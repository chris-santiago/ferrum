//! prepare_render_inputs(spec, batch) →
//!   1. Apply Phase 5 transforms.
//!   2. Build provisional ResolvedScales for tick-label generation.
//!   3. Derive AxesInput (titles, tick_labels).
//!   4. Group rows by facet field (if facet).
//!   5. Build LegendEntry list (if color encoding).
//!   6. (Phase 8a) Build per-layer prepared inputs; swap x↔y if CoordFlip.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, LargeStringArray, StringArray, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::layout::{
    AxesInput, AxisInput, AxisOrient, ColorbarInput, FacetGroup, FacetKey, LegendEntry,
    TickProjection,
};
use crate::spec::chart::ChartSpec;
use crate::transform::context::TransformContext;
use crate::transform::core::{apply_transforms_named, FINAL_OUTPUT_KEY};

use super::scale_resolve::ResolvedScales;
use super::{RenderError, RenderWarning};

mod extent;
mod legend;

// Re-exported into this module's namespace so the orchestrator and the inline
// `#[cfg(test)] mod tests` (which uses `super::*`) resolve the extracted helpers
// from their cohesive submodules without a path change.
use extent::fix_transform_extents_for_facet;

/// Key used when unifying wrap-mode and grid-mode partitions: (col_val, Option<row_val>).
/// `None` row value = wrap mode (single-field facet).
type FacetPartitionKey = (String, Option<String>);

/// Per-layer prepared rendering data. When ChartSpec.layers.is_none(), exactly one
/// LayerPrepared is constructed from the chart-level mark + encoding.
#[derive(Debug, Clone)]
pub struct LayerPrepared {
    pub mark: crate::spec::mark::Mark,
    pub encoding: crate::spec::encoding::Encoding,
    pub transforms: Vec<crate::transform::core::TransformSpec>,
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
    /// Name of the chart-level transform output this layer reads from.
    /// `None` ⇒ pipeline final output (resolved via [`FINAL_OUTPUT_KEY`]).
    pub data_source: Option<String>,
    /// Phase 9c — position adjustment for this layer. Merged from
    /// `Layer.position` (preferred) or `ChartSpec.position` (chart-level
    /// fallback for single-layer charts).
    pub position: Option<crate::spec::position::PositionAdjust>,
    /// Pixel-level blend mode for this layer's MarkBatch.
    pub blend: Option<ferrum_scene::BlendMode>,
    /// Whether this layer resolves its own independent y-scale slot instead of
    /// sharing the primary (layer 0) y-scale (secondary-y-axis, GH #52). Mirrors
    /// [`crate::spec::layer::Layer::independent_y`]; always `false` for a
    /// single-layer (chart-only) chart, whose sole layer is the primary.
    pub independent_y: bool,
}

impl LayerPrepared {
    /// Build a single layer from chart-level fields (single-layer mode).
    pub(crate) fn from_chart_only(spec: &crate::spec::chart::ChartSpec) -> Self {
        Self {
            mark: spec.mark,
            encoding: spec.encoding.clone(),
            transforms: spec.transforms.clone(),
            mark_style: spec.mark_style.clone(),
            data_source: None,
            position: spec.position.clone(),
            blend: None,
            independent_y: false,
        }
    }

    /// Build a layer by inheriting unset encoding channels from chart-level.
    /// See [`crate::spec::encoding::Encoding::inherit_from`] for the policy.
    pub(crate) fn from_chart_and_layer(
        spec: &crate::spec::chart::ChartSpec,
        layer: &crate::spec::layer::Layer,
    ) -> Self {
        let mut encoding = layer.encoding.clone();
        // Layers routed to their own data via data_source are self-contained —
        // only inherit non-positional channels (color, size, etc.) from the
        // chart level. Inheriting x/y would inject the primary batch's field
        // into a layer that reads a different named output, causing marks like
        // rule to iterate over rows they don't own.
        if layer.data_source.is_some() {
            encoding.inherit_non_positional(&spec.encoding);
        } else {
            encoding.inherit_from(&spec.encoding);
        }
        Self {
            mark: layer.mark,
            encoding,
            transforms: layer.transforms.clone(),
            mark_style: layer.mark_style.clone().or_else(|| spec.mark_style.clone()),
            data_source: layer.data_source.clone(),
            position: layer.position.clone().or_else(|| spec.position.clone()),
            blend: layer.blend,
            independent_y: layer.independent_y,
        }
    }
}

/// Normalize Arrow string columns to `Utf8` (`StringArray`).
///
/// Polars exports string columns as `Utf8View` (`StringViewArray`) or `LargeUtf8`
/// (`LargeStringArray`) depending on version. The rest of the render pipeline
/// (scale_resolve, draw, mark renderers) downcasts to `StringArray`. Converting
/// once here keeps every consumer simple and avoids per-site downcast forks.
fn normalize_string_views(batch: &RecordBatch) -> RecordBatch {
    let schema = batch.schema();
    let mut new_fields: Vec<Arc<Field>> = Vec::with_capacity(schema.fields().len());
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut changed = false;
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match field.data_type() {
            DataType::Utf8View => {
                if let Some(view) = col.as_any().downcast_ref::<StringViewArray>() {
                    let owned: StringArray = view
                        .iter()
                        .map(|opt| opt.map(|s| s.to_string()))
                        .collect::<Vec<Option<String>>>()
                        .into();
                    new_cols.push(Arc::new(owned));
                    new_fields.push(Arc::new(Field::new(
                        field.name(),
                        DataType::Utf8,
                        field.is_nullable(),
                    )));
                    changed = true;
                    continue;
                }
            }
            DataType::LargeUtf8 => {
                // Polars produces LargeUtf8 (LargeStringArray) for string columns.
                if let Some(large) = col.as_any().downcast_ref::<LargeStringArray>() {
                    let owned: StringArray = large
                        .iter()
                        .map(|opt| opt.map(|s| s.to_string()))
                        .collect::<Vec<Option<String>>>()
                        .into();
                    new_cols.push(Arc::new(owned));
                    new_fields.push(Arc::new(Field::new(
                        field.name(),
                        DataType::Utf8,
                        field.is_nullable(),
                    )));
                    changed = true;
                    continue;
                }
            }
            _ => {}
        }
        new_cols.push(col.clone());
        new_fields.push(field.clone());
    }
    if !changed {
        return batch.clone();
    }
    let new_schema = Arc::new(Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_cols)
        .expect("normalized batch must construct: same row count + matched dtypes")
}

#[derive(Debug)]
pub struct PreparedInputs {
    /// All chart-level transform outputs, keyed by their `name` (when present)
    /// plus `FINAL_OUTPUT_KEY` ("__final__") for the pipeline tail. Layers
    /// with `data_source: Some(name)` look up their input batch here; layers
    /// with `data_source: None` resolve to `FINAL_OUTPUT_KEY` via
    /// [`PreparedInputs::final_batch`].
    pub transform_outputs: HashMap<String, RecordBatch>,
    /// Scales resolved once during prepare **for tick-label generation only**.
    /// Their pixel ranges are not panel-specific, so the final per-panel scales —
    /// whose ranges differ per panel — are resolved fresh by
    /// `scene_build::resolve_panel_scales` (SPINE-04). Do not consume these for
    /// per-panel mark/grid positioning; use the per-panel scales there instead.
    pub provisional_scales: ResolvedScales,
    pub axes: AxesInput,
    pub facet_groups: Vec<FacetGroup>,
    pub legend_entries: Vec<LegendEntry>,
    /// Legend title (Themes-T2.5b). Defaults to the color encoding's field
    /// name; None when no categorical color encoding drives a legend.
    pub legend_title: Option<String>,
    /// Continuous-colorbar input. Built from a Continuous color scale's
    /// domain + scheme; consumed by `compute_layout` to allocate a colorbar
    /// in the legend gutter. Mutually exclusive with `legend_entries`.
    pub colorbar: Option<ColorbarInput>,
    pub warnings: Vec<RenderWarning>,
    /// One entry per layer. Single-layer charts have len() == 1.
    pub layers: Vec<LayerPrepared>,
    /// True when spec.coord == Some(CoordKind::Flip). The draw loop uses this
    /// to know that x/y have already been swapped in each layer's encoding.
    pub coord_flipped: bool,
    /// D13: all per-encoding legend style overrides extracted from
    /// `encoding.color.legend.*`. Grouped here to keep `PreparedInputs`
    /// flat and to make adding new override fields a one-location change.
    pub legend_overrides: LegendPreparedOverrides,
    /// Multivariate B1: auxiliary (size / shape) legend blocks. Built from the
    /// resolved size/shape scales, stacked beneath the color legend by
    /// `compute_layout`. Empty when neither size nor shape is encoded (or both
    /// suppressed via `legend=None`). A size/shape channel that shares its
    /// field with the color channel is merged into the color legend rather than
    /// emitted here (Vega-Lite same-field merge).
    pub aux_legends: Vec<crate::layout::AuxLegendInput>,
    /// The tick-count hints (`Axis(tick_count=N)`, default 10) used to generate
    /// the x/y tick labels + projections. Carried so the post-prepare
    /// tick_extra/tick_min_step adjustment (B5 unit 2) can regenerate the SAME
    /// raw tick values that produced the current labels.
    pub x_tick_count: usize,
    pub y_tick_count: usize,
}

/// Per-encoding legend style overrides extracted from `encoding.color.legend.*`.
///
/// All fields are `Option` so `Default` (all `None`) represents "use theme defaults".
/// Populated by `prepare_render_inputs` from the `color_legend_extra` JSON map and
/// consumed by `legend_overrides_from_prep` (layout) and `prepare_and_layout`
/// (effective-theme construction).
#[derive(Debug, Clone, Default)]
pub struct LegendPreparedOverrides {
    /// D13: orient override from `encoding.color.legend.orient`.
    /// `None` means use the theme default.
    pub orient: Option<crate::layout::LegendOrient>,
    /// D13: title override from `encoding.color.legend.title`.
    /// `Some(s)` replaces the default field-name legend title.
    pub title: Option<String>,
    /// D13: title font size override from `encoding.color.legend.titleFontSize`.
    pub title_font_size: Option<f64>,
    /// D13: columns override from `encoding.color.legend.columns`.
    /// When `Some`, categorical legend entries are arranged in N columns.
    pub columns: Option<u32>,
    /// D13+: maximum number of colorbar ticks from `encoding.color.legend.tickCount`.
    pub tick_count: Option<usize>,
    /// D13+: label font size from `encoding.color.legend.labelFontSize`.
    pub label_font_size: Option<f64>,
    /// D13+: colorbar gradient bar length in pixels from `encoding.color.legend.gradientLength`.
    pub gradient_length: Option<f64>,
    /// D13+: colorbar gradient bar thickness in pixels from `encoding.color.legend.gradientThickness`.
    pub gradient_thickness: Option<f64>,
    /// D13+: direction override from `encoding.color.legend.direction`.
    pub direction: Option<crate::layout::LegendDirection>,
    /// D13+: explicit tick/entry values from `encoding.color.legend.values`.
    pub values: Option<Vec<String>>,
    /// D13+: legend type override from `encoding.color.legend.type`.
    /// "gradient" forces colorbar rendering; "symbol" forces discrete entries.
    pub legend_type: Option<String>,
    /// B5: symbol shape override from `encoding.color.legend.symbol_type`.
    /// Per-channel wins over chart-level `configure_legend(symbol_type=...)`.
    pub symbol_type: Option<String>,
    /// B5 unit 3: stroke width of legend symbols from
    /// `encoding.color.legend.symbol_stroke_width`.
    pub symbol_stroke_width: Option<f64>,
    /// B5 unit 3: per-legend vertical entry spacing from
    /// `encoding.color.legend.row_padding` (replaces `LEGEND_ENTRY_ROW_PAD`).
    pub row_padding: Option<f64>,
    /// B5 unit 3: per-legend horizontal entry spacing from
    /// `encoding.color.legend.column_padding`.
    pub column_padding: Option<f64>,
    /// B5 unit 3: max legend-label pixel width from
    /// `encoding.color.legend.label_limit`. Labels wider than this are truncated
    /// with an ellipsis.
    pub label_limit: Option<f64>,
    /// B5 unit 3: cap the legend group's total height from
    /// `encoding.color.legend.clip_height`. Overflow is hard-clipped.
    pub clip_height: Option<f64>,
    /// B5 unit 3: minimum step between colorbar ticks (data units) from
    /// `encoding.color.legend.tick_min_step`.
    pub tick_min_step: Option<f64>,
    /// B5 unit 3: coarse draw order from `encoding.color.legend.zindex`.
    /// `>= 1` routes the legend above marks; `<= 0` (default) below.
    pub zindex: Option<i64>,
    /// B5 unit 6a: legend swatch area (px²) from `encoding.color.legend.symbol_size`.
    pub symbol_size: Option<f64>,
    /// B5 unit 6a: legend entry-label fill color from
    /// `encoding.color.legend.label_color`.
    pub label_color: Option<String>,
    /// B5 unit 6a: extra plot→legend gap (px) from `encoding.color.legend.offset`.
    pub offset: Option<f64>,
    /// B5 unit 6a: internal legend box padding (px) from
    /// `encoding.color.legend.padding`.
    pub padding: Option<f64>,
    /// B5 unit 6a: legend title→entry gap (px) from
    /// `encoding.color.legend.title_padding`.
    pub title_padding: Option<f64>,
}

impl PreparedInputs {
    /// The final transform-pipeline output — i.e. `transform_outputs[FINAL_OUTPUT_KEY]`.
    /// Used by the render orchestrator for facet filtering, the colorbar legend
    /// scale rebuild, and any other consumer that needs the chart-level tail.
    pub fn final_batch(&self) -> &RecordBatch {
        self.transform_outputs
            .get(FINAL_OUTPUT_KEY)
            .expect("apply_transforms_named publishes FINAL_OUTPUT_KEY unconditionally")
    }
}

/// Which positional channel an axis-input derivation is for.
///
/// Carries the two pieces of behavior that differ between the x and y axis
/// derivations so `build_axis_input` can be called once per channel instead of
/// the per-axis block being hand-written twice (SPINE-08):
/// - the **default orient** (`Bottom` for x, `Left` for y), used when no
///   per-channel `Axis(orient=...)` override is present;
/// - the **non-ordinal reverse policy** (only y reverses labels/fractions so
///   high domain values sit at the top), threaded into `build_axis_tick_inputs`
///   and `encoding_axis_style_overrides`' validation channel token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    X,
    Y,
}

impl Channel {
    /// The default axis side when no per-channel `Axis(orient=...)` is set.
    fn default_orient(self) -> AxisOrient {
        match self {
            Channel::X => AxisOrient::Bottom,
            Channel::Y => AxisOrient::Left,
        }
    }

    /// Whether this channel applies the non-ordinal label/fraction reversal.
    /// Only y reverses (the pixel range is already inverted in scale_resolve);
    /// x never does.
    fn reverses(self) -> bool {
        matches!(self, Channel::Y)
    }

    /// The static channel token used for axis-orient/style validation messages.
    fn token(self) -> &'static str {
        match self {
            Channel::X => "x",
            Channel::Y => "y",
        }
    }
}

/// Resolve a positional axis title via the 3-way idiom shared by the spec-level
/// encoding title and the per-channel `Axis(title=...)` override (SPINE-08):
///   - **absent** (`None`)            → fall through to `fallback`;
///   - **present + empty/whitespace** → explicit suppress, returns `None`;
///   - **present + non-empty**        → use it.
///
/// The empty-string suppression contract comes from Python forwarding
/// `title = ""` only when `title=None` was explicitly passed; an absent key
/// means "use the default".
fn resolve_axis_title(value: Option<&str>, fallback: Option<String>) -> Option<String> {
    match value {
        Some(s) if s.trim().is_empty() => None, // explicit suppress: don't fall back
        Some(s) => Some(s.to_owned()),          // explicit non-empty title
        None => fallback,                       // absent: use the default
    }
}

/// Build one positional axis's [`AxisInput`] from its resolved scale and
/// encoding (SPINE-08).
///
/// This is the single derivation the x and y axes both run; the per-channel
/// differences (orient default + the non-ordinal-y label reversal) are carried
/// by [`Channel`]. It performs, in order: the 3-way title resolution
/// (spec-level title → layer-0 title → field name), the show-toggle extraction,
/// the tick-label/projection build (threading numeric format forward, applying
/// temporal format directly, reversing for non-ordinal y), the theme-gated
/// minor-fraction projection, the per-channel style-override parse, the orient
/// default, and the final `AxisInput` assembly.
///
/// `rendering_enc` is the layer-0 (post-CoordFlip) encoding for this channel;
/// `spec_enc` is the user-facing spec-level encoding (its title wins). Returns
/// the assembled `AxisInput` plus the resolved show-axis-band toggle inputs are
/// kept on the `AxisInput` itself.
fn build_axis_input(
    channel: Channel,
    rendering_enc: Option<&crate::spec::encoding::EncodingSpec>,
    spec_enc: Option<&crate::spec::encoding::EncodingSpec>,
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    theme: &crate::layout::ThemeInputs,
) -> Result<AxisInput, RenderError> {
    // Axis title resolution priority:
    //   1. Spec-level encoding title (set by user via .encode(y=Y(..., title=...)))
    //   2. Layer-0 encoding title (set by desugar for internal column names)
    //   3. Field name (fallback)
    // User-explicit titles always win; layer-level titles override the field
    // name for diagnostic charts whose layer-0 encoding references a column
    // with a non-semantic name (e.g. "lower_whisker" / "param_value").
    let field_title = rendering_enc.and_then(|e| {
        // Explicit spec-level title takes highest priority.
        let spec_title =
            resolve_axis_title(spec_enc.and_then(|p| p.title.as_deref()), None);
        if let Some(spec_title) = spec_title {
            return Some(spec_title);
        }
        if spec_enc.and_then(|p| p.title.as_deref()).is_some() {
            // Spec-level title present but empty → explicit suppress, no fallback.
            return None;
        }
        // Layer-0 title (desugar for diagnostic column names), then field name.
        resolve_axis_title(e.title.as_deref(), Some(e.field.clone()))
    });

    // D7 + D12 (B5-typed): per-axis style fields from the typed `encoding.axis`
    // style spec. The show toggles default to `true` (and title falls through to
    // the field name) so SVG output is byte-identical when the encoding carries
    // no axis overrides.
    let enc_axis = rendering_enc.and_then(|e| e.axis.as_ref());
    let show_labels = enc_axis.and_then(|a| a.labels).unwrap_or(true);
    let show_ticks = enc_axis.and_then(|a| a.ticks).unwrap_or(true);
    let show_domain = enc_axis.and_then(|a| a.domain).unwrap_or(true);
    let show_grid = enc_axis.and_then(|a| a.grid).unwrap_or(true);
    // Axis(title=...): the outer Option distinguishes "key absent" (fall through
    // to the field-name default) from "key present but empty" (explicit suppress).
    let title = resolve_axis_title(enc_axis.and_then(|a| a.title.as_deref()), field_title);

    // D12 + D3: resolve the tick label format. A per-channel `Axis(label_format=,
    // label_format_type=)` takes precedence over the shorthand
    // `encoding.format`/`format_type`. Chart-level
    // `configure_axis(label_format_raw=...)` is applied later in `render/mod.rs`
    // only when no per-encoding override exists.
    let (tick_format, tick_format_type) = resolve_axis_label_format(rendering_enc);

    // Grid item 18: minor tick fractions from the resolved scale, projected
    // through the same `[0,1]`-range scale that places majors. The
    // `theme.grid.minor` gate (default `false`) is the single source of truth:
    // when off, `minor` is empty → no minors built → default output
    // byte-identical.
    let minor_fractions = if theme.grid.minor {
        scale.minor_tick_fractions()
    } else {
        Vec::new()
    };

    // Derive labels + scale-projected tick fractions + threaded label format
    // through the shared `build_axis_tick_inputs` helper. Global axes use
    // `Thread` mode (numeric format is threaded onto the override for central
    // application; temporal is formatted directly). `channel.reverses()` carries
    // the non-ordinal-y reversal; the theme-gated minor fractions are passed in
    // (not reversed).
    let (tick_labels, tick_projection, label_format_override) = build_axis_tick_inputs(
        scale,
        tick_count,
        TickFormatMode::Thread {
            format: tick_format,
            format_type: tick_format_type.as_deref(),
        },
        channel.reverses(),
        minor_fractions,
    );

    // B5: per-channel axis STYLING + positioning overrides.
    let mut overrides = encoding_axis_style_overrides(enc_axis.map(Box::as_ref), channel.token())?;
    // Seed the per-channel `label_format` (resolved via the temporal/numeric
    // threading above) onto the bundle.
    overrides.label_format = label_format_override;

    // Per-channel `orient` selects the axis side; absent → the dimension default
    // (Bottom for x, Left for y). The `orient` override input stays in the bundle
    // so a later chart-level `configure_axis(orient=...)` fills it only when
    // still `None` (per-channel wins), then `resolve_orient` re-syncs.
    let orient = overrides.orient.unwrap_or(channel.default_orient());

    Ok(AxisInput {
        orient,
        title,
        tick_labels,
        show_labels,
        show_ticks,
        show_domain,
        show_grid,
        tick_format: None, // already applied above
        tick_format_type: None,
        tick_projection,
        // Explicit-range ordinal axes (GH #39 phase 2): carry the scale's absolute
        // band centers so layout places tick labels/grid lines at the same pixels
        // the marks get. `None` for continuous axes and for ordinal axes without an
        // explicit range — the latter keeps `uniform_center`, byte-identical.
        categorical_positions: scale.explicit_band_centers(),
        overrides,
    })
}

/// Prepare a single leaf's render inputs (transforms, per-layer encodings,
/// provisional scales/axes, legends).
///
/// `leaf_scales` is the D4b composite seam: `Some` only for a composite leaf,
/// threading a resolved-domain context into the provisional scale pass so the
/// leaf's provisional axes (and thus tick labels) resolve on the same auto path,
/// seeded by the shared domain, that the per-panel final pass will use. `None`
/// for every standalone (flat/facet) render reproduces the pre-D4b behavior
/// byte-for-byte.
pub fn prepare_render_inputs(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &crate::layout::ThemeInputs,
    leaf_scales: Option<&crate::render::scale_resolve::LeafScaleContext>,
) -> Result<PreparedInputs, RenderError> {
    if batch.num_rows() == 0 {
        return Err(RenderError::EmptyBatch);
    }

    // Normalize Utf8View columns (e.g. from polars) to Utf8 so downstream
    // downcasts to StringArray succeed uniformly.
    let normalized = normalize_string_views(batch);

    // Run the Phase 5 transform pipeline, applying the facet-aware partition →
    // per-panel transforms → inject → concat path when a facet is present and
    // the plain single-partition path otherwise. The D10 impute step runs once
    // inside, in both branches (SPINE-12).
    let transform_outputs = run_transforms_with_facet(spec, &normalized)?;
    let transformed = transform_outputs
        .get(FINAL_OUTPUT_KEY)
        .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY")
        .clone();

    // --- Phase 8a: per-layer inputs + CoordFlip ---

    // Build per-layer prepared inputs (swapping x↔y / x2↔y2 under CoordFlip).
    let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));
    let layers = build_layers(spec, coord_flipped);

    // Validate every layer's data_source resolves to a known transform output.
    // Fail-fast here so the per-panel render loop can unconditionally `.get()`.
    for (i, layer) in layers.iter().enumerate() {
        if let Some(name) = &layer.data_source {
            if !transform_outputs.contains_key(name) {
                let mut keys: Vec<&str> =
                    transform_outputs.keys().map(|s| s.as_str()).collect();
                keys.sort_unstable();
                return Err(RenderError::TransformFailed(format!(
                    "layer {i} data_source '{name}' not found in transform outputs; \
                     available keys: [{}]",
                    keys.join(", ")
                )));
            }
        }
    }

    // Build provisional scales and axes using the first layer's resolved encoding,
    // which already incorporates CoordFlip. For single-layer non-flipped specs this
    // is identical to what Phase 7 computed (same encoding, same spec fields).
    //
    // We need a ChartSpec whose encoding reflects the (possibly swapped) channels.
    // Clone spec and substitute the rendering encoding so resolve_scales works
    // correctly. For back-compat (single-layer, no flip), this clone is structurally
    // equal to spec itself — goldens should be byte-identical.
    let rendering_encoding = layers[0].encoding.clone();
    let rendering_spec = ChartSpec {
        encoding: rendering_encoding.clone(),
        ..spec.clone()
    };

    let (provisional_scales, scale_warnings) = crate::render::scale_resolve::resolve_scales_with_leaf_context(
        &rendering_spec,
        &transformed,
        &transform_outputs,
        (0.0, 1.0),
        (0.0, 1.0),
        theme,
        leaf_scales,
    )?;

    // Derive both axis inputs + their tick counts (SPINE-08/SPINE-12). The
    // rendering encoding is post-CoordFlip; the spec-level encoding (whose
    // explicit title wins) is threaded through for the title 3-way. Also
    // derives the provisional secondary y-axis inputs (secondary-y-axis, GH
    // #52) from `layers`/`transformed`/`transform_outputs`, pushing any scale
    // warnings into `scale_warnings` alongside the primary's.
    let mut scale_warnings = scale_warnings;
    let (axes, x_tick_count, y_tick_count) = build_axes(
        spec,
        &rendering_encoding,
        &provisional_scales,
        theme,
        &layers,
        &transformed,
        &transform_outputs,
        &mut scale_warnings,
    )?;

    let facet_groups = if let Some(fspec) = &spec.facet {
        if let Some(row_field) = &fspec.row {
            // Grid mode: produce one FacetGroup per (row_val, col_val) pair in
            // row-major order. Each group's `key` is the col dimension (drives the
            // column-header strip title); `row_key` is the row dimension (drives the
            // row-header strip title and secondary batch filter).
            group_rows_by_two_fields(&transformed, &fspec.field, row_field)?
        } else {
            // Wrap mode: single-field grouping — behavior unchanged.
            group_rows_by_field(&transformed, &fspec.field)?
        }
    } else {
        Vec::new()
    };

    // Color legend / colorbar / aux-legend construction (categorical entries,
    // continuous colorbar, conditional-color fallback, per-channel style
    // overrides, size/shape aux legends, and the same-field color+size merge).
    // See `legend::build_color_legend` for the full behavior.
    let legend::ColorLegendBundle {
        legend_entries,
        colorbar,
        legend_title,
        legend_overrides,
        aux_legends,
    } = legend::build_color_legend(spec, &transformed, &provisional_scales);

    Ok(PreparedInputs {
        transform_outputs,
        provisional_scales,
        axes,
        facet_groups,
        legend_entries,
        legend_title,
        colorbar,
        warnings: scale_warnings,
        layers,
        coord_flipped,
        legend_overrides,
        aux_legends,
        x_tick_count,
        y_tick_count,
    })
}

/// Run the Phase 5 transform pipeline, producing the named-output map (SPINE-12).
///
/// When faceting is active, partition the input batch by the facet column(s)
/// BEFORE running transforms so each panel gets its own data subset and
/// transforms execute independently per panel; the facet column(s) are
/// re-injected into every output and per-key batches are concatenated across
/// panels. When there is no facet, the pipeline is the plain single-partition
/// path (one partition = full batch). Both branches apply the D10 impute step
/// (`apply_impute_to_final`) exactly once on `FINAL_OUTPUT_KEY`.
fn run_transforms_with_facet(
    spec: &ChartSpec,
    normalized: &RecordBatch,
) -> Result<HashMap<String, RecordBatch>, RenderError> {
    let ctx = TransformContext::default();
    let Some(fspec) = &spec.facet else {
        // No facet: unchanged pipeline (single partition = full batch).
        let mut outputs = apply_transforms_named(&spec.transforms, normalized, &ctx)
            .map_err(|e| RenderError::TransformFailed(e.to_string()))?;
        apply_impute_to_final(&mut outputs, spec);
        return Ok(outputs);
    };

    // Facet-before-transform: partition → per-panel transforms → inject facet column(s) → concat.
    //
    // Grid mode (fspec.row is set): partition on the composite (col_val, row_val) key so each
    // (row, col) cell is a distinct partition. Both the col field and the row field are injected
    // back into every transform output so per-panel filtering can use either.
    //
    // Wrap mode (fspec.row is None): partition on fspec.field only — behavior unchanged.
    // Each partition carries (col_val, optional_row_val) → RecordBatch.
    // Grid mode populates the row value; wrap mode leaves it None.
    let partitions: Vec<(FacetPartitionKey, RecordBatch)> = if let Some(row_field) = &fspec.row {
        partition_batch_by_two_fields(normalized, &fspec.field, row_field)?
            .into_iter()
            .map(|((col_val, row_val), batch)| ((col_val, Some(row_val)), batch))
            .collect()
    } else {
        partition_batch_by_field(normalized, &fspec.field)?
            .into_iter()
            .map(|(col_val, batch)| ((col_val, None), batch))
            .collect()
    };

    // Pin a shared value-axis extent from the full (pre-partition) batch for
    // every extent-carrying transform (Kde / Bin / Violin / Kde2D / Bin2D)
    // without an explicit extent. Otherwise each partition would use its own
    // range, causing panels (and hue groups within panels) to render on
    // different value scales and making positions non-comparable. By fixing
    // the extent to the global range before partitioning, all panels and
    // groups share the same value axis. This is the correct default for
    // faceted density / histogram / violin / 2-D density / heatmap charts
    // (archaeology bug #7; extended to Kde2D/Bin2D by R5).
    let effective_transforms = fix_transform_extents_for_facet(&spec.transforms, normalized);

    let mut merged: HashMap<String, Vec<RecordBatch>> = HashMap::new();
    for ((col_value, row_value), partition_batch) in &partitions {
        let mut panel_outputs = apply_transforms_named(&effective_transforms, partition_batch, &ctx)
            .map_err(|e| RenderError::TransformFailed(e.to_string()))?;
        // D10: per-panel imputation.
        apply_impute_to_final(&mut panel_outputs, spec);
        // Re-inject the col field (and row field when in grid mode) into every
        // output batch. Transforms like Smooth replace the batch entirely, losing
        // any facet columns that were present on the input partition.
        for batch in panel_outputs.values_mut() {
            *batch = inject_facet_column(batch, &fspec.field, col_value);
            if let (Some(row_field), Some(row_val)) = (&fspec.row, row_value) {
                *batch = inject_facet_column(batch, row_field, row_val);
            }
        }
        for (key, batch) in panel_outputs {
            merged.entry(key).or_default().push(batch);
        }
    }
    // Concat per-key batches across all panels into a single map.
    let mut combined: HashMap<String, RecordBatch> = HashMap::new();
    for (key, batches) in merged {
        if batches.len() == 1 {
            combined.insert(key, batches.into_iter().next().unwrap());
        } else {
            let schema = batches[0].schema();
            let merged_batch = arrow::compute::concat_batches(&schema, &batches).map_err(|e| {
                RenderError::TransformFailed(format!("concat facet partitions for key '{key}': {e}"))
            })?;
            combined.insert(key, merged_batch);
        }
    }
    Ok(combined)
}

/// Apply the D10 impute step in place on the pipeline's `FINAL_OUTPUT_KEY`
/// batch, replacing it only when imputation added rows (SPINE-12).
///
/// This is the single call site for both the facet (per-panel) and non-facet
/// transform branches; it formerly appeared verbatim in each.
fn apply_impute_to_final(outputs: &mut HashMap<String, RecordBatch>, spec: &ChartSpec) {
    let final_batch = outputs
        .get(FINAL_OUTPUT_KEY)
        .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY");
    let imputed = apply_impute(final_batch, spec);
    if imputed.num_rows() != final_batch.num_rows() {
        outputs.insert(FINAL_OUTPUT_KEY.to_string(), imputed);
    }
}

/// Build the per-layer prepared inputs, applying the CoordFlip x↔y / x2↔y2 swap
/// to every layer when `coord_flipped` (SPINE-12).
///
/// Single-layer charts (`spec.layers == None`) produce exactly one
/// [`LayerPrepared`] from the chart-level mark + encoding; multi-layer charts
/// produce one per `Layer` (inheriting unset channels from chart level).
fn build_layers(spec: &ChartSpec, coord_flipped: bool) -> Vec<LayerPrepared> {
    let raw: Vec<LayerPrepared> = match &spec.layers {
        None => vec![LayerPrepared::from_chart_only(spec)],
        Some(layer_vec) => layer_vec
            .iter()
            .map(|l| LayerPrepared::from_chart_and_layer(spec, l))
            .collect(),
    };
    if !coord_flipped {
        return raw;
    }
    raw.into_iter()
        .map(|mut lp| {
            let tmp = lp.encoding.x.take();
            lp.encoding.x = lp.encoding.y.take();
            lp.encoding.y = tmp;
            // Phase 10c-pre: x2/y2 must swap together with x/y so paired
            // endpoints (segment, ribbon) remain self-consistent under flip.
            let tmp2 = lp.encoding.x2.take();
            lp.encoding.x2 = lp.encoding.y2.take();
            lp.encoding.y2 = tmp2;
            lp
        })
        .collect()
}

/// Build both positional [`AxisInput`]s and their tick counts (SPINE-08/12).
///
/// `rendering_encoding` is the layer-0 (post-CoordFlip) encoding; `spec`'s
/// encoding supplies the user-facing spec-level title that wins the 3-way title
/// resolution. The per-channel derivation runs once each through
/// [`build_axis_input`], with [`Channel`] carrying the orient default and the
/// non-ordinal-y reversal. Returns `(axes, x_tick_count, y_tick_count)`.
#[allow(clippy::too_many_arguments)]
fn build_axes(
    spec: &ChartSpec,
    rendering_encoding: &crate::spec::encoding::Encoding,
    provisional_scales: &ResolvedScales,
    theme: &crate::layout::ThemeInputs,
    layers: &[LayerPrepared],
    transformed: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(AxesInput, usize, usize), RenderError> {
    // D3 (flexibility campaign): per-channel `Axis(tick_count=N)` controls the
    // target tick count for continuous + temporal axes (default 10). Limiting it
    // is what de-clutters dense temporal axes (e.g. a 72-row series no longer
    // renders 72 overlapping ticks). Read before tick generation so the count
    // feeds every downstream tick call (labels, fractions, minors).
    let x_tick_count = encoding_axis_tick_count(rendering_encoding.x.as_ref()).unwrap_or(10);
    let y_tick_count = encoding_axis_tick_count(rendering_encoding.y.as_ref()).unwrap_or(10);

    let axes = AxesInput {
        x: build_axis_input(
            Channel::X,
            rendering_encoding.x.as_ref(),
            spec.encoding.x.as_ref(),
            &provisional_scales.x,
            x_tick_count,
            theme,
        )?,
        y: build_axis_input(
            Channel::Y,
            rendering_encoding.y.as_ref(),
            spec.encoding.y.as_ref(),
            &provisional_scales.y,
            y_tick_count,
            theme,
        )?,
        show_x: spec.axis_x.unwrap_or(true),
        show_y: spec.axis_y.unwrap_or(true),
        secondary_y: build_secondary_y_axis_inputs(
            spec,
            layers,
            transformed,
            transform_outputs,
            theme,
            warnings,
        )?,
    };
    Ok((axes, x_tick_count, y_tick_count))
}

/// Provisional secondary y-axis inputs, one per `independent_y` layer in layer
/// order (secondary-y-axis, GH #52). Mirrors the primary y axis's
/// provisional-then-final pattern directly above: each independent layer's own
/// y-scale is resolved against the full (pre-panel) batch with a `(0.0, 1.0)`
/// placeholder pixel range, using the SAME per-layer encoding merge (chart
/// encoding overlaid by the layer's own, `layers: None` so the y-domain isn't
/// re-unioned with sibling layers) that
/// [`render::scene_build::resolve_layer_y_scale`](crate::render::scene_build)
/// (Task 2) applies per-panel later. Domain-derived tick labels/fractions do
/// not depend on the placeholder pixel range, so the two resolutions agree —
/// one logical scale, computed twice only because layout must run before
/// panels exist. `compute_layout` (Task 3) reserves one right-side margin band
/// per returned axis and places it stacked outward from the primary.
///
/// Empty when no layer sets `independent_y` — the byte-stable gate mirrors
/// `resolve_panel_scales`'s Task 2 gate exactly, so `AxesInput.secondary_y`
/// stays empty and layout/scene output for the shared path is unchanged.
fn build_secondary_y_axis_inputs(
    spec: &ChartSpec,
    layers: &[LayerPrepared],
    transformed: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &crate::layout::ThemeInputs,
    warnings: &mut Vec<RenderWarning>,
) -> Result<Vec<AxisInput>, RenderError> {
    if !layers.iter().skip(1).any(|l| l.independent_y) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for layer in layers.iter().skip(1) {
        if !layer.independent_y {
            continue;
        }
        let layer_batch: &RecordBatch = match &layer.data_source {
            Some(name) => transform_outputs
                .get(name)
                .expect("layer.data_source validated by prepare_render_inputs"),
            None => transformed,
        };
        // Same per-layer encoding merge `resolve_layer_y_scale` (scene_build.rs,
        // Task 2) uses: the layer's own encoding overlays the chart-level
        // encoding, and `layers: None` stops the y-domain union from re-unioning
        // sibling layers' fields so this slot spans exactly its own data.
        let mut layer_encoding = spec.encoding.clone();
        layer_encoding.overlay_from(&layer.encoding);
        let layer_spec = ChartSpec {
            mark: layer.mark,
            encoding: layer_encoding.clone(),
            layers: None,
            ..spec.clone()
        };
        let (layer_scales, layer_warnings) =
            crate::render::scale_resolve::resolve_scales_with_leaf_context(
                &layer_spec,
                layer_batch,
                transform_outputs,
                (0.0, 1.0),
                (0.0, 1.0),
                theme,
                None,
            )?;
        warnings.extend(layer_warnings);
        let y_tick_count = encoding_axis_tick_count(layer_encoding.y.as_ref()).unwrap_or(10);
        let axis_input = build_axis_input(
            Channel::Y,
            layer_encoding.y.as_ref(),
            layer_encoding.y.as_ref(),
            &layer_scales.y,
            y_tick_count,
            theme,
        )?;
        out.push(axis_input);
    }
    Ok(out)
}

/// D10: fill missing (group × x-value) combinations in the batch with a constant y value.
///
/// When `encoding.y.impute = {"value": N}` is set and the encoding has both an x and
/// color channel, this synthesizes zero-rows for every (x-value, color-group) pair
/// that is absent from the data, ensuring that line charts and area charts connect
/// correctly even when some groups are missing observations at certain x ticks.
///
/// The imputed rows carry the x and color values from the (x, group) key and the
/// impute constant for y. All other columns default to null. No-ops when any of
/// these conditions hold: no x encoding, no color encoding, impute value absent,
/// or the batch is already complete.
fn apply_impute(
    batch: &RecordBatch,
    spec: &ChartSpec,
) -> RecordBatch {
    use arrow::array::{Float64Array, StringArray};

    // Only handle `encoding.y.impute = {"value": <number>}`.
    let impute_value = spec
        .encoding
        .y
        .as_ref()
        .and_then(|y| y.impute.as_ref())
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("value"))
        .and_then(|v| v.as_f64());
    let Some(fill) = impute_value else { return batch.clone(); };

    let x_enc = match spec.encoding.x.as_ref() { Some(e) => e, None => return batch.clone() };
    let color_enc = match spec.encoding.color.as_ref() { Some(e) => e, None => return batch.clone() };
    let y_enc = match spec.encoding.y.as_ref() { Some(e) => e, None => return batch.clone() };

    let x_field = &x_enc.field;
    let color_field = &color_enc.field;
    let y_field = &y_enc.field;

    // Collect distinct x values and groups. Only handles Float64 x + Utf8 color.
    let x_col = match batch.column_by_name(x_field) { Some(c) => c, None => return batch.clone() };
    let color_col = match batch.column_by_name(color_field) { Some(c) => c, None => return batch.clone() };
    let x_arr = match x_col.as_any().downcast_ref::<Float64Array>() { Some(a) => a, None => return batch.clone() };
    let color_arr = match color_col.as_any().downcast_ref::<StringArray>() { Some(a) => a, None => return batch.clone() };

    // Collect all (x, group) pairs and the full domain of each.
    use std::collections::HashSet;
    let mut x_vals: Vec<f64> = x_arr.iter().flatten().collect();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    x_vals.dedup();
    let mut groups: Vec<String> = color_arr.iter().flatten().map(str::to_owned).collect();
    groups.sort_unstable();
    groups.dedup();

    if x_vals.is_empty() || groups.is_empty() { return batch.clone(); }

    // Build the set of existing (x, group) keys.
    let mut existing: HashSet<(u64, String)> = HashSet::new();
    for i in 0..batch.num_rows() {
        if x_arr.is_null(i) || color_arr.is_null(i) { continue; }
        let xv = x_arr.value(i);
        let gv = color_arr.value(i).to_owned();
        existing.insert((xv.to_bits(), gv));
    }

    // Build synthetic rows for missing (x, group) pairs.
    let mut new_x: Vec<Option<f64>> = Vec::new();
    let mut new_group: Vec<Option<String>> = Vec::new();
    let mut new_y: Vec<Option<f64>> = Vec::new();
    for xv in &x_vals {
        for gv in &groups {
            if existing.contains(&(xv.to_bits(), gv.clone())) { continue; }
            new_x.push(Some(*xv));
            new_group.push(Some(gv.clone()));
            new_y.push(Some(fill));
        }
    }
    if new_x.is_empty() { return batch.clone(); }

    // Append synthetic rows: build a small batch with (x, color, y) and null all other cols.
    let n_new = new_x.len();
    let n_orig = batch.num_rows();
    let schema = batch.schema();
    let mut combined_cols: Vec<ArrayRef> = Vec::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let orig_col = batch.column(col_idx);
        match field.name().as_str() {
            name if name == x_field => {
                let combined: ArrayRef = Arc::new(Float64Array::from(
                    (0..n_orig).map(|i| if x_arr.is_null(i) { None } else { Some(x_arr.value(i)) })
                        .chain(new_x.iter().copied())
                        .collect::<Vec<Option<f64>>>(),
                ));
                combined_cols.push(combined);
            }
            name if name == color_field => {
                let orig_str = orig_col.as_any().downcast_ref::<StringArray>();
                let combined: ArrayRef = if let Some(orig_str) = orig_str {
                    Arc::new((0..n_orig).map(|i| if orig_str.is_null(i) { None } else { Some(orig_str.value(i)) })
                        .chain(new_group.iter().map(|v| v.as_deref()))
                        .collect::<Vec<Option<&str>>>()
                        .into_iter()
                        .collect::<StringArray>())
                } else {
                    return batch.clone();
                };
                combined_cols.push(combined);
            }
            name if name == y_field => {
                let orig_f64 = orig_col.as_any().downcast_ref::<Float64Array>();
                let combined: ArrayRef = if let Some(orig_f64) = orig_f64 {
                    Arc::new(Float64Array::from(
                        (0..n_orig).map(|i| if orig_f64.is_null(i) { None } else { Some(orig_f64.value(i)) })
                            .chain(new_y.iter().copied())
                            .collect::<Vec<Option<f64>>>(),
                    ))
                } else {
                    return batch.clone();
                };
                combined_cols.push(combined);
            }
            _ => {
                // Append nulls for synthetic rows.
                let extended = arrow::compute::concat(&[
                    orig_col.as_ref(),
                    arrow::array::new_null_array(orig_col.data_type(), n_new).as_ref(),
                ]);
                match extended {
                    Ok(arr) => combined_cols.push(arr),
                    Err(_) => return batch.clone(),
                }
            }
        }
    }
    match RecordBatch::try_new(schema, combined_cols) {
        Ok(b) => b,
        Err(_) => batch.clone(),
    }
}

// ── Tick-label format application ────────────────────────────────────────────
//
// All number/time formatting is delegated to `crate::render::format`, the single
// source of truth for the d3-format grammar and chrono time formatting. This
// module only re-parses pre-computed label strings back to numbers (or epoch-ms
// for temporal axes) and re-applies the requested format.

/// D3 (flexibility campaign): per-channel `Axis(tick_count=N)`. Read from the
/// encoding's typed `axis` style spec. `None` → use the caller's default (10).
fn encoding_axis_tick_count(enc: Option<&crate::spec::encoding::EncodingSpec>) -> Option<usize> {
    enc.and_then(|e| e.axis.as_ref())
        .and_then(|a| a.tick_count)
        .map(|n| n as usize)
}

/// Parse an axis `orient` string into an [`AxisOrient`], validating it against
/// the channel dimension: x accepts top/bottom, y accepts left/right. A
/// cross-dimension value fails loud per the B5 contract. Shared by the
/// per-channel (here) and chart-level (`render::mod`) apply paths so the two
/// cannot drift on the accepted token set or the dimension check.
pub(crate) fn parse_axis_orient(
    value: &str,
    channel: &'static str,
) -> Result<AxisOrient, RenderError> {
    let orient = match value.trim().to_ascii_lowercase().as_str() {
        "top" => AxisOrient::Top,
        "bottom" => AxisOrient::Bottom,
        "left" => AxisOrient::Left,
        "right" => AxisOrient::Right,
        _ => {
            return Err(RenderError::InvalidAxisOrient {
                channel,
                orient: value.to_owned(),
            })
        }
    };
    let ok = match channel {
        "x" => matches!(orient, AxisOrient::Top | AxisOrient::Bottom),
        _ => matches!(orient, AxisOrient::Left | AxisOrient::Right),
    };
    if ok {
        Ok(orient)
    } else {
        Err(RenderError::InvalidAxisOrient { channel, orient: value.to_owned() })
    }
}

/// Parse a `label_overlap` token (`fm.Axis(label_overlap=...)` /
/// `configure_axis(label_overlap=...)`) into a [`LabelOverlap`] strategy (B5 unit
/// 6b). Maps the Vega-style values onto the existing cascade primitives:
/// - `"true"` → [`LabelOverlap::ShowAll`] (show every label, may overlap).
/// - `"false"` / `"greedy"` → [`LabelOverlap::Greedy`] (cascade default cull).
/// - `"parity"` → [`LabelOverlap::Parity`] (stride-2 decimation).
/// - `"rotate"` → [`LabelOverlap::Rotate`] (force the rotate stage).
///
/// An unrecognized token is **bounded to the nearest existing behavior**
/// (`Greedy`, the cascade default) rather than failing loud, since
/// `label_overlap` only biases an already-graceful cascade — an unknown value
/// degrades to the default rather than erroring on a render. Returns `None` for
/// an empty/whitespace token so callers leave the cascade unmodified.
pub(crate) fn parse_label_overlap(value: &str) -> Option<crate::layout::LabelOverlap> {
    use crate::layout::LabelOverlap;
    match value.trim().to_ascii_lowercase().as_str() {
        "" => None,
        "true" => Some(LabelOverlap::ShowAll),
        "parity" => Some(LabelOverlap::Parity),
        "rotate" => Some(LabelOverlap::Rotate),
        // "false", "greedy", and any unrecognized token bound to the cascade
        // default (greedy) — the nearest existing behavior.
        _ => Some(LabelOverlap::Greedy),
    }
}

/// Parse a `title_orient` string into an [`AxisOrient`]. Unlike axis `orient`,
/// all four sides are valid (a title may run perpendicular to its axis), so this
/// only rejects an unrecognized token. Shared with the chart-level apply path.
pub(crate) fn parse_title_orient(
    value: &str,
    channel: &'static str,
) -> Result<AxisOrient, RenderError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "top" => Ok(AxisOrient::Top),
        "bottom" => Ok(AxisOrient::Bottom),
        "left" => Ok(AxisOrient::Left),
        "right" => Ok(AxisOrient::Right),
        _ => Err(RenderError::InvalidAxisOrient { channel, orient: value.to_owned() }),
    }
}

/// Parse a per-channel typed [`AxisStyleSpec`] into the bundled
/// [`AxisStyleOverrides`] ready to drop into an `AxisInput` (B5). A fresh-build
/// over a `Default` bundle via the canonical [`crate::render::axis_style_fill_from`]
/// merge (`fill_only_if_none = false`), the same merge the chart-level
/// `configure_axis` path uses (`fill_only_if_none = true`) — so the two cannot
/// drift on the field set, color parsing, or orient/title_orient validation.
///
/// `label_format` is left `None` here (the fresh-build path never writes it):
/// prepare resolves it through the temporal/numeric format threading
/// (`apply_axis_format_or_thread`) and seeds it onto the bundle afterward. An
/// unparseable color hex string yields `None` (theme fallback) rather than
/// failing. `orient` is the validated override input; the concrete axis side is
/// resolved into `AxisInput.orient` once all override layers merge.
fn encoding_axis_style_overrides(
    axis: Option<&crate::render::chart_config::AxisStyleSpec>,
    channel: &'static str,
) -> Result<crate::layout::AxisStyleOverrides, RenderError> {
    let mut overrides = crate::layout::AxisStyleOverrides::default();
    if let Some(a) = axis {
        crate::render::axis_style_fill_from(&mut overrides, a, channel, false)?;
    }
    Ok(overrides)
}

/// Apply the per-axis `tick_min_step` / `tick_extra` adjustments (B5 unit 2) to a
/// continuous axis's already-generated ticks, mutating the axis's `tick_labels`
/// and projected `tick_projection.major` fractions in lockstep.
///
/// - `tick_min_step` (data units): greedily drop any tick whose raw value is
///   within `min_step` of the previously kept tick (in domain order), thinning a
///   dense axis without disturbing the surviving labels/fractions.
/// - `tick_extra`: append a tick at each domain boundary (`scale min`/`max`) when
///   not already present (within a tiny epsilon), labeled via `format_numeric`
///   and projected to its domain fraction via the scale.
///
/// Both are no-ops for ordinal axes (no `tick_projection`), when neither field is
/// set, or when the tick count drifted, so default output is byte-identical. The
/// effective `tick_extra`/`tick_min_step` are read off the axis input AFTER the
/// chart-level config merge, so the per-channel value already won when present.
/// `tick_extra` formats boundary labels with `format_numeric`, so it is scoped to
/// non-temporal continuous axes (temporal boundary formatting needs spacing
/// context the scale only has internally — flagged as a bounded interpretation).
///
/// `reversed` mirrors the non-ordinal-y label reversal so the raw values pulled
/// from the scale stay index-aligned with the (reversed) labels/fractions.
pub(crate) fn adjust_axis_ticks(
    axis: &mut AxisInput,
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    reversed: bool,
) {
    let tick_min_step = axis.overrides.tick_min_step;
    let tick_extra = axis.overrides.tick_extra.unwrap_or(false);
    if tick_min_step.is_none() && !tick_extra {
        return;
    }
    // Only continuous axes carry a projection; ordinal axes have no numeric
    // domain to thin or bound.
    let Some(proj) = axis.tick_projection.as_mut() else { return };
    let Some(raw_src) = scale.tick_values_raw(tick_count) else { return };
    let is_temporal = matches!(scale, crate::render::scale_resolve::ScaleKind::Time(_));

    let mut raw: Vec<f64> = raw_src;
    if reversed {
        raw.reverse();
    }
    // Guard against index drift (a stale tick count): only act when the raw
    // values, labels, and projected fractions all line up.
    if raw.len() != axis.tick_labels.len() || raw.len() != proj.major.len() {
        return;
    }

    // ── tick_min_step: thin ticks closer than min_step in data space ─────────
    if let Some(min_step) = tick_min_step {
        if min_step > 0.0 && raw.len() > 1 {
            // Walk in domain order so "closer than min_step to the last kept" is
            // well-defined regardless of the reversed display order.
            let mut order: Vec<usize> = (0..raw.len()).collect();
            order.sort_by(|&a, &b| raw[a].total_cmp(&raw[b]));
            let mut keep = vec![false; raw.len()];
            let mut last_kept: Option<f64> = None;
            for &idx in &order {
                let v = raw[idx];
                let far_enough = last_kept.is_none_or(|lk| (v - lk).abs() >= min_step);
                if far_enough {
                    keep[idx] = true;
                    last_kept = Some(v);
                }
            }
            let mut k = 0;
            raw.retain(|_| { let keep_it = keep[k]; k += 1; keep_it });
            let mut k = 0;
            axis.tick_labels.retain(|_| { let keep_it = keep[k]; k += 1; keep_it });
            let mut k = 0;
            proj.major.retain(|_| { let keep_it = keep[k]; k += 1; keep_it });
        }
    }

    // ── tick_extra: append domain-boundary ticks if missing ──────────────────
    if tick_extra && !is_temporal {
        if let Some((d_lo, d_hi)) = scale.data_domain() {
            let eps = ((d_hi - d_lo).abs() * 1e-9).max(f64::MIN_POSITIVE);
            for boundary in [d_lo, d_hi] {
                if !boundary.is_finite() {
                    continue;
                }
                if raw.iter().any(|&v| (v - boundary).abs() <= eps) {
                    continue;
                }
                let frac = scale.value_fractions(&[boundary]);
                let Some(&f) = frac.first() else { continue };
                if !f.is_finite() {
                    continue;
                }
                raw.push(boundary);
                axis.tick_labels.push(crate::render::format::format_numeric(boundary));
                proj.major.push(f);
            }
        }
    }
}

/// Resolve the tick-label format `(spec, type)` for a positional channel.
///
/// Precedence: a per-channel `Axis(label_format=, label_format_type=)` (in the
/// `encoding.axis` map) wins over the shorthand `encoding.format`/`format_type`.
/// Returns `(None, None)` when neither is set so default formatting is preserved.
///
/// `pub(crate)` so `scene_build.rs` can re-derive the format spec for
/// independent-axis per-panel label formatting without duplicating the
/// precedence logic.
pub(crate) fn resolve_axis_label_format(
    enc: Option<&crate::spec::encoding::EncodingSpec>,
) -> (Option<String>, Option<String>) {
    let Some(e) = enc else { return (None, None) };
    let axis = e.axis.as_ref();
    if let Some(fmt) = axis.and_then(|a| a.label_format.clone()) {
        let ty = axis.and_then(|a| a.label_format_type.clone());
        return (Some(fmt), ty);
    }
    (e.format.clone(), e.format_type.clone())
}

/// Apply a resolved tick-label format to one axis, returning the (possibly
/// reformatted) labels and the `label_format_override` to thread into the
/// `AxisInput`.
///
/// Two paths:
/// - **Temporal** axis with an explicit strftime pattern (`format` contains `%`
///   and `format_type` is `"time"` or unset): the raw epoch-ms tick values are
///   pulled from the scale and formatted directly via `chrono` here. No override
///   is threaded (the strings are already final; re-parsing them downstream
///   would fail). Returns `(formatted, None)`.
/// - **Numeric** axis (or any non-temporal): the spec is threaded forward as the
///   `label_format_override` (D3 root-cause fix for `prepare.rs:538`), and the
///   labels are returned unchanged. `render/mod.rs::apply_label_format_to_axis`
///   then applies it centrally — lossless because numeric label strings reparse
///   to f64. Threading (not pre-applying) also lets chart-level
///   `configure_axis` defer to the per-channel override via its `is_none()` gate.
fn apply_axis_format_or_thread(
    labels: Vec<String>,
    format: Option<String>,
    format_type: Option<&str>,
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    reversed: bool,
) -> (Vec<String>, Option<String>) {
    use crate::render::format::format_time_spec;
    let Some(fmt) = format else { return (labels, None) };
    let is_time_pattern =
        fmt.contains('%') && (format_type == Some("time") || format_type.is_none());
    if is_time_pattern {
        if let Some(mut values) = scale.temporal_tick_values(tick_count) {
            if reversed {
                values.reverse();
            }
            // Guard against index drift between the cached labels and a fresh
            // tick computation: only substitute when the counts line up.
            if values.len() == labels.len() {
                let out = values
                    .into_iter()
                    .map(|ms| format_time_spec(ms, &fmt))
                    .collect();
                return (out, None);
            }
        }
        // Temporal pattern but no raw values available (non-Time scale or drift):
        // fall back to the string-level path, then thread nothing further.
        return (apply_tick_format(labels, Some(&fmt), format_type), None);
    }
    // Numeric / non-temporal: thread the spec to the override, apply later.
    (labels, Some(fmt))
}

/// How [`build_axis_tick_inputs`] applies the resolved tick-label format. The two
/// modes are the two byte-distinct format disciplines the render spine already
/// has — surfaced as one enum so the shared and independent axis paths run the
/// same tick-derivation function while keeping their (intentionally different)
/// format handling.
pub(crate) enum TickFormatMode<'a> {
    /// Shared/global axes: format temporal axes directly, but THREAD a numeric
    /// format forward (returned as the second tuple element) so chart-level
    /// `configure_axis` can still defer to the per-channel override via its
    /// `is_none()` gate, with `apply_label_format_to_axis` applying it centrally
    /// later. Mirrors [`apply_axis_format_or_thread`].
    Thread { format: Option<String>, format_type: Option<&'a str> },
    /// Independent per-panel axes: apply the format IMMEDIATELY via
    /// [`apply_tick_format`] and thread nothing (the per-panel layout is built
    /// directly in `scene_build`, bypassing `apply_label_format_to_axis`).
    Immediate { format: Option<&'a str>, format_type: Option<&'a str> },
}

/// Derive one axis's tick labels and [`TickProjection`] from a resolved scale —
/// the single source of truth for the sequence the shared (global) and
/// independent (per-panel facet) axis paths both run: `tick_labels(count)`, the
/// non-ordinal-y label/fraction reversal, format application, and the
/// fraction-projection build.
///
/// Returns `(labels, projection, threaded_label_format)`. The threaded format is
/// `Some` only under [`TickFormatMode::Thread`] for a numeric axis (the caller
/// seeds it onto `AxisInput.overrides.label_format`); it is always `None` under
/// [`TickFormatMode::Immediate`].
///
/// `minor` is supplied by the caller (already projected, NOT reversed) so this fn
/// stays theme-agnostic: the shared path passes the `theme.grid.minor`-gated
/// `minor_tick_fractions()`; the independent path passes `Vec::new()` (per-panel
/// independent axes do not rebuild minor ticks — preserved from the pre-extraction
/// behavior). An ordinal scale yields no projection (`None`), matching uniform-
/// slot placement.
pub(crate) fn build_axis_tick_inputs(
    scale: &crate::render::scale_resolve::ScaleKind,
    tick_count: usize,
    mode: TickFormatMode<'_>,
    is_y: bool,
    minor: Vec<f64>,
) -> (Vec<String>, Option<TickProjection>, Option<String>) {
    use crate::render::scale_resolve::ScaleKind;
    // Non-ordinal y axes display high domain values at the top, so labels and the
    // projected fractions are reversed in lockstep (the pixel range is already
    // inverted in scale_resolve). Ordinal y keeps domain order (top-down for
    // heatmaps/confusion matrices).
    let reverse = is_y && !matches!(scale, ScaleKind::Ordinal(_));

    let mut labels = scale.tick_labels(tick_count);
    let mut fractions = scale.tick_fractions(tick_count);
    let padding_frac = scale.padding_fraction();
    if reverse {
        labels.reverse();
        fractions.reverse();
    }

    let (labels, threaded) = match mode {
        TickFormatMode::Thread { format, format_type } => apply_axis_format_or_thread(
            labels,
            format,
            format_type,
            scale,
            tick_count,
            // The temporal raw values must be reversed in lockstep with the
            // already-reversed labels so index `i` still aligns.
            reverse,
        ),
        TickFormatMode::Immediate { format, format_type } => {
            (apply_tick_format(labels, format, format_type), None)
        }
    };

    // A non-empty major-fraction vec means a continuous scale → scale-projected
    // placement; an empty vec means ordinal/discretizing → no carrier (uniform
    // slots), byte-identical to the pre-extraction `None`.
    let projection = (!fractions.is_empty()).then(|| TickProjection {
        padding_frac,
        major: fractions,
        minor,
    });

    (labels, projection, threaded)
}

/// D12: apply an encoding-level `format` string to pre-computed tick label strings.
///
/// The scale's `tick_labels()` method returns pre-formatted strings (via
/// `format_numeric`). When the encoding carries an explicit `format` string
/// (e.g. `.2f`, `~s`, `,`), we re-parse each label back to f64 and re-format it
/// via the canonical d3-format grammar in `crate::render::format`.
///
/// When `format_type == "time"`, labels are treated as epoch-ms integers. A
/// format string containing `%` is interpreted as a `chrono` strftime pattern
/// (e.g. `"%b %Y"`); otherwise the default spacing-keyed temporal formatter is
/// used, preserving the prior byte output. Labels that fail to parse are left
/// unchanged (ordinal labels, already-formatted strings, etc.).
pub(crate) fn apply_tick_format(
    labels: Vec<String>,
    format: Option<&str>,
    format_type: Option<&str>,
) -> Vec<String> {
    use crate::render::format::{format_time, format_time_spec, parse_format_spec, format_parsed};
    let is_time = format_type == Some("time");
    // Non-time with no format string is a pure pass-through (byte-identical).
    if !is_time && format.is_none() {
        return labels;
    }
    let time_pattern = format.filter(|f| f.contains('%'));
    let num_spec = format.map(parse_format_spec);
    labels
        .into_iter()
        .map(|raw| {
            if is_time {
                let epoch_ms = raw
                    .parse::<i64>()
                    .ok()
                    .or_else(|| raw.parse::<f64>().ok().map(|f| f as i64));
                match (epoch_ms, time_pattern) {
                    (Some(ms), Some(pat)) => format_time_spec(ms, pat),
                    (Some(ms), None) => format_time(ms, 86_400_000),
                    (None, _) => raw,
                }
            } else if let (Ok(v), Some(spec)) = (raw.parse::<f64>(), num_spec.as_ref()) {
                format_parsed(v, spec)
            } else {
                raw // ordinal — pass through
            }
        })
        .collect()
}

#[cfg(test)]
mod tick_format_tests {
    use super::*;

    /// Regression: apply_tick_format with a d3 percent spec must format numerics.
    /// This exercises the code path used by scene_build.rs independent-axis label
    /// formatting — where apply_tick_format is called directly with the raw format
    /// spec from resolve_axis_label_format.
    #[test]
    fn tick_format_percent_spec_applied_to_numerics() {
        let labels = vec!["0".to_string(), "0.5".to_string(), "1".to_string()];
        let out = apply_tick_format(labels, Some(".0%"), None);
        // ".0%" of 0.5 = "50%"
        assert_eq!(out, vec!["0%", "50%", "100%"]);
    }

    fn fmt_labels(labels: &[&str], format: &str) -> Vec<String> {
        apply_tick_format(
            labels.iter().map(|s| s.to_string()).collect(),
            Some(format),
            None,
        )
    }

    #[test]
    fn currency_format_prepends_dollar() {
        // "$,.0f" should produce "$10", "$1,000", etc.
        assert_eq!(fmt_labels(&["10", "1000", "50"], "$,.0f"), vec!["$10", "$1,000", "$50"]);
    }
    #[test]
    fn dotf_two_decimals() {
        assert_eq!(fmt_labels(&["2", "2.5", "3"], ".2f"), vec!["2.00", "2.50", "3.00"]);
    }
    #[test]
    fn dot1f_one_decimal() {
        assert_eq!(fmt_labels(&["1", "1.5", "2"], ".1f"), vec!["1.0", "1.5", "2.0"]);
    }
    #[test]
    fn percent_format() {
        assert_eq!(fmt_labels(&["0.1", "0.123", "0.5"], ".1%"), vec!["10.0%", "12.3%", "50.0%"]);
    }
    #[test]
    fn comma_thousands_integer() {
        assert_eq!(
            fmt_labels(&["1000", "10000", "1234567"], ","),
            vec!["1,000", "10,000", "1,234,567"]
        );
    }
    #[test]
    fn comma_dotf_thousands_decimal() {
        assert_eq!(fmt_labels(&["1000", "2000", "3000"], ",.0f"), vec!["1,000", "2,000", "3,000"]);
    }
    #[test]
    fn d_integer_format() {
        assert_eq!(fmt_labels(&["42", "1000", "3.7"], "d"), vec!["42", "1000", "4"]);
    }
    #[test]
    fn si_kilo() {
        // d3 `s` precision is SIGNIFICANT digits: ".2s" of 1200 → "1.2k".
        assert_eq!(fmt_labels(&["1200", "1500"], ".2s"), vec!["1.2k", "1.5k"]);
        // ".3s" keeps three sig figs → "1.20k".
        assert_eq!(fmt_labels(&["1200"], ".3s"), vec!["1.20k"]);
    }
    #[test]
    fn si_trim_megabyte() {
        // The audit-motivating case: "~s" trims insignificant zeros → "1.5M".
        assert_eq!(fmt_labels(&["1500000"], "~s"), vec!["1.5M"]);
    }
    #[test]
    fn ordinal_passthrough() {
        assert_eq!(fmt_labels(&["setosa", "versicolor"], ".2f"), vec!["setosa", "versicolor"]);
    }
    #[test]
    fn no_format_passthrough() {
        let labels = vec!["1".to_string(), "2".to_string()];
        let out = apply_tick_format(labels.clone(), None, None);
        assert_eq!(out, labels);
    }
    #[test]
    fn time_pattern_via_chrono() {
        // 2020-01-01T00:00:00Z = 1577836800000 ms; "%b %Y" → "Jan 2020".
        let out = apply_tick_format(
            vec!["1577836800000".to_string()],
            Some("%b %Y"),
            Some("time"),
        );
        assert_eq!(out, vec!["Jan 2020"]);
    }
    #[test]
    fn time_default_when_no_pattern() {
        // No `%` pattern → default day-spacing temporal formatter (byte-stable).
        let out = apply_tick_format(
            vec!["1577836800000".to_string()],
            Some(","),
            Some("time"),
        );
        assert_eq!(out, vec!["2020-01-01"]);
    }
}

/// Resolve a named column from `batch` as a `StringArray`, returning a typed
/// error when the column is absent or has a non-Utf8 type. Shared by all four
/// facet distinct-value scan helpers below so the column-lookup + downcast
/// pattern lives in exactly one place.
fn facet_str_arr<'a>(
    batch: &'a RecordBatch,
    field: &str,
) -> Result<&'a arrow::array::StringArray, RenderError> {
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    col.as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!(
                "facet field '{field}' must be Utf8 (Phase 7 limitation)"
            ))
        })
}

/// Partition a RecordBatch by a Utf8 field, returning `(value, filtered_batch)`
/// pairs in first-appearance order. Used by facet-before-transform to split the
/// input into per-panel subsets before running transforms.
fn partition_batch_by_field(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<(String, RecordBatch)>, RenderError> {
    use arrow::array::BooleanArray;
    use arrow::compute::filter_record_batch;
    let arr = facet_str_arr(batch, field)?;
    // Collect distinct values in first-appearance order.
    let mut order: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for v in arr.iter().flatten() {
        let s = v.to_string();
        if seen.insert(s.clone()) {
            order.push(s);
        }
    }
    let mut result = Vec::with_capacity(order.len());
    for value in order {
        let mask: BooleanArray = arr
            .iter()
            .map(|v| Some(v.map(|s| s == value.as_str()).unwrap_or(false)))
            .collect();
        let filtered = filter_record_batch(batch, &mask)
            .map_err(|e| RenderError::ScaleResolutionFailed(format!("partition filter: {e}")))?;
        result.push((value, filtered));
    }
    Ok(result)
}

/// Ensure a RecordBatch has a Utf8 column named `field` with the constant `value`.
/// If the column already exists, return the batch unchanged. Otherwise, append a
/// new Utf8 column filled with `value` repeated for every row. This is used to
/// re-inject the facet column into transform outputs that replace the batch
/// entirely (e.g. Smooth, KDE, Histogram).
fn inject_facet_column(batch: &RecordBatch, field: &str, value: &str) -> RecordBatch {
    if batch.column_by_name(field).is_some() {
        return batch.clone();
    }
    let n = batch.num_rows();
    let constant: ArrayRef = Arc::new(StringArray::from(vec![value; n]));
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(field, DataType::Utf8, false)));
    let new_schema = Arc::new(Schema::new(fields));
    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(constant);
    RecordBatch::try_new(new_schema, columns)
        .expect("inject_facet_column: schema + columns must be consistent")
}

fn group_rows_by_field(batch: &RecordBatch, field: &str) -> Result<Vec<FacetGroup>, RenderError> {
    let arr = facet_str_arr(batch, field)?;
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for v in arr.iter().flatten() {
        let s = v.to_string();
        if !counts.contains_key(&s) {
            order.push(s.clone());
        }
        *counts.entry(s).or_insert(0) += 1;
    }
    Ok(order
        .into_iter()
        .map(|v| FacetGroup {
            key: FacetKey { field: field.to_string(), value: v.clone() },
            n_rows: counts[&v],
            row_key: None,
        })
        .collect())
}

/// Grid-mode two-field grouping: one `FacetGroup` per `(row_val, col_val)` pair
/// in row-major order.
///
/// Distinct row values appear in first-appearance order from the data; within
/// each row the col values appear in first-appearance order. `key` carries the
/// col dimension (drives the column-header strip); `row_key` carries the row
/// dimension (drives the row-header strip and secondary batch filter).
fn group_rows_by_two_fields(
    batch: &RecordBatch,
    col_field: &str,
    row_field: &str,
) -> Result<Vec<FacetGroup>, RenderError> {
    let col_arr = facet_str_arr(batch, col_field)?;
    let row_arr = facet_str_arr(batch, row_field)?;

    // Collect distinct row values and col values (first-appearance order).
    let mut row_order: Vec<String> = Vec::new();
    let mut row_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut col_order: Vec<String> = Vec::new();
    let mut col_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Count rows per (row_val, col_val) pair.
    let mut counts: std::collections::HashMap<(String, String), u64> = std::collections::HashMap::new();

    for i in 0..batch.num_rows() {
        let row_v = match row_arr.is_null(i) { true => continue, false => row_arr.value(i).to_string() };
        let col_v = match col_arr.is_null(i) { true => continue, false => col_arr.value(i).to_string() };
        if row_seen.insert(row_v.clone()) {
            row_order.push(row_v.clone());
        }
        if col_seen.insert(col_v.clone()) {
            col_order.push(col_v.clone());
        }
        *counts.entry((row_v, col_v)).or_insert(0) += 1;
    }

    // Emit groups in row-major order: (r0,c0), (r0,c1), ..., (r1,c0), ...
    let mut groups = Vec::with_capacity(row_order.len() * col_order.len());
    for row_v in &row_order {
        for col_v in &col_order {
            let n_rows = counts.get(&(row_v.clone(), col_v.clone())).copied().unwrap_or(0);
            groups.push(FacetGroup {
                key: FacetKey { field: col_field.to_string(), value: col_v.clone() },
                n_rows,
                row_key: Some(FacetKey { field: row_field.to_string(), value: row_v.clone() }),
            });
        }
    }
    Ok(groups)
}

/// `(col_val, row_val)` composite key for a single grid-mode partition.
type GridPartitionKey = (String, String);

/// Partition a RecordBatch by two Utf8 fields (col_field, row_field), returning
/// `((col_val, row_val), filtered_batch)` pairs in row-major first-appearance
/// order. Used by facet-before-transform in grid mode.
fn partition_batch_by_two_fields(
    batch: &RecordBatch,
    col_field: &str,
    row_field: &str,
) -> Result<Vec<(GridPartitionKey, RecordBatch)>, RenderError> {
    use arrow::array::{Array, BooleanArray};
    use arrow::compute::filter_record_batch;

    let col_arr = facet_str_arr(batch, col_field)?;
    let row_arr = facet_str_arr(batch, row_field)?;

    // Collect distinct (row_val, col_val) pairs in row-major first-appearance order.
    let mut row_order: Vec<String> = Vec::new();
    let mut row_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut col_order: Vec<String> = Vec::new();
    let mut col_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for i in 0..batch.num_rows() {
        if row_arr.is_null(i) || col_arr.is_null(i) { continue; }
        let rv = row_arr.value(i).to_string();
        let cv = col_arr.value(i).to_string();
        if row_seen.insert(rv.clone()) { row_order.push(rv); }
        if col_seen.insert(cv.clone()) { col_order.push(cv); }
    }

    let mut result = Vec::with_capacity(row_order.len() * col_order.len());
    for row_v in &row_order {
        for col_v in &col_order {
            let mask: BooleanArray = (0..batch.num_rows())
                .map(|i| {
                    if col_arr.is_null(i) || row_arr.is_null(i) { return Some(false); }
                    Some(col_arr.value(i) == col_v.as_str() && row_arr.value(i) == row_v.as_str())
                })
                .collect();
            // Skip (row_v, col_v) pairs that have no rows in this batch (sparse cross-products).
            if !mask.iter().any(|v| v == Some(true)) {
                continue;
            }
            let filtered = filter_record_batch(batch, &mask)
                .map_err(|e| RenderError::ScaleResolutionFailed(format!("partition filter: {e}")))?;
            result.push(((col_v.clone(), row_v.clone()), filtered));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AuxLegendInput, SymbolKind};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn parse_label_overlap_maps_vega_tokens() {
        use crate::layout::LabelOverlap;
        assert_eq!(parse_label_overlap("true"), Some(LabelOverlap::ShowAll));
        assert_eq!(parse_label_overlap("parity"), Some(LabelOverlap::Parity));
        assert_eq!(parse_label_overlap("rotate"), Some(LabelOverlap::Rotate));
        assert_eq!(parse_label_overlap("greedy"), Some(LabelOverlap::Greedy));
        assert_eq!(parse_label_overlap("false"), Some(LabelOverlap::Greedy));
        // Case-insensitive + whitespace-tolerant.
        assert_eq!(parse_label_overlap("  PARITY "), Some(LabelOverlap::Parity));
        // Empty → no override (leave the cascade unmodified).
        assert_eq!(parse_label_overlap(""), None);
        // Unrecognized → bounded to the cascade default (greedy), not an error.
        assert_eq!(parse_label_overlap("nonsense"), Some(LabelOverlap::Greedy));
    }

    // --- SPINE-08: build_axis_input + resolve_axis_title -------------------------

    /// The 3-way title idiom: absent → fallback; present-empty → suppress;
    /// present-nonempty → use it (whitespace counts as empty).
    #[test]
    fn resolve_axis_title_three_way() {
        assert_eq!(resolve_axis_title(None, Some("field".into())), Some("field".into()));
        assert_eq!(resolve_axis_title(None, None), None);
        assert_eq!(resolve_axis_title(Some(""), Some("field".into())), None);
        assert_eq!(resolve_axis_title(Some("   "), Some("field".into())), None);
        assert_eq!(resolve_axis_title(Some("Custom"), Some("field".into())), Some("Custom".into()));
        // Explicit non-empty wins even with no fallback.
        assert_eq!(resolve_axis_title(Some("Custom"), None), Some("Custom".into()));
    }

    /// `Channel` carries the orient default and the reverse policy.
    #[test]
    fn channel_carries_orient_default_and_reverse_policy() {
        assert_eq!(Channel::X.default_orient(), AxisOrient::Bottom);
        assert_eq!(Channel::Y.default_orient(), AxisOrient::Left);
        assert!(!Channel::X.reverses());
        assert!(Channel::Y.reverses());
        assert_eq!(Channel::X.token(), "x");
        assert_eq!(Channel::Y.token(), "y");
    }

    /// A continuous (Linear) scale over [0, 10]. Tick labels run "0".."10" in
    /// domain order; the non-ordinal-y reversal flips them.
    fn linear_scale_0_10() -> crate::render::scale_resolve::ScaleKind {
        use crate::render::scale_resolve::ScaleKind;
        use crate::scale::linear::LinearScale;
        ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 10.0],
            vec![0.0, 1.0],
            false,
            false,
        ))
    }

    /// An ordinal scale over three categories (no continuum to reverse).
    fn ordinal_scale_abc() -> crate::render::scale_resolve::ScaleKind {
        use crate::render::scale_resolve::ScaleKind;
        use crate::scale::ordinal::OrdinalScale;
        ScaleKind::Ordinal(OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into()],
            vec![0.0, 1.0],
            0.0,
        ))
    }

    /// x and y axis inputs built from the SAME continuous scale agree on every
    /// field EXCEPT that the non-ordinal y reverses its tick labels and
    /// projected major fractions, and picks the Left orient default. This is the
    /// load-bearing parity check for the SPINE-08 extraction: the y derivation is
    /// NOT mechanically identical to x — it interleaves the reversal.
    #[test]
    fn build_axis_input_x_vs_y_continuous_reverse_parity() {
        let theme = crate::layout::ThemeInputs::default();
        let x_enc = EncodingSpec { field: "v".into(), ..Default::default() };
        let y_enc = EncodingSpec { field: "v".into(), ..Default::default() };
        let scale = linear_scale_0_10();

        let x = build_axis_input(Channel::X, Some(&x_enc), Some(&x_enc), &scale, 10, &theme)
            .expect("x axis input");
        let y = build_axis_input(Channel::Y, Some(&y_enc), Some(&y_enc), &scale, 10, &theme)
            .expect("y axis input");

        // Orient defaults differ by channel.
        assert_eq!(x.orient, AxisOrient::Bottom);
        assert_eq!(y.orient, AxisOrient::Left);
        // Title falls through to the field name on both.
        assert_eq!(x.title.as_deref(), Some("v"));
        assert_eq!(y.title.as_deref(), Some("v"));
        // Show toggles default true on both.
        assert!(x.show_labels && x.show_ticks && x.show_domain && x.show_grid);
        assert!(y.show_labels && y.show_ticks && y.show_domain && y.show_grid);

        // The y tick labels are the x tick labels reversed (non-ordinal-y rule).
        assert!(!x.tick_labels.is_empty(), "continuous axis must have labels");
        let mut x_reversed = x.tick_labels.clone();
        x_reversed.reverse();
        assert_eq!(y.tick_labels, x_reversed);

        // The projected major fractions are likewise reversed in lockstep.
        let x_major = x.tick_projection.as_ref().expect("x projection").major.clone();
        let y_major = y.tick_projection.as_ref().expect("y projection").major.clone();
        let mut x_major_reversed = x_major.clone();
        x_major_reversed.reverse();
        assert_eq!(y_major, x_major_reversed);
    }

    /// On an ORDINAL scale the non-ordinal-y reversal does NOT apply: y keeps the
    /// same order as x (top-down for heatmaps / confusion matrices). The only
    /// remaining difference is the orient default.
    #[test]
    fn build_axis_input_y_ordinal_does_not_reverse() {
        let theme = crate::layout::ThemeInputs::default();
        let enc = EncodingSpec { field: "cat".into(), ..Default::default() };
        let scale = ordinal_scale_abc();

        let x = build_axis_input(Channel::X, Some(&enc), Some(&enc), &scale, 10, &theme)
            .expect("x axis input");
        let y = build_axis_input(Channel::Y, Some(&enc), Some(&enc), &scale, 10, &theme)
            .expect("y axis input");

        assert_eq!(x.tick_labels, y.tick_labels, "ordinal y must not reverse");
        assert_eq!(x.orient, AxisOrient::Bottom);
        assert_eq!(y.orient, AxisOrient::Left);
    }

    /// Spec-level title wins over the layer-0 title, which wins over the field
    /// name; an explicit empty spec-level title suppresses entirely.
    #[test]
    fn build_axis_input_title_precedence() {
        let theme = crate::layout::ThemeInputs::default();
        let scale = linear_scale_0_10();

        // spec-level title wins over layer-0 title.
        let rendering = EncodingSpec {
            field: "v".into(),
            title: Some("layer".into()),
            ..Default::default()
        };
        let spec_enc = EncodingSpec {
            field: "v".into(),
            title: Some("spec".into()),
            ..Default::default()
        };
        let a = build_axis_input(Channel::X, Some(&rendering), Some(&spec_enc), &scale, 10, &theme)
            .expect("axis input");
        assert_eq!(a.title.as_deref(), Some("spec"));

        // explicit empty spec-level title suppresses (no fallback to field/layer).
        let spec_suppress = EncodingSpec {
            field: "v".into(),
            title: Some("".into()),
            ..Default::default()
        };
        let b = build_axis_input(
            Channel::X,
            Some(&rendering),
            Some(&spec_suppress),
            &scale,
            10,
            &theme,
        )
        .expect("axis input");
        assert_eq!(b.title, None);

        // absent spec-level title falls through to the layer-0 title.
        let spec_absent = EncodingSpec { field: "v".into(), ..Default::default() };
        let c = build_axis_input(Channel::X, Some(&rendering), Some(&spec_absent), &scale, 10, &theme)
            .expect("axis input");
        assert_eq!(c.title.as_deref(), Some("layer"));
    }

    // --- archaeology #7: generalized facet extent pin ---------------------------

    /// A two-column batch: a Float64 value field `v` plus a Utf8 hue field `g`.
    /// Used to drive `fix_transform_extents_for_facet` over the full pre-facet
    /// dataset (the global extent must ignore `g`).
    ///
    /// Values are [0.3, 0.5, 1.2, 4.8, 9.7] (raw range 0.3..9.7), chosen so
    /// that nicing with bin_count=10 actually changes the extent: nice_step
    /// produces step=1.0, which rounds 0.3 down to 0.0 and 9.7 up to 10.0.
    fn extent_pin_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.3, 0.5, 1.2, 4.8, 9.7])),
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "b"])),
            ],
        )
        .unwrap()
    }

    /// A faceted `Bin` transform with `extent=None` gets its extent pinned to the
    /// niced global range over the full pre-facet batch.
    #[test]
    fn fix_extents_pins_bin_to_niced_global_extent() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::core::TransformSpec;
        let batch = extent_pin_batch();
        let spec = BinSpec {
            field: "v".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Bin(spec.clone())], &batch);
        let TransformSpec::Bin(pinned) = &out[0] else {
            panic!("expected a Bin transform back");
        };
        // Concrete expected value: raw range is (0.3, 9.7); nice_step(0.3, 9.7, 10)
        // → step0=0.94, log10(0.94)≈-0.027 → floor=-1 → pow10=0.1, frac=9.4 ≥ 7.5
        // → nice_frac=10.0 → step=1.0; niced lo=floor(0.3/1.0)*1.0=0.0,
        // niced hi=ceil(9.7/1.0)*1.0=10.0.  The niced range differs from the raw
        // range, so this assertion actively exercises the nicing path.
        assert_eq!(
            pinned.extent,
            Some((0.0, 10.0)),
            "Bin extent must be the niced (0.0, 10.0), not the raw (0.3, 9.7)"
        );
        // Also confirm orchestration: the pinned value matches global_extent's output.
        assert_eq!(
            pinned.extent,
            crate::transform::bin::global_extent(&spec, &batch),
            "Bin extent must be pinned to the niced global extent"
        );
    }

    /// A faceted `Violin` transform with `extent=None` gets its extent pinned to
    /// the global range over the full pre-facet batch.
    #[test]
    fn fix_extents_pins_violin_to_global_extent() {
        use crate::transform::core::TransformSpec;
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::violin::ViolinSpec;
        let batch = extent_pin_batch();
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: Vec::new(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Violin(spec.clone())], &batch);
        let TransformSpec::Violin(pinned) = &out[0] else {
            panic!("expected a Violin transform back");
        };
        // Violin pins to the RAW global (min, max) over v = (0.3, 9.7). Unlike
        // Bin, KDE/Violin do not nice their extent (no bin edges to align), so
        // the pinned value is the unrounded global range, not (0.0, 10.0).
        assert_eq!(
            pinned.extent,
            Some((0.3, 9.7)),
            "Violin extent must be pinned to the raw global (min, max) over v"
        );
    }

    /// The previously-excluded multi-group case: a faceted `Kde` transform WITH a
    /// `groupby` (hue) and `extent=None` now gets pinned to the global extent over
    /// the full dataset, so panels and hue groups share one value axis (spec §8).
    #[test]
    fn fix_extents_pins_kde_with_groupby_multi_group() {
        use crate::transform::core::TransformSpec;
        use crate::transform::kde::KdeSpec;
        let batch = extent_pin_batch();
        let spec = KdeSpec {
            field: "v".into(),
            bandwidth: crate::transform::kde::BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: "gaussian".into(),
            groupby: vec!["g".into()],
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Kde(spec.clone())], &batch);
        let TransformSpec::Kde(pinned) = &out[0] else {
            panic!("expected a Kde transform back");
        };
        // Raw global extent over v = (0.3, 9.7) regardless of the `g` groupby —
        // the multi-group fix the old groupby.is_some() early-return blocked. KDE
        // does not nice (only Bin does), so the pin is the raw global min/max.
        assert_eq!(
            pinned.extent,
            Some((0.3, 9.7)),
            "KDE with a groupby must still be pinned to the full-dataset extent"
        );
    }

    /// A user-provided explicit `extent` must never be overridden by the pin.
    #[test]
    fn fix_extents_respects_user_extent() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::core::TransformSpec;
        use crate::transform::kde::KdeSpec;
        let batch = extent_pin_batch();
        let user = Some((-5.0, 100.0));
        let bin = BinSpec {
            field: "v".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: user,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            name: None,
        };
        let kde = KdeSpec {
            field: "v".into(),
            bandwidth: crate::transform::kde::BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: user,
            cumulative: false,
            shared_extent: false,
            kernel: "gaussian".into(),
            groupby: vec![],
            name: None,
        };
        let out = fix_transform_extents_for_facet(
            &[TransformSpec::Bin(bin), TransformSpec::Kde(kde)],
            &batch,
        );
        let TransformSpec::Bin(pinned_bin) = &out[0] else {
            panic!("expected a Bin transform back");
        };
        let TransformSpec::Kde(pinned_kde) = &out[1] else {
            panic!("expected a Kde transform back");
        };
        assert_eq!(pinned_bin.extent, user, "user Bin extent must be preserved");
        assert_eq!(pinned_kde.extent, user, "user KDE extent must be preserved");
    }

    /// Pins the niced-vs-raw contract between `bin::global_extent`, `kde::global_extent`,
    /// and `violin::global_extent`.
    ///
    /// For the shared fixture data [0.3, 0.5, 1.2, 4.8, 9.7] (raw range 0.3..9.7):
    ///
    /// - **Bin** nices its extent to align bin edges: step=1.0 → (0.0, 10.0).
    /// - **KDE** and **Violin** pin to the raw (min, max) because they have no bin
    ///   edges to align: (0.3, 9.7).
    ///
    /// This test guards the exact defect class that regressed in archaeology bug #7:
    /// the `fix_extents_pins_violin_to_global_extent` and
    /// `fix_extents_pins_kde_with_groupby_multi_group` tests had their assertions
    /// set to the niced value (0.0, 10.0) instead of the raw value (0.3, 9.7).
    /// Any future change that makes KDE/Violin nice, or makes Bin not nice, will
    /// trip this test and force an explicit re-evaluation of the contract.
    #[test]
    fn global_extent_nices_for_bin_but_raw_for_kde_and_violin() {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::kde::{BandwidthSpec, KdeSpec};
        use crate::transform::violin::ViolinSpec;

        let batch = extent_pin_batch();

        let bin_spec = BinSpec {
            field: "v".into(),
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            name: None,
        };
        let kde_spec = KdeSpec {
            field: "v".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: "gaussian".into(),
            groupby: vec![],
            name: None,
        };
        let violin_spec = ViolinSpec {
            field: "v".into(),
            groupby: Vec::new(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };

        // Bin nices: raw (0.3, 9.7) with bin_count=10 → step=1.0 → (0.0, 10.0).
        assert_eq!(
            crate::transform::bin::global_extent(&bin_spec, &batch),
            Some((0.0, 10.0)),
            "Bin must return the NICED global extent, not the raw (0.3, 9.7)"
        );
        // KDE does not nice: returns the raw (min, max).
        assert_eq!(
            crate::transform::kde::global_extent(&kde_spec, &batch),
            Some((0.3, 9.7)),
            "KDE must return the RAW global extent (0.3, 9.7), not the niced (0.0, 10.0)"
        );
        // Violin does not nice: returns the raw (min, max).
        assert_eq!(
            crate::transform::violin::global_extent(&violin_spec, &batch),
            Some((0.3, 9.7)),
            "Violin must return the RAW global extent (0.3, 9.7), not the niced (0.0, 10.0)"
        );
    }

    // --- archaeology R5: 2-D facet extent pin (Kde2D / Bin2D) -------------------

    /// A three-column batch: Float64 `x`, Float64 `y`, and a Utf8 facet field `p`.
    /// Panel "A" lives at x in 0..1, y in 0..1; panel "B" lives at x in 10..11,
    /// y in 100..101. The two panels' x AND y ranges are fully DISJOINT, so a
    /// per-panel extent (the pre-R5 bug) would give panel A x∈[0,1]/y∈[0,1] and
    /// panel B x∈[10,11]/y∈[100,101). The pin computes the global extent over the
    /// whole batch → x∈[0,11], y∈[0,101] for BOTH panels.
    fn extent_pin_batch_2d() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("p", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    0.0, 0.5, 1.0, // panel A
                    10.0, 10.5, 11.0, // panel B
                ])),
                Arc::new(Float64Array::from(vec![
                    0.0, 0.5, 1.0, // panel A
                    100.0, 100.5, 101.0, // panel B
                ])),
                Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
            ],
        )
        .unwrap()
    }

    /// A faceted `Kde2D` with no extent gets pinned to the RAW global 2-D range
    /// over the full pre-facet batch, so every panel shares the same x AND y
    /// extents. Discriminating: an unpinned (per-panel) extent would not span the
    /// global x∈[0,11]/y∈[0,101] range.
    #[test]
    fn fix_extents_pins_kde2d_to_global_2d_extent() {
        use crate::transform::core::TransformSpec;
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::kde_2d::Kde2DSpec;
        let batch = extent_pin_batch_2d();
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            groupby: vec![],
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Kde2D(spec.clone())], &batch);
        let TransformSpec::Kde2D(pinned) = &out[0] else {
            panic!("expected a Kde2D transform back");
        };
        // Raw global extent over the whole batch: x∈[0,11], y∈[0,101]. Kde2D does
        // not nice. The disjoint per-panel ranges make this fail if unpinned.
        assert_eq!(
            pinned.extent,
            Some((0.0, 11.0, 0.0, 101.0)),
            "Kde2D extent must be pinned to the raw global 2-D range across panels"
        );
        // Orchestration sanity: the pinned value matches global_extent's output.
        assert_eq!(
            pinned.extent,
            crate::transform::kde_2d::global_extent(&spec, &batch),
        );
    }

    /// A faceted `Bin2D` with no extent gets pinned per-axis to the RAW global
    /// range over the full pre-facet batch, so every panel shares the same x AND
    /// y bin edges. Bin2D never nices, so the pin is the raw range.
    #[test]
    fn fix_extents_pins_bin2d_to_global_2d_extent() {
        use crate::transform::bin_2d::Bin2DSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::core::TransformSpec;
        let batch = extent_pin_batch_2d();
        let spec = Bin2DSpec {
            x: "x".into(),
            y: "y".into(),
            bins_x: BinMode::Fixed { n: 10 },
            bins_y: BinMode::Fixed { n: 10 },
            extent_x: None,
            extent_y: None,
            cumulative: false,
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Bin2D(spec.clone())], &batch);
        let TransformSpec::Bin2D(pinned) = &out[0] else {
            panic!("expected a Bin2D transform back");
        };
        assert_eq!(
            pinned.extent_x,
            Some((0.0, 11.0)),
            "Bin2D extent_x must be the raw global x-range across panels"
        );
        assert_eq!(
            pinned.extent_y,
            Some((0.0, 101.0)),
            "Bin2D extent_y must be the raw global y-range across panels"
        );
    }

    /// A user-provided explicit 2-D extent must never be overridden by the pin.
    /// Covers both Kde2D (4-tuple) and Bin2D (per-axis extent_x/extent_y).
    #[test]
    fn fix_extents_respects_user_2d_extent() {
        use crate::transform::bin_2d::Bin2DSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::core::TransformSpec;
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::kde_2d::Kde2DSpec;
        let batch = extent_pin_batch_2d();

        let user_kde = (-5.0, 50.0, -5.0, 500.0);
        let kde = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: Some(user_kde),
            groupby: vec![],
            name: None,
        };
        // Bin2D with BOTH axes user-set must be left fully untouched.
        let user_bin_x = (-1.0, 20.0);
        let user_bin_y = (-2.0, 200.0);
        let bin = Bin2DSpec {
            x: "x".into(),
            y: "y".into(),
            bins_x: BinMode::Fixed { n: 10 },
            bins_y: BinMode::Fixed { n: 10 },
            extent_x: Some(user_bin_x),
            extent_y: Some(user_bin_y),
            cumulative: false,
            name: None,
        };
        let out = fix_transform_extents_for_facet(
            &[TransformSpec::Kde2D(kde), TransformSpec::Bin2D(bin)],
            &batch,
        );
        let TransformSpec::Kde2D(pinned_kde) = &out[0] else {
            panic!("expected a Kde2D transform back");
        };
        let TransformSpec::Bin2D(pinned_bin) = &out[1] else {
            panic!("expected a Bin2D transform back");
        };
        assert_eq!(pinned_kde.extent, Some(user_kde), "user Kde2D extent must be preserved");
        assert_eq!(pinned_bin.extent_x, Some(user_bin_x), "user Bin2D extent_x must be preserved");
        assert_eq!(pinned_bin.extent_y, Some(user_bin_y), "user Bin2D extent_y must be preserved");
    }

    /// A faceted `Bin2D` with `extent_x` user-set but `extent_y = None` goes through
    /// the dispatch arm's `.or` composition: the user-provided `extent_x` is kept
    /// unchanged while `extent_y` is pinned to the global y-range over the full
    /// pre-facet batch. This guards against a future x/y transposition in the arm
    /// (e.g. accidentally writing `extent_x: spec.extent_y.or(...)`) by asserting
    /// the final pinned values at both axes simultaneously.
    #[test]
    fn fix_extents_bin2d_partial_user_extent_keeps_x_pins_y() {
        use crate::transform::bin_2d::Bin2DSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::core::TransformSpec;
        let batch = extent_pin_batch_2d();

        // User has explicitly set extent_x but left extent_y as None.
        let user_x = (2.0, 15.0);
        let spec = Bin2DSpec {
            x: "x".into(),
            y: "y".into(),
            bins_x: BinMode::Fixed { n: 10 },
            bins_y: BinMode::Fixed { n: 10 },
            extent_x: Some(user_x),
            extent_y: None,
            cumulative: false,
            name: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Bin2D(spec.clone())], &batch);
        let TransformSpec::Bin2D(pinned) = &out[0] else {
            panic!("expected a Bin2D transform back");
        };
        // The user-provided extent_x must survive unchanged.
        assert_eq!(
            pinned.extent_x,
            Some(user_x),
            "user extent_x must be preserved when extent_y is None"
        );
        // extent_y was None; the pin must fill it with the global y-range [0, 101].
        // Raw y values in the fixture: 0.0, 0.5, 1.0, 100.0, 100.5, 101.0.
        assert_eq!(
            pinned.extent_y,
            Some((0.0, 101.0)),
            "extent_y must be pinned to the global y-range when user left it None"
        );
    }

    // --- archaeology #7 round-7 T1: Hex / Raster / DataBin dispatch tests -------

    /// A faceted `Hex` with no extent gets both axes pinned to the RAW global
    /// range over the full pre-facet batch. The two-panel fixture has disjoint
    /// x ∈ [0,1] vs [10,11] and y ∈ [0,1] vs [100,101]; the global raw range is
    /// x∈[0,11], y∈[0,101]. An un-pinned hex lattice would anchor differently per
    /// panel, making the bin edges incomparable across panels.
    #[test]
    fn fix_extents_pins_hex_to_global_2d_extent() {
        use crate::transform::core::TransformSpec;
        use crate::transform::hex::HexSpec;
        let batch = extent_pin_batch_2d();
        let spec = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: None,
            aggregate: "count".into(),
            field: None,
            name: None,
            extent_x: None,
            extent_y: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Hex(spec.clone())], &batch);
        let TransformSpec::Hex(pinned) = &out[0] else {
            panic!("expected a Hex transform back");
        };
        assert_eq!(
            pinned.extent_x,
            Some((0.0, 11.0)),
            "Hex extent_x must be pinned to the raw global x-range across panels"
        );
        assert_eq!(
            pinned.extent_y,
            Some((0.0, 101.0)),
            "Hex extent_y must be pinned to the raw global y-range across panels"
        );
    }

    /// A faceted `Hex` with `extent_x` user-set but `extent_y = None` keeps the
    /// user value on x and pins y to the global raw range. Guards against an
    /// accidental x/y transposition in the dispatch arm.
    #[test]
    fn fix_extents_hex_partial_user_extent_keeps_x_pins_y() {
        use crate::transform::core::TransformSpec;
        use crate::transform::hex::HexSpec;
        let batch = extent_pin_batch_2d();

        let user_x = (2.0, 15.0);
        let spec = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: None,
            aggregate: "count".into(),
            field: None,
            name: None,
            extent_x: Some(user_x),
            extent_y: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Hex(spec.clone())], &batch);
        let TransformSpec::Hex(pinned) = &out[0] else {
            panic!("expected a Hex transform back");
        };
        assert_eq!(
            pinned.extent_x,
            Some(user_x),
            "user extent_x must be preserved when extent_y is None"
        );
        assert_eq!(
            pinned.extent_y,
            Some((0.0, 101.0)),
            "extent_y must be pinned to the global y-range when user left it None"
        );
    }

    /// A faceted `Raster` with no extent gets both axes pinned to the RAW global
    /// range over the full pre-facet batch: x∈[0,11], y∈[0,101].
    #[test]
    fn fix_extents_pins_raster_to_global_2d_extent() {
        use crate::transform::core::TransformSpec;
        use crate::transform::raster::{RasterSpec, ResolutionSpec};
        let batch = extent_pin_batch_2d();
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "count".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(4),
            min_count: None,
            log_scale: false,
            name: None,
            extent_x: None,
            extent_y: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Raster(spec.clone())], &batch);
        let TransformSpec::Raster(pinned) = &out[0] else {
            panic!("expected a Raster transform back");
        };
        assert_eq!(
            pinned.extent_x,
            Some((0.0, 11.0)),
            "Raster extent_x must be pinned to the raw global x-range across panels"
        );
        assert_eq!(
            pinned.extent_y,
            Some((0.0, 101.0)),
            "Raster extent_y must be pinned to the raw global y-range across panels"
        );
    }

    /// A faceted `Raster` with `extent_x` user-set but `extent_y = None` keeps
    /// the user value on x and pins y to the global raw range. Guards against an
    /// accidental x/y transposition in the dispatch arm.
    #[test]
    fn fix_extents_raster_partial_user_extent_keeps_x_pins_y() {
        use crate::transform::core::TransformSpec;
        use crate::transform::raster::{RasterSpec, ResolutionSpec};
        let batch = extent_pin_batch_2d();

        let user_x = (2.0, 15.0);
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "count".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(4),
            min_count: None,
            log_scale: false,
            name: None,
            extent_x: Some(user_x),
            extent_y: None,
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::Raster(spec.clone())], &batch);
        let TransformSpec::Raster(pinned) = &out[0] else {
            panic!("expected a Raster transform back");
        };
        assert_eq!(
            pinned.extent_x,
            Some(user_x),
            "user extent_x must be preserved when extent_y is None"
        );
        assert_eq!(
            pinned.extent_y,
            Some((0.0, 101.0)),
            "extent_y must be pinned to the global y-range when user left it None"
        );
    }

    /// A faceted `DataBin` with `extent = None` gets pinned to the RAW global
    /// `(min, max)` over the full pre-facet batch. The fixture has `v` values
    /// [0.3, 0.5, 1.2, 4.8, 9.7] (raw range 0.3..9.7). Unlike `Bin`, DataBin
    /// does not nice the pinned extent — the raw range drives the bin boundary
    /// computation directly.
    #[test]
    fn fix_extents_pins_data_bin_to_global_raw_extent() {
        use crate::transform::core::TransformSpec;
        use crate::transform::data_bin::DataBinSpec;
        let batch = extent_pin_batch();
        let spec = DataBinSpec {
            field: "v".into(),
            as_: None,
            maxbins: Some(10),
            step: None,
            nice: false,
            name: None,
            extent: None,
        };
        let out =
            fix_transform_extents_for_facet(&[TransformSpec::DataBin(spec.clone())], &batch);
        let TransformSpec::DataBin(pinned) = &out[0] else {
            panic!("expected a DataBin transform back");
        };
        // Raw range over [0.3, 0.5, 1.2, 4.8, 9.7] = (0.3, 9.7). DataBin never
        // nices the pin, so the pinned value is the exact raw (min, max).
        assert_eq!(
            pinned.extent,
            Some((0.3, 9.7)),
            "DataBin extent must be pinned to the raw global (min, max) across panels"
        );
    }

    /// A faceted `DataBin` with a user-provided `extent` must never have it
    /// overridden. The dispatch arm guards on `spec.extent.is_none()`, so this
    /// confirms the guard fires correctly and the user value is left untouched.
    #[test]
    fn fix_extents_data_bin_user_extent_is_preserved() {
        use crate::transform::core::TransformSpec;
        use crate::transform::data_bin::DataBinSpec;
        let batch = extent_pin_batch();

        let user_extent = (0.0, 20.0);
        let spec = DataBinSpec {
            field: "v".into(),
            as_: None,
            maxbins: Some(10),
            step: None,
            nice: false,
            name: None,
            extent: Some(user_extent),
        };
        let out =
            fix_transform_extents_for_facet(&[TransformSpec::DataBin(spec.clone())], &batch);
        let TransformSpec::DataBin(pinned) = &out[0] else {
            panic!("expected a DataBin transform back");
        };
        assert_eq!(
            pinned.extent,
            Some(user_extent),
            "user DataBin extent must never be clobbered by the pin"
        );
    }

    // --- archaeology #7 round-3 T1: DensityData facet extent pin ----------------

    /// A batch with two panels whose value ranges are fully DISJOINT.
    /// Panel "A" has x in [1, 3], panel "B" has x in [10, 30].
    /// Used to drive `fix_transform_extents_for_facet` for DensityData: an
    /// unpinned per-panel KDE would give each panel a different x extent.
    fn density_disjoint_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("panel", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    1.0, 2.0, 3.0,    // panel A
                    10.0, 20.0, 30.0, // panel B
                ])),
                Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
            ],
        )
        .unwrap()
    }

    fn base_density_spec() -> crate::transform::density_data::DensityDataSpec {
        crate::transform::density_data::DensityDataSpec {
            field: "x".into(),
            bandwidth: None,
            groupby: None,
            extent: None,
            steps: None,
            cumulative: false,
            as_: ("value".into(), "density".into()),
            name: None,
        }
    }

    /// A faceted `DensityData` with `extent=None` must be pinned to the RAW
    /// global (min, max) over the full pre-facet batch so every panel shares
    /// the same value axis. This is discriminating: without the T1 fix the
    /// DensityData arm did not exist in `fix_transform_extents_for_facet` and
    /// the spec fell through the `_ => t.clone()` arm with extent still `None`.
    #[test]
    fn fix_extents_pins_density_data_to_global_extent() {
        use crate::transform::core::TransformSpec;
        let batch = density_disjoint_batch();
        let spec = base_density_spec(); // extent: None
        let out = fix_transform_extents_for_facet(&[TransformSpec::DensityData(spec.clone())], &batch);
        let TransformSpec::DensityData(pinned) = &out[0] else {
            panic!("expected a DensityData transform back");
        };
        // Raw global extent over full batch: x ∈ [1.0, 30.0].
        // DensityData does not nice (mirrors KDE), so the pin is the raw (min, max).
        assert_eq!(
            pinned.extent,
            Some((1.0, 30.0)),
            "DensityData extent must be pinned to the raw global (1.0, 30.0)"
        );
        // Orchestration sanity: matches global_extent's output directly.
        assert_eq!(
            pinned.extent,
            crate::transform::density_data::global_extent(&spec, &batch),
            "pinned extent must equal global_extent's own output"
        );
    }

    /// A user-provided explicit `DensityData` `extent` must never be overridden.
    #[test]
    fn fix_extents_density_data_respects_user_extent() {
        use crate::transform::core::TransformSpec;
        let batch = density_disjoint_batch();
        let user = Some((-5.0, 50.0));
        let spec = crate::transform::density_data::DensityDataSpec {
            extent: user,
            ..base_density_spec()
        };
        let out = fix_transform_extents_for_facet(&[TransformSpec::DensityData(spec)], &batch);
        let TransformSpec::DensityData(pinned) = &out[0] else {
            panic!("expected a DensityData transform back");
        };
        assert_eq!(pinned.extent, user, "user DensityData extent must be preserved");
    }

    /// `DensityData::global_extent` returns raw (min, max) — no nicing, mirroring KDE.
    /// Guards the contract that DensityData uses raw extent (not niced like Bin).
    #[test]
    fn density_data_global_extent_is_raw_not_niced() {
        use crate::transform::density_data;
        let batch = density_disjoint_batch();
        let spec = base_density_spec();
        assert_eq!(
            density_data::global_extent(&spec, &batch),
            Some((1.0, 30.0)),
            "DensityData global_extent must return the raw (min, max), not a niced value"
        );
    }

    fn batch3() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
            ],
        )
        .unwrap()
    }

    fn spec_color_facet() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
                resolve: crate::layout::facet::FacetResolve::default(),
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        }
    }

    #[test]
    fn prepare_returns_axes_and_groups_and_legend() {
        let spec = spec_color_facet();
        let batch = batch3();
        let prep = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prep.axes.x.title.as_deref(), Some("x"));
        assert_eq!(prep.axes.y.title.as_deref(), Some("y"));
        assert!(!prep.axes.x.tick_labels.is_empty());
        assert_eq!(prep.facet_groups.len(), 2);
        assert_eq!(prep.facet_groups[0].n_rows, 2);
        assert_eq!(prep.facet_groups[1].n_rows, 1);
        assert_eq!(prep.legend_entries.len(), 2);
        assert_eq!(prep.legend_entries[0].label, "a");
    }

    // ── Multivariate B1: size / shape / merged legends ───────────────────

    fn batch_pop() -> RecordBatch {
        // x, y, a numeric "pop" field (size/color domain ≈ [10, 100]), and a
        // categorical "region" field for color/shape.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("pop", DataType::Float64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("grp", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(Float64Array::from(vec![10.0, 40.0, 70.0, 100.0])),
                Arc::new(StringArray::from(vec!["AS", "EU", "AS", "AF"])),
                Arc::new(StringArray::from(vec!["lo", "hi", "lo", "hi"])),
            ],
        )
        .unwrap()
    }

    fn base_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    fn prep(spec: &ChartSpec, batch: &RecordBatch) -> PreparedInputs {
        prepare_render_inputs(spec, batch, &crate::layout::ThemeInputs::default(), None).unwrap()
    }

    #[test]
    fn size_encoding_produces_graduated_size_legend() {
        let mut spec = base_spec();
        spec.encoding.size = Some(EncodingSpec { field: "pop".into(), type_: None, ..Default::default() });
        let p = prep(&spec, &batch_pop());
        assert_eq!(p.aux_legends.len(), 1, "expected exactly one (size) aux legend");
        let AuxLegendInput::Size { entries, title } = &p.aux_legends[0] else {
            panic!("expected a Size aux legend, got {:?}", p.aux_legends[0]);
        };
        assert_eq!(title.as_deref(), Some("pop"));
        // ~5 nice round values across the [10, 100] domain. nice_ticks(10,100,5)
        // yields 20,40,60,80,100 — labels are the round values, not raw quantiles.
        assert!(
            (4..=6).contains(&entries.len()),
            "expected ~5 graduated entries, got {}: {:?}",
            entries.len(), entries
        );
        // Labels are human-friendly round numbers.
        for e in entries {
            assert!(e.radius > 0.0, "size symbol radius must be positive");
            let v: f64 = e.label.parse().expect("size legend label must be numeric");
            assert_eq!(v.fract(), 0.0, "expected round value label, got {}", e.label);
        }
        // Radii increase with value (larger pop → bigger symbol).
        assert!(
            entries.first().unwrap().radius < entries.last().unwrap().radius,
            "radii should grow across the size domain"
        );
    }

    #[test]
    fn shape_encoding_produces_shape_legend_one_per_category() {
        let mut spec = base_spec();
        spec.encoding.shape = Some(EncodingSpec { field: "region".into(), type_: None, ..Default::default() });
        let p = prep(&spec, &batch_pop());
        assert_eq!(p.aux_legends.len(), 1);
        let AuxLegendInput::Shape { entries, title } = &p.aux_legends[0] else {
            panic!("expected a Shape aux legend");
        };
        assert_eq!(title.as_deref(), Some("region"));
        // Distinct regions in encounter order: AS, EU, AF.
        let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, vec!["AS", "EU", "AF"]);
        // First three palette shapes: circle, square, cross.
        assert_eq!(entries[0].shape_name, "circle");
        assert_eq!(entries[1].shape_name, "square");
        assert_eq!(entries[2].shape_name, "cross");
    }

    #[test]
    fn color_size_shape_together_produce_two_aux_blocks_plus_color() {
        // color on region (categorical → color legend), size on pop, shape on a
        // 4th distinct field. Color legend is separate (legend_entries); size +
        // shape are two stacked aux blocks.
        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec { field: "region".into(), type_: None, ..Default::default() });
        spec.encoding.size = Some(EncodingSpec { field: "pop".into(), type_: None, ..Default::default() });
        spec.encoding.shape = Some(EncodingSpec { field: "grp".into(), type_: None, ..Default::default() });
        let p = prep(&spec, &batch_pop());
        // Color drives a categorical legend (legend_entries), not an aux block.
        assert!(!p.legend_entries.is_empty(), "color legend entries expected");
        // Two aux blocks: size then shape, stable order.
        assert_eq!(p.aux_legends.len(), 2, "expected size + shape aux blocks");
        assert!(matches!(p.aux_legends[0], AuxLegendInput::Size { .. }), "size first");
        assert!(matches!(p.aux_legends[1], AuxLegendInput::Shape { .. }), "shape second");
    }

    #[test]
    fn color_and_size_on_same_field_merge_into_one_block() {
        // color (continuous) AND size both on "pop" → a single combined block:
        // a size legend whose symbols also carry color (color_hex), and NO
        // colorbar (it would be the second, redundant legend).
        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "pop".into(),
            type_: Some(crate::spec::encoding::DataType::Quantitative),
            ..Default::default()
        });
        spec.encoding.size = Some(EncodingSpec { field: "pop".into(), type_: None, ..Default::default() });
        let p = prep(&spec, &batch_pop());
        // Exactly one merged block, and the colorbar is suppressed.
        assert_eq!(p.aux_legends.len(), 1, "merge → single block, not two");
        assert!(p.colorbar.is_none(), "colorbar must be suppressed in the merged case");
        let AuxLegendInput::Size { entries, .. } = &p.aux_legends[0] else {
            panic!("merged block should be a Size legend carrying color");
        };
        assert!(
            entries.iter().all(|e| e.color_hex.is_some()),
            "merged size entries must carry per-entry color"
        );
    }

    #[test]
    fn size_legend_disabled_suppresses_block() {
        let mut spec = base_spec();
        spec.encoding.size = Some(EncodingSpec {
            field: "pop".into(),
            legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
                disabled: Some(true),
                ..Default::default()
            })),
            ..Default::default()
        });
        let p = prep(&spec, &batch_pop());
        assert!(p.aux_legends.is_empty(), "legend=None on size must suppress the block");
    }

    #[test]
    fn color_only_chart_has_no_aux_legends() {
        // Regression: a color-only chart produces zero aux legends (the color
        // legend / colorbar path is untouched).
        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec { field: "region".into(), type_: None, ..Default::default() });
        let p = prep(&spec, &batch_pop());
        assert!(p.aux_legends.is_empty());
        assert!(!p.legend_entries.is_empty(), "color legend still present");
    }

    #[test]
    fn empty_batch_errors() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(Vec::<f64>::new())),
                Arc::new(Float64Array::from(Vec::<f64>::new())),
            ],
        )
        .unwrap();
        let mut spec = spec_color_facet();
        spec.encoding.color = None;
        spec.facet = None;
        let err = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap_err();
        assert!(matches!(err, RenderError::EmptyBatch));
    }

    // --- Phase 8a Task 6 tests ---

    /// Helper: simple 2-column float batch with named fields.
    fn price_weight_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Float64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    /// Helper: single-layer spec with x="price", y="weight".
    fn single_layer_spec() -> ChartSpec {
        ChartSpec {
            data: crate::spec::data_ref::DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "weight".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        }
    }

    #[test]
    fn prepare_single_layer_produces_one_layer_prepared() {
        let spec = single_layer_spec();
        let batch = price_weight_batch();
        let prepared = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prepared.layers.len(), 1);
        assert_eq!(prepared.layers[0].mark, Mark::Point);
        assert!(!prepared.coord_flipped);
        // Encoding fields should match spec
        assert_eq!(prepared.layers[0].encoding.x.as_ref().unwrap().field, "price");
        assert_eq!(prepared.layers[0].encoding.y.as_ref().unwrap().field, "weight");
    }

    #[test]
    fn prepare_multi_layer_produces_multiple_layer_prepared() {
        use crate::spec::layer::Layer;
        let mut spec = single_layer_spec();
        // Two layers: point on price/weight, line inheriting chart encoding
        spec.layers = Some(vec![
            Layer {
                mark: Mark::Point,
                encoding: Encoding {
                    x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
                    y: Some(EncodingSpec { field: "weight".into(), type_: None, ..Default::default() }),
                    ..Default::default()
                },
                transforms: vec![],
                mark_style: None,
                data_source: None,
                position: None,
                blend: None,
                name: None,
                independent_y: false,
            },
            Layer {
                mark: Mark::Line,
                encoding: Encoding::default(), // inherits from chart-level
                transforms: vec![],
                mark_style: None,
                data_source: None,
            position: None, blend: None, name: None, independent_y: false,
            },
        ]);
        let batch = price_weight_batch();
        let prepared = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prepared.layers.len(), 2);
        assert_eq!(prepared.layers[0].mark, Mark::Point);
        assert_eq!(prepared.layers[1].mark, Mark::Line);
        // Layer 2 inherits chart-level encoding
        assert_eq!(prepared.layers[1].encoding.x.as_ref().unwrap().field, "price");
        assert_eq!(prepared.layers[1].encoding.y.as_ref().unwrap().field, "weight");
    }

    // --- Phase 8b Task 9: named-output transform routing ---

    /// Build a ChartSpec with one Bin transform whose `name` is configurable,
    /// and `bin_count` such that the transform succeeds on price_weight_batch().
    ///
    /// When `name` is None, the bin transform is unnamed → it chains, so
    /// `__final__` has bin output schema and the encoding is pointed at bin
    /// columns. When `name` is Some, fan-out semantics apply: `__final__`
    /// retains the original schema, so the encoding stays on the original
    /// columns to keep `resolve_scales` happy.
    fn spec_with_one_bin(name: Option<String>) -> ChartSpec {
        use crate::transform::bin::BinSpec;
        use crate::transform::bin_mode::BinMode;
        use crate::transform::core::TransformSpec;
        let named = name.is_some();
        let mut spec = single_layer_spec();
        spec.transforms = vec![TransformSpec::Bin(BinSpec {
            field: "price".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((10.0, 30.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            name,
        })];
        if !named {
            // Unnamed/chained: after Bin, the encoding fields ("price", "weight")
            // no longer exist in __final__ — point at bin output columns so
            // resolve_scales doesn't fail.
            spec.encoding.x = Some(crate::spec::encoding::EncodingSpec {
                field: "bin_start".into(),
                type_: None,
                ..Default::default()
            });
            spec.encoding.y = Some(crate::spec::encoding::EncodingSpec {
                field: "count".into(),
                type_: None,
                ..Default::default()
            });
        }
        // Named/fan-out: __final__ keeps the original price/weight schema, so
        // the chart-level encoding (price, weight) from `single_layer_spec()`
        // still resolves against __final__ correctly.
        spec
    }

    #[test]
    fn data_source_none_uses_final_pipeline_output() {
        pyo3::Python::initialize();
        let spec = spec_with_one_bin(None);
        let batch = price_weight_batch();
        let prep = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        // __final__ is always present.
        assert!(
            prep.transform_outputs.contains_key("__final__"),
            "transform_outputs must always publish __final__"
        );
        // Bin had no name → its output is NOT separately keyed.
        assert_eq!(
            prep.transform_outputs.len(),
            1,
            "expected only __final__, got keys: {:?}",
            prep.transform_outputs.keys().collect::<Vec<_>>()
        );
        // final_batch() returns the FINAL_OUTPUT_KEY entry.
        let final_batch = prep.transform_outputs.get("__final__").unwrap();
        assert_eq!(prep.final_batch().num_rows(), final_batch.num_rows());
        assert_eq!(prep.final_batch().num_columns(), final_batch.num_columns());
        assert_eq!(
            prep.final_batch().schema(),
            final_batch.schema(),
            "final_batch() and __final__ schemas must match"
        );
    }

    #[test]
    fn data_source_some_publishes_named_transform_output() {
        pyo3::Python::initialize();
        let spec = spec_with_one_bin(Some("box".into()));
        let batch = price_weight_batch();
        let prep = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert!(prep.transform_outputs.contains_key("box"));
        assert!(prep.transform_outputs.contains_key("__final__"));
        // Under fan-out semantics, named transforms run on the ORIGINAL input
        // and do NOT advance the chained pipeline. The named "box" output is
        // the bin output; __final__ is the original input (since no unnamed
        // transforms advanced the chain).
        let named = prep.transform_outputs.get("box").unwrap();
        let fin = prep.transform_outputs.get("__final__").unwrap();
        // The named bin output has the bin schema (bin_start/bin_end/count/density).
        let named_schema = named.schema();
        let named_fields: Vec<&str> = named_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(
            named_fields.contains(&"bin_start") && named_fields.contains(&"count"),
            "named output should have bin schema, got: {:?}",
            named_fields
        );
        // __final__ retains the original schema (price + weight).
        let final_schema = fin.schema();
        let final_fields: Vec<&str> = final_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(
            final_fields.contains(&"price") && final_fields.contains(&"weight"),
            "__final__ should have original schema, got: {:?}",
            final_fields
        );
        // And — proving the change — the named output and __final__ schemas differ.
        assert_ne!(named.schema(), fin.schema());
    }

    #[test]
    fn unknown_data_source_raises_clear_error() {
        pyo3::Python::initialize();
        use crate::spec::layer::Layer;
        // Pipeline publishes "step1"; layer asks for "missing".
        let mut spec = spec_with_one_bin(Some("step1".into()));
        spec.layers = Some(vec![Layer {
            mark: Mark::Point,
            encoding: Encoding::default(),
            transforms: vec![],
            mark_style: None,
            data_source: Some("missing".into()),
            position: None, blend: None, name: None, independent_y: false,
        }]);
        let batch = price_weight_batch();
        let err = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("missing"), "error must name the bogus key: {msg}");
        // Available keys list must mention either the named transform or the sentinel.
        assert!(
            msg.contains("step1") || msg.contains("__final__"),
            "error must list available keys: {msg}"
        );
    }

    #[test]
    fn prepare_coord_flip_swaps_x_y_in_each_layer() {
        use crate::spec::coord::CoordKind;
        let mut spec = single_layer_spec(); // x="price", y="weight"
        spec.coord = Some(CoordKind::Flip);
        let batch = price_weight_batch();
        let prepared = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert!(prepared.coord_flipped);
        // After flip: x should have "weight", y should have "price"
        assert_eq!(
            prepared.layers[0].encoding.x.as_ref().unwrap().field,
            "weight",
            "CoordFlip should swap x←weight (was y)"
        );
        assert_eq!(
            prepared.layers[0].encoding.y.as_ref().unwrap().field,
            "price",
            "CoordFlip should swap y←price (was x)"
        );
        // Axes titles should also reflect the flip
        assert_eq!(prepared.axes.x.title.as_deref(), Some("weight"));
        assert_eq!(prepared.axes.y.title.as_deref(), Some("price"));
    }

    // --- D3 (flexibility campaign): per-channel Axis(label_format, tick_count) ---

    /// Build an `EncodingSpec` carrying a per-channel `axis` from a JSON object,
    /// deserialized through the typed `AxisStyleSpec` (so the test exercises the
    /// real wire path, including `deny_unknown_fields`).
    fn enc_with_axis(field: &str, axis: serde_json::Value) -> EncodingSpec {
        let axis: crate::render::chart_config::AxisStyleSpec =
            serde_json::from_value(axis).expect("valid axis style json");
        EncodingSpec {
            field: field.into(),
            axis: Some(Box::new(axis)),
            ..Default::default()
        }
    }

    #[test]
    fn per_channel_label_format_reaches_override() {
        // A per-channel `Axis(label_format=",.0f")` on x must reach the axis
        // `label_format_override` (root-cause fix for the hardcoded `None`).
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis(
            "price",
            serde_json::json!({ "label_format": ",.0f" }),
        ));
        let batch = price_weight_batch();
        let prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prep.axes.x.overrides.label_format.as_deref(), Some(",.0f"));
    }

    #[test]
    fn per_channel_axis_styling_reaches_axis_input() {
        // B5: a per-channel `Axis(grid_color=, label_color=, domain_width=)` must
        // land on the AxisInput's per-axis style override fields (which
        // build_axis/build_grid consult with a theme fallback).
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis(
            "price",
            serde_json::json!({ "grid_color": "#cccccc", "label_color": "#ff00ff", "domain_width": 3.0 }),
        ));
        let batch = price_weight_batch();
        let prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        let gx = prep.axes.x.overrides.grid_color.expect("per-channel grid_color must reach x AxisInput");
        assert_eq!([gx.red, gx.green, gx.blue], [0xcc, 0xcc, 0xcc]);
        let lx = prep.axes.x.overrides.label_color.expect("per-channel label_color must reach x AxisInput");
        assert_eq!([lx.red, lx.green, lx.blue], [0xff, 0x00, 0xff]);
        assert_eq!(prep.axes.x.overrides.domain_width, Some(3.0));
        // The y-axis must be untouched (per-channel applies only to its own axis).
        assert!(prep.axes.y.overrides.grid_color.is_none());
        assert!(prep.axes.y.overrides.label_color.is_none());
    }

    // ── B5 unit 2: orphan positioning / tick fields ─────────────────────────

    #[test]
    fn per_channel_orient_reaches_axis_input() {
        // `Axis(orient="top")` on x must set the x AxisInput's orient to Top.
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis("price", serde_json::json!({ "orient": "top" })));
        let batch = price_weight_batch();
        let prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prep.axes.x.orient, AxisOrient::Top);
        // y untouched (default Left).
        assert_eq!(prep.axes.y.orient, AxisOrient::Left);
    }

    #[test]
    fn cross_dimension_orient_fails_loud() {
        // `Axis(orient="left")` on the x channel is a cross-dimension error and
        // must surface a RenderError rather than silently dropping.
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis("price", serde_json::json!({ "orient": "left" })));
        let batch = price_weight_batch();
        let err = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None)
            .expect_err("x orient='left' must fail loud");
        match err {
            crate::render::RenderError::InvalidAxisOrient { channel, orient } => {
                assert_eq!(channel, "x");
                assert_eq!(orient, "left");
            }
            other => panic!("expected InvalidAxisOrient, got {other:?}"),
        }
    }

    #[test]
    fn per_channel_orphan_positioning_fields_reach_axis_input() {
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis(
            "price",
            serde_json::json!({
                "translate": 12.0,
                "min_band": 70.0,
                "max_band": 120.0,
                "grid_opacity": 0.25,
                "title_orient": "bottom",
                "zindex": 1
            }),
        ));
        let batch = price_weight_batch();
        let prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prep.axes.x.overrides.translate, Some(12.0));
        assert_eq!(prep.axes.x.overrides.min_band, Some(70.0));
        assert_eq!(prep.axes.x.overrides.max_band, Some(120.0));
        assert_eq!(prep.axes.x.overrides.grid_opacity, Some(0.25));
        assert_eq!(prep.axes.x.overrides.title_orient, Some(AxisOrient::Bottom));
        assert_eq!(prep.axes.x.overrides.zindex, Some(1));
    }

    #[test]
    fn tick_min_step_thins_dense_ticks() {
        // A continuous x-axis with tick_min_step set must drop ticks closer than
        // the step in data space, leaving fewer labels.
        let mut spec = single_layer_spec();
        let batch = price_weight_batch();
        let baseline =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        let base_count = baseline.axes.x.tick_labels.len();

        // Pick a min_step larger than the natural tick spacing to force thinning.
        spec.encoding.x = Some(enc_with_axis("price", serde_json::json!({ "tick_min_step": 1e9 })));
        let mut prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        // The adjustment runs in prepare_and_layout; invoke it directly here.
        let tc = prep.x_tick_count;
        adjust_axis_ticks(&mut prep.axes.x, &prep.provisional_scales.x, tc, false);
        assert!(
            prep.axes.x.tick_labels.len() < base_count,
            "tick_min_step=1e9 must thin ticks: base={base_count}, after={}",
            prep.axes.x.tick_labels.len()
        );
        // At least one tick survives.
        assert!(!prep.axes.x.tick_labels.is_empty());
        // Labels and projected fractions stay index-aligned.
        if let Some(proj) = prep.axes.x.tick_projection.as_ref() {
            assert_eq!(proj.major.len(), prep.axes.x.tick_labels.len());
        }
    }

    #[test]
    fn tick_extra_appends_domain_boundaries() {
        // tick_extra must add a tick at each domain boundary not already present.
        // Use awkward bounds (3.0 .. 97.0) so the scale's nice ticks (0,20,40,…)
        // do NOT already include the exact boundaries, forcing an actual append.
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Float64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![3.0, 50.0, 97.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis("price", serde_json::json!({ "tick_extra": true })));
        let mut prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        let before = prep.axes.x.tick_labels.len();
        let tc = prep.x_tick_count;
        adjust_axis_ticks(&mut prep.axes.x, &prep.provisional_scales.x, tc, false);
        let after = prep.axes.x.tick_labels.len();
        assert!(after > before, "tick_extra must append boundary ticks: {before} -> {after}");
        // Labels and projected fractions stay index-aligned.
        let proj = prep.axes.x.tick_projection.as_ref().unwrap();
        assert_eq!(proj.major.len(), prep.axes.x.tick_labels.len(), "labels/fractions aligned");
        // The exact domain-boundary labels ("3" and "97") are present.
        assert!(prep.axes.x.tick_labels.iter().any(|l| l == "3"), "domain min tick present");
        assert!(prep.axes.x.tick_labels.iter().any(|l| l == "97"), "domain max tick present");
    }

    #[test]
    fn per_channel_axis_style_wins_over_chart_config() {
        // B5 cascade: a per-channel `Axis(grid_width=...)` on x, applied in
        // `prepare_render_inputs`, is set BEFORE the chart-level
        // `configure_axis(grid_width=...)` fill (which only fills `None`), so the
        // per-channel value wins. We assert the prep-stage value here, then verify
        // the chart-level apply does not overwrite it.
        let mut spec = single_layer_spec();
        spec.encoding.x =
            Some(enc_with_axis("price", serde_json::json!({ "grid_width": 4.0 })));
        let batch = price_weight_batch();
        let mut prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        assert_eq!(prep.axes.x.overrides.grid_width, Some(4.0), "per-channel value set in prep");
        // Chart-level configure_axis(grid_width=1.0) must NOT overwrite per-channel.
        let cfg = crate::render::chart_config::AxisConfigSpec {
            style: crate::render::chart_config::AxisStyleSpec {
                grid_width: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        crate::render::apply_axis_style_to_axis_input(&mut prep.axes.x, &cfg.style).unwrap();
        assert_eq!(prep.axes.x.overrides.grid_width, Some(4.0), "per-channel must win over configure_axis");
    }

    #[test]
    fn per_channel_show_toggle_wins_over_chart_config() {
        // B5 Unit-1 fix: a per-channel `Axis(grid=False)`/`Axis(domain=False)` on x
        // must survive a conflicting chart-level `configure_axis(grid=True)` /
        // `configure_axis(domain=True)`. The per-channel prepare path is the sole
        // owner of `AxisInput.show_*`; the chart-level toggle flows through its
        // global theme/gate path, not by clobbering the per-axis `show_*` value.
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis(
            "price",
            serde_json::json!({ "grid": false, "domain": false }),
        ));
        let batch = price_weight_batch();
        let mut prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        // Per-channel suppression is set in prep; y is untouched (defaults true).
        assert!(!prep.axes.x.show_grid, "per-channel grid=False set in prep");
        assert!(!prep.axes.x.show_domain, "per-channel domain=False set in prep");
        assert!(prep.axes.y.show_grid, "y grid default true");
        assert!(prep.axes.y.show_domain, "y domain default true");

        // Chart-level configure_axis(grid=True, domain=True) applies to BOTH axes
        // via the shared `axis` key, the same wire step `prepare_and_layout` runs.
        let cfg = crate::render::chart_config::AxisConfigSpec {
            style: crate::render::chart_config::AxisStyleSpec {
                grid: Some(true),
                domain: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        crate::render::apply_axis_config_to_axis_input(&mut prep.axes.x, Some(&cfg)).unwrap();
        crate::render::apply_axis_config_to_axis_input(&mut prep.axes.y, Some(&cfg)).unwrap();

        // Per-channel x suppression survives the conflicting chart-level toggle.
        assert!(
            !prep.axes.x.show_grid,
            "per-channel grid=False must survive configure_axis(grid=True)"
        );
        assert!(
            !prep.axes.x.show_domain,
            "per-channel domain=False must survive configure_axis(domain=True)"
        );
        // The other axis (no per-channel override) is unaffected: still shown.
        assert!(prep.axes.y.show_grid, "y axis show_grid unaffected");
        assert!(prep.axes.y.show_domain, "y axis show_domain unaffected");
    }

    #[test]
    fn per_channel_label_format_formats_after_central_apply() {
        // End-to-end through render::mod's central application: the numeric spec
        // must actually reformat the labels. We replicate the override-apply step
        // that `render_pipeline` performs.
        let mut spec = single_layer_spec();
        spec.encoding.x = Some(enc_with_axis(
            "price",
            serde_json::json!({ "label_format": "$,.2f" }),
        ));
        let batch = price_weight_batch();
        let mut prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();
        // Apply the threaded override exactly as render/mod.rs does.
        prep.axes.x.tick_labels = apply_tick_format(
            std::mem::take(&mut prep.axes.x.tick_labels),
            prep.axes.x.overrides.label_format.as_deref(),
            None,
        );
        // Every numeric label should now carry the "$" prefix and 2 decimals.
        assert!(
            prep.axes.x.tick_labels.iter().all(|l| l.starts_with('$') && l.contains('.')),
            "got: {:?}",
            prep.axes.x.tick_labels
        );
    }

    #[test]
    fn per_channel_tick_count_limits_numeric_ticks() {
        // Default tick generation yields several ticks over [10, 30]; with
        // `tick_count=2` the axis should produce noticeably fewer labels.
        let batch = price_weight_batch();

        let mut default_spec = single_layer_spec();
        default_spec.encoding.x =
            Some(EncodingSpec { field: "price".into(), ..Default::default() });
        let default_prep =
            prepare_render_inputs(&default_spec, &batch, &crate::layout::ThemeInputs::default(), None)
                .unwrap();

        let mut limited_spec = single_layer_spec();
        limited_spec.encoding.x =
            Some(enc_with_axis("price", serde_json::json!({ "tick_count": 2 })));
        let limited_prep =
            prepare_render_inputs(&limited_spec, &batch, &crate::layout::ThemeInputs::default(), None)
                .unwrap();

        assert!(
            limited_prep.axes.x.tick_labels.len() < default_prep.axes.x.tick_labels.len(),
            "tick_count=2 should produce fewer labels than default; limited={:?} default={:?}",
            limited_prep.axes.x.tick_labels,
            default_prep.axes.x.tick_labels
        );
    }

    // --- F8: colorbar tick formatting via full d3 grammar ---

    /// Build a spec with a continuous color encoding on "pop" so that the colorbar
    /// code path activates. Returns a `ColorbarInput` with `format_spec` applied.
    fn colorbar_with_format(format_spec: &str) -> crate::layout::ColorbarInput {
        use crate::spec::encoding::DataType;
        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "pop".into(),
            type_: Some(DataType::Quantitative),
            legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
                format: Some(format_spec.to_owned()),
                ..Default::default()
            })),
            ..Default::default()
        });
        let p = prep(&spec, &batch_pop());
        p.colorbar.expect("continuous color encoding must produce a colorbar")
    }

    /// (a) Byte-stability: `.2f` colorbar output must match today's expected strings.
    ///
    /// Domain [10, 100]; 5 ticks at t=0,0.25,0.5,0.75,1.0 → v=10,32.5,55,77.5,100.
    /// `.2f` → fixed 2 decimal places.
    #[test]
    fn colorbar_format_dot2f_byte_stable() {
        let colorbar = colorbar_with_format(".2f");
        assert_eq!(
            colorbar.tick_labels,
            vec!["10.00", "32.50", "55.00", "77.50", "100.00"],
            "`.2f` colorbar tick labels must match fixed-2-decimal output"
        );
    }

    /// (a) Byte-stability: `.0%` colorbar output must match today's expected strings.
    ///
    /// Domain [0.0, 1.0] (use batch_pop() but override to a [0,1] range by using
    /// a spec whose pop values happen to span [10,100]; for a cleaner [0,1] domain
    /// we build a dedicated batch).
    #[test]
    fn colorbar_format_dot0pct_byte_stable() {
        use crate::spec::encoding::DataType;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDt, Field, Schema};

        // Build a batch with "pop" in [0.0, 1.0] so the colorbar domain is [0,1].
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDt::Float64, false),
            Field::new("y", ArrowDt::Float64, false),
            Field::new("pop", ArrowDt::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
            ],
        )
        .unwrap();

        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "pop".into(),
            type_: Some(DataType::Quantitative),
            legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
                format: Some(".0%".to_owned()),
                ..Default::default()
            })),
            ..Default::default()
        });
        let p = prep(&spec, &batch);
        let colorbar = p.colorbar.expect("continuous color encoding must produce a colorbar");

        // 5 ticks at t=0,0.25,0.5,0.75,1.0 → v=0,0.25,0.5,0.75,1.0
        // `.0%` → multiply by 100, 0 decimals, append `%` → "0%","25%","50%","75%","100%"
        assert_eq!(
            colorbar.tick_labels,
            vec!["0%", "25%", "50%", "75%", "100%"],
            "`.0%` colorbar tick labels must match percent-0-decimal output"
        );
    }

    /// (b) Capability widening: `,` (grouped thousands) now formats via the full
    ///     d3 grammar rather than falling through to the auto-precision path.
    ///
    /// Domain [10, 100] from batch_pop(); 5 ticks → 10, 32.5, 55, 77.5, 100.
    /// With `,` (no type) these are integer-valued or near-integer; the d3 grammar
    /// groups integer-valued inputs as plain integers: "10", "33", "55", "78", "100"
    /// (or with grouping applied, which only triggers at ≥4 digits). What matters is
    /// that the output does NOT look like the old fallback (auto-precision float).
    #[test]
    fn colorbar_format_comma_grouped_widened() {
        // Domain [1000, 100000] so we can see grouping separators.
        use crate::spec::encoding::DataType;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDt, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDt::Float64, false),
            Field::new("y", ArrowDt::Float64, false),
            Field::new("pop", ArrowDt::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1000.0, 100_000.0])),
            ],
        )
        .unwrap();

        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "pop".into(),
            type_: Some(DataType::Quantitative),
            legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
                format: Some(",.0f".to_owned()),
                ..Default::default()
            })),
            ..Default::default()
        });
        let p = prep(&spec, &batch);
        let colorbar = p.colorbar.expect("continuous color encoding must produce a colorbar");

        // 5 ticks at t=0..1 over [1000, 100000]:
        // v = 1000, 25750, 50500, 75250, 100000
        // `,.0f` → integer with grouping: "1,000" "25,750" "50,500" "75,250" "100,000"
        assert!(
            colorbar.tick_labels.iter().any(|l| l.contains(',')),
            "`,` grouped format must produce commas in large-number colorbar ticks; got {:?}",
            colorbar.tick_labels
        );
        // None of the ticks should contain a decimal point (`.0f` rounds to integer).
        assert!(
            colorbar.tick_labels.iter().all(|l| !l.contains('.')),
            "`.0f` format must not produce decimal points; got {:?}",
            colorbar.tick_labels
        );
    }

    /// (b) Capability widening: `~s` (SI with trim) formats via the full grammar.
    ///
    /// Domain [1000, 1_000_000]; first tick (1000) → "1k", last (1M) → "1M".
    #[test]
    fn colorbar_format_si_trim_widened() {
        use crate::spec::encoding::DataType;
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType as ArrowDt, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDt::Float64, false),
            Field::new("y", ArrowDt::Float64, false),
            Field::new("pop", ArrowDt::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1_000.0, 1_000_000.0])),
            ],
        )
        .unwrap();

        let mut spec = base_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "pop".into(),
            type_: Some(DataType::Quantitative),
            legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec {
                format: Some("~s".to_owned()),
                ..Default::default()
            })),
            ..Default::default()
        });
        let p = prep(&spec, &batch);
        let colorbar = p.colorbar.expect("continuous color encoding must produce a colorbar");

        // First tick (v=1000) → "1k", last tick (v=1_000_000) → "1M".
        assert_eq!(
            colorbar.tick_labels.first().map(String::as_str),
            Some("1k"),
            "`~s` must format 1000 as '1k'; got {:?}",
            colorbar.tick_labels
        );
        assert_eq!(
            colorbar.tick_labels.last().map(String::as_str),
            Some("1M"),
            "`~s` must format 1_000_000 as '1M'; got {:?}",
            colorbar.tick_labels
        );
    }

    // ── #9 [FA-15]: conditional-color legend regression ──────────────────────

    /// Build a batch with an `x`, `y`, and `cat` (categorical) column.
    fn cat_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("cat", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "c"])),
            ],
        )
        .unwrap()
    }

    /// A spec with `color = None` and a `Color` conditional using Field{name="cat"} in
    /// `if_selected` must produce non-empty `legend_entries` whose labels match the
    /// field's distinct values in first-appearance order ("a", "b", "c").
    #[test]
    fn conditional_color_field_builds_legend_entries() {
        use ferrum_scene::{ChannelName, ConditionalEncoding, EncodingValue};

        let mut spec = base_spec();
        // No base color encoding — this is the conditional-only case.
        spec.encoding.color = None;
        spec.conditionals = vec![ConditionalEncoding {
            selection_name: "sel".into(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Field { name: "cat".into() },
            if_not: EncodingValue::Color {
                value: ferrum_scene::Color { r: 200, g: 200, b: 200, a: 255 },
            },
        }];

        let batch = cat_batch();
        let prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default(), None).unwrap();

        assert!(
            !prep.legend_entries.is_empty(),
            "conditional Color Field must produce legend entries when base color is absent"
        );
        let labels: Vec<&str> = prep.legend_entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["a", "b", "c"],
            "legend entries must match distinct values in first-appearance order"
        );
        // All symbols should be Circle to match the categorical convention.
        for entry in &prep.legend_entries {
            assert_eq!(
                entry.symbol,
                SymbolKind::Circle,
                "conditional-color legend entries must use SymbolKind::Circle"
            );
        }
        // Legend title must be the conditional field name.
        assert_eq!(
            prep.legend_title.as_deref(),
            Some("cat"),
            "legend title must be the conditional color field name"
        );
    }

    /// A chart WITH a base color encoding must be byte-identical to the pre-fix
    /// behavior — the new path must not activate when `provisional_scales.color` is Some.
    #[test]
    fn base_color_chart_legend_unchanged_by_conditional_fix() {
        // Use the existing `base_spec` + `batch_pop` helpers from the aux-legend tests.
        // Those live in the same mod so they are accessible here.
        let mut spec = base_spec();
        // A categorical base color encoding drives `provisional_scales.color = Some(Categorical)`.
        spec.encoding.color = Some(EncodingSpec {
            field: "region".into(),
            type_: None,
            ..Default::default()
        });
        // Also add a conditional — must NOT affect the base-color legend.
        use ferrum_scene::{ChannelName, ConditionalEncoding, EncodingValue};
        spec.conditionals = vec![ConditionalEncoding {
            selection_name: "sel".into(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Field { name: "region".into() },
            if_not: EncodingValue::Color {
                value: ferrum_scene::Color { r: 200, g: 200, b: 200, a: 255 },
            },
        }];

        let p = prep(&spec, &batch_pop());
        // The categorical arm (not the conditional arm) should have populated legend_entries.
        assert!(
            !p.legend_entries.is_empty(),
            "base-color chart must still produce legend entries"
        );
        // Entries must equal the domain of the base color field, not vary with the conditional.
        // `batch_pop` has "Asia", "Europe", "Americas" in first-appearance order.
        let labels: Vec<&str> = p.legend_entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["AS", "EU", "AF"],
            "base-color chart legend entries must not be affected by a conditional"
        );
    }

    // ── SPINE-03: build_axis_tick_inputs characterization ────────────────────
    //
    // These pin that the single `build_axis_tick_inputs` helper reproduces the
    // OLD open-coded independent-axis sequence (formerly in scene_build.rs:
    // tick_labels → non-ordinal-y reverse → apply_tick_format → projection with
    // minor: Vec::new()) byte-for-byte. The oracle below is that exact old
    // sequence, inlined here so any drift trips the test.

    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;
    use crate::scale::ordinal::OrdinalScale;

    fn linear_scale(lo: f64, hi: f64) -> ScaleKind {
        ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi],
            vec![0.0, 1.0],
            false,
            false,
        ))
    }

    /// The old independent-X open-coded sequence (no reverse), used as an oracle.
    fn old_independent_x(
        scale: &ScaleKind,
        count: usize,
        fmt: Option<&str>,
        fmt_type: Option<&str>,
    ) -> (Vec<String>, Option<TickProjection>) {
        let labels = apply_tick_format(scale.tick_labels(count), fmt, fmt_type);
        let proj = if matches!(scale, ScaleKind::Ordinal(_)) {
            None
        } else {
            let fractions = scale.tick_fractions(count);
            (!fractions.is_empty()).then(|| TickProjection {
                padding_frac: scale.padding_fraction(),
                major: fractions,
                minor: Vec::new(),
            })
        };
        (labels, proj)
    }

    /// The old independent-Y open-coded sequence (non-ordinal reverse), as oracle.
    fn old_independent_y(
        scale: &ScaleKind,
        count: usize,
        fmt: Option<&str>,
        fmt_type: Option<&str>,
    ) -> (Vec<String>, Option<TickProjection>) {
        let mut raw = scale.tick_labels(count);
        if !matches!(scale, ScaleKind::Ordinal(_)) {
            raw.reverse();
        }
        let labels = apply_tick_format(raw, fmt, fmt_type);
        let proj = if matches!(scale, ScaleKind::Ordinal(_)) {
            None
        } else {
            let mut fractions = scale.tick_fractions(count);
            (!fractions.is_empty()).then(|| {
                fractions.reverse();
                TickProjection {
                    padding_frac: scale.padding_fraction(),
                    major: fractions,
                    minor: Vec::new(),
                }
            })
        };
        (labels, proj)
    }

    /// X (no reverse), no format: build_axis_tick_inputs(Immediate) == oracle.
    #[test]
    fn build_axis_tick_inputs_x_matches_old_independent_no_format() {
        let scale = linear_scale(0.0, 100.0);
        let (labels, proj, threaded) = build_axis_tick_inputs(
            &scale,
            10,
            TickFormatMode::Immediate { format: None, format_type: None },
            false,
            Vec::new(),
        );
        let (o_labels, o_proj) = old_independent_x(&scale, 10, None, None);
        assert_eq!(labels, o_labels);
        assert_eq!(proj, o_proj);
        assert_eq!(threaded, None, "Immediate mode threads nothing");
    }

    /// X with a d3 numeric format applied immediately.
    #[test]
    fn build_axis_tick_inputs_x_matches_old_independent_with_format() {
        let scale = linear_scale(0.0, 1.0);
        let (labels, proj, _t) = build_axis_tick_inputs(
            &scale,
            5,
            TickFormatMode::Immediate { format: Some(".0%"), format_type: None },
            false,
            Vec::new(),
        );
        let (o_labels, o_proj) = old_independent_x(&scale, 5, Some(".0%"), None);
        assert_eq!(labels, o_labels);
        assert_eq!(proj, o_proj);
    }

    /// Y (non-ordinal reverse) matches the old reversed-label / reversed-fraction
    /// sequence in lockstep.
    #[test]
    fn build_axis_tick_inputs_y_matches_old_independent_reversed() {
        let scale = linear_scale(-50.0, 50.0);
        let (labels, proj, _t) = build_axis_tick_inputs(
            &scale,
            8,
            TickFormatMode::Immediate { format: None, format_type: None },
            true,
            Vec::new(),
        );
        let (o_labels, o_proj) = old_independent_y(&scale, 8, None, None);
        assert_eq!(labels, o_labels);
        assert_eq!(proj, o_proj);
        // Sanity: the y projection's fractions are descending (reversed).
        let major = &proj.unwrap().major;
        assert!(
            major.windows(2).all(|w| w[0] >= w[1]),
            "reversed y fractions must be non-increasing"
        );
    }

    /// Ordinal scale: no projection on either path, no y-reverse of labels.
    #[test]
    fn build_axis_tick_inputs_ordinal_has_no_projection() {
        let scale = ScaleKind::Ordinal(OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into()],
            vec![0.0, 30.0],
            0.1,
        ));
        let (x_labels, x_proj, _t) = build_axis_tick_inputs(
            &scale,
            10,
            TickFormatMode::Immediate { format: None, format_type: None },
            false,
            Vec::new(),
        );
        assert!(x_proj.is_none(), "ordinal scale yields no tick projection");
        let (y_labels, y_proj, _t) = build_axis_tick_inputs(
            &scale,
            10,
            TickFormatMode::Immediate { format: None, format_type: None },
            true,
            Vec::new(),
        );
        assert!(y_proj.is_none());
        // Ordinal labels are NOT reversed for y (top-down convention preserved).
        assert_eq!(x_labels, y_labels);
    }

    /// MOD-09 seam: the shared (`Thread`) and independent (`Immediate`) paths run
    /// the SAME `build_axis_tick_inputs` and agree on labels + major fractions for
    /// the same scale, while the independent path keeps `minor` EMPTY even when the
    /// shared path is handed a non-empty minor vec. This pins the byte-identity
    /// invariant the independent-facet goldens rely on: routing both axis paths
    /// through one helper must not start emitting per-panel minor ticks.
    #[test]
    fn build_axis_tick_inputs_independent_keeps_minor_empty_while_shared_carries_it() {
        let scale = linear_scale(0.0, 100.0);

        // Shared/global path: numeric format threads forward (not applied here),
        // and a non-empty minor vec is carried into the projection.
        let shared_minor = vec![0.1_f64, 0.3, 0.7];
        let (shared_labels, shared_proj, shared_threaded) = build_axis_tick_inputs(
            &scale,
            10,
            TickFormatMode::Thread { format: None, format_type: None },
            false,
            shared_minor.clone(),
        );

        // Independent/per-panel path: format applied immediately, minor empty.
        let (indep_labels, indep_proj, indep_threaded) = build_axis_tick_inputs(
            &scale,
            10,
            TickFormatMode::Immediate { format: None, format_type: None },
            false,
            Vec::new(),
        );

        // Same scale, same count, no format on either → identical labels + majors.
        assert_eq!(shared_labels, indep_labels);
        let shared_proj = shared_proj.expect("continuous scale yields a projection");
        let indep_proj = indep_proj.expect("continuous scale yields a projection");
        assert_eq!(shared_proj.major, indep_proj.major, "major fractions must agree");
        assert_eq!(shared_proj.padding_frac, indep_proj.padding_frac);

        // The minor vec is the only intentional divergence: shared carries the
        // caller's minor ticks; independent stays empty (no per-panel minors).
        assert_eq!(shared_proj.minor, shared_minor, "shared path carries minor ticks");
        assert!(indep_proj.minor.is_empty(), "independent path keeps minor empty");

        // Neither mode threads a format when none is supplied; only Thread can
        // ever thread, and only for a numeric format string.
        assert_eq!(shared_threaded, None);
        assert_eq!(indep_threaded, None);
    }

    // ── #52 Task 3: provisional secondary y-axis inputs ──────────────────────

    /// `prepare_render_inputs` derives one `AxesInput.secondary_y` entry per
    /// `independent_y` layer, titled from that layer's own y encoding and
    /// carrying tick labels resolved against ITS OWN domain — not the
    /// primary's (secondary-y-axis, GH #52 Task 3; the layout stage this feeds
    /// reserves one right-side band + emits one axis per entry).
    #[test]
    fn prepare_render_inputs_independent_y_layer_produces_secondary_axis_input() {
        use crate::spec::layer::Layer;

        let primary = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y0".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: false,
        };
        let secondary = Layer {
            mark: Mark::Line,
            encoding: Encoding {
                y: Some(EncodingSpec {
                    field: "y1".into(),
                    title: Some("Secondary Title".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            mark_style: None,
            data_source: None,
            position: None,
            blend: None,
            name: None,
            independent_y: true,
        };

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: Some(vec![primary, secondary]),
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y0", DataType::Float64, false),
            Field::new("y1", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .unwrap();

        let theme = crate::layout::ThemeInputs::default();
        let prep = prepare_render_inputs(&spec, &batch, &theme, None).unwrap();

        assert_eq!(prep.axes.secondary_y.len(), 1, "one independent_y layer → one secondary axis input");
        let secondary_axis = &prep.axes.secondary_y[0];
        assert_eq!(secondary_axis.title.as_deref(), Some("Secondary Title"));
        assert_ne!(
            secondary_axis.tick_labels, prep.axes.y.tick_labels,
            "the secondary axis resolves its OWN domain, not the primary's"
        );
    }

    /// Wire back-compat: no layer sets `independent_y` → `secondary_y` stays
    /// empty. Byte-stable gate mirroring the scene_build.rs Task 2 gate.
    #[test]
    fn prepare_render_inputs_no_independent_y_layer_leaves_secondary_y_empty() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();

        let theme = crate::layout::ThemeInputs::default();
        let prep = prepare_render_inputs(&spec, &batch, &theme, None).unwrap();
        assert!(prep.axes.secondary_y.is_empty());
    }
}

// ── Grid-mode composite-key partitioning unit tests ──────────────────────────

#[cfg(test)]
mod grid_partition_tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a 3×3 grid batch: 3 row categories × 3 col categories, 2 data rows each.
    fn grid_3x3_batch() -> RecordBatch {
        let mut col_vals: Vec<&str> = Vec::new();
        let mut row_vals: Vec<&str> = Vec::new();
        let cols = ["c1", "c2", "c3"];
        let rows = ["r1", "r2", "r3"];
        for &r in &rows {
            for &c in &cols {
                // 2 data rows per (row, col) cell.
                col_vals.push(c); row_vals.push(r);
                col_vals.push(c); row_vals.push(r);
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("col_cat", DataType::Utf8, false),
            Field::new("row_cat", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(col_vals)),
            Arc::new(StringArray::from(row_vals)),
        ]).unwrap()
    }

    #[test]
    fn partition_by_two_fields_produces_correct_count() {
        let batch = grid_3x3_batch();
        let partitions = partition_batch_by_two_fields(&batch, "col_cat", "row_cat").unwrap();
        // 3 row values × 3 col values = 9 partitions.
        assert_eq!(
            partitions.len(), 9,
            "Expected 9 composite partitions for a 3×3 grid; got {}",
            partitions.len()
        );
    }

    #[test]
    fn partition_by_two_fields_each_partition_has_correct_row_count() {
        let batch = grid_3x3_batch();
        let partitions = partition_batch_by_two_fields(&batch, "col_cat", "row_cat").unwrap();
        for ((col_v, row_v), part_batch) in &partitions {
            assert_eq!(
                part_batch.num_rows(), 2,
                "Partition ({col_v}, {row_v}) must have 2 rows; got {}",
                part_batch.num_rows()
            );
        }
    }

    #[test]
    fn partition_by_two_fields_row_major_order() {
        let batch = grid_3x3_batch();
        let partitions = partition_batch_by_two_fields(&batch, "col_cat", "row_cat").unwrap();
        // Row-major: (r1,c1), (r1,c2), (r1,c3), (r2,c1), ...
        let keys: Vec<(&str, &str)> = partitions.iter()
            .map(|((cv, rv), _)| (cv.as_str(), rv.as_str()))
            .collect();
        assert_eq!(keys[0], ("c1", "r1"), "First partition should be (c1, r1)");
        assert_eq!(keys[1], ("c2", "r1"), "Second partition should be (c2, r1)");
        assert_eq!(keys[2], ("c3", "r1"), "Third partition should be (c3, r1)");
        assert_eq!(keys[3], ("c1", "r2"), "Fourth partition should be (c1, r2)");
        assert_eq!(keys[8], ("c3", "r3"), "Last partition should be (c3, r3)");
    }

    #[test]
    fn group_rows_by_two_fields_produces_correct_count_and_row_keys() {
        let batch = grid_3x3_batch();
        let groups = group_rows_by_two_fields(&batch, "col_cat", "row_cat").unwrap();
        assert_eq!(groups.len(), 9, "Expected 9 groups for a 3×3 grid; got {}", groups.len());

        for group in &groups {
            assert!(
                group.row_key.is_some(),
                "Every grid-mode group must have a row_key; got None for col={}",
                group.key.value
            );
        }

        // First group: col="c1", row="r1".
        assert_eq!(groups[0].key.value, "c1");
        assert_eq!(groups[0].row_key.as_ref().unwrap().value, "r1");
        // Last group: col="c3", row="r3".
        assert_eq!(groups[8].key.value, "c3");
        assert_eq!(groups[8].row_key.as_ref().unwrap().value, "r3");
        // All groups have 2 rows each (from the fixture).
        for g in &groups {
            assert_eq!(g.n_rows, 2, "Expected 2 rows per group; got {}", g.n_rows);
        }
    }

    #[test]
    fn wrap_mode_groups_have_no_row_key() {
        // Single-field (wrap mode) grouping must leave row_key = None.
        let batch = grid_3x3_batch();
        let groups = group_rows_by_field(&batch, "col_cat").unwrap();
        for g in &groups {
            assert!(
                g.row_key.is_none(),
                "Wrap-mode groups must have row_key = None; got {:?}",
                g.row_key
            );
        }
    }

    // ── T1.9 (SPINE-06): facet_str_arr contract ──────────────────────────────

    /// A minimal batch with one Utf8 column `"cat"` and one Float64 column `"v"`.
    /// Used by all three `facet_str_arr` tests below.
    fn facet_str_arr_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    /// Happy path: `facet_str_arr` on a present Utf8 column returns the
    /// underlying `StringArray` with the correct values.
    #[test]
    fn facet_str_arr_returns_correct_utf8_array() {
        let batch = facet_str_arr_batch();
        let arr = facet_str_arr(&batch, "cat").expect("cat is a Utf8 column");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), "a");
        assert_eq!(arr.value(1), "b");
        assert_eq!(arr.value(2), "a");
    }

    /// Missing column: `facet_str_arr` must return `RenderError::UnknownColumn`
    /// (not panic) when the field is absent from the batch.
    #[test]
    fn facet_str_arr_missing_column_errors_unknown_column() {
        let batch = facet_str_arr_batch();
        let err = facet_str_arr(&batch, "nonexistent").unwrap_err();
        assert!(
            matches!(err, RenderError::UnknownColumn { .. }),
            "expected UnknownColumn, got: {err:?}"
        );
    }

    /// Non-Utf8 column: `facet_str_arr` on a Float64 column must return a
    /// `RenderError::ScaleResolutionFailed` containing "Phase 7 limitation"
    /// (the SPINE-06 error contract pinned for the three call sites).
    #[test]
    fn facet_str_arr_non_utf8_column_errors_phase7_limitation() {
        let batch = facet_str_arr_batch();
        let err = facet_str_arr(&batch, "v").unwrap_err();
        match &err {
            RenderError::ScaleResolutionFailed(msg) => {
                assert!(
                    msg.contains("Phase 7 limitation"),
                    "error message must contain 'Phase 7 limitation', got: {msg:?}"
                );
            }
            other => panic!("expected ScaleResolutionFailed, got: {other:?}"),
        }
    }
}
