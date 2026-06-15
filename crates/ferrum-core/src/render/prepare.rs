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
    AuxLegendInput, AxesInput, AxisInput, AxisOrient, ColorbarInput, FacetGroup, FacetKey,
    LegendEntry, LegendOrient, ShapeLegendEntry, SizeLegendEntry, SymbolKind, TickProjection,
};
use crate::spec::chart::ChartSpec;
use crate::transform::context::TransformContext;
use crate::transform::core::{apply_transforms_named, FINAL_OUTPUT_KEY};

use super::scale_resolve::ResolvedScales;
use super::{RenderError, RenderWarning};

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

pub fn prepare_render_inputs(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &crate::layout::ThemeInputs,
) -> Result<PreparedInputs, RenderError> {
    if batch.num_rows() == 0 {
        return Err(RenderError::EmptyBatch);
    }

    // Normalize Utf8View columns (e.g. from polars) to Utf8 so downstream
    // downcasts to StringArray succeed uniformly.
    let normalized = normalize_string_views(batch);

    // Build the named-output map. When faceting is active, partition the input
    // batch by the facet column(s) BEFORE running transforms so each panel gets
    // its own data subset and transforms execute independently per panel.
    // When there is no facet, the pipeline is unchanged (single partition = full batch).
    let ctx = TransformContext::default();
    let transform_outputs = if let Some(fspec) = &spec.facet {
        // Facet-before-transform: partition → per-panel transforms → inject facet column(s) → concat.
        //
        // Grid mode (fspec.row is set): partition on the composite (col_val, row_val) key so each
        // (row, col) cell is a distinct partition. Both the col field and the row field are injected
        // back into every transform output so per-panel filtering can use either.
        //
        // Wrap mode (fspec.row is None): partition on fspec.field only — behavior unchanged.
        // Each partition carries (col_val, optional_row_val) → RecordBatch.
        // Grid mode populates the row value; wrap mode leaves it None.
        let partitions: Vec<(FacetPartitionKey, RecordBatch)> =
            if let Some(row_field) = &fspec.row {
                partition_batch_by_two_fields(&normalized, &fspec.field, row_field)?
                    .into_iter()
                    .map(|((col_val, row_val), batch)| ((col_val, Some(row_val)), batch))
                    .collect()
            } else {
                partition_batch_by_field(&normalized, &fspec.field)?
                    .into_iter()
                    .map(|(col_val, batch)| ((col_val, None), batch))
                    .collect()
            };

        // Pre-compute the global KDE extent from the full (pre-partition) batch.
        // When a KDE transform has `shared_extent=false` and `extent=None`, each
        // partition would use its own range, causing panels to render on different
        // x-scales and making peak positions non-comparable. By fixing the extent
        // to the global range before partitioning, all panels evaluate the KDE on
        // the same x-range, so marks land at comparable pixel positions on the
        // shared axis. This is the correct default for faceted density charts.
        let effective_transforms = fix_kde_extents_for_facet(&spec.transforms, &normalized);

        let mut merged: HashMap<String, Vec<RecordBatch>> = HashMap::new();
        for ((col_value, row_value), partition_batch) in &partitions {
            let mut panel_outputs = apply_transforms_named(&effective_transforms, partition_batch, &ctx)
                .map_err(|e| RenderError::TransformFailed(e.to_string()))?;
            // D10: per-panel imputation
            {
                let final_batch = panel_outputs
                    .get(FINAL_OUTPUT_KEY)
                    .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY");
                let imputed = apply_impute(final_batch, spec);
                if imputed.num_rows() != final_batch.num_rows() {
                    panel_outputs.insert(FINAL_OUTPUT_KEY.to_string(), imputed);
                }
            }
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
                let merged_batch = arrow::compute::concat_batches(&schema, &batches)
                    .map_err(|e| RenderError::TransformFailed(format!(
                        "concat facet partitions for key '{key}': {e}"
                    )))?;
                combined.insert(key, merged_batch);
            }
        }
        combined
    } else {
        // No facet: unchanged pipeline (single partition = full batch).
        let mut outputs = apply_transforms_named(&spec.transforms, &normalized, &ctx)
            .map_err(|e| RenderError::TransformFailed(e.to_string()))?;
        // D10: apply imputation (fill missing group×x combinations with a constant y)
        // on the final batch, when encoding.y.impute = {"value": N} is set.
        {
            let final_batch = outputs
                .get(FINAL_OUTPUT_KEY)
                .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY");
            let imputed = apply_impute(final_batch, spec);
            if imputed.num_rows() != final_batch.num_rows() {
                outputs.insert(FINAL_OUTPUT_KEY.to_string(), imputed);
            }
        }
        outputs
    };
    let transformed = transform_outputs
        .get(FINAL_OUTPUT_KEY)
        .expect("apply_transforms_named must publish FINAL_OUTPUT_KEY")
        .clone();

    // --- Phase 8a: per-layer inputs + CoordFlip ---

    // Build per-layer prepared inputs
    let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));

    let layers: Vec<LayerPrepared> = {
        let raw: Vec<LayerPrepared> = match &spec.layers {
            None => vec![LayerPrepared::from_chart_only(spec)],
            Some(layer_vec) => layer_vec
                .iter()
                .map(|l| LayerPrepared::from_chart_and_layer(spec, l))
                .collect(),
        };
        if coord_flipped {
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
        } else {
            raw
        }
    };

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

    let (provisional_scales, scale_warnings) = crate::render::scale_resolve::resolve_scales_with_outputs(
        &rendering_spec,
        &transformed,
        &transform_outputs,
        (0.0, 1.0),
        (0.0, 1.0),
        theme,
    )?;

    // Axis title resolution priority:
    //   1. Spec-level encoding title (set by user via .encode(y=Y(..., title=...)))
    //   2. Layer-0 encoding title (set by desugar for internal column names)
    //   3. Field name (fallback)
    // User-explicit titles always win; layer-level titles override the field
    // name for diagnostic charts whose layer-0 encoding references a column
    // with a non-semantic name (e.g. "lower_whisker" / "param_value").
    //
    // Empty-string suppression contract: Python forwards `out["title"] = ""`
    // ONLY when `title=None` is explicitly passed by the caller. An absent key
    // means "use the field name" (default); a present empty string means
    // "suppress entirely". We resolve here at the boundary so the rest of the
    // pipeline naturally sees `None` and emits neither a margin nor a text node.
    let x_field = rendering_encoding
        .x
        .as_ref()
        .and_then(|e| {
            // Explicit spec-level title takes highest priority.
            if let Some(explicit) = spec.encoding.x.as_ref().and_then(|p| p.title.as_deref()) {
                if explicit.trim().is_empty() {
                    return None; // explicit suppress: title=None → ""
                }
                return Some(explicit.to_owned());
            }
            // Layer-0 title (desugar for diagnostic column names).
            if let Some(layer_title) = e.title.as_deref() {
                if layer_title.trim().is_empty() {
                    return None;
                }
                return Some(layer_title.to_owned());
            }
            // Fallback: field name.
            Some(e.field.clone())
        });
    let y_field = rendering_encoding
        .y
        .as_ref()
        .and_then(|e| {
            if let Some(explicit) = spec.encoding.y.as_ref().and_then(|p| p.title.as_deref()) {
                if explicit.trim().is_empty() {
                    return None;
                }
                return Some(explicit.to_owned());
            }
            if let Some(layer_title) = e.title.as_deref() {
                if layer_title.trim().is_empty() {
                    return None;
                }
                return Some(layer_title.to_owned());
            }
            Some(e.field.clone())
        });
    // D3 (flexibility campaign): per-channel `Axis(tick_count=N)` controls the
    // target tick count for continuous + temporal axes (default 10). Limiting it
    // is what de-clutters dense temporal axes (e.g. a 72-row series no longer
    // renders 72 overlapping ticks). Read before tick generation so the count
    // feeds every downstream tick call (labels, fractions, minors).
    let x_tick_count = encoding_axis_tick_count(rendering_encoding.x.as_ref()).unwrap_or(10);
    let y_tick_count = encoding_axis_tick_count(rendering_encoding.y.as_ref()).unwrap_or(10);

    let x_tick_labels = provisional_scales.x.tick_labels(x_tick_count);
    // Continuous-axis scale projection (2026-05-30): per-tick domain fractions
    // and the scale's padding fraction, so layout can place continuous ticks at
    // the SAME pixels that data marks land on. `tick_fractions` projects exactly
    // the same tick values as `tick_labels` (both via `ticks_internal(count)`), so
    // the carrier stays index-aligned with the labels. Ordinal/discretizing
    // scales return an empty vec → carrier is `None` → uniform-slot placement.
    let x_tick_fractions = provisional_scales.x.tick_fractions(x_tick_count);
    let x_scale_padding_frac = provisional_scales.x.padding_fraction();
    // Y-axis tick labels arrive in domain order (low → high). `layout_y_axis`
    // places the first label at the TOP of the panel, which is the correct
    // top-down convention for ordinal y (heatmaps, confusion matrices) but
    // INVERTS quantitative/temporal labels relative to the data placement
    // (scale_resolve.rs inverts the pixel range for non-ordinal y so high
    // data → top pixel). Reverse the tick labels here for non-ordinal y so
    // the axis labels and data points share the same orientation. The projected
    // fractions must be reversed in lockstep so `label[i]` aligns with
    // `fraction[i]`.
    let mut y_tick_labels = provisional_scales.y.tick_labels(y_tick_count);
    let mut y_tick_fractions = provisional_scales.y.tick_fractions(y_tick_count);
    let y_scale_padding_frac = provisional_scales.y.padding_fraction();
    if !matches!(provisional_scales.y, crate::render::scale_resolve::ScaleKind::Ordinal(_)) {
        y_tick_labels.reverse();
        y_tick_fractions.reverse();
    }
    // D7 + D12 (B5-typed): extract per-axis style fields from the typed
    // `encoding.axis` style spec. The show toggles default to `true` (and title
    // falls through to the field name) so SVG output is byte-identical when the
    // encoding carries no axis overrides.
    let x_enc_axis = rendering_encoding.x.as_ref().and_then(|e| e.axis.as_ref());
    let y_enc_axis = rendering_encoding.y.as_ref().and_then(|e| e.axis.as_ref());
    let x_axis_labels = x_enc_axis.and_then(|a| a.labels).unwrap_or(true);
    let x_axis_ticks = x_enc_axis.and_then(|a| a.ticks).unwrap_or(true);
    let x_axis_domain = x_enc_axis.and_then(|a| a.domain).unwrap_or(true);
    let x_axis_grid = x_enc_axis.and_then(|a| a.grid).unwrap_or(true);
    // Axis(title=...) resolution: the outer Option distinguishes "key absent" from
    // "key present but empty".
    //   - None (absent)  → fall through to x_field (field-name default)
    //   - Some("") or Some("  ") → explicit suppress; final title is None, no fallback
    //   - Some("Custom") → use "Custom" directly, no fallback
    let x_axis_title: Option<String> = match x_enc_axis.and_then(|a| a.title.as_deref()) {
        Some(s) if s.trim().is_empty() => None, // explicit suppress: don't fall back
        Some(s) => Some(s.to_owned()),           // explicit non-empty title
        None => x_field,                         // absent: fall through to field default
    };
    let y_axis_labels = y_enc_axis.and_then(|a| a.labels).unwrap_or(true);
    let y_axis_ticks = y_enc_axis.and_then(|a| a.ticks).unwrap_or(true);
    let y_axis_domain = y_enc_axis.and_then(|a| a.domain).unwrap_or(true);
    let y_axis_grid = y_enc_axis.and_then(|a| a.grid).unwrap_or(true);
    // Same three-way resolution as x_axis_title above.
    let y_axis_title: Option<String> = match y_enc_axis.and_then(|a| a.title.as_deref()) {
        Some(s) if s.trim().is_empty() => None,
        Some(s) => Some(s.to_owned()),
        None => y_field,
    };
    // D12 + D3: resolve the tick label format. A per-channel `Axis(label_format=,
    // label_format_type=)` (carried in `encoding.axis`) takes precedence over the
    // shorthand `encoding.format`/`format_type`, so an explicit `Axis(...)` always
    // wins. Chart-level `configure_axis(label_format_raw=...)` is applied later in
    // `render/mod.rs` only when no per-encoding override exists.
    let (x_tick_format, x_tick_format_type) = resolve_axis_label_format(rendering_encoding.x.as_ref());
    let (y_tick_format, y_tick_format_type) = resolve_axis_label_format(rendering_encoding.y.as_ref());
    // Apply the format. For temporal axes with an explicit strftime pattern, the
    // raw epoch-ms tick values are formatted directly here (the pre-computed
    // strings have already lost the timestamp), and no override is carried
    // forward. For numeric axes the spec is threaded to `label_format_override`
    // (D3 root-cause fix for `prepare.rs:538`) so `apply_label_format_to_axis`
    // applies it centrally — re-parsing the numeric label strings is lossless.
    let (x_tick_labels, x_label_format_override) = apply_axis_format_or_thread(
        x_tick_labels,
        x_tick_format,
        x_tick_format_type.as_deref(),
        &provisional_scales.x,
        x_tick_count,
        false,
    );
    let (y_tick_labels, y_label_format_override) = apply_axis_format_or_thread(
        y_tick_labels,
        y_tick_format,
        y_tick_format_type.as_deref(),
        &provisional_scales.y,
        y_tick_count,
        // Non-ordinal y labels were reversed above; the raw temporal values must
        // be reversed in lockstep so index `i` still aligns.
        !matches!(provisional_scales.y, crate::render::scale_resolve::ScaleKind::Ordinal(_)),
    );

    // Grid item 18: minor tick fractions from the provisional scales, projected
    // through the same `[0,1]`-range scale that places majors. Carried into the
    // AxisInput's `TickProjection.minor` so layout can build
    // `AxisLayout.minor_ticks`. The `theme.grid.minor` gate (default `false`) is
    // the single source of truth: when off, `minor` is empty → no minors built →
    // default output byte-identical. "Minor enabled" downstream is derived purely
    // from `minor` being non-empty. Categorical/discretizing scales yield an
    // empty vec (no continuum to subdivide), so they stay minor-free regardless.
    let x_minor_fractions = if theme.grid.minor {
        provisional_scales.x.minor_tick_fractions()
    } else {
        Vec::new()
    };
    let y_minor_fractions = if theme.grid.minor {
        provisional_scales.y.minor_tick_fractions()
    } else {
        Vec::new()
    };

    // B5: per-channel axis STYLING + positioning overrides (grid color/dash/
    // width, label color/font-size, domain color/width, title styling,
    // tick_values, padding, and the unit-2 orphans orient/translate/extents/
    // grid_opacity/title_orient/zindex/tick_min_step/tick_extra).
    let mut x_axis_style = encoding_axis_style_overrides(x_enc_axis.map(Box::as_ref), "x")?;
    let mut y_axis_style = encoding_axis_style_overrides(y_enc_axis.map(Box::as_ref), "y")?;

    // Continuous-axis scale projection: an empty major-fraction vec means the
    // axis is categorical/discretizing (ordinal) → carrier is `None`, so layout
    // keeps the uniform-slot formula byte-identically. A non-empty vec means a
    // continuous scale → carrier drives scale-projected placement, and groups the
    // padding fraction + projected major/minor fractions that share the inset.
    let x_tick_projection = (!x_tick_fractions.is_empty()).then(|| TickProjection {
        padding_frac: x_scale_padding_frac,
        major: x_tick_fractions,
        minor: x_minor_fractions,
    });
    let y_tick_projection = (!y_tick_fractions.is_empty()).then(|| TickProjection {
        padding_frac: y_scale_padding_frac,
        major: y_tick_fractions,
        minor: y_minor_fractions,
    });

    // Per-channel `orient` selects the axis side; absent → the dimension default
    // (Bottom for x, Left for y). Validated above against the channel dimension.
    // Stored as the concrete `AxisInput.orient`; the `orient` override input stays
    // in the bundle so a later chart-level `configure_axis(orient=...)` fills it
    // only when still `None` (per-channel wins), then `resolve_orient` re-syncs.
    let x_orient = x_axis_style.orient.unwrap_or(AxisOrient::Bottom);
    let y_orient = y_axis_style.orient.unwrap_or(AxisOrient::Left);

    // Seed the per-channel `label_format` (resolved via the temporal/numeric
    // threading above) onto the bundle. `label_angle` is already on the bundle
    // (parsed from the same `encoding.axis` spec as `x_label_angle`).
    x_axis_style.label_format = x_label_format_override;
    y_axis_style.label_format = y_label_format_override;

    let axes = AxesInput {
        x: AxisInput {
            orient: x_orient,
            title: x_axis_title,
            tick_labels: x_tick_labels,
            show_labels: x_axis_labels,
            show_ticks: x_axis_ticks,
            show_domain: x_axis_domain,
            show_grid: x_axis_grid,
            tick_format: None, // already applied above
            tick_format_type: None,
            tick_projection: x_tick_projection,
            overrides: x_axis_style,
        },
        y: AxisInput {
            orient: y_orient,
            title: y_axis_title,
            tick_labels: y_tick_labels,
            show_labels: y_axis_labels,
            show_ticks: y_axis_ticks,
            show_domain: y_axis_domain,
            show_grid: y_axis_grid,
            tick_format: None,
            tick_format_type: None,
            tick_projection: y_tick_projection,
            overrides: y_axis_style,
        },
        show_x: spec.axis_x.unwrap_or(true),
        show_y: spec.axis_y.unwrap_or(true),
    };

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

    // Schwabish SB3 (2026-05-11): respect ``legend={"disabled": true}`` on the
    // color encoding by emitting no legend entries AND no colorbar. The
    // Python ``Color`` class translates ``legend=None`` / ``legend=False``
    // from ``encode(color=Color(field, legend=None))`` into this JSON shape
    // so direct-label diagnostic charts can opt out of redundant legends.
    let legend_disabled = spec
        .encoding
        .color
        .as_ref()
        .and_then(|c| c.legend.as_ref())
        .and_then(|l| l.disabled)
        .unwrap_or(false);
    let (legend_entries, colorbar): (Vec<LegendEntry>, Option<ColorbarInput>) =
        if legend_disabled {
            (Vec::new(), None)
        } else {
            match &provisional_scales.color {
            Some(super::scale_resolve::ColorScale::Categorical { domain, .. }) => {
                let entries = domain.iter()
                    .map(|v| LegendEntry { label: v.clone(), symbol: SymbolKind::Circle })
                    .collect();
                (entries, None)
            }
            Some(super::scale_resolve::ColorScale::Continuous { domain, scheme, .. }) => {
                // Sample the scheme at 11 evenly-spaced positions so the
                // gradient looks smooth without bloating the SVG. The
                // renderer emits these as `linearGradient` stops.
                let n_stops = 11;
                let stops: Vec<(f64, String)> = (0..n_stops).map(|i| {
                    let t = i as f64 / (n_stops - 1) as f64;
                    let color = scheme.sample(t);
                    (t, super::color::fmt_svg(color))
                }).collect();
                // Tick labels: check for explicit tickLabels override from the
                // typed legend spec (e.g. ["Low", "High"] for SHAP beeswarm),
                // else compute 5 ticks across the domain at 0, 0.25, 0.5, 0.75, 1.0.
                // When `format=` is set, apply a Python-style format spec to each tick value.
                let color_legend = spec
                    .encoding
                    .color
                    .as_ref()
                    .and_then(|c| c.legend.as_ref());
                let custom_tick_labels: Option<Vec<String>> =
                    color_legend.and_then(|l| l.tick_labels.clone());
                let format_spec: Option<&str> =
                    color_legend.and_then(|l| l.format.as_deref());
                // `cb_domain` is the numeric span the labels cover — carried for
                // `tick_min_step` thinning in `compute_layout`. `None` for explicit
                // non-numeric label overrides (their step is undefined).
                let (tick_labels, cb_domain) = if let Some(labels) = custom_tick_labels {
                    (labels, None)
                } else {
                    let (lo, hi) = *domain;
                    let labels = (0..5).map(|i| {
                        let t = i as f64 / 4.0;
                        let v = lo + t * (hi - lo);
                        if let Some(spec_str) = format_spec {
                            crate::render::format::format_with_spec(v, Some(spec_str))
                        } else {
                            format_colorbar_tick(v, lo, hi)
                        }
                    }).collect();
                    (labels, Some((lo, hi)))
                };
                (Vec::new(), Some(ColorbarInput { stops, tick_labels, domain: cb_domain }))
            }
            None => (Vec::new(), None),
            }
        };

    // Legend title (Themes-T2.5b): default to the color encoding's field name.
    let legend_title = if !legend_entries.is_empty() || colorbar.is_some() {
        spec.encoding.color.as_ref().map(|c| c.field.clone())
    } else {
        None
    };

    // D13 (B5-typed): extract legend style overrides from the typed
    // `encoding.color.legend` spec. Per-channel precedence: these win over
    // chart-level `configure_legend` (which fills only what is still `None`).
    let color_legend = spec
        .encoding
        .color
        .as_ref()
        .and_then(|c| c.legend.as_ref());
    let legend_overrides = LegendPreparedOverrides {
        orient: color_legend.and_then(|l| l.orient.as_deref()).and_then(|s| match s {
            "right"  => Some(LegendOrient::Right),
            "left"   => Some(LegendOrient::Left),
            "top"    => Some(LegendOrient::Top),
            "bottom" => Some(LegendOrient::Bottom),
            _ => None,
        }),
        title: color_legend.and_then(|l| l.title.clone()),
        title_font_size: color_legend.and_then(|l| l.title_font_size),
        columns: color_legend.and_then(|l| l.columns),
        tick_count: color_legend.and_then(|l| l.tick_count).map(|n| n as usize),
        label_font_size: color_legend.and_then(|l| l.label_font_size),
        gradient_length: color_legend.and_then(|l| l.gradient_length),
        gradient_thickness: color_legend.and_then(|l| l.gradient_thickness),
        direction: color_legend.and_then(|l| l.direction.as_deref()).and_then(|s| match s {
            "horizontal" => Some(crate::layout::LegendDirection::Horizontal),
            "vertical"   => Some(crate::layout::LegendDirection::Vertical),
            _ => None,
        }),
        // `values`: explicit tick/entry labels for the legend. Accepts an array of
        // strings or numbers. Numbers are formatted to a short decimal string.
        values: color_legend.and_then(|l| l.values.as_ref()).map(|arr| {
            arr.iter()
                .map(|item| {
                    if let Some(s) = item.as_str() {
                        s.to_string()
                    } else if let Some(n) = item.as_f64() {
                        if n.fract() == 0.0 && n.abs() < 1e15 {
                            format!("{}", n as i64)
                        } else {
                            format!("{:.4}", n).trim_end_matches('0').trim_end_matches('.').to_string()
                        }
                    } else {
                        item.to_string()
                    }
                })
                .collect()
        }),
        // `type`: "gradient" forces colorbar path; "symbol" forces categorical entries.
        legend_type: color_legend.and_then(|l| l.legend_type.clone()),
        symbol_type: color_legend.and_then(|l| l.symbol_type.clone()),
        // B5 unit 3: orphan legend styling/positioning fields. Per-channel here;
        // chart-level `configure_legend(...)` fills any that stay `None`.
        symbol_stroke_width: color_legend.and_then(|l| l.symbol_stroke_width),
        row_padding: color_legend.and_then(|l| l.row_padding),
        column_padding: color_legend.and_then(|l| l.column_padding),
        label_limit: color_legend.and_then(|l| l.label_limit),
        clip_height: color_legend.and_then(|l| l.clip_height),
        tick_min_step: color_legend.and_then(|l| l.tick_min_step),
        zindex: color_legend.and_then(|l| l.zindex),
        // B5 unit 6a orphans. Per-channel here; chart-level `configure_legend`
        // fills any that stay `None`.
        symbol_size: color_legend.and_then(|l| l.symbol_size),
        label_color: color_legend.and_then(|l| l.label_color.clone()),
        offset: color_legend.and_then(|l| l.offset),
        padding: color_legend.and_then(|l| l.padding),
        title_padding: color_legend.and_then(|l| l.title_padding),
    };

    // Multivariate B1: build size/shape auxiliary legends from the resolved
    // scales. A size/shape channel that shares its field with the color channel
    // is merged into the color legend rather than emitted as a separate block.
    let aux_legends = build_aux_legends(spec, &provisional_scales);

    // Same-field merge: when color (continuous) and size share a field, the
    // combined block is the size legend whose symbols also carry color
    // (`color_hex`). Suppress the now-redundant colorbar so a single combined
    // legend renders rather than a colorbar plus a size legend.
    let colorbar = {
        let merged_color_size = aux_legends.iter().any(|a| matches!(
            a,
            AuxLegendInput::Size { entries, .. }
                if entries.iter().any(|e| e.color_hex.is_some())
        ));
        if merged_color_size { None } else { colorbar }
    };
    let legend_title = if colorbar.is_none() && legend_entries.is_empty() {
        None
    } else {
        legend_title
    };

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

/// Read whether a channel's typed `legend` spec carries `disabled: true`
/// (`legend=None` / `legend=False` from the Python `Size`/`Shape` classes).
fn legend_channel_disabled(enc: Option<&crate::spec::encoding::EncodingSpec>) -> bool {
    enc.and_then(|e| e.legend.as_ref())
        .and_then(|l| l.disabled)
        .unwrap_or(false)
}

/// Optional explicit legend title from a channel's typed `legend.title`,
/// falling back to the channel's field name.
fn aux_legend_title(enc: &crate::spec::encoding::EncodingSpec) -> Option<String> {
    let explicit = enc.legend.as_ref().and_then(|l| l.title.clone());
    explicit.or_else(|| Some(enc.field.clone()))
}

/// Build the size/shape auxiliary legend blocks.
///
/// Size: graduated symbols at ~5 nice round values spanning the size domain
/// (`nice_ticks`), each scaled to the size scale's pixel radius and labeled
/// with the value. Shape: one glyph per category in the shape scale.
///
/// Same-field merge (Vega-Lite behavior): when the size channel shares its
/// field with a *continuous* color encoding, the size legend's symbols also
/// carry the color the shared field maps to (`color_hex`), and the colorbar is
/// expected to be suppressed by the caller — a single combined block. A size or
/// shape channel that shares its field with a categorical color encoding is
/// suppressed entirely (the color legend already labels that field).
fn build_aux_legends(
    spec: &ChartSpec,
    scales: &ResolvedScales,
) -> Vec<AuxLegendInput> {
    use crate::render::format::format_with_spec;
    use crate::scale::ticks::nice_ticks;

    let color_field = spec.encoding.color.as_ref().map(|c| c.field.as_str());
    let color_is_continuous = matches!(
        scales.color,
        Some(crate::render::scale_resolve::ColorScale::Continuous { .. })
    );

    let mut out = Vec::new();

    // ── Size legend ──────────────────────────────────────────────────────
    if let (Some(size_scale), Some(size_enc)) = (&scales.size, &spec.encoding.size) {
        let disabled = legend_channel_disabled(spec.encoding.size.as_ref());
        let same_field_as_color = color_field == Some(size_enc.field.as_str());
        // Merge into a categorical color legend → suppress (color labels it).
        let suppressed = disabled || (same_field_as_color && !color_is_continuous);
        if !suppressed {
            let format_spec = size_enc
                .legend
                .as_ref()
                .and_then(|l| l.format.as_deref());
            if let Some((lo, hi)) = size_scale.inner.data_domain() {
                let values = nice_ticks(lo, hi, 5);
                let entries: Vec<SizeLegendEntry> = values
                    .iter()
                    .filter_map(|&v| {
                        // The size scale maps a value to a mark *area* (square
                        // pixels); the point mark draws radius = sqrt(area/π).
                        // Match that exactly so legend symbols equal the marks.
                        let area = size_scale.inner.to_pixel_f64(v)?;
                        let radius = (area / std::f64::consts::PI).sqrt();
                        if !(radius.is_finite() && radius > 0.0) {
                            return None;
                        }
                        // Merge color+size on the same continuous field: sample
                        // the color scale at the value so the symbol varies in
                        // both radius and color.
                        let color_hex = if same_field_as_color && color_is_continuous {
                            scales
                                .color
                                .as_ref()
                                .and_then(|c| c.lookup_f64(v))
                                .map(crate::render::color::fmt_svg)
                        } else {
                            None
                        };
                        Some(SizeLegendEntry {
                            label: format_with_spec(v, format_spec),
                            radius,
                            color_hex,
                        })
                    })
                    .collect();
                if !entries.is_empty() {
                    out.push(AuxLegendInput::Size {
                        title: aux_legend_title(size_enc),
                        entries,
                    });
                }
            }
        }
    }

    // ── Shape legend ─────────────────────────────────────────────────────
    if let (Some(shape_scale), Some(shape_enc)) = (&scales.shape, &spec.encoding.shape) {
        let disabled = legend_channel_disabled(spec.encoding.shape.as_ref());
        let same_field_as_color = color_field == Some(shape_enc.field.as_str());
        // Shape always maps a categorical field; if color encodes the same
        // categorical field, the color legend already enumerates it — suppress.
        let suppressed = disabled || same_field_as_color;
        if !suppressed {
            let entries: Vec<ShapeLegendEntry> = shape_scale
                .domain
                .iter()
                .zip(shape_scale.shapes.iter())
                .map(|(label, kind)| ShapeLegendEntry {
                    label: label.clone(),
                    shape_name: kind.name().to_string(),
                })
                .collect();
            if !entries.is_empty() {
                out.push(AuxLegendInput::Shape {
                    title: aux_legend_title(shape_enc),
                    entries,
                });
            }
        }
    }

    out
}


/// Format a single colorbar tick value into a short human-readable label.
/// Picks decimal precision from the domain span so that small ranges still
/// show enough digits and large ranges don't waste pixels on noise.
fn format_colorbar_tick(value: f64, lo: f64, hi: f64) -> String {
    let span = (hi - lo).abs();
    let precision: usize = if span == 0.0 || !span.is_finite() {
        2
    } else if span >= 100.0 {
        0
    } else if span >= 10.0 {
        1
    } else if span >= 1.0 {
        2
    } else {
        3
    };
    let s = format!("{:.*}", precision, value);
    // Strip trailing zeros / decimal point when the integer form is exact.
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if trimmed.is_empty() { "0".into() } else { trimmed }
    } else {
        s
    }
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
/// [`AxisStyleOverrides`] ready to drop into an `AxisInput` (B5). Color strings
/// are pre-parsed to `Srgba<u8>`; an unparseable hex string yields `None` (theme
/// fallback) rather than failing, matching the chart-level apply behavior.
///
/// `label_format` is left `None` here: prepare resolves it through the temporal/
/// numeric format threading (`apply_axis_format_or_thread`) and seeds it onto the
/// bundle at construction. `orient` is the validated override input; the concrete
/// axis side is resolved into `AxisInput.orient` once all override layers merge.
fn encoding_axis_style_overrides(
    axis: Option<&crate::render::chart_config::AxisStyleSpec>,
    channel: &'static str,
) -> Result<crate::layout::AxisStyleOverrides, RenderError> {
    let Some(a) = axis else { return Ok(crate::layout::AxisStyleOverrides::default()) };
    let parse = |c: &Option<String>| {
        c.as_deref()
            .and_then(|s| crate::render::color::from_hex_str(s).ok())
    };
    let orient = a
        .orient
        .as_deref()
        .map(|s| parse_axis_orient(s, channel))
        .transpose()?;
    let title_orient = a
        .title_orient
        .as_deref()
        .map(|s| parse_title_orient(s, channel))
        .transpose()?;
    Ok(crate::layout::AxisStyleOverrides {
        tick_values: a.values.clone(),
        label_angle: a.label_angle,
        label_format: None,
        title_font_size: a.title_font_size,
        title_color: parse(&a.title_color),
        title_padding: a.title_padding,
        label_padding: a.label_padding,
        label_color: parse(&a.label_color),
        label_font_size: a.label_font_size,
        grid_color: parse(&a.grid_color),
        grid_dash: a.grid_dash.clone(),
        grid_width: a.grid_width,
        domain_color: parse(&a.domain_color),
        domain_width: a.domain_width,
        orient,
        translate: a.translate,
        min_extent: a.min_extent,
        max_extent: a.max_extent,
        grid_opacity: a.grid_opacity,
        title_orient,
        zindex: a.zindex,
        tick_extra: a.tick_extra,
        tick_min_step: a.tick_min_step,
        offset: a.offset,
        label_flush: a.label_flush,
        label_overlap: a.label_overlap.as_deref().and_then(parse_label_overlap),
    })
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

/// Partition a RecordBatch by a Utf8 field, returning `(value, filtered_batch)`
/// For each KDE transform in `transforms` that has `extent=None` and
/// `groupby=None` (single-group path), pre-compute the global extent from
/// the full `batch` and return a new transform list where those KDE specs
/// carry an explicit `extent`. This ensures that all facet partitions evaluate
/// the KDE on the same x-range so mark positions are comparable across panels.
///
/// KDE transforms that already have an explicit `extent`, or that use a
/// `groupby` (multi-group path), or that have `shared_extent=true`, are left
/// unchanged.
///
/// All non-KDE transforms are passed through unchanged.
fn fix_kde_extents_for_facet(
    transforms: &[crate::transform::core::TransformSpec],
    batch: &RecordBatch,
) -> Vec<crate::transform::core::TransformSpec> {
    use arrow::array::Float64Array;
    use crate::transform::core::TransformSpec;

    transforms.iter().map(|t| {
        let TransformSpec::Kde(kde_spec) = t else { return t.clone(); };
        // Only fill-in extent when the spec doesn't already have one, has no
        // groupby (the grouped path manages its own shared_extent logic), and
        // shared_extent is false (shared_extent=true already handles this).
        if kde_spec.extent.is_some() || kde_spec.groupby.is_some() {
            return t.clone();
        }
        // Compute global [lo, hi] from the full batch's KDE field.
        let extent = batch.column_by_name(&kde_spec.field)
            .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
            .and_then(|arr| {
                let (lo, hi) = (0..arr.len()).fold(
                    (f64::INFINITY, f64::NEG_INFINITY),
                    |(lo, hi), i| {
                        if arr.is_null(i) { return (lo, hi); }
                        let v = arr.value(i);
                        if v.is_nan() { return (lo, hi); }
                        (lo.min(v), hi.max(v))
                    },
                );
                if lo.is_finite() && hi.is_finite() && lo < hi { Some((lo, hi)) } else { None }
            });
        match extent {
            Some((lo, hi)) => TransformSpec::Kde(crate::transform::kde::KdeSpec {
                extent: Some((lo, hi)),
                ..kde_spec.clone()
            }),
            None => t.clone(),
        }
    }).collect()
}

/// pairs in first-appearance order. Used by facet-before-transform to split the
/// input into per-panel subsets before running transforms.
fn partition_batch_by_field(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<(String, RecordBatch)>, RenderError> {
    use arrow::array::{Array, BooleanArray, StringArray};
    use arrow::compute::filter_record_batch;
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!(
                "facet field '{field}' must be Utf8 (Phase 7 limitation)"
            ))
        })?;
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
    use arrow::array::StringArray;
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!(
                "facet field '{field}' must be Utf8 (Phase 7 limitation)"
            ))
        })?;
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
    use arrow::array::StringArray;

    let get_str_arr = |field: &str| -> Result<&StringArray, RenderError> {
        let col = batch.column_by_name(field)
            .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
        col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!(
                "facet field '{field}' must be Utf8 (Phase 7 limitation)"
            ))
        })
    };

    let col_arr = get_str_arr(col_field)?;
    let row_arr = get_str_arr(row_field)?;

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
    use arrow::array::{Array, BooleanArray, StringArray};
    use arrow::compute::filter_record_batch;

    let get_str_arr = |field: &str| -> Result<&StringArray, RenderError> {
        let col = batch.column_by_name(field)
            .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
        col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!(
                "facet field '{field}' must be Utf8 (Phase 7 limitation)"
            ))
        })
    };

    let col_arr = get_str_arr(col_field)?;
    let row_arr = get_str_arr(row_field)?;

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
        let prep = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
        prepare_render_inputs(spec, batch, &crate::layout::ThemeInputs::default()).unwrap()
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
        let err = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap_err();
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
        let prepared = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            },
            Layer {
                mark: Mark::Line,
                encoding: Encoding::default(), // inherits from chart-level
                transforms: vec![],
                mark_style: None,
                data_source: None,
            position: None, blend: None, name: None,
            },
        ]);
        let batch = price_weight_batch();
        let prepared = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
        use crate::transform::core::TransformSpec;
        let named = name.is_some();
        let mut spec = single_layer_spec();
        spec.transforms = vec![TransformSpec::Bin(BinSpec {
            field: "price".into(),
            bin_count: Some(2),
            bin_width: None,
            extent: Some((10.0, 30.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
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
        let prep = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
        let prep = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            position: None, blend: None, name: None,
        }]);
        let batch = price_weight_batch();
        let err = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap_err();
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
        let prepared = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
        let err = prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default())
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
                "min_extent": 70.0,
                "max_extent": 120.0,
                "grid_opacity": 0.25,
                "title_orient": "bottom",
                "zindex": 1
            }),
        ));
        let batch = price_weight_batch();
        let prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
        assert_eq!(prep.axes.x.overrides.translate, Some(12.0));
        assert_eq!(prep.axes.x.overrides.min_extent, Some(70.0));
        assert_eq!(prep.axes.x.overrides.max_extent, Some(120.0));
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
        let base_count = baseline.axes.x.tick_labels.len();

        // Pick a min_step larger than the natural tick spacing to force thinning.
        spec.encoding.x = Some(enc_with_axis("price", serde_json::json!({ "tick_min_step": 1e9 })));
        let mut prep =
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&spec, &batch, &crate::layout::ThemeInputs::default()).unwrap();
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
            prepare_render_inputs(&default_spec, &batch, &crate::layout::ThemeInputs::default())
                .unwrap();

        let mut limited_spec = single_layer_spec();
        limited_spec.encoding.x =
            Some(enc_with_axis("price", serde_json::json!({ "tick_count": 2 })));
        let limited_prep =
            prepare_render_inputs(&limited_spec, &batch, &crate::layout::ThemeInputs::default())
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
}

// ── Grid-mode composite-key partitioning unit tests ──────────────────────────

#[cfg(test)]
mod grid_partition_tests {
    use super::*;
    use arrow::array::StringArray;
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
}
