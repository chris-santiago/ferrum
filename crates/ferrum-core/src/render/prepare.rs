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
    LegendOrient, SymbolKind,
};
use crate::spec::chart::ChartSpec;
use crate::transform::context::TransformContext;
use crate::transform::core::{apply_transforms_named, FINAL_OUTPUT_KEY};

use super::scale_resolve::ResolvedScales;
use super::{RenderError, RenderWarning};

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
        encoding.inherit_from(&spec.encoding);
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
    /// D13: legend orient override from `encoding.color.legend.orient`.
    /// `None` means use the theme default. The renderer applies this by
    /// temporarily replacing `theme.legend_orient` before calling
    /// `compute_layout`.
    pub legend_orient_override: Option<crate::layout::LegendOrient>,
    /// D13: legend title override from `encoding.color.legend.title`.
    /// `Some(s)` replaces the default field-name legend title.
    pub legend_title_override: Option<String>,
    /// D13: legend title font size override from `encoding.color.legend.titleFontSize`.
    pub legend_title_font_size_override: Option<f64>,
    /// D13: legend columns override from `encoding.color.legend.columns`.
    /// When `Some`, categorical legend entries are arranged in N columns instead of
    /// the default single vertical column.
    pub legend_columns_override: Option<u32>,
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
        // Facet-before-transform: partition → per-panel transforms → inject facet column → concat
        let partitions = partition_batch_by_field(&normalized, &fspec.field)?;
        let mut merged: HashMap<String, Vec<RecordBatch>> = HashMap::new();
        for (facet_value, partition_batch) in &partitions {
            let mut panel_outputs = apply_transforms_named(&spec.transforms, partition_batch, &ctx)
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
            // Ensure every output batch has the facet column (transforms like
            // Smooth replace the batch entirely, losing the facet column).
            for batch in panel_outputs.values_mut() {
                *batch = inject_facet_column(batch, &fspec.field, facet_value);
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
    let x_field = rendering_encoding
        .x
        .as_ref()
        .map(|e| {
            spec.encoding.x.as_ref().and_then(|p| p.title.clone())
                .or_else(|| e.title.clone())
                .unwrap_or_else(|| e.field.clone())
        });
    let y_field = rendering_encoding
        .y
        .as_ref()
        .map(|e| {
            spec.encoding.y.as_ref().and_then(|p| p.title.clone())
                .or_else(|| e.title.clone())
                .unwrap_or_else(|| e.field.clone())
        });
    let x_tick_labels = provisional_scales.x.tick_labels(10);
    // Y-axis tick labels arrive in domain order (low → high). `layout_y_axis`
    // places the first label at the TOP of the panel, which is the correct
    // top-down convention for ordinal y (heatmaps, confusion matrices) but
    // INVERTS quantitative/temporal labels relative to the data placement
    // (scale_resolve.rs inverts the pixel range for non-ordinal y so high
    // data → top pixel). Reverse the tick labels here for non-ordinal y so
    // the axis labels and data points share the same orientation.
    let mut y_tick_labels = provisional_scales.y.tick_labels(10);
    if !matches!(provisional_scales.y, crate::render::scale_resolve::ScaleKind::Ordinal(_)) {
        y_tick_labels.reverse();
    }
    // D7 + D12: extract per-axis style fields from encoding.axis and encoding.format.
    // All new fields default to the safe backward-compat value so SVG output is
    // byte-identical when the encoding carries no axis/format overrides.
    let x_enc_axis = rendering_encoding.x.as_ref().and_then(|e| e.axis.as_ref());
    let y_enc_axis = rendering_encoding.y.as_ref().and_then(|e| e.axis.as_ref());
    let x_axis_labels = x_enc_axis
        .and_then(|a| a.extra.get("labels"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let x_axis_ticks = x_enc_axis
        .and_then(|a| a.extra.get("ticks"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let x_axis_domain = x_enc_axis
        .and_then(|a| a.extra.get("domain"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let x_axis_grid = x_enc_axis
        .and_then(|a| a.extra.get("grid"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let x_label_angle = x_enc_axis
        .and_then(|a| a.extra.get("labelAngle").or_else(|| a.extra.get("label_angle")))
        .and_then(|v| v.as_f64());
    let x_axis_title = x_enc_axis
        .and_then(|a| a.extra.get("title"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let y_axis_labels = y_enc_axis
        .and_then(|a| a.extra.get("labels"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let y_axis_ticks = y_enc_axis
        .and_then(|a| a.extra.get("ticks"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let y_axis_domain = y_enc_axis
        .and_then(|a| a.extra.get("domain"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let y_axis_grid = y_enc_axis
        .and_then(|a| a.extra.get("grid"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let y_label_angle = y_enc_axis
        .and_then(|a| a.extra.get("labelAngle").or_else(|| a.extra.get("label_angle")))
        .and_then(|v| v.as_f64());
    let y_axis_title = y_enc_axis
        .and_then(|a| a.extra.get("title"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    // D12: apply encoding.format to x/y tick labels.
    let x_tick_format = rendering_encoding.x.as_ref().and_then(|e| e.format.clone());
    let x_tick_format_type = rendering_encoding.x.as_ref().and_then(|e| e.format_type.clone());
    let y_tick_format = rendering_encoding.y.as_ref().and_then(|e| e.format.clone());
    let y_tick_format_type = rendering_encoding.y.as_ref().and_then(|e| e.format_type.clone());
    // Apply format to pre-computed tick label strings.
    let x_tick_labels = apply_tick_format(x_tick_labels, x_tick_format.as_deref(), x_tick_format_type.as_deref());
    let y_tick_labels = apply_tick_format(y_tick_labels, y_tick_format.as_deref(), y_tick_format_type.as_deref());

    let axes = AxesInput {
        x: AxisInput {
            orient: AxisOrient::Bottom,
            title: x_axis_title.or(x_field),
            tick_labels: x_tick_labels,
            label_angle_override: x_label_angle,
            show_labels: x_axis_labels,
            show_ticks: x_axis_ticks,
            show_domain: x_axis_domain,
            show_grid: x_axis_grid,
            tick_format: None, // already applied above
            tick_format_type: None,
        },
        y: AxisInput {
            orient: AxisOrient::Left,
            title: y_axis_title.or(y_field),
            tick_labels: y_tick_labels,
            label_angle_override: y_label_angle,
            show_labels: y_axis_labels,
            show_ticks: y_axis_ticks,
            show_domain: y_axis_domain,
            show_grid: y_axis_grid,
            tick_format: None,
            tick_format_type: None,
        },
        show_x: spec.axis_x.unwrap_or(true),
        show_y: spec.axis_y.unwrap_or(true),
    };

    let facet_groups = if let Some(fspec) = &spec.facet {
        group_rows_by_field(&transformed, &fspec.field)?
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
        .and_then(|l| l.extra.get("disabled"))
        .and_then(|v| v.as_bool())
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
            Some(super::scale_resolve::ColorScale::Continuous { domain, scheme }) => {
                // Sample the scheme at 11 evenly-spaced positions so the
                // gradient looks smooth without bloating the SVG. The
                // renderer emits these as `linearGradient` stops.
                let n_stops = 11;
                let stops: Vec<(f64, String)> = (0..n_stops).map(|i| {
                    let t = i as f64 / (n_stops - 1) as f64;
                    let color = scheme.sample(t);
                    (t, super::color::fmt_svg(color))
                }).collect();
                // Tick labels: check for explicit tickLabels override from
                // legend extra (e.g. ["Low", "High"] for SHAP beeswarm),
                // else compute 5 ticks across the domain at 0, 0.25, 0.5, 0.75, 1.0.
                // When `format=` is set, apply a Python-style format spec to each tick value.
                let legend_extra_ref = spec
                    .encoding
                    .color
                    .as_ref()
                    .and_then(|c| c.legend.as_ref())
                    .map(|l| &l.extra);
                let custom_tick_labels: Option<Vec<String>> = legend_extra_ref
                    .and_then(|extra| extra.get("tickLabels"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect());
                let format_spec: Option<&str> = legend_extra_ref
                    .and_then(|extra| extra.get("format"))
                    .and_then(|v| v.as_str());
                let tick_labels = if let Some(labels) = custom_tick_labels {
                    labels
                } else {
                    let (lo, hi) = *domain;
                    (0..5).map(|i| {
                        let t = i as f64 / 4.0;
                        let v = lo + t * (hi - lo);
                        if let Some(spec_str) = format_spec {
                            apply_format_spec(v, spec_str)
                        } else {
                            format_colorbar_tick(v, lo, hi)
                        }
                    }).collect()
                };
                (Vec::new(), Some(ColorbarInput { stops, tick_labels }))
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

    // D13: extract legend style overrides from encoding.color.legend extra fields.
    let color_legend_extra = spec
        .encoding
        .color
        .as_ref()
        .and_then(|c| c.legend.as_ref())
        .map(|l| &l.extra);
    let legend_orient_override = color_legend_extra
        .and_then(|extra| extra.get("orient"))
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "right" => Some(LegendOrient::Right),
            "left" => Some(LegendOrient::Left),
            "top" => Some(LegendOrient::Top),
            "bottom" => Some(LegendOrient::Bottom),
            _ => None,
        });
    let legend_title_override = color_legend_extra
        .and_then(|extra| extra.get("title"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let legend_title_font_size_override = color_legend_extra
        .and_then(|extra| {
            extra.get("titleFontSize").or_else(|| extra.get("title_font_size"))
        })
        .and_then(|v| v.as_f64());
    let legend_columns_override = color_legend_extra
        .and_then(|extra| extra.get("columns"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);

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
        legend_orient_override,
        legend_title_override,
        legend_title_font_size_override,
        legend_columns_override,
    })
}

/// Apply a Python-style format spec string to a numeric value.
///
/// Supports the subset commonly used for chart tick labels:
/// - `".Nf"` — fixed-point with N decimal places (e.g. `".2f"` → `"3.14"`)
/// - `".N"` — same as `.Nf` (vega-lite shorthand)
/// - `".N%"` or `"%"` — multiply by 100, format with N decimal places, append `%`
///   (e.g. `".0%"` → `"75%"`, `".1%"` → `"74.5%"`)
/// - `".Ne"` or `".Ng"` — scientific / general notation (falls back to `format_colorbar_tick`)
///
/// Unrecognized specs fall back to `format_colorbar_tick`.
fn apply_format_spec(value: f64, spec: &str) -> String {
    let s = spec.trim();
    // Percent: optional leading `.N` then `%`
    if s.ends_with('%') {
        let prefix = s.trim_end_matches('%');
        let precision: usize = if prefix.is_empty() {
            0
        } else if let Some(n) = prefix.strip_prefix('.') {
            n.parse().unwrap_or(0)
        } else {
            0
        };
        let pct = value * 100.0;
        return format!("{:.prec$}%", pct, prec = precision);
    }
    // Fixed-point: `.Nf` or `.N` (no suffix letter, or `f`)
    let prefix = if s.ends_with('f') { s.trim_end_matches('f') } else { s };
    if let Some(n) = prefix.strip_prefix('.') {
        if let Ok(precision) = n.parse::<usize>() {
            return format!("{:.prec$}", value, prec = precision);
        }
    }
    // Fallback: auto-precision
    format_colorbar_tick(value, value, value)
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

/// D12: apply an encoding-level `format` string to pre-computed tick label strings.
///
/// The scale's `tick_labels()` method returns pre-formatted strings. When the
/// encoding carries an explicit `format` string (e.g. `.2f`), we re-parse each
/// label back to a float and re-format it per the spec. When `format_type` is
/// `"time"`, we treat the label as an epoch-ms integer and use `format_time`.
/// Labels that fail to parse are left unchanged (ordinal labels, already-formatted
/// time strings, etc.).
fn apply_tick_format(
    labels: Vec<String>,
    format: Option<&str>,
    format_type: Option<&str>,
) -> Vec<String> {
    use crate::render::format::{format_numeric, format_time};
    let Some(fmt) = format else { return labels };
    labels
        .into_iter()
        .map(|raw| {
            if format_type == Some("time") {
                // Try to parse as i64 epoch-ms.
                if let Ok(epoch_ms) = raw.parse::<i64>() {
                    return format_time(epoch_ms, 86_400_000);
                }
                // Also try f64 (tick_labels may produce "1.7e12" style).
                if let Ok(f) = raw.parse::<f64>() {
                    return format_time(f as i64, 86_400_000);
                }
                raw
            } else {
                // Re-parse to f64 and apply the numeric format spec.
                if let Ok(v) = raw.parse::<f64>() {
                    let trimmed = fmt.strip_prefix('.').unwrap_or(fmt);
                    let (digits_part, fmt_char) = match trimmed.chars().last() {
                        Some(c @ ('f' | 'e' | 'g')) => (&trimmed[..trimmed.len() - 1], c),
                        _ => return format_numeric(v),
                    };
                    let n: usize = digits_part.parse().unwrap_or(2);
                    match fmt_char {
                        'f' => format!("{v:.*}", n),
                        'e' => format!("{v:.*e}", n),
                        'g' | _ => format_numeric(v),
                    }
                } else {
                    raw // ordinal — pass through
                }
            }
        })
        .collect()
}

/// Partition a RecordBatch by a Utf8 field, returning `(value, filtered_batch)`
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
        })
        .collect())
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
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
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
            },
            Layer {
                mark: Mark::Line,
                encoding: Encoding::default(), // inherits from chart-level
                transforms: vec![],
                mark_style: None,
                data_source: None,
            position: None, blend: None,
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
            position: None, blend: None,
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
}
