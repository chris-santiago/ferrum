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
    AxesInput, AxisInput, AxisOrient, FacetGroup, FacetKey, LegendEntry, SymbolKind,
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
        }
    }

    /// Build a layer by inheriting from chart-level when layer's encoding fields are None.
    pub(crate) fn from_chart_and_layer(
        spec: &crate::spec::chart::ChartSpec,
        layer: &crate::spec::layer::Layer,
    ) -> Self {
        let mut encoding = layer.encoding.clone();
        // Inherit chart-level encoding when layer encoding fields are unset.
        if encoding.x.is_none() {
            encoding.x = spec.encoding.x.clone();
        }
        if encoding.y.is_none() {
            encoding.y = spec.encoding.y.clone();
        }
        if encoding.color.is_none() {
            encoding.color = spec.encoding.color.clone();
        }
        // Also inherit size/shape/opacity (Phase 8a channels) if present
        if encoding.size.is_none() {
            encoding.size = spec.encoding.size.clone();
        }
        if encoding.shape.is_none() {
            encoding.shape = spec.encoding.shape.clone();
        }
        if encoding.opacity.is_none() {
            encoding.opacity = spec.encoding.opacity.clone();
        }
        // Phase 8b Task 22: paired-channel endpoints (ribbon mark and future scale_resolve work).
        if encoding.x2.is_none() {
            encoding.x2 = spec.encoding.x2.clone();
        }
        if encoding.y2.is_none() {
            encoding.y2 = spec.encoding.y2.clone();
        }
        Self {
            mark: layer.mark,
            encoding,
            transforms: layer.transforms.clone(),
            mark_style: layer.mark_style.clone().or_else(|| spec.mark_style.clone()),
            data_source: layer.data_source.clone(),
            position: layer.position.clone().or_else(|| spec.position.clone()),
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
    /// Final output of the chart-level transform pipeline. Equal to
    /// `transform_outputs[FINAL_OUTPUT_KEY]`. Retained as a field so existing
    /// consumers (facet filtering, scale resolution, legend) continue to work
    /// unchanged when no layer sets `data_source`.
    pub transformed: RecordBatch,
    /// All chart-level transform outputs, keyed by their `name` (when present)
    /// plus `FINAL_OUTPUT_KEY` ("__final__") for the pipeline tail. Layers
    /// with `data_source: Some(name)` look up their input batch here; layers
    /// with `data_source: None` resolve to `FINAL_OUTPUT_KEY`.
    pub transform_outputs: HashMap<String, RecordBatch>,
    pub provisional_scales: ResolvedScales,
    pub axes: AxesInput,
    pub facet_groups: Vec<FacetGroup>,
    pub legend_entries: Vec<LegendEntry>,
    pub warnings: Vec<RenderWarning>,
    /// One entry per layer. Single-layer charts have len() == 1.
    pub layers: Vec<LayerPrepared>,
    /// True when spec.coord == Some(CoordKind::Flip). The draw loop uses this
    /// to know that x/y have already been swapped in each layer's encoding.
    pub coord_flipped: bool,
}

pub fn prepare_render_inputs(
    spec: &ChartSpec,
    batch: &RecordBatch,
) -> Result<PreparedInputs, RenderError> {
    if batch.num_rows() == 0 {
        return Err(RenderError::EmptyBatch);
    }

    // Normalize Utf8View columns (e.g. from polars) to Utf8 so downstream
    // downcasts to StringArray succeed uniformly.
    let normalized = normalize_string_views(batch);

    // Build the named-output map. When there are no transforms, the helper
    // still publishes a `FINAL_OUTPUT_KEY` entry pointing at the input batch.
    let ctx = TransformContext::default();
    let transform_outputs = apply_transforms_named(&spec.transforms, &normalized, &ctx)
        .map_err(|e| RenderError::TransformFailed(e.to_string()))?;
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
        &crate::layout::ThemeInputs::default(),
    )?;

    let x_field = rendering_encoding.x.as_ref().map(|e| e.field.clone());
    let y_field = rendering_encoding.y.as_ref().map(|e| e.field.clone());
    let x_tick_labels = provisional_scales.x.tick_labels(10);
    let y_tick_labels = provisional_scales.y.tick_labels(10);
    let axes = AxesInput {
        x: AxisInput {
            orient: AxisOrient::Bottom,
            title: x_field,
            tick_labels: x_tick_labels,
            label_angle_override: None,
        },
        y: AxisInput {
            orient: AxisOrient::Left,
            title: y_field,
            tick_labels: y_tick_labels,
            label_angle_override: None,
        },
    };

    let facet_groups = if let Some(fspec) = &spec.facet {
        group_rows_by_field(&transformed, &fspec.field)?
    } else {
        Vec::new()
    };

    let legend_entries = match &provisional_scales.color {
        Some(super::scale_resolve::ColorScale::Categorical { domain, .. }) => domain
            .iter()
            .map(|v| LegendEntry { label: v.clone(), symbol: SymbolKind::Circle })
            .collect(),
        None => Vec::new(),
    };

    Ok(PreparedInputs {
        transformed,
        transform_outputs,
        provisional_scales,
        axes,
        facet_groups,
        legend_entries,
        warnings: scale_warnings,
        layers,
        coord_flipped,
    })
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
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        }
    }

    #[test]
    fn prepare_returns_axes_and_groups_and_legend() {
        let spec = spec_color_facet();
        let batch = batch3();
        let prep = prepare_render_inputs(&spec, &batch).unwrap();
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
        let err = prepare_render_inputs(&spec, &batch).unwrap_err();
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
        }
    }

    #[test]
    fn prepare_single_layer_produces_one_layer_prepared() {
        let spec = single_layer_spec();
        let batch = price_weight_batch();
        let prepared = prepare_render_inputs(&spec, &batch).unwrap();
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
            },
            Layer {
                mark: Mark::Line,
                encoding: Encoding::default(), // inherits from chart-level
                transforms: vec![],
                mark_style: None,
                data_source: None,
            position: None,
            },
        ]);
        let batch = price_weight_batch();
        let prepared = prepare_render_inputs(&spec, &batch).unwrap();
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
        let prep = prepare_render_inputs(&spec, &batch).unwrap();
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
        // prep.transformed is a clone of __final__.
        let final_batch = prep.transform_outputs.get("__final__").unwrap();
        assert_eq!(prep.transformed.num_rows(), final_batch.num_rows());
        assert_eq!(prep.transformed.num_columns(), final_batch.num_columns());
        assert_eq!(
            prep.transformed.schema(),
            final_batch.schema(),
            "transformed and __final__ schemas must match"
        );
    }

    #[test]
    fn data_source_some_publishes_named_transform_output() {
        pyo3::Python::initialize();
        let spec = spec_with_one_bin(Some("box".into()));
        let batch = price_weight_batch();
        let prep = prepare_render_inputs(&spec, &batch).unwrap();
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
            position: None,
        }]);
        let batch = price_weight_batch();
        let err = prepare_render_inputs(&spec, &batch).unwrap_err();
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
        let prepared = prepare_render_inputs(&spec, &batch).unwrap();
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
