//! prepare_render_inputs(spec, batch) →
//!   1. Apply Phase 5 transforms.
//!   2. Build provisional ResolvedScales for tick-label generation.
//!   3. Derive AxesInput (titles, tick_labels).
//!   4. Group rows by facet field (if facet).
//!   5. Build LegendEntry list (if color encoding).

use arrow::record_batch::RecordBatch;

use crate::layout::{
    AxesInput, AxisInput, AxisOrient, FacetGroup, FacetKey, LegendEntry, SymbolKind,
};
use crate::spec::chart::ChartSpec;
use crate::transform::core::apply_transforms;

use super::scale_resolve::{resolve_scales, ResolvedScales};
use super::{RenderError, RenderWarning};

#[derive(Debug)]
pub struct PreparedInputs {
    pub transformed: RecordBatch,
    pub provisional_scales: ResolvedScales,
    pub axes: AxesInput,
    pub facet_groups: Vec<FacetGroup>,
    pub legend_entries: Vec<LegendEntry>,
    pub warnings: Vec<RenderWarning>,
}

pub fn prepare_render_inputs(
    spec: &ChartSpec,
    batch: &RecordBatch,
) -> Result<PreparedInputs, RenderError> {
    if batch.num_rows() == 0 {
        return Err(RenderError::EmptyBatch);
    }

    let transformed = if spec.transforms.is_empty() {
        batch.clone()
    } else {
        apply_transforms(&spec.transforms, batch)
            .map_err(|e| RenderError::TransformFailed(e.to_string()))?
    };

    let (provisional_scales, scale_warnings) =
        resolve_scales(spec, &transformed, (0.0, 1.0), (0.0, 1.0))?;

    let x_field = spec.encoding.x.as_ref().map(|e| e.field.clone());
    let y_field = spec.encoding.y.as_ref().map(|e| e.field.clone());
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
        provisional_scales,
        axes,
        facet_groups,
        legend_entries,
        warnings: scale_warnings,
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
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "species".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
            }),
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
}
