use arrow::array::RecordBatch;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::aggregate::AggregateSpec;
use crate::transform::bin::{self, BinSpec};
use crate::transform::kde::KdeSpec;
use crate::transform::smooth::SmoothSpec;
use crate::transform::summary::SummarySpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TransformSpec {
    Bin(BinSpec),
    Kde(KdeSpec),
    Smooth(SmoothSpec),
    Aggregate(AggregateSpec),
    Summary(SummarySpec),
}

impl TransformSpec {
    pub(crate) fn apply(&self, batch: &RecordBatch) -> PyResult<RecordBatch> {
        match self {
            Self::Bin(s)       => bin::apply(s, batch),
            Self::Kde(s)       => crate::transform::kde::apply(s, batch),
            Self::Smooth(s)    => crate::transform::smooth::apply(s, batch),
            Self::Aggregate(s) => crate::transform::aggregate::apply(s, batch),
            Self::Summary(s)   => crate::transform::summary::apply(s, batch),
        }
    }
}

pub(crate) fn apply_transforms(
    specs: &[TransformSpec],
    batch: &RecordBatch,
) -> PyResult<RecordBatch> {
    let mut current = batch.clone(); // Arrow Arc-clones; cheap
    for spec in specs {
        current = spec.apply(&current)?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    use crate::transform::aggregate::{AggregateSpec, AggregateOp, AggFn};
    use crate::transform::bin::BinSpec;

    fn make_one_col_batch(name: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(name, DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    #[test]
    fn test_transform_spec_bin_round_trip() {
        let original = TransformSpec::Bin(BinSpec {
            field: "x".into(),
            bin_count: Some(10),
            bin_width: None,
            extent: None,
            nice: true,
        });
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""type":"bin""#), "missing tag: {json}");
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_apply_transforms_empty_returns_input_unchanged() {
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0]);
        let out = apply_transforms(&[], &batch).unwrap();
        assert_eq!(out.num_rows(), 3);
        assert_eq!(out.num_columns(), 1);
        assert_eq!(out.schema().field(0).name(), "x");
    }

    #[test]
    fn test_pipeline_bin_then_aggregate() {
        pyo3::Python::initialize();
        // Bin produces {bin_start, bin_end, count, density}; aggregate over count by bin_start.
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let pipeline = vec![
            TransformSpec::Bin(BinSpec {
                field: "x".into(),
                bin_count: Some(5),
                bin_width: None,
                extent: Some((1.0, 10.0)),
                nice: false,
            }),
            TransformSpec::Aggregate(AggregateSpec {
                ops: vec![AggregateOp {
                    field: "count".into(),
                    fn_: AggFn::Sum,
                    as_: "total_count".into(),
                }],
                groupby: vec![],
            }),
        ];

        // bin produces UInt64 count, but stat_aggregate requires Float64 for op fields.
        // The pipeline is expected to fail with a clear schema-mismatch error from stat_aggregate.
        let err = apply_transforms(&pipeline, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Float64") || msg.contains("dtype"),
            "expected dtype error from stat_aggregate; got: {msg}");
    }

    #[test]
    fn test_pipeline_schema_mismatch_after_bin() {
        pyo3::Python::initialize();
        // After stat_bin, the input column "x" no longer exists. A follow-up aggregate
        // referring to "x" must raise PyValueError mentioning the missing column.
        let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let pipeline = vec![
            TransformSpec::Bin(BinSpec {
                field: "x".into(),
                bin_count: Some(3),
                bin_width: None,
                extent: Some((1.0, 5.0)),
                nice: false,
            }),
            TransformSpec::Aggregate(AggregateSpec {
                ops: vec![AggregateOp {
                    field: "x".into(),
                    fn_: AggFn::Mean,
                    as_: "m".into(),
                }],
                groupby: vec![],
            }),
        ];
        let err = apply_transforms(&pipeline, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'x'") && (msg.contains("not found") || msg.contains("missing")),
            "expected missing-column error; got: {msg}");
    }
}
