use arrow::array::RecordBatch;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::bin::{self, BinSpec};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TransformSpec {
    Bin(BinSpec),
}

impl TransformSpec {
    pub(crate) fn apply(&self, batch: &RecordBatch) -> PyResult<RecordBatch> {
        match self {
            Self::Bin(s) => bin::apply(s, batch),
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
}
