//! Data transform: Filter — row filter using expression evaluator.
//!
//! Evaluates a predicate expression (via `expr.rs`) per row and retains
//! only rows where the expression is truthy.

use arrow::array::{Array, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, TimeUnit};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::expr::{Expr, ExprValue};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FilterSpec {
    pub predicate: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Crossfilter reference (D6): names a selection parameter whose live extent
    /// filters this layer's rows in WASM. Inert in the static renderer — a param
    /// filter carries `predicate: "true"`, so `apply()` keeps every row (the
    /// correct static semantics for an empty selection).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub param: Option<String>,
}

pub(crate) fn apply(spec: &FilterSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let expr = Expr::parse(&spec.predicate).map_err(|e| {
        PyValueError::new_err(format!("data_filter: invalid predicate: {e}"))
    })?;

    let schema = batch.schema();
    let n_rows = batch.num_rows();
    let mut keep_indices: Vec<u32> = Vec::with_capacity(n_rows);

    for row in 0..n_rows {
        let result = expr.eval(&|field_name: &str| {
            field_value(batch, &schema, field_name, row)
        });
        if result.is_truthy() {
            keep_indices.push(row as u32);
        }
    }

    let indices = UInt32Array::from(keep_indices);
    let columns: Vec<_> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &indices, None))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(format!("data_filter: take failed: {e}")))?;

    RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_filter: {e}")))
}

/// Extract a field value from a row as an ExprValue.
fn field_value(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
    field_name: &str,
    row: usize,
) -> ExprValue {
    let idx = match schema.index_of(field_name) {
        Ok(i) => i,
        Err(_) => return ExprValue::Null,
    };
    let col = batch.column(idx);
    if col.is_null(row) {
        return ExprValue::Null;
    }
    match schema.field(idx).data_type() {
        DataType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|arr| ExprValue::Number(arr.value(row)))
            .unwrap_or(ExprValue::Null),
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|arr| ExprValue::Str(arr.value(row).to_string()))
            .unwrap_or(ExprValue::Null),
        DataType::Int32 => col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64))
            .unwrap_or(ExprValue::Null),
        DataType::Int64 => col
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64))
            .unwrap_or(ExprValue::Null),
        DataType::Float32 => col
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64))
            .unwrap_or(ExprValue::Null),
        DataType::Boolean => col
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .map(|arr| ExprValue::Bool(arr.value(row)))
            .unwrap_or(ExprValue::Null),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => col
            .as_any()
            .downcast_ref::<arrow::array::TimestampNanosecondArray>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64 / 1_000_000.0))
            .unwrap_or(ExprValue::Null),
        DataType::Timestamp(TimeUnit::Microsecond, _) => col
            .as_any()
            .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64 / 1_000.0))
            .unwrap_or(ExprValue::Null),
        DataType::Timestamp(TimeUnit::Millisecond, _) => col
            .as_any()
            .downcast_ref::<arrow::array::TimestampMillisecondArray>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64))
            .unwrap_or(ExprValue::Null),
        DataType::Timestamp(TimeUnit::Second, _) => col
            .as_any()
            .downcast_ref::<arrow::array::TimestampSecondArray>()
            .map(|arr| ExprValue::Number(arr.value(row) as f64 * 1_000.0))
            .unwrap_or(ExprValue::Null),
        _ => ExprValue::Null,
    }
}

/// Shared helper: extract field value from a RecordBatch row.
/// Used by both filter and calculate transforms.
pub(crate) fn batch_field_value(
    batch: &RecordBatch,
    field_name: &str,
    row: usize,
) -> ExprValue {
    field_value(batch, &batch.schema(), field_name, row)
}

// No PyO3 wrapper: `DataFilter` is constructed only via the dict-emitting
// `transform_filter` Python function and carried through the `transforms_json`
// serde path (SEAM-02). `FilterSpec` above is the serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn filter_gt_predicate() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "datum.x > 3".into(),
            name: None,
            param: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let xs = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xs.value(0), 4.0);
        assert_eq!(xs.value(1), 5.0);
    }

    #[test]
    fn filter_compound_predicate() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "datum.x >= 2 && datum.y <= 30".into(),
            name: None,
            param: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let xs = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xs.value(0), 2.0);
        assert_eq!(xs.value(1), 3.0);
    }

    #[test]
    fn filter_all_rows_removed() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "datum.x > 100".into(),
            name: None,
            param: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn filter_with_param_round_trips() {
        let spec = FilterSpec {
            predicate: "true".into(),
            name: None,
            param: Some("brush".into()),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json, r#"{"predicate":"true","param":"brush"}"#);
        let parsed: FilterSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn filter_without_param_omits_key() {
        let spec = FilterSpec {
            predicate: "datum.x > 3".into(),
            name: None,
            param: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json, r#"{"predicate":"datum.x > 3"}"#);
        assert!(!json.contains("param"));
    }

    #[test]
    fn filter_param_is_inert_statically() {
        // A param filter carries `predicate: "true"` → apply keeps all rows.
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "true".into(),
            name: None,
            param: Some("brush".into()),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), batch.num_rows());
    }

    #[test]
    fn filter_invalid_expr_errors() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "invalid!!!".into(),
            name: None,
            param: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("data_filter"));
    }
}
