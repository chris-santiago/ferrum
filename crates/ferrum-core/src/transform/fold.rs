//! Data transform: Fold — wide to long (melt).
//!
//! Similar to `unpivot` but with Vega-lite's `fold` semantics: takes a list
//! of fields to fold and produces `(key, value)` columns. All non-folded
//! columns are preserved as-is, repeated for each fold entry.

use arrow::array::{Array, ArrayRef, RecordBatch, StringBuilder};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct FoldSpec {
    /// Column names to fold (melt).
    pub fields: Vec<String>,
    /// Output column names: (key_col, value_col). Defaults to ("key", "value").
    #[serde(default = "default_fold_as")]
    pub as_: (String, String),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_fold_as() -> (String, String) {
    ("key".into(), "value".into())
}

pub(crate) fn apply(spec: &FoldSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    if spec.fields.is_empty() {
        return Err(PyValueError::new_err("data_fold: fields must be non-empty"));
    }

    // Validate fold fields exist and collect their indices and dtypes.
    let mut fold_indices: Vec<usize> = Vec::with_capacity(spec.fields.len());
    for f in &spec.fields {
        let idx = schema.index_of(f).map_err(|_| {
            PyValueError::new_err(format!("data_fold: column '{f}' not found"))
        })?;
        fold_indices.push(idx);
    }

    // Determine unified value dtype (widen to Float64 if mixed numeric).
    let value_dtypes: Vec<&DataType> = fold_indices
        .iter()
        .map(|&i| schema.field(i).data_type())
        .collect();
    let unified_dtype = unify_dtypes(&value_dtypes)?;

    // Identify non-fold columns (preserved columns).
    let preserved_indices: Vec<usize> = (0..schema.fields().len())
        .filter(|i| !fold_indices.contains(i))
        .collect();

    let k = spec.fields.len();
    let out_rows = n_rows * k;

    // Build key column: repeat each field name n_rows times.
    let mut key_builder = StringBuilder::with_capacity(out_rows, out_rows * 8);
    for field_name in &spec.fields {
        for _ in 0..n_rows {
            key_builder.append_value(field_name);
        }
    }
    let key_col: ArrayRef = Arc::new(key_builder.finish());

    // Build value column: stack fold columns vertically, cast to unified dtype.
    let fold_arrays_cast: Vec<ArrayRef> = fold_indices
        .iter()
        .map(|&i| {
            cast(batch.column(i), &unified_dtype).map_err(|e| {
                PyValueError::new_err(format!("data_fold: cast: {e}"))
            })
        })
        .collect::<PyResult<_>>()?;
    let refs: Vec<&dyn Array> = fold_arrays_cast.iter().map(|a| a.as_ref()).collect();
    let value_col: ArrayRef = concat(&refs)
        .map_err(|e| PyValueError::new_err(format!("data_fold: concat: {e}")))?;

    // Build preserved columns: replicate each k times.
    let preserved_cols: Vec<ArrayRef> = preserved_indices
        .iter()
        .map(|&i| {
            let col = batch.column(i);
            let repeats: Vec<&dyn Array> = (0..k).map(|_| col.as_ref()).collect();
            concat(&repeats).map_err(|e| {
                PyValueError::new_err(format!("data_fold: replicate: {e}"))
            })
        })
        .collect::<PyResult<_>>()?;

    // Build output schema.
    let mut out_fields: Vec<Field> = preserved_indices
        .iter()
        .map(|&i| schema.field(i).as_ref().clone())
        .collect();
    out_fields.push(Field::new(&spec.as_.0, DataType::Utf8, false));
    out_fields.push(Field::new(&spec.as_.1, unified_dtype, true));
    let out_schema = Arc::new(Schema::new(out_fields));

    let mut out_cols = preserved_cols;
    out_cols.push(key_col);
    out_cols.push(value_col);

    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_fold: {e}")))
}

fn unify_dtypes(dtypes: &[&DataType]) -> PyResult<DataType> {
    if dtypes.is_empty() {
        return Err(PyValueError::new_err("data_fold: no fields to fold"));
    }
    if dtypes.iter().all(|d| *d == dtypes[0]) {
        return Ok(dtypes[0].clone());
    }
    // All numeric → Float64.
    if dtypes.iter().all(|d| is_numeric(d)) {
        return Ok(DataType::Float64);
    }
    Err(PyValueError::new_err(
        "data_fold: fold fields have incompatible types; all must be numeric or same type",
    ))
}

fn is_numeric(d: &DataType) -> bool {
    matches!(
        d,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DataFold")]
#[derive(Debug, Clone)]
pub(crate) struct PyDataFold(pub(crate) TransformSpec);

#[pymethods]
impl PyDataFold {
    #[new]
    #[pyo3(signature = (fields, *, as_key = "key", as_value = "value", name = None))]
    fn new(fields: Vec<String>, as_key: &str, as_value: &str, name: Option<String>) -> PyResult<Self> {
        if fields.is_empty() {
            return Err(PyValueError::new_err("DataFold: fields must be non-empty"));
        }
        Ok(PyDataFold(TransformSpec::Fold(FoldSpec {
            fields,
            as_: (as_key.into(), as_value.into()),
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Fold(s) => format!("DataFold(fields={:?})", s.fields),
            _ => "DataFold(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn fold_wide_to_long() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["x", "y"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
            ],
        )
        .unwrap();

        let spec = FoldSpec {
            fields: vec!["a".into(), "b".into()],
            as_: ("key".into(), "value".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // 2 rows * 2 fold fields = 4 output rows.
        assert_eq!(out.num_rows(), 4);
        assert_eq!(out.num_columns(), 3); // id, key, value

        let ids = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let keys = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let vals = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

        // First 2 rows: key="a", values 1,2
        assert_eq!(keys.value(0), "a");
        assert_eq!(keys.value(1), "a");
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(1), 2.0);

        // Next 2 rows: key="b", values 3,4
        assert_eq!(keys.value(2), "b");
        assert_eq!(keys.value(3), "b");
        assert_eq!(vals.value(2), 3.0);
        assert_eq!(vals.value(3), 4.0);

        // ids are repeated
        assert_eq!(ids.value(0), "x");
        assert_eq!(ids.value(1), "y");
        assert_eq!(ids.value(2), "x");
        assert_eq!(ids.value(3), "y");
    }
}
