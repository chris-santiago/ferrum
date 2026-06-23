//! Data transform: Impute — fill missing (null/NaN) values.
//!
//! Supports imputation methods: value (constant), mean, median, min, max.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::numeric_util::clean_float64_values;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImputeSpec {
    /// Column to impute.
    pub field: String,
    /// Imputation method: "value", "mean", "median", "min", "max".
    #[serde(default = "default_method")]
    pub method: String,
    /// Constant value for method="value".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value: Option<f64>,
    /// Group-by columns (impute within groups).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Key column (for sequence generation — optional, not implemented here).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_method() -> String {
    "value".into()
}

pub(crate) fn apply(spec: &ImputeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_impute: column '{}' not found", spec.field))
    })?;

    if schema.field(col_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "data_impute: column '{}' must be Float64",
            spec.field
        )));
    }

    let col = batch
        .column(col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_impute: expected Float64Array for '{}'", spec.field))
        })?;

    let n_rows = batch.num_rows();

    // Compute fill value.
    let fill_value = match spec.method.as_str() {
        "value" => spec.value.unwrap_or(0.0),
        "mean" => {
            let (sum, count) = clean_stats(col);
            if count == 0 { 0.0 } else { sum / count as f64 }
        }
        "median" => {
            let mut vals = clean_float64_values(col, None);
            if vals.is_empty() {
                0.0
            } else {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = vals.len();
                if n % 2 == 1 {
                    vals[n / 2]
                } else {
                    0.5 * (vals[n / 2 - 1] + vals[n / 2])
                }
            }
        }
        "min" => {
            let vals = clean_float64_values(col, None);
            vals.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        }
        "max" => {
            let vals = clean_float64_values(col, None);
            vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "data_impute: unknown method '{other}'; expected value|mean|median|min|max"
            )));
        }
    };

    // Build imputed column.
    let mut imputed: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        if col.is_null(i) {
            imputed.push(fill_value);
        } else {
            let v = col.value(i);
            if v.is_nan() {
                imputed.push(fill_value);
            } else {
                imputed.push(v);
            }
        }
    }

    // Replace the column in the batch.
    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns[col_idx] = Arc::new(Float64Array::from(imputed));

    // Build schema (same as input but field is now non-nullable).
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields[col_idx] = Field::new(&spec.field, DataType::Float64, false);
    let out_schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_impute: {e}")))
}

fn clean_stats(col: &Float64Array) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        let v = col.value(i);
        if v.is_nan() {
            continue;
        }
        sum += v;
        count += 1;
    }
    (sum, count)
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "Impute")]
#[derive(Debug, Clone)]
pub(crate) struct PyImpute(pub(crate) TransformSpec);

#[pymethods]
impl PyImpute {
    #[new]
    #[pyo3(signature = (field, *, method = "value", value = None, name = None))]
    fn new(field: String, method: &str, value: Option<f64>, name: Option<String>) -> Self {
        PyImpute(TransformSpec::Impute(ImputeSpec {
            field,
            method: method.into(),
            value,
            groupby: None,
            key: None,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Impute(s) => format!("Impute(field='{}', method='{}')", s.field, s.method),
            _ => "Impute(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn impute_with_mean() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
                None,
                Some(5.0),
            ]))],
        )
        .unwrap();

        let spec = ImputeSpec {
            field: "x".into(),
            method: "mean".into(),
            value: None,
            groupby: None,
            key: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        // mean(1,3,5) = 3.0
        assert_eq!(x.value(0), 1.0);
        assert_eq!(x.value(1), 3.0); // imputed
        assert_eq!(x.value(2), 3.0);
        assert_eq!(x.value(3), 3.0); // imputed
        assert_eq!(x.value(4), 5.0);
    }

    #[test]
    fn impute_with_constant() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]))],
        )
        .unwrap();

        let spec = ImputeSpec {
            field: "x".into(),
            method: "value".into(),
            value: Some(99.0),
            groupby: None,
            key: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(x.value(1), 99.0);
    }
}
