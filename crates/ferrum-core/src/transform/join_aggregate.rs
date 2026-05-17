//! Data transform: JoinAggregate — add aggregate columns without collapsing rows.
//!
//! Computes grouped aggregates and joins them back to the original rows,
//! so the output has the same number of rows as the input plus new columns.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::aggregate::AggFn;

/// A single aggregation specification for JoinAggregate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct AggSpec {
    pub field: String,
    #[serde(rename = "fn")]
    pub fn_: AggFn,
    #[serde(rename = "as")]
    pub as_: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct JoinAggregateSpec {
    pub aggregates: Vec<AggSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &JoinAggregateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();
    let groupby = spec.groupby.clone().unwrap_or_default();

    if spec.aggregates.is_empty() {
        return Err(PyValueError::new_err(
            "data_join_aggregate: aggregates must be non-empty",
        ));
    }

    // Build group keys for each row.
    let group_keys = build_group_keys(batch, &schema, &groupby)?;

    // Collect row indices per group.
    let mut groups: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();
    for (row, key) in group_keys.iter().enumerate() {
        groups.entry(key.clone()).or_default().push(row);
    }

    // For each aggregate, compute per-group value, then broadcast back.
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(spec.aggregates.len());

    for agg in &spec.aggregates {
        let col_idx = schema.index_of(&agg.field).map_err(|_| {
            PyValueError::new_err(format!(
                "data_join_aggregate: column '{}' not found",
                agg.field
            ))
        })?;

        // Compute per-group aggregate.
        let mut row_values: Vec<f64> = vec![f64::NAN; n_rows];

        for (_key, rows) in &groups {
            let agg_val = compute_agg(batch.column(col_idx).as_ref(), rows, agg.fn_)?;
            for &r in rows {
                row_values[r] = agg_val;
            }
        }

        new_cols.push(Arc::new(Float64Array::from(row_values)));
    }

    // Build output schema: original columns + new aggregate columns.
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    for agg in &spec.aggregates {
        fields.push(Field::new(&agg.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.extend(new_cols);

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_join_aggregate: {e}")))
}

fn build_group_keys(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
    groupby: &[String],
) -> PyResult<Vec<Vec<String>>> {
    let n_rows = batch.num_rows();
    let mut keys = vec![Vec::new(); n_rows];

    for g in groupby {
        let idx = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "data_join_aggregate: groupby column '{g}' not found"
            ))
        })?;
        let col = batch.column(idx);
        for row in 0..n_rows {
            let s = extract_key_str(col.as_ref(), row);
            keys[row].push(s);
        }
    }
    Ok(keys)
}

fn extract_key_str(col: &dyn Array, row: usize) -> String {
    if col.is_null(row) {
        return "__null__".to_string();
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return arr.value(row).to_string();
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return format!("{}", arr.value(row));
    }
    "__unknown__".to_string()
}

fn compute_agg(col: &dyn Array, rows: &[usize], fn_: AggFn) -> PyResult<f64> {
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        let vals: Vec<f64> = rows
            .iter()
            .filter_map(|&r| {
                if arr.is_null(r) {
                    return None;
                }
                let v = arr.value(r);
                if v.is_nan() {
                    return None;
                }
                Some(v)
            })
            .collect();

        if vals.is_empty() {
            return Ok(if matches!(fn_, AggFn::Count) {
                0.0
            } else {
                f64::NAN
            });
        }

        Ok(match fn_ {
            AggFn::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
            AggFn::Sum => vals.iter().sum(),
            AggFn::Count => vals.len() as f64,
            AggFn::Min => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggFn::Max => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggFn::Median => {
                let mut sorted = vals;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
                }
            }
        })
    } else {
        // Count on non-numeric columns.
        if matches!(fn_, AggFn::Count) {
            let non_null = rows.iter().filter(|&&r| !col.is_null(r)).count();
            Ok(non_null as f64)
        } else {
            Err(PyValueError::new_err(
                "data_join_aggregate: non-count aggregation requires Float64 column",
            ))
        }
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "JoinAggregate")]
#[derive(Debug, Clone)]
pub(crate) struct PyJoinAggregate(pub(crate) TransformSpec);

#[pymethods]
impl PyJoinAggregate {
    #[new]
    #[pyo3(signature = (*, name = None))]
    fn new(name: Option<String>) -> Self {
        PyJoinAggregate(TransformSpec::JoinAggregate(JoinAggregateSpec {
            aggregates: Vec::new(),
            groupby: None,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        "JoinAggregate(...)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn join_aggregate_preserves_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap();

        let spec = JoinAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Mean,
                as_: "group_mean".into(),
            }],
            groupby: Some(vec!["group".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // Same row count as input.
        assert_eq!(out.num_rows(), 5);
        // Original columns + new aggregate column.
        assert_eq!(out.num_columns(), 3);

        let means = out
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        // Group "a": mean(1,2) = 1.5
        assert!((means.value(0) - 1.5).abs() < 1e-12);
        assert!((means.value(1) - 1.5).abs() < 1e-12);
        // Group "b": mean(3,4,5) = 4.0
        assert!((means.value(2) - 4.0).abs() < 1e-12);
        assert!((means.value(3) - 4.0).abs() < 1e-12);
        assert!((means.value(4) - 4.0).abs() < 1e-12);
    }
}
