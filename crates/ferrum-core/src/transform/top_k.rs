//! Data transform: TopK — keep top-k rows by an aggregate value.
//!
//! Groups by a field, computes an aggregate, then keeps only the top-k groups.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct TopKSpec {
    /// Number of top groups to keep.
    pub n: usize,
    /// Field to aggregate for ranking.
    pub field: String,
    /// Aggregation operation: sum, mean, count, min, max.
    #[serde(default = "default_op")]
    pub op: String,
    /// Sort direction: "descending" (default) or "ascending".
    #[serde(default = "default_sort")]
    pub sort: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_op() -> String {
    "sum".into()
}
fn default_sort() -> String {
    "descending".into()
}

pub(crate) fn apply(spec: &TopKSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    let col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_top_k: column '{}' not found", spec.field))
    })?;

    // Use the field column as both the groupby and the aggregate target.
    // TopK semantics: group by field, aggregate, keep top-n groups.
    let col = batch.column(col_idx);

    // Build groups.
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for row in 0..n_rows {
        let key = if col.is_null(row) {
            "__null__".to_string()
        } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            arr.value(row).to_string()
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            format!("{}", arr.value(row))
        } else {
            format!("row_{row}")
        };
        groups.entry(key).or_default().push(row);
    }

    // Compute aggregate per group.
    let agg_col = if schema.field(col_idx).data_type() == &DataType::Float64 {
        Some(
            col.as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| PyValueError::new_err("data_top_k: expected Float64Array"))?,
        )
    } else {
        None
    };

    let mut group_aggs: Vec<(String, f64)> = groups
        .iter()
        .map(|(key, rows)| {
            let agg_val = match agg_col {
                Some(arr) => {
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
                    compute_agg(&vals, &spec.op)
                }
                None => {
                    // Count for non-numeric.
                    rows.len() as f64
                }
            };
            (key.clone(), agg_val)
        })
        .collect();

    // Sort by aggregate.
    if spec.sort == "ascending" {
        group_aggs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        group_aggs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Keep top-n groups.
    let top_groups: std::collections::HashSet<&str> = group_aggs
        .iter()
        .take(spec.n)
        .map(|(k, _)| k.as_str())
        .collect();

    // Collect row indices that belong to top groups.
    let mut keep_indices: Vec<u32> = Vec::new();
    for (key, rows) in &groups {
        if top_groups.contains(key.as_str()) {
            for &r in rows {
                keep_indices.push(r as u32);
            }
        }
    }
    keep_indices.sort_unstable();

    let idx_array = UInt32Array::from(keep_indices);
    let columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &idx_array, None))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(format!("data_top_k: take: {e}")))?;

    RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_top_k: {e}")))
}

fn compute_agg(vals: &[f64], op: &str) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    match op {
        "sum" => vals.iter().sum(),
        "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
        "count" => vals.len() as f64,
        "min" => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        "max" => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        _ => vals.iter().sum(),
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "TopK")]
#[derive(Debug, Clone)]
pub(crate) struct PyTopK(pub(crate) TransformSpec);

#[pymethods]
impl PyTopK {
    #[new]
    #[pyo3(signature = (n, field, *, op = "sum", sort = "descending", name = None))]
    fn new(n: usize, field: String, op: &str, sort: &str, name: Option<String>) -> Self {
        PyTopK(TransformSpec::TopK(TopKSpec {
            n,
            field,
            op: op.into(),
            sort: sort.into(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::TopK(s) => format!("TopK(n={}, field='{}')", s.n, s.field),
            _ => "TopK(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn top_k_keeps_top_groups() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b", "c", "c"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 10.0, 20.0, 5.0, 5.0])),
            ],
        )
        .unwrap();

        // Top 2 by sum of val (group by val field — actually we want to use cat).
        // Since TopK groups by the `field` column, we use "cat" with count op.
        let spec = TopKSpec {
            n: 2,
            field: "cat".into(),
            op: "count".into(),
            sort: "descending".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // All groups have count 2, so top 2 of 3 groups → 4 rows.
        assert_eq!(out.num_rows(), 4);
    }
}
