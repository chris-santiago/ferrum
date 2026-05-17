//! Data transform: DataAggregate — group-by aggregation (data transform variant).
//!
//! Same semantics as stat_aggregate but exposed as a data transform.
//! Delegates to the existing aggregate module's logic.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::aggregate::AggFn;
use crate::transform::join_aggregate::AggSpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DataAggregateSpec {
    pub aggregates: Vec<AggSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &DataAggregateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();
    let groupby = spec.groupby.clone().unwrap_or_default();

    if spec.aggregates.is_empty() {
        return Err(PyValueError::new_err(
            "data_aggregate: aggregates must be non-empty",
        ));
    }

    // Validate aggregate fields.
    for agg in &spec.aggregates {
        schema.index_of(&agg.field).map_err(|_| {
            PyValueError::new_err(format!(
                "data_aggregate: column '{}' not found",
                agg.field
            ))
        })?;
    }

    // Build group keys.
    let group_col_indices: Vec<usize> = groupby
        .iter()
        .map(|g| {
            schema.index_of(g).map_err(|_| {
                PyValueError::new_err(format!(
                    "data_aggregate: groupby column '{g}' not found"
                ))
            })
        })
        .collect::<PyResult<_>>()?;

    let mut groups: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();
    for row in 0..n_rows {
        let key: Vec<String> = group_col_indices
            .iter()
            .map(|&gi| extract_key(batch.column(gi).as_ref(), row))
            .collect();
        groups.entry(key).or_default().push(row);
    }

    if groupby.is_empty() {
        groups.clear();
        groups.insert(Vec::new(), (0..n_rows).collect());
    }

    let n_out_rows = groups.len();

    // Compute aggregates per group.
    let mut group_key_vecs: Vec<Vec<String>> = Vec::with_capacity(n_out_rows);
    let mut agg_results: Vec<Vec<f64>> = vec![Vec::with_capacity(n_out_rows); spec.aggregates.len()];

    for (key, rows) in &groups {
        group_key_vecs.push(key.clone());
        for (ai, agg) in spec.aggregates.iter().enumerate() {
            let col_idx = schema.index_of(&agg.field).map_err(|_| {
                PyValueError::new_err(format!(
                    "data_aggregate: column '{}' not found",
                    agg.field
                ))
            })?;
            let col = batch.column(col_idx);
            let val = compute_agg(col.as_ref(), rows, agg.fn_);
            agg_results[ai].push(val);
        }
    }

    // Build output schema: groupby columns (Utf8) + aggregate columns (Float64).
    let mut fields: Vec<Field> = Vec::new();
    for (gi, g) in groupby.iter().enumerate() {
        let dtype = schema.field(group_col_indices[gi]).data_type().clone();
        fields.push(Field::new(g, dtype, true));
    }
    for agg in &spec.aggregates {
        fields.push(Field::new(&agg.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    // Build columns.
    let mut columns: Vec<ArrayRef> = Vec::new();

    for (gi, _) in groupby.iter().enumerate() {
        let dtype = schema.field(group_col_indices[gi]).data_type();
        match dtype {
            DataType::Utf8 => {
                let vals: Vec<String> = group_key_vecs.iter().map(|k| k[gi].clone()).collect();
                columns.push(Arc::new(StringArray::from(vals)));
            }
            DataType::Float64 => {
                let vals: Vec<f64> = group_key_vecs
                    .iter()
                    .map(|k| k[gi].parse::<f64>().unwrap_or(f64::NAN))
                    .collect();
                columns.push(Arc::new(Float64Array::from(vals)));
            }
            _ => {
                // Fallback to Utf8.
                let vals: Vec<String> = group_key_vecs.iter().map(|k| k[gi].clone()).collect();
                columns.push(Arc::new(StringArray::from(vals)));
            }
        }
    }

    for agg_vals in agg_results {
        columns.push(Arc::new(Float64Array::from(agg_vals)));
    }

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_aggregate: {e}")))
}

fn extract_key(col: &dyn Array, row: usize) -> String {
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

fn compute_agg(col: &dyn Array, rows: &[usize], fn_: AggFn) -> f64 {
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
            return if matches!(fn_, AggFn::Count) {
                0.0
            } else {
                f64::NAN
            };
        }

        match fn_ {
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
        }
    } else {
        // Count on non-numeric.
        if matches!(fn_, AggFn::Count) {
            rows.iter().filter(|&&r| !col.is_null(r)).count() as f64
        } else {
            f64::NAN
        }
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DataAggregate")]
#[derive(Debug, Clone)]
pub(crate) struct PyDataAggregate(pub(crate) TransformSpec);

#[pymethods]
impl PyDataAggregate {
    #[new]
    #[pyo3(signature = (*, groupby = None, name = None))]
    fn new(groupby: Option<Vec<String>>, name: Option<String>) -> Self {
        PyDataAggregate(TransformSpec::DataAggregate(DataAggregateSpec {
            aggregates: Vec::new(),
            groupby,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        "DataAggregate(...)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::aggregate::AggFn;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn data_aggregate_grouped_mean() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 10.0, 20.0])),
            ],
        )
        .unwrap();

        let spec = DataAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Mean,
                as_: "mean_val".into(),
            }],
            groupby: Some(vec!["group".into()]),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.num_columns(), 2); // group + mean_val

        let groups = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let means = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        let a_idx = (0..groups.len()).find(|&i| groups.value(i) == "a").unwrap();
        let b_idx = (0..groups.len()).find(|&i| groups.value(i) == "b").unwrap();
        assert!((means.value(a_idx) - 2.0).abs() < 1e-12);
        assert!((means.value(b_idx) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn data_aggregate_no_groupby() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap();

        let spec = DataAggregateSpec {
            aggregates: vec![AggSpec {
                field: "val".into(),
                fn_: AggFn::Sum,
                as_: "total".into(),
            }],
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let total = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((total.value(0) - 15.0).abs() < 1e-12);
    }
}
