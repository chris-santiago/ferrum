//! Data transform: DataAggregate — group-by aggregation (data transform variant).
//!
//! Same semantics as stat_aggregate but exposed as a data transform.
//! Delegates to the existing aggregate module's logic.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::aggregate::AggFn;
use crate::transform::group_key::{
    groupby_field_nullable, groupby_key_at, is_groupby_supported_dtype, materialize_groupby_col,
    KeyValue,
};
use crate::transform::join_aggregate::AggSpec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

    // Validate groupby dtypes and capture them for shared key extraction (FA-7).
    let group_dtypes: Vec<DataType> = group_col_indices
        .iter()
        .zip(groupby.iter())
        .map(|(&gi, g)| {
            let dt = schema.field(gi).data_type().clone();
            if !is_groupby_supported_dtype(&dt) {
                return Err(PyValueError::new_err(format!(
                    "data_aggregate: groupby column '{g}' has unsupported dtype {dt:?}; \
                     supported: Utf8/LargeUtf8, Float64/Float32, \
                     Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
                )));
            }
            Ok(dt)
        })
        .collect::<PyResult<_>>()?;

    let mut groups: BTreeMap<Vec<KeyValue>, Vec<usize>> = BTreeMap::new();
    for row in 0..n_rows {
        let mut key = Vec::with_capacity(group_col_indices.len());
        for (gpos, &gi) in group_col_indices.iter().enumerate() {
            let kv = groupby_key_at(batch.column(gi).as_ref(), &group_dtypes[gpos], row)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "data_aggregate: internal error extracting groupby key at row {row}"
                    ))
                })?;
            key.push(kv);
        }
        groups.entry(key).or_default().push(row);
    }

    if groupby.is_empty() {
        groups.clear();
        groups.insert(Vec::new(), (0..n_rows).collect());
    }

    let n_out_rows = groups.len();

    // Compute aggregates per group.
    let mut group_key_vecs: Vec<Vec<KeyValue>> = Vec::with_capacity(n_out_rows);
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

    // Build output schema: groupby columns (original dtype, nullable for FA-9
    // null keys) + aggregate columns (Float64).
    let mut fields: Vec<Field> = Vec::new();
    for (gi, g) in groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), groupby_field_nullable()));
    }
    for agg in &spec.aggregates {
        fields.push(Field::new(&agg.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    // Build columns. The groupby output array dtype must match the declared
    // schema field; materialize_groupby_col guarantees this (the previous code
    // declared the original dtype but emitted a StringArray).
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (gi, dt) in group_dtypes.iter().enumerate() {
        let col = materialize_groupby_col(&group_key_vecs, gi, dt)
            .map_err(PyValueError::new_err)?;
        columns.push(col);
    }

    for agg_vals in agg_results {
        columns.push(Arc::new(Float64Array::from(agg_vals)));
    }

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_aggregate: {e}")))
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

        use crate::transform::aggregate::aggregate as agg_fn;
        agg_fn(&vals, fn_, rows.len())
    } else {
        // Count on non-numeric.
        if matches!(fn_, AggFn::Count) {
            rows.iter().filter(|&&r| !col.is_null(r)).count() as f64
        } else {
            f64::NAN
        }
    }
}

// No PyO3 wrapper: `DataAggregate` is constructed only via the dict-emitting
// `transform_aggregate` Python function and carried through the
// `transforms_json` serde path (SEAM-02). `DataAggregateSpec` above is the
// serde target.

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
