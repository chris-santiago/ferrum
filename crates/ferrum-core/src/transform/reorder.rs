//! Reorder transform — apply a permutation to the input batch.
//!
//! `by` names an Int64 index column where output.row[i] == input.row[by[i]].
//! When `drop_index=true` (default), the `by` column is omitted from output.
//!
//! Used by `clustermap()` to reorder rows/columns by Linkage's `order` named output.

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, UInt64Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReorderSpec {
    pub by: String,
    #[serde(default = "default_drop_index")]
    pub drop_index: bool,
    /// Optional name of a sibling transform's named output to read the index
    /// column from (Phase 9 finalize). When set, Reorder uses `<from>` as the
    /// batch source for the `by` column, but applies the permutation to the
    /// chained input batch. Used by clustermap to consume Linkage's
    /// `<name>_order` secondary output across the transform chain.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_drop_index() -> bool { true }

pub(crate) fn apply(spec: &ReorderSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_outputs(spec, batch, None)
}

pub(crate) fn apply_with_outputs(
    spec: &ReorderSpec,
    batch: &RecordBatch,
    named_outputs: Option<&std::collections::HashMap<String, RecordBatch>>,
) -> PyResult<RecordBatch> {
    // Determine which batch to read the index column from.
    let index_batch: &RecordBatch = match (&spec.from, named_outputs) {
        (Some(name), Some(outs)) => outs.get(name).ok_or_else(|| PyValueError::new_err(
            format!("stat_reorder: from='{name}' not found in named outputs; available: {:?}",
                outs.keys().collect::<Vec<_>>())))?,
        (Some(name), None) => return Err(PyValueError::new_err(format!(
            "stat_reorder: from='{name}' set but no transform context available"))),
        (None, _) => batch,
    };
    let schema = index_batch.schema();
    let idx = schema.index_of(&spec.by).map_err(|_| PyValueError::new_err(
        format!("stat_reorder: index column '{}' not found", spec.by)))?;
    if schema.field(idx).data_type() != &DataType::Int64 {
        return Err(PyValueError::new_err(format!(
            "stat_reorder: '{}' must be Int64 (got {:?})",
            spec.by, schema.field(idx).data_type())));
    }
    let idx_arr = index_batch.column(idx).as_any().downcast_ref::<Int64Array>().unwrap();
    let n = idx_arr.len();
    if spec.from.is_some() && n != batch.num_rows() {
        return Err(PyValueError::new_err(format!(
            "stat_reorder: from='{}' produced {} indices but batch has {} rows",
            spec.from.as_deref().unwrap_or(""), n, batch.num_rows())));
    }
    // arrow::compute::take requires UInt64 indices.
    let take_indices: UInt64Array = (0..n)
        .map(|i| {
            let v = idx_arr.value(i);
            if v < 0 || (v as usize) >= batch.num_rows() {
                return Err(PyValueError::new_err(format!(
                    "stat_reorder: index {v} out of bounds (n={})", batch.num_rows())));
            }
            Ok(v as u64)
        })
        .collect::<PyResult<Vec<u64>>>()?
        .into();

    let batch_schema = batch.schema();
    let in_batch_idx = batch_schema.index_of(&spec.by).ok();
    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut out_fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
    for (i, field) in batch_schema.fields().iter().enumerate() {
        if spec.drop_index && in_batch_idx == Some(i) {
            continue;
        }
        let permuted = take(&batch.column(i), &take_indices, None)
            .map_err(|e| PyValueError::new_err(format!("stat_reorder: take: {e}")))?;
        out_cols.push(permuted);
        out_fields.push(field.as_ref().clone());
    }
    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("stat_reorder: {e}")))
}

// PyO3 wrapper — mirror Unpivot pattern.
use crate::transform::core::TransformSpec;

/// Row-reordering transform driven by an integer index column.
///
/// Permutes all rows of the input batch according to the values in the
/// ``by`` column, which must be an integer column containing a permutation
/// of ``0..n-1``. When ``from_`` is set, the batch is consumed from a
/// named sibling transform's output (e.g. a ``Linkage`` that emits
/// ``new_idx``) rather than from the primary data source.
///
/// Parameters
/// ----------
/// by : str
///     Integer column containing the target row index permutation. Must be
///     of type Int64 or UInt64 and contain a valid permutation of
///     ``0..n-1``.
/// drop_index : bool, default True
///     When True, the ``by`` column is removed from the output. Set to
///     False to retain it for debugging.
/// from_ : str, optional
///     Name of a sibling transform whose output supplies this transform's
///     input batch. Used in clustermap pipelines where ``Linkage`` feeds
///     directly into ``Reorder``.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
#[pyclass(eq, module = "ferrum._core", name = "Reorder")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyReorder(pub(crate) TransformSpec);

#[pymethods]
impl PyReorder {
    #[new]
    #[pyo3(signature = (by, *, drop_index = true, from_ = None, name = None))]
    fn new(by: &str, drop_index: bool, from_: Option<String>, name: Option<String>) -> PyResult<Self> {
        if by.is_empty() {
            return Err(PyValueError::new_err("Reorder: by must be non-empty"));
        }
        Ok(PyReorder(TransformSpec::Reorder(ReorderSpec {
            by: by.into(), drop_index, from: from_, name,
        })))
    }
    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Reorder(s) => format!(
                "Reorder(by='{}', drop_index={}, from={:?})",
                s.by, s.drop_index, s.from),
            #[allow(unreachable_patterns)] _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};

    fn make_batch_with_idx(idx: Vec<i64>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("new_idx", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(idx)),
            Arc::new(Float64Array::from(vals)),
        ]).unwrap()
    }

    #[test]
    fn reorder_5_rows_correctness() {
        let batch = make_batch_with_idx(vec![4, 3, 2, 1, 0], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, from: None, name: None };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 1);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((0..5).map(|i| v.value(i)).collect::<Vec<_>>(),
                   vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn reorder_identity_permutation_unchanged() {
        let batch = make_batch_with_idx(vec![0, 1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, from: None, name: None };
        let out = apply(&spec, &batch).unwrap();
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((0..5).map(|i| v.value(i)).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reorder_keeps_index_when_drop_false() {
        let batch = make_batch_with_idx(vec![1, 0], vec![10.0, 20.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: false, from: None, name: None };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
    }

    #[test]
    fn reorder_out_of_bounds_errors() {
        pyo3::Python::initialize();
        let batch = make_batch_with_idx(vec![0, 99], vec![1.0, 2.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, from: None, name: None };
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }
}
