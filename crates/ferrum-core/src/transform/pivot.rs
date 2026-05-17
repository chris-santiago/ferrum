//! Data transform: Pivot — long to wide.
//!
//! Groups by `groupby` columns, then for each unique value of `field`,
//! creates a new column containing the aggregated `value` column.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct PivotSpec {
    /// Column whose unique values become new column headers.
    pub field: String,
    /// Column whose values fill the pivoted cells.
    pub value: String,
    /// Columns to group by. If None, each row is its own group.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Maximum number of pivot columns to create.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub limit: Option<usize>,
    /// Aggregation operation when multiple values map to same cell.
    #[serde(default = "default_op")]
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_op() -> String {
    "sum".into()
}

pub(crate) fn apply(spec: &PivotSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    // Validate field column (must be Utf8).
    let field_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_pivot: column '{}' not found", spec.field))
    })?;
    let field_col = batch
        .column(field_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_pivot: column '{}' must be Utf8",
                spec.field
            ))
        })?;

    // Validate value column (must be Float64).
    let value_idx = schema.index_of(&spec.value).map_err(|_| {
        PyValueError::new_err(format!("data_pivot: column '{}' not found", spec.value))
    })?;
    let value_col = batch
        .column(value_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_pivot: column '{}' must be Float64",
                spec.value
            ))
        })?;

    // Resolve groupby columns.
    let groupby = spec.groupby.clone().unwrap_or_default();
    let group_indices: Vec<usize> = groupby
        .iter()
        .map(|g| {
            schema.index_of(g).map_err(|_| {
                PyValueError::new_err(format!("data_pivot: groupby column '{g}' not found"))
            })
        })
        .collect::<PyResult<_>>()?;

    // Collect unique pivot values (ordered by first appearance).
    let mut pivot_values: Vec<String> = Vec::new();
    let mut pivot_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..n_rows {
        if field_col.is_null(i) {
            continue;
        }
        let v = field_col.value(i).to_string();
        if pivot_set.insert(v.clone()) {
            pivot_values.push(v);
        }
    }

    // Apply limit.
    if let Some(limit) = spec.limit {
        pivot_values.truncate(limit);
    }

    // Build group key → { pivot_value → Vec<f64> } map.
    // Group key is a Vec of string representations for simplicity.
    let mut groups: BTreeMap<Vec<String>, BTreeMap<String, Vec<f64>>> = BTreeMap::new();

    for row in 0..n_rows {
        if field_col.is_null(row) {
            continue;
        }
        let pv = field_col.value(row).to_string();
        if !pivot_set.contains(&pv) {
            continue; // Pruned by limit.
        }

        let key: Vec<String> = group_indices
            .iter()
            .map(|&gi| extract_string_key(batch.column(gi).as_ref(), row))
            .collect();

        let val = if value_col.is_null(row) {
            f64::NAN
        } else {
            value_col.value(row)
        };

        groups
            .entry(key)
            .or_default()
            .entry(pv)
            .or_default()
            .push(val);
    }

    // Build output: groupby columns + one Float64 column per pivot value.
    let n_out_rows = groups.len();

    // Groupby output columns.
    let mut out_fields: Vec<Field> = groupby
        .iter()
        .enumerate()
        .map(|(i, name)| {
            Field::new(name, schema.field(group_indices[i]).data_type().clone(), true)
        })
        .collect();
    for pv in &pivot_values {
        out_fields.push(Field::new(pv, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(out_fields));

    // Fill columns.
    let mut group_key_vecs: Vec<Vec<String>> = Vec::with_capacity(n_out_rows);
    let mut pivot_data: Vec<Vec<f64>> = vec![Vec::with_capacity(n_out_rows); pivot_values.len()];

    for (key, pivot_map) in &groups {
        group_key_vecs.push(key.clone());
        for (pi, pv) in pivot_values.iter().enumerate() {
            let agg_val = match pivot_map.get(pv) {
                Some(vals) => aggregate_values(vals, &spec.op),
                None => f64::NAN,
            };
            pivot_data[pi].push(agg_val);
        }
    }

    // Build Arrow arrays.
    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(groupby.len() + pivot_values.len());
    for (gi, _) in groupby.iter().enumerate() {
        let vals: Vec<String> = group_key_vecs.iter().map(|k| k[gi].clone()).collect();
        // Emit as Utf8 for simplicity (groupby keys).
        out_cols.push(Arc::new(StringArray::from(vals)));
    }
    for pv_data in pivot_data {
        out_cols.push(Arc::new(Float64Array::from(pv_data)));
    }

    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_pivot: {e}")))
}

fn extract_string_key(col: &dyn Array, row: usize) -> String {
    if col.is_null(row) {
        return String::new();
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return arr.value(row).to_string();
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return format!("{}", arr.value(row));
    }
    String::new()
}

fn aggregate_values(vals: &[f64], op: &str) -> f64 {
    let clean: Vec<f64> = vals.iter().copied().filter(|v| !v.is_nan()).collect();
    if clean.is_empty() {
        return f64::NAN;
    }
    match op {
        "sum" => clean.iter().sum(),
        "mean" => clean.iter().sum::<f64>() / clean.len() as f64,
        "count" => clean.len() as f64,
        "min" => clean.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        "max" => clean.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        "first" => clean[0],
        "last" => *clean.last().unwrap(),
        _ => clean.iter().sum(), // Default to sum.
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DataPivot")]
#[derive(Debug, Clone)]
pub(crate) struct PyDataPivot(pub(crate) TransformSpec);

#[pymethods]
impl PyDataPivot {
    #[new]
    #[pyo3(signature = (field, value, *, groupby = None, limit = None, op = "sum", name = None))]
    fn new(
        field: String,
        value: String,
        groupby: Option<Vec<String>>,
        limit: Option<usize>,
        op: &str,
        name: Option<String>,
    ) -> PyResult<Self> {
        Ok(PyDataPivot(TransformSpec::Pivot(PivotSpec {
            field,
            value,
            groupby,
            limit,
            op: op.into(),
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Pivot(s) => format!("DataPivot(field='{}', value='{}')", s.field, s.value),
            _ => "DataPivot(?)".to_string(),
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
    fn pivot_basic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap();

        let spec = PivotSpec {
            field: "key".into(),
            value: "val".into(),
            groupby: Some(vec!["group".into()]),
            limit: None,
            op: "sum".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        // Columns: group, x, y
        assert_eq!(out.num_columns(), 3);
    }
}
