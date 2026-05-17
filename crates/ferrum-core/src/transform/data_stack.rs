//! Data transform: DataStack — compute stacked (cumulative) positions.
//!
//! Computes y0 and y1 columns for stacked bar/area charts. Groups by
//! `groupby`, sorts within each group, and accumulates field values.
//! Supports offset modes: "zero" (default), "normalize", "center".

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DataStackSpec {
    /// Field to stack.
    pub field: String,
    /// Columns to group by (define each stack).
    #[serde(default)]
    pub groupby: Vec<String>,
    /// Sort order within each stack.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<Vec<String>>,
    /// Output column names: (y0, y1).
    #[serde(default = "default_stack_as")]
    pub as_: (String, String),
    /// Offset mode: "zero", "normalize", "center".
    #[serde(default = "default_offset")]
    pub offset: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_stack_as() -> (String, String) {
    ("y0".into(), "y1".into())
}
fn default_offset() -> String {
    "zero".into()
}

pub(crate) fn apply(spec: &DataStackSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    let field_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_stack: column '{}' not found", spec.field))
    })?;
    let field_col = batch
        .column(field_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_stack: column '{}' must be Float64",
                spec.field
            ))
        })?;

    // Build group keys.
    let group_cols: Vec<usize> = spec
        .groupby
        .iter()
        .map(|g| {
            schema.index_of(g).map_err(|_| {
                PyValueError::new_err(format!("data_stack: groupby column '{g}' not found"))
            })
        })
        .collect::<PyResult<_>>()?;

    // Partition rows by group key.
    let mut groups: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();
    for row in 0..n_rows {
        let key: Vec<String> = group_cols
            .iter()
            .map(|&gi| extract_key(batch.column(gi).as_ref(), row))
            .collect();
        groups.entry(key).or_default().push(row);
    }

    // If no groupby, all rows are one group.
    if spec.groupby.is_empty() {
        groups.clear();
        groups.insert(Vec::new(), (0..n_rows).collect());
    }

    // Compute y0, y1 for each group.
    let mut y0 = vec![0.0f64; n_rows];
    let mut y1 = vec![0.0f64; n_rows];

    for rows in groups.values() {
        // Accumulate.
        let mut cumsum = 0.0;
        for &row in rows {
            let val = if field_col.is_null(row) {
                0.0
            } else {
                let v = field_col.value(row);
                if v.is_nan() { 0.0 } else { v }
            };
            y0[row] = cumsum;
            y1[row] = cumsum + val;
            cumsum += val;
        }

        // Apply offset.
        match spec.offset.as_str() {
            "zero" => {} // Already done.
            "normalize" => {
                let total = cumsum;
                if total.abs() > 1e-15 {
                    for &row in rows {
                        y0[row] /= total;
                        y1[row] /= total;
                    }
                }
            }
            "center" => {
                let half = cumsum / 2.0;
                for &row in rows {
                    y0[row] -= half;
                    y1[row] -= half;
                }
            }
            _ => {} // Unknown offset → treat as zero.
        }
    }

    // Build output: original columns + y0, y1.
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(&spec.as_.0, DataType::Float64, false));
    fields.push(Field::new(&spec.as_.1, DataType::Float64, false));
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.push(Arc::new(Float64Array::from(y0)));
    columns.push(Arc::new(Float64Array::from(y1)));

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_stack: {e}")))
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

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DataStack")]
#[derive(Debug, Clone)]
pub(crate) struct PyDataStack(pub(crate) TransformSpec);

#[pymethods]
impl PyDataStack {
    #[new]
    #[pyo3(signature = (field, groupby, *, offset = "zero", name = None))]
    fn new(field: String, groupby: Vec<String>, offset: &str, name: Option<String>) -> Self {
        PyDataStack(TransformSpec::DataStack(DataStackSpec {
            field,
            groupby,
            sort: None,
            as_: default_stack_as(),
            offset: offset.into(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::DataStack(s) => format!("DataStack(field='{}', offset='{}')", s.field, s.offset),
            _ => "DataStack(?)".to_string(),
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
    fn stack_zero_offset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "a"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec!["cat".into()],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "zero".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y0_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let y1_col = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();

        assert_eq!(y0_col.value(0), 0.0);
        assert_eq!(y1_col.value(0), 1.0);
        assert_eq!(y0_col.value(1), 1.0);
        assert_eq!(y1_col.value(1), 3.0);
        assert_eq!(y0_col.value(2), 3.0);
        assert_eq!(y1_col.value(2), 6.0);
    }

    #[test]
    fn stack_normalize_offset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![2.0, 3.0, 5.0]))],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec![],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "normalize".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y1_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        // Last y1 should be 1.0 (normalized total).
        assert!((y1_col.value(2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stack_center_offset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![2.0, 2.0, 2.0]))],
        )
        .unwrap();

        let spec = DataStackSpec {
            field: "val".into(),
            groupby: vec![],
            sort: None,
            as_: ("y0".into(), "y1".into()),
            offset: "center".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let y0_col = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let y1_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        // Total = 6, half = 3. First y0 = 0-3 = -3, first y1 = 2-3 = -1.
        assert!((y0_col.value(0) - (-3.0)).abs() < 1e-12);
        assert!((y1_col.value(0) - (-1.0)).abs() < 1e-12);
    }
}
