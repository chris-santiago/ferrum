//! Data transform: Flatten — expand list/array columns into separate rows.
//!
//! For each row, if the specified list columns contain N elements, the row
//! is replicated N times with the list elements distributed across rows.
//! Non-list columns are repeated.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlattenSpec {
    /// Column names to flatten (must be list/array typed or semi-colon strings).
    pub fields: Vec<String>,
    /// Optional output names for flattened columns.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub as_: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &FlattenSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    if spec.fields.is_empty() {
        return Err(PyValueError::new_err("data_flatten: fields must be non-empty"));
    }

    // Validate fields exist.
    let field_indices: Vec<usize> = spec
        .fields
        .iter()
        .map(|f| {
            schema.index_of(f).map_err(|_| {
                PyValueError::new_err(format!("data_flatten: column '{f}' not found"))
            })
        })
        .collect::<PyResult<_>>()?;

    // For Arrow list columns, expand; for string columns with delimiters, split.
    // For now, handle ListArray(Float64) and Utf8 (semicolon-delimited strings).
    // Determine the number of elements per row from the first flatten field.
    let first_col = batch.column(field_indices[0]);
    let lengths = compute_row_lengths(first_col.as_ref(), n_rows)?;

    // Build expanded output for each column.
    let non_flatten_indices: Vec<usize> = (0..schema.fields().len())
        .filter(|i| !field_indices.contains(i))
        .collect();

    let total_out_rows: usize = lengths.iter().sum();

    // Expand non-flatten columns by repeating.
    let mut out_cols: Vec<ArrayRef> = Vec::new();
    let mut out_fields: Vec<Field> = Vec::new();

    for &i in &non_flatten_indices {
        let col = batch.column(i);
        let expanded = repeat_by_lengths(col.as_ref(), &lengths, total_out_rows)?;
        out_cols.push(expanded);
        out_fields.push(schema.field(i).as_ref().clone());
    }

    // Expand flatten columns by extracting elements.
    let as_names = spec.as_.clone().unwrap_or_else(|| spec.fields.clone());
    for (fi, &col_idx) in field_indices.iter().enumerate() {
        let col = batch.column(col_idx);
        let (expanded, dtype) = flatten_column(col.as_ref(), &lengths, total_out_rows)?;
        let col_name = as_names.get(fi).unwrap_or(&spec.fields[fi]);
        out_cols.push(expanded);
        out_fields.push(Field::new(col_name, dtype, true));
    }

    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_flatten: {e}")))
}

/// Compute the number of elements per row for a list or delimited column.
fn compute_row_lengths(col: &dyn Array, n_rows: usize) -> PyResult<Vec<usize>> {
    // Try ListArray<Float64>.
    if let Some(list) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
        let mut lengths = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            if list.is_null(i) {
                lengths.push(0);
            } else {
                lengths.push(list.value_length(i) as usize);
            }
        }
        return Ok(lengths);
    }
    // Try Utf8 with semicolon delimiter.
    if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
        let mut lengths = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            if str_arr.is_null(i) || str_arr.value(i).is_empty() {
                lengths.push(1); // Keep the row with null/empty.
            } else {
                lengths.push(str_arr.value(i).split(';').count());
            }
        }
        return Ok(lengths);
    }
    // Non-list, non-string: treat as scalar (length 1 per row).
    Ok(vec![1; n_rows])
}

/// Repeat each row element according to lengths.
fn repeat_by_lengths(col: &dyn Array, lengths: &[usize], total: usize) -> PyResult<ArrayRef> {
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        let mut out = Vec::with_capacity(total);
        for (i, &len) in lengths.iter().enumerate() {
            let v = if arr.is_null(i) { f64::NAN } else { arr.value(i) };
            for _ in 0..len {
                out.push(v);
            }
        }
        return Ok(Arc::new(Float64Array::from(out)));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        let mut out: Vec<Option<String>> = Vec::with_capacity(total);
        for (i, &len) in lengths.iter().enumerate() {
            let v = if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i).to_string())
            };
            for _ in 0..len {
                out.push(v.clone());
            }
        }
        return Ok(Arc::new(StringArray::from(out)));
    }
    // Fallback: null array.
    let out: Vec<Option<f64>> = vec![None; total];
    Ok(Arc::new(Float64Array::from(out)))
}

/// Flatten a column: extract list elements or split strings.
fn flatten_column(
    col: &dyn Array,
    lengths: &[usize],
    total: usize,
) -> PyResult<(ArrayRef, DataType)> {
    // Try ListArray.
    if let Some(list) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
        let values = list.values();
        if let Some(float_vals) = values.as_any().downcast_ref::<Float64Array>() {
            let mut out = Vec::with_capacity(total);
            for i in 0..list.len() {
                if list.is_null(i) {
                    // Skip (length is 0 from compute_row_lengths).
                } else {
                    let start = list.value_offsets()[i] as usize;
                    let end = list.value_offsets()[i + 1] as usize;
                    for j in start..end {
                        out.push(if float_vals.is_null(j) {
                            f64::NAN
                        } else {
                            float_vals.value(j)
                        });
                    }
                }
            }
            return Ok((Arc::new(Float64Array::from(out)), DataType::Float64));
        }
    }
    // Try Utf8 split.
    if let Some(str_arr) = col.as_any().downcast_ref::<StringArray>() {
        let mut out: Vec<Option<String>> = Vec::with_capacity(total);
        for (i, &_len) in lengths.iter().enumerate() {
            if str_arr.is_null(i) || str_arr.value(i).is_empty() {
                out.push(None);
            } else {
                for part in str_arr.value(i).split(';') {
                    out.push(Some(part.trim().to_string()));
                }
            }
        }
        return Ok((Arc::new(StringArray::from(out)), DataType::Utf8));
    }
    // Fallback.
    let out: Vec<Option<f64>> = vec![None; total];
    Ok((Arc::new(Float64Array::from(out)), DataType::Float64))
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "Flatten")]
#[derive(Debug, Clone)]
pub(crate) struct PyFlatten(pub(crate) TransformSpec);

#[pymethods]
impl PyFlatten {
    #[new]
    #[pyo3(signature = (fields, *, as_ = None, name = None))]
    fn new(fields: Vec<String>, as_: Option<Vec<String>>, name: Option<String>) -> PyResult<Self> {
        if fields.is_empty() {
            return Err(PyValueError::new_err("Flatten: fields must be non-empty"));
        }
        Ok(PyFlatten(TransformSpec::Flatten(FlattenSpec { fields, as_, name })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Flatten(s) => format!("Flatten(fields={:?})", s.fields),
            _ => "Flatten(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, Float64Array};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn flatten_semicolon_delimited_strings() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, false),
            Field::new("tags", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["a;b;c", "d;e"])),
            ],
        )
        .unwrap();

        let spec = FlattenSpec {
            fields: vec!["tags".into()],
            as_: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // Row 0 expands to 3, row 1 to 2 → 5 total rows.
        assert_eq!(out.num_rows(), 5);

        let ids = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ids.value(0), 1.0);
        assert_eq!(ids.value(1), 1.0);
        assert_eq!(ids.value(2), 1.0);
        assert_eq!(ids.value(3), 2.0);
        assert_eq!(ids.value(4), 2.0);

        let tags = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(tags.value(0), "a");
        assert_eq!(tags.value(1), "b");
        assert_eq!(tags.value(2), "c");
        assert_eq!(tags.value(3), "d");
        assert_eq!(tags.value(4), "e");
    }
}
