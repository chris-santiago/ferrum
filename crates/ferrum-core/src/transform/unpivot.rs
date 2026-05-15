//! Unpivot transform — wide → long reshape.
//!
//! Output schema:
//!   [id_vars..., var_name: Utf8, value_name: <unified-dtype>]
//!
//! Dtype rule (homogeneous-or-numeric):
//!   - All value columns must share a dtype, OR all be numeric.
//!   - Numeric mixed types widen to the widest (Int32+Float64 → Float64).
//!   - Mixed non-numeric types → error.
//!
//! Used by `heatmap()` (wide-matrix input) and `clustermap()` reshape.

use arrow::array::{Array, ArrayRef, RecordBatch, StringArray, StringBuilder};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct UnpivotSpec {
    #[serde(default)]
    pub id_vars: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value_vars: Option<Vec<String>>,
    #[serde(default = "default_var_name")]
    pub var_name: String,
    #[serde(default = "default_value_name")]
    pub value_name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_var_name() -> String { "variable".into() }
fn default_value_name() -> String { "value".into() }

pub(crate) fn apply(spec: &UnpivotSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    // Resolve value_vars: either explicit, or all non-id columns.
    let value_var_names: Vec<String> = match &spec.value_vars {
        Some(v) => v.clone(),
        None => schema.fields().iter()
            .map(|f| f.name().to_string())
            .filter(|n| !spec.id_vars.contains(n))
            .collect(),
    };

    if value_var_names.is_empty() {
        return Err(PyValueError::new_err(
            "stat_unpivot: no value_vars to melt (id_vars covers all columns)"
        ));
    }

    // Validate dtypes: must be homogeneous OR all-numeric.
    let value_dtypes: Vec<&DataType> = value_var_names.iter()
        .map(|n| {
            let i = schema.index_of(n).map_err(|_| PyValueError::new_err(
                format!("stat_unpivot: column '{n}' not found")
            ))?;
            Ok(schema.field(i).data_type())
        })
        .collect::<PyResult<_>>()?;

    let unified_dtype = unify_value_dtype(&value_dtypes)?;

    // Cast each value column to the unified dtype.
    let value_columns_cast: Vec<ArrayRef> = value_var_names.iter()
        .map(|n| {
            let i = schema.index_of(n).unwrap();
            cast(&batch.column(i), &unified_dtype)
                .map_err(|e| PyValueError::new_err(format!("stat_unpivot: cast '{n}': {e}")))
        })
        .collect::<PyResult<_>>()?;

    // Stack value columns vertically.
    let stacked_value: ArrayRef = {
        let refs: Vec<&dyn Array> = value_columns_cast.iter().map(|a| a.as_ref()).collect();
        concat(&refs).map_err(|e| PyValueError::new_err(format!("stat_unpivot: concat: {e}")))?
    };

    // Build var_name column (Utf8): repeat each name n_rows times in row-major order.
    let mut var_builder = StringBuilder::with_capacity(
        n_rows * value_var_names.len(),
        n_rows * value_var_names.len() * 8,
    );
    for name in &value_var_names {
        for _ in 0..n_rows {
            var_builder.append_value(name);
        }
    }
    let var_arr: ArrayRef = Arc::new(var_builder.finish());

    // Build id columns: take indices [0..n_rows] cycled per value_var.
    let id_columns_replicated: Vec<ArrayRef> = spec.id_vars.iter()
        .map(|n| {
            let i = schema.index_of(n).map_err(|_| PyValueError::new_err(
                format!("stat_unpivot: id_var '{n}' not found")
            ))?;
            // Concat the original id-column with itself k times where k = value_vars.len()
            let one = batch.column(i);
            let repeats: Vec<&dyn Array> = (0..value_var_names.len())
                .map(|_| one.as_ref()).collect();
            concat(&repeats).map_err(|e| PyValueError::new_err(format!("stat_unpivot: id-replicate: {e}")))
        })
        .collect::<PyResult<_>>()?;

    // Assemble output schema: id_vars... + var_name + value_name.
    let mut fields: Vec<Field> = spec.id_vars.iter().map(|n| {
        let i = schema.index_of(n).unwrap();
        let f = schema.field(i);
        Field::new(f.name(), f.data_type().clone(), f.is_nullable())
    }).collect();
    fields.push(Field::new(&spec.var_name, DataType::Utf8, false));
    fields.push(Field::new(&spec.value_name, unified_dtype, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols = id_columns_replicated;
    cols.push(var_arr);
    cols.push(stacked_value);
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_unpivot: {e}")))
}

fn unify_value_dtype(dtypes: &[&DataType]) -> PyResult<DataType> {
    if dtypes.is_empty() {
        return Err(PyValueError::new_err("stat_unpivot: no value columns"));
    }
    // Homogeneous fast path.
    if dtypes.iter().all(|d| *d == dtypes[0]) {
        return Ok(dtypes[0].clone());
    }
    // Mixed: must be all-numeric to widen.
    let all_numeric = dtypes.iter().all(|d| is_numeric(d));
    if !all_numeric {
        let names: Vec<String> = dtypes.iter().map(|d| format!("{d}")).collect();
        return Err(PyValueError::new_err(format!(
            "stat_unpivot: value_vars have heterogeneous non-numeric types: [{}]; \
             cast to a common type before unpivot", names.join(", ")
        )));
    }
    // Widen to Float64 if any float; else widest int. Phase 9 keeps this simple:
    // any mixed-numeric → Float64 (covers all observed cases for heatmap/clustermap).
    Ok(DataType::Float64)
}

fn is_numeric(d: &DataType) -> bool {
    matches!(d,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
        | DataType::Float32 | DataType::Float64)
}

// ---------- PyO3 wrapper ----------

use crate::transform::core::TransformSpec;

/// Melt / wide-to-long reshaping transform.
///
/// Converts a wide-format batch to long format by stacking one or more
/// value columns into ``var_name`` / ``value_name`` column pairs, while
/// preserving ``id_vars`` columns as repeated identifiers. Numeric and
/// string value columns are both supported; all selected value columns
/// must share a compatible Arrow data type.
///
/// Parameters
/// ----------
/// id_vars : list of str, default []
///     Columns to keep as identifier variables (repeated per unpivoted row).
/// value_vars : list of str, optional
///     Columns to unpivot. When omitted, all columns not in ``id_vars``
///     are unpivoted.
/// var_name : str, default "variable"
///     Name of the new column that holds the original column names.
/// value_name : str, default "value"
///     Name of the new column that holds the original values.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_line().encode(
/// ...     x="variable", y="value", color="id",
/// ...     transform=fm.Unpivot(["score_a", "score_b"], variable="variable", value="value"),
/// ... )
#[pyclass(eq, module = "ferrum._core", name = "Unpivot")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyUnpivot(pub(crate) TransformSpec);

#[pymethods]
impl PyUnpivot {
    #[new]
    #[pyo3(signature = (
        *,
        id_vars = Vec::<String>::new(),
        value_vars = None,
        var_name = "variable",
        value_name = "value",
        name = None,
    ))]
    fn new(
        id_vars: Vec<String>,
        value_vars: Option<Vec<String>>,
        var_name: &str,
        value_name: &str,
        name: Option<String>,
    ) -> PyResult<Self> {
        if var_name.is_empty() || value_name.is_empty() {
            return Err(PyValueError::new_err("Unpivot: var_name and value_name must be non-empty"));
        }
        Ok(PyUnpivot(TransformSpec::Unpivot(UnpivotSpec {
            id_vars, value_vars, var_name: var_name.into(), value_name: value_name.into(), name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Unpivot(s) => format!(
                "Unpivot(id_vars={:?}, value_vars={:?}, var_name='{}', value_name='{}')",
                s.id_vars, s.value_vars, s.var_name, s.value_name,
            ),
            #[allow(unreachable_patterns)] _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array};

    fn batch_3x4() -> RecordBatch {
        // 3 rows × 4 numeric value columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::Int32, false),
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
            Field::new("d", DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![7.0, 8.0, 9.0])),
            Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0])),
        ]).unwrap()
    }

    #[test]
    fn unpivot_3x4_numeric_correctness() {
        let batch = batch_3x4();
        let spec = UnpivotSpec {
            id_vars: vec!["row_id".into()],
            value_vars: None,
            var_name: "variable".into(),
            value_name: "value".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 12);  // 3 rows × 4 value cols
        assert_eq!(out.num_columns(), 3); // row_id, variable, value
        let vars = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let vals = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        // First 3 rows should be variable="a", value=1,2,3 (the column order).
        assert_eq!(vars.value(0), "a"); assert_eq!(vals.value(0), 1.0);
        assert_eq!(vars.value(1), "a"); assert_eq!(vals.value(1), 2.0);
        assert_eq!(vars.value(3), "b"); assert_eq!(vals.value(3), 4.0);
        assert_eq!(vars.value(11), "d"); assert_eq!(vals.value(11), 12.0);
    }

    #[test]
    fn unpivot_widens_int_and_float_to_float64() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Float64Array::from(vec![3.5, 4.5])),
        ]).unwrap();
        let spec = UnpivotSpec {
            id_vars: vec![],
            value_vars: None,
            var_name: "k".into(),
            value_name: "v".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
        let vals = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(2), 3.5);
    }

    #[test]
    fn unpivot_homogeneous_utf8_works() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(StringArray::from(vec!["p", "q"])),
        ]).unwrap();
        let spec = UnpivotSpec {
            id_vars: vec![], value_vars: None,
            var_name: "k".into(), value_name: "v".into(), name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(1).data_type(), &DataType::Utf8);
    }

    #[test]
    fn unpivot_mixed_int_and_utf8_errors() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["x"])),
        ]).unwrap();
        let spec = UnpivotSpec {
            id_vars: vec![], value_vars: None,
            var_name: "k".into(), value_name: "v".into(), name: None,
        };
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(err.contains("heterogeneous non-numeric"), "got: {err}");
    }

    #[test]
    fn unpivot_preserves_id_dtypes() {
        let batch = batch_3x4();
        let spec = UnpivotSpec {
            id_vars: vec!["row_id".into()],
            value_vars: None,
            var_name: "k".into(), value_name: "v".into(), name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int32);
    }
}
