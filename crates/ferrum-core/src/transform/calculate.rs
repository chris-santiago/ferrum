//! Data transform: Calculate — add a derived column via expression evaluator.
//!
//! Evaluates an expression per row. A new `as_field` appends the result as a
//! column; an existing `as_field` is overwritten in place (same schema
//! position, the expression's values and dtype), so a batch never holds two
//! columns of one name.

use arrow::array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::expr::{Expr, ExprValue};
use crate::transform::filter::{batch_field_value, ensure_expr_fields_readable};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CalculateSpec {
    /// Output column name.
    pub as_field: String,
    /// Expression string (e.g. "datum.x * 2 + datum.y").
    pub expr: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &CalculateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let parsed = Expr::parse(&spec.expr).map_err(|e| {
        PyValueError::new_err(format!("data_calculate: invalid expression: {e}"))
    })?;

    // A bare field reference (`datum.x` or `datum["x"]`, no operators)
    // carries that column's array through unchanged — dtype, nulls and
    // dictionary encoding included — rather than going through the
    // row-by-row `ExprValue` evaluator below, which can only represent
    // Number/Str/Bool/Null and so would collapse an Int64, Date32, Datetime
    // or Dictionary column to Float64 or Utf8. A bare reference to a column
    // the rows lack takes the path below, whose check refuses it like any
    // other expression naming a missing column.
    let bare_ref_idx = parsed
        .as_bare_field_ref()
        .and_then(|field_name| batch.schema().index_of(field_name).ok());
    let (dtype, col): (DataType, ArrayRef) = if let Some(idx) = bare_ref_idx {
        (batch.schema().field(idx).data_type().clone(), batch.column(idx).clone())
    } else {
        ensure_expr_fields_readable(&parsed, &batch.schema(), "data_calculate")?;

        let n_rows = batch.num_rows();

        // Evaluate the parsed expression for each row to determine result type.
        let mut results: Vec<ExprValue> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let val = parsed.eval(&|field_name: &str| batch_field_value(batch, field_name, row));
            results.push(val);
        }

        // Determine output dtype from the results.
        infer_and_build_column(&spec.expr, &results)?
    };

    let old_schema = batch.schema();
    let mut fields: Vec<Field> = old_schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    let field = Field::new(&spec.as_field, dtype, true);
    match old_schema.index_of(&spec.as_field) {
        Ok(existing) => {
            fields[existing] = field;
            columns[existing] = col;
        }
        Err(_) => {
            fields.push(field);
            columns.push(col);
        }
    }
    let new_schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(new_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_calculate: {e}")))
}

/// Infer the Arrow dtype from expression results and build the column array.
///
/// An array-valued result (e.g. a bare `[1, 2, 3]` expression, or a ternary
/// branch that evaluates to an array) has no column representation: per the
/// batch's raise-never-null policy, `[Node]` arrays exist only as
/// `indexof(...)` intermediates, not as materializable values, so this
/// raises a typed error naming the offending expression and row rather than
/// silently emitting a debug-formatted string or an all-NaN column. That
/// guard runs first and unconditionally, which is what makes the
/// `ExprValue::Array` arms in the branches below genuinely unreachable
/// (the same early-return-guarantees-the-match pattern `eval_call` in
/// `expr.rs` uses for its `BinOp::And | BinOp::Or` arm).
fn infer_and_build_column(expr: &str, results: &[ExprValue]) -> PyResult<(DataType, ArrayRef)> {
    if let Some(row) = results.iter().position(|v| matches!(v, ExprValue::Array(_))) {
        return Err(PyValueError::new_err(format!(
            "data_calculate: expression '{expr}' evaluated to an array value at row {row}; \
             arrays are only valid as an indexof(...) argument, not as a calculate column result"
        )));
    }

    // Check if all results are the same type.
    let has_str = results.iter().any(|v| matches!(v, ExprValue::Str(_)));
    let has_bool = results.iter().any(|v| matches!(v, ExprValue::Bool(_)));
    let has_num = results.iter().any(|v| matches!(v, ExprValue::Number(_)));

    let (dtype, col): (DataType, ArrayRef) = if has_str {
        // Coerce everything to string.
        let values: Vec<Option<String>> = results
            .iter()
            .map(|v| match v {
                ExprValue::Str(s) => Some(s.clone()),
                ExprValue::Number(n) => Some(format!("{n}")),
                ExprValue::Bool(b) => Some(b.to_string()),
                ExprValue::Null => None,
                ExprValue::Array(_) => unreachable!("ruled out above"),
            })
            .collect();
        (DataType::Utf8, Arc::new(StringArray::from(values)))
    } else if has_bool && !has_num {
        let values: Vec<Option<bool>> = results
            .iter()
            .map(|v| match v {
                ExprValue::Bool(b) => Some(*b),
                ExprValue::Null => None,
                _ => None,
            })
            .collect();
        (DataType::Boolean, Arc::new(BooleanArray::from(values)))
    } else {
        // Default: Float64.
        let values: Vec<Option<f64>> = results
            .iter()
            .map(|v| match v {
                ExprValue::Number(n) => Some(*n),
                ExprValue::Bool(true) => Some(1.0),
                ExprValue::Bool(false) => Some(0.0),
                ExprValue::Null => None,
                ExprValue::Str(_) => Some(f64::NAN),
                ExprValue::Array(_) => unreachable!("ruled out above"),
            })
            .collect();
        (DataType::Float64, Arc::new(Float64Array::from(values)))
    };
    Ok((dtype, col))
}

// No PyO3 wrapper: `DataCalculate` is constructed only via the dict-emitting
// `transform_calculate` Python function and carried through the
// `transforms_json` serde path (SEAM-02). The empty-`expr` check the removed
// `#[new]` performed is not reachable via the dict path (which only validates
// `as_`); the empty-`as_field` check is covered by `transform_calculate`'s
// `_validate_as_str`. `CalculateSpec` above is the serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn calculate_adds_derived_column() {
        let batch = make_batch();
        let spec = CalculateSpec {
            as_field: "z".into(),
            expr: "datum.x + datum.y".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 3);
        assert_eq!(out.schema().field(2).name(), "z");
        let z = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(z.value(0), 11.0);
        assert_eq!(z.value(1), 22.0);
        assert_eq!(z.value(2), 33.0);
    }

    #[test]
    fn calculate_multiplication() {
        let batch = make_batch();
        let spec = CalculateSpec {
            as_field: "product".into(),
            expr: "datum.x * datum.y".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let p = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(p.value(0), 10.0);
        assert_eq!(p.value(1), 40.0);
        assert_eq!(p.value(2), 90.0);
    }

    #[test]
    fn calculate_preserves_original_columns() {
        let batch = make_batch();
        let spec = CalculateSpec {
            as_field: "z".into(),
            expr: "datum.x * 2".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 3);
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(x.value(0), 1.0);
        assert_eq!(x.value(1), 2.0);
    }

    fn calc(as_field: &str, expr: &str) -> CalculateSpec {
        CalculateSpec { as_field: as_field.into(), expr: expr.into(), name: None }
    }

    /// An existing `as_field` is overwritten at its own schema position, and
    /// a read by name sees the new values. Appending a second `x` would leave
    /// the old values first in name lookup.
    #[test]
    fn an_existing_column_is_replaced_in_place() {
        let out = apply(&calc("x", "datum.x * 100"), &make_batch()).unwrap();
        let schema = out.schema();
        let names: Vec<&String> = schema.fields().iter().map(|f| f.name()).collect();
        assert_eq!(names, ["x", "y"], "one column per name, original order");
        let idx = out.schema().index_of("x").unwrap();
        let x = out.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(x.values().to_vec(), [100.0, 200.0, 300.0]);
        assert_eq!(out.column(1), make_batch().column(1), "other columns are untouched");
    }

    /// The replaced column takes the expression's dtype, not the old one.
    #[test]
    fn a_replaced_column_takes_the_expression_dtype() {
        let out = apply(&calc("y", "datum.x > 1 ? 'big' : 'small'"), &make_batch()).unwrap();
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(1).name(), "y");
        assert_eq!(out.schema().field(1).data_type(), &DataType::Utf8);
        let y = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((y.value(0), y.value(2)), ("small", "big"));
    }

    /// A new name appends exactly the batch the append rule always built.
    #[test]
    fn a_new_name_appends_the_same_batch_as_before() {
        let batch = make_batch();
        let out = apply(&calc("z", "datum.x + datum.y"), &batch).unwrap();
        let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new("z", DataType::Float64, true));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(Float64Array::from(vec![11.0, 22.0, 33.0])));
        let expected = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        assert_eq!(out, expected);
    }

    // ─── Array-valued results raise, never silently null (rust-quality
    // Required Fix 3 / orchestrator adjudication) ──────────────────────────

    #[test]
    fn calculate_bare_array_literal_raises_typed_error() {
        pyo3::Python::initialize();
        let batch = make_batch();
        let spec = CalculateSpec {
            as_field: "arr".into(),
            expr: "[1, 2, 3]".into(),
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("data_calculate:"), "got: {msg}");
        assert!(msg.contains("[1, 2, 3]"), "got: {msg}");
        assert!(msg.contains("array"), "got: {msg}");
    }

    #[test]
    fn calculate_ternary_branch_array_raises_typed_error() {
        // The array can also arrive via a ternary branch rather than being
        // the whole expression — both are reachable from `to_svg`.
        pyo3::Python::initialize();
        let batch = make_batch();
        let spec = CalculateSpec {
            as_field: "arr".into(),
            expr: "datum.x > 1 ? \"hi\" : [1, 2]".into(),
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("data_calculate:"), "got: {msg}");
        assert!(msg.contains("row 0"), "got: {msg}");
    }

    // ─── Bare field reference preserves the column's Arrow type ───────────
    // `Calculate { expr: "datum.<col>" }` (or the bracketed
    // `datum["<col>"]` form) carries the column's array through unchanged —
    // dtype, nulls and dictionary encoding — instead of re-inferring it from
    // the evaluated `ExprValue`s, which can only represent Number/Str/Bool/
    // Null and so collapses Int64 to Float64, Date32/Datetime to Float64,
    // and Dictionary to Utf8. ─────────────────────────────────────────────

    use arrow::array::{Array, Date32Array, DictionaryArray, Int64Array, StringDictionaryBuilder, TimestampMillisecondArray};
    use arrow::datatypes::Int32Type;

    /// A batch with the standard `x`/`y` Float64 pair plus one extra column
    /// of whatever dtype the bare-reference case under test needs.
    fn make_batch_with_extra(name: &str, dtype: DataType, col: ArrayRef) -> RecordBatch {
        let base = make_batch();
        let mut fields: Vec<Field> = base.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new(name, dtype, true));
        let mut columns = base.columns().to_vec();
        columns.push(col);
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    /// Asserts a bare `datum.<name>` reference carries `col` through with
    /// `dtype` unchanged, both appended under a new name and replacing an
    /// existing column.
    fn assert_bare_ref_preserves(name: &str, dtype: DataType, col: ArrayRef) {
        // Formatting a `PyErr`'s `Display` (as an `.unwrap()` panic message
        // would) needs a live interpreter, so every caller through this
        // helper initializes one up front.
        pyo3::Python::initialize();
        let batch = make_batch_with_extra(name, dtype.clone(), col.clone());

        // New name: appended as a fourth column, values and dtype untouched.
        let out = apply(&calc("new_name", &format!("datum.{name}")), &batch).unwrap();
        assert_eq!(out.num_columns(), 4, "dtype {dtype:?}");
        let idx = out.schema().index_of("new_name").unwrap();
        assert_eq!(out.schema().field(idx).data_type(), &dtype, "dtype {dtype:?}");
        assert_eq!(out.column(idx), &col, "dtype {dtype:?}: values must be byte-identical");

        // Replace in place: same position, same dtype, same values.
        let out = apply(&calc(name, &format!("datum.{name}")), &batch).unwrap();
        assert_eq!(out.num_columns(), 3, "dtype {dtype:?}");
        let idx = out.schema().index_of(name).unwrap();
        assert_eq!(out.schema().field(idx).data_type(), &dtype, "dtype {dtype:?}");
        assert_eq!(out.column(idx), &col, "dtype {dtype:?}: values must be byte-identical");
    }

    #[test]
    fn bare_reference_preserves_int64() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(3)]));
        assert_bare_ref_preserves("n", DataType::Int64, col);
    }

    #[test]
    fn bare_reference_preserves_date32() {
        let col: ArrayRef = Arc::new(Date32Array::from(vec![Some(0_i32), None, Some(19_000)]));
        assert_bare_ref_preserves("d", DataType::Date32, col);
    }

    #[test]
    fn bare_reference_preserves_datetime() {
        let dtype = DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None);
        let col: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![
            Some(0_i64),
            None,
            Some(1_700_000_000_000),
        ]));
        assert_bare_ref_preserves("t", dtype, col);
    }

    #[test]
    fn bare_reference_preserves_dictionary() {
        let mut builder = StringDictionaryBuilder::<Int32Type>::new();
        builder.append_value("a");
        builder.append_null();
        builder.append_value("a");
        let dict: DictionaryArray<Int32Type> = builder.finish();
        let dtype = dict.data_type().clone();
        let col: ArrayRef = Arc::new(dict);
        assert_bare_ref_preserves("cat", dtype, col);
    }

    #[test]
    fn bare_reference_preserves_boolean() {
        let col: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
        assert_bare_ref_preserves("flag", DataType::Boolean, col);
    }

    #[test]
    fn bare_reference_preserves_utf8_with_nulls() {
        let col: ArrayRef = Arc::new(StringArray::from(vec![Some("hi"), None, Some("there")]));
        assert_bare_ref_preserves("s", DataType::Utf8, col);
    }

    #[test]
    fn bare_reference_bracketed_form_preserves_dtype() {
        // `datum["name"]` is the other bare-reference shape `as_bare_field_ref`
        // must recognize — same rule, not just the dotted form.
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(5_i64), Some(6), Some(7)]));
        let batch = make_batch_with_extra("full name", DataType::Int64, col.clone());
        let out = apply(&calc("z", "datum[\"full name\"]"), &batch).unwrap();
        let idx = out.schema().index_of("z").unwrap();
        assert_eq!(out.schema().field(idx).data_type(), &DataType::Int64);
        assert_eq!(out.column(idx), &col);
    }

    #[test]
    fn non_bare_expression_still_infers_as_float64() {
        // An operator present (even `+ 0`) takes the existing inference path:
        // an Int64 column still collapses to Float64, unchanged behavior.
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), Some(2), Some(3)]));
        let batch = make_batch_with_extra("n", DataType::Int64, col);
        let out = apply(&calc("z", "datum.n + 0"), &batch).unwrap();
        let idx = out.schema().index_of("z").unwrap();
        assert_eq!(out.schema().field(idx).data_type(), &DataType::Float64);
        let z = out.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(z.values().to_vec(), [1.0, 2.0, 3.0]);
    }

    /// An expression naming a column the rows lack is refused, naming the
    /// column and the columns the rows carry, in every shape: a bare
    /// reference (which would otherwise take the dtype-preserving path's
    /// miss and emit an all-null column), an operator over it, and a
    /// replacement of an existing column. The rejected alternative is the
    /// Vega-style missing-field `Null`, an all-null or all-NaN column that
    /// blanks whatever reads it.
    #[test]
    fn a_missing_column_is_refused_naming_it_and_the_available_columns() {
        pyo3::Python::initialize();
        let batch = make_batch();
        for (as_field, expr) in [
            ("z", "datum.nonexistent"),
            ("z", "datum[\"nonexistent\"]"),
            ("z", "datum.nonexistent * 2"),
            ("x", "datum.x + datum.nonexistent"),
        ] {
            let msg = apply(&calc(as_field, expr), &batch)
                .expect_err(&format!("{expr}: reads a column the rows lack"))
                .to_string();
            assert_eq!(
                msg,
                r#"ValueError: data_calculate: column 'nonexistent' not found; available: ["x", "y"]"#,
                "{expr}"
            );
        }
    }

    /// A column that is present but all null is read as written, on both
    /// the bare-reference path and the evaluator; an expression with no
    /// column reference computes the same way.
    #[test]
    fn an_all_null_column_and_a_literal_expression_still_compute() {
        let batch = make_batch_with_extra(
            "n",
            DataType::Float64,
            Arc::new(Float64Array::from(vec![None, None, None])),
        );
        let column = |expr: &str| {
            let out = apply(&calc("z", expr), &batch).unwrap();
            out.column(out.schema().index_of("z").unwrap()).clone()
        };
        assert_eq!(column("datum.n").null_count(), 3);
        let doubled = column("datum.n * 2");
        assert_eq!(doubled.len(), 3);
        let literal = column("1 + 2");
        let literal = literal.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(literal.values().to_vec(), vec![3.0, 3.0, 3.0]);
    }
}
