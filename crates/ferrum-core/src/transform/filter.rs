//! Data transform: Filter — row filter using expression evaluator.
//!
//! Evaluates a predicate expression (via `expr.rs`) per row and retains
//! only rows where the expression is truthy.

use arrow::array::{Array, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, TimeUnit};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::expr::{Expr, ExprValue};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FilterSpec {
    pub predicate: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// Crossfilter reference (D6): names a selection parameter whose live extent
    /// filters this layer's rows in WASM. Inert in the static renderer — a param
    /// filter carries `predicate: "true"`, so `apply()` keeps every row (the
    /// correct static semantics for an empty selection).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub param: Option<String>,
}

pub(crate) fn apply(spec: &FilterSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let expr = Expr::parse(&spec.predicate).map_err(|e| {
        PyValueError::new_err(format!("data_filter: invalid predicate: {e}"))
    })?;

    let schema = batch.schema();
    ensure_expr_fields_supported(&expr, &schema, "data_filter")?;

    let n_rows = batch.num_rows();
    let mut keep_indices: Vec<u32> = Vec::with_capacity(n_rows);

    for row in 0..n_rows {
        let result = expr.eval(&|field_name: &str| {
            field_value(batch, &schema, field_name, row)
        });
        if result.is_truthy() {
            keep_indices.push(row as u32);
        }
    }

    let indices = UInt32Array::from(keep_indices);
    let columns: Vec<_> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &indices, None))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(format!("data_filter: take failed: {e}")))?;

    RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_filter: {e}")))
}

/// Dtypes [`field_value`] can represent as an [`ExprValue`].
///
/// Batch-D dtype policy (spec §4.1): every integer / unsigned / float dtype
/// reads as a `Number`, so an expression over a `UInt32` or `Int16` column
/// behaves exactly like the same data as `Int64`. Anything outside this set is
/// refused by [`ensure_expr_fields_supported`] before evaluation starts —
/// before that check existed, an unsupported dtype silently read as `Null` on
/// every row, which made `transform_filter` drop the entire dataset and
/// `transform_calculate` emit an all-null column.
///
/// `expr_dtype_support_matches_field_value` pins this list against
/// `field_value`'s own dispatch, so widening one without the other fails there.
pub(crate) fn expr_dtype_supported(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Boolean
            | DataType::Timestamp(_, _)
    )
}

/// Refuse, before evaluation, any column the expression reads whose dtype
/// [`field_value`] cannot represent.
///
/// A dtype is a schema property, so this runs once per batch rather than per
/// row. Fields the expression names but the schema lacks are left alone —
/// `datum.missing` evaluating to `Null` is Vega's documented behavior, and is
/// unrelated to the dtype policy. `ctx` is the calling transform's error prefix.
pub(crate) fn ensure_expr_fields_supported(
    expr: &Expr,
    schema: &arrow::datatypes::SchemaRef,
    ctx: &str,
) -> PyResult<()> {
    for name in expr.field_refs() {
        let Ok(idx) = schema.index_of(&name) else {
            continue;
        };
        let dtype = schema.field(idx).data_type();
        if !expr_dtype_supported(dtype) {
            return Err(PyValueError::new_err(format!(
                "{ctx}: column '{name}' has dtype {dtype:?}, which expressions cannot read; \
                 supported: any numeric (int/uint/float), Utf8/LargeUtf8, Boolean, Timestamp"
            )));
        }
    }
    Ok(())
}

/// Extract a field value from a row as an ExprValue.
///
/// Callers must have run [`ensure_expr_fields_supported`] first; the trailing
/// arm is unreachable for any column that check admits.
fn field_value(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
    field_name: &str,
    row: usize,
) -> ExprValue {
    let idx = match schema.index_of(field_name) {
        Ok(i) => i,
        Err(_) => return ExprValue::Null,
    };
    let col = batch.column(idx);
    if col.is_null(row) {
        return ExprValue::Null;
    }
    // Read one numeric cell, widening to f64.
    macro_rules! num {
        ($t:ty) => {
            col.as_any()
                .downcast_ref::<$t>()
                .map(|arr| ExprValue::Number(arr.value(row) as f64))
                .unwrap_or(ExprValue::Null)
        };
    }
    // Same, for a temporal tick converted to the milliseconds the evaluator
    // works in. The conversion is spelled out per unit rather than folded into
    // one scale factor: `x / 1_000_000.0` and `x * 1e-6` differ by an ULP, and
    // these values feed byte-compared output.
    macro_rules! millis {
        ($t:ty, |$v:ident| $conv:expr) => {
            col.as_any()
                .downcast_ref::<$t>()
                .map(|arr| {
                    let $v = arr.value(row) as f64;
                    ExprValue::Number($conv)
                })
                .unwrap_or(ExprValue::Null)
        };
    }
    match schema.field(idx).data_type() {
        DataType::Float64 => num!(Float64Array),
        DataType::Float32 => num!(arrow::array::Float32Array),
        DataType::Float16 => col
            .as_any()
            .downcast_ref::<arrow::array::Float16Array>()
            .map(|arr| ExprValue::Number(arr.value(row).to_f64()))
            .unwrap_or(ExprValue::Null),
        DataType::Int8 => num!(arrow::array::Int8Array),
        DataType::Int16 => num!(arrow::array::Int16Array),
        DataType::Int32 => num!(arrow::array::Int32Array),
        DataType::Int64 => num!(arrow::array::Int64Array),
        DataType::UInt8 => num!(arrow::array::UInt8Array),
        DataType::UInt16 => num!(arrow::array::UInt16Array),
        DataType::UInt32 => num!(arrow::array::UInt32Array),
        DataType::UInt64 => num!(arrow::array::UInt64Array),
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|arr| ExprValue::Str(arr.value(row).to_string()))
            .unwrap_or(ExprValue::Null),
        DataType::LargeUtf8 => col
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
            .map(|arr| ExprValue::Str(arr.value(row).to_string()))
            .unwrap_or(ExprValue::Null),
        DataType::Boolean => col
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .map(|arr| ExprValue::Bool(arr.value(row)))
            .unwrap_or(ExprValue::Null),
        // Timestamps read as epoch milliseconds, whatever their stored unit.
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            millis!(arrow::array::TimestampNanosecondArray, |v| v / 1_000_000.0)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            millis!(arrow::array::TimestampMicrosecondArray, |v| v / 1_000.0)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            num!(arrow::array::TimestampMillisecondArray)
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            millis!(arrow::array::TimestampSecondArray, |v| v * 1_000.0)
        }
        _ => ExprValue::Null,
    }
}

/// Shared helper: extract field value from a RecordBatch row.
/// Used by both filter and calculate transforms.
pub(crate) fn batch_field_value(
    batch: &RecordBatch,
    field_name: &str,
    row: usize,
) -> ExprValue {
    field_value(batch, &batch.schema(), field_name, row)
}

// No PyO3 wrapper: `DataFilter` is constructed only via the dict-emitting
// `transform_filter` Python function and carried through the `transforms_json`
// serde path (SEAM-02). `FilterSpec` above is the serde target.

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
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn filter_gt_predicate() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "datum.x > 3".into(),
            name: None,
            param: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let xs = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xs.value(0), 4.0);
        assert_eq!(xs.value(1), 5.0);
    }

    #[test]
    fn filter_compound_predicate() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "datum.x >= 2 && datum.y <= 30".into(),
            name: None,
            param: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let xs = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xs.value(0), 2.0);
        assert_eq!(xs.value(1), 3.0);
    }

    #[test]
    fn filter_all_rows_removed() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "datum.x > 100".into(),
            name: None,
            param: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn filter_with_param_round_trips() {
        let spec = FilterSpec {
            predicate: "true".into(),
            name: None,
            param: Some("brush".into()),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json, r#"{"predicate":"true","param":"brush"}"#);
        let parsed: FilterSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn filter_without_param_omits_key() {
        let spec = FilterSpec {
            predicate: "datum.x > 3".into(),
            name: None,
            param: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json, r#"{"predicate":"datum.x > 3"}"#);
        assert!(!json.contains("param"));
    }

    #[test]
    fn filter_param_is_inert_statically() {
        // A param filter carries `predicate: "true"` → apply keeps all rows.
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "true".into(),
            name: None,
            param: Some("brush".into()),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), batch.num_rows());
    }

    #[test]
    fn filter_invalid_expr_errors() {
        let batch = make_batch();
        let spec = FilterSpec {
            predicate: "invalid!!!".into(),
            name: None,
            param: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("data_filter"));
    }

    // ---- Batch-D dtype policy (spec §4.1) ----

    fn one_col(name: &str, col: arrow::array::ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            name,
            col.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn filter_gt_one(batch: &RecordBatch) -> PyResult<RecordBatch> {
        apply(
            &FilterSpec {
                predicate: "datum.n > 1".into(),
                name: None,
                param: None,
            },
            batch,
        )
    }

    #[test]
    fn filter_reads_every_int_and_uint_dtype() {
        // Each of these silently evaluated to Null before the widening, so the
        // predicate dropped every row.
        let cases: Vec<arrow::array::ArrayRef> = vec![
            Arc::new(arrow::array::Int8Array::from(vec![1_i8, 2, 3])),
            Arc::new(arrow::array::Int16Array::from(vec![1_i16, 2, 3])),
            Arc::new(arrow::array::Int32Array::from(vec![1_i32, 2, 3])),
            Arc::new(arrow::array::Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(arrow::array::UInt8Array::from(vec![1_u8, 2, 3])),
            Arc::new(arrow::array::UInt16Array::from(vec![1_u16, 2, 3])),
            Arc::new(arrow::array::UInt32Array::from(vec![1_u32, 2, 3])),
            Arc::new(arrow::array::UInt64Array::from(vec![1_u64, 2, 3])),
            Arc::new(arrow::array::Float32Array::from(vec![1.0_f32, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ];
        for col in cases {
            let dtype = col.data_type().clone();
            let out = filter_gt_one(&one_col("n", col)).unwrap();
            assert_eq!(out.num_rows(), 2, "dtype {dtype:?} must keep rows 2 and 3");
        }
    }

    #[test]
    fn filter_reads_large_utf8() {
        let col: arrow::array::ArrayRef =
            Arc::new(arrow::array::LargeStringArray::from(vec!["a", "b", "a"]));
        let out = apply(
            &FilterSpec {
                predicate: "datum.s == 'a'".into(),
                name: None,
                param: None,
            },
            &one_col("s", col),
        )
        .unwrap();
        assert_eq!(out.num_rows(), 2);
    }

    #[test]
    fn filter_unsupported_dtype_raises_naming_transform_column_and_dtype() {
        pyo3::Python::initialize();
        let values: arrow::array::ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let list = arrow::array::ListArray::new(
            Arc::new(Field::new("item", DataType::Float64, true)),
            arrow::buffer::OffsetBuffer::new(vec![0_i32, 1, 2].into()),
            values,
            None,
        );
        let msg = filter_gt_one(&one_col("n", Arc::new(list)))
            .unwrap_err()
            .to_string();

        assert!(msg.contains("data_filter"), "err: {msg}");
        assert!(msg.contains("'n'"), "err must name the column: {msg}");
        assert!(msg.contains("List"), "err must name the dtype: {msg}");
    }

    #[test]
    fn filter_missing_field_still_evaluates_to_null() {
        // Vega semantics for `datum.absent` are unchanged by the dtype policy:
        // an unknown field is Null, not an error.
        let batch = make_batch();
        let out = apply(
            &FilterSpec {
                predicate: "datum.absent > 1".into(),
                name: None,
                param: None,
            },
            &batch,
        )
        .unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn expr_dtype_support_matches_field_value() {
        // Pins `expr_dtype_supported` against `field_value`'s own dispatch:
        // every admitted dtype must read as a non-Null value from a non-null
        // cell, and every refused dtype must not.
        let cases: Vec<arrow::array::ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(arrow::array::Float32Array::from(vec![1.0_f32])),
            Arc::new(arrow::array::Int8Array::from(vec![1_i8])),
            Arc::new(arrow::array::Int16Array::from(vec![1_i16])),
            Arc::new(arrow::array::Int32Array::from(vec![1_i32])),
            Arc::new(arrow::array::Int64Array::from(vec![1_i64])),
            Arc::new(arrow::array::UInt8Array::from(vec![1_u8])),
            Arc::new(arrow::array::UInt16Array::from(vec![1_u16])),
            Arc::new(arrow::array::UInt32Array::from(vec![1_u32])),
            Arc::new(arrow::array::UInt64Array::from(vec![1_u64])),
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(arrow::array::LargeStringArray::from(vec!["a"])),
            Arc::new(arrow::array::BooleanArray::from(vec![true])),
            Arc::new(arrow::array::TimestampMillisecondArray::from(vec![1_i64])),
            Arc::new(arrow::array::TimestampSecondArray::from(vec![1_i64])),
            Arc::new(arrow::array::TimestampMicrosecondArray::from(vec![1_i64])),
            Arc::new(arrow::array::TimestampNanosecondArray::from(vec![1_i64])),
            Arc::new(arrow::array::Date32Array::from(vec![1_i32])),
            Arc::new(arrow::array::Time32SecondArray::from(vec![1_i32])),
        ];
        for col in cases {
            let dtype = col.data_type().clone();
            let batch = one_col("f", col);
            let readable = field_value(&batch, &batch.schema(), "f", 0) != ExprValue::Null;
            assert_eq!(
                expr_dtype_supported(&dtype),
                readable,
                "expr_dtype_supported and field_value disagree on {dtype:?}"
            );
        }
    }
}
