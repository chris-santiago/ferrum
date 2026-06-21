//! Shared group-keying for the stat transforms (aggregate, summary, error_extent,
//! box_stats, violin).
//!
//! All grouping transforms partition rows by one or more groupby columns, then
//! reconstruct those columns in their output batch with the *original* dtype
//! preserved. This module is the single source of truth for:
//!   - which Arrow dtypes are valid groupby keys ([`is_groupby_supported_dtype`]),
//!   - how a single row's key is extracted ([`groupby_key_at`]), and
//!   - how the keys are materialized back into an output column ([`materialize_groupby_col`]).
//!
//! Before this module existed, four sibling transforms each carried their own
//! `Str`/`Float`-only `KeyValue` enum and rejected integer/uint/bool groupby
//! columns. Centralizing the logic here means int/uint/bool/null support exists
//! once and cannot drift.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Internal representation of a group key value. Order matters: `BTreeMap` relies on `Ord`.
///
/// `Null` sorts first (lowest), then the typed variants. This ensures that null
/// rows from any dtype always collapse into the same group without colliding with
/// any real value (avoids the "0 as null sentinel" problem for integer columns).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum KeyValue {
    Null,       // null row for any dtype — groups all nulls together.
    Str(String),
    Float(u64), // f64 bits — NaN rows use Null, not Float(NAN.to_bits()).
    Int(i64),   // signed integer types (i8/i16/i32/i64), widened to i64 for grouping.
    UInt(u64),  // unsigned integer types (u8/u16/u32/u64), widened to u64 for grouping.
    Bool(u8),   // boolean: 0 = false, 1 = true (nulls use Null variant above).
}

/// Returns `true` for the Arrow dtypes supported as groupby keys.
pub(crate) fn is_groupby_supported_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Float64
            | DataType::Float32
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
    )
}

/// Result of [`group_partition`]: the distinct group keys in first-appearance
/// order, the per-key row-index map, and the group column's original dtype.
pub(crate) type GroupPartition = (Vec<KeyValue>, BTreeMap<KeyValue, Vec<usize>>, DataType);

/// Partition a batch's rows by a single groupby column.
///
/// This is the one partition path shared by every grouped stat transform
/// (`bin`, `kde`, `kde_2d`, `smooth`). It validates the column dtype, then walks
/// the rows once to build a [`GroupPartition`]:
///   - `group_order`: the distinct group keys in **first-appearance** order,
///   - `group_idx_map`: each key's row indices (a `BTreeMap` so per-group output
///     materialization is deterministic), and
///   - the group column's original [`DataType`], so callers can reconstruct the
///     output column with [`materialize_groupby_col`] and the dtype preserved
///     end-to-end.
///
/// **Null-key policy (centralized here — XFORM-09):** rows whose group key is
/// [`KeyValue::Null`] are skipped, so a null group never becomes its own series.
/// Every grouped transform shares this one policy; previously `kde`/`kde_2d`/
/// `smooth` each had an undocumented inline `is_null` skip that could drift from
/// `bin`'s. Keeping the skip in one place means the policy cannot diverge across
/// the family.
///
/// `ctx` is the transform-name prefix (e.g. `"stat_kde"`) used only in the
/// error messages, matching the convention in [`crate::transform::numeric_util`].
pub(crate) fn group_partition(
    batch: &RecordBatch,
    group_col: &str,
    ctx: &str,
) -> PyResult<GroupPartition> {
    let schema = batch.schema();
    let gi = schema.index_of(group_col).map_err(|_| {
        PyValueError::new_err(format!("{ctx}: groupby column '{group_col}' not found"))
    })?;
    let gtype = schema.field(gi).data_type().clone();
    if !is_groupby_supported_dtype(&gtype) {
        return Err(PyValueError::new_err(format!(
            "{ctx}: groupby column '{group_col}' has unsupported dtype {gtype:?}; \
             supported: Utf8/LargeUtf8, Float64/Float32, \
             Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
        )));
    }
    let garr = batch.column(gi);

    let mut group_order: Vec<KeyValue> = Vec::new();
    let mut group_idx_map: BTreeMap<KeyValue, Vec<usize>> = BTreeMap::new();
    for i in 0..garr.len() {
        let gv = groupby_key_at(garr.as_ref(), &gtype, i).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{ctx}: internal error extracting groupby key at row {i}"
            ))
        })?;
        // Null group keys collapse into their own KeyValue::Null, which the
        // centralized policy skips (see the doc comment above).
        if matches!(gv, KeyValue::Null) {
            continue;
        }
        if !group_idx_map.contains_key(&gv) {
            group_order.push(gv.clone());
        }
        group_idx_map.entry(gv).or_default().push(i);
    }

    Ok((group_order, group_idx_map, gtype))
}

/// Extract the groupby [`KeyValue`] for a single row from an Arrow column.
///
/// Signed integers (i8/i16/i32/i64) are widened to `KeyValue::Int(i64)`.
/// Unsigned integers (u8/u16/u32/u64) are widened to `KeyValue::UInt(u64)`.
/// Float32 is widened to `KeyValue::Float` (f64 bits) to match Float64 grouping.
/// Null rows always produce `KeyValue::Null` regardless of dtype, so nulls from
/// the same column always collapse into one group with no risk of colliding with
/// a real value.
///
/// Returns `None` only if `dt` is not a supported groupby dtype (should not
/// happen after validation, but guards against logic errors).
pub(crate) fn groupby_key_at(arr: &dyn Array, dt: &DataType, row: usize) -> Option<KeyValue> {
    macro_rules! int_key {
        ($arr_ty:ty, $variant:expr) => {{
            let a = arr.as_any().downcast_ref::<$arr_ty>()?;
            return Some(if a.is_null(row) {
                KeyValue::Null
            } else {
                $variant(a.value(row) as _)
            });
        }};
    }

    match dt {
        DataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>()?;
            Some(if a.is_null(row) {
                KeyValue::Null
            } else {
                KeyValue::Str(a.value(row).to_string())
            })
        }
        DataType::LargeUtf8 => {
            use arrow::array::LargeStringArray;
            let a = arr.as_any().downcast_ref::<LargeStringArray>()?;
            Some(if a.is_null(row) {
                KeyValue::Null
            } else {
                KeyValue::Str(a.value(row).to_string())
            })
        }
        DataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>()?;
            Some(if a.is_null(row) {
                KeyValue::Null
            } else {
                KeyValue::Float(a.value(row).to_bits())
            })
        }
        DataType::Float32 => {
            use arrow::array::Float32Array;
            let a = arr.as_any().downcast_ref::<Float32Array>()?;
            Some(if a.is_null(row) {
                KeyValue::Null
            } else {
                KeyValue::Float((a.value(row) as f64).to_bits())
            })
        }
        DataType::Int64 => int_key!(Int64Array, KeyValue::Int),
        DataType::Int32 => int_key!(Int32Array, KeyValue::Int),
        DataType::Int16 => int_key!(Int16Array, KeyValue::Int),
        DataType::Int8 => int_key!(Int8Array, KeyValue::Int),
        DataType::UInt64 => int_key!(UInt64Array, KeyValue::UInt),
        DataType::UInt32 => int_key!(UInt32Array, KeyValue::UInt),
        DataType::UInt16 => int_key!(UInt16Array, KeyValue::UInt),
        DataType::UInt8 => int_key!(UInt8Array, KeyValue::UInt),
        DataType::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>()?;
            Some(if a.is_null(row) {
                KeyValue::Null
            } else {
                KeyValue::Bool(a.value(row) as u8)
            })
        }
        _ => None,
    }
}

/// Reconstruct the output Arrow column for a single groupby dimension from the
/// materialized group keys. The output dtype matches the original input column
/// dtype so the schema is preserved end-to-end.
///
/// `keys` is the per-group list of key tuples; `gi` selects which groupby
/// dimension to materialize. Signed-integer variants are narrowed back to their
/// original width.
///
/// FA-9: a [`KeyValue::Null`] group (rows whose groupby column was null) is
/// emitted as an actual Arrow null in the output column — for every dtype,
/// including integers and booleans. This keeps a null-keyed group DISTINCT from
/// a genuine `0` / `false` / empty-string key all the way to the downstream
/// category scale, which treats the null as its own category. The change is
/// wire-neutral: Arrow nulls travel over the C Data Interface unchanged, so no
/// JSON/transport contract is affected. Callers that build a non-nullable output
/// field must mark the groupby field as nullable (see [`groupby_field_nullable`]).
pub(crate) fn materialize_groupby_col(
    keys: &[Vec<KeyValue>],
    gi: usize,
    dt: &DataType,
) -> Result<ArrayRef, String> {
    macro_rules! build_int {
        ($arr_ty:ty, $native:ty) => {{
            let v: Vec<Option<$native>> = keys
                .iter()
                .map(|k| match k[gi] {
                    KeyValue::Int(x) => Some(x as $native),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::Int or Null for signed-integer dtype"),
                })
                .collect();
            Ok(Arc::new(<$arr_ty>::from(v)) as ArrayRef)
        }};
    }
    macro_rules! build_uint {
        ($arr_ty:ty, $native:ty) => {{
            let v: Vec<Option<$native>> = keys
                .iter()
                .map(|k| match k[gi] {
                    KeyValue::UInt(x) => Some(x as $native),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::UInt or Null for unsigned-integer dtype"),
                })
                .collect();
            Ok(Arc::new(<$arr_ty>::from(v)) as ArrayRef)
        }};
    }

    match dt {
        DataType::Utf8 => {
            let v: Vec<Option<String>> = keys
                .iter()
                .map(|k| match &k[gi] {
                    KeyValue::Str(s) => Some(s.clone()),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::Str or Null for Utf8 dtype"),
                })
                .collect();
            Ok(Arc::new(StringArray::from(v)) as ArrayRef)
        }
        DataType::LargeUtf8 => {
            use arrow::array::LargeStringArray;
            let v: Vec<Option<String>> = keys
                .iter()
                .map(|k| match &k[gi] {
                    KeyValue::Str(s) => Some(s.clone()),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::Str or Null for LargeUtf8 dtype"),
                })
                .collect();
            Ok(Arc::new(LargeStringArray::from(v)) as ArrayRef)
        }
        DataType::Float64 => {
            let v: Vec<Option<f64>> = keys
                .iter()
                .map(|k| match k[gi] {
                    KeyValue::Float(bits) => Some(f64::from_bits(bits)),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::Float or Null for Float64 dtype"),
                })
                .collect();
            Ok(Arc::new(Float64Array::from(v)) as ArrayRef)
        }
        DataType::Float32 => {
            use arrow::array::Float32Array;
            let v: Vec<Option<f32>> = keys
                .iter()
                .map(|k| match k[gi] {
                    KeyValue::Float(bits) => Some(f64::from_bits(bits) as f32),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::Float or Null for Float32 dtype"),
                })
                .collect();
            Ok(Arc::new(Float32Array::from(v)) as ArrayRef)
        }
        DataType::Int64 => build_int!(Int64Array, i64),
        DataType::Int32 => build_int!(Int32Array, i32),
        DataType::Int16 => build_int!(Int16Array, i16),
        DataType::Int8 => build_int!(Int8Array, i8),
        DataType::UInt64 => build_uint!(UInt64Array, u64),
        DataType::UInt32 => build_uint!(UInt32Array, u32),
        DataType::UInt16 => build_uint!(UInt16Array, u16),
        DataType::UInt8 => build_uint!(UInt8Array, u8),
        DataType::Boolean => {
            let v: Vec<Option<bool>> = keys
                .iter()
                .map(|k| match k[gi] {
                    KeyValue::Bool(b) => Some(b != 0),
                    KeyValue::Null => None,
                    _ => unreachable!("expected KeyValue::Bool or Null for Boolean dtype"),
                })
                .collect();
            Ok(Arc::new(BooleanArray::from(v)) as ArrayRef)
        }
        other => Err(format!(
            "group_key: cannot materialize groupby output for dtype {other:?}"
        )),
    }
}

/// Whether the output groupby field must be declared nullable.
///
/// [`materialize_groupby_col`] emits an Arrow null for any [`KeyValue::Null`]
/// group (FA-9), so the field must be nullable whenever a null-keyed group can
/// appear. Always returning `true` is correct and keeps the schema honest; a
/// non-nullable declaration would make `RecordBatch::try_new` reject a column
/// that legitimately carries a null group key.
pub(crate) fn groupby_field_nullable() -> bool {
    true
}

/// Build an empty (zero-row) output column for the given groupby dtype.
///
/// Used by transforms (e.g. violin) that emit a valid, correctly-typed output
/// batch even when the input is empty.
pub(crate) fn empty_groupby_col(dt: &DataType) -> Result<ArrayRef, String> {
    materialize_groupby_col(&[], 0, dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;

    #[test]
    fn null_and_nan_float_keys_are_distinct() {
        // FA-3 contract: a null float key (KeyValue::Null) and a genuine NaN
        // value key (KeyValue::Float(NAN.to_bits())) must not collide.
        let arr = Float64Array::from(vec![None, Some(f64::NAN), Some(1.0)]);
        let k_null = groupby_key_at(&arr, &DataType::Float64, 0).unwrap();
        let k_nan = groupby_key_at(&arr, &DataType::Float64, 1).unwrap();
        let k_real = groupby_key_at(&arr, &DataType::Float64, 2).unwrap();
        assert_eq!(k_null, KeyValue::Null);
        assert!(matches!(k_nan, KeyValue::Float(_)));
        assert_ne!(k_null, k_nan, "null and NaN must be distinct keys");
        assert_ne!(k_nan, k_real);
    }

    #[test]
    fn int64_key_extract_and_materialize_roundtrip() {
        let arr = Int64Array::from(vec![Some(10), None, Some(20)]);
        let keys: Vec<Vec<KeyValue>> = (0..3)
            .map(|r| vec![groupby_key_at(&arr, &DataType::Int64, r).unwrap()])
            .collect();
        assert_eq!(keys[0][0], KeyValue::Int(10));
        assert_eq!(keys[1][0], KeyValue::Null);
        assert_eq!(keys[2][0], KeyValue::Int(20));

        let col = materialize_groupby_col(&keys, 0, &DataType::Int64).unwrap();
        let out = col.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(out.value(0), 10);
        // FA-9: the null int key materializes as an actual Arrow null, NOT a 0
        // proxy, so a null group stays distinct from a genuine 0 key downstream.
        assert!(out.is_null(1), "null int key materializes as Arrow null");
        assert_eq!(out.value(2), 20);
    }

    #[test]
    fn fa9_null_int_key_distinct_from_real_zero_key() {
        // FA-9 regression: {Null, Int(0), Int(5)} must produce three
        // distinguishable output values — null as an Arrow null, 0 and 5 as
        // genuine values — so a downstream category scale shows three groups.
        let keys: Vec<Vec<KeyValue>> = vec![
            vec![KeyValue::Null],
            vec![KeyValue::Int(0)],
            vec![KeyValue::Int(5)],
        ];
        let col = materialize_groupby_col(&keys, 0, &DataType::Int64).unwrap();
        let out = col.as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(out.is_null(0), "null key must be a distinct Arrow null");
        assert!(!out.is_null(1));
        assert_eq!(out.value(1), 0, "real 0 key stays 0");
        assert!(!out.is_null(2));
        assert_eq!(out.value(2), 5);
    }

    #[test]
    fn unsupported_dtype_rejected() {
        assert!(!is_groupby_supported_dtype(&DataType::Date32));
        assert!(is_groupby_supported_dtype(&DataType::Int32));
        assert!(is_groupby_supported_dtype(&DataType::Boolean));
        assert!(is_groupby_supported_dtype(&DataType::Utf8));
    }

    // ── group_partition (T0.3 unification) ──────────────────────────────────

    use arrow::array::BooleanArray;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn batch_with_group(col: ArrayRef) -> RecordBatch {
        let nullable = col.null_count() > 0;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "g",
            col.data_type().clone(),
            nullable,
        )]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    #[test]
    fn group_partition_int64_first_appearance_order() {
        // Values 20, 10, 20, 10 → first-appearance order is [20, 10].
        let col: ArrayRef = Arc::new(Int64Array::from(vec![20_i64, 10, 20, 10]));
        let batch = batch_with_group(col);
        let (order, map, dt) = group_partition(&batch, "g", "test").unwrap();
        assert_eq!(dt, DataType::Int64);
        assert_eq!(order, vec![KeyValue::Int(20), KeyValue::Int(10)]);
        assert_eq!(map.get(&KeyValue::Int(20)).unwrap(), &vec![0_usize, 2]);
        assert_eq!(map.get(&KeyValue::Int(10)).unwrap(), &vec![1_usize, 3]);
    }

    #[test]
    fn group_partition_boolean_groups() {
        let col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true]));
        let batch = batch_with_group(col);
        let (order, map, dt) = group_partition(&batch, "g", "test").unwrap();
        assert_eq!(dt, DataType::Boolean);
        assert_eq!(order, vec![KeyValue::Bool(1), KeyValue::Bool(0)]);
        assert_eq!(map.get(&KeyValue::Bool(1)).unwrap(), &vec![0_usize, 2]);
        assert_eq!(map.get(&KeyValue::Bool(0)).unwrap(), &vec![1_usize]);
    }

    #[test]
    fn group_partition_skips_null_keys() {
        // Centralized null-key policy: null rows are excluded from the partition.
        let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(1), None]));
        let batch = batch_with_group(col);
        let (order, map, _) = group_partition(&batch, "g", "test").unwrap();
        assert_eq!(order, vec![KeyValue::Int(1)]);
        assert_eq!(map.get(&KeyValue::Int(1)).unwrap(), &vec![0_usize, 2]);
        assert!(!map.contains_key(&KeyValue::Null), "null key must be skipped");
    }

    #[test]
    fn group_partition_unsupported_dtype_errors() {
        pyo3::Python::initialize();
        use arrow::array::Date32Array;
        let col: ArrayRef = Arc::new(Date32Array::from(vec![0_i32, 1]));
        let batch = batch_with_group(col);
        let err = group_partition(&batch, "g", "stat_kde").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported dtype"), "err: {msg}");
        assert!(msg.contains("stat_kde"), "err must carry ctx: {msg}");
    }

    #[test]
    fn group_partition_missing_column_errors() {
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2]));
        let batch = batch_with_group(col);
        let err = group_partition(&batch, "ghost", "stat_smooth").unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }
}
