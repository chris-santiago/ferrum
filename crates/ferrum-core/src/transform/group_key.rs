//! The crate's grouping toolkit: everything every grouped transform needs to
//! turn a `groupby` request into partitioned rows and back into an output
//! batch whose group columns carry their *original* dtypes.
//!
//! What lives here (the charter grew well past the original three-item
//! key-handling list, and a refusal policy landing inside a stacking helper
//! was the cost of not saying so — batch-F close review):
//!   - **Key vocabulary** — which Arrow dtypes are valid groupby keys
//!     ([`is_groupby_supported_dtype`]), how one row's key is extracted
//!     ([`groupby_key_at`]), and how keys are materialized back into an
//!     output column ([`materialize_groupby_col`], [`groupby_field_nullable`]).
//!   - **Partitioning** — [`group_partition_multi`], the shared partitioner.
//!     It owns the policies that belong to the groupby *request* itself:
//!     dtype support, first-appearance group order, the null-key skip, and
//!     the repeated-name refusal. Most grouped transforms route through it;
//!     the ones that still hand-roll their own loop over the key primitives
//!     above are named in its doc, and moving them over is a filed follow-up.
//!   - **Grouped-execution templates** — [`grouped_two_column_fit`] (the
//!     `RegressionData`/`LoessData` shape: fit per group, stack two Float64
//!     grids) and [`stack_grouped_batches`] (the `Smooth`/`Robust` shape:
//!     stack already-computed per-group batches). Both append the group
//!     columns; the second also refuses a groupby name that collides with
//!     the transform's own output columns.
//!   - **Boundary parsing** — the wire back-compat shim
//!     ([`deserialize_groupby_vec`]) and the PyO3 argument parser
//!     ([`parse_groupby_pyany`]), so `null`/`"col"`/`["a","b"]` normalize to
//!     `Vec<String>` in exactly one place.
//!
//! Before this module existed, four sibling transforms each carried their own
//! `Str`/`Float`-only `KeyValue` enum and rejected integer/uint/bool groupby
//! columns. Centralizing the logic here means int/uint/bool/null support exists
//! once and cannot drift.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
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

/// Result of [`group_partition_multi`]: distinct group key tuples in first-appearance
/// order, the per-tuple row-index map, and each group column's original dtype.
pub(crate) type GroupPartitionMulti = (
    Vec<Vec<KeyValue>>,
    std::collections::BTreeMap<Vec<KeyValue>, Vec<usize>>,
    Vec<DataType>,
);

/// Partition a batch's rows by multiple groupby columns.
///
/// The single grouping entry point for all per-group transforms (bin, kde,
/// kde_2d, smooth, ...): first-appearance group order, null-key skip,
/// dtype-preservation. When `group_cols` is empty, every row keys to the empty
/// `Vec`, so the result is ONE all-rows group — `stat_outliers`' ungrouped path
/// depends on exactly this (batch A migrated it off an explicit all-rows
/// insert), so tightening this to an empty partition silently breaks it. A
/// single-element `group_cols` yields the conventional single-column grouping.
///
/// **Null-key policy:** a row is skipped if ANY of its group-column values is null.
/// This mirrors the single-column behaviour (a null in the groupby column is not
/// assigned to any group). The null skip applies when *any* dimension of the
/// composite key contains [`KeyValue::Null`].
///
/// **Repeated-name policy:** `group_cols` may not repeat a name (e.g.
/// `groupby=["g", "g"]`). A repeat is a malformed *request*, not a property of
/// any one transform's output shape: a grouped transform appends one output
/// field per groupby dimension, so a repeat means two identically-named Arrow
/// fields, and `Schema::index_of` silently resolves to the FIRST match — every
/// downstream reader of "the groupby column" would see the wrong column's
/// values. The refusal lives here rather than in any single execution
/// template: batch F installed it in [`stack_grouped_batches`] instead, which
/// closed `smooth`/`robust` and left every other grouped transform corrupting
/// silently (batch-F close review, F1). Installed here it covers every caller
/// of this function — `bin`, `kde`, `kde_2d`, `smooth`, `robust`, `outliers`,
/// `top_k`, `impute`, plus `regression_data`/`loess_data` through
/// [`grouped_two_column_fit`]. It does NOT cover the transforms that still
/// hand-roll their own partition loop over this module's key primitives
/// (`summary`, `violin`, `aggregate`, `box_stats`, `error_extent`,
/// `density_data`, `pivot`, `data_aggregate`); migrating those onto this
/// partitioner is the filed follow-up that also generalizes
/// `stack_grouped_batches`'s output-collision guard. `ctx` keeps the error in
/// the calling transform's voice.
///
/// **That residual is a duplicated rule, not an open hole.** No public API
/// path reaches a hand-rolled partitioner with a repeated groupby name:
/// `aggregate`, `summary`, `violin`, `box_stats`, and `error_extent` each
/// refuse the same repeat at their own PyO3 constructor (`"Aggregate:
/// duplicate groupby field 'g'"` and its four siblings), and `density_data`
/// (`PyDensityData::new` takes only `field`/`name`), `pivot`, and
/// `data_aggregate` (no `#[pyclass]` at all — wire-only) cannot be
/// constructed with a `groupby` from Python in the first place. So the
/// invariant holds everywhere; what differs is WHERE and in WHOSE WORDS it
/// is enforced — one rule with two messages at two lifecycle stages
/// (construction time for those five, partition time here). The follow-up is
/// therefore a consolidation for message consistency and wire-path coverage,
/// not a corruption fix; scope it accordingly.
pub(crate) fn group_partition_multi(
    batch: &RecordBatch,
    group_cols: &[String],
    ctx: &str,
) -> PyResult<GroupPartitionMulti> {
    let schema = batch.schema();
    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(group_cols.len());
    let mut group_arrays: Vec<usize> = Vec::with_capacity(group_cols.len());
    let mut seen_group_cols: std::collections::HashSet<&str> =
        std::collections::HashSet::with_capacity(group_cols.len());

    for col_name in group_cols {
        let gi = schema.index_of(col_name).map_err(|_| {
            PyValueError::new_err(format!("{ctx}: groupby column '{col_name}' not found"))
        })?;
        // Checked after the existence lookup so a missing column still reports
        // "not found" first, as it did before this guard moved here.
        if !seen_group_cols.insert(col_name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "{ctx}: groupby column '{col_name}' is repeated in `groupby`; \
                 each groupby column must have a distinct name (got groupby={group_cols:?})"
            )));
        }
        let gtype = schema.field(gi).data_type().clone();
        if !is_groupby_supported_dtype(&gtype) {
            return Err(PyValueError::new_err(format!(
                "{ctx}: groupby column '{col_name}' has unsupported dtype {gtype:?}; \
                 supported: Utf8/LargeUtf8, Float64/Float32, \
                 Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
            )));
        }
        group_dtypes.push(gtype);
        group_arrays.push(gi);
    }

    let n_rows = batch.num_rows();
    let mut group_order: Vec<Vec<KeyValue>> = Vec::new();
    let mut group_idx_map: std::collections::BTreeMap<Vec<KeyValue>, Vec<usize>> =
        std::collections::BTreeMap::new();

    for row in 0..n_rows {
        let mut key: Vec<KeyValue> = Vec::with_capacity(group_cols.len());
        let mut has_null = false;
        for (dim, &col_idx) in group_arrays.iter().enumerate() {
            let arr = batch.column(col_idx);
            let kv = groupby_key_at(arr.as_ref(), &group_dtypes[dim], row).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{ctx}: internal error extracting groupby key at row {row}"
                ))
            })?;
            if matches!(kv, KeyValue::Null) {
                has_null = true;
                break;
            }
            key.push(kv);
        }
        if has_null {
            continue;
        }
        if !group_idx_map.contains_key(&key) {
            group_order.push(key.clone());
        }
        group_idx_map.entry(key).or_default().push(row);
    }

    Ok((group_order, group_idx_map, group_dtypes))
}

/// Partition `batch` by `group_cols`, run `fit_group` on each partition's row
/// indices, and stack the resulting two-column Float64 grids into one batch
/// with the group columns appended at their original dtype (XFORM-03).
///
/// This is the grouped-execution template `RegressionData` and `LoessData`
/// share (R1 design finding 2: the two `apply_grouped` bodies were 49 lines
/// differing in six tokens — the spec type and five error-message prefixes).
/// `label` supplies the transform's error voice; `as_` names the two output
/// columns; `fit_group` is the per-group fit, which must return a batch whose
/// first two columns are Float64 (each transform's own `apply_one_group`,
/// including its below-threshold NaN grid).
///
/// A group's own degenerate output still contributes its rows and its key, so
/// the group stays in the output schema, the color-scale domain, and the
/// legend rather than silently vanishing — see each caller's `all_nan_output`
/// doc for why that is the right disposition.
pub(crate) fn grouped_two_column_fit(
    batch: &RecordBatch,
    group_cols: &[String],
    label: &str,
    as_: &(String, String),
    fit_group: impl Fn(&[usize]) -> PyResult<RecordBatch>,
) -> PyResult<RecordBatch> {
    use arrow::datatypes::{Field, Schema};

    let (group_order, group_idx_map, gtypes) = group_partition_multi(batch, group_cols, label)?;

    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();

    for g in &group_order {
        let ixs = group_idx_map.get(g).ok_or_else(|| {
            PyValueError::new_err(format!("{label}: missing group key {g:?} in index map"))
        })?;
        let out = fit_group(ixs)?;
        let out_x = out.column(0).as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err(format!("{label}: expected Float64Array in grouped output"))
        })?;
        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err(format!("{label}: expected Float64Array in grouped output"))
        })?;
        for i in 0..out.num_rows() {
            all_x.push(out_x.value(i));
            all_y.push(out_y.value(i));
            group_keys_out.push(g.clone());
        }
    }

    let mut out_fields = vec![
        Field::new(&as_.0, DataType::Float64, false),
        Field::new(&as_.1, DataType::Float64, false),
    ];
    let mut out_cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_x)),
        Arc::new(Float64Array::from(all_y)),
    ];

    for (gi, (col_name, dtype)) in group_cols.iter().zip(gtypes.iter()).enumerate() {
        out_fields.push(Field::new(col_name, dtype.clone(), groupby_field_nullable()));
        let group_col_arr =
            materialize_groupby_col(&group_keys_out, gi, dtype).map_err(PyValueError::new_err)?;
        out_cols.push(group_col_arr);
    }

    RecordBatch::try_new(Arc::new(Schema::new(out_fields)), out_cols)
        .map_err(|e| PyValueError::new_err(format!("{label}: {e}")))
}

/// Stack a transform's already-computed per-group output batches into one
/// batch, appending the group-key columns at their ORIGINAL dtypes
/// (XFORM-03).
///
/// This is the grouped-stacking template `smooth::apply_grouped` and
/// `robust::apply_grouped` share (batch-F T2h hoist: the two bodies were
/// ~30 lines differing only in the context label passed through error
/// messages). `ctx` supplies that label; `per_group_batches` is one
/// `(group key, per-group output batch)` pair per distinct group, in the
/// order they should be stacked; `group_cols` / `gtypes` are
/// [`group_partition_multi`]'s own output, so callers pass them straight
/// through.
///
/// Reconciles schema divergence across groups via [`concat_by_name`] (a
/// degenerate fit for one group can legitimately emit a narrower batch than
/// its siblings — see that function's module doc), then appends one output
/// field per groupby dimension, materialized via [`materialize_groupby_col`]
/// so an `Int64`/`Boolean`/etc. groupby column round-trips at its original
/// dtype rather than widening to `String`.
///
/// **Output-collision guard:** the output must not carry two fields with the
/// same name. This refuses the case that is a property of *appending group
/// columns*: a groupby column's name matches one of the per-group output's
/// own field names (e.g. grouping by a column literally named `x` or
/// `residual`). Arrow does not reject duplicate schema field names, and
/// `Schema::index_of` silently resolves to the FIRST match, so every
/// downstream reader of "the groupby column" would silently see the wrong
/// column's values instead. The error names both the colliding column and
/// this transform's actual output columns, so the message is actionable
/// from Python — a caller can rename the offending groupby column, but
/// cannot rename a transform's fixed output schema. This is a behavior
/// change versus the pre-hoist `smooth`/`robust` bodies, which had no such
/// guard — the collision was previously mis-resolved silently, not merely
/// undocumented. It is still installed here rather than at the ~13 sites
/// that append group columns; generalizing it is a filed follow-up.
///
/// The sibling case — `group_cols` itself repeating a name — is NOT checked
/// here: it is a property of the groupby *request* rather than of stacking,
/// so it belongs to (and lives in) [`group_partition_multi`], which both
/// callers of this function run first and whose output `group_cols`/`gtypes`
/// they pass straight through. See that function's "Repeated-name policy".
pub(crate) fn stack_grouped_batches(
    ctx: &str,
    per_group_batches: &[(Vec<KeyValue>, RecordBatch)],
    group_cols: &[String],
    gtypes: &[DataType],
) -> PyResult<RecordBatch> {
    let total_rows: usize = per_group_batches.iter().map(|(_, b)| b.num_rows()).sum();
    let stacked = crate::transform::concat::concat_by_name(
        ctx,
        per_group_batches.iter().map(|(_, b)| b.clone()).collect(),
    )
    .map_err(PyValueError::new_err)?;

    let output_field_names: Vec<String> =
        stacked.schema().fields().iter().map(|f| f.name().clone()).collect();
    for col_name in group_cols {
        if output_field_names.iter().any(|f| f == col_name) {
            return Err(PyValueError::new_err(format!(
                "{ctx}: groupby column '{col_name}' collides with this transform's own output \
                 column '{col_name}' (output columns: {}); rename the data column before grouping",
                output_field_names.join(", ")
            )));
        }
    }

    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(total_rows);
    for (g, b) in per_group_batches {
        for _ in 0..b.num_rows() {
            group_keys_out.push(g.clone());
        }
    }
    let mut out_fields: Vec<Field> =
        stacked.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut out_cols: Vec<ArrayRef> = stacked.columns().to_vec();
    for (gi, (col_name, dtype)) in group_cols.iter().zip(gtypes.iter()).enumerate() {
        out_fields.push(Field::new(col_name, dtype.clone(), groupby_field_nullable()));
        let group_col_arr =
            materialize_groupby_col(&group_keys_out, gi, dtype).map_err(PyValueError::new_err)?;
        out_cols.push(group_col_arr);
    }

    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols).map_err(|e| PyValueError::new_err(format!("{ctx}: {e}")))
}

/// Serde deserializer that accepts `null`, a bare JSON string, or a JSON array
/// of strings, normalising all three to `Vec<String>`:
///   - `null`      → `[]`  (ungrouped)
///   - `"col"`     → `["col"]`  (single-column — legacy wire format)
///   - `["a","b"]` → `["a","b"]`  (multi-column — canonical wire format)
///
/// This is the wire back-compat shim required by D-GROUPBY-1: old serialized specs
/// that carried a bare string `"groupby":"col"` remain readable after the migration
/// to `Vec<String>`. Serialize always emits the array form (via the struct's
/// `#[serde(skip_serializing_if = "Vec::is_empty")]` attribute).
pub(crate) fn deserialize_groupby_vec<'de, D>(
    deserializer: D,
) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct GroupbyVisitor;

    impl<'de> Visitor<'de> for GroupbyVisitor {
        type Value = Vec<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "null, a string, or an array of strings")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Vec<String>, E> {
            Ok(Vec::new())
        }

        fn visit_none<E: de::Error>(self) -> Result<Vec<String>, E> {
            Ok(Vec::new())
        }

        fn visit_some<D2: serde::Deserializer<'de>>(
            self,
            d: D2,
        ) -> Result<Vec<String>, D2::Error> {
            d.deserialize_any(GroupbyVisitor)
        }

        fn visit_str<E: de::Error>(self, s: &str) -> Result<Vec<String>, E> {
            Ok(vec![s.to_owned()])
        }

        fn visit_string<E: de::Error>(self, s: String) -> Result<Vec<String>, E> {
            Ok(vec![s])
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<String>, A::Error> {
            let mut out = Vec::new();
            while let Some(elem) = seq.next_element::<String>()? {
                out.push(elem);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_any(GroupbyVisitor)
}

/// Parse the `groupby` PyO3 constructor argument into `Vec<String>`.
///
/// Accepts:
///   - `None` → `[]`
///   - A Python `str` → `["col"]`
///   - A Python `list[str]` → the list contents
///
/// This is the Python-boundary equivalent of [`deserialize_groupby_vec`] — both
/// normalise the same three input forms to the canonical `Vec<String>` shape so the
/// PyO3 constructors and the JSON wire layer agree on what "ungrouped" means.
pub(crate) fn parse_groupby_pyany(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<String>> {
    match obj {
        None => Ok(Vec::new()),
        Some(o) => {
            if let Ok(s) = o.extract::<String>() {
                Ok(vec![s])
            } else if let Ok(v) = o.extract::<Vec<String>>() {
                Ok(v)
            } else {
                Err(PyValueError::new_err(
                    "groupby must be None, a str, or a list[str]",
                ))
            }
        }
    }
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

    // ── group_partition_multi single-column coverage (T0.3 / T3.8a) ──────────
    //
    // These tests previously called the now-deleted `group_partition` helper.
    // They are rewritten to call `group_partition_multi` with a 1-element slice,
    // which is the canonical entry point after T3.8a. The single-key case
    // exercises identical code paths inside `group_partition_multi`; only the
    // return shape differs (Vec<Vec<KeyValue>> keys, Vec<DataType> dtypes).

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
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(Int64Array::from(vec![20_i64, 10, 20, 10]));
        let batch = batch_with_group(col);
        let (order, map, dtypes) =
            group_partition_multi(&batch, &["g".to_string()], "test").unwrap();
        assert_eq!(dtypes[0], DataType::Int64);
        assert_eq!(
            order,
            vec![vec![KeyValue::Int(20)], vec![KeyValue::Int(10)]]
        );
        assert_eq!(map.get(&vec![KeyValue::Int(20)]).unwrap(), &vec![0_usize, 2]);
        assert_eq!(map.get(&vec![KeyValue::Int(10)]).unwrap(), &vec![1_usize, 3]);
    }

    #[test]
    fn group_partition_boolean_groups() {
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true]));
        let batch = batch_with_group(col);
        let (order, map, dtypes) =
            group_partition_multi(&batch, &["g".to_string()], "test").unwrap();
        assert_eq!(dtypes[0], DataType::Boolean);
        assert_eq!(
            order,
            vec![vec![KeyValue::Bool(1)], vec![KeyValue::Bool(0)]]
        );
        assert_eq!(map.get(&vec![KeyValue::Bool(1)]).unwrap(), &vec![0_usize, 2]);
        assert_eq!(map.get(&vec![KeyValue::Bool(0)]).unwrap(), &vec![1_usize]);
    }

    #[test]
    fn group_partition_skips_null_keys() {
        // Centralized null-key policy: null rows are excluded from the partition.
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(1), None]));
        let batch = batch_with_group(col);
        let (order, map, _) =
            group_partition_multi(&batch, &["g".to_string()], "test").unwrap();
        assert_eq!(order, vec![vec![KeyValue::Int(1)]]);
        assert_eq!(map.get(&vec![KeyValue::Int(1)]).unwrap(), &vec![0_usize, 2]);
        assert!(
            !map.contains_key(&vec![KeyValue::Null]),
            "null key must be skipped"
        );
    }

    #[test]
    fn group_partition_unsupported_dtype_errors() {
        pyo3::Python::initialize();
        use arrow::array::Date32Array;
        let col: ArrayRef = Arc::new(Date32Array::from(vec![0_i32, 1]));
        let batch = batch_with_group(col);
        let err = group_partition_multi(&batch, &["g".to_string()], "stat_kde").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported dtype"), "err: {msg}");
        assert!(msg.contains("stat_kde"), "err must carry ctx: {msg}");
    }

    #[test]
    fn group_partition_missing_column_errors() {
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2]));
        let batch = batch_with_group(col);
        let err =
            group_partition_multi(&batch, &["ghost".to_string()], "stat_smooth").unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    // ── T3.8a regression: deserialize_groupby_vec wire formats ──────────────

    /// `deserialize_groupby_vec` accepts the three legacy wire forms:
    ///   - JSON `null` → `[]`
    ///   - JSON bare string `"col"` → `["col"]`
    ///   - JSON array `["a","b"]` → `["a","b"]`
    #[test]
    fn deserialize_groupby_vec_accepts_null_string_and_array() {
        use serde_json;

        // Wrap in a helper struct so we can drive the serde machinery.
        #[derive(serde::Deserialize)]
        struct W {
            #[serde(deserialize_with = "deserialize_groupby_vec")]
            groupby: Vec<String>,
        }

        let null_form: W = serde_json::from_str(r#"{"groupby": null}"#).unwrap();
        assert!(null_form.groupby.is_empty(), "null must deserialize to []");

        let str_form: W = serde_json::from_str(r#"{"groupby": "col"}"#).unwrap();
        assert_eq!(str_form.groupby, vec!["col"], "bare string must deserialize to [\"col\"]");

        let arr_form: W = serde_json::from_str(r#"{"groupby": ["a","b"]}"#).unwrap();
        assert_eq!(arr_form.groupby, vec!["a", "b"], "array must deserialize as-is");
    }

    /// When `groupby` is absent from the JSON, `default` yields `[]`.
    #[test]
    fn deserialize_groupby_vec_missing_field_yields_empty() {
        #[derive(serde::Deserialize)]
        struct W {
            #[serde(default, deserialize_with = "deserialize_groupby_vec")]
            groupby: Vec<String>,
        }

        let absent: W = serde_json::from_str(r#"{}"#).unwrap();
        assert!(absent.groupby.is_empty(), "missing field must default to []");
    }

    // ── T3.8a regression: group_partition_multi ──────────────────────────────

    /// `group_partition_multi` on two string columns produces composite keys
    /// (one `Vec<KeyValue>` per group) and preserves first-appearance order.
    #[test]
    fn group_partition_multi_two_string_columns() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("g1", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("g2", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "B", "B"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "y", "x", "y"])) as ArrayRef,
            ],
        )
        .unwrap();

        let (order, map, dtypes) =
            group_partition_multi(&batch, &["g1".to_string(), "g2".to_string()], "test").unwrap();

        assert_eq!(dtypes.len(), 2);
        assert_eq!(dtypes[0], arrow::datatypes::DataType::Utf8);
        assert_eq!(dtypes[1], arrow::datatypes::DataType::Utf8);

        // Four distinct groups in first-appearance order.
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], vec![KeyValue::Str("A".into()), KeyValue::Str("x".into())]);
        assert_eq!(order[1], vec![KeyValue::Str("A".into()), KeyValue::Str("y".into())]);
        assert_eq!(order[2], vec![KeyValue::Str("B".into()), KeyValue::Str("x".into())]);
        assert_eq!(order[3], vec![KeyValue::Str("B".into()), KeyValue::Str("y".into())]);

        assert_eq!(map[&order[0]], vec![0]);
        assert_eq!(map[&order[1]], vec![1]);
        assert_eq!(map[&order[2]], vec![2]);
        assert_eq!(map[&order[3]], vec![3]);
    }

    /// Rows where any dimension is null are skipped by `group_partition_multi`.
    #[test]
    fn group_partition_multi_skips_null_rows() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("g1", arrow::datatypes::DataType::Utf8, true),
            arrow::datatypes::Field::new("g2", arrow::datatypes::DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("A"), None, Some("B")])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("x"), Some("x"), Some("y")])) as ArrayRef,
            ],
        )
        .unwrap();

        let (order, map, _) =
            group_partition_multi(&batch, &["g1".to_string(), "g2".to_string()], "test").unwrap();

        // Row 1 has g1=null → excluded; only rows 0 and 2 appear.
        assert_eq!(order.len(), 2);
        assert_eq!(map[&order[0]], vec![0]);
        assert_eq!(map[&order[1]], vec![2]);
    }

    // ── BUG-HUNT (step4): partition order, multi-col, deserializer adversarial ──

    /// `group_order` is FIRST-APPEARANCE order, but `group_idx_map` is a BTreeMap
    /// keyed by `Vec<KeyValue>` (Ord). When first-appearance order DIFFERS from
    /// sorted order, the returned `group_order` must follow first-appearance, not
    /// the BTreeMap's sorted iteration — otherwise downstream group columns would
    /// be emitted in the wrong order. Insert "B" before "A" so the two orders differ.
    #[test]
    fn bughunt_group_order_is_first_appearance_not_sorted() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["B", "A", "B", "A", "C"])) as ArrayRef],
        )
        .unwrap();
        let (order, _, _) =
            group_partition_multi(&batch, &["g".to_string()], "test").unwrap();
        assert_eq!(
            order,
            vec![
                vec![KeyValue::Str("B".into())],
                vec![KeyValue::Str("A".into())],
                vec![KeyValue::Str("C".into())],
            ],
            "group_order must be first-appearance (B,A,C), not sorted (A,B,C)"
        );
    }

    /// A 3-column groupby with interleaved combos: the composite-key partition must
    /// produce exactly the distinct tuples, in first-appearance order, with each
    /// row routed to the correct tuple. Tests joint partitioning across 3 dims.
    #[test]
    fn bughunt_group_partition_multi_three_columns_joint() {
        pyo3::Python::initialize();
        use arrow::array::{Int64Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        // 5 rows; distinct (a,b,c): (A,1,x)[rows 0,3], (A,2,x)[1], (B,1,x)[2], (A,1,y)[4]
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "B", "A", "A"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1_i64, 2, 1, 1, 1])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "x", "x", "x", "y"])) as ArrayRef,
            ],
        )
        .unwrap();
        let (order, map, dtypes) = group_partition_multi(
            &batch,
            &["a".to_string(), "b".to_string(), "c".to_string()],
            "test",
        )
        .unwrap();
        assert_eq!(dtypes, vec![DataType::Utf8, DataType::Int64, DataType::Utf8]);
        assert_eq!(order.len(), 4, "expected 4 distinct triples");
        // First-appearance: (A,1,x), (A,2,x), (B,1,x), (A,1,y)
        let k_a1x = vec![KeyValue::Str("A".into()), KeyValue::Int(1), KeyValue::Str("x".into())];
        assert_eq!(order[0], k_a1x);
        assert_eq!(map[&k_a1x], vec![0, 3], "(A,1,x) must collect rows 0 and 3");
        let k_a1y = vec![KeyValue::Str("A".into()), KeyValue::Int(1), KeyValue::Str("y".into())];
        assert_eq!(map[&k_a1y], vec![4], "(A,1,y) must be distinct from (A,1,x)");
    }

    /// Null-skip is ANY-dimension: a row with a null in the SECOND column (first
    /// non-null) must still be skipped entirely. Pins that the early `break` on the
    /// first null dimension doesn't accidentally retain rows whose later dims are null.
    #[test]
    fn bughunt_group_partition_multi_null_in_later_dim_skips_row() {
        pyo3::Python::initialize();
        use arrow::array::{Int64Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A", "A", "A"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(1)])) as ArrayRef,
            ],
        )
        .unwrap();
        let (order, map, _) =
            group_partition_multi(&batch, &["a".to_string(), "b".to_string()], "test").unwrap();
        assert_eq!(order.len(), 1, "only (A,1) survives; (A,null) row is skipped");
        assert_eq!(map[&order[0]], vec![0, 2], "rows 0 and 2 only; row 1 (b=null) excluded");
    }

    /// Float group keys: a genuine NaN value and a null must NOT collide (FA-3),
    /// and both must be distinct from a finite key, when used as a 2-D composite.
    #[test]
    fn bughunt_group_partition_multi_float_nan_vs_null_distinct() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.0), Some(f64::NAN), None, Some(1.0)])) as ArrayRef],
        )
        .unwrap();
        let (order, map, _) =
            group_partition_multi(&batch, &["g".to_string()], "test").unwrap();
        // null row skipped; 1.0 and NaN form two distinct groups.
        assert_eq!(order.len(), 2, "1.0 and NaN are two groups; null is skipped");
        let k_one = vec![KeyValue::Float(1.0_f64.to_bits())];
        assert_eq!(map[&k_one], vec![0, 3], "the two 1.0 rows group together");
        assert!(
            map.keys().any(|k| matches!(k[0], KeyValue::Float(b) if f64::from_bits(b).is_nan())),
            "a NaN-keyed group must exist, distinct from null"
        );
    }

    /// `deserialize_groupby_vec` must REJECT a non-string array element (e.g. an
    /// integer), not silently coerce or drop it. Groupby column names are strings.
    #[test]
    fn bughunt_deserialize_groupby_vec_rejects_non_string_element() {
        #[derive(serde::Deserialize)]
        struct W {
            #[serde(deserialize_with = "deserialize_groupby_vec")]
            #[allow(dead_code)]
            groupby: Vec<String>,
        }
        let r: Result<W, _> = serde_json::from_str(r#"{"groupby": ["a", 7]}"#);
        assert!(r.is_err(), "an integer array element must be rejected, not coerced");
    }

    /// `deserialize_groupby_vec` on an empty array yields `[]` (ungrouped), exactly
    /// like null/absent — three spellings of "no grouping" must all converge.
    #[test]
    fn bughunt_deserialize_groupby_vec_empty_array_is_ungrouped() {
        #[derive(serde::Deserialize)]
        struct W {
            #[serde(deserialize_with = "deserialize_groupby_vec")]
            groupby: Vec<String>,
        }
        let w: W = serde_json::from_str(r#"{"groupby": []}"#).unwrap();
        assert!(w.groupby.is_empty(), "empty array must mean ungrouped");
    }

    /// `materialize_groupby_col` indexes `k[gi]`; an out-of-range `gi` would panic.
    /// Pins that materializing dim 0 of a single-dim key vec is valid and that the
    /// null/value mix produces the right Arrow null mask for a string column.
    #[test]
    fn bughunt_materialize_string_col_null_mask_correct() {
        let keys: Vec<Vec<KeyValue>> = vec![
            vec![KeyValue::Str("A".into())],
            vec![KeyValue::Null],
            vec![KeyValue::Str("B".into())],
        ];
        let col = materialize_groupby_col(&keys, 0, &DataType::Utf8).unwrap();
        let s = col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(s.value(0), "A");
        assert!(s.is_null(1), "Null key must materialize as Arrow null, not empty string");
        assert_eq!(s.value(2), "B");
    }

    // ── stack_grouped_batches (batch-F T2h hoist) ─────────────────────────

    fn xy_group_batch(x: Vec<f64>, y: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(x)) as ArrayRef,
                Arc::new(Float64Array::from(y)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    /// The ordinary case: two groups' per-group output batches stack into
    /// one batch with the group column appended at its original dtype,
    /// rows in the same order the per-group batches were supplied.
    #[test]
    fn stack_grouped_batches_happy_path_appends_group_column() {
        let per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = vec![
            (vec![KeyValue::Str("a".into())], xy_group_batch(vec![1.0, 2.0], vec![10.0, 20.0])),
            (vec![KeyValue::Str("b".into())], xy_group_batch(vec![3.0], vec![30.0])),
        ];
        let out = stack_grouped_batches(
            "test",
            &per_group_batches,
            &["g".to_string()],
            &[DataType::Utf8],
        )
        .unwrap();

        assert_eq!(out.num_rows(), 3);
        assert_eq!(out.schema().fields().len(), 3);
        assert_eq!(out.schema().field(2).name(), "g");
        let g_col = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(g_col.value(0), "a");
        assert_eq!(g_col.value(1), "a");
        assert_eq!(g_col.value(2), "b");
    }

    /// Batch-F T2h rider 1: a groupby column whose name collides with one of
    /// the per-group output's own field names must refuse with a typed
    /// error naming the colliding column, rather than appending a second
    /// same-named field that `Schema::index_of` would silently resolve to
    /// the wrong one.
    #[test]
    fn stack_grouped_batches_rejects_name_collision_with_output_column() {
        pyo3::Python::initialize();
        let per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = vec![
            (vec![KeyValue::Str("a".into())], xy_group_batch(vec![1.0], vec![10.0])),
        ];
        let err = stack_grouped_batches(
            "test",
            &per_group_batches,
            &["y".to_string()],
            &[DataType::Utf8],
        )
        .expect_err("groupby column 'y' collides with the output's own 'y' field");
        let msg = err.to_string();
        assert!(msg.contains('y'), "error must name the colliding column: {msg}");
        assert!(msg.contains("collides"), "error must explain the collision: {msg}");
        assert!(
            msg.contains("x") && msg.contains("y"),
            "error must name the transform's actual output columns so the message is \
             actionable from Python: {msg}"
        );
    }

    /// Task 2h review finding 3: the moved multi-column append loop
    /// (`group_cols.iter().zip(gtypes.iter()).enumerate()`) is where a
    /// transposed `gi`/dtype pairing would hide, and nothing exercised
    /// `gi > 0` through this helper. Two group columns of different dtypes
    /// (Utf8 then Int64) pins that each output column gets ITS OWN dtype and
    /// values, not the first dimension's for both.
    #[test]
    fn stack_grouped_batches_two_column_groupby_preserves_each_dtype() {
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = vec![
            (
                vec![KeyValue::Str("a".into()), KeyValue::Int(1)],
                xy_group_batch(vec![1.0, 2.0], vec![10.0, 20.0]),
            ),
            (
                vec![KeyValue::Str("b".into()), KeyValue::Int(2)],
                xy_group_batch(vec![3.0], vec![30.0]),
            ),
        ];
        let out = stack_grouped_batches(
            "test",
            &per_group_batches,
            &["g1".to_string(), "g2".to_string()],
            &[DataType::Utf8, DataType::Int64],
        )
        .unwrap();

        assert_eq!(out.num_rows(), 3);
        assert_eq!(out.schema().fields().len(), 4, "x, y, g1, g2");
        assert_eq!(out.schema().field(2).name(), "g1");
        assert_eq!(out.schema().field(2).data_type(), &DataType::Utf8);
        assert_eq!(out.schema().field(3).name(), "g2");
        assert_eq!(out.schema().field(3).data_type(), &DataType::Int64);

        let g1 = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(g1.value(0), "a");
        assert_eq!(g1.value(1), "a");
        assert_eq!(g1.value(2), "b");

        let g2 = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(g2.value(0), 1);
        assert_eq!(g2.value(1), 1);
        assert_eq!(g2.value(2), 2);
    }

    /// Task 2h review finding 2, re-homed by the batch-F close review (F1):
    /// `group_cols` repeating a name (e.g. a caller passing
    /// `groupby=["g", "g"]`) must refuse — the guard exists to establish "no
    /// duplicate field names in the output", not merely "no group column
    /// collides with a fit column". It lives in `group_partition_multi` (the
    /// shared partitioner) rather than in one stacking template, so
    /// `kde`/`bin`/`kde_2d`/`top_k`/`impute`/`outliers` are covered by the
    /// same eight lines that cover `smooth`/`robust`.
    #[test]
    fn group_partition_multi_rejects_duplicate_name_within_group_cols() {
        pyo3::Python::initialize();
        let batch = xy_group_batch(vec![1.0, 2.0], vec![10.0, 20.0]);
        // `x` exists in this batch, so the refusal is about the REPEAT, not
        // about a missing column.
        let err = group_partition_multi(&batch, &["x".to_string(), "x".to_string()], "test")
            .expect_err("groupby=['x', 'x'] must refuse rather than key on one column twice");
        let msg = err.to_string();
        assert!(msg.contains('x'), "error must name the repeated column: {msg}");
        assert!(msg.contains("repeated"), "error must explain the duplication: {msg}");
        assert!(msg.contains("test"), "error must stay in the caller's voice: {msg}");
    }

    /// The precedence half of the same guard: a repeated name that does not
    /// exist still reports "not found" first, exactly as it did before the
    /// repeat check moved into this function.
    #[test]
    fn group_partition_multi_missing_column_outranks_the_repeat_check() {
        pyo3::Python::initialize();
        let batch = xy_group_batch(vec![1.0], vec![10.0]);
        let err =
            group_partition_multi(&batch, &["ghost".to_string(), "ghost".to_string()], "test")
                .expect_err("a missing groupby column must still refuse");
        let msg = err.to_string();
        assert!(msg.contains("not found"), "missing column must report 'not found': {msg}");
    }
}
