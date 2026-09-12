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
//!   - **Partitioning** — [`group_partition_multi`], the shared partitioner,
//!     and [`validate_groupby_columns`], the request-level validator it is
//!     built on. Between them they own every policy that belongs to the
//!     groupby *request* itself: dtype support, the repeated-name refusal,
//!     first-appearance group order, and the null-key policy
//!     ([`NullKeyPolicy`]). Every grouped transform routes through the
//!     partitioner; the one exception, `data_window` — whose partitioning is
//!     a genuinely different algorithm (adjacent peer runs over *sorted*
//!     rows) — still shares the validator, so the three request-level
//!     refusals read identically everywhere (GH #179).
//!   - **Output-collision refusal** — [`refuse_groupby_output_collision`],
//!     called by every site that appends group columns to an output schema.
//!   - **Grouped-execution templates** — [`grouped_two_column_fit`] (the
//!     `RegressionData`/`LoessData` shape: fit per group, stack two Float64
//!     grids) and [`stack_grouped_batches`] (the `Smooth`/`Robust` shape:
//!     stack already-computed per-group batches). Both append the group
//!     columns and both refuse an output collision.
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

use crate::transform::context::{TransformContext, TransformWarning};

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
/// **The [`KeyValue::Null`] arm is no longer reachable from any production call
/// site** (GH #197). It emits an actual Arrow null for every dtype, which is
/// FA-9's contract and was live until the adjudicated null-key policy landed:
/// every transform that appends a group column now partitions under
/// [`NullKeyPolicy::Skip`], so no null key survives to be materialized, and the
/// two policies that DO keep null keys ([`NullKeyPolicy::Pool`]'s row-preserving
/// transforms) append no group column at all. The arm is kept rather than
/// deleted because it is the typed, non-panicking branch of a `match` whose
/// other arms `unreachable!()`, and because
/// `fa9_null_int_key_distinct_from_real_zero_key` (tests) still pins the FA-9
/// behavior it encodes — but a reader should not take it as describing
/// something the crate does today. Whether it survives is a filed follow-up
/// (task-3 report concern 2 / 7): either it becomes explicitly test-only or the
/// function's contract narrows to non-null keys.
///
/// FA-9's original rationale, for that decision: a null-keyed group stayed
/// DISTINCT from a genuine `0` / `false` / empty-string key all the way to the
/// downstream category scale. The emission is wire-neutral — Arrow nulls travel
/// over the C Data Interface unchanged — and callers building a non-nullable
/// output field must mark the groupby field nullable (see
/// [`groupby_field_nullable`]).
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

/// Result of [`group_partition_multi`]: the distinct groups in first-appearance
/// order, each paired with its member row indices, plus each group column's
/// original dtype.
///
/// The key/rows pairing is materialized rather than left as an order vector
/// plus a lookup map: every caller iterated the order and then looked each key
/// back up, so each one carried its own unreachable "missing group key in
/// index map" error arm. Pairing them here deletes that arm eleven times over.
#[derive(Debug)]
pub(crate) struct GroupPartition {
    /// One entry per distinct group key, in first-appearance order, paired
    /// with the input row indices carrying that key (ascending).
    pub groups: Vec<(Vec<KeyValue>, Vec<usize>)>,
    /// Each `group_cols` entry's original Arrow dtype, in the same order.
    pub dtypes: Vec<DataType>,
}

/// What a grouped transform does with rows whose `groupby` key contains a null
/// (GH #197), and — since the two must never drift apart — what its user-facing
/// warning says happened when EVERY row is null-keyed.
///
/// The adjudicated policy is [`NullKeyPolicy::Skip`]: a null-keyed row belongs
/// to no group, so an all-null key column yields the empty group set. The other
/// two variants are the policy's stated exemptions, spelled at the call site
/// that takes them rather than hidden in a hand-rolled loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NullKeyPolicy {
    /// Drop null-keyed rows from every group. With no surviving row the
    /// transform emits its (schema-correct) empty grouped output. This is the
    /// policy for every transform that appends the group key as an output
    /// column: a row with no key has no row to be written to.
    Skip,
    /// Drop them from the groups as [`NullKeyPolicy::Skip`] does, but the
    /// transform still computes a value for them from the whole batch —
    /// `Impute`'s whole-column fill and `Smooth`/`Robust`'s pooled whole-batch
    /// fit. Both contracts pre-date this policy and are load-bearing (row
    /// preservation; "the pooled fit IS the degraded-but-meaningful render"),
    /// so they are exempt from the empty-output arm and say so in the warning.
    SkipWithWholeBatchFallback,
    /// Keep null-keyed rows as one pooled null-keyed group. For the transforms
    /// whose contract is one output value per INPUT row (`data_stack`,
    /// `data_join_aggregate`) and which therefore append no group column:
    /// skipping a row would leave it silently holding a default value instead
    /// of a computed one, which is the silent-wrong outcome this campaign
    /// exists to remove.
    Pool,
}

impl NullKeyPolicy {
    /// Whether a null-containing key excludes its row from every group.
    fn skips_null_rows(self) -> bool {
        !matches!(self, NullKeyPolicy::Pool)
    }

    /// Whether the batch degraded to "no grouping happened", given how many of
    /// its rows carried a null ANYWHERE in the key and how many were null in
    /// EVERY dimension.
    ///
    /// The two counts differ only for a multi-column `groupby`, and only
    /// [`NullKeyPolicy::Pool`] can tell them apart: under the skipping
    /// policies a single null dimension already excludes the row, so a batch
    /// whose every row has *some* null produces no groups at all. Under `Pool`
    /// those rows still key on their non-null dimensions — `groupby=["a","b"]`
    /// with an all-null `a` and a varying `b` splits into one group per `b`,
    /// which is not a degrade and must not claim to be one. Only a key that is
    /// null in every dimension pools everything into the one unlabeled group
    /// the warning describes.
    fn degraded_to_no_grouping(self, any_null: usize, fully_null: usize, n_rows: usize) -> bool {
        match self {
            NullKeyPolicy::Skip | NullKeyPolicy::SkipWithWholeBatchFallback => any_null == n_rows,
            NullKeyPolicy::Pool => fully_null == n_rows,
        }
    }

    /// The user-facing clause describing what the transform did when EVERY row
    /// was null-keyed, for [`TransformWarning::GroupKeyAllNull`]'s text.
    fn all_null_disposition(self) -> &'static str {
        match self {
            NullKeyPolicy::Skip => "the grouped output is empty",
            NullKeyPolicy::SkipWithWholeBatchFallback => {
                "the whole batch was used as one ungrouped computation instead"
            }
            NullKeyPolicy::Pool => "all rows were treated as one unlabeled group",
        }
    }
}

/// Refuse a `groupby` request that names the same column twice.
///
/// A repeat is malformed independently of any data: a grouped transform
/// appends one output field per groupby dimension, so a repeat means two
/// identically-named Arrow fields, and `Schema::index_of` silently resolves
/// such a name to the FIRST match — every downstream reader of "the groupby
/// column" would see the wrong column's values.
///
/// Split out of [`validate_groupby_columns`] so the rule can also fire at the
/// earliest boundary that has the facts: the five PyO3 constructors that used
/// to carry their own `"…: duplicate groupby field 'g'"` loop now call this
/// instead. One rule, one message, still refused at construction where the
/// caller is — with the partition-time call covering the wire path, which has
/// no constructor to refuse at (GH #179 residual 2 and 3). `label` keeps the
/// error in the calling transform's voice.
pub(crate) fn refuse_repeated_groupby_names(label: &str, group_cols: &[String]) -> PyResult<()> {
    let mut seen: std::collections::HashSet<&str> =
        std::collections::HashSet::with_capacity(group_cols.len());
    for col_name in group_cols {
        if !seen.insert(col_name.as_str()) {
            return Err(repeated_groupby_name_error(label, col_name, group_cols));
        }
    }
    Ok(())
}

/// The repeated-name error text itself, so
/// [`refuse_repeated_groupby_names`] (which scans a whole request up front)
/// and [`validate_groupby_columns`] (which must interleave the check with its
/// per-column existence lookup, to keep "not found" first for an earlier
/// missing name) produce byte-identical messages.
fn repeated_groupby_name_error(label: &str, col_name: &str, group_cols: &[String]) -> PyErr {
    PyValueError::new_err(format!(
        "{label}: groupby column '{col_name}' is repeated in `groupby`; \
         each groupby column must have a distinct name (got groupby={group_cols:?})"
    ))
}

/// Validate a `groupby` request against a batch: every column exists, no name
/// repeats, every dtype is a supported key type. Returns each column's index
/// and dtype, in request order.
///
/// Split out of [`group_partition_multi`] so the one transform whose
/// partitioning is a genuinely different algorithm (`data_window` partitions
/// *sorted* rows into adjacent peer runs, not into whole groups) enforces the
/// same three rules in the same words instead of hand-rolling them.
///
/// **Repeated-name policy:** the same rule as
/// [`refuse_repeated_groupby_names`], in the same words (both build the text
/// with [`repeated_groupby_name_error`]), but interleaved with the existence
/// lookup rather than run up front. The interleaving is what keeps a MISSING
/// column reporting "not found", as it did before either guard lived here: for
/// `["ghost", "ghost"]` the first `ghost` fails `index_of` before its repeat
/// can be seen. It is a genuine precedence, not an impossibility — for
/// `["g", "g", "ghost"]` the repeat fires first and `ghost` is never looked
/// up. Each name is reported by whichever rule its own position reaches first.
/// `label` keeps the error in the calling transform's voice.
pub(crate) fn validate_groupby_columns(
    batch: &RecordBatch,
    group_cols: &[String],
    label: &str,
) -> PyResult<Vec<(usize, DataType)>> {
    let schema = batch.schema();
    let mut resolved: Vec<(usize, DataType)> = Vec::with_capacity(group_cols.len());

    for (i, col_name) in group_cols.iter().enumerate() {
        let gi = schema.index_of(col_name).map_err(|_| {
            PyValueError::new_err(format!("{label}: groupby column '{col_name}' not found"))
        })?;
        // Checked against the names already resolved, AFTER this one's
        // existence lookup, so an earlier missing column still reports "not
        // found" first.
        if group_cols[..i].contains(col_name) {
            return Err(repeated_groupby_name_error(label, col_name, group_cols));
        }
        let gtype = schema.field(gi).data_type().clone();
        if !is_groupby_supported_dtype(&gtype) {
            return Err(PyValueError::new_err(format!(
                "{label}: groupby column '{col_name}' has unsupported dtype {gtype:?}; \
                 supported: Utf8/LargeUtf8, Float64/Float32, \
                 Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
            )));
        }
        resolved.push((gi, gtype));
    }
    Ok(resolved)
}

/// Refuse a `groupby` name that collides with one of the transform's own
/// output column names.
///
/// Arrow does not reject duplicate schema field names, and `Schema::index_of`
/// silently resolves to the FIRST match, so appending a group column named like
/// an output column (grouping by a column literally called `x`, `density`, or
/// `mean`) makes every downstream reader of either column read the wrong one.
/// The error names both the colliding column and this transform's actual output
/// columns, so it is actionable from Python: a caller can rename the offending
/// data column, but cannot rename a transform's fixed output schema.
///
/// Extracted from [`stack_grouped_batches`], where batch F installed it for
/// `smooth`/`robust` alone; it is now called at every site that appends group
/// columns to an output schema (GH #179 residual 1).
///
/// **Call it BEFORE partitioning.** Every migrated site does, so a `groupby`
/// name that collides with an output column is refused on the same input
/// whether or not the column exists, whether or not the batch has rows, and
/// without first walking every row to build a partition that is about to be
/// thrown away. The visible consequence is a precedence: for
/// `groupby=["density"]` where the batch has no `density` column, `stat_kde`
/// reports the collision rather than "not found". Both are true and both
/// refuse; the collision is the more useful one, because it names the fixed
/// output schema the caller has to work around rather than a column they can
/// simply add.
///
/// [`stack_grouped_batches`] is the one caller that runs it AFTER partitioning,
/// and cannot do otherwise: the output field names it checks against are read
/// off the stacked per-group batches, which do not exist until the groups have
/// been computed. That is a structural exception, not a second policy.
pub(crate) fn refuse_groupby_output_collision(
    label: &str,
    group_cols: &[String],
    output_fields: &[String],
) -> PyResult<()> {
    for col_name in group_cols {
        if output_fields.iter().any(|f| f == col_name) {
            return Err(PyValueError::new_err(format!(
                "{label}: groupby column '{col_name}' collides with this transform's own output \
                 column '{col_name}' (output columns: {}); rename the data column before grouping",
                output_fields.join(", ")
            )));
        }
    }
    Ok(())
}

/// Partition a batch's rows by multiple groupby columns.
///
/// The single grouping entry point for every per-group transform: one
/// validation pass ([`validate_groupby_columns`]), first-appearance group
/// order, one null-key policy, dtype-preservation. When `group_cols` is empty,
/// every row keys to the empty `Vec`, so the result is ONE all-rows group —
/// `stat_outliers`' ungrouped path depends on exactly this (batch A migrated it
/// off an explicit all-rows insert), so tightening this to an empty partition
/// silently breaks it. Note the boundary case that follows from it: an empty
/// `group_cols` on a ZERO-ROW batch yields zero groups, not one empty group,
/// since the group is discovered from the rows. Transforms that must emit a
/// whole-batch row even for empty input keep their own explicit ungrouped arm.
///
/// **Null-key policy (GH #197):** decided by `null_keys` — see
/// [`NullKeyPolicy`], whose three variants are the adjudicated policy and its
/// two stated exemptions. When EVERY row of a non-empty batch is null-keyed,
/// this pushes one [`TransformWarning::GroupKeyAllNull`] onto `warn_ctx`,
/// naming the policy's own disposition text, so "the chart came back empty"
/// stops being indistinguishable from "the transform worked". Emitting it here
/// rather than at each caller is what makes the family answer with one voice.
///
/// **Repeated-name policy:** see [`validate_groupby_columns`]. Installed in the
/// shared validator, it covers every grouped transform in the crate: batch F
/// had installed it in [`stack_grouped_batches`] instead, which closed
/// `smooth`/`robust` and left every hand-rolled partitioner corrupting silently
/// (batch-F close review F1, then GH #179). The claim that once stood here —
/// that the residual was "a duplicated rule, not an open hole" because the
/// hand-rolled partitioners were unreachable with a repeat from Python — was
/// wrong: `fm.transform_pivot`, `fm.transform_aggregate`, and
/// `fm.transform_density` are typed public functions that emit those specs as
/// dicts, so `groupby=["g", "g"]` reached them and produced two identically
/// named output fields. The hole is closed by migration, not by documentation.
pub(crate) fn group_partition_multi(
    batch: &RecordBatch,
    group_cols: &[String],
    label: &str,
    null_keys: NullKeyPolicy,
    warn_ctx: &TransformContext,
) -> PyResult<GroupPartition> {
    let resolved = validate_groupby_columns(batch, group_cols, label)?;

    let n_rows = batch.num_rows();
    let mut index_of: std::collections::BTreeMap<Vec<KeyValue>, usize> =
        std::collections::BTreeMap::new();
    let mut groups: Vec<(Vec<KeyValue>, Vec<usize>)> = Vec::new();
    // Two counts, because the degrade condition differs by policy — see
    // `NullKeyPolicy::degraded_to_no_grouping`.
    let mut any_null_rows = 0_usize;
    let mut fully_null_rows = 0_usize;

    for row in 0..n_rows {
        let mut key: Vec<KeyValue> = Vec::with_capacity(group_cols.len());
        let mut has_null = false;
        let mut non_null_dims = 0_usize;
        for (col_idx, dtype) in &resolved {
            let arr = batch.column(*col_idx);
            let kv = groupby_key_at(arr.as_ref(), dtype, row).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{label}: internal error extracting groupby key at row {row}"
                ))
            })?;
            if matches!(kv, KeyValue::Null) {
                has_null = true;
                if null_keys.skips_null_rows() {
                    break;
                }
            } else {
                non_null_dims += 1;
            }
            key.push(kv);
        }
        if has_null {
            any_null_rows += 1;
            if null_keys.skips_null_rows() {
                // `non_null_dims` is not meaningful here: the dimension loop
                // `break`s at the first null, so later dimensions were never
                // read. The skipping policies key their degrade off
                // `any_null_rows` alone, which is exactly why they may break.
                continue;
            }
            if non_null_dims == 0 {
                fully_null_rows += 1;
            }
        }
        match index_of.get(&key) {
            Some(&gi) => groups[gi].1.push(row),
            None => {
                index_of.insert(key.clone(), groups.len());
                groups.push((key, vec![row]));
            }
        }
    }

    warn_if_group_keys_all_null(
        label,
        group_cols,
        null_keys,
        NullKeyCounts { any_null: any_null_rows, fully_null: fully_null_rows, n_rows },
        warn_ctx,
    );

    Ok(GroupPartition {
        groups,
        dtypes: resolved.into_iter().map(|(_, dt)| dt).collect(),
    })
}

/// How many of a batch's rows carried a null in their groupby key, counted
/// two ways because [`NullKeyPolicy::degraded_to_no_grouping`] needs both.
/// Bundled so [`warn_if_group_keys_all_null`]'s signature reads as one fact
/// about one batch rather than three loose `usize`s a caller could transpose.
pub(crate) struct NullKeyCounts {
    /// Rows with a null in ANY key dimension.
    pub any_null: usize,
    /// Rows whose key was null in EVERY dimension. Only meaningful under
    /// [`NullKeyPolicy::Pool`], which is the only policy that reads it — the
    /// skipping policies stop at a row's first null dimension.
    pub fully_null: usize,
    /// The batch's row count.
    pub n_rows: usize,
}

/// Emit GH #197's all-null-group-key warning if this request degraded to "no
/// grouping happened".
///
/// The single place that builds a [`TransformWarning::GroupKeyAllNull`], so
/// every grouped transform reports the condition in the same words with the
/// disposition its own [`NullKeyPolicy`] promises. [`group_partition_multi`]
/// calls it for the transforms it partitions; `data_window` calls it directly,
/// because its partitioning is a different algorithm (adjacent peer runs over
/// sorted rows) but its null-key exemption is the same `Pool` one its
/// row-preserving siblings take — and an exemption that does not warn is the
/// silent degrade this channel exists to remove.
///
/// Silent when no grouping was requested, when the batch is empty, or when any
/// row produced a usable key: nothing degraded in those cases.
pub(crate) fn warn_if_group_keys_all_null(
    label: &str,
    group_cols: &[String],
    null_keys: NullKeyPolicy,
    counts: NullKeyCounts,
    warn_ctx: &TransformContext,
) {
    if group_cols.is_empty() || counts.n_rows == 0 {
        return;
    }
    if !null_keys.degraded_to_no_grouping(counts.any_null, counts.fully_null, counts.n_rows) {
        return;
    }
    warn_ctx.warn(TransformWarning::GroupKeyAllNull {
        transform: label.to_string(),
        columns: group_cols.join(", "),
        disposition: null_keys.all_null_disposition().to_string(),
    });
}

/// Give an UNGROUPED request exactly one whole-batch group, even when the
/// batch has no rows.
///
/// [`group_partition_multi`] discovers groups from rows, so an empty
/// `group_cols` on a zero-row batch yields zero groups rather than one empty
/// group. For a reducing transform whose ungrouped contract is "collapse the
/// whole batch to ONE summary row" (`stat_aggregate`, `data_aggregate`), that
/// boundary is observable: zero rows in must still produce one row out, with
/// `count()` = 0 and the value statistics NaN. Called by exactly those
/// transforms, right after partitioning, so the rule is written once.
pub(crate) fn ensure_ungrouped_whole_batch_group(
    partition: &mut GroupPartition,
    group_cols: &[String],
) {
    if group_cols.is_empty() && partition.groups.is_empty() {
        partition.groups.push((Vec::new(), Vec::new()));
    }
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
    warn_ctx: &TransformContext,
    fit_group: impl Fn(&[usize]) -> PyResult<RecordBatch>,
) -> PyResult<RecordBatch> {
    use arrow::datatypes::{Field, Schema};

    refuse_groupby_output_collision(label, group_cols, &[as_.0.clone(), as_.1.clone()])?;
    let partition =
        group_partition_multi(batch, group_cols, label, NullKeyPolicy::Skip, warn_ctx)?;

    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();

    for (g, ixs) in &partition.groups {
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

    for (gi, (col_name, dtype)) in group_cols.iter().zip(partition.dtypes.iter()).enumerate() {
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
/// **Output-collision guard:** delegated to
/// [`refuse_groupby_output_collision`], which batch F first wrote inline here
/// and GH #179 hoisted so that all thirteen group-column append sites refuse
/// the same shape in the same words.
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
    refuse_groupby_output_collision(ctx, group_cols, &output_field_names)?;

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
mod tests;
