//! Data transform: Impute — fill missing (null/NaN) values.
//!
//! Supports imputation methods: value (constant), mean, median, min, max.
//!
//! ## Groupby + key/keyvals (batch-D M3, F-L05-04)
//!
//! `impute` is row-preserving, unlike the concat-and-rebuild template
//! `smooth.rs`/`regression_data.rs` use for grouped fits: every input row
//! keeps its position, so grouping here is partition-then-scatter (the
//! `data_window` shape) rather than partition-fit-concat. Rows partition via
//! [`group_key::group_partition_multi`] (empty `groupby` yields exactly one
//! implicit group holding every row, in original order — which is why the
//! ungrouped path below is byte-identical to the pre-groupby implementation:
//! it is the SAME per-group computation, just over the one group that
//! happens to be the whole batch). A row whose groupby key is null belongs
//! to no group under that helper's null-key policy; since this transform
//! must still emit a row for it, such a row falls back to the whole-column
//! statistic instead (a case that never arises when `groupby` is unset).
//! The same whole-column statistic is also the fallback for a group whose
//! own cleaned value set is empty (every value in that group is null/NaN):
//! without it, `mean`/`median` would silently fill with `0.0` and `min`/`max`
//! would fill with `±f64::INFINITY`, either of which is worse than falling
//! back to a wider window — `0.0` is a real plottable value and `±inf`
//! drops the group's rows from the render entirely.
//!
//! `key=`/`keyvals=` implement Vega's keyed-imputation semantics: within
//! each group, the key universe (`keyvals` if given, else the distinct `key`
//! values observed across the whole batch, in first-appearance order) minus
//! the key values actually present in that group is the set of missing
//! keys, and each gets one appended row — the group's own columns, the
//! missing key, and the group's own fill value; every other column is null.
//! `key` without `groupby` is Vega's "no groupby means one implicit group"
//! rule: the whole batch is that one group. Appended rows land after all
//! original rows, in group-then-key order (both deterministic). `keyvals`
//! is meaningless without `key` (there is no universe to speak of), so it
//! is a refusal, not a silent no-op, to pass one without the other.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::transform::group_key::{
    group_partition_multi, groupby_key_at, is_groupby_supported_dtype, materialize_groupby_col,
    KeyValue,
};
use crate::transform::numeric_util::{clean_float64_values, coerce_to_float64};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImputeSpec {
    /// Column to impute.
    pub field: String,
    /// Imputation method: "value", "mean", "median", "min", "max".
    #[serde(default = "default_method")]
    pub method: String,
    /// Constant value for method="value".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value: Option<f64>,
    /// Group-by columns (impute within groups).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Key column: within each group, missing values of this column (against
    /// the key universe — see `keyvals`) get an appended row.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub key: Option<String>,
    /// The key universe. Requires `key` (a refusal, not a silent no-op, if
    /// `key` is unset). When absent, the universe is the distinct `key`
    /// values observed across the whole batch. Each value is typed per the
    /// key column's own dtype at apply time: an integral JSON float (e.g.
    /// `1.0`) coerces onto an Int/UInt key column (M1's "coerce numerics,
    /// raise on the rest" posture — a non-integral float against an
    /// Int/UInt key column stays a typed refusal, since that coercion would
    /// be lossy). Any other dtype mismatch (e.g. a string against a numeric
    /// key column) is a typed refusal, not a silent drop.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub keyvals: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

/// The single source of truth for `Impute`'s free-string `method`
/// vocabulary. Read by [`validate_method`] (the enforcement site), its error
/// message, and `registry::entry_for_kind("impute")`'s `methods` list — R1
/// design finding 1, mirroring the `smooth::METHODS` precedent.
pub(crate) const METHODS: &[&str] = &["value", "mean", "median", "min", "max"];

fn default_method() -> String {
    "value".into()
}

/// Per-group context shared by the row-fill scatter and key-row generation:
/// one `group_partition_multi` call's groups (order + row-index map), their
/// columns' original dtypes, each group's own computed fill value, and the
/// groupby columns' schema indices. Bundled because every downstream step
/// needs the same five pieces of that one call.
struct GroupContext<'a> {
    order: &'a [Vec<KeyValue>],
    idx_map: &'a BTreeMap<Vec<KeyValue>, Vec<usize>>,
    dtypes: &'a [DataType],
    col_indices: &'a [usize],
    fill_values: &'a [f64],
}

/// The key column's identity, bundled for the same reason as
/// [`GroupContext`]: `generate_missing_key_rows` needs the key column's
/// name, schema index, and dtype together, and bundling them keeps the
/// function's argument count under clippy's `too_many_arguments` threshold
/// while letting the caller pass `key_name` once instead of re-deriving it
/// from `spec.key` inside the callee.
struct KeyMeta<'a> {
    name: &'a str,
    col_idx: usize,
    dtype: &'a DataType,
}

pub(crate) fn apply(spec: &ImputeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    validate_method(&spec.method)?;
    if spec.keyvals.is_some() && spec.key.is_none() {
        return Err(PyValueError::new_err(
            "data_impute: keyvals requires key=; the key universe is meaningless without a key column",
        ));
    }

    let schema = batch.schema();

    let field_col_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_impute: column '{}' not found", spec.field))
    })?;

    // Any numeric dtype is accepted; the imputed column is emitted as Float64
    // (the fill value is a float, so the column widens by construction).
    let col = coerce_to_float64(batch.column(field_col_idx), "data_impute", &spec.field)?;

    let n_rows = batch.num_rows();

    // Partition rows for a per-group fill value (M3, F-L05-04). See the
    // module doc for why the ungrouped path is byte-identical by
    // construction rather than a separate branch.
    let groupby: &[String] = spec.groupby.as_deref().unwrap_or(&[]);
    let (group_order, group_idx_map, group_dtypes) =
        group_partition_multi(batch, groupby, "data_impute")?;
    let groupby_col_indices: Vec<usize> = groupby
        .iter()
        .map(|name| {
            schema.index_of(name).map_err(|_| {
                PyValueError::new_err(format!("data_impute: groupby column '{name}' not found"))
            })
        })
        .collect::<PyResult<_>>()?;

    // The whole-column statistic, computed once up front. It serves two
    // fallback cases below: a row whose groupby key is null (belongs to no
    // group) and a group whose own cleaned value set is empty. When even the
    // whole column has no usable values (a pathological empty-batch input),
    // `degenerate_default` reproduces the pre-groupby implementation's
    // degenerate return (`0.0` for mean/median, `±f64::INFINITY` for
    // min/max) — unchanged from before this transform supported grouping.
    let whole_column_fill = fill_value_for(&col, None, &spec.method, spec.value)
        .unwrap_or_else(|| degenerate_default(&spec.method));

    let mut row_fill = vec![f64::NAN; n_rows];
    let mut assigned = vec![false; n_rows];
    let mut group_fill_values: Vec<f64> = Vec::with_capacity(group_order.len());
    for key in &group_order {
        let rows = group_idx_map.get(key).ok_or_else(|| {
            PyValueError::new_err(format!("data_impute: missing group key {key:?} in index map"))
        })?;
        // A group whose own cleaned value set is empty (every value in the
        // group is null/NaN) falls back to the whole-column statistic
        // instead of `fill_value_for`'s degenerate return — see the module
        // doc. Appended key rows for this group read the same value via
        // `group_fill_values`/`groups.fill_values`.
        let fv = fill_value_for(&col, Some(rows), &spec.method, spec.value)
            .unwrap_or(whole_column_fill);
        group_fill_values.push(fv);
        for &r in rows {
            row_fill[r] = fv;
            assigned[r] = true;
        }
    }
    // A row whose groupby key contains a null value belongs to no group
    // (`group_partition_multi`'s null-key policy). It still needs a fill
    // value to keep this transform row-preserving, so it falls back to the
    // same whole-column statistic the ungrouped path would use. This never
    // triggers when `groupby` is empty (every row is in the one implicit
    // group) or has no null keys.
    for (r, is_assigned) in assigned.iter().enumerate() {
        if !is_assigned {
            row_fill[r] = whole_column_fill;
        }
    }

    let mut imputed: Vec<f64> = Vec::with_capacity(n_rows);
    for (i, &fill) in row_fill.iter().enumerate() {
        if col.is_null(i) {
            imputed.push(fill);
        } else {
            let v = col.value(i);
            imputed.push(if v.is_nan() { fill } else { v });
        }
    }

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns[field_col_idx] = Arc::new(Float64Array::from(imputed));

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields[field_col_idx] = Field::new(&spec.field, DataType::Float64, false);

    // `key=`/`keyvals=` (Vega semantics — see the module doc).
    if let Some(key_name) = &spec.key {
        let key_col_idx = schema.index_of(key_name).map_err(|_| {
            PyValueError::new_err(format!("data_impute: key column '{key_name}' not found"))
        })?;
        if key_col_idx == field_col_idx {
            return Err(PyValueError::new_err(format!(
                "data_impute: key column '{key_name}' cannot be the same as the imputed field '{}'",
                spec.field
            )));
        }
        if groupby_col_indices.contains(&key_col_idx) {
            return Err(PyValueError::new_err(format!(
                "data_impute: key column '{key_name}' cannot also be a groupby column"
            )));
        }
        let key_dtype = schema.field(key_col_idx).data_type().clone();
        if !is_groupby_supported_dtype(&key_dtype) {
            return Err(PyValueError::new_err(format!(
                "data_impute: key column '{key_name}' has unsupported dtype {key_dtype:?}; \
                 supported: Utf8/LargeUtf8, Float64/Float32, \
                 Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
            )));
        }

        let groups = GroupContext {
            order: &group_order,
            idx_map: &group_idx_map,
            dtypes: &group_dtypes,
            col_indices: &groupby_col_indices,
            fill_values: &group_fill_values,
        };
        let key_meta = KeyMeta { name: key_name, col_idx: key_col_idx, dtype: &key_dtype };
        if let Some(appended_cols) =
            generate_missing_key_rows(spec, batch, &schema, field_col_idx, &key_meta, &groups)?
        {
            for (idx, field) in fields.iter_mut().enumerate() {
                if idx == field_col_idx || idx == key_col_idx || groupby_col_indices.contains(&idx) {
                    continue;
                }
                // Every other column may carry a null in an appended row.
                *field = Field::new(field.name(), field.data_type().clone(), true);
            }
            for (idx, appended_col) in appended_cols.into_iter().enumerate() {
                columns[idx] =
                    arrow::compute::concat(&[columns[idx].as_ref(), appended_col.as_ref()])
                        .map_err(|e| PyValueError::new_err(format!("data_impute: {e}")))?;
            }
        }
    }

    let out_schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_impute: {e}")))
}

fn validate_method(method: &str) -> PyResult<()> {
    if METHODS.contains(&method) {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "data_impute: unknown method '{method}'; expected {}",
        METHODS.join("|"),
    )))
}

/// The fill value for one method, restricted to `rows` when given (a group's
/// row indices) or the whole column when `None`. With `rows = None`, this
/// iterates the column in the exact same order (`0..len`) the pre-groupby
/// implementation always did, so the ungrouped path's arithmetic — including
/// summation order for `mean` — is unchanged.
///
/// Returns `None` when `method` is `mean`/`median`/`min`/`max` and the
/// restricted value set is empty (every value at `rows` is null/NaN) — the
/// caller is expected to fall back to a wider window (see `apply`'s
/// `whole_column_fill`) rather than accept this function's own degenerate
/// default. `method = "value"` never returns `None`, since the constant
/// doesn't depend on the data. `method` is assumed already validated by
/// [`validate_method`] (called once at `apply`'s entry), so this function is
/// infallible.
fn fill_value_for(
    col: &Float64Array,
    rows: Option<&[usize]>,
    method: &str,
    value: Option<f64>,
) -> Option<f64> {
    match method {
        "value" => Some(value.unwrap_or(0.0)),
        "mean" => {
            let vals = clean_float64_values(col, rows);
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        }
        "median" => {
            let mut vals = clean_float64_values(col, rows);
            if vals.is_empty() {
                None
            } else {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = vals.len();
                Some(if n % 2 == 1 {
                    vals[n / 2]
                } else {
                    0.5 * (vals[n / 2 - 1] + vals[n / 2])
                })
            }
        }
        "min" => {
            let vals = clean_float64_values(col, rows);
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
            }
        }
        "max" => {
            let vals = clean_float64_values(col, rows);
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
            }
        }
        other => unreachable!(
            "data_impute: method '{other}' should have been refused by validate_method at apply's entry"
        ),
    }
}

/// `fill_value_for`'s degenerate default for the pathological case where
/// even the whole column has no usable values — unchanged from the
/// pre-groupby implementation's behavior. Never reached for `method =
/// "value"` (that arm of `fill_value_for` never returns `None`).
fn degenerate_default(method: &str) -> f64 {
    match method {
        "mean" | "median" => 0.0,
        "min" => f64::INFINITY,
        "max" => f64::NEG_INFINITY,
        other => unreachable!(
            "data_impute: method '{other}' should have been refused by validate_method at apply's entry"
        ),
    }
}

/// An integral JSON float (e.g. `1.0`, produced by `json.dumps` for a Python
/// `float` that happens to be whole-valued) narrowed to `i64`, or `None` if
/// it has a fractional part or doesn't fit in range. Used to let `keyvals`
/// coerce a whole-valued float onto an Int key column (M1's "coerce
/// numerics, raise on the rest" posture) without silently truncating a
/// genuinely fractional or out-of-range value. The caller is responsible for
/// further range-checking against the specific signed-integer width.
/// The upper bound is strict (`f < -(i64::MIN as f64)`) to prevent saturation
/// before the range check; `i64::MAX as f64` equals 2^63, one above the actual max.
fn integral_f64_as_i64(f: f64) -> Option<i64> {
    (f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f < -(i64::MIN as f64))
        .then_some(f as i64)
}

/// Same as [`integral_f64_as_i64`], narrowed to `u64` (non-negative only).
/// The caller is responsible for further range-checking against the specific
/// unsigned-integer width.
/// The upper bound is strict (`f < 18446744073709551616.0`, i.e. 2^64) to prevent
/// saturation before the range check; `u64::MAX as f64` equals 2^64, one above the actual max.
fn integral_f64_as_u64(f: f64) -> Option<u64> {
    (f.is_finite() && f.fract() == 0.0 && (0.0..18446744073709551616.0).contains(&f))
        .then_some(f as u64)
}

/// Range bounds for a signed integer dtype, returning (min, max) as i64.
/// Called by `keyvalue_from_json` to range-check a keyval before narrowing.
fn signed_int_bounds(dt: &DataType) -> Option<(i64, i64)> {
    match dt {
        DataType::Int8 => Some((i8::MIN as i64, i8::MAX as i64)),
        DataType::Int16 => Some((i16::MIN as i64, i16::MAX as i64)),
        DataType::Int32 => Some((i32::MIN as i64, i32::MAX as i64)),
        DataType::Int64 => Some((i64::MIN, i64::MAX)),
        _ => None,
    }
}

/// Range bounds for an unsigned integer dtype, returning (min, max) as u64.
/// Called by `keyvalue_from_json` to range-check a keyval before narrowing.
fn unsigned_int_bounds(dt: &DataType) -> Option<(u64, u64)> {
    match dt {
        DataType::UInt8 => Some((u8::MIN as u64, u8::MAX as u64)),
        DataType::UInt16 => Some((u16::MIN as u64, u16::MAX as u64)),
        DataType::UInt32 => Some((u32::MIN as u64, u32::MAX as u64)),
        DataType::UInt64 => Some((u64::MIN, u64::MAX)),
        _ => None,
    }
}

/// Convert one `keyvals` JSON entry to a [`KeyValue`] typed per the key
/// column's own dtype. A JSON value that doesn't fit that dtype's family
/// (e.g. a string against a numeric key column, or a non-integral float
/// against an Int/UInt key column — the genuinely lossy case) is a typed
/// refusal.
fn keyvalue_from_json(v: &serde_json::Value, dt: &DataType, key_name: &str) -> PyResult<KeyValue> {
    let incompatible = || {
        PyValueError::new_err(format!(
            "data_impute: keyvals value {v} is not compatible with key column '{key_name}' dtype {dt:?}"
        ))
    };
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => {
            v.as_str().map(|s| KeyValue::Str(s.to_string())).ok_or_else(incompatible)
        }
        DataType::Float64 => {
            v.as_f64().map(|f| KeyValue::Float(f.to_bits())).ok_or_else(incompatible)
        }
        // Float32 key values are widened to f64 through an `f32` round-trip
        // in `group_key::groupby_key_at` — matching that exactly here (not
        // just `as_f64()`) is required, since a JSON value that isn't
        // exactly representable in f32 would otherwise never equal any
        // value actually present in the group.
        DataType::Float32 => v
            .as_f64()
            .map(|f| KeyValue::Float((f as f32 as f64).to_bits()))
            .ok_or_else(incompatible),
        // M1's "coerce numerics, raise on the rest": an integral JSON float
        // (e.g. `1.0`, from `df["month"].unique().to_list()` off a float
        // column) is accepted; a non-integral float is the genuinely lossy
        // case and stays refused. Range-check against the target dtype's width,
        // both in the fast path (as_i64) and the integral-float fallback, to
        // prevent silent wrapping when narrowing to smaller integer types.
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            let (min_bound, max_bound) =
                signed_int_bounds(dt).unwrap_or_else(|| unreachable!(
                    "signed_int_bounds should always succeed for Int8/Int16/Int32/Int64"
                ));
            let val = v
                .as_i64()
                .or_else(|| v.as_f64().and_then(integral_f64_as_i64))
                .ok_or_else(incompatible)?;
            if val >= min_bound && val <= max_bound {
                Ok(KeyValue::Int(val))
            } else {
                Err(incompatible())
            }
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let (min_bound, max_bound) =
                unsigned_int_bounds(dt).unwrap_or_else(|| unreachable!(
                    "unsigned_int_bounds should always succeed for UInt8/UInt16/UInt32/UInt64"
                ));
            let val = v
                .as_u64()
                .or_else(|| v.as_f64().and_then(integral_f64_as_u64))
                .ok_or_else(incompatible)?;
            if val >= min_bound && val <= max_bound {
                Ok(KeyValue::UInt(val))
            } else {
                Err(incompatible())
            }
        }
        DataType::Boolean => v.as_bool().map(|b| KeyValue::Bool(b as u8)).ok_or_else(incompatible),
        other => Err(PyValueError::new_err(format!(
            "data_impute: key column '{key_name}' has unsupported dtype {other:?}"
        ))),
    }
}

/// Build the appended rows for `key=`/`keyvals=`, one per (group, missing
/// key) pair in group-then-key order, or `None` when nothing is missing.
/// Returns one `ArrayRef` per schema column, each of the same length M —
/// ready for the caller to `concat` onto the original per-column arrays.
fn generate_missing_key_rows(
    spec: &ImputeSpec,
    batch: &RecordBatch,
    schema: &SchemaRef,
    field_col_idx: usize,
    key: &KeyMeta,
    groups: &GroupContext,
) -> PyResult<Option<Vec<ArrayRef>>> {
    let key_arr = batch.column(key.col_idx);

    // The key universe: `keyvals` if given (first occurrence wins on a
    // duplicate), else the distinct key values observed across the WHOLE
    // batch, in first-appearance order.
    let mut universe: Vec<KeyValue> = Vec::new();
    let mut seen: BTreeSet<KeyValue> = BTreeSet::new();
    if let Some(raw_vals) = &spec.keyvals {
        for v in raw_vals {
            let kv = keyvalue_from_json(v, key.dtype, key.name)?;
            if seen.insert(kv.clone()) {
                universe.push(kv);
            }
        }
    } else {
        for row in 0..batch.num_rows() {
            if let Some(kv) = groupby_key_at(key_arr.as_ref(), key.dtype, row) {
                if matches!(kv, KeyValue::Null) {
                    continue;
                }
                if seen.insert(kv.clone()) {
                    universe.push(kv);
                }
            }
        }
    }
    if universe.is_empty() {
        return Ok(None);
    }

    let mut appended_group_keys: Vec<Vec<KeyValue>> = Vec::new();
    let mut appended_key_values: Vec<KeyValue> = Vec::new();
    let mut appended_field_values: Vec<f64> = Vec::new();

    for (gi, group_key) in groups.order.iter().enumerate() {
        let rows = groups.idx_map.get(group_key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_impute: missing group key {group_key:?} in index map"
            ))
        })?;
        let mut present: BTreeSet<KeyValue> = BTreeSet::new();
        for &row in rows {
            if let Some(kv) = groupby_key_at(key_arr.as_ref(), key.dtype, row) {
                if !matches!(kv, KeyValue::Null) {
                    present.insert(kv);
                }
            }
        }
        for kv in &universe {
            if !present.contains(kv) {
                appended_group_keys.push(group_key.clone());
                appended_key_values.push(kv.clone());
                appended_field_values.push(groups.fill_values[gi]);
            }
        }
    }

    let m = appended_field_values.len();
    if m == 0 {
        return Ok(None);
    }

    let mut appended_cols: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == field_col_idx {
            appended_cols.push(Arc::new(Float64Array::from(appended_field_values.clone())));
        } else if idx == key.col_idx {
            let wrapped: Vec<Vec<KeyValue>> =
                appended_key_values.iter().cloned().map(|k| vec![k]).collect();
            appended_cols.push(
                materialize_groupby_col(&wrapped, 0, key.dtype).map_err(PyValueError::new_err)?,
            );
        } else if let Some(gi) = groups.col_indices.iter().position(|&gci| gci == idx) {
            appended_cols.push(
                materialize_groupby_col(&appended_group_keys, gi, &groups.dtypes[gi])
                    .map_err(PyValueError::new_err)?,
            );
        } else {
            appended_cols.push(arrow::array::new_null_array(field.data_type(), m));
        }
    }
    Ok(Some(appended_cols))
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "Impute")]
#[derive(Debug, Clone)]
pub(crate) struct PyImpute(pub(crate) TransformSpec);

#[pymethods]
impl PyImpute {
    #[new]
    #[pyo3(signature = (field, *, method = "value", value = None, name = None))]
    fn new(field: String, method: &str, value: Option<f64>, name: Option<String>) -> Self {
        PyImpute(TransformSpec::Impute(ImputeSpec {
            field,
            method: method.into(),
            value,
            groupby: None,
            key: None,
            keyvals: None,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Impute(s) => format!("Impute(field='{}', method='{}')", s.field, s.method),
            _ => "Impute(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn spec(
        field: &str,
        method: &str,
        value: Option<f64>,
        groupby: Option<Vec<&str>>,
        key: Option<&str>,
        keyvals: Option<Vec<serde_json::Value>>,
    ) -> ImputeSpec {
        ImputeSpec {
            field: field.into(),
            method: method.into(),
            value,
            groupby: groupby.map(|g| g.into_iter().map(String::from).collect()),
            key: key.map(String::from),
            keyvals,
            name: None,
        }
    }

    #[test]
    fn impute_with_mean() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
                None,
                Some(5.0),
            ]))],
        )
        .unwrap();

        let s = spec("x", "mean", None, None, None, None);
        let out = apply(&s, &batch).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        // mean(1,3,5) = 3.0
        assert_eq!(x.value(0), 1.0);
        assert_eq!(x.value(1), 3.0); // imputed
        assert_eq!(x.value(2), 3.0);
        assert_eq!(x.value(3), 3.0); // imputed
        assert_eq!(x.value(4), 5.0);
    }

    #[test]
    fn impute_with_constant() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]))],
        )
        .unwrap();

        let s = spec("x", "value", Some(99.0), None, None, None);
        let out = apply(&s, &batch).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(x.value(1), 99.0);
    }

    // ── M3 groupby coverage (F-L05-04) ───────────────────────────────────

    /// The finding's own fixture: `transform_impute(groupby=["g"])` used to
    /// fill BOTH groups' gaps with the global mean (65). Group "a"'s own
    /// mean(100,120)=110; group "b"'s own mean(10,30)=20.
    #[test]
    fn f_l05_04_per_group_mean_not_global_blend() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("sales", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "a", "b", "b", "b"])),
                Arc::new(Float64Array::from(vec![
                    Some(100.0),
                    None,
                    Some(120.0),
                    Some(10.0),
                    None,
                    Some(30.0),
                ])),
            ],
        )
        .unwrap();

        let s = spec("sales", "mean", None, Some(vec!["g"]), None, None);
        let out = apply(&s, &batch).unwrap();
        let sales = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (0..6).map(|i| sales.value(i)).collect::<Vec<_>>(),
            vec![100.0, 110.0, 120.0, 10.0, 20.0, 30.0],
            "group 'a' fills with its own mean 110, group 'b' with its own mean 20 — \
             NOT the global blended mean 65"
        );
        // Row-preserving: same row count and the "g" column is untouched.
        assert_eq!(out.num_rows(), 6);
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(g.value(0), "a");
        assert_eq!(g.value(5), "b");
    }

    #[test]
    fn ungrouped_impute_byte_identical_to_single_global_group() {
        // Same values as the F-L05-04 fixture but ungrouped: the global mean
        // over all 4 non-null values (100,120,10,30) is 65.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "sales",
            DataType::Float64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(100.0),
                None,
                Some(120.0),
                Some(10.0),
                None,
                Some(30.0),
            ]))],
        )
        .unwrap();

        let s = spec("sales", "mean", None, None, None, None);
        let out = apply(&s, &batch).unwrap();
        let sales = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (0..6).map(|i| sales.value(i)).collect::<Vec<_>>(),
            vec![100.0, 65.0, 120.0, 10.0, 65.0, 30.0],
        );
        assert!(!out.schema().field(0).is_nullable(), "field column stays non-nullable");
    }

    #[test]
    fn int_groupby_column_partitions_correctly() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![Some(10.0), None, Some(100.0), None])),
            ],
        )
        .unwrap();

        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let out = apply(&s, &batch).unwrap();
        let v = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.value(1), 10.0, "group 1's only value fills group 1's gap");
        assert_eq!(v.value(3), 100.0, "group 2's only value fills group 2's gap");
    }

    /// A row whose groupby key is null belongs to no group; it still gets a
    /// fill (row-preserving), falling back to the whole-column statistic.
    #[test]
    fn null_groupby_key_row_falls_back_to_whole_column_statistic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("a"), None])),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None])),
            ],
        )
        .unwrap();

        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let out = apply(&s, &batch).unwrap();
        let v = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group "a" has no gaps. The null-key row (excluded from any group)
        // falls back to the whole column's mean: mean(10, 20) = 15.
        assert_eq!(v.value(2), 15.0);
    }

    // ── key=/keyvals= generation (§4.3) ──────────────────────────────────

    fn key_fixture() -> RecordBatch {
        // Group 1: months {1, 2}; group 2: months {2, 3}. Universe
        // (batch-wide first-appearance) = [1, 2, 3].
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Float64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 30.0, 100.0, 300.0])),
            ],
        )
        .unwrap()
    }

    fn col_f64(out: &RecordBatch, idx: usize) -> Vec<f64> {
        out.column(idx).as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
    }

    /// The "g" groupby column stays Int64 (pass-through, never coerced).
    fn col_i64(out: &RecordBatch, idx: usize) -> Vec<i64> {
        out.column(idx).as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec()
    }

    /// Group 1 is missing month 3 (appended fill = mean(10,30) = 20); group 2
    /// is missing month 1 (appended fill = mean(100,300) = 200). Wrong group
    /// pairing (e.g. using the other group's fill) would fail this.
    #[test]
    fn missing_key_rows_generated_per_group_with_that_groups_fill() {
        let batch = key_fixture();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 6, "4 original + 2 appended (one per group's missing month)");

        let g = col_i64(&out, 0);
        let month = col_f64(&out, 1);
        let sales = col_f64(&out, 2);
        let mut rows: Vec<(i64, i64, f64)> =
            (0..6).map(|i| (g[i], month[i] as i64, sales[i])).collect();
        rows.sort_by_key(|&(g, m, _)| (g, m));
        assert_eq!(
            rows,
            vec![
                (1, 1, 10.0),
                (1, 2, 30.0),
                (1, 3, 20.0),  // appended: group 1's own fill
                (2, 1, 200.0), // appended: group 2's own fill
                (2, 2, 100.0),
                (2, 3, 300.0),
            ]
        );
        // Appended rows widen every non-field/non-groupby/non-key column's
        // nullability; here there are none besides those three, so nothing
        // else to check, but the field/groupby/key columns themselves must
        // stay non-null throughout.
        assert!(!out.schema().field(2).is_nullable(), "imputed field column stays non-nullable");
    }

    /// `keyvals=` overrides the batch-wide universe: a value absent from the
    /// data entirely still generates one appended row per group.
    #[test]
    fn keyvals_override_expands_the_universe_beyond_observed_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Float64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 100.0, 200.0])),
            ],
        )
        .unwrap();

        // Without keyvals: both groups already have {1, 2} — the whole
        // observed universe — so nothing is missing.
        let s_default = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out_default = apply(&s_default, &batch).unwrap();
        assert_eq!(out_default.num_rows(), 4, "no keyvals override: nothing is missing");

        // With keyvals=[1,2,3]: month 3 is now in the universe, missing from
        // BOTH groups, so 2 rows get appended.
        let s_override = spec(
            "sales",
            "mean",
            None,
            Some(vec!["g"]),
            Some("month"),
            Some(vec![serde_json::json!(1.0), serde_json::json!(2.0), serde_json::json!(3.0)]),
        );
        let out_override = apply(&s_override, &batch).unwrap();
        assert_eq!(out_override.num_rows(), 6, "keyvals=[1,2,3] adds month 3 for both groups");
        let month = col_f64(&out_override, 1);
        let sales = col_f64(&out_override, 2);
        let mut appended: Vec<f64> = (4..6).map(|i| sales[i]).collect();
        appended.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(appended, vec![15.0, 150.0], "group 1 fill=mean(10,20)=15, group 2 fill=mean(100,200)=150");
        assert!(month[4..6].iter().all(|&m| m == 3.0));
    }

    /// `key` without `groupby`: Vega's "no groupby means one implicit group"
    /// rule — the whole batch is treated as a single group.
    #[test]
    fn key_without_groupby_treats_whole_batch_as_one_group() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Float64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();

        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1.0), serde_json::json!(2.0), serde_json::json!(3.0)]),
        );
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 3, "month 3 is missing from the one implicit group");
        let month = col_f64(&out, 0);
        let sales = col_f64(&out, 1);
        assert_eq!(month[2], 3.0);
        assert_eq!(sales[2], 15.0, "the implicit group's own fill: mean(10,20)=15");
    }

    /// The key column's own dtype (Utf8 here) is preserved on appended rows,
    /// not coerced or corrupted.
    #[test]
    fn key_column_dtype_preserved_on_appended_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Utf8, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(StringArray::from(vec!["Jan", "Feb", "Feb", "Mar"])),
                Arc::new(Float64Array::from(vec![10.0, 30.0, 100.0, 300.0])),
            ],
        )
        .unwrap();

        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 6);
        let month = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let g = col_i64(&out, 0);
        let sales = col_f64(&out, 2);
        let mut appended: Vec<(i64, String, f64)> = (4..6)
            .map(|i| (g[i], month.value(i).to_string(), sales[i]))
            .collect();
        appended.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            appended,
            vec![(1, "Mar".to_string(), 20.0), (2, "Jan".to_string(), 200.0)],
            "appended rows carry a real Utf8 category, not a numeral or corrupted value"
        );
    }

    /// Ungrouped, no key: identical output shape and schema to the
    /// pre-M3 implementation (no appended rows, no schema widening).
    #[test]
    fn ungrouped_no_key_is_byte_identical_shape() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Float64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]))],
        )
        .unwrap();
        let s = spec("x", "mean", None, None, None, None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 3, "no rows appended when key is unset");
        assert_eq!(out.num_columns(), 1);
        assert!(!out.schema().field(0).is_nullable());
    }

    #[test]
    fn keyvals_type_mismatch_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec(
            "sales",
            "mean",
            None,
            Some(vec!["g"]),
            Some("month"),
            Some(vec![serde_json::json!("not-a-number")]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("data_impute"), "err: {msg}");
        assert!(msg.contains("keyvals"), "err: {msg}");
        assert!(msg.contains("'month'"), "err must name the key column: {msg}");
    }

    #[test]
    fn key_same_as_field_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("sales"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("cannot be the same as the imputed field"), "err: {msg}");
    }

    #[test]
    fn key_same_as_groupby_column_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("g"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("cannot also be a groupby column"), "err: {msg}");
    }

    #[test]
    fn unsupported_key_dtype_refuses() {
        pyo3::Python::initialize();
        // Date32 is not a supported groupby/key dtype.
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("d", DataType::Date32, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(arrow::array::Date32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, Some(vec!["g"]), Some("d"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("unsupported dtype"), "err: {msg}");
        assert!(msg.contains("'d'"), "err: {msg}");
    }

    #[test]
    fn unknown_method_refuses_even_on_empty_batch() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(Vec::<f64>::new()))]).unwrap();
        let s = spec("x", "bogus", None, None, None, None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("unknown method 'bogus'"), "err: {msg}");
    }

    #[test]
    fn boolean_key_column_supported() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("flag", DataType::Boolean, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // Universe = {true, false}; group 1 only has `true`, so `false` is
        // appended for it; group 2 only has `false`, so `true` is appended.
        let s = spec("v", "mean", None, Some(vec!["g"]), Some("flag"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 4);
    }

    // ── Round-2 fixes (rust-quality-reviewer required fixes 1-6) ─────────

    /// Fix 1: a group whose own cleaned value set is empty (every value is
    /// null/NaN) must fall back to the WHOLE-COLUMN statistic instead of
    /// `fill_value_for`'s degenerate return (`0.0` for mean/median, `±inf`
    /// for min/max), for every data-dependent method.
    #[test]
    fn empty_group_falls_back_to_whole_column_statistic_for_every_method() {
        let make_batch = || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("sales", DataType::Float64, true),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                    Arc::new(Float64Array::from(vec![Some(100.0), Some(120.0), None, None])),
                ],
            )
            .unwrap()
        };
        // Group "b" has zero non-null values, so its OWN fill would be
        // degenerate (0.0 for mean/median, ±inf for min/max) without the
        // fallback. The whole-column statistic over [100, 120] is the
        // correct fallback for every method.
        let expected: &[(&str, f64)] =
            &[("mean", 110.0), ("median", 110.0), ("min", 100.0), ("max", 120.0)];
        for &(method, fallback) in expected {
            let batch = make_batch();
            let s = spec("sales", method, None, Some(vec!["g"]), None, None);
            let out = apply(&s, &batch).unwrap();
            let sales = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(out.num_rows(), 4, "method {method}: row-preserving even for an all-null group");
            assert_eq!(sales.value(0), 100.0, "method {method}: group a's real value untouched");
            assert_eq!(sales.value(1), 120.0, "method {method}: group a's real value untouched");
            assert_eq!(
                sales.value(2),
                fallback,
                "method {method}: group b (all-null) must fall back to the whole-column \
                 statistic {fallback}, not a degenerate 0.0/±inf"
            );
            assert_eq!(
                sales.value(3),
                fallback,
                "method {method}: group b's second row same fallback"
            );
        }
    }

    /// Fix 1 (continued): the same fallback must be the value carried by
    /// an APPENDED key row generated for the empty group, not the
    /// degenerate `fill_value_for` return.
    #[test]
    fn empty_group_fallback_also_used_for_that_groups_appended_key_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
                Arc::new(Int64Array::from(vec![1i64, 2, 1])),
                Arc::new(Float64Array::from(vec![Some(100.0), Some(120.0), None])),
            ],
        )
        .unwrap();
        // Universe (batch-wide) = {1, 2}; group "b" only has month 1, so
        // month 2 is missing and gets appended for it.
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 4, "3 original + 1 appended (group b's missing month 2)");
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let month = col_i64(&out, 1);
        let sales = col_f64(&out, 2);
        let appended = (0..4)
            .find(|&i| g.value(i) == "b" && month[i] == 2)
            .expect("appended row for group b, month 2");
        assert_eq!(
            sales[appended], 110.0,
            "group b's appended row must use the whole-column fallback (110), not 0.0"
        );
    }

    /// Fix 2: a Float32 key column's `keyvals` entries must round-trip
    /// through `f32` the same way `group_key::groupby_key_at` does, or
    /// every keyvals entry is judged missing from every group (spurious
    /// duplicate rows).
    #[test]
    fn float32_key_column_with_fractional_keyvals_matches_observed_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Float32, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Float32Array::from(vec![0.1f32, 0.2, 0.2, 0.3])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )
        .unwrap();
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(0.1), serde_json::json!(0.2), serde_json::json!(0.3)]),
        );
        let out = apply(&s, &batch).unwrap();
        assert_eq!(
            out.num_rows(),
            4,
            "keyvals=[0.1,0.2,0.3] already covers every observed Float32 month; a \
             Float32-bits mismatch would spuriously append one row per keyvals entry"
        );
    }

    /// Fix 4: M1's "coerce numerics, raise on the rest" — an integral JSON
    /// float against an Int key column is a numeric, not a dtype mismatch.
    #[test]
    fn keyvals_integral_float_coerces_onto_int_key_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1.0), serde_json::json!(2.0), serde_json::json!(3.0)]),
        );
        let out = apply(&s, &batch).unwrap();
        assert_eq!(
            out.num_rows(),
            3,
            "1.0/2.0 coerce onto Int64 and match the two observed rows; 3.0 is appended"
        );
    }

    /// Fix 4 (continued): the genuinely lossy case — a non-integral float
    /// against an Int key column — must stay a typed refusal.
    #[test]
    fn keyvals_non_integral_float_against_int_key_column_still_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        let s = spec("sales", "mean", None, None, Some("month"), Some(vec![serde_json::json!(1.5)]));
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
    }

    /// Fix 3: `keyvals` without `key` is a refusal, not a silent no-op.
    #[test]
    fn keyvals_without_key_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec(
            "sales",
            "mean",
            None,
            Some(vec!["g"]),
            None,
            Some(vec![serde_json::json!(1.0)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("keyvals requires key"), "err: {msg}");
    }

    /// Fix 1 (range check), part 1: an integer keyval that overflows the key
    /// column's specific integer width must refuse with an incompatible error,
    /// not silently wrap. `Int32` has max = 2,147,483,647; `3000000000` is
    /// out of range, so it must refuse.
    #[test]
    fn keyvals_out_of_range_int32_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int32, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Int32Array::from(vec![1i32, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // 3000000000 is out of range for Int32 (max 2147483647).
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1), serde_json::json!(3000000000i64)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
        assert!(msg.contains("Int32"), "error must name the dtype 'Int32': {msg}");
    }

    /// Fix 1 (range check), part 2: an integer keyval that would wrap to an
    /// existing key's value when narrowed (the worse shape: it manufactures
    /// a duplicate of an already-present key) must refuse, not silently
    /// duplicate. `Int8` has max = 127; `257 % 256 = 1`, so 257 wrapped as
    /// Int8 is 1, which is already in the key column and would create a
    /// duplicate row.
    #[test]
    fn keyvals_wrap_onto_existing_key_int8_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int8, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Int8Array::from(vec![1i8, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // 257 is out of range for Int8 (max 127). It would wrap to 1 (257 % 256 = 1 as u8,
        // reinterpreted as signed), creating a duplicate of the existing key 1.
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1), serde_json::json!(2), serde_json::json!(257i64)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
        assert!(msg.contains("Int8"), "error must name the dtype 'Int8': {msg}");
    }

    /// Fix 6 (extra-column coverage): §4.3's "every other column is null"
    /// on an appended row, previously unreached by any fixture in either
    /// suite.
    #[test]
    fn appended_row_leaves_unrelated_column_null_and_widens_its_nullability() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
            Field::new("note", DataType::Utf8, false), // neither field, key, nor groupby
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![1i64, 2, 2, 3])),
                Arc::new(Float64Array::from(vec![10.0, 30.0, 100.0, 300.0])),
                Arc::new(StringArray::from(vec!["p", "q", "r", "s"])),
            ],
        )
        .unwrap();
        // Universe (batch-wide) = {1, 2, 3}. Group 1 (months {1,2}) is
        // missing 3; group 2 (months {2,3}) is missing 1.
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 6, "4 original + 2 appended");
        assert!(
            out.schema().field(3).is_nullable(),
            "'note' must widen to nullable once a row is appended"
        );
        let note = out.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        assert!(note.is_null(4), "appended row 0 leaves 'note' null");
        assert!(note.is_null(5), "appended row 1 leaves 'note' null");
        for i in 0..4 {
            assert!(!note.is_null(i), "original row {i}'s 'note' must stay untouched");
        }
    }

    /// Finding 1 (cycle 3): the 64-bit boundary (2^63 for signed, 2^64 for unsigned)
    /// must refuse, not saturate to the max representable value. The hardening tightens
    /// the upper bound from `f <= i64::MAX as f64` (which equals 2^63, one above actual max)
    /// to `f < -(i64::MIN as f64)` (which equals 2^63). `2^63` itself is a keyval user might
    /// produce from Python via `2**63`, and it must refuse with an incompatible error,
    /// not silently key `i64::MAX`.
    #[test]
    fn keyvals_2pow63_on_int64_key_column_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // 2^63 is 9223372036854775808.0, which is out of range for Int64 (max 9223372036854775807).
        // Expressed as JSON, this becomes the literal number 9223372036854775808.
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1), serde_json::json!(9223372036854775808.0)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
        assert!(msg.contains("Int64"), "error must name the dtype 'Int64': {msg}");
    }
}
