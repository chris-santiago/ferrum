//! Centralized Arrow numeric/string column casting helpers.
//!
//! Renderer code reaches for Arrow columns at three different layers
//! (`scale_resolve` for domain inference, `draw` for per-row coordinate
//! extraction, and per-mark renderers for value lookups). Each layer
//! previously hand-rolled its own dtype dispatch chain over Float/Int/UInt
//! variants — five copies of the same downcast ladder across `draw.rs`
//! (`col_as_f64`, `col_as_str`) and `scale_resolve.rs` (`column_min_max_f64`,
//! `numeric_extent`, `distinct_values_in_order`).
//!
//! This module consolidates that surface so adding a new Arrow primitive
//! is a one-line edit, dtype support is uniform across the renderer, and
//! the F16 widening (color type inference) lands by changing one call site
//! instead of five.

use arrow::array::{
    Array, BooleanArray,
    Date32Array, Date64Array,
    DurationMillisecondArray, DurationMicrosecondArray,
    DurationNanosecondArray, DurationSecondArray,
    Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array,
    LargeStringArray, StringArray,
    TimestampMillisecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::record_batch::RecordBatch;

use super::RenderError;

/// Multiplier to convert a Duration tick in the given `unit` to nanoseconds.
///
/// Both `col_as_f64` and `min_max_f64` normalize Duration values to nanoseconds
/// so that the two functions always agree numerically (T0-closeout / D-DTYPE-1).
/// This is the single source of truth for that scale factor; a third divergence
/// is unrepresentable because both callers delegate here.
///
/// Values on live Python paths are pre-cast to `Timestamp(ms)` by
/// `_coerce.normalize_for_rust` before crossing the CDI boundary, so these arms
/// are unreachable in production. The helper is retained for internal callers and
/// will be deleted together with its call sites in T6.2.
#[inline]
fn duration_unit_scale(unit: &TimeUnit) -> f64 {
    match unit {
        TimeUnit::Nanosecond  => 1.0,
        TimeUnit::Microsecond => 1_000.0,
        TimeUnit::Millisecond => 1_000_000.0,
        TimeUnit::Second      => 1_000_000_000.0,
    }
}

/// Category label for a null value in an ordinal/nominal column (FA-9).
///
/// A null row in a categorical column becomes its own distinct category under
/// this label, so a null groupby key stays separate from a real `0`/`false`/`""`
/// value all the way to the category scale. `distinct_values_in_order` and
/// `col_as_ordinal_category_str` both use this constant, so the per-row band
/// lookup matches the domain entry.
pub(crate) const NULL_CATEGORY: &str = "null";

/// Authoritative predicate: true iff `dtype` is in the set that
/// `col_as_f64` can read and `min_max_f64` can range over.
///
/// This is the single source of truth for "can this dtype be treated as a
/// continuous/quantitative value?".  `is_numeric`, `col_as_f64`, and
/// `min_max_f64` all branch from this predicate so they can never claim
/// different coverage (RSUP-02 / D-DTYPE-1).
///
/// Covered set: Float64/32, Int64/32/16/8, UInt64/32/16/8,
/// Timestamp(any unit), Duration(any unit), Date32, Date64.
///
/// Note: on live Python paths Date32/Date64/Duration never arrive — they
/// are pre-cast to Timestamp(ms) by `_coerce.normalize_for_rust` before
/// crossing the CDI boundary. The predicate covers them so that internal
/// callers and future code paths are consistent with `col_as_f64`.
pub(crate) fn supported_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float64 | DataType::Float32
            | DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8
            | DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Date32
            | DataType::Date64
    )
}

/// True for Arrow dtypes that should route as continuous/quantitative when
/// inferring an encoding type from data alone.
///
/// Used by `scale_resolve::build_color_scale` to validate that a column
/// marked quantitative or temporal is actually a numeric dtype, replacing
/// the pre-F16 narrow `Float64|UInt64` check.
///
/// Delegates to [`supported_numeric_dtype`] so the coverage is always in
/// sync with `col_as_f64` and `min_max_f64` (D-DTYPE-1).
pub(crate) fn is_numeric(dtype: &DataType) -> bool {
    supported_numeric_dtype(dtype)
}

/// Read any supported numeric Arrow column as `Vec<Option<f64>>`, preserving
/// null positions. Returns `Err(ScaleResolutionFailed)` for unsupported
/// dtypes — matches the prior `draw::col_as_f64` semantics exactly.
///
/// Supported dtypes: Float64/32, Int64/32/16/8, UInt64/32/16/8,
/// TimestampMillisecond.
pub(crate) fn col_as_f64(batch: &RecordBatch, field: &str) -> Result<Vec<Option<f64>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    macro_rules! collect_as {
        ($t:ty) => {{
            let a = col.as_any().downcast_ref::<$t>().expect("dtype matched");
            return Ok(a.iter().map(|v| v.map(|x| x as f64)).collect());
        }};
    }
    match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().expect("Float64");
            return Ok(a.iter().collect());
        }
        DataType::Float32 => collect_as!(Float32Array),
        DataType::Int64 => collect_as!(Int64Array),
        DataType::Int32 => collect_as!(Int32Array),
        DataType::Int16 => collect_as!(Int16Array),
        DataType::Int8 => collect_as!(Int8Array),
        DataType::UInt64 => collect_as!(UInt64Array),
        DataType::UInt32 => collect_as!(UInt32Array),
        DataType::UInt16 => collect_as!(UInt16Array),
        DataType::UInt8 => collect_as!(UInt8Array),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => collect_as!(TimestampNanosecondArray),
        DataType::Timestamp(TimeUnit::Microsecond, _) => collect_as!(TimestampMicrosecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => collect_as!(TimestampMillisecondArray),
        DataType::Timestamp(TimeUnit::Second, _) => collect_as!(TimestampSecondArray),
        // Duration and Date arms below are unreachable on live Python paths:
        // `_coerce.normalize_for_rust` pre-casts both to Timestamp(ms) before
        // the CDI boundary crossing. They remain for correctness of internal
        // callers and are tracked for removal in T6.2.
        //
        // Duration values are normalized to nanoseconds via `duration_unit_scale`
        // so this function and `min_max_f64` always agree numerically (T0-closeout).
        DataType::Duration(TimeUnit::Nanosecond) => {
            let scale = duration_unit_scale(&TimeUnit::Nanosecond);
            let a = col.as_any().downcast_ref::<DurationNanosecondArray>().expect("dtype matched");
            Ok(a.iter().map(|v| v.map(|x| x as f64 * scale)).collect())
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let scale = duration_unit_scale(&TimeUnit::Microsecond);
            let a = col.as_any().downcast_ref::<DurationMicrosecondArray>().expect("dtype matched");
            Ok(a.iter().map(|v| v.map(|x| x as f64 * scale)).collect())
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let scale = duration_unit_scale(&TimeUnit::Millisecond);
            let a = col.as_any().downcast_ref::<DurationMillisecondArray>().expect("dtype matched");
            Ok(a.iter().map(|v| v.map(|x| x as f64 * scale)).collect())
        }
        DataType::Duration(TimeUnit::Second) => {
            let scale = duration_unit_scale(&TimeUnit::Second);
            let a = col.as_any().downcast_ref::<DurationSecondArray>().expect("dtype matched");
            Ok(a.iter().map(|v| v.map(|x| x as f64 * scale)).collect())
        }
        // Date32: days since epoch → epoch-milliseconds (matches JS Date / Altair convention).
        DataType::Date32 => {
            let a = col.as_any().downcast_ref::<Date32Array>().expect("dtype matched");
            Ok(a.iter().map(|v| v.map(|d| d as f64 * 86_400_000.0)).collect())
        }
        // Date64: milliseconds since epoch → f64 directly.
        DataType::Date64 => {
            let a = col.as_any().downcast_ref::<Date64Array>().expect("dtype matched");
            Ok(a.iter().map(|v| v.map(|d| d as f64)).collect())
        }
        other => Err(RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{other:?}"),
            context: None,
        }),
    }
}

/// Read a Utf8 column as `Vec<Option<String>>`. Returns
/// `Err(ScaleResolutionFailed)` for any other dtype.
///
/// Strings should already have been normalized to Utf8 by
/// `prepare::normalize_string_views` upstream; this function only handles
/// the post-normalization case, matching prior `draw::col_as_str`.
pub(crate) fn col_as_str(batch: &RecordBatch, field: &str) -> Result<Vec<Option<String>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect())
    } else {
        Err(RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?} (expected Utf8)", col.data_type()),
            context: None,
        })
    }
}

/// Read any column that can serve as an ordinal category key as
/// `Vec<Option<String>>`.
///
/// This is the correct helper to use when a scale is `ScaleKind::Ordinal` and
/// you need per-row category strings for `to_pixel_str`. Unlike `col_as_str`,
/// it accepts non-string numeric dtypes by stringifying them — using the same
/// formatting that `distinct_values_in_order` applies when building the ordinal
/// domain, so the per-row strings always match the domain entries.
///
/// Supported dtypes and their formatting:
/// - Utf8 / LargeUtf8 → value as-is (identity, same as `col_as_str`)
/// - Int64/32/16/8, UInt64/32/16/8 → native `.to_string()` (e.g. `2000i64` → `"2000"`)
/// - Float64/32 → `format!("{v}")` when the value has a non-zero fractional
///   part (e.g. `1.5` → `"1.5"`), or integer-formatted otherwise (e.g.
///   `2000.0` → `"2000"`), so integer-valued float columns behave identically
///   to Int64 ordinal columns.
/// - Boolean → `"true"` / `"false"` (same as `distinct_values_in_order`).
///
/// Returns `Err(UnsupportedDtype)` only for dtypes that cannot be sensibly
/// turned into category strings (timestamps, durations, etc.).
pub(crate) fn col_as_ordinal_category_str(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<Option<String>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;

    // Macro for integer types: native value → .to_string()
    macro_rules! collect_int_as_str {
        ($t:ty) => {
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                return Ok(a.iter().map(|v| v.map(|x| x.to_string())).collect());
            }
        };
    }

    // Utf8 / LargeUtf8 — identity (most common, checked first).
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect());
    }

    // Integer types — .to_string() matches distinct_values_in_order exactly.
    collect_int_as_str!(Int64Array);
    collect_int_as_str!(Int32Array);
    collect_int_as_str!(Int16Array);
    collect_int_as_str!(Int8Array);
    collect_int_as_str!(UInt64Array);
    collect_int_as_str!(UInt32Array);
    collect_int_as_str!(UInt16Array);
    collect_int_as_str!(UInt8Array);

    // Float types — delegate to the shared helper so this function and
    // `distinct_values_in_order` always produce matching strings.
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(a.iter().map(|v| v.map(float_as_ordinal_str)).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        // Widen to f64 before formatting to reuse the same helper.
        return Ok(a.iter().map(|v| v.map(|x| float_as_ordinal_str(x as f64))).collect());
    }

    // Boolean — "true" / "false".
    if let Some(a) = col.as_any().downcast_ref::<arrow::array::BooleanArray>() {
        return Ok(a.iter().map(|v| v.map(|b| b.to_string())).collect());
    }

    Err(RenderError::UnsupportedDtype {
        field: field.to_string(),
        dtype: format!("{:?} (cannot convert to ordinal category string)", col.data_type()),
        context: None,
    })
}

/// Distinct categories for a **positional** (x:N / y:N) ordinal domain, with a
/// null row surfaced as its own [`NULL_CATEGORY`] category (FA-9).
///
/// This is the positional counterpart to [`distinct_values_in_order`]. The base
/// helper drops nulls — correct for color/shape/legend domains, where a null
/// must NOT consume a palette slot or legend entry. Positional axes are
/// different: a null groupby key (e.g. an aggregate's null group) must occupy
/// its own band so the bar renders as a distinct category, satisfying the FA-9
/// invariant that a null group and a real `0`/`false`/`""` group are two
/// categories. Null-free columns are byte-identical to [`distinct_values_in_order`].
pub(crate) fn distinct_positional_categories(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<String>, RenderError> {
    let mut out = distinct_values_in_order(batch, field)?;
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    // Append the null category (first-appearance order keeps it after real
    // values, matching how the per-row mapper places it). `distinct_values_in_order`
    // never emits NULL_CATEGORY for real values (no dtype stringifies to "null"
    // except a literal Utf8 "null", which is an accepted edge-case merge).
    if col.null_count() > 0 && !out.iter().any(|s| s == NULL_CATEGORY) {
        out.push(NULL_CATEGORY.to_string());
    }
    Ok(out)
}

/// Per-row category strings for a **positional** (x:N / y:N) axis, mapping a
/// null row to [`NULL_CATEGORY`] so it lands in the null band produced by
/// [`distinct_positional_categories`] (FA-9).
///
/// The positional counterpart to [`col_as_ordinal_category_str`], which keeps
/// nulls as `None` (correct for color/shape, where a null row draws no mark and
/// no legend entry).
pub(crate) fn col_as_positional_category_str(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<Option<String>>, RenderError> {
    let per_row = col_as_ordinal_category_str(batch, field)?;
    Ok(per_row
        .into_iter()
        .map(|o| o.or_else(|| Some(NULL_CATEGORY.to_string())))
        .collect())
}

/// `(min, max)` over a numeric Arrow column, including non-finite values.
/// Returns `Err(message)` for unsupported dtypes — matches the prior
/// `scale_resolve::column_min_max_f64` semantics exactly. Caller is
/// responsible for wrapping the `Err` into a typed `RenderError`.
///
/// **Caveat**: this function does not filter NaN. For Float columns the
/// `f64::min`/`max` IEEE-754 semantics treat NaN as "absent" (non-NaN wins),
/// which is the prior behavior. An all-NaN column returns `(+INF, -INF)` —
/// caller should validate.
pub(crate) fn min_max_f64(col: &dyn Array) -> Result<(f64, f64), String> {
    macro_rules! min_max_int {
        ($t:ty, $native:ty) => {{
            let a = col.as_any().downcast_ref::<$t>().expect("dtype matched");
            let min = a.iter().flatten().fold(<$native>::MAX, <$native>::min) as f64;
            let max = a.iter().flatten().fold(<$native>::MIN, <$native>::max) as f64;
            return Ok((min, max));
        }};
    }
    match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().expect("Float64");
            let min = a.iter().flatten().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);
            let max = a.iter().flatten().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
            Ok((min, max))
        }
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>().expect("Float32");
            let min = a.iter().flatten().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min) as f64;
            let max = a.iter().flatten().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max) as f64;
            Ok((min, max))
        }
        DataType::Int64 => min_max_int!(Int64Array, i64),
        DataType::Int32 => min_max_int!(Int32Array, i32),
        DataType::Int16 => min_max_int!(Int16Array, i16),
        DataType::Int8 => min_max_int!(Int8Array, i8),
        DataType::UInt64 => min_max_int!(UInt64Array, u64),
        DataType::UInt32 => min_max_int!(UInt32Array, u32),
        DataType::UInt16 => min_max_int!(UInt16Array, u16),
        DataType::UInt8 => min_max_int!(UInt8Array, u8),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => min_max_int!(TimestampNanosecondArray, i64),
        DataType::Timestamp(TimeUnit::Microsecond, _) => min_max_int!(TimestampMicrosecondArray, i64),
        DataType::Timestamp(TimeUnit::Millisecond, _) => min_max_int!(TimestampMillisecondArray, i64),
        DataType::Timestamp(TimeUnit::Second, _) => min_max_int!(TimestampSecondArray, i64),
        // Duration and Date arms: unreachable on live Python paths because
        // `_coerce.normalize_for_rust` pre-casts them to Timestamp(ms) before
        // crossing the CDI boundary. Covered here so that `min_max_f64` agrees
        // with `col_as_f64` and `supported_numeric_dtype` (D-DTYPE-1 / T6.2).
        //
        // Duration values are normalized to nanoseconds via `duration_unit_scale`
        // so this function and `col_as_f64` always agree numerically (T0-closeout).
        DataType::Duration(TimeUnit::Nanosecond) => {
            let scale = duration_unit_scale(&TimeUnit::Nanosecond);
            let a = col.as_any().downcast_ref::<DurationNanosecondArray>().expect("dtype matched");
            let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64 * scale;
            let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64 * scale;
            Ok((min, max))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let scale = duration_unit_scale(&TimeUnit::Microsecond);
            let a = col.as_any().downcast_ref::<DurationMicrosecondArray>().expect("dtype matched");
            let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64 * scale;
            let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64 * scale;
            Ok((min, max))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let scale = duration_unit_scale(&TimeUnit::Millisecond);
            let a = col.as_any().downcast_ref::<DurationMillisecondArray>().expect("dtype matched");
            let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64 * scale;
            let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64 * scale;
            Ok((min, max))
        }
        DataType::Duration(TimeUnit::Second) => {
            let scale = duration_unit_scale(&TimeUnit::Second);
            let a = col.as_any().downcast_ref::<DurationSecondArray>().expect("dtype matched");
            let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64 * scale;
            let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64 * scale;
            Ok((min, max))
        }
        DataType::Date32 => {
            let a = col.as_any().downcast_ref::<Date32Array>().expect("Date32");
            let min = a.iter().flatten().fold(i32::MAX, i32::min) as f64 * 86_400_000.0;
            let max = a.iter().flatten().fold(i32::MIN, i32::max) as f64 * 86_400_000.0;
            Ok((min, max))
        }
        DataType::Date64 => {
            let a = col.as_any().downcast_ref::<Date64Array>().expect("Date64");
            let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64;
            let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64;
            Ok((min, max))
        }
        other => Err(format!("unsupported column dtype: {other:?}")),
    }
}

/// `(min, max)` over the *finite* values of a numeric column. Returns
/// `None` when no finite values are present or when the dtype is
/// unsupported — matches the prior `scale_resolve::numeric_extent`
/// semantics, but supports the full numeric dtype set (callers that
/// previously routed Int32/Float32/etc. to a categorical path will
/// continue to do so until F16 widens the routing predicate; this
/// function is unreachable for those dtypes today).
pub(crate) fn finite_min_max_f64(col: &dyn Array) -> Option<(f64, f64)> {
    let (lo, hi) = min_max_f64(col).ok()?;
    if lo.is_finite() && hi.is_finite() && lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// Format a float value as an ordinal category string.
///
/// Whole-number floats (e.g. `2000.0`) are formatted as integers (`"2000"`)
/// so that an integer-valued Float64 ordinal column produces the same domain
/// strings as an equivalent Int64 column. Fractional values (e.g. `1.5`)
/// keep their decimal representation (`"1.5"`). Non-finite values fall back
/// to `format!("{v}")` (i.e. `"inf"`, `"-inf"`, `"NaN"`).
///
/// This helper is the single source of truth for float-ordinal stringification
/// and is shared between `distinct_values_in_order` (domain building) and
/// `col_as_ordinal_category_str` (per-row lookup), so the two can never drift.
#[inline]
fn float_as_ordinal_str(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Enumerate distinct values in encounter order, stringified, from a column
/// whose dtype is one of `Utf8`, `LargeUtf8`, `Int*`, `UInt*`, `Float32`,
/// `Float64`, or `Boolean`. Returns `Err(UnsupportedDtype)` for any other
/// dtype.
///
/// Stringification rules (must agree exactly with `col_as_ordinal_category_str`):
/// - Utf8 / LargeUtf8 → identity
/// - Int*/UInt* → `.to_string()`
/// - Float64/32 → `float_as_ordinal_str` (integer-formatted for whole values)
/// - Boolean → `"true"` / `"false"`
pub(crate) fn distinct_values_in_order(batch: &RecordBatch, field: &str) -> Result<Vec<String>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();
    let mut push = |s: String| {
        if seen.insert(s.clone()) {
            out.push(s);
        }
    };
    // Macro to avoid repeating the downcast-and-stringify loop for each integer type.
    macro_rules! push_int {
        ($t:ty) => {
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                for v in a.iter().flatten() { push(v.to_string()); }
                return Ok(out);
            }
        };
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        // Polars produces LargeUtf8 (LargeStringArray) for string columns.
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        // Float64 ordinal columns: integer-valued floats format as integers
        // (e.g. 2000.0 → "2000") to match Int64 domain strings. Fractional
        // values keep their decimal form (e.g. 1.5 → "1.5").
        for v in a.iter().flatten() { push(float_as_ordinal_str(v)); }
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        // Same rule for Float32; widen to f64 first to reuse the same helper.
        for v in a.iter().flatten() { push(float_as_ordinal_str(v as f64)); }
    } else {
        // Non-Int64 integer variants — pandas/pyarrow users commonly have Int32, UInt8, etc.
        // Each arm stringifies the native value so downstream encoding treats them as nominal.
        push_int!(Int32Array);
        push_int!(Int16Array);
        push_int!(Int8Array);
        push_int!(UInt64Array);
        push_int!(UInt32Array);
        push_int!(UInt16Array);
        push_int!(UInt8Array);
        return Err(RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?} (cannot enumerate distinct values)", col.data_type()),
            context: None,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Date32Array, Date64Array,
        DurationMillisecondArray, DurationMicrosecondArray,
        DurationNanosecondArray, DurationSecondArray,
        Float64Array, Int32Array, Int8Array, StringArray, UInt8Array, UInt32Array,
        TimestampMillisecondArray,
    };
    use arrow::datatypes::{Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn batch_f64(name: &str, values: Vec<Option<f64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    #[test]
    fn col_as_f64_handles_float64_with_nulls() {
        let b = batch_f64("x", vec![Some(1.0), None, Some(3.0)]);
        let out = col_as_f64(&b, "x").unwrap();
        assert_eq!(out, vec![Some(1.0), None, Some(3.0)]);
    }

    #[test]
    fn col_as_f64_widens_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), None]))],
        ).unwrap();
        let out = col_as_f64(&b, "x").unwrap();
        assert_eq!(out, vec![Some(1.0), Some(2.0), None]);
    }

    #[test]
    fn col_as_f64_errors_on_string() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        ).unwrap();
        let err = col_as_f64(&b, "s").unwrap_err();
        assert!(format!("{err}").contains("unsupported dtype"), "{err}");
    }

    #[test]
    fn col_as_str_reads_utf8() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]))],
        ).unwrap();
        let out = col_as_str(&b, "s").unwrap();
        assert_eq!(out, vec![Some("a".into()), None, Some("b".into())]);
    }

    #[test]
    fn min_max_f64_float64() {
        let b = batch_f64("x", vec![Some(3.0), Some(1.0), Some(2.0)]);
        let col = b.column(0).as_ref();
        let (lo, hi) = min_max_f64(col).unwrap();
        assert_eq!((lo, hi), (1.0, 3.0));
    }

    #[test]
    fn min_max_f64_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(5), Some(-3), Some(8)]))],
        ).unwrap();
        let (lo, hi) = min_max_f64(b.column(0).as_ref()).unwrap();
        assert_eq!((lo, hi), (-3.0, 8.0));
    }

    #[test]
    fn finite_min_max_f64_filters_nan() {
        let b = batch_f64("x", vec![Some(1.0), Some(f64::NAN), Some(3.0)]);
        let result = finite_min_max_f64(b.column(0).as_ref());
        // Per IEEE-754 f64::min/max, NaN is absorbed; non-NaN values dominate.
        assert_eq!(result, Some((1.0, 3.0)));
    }

    #[test]
    fn finite_min_max_f64_all_nan_returns_none() {
        let b = batch_f64("x", vec![Some(f64::NAN), Some(f64::NAN)]);
        let result = finite_min_max_f64(b.column(0).as_ref());
        // All-NaN: min_max returns (+INF, -INF); finite_min_max should None.
        assert_eq!(result, None);
    }

    #[test]
    fn distinct_values_in_order_strings() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["b", "a", "b", "c", "a"]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "s").unwrap();
        assert_eq!(out, vec!["b", "a", "c"]);
    }

    #[test]
    fn is_numeric_covers_supported_dtypes() {
        assert!(is_numeric(&DataType::Float64));
        assert!(is_numeric(&DataType::Float32));
        assert!(is_numeric(&DataType::Int8));
        assert!(is_numeric(&DataType::UInt32));
        assert!(is_numeric(&DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None)));
        assert!(!is_numeric(&DataType::Utf8));
        assert!(!is_numeric(&DataType::Boolean));
    }

    // ── distinct_values_in_order: integer-type columns ───────────────────────

    #[test]
    fn distinct_values_in_order_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![
                Some(3), Some(1), Some(3), Some(2), Some(1),
            ]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["3", "1", "2"]);
    }

    #[test]
    fn distinct_values_in_order_uint8() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::UInt8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt8Array::from(vec![Some(10u8), Some(20u8), Some(10u8)]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["10", "20"]);
    }

    #[test]
    fn distinct_values_in_order_int8() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int8Array::from(vec![Some(-1i8), Some(2i8), Some(-1i8)]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["-1", "2"]);
    }

    #[test]
    fn distinct_values_in_order_uint32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::UInt32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt32Array::from(vec![Some(100u32), Some(200u32), Some(100u32)]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["100", "200"]);
    }

    // ── col_as_f64: Duration and Date types ──────────────────────────────────

    #[test]
    fn col_as_f64_duration_nanosecond() {
        // 1 second = 1_000_000_000 nanoseconds → f64 1_000_000_000.0
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Duration(TimeUnit::Nanosecond), true),
        ]));
        let arr = DurationNanosecondArray::from(vec![Some(1_000_000_000i64), None, Some(2_000_000_000i64)]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let out = col_as_f64(&b, "d").unwrap();
        assert_eq!(out, vec![Some(1_000_000_000.0), None, Some(2_000_000_000.0)]);
    }

    #[test]
    fn col_as_f64_duration_microsecond() {
        // 1 second = 1_000_000 microseconds → f64 1_000_000_000.0 nanoseconds
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Duration(TimeUnit::Microsecond), true),
        ]));
        let arr = DurationMicrosecondArray::from(vec![Some(1_000_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let out = col_as_f64(&b, "d").unwrap();
        assert_eq!(out, vec![Some(1_000_000_000.0), None]);
    }

    #[test]
    fn col_as_f64_duration_millisecond() {
        // 1000 ms = 1 second → 1_000_000_000 ns
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Duration(TimeUnit::Millisecond), true),
        ]));
        let arr = DurationMillisecondArray::from(vec![Some(1000i64)]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let out = col_as_f64(&b, "d").unwrap();
        assert_eq!(out, vec![Some(1_000_000_000.0)]);
    }

    #[test]
    fn col_as_f64_duration_second() {
        // 1 s → 1_000_000_000 ns
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Duration(TimeUnit::Second), true),
        ]));
        let arr = DurationSecondArray::from(vec![Some(1i64)]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let out = col_as_f64(&b, "d").unwrap();
        assert_eq!(out, vec![Some(1_000_000_000.0)]);
    }

    #[test]
    fn col_as_f64_date32_days_to_epoch_millis() {
        // Day 1 (1970-01-02) → 86_400_000 ms
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Date32, true),
        ]));
        let arr = Date32Array::from(vec![Some(1i32), None, Some(0i32)]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let out = col_as_f64(&b, "d").unwrap();
        assert_eq!(out, vec![Some(86_400_000.0), None, Some(0.0)]);
    }

    #[test]
    fn col_as_f64_date64_ms_to_f64() {
        // Date64 stores milliseconds since epoch directly.
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Date64, true),
        ]));
        let arr = Date64Array::from(vec![Some(86_400_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let out = col_as_f64(&b, "d").unwrap();
        assert_eq!(out, vec![Some(86_400_000.0), None]);
    }

    // ── col_as_ordinal_category_str ──────────────────────────────────────────

    // ── distinct_values_in_order: Float32 / Float64 columns ─────────────────

    /// F1 regression: Float64 ordinal columns with integer-valued data must be
    /// supported by `distinct_values_in_order` (previously failed with
    /// `UnsupportedDtype`), and must produce the same strings as
    /// `col_as_ordinal_category_str` for the same data.
    #[test]
    fn distinct_values_in_order_float64_whole_values_format_as_integer() {
        // [2000.0, 2001.0, 2000.0] → ["2000", "2001"] (dedup, order-preserving)
        let schema = Arc::new(Schema::new(vec![Field::new("yr", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(2000.0), Some(2001.0), Some(2000.0),
            ]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "yr").unwrap();
        assert_eq!(out, vec!["2000", "2001"]);
    }

    #[test]
    fn distinct_values_in_order_float64_fractional_preserves_decimal() {
        // [1.5, 2.5] → ["1.5", "2.5"]
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.5), Some(2.5)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["1.5", "2.5"]);
    }

    #[test]
    fn distinct_values_in_order_float64_mixed_whole_and_fractional() {
        // [2.0, 1.5] → ["2", "1.5"]: whole values integer-formatted, fractional kept.
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(2.0), Some(1.5)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["2", "1.5"]);
    }

    /// Invariant: domain strings produced by `distinct_values_in_order` and
    /// per-row strings produced by `col_as_ordinal_category_str` must agree
    /// for Float64 columns. A value present in the domain must always hit a
    /// domain entry when looked up — this is the prerequisite for correct
    /// ordinal scale resolution and mark drawing.
    #[test]
    fn float64_distinct_values_and_category_str_agree() {
        let values = vec![Some(2000.0), Some(2001.0), Some(1.5), Some(2000.0), None];
        let schema = Arc::new(Schema::new(vec![Field::new("yr", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(values))],
        )
        .unwrap();

        let domain = distinct_values_in_order(&b, "yr").unwrap();
        let per_row = col_as_ordinal_category_str(&b, "yr").unwrap();

        // Every non-null per-row string must be in the domain.
        for s in per_row.iter().flatten() {
            assert!(
                domain.contains(s),
                "per-row string {s:?} not found in domain {domain:?}; domain/drawer mismatch"
            );
        }
        // Domain must be ["2000", "2001", "1.5"] (encounter order, deduped);
        // the color/shape domain helper drops nulls.
        assert_eq!(domain, vec!["2000", "2001", "1.5"]);

        // FA-9: the positional variants surface the null as its own category so a
        // null group renders a distinct x:N band.
        let pos_domain = distinct_positional_categories(&b, "yr").unwrap();
        assert_eq!(pos_domain, vec!["2000", "2001", "1.5", NULL_CATEGORY]);
        let pos_rows = col_as_positional_category_str(&b, "yr").unwrap();
        assert_eq!(pos_rows[4], Some(NULL_CATEGORY.to_string()));
    }

    #[test]
    fn distinct_values_in_order_float32_whole_values_format_as_integer() {
        use arrow::array::Float32Array;
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float32Array::from(vec![Some(10.0f32), Some(20.0f32), Some(10.0f32)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["10", "20"]);
    }

    #[test]
    fn distinct_values_in_order_float32_fractional() {
        use arrow::array::Float32Array;
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float32Array::from(vec![Some(1.5f32), Some(2.5f32)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["1.5", "2.5"]);
    }

    #[test]
    fn col_as_ordinal_category_str_utf8_passthrough() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "s").unwrap();
        assert_eq!(out, vec![Some("a".into()), None, Some("b".into())]);
    }

    #[test]
    fn col_as_ordinal_category_str_int64_stringifies() {
        // Int64 values must stringify the same way distinct_values_in_order does.
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("year", DataType::Int64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![Some(2000i64), Some(2001i64), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "year").unwrap();
        assert_eq!(out, vec![Some("2000".into()), Some("2001".into()), None]);
    }

    #[test]
    fn col_as_ordinal_category_str_int32_stringifies() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(10), Some(20), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "x").unwrap();
        assert_eq!(out, vec![Some("10".into()), Some("20".into()), None]);
    }

    #[test]
    fn col_as_ordinal_category_str_float64_whole_values_format_as_integer() {
        // Float64 ordinal columns with integer-valued data (e.g. 2000.0) must
        // produce "2000", not "2000.0", to match Int64 domain strings.
        let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(2000.0), Some(2001.0), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "y").unwrap();
        assert_eq!(out, vec![Some("2000".into()), Some("2001".into()), None]);
    }

    #[test]
    fn col_as_ordinal_category_str_float64_fractional_preserves_decimal() {
        // Float64 with a non-zero fractional part must keep the decimal.
        let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.5), Some(2.0), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "y").unwrap();
        assert_eq!(out[0], Some("1.5".into()));
        assert_eq!(out[1], Some("2".into()));
        assert_eq!(out[2], None);
    }

    #[test]
    fn col_as_ordinal_category_str_uint8_stringifies() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::UInt8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt8Array::from(vec![Some(5u8), Some(10u8), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "x").unwrap();
        assert_eq!(out, vec![Some("5".into()), Some("10".into()), None]);
    }

    // ── RSUP-02 regression: supported_numeric_dtype / is_numeric / col_as_f64 /
    //    min_max_f64 must agree on every dtype in the supported set ─────────────
    //
    // This test would FAIL on the old code because `is_numeric` and `min_max_f64`
    // omitted Date32/Date64/Duration — returning false/Err for dtypes that
    // `col_as_f64` could read successfully.  After D-DTYPE-1 all three branch
    // from `supported_numeric_dtype` and must agree for every dtype below.

    /// Helper: build a one-element RecordBatch with a single column of the given
    /// dtype containing a plausible non-null value, ready for agreement checks.
    fn one_elem_batch_for_dtype(dtype: &DataType) -> RecordBatch {
        use std::sync::Arc as A;
        let field = Field::new("v", dtype.clone(), true);
        let schema = Arc::new(Schema::new(vec![field]));
        let col: Arc<dyn Array> = match dtype {
            DataType::Float64 => A::new(Float64Array::from(vec![Some(1.0f64)])),
            DataType::Float32 => A::new(arrow::array::Float32Array::from(vec![Some(1.0f32)])),
            DataType::Int64 => A::new(arrow::array::Int64Array::from(vec![Some(1i64)])),
            DataType::Int32 => A::new(Int32Array::from(vec![Some(1i32)])),
            DataType::Int16 => A::new(arrow::array::Int16Array::from(vec![Some(1i16)])),
            DataType::Int8 => A::new(Int8Array::from(vec![Some(1i8)])),
            DataType::UInt64 => A::new(arrow::array::UInt64Array::from(vec![Some(1u64)])),
            DataType::UInt32 => A::new(UInt32Array::from(vec![Some(1u32)])),
            DataType::UInt16 => A::new(arrow::array::UInt16Array::from(vec![Some(1u16)])),
            DataType::UInt8 => A::new(UInt8Array::from(vec![Some(1u8)])),
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                A::new(TimestampMillisecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                A::new(arrow::array::TimestampMicrosecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                A::new(TimestampNanosecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Second, tz) => {
                A::new(TimestampSecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                A::new(DurationNanosecondArray::from(vec![Some(1i64)]))
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                A::new(DurationMicrosecondArray::from(vec![Some(1i64)]))
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                A::new(DurationMillisecondArray::from(vec![Some(1i64)]))
            }
            DataType::Duration(TimeUnit::Second) => {
                A::new(DurationSecondArray::from(vec![Some(1i64)]))
            }
            DataType::Date32 => A::new(Date32Array::from(vec![Some(1i32)])),
            DataType::Date64 => A::new(Date64Array::from(vec![Some(1i64)])),
            other => panic!("one_elem_batch_for_dtype: unhandled dtype {other:?}"),
        };
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// RSUP-02 regression: `supported_numeric_dtype`, `is_numeric`, `col_as_f64`,
    /// and `min_max_f64` must all agree — each returning true/Ok — for every
    /// dtype in the covered set, AND must agree NUMERICALLY (same f64 scale).
    ///
    /// On the old code this test would fail for `Date32`, `Date64`, and all four
    /// `Duration` variants because `is_numeric` returned `false` and `min_max_f64`
    /// returned `Err` for those dtypes, even though `col_as_f64` could read them.
    ///
    /// After T0-closeout the test additionally fails on the old code for
    /// `Duration(Microsecond)`, `Duration(Millisecond)`, and `Duration(Second)`
    /// because `min_max_f64` used raw i64 ticks while `col_as_f64` normalized to
    /// nanoseconds — e.g. Duration(ms) value=1 gave min_max → 1.0 but col_as_f64
    /// → 1_000_000.0 (off by 1e6). The numeric equality assertions below catch
    /// this: min_max_f64 min/max must equal the min/max of col_as_f64's output.
    #[test]
    fn supported_numeric_dtype_is_numeric_col_as_f64_min_max_agree() {
        let dtypes = [
            DataType::Float64,
            DataType::Float32,
            DataType::Int64,
            DataType::Int32,
            DataType::Int16,
            DataType::Int8,
            DataType::UInt64,
            DataType::UInt32,
            DataType::UInt16,
            DataType::UInt8,
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            DataType::Timestamp(TimeUnit::Second, None),
            // The next four are the latent-bug cases (RSUP-02): old code returned
            // false / Err for these while col_as_f64 could read them.
            // T0-closeout additionally requires numeric agreement (nanosecond scale).
            DataType::Duration(TimeUnit::Nanosecond),
            DataType::Duration(TimeUnit::Microsecond),
            DataType::Duration(TimeUnit::Millisecond),
            DataType::Duration(TimeUnit::Second),
            DataType::Date32,
            DataType::Date64,
        ];

        for dtype in &dtypes {
            let batch = one_elem_batch_for_dtype(dtype);

            // 1. supported_numeric_dtype and is_numeric must agree and both be true.
            assert!(
                supported_numeric_dtype(dtype),
                "supported_numeric_dtype returned false for {dtype:?}"
            );
            assert!(
                is_numeric(dtype),
                "is_numeric returned false for {dtype:?} (disagreement with supported_numeric_dtype)"
            );

            // 2. col_as_f64 must succeed.
            let f64_vals = col_as_f64(&batch, "v").unwrap_or_else(|e| {
                panic!("col_as_f64 returned Err for {dtype:?}: {e:?}")
            });

            // 3. min_max_f64 must succeed.
            let (mm_min, mm_max) = min_max_f64(batch.column(0).as_ref()).unwrap_or_else(|e| {
                panic!("min_max_f64 returned Err for {dtype:?}: {e}")
            });

            // 4. NUMERIC AGREEMENT: min_max_f64 values must match the min/max of
            //    col_as_f64's output. This is the key invariant — both must produce
            //    the same f64 scale (e.g. nanoseconds for Duration). On the old code
            //    this assertion fails for Duration(µs/ms/s) because min_max_f64 used
            //    raw ticks while col_as_f64 normalized to nanoseconds.
            let non_null_vals: Vec<f64> = f64_vals.into_iter().flatten().collect();
            let expected_min = non_null_vals.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let expected_max = non_null_vals.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            assert_eq!(
                mm_min, expected_min,
                "min_max_f64 min disagrees with col_as_f64 min for {dtype:?}: \
                 min_max_f64={mm_min} vs col_as_f64_min={expected_min}"
            );
            assert_eq!(
                mm_max, expected_max,
                "min_max_f64 max disagrees with col_as_f64 max for {dtype:?}: \
                 min_max_f64={mm_max} vs col_as_f64_max={expected_max}"
            );
        }

        // Sanity: non-numeric dtypes return false from both predicates.
        assert!(!supported_numeric_dtype(&DataType::Utf8));
        assert!(!supported_numeric_dtype(&DataType::Boolean));
        assert!(!is_numeric(&DataType::Utf8));
        assert!(!is_numeric(&DataType::Boolean));
    }
}
