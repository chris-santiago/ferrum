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
    Array, BooleanArray, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array,
    LargeStringArray, StringArray,
    TimestampMillisecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::record_batch::RecordBatch;

use super::RenderError;

/// True for Arrow dtypes that should route as continuous/quantitative when
/// inferring an encoding type from data alone.
///
/// Used by `scale_resolve::build_color_scale` to validate that a column
/// marked quantitative or temporal is actually a numeric dtype, replacing
/// the pre-F16 narrow `Float64|UInt64` check.
pub(crate) fn is_numeric(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float64 | DataType::Float32
            | DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8
            | DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8
            | DataType::Timestamp(_, _)
    )
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

/// Enumerate distinct values in encounter order, stringified, from a
/// column whose dtype is one of `Utf8`, `LargeUtf8`, `Int64`, or
/// `Boolean`. Returns `Err(ScaleResolutionFailed)` for any other dtype.
///
/// Matches the prior `scale_resolve::distinct_values_in_order` semantics.
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
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        // Polars produces LargeUtf8 (LargeStringArray) for string columns.
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else {
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
    use arrow::array::{Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{Field, Schema};
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
}
