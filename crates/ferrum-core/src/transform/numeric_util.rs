//! Shared numeric helpers for the transform layer.
//!
//! Kept `pub(crate)` — not part of the public API surface.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Coerce a numeric Arrow column to a `Float64Array`.
///
/// - If the column is already Float64, it is downcast and cloned.
/// - If the column is any other numeric type (Int8/16/32/64, UInt8/16/32/64,
///   Float16/Float32), it is cast to Float64 via the arrow-cast kernel.
/// - Any non-numeric type (e.g. Utf8) is rejected with a clear `PyValueError`.
///
/// `ctx` is the transform-name prefix (e.g. `"stat_kde"`) and `field` is the
/// column name, both used only in error messages. This lets KDE and other
/// numeric transforms accept integer-typed columns instead of hard-erroring.
pub(crate) fn coerce_to_float64(
    col: &ArrayRef,
    ctx: &str,
    field: &str,
) -> PyResult<Float64Array> {
    match col.data_type() {
        DataType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{ctx}: expected Float64Array for column '{field}'"
                ))
            }),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32 => {
            let casted = arrow::compute::cast(col, &DataType::Float64).map_err(|e| {
                PyValueError::new_err(format!(
                    "{ctx}: could not cast column '{field}' to Float64: {e}"
                ))
            })?;
            casted
                .as_any()
                .downcast_ref::<Float64Array>()
                .cloned()
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "{ctx}: cast to Float64 failed for column '{field}'"
                    ))
                })
        }
        other => Err(PyValueError::new_err(format!(
            "{ctx}: column '{field}' must be numeric, got {other:?}"
        ))),
    }
}

/// Extract clean `f64` values from a Float64Array, dropping null and NaN entries.
///
/// When `only_indices` is `Some(ixs)`, only the rows at those positions are
/// considered (used by groupby partitioning in bin/kde).  When it is `None`,
/// all rows are scanned.
pub(crate) fn clean_float64_values(
    arr: &Float64Array,
    only_indices: Option<&[usize]>,
) -> Vec<f64> {
    let push = |i: usize, out: &mut Vec<f64>| {
        if arr.is_null(i) {
            return;
        }
        let v = arr.value(i);
        if v.is_nan() {
            return;
        }
        out.push(v);
    };

    match only_indices {
        Some(ixs) => {
            let mut out = Vec::with_capacity(ixs.len());
            for &i in ixs {
                push(i, &mut out);
            }
            out
        }
        None => {
            let mut out = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                push(i, &mut out);
            }
            out
        }
    }
}

/// Raw `(lo, hi)` extent of a Float64-coercible column, dropping null/NaN.
///
/// Coerces `field` to Float64 (so integer columns are handled the same as float
/// ones), drops null and NaN entries via [`clean_float64_values`], then folds to
/// the min/max. Returns `None` when the field is missing or non-numeric, when the
/// cleaned column is empty, or when the extent is degenerate (`lo >= hi` or
/// non-finite).
///
/// This is the single source for the cast→clean→fold extent computation that the
/// grouped stat transforms (`bin`, `kde`) need before per-group partitioning.
pub(crate) fn raw_float64_extent(batch: &RecordBatch, field: &str) -> Option<(f64, f64)> {
    let schema = batch.schema();
    let idx = schema.index_of(field).ok()?;
    let arr = coerce_to_float64(batch.column(idx), "raw_float64_extent", field).ok()?;
    let clean = clean_float64_values(&arr, None);
    if clean.is_empty() {
        return None;
    }
    let (lo, hi) = clean
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    if lo.is_finite() && hi.is_finite() && lo < hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// Resolve the extent a grouped stat transform should apply across all groups.
///
/// Returns:
///   - `spec_extent` unchanged whenever the caller already pinned one (an
///     explicit extent always wins, and is shared across groups), else
///   - `None` when `shared` is `false` (default / layer mode → each group uses
///     its own data-derived extent), else
///   - the global [`raw_float64_extent`] over `field` when `shared` is `true`
///     and no explicit extent was set (so every group gets the same edges —
///     required for `multiple="stack"`/`"fill"` alignment).
///
/// This unifies the previously divergent shared-extent blocks in `bin` (a
/// `Cow`-patched spec) and `kde` (a cloned-spec rebuild with an inline fold).
/// The numeric result is identical to both prior copies for the existing path.
pub(crate) fn resolve_shared_extent(
    spec_extent: Option<(f64, f64)>,
    shared: bool,
    batch: &RecordBatch,
    field: &str,
) -> Option<(f64, f64)> {
    if spec_extent.is_some() {
        return spec_extent;
    }
    if !shared {
        return None;
    }
    raw_float64_extent(batch, field)
}

/// Linearly-interpolated quantile over an already-sorted (ascending) sequence.
///
/// Matches the numpy/pandas `linear` interpolation convention.
/// Returns `NaN` for empty slices.
pub(crate) fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray, UInt64Array};
    use std::sync::Arc;

    fn arr_from_options(vals: Vec<Option<f64>>) -> Float64Array {
        Float64Array::from(vals)
    }

    fn arr_from_vals(vals: Vec<f64>) -> Float64Array {
        Float64Array::from(vals)
    }

    // ---------- clean_float64_values ----------

    #[test]
    fn clean_drops_nulls() {
        let arr = arr_from_options(vec![Some(1.0), None, Some(3.0)]);
        let result = clean_float64_values(&arr, None);
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn clean_drops_nan() {
        let arr = arr_from_vals(vec![1.0, f64::NAN, 3.0]);
        let result = clean_float64_values(&arr, None);
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn clean_drops_null_and_nan_together() {
        let arr = arr_from_options(vec![Some(1.0), None, Some(f64::NAN), Some(3.0)]);
        let result = clean_float64_values(&arr, None);
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn clean_empty_input() {
        let arr = arr_from_vals(vec![]);
        let result = clean_float64_values(&arr, None);
        assert!(result.is_empty());
    }

    #[test]
    fn clean_all_null() {
        let arr = arr_from_options(vec![None, None]);
        let result = clean_float64_values(&arr, None);
        assert!(result.is_empty());
    }

    #[test]
    fn clean_only_indices_subset() {
        // Rows: [10.0, NaN, 30.0, null, 50.0]
        // only_indices = [0, 2, 4] → should return [10.0, 30.0, 50.0]
        let arr = arr_from_options(vec![
            Some(10.0),
            Some(f64::NAN),
            Some(30.0),
            None,
            Some(50.0),
        ]);
        let result = clean_float64_values(&arr, Some(&[0, 2, 4]));
        assert_eq!(result, vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn clean_only_indices_skips_nan_in_subset() {
        // All indices provided, but index 1 is NaN and index 3 is null.
        let arr = arr_from_options(vec![
            Some(1.0),
            Some(f64::NAN),
            Some(3.0),
            None,
            Some(5.0),
        ]);
        let result = clean_float64_values(&arr, Some(&[0, 1, 2, 3, 4]));
        assert_eq!(result, vec![1.0, 3.0, 5.0]);
    }

    // ---------- quantile_sorted ----------

    #[test]
    fn quantile_empty_returns_nan() {
        let q = quantile_sorted(&[], 0.5);
        assert!(q.is_nan());
    }

    #[test]
    fn quantile_single_element() {
        let q = quantile_sorted(&[42.0], 0.5);
        assert_eq!(q, 42.0);
    }

    #[test]
    fn quantile_median_even() {
        // [1,2,3,4] median → 2.5
        let q = quantile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.5);
        assert!((q - 2.5).abs() < 1e-12);
    }

    #[test]
    fn quantile_quartiles_five_elements() {
        // numpy.quantile([1,2,3,4,5], [0.25,0.5,0.75]) == [2.0, 3.0, 4.0]
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile_sorted(&data, 0.25) - 2.0).abs() < 1e-12);
        assert!((quantile_sorted(&data, 0.50) - 3.0).abs() < 1e-12);
        assert!((quantile_sorted(&data, 0.75) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_p0_and_p1_are_endpoints() {
        let data = [3.0f64, 7.0, 11.0];
        assert_eq!(quantile_sorted(&data, 0.0), 3.0);
        assert_eq!(quantile_sorted(&data, 1.0), 11.0);
    }

    // ---------- coerce_to_float64 ----------

    #[test]
    fn coerce_float64_passthrough() {
        let col: ArrayRef = Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5]));
        let out = coerce_to_float64(&col, "ctx", "f").unwrap();
        assert_eq!(out.values(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn coerce_int64_casts() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2, 3]));
        let out = coerce_to_float64(&col, "ctx", "f").unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn coerce_int32_casts() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![10_i32, 20]));
        let out = coerce_to_float64(&col, "ctx", "f").unwrap();
        assert_eq!(out.values(), &[10.0, 20.0]);
    }

    #[test]
    fn coerce_uint64_casts() {
        let col: ArrayRef = Arc::new(UInt64Array::from(vec![5_u64, 6]));
        let out = coerce_to_float64(&col, "ctx", "f").unwrap();
        assert_eq!(out.values(), &[5.0, 6.0]);
    }

    #[test]
    fn coerce_preserves_nulls() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(1_i64), None, Some(3)]));
        let out = coerce_to_float64(&col, "ctx", "f").unwrap();
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
        assert_eq!(out.value(0), 1.0);
        assert_eq!(out.value(2), 3.0);
    }

    #[test]
    fn coerce_string_rejected() {
        pyo3::Python::initialize();
        let col: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let err = coerce_to_float64(&col, "stat_kde", "x").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must be numeric"), "err: {msg}");
        assert!(msg.contains("stat_kde"), "err should carry ctx: {msg}");
        assert!(msg.contains("'x'"), "err should carry field: {msg}");
    }

    #[test]
    fn quantile_linear_interpolation_fractional() {
        // [0.0, 10.0], p=0.3 → h = 0.3*1 = 0.3, lo=0, hi=1, frac=0.3 → 0*(0.7)+10*(0.3) = 3.0
        let data = [0.0f64, 10.0];
        let q = quantile_sorted(&data, 0.3);
        assert!((q - 3.0).abs() < 1e-12);
    }

    // ---------- raw_float64_extent / resolve_shared_extent ----------

    use arrow::datatypes::{Field, Schema};

    fn one_col_batch(name: &str, col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            name,
            col.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    #[test]
    fn raw_extent_basic_min_max() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![3.0, 1.0, 5.0, 2.0])));
        assert_eq!(raw_float64_extent(&b, "x"), Some((1.0, 5.0)));
    }

    #[test]
    fn raw_extent_coerces_int_column() {
        use arrow::array::Int64Array;
        let b = one_col_batch("x", Arc::new(Int64Array::from(vec![1_i64, 5, 3])));
        assert_eq!(raw_float64_extent(&b, "x"), Some((1.0, 5.0)));
    }

    #[test]
    fn raw_extent_degenerate_returns_none() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0])));
        assert_eq!(raw_float64_extent(&b, "x"), None);
    }

    #[test]
    fn raw_extent_missing_field_returns_none() {
        pyo3::Python::initialize();
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 2.0])));
        assert_eq!(raw_float64_extent(&b, "ghost"), None);
    }

    #[test]
    fn resolve_shared_extent_explicit_wins() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 9.0])));
        assert_eq!(
            resolve_shared_extent(Some((0.0, 100.0)), true, &b, "x"),
            Some((0.0, 100.0)),
            "an explicit extent must be returned unchanged"
        );
    }

    #[test]
    fn resolve_shared_extent_not_shared_is_none() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 9.0])));
        assert_eq!(resolve_shared_extent(None, false, &b, "x"), None);
    }

    #[test]
    fn resolve_shared_extent_shared_computes_global() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 9.0, 4.0])));
        assert_eq!(resolve_shared_extent(None, true, &b, "x"), Some((1.0, 9.0)));
    }
}
