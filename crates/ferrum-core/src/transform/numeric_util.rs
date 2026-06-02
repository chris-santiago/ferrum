//! Shared numeric helpers for the transform layer.
//!
//! Kept `pub(crate)` — not part of the public API surface.

use arrow::array::{Array, Float64Array};

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
    use arrow::array::Float64Array;
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

    #[test]
    fn quantile_linear_interpolation_fractional() {
        // [0.0, 10.0], p=0.3 → h = 0.3*1 = 0.3, lo=0, hi=1, frac=0.3 → 0*(0.7)+10*(0.3) = 3.0
        let data = [0.0f64, 10.0];
        let q = quantile_sorted(&data, 0.3);
        assert!((q - 3.0).abs() < 1e-12);
    }
}
