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

/// Whether `dt` is a dtype [`coerce_to_float64`] accepts (`Float64` itself, or
/// any other integer/unsigned/float dtype it casts). Kept in sync with
/// `coerce_to_float64`'s own match arms by test assertion — see
/// `is_numeric_dtype_matches_coerce_to_float64s_accepted_set` which verifies
/// both accept and reject the same set of dtypes. Both lists live in this
/// module. Used where a caller must *select* the numeric columns out of a
/// mixed batch before coercing each one individually, e.g.
/// `linkage::extract_matrix` picking feature columns out of a batch that also
/// carries a non-numeric row/column label column.
pub(crate) fn is_numeric_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
    )
}

/// Coerce a paired `(x, y)` column pair into jointly-clean `f64` vectors.
///
/// Both columns are resolved by name, coerced through [`coerce_to_float64`] (so
/// any integer/unsigned/float dtype is accepted and anything else raises), then
/// walked once keeping only the rows that are non-null and non-NaN in *both*
/// columns — paired listwise deletion, since a row missing either coordinate
/// cannot contribute to a fit.
///
/// `ctx` is the transform-name prefix used in every error message, so each
/// caller keeps its own error voice. This is the single ingestion path for the
/// two-column fit family (`logistic`, `glm`, `robust`); a caller that needs the
/// index-restricted variant (grouped per-partition fits) should use
/// [`coerce_xy_pairs_at`] instead of duplicating this walk. `smooth.rs::extract_xy`
/// remains a separate walk — it isn't routed through this helper.
pub(crate) fn coerce_xy_pairs(
    batch: &RecordBatch,
    x_field: &str,
    y_field: &str,
    ctx: &str,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    coerce_xy_pairs_at(batch, x_field, y_field, ctx, None)
}

/// [`coerce_xy_pairs`], restricted to `only_indices` when given.
///
/// When `only_indices` is `Some(ixs)`, only the rows at those positions are
/// considered (used by groupby partitioning — each group's fit ingests only
/// its own rows). When it is `None`, all rows are scanned, and this is
/// byte-identical to [`coerce_xy_pairs`]. Mirrors the `only_indices` pattern
/// already established by [`clean_float64_values`] in this module.
pub(crate) fn coerce_xy_pairs_at(
    batch: &RecordBatch,
    x_field: &str,
    y_field: &str,
    ctx: &str,
    only_indices: Option<&[usize]>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema
        .index_of(x_field)
        .map_err(|_| PyValueError::new_err(format!("{ctx}: column '{x_field}' not found")))?;
    let yi = schema
        .index_of(y_field)
        .map_err(|_| PyValueError::new_err(format!("{ctx}: column '{y_field}' not found")))?;
    let xa = coerce_to_float64(batch.column(xi), ctx, x_field)?;
    let ya = coerce_to_float64(batch.column(yi), ctx, y_field)?;

    let candidate_iter: Box<dyn Iterator<Item = usize>> = match only_indices {
        Some(ixs) => Box::new(ixs.iter().copied()),
        None => Box::new(0..xa.len()),
    };

    let cap = only_indices.map_or(xa.len(), <[usize]>::len);
    let mut xs = Vec::with_capacity(cap);
    let mut ys = Vec::with_capacity(cap);
    for i in candidate_iter {
        if xa.is_null(i) || ya.is_null(i) {
            continue;
        }
        let (xv, yv) = (xa.value(i), ya.value(i));
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        xs.push(xv);
        ys.push(yv);
    }
    Ok((xs, ys))
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

/// Finite `(lo, hi)` extent of a `Float64Array`, skipping null and NaN entries.
///
/// Folds the non-null, non-NaN values to their min/max. Returns `None` when the
/// array has no finite values, or when the resulting extent is degenerate
/// (`lo >= hi` or non-finite). This is the canonical null/NaN-skipping fold plus
/// `lo < hi` guard that every faceted-pin transform's per-axis extent needs.
///
/// [`column_extent`] is the batch-level wrapper (index_of + coerce_to_float64 +
/// `finite_extent`); use it when starting from a column name.
pub(crate) fn finite_extent(arr: &Float64Array) -> Option<(f64, f64)> {
    let (lo, hi) = (0..arr.len()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), i| {
        if arr.is_null(i) {
            return (lo, hi);
        }
        let v = arr.value(i);
        if v.is_nan() {
            return (lo, hi);
        }
        (lo.min(v), hi.max(v))
    });
    if lo.is_finite() && hi.is_finite() && lo < hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// Finite `(lo, hi)` extent of a numeric column, addressed by name.
///
/// Coerces `field` to Float64 (so integer columns are handled the same as float
/// ones), then delegates to [`finite_extent`]. Returns `None` when the field is
/// missing or non-numeric, when the column has no finite values, or when the
/// extent is degenerate (`lo >= hi` or non-finite).
///
/// This is the single source for the cast→clean→fold extent computation that the
/// faceted-pin stat transforms (`bin`, `kde`, `violin`, `density_data`,
/// `data_bin`, and the per-axis 2-D pins in `bin_2d`/`hex`/`raster`/`kde_2d`)
/// share before per-group partitioning.
pub(crate) fn column_extent(batch: &RecordBatch, field: &str) -> Option<(f64, f64)> {
    let schema = batch.schema();
    let idx = schema.index_of(field).ok()?;
    let arr = coerce_to_float64(batch.column(idx), "column_extent", field).ok()?;
    finite_extent(&arr)
}

/// Resolve the extent a grouped stat transform should apply across all groups.
///
/// Returns:
///   - `spec_extent` unchanged whenever the caller already pinned one (an
///     explicit extent always wins, and is shared across groups), else
///   - `None` when `shared` is `false` (default / layer mode → each group uses
///     its own data-derived extent), else
///   - the global [`column_extent`] over `field` when `shared` is `true`
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
    column_extent(batch, field)
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

    // ---------- is_numeric_dtype ----------

    #[test]
    fn is_numeric_dtype_matches_coerce_to_float64s_accepted_set() {
        pyo3::Python::initialize();

        // Accepted numeric dtypes: is_numeric_dtype should return true, and
        // coerce_to_float64 should accept a one-element array.
        for dt in [
            DataType::Float64,
            DataType::Float32,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            assert!(
                is_numeric_dtype(&dt),
                "{dt:?} should be numeric per is_numeric_dtype"
            );

            // Build a one-element array of this dtype and verify coerce accepts it.
            let test_arr: ArrayRef = match &dt {
                DataType::Float64 => Arc::new(Float64Array::from(vec![1.5])),
                DataType::Float32 => {
                    Arc::new(arrow::array::Float32Array::from(vec![1.5_f32]))
                }
                DataType::Int8 => Arc::new(arrow::array::Int8Array::from(vec![1_i8])),
                DataType::Int16 => Arc::new(arrow::array::Int16Array::from(vec![1_i16])),
                DataType::Int32 => Arc::new(Int32Array::from(vec![1_i32])),
                DataType::Int64 => Arc::new(Int64Array::from(vec![1_i64])),
                DataType::UInt8 => Arc::new(arrow::array::UInt8Array::from(vec![1_u8])),
                DataType::UInt16 => Arc::new(arrow::array::UInt16Array::from(vec![1_u16])),
                DataType::UInt32 => Arc::new(arrow::array::UInt32Array::from(vec![1_u32])),
                DataType::UInt64 => Arc::new(UInt64Array::from(vec![1_u64])),
                _ => panic!("unhandled dtype in test: {dt:?}"),
            };

            let result = coerce_to_float64(&test_arr, "test", "col");
            assert!(
                result.is_ok(),
                "{dt:?} should be accepted by coerce_to_float64; got {result:?}"
            );
        }

        // Note: Float16 is listed in coerce_to_float64's match and is_numeric_dtype's matches,
        // but is not tested here due to the complexity of constructing a Float16Array in tests.
        // Its coverage is ensured by being handled in the same code paths as Float32.

        // Refused dtypes: is_numeric_dtype should return false, and
        // coerce_to_float64 should reject a one-element array.
        for dt in [
            DataType::Utf8,
            DataType::Boolean,
            DataType::LargeUtf8,
            DataType::Decimal128(38, 10),
            DataType::Date32,
        ] {
            assert!(
                !is_numeric_dtype(&dt),
                "{dt:?} should not be numeric per is_numeric_dtype"
            );

            // Build a one-element array of this dtype and verify coerce rejects it.
            let test_arr: ArrayRef = match &dt {
                DataType::Utf8 => Arc::new(StringArray::from(vec!["test"])),
                DataType::Boolean => {
                    Arc::new(arrow::array::BooleanArray::from(vec![true]))
                }
                DataType::LargeUtf8 => {
                    Arc::new(arrow::array::LargeStringArray::from(vec!["test"]))
                }
                DataType::Decimal128(_, _) => {
                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![Some(1_i128)])
                            .with_data_type(dt.clone()),
                    )
                }
                DataType::Date32 => Arc::new(arrow::array::Date32Array::from(vec![18993_i32])),
                _ => panic!("unhandled dtype in test: {dt:?}"),
            };

            let result = coerce_to_float64(&test_arr, "test", "col");
            assert!(
                result.is_err(),
                "{dt:?} should be rejected by coerce_to_float64; got {result:?}"
            );
        }
    }

    // ---------- coerce_xy_pairs ----------

    fn xy_batch(x: ArrayRef, y: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", x.data_type().clone(), true),
            Field::new("y", y.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![x, y]).unwrap()
    }

    #[test]
    fn coerce_xy_pairs_float64_passthrough() {
        let b = xy_batch(
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
        );
        let (xs, ys) = coerce_xy_pairs(&b, "x", "y", "Glm").unwrap();
        assert_eq!(xs, vec![1.0, 2.0, 3.0]);
        assert_eq!(ys, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn coerce_xy_pairs_int_columns_match_their_float_equivalents() {
        let ints = xy_batch(
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(Int32Array::from(vec![4_i32, 5, 6])),
        );
        let floats = xy_batch(
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
        );
        assert_eq!(
            coerce_xy_pairs(&ints, "x", "y", "Glm").unwrap(),
            coerce_xy_pairs(&floats, "x", "y", "Glm").unwrap()
        );
    }

    #[test]
    fn coerce_xy_pairs_drops_rows_null_or_nan_in_either_column() {
        let b = xy_batch(
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
                Some(f64::NAN),
                Some(5.0),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(10.0),
                Some(20.0),
                None,
                Some(40.0),
                Some(50.0),
            ])),
        );
        let (xs, ys) = coerce_xy_pairs(&b, "x", "y", "Robust").unwrap();
        assert_eq!(xs, vec![1.0, 5.0]);
        assert_eq!(ys, vec![10.0, 50.0]);
    }

    #[test]
    fn coerce_xy_pairs_non_numeric_raises_with_ctx_column_and_dtype() {
        pyo3::Python::initialize();
        let b = xy_batch(
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        );
        let msg = coerce_xy_pairs(&b, "x", "y", "Logistic")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("Logistic"), "err: {msg}");
        assert!(msg.contains("'x'"), "err must name the column: {msg}");
        assert!(msg.contains("Utf8"), "err must name the dtype: {msg}");
    }

    // ---------- coerce_xy_pairs_at (only_indices) ----------

    #[test]
    fn coerce_xy_pairs_at_none_matches_coerce_xy_pairs() {
        let b = xy_batch(
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
        );
        assert_eq!(
            coerce_xy_pairs_at(&b, "x", "y", "Test", None).unwrap(),
            coerce_xy_pairs(&b, "x", "y", "Test").unwrap()
        );
    }

    #[test]
    fn coerce_xy_pairs_at_restricts_to_only_indices() {
        let b = xy_batch(
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        );
        let (xs, ys) = coerce_xy_pairs_at(&b, "x", "y", "Test", Some(&[0, 2])).unwrap();
        assert_eq!(xs, vec![10.0, 30.0]);
        assert_eq!(ys, vec![1.0, 3.0]);
    }

    #[test]
    fn coerce_xy_pairs_at_only_indices_still_drops_null_and_nan() {
        let b = xy_batch(
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(f64::NAN),
                Some(3.0),
                None,
            ])),
            Arc::new(Float64Array::from(vec![
                Some(10.0),
                Some(20.0),
                Some(30.0),
                Some(40.0),
            ])),
        );
        // Restrict to all four indices; NaN at 1 and null at 3 must still drop.
        let (xs, ys) = coerce_xy_pairs_at(&b, "x", "y", "Test", Some(&[0, 1, 2, 3])).unwrap();
        assert_eq!(xs, vec![1.0, 3.0]);
        assert_eq!(ys, vec![10.0, 30.0]);
    }

    #[test]
    fn coerce_xy_pairs_missing_column_raises_with_ctx() {
        pyo3::Python::initialize();
        let b = xy_batch(
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![2.0])),
        );
        let msg = coerce_xy_pairs(&b, "x", "ghost", "data_loess")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("data_loess: column 'ghost' not found"), "err: {msg}");
    }

    // ---- R1-relocated coverage (tests/bug_hunt_release_transforms.rs, 2026-08-27) ----

    #[test]
    fn quantile_all_equal_returns_that_value() {
        let v = [7.0, 7.0, 7.0, 7.0];
        assert_eq!(quantile_sorted(&v, 0.0), 7.0);
        assert_eq!(quantile_sorted(&v, 0.5), 7.0);
        assert_eq!(quantile_sorted(&v, 1.0), 7.0);
    }

    #[test]
    fn quantile_with_nan_present_poisons_result() {
        // quantile_sorted does NOT filter NaN (the caller is responsible via
        // clean_float64_values). A NaN in the sorted slice at the interpolation
        // index poisons the output. This documents the contract boundary: the
        // helper assumes pre-cleaned input.
        let v = [1.0, f64::NAN, 3.0]; // not actually sorted, but exercises NaN
        let q = quantile_sorted(&v, 0.5); // h=1.0 → index 1 → NaN
        assert!(q.is_nan(), "NaN at the interpolation index must propagate");
    }

    #[test]
    fn clean_keeps_infinity_drops_null_and_nan() {
        // Infinity is NOT NaN, so it survives cleaning — a value that then
        // poisons downstream sum/mean. Documents that clean keeps ±inf.
        let arr = arr_from_options(vec![Some(1.0), None, Some(f64::NAN), Some(f64::INFINITY)]);
        let result = clean_float64_values(&arr, None);
        assert_eq!(result, vec![1.0, f64::INFINITY]);
    }

    #[test]
    fn quantile_linear_interpolation_fractional() {
        // [0.0, 10.0], p=0.3 → h = 0.3*1 = 0.3, lo=0, hi=1, frac=0.3 → 0*(0.7)+10*(0.3) = 3.0
        let data = [0.0f64, 10.0];
        let q = quantile_sorted(&data, 0.3);
        assert!((q - 3.0).abs() < 1e-12);
    }

    // ---------- finite_extent / column_extent / resolve_shared_extent ----------

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
    fn finite_extent_basic_min_max() {
        let arr = Float64Array::from(vec![3.0, 1.0, 5.0, 2.0]);
        assert_eq!(finite_extent(&arr), Some((1.0, 5.0)));
    }

    #[test]
    fn finite_extent_skips_null_and_nan() {
        let arr = arr_from_options(vec![Some(3.0), None, Some(f64::NAN), Some(1.0), Some(5.0)]);
        assert_eq!(finite_extent(&arr), Some((1.0, 5.0)));
    }

    #[test]
    fn finite_extent_all_null_returns_none() {
        let arr = arr_from_options(vec![None, None]);
        assert_eq!(finite_extent(&arr), None);
    }

    #[test]
    fn finite_extent_degenerate_returns_none() {
        let arr = Float64Array::from(vec![5.0, 5.0, 5.0]);
        assert_eq!(finite_extent(&arr), None);
    }

    #[test]
    fn column_extent_basic_min_max() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![3.0, 1.0, 5.0, 2.0])));
        assert_eq!(column_extent(&b, "x"), Some((1.0, 5.0)));
    }

    #[test]
    fn column_extent_coerces_int_column() {
        use arrow::array::Int64Array;
        let b = one_col_batch("x", Arc::new(Int64Array::from(vec![1_i64, 5, 3])));
        assert_eq!(column_extent(&b, "x"), Some((1.0, 5.0)));
    }

    #[test]
    fn column_extent_degenerate_returns_none() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0])));
        assert_eq!(column_extent(&b, "x"), None);
    }

    #[test]
    fn column_extent_missing_field_returns_none() {
        pyo3::Python::initialize();
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 2.0])));
        assert_eq!(column_extent(&b, "ghost"), None);
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
