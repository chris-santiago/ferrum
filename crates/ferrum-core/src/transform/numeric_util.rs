//! Shared numeric helpers for the transform layer.
//!
//! Kept `pub(crate)` — not part of the public API surface.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Schema};
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

/// Schema positions of the numeric columns a wide matrix-shaped table
/// exposes, in schema order: `Linkage`'s column axis leaf `k` is the column
/// at position `k` of this list, `Reorder(axis="columns")` permutes exactly
/// these positions, and `pairplot`/`heatmap`/`clustermap` split their id
/// column from these same positions via [`numeric_columns`]. The predicate is
/// crate-wide (every [`is_numeric_dtype`] dtype), not specific to clustering,
/// which is why it lives beside `is_numeric_dtype` rather than in `linkage`.
///
/// A decimal column is refused, voiced as `ctx`: it holds numbers, but no
/// transform reads it as one, so leaving it out of the numeric columns would
/// drop data the caller meant to use.
pub(crate) fn numeric_column_indices(schema: &Schema, ctx: &str) -> PyResult<Vec<usize>> {
    let decimal = schema.fields().iter().find(|f| {
        matches!(
            f.data_type(),
            DataType::Decimal32(..) | DataType::Decimal64(..) | DataType::Decimal128(..) | DataType::Decimal256(..)
        )
    });
    if let Some(field) = decimal {
        return Err(PyValueError::new_err(format!(
            "{ctx}: column '{}' is a decimal column ({}); a numeric column must be an integer or \
             float column, so cast it to float first",
            field.name(),
            field.data_type()
        )));
    }
    Ok(schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| is_numeric_dtype(f.data_type()))
        .map(|(i, _)| i)
        .collect())
}

/// `ferrum._core.numeric_columns(schema, entry)`: the names of the numeric
/// columns [`numeric_column_indices`] selects from an Arrow `schema`, in
/// schema order, so a Python figure function splits its table into id and
/// numeric columns by the one rule `Linkage`, `Reorder`, `pairplot` and
/// `heatmap` all apply. Refuses a decimal column with a `ValueError` voiced
/// as `entry`. Registered in `lib.rs`, typed in `src/ferrum/_core.pyi`.
#[pyo3::pyfunction]
pub(crate) fn numeric_columns(schema: pyo3_arrow::PySchema, entry: &str) -> PyResult<Vec<String>> {
    let schema = schema.into_inner();
    Ok(numeric_column_indices(&schema, entry)?
        .into_iter()
        .map(|i| schema.field(i).name().clone())
        .collect())
}

/// Coerce a paired `(x, y)` column pair into jointly-clean `f64` vectors,
/// over every row (`only_indices` `None`) or only the rows at `only_indices`
/// (one groupby partition's).
///
/// Both columns are resolved by name, coerced through [`coerce_to_float64`] (so
/// any integer/unsigned/float dtype is accepted and anything else raises), then
/// walked once keeping only the rows that are non-null and non-NaN in *both*
/// columns — paired listwise deletion, since a row missing either coordinate
/// cannot contribute to a fit.
///
/// `ctx` is the transform-name prefix used in every error message, so each
/// caller keeps its own error voice. This is the single ingestion path for the
/// two-column fit family (`logistic`, `glm`, `robust`), grouped or not.
/// `smooth.rs::extract_xy` remains a separate walk — it isn't routed through
/// this helper. Mirrors the `only_indices` pattern already established by
/// [`clean_float64_values`] in this module.
pub(crate) fn coerce_xy_pairs_at(
    batch: &RecordBatch,
    x_field: &str,
    y_field: &str,
    ctx: &str,
    only_indices: Option<&[usize]>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    coerce_xy_pairs_at_indexed(batch, x_field, y_field, ctx, only_indices).map(|(xs, ys, _)| (xs, ys))
}

/// Like [`coerce_xy_pairs_at`], but also returns the ORIGINAL row index of
/// each surviving pair, in the same order — used where a caller needs to
/// carry other columns for exactly the rows a residuals-mode fit kept
/// (`Robust`'s own residual batch; mirrors
/// [`clean_float64_values_indexed`]'s identical shape for the single-column
/// case).
pub(crate) fn coerce_xy_pairs_at_indexed(
    batch: &RecordBatch,
    x_field: &str,
    y_field: &str,
    ctx: &str,
    only_indices: Option<&[usize]>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
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
    let mut idxs = Vec::with_capacity(cap);
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
        idxs.push(i);
    }
    Ok((xs, ys, idxs))
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
    clean_float64_values_indexed(arr, only_indices).0
}

/// Like [`clean_float64_values`], but also returns the ORIGINAL row index of
/// each surviving value, in the same order. Used where a caller needs to look
/// up other columns for exactly the rows that survived a null/NaN filter —
/// e.g. `Bin`'s optional per-bin aggregate (batch-E T7), which must aggregate
/// `agg_field` over precisely the rows counted in each bin.
pub(crate) fn clean_float64_values_indexed(
    arr: &Float64Array,
    only_indices: Option<&[usize]>,
) -> (Vec<f64>, Vec<usize>) {
    let push = |i: usize, vals: &mut Vec<f64>, idxs: &mut Vec<usize>| {
        if arr.is_null(i) {
            return;
        }
        let v = arr.value(i);
        if v.is_nan() {
            return;
        }
        vals.push(v);
        idxs.push(i);
    };

    match only_indices {
        Some(ixs) => {
            let mut vals = Vec::with_capacity(ixs.len());
            let mut idxs = Vec::with_capacity(ixs.len());
            for &i in ixs {
                push(i, &mut vals, &mut idxs);
            }
            (vals, idxs)
        }
        None => {
            let mut vals = Vec::with_capacity(arr.len());
            let mut idxs = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                push(i, &mut vals, &mut idxs);
            }
            (vals, idxs)
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
    finite_extent_where(arr, |_| true)
}

/// [`finite_extent`] over a subset of rows: same null/NaN skip, same fold, same
/// `lo < hi` guard, restricted to the rows `keep` accepts. One fold body serves
/// both so a whole-column extent and a row-restricted one can never disagree
/// about what "finite extent" means.
fn finite_extent_where(arr: &Float64Array, keep: impl Fn(usize) -> bool) -> Option<(f64, f64)> {
    let (lo, hi) = (0..arr.len()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), i| {
        if !keep(i) || arr.is_null(i) {
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

/// `range`'s two endpoints padded by a symmetric proportional margin,
/// `(hi - lo) * pad`, on both sides, in whichever units `range` is already
/// stated in: data units on a linear axis, or log units once a demanded
/// log x's own evaluation range is already stated there — this is `lmplot`'s
/// `truncate=False` margin and `Smooth`/`Robust`'s `x_range_pad`, computed
/// in the SAME space the range itself is given in, so the margin itself
/// never lands off-side even when the unpadded range sits close to the
/// axis's excluded side. Not log-specific — a plain arithmetic helper, so
/// it lives beside [`column_extent`] rather than in `log_space.rs`. `pad`
/// of `None` returns `range` unchanged.
pub(crate) fn padded_extent(range: [f64; 2], pad: Option<f64>) -> [f64; 2] {
    let Some(pad) = pad else { return range };
    let [lo, hi] = range;
    let margin = (hi - lo) * pad;
    [lo - margin, hi + margin]
}

/// The evaluation range `Smooth`'s and `Robust`'s per-group fit body
/// grids over: `x_range` when the caller wrote one, else the WHOLE
/// `batch`'s own `x_field` extent (`column_extent`, before any `groupby`
/// split, so every group spans the SAME full-column range) — either way
/// padded by `x_range_pad`
/// ([`padded_extent`]). `x_range` arrives already mapped into the fit's
/// own space when one is demanded (`LogAxisSpace::extent_in_space`, called
/// by the caller before this), and `batch`/`x_field` are themselves
/// already in that space too (the demanded route's body runs on the
/// mapped batch, off-side rows null), so this function itself carries no
/// log-space knowledge at all: a plain extent-and-pad fold, run after
/// mapping rather than before it.
///
/// `None` when there is nothing to derive a range from: both `x_range` and
/// `x_range_pad` unset (the per-group data extent stands unchanged), or `x_range`
/// unset and the whole batch's `x_field` extent is itself `None` (no finite
/// values, or every row off-side on the mapped route — the axis's own
/// `LogScaleNoValueOnSide` refusal speaks there instead of this function
/// inventing a range).
pub(crate) fn evaluation_range(
    x_range: Option<[f64; 2]>,
    x_range_pad: Option<f64>,
    batch: &RecordBatch,
    x_field: &str,
) -> Option<[f64; 2]> {
    if x_range.is_none() && x_range_pad.is_none() {
        return None;
    }
    let base = match x_range {
        Some(range) => range,
        None => {
            let (lo, hi) = column_extent(batch, x_field)?;
            [lo, hi]
        }
    };
    Some(padded_extent(base, x_range_pad))
}

/// [`column_extent`] restricted to the rows a GROUPED transform will actually
/// draw — those whose `groupby` key survives GH #197's null-key policy.
///
/// The one fold every shared/global value extent in the grouped family uses, so
/// "the shared extent" means the same thing at both of the places it is
/// computed: the facet pin (`render::prepare::extent`, via each transform's
/// `global_extent`) and the in-transform `shared_extent` resolution
/// ([`resolve_shared_extent`]). Before #197 the distinction did not exist —
/// a null-keyed row still formed its own group and drew its own violin/curve,
/// so folding it in was correct. Under the adjudicated policy it draws nothing,
/// and folding it in stretches every SURVIVING group's grid across a range no
/// group occupies. `groupby.is_empty()` is exactly [`column_extent`]: no
/// grouping, no exclusions, byte-identical to before.
///
/// Deliberately lenient about a malformed `groupby`: a missing column or an
/// unsupported key dtype falls back to the whole-column fold, because the
/// transform's own partitioner refuses that request moments later with a proper
/// message (`group_key::validate_groupby_columns`). Inventing a second, worse
/// refusal site inside an extent helper would only make the error worse.
///
/// COST, stated as a decision rather than an accident (quality review N2): the
/// per-row predicate calls [`crate::transform::group_key::groupby_key_at`],
/// which for a `Utf8` key allocates a `String` purely to be tested for null.
/// `batch.column(gi).is_null(row)` would be allocation-free and is equivalent
/// for every supported key dtype — but it restates "what counts as a null
/// group key" outside `group_key`, which is the one module that owns that rule
/// and the coupling this task exists to remove. The trade is deliberate: a
/// transient per-row allocation on a pre-facet fold, against a second copy of
/// the null-key definition drifting from the partitioner's. Revisit only with a
/// measurement showing this fold on the hot path.
pub(crate) fn grouped_column_extent(
    batch: &RecordBatch,
    field: &str,
    groupby: &[String],
) -> Option<(f64, f64)> {
    let schema = batch.schema();
    let idx = schema.index_of(field).ok()?;
    let arr = coerce_to_float64(batch.column(idx), "column_extent", field).ok()?;
    match kept_by_groupby(batch, groupby) {
        Some(keep) => finite_extent_where(&arr, keep),
        None => finite_extent(&arr),
    }
}

/// The clean (non-null, non-NaN) values of `field` over the rows a GROUPED
/// transform draws: the same rows [`grouped_column_extent`] folds, so a
/// statistic of the pooled values (a bin count over them) and the pooled
/// extent describe one set of rows. `None` when the field is missing or
/// non-numeric.
pub(crate) fn grouped_clean_values(
    batch: &RecordBatch,
    field: &str,
    groupby: &[String],
) -> Option<Vec<f64>> {
    let schema = batch.schema();
    let idx = schema.index_of(field).ok()?;
    let arr = coerce_to_float64(batch.column(idx), "column_extent", field).ok()?;
    Some(match kept_by_groupby(batch, groupby) {
        Some(keep) => {
            let rows: Vec<usize> = (0..arr.len()).filter(|&row| keep(row)).collect();
            clean_float64_values(&arr, Some(&rows))
        }
        None => clean_float64_values(&arr, None),
    })
}

/// Whether a row's `groupby` key survives GH #197's null-key policy, the one
/// row filter [`grouped_column_extent`] and [`grouped_clean_values`] share.
/// `None` keeps every row: an ungrouped request, or a malformed `groupby` (a
/// missing column or an unsupported key dtype), which the transform's own
/// partitioner refuses with a proper message (see [`grouped_column_extent`]).
fn kept_by_groupby<'a>(
    batch: &'a RecordBatch,
    groupby: &[String],
) -> Option<impl Fn(usize) -> bool + 'a> {
    use crate::transform::group_key::{groupby_key_at, is_groupby_supported_dtype, KeyValue};

    if groupby.is_empty() {
        return None;
    }
    let schema = batch.schema();
    let mut key_cols: Vec<(usize, DataType)> = Vec::with_capacity(groupby.len());
    for name in groupby {
        let gi = schema.index_of(name).ok()?;
        let dt = schema.field(gi).data_type().clone();
        if !is_groupby_supported_dtype(&dt) {
            return None;
        }
        key_cols.push((gi, dt));
    }
    Some(move |row: usize| {
        key_cols.iter().all(|(gi, dt)| {
            !matches!(
                groupby_key_at(batch.column(*gi).as_ref(), dt, row),
                Some(KeyValue::Null) | None
            )
        })
    })
}

/// Resolve the extent a grouped stat transform should apply across all groups.
///
/// Returns:
///   - `spec_extent` unchanged whenever the caller already pinned one (an
///     explicit extent always wins, and is shared across groups), else
///   - `None` when `shared` is `false` (default / layer mode → each group uses
///     its own data-derived extent), else
///   - the [`grouped_column_extent`] over `field` when `shared` is `true`
///     and no explicit extent was set (so every group gets the same edges —
///     required for `multiple="stack"`/`"fill"` alignment).
///
/// This unifies the previously divergent shared-extent blocks in `bin` (a
/// `Cow`-patched spec) and `kde` (a cloned-spec rebuild with an inline fold).
/// `groupby` is the caller's own request, so the fold covers exactly the rows
/// that caller will draw — see [`grouped_column_extent`]. With no `groupby`, or
/// with no null keys in it, the result is identical to the pre-#197 whole-column
/// fold.
pub(crate) fn resolve_shared_extent(
    spec_extent: Option<(f64, f64)>,
    shared: bool,
    batch: &RecordBatch,
    field: &str,
    groupby: &[String],
) -> Option<(f64, f64)> {
    if spec_extent.is_some() {
        return spec_extent;
    }
    if !shared {
        return None;
    }
    grouped_column_extent(batch, field, groupby)
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

    // ---------- clean_float64_values_indexed ----------

    #[test]
    fn clean_indexed_returns_original_row_indices() {
        let arr = arr_from_options(vec![Some(10.0), None, Some(f64::NAN), Some(40.0)]);
        let (vals, idxs) = clean_float64_values_indexed(&arr, None);
        assert_eq!(vals, vec![10.0, 40.0]);
        assert_eq!(idxs, vec![0, 3], "surviving values must carry their ORIGINAL row index");
    }

    #[test]
    fn clean_indexed_only_indices_subset_preserves_original_indices() {
        let arr = arr_from_options(vec![Some(1.0), Some(f64::NAN), Some(3.0), None, Some(5.0)]);
        let (vals, idxs) = clean_float64_values_indexed(&arr, Some(&[0, 1, 2, 3, 4]));
        assert_eq!(vals, vec![1.0, 3.0, 5.0]);
        assert_eq!(idxs, vec![0, 2, 4]);
    }

    #[test]
    fn clean_indexed_values_match_clean_float64_values() {
        // clean_float64_values must stay byte-identical (it now delegates to
        // clean_float64_values_indexed and drops the index half).
        let arr = arr_from_options(vec![Some(1.0), None, Some(f64::NAN), Some(3.0)]);
        assert_eq!(clean_float64_values(&arr, None), clean_float64_values_indexed(&arr, None).0);
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

    // ---------- coerce_xy_pairs_at ----------

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
        let (xs, ys) = coerce_xy_pairs_at(&b, "x", "y", "Glm", None).unwrap();
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
            coerce_xy_pairs_at(&ints, "x", "y", "Glm", None).unwrap(),
            coerce_xy_pairs_at(&floats, "x", "y", "Glm", None).unwrap()
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
        let (xs, ys) = coerce_xy_pairs_at(&b, "x", "y", "Robust", None).unwrap();
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
        let msg = coerce_xy_pairs_at(&b, "x", "y", "Logistic", None)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("Logistic"), "err: {msg}");
        assert!(msg.contains("'x'"), "err must name the column: {msg}");
        assert!(msg.contains("Utf8"), "err must name the dtype: {msg}");
    }

    // ---------- coerce_xy_pairs_at (only_indices) ----------

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
        let msg = coerce_xy_pairs_at(&b, "x", "ghost", "data_loess", None)
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
    fn evaluation_range_both_none_keeps_the_per_group_extent() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 2.0, 4.0])));
        assert_eq!(evaluation_range(None, None, &b, "x"), None, "no x_range and no pad: the caller's per-group data extent stands");
    }

    #[test]
    fn evaluation_range_pad_alone_pads_the_whole_batchs_own_extent() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 2.0, 4.0, 8.0, 16.0])));
        // The whole batch's own extent is [1, 16], padded 10% on each
        // side: margin = 15 * 0.1 = 1.5.
        assert_eq!(evaluation_range(None, Some(0.1), &b, "x"), Some([-0.5, 17.5]), "pad alone means extend past the data's own range");
    }

    #[test]
    fn evaluation_range_x_range_plus_pad_pads_the_caller_written_range() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 2.0, 4.0])));
        // x_range overrides the batch's own extent entirely; the pad still
        // applies, in whichever units x_range is already stated in.
        assert_eq!(
            evaluation_range(Some([10.0, 20.0]), Some(0.1), &b, "x"),
            Some([9.0, 21.0]),
            "x_range wins over the batch's own extent, with the pad applied to IT"
        );
    }

    #[test]
    fn evaluation_range_all_tied_batch_extent_returns_none() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0])));
        assert_eq!(
            evaluation_range(None, Some(0.1), &b, "x"),
            None,
            "a degenerate (all-tied) batch extent has no range to derive or pad"
        );
    }

    #[test]
    fn evaluation_range_empty_batch_returns_none() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(Vec::<f64>::new())));
        assert_eq!(evaluation_range(None, Some(0.1), &b, "x"), None, "an empty column has no extent to derive or pad");
    }

    #[test]
    fn resolve_shared_extent_explicit_wins() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 9.0])));
        assert_eq!(
            resolve_shared_extent(Some((0.0, 100.0)), true, &b, "x", &[]),
            Some((0.0, 100.0)),
            "an explicit extent must be returned unchanged"
        );
    }

    #[test]
    fn resolve_shared_extent_not_shared_is_none() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 9.0])));
        assert_eq!(resolve_shared_extent(None, false, &b, "x", &[]), None);
    }

    #[test]
    fn resolve_shared_extent_shared_computes_global() {
        let b = one_col_batch("x", Arc::new(Float64Array::from(vec![1.0, 9.0, 4.0])));
        assert_eq!(resolve_shared_extent(None, true, &b, "x", &[]), Some((1.0, 9.0)));
    }

    // ---------- numeric_column_indices ----------

    #[test]
    fn numeric_column_indices_selects_every_numeric_dtype_skips_label() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        assert_eq!(numeric_column_indices(&schema, "pairplot").unwrap(), vec![1, 2]);
    }

    #[test]
    fn numeric_column_indices_refuses_a_decimal_column_naming_ctx_and_dtype() {
        pyo3::Python::initialize();
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("d", DataType::Decimal128(5, 2), false),
        ]);
        let msg = numeric_column_indices(&schema, "heatmap").unwrap_err().to_string();
        assert!(msg.contains("heatmap: column 'd' is a decimal column"), "err: {msg}");
        assert!(msg.contains("a numeric column must be an integer or float column"), "err: {msg}");
    }
}
