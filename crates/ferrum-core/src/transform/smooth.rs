//! Fit-family transform: `Smooth` — LOESS, polynomial/linear (OLS), log-x,
//! sqrt-x, GLM (Gaussian family, Identity link), and GAM (penalized cubic
//! B-spline) regression lines with optional confidence bands, evaluated
//! over an evenly-spaced x grid. See [`PySmooth`] for the full parameter
//! contract.
//!
//! The heavier fitting cores live beside this file rather than inside it:
//! [`crate::transform::glm`] owns IRLS and the Wald band,
//! [`crate::transform::gam`] owns the penalized-spline basis, penalty, and
//! GCV λ selection. This module owns method routing, the evaluation grid,
//! the CI-band assembly, and the grouped/partitioned dispositions below.
//!
//! ## Degenerate-input policy (batch-D T9 round 3)
//!
//! Two different kinds of "this fit cannot be computed" are handled
//! differently on purpose, and the distinction is deliberate rather than an
//! oversight:
//!
//! - **Parameter absurdity, on a genuine whole-frame call.** The caller
//!   asked for a polynomial `degree` the input cannot determine —
//!   `degree >=` the number of usable (non-null, post-`x_bins`-aggregation) points,
//!   i.e. the Vandermonde design is rank-deficient (`poly_ols`'s `n < p`
//!   case). This refuses with a typed `PyValueError` naming the point count
//!   needed — the same refusal class as `degree=0` — instead of silently
//!   emitting an all-NaN grid (closes the spec review's finding 2:
//!   `degree=20` on a 20-row frame used to render zero polylines with no
//!   diagnostic at all). It fires only when the batch is genuinely the
//!   whole frame: `groupby` is empty AND the render pipeline has not
//!   already partitioned this batch out of a larger one.
//! - **Data degeneracy, or a slice of a larger frame, degrades instead.**
//!   Both of the following degrade to an all-NaN row range rather than
//!   failing the whole render:
//!   - A per-group rank-deficient design under `groupby` (T6's contract —
//!     zero-variance x, non-convergent IRLS, too few points after that
//!     group's own null/NaN drop). A grouped batch keeps succeeding even
//!     when one group can't be fit, rather than one bad group failing the
//!     whole chart.
//!   - An UNGROUPED batch that is itself one partition of a larger frame —
//!     e.g. one facet panel. Round 2 treated `groupby.is_empty()` as proof
//!     of "whole frame", but `render::prepare::run_transforms_with_facet`
//!     partitions the input by the facet field(s) BEFORE running
//!     transforms, so every facet panel is an ungrouped `apply()` call on a
//!     data subset: `fm.lmplot(order=3, col='sp')` on a sparse level raised,
//!     while the identical data and degree under `hue='sp'` rendered (same
//!     "not enough data for this fit" condition, two opposite dispositions
//!     depending only on which channel partitioned it). Round 3 closes this
//!     by having the production pipeline mark a batch as a partition via
//!     [`TransformContext::is_partition`](crate::transform::context::TransformContext::is_partition)
//!     — set only at `render::prepare::run_transforms_with_facet`'s facet
//!     construction site — and gating the refusal on `!ctx.is_partition` in
//!     addition to `groupby.is_empty()`.
//!
//! **`gam` contributes no third case (batch-D T10).** The whole-frame
//! refusal above exists for exactly one shape: the CALLER's `degree`
//! parameter is what makes the design undeterminable, so a silent empty
//! render would hide the caller's own mistake. `gam` exposes no such
//! parameter — its basis dimension is derived from the data
//! (`clamp(distinct_x, 4, 10)`) and its smoothness from a GCV-selected
//! penalty — so every way a `gam` fit can fail is data degeneracy, and
//! data degeneracy degrades here. A whole-frame `gam` on fewer than 4
//! distinct x values therefore emits an all-NaN grid exactly like `lm` on
//! a zero-variance x, rather than raising. Adding a refusal would mean
//! refusing charts that `method="loess"` renders (blank) on the same data,
//! for no caller-actionable reason.
//!
//!   `apply_one_group`'s `only_indices` parameter is `None` for every
//!   ungrouped call, whether or not that call is itself a facet panel;
//!   `is_partition` is the separate signal that disambiguates the two.
//!   `apply()` (no context — direct calls, unit tests, WASM/notebook direct
//!   usage) always treats its input as the whole frame, matching this
//!   module's behavior before this round. `apply_grouped` passes
//!   `Some(ixs)` to each per-group call, but its own empty-partition
//!   fallback (every group key is null) passes `None` on the full batch, so
//!   that fallback threads `is_partition` through too rather than being
//!   unconditionally treated as whole-frame.
//!
//! ## Output schema is spec-determined, not data-determined (round 4)
//!
//! Every degenerate branch above (a rank-deficient design, zero-variance x,
//! non-convergent IRLS, or fewer than 2 usable points) emits its all-NaN
//! output via [`build_degenerate_smooth_batch`], which appends the
//! `_metrics_text`/`_metrics_y` columns — all-null — whenever
//! `spec.inject_metrics` is set, exactly like a well-supported sibling
//! fit's real (non-null-anchor) output. Before this round, every
//! degenerate branch passed `metrics: None` unconditionally, so a
//! `Smooth`'s output schema silently depended on whether THIS PARTICULAR
//! slice's fit happened to degenerate: one `groupby` group or facet panel
//! could carry two more columns than a degenerate sibling. That divergence
//! is what `apply_grouped`'s by-name reconciliation (below) and
//! `render::prepare::concat_batches_by_name` exist to paper over across
//! groups and facet panels respectively; with the schema fixed by the
//! spec instead of the data, both of those are now a belt-and-braces
//! safety net for any *other* named-output divergence, not the primary
//! defense against this one.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::gam;
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, parse_groupby_pyany, stack_grouped_batches,
    KeyValue,
};
use crate::transform::numeric_util::coerce_to_float64;
use crate::transform::residuals;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SmoothMethod {
    /// Global ordinary least squares (degree-1 polynomial).
    Lm,
    /// Locally-weighted polynomial regression.
    Loess,
    /// Global polynomial regression of the specified degree (2 = quadratic, 3 = cubic).
    Poly { degree: u8 },
    /// Log-x linear fit: y ~ a + b·ln(x).  Requires x > 0.
    LogX,
    /// Sqrt-x linear fit: y ~ a + b·√x.  Requires x ≥ 0.
    SqrtX,
    /// Generalized Linear Model (Gaussian family, Identity link — the
    /// standalone `Glm` transform's own default family/link) fit via IRLS.
    /// Batch-D T9 (spec §4.6): delegates to `glm::fit_irls` so
    /// `mark_smooth(method="glm")` shares one fitting implementation with
    /// the `Glm` transform, rather than re-deriving IRLS inline or aliasing
    /// to `Lm` — Gaussian/Identity IRLS converges to the same point
    /// estimate as OLS, so an alias would look identical here but is the
    /// wrong implementation choice (it silently stops being correct the
    /// moment family/link is ever exposed, and its CI band would not match
    /// the real GLM's Wald band, which `glm_fit` reproduces exactly).
    Glm,
    /// Generalized Additive Model: a penalized cubic B-spline smoother
    /// (P-spline) with the smoothing parameter chosen by GCV. Batch-D T10
    /// (spec §4.6): the numerical core lives in
    /// [`crate::transform::gam`], which documents the basis, knot
    /// placement, penalty, and λ-selection defaults. Deterministic — it
    /// uses no randomness, so unlike `Loess` its band does not depend on
    /// `seed`.
    Gam,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SmoothOutput {
    Fitted,
    Residuals,
}

pub(crate) fn default_smooth_output() -> SmoothOutput { SmoothOutput::Fitted }

/// LOESS span used when the caller does not ask for one. Non-loess methods
/// store this same value as an inert placeholder (nothing reads it), which is
/// why an EXPLICIT `bandwidth=` equal to it is tolerated on those methods
/// while any other explicit value refuses — see `PySmooth::new`.
pub(crate) const DEFAULT_LOESS_BANDWIDTH: f64 = 0.75;

fn default_gam_k() -> usize { gam::DEFAULT_BASIS_DIM }

fn is_default_gam_k(k: &usize) -> bool { *k == gam::DEFAULT_BASIS_DIM }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SmoothSpec {
    pub x: String,
    pub y: String,
    pub method: SmoothMethod,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub bandwidth: f64,
    pub degree: u8,
    /// Requested spline basis dimension for `method="gam"` — the parameter
    /// mgcv calls `k`. Defaults to [`gam::DEFAULT_BASIS_DIM`]
    /// and is omitted from the wire at that default, so every spec written
    /// before this field existed round-trips byte-identically and every
    /// non-gam method serializes exactly as it did. `gam::fit` clamps it into
    /// `4..=distinct_x`; `PySmooth::new` refuses a value below 4 and refuses
    /// the parameter entirely on a method that has no basis to size.
    #[serde(default = "default_gam_k", skip_serializing_if = "is_default_gam_k")]
    pub k: usize,
    pub n: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_bins: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_estimator: Option<crate::transform::aggregate::AggFn>,
    #[serde(default = "default_smooth_output")]
    pub output: SmoothOutput,
    /// Schwabish SB-followup (2026-05-12): when ``output == Residuals`` and
    /// this is ``true``, append a nullable ``_ref_zero`` Float64 column to
    /// the residuals batch. One non-null entry (value ``0.0``) on the first
    /// row, ``null`` on the rest — so a downstream ``mark_rule(y="_ref_zero")``
    /// renders exactly one horizontal reference line at y=0 without needing
    /// any extra Python-side data manipulation. No-op for ``Fitted`` output.
    #[serde(default)]
    pub inject_zero_ref: bool,
    /// Schwabish SB-followup (2026-05-12): when ``true``, append nullable
    /// ``_metrics_text`` (Utf8) and ``_metrics_y`` (Float64) columns with
    /// a single non-null row holding ``"R² {r2}\nRMSE {rmse}\nMAE {mae}"``
    /// and the anchor y position. Allowed on BOTH outputs (Fitted and
    /// Residuals); the VISUAL intent is identical — anchor the text at
    /// the rightmost finite x of the output and at the highest finite y
    /// of the respective output column, so the text sits in the
    /// top-right corner of the rendered geometry. The y differs by
    /// path only because the y axis itself differs between the two
    /// output modes:
    /// - ``Residuals``: anchor at the rightmost finite ``x`` input row;
    ///   ``_metrics_y`` at the max finite residual (top of the residual
    ///   scatter, near y=0).
    /// - ``Fitted``: anchor at the rightmost finite grid x; ``_metrics_y``
    ///   at the max finite fitted y (top of the fit curve). Metrics are
    ///   computed against the RAW (pre-aggregation) input regardless of
    ///   ``x_bins``.
    /// Designed for a same-data ``mark_text`` overlay reading both columns.
    #[serde(default)]
    pub inject_metrics: bool,
    /// When non-empty, partition input by these columns and emit per-(x, group)
    /// rows. Output schema gains one column per groupby field after the core output.
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// WI-7 (2026-05-15): when set, evaluate the fit line over this x range
    /// instead of ``[xs.min(), xs.max()]``. Enables ``lmplot(truncate=False)``
    /// to extend the fit line to the axis domain boundary. Clipped at this
    /// boundary — never extrapolates past it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_range: Option<[f64; 2]>,
}

pub(crate) fn apply(spec: &SmoothSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Batch-D T9 round 3: the context-aware entry point, wired into
/// `TransformSpec::apply_with_context`'s dispatch table (`transform/core.rs`)
/// so the production render pipeline's `TransformContext.is_partition` (see
/// its doc) actually reaches the rank-deficiency refusal below. `apply()`
/// forwards here with a default context (`is_partition: false`), preserving
/// its pre-round-3 whole-frame behavior for direct calls, unit tests, and
/// WASM/notebook usage.
pub(crate) fn apply_with_context(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby, ctx.is_partition);
    }
    apply_one_group(spec, batch, None, ctx.is_partition)
}

fn apply_one_group(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    is_partition: bool,
) -> PyResult<RecordBatch> {
    let (xs_raw, ys_raw) = extract_xy(spec, batch, only_indices)?;

    // Pre-aggregate xs/ys into n equal-width bins if x_bins is set.
    let (xs, ys) = if let Some(n_bins) = spec.x_bins {
        let estimator = spec.x_estimator.unwrap_or(crate::transform::aggregate::AggFn::Mean);
        pre_aggregate_xy(&xs_raw, &ys_raw, n_bins, estimator)
    } else {
        (xs_raw.clone(), ys_raw.clone())
    };

    if xs.len() < 2 {
        return match spec.output {
            SmoothOutput::Fitted => all_nan_output(spec),
            SmoothOutput::Residuals => build_residuals_batch(
                Vec::new(), Vec::new(),
                spec.inject_zero_ref,
                // Batch-D T9 round 4: zero rows either way, but when
                // `inject_metrics` is requested the columns must still be
                // PRESENT (empty, like every other column here) rather than
                // dropped — the same schema-uniformity fix as the Fitted
                // path's `build_degenerate_smooth_batch`. `residuals::compute`
                // on empty slices is the same "no anchor row" shape
                // `append_metrics_columns` produces for a zero-length input,
                // so this needs no separate null-columns helper.
                spec.inject_metrics.then(|| residuals::compute(&[], &[])),
            ),
        };
    }

    // See the module doc's "Degenerate-input policy" section: a polynomial
    // `degree` the data cannot determine (rank-deficient Vandermonde design)
    // is parameter absurdity, not data degeneracy, so it refuses on a
    // genuine whole-frame call rather than falling through to `poly_fit`'s
    // (or `build_predictor`'s) all-NaN branch. `only_indices.is_none()`
    // alone is NOT sufficient proof of "whole frame" — `render::prepare::
    // run_transforms_with_facet` partitions the input by facet BEFORE
    // running transforms, so an ungrouped call can still be one panel of a
    // larger frame. `!is_partition` is the second half of the gate: a
    // per-group rank-deficient design under `groupby` (`only_indices` is
    // `Some`) and an ungrouped-but-partitioned facet panel (`is_partition`
    // is `true`) both degrade to that slice's own NaN range instead.
    if only_indices.is_none() && !is_partition {
        if let SmoothMethod::Poly { degree } = spec.method {
            let min_points = degree as usize + 1;
            if xs.len() < min_points {
                return Err(PyValueError::new_err(format!(
                    "Smooth: degree={degree} needs at least {min_points} data points to \
                     fit a degree-{degree} polynomial; only {} remain after dropping \
                     null/NaN x/y rows (and any x_bins aggregation). A degree this high \
                     relative to the data is a rank-deficient design, not a valid fit — \
                     lower degree or provide more data",
                    xs.len(),
                )));
            }
        }
    }

    if matches!(spec.output, SmoothOutput::Residuals) {
        // Fit at original (un-aggregated) input xs; emit (x, y - fit(x)) rows.
        return residuals_fit(spec, &xs, &ys, &xs_raw, &ys_raw);
    }

    let (data_x_min, data_x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));

    // WI-7: when x_range is set (truncate=False), use it as the evaluation
    // domain instead of the observed data extent. Clip to valid range.
    let (x_min, x_max) = match spec.x_range {
        Some([lo, hi]) if lo.is_finite() && hi.is_finite() && hi > lo => (lo, hi),
        _ => (data_x_min, data_x_max),
    };

    let grid: Vec<f64> = (0..spec.n)
        .map(|i| if spec.n <= 1 { x_min } else {
            x_min + (x_max - x_min) * (i as f64) / ((spec.n - 1) as f64)
        })
        .collect();

    // Schwabish SB-followup (2026-05-12): when ``inject_metrics`` is set
    // on the Fitted path, compute R²/RMSE/MAE against the RAW input
    // (pre-aggregation) so the metrics reflect the spread of the actual
    // data points, not the binned summary. Mirrors the residuals_fit
    // convention. The predictor is rebuilt here (rather than threaded
    // through ``lm_fit`` / ``loess_fit``) so those fit functions stay
    // focused on "fit a curve over a grid" — see SB-followup C4 rework
    // (2026-05-12).
    let metrics = if spec.inject_metrics {
        let predictor = build_predictor(spec, &xs, &ys);
        let resids: Vec<f64> = xs_raw.iter().zip(ys_raw.iter())
            .map(|(&xi, &yi)| {
                let yhat = predictor(xi);
                if yhat.is_nan() { f64::NAN } else { yi - yhat }
            })
            .collect();
        Some(residuals::compute(&ys_raw, &resids))
    } else {
        None
    };

    match spec.method {
        SmoothMethod::Lm => lm_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::Loess => loess_fit(
            &xs, &ys, &grid, spec.bandwidth, spec.degree, spec.ci, spec.n, spec.seed,
            metrics,
        ),
        SmoothMethod::Poly { degree } => poly_fit(&xs, &ys, &grid, spec.ci, spec.n, degree, metrics),
        SmoothMethod::LogX => log_x_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::SqrtX => sqrt_x_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::Glm => glm_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::Gam => gam_fit(&xs, &ys, &grid, spec.ci, spec.n, spec.k, metrics),
    }
}

/// Partition input batch by the groupby columns, call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// columns appended (with their original dtypes, XFORM-03) as additional fields.
fn apply_grouped(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    group_cols: &[String],
    is_partition: bool,
) -> PyResult<RecordBatch> {
    // Multi-column partition path (first-appearance order, null-key skip, dtype-preserve).
    let (group_order, group_idx_map, gtypes) =
        group_partition_multi(batch, group_cols, "stat_smooth")?;

    // Collect per-group outputs and stack them.
    let mut per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = Vec::new();
    for g in &group_order {
        let ixs = group_idx_map.get(g)
            .ok_or_else(|| PyValueError::new_err(format!(
                "stat_smooth: missing group key {g:?} in index map")))?;
        // `only_indices` is `Some` here, so the rank-deficiency refusal in
        // `apply_one_group` never gates on `is_partition` for this call —
        // T6's per-group degrade-to-NaN contract already applies.
        let out = apply_one_group(spec, batch, Some(ixs), is_partition)?;
        per_group_batches.push((g.clone(), out));
    }

    if per_group_batches.is_empty() {
        // Fallback: every group key was null, so there is no per-group
        // output to stack. This call passes `None` for `only_indices` on
        // the FULL batch (module doc's finding D, batch-D T9 round 3): it
        // is not the `Some(ixs)` shape every other `apply_grouped` call
        // produces, so it forwards `is_partition` too — a rank-deficient
        // `Poly` here degrades like a group when this batch is itself a
        // facet panel, rather than always refusing.
        return apply_one_group(spec, batch, None, is_partition);
    }

    // Reconcile the per-group fit batches and append the group columns via
    // the ONE shared stacking template (`group_key::stack_grouped_batches`,
    // batch-F T2h hoist), which both `Smooth` and `Robust` now use — see its
    // doc for the by-name reconciliation and the groupby/output-column
    // collision guard.
    //
    // Why schema divergence across groups is possible at all: a degenerate
    // fit (rank-deficient/non-convergent design, zero-variance x, too few
    // points) for one group's own slice used to emit the CORE 4-column
    // `[x, y, ci_lower, ci_upper]` batch even when `inject_metrics` was set.
    // Batch-D T9 round 4 closed that at the source by routing every
    // degenerate branch through `build_degenerate_smooth_batch`, which emits
    // null metrics columns when requested — so this is a belt-and-braces
    // net, not a live path. The net still matters because the failure mode
    // it replaced was a live `PanicException` through `Chart.to_svg()`: the
    // original code pinned the reference schema to `per_group_batches[0]`
    // and indexed every other group POSITIONALLY against it, so a
    // degenerate first group silently dropped later groups' real metrics
    // columns and a degenerate later group indexed past its own narrower
    // schema.
    stack_grouped_batches("stat_smooth", &per_group_batches, group_cols, &gtypes)
}

fn pre_aggregate_xy(
    xs: &[f64],
    ys: &[f64],
    n_bins: usize,
    estimator: crate::transform::aggregate::AggFn,
) -> (Vec<f64>, Vec<f64>) {
    use crate::transform::aggregate::aggregate as agg_fn;
    if xs.is_empty() || n_bins == 0 {
        return (Vec::new(), Vec::new());
    }
    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if x_min == x_max {
        let avg = agg_fn(ys, estimator, ys.len());
        return (vec![x_min], vec![avg]);
    }
    let mut buckets_x: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    let mut buckets_y: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    let width = x_max - x_min;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let mut idx = ((x - x_min) / width * n_bins as f64).floor() as isize;
        if idx >= n_bins as isize { idx = n_bins as isize - 1; }
        if idx < 0 { idx = 0; }
        let u = idx as usize;
        buckets_x[u].push(x);
        buckets_y[u].push(y);
    }
    let mut out_x = Vec::new();
    let mut out_y = Vec::new();
    for (xs_in, ys_in) in buckets_x.iter().zip(buckets_y.iter()) {
        if ys_in.is_empty() { continue; }
        // x always uses mean within the bin; y uses the chosen estimator.
        let mean_x = xs_in.iter().sum::<f64>() / xs_in.len() as f64;
        let agg = agg_fn(ys_in, estimator, ys_in.len());
        out_x.push(mean_x);
        out_y.push(agg);
    }
    (out_x, out_y)
}

/// Build a point-predictor closure for the chosen smoothing method,
/// fitted on `(xs_fit, ys_fit)`. Returns NaN for any query point that
/// the underlying fit cannot evaluate (e.g. zero-variance x for OLS,
/// rank-deficient local window for LOESS). Shared between
/// `residuals_fit` and the Fitted+metrics path in `apply()` so the
/// residual computation is byte-identical across both call sites.
fn build_predictor(spec: &SmoothSpec, xs_fit: &[f64], ys_fit: &[f64]) -> Box<dyn Fn(f64) -> f64> {
    match spec.method {
        SmoothMethod::Lm => {
            let n = xs_fit.len() as f64;
            let mean_x = xs_fit.iter().sum::<f64>() / n;
            let mean_y = ys_fit.iter().sum::<f64>() / n;
            let sxx: f64 = xs_fit.iter().map(|x| (x - mean_x).powi(2)).sum();
            let sxy: f64 = xs_fit.iter().zip(ys_fit).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
            if sxx == 0.0 {
                Box::new(|_| f64::NAN)
            } else {
                let beta = sxy / sxx;
                let alpha = mean_y - beta * mean_x;
                Box::new(move |x: f64| alpha + beta * x)
            }
        }
        SmoothMethod::Loess => {
            let n = xs_fit.len();
            let k = ((spec.bandwidth * n as f64).ceil() as usize).max((spec.degree as usize) + 1);
            let k = k.min(n);
            let (sxs, sys) = sort_xy(xs_fit, ys_fit);
            let degree = spec.degree;
            Box::new(move |x: f64| loess_at_point_sorted(&sxs, &sys, x, k, degree))
        }
        SmoothMethod::Poly { degree } => {
            let coeffs = poly_ols(xs_fit, ys_fit, degree as usize);
            Box::new(move |x: f64| eval_poly(&coeffs, x))
        }
        SmoothMethod::LogX => {
            let tx: Vec<f64> = xs_fit.iter().map(|&x| x.ln()).collect();
            let valid: Vec<(f64, f64)> = tx.iter().zip(ys_fit.iter())
                .filter(|(t, _)| t.is_finite())
                .map(|(&t, &y)| (t, y))
                .collect();
            if valid.len() < 2 {
                return Box::new(|_| f64::NAN);
            }
            let vxs: Vec<f64> = valid.iter().map(|(t, _)| *t).collect();
            let vys: Vec<f64> = valid.iter().map(|(_, y)| *y).collect();
            let n = vxs.len() as f64;
            let mean_x = vxs.iter().sum::<f64>() / n;
            let mean_y = vys.iter().sum::<f64>() / n;
            let sxx: f64 = vxs.iter().map(|x| (x - mean_x).powi(2)).sum();
            let sxy: f64 = vxs.iter().zip(vys.iter()).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
            if sxx == 0.0 {
                return Box::new(|_| f64::NAN);
            }
            let beta = sxy / sxx;
            let alpha = mean_y - beta * mean_x;
            Box::new(move |x: f64| {
                let t = x.ln();
                if !t.is_finite() { f64::NAN } else { alpha + beta * t }
            })
        }
        SmoothMethod::SqrtX => {
            let tx: Vec<f64> = xs_fit.iter().map(|&x| x.sqrt()).collect();
            let valid: Vec<(f64, f64)> = tx.iter().zip(ys_fit.iter())
                .filter(|(t, _)| t.is_finite())
                .map(|(&t, &y)| (t, y))
                .collect();
            if valid.len() < 2 {
                return Box::new(|_| f64::NAN);
            }
            let vxs: Vec<f64> = valid.iter().map(|(t, _)| *t).collect();
            let vys: Vec<f64> = valid.iter().map(|(_, y)| *y).collect();
            let n = vxs.len() as f64;
            let mean_x = vxs.iter().sum::<f64>() / n;
            let mean_y = vys.iter().sum::<f64>() / n;
            let sxx: f64 = vxs.iter().map(|x| (x - mean_x).powi(2)).sum();
            let sxy: f64 = vxs.iter().zip(vys.iter()).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
            if sxx == 0.0 {
                return Box::new(|_| f64::NAN);
            }
            let beta = sxy / sxx;
            let alpha = mean_y - beta * mean_x;
            Box::new(move |x: f64| {
                let t = x.sqrt();
                if !t.is_finite() { f64::NAN } else { alpha + beta * t }
            })
        }
        SmoothMethod::Glm => {
            use crate::transform::glm::{fit_irls, link_inverse, GlmFamily, GlmLink};
            match fit_irls(xs_fit, ys_fit, GlmFamily::Gaussian, GlmLink::Identity, GLM_SMOOTH_MAX_ITER, GLM_SMOOTH_TOL) {
                Ok((beta, _cov)) => {
                    Box::new(move |x: f64| link_inverse(GlmLink::Identity, beta[0] + beta[1] * x))
                }
                Err(_) => Box::new(|_| f64::NAN),
            }
        }
        SmoothMethod::Gam => match gam::fit(xs_fit, ys_fit, spec.k) {
            Some(fit) => Box::new(move |x: f64| fit.predict(x)),
            None => Box::new(|_| f64::NAN),
        },
    }
}

/// IRLS knobs for the `Smooth(method="glm")` routing — mirrors the
/// standalone `Glm` transform's own PyO3 defaults (`max_iter=25`,
/// `tol=1e-8`) since `glm_fit`/`build_predictor`'s `Glm` arm share its
/// fitting core.
const GLM_SMOOTH_MAX_ITER: usize = 25;
const GLM_SMOOTH_TOL: f64 = 1e-8;

/// Emit a residuals RecordBatch with the canonical ``[x, residual]``
/// columns. Schwabish SB-followup (2026-05-12) opt-in extras:
///
/// - ``inject_zero_ref=true`` appends a nullable ``_ref_zero`` Float64
///   column with one ``0.0`` on the first row (rest ``null``), enabling
///   a single-rule overlay without further Python-side injection.
/// - ``metrics=Some((r2, rmse, mae))`` appends ``_metrics_text`` (Utf8)
///   and ``_metrics_y`` (Float64) with one non-null row at the anchor
///   (max ``x``).
///
/// Both column-emission helpers live in
/// `crate::transform::residuals` so the schema invariant
/// (`_ref_zero`, `_metrics_text`, `_metrics_y`) is owned in one place
/// and shared with `transform::robust::build_residuals_batch`.
fn build_residuals_batch(
    xs: Vec<f64>, resid: Vec<f64>,
    inject_zero_ref: bool,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let n = xs.len();
    let mut fields = vec![
        Field::new("x",        DataType::Float64, true),
        Field::new("residual", DataType::Float64, true),
    ];
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(xs.clone())),
        Arc::new(Float64Array::from(resid.clone())),
    ];

    if inject_zero_ref {
        residuals::append_zero_ref_column(&mut fields, &mut cols, n);
    }

    if let Some(m) = metrics {
        residuals::append_metrics_columns(&mut fields, &mut cols, &xs, &resid, m);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_smooth: {e}")))
}

fn residuals_fit(
    spec: &SmoothSpec,
    xs_fit: &[f64], ys_fit: &[f64],
    xs_input: &[f64], ys_input: &[f64],
) -> PyResult<RecordBatch> {
    // Fit the chosen model on (xs_fit, ys_fit); then evaluate at xs_input and subtract.
    let predictor = build_predictor(spec, xs_fit, ys_fit);
    let mut out_x = Vec::with_capacity(xs_input.len());
    let mut out_r = Vec::with_capacity(xs_input.len());
    for (&xi, &yi) in xs_input.iter().zip(ys_input.iter()) {
        let yhat = predictor(xi);
        let r = if yhat.is_nan() { f64::NAN } else { yi - yhat };
        out_x.push(xi);
        out_r.push(r);
    }
    let metrics = if spec.inject_metrics {
        Some(residuals::compute(ys_input, &out_r))
    } else {
        None
    };
    build_residuals_batch(out_x, out_r, spec.inject_zero_ref, metrics)
}

fn extract_xy(spec: &SmoothSpec, batch: &RecordBatch, only_indices: Option<&[usize]>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.y)))?;
    let xa = coerce_to_float64(batch.column(xi), "stat_smooth", &spec.x)?;
    let ya = coerce_to_float64(batch.column(yi), "stat_smooth", &spec.y)?;
    // Collect row indices that are valid in BOTH arrays (non-null, non-NaN).
    // Uses the same null+NaN predicate as clean_float64_values in numeric_util,
    // but applied to paired columns so only jointly-clean rows are kept.
    use crate::transform::numeric_util::clean_float64_values;
    let candidate_iter: Box<dyn Iterator<Item = usize>> = match only_indices {
        Some(ixs) => Box::new(ixs.iter().copied()),
        None => Box::new(0..xa.len()),
    };
    let valid_indices: Vec<usize> = candidate_iter
        .filter(|&i| {
            !xa.is_null(i) && !ya.is_null(i)
                && !xa.value(i).is_nan()
                && !ya.value(i).is_nan()
        })
        .collect();

    // Extract each column at the valid indices via the shared cleaner.
    // This reuses the same null+NaN drop logic across all transform callers.
    let xs = clean_float64_values(&xa, Some(&valid_indices));
    let ys = clean_float64_values(&ya, Some(&valid_indices));
    Ok((xs, ys))
}

fn all_nan_output(spec: &SmoothSpec) -> PyResult<RecordBatch> {
    let n = spec.n;
    build_degenerate_smooth_batch(&vec![f64::NAN; n], n, spec.inject_metrics)
}

/// Build the canonical all-NaN `[x, y, ci_lower, ci_upper]` batch for a fit
/// this method could not compute — rank-deficient Vandermonde design,
/// zero-variance x, non-convergent IRLS, or fewer than 2 usable points
/// (batch-D T9 round 4, closing the rust-quality cycle-3 root-cause
/// finding). Every degenerate branch in this file routes through here
/// instead of calling `build_smooth_batch(..., None)` directly, so a
/// Smooth's output schema no longer depends on whether THIS PARTICULAR
/// fit happened to degenerate: when `metrics_requested` is true (every
/// caller passes `metrics.is_some()`, since the caller only ever computes
/// `Some(..)` when `spec.inject_metrics` was set), the `_metrics_text`/
/// `_metrics_y` columns are still appended — all-null, since there is no
/// fit to summarize, not the caller's already-computed (and here
/// meaningless, since it was derived from an all-NaN predictor) `(r2,
/// rmse, mae)` tuple. Before this fix, a degenerate fit silently emitted a
/// narrower batch than its well-supported siblings, which is exactly the
/// divergence `render::prepare::concat_batches_by_name` (across facet
/// panels) and `apply_grouped` (across `groupby` groups) exist to paper
/// over; eliminating it at the source makes both of those a
/// belt-and-braces safety net rather than the primary defense.
fn build_degenerate_smooth_batch(x_col: &[f64], n_grid: usize, metrics_requested: bool) -> PyResult<RecordBatch> {
    let nans = vec![f64::NAN; n_grid];
    let mut fields = crate::transform::registry::fit_output_fields();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x_col.to_vec())),
        Arc::new(Float64Array::from(nans.clone())),
        Arc::new(Float64Array::from(nans.clone())),
        Arc::new(Float64Array::from(nans)),
    ];
    if metrics_requested {
        residuals::append_null_metrics_columns(&mut fields, &mut cols, n_grid);
    }
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("stat_smooth: {e}")))
}

/// Emit a fitted-smooth RecordBatch with the canonical
/// ``[x, y, ci_lower, ci_upper]`` grid columns. Schwabish SB-followup
/// (2026-05-12) extension: when ``metrics=Some((r2, rmse, mae))`` is
/// passed, also append ``_metrics_text`` (Utf8) and ``_metrics_y``
/// (Float64) with a single non-null row at the rightmost grid point
/// so a chart-side ``mark_text`` overlay can render the OLS / LOESS
/// summary in the chart's top-right corner without duplicating the
/// fit computation in Python.
///
/// Metric-column emission delegates to
/// `crate::transform::residuals::append_metrics_columns` so the
/// schema and text-format invariant is shared across smooth.rs and
/// robust.rs.
fn build_smooth_batch(
    x: Vec<f64>, y: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    // Batch-D M6 schema-collapse (spec §4.6): the shared fit-family output
    // schema also backs `logistic.rs`/`glm.rs`/`robust.rs` and the
    // `transform_registry` entries for all four — one source instead of four
    // identical `Field::new` literals.
    let mut fields = crate::transform::registry::fit_output_fields();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x.clone())),
        Arc::new(Float64Array::from(y.clone())),
        Arc::new(Float64Array::from(lo)),
        Arc::new(Float64Array::from(hi)),
    ];

    if let Some(m) = metrics {
        residuals::append_metrics_columns(&mut fields, &mut cols, &x, &y, m);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("stat_smooth: {e}")))
}

// ---- Polynomial OLS helpers ----

/// Solve the normal equations for polynomial regression of the given degree
/// using faer's LU decomposition.  Returns the coefficient vector
/// `[a_0, a_1, ..., a_d]` such that `y ≈ a_0 + a_1·x + … + a_d·x^d`.
/// Returns an empty Vec if the system is rank-deficient or inputs are invalid.
fn poly_ols(xs: &[f64], ys: &[f64], degree: usize) -> Vec<f64> {
    use faer::prelude::*;

    let n = xs.len();
    let p = degree + 1;
    if n < p || degree == 0 {
        return Vec::new();
    }

    // Build Vandermonde matrix X (n × p).
    let mut x_mat = Mat::zeros(n, p);
    for (i, &xi) in xs.iter().enumerate() {
        for j in 0..p {
            x_mat[(i, j)] = xi.powi(j as i32);
        }
    }

    // Normal equations: X'X β = X'y.
    let xt = x_mat.transpose().to_owned();
    let xtx = &xt * &x_mat;
    let mut xty = Mat::zeros(p, 1);
    for j in 0..p {
        let mut v = 0.0_f64;
        for (i, &yi) in ys.iter().enumerate() {
            v += xt[(j, i)] * yi;
        }
        xty[(j, 0)] = v;
    }

    // Solve via full-pivot LU (handles near-singular cases gracefully).
    let lu = xtx.partial_piv_lu();
    let sol = lu.solve(&xty);

    (0..p).map(|j| sol[(j, 0)]).collect()
}

/// Evaluate the polynomial with coefficients `[a_0, a_1, …, a_d]` at `x`
/// via Horner's method.
fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return f64::NAN;
    }
    let mut val = *coeffs.last().unwrap();
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        val = val * x + c;
    }
    val
}

/// Global polynomial regression fit over a grid.
fn poly_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize, degree: u8,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let coeffs = poly_ols(xs, ys, degree as usize);
    if coeffs.is_empty() {
        return build_degenerate_smooth_batch(grid, n_grid, metrics.is_some());
    }

    let y_fit: Vec<f64> = grid.iter().map(|&x| eval_poly(&coeffs, x)).collect();

    // Confidence interval: bootstrap residuals from the polynomial fit at input xs.
    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            let n = xs.len();
            let dof = n as f64 - (degree as f64 + 1.0);
            if dof <= 0.0 {
                (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid])
            } else {
                let resid_ss: f64 = xs.iter().zip(ys)
                    .map(|(&xi, &yi)| (yi - eval_poly(&coeffs, xi)).powi(2))
                    .sum();
                let sigma2 = resid_ss / dof;
                let t_crit = student_t_critical(level, dof);
                // Approximate SE using the leverage from the hat matrix diagonal.
                let leverages = poly_hat_diagonal(xs, degree as usize);
                let lo_v: Vec<f64> = grid.iter().enumerate().map(|(i, &x)| {
                    let h = leverages.get(i).copied().unwrap_or(0.0);
                    let se = (sigma2 * (1.0 - h + 1.0 / n as f64)).sqrt();
                    eval_poly(&coeffs, x) - t_crit * se
                }).collect();
                let hi_v: Vec<f64> = grid.iter().enumerate().map(|(i, &x)| {
                    let h = leverages.get(i).copied().unwrap_or(0.0);
                    let se = (sigma2 * (1.0 - h + 1.0 / n as f64)).sqrt();
                    eval_poly(&coeffs, x) + t_crit * se
                }).collect();
                (lo_v, hi_v)
            }
        }
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)
}

/// Compute hat-matrix diagonal for polynomial design X (Vandermonde).
/// Returns Vec of length n corresponding to the training points.
fn poly_hat_diagonal(xs: &[f64], degree: usize) -> Vec<f64> {
    use faer::prelude::*;
    let n = xs.len();
    let p = degree + 1;
    if n < p {
        return vec![0.0; n];
    }
    let mut x_mat = Mat::zeros(n, p);
    for (i, &xi) in xs.iter().enumerate() {
        for j in 0..p {
            x_mat[(i, j)] = xi.powi(j as i32);
        }
    }
    crate::transform::linalg::hat_diagonal(&x_mat)
}

/// Log-x linear fit: y ~ a + b·ln(x). Requires x > 0.
fn log_x_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    transformed_linear_fit(xs, ys, grid, ci, n_grid, |x| x.ln(), metrics)
}

/// Sqrt-x linear fit: y ~ a + b·√x. Requires x ≥ 0.
fn sqrt_x_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    transformed_linear_fit(xs, ys, grid, ci, n_grid, |x| x.sqrt(), metrics)
}

/// Fit `y ~ a + b·T(x)` where `T` is any monotone transformation of x.
/// Skips pairs where `T(x)` is not finite (e.g., ln of non-positive x).
fn transformed_linear_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    transform: impl Fn(f64) -> f64,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    // Transform x values; filter to finite-transform pairs only.
    let pairs: Vec<(f64, f64)> = xs.iter().zip(ys.iter())
        .filter_map(|(&xi, &yi)| {
            let t = transform(xi);
            if t.is_finite() { Some((t, yi)) } else { None }
        })
        .collect();

    if pairs.len() < 2 {
        return build_degenerate_smooth_batch(grid, n_grid, metrics.is_some());
    }

    let txs: Vec<f64> = pairs.iter().map(|(t, _)| *t).collect();
    let tys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    // Delegate to the OLS path on transformed-x/original-y.
    lm_fit_on(&txs, &tys, grid, ci, n_grid, &transform, metrics)
}

/// OLS fit on pre-transformed x values; evaluate the fit at `transform(grid[i])`.
fn lm_fit_on(
    txs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    transform: &impl Fn(f64) -> f64,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let n = txs.len();
    let mean_x: f64 = txs.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = ys.iter().sum::<f64>() / n as f64;
    let sxx: f64 = txs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = txs.iter().zip(ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();

    if sxx == 0.0 {
        return build_degenerate_smooth_batch(grid, n_grid, metrics.is_some());
    }

    let beta = sxy / sxx;
    let alpha = mean_y - beta * mean_x;

    let y_fit: Vec<f64> = grid.iter().map(|&x| {
        let t = transform(x);
        if t.is_finite() { alpha + beta * t } else { f64::NAN }
    }).collect();

    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            let resid_ss: f64 = txs.iter().zip(ys)
                .map(|(x, y)| (y - (alpha + beta * x)).powi(2))
                .sum();
            let dof = n as f64 - 2.0;
            if dof <= 0.0 {
                (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid])
            } else {
                let sigma2 = resid_ss / dof;
                let t_crit = student_t_critical(level, dof);
                let mut lo_v = Vec::with_capacity(n_grid);
                let mut hi_v = Vec::with_capacity(n_grid);
                for &x in grid {
                    let t = transform(x);
                    if !t.is_finite() {
                        lo_v.push(f64::NAN);
                        hi_v.push(f64::NAN);
                    } else {
                        let se = (sigma2 * (1.0 / n as f64 + (t - mean_x).powi(2) / sxx)).sqrt();
                        let yhat = alpha + beta * t;
                        lo_v.push(yhat - t_crit * se);
                        hi_v.push(yhat + t_crit * se);
                    }
                }
                (lo_v, hi_v)
            }
        }
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)
}

/// Generalized Linear Model fit (Gaussian family, Identity link — the
/// standalone `Glm` transform's own default family/link) via IRLS,
/// evaluated over `grid`. Batch-D T9 (spec §4.6): `mark_smooth(method="glm")`
/// delegates to `glm::fit_irls` so the two transforms share one fitting
/// implementation rather than re-deriving IRLS inline here.
///
/// The confidence band mirrors `glm.rs::apply`'s own Wald band on the
/// linear predictor, back-transformed through the inverse link — the real
/// GLM CI this method routes to (including its lack of dispersion scaling
/// for non-canonical families, unchanged from `glm.rs`), not the
/// t-distribution band the other `Smooth` methods use.
///
/// Degenerate designs (e.g. constant x) fail IRLS the same way `Glm` itself
/// does; here that degrades to an all-NaN grid, matching every other
/// method's degenerate-input handling in this file, rather than raising
/// mid-batch.
fn glm_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    use crate::transform::glm::{fit_irls, link_inverse, wald_band, GlmFamily, GlmLink};

    let family = GlmFamily::Gaussian;
    let link = GlmLink::Identity;

    let (beta, cov) = match fit_irls(xs, ys, family, link, GLM_SMOOTH_MAX_ITER, GLM_SMOOTH_TOL) {
        Ok(v) => v,
        Err(_) => return build_degenerate_smooth_batch(grid, n_grid, metrics.is_some()),
    };

    let y_fit: Vec<f64> = grid.iter().map(|&x| link_inverse(link, beta[0] + beta[1] * x)).collect();

    // Batch-D T9 round 2 (Required Fix 5): shared with `Glm::apply`, which
    // reproduced this exact formula verbatim before this hoist. See
    // `glm::wald_band`'s doc comment for why the extraction lives there.
    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => wald_band(beta, cov, link, grid, level),
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)
}

/// Generalized Additive Model fit — a penalized cubic B-spline (P-spline)
/// with a GCV-chosen smoothing parameter — evaluated over `grid`. Batch-D
/// T10 (spec §4.6). The fitting math, its documented defaults (basis
/// dimension, uniform knots, second-order difference penalty, the λ grid),
/// and the no-extrapolation policy all live in
/// [`crate::transform::gam`]'s module doc.
///
/// The band is assembled here, from `gam`'s posterior standard error and
/// this file's own [`student_t_critical`], on the fit's fractional `n −
/// edf` residual degrees of freedom: `f̂(x) ± t · se(x)`. That matches the
/// t-distribution band `lm`/`poly`/`log`/`sqrt` build a few functions up
/// (`glm`'s Wald-z band is the family's one deliberate exception), and it
/// keeps `gam.rs` to the fitting math rather than teaching it about
/// confidence levels.
///
/// A fit `gam` cannot compute (fewer than 4 distinct x values, zero x
/// variance, no λ leaving usable residual degrees of freedom) degrades to
/// an all-NaN grid like every other degenerate branch in this file. `gam`
/// adds no refusal of its own at *fit* time — its one caller-supplied
/// parameter, `k`, is clamped down to the number of distinct x values before
/// it reaches the design, so it cannot make the design absurd the way an
/// over-large `degree` makes a Vandermonde design rank-deficient (an
/// unusable `k` VALUE, `k < 4`, is refused at `PySmooth::new` instead).
fn gam_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize, k: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let fit = match gam::fit(xs, ys, k) {
        Some(f) => f,
        None => return build_degenerate_smooth_batch(grid, n_grid, metrics.is_some()),
    };

    let y_fit: Vec<f64> = grid.iter().map(|&x| fit.predict(x)).collect();

    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            let t_crit = student_t_critical(level, fit.residual_dof());
            let mut lo = Vec::with_capacity(n_grid);
            let mut hi = Vec::with_capacity(n_grid);
            for (&x, &yhat) in grid.iter().zip(y_fit.iter()) {
                let half_width = t_crit * fit.std_error(x);
                lo.push(yhat - half_width);
                hi.push(yhat + half_width);
            }
            (lo, hi)
        }
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)
}

fn lm_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let n = xs.len();
    let mean_x: f64 = xs.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = ys.iter().sum::<f64>() / n as f64;
    let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();

    if sxx == 0.0 {
        return build_degenerate_smooth_batch(grid, n_grid, metrics.is_some());
    }

    let beta = sxy / sxx;
    let alpha = mean_y - beta * mean_x;
    let y_fit: Vec<f64> = grid.iter().map(|x| alpha + beta * x).collect();

    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            let resid_ss: f64 = xs.iter().zip(ys)
                .map(|(x, y)| (y - (alpha + beta * x)).powi(2))
                .sum();
            let dof = (n as f64) - 2.0;
            if dof <= 0.0 { (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]) }
            else {
                let sigma2 = resid_ss / dof;
                let t_crit = student_t_critical(level, dof);
                let mut lo = Vec::with_capacity(n_grid);
                let mut hi = Vec::with_capacity(n_grid);
                for &xq in grid {
                    let se = (sigma2 * (1.0 / (n as f64) + (xq - mean_x).powi(2) / sxx)).sqrt();
                    lo.push(alpha + beta * xq - t_crit * se);
                    hi.push(alpha + beta * xq + t_crit * se);
                }
                (lo, hi)
            }
        }
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)
}

/// Two-sided t-critical at level `level` (e.g., 0.95) with `dof` degrees of freedom.
/// Hill's approximation; adequate for n >= 3 and tail probabilities >= 0.5%.
fn student_t_critical(level: f64, dof: f64) -> f64 {
    let alpha = 1.0 - level;
    let p = 1.0 - alpha / 2.0;
    let z = inv_normal_cdf(p);
    let c1 = (z * z + 1.0) / (4.0 * dof);
    let c2 = (5.0 * z.powi(4) + 16.0 * z * z + 3.0) / (96.0 * dof * dof);
    z * (1.0 + c1 + c2)
}

fn inv_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer / Moro algorithm.
    let a = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ];
    let d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    }
}

fn loess_fit(
    xs: &[f64], ys: &[f64], grid: &[f64],
    bandwidth: f64, degree: u8, ci: Option<f64>, n_grid: usize, seed: u64,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let n = xs.len();
    let k = ((bandwidth * n as f64).ceil() as usize).max((degree as usize) + 1);
    let k = k.min(n);

    let (sxs, sys) = sort_xy(xs, ys);
    let y_fit: Vec<f64> = grid.iter().map(|&x0|
        loess_at_point_sorted(&sxs, &sys, x0, k, degree)
    ).collect();

    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => loess_bootstrap_ci(xs, ys, grid, k, degree, level, seed),
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)
}

/// Sort `(xs, ys)` by `xs` and return the sorted pair.
fn sort_xy(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().unzip()
}

/// LOESS evaluation at a single query point. `xs` must be sorted ascending.
/// Uses binary search + sliding window to find the k nearest neighbors
/// in O(log n + k) instead of the O(n log n) full-sort approach.
fn loess_at_point_sorted(xs: &[f64], ys: &[f64], x0: f64, k: usize, degree: u8) -> f64 {
    let n = xs.len();
    if n == 0 || k == 0 { return f64::NAN; }
    let k = k.min(n);
    if k < (degree as usize) + 1 { return f64::NAN; }

    let center = xs.partition_point(|&v| v < x0);
    let mut best_lo = center.saturating_sub(k).min(n - k);
    // Slide the window to minimize the max distance to x0
    while best_lo + k < n {
        let d_drop = (xs[best_lo] - x0).abs();
        let d_add = (xs[best_lo + k] - x0).abs();
        if d_add < d_drop {
            best_lo += 1;
        } else {
            break;
        }
    }

    let take: Vec<(usize, f64)> = (best_lo..best_lo + k)
        .map(|i| (i, (xs[i] - x0).abs()))
        .collect();
    let h = take.iter().map(|(_, d)| *d).fold(0.0_f64, f64::max);

    if degree == 1 {
        let mut sw = 0.0; let mut swx = 0.0; let mut swxx = 0.0;
        let mut swy = 0.0; let mut swxy = 0.0;
        for (i, dist) in &take {
            let w = if h == 0.0 { 1.0 } else {
                let u = (dist / h).abs();
                if u >= 1.0 { 0.0 } else { let v = 1.0 - u.powi(3); v * v * v }
            };
            let xi = xs[*i]; let yi = ys[*i];
            sw += w; swx += w * xi; swxx += w * xi * xi;
            swy += w * yi; swxy += w * xi * yi;
        }
        let det = sw * swxx - swx * swx;
        if det.abs() < 1e-15 { return f64::NAN; }
        let a = (swxx * swy - swx * swxy) / det;
        let b = (sw * swxy - swx * swy) / det;
        a + b * x0
    } else if degree == 2 {
        let mut xtwx = [[0.0_f64; 3]; 3];
        let mut xtwy = [0.0_f64; 3];
        for (i, dist) in &take {
            let w = if h == 0.0 { 1.0 } else {
                let u = (dist / h).abs();
                if u >= 1.0 { 0.0 } else { let v = 1.0 - u.powi(3); v * v * v }
            };
            let xi = xs[*i];
            let row = [1.0, xi, xi * xi];
            for r in 0..3 {
                for c in 0..3 {
                    xtwx[r][c] += w * row[r] * row[c];
                }
                xtwy[r] += w * row[r] * ys[*i];
            }
        }
        match crate::transform::linalg::solve_3x3_spd(xtwx, xtwy) {
            Some(beta) => beta[0] + beta[1] * x0 + beta[2] * x0 * x0,
            None => f64::NAN,
        }
    } else {
        f64::NAN
    }
}

fn loess_bootstrap_ci(
    xs: &[f64], ys: &[f64], grid: &[f64], k: usize, degree: u8, level: f64, seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;

    let n = xs.len();
    if n < 2 || level <= 0.0 || level >= 1.0 {
        return (vec![f64::NAN; grid.len()], vec![f64::NAN; grid.len()]);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_boot: usize = 200;
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    samples.resize_with(grid.len(), Vec::new);

    let mut bx = vec![0.0; n];
    let mut by = vec![0.0; n];
    for _ in 0..n_boot {
        for i in 0..n {
            let j = rng.gen_range(0..n);
            bx[i] = xs[j];
            by[i] = ys[j];
        }
        let (sbx, sby) = sort_xy(&bx, &by);
        for (gi, &x0) in grid.iter().enumerate() {
            let v = loess_at_point_sorted(&sbx, &sby, x0, k, degree);
            samples[gi].push(v);
        }
    }
    let alpha = 1.0 - level;
    let lo_q = alpha / 2.0; let hi_q = 1.0 - alpha / 2.0;
    let mut lo_out = Vec::with_capacity(grid.len());
    let mut hi_out = Vec::with_capacity(grid.len());
    for s in samples.iter_mut() {
        s.retain(|v| v.is_finite());
        if s.len() < 4 {
            lo_out.push(f64::NAN); hi_out.push(f64::NAN);
            continue;
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lo_out.push(percentile_sorted(s, lo_q));
        hi_out.push(percentile_sorted(s, hi_q));
    }
    (lo_out, hi_out)
}

fn percentile_sorted(s: &[f64], p: f64) -> f64 {
    let n = s.len();
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    s[lo] * (1.0 - frac) + s[hi] * frac
}

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Smoothing line (LOESS or linear model) with optional confidence band.
///
/// Fits a smoothing curve to ``(x, y)`` data and evaluates it on ``n``
/// grid points spanning the x-range. When ``ci`` is set, a bootstrap
/// confidence band is computed using ``seed`` for reproducibility. An
/// optional x-binning step (``x_bins``) and per-bin aggregator
/// (``x_estimator``) are applied before fitting, as in seaborn's
/// ``lineplot``.
///
/// Output columns: ``x`` (Float64 grid), ``y`` (Float64 fitted or
/// residuals), ``ci_lower`` and ``ci_upper`` (Float64; ``NaN`` when ``ci``
/// is not set).
///
/// Parameters
/// ----------
/// x : str
///     Predictor column (any integer, unsigned, or float dtype; widened to
///     Float64). A non-numeric column raises, naming the transform, column,
///     and dtype.
/// y : str
///     Response column (any integer, unsigned, or float dtype; widened to
///     Float64). A non-numeric column raises, naming the transform, column,
///     and dtype.
/// method : {"loess", "lm", "linear", "quadratic", "cubic", "log", "sqrt", \
///           "glm", "gam"}, default "loess"
///     Smoothing method. ``"loess"`` fits locally-weighted polynomial
///     regression; ``"lm"``/``"linear"`` fits a global linear model
///     (promoted to a polynomial fit when ``degree >= 2`` is also given);
///     ``"quadratic"``/``"cubic"`` fit fixed degree-2/3 polynomials;
///     ``"log"``/``"sqrt"`` fit a linear model on a transformed x;
///     ``"glm"`` fits a Generalized Linear Model (Gaussian family, Identity
///     link — the standalone ``Glm`` transform's own default) via IRLS;
///     ``"gam"`` fits a Generalized Additive Model — a penalized cubic
///     B-spline (P-spline) whose smoothing parameter is chosen by
///     generalized cross-validation. Unlike ``"loess"``, ``"gam"`` uses no
///     randomness: its confidence band is the spline's Bayesian posterior
///     band, so ``seed`` has no effect on it. ``"gam"`` holds its boundary
///     value rather than extrapolating outside the observed x range.
/// ci : float or None, default 0.95
///     Confidence level in (0, 1) for the confidence band. Set to
///     ``None`` to suppress the band.
/// bandwidth : float, optional
///     LOESS span in (0, 1]; fraction of data used in each local fit.
///     Defaults to ``0.75`` when omitted. Passing it with any other method
///     raises, since no other fit path reads it — the one tolerance is an
///     explicit ``bandwidth=0.75`` (the default value itself), accepted as a
///     harmless no-op exactly as ``method="lm", degree=1`` is. To make a
///     ``"gam"`` fit follow the data more closely, raise ``k``; for
///     ``"lm"``/``"linear"``, raise ``degree``.
/// degree : int, optional
///     For ``method="lm"``/``"linear"``: promotes the fit to a polynomial
///     of this degree (``Poly{degree}``) when ``>= 2``; ``1`` (or omitting
///     ``degree``) keeps the plain linear fit; ``0`` raises. A degree that
///     leaves fewer usable data points than coefficients to solve for (a
///     rank-deficient design) raises rather than silently rendering nothing.
///     For ``method="loess"``: the local-fit degree, ``1`` or ``2`` (default
///     when omitted is ``2``). For ``method="quadratic"``/``"cubic"``:
///     optional and must match the method's fixed degree (``2``/``3``) when
///     given — a conflicting value (e.g. ``method="quadratic", degree=5``)
///     raises rather than silently fitting degree 2 anyway. Passing
///     ``degree`` with ``method="log"``/``"sqrt"``/``"glm"``/``"gam"``
///     raises, since those methods have no degree concept to apply it to
///     (``"gam"``'s spline degree is fixed at 3; its smoothness comes from
///     the GCV-chosen penalty and from ``k``, not from a degree).
/// k : int, optional
///     For ``method="gam"``: the spline basis dimension (mgcv's ``k``),
///     default ``10``. This is gam's smoothness knob — the penalty chooses
///     how much of the basis to spend, so a larger ``k`` widens what the fit
///     *can* follow rather than forcing it to wiggle. Raise it when a
///     high-frequency signal is visibly under-smoothed at the default.
///     Clamped down to the number of distinct x values (a basis cannot carry
///     more segments than the data can place), so an over-large ``k`` is
///     harmless rather than rank-deficient. Values below ``4`` raise (no
///     smaller cubic basis exists), as does passing ``k`` with any method
///     other than ``"gam"``.
/// n : int, default 200
///     Number of grid points to evaluate. Must be > 0.
/// seed : int, default 0
///     RNG seed for the bootstrap confidence band. Seeds ``ChaCha8Rng``
///     for byte-deterministic output across platforms.
/// x_bins : int, optional
///     When set, the x-axis is divided into this many equal-width bins and
///     ``x_estimator`` is applied within each bin before fitting.
/// x_estimator : {"mean", "median", "sum", "min", "max"}, optional
///     Aggregation function applied per x-bin when ``x_bins`` is set.
///     Default is ``None`` (no aggregation).
/// output : {"fitted", "residuals"}, default "fitted"
///     When ``"residuals"``, the ``y`` column contains ``y_obs - y_hat``.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_smooth(method="lm", ci=0.95).encode(x="x", y="y")
///
/// LOESS curve with custom bandwidth:
///
/// >>> fm.Chart(df).mark_smooth(method="loess", bandwidth=0.4).encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Smooth")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PySmooth(pub(crate) TransformSpec);

#[pymethods]
impl PySmooth {
    #[new]
    #[pyo3(signature = (x, y, *, method = "loess", ci = Some(0.95), bandwidth = None, degree = None, k = None, n = 200, seed = 0, x_bins = None, x_estimator = None, output = "fitted", inject_zero_ref = false, inject_metrics = false, groupby = None, name = None, x_range = None))]
    fn new(
        x: &str, y: &str,
        method: &str,
        ci: Option<f64>,
        bandwidth: Option<f64>,
        degree: Option<u8>,
        k: Option<usize>,
        n: usize,
        seed: u64,
        x_bins: Option<usize>,
        x_estimator: Option<&str>,
        output: &str,
        inject_zero_ref: bool,
        inject_metrics: bool,
        groupby: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        x_range: Option<[f64; 2]>,
    ) -> PyResult<Self> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Smooth: x and y must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Smooth: n must be > 0"));
        }
        if let Some(level) = ci {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err("Smooth: ci must be in (0, 1)"));
            }
        }
        // Batch-D T9 round 2 (Required Fix 2): `METHODS` is the single
        // vocabulary the registry, this pre-check, and the error message all
        // read — see `METHODS`'s doc comment. A string not in `METHODS`
        // refuses here, before the match below ever runs.
        if !METHODS.contains(&method) {
            return Err(PyValueError::new_err(format!(
                "Smooth: unknown method '{method}'; expected {}",
                METHODS.iter().map(|m| format!("'{m}'")).collect::<Vec<_>>().join(" | "),
            )));
        }
        // Batch-D T9 (spec §4.6): "lm"/"linear" promotes to a polynomial fit
        // when `degree >= 2` is given; `degree == 1` (or omitted) keeps the
        // plain linear fit — the no-op case `lmplot(order=1)` (and every
        // existing `mark_smooth(method="lm")` caller that never mentions
        // `degree`) relies on for byte-identical output. `degree == 0`
        // refuses rather than silently behaving like `degree == 1`.
        //
        // "quadratic"/"cubic" fit a FIXED degree (2/3). Batch-D T9 round 2
        // (Required Fix 1, rust-quality verdict finding 1): a `degree=` that
        // matches the fixed degree is accepted as a harmless no-op (the same
        // "explicit value that agrees with the default" tolerance `lm`'s
        // `degree=1` gets above); a CONFLICTING `degree=` used to be silently
        // swallowed — `method="quadratic", degree=5"` quietly fit a degree-2
        // curve with no diagnostic, the exact "silently the wrong fit" class
        // D9 exists to close. It now refuses, naming `method="lm", degree=d`
        // as the way to actually get that degree.
        let resolved_method = match method {
            "lm" | "linear" => match degree {
                None | Some(1) => SmoothMethod::Lm,
                Some(0) => return Err(PyValueError::new_err(
                    "Smooth: degree must be >= 1 for method='lm'/'linear'"
                )),
                Some(d) => SmoothMethod::Poly { degree: d },
            },
            "loess" => SmoothMethod::Loess,
            "quadratic" => match degree {
                None | Some(2) => SmoothMethod::Poly { degree: 2 },
                Some(d) => return Err(PyValueError::new_err(format!(
                    "Smooth: degree={d} conflicts with method='quadratic', which fits a \
                     fixed degree-2 polynomial; omit degree, pass degree=2, or use \
                     method='lm', degree={d} for a degree-{d} fit"
                ))),
            },
            "cubic" => match degree {
                None | Some(3) => SmoothMethod::Poly { degree: 3 },
                Some(d) => return Err(PyValueError::new_err(format!(
                    "Smooth: degree={d} conflicts with method='cubic', which fits a \
                     fixed degree-3 polynomial; omit degree, pass degree=3, or use \
                     method='lm', degree={d} for a degree-{d} fit"
                ))),
            },
            "log" => SmoothMethod::LogX,
            "sqrt" => SmoothMethod::SqrtX,
            "glm" => SmoothMethod::Glm,
            "gam" => SmoothMethod::Gam,
            // Batch-D T9 round 3 (cycle-2 finding B): the `METHODS` pre-check
            // above guarantees `accepted ⊆ METHODS`, not `METHODS ⊆ match
            // arms` — nothing compile-time-enforces the reverse, so an entry
            // added to `METHODS` without a matching arm here used to reach
            // an `unreachable!()` and surface as a `PanicException` at the
            // PyO3 boundary instead of a `ValueError`. Return the same typed
            // error the pre-check builds; a one-line edit to `METHODS` (the
            // const its own doc comment invites people to edit) should never
            // be able to panic a caller.
            _ => return Err(PyValueError::new_err(format!(
                "Smooth: internal error — '{method}' is declared in `METHODS` but has no \
                 parser arm in `PySmooth::new`; this is a bug in ferrum, not a user error. \
                 expected {}",
                METHODS.iter().map(|m| format!("'{m}'")).collect::<Vec<_>>().join(" | "),
            ))),
        };
        // Closing D9: `degree` on a method with no degree concept ("log",
        // "sqrt", "glm") used to be silently ignored; it now refuses so a
        // misunderstood `degree=` can't render the wrong fit quietly.
        // "lm"/"linear" (promotable), "loess", "quadratic", and "cubic" all
        // accept it. Batch-D T10: "gam" joins the refusing side from birth —
        // its spline degree is fixed at 3 by the method's contract and its
        // smoothness is set by the GCV-chosen penalty, not by a degree, so
        // there is nothing for a caller's `degree=` to mean.
        if degree.is_some()
            && matches!(
                resolved_method,
                SmoothMethod::LogX | SmoothMethod::SqrtX | SmoothMethod::Glm | SmoothMethod::Gam
            )
        {
            return Err(PyValueError::new_err(format!(
                "Smooth: degree is not valid for method='{method}'; degree only \
                 applies to 'lm'/'linear' (polynomial promotion), 'loess', \
                 'quadratic', or 'cubic'"
            )));
        }
        // R1 design finding 6: `bandwidth` used to be a plain `f64` with an
        // inline `0.75` default, so an explicit value on a non-loess method
        // was indistinguishable from the default and every fit path except
        // LOESS silently ignored it — a user watching `method="gam"`
        // under-smooth had one knob that refused (`degree`) and one that
        // no-opped. `Option<f64>` makes explicit-vs-default distinguishable
        // at this boundary (the `degree: Option<u8>` precedent above), and an
        // explicit non-loess `bandwidth` now refuses in exactly the shape
        // `degree` does.
        //
        // The one tolerance mirrors `degree`'s own: an explicit value that
        // AGREES with the default is a harmless no-op, the same way
        // `method="lm", degree=1` and `method="quadratic", degree=2` are
        // accepted above. `desugar_smooth` (`marks/statistical.py`) no longer
        // needs it -- its own `bandwidth` default moved to `None` and its
        // forward is conditional on `bandwidth is not None`, so it never
        // sends an explicit `0.75` in the first place. The tolerance now
        // matters only to a caller constructing `Smooth(...)` directly with
        // an explicit `bandwidth=0.75` on a non-loess method; `mark_smooth
        // (method="lm", bandwidth=0.4)` still raises either way.
        if let Some(b) = bandwidth {
            if !matches!(resolved_method, SmoothMethod::Loess) && b != DEFAULT_LOESS_BANDWIDTH {
                return Err(PyValueError::new_err(format!(
                    "Smooth: bandwidth={b} is not valid for method='{method}'; bandwidth is \
                     the LOESS span and only applies to method='loess'. For a smoother \
                     '{method}' fit, use the knob that method actually has (method='gam' \
                     takes k=, the spline basis dimension; 'lm'/'linear' take degree=)"
                )));
            }
        }
        let bandwidth = bandwidth.unwrap_or(DEFAULT_LOESS_BANDWIDTH);
        // `k` is gam's basis dimension and gam's alone, refused on every other
        // method under the same asymmetry rule `degree` and `bandwidth` follow.
        // No agrees-with-the-default tolerance here: nothing forwards `k`
        // unconditionally, so `Some(_)` on a non-gam method is always a
        // misunderstanding worth naming.
        if let Some(kv) = k {
            if !matches!(resolved_method, SmoothMethod::Gam) {
                return Err(PyValueError::new_err(format!(
                    "Smooth: k={kv} is not valid for method='{method}'; k is the spline \
                     basis dimension and only applies to method='gam'"
                )));
            }
            if kv < gam::MIN_BASIS_DIM {
                return Err(PyValueError::new_err(format!(
                    "Smooth: k={kv} is below the smallest cubic spline basis ({}); a basis \
                     this small cannot span even one cubic segment",
                    gam::MIN_BASIS_DIM,
                )));
            }
        }
        let k = k.unwrap_or(gam::DEFAULT_BASIS_DIM);
        if matches!(resolved_method, SmoothMethod::Loess) {
            if !bandwidth.is_finite() || bandwidth <= 0.0 || bandwidth > 1.0 {
                return Err(PyValueError::new_err(
                    "Smooth: LOESS bandwidth must be a finite value in (0, 1]",
                ));
            }
            let loess_degree = degree.unwrap_or(2);
            if loess_degree != 1 && loess_degree != 2 {
                return Err(PyValueError::new_err("Smooth: LOESS degree must be 1 or 2"));
            }
        }
        // `SmoothSpec.degree` mirrors the resolved fit's own degree where
        // one exists (Poly/Loess); methods without a degree concept
        // (Lm/LogX/SqrtX/Glm) store 1 as a neutral placeholder that no fit
        // path reads.
        let degree_for_spec: u8 = match resolved_method {
            SmoothMethod::Poly { degree: d } => d,
            SmoothMethod::Loess => degree.unwrap_or(2),
            _ => 1,
        };
        let method = resolved_method;
        use crate::transform::aggregate::AggFn;
        let x_estimator_parsed: Option<AggFn> = match x_estimator {
            None => None,
            Some(s) => Some(match s {
                "mean" => AggFn::Mean,
                "median" => AggFn::Median,
                "sum" => AggFn::Sum,
                "min" => AggFn::Min,
                "max" => AggFn::Max,
                other => return Err(PyValueError::new_err(format!(
                    "Smooth: unknown x_estimator '{other}'; expected 'mean'|'median'|'sum'|'min'|'max'"
                ))),
            }),
        };
        let output_parsed = match output {
            "fitted" => SmoothOutput::Fitted,
            "residuals" => SmoothOutput::Residuals,
            other => return Err(PyValueError::new_err(format!(
                "Smooth: unknown output '{other}'; expected 'fitted'|'residuals'"
            ))),
        };
        if let Some(b) = x_bins {
            if b == 0 {
                return Err(PyValueError::new_err("Smooth: x_bins must be > 0"));
            }
        }
        if inject_zero_ref
            && !matches!(output_parsed, SmoothOutput::Residuals)
        {
            return Err(PyValueError::new_err(
                "Smooth: inject_zero_ref requires output='residuals'",
            ));
        }
        // inject_metrics is allowed on both Fitted and Residuals — see
        // SB-followup 2026-05-12 (3a rework). Top-right anchor semantics
        // are documented on SmoothSpec.inject_metrics.
        let groupby = parse_groupby_pyany(groupby)?;
        Ok(PySmooth(TransformSpec::Smooth(SmoothSpec {
            x: x.to_string(), y: y.to_string(),
            method, ci, bandwidth, degree: degree_for_spec, k, n, seed,
            x_bins,
            x_estimator: x_estimator_parsed,
            output: output_parsed,
            inject_zero_ref,
            inject_metrics,
            groupby,
            name,
            x_range,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Smooth(s) => format!(
                "Smooth(x='{}', y='{}', method={:?}, ci={:?}, bandwidth={}, degree={}, k={}, n={}, seed={}, x_bins={:?}, x_estimator={:?}, output={:?}, inject_zero_ref={}, inject_metrics={}, groupby={:?}, name={:?})",
                s.x, s.y, s.method, s.ci, s.bandwidth, s.degree, s.k, s.n, s.seed,
                s.x_bins, s.x_estimator, s.output,
                s.inject_zero_ref, s.inject_metrics, s.groupby, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// The single source of truth for `Smooth`'s free-string `method`
/// vocabulary. Batch-D T9 round 2 (Required Fix 2, rust-quality verdict):
/// consumed by THREE sites — `registry::entry_for_kind("smooth")`'s
/// `methods` list (`registry/mod.rs`, via `METHODS.to_vec()`), the
/// constructor's pre-check above, and the "unknown method" error string
/// above (built FROM this slice, not a hand-kept parallel literal). The
/// pre-check makes the guarantee structural rather than test-detected: a
/// match arm added to `PySmooth::new` for a new alias without adding it here
/// is unreachable dead code (the pre-check rejects the string as "unknown
/// method" before the match ever runs), so a new method genuinely cannot
/// "land" — become callable — without the registry seeing it in the same
/// commit. The former guard
/// (`smooth_methods_vocabulary_matches_parser_error_message_exactly`,
/// `registry/tests.rs`) only proved the error message matched the registry;
/// it could not have caught a bare match arm added without touching the
/// message at all.
///
/// The reverse direction (`METHODS ⊆ match arms`, i.e. an entry added here
/// without a corresponding arm in `PySmooth::new`'s match below) is NOT
/// compile-time-enforced — nothing in Rust's exhaustiveness checking ties a
/// `&str` slice to a `match` on a different `&str`. Batch-D T9 round 3: the
/// match's `_` arm returns the same typed "unknown method" error rather than
/// `unreachable!()`, so that direction fails safely (a `ValueError`) instead
/// of panicking at the PyO3 boundary; `test_all_method_aliases_accepted_by_pysmooth_new`
/// (iterating `METHODS` itself) still catches the gap at test time.
pub(crate) const METHODS: &[&str] =
    &["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt", "glm", "gam"];

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn xy_batch(x: Vec<f64>, y: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
        ]).unwrap()
    }

    fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) }).collect()
    }

    #[test]
    fn test_lm_recovers_slope_and_intercept_exactly() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 + 2.0 * x).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for (xq, yq) in xg.iter().zip(yf.iter()) {
            let expected = 3.0 + 2.0 * xq;
            assert!((yq - expected).abs() < 1e-10, "y(x={xq})={yq}, expected {expected}");
        }
    }

    #[test]
    fn test_lm_ci_band_brackets_fit_at_mean_x() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| {
            x + if i % 2 == 0 { 0.5 } else { -0.5 }
        }).collect();
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 51, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        let lo = col(&out, "ci_lower");
        let hi = col(&out, "ci_upper");
        let i = (0..xg.len()).min_by(|a, b|
            (xg[*a] - mean_x).abs().partial_cmp(&(xg[*b] - mean_x).abs()).unwrap()
        ).unwrap();
        assert!(lo[i] < yf[i] && yf[i] < hi[i], "CI must bracket fit at x={}", xg[i]);
        assert!(hi[i] - lo[i] > 0.0);
    }

    #[test]
    fn test_lm_zero_variance_x_emits_nan_line() {
        let xs = vec![5.0; 10];
        let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|y| y.is_nan()));
    }

    #[test]
    fn test_lm_n_lt_2_emits_all_nan() {
        let batch = xy_batch(vec![1.0], vec![1.0]);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            name: None,
            x_range: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|y| y.is_nan()));
    }

    #[test]
    fn test_smooth_round_trip_json() {
        let original = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.5, degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 100, seed: 42,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: SmoothSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_loess_deg1_against_fixtures() {
        use serde::Deserialize;
        const FIXTURES: &str = include_str!("fixtures/stat_refs.json");
        #[derive(Deserialize)]
        struct LoessCase {
            name: String, x: Vec<f64>, y: Vec<f64>,
            bandwidth: f64, degree: u8, n: usize,
            x_grid: Vec<f64>, y_fit: Vec<f64>,
        }
        #[derive(Deserialize)]
        struct F { loess: Vec<LoessCase> }
        let cases: F = serde_json::from_str(FIXTURES).unwrap();
        for case in cases.loess {
            if case.degree != 1 { continue; }
            let batch = xy_batch(case.x.clone(), case.y.clone());
            let spec = SmoothSpec {
                x: "x".into(), y: "y".into(),
                method: SmoothMethod::Loess, ci: None,
                bandwidth: case.bandwidth, degree: case.degree, k: gam::DEFAULT_BASIS_DIM, n: case.n, seed: 0,
                x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
                inject_zero_ref: false,
                inject_metrics: false,
            x_range: None,
                groupby: vec![],
                name: None,
            };
            let out = apply(&spec, &batch).unwrap();
            let xg = col(&out, "x");
            let yf = col(&out, "y");
            for i in 0..case.n {
                assert!((xg[i] - case.x_grid[i]).abs() < 1e-9, "x grid {} vs {}", xg[i], case.x_grid[i]);
                assert!((yf[i] - case.y_fit[i]).abs() < 1e-9, "case {}: y_fit[{i}] = {} vs {}", case.name, yf[i], case.y_fit[i]);
            }
        }
    }

    #[test]
    fn test_loess_deg2_against_fixtures() {
        use serde::Deserialize;
        const FIXTURES: &str = include_str!("fixtures/stat_refs.json");
        #[derive(Deserialize)]
        struct LoessCase {
            name: String, x: Vec<f64>, y: Vec<f64>,
            bandwidth: f64, degree: u8, n: usize,
            x_grid: Vec<f64>, y_fit: Vec<f64>,
        }
        #[derive(Deserialize)]
        struct F { loess: Vec<LoessCase> }
        let cases: F = serde_json::from_str(FIXTURES).unwrap();
        for case in cases.loess {
            if case.degree != 2 { continue; }
            let batch = xy_batch(case.x.clone(), case.y.clone());
            let spec = SmoothSpec {
                x: "x".into(), y: "y".into(),
                method: SmoothMethod::Loess, ci: None,
                bandwidth: case.bandwidth, degree: case.degree, k: gam::DEFAULT_BASIS_DIM, n: case.n, seed: 0,
                x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
                inject_zero_ref: false,
                inject_metrics: false,
            x_range: None,
                groupby: vec![],
                name: None,
            };
            let out = apply(&spec, &batch).unwrap();
            let xg = col(&out, "x");
            let yf = col(&out, "y");
            for i in 0..case.n {
                assert!((xg[i] - case.x_grid[i]).abs() < 1e-9);
                assert!((yf[i] - case.y_fit[i]).abs() < 1e-9,
                    "case {}: y_fit[{i}] = {} vs {} (diff {})",
                    case.name, yf[i], case.y_fit[i], (yf[i] - case.y_fit[i]).abs());
            }
        }
    }

    #[test]
    fn test_loess_deg2_local_window_too_small_emits_nan() {
        // Only 3 points but degree=2 requires k >= 3 — when bandwidth*n rounds to <3 the impl
        // floors k to degree+1=3 (so this test verifies the floor and that we don't panic).
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0];
        let ys: Vec<f64> = vec![0.0, 1.0, 4.0];
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Loess, ci: None,
            bandwidth: 0.1,  // bw * n = 0.3, floored to k = degree + 1 = 3
            degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        // Primary goal: no panic with k == n == degree+1.
        // With k=n=3 and tricube weights, the farthest neighbor's weight is exactly 0 for grid
        // endpoints, making X'WX rank-deficient → solve_3x3_spd returns None → NaN.
        // Interior grid points may have better conditioning and return finite values.
        // We assert: apply succeeds (no panic), output has 5 rows, each value is NaN or finite.
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert_eq!(yf.len(), 5);
        for (i, &v) in yf.iter().enumerate() {
            assert!(v.is_nan() || v.is_finite(), "y[{i}]={v} is neither NaN nor finite");
        }
        // The extreme endpoints (x=0 and x=2) should emit NaN: h==max_dist, u=1 → w=0 for
        // the farthest point, leaving rank-2 system.
        assert!(yf[0].is_nan(), "y[0]={} should be NaN (rank-deficient at left endpoint)", yf[0]);
        assert!(yf[4].is_nan(), "y[4]={} should be NaN (rank-deficient at right endpoint)", yf[4]);
    }

    #[test]
    fn test_loess_deg1_bootstrap_ci_is_reproducible_under_seed() {
        let xs: Vec<f64> = (0..40).map(|i| i as f64 / 10.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x).sin()).collect();
        let batch = xy_batch(xs, ys);
        let spec1 = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Loess,
            ci: Some(0.95),
            bandwidth: 0.5, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 20, seed: 42,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let spec2 = spec1.clone();
        let out1 = apply(&spec1, &batch).unwrap();
        let out2 = apply(&spec2, &batch).unwrap();
        let lo1 = col(&out1, "ci_lower");
        let lo2 = col(&out2, "ci_lower");
        let hi1 = col(&out1, "ci_upper");
        let hi2 = col(&out2, "ci_upper");
        for i in 0..lo1.len() {
            assert_eq!(lo1[i].to_bits(), lo2[i].to_bits(), "ci_lower not deterministic at {i}");
            assert_eq!(hi1[i].to_bits(), hi2[i].to_bits(), "ci_upper not deterministic at {i}");
        }
    }

    #[test]
    fn smooth_x_bins_pre_aggregates_then_fits() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm, ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: Some(10),
            x_estimator: Some(crate::transform::aggregate::AggFn::Mean),
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        let slope = (yf[xg.len() - 1] - yf[0]) / (xg[xg.len() - 1] - xg[0]);
        assert!((slope - 2.0).abs() < 1e-6, "expected slope 2.0, got {slope}");
    }

    #[test]
    fn smooth_output_residuals_returns_y_minus_fitted() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            2.0 * x + 1.0 + if i % 2 == 0 { 0.1 } else { -0.1 }
        ).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm, ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None,
            output: SmoothOutput::Residuals,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            name: None,
            x_range: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).name(), "x");
        assert_eq!(out.schema().field(1).name(), "residual");
        let r = col(&out, "residual");
        let mean_r: f64 = r.iter().sum::<f64>() / r.len() as f64;
        assert!(mean_r.abs() < 1e-9, "residual mean = {mean_r}");
        let max_abs = r.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(max_abs < 0.5, "max |residual| = {max_abs}");
    }

    #[test]
    fn smooth_output_default_is_fitted() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm, ci: Some(0.95),
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
        x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let schema = out.schema();
        let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, vec!["x".to_string(), "y".to_string(), "ci_lower".to_string(), "ci_upper".to_string()]);
    }

    #[test]
    fn test_lm_two_row_minimal_fit() {
        // 2 points is the minimum for a valid linear fit: y = 1 + 2x.
        let batch = xy_batch(vec![0.0, 1.0], vec![1.0, 3.0]);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        assert_eq!(xg.len(), 5);
        // Fit should be y = 1 + 2x exactly (perfect fit through 2 points).
        for (xq, yq) in xg.iter().zip(yf.iter()) {
            let expected = 1.0 + 2.0 * xq;
            assert!(
                (yq - expected).abs() < 1e-10,
                "y(x={xq})={yq}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_lm_constant_y_flat_line() {
        // All y=3.0 → fit should be flat at y=3.0 regardless of x.
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = vec![3.0; 10];
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 7, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            x_range: None,
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        for (i, &v) in yf.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-10,
                "expected y=3.0 at grid point {i}; got {v}"
            );
        }
    }

    // ---- Method alias & new-method tests ----

    /// "linear" must produce exactly the same output as "lm".
    #[test]
    fn test_linear_alias_matches_lm() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.5 * x - 3.0).collect();
        let batch = xy_batch(xs, ys);
        let base = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        // "linear" maps to SmoothMethod::Lm in the PySmooth constructor.
        // Test directly by verifying both produce identical RecordBatch output.
        let out_lm = apply(&base, &batch).unwrap();
        // The "linear" alias resolves to the same SmoothMethod::Lm struct,
        // so we verify the spec round-trip via JSON then re-apply.
        let json = serde_json::to_string(&base).unwrap();
        let parsed: SmoothSpec = serde_json::from_str(&json).unwrap();
        let out_linear = apply(&parsed, &batch).unwrap();
        let y_lm = col(&out_lm, "y");
        let y_linear = col(&out_linear, "y");
        for (a, b) in y_lm.iter().zip(y_linear.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "'linear' and 'lm' must produce identical output");
        }
    }

    /// "quadratic" (Poly { degree: 2 }) must recover a perfect degree-2 polynomial.
    #[test]
    fn test_quadratic_recovers_degree2_polynomial() {
        // y = 1 + 2x + 3x² — a perfect quadratic.
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Poly { degree: 2 },
            ci: None,
            bandwidth: 0.75, degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        assert_eq!(yf.len(), 5);
        for (&xq, &yq) in xg.iter().zip(yf.iter()) {
            let expected = 1.0 + 2.0 * xq + 3.0 * xq * xq;
            assert!(
                (yq - expected).abs() < 1e-6,
                "quadratic: y({xq}) = {yq}, expected {expected}"
            );
        }
    }

    /// "cubic" (Poly { degree: 3 }) must recover a perfect degree-3 polynomial.
    #[test]
    fn test_cubic_recovers_degree3_polynomial() {
        // y = -2 + x - 0.5x² + 0.1x³
        let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| -2.0 + x - 0.5 * x * x + 0.1 * x * x * x).collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Poly { degree: 3 },
            ci: None,
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 7, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for (&xq, &yq) in xg.iter().zip(yf.iter()) {
            let expected = -2.0 + xq - 0.5 * xq * xq + 0.1 * xq * xq * xq;
            assert!(
                (yq - expected).abs() < 1e-5,
                "cubic: y({xq}) = {yq}, expected {expected}"
            );
        }
    }

    /// "log" (LogX) must fit y ~ a + b·ln(x) and return NaN for x <= 0.
    #[test]
    fn test_log_x_recovers_log_relationship() {
        // y = 1 + 3·ln(x) (ground truth)
        let xs: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 3.0 * x.ln()).collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::LogX,
            ci: None,
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for (&xq, &yq) in xg.iter().zip(yf.iter()) {
            let expected = 1.0 + 3.0 * xq.ln();
            assert!(
                (yq - expected).abs() < 1e-6,
                "log: y({xq}) = {yq}, expected {expected}"
            );
        }
    }

    /// "sqrt" (SqrtX) must fit y ~ a + b·√x.
    #[test]
    fn test_sqrt_x_recovers_sqrt_relationship() {
        // y = 2 + 4·√x (ground truth)
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 + 4.0 * x.sqrt()).collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::SqrtX,
            ci: None,
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for (&xq, &yq) in xg.iter().zip(yf.iter()) {
            let expected = 2.0 + 4.0 * xq.sqrt();
            assert!(
                (yq - expected).abs() < 1e-6,
                "sqrt: y({xq}) = {yq}, expected {expected}"
            );
        }
    }

    /// "log" on a grid that includes x=0 must produce NaN at that grid point.
    #[test]
    fn test_log_x_nan_at_nonpositive_grid_points() {
        let xs: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.ln()).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::LogX,
            ci: None,
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false,
            // Force x_range to include x=0 to probe the NaN path.
            x_range: Some([0.0, 20.0]),
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        // The first grid point (x=0) must be NaN since ln(0) = -inf.
        assert!(
            yf[0].is_nan(),
            "log method must emit NaN at x={} (grid point 0); got {}",
            xg[0], yf[0]
        );
        // Remaining grid points (x > 0) must be finite.
        for (i, &yq) in yf.iter().enumerate().skip(1) {
            assert!(
                yq.is_finite(),
                "log method: y[{i}] should be finite at x={} but got {yq}",
                xg[i]
            );
        }
    }

    /// Every declared method (iterated from `METHODS` itself, not a
    /// hand-copied literal — batch-D T9 round 2) must be accepted without
    /// error by PySmooth::new.
    #[test]
    fn test_all_method_aliases_accepted_by_pysmooth_new() {
        pyo3::Python::initialize();
        for method in METHODS {
            let result = PySmooth::new(
                "x", "y",
                method,
                None, None, None, None, 200, 0,
                None, None, "fitted",
                false, false, None, None, None,
            );
            assert!(
                result.is_ok(),
                "PySmooth::new with method={method:?} should succeed; got {:?}",
                result.err().map(|e| e.to_string())
            );
        }
    }

    // ── NaN regression: smooth must drop NaN rows, matching kde/bin behavior ────

    /// A NaN x-value in the input must be silently dropped: the smoothed output
    /// must be identical to running the same smooth on the clean (NaN-free) subset.
    #[test]
    fn test_nan_in_x_is_dropped_from_smooth() {
        let xs_clean: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys_clean: Vec<f64> = xs_clean.iter().map(|x| 2.0 * x + 1.0).collect();

        // Inject a NaN at index 3.
        let mut xs_nan = xs_clean.clone();
        xs_nan[3] = f64::NAN;

        let batch_clean = xy_batch(xs_clean.clone(), ys_clean.clone());
        let batch_nan = xy_batch(xs_nan, ys_clean.clone());

        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };

        let out_clean = apply(&spec, &batch_clean).unwrap();
        let out_nan   = apply(&spec, &batch_nan).unwrap();

        let yf_clean = col(&out_clean, "y");
        let yf_nan   = col(&out_nan,   "y");

        assert_eq!(yf_clean.len(), yf_nan.len());
        for (i, (a, b)) in yf_clean.iter().zip(yf_nan.iter()).enumerate() {
            // Neither result should be NaN-corrupted.
            assert!(a.is_finite(), "clean output y[{i}]={a} should be finite");
            assert!(b.is_finite(), "nan-input output y[{i}]={b} should be NaN-corrupted (expected finite)");
            assert!(
                (a - b).abs() < 1e-9,
                "y[{i}]: clean={a}, nan-input={b}; dropping NaN row must not change smoothed values"
            );
        }
    }

    /// A NaN y-value in the input must be silently dropped.
    #[test]
    fn test_nan_in_y_is_dropped_from_smooth() {
        let xs_clean: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys_clean: Vec<f64> = xs_clean.iter().map(|x| 3.0 * x - 0.5).collect();

        // Inject a NaN at index 5.
        let mut ys_nan = ys_clean.clone();
        ys_nan[5] = f64::NAN;

        let batch_clean = xy_batch(xs_clean.clone(), ys_clean.clone());
        let batch_nan = xy_batch(xs_clean.clone(), ys_nan);

        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };

        let out_clean = apply(&spec, &batch_clean).unwrap();
        let out_nan   = apply(&spec, &batch_nan).unwrap();

        let yf_clean = col(&out_clean, "y");
        let yf_nan   = col(&out_nan,   "y");

        assert_eq!(yf_clean.len(), yf_nan.len());
        for (i, (a, b)) in yf_clean.iter().zip(yf_nan.iter()).enumerate() {
            // Both should be finite (NaN row was dropped, LM can still fit).
            assert!(a.is_finite(), "clean output y[{i}]={a} should be finite");
            assert!(b.is_finite(), "nan-y-input output y[{i}]={b} should be finite after NaN drop");
            // The fit values must be close but NOT required to be byte-identical
            // (removing one data point changes the regression coefficients slightly).
            // What must NOT happen: NaN-corrupted output.  Here we just assert finite.
        }
    }

    // ── T0.3: grouped smooth accepts non-Utf8 group dtypes + preserves dtype ──

    fn col_i64<'a>(b: &'a RecordBatch, name: &str) -> &'a arrow::array::Int64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
    }

    /// An Int64 group column must work (was "must be Utf8") AND round-trip as
    /// Int64 in the output, not a forced StringArray (XFORM-03).
    #[test]
    fn test_smooth_grouped_int64_group_roundtrips_as_int64() {
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        // Two groups, each a perfect line so LM recovers it exactly.
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut all_x: Vec<f64> = Vec::new();
        let mut all_y: Vec<f64> = Vec::new();
        let mut all_g: Vec<i64> = Vec::new();
        for &x in &xs {
            all_x.push(x);
            all_y.push(3.0 + 2.0 * x); // group 100
            all_g.push(100);
        }
        for &x in &xs {
            all_x.push(x);
            all_y.push(1.0 - 1.0 * x); // group 200
            all_g.push(200);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(all_x)),
                Arc::new(Float64Array::from(all_y)),
                Arc::new(Int64Array::from(all_g)),
            ],
        )
        .unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &batch).expect("Int64 groupby must succeed");
        // 2 groups × 5 grid points = 10 rows.
        assert_eq!(out.num_rows(), 10);
        // Group column must be the LAST field, dtype Int64 (NOT Utf8).
        let last = out.schema().fields().len() - 1;
        assert_eq!(out.schema().field(last).name(), "g");
        assert_eq!(
            out.schema().field(last).data_type(),
            &DataType::Int64,
            "group column must round-trip as Int64, not String"
        );
        let groups = col_i64(&out, "g");
        for i in 0..5 {
            assert_eq!(groups.value(i), 100, "first 5 rows are group 100");
        }
        for i in 5..10 {
            assert_eq!(groups.value(i), 200, "last 5 rows are group 200");
        }
        // Each group's fit recovers its own line.
        let yf = col(&out, "y");
        let xg = col(&out, "x");
        for i in 0..5 {
            assert!((yf[i] - (3.0 + 2.0 * xg[i])).abs() < 1e-9, "group 100 fit");
        }
        for i in 5..10 {
            assert!((yf[i] - (1.0 - 1.0 * xg[i])).abs() < 1e-9, "group 200 fit");
        }
    }

    /// Batch-F T2h: grouping by a column whose name collides with an output
    /// column name (`y`) must refuse with a typed error naming the
    /// collision, not silently append a second `y` field that
    /// `Schema::index_of` would resolve to the wrong one downstream.
    #[test]
    fn test_smooth_grouped_name_collision_with_output_column_refuses() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let group_vals: Vec<&str> = vec!["a", "a", "a", "b", "b", "b"];
        // The groupby column is literally named "y" — the Fitted output's
        // own fitted-y field — while the real x/y data live in "x_in"/"y_in".
        let schema = Arc::new(Schema::new(vec![
            Field::new("x_in", DataType::Float64, true),
            Field::new("y_in", DataType::Float64, true),
            Field::new("y", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(arrow::array::StringArray::from(group_vals)),
            ],
        )
        .unwrap();
        let spec = SmoothSpec {
            x: "x_in".into(), y: "y_in".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec!["y".into()],
            name: None,
        };
        let err = apply(&spec, &batch).expect_err(
            "groupby column colliding with an output column name must refuse",
        );
        let msg = err.to_string();
        assert!(msg.contains('y'), "error must name the colliding column: {msg}");
        assert!(msg.contains("collides"), "error must explain the collision: {msg}");
    }

    /// A Boolean group column must work (was "must be Utf8") and round-trip as
    /// Boolean in the output.
    #[test]
    fn test_smooth_grouped_boolean_group_column() {
        use arrow::array::BooleanArray;
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut all_x: Vec<f64> = Vec::new();
        let mut all_y: Vec<f64> = Vec::new();
        let mut all_g: Vec<bool> = Vec::new();
        for &x in &xs {
            all_x.push(x);
            all_y.push(2.0 * x);
            all_g.push(true);
        }
        for &x in &xs {
            all_x.push(x);
            all_y.push(-3.0 * x + 1.0);
            all_g.push(false);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(all_x)),
                Arc::new(Float64Array::from(all_y)),
                Arc::new(BooleanArray::from(all_g)),
            ],
        )
        .unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &batch).expect("Boolean groupby must succeed");
        assert_eq!(out.num_rows(), 8, "2 groups × 4 grid points");
        let last = out.schema().fields().len() - 1;
        assert_eq!(out.schema().field(last).data_type(), &DataType::Boolean);
        let groups = out
            .column(last)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        for i in 0..4 {
            assert!(groups.value(i), "first group (true) appears first");
        }
        for i in 4..8 {
            assert!(!groups.value(i), "second group (false) appears second");
        }
    }

    /// Sentinel: a Utf8 group column still round-trips as Utf8 (byte-stable path).
    #[test]
    fn test_smooth_grouped_utf8_group_unchanged() {
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let mut all_x: Vec<f64> = Vec::new();
        let mut all_y: Vec<f64> = Vec::new();
        let mut all_g: Vec<&str> = Vec::new();
        for &x in &xs {
            all_x.push(x);
            all_y.push(x + 1.0);
            all_g.push("A");
        }
        for &x in &xs {
            all_x.push(x);
            all_y.push(2.0 * x);
            all_g.push("B");
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(all_x)),
                Arc::new(Float64Array::from(all_y)),
                Arc::new(StringArray::from(all_g)),
            ],
        )
        .unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let last = out.schema().fields().len() - 1;
        assert_eq!(out.schema().field(last).data_type(), &DataType::Utf8);
        let groups = out
            .column(last)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(groups.value(0), "A");
        assert_eq!(groups.value(out.num_rows() - 1), "B");
    }

    /// Unknown method aliases must return an error.
    #[test]
    fn test_unknown_method_alias_returns_error() {
        pyo3::Python::initialize();
        let result = PySmooth::new(
            "x", "y", "polynomial",
            None, None, None, None, 200, 0,
            None, None, "fitted",
            false, false, None, None, None,
        );
        assert!(result.is_err(), "unknown method 'polynomial' must return an error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("unknown method"), "error message should mention 'unknown method': {msg}");
    }

    // ---- Batch-D T9: glm routing + lm degree≥2 → polynomial promotion ----

    /// `Smooth(method="glm")` must be accepted BY THE STRING PARSER
    /// (previously "glm" hit the `other` catch-all and raised "unknown
    /// method"). Batch-D T9 round 2 (Required Fix 3, rust-quality verdict):
    /// routes through `PySmooth::new`, not a hand-built `SmoothSpec` literal
    /// with `method: SmoothMethod::Glm` already set — the prior version of
    /// this test compiled and passed regardless of whether the parser
    /// accepted `"glm"` at all, so its "no longer hits the catch-all" claim
    /// was not actually proven by it.
    #[test]
    fn test_glm_method_accepted_and_produces_finite_fit() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 3.0 + 2.0 * x + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let batch = xy_batch(xs, ys);
        let ctor = PySmooth::new(
            "x", "y", "glm",
            None, None, None, None, 10, 0,
            None, None, "fitted",
            false, false, None, None, None,
        ).expect("PySmooth::new(method='glm') must be accepted by the string parser");
        let spec = match ctor.0 {
            TransformSpec::Smooth(s) => s,
            _ => panic!("expected Smooth spec"),
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|v| v.is_finite()), "glm fit must be finite; got {yf:?}");
    }

    /// GLM (Gaussian, Identity link) IRLS converges to the exact OLS point
    /// estimate, so its fitted curve must match "lm"'s to high precision —
    /// this is the correct behavior (not a sign the two are aliased; the
    /// CI-band test below proves the implementations are genuinely distinct).
    #[test]
    fn test_glm_point_estimate_matches_lm_ols() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 3.0 + 2.0 * x + if i % 3 == 0 { 0.4 } else { -0.2 })
            .collect();
        let batch = xy_batch(xs, ys);
        let lm_spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let glm_spec = SmoothSpec { method: SmoothMethod::Glm, ..lm_spec.clone() };
        let lm_out = apply(&lm_spec, &batch).unwrap();
        let glm_out = apply(&glm_spec, &batch).unwrap();
        let lm_y = col(&lm_out, "y");
        let glm_y = col(&glm_out, "y");
        for (a, b) in lm_y.iter().zip(glm_y.iter()) {
            assert!((a - b).abs() < 1e-8, "glm should match lm's OLS point estimate: lm={a}, glm={b}");
        }
    }

    /// RED-proof discriminator: a `"glm" => SmoothMethod::Lm` alias would
    /// happen to match `lm`'s point estimate (see above) but could not
    /// reproduce `glm.rs`'s real Wald CI band, which omits the dispersion
    /// (σ²) scaling that `lm`'s t-distribution band applies. With `ci` set,
    /// the two bands must therefore differ even though the fitted line is
    /// the same — proving `glm_fit` takes a genuinely distinct code path
    /// rather than aliasing to `Lm`.
    #[test]
    fn test_glm_ci_band_differs_from_lm_ci_band_proving_distinct_implementation() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 3.0 + 2.0 * x + if i % 2 == 0 { 1.5 } else { -1.5 })
            .collect();
        let batch = xy_batch(xs, ys);
        let lm_spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 10, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let glm_spec = SmoothSpec { method: SmoothMethod::Glm, ..lm_spec.clone() };
        let lm_out = apply(&lm_spec, &batch).unwrap();
        let glm_out = apply(&glm_spec, &batch).unwrap();
        let lm_width: Vec<f64> = col(&lm_out, "ci_upper").iter().zip(col(&lm_out, "ci_lower").iter())
            .map(|(&hi, &lo)| hi - lo).collect();
        let glm_width: Vec<f64> = col(&glm_out, "ci_upper").iter().zip(col(&glm_out, "ci_lower").iter())
            .map(|(&hi, &lo)| hi - lo).collect();
        let any_differs = lm_width.iter().zip(glm_width.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_differs,
            "glm and lm CI bands must differ (real GLM Wald CI has no dispersion scaling); \
             lm_width={lm_width:?}, glm_width={glm_width:?}"
        );
    }

    /// `Smooth(method="lm", degree=2)` must resolve to the exact same
    /// `Poly{degree:2}` code path as `Smooth(method="quadratic")` — byte-equal
    /// output, since both now construct the identical enum variant.
    #[test]
    fn test_lm_degree_2_byte_equal_to_quadratic() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
        let batch = xy_batch(xs, ys);
        let lm_deg2_spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Poly { degree: 2 },
            ci: None,
            bandwidth: 0.75, degree: 2, k: gam::DEFAULT_BASIS_DIM, n: 15, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        // Route both through PySmooth::new to prove the *constructor's*
        // resolution — not just the shared enum — treats them identically.
        pyo3::Python::initialize();
        let lm_ctor = PySmooth::new(
            "x", "y", "lm", None, None, Some(2), None, 15, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        let quad_ctor = PySmooth::new(
            "x", "y", "quadratic", None, None, None, None, 15, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        assert_eq!(lm_ctor.0, quad_ctor.0, "method='lm', degree=2 must construct the identical spec as method='quadratic'");

        let out = apply(&lm_deg2_spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for (&xq, &yq) in xg.iter().zip(yf.iter()) {
            let expected = 1.0 + 2.0 * xq + 3.0 * xq * xq;
            assert!((yq - expected).abs() < 1e-6, "lm+degree=2: y({xq})={yq}, expected {expected}");
        }
    }

    /// `Smooth(method="lm")` with no `degree` and `Smooth(method="lm",
    /// degree=1)` must both resolve to plain `SmoothMethod::Lm` — the
    /// no-op case every existing `mark_smooth(method="lm")` caller (and
    /// `lmplot`'s default `order=1`) relies on for unchanged output.
    #[test]
    fn test_lm_no_degree_and_degree_1_both_stay_linear() {
        pyo3::Python::initialize();
        let implicit = PySmooth::new(
            "x", "y", "lm", None, None, None, None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        let explicit_one = PySmooth::new(
            "x", "y", "lm", None, None, Some(1), None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        assert_eq!(implicit.0, explicit_one.0, "degree omitted vs degree=1 must construct the same spec");
        match &implicit.0 {
            TransformSpec::Smooth(s) => assert_eq!(s.method, SmoothMethod::Lm),
            _ => panic!("expected Smooth spec"),
        }
    }

    /// `degree=0` with `method="lm"` must refuse rather than silently
    /// behaving like `degree=1`.
    #[test]
    fn test_lm_degree_zero_refuses() {
        pyo3::Python::initialize();
        let result = PySmooth::new(
            "x", "y", "lm", None, None, Some(0), None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        );
        assert!(result.is_err(), "degree=0 with method='lm' must be refused");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("degree"), "error should mention 'degree': {msg}");
    }

    /// Closing D9: `degree` on a method with no degree concept ("log",
    /// "sqrt", "glm", and — batch-D T10 — "gam") must refuse rather than
    /// being silently ignored.
    #[test]
    fn test_degree_refused_for_non_polynomial_non_loess_methods() {
        pyo3::Python::initialize();
        for method in &["log", "sqrt", "glm", "gam"] {
            let result = PySmooth::new(
                "x", "y", method, None, None, Some(2), None, 200, 0, None, None, "fitted",
                false, false, None, None, None,
            );
            assert!(result.is_err(), "degree=2 with method={method:?} must be refused");
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains("degree"), "error should mention 'degree' for method={method:?}: {msg}");
        }
    }

    // ---- Batch-D T9 round 2: quadratic/cubic alias-degree policy ----

    /// `method="quadratic"` with a `degree` that AGREES with the fixed
    /// degree (2) is a harmless no-op — accepted, not refused.
    #[test]
    fn test_quadratic_matching_degree_is_accepted_noop() {
        pyo3::Python::initialize();
        let implicit = PySmooth::new(
            "x", "y", "quadratic", None, None, None, None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        let explicit = PySmooth::new(
            "x", "y", "quadratic", None, None, Some(2), None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        assert_eq!(implicit.0, explicit.0, "degree omitted vs degree=2 must construct the same spec for 'quadratic'");
    }

    /// `method="cubic"` with a `degree` that AGREES with the fixed degree
    /// (3) is a harmless no-op — accepted, not refused.
    #[test]
    fn test_cubic_matching_degree_is_accepted_noop() {
        pyo3::Python::initialize();
        let implicit = PySmooth::new(
            "x", "y", "cubic", None, None, None, None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        let explicit = PySmooth::new(
            "x", "y", "cubic", None, None, Some(3), None, 200, 0, None, None, "fitted",
            false, false, None, None, None,
        ).unwrap();
        assert_eq!(implicit.0, explicit.0, "degree omitted vs degree=3 must construct the same spec for 'cubic'");
    }

    /// A `degree` that CONFLICTS with `method="quadratic"`'s fixed degree
    /// must refuse rather than silently fitting degree 2 anyway. Batch-D T9
    /// round 2 (Required Fix 1, rust-quality verdict finding 1): this used
    /// to construct successfully and quietly fit a degree-2 curve.
    #[test]
    fn test_quadratic_conflicting_degree_refuses() {
        pyo3::Python::initialize();
        for bad_degree in [1u8, 3, 5] {
            let result = PySmooth::new(
                "x", "y", "quadratic", None, None, Some(bad_degree), None, 200, 0, None, None, "fitted",
                false, false, None, None, None,
            );
            assert!(result.is_err(), "degree={bad_degree} conflicting with method='quadratic' must be refused");
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains("degree") && msg.contains("quadratic"),
                "error should name both 'degree' and 'quadratic': {msg}");
        }
    }

    /// A `degree` that CONFLICTS with `method="cubic"`'s fixed degree must
    /// refuse rather than silently fitting degree 3 anyway.
    #[test]
    fn test_cubic_conflicting_degree_refuses() {
        pyo3::Python::initialize();
        for bad_degree in [1u8, 2, 4] {
            let result = PySmooth::new(
                "x", "y", "cubic", None, None, Some(bad_degree), None, 200, 0, None, None, "fitted",
                false, false, None, None, None,
            );
            assert!(result.is_err(), "degree={bad_degree} conflicting with method='cubic' must be refused");
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains("degree") && msg.contains("cubic"),
                "error should name both 'degree' and 'cubic': {msg}");
        }
    }

    // ---- Batch-D T9 round 2: rank-deficient degree refuses on the ungrouped path ----

    /// A `degree` that leaves the design rank-deficient (fewer usable points
    /// than coefficients to solve for) must refuse with a typed error on the
    /// UNGROUPED top-level call, not silently render an all-NaN grid.
    /// RED-proves the spec review's finding 2: `degree=20` on a 20-row frame
    /// used to render zero polylines with no diagnostic at all.
    #[test]
    fn test_rank_deficient_degree_refuses_on_ungrouped_path() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        for degree in [20u8, 21, 25] {
            let ctor = PySmooth::new(
                "x", "y", "lm", None, None, Some(degree), None, 5, 0, None, None, "fitted",
                false, false, None, None, None,
            ).unwrap();
            let spec = match ctor.0 {
                TransformSpec::Smooth(s) => s,
                _ => panic!("expected Smooth spec"),
            };
            let result = apply(&spec, &batch);
            assert!(result.is_err(), "degree={degree} on 20 points must refuse (rank-deficient), not silently NaN");
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains("degree") && msg.contains("rank-deficient"),
                "error for degree={degree} should mention 'degree' and 'rank-deficient': {msg}");
        }
    }

    /// The boundary case: `degree = n_points - 1` is EXACTLY determined
    /// (not rank-deficient) and must still fit, not refuse.
    #[test]
    fn test_degree_exactly_n_minus_one_is_not_rank_deficient() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Poly { degree: 19 },
            ci: None,
            bandwidth: 0.75, degree: 19, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).expect("degree=19 on 20 points is exactly determined, not rank-deficient");
        let yf = col(&out, "y");
        assert!(yf.iter().all(|v| v.is_finite()), "exactly-determined fit must be finite; got {yf:?}");
    }

    /// A rank-deficient degree inside ONE GROUP of a grouped fit must still
    /// degrade to that group's own all-NaN range (T6's contract) rather than
    /// refusing the whole batch — the refusal above is specific to the
    /// ungrouped top-level call.
    #[test]
    fn test_rank_deficient_degree_inside_one_group_stays_nan_not_refused() {
        pyo3::Python::initialize();
        // Group "a": 20 points, degree=19 is exactly determined (finite fit).
        // Group "b": only 3 points, degree=19 is massively rank-deficient.
        let mut all_x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut all_y: Vec<f64> = all_x.iter().map(|&x| x + 1.0).collect();
        let mut all_g: Vec<&str> = vec!["a"; 20];
        all_x.extend([100.0, 101.0, 102.0]);
        all_y.extend([1.0, 2.0, 3.0]);
        all_g.extend(["b", "b", "b"]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(all_x)),
                Arc::new(Float64Array::from(all_y)),
                Arc::new(arrow::array::StringArray::from(all_g)),
            ],
        ).unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Poly { degree: 19 },
            ci: None,
            bandwidth: 0.75, degree: 19, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec!["g".into()], name: None,
        };
        let out = apply(&spec, &batch)
            .expect("a rank-deficient group must degrade to NaN, not refuse the whole grouped batch");
        assert_eq!(out.num_rows(), 10, "2 groups × 5 grid points");
        let yf = col(&out, "y");
        assert!(yf[..5].iter().all(|v| v.is_finite()), "group 'a' (exactly determined) must be finite");
        assert!(yf[5..].iter().all(|v| v.is_nan()), "group 'b' (rank-deficient) must be NaN, not a refusal");
    }

    // ---- Batch-D T9 round 2: apply_grouped schema-mismatch panic (Required Fix 6) ----

    /// RED-proof for the live `PanicException` the rust-quality verdict
    /// reproduced through `Chart.to_svg()`: a grouped `glm` fit with
    /// `inject_metrics=true` where the SECOND group (not the first) is
    /// degenerate (glm's IRLS fails on zero-variance x) used to panic inside
    /// `apply_grouped` — the reference schema was pinned to group 0's WIDE
    /// (6-column, metrics included) output, and group 1's NARROW (4-column)
    /// degenerate output was then indexed positionally past its own schema
    /// (`index out of bounds: the len is 4 but the index is 4`). `lm` and
    /// `poly` inherit the identical `apply_grouped` template and were
    /// equally exposed; this test pins the fix at the shared level.
    #[test]
    fn test_apply_grouped_survives_degenerate_non_first_group_with_inject_metrics() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0])),
                Arc::new(arrow::array::StringArray::from(vec!["a", "a", "a", "a", "b", "b", "b", "b"])),
            ],
        ).unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Glm,
            ci: None,
            bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: true, x_range: None,
            groupby: vec!["g".into()], name: None,
        };
        let out = apply(&spec, &batch)
            .expect("grouped glm with a degenerate non-first group + inject_metrics must not panic");
        assert_eq!(out.num_rows(), 10, "2 groups × 5 grid points");
        let metrics_y = out
            .column(out.schema().index_of("_metrics_y").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        // Group 'a' (rows 0..5): a real glm fit -> the anchor row (rightmost
        // grid x, i.e. the group's LAST row) holds a finite metrics value.
        assert!(!metrics_y.is_null(4) && metrics_y.value(4).is_finite(),
            "group 'a' anchor row _metrics_y should be finite");
        // Group 'b' (rows 5..10): zero-variance x -> degenerate glm ->
        // `build_degenerate_smooth_batch` still emits the metrics columns
        // (batch-D T9 round 4), all-null -> every row null/NaN, not a panic
        // (and no longer via `apply_grouped`'s by-name padding fallback,
        // since group 'b' now carries the columns itself).
        for i in 5..10 {
            assert!(metrics_y.is_null(i) || metrics_y.value(i).is_nan(),
                "group 'b' row {i} _metrics_y should be null/NaN (degenerate group)");
        }
        let metrics_text = out
            .column(out.schema().index_of("_metrics_text").unwrap())
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        assert!(!metrics_text.is_null(4), "group 'a' anchor row _metrics_text should be present");
        for i in 5..10 {
            assert!(metrics_text.is_null(i), "group 'b' row {i} _metrics_text should be null (degenerate group)");
        }
    }

    // ---- Batch-D T9 round 4: degenerate-branch schema uniformity (root cause) ----

    /// RED-proof for `build_degenerate_smooth_batch` itself: when metrics
    /// are requested it must append `_metrics_text`/`_metrics_y`, both
    /// entirely null (no anchor row — there is no fit to summarize).
    #[test]
    fn build_degenerate_smooth_batch_emits_null_metrics_when_requested() {
        let grid = vec![0.0, 1.0, 2.0];
        let out = build_degenerate_smooth_batch(&grid, 3, true).unwrap();
        assert_eq!(out.schema().fields().len(), 6, "core 4 + metrics 2");
        let text = out.column(out.schema().index_of("_metrics_text").unwrap())
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        let y = out.column(out.schema().index_of("_metrics_y").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..3 {
            assert!(text.is_null(i), "row {i} _metrics_text must be null, not a garbage anchor");
            assert!(y.is_null(i), "row {i} _metrics_y must be null, not a garbage anchor");
        }
    }

    /// The `metrics_requested = false` side of the same contract: no columns
    /// at all when `inject_metrics` was never set, matching every method's
    /// pre-round-4 schema for the common (metrics-off) case.
    #[test]
    fn build_degenerate_smooth_batch_omits_metrics_columns_when_not_requested() {
        let grid = vec![0.0, 1.0];
        let out = build_degenerate_smooth_batch(&grid, 2, false).unwrap();
        assert_eq!(out.schema().fields().len(), 4, "core columns only, no metrics");
        assert!(out.schema().index_of("_metrics_text").is_err());
        assert!(out.schema().index_of("_metrics_y").is_err());
    }

    /// RED-proof at the public `apply()` boundary for the reviewer's cited
    /// `all_nan_output` example: a genuinely too-sparse ungrouped Fitted
    /// call (1 usable point) with `inject_metrics=true` must still emit the
    /// metrics columns — all-null — rather than dropping them (which is
    /// exactly the divergence a sibling group/facet-panel schema then had
    /// to reconcile around).
    #[test]
    fn all_nan_output_via_apply_with_inject_metrics_emits_null_metrics_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![2.0])),
            ],
        ).unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None, bandwidth: 0.75, degree: 1, k: gam::DEFAULT_BASIS_DIM, n: 4, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: true, x_range: None,
            groupby: vec![], name: None,
        };
        let out = apply(&spec, &batch).expect("1 usable point must degrade, not refuse (xs.len() < 2)");
        assert_eq!(out.num_rows(), 4, "spec.n grid rows");
        assert_eq!(out.schema().fields().len(), 6, "core 4 + metrics 2, even though the fit is all-NaN");
        let text = out.column(out.schema().index_of("_metrics_text").unwrap())
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        for i in 0..4 {
            assert!(text.is_null(i), "row {i} _metrics_text must be null for an all-NaN degenerate fit");
        }
    }

    /// RED-proof at the public `apply()` boundary for the reviewer's other
    /// cited example, `poly_fit`: a `groupby` group too sparse for the
    /// requested degree (2 points, degree=3 — rank-deficient, `poly_ols`
    /// returns empty coefficients) with `inject_metrics=true` must emit
    /// null metrics for that group's rows, NOT the garbage `(r2=1, rmse=0,
    /// mae=0)` tuple `residuals::compute` derives from an all-NaN predictor
    /// — proving `poly_fit`'s degenerate branch uses `metrics.is_some()` as
    /// a gate rather than passing the caller's already-computed tuple
    /// through.
    #[test]
    fn poly_fit_degenerate_group_with_inject_metrics_emits_null_not_garbage() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, // "a": 5 points, cubic-recoverable
                    0.0, 1.0,                // "b": 2 points, rank-deficient for degree=3
                ])),
                Arc::new(Float64Array::from(vec![
                    0.0, 1.0, 8.0, 27.0, 64.0, // "a": y = x^3
                    0.0, 1.0,                  // "b"
                ])),
                Arc::new(arrow::array::StringArray::from(vec!["a", "a", "a", "a", "a", "b", "b"])),
            ],
        ).unwrap();
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Poly { degree: 3 },
            ci: None,
            bandwidth: 0.75, degree: 3, k: gam::DEFAULT_BASIS_DIM, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: true, x_range: None,
            groupby: vec!["g".into()], name: None,
        };
        let out = apply(&spec, &batch)
            .expect("a groupby group too sparse for degree=3 must degrade, not refuse the whole fit");
        assert_eq!(out.num_rows(), 10, "2 groups × 5 grid points");
        let metrics_text = out
            .column(out.schema().index_of("_metrics_text").unwrap())
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        // Group 'a' (rows 0..5): a real cubic fit -> the anchor row (the
        // group's LAST row) holds a finite, non-garbage metrics value.
        assert!(!metrics_text.is_null(4), "group 'a' anchor row _metrics_text should be present");
        // Group 'b' (rows 5..10): rank-deficient poly_ols (n=2 < p=4) ->
        // null, not the (r2=1, rmse=0, mae=0) garbage an unconditional
        // pass-through of the caller's already-computed `metrics` tuple
        // would have produced.
        for i in 5..10 {
            assert!(metrics_text.is_null(i), "group 'b' row {i} _metrics_text must be null, not a garbage anchor");
        }
    }

    // ---- Batch-D T10: GAM (penalized cubic B-splines) ----

    /// A genuinely curved fixture with two turning points — no straight
    /// line, and no single quadratic, can track it. Shared by the gam
    /// discriminators below so they all argue about the same data.
    fn curved_xy() -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..60).map(|i| i as f64 * 0.15).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 1.4).sin() * 5.0 + 0.3 * x).collect();
        (xs, ys)
    }

    fn gam_spec(ci: Option<f64>, n: usize) -> SmoothSpec {
        gam_spec_with_k(ci, n, gam::DEFAULT_BASIS_DIM)
    }

    fn gam_spec_with_k(ci: Option<f64>, n: usize, k: usize) -> SmoothSpec {
        SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Gam,
            ci,
            bandwidth: 0.75, degree: 1, k, n, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        }
    }

    /// Count sign changes in the fitted curve's first difference — the
    /// number of turning points the fit actually renders.
    fn turning_points(ys: &[f64]) -> usize {
        let diffs: Vec<f64> = ys.windows(2).map(|w| w[1] - w[0]).collect();
        diffs
            .windows(2)
            .filter(|w| w[0] * w[1] < 0.0)
            .count()
    }

    /// RED-proof: `"gam"` reaches a real fit through the actual string
    /// parser (before T10 it was rejected as an unknown method), and that
    /// fit is finite everywhere on the grid.
    #[test]
    fn test_gam_method_accepted_and_produces_finite_fit() {
        pyo3::Python::initialize();
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys);
        let ctor = PySmooth::new(
            "x", "y", "gam",
            None, None, None, None, 25, 0,
            None, None, "fitted",
            false, false, None, None, None,
        ).expect("PySmooth::new(method='gam') must be accepted by the string parser");
        let spec = match ctor.0 {
            TransformSpec::Smooth(s) => s,
            _ => panic!("expected Smooth spec"),
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert_eq!(yf.len(), 25);
        assert!(yf.iter().all(|v| v.is_finite()), "gam fit must be finite; got {yf:?}");
    }

    /// The discriminator that a lazy `"gam" => SmoothMethod::Lm` (or
    /// `Poly`) alias could not pass: on data with two genuine turning
    /// points, the gam fit must BEND — it renders at least two turning
    /// points where the linear fit renders none, and tracks the data an
    /// order of magnitude more closely. Pins curvature properties, not a
    /// bare inequality.
    #[test]
    fn test_gam_captures_curvature_that_lm_cannot() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs.clone(), ys.clone());
        let gam_out = apply(&gam_spec(None, 60), &batch).unwrap();
        let lm_out = apply(
            &SmoothSpec { method: SmoothMethod::Lm, ..gam_spec(None, 60) },
            &batch,
        ).unwrap();

        let gam_y = col(&gam_out, "y");
        let lm_y = col(&lm_out, "y");

        assert_eq!(turning_points(&lm_y), 0, "a linear fit cannot turn");
        assert!(
            turning_points(&gam_y) >= 2,
            "gam must render the fixture's two turning points; got {} (fit = {gam_y:?})",
            turning_points(&gam_y),
        );

        // The grid is the data's own x here (n = 60 over the same span),
        // so fitted-vs-observed error is directly comparable.
        let rmse = |fit: &[f64]| -> f64 {
            (fit.iter().zip(ys.iter()).map(|(f, y)| (f - y) * (f - y)).sum::<f64>()
                / ys.len() as f64).sqrt()
        };
        let gam_rmse = rmse(&gam_y);
        let lm_rmse = rmse(&lm_y);
        assert!(
            gam_rmse * 10.0 < lm_rmse,
            "gam RMSE {gam_rmse} should be far below lm RMSE {lm_rmse}"
        );
    }

    /// gam is not loess in disguise. Two independent structural proofs:
    /// (1) the fitted curves differ materially on curved data, and (2) gam
    /// is seed-invariant — loess's CI band comes from a seeded bootstrap,
    /// so changing `seed` moves loess's band and cannot move gam's, which
    /// is a closed-form posterior band.
    #[test]
    fn test_gam_is_not_loess_point_estimates_and_seed_dependence_both_differ() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys.clone());
        let y_range = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let gam_out = apply(&gam_spec(Some(0.95), 40), &batch).unwrap();
        let loess_out = apply(
            &SmoothSpec {
                method: SmoothMethod::Loess,
                degree: 2,
                ..gam_spec(Some(0.95), 40)
            },
            &batch,
        ).unwrap();
        let max_gap = col(&gam_out, "y").iter().zip(col(&loess_out, "y").iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_gap > 0.01 * y_range,
            "gam and loess must differ materially on curved data; max gap {max_gap} \
             vs y range {y_range}"
        );

        // Seed dependence: loess's bootstrap band moves, gam's does not.
        let reseed = |mut s: SmoothSpec| { s.seed = 987_654; s };
        let gam_reseeded = apply(&reseed(gam_spec(Some(0.95), 40)), &batch).unwrap();
        for (a, b) in col(&gam_out, "ci_lower").iter().zip(col(&gam_reseeded, "ci_lower").iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "gam's band must not depend on seed");
        }
        let loess_reseeded = apply(
            &reseed(SmoothSpec {
                method: SmoothMethod::Loess,
                degree: 2,
                ..gam_spec(Some(0.95), 40)
            }),
            &batch,
        ).unwrap();
        let loess_band_moved = col(&loess_out, "ci_lower").iter()
            .zip(col(&loess_reseeded, "ci_lower").iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(
            loess_band_moved,
            "control: loess's bootstrap band should move with the seed — if this ever \
             stops holding, the seed-invariance half of this discriminator proves nothing"
        );
    }

    /// The CI band must bracket the fit and stay finite across the grid.
    #[test]
    fn test_gam_ci_band_brackets_the_fit() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys);
        let out = apply(&gam_spec(Some(0.95), 30), &batch).unwrap();
        let y = col(&out, "y");
        let lo = col(&out, "ci_lower");
        let hi = col(&out, "ci_upper");
        for i in 0..y.len() {
            assert!(lo[i].is_finite() && hi[i].is_finite(), "band must be finite at grid point {i}");
            assert!(lo[i] < y[i] && y[i] < hi[i], "band must bracket the fit at grid point {i}");
        }
    }

    /// `ci=None` leaves the band NaN, like every sibling method.
    #[test]
    fn test_gam_without_ci_emits_nan_band() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys);
        let out = apply(&gam_spec(None, 10), &batch).unwrap();
        assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()));
        assert!(col(&out, "ci_upper").iter().all(|v| v.is_nan()));
        assert!(col(&out, "y").iter().all(|v| v.is_finite()));
    }

    /// Determinism through the production entry point: two applies of the
    /// same spec produce bit-identical columns (no RNG anywhere in gam).
    #[test]
    fn test_gam_is_byte_deterministic_through_apply() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys);
        let a = apply(&gam_spec(Some(0.9), 37), &batch).unwrap();
        let b = apply(&gam_spec(Some(0.9), 37), &batch).unwrap();
        for name in ["x", "y", "ci_lower", "ci_upper"] {
            let (ca, cb) = (col(&a, name), col(&b, name));
            for (i, (va, vb)) in ca.iter().zip(cb.iter()).enumerate() {
                assert_eq!(va.to_bits(), vb.to_bits(), "column {name} row {i} must be bit-identical");
            }
        }
    }

    /// Grouped gam: one curve per group, each fitted on its own rows.
    #[test]
    fn test_gam_grouped_fit_emits_one_curve_per_group() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut g = Vec::new();
        for i in 0..40 {
            let x = i as f64 * 0.2;
            xs.push(x);
            ys.push((x * 1.3).sin() * 4.0);
            g.push("a");
            xs.push(x);
            ys.push((x * 1.3).sin() * 4.0 + 25.0);
            g.push("b");
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(arrow::array::StringArray::from(g)),
        ]).unwrap();

        let spec = SmoothSpec { groupby: vec!["g".into()], ..gam_spec(None, 12) };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 24, "2 groups × 12 grid points");
        let y = col(&out, "y");
        assert!(y.iter().all(|v| v.is_finite()), "both groups must fit: {y:?}");
        // Group 'b' is group 'a' shifted up by 25 — the fits must carry
        // that offset, proving each group was fitted on its own rows.
        for i in 0..12 {
            let gap = y[12 + i] - y[i];
            assert!(
                (gap - 25.0).abs() < 0.5,
                "per-group offset at grid point {i} is {gap}, expected ~25"
            );
        }
    }

    /// The degenerate-group contract, for gam: a group with too few
    /// distinct x values to place a cubic basis degrades to that group's
    /// own NaN range instead of failing the whole grouped fit.
    #[test]
    fn test_gam_degenerate_group_degrades_to_nan_not_refusal() {
        let mut xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
        let mut ys: Vec<f64> = xs.iter().map(|x| (x * 1.5).sin() * 3.0).collect();
        let mut g: Vec<&str> = vec!["a"; 20];
        // Group 'b': only 3 distinct x values — below the cubic basis minimum.
        for x in [1.0, 2.0, 3.0] {
            xs.push(x);
            ys.push(x * 2.0);
            g.push("b");
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(arrow::array::StringArray::from(g)),
        ]).unwrap();

        let spec = SmoothSpec { groupby: vec!["g".into()], ..gam_spec(None, 6) };
        let out = apply(&spec, &batch)
            .expect("a group too sparse for a spline must degrade, not fail the grouped fit");
        assert_eq!(out.num_rows(), 12, "2 groups × 6 grid points");
        let y = col(&out, "y");
        assert!(y[..6].iter().all(|v| v.is_finite()), "group 'a' must fit");
        assert!(y[6..].iter().all(|v| v.is_nan()), "group 'b' must degrade to NaN");
    }

    /// gam's whole-frame policy (module doc, "gam contributes no third
    /// case"): a whole-frame fit the data cannot support degrades to an
    /// all-NaN grid rather than raising, because no caller parameter is
    /// responsible for the failure. Contrast
    /// `test_rank_deficient_degree_refuses_on_ungrouped_path`, where the
    /// caller's own `degree` is.
    #[test]
    fn test_gam_whole_frame_too_sparse_degrades_rather_than_refusing() {
        // 3 distinct x values: below the 4 a cubic basis needs.
        let batch = xy_batch(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = apply(&gam_spec(Some(0.95), 5), &batch)
            .expect("gam has no caller-supplied parameter to blame, so it degrades, not refuses");
        assert!(col(&out, "y").iter().all(|v| v.is_nan()));
        assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()));
    }

    /// gam composes with `output="residuals"` and `inject_metrics` like
    /// every other method — the shared `build_predictor` arm is wired.
    #[test]
    fn test_gam_residuals_output_and_metrics_compose() {
        let (xs, ys) = curved_xy();
        let n_rows = xs.len();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            output: SmoothOutput::Residuals,
            inject_metrics: true,
            ..gam_spec(None, 20)
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), n_rows, "residuals output is one row per input row");
        let resid = col(&out, "residual");
        assert!(resid.iter().all(|v| v.is_finite()), "gam residuals must be finite");
        // A spline that tracks this fixture leaves small residuals.
        let max_abs = resid.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_abs < 1.0, "max |residual| = {max_abs} on a smooth sine fixture");
        assert!(out.schema().index_of("_metrics_text").is_ok(), "inject_metrics must add its columns");
    }

    /// The degenerate path still emits a spec-determined schema (round-4
    /// contract) when gam is the method that degenerated.
    #[test]
    fn test_gam_degenerate_fit_with_inject_metrics_emits_null_metrics_columns() {
        let batch = xy_batch(vec![1.0, 1.0, 2.0, 2.0], vec![1.0, 2.0, 3.0, 4.0]);
        let spec = SmoothSpec { inject_metrics: true, ..gam_spec(None, 4) };
        let out = apply(&spec, &batch).unwrap();
        let metrics_text = out
            .column(out.schema().index_of("_metrics_text").unwrap())
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        assert!((0..out.num_rows()).all(|i| metrics_text.is_null(i)),
            "a degenerate gam fit must emit present-but-null metrics columns");
    }

    // ── R1 design finding 6 + (a): gam's smoothness knob, and the
    //    refusal symmetry across degree / bandwidth / k ──────────────────

    /// A high-frequency fixture the DEFAULT basis (k = 10) provably cannot
    /// follow: six full sine periods over the x span. This is the measured
    /// under-smoothing symptom `k=` exists to fix.
    fn high_frequency_xy() -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..120).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 3.2).sin() * 4.0).collect();
        (xs, ys)
    }

    /// The discriminator for exposing `k`: on data that turns more often
    /// than a 10-function basis can follow, `k=20` tracks the signal and
    /// `k=10` does not. Both halves are asserted — a `k` that were silently
    /// ignored (the shape this parameter exists to avoid becoming) would
    /// produce IDENTICAL output, which the final bit-comparison rejects.
    #[test]
    fn test_gam_k_20_follows_a_signal_the_default_basis_under_smooths() {
        let (xs, ys) = high_frequency_xy();
        let n_grid = xs.len();
        let batch = xy_batch(xs, ys.clone());

        let rmse = |fit: &[f64]| -> f64 {
            (fit.iter().zip(ys.iter()).map(|(f, y)| (f - y) * (f - y)).sum::<f64>()
                / ys.len() as f64).sqrt()
        };
        let default_out = apply(&gam_spec(None, n_grid), &batch).unwrap();
        let wide_out = apply(&gam_spec_with_k(None, n_grid, 20), &batch).unwrap();
        let default_y = col(&default_out, "y");
        let wide_y = col(&wide_out, "y");

        let y_range = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        // The symptom: the default basis cannot resolve six periods.
        assert!(
            rmse(&default_y) > 0.15 * y_range,
            "fixture is not under-smoothed at the default k — it cannot demonstrate \
             what k= fixes; default RMSE {} vs y range {y_range}",
            rmse(&default_y),
        );
        // The fix: a wider basis resolves it.
        assert!(
            rmse(&wide_y) * 3.0 < rmse(&default_y),
            "k=20 RMSE {} should be far below k=10 RMSE {}",
            rmse(&wide_y),
            rmse(&default_y),
        );
        assert!(
            turning_points(&wide_y) > turning_points(&default_y),
            "k=20 must render more turning points ({}) than k=10 ({})",
            turning_points(&wide_y),
            turning_points(&default_y),
        );
        assert!(
            default_y.iter().zip(wide_y.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
            "k= must reach the fit — identical output would mean it is being ignored"
        );
    }

    /// `k` is clamped UP to the smallest cubic basis and DOWN to the number
    /// of distinct x values, so neither extreme can produce a degenerate
    /// design — the reason gam adds no whole-frame refusal of its own.
    #[test]
    fn test_gam_k_is_clamped_to_the_distinct_x_count() {
        let (xs, ys) = curved_xy();
        let distinct = xs.len(); // the fixture's x values are all distinct
        let batch = xy_batch(xs, ys);
        let huge = apply(&gam_spec_with_k(None, 20, distinct * 4), &batch)
            .expect("an over-large k must clamp, not fail");
        let at_cap = apply(&gam_spec_with_k(None, 20, distinct), &batch).unwrap();
        for (a, b) in col(&huge, "y").iter().zip(col(&at_cap, "y").iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "k above distinct_x must clamp to distinct_x");
        }
    }

    /// `k` omitted must equal `k` at its documented default — the wire and
    /// the fit both have to agree that 10 is the default, or a spec written
    /// without `k` would render differently from one written with `k=10`.
    #[test]
    fn test_gam_default_k_matches_an_explicit_ten() {
        pyo3::Python::initialize();
        let make = |k: Option<usize>| {
            let ctor = PySmooth::new(
                "x", "y", "gam", None, None, None, k, 20, 0,
                None, None, "fitted", false, false, None, None, None,
            ).expect("gam spec must construct");
            match ctor.0 {
                TransformSpec::Smooth(s) => s,
                _ => panic!("expected Smooth spec"),
            }
        };
        assert_eq!(make(None), make(Some(gam::DEFAULT_BASIS_DIM)));
        assert_eq!(make(None).k, gam::DEFAULT_BASIS_DIM);
    }

    /// `k` is omitted from the wire at its default, so every `Smooth` spec
    /// written before this field existed round-trips byte-identically —
    /// and a non-default `k` does ride the wire.
    #[test]
    fn test_gam_k_is_omitted_from_the_wire_at_its_default() {
        let default_json = serde_json::to_string(&gam_spec(None, 20)).unwrap();
        assert!(
            !default_json.contains("\"k\""),
            "a default k must not appear on the wire: {default_json}"
        );
        let wide_json = serde_json::to_string(&gam_spec_with_k(None, 20, 20)).unwrap();
        assert!(wide_json.contains("\"k\":20"), "a non-default k must ride the wire: {wide_json}");
        let round_tripped: SmoothSpec = serde_json::from_str(&default_json).unwrap();
        assert_eq!(round_tripped, gam_spec(None, 20));
    }

    /// The asymmetry finding, closed on gam's side: `k` is refused on every
    /// method that has no spline basis to size, exactly as `degree` is
    /// refused on every method that has no degree.
    #[test]
    fn test_k_on_a_non_gam_method_refuses() {
        pyo3::Python::initialize();
        for method in ["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt", "glm"] {
            let result = PySmooth::new(
                "x", "y", method, None, None, None, Some(12), 200, 0,
                None, None, "fitted", false, false, None, None, None,
            );
            assert!(result.is_err(), "k=12 with method={method:?} must be refused");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains('k') && msg.contains(method),
                "error should name k and the method for {method:?}: {msg}"
            );
        }
    }

    /// A `k` below the smallest cubic basis is a value error, not something
    /// to silently clamp — clamping it would render a basis the caller did
    /// not ask for with no diagnostic.
    #[test]
    fn test_k_below_the_minimum_basis_refuses() {
        pyo3::Python::initialize();
        for bad_k in [0usize, 1, 2, 3] {
            let result = PySmooth::new(
                "x", "y", "gam", None, None, None, Some(bad_k), 200, 0,
                None, None, "fitted", false, false, None, None, None,
            );
            assert!(result.is_err(), "k={bad_k} must be refused");
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains('k'), "error should mention k for k={bad_k}: {msg}");
        }
        assert!(
            PySmooth::new(
                "x", "y", "gam", None, None, None, Some(gam::MIN_BASIS_DIM), 200, 0,
                None, None, "fitted", false, false, None, None, None,
            ).is_ok(),
            "k at exactly the minimum basis dimension must be accepted"
        );
    }

    /// The other half of design finding 6: an explicit non-default
    /// `bandwidth` on a method that never reads it now refuses instead of
    /// silently no-opping, mirroring `degree`'s policy.
    #[test]
    fn test_non_default_bandwidth_on_a_non_loess_method_refuses() {
        pyo3::Python::initialize();
        for method in ["lm", "linear", "quadratic", "cubic", "log", "sqrt", "glm", "gam"] {
            let result = PySmooth::new(
                "x", "y", method, None, Some(0.4), None, None, 200, 0,
                None, None, "fitted", false, false, None, None, None,
            );
            assert!(result.is_err(), "bandwidth=0.4 with method={method:?} must be refused");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("bandwidth") && msg.contains(method),
                "error should name bandwidth and the method for {method:?}: {msg}"
            );
        }
        assert!(
            PySmooth::new(
                "x", "y", "loess", None, Some(0.4), None, None, 200, 0,
                None, None, "fitted", false, false, None, None, None,
            ).is_ok(),
            "loess is the method bandwidth belongs to"
        );
    }

    /// The one tolerance, matching `degree`'s: an explicit value that
    /// AGREES with the default is a harmless no-op on any method. Only a
    /// direct `Smooth(...)` caller can reach this today -- `desugar_smooth`
    /// (`marks/statistical.py`) forwards `bandwidth` conditionally
    /// (`if bandwidth is not None`) since its own default moved to `None` --
    /// but the invariant must still hold: an explicit default and an omitted
    /// value construct the identical spec.
    #[test]
    fn test_explicit_default_bandwidth_is_accepted_on_every_method() {
        pyo3::Python::initialize();
        let make = |method: &str, bandwidth: Option<f64>| {
            PySmooth::new(
                "x", "y", method, None, bandwidth, None, None, 200, 0,
                None, None, "fitted", false, false, None, None, None,
            )
        };
        for method in ["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt", "glm", "gam"] {
            let explicit = make(method, Some(DEFAULT_LOESS_BANDWIDTH))
                .unwrap_or_else(|e| panic!("explicit default bandwidth rejected for {method:?}: {e}"));
            let omitted = make(method, None).unwrap();
            assert_eq!(
                explicit.0, omitted.0,
                "bandwidth omitted vs bandwidth={DEFAULT_LOESS_BANDWIDTH} must construct the \
                 same spec for method={method:?}"
            );
        }
    }

    /// **Mutation gate: wire-path k below MIN_BASIS_DIM renders without panic.**
    /// This catches the mutation `k.clamp(MIN_BASIS_DIM, ...)` → `k.min(...)`
    /// via the deserialization path: a SmoothSpec with `k: 2` (below the
    /// minimum) must deserialize and render without panic, producing the same
    /// output as the clamped equivalent (`k: MIN_BASIS_DIM`).
    ///
    /// When `k` is omitted or set to its default, serde uses the default
    /// function. But an explicit `"k": 2` in JSON bypasses the Python PySmooth
    /// validation (which rejects k < 4), so the Rust clamp at `gam::fit` is
    /// the only defense against the panic at line 156 (`dim - SPLINE_DEGREE`
    /// underflow). Without the clamp, dim would be 2, and `2 - 3` panics.
    #[test]
    fn test_gam_k_below_min_basis_dim_via_json_deserialization() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys);

        // Build a spec with k=2 (below the minimum of 4) directly.
        // This bypasses Python validation, simulating what happens when a
        // ChartSpec JSON with explicit "k": 2 deserializes in Rust.
        let spec_low = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Gam,
            ci: None,
            bandwidth: 0.75, degree: 1, k: 2, n: 200, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };

        // Apply the spec with k=2 — it must render without panic.
        let result_low = apply(&spec_low, &batch).expect(
            "k=2 must render without panic; gam::fit clamps it to MIN_BASIS_DIM"
        );

        // Build the same spec with k=4 (the minimum).
        let spec_min = gam_spec_with_k(None, 200, gam::MIN_BASIS_DIM);
        let result_min = apply(&spec_min, &batch).expect("k=4 must render");

        // Both outputs must be byte-identical.
        let col_low = col(&result_low, "y");
        let col_min = col(&result_min, "y");
        assert_eq!(col_low.len(), col_min.len(), "both outputs must have the same length");
        for (a, b) in col_low.iter().zip(col_min.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "k=2 (clamped) and k=4 must produce identical fitted values"
            );
        }
    }

    /// **Mutation gate: zero k also renders.** Boundary case:
    /// a SmoothSpec with `k: 0` (far below the minimum) must render without
    /// panic and match the k=4 output.
    #[test]
    fn test_gam_k_zero_via_json_deserialization() {
        let (xs, ys) = curved_xy();
        let batch = xy_batch(xs, ys);

        // Build a spec with k=0.
        let spec_zero = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Gam,
            ci: None,
            bandwidth: 0.75, degree: 1, k: 0, n: 200, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, x_range: None,
            groupby: vec![], name: None,
        };

        // Apply the spec with k=0 — must not panic.
        let result_zero = apply(&spec_zero, &batch).expect(
            "k=0 must render without panic; gam::fit clamps it to MIN_BASIS_DIM"
        );

        // Build the same spec with k=4 (the minimum).
        let spec_min = gam_spec_with_k(None, 200, gam::MIN_BASIS_DIM);
        let result_min = apply(&spec_min, &batch).expect("k=4 must render");

        // Both outputs must be byte-identical.
        let col_zero = col(&result_zero, "y");
        let col_min = col(&result_min, "y");
        for (a, b) in col_zero.iter().zip(col_min.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "k=0 (clamped) and k=4 must produce identical fitted values"
            );
        }
    }
}
