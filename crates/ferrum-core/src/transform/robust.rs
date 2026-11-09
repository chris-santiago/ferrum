//! Robust linear regression transform (Phase 9b).
//!
//! Fits `y ≈ α + β·x` by Huber M-estimation via iteratively reweighted least
//! squares (IRLS) on a 2×2 design `X = [1, x]`. Emits a fitted line over an
//! evenly-spaced grid spanning the input x-range, with optional sandwich-form
//! Wald confidence band on η.
//!
//! Huber's ψ(u) = u for |u| ≤ c, else c·sign(u). The default `huber_c = 1.345`
//! gives ~95% asymptotic efficiency under a Gaussian model while bounding
//! influence of large residuals.
//!
//! Sandwich-form CI (Huber's Proposition 2):
//!
//! ```text
//!   Cov(β̂) ≈ s² · (X'X)⁻¹ · Σ ψ²(uᵢ) / (Σ ψ'(uᵢ))²
//! ```
//!
//! where s is the MAD-scale of residuals and ψ'(u) = 1 for |u| ≤ c, else 0.
//! Then `Var(η at x) = [1, x] · Cov · [1, x]'`, and the CI on η is
//! `η ± z_critical · sqrt(Var(η))`.
//!
//! Output schema (default, `output = Fitted`):
//!   `[x: Float64, y: Float64, ci_lower: Float64, ci_upper: Float64]`
//! With `output = Residuals`, mirrors the Smooth contract:
//!   `[x: Float64, residual: Float64]` at the original (un-gridded) xs.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::color_bin_group::SlotSpace;
use crate::transform::context::{describe_group_keys, TransformContext, TransformWarning};
use crate::transform::core::{SpaceSlot, TransformSpec};
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, parse_groupby_pyany, pooled_group_key,
    stack_grouped_batches,
    NullKeyPolicy,
    KeyValue,
};
use crate::transform::log_space::run_fit_in_log_space;
use crate::transform::logistic::inv_normal_cdf;
use crate::transform::numeric_util::coerce_xy_pairs_at_indexed;
use crate::transform::residuals::{self, ResidualCarry, ResidualColumns};
use crate::transform::smooth::SmoothOutput;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RobustSpec {
    pub x: String,
    pub y: String,
    pub n_grid: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub huber_c: f64,
    pub max_iter: usize,
    pub tol: f64,
    #[serde(default = "crate::transform::smooth::default_smooth_output")]
    pub output: SmoothOutput,
    /// See SmoothSpec.inject_zero_ref — same semantics on the Robust
    /// transform's residuals batch. Schwabish SB-followup (2026-05-12).
    #[serde(default)]
    pub inject_zero_ref: bool,
    /// See SmoothSpec.inject_metrics. Schwabish SB-followup (2026-05-12).
    #[serde(default)]
    pub inject_metrics: bool,
    /// See SmoothSpec.groupby: when non-empty, partition input by these
    /// columns and refit independently per group, re-emitting the group-key
    /// columns (with original dtypes, XFORM-03) as additional output fields.
    /// Batch-F F-L10-04: lets ``residplot(robust=True)`` compose with
    /// ``color=``/``label=`` grouping the same way ``Smooth`` already does.
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
    /// instead of ``[xs.min(), xs.max()]``. Enables ``lmplot(truncate=False)``.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_range: Option<[f64; 2]>,
    /// See `SmoothSpec.x_range_pad` — same contract: a proportional margin
    /// applied to the evaluation range `apply_one_group` resolves
    /// (`x_range` when set, else the whole batch's own `x` extent),
    /// in whichever units the fit runs in. `x_range_pad` alone is not a
    /// no-op: it is the request to extend past the data's own range.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_range_pad: Option<f64>,
    /// See `SmoothSpec.residual_columns` — the identical internal naming
    /// setting, mirrored here so `Robust` gives its own computed columns
    /// fresh names the same way.
    #[serde(default, skip_serializing_if = "ResidualColumns::is_default")]
    pub residual_columns: ResidualColumns,
}

// ---------- Main apply ----------

/// Entry point: dispatches to the grouped or ungrouped fit depending on
/// `spec.groupby`. Mirrors `smooth::apply_with_context`'s dispatch shape;
/// Robust has no context-dependent whole-frame refusal (Smooth's
/// rank-deficient-`degree` case), so it never reads `ctx.is_partition` — the
/// context is threaded through solely for the #162 degrade-warning sink.
pub(crate) fn apply(spec: &RobustSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point (#162): identical dispatch to [`apply`], plus
/// `ctx.warnings` as the sink for this transform's own degrade sites (see
/// `apply_one_group`'s `xs.len() < 2` and zero-variance-x branches below).
///
/// When a drawn log axis demands this fit on its `X` slot, its `Y` slot, or
/// both independently, the fit runs in that space instead
/// ([`apply_in_log_space`]) — the same mechanism `Smooth` joins
/// (`smooth::apply_in_log_space`), since Robust's linear fit is the same
/// shape on either side of the mapping.
pub(crate) fn apply_with_context(
    spec: &RobustSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // The fitted path never reads `residual_columns`; residual mode runs
    // this before any dispatch, mirroring `smooth::apply_with_context`.
    if matches!(spec.output, SmoothOutput::Residuals) {
        spec.residual_columns.check("Robust")?;
    }
    let x_slot = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::X));
    let y_slot = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::Y));
    if x_slot.is_none() && y_slot.is_none() {
        return apply_in_data_space(spec, &spec.x, batch, ctx);
    }
    apply_in_log_space(spec, batch, ctx, x_slot, y_slot)
}

/// [`apply_with_context`]'s ordinary (data-space) body, with `x`/`y` (and
/// `x_range`, when the caller wrote one) already in this fit's own space
/// — data units when not demanded, the mapped batch's log units under
/// [`apply_in_log_space`]. `apply_one_group` resolves the actual
/// evaluation range via
/// [`crate::transform::numeric_util::evaluation_range`], padding once in
/// whichever space this body already runs in.
///
/// `predictor_field` is the caller's OWN predictor column name, never a
/// demanded slot's log-mapped private name: `spec.x` itself when not
/// demanded, but [`apply_in_log_space`]'s own `spec.x` (captured before it
/// builds the mapped spec passed as `spec` here) when it is — mirrors
/// `smooth::apply_in_data_space`'s identical parameter, for the identical
/// reason (`build_residuals_batch`'s carry needs to ask about the caller's
/// own column, not this call's internal fit column).
fn apply_in_data_space(spec: &RobustSpec, predictor_field: &str, batch: &RecordBatch, ctx: &TransformContext) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, predictor_field, batch, &spec.groupby, ctx);
    }
    apply_one_group(spec, predictor_field, batch, None, ctx, None)
}

/// `apply_with_context` under a drawn log axis's demand, mirroring
/// `smooth::apply_in_log_space`: a caller-written `x_range` is mapped into
/// the demanded `x` axis's units (unpadded) before the ordinary body runs
/// over the mapped columns. Robust has one method (Huber IRLS, always
/// linear in its own inputs), so there is no predictor-transform swap and
/// no sqrt-style refusal to make.
fn apply_in_log_space(
    spec: &RobustSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
    x_slot: Option<&SlotSpace>,
    y_slot: Option<&SlotSpace>,
) -> PyResult<RecordBatch> {
    // See `smooth::apply_in_log_space`'s identical call for the full
    // contract: `extent_in_space` never returns `Ok(None)` for a `Some`
    // input, so there is no fallback to write.
    let x_range = x_slot.map_or(Ok(spec.x_range), |slot| {
        slot.space
            .extent_in_space("Robust", "x_range", spec.x_range.map(|[a, b]| (a, b)))
            .map(|r| r.map(|(a, b)| [a, b]))
    })?;
    let mapped_spec = RobustSpec { x_range, ..spec.clone() };
    run_fit_in_log_space(
        ctx,
        "Robust",
        batch,
        &TransformSpec::fit_space_columns(&spec.x, &spec.y, spec.output, &spec.residual_columns.x),
        (&spec.x, x_slot),
        (&spec.y, y_slot),
        |mapped_batch, x, y, ctx| {
            apply_in_data_space(&RobustSpec { x: x.to_string(), y: y.to_string(), ..mapped_spec.clone() }, &spec.x, mapped_batch, ctx)
        },
    )
}

/// Fit one group's (or the whole frame's) rows. `only_indices` restricts the
/// x/y extraction to those row positions when set (a groupby partition);
/// `None` scans every row, byte-identical to this function's pre-groupby body.
/// `group` is a human-readable label for the groupby key (`describe_group_keys`),
/// `None` for a whole-frame call — attached to any degrade warning pushed here.
fn apply_one_group(
    spec: &RobustSpec,
    predictor_field: &str,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    ctx: &TransformContext,
    group: Option<&str>,
) -> PyResult<RecordBatch> {
    let (xs, ys, row_idx) = coerce_xy_pairs_at_indexed(batch, &spec.x, &spec.y, "Robust", only_indices)?;

    let n_grid = spec.n_grid.max(1);

    if xs.len() < 2 {
        ctx.warn(TransformWarning::DegenerateFit {
            transform: "Robust".into(),
            group: group.map(str::to_string),
            reason: "fewer than 2 usable (non-null/non-NaN) x/y points".into(),
            panel: None,
            step: ctx.step.clone(),
        });
        return match spec.output {
            SmoothOutput::Fitted => build_nan_fitted(n_grid, spec.inject_metrics),
            SmoothOutput::Residuals => build_residuals_batch(
                &ResidualCarry { batch, row_idx: &[], predictor_field, group_cols: &spec.groupby },
                &spec.residual_columns,
                Vec::new(), Vec::new(), &[],
                spec.inject_zero_ref, spec.inject_metrics,
            ),
        };
    }

    let (data_x_min, data_x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    // See smooth::apply_one_group's identical call for the full contract.
    let (x_min, x_max) = match crate::transform::numeric_util::evaluation_range(spec.x_range, spec.x_range_pad, batch, &spec.x) {
        Some([lo, hi]) if lo.is_finite() && hi.is_finite() && hi > lo => (lo, hi),
        _ => (data_x_min, data_x_max),
    };
    if !(x_max > x_min) {
        // Zero-variance x → no slope identifiable → NaN line.
        ctx.warn(TransformWarning::DegenerateFit {
            transform: "Robust".into(),
            group: group.map(str::to_string),
            reason: "zero x variance".into(),
            panel: None,
            step: ctx.step.clone(),
        });
        return match spec.output {
            SmoothOutput::Fitted => build_nan_fitted(n_grid, spec.inject_metrics),
            SmoothOutput::Residuals => {
                let nans = vec![f64::NAN; xs.len()];
                build_residuals_batch(
                    &ResidualCarry { batch, row_idx: &row_idx, predictor_field, group_cols: &spec.groupby },
                    &spec.residual_columns,
                    xs.clone(), nans, &ys,
                    spec.inject_zero_ref, spec.inject_metrics,
                )
            }
        };
    }

    // 1) OLS init.
    let (mut alpha, mut beta) = ols_2x2(&xs, &ys);

    // 2) IRLS loop.
    let c = spec.huber_c;
    let mut s = 0.0_f64;
    let mut weights: Vec<f64> = vec![1.0; xs.len()];
    let mut u_vec: Vec<f64> = vec![0.0; xs.len()];

    for _ in 0..spec.max_iter.max(1) {
        // Residuals.
        let r: Vec<f64> = xs.iter().zip(ys.iter())
            .map(|(&xi, &yi)| yi - (alpha + beta * xi))
            .collect();

        // Scale via MAD × 1.4826.
        s = mad_scale(&r);
        if s == 0.0 {
            // Exact fit (or degenerate) → no further update possible.
            break;
        }

        // u = r / s, weights w = ψ(u)/u.
        for i in 0..xs.len() {
            let ui = r[i] / s;
            u_vec[i] = ui;
            weights[i] = huber_weight(ui, c);
        }

        // Weighted LS solve.
        let (a_new, b_new) = match wls_2x2(&xs, &ys, &weights) {
            Some(v) => v,
            None => break,
        };
        if !a_new.is_finite() || !b_new.is_finite() { break; }

        let da = (a_new - alpha).abs();
        let db = (b_new - beta).abs();
        alpha = a_new;
        beta = b_new;
        if da.max(db) < spec.tol {
            break;
        }
    }

    // After loop: recompute residuals, u, weights at final β for output & CI.
    let r_final: Vec<f64> = xs.iter().zip(ys.iter())
        .map(|(&xi, &yi)| yi - (alpha + beta * xi))
        .collect();

    if matches!(spec.output, SmoothOutput::Residuals) {
        return build_residuals_batch(
            &ResidualCarry { batch, row_idx: &row_idx, predictor_field, group_cols: &spec.groupby },
            &spec.residual_columns,
            xs.clone(), r_final, &ys,
            spec.inject_zero_ref, spec.inject_metrics,
        );
    }

    // Schwabish SB-followup (C5, 2026-05-12): when ``inject_metrics``
    // is set on the Fitted path, compute R²/RMSE/MAE against the input
    // (xs, ys) using the converged (α, β). Mirrors the Smooth Fitted
    // path's contract — text and y-anchor format are byte-identical
    // across the two transforms because both go through
    // ``residuals::append_metrics_columns``.
    let metrics = spec.inject_metrics
        .then(|| residuals::compute(&ys, &r_final));

    // Refresh scale + u for CI computation.
    s = mad_scale(&r_final);
    let n = xs.len();

    // Grid and fitted line.
    let grid: Vec<f64> = grid_vec(x_min, x_max, n_grid);
    let y_fit: Vec<f64> = grid.iter().map(|&xq| alpha + beta * xq).collect();

    // `band_undefined` collects WHY the band (not the line) is missing,
    // across every branch below, reported once at the end — mirroring
    // `smooth::apply_one_group`'s post-dispatch report, so a degenerate
    // band is never reported from more than one place in this function.
    let (lo, hi, band_undefined): (Vec<f64>, Vec<f64>, Option<String>) = if s == 0.0 {
        // Exact fit (or every residual on the fitted line): the robust
        // scale is 0, so the sandwich band has no width to compute. `n <=
        // 2` is the same residual-degrees-of-freedom shape `lm` reports for
        // a 2-point fit (a 2-point line is always exact, so this is the
        // only way `n <= 2` reaches here); `n > 2` is a genuine zero-MAD
        // degeneracy where at least half the points lie exactly on the
        // fitted line.
        let reason = if n <= 2 {
            crate::transform::context::zero_residual_dof_reason(n, 2)
        } else {
            "the residuals' median absolute deviation is 0 (at least half the points lie on \
             the line), so the robust scale gives the band no width"
                .to_string()
        };
        (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid], Some(reason))
    } else {
        for i in 0..xs.len() {
            u_vec[i] = r_final[i] / s;
        }
        match spec.ci {
            None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid], None),
            Some(level) => {
                if !(level > 0.0 && level < 1.0) {
                    return Err(PyValueError::new_err("Robust: ci must be in (0, 1)"));
                }
                let z = inv_normal_cdf(1.0 - (1.0 - level) / 2.0);
                // Cov(β̂) = s² · (X'X)⁻¹ · Σψ²(u) / (Σψ'(u))²
                match invert_2x2(xtx_unweighted(&xs)) {
                    // The line fit fine (IRLS converged); only the
                    // sandwich covariance is undefined because the design
                    // is singular.
                    None => (
                        vec![f64::NAN; n_grid],
                        vec![f64::NAN; n_grid],
                        Some(
                            "the design is singular, so the sandwich covariance has no inverse"
                                .to_string(),
                        ),
                    ),
                    Some(xtx_inv) => {
                        let mut sum_psi_sq = 0.0_f64;
                        let mut sum_psi_prime = 0.0_f64;
                        for &ui in &u_vec {
                            let psi = huber_psi(ui, c);
                            sum_psi_sq += psi * psi;
                            sum_psi_prime += if ui.abs() <= c { 1.0 } else { 0.0 };
                        }
                        if sum_psi_prime <= 0.0 {
                            // Every residual exceeds the Huber threshold,
                            // so Σψ'(u) is zero and the sandwich covariance
                            // has no scalar to scale by. The line still
                            // drew; only the band is lost.
                            (
                                vec![f64::NAN; n_grid],
                                vec![f64::NAN; n_grid],
                                Some(
                                    "every residual is beyond the Huber threshold, leaving no \
                                     usable weights for the sandwich covariance"
                                        .to_string(),
                                ),
                            )
                        } else {
                            let scalar = s * s * sum_psi_sq / (sum_psi_prime * sum_psi_prime);
                            let cov = scale_2x2(xtx_inv, scalar);
                            let mut lo = Vec::with_capacity(n_grid);
                            let mut hi = Vec::with_capacity(n_grid);
                            let mut undefined = 0usize;
                            for &xq in &grid {
                                let v = cov[0][0] + 2.0 * xq * cov[0][1] + xq * xq * cov[1][1];
                                let se = if v >= 0.0 && v.is_finite() {
                                    v.sqrt()
                                } else {
                                    undefined += 1;
                                    f64::NAN
                                };
                                let eta = alpha + beta * xq;
                                lo.push(eta - z * se);
                                hi.push(eta + z * se);
                            }
                            // A negative or non-finite sandwich variance at
                            // only some grid points is a numerical artifact
                            // (x's spread tiny next to its magnitude), not a
                            // lost fit, so only those points' band is
                            // undefined.
                            let reason = (undefined > 0).then(|| {
                                crate::transform::context::partial_band_reason(
                                    "the sandwich variance is negative (x's spread is tiny \
                                     next to its magnitude; centering x avoids it)",
                                    undefined,
                                    n_grid,
                                )
                            });
                            (lo, hi, reason)
                        }
                    }
                }
            }
        }
    };

    if let (Some(level), Some(reason)) = (spec.ci, band_undefined) {
        ctx.warn(TransformWarning::IntervalUndefined {
            transform: "Robust".into(),
            group: group.map(str::to_string),
            ci: level,
            reason,
            panel: None,
            step: ctx.step.clone(),
        });
    }

    build_fitted_batch(grid, y_fit, lo, hi, metrics)
}

/// Partition input batch by the groupby columns, call `apply_one_group` per
/// partition, then stack the results into a single batch with the group
/// columns appended (with their original dtypes, XFORM-03) as additional
/// fields.
///
/// Mirrors `smooth::apply_grouped`'s stacking shape (batch-D T9/T3.8a):
/// first-appearance group order, null-key skip (`group_partition_multi`),
/// and dtype-preserving group-column re-emission, both via the shared
/// `group_key::stack_grouped_batches` template (batch-F T2h hoist — the two
/// `apply_grouped` bodies were ~30 lines differing only in the per-group fit
/// call and the error-context string). Robust's schema is spec-determined —
/// the same `output`/`inject_metrics` combination on every group — so no
/// group can legitimately diverge from its siblings; the by-name schema
/// reconciliation inside `stack_grouped_batches` is belt-and-braces the same
/// way it is for Smooth.
fn apply_grouped(
    spec: &RobustSpec,
    predictor_field: &str,
    batch: &RecordBatch,
    group_cols: &[String],
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // `SkipWithWholeBatchFallback` is #197's stated exemption for this
    // transform: the all-null-key arm below re-fits the whole batch rather
    // than emitting nothing, and the warning says so.
    let partition = group_partition_multi(
        batch,
        group_cols,
        "Robust",
        NullKeyPolicy::SkipWithWholeBatchFallback,
        ctx,
    )?;

    let mut per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = Vec::new();
    for (g, ixs) in &partition.groups {
        let label = describe_group_keys(group_cols, g);
        let out = apply_one_group(spec, predictor_field, batch, Some(ixs), ctx, Some(&label))?;
        per_group_batches.push((g.clone(), out));
    }

    if per_group_batches.is_empty() {
        // Every group key was null: there is no per-group output to stack.
        // Fall back to the full batch on `None` indices, mirroring
        // `smooth::apply_grouped`'s identical fallback — the pooled fit IS
        // the null-keyed group, stacked with a key that nulls only the
        // `groupby` column(s) actually all-null (`pooled_group_key`'s
        // doc), so a facet field folded into `group_cols` keeps its value.
        let pooled = apply_one_group(spec, predictor_field, batch, None, ctx, None)?;
        let key = pooled_group_key(batch, group_cols)?;
        return stack_grouped_batches("Robust", &[(key, pooled)], group_cols, &partition.dtypes);
    }

    stack_grouped_batches("Robust", &per_group_batches, group_cols, &partition.dtypes)
}

// ---------- Helpers ----------

fn grid_vec(x_min: f64, x_max: f64, n_grid: usize) -> Vec<f64> {
    (0..n_grid)
        .map(|i| if n_grid <= 1 { x_min }
             else { x_min + (x_max - x_min) * (i as f64) / ((n_grid - 1) as f64) })
        .collect()
}

/// Closed-form 2×2 OLS for X=[1, x]. Returns (α, β); falls back to (0, 0) on
/// zero-variance x (caller already guards but keep safe).
fn ols_2x2(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n == 0.0 { return (0.0, 0.0); }
    let mean_x: f64 = xs.iter().sum::<f64>() / n;
    let mean_y: f64 = ys.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
    if sxx == 0.0 { return (mean_y, 0.0); }
    let beta = sxy / sxx;
    let alpha = mean_y - beta * mean_x;
    (alpha, beta)
}

/// Weighted least squares on the same 2×2 design X=[1, x].
/// Solves (X'WX)β = X'Wy. Returns None on singular system.
fn wls_2x2(xs: &[f64], ys: &[f64], w: &[f64]) -> Option<(f64, f64)> {
    let mut wxtx = [[0.0_f64; 2]; 2];
    let mut wxty = [0.0_f64; 2];
    for i in 0..xs.len() {
        let wi = w[i];
        let xi = xs[i];
        let yi = ys[i];
        wxtx[0][0] += wi;
        wxtx[0][1] += wi * xi;
        wxtx[1][0] += wi * xi;
        wxtx[1][1] += wi * xi * xi;
        wxty[0] += wi * yi;
        wxty[1] += wi * xi * yi;
    }
    let inv = invert_2x2(wxtx)?;
    let a = inv[0][0] * wxty[0] + inv[0][1] * wxty[1];
    let b = inv[1][0] * wxty[0] + inv[1][1] * wxty[1];
    Some((a, b))
}

use crate::transform::linalg::{invert_2x2, scale_2x2, xtx_unweighted};

/// MAD-based robust scale estimate: s = 1.4826 · median(|r - median(r)|).
fn mad_scale(r: &[f64]) -> f64 {
    if r.is_empty() { return 0.0; }
    let med = median(r);
    let abs_dev: Vec<f64> = r.iter().map(|&v| (v - med).abs()).collect();
    1.4826 * median(&abs_dev)
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

/// Huber ψ(u) = u for |u| ≤ c, else c·sign(u).
fn huber_psi(u: f64, c: f64) -> f64 {
    if u.abs() <= c { u } else if u > 0.0 { c } else { -c }
}

/// Huber weight w(u) = ψ(u)/u, with w(0) = 1.
fn huber_weight(u: f64, c: f64) -> f64 {
    if u == 0.0 { 1.0 }
    else if u.abs() <= c { 1.0 }
    else { c / u.abs() }
}

/// Schwabish SB-followup (C5 audit-rework, 2026-05-12): accepts an
/// optional ``metrics`` tuple so the Fitted path can carry the same
/// ``_metrics_text`` / ``_metrics_y`` annotation columns Smooth emits.
/// Delegates column emission to ``crate::transform::residuals`` so the
/// schema + text-format invariant lives in one place across Smooth and
/// Robust.
fn build_fitted_batch(
    x: Vec<f64>, y: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    // Batch-D M6 schema-collapse (spec §4.6): shared with `smooth.rs`
    // (Fitted)/`logistic.rs`/`glm.rs` and the `transform_registry` entries
    // for all four.
    let mut fields = crate::transform::registry::fit_output_fields();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x.clone())),
        Arc::new(Float64Array::from(y.clone())),
        Arc::new(Float64Array::from(lo)),
        Arc::new(Float64Array::from(hi)),
    ];
    if let Some(m) = metrics {
        // The Fitted path never reads `residual_columns`, so its metrics
        // columns stay the stock names.
        residuals::append_metrics_columns(
            &mut fields, &mut cols, &x, &y, m,
            residuals::DEFAULT_RESIDUAL_METRICS_TEXT, residuals::DEFAULT_RESIDUAL_METRICS_Y,
        );
    }
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("Robust: {e}")))
}

fn build_nan_fitted(n_grid: usize, metrics_requested: bool) -> PyResult<RecordBatch> {
    let nans = vec![f64::NAN; n_grid];
    let mut fields = crate::transform::registry::fit_output_fields();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(nans.clone())),
        Arc::new(Float64Array::from(nans.clone())),
        Arc::new(Float64Array::from(nans.clone())),
        Arc::new(Float64Array::from(nans.clone())),
    ];
    if metrics_requested {
        residuals::append_null_metrics_columns(&mut fields, &mut cols, n_grid);
    }
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("Robust: {e}")))
}

/// Schwabish SB-followup (2026-05-12): mirrors
/// ``smooth::build_residuals_batch`` — emits optional ``_ref_zero`` /
/// ``_metrics_text`` / ``_metrics_y`` columns when the corresponding
/// inject flag is set. The metrics computation needs the original
/// ``y_obs`` to derive R² / RMSE / MAE; delegates column emission to
/// `crate::transform::residuals` so the schema + text-format invariant
/// is owned in one home (post-C4 rework, 2026-05-12).
///
/// Row-preserving (the `Bin(per_row=True)` contract, applied here): every
/// other column `carry.batch` holds is carried at exactly the row
/// positions `carry.row_idx` names, in order — see
/// `residuals::append_carried_columns`'s own doc for the exclude/collision
/// rule, derived there from the computed columns this builds below rather
/// than hand-listed again here; mirrors `smooth::fits::build_residuals_batch`.
fn build_residuals_batch(
    carry: &ResidualCarry,
    names: &ResidualColumns,
    xs: Vec<f64>, resid: Vec<f64>, ys_obs: &[f64],
    inject_zero_ref: bool, inject_metrics: bool,
) -> PyResult<RecordBatch> {
    let n = xs.len();
    let mut fields = vec![
        Field::new(&names.x,        DataType::Float64, true),
        Field::new(&names.residual, DataType::Float64, true),
    ];
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(xs.clone())),
        Arc::new(Float64Array::from(resid.clone())),
    ];

    if inject_zero_ref {
        residuals::append_zero_ref_column(&mut fields, &mut cols, n, &names.ref_zero);
    }

    if inject_metrics {
        let metrics = residuals::compute(ys_obs, &resid);
        residuals::append_metrics_columns(&mut fields, &mut cols, &xs, &resid, metrics, &names.metrics_text, &names.metrics_y);
    }

    residuals::append_carried_columns("Robust", &mut fields, &mut cols, carry)?;

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("Robust: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

/// Robust regression fit (Huber M-estimator via IRLS).
///
/// Fits a linear model using an iteratively reweighted least squares
/// procedure with Huber weights, making the fit resistant to outliers.
/// Evaluates the fitted line on a grid of ``n_grid`` evenly-spaced x-values.
/// When ``ci`` is set, also emits Wald-style confidence bands.
///
/// Output columns: ``x`` (Float64 grid points), ``y`` (Float64 fitted
/// values), ``ci_lower`` and ``ci_upper`` (Float64; ``NaN`` when ``ci`` is
/// not set).
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
/// n_grid : int, default 100
///     Number of evenly-spaced grid points for the fitted line. Must be > 0.
/// ci : float, optional
///     Confidence level in (0, 1) for confidence bands (e.g. 0.95). When
///     omitted, ``ci_lower`` and ``ci_upper`` are ``NaN``.
/// huber_c : float, default 1.345
///     Huber loss transition parameter (``c`` in ``ρ(r) = r²/2`` for
///     ``|r| ≤ c``, else ``c|r| - c²/2``). Must be finite and > 0.
/// max_iter : int, default 25
///     Maximum IRLS iterations. Must be > 0.
/// tol : float, default 1e-8
///     Convergence tolerance (relative change in coefficients). Must be
///     finite and > 0.
/// output : {"fitted", "residuals"}, default "fitted"
///     When ``"residuals"``, the ``y`` column contains ``y_obs - y_fitted``
///     instead of the fitted values.
/// groupby : str | list[str], optional
///     When set, partition the input by these column(s) and refit
///     independently per group, appending the group-key column(s) (at
///     their original dtype) to the output. Mirrors ``Smooth(groupby=)``.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_smooth(method="robust").encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Robust")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyRobust(pub(crate) TransformSpec);

#[pymethods]
impl PyRobust {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        n_grid = 100,
        ci = None,
        huber_c = 1.345,
        max_iter = 25,
        tol = 1e-8,
        output = "fitted",
        inject_zero_ref = false,
        inject_metrics = false,
        groupby = None,
        name = None,
        x_range = None,
        x_range_pad = None,
    ))]
    fn new(
        x: &str,
        y: &str,
        n_grid: usize,
        ci: Option<f64>,
        huber_c: f64,
        max_iter: usize,
        tol: f64,
        output: &str,
        inject_zero_ref: bool,
        inject_metrics: bool,
        groupby: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        x_range: Option<[f64; 2]>,
        x_range_pad: Option<f64>,
    ) -> PyResult<Self> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Robust: x and y must be non-empty"));
        }
        if let Some([a, b]) = x_range {
            if !a.is_finite() || !b.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "Robust: x_range must be (lo, hi) with both finite; received [{a}, {b}]"
                )));
            }
        }
        if let Some(pad) = x_range_pad {
            if !pad.is_finite() || pad < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "Robust: x_range_pad must be a finite fraction >= 0; received {pad}"
                )));
            }
        }
        if n_grid == 0 {
            return Err(PyValueError::new_err("Robust: n_grid must be > 0"));
        }
        if max_iter == 0 {
            return Err(PyValueError::new_err("Robust: max_iter must be > 0"));
        }
        if !(tol.is_finite() && tol > 0.0) {
            return Err(PyValueError::new_err("Robust: tol must be finite and > 0"));
        }
        if !(huber_c.is_finite() && huber_c > 0.0) {
            return Err(PyValueError::new_err("Robust: huber_c must be finite and > 0"));
        }
        if let Some(level) = ci {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err("Robust: ci must be in (0, 1)"));
            }
        }
        let output_parsed = match output {
            "fitted" => SmoothOutput::Fitted,
            "residuals" => SmoothOutput::Residuals,
            other => return Err(PyValueError::new_err(format!(
                "Robust: unknown output '{other}'; expected 'fitted'|'residuals'"
            ))),
        };

        if inject_zero_ref
            && !matches!(output_parsed, SmoothOutput::Residuals)
        {
            return Err(PyValueError::new_err(
                "Robust: inject_zero_ref requires output='residuals'",
            ));
        }
        // inject_metrics is allowed on both Fitted and Residuals — mirrors
        // Smooth's policy (post-C5 audit-rework, 2026-05-12). On Fitted the
        // metrics annotation lands on the fit-grid output via
        // ``residuals::append_metrics_columns`` after the IRLS converges.
        let groupby = parse_groupby_pyany(groupby)?;
        Ok(PyRobust(TransformSpec::Robust(RobustSpec {
            x: x.to_string(),
            y: y.to_string(),
            n_grid,
            ci,
            huber_c,
            max_iter,
            tol,
            output: output_parsed,
            inject_zero_ref,
            inject_metrics,
            groupby,
            name,
            x_range,
            x_range_pad,
            residual_columns: ResidualColumns::default(),
        })))
    }

    /// Internal: mirrors `PySmooth`'s identical private method (see its
    /// doc for the full contract) — gives `Robust`'s own computed
    /// residual-mode columns fresh names, refusing on a `Fitted` spec or
    /// a malformed name set, with `Robust`'s own voice.
    #[pyo3(signature = (*, x=None, residual=None, ref_zero=None, metrics_text=None, metrics_y=None))]
    fn _with_residual_columns(
        &self,
        x: Option<String>,
        residual: Option<String>,
        ref_zero: Option<String>,
        metrics_text: Option<String>,
        metrics_y: Option<String>,
    ) -> PyResult<Self> {
        // `PyRobust` only ever wraps `TransformSpec::Robust` (its only
        // constructor, `new` above, guarantees it), so the other arm is
        // unreachable through the public API; it still returns a typed
        // error rather than `unreachable!()`, mirroring `PySmooth`'s own
        // identical method, so a future internal caller that got this
        // wrong fails safely at the PyO3 boundary instead of panicking.
        let TransformSpec::Robust(spec) = &self.0 else {
            return Err(PyValueError::new_err(
                "Robust: internal error — _with_residual_columns called on a spec that is not \
                 Robust; this is a bug in ferrum, not a user error",
            ));
        };
        if !matches!(spec.output, SmoothOutput::Residuals) {
            return Err(PyValueError::new_err(
                "Robust: residual column names apply only to output='residuals'",
            ));
        }
        let mut names = spec.residual_columns.clone();
        if let Some(v) = x { names.x = v; }
        if let Some(v) = residual { names.residual = v; }
        if let Some(v) = ref_zero { names.ref_zero = v; }
        if let Some(v) = metrics_text { names.metrics_text = v; }
        if let Some(v) = metrics_y { names.metrics_y = v; }
        names.check("Robust")?;
        let mut spec = spec.clone();
        spec.residual_columns = names;
        Ok(Self(TransformSpec::Robust(spec)))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Robust(s) => format!(
                "Robust(x='{}', y='{}', n_grid={}, ci={:?}, huber_c={}, max_iter={}, tol={}, output={:?}, inject_zero_ref={}, inject_metrics={}, groupby={:?}, name={:?}, x_range={:?}, x_range_pad={:?})",
                s.x, s.y, s.n_grid, s.ci, s.huber_c, s.max_iter, s.tol, s.output,
                s.inject_zero_ref, s.inject_metrics, s.groupby, s.name, s.x_range, s.x_range_pad,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::color_bin_group::StepSpace;
    use arrow::array::{Array, Float64Array, RecordBatch, StringArray};
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

    /// `xy_batch` plus a Utf8 groupby column `g`, for the grouped-fit tests
    /// below. Mirrors `smooth.rs`'s own groupby test fixtures.
    fn xy_batch_with_group(x: Vec<f64>, y: Vec<f64>, g: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
            Arc::new(StringArray::from(g)),
        ]).unwrap()
    }

    fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) }).collect()
    }

    fn spec(n_grid: usize, ci: Option<f64>, huber_c: f64, output: SmoothOutput) -> RobustSpec {
        RobustSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            n_grid, ci, huber_c,
            max_iter: 50, tol: 1e-10,
            output,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            x_range: None, x_range_pad: None,
            name: None,
        }
    }

    #[test]
    fn test_robust_round_trip_json() {
        let s = RobustSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            n_grid: 100, ci: Some(0.95),
            huber_c: 1.345, max_iter: 25, tol: 1e-8,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            name: None,
            x_range: None, x_range_pad: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: RobustSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);

        // Also check residuals variant + name.
        let s2 = RobustSpec {
            residual_columns: Default::default(),
            x: "a".into(), y: "b".into(),
            n_grid: 50, ci: None,
            huber_c: 2.0, max_iter: 10, tol: 1e-6,
            output: SmoothOutput::Residuals,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            name: Some("r1".into()),
            x_range: None, x_range_pad: None,
        };
        let json2 = serde_json::to_string(&s2).unwrap();
        let parsed2: RobustSpec = serde_json::from_str(&json2).unwrap();
        assert_eq!(parsed2, s2);
    }

    #[test]
    fn test_robust_clean_linear_recovers_ols() {
        // y = 2x + 1 exactly → Robust should converge to (1, 2) within 1e-6.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(20, None, 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        let x = col(&out, "x");
        let y = col(&out, "y");
        // On a perfect line, fit at x[0] should be 1.0, at x[last] 2*19 + 1 = 39.0.
        assert!((y[0] - (2.0 * x[0] + 1.0)).abs() < 1e-6,
            "fitted line should match y=2x+1; got y[0]={}", y[0]);
        let last = y.len() - 1;
        assert!((y[last] - (2.0 * x[last] + 1.0)).abs() < 1e-6,
            "fitted line should match y=2x+1; got y[last]={}", y[last]);
    }

    /// Compute (α, β) directly via Robust IRLS for tests.
    fn fit_ab(xs: &[f64], ys: &[f64], huber_c: f64, max_iter: usize, tol: f64) -> (f64, f64) {
        // Reproduce the IRLS path to expose β for property tests.
        let (mut alpha, mut beta) = ols_2x2(xs, ys);
        for _ in 0..max_iter {
            let r: Vec<f64> = xs.iter().zip(ys.iter())
                .map(|(&xi, &yi)| yi - (alpha + beta * xi)).collect();
            let s = mad_scale(&r);
            if s == 0.0 { break; }
            let w: Vec<f64> = r.iter().map(|&ri| huber_weight(ri / s, huber_c)).collect();
            let (a_new, b_new) = wls_2x2(xs, ys, &w).unwrap_or((alpha, beta));
            let da = (a_new - alpha).abs(); let db = (b_new - beta).abs();
            alpha = a_new; beta = b_new;
            if da.max(db) < tol { break; }
        }
        (alpha, beta)
    }

    fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
        let (_, b) = ols_2x2(xs, ys);
        b
    }

    #[test]
    fn test_robust_beats_ols_under_10pct_outliers() {
        // True slope 2.0; insert 10% large positive y-outliers.
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        // 10% of 50 = 5 outliers.
        for &i in &[5usize, 15, 25, 35, 45] {
            ys[i] += 100.0;
        }
        let (_, beta_r) = fit_ab(&xs, &ys, 1.345, 50, 1e-10);
        let beta_ols = ols_slope(&xs, &ys);
        assert!((beta_r - 2.0).abs() < 0.1,
            "Robust slope should be within 0.1 of true=2.0; got {beta_r}");
        // OLS should be more biased than Robust.
        assert!((beta_ols - 2.0).abs() > (beta_r - 2.0).abs(),
            "Robust ({beta_r}) should be closer to 2.0 than OLS ({beta_ols})");
    }

    #[test]
    fn test_robust_under_30pct_outliers() {
        // 30% outliers spread uniformly across the x range, alternating sign so
        // they don't induce a directional pull on β; Huber should largely
        // down-weight them and recover the true slope.
        let xs: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let n_out = 18; // 30% of 60
        for k in 0..n_out {
            // Spread outlier indices evenly across the full x range.
            let i = (k as f64 * (xs.len() as f64) / (n_out as f64)).floor() as usize;
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            ys[i] += sign * 50.0;
        }
        let (_, beta_r) = fit_ab(&xs, &ys, 1.345, 100, 1e-10);
        assert!((beta_r - 2.0).abs() < 0.2,
            "Robust slope should be within 0.2 of true=2.0 under 30% outliers; got {beta_r}");
    }

    #[test]
    fn test_robust_ci_brackets_fit() {
        // Noisy linear data so MAD scale > 0 → finite CI band.
        let xs: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 2.0 * x + 1.0 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let batch = xy_batch(xs.clone(), ys);
        let s = spec(30, Some(0.95), 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        let y = col(&out, "y");
        let lo = col(&out, "ci_lower");
        let hi = col(&out, "ci_upper");

        // Mean x is at index ≈ 14 of grid. Check at least one mid-grid index brackets fit.
        let mid = y.len() / 2;
        assert!(lo[mid].is_finite() && hi[mid].is_finite(),
            "CI must be finite at mid-grid; got lo={}, hi={}", lo[mid], hi[mid]);
        assert!(lo[mid] <= y[mid] + 1e-9 && y[mid] - 1e-9 <= hi[mid],
            "CI must bracket fit at mid; lo={}, y={}, hi={}", lo[mid], y[mid], hi[mid]);
    }

    #[test]
    fn test_robust_residuals_schema_and_zero_sum() {
        // For perfectly linear data, all residuals are ~0. The response
        // column "y" is carried alongside the computed [x, residual] pair —
        // it is not one of this transform's own output names, so the
        // row-preserving contract keeps it under its own name.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let s = spec(100, None, 1.345, SmoothOutput::Residuals);
        let out = apply(&s, &batch).unwrap();

        // Schema check: the computed [x, residual] pair plus the carried "y".
        let schema = out.schema();
        assert_eq!(schema.fields().len(), 3, "x, residual, and the carried response column y");
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "residual");
        assert_eq!(schema.field(2).name(), "y");
        assert_eq!(col(&out, "y"), ys, "the carried y column holds the caller's own response values");

        // Sum-of-residuals ≈ 0.
        let r = col(&out, "residual");
        assert_eq!(r.len(), xs.len(), "residuals should be at original xs");
        let sum: f64 = r.iter().sum();
        assert!(sum.abs() < 1e-6, "sum of residuals on clean linear should be ~0; got {sum}");
    }

    #[test]
    fn test_robust_default_fitted_schema() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(25, Some(0.95), 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        let schema = out.schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "y");
        assert_eq!(schema.field(2).name(), "ci_lower");
        assert_eq!(schema.field(3).name(), "ci_upper");
        assert_eq!(out.num_rows(), 25);
    }

    #[test]
    fn test_robust_fitted_inject_metrics_columns_present() {
        // Schwabish C5: inject_metrics=true on Fitted output should
        // append _metrics_text + _metrics_y columns with exactly one
        // non-null row, mirroring Smooth's behavior.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 0.5 * x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let s = RobustSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            n_grid: 30, ci: Some(0.95),
            huber_c: 1.345, max_iter: 50, tol: 1e-10,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: true,
            groupby: vec![],
            x_range: None, x_range_pad: None,
            name: None,
        };
        let out = apply(&s, &batch).unwrap();
        let schema = out.schema();
        // Base 4 columns + 2 injected columns.
        assert_eq!(schema.fields().len(), 6);
        assert_eq!(schema.field(4).name(), "_metrics_text");
        assert_eq!(schema.field(5).name(), "_metrics_y");

        // Exactly one non-null row in _metrics_text; format starts with "R²".
        let text_arr = out.column(4).as_any()
            .downcast_ref::<StringArray>()
            .expect("_metrics_text column should be Utf8");
        let non_null: Vec<usize> = (0..text_arr.len())
            .filter(|&i| !text_arr.is_null(i))
            .collect();
        assert_eq!(non_null.len(), 1, "exactly one non-null _metrics_text row");
        let text = text_arr.value(non_null[0]);
        assert!(text.starts_with("R² "),
            "metrics text must start with 'R² '; got {text:?}");

        // _metrics_y at the same row is finite (anchor y).
        let y_anchor = col(&out, "_metrics_y")[non_null[0]];
        assert!(y_anchor.is_finite(),
            "_metrics_y anchor must be finite; got {y_anchor}");
    }

    #[test]
    fn test_robust_huber_c_sensitivity() {
        // Outlier-laden data: small c should pull β closer to true 2.0 than huge c.
        let xs: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        for i in (0..40).step_by(4) {
            ys[i] += 50.0;
        }
        let (_, beta_small) = fit_ab(&xs, &ys, 0.5, 100, 1e-10);
        let (_, beta_huge) = fit_ab(&xs, &ys, 10.0, 100, 1e-10);
        assert!((beta_small - 2.0).abs() < (beta_huge - 2.0).abs(),
            "small huber_c={beta_small} should be closer to 2.0 than huge huber_c={beta_huge}");
    }

    #[test]
    fn test_robust_max_iter_1_finite() {
        // Single-pass: should still produce finite output (won't have fully converged).
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0 + if (x as i64) % 3 == 0 { 5.0 } else { 0.0 }).collect();
        let batch = xy_batch(xs, ys);
        let s = RobustSpec {
            residual_columns: Default::default(),
            x: "x".into(), y: "y".into(),
            n_grid: 10, ci: None,
            huber_c: 1.345, max_iter: 1, tol: 1e-8,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            x_range: None, x_range_pad: None,
            name: None,
        };
        let out = apply(&s, &batch).unwrap();
        let y = col(&out, "y");
        for v in &y {
            assert!(v.is_finite(), "max_iter=1 must produce finite output; got {v}");
        }
    }

    #[test]
    fn test_robust_zero_variance_x_produces_nan_line() {
        let xs: Vec<f64> = vec![3.0; 10];
        let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(15, None, 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        let y = col(&out, "y");
        assert_eq!(out.num_rows(), 15);
        assert!(y.iter().all(|v| v.is_nan()), "zero-variance x must give NaN line; got {y:?}");
    }

    #[test]
    fn test_robust_empty_after_filter_produces_nan() {
        // All-null y after filter → empty.
        let xs: Vec<f64> = vec![1.0, 2.0, 3.0];
        let ys: Vec<f64> = vec![f64::NAN, f64::NAN, f64::NAN];
        let batch = xy_batch(xs, ys);
        let s = spec(8, None, 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 8);
        let y = col(&out, "y");
        assert!(y.iter().all(|v| v.is_nan()), "empty input must give NaN line");
    }

    #[test]
    fn test_robust_n_grid_size() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(50, None, 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 50);
    }

    // ── Batch-F F-L10-04: Robust groupby ─────────────────────────────────

    #[test]
    fn test_robust_grouped_refit_differs_per_group() {
        // Group "a": slope 2; group "b": slope -3. A grouped fit must
        // recover each group's own line independently rather than pooling
        // across groups, and the output schema gains the "g" column.
        let n_per_group = 20;
        let xs_a: Vec<f64> = (0..n_per_group).map(|i| i as f64).collect();
        let ys_a: Vec<f64> = xs_a.iter().map(|&x| 2.0 * x + 1.0).collect();
        let xs_b: Vec<f64> = (0..n_per_group).map(|i| i as f64).collect();
        let ys_b: Vec<f64> = xs_b.iter().map(|&x| -3.0 * x + 5.0).collect();

        let mut xs = xs_a;
        xs.extend(xs_b);
        let mut ys = ys_a;
        ys.extend(ys_b);
        let groups: Vec<&str> = std::iter::repeat("a")
            .take(n_per_group)
            .chain(std::iter::repeat("b").take(n_per_group))
            .collect();

        let batch = xy_batch_with_group(xs, ys, groups);
        let mut s = spec(10, None, 1.345, SmoothOutput::Fitted);
        s.groupby = vec!["g".into()];
        let out = apply(&s, &batch).unwrap();

        let schema = out.schema();
        assert_eq!(schema.fields().len(), 5, "core 4 columns + 1 groupby column");
        assert_eq!(schema.field(4).name(), "g");
        assert_eq!(out.num_rows(), 20, "10 grid points × 2 groups");

        let g_col = out.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let x_col = col(&out, "x");
        let y_col = col(&out, "y");
        for i in 0..out.num_rows() {
            let (x, y) = (x_col[i], y_col[i]);
            match g_col.value(i) {
                "a" => assert!((y - (2.0 * x + 1.0)).abs() < 1e-6,
                    "group a fit should match y=2x+1 at x={x}; got y={y}"),
                "b" => assert!((y - (-3.0 * x + 5.0)).abs() < 1e-6,
                    "group b fit should match y=-3x+5 at x={x}; got y={y}"),
                other => panic!("unexpected group key {other:?}"),
            }
        }
    }

    #[test]
    fn test_robust_grouped_key_dtype_preserved() {
        // Int64 groupby column must round-trip as Int64 in the output
        // (XFORM-03), not get widened to a string — mirrors Smooth's
        // groupby dtype-preservation contract.
        use arrow::array::Int64Array;
        let n_per_group = 15;
        let xs: Vec<f64> = (0..2 * n_per_group).map(|i| (i % n_per_group) as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| if i < n_per_group { 2.0 * x } else { -1.0 * x + 10.0 })
            .collect();
        let groups: Vec<i64> = (0..2 * n_per_group)
            .map(|i| if i < n_per_group { 1_i64 } else { 2_i64 })
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(Int64Array::from(groups)),
        ]).unwrap();

        let mut s = spec(5, None, 1.345, SmoothOutput::Fitted);
        s.groupby = vec!["g".into()];
        let out = apply(&s, &batch).unwrap();

        let out_schema = out.schema();
        assert_eq!(out_schema.field(4).name(), "g");
        assert_eq!(out_schema.field(4).data_type(), &DataType::Int64,
            "groupby column dtype must be preserved as Int64, not widened to a string");

        let g_col = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        let distinct: std::collections::BTreeSet<i64> =
            (0..g_col.len()).map(|i| g_col.value(i)).collect();
        assert_eq!(distinct, [1_i64, 2].into_iter().collect());
    }

    /// Batch-F T2h: grouping by a column whose name collides with an output
    /// column name (`ci_lower`) must refuse with a typed error naming the
    /// collision, not silently append a second `ci_lower` field that
    /// `Schema::index_of` would resolve to the wrong one downstream. Mirrors
    /// Smooth's identical collision-guard contract, since both transforms
    /// share `group_key::stack_grouped_batches`.
    #[test]
    fn test_robust_grouped_name_collision_with_output_column_refuses() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let group_vals: Vec<&str> = vec!["a", "a", "a", "b", "b", "b"];
        // The groupby column is literally named "ci_lower" — one of the
        // Fitted output's own field names — while the real x/y data live in
        // "x_in"/"y_in".
        let schema = Arc::new(Schema::new(vec![
            Field::new("x_in", DataType::Float64, true),
            Field::new("y_in", DataType::Float64, true),
            Field::new("ci_lower", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(group_vals)),
            ],
        )
        .unwrap();
        let mut s = spec(4, None, 1.345, SmoothOutput::Fitted);
        s.x = "x_in".into();
        s.y = "y_in".into();
        s.groupby = vec!["ci_lower".into()];
        let err = apply(&s, &batch).expect_err(
            "groupby column colliding with an output column name must refuse",
        );
        let msg = err.to_string();
        assert!(msg.contains("ci_lower"), "error must name the colliding column: {msg}");
        assert!(msg.contains("collides"), "error must explain the collision: {msg}");
    }

    /// Task 2h review finding 2, reproduced end-to-end for Robust:
    /// `groupby=['g', 'g']` must refuse rather than silently emit two
    /// identically-named 'g' fields (the exact live repro the review used:
    /// `Robust(x='x_in', y='y_in', groupby=['g','g'])`).
    #[test]
    fn test_robust_grouped_duplicate_groupby_name_refuses() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let group_vals: Vec<&str> = vec!["a", "a", "a", "b", "b", "b"];
        let schema = Arc::new(Schema::new(vec![
            Field::new("x_in", DataType::Float64, true),
            Field::new("y_in", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(group_vals)),
            ],
        )
        .unwrap();
        let mut s = spec(4, None, 1.345, SmoothOutput::Fitted);
        s.x = "x_in".into();
        s.y = "y_in".into();
        s.groupby = vec!["g".into(), "g".into()];
        let err = apply(&s, &batch)
            .expect_err("groupby=['g', 'g'] must refuse rather than emit two 'g' fields");
        let msg = err.to_string();
        assert!(msg.contains('g'), "error must name the repeated column: {msg}");
        assert!(msg.contains("repeated"), "error must explain the duplication: {msg}");
    }

    #[test]
    fn test_robust_grouped_residuals_output() {
        // The grouped path must work for output="residuals" too (cross-task
        // note: BOTH output modes), mirroring Smooth's grouped-residuals
        // contract. The response column "y" rides along as a carried
        // column, and the groupby column "g" is appended once by the
        // stacking template, not carried a second time.
        let n_per_group = 10;
        let xs_a: Vec<f64> = (0..n_per_group).map(|i| i as f64).collect();
        let ys_a: Vec<f64> = xs_a.iter().map(|&x| 2.0 * x + 1.0).collect();
        let xs_b: Vec<f64> = (0..n_per_group).map(|i| i as f64).collect();
        let ys_b: Vec<f64> = xs_b.iter().map(|&x| -1.0 * x).collect();

        let mut xs = xs_a;
        xs.extend(xs_b);
        let mut ys = ys_a;
        ys.extend(ys_b);
        let groups: Vec<&str> = std::iter::repeat("a")
            .take(n_per_group)
            .chain(std::iter::repeat("b").take(n_per_group))
            .collect();

        let batch = xy_batch_with_group(xs, ys.clone(), groups);
        let mut s = spec(10, None, 1.345, SmoothOutput::Residuals);
        s.groupby = vec!["g".into()];
        let out = apply(&s, &batch).unwrap();

        let schema = out.schema();
        assert_eq!(schema.fields().len(), 4, "residuals [x, residual] + carried y + 1 groupby column");
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "residual");
        assert_eq!(schema.field(2).name(), "y");
        assert_eq!(schema.field(3).name(), "g");
        assert_eq!(out.num_rows(), 2 * n_per_group);
        assert_eq!(col(&out, "y"), ys, "the carried y column holds every group's own response values, in order");

        // Clean per-group linear data → residuals ~0 for both groups.
        let r = col(&out, "residual");
        for &v in &r {
            assert!(v.abs() < 1e-6, "residual on clean linear per-group data should be ~0; got {v}");
        }
    }

    /// With the groups interleaved in the input, each group's emitted rows
    /// carry its OWN input row's values, in that row's own position within
    /// the group — mirrors `smooth::tests`'s identical contract (see its
    /// doc for why a contiguous fixture cannot discriminate this).
    #[test]
    fn test_robust_grouped_residuals_carry_their_own_values_under_interleaved_groups() {
        let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| if i % 2 == 0 { 2.0 * x } else { -1.0 * x }).collect();
        let id: Vec<&str> = vec!["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"];
        let g: Vec<&str> = vec!["a", "b", "a", "b", "a", "b", "a", "b"];
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("id", DataType::Utf8, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(id)),
            Arc::new(StringArray::from(g)),
        ]).unwrap();
        let mut s_spec = spec(10, None, 1.345, SmoothOutput::Residuals);
        s_spec.groupby = vec!["g".into()];
        let out = apply(&s_spec, &batch).unwrap();

        assert_eq!(out.num_rows(), 8);
        let id_out = out.column(out.schema().index_of("id").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            (0..8).map(|i| id_out.value(i)).collect::<Vec<_>>(),
            vec!["r0", "r2", "r4", "r6", "r1", "r3", "r5", "r7"],
            "group a's own rows (even input positions), then group b's own rows (odd), each in its original relative order",
        );
        let g_out = out.column(out.schema().index_of("g").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            (0..8).map(|i| g_out.value(i)).collect::<Vec<_>>(),
            vec!["a", "a", "a", "a", "b", "b", "b", "b"],
        );
    }

    // ── Residuals mode carries the caller's other input columns ─────────

    /// Every other column the caller's batch holds rides along on the
    /// residuals batch, copied at exactly the rows the fit kept (one row
    /// is dropped for a NaN `y`), in the surviving rows' own order, at
    /// every dtype — mirrors `smooth::tests`'s identical contract.
    #[test]
    fn test_robust_residuals_carries_other_input_columns_at_surviving_rows_in_order() {
        let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let w: Vec<f64> = vec![1.0, 3.0, f64::NAN, 7.0, 9.0]; // row 2 dropped (NaN y)
        let s: Vec<Option<f64>> = vec![Some(10.0), None, Some(30.0), Some(f64::NAN), Some(50.0)];
        let id: Vec<&str> = vec!["r0", "r1", "r2", "r3", "r4"];
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
            Field::new("s", DataType::Float64, true),
            Field::new("id", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(v.clone())),
            Arc::new(Float64Array::from(w.clone())),
            Arc::new(Float64Array::from(s.clone())),
            Arc::new(StringArray::from(id.clone())),
        ]).unwrap();
        let mut s_spec = spec(100, None, 1.345, SmoothOutput::Residuals);
        s_spec.x = "v".into();
        s_spec.y = "w".into();
        let out = apply(&s_spec, &batch).unwrap();

        assert_eq!(out.num_rows(), 4, "row 2 (NaN y) is dropped from the fit, and from every carried column");
        let schema = out.schema();
        let mut names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["id", "residual", "s", "v", "w", "x"]);

        assert_eq!(col(&out, "v"), vec![0.0, 1.0, 3.0, 4.0], "the predictor's own column carries too");
        assert_eq!(col(&out, "x"), vec![0.0, 1.0, 3.0, 4.0], "the computed x holds the same surviving values");
        assert_eq!(col(&out, "w"), vec![1.0, 3.0, 7.0, 9.0], "the response column rides along as a carried column");

        let s_out = out.column(out.schema().index_of("s").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(s_out.value(0), 10.0);
        assert!(s_out.is_null(1), "the caller's own null in a carried column stays null");
        assert!(!s_out.is_null(2) && s_out.value(2).is_nan(), "a NaN (never nulled) in a carried column is copied as the caller's row held it");
        assert_eq!(s_out.value(3), 50.0);

        let id_out = out.column(out.schema().index_of("id").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((0..4).map(|i| id_out.value(i)).collect::<Vec<_>>(), vec!["r0", "r1", "r3", "r4"]);
    }

    /// An input column named like one of the step's own output columns is
    /// refused rather than shadowed — the `Bin(per_row=True)` collision
    /// text, applied to `Robust`'s residual mode, with `Robust`'s own
    /// prefix, for every name the step can emit: the always-emitted
    /// `residual`, the computed `x` when the predictor is a different
    /// column (`v`), and `_ref_zero`/`_metrics_text`/`_metrics_y` only
    /// when their own inject flag asks for them.
    #[test]
    fn test_robust_residuals_refuses_an_input_column_named_like_each_column_it_emits() {
        pyo3::Python::initialize();
        let base = || {
            let mut s_spec = spec(100, None, 1.345, SmoothOutput::Residuals);
            s_spec.x = "v".into();
            s_spec.y = "w".into();
            s_spec
        };
        for (taken, s_spec) in [
            ("residual", base()),
            ("x", base()),
            ("_ref_zero", RobustSpec { inject_zero_ref: true, ..base() }),
            ("_metrics_text", RobustSpec { inject_metrics: true, ..base() }),
            ("_metrics_y", RobustSpec { inject_metrics: true, ..base() }),
        ] {
            let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
            let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
            let taken_col: Vec<f64> = vec![-1.0, -2.0, -3.0, -4.0];
            let schema = Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, true),
                Field::new("w", DataType::Float64, true),
                Field::new(taken, DataType::Float64, true),
            ]));
            let batch = RecordBatch::try_new(schema, vec![
                Arc::new(Float64Array::from(v)),
                Arc::new(Float64Array::from(w)),
                Arc::new(Float64Array::from(taken_col)),
            ]).unwrap();
            let err = apply(&s_spec, &batch).expect_err(&format!("a data column named '{taken}' must refuse"));
            assert_eq!(
                err.to_string(),
                format!(
                    "ValueError: Robust: residual output column '{taken}' collides with an \
                     input column of the same name; rename the input column"
                ),
                "{taken}",
            );
        }
    }

    /// An optional column this call does NOT emit is carried like any
    /// other column — mirrors `smooth::tests`'s identical contract.
    #[test]
    fn test_robust_residuals_carries_an_input_column_named_like_an_optional_column_it_does_not_emit() {
        for taken in ["_ref_zero", "_metrics_text", "_metrics_y"] {
            let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
            let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
            let taken_col: Vec<f64> = vec![9.0, 8.0, 7.0, 6.0];
            let schema = Arc::new(Schema::new(vec![
                Field::new("v", DataType::Float64, true),
                Field::new("w", DataType::Float64, true),
                Field::new(taken, DataType::Float64, true),
            ]));
            let batch = RecordBatch::try_new(schema, vec![
                Arc::new(Float64Array::from(v)),
                Arc::new(Float64Array::from(w)),
                Arc::new(Float64Array::from(taken_col)),
            ]).unwrap();
            let mut s_spec = spec(100, None, 1.345, SmoothOutput::Residuals);
            s_spec.x = "v".into();
            s_spec.y = "w".into();
            let out = apply(&s_spec, &batch).expect(&format!("'{taken}' is an ordinary carried column when not requested"));
            assert_eq!(col(&out, taken), vec![9.0, 8.0, 7.0, 6.0], "{taken}");
        }
    }

    /// Fewer than 2 usable points degrades to the existing zero-row
    /// residuals batch (unchanged), but its schema still carries every
    /// other input column, at its own dtype.
    #[test]
    fn test_robust_residuals_degenerate_batch_still_carries_input_schema() {
        let v: Vec<f64> = vec![1.0];
        let w: Vec<f64> = vec![2.0];
        let id: Vec<&str> = vec!["only-row"];
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
            Field::new("id", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(v)),
            Arc::new(Float64Array::from(w)),
            Arc::new(StringArray::from(id)),
        ]).unwrap();
        let mut s_spec = spec(100, None, 1.345, SmoothOutput::Residuals);
        s_spec.x = "v".into();
        s_spec.y = "w".into();
        let out = apply(&s_spec, &batch).expect("fewer than 2 points degrades, not fails");
        assert_eq!(out.num_rows(), 0, "a single usable row still can't fit a line");
        let schema = out.schema();
        let mut names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["id", "residual", "v", "w", "x"]);
        assert_eq!(
            out.schema().field(out.schema().index_of("id").unwrap()).data_type(),
            &DataType::Utf8,
            "the carried column keeps its own dtype even with no rows to carry",
        );
    }

    /// A carried (non-positional) column rides through a demanded log x
    /// unmapped: its own values are copied straight from the caller's
    /// batch, never forward/backward-mapped the way the predictor's own
    /// `x` slot is — and the predictor's own column, literally named `x`
    /// here, is still not carried a second time even though the fit now
    /// reads it through a log-mapped private column, not its own name.
    #[test]
    fn robust_residuals_on_a_demanded_log_x_carries_other_columns_unmapped() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
        let weights: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys.clone())),
            Arc::new(Float64Array::from(weights.clone())),
        ]).unwrap();
        let s = spec(6, None, 1.345, SmoothOutput::Residuals);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s, &batch, &ctx).expect("a clean log-x fit emits residuals");
        assert_eq!(out.num_rows(), 6, "every row survives (no nulls to drop)");
        assert_eq!(col(&out, "w"), weights, "the carried column is copied straight from the caller, never log-mapped");

        let schema = out.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names, vec!["x", "residual", "y", "w"],
            "the caller's own x column is not carried a second time even under a demanded log x",
        );
    }

    /// When the predictor is NOT named `x` (it is `v` here), a demanded
    /// log x maps only a fresh, private column for the fit itself —
    /// mirrors `smooth::tests`'s identical contract.
    #[test]
    fn robust_residuals_on_a_demanded_log_x_keeps_a_predictor_not_named_x_in_data_units() {
        pyo3::Python::initialize();
        let v: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let w: Vec<f64> = v.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(v.clone())),
            Arc::new(Float64Array::from(w)),
        ]).unwrap();
        let mut s_spec = spec(6, None, 1.345, SmoothOutput::Residuals);
        s_spec.x = "v".into();
        s_spec.y = "w".into();
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s_spec, &batch, &ctx).expect("a clean log-x fit emits residuals");

        let schema = out.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(
            names.iter().all(|n| !n.starts_with("__log_space_")),
            "no private log-space column leaks into the output: {names:?}",
        );
        assert_eq!(names, vec!["x", "residual", "v", "w"], "v carries under its own name, not folded into x");
        assert_eq!(col(&out, "v"), v, "the carried predictor is the caller's own data-space values, never log-mapped");
        let xg = col(&out, "x");
        for (xq, vq) in xg.iter().zip(v.iter()) {
            assert!(
                (xq - vq).abs() < 1e-9,
                "the computed x, mapped back, must equal the predictor's own data-space value: x={xq} v={vq}",
            );
        }
    }

    // ── The internal residual_columns naming setting ─────────────────────

    /// Setting every computed column's name moves them all out of the
    /// way — mirrors `smooth::tests`'s identical contract, with `Robust`'s
    /// own prefix.
    #[test]
    fn test_robust_residuals_emits_its_columns_under_the_names_its_setting_gives() {
        let v: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let w: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
        let x_col: Vec<f64> = vec![100.0, 101.0, 102.0, 103.0];
        let residual_col: Vec<f64> = vec![200.0, 201.0, 202.0, 203.0];
        let ref_zero_col: Vec<f64> = vec![300.0, 301.0, 302.0, 303.0];
        let metrics_text_col: Vec<f64> = vec![400.0, 401.0, 402.0, 403.0];
        let metrics_y_col: Vec<f64> = vec![500.0, 501.0, 502.0, 503.0];
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
            Field::new("x", DataType::Float64, true),
            Field::new("residual", DataType::Float64, true),
            Field::new("_ref_zero", DataType::Float64, true),
            Field::new("_metrics_text", DataType::Float64, true),
            Field::new("_metrics_y", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(v)),
            Arc::new(Float64Array::from(w)),
            Arc::new(Float64Array::from(x_col.clone())),
            Arc::new(Float64Array::from(residual_col.clone())),
            Arc::new(Float64Array::from(ref_zero_col.clone())),
            Arc::new(Float64Array::from(metrics_text_col.clone())),
            Arc::new(Float64Array::from(metrics_y_col.clone())),
        ]).unwrap();
        let mut s_spec = spec(100, None, 1.345, SmoothOutput::Residuals);
        s_spec.x = "v".into();
        s_spec.y = "w".into();
        s_spec.inject_zero_ref = true;
        s_spec.inject_metrics = true;
        s_spec.residual_columns = ResidualColumns {
            x: "x_1".into(), residual: "residual_1".into(), ref_zero: "_ref_zero_1".into(),
            metrics_text: "_metrics_text_1".into(), metrics_y: "_metrics_y_1".into(),
        };
        let out = apply(&s_spec, &batch).unwrap();

        let schema = out.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec!["x_1", "residual_1", "_ref_zero_1", "_metrics_text_1", "_metrics_y_1", "v", "w", "x", "residual", "_ref_zero", "_metrics_text", "_metrics_y"],
            "the computed columns under their set names, then every input column in input order",
        );
        assert_eq!(col(&out, "x"), x_col, "the data's own 'x' column, not the computed output, carries verbatim");
        assert_eq!(col(&out, "residual"), residual_col);
        assert_eq!(col(&out, "_ref_zero"), ref_zero_col);
        assert_eq!(col(&out, "_metrics_text"), metrics_text_col);
        assert_eq!(col(&out, "_metrics_y"), metrics_y_col);
    }

    /// `_with_residual_columns` only reaches a non-Robust spec through a
    /// `PyRobust` built directly in Rust, never through the public
    /// `fm.Robust` door (whose one constructor, `PyRobust::new`, only ever
    /// builds `TransformSpec::Robust`) — still a typed error, not a panic,
    /// at that PyO3 boundary.
    #[test]
    fn test_with_residual_columns_on_a_non_robust_spec_is_a_typed_error_not_a_panic() {
        pyo3::Python::initialize();
        let wrong = PyRobust(TransformSpec::Identity(crate::transform::identity::IdentitySpec {
            name: None,
            statistics: false,
        }));
        let err = wrong._with_residual_columns(None, None, None, None, None)
            .expect_err("a non-Robust spec must be refused, not panic");
        assert!(err.to_string().contains("Robust"), "{err}");
    }

    #[test]
    fn test_robust_empty_groupby_byte_identical_to_ungrouped() {
        // A groupby-shaped column can be present in the input batch without
        // being requested (`spec.groupby == []`): the fit must ignore it
        // entirely, producing the exact same canonical
        // `[x, y, ci_lower, ci_upper]` schema and fitted values as if the
        // column didn't exist — pins the "empty groupby is byte-identical to
        // today" contract (batch-F task 2 intent).
        // Alternating +/-0.5 noise (as in `test_robust_ci_brackets_fit`) keeps
        // the MAD residual scale > 0, so `ci_lower`/`ci_upper` are finite
        // rather than the degenerate exact-fit NaN band. A perfectly linear
        // fixture would drive `s == 0` and emit NaN CI bounds instead — and
        // `assert_eq!` on `Vec<f64>` FAILS when the compared vectors carry
        // NaN at matching positions (NaN != NaN), so a degenerate fixture
        // would make this assertion spuriously FAIL, not vacuously pass. The
        // noisy fixture is required for the CI assertions to mean anything.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 2.0 * x + 1.0 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let plain_batch = xy_batch(xs.clone(), ys.clone());
        let groups: Vec<&str> = (0..20).map(|i| if i % 2 == 0 { "a" } else { "b" }).collect();
        let batch_with_unused_col = xy_batch_with_group(xs, ys, groups);

        let s = spec(20, Some(0.95), 1.345, SmoothOutput::Fitted); // groupby: vec![]
        let out_plain = apply(&s, &plain_batch).unwrap();
        let out_with_unused_col = apply(&s, &batch_with_unused_col).unwrap();

        assert_eq!(
            out_plain.schema(), out_with_unused_col.schema(),
            "an unrequested groupby column in the input batch must not change the output schema"
        );
        assert_eq!(col(&out_plain, "x"), col(&out_with_unused_col, "x"));
        assert_eq!(col(&out_plain, "y"), col(&out_with_unused_col, "y"));
        assert_eq!(col(&out_plain, "ci_lower"), col(&out_with_unused_col, "ci_lower"));
        assert_eq!(col(&out_plain, "ci_upper"), col(&out_with_unused_col, "ci_upper"));
    }

    // ── #162: degrade-warning channel ────────────────────────────────────

    use crate::transform::context::{TransformContext, TransformWarning};

    /// RED (pre-#162): this assertion fails on the pre-change code, since
    /// `Robust::apply` had no way to report a degenerate fit at all.
    #[test]
    fn robust_below_threshold_pushes_a_degenerate_fit_warning() {
        let batch = xy_batch(vec![1.0], vec![1.0]);
        let s = spec(8, None, 1.345, SmoothOutput::Fitted);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 8, "output is unchanged: still the NaN-filled grid");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            &warnings[0],
            TransformWarning::DegenerateFit { transform, group, .. }
                if transform == "Robust" && group.is_none()
        ));
    }

    /// Zero-variance x is the second degenerate site (`x_max > x_min` fails)
    /// and must warn too, distinctly from the below-threshold case.
    #[test]
    fn robust_zero_variance_x_pushes_a_degenerate_fit_warning() {
        let batch = xy_batch(vec![3.0; 10], (0..10).map(|i| i as f64).collect());
        let s = spec(15, None, 1.345, SmoothOutput::Fitted);
        let ctx = TransformContext::default();
        let _ = apply_with_context(&s, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            &warnings[0],
            TransformWarning::DegenerateFit { transform, reason, .. }
                if transform == "Robust" && reason.contains("zero x variance")
        ));
    }

    /// A grouped call attributes the warning to the specific degenerate
    /// group, not the whole batch — the healthy group must NOT warn.
    #[test]
    fn robust_grouped_degenerate_group_warning_names_the_group() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0];
        let ys: Vec<f64> = vec![0.0, 2.0, 4.0, 6.0, 8.0, 1.0];
        let groups: Vec<&str> = vec!["healthy", "healthy", "healthy", "healthy", "healthy", "sparse"];
        let batch = xy_batch_with_group(xs, ys, groups);
        let mut s = spec(5, None, 1.345, SmoothOutput::Fitted);
        s.groupby = vec!["g".into()];
        let ctx = TransformContext::default();
        let _ = apply_with_context(&s, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "only the sparse group should warn; got {warnings:?}");
        assert!(matches!(
            &warnings[0],
            TransformWarning::DegenerateFit { transform, group, .. }
                if transform == "Robust" && group.as_deref() == Some("g=sparse")
        ));
    }

    /// Bare `apply()` (no context) must not panic — the ephemeral default
    /// sink absorbs the warning and is simply dropped, per
    /// `TransformContext::warnings`'s doc.
    #[test]
    fn robust_bare_apply_with_degenerate_input_does_not_panic() {
        let batch = xy_batch(vec![1.0], vec![1.0]);
        let s = spec(8, None, 1.345, SmoothOutput::Fitted);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 8);
    }

    /// GH #197 EXEMPTION: an all-null `groupby` column leaves this transform
    /// with nothing to partition, and its documented answer is the pooled
    /// whole-batch fit rather than an empty output — the degraded-but-
    /// meaningful render. The fit's own columns equal the ungrouped fit's;
    /// what is new is that the output ALSO carries its `groupby` column, all
    /// null at its original dtype — the pooled fit IS the null-keyed group —
    /// and that it SAYS so, with its own disposition text rather than a
    /// claim of emptiness.
    #[test]
    fn robust_all_null_group_keys_pool_the_whole_batch_and_warn() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(x.clone())),
                Arc::new(Float64Array::from(y.clone())),
                Arc::new(StringArray::from(vec![None, None, None, None] as Vec<Option<&str>>)),
            ],
        )
        .unwrap();
        let s = RobustSpec { groupby: vec!["g".into()], ..spec(10, None, 1.345, SmoothOutput::Fitted) };
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 10, "the pooled whole-batch fit still renders");

        let g = out.column(out.schema().index_of("g").unwrap());
        assert_eq!(g.data_type(), &DataType::Utf8, "the groupby column keeps its original dtype");
        assert_eq!(g.null_count(), g.len(), "the pooled fit IS the null-keyed group");

        let ungrouped = apply(&spec(10, None, 1.345, SmoothOutput::Fitted), &xy_batch(x, y)).unwrap();
        for field in ungrouped.schema().fields() {
            assert_eq!(
                out.column_by_name(field.name()),
                ungrouped.column_by_name(field.name()),
                "the pooled fit's own column '{}' equals the ungrouped fit's",
                field.name()
            );
        }

        let warnings = ctx.warnings.borrow();
        let all_null: Vec<&TransformWarning> = warnings
            .iter()
            .filter(|w| matches!(w, TransformWarning::GroupKeyAllNull { .. }))
            .collect();
        assert_eq!(all_null.len(), 1, "got {warnings:?}");
        assert!(
            all_null[0]
                .to_string()
                .contains("the whole batch was used as one ungrouped computation instead"),
            "the warning states this transform's OWN disposition: {}",
            all_null[0]
        );
    }

    // ── An undefined confidence band warns ────────────────────────────────

    /// A singular `X'X`: `x_range` overrides the degenerate-x guard's
    /// own extent (it reads zero-variance from `x_range`'s span, not the
    /// data's), so a batch whose real `x` is constant reaches the sandwich
    /// covariance's own singular-design check instead of the earlier
    /// whole-fit `DegenerateFit` guard. The line (a flat OLS/IRLS fit
    /// through the constant x) still draws; only the band is undefined.
    #[test]
    fn robust_singular_design_warns_interval_undefined() {
        let batch = xy_batch(vec![5.0, 5.0, 5.0, 5.0], vec![1.0, 2.0, 3.0, 4.0]);
        let mut s = spec(5, Some(0.95), 1.345, SmoothOutput::Fitted);
        s.x_range = Some([0.0, 10.0]);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).expect("a constant x still fits a flat line");
        assert!(!col(&out, "y").iter().all(|v| v.is_nan()), "the line must draw");
        assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()), "the band stays undefined");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(
                &warnings[0],
                TransformWarning::IntervalUndefined { transform, group, ci, reason, .. }
                    if transform == "Robust" && group.is_none() && *ci == 0.95 && reason.contains("singular")
            ),
            "{warnings:?}"
        );
    }

    /// A zero `Σψ'(u)`: an unreasonably tiny `huber_c` (built
    /// directly, bypassing the PyO3 constructor's `> 0` validation, as a
    /// constructed fixture) makes every finite residual exceed the Huber
    /// threshold, so no row contributes to the sandwich covariance's
    /// scalar. The line still draws; only the band is undefined.
    #[test]
    fn robust_zero_sum_psi_prime_warns_interval_undefined() {
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        // A small alternating offset keeps every residual nonzero (so the
        // MAD scale is nonzero and the exact-fit early return never fires)
        // without needing a real outlier -- `huber_c=1e-12` alone is what
        // forces every one of these ordinary residuals past the threshold.
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 2.0 * x + 1.0 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let batch = xy_batch(xs, ys);
        let s = spec(5, Some(0.95), 1e-12, SmoothOutput::Fitted);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).expect("an exact line still fits");
        assert!(!col(&out, "y").iter().all(|v| v.is_nan()), "the line must draw");
        assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()), "the band stays undefined");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(
                &warnings[0],
                TransformWarning::IntervalUndefined { transform, reason, .. }
                    if transform == "Robust" && reason.contains("Huber")
            ),
            "{warnings:?}"
        );
    }

    /// `n <= 2` leaves zero residual degrees of freedom for the 2-parameter
    /// fit — the same shape `lm` reports, worded identically
    /// (`zero_residual_dof_reason`). A 2-point fit is always exact, so this
    /// is the only way `n <= 2` reaches the band-undefined report at all.
    #[test]
    fn robust_n_eq_2_warns_zero_residual_dof() {
        let batch = xy_batch(vec![0.0, 1.0], vec![0.0, 2.0]);
        let s = spec(5, Some(0.95), 1.345, SmoothOutput::Fitted);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).expect("a 2-point line always fits exactly");
        assert!(!col(&out, "y").iter().all(|v| v.is_nan()), "the line must draw");
        assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()), "the band stays undefined");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(
                &warnings[0],
                TransformWarning::IntervalUndefined { transform, group, ci, reason, .. }
                    if transform == "Robust" && group.is_none() && *ci == 0.95
                       && reason == "2 points leave no residual degrees of freedom for a 2-parameter fit; the band needs at least 3"
            ),
            "{warnings:?}"
        );
    }

    /// A zero-MAD exact fit (`n > 2`, every point exactly on the fitted
    /// line) gives the robust scale no width to report a band with — a
    /// genuine zero-MAD degeneracy, distinct from `n <= 2`'s residual-dof
    /// shape above.
    #[test]
    fn robust_exact_fit_with_n_gt_2_warns_zero_mad() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(5, Some(0.95), 1.345, SmoothOutput::Fitted);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).expect("an exact line always fits exactly");
        assert!(!col(&out, "y").iter().all(|v| v.is_nan()), "the line must draw");
        assert!(col(&out, "ci_lower").iter().all(|v| v.is_nan()), "the band stays undefined");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(
                &warnings[0],
                TransformWarning::IntervalUndefined { transform, group, ci, reason, .. }
                    if transform == "Robust" && group.is_none() && *ci == 0.95
                       && reason == "the residuals' median absolute deviation is 0 (at least half the points lie on the line), so the robust scale gives the band no width"
            ),
            "{warnings:?}"
        );
    }

    /// `ci=None` on either degenerate-band shape above stays silent — the
    /// band was never requested, so there is nothing to report undefined.
    #[test]
    fn robust_degenerate_band_shapes_stay_silent_without_ci() {
        for (xs, ys) in [
            (vec![0.0, 1.0], vec![0.0, 2.0]),
            (vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 3.0, 5.0, 7.0]),
        ] {
            let batch = xy_batch(xs, ys);
            let s = spec(5, None, 1.345, SmoothOutput::Fitted);
            let ctx = TransformContext::default();
            let out = apply_with_context(&s, &batch, &ctx).expect("fits");
            assert!(!col(&out, "y").iter().all(|v| v.is_nan()), "the line must draw");
            assert!(ctx.warnings.borrow().is_empty(), "got {:?}", ctx.warnings.borrow());
        }
    }

    /// A negative or non-finite sandwich variance at only SOME grid
    /// points — a numerical artifact of `x`'s spread being tiny next to
    /// its magnitude, not a lost fit — warns with the point count, rather
    /// than leaving holes in the band silently. Reached through the public
    /// door with a real (not hand-built) fixture: `x` centered on a large
    /// constant (`c = 1e9`) with a small spread (`0.1`), enough points
    /// (`n = 12`) that the design is not singular (the earlier branch),
    /// but the per-grid-point quadratic form still cancels at a few of the
    /// 50 grid points.
    #[test]
    fn robust_negative_sandwich_variance_at_some_grid_points_warns_partial() {
        let c = 1_000_000_000.0;
        let spread = 0.1;
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| c + (i as f64 - (n as f64 - 1.0) / 2.0) * spread).collect();
        let ys: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| 2.0 * (x - c) + 1.0 + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let batch = xy_batch(xs, ys);
        let s = spec(50, Some(0.95), 1.345, SmoothOutput::Fitted);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).expect("fits despite the extreme offset");
        assert!(!col(&out, "y").iter().all(|v| v.is_nan()), "the line must draw");
        let ci_lower = col(&out, "ci_lower");
        assert!(ci_lower.iter().any(|v| v.is_nan()), "some grid points must stay undefined");
        assert!(ci_lower.iter().any(|v| !v.is_nan()), "but not every grid point");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        match &warnings[0] {
            TransformWarning::IntervalUndefined { transform, group, ci, reason, .. } => {
                assert_eq!(transform, "Robust");
                assert!(group.is_none());
                assert_eq!(*ci, 0.95);
                assert!(reason.contains("the sandwich variance is negative"), "{reason}");
                assert!(reason.contains("of 50 grid points"), "{reason}");
            }
            other => panic!("expected IntervalUndefined, got {other:?}"),
        }
    }

    /// The same negative-sandwich-variance shape as the ungrouped test
    /// above, reached once per `groupby` partition: each group's own
    /// warning names that group rather than leaving `group` unset, so a
    /// grouped chart's facet or legend can tell which group's band is
    /// undefined.
    #[test]
    fn robust_grouped_negative_sandwich_variance_warns_naming_each_group() {
        let c = 1_000_000_000.0;
        let spread = 0.1;
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| c + (i as f64 - (n as f64 - 1.0) / 2.0) * spread).collect();
        let ys: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| 2.0 * (x - c) + 1.0 + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let mut all_xs = Vec::new();
        let mut all_ys = Vec::new();
        let mut all_g = Vec::new();
        for group in ["a", "b"] {
            all_xs.extend(xs.iter().copied());
            all_ys.extend(ys.iter().copied());
            all_g.extend(std::iter::repeat(group).take(xs.len()));
        }
        let batch = xy_batch_with_group(all_xs, all_ys, all_g);
        let mut s = spec(50, Some(0.95), 1.345, SmoothOutput::Fitted);
        s.groupby = vec!["g".into()];
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).expect("fits despite the extreme offset");
        assert!(col(&out, "y").iter().all(|v| v.is_finite()), "both groups' lines must draw");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 2, "one warning per group, got {warnings:?}");
        let mut seen_groups: Vec<String> = Vec::new();
        for w in warnings.iter() {
            match w {
                TransformWarning::IntervalUndefined { transform, group, ci, reason, .. } => {
                    assert_eq!(transform, "Robust");
                    let label = group.clone().expect("a grouped band warning must name its group");
                    seen_groups.push(label);
                    assert_eq!(*ci, 0.95);
                    assert!(reason.contains("the sandwich variance is negative"), "{reason}");
                    assert!(reason.contains("of 50 grid points"), "{reason}");
                }
                other => panic!("expected IntervalUndefined, got {other:?}"),
            }
        }
        seen_groups.sort();
        assert_eq!(seen_groups, vec!["g=a".to_string(), "g=b".to_string()]);
    }

    // ── A drawn log axis demands this fit in its own space ───────────────

    type LogSpace = crate::transform::color_bin_group::LogAxisSpace;

    fn ctx_demanding(space: StepSpace) -> TransformContext {
        TransformContext { log_space: Some(std::rc::Rc::new(space)), ..TransformContext::default() }
    }

    /// The `x` slot demanded along `x` in `x_space`, the `y` slot along `y`
    /// in `y_space`, whichever is `Some` — mirrors `smooth::tests`' and
    /// `kde_2d::tests`' identical helper.
    fn axis_slots(x_space: Option<LogSpace>, y_space: Option<LogSpace>) -> StepSpace {
        use crate::spec::layer::PositionalAxis;
        StepSpace::for_slots(
            [(SpaceSlot::X, PositionalAxis::X, x_space), (SpaceSlot::Y, PositionalAxis::Y, y_space)]
                .into_iter()
                .filter_map(|(slot, axis, space)| space.map(|space| (slot, SlotSpace { space, axes: vec![axis], records_caller_input: true }))),
        )
    }

    fn positive_space() -> LogSpace {
        LogSpace { base: 10.0, side: crate::scale::log::LogSide::Positive }
    }

    /// Robust's Huber fit, drawn on a demanded log x, fits a straight line
    /// in log(x): data generated as `y = a + b*log10(x)` has no outliers to
    /// down-weight, so Huber IRLS converges to (close to) the exact OLS
    /// line in log units, and the mapped-back grid stays strictly positive.
    #[test]
    fn robust_on_a_demanded_log_x_fits_a_straight_line_in_log_units() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 3.0 + 2.0 * x.log10()).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(6, None, 1.345, SmoothOutput::Fitted);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        let grid = col(&out, "x");
        let fitted = col(&out, "y");
        assert!(grid.iter().all(|v| v.is_finite() && *v > 0.0), "{grid:?}");
        for (&x, &y) in grid.iter().zip(fitted.iter()) {
            let expected = 3.0 + 2.0 * x.log10();
            assert!((y - expected).abs() < 1e-6, "x={x} y={y} expected={expected}");
        }
        let recorded = ctx.log_input.borrow();
        let column = recorded
            .get(&(crate::spec::layer::PositionalAxis::X, "x".to_string()))
            .expect("the walk's root records the caller's column");
        assert_eq!(column.len(), 6, "the recorded column is the caller's whole input");
    }

    /// A demanded x with off-side (`<= 0`) rows and a genuine null mixed
    /// in among on-side ones: the mapped-off-side rows never reach the
    /// fit body, so the output grid draws no off-side x, but
    /// `ctx.log_input` still records the caller's WHOLE, UNTOUCHED column
    /// — mirrors `smooth::tests`'s identical case.
    #[test]
    fn robust_on_a_demanded_x_with_off_side_and_null_rows_records_the_caller_column_untouched_and_draws_no_off_side_x() {
        pyo3::Python::initialize();
        let xs: Vec<Option<f64>> = vec![Some(-2.0), None, Some(1.0), Some(2.0), Some(4.0), Some(8.0), Some(16.0)];
        let ys: Vec<Option<f64>> = vec![
            Some(0.0),
            Some(0.0),
            Some(3.0),
            Some(3.0 + 2.0 * 2f64.log10()),
            Some(3.0 + 2.0 * 4f64.log10()),
            Some(3.0 + 2.0 * 8f64.log10()),
            Some(3.0 + 2.0 * 16f64.log10()),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs.clone())),
            Arc::new(Float64Array::from(ys)),
        ])
        .unwrap();
        let s = spec(5, None, 1.345, SmoothOutput::Fitted);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s, &batch, &ctx).expect("the on-side rows alone still fit a line");
        let grid = col(&out, "x");
        assert!(grid.iter().all(|v| v.is_finite() && *v > 0.0), "the grid must draw no off-side x: {grid:?}");
        let recorded = ctx.log_input.borrow();
        let column = recorded
            .get(&(crate::spec::layer::PositionalAxis::X, "x".to_string()))
            .expect("the demanded X slot records the caller's column");
        assert_eq!(
            column.len(),
            xs.len(),
            "the recorded column is the caller's WHOLE input length, off-side and null rows included, not filtered down to the rows the fit itself uses"
        );
        let recorded_f64 = column.as_any().downcast_ref::<Float64Array>().expect("caller x is Float64");
        assert!(recorded_f64.is_null(1), "the caller's own null stays null in the recorded copy, not silently filled in");
        assert_eq!(recorded_f64.value(0), -2.0, "the off-side row's own value is recorded verbatim, not nulled out by the log mapping that only touches the fit's own input");
    }

    /// `PyRobust::new` refuses a non-finite `x_range` endpoint, at either
    /// end, for each of NaN/+inf/-inf — a regression test for the
    /// constructor's own `x_range.is_finite()` guard: were that check
    /// deleted, every one of these six cases would construct successfully
    /// instead of refusing (`result.err()` would be `None`), and the test
    /// would panic at `.err().unwrap_or_else(..)` ("must be refused, not
    /// accepted"), never reaching the `assert_eq!` on the refusal text.
    /// Mirrors `smooth::tests`'s identical `PySmooth` case.
    #[test]
    fn pyrobust_new_refuses_a_non_finite_x_range_at_either_end() {
        pyo3::Python::initialize();
        for (a, b) in [
            (f64::NAN, 8.0),
            (8.0, f64::NAN),
            (f64::INFINITY, 8.0),
            (8.0, f64::INFINITY),
            (f64::NEG_INFINITY, 8.0),
            (8.0, f64::NEG_INFINITY),
        ] {
            let result = PyRobust::new(
                "x", "y", 100, None, 1.345, 25, 1e-8, "fitted",
                false, false, None, None, Some([a, b]), None,
            );
            let err = result.err().unwrap_or_else(|| panic!("x_range=[{a}, {b}] must be refused, not accepted"));
            assert_eq!(
                err.to_string(),
                format!("ValueError: Robust: x_range must be (lo, hi) with both finite; received [{a}, {b}]"),
                "x_range=[{a}, {b}]"
            );
        }
    }

    /// `x_range_pad` is taken AFTER `x_range` is mapped into the demanded
    /// axis's own log10 units, so the padded ends are EXACTLY
    /// `10^(-0.05*log10(16))` and `10^(1.05*log10(16))` for `x_range=[1,
    /// 16]`, `pad=0.05` — see `smooth::tests`'s identical case for the
    /// full contract.
    #[test]
    fn robust_x_range_pad_is_taken_in_the_demanded_axis_own_log_units() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let ys: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let batch = xy_batch(xs, ys);
        let mut s = spec(10, None, 1.345, SmoothOutput::Fitted);
        s.x_range = Some([1.0, 16.0]);
        s.x_range_pad = Some(0.05);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        let grid = col(&out, "x");
        let log16 = 16f64.log10();
        let expected_lo = 10f64.powf(-0.05 * log16);
        let expected_hi = 10f64.powf(1.05 * log16);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(
            close(grid.iter().cloned().reduce(f64::min).unwrap(), expected_lo),
            "expected the padded low end {expected_lo}, got {grid:?}"
        );
        assert!(
            close(grid.iter().cloned().reduce(f64::max).unwrap(), expected_hi),
            "expected the padded high end {expected_hi}, got {grid:?}"
        );
    }

    /// `x_range_pad` ALONE (no explicit `x_range`) pads the batch's own x
    /// extent in the demanded axis's own LOG units, not data units: this
    /// fixture's x spans exactly `[1, 16]`, the same span the test above
    /// passes as an EXPLICIT `x_range`, so the two tests' expected padded
    /// ends are identical — see `smooth::tests`'s identical case for the
    /// full contract.
    #[test]
    fn robust_x_range_pad_alone_pads_the_batchs_own_extent_in_the_demanded_axis_own_log_units() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let ys: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let batch = xy_batch(xs, ys);
        let mut s = spec(10, None, 1.345, SmoothOutput::Fitted);
        s.x_range = None;
        s.x_range_pad = Some(0.05);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        let grid = col(&out, "x");
        let log16 = 16f64.log10();
        let expected_lo = 10f64.powf(-0.05 * log16);
        let expected_hi = 10f64.powf(1.05 * log16);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(
            close(grid.iter().cloned().reduce(f64::min).unwrap(), expected_lo),
            "expected the padded low end {expected_lo}, got {grid:?}"
        );
        assert!(
            close(grid.iter().cloned().reduce(f64::max).unwrap(), expected_hi),
            "expected the padded high end {expected_hi}, got {grid:?}"
        );
    }

    /// A range whose DATA-SPACE pad would go off-side: `x_range=[1, 100]`
    /// padded 5% in data units would give a negative, unplaceable low end.
    /// Padded in LOG units instead, the ends are exactly `10^-0.1` and
    /// `10^2.1`, both strictly positive — see `smooth::tests`'s identical
    /// case for the full contract.
    #[test]
    fn robust_x_range_pad_in_log_units_never_goes_off_side_where_a_data_space_pad_would() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![1.0, 10.0, 100.0];
        let ys: Vec<f64> = vec![1.0, 2.0, 3.0];
        let batch = xy_batch(xs, ys);
        let mut s = spec(10, None, 1.345, SmoothOutput::Fitted);
        s.x_range = Some([1.0, 100.0]);
        s.x_range_pad = Some(0.05);
        let ctx = ctx_demanding(axis_slots(Some(positive_space()), None));
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        let grid = col(&out, "x");
        let expected_lo = 10f64.powf(-0.1);
        let expected_hi = 10f64.powf(2.1);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(grid.iter().all(|v| v.is_finite() && *v > 0.0), "every grid point must stay on-side: {grid:?}");
        assert!(
            close(grid.iter().cloned().reduce(f64::min).unwrap(), expected_lo),
            "expected the padded low end {expected_lo}, got {grid:?}"
        );
        assert!(
            close(grid.iter().cloned().reduce(f64::max).unwrap(), expected_hi),
            "expected the padded high end {expected_hi}, got {grid:?}"
        );
    }
}
