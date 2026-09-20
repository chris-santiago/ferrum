//! The fit implementations: per-method point predictors, per-method
//! grid-and-band fits, and the batch-assembly helpers shared across them.
//! Wave 6A `#163`: split out of `transform/smooth.rs` (past the CLAUDE.md
//! test-vs-source promotion trigger) rather than left inline, since it is
//! the largest cohesive block in the file — every numeric fitting routine
//! plus the `[x, y, ci_lower, ci_upper]`/`[x, residual]` batch emitters they
//! share. `mod.rs`'s `apply_one_group` calls directly into several of these
//! (the per-method dispatch, the degenerate-input batch builders), so those
//! items are `pub(super)` — visible to `smooth` and its descendants
//! (`binding`, `tests`), never beyond the module tree, mirroring
//! `render/scale_resolve`'s `pub(in crate::render)` split precedent.
//! `extract_xy`/`all_nan_output` do NOT live here — neither is a fit
//! function nor a helper other `fits.rs` functions share; both stay in
//! `mod.rs` beside their sole caller, `apply_one_group`.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use std::sync::Arc;

use crate::transform::context::{TransformContext, TransformWarning};
use crate::transform::gam;
use crate::transform::residuals;

use super::{SmoothMethod, SmoothSpec};

/// Build a point-predictor closure for the chosen smoothing method,
/// fitted on `(xs_fit, ys_fit)`. Returns NaN for any query point that
/// the underlying fit cannot evaluate (e.g. zero-variance x for OLS,
/// rank-deficient local window for LOESS). Shared between
/// `residuals_fit` and the Fitted+metrics path in `apply()` so the
/// residual computation is byte-identical across both call sites.
pub(super) fn build_predictor(spec: &SmoothSpec, xs_fit: &[f64], ys_fit: &[f64]) -> Box<dyn Fn(f64) -> f64> {
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
pub(super) fn build_residuals_batch(
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

pub(super) fn residuals_fit(
    spec: &SmoothSpec,
    xs_fit: &[f64], ys_fit: &[f64],
    xs_input: &[f64], ys_input: &[f64],
) -> PyResult<RecordBatch> {
    // Fit the chosen model on (xs_fit, ys_fit); then evaluate at xs_input and subtract.
    let predictor = build_predictor(spec, xs_fit, ys_fit);
    residuals_from_predictor(&*predictor, xs_input, ys_input, spec.inject_zero_ref, spec.inject_metrics)
}

/// The shared tail of [`residuals_fit`]/[`gam_residuals_fit`]: evaluate
/// `predictor` at each `xs_input` point, subtract from `ys_input`, and emit
/// the `[x, residual]` batch (plus the optional `_ref_zero`/metrics
/// columns). Factored out so `gam_residuals_fit` — which builds its own
/// predictor directly from `gam::fit` instead of going through
/// [`build_predictor`], so it can also report a [`GamFitOutcome`] — does not
/// need to re-duplicate this loop.
fn residuals_from_predictor(
    predictor: &dyn Fn(f64) -> f64,
    xs_input: &[f64], ys_input: &[f64],
    inject_zero_ref: bool, inject_metrics: bool,
) -> PyResult<RecordBatch> {
    let mut out_x = Vec::with_capacity(xs_input.len());
    let mut out_r = Vec::with_capacity(xs_input.len());
    for (&xi, &yi) in xs_input.iter().zip(ys_input.iter()) {
        let yhat = predictor(xi);
        let r = if yhat.is_nan() { f64::NAN } else { yi - yhat };
        out_x.push(xi);
        out_r.push(r);
    }
    let metrics = if inject_metrics {
        Some(residuals::compute(ys_input, &out_r))
    } else {
        None
    };
    build_residuals_batch(out_x, out_r, inject_zero_ref, metrics)
}

/// `SmoothOutput::Residuals` counterpart to [`gam_fit`]: fits `gam` directly
/// (rather than through [`build_predictor`]'s generic dispatch, which
/// discards whether `gam::fit` clamped `k`) so its caller can report the
/// same [`GamFitOutcome`] the Fitted path reports via [`report_gam_outcome`].
pub(super) fn gam_residuals_fit(
    spec: &SmoothSpec,
    xs_fit: &[f64], ys_fit: &[f64],
    xs_input: &[f64], ys_input: &[f64],
) -> PyResult<(RecordBatch, GamFitOutcome)> {
    let (predictor, outcome): (Box<dyn Fn(f64) -> f64>, GamFitOutcome) = match gam::fit(xs_fit, ys_fit, spec.k) {
        Some(fit) => {
            let outcome = GamFitOutcome::Fit { requested: spec.k, applied: fit.basis_dim() };
            (Box::new(move |x: f64| fit.predict(x)), outcome)
        }
        None => (Box::new(|_: f64| f64::NAN), GamFitOutcome::Degenerate),
    };
    let out = residuals_from_predictor(
        &*predictor, xs_input, ys_input, spec.inject_zero_ref, spec.inject_metrics,
    )?;
    Ok((out, outcome))
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
pub(super) fn build_degenerate_smooth_batch(x_col: &[f64], n_grid: usize, metrics_requested: bool) -> PyResult<RecordBatch> {
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
pub(super) fn poly_fit(
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
pub(super) fn log_x_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    transformed_linear_fit(xs, ys, grid, ci, n_grid, |x| x.ln(), metrics)
}

/// Sqrt-x linear fit: y ~ a + b·√x. Requires x ≥ 0.
pub(super) fn sqrt_x_fit(
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
pub(super) fn glm_fit(
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
/// What [`gam_fit`] found while fitting — the #162 signal its caller
/// (`apply_one_group`) turns into a `TransformWarning`. Kept as a return
/// value rather than threading `ctx`/`group` straight into `gam_fit` itself,
/// which would push its argument count past clippy's `too_many_arguments`
/// threshold for no real benefit — the caller already has both and is a
/// one-line `match` away from reporting either shape.
pub(super) enum GamFitOutcome {
    /// `gam::fit` returned `None` (fewer than 4 distinct x values, zero x
    /// variance, or no usable smoothing parameter).
    Degenerate,
    /// A fit was produced; `requested`/`applied` are `k` and the basis
    /// dimension actually used (differ exactly when `gam::fit` clamped it).
    Fit { requested: usize, applied: usize },
}

/// Turn a [`GamFitOutcome`] into the #162 warning it represents, shared
/// between the Fitted path (`apply_one_group`'s `gam_fit` call) and the
/// Residuals path (`gam_residuals_fit`, below) — both call `gam::fit` with
/// the same `k` and need the identical clamp/degenerate report.
pub(super) fn report_gam_outcome(ctx: &TransformContext, group: Option<&str>, outcome: GamFitOutcome) {
    match outcome {
        GamFitOutcome::Degenerate => ctx.warn(TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: group.map(str::to_string),
            reason: "fewer than 4 distinct x values, zero x variance, or no usable \
                     smoothing parameter for a gam fit"
                .into(),
        }),
        GamFitOutcome::Fit { requested, applied } if requested != applied => {
            // #162's named case: `k` is clamped down to the number of
            // distinct x values (`gam.rs`'s `fit`,
            // `k.clamp(MIN_BASIS_DIM, distinct.len())`).
            ctx.warn(TransformWarning::ParameterClamped {
                transform: "Smooth".into(),
                group: group.map(str::to_string),
                param: "k".into(),
                requested: requested.to_string(),
                applied: applied.to_string(),
            });
        }
        GamFitOutcome::Fit { .. } => {}
    }
}

pub(super) fn gam_fit(
    xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize, k: usize,
    metrics: Option<(f64, f64, f64)>,
) -> PyResult<(RecordBatch, GamFitOutcome)> {
    let fit = match gam::fit(xs, ys, k) {
        Some(f) => f,
        None => {
            let out = build_degenerate_smooth_batch(grid, n_grid, metrics.is_some())?;
            return Ok((out, GamFitOutcome::Degenerate));
        }
    };
    let outcome = GamFitOutcome::Fit { requested: k, applied: fit.basis_dim() };

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

    let out = build_smooth_batch(grid.to_vec(), y_fit, lo, hi, metrics)?;
    Ok((out, outcome))
}

pub(super) fn lm_fit(
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
pub(super) fn student_t_critical(level: f64, dof: f64) -> f64 {
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

pub(super) fn loess_fit(
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

