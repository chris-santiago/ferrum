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

use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, parse_groupby_pyany, KeyValue,
};
use crate::transform::logistic::inv_normal_cdf;
use crate::transform::numeric_util::coerce_xy_pairs_at;
use crate::transform::residuals;
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
}

// ---------- Main apply ----------

/// Entry point: dispatches to the grouped or ungrouped fit depending on
/// `spec.groupby`. Mirrors `smooth::apply_with_context`'s dispatch shape,
/// minus the `TransformContext` threading — Robust has no context-dependent
/// whole-frame refusal (Smooth's rank-deficient-`degree` case), so it needs
/// no `is_partition` signal and stays reachable through the plain `apply()`
/// path (`TransformSpec::apply_with_context`'s default fallback).
pub(crate) fn apply(spec: &RobustSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby);
    }
    apply_one_group(spec, batch, None)
}

/// Fit one group's (or the whole frame's) rows. `only_indices` restricts the
/// x/y extraction to those row positions when set (a groupby partition);
/// `None` scans every row, byte-identical to this function's pre-groupby body.
fn apply_one_group(
    spec: &RobustSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
) -> PyResult<RecordBatch> {
    let (xs, ys) = coerce_xy_pairs_at(batch, &spec.x, &spec.y, "Robust", only_indices)?;

    let n_grid = spec.n_grid.max(1);

    if xs.len() < 2 {
        return match spec.output {
            SmoothOutput::Fitted => build_nan_fitted(n_grid, spec.inject_metrics),
            SmoothOutput::Residuals => build_residuals_batch(
                Vec::new(), Vec::new(), &[],
                spec.inject_zero_ref, spec.inject_metrics,
            ),
        };
    }

    let (data_x_min, data_x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    // WI-7: use x_range when set (truncate=False); else fall back to data extent.
    let (x_min, x_max) = match spec.x_range {
        Some([lo, hi]) if lo.is_finite() && hi.is_finite() && hi > lo => (lo, hi),
        _ => (data_x_min, data_x_max),
    };
    if !(x_max > x_min) {
        // Zero-variance x → no slope identifiable → NaN line.
        return match spec.output {
            SmoothOutput::Fitted => build_nan_fitted(n_grid, spec.inject_metrics),
            SmoothOutput::Residuals => {
                let nans = vec![f64::NAN; xs.len()];
                build_residuals_batch(
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
    if s == 0.0 {
        // Exact fit: return grid with fitted line and zero-width / NaN CI.
        let grid: Vec<f64> = grid_vec(x_min, x_max, n_grid);
        let y_fit: Vec<f64> = grid.iter().map(|&xq| alpha + beta * xq).collect();
        let nans = vec![f64::NAN; n_grid];
        return build_fitted_batch(grid, y_fit, nans.clone(), nans, metrics);
    }
    for i in 0..xs.len() {
        u_vec[i] = r_final[i] / s;
    }

    // Grid and fitted line.
    let grid: Vec<f64> = grid_vec(x_min, x_max, n_grid);
    let y_fit: Vec<f64> = grid.iter().map(|&xq| alpha + beta * xq).collect();

    // Sandwich CI.
    let (lo, hi) = match spec.ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err("Robust: ci must be in (0, 1)"));
            }
            let z = inv_normal_cdf(1.0 - (1.0 - level) / 2.0);
            // Cov(β̂) = s² · (X'X)⁻¹ · Σψ²(u) / (Σψ'(u))²
            let xtx_inv = match invert_2x2(xtx_unweighted(&xs)) {
                Some(m) => m,
                None => {
                    return build_fitted_batch(
                        grid, y_fit,
                        vec![f64::NAN; n_grid], vec![f64::NAN; n_grid],
                        metrics,
                    );
                }
            };
            let mut sum_psi_sq = 0.0_f64;
            let mut sum_psi_prime = 0.0_f64;
            for &ui in &u_vec {
                let psi = huber_psi(ui, c);
                sum_psi_sq += psi * psi;
                sum_psi_prime += if ui.abs() <= c { 1.0 } else { 0.0 };
            }
            let scalar = if sum_psi_prime > 0.0 {
                s * s * sum_psi_sq / (sum_psi_prime * sum_psi_prime)
            } else {
                f64::NAN
            };
            let cov = scale_2x2(xtx_inv, scalar);

            let mut lo = Vec::with_capacity(n_grid);
            let mut hi = Vec::with_capacity(n_grid);
            for &xq in &grid {
                let v = cov[0][0]
                    + 2.0 * xq * cov[0][1]
                    + xq * xq * cov[1][1];
                let se = if v >= 0.0 && v.is_finite() { v.sqrt() } else { f64::NAN };
                let eta = alpha + beta * xq;
                lo.push(eta - z * se);
                hi.push(eta + z * se);
            }
            (lo, hi)
        }
    };

    build_fitted_batch(grid, y_fit, lo, hi, metrics)
}

/// Partition input batch by the groupby columns, call `apply_one_group` per
/// partition, then stack the results into a single batch with the group
/// columns appended (with their original dtypes, XFORM-03) as additional
/// fields.
///
/// Mirrors `smooth::apply_grouped`'s stacking shape (batch-D T9/T3.8a):
/// first-appearance group order, null-key skip (`group_partition_multi`),
/// schema reconciliation via `concat::concat_by_name` (Robust's schema is
/// spec-determined — the same `output`/`inject_metrics` combination on every
/// group — so no group can legitimately diverge from its siblings; this is
/// belt-and-braces the same way it is for Smooth), and dtype-preserving
/// group-column re-emission via the shared `group_key` helpers.
///
/// DUPLICATION-PENDING (rust-coder, batch-F task 2): the per-group loop plus
/// the stack-and-append-group-columns block below (~30 lines) is structurally
/// identical to `smooth::apply_grouped`'s (`smooth.rs:394-460`), differing
/// only in the per-group fit call and the error-context string. `group_key.rs`
/// is outside this task's declared footprint (robust.rs, registry/mod.rs,
/// _core.pyi), so it isn't hoisted here; a follow-up should extract a shared
/// `group_key::stack_grouped_batches(context, per_group_batches, group_cols,
/// gtypes) -> PyResult<RecordBatch>` helper covering both call sites.
fn apply_grouped(
    spec: &RobustSpec,
    batch: &RecordBatch,
    group_cols: &[String],
) -> PyResult<RecordBatch> {
    let (group_order, group_idx_map, gtypes) = group_partition_multi(batch, group_cols, "Robust")?;

    let mut per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = Vec::new();
    for g in &group_order {
        let ixs = group_idx_map.get(g).ok_or_else(|| {
            PyValueError::new_err(format!("Robust: missing group key {g:?} in index map"))
        })?;
        let out = apply_one_group(spec, batch, Some(ixs))?;
        per_group_batches.push((g.clone(), out));
    }

    if per_group_batches.is_empty() {
        // Every group key was null: there is no per-group output to stack.
        // Fall back to the full batch on `None` indices, mirroring
        // `smooth::apply_grouped`'s identical fallback.
        return apply_one_group(spec, batch, None);
    }

    let total_rows: usize = per_group_batches.iter().map(|(_, b)| b.num_rows()).sum();
    let stacked = crate::transform::concat::concat_by_name(
        "Robust",
        per_group_batches.iter().map(|(_, b)| b.clone()).collect(),
    )
    .map_err(PyValueError::new_err)?;

    // Append the group columns, preserving their ORIGINAL dtypes (XFORM-03).
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(total_rows);
    for (g, b) in &per_group_batches {
        for _ in 0..b.num_rows() {
            group_keys_out.push(g.clone());
        }
    }
    let mut out_fields: Vec<Field> =
        stacked.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut out_cols: Vec<ArrayRef> = stacked.columns().to_vec();
    for (gi, (col_name, dtype)) in group_cols.iter().zip(gtypes.iter()).enumerate() {
        out_fields.push(Field::new(col_name, dtype.clone(), groupby_field_nullable()));
        let group_col_arr =
            materialize_groupby_col(&group_keys_out, gi, dtype).map_err(PyValueError::new_err)?;
        out_cols.push(group_col_arr);
    }

    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("Robust: {e}")))
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
        residuals::append_metrics_columns(&mut fields, &mut cols, &x, &y, m);
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
fn build_residuals_batch(
    xs: Vec<f64>, resid: Vec<f64>, ys_obs: &[f64],
    inject_zero_ref: bool, inject_metrics: bool,
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

    if inject_metrics {
        let metrics = residuals::compute(ys_obs, &resid);
        residuals::append_metrics_columns(&mut fields, &mut cols, &xs, &resid, metrics);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("Robust: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

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
    ) -> PyResult<Self> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Robust: x and y must be non-empty"));
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
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Robust(s) => format!(
                "Robust(x='{}', y='{}', n_grid={}, ci={:?}, huber_c={}, max_iter={}, tol={}, output={:?}, inject_zero_ref={}, inject_metrics={}, groupby={:?}, name={:?})",
                s.x, s.y, s.n_grid, s.ci, s.huber_c, s.max_iter, s.tol, s.output,
                s.inject_zero_ref, s.inject_metrics, s.groupby, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            x: "x".into(), y: "y".into(),
            n_grid, ci, huber_c,
            max_iter: 50, tol: 1e-10,
            output,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            x_range: None,
            name: None,
        }
    }

    #[test]
    fn test_robust_round_trip_json() {
        let s = RobustSpec {
            x: "x".into(), y: "y".into(),
            n_grid: 100, ci: Some(0.95),
            huber_c: 1.345, max_iter: 25, tol: 1e-8,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            name: None,
            x_range: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: RobustSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);

        // Also check residuals variant + name.
        let s2 = RobustSpec {
            x: "a".into(), y: "b".into(),
            n_grid: 50, ci: None,
            huber_c: 2.0, max_iter: 10, tol: 1e-6,
            output: SmoothOutput::Residuals,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            name: Some("r1".into()),
            x_range: None,
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
        // For perfectly linear data, all residuals are ~0.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let s = spec(100, None, 1.345, SmoothOutput::Residuals);
        let out = apply(&s, &batch).unwrap();

        // Schema check.
        let schema = out.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "residual");

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
            x: "x".into(), y: "y".into(),
            n_grid: 30, ci: Some(0.95),
            huber_c: 1.345, max_iter: 50, tol: 1e-10,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: true,
            groupby: vec![],
            x_range: None,
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
            x: "x".into(), y: "y".into(),
            n_grid: 10, ci: None,
            huber_c: 1.345, max_iter: 1, tol: 1e-8,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            groupby: vec![],
            x_range: None,
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

    #[test]
    fn test_robust_grouped_residuals_output() {
        // The grouped path must work for output="residuals" too (cross-task
        // note: BOTH output modes), mirroring Smooth's grouped-residuals
        // contract.
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

        let batch = xy_batch_with_group(xs, ys, groups);
        let mut s = spec(10, None, 1.345, SmoothOutput::Residuals);
        s.groupby = vec!["g".into()];
        let out = apply(&s, &batch).unwrap();

        let schema = out.schema();
        assert_eq!(schema.fields().len(), 3, "residuals [x, residual] + 1 groupby column");
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(schema.field(1).name(), "residual");
        assert_eq!(schema.field(2).name(), "g");
        assert_eq!(out.num_rows(), 2 * n_per_group);

        // Clean per-group linear data → residuals ~0 for both groups.
        let r = col(&out, "residual");
        for &v in &r {
            assert!(v.abs() < 1e-6, "residual on clean linear per-group data should be ~0; got {v}");
        }
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
        // rather than the degenerate exact-fit NaN band — a perfectly linear
        // fixture would make this assertion vacuously true via NaN != NaN.
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
}
