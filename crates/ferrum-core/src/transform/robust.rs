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

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::logistic::inv_normal_cdf;
use crate::transform::residuals;
use crate::transform::smooth::SmoothOutput;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

// ---------- Main apply ----------

pub(crate) fn apply(spec: &RobustSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let (xs, ys) = extract_xy(spec, batch)?;

    let n_grid = spec.n_grid.max(1);

    if xs.len() < 2 {
        return match spec.output {
            SmoothOutput::Fitted => build_nan_fitted(n_grid),
            SmoothOutput::Residuals => build_residuals_batch(
                Vec::new(), Vec::new(), &[],
                spec.inject_zero_ref, spec.inject_metrics,
            ),
        };
    }

    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if !(x_max > x_min) {
        // Zero-variance x → no slope identifiable → NaN line.
        return match spec.output {
            SmoothOutput::Fitted => build_nan_fitted(n_grid),
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

    // Refresh scale + u for CI computation.
    s = mad_scale(&r_final);
    if s == 0.0 {
        // Exact fit: return grid with fitted line and zero-width / NaN CI.
        let grid: Vec<f64> = grid_vec(x_min, x_max, n_grid);
        let y_fit: Vec<f64> = grid.iter().map(|&xq| alpha + beta * xq).collect();
        let nans = vec![f64::NAN; n_grid];
        return build_fitted_batch(grid, y_fit, nans.clone(), nans);
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

    build_fitted_batch(grid, y_fit, lo, hi)
}

// ---------- Helpers ----------

fn grid_vec(x_min: f64, x_max: f64, n_grid: usize) -> Vec<f64> {
    (0..n_grid)
        .map(|i| if n_grid <= 1 { x_min }
             else { x_min + (x_max - x_min) * (i as f64) / ((n_grid - 1) as f64) })
        .collect()
}

fn extract_xy(spec: &RobustSpec, batch: &RecordBatch) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("Robust: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("Robust: column '{}' not found", spec.y)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("Robust: '{}' must be Float64", spec.x)));
    }
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("Robust: '{}' must be Float64", spec.y)));
    }
    let xa = batch.column(xi).as_any().downcast_ref::<Float64Array>().unwrap();
    let ya = batch.column(yi).as_any().downcast_ref::<Float64Array>().unwrap();
    let mut xs = Vec::with_capacity(xa.len());
    let mut ys = Vec::with_capacity(ya.len());
    for i in 0..xa.len() {
        if xa.is_null(i) || ya.is_null(i) { continue; }
        let xv = xa.value(i); let yv = ya.value(i);
        if xv.is_nan() || yv.is_nan() { continue; }
        xs.push(xv); ys.push(yv);
    }
    Ok((xs, ys))
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

/// Unweighted X'X for X = [1, x].
fn xtx_unweighted(xs: &[f64]) -> [[f64; 2]; 2] {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    [[n, sx], [sx, sxx]]
}

fn invert_2x2(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !det.is_finite() || det.abs() < 1e-15 { return None; }
    let inv_det = 1.0 / det;
    Some([
        [ a[1][1] * inv_det, -a[0][1] * inv_det],
        [-a[1][0] * inv_det,  a[0][0] * inv_det],
    ])
}

fn scale_2x2(a: [[f64; 2]; 2], k: f64) -> [[f64; 2]; 2] {
    [[a[0][0] * k, a[0][1] * k], [a[1][0] * k, a[1][1] * k]]
}

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

fn build_fitted_batch(
    x: Vec<f64>, y: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>,
) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x",        DataType::Float64, true),
        Field::new("y",        DataType::Float64, true),
        Field::new("ci_lower", DataType::Float64, true),
        Field::new("ci_upper", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x)),
        Arc::new(Float64Array::from(y)),
        Arc::new(Float64Array::from(lo)),
        Arc::new(Float64Array::from(hi)),
    ];
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("Robust: {e}")))
}

fn build_nan_fitted(n_grid: usize) -> PyResult<RecordBatch> {
    let nans = vec![f64::NAN; n_grid];
    build_fitted_batch(nans.clone(), nans.clone(), nans.clone(), nans)
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
///     Predictor column (must be Float64).
/// y : str
///     Response column (must be Float64).
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
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
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
        name = None,
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
        name: Option<String>,
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

        if (inject_zero_ref || inject_metrics)
            && !matches!(output_parsed, SmoothOutput::Residuals)
        {
            return Err(PyValueError::new_err(
                "Robust: inject_zero_ref / inject_metrics require output='residuals'",
            ));
        }
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
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Robust(s) => format!(
                "Robust(x='{}', y='{}', n_grid={}, ci={:?}, huber_c={}, max_iter={}, tol={}, output={:?}, inject_zero_ref={}, inject_metrics={}, name={:?})",
                s.x, s.y, s.n_grid, s.ci, s.huber_c, s.max_iter, s.tol, s.output,
                s.inject_zero_ref, s.inject_metrics, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

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

    fn spec(n_grid: usize, ci: Option<f64>, huber_c: f64, output: SmoothOutput) -> RobustSpec {
        RobustSpec {
            x: "x".into(), y: "y".into(),
            n_grid, ci, huber_c,
            max_iter: 50, tol: 1e-10,
            output,
            inject_zero_ref: false,
            inject_metrics: false,
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
            name: None,
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
            name: Some("r1".into()),
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
}
