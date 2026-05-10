//! Logistic-regression transform (Phase 9b).
//!
//! Fits a univariate logistic regression `logit(P(y=1)) = β₀ + β₁·x` by
//! Iteratively Reweighted Least Squares (IRLS) and emits a probability curve
//! over an evenly-spaced grid spanning the input x-range. When `ci` is set,
//! also reports a Wald confidence band on the linear predictor, transformed
//! through the inverse-link (sigmoid) to probability space.
//!
//! Output schema: `[x: Float64, y: Float64, ci_lower: Float64, ci_upper: Float64]`
//! where `y` is the fitted probability `P(y=1 | x)`.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LogisticSpec {
    pub x: String,
    pub y: String,
    pub n_grid: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub max_iter: usize,
    pub tol: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &LogisticSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let (xs, ys) = extract_xy(spec, batch)?;
    validate_binary_y(&ys)?;

    if xs.len() < 2 {
        return Err(PyValueError::new_err(
            "Logistic: need at least 2 non-null observations",
        ));
    }

    // Singular design: zero variance in x => X'WX is singular.
    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if !(x_max > x_min) {
        return Err(PyValueError::new_err(
            "Logistic: singular design (X'WX has zero determinant); x has zero variance",
        ));
    }

    let (beta, cov) = fit_irls(&xs, &ys, spec.max_iter, spec.tol)?;

    let n_grid = spec.n_grid.max(1);
    let grid: Vec<f64> = (0..n_grid)
        .map(|i| if n_grid <= 1 { x_min } else {
            x_min + (x_max - x_min) * (i as f64) / ((n_grid - 1) as f64)
        })
        .collect();

    let y_fit: Vec<f64> = grid.iter().map(|&xq| sigmoid(beta[0] + beta[1] * xq)).collect();

    let (lo, hi) = match spec.ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err(
                    "Logistic: ci must be in (0, 1)",
                ));
            }
            let z = inv_normal_cdf(1.0 - (1.0 - level) / 2.0);
            let mut lo = Vec::with_capacity(n_grid);
            let mut hi = Vec::with_capacity(n_grid);
            for &xq in &grid {
                // η = β0 + β1 x; Var(η) = [1, x] Σ [1, x]'
                let v = cov[0][0]
                    + 2.0 * xq * cov[0][1]
                    + xq * xq * cov[1][1];
                let se = if v >= 0.0 { v.sqrt() } else { f64::NAN };
                let eta = beta[0] + beta[1] * xq;
                lo.push(sigmoid(eta - z * se));
                hi.push(sigmoid(eta + z * se));
            }
            (lo, hi)
        }
    };

    build_logistic_batch(grid, y_fit, lo, hi)
}

fn extract_xy(spec: &LogisticSpec, batch: &RecordBatch) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("Logistic: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("Logistic: column '{}' not found", spec.y)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("Logistic: '{}' must be Float64", spec.x)));
    }
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("Logistic: '{}' must be Float64", spec.y)));
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

fn validate_binary_y(ys: &[f64]) -> PyResult<()> {
    for &y in ys {
        if y != 0.0 && y != 1.0 {
            return Err(PyValueError::new_err(format!(
                "Logistic: y must be binary (only 0 or 1); found value {y}",
            )));
        }
    }
    Ok(())
}

fn sigmoid(z: f64) -> f64 {
    // Stable sigmoid: avoid overflow for large |z|.
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Fit logistic regression β = (β₀, β₁) by IRLS. Returns (β, Σ̂).
fn fit_irls(
    xs: &[f64],
    ys: &[f64],
    max_iter: usize,
    tol: f64,
) -> PyResult<([f64; 2], [[f64; 2]; 2])> {
    let n = xs.len();
    let mut beta = [0.0_f64, 0.0_f64];
    let mut eta = vec![0.0_f64; n];
    let mut mu = vec![0.5_f64; n];

    let mut prev_ll = log_likelihood(ys, &mu);
    let mut last_cov: Option<[[f64; 2]; 2]> = None;

    for it in 0..max_iter.max(1) {
        // Weights and working response.
        let mut wxtx = [[0.0_f64; 2]; 2]; // X' W X
        let mut wxtz = [0.0_f64; 2];      // X' W z
        for i in 0..n {
            let w = (mu[i] * (1.0 - mu[i])).max(1e-12);
            let z = eta[i] + (ys[i] - mu[i]) / w;
            let xi = xs[i];
            wxtx[0][0] += w;
            wxtx[0][1] += w * xi;
            wxtx[1][0] += w * xi;
            wxtx[1][1] += w * xi * xi;
            wxtz[0] += w * z;
            wxtz[1] += w * xi * z;
        }

        let inv = invert_2x2(wxtx).ok_or_else(|| PyValueError::new_err(
            "Logistic: singular design (X'WX has zero determinant)"
        ))?;

        let b0 = inv[0][0] * wxtz[0] + inv[0][1] * wxtz[1];
        let b1 = inv[1][0] * wxtz[0] + inv[1][1] * wxtz[1];
        beta = [b0, b1];
        last_cov = Some(inv);

        // Perfect-separation guard: blow-up in coefficients.
        if beta[0].abs() > 1e3 || beta[1].abs() > 1e3 {
            return Err(PyValueError::new_err(
                "Logistic: perfect separation detected (coefficients diverged)",
            ));
        }

        // Update η, μ.
        for i in 0..n {
            eta[i] = beta[0] + beta[1] * xs[i];
            mu[i] = sigmoid(eta[i]);
        }

        // Perfect-separation guard: probabilities saturate early.
        if it < 5 {
            let saturated = mu.iter().all(|&p| p < 1e-6 || p > 1.0 - 1e-6);
            if saturated {
                return Err(PyValueError::new_err(
                    "Logistic: perfect separation detected (fitted probabilities saturated)",
                ));
            }
        }

        let ll = log_likelihood(ys, &mu);
        if (ll - prev_ll).abs() < tol {
            check_separation_at_convergence(ys, &mu)?;
            return Ok((beta, inv));
        }
        prev_ll = ll;
    }

    // Did not strictly converge under tol; check final fit and return.
    check_separation_at_convergence(ys, &mu)?;
    Ok((beta, last_cov.unwrap_or([[f64::NAN; 2]; 2])))
}

/// Detect perfect separation by checking whether every observation is
/// classified with near-certainty on the correct side. With binary labels,
/// `y * μ + (1-y) * (1-μ)` is the probability the model assigns to the
/// observed label; this is ≥ 1 - 1e-3 for every i iff the fit perfectly
/// separates the classes.
fn check_separation_at_convergence(ys: &[f64], mu: &[f64]) -> PyResult<()> {
    if ys.is_empty() { return Ok(()); }
    let mut all_perfect = true;
    for (&y, &p) in ys.iter().zip(mu.iter()) {
        let p_correct = y * p + (1.0 - y) * (1.0 - p);
        if p_correct < 1.0 - 1e-3 {
            all_perfect = false;
            break;
        }
    }
    if all_perfect {
        return Err(PyValueError::new_err(
            "Logistic: perfect separation detected (model classifies every observation with probability > 0.999)",
        ));
    }
    Ok(())
}

fn log_likelihood(ys: &[f64], mu: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for (y, p) in ys.iter().zip(mu.iter()) {
        let pc = p.clamp(1e-12, 1.0 - 1e-12);
        acc += y * pc.ln() + (1.0 - y) * (1.0 - pc).ln();
    }
    acc
}

fn invert_2x2(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-15 { return None; }
    let inv_det = 1.0 / det;
    Some([
        [ a[1][1] * inv_det, -a[0][1] * inv_det],
        [-a[1][0] * inv_det,  a[0][0] * inv_det],
    ])
}

fn build_logistic_batch(
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
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("Logistic: {e}")))
}

/// Inverse standard normal CDF (Beasley-Springer / Moro).
fn inv_normal_cdf(p: f64) -> f64 {
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

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

#[pyclass(eq, module = "ferrum._core", name = "Logistic")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyLogistic(pub(crate) TransformSpec);

#[pymethods]
impl PyLogistic {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        n_grid = 100,
        ci = None,
        max_iter = 25,
        tol = 1e-8,
        name = None,
    ))]
    fn new(
        x: &str,
        y: &str,
        n_grid: usize,
        ci: Option<f64>,
        max_iter: usize,
        tol: f64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Logistic: x and y must be non-empty"));
        }
        if n_grid == 0 {
            return Err(PyValueError::new_err("Logistic: n_grid must be > 0"));
        }
        if max_iter == 0 {
            return Err(PyValueError::new_err("Logistic: max_iter must be > 0"));
        }
        if !(tol.is_finite() && tol > 0.0) {
            return Err(PyValueError::new_err("Logistic: tol must be finite and > 0"));
        }
        if let Some(level) = ci {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err("Logistic: ci must be in (0, 1)"));
            }
        }
        Ok(PyLogistic(TransformSpec::Logistic(LogisticSpec {
            x: x.to_string(),
            y: y.to_string(),
            n_grid,
            ci,
            max_iter,
            tol,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Logistic(s) => format!(
                "Logistic(x='{}', y='{}', n_grid={}, ci={:?}, max_iter={}, tol={})",
                s.x, s.y, s.n_grid, s.ci, s.max_iter, s.tol,
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

    fn default_spec(n_grid: usize, ci: Option<f64>) -> LogisticSpec {
        LogisticSpec {
            x: "x".into(), y: "y".into(),
            n_grid, ci, max_iter: 25, tol: 1e-8, name: None,
        }
    }

    #[test]
    fn test_logistic_round_trip_json() {
        let original = LogisticSpec {
            x: "x".into(), y: "y".into(),
            n_grid: 100, ci: Some(0.95), max_iter: 25, tol: 1e-8, name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: LogisticSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_logistic_well_separated_data() {
        // x in [-5, 5]; y=0 mostly for x<0, y=1 mostly for x>0, with overlap to avoid
        // perfect separation.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..20 {
            let x = -5.0 + i as f64 * 0.5;
            xs.push(x);
            // Probabilistic-ish but deterministic: y=1 if x > 0, with a few mis-labels
            // near zero to break separation.
            let label = if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 };
            ys.push(label);
        }
        let batch = xy_batch(xs.clone(), ys);
        let spec = default_spec(50, None);
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        // Probability should be < 0.1 at low end, > 0.9 at high end.
        assert!(yf[0] < 0.1, "fitted prob at min(x)={} should be < 0.1, got {}", xg[0], yf[0]);
        assert!(*yf.last().unwrap() > 0.9,
            "fitted prob at max(x)={} should be > 0.9, got {}", xg.last().unwrap(), yf.last().unwrap());
        // Monotone increasing (slope > 0).
        for i in 1..yf.len() {
            assert!(yf[i] >= yf[i-1] - 1e-12, "fit should be monotone non-decreasing");
        }
    }

    #[test]
    fn test_logistic_converges_within_max_iter() {
        let xs: Vec<f64> = (0..40).map(|i| -4.0 + i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            if x > 0.0 { 1.0 } else if x < -1.0 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 }
        ).collect();
        // tight max_iter still produces a usable result; we just check no error and finite.
        let batch = xy_batch(xs, ys);
        let spec = LogisticSpec {
            x: "x".into(), y: "y".into(),
            n_grid: 10, ci: None, max_iter: 25, tol: 1e-8, name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|y| y.is_finite()));
    }

    #[test]
    fn test_logistic_non_binary_y_raises() {
        pyo3::Python::initialize();
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let ys = vec![0.0, 0.5, 1.0, 0.0, 1.0, 0.0];
        let batch = xy_batch(xs, ys);
        let spec = default_spec(10, None);
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("binary"), "expected 'binary' in error; got: {msg}");
    }

    #[test]
    fn test_logistic_perfect_separation_raises() {
        pyo3::Python::initialize();
        // All y=0 for x<<0, all y=1 for x>>0 with a huge gap → perfect separation.
        let xs: Vec<f64> = vec![-100.0, -90.0, -80.0, 80.0, 90.0, 100.0];
        let ys: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let batch = xy_batch(xs, ys);
        let spec = default_spec(10, None);
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.to_lowercase().contains("separation"),
            "expected 'separation' in error; got: {msg}");
    }

    #[test]
    fn test_logistic_singular_design_raises() {
        pyo3::Python::initialize();
        // All x identical → zero variance → singular X'WX.
        let xs = vec![3.0; 10];
        let ys: Vec<f64> = (0..10).map(|i| (i % 2) as f64).collect();
        let batch = xy_batch(xs, ys);
        let spec = default_spec(10, None);
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.to_lowercase().contains("singular"),
            "expected 'singular' in error; got: {msg}");
    }

    #[test]
    fn test_logistic_ci_brackets_fit() {
        let xs: Vec<f64> = (0..30).map(|i| -3.0 + i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 }
        ).collect();
        let batch = xy_batch(xs, ys);
        let spec = default_spec(20, Some(0.95));
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        let lo = col(&out, "ci_lower");
        let hi = col(&out, "ci_upper");
        for i in 0..yf.len() {
            assert!(lo[i].is_finite() && hi[i].is_finite(),
                "CI must be finite at i={i}; got lo={}, hi={}", lo[i], hi[i]);
            assert!(lo[i] <= yf[i] + 1e-9 && yf[i] - 1e-9 <= hi[i],
                "CI must bracket fit at i={i}: lo={}, y={}, hi={}", lo[i], yf[i], hi[i]);
        }
    }

    #[test]
    fn test_logistic_schema_is_x_y_lo_hi_with_nan_when_no_ci() {
        let xs: Vec<f64> = (0..20).map(|i| -2.0 + i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 }
        ).collect();
        let batch = xy_batch(xs, ys);
        let spec = default_spec(15, None);
        let out = apply(&spec, &batch).unwrap();
        let schema = out.schema();
        let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, vec![
            "x".to_string(), "y".to_string(),
            "ci_lower".to_string(), "ci_upper".to_string(),
        ]);
        let lo = col(&out, "ci_lower");
        let hi = col(&out, "ci_upper");
        assert!(lo.iter().all(|v| v.is_nan()), "ci_lower must be all-NaN when ci=None");
        assert!(hi.iter().all(|v| v.is_nan()), "ci_upper must be all-NaN when ci=None");
    }

    #[test]
    fn test_logistic_n_grid_100_produces_100_rows() {
        let xs: Vec<f64> = (0..40).map(|i| -4.0 + i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 }
        ).collect();
        let batch = xy_batch(xs, ys);
        let spec = default_spec(100, None);
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 100);
    }

    #[test]
    fn test_logistic_tight_tol_still_converges() {
        let xs: Vec<f64> = (0..30).map(|i| -3.0 + i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 }
        ).collect();
        let batch = xy_batch(xs, ys);
        let spec = LogisticSpec {
            x: "x".into(), y: "y".into(),
            n_grid: 10, ci: None, max_iter: 100, tol: 1e-14, name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|y| y.is_finite() && (0.0..=1.0).contains(y)));
    }

    #[test]
    fn test_logistic_slope_positive_on_increasing_data() {
        let xs: Vec<f64> = (0..30).map(|i| -3.0 + i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)|
            if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 }
        ).collect();
        let batch = xy_batch(xs, ys);
        let (beta, _cov) = fit_irls(
            &col_from(&batch, "x"),
            &col_from(&batch, "y"),
            25, 1e-8,
        ).unwrap();
        assert!(beta[1] > 0.0, "slope should be positive on increasing data; got {}", beta[1]);
    }

    fn col_from(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }
}
