//! Generalized Linear Model transform (Phase 9b).
//!
//! Fits a univariate GLM `g(E[y|x]) = β₀ + β₁·x` by Iteratively Reweighted
//! Least Squares (IRLS) for an arbitrary family/link combination, and emits a
//! fitted-mean curve over an evenly-spaced grid spanning the input x-range.
//! When `ci` is set, also reports a Wald confidence band on the linear
//! predictor η, transformed through the inverse-link to mean (response) space.
//!
//! Supports five families:
//!   - Gaussian (V(μ)=1)
//!   - Binomial (V(μ)=μ(1-μ))
//!   - Poisson  (V(μ)=μ)
//!   - Gamma    (V(μ)=μ²)
//!   - InverseGaussian (V(μ)=μ³)
//!
//! Supports seven links:
//!   - Identity, Log, Logit, Probit, Inverse, InverseSquared, Sqrt
//!
//! Family/link compatibility is validated up-front; canonical pairings are:
//! Gaussian↔Identity, Binomial↔Logit, Poisson↔Log, Gamma↔Inverse,
//! InverseGaussian↔InverseSquared.
//!
//! Output schema: `[x: Float64, y: Float64, ci_lower: Float64, ci_upper: Float64]`
//! where `y` is the fitted mean E[y|x] in response space.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::logistic::inv_normal_cdf;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GlmFamily {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GlmLink {
    Identity,
    Log,
    Logit,
    Probit,
    Inverse,
    InverseSquared,
    Sqrt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct GlmSpec {
    pub x: String,
    pub y: String,
    pub family: GlmFamily,
    pub link: GlmLink,
    pub n_grid: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub max_iter: usize,
    pub tol: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

// ---------- Family/Link helpers ----------

const MU_EPS: f64 = 1e-9;
const PROB_EPS: f64 = 1e-9;

pub(crate) fn canonical_link(family: GlmFamily) -> GlmLink {
    match family {
        GlmFamily::Gaussian => GlmLink::Identity,
        GlmFamily::Binomial => GlmLink::Logit,
        GlmFamily::Poisson => GlmLink::Log,
        GlmFamily::Gamma => GlmLink::Inverse,
        GlmFamily::InverseGaussian => GlmLink::InverseSquared,
    }
}

fn family_name(family: GlmFamily) -> &'static str {
    match family {
        GlmFamily::Gaussian => "gaussian",
        GlmFamily::Binomial => "binomial",
        GlmFamily::Poisson => "poisson",
        GlmFamily::Gamma => "gamma",
        GlmFamily::InverseGaussian => "inverse_gaussian",
    }
}

fn link_name(link: GlmLink) -> &'static str {
    match link {
        GlmLink::Identity => "identity",
        GlmLink::Log => "log",
        GlmLink::Logit => "logit",
        GlmLink::Probit => "probit",
        GlmLink::Inverse => "inverse",
        GlmLink::InverseSquared => "inverse_squared",
        GlmLink::Sqrt => "sqrt",
    }
}

fn valid_links_for(family: GlmFamily) -> &'static [GlmLink] {
    match family {
        GlmFamily::Gaussian => &[GlmLink::Identity, GlmLink::Log, GlmLink::Inverse],
        GlmFamily::Binomial => &[GlmLink::Logit, GlmLink::Probit, GlmLink::Log],
        GlmFamily::Poisson => &[GlmLink::Log, GlmLink::Identity, GlmLink::Sqrt],
        GlmFamily::Gamma => &[GlmLink::Inverse, GlmLink::Identity, GlmLink::Log],
        GlmFamily::InverseGaussian => &[GlmLink::InverseSquared, GlmLink::Identity, GlmLink::Log],
    }
}

pub(crate) fn validate_pair(family: GlmFamily, link: GlmLink) -> PyResult<()> {
    let valid = valid_links_for(family);
    if valid.iter().any(|&l| l == link) {
        return Ok(());
    }
    let valid_names: Vec<String> = valid.iter().map(|l| format!("'{}'", link_name(*l))).collect();
    Err(PyValueError::new_err(format!(
        "Glm: family '{}' incompatible with link '{}'; valid links: [{}]",
        family_name(family),
        link_name(link),
        valid_names.join(", "),
    )))
}

pub(crate) fn variance(family: GlmFamily, mu: f64) -> f64 {
    match family {
        GlmFamily::Gaussian => 1.0,
        GlmFamily::Binomial => mu * (1.0 - mu),
        GlmFamily::Poisson => mu,
        GlmFamily::Gamma => mu * mu,
        GlmFamily::InverseGaussian => mu * mu * mu,
    }
}

/// Forward link g(μ).
pub(crate) fn link_apply(link: GlmLink, mu: f64) -> f64 {
    match link {
        GlmLink::Identity => mu,
        GlmLink::Log => clip_positive(mu).ln(),
        GlmLink::Logit => {
            let m = mu.clamp(PROB_EPS, 1.0 - PROB_EPS);
            (m / (1.0 - m)).ln()
        }
        GlmLink::Probit => {
            let m = mu.clamp(PROB_EPS, 1.0 - PROB_EPS);
            inv_normal_cdf(m)
        }
        GlmLink::Inverse => 1.0 / clip_positive(mu),
        GlmLink::InverseSquared => {
            let m = clip_positive(mu);
            1.0 / (m * m)
        }
        GlmLink::Sqrt => clip_positive(mu).sqrt(),
    }
}

/// Inverse link g⁻¹(η).
pub(crate) fn link_inverse(link: GlmLink, eta: f64) -> f64 {
    match link {
        GlmLink::Identity => eta,
        GlmLink::Log => eta.exp(),
        GlmLink::Logit => {
            // Stable sigmoid.
            if eta >= 0.0 {
                let e = (-eta).exp();
                1.0 / (1.0 + e)
            } else {
                let e = eta.exp();
                e / (1.0 + e)
            }
        }
        GlmLink::Probit => normal_cdf(eta),
        GlmLink::Inverse => {
            // 1/η; guard against η→0 by clipping.
            let e = if eta.abs() < MU_EPS {
                if eta < 0.0 { -MU_EPS } else { MU_EPS }
            } else { eta };
            1.0 / e
        }
        GlmLink::InverseSquared => {
            // 1/sqrt(η); η must be > 0.
            let e = if eta < MU_EPS { MU_EPS } else { eta };
            1.0 / e.sqrt()
        }
        GlmLink::Sqrt => eta * eta,
    }
}

/// Derivative g'(μ) of the link w.r.t. μ.
pub(crate) fn link_derivative(link: GlmLink, mu: f64) -> f64 {
    match link {
        GlmLink::Identity => 1.0,
        GlmLink::Log => 1.0 / clip_positive(mu),
        GlmLink::Logit => {
            let m = mu.clamp(PROB_EPS, 1.0 - PROB_EPS);
            1.0 / (m * (1.0 - m))
        }
        GlmLink::Probit => {
            let m = mu.clamp(PROB_EPS, 1.0 - PROB_EPS);
            let z = inv_normal_cdf(m);
            1.0 / normal_pdf(z)
        }
        GlmLink::Inverse => {
            let m = clip_positive(mu);
            -1.0 / (m * m)
        }
        GlmLink::InverseSquared => {
            let m = clip_positive(mu);
            -2.0 / (m * m * m)
        }
        GlmLink::Sqrt => {
            let m = clip_positive(mu);
            1.0 / (2.0 * m.sqrt())
        }
    }
}

fn clip_positive(mu: f64) -> f64 {
    if mu < MU_EPS { MU_EPS } else { mu }
}

/// Standard normal CDF via erf: Φ(z) = 0.5 * (1 + erf(z/√2)).
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal PDF.
fn normal_pdf(z: f64) -> f64 {
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    inv_sqrt_2pi * (-0.5 * z * z).exp()
}

/// Error function via Abramowitz-Stegun 7.1.26 (max error ~1.5e-7).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

// ---------- Main apply ----------

pub(crate) fn apply(spec: &GlmSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    validate_pair(spec.family, spec.link)?;
    let (xs, ys) = extract_xy(spec, batch)?;

    if xs.len() < 2 {
        return Err(PyValueError::new_err(
            "Glm: need at least 2 non-null observations",
        ));
    }

    // Family-specific input validation.
    match spec.family {
        GlmFamily::Binomial => {
            for &y in &ys {
                if y != 0.0 && y != 1.0 {
                    return Err(PyValueError::new_err(format!(
                        "Glm: binomial family requires binary y (0 or 1); found {y}",
                    )));
                }
            }
        }
        GlmFamily::Poisson => {
            for &y in &ys {
                if y < 0.0 || y.fract() != 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "Glm: poisson family requires non-negative integer y; found {y}",
                    )));
                }
            }
        }
        GlmFamily::Gamma | GlmFamily::InverseGaussian => {
            for &y in &ys {
                if y <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "Glm: {} family requires strictly positive y; found {y}",
                        family_name(spec.family),
                    )));
                }
            }
        }
        GlmFamily::Gaussian => {}
    }

    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if !(x_max > x_min) {
        return Err(PyValueError::new_err(
            "Glm: singular design (X'WX has zero determinant); x has zero variance",
        ));
    }

    let (beta, cov) = fit_irls(&xs, &ys, spec.family, spec.link, spec.max_iter, spec.tol)?;

    let n_grid = spec.n_grid.max(1);
    let grid: Vec<f64> = (0..n_grid)
        .map(|i| if n_grid <= 1 { x_min } else {
            x_min + (x_max - x_min) * (i as f64) / ((n_grid - 1) as f64)
        })
        .collect();

    let y_fit: Vec<f64> = grid.iter()
        .map(|&xq| link_inverse(spec.link, beta[0] + beta[1] * xq))
        .collect();

    let (lo, hi) = match spec.ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err("Glm: ci must be in (0, 1)"));
            }
            let z = inv_normal_cdf(1.0 - (1.0 - level) / 2.0);
            let mut lo = Vec::with_capacity(n_grid);
            let mut hi = Vec::with_capacity(n_grid);
            for &xq in &grid {
                // Var(η) = [1, x] Σ [1, x]'.
                let v = cov[0][0]
                    + 2.0 * xq * cov[0][1]
                    + xq * xq * cov[1][1];
                let se = if v >= 0.0 { v.sqrt() } else { f64::NAN };
                let eta = beta[0] + beta[1] * xq;
                lo.push(link_inverse(spec.link, eta - z * se));
                hi.push(link_inverse(spec.link, eta + z * se));
            }
            (lo, hi)
        }
    };

    build_glm_batch(grid, y_fit, lo, hi)
}

fn extract_xy(spec: &GlmSpec, batch: &RecordBatch) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("Glm: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("Glm: column '{}' not found", spec.y)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("Glm: '{}' must be Float64", spec.x)));
    }
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("Glm: '{}' must be Float64", spec.y)));
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

/// Compute initial μ for IRLS given family and observations.
fn initial_mu(family: GlmFamily, ys: &[f64]) -> Vec<f64> {
    let eps = 1e-3;
    ys.iter().map(|&y| match family {
        GlmFamily::Gaussian => y,
        GlmFamily::Binomial => ((y + 0.5) / 2.0).clamp(PROB_EPS, 1.0 - PROB_EPS),
        GlmFamily::Poisson => y + eps,
        GlmFamily::Gamma => if y > eps { y } else { eps },
        GlmFamily::InverseGaussian => if y > eps { y } else { eps },
    }).collect()
}

/// Fit GLM β = (β₀, β₁) by IRLS. Returns (β, Σ̂ = (X'WX)⁻¹).
pub(crate) fn fit_irls(
    xs: &[f64],
    ys: &[f64],
    family: GlmFamily,
    link: GlmLink,
    max_iter: usize,
    tol: f64,
) -> PyResult<([f64; 2], [[f64; 2]; 2])> {
    let n = xs.len();
    let mut mu = initial_mu(family, ys);
    // Clip μ to the link's domain for stability.
    for m in mu.iter_mut() {
        *m = clip_for_link(link, *m);
    }
    let mut eta: Vec<f64> = mu.iter().map(|&m| link_apply(link, m)).collect();

    let mut beta = [0.0_f64, 0.0_f64];
    let mut prev_dev = deviance(family, ys, &mu);
    let mut last_cov: Option<[[f64; 2]; 2]> = None;

    for _ in 0..max_iter.max(1) {
        let mut wxtx = [[0.0_f64; 2]; 2];
        let mut wxtz = [0.0_f64; 2];

        for i in 0..n {
            let mu_i = mu[i];
            let v = variance(family, mu_i).max(1e-12);
            let gprime = link_derivative(link, mu_i);
            // w_i = 1 / (V(μ) · (g'(μ))²)
            let w = 1.0 / (v * gprime * gprime).max(1e-15);
            // z_i = η_i + (y_i - μ_i) · g'(μ_i)
            let z = eta[i] + (ys[i] - mu_i) * gprime;
            let xi = xs[i];

            wxtx[0][0] += w;
            wxtx[0][1] += w * xi;
            wxtx[1][0] += w * xi;
            wxtx[1][1] += w * xi * xi;
            wxtz[0] += w * z;
            wxtz[1] += w * xi * z;
        }

        let inv = invert_2x2(wxtx).ok_or_else(|| PyValueError::new_err(
            "Glm: singular design (X'WX has zero determinant)"
        ))?;

        let b0 = inv[0][0] * wxtz[0] + inv[0][1] * wxtz[1];
        let b1 = inv[1][0] * wxtz[0] + inv[1][1] * wxtz[1];
        beta = [b0, b1];
        last_cov = Some(inv);

        if !beta[0].is_finite() || !beta[1].is_finite() {
            return Err(PyValueError::new_err(
                "Glm: IRLS diverged (non-finite coefficients)",
            ));
        }

        // Update η, μ.
        for i in 0..n {
            eta[i] = beta[0] + beta[1] * xs[i];
            let m_raw = link_inverse(link, eta[i]);
            mu[i] = clip_for_link(link, m_raw);
        }

        let dev = deviance(family, ys, &mu);
        let denom = prev_dev.abs().max(1.0);
        if (dev - prev_dev).abs() / denom < tol {
            return Ok((beta, inv));
        }
        prev_dev = dev;
    }

    Ok((beta, last_cov.unwrap_or([[f64::NAN; 2]; 2])))
}

/// Clip μ to a numerically safe region for the given link's domain.
fn clip_for_link(link: GlmLink, mu: f64) -> f64 {
    match link {
        GlmLink::Logit | GlmLink::Probit => mu.clamp(PROB_EPS, 1.0 - PROB_EPS),
        GlmLink::Log | GlmLink::Inverse | GlmLink::InverseSquared | GlmLink::Sqrt => {
            clip_positive(mu)
        }
        GlmLink::Identity => mu,
    }
}

/// Deviance for convergence check. Not a true deviance for all families
/// (we use SSR for Gaussian and log-likelihood-based forms otherwise), but it
/// monotonically tracks fit quality which is all IRLS needs.
fn deviance(family: GlmFamily, ys: &[f64], mu: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for (&y, &m) in ys.iter().zip(mu.iter()) {
        match family {
            GlmFamily::Gaussian => {
                let r = y - m;
                acc += r * r;
            }
            GlmFamily::Binomial => {
                let p = m.clamp(PROB_EPS, 1.0 - PROB_EPS);
                // -2 log L for Bernoulli observation.
                acc += -2.0 * (y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            }
            GlmFamily::Poisson => {
                let m_safe = clip_positive(m);
                // 2 (y log(y/μ) - (y - μ)); use y log y = 0 for y=0.
                let ylogterm = if y > 0.0 { y * (y / m_safe).ln() } else { 0.0 };
                acc += 2.0 * (ylogterm - (y - m_safe));
            }
            GlmFamily::Gamma => {
                let m_safe = clip_positive(m);
                let y_safe = clip_positive(y);
                // 2 (-log(y/μ) + (y - μ)/μ)
                acc += 2.0 * (-((y_safe / m_safe).ln()) + (y_safe - m_safe) / m_safe);
            }
            GlmFamily::InverseGaussian => {
                let m_safe = clip_positive(m);
                let y_safe = clip_positive(y);
                // (y - μ)² / (μ² y)
                let r = y_safe - m_safe;
                acc += (r * r) / (m_safe * m_safe * y_safe);
            }
        }
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

fn build_glm_batch(
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
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("Glm: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

fn parse_family(s: &str) -> PyResult<GlmFamily> {
    match s {
        "gaussian" => Ok(GlmFamily::Gaussian),
        "binomial" => Ok(GlmFamily::Binomial),
        "poisson" => Ok(GlmFamily::Poisson),
        "gamma" => Ok(GlmFamily::Gamma),
        "inverse_gaussian" => Ok(GlmFamily::InverseGaussian),
        other => Err(PyValueError::new_err(format!(
            "Glm: unknown family '{other}'; expected one of \
             'gaussian', 'binomial', 'poisson', 'gamma', 'inverse_gaussian'",
        ))),
    }
}

fn parse_link(s: &str) -> PyResult<GlmLink> {
    match s {
        "identity" => Ok(GlmLink::Identity),
        "log" => Ok(GlmLink::Log),
        "logit" => Ok(GlmLink::Logit),
        "probit" => Ok(GlmLink::Probit),
        "inverse" => Ok(GlmLink::Inverse),
        "inverse_squared" => Ok(GlmLink::InverseSquared),
        "sqrt" => Ok(GlmLink::Sqrt),
        other => Err(PyValueError::new_err(format!(
            "Glm: unknown link '{other}'; expected one of \
             'identity', 'log', 'logit', 'probit', 'inverse', 'inverse_squared', 'sqrt'",
        ))),
    }
}

#[pyclass(eq, module = "ferrum._core", name = "Glm")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyGlm(pub(crate) TransformSpec);

#[pymethods]
impl PyGlm {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        family = "gaussian".to_string(),
        link = None,
        n_grid = 100,
        ci = None,
        max_iter = 25,
        tol = 1e-8,
        name = None,
    ))]
    fn new(
        x: &str,
        y: &str,
        family: String,
        link: Option<String>,
        n_grid: usize,
        ci: Option<f64>,
        max_iter: usize,
        tol: f64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Glm: x and y must be non-empty"));
        }
        if n_grid == 0 {
            return Err(PyValueError::new_err("Glm: n_grid must be > 0"));
        }
        if max_iter == 0 {
            return Err(PyValueError::new_err("Glm: max_iter must be > 0"));
        }
        if !(tol.is_finite() && tol > 0.0) {
            return Err(PyValueError::new_err("Glm: tol must be finite and > 0"));
        }
        if let Some(level) = ci {
            if !(level > 0.0 && level < 1.0) {
                return Err(PyValueError::new_err("Glm: ci must be in (0, 1)"));
            }
        }

        let family = parse_family(&family)?;
        let link = match link {
            Some(s) => parse_link(&s)?,
            None => canonical_link(family),
        };
        validate_pair(family, link)?;

        Ok(PyGlm(TransformSpec::Glm(GlmSpec {
            x: x.to_string(),
            y: y.to_string(),
            family,
            link,
            n_grid,
            ci,
            max_iter,
            tol,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Glm(s) => format!(
                "Glm(x='{}', y='{}', family='{}', link='{}', n_grid={}, ci={:?}, max_iter={}, tol={})",
                s.x, s.y, family_name(s.family), link_name(s.link),
                s.n_grid, s.ci, s.max_iter, s.tol,
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

    fn spec(family: GlmFamily, link: GlmLink, n_grid: usize, ci: Option<f64>) -> GlmSpec {
        GlmSpec {
            x: "x".into(), y: "y".into(),
            family, link, n_grid, ci,
            max_iter: 50, tol: 1e-9, name: None,
        }
    }

    #[test]
    fn test_glm_round_trip_json_each_family() {
        for fam in [
            GlmFamily::Gaussian, GlmFamily::Binomial, GlmFamily::Poisson,
            GlmFamily::Gamma, GlmFamily::InverseGaussian,
        ] {
            let link = canonical_link(fam);
            let s = GlmSpec {
                x: "x".into(), y: "y".into(),
                family: fam, link, n_grid: 50,
                ci: Some(0.95), max_iter: 25, tol: 1e-8, name: None,
            };
            let json = serde_json::to_string(&s).unwrap();
            let parsed: GlmSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, s, "round-trip failed for family {:?}", fam);
        }
    }

    #[test]
    fn test_glm_gaussian_identity_recovers_ols() {
        // y = 2x + 1 exactly.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let (beta, _cov) = fit_irls(&xs, &ys, GlmFamily::Gaussian, GlmLink::Identity, 25, 1e-12).unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-9, "intercept should be 1.0; got {}", beta[0]);
        assert!((beta[1] - 2.0).abs() < 1e-9, "slope should be 2.0; got {}", beta[1]);
    }

    #[test]
    fn test_glm_binomial_logit_separated_data() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..20 {
            let x = -5.0 + i as f64 * 0.5;
            xs.push(x);
            let label = if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 };
            ys.push(label);
        }
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::Binomial, GlmLink::Logit, 50, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf[0] < 0.1, "prob at low x should be < 0.1; got {}", yf[0]);
        assert!(*yf.last().unwrap() > 0.9, "prob at high x should be > 0.9; got {}", yf.last().unwrap());
    }

    #[test]
    fn test_glm_poisson_log_counts() {
        // counts increase with x → β₁ > 0; all fitted means positive.
        let xs: Vec<f64> = (0..15).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| ((x * 1.5).exp().round() + (i % 2) as f64).max(0.0))
            .collect();
        let batch = xy_batch(xs.clone(), ys.clone());
        let s = spec(GlmFamily::Poisson, GlmLink::Log, 30, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|&v| v > 0.0 && v.is_finite()),
            "all fitted means must be finite and positive; got {:?}", yf);
        // Slope should be positive.
        let (beta, _) = fit_irls(&xs, &ys, GlmFamily::Poisson, GlmLink::Log, 50, 1e-9).unwrap();
        assert!(beta[1] > 0.0, "Poisson log slope should be positive on increasing counts; got {}", beta[1]);
    }

    #[test]
    fn test_glm_gamma_inverse_positive_data() {
        // Gamma w/ inverse link: y > 0, expect positive fitted means, convergence.
        let xs: Vec<f64> = (1..20).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 / (0.5 + 0.3 * x)).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::Gamma, GlmLink::Inverse, 30, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|&v| v > 0.0 && v.is_finite()),
            "Gamma fitted means must be positive & finite; got {:?}", yf);
    }

    #[test]
    fn test_glm_inverse_gaussian_canonical() {
        let xs: Vec<f64> = (1..20).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 / (1.0 + 0.2 * x).sqrt()).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::InverseGaussian, GlmLink::InverseSquared, 30, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|&v| v > 0.0 && v.is_finite()),
            "InverseGaussian fitted means must be positive & finite; got {:?}", yf);
    }

    #[test]
    fn test_glm_gaussian_log_non_canonical() {
        // y = exp(0.5 + 0.3 x), strictly positive; Gaussian + Log link.
        let xs: Vec<f64> = (0..15).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (0.5 + 0.3 * x).exp()).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::Gaussian, GlmLink::Log, 30, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|&v| v > 0.0 && v.is_finite()),
            "Gaussian+Log fitted means must be positive & finite; got {:?}", yf);
    }

    #[test]
    fn test_glm_binomial_probit_non_canonical() {
        // Steep separation so probit picks up the trend.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..20 {
            let x = -3.0 + i as f64 * 0.3;
            xs.push(x);
            let label = if x > 0.5 { 1.0 } else if x < -0.5 { 0.0 } else if i % 2 == 0 { 1.0 } else { 0.0 };
            ys.push(label);
        }
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::Binomial, GlmLink::Probit, 30, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf[0] < 0.5, "probit prob at low x should be < 0.5; got {}", yf[0]);
        assert!(*yf.last().unwrap() > 0.5, "probit prob at high x should be > 0.5; got {}", yf.last().unwrap());
    }

    #[test]
    fn test_glm_poisson_sqrt_non_canonical() {
        let xs: Vec<f64> = (0..15).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (1.0 + 2.0 * x).powi(2).round()).collect();
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::Poisson, GlmLink::Sqrt, 30, None);
        let out = apply(&s, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|&v| v > 0.0 && v.is_finite()),
            "Poisson+Sqrt fitted means must be positive & finite; got {:?}", yf);
    }

    #[test]
    fn test_glm_invalid_pair_error_message() {
        pyo3::Python::initialize();
        let err = validate_pair(GlmFamily::Binomial, GlmLink::Inverse).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("logit"), "error should list 'logit'; got: {msg}");
        assert!(msg.contains("probit"), "error should list 'probit'; got: {msg}");
        assert!(msg.contains("log"), "error should list 'log'; got: {msg}");
    }

    #[test]
    fn test_glm_ci_brackets_fit_gaussian_identity() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Add a little noise so cov is non-singular but finite.
        let ys: Vec<f64> = xs.iter().enumerate()
            .map(|(i, &x)| 2.0 * x + 1.0 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let batch = xy_batch(xs, ys);
        let s = spec(GlmFamily::Gaussian, GlmLink::Identity, 20, Some(0.95));
        let out = apply(&s, &batch).unwrap();
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
    fn test_glm_canonical_link_table() {
        assert_eq!(canonical_link(GlmFamily::Gaussian), GlmLink::Identity);
        assert_eq!(canonical_link(GlmFamily::Binomial), GlmLink::Logit);
        assert_eq!(canonical_link(GlmFamily::Poisson), GlmLink::Log);
        assert_eq!(canonical_link(GlmFamily::Gamma), GlmLink::Inverse);
        assert_eq!(canonical_link(GlmFamily::InverseGaussian), GlmLink::InverseSquared);
    }
}
