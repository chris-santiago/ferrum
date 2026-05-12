use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::residuals;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SmoothMethod {
    Lm,
    Loess,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SmoothOutput {
    Fitted,
    Residuals,
}

pub(crate) fn default_smooth_output() -> SmoothOutput { SmoothOutput::Fitted }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct SmoothSpec {
    pub x: String,
    pub y: String,
    pub method: SmoothMethod,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub bandwidth: f64,
    pub degree: u8,
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &SmoothSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let (xs_raw, ys_raw) = extract_xy(spec, batch)?;

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
                spec.inject_zero_ref, None,
            ),
        };
    }

    if matches!(spec.output, SmoothOutput::Residuals) {
        // Fit at original (un-aggregated) input xs; emit (x, y - fit(x)) rows.
        return residuals_fit(spec, &xs, &ys, &xs_raw, &ys_raw);
    }

    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));

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
    }
}

fn pre_aggregate_xy(
    xs: &[f64],
    ys: &[f64],
    n_bins: usize,
    estimator: crate::transform::aggregate::AggFn,
) -> (Vec<f64>, Vec<f64>) {
    use crate::transform::aggregate::AggFn;
    if xs.is_empty() || n_bins == 0 {
        return (Vec::new(), Vec::new());
    }
    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if x_min == x_max {
        let avg = match estimator {
            AggFn::Mean => ys.iter().sum::<f64>() / ys.len() as f64,
            AggFn::Sum => ys.iter().sum(),
            AggFn::Count => ys.len() as f64,
            AggFn::Min => ys.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggFn::Max => ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggFn::Median => median(ys),
        };
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
        let agg = match estimator {
            AggFn::Mean => ys_in.iter().sum::<f64>() / ys_in.len() as f64,
            AggFn::Sum => ys_in.iter().sum(),
            AggFn::Count => ys_in.len() as f64,
            AggFn::Min => ys_in.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggFn::Max => ys_in.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggFn::Median => median(ys_in),
        };
        out_x.push(mean_x);
        out_y.push(agg);
    }
    (out_x, out_y)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 { return f64::NAN; }
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
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
    }
}

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

fn extract_xy(spec: &SmoothSpec, batch: &RecordBatch) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.y)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("stat_smooth: '{}' must be Float64", spec.x)));
    }
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("stat_smooth: '{}' must be Float64", spec.y)));
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

fn all_nan_output(spec: &SmoothSpec) -> PyResult<RecordBatch> {
    let n = spec.n;
    let nans = vec![f64::NAN; n];
    build_smooth_batch(nans.clone(), nans.clone(), nans.clone(), nans, None)
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
    let mut fields = vec![
        Field::new("x",        DataType::Float64, true),
        Field::new("y",        DataType::Float64, true),
        Field::new("ci_lower", DataType::Float64, true),
        Field::new("ci_upper", DataType::Float64, true),
    ];
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
        return build_smooth_batch(
            grid.to_vec(),
            vec![f64::NAN; n_grid],
            vec![f64::NAN; n_grid],
            vec![f64::NAN; n_grid],
            None,
        );
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
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
///     Predictor column (must be Float64).
/// y : str
///     Response column (must be Float64).
/// method : {"loess", "lm"}, default "loess"
///     Smoothing method. ``"loess"`` fits locally-weighted polynomial
///     regression; ``"lm"`` fits a global linear model.
/// ci : float or None, default 0.95
///     Confidence level in (0, 1) for bootstrap confidence bands. Set to
///     ``None`` to suppress the band.
/// bandwidth : float, default 0.75
///     LOESS span in (0, 1]; fraction of data used in each local fit.
///     Ignored when ``method="lm"``.
/// degree : {1, 2}, default 2
///     Polynomial degree for LOESS. Ignored when ``method="lm"``.
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
#[pyclass(eq, module = "ferrum._core", name = "Smooth")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PySmooth(pub(crate) TransformSpec);

#[pymethods]
impl PySmooth {
    #[new]
    #[pyo3(signature = (x, y, *, method = "loess", ci = Some(0.95), bandwidth = 0.75, degree = 2, n = 200, seed = 0, x_bins = None, x_estimator = None, output = "fitted", inject_zero_ref = false, inject_metrics = false, name = None))]
    fn new(
        x: &str, y: &str,
        method: &str,
        ci: Option<f64>,
        bandwidth: f64,
        degree: u8,
        n: usize,
        seed: u64,
        x_bins: Option<usize>,
        x_estimator: Option<&str>,
        output: &str,
        inject_zero_ref: bool,
        inject_metrics: bool,
        name: Option<String>,
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
        let method = match method {
            "lm" => SmoothMethod::Lm,
            "loess" => SmoothMethod::Loess,
            other => return Err(PyValueError::new_err(format!(
                "Smooth: unknown method '{other}'; expected 'lm' | 'loess'"
            ))),
        };
        if matches!(method, SmoothMethod::Loess) {
            if !bandwidth.is_finite() || bandwidth <= 0.0 || bandwidth > 1.0 {
                return Err(PyValueError::new_err(
                    "Smooth: LOESS bandwidth must be a finite value in (0, 1]",
                ));
            }
            if degree != 1 && degree != 2 {
                return Err(PyValueError::new_err("Smooth: LOESS degree must be 1 or 2"));
            }
        }
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
        Ok(PySmooth(TransformSpec::Smooth(SmoothSpec {
            x: x.to_string(), y: y.to_string(),
            method, ci, bandwidth, degree, n, seed,
            x_bins,
            x_estimator: x_estimator_parsed,
            output: output_parsed,
            inject_zero_ref,
            inject_metrics,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Smooth(s) => format!(
                "Smooth(x='{}', y='{}', method={:?}, ci={:?}, bandwidth={}, degree={}, n={}, seed={}, x_bins={:?}, x_estimator={:?}, output={:?}, inject_zero_ref={}, inject_metrics={}, name={:?})",
                s.x, s.y, s.method, s.ci, s.bandwidth, s.degree, s.n, s.seed,
                s.x_bins, s.x_estimator, s.output,
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

    #[test]
    fn test_lm_recovers_slope_and_intercept_exactly() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 + 2.0 * x).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
            bandwidth: 0.0, degree: 1, n: 51, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            name: None,
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
            bandwidth: 0.5, degree: 2, n: 100, seed: 42,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
                bandwidth: case.bandwidth, degree: case.degree, n: case.n, seed: 0,
                x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
                inject_zero_ref: false,
                inject_metrics: false,
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
                bandwidth: case.bandwidth, degree: case.degree, n: case.n, seed: 0,
                x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
                inject_zero_ref: false,
                inject_metrics: false,
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
            degree: 2, n: 5, seed: 0,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
            bandwidth: 0.5, degree: 1, n: 20, seed: 42,
            x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
            x_bins: Some(10),
            x_estimator: Some(crate::transform::aggregate::AggFn::Mean),
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
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
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
            x_bins: None, x_estimator: None,
            output: SmoothOutput::Residuals,
            inject_zero_ref: false,
            inject_metrics: false,
            name: None,
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
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
            x_bins: None, x_estimator: None,
            output: SmoothOutput::Fitted,
            inject_zero_ref: false,
            inject_metrics: false,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let schema = out.schema();
        let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, vec!["x".to_string(), "y".to_string(), "ci_lower".to_string(), "ci_upper".to_string()]);
    }
}
