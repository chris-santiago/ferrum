//! Rust-native statistics for model diagnostics.
//!
//! Replaces `src/ferrum/_diagnostics/stats.py`. All functions accept Arrow
//! arrays via pyo3-arrow and return Arrow RecordBatches — zero-copy on the
//! polars → Rust path.

use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, Int64Array, ArrayRef};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::{PyArray, PyRecordBatch};

use faer::Side;

use crate::diagnostics;
use crate::transform::linalg;
use crate::transform::linkage::DistanceMetric;

// ---------------------------------------------------------------------------
// Internal pure-Rust implementations
// ---------------------------------------------------------------------------

fn studentized_residual_vec(
    y_true: &[f64],
    y_pred: &[f64],
    h_diag: Option<&[f64]>,
) -> Vec<f64> {
    let n = y_true.len();
    let r: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect();
    match h_diag {
        None => {
            let sigma = if n > 1 {
                let mean = r.iter().sum::<f64>() / n as f64;
                let var = r.iter().map(|ri| (ri - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                var.sqrt()
            } else {
                1.0
            };
            if sigma > 0.0 {
                r.iter().map(|ri| ri / sigma).collect()
            } else {
                vec![0.0; n]
            }
        }
        Some(h) => {
            let sse: f64 = r.iter().map(|ri| ri * ri).sum();
            let p_eff = h.iter().sum::<f64>().round() as usize;
            let sigma_sq = sse / (n.saturating_sub(p_eff).max(1)) as f64;
            let sigma = sigma_sq.sqrt();
            if sigma <= 0.0 {
                return vec![0.0; n];
            }
            r.iter()
                .zip(h)
                .map(|(ri, hi)| ri / (sigma * (1.0 - hi).max(1e-12).sqrt()))
                .collect()
        }
    }
}

fn cooks_distance_vec(
    y_true: &[f64],
    y_pred: &[f64],
    h_diag: &[f64],
) -> Vec<f64> {
    let n = y_true.len();
    let r: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect();
    let p_eff = h_diag.iter().sum::<f64>().round() as usize;
    if n <= p_eff {
        return vec![0.0; n];
    }
    // B19: p_eff=0 would cause division by zero (p_eff as f64 * sigma_sq = 0)
    // producing NaN. Cook's distance is undefined without predictors; return zeros.
    if p_eff == 0 {
        return vec![0.0; n];
    }
    let sse: f64 = r.iter().map(|ri| ri * ri).sum();
    let sigma_sq = sse / (n - p_eff).max(1) as f64;
    if sigma_sq <= 0.0 {
        return vec![0.0; n];
    }
    r.iter()
        .zip(h_diag)
        .map(|(ri, hi)| {
            // B18: leverage hi can be negative (e.g. from numerical issues in the
            // hat matrix). Cook's distance is non-negative by definition, so clamp
            // the leverage to [0, ∞) before computing the influence term.
            let hi_clamped = hi.max(0.0);
            let one_minus_h_sq = (1.0 - hi_clamped).powi(2);
            let denom = if one_minus_h_sq > 0.0 { one_minus_h_sq } else { 1e-24 };
            (ri * ri / (p_eff as f64 * sigma_sq)) * (hi_clamped / denom)
        })
        .collect()
}

fn rankdata_average_vec(arr: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    for (rank_idx, &orig_idx) in order.iter().enumerate() {
        ranks[orig_idx] = (rank_idx + 1) as f64;
    }
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && arr[order[j]] == arr[order[i]] {
            j += 1;
        }
        if j - i > 1 {
            let mean_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0;
            for k in i..j {
                ranks[order[k]] = mean_rank;
            }
        }
        i = j;
    }
    ranks
}

fn variance_rank_vec(x_flat: &[f64], n: usize, p: usize) -> Vec<f64> {
    // B21: n=0 produces 0.0/0.0 = NaN through mean/variance computation.
    // Variance of an empty column is defined as 0.
    if n == 0 {
        return vec![0.0; p];
    }
    let mut result = vec![0.0; p];
    for j in 0..p {
        let mean: f64 = (0..n).map(|i| x_flat[i * p + j]).sum::<f64>() / n as f64;
        let var: f64 = (0..n).map(|i| (x_flat[i * p + j] - mean).powi(2)).sum::<f64>() / n as f64;
        result[j] = var;
    }
    result
}

fn covariance_rank_vec(x_flat: &[f64], n: usize, p: usize, y: &[f64]) -> Vec<f64> {
    if n < 2 {
        return vec![0.0; p];
    }
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ym: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();
    let mut result = vec![0.0; p];
    for j in 0..p {
        let x_mean: f64 = (0..n).map(|i| x_flat[i * p + j]).sum::<f64>() / n as f64;
        let cov: f64 = (0..n).map(|i| (x_flat[i * p + j] - x_mean) * ym[i]).sum::<f64>()
            / (n - 1) as f64;
        result[j] = cov.abs();
    }
    result
}

fn phi_inv(p: f64) -> f64 {
    if p <= 0.0 { return -8.0; }
    if p >= 1.0 { return 8.0; }
    let a = [
        -3.96968302866538e01, 2.20946098424520e02, -2.75928510446969e02,
        1.38357751867269e02, -3.06647980661472e01, 2.50662827745924e00,
    ];
    let b = [
        -5.44760987982241e01, 1.61585836858041e02, -1.55698979859887e02,
        6.68013118877197e01, -1.32806815528857e01,
    ];
    let c = [
        -7.78489400243029e-03, -3.22396458041137e-01, -2.40075827716184e00,
        -2.54973253934373e00, 4.37466414146497e00, 2.93816398269878e00,
    ];
    let d = [
        7.78469570904146e-03, 3.22467129070040e-01,
        2.44513413714300e00, 3.75440866190742e00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q
            / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
}

pub fn shapiro_w_scalar(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return f64::NAN; }
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let m: Vec<f64> = (1..=n)
        .map(|k| phi_inv((k as f64 - 0.375) / (n as f64 + 0.25)))
        .collect();
    let m_norm_sq: f64 = m.iter().map(|v| v * v).sum();
    let u = 1.0 / (n as f64).sqrt();

    let a_n = -2.706056 * u.powi(5) + 4.434685 * u.powi(4) - 2.071190 * u.powi(3)
        - 0.147981 * u.powi(2) + 0.221157 * u + m[n - 1] / m_norm_sq.sqrt();
    let a_nm1 = -3.582633 * u.powi(5) + 5.682633 * u.powi(4) - 1.752460 * u.powi(3)
        - 0.293762 * u.powi(2) + 0.042981 * u + m[n - 2] / m_norm_sq.sqrt();

    let eps = (m_norm_sq - 2.0 * m[n - 1].powi(2) - 2.0 * m[n - 2].powi(2))
        / (1.0 - 2.0 * a_n.powi(2) - 2.0 * a_nm1.powi(2));

    let mut a = vec![0.0; n];
    a[n - 1] = a_n;
    a[n - 2] = a_nm1;
    a[0] = -a_n;
    a[1] = -a_nm1;
    if n > 4 {
        // Guard against negative eps (can occur for small n with near-linear input);
        // eps.max(0.0) prevents NaN propagation through the W statistic.
        let eps_sqrt = eps.max(0.0).sqrt();
        for i in 2..n - 2 {
            a[i] = if eps_sqrt > 0.0 { m[i] / eps_sqrt } else { 0.0 };
        }
    }

    let numer: f64 = a.iter().zip(sorted.iter()).map(|(ai, xi)| ai * xi).sum::<f64>().powi(2);
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let denom: f64 = sorted.iter().map(|xi| (xi - mean).powi(2)).sum();
    if denom <= 0.0 { return 1.0; }
    // B20: The Royston polynomial approximation can produce W slightly above 1.0
    // for small n with skewed data (e.g. n=3, W≈1.009). Clamp to the valid range.
    (numer / denom).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Helpers for Arrow extraction
// ---------------------------------------------------------------------------

fn extract_f64_column(batch: &RecordBatch, idx: usize) -> PyResult<Vec<f64>> {
    let arr = batch.column(idx);
    let f64_arr = arr.as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err(format!(
            "Column {} must be Float64, got {:?}", batch.schema().field(idx).name(), arr.data_type()
        )))?;
    Ok(f64_arr.values().to_vec())
}

fn batch_to_flat_f64(batch: &RecordBatch) -> PyResult<(Vec<f64>, usize, usize, Vec<String>)> {
    let n = batch.num_rows();
    let p = batch.num_columns();
    let mut flat = vec![0.0; n * p];
    let mut col_names = Vec::with_capacity(p);
    for j in 0..p {
        col_names.push(batch.schema().field(j).name().clone());
        let col = extract_f64_column(batch, j)?;
        for i in 0..n {
            flat[i * p + j] = col[i];
        }
    }
    Ok((flat, n, p, col_names))
}

fn f64_array(v: Vec<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(v))
}

// ---------------------------------------------------------------------------
// PyO3-exposed functions (Arrow in, Arrow out)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn hat_matrix_stats(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    y_true: PyArray,
    y_pred: PyArray,
) -> PyResult<PyRecordBatch> {
    let x_batch: RecordBatch = x_table.into();
    let y_true_arr = y_true.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("y_true must be Float64"))?.clone();
    let y_pred_arr = y_pred.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("y_pred must be Float64"))?.clone();

    let n = x_batch.num_rows();
    let p = x_batch.num_columns();
    let yt = y_true_arr.values();
    let yp = y_pred_arr.values();

    if yt.len() != n || yp.len() != n {
        return Err(PyValueError::new_err("y_true/y_pred length must match X row count"));
    }

    // Build design matrix with intercept column prepended
    let p_with_intercept = p + 1;
    let mut flat = vec![0.0; n * p_with_intercept];
    for i in 0..n {
        flat[i * p_with_intercept] = 1.0; // intercept
    }
    for j in 0..p {
        let col = extract_f64_column(&x_batch, j)?;
        for i in 0..n {
            flat[i * p_with_intercept + j + 1] = col[i];
        }
    }
    let x_mat = linalg::mat_from_flat(&flat, n, p_with_intercept)?;
    let h = linalg::hat_diagonal(&x_mat);

    let stud = studentized_residual_vec(yt, yp, Some(&h));
    let cooks = cooks_distance_vec(yt, yp, &h);

    let schema = Arc::new(Schema::new(vec![
        Field::new("studentized_residual", DataType::Float64, false),
        Field::new("cooks_distance", DataType::Float64, false),
        Field::new("leverage", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        f64_array(stud),
        f64_array(cooks),
        f64_array(h),
    ]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(batch))
}

#[pyfunction]
pub fn studentized_residual_no_x(
    _py: Python<'_>,
    y_true: PyArray,
    y_pred: PyArray,
) -> PyResult<PyArray> {
    let yt = y_true.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("y_true must be Float64"))?.clone();
    let yp = y_pred.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("y_pred must be Float64"))?.clone();
    let stud = studentized_residual_vec(yt.values(), yp.values(), None);
    Ok(PyArray::from_array_ref(Arc::new(Float64Array::from(stud))))
}

#[pyfunction]
pub fn py_shapiro_w(_py: Python<'_>, x: PyArray) -> PyResult<f64> {
    let arr = x.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("x must be Float64"))?;
    if arr.len() < 3 {
        return Err(PyValueError::new_err("shapiro_w requires n >= 3"));
    }
    Ok(shapiro_w_scalar(arr.values()))
}

#[pyfunction]
pub fn py_rankdata_average(
    _py: Python<'_>,
    x: PyArray,
) -> PyResult<PyArray> {
    let arr = x.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("x must be Float64"))?;
    let ranks = rankdata_average_vec(arr.values());
    Ok(PyArray::from_array_ref(Arc::new(Float64Array::from(ranks))))
}

#[pyfunction]
pub fn py_rank1d(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    algorithm: &str,
    top_k: Option<usize>,
) -> PyResult<PyRecordBatch> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, p, col_names) = batch_to_flat_f64(&batch)?;

    let scores: Vec<f64> = match algorithm {
        "shapiro" => (0..p)
            .map(|j| {
                let col: Vec<f64> = (0..n).map(|i| flat[i * p + j]).collect();
                shapiro_w_scalar(&col)
            })
            .collect(),
        "variance" => variance_rank_vec(&flat, n, p),
        "covariance" => {
            return Err(PyValueError::new_err(
                "rank1d(algorithm='covariance') requires y; use py_rank1d_with_y",
            ));
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "rank1d algorithm must be 'shapiro' or 'variance'; got '{other}'"
            )));
        }
    };

    rank1d_from_scores(&col_names, &scores, top_k)
}

#[pyfunction]
pub fn py_rank1d_with_y(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    y: PyArray,
    algorithm: &str,
    top_k: Option<usize>,
) -> PyResult<PyRecordBatch> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, p, col_names) = batch_to_flat_f64(&batch)?;
    let y_arr = y.array().as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("y must be Float64"))?.clone();
    let y_vals = y_arr.values();

    let scores: Vec<f64> = match algorithm {
        "covariance" => covariance_rank_vec(&flat, n, p, y_vals),
        "shapiro" => (0..p)
            .map(|j| {
                let col: Vec<f64> = (0..n).map(|i| flat[i * p + j]).collect();
                shapiro_w_scalar(&col)
            })
            .collect(),
        "variance" => variance_rank_vec(&flat, n, p),
        other => {
            return Err(PyValueError::new_err(format!(
                "rank1d algorithm must be 'shapiro', 'variance', or 'covariance'; got '{other}'"
            )));
        }
    };

    rank1d_from_scores(&col_names, &scores, top_k)
}

fn rank1d_from_scores(
    col_names: &[String],
    scores: &[f64],
    top_k: Option<usize>,
) -> PyResult<PyRecordBatch> {
    let p = scores.len();
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

    let limit = top_k.unwrap_or(p).min(p);
    let features: Vec<String> = order[..limit].iter().map(|&i| col_names[i].clone()).collect();
    let score_vals: Vec<f64> = order[..limit].iter().map(|&i| scores[i]).collect();
    let ranks: Vec<i64> = (1..=limit as i64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("feature", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("rank", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from(features)) as ArrayRef,
        f64_array(score_vals),
        Arc::new(Int64Array::from(ranks)) as ArrayRef,
    ]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(batch))
}

#[pyfunction]
pub fn py_rank2d(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    algorithm: &str,
) -> PyResult<PyRecordBatch> {
    let rb: RecordBatch = x_table.into();
    let (flat, n, p, col_names) = batch_to_flat_f64(&rb)?;

    let c_flat: Vec<f64> = match algorithm {
        "pearson" => {
            let mat = linalg::mat_from_flat(&flat, n, p)?;
            let c = linalg::corrcoef(&mat);
            let mut out = vec![0.0; p * p];
            for i in 0..p {
                for j in 0..p {
                    out[i * p + j] = c[(i, j)];
                }
            }
            out
        }
        "spearman" => {
            // Rank each column, then compute pearson on ranks
            let mut ranked_flat = vec![0.0; n * p];
            for j in 0..p {
                let col: Vec<f64> = (0..n).map(|i| flat[i * p + j]).collect();
                let ranks = rankdata_average_vec(&col);
                for i in 0..n {
                    ranked_flat[i * p + j] = ranks[i];
                }
            }
            let mat = linalg::mat_from_flat(&ranked_flat, n, p)?;
            let c = linalg::corrcoef(&mat);
            let mut out = vec![0.0; p * p];
            for i in 0..p {
                for j in 0..p {
                    out[i * p + j] = c[(i, j)];
                }
            }
            out
        }
        "kendall" => {
            let mut c_out = vec![0.0; p * p];
            for i in 0..p {
                c_out[i * p + i] = 1.0;
                for j in (i + 1)..p {
                    let x_col: Vec<f64> = (0..n).map(|r| flat[r * p + i]).collect();
                    let y_col: Vec<f64> = (0..n).map(|r| flat[r * p + j]).collect();
                    let result = diagnostics::kendall_tau_b(&x_col, &y_col);
                    let tau = if result.tau.is_finite() { result.tau } else { 0.0 };
                    c_out[i * p + j] = tau;
                    c_out[j * p + i] = tau;
                }
            }
            c_out
        }
        "covariance" => {
            if p == 1 {
                let mean = flat.iter().sum::<f64>() / n as f64;
                let var: f64 = flat.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (n.saturating_sub(1).max(1)) as f64;
                vec![var]
            } else {
                // np.cov(X, rowvar=False) equivalent
                let mut means = vec![0.0; p];
                for j in 0..p {
                    means[j] = (0..n).map(|i| flat[i * p + j]).sum::<f64>() / n as f64;
                }
                let mut c_out = vec![0.0; p * p];
                for i in 0..p {
                    for j in i..p {
                        let cov: f64 = (0..n)
                            .map(|r| (flat[r * p + i] - means[i]) * (flat[r * p + j] - means[j]))
                            .sum::<f64>()
                            / (n.saturating_sub(1).max(1)) as f64;
                        c_out[i * p + j] = cov;
                        c_out[j * p + i] = cov;
                    }
                }
                c_out
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "rank2d algorithm must be 'pearson', 'spearman', 'kendall', or 'covariance'; got '{other}'"
            )));
        }
    };

    // Build long-form output: (feature_x, feature_y, correlation)
    let mut feat_x = Vec::with_capacity(p * p);
    let mut feat_y = Vec::with_capacity(p * p);
    let mut corr = Vec::with_capacity(p * p);
    for i in 0..p {
        for j in 0..p {
            feat_x.push(col_names[i].clone());
            feat_y.push(col_names[j].clone());
            corr.push(c_flat[i * p + j]);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("feature_x", DataType::Utf8, false),
        Field::new("feature_y", DataType::Utf8, false),
        Field::new("correlation", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from(feat_x)) as ArrayRef,
        Arc::new(StringArray::from(feat_y)) as ArrayRef,
        f64_array(corr),
    ]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(batch))
}

// ---------------------------------------------------------------------------
// Distance-metric string parser (shared by silhouette, mds, calinski)
// ---------------------------------------------------------------------------

fn parse_distance_metric(s: &str) -> PyResult<DistanceMetric> {
    match s {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "cosine" => Ok(DistanceMetric::Cosine),
        "correlation" => Ok(DistanceMetric::Correlation),
        "chebyshev" => Ok(DistanceMetric::Chebyshev),
        other => Err(PyValueError::new_err(format!(
            "Unknown distance metric '{other}'; expected euclidean|manhattan|cosine|correlation|chebyshev"
        ))),
    }
}

// ---------------------------------------------------------------------------
// PCA (thin SVD via faer)
// ---------------------------------------------------------------------------

fn pca_svd(
    batch: &RecordBatch,
    n_components: Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (flat, n, p, _col_names) = batch_to_flat_f64(batch)?;
    if n == 0 || p == 0 {
        return Err(PyValueError::new_err("pca: input must have at least 1 row and 1 column"));
    }
    let k = n_components.unwrap_or_else(|| n.min(p)).min(n.min(p));
    if k == 0 {
        return Err(PyValueError::new_err("pca: n_components must be >= 1"));
    }

    let mut x_mat = linalg::mat_from_flat(&flat, n, p)?;
    for j in 0..p {
        let mean: f64 = (0..n).map(|i| x_mat[(i, j)]).sum::<f64>() / n as f64;
        for i in 0..n {
            x_mat[(i, j)] -= mean;
        }
    }

    let svd = x_mat.thin_svd().map_err(|e| PyValueError::new_err(format!("pca SVD failed: {e:?}")))?;
    let u = svd.U();
    let s_diag = svd.S();

    let s_vals: Vec<f64> = s_diag.column_vector().iter().copied().collect();
    let s_sq: Vec<f64> = s_vals.iter().map(|v| v * v).collect();
    let total_var: f64 = s_sq.iter().sum();

    let evr: Vec<f64> = if total_var > 0.0 {
        s_sq[..k].iter().map(|v| v / total_var).collect()
    } else {
        vec![0.0; k]
    };

    let mut scores = vec![vec![0.0; n]; k];
    for c in 0..k {
        let sv = s_vals[c];
        for i in 0..n {
            scores[c][i] = u[(i, c)] * sv;
        }
    }

    Ok((scores, evr))
}

#[pyfunction]
#[pyo3(signature = (x_table, n_components=None))]
pub fn pca_scores(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    n_components: Option<usize>,
) -> PyResult<PyRecordBatch> {
    let batch: RecordBatch = x_table.into();
    let (scores, _evr) = pca_svd(&batch, n_components)?;
    let k = scores.len();

    let fields: Vec<Field> = (0..k)
        .map(|i| Field::new(format!("dim_{i}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = scores.into_iter().map(|col| f64_array(col)).collect();
    let out = RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(out))
}

#[pyfunction]
#[pyo3(signature = (x_table, n_components=None))]
pub fn pca_variance(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    n_components: Option<usize>,
) -> PyResult<PyRecordBatch> {
    let batch: RecordBatch = x_table.into();
    let (_scores, evr) = pca_svd(&batch, n_components)?;
    let k = evr.len();

    let mut cum = Vec::with_capacity(k);
    let mut running = 0.0;
    for &v in &evr {
        running += v;
        cum.push(running);
    }

    let components: Vec<i64> = (1..=k as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("component", DataType::Int64, false),
        Field::new("explained_variance_ratio", DataType::Float64, false),
        Field::new("cumulative_variance_ratio", DataType::Float64, false),
    ]));
    let out = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(components)) as ArrayRef,
        f64_array(evr),
        f64_array(cum),
    ]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(out))
}

// ---------------------------------------------------------------------------
// Classical MDS
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (x_table, n_components, input_type="features", metric="euclidean"))]
pub fn mds_classical(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    n_components: usize,
    input_type: &str,
    metric: &str,
) -> PyResult<PyRecordBatch> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, feat, _col_names) = batch_to_flat_f64(&batch)?;
    if n < 2 {
        return Err(PyValueError::new_err("mds_classical: need at least 2 observations"));
    }
    if n_components == 0 {
        return Err(PyValueError::new_err("mds_classical: n_components must be >= 1"));
    }
    let is_distance = match input_type {
        "distance" => true,
        "features" => false,
        other => return Err(PyValueError::new_err(format!(
            "mds_classical: input_type must be 'features' or 'distance'; got '{other}'"
        ))),
    };

    let full_dist = if is_distance {
        if n != feat {
            return Err(PyValueError::new_err(
                "mds_classical: when is_distance=true, input must be a square n×n distance matrix"
            ));
        }
        flat
    } else {
        let dm = parse_distance_metric(metric)?;
        let condensed = crate::transform::linkage::condensed_distances(&flat, n, feat, dm)?;
        let mut full = vec![0.0; n * n];
        let mut idx = 0;
        for i in 0..n - 1 {
            for j in i + 1..n {
                full[i * n + j] = condensed[idx];
                full[j * n + i] = condensed[idx];
                idx += 1;
            }
        }
        full
    };

    // Double-centering: B_ij = -0.5 * (D_ij^2 - row_mean_i - col_mean_j + grand_mean)
    let mut d_sq = vec![0.0; n * n];
    for i in 0..n * n {
        d_sq[i] = full_dist[i] * full_dist[i];
    }

    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += d_sq[i * n + j];
        }
        row_means[i] = s / n as f64;
    }
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s += d_sq[i * n + j];
        }
        col_means[j] = s / n as f64;
    }
    for &v in &d_sq {
        grand_mean += v;
    }
    grand_mean /= (n * n) as f64;

    let mut b_flat = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b_flat[i * n + j] = -0.5 * (d_sq[i * n + j] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    let b_mat = linalg::mat_from_flat(&b_flat, n, n)?;
    let eigen = b_mat.self_adjoint_eigen(Side::Lower)
        .map_err(|e| PyValueError::new_err(format!("mds_classical eigendecomposition failed: {e:?}")))?;

    let eigvals: Vec<f64> = eigen.S().column_vector().iter().copied().collect();
    let eigvecs = eigen.U();

    // faer returns eigenvalues in ascending order; take the last n_components (largest)
    let nc = n_components.min(n);
    let mut coords = vec![vec![0.0; n]; nc];
    for c in 0..nc {
        let eig_idx = n - 1 - c; // largest first
        let lam = eigvals[eig_idx].max(0.0).sqrt();
        for i in 0..n {
            coords[c][i] = eigvecs[(i, eig_idx)] * lam;
        }
    }

    let fields: Vec<Field> = (0..nc)
        .map(|i| Field::new(format!("dim_{i}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = coords.into_iter().map(|col| f64_array(col)).collect();
    let out = RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(out))
}

// ---------------------------------------------------------------------------
// Silhouette
// ---------------------------------------------------------------------------

fn silhouette_samples_vec(
    flat: &[f64],
    n: usize,
    feat: usize,
    labels: &[i64],
    metric: DistanceMetric,
) -> PyResult<Vec<f64>> {
    if n == 0 {
        return Ok(vec![]);
    }

    // Compute full pairwise distance matrix
    let condensed = crate::transform::linkage::condensed_distances(flat, n, feat, metric)?;
    let mut dist = vec![0.0; n * n];
    let mut idx = 0;
    for i in 0..n - 1 {
        for j in i + 1..n {
            dist[i * n + j] = condensed[idx];
            dist[j * n + i] = condensed[idx];
            idx += 1;
        }
    }

    // Collect unique cluster labels and build cluster membership
    let mut unique_labels: Vec<i64> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Ok(vec![0.0; n]);
    }

    // Map label -> cluster index
    let label_to_idx: std::collections::HashMap<i64, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(idx, &lab)| (lab, idx))
        .collect();

    // Build cluster membership lists
    let mut cluster_members: Vec<Vec<usize>> = vec![vec![]; n_clusters];
    for (i, &lab) in labels.iter().enumerate() {
        cluster_members[label_to_idx[&lab]].push(i);
    }

    let mut silhouette = vec![0.0; n];
    for i in 0..n {
        let ci = label_to_idx[&labels[i]];
        let cluster_size = cluster_members[ci].len();

        // a(i) = mean intra-cluster distance
        let a_i = if cluster_size <= 1 {
            0.0
        } else {
            let sum: f64 = cluster_members[ci].iter()
                .filter(|&&j| j != i)
                .map(|&j| dist[i * n + j])
                .sum();
            sum / (cluster_size - 1) as f64
        };

        // b(i) = min mean inter-cluster distance
        let mut b_i = f64::INFINITY;
        for (c_idx, members) in cluster_members.iter().enumerate() {
            if c_idx == ci || members.is_empty() {
                continue;
            }
            let sum: f64 = members.iter().map(|&j| dist[i * n + j]).sum();
            let mean_d = sum / members.len() as f64;
            if mean_d < b_i {
                b_i = mean_d;
            }
        }

        let denom = a_i.max(b_i);
        silhouette[i] = if denom > 0.0 {
            (b_i - a_i) / denom
        } else {
            0.0
        };
    }

    Ok(silhouette)
}

fn extract_labels(labels: &PyArray) -> PyResult<Vec<i64>> {
    let arr = labels.array();
    if let Some(i64_arr) = arr.as_any().downcast_ref::<Int64Array>() {
        return Ok(i64_arr.values().to_vec());
    }
    if let Some(f64_arr) = arr.as_any().downcast_ref::<Float64Array>() {
        return Ok(f64_arr.values().iter().map(|v| *v as i64).collect());
    }
    Err(PyValueError::new_err("labels must be Int64 or Float64"))
}

#[pyfunction]
#[pyo3(signature = (x_table, labels, metric="euclidean"))]
pub fn silhouette_samples(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    labels: PyArray,
    metric: &str,
) -> PyResult<PyArray> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, feat, _col_names) = batch_to_flat_f64(&batch)?;
    let lab = extract_labels(&labels)?;
    if lab.len() != n {
        return Err(PyValueError::new_err("labels length must match X row count"));
    }
    let dm = parse_distance_metric(metric)?;
    let sv = silhouette_samples_vec(&flat, n, feat, &lab, dm)?;
    Ok(PyArray::from_array_ref(Arc::new(Float64Array::from(sv))))
}

#[pyfunction]
#[pyo3(signature = (x_table, labels, metric="euclidean"))]
pub fn silhouette_score(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    labels: PyArray,
    metric: &str,
) -> PyResult<f64> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, feat, _col_names) = batch_to_flat_f64(&batch)?;
    let lab = extract_labels(&labels)?;
    if lab.len() != n {
        return Err(PyValueError::new_err("labels length must match X row count"));
    }
    let dm = parse_distance_metric(metric)?;
    let sv = silhouette_samples_vec(&flat, n, feat, &lab, dm)?;
    if sv.is_empty() {
        return Ok(0.0);
    }
    let mean = sv.iter().sum::<f64>() / sv.len() as f64;
    Ok(mean)
}

// ---------------------------------------------------------------------------
// Calinski-Harabasz score
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn calinski_harabasz_score(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    labels: PyArray,
) -> PyResult<f64> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, p, _col_names) = batch_to_flat_f64(&batch)?;
    let lab = extract_labels(&labels)?;
    if lab.len() != n {
        return Err(PyValueError::new_err("labels length must match X row count"));
    }

    // Overall centroid
    let mut x_bar = vec![0.0; p];
    for j in 0..p {
        let sum: f64 = (0..n).map(|i| flat[i * p + j]).sum();
        x_bar[j] = sum / n as f64;
    }

    // Unique clusters
    let mut unique_labels: Vec<i64> = lab.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();

    if k < 2 || n <= k {
        return Err(PyValueError::new_err(
            "calinski_harabasz_score requires at least 2 clusters and n > k"
        ));
    }

    let label_to_idx: std::collections::HashMap<i64, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(idx, &lab)| (lab, idx))
        .collect();

    // Build cluster membership
    let mut cluster_members: Vec<Vec<usize>> = vec![vec![]; k];
    for (i, &lab_val) in lab.iter().enumerate() {
        cluster_members[label_to_idx[&lab_val]].push(i);
    }

    let mut between_scatter = 0.0;
    let mut within_scatter = 0.0;

    for members in &cluster_members {
        let n_c = members.len() as f64;
        if members.is_empty() {
            continue;
        }

        // Cluster centroid
        let mut mu_c = vec![0.0; p];
        for &i in members {
            for j in 0..p {
                mu_c[j] += flat[i * p + j];
            }
        }
        for j in 0..p {
            mu_c[j] /= n_c;
        }

        // Between scatter: n_c * ||mu_c - x_bar||^2
        let mut dist_sq = 0.0;
        for j in 0..p {
            dist_sq += (mu_c[j] - x_bar[j]).powi(2);
        }
        between_scatter += n_c * dist_sq;

        // Within scatter: sum_i ||x_i - mu_c||^2
        for &i in members {
            let mut d2 = 0.0;
            for j in 0..p {
                d2 += (flat[i * p + j] - mu_c[j]).powi(2);
            }
            within_scatter += d2;
        }
    }

    if within_scatter == 0.0 {
        return Ok(f64::INFINITY);
    }

    let ch = (between_scatter / (k - 1) as f64) / (within_scatter / (n - k) as f64);
    Ok(ch)
}

// ---------------------------------------------------------------------------
// t-SNE embedding via manifolds-rs
// ---------------------------------------------------------------------------

fn flat_to_faer_compat(flat: &[f64], n: usize, p: usize) -> faer_compat::Mat<f64> {
    let mut mat = faer_compat::Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            mat[(i, j)] = flat[i * p + j];
        }
    }
    mat
}

#[pyfunction]
#[pyo3(signature = (x_table, n_components, seed, perplexity=None, learning_rate=None, n_iter=None))]
pub fn tsne_embedding(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    n_components: usize,
    seed: u64,
    perplexity: Option<f64>,
    learning_rate: Option<f64>,
    n_iter: Option<usize>,
) -> PyResult<PyRecordBatch> {
    if n_components != 2 {
        return Err(PyValueError::new_err(
            "tsne_embedding: manifolds-rs only supports n_components=2"
        ));
    }
    let batch: RecordBatch = x_table.into();
    let (flat, n, p, _col_names) = batch_to_flat_f64(&batch)?;
    if n < 4 {
        return Err(PyValueError::new_err(
            "tsne_embedding: need at least 4 observations"
        ));
    }

    let mat = flat_to_faer_compat(&flat, n, p);

    let params = manifolds_rs::TsneParams::new(
        Some(n_components),
        perplexity,
        None, // init_range
        learning_rate,
        n_iter,
        None, // ann_type (default: kmknn)
        None, // theta (default: 0.5)
        None, // n_interp_points
    );

    let result = manifolds_rs::tsne(
        mat.as_ref(),
        None, // no precomputed kNN
        &params,
        "bh",
        seed as usize,
        false, // verbose
    ).map_err(|e| PyValueError::new_err(format!("tsne_embedding failed: {e}")))?;

    // result is Vec<Vec<f64>> with outer len = n_components, inner len = n_samples
    embedding_to_record_batch(&result, n_components)
}

// ---------------------------------------------------------------------------
// UMAP embedding via manifolds-rs
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (x_table, n_components, seed, n_neighbors=None, min_dist=None, n_epochs=None))]
pub fn umap_embedding(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    n_components: usize,
    seed: u64,
    n_neighbors: Option<usize>,
    min_dist: Option<f64>,
    n_epochs: Option<usize>,
) -> PyResult<PyRecordBatch> {
    let batch: RecordBatch = x_table.into();
    let (flat, n, p, _col_names) = batch_to_flat_f64(&batch)?;
    if n < 4 {
        return Err(PyValueError::new_err(
            "umap_embedding: need at least 4 observations"
        ));
    }

    let mat = flat_to_faer_compat(&flat, n, p);

    let min_dist_val = min_dist.unwrap_or(0.1);
    let spread = 1.0;
    let params = manifolds_rs::UmapParams::new(
        Some(n_components),
        n_neighbors,
        None, // optimiser (default: adam_parallel)
        None, // ann_type (default: kmknn)
        None, // initialisation (default: spectral)
        None, // init_range
        None, // nn_params
        Some(manifolds_rs::prelude::UmapOptimParams::from_min_dist_spread(
            min_dist_val,
            spread,
            None, // lr
            None, // gamma
            n_epochs,
            None, // neg_sample_rate
            None, // beta1
            None, // beta2
            None, // eps
        )),
        None, // umap_graph_params
        None, // randomised
    );

    let result = manifolds_rs::umap(
        mat.as_ref(),
        None, // no precomputed kNN
        &params,
        seed as usize,
        false, // verbose
    ).map_err(|e| PyValueError::new_err(format!("umap_embedding failed: {e}")))?;

    // result is Vec<Vec<f64>> with outer len = n_components, inner len = n_samples
    embedding_to_record_batch(&result, n_components)
}

/// Convert manifolds-rs output (Vec<Vec<f64>> with [n_dim][n_samples]) to a
/// RecordBatch with columns dim_0, dim_1, ...
fn embedding_to_record_batch(
    embedding: &[Vec<f64>],
    n_components: usize,
) -> PyResult<PyRecordBatch> {
    let nc = embedding.len().min(n_components);
    let fields: Vec<Field> = (0..nc)
        .map(|i| Field::new(format!("dim_{i}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = embedding[..nc]
        .iter()
        .map(|col| f64_array(col.clone()))
        .collect();
    let out = RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(out))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Test coverage lives in `tests.rs` (precedent: `render/scale_resolve/tests.rs`,
// `projection/mod.rs`) — the module grew too large for an inline
// `mod tests { ... }` block once the mirror-coupled coverage from the R1
// bug-hunt suites was relocated in-crate to exercise these real stats
// transforms directly instead of a duplicated-formula copy.
#[cfg(test)]
mod tests;
