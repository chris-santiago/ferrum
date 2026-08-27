//! Phase 10 model-diagnostics — sole Rust contribution.
//!
//! `kendall_tau_b` implements Knight's O(n log n) merge-sort variant
//! for Kendall's tau-b rank correlation. Used by
//! `ModelSource.rank2d(algorithm="kendall")` when n is large enough
//! that the O(n²) pure-Python loop is too slow.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_arrow::{PyArray, PyRecordBatch};

#[derive(Debug, Clone, Copy)]
pub struct KendallResult {
    pub tau: f64,
    pub n_concordant: u64,
    pub n_discordant: u64,
    pub n_tied_x: u64,
    pub n_tied_y: u64,
    pub n_tied_both: u64,
}

/// Stable-sort `idx` by `x[idx]`, counting tied x pairs.
fn sort_by_key_count_ties(x: &[f64], idx: &mut [usize]) -> u64 {
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ties: u64 = 0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let run = (j - i) as u64;
        if run > 1 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

/// Merge-sort by `y[idx[..]]`, counting inversions (= n_discordant).
fn count_inversions(y: &[f64], idx: &mut [usize]) -> u64 {
    let mut buf = vec![0usize; idx.len()];
    merge_sort(y, idx, &mut buf, 0, idx.len())
}

fn merge_sort(y: &[f64], idx: &mut [usize], buf: &mut [usize], lo: usize, hi: usize) -> u64 {
    if hi - lo <= 1 {
        return 0;
    }
    let mid = (lo + hi) / 2;
    let l = merge_sort(y, idx, buf, lo, mid);
    let r = merge_sort(y, idx, buf, mid, hi);
    let m = merge(y, idx, buf, lo, mid, hi);
    l + r + m
}

fn merge(y: &[f64], idx: &mut [usize], buf: &mut [usize], lo: usize, mid: usize, hi: usize) -> u64 {
    let mut i = lo;
    let mut j = mid;
    let mut k = lo;
    let mut inv: u64 = 0;
    while i < mid && j < hi {
        if y[idx[i]] <= y[idx[j]] {
            buf[k] = idx[i];
            i += 1;
        } else {
            buf[k] = idx[j];
            inv += (mid - i) as u64;
            j += 1;
        }
        k += 1;
    }
    while i < mid {
        buf[k] = idx[i];
        i += 1;
        k += 1;
    }
    while j < hi {
        buf[k] = idx[j];
        j += 1;
        k += 1;
    }
    idx[lo..hi].copy_from_slice(&buf[lo..hi]);
    inv
}

/// Count y-ties given indices already sorted by y.
fn count_y_ties_after_sort(y: &[f64], idx: &[usize]) -> u64 {
    let mut ties: u64 = 0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && y[idx[j]] == y[idx[i]] {
            j += 1;
        }
        let run = (j - i) as u64;
        if run > 1 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

/// Count joint (x, y) ties. `idx` must be sorted by x then by y within
/// each x-tie group.
fn count_xy_ties(x: &[f64], y: &[f64], idx: &[usize]) -> u64 {
    let mut ties: u64 = 0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && x[idx[j]] == x[idx[i]] && y[idx[j]] == y[idx[i]] {
            j += 1;
        }
        let run = (j - i) as u64;
        if run > 1 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

pub fn kendall_tau_b(x: &[f64], y: &[f64]) -> KendallResult {
    assert_eq!(x.len(), y.len(), "x and y must be the same length");
    let n = x.len() as u64;
    if n < 2 {
        return KendallResult {
            tau: f64::NAN,
            n_concordant: 0,
            n_discordant: 0,
            n_tied_x: 0,
            n_tied_y: 0,
            n_tied_both: 0,
        };
    }

    let n0 = n * (n - 1) / 2;

    // Step 1: sort by x (stable), counting tied x pairs.
    let mut idx: Vec<usize> = (0..x.len()).collect();
    let n_tied_x = sort_by_key_count_ties(x, &mut idx);

    // Within tied-x runs, sort by y so count_xy_ties sees joint ties as
    // contiguous runs.
    {
        let mut i = 0;
        while i < idx.len() {
            let mut j = i + 1;
            while j < idx.len() && x[idx[j]] == x[idx[i]] {
                j += 1;
            }
            idx[i..j].sort_by(|&a, &b| {
                y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
            i = j;
        }
    }
    let n_tied_both = count_xy_ties(x, y, &idx);

    // Step 2: count discordant pairs via merge-sort on y.
    let n_discordant = count_inversions(y, &mut idx);

    // Step 3: count y-ties (idx is now sorted by y).
    let n_tied_y = count_y_ties_after_sort(y, &idx);

    // Concordant pairs: n0 = C + D + T_x + T_y - T_both
    //   so C = n0 - D - T_x - T_y + T_both
    let n_concordant = (n0 as i128
        - n_discordant as i128
        - n_tied_x as i128
        - n_tied_y as i128
        + n_tied_both as i128)
        .max(0) as u64;

    // tau-b = (C - D) / sqrt((n0 - T_x) * (n0 - T_y))
    let denom = (((n0 - n_tied_x) as f64) * ((n0 - n_tied_y) as f64)).sqrt();
    let tau = if denom > 0.0 {
        (n_concordant as f64 - n_discordant as f64) / denom
    } else {
        f64::NAN
    };

    KendallResult {
        tau,
        n_concordant,
        n_discordant,
        n_tied_x,
        n_tied_y,
        n_tied_both,
    }
}

#[pyfunction]
#[pyo3(name = "kendall_tau_b")]
pub fn py_kendall_tau_b<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must be the same length"));
    }
    let r = kendall_tau_b(&x, &y);
    let d = PyDict::new(py);
    d.set_item("tau", r.tau)?;
    d.set_item("n_concordant", r.n_concordant)?;
    d.set_item("n_discordant", r.n_discordant)?;
    d.set_item("n_tied_x", r.n_tied_x)?;
    d.set_item("n_tied_y", r.n_tied_y)?;
    d.set_item("n_tied_both", r.n_tied_both)?;
    Ok(d)
}

// ===========================================================================
// Classification diagnostic curve kernels (sklearn-parity)
//
// These reproduce scikit-learn's exact conventions so existing golden SVGs do
// not shift. They emit the numeric core only; the Python adapter attaches class
// labels, broadcasts scalar metrics, and formats values.
// ===========================================================================

// --- Arrow extraction helpers ------------------------------------------------

/// Reject Arrow null entries (validity-bitmap nulls).
///
/// Returns an error message string so the check can be tested without a live
/// Python interpreter. Call sites convert it to `PyValueError` via
/// `.map_err(PyValueError::new_err)`.
///
/// NaN is a valid f64 *value* and is never counted by `null_count`, so a
/// `Float64Array` containing NaN has `null_count == 0` and passes this check.
/// Only genuine Arrow nulls (entries absent from the validity bitmap) are
/// rejected.
fn check_no_nulls(null_count: usize, name: &str) -> Result<(), String> {
    if null_count > 0 {
        return Err(format!("{name} must not contain nulls"));
    }
    Ok(())
}

fn as_f64_slice<'a>(arr: &'a PyArray, name: &str) -> PyResult<&'a [f64]> {
    let array_ref = arr.array();
    check_no_nulls(array_ref.null_count(), name).map_err(PyValueError::new_err)?;
    array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .map(|a| a.values().as_ref())
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be Float64")))
}

fn as_i64_slice<'a>(arr: &'a PyArray, name: &str) -> PyResult<&'a [i64]> {
    let array_ref = arr.array();
    check_no_nulls(array_ref.null_count(), name).map_err(PyValueError::new_err)?;
    array_ref
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|a| a.values().as_ref())
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be Int64")))
}

fn check_equal_len(a: usize, b: usize, an: &str, bn: &str) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "{an} and {bn} must be the same length ({a} != {b})"
        )));
    }
    Ok(())
}

fn record_batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> PyResult<PyRecordBatch> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRecordBatch::new(batch))
}

// --- Binary classifier curve (sklearn `_binary_clf_curve`) -------------------

/// Sweep thresholds in descending-score order, aggregating ties.
///
/// Mirrors `sklearn.metrics._ranking._binary_clf_curve`: sort by descending
/// score, keep only the indices where the score changes (tie aggregation), and
/// return cumulative true/false positive counts plus the distinct thresholds.
/// `fps`, `tps`, and `thresholds` are equal length.
fn binary_clf_curve(y_true: &[f64], y_score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y_true.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Descending by score; stable on ties so the result is deterministic.
    order.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Cumulative true positives across the sorted sweep.
    let mut cum_tps = vec![0.0; n];
    let mut running = 0.0;
    for (k, &idx) in order.iter().enumerate() {
        running += y_true[idx];
        cum_tps[k] = running;
    }

    // Distinct-value indices: the last index of each run of tied scores.
    let mut tps = Vec::new();
    let mut fps = Vec::new();
    let mut thresholds = Vec::new();
    for k in 0..n {
        let is_last = k + 1 == n || y_score[order[k + 1]] != y_score[order[k]];
        if is_last {
            let tp = cum_tps[k];
            // fps = (count so far) - tps = (k + 1) - tp
            let fp = (k + 1) as f64 - tp;
            tps.push(tp);
            fps.push(fp);
            thresholds.push(y_score[order[k]]);
        }
    }
    (fps, tps, thresholds)
}

// --- ROC curve ---------------------------------------------------------------

/// Pure-Rust ROC curve core: returns (fpr, tpr, thresholds), each equal length,
/// with the leading (0, 0) origin and +inf threshold sentinel sklearn emits.
fn roc_curve_core(
    y_true: &[f64],
    y_score: &[f64],
    drop_intermediate: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut fps, mut tps, mut thresholds) = binary_clf_curve(y_true, y_score);

    // sklearn: drop collinear interior points, keeping endpoints plus any point
    // where the second difference of fps or tps is nonzero (a slope change).
    if drop_intermediate && fps.len() > 2 {
        let m = fps.len();
        let mut keep = vec![false; m];
        keep[0] = true;
        keep[m - 1] = true;
        for i in 1..m - 1 {
            let d2_fps = (fps[i + 1] - fps[i]) - (fps[i] - fps[i - 1]);
            let d2_tps = (tps[i + 1] - tps[i]) - (tps[i] - tps[i - 1]);
            if d2_fps != 0.0 || d2_tps != 0.0 {
                keep[i] = true;
            }
        }
        let filter = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&keep)
                .filter_map(|(x, &k)| k.then_some(*x))
                .collect()
        };
        fps = filter(&fps);
        tps = filter(&tps);
        thresholds = filter(&thresholds);
    }

    // Prepend the (0, 0) origin with the leading +inf sentinel sklearn emits.
    let fp_last = *fps.last().unwrap_or(&0.0);
    let tp_last = *tps.last().unwrap_or(&0.0);

    let mut fpr = Vec::with_capacity(fps.len() + 1);
    let mut tpr = Vec::with_capacity(tps.len() + 1);
    let mut thr = Vec::with_capacity(thresholds.len() + 1);
    // sklearn: when all-positive (fp_last == 0) every fpr value is NaN because
    // it is computed as fps / fps[-1] = x / 0.  The leading origin slot follows
    // the same division, so it is also NaN.  Symmetric reasoning applies to tpr
    // when all-negative (tp_last == 0).  We implement this via the same division
    // so the 0/0 → NaN semantic falls out naturally, matching sklearn 1.7.2.
    fpr.push(0.0_f64 / fp_last); // NaN when fp_last == 0 (all-positive input)
    tpr.push(0.0_f64 / tp_last); // NaN when tp_last == 0 (all-negative input)
    thr.push(f64::INFINITY);
    for &f in &fps {
        fpr.push(f / fp_last);
    }
    for &t in &tps {
        tpr.push(t / tp_last);
    }
    thr.extend_from_slice(&thresholds);

    (fpr, tpr, thr)
}

#[pyfunction]
pub fn roc_curve_kernel(
    _py: Python<'_>,
    y_true: PyArray,
    y_score: PyArray,
    drop_intermediate: bool,
) -> PyResult<PyRecordBatch> {
    let yt = as_f64_slice(&y_true, "y_true")?;
    let ys = as_f64_slice(&y_score, "y_score")?;
    check_equal_len(yt.len(), ys.len(), "y_true", "y_score")?;

    let (fpr, tpr, thr) = roc_curve_core(yt, ys, drop_intermediate);

    record_batch(
        vec![
            Field::new("fpr", DataType::Float64, false),
            Field::new("tpr", DataType::Float64, false),
            Field::new("threshold", DataType::Float64, false),
        ],
        vec![
            Arc::new(Float64Array::from(fpr)),
            Arc::new(Float64Array::from(tpr)),
            Arc::new(Float64Array::from(thr)),
        ],
    )
}

/// Trapezoidal area under the (fpr, tpr) curve; NaN if only one class present.
fn roc_auc_core(y_true: &[f64], y_score: &[f64]) -> f64 {
    let (fps, tps, _) = binary_clf_curve(y_true, y_score);
    let fp_last = *fps.last().unwrap_or(&0.0);
    let tp_last = *tps.last().unwrap_or(&0.0);
    if fp_last <= 0.0 || tp_last <= 0.0 {
        // Only one class present; AUC is undefined.
        return f64::NAN;
    }

    // Trapezoidal area under the (fpr, tpr) curve, including the (0,0) origin.
    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    for &f in &fps {
        fpr.push(f / fp_last);
    }
    for &t in &tps {
        tpr.push(t / tp_last);
    }
    let mut area = 0.0;
    for i in 1..fpr.len() {
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0;
    }
    area
}

#[pyfunction]
pub fn roc_auc(_py: Python<'_>, y_true: PyArray, y_score: PyArray) -> PyResult<f64> {
    let yt = as_f64_slice(&y_true, "y_true")?;
    let ys = as_f64_slice(&y_score, "y_score")?;
    check_equal_len(yt.len(), ys.len(), "y_true", "y_score")?;
    Ok(roc_auc_core(yt, ys))
}

// --- Precision-recall curve --------------------------------------------------

/// Reversed precision and recall arrays (length `m`, no trailing endpoint).
///
/// Iterates `k` in `(0..m).rev()` to match sklearn's `precision_recall_curve`
/// ordering (highest threshold first).
///
/// - `precision[k] = tps[k] / (tps[k] + fps[k])`, or `1.0` when the denominator
///   is zero (no predictions above the threshold).
/// - `recall[k]    = tps[k] / total_pos`,          or `1.0` when `total_pos == 0`
///   (callers that guard against the all-negative case before calling may rely on
///   the guarded form; both branches are numerically correct).
fn reversed_pr(fps: &[f64], tps: &[f64], total_pos: f64) -> (Vec<f64>, Vec<f64>) {
    let m = tps.len();
    let mut precision = Vec::with_capacity(m);
    let mut recall = Vec::with_capacity(m);
    for k in (0..m).rev() {
        let denom = tps[k] + fps[k];
        precision.push(if denom > 0.0 { tps[k] / denom } else { 1.0 });
        recall.push(if total_pos > 0.0 { tps[k] / total_pos } else { 1.0 });
    }
    (precision, recall)
}

/// Pure-Rust PR curve core: reversed precision/recall plus the (1, 0) endpoint.
/// `threshold` is equal length to precision/recall, with NaN in the final slot
/// (sklearn's threshold array is one element shorter).
fn pr_curve_core(y_true: &[f64], y_score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (fps, tps, thresholds) = binary_clf_curve(y_true, y_score);
    let total_pos = *tps.last().unwrap_or(&0.0);

    // sklearn precision_recall_curve: precision = tps / (tps + fps),
    // recall = tps / tps[-1]; reverse, then append the (1, 0) endpoint.
    let m = tps.len();
    let (mut precision, mut recall) = reversed_pr(&fps, &tps, total_pos);

    // Reversed threshold column (one-to-one with precision/recall).
    let mut threshold: Vec<f64> = thresholds.iter().rev().cloned().collect();
    threshold.reserve(1);

    // Trailing endpoint (precision=1, recall=0); its threshold slot is NaN
    // because sklearn's threshold array is one element shorter.
    precision.push(1.0);
    recall.push(0.0);
    threshold.push(f64::NAN);

    debug_assert_eq!(precision.len(), m + 1);
    debug_assert_eq!(recall.len(), m + 1);
    debug_assert_eq!(threshold.len(), m + 1);

    (precision, recall, threshold)
}

#[pyfunction]
pub fn pr_curve_kernel(
    _py: Python<'_>,
    y_true: PyArray,
    y_score: PyArray,
) -> PyResult<PyRecordBatch> {
    let yt = as_f64_slice(&y_true, "y_true")?;
    let ys = as_f64_slice(&y_score, "y_score")?;
    check_equal_len(yt.len(), ys.len(), "y_true", "y_score")?;

    let (precision, recall, threshold) = pr_curve_core(yt, ys);

    record_batch(
        vec![
            Field::new("precision", DataType::Float64, false),
            Field::new("recall", DataType::Float64, false),
            Field::new("threshold", DataType::Float64, false),
        ],
        vec![
            Arc::new(Float64Array::from(precision)),
            Arc::new(Float64Array::from(recall)),
            Arc::new(Float64Array::from(threshold)),
        ],
    )
}

/// Step-function average precision: Σ (Rₙ − Rₙ₋₁)·Pₙ, NOT trapezoidal.
fn average_precision_core(y_true: &[f64], y_score: &[f64]) -> f64 {
    let (fps, tps, _) = binary_clf_curve(y_true, y_score);
    let total_pos = *tps.last().unwrap_or(&0.0);
    if total_pos <= 0.0 {
        // sklearn.metrics.average_precision_score returns 0.0 (with a warning)
        // when y_true has no positive class. Byte-parity requires 0.0, not NaN.
        return 0.0;
    }

    // Reversed precision/recall with the (P=1, R=0) endpoint, matching
    // sklearn's precision_recall_curve.  The `total_pos > 0` guard above
    // ensures the `else 1.0` recall branch in `reversed_pr` is never reached.
    let (mut precision, mut recall) = reversed_pr(&fps, &tps, total_pos);
    precision.push(1.0);
    recall.push(0.0);

    let mut ap = 0.0;
    for i in 0..precision.len() - 1 {
        ap += (recall[i] - recall[i + 1]) * precision[i];
    }
    ap
}

#[pyfunction]
pub fn average_precision(
    _py: Python<'_>,
    y_true: PyArray,
    y_score: PyArray,
) -> PyResult<f64> {
    let yt = as_f64_slice(&y_true, "y_true")?;
    let ys = as_f64_slice(&y_score, "y_score")?;
    check_equal_len(yt.len(), ys.len(), "y_true", "y_score")?;
    Ok(average_precision_core(yt, ys))
}

// --- Calibration curve -------------------------------------------------------

fn quantile_edges(y_prob: &[f64], n_bins: usize) -> Vec<f64> {
    // np.percentile(y_prob, linspace(0,100,n_bins+1)) with linear interpolation.
    let mut sorted = y_prob.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mut edges = Vec::with_capacity(n_bins + 1);
    if n == 0 {
        return vec![0.0; n_bins + 1];
    }
    for k in 0..=n_bins {
        let q = k as f64 / n_bins as f64;
        // numpy "linear" percentile: rank = q * (n - 1)
        let rank = q * (n - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        let frac = rank - lo as f64;
        let val = sorted[lo] + (sorted[hi.min(n - 1)] - sorted[lo]) * frac;
        edges.push(val);
    }
    edges
}

/// Pure-Rust calibration core: bins by strategy, drops empty bins, returns
/// (mean_predicted, fraction_positive, count) over the surviving bins.
fn calibration_core(
    y_true: &[f64],
    y_prob: &[f64],
    n_bins: usize,
    strategy: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i64>)> {
    let edges: Vec<f64> = match strategy {
        "uniform" => (0..=n_bins).map(|k| k as f64 / n_bins as f64).collect(),
        "quantile" => quantile_edges(y_prob, n_bins),
        other => {
            return Err(PyValueError::new_err(format!(
                "strategy must be 'uniform' or 'quantile'; got '{other}'"
            )))
        }
    };

    // sklearn: binids = searchsorted(edges[1:-1], y_prob) (side="left").
    let inner = &edges[1..edges.len() - 1];
    let mut sum_prob = vec![0.0; n_bins];
    let mut sum_true = vec![0.0; n_bins];
    let mut count = vec![0i64; n_bins];
    for (&p, &t) in y_prob.iter().zip(y_true.iter()) {
        // searchsorted left: first index where inner[idx] >= p.
        let bin = inner.partition_point(|&e| e < p).min(n_bins - 1);
        sum_prob[bin] += p;
        sum_true[bin] += t;
        count[bin] += 1;
    }

    // Drop empty bins, preserving ascending bin order.
    let mut mean_predicted = Vec::new();
    let mut fraction_positive = Vec::new();
    let mut counts = Vec::new();
    for b in 0..n_bins {
        if count[b] != 0 {
            let c = count[b] as f64;
            mean_predicted.push(sum_prob[b] / c);
            fraction_positive.push(sum_true[b] / c);
            counts.push(count[b]);
        }
    }
    Ok((mean_predicted, fraction_positive, counts))
}

#[pyfunction]
pub fn calibration_kernel(
    _py: Python<'_>,
    y_true: PyArray,
    y_prob: PyArray,
    n_bins: u32,
    strategy: &str,
) -> PyResult<PyRecordBatch> {
    let yt = as_f64_slice(&y_true, "y_true")?;
    let yp = as_f64_slice(&y_prob, "y_prob")?;
    check_equal_len(yt.len(), yp.len(), "y_true", "y_prob")?;
    if n_bins == 0 {
        return Err(PyValueError::new_err("n_bins must be >= 1"));
    }

    let (mean_predicted, fraction_positive, counts) =
        calibration_core(yt, yp, n_bins as usize, strategy)?;

    record_batch(
        vec![
            Field::new("mean_predicted", DataType::Float64, false),
            Field::new("fraction_positive", DataType::Float64, false),
            Field::new("count", DataType::Int64, false),
        ],
        vec![
            Arc::new(Float64Array::from(mean_predicted)),
            Arc::new(Float64Array::from(fraction_positive)),
            Arc::new(Int64Array::from(counts)),
        ],
    )
}

// --- Confusion matrix --------------------------------------------------------

/// Pure-Rust confusion-matrix core: mirrors sklearn's `confusion_matrix`.
///
/// `y_true` and `y_pred` are integer class labels; `labels` determines which
/// classes appear in the matrix and their row/column order. Pairs where either
/// element is absent from `labels` are silently ignored, matching sklearn.
///
/// Returns the L×L confusion matrix as a flat row-major `Vec<f64>` of length
/// `labels.len() * labels.len()`, after optional normalization.
///
/// `normalize` must be one of `""` (no normalization), `"true"` (normalize over
/// actual/row), `"pred"` (normalize over predicted/column), or `"all"` (normalize
/// over all entries).
fn confusion_matrix_core(
    y_true: &[i64],
    y_pred: &[i64],
    labels: &[i64],
    normalize: &str,
) -> PyResult<Vec<f64>> {
    let l = labels.len();
    let label_idx: std::collections::HashMap<i64, usize> =
        labels.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    // Counts: cm[i][j] = #(true == labels[i] & pred == labels[j]).
    // Pairs whose label is not in `labels` are ignored, matching sklearn.
    let mut cm = vec![0.0; l * l];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if let (Some(&i), Some(&j)) = (label_idx.get(&t), label_idx.get(&p)) {
            cm[i * l + j] += 1.0;
        }
    }

    match normalize {
        "" => {}
        "true" => {
            for i in 0..l {
                let row_sum: f64 = (0..l).map(|j| cm[i * l + j]).sum();
                if row_sum != 0.0 {
                    for j in 0..l {
                        cm[i * l + j] /= row_sum;
                    }
                }
            }
        }
        "pred" => {
            for j in 0..l {
                let col_sum: f64 = (0..l).map(|i| cm[i * l + j]).sum();
                if col_sum != 0.0 {
                    for i in 0..l {
                        cm[i * l + j] /= col_sum;
                    }
                }
            }
        }
        "all" => {
            let total: f64 = cm.iter().sum();
            if total != 0.0 {
                for v in cm.iter_mut() {
                    *v /= total;
                }
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "normalize must be '', 'true', 'pred', or 'all'; got '{other}'"
            )))
        }
    }

    Ok(cm)
}

#[pyfunction]
pub fn confusion_kernel(
    _py: Python<'_>,
    y_true: PyArray,
    y_pred: PyArray,
    labels: PyArray,
    normalize: &str,
) -> PyResult<PyRecordBatch> {
    let yt = as_i64_slice(&y_true, "y_true")?;
    let yp = as_i64_slice(&y_pred, "y_pred")?;
    let labs = as_i64_slice(&labels, "labels")?;
    check_equal_len(yt.len(), yp.len(), "y_true", "y_pred")?;

    let l = labs.len();
    let cm = confusion_matrix_core(yt, yp, labs, normalize)?;

    // Dense L×L in row-major order.
    let mut row = Vec::with_capacity(l * l);
    let mut col = Vec::with_capacity(l * l);
    let mut value = Vec::with_capacity(l * l);
    for i in 0..l {
        for j in 0..l {
            row.push(i as i64);
            col.push(j as i64);
            value.push(cm[i * l + j]);
        }
    }

    record_batch(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ],
        vec![
            Arc::new(Int64Array::from(row)),
            Arc::new(Int64Array::from(col)),
            Arc::new(Float64Array::from(value)),
        ],
    )
}

// --- Precision/recall/F1 at thresholds ---------------------------------------

/// Pure-Rust PRF core: returns (precision, recall, f1, queue_rate) for each
/// threshold. Zero-division yields 0.0 for all three rate metrics, matching
/// sklearn's `zero_division=0` default.
fn prf_at_thresholds_core(
    y_true: &[f64],
    y_score: &[f64],
    thresholds: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y_true.len();
    let mut precision = Vec::with_capacity(thresholds.len());
    let mut recall = Vec::with_capacity(thresholds.len());
    let mut f1 = Vec::with_capacity(thresholds.len());
    let mut queue_rate = Vec::with_capacity(thresholds.len());

    for &t in thresholds {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;
        let mut predicted_pos = 0.0;
        for i in 0..n {
            let pred = y_score[i] >= t;
            let actual = y_true[i] != 0.0;
            if pred {
                predicted_pos += 1.0;
                if actual {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            } else if actual {
                fn_ += 1.0;
            }
        }
        // zero_division -> 0.0 for all three metrics.
        let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        precision.push(p);
        recall.push(r);
        f1.push(f);
        queue_rate.push(if n > 0 { predicted_pos / n as f64 } else { 0.0 });
    }

    (precision, recall, f1, queue_rate)
}

#[pyfunction]
pub fn prf_at_thresholds(
    _py: Python<'_>,
    y_true: PyArray,
    y_score: PyArray,
    thresholds: PyArray,
) -> PyResult<PyRecordBatch> {
    let yt = as_f64_slice(&y_true, "y_true")?;
    let ys = as_f64_slice(&y_score, "y_score")?;
    let thr = as_f64_slice(&thresholds, "thresholds")?;
    check_equal_len(yt.len(), ys.len(), "y_true", "y_score")?;

    let (precision, recall, f1, queue_rate) = prf_at_thresholds_core(yt, ys, thr);

    record_batch(
        vec![
            Field::new("precision", DataType::Float64, false),
            Field::new("recall", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("queue_rate", DataType::Float64, false),
        ],
        vec![
            Arc::new(Float64Array::from(precision)),
            Arc::new(Float64Array::from(recall)),
            Arc::new(Float64Array::from(f1)),
            Arc::new(Float64Array::from(queue_rate)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn kendall_perfectly_concordant() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, 1.0, 1e-12));
        assert_eq!(r.n_discordant, 0);
        assert_eq!(r.n_concordant, 10);
    }

    #[test]
    fn kendall_perfectly_discordant() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, -1.0, 1e-12));
        assert_eq!(r.n_concordant, 0);
        assert_eq!(r.n_discordant, 10);
    }

    #[test]
    fn kendall_all_tied_y() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 5.0, 5.0, 5.0];
        let r = kendall_tau_b(&x, &y);
        assert!(r.tau.is_nan());
        assert_eq!(r.n_tied_y, 6);
    }

    #[test]
    fn kendall_with_ties_in_x() {
        // x = [1,1,2,2], y = [1,2,1,2]
        // n0=6, T_x=2, T_y=2, T_both=0, D=1, C=1, tau = 0.
        let x = [1.0, 1.0, 2.0, 2.0];
        let y = [1.0, 2.0, 1.0, 2.0];
        let r = kendall_tau_b(&x, &y);
        assert_eq!(r.n_tied_x, 2);
        assert_eq!(r.n_tied_y, 2);
        assert!(approx_eq(r.tau, 0.0, 1e-12));
    }

    #[test]
    fn kendall_n_less_than_2_returns_nan() {
        let r = kendall_tau_b(&[1.0], &[2.0]);
        assert!(r.tau.is_nan());
    }

    // ---- R1-relocated coverage (tests/bug_hunt_stats_transforms.rs round2_tests,
    // 2026-08-27) — every test below calls the real `kendall_tau_b` above rather
    // than the mirror's inlined copy. `bug_hunt_r2_kendall_all_joint_ties` was
    // dropped as a duplicate of `kendall_all_tied_both` immediately below (both
    // assert an all-tied-in-both-columns group produces `tau = NaN` with matching
    // tie counts).

    #[test]
    fn kendall_all_tied_both() {
        // n0 = 6, T_x = 6, T_y = 6 → denom = sqrt(0*0) = 0 → NaN
        let x = [5.0, 5.0, 5.0, 5.0];
        let y = [3.0, 3.0, 3.0, 3.0];
        let r = kendall_tau_b(&x, &y);
        assert!(r.tau.is_nan(), "all-tied tau should be NaN, got {}", r.tau);
        assert_eq!(r.n_tied_x, 6, "n_tied_x for 4 elements all tied should be C(4,2)=6");
        assert_eq!(r.n_tied_y, 6, "n_tied_y for 4 elements all tied should be C(4,2)=6");
        assert_eq!(r.n_tied_both, 6, "n_tied_both should be 6");
    }

    #[test]
    fn kendall_tied_x_distinct_y_hand_computed() {
        // x=[1,1,2,2], y=[1,2,3,4]. n0=6, T_x=2, T_y=0, T_both=0, D=0, C=4.
        // tau = 4/sqrt(4*6) = 4/sqrt(24).
        let x = [1.0, 1.0, 2.0, 2.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let r = kendall_tau_b(&x, &y);
        assert!(r.tau.is_finite(), "tau with tied x should be finite, got {}", r.tau);
        assert_eq!(r.n_tied_x, 2);
        assert_eq!(r.n_tied_y, 0);
        let expected = 4.0 / (24.0_f64).sqrt();
        assert!(approx_eq(r.tau, expected, 1e-10), "tau should be {expected}, got {}", r.tau);
    }

    #[test]
    fn kendall_heavy_ties_bounds() {
        // x = [1,1,1,2,2,2], y = [1,1,2,1,2,2]. n0=15, T_x=T_y=6 (C(3,2)*2 each).
        let x = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let y = [1.0, 1.0, 2.0, 1.0, 2.0, 2.0];
        let r = kendall_tau_b(&x, &y);
        assert_eq!(r.n_tied_x, 6, "n_tied_x should be 6, got {}", r.n_tied_x);
        assert_eq!(r.n_tied_y, 6, "n_tied_y should be 6, got {}", r.n_tied_y);
        assert!(r.tau.is_finite(), "tau with heavy ties should be finite, got {}", r.tau);
        assert!(r.tau >= -1.0 && r.tau <= 1.0, "tau must be in [-1,1], got {}", r.tau);
    }

    #[test]
    fn kendall_identical_arrays_with_ties_is_one() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0];
        let y = [3.0, 1.0, 4.0, 1.0, 5.0];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, 1.0, 1e-10), "identical arrays tau should be 1.0, got {}", r.tau);
    }

    #[test]
    fn kendall_nan_in_x_no_panic() {
        // NaN breaks == comparison; partial_cmp falls back to Equal. Contract: no panic.
        let x = [1.0, f64::NAN, 3.0, 2.0, f64::NAN];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let _ = kendall_tau_b(&x, &y);
    }

    #[test]
    fn kendall_nan_in_y_no_panic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [f64::NAN, 2.0, f64::NAN, 4.0, 5.0];
        let _ = kendall_tau_b(&x, &y);
    }

    #[test]
    fn kendall_infinity_identical_monotonic_is_one() {
        // +Inf > everything, -Inf < everything → identical monotonic sequences.
        let x = [f64::NEG_INFINITY, 0.0, 1.0, f64::INFINITY];
        let y = [f64::NEG_INFINITY, 0.0, 1.0, f64::INFINITY];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, 1.0, 1e-10), "identical monotonic with Inf: tau should be 1.0, got {}", r.tau);
    }

    #[test]
    fn kendall_n2_discordant_is_negative_one() {
        let x = [1.0, 2.0];
        let y = [2.0, 1.0];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, -1.0, 1e-10), "n=2 discordant pair: tau should be -1.0, got {}", r.tau);
        assert_eq!(r.n_concordant, 0);
        assert_eq!(r.n_discordant, 1);
    }

    #[test]
    fn kendall_n2_all_tied_is_nan() {
        // n0=1, T_x=1, T_y=1, T_both=1 → denom = sqrt(0*0) = 0 → NaN.
        let x = [1.0, 1.0];
        let y = [2.0, 2.0];
        let r = kendall_tau_b(&x, &y);
        assert!(r.tau.is_nan(), "n=2 all-tied: tau should be NaN, got {}", r.tau);
    }

    #[test]
    fn kendall_joint_ties_subset_of_x_and_y_ties() {
        // x=[1,1,2,2], y=[3,3,4,4]. T_x=2, T_y=2, T_both=2 (pairs (0,1),(2,3)).
        // All non-joint-tied pairs are concordant: (0,2),(0,3),(1,2),(1,3) = 4.
        let x = [1.0, 1.0, 2.0, 2.0];
        let y = [3.0, 3.0, 4.0, 4.0];
        let r = kendall_tau_b(&x, &y);
        assert_eq!(r.n_tied_x, 2);
        assert_eq!(r.n_tied_y, 2);
        assert_eq!(r.n_tied_both, 2);
        assert_eq!(r.n_concordant, 4, "concordant should be 4, got {}", r.n_concordant);
        assert_eq!(r.n_discordant, 0);
        assert!(approx_eq(r.tau, 1.0, 1e-10), "tau should be 1.0, got {}", r.tau);
    }

    #[test]
    fn kendall_mixed_concordant_discordant_hand_computed() {
        // x=[1,2,3,4], y=[1,3,2,4]. Pairs: (0,1)C,(0,2)C,(0,3)C,(1,2)D,(1,3)C,(2,3)C.
        // C=5, D=1. tau = (5-1)/6 = 2/3.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 4.0];
        let r = kendall_tau_b(&x, &y);
        assert_eq!(r.n_concordant, 5);
        assert_eq!(r.n_discordant, 1);
        let expected = 4.0 / 6.0;
        assert!(approx_eq(r.tau, expected, 1e-10), "tau should be {expected}, got {}", r.tau);
    }

    #[test]
    fn kendall_symmetry_tau_xy_equals_tau_yx() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
        let r_xy = kendall_tau_b(&x, &y);
        let r_yx = kendall_tau_b(&y, &x);
        assert!(approx_eq(r_xy.tau, r_yx.tau, 1e-10), "tau(x,y) should equal tau(y,x): {} vs {}", r_xy.tau, r_yx.tau);
    }

    #[test]
    fn kendall_antisymmetry_under_negation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 3.0, 1.0, 5.0, 4.0];
        let y_neg: Vec<f64> = y.iter().map(|&v| -v).collect();
        let r = kendall_tau_b(&x, &y);
        let r_neg = kendall_tau_b(&x, &y_neg);
        assert!(approx_eq(r.tau, -r_neg.tau, 1e-10), "tau(x,y) should be -tau(x,-y): {} vs {}", r.tau, -r_neg.tau);
    }

    #[test]
    fn kendall_bounds_across_tie_patterns() {
        let test_cases: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0, 2.0, 2.0, 3.0, 3.0], vec![5.0, 4.0, 4.0, 3.0, 3.0]),
            (vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0], vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
            ((0..20).map(|i| (i % 5) as f64).collect(), (0..20).map(|i| (i % 7) as f64).collect()),
        ];
        for (idx, (x, y)) in test_cases.iter().enumerate() {
            let r = kendall_tau_b(x, y);
            if r.tau.is_finite() {
                assert!(r.tau >= -1.0 && r.tau <= 1.0, "test case {idx}: tau must be in [-1,1], got {}", r.tau);
            }
        }
    }

    #[test]
    fn kendall_large_monotonic_n100_is_one() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, 1.0, 1e-10), "monotonic n=100: tau should be 1.0, got {}", r.tau);
        assert_eq!(r.n_concordant, 100 * 99 / 2);
        assert_eq!(r.n_discordant, 0);
    }

    #[test]
    fn kendall_large_anti_monotonic_n100_is_negative_one() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).rev().map(|i| i as f64).collect();
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, -1.0, 1e-10), "anti-monotonic n=100: tau should be -1.0, got {}", r.tau);
        assert_eq!(r.n_concordant, 0);
        assert_eq!(r.n_discordant, 100 * 99 / 2);
    }

    #[test]
    fn kendall_pair_count_invariant_holds() {
        // For any input: C + D + T_x + T_y - T_both = n0.
        let test_cases: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 3.0, 1.0, 4.0, 2.0]),
            (vec![1.0, 1.0, 2.0, 2.0, 3.0], vec![3.0, 1.0, 2.0, 2.0, 1.0]),
            (vec![5.0, 5.0, 5.0], vec![1.0, 2.0, 3.0]),
            (vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0]),
        ];
        for (idx, (x, y)) in test_cases.iter().enumerate() {
            let r = kendall_tau_b(x, y);
            let n = x.len() as u64;
            let n0 = n * (n - 1) / 2;
            let sum = r.n_concordant as i128 + r.n_discordant as i128
                + r.n_tied_x as i128 + r.n_tied_y as i128
                - r.n_tied_both as i128;
            assert_eq!(sum, n0 as i128, "test case {idx}: C+D+T_x+T_y-T_both ({sum}) should equal n0 ({n0})");
        }
    }

    // ---- Curve kernels (sklearn-parity) ----

    fn slices_approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(approx_eq(*x, *y, tol), "mismatch: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn roc_perfect_classifier() {
        // Scores perfectly separate the classes.
        let yt = [0.0, 0.0, 1.0, 1.0];
        let ys = [0.1, 0.2, 0.8, 0.9];
        let (fpr, tpr, thr) = roc_curve_core(&yt, &ys, true);
        // Perfect classifier reaches (0, 1) before any false positive.
        assert_eq!(fpr[0], 0.0);
        assert_eq!(tpr[0], 0.0);
        assert!(thr[0].is_infinite());
        assert!(approx_eq(roc_auc_core(&yt, &ys), 1.0, 1e-12));
    }

    #[test]
    fn roc_matches_sklearn_example() {
        // Verified against sklearn 1.7.2 (drop_intermediate has no effect here).
        let yt = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let ys = [0.1, 0.4, 0.35, 0.8, 0.8, 0.4, 0.7];
        let (fpr, tpr, thr) = roc_curve_core(&yt, &ys, true);
        slices_approx_eq(
            &fpr,
            &[0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 1.0],
            1e-12,
        );
        slices_approx_eq(&tpr, &[0.0, 0.5, 0.75, 0.75, 1.0, 1.0], 1e-12);
        assert!(thr[0].is_infinite());
        slices_approx_eq(&thr[1..], &[0.8, 0.7, 0.4, 0.35, 0.1], 1e-12);
        assert!(approx_eq(roc_auc_core(&yt, &ys), 0.8333333333333334, 1e-12));
    }

    #[test]
    fn roc_ties_aggregate_into_one_point() {
        // Two tied scores (0.8) collapse to a single threshold point.
        let yt = [0.0, 1.0, 1.0, 0.0];
        let ys = [0.8, 0.8, 0.5, 0.5];
        let (_, _, thr) = roc_curve_core(&yt, &ys, false);
        // Distinct thresholds = {0.8, 0.5} plus the +inf sentinel.
        assert_eq!(thr.len(), 3);
        assert!(thr[0].is_infinite());
        slices_approx_eq(&thr[1..], &[0.8, 0.5], 1e-12);
    }

    #[test]
    fn roc_single_class_yields_nan_auc() {
        let yt = [1.0, 1.0, 1.0];
        let ys = [0.1, 0.2, 0.3];
        assert!(roc_auc_core(&yt, &ys).is_nan());
        let (fpr, _, _) = roc_curve_core(&yt, &ys, true);
        // No negatives: ALL fpr values are NaN (including the leading origin
        // slot) because fpr = fps / fps[-1] = x / 0, matching sklearn 1.7.2.
        assert!(fpr.iter().all(|v| v.is_nan()));
    }

    // --- Degenerate single-class input: sklearn 1.7.2 byte-parity ----

    #[test]
    fn roc_all_negative_tpr_all_nan() {
        // y_true = [0, 0, 0]: zero positives → tpr = tps / tps[-1] = x / 0 → all NaN.
        // Verified against sklearn 1.7.2:
        //   roc_curve([0,0,0], [0.1, 0.5, 0.9]) →
        //     fpr = [0, 1/3, 2/3, 1], tpr = [nan, nan, nan, nan],
        //     thresholds = [+inf, 0.9, 0.5, 0.1]
        let yt = [0.0, 0.0, 0.0];
        let ys = [0.1, 0.5, 0.9];
        let (fpr, tpr, thr) = roc_curve_core(&yt, &ys, false);
        // All tpr values are NaN.
        assert!(
            tpr.iter().all(|v| v.is_nan()),
            "all-negative y_true must yield all-NaN tpr; got {tpr:?}"
        );
        // fpr is finite and matches the expected rates (0 positives → fps == cumulative counts).
        assert!(fpr.iter().all(|v| v.is_finite()), "fpr must be finite; got {fpr:?}");
        slices_approx_eq(&fpr, &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], 1e-12);
        // Leading threshold is +inf; remaining descend from highest score.
        assert!(thr[0].is_infinite());
        slices_approx_eq(&thr[1..], &[0.9, 0.5, 0.1], 1e-12);
    }

    #[test]
    fn roc_all_positive_fpr_all_nan() {
        // y_true = [1, 1, 1]: zero negatives → fpr = fps / fps[-1] = x / 0 → all NaN.
        // Verified against sklearn 1.7.2:
        //   roc_curve([1,1,1], [0.1, 0.5, 0.9]) →
        //     fpr = [nan, nan, nan, nan], tpr = [0, 1/3, 2/3, 1],
        //     thresholds = [+inf, 0.9, 0.5, 0.1]
        let yt = [1.0, 1.0, 1.0];
        let ys = [0.1, 0.5, 0.9];
        let (fpr, tpr, thr) = roc_curve_core(&yt, &ys, false);
        // All fpr values are NaN.
        assert!(
            fpr.iter().all(|v| v.is_nan()),
            "all-positive y_true must yield all-NaN fpr; got {fpr:?}"
        );
        // tpr is finite and matches the expected rates.
        assert!(tpr.iter().all(|v| v.is_finite()), "tpr must be finite; got {tpr:?}");
        slices_approx_eq(&tpr, &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], 1e-12);
        // Leading threshold is +inf; remaining descend from highest score.
        assert!(thr[0].is_infinite());
        slices_approx_eq(&thr[1..], &[0.9, 0.5, 0.1], 1e-12);
    }

    #[test]
    fn roc_drop_intermediate_prunes_collinear_points() {
        // Verified against sklearn 1.7.2: drop keeps fewer points than no-drop.
        let yt = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let ys: Vec<f64> = (0..10).map(|i| 0.9 - 0.08 * i as f64).collect();
        let (fpr_drop, tpr_drop, _) = roc_curve_core(&yt, &ys, true);
        let (fpr_nodrop, tpr_nodrop, _) = roc_curve_core(&yt, &ys, false);
        assert!(fpr_drop.len() < fpr_nodrop.len());
        // Endpoints preserved.
        assert_eq!(fpr_drop[0], fpr_nodrop[0]);
        assert!(approx_eq(
            *fpr_drop.last().unwrap(),
            *fpr_nodrop.last().unwrap(),
            1e-12
        ));
        assert!(approx_eq(
            *tpr_drop.last().unwrap(),
            *tpr_nodrop.last().unwrap(),
            1e-12
        ));
    }

    #[test]
    fn pr_curve_matches_sklearn_example() {
        // Verified against sklearn 1.7.2.
        let yt = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let ys = [0.1, 0.4, 0.35, 0.8, 0.8, 0.4, 0.7];
        let (p, r, t) = pr_curve_core(&yt, &ys);
        slices_approx_eq(
            &p,
            &[4.0 / 7.0, 2.0 / 3.0, 0.6, 1.0, 1.0, 1.0],
            1e-12,
        );
        slices_approx_eq(&r, &[1.0, 1.0, 0.75, 0.75, 0.5, 0.0], 1e-12);
        // Final threshold slot is NaN; preceding values match sklearn.
        assert!(t.last().unwrap().is_nan());
        slices_approx_eq(&t[..t.len() - 1], &[0.1, 0.35, 0.4, 0.7, 0.8], 1e-12);
        assert!(approx_eq(
            average_precision_core(&yt, &ys),
            0.9166666666666666,
            1e-12
        ));
    }

    #[test]
    fn pr_all_correct_perfect_precision() {
        let yt = [0.0, 0.0, 1.0, 1.0];
        let ys = [0.1, 0.2, 0.8, 0.9];
        // Perfect separation gives AP = 1.0.
        assert!(approx_eq(average_precision_core(&yt, &ys), 1.0, 1e-12));
    }

    #[test]
    fn calibration_uniform_matches_sklearn() {
        // Verified against sklearn 1.7.2 calibration_curve(n_bins=4, uniform).
        let yt = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let yp = [0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.05, 0.6, 0.55];
        let (mp, fp, cnt) = calibration_core(&yt, &yp, 4, "uniform").unwrap();
        slices_approx_eq(&fp, &[0.0, 0.5, 1.0, 1.0], 1e-12);
        slices_approx_eq(
            &mp,
            &[0.11666666666666665, 0.35, 0.6166666666666667, 0.85],
            1e-9,
        );
        assert_eq!(cnt, vec![3, 2, 3, 2]);
    }

    #[test]
    fn calibration_quantile_matches_sklearn() {
        let yt = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let yp = [0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.05, 0.6, 0.55];
        let (mp, fp, _) = calibration_core(&yt, &yp, 4, "quantile").unwrap();
        slices_approx_eq(&fp, &[0.0, 0.5, 1.0, 1.0], 1e-12);
        slices_approx_eq(
            &mp,
            &[0.11666666666666665, 0.35, 0.575, 0.8],
            1e-9,
        );
    }

    #[test]
    fn calibration_drops_empty_bins() {
        // 20 bins over 10 distinct probs leaves many empty; survivors only.
        let yt = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let yp = [0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.05, 0.6, 0.55];
        let (mp, _, cnt) = calibration_core(&yt, &yp, 20, "uniform").unwrap();
        // Exactly 10 non-empty bins (one per distinct prob), all count 1.
        assert_eq!(mp.len(), 10);
        assert!(cnt.iter().all(|&c| c == 1));
    }

    #[test]
    fn calibration_quantile_drops_empty_bins() {
        // 10 bins over only 2 distinct prob values (0.2 and 0.8); quantile
        // edges collapse, leaving only 2 non-empty bins. Expected values
        // cross-checked against sklearn 1.7.2:
        //   calibration_curve(yt, yp, n_bins=10, strategy="quantile")
        // -> mean_predicted=[0.2, 0.8], fraction_positive=[0.5, 0.6667],
        //    counts=[2, 3].
        let yt = [0.0, 1.0, 1.0, 0.0, 1.0];
        let yp = [0.2, 0.2, 0.8, 0.8, 0.8];
        let (mp, fp, cnt) = calibration_core(&yt, &yp, 10, "quantile").unwrap();
        assert_eq!(cnt.len(), 2, "only 2 non-empty bins should survive");
        slices_approx_eq(&mp, &[0.2, 0.8], 1e-12);
        slices_approx_eq(&fp, &[0.5, 2.0 / 3.0], 1e-9);
        assert_eq!(cnt, vec![2, 3]);
    }

    #[test]
    fn prf_zero_division_yields_zero() {
        // At a threshold above every score, no positives are predicted.
        // This test calls prf_at_thresholds_core directly so the real
        // zero-division path is exercised, not a hand-copied duplicate.
        let yt = [0.0, 1.0, 1.0];
        let ys = [0.2, 0.5, 0.9];
        let thr = [1.5]; // above all scores
        let (precision, recall, f1, queue_rate) = prf_at_thresholds_core(&yt, &ys, &thr);
        assert_eq!(precision[0], 0.0);
        assert_eq!(recall[0], 0.0);
        assert_eq!(f1[0], 0.0);
        assert_eq!(queue_rate[0], 0.0);
    }

    #[test]
    fn binary_clf_curve_tie_aggregation() {
        // Tied top scores accumulate their TPs into a single point.
        let yt = [1.0, 1.0, 0.0];
        let ys = [0.9, 0.9, 0.1];
        let (fps, tps, thr) = binary_clf_curve(&yt, &ys);
        assert_eq!(thr.len(), 2); // {0.9, 0.1}
        assert!(approx_eq(tps[0], 2.0, 1e-12));
        assert!(approx_eq(fps[0], 0.0, 1e-12));
    }

    #[test]
    fn average_precision_all_negative_returns_zero() {
        // sklearn.metrics.average_precision_score([0,0,0], [0.1,0.5,0.9]) == 0.0
        // (sklearn emits a UserWarning but returns 0.0, not NaN).
        let yt = [0.0, 0.0, 0.0];
        let ys = [0.1, 0.5, 0.9];
        let ap = average_precision_core(&yt, &ys);
        assert_eq!(ap, 0.0, "all-negative y_true must yield 0.0, not NaN");
    }

    #[test]
    fn confusion_matrix_core_3class_normalize_variants() {
        // Cross-checked against sklearn 1.7.2:
        //   y_true = [0,0,1,1,2,2,5], y_pred = [0,1,1,2,2,0,5], labels = [0,1,2]
        // The (true=5,pred=5) pair is absent from labels and must be ignored.
        //
        // Raw counts (no normalize):
        //   [[1,1,0],[0,1,1],[1,0,1]]
        //
        // normalize="true":
        //   [[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]]
        //
        // normalize="all" (total=6, each entry / 6):
        //   [[1/6,1/6,0],[0,1/6,1/6],[1/6,0,1/6]]
        let y_true = [0i64, 0, 1, 1, 2, 2, 5];
        let y_pred = [0i64, 1, 1, 2, 2, 0, 5];
        let labels = [0i64, 1, 2];
        let l = labels.len();

        // ---- no normalize ----
        let cm = confusion_matrix_core(&y_true, &y_pred, &labels, "").unwrap();
        let expected_raw = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        for (k, (&got, &exp)) in cm.iter().zip(expected_raw.iter()).enumerate() {
            assert!(
                approx_eq(got, exp, 1e-12),
                "raw cm[{k}]: got {got}, expected {exp}"
            );
        }

        // ---- normalize = "true" ----
        let cm_true = confusion_matrix_core(&y_true, &y_pred, &labels, "true").unwrap();
        let expected_true = [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5];
        for (k, (&got, &exp)) in cm_true.iter().zip(expected_true.iter()).enumerate() {
            assert!(
                approx_eq(got, exp, 1e-12),
                "true-norm cm[{k}]: got {got}, expected {exp}"
            );
        }

        // ---- normalize = "all" ----
        let cm_all = confusion_matrix_core(&y_true, &y_pred, &labels, "all").unwrap();
        let sixth = 1.0 / 6.0;
        let expected_all = [sixth, sixth, 0.0, 0.0, sixth, sixth, sixth, 0.0, sixth];
        for (k, (&got, &exp)) in cm_all.iter().zip(expected_all.iter()).enumerate() {
            assert!(
                approx_eq(got, exp, 1e-9),
                "all-norm cm[{k}]: got {got}, expected {exp}"
            );
        }

        // ---- label NOT in array is silently ignored ----
        // Verify by checking the 3×3 result has exactly 6 total raw counts
        // (the (5,5) pair contributes nothing).
        let raw_sum: f64 = confusion_matrix_core(&y_true, &y_pred, &labels, "")
            .unwrap()
            .iter()
            .sum();
        assert_eq!(raw_sum, 6.0, "pair with label=5 outside labels must be ignored");

        // Drop unused variable warning suppression.
        let _ = l;
    }

    // ---- Null-reject guard (check_no_nulls) ----

    /// A null entry (`null_count > 0`) must be rejected.
    ///
    /// `check_no_nulls` returns `Result<(), String>` so it can be exercised
    /// without a live Python interpreter (no `PyValueError::new_err` needed).
    #[test]
    fn null_guard_rejects_nonzero_null_count() {
        // Build a Float64Array that genuinely has Arrow nulls.
        let arr = Float64Array::from(vec![Some(1.0_f64), None, Some(2.0)]);
        let null_count = arr.null_count();
        assert!(null_count > 0, "test array must have at least one null");

        let result = check_no_nulls(null_count, "y_score");
        assert!(
            result.is_err(),
            "check_no_nulls must return Err when null_count > 0"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("y_score") && msg.contains("nulls"),
            "error message must mention the field name and 'nulls'; got: {msg}"
        );
    }

    /// `null_count == 0` must pass, even when the backing buffer contains NaN.
    ///
    /// Arrow stores NaN as a regular f64 value; the validity bitmap is fully
    /// set, so `null_count` is zero. The guard must NOT fire.
    #[test]
    fn null_guard_passes_nan_values_with_zero_null_count() {
        // A Float64Array built from plain f64 values (including NaN) has no
        // Arrow nulls, so null_count() == 0.
        let arr = Float64Array::from(vec![1.0_f64, f64::NAN, 2.0]);
        assert_eq!(arr.null_count(), 0, "NaN values must not set the null bitmap");
        assert!(
            check_no_nulls(arr.null_count(), "y_score").is_ok(),
            "NaN-containing array with null_count==0 must pass the guard"
        );
    }
}
