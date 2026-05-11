//! Phase 10 model-diagnostics — sole Rust contribution.
//!
//! `kendall_tau_b` implements Knight's O(n log n) merge-sort variant
//! for Kendall's tau-b rank correlation. Used by
//! `ModelSource.rank2d(algorithm="kendall")` when n is large enough
//! that the O(n²) pure-Python loop is too slow.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
