//! Shared linear-algebra helpers for the stat engine.
//!
//! - Fixed-size solvers: `invert_2x2`, `solve_3x3_spd` (hand-rolled, faster
//!   than faer dispatch overhead at these sizes).
//! - `faer`-backed routines for arbitrary-size problems: `hat_diagonal`,
//!   `corrcoef`.

use faer::prelude::*;
use faer::linalg::solvers::Llt;
use faer::Side;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

// ---- Fixed-size helpers (shared across glm, logistic, robust, smooth) ----

/// Invert a 2×2 matrix via Cramer's rule.
/// Returns `None` when the determinant is near-zero (singular).
pub(crate) fn invert_2x2(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !det.is_finite() || det.abs() < 1e-15 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [a[1][1] * inv_det, -a[0][1] * inv_det],
        [-a[1][0] * inv_det, a[0][0] * inv_det],
    ])
}

/// Scalar-multiply a 2×2 matrix.
pub(crate) fn scale_2x2(a: [[f64; 2]; 2], k: f64) -> [[f64; 2]; 2] {
    [
        [a[0][0] * k, a[0][1] * k],
        [a[1][0] * k, a[1][1] * k],
    ]
}

/// Build unweighted X'X for the univariate design X = [1, x].
pub(crate) fn xtx_unweighted(xs: &[f64]) -> [[f64; 2]; 2] {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    [[n, sx], [sx, sxx]]
}

/// Solve M x = b for a 3×3 symmetric positive-definite M via Cholesky.
/// Returns `None` if M is not positive-definite.
pub(crate) fn solve_3x3_spd(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let l00_sq = m[0][0];
    if !(l00_sq > 0.0) {
        return None;
    }
    let l00 = l00_sq.sqrt();

    let l10 = m[1][0] / l00;
    let l11_sq = m[1][1] - l10 * l10;
    if !(l11_sq > 0.0) {
        return None;
    }
    let l11 = l11_sq.sqrt();

    let l20 = m[2][0] / l00;
    let l21 = (m[2][1] - l20 * l10) / l11;
    let l22_sq = m[2][2] - l20 * l20 - l21 * l21;
    if !(l22_sq > 0.0) {
        return None;
    }
    let l22 = l22_sq.sqrt();

    // Forward sub: L y = b
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let y2 = (b[2] - l20 * y0 - l21 * y1) / l22;

    // Back sub: L' x = y
    let x2 = y2 / l22;
    let x1 = (y1 - l21 * x2) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2) / l00;

    Some([x0, x1, x2])
}

// ---- faer-backed routines ----


/// Hat matrix diagonal h_ii via Cholesky of the normal equations.
///
/// Given design matrix X (n × p), computes h_ii = X_i' (X'X)^{-1} X_i
/// for each row i. Uses Cholesky factorization of X'X (SPD when X has
/// full column rank). Returns Vec<f64> of length n, clamped to [0, 1-eps].
///
/// Falls back to zero leverage if Cholesky fails (rank-deficient X).
pub(crate) fn hat_diagonal(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return vec![0.0; n];
    }

    // X'X (p × p)
    let xtx = x.transpose() * x;

    // Cholesky of X'X
    let llt = match Llt::new(xtx.as_ref(), Side::Lower) {
        Ok(l) => l,
        Err(_) => return vec![0.0; n],
    };

    // Solve (X'X) Z = X' → Z = (X'X)^{-1} X' (p × n)
    let xt = x.transpose().to_owned();
    let z = llt.solve(&xt);

    // h_ii = sum_j X[i,j] * Z[j,i] = dot(X[i,:], Z[:,i])
    let mut h = vec![0.0; n];
    for i in 0..n {
        let mut val = 0.0_f64;
        for j in 0..p {
            val += x[(i, j)] * z[(j, i)];
        }
        h[i] = val.clamp(0.0, 1.0 - 1e-12);
    }
    h
}


/// Pearson correlation matrix for X (n × p). Returns a p × p Mat<f64>.
///
/// Each entry C\[i,j\] = cor(X\[:,i\], X\[:,j\]). Diagonal is pinned to 1.0
/// to avoid floating-point drift. Zero-variance columns yield 0.0 correlation.
pub(crate) fn corrcoef(x: &Mat<f64>) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();
    if p == 0 || n == 0 {
        return Mat::zeros(p, p);
    }

    // Column means
    let mut means = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += x[(i, j)];
        }
        means[j] = s / n as f64;
    }

    // Centered copy
    let mut xc = Mat::zeros(n, p);
    for j in 0..p {
        for i in 0..n {
            xc[(i, j)] = x[(i, j)] - means[j];
        }
    }

    // Gram matrix Xc' Xc
    let gram = xc.transpose() * &xc;

    // Column norms (sqrt of diagonal of gram)
    let mut norms = vec![0.0; p];
    for j in 0..p {
        norms[j] = gram[(j, j)].sqrt();
    }

    // Normalize to correlation
    let mut c = Mat::zeros(p, p);
    for i in 0..p {
        c[(i, i)] = 1.0;
        for j in (i + 1)..p {
            let denom = norms[i] * norms[j];
            let corr = if denom > 0.0 {
                (gram[(i, j)] / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            c[(i, j)] = corr;
            c[(j, i)] = corr;
        }
    }
    c
}


/// Build a faer Mat<f64> from a flat row-major slice and dimensions.
///
/// # Errors
/// Returns `PyValueError` if `data.len() != nrows * ncols`.
pub(crate) fn mat_from_flat(data: &[f64], nrows: usize, ncols: usize) -> PyResult<Mat<f64>> {
    if data.len() != nrows * ncols {
        return Err(PyValueError::new_err(format!(
            "mat_from_flat: data length mismatch — expected {} ({}×{}), got {}",
            nrows * ncols, nrows, ncols, data.len()
        )));
    }
    let mut m = Mat::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            m[(i, j)] = data[i * ncols + j];
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // ---- invert_2x2 ----

    #[test]
    fn test_invert_2x2_identity() {
        let m = [[1.0, 0.0], [0.0, 1.0]];
        let inv = invert_2x2(m).unwrap();
        assert!(approx_eq(inv[0][0], 1.0, 1e-12));
        assert!(approx_eq(inv[1][1], 1.0, 1e-12));
    }

    #[test]
    fn test_invert_2x2_singular_returns_none() {
        let m = [[1.0, 2.0], [2.0, 4.0]];
        assert!(invert_2x2(m).is_none());
    }

    #[test]
    fn test_invert_2x2_general() {
        let m = [[4.0, 7.0], [2.0, 6.0]];
        let inv = invert_2x2(m).unwrap();
        // M * M^-1 = I
        let p00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let p11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];
        assert!(approx_eq(p00, 1.0, 1e-12));
        assert!(approx_eq(p11, 1.0, 1e-12));
    }

    // ---- solve_3x3_spd ----

    #[test]
    fn test_solve_3x3_spd_identity_returns_rhs() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [5.0, -3.0, 7.0];
        let x = solve_3x3_spd(m, b).unwrap();
        assert!(approx_eq(x[0], 5.0, 1e-12));
        assert!(approx_eq(x[1], -3.0, 1e-12));
        assert!(approx_eq(x[2], 7.0, 1e-12));
    }

    #[test]
    fn test_solve_3x3_spd_diagonal_returns_rhs_div_diag() {
        let m = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]];
        let b = [4.0, 8.0, 16.0];
        let x = solve_3x3_spd(m, b).unwrap();
        assert!(approx_eq(x[0], 2.0, 1e-12));
        assert!(approx_eq(x[1], 2.0, 1e-12));
        assert!(approx_eq(x[2], 2.0, 1e-12));
    }

    #[test]
    fn test_solve_3x3_spd_general_case() {
        let m = [
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ];
        let b = [-20.0, -43.0, 192.0];
        let x = solve_3x3_spd(m, b).unwrap();
        assert!(approx_eq(x[0], 1.0, 1e-9), "x[0] = {}", x[0]);
        assert!(approx_eq(x[1], 2.0, 1e-9), "x[1] = {}", x[1]);
        assert!(approx_eq(x[2], 3.0, 1e-9), "x[2] = {}", x[2]);
    }

    #[test]
    fn test_solve_3x3_spd_round_trip() {
        let a = 0.5;
        let b = 1.5;
        let c = 3.0;
        let xs = [a, b, c];
        let mut xt_x = [[0.0; 3]; 3];
        for &xi in &xs {
            let row = [1.0, xi, xi * xi];
            for i in 0..3 {
                for j in 0..3 {
                    xt_x[i][j] += row[i] * row[j];
                }
            }
        }
        let beta_true = [1.0, -2.0, 0.5];
        let mut rhs = [0.0; 3];
        for &xi in &xs {
            let row = [1.0, xi, xi * xi];
            let yi = beta_true
                .iter()
                .zip(row.iter())
                .map(|(b, r)| b * r)
                .sum::<f64>();
            for i in 0..3 {
                rhs[i] += row[i] * yi;
            }
        }
        let beta_solved = solve_3x3_spd(xt_x, rhs).unwrap();
        for i in 0..3 {
            assert!(
                approx_eq(beta_solved[i], beta_true[i], 1e-9),
                "beta[{i}] = {} vs {}",
                beta_solved[i],
                beta_true[i]
            );
        }
    }

    #[test]
    fn test_solve_3x3_spd_singular_returns_none() {
        let m = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        let b = [1.0, 2.0, 3.0];
        assert!(solve_3x3_spd(m, b).is_none());
    }

    // ---- hat_diagonal ----

    #[test]
    fn test_hat_diagonal_identity_design() {
        // X = I_3 → H = I → h_ii = 1 (clamped to 1 - 1e-12)
        let mut x = Mat::zeros(3, 3);
        x[(0, 0)] = 1.0;
        x[(1, 1)] = 1.0;
        x[(2, 2)] = 1.0;
        let h = hat_diagonal(&x);
        for &hi in &h {
            assert!(approx_eq(hi, 1.0 - 1e-12, 1e-10));
        }
    }

    #[test]
    fn test_hat_diagonal_intercept_only() {
        // X = [1, 1, 1, 1]' → h_ii = 1/n = 0.25
        let n = 4;
        let mut x = Mat::zeros(n, 1);
        for i in 0..n {
            x[(i, 0)] = 1.0;
        }
        let h = hat_diagonal(&x);
        for &hi in &h {
            assert!(approx_eq(hi, 0.25, 1e-12));
        }
    }

    #[test]
    fn test_hat_diagonal_sum_equals_p() {
        // tr(H) = p for any X with rank p
        let n = 10;
        let p = 3;
        let mut x = Mat::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = i as f64;
            x[(i, 2)] = (i as f64) * (i as f64);
        }
        let h = hat_diagonal(&x);
        let trace: f64 = h.iter().sum();
        assert!(approx_eq(trace, p as f64, 1e-9), "trace = {}", trace);
    }

    // ---- corrcoef ----

    #[test]
    fn test_corrcoef_identity_correlation() {
        // Perfectly correlated: col0 = col1
        let mut x = Mat::zeros(4, 2);
        for i in 0..4 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = i as f64;
        }
        let c = corrcoef(&x);
        assert!(approx_eq(c[(0, 0)], 1.0, 1e-12));
        assert!(approx_eq(c[(1, 1)], 1.0, 1e-12));
        assert!(approx_eq(c[(0, 1)], 1.0, 1e-12));
    }

    #[test]
    fn test_corrcoef_negative_correlation() {
        let mut x = Mat::zeros(4, 2);
        for i in 0..4 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = -(i as f64);
        }
        let c = corrcoef(&x);
        assert!(approx_eq(c[(0, 1)], -1.0, 1e-12));
    }

    #[test]
    fn test_corrcoef_zero_variance() {
        // Constant column → 0 correlation
        let mut x = Mat::zeros(4, 2);
        for i in 0..4 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = 5.0;
        }
        let c = corrcoef(&x);
        assert!(approx_eq(c[(0, 1)], 0.0, 1e-12));
        assert!(approx_eq(c[(1, 1)], 1.0, 1e-12));
    }
}
