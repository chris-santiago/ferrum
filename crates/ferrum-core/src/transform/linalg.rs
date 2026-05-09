//! Small linear-algebra helpers used by the stat engine.
//! Currently: Cholesky solve for a 3x3 symmetric positive-definite system.

/// Solves M x = b for a 3x3 symmetric positive-definite M via Cholesky factorization.
/// Returns None if M is not positive-definite (e.g. rank-deficient or near-singular).
///
/// Algorithm:
///   1. Factor M = L L' where L is 3x3 lower-triangular with positive diagonal.
///   2. Solve L y = b (forward substitution).
///   3. Solve L' x = y (backward substitution).
///
/// LOESS degree=2 calls this on the weighted normal-equations matrix
/// X' W X where X has rows [1, x_i, x_i^2] and W is diagonal with tricube weights;
/// SPD is guaranteed when at least 3 distinct x_i fall in the local window with positive weight.
pub(crate) fn solve_3x3_spd(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    // Cholesky factor in-place into l[i][j] for j <= i.
    let l00_sq = m[0][0];
    if !(l00_sq > 0.0) { return None; }
    let l00 = l00_sq.sqrt();

    let l10 = m[1][0] / l00;
    let l11_sq = m[1][1] - l10 * l10;
    if !(l11_sq > 0.0) { return None; }
    let l11 = l11_sq.sqrt();

    let l20 = m[2][0] / l00;
    let l21 = (m[2][1] - l20 * l10) / l11;
    let l22_sq = m[2][2] - l20 * l20 - l21 * l21;
    if !(l22_sq > 0.0) { return None; }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_solve_3x3_spd_identity_returns_rhs() {
        // I * x = b → x = b
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
        // M = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] (classic Cholesky example, SPD)
        // L should be [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        // Pick rhs b = M @ [1, 2, 3] = [4 + 24 - 48, 12 + 74 - 129, -16 - 86 + 294] = [-20, -43, 192]
        let m = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
        let b = [-20.0, -43.0, 192.0];
        let x = solve_3x3_spd(m, b).unwrap();
        assert!(approx_eq(x[0], 1.0, 1e-9), "x[0] = {}", x[0]);
        assert!(approx_eq(x[1], 2.0, 1e-9), "x[1] = {}", x[1]);
        assert!(approx_eq(x[2], 3.0, 1e-9), "x[2] = {}", x[2]);
    }

    #[test]
    fn test_solve_3x3_spd_round_trip() {
        // Synthesize a vandermonde-like SPD: X' X where X = [[1,a,a^2], [1,b,b^2], [1,c,c^2]]
        let a = 0.5; let b = 1.5; let c = 3.0;
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
        // RHS: X' y where y = X * [1, -2, 0.5]
        let beta_true = [1.0, -2.0, 0.5];
        let mut rhs = [0.0; 3];
        for &xi in &xs {
            let row = [1.0, xi, xi * xi];
            let yi = beta_true.iter().zip(row.iter()).map(|(b, r)| b * r).sum::<f64>();
            for i in 0..3 { rhs[i] += row[i] * yi; }
        }
        let beta_solved = solve_3x3_spd(xt_x, rhs).unwrap();
        for i in 0..3 {
            assert!(approx_eq(beta_solved[i], beta_true[i], 1e-9),
                "beta[{i}] = {} vs {}", beta_solved[i], beta_true[i]);
        }
    }

    #[test]
    fn test_solve_3x3_spd_singular_returns_none() {
        // Rank-deficient: rows 1 and 2 are identical → not SPD.
        let m = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        let b = [1.0, 2.0, 3.0];
        assert!(solve_3x3_spd(m, b).is_none());
    }
}
