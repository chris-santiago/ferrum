//! Penalized cubic B-spline smoother — the numerical core behind
//! `Smooth(method="gam")` (batch-D T10, spec §4.6).
//!
//! This is a real generalized additive model in the single-smooth case:
//! one smooth term `f(x)` fitted by penalized least squares, with the
//! smoothing parameter chosen by generalized cross-validation. It is not an
//! alias for any other method in [`crate::transform::smooth`], and it uses
//! no randomness at all — see "Determinism" below.
//!
//! ## The model
//!
//! `y = f(x) + ε`, with `f(x) = Σ_j β_j B_j(x)` over a **cubic B-spline
//! basis on uniformly spaced knots**, fitted by minimizing
//!
//! ```text
//!   ‖y − Bβ‖² + λ · βᵀSβ
//! ```
//!
//! This is the P-spline construction (Eilers & Marx 1996): a rich,
//! equally-spaced B-spline basis deliberately larger than the wiggliness
//! the data support, with a **second-order difference penalty**
//! `S = D₂ᵀD₂` on adjacent coefficients doing the smoothing. The
//! alternative (an integrated-squared-second-derivative penalty on
//! data-quantile knots) needs the same machinery for the same practical
//! answer; the difference penalty is exactly representable in the
//! coefficient basis, so no numerical quadrature enters the fit.
//!
//! ## Documented defaults
//!
//! - **Basis dimension** `k = clamp(distinct_x, 4, 10)` — 10 is mgcv's own
//!   default basis dimension for a single smooth, and the lower bound is
//!   `SPLINE_DEGREE + 1`, the smallest basis that spans a single cubic
//!   segment. Capping at the number of *distinct* x values keeps the basis
//!   from carrying more segments than the data can place.
//! - **Knots**: `k − 3` equal-width segments spanning `[min(x), max(x)]`,
//!   extended uniformly by three spacings on each side (the standard
//!   P-spline knot vector). Uniform spacing is what makes the difference
//!   penalty a meaningful roughness measure.
//! - **Penalty order**: 2. Its null space is exactly the set of
//!   coefficient vectors linear in the coefficient index, which — because
//!   this basis reproduces linear functions exactly (see
//!   `basis_reproduces_linear_functions_at_greville_abscissae`) — is the
//!   set of straight lines. So `λ → ∞` collapses the fit to the OLS line
//!   rather than to a constant, and no amount of smoothing can bias the
//!   fit away from a genuinely linear trend.
//! - **Smoothing parameter**: chosen by **GCV**,
//!   `GCV(λ) = n·RSS(λ) / (n − edf(λ))²` with `edf(λ) = tr(B(BᵀB+λS)⁻¹Bᵀ)`,
//!   minimized over a fixed 33-point geometric grid
//!   `λ = λ_scale · 10^e`, `e ∈ {−8, −7.5, …, 8}`. `λ_scale =
//!   tr(BᵀB)/tr(S)` makes the grid invariant to the scale of x and to the
//!   number of observations, so the same 33 candidates are meaningful for
//!   every input. Ties resolve to the smallest λ (the scan is ascending and
//!   the comparison is strict).
//! - **Uncertainty**: the Bayesian posterior covariance
//!   `V_β = σ̂²(BᵀB + λS)⁻¹` with `σ̂² = RSS/(n − edf)` — the
//!   Wahba/Silverman/Nychka covariance that mgcv reports by default, which
//!   accounts for the penalty's shrinkage bias instead of pretending the
//!   fit is unbiased. [`GamFit::std_error`] returns `√(b(x)ᵀ V_β b(x))`;
//!   `smooth.rs` assembles the band as `f̂(x) ± t · se(x)` on `n − edf`
//!   degrees of freedom, matching the t-distribution band its `lm`,
//!   polynomial, `log`, and `sqrt` siblings use (`glm`'s Wald-z band is the
//!   odd one out there, for its own reasons).
//!
//! ## Evaluation outside the data range
//!
//! A B-spline basis is only defined on its knot span. Rather than let a
//! cubic run away outside `[min(x), max(x)]` — which `x_range`
//! (`lmplot(truncate=False)`) can ask for — [`GamFit::predict`] and
//! [`GamFit::std_error`] **hold the boundary value**: the fit is flat, and
//! the band keeps its boundary width, beyond the observed data. This is a
//! deliberate refusal to extrapolate a penalized spline, stated here
//! because it is the one place gam's grid output differs in kind from a
//! polynomial method's.
//!
//! ## Degenerate input
//!
//! [`fit`] returns `None` — never an error — when the data cannot support
//! any smooth at all: fewer than 4 finite `(x, y)` pairs, fewer than 4
//! distinct x values, zero x variance, a non-positive-definite penalized
//! normal-equation matrix at every candidate λ, or no candidate leaving
//! more than [`MIN_RESIDUAL_DOF`] residual degrees of freedom. `smooth.rs`
//! turns `None` into the same all-NaN grid every other method's degenerate
//! branch emits. **gam adds no whole-frame refusal**: the polynomial
//! methods refuse a rank-deficient whole-frame fit because the *caller's*
//! `degree` is what makes the design undeterminable, and gam has no such
//! parameter — its basis dimension is derived from the data, so every gam
//! degeneracy is data degeneracy, which this file's siblings degrade on
//! rather than refuse. See `smooth.rs`'s "Degenerate-input policy".
//!
//! ## Determinism
//!
//! No RNG, no iteration to a tolerance, no parallel reduction: the fit is a
//! fixed 33-candidate scan of closed-form solves, every sum accumulated in
//! a fixed order over `dim ≤ 10` sized problems. Two runs on the same input
//! are byte-identical, which the CLAUDE.md byte-deterministic-render
//! constraint requires and `fit_is_byte_deterministic_across_runs` pins.

use faer::prelude::*;

use crate::transform::linalg::invert_spd;

/// B-spline degree. Cubic is the method's contract, not a tunable.
const SPLINE_DEGREE: usize = 3;
/// Number of non-zero basis functions at any x (`SPLINE_DEGREE + 1`).
const SPAN: usize = SPLINE_DEGREE + 1;
/// Largest basis dimension, matching mgcv's default `k = 10` for a single
/// smooth term. The penalty, not the basis size, controls smoothness.
const MAX_BASIS_DIM: usize = 10;
/// Smallest basis dimension: one cubic segment.
const MIN_BASIS_DIM: usize = SPAN;
/// Difference-penalty order. See the module doc's null-space note.
const PENALTY_ORDER: usize = 2;
/// GCV λ grid: `λ_scale · 10^e` for `e` from this exponent, in
/// [`LAMBDA_LOG10_STEP`] increments, [`LAMBDA_GRID_LEN`] candidates.
const LAMBDA_LOG10_MIN: f64 = -8.0;
const LAMBDA_LOG10_STEP: f64 = 0.5;
const LAMBDA_GRID_LEN: usize = 33;
/// A candidate λ must leave more than this many residual degrees of
/// freedom for `σ̂² = RSS/(n − edf)` to mean anything. Candidates below it
/// are skipped, which is how an interpolating (λ ≈ 0) fit on a small
/// sample is rejected in favor of a smoother one.
const MIN_RESIDUAL_DOF: f64 = 1.0;

/// A cubic B-spline basis on uniformly spaced knots over `[x_min, x_max]`.
///
/// Knots are implicit: `t_j = x_min + (j − SPLINE_DEGREE)·h` for
/// `j = 0 ..= n_segments + 2·SPLINE_DEGREE`, so the basis has
/// `dim = n_segments + SPLINE_DEGREE` functions and `t_SPLINE_DEGREE`
/// lands exactly on `x_min`.
#[derive(Debug, Clone)]
struct SplineBasis {
    x_min: f64,
    /// Knot spacing `(x_max − x_min) / n_segments`.
    h: f64,
    dim: usize,
}

impl SplineBasis {
    /// `dim` must be `>= MIN_BASIS_DIM` and `x_max > x_min` (both are
    /// checked by [`fit`], the only constructor's caller).
    fn new(x_min: f64, x_max: f64, dim: usize) -> Self {
        let n_segments = dim - SPLINE_DEGREE;
        Self { x_min, h: (x_max - x_min) / n_segments as f64, dim }
    }

    fn n_segments(&self) -> usize {
        self.dim - SPLINE_DEGREE
    }

    /// The `SPAN` non-zero basis values at `x`, with the index of the first.
    ///
    /// Uniform knots collapse the Cox–de Boor recursion to the classical
    /// closed-form cubic blending polynomials in the local coordinate
    /// `v ∈ [0, 1]`, so this is exact arithmetic with no recursion. `x`
    /// outside `[x_min, x_max]` is clamped to the boundary segment — the
    /// documented no-extrapolation policy.
    fn row(&self, x: f64) -> (usize, [f64; SPAN]) {
        let n_segments = self.n_segments();
        let u = ((x - self.x_min) / self.h).clamp(0.0, n_segments as f64);
        let segment = (u.floor() as usize).min(n_segments - 1);
        let v = u - segment as f64;
        let w = 1.0 - v;
        (
            segment,
            [
                w * w * w / 6.0,
                (3.0 * v * v * v - 6.0 * v * v + 4.0) / 6.0,
                (-3.0 * v * v * v + 3.0 * v * v + 3.0 * v + 1.0) / 6.0,
                v * v * v / 6.0,
            ],
        )
    }
}

/// A fitted penalized cubic B-spline, ready to evaluate anywhere.
#[derive(Debug, Clone)]
pub(crate) struct GamFit {
    basis: SplineBasis,
    beta: Vec<f64>,
    /// Bayesian posterior covariance `σ̂²(BᵀB + λS)⁻¹`, row-major `dim×dim`.
    cov: Vec<f64>,
    residual_dof: f64,
}

impl GamFit {
    /// `f̂(x)`. NaN for non-finite `x`; boundary-held outside the data range.
    pub(crate) fn predict(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        let (start, values) = self.basis.row(x);
        (0..SPAN).map(|a| self.beta[start + a] * values[a]).sum()
    }

    /// `√(b(x)ᵀ V_β b(x))` — the posterior standard error of `f̂(x)`, which
    /// `smooth.rs` scales by a t-critical value to build the CI band.
    pub(crate) fn std_error(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        let (start, values) = self.basis.row(x);
        let dim = self.basis.dim;
        let mut variance = 0.0;
        for a in 0..SPAN {
            for b in 0..SPAN {
                variance += values[a] * values[b] * self.cov[(start + a) * dim + start + b];
            }
        }
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    }

    /// `n − edf`: the residual degrees of freedom the CI band's
    /// t-distribution uses. Fractional, as a penalized fit's always is.
    pub(crate) fn residual_dof(&self) -> f64 {
        self.residual_dof
    }
}

/// One λ's penalized solve: `β = (BᵀB + λS)⁻¹Bᵀy`, the inverse itself
/// (needed for both the covariance and the effective degrees of freedom),
/// and `edf = tr((BᵀB + λS)⁻¹BᵀB)`.
struct PenalizedSolve {
    beta: Vec<f64>,
    /// `(BᵀB + λS)⁻¹`, row-major `dim×dim`.
    inverse: Vec<f64>,
    edf: f64,
}

/// The GCV-selected candidate, carrying everything [`fit`] needs to finish.
struct GcvCandidate {
    gcv: f64,
    solve: PenalizedSolve,
    rss: f64,
    residual_dof: f64,
}

/// Fit a penalized cubic B-spline to `(xs, ys)`.
///
/// Returns `None` for degenerate input rather than an error — see the
/// module doc's "Degenerate input" section for the exhaustive list and for
/// why gam contributes no whole-frame refusal of its own.
pub(crate) fn fit(xs: &[f64], ys: &[f64]) -> Option<GamFit> {
    let pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();
    let n = pairs.len();
    if n < MIN_BASIS_DIM {
        return None;
    }

    // Both extremes are finite: `pairs` is non-empty and filtered to finite
    // values, so `<=` is a total comparison here (no NaN can reach it).
    let x_min = pairs.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = pairs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    if x_max <= x_min {
        return None;
    }

    let mut distinct: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    distinct.dedup();
    if distinct.len() < MIN_BASIS_DIM {
        return None;
    }

    let dim = distinct.len().min(MAX_BASIS_DIM);
    let basis = SplineBasis::new(x_min, x_max, dim);
    let rows: Vec<(usize, [f64; SPAN])> = pairs.iter().map(|&(x, _)| basis.row(x)).collect();
    let (btb, bty) = normal_equations(&rows, &pairs, dim);
    let penalty = difference_penalty(dim);
    let trace_btb: f64 = (0..dim).map(|i| btb[i * dim + i]).sum();
    let trace_penalty: f64 = (0..dim).map(|i| penalty[i * dim + i]).sum();
    let lambda_scale = if trace_btb > 0.0 && trace_penalty > 0.0 {
        trace_btb / trace_penalty
    } else {
        1.0
    };

    let mut best: Option<GcvCandidate> = None;
    for step in 0..LAMBDA_GRID_LEN {
        let lambda =
            lambda_scale * 10f64.powf(LAMBDA_LOG10_MIN + LAMBDA_LOG10_STEP * step as f64);
        let Some(solve) = solve_penalized(&btb, &bty, &penalty, dim, lambda) else {
            continue;
        };
        // `solve_penalized` already rejected a non-finite `edf`, so this is
        // a total comparison.
        let residual_dof = n as f64 - solve.edf;
        if residual_dof <= MIN_RESIDUAL_DOF {
            continue;
        }
        let rss = residual_sum_of_squares(&rows, &pairs, &solve.beta);
        if !rss.is_finite() {
            continue;
        }
        let gcv = n as f64 * rss / (residual_dof * residual_dof);
        if !gcv.is_finite() {
            continue;
        }
        // Strict `<` over an ascending scan: ties keep the smallest λ.
        if best.as_ref().is_none_or(|b| gcv < b.gcv) {
            best = Some(GcvCandidate { gcv, solve, rss, residual_dof });
        }
    }

    let best = best?;
    let sigma2 = best.rss / best.residual_dof;
    let cov: Vec<f64> = best.solve.inverse.iter().map(|v| v * sigma2).collect();
    Some(GamFit {
        basis,
        beta: best.solve.beta,
        cov,
        residual_dof: best.residual_dof,
    })
}

/// `(BᵀB, Bᵀy)` for the sparse spline design, accumulated over each row's
/// `SPAN` non-zero entries only (the basis is banded, so the full `n×dim`
/// matrix is never materialized). Both outputs are row-major `dim×dim` and
/// `dim`-length respectively.
///
/// Shared with the λ-behavior tests below so they exercise the same design
/// this function builds for production rather than a hand-rolled mirror of
/// it — a mirror could drift from the real basis and still "pass".
fn normal_equations(
    rows: &[(usize, [f64; SPAN])],
    pairs: &[(f64, f64)],
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut btb = vec![0.0; dim * dim];
    let mut bty = vec![0.0; dim];
    for (&(start, values), &(_, y)) in rows.iter().zip(pairs.iter()) {
        for a in 0..SPAN {
            bty[start + a] += values[a] * y;
            for b in 0..SPAN {
                btb[(start + a) * dim + start + b] += values[a] * values[b];
            }
        }
    }
    (btb, bty)
}

/// `S = D_kᵀD_k` for the order-`PENALTY_ORDER` difference operator, as a
/// row-major `dim×dim` matrix.
fn difference_penalty(dim: usize) -> Vec<f64> {
    // Order-2 difference stencil: [1, -2, 1].
    const STENCIL: [f64; PENALTY_ORDER + 1] = [1.0, -2.0, 1.0];
    let mut penalty = vec![0.0; dim * dim];
    for row in 0..dim.saturating_sub(PENALTY_ORDER) {
        for (a, &da) in STENCIL.iter().enumerate() {
            for (b, &db) in STENCIL.iter().enumerate() {
                penalty[(row + a) * dim + row + b] += da * db;
            }
        }
    }
    penalty
}

/// Solve `(BᵀB + λS)β = Bᵀy`, returning the inverse and effective degrees
/// of freedom alongside. `None` when the matrix is not numerically
/// positive-definite — the caller skips that λ.
fn solve_penalized(
    btb: &[f64],
    bty: &[f64],
    penalty: &[f64],
    dim: usize,
    lambda: f64,
) -> Option<PenalizedSolve> {
    let mut a = Mat::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            a[(i, j)] = btb[i * dim + j] + lambda * penalty[i * dim + j];
        }
    }
    let inverse_mat = invert_spd(&a)?;
    let mut inverse = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            inverse[i * dim + j] = inverse_mat[(i, j)];
        }
    }

    let beta: Vec<f64> = (0..dim)
        .map(|i| (0..dim).map(|j| inverse[i * dim + j] * bty[j]).sum())
        .collect();
    if !beta.iter().all(|v| v.is_finite()) {
        return None;
    }

    // edf = tr(A⁻¹BᵀB); both matrices are symmetric, so the transpose in
    // the trace's inner product is immaterial, but index it honestly.
    let edf: f64 = (0..dim)
        .map(|i| (0..dim).map(|j| inverse[i * dim + j] * btb[j * dim + i]).sum::<f64>())
        .sum();
    if !edf.is_finite() {
        return None;
    }

    Some(PenalizedSolve { beta, inverse, edf })
}

fn residual_sum_of_squares(
    rows: &[(usize, [f64; SPAN])],
    pairs: &[(f64, f64)],
    beta: &[f64],
) -> f64 {
    rows.iter()
        .zip(pairs.iter())
        .map(|(&(start, values), &(_, y))| {
            let fitted: f64 = (0..SPAN).map(|a| beta[start + a] * values[a]).sum();
            (y - fitted) * (y - fitted)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every B-spline basis is a partition of unity — the single property
    /// that makes `Σ β_j B_j(x)` an interpolation rather than an arbitrary
    /// weighted sum. Closed-form, exact.
    #[test]
    fn basis_is_a_partition_of_unity() {
        let basis = SplineBasis::new(-2.0, 7.0, 10);
        for i in 0..=400 {
            let x = -2.0 + 9.0 * i as f64 / 400.0;
            let (_, values) = basis.row(x);
            let sum: f64 = values.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "Σ B_j({x}) = {sum}, expected 1");
        }
    }

    /// Linear precision: `Σ ξ_j B_j(x) = x` where `ξ_j` is basis function
    /// `j`'s Greville abscissa (`x_min + h·(j − 1)` for this uniform cubic
    /// knot vector). This is what makes the order-2 difference penalty's
    /// null space *the straight lines* — see the module doc — so it is a
    /// load-bearing property, not a basis trivia check.
    #[test]
    fn basis_reproduces_linear_functions_at_greville_abscissae() {
        let basis = SplineBasis::new(1.5, 9.5, 7);
        for i in 0..=200 {
            let x = 1.5 + 8.0 * i as f64 / 200.0;
            let (start, values) = basis.row(x);
            let reconstructed: f64 = (0..SPAN)
                .map(|a| {
                    let j = start + a;
                    let greville = basis.x_min + basis.h * (j as f64 - 1.0);
                    greville * values[a]
                })
                .sum();
            assert!(
                (reconstructed - x).abs() < 1e-10,
                "Σ ξ_j B_j({x}) = {reconstructed}"
            );
        }
    }

    /// Closed form: an exactly linear response is in both the basis span
    /// AND the penalty's null space, so *every* λ reproduces it exactly and
    /// the fit must return the generating line to machine precision.
    #[test]
    fn fit_reproduces_an_exactly_linear_function() {
        let xs: Vec<f64> = (0..25).map(|i| i as f64 * 0.4).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 - 2.0 * x).collect();
        let fit = fit(&xs, &ys).expect("linear data must fit");
        for i in 0..=50 {
            let x = 9.6 * i as f64 / 50.0;
            let expected = 3.0 - 2.0 * x;
            assert!(
                (fit.predict(x) - expected).abs() < 1e-8,
                "f({x}) = {}, expected {expected}",
                fit.predict(x)
            );
        }
    }

    /// The constant case of the same closed form.
    #[test]
    fn fit_reproduces_a_constant() {
        let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let ys = vec![4.25; 12];
        let fit = fit(&xs, &ys).expect("constant data must fit");
        for i in 0..=20 {
            let x = 11.0 * i as f64 / 20.0;
            assert!((fit.predict(x) - 4.25).abs() < 1e-9, "f({x}) = {}", fit.predict(x));
        }
    }

    /// The penalty's null space, verified directly: at a huge λ the
    /// penalized fit must converge to the ordinary least-squares LINE (not
    /// to a constant, and not to zero), on data that is genuinely curved.
    /// The expected line is computed in the test from the closed-form OLS
    /// formulas, independent of any production code.
    #[test]
    fn infinite_lambda_collapses_to_the_ols_line() {
        let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.25).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 1.3).sin() * 4.0 + 0.5 * x).collect();

        // Closed-form OLS on (xs, ys).
        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let sxx: f64 = xs.iter().map(|x| (x - mean_x) * (x - mean_x)).sum();
        let sxy: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let slope = sxy / sxx;
        let intercept = mean_y - slope * mean_x;

        // Build the design through the production helper, then force a huge λ.
        let dim = MAX_BASIS_DIM;
        let basis = SplineBasis::new(0.0, 7.25, dim);
        let pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
        let rows: Vec<(usize, [f64; SPAN])> = xs.iter().map(|&x| basis.row(x)).collect();
        let (btb, bty) = normal_equations(&rows, &pairs, dim);
        let penalty = difference_penalty(dim);
        let trace_btb: f64 = (0..dim).map(|i| btb[i * dim + i]).sum();
        let trace_penalty: f64 = (0..dim).map(|i| penalty[i * dim + i]).sum();
        let lambda = 1e12 * trace_btb / trace_penalty;
        let solve = solve_penalized(&btb, &bty, &penalty, dim, lambda).expect("solve");

        for i in 0..=20 {
            let x = 7.25 * i as f64 / 20.0;
            let (start, values) = basis.row(x);
            let fitted: f64 = (0..SPAN).map(|a| solve.beta[start + a] * values[a]).sum();
            let expected = intercept + slope * x;
            assert!(
                (fitted - expected).abs() < 1e-4,
                "λ→∞ fit at x={x} is {fitted}, OLS line gives {expected}"
            );
        }
        // ...and the effective degrees of freedom collapse to the null
        // space's dimension: intercept + slope = 2.
        assert!(
            (solve.edf - 2.0).abs() < 1e-3,
            "λ→∞ edf = {}, expected 2 (the linear null space)",
            solve.edf
        );
    }

    /// The other end of the same dial: a tiny λ leaves the basis nearly
    /// unpenalized, so it must track the data more closely than a large λ.
    #[test]
    fn smaller_lambda_tracks_the_data_more_closely() {
        let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 1.1).sin() * 3.0).collect();
        let dim = MAX_BASIS_DIM;
        let basis = SplineBasis::new(0.0, 7.8, dim);
        let rows: Vec<(usize, [f64; SPAN])> = xs.iter().map(|&x| basis.row(x)).collect();
        let pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
        let (btb, bty) = normal_equations(&rows, &pairs, dim);
        let penalty = difference_penalty(dim);
        let scale: f64 = (0..dim).map(|i| btb[i * dim + i]).sum::<f64>()
            / (0..dim).map(|i| penalty[i * dim + i]).sum::<f64>();

        let loose = solve_penalized(&btb, &bty, &penalty, dim, scale * 1e-8).expect("loose");
        let tight = solve_penalized(&btb, &bty, &penalty, dim, scale * 1e4).expect("tight");
        let loose_rss = residual_sum_of_squares(&rows, &pairs, &loose.beta);
        let tight_rss = residual_sum_of_squares(&rows, &pairs, &tight.beta);
        assert!(
            loose_rss < tight_rss,
            "small λ RSS {loose_rss} should be below large λ RSS {tight_rss}"
        );
        assert!(
            loose.edf > tight.edf,
            "small λ edf {} should exceed large λ edf {}",
            loose.edf,
            tight.edf
        );
    }

    /// GCV must actually pick a middle course: on curved data it spends
    /// more than the 2 degrees of freedom of a straight line, and fewer
    /// than the basis dimension (which would be an interpolation).
    #[test]
    fn gcv_selects_effective_df_between_a_line_and_an_interpolation() {
        let xs: Vec<f64> = (0..60).map(|i| i as f64 * 0.15).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 1.2).sin() * 5.0).collect();
        let fit = fit(&xs, &ys).expect("curved data must fit");
        let edf = xs.len() as f64 - fit.residual_dof();
        assert!(
            edf > 2.5,
            "GCV should spend more than a straight line's 2 df on curved data; edf = {edf}"
        );
        assert!(
            edf <= MAX_BASIS_DIM as f64 + 1e-9,
            "edf {edf} cannot exceed the basis dimension"
        );
    }

    /// Wigglier data should buy more effective degrees of freedom than
    /// near-linear data at the same sample size — the GCV selection is
    /// responding to the signal, not returning a fixed λ.
    #[test]
    fn gcv_spends_more_df_on_wigglier_data() {
        let xs: Vec<f64> = (0..60).map(|i| i as f64 * 0.15).collect();
        let straight: Vec<f64> = xs.iter().map(|x| 2.0 + 0.5 * x).collect();
        let wiggly: Vec<f64> = xs.iter().map(|x| (x * 2.2).sin() * 5.0).collect();
        let straight_fit = fit(&xs, &straight).expect("linear fit");
        let wiggly_fit = fit(&xs, &wiggly).expect("wiggly fit");
        let straight_edf = xs.len() as f64 - straight_fit.residual_dof();
        let wiggly_edf = xs.len() as f64 - wiggly_fit.residual_dof();
        assert!(
            wiggly_edf > straight_edf + 1.0,
            "wiggly edf {wiggly_edf} should clearly exceed straight-line edf {straight_edf}"
        );
    }

    #[test]
    fn fewer_than_four_distinct_x_returns_none() {
        let xs = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(fit(&xs, &ys).is_none(), "3 distinct x cannot support a cubic basis");
    }

    #[test]
    fn constant_x_returns_none() {
        let xs = vec![5.0; 10];
        let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(fit(&xs, &ys).is_none());
    }

    #[test]
    fn too_few_points_returns_none() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        assert!(fit(&xs, &ys).is_none());
    }

    #[test]
    fn non_finite_pairs_are_dropped_not_propagated() {
        let mut xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|x| x * 2.0).collect();
        xs.push(f64::INFINITY);
        ys.push(1.0);
        let fit = fit(&xs, &ys).expect("the finite majority must still fit");
        assert!(fit.predict(10.0).is_finite(), "an infinite x must not poison the fit");
        assert!((fit.predict(10.0) - 20.0).abs() < 1e-8);
    }

    /// The documented no-extrapolation policy: outside `[min(x), max(x)]`
    /// both the fit and the band hold their boundary value.
    #[test]
    fn evaluation_outside_the_data_range_holds_the_boundary_value() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 3.0 * x).collect();
        let fit = fit(&xs, &ys).expect("fit");
        let at_low = fit.predict(0.0);
        let at_high = fit.predict(19.0);
        assert!((fit.predict(-50.0) - at_low).abs() < 1e-9, "left of the data must hold f(x_min)");
        assert!((fit.predict(999.0) - at_high).abs() < 1e-9, "right of the data must hold f(x_max)");
        assert!((fit.std_error(-50.0) - fit.std_error(0.0)).abs() < 1e-9);
        assert!(fit.predict(f64::NAN).is_nan());
        assert!(fit.std_error(f64::NAN).is_nan());
    }

    /// Noisier data must widen the posterior band: `V_β` scales with `σ̂²`.
    #[test]
    fn standard_error_grows_with_residual_noise() {
        let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
        let quiet: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| 2.0 * x + if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        let noisy: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| 2.0 * x + if i % 2 == 0 { 2.5 } else { -2.5 })
            .collect();
        let quiet_fit = fit(&xs, &quiet).expect("quiet fit");
        let noisy_fit = fit(&xs, &noisy).expect("noisy fit");
        let x = 5.0;
        assert!(
            noisy_fit.std_error(x) > quiet_fit.std_error(x) * 5.0,
            "noisy se {} should dwarf quiet se {}",
            noisy_fit.std_error(x),
            quiet_fit.std_error(x)
        );
        assert!(quiet_fit.std_error(x) > 0.0);
    }

    /// No RNG, no tolerance loop: repeated fits on the same input must be
    /// bit-identical, which the byte-deterministic-render constraint needs.
    #[test]
    fn fit_is_byte_deterministic_across_runs() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.17).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 1.7).sin() * 3.0 + x * 0.4).collect();
        let a = fit(&xs, &ys).expect("fit a");
        let b = fit(&xs, &ys).expect("fit b");
        assert_eq!(a.residual_dof().to_bits(), b.residual_dof().to_bits());
        for i in 0..=100 {
            let x = 8.33 * i as f64 / 100.0;
            assert_eq!(
                a.predict(x).to_bits(),
                b.predict(x).to_bits(),
                "predictions must be bit-identical at x={x}"
            );
            assert_eq!(a.std_error(x).to_bits(), b.std_error(x).to_bits());
        }
    }

    /// The whole point of the method: a penalized spline must follow a
    /// multi-turn curve that no single global polynomial of low degree can.
    #[test]
    fn fit_tracks_a_multi_turn_curve() {
        let xs: Vec<f64> = (0..80).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x * 1.5).sin() * 4.0).collect();
        let fit = fit(&xs, &ys).expect("fit");
        let max_error = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| (fit.predict(x) - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 0.5, "max |f̂ − y| = {max_error} on a smooth sine");
    }
}
