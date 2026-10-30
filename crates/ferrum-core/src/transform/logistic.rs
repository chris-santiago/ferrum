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

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::Schema;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::{TransformContext, TransformWarning, Unfittable};
use crate::transform::context::describe_group_keys;
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, parse_groupby_pyany, pooled_group_key,
    stack_grouped_batches, FitRows, KeyValue,
    NullKeyPolicy,
};
use crate::transform::numeric_util::coerce_xy_pairs_at;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LogisticSpec {
    pub x: String,
    pub y: String,
    pub n_grid: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub max_iter: usize,
    pub tol: f64,
    /// When non-empty, partition the input by these columns and fit each
    /// group on its own rows, appending the group-key columns (with their
    /// original dtypes) to the output, as `Robust(groupby=)` does. A null key
    /// belongs to no group; a batch whose every key is null is fit whole.
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

/// Entry point with no render context: [`apply_with_context`] with a
/// default context.
pub(crate) fn apply(spec: &LogisticSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Fit the whole batch, or, under `spec.groupby`, each group on its own rows
/// and stack the per-group curves with the group columns appended, as
/// `robust::apply_grouped` does. `SkipWithWholeBatchFallback` is the
/// partition's policy: a batch whose every key is null is fit whole, and the
/// partition's warning says so.
pub(crate) fn apply_with_context(
    spec: &LogisticSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if spec.groupby.is_empty() {
        return fit(spec, batch, None, ctx);
    }
    let partition = group_partition_multi(
        batch,
        &spec.groupby,
        "Logistic",
        NullKeyPolicy::SkipWithWholeBatchFallback,
        ctx,
    )?;
    let mut per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = Vec::new();
    for (key, rows) in &partition.groups {
        let group = FitRows { indices: rows, label: describe_group_keys(&spec.groupby, key) };
        per_group_batches.push((key.clone(), fit(spec, batch, Some(&group), ctx)?));
    }
    if per_group_batches.is_empty() {
        // Every group key was null: the pooled fit IS the null-keyed group,
        // stacked with a key that nulls only the `groupby` column(s)
        // actually all-null (`pooled_group_key`'s doc), so a facet field
        // folded into `spec.groupby` keeps its value, as
        // `robust::apply_grouped`'s fallback does.
        let pooled = fit(spec, batch, None, ctx)?;
        let key = pooled_group_key(batch, &spec.groupby)?;
        return stack_grouped_batches("Logistic", &[(key, pooled)], &spec.groupby, &partition.dtypes);
    }
    stack_grouped_batches("Logistic", &per_group_batches, &spec.groupby, &partition.dtypes)
}

/// Fit `batch`'s rows, or only one group's (`group`). Rows that cannot be fit
/// refuse the whole frame and degrade a group to its empty fit, with a
/// warning ([`TransformContext::degrade_group_or_refuse`]).
fn fit(
    spec: &LogisticSpec,
    batch: &RecordBatch,
    group: Option<&FitRows<'_>>,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let only_indices = group.map(|group| group.indices);
    let unfittable = |failure: Unfittable| -> PyResult<RecordBatch> {
        ctx.degrade_group_or_refuse("Logistic", group.map(|group| group.label.as_str()), failure)?;
        let nans = vec![f64::NAN; spec.n_grid.max(1)];
        build_logistic_batch(nans.clone(), nans.clone(), nans.clone(), nans)
    };
    let (xs, ys) = coerce_xy_pairs_at(batch, &spec.x, &spec.y, "Logistic", only_indices)?;
    if let Some(y) = ys.iter().find(|&&y| y != 0.0 && y != 1.0) {
        return unfittable(Unfittable {
            reason: format!("y is not binary (found {y})"),
            refusal: format!("Logistic: y must be binary (only 0 or 1); found value {y}"),
        });
    }

    if xs.len() < 2 {
        return unfittable(Unfittable {
            reason: "fewer than 2 usable (non-null/non-NaN) x/y points".into(),
            refusal: "Logistic: need at least 2 non-null observations".into(),
        });
    }

    // Singular design: zero variance in x => X'WX is singular.
    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if !(x_max > x_min) {
        return unfittable(Unfittable {
            reason: "zero x variance".into(),
            refusal: "Logistic: singular design (X'WX has zero determinant); x has zero variance".into(),
        });
    }

    let (beta, cov) = match fit_irls(&xs, &ys, spec.max_iter, spec.tol) {
        Ok(fit) => fit,
        Err(failure) => return unfittable(failure),
    };

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
            let mut undefined = 0usize;
            for &xq in &grid {
                // η = β0 + β1 x; Var(η) = [1, x] Σ [1, x]'
                let v = cov[0][0]
                    + 2.0 * xq * cov[0][1]
                    + xq * xq * cov[1][1];
                let se = if v >= 0.0 { v.sqrt() } else { undefined += 1; f64::NAN };
                let eta = beta[0] + beta[1] * xq;
                lo.push(sigmoid(eta - z * se));
                hi.push(sigmoid(eta + z * se));
            }
            // The fit line drew fine (IRLS converged); a negative Var(η)
            // at some grid points is a numerical artifact of the Wald
            // covariance (x's spread tiny next to its magnitude), not a
            // lost fit, so only the band there is undefined.
            if undefined > 0 {
                let reason = crate::transform::context::partial_band_reason(
                    "the Wald variance is negative (x's spread is tiny next to its magnitude; \
                     centering x avoids it)",
                    undefined,
                    n_grid,
                );
                ctx.warn(TransformWarning::IntervalUndefined {
                    transform: "Logistic".into(),
                    group: group.map(|group| group.label.clone()),
                    ci: level,
                    reason,
                    panel: None,
                });
            }
            (lo, hi)
        }
    };

    build_logistic_batch(grid, y_fit, lo, hi)
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

/// Fit logistic regression β = (β₀, β₁) by IRLS. Returns (β, Σ̂), or why
/// the rows cannot be fit (a singular weighted design, perfect separation).
fn fit_irls(
    xs: &[f64],
    ys: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<([f64; 2], [[f64; 2]; 2]), Unfittable> {
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

        let inv = invert_2x2(wxtx)
            .ok_or_else(|| Unfittable::new("Logistic", "singular design (X'WX has zero determinant)"))?;

        let b0 = inv[0][0] * wxtz[0] + inv[0][1] * wxtz[1];
        let b1 = inv[1][0] * wxtz[0] + inv[1][1] * wxtz[1];
        beta = [b0, b1];
        last_cov = Some(inv);

        // Perfect-separation guard: blow-up in coefficients.
        if beta[0].abs() > 1e3 || beta[1].abs() > 1e3 {
            return Err(Unfittable::new("Logistic", "perfect separation detected (coefficients diverged)"));
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
                return Err(Unfittable::new(
                    "Logistic",
                    "perfect separation detected (fitted probabilities saturated)",
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
fn check_separation_at_convergence(ys: &[f64], mu: &[f64]) -> Result<(), Unfittable> {
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
        return Err(Unfittable::new(
            "Logistic",
            "perfect separation detected (model classifies every observation with probability > 0.999)",
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

use crate::transform::linalg::invert_2x2;

fn build_logistic_batch(
    x: Vec<f64>, y: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>,
) -> PyResult<RecordBatch> {
    // Batch-D M6 schema-collapse (spec §4.6): shared with `smooth.rs`
    // (Fitted)/`glm.rs`/`robust.rs` (Fitted) and the `transform_registry`
    // entries for all four.
    let schema = Arc::new(Schema::new(crate::transform::registry::fit_output_fields()));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x)),
        Arc::new(Float64Array::from(y)),
        Arc::new(Float64Array::from(lo)),
        Arc::new(Float64Array::from(hi)),
    ];
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("Logistic: {e}")))
}

/// Inverse standard normal CDF (Beasley-Springer / Moro).
pub(crate) fn inv_normal_cdf(p: f64) -> f64 {
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

/// Logistic regression fit with optional confidence band.
///
/// Fits a binary logistic regression model ``P(y=1|x)`` via IRLS (Binomial
/// family, logit link) and evaluates the fitted probability on an evenly-
/// spaced grid. When ``ci`` is set, also produces Wald confidence bands.
///
/// Equivalent to ``Glm(family="binomial", link="logit")`` but provided as
/// a convenience class with a simpler signature.
///
/// Output columns: ``x`` (Float64), ``y`` (Float64 fitted probability),
/// ``ci_lower`` and ``ci_upper`` (Float64; ``NaN`` when ``ci`` is not set).
///
/// Parameters
/// ----------
/// x : str
///     Predictor column (any integer, unsigned, or float dtype; widened to
///     Float64). A non-numeric column raises, naming the transform, column,
///     and dtype.
/// y : str
///     Binary response column (any integer, unsigned, or float dtype;
///     widened to Float64; values should be 0 or 1). A non-numeric column
///     raises, naming the transform, column, and dtype.
/// n_grid : int, default 100
///     Number of evenly-spaced grid points to evaluate. Must be > 0.
/// ci : float, optional
///     Confidence level in (0, 1) for Wald bands (e.g. 0.95). When
///     omitted, ``ci_lower`` and ``ci_upper`` are ``NaN``.
/// max_iter : int, default 25
///     Maximum IRLS iterations. Must be > 0.
/// tol : float, default 1e-8
///     Convergence tolerance (relative change in coefficients). Must be
///     finite and > 0.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_smooth(method="logistic").encode(x="x", y="binary_y")
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
        groupby = None,
        name = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: &str,
        y: &str,
        n_grid: usize,
        ci: Option<f64>,
        max_iter: usize,
        tol: f64,
        groupby: Option<&Bound<'_, PyAny>>,
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
        let groupby = parse_groupby_pyany(groupby)?;
        Ok(PyLogistic(TransformSpec::Logistic(LogisticSpec {
            x: x.to_string(),
            y: y.to_string(),
            n_grid,
            ci,
            max_iter,
            tol,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Logistic(s) => format!(
                "Logistic(x='{}', y='{}', n_grid={}, ci={:?}, max_iter={}, tol={}, groupby={:?}, name={:?})",
                s.x, s.y, s.n_grid, s.ci, s.max_iter, s.tol, s.groupby, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, RecordBatch};
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
            n_grid, ci, max_iter: 25, tol: 1e-8, groupby: Vec::new(), name: None,
        }
    }

    #[test]
    fn test_logistic_round_trip_json() {
        let original = LogisticSpec {
            x: "x".into(), y: "y".into(),
            n_grid: 100, ci: Some(0.95), max_iter: 25, tol: 1e-8, groupby: Vec::new(), name: None,
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
            n_grid: 10, ci: None, max_iter: 25, tol: 1e-8, groupby: Vec::new(), name: None,
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
            n_grid: 10, ci: None, max_iter: 100, tol: 1e-14, groupby: Vec::new(), name: None,
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

    /// `x`, `y` and a Utf8 group `g` (nullable).
    fn grouped_batch(x: Vec<f64>, y: Vec<f64>, g: Vec<Option<&str>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
            Arc::new(arrow::array::StringArray::from(g)),
        ]).unwrap()
    }

    fn str_col(b: &RecordBatch, name: &str) -> Vec<String> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    /// Under `groupby`, each group is fit on its own rows: the grouped
    /// output is each group's ungrouped fit stacked, with the group column
    /// appended. Rejects one pooled fit over both groups.
    #[test]
    fn grouped_fit_is_each_groups_own_fit() {
        pyo3::Python::initialize();
        let (xa, ya, xb, yb): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0], vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        let n = xa.len();
        let mut grouped = default_spec(10, None);
        grouped.groupby = vec!["g".into()];
        let batch = grouped_batch(
            xa.iter().chain(&xb).copied().collect(),
            ya.iter().chain(&yb).copied().collect(),
            std::iter::repeat_n(Some("a"), n).chain(std::iter::repeat_n(Some("b"), xb.len())).collect(),
        );
        let out = apply(&grouped, &batch).unwrap();
        let ungrouped = default_spec(10, None);
        let a = apply(&ungrouped, &xy_batch(xa, ya)).unwrap();
        let b = apply(&ungrouped, &xy_batch(xb, yb)).unwrap();
        let expected: Vec<f64> = col(&a, "y").into_iter().chain(col(&b, "y")).collect();
        assert_eq!(col(&out, "y"), expected);
        let groups = str_col(&out, "g");
        assert_eq!(groups.iter().filter(|g| *g == "a").count(), a.num_rows());
        assert_eq!(groups.iter().filter(|g| *g == "b").count(), b.num_rows());
    }

    /// A null key belongs to no group; a batch whose every key is null is
    /// fit whole, as the ungrouped fit, and the partition says so. The
    /// pooled fit IS the null-keyed group: its output ALSO carries `g`, all
    /// null at its original dtype, rather than lacking it.
    #[test]
    fn all_null_keys_fit_the_whole_batch_and_warn() {
        pyo3::Python::initialize();
        let (xa, ya, _, _): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0], vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        let mut grouped = default_spec(10, None);
        grouped.groupby = vec!["g".into()];
        let batch = grouped_batch(xa.clone(), ya.clone(), vec![None; xa.len()]);
        let ctx = TransformContext::default();
        let out = apply_with_context(&grouped, &batch, &ctx).unwrap();
        let whole = apply(&default_spec(10, None), &xy_batch(xa, ya)).unwrap();
        assert_eq!(col(&out, "y"), col(&whole, "y"));

        let g = out.column(out.schema().index_of("g").unwrap());
        assert_eq!(g.data_type(), &DataType::Utf8, "the groupby column keeps its original dtype");
        assert_eq!(g.null_count(), g.len(), "the pooled fit IS the null-keyed group");

        let said: Vec<String> = ctx.warnings.borrow().iter().map(|w| w.to_string()).collect();
        assert_eq!(
            said,
            ["Logistic: every row has a null in its groupby key (g), so the data could not be split into groups; \
              the whole batch was used as one ungrouped computation instead"]
        );
    }

    /// The PyO3 constructor and the wire both accept `groupby`.
    #[test]
    fn groupby_round_trips_on_the_wire() {
        let mut grouped = default_spec(10, None);
        grouped.groupby = vec!["g".into()];
        let json = serde_json::to_string(&grouped).unwrap();
        assert!(json.contains("\"groupby\":[\"g\"]"), "{json}");
        let parsed: LogisticSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, grouped);
        let ungrouped = serde_json::to_string(&default_spec(10, None)).unwrap();
        assert!(!ungrouped.contains("groupby"), "an ungrouped spec serializes as before: {ungrouped}");
    }

    /// A group that cannot be fit draws no fit (its curve is all NaN) and a
    /// warning names it and the reason, while the other group fits as it
    /// does alone, as `Smooth`'s and `Robust`'s groups degrade. The same rows
    /// ungrouped still refuse.
    #[test]
    fn a_group_that_cannot_be_fit_degrades_with_a_warning_and_the_others_draw() {
        pyo3::Python::initialize();
        let (gx, gy): (Vec<f64>, Vec<f64>) = (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let n = gx.len();
        let mut grouped = default_spec(10, None);
        grouped.groupby = vec!["g".into()];
        let cases: Vec<(&str, Vec<f64>, Vec<f64>, &str)> = vec![
            ("one point", vec![1.0], vec![gy[0]], "fewer than 2 usable (non-null/non-NaN) x/y points"),
            ("constant x", vec![2.0, 2.0, 2.0], vec![gy[0], gy[1], gy[2]], "zero x variance"),
            ("non-binary y", vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 1.0], "y is not binary (found 2)"),
        ];
        for (label, bx, by, reason) in cases {
            let batch = grouped_batch(
                gx.iter().chain(&bx).copied().collect(),
                gy.iter().chain(&by).copied().collect(),
                std::iter::repeat_n(Some("a"), n).chain(std::iter::repeat_n(Some("b"), bx.len())).collect(),
            );
            let ctx = TransformContext::default();
            let out = apply_with_context(&grouped, &batch, &ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            let alone = apply(&default_spec(10, None), &xy_batch(gx.clone(), gy.clone())).unwrap();
            let ys = col(&out, "y");
            let groups = str_col(&out, "g");
            let a: Vec<f64> = ys.iter().zip(&groups).filter(|(_, g)| *g == "a").map(|(y, _)| *y).collect();
            assert_eq!(a, col(&alone, "y"), "{label}: the good group fits as alone");
            assert!(ys.iter().zip(&groups).filter(|(_, g)| *g == "b").all(|(y, _)| y.is_nan()), "{label}: no fit for b");
            let said: Vec<String> = ctx.warnings.borrow().iter().map(|w| w.to_string()).collect();
            assert_eq!(
                said,
                [format!("Logistic: could not fit group g=b ({reason}); it degraded to an empty/flat output instead of failing the whole chart")],
                "{label}"
            );
            assert!(apply(&default_spec(10, None), &xy_batch(bx, by)).is_err(), "{label}: ungrouped still refuses");
        }
    }

    /// Every way one group's fit can fail degrades that group alone: no fit
    /// for it, a warning naming it and the reason, and the other group fits
    /// as it does alone. The same rows ungrouped refuse with the reason's
    /// own message.
    #[test]
    fn every_per_group_fit_failure_degrades_the_group_and_refuses_ungrouped() {
        pyo3::Python::initialize();
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, LogisticSpec, (Vec<f64>, Vec<f64>), (Vec<f64>, Vec<f64>), &str)> = vec![
            (
                "perfect separation",
                default_spec(10, None),
                (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
                (vec![-100.0, -90.0, -80.0, 80.0, 90.0, 100.0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                "perfect separation detected (model classifies every observation with probability > 0.999)",
            ),
            (
                "singular weighted design",
                default_spec(10, None),
                (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
                (vec![0.0, 1e-9, 2e-9], vec![0.0, 1.0, 0.0]),
                "singular design (X'WX has zero determinant)",
            ),
        ];
        for (label, spec, (gx, gy), (bx, by), reason) in cases {
            let mut grouped = spec.clone();
            grouped.groupby = vec!["g".into()];
            let n = gx.len();
            let batch = grouped_batch(
                gx.iter().chain(&bx).copied().collect(),
                gy.iter().chain(&by).copied().collect(),
                std::iter::repeat_n(Some("a"), n).chain(std::iter::repeat_n(Some("b"), bx.len())).collect(),
            );
            let ctx = TransformContext::default();
            let out = apply_with_context(&grouped, &batch, &ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            let alone = apply(&spec, &xy_batch(gx, gy)).unwrap();
            let ys = col(&out, "y");
            let groups = str_col(&out, "g");
            let a: Vec<f64> = ys.iter().zip(&groups).filter(|(_, g)| *g == "a").map(|(y, _)| *y).collect();
            assert_eq!(a, col(&alone, "y"), "{label}: the good group fits as alone");
            assert!(ys.iter().zip(&groups).filter(|(_, g)| *g == "b").all(|(y, _)| y.is_nan()), "{label}: no fit for b");
            let said: Vec<String> = ctx.warnings.borrow().iter().map(|w| w.to_string()).collect();
            assert_eq!(
                said,
                [format!("Logistic: could not fit group g=b ({reason}); it degraded to an empty/flat output instead of failing the whole chart")],
                "{label}"
            );
            let refused = apply(&spec, &xy_batch(bx, by)).expect_err(label).to_string();
            assert_eq!(refused, format!("ValueError: Logistic: {reason}"), "{label}: ungrouped refuses");
        }
    }

    /// A genuinely reachable cancellation shape, not a hand-built one: at
    /// `x` centered on a large constant (`c = 37e6`) with a small spread
    /// (`±0.5`), `Var(η) = [1, x] Σ [1, x]'` is positive in exact
    /// arithmetic (its minimum over `x` is `1/Σw`), but floating-point
    /// cancellation at `x`'s magnitude makes it compute negative at some
    /// grid points. The fit line still converges fine; only the band is
    /// undefined where it cancels.
    #[test]
    fn large_offset_x_reaches_the_wald_variance_cancellation_branch() {
        let c = 37_000_000.0;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..3 {
            xs.extend([c - 0.5, c - 0.5, c + 0.5, c + 0.5]);
            ys.extend([0.0, 1.0, 0.0, 1.0]);
        }
        let batch = xy_batch(xs, ys);
        let spec = default_spec(100, Some(0.95));
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert!(col(&out, "y").iter().all(|v| v.is_finite()), "the fit line must draw");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        match &warnings[0] {
            TransformWarning::IntervalUndefined { transform, group, ci, reason, .. } => {
                assert_eq!(transform, "Logistic");
                assert!(group.is_none());
                assert_eq!(*ci, 0.95);
                assert!(reason.contains("of 100 grid points"), "{reason}");
            }
            other => panic!("expected IntervalUndefined, got {other:?}"),
        }
    }

    /// The same Wald-variance-cancellation shape as the ungrouped test
    /// above, reached once per `groupby` partition: each group's own
    /// warning names that group rather than leaving `group` unset, so a
    /// grouped chart's facet or legend can tell which group's band is
    /// undefined.
    #[test]
    fn large_offset_x_grouped_reaches_the_wald_variance_cancellation_branch_naming_each_group() {
        let c = 37_000_000.0;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut gs: Vec<Option<&str>> = Vec::new();
        for group in ["a", "b"] {
            for _ in 0..3 {
                xs.extend([c - 0.5, c - 0.5, c + 0.5, c + 0.5]);
                ys.extend([0.0, 1.0, 0.0, 1.0]);
                gs.extend([Some(group); 4]);
            }
        }
        let batch = grouped_batch(xs, ys, gs);
        let mut spec = default_spec(100, Some(0.95));
        spec.groupby = vec!["g".into()];
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert!(col(&out, "y").iter().all(|v| v.is_finite()), "both groups' lines must draw");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 2, "one warning per group, got {warnings:?}");
        let mut seen_groups: Vec<String> = Vec::new();
        for w in warnings.iter() {
            match w {
                TransformWarning::IntervalUndefined { transform, group, ci, reason, .. } => {
                    assert_eq!(transform, "Logistic");
                    let label = group.clone().expect("a grouped band warning must name its group");
                    seen_groups.push(label);
                    assert_eq!(*ci, 0.95);
                    assert!(reason.contains("of 100 grid points"), "{reason}");
                }
                other => panic!("expected IntervalUndefined, got {other:?}"),
            }
        }
        seen_groups.sort();
        assert_eq!(seen_groups, vec!["g=a".to_string(), "g=b".to_string()]);
    }
}
