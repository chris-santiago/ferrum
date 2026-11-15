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
use crate::transform::core::{SpaceSlot, TransformSpec};
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, parse_groupby_pyany, pooled_group_key,
    stack_grouped_batches, FitRows, KeyValue,
    FIT_NULL_KEYS,
};
use crate::transform::log_space::run_fit_in_log_space;
use crate::transform::numeric_util::{coerce_xy_pairs_at, FEWER_THAN_TWO_USABLE_XY_POINTS};

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

/// When a drawn log axis demands this fit's `x`, the fit runs over `x`
/// mapped into that axis's own log units ([`run_fit_in_log_space`]); the
/// response `y` is a probability and never joins this demand (a `Logistic`
/// under a log y keeps fitting in data space — only `x` has a drawable
/// positional space here). Otherwise dispatches to [`apply_in_data_space`]
/// unchanged.
pub(crate) fn apply_with_context(
    spec: &LogisticSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let Some(slot) = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::X)) else {
        return apply_in_data_space(spec, batch, ctx);
    };
    run_fit_in_log_space(
        ctx,
        "Logistic",
        batch,
        &TransformSpec::fit_x_only_space_columns(&spec.x),
        (&spec.x, Some(slot)),
        (&spec.y, None),
        |mapped_batch, x, _y, ctx| apply_in_data_space(&LogisticSpec { x: x.to_string(), ..spec.clone() }, mapped_batch, ctx),
    )
}

/// Fit the whole batch, or, under `spec.groupby`, each group on its own rows
/// and stack the per-group curves with the group columns appended, as
/// `robust::apply_grouped` does. `SkipWithWholeBatchFallback` is the
/// partition's policy: a batch whose every key is null is fit whole, and the
/// partition's warning says so.
fn apply_in_data_space(
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
        FIT_NULL_KEYS,
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

/// Fit `batch`'s rows, or only one group's (`group`). Data the fit cannot
/// use (too few usable points, no x variance, one class, separation, a
/// singular design) degrades the call, panel or group to its empty fit with a
/// warning ([`Unfittable::degenerate`]). A `y` that is not 0 or 1 is a
/// response the method cannot read ([`Unfittable::malformed`] semantics): it
/// refuses the genuine whole frame and degrades a group or a panel
/// ([`TransformContext::degrade_group_or_refuse`]).
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
            refusal: Some(format!("Logistic: y must be binary (only 0 or 1); found value {y}")),
        });
    }

    if xs.len() < 2 {
        return unfittable(Unfittable::degenerate(FEWER_THAN_TWO_USABLE_XY_POINTS));
    }

    // Singular design: zero variance in x => X'WX is singular.
    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if !(x_max > x_min) {
        return unfittable(Unfittable::degenerate("zero x variance"));
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
                    step: ctx.step.clone(),
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
            .ok_or_else(|| Unfittable::degenerate("singular design (X'WX has zero determinant)"))?;

        let b0 = inv[0][0] * wxtz[0] + inv[0][1] * wxtz[1];
        let b1 = inv[1][0] * wxtz[0] + inv[1][1] * wxtz[1];
        beta = [b0, b1];
        last_cov = Some(inv);

        // Perfect-separation guard: blow-up in coefficients.
        if beta[0].abs() > 1e3 || beta[1].abs() > 1e3 {
            return Err(Unfittable::degenerate("perfect separation detected (coefficients diverged)"));
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
                return Err(Unfittable::degenerate(
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
        return Err(Unfittable::degenerate(
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
/// Data the fit cannot use (fewer than 2 usable points, no ``x`` variance,
/// one class, perfect separation) warns and draws an empty fit, naming the
/// facet panel, group or call, as ``Smooth`` and ``Robust`` do. A ``y`` that
/// is not 0 or 1 is refused when the call is the whole frame, and degrades a
/// facet panel or a ``groupby`` group.
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
/// >>> import polars as pl
/// >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "y": [0, 0, 1, 0, 1, 0, 1, 1]})
/// >>> chart = fm.lmplot(df, x="x", y="y", method="logistic")
/// >>> svg = chart.to_svg()
///
/// The same fit as a transform, drawn as a line:
///
/// >>> chart = fm.Chart(df).transform(fm.Logistic("x", "y")).mark_line().encode(x="x", y="y")
/// >>> svg = chart.to_svg()
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
        let groupby = parse_groupby_pyany("Logistic", groupby)?;
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

    // ── A drawn log axis demands this fit in its own space ───────────────

    use crate::transform::color_bin_group::{SlotSpace, StepSpace};
    type LogSpace = crate::transform::color_bin_group::LogAxisSpace;

    fn ctx_demanding(space: StepSpace) -> TransformContext {
        TransformContext { log_space: Some(std::rc::Rc::new(space)), ..TransformContext::default() }
    }

    /// The `x` slot demanded along `x` in `x_space` — `Logistic` and `Glm`
    /// carry only `x` ([`TransformSpec::fit_x_only_space_columns`]), so
    /// unlike `smooth::tests`'/`robust::tests`'/`kde_2d::tests`' identical
    /// helper, there is no `y_space` arm to mirror.
    fn x_slot(x_space: LogSpace) -> StepSpace {
        StepSpace::for_slots(std::iter::once((SpaceSlot::X, SlotSpace { space: x_space, axes: vec![crate::spec::layer::PositionalAxis::X], records_caller_input: true })))
    }

    fn positive_space() -> LogSpace {
        LogSpace { base: 10.0, side: crate::scale::log::LogSide::Positive }
    }

    /// A call (here a facet panel) whose every `x` is off a log x axis's side
    /// is cut to zero rows by the log seam: the fit's degenerate batch comes
    /// back, its warning goes to the scratch sink, the panel is recorded as
    /// one the axis emptied, and nothing refuses, so the axis's own report is
    /// the only one.
    #[test]
    fn logistic_panel_wholly_off_a_log_axis_is_cut_and_raises_nothing() {
        pyo3::Python::initialize();
        let batch = xy_batch(vec![-3.0, -2.0, -1.0, -3.0], vec![0.0, 1.0, 0.0, 1.0]);
        let ctx = TransformContext {
            is_partition: true,
            log_space: Some(std::rc::Rc::new(x_slot(positive_space()))),
            ..TransformContext::default()
        };
        let out = apply_with_context(&default_spec(10, None), &batch, &ctx).expect("Logistic: a panel the axis emptied must not refuse");
        assert_eq!(out.num_rows(), 0);
        assert!(ctx.warnings.borrow().is_empty(), "the fit's own warning goes to the scratch sink");
        let recorded = ctx.all_null_partitions.borrow();
        assert_eq!(recorded.len(), 1);
        assert!(matches!(recorded[0].outcome, crate::transform::context::AllNullOutcome::OffSide));
    }

    /// A zero-row whole frame has nothing to fit and degrades, naming the call.
    #[test]
    fn logistic_zero_row_whole_frame_degrades() {
        pyo3::Python::initialize();
        let ctx = TransformContext::default();
        let out = apply_with_context(&default_spec(10, None), &xy_batch(vec![], vec![]), &ctx).expect("Logistic: a zero-row frame degrades");
        assert_degrades(
            &ctx,
            &out,
            whole_call_sentence(FEWER_THAN_TWO_USABLE_XY_POINTS),
            "zero rows",
        );
    }

    /// Demanded on a positive-side log x, `Logistic` fits over `x` mapped
    /// into log10 units, then maps its `x` grid back
    /// ([`run_fit_in_log_space`]) — this must equal, point for point,
    /// running the SAME fit directly (no demand) on a batch whose `x` is
    /// already substituted with `log10(x)`: the grid mapped back through
    /// `10^v`, and the fitted probability `y` (which never joins the x
    /// demand — the response sits on the link's own scale) identical
    /// outright, since both routes fit the identical log10(x) values.
    #[test]
    fn logistic_on_a_demanded_x_equals_its_own_fit_on_log10_substituted_data() {
        pyo3::Python::initialize();
        let xs_raw: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
        // Two mislabels (index 3 flipped up, index 4 flipped down) break
        // perfect separation near the log10(x)=1 boundary, as
        // `test_logistic_well_separated_data` does for its own data.
        let ys: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let xs_log: Vec<f64> = xs_raw.iter().map(|x| x.log10()).collect();
        let batch_raw = xy_batch(xs_raw, ys.clone());
        let batch_log = xy_batch(xs_log, ys);
        let spec = default_spec(8, None);
        let ctx = ctx_demanding(x_slot(positive_space()));
        let demanded = apply_with_context(&spec, &batch_raw, &ctx).expect("demanded fit on positive x");
        let oracle = apply_with_context(&spec, &batch_log, &TransformContext::default()).expect("oracle fit on log10(x)-substituted data");
        let demanded_x = col(&demanded, "x");
        let oracle_x = col(&oracle, "x");
        assert_eq!(demanded_x.len(), oracle_x.len());
        for (i, (&dx, &ox)) in demanded_x.iter().zip(oracle_x.iter()).enumerate() {
            assert!((dx - 10f64.powf(ox)).abs() < 1e-9, "row {i}: demanded grid x={dx} must equal 10^(oracle log10 grid)={}", 10f64.powf(ox));
        }
        let demanded_y = col(&demanded, "y");
        let oracle_y = col(&oracle, "y");
        for (i, (&dy, &oy)) in demanded_y.iter().zip(oracle_y.iter()).enumerate() {
            assert!((dy - oy).abs() < 1e-9, "row {i}: the fitted probability never joins the x demand, so it must match the oracle exactly: {dy} vs {oy}");
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


    /// The one warning a call that cannot be fit raises, and its all-NaN fit.
    fn assert_degrades(ctx: &TransformContext, out: &RecordBatch, expected: String, label: &str) {
        assert!(col(out, "y").iter().all(|y| y.is_nan()), "{label}: no fit drawn");
        let said: Vec<String> = ctx.warnings.borrow().iter().map(|w| w.to_string()).collect();
        assert_eq!(said, [expected], "{label}");
    }

    /// The whole-call sentence of a `Logistic` fit that cannot be computed.
    fn whole_call_sentence(reason: &str) -> String {
        format!("Logistic: could not fit the data ({reason}); it degraded to an empty/flat output instead of failing the whole chart")
    }

    /// Data the fit cannot use degrades the whole frame, a facet panel
    /// (`is_partition`, no group) and the call alike, never refusing.
    fn assert_whole_frame_and_panel_degrade(spec: &LogisticSpec, x: Vec<f64>, y: Vec<f64>, reason: &str, label: &str) {
        let whole = TransformContext::default();
        let out = apply_with_context(spec, &xy_batch(x.clone(), y.clone()), &whole).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_degrades(&whole, &out, whole_call_sentence(reason), label);
        let panel = TransformContext { is_partition: true, ..TransformContext::default() };
        let out = apply_with_context(spec, &xy_batch(x, y), &panel).unwrap_or_else(|e| panic!("{label} (panel): {e}"));
        assert_degrades(&panel, &out, whole_call_sentence(reason), label);
    }

    #[test]
    fn test_logistic_perfect_separation_degrades() {
        pyo3::Python::initialize();
        // All y=0 for x<<0, all y=1 for x>>0 with a huge gap → perfect separation.
        let xs: Vec<f64> = vec![-100.0, -90.0, -80.0, 80.0, 90.0, 100.0];
        let ys: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_whole_frame_and_panel_degrade(
            &default_spec(10, None),
            xs,
            ys,
            "perfect separation detected (model classifies every observation with probability > 0.999)",
            "perfect separation",
        );
    }

    /// Coefficients that blow up on the first weighted step (the weights are
    /// near-flat over a tiny `x` spread) are separation the fit cannot use:
    /// the whole call and a panel degrade with that reason.
    #[test]
    fn test_logistic_diverging_coefficients_degrade() {
        pyo3::Python::initialize();
        assert_whole_frame_and_panel_degrade(
            &default_spec(10, None),
            vec![0.0, 1e-4, 2e-4, 3e-4],
            vec![0.0, 0.0, 1.0, 1.0],
            "perfect separation detected (coefficients diverged)",
            "diverged coefficients",
        );
        assert_whole_frame_and_panel_degrade(
            &default_spec(10, None),
            vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0],
            vec![0.0, 0.0, 1.0, 1.0],
            "perfect separation detected (coefficients diverged)",
            "diverged coefficients at a large offset",
        );
    }

    /// One class over an `x` whose spread is tiny next to its magnitude
    /// saturates every fitted probability within the first five steps (the
    /// weighted solve loses the intercept to cancellation): separation the
    /// fit cannot use, so the whole call and a panel degrade.
    #[test]
    fn test_logistic_saturated_probabilities_degrade() {
        pyo3::Python::initialize();
        let cases = [
            (vec![1_000_000.02, 1_000_000.0, 999_999.96], vec![1.0; 3]),
            (vec![999_999.99, 1_000_000.0, 1_000_000.03], vec![1.0; 3]),
            (vec![1_000_000.03, 1_000_000.01, 999_999.99], vec![0.0; 3]),
        ];
        for (i, (xs, ys)) in cases.into_iter().enumerate() {
            assert_whole_frame_and_panel_degrade(
                &default_spec(10, None),
                xs,
                ys,
                "perfect separation detected (fitted probabilities saturated)",
                &format!("saturated probabilities, case {i}"),
            );
        }
    }

    /// A `y` that is not 0 or 1 refuses the genuine whole frame whatever its
    /// count of usable rows, and degrades a facet panel.
    #[test]
    fn test_logistic_response_outside_the_method_refuses_the_whole_frame_at_any_row_count() {
        pyo3::Python::initialize();
        for (label, xs, ys) in [
            ("one row", vec![1.0], vec![2.0]),
            ("two rows", vec![1.0, 2.0], vec![0.0, 2.0]),
            ("many rows", vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 2.0, 0.0, 1.0]),
        ] {
            let refused = apply(&default_spec(10, None), &xy_batch(xs.clone(), ys.clone())).expect_err(label).to_string();
            assert_eq!(refused, "ValueError: Logistic: y must be binary (only 0 or 1); found value 2", "{label}");
            let panel = TransformContext { is_partition: true, ..TransformContext::default() };
            let out = apply_with_context(&default_spec(10, None), &xy_batch(xs, ys), &panel).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_degrades(&panel, &out, whole_call_sentence("y is not binary (found 2)"), label);
        }
    }

    #[test]
    fn test_logistic_singular_design_degrades() {
        pyo3::Python::initialize();
        // All x identical → zero variance → singular X'WX.
        let xs = vec![3.0; 10];
        let ys: Vec<f64> = (0..10).map(|i| (i % 2) as f64).collect();
        assert_whole_frame_and_panel_degrade(&default_spec(10, None), xs, ys, "zero x variance", "constant x");
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
        assert!(!ungrouped.contains("groupby"), "an ungrouped spec omits groupby: {ungrouped}");
    }

    /// A group that cannot be fit draws no fit (its curve is all NaN) and a
    /// warning names it and the reason, while the other group fits as it
    /// does alone, as `Smooth`'s and `Robust`'s groups degrade. The same rows
    /// ungrouped degrade the call too, except a `y` that is not binary, which
    /// refuses the whole frame and degrades a facet panel.
    #[test]
    fn a_group_that_cannot_be_fit_degrades_with_a_warning_and_the_others_draw() {
        pyo3::Python::initialize();
        let (gx, gy): (Vec<f64>, Vec<f64>) = (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let n = gx.len();
        let mut grouped = default_spec(10, None);
        grouped.groupby = vec!["g".into()];
        let cases: Vec<(&str, Vec<f64>, Vec<f64>, &str)> = vec![
            ("one point", vec![1.0], vec![gy[0]], FEWER_THAN_TWO_USABLE_XY_POINTS),
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
            if reason.starts_with("y is not binary") {
                let refused = apply(&default_spec(10, None), &xy_batch(bx.clone(), by.clone())).expect_err(label).to_string();
                assert_eq!(refused, "ValueError: Logistic: y must be binary (only 0 or 1); found value 2", "{label}: whole frame refuses");
                let panel = TransformContext { is_partition: true, ..TransformContext::default() };
                let out = apply_with_context(&default_spec(10, None), &xy_batch(bx, by), &panel).unwrap_or_else(|e| panic!("{label} (panel): {e}"));
                assert_degrades(&panel, &out, whole_call_sentence(reason), label);
            } else {
                assert_whole_frame_and_panel_degrade(&default_spec(10, None), bx, by, reason, label);
            }
        }
    }

    /// Every way one group's fit can fail degrades that group alone: no fit
    /// for it, a warning naming it and the reason, and the other group fits
    /// as it does alone. The same rows ungrouped degrade the call as well.
    #[test]
    fn every_per_group_fit_failure_degrades_the_group_and_the_ungrouped_call() {
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
            assert_whole_frame_and_panel_degrade(&spec, bx, by, reason, label);
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

    /// Two frames of one fit: the first eight rows are the points, then a row
    /// whose `x` is `inf_x` and a row whose `y` is `inf_y`. `None` is the null
    /// twin and an infinity is the value the twin stands in for.
    fn frame_with_unusable(inf_x: Option<f64>, inf_y: Option<f64>, ys: [f64; 8]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        let mut xs: Vec<Option<f64>> = (1..=8).map(|i| Some(i as f64)).collect();
        let mut ys: Vec<Option<f64>> = ys.iter().map(|&y| Some(y)).collect();
        xs.extend([inf_x, Some(9.0)]);
        ys.extend([Some(1.0), inf_y]);
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(xs)), Arc::new(Float64Array::from(ys))],
        )
        .unwrap()
    }

    /// A point with an infinite `x` or `y` is skipped as a null point is: the
    /// fit is the null twin's.
    #[test]
    fn an_infinite_coordinate_is_skipped_as_a_null_one_is() {
        pyo3::Python::initialize();
        let ys = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let fit = |inf_x, inf_y| apply(&default_spec(20, None), &frame_with_unusable(inf_x, inf_y, ys)).unwrap();
        let twin = fit(None, None);
        for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(fit(Some(infinite), Some(infinite)), twin);
        }
        assert!(col(&twin, "y").iter().all(|v| v.is_finite()));
    }
}
