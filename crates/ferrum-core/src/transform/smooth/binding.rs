//! The PyO3 binding: `PySmooth`, its `#[pymethods] impl`, and the `METHODS`
//! vocabulary constant they both read. Wave 6A `#163`: split out of
//! `transform/smooth.rs` as a self-contained unit (its own local `use
//! pyo3::prelude::*`) rather than left inline in `mod.rs`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::transform::core::TransformSpec;
use crate::transform::gam;
use crate::transform::group_key::parse_groupby_pyany;

use super::{SmoothMethod, SmoothOutput, SmoothSpec, DEFAULT_LOESS_BANDWIDTH};

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
///     Predictor column (any integer, unsigned, or float dtype; widened to
///     Float64). A non-numeric column raises, naming the transform, column,
///     and dtype.
/// y : str
///     Response column (any integer, unsigned, or float dtype; widened to
///     Float64). A non-numeric column raises, naming the transform, column,
///     and dtype.
/// method : {"loess", "lm", "linear", "quadratic", "cubic", "log", "sqrt", \
///           "glm", "gam"}, default "loess"
///     Smoothing method. ``"loess"`` fits locally-weighted polynomial
///     regression; ``"lm"``/``"linear"`` fits a global linear model
///     (promoted to a polynomial fit when ``degree >= 2`` is also given);
///     ``"quadratic"``/``"cubic"`` fit fixed degree-2/3 polynomials;
///     ``"log"``/``"sqrt"`` fit a linear model on a transformed x;
///     ``"glm"`` fits a Generalized Linear Model (Gaussian family, Identity
///     link — the standalone ``Glm`` transform's own default) via IRLS;
///     ``"gam"`` fits a Generalized Additive Model — a penalized cubic
///     B-spline (P-spline) whose smoothing parameter is chosen by
///     generalized cross-validation. Unlike ``"loess"``, ``"gam"`` uses no
///     randomness: its confidence band is the spline's Bayesian posterior
///     band, so ``seed`` has no effect on it. ``"gam"`` holds its boundary
///     value rather than extrapolating outside the observed x range.
/// ci : float or None, default 0.95
///     Confidence level in (0, 1) for the confidence band. Set to
///     ``None`` to suppress the band.
/// bandwidth : float, optional
///     LOESS span in (0, 1]; fraction of data used in each local fit.
///     Defaults to ``0.75`` when omitted. Passing it with any other method
///     raises, since no other fit path reads it — the one tolerance is an
///     explicit ``bandwidth=0.75`` (the default value itself), accepted as a
///     harmless no-op exactly as ``method="lm", degree=1`` is. To make a
///     ``"gam"`` fit follow the data more closely, raise ``k``; for
///     ``"lm"``/``"linear"``, raise ``degree``.
/// degree : int, optional
///     For ``method="lm"``/``"linear"``: promotes the fit to a polynomial
///     of this degree (``Poly{degree}``) when ``>= 2``; ``1`` (or omitting
///     ``degree``) keeps the plain linear fit; ``0`` raises. A degree that
///     leaves fewer usable data points than coefficients to solve for (a
///     rank-deficient design) raises rather than silently rendering nothing.
///     For ``method="loess"``: the local-fit degree, ``1`` or ``2`` (default
///     when omitted is ``2``). For ``method="quadratic"``/``"cubic"``:
///     optional and must match the method's fixed degree (``2``/``3``) when
///     given — a conflicting value (e.g. ``method="quadratic", degree=5``)
///     raises rather than silently fitting degree 2 anyway. Passing
///     ``degree`` with ``method="log"``/``"sqrt"``/``"glm"``/``"gam"``
///     raises, since those methods have no degree concept to apply it to
///     (``"gam"``'s spline degree is fixed at 3; its smoothness comes from
///     the GCV-chosen penalty and from ``k``, not from a degree).
/// k : int, optional
///     For ``method="gam"``: the spline basis dimension (mgcv's ``k``),
///     default ``10``. This is gam's smoothness knob — the penalty chooses
///     how much of the basis to spend, so a larger ``k`` widens what the fit
///     *can* follow rather than forcing it to wiggle. Raise it when a
///     high-frequency signal is visibly under-smoothed at the default.
///     Clamped down to the number of distinct x values (a basis cannot carry
///     more segments than the data can place), so an over-large ``k`` is
///     harmless rather than rank-deficient. Values below ``4`` raise (no
///     smaller cubic basis exists), as does passing ``k`` with any method
///     other than ``"gam"``.
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
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_smooth(method="lm", ci=0.95).encode(x="x", y="y")
///
/// LOESS curve with custom bandwidth:
///
/// >>> fm.Chart(df).mark_smooth(method="loess", bandwidth=0.4).encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Smooth")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PySmooth(pub(crate) TransformSpec);

#[pymethods]
impl PySmooth {
    #[new]
    #[pyo3(signature = (x, y, *, method = "loess", ci = Some(0.95), bandwidth = None, degree = None, k = None, n = 200, seed = 0, x_bins = None, x_estimator = None, output = "fitted", inject_zero_ref = false, inject_metrics = false, groupby = None, name = None, x_range = None))]
    pub(super) fn new(
        x: &str, y: &str,
        method: &str,
        ci: Option<f64>,
        bandwidth: Option<f64>,
        degree: Option<u8>,
        k: Option<usize>,
        n: usize,
        seed: u64,
        x_bins: Option<usize>,
        x_estimator: Option<&str>,
        output: &str,
        inject_zero_ref: bool,
        inject_metrics: bool,
        groupby: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        x_range: Option<[f64; 2]>,
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
        // Batch-D T9 round 2 (Required Fix 2): `METHODS` is the single
        // vocabulary the registry, this pre-check, and the error message all
        // read — see `METHODS`'s doc comment. A string not in `METHODS`
        // refuses here, before the match below ever runs.
        if !METHODS.contains(&method) {
            return Err(PyValueError::new_err(format!(
                "Smooth: unknown method '{method}'; expected {}",
                METHODS.iter().map(|m| format!("'{m}'")).collect::<Vec<_>>().join(" | "),
            )));
        }
        // Batch-D T9 (spec §4.6): "lm"/"linear" promotes to a polynomial fit
        // when `degree >= 2` is given; `degree == 1` (or omitted) keeps the
        // plain linear fit — the no-op case `lmplot(order=1)` (and every
        // existing `mark_smooth(method="lm")` caller that never mentions
        // `degree`) relies on for byte-identical output. `degree == 0`
        // refuses rather than silently behaving like `degree == 1`.
        //
        // "quadratic"/"cubic" fit a FIXED degree (2/3). Batch-D T9 round 2
        // (Required Fix 1, rust-quality verdict finding 1): a `degree=` that
        // matches the fixed degree is accepted as a harmless no-op (the same
        // "explicit value that agrees with the default" tolerance `lm`'s
        // `degree=1` gets above); a CONFLICTING `degree=` used to be silently
        // swallowed — `method="quadratic", degree=5"` quietly fit a degree-2
        // curve with no diagnostic, the exact "silently the wrong fit" class
        // D9 exists to close. It now refuses, naming `method="lm", degree=d`
        // as the way to actually get that degree.
        let resolved_method = match method {
            "lm" | "linear" => match degree {
                None | Some(1) => SmoothMethod::Lm,
                Some(0) => return Err(PyValueError::new_err(
                    "Smooth: degree must be >= 1 for method='lm'/'linear'"
                )),
                Some(d) => SmoothMethod::Poly { degree: d },
            },
            "loess" => SmoothMethod::Loess,
            "quadratic" => match degree {
                None | Some(2) => SmoothMethod::Poly { degree: 2 },
                Some(d) => return Err(PyValueError::new_err(format!(
                    "Smooth: degree={d} conflicts with method='quadratic', which fits a \
                     fixed degree-2 polynomial; omit degree, pass degree=2, or use \
                     method='lm', degree={d} for a degree-{d} fit"
                ))),
            },
            "cubic" => match degree {
                None | Some(3) => SmoothMethod::Poly { degree: 3 },
                Some(d) => return Err(PyValueError::new_err(format!(
                    "Smooth: degree={d} conflicts with method='cubic', which fits a \
                     fixed degree-3 polynomial; omit degree, pass degree=3, or use \
                     method='lm', degree={d} for a degree-{d} fit"
                ))),
            },
            "log" => SmoothMethod::LogX,
            "sqrt" => SmoothMethod::SqrtX,
            "glm" => SmoothMethod::Glm,
            "gam" => SmoothMethod::Gam,
            // Batch-D T9 round 3 (cycle-2 finding B): the `METHODS` pre-check
            // above guarantees `accepted ⊆ METHODS`, not `METHODS ⊆ match
            // arms` — nothing compile-time-enforces the reverse, so an entry
            // added to `METHODS` without a matching arm here used to reach
            // an `unreachable!()` and surface as a `PanicException` at the
            // PyO3 boundary instead of a `ValueError`. Return the same typed
            // error the pre-check builds; a one-line edit to `METHODS` (the
            // const its own doc comment invites people to edit) should never
            // be able to panic a caller.
            _ => return Err(PyValueError::new_err(format!(
                "Smooth: internal error — '{method}' is declared in `METHODS` but has no \
                 parser arm in `PySmooth::new`; this is a bug in ferrum, not a user error. \
                 expected {}",
                METHODS.iter().map(|m| format!("'{m}'")).collect::<Vec<_>>().join(" | "),
            ))),
        };
        // Closing D9: `degree` on a method with no degree concept ("log",
        // "sqrt", "glm") used to be silently ignored; it now refuses so a
        // misunderstood `degree=` can't render the wrong fit quietly.
        // "lm"/"linear" (promotable), "loess", "quadratic", and "cubic" all
        // accept it. Batch-D T10: "gam" joins the refusing side from birth —
        // its spline degree is fixed at 3 by the method's contract and its
        // smoothness is set by the GCV-chosen penalty, not by a degree, so
        // there is nothing for a caller's `degree=` to mean.
        if degree.is_some()
            && matches!(
                resolved_method,
                SmoothMethod::LogX | SmoothMethod::SqrtX | SmoothMethod::Glm | SmoothMethod::Gam
            )
        {
            return Err(PyValueError::new_err(format!(
                "Smooth: degree is not valid for method='{method}'; degree only \
                 applies to 'lm'/'linear' (polynomial promotion), 'loess', \
                 'quadratic', or 'cubic'"
            )));
        }
        // R1 design finding 6: `bandwidth` used to be a plain `f64` with an
        // inline `0.75` default, so an explicit value on a non-loess method
        // was indistinguishable from the default and every fit path except
        // LOESS silently ignored it — a user watching `method="gam"`
        // under-smooth had one knob that refused (`degree`) and one that
        // no-opped. `Option<f64>` makes explicit-vs-default distinguishable
        // at this boundary (the `degree: Option<u8>` precedent above), and an
        // explicit non-loess `bandwidth` now refuses in exactly the shape
        // `degree` does.
        //
        // The one tolerance mirrors `degree`'s own: an explicit value that
        // AGREES with the default is a harmless no-op, the same way
        // `method="lm", degree=1` and `method="quadratic", degree=2` are
        // accepted above. `desugar_smooth` (`marks/statistical.py`) no longer
        // needs it -- its own `bandwidth` default moved to `None` and its
        // forward is conditional on `bandwidth is not None`, so it never
        // sends an explicit `0.75` in the first place. The tolerance now
        // matters only to a caller constructing `Smooth(...)` directly with
        // an explicit `bandwidth=0.75` on a non-loess method; `mark_smooth
        // (method="lm", bandwidth=0.4)` still raises either way.
        if let Some(b) = bandwidth {
            if !matches!(resolved_method, SmoothMethod::Loess) && b != DEFAULT_LOESS_BANDWIDTH {
                return Err(PyValueError::new_err(format!(
                    "Smooth: bandwidth={b} is not valid for method='{method}'; bandwidth is \
                     the LOESS span and only applies to method='loess'. For a smoother \
                     '{method}' fit, use the knob that method actually has (method='gam' \
                     takes k=, the spline basis dimension; 'lm'/'linear' take degree=)"
                )));
            }
        }
        let bandwidth = bandwidth.unwrap_or(DEFAULT_LOESS_BANDWIDTH);
        // `k` is gam's basis dimension and gam's alone, refused on every other
        // method under the same asymmetry rule `degree` and `bandwidth` follow.
        // No agrees-with-the-default tolerance here: nothing forwards `k`
        // unconditionally, so `Some(_)` on a non-gam method is always a
        // misunderstanding worth naming.
        if let Some(kv) = k {
            if !matches!(resolved_method, SmoothMethod::Gam) {
                return Err(PyValueError::new_err(format!(
                    "Smooth: k={kv} is not valid for method='{method}'; k is the spline \
                     basis dimension and only applies to method='gam'"
                )));
            }
            if kv < gam::MIN_BASIS_DIM {
                return Err(PyValueError::new_err(format!(
                    "Smooth: k={kv} is below the smallest cubic spline basis ({}); a basis \
                     this small cannot span even one cubic segment",
                    gam::MIN_BASIS_DIM,
                )));
            }
        }
        let k = k.unwrap_or(gam::DEFAULT_BASIS_DIM);
        if matches!(resolved_method, SmoothMethod::Loess) {
            if !bandwidth.is_finite() || bandwidth <= 0.0 || bandwidth > 1.0 {
                return Err(PyValueError::new_err(
                    "Smooth: LOESS bandwidth must be a finite value in (0, 1]",
                ));
            }
            let loess_degree = degree.unwrap_or(2);
            if loess_degree != 1 && loess_degree != 2 {
                return Err(PyValueError::new_err("Smooth: LOESS degree must be 1 or 2"));
            }
        }
        // `SmoothSpec.degree` mirrors the resolved fit's own degree where
        // one exists (Poly/Loess); methods without a degree concept
        // (Lm/LogX/SqrtX/Glm) store 1 as a neutral placeholder that no fit
        // path reads.
        let degree_for_spec: u8 = match resolved_method {
            SmoothMethod::Poly { degree: d } => d,
            SmoothMethod::Loess => degree.unwrap_or(2),
            _ => 1,
        };
        let method = resolved_method;
        use crate::transform::aggregate::XEstimator;
        let x_estimator_parsed = x_estimator
            .map(|s| {
                XEstimator::parse(s).map(XEstimator::agg_fn).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Smooth: unknown x_estimator '{s}'; expected {}",
                        XEstimator::accepted_values()
                    ))
                })
            })
            .transpose()?;
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
        let groupby = parse_groupby_pyany(groupby)?;
        Ok(PySmooth(TransformSpec::Smooth(SmoothSpec {
            x: x.to_string(), y: y.to_string(),
            method, ci, bandwidth, degree: degree_for_spec, k, n, seed,
            x_bins,
            x_estimator: x_estimator_parsed,
            output: output_parsed,
            inject_zero_ref,
            inject_metrics,
            groupby,
            name,
            x_range,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Smooth(s) => format!(
                "Smooth(x='{}', y='{}', method={:?}, ci={:?}, bandwidth={}, degree={}, k={}, n={}, seed={}, x_bins={:?}, x_estimator={:?}, output={:?}, inject_zero_ref={}, inject_metrics={}, groupby={:?}, name={:?}, x_range={:?})",
                s.x, s.y, s.method, s.ci, s.bandwidth, s.degree, s.k, s.n, s.seed,
                // The canonical `AggFn` spelling, as `AggregateOp`'s `fn_=`.
                s.x_bins, s.x_estimator.map(<&'static str>::from), s.output,
                s.inject_zero_ref, s.inject_metrics, s.groupby, s.name, s.x_range,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// The single source of truth for `Smooth`'s free-string `method`
/// vocabulary. Batch-D T9 round 2 (Required Fix 2, rust-quality verdict):
/// consumed by THREE sites — `registry::entry_for_kind("smooth")`'s
/// `methods` list (`registry/mod.rs`, via `METHODS.to_vec()`), the
/// constructor's pre-check above, and the "unknown method" error string
/// above (built FROM this slice, not a hand-kept parallel literal). The
/// pre-check makes the guarantee structural rather than test-detected: a
/// match arm added to `PySmooth::new` for a new alias without adding it here
/// is unreachable dead code (the pre-check rejects the string as "unknown
/// method" before the match ever runs), so a new method genuinely cannot
/// "land" — become callable — without the registry seeing it in the same
/// commit. The former guard
/// (`smooth_methods_vocabulary_matches_parser_error_message_exactly`,
/// `registry/tests.rs`) only proved the error message matched the registry;
/// it could not have caught a bare match arm added without touching the
/// message at all.
///
/// The reverse direction (`METHODS ⊆ match arms`, i.e. an entry added here
/// without a corresponding arm in `PySmooth::new`'s match below) is NOT
/// compile-time-enforced — nothing in Rust's exhaustiveness checking ties a
/// `&str` slice to a `match` on a different `&str`. Batch-D T9 round 3: the
/// match's `_` arm returns the same typed "unknown method" error rather than
/// `unreachable!()`, so that direction fails safely (a `ValueError`) instead
/// of panicking at the PyO3 boundary; `test_all_method_aliases_accepted_by_pysmooth_new`
/// (iterating `METHODS` itself) still catches the gap at test time.
pub(crate) const METHODS: &[&str] =
    &["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt", "glm", "gam"];

