use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{continuous_common, resolve_continuous, scale_spec_to_py_dict};
use super::ticks::{minor_ticks_log, nice_ticks, Tick};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct LogScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    base: f64,
    clamp: bool,
}

impl LogScaleData {
    /// Number of decades below the reference endpoint used as a floor when
    /// [`sanitize_domain`](Self::sanitize_domain) replaces a zero (or
    /// non-finite) domain endpoint. Six decades keeps the floor comfortably
    /// below any realistic data value while staying far from `f64`
    /// underflow.
    const ZERO_FLOOR_DECADES: f64 = 6.0;

    /// Construct a [`LogScaleData`], sanitizing the domain so it can never
    /// store an endpoint that `ln()` cannot handle (see
    /// [`sanitize_domain`](Self::sanitize_domain)).
    fn new(domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool) -> Self {
        Self { domain: Self::sanitize_domain(domain), range, base, clamp }
    }

    /// Whether `x` can stand as a log-domain endpoint: nonzero and finite.
    fn is_loggable(x: f64) -> bool {
        x != 0.0 && x.is_finite()
    }

    /// Replace a domain endpoint that cannot be logarithmed (exactly zero,
    /// or non-finite) with a small floor value derived from the other,
    /// valid endpoint — the d3/vega-lite convention for a log domain that
    /// touches zero, rather than rejecting it outright.
    ///
    /// This matters for auto-inferred domains: e.g. a `nice`-extended
    /// histogram bin edge can legitimately land exactly on `0` even though
    /// every raw data value is strictly positive (GH #49). Without this,
    /// `ln(0) = -inf` propagates into the `as i64` exponent cast in
    /// [`ticks`](Self::ticks) and [`minor_ticks_log`](super::ticks::minor_ticks_log),
    /// which saturates to `i64::MIN`/`i64::MAX` and then panics on the
    /// subsequent `i64` subtraction (debug builds) or silently produces a
    /// bogus tick set (release builds).
    ///
    /// Well-formed domains — both endpoints finite, nonzero, same sign —
    /// pass through completely unchanged; this must never alter behavior
    /// for a valid log domain.
    fn sanitize_domain(domain: [f64; 2]) -> [f64; 2] {
        let [d0, d1] = domain;
        let d0_ok = Self::is_loggable(d0);
        let d1_ok = Self::is_loggable(d1);
        if d0_ok && d1_ok {
            return domain;
        }
        let reference = if d0_ok {
            d0
        } else if d1_ok {
            d1
        } else {
            // Neither endpoint carries a usable sign/magnitude signal (e.g.
            // both exactly 0, or both non-finite). Fall back to the same
            // [1, 10] sentinel `LogScale::new` uses when no domain is
            // supplied at all.
            return [1.0, 10.0];
        };
        let sign = reference.signum();
        let floor = sign * (reference.abs() / 10f64.powf(Self::ZERO_FLOOR_DECADES)).max(f64::MIN_POSITIVE);
        if d0_ok { [d0, floor] } else { [floor, d1] }
    }

    /// Snap a raw exponent to the nearest integer when within floating-point epsilon.
    /// Prevents `ln(1000)/ln(10) = 2.9999999999999996` from flooring to 2 instead of 3.
    fn snap_exp(raw: f64) -> f64 {
        let rounded = raw.round();
        if (raw - rounded).abs() < 1e-10 {
            rounded
        } else {
            raw
        }
    }

    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let neg = d0 < 0.0;
        let sign = if neg { -1.0 } else { 1.0 };
        if (x * sign) <= 0.0 && !self.clamp { return f64::NAN; }
        let log_base = self.base.ln();
        let lx = (x * sign).max(f64::MIN_POSITIVE).ln() / log_base;
        let ld0 = (d0 * sign).ln() / log_base;
        let ld1 = (d1 * sign).ln() / log_base;
        let t = (lx - ld0) / (ld1 - ld0);
        let mapped = r0 + t * (r1 - r0);
        if self.clamp {
            let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
            mapped.clamp(lo, hi)
        } else if (x * sign) < (d0 * sign).min(d1 * sign) || (x * sign) > (d0 * sign).max(d1 * sign) {
            f64::NAN
        } else {
            mapped
        }
    }

    fn invert(&self, y: f64) -> f64 {
        if y.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let neg = d0 < 0.0;
        let sign = if neg { -1.0 } else { 1.0 };
        let log_base = self.base.ln();
        let ld0 = (d0 * sign).ln() / log_base;
        let ld1 = (d1 * sign).ln() / log_base;
        let t = (y - r0) / (r1 - r0);
        let lmapped = ld0 + t * (ld1 - ld0);
        let mapped = sign * self.base.powf(lmapped);
        if self.clamp {
            let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
            mapped.clamp(lo, hi)
        } else if y < r0.min(r1) || y > r0.max(r1) {
            f64::NAN
        } else {
            mapped
        }
    }

    fn ticks(&self, count: usize) -> Vec<f64> {
        let neg = self.domain[0] < 0.0;
        let sign: f64 = if neg { -1.0 } else { 1.0 };
        let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
        let hi = (self.domain[0] * sign).max(self.domain[1] * sign);
        let log_base = self.base.ln();
        let lo_exp = Self::snap_exp(lo.ln() / log_base).floor() as i64;
        let hi_exp = Self::snap_exp(hi.ln() / log_base).ceil() as i64;
        let span_decades = (hi_exp - lo_exp).max(1) as usize;

        // Ticks are generated in ascending-exponent order.
        // For positive domains, ascending exponent = ascending value.
        // For negative domains, ascending exponent = descending value (more negative).
        // Reversal condition must account for the sign flip.
        let should_reverse = if neg {
            // Negative: ticks come out descending on the number line.
            // Reverse only if the domain is ascending (d[0] < d[1]).
            self.domain[0] < self.domain[1]
        } else {
            // Positive: ticks come out ascending on the number line.
            // Reverse only if the domain is descending (d[0] > d[1]).
            self.domain[0] > self.domain[1]
        };

        if span_decades >= count {
            let mut out: Vec<f64> = (lo_exp..=hi_exp)
                .map(|e| sign * self.base.powi(e as i32))
                .filter(|t| (t.abs() >= lo) && (t.abs() <= hi))
                .collect();
            if should_reverse { out.reverse(); }
            out
        } else {
            let lvals = nice_ticks(lo.ln() / log_base, hi.ln() / log_base, count);
            let mut out: Vec<f64> = lvals.into_iter().map(|lv| sign * self.base.powf(lv)).collect();
            if should_reverse { out.reverse(); }
            out
        }
    }

    fn nice(self) -> Self {
        let neg = self.domain[0] < 0.0;
        let sign: f64 = if neg { -1.0 } else { 1.0 };
        let log_base = self.base.ln();
        let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
        let hi = (self.domain[0] * sign).max(self.domain[1] * sign);
        let lo_exp = Self::snap_exp(lo.ln() / log_base).floor();
        let hi_exp = Self::snap_exp(hi.ln() / log_base).ceil();
        // For positive domains: new_lo < new_hi (smallest to largest absolute value)
        // For negative domains: new_lo = sign * base^lo_exp is closest to zero (e.g. -1),
        // and new_hi = sign * base^hi_exp is most negative (e.g. -1000). Swap them so
        // the magnitude ordering matches the number-line ordering before applying the
        // ascending/descending domain logic below.
        let (new_lo, new_hi) = if neg {
            (sign * self.base.powf(hi_exp), sign * self.base.powf(lo_exp))
        } else {
            (sign * self.base.powf(lo_exp), sign * self.base.powf(hi_exp))
        };
        let new_domain = if self.domain[0] <= self.domain[1] {
            [new_lo, new_hi]
        } else {
            [new_hi, new_lo]
        };
        Self { domain: new_domain, range: self.range, base: self.base, clamp: self.clamp }
    }
}

/// Continuous logarithmic scale.
///
/// Maps a numeric domain to a numeric range via a logarithmic transformation.
/// Useful for data spanning several orders of magnitude. Domain must not
/// contain zero and both endpoints must share the same sign.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``. Neither endpoint may be 0 and both
///     must have the same sign.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// base : float, default 10.0
///     Logarithm base. Must be finite, positive, and not equal to 1.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to the nearest power of ``base``.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         x=fr.X("value", scale=fr.LogScale(domain=[1, 10_000], range=[0, 400]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LogScale {
    data: LogScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    domain_user_set: bool,
}

impl LogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, base: f64, clamp: bool, nice: bool) -> Self {
        let mut d = LogScaleData::new([domain[0], domain[1]], [range[0], range[1]], base, clamp);
        if nice { d = d.nice(); }
        LogScale { data: d, padding: None, range_user_set: true, domain_user_set: true }
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 { self.data.scale(x) }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return minor ticks using the standard 2-9 intra-decade multiples.
    ///
    /// For base-10 log scales this produces the visually-standard minor ticks
    /// that crowd towards the top of each decade.  For other bases the minor
    /// tick list is empty (2-9 multiples are not meaningful for non-decimal
    /// logarithm bases).
    ///
    /// The major tick count is fixed at 10 (the conventional default) to
    /// determine which decade boundaries act as major ticks.  The minor tick
    /// *positions* are always the 2-9 intra-decade multiples, independent of
    /// any count parameter.
    // Wired to the render layer via `ScaleKind::minor_tick_fractions`
    // (`render/scale_resolve/mod.rs`, dispatched through `dispatch_continuous!`).
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.data.ticks(10);
        let [lo, hi] = self.data.domain;
        minor_ticks_log(lo, hi, self.data.base, &majors)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] { self.data.range }

    pub(crate) fn domain_pair(&self) -> [f64; 2] { self.data.domain }

    pub(crate) fn repr_string(&self) -> String {
        let LogScaleData { domain, range, base, clamp } = &self.data;
        let domain_s = if self.domain_user_set {
            format!("[{}, {}]", domain[0], domain[1])
        } else {
            "None".to_string()
        };
        let range_s = if self.range_user_set {
            format!("[{}, {}]", range[0], range[1])
        } else {
            "None".to_string()
        };
        format!(
            "LogScale(domain={}, range={}, base={}, clamp={})",
            domain_s, range_s, base, if *clamp { "True" } else { "False" }
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    /// `nice` is baked into the domain at construction, so it is always `false`
    /// here — matching what the legacy `_scale_to_dict` omitted.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Log {
            base: self.data.base,
            common: continuous_common(
                self.data.domain,
                self.domain_user_set,
                self.data.range,
                self.range_user_set,
                self.data.clamp,
                self.padding,
            ),
            nice: false,
        }
    }
}

#[pymethods]
impl LogScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, base = 10.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        base: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        if !base.is_finite() || base <= 0.0 || base == 1.0 {
            return Err(PyValueError::new_err(format!(
                "base must be finite, > 0, and != 1; got {base}"
            )));
        }
        // Sentinel [1.0, 10.0] when no domain supplied; render-time inference
        // replaces it before any scale computation occurs.
        let resolved = resolve_continuous(domain, range, [1.0, 10.0])?;
        if resolved.domain_user_set {
            let [d0, d1] = resolved.domain;
            if d0 == 0.0 || d1 == 0.0 {
                return Err(PyValueError::new_err(
                    "log scale domain must not contain 0",
                ));
            }
            if d0.signum() != d1.signum() {
                return Err(PyValueError::new_err(
                    "log scale domain endpoints must have the same sign",
                ));
            }
        }
        let mut d = LogScaleData::new(resolved.domain, resolved.range, base, clamp);
        if nice && resolved.domain_user_set {
            d = d.nice();
        }
        Ok(LogScale {
            data: d,
            padding,
            range_user_set: resolved.range_user_set,
            domain_user_set: resolved.domain_user_set,
        })
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.data.scale(x) }
    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.data.invert(y) }

    /// Return approximately ``count`` tick values spaced logarithmically within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest power of ``base``.
    fn nice(&self) -> Self {
        LogScale {
            data: self.data.clone().nice(),
            padding: self.padding,
            range_user_set: self.range_user_set,
            domain_user_set: self.domain_user_set,
        }
    }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.padding }

    /// Input domain as ``[min, max]``, or ``None`` when data-derived.
    #[getter]
    fn domain(&self) -> Option<Vec<f64>> {
        if self.domain_user_set { Some(self.data.domain.to_vec()) } else { None }
    }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.range_user_set { Some(self.data.range.to_vec()) } else { None }
    }

    /// Logarithm base (default 10.0).
    #[getter]
    fn base(&self) -> f64 { self.data.base }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.data.clamp }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool) -> LogScaleData {
        LogScaleData::new(domain, range, base, clamp)
    }

    #[test]
    fn log_scale_basic_decades() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!((s.scale(1.0) - 0.0).abs() < 1e-12);
        assert!((s.scale(10.0) - 1.0).abs() < 1e-12);
        assert!((s.scale(1000.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn log_inversion_round_trip() {
        let s = d([1.0, 1_000_000.0], [0.0, 6.0], 10.0, false);
        for x in [1.0, 10.0, 100.0, 12345.0, 999999.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back / x - 1.0).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn log_negative_domain_supported() {
        let s = d([-1000.0, -1.0], [0.0, 3.0], 10.0, false);
        let y = s.scale(-10.0);
        let back = s.invert(y);
        assert!((back / -10.0 - 1.0).abs() < 1e-9, "negative round-trip failed: got {back}");
    }

    #[test]
    fn log_ticks_one_per_decade() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        let t = s.ticks(4);
        assert!(t.len() >= 3, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn log_nice_rounds_to_decades() {
        let s = d([3.0, 700.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - 1.0).abs() < 1e-9);
        assert!((n.domain[1] - 1000.0).abs() < 1e-9);
    }

    /// Named-field conversion (T2.5): user-set domain/range/base round-trip,
    /// and an unset domain reports `None` while keeping the [1, 10] sentinel.
    #[test]
    fn log_named_fields_round_trip() {
        let with_domain = LogScale::new(
            Some(vec![1.0, 1000.0]), Some(vec![0.0, 300.0]), 10.0, false, false, Some(0.1),
        ).unwrap();
        assert_eq!(with_domain.domain(), Some(vec![1.0, 1000.0]));
        assert_eq!(with_domain.range(), Some(vec![0.0, 300.0]));
        assert_eq!(with_domain.base(), 10.0);
        assert_eq!(with_domain.padding(), Some(0.1));

        let no_domain = LogScale::new(None, None, 2.0, false, false, None).unwrap();
        assert_eq!(no_domain.domain(), None);
        assert_eq!(no_domain.range(), None);
        assert_eq!(no_domain.domain_pair(), [1.0, 10.0]);
        assert_eq!(no_domain.base(), 2.0);
    }

    // F17 — Underflow contract:
    //
    // For a positive domain, inputs ≤ 0 are out-of-domain. Without clamp,
    // the scale returns NaN; ScaleKind::to_pixel_f64 maps NaN to None so
    // mark renderers drop the row, matching Linear's out-of-domain
    // behavior. With clamp=true, out-of-domain inputs map to the range
    // endpoint (clamped) without producing NaN.
    //
    // This is intentional: callers either pass clamp=true (to render
    // off-axis points at the boundary) or drop the row. The earlier
    // F17 finding flagged a potential silent-NaN-leak path; these
    // tests are the contract pin.

    #[test]
    fn log_zero_input_returns_nan_unclamped() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(s.scale(0.0).is_nan(), "x=0 must be NaN on positive log domain");
    }

    #[test]
    fn log_negative_input_returns_nan_unclamped() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(s.scale(-1.0).is_nan(), "x=-1 must be NaN on positive log domain");
    }

    #[test]
    fn log_zero_input_clamps_to_range_when_clamp_enabled() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, true);
        let y = s.scale(0.0);
        assert!(y.is_finite(), "x=0 with clamp must be finite, got {y}");
        // 0 < domain[0], so the clamped result is the lower range bound.
        assert_eq!(y, 0.0);
    }

    // ── GH #49 — zero-touching domain regression ────────────────────────────
    //
    // Auto-inferred log domains can legitimately touch 0 (e.g. a
    // `nice`-extended histogram bin edge), even though the constructor
    // rejects a user-supplied domain containing 0. `sanitize_domain` clamps
    // the zero endpoint to a small floor rather than letting `ln(0) = -inf`
    // reach the `as i64` exponent cast in `ticks()`, which previously
    // saturated to `i64::MIN` and panicked on the next subtraction.

    #[test]
    fn log_ticks_finite_for_positive_domain_touching_zero() {
        let s = d([0.0, 1000.0], [0.0, 3.0], 10.0, false);
        let t = s.ticks(10);
        assert!(!t.is_empty(), "expected a non-empty tick set, got {t:?}");
        assert!(t.iter().all(|v| v.is_finite()), "all ticks must be finite: {t:?}");
        // The floored zero endpoint must not spuriously introduce a decade
        // above the real, finite endpoint.
        assert!(t.iter().all(|&v| v <= 1000.0 + 1e-6), "tick exceeds domain max: {t:?}");
    }

    #[test]
    fn log_ticks_finite_for_negative_domain_touching_zero() {
        let s = d([-1000.0, 0.0], [0.0, 3.0], 10.0, false);
        let t = s.ticks(10);
        assert!(!t.is_empty(), "expected a non-empty tick set, got {t:?}");
        assert!(t.iter().all(|v| v.is_finite()), "all ticks must be finite: {t:?}");
        assert!(t.iter().all(|&v| v >= -1000.0 - 1e-6), "tick exceeds domain min: {t:?}");
    }

    #[test]
    fn log_scale_finite_for_positive_domain_touching_zero() {
        let s = d([0.0, 1000.0], [0.0, 3.0], 10.0, false);
        // Interior, in-domain samples must map to finite pixel coordinates.
        for x in [1.0, 10.0, 500.0, 1000.0] {
            let y = s.scale(x);
            assert!(y.is_finite(), "scale({x}) must be finite, got {y}");
        }
    }

    #[test]
    fn log_scale_finite_for_negative_domain_touching_zero() {
        let s = d([-1000.0, 0.0], [0.0, 3.0], 10.0, false);
        for x in [-1.0, -10.0, -500.0, -1000.0] {
            let y = s.scale(x);
            assert!(y.is_finite(), "scale({x}) must be finite, got {y}");
        }
    }

    #[test]
    fn log_minor_ticks_finite_for_domain_touching_zero() {
        // `minor_ticks_internal` reads the stored domain directly (not
        // through `ticks()`), so it needs its own coverage: the sanitized
        // domain must also keep `minor_ticks_log` (ticks.rs) from hitting
        // the same `ln(0) = -inf` → `as i64` saturation.
        let scale = LogScale::new_internal(vec![0.0, 1000.0], vec![0.0, 600.0], 10.0, false, false);
        let minors = scale.minor_ticks_internal();
        assert!(
            minors.iter().all(|t| t.position.is_finite()),
            "all minor ticks must be finite: {minors:?}",
        );
    }

    #[test]
    fn log_sanitize_domain_noop_for_well_formed_domains() {
        // Existing, valid log domains must be returned byte-identical —
        // sanitization must never alter behavior for well-formed input.
        assert_eq!(LogScaleData::sanitize_domain([1.0, 1000.0]), [1.0, 1000.0]);
        assert_eq!(LogScaleData::sanitize_domain([-1000.0, -1.0]), [-1000.0, -1.0]);
    }

    #[test]
    fn log_sanitize_domain_both_zero_falls_back_to_sentinel() {
        assert_eq!(LogScaleData::sanitize_domain([0.0, 0.0]), [1.0, 10.0]);
    }

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Regression: major tick positions must be stable (not changed by adding minor support).
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10 internally.
    /// The stability check compares `ticks_internal(10)` before and after — the
    /// same count that `minor_ticks_internal()` uses — so the comparison is exact.
    ///
    /// For [1, 1000] with count=10, span_decades=3 < 10 so the decade-per-tick path
    /// is NOT triggered; but for count=3 (span_decades >= count) it would be.  We
    /// also verify the count=3 path produces the expected integer-exponent ticks.
    #[test]
    fn log_major_positions_unchanged() {
        let scale = LogScale::new_internal(
            vec![1.0, 1000.0], vec![0.0, 600.0], 10.0, false, false,
        );
        // Verify count=3 triggers integer-exponent path (3 decades >= count=3).
        let majors3 = scale.ticks_internal(3);
        assert!(majors3.iter().any(|&v| (v - 1.0).abs() < 1e-9), "1 not in {majors3:?}");
        assert!(majors3.iter().any(|&v| (v - 10.0).abs() < 1e-9), "10 not in {majors3:?}");
        assert!(majors3.iter().any(|&v| (v - 100.0).abs() < 1e-9), "100 not in {majors3:?}");
        assert!(majors3.iter().any(|&v| (v - 1000.0).abs() < 1e-9), "1000 not in {majors3:?}");

        // Stability: calling minor_ticks_internal() does not mutate the scale.
        // Compare at count=10 (the count minor_ticks_internal uses internally).
        let before = scale.ticks_internal(10);
        let _ = scale.minor_ticks_internal();
        let after = scale.ticks_internal(10);
        assert_eq!(before, after);
    }

    /// Log minors for [1, 100] land at 2-9 and 20-90.
    ///
    /// `minor_ticks_internal()` uses count=10 for majors internally.
    /// For [1,100] with count=10, the major ticks include 1, 10, 100 as the
    /// decade boundaries, so the 2-9 and 20-90 multiples are all produced.
    #[test]
    fn log_minors_at_standard_multiples() {
        let scale = LogScale::new_internal(
            vec![1.0, 100.0], vec![0.0, 600.0], 10.0, false, false,
        );
        let minors = scale.minor_ticks_internal();
        assert_eq!(minors.len(), 16, "expected 16 minors, got {}: {:?}", minors.len(), minors);

        let positions: Vec<f64> = minors.iter().map(|t| t.position).collect();
        for m in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                  20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0] {
            assert!(
                positions.iter().any(|&p| (p - m).abs() < 1e-9),
                "{m} not in {positions:?}",
            );
        }
        assert!(minors.iter().all(|t| !t.is_major));
    }

    /// Log minors have non-uniform spacing in log space (crowding at top of decade).
    ///
    /// `minor_ticks_internal()` uses count=10 for majors internally.
    #[test]
    fn log_minors_nonuniform_in_linear_space() {
        let scale = LogScale::new_internal(
            vec![1.0, 100.0], vec![0.0, 600.0], 10.0, false, false,
        );
        let minors = scale.minor_ticks_internal();
        // First decade: 2,3,...,9 — uniform in LINEAR space but NOT in LOG space.
        let first_decade: Vec<f64> = minors
            .iter()
            .filter(|t| t.position >= 2.0 && t.position < 10.0)
            .map(|t| t.position)
            .collect();
        let log_gaps: Vec<f64> = first_decade
            .windows(2)
            .map(|w| w[1].ln() - w[0].ln())
            .collect();
        let first_gap = log_gaps[0];
        let all_equal = log_gaps.iter().all(|&g| (g - first_gap).abs() < 1e-9);
        assert!(!all_equal, "log-space gaps should be non-uniform: {log_gaps:?}");
    }

    /// Log major positions are unchanged by the introduction of minor ticks.
    ///
    /// Both calls use count=10 (the fixed internal count for `minor_ticks_internal`).
    #[test]
    fn log_majors_are_preserved_after_minor_call() {
        let scale = LogScale::new_internal(
            vec![1.0, 1000.0], vec![0.0, 600.0], 10.0, false, false,
        );
        let before = scale.ticks_internal(10);
        let _ = scale.minor_ticks_internal(); // side-effect free
        let after = scale.ticks_internal(10);
        assert_eq!(before, after);
    }
}
