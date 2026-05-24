use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::ticks::nice_ticks;

#[derive(Debug, Clone, PartialEq)]
struct LogScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    base: f64,
    clamp: bool,
}

impl LogScaleData {
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
        let lo_exp = (lo.ln() / log_base).floor() as i64;
        let hi_exp = (hi.ln() / log_base).ceil() as i64;
        let span_decades = (hi_exp - lo_exp).max(1) as usize;
        if span_decades >= count {
            let mut out: Vec<f64> = (lo_exp..=hi_exp)
                .map(|e| sign * self.base.powi(e as i32))
                .filter(|t| (t.abs() >= lo) && (t.abs() <= hi))
                .collect();
            if self.domain[0] > self.domain[1] { out.reverse(); }
            out
        } else {
            let lvals = nice_ticks(lo.ln() / log_base, hi.ln() / log_base, count);
            let mut out: Vec<f64> = lvals.into_iter().map(|lv| sign * self.base.powf(lv)).collect();
            if self.domain[0] > self.domain[1] { out.reverse(); }
            out
        }
    }

    fn nice(self) -> Self {
        let neg = self.domain[0] < 0.0;
        let sign: f64 = if neg { -1.0 } else { 1.0 };
        let log_base = self.base.ln();
        let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
        let hi = (self.domain[0] * sign).max(self.domain[1] * sign);
        let lo_exp = (lo.ln() / log_base).floor();
        let hi_exp = (hi.ln() / log_base).ceil();
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
pub struct LogScale(LogScaleData, Option<f64>, bool);

impl LogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, base: f64, clamp: bool, nice: bool) -> Self {
        let mut d = LogScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            base,
            clamp,
        };
        if nice { d = d.nice(); }
        LogScale(d, None, true)
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 { self.0.scale(x) }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    pub(crate) fn range_pair(&self) -> [f64; 2] { self.0.range }

    pub(crate) fn domain_pair(&self) -> [f64; 2] { self.0.domain }

    pub(crate) fn repr_string(&self) -> String {
        let LogScaleData { domain, range, base, clamp } = &self.0;
        format!(
            "LogScale(domain=[{}, {}], range=[{}, {}], base={}, clamp={})",
            domain[0], domain[1], range[0], range[1], base, if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl LogScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, base = 10.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Option<Vec<f64>>,
        base: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        let range_user_set = range.is_some();
        let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
        validate_continuous_pair(&domain, &r)?;
        if !base.is_finite() || base <= 0.0 || base == 1.0 {
            return Err(PyValueError::new_err(format!(
                "base must be finite, > 0, and != 1; got {base}"
            )));
        }
        if domain[0] == 0.0 || domain[1] == 0.0 {
            return Err(PyValueError::new_err(
                "log scale domain must not contain 0",
            ));
        }
        if domain[0].signum() != domain[1].signum() {
            return Err(PyValueError::new_err(
                "log scale domain endpoints must have the same sign",
            ));
        }
        let mut d = LogScaleData {
            domain: [domain[0], domain[1]],
            range:  [r[0],  r[1]],
            base,
            clamp,
        };
        if nice {
            d = d.nice();
        }
        Ok(LogScale(d, padding, range_user_set))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }
    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` tick values spaced logarithmically within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest power of ``base``.
    fn nice(&self) -> Self { LogScale(self.0.clone().nice(), self.1, self.2) }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.1 }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.to_vec() }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.2 { Some(self.0.range.to_vec()) } else { None }
    }

    /// Logarithm base (default 10.0).
    #[getter]
    fn base(&self) -> f64 { self.0.base }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool) -> LogScaleData {
        LogScaleData { domain, range, base, clamp }
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
}
