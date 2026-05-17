use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::ticks::{nice_step, nice_ticks};

#[derive(Debug, Clone, PartialEq)]
struct PowScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    exponent: f64,
    clamp: bool,
}

impl PowScaleData {
    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let pow_fwd = |v: f64| v.signum() * v.abs().powf(self.exponent);
        let t = (pow_fwd(x) - pow_fwd(d0)) / (pow_fwd(d1) - pow_fwd(d0));
        let mapped = r0 + t * (r1 - r0);
        if self.clamp {
            let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
            mapped.clamp(lo, hi)
        } else if x < d0.min(d1) || x > d0.max(d1) {
            f64::NAN
        } else {
            mapped
        }
    }

    fn invert(&self, y: f64) -> f64 {
        if y.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let pow_fwd = |v: f64| v.signum() * v.abs().powf(self.exponent);
        let pow_inv = |v: f64| v.signum() * v.abs().powf(1.0 / self.exponent);
        let t = (y - r0) / (r1 - r0);
        let lmapped = pow_fwd(d0) + t * (pow_fwd(d1) - pow_fwd(d0));
        let mapped = pow_inv(lmapped);
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
        nice_ticks(self.domain[0], self.domain[1], count)
    }

    fn nice(self) -> Self {
        let step = nice_step(self.domain[0], self.domain[1], 10);
        if !step.is_finite() || step == 0.0 {
            return self;
        }
        let lo_min = self.domain[0].min(self.domain[1]);
        let hi_max = self.domain[0].max(self.domain[1]);
        let nice_lo = (lo_min / step).floor() * step;
        let nice_hi = (hi_max / step).ceil() * step;
        let new_domain = if self.domain[0] <= self.domain[1] {
            [nice_lo, nice_hi]
        } else {
            [nice_hi, nice_lo]
        };
        Self { domain: new_domain, range: self.range, exponent: self.exponent, clamp: self.clamp }
    }
}

/// Continuous power scale.
///
/// Maps a numeric domain to a numeric range via a power transformation
/// (x^exponent). Useful for data where perceptual linearity requires
/// non-linear scaling (e.g., bubble area encoding with exponent=0.5).
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// exponent : float, default 2.0
///     The power exponent. Must be finite and positive.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to "nice" values for tick generation.
/// padding : float, optional
///     Fractional inward pixel padding.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct PowScale(PowScaleData, Option<f64>, bool);

impl PowScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, exponent: f64, clamp: bool) -> Self {
        let d = PowScaleData {
            domain: [domain[0], domain[1]],
            range: [range[0], range[1]],
            exponent,
            clamp,
        };
        PowScale(d, None, true)
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale(x)
    }

    /// Crate-internal tick call.
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.0.ticks(count)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.0.range
    }

    pub(crate) fn domain_pair(&self) -> [f64; 2] {
        self.0.domain
    }

    fn repr_string(&self) -> String {
        let PowScaleData { domain, range, exponent, clamp } = &self.0;
        format!(
            "PowScale(domain=[{}, {}], range=[{}, {}], exponent={}, clamp={})",
            domain[0], domain[1], range[0], range[1], exponent,
            if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl PowScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, exponent = 2.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Option<Vec<f64>>,
        exponent: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        let range_user_set = range.is_some();
        let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
        validate_continuous_pair(&domain, &r)?;
        if !exponent.is_finite() || exponent <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "exponent must be finite and > 0; got {exponent}"
            )));
        }
        let mut d = PowScaleData {
            domain: [domain[0], domain[1]],
            range: [r[0], r[1]],
            exponent,
            clamp,
        };
        if nice {
            d = d.nice();
        }
        Ok(PowScale(d, padding, range_user_set))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self { PowScale(self.0.clone().nice(), self.1, self.2) }

    /// Fractional inward pixel padding.
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

    /// The power exponent.
    #[getter]
    fn exponent(&self) -> f64 { self.0.exponent }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

    fn __repr__(&self) -> String { self.repr_string() }
}

/// Continuous square-root scale (convenience for PowScale with exponent=0.5).
///
/// Equivalent to ``PowScale(exponent=0.5, ...)``. Commonly used for area
/// encodings where perceived size should scale linearly with value.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to "nice" values for tick generation.
/// padding : float, optional
///     Fractional inward pixel padding.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SqrtScale(PowScaleData, Option<f64>, bool);

#[pymethods]
impl SqrtScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Option<Vec<f64>>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        let range_user_set = range.is_some();
        let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
        validate_continuous_pair(&domain, &r)?;
        let mut d = PowScaleData {
            domain: [domain[0], domain[1]],
            range: [r[0], r[1]],
            exponent: 0.5,
            clamp,
        };
        if nice {
            d = d.nice();
        }
        Ok(SqrtScale(d, padding, range_user_set))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self { SqrtScale(self.0.clone().nice(), self.1, self.2) }

    /// Fractional inward pixel padding.
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

    /// The power exponent (always 0.5 for SqrtScale).
    #[getter]
    fn exponent(&self) -> f64 { 0.5 }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

    fn __repr__(&self) -> String {
        let PowScaleData { domain, range, clamp, .. } = &self.0;
        format!(
            "SqrtScale(domain=[{}, {}], range=[{}, {}], clamp={})",
            domain[0], domain[1], range[0], range[1],
            if *clamp { "True" } else { "False" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], exponent: f64, clamp: bool) -> PowScaleData {
        PowScaleData { domain, range, exponent, clamp }
    }

    #[test]
    fn pow_scale_basic() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        // x=0 => 0^2=0, t=0/(100^2 - 0) = 0 => 0.0
        assert!((s.scale(0.0) - 0.0).abs() < 1e-12);
        // x=100 => 100^2=10000, t=10000/10000=1 => 1.0
        assert!((s.scale(100.0) - 1.0).abs() < 1e-12);
        // x=50 => 50^2=2500, t=2500/10000=0.25
        assert!((s.scale(50.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pow_scale_inversion_round_trip() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        for x in [0.0, 10.0, 25.0, 50.0, 75.0, 100.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn sqrt_scale_has_exponent_half() {
        let s = d([0.0, 100.0], [0.0, 1.0], 0.5, false);
        // x=25 => 25^0.5=5, domain: 0^0.5=0, 100^0.5=10, t=5/10=0.5
        assert!((s.scale(25.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pow_scale_clamp() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, true);
        assert_eq!(s.scale(-10.0), 0.0);
        assert_eq!(s.scale(200.0), 1.0);
    }

    #[test]
    fn pow_scale_out_of_domain_nan() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        assert!(s.scale(-1.0).is_nan());
        assert!(s.scale(101.0).is_nan());
    }

    #[test]
    fn pow_scale_nan_propagates() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        assert!(s.scale(f64::NAN).is_nan());
        assert!(s.invert(f64::NAN).is_nan());
    }

    #[test]
    fn pow_scale_to_dict_exponent() {
        // Verify that PowScale stores exponent correctly
        let s = PowScale(
            PowScaleData { domain: [0.0, 10.0], range: [0.0, 1.0], exponent: 3.0, clamp: false },
            None,
            true,
        );
        assert_eq!(s.0.exponent, 3.0);
    }

    #[test]
    fn sqrt_scale_exponent_is_half() {
        let s = SqrtScale(
            PowScaleData { domain: [0.0, 100.0], range: [0.0, 1.0], exponent: 0.5, clamp: false },
            None,
            true,
        );
        assert_eq!(s.0.exponent, 0.5);
    }
}
