use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::ticks::{nice_step, nice_ticks};

/// Internal data for a linear-affine scale. Shared by [`LinearScale`] and
/// [`super::time::TimeScale`] (time scales use the same domain-to-range
/// mapping but add time-aware tick generation).
///
/// Kept `pub(super)` so `scale::time` can reach it without exposing the
/// data shape to the rest of the crate. The PyO3 newtypes wrap this struct
/// alongside their padding field; render-side callers go through the
/// newtypes' `scale_internal` / `range_pair` / `ticks_internal` accessors.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LinearScaleData {
    pub(super) domain: [f64; 2],
    pub(super) range: [f64; 2],
    pub(super) clamp: bool,
}

impl LinearScaleData {
    pub(super) fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let t = (x - d0) / (d1 - d0);
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

    pub(super) fn invert(&self, y: f64) -> f64 {
        if y.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let t = (y - r0) / (r1 - r0);
        let mapped = d0 + t * (d1 - d0);
        if self.clamp {
            let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
            mapped.clamp(lo, hi)
        } else if y < r0.min(r1) || y > r0.max(r1) {
            f64::NAN
        } else {
            mapped
        }
    }

    pub(super) fn ticks(&self, count: usize) -> Vec<f64> {
        nice_ticks(self.domain[0], self.domain[1], count)
    }

    pub(super) fn nice(self) -> Self {
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
        Self { domain: new_domain, range: self.range, clamp: self.clamp }
    }
}

/// Continuous linear scale.
///
/// Maps a numeric domain to a numeric range via affine transformation.
/// Domain endpoints are derived from data min/max when not supplied;
/// range is derived from the axis pixel extent.
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
///
/// Examples
/// --------
/// Scales are normally constructed implicitly by ``Chart.encode(...)``.
/// Pass an instance explicitly to override the defaults::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         x=fr.X("value", scale=fr.LinearScale(domain=[0, 100], range=[0, 400]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearScale(LinearScaleData, Option<f64>);

impl LinearScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `padding` defaults to `None`; the renderer applies its own padding fraction
    /// before constructing the scale, so this field is meaningful only on
    /// user-supplied scale specs that roundtrip through serde.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let mut d = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            d = d.nice();
        }
        LinearScale(d, None)
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale(x)
    }

    /// Crate-internal tick call.
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.0.ticks(count)
    }

    /// Pixel-range pair `[lo, hi]` of the underlying scale. Used by `ScaleKind::pixel_range`.
    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.0.range
    }

    pub(crate) fn repr_string(&self) -> String {
        let LinearScaleData { domain, range, clamp } = &self.0;
        format!(
            "LinearScale(domain=[{}, {}], range=[{}, {}], clamp={})",
            domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl LinearScale {
    #[new]
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Vec<f64>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let mut d = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            d = d.nice();
        }
        Ok(LinearScale(d, padding))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 {
        self.0.scale(x)
    }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 {
        self.0.invert(y)
    }

    /// Return approximately ``count`` evenly-spaced tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> {
        self.0.ticks(count)
    }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self {
        LinearScale(self.0.clone().nice(), self.1)
    }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset; an explicit value
    /// (including 0.0) overrides the default at render time.
    #[getter]
    fn padding(&self) -> Option<f64> {
        self.1
    }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> {
        self.0.domain.to_vec()
    }

    /// Output range as ``[lo, hi]`` pixel coordinates.
    #[getter]
    fn range(&self) -> Vec<f64> {
        self.0.range.to_vec()
    }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool {
        self.0.clamp
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], clamp: bool) -> LinearScaleData {
        LinearScaleData { domain, range, clamp }
    }

    #[test]
    fn linear_scale_basic() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        assert!((s.scale(5.0) - 0.5).abs() < 1e-12);
        assert!((s.scale(0.0) - 0.0).abs() < 1e-12);
        assert!((s.scale(10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_inversion_round_trip() {
        let s = d([-50.0, 50.0], [0.0, 100.0], false);
        for x in [-50.0, -25.0, 0.0, 17.5, 50.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn linear_out_of_domain_returns_nan_when_unclamped() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        assert!(s.scale(-1.0).is_nan());
        assert!(s.scale(11.0).is_nan());
    }

    #[test]
    fn linear_clamp_clamps_output() {
        let s = d([0.0, 10.0], [0.0, 1.0], true);
        assert_eq!(s.scale(-1.0), 0.0);
        assert_eq!(s.scale(11.0), 1.0);
    }

    #[test]
    fn linear_nan_propagates() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        assert!(s.scale(f64::NAN).is_nan());
        assert!(s.invert(f64::NAN).is_nan());
    }

    #[test]
    fn linear_ticks_default_count() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        let t = s.ticks(10);
        assert!(t.len() >= 5, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn linear_nice_idempotent() {
        let s = d([0.13, 9.7], [0.0, 1.0], false);
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1, n2);
    }
}
