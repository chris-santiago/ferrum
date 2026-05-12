use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::ticks::{nice_step, nice_ticks};

fn symlog_fwd(x: f64, c: f64) -> f64 {
    x.signum() * (x.abs() / c).ln_1p()
}

fn symlog_inv(y: f64, c: f64) -> f64 {
    y.signum() * c * (y.abs().exp() - 1.0)
}

#[derive(Debug, Clone, PartialEq)]
struct SymlogScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    constant: f64,
    clamp: bool,
}

impl SymlogScaleData {
    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let f = |v: f64| symlog_fwd(v, self.constant);
        let t = (f(x) - f(d0)) / (f(d1) - f(d0));
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
        let f = |v: f64| symlog_fwd(v, self.constant);
        let t = (y - r0) / (r1 - r0);
        let lmapped = f(d0) + t * (f(d1) - f(d0));
        let mapped = symlog_inv(lmapped, self.constant);
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
        Self { domain: new_domain, range: self.range, constant: self.constant, clamp: self.clamp }
    }
}

/// Symmetric logarithmic scale.
///
/// Maps a numeric domain — including zero and negative values — to a range
/// using a bi-symmetric log transform. The transformation is linear in
/// ``[-constant, +constant]`` and logarithmic outside that band, so zero and
/// sign changes are handled without special-casing.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``. May span zero or be entirely negative.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// constant : float, default 1.0
///     Half-width of the linear region around zero. Must be finite and
///     positive.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to "nice" values for tick generation.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         y=fr.Y("delta", scale=fr.SymlogScale(domain=[-1000, 1000], range=[400, 0]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SymlogScale(SymlogScaleData, Option<f64>);

impl SymlogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, constant: f64, clamp: bool, nice: bool) -> Self {
        let mut d = SymlogScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            constant,
            clamp,
        };
        if nice { d = d.nice(); }
        SymlogScale(d, None)
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 { self.0.scale(x) }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    pub(crate) fn range_pair(&self) -> [f64; 2] { self.0.range }

    pub(crate) fn repr_string(&self) -> String {
        let SymlogScaleData { domain, range, constant, clamp } = &self.0;
        format!(
            "SymlogScale(domain=[{}, {}], range=[{}, {}], constant={}, clamp={})",
            domain[0], domain[1], range[0], range[1], constant, if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl SymlogScale {
    #[new]
    #[pyo3(signature = (*, domain, range, constant = 1.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Vec<f64>,
        constant: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        if !constant.is_finite() || constant <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "constant must be finite and > 0; got {constant}"
            )));
        }
        let mut d = SymlogScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            constant,
            clamp,
        };
        if nice {
            d = d.nice();
        }
        Ok(SymlogScale(d, padding))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }
    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self { SymlogScale(self.0.clone().nice(), self.1) }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.1 }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.to_vec() }

    /// Output range as ``[lo, hi]`` pixel coordinates.
    #[getter]
    fn range(&self) -> Vec<f64> { self.0.range.to_vec() }

    /// Half-width of the linear region around zero (default 1.0).
    #[getter]
    fn constant(&self) -> f64 { self.0.constant }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool) -> SymlogScaleData {
        SymlogScaleData { domain, range, constant, clamp }
    }

    #[test]
    fn symlog_scale_handles_zero() {
        let s = d([-100.0, 100.0], [0.0, 1.0], 1.0, false);
        let y = s.scale(0.0);
        assert!(y.is_finite(), "scale(0) returned {y}");
        assert!((y - 0.5).abs() < 1e-12, "expected 0.5, got {y}");
    }

    #[test]
    fn symlog_inversion_round_trip_across_zero() {
        let s = d([-1000.0, 1000.0], [0.0, 1.0], 1.0, false);
        for x in [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-6, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn symlog_constant_changes_curvature() {
        // Larger constant → more linear (less compression) → x=50 on [-100,100] maps closer to 0.5
        // Smaller constant → more log-like → x=50 maps closer to 1.0 (compressed toward endpoint)
        let s1 = d([-100.0, 100.0], [0.0, 1.0], 1.0,   false);
        let s2 = d([-100.0, 100.0], [0.0, 1.0], 100.0, false);
        let y1 = s1.scale(50.0);
        let y2 = s2.scale(50.0);
        assert!(y1 > y2, "expected y1={y1} > y2={y2}: smaller constant compresses more");
        assert!(y1 > 0.5 && y1 < 1.0, "y1={y1} out of (0.5,1.0)");
        assert!(y2 > 0.5 && y2 < 1.0, "y2={y2} out of (0.5,1.0)");
    }
}
