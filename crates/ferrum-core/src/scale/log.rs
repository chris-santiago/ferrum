use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

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
pub struct LogScale(Scale);

impl LogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, base: f64, clamp: bool, nice: bool) -> Self {
        let mut s = super::core::Scale::Log {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            base,
            clamp,
        };
        if nice { s = s.nice(); }
        LogScale(s)
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.0.ticks(Some(count))
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        match &self.0 {
            super::core::Scale::Log { range, .. } => *range,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Log { domain, range, base, clamp } => format!(
                "LogScale(domain=[{}, {}], range=[{}, {}], base={}, clamp={})",
                domain[0], domain[1], range[0], range[1], base, if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl LogScale {
    #[new]
    #[pyo3(signature = (*, domain, range, base = 10.0, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, base: f64, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
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
        let mut s = Scale::Log {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            base,
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(LogScale(s))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    /// Return approximately ``count`` tick values spaced logarithmically within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(Some(count)) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest power of ``base``.
    fn nice(&self) -> Self { LogScale(self.0.clone().nice()) }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Log { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Output range as ``[lo, hi]`` pixel coordinates.
    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Log { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Logarithm base (default 10.0).
    #[getter]
    fn base(&self) -> f64 {
        match &self.0 {
            Scale::Log { base, .. } => *base,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Log { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
