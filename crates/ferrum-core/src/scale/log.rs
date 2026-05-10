use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

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

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(Some(count)) }

    fn nice(&self) -> Self { LogScale(self.0.clone().nice()) }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Log { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Log { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn base(&self) -> f64 {
        match &self.0 {
            Scale::Log { base, .. } => *base,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

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
