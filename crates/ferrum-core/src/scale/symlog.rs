use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SymlogScale(Scale);

impl SymlogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, constant: f64, clamp: bool, nice: bool) -> Self {
        let mut s = super::core::Scale::Symlog {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            constant,
            clamp,
        };
        if nice { s = s.nice(); }
        SymlogScale(s)
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.0.ticks(Some(count))
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        match &self.0 {
            super::core::Scale::Symlog { range, .. } => *range,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Symlog { domain, range, constant, clamp } => format!(
                "SymlogScale(domain=[{}, {}], range=[{}, {}], constant={}, clamp={})",
                domain[0], domain[1], range[0], range[1], constant, if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl SymlogScale {
    #[new]
    #[pyo3(signature = (*, domain, range, constant = 1.0, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, constant: f64, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        if !constant.is_finite() || constant <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "constant must be finite and > 0; got {constant}"
            )));
        }
        let mut s = Scale::Symlog {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            constant,
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(SymlogScale(s))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(Some(count)) }

    fn nice(&self) -> Self { SymlogScale(self.0.clone().nice()) }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Symlog { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Symlog { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn constant(&self) -> f64 {
        match &self.0 {
            Scale::Symlog { constant, .. } => *constant,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Symlog { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
