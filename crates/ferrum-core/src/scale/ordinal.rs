use pyo3::prelude::*;

use super::core::{validate_ordinal, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalScale(Scale);

impl OrdinalScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Ordinal { domain, range, padding } => format!(
                "OrdinalScale(domain={:?}, range=[{}, {}], padding={})",
                domain, range.first().copied().unwrap_or(0.0), range.last().copied().unwrap_or(0.0), padding
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl OrdinalScale {
    #[new]
    #[pyo3(signature = (*, domain, range, padding = 0.0))]
    fn new(domain: Vec<String>, range: Vec<f64>, padding: f64) -> PyResult<Self> {
        validate_ordinal(&domain, &range, padding)?;
        Ok(OrdinalScale(Scale::Ordinal { domain, range, padding }))
    }

    fn scale(&self, value: &str) -> f64 {
        self.0.scale_str(value)
    }

    fn invert(&self, y: f64) -> Option<String> {
        self.0.invert_band(y)
    }

    fn ticks(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn nice(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn domain(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Ordinal { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn padding(&self) -> f64 {
        match &self.0 {
            Scale::Ordinal { padding, .. } => *padding,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
