use pyo3::prelude::*;

use super::core::{validate_threshold, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdScale(Scale);

impl ThresholdScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Threshold { domain, range } => format!(
                "ThresholdScale(domain={:?}, range={:?})",
                domain, range
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl ThresholdScale {
    #[new]
    #[pyo3(signature = (*, domain, range))]
    fn new(domain: Vec<f64>, range: Vec<f64>) -> PyResult<Self> {
        validate_threshold(&domain, &range)?;
        Ok(ThresholdScale(Scale::Threshold { domain, range }))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }

    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    fn ticks(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn nice(&self) -> Self { self.clone() }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
