use pyo3::prelude::*;

use super::core::{validate_quantile, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileScale(Scale);

impl QuantileScale {
    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Quantile { domain, range, quantiles } => format!(
                "QuantileScale(domain=<{} samples>, range={:?}, quantiles={:?})",
                domain.len(), range, quantiles
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl QuantileScale {
    #[new]
    #[pyo3(signature = (*, domain, range))]
    fn new(domain: Vec<f64>, range: Vec<f64>) -> PyResult<Self> {
        validate_quantile(&domain, &range)?;
        let mut sorted = domain.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let quantiles = Scale::compute_quantile_cuts(&sorted, range.len());
        Ok(QuantileScale(Scale::Quantile {
            domain: sorted,
            range,
            quantiles,
        }))
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }

    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    #[pyo3(signature = (count = None))]
    fn ticks(&self, count: Option<usize>) -> Vec<f64> { self.0.ticks(count) }

    fn nice(&self) -> Self { self.clone() }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { quantiles, .. } => quantiles.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
