use pyo3::prelude::*;

use super::core::{validate_quantile, Scale};

/// Quantile-binned discrete scale.
///
/// Partitions a numeric sample domain into ``len(range)`` equal-probability
/// bins by computing quantile cut-points, then maps each input value to its
/// bin's range value. Useful for diverging or sequential color encodings
/// where data density matters more than equal-width intervals.
///
/// Parameters
/// ----------
/// domain : list[float]
///     Numeric sample values. The scale sorts these internally to compute
///     quantile boundaries.
/// range : list[float]
///     Discrete output values, one per bin. The number of bins equals
///     ``len(range)``.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     scale = fr.QuantileScale(domain=data, range=[0.0, 0.5, 1.0])
///     # Three equal-frequency bins mapped to 0.0, 0.5, or 1.0.
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

    /// Map a single input value ``x`` to its bin's range value.
    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }

    /// Return the ``[lo, hi]`` domain extent of the bin that contains range
    /// value ``y``.
    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    /// Return tick values (the quantile cut-points).
    #[pyo3(signature = (count = None))]
    fn ticks(&self, count: Option<usize>) -> Vec<f64> { self.0.ticks(count) }

    /// Return this scale unchanged (quantile scales have no "nice" rounding).
    fn nice(&self) -> Self { self.clone() }

    /// Sorted sample values used to compute quantile boundaries.
    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Discrete output values, one per bin.
    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Quantile { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Computed quantile cut-point boundaries.
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
