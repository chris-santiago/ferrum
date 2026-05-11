use pyo3::prelude::*;

use super::core::{validate_threshold, Scale};

/// Threshold-based discrete scale.
///
/// Partitions the numeric domain into ``len(range)`` bins defined by
/// explicit threshold values, then maps each input to its bin's range value.
/// Unlike ``QuantileScale``, bin widths are user-specified rather than
/// equal-frequency. The number of range values must equal the number of
/// thresholds plus one (``len(range) == len(domain) + 1``).
///
/// Parameters
/// ----------
/// domain : list[float]
///     Sorted threshold values that define bin boundaries. ``len(domain)``
///     must be at least 1.
/// range : list[float]
///     Discrete output values. ``len(range)`` must equal ``len(domain) + 1``.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     scale = fr.ThresholdScale(domain=[0.0, 0.5], range=[-1.0, 0.0, 1.0])
///     # x < 0   → -1.0,  0 ≤ x < 0.5 → 0.0,  x ≥ 0.5 → 1.0
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

    /// Map a single input value ``x`` to its bin's range value.
    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }

    /// Return the ``[lo, hi]`` domain extent of the bin that contains range
    /// value ``y``.
    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    /// Return the threshold break values (the domain).
    fn ticks(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Return this scale unchanged (threshold scales have no "nice" rounding).
    fn nice(&self) -> Self { self.clone() }

    /// Sorted threshold values that define bin boundaries.
    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Threshold { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Discrete output values, one per bin.
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
