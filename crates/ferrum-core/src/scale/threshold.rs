use pyo3::prelude::*;

use super::core::validate_threshold;

#[derive(Debug, Clone, PartialEq)]
struct ThresholdScaleData {
    domain: Vec<f64>,
    range: Vec<f64>,
}

impl ThresholdScaleData {
    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let idx = self.domain.partition_point(|t| *t <= x);
        self.range[idx]
    }

    fn invert_extent(&self, y: f64) -> (f64, f64) {
        if y.is_nan() { return (f64::NAN, f64::NAN); }
        let idx = match self.range.iter().position(|r| *r == y) {
            Some(i) => i,
            None => return (f64::NAN, f64::NAN),
        };
        let lo = if idx == 0 { f64::NEG_INFINITY } else { self.domain[idx - 1] };
        let hi = if idx >= self.domain.len() { f64::INFINITY } else { self.domain[idx] };
        (lo, hi)
    }
}

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
pub struct ThresholdScale(ThresholdScaleData);

impl ThresholdScale {
    pub(crate) fn repr_string(&self) -> String {
        format!(
            "ThresholdScale(domain={:?}, range={:?})",
            self.0.domain, self.0.range
        )
    }
}

#[pymethods]
impl ThresholdScale {
    #[new]
    #[pyo3(signature = (*, domain, range))]
    fn new(domain: Vec<f64>, range: Vec<f64>) -> PyResult<Self> {
        validate_threshold(&domain, &range)?;
        Ok(ThresholdScale(ThresholdScaleData { domain, range }))
    }

    /// Map a single input value ``x`` to its bin's range value.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }

    /// Return the ``[lo, hi]`` domain extent of the bin that contains range
    /// value ``y``.
    fn invert_extent(&self, y: f64) -> (f64, f64) { self.0.invert_extent(y) }

    /// Return the threshold break values (the domain).
    fn ticks(&self) -> Vec<f64> { self.0.domain.clone() }

    /// Return this scale unchanged (threshold scales have no "nice" rounding).
    fn nice(&self) -> Self { self.clone() }

    /// Sorted threshold values that define bin boundaries.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.clone() }

    /// Discrete output values, one per bin.
    #[getter]
    fn range(&self) -> Vec<f64> { self.0.range.clone() }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_scale_basic() {
        let s = ThresholdScaleData {
            domain: vec![0.0, 10.0],
            range: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(s.scale(-1.0), 1.0);
        assert_eq!(s.scale(0.0), 2.0);
        assert_eq!(s.scale(5.0), 2.0);
        assert_eq!(s.scale(10.0), 3.0);
        assert_eq!(s.scale(20.0), 3.0);
    }

    #[test]
    fn threshold_invert_extent_round_trip() {
        let s = ThresholdScaleData {
            domain: vec![0.0, 10.0],
            range: vec![1.0, 2.0, 3.0],
        };
        let (lo, hi) = s.invert_extent(2.0);
        assert_eq!((lo, hi), (0.0, 10.0));
        let (lo, hi) = s.invert_extent(1.0);
        assert!(lo.is_infinite() && lo.is_sign_negative());
        assert_eq!(hi, 0.0);
        let (lo, hi) = s.invert_extent(3.0);
        assert_eq!(lo, 10.0);
        assert!(hi.is_infinite() && hi.is_sign_positive());
    }

    #[test]
    fn threshold_invert_extent_unknown_returns_nan() {
        let s = ThresholdScaleData {
            domain: vec![0.0],
            range: vec![1.0, 2.0],
        };
        let (lo, hi) = s.invert_extent(99.0);
        assert!(lo.is_nan() && hi.is_nan());
    }
}
