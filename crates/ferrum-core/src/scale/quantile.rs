use pyo3::prelude::*;

use super::core::{compute_quantile_cuts, scale_spec_to_py_dict, validate_quantile};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct QuantileScaleData {
    domain: Vec<f64>,
    range: Vec<f64>,
    quantiles: Vec<f64>,
}

impl QuantileScaleData {
    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let idx = self.quantiles.partition_point(|q| *q <= x);
        self.range[idx]
    }

    fn invert_extent(&self, y: f64) -> (f64, f64) {
        if y.is_nan() { return (f64::NAN, f64::NAN); }
        let idx = match self.range.iter().position(|r| *r == y) {
            Some(i) => i,
            None => return (f64::NAN, f64::NAN),
        };
        let lo = if idx == 0 { f64::NEG_INFINITY } else { self.quantiles[idx - 1] };
        let hi = if idx >= self.quantiles.len() { f64::INFINITY } else { self.quantiles[idx] };
        (lo, hi)
    }

    fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        let target = count.unwrap_or_else(|| crate::scale::ticks::sturges_floor(self.domain.len()));
        if target >= self.quantiles.len() {
            self.quantiles.clone()
        } else {
            let step = self.quantiles.len() as f64 / target as f64;
            (0..target)
                .map(|i| self.quantiles[((i as f64 + 0.5) * step).floor() as usize])
                .collect()
        }
    }
}

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
pub struct QuantileScale(QuantileScaleData);

impl QuantileScale {
    pub(crate) fn repr_string(&self) -> String {
        format!(
            "QuantileScale(domain=<{} samples>, range={:?}, quantiles={:?})",
            self.0.domain.len(), self.0.range, self.0.quantiles
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `domain` is the sorted sample, `range` the discrete numeric outputs. The
    /// computed `quantiles` cut-points are deterministic from the sample, so they
    /// are **not** transmitted (recomputed on resolution).
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Quantile {
            domain: Some(self.0.domain.clone()),
            range: Some(self.0.range.clone()),
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
        let quantiles = compute_quantile_cuts(&sorted, range.len());
        Ok(QuantileScale(QuantileScaleData {
            domain: sorted,
            range,
            quantiles,
        }))
    }

    /// Map a single input value ``x`` to its bin's range value.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }

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
    fn domain(&self) -> Vec<f64> { self.0.domain.clone() }

    /// Discrete output values, one per bin.
    #[getter]
    fn range(&self) -> Vec<f64> { self.0.range.clone() }

    /// Computed quantile cut-point boundaries.
    #[getter]
    fn quantiles(&self) -> Vec<f64> { self.0.quantiles.clone() }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_scale_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = compute_quantile_cuts(&sorted, 3);
        let s = QuantileScaleData {
            domain: sorted,
            range: vec![10.0, 20.0, 30.0],
            quantiles: cuts,
        };
        assert_eq!(s.scale(0.0), 10.0);
        assert_eq!(s.scale(2.5), 20.0);
        assert_eq!(s.scale(10.0), 30.0);
    }

    #[test]
    fn quantile_invert_extent_round_trip() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = compute_quantile_cuts(&sorted, 3);
        let s = QuantileScaleData {
            domain: sorted,
            range: vec![10.0, 20.0, 30.0],
            quantiles: cuts.clone(),
        };
        let (lo, hi) = s.invert_extent(20.0);
        assert!((lo - cuts[0]).abs() < 1e-9);
        assert!((hi - cuts[1]).abs() < 1e-9);
    }

    #[test]
    fn quantile_ticks_default_uses_sturges_floor() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = compute_quantile_cuts(&sorted, 10);
        let s = QuantileScaleData {
            domain: sorted,
            range: vec![0.0; 10],
            quantiles: cuts,
        };
        let t = s.ticks(None);
        assert_eq!(t.len(), 4);
    }
}
