use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{
    is_strictly_ascending, not_strictly_ascending_message, scale_spec_to_py_dict, validate_finite,
};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct BinOrdinalScaleData {
    bins: Vec<f64>,
    scheme: Option<String>,
}

impl BinOrdinalScaleData {
    /// Map a continuous value to its bin index (0-based).
    /// Values below first bin edge go to bin 0; values at or above
    /// the last edge go to the last bin.
    fn bin_index(&self, x: f64) -> Option<usize> {
        if x.is_nan() { return None; }
        if self.bins.is_empty() { return None; }
        // partition_point finds first bin edge > x
        let idx = self.bins.partition_point(|b| *b <= x);
        // idx ranges from 0 (x < bins[0]) to bins.len() (x >= all)
        // We have bins.len() - 1 intervals + 2 open ends = bins.len() + 1 bins total,
        // but typically the number of colors = bins.len() + 1 (one per interval).
        Some(idx)
    }

    /// Number of output bins (intervals defined by the edges).
    fn num_bins(&self) -> usize {
        if self.bins.is_empty() { 1 } else { self.bins.len() + 1 }
    }
}

/// Binned ordinal color scale.
///
/// Defines explicit bin edges for a continuous domain and maps each bin
/// to a color from a named scheme. The number of output colors equals
/// ``len(bins) + 1`` (one per interval between and beyond the edges).
///
/// Parameters
/// ----------
/// bins : list[float]
///     Sorted bin edge values. Must be strictly ascending.
/// scheme : str, optional
///     Name of the color scheme to draw bin colors from. When ``None``,
///     the renderer uses the theme's default categorical or sequential
///     scheme depending on bin count.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct BinOrdinalScale(BinOrdinalScaleData);

impl BinOrdinalScale {
    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `bins` is constructor-guaranteed non-empty; `scheme` is emitted only when
    /// a non-empty string (mirroring the legacy `if scale.scheme:` guard).
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::BinOrdinal {
            bins: if self.0.bins.is_empty() {
                None
            } else {
                Some(self.0.bins.clone())
            },
            scheme: self.0.scheme.as_ref().filter(|s| !s.is_empty()).cloned(),
        }
    }
}

#[pymethods]
impl BinOrdinalScale {
    #[new]
    #[pyo3(signature = (*, bins, scheme = None))]
    fn new(bins: Vec<f64>, scheme: Option<String>) -> PyResult<Self> {
        if bins.is_empty() {
            return Err(PyValueError::new_err("bins must be non-empty"));
        }
        validate_finite("bins", &bins)?;
        if !is_strictly_ascending(&bins) {
            return Err(PyValueError::new_err(not_strictly_ascending_message("bins")));
        }
        Ok(BinOrdinalScale(BinOrdinalScaleData { bins, scheme }))
    }

    /// Map a continuous value ``x`` to its bin index (0-based).
    ///
    /// Returns ``None`` if ``x`` is NaN.
    fn scale(&self, x: f64) -> Option<usize> {
        self.0.bin_index(x)
    }

    /// Return the bin edge values.
    fn ticks(&self) -> Vec<f64> {
        self.0.bins.clone()
    }

    /// Return this scale unchanged.
    fn nice(&self) -> Self { self.clone() }

    /// Number of output bins (colors).
    #[getter]
    fn num_bins(&self) -> usize { self.0.num_bins() }

    /// Bin edge values.
    #[getter]
    fn bins(&self) -> Vec<f64> { self.0.bins.clone() }

    /// Color scheme name, or ``None``.
    #[getter]
    fn scheme(&self) -> Option<String> { self.0.scheme.clone() }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        format!(
            "BinOrdinalScale(bins={:?}, scheme={:?})",
            self.0.bins, self.0.scheme
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_ordinal_basic() {
        let s = BinOrdinalScaleData {
            bins: vec![10.0, 20.0, 30.0],
            scheme: Some("blues".into()),
        };
        // 4 bins: (-inf, 10), [10, 20), [20, 30), [30, inf)
        assert_eq!(s.bin_index(5.0), Some(0));
        assert_eq!(s.bin_index(10.0), Some(1));
        assert_eq!(s.bin_index(15.0), Some(1));
        assert_eq!(s.bin_index(20.0), Some(2));
        assert_eq!(s.bin_index(25.0), Some(2));
        assert_eq!(s.bin_index(30.0), Some(3));
        assert_eq!(s.bin_index(50.0), Some(3));
    }

    #[test]
    fn bin_ordinal_nan() {
        let s = BinOrdinalScaleData {
            bins: vec![10.0],
            scheme: None,
        };
        assert_eq!(s.bin_index(f64::NAN), None);
    }

    #[test]
    fn bin_ordinal_num_bins() {
        let s = BinOrdinalScaleData {
            bins: vec![1.0, 2.0, 3.0, 4.0],
            scheme: None,
        };
        assert_eq!(s.num_bins(), 5);
    }
}
