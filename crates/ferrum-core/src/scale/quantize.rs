use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::validate_finite;

#[derive(Debug, Clone, PartialEq)]
struct QuantizeScaleData {
    domain: [f64; 2],
    range: Vec<String>,
}

impl QuantizeScaleData {
    /// Map a continuous value to one of the discrete range values.
    /// The domain is divided into equal-width bins.
    fn scale(&self, x: f64) -> Option<&str> {
        if x.is_nan() { return None; }
        let [d0, d1] = self.domain;
        let n = self.range.len();
        if n == 0 { return None; }
        // Normalize x to [0, 1] within domain
        let t = (x - d0) / (d1 - d0);
        let t_clamped = t.clamp(0.0, 1.0 - f64::EPSILON);
        let idx = (t_clamped * n as f64).floor() as usize;
        let idx = idx.min(n - 1);
        Some(&self.range[idx])
    }

    /// Return the bin thresholds (n-1 interior break points for n range values).
    fn thresholds(&self) -> Vec<f64> {
        let [d0, d1] = self.domain;
        let n = self.range.len();
        if n <= 1 { return Vec::new(); }
        let step = (d1 - d0) / n as f64;
        (1..n).map(|i| d0 + i as f64 * step).collect()
    }
}

/// Quantize scale for binning continuous data into discrete colors.
///
/// Divides a continuous numeric domain into equal-width bins and maps
/// each bin to a discrete range value (typically a color string). Unlike
/// ``ThresholdScale``, the bin widths are uniform rather than user-specified.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``. Divided into ``len(range)``
///     equal-width bins.
/// range : list[str]
///     Discrete output values (typically color hex strings), one per bin.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizeScale(QuantizeScaleData);

#[pymethods]
impl QuantizeScale {
    #[new]
    #[pyo3(signature = (*, domain, range))]
    fn new(domain: Vec<f64>, range: Vec<String>) -> PyResult<Self> {
        if domain.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "domain must have length 2; got {}",
                domain.len()
            )));
        }
        validate_finite("domain", &domain)?;
        if domain[0] == domain[1] {
            return Err(PyValueError::new_err(
                "domain endpoints must differ (lo != hi)"
            ));
        }
        if range.is_empty() {
            return Err(PyValueError::new_err("range must be non-empty"));
        }
        Ok(QuantizeScale(QuantizeScaleData {
            domain: [domain[0], domain[1]],
            range,
        }))
    }

    /// Map a continuous value ``x`` to its discrete range value (color).
    ///
    /// Returns ``None`` if ``x`` is NaN.
    fn scale(&self, x: f64) -> Option<String> {
        self.0.scale(x).map(|s| s.to_owned())
    }

    /// Return the computed bin thresholds (interior break points).
    fn thresholds(&self) -> Vec<f64> {
        self.0.thresholds()
    }

    /// Return this scale unchanged (quantize scales have no "nice" rounding).
    fn nice(&self) -> Self { self.clone() }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.to_vec() }

    /// Discrete output values, one per bin.
    #[getter]
    fn range(&self) -> Vec<String> { self.0.range.clone() }

    fn __repr__(&self) -> String {
        format!(
            "QuantizeScale(domain=[{}, {}], range={:?})",
            self.0.domain[0], self.0.domain[1], self.0.range
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_scale_basic() {
        let s = QuantizeScaleData {
            domain: [0.0, 100.0],
            range: vec!["low".into(), "mid".into(), "high".into()],
        };
        assert_eq!(s.scale(10.0), Some("low"));
        assert_eq!(s.scale(50.0), Some("mid"));
        assert_eq!(s.scale(90.0), Some("high"));
    }

    #[test]
    fn quantize_scale_boundaries() {
        let s = QuantizeScaleData {
            domain: [0.0, 100.0],
            range: vec!["a".into(), "b".into()],
        };
        // < 50 => "a", >= 50 => "b"
        assert_eq!(s.scale(0.0), Some("a"));
        assert_eq!(s.scale(49.9), Some("a"));
        assert_eq!(s.scale(50.0), Some("b"));
        assert_eq!(s.scale(99.9), Some("b"));
    }

    #[test]
    fn quantize_scale_clamps_extremes() {
        let s = QuantizeScaleData {
            domain: [0.0, 100.0],
            range: vec!["a".into(), "b".into(), "c".into()],
        };
        // Values outside domain are clamped
        assert_eq!(s.scale(-10.0), Some("a"));
        assert_eq!(s.scale(200.0), Some("c"));
    }

    #[test]
    fn quantize_scale_nan_returns_none() {
        let s = QuantizeScaleData {
            domain: [0.0, 100.0],
            range: vec!["a".into(), "b".into()],
        };
        assert_eq!(s.scale(f64::NAN), None);
    }

    #[test]
    fn quantize_thresholds() {
        let s = QuantizeScaleData {
            domain: [0.0, 100.0],
            range: vec!["a".into(), "b".into(), "c".into()],
        };
        let t = s.thresholds();
        assert_eq!(t.len(), 2);
        assert!((t[0] - 100.0 / 3.0).abs() < 1e-9);
        assert!((t[1] - 200.0 / 3.0).abs() < 1e-9);
    }
}
