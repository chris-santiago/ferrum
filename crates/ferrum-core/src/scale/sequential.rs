use pyo3::prelude::*;

use super::core::scale_spec_to_py_dict;
use crate::spec::encoding::ScaleSpec;

/// Sequential color-mapping scale.
///
/// Maps a continuous numeric domain to a named sequential color scheme.
/// The renderer uses the scheme name to look up a palette and interpolate
/// colors across the ``[0, 1]`` normalized domain. Commonly used for
/// heatmaps, choropleths, and density visualizations.
///
/// Parameters
/// ----------
/// scheme : str, optional
///     Name of the sequential color scheme (e.g., ``"viridis"``,
///     ``"blues"``, ``"inferno"``). When ``None``, the renderer falls back
///     to the theme's default sequential scheme.
/// domain : tuple[float, float], optional
///     Input domain as ``[min, max]``. When ``None``, the renderer derives
///     from data extent.
/// reverse : bool, default False
///     Reverse the color interpolation direction.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialScale {
    scheme: Option<String>,
    domain: Option<[f64; 2]>,
    reverse: bool,
}

impl SequentialScale {
    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `scheme` is emitted only when a non-empty string (mirroring the legacy
    /// `if scale.scheme:` guard); `reverse` is always carried.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Sequential {
            scheme: self.scheme.as_ref().filter(|s| !s.is_empty()).cloned(),
            domain: self.domain.map(|d| d.to_vec()),
            reverse: self.reverse,
        }
    }
}

#[pymethods]
impl SequentialScale {
    #[new]
    #[pyo3(signature = (*, scheme = None, domain = None, reverse = false))]
    fn new(
        scheme: Option<String>,
        domain: Option<Vec<f64>>,
        reverse: bool,
    ) -> PyResult<Self> {
        let d = domain.map(|v| {
            if v.len() >= 2 { [v[0], v[1]] } else { [0.0, 1.0] }
        });
        Ok(SequentialScale { scheme, domain: d, reverse })
    }

    /// Name of the sequential color scheme, or ``None`` for theme default.
    #[getter]
    fn scheme(&self) -> Option<String> { self.scheme.clone() }

    /// Input domain as ``[min, max]``, or ``None`` when data-derived.
    #[getter]
    fn domain(&self) -> Option<Vec<f64>> {
        self.domain.map(|d| d.to_vec())
    }

    /// Whether the color direction is reversed.
    #[getter]
    fn reverse(&self) -> bool { self.reverse }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        format!(
            "SequentialScale(scheme={:?}, domain={:?}, reverse={})",
            self.scheme, self.domain,
            if self.reverse { "True" } else { "False" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_scale_with_scheme() {
        let s = SequentialScale {
            scheme: Some("viridis".into()),
            domain: Some([0.0, 100.0]),
            reverse: false,
        };
        assert_eq!(s.scheme, Some("viridis".into()));
        assert_eq!(s.domain, Some([0.0, 100.0]));
        assert!(!s.reverse);
    }

    #[test]
    fn sequential_scale_defaults() {
        let s = SequentialScale {
            scheme: None,
            domain: None,
            reverse: false,
        };
        assert_eq!(s.scheme, None);
        assert_eq!(s.domain, None);
    }

    #[test]
    fn sequential_scale_reverse() {
        let s = SequentialScale {
            scheme: Some("blues".into()),
            domain: Some([0.0, 1.0]),
            reverse: true,
        };
        assert!(s.reverse);
    }
}
