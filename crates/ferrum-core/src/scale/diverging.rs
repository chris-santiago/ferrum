use pyo3::prelude::*;

use super::core::scale_spec_to_py_dict;
use crate::spec::encoding::ScaleSpec;

/// Diverging color-mapping scale.
///
/// Maps a continuous numeric domain with a meaningful midpoint to a
/// diverging color scheme. The domain is typically ``[lo, mid, hi]`` where
/// ``mid`` is a neutral value (often zero or the mean). Colors interpolate
/// from one extreme through a neutral center to the other extreme.
///
/// Parameters
/// ----------
/// scheme : str, optional
///     Name of the diverging color scheme (e.g., ``"rdbu"``,
///     ``"brbg"``, ``"piyg"``). When ``None``, the renderer falls back
///     to the theme's default diverging scheme.
/// domain : tuple[float, float, float], optional
///     Input domain as ``[lo, mid, hi]``. When ``None``, the renderer
///     derives from data extent with midpoint at 0.
/// domain_mid : float, optional
///     Alternative way to set just the midpoint while letting the renderer
///     derive ``lo`` and ``hi`` from data. Ignored when ``domain`` is set.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct DivergingScale {
    scheme: Option<String>,
    domain: Option<[f64; 3]>,
    domain_mid: Option<f64>,
}

impl DivergingScale {
    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `scheme` is emitted only when a non-empty string; `domain_mid` is carried
    /// whenever set (the legacy `is not None` guard, so `0.0` is preserved).
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Diverging {
            scheme: self.scheme.as_ref().filter(|s| !s.is_empty()).cloned(),
            domain: self.domain.map(|d| d.to_vec()),
            domain_mid: self.domain_mid,
        }
    }
}

#[pymethods]
impl DivergingScale {
    #[new]
    #[pyo3(signature = (*, scheme = None, domain = None, domain_mid = None))]
    fn new(
        scheme: Option<String>,
        domain: Option<Vec<f64>>,
        domain_mid: Option<f64>,
    ) -> PyResult<Self> {
        let d = domain.map(|v| {
            if v.len() >= 3 {
                [v[0], v[1], v[2]]
            } else if v.len() == 2 {
                [v[0], (v[0] + v[1]) / 2.0, v[1]]
            } else {
                [0.0, 0.5, 1.0]
            }
        });
        Ok(DivergingScale { scheme, domain: d, domain_mid })
    }

    /// Name of the diverging color scheme, or ``None`` for theme default.
    #[getter]
    fn scheme(&self) -> Option<String> { self.scheme.clone() }

    /// Input domain as ``[lo, mid, hi]``, or ``None`` when data-derived.
    #[getter]
    fn domain(&self) -> Option<Vec<f64>> {
        self.domain.map(|d| d.to_vec())
    }

    /// Explicit midpoint for the domain, or ``None``.
    #[getter]
    fn domain_mid(&self) -> Option<f64> { self.domain_mid }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        format!(
            "DivergingScale(scheme={:?}, domain={:?}, domain_mid={:?})",
            self.scheme, self.domain, self.domain_mid
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diverging_scale_with_full_domain() {
        let s = DivergingScale {
            scheme: Some("rdbu".into()),
            domain: Some([-1.0, 0.0, 1.0]),
            domain_mid: None,
        };
        assert_eq!(s.scheme, Some("rdbu".into()));
        assert_eq!(s.domain, Some([-1.0, 0.0, 1.0]));
    }

    #[test]
    fn diverging_scale_defaults() {
        let s = DivergingScale {
            scheme: None,
            domain: None,
            domain_mid: Some(0.0),
        };
        assert_eq!(s.scheme, None);
        assert_eq!(s.domain, None);
        assert_eq!(s.domain_mid, Some(0.0));
    }

    #[test]
    fn diverging_scale_repr() {
        let s = DivergingScale {
            scheme: Some("brbg".into()),
            domain: Some([-5.0, 0.0, 5.0]),
            domain_mid: None,
        };
        let r = format!("{:?}", s);
        assert!(r.contains("brbg"));
    }
}
