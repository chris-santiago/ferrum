use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{scale_spec_to_py_dict, validate_finite, DEGENERATE_DOMAIN_MESSAGE};
use crate::spec::encoding::ScaleSpec;

/// Diverging color-mapping scale.
///
/// Maps a continuous numeric domain with a meaningful midpoint to a
/// diverging color scheme. The domain is typically ``[lo, mid, hi]`` where
/// ``mid`` is a neutral value (often zero or the mean). Colors interpolate
/// from one extreme through a neutral center to the other extreme.
///
/// On a positional (x/y) channel the domain collapses to its outer bounds
/// ``[lo, hi]``; the midpoint only affects color mapping.
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
        let d = match domain {
            Some(v) => {
                // Reject a domain with fewer than 2 elements instead of
                // silently substituting the [0.0, 0.5, 1.0] placeholder as if
                // it were the user's explicit intent (GH #69) — the getter,
                // the wire dict, and `positional_extent` would otherwise all
                // report a domain the user never asked for.
                if v.len() < 2 {
                    return Err(PyValueError::new_err(format!(
                        "domain must have length >= 2 ([lo, hi]) or 3 ([lo, mid, hi]); got {}",
                        v.len()
                    )));
                }
                validate_finite("domain", &v)?;
                // Reject a degenerate domain (lo == hi) the same way
                // `resolve_continuous`/`QuantizeScale` do for their own
                // domains (GH #69 cohesion fix) — a zero-width diverging
                // domain has no meaningful midpoint and would divide by zero
                // in `positional_extent`'s downstream consumers.
                if v.len() >= 3 {
                    if v[0] == v[2] {
                        return Err(PyValueError::new_err(DEGENERATE_DOMAIN_MESSAGE));
                    }
                    Some([v[0], v[1], v[2]])
                } else {
                    if v[0] == v[1] {
                        return Err(PyValueError::new_err(DEGENERATE_DOMAIN_MESSAGE));
                    }
                    Some([v[0], (v[0] + v[1]) / 2.0, v[1]])
                }
            }
            None => None,
        };
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

    // ── constructor validation (GH #69) ──────────────────────────────────────

    /// A 2-element degenerate domain (`lo == hi`) must be rejected, matching
    /// `resolve_continuous`/`QuantizeScale`'s own domain-degeneracy check.
    #[test]
    fn diverging_scale_new_rejects_degenerate_two_element_domain() {
        assert!(DivergingScale::new(None, Some(vec![5.0, 5.0]), None).is_err());
    }

    /// A 3-element domain with `lo == hi` (matching first/last, ignoring the
    /// midpoint) must also be rejected.
    #[test]
    fn diverging_scale_new_rejects_degenerate_three_element_domain() {
        assert!(DivergingScale::new(None, Some(vec![5.0, 5.0, 5.0]), None).is_err());
        assert!(DivergingScale::new(None, Some(vec![5.0, 7.0, 5.0]), None).is_err());
    }

    /// A well-formed domain is unaffected by the new degeneracy check
    /// (non-regression pin for the existing 2-/3-element expansion).
    #[test]
    fn diverging_scale_new_accepts_well_formed_domain() {
        let s = DivergingScale::new(None, Some(vec![-10.0, 30.0]), None).unwrap();
        assert_eq!(s.domain, Some([-10.0, 10.0, 30.0]));
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
