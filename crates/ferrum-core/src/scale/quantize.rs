use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{
    degenerate_ratio, scale_spec_to_py_dict, uniform_bin_thresholds, validate_finite,
    DEGENERATE_DOMAIN_MESSAGE,
};
use crate::spec::encoding::ScaleSpec;

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
        // Normalize x to [0, 1] within domain. A degenerate domain (d0 ==
        // d1) cannot actually reach this line for a real caller:
        // `QuantizeScale::new` rejects `d0 == d1` up front (see below), this
        // struct's only non-test construction site is inside that
        // constructor after the rejection, and `ScaleSpec::Quantize` never
        // builds a `QuantizeScaleData` in the render pipeline (it routes to
        // the Linear fallback instead). This guard is convention-uniformity
        // on a private struct — defense-in-depth so `QuantizeScaleData`
        // matches every other continuous scale's degenerate-domain contract
        // (GH #104) if a future non-test caller ever constructs one
        // directly — not a defense against a live data path. If it did fire,
        // `degenerate_ratio` resolving to 0.5 means the MIDDLE bin here
        // (rather than 0.5 as a "range midpoint", which is what the guard
        // means for every affine-continuous scale), matching the batch's
        // uniform "center a degenerate domain" semantics, rather than bin 0
        // (which `NaN.clamp(...)` then `as usize`'s saturating-NaN-to-0 cast
        // would otherwise resolve to silently, with no NaN ever escaping —
        // benign, but an unstated, arbitrary bin-0 default).
        let t = degenerate_ratio(x - d0, d1 - d0);
        let t_clamped = t.clamp(0.0, 1.0 - f64::EPSILON);
        let idx = (t_clamped * n as f64).floor() as usize;
        let idx = idx.min(n - 1);
        Some(&self.range[idx])
    }

    /// Return the bin thresholds (n-1 interior break points for n range values).
    ///
    /// Delegates to [`uniform_bin_thresholds`] so this scale and the render-side
    /// discretizing color resolver share one definition of quantize bin
    /// geometry.
    fn thresholds(&self) -> Vec<f64> {
        let [d0, d1] = self.domain;
        uniform_bin_thresholds(d0, d1, self.range.len())
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

impl QuantizeScale {
    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `range` carries color strings (distinct from `Quantile`/`Threshold`'s
    /// numeric range); the constructor guarantees both `domain` (length 2) and
    /// `range` (non-empty).
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Quantize {
            domain: Some(self.0.domain.to_vec()),
            range: if self.0.range.is_empty() {
                None
            } else {
                Some(self.0.range.clone())
            },
        }
    }
}

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
            return Err(PyValueError::new_err(DEGENERATE_DOMAIN_MESSAGE));
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

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

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

    /// #99/#104 residue: a degenerate equal-endpoint domain (`d0 == d1`,
    /// e.g. a constant-valued data column) used to divide by zero
    /// (`0/0 = NaN`) in the `t` ratio. `NaN.clamp(...)` returns `NaN`
    /// unchanged, and `(NaN * n).floor() as usize` saturates to `0` (never
    /// panics, no NaN escapes) — so this was silently, arbitrarily always
    /// bin 0. Under the batch's "center a degenerate domain" convention it
    /// must instead resolve to the MIDDLE bin.
    ///
    /// This exercises `QuantizeScaleData` directly via a struct literal,
    /// which only in-module test code can write — for the contract that
    /// actually holds for every real caller, `QuantizeScale::new` (the
    /// public constructor) must reject a degenerate domain outright, pinned
    /// below alongside the struct-literal case so the test covers both the
    /// defense-in-depth branch and the real-caller-facing rejection.
    #[test]
    fn quantize_scale_degenerate_domain_selects_middle_bin() {
        let three = QuantizeScaleData {
            domain: [5.0, 5.0],
            range: vec!["low".into(), "mid".into(), "high".into()],
        };
        assert_eq!(
            three.scale(5.0),
            Some("mid"),
            "degenerate 3-bin domain must select the middle bin, not bin 0"
        );

        let two = QuantizeScaleData {
            domain: [5.0, 5.0],
            range: vec!["a".into(), "b".into()],
        };
        assert_eq!(
            two.scale(5.0),
            Some("b"),
            "degenerate 2-bin domain must select the upper-middle bin (floor(0.5*2)=1), not bin 0"
        );

        // The contract that actually holds for real callers: the public
        // constructor rejects a degenerate domain before a QuantizeScaleData
        // with d0 == d1 can ever exist outside a test's struct literal.
        assert!(
            QuantizeScale::new(vec![5.0, 5.0], vec!["a".into(), "b".into()]).is_err(),
            "QuantizeScale::new must reject a degenerate domain (lo == hi)"
        );
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
