use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::scale_spec_to_py_dict;
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct PointScaleData {
    domain: Vec<String>,
    padding: f64,
    align: f64,
    reverse: bool,
}

impl PointScaleData {
    fn scale_str(&self, s: &str, range_lo: f64, range_hi: f64) -> f64 {
        let idx = match self.domain.iter().position(|c| c == s) {
            Some(i) => i,
            None => return f64::NAN,
        };
        let n = self.domain.len();
        if n <= 1 {
            // Single category: place at center of extent
            let center = (range_lo + range_hi) / 2.0;
            return center;
        }
        let extent = range_hi - range_lo;
        // A point scale is essentially a band scale with bandwidth=0.
        // step = extent / (n - 1 + padding * 2)
        let denom = (n as f64 - 1.0) + self.padding * 2.0;
        let step = extent / denom;
        let start = range_lo + self.padding * step
            + self.align * (extent - denom * step).max(0.0);
        let pos = start + (idx as f64) * step;
        if self.reverse {
            range_hi - (pos - range_lo)
        } else {
            pos
        }
    }
}

/// Discrete point scale for dot plots.
///
/// Maps a categorical (string) domain to evenly-spaced point positions
/// (zero bandwidth). Similar to a band scale with bandwidth=0. Useful
/// for dot plots, strip plots, and Cleveland-style charts.
///
/// Parameters
/// ----------
/// domain : list[str], optional
///     Ordered list of category labels. When ``None``, the renderer derives
///     the domain from data.
/// padding : float, default 0.5
///     Outer padding expressed as a fraction of step size.
/// align : float, default 0.5
///     Alignment within leftover space, in ``[0.0, 1.0]``.
/// reverse : bool, default False
///     Reverse the category order within the range.
/// range : list[float], optional
///     Pixel extent ``[lo, hi]``. When ``None``, the renderer fills from
///     the plot-area dimensions.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct PointScale {
    data: PointScaleData,
    range: Option<[f64; 2]>,
}

impl PointScale {
    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// The explicit `range` (`PointScale(..., range=[lo, hi])`) IS carried into
    /// the wire form (issue #39 fix, previously silently dropped by the legacy
    /// `_scale_to_dict` deserialiser).
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Point {
            domain: if self.data.domain.is_empty() {
                None
            } else {
                Some(self.data.domain.clone())
            },
            padding: self.data.padding,
            align: self.data.align,
            reverse: self.data.reverse,
            range: self.range.map(|r| r.to_vec()),
        }
    }
}

#[pymethods]
impl PointScale {
    #[new]
    #[pyo3(signature = (*, domain = None, padding = 0.5, align = 0.5, reverse = false, range = None))]
    fn new(
        domain: Option<Vec<String>>,
        padding: f64,
        align: f64,
        reverse: bool,
        range: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if !padding.is_finite() || padding < 0.0 {
            return Err(PyValueError::new_err(format!(
                "padding must be >= 0; got {padding}"
            )));
        }
        if !align.is_finite() || !(0.0..=1.0).contains(&align) {
            return Err(PyValueError::new_err(format!(
                "align must be in [0, 1]; got {align}"
            )));
        }
        let r = range.map(|v| {
            if v.len() >= 2 { [v[0], v[1]] } else { [0.0, 1.0] }
        });
        Ok(PointScale {
            data: PointScaleData {
                domain: domain.unwrap_or_default(),
                padding,
                align,
                reverse,
            },
            range: r,
        })
    }

    /// Map a category label to its point pixel coordinate.
    ///
    /// Returns ``f64::NAN`` for labels not in the domain.
    fn scale(&self, value: &str) -> f64 {
        let [r0, r1] = self.range.unwrap_or([0.0, 1.0]);
        self.data.scale_str(value, r0, r1)
    }

    /// Return the domain categories in order.
    fn ticks(&self) -> Vec<String> {
        self.data.domain.clone()
    }

    /// Return this scale unchanged (point scales have no numeric "nice" rounding).
    fn nice(&self) -> Self { self.clone() }

    /// Ordered list of category labels.
    #[getter]
    fn domain(&self) -> Vec<String> { self.data.domain.clone() }

    /// Pixel extent of the scale, or ``None`` when auto-derived.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        self.range.map(|r| r.to_vec())
    }

    /// Outer padding as a fraction of step size.
    #[getter]
    fn padding(&self) -> f64 { self.data.padding }

    /// Alignment within leftover space.
    #[getter]
    fn align(&self) -> f64 { self.data.align }

    /// Whether category order is reversed.
    #[getter]
    fn reverse(&self) -> bool { self.data.reverse }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        format!(
            "PointScale(domain={:?}, padding={}, align={}, reverse={})",
            self.data.domain, self.data.padding, self.data.align,
            if self.data.reverse { "True" } else { "False" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_scale_basic_positions() {
        let s = PointScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding: 0.0,
            align: 0.5,
            reverse: false,
        };
        // n=3, padding=0: step = 300 / (3-1+0) = 150
        let ya = s.scale_str("a", 0.0, 300.0);
        let yb = s.scale_str("b", 0.0, 300.0);
        let yc = s.scale_str("c", 0.0, 300.0);
        assert!((ya - 0.0).abs() < 1e-9, "ya={ya}");
        assert!((yb - 150.0).abs() < 1e-9, "yb={yb}");
        assert!((yc - 300.0).abs() < 1e-9, "yc={yc}");
    }

    #[test]
    fn point_scale_with_padding() {
        let s = PointScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding: 0.5,
            align: 0.5,
            reverse: false,
        };
        // n=3, padding=0.5: denom = 2 + 1.0 = 3.0, step = 300/3 = 100
        // start = 0 + 0.5*100 = 50
        let ya = s.scale_str("a", 0.0, 300.0);
        let yb = s.scale_str("b", 0.0, 300.0);
        let yc = s.scale_str("c", 0.0, 300.0);
        assert!((ya - 50.0).abs() < 1e-9, "ya={ya}");
        assert!((yb - 150.0).abs() < 1e-9, "yb={yb}");
        assert!((yc - 250.0).abs() < 1e-9, "yc={yc}");
    }

    #[test]
    fn point_scale_reverse() {
        let s = PointScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding: 0.0,
            align: 0.5,
            reverse: true,
        };
        let ya = s.scale_str("a", 0.0, 300.0);
        let yc = s.scale_str("c", 0.0, 300.0);
        // reversed: "a" at 300, "c" at 0
        assert!((ya - 300.0).abs() < 1e-9, "ya={ya}");
        assert!((yc - 0.0).abs() < 1e-9, "yc={yc}");
    }

    #[test]
    fn point_scale_single_category() {
        let s = PointScaleData {
            domain: vec!["x".into()],
            padding: 0.5,
            align: 0.5,
            reverse: false,
        };
        let y = s.scale_str("x", 0.0, 100.0);
        assert!((y - 50.0).abs() < 1e-9, "single category at center: y={y}");
    }

    #[test]
    fn point_scale_unknown_returns_nan() {
        let s = PointScaleData {
            domain: vec!["a".into()],
            padding: 0.0,
            align: 0.5,
            reverse: false,
        };
        assert!(s.scale_str("z", 0.0, 100.0).is_nan());
    }
}
