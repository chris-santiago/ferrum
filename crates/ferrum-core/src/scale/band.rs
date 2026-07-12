use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{scale_spec_to_py_dict, validate_band_point_range};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct BandScaleData {
    domain: Vec<String>,
    padding_inner: f64,
    padding_outer: f64,
    align: f64,
}

impl BandScaleData {
    /// Compute bandwidth and step given a pixel extent.
    ///
    /// `bandwidth` is always non-negative, even when `extent` is negative
    /// (an inverted explicit `range=[hi, lo]`, GH #69): d3's band scale never
    /// reports a negative bandwidth, and downstream `cx - bandwidth/2`
    /// consumers would silently flip sides if it went negative. `step` stays
    /// signed — it drives `scale_str`'s position arithmetic, which must place
    /// bands in descending order for a descending range.
    fn layout(&self, extent: f64) -> (f64, f64) {
        let n = self.domain.len() as f64;
        if n == 0.0 { return (0.0, 0.0); }
        // step = extent / (n + padding_outer * 2 + padding_inner * (n - 1) - padding_inner)
        // Simplified: step = extent / (n - padding_inner + 2 * padding_outer + padding_inner * n - padding_inner)
        // D3 formula: step = extent / max(1, n - paddingInner + paddingOuter * 2)
        let denom = (n - self.padding_inner + self.padding_outer * 2.0).max(1.0);
        let step = extent / denom;
        let bandwidth = (step * (1.0 - self.padding_inner)).abs();
        (bandwidth, step)
    }

    fn scale_str(&self, s: &str, range_lo: f64, range_hi: f64) -> f64 {
        let idx = match self.domain.iter().position(|c| c == s) {
            Some(i) => i,
            None => return f64::NAN,
        };
        let extent = range_hi - range_lo;
        let (_bandwidth, step) = self.layout(extent);
        let start = range_lo + self.padding_outer * step
            + self.align * (extent - (self.domain.len() as f64 - self.padding_inner + self.padding_outer * 2.0) * step).max(0.0);
        start + (idx as f64) * step + step * self.padding_inner / 2.0
    }
}

/// Discrete band scale for bar charts.
///
/// Maps a categorical (string) domain to pixel bands with configurable
/// inner and outer padding. Each category occupies a band of equal width
/// within the range, suitable for bar/column charts.
///
/// Parameters
/// ----------
/// domain : list[str], optional
///     Ordered list of category labels. When ``None``, the renderer derives
///     the domain from data.
/// padding : float, default 0.1
///     Shorthand that sets both ``padding_inner`` and ``padding_outer`` when
///     those are not given explicitly.
/// padding_inner : float, optional
///     Fractional inner padding between bands, in ``[0.0, 1.0)``.
/// padding_outer : float, optional
///     Fractional outer padding before the first and after the last band.
/// align : float, default 0.5
///     Alignment within leftover space, in ``[0.0, 1.0]``.
/// range : list[float], optional
///     Pixel extent ``[lo, hi]``. When ``None``, the renderer fills from
///     the plot-area dimensions.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct BandScale {
    data: BandScaleData,
    range: Option<[f64; 2]>,
}

impl BandScale {
    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// One remaining faithful-reproduction trap from the legacy `_scale_to_dict`:
    /// it emitted `paddingInner`/`paddingOuter`/`align` but **no** top-level
    /// `padding`, so on deserialize `ScaleSpec::Band.padding` took its serde
    /// default (`default_band_padding` = 0.1) regardless of the constructor's
    /// `padding` shorthand. We reproduce that default here.
    ///
    /// The explicit `range` (`BandScale(..., range=[lo, hi])`) IS carried into
    /// the wire form (issue #39 fix, previously silently dropped).
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Band {
            domain: if self.data.domain.is_empty() {
                None
            } else {
                Some(self.data.domain.clone())
            },
            padding: crate::spec::encoding::default_band_padding(),
            padding_inner: Some(self.data.padding_inner),
            padding_outer: Some(self.data.padding_outer),
            align: self.data.align,
            range: self.range.map(|r| r.to_vec()),
        }
    }
}

#[pymethods]
impl BandScale {
    #[new]
    #[pyo3(signature = (*, domain = None, padding = 0.1, padding_inner = None, padding_outer = None, align = 0.5, range = None))]
    fn new(
        domain: Option<Vec<String>>,
        padding: f64,
        padding_inner: Option<f64>,
        padding_outer: Option<f64>,
        align: f64,
        range: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let pi = padding_inner.unwrap_or(padding);
        let po = padding_outer.unwrap_or(padding);
        if !pi.is_finite() || !(0.0..1.0).contains(&pi) {
            return Err(PyValueError::new_err(format!(
                "padding_inner must be in [0, 1); got {pi}"
            )));
        }
        if !po.is_finite() || po < 0.0 {
            return Err(PyValueError::new_err(format!(
                "padding_outer must be >= 0; got {po}"
            )));
        }
        if !align.is_finite() || !(0.0..=1.0).contains(&align) {
            return Err(PyValueError::new_err(format!(
                "align must be in [0, 1]; got {align}"
            )));
        }
        let r = match range {
            Some(v) => {
                validate_band_point_range(&v)?;
                Some([v[0], v[1]])
            }
            None => None,
        };
        Ok(BandScale {
            data: BandScaleData {
                domain: domain.unwrap_or_default(),
                padding_inner: pi,
                padding_outer: po,
                align,
            },
            range: r,
        })
    }

    /// Map a category label to its band-center pixel coordinate.
    ///
    /// Returns ``f64::NAN`` for labels not in the domain.
    fn scale(&self, value: &str) -> f64 {
        let [r0, r1] = self.range.unwrap_or([0.0, 1.0]);
        self.data.scale_str(value, r0, r1)
    }

    /// Compute the bandwidth (bar width) in pixels.
    fn bandwidth(&self) -> f64 {
        let [r0, r1] = self.range.unwrap_or([0.0, 1.0]);
        let (bw, _) = self.data.layout(r1 - r0);
        bw
    }

    /// Return the domain categories in order.
    fn ticks(&self) -> Vec<String> {
        self.data.domain.clone()
    }

    /// Return this scale unchanged (band scales have no numeric "nice" rounding).
    fn nice(&self) -> Self { self.clone() }

    /// Ordered list of category labels.
    #[getter]
    fn domain(&self) -> Vec<String> { self.data.domain.clone() }

    /// Pixel extent of the scale, or ``None`` when auto-derived.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        self.range.map(|r| r.to_vec())
    }

    /// Fractional inner padding between bands.
    #[getter]
    fn padding_inner(&self) -> f64 { self.data.padding_inner }

    /// Fractional outer padding before/after bands.
    #[getter]
    fn padding_outer(&self) -> f64 { self.data.padding_outer }

    /// Alignment within leftover space.
    #[getter]
    fn align(&self) -> f64 { self.data.align }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        format!(
            "BandScale(domain={:?}, padding_inner={}, padding_outer={}, align={})",
            self.data.domain, self.data.padding_inner, self.data.padding_outer, self.data.align
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_scale_basic_layout() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding_inner: 0.0,
            padding_outer: 0.0,
            align: 0.5,
        };
        let (bw, step) = s.layout(300.0);
        assert!((step - 100.0).abs() < 1e-9, "step={step}");
        assert!((bw - 100.0).abs() < 1e-9, "bandwidth={bw}");
    }

    #[test]
    fn band_scale_with_padding() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into()],
            padding_inner: 0.2,
            padding_outer: 0.1,
            align: 0.5,
        };
        let (bw, step) = s.layout(200.0);
        // denom = 2 - 0.2 + 0.1*2 = 2.0, step = 100
        assert!((step - 100.0).abs() < 1e-9, "step={step}");
        // bandwidth = step * (1 - padding_inner) = 100 * 0.8 = 80
        assert!((bw - 80.0).abs() < 1e-9, "bandwidth={bw}");
    }

    #[test]
    fn band_scale_center_positions() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding_inner: 0.0,
            padding_outer: 0.0,
            align: 0.5,
        };
        let ya = s.scale_str("a", 0.0, 300.0);
        let yb = s.scale_str("b", 0.0, 300.0);
        let yc = s.scale_str("c", 0.0, 300.0);
        // With no padding: step=100, centers at 0, 100, 200
        assert!((ya - 0.0).abs() < 1e-9, "ya={ya}");
        assert!((yb - 100.0).abs() < 1e-9, "yb={yb}");
        assert!((yc - 200.0).abs() < 1e-9, "yc={yc}");
    }

    #[test]
    fn band_scale_unknown_returns_nan() {
        let s = BandScaleData {
            domain: vec!["a".into()],
            padding_inner: 0.0,
            padding_outer: 0.0,
            align: 0.5,
        };
        assert!(s.scale_str("z", 0.0, 100.0).is_nan());
    }
}
