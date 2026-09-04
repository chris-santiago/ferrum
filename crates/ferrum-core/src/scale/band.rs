use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{scale_spec_to_py_dict, validate_band_point_range};
use super::discrete::{DiscreteGeometry, DiscreteLayout};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct BandScaleData {
    domain: Vec<String>,
    padding_inner: f64,
    padding_outer: f64,
    align: f64,
}

impl BandScaleData {
    /// Pixel geometry of this band scale over `[range_lo, range_hi]`.
    ///
    /// The band formulas themselves live in [`crate::scale::discrete`] (which
    /// documents where the model departs from upstream d3), and the render-side
    /// `OrdinalScale` calls them too — this facade and the renderer compute band
    /// geometry from one implementation (F-L04-03). Unification was not
    /// output-preserving for this facade: its own placement inset each band
    /// half an inner gap into its slot, which let the last band run past
    /// `range_hi`, so the shared model uses upstream d3's placement and
    /// `scale()` now returns a different pixel for `padding_inner > 0`.
    ///
    /// `bandwidth` is always non-negative, even when the extent is negative
    /// (an inverted explicit `range=[hi, lo]`, GH #69): d3's band scale never
    /// reports a negative bandwidth, and downstream `cx - bandwidth/2`
    /// consumers would silently flip sides if it went negative. `step` stays
    /// signed — it drives the position arithmetic, which must place bands in
    /// descending order for a descending range.
    fn geometry(&self, range_lo: f64, range_hi: f64) -> DiscreteGeometry {
        DiscreteLayout::band(self.padding_inner, self.padding_outer, self.align).geometry(
            self.domain.len(),
            range_lo,
            range_hi,
        )
    }

    /// The leading edge of `s`'s band (d3's `band(x)`), not its middle. For an
    /// inverted range the step is negative, so this is the band's
    /// high-coordinate edge.
    fn scale_str(&self, s: &str, range_lo: f64, range_hi: f64) -> f64 {
        let idx = match self.domain.iter().position(|c| c == s) {
            Some(i) => i,
            None => return f64::NAN,
        };
        self.geometry(range_lo, range_hi).lead(idx)
    }
}

/// Discrete band scale for bar charts.
///
/// Maps a categorical (string) domain to pixel bands with configurable
/// inner and outer padding. Each category occupies a band of equal width
/// within the range, suitable for bar/column charts.
///
/// Every parameter below moves rendered geometry: a ``BandScale`` passed to an
/// encoding resolves through the same band model this class computes with, so
/// ``scale()``/``bandwidth()`` here describe what the chart draws. ``scale(v)``
/// is the band's *leading* edge, so for an ascending range the band spans
/// ``[scale(v), scale(v) + bandwidth()]`` and a mark for ``v`` is drawn at
/// ``scale(v) + bandwidth() / 2``. For a descending ``range=[hi, lo]`` the
/// bands run the other way: subtract instead of adding (the general form is
/// ``scale(v) + bandwidth() / 2 * sign``, with ``sign`` the sign of
/// ``range[1] - range[0]``). The inner gap follows each band, so the bands
/// always stay inside the declared range.
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
///     Fractional inner padding between bands, in ``[0.0, 1.0)``. Narrows each
///     band: ``bandwidth = |step| * (1 - padding_inner)``, with the freed space
///     becoming the gap that follows it.
/// padding_outer : float, optional
///     Fractional outer padding before the first and after the last band.
/// align : float, default 0.5
///     Where the bands sit within any *leftover* space, in ``[0.0, 1.0]``
///     (0 = against the low end of the range, 1 = against the high end).
///
///     Leftover exists only when the band denominator
///     ``n - padding_inner + 2 * padding_outer`` falls below 1 and is clamped
///     to 1: in practice, a single-category domain with a large
///     ``padding_inner``. Otherwise the bands fill the range exactly and
///     ``align`` moves nothing, whatever value it takes. Example: one category
///     over ``range=[0, 100]`` with ``padding_inner=0.5`` gives a 50px band and
///     50px of leftover, so the band's leading edge sits at 0 (``align=0``),
///     25 (``align=0.5``) or 50 (``align=1``, flush against the high end).
///
///     An inverted ``range=[hi, lo]`` has no leftover to distribute in the
///     direction ``align`` names, so ``align`` is inert there. d3 also routes
///     *outer* padding through ``align``; this scale applies outer padding
///     directly, so the two agree whenever ``align=0.5`` or
///     ``padding_outer=0`` and differ only for a non-default ``align``
///     combined with outer padding.
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

    /// Map a category label to the leading pixel edge of its band.
    ///
    /// Changed in the batch-C scale unification (F-L04-03): for
    /// ``padding_inner > 0`` this returns d3's band edge, half an inner gap
    /// lower than earlier releases reported (for
    /// ``BandScale(domain=list("abcd"), range=[40, 260])`` — default
    /// ``padding=0.1`` — ``scale("a")`` is 45.366, was 48.049). The old value
    /// placed the band block so that the last band ran past ``range[1]``.
    ///
    /// This is d3's ``band(x)``. For an ascending range it is the band's
    /// low-coordinate edge: the band spans ``[scale(v), scale(v) + bandwidth()]``
    /// and the pixel a mark for ``v`` is drawn at is
    /// ``scale(v) + bandwidth() / 2``. For a descending ``range=[hi, lo]`` the
    /// band runs back from this edge instead, so the mark sits at
    /// ``scale(v) - bandwidth() / 2``. Returns ``f64::NAN`` for labels not in
    /// the domain.
    fn scale(&self, value: &str) -> f64 {
        let [r0, r1] = self.range.unwrap_or([0.0, 1.0]);
        self.data.scale_str(value, r0, r1)
    }

    /// Compute the bandwidth (bar width) in pixels.
    fn bandwidth(&self) -> f64 {
        let [r0, r1] = self.range.unwrap_or([0.0, 1.0]);
        self.data.geometry(r0, r1).bandwidth()
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
    use crate::scale::ordinal::OrdinalScale;

    // ── facade ↔ render seam oracle (spec §10, F-L04-03) ────────────────────

    /// The sharpest seam-check this change has: for identical parameters the
    /// compute facade and the render-side scale must describe the SAME
    /// geometry. `BandScale.bandwidth()` must equal `OrdinalScale::bandwidth()`,
    /// and the facade's band — `[scale(v), scale(v) + bandwidth()]` — must be
    /// centered on the pixel the renderer places the mark at.
    ///
    /// This is what F-L04-03 broke: the facade computed the d3 model while the
    /// renderer computed a symmetric one in which padding moved nothing, so the
    /// two disagreed for every padded scale. A single shared implementation
    /// (`crate::scale::discrete`) is what makes this test pass; re-forking
    /// either side fails it.
    #[test]
    fn facade_and_render_scale_agree_on_band_geometry() {
        let cats = ["a", "b", "c", "d"];
        let domain: Vec<String> = cats.iter().map(|s| s.to_string()).collect();
        for (lo, hi) in [(40.0, 260.0), (0.0, 500.0), (260.0, 40.0)] {
            for (pi, po, align) in [
                (0.0, 0.0, 0.5),
                (0.1, 0.1, 0.5),
                (0.5, 0.0, 0.0),
                (0.2, 0.35, 1.0),
                (0.0, 0.5, 0.5),
            ] {
                let facade = BandScale {
                    data: BandScaleData {
                        domain: domain.clone(),
                        padding_inner: pi,
                        padding_outer: po,
                        align,
                    },
                    range: Some([lo, hi]),
                };
                let rendered = OrdinalScale::new_internal(
                    domain.clone(),
                    vec![lo, hi],
                    DiscreteLayout::band(pi, po, align),
                );
                let ctx = format!("range=[{lo}, {hi}] pi={pi} po={po} align={align}");
                assert_eq!(
                    facade.bandwidth(),
                    rendered.bandwidth(),
                    "bandwidth disagreement ({ctx})"
                );
                // The facade reports the band's leading edge; the renderer
                // places marks at its center. `step` carries the sign, so a
                // descending range walks the half-band the other way.
                let half = facade.bandwidth() / 2.0 * facade.data.geometry(lo, hi).step().signum();
                for cat in cats {
                    let center = rendered.scale_internal(cat).expect("category in domain");
                    assert!(
                        (facade.scale(cat) + half - center).abs() < 1e-9,
                        "center disagreement for {cat} ({ctx}): facade lead {} + half {half} vs render {center}",
                        facade.scale(cat)
                    );
                }
            }
        }
    }

    #[test]
    fn band_scale_basic_layout() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding_inner: 0.0,
            padding_outer: 0.0,
            align: 0.5,
        };
        let g = s.geometry(0.0, 300.0);
        assert!((g.step() - 100.0).abs() < 1e-9, "step={}", g.step());
        assert!((g.bandwidth() - 100.0).abs() < 1e-9, "bandwidth={}", g.bandwidth());
    }

    #[test]
    fn band_scale_with_padding() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into()],
            padding_inner: 0.2,
            padding_outer: 0.1,
            align: 0.5,
        };
        let g = s.geometry(0.0, 200.0);
        // denom = 2 - 0.2 + 0.1*2 = 2.0, step = 100
        assert!((g.step() - 100.0).abs() < 1e-9, "step={}", g.step());
        // bandwidth = step * (1 - padding_inner) = 100 * 0.8 = 80
        assert!((g.bandwidth() - 80.0).abs() < 1e-9, "bandwidth={}", g.bandwidth());
    }

    #[test]
    fn band_scale_lead_positions() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into(), "c".into()],
            padding_inner: 0.0,
            padding_outer: 0.0,
            align: 0.5,
        };
        let ya = s.scale_str("a", 0.0, 300.0);
        let yb = s.scale_str("b", 0.0, 300.0);
        let yc = s.scale_str("c", 0.0, 300.0);
        // With no padding: step=100, band leading edges at 0, 100, 200
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

    // ── denominator clamp, degenerate domains, sign (ported from
    // tests/bug_hunt_band_point_range.rs, R1) ────────────────────────────────

    /// The d3 denominator clamp: n=1, padding_inner=0.9, padding_outer=0 gives
    /// n - pi + 2*po = 0.1, clamped to 1.0 → step = extent, bandwidth = extent
    /// * (1 - 0.9). Without the clamp step would be 10x the extent.
    #[test]
    fn band_denominator_clamps_below_one() {
        let s = BandScaleData {
            domain: vec!["a".into()],
            padding_inner: 0.9,
            padding_outer: 0.0,
            align: 0.5,
        };
        let g = s.geometry(0.0, 200.0);
        assert!((g.step() - 200.0).abs() < 1e-9, "denominator must clamp to 1.0; step={}", g.step());
        assert!(
            (g.bandwidth() - 20.0).abs() < 1e-9,
            "bandwidth = extent * (1 - pi); got {}",
            g.bandwidth()
        );
    }

    /// Empty domain: the geometry is inert (zero step, zero bandwidth) and
    /// `scale_str` returns NaN — no division by the n==0 denominator.
    #[test]
    fn band_empty_domain_layout_zero_and_nan_lookup() {
        let s = BandScaleData {
            domain: Vec::new(),
            padding_inner: 0.1,
            padding_outer: 0.1,
            align: 0.5,
        };
        let g = s.geometry(0.0, 300.0);
        assert_eq!((g.bandwidth(), g.step()), (0.0, 0.0));
        assert!(s.scale_str("a", 0.0, 300.0).is_nan());
    }

    /// Regression test (GH #69): `BandScaleData::layout` used to return a
    /// NEGATIVE bandwidth for an inverted range (extent < 0 → step < 0 →
    /// bandwidth = step * (1 - pi) < 0). The pyclass getter
    /// `BandScale::bandwidth()` shipped that sign to Python; d3 never reports
    /// a negative bandwidth and `cx - bandwidth/2` consumers would silently
    /// flip sides. Fixed by taking `.abs()` of the bandwidth (not the signed
    /// `step`, which still drives `scale_str`'s descending-position
    /// arithmetic).
    #[test]
    fn band_bandwidth_non_negative_for_inverted_range() {
        let s = BandScaleData {
            domain: vec!["a".into(), "b".into()],
            padding_inner: 0.0,
            padding_outer: 0.0,
            align: 0.5,
        };
        let bw = s.geometry(260.0, 40.0).bandwidth();
        // n=2, pi=po=0 → denom=2, step = -220/2 = -110, bandwidth = |step| = 110.0.
        assert!((bw - 110.0).abs() < 1e-9, "bandwidth must be |step| = 110.0 for an inverted range; got {bw}");
        assert!(bw >= 0.0, "bandwidth must be non-negative for an inverted range; got {bw}");
    }

    /// align leftover activation: the ONLY reachable leftover > 0 case is the
    /// denominator clamp (denom_raw < 1). n=1, pi=0.5, po=0 over [0, 100]:
    /// step = 100, leftover = 100 - 0.5*100 = 50. `scale()` is the band's
    /// leading edge, so align=0 puts it at 0 (band [0, 50]) and align=1 shifts
    /// it by the full leftover to 50 (band [50, 100]).
    #[test]
    fn band_align_shifts_within_clamped_leftover() {
        let mk = |align: f64| BandScaleData {
            domain: vec!["a".into()],
            padding_inner: 0.5,
            padding_outer: 0.0,
            align,
        };
        let p0 = mk(0.0).scale_str("a", 0.0, 100.0);
        let p1 = mk(1.0).scale_str("a", 0.0, 100.0);
        assert!((p0 - 0.0).abs() < 1e-9, "align=0 lead; got {p0}");
        assert!((p1 - 50.0).abs() < 1e-9, "align=1 lead must shift by the leftover; got {p1}");
    }
}
