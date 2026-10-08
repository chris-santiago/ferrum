use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{scale_spec_to_py_dict, validate_band_point_range};
use super::discrete::{BandPadding, BandPaddingError, DiscreteGeometry, DiscreteLayout};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct BandScaleData {
    domain: Vec<String>,
    /// The padding as the user stated it, for the getters, the wire and repr.
    padding: BandPadding,
    /// `padding` resolved at `align`, once, at construction.
    layout: DiscreteLayout,
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
        self.layout.geometry(self.domain.len(), range_lo, range_hi)
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
/// ``scale()``/``bandwidth()`` here describe the category's *slot*, not what a
/// mark draws inside it. ``scale(v)`` is the slot's *leading* edge, so for an
/// ascending range the slot spans ``[scale(v), scale(v) + bandwidth()]`` and
/// its center sits at ``scale(v) + bandwidth() / 2``. For a descending
/// ``range=[hi, lo]`` the slot runs the other way: subtract instead of adding
/// (the general form is ``scale(v) + bandwidth() / 2 * sign``, with ``sign``
/// the sign of ``range[1] - range[0]``). The inner gap follows each slot, so
/// slots always stay inside the declared range.
///
/// A mark never draws the full slot: each mark family scales `bandwidth()` by
/// its own fraction to get the geometry it actually paints — bar's vertical
/// and horizontal paths (`build_ordinal`, `build_ordinal_y` in
/// `render/marks/bar.rs`) at a fixed 0.8, box/tick's crossbar and rug paths
/// (`build_ordinal_range` in `render/marks/rect.rs`; `build_geometry` and
/// `rug_tick_len` in `render/marks/tick.rs`) at their configurable
/// `band_size` (default 0.6), and heatmap cells (`build_heatmap` in
/// `render/marks/rect.rs`) at 1.0 (they tile the slot with no gap). The scale
/// itself knows nothing about any of these fractions; it only ever reports
/// the slot.
///
/// Parameters
/// ----------
/// domain : list[str], optional
///     Ordered list of category labels. When ``None``, the renderer derives
///     the domain from data.
/// padding : float, optional
///     Shorthand that sets both ``padding_inner`` and ``padding_outer`` when
///     those are not given explicitly, in ``[0.0, 1.0]``.
/// padding_inner : float, optional
///     Fractional inner padding between bands, in ``[0.0, 1.0]``. Narrows each
///     band: ``bandwidth = |step| * (1 - padding_inner)``, with the freed space
///     becoming the gap that follows it. Falls back to ``padding``, then to 0.
/// padding_outer : float, optional
///     Fractional outer padding before the first and after the last band, a
///     finite number ``>= 0``. Falls back to ``padding``, then to 0.
///
///     With no padding argument at all, the bands lay out exactly as an
///     auto-inferred categorical axis does: no inner or outer padding. An
///     explicit ``0.0`` is kept as set, and is distinct from ``None``.
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
    /// Carries the padding exactly as the user set it, so only the padding
    /// keys the user passed reach the wire, and the renderer resolves them by
    /// the same rule this class does.
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
            padding: self.data.padding,
            align: self.data.layout.align(),
            range: self.range.map(|r| r.to_vec()),
        }
    }
}

#[pymethods]
impl BandScale {
    #[new]
    #[pyo3(signature = (*, domain = None, padding = None, padding_inner = None, padding_outer = None, align = 0.5, range = None))]
    fn new(
        domain: Option<Vec<String>>,
        padding: Option<f64>,
        padding_inner: Option<f64>,
        padding_outer: Option<f64>,
        align: f64,
        range: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let refuse = |e: BandPaddingError| {
            PyValueError::new_err(format!("BandScale({}=...) {}", e.key().argument(), e.reason()))
        };
        let padding = BandPadding::new(padding, padding_inner, padding_outer).map_err(refuse)?;
        if !align.is_finite() || !(0.0..=1.0).contains(&align) {
            return Err(PyValueError::new_err(format!(
                "BandScale(align=...) must be a number in [0, 1]; received {align}"
            )));
        }
        let r = match range {
            Some(v) => {
                validate_band_point_range(&v)?;
                Some([v[0], v[1]])
            }
            None => None,
        };
        // `padding` is already validated above (`BandPadding::new`), so this
        // resolves through `BandPadding::resolve` directly rather than
        // `BandPadding::layout`, whose own `self.validate()` would only
        // re-check values this constructor has already checked.
        let layout = padding.resolve(align);
        Ok(BandScale {
            data: BandScaleData { domain: domain.unwrap_or_default(), padding, layout },
            range: r,
        })
    }

    /// Map a category label to the leading pixel edge of its slot.
    ///
    /// Changed in the batch-C scale unification (F-L04-03): for
    /// ``padding_inner > 0`` this returns d3's slot edge, half an inner gap
    /// lower than earlier releases reported (for
    /// ``BandScale(domain=list("abcd"), range=[40, 260], padding=0.1)``,
    /// ``scale("a")`` is 45.366, was 48.049). The old value
    /// placed the slot block so that the last slot ran past ``range[1]``.
    ///
    /// This is d3's ``band(x)``. For an ascending range it is the slot's
    /// low-coordinate edge: the slot spans ``[scale(v), scale(v) + bandwidth()]``
    /// and its center sits at ``scale(v) + bandwidth() / 2``. For a
    /// descending ``range=[hi, lo]`` the slot runs back from this edge
    /// instead, so the center sits at ``scale(v) - bandwidth() / 2``. Returns
    /// ``NaN`` for labels not in the domain. A mark drawn for ``v`` does not
    /// fill this slot — see the struct-level doc above for how each mark
    /// family scales ``bandwidth()`` down to its own drawn fraction.
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

    /// The ``padding`` shorthand as passed, or ``None`` when unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.data.padding.padding() }

    /// Fractional inner padding between bands: ``padding_inner``, else
    /// ``padding``, else ``None`` (no inner padding, as on an auto-inferred
    /// categorical axis).
    #[getter]
    fn padding_inner(&self) -> Option<f64> { self.data.padding.inner() }

    /// Fractional outer padding before/after bands: ``padding_outer``, else
    /// ``padding``, else ``None`` (no outer padding).
    #[getter]
    fn padding_outer(&self) -> Option<f64> { self.data.padding.outer() }

    /// Alignment within leftover space.
    #[getter]
    fn align(&self) -> f64 { self.data.layout.align() }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        let arg = |v: Option<f64>| v.map_or_else(|| "None".to_string(), |v| v.to_string());
        let padding = &self.data.padding;
        format!(
            "BandScale(domain={:?}, padding={}, padding_inner={}, padding_outer={}, align={})",
            self.data.domain,
            arg(padding.padding()),
            arg(padding.inner_as_set()),
            arg(padding.outer_as_set()),
            self.data.layout.align()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scale::ordinal::OrdinalScale;

    /// A facade with both sides set explicitly, resolved through
    /// [`BandPadding`] as the constructor resolves them.
    fn data(domain: Vec<String>, padding_inner: f64, padding_outer: f64, align: f64) -> BandScaleData {
        let padding = BandPadding::new(None, Some(padding_inner), Some(padding_outer))
            .expect("test padding is in bounds");
        let layout = padding.layout(align).expect("test padding is in bounds");
        BandScaleData { domain, padding, layout }
    }

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
                    data: data(domain.clone(), pi, po, align),
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
        let s = data(vec!["a".into(), "b".into(), "c".into()], 0.0, 0.0, 0.5);
        let g = s.geometry(0.0, 300.0);
        assert!((g.step() - 100.0).abs() < 1e-9, "step={}", g.step());
        assert!((g.bandwidth() - 100.0).abs() < 1e-9, "bandwidth={}", g.bandwidth());
    }

    #[test]
    fn band_scale_with_padding() {
        let s = data(vec!["a".into(), "b".into()], 0.2, 0.1, 0.5);
        let g = s.geometry(0.0, 200.0);
        // denom = 2 - 0.2 + 0.1*2 = 2.0, step = 100
        assert!((g.step() - 100.0).abs() < 1e-9, "step={}", g.step());
        // bandwidth = step * (1 - padding_inner) = 100 * 0.8 = 80
        assert!((g.bandwidth() - 80.0).abs() < 1e-9, "bandwidth={}", g.bandwidth());
    }

    #[test]
    fn band_scale_lead_positions() {
        let s = data(vec!["a".into(), "b".into(), "c".into()], 0.0, 0.0, 0.5);
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
        let s = data(vec!["a".into()], 0.0, 0.0, 0.5);
        assert!(s.scale_str("z", 0.0, 100.0).is_nan());
    }

    // ── denominator clamp, degenerate domains, sign (ported from
    // tests/bug_hunt_band_point_range.rs, R1) ────────────────────────────────

    /// The d3 denominator clamp: n=1, padding_inner=0.9, padding_outer=0 gives
    /// n - pi + 2*po = 0.1, clamped to 1.0 → step = extent, bandwidth = extent
    /// * (1 - 0.9). Without the clamp step would be 10x the extent.
    #[test]
    fn band_denominator_clamps_below_one() {
        let s = data(vec!["a".into()], 0.9, 0.0, 0.5);
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
        let s = data(Vec::new(), 0.1, 0.1, 0.5);
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
        let s = data(vec!["a".into(), "b".into()], 0.0, 0.0, 0.5);
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
        let mk = |align: f64| data(vec!["a".into()], 0.5, 0.0, align);
        let p0 = mk(0.0).scale_str("a", 0.0, 100.0);
        let p1 = mk(1.0).scale_str("a", 0.0, 100.0);
        assert!((p0 - 0.0).abs() < 1e-9, "align=0 lead; got {p0}");
        assert!((p1 - 50.0).abs() < 1e-9, "align=1 lead must shift by the leftover; got {p1}");
    }

    // ── unset padding, the constructor door and the wire (#157) ─────────────

    fn abcd() -> Option<Vec<String>> {
        Some(["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect())
    }

    fn band_scale(
        padding: Option<f64>,
        padding_inner: Option<f64>,
        padding_outer: Option<f64>,
        range: Option<Vec<f64>>,
    ) -> BandScale {
        BandScale::new(abcd(), padding, padding_inner, padding_outer, 0.5, range).expect("valid BandScale")
    }

    fn wire(scale: &BandScale) -> serde_json::Value {
        serde_json::to_value(scale.to_scale_spec()).unwrap()
    }

    /// `BandScale()` lays out as the auto-inferred categorical axis:
    /// `bandwidth()` over `[0, 400]` is 100 (it was 87.805 under the hidden
    /// `padding = 0.1`), and the leading edges sit on the unpadded slots.
    #[test]
    fn unset_band_scale_bandwidth_matches_the_auto_inferred_layout() {
        let s = band_scale(None, None, None, Some(vec![0.0, 400.0]));
        assert_eq!(s.bandwidth(), 100.0);
        assert_eq!(["a", "b", "c", "d"].map(|c| s.scale(c)), [0.0, 100.0, 200.0, 300.0]);
        assert_eq!(s.data.layout, DiscreteLayout::UNPADDED);
        // An explicit `padding` keeps d3's math: 400 / 4.1 · 0.9.
        let padded = band_scale(Some(0.1), None, None, Some(vec![0.0, 400.0]));
        assert!((padded.bandwidth() - 87.805).abs() < 1e-3, "{}", padded.bandwidth());
    }

    /// The facade resolves exactly the layout its own wire form resolves to:
    /// a `to_scale_spec` round trip through JSON reproduces the same
    /// `BandPadding`, so facade and renderer read one value.
    #[test]
    fn facade_layout_is_the_layout_its_wire_form_resolves_to() {
        for (p, pi, po) in [
            (None, None, None),
            (Some(0.1), None, None),
            (Some(0.0), None, None),
            (None, Some(0.2), None),
            (None, None, Some(0.3)),
            (Some(0.4), Some(0.2), None),
            (None, Some(0.2), Some(1.5)),
        ] {
            let s = band_scale(p, pi, po, None);
            let json = serde_json::to_string(&s.to_scale_spec()).unwrap();
            let ScaleSpec::Band { padding, align, .. } = serde_json::from_str(&json).unwrap() else {
                panic!("expected a band spec: {json}")
            };
            assert_eq!(padding, s.data.padding, "{json}");
            assert_eq!(padding.layout(align).unwrap(), s.data.layout, "{json}");
        }
    }

    /// `to_scale_spec` emits only the padding keys the user passed.
    #[test]
    fn wire_carries_only_user_set_padding_keys() {
        let keys = |s: &BandScale| {
            let mut k: Vec<String> = wire(s).as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(keys(&band_scale(None, None, None, None)), ["align", "domain", "type"]);
        assert_eq!(wire(&band_scale(Some(0.0), None, None, None))["padding"], serde_json::json!(0.0));
        assert_eq!(keys(&band_scale(None, Some(0.2), None, None)), ["align", "domain", "paddingInner", "type"]);
        assert_eq!(
            keys(&band_scale(Some(0.1), None, Some(0.3), None)),
            ["align", "domain", "padding", "paddingOuter", "type"]
        );
    }

    /// The getters report the user's arguments: `padding` as passed,
    /// `padding_inner`/`padding_outer` as `side ?? padding`, `None` when unset.
    #[test]
    fn getters_report_user_values_or_none() {
        let get = |s: &BandScale| (s.padding(), s.padding_inner(), s.padding_outer());
        assert_eq!(get(&band_scale(None, None, None, None)), (None, None, None));
        assert_eq!(get(&band_scale(Some(0.1), None, None, None)), (Some(0.1), Some(0.1), Some(0.1)));
        assert_eq!(get(&band_scale(Some(0.0), None, None, None)), (Some(0.0), Some(0.0), Some(0.0)));
        assert_eq!(get(&band_scale(None, Some(0.2), None, None)), (None, Some(0.2), None));
        assert_eq!(get(&band_scale(None, None, Some(0.3), None)), (None, None, Some(0.3)));
        assert_eq!(get(&band_scale(Some(0.4), Some(0.2), None, None)), (Some(0.4), Some(0.2), Some(0.4)));
    }

    /// Equality and repr reflect what the user set: `BandScale()` equals
    /// `BandScale(padding=None)` and differs from `BandScale(padding=0.0)`.
    #[test]
    fn equality_and_repr_reflect_user_set_values() {
        let unset = band_scale(None, None, None, None);
        let zero = band_scale(Some(0.0), None, None, None);
        assert_eq!(unset, band_scale(None, None, None, None));
        assert_ne!(unset, zero);
        assert_eq!(
            unset.__repr__(),
            r#"BandScale(domain=["a", "b", "c", "d"], padding=None, padding_inner=None, padding_outer=None, align=0.5)"#
        );
        assert_eq!(
            band_scale(Some(0.4), Some(0.2), None, None).__repr__(),
            r#"BandScale(domain=["a", "b", "c", "d"], padding=0.4, padding_inner=0.2, padding_outer=None, align=0.5)"#
        );
        assert_ne!(unset.__repr__(), zero.__repr__());
    }

    /// The constructor refuses an out-of-bounds padding naming `BandScale`
    /// and the argument passed, including a `padding` that explicit inner
    /// and outer values mask; d3's open-ended outer padding is accepted.
    #[test]
    fn constructor_refuses_out_of_bounds_padding_naming_the_argument() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let refusal = |p: Option<f64>, pi: Option<f64>, po: Option<f64>| {
                BandScale::new(abcd(), p, pi, po, 0.5, None)
                    .expect_err("out-of-bounds padding")
                    .value(py)
                    .to_string()
            };
            assert_eq!(refusal(Some(1.5), None, None), "BandScale(padding=...) must be a number in [0, 1]; received 1.5");
            assert_eq!(
                refusal(None, Some(f64::NAN), None),
                "BandScale(padding_inner=...) must be a number in [0, 1]; received NaN"
            );
            assert_eq!(
                refusal(None, Some(f64::INFINITY), None),
                "BandScale(padding_inner=...) must be a number in [0, 1]; received inf"
            );
            assert_eq!(
                refusal(None, None, Some(-0.5)),
                "BandScale(padding_outer=...) must be a finite number >= 0; received -0.5"
            );
            assert_eq!(
                refusal(Some(-1.0), Some(0.2), Some(0.3)),
                "BandScale(padding=...) must be a number in [0, 1]; received -1"
            );
            let align_refusal = |align: f64| {
                BandScale::new(abcd(), None, None, None, align, None)
                    .expect_err("out-of-bounds align")
                    .value(py)
                    .to_string()
            };
            assert_eq!(align_refusal(1.5), "BandScale(align=...) must be a number in [0, 1]; received 1.5");
            assert_eq!(align_refusal(f64::NAN), "BandScale(align=...) must be a number in [0, 1]; received NaN");
        });
        assert!(BandScale::new(abcd(), None, Some(1.0), Some(3.0), 0.5, None).is_ok());
    }
}
