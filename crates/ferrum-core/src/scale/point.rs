use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{scale_spec_to_py_dict, validate_band_point_range};
use super::discrete::DiscreteLayout;
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct PointScaleData {
    domain: Vec<String>,
    padding: f64,
    align: f64,
    reverse: bool,
}

impl PointScaleData {
    /// Map a category to its point pixel, `reverse` included.
    ///
    /// The point formulas live in [`crate::scale::discrete`], which the
    /// render-side `OrdinalScale` also calls (F-L04-03). `reverse` stays here:
    /// it is a facade-level mirroring of the resolved positions, whereas the
    /// renderer implements `PointScale(reverse=True)` by reversing the resolved
    /// domain (GH #65) so axis ticks follow the marks.
    fn scale_str(&self, s: &str, range_lo: f64, range_hi: f64) -> f64 {
        let idx = match self.domain.iter().position(|c| c == s) {
            Some(i) => i,
            None => return f64::NAN,
        };
        let pos = DiscreteLayout::point(self.padding, self.align)
            .geometry(self.domain.len(), range_lo, range_hi)
            .position(idx);
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
/// A ``PointScale`` passed to an encoding resolves through the same point
/// model this class computes with, so ``scale()`` here gives the pixel a mark
/// for that category is drawn at.
///
/// Parameters
/// ----------
/// domain : list[str], optional
///     Ordered list of category labels. When ``None``, the renderer derives
///     the domain from data.
/// padding : float, default 0.5
///     Outer padding expressed as a fraction of step size:
///     ``step = extent / (n - 1 + 2 * padding)``. ``0.0`` puts the first and
///     last categories exactly on the range endpoints; the default ``0.5``
///     holds half a step at each end, which places the points on the same
///     pixels an unpadded band scale centers its bands at.
/// align : float, default 0.5
///     Where the points sit within any *leftover* space, in ``[0.0, 1.0]``.
///
///     A point scale never has leftover: its padded positions fill the range
///     exactly for any ``padding``, so ``align`` is algebraically inert here:
///     it is accepted and validated, but no value of it moves a point. Use
///     ``padding`` to control the space at the ends. (d3 reaches the same
///     positions by distributing that end padding through ``align``, where a
///     non-default ``align`` does shift them; the two agree at the default
///     ``align=0.5``.)
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
        let r = match range {
            Some(v) => {
                validate_band_point_range(&v)?;
                Some([v[0], v[1]])
            }
            None => None,
        };
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
    /// Returns ``NaN`` for labels not in the domain.
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
    use crate::scale::discrete::DiscreteLayout;
    use crate::scale::ordinal::OrdinalScale;

    // ── facade ↔ render seam oracle (spec §10, F-L04-03) ────────────────────

    /// The point half of the facade/render seam check: for identical
    /// parameters, `PointScale.scale()` must equal the pixel the renderer
    /// places the mark at. Before F-L04-03 the renderer put point scales on
    /// band centers (`(i + 0.5)·step`) and ignored `padding` entirely, so the
    /// two agreed only by the `padding = 0.5` coincidence; the sweep below
    /// includes paddings on both sides of it.
    ///
    /// `reverse` is deliberately excluded: the facade mirrors positions while
    /// the resolver reverses the domain vector (GH #65). Those are equivalent
    /// transforms of the same base positions, which is what this pins.
    #[test]
    fn facade_and_render_scale_agree_on_point_positions() {
        let cats = ["a", "b", "c", "d"];
        let domain: Vec<String> = cats.iter().map(|s| s.to_string()).collect();
        for (lo, hi) in [(40.0, 260.0), (0.0, 500.0), (260.0, 40.0)] {
            for (padding, align) in [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (0.25, 0.0), (0.75, 1.0)] {
                let facade = PointScale {
                    data: PointScaleData {
                        domain: domain.clone(),
                        padding,
                        align,
                        reverse: false,
                    },
                    range: Some([lo, hi]),
                };
                let rendered = OrdinalScale::new_internal(
                    domain.clone(),
                    vec![lo, hi],
                    DiscreteLayout::point(padding, align),
                );
                for cat in cats {
                    let center = rendered.scale_internal(cat).expect("category in domain");
                    assert!(
                        (facade.scale(cat) - center).abs() < 1e-9,
                        "position disagreement for {cat} (range=[{lo}, {hi}] padding={padding} \
                         align={align}): facade {} vs render {center}",
                        facade.scale(cat)
                    );
                }
            }
        }
    }

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

    // ── reverse (GH #65) composition corners (ported from
    // tests/bug_hunt_band_point_range.rs, R1) ────────────────────────────────

    fn point(domain: &[&str], padding: f64, align: f64, reverse: bool) -> PointScaleData {
        PointScaleData {
            domain: domain.iter().map(|s| s.to_string()).collect(),
            padding,
            align,
            reverse,
        }
    }

    /// `reverse=true` over an INVERTED range is a double reversal: every
    /// category lands exactly where the forward scale over the forward range
    /// puts it.
    #[test]
    fn point_reverse_composed_with_inverted_range_equals_forward() {
        let fwd = point(&["a", "b", "c"], 0.0, 0.5, false);
        let dbl = point(&["a", "b", "c"], 0.0, 0.5, true);
        for cat in ["a", "b", "c"] {
            let forward = fwd.scale_str(cat, 0.0, 300.0);
            let double = dbl.scale_str(cat, 300.0, 0.0);
            assert!(
                (forward - double).abs() < 1e-9,
                "reverse over [300,0] must equal forward over [0,300] for {cat}: {forward} vs {double}"
            );
        }
    }

    /// Single category ignores reverse, padding, and align: always the range
    /// midpoint (the `n <= 1` early return), across a sweep of combinations.
    #[test]
    fn point_single_category_ignores_reverse_padding_align() {
        for reverse in [false, true] {
            for padding in [0.0, 0.5, 10.0] {
                for align in [0.0, 0.5, 1.0] {
                    let s = point(&["x"], padding, align, reverse);
                    let p = s.scale_str("x", 40.0, 260.0);
                    assert_eq!(
                        p, 150.0,
                        "single category must sit at midpoint (reverse={reverse}, padding={padding}, align={align})"
                    );
                }
            }
        }
    }

    /// Zero-extent range: every position collapses to the single pixel, with
    /// and without reverse (reverse of a constant is the same constant).
    #[test]
    fn point_zero_extent_range_positions_collapse() {
        for reverse in [false, true] {
            let s = point(&["a", "b", "c"], 0.5, 0.5, reverse);
            for cat in ["a", "b", "c"] {
                let p = s.scale_str(cat, 100.0, 100.0);
                assert_eq!(p, 100.0, "zero-extent position for {cat} (reverse={reverse}) must be 100");
            }
        }
    }

    /// reverse endpoint oracle with padding: n=3, padding=0.5 over [0, 300] →
    /// step=100, forward a/b/c at 50/150/250; reversed, a mirrors to 250.
    /// (Mirror about the range midpoint, per the domain-reversal equivalence
    /// documented in `render/scale_resolve/positional.rs`.)
    #[test]
    fn point_reverse_with_padding_mirrors_about_midpoint() {
        let s = point(&["a", "b", "c"], 0.5, 0.5, true);
        assert!((s.scale_str("a", 0.0, 300.0) - 250.0).abs() < 1e-9);
        assert!((s.scale_str("b", 0.0, 300.0) - 150.0).abs() < 1e-9);
        assert!((s.scale_str("c", 0.0, 300.0) - 50.0).abs() < 1e-9);
    }
}
