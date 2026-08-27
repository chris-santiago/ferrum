use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};

use super::core::{scale_spec_to_py_dict, validate_ordinal, validate_ordinal_domain};
use crate::spec::encoding::ScaleSpec;

/// A single element of an `OrdinalScale` range: either a pixel coordinate
/// (number) or a color string.
///
/// This enum exists so the PyO3 `OrdinalScale.range` getter can round-trip
/// whatever the user passed in — numbers for positional/axis scales, CSS color
/// strings for categorical color scales — without loss.  The internal
/// `OrdinalScaleData.range` field remains `Vec<f64>` for all arithmetic.
///
/// The `#[derive(FromPyObject)]` implementation in PyO3 0.28 tries each variant
/// in order; `Number` is tried first so integer literals (which Python can
/// extract as both `f64` and `String`) are always captured as numbers.
///
/// The `#[serde(untagged)]` representation makes a `Vec<OrdinalRangeValue>`
/// serialize to (and deserialize from) a plain JSON array — `[0.0, 300.0]`
/// for a numeric range, `["#ccc", "#e4572e"]` for a color range, or a mix.
/// This is the exact shape the Python side already emits for `scale.range`
/// (a flat `list[float | str]`), so the typed wire representation requires no
/// Python-side change (F2 fix, 2026-05-31).  As with `FromPyObject`, variant
/// order matters: serde tries `Number` first, so JSON numbers never get
/// captured as strings.
// `pub` (not `pub(crate)`) only so it can appear in the `pub` `ScaleSpec::Ordinal`
// field without tripping `private_interfaces`. The `scale` module is itself
// private (`mod scale;` in lib.rs), so this type stays crate-internal in
// practice — exactly like `ScaleSpec` and `ContinuousScaleCommon`.
#[derive(Debug, Clone, PartialEq, FromPyObject, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OrdinalRangeValue {
    #[pyo3(transparent)]
    Number(f64),
    #[pyo3(transparent)]
    Str(String),
}

impl OrdinalRangeValue {
    /// The pixel coordinate, if this entry is a `Number`.
    pub(crate) fn as_number(&self) -> Option<f64> {
        match self {
            OrdinalRangeValue::Number(f) => Some(*f),
            OrdinalRangeValue::Str(_) => None,
        }
    }

    /// The string value, if this entry is a `Str` (e.g. a CSS color).
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            OrdinalRangeValue::Str(s) => Some(s),
            OrdinalRangeValue::Number(_) => None,
        }
    }

    /// Extract the numeric pixel coordinates from a typed ordinal range.
    ///
    /// Returns the numbers in order, dropping any string (color) entries. An
    /// empty result means the range carried no pixel coordinates — the caller
    /// (positional resolver) should fall back to the plot-area extent.
    pub(crate) fn numbers(range: &[OrdinalRangeValue]) -> Vec<f64> {
        range.iter().filter_map(OrdinalRangeValue::as_number).collect()
    }

    /// Extract the color strings from a typed ordinal range, but only when the
    /// range is entirely strings.
    ///
    /// Returns `Some` only for a non-empty all-string range (the color path);
    /// `None` when the range is empty or contains any numeric entry (a pixel
    /// range intended for the positional resolver). This preserves the exact
    /// all-or-nothing semantics of the previous JSON sniffer.
    pub(crate) fn all_strings(range: &[OrdinalRangeValue]) -> Option<Vec<String>> {
        if range.is_empty() {
            return None;
        }
        let strings: Vec<String> =
            range.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
        if strings.len() == range.len() {
            Some(strings)
        } else {
            None
        }
    }

    /// Convert to a Python object (`float` or `str`) for use in getters.
    fn into_py_any(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            OrdinalRangeValue::Number(f) => {
                Ok(f.into_pyobject(py).map(|b| b.into_any().unbind())?)
            }
            OrdinalRangeValue::Str(s) => {
                Ok(s.into_pyobject(py).map(|b| b.into_any().unbind())?)
            }
        }
    }
}

/// Provenance of an `OrdinalScale`'s pixel range (GH #79b).
///
/// Replaces the former `range_user_set`/`explicit_pixel_range` boolean pair
/// (bug-hunt design review, GH #69). A call-site audit found the two facts
/// are never independently meaningful on one instance — `explicit_pixel_range
/// == true` never occurs without `range_user_set == true` also being true,
/// because `with_explicit_range` (the only writer of the old
/// `explicit_pixel_range`) is chained exclusively onto a fresh
/// `new_internal` result, which unconditionally hardcoded
/// `range_user_set = true`. Exactly three (not four) states are reachable,
/// modeled here as one enum instead of two independent flags:
///
/// - [`Default`](Self::Default) — old `(range_user_set: false,
///   explicit_pixel_range: false)`. Only reachable from `OrdinalScale::new`
///   (the PyO3 constructor) when no `range=` kwarg was passed. The `.range`
///   getter returns `None`; the explicit-range accessors return `None`/no-op.
/// - [`UserSet`](Self::UserSet) — old `(true, false)`. Reachable two ways
///   that share the exact same reader-visible behavior: `OrdinalScale::new`
///   with a `range=` kwarg (feeds the `.range` getter and
///   `to_scale_spec()`'s wire form via `range_orig`), and every
///   `new_internal` render-resolver instance that is never subsequently
///   marked explicit via `with_explicit_range(true)` (the panel-extent
///   fallback). The `.range` getter is declared unread on render-internal
///   instances — render code never calls it — but its `Some`-returning
///   behavior is preserved byte-for-byte in case a future/test caller
///   invokes it. The explicit-range accessors return `None`/no-op for this
///   variant either way.
/// - [`ExplicitPixel`](Self::ExplicitPixel) — old `(true, true)`. Set only
///   via `with_explicit_range(true)` at the render-internal resolver call
///   sites (`render::scale_resolve::positional`) that resolved a
///   user-supplied `BandScale`/`PointScale`/positional-`OrdinalScale` pixel
///   range (GH #39 phase 2), as opposed to the panel-extent fallback. Feeds
///   `explicit_band_extent()`, `explicit_band_centers()`, and
///   `translate_explicit_range()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangeProvenance {
    Default,
    UserSet,
    ExplicitPixel,
}

impl RangeProvenance {
    /// Old `range_user_set` answer: `false` only for [`Default`](Self::Default).
    fn range_is_user_set(self) -> bool {
        !matches!(self, RangeProvenance::Default)
    }

    /// Old `explicit_pixel_range` answer: `true` only for
    /// [`ExplicitPixel`](Self::ExplicitPixel).
    fn is_explicit_pixel(self) -> bool {
        matches!(self, RangeProvenance::ExplicitPixel)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OrdinalScaleData {
    domain: Vec<String>,
    range: Vec<f64>,
    padding: f64,
}

impl OrdinalScaleData {
    fn layout(&self) -> (f64, f64, f64) {
        let r_lo = *self.range.first().unwrap();
        let r_hi = *self.range.last().unwrap();
        let n = self.domain.len() as f64;
        let step = (r_hi - r_lo) / n;
        let half_band = step.abs() * (1.0 - self.padding) / 2.0;
        let first_center = r_lo + step / 2.0;
        (first_center, step, half_band)
    }

    fn scale_str(&self, s: &str) -> f64 {
        let idx = match self.domain.iter().position(|c| c == s) {
            Some(i) => i,
            None => return f64::NAN,
        };
        let (first_center, step, _) = self.layout();
        first_center + (idx as f64) * step
    }

    fn invert_band(&self, y: f64) -> Option<String> {
        if y.is_nan() { return None; }
        let (first_center, step, half_band) = self.layout();
        if step == 0.0 { return None; }
        let raw = (y - first_center) / step;
        let idx = raw.round() as i64;
        if idx < 0 || idx as usize >= self.domain.len() { return None; }
        let center = first_center + (idx as f64) * step;
        if (y - center).abs() <= half_band {
            Some(self.domain[idx as usize].clone())
        } else {
            None
        }
    }
}

/// Discrete ordinal scale.
///
/// Maps a categorical (string) domain to an evenly-divided numeric range.
/// Each category is assigned a band center within the range, with optional
/// inner padding between bands. Tick generation returns the category list.
///
/// The ``range`` parameter accepts either pixel coordinates (``list[float]``)
/// for positional/axis use, or CSS color strings (``list[str]``) for
/// categorical color encoding. Numbers and strings may be mixed.
///
/// Parameters
/// ----------
/// domain : list[str]
///     Ordered list of category labels.
/// range : list[float | str], optional
///     Pixel positions (numbers) or color strings for the scale output.
///     When absent the renderer fills in the plot-area extent.
/// padding : float, default 0.0
///     Fractional inner padding between bands, in ``[0.0, 1.0)``.
///
/// Examples
/// --------
/// Positional ordinal scale (fix category order)::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         x=fr.X("group", scale=fr.OrdinalScale(domain=["A", "B", "C"], range=[0, 300]))
///     )
///
/// Color ordinal scale (accent one category)::
///
///     chart = fr.Chart(df).encode(
///         color=fr.Color("c:N", scale=fr.OrdinalScale(
///             domain=["A", "B", "C"],
///             range=["#cccccc", "#cccccc", "#e4572e"],
///         ))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalScale {
    data: OrdinalScaleData,
    /// Range-provenance origin (GH #79b). See [`RangeProvenance`] for the
    /// full old-booleans mapping.
    range_provenance: RangeProvenance,
    /// Original range values as supplied by the user, preserved for the
    /// Python getter. `None` when the user did not supply a range.
    range_orig: Option<Vec<OrdinalRangeValue>>,
}

impl OrdinalScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    ///
    /// Always supplies a numeric range; `range_orig` is set to match.
    pub(crate) fn new_internal(domain: Vec<String>, range: Vec<f64>, padding: f64) -> Self {
        let range_orig: Vec<OrdinalRangeValue> =
            range.iter().map(|&f| OrdinalRangeValue::Number(f)).collect();
        OrdinalScale {
            data: OrdinalScaleData { domain, range, padding },
            range_provenance: RangeProvenance::UserSet,
            range_orig: Some(range_orig),
        }
    }

    /// Marks this scale's pixel range as explicitly supplied by the user
    /// (vs. the panel-extent fallback `new_internal` defaults to). Chainable
    /// builder for the render-internal resolver call sites that know whether
    /// the `ScaleSpec` carried a usable range (GH #39 phase 2). Never called
    /// with `true` outside `render::scale_resolve::positional` — every other
    /// `new_internal` call site keeps the default `false`. Always called
    /// directly on a fresh `new_internal` result (never on a PyO3-constructed
    /// instance), so `explicit == false` reverts to `RangeProvenance::UserSet`
    /// — `new_internal`'s own provenance — rather than losing information.
    pub(crate) fn with_explicit_range(mut self, explicit: bool) -> Self {
        self.range_provenance =
            if explicit { RangeProvenance::ExplicitPixel } else { RangeProvenance::UserSet };
        self
    }

    /// Signed pixel extent (`r1 − r0`, in range order) of this scale's band
    /// range, but only when [`with_explicit_range`](Self::with_explicit_range)
    /// recorded it as user-supplied. `None` for the panel-extent fallback,
    /// even though that range is numerically valid — explicitness is recorded
    /// at construction, never inferred by comparing floats.
    pub(crate) fn explicit_band_extent(&self) -> Option<f64> {
        if !self.range_provenance.is_explicit_pixel() {
            return None;
        }
        let r0 = *self.data.range.first()?;
        let r1 = *self.data.range.last()?;
        Some(r1 - r0)
    }

    /// Absolute band-center pixels, one per domain category in order, but only
    /// when [`with_explicit_range`](Self::with_explicit_range) recorded this
    /// scale's range as user-supplied (GH #39 phase 2, band-geometry
    /// unification). These are the same pixels [`scale_internal`](Self::scale_internal)
    /// yields for each category — i.e. the pixels marks are placed at — so a
    /// categorical axis that consumes them agrees with its marks. `None` for the
    /// panel-extent fallback, keeping the layout `uniform_center` path (and thus
    /// every no-range golden) byte-identical. Explicitness is recorded at
    /// construction, never inferred by comparing floats.
    pub(crate) fn explicit_band_centers(&self) -> Option<Vec<f64>> {
        if !self.range_provenance.is_explicit_pixel() {
            return None;
        }
        Some(self.data.domain.iter().map(|c| self.data.scale_str(c)).collect())
    }

    /// Crate-internal lookup. Returns `None` if `value` is not in the domain.
    pub(crate) fn scale_internal(&self, value: &str) -> Option<f64> {
        let v = self.data.scale_str(value);
        if v.is_nan() { None } else { Some(v) }
    }

    /// Crate-internal tick call (returns the categorical domain).
    pub(crate) fn ticks_internal(&self) -> Vec<String> {
        self.data.domain.clone()
    }

    /// Per-category band width in pixels (the full step between consecutive
    /// band centers). Used by render-side position adjustments (Phase 9c
    /// Dodge) to compute sub-band offsets. The returned value is positive even
    /// when the range is reversed (lo > hi).
    pub(crate) fn bandwidth(&self) -> f64 {
        if self.data.domain.is_empty() { return 0.0; }
        let r_lo = *self.data.range.first().unwrap();
        let r_hi = *self.data.range.last().unwrap();
        ((r_hi - r_lo) / self.data.domain.len() as f64).abs()
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        [
            *self.data.range.first().unwrap(),
            *self.data.range.last().unwrap(),
        ]
    }

    /// Re-anchor this scale's pixel range by `offset`, used to place an
    /// EXPLICITLY user-supplied Band/Point/positional-Ordinal range onto a
    /// facet panel other than the reference panel (GH #70).
    ///
    /// An explicit range is chart-absolute by design (#39 phase 2): every
    /// panel's resolver call receives the SAME literal `[lo, hi]`, so under
    /// faceting every panel's marks — and, via `explicit_band_centers`, its
    /// axis ticks — would otherwise land at the identical absolute pixels,
    /// correct for at most one panel and overlapping/clipped for the rest.
    /// The render-side caller (`scene_build::resolve_panel_scales`) translates
    /// every panel but the reference (panel 0, where `offset == 0.0`) by that
    /// panel's own displacement from panel 0's plot-area origin, so the same
    /// user-specified extent re-anchors inside each panel's own window.
    ///
    /// No-op when this scale's range was never recorded as explicit
    /// (`range_provenance` is not [`RangeProvenance::ExplicitPixel`]) — the
    /// panel-extent fallback already resolves per-panel and must not be
    /// shifted again.
    ///
    /// Only `data.range` is load-bearing at render time (mark placement,
    /// `explicit_band_extent()`, `explicit_band_centers()`); this deliberately
    /// leaves the wire/getter-facing `range_orig` untouched. A render-internal
    /// `OrdinalScale` (built via `new_internal` + `with_explicit_range`) never
    /// flows back through the `.range` Python getter or `to_scale_spec()` — those
    /// exist for the PyO3-constructed path — so mutating `range_orig` here would
    /// be unread, cosmetic-looking state with no consumer, and a future reader
    /// might mistake it for load-bearing.
    pub(crate) fn translate_explicit_range(&mut self, offset: f64) {
        if !self.range_provenance.is_explicit_pixel() || offset == 0.0 {
            return;
        }
        for r in self.data.range.iter_mut() {
            *r += offset;
        }
    }

    pub(crate) fn repr_string(&self) -> String {
        let OrdinalScaleData { domain, range, padding } = &self.data;
        format!(
            "OrdinalScale(domain={:?}, range=[{}, {}], padding={})",
            domain, range.first().copied().unwrap_or(0.0), range.last().copied().unwrap_or(0.0), padding
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `range` carries the user's original polymorphic values (numbers and/or
    /// color strings) via `OrdinalRangeValue`, and is emitted only when present
    /// and non-empty — mirroring the `if scale.domain/range:` guards the legacy
    /// `_scale_to_dict` applied.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Ordinal {
            domain: if self.data.domain.is_empty() {
                None
            } else {
                Some(self.data.domain.clone())
            },
            range: self.range_orig.as_ref().filter(|v| !v.is_empty()).cloned(),
            padding: self.data.padding,
        }
    }
}

#[pymethods]
impl OrdinalScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, padding = 0.0))]
    fn new(domain: Vec<String>, range: Option<Vec<OrdinalRangeValue>>, padding: f64) -> PyResult<Self> {
        let range_provenance =
            if range.is_some() { RangeProvenance::UserSet } else { RangeProvenance::Default };
        let range_orig = range.clone();

        // Extract numeric values from the range for the internal `OrdinalScaleData`.
        // A string-only range (e.g. color strings) has no pixel coordinates;
        // use the default placeholder [0.0, 1.0] — it will never be used for
        // positional math when the scale is used purely as a color mapping.
        let numeric_range: Vec<f64> = range
            .as_deref()
            .map(|vals| {
                vals.iter()
                    .filter_map(|v| if let OrdinalRangeValue::Number(f) = v { Some(*f) } else { None })
                    .collect()
            })
            .unwrap_or_default();

        // Validate domain non-emptiness, duplicates, and padding regardless of range kind.
        validate_ordinal_domain(&domain, padding)?;

        // Validate numeric range values only when there are any (a string-only
        // range skips the numeric extent check — color strings have no pixel invariant).
        if !numeric_range.is_empty() {
            validate_ordinal(&domain, &numeric_range, padding)?;
        }

        // Internal data uses numeric range; fall back to [0.0, 1.0] placeholder
        // when none exist (string-only color range).
        let internal_range = if numeric_range.len() >= 2 {
            numeric_range
        } else {
            vec![0.0, 1.0]
        };

        Ok(OrdinalScale {
            data: OrdinalScaleData { domain, range: internal_range, padding },
            range_provenance,
            range_orig,
        })
    }

    /// Map a category label to its band-center pixel coordinate.
    ///
    /// Returns ``f64::NAN`` for labels not in the domain.
    fn scale(&self, value: &str) -> f64 {
        self.data.scale_str(value)
    }

    /// Return the category label whose band contains pixel coordinate ``y``,
    /// or ``None`` if ``y`` is out of range.
    fn invert(&self, y: f64) -> Option<String> {
        self.data.invert_band(y)
    }

    /// Return the domain categories in order.
    fn ticks(&self) -> Vec<String> {
        self.data.domain.clone()
    }

    /// Return this scale unchanged (ordinal scales have no numeric "nice" rounding).
    fn nice(&self) -> Self {
        self.clone()
    }

    /// Ordered list of category labels.
    #[getter]
    fn domain(&self) -> Vec<String> {
        self.data.domain.clone()
    }

    /// Scale range as originally supplied by the user: a list of floats, strings,
    /// or a mix. Returns ``None`` when no range was given (the renderer fills in
    /// the plot-area extent at render time).
    ///
    /// Round-trips losslessly: pixel coordinates come back as floats, color
    /// strings come back as strings.
    #[getter]
    fn range(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if !self.range_provenance.range_is_user_set() {
            return Ok(None);
        }
        let items: Vec<Py<PyAny>> = self
            .range_orig
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|v| v.clone().into_py_any(py))
            .collect::<PyResult<_>>()?;
        Ok(Some(PyList::new(py, items)?.into_any().unbind()))
    }

    /// Fractional inner padding between bands.
    #[getter]
    fn padding(&self) -> f64 {
        self.data.padding
    }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: Vec<&str>, range: Vec<f64>, padding: f64) -> OrdinalScaleData {
        OrdinalScaleData {
            domain: domain.into_iter().map(String::from).collect(),
            range,
            padding,
        }
    }

    #[test]
    fn ordinal_band_centers_no_padding() {
        let s = d(vec!["a", "b", "c"], vec![0.0, 30.0], 0.0);
        assert!((s.scale_str("a") - 5.0).abs() < 1e-12);
        assert!((s.scale_str("b") - 15.0).abs() < 1e-12);
        assert!((s.scale_str("c") - 25.0).abs() < 1e-12);
    }

    #[test]
    fn ordinal_invert_round_trip() {
        let s = d(vec!["a", "b", "c"], vec![0.0, 30.0], 0.0);
        for cat in ["a", "b", "c"] {
            let y = s.scale_str(cat);
            let back = s.invert_band(y);
            assert_eq!(back.as_deref(), Some(cat), "round-trip failed for {cat}");
        }
    }

    #[test]
    fn ordinal_invert_outside_band_returns_none() {
        let s = d(vec!["a", "b", "c"], vec![0.0, 30.0], 0.5);
        assert!(s.invert_band(10.0).is_none());
    }

    #[test]
    fn ordinal_unknown_category_returns_nan() {
        let s = d(vec!["a"], vec![0.0, 10.0], 0.0);
        assert!(s.scale_str("z").is_nan());
    }

    // ── explicit_band_centers (GH #39 phase 2) ───────────────────────────────

    /// An explicit range reports absolute band centers, one per category in
    /// domain order — the same pixels `scale_internal` yields for marks. The
    /// canonical oracle: domain [a,b,c,d] over [40, 260] → step 55 → centers
    /// 67.5 / 122.5 / 177.5 / 232.5.
    #[test]
    fn explicit_band_centers_match_scale_internal_oracle() {
        let scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![40.0, 260.0],
            0.0,
        )
        .with_explicit_range(true);
        let centers = scale.explicit_band_centers().expect("explicit → Some");
        assert_eq!(centers, vec![67.5, 122.5, 177.5, 232.5]);
        // Each center equals the mark pixel for that category.
        for (cat, c) in ["a", "b", "c", "d"].iter().zip(&centers) {
            assert_eq!(scale.scale_internal(cat), Some(*c));
        }
    }

    /// The panel-extent fallback (never marked explicit) reports `None`, so the
    /// layout `uniform_center` path stays byte-identical.
    #[test]
    fn explicit_band_centers_none_without_explicit_flag() {
        let scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![0.0, 220.0],
            0.0,
        );
        assert_eq!(scale.explicit_band_centers(), None);
    }

    /// A reversed explicit range `[hi, lo]` is passed through as-is: band centers
    /// descend, matching the mark placement (`scale_internal`).
    #[test]
    fn explicit_band_centers_honor_reversed_range() {
        let scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![260.0, 40.0],
            0.0,
        )
        .with_explicit_range(true);
        let centers = scale.explicit_band_centers().expect("explicit → Some");
        assert_eq!(centers, vec![232.5, 177.5, 122.5, 67.5]);
        for (cat, c) in ["a", "b", "c", "d"].iter().zip(&centers) {
            assert_eq!(scale.scale_internal(cat), Some(*c));
        }
    }

    // ── translate_explicit_range (GH #70 facet re-anchoring) ─────────────────

    /// An explicit range shifts both endpoints — and thus every band center —
    /// by `offset`.
    #[test]
    fn translate_explicit_range_shifts_explicit_range() {
        let mut scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![40.0, 260.0],
            0.0,
        )
        .with_explicit_range(true);
        scale.translate_explicit_range(100.0);
        assert_eq!(scale.range_pair(), [140.0, 360.0]);
        assert_eq!(scale.explicit_band_extent(), Some(220.0), "shifting preserves the signed extent");
    }

    /// A non-explicit (panel-extent fallback) range is untouched by
    /// `translate_explicit_range`, even with a non-zero offset — the
    /// fallback already resolves per-panel and must not be shifted again.
    #[test]
    fn translate_explicit_range_noop_when_not_explicit() {
        let mut scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![40.0, 260.0],
            0.0,
        ); // no `.with_explicit_range(true)` — range_provenance stays UserSet
        scale.translate_explicit_range(100.0);
        assert_eq!(scale.range_pair(), [40.0, 260.0]);
    }

    /// An explicit range with a zero offset (panel 0 / the reference panel,
    /// or any standalone non-faceted render) is untouched.
    #[test]
    fn translate_explicit_range_noop_when_offset_zero() {
        let mut scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![40.0, 260.0],
            0.0,
        )
        .with_explicit_range(true);
        scale.translate_explicit_range(0.0);
        assert_eq!(scale.range_pair(), [40.0, 260.0]);
    }

    // ── OrdinalRangeValue round-trip tests ───────────────────────────────────

    /// A numeric range stores numbers and the internal data uses those numbers.
    #[test]
    fn ordinal_scale_numeric_range_roundtrip() {
        let scale = OrdinalScale::new_internal(
            vec!["A".into(), "B".into(), "C".into()],
            vec![0.0, 300.0],
            0.0,
        );
        // Internal range preserved for positional math.
        assert_eq!(scale.data.range, vec![0.0, 300.0]);
        // range_orig contains numbers.
        let orig = scale.range_orig.as_ref().unwrap();
        assert_eq!(orig.len(), 2);
        assert!(matches!(orig[0], OrdinalRangeValue::Number(f) if f == 0.0));
        assert!(matches!(orig[1], OrdinalRangeValue::Number(f) if f == 300.0));
    }

    /// A string-only range (color strings) stores the strings in range_orig
    /// and falls back to the [0.0, 1.0] placeholder in data.range.
    ///
    /// This test uses the crate-private fields directly (no PyO3 runtime needed).
    #[test]
    fn ordinal_scale_string_range_stored_losslessly() {
        // Build directly using struct literal (mirrors what `new()` produces).
        let scale = OrdinalScale {
            data: OrdinalScaleData {
                domain: vec!["A".into(), "B".into(), "C".into()],
                range: vec![0.0, 1.0], // placeholder for string-only range
                padding: 0.0,
            },
            range_provenance: RangeProvenance::UserSet,
            range_orig: Some(vec![
                OrdinalRangeValue::Str("#cccccc".into()),
                OrdinalRangeValue::Str("#cccccc".into()),
                OrdinalRangeValue::Str("#e4572e".into()),
            ]),
        };
        let orig = scale.range_orig.as_ref().unwrap();
        assert_eq!(orig.len(), 3);
        assert!(matches!(&orig[0], OrdinalRangeValue::Str(s) if s == "#cccccc"));
        assert!(matches!(&orig[1], OrdinalRangeValue::Str(s) if s == "#cccccc"));
        assert!(matches!(&orig[2], OrdinalRangeValue::Str(s) if s == "#e4572e"));
        // Internal data range is the placeholder — string ranges are NOT used for math.
        assert_eq!(scale.data.range, vec![0.0, 1.0]);
    }

    // ── OrdinalRangeValue serde + typed-accessor tests (F2) ──────────────────

    /// A numeric range deserializes from a plain JSON number array and
    /// serializes back to one. This is the exact JSON Python emits for
    /// `scale.range = [0.0, 300.0]`.
    #[test]
    fn ordinal_range_value_numeric_json_round_trip() {
        let json = "[0.0,300.0]";
        let parsed: Vec<OrdinalRangeValue> = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            vec![OrdinalRangeValue::Number(0.0), OrdinalRangeValue::Number(300.0)]
        );
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
    }

    /// Integer JSON literals (`[0, 300]`) deserialize as `Number` — variant
    /// order means numbers are never captured as strings.
    #[test]
    fn ordinal_range_value_integer_json_parses_as_number() {
        let parsed: Vec<OrdinalRangeValue> = serde_json::from_str("[0,300]").unwrap();
        assert_eq!(
            parsed,
            vec![OrdinalRangeValue::Number(0.0), OrdinalRangeValue::Number(300.0)]
        );
    }

    /// A string range deserializes from a plain JSON string array.
    #[test]
    fn ordinal_range_value_string_json_round_trip() {
        let json = r##"["#ccc","#e4572e"]"##;
        let parsed: Vec<OrdinalRangeValue> = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            vec![
                OrdinalRangeValue::Str("#ccc".into()),
                OrdinalRangeValue::Str("#e4572e".into())
            ]
        );
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
    }

    /// `numbers()` extracts the `Number` arms, dropping strings.
    #[test]
    fn ordinal_range_value_numbers_drops_strings() {
        let range = vec![
            OrdinalRangeValue::Number(10.0),
            OrdinalRangeValue::Str("#ccc".into()),
            OrdinalRangeValue::Number(20.0),
        ];
        assert_eq!(OrdinalRangeValue::numbers(&range), vec![10.0, 20.0]);
    }

    /// `all_strings()` returns `Some` only for a non-empty all-string range,
    /// matching the old JSON sniffer's all-or-nothing semantics.
    #[test]
    fn ordinal_range_value_all_strings_semantics() {
        let strings = vec![
            OrdinalRangeValue::Str("#ccc".into()),
            OrdinalRangeValue::Str("#e4572e".into()),
        ];
        assert_eq!(
            OrdinalRangeValue::all_strings(&strings),
            Some(vec!["#ccc".to_string(), "#e4572e".to_string()])
        );
        // Mixed → None (a numeric pixel entry disqualifies the color path).
        let mixed = vec![OrdinalRangeValue::Number(0.0), OrdinalRangeValue::Str("#ccc".into())];
        assert_eq!(OrdinalRangeValue::all_strings(&mixed), None);
        // Empty → None.
        assert_eq!(OrdinalRangeValue::all_strings(&[]), None);
    }

    // ── RangeProvenance mapping (GH #79b) ─────────────────────────────────────

    /// A `new_internal` instance, never marked explicit, answers exactly as
    /// the old `(range_user_set: true, explicit_pixel_range: false)` pair
    /// did: the `.range`-getter fact is `true` (even though unread on
    /// render-internal instances) and the explicit-pixel fact is `false`.
    #[test]
    fn range_provenance_new_internal_matches_old_booleans() {
        let scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![0.0, 100.0],
            0.0,
        );
        assert!(scale.range_provenance.range_is_user_set(), "old range_user_set was true");
        assert!(!scale.range_provenance.is_explicit_pixel(), "old explicit_pixel_range was false");
        assert_eq!(scale.explicit_band_extent(), None);
        assert_eq!(scale.explicit_band_centers(), None);
    }

    /// `OrdinalScale::new` (the PyO3 constructor) with a `range=` kwarg
    /// matches old `(range_user_set: true, explicit_pixel_range: false)`:
    /// the `.range` getter fires, and the render-side explicit accessors do
    /// not (a PyO3-constructed instance is never used positionally without
    /// first round-tripping through `ScaleSpec::Ordinal` and `new_internal`).
    #[test]
    fn range_provenance_pyclass_new_with_range_matches_old_booleans() {
        let scale = OrdinalScale::new(
            vec!["a".into(), "b".into()],
            Some(vec![OrdinalRangeValue::Number(0.0), OrdinalRangeValue::Number(100.0)]),
            0.0,
        )
        .unwrap();
        assert!(scale.range_provenance.range_is_user_set(), "old range_user_set was true");
        assert!(!scale.range_provenance.is_explicit_pixel(), "old explicit_pixel_range was false");
        assert_eq!(scale.explicit_band_extent(), None);
    }

    /// `OrdinalScale::new` without a `range=` kwarg matches old
    /// `(range_user_set: false, explicit_pixel_range: false)`: the `.range`
    /// getter returns `None` and the explicit accessors are also inert.
    #[test]
    fn range_provenance_pyclass_new_without_range_matches_old_booleans() {
        let scale = OrdinalScale::new(vec!["a".into(), "b".into()], None, 0.0).unwrap();
        assert!(!scale.range_provenance.range_is_user_set(), "old range_user_set was false");
        assert!(!scale.range_provenance.is_explicit_pixel(), "old explicit_pixel_range was false");
        assert_eq!(scale.explicit_band_extent(), None);
    }

    /// `with_explicit_range(true)` flips only the explicit-side answer: the
    /// `.range`-getter fact (`range_is_user_set`) stays `true`, matching old
    /// behavior where `explicit_pixel_range` was set independently of
    /// `range_user_set` (which `new_internal` had already hardcoded `true`).
    #[test]
    fn with_explicit_range_true_flips_only_explicit_side() {
        let scale = OrdinalScale::new_internal(
            vec!["a".into(), "b".into()],
            vec![0.0, 100.0],
            0.0,
        )
        .with_explicit_range(true);
        assert!(scale.range_provenance.range_is_user_set(), "range_is_user_set unaffected");
        assert!(scale.range_provenance.is_explicit_pixel());
        assert_eq!(scale.explicit_band_extent(), Some(100.0));
    }

    /// `with_explicit_range(false)` is the default state `new_internal`
    /// already produces — a no-op that keeps both facts identical to a plain
    /// `new_internal` result.
    #[test]
    fn with_explicit_range_false_is_noop_vs_new_internal_default() {
        let plain = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![0.0, 100.0], 0.0);
        let explicit_false = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![0.0, 100.0], 0.0)
            .with_explicit_range(false);
        assert_eq!(plain.range_provenance, explicit_false.range_provenance);
    }

    // ── OrdinalScaleData degenerate / signed-extent cases (ported from
    // tests/bug_hunt_band_point_range.rs, R1) ────────────────────────────────

    /// Zero-width explicit range [150, 150]: step = 0, every center collapses
    /// to 150, no NaN anywhere, bandwidth 0. Exercises the division
    /// `(r_hi - r_lo)/n` with a zero numerator (a user-reachable state via
    /// `BandScale(range=[c, c])`).
    #[test]
    fn ordinal_zero_width_range_centers_collapse_finitely() {
        let s = d(vec!["a", "b", "c"], vec![150.0, 150.0], 0.0);
        for cat in ["a", "b", "c"] {
            let p = s.scale_str(cat);
            assert!(p.is_finite(), "center for {cat} must be finite, got {p}");
            assert_eq!(p, 150.0, "all centers must collapse to the range pixel");
        }
        let ord = OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into()],
            vec![150.0, 150.0],
            0.0,
        );
        assert_eq!(ord.bandwidth(), 0.0);
    }

    /// Zero-width range + invert: the `step == 0.0` guard must return `None`
    /// (not divide by zero into a bogus index).
    #[test]
    fn ordinal_zero_width_range_invert_returns_none() {
        let s = d(vec!["a", "b"], vec![150.0, 150.0], 0.0);
        assert_eq!(s.invert_band(150.0), None, "step==0 invert must refuse, not divide by 0");
        assert_eq!(s.invert_band(0.0), None);
    }

    /// Inverted explicit range [260, 40] (hi < lo): 4 categories, step = -55,
    /// centers descend 232.5 / 177.5 / 122.5 / 67.5 — the signed-extent oracle
    /// consumed by `explicit_band_centers` — and `OrdinalScale::bandwidth()`
    /// stays positive (55).
    #[test]
    fn ordinal_inverted_range_descending_centers_positive_bandwidth() {
        let s = d(vec!["a", "b", "c", "d"], vec![260.0, 40.0], 0.0);
        assert_eq!(s.scale_str("a"), 232.5);
        assert_eq!(s.scale_str("b"), 177.5);
        assert_eq!(s.scale_str("c"), 122.5);
        assert_eq!(s.scale_str("d"), 67.5);
        let ord = OrdinalScale::new_internal(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![260.0, 40.0],
            0.0,
        );
        assert_eq!(ord.bandwidth(), 55.0, "bandwidth must be |step| under a reversed range");
    }

    /// Inverted range + inner padding: `invert(scale(cat))` round-trips for
    /// every category. The `half_band` term uses `step.abs()`, so a negative
    /// step must not make the acceptance window empty.
    #[test]
    fn ordinal_inverted_range_invert_round_trip_with_padding() {
        let s = d(vec!["a", "b", "c", "d"], vec![260.0, 40.0], 0.4);
        for cat in ["a", "b", "c", "d"] {
            let y = s.scale_str(cat);
            assert_eq!(
                s.invert_band(y).as_deref(),
                Some(cat),
                "invert(scale({cat})) must round-trip under an inverted range"
            );
        }
    }

    /// Inverted range invert: a pixel just outside the band gap (between
    /// bands) returns `None` even with the negative step (the
    /// `|y - center| <= half_band` window must be evaluated symmetrically).
    #[test]
    fn ordinal_inverted_range_invert_rejects_gap_pixels() {
        // padding 0.5 → half_band = 55 * 0.5 / 2 = 13.75. Band "a" center
        // 232.5, band "b" center 177.5; midpoint 205.0 sits in the padding gap.
        let s = d(vec!["a", "b", "c", "d"], vec![260.0, 40.0], 0.5);
        assert_eq!(s.invert_band(205.0), None, "gap pixel must not resolve to a category");
    }

    /// Single category over an explicit range: center = midpoint of the range
    /// (step = extent, first_center = lo + extent/2), and bandwidth equals the
    /// full extent regardless of range order.
    #[test]
    fn ordinal_single_category_explicit_range_center_is_midpoint() {
        let s = d(vec!["only"], vec![40.0, 260.0], 0.0);
        assert_eq!(s.scale_str("only"), 150.0);
        let ord = OrdinalScale::new_internal(vec!["only".into()], vec![40.0, 260.0], 0.0);
        assert_eq!(ord.bandwidth(), 220.0);
        // Under the inverted form the midpoint is the same pixel.
        let inv = d(vec!["only"], vec![260.0, 40.0], 0.0);
        assert_eq!(inv.scale_str("only"), 150.0);
    }

    /// Empty domain (0-row data resolved with an explicit range): step is
    /// `extent / 0` = ±inf internally, but every public lookup stays safe —
    /// `scale_str` returns NaN (category can't be found), invert returns
    /// `None`, bandwidth returns 0. Pins that the inf never escapes.
    #[test]
    fn ordinal_empty_domain_lookups_are_contained() {
        let s = d(Vec::new(), vec![40.0, 260.0], 0.0);
        assert!(s.scale_str("anything").is_nan());
        let ord = OrdinalScale::new_internal(Vec::new(), vec![40.0, 260.0], 0.0);
        assert_eq!(ord.bandwidth(), 0.0);
        // invert on the inf-step layout: raw = (y - inf)/inf = NaN; NaN.round()
        // as i64 saturates to 0; 0 >= domain.len()==0 → None. Must not panic.
        assert_eq!(s.invert_band(150.0), None);
        assert_eq!(s.invert_band(f64::NAN), None);
    }

    /// Extreme range magnitudes (1e300 span): centers stay finite for a small
    /// domain (no overflow in `(r_hi - r_lo) / n` or `first_center + i * step`).
    #[test]
    fn ordinal_huge_range_no_overflow() {
        let s = d(vec!["a", "b"], vec![-1e300, 1e300], 0.0);
        let a = s.scale_str("a");
        let b = s.scale_str("b");
        assert!(a.is_finite() && b.is_finite(), "centers must be finite: a={a}, b={b}");
        assert!(a < b);
        assert_eq!(a, -5e299);
        assert_eq!(b, 5e299);
    }
}
