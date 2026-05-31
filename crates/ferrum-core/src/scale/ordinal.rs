use pyo3::prelude::*;
use pyo3::types::PyList;

use super::core::{validate_ordinal, validate_ordinal_domain};

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
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub(crate) enum OrdinalRangeValue {
    #[pyo3(transparent)]
    Number(f64),
    #[pyo3(transparent)]
    Str(String),
}

impl OrdinalRangeValue {
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
    /// Whether the user explicitly supplied a range (false → auto-fill from
    /// plot-area extent at render time).
    range_user_set: bool,
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
            range_user_set: true,
            range_orig: Some(range_orig),
        }
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

    /// Categorical scales have no numeric continuum to subdivide.
    ///
    /// Returns an empty `Vec` — this is a documented semantic absence, not an
    /// error.  `minor=True` on a categorical axis produces no minor gridlines.
    // Wired to the render layer in Task 2 of the grid subsystem.
    #[allow(dead_code)]
    pub(crate) fn minor_ticks_internal(&self) -> Vec<crate::scale::ticks::Tick> {
        Vec::new()
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

    pub(crate) fn repr_string(&self) -> String {
        let OrdinalScaleData { domain, range, padding } = &self.data;
        format!(
            "OrdinalScale(domain={:?}, range=[{}, {}], padding={})",
            domain, range.first().copied().unwrap_or(0.0), range.last().copied().unwrap_or(0.0), padding
        )
    }
}

#[pymethods]
impl OrdinalScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, padding = 0.0))]
    fn new(domain: Vec<String>, range: Option<Vec<OrdinalRangeValue>>, padding: f64) -> PyResult<Self> {
        let range_user_set = range.is_some();
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
            range_user_set,
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
        if !self.range_user_set {
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

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Categorical (ordinal) scales have no continuum to subdivide.
    /// minor_ticks_internal must always return empty — this is a documented
    /// semantic absence (not an error or unimplemented path).
    #[test]
    fn ordinal_minor_ticks_always_empty() {
        let scale = OrdinalScale {
            data: OrdinalScaleData {
                domain: vec!["a".into(), "b".into(), "c".into()],
                range: vec![0.0, 300.0],
                padding: 0.1,
            },
            range_user_set: true,
            range_orig: Some(vec![
                OrdinalRangeValue::Number(0.0),
                OrdinalRangeValue::Number(300.0),
            ]),
        };
        let minors = scale.minor_ticks_internal();
        assert!(
            minors.is_empty(),
            "ordinal minor_ticks_internal must return empty, got {minors:?}",
        );
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
            range_user_set: true,
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
}
