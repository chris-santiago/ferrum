use pyo3::prelude::*;

use super::core::validate_ordinal;

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
/// Parameters
/// ----------
/// domain : list[str]
///     Ordered list of category labels.
/// range : list[float]
///     Pixel positions for the scale endpoints. The scale divides the
///     interval between ``range[0]`` and ``range[-1]`` evenly across the
///     domain categories.
/// padding : float, default 0.0
///     Fractional inner padding between bands, in ``[0.0, 1.0)``.
///
/// Examples
/// --------
/// Ordinal scales are normally constructed implicitly when ``Chart.encode``
/// detects a categorical column. Pass an instance to fix the category order::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         x=fr.X("group", scale=fr.OrdinalScale(domain=["A", "B", "C"], range=[0, 300]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalScale(OrdinalScaleData, bool);

impl OrdinalScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    pub(crate) fn new_internal(domain: Vec<String>, range: Vec<f64>, padding: f64) -> Self {
        OrdinalScale(OrdinalScaleData { domain, range, padding }, true)
    }

    /// Crate-internal lookup. Returns `None` if `value` is not in the domain.
    pub(crate) fn scale_internal(&self, value: &str) -> Option<f64> {
        let v = self.0.scale_str(value);
        if v.is_nan() { None } else { Some(v) }
    }

    /// Crate-internal tick call (returns the categorical domain).
    pub(crate) fn ticks_internal(&self) -> Vec<String> {
        self.0.domain.clone()
    }

    /// Per-category band width in pixels (the full step between consecutive
    /// band centers). Used by render-side position adjustments (Phase 9c
    /// Dodge) to compute sub-band offsets. The returned value is positive even
    /// when the range is reversed (lo > hi).
    pub(crate) fn bandwidth(&self) -> f64 {
        if self.0.domain.is_empty() { return 0.0; }
        let r_lo = *self.0.range.first().unwrap();
        let r_hi = *self.0.range.last().unwrap();
        ((r_hi - r_lo) / self.0.domain.len() as f64).abs()
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        [
            *self.0.range.first().unwrap(),
            *self.0.range.last().unwrap(),
        ]
    }

    pub(crate) fn repr_string(&self) -> String {
        let OrdinalScaleData { domain, range, padding } = &self.0;
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
    fn new(domain: Vec<String>, range: Option<Vec<f64>>, padding: f64) -> PyResult<Self> {
        let range_user_set = range.is_some();
        let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
        validate_ordinal(&domain, &r, padding)?;
        Ok(OrdinalScale(OrdinalScaleData { domain, range: r, padding }, range_user_set))
    }

    /// Map a category label to its band-center pixel coordinate.
    ///
    /// Returns ``f64::NAN`` for labels not in the domain.
    fn scale(&self, value: &str) -> f64 {
        self.0.scale_str(value)
    }

    /// Return the category label whose band contains pixel coordinate ``y``,
    /// or ``None`` if ``y`` is out of range.
    fn invert(&self, y: f64) -> Option<String> {
        self.0.invert_band(y)
    }

    /// Return the domain categories in order.
    fn ticks(&self) -> Vec<String> {
        self.0.domain.clone()
    }

    /// Return this scale unchanged (ordinal scales have no numeric "nice" rounding).
    fn nice(&self) -> Self {
        self.clone()
    }

    /// Ordered list of category labels.
    #[getter]
    fn domain(&self) -> Vec<String> {
        self.0.domain.clone()
    }

    /// Pixel extent of the scale as the full range list, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.1 { Some(self.0.range.clone()) } else { None }
    }

    /// Fractional inner padding between bands.
    #[getter]
    fn padding(&self) -> f64 {
        self.0.padding
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
}
