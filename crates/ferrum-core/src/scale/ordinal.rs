use pyo3::prelude::*;

use super::core::{validate_ordinal, Scale};

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
pub struct OrdinalScale(Scale);

impl OrdinalScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    pub(crate) fn new_internal(domain: Vec<String>, range: Vec<f64>, padding: f64) -> Self {
        OrdinalScale(Scale::Ordinal { domain, range, padding })
    }

    /// Crate-internal lookup. Returns `None` if `value` is not in the domain.
    pub(crate) fn scale_internal(&self, value: &str) -> Option<f64> {
        let v = self.0.scale_str(value);
        if v.is_nan() { None } else { Some(v) }
    }

    /// Crate-internal tick call (returns the categorical domain).
    pub(crate) fn ticks_internal(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Per-category band width in pixels (the full step between consecutive
    /// band centers). Used by render-side position adjustments (Phase 9c
    /// Dodge) to compute sub-band offsets. The returned value is positive even
    /// when the range is reversed (lo > hi).
    pub(crate) fn bandwidth(&self) -> f64 {
        match &self.0 {
            Scale::Ordinal { domain, range, .. } => {
                if domain.is_empty() { return 0.0; }
                let r_lo = *range.first().unwrap();
                let r_hi = *range.last().unwrap();
                ((r_hi - r_lo) / domain.len() as f64).abs()
            }
            #[allow(unreachable_patterns)]
            _ => 0.0,
        }
    }

    /// Pixel-range endpoints `[lo, hi]` of the underlying scale.
    pub(crate) fn range_pair(&self) -> [f64; 2] {
        match &self.0 {
            Scale::Ordinal { range, .. } => [
                *range.first().unwrap(),
                *range.last().unwrap(),
            ],
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Ordinal { domain, range, padding } => format!(
                "OrdinalScale(domain={:?}, range=[{}, {}], padding={})",
                domain, range.first().copied().unwrap_or(0.0), range.last().copied().unwrap_or(0.0), padding
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl OrdinalScale {
    #[new]
    #[pyo3(signature = (*, domain, range, padding = 0.0))]
    fn new(domain: Vec<String>, range: Vec<f64>, padding: f64) -> PyResult<Self> {
        validate_ordinal(&domain, &range, padding)?;
        Ok(OrdinalScale(Scale::Ordinal { domain, range, padding }))
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
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Return this scale unchanged (ordinal scales have no numeric "nice" rounding).
    fn nice(&self) -> Self {
        self.clone()
    }

    /// Ordered list of category labels.
    #[getter]
    fn domain(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Pixel extent of the scale as the full range list.
    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Ordinal { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Fractional inner padding between bands.
    #[getter]
    fn padding(&self) -> f64 {
        match &self.0 {
            Scale::Ordinal { padding, .. } => *padding,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}
