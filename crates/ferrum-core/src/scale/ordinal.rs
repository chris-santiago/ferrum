use pyo3::prelude::*;

use super::core::{validate_ordinal, Scale};

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

    fn scale(&self, value: &str) -> f64 {
        self.0.scale_str(value)
    }

    fn invert(&self, y: f64) -> Option<String> {
        self.0.invert_band(y)
    }

    fn ticks(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn nice(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn domain(&self) -> Vec<String> {
        match &self.0 {
            Scale::Ordinal { domain, .. } => domain.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Ordinal { range, .. } => range.clone(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

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
