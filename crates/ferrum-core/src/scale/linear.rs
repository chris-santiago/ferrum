use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearScale(Scale);

impl LinearScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let mut s = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            s = s.nice();
        }
        LinearScale(s)
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    /// Crate-internal tick call.
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.0.ticks(Some(count))
    }

    /// Pixel-range pair `[lo, hi]` of the underlying scale. Used by `ScaleKind::pixel_range`.
    pub(crate) fn range_pair(&self) -> [f64; 2] {
        match &self.0 {
            Scale::Linear { range, .. } => *range,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Linear { domain, range, clamp } => format!(
                "LinearScale(domain=[{}, {}], range=[{}, {}], clamp={})",
                domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl LinearScale {
    #[new]
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let mut s = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(LinearScale(s))
    }

    fn scale(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    fn invert(&self, y: f64) -> f64 {
        self.0.invert_f64(y)
    }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> {
        self.0.ticks(Some(count))
    }

    fn nice(&self) -> Self {
        LinearScale(self.0.clone().nice())
    }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Linear { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}
