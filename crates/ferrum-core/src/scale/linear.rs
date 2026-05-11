use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};

/// Continuous linear scale.
///
/// Maps a numeric domain to a numeric range via affine transformation.
/// Domain endpoints are derived from data min/max when not supplied;
/// range is derived from the axis pixel extent.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to "nice" values for tick generation.
///
/// Examples
/// --------
/// Scales are normally constructed implicitly by ``Chart.encode(...)``.
/// Pass an instance explicitly to override the defaults::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         x=fr.X("value", scale=fr.LinearScale(domain=[0, 100], range=[0, 400]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearScale(Scale, Option<f64>);

impl LinearScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `padding` defaults to `None`; the renderer applies its own padding fraction
    /// before constructing the scale, so this field is meaningful only on
    /// user-supplied scale specs that roundtrip through serde.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let mut s = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            s = s.nice();
        }
        LinearScale(s, None)
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
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Vec<f64>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let mut s = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            s = s.nice();
        }
        Ok(LinearScale(s, padding))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 {
        self.0.invert_f64(y)
    }

    /// Return approximately ``count`` evenly-spaced tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> {
        self.0.ticks(Some(count))
    }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self {
        LinearScale(self.0.clone().nice(), self.1)
    }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset; an explicit value
    /// (including 0.0) overrides the default at render time.
    #[getter]
    fn padding(&self) -> Option<f64> {
        self.1
    }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Output range as ``[lo, hi]`` pixel coordinates.
    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
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
