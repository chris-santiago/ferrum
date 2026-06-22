use pyo3::prelude::*;

use super::core::resolve_continuous;
use super::ticks::{minor_ticks_default, nice_step, nice_ticks, Tick};

/// Internal data for a linear-affine scale. Shared by [`LinearScale`] and
/// [`super::time::TimeScale`] (time scales use the same domain-to-range
/// mapping but add time-aware tick generation).
///
/// Kept `pub(super)` so `scale::time` can reach it without exposing the
/// data shape to the rest of the crate. The PyO3 newtypes wrap this struct
/// alongside their padding field; render-side callers go through the
/// newtypes' `scale_internal` / `range_pair` / `ticks_internal` accessors.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LinearScaleData {
    pub(super) domain: [f64; 2],
    pub(super) range: [f64; 2],
    pub(super) clamp: bool,
}

impl LinearScaleData {
    pub(super) fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let t = (x - d0) / (d1 - d0);
        let mapped = r0 + t * (r1 - r0);
        if self.clamp {
            let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
            mapped.clamp(lo, hi)
        } else if x < d0.min(d1) || x > d0.max(d1) {
            f64::NAN
        } else {
            mapped
        }
    }

    pub(super) fn invert(&self, y: f64) -> f64 {
        if y.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let t = (y - r0) / (r1 - r0);
        let mapped = d0 + t * (d1 - d0);
        if self.clamp {
            let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
            mapped.clamp(lo, hi)
        } else if y < r0.min(r1) || y > r0.max(r1) {
            f64::NAN
        } else {
            mapped
        }
    }

    pub(super) fn ticks(&self, count: usize) -> Vec<f64> {
        nice_ticks(self.domain[0], self.domain[1], count)
    }

    pub(super) fn nice(self) -> Self {
        let step = nice_step(self.domain[0], self.domain[1], 10);
        if !step.is_finite() || step == 0.0 {
            return self;
        }
        let lo_min = self.domain[0].min(self.domain[1]);
        let hi_max = self.domain[0].max(self.domain[1]);
        let nice_lo = (lo_min / step).floor() * step;
        let nice_hi = (hi_max / step).ceil() * step;
        let new_domain = if self.domain[0] <= self.domain[1] {
            [nice_lo, nice_hi]
        } else {
            [nice_hi, nice_lo]
        };
        Self { domain: new_domain, range: self.range, clamp: self.clamp }
    }
}

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
pub struct LinearScale {
    data: LinearScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    domain_user_set: bool,
}

impl LinearScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `padding` defaults to `None`; the renderer applies its own padding fraction
    /// before constructing the scale, so this field is meaningful only on
    /// user-supplied scale specs that roundtrip through serde.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let mut d = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        if nice {
            d = d.nice();
        }
        LinearScale { data: d, padding: None, range_user_set: true, domain_user_set: true }
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.data.scale(x)
    }

    /// Crate-internal tick call.
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.data.ticks(count)
    }

    /// Return minor ticks between the major ticks for this scale.
    ///
    /// Linear scales use the default subdivision algorithm: each major interval
    /// is divided into [`DEFAULT_MINOR_SUBDIVISIONS`] sub-intervals (5), yielding
    /// 4 interior minor ticks per gap.  For a plain linear scale the transformed
    /// space is identical to the data domain, so the subdivision is uniform in
    /// data space.
    ///
    /// The major tick count is fixed at 10 (the conventional default).  The
    /// minor tick *density* is always controlled by the locked
    /// `DEFAULT_MINOR_SUBDIVISIONS` constant (5 sub-intervals → 4 interior
    /// minors per major gap); there is no per-call override.
    // Wired to the render layer in Task 2 of the grid subsystem.
    #[allow(dead_code)]
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.data.ticks(10);
        minor_ticks_default(&majors, |x| x)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.data.range
    }

    pub(crate) fn domain_pair(&self) -> [f64; 2] {
        self.data.domain
    }

    pub(crate) fn repr_string(&self) -> String {
        let LinearScaleData { domain, range, clamp } = &self.data;
        let domain_s = if self.domain_user_set {
            format!("[{}, {}]", domain[0], domain[1])
        } else {
            "None".to_string()
        };
        let range_s = if self.range_user_set {
            format!("[{}, {}]", range[0], range[1])
        } else {
            "None".to_string()
        };
        format!(
            "LinearScale(domain={}, range={}, clamp={})",
            domain_s, range_s, if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl LinearScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        // Sentinel [0.0, 1.0] when no domain supplied; render-time inference
        // replaces it before any scale computation occurs.
        let resolved = resolve_continuous(domain, range, [0.0, 1.0])?;
        let mut d = LinearScaleData {
            domain: resolved.domain,
            range: resolved.range,
            clamp,
        };
        if nice && resolved.domain_user_set {
            d = d.nice();
        }
        Ok(LinearScale {
            data: d,
            padding,
            range_user_set: resolved.range_user_set,
            domain_user_set: resolved.domain_user_set,
        })
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 {
        self.data.scale(x)
    }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 {
        self.data.invert(y)
    }

    /// Return approximately ``count`` evenly-spaced tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> {
        self.data.ticks(count)
    }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self {
        LinearScale {
            data: self.data.clone().nice(),
            padding: self.padding,
            range_user_set: self.range_user_set,
            domain_user_set: self.domain_user_set,
        }
    }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset; an explicit value
    /// (including 0.0) overrides the default at render time.
    #[getter]
    fn padding(&self) -> Option<f64> {
        self.padding
    }

    /// Input domain as ``[min, max]``, or ``None`` when data-derived.
    #[getter]
    fn domain(&self) -> Option<Vec<f64>> {
        if self.domain_user_set { Some(self.data.domain.to_vec()) } else { None }
    }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.range_user_set { Some(self.data.range.to_vec()) } else { None }
    }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool {
        self.data.clamp
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], clamp: bool) -> LinearScaleData {
        LinearScaleData { domain, range, clamp }
    }

    #[test]
    fn linear_scale_basic() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        assert!((s.scale(5.0) - 0.5).abs() < 1e-12);
        assert!((s.scale(0.0) - 0.0).abs() < 1e-12);
        assert!((s.scale(10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_inversion_round_trip() {
        let s = d([-50.0, 50.0], [0.0, 100.0], false);
        for x in [-50.0, -25.0, 0.0, 17.5, 50.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn linear_out_of_domain_returns_nan_when_unclamped() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        assert!(s.scale(-1.0).is_nan());
        assert!(s.scale(11.0).is_nan());
    }

    #[test]
    fn linear_clamp_clamps_output() {
        let s = d([0.0, 10.0], [0.0, 1.0], true);
        assert_eq!(s.scale(-1.0), 0.0);
        assert_eq!(s.scale(11.0), 1.0);
    }

    #[test]
    fn linear_nan_propagates() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        assert!(s.scale(f64::NAN).is_nan());
        assert!(s.invert(f64::NAN).is_nan());
    }

    #[test]
    fn linear_ticks_default_count() {
        let s = d([0.0, 10.0], [0.0, 1.0], false);
        let t = s.ticks(10);
        assert!(t.len() >= 5, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn linear_nice_idempotent() {
        let s = d([0.13, 9.7], [0.0, 1.0], false);
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1, n2);
    }

    /// Named-field conversion (T2.5): a user-set domain/range round-trips
    /// through the PyO3 getters, and an unset domain reports `None` while still
    /// carrying the [0, 1] sentinel internally for render-time inference.
    #[test]
    fn linear_named_fields_round_trip() {
        let with_domain = LinearScale::new(
            Some(vec![2.0, 8.0]), Some(vec![0.0, 100.0]), true, false, Some(0.25),
        ).unwrap();
        assert_eq!(with_domain.domain(), Some(vec![2.0, 8.0]));
        assert_eq!(with_domain.range(), Some(vec![0.0, 100.0]));
        assert!(with_domain.clamp());
        assert_eq!(with_domain.padding(), Some(0.25));

        let no_domain = LinearScale::new(None, None, false, false, None).unwrap();
        assert_eq!(no_domain.domain(), None);
        assert_eq!(no_domain.range(), None);
        // Sentinel preserved internally so render-time inference can replace it.
        assert_eq!(no_domain.domain_pair(), [0.0, 1.0]);
    }

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Regression: major positions for [0, 10] must be exactly 0,1,2,...,10.
    #[test]
    fn linear_major_positions_unchanged() {
        let scale = LinearScale::new_internal(
            vec![0.0, 10.0], vec![0.0, 600.0], false, false,
        );
        let majors = scale.ticks_internal(10);
        assert_eq!(majors.first().copied(), Some(0.0));
        assert_eq!(majors.last().copied(), Some(10.0));
        assert_eq!(majors.len(), 11);
    }

    /// Minor count per major interval = DEFAULT_MINOR_SUBDIVISIONS - 1 = 4.
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10.  For domain
    /// [0, 10] that gives 11 major positions (0..=10), so 10 intervals × 4
    /// interior minors = 40 total minors.
    #[test]
    fn linear_minor_count_per_interval() {
        let scale = LinearScale::new_internal(
            vec![0.0, 10.0], vec![0.0, 600.0], false, false,
        );
        // minor_ticks_internal() uses the fixed major count of 10 internally.
        let majors = scale.ticks_internal(10);
        let minors = scale.minor_ticks_internal();
        // 10 intervals × (DEFAULT_MINOR_SUBDIVISIONS - 1) = 10 × 4 = 40
        let expected = (majors.len() - 1) * 4; // 40
        assert_eq!(minors.len(), expected, "minors: {minors:?}");
        assert!(minors.iter().all(|t| !t.is_major));
    }

    /// Minors lie strictly between consecutive major positions.
    #[test]
    fn linear_minors_strictly_between_majors() {
        let scale = LinearScale::new_internal(
            vec![0.0, 100.0], vec![0.0, 600.0], false, false,
        );
        // minor_ticks_internal() uses the fixed major count of 10 internally.
        let majors = scale.ticks_internal(10);
        let minors = scale.minor_ticks_internal();
        let major_set: std::collections::HashSet<u64> =
            majors.iter().map(|&v| v.to_bits()).collect();
        for m in &minors {
            assert!(
                !major_set.contains(&m.position.to_bits()),
                "minor at {} coincides with major",
                m.position,
            );
            assert!(m.position >= 0.0 && m.position <= 100.0, "minor out of domain: {}", m.position);
        }
    }
}
