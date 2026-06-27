use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{continuous_common, resolve_continuous, scale_spec_to_py_dict};
use super::ticks::{minor_ticks_default, nice_step, nice_ticks, Tick};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
struct PowScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    exponent: f64,
    clamp: bool,
}

impl PowScaleData {
    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let pow_fwd = |v: f64| v.signum() * v.abs().powf(self.exponent);
        let t = (pow_fwd(x) - pow_fwd(d0)) / (pow_fwd(d1) - pow_fwd(d0));
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

    fn invert(&self, y: f64) -> f64 {
        if y.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let pow_fwd = |v: f64| v.signum() * v.abs().powf(self.exponent);
        let pow_inv = |v: f64| v.signum() * v.abs().powf(1.0 / self.exponent);
        let t = (y - r0) / (r1 - r0);
        let lmapped = pow_fwd(d0) + t * (pow_fwd(d1) - pow_fwd(d0));
        let mapped = pow_inv(lmapped);
        if self.clamp {
            let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
            mapped.clamp(lo, hi)
        } else if y < r0.min(r1) || y > r0.max(r1) {
            f64::NAN
        } else {
            mapped
        }
    }

    fn ticks(&self, count: usize) -> Vec<f64> {
        nice_ticks(self.domain[0], self.domain[1], count)
    }

    fn nice(self) -> Self {
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
        Self { domain: new_domain, range: self.range, exponent: self.exponent, clamp: self.clamp }
    }
}

/// Continuous power scale.
///
/// Maps a numeric domain to a numeric range via a power transformation
/// (x^exponent). Useful for data where perceptual linearity requires
/// non-linear scaling (e.g., bubble area encoding with exponent=0.5).
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// exponent : float, default 2.0
///     The power exponent. Must be finite and positive.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to "nice" values for tick generation.
/// padding : float, optional
///     Fractional inward pixel padding.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct PowScale {
    data: PowScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    domain_user_set: bool,
}

impl PowScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, exponent: f64, clamp: bool) -> Self {
        let d = PowScaleData {
            domain: [domain[0], domain[1]],
            range: [range[0], range[1]],
            exponent,
            clamp,
        };
        PowScale { data: d, padding: None, range_user_set: true, domain_user_set: true }
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.data.scale(x)
    }

    /// Crate-internal tick call.
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.data.ticks(count)
    }

    /// Return minor ticks subdivided in the **power-transformed** space.
    ///
    /// Major ticks from `nice_ticks` are evenly spaced in the raw domain, but
    /// they map to non-uniform pixel positions because the power transform is
    /// non-linear.  To produce visually-uniform minor ticks, this method
    /// converts each major position to power space (`x^exponent`), subdivides
    /// uniformly there, and maps back via the inverse (`y^(1/exponent)`).
    ///
    /// The major tick count is fixed at 10 (the conventional default).  Minor
    /// tick density is always `DEFAULT_MINOR_SUBDIVISIONS` (5 sub-intervals →
    /// 4 interior minors per gap); there is no per-call override.
    // Wired to the render layer via `ScaleKind::minor_tick_fractions`
    // (`render/scale_resolve/mod.rs`, dispatched through `dispatch_continuous!`).
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.data.ticks(10);
        let exp = self.data.exponent;
        // Forward: data → transformed (same sign preserved via signum).
        let fwd = |v: f64| v.signum() * v.abs().powf(exp);
        // Inverse: transformed → data.
        let inv = move |t: f64| t.signum() * t.abs().powf(1.0 / exp);
        let transformed: Vec<f64> = majors.iter().map(|&v| fwd(v)).collect();
        minor_ticks_default(&transformed, inv)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.data.range
    }

    pub(crate) fn domain_pair(&self) -> [f64; 2] {
        self.data.domain
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    /// The `Pow` variant carries no `nice` field (nice is baked into the domain
    /// at construction), matching what the legacy `_scale_to_dict` emitted.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Pow {
            exponent: self.data.exponent,
            common: continuous_common(
                self.data.domain,
                self.domain_user_set,
                self.data.range,
                self.range_user_set,
                self.data.clamp,
                self.padding,
            ),
        }
    }

    fn repr_string(&self) -> String {
        let PowScaleData { domain, range, exponent, clamp } = &self.data;
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
            "PowScale(domain={}, range={}, exponent={}, clamp={})",
            domain_s, range_s, exponent, if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl PowScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, exponent = 2.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        exponent: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        if !exponent.is_finite() || exponent <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "exponent must be finite and > 0; got {exponent}"
            )));
        }
        // Sentinel [0.0, 1.0] when no domain supplied; render-time inference
        // replaces it before any scale computation occurs.
        let resolved = resolve_continuous(domain, range, [0.0, 1.0])?;
        let mut d = PowScaleData {
            domain: resolved.domain,
            range: resolved.range,
            exponent,
            clamp,
        };
        if nice && resolved.domain_user_set {
            d = d.nice();
        }
        Ok(PowScale {
            data: d,
            padding,
            range_user_set: resolved.range_user_set,
            domain_user_set: resolved.domain_user_set,
        })
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.data.scale(x) }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.data.invert(y) }

    /// Return approximately ``count`` tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self {
        PowScale {
            data: self.data.clone().nice(),
            padding: self.padding,
            range_user_set: self.range_user_set,
            domain_user_set: self.domain_user_set,
        }
    }

    /// Fractional inward pixel padding.
    #[getter]
    fn padding(&self) -> Option<f64> { self.padding }

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

    /// The power exponent.
    #[getter]
    fn exponent(&self) -> f64 { self.data.exponent }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.data.clamp }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

/// Continuous square-root scale (convenience for PowScale with exponent=0.5).
///
/// Equivalent to ``PowScale(exponent=0.5, ...)``. Commonly used for area
/// encodings where perceived size should scale linearly with value.
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
/// padding : float, optional
///     Fractional inward pixel padding.
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SqrtScale {
    data: PowScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    domain_user_set: bool,
}

impl SqrtScale {
    /// Crate-internal scale call (no PyO3 boundary).
    // Dead on the render path by design: `SqrtScale` is a PyO3 user-query facade
    // (see the `scale/mod.rs` module doc). A `ScaleSpec::Sqrt` resolves to a
    // `PowScale` (exponent 0.5) at render, so these `_internal` helpers exist only
    // for direct Python `.scale()`/`.ticks()` queries. Kept as part of the PyO3 API
    // surface; the allow suppresses the render-side dead-code lint.
    #[allow(dead_code)]
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.data.scale(x)
    }

    /// Crate-internal tick call.
    // Dead on the render path by design (see `scale_internal` above): render uses
    // `PowScale`, this serves direct Python `.ticks()` queries on the PyO3 facade.
    #[allow(dead_code)]
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.data.ticks(count)
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `ScaleSpec::Sqrt` carries no `exponent` field — the exponent is implicitly
    /// 0.5 and is resolved as `ScaleKind::Pow(0.5)` on the render path. The legacy
    /// `_scale_to_dict` emitted an `"exponent": 0.5` key, but the deserialiser
    /// dropped it (the `Sqrt` variant has no such field); omitting it here yields
    /// the identical stored `ScaleSpec`.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Sqrt {
            common: continuous_common(
                self.data.domain,
                self.domain_user_set,
                self.data.range,
                self.range_user_set,
                self.data.clamp,
                self.padding,
            ),
        }
    }
}

#[pymethods]
impl SqrtScale {
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
        let mut d = PowScaleData {
            domain: resolved.domain,
            range: resolved.range,
            exponent: 0.5,
            clamp,
        };
        if nice && resolved.domain_user_set {
            d = d.nice();
        }
        Ok(SqrtScale {
            data: d,
            padding,
            range_user_set: resolved.range_user_set,
            domain_user_set: resolved.domain_user_set,
        })
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.data.scale(x) }

    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.data.invert(y) }

    /// Return approximately ``count`` tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self {
        SqrtScale {
            data: self.data.clone().nice(),
            padding: self.padding,
            range_user_set: self.range_user_set,
            domain_user_set: self.domain_user_set,
        }
    }

    /// Fractional inward pixel padding.
    #[getter]
    fn padding(&self) -> Option<f64> { self.padding }

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

    /// The power exponent (always 0.5 for SqrtScale).
    #[getter]
    fn exponent(&self) -> f64 { 0.5 }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.data.clamp }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String {
        let PowScaleData { domain, range, clamp, .. } = &self.data;
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
            "SqrtScale(domain={}, range={}, clamp={})",
            domain_s, range_s, if *clamp { "True" } else { "False" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], exponent: f64, clamp: bool) -> PowScaleData {
        PowScaleData { domain, range, exponent, clamp }
    }

    #[test]
    fn pow_scale_basic() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        // x=0 => 0^2=0, t=0/(100^2 - 0) = 0 => 0.0
        assert!((s.scale(0.0) - 0.0).abs() < 1e-12);
        // x=100 => 100^2=10000, t=10000/10000=1 => 1.0
        assert!((s.scale(100.0) - 1.0).abs() < 1e-12);
        // x=50 => 50^2=2500, t=2500/10000=0.25
        assert!((s.scale(50.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pow_scale_inversion_round_trip() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        for x in [0.0, 10.0, 25.0, 50.0, 75.0, 100.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn sqrt_scale_has_exponent_half() {
        let s = d([0.0, 100.0], [0.0, 1.0], 0.5, false);
        // x=25 => 25^0.5=5, domain: 0^0.5=0, 100^0.5=10, t=5/10=0.5
        assert!((s.scale(25.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pow_scale_clamp() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, true);
        assert_eq!(s.scale(-10.0), 0.0);
        assert_eq!(s.scale(200.0), 1.0);
    }

    #[test]
    fn pow_scale_out_of_domain_nan() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        assert!(s.scale(-1.0).is_nan());
        assert!(s.scale(101.0).is_nan());
    }

    #[test]
    fn pow_scale_nan_propagates() {
        let s = d([0.0, 100.0], [0.0, 1.0], 2.0, false);
        assert!(s.scale(f64::NAN).is_nan());
        assert!(s.invert(f64::NAN).is_nan());
    }

    #[test]
    fn pow_scale_to_dict_exponent() {
        // Verify that PowScale stores exponent correctly
        let s = PowScale {
            data: PowScaleData { domain: [0.0, 10.0], range: [0.0, 1.0], exponent: 3.0, clamp: false },
            padding: None,
            range_user_set: true,
            domain_user_set: true,
        };
        assert_eq!(s.data.exponent, 3.0);
    }

    #[test]
    fn sqrt_scale_exponent_is_half() {
        let s = SqrtScale {
            data: PowScaleData { domain: [0.0, 100.0], range: [0.0, 1.0], exponent: 0.5, clamp: false },
            padding: None,
            range_user_set: true,
            domain_user_set: true,
        };
        assert_eq!(s.data.exponent, 0.5);
    }

    /// Named-field conversion (T2.5): Pow/Sqrt user-set domain/range round-trip,
    /// and an unset domain reports `None` while keeping the [0, 1] sentinel.
    #[test]
    fn pow_sqrt_named_fields_round_trip() {
        let pow = PowScale::new(
            Some(vec![0.0, 100.0]), Some(vec![0.0, 600.0]), 2.0, false, false, Some(0.05),
        ).unwrap();
        assert_eq!(pow.domain(), Some(vec![0.0, 100.0]));
        assert_eq!(pow.range(), Some(vec![0.0, 600.0]));
        assert_eq!(pow.exponent(), 2.0);
        assert_eq!(pow.padding(), Some(0.05));

        let pow_none = PowScale::new(None, None, 2.0, false, false, None).unwrap();
        assert_eq!(pow_none.domain(), None);
        assert_eq!(pow_none.range(), None);
        assert_eq!(pow_none.domain_pair(), [0.0, 1.0]);

        let sqrt = SqrtScale::new(
            Some(vec![0.0, 49.0]), Some(vec![0.0, 7.0]), false, false, None,
        ).unwrap();
        assert_eq!(sqrt.domain(), Some(vec![0.0, 49.0]));
        assert_eq!(sqrt.range(), Some(vec![0.0, 7.0]));
        assert_eq!(sqrt.exponent(), 0.5);
    }

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Regression: pow scale major positions must be unchanged by minor tick addition.
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10 internally;
    /// the explicit `ticks_internal(5)` call here uses count=5.  We compare
    /// the major list at count=10 (what minor_ticks_internal uses) to verify
    /// no mutation occurs.
    #[test]
    fn pow_major_positions_unchanged() {
        let scale = PowScale::new_internal(vec![0.0, 100.0], vec![0.0, 600.0], 2.0, false);
        let before = scale.ticks_internal(10);
        let _ = scale.minor_ticks_internal();
        let after = scale.ticks_internal(10);
        assert_eq!(before, after);
    }

    /// Pow(exponent=2) minors must be evenly spaced in the TRANSFORMED (squared) space,
    /// not in the raw data domain.
    ///
    /// For [0,100] with exponent=2, the transformed range is [0,10000].
    /// `minor_ticks_internal()` uses the fixed major count of 10 internally.
    /// Major ticks at (nice_ticks with count=10 in raw space) e.g. 0,10,20,...,100.
    /// In transformed space: 0,100,400,900,...,10000 — NOT uniform.
    /// Minors are computed by subdividing uniformly in transformed space, then
    /// inverting; in the raw domain they will be at non-uniform positions.
    #[test]
    fn pow_minors_evenly_spaced_in_transformed_space() {
        let scale = PowScale::new_internal(vec![0.0, 100.0], vec![0.0, 600.0], 2.0, false);
        // minor_ticks_internal uses fixed count=10 for majors.
        let majors = scale.ticks_internal(10);
        let minors = scale.minor_ticks_internal();

        // Minors should be non-empty.
        assert!(!minors.is_empty(), "expected non-empty minors for pow scale");

        // All minors must be within domain and not coincide with majors.
        let major_set: std::collections::HashSet<u64> =
            majors.iter().map(|&v| v.to_bits()).collect();
        for m in &minors {
            assert!(!major_set.contains(&m.position.to_bits()),
                "minor at {} coincides with major", m.position);
            assert!(m.position >= 0.0 && m.position <= 100.0,
                "minor out of domain: {}", m.position);
            assert!(!m.is_major);
        }

        // Verify the minors are evenly spaced in transformed (squared) space:
        // Pick the first group (between major[0] and major[1]).
        if majors.len() >= 2 {
            let m0 = majors[0].signum() * majors[0].abs().powf(2.0);
            let m1 = majors[1].signum() * majors[1].abs().powf(2.0);
            // DEFAULT_MINOR_SUBDIVISIONS = 5 → 4 interior positions
            let step = (m1 - m0) / 5.0;
            let expected_transformed: Vec<f64> = (1..5).map(|i| m0 + i as f64 * step).collect();
            let minor_positions_in_group: Vec<f64> = minors
                .iter()
                .filter(|t| t.position > majors[0] && t.position < majors[1])
                .map(|t| t.position.signum() * t.position.abs().powf(2.0))
                .collect();
            assert_eq!(
                minor_positions_in_group.len(),
                expected_transformed.len(),
                "wrong count of minors in first interval"
            );
            for (got, exp) in minor_positions_in_group.iter().zip(&expected_transformed) {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "minor in transformed space: expected {exp}, got {got}"
                );
            }
        }
    }

}
