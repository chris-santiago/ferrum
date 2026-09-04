use pyo3::prelude::*;

use super::core::{continuous_common, degenerate_ratio, resolve_continuous, scale_spec_to_py_dict};
use super::ticks::{minor_ticks_default, nice_step, nice_ticks, Tick};
use crate::spec::encoding::ScaleSpec;

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
        // Degenerate domain (d0 == d1, e.g. a constant-valued data column):
        // see `degenerate_ratio`'s doc comment (GH #104) for why this must
        // resolve to the range midpoint rather than 0/0 = NaN. `TimeScale`
        // shares this exact struct/method, so this guard covers it too.
        let t = degenerate_ratio(x - d0, d1 - d0);
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
/// reverse : bool, default False
///     Swap the resolved domain endpoints when this scale resolves inside a
///     chart render, producing a descending axis — equivalent, AT RENDER
///     TIME, to writing ``domain=[hi, lo]`` for an explicit domain (an
///     auto-inferred domain keeps its usual padding before the swap). The
///     swap applies only at render resolution: this object's own
///     ``scale()``/``invert()``/``ticks()`` and its ``domain`` getter keep
///     reporting the constructor's domain unchanged. This diverges from
///     ``PointScale``'s identically-named ``reverse``, which DOES apply
///     inside ``PointScale.scale()``.
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
    reverse: bool,
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
        LinearScale { data: d, padding: None, range_user_set: true, domain_user_set: true, reverse: false }
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
    // Wired to the render layer via `ScaleKind::minor_tick_fractions`
    // (`render/scale_resolve/mod.rs`, dispatched through `dispatch_continuous!`).
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

    /// This scale's domain endpoints rounded outward to "nice" values — the
    /// exact rounding `LinearScale(nice=True)` applies at construction
    /// (`LinearScaleData::nice`, `nice_step` with a count-10 tick target) —
    /// without mutating `self`. `ScaleKind::niced_domain` dispatches here so
    /// the chart-level `configure_axis(nice=True)` cascade
    /// (`scale_resolve::apply_axis_domain_config`) rounds identically to the
    /// encoding-level `Scale(nice=True)` surface, on THIS scale's own domain,
    /// rather than re-implementing linear rounding at the config seam.
    pub(crate) fn nice_domain_pair(&self) -> [f64; 2] {
        self.data.clone().nice().domain
    }

    /// Replace this scale's data-space domain in place, keeping its range and
    /// every kind-specific parameter.
    ///
    /// The sibling of [`domain_pair`](Self::domain_pair), added for the
    /// chart-level scale-domain config (D3, spec §4.2), which adjusts a
    /// RESOLVED domain rather than building a new scale — reconstructing via
    /// `new_internal` would have to re-supply parameters the caller cannot
    /// see. Because it is a second way into the domain field, it must apply
    /// whatever validation this kind's own constructor applies, or a config
    /// domain could store a value construction would have refused. `LinearScaleData` has no sanitizer — `new_internal` writes the pair straight through — so a raw write here matches construction exactly.
    ///
    /// `domain_user_set` flips to `true`: the domain now IS explicitly set (by
    /// the chart config), so `repr_string`/the `domain` getter must stop
    /// reporting it as data-derived.
    pub(crate) fn set_domain_pair(&mut self, domain: [f64; 2]) {
        self.data.domain = domain;
        self.domain_user_set = true;
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
        // `reverse` only appears when non-default (mirrors `TimeScale::repr_string`'s
        // `utc` prefix), so the default-shaped repr stays byte-identical to before.
        let reverse_s = if self.reverse { ", reverse=True" } else { "" };
        format!(
            "LinearScale(domain={}, range={}, clamp={}{})",
            domain_s, range_s, if *clamp { "True" } else { "False" }, reverse_s
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    ///
    /// `nice`/`zero` are always `false`: `nice` is baked into the domain at
    /// construction (no field survives) and `zero` is not a `LinearScale`
    /// concept — matching what the legacy `_scale_to_dict` omitted.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Linear {
            common: continuous_common(
                self.data.domain,
                self.domain_user_set,
                self.data.range,
                self.range_user_set,
                self.data.clamp,
                self.padding,
                self.reverse,
            ),
            nice: false,
            zero: false,
        }
    }
}

#[pymethods]
impl LinearScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, clamp = false, nice = false, padding = None, reverse = false))]
    fn new(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
        reverse: bool,
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
            reverse,
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
            reverse: self.reverse,
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

    /// Whether this scale's domain is swapped when it resolves inside a
    /// chart render (descending axis). Does not affect this object's own
    /// `scale`/`invert`/`ticks`/`domain` — unlike `PointScale::reverse`,
    /// which DOES apply inside `PointScale::scale`.
    #[getter]
    fn reverse(&self) -> bool {
        self.reverse
    }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
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

    /// #99/#104 residue: a degenerate equal-endpoint domain (`d0 == d1`,
    /// e.g. a constant-valued data column) used to divide by zero
    /// (`0/0 = NaN`) in the `t` ratio. It must instead resolve to the range
    /// midpoint — finite, never NaN — on both `clamp` arms. `TimeScale`
    /// reuses this exact struct, so this guard covers it too (see
    /// `time.rs::test_time_scale_degenerate_single_instant_domain_returns_range_midpoint`
    /// for a dedicated pin at that call site).
    #[test]
    fn linear_scale_degenerate_domain_returns_range_midpoint_not_nan() {
        let unclamped = d([5.0, 5.0], [0.0, 100.0], false);
        let mapped = unclamped.scale(5.0);
        assert!(mapped.is_finite(), "degenerate-domain scale() must be finite, got NaN");
        assert_eq!(mapped, 50.0, "degenerate domain must map to the range midpoint");

        let clamped = d([5.0, 5.0], [0.0, 100.0], true);
        let mapped_clamped = clamped.scale(5.0);
        assert!(mapped_clamped.is_finite(), "clamp=true must also be finite for a degenerate domain");
        assert_eq!(mapped_clamped, 50.0, "clamp=true degenerate domain must also map to the midpoint");
    }

    /// GH #104 quality-review remediation (S2-3): `degenerate_ratio` must key
    /// on *actual* degeneracy — both the numerator AND the denominator being
    /// zero (`x == d0 == d1`) — not a bare `denom == 0.0`. A degenerate
    /// domain (`d0 == d1`) queried at a DIFFERENT point (`x != d0`) is a
    /// genuine `k/0 = ±inf`, not a `0/0`, and must keep its pre-guard
    /// clamped-endpoint behavior on the `clamp == true` arm: `(+inf).clamp`
    /// saturates to the range's high endpoint, `(-inf).clamp` to the low
    /// endpoint — exactly what every affine-continuous scale already did for
    /// an out-of-domain value before `degenerate_ratio` existed. Tests both
    /// arms (`x` above and below the collapsed domain point) so a future
    /// widening of the `denom == 0.0` check alone (dropping the numerator
    /// half) breaks this immediately.
    #[test]
    fn linear_scale_degenerate_domain_off_point_keeps_clamped_endpoint_not_midpoint() {
        let clamped = d([5.0, 5.0], [0.0, 100.0], true);
        assert_eq!(
            clamped.scale(10.0), 100.0,
            "x above the collapsed domain point must clamp to the range's high endpoint (+inf.clamp), not the midpoint"
        );
        assert_eq!(
            clamped.scale(0.0), 0.0,
            "x below the collapsed domain point must clamp to the range's low endpoint (-inf.clamp), not the midpoint"
        );
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
            Some(vec![2.0, 8.0]), Some(vec![0.0, 100.0]), true, false, Some(0.25), false,
        ).unwrap();
        assert_eq!(with_domain.domain(), Some(vec![2.0, 8.0]));
        assert_eq!(with_domain.range(), Some(vec![0.0, 100.0]));
        assert!(with_domain.clamp());
        assert_eq!(with_domain.padding(), Some(0.25));
        assert!(!with_domain.reverse());

        let no_domain = LinearScale::new(None, None, false, false, None, false).unwrap();
        assert_eq!(no_domain.domain(), None);
        assert_eq!(no_domain.range(), None);
        // Sentinel preserved internally so render-time inference can replace it.
        assert_eq!(no_domain.domain_pair(), [0.0, 1.0]);
    }

    // ── `reverse` kwarg (F-L04-07, batch-C task 2) ──────────────────────────

    /// `reverse=True` round-trips through the getter and into the wire
    /// `ScaleSpec::Linear.common.reverse` bit unswapped — the actual domain
    /// swap happens downstream at the resolver (`apply_domain_reverse`), not
    /// here (see `continuous_common`'s doc).
    #[test]
    fn linear_reverse_round_trips_through_to_scale_spec() {
        let s = LinearScale::new(Some(vec![2.0, 8.0]), None, false, false, None, true).unwrap();
        assert!(s.reverse());
        match s.to_scale_spec() {
            ScaleSpec::Linear { common, .. } => {
                assert!(common.reverse, "reverse=True must survive to the wire spec");
                // The domain itself is NOT swapped by the pyclass — only the flag travels.
                assert_eq!(common.domain, Some(vec![2.0, 8.0]));
            }
            other => panic!("expected ScaleSpec::Linear, got {other:?}"),
        }
    }

    /// Default (`reverse` unset) emits `reverse: false` on the spec, and the
    /// wire serialization omits the key entirely (`skip_serializing_if`),
    /// matching every pre-existing baseline in `test_scale_spec_parity`.
    #[test]
    fn linear_reverse_default_emits_no_reverse_key() {
        let s = LinearScale::new(None, None, false, false, None, false).unwrap();
        assert!(!s.reverse());
        match s.to_scale_spec() {
            ScaleSpec::Linear { common, .. } => assert!(!common.reverse),
            other => panic!("expected ScaleSpec::Linear, got {other:?}"),
        }
        let json = serde_json::to_string(&s.to_scale_spec()).unwrap();
        assert!(!json.contains("reverse"), "default reverse must not appear on the wire: {json}");
    }

    /// `repr_string()` pinned in both directions (quality-review F2): the
    /// default-shaped repr is byte-identical to before this change (no
    /// `reverse` mention at all — not even `reverse=False`), and
    /// `reverse=True` appends the exact `, reverse=True)` suffix.
    #[test]
    fn linear_repr_pins_both_reverse_branches() {
        let default_scale = LinearScale::new(None, None, false, false, None, false).unwrap();
        assert_eq!(
            default_scale.repr_string(),
            "LinearScale(domain=None, range=None, clamp=False)",
        );

        let reversed_scale = LinearScale::new(None, None, false, false, None, true).unwrap();
        assert_eq!(
            reversed_scale.repr_string(),
            "LinearScale(domain=None, range=None, clamp=False, reverse=True)",
        );
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
