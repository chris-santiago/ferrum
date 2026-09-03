use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{continuous_common, degenerate_ratio, resolve_continuous, scale_spec_to_py_dict};
use super::ticks::{minor_ticks_default, nice_step, nice_ticks, Tick};
use crate::spec::encoding::ScaleSpec;

fn symlog_fwd(x: f64, c: f64) -> f64 {
    x.signum() * (x.abs() / c).ln_1p()
}

fn symlog_inv(y: f64, c: f64) -> f64 {
    y.signum() * c * (y.abs().exp() - 1.0)
}

#[derive(Debug, Clone, PartialEq)]
struct SymlogScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    constant: f64,
    clamp: bool,
}

impl SymlogScaleData {
    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let f = |v: f64| symlog_fwd(v, self.constant);
        // Degenerate domain (d0 == d1, e.g. a constant-valued data column):
        // see `degenerate_ratio`'s doc comment (GH #104) for why this must
        // resolve to the range midpoint rather than 0/0 = NaN.
        let t = degenerate_ratio(f(x) - f(d0), f(d1) - f(d0));
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
        let f = |v: f64| symlog_fwd(v, self.constant);
        let t = (y - r0) / (r1 - r0);
        let lmapped = f(d0) + t * (f(d1) - f(d0));
        let mapped = symlog_inv(lmapped, self.constant);
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
        Self { domain: new_domain, range: self.range, constant: self.constant, clamp: self.clamp }
    }
}

/// Symmetric logarithmic scale.
///
/// Maps a numeric domain — including zero and negative values — to a range
/// using a bi-symmetric log transform. The transformation is linear in
/// ``[-constant, +constant]`` and logarithmic outside that band, so zero and
/// sign changes are handled without special-casing.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``. May span zero or be entirely negative.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// constant : float, default 1.0
///     Half-width of the linear region around zero. Must be finite and
///     positive.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to "nice" values for tick generation.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         y=fr.Y("delta", scale=fr.SymlogScale(domain=[-1000, 1000], range=[400, 0]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct SymlogScale {
    data: SymlogScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    domain_user_set: bool,
}

impl SymlogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, constant: f64, clamp: bool, nice: bool) -> Self {
        let mut d = SymlogScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            constant,
            clamp,
        };
        if nice { d = d.nice(); }
        SymlogScale { data: d, padding: None, range_user_set: true, domain_user_set: true }
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 { self.data.scale(x) }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return minor ticks subdivided in **symlog-transformed** space.
    ///
    /// Symlog maps raw values via `sign(x) * ln(1 + |x|/c)`.  Major ticks
    /// from `nice_ticks` are evenly spaced in the raw domain; to produce
    /// visually-uniform minors this method converts each major to symlog space,
    /// subdivides uniformly there, and maps back via the symlog inverse.
    ///
    /// The major tick count is fixed at 10 (the conventional default).  Minor
    /// tick density is always `DEFAULT_MINOR_SUBDIVISIONS` (5 sub-intervals →
    /// 4 interior minors per gap); there is no per-call override.
    // Wired to the render layer via `ScaleKind::minor_tick_fractions`
    // (`render/scale_resolve/mod.rs`, dispatched through `dispatch_continuous!`).
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.data.ticks(10);
        let c = self.data.constant;
        let fwd = move |v: f64| symlog_fwd(v, c);
        let inv = move |t: f64| symlog_inv(t, c);
        let transformed: Vec<f64> = majors.iter().map(|&v| fwd(v)).collect();
        minor_ticks_default(&transformed, inv)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] { self.data.range }

    pub(crate) fn domain_pair(&self) -> [f64; 2] { self.data.domain }

    /// This scale's domain endpoints rounded outward to "nice" values — the
    /// exact rounding `SymlogScale(nice=True)` applies at construction — see
    /// [`LinearScale::nice_domain_pair`](super::linear::LinearScale::nice_domain_pair)
    /// for why `ScaleKind::niced_domain` dispatches to a kind-specific
    /// method like this one instead of a shared inline rounding.
    pub(crate) fn nice_domain_pair(&self) -> [f64; 2] { self.data.clone().nice().domain }

    /// Replace this scale's data-space domain in place, keeping its range and
    /// every kind-specific parameter.
    ///
    /// The sibling of [`domain_pair`](Self::domain_pair), added for the
    /// chart-level scale-domain config (D3, spec §4.2), which adjusts a
    /// RESOLVED domain rather than building a new scale — reconstructing via
    /// `new_internal` would have to re-supply parameters the caller cannot
    /// see. Because it is a second way into the domain field, it must apply
    /// whatever validation this kind's own constructor applies, or a config
    /// domain could store a value construction would have refused. `SymlogScaleData` has no sanitizer (symlog is defined through zero, so a zero endpoint is legal here) — `new_internal` writes the pair straight through.
    ///
    /// `domain_user_set` flips to `true`: the domain now IS explicitly set (by
    /// the chart config), so `repr_string`/the `domain` getter must stop
    /// reporting it as data-derived.
    pub(crate) fn set_domain_pair(&mut self, domain: [f64; 2]) {
        self.data.domain = domain;
        self.domain_user_set = true;
    }

    pub(crate) fn repr_string(&self) -> String {
        let SymlogScaleData { domain, range, constant, clamp } = &self.data;
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
            "SymlogScale(domain={}, range={}, constant={}, clamp={})",
            domain_s, range_s, constant, if *clamp { "True" } else { "False" }
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    /// `nice` is baked into the domain at construction, so it is always `false`
    /// here — matching what the legacy `_scale_to_dict` omitted.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Symlog {
            constant: self.data.constant,
            common: continuous_common(
                self.data.domain,
                self.domain_user_set,
                self.data.range,
                self.range_user_set,
                self.data.clamp,
                self.padding,
            ),
            nice: false,
        }
    }
}

#[pymethods]
impl SymlogScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, constant = 1.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        constant: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        if !constant.is_finite() || constant <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "constant must be finite and > 0; got {constant}"
            )));
        }
        // Sentinel [-1.0, 1.0] when no domain supplied; render-time inference
        // replaces it before any scale computation occurs.
        let resolved = resolve_continuous(domain, range, [-1.0, 1.0])?;
        let mut d = SymlogScaleData {
            domain: resolved.domain,
            range: resolved.range,
            constant,
            clamp,
        };
        if nice && resolved.domain_user_set {
            d = d.nice();
        }
        Ok(SymlogScale {
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
        SymlogScale {
            data: self.data.clone().nice(),
            padding: self.padding,
            range_user_set: self.range_user_set,
            domain_user_set: self.domain_user_set,
        }
    }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
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

    /// Half-width of the linear region around zero (default 1.0).
    #[getter]
    fn constant(&self) -> f64 { self.data.constant }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.data.clamp }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool) -> SymlogScaleData {
        SymlogScaleData { domain, range, constant, clamp }
    }

    #[test]
    fn symlog_scale_handles_zero() {
        let s = d([-100.0, 100.0], [0.0, 1.0], 1.0, false);
        let y = s.scale(0.0);
        assert!(y.is_finite(), "scale(0) returned {y}");
        assert!((y - 0.5).abs() < 1e-12, "expected 0.5, got {y}");
    }

    /// #99/#104 residue: a degenerate equal-endpoint domain (`d0 == d1`,
    /// e.g. a constant-valued data column) used to divide by zero
    /// (`0/0 = NaN`) in the `t` ratio. It must instead resolve to the range
    /// midpoint — finite, never NaN — on both `clamp` arms.
    #[test]
    fn symlog_scale_degenerate_domain_returns_range_midpoint_not_nan() {
        let unclamped = d([5.0, 5.0], [0.0, 100.0], 1.0, false);
        let mapped = unclamped.scale(5.0);
        assert!(mapped.is_finite(), "degenerate-domain scale() must be finite, got NaN");
        assert_eq!(mapped, 50.0, "degenerate domain must map to the range midpoint");

        let clamped = d([5.0, 5.0], [0.0, 100.0], 1.0, true);
        let mapped_clamped = clamped.scale(5.0);
        assert!(mapped_clamped.is_finite(), "clamp=true must also be finite for a degenerate domain");
        assert_eq!(mapped_clamped, 50.0, "clamp=true degenerate domain must also map to the midpoint");
    }

    #[test]
    fn symlog_inversion_round_trip_across_zero() {
        let s = d([-1000.0, 1000.0], [0.0, 1.0], 1.0, false);
        for x in [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-6, "round-trip failed at x={x}: got {back}");
        }
    }

    /// Named-field conversion (T2.5): user-set domain/range/constant round-trip,
    /// and an unset domain reports `None` while keeping the [-1, 1] sentinel.
    #[test]
    fn symlog_named_fields_round_trip() {
        let with_domain = SymlogScale::new(
            Some(vec![-1000.0, 1000.0]), Some(vec![0.0, 400.0]), 1.0, false, false, Some(0.2),
        ).unwrap();
        assert_eq!(with_domain.domain(), Some(vec![-1000.0, 1000.0]));
        assert_eq!(with_domain.range(), Some(vec![0.0, 400.0]));
        assert_eq!(with_domain.constant(), 1.0);
        assert_eq!(with_domain.padding(), Some(0.2));

        let no_domain = SymlogScale::new(None, None, 5.0, false, false, None).unwrap();
        assert_eq!(no_domain.domain(), None);
        assert_eq!(no_domain.range(), None);
        assert_eq!(no_domain.domain_pair(), [-1.0, 1.0]);
        assert_eq!(no_domain.constant(), 5.0);
    }

    #[test]
    fn symlog_constant_changes_curvature() {
        // Larger constant → more linear (less compression) → x=50 on [-100,100] maps closer to 0.5
        // Smaller constant → more log-like → x=50 maps closer to 1.0 (compressed toward endpoint)
        let s1 = d([-100.0, 100.0], [0.0, 1.0], 1.0,   false);
        let s2 = d([-100.0, 100.0], [0.0, 1.0], 100.0, false);
        let y1 = s1.scale(50.0);
        let y2 = s2.scale(50.0);
        assert!(y1 > y2, "expected y1={y1} > y2={y2}: smaller constant compresses more");
        assert!(y1 > 0.5 && y1 < 1.0, "y1={y1} out of (0.5,1.0)");
        assert!(y2 > 0.5 && y2 < 1.0, "y2={y2} out of (0.5,1.0)");
    }

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Regression: symlog major positions are unchanged after adding minor support.
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10 internally.
    #[test]
    fn symlog_major_positions_unchanged() {
        let scale = SymlogScale::new_internal(
            vec![-100.0, 100.0], vec![0.0, 600.0], 1.0, false, false,
        );
        // minor_ticks_internal uses count=10 for majors; compare at the same count.
        let before = scale.ticks_internal(10);
        let _ = scale.minor_ticks_internal();
        let after = scale.ticks_internal(10);
        assert_eq!(before, after);
    }

    /// Symlog minors must be evenly spaced in the SYMLOG-transformed space,
    /// not in the raw data domain.  In transformed space minor gaps are uniform;
    /// in raw domain they are non-uniform (closer together near zero).
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10.
    #[test]
    fn symlog_minors_evenly_spaced_in_transformed_space() {
        let constant = 1.0_f64;
        let scale = SymlogScale::new_internal(
            vec![-100.0, 100.0], vec![0.0, 600.0], constant, false, false,
        );
        // minor_ticks_internal uses count=10 for majors.
        let majors = scale.ticks_internal(10);
        let minors = scale.minor_ticks_internal();

        assert!(!minors.is_empty(), "expected non-empty symlog minors");

        // Minors must not coincide with majors.
        let major_set: std::collections::HashSet<u64> =
            majors.iter().map(|&v| v.to_bits()).collect();
        for m in &minors {
            assert!(!major_set.contains(&m.position.to_bits()),
                "symlog minor at {} coincides with major", m.position);
            assert!(!m.is_major);
            assert!(m.position.is_finite(), "symlog minor is non-finite");
        }

        // Verify uniformity in symlog space for the first major interval.
        // DEFAULT_MINOR_SUBDIVISIONS = 5 → step = (t1-t0)/5, 4 interior minors.
        if majors.len() >= 2 {
            let t0 = symlog_fwd(majors[0], constant);
            let t1 = symlog_fwd(majors[1], constant);
            let step = (t1 - t0) / 5.0; // 5 = DEFAULT_MINOR_SUBDIVISIONS
            // Interior transformed positions: t0+step, t0+2*step, t0+3*step, t0+4*step.
            let group_minors: Vec<f64> = minors
                .iter()
                .filter(|m| {
                    m.position > majors[0].min(majors[1]) &&
                    m.position < majors[0].max(majors[1])
                })
                .map(|m| symlog_fwd(m.position, constant))
                .collect();
            assert_eq!(group_minors.len(), 4,
                "expected 4 interior minors in first interval, got {}: {group_minors:?}",
                group_minors.len());
            for (i, &got) in group_minors.iter().enumerate() {
                let expected = t0 + (i as f64 + 1.0) * step;
                assert!(
                    (got - expected).abs() < 1e-9,
                    "symlog minor {i} in transformed space: expected {expected:.6}, got {got:.6}"
                );
            }
        }
    }
}
