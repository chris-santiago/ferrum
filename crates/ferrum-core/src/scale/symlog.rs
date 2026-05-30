use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::ticks::{minor_ticks_default, nice_step, nice_ticks, Tick};

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
        let t = (f(x) - f(d0)) / (f(d1) - f(d0));
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
pub struct SymlogScale(SymlogScaleData, Option<f64>, bool);

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
        SymlogScale(d, None, true)
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 { self.0.scale(x) }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

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
    // Wired to the render layer in Task 2 of the grid subsystem.
    #[allow(dead_code)]
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.0.ticks(10);
        let c = self.0.constant;
        let fwd = move |v: f64| symlog_fwd(v, c);
        let inv = move |t: f64| symlog_inv(t, c);
        let transformed: Vec<f64> = majors.iter().map(|&v| fwd(v)).collect();
        minor_ticks_default(&transformed, inv)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] { self.0.range }

    pub(crate) fn domain_pair(&self) -> [f64; 2] { self.0.domain }

    pub(crate) fn repr_string(&self) -> String {
        let SymlogScaleData { domain, range, constant, clamp } = &self.0;
        format!(
            "SymlogScale(domain=[{}, {}], range=[{}, {}], constant={}, clamp={})",
            domain[0], domain[1], range[0], range[1], constant, if *clamp { "True" } else { "False" }
        )
    }
}

#[pymethods]
impl SymlogScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, constant = 1.0, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Option<Vec<f64>>,
        constant: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        let range_user_set = range.is_some();
        let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
        validate_continuous_pair(&domain, &r)?;
        if !constant.is_finite() || constant <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "constant must be finite and > 0; got {constant}"
            )));
        }
        let mut d = SymlogScaleData {
            domain: [domain[0], domain[1]],
            range:  [r[0],  r[1]],
            constant,
            clamp,
        };
        if nice {
            d = d.nice();
        }
        Ok(SymlogScale(d, padding, range_user_set))
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }
    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` tick values within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.0.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to "nice" values.
    fn nice(&self) -> Self { SymlogScale(self.0.clone().nice(), self.1, self.2) }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.1 }

    /// Input domain as ``[min, max]``.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.to_vec() }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.2 { Some(self.0.range.to_vec()) } else { None }
    }

    /// Half-width of the linear region around zero (default 1.0).
    #[getter]
    fn constant(&self) -> f64 { self.0.constant }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

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

    #[test]
    fn symlog_inversion_round_trip_across_zero() {
        let s = d([-1000.0, 1000.0], [0.0, 1.0], 1.0, false);
        for x in [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back - x).abs() < 1e-6, "round-trip failed at x={x}: got {back}");
        }
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
