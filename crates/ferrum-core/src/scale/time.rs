use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::linear::LinearScaleData;
use super::ticks::{calendar_ticks, minor_ticks_default, nice_calendar_interval, nice_time_interval_ms, CalendarInterval, Tick};

/// Continuous temporal scale backed by Unix epoch milliseconds.
///
/// Maps an epoch-millisecond domain to a numeric range. Tick generation
/// uses time-aware "nice" intervals (seconds, minutes, hours, days, months,
/// years) rather than purely numeric rounding. Domain values are
/// floating-point epoch milliseconds (UTC).
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[t_min, t_max]`` in epoch milliseconds (UTC).
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Extend domain endpoints to the nearest calendar interval boundary.
///
/// Examples
/// --------
/// Ferrum converts datetime columns automatically; a ``TimeScale`` is
/// constructed implicitly when the channel data type is temporal::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(x=fr.X("date:T"))
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct TimeScale(LinearScaleData, Option<f64>, bool, bool);

impl TimeScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `TimeScale` reuses [`LinearScaleData`]: the domain-to-range mapping is
    /// affine over epoch milliseconds, only tick/nice behavior is time-aware.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let inner = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale(inner, None, true, false);
        if nice { s.time_nice() } else { s }
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale(x)
    }

    /// Crate-internal tick call (uses time-aware nice intervals).
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.time_ticks(count)
    }

    /// Return minor ticks subdivided uniformly between major calendar ticks.
    ///
    /// Time is already in a linear (epoch-ms) space, so the default
    /// subdivision algorithm applies directly: each major interval is divided
    /// into `DEFAULT_MINOR_SUBDIVISIONS` sub-intervals in epoch-ms, yielding
    /// 4 interior minor ticks per major gap.
    ///
    /// The major tick count is fixed at 10 (the conventional default).  Minor
    /// tick density is always `DEFAULT_MINOR_SUBDIVISIONS` (5 sub-intervals →
    /// 4 interior minors per gap); there is no per-call override.
    // Wired to the render layer in Task 2 of the grid subsystem.
    #[allow(dead_code)]
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.time_ticks(10);
        // Time domain is already linear (epoch-ms), so the identity transform
        // gives visually-uniform minor ticks.
        minor_ticks_default(&majors, |x| x)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.0.range
    }

    pub(crate) fn domain_pair(&self) -> [f64; 2] {
        self.0.domain
    }

    pub(crate) fn repr_string(&self) -> String {
        let LinearScaleData { domain, range, clamp } = &self.0;
        let prefix = if self.3 { "TimeScale(utc=True, " } else { "TimeScale(" };
        format!(
            "{}domain=[{}, {}], range=[{}, {}], clamp={})",
            prefix, domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
        )
    }

    fn time_ticks(&self, count: usize) -> Vec<f64> {
        let [d0, d1] = self.0.domain;
        // Pass domain values directly; calendar_ticks handles reversal when d0 > d1.
        calendar_ticks(d0, d1, count)
    }

    fn time_nice(&self) -> Self {
        let [d0, d1] = self.0.domain;
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let span = hi - lo;
        let cal = nice_calendar_interval(span, 10);
        let (new_lo, new_hi) = match cal {
            CalendarInterval::Month | CalendarInterval::Year => {
                use chrono::{Datelike, TimeZone, Utc};
                // Safe conversion: fall back to approximate arithmetic on out-of-range timestamps.
                let Some(dt_lo) = Utc.timestamp_millis_opt(lo as i64).single() else {
                    let iv = nice_time_interval_ms(span, 10);
                    if !iv.is_finite() || iv <= 0.0 { return self.clone(); }
                    return TimeScale(LinearScaleData {
                        domain: [(lo / iv).floor() * iv, (hi / iv).ceil() * iv],
                        range: self.0.range, clamp: self.0.clamp,
                    }, self.1, self.2, self.3);
                };
                let Some(dt_hi) = Utc.timestamp_millis_opt(hi as i64).single() else {
                    return self.clone();
                };
                let snapped_lo = if cal == CalendarInterval::Year {
                    Utc.with_ymd_and_hms(dt_lo.year(), 1, 1, 0, 0, 0)
                        .single().map(|t| t.timestamp_millis() as f64)
                        .unwrap_or(lo)
                } else {
                    Utc.with_ymd_and_hms(dt_lo.year(), dt_lo.month(), 1, 0, 0, 0)
                        .single().map(|t| t.timestamp_millis() as f64)
                        .unwrap_or(lo)
                };
                let (ny, nm) = if cal == CalendarInterval::Year {
                    (dt_hi.year() + 1, 1u32)
                } else {
                    let m = dt_hi.month() + 1;
                    if m > 12 { (dt_hi.year() + 1, 1u32) } else { (dt_hi.year(), m) }
                };
                let snapped_hi = Utc.with_ymd_and_hms(ny, nm, 1, 0, 0, 0)
                    .single().map(|t| t.timestamp_millis() as f64)
                    .unwrap_or(hi);
                (snapped_lo, snapped_hi)
            }
            _ => {
                let interval = nice_time_interval_ms(span, 10);
                if !interval.is_finite() || interval <= 0.0 {
                    return self.clone();
                }
                ((lo / interval).floor() * interval, (hi / interval).ceil() * interval)
            }
        };
        let new_domain = if d0 <= d1 { [new_lo, new_hi] } else { [new_hi, new_lo] };
        TimeScale(
            LinearScaleData { domain: new_domain, range: self.0.range, clamp: self.0.clamp },
            self.1, self.2, self.3,
        )
    }
}

#[pymethods]
impl TimeScale {
    #[new]
    #[pyo3(signature = (*, domain, range = None, clamp = false, nice = false, padding = None, utc = false))]
    fn new(
        domain: Vec<f64>,
        range: Option<Vec<f64>>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
        utc: bool,
    ) -> PyResult<Self> {
        let range_user_set = range.is_some();
        let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
        validate_continuous_pair(&domain, &r)?;
        let inner = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [r[0],  r[1]],
            clamp,
        };
        let s = TimeScale(inner, padding, range_user_set, utc);
        if nice {
            Ok(s.time_nice())
        } else {
            Ok(s)
        }
    }

    /// Map an epoch-millisecond value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }
    /// Invert a range coordinate ``y`` back to an epoch-millisecond value.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` time-aligned tick values within the domain.
    ///
    /// Tick granularity snaps to calendar intervals (seconds, minutes, hours,
    /// days, months, or years) based on the domain span.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.time_ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest calendar interval.
    fn nice(&self) -> Self { self.time_nice() }

    /// Input domain as ``[t_min, t_max]`` in epoch milliseconds.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.to_vec() }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.2 { Some(self.0.range.to_vec()) } else { None }
    }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.1 }

    /// Whether this is a UTC time scale (affects type serialization).
    #[getter]
    fn utc(&self) -> bool { self.3 }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_scale_round_trip_ms() {
        // 2026-01-01 00:00:00 UTC = 1767225600000.0 ms
        // 2026-12-31 23:59:59 UTC ≈ 1798761599000.0 ms
        let t = TimeScale::new(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            Some(vec![0.0, 1000.0]),
            false,
            false,
            None,
            false,
        ).unwrap();
        let mid = (1_767_225_600_000.0 + 1_798_761_599_000.0) / 2.0;
        let y = t.scale(mid);
        let back = t.invert(y);
        assert!((back - mid).abs() < 1e-3, "round-trip failed: got {back}");
    }

    #[test]
    fn test_time_ticks_returns_some_ticks_for_year_span() {
        let t = TimeScale::new(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            Some(vec![0.0, 1000.0]),
            false,
            false,
            None,
            false,
        ).unwrap();
        let ticks = t.ticks(10);
        assert!(!ticks.is_empty(), "expected non-empty ticks");
    }

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Regression: time scale major positions are unchanged after adding minor support.
    ///
    /// Both `ticks_internal(10)` and `minor_ticks_internal()` use count=10 for
    /// the major tick generation, so the comparison is apples-to-apples.
    #[test]
    fn test_time_major_positions_unchanged() {
        // 2026-01-01 to 2026-12-31 (month-level major ticks).
        let t = TimeScale::new_internal(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            vec![0.0, 1000.0],
            false,
            false,
        );
        let before = t.ticks_internal(10);
        let _ = t.minor_ticks_internal();
        let after = t.ticks_internal(10);
        assert_eq!(before, after, "major ticks changed after calling minor_ticks_internal");
    }

    /// Time minor ticks are present and lie between major ticks.
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10 internally.
    #[test]
    fn test_time_minor_ticks_exist_between_majors() {
        // Use a span where major ticks fall at month boundaries.
        // 2026-01-01 = 1767225600000 ms, 2026-12-31 ≈ 1798761599000 ms.
        let t = TimeScale::new_internal(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            vec![0.0, 1000.0],
            false,
            false,
        );
        // minor_ticks_internal uses count=10 for majors internally.
        let majors = t.ticks_internal(10);
        let minors = t.minor_ticks_internal();

        // Minors must be non-empty when there are at least 2 major ticks.
        if majors.len() >= 2 {
            assert!(!minors.is_empty(), "expected minor ticks between month boundaries");
        }

        // Minors must lie within the domain and not coincide with majors.
        let lo = 1_767_225_600_000.0_f64;
        let hi = 1_798_761_599_000.0_f64;
        let major_set: std::collections::HashSet<u64> =
            majors.iter().map(|&v| v.to_bits()).collect();
        for m in &minors {
            assert!(!major_set.contains(&m.position.to_bits()),
                "time minor at {} coincides with major", m.position);
            assert!(m.position >= lo && m.position <= hi,
                "time minor {} outside domain [{lo},{hi}]", m.position);
            assert!(!m.is_major);
        }
    }

    /// Minor count per major interval = 4 (DEFAULT_MINOR_SUBDIVISIONS - 1).
    /// Use a small fixed-interval domain to make the count deterministic.
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10; the 10-second
    /// span with count=10 produces major ticks at 1-second intervals, giving a
    /// predictable 4 minors per interval.
    #[test]
    fn test_time_minor_count_per_interval() {
        // Span of exactly 10 seconds — major ticks at 1-second intervals.
        let lo = 0.0_f64;
        let hi = 10_000.0; // 10 seconds in ms
        let t = TimeScale::new_internal(vec![lo, hi], vec![0.0, 1000.0], false, false);
        // minor_ticks_internal uses count=10 for majors; ticks_internal(10) gives same list.
        let majors = t.ticks_internal(10);
        let minors = t.minor_ticks_internal();
        if majors.len() >= 2 {
            // (DEFAULT_MINOR_SUBDIVISIONS - 1) = 4 interior minors per interval.
            let expected = (majors.len() - 1) * 4;
            assert_eq!(minors.len(), expected,
                "expected {expected} minors for {}-interval time scale, got {}: {minors:?}",
                majors.len() - 1,
                minors.len(),
            );
        }
    }
}
