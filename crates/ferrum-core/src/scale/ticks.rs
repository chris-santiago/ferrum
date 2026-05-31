//! Shared tick-generation and binning helpers.

// ── Tick type ────────────────────────────────────────────────────────────────

/// A single scale tick position with a major/minor classification.
///
/// Major ticks (`is_major = true`) correspond to the positions produced by the
/// existing `ticks_internal` / `nice_ticks` methods.  Minor ticks
/// (`is_major = false`) are the interior subdivision ticks produced by
/// `minor_ticks_*` helpers.
///
/// The `position` is always in the scale's **data domain** (epoch-ms for time
/// scales), not in pixel coordinates.
///
/// # Note
/// This type is consumed by the scale's `minor_ticks_internal()` methods,
/// which are wired into the render layer in the grid/minor-tick rendering task
/// (Task 2 of the grid subsystem).  The `#[allow(dead_code)]` suppresses the
/// compiler's dead-code lint while Task 2 is still pending.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tick {
    pub(crate) position: f64,
    pub(crate) is_major: bool,
}

// ── Minor tick generation ────────────────────────────────────────────────────

/// Number of sub-intervals each major interval is divided into for the
/// default (linear / pow / sqrt / symlog / time) minor tick algorithm.
///
/// `5` sub-intervals → 4 interior minor ticks per major interval.  This
/// matches the conventional matplotlib/D3 default for linear scales.
// Consumed by minor_ticks_default, which is wired to render in Task 2.
#[allow(dead_code)]
const DEFAULT_MINOR_SUBDIVISIONS: usize = 5;

/// Generate minor ticks for the **default** (linear / pow / sqrt / symlog /
/// time) algorithm.
// Wired to the render layer in Task 2 of the grid subsystem.
#[allow(dead_code)]
///
/// The minor positions are computed in the *transformed* space (the space
/// where the major ticks are evenly spaced).  Pass pre-transformed major
/// positions (i.e., the major tick positions expressed in a uniform
/// coordinate) as `transformed_majors`, and a `to_data` closure that maps
/// them back to data-domain values.  For a plain linear scale the transform
/// is the identity, so `transformed_majors == major_data_positions` and
/// `to_data = |x| x`.
///
/// Returns only interior minor ticks (those strictly between consecutive
/// major ticks, not coincident with any major).
pub(crate) fn minor_ticks_default(
    transformed_majors: &[f64],
    to_data: impl Fn(f64) -> f64,
) -> Vec<Tick> {
    if transformed_majors.len() < 2 {
        return Vec::new();
    }

    let n = DEFAULT_MINOR_SUBDIVISIONS; // 5 sub-intervals → 4 interior minors
    let mut out: Vec<Tick> = Vec::new();

    for window in transformed_majors.windows(2) {
        let t0 = window[0];
        let t1 = window[1];
        let step = (t1 - t0) / n as f64;
        // i = 1 .. n-1  → strictly interior (skip endpoints which are major)
        for i in 1..n {
            let t = t0 + (i as f64) * step;
            let pos = to_data(t);
            if pos.is_finite() {
                out.push(Tick { position: pos, is_major: false });
            }
        }
    }

    out
}

/// Generate minor ticks for **log** scales using the standard 2-9
/// intra-decade multiples.
// Wired to the render layer in Task 2 of the grid subsystem.
#[allow(dead_code)]
///
/// For a base-10 log scale with positive domain `[lo, hi]`, the minor ticks
/// are the non-power-of-10 multiples `{2,3,4,5,6,7,8,9} × 10^e` that fall
/// strictly inside `[lo, hi]` and do not coincide with any major tick.
///
/// `major_positions` must already be filtered to the domain (as returned by
/// the log scale's `ticks_internal`).  The `base` parameter is the log base
/// (typically 10.0).
///
/// Negative domains are handled by reflecting through zero: the multiples
/// become `{-2,-3,-4,-5,-6,-7,-8,-9} × base^e`.
pub(crate) fn minor_ticks_log(
    lo: f64,
    hi: f64,
    base: f64,
    major_positions: &[f64],
) -> Vec<Tick> {
    if major_positions.is_empty() || !lo.is_finite() || !hi.is_finite() {
        return Vec::new();
    }
    // We only support base-10; for other bases fall back to empty (log minors
    // are only visually meaningful for base-10, where 2-9 multiples are
    // familiar).  Base-2 has no useful 2-9 multiples within a decade.
    if (base - 10.0).abs() > 1e-9 {
        return Vec::new();
    }

    let neg = lo < 0.0;
    let sign: f64 = if neg { -1.0 } else { 1.0 };
    let abs_lo = (lo * sign).min(hi * sign);
    let abs_hi = (lo * sign).max(hi * sign);

    let log_base = base.ln();
    let lo_exp = (abs_lo.ln() / log_base).floor() as i64 - 1;
    let hi_exp = (abs_hi.ln() / log_base).ceil() as i64 + 1;

    // Build a set of major positions for dedup (in abs value space).
    let major_set: std::collections::HashSet<u64> = major_positions
        .iter()
        .map(|&v| (v * sign).to_bits())
        .collect();

    let multiples: [u64; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
    let mut out: Vec<Tick> = Vec::new();

    for e in lo_exp..=hi_exp {
        let decade = base.powi(e as i32);
        for &m in &multiples {
            let abs_pos = (m as f64) * decade;
            if abs_pos <= abs_lo || abs_pos >= abs_hi {
                continue;
            }
            let pos = sign * abs_pos;
            // Skip if coincident with a major tick.
            if major_set.contains(&abs_pos.to_bits()) {
                continue;
            }
            if pos.is_finite() {
                out.push(Tick { position: pos, is_major: false });
            }
        }
    }

    // Sort by absolute value ascending, then apply domain order.
    out.sort_by(|a, b| {
        (a.position * sign)
            .partial_cmp(&(b.position * sign))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // If the domain is descending, reverse.
    if lo > hi {
        out.reverse();
    }

    out
}

pub(crate) fn sturges_floor(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let v = ((n as f64).log2() + 1.0).ceil();
    if v < 1.0 { 1 } else { v as usize }
}

pub(crate) fn nice_step(d_lo: f64, d_hi: f64, count: usize) -> f64 {
    if count == 0 || !d_lo.is_finite() || !d_hi.is_finite() {
        return f64::NAN;
    }
    let span = (d_hi - d_lo).abs();
    if span == 0.0 {
        return 0.0;
    }
    let step0 = span / (count as f64);
    let exp = step0.log10().floor();
    let pow10 = 10f64.powf(exp);
    let frac = step0 / pow10;
    let nice_frac = if frac >= 7.5 {
        10.0
    } else if frac >= 3.5 {
        5.0
    } else if frac >= 1.5 {
        2.0
    } else {
        1.0
    };
    nice_frac * pow10
}

pub(crate) fn nice_ticks(d_lo: f64, d_hi: f64, count: usize) -> Vec<f64> {
    if count == 0 || !d_lo.is_finite() || !d_hi.is_finite() {
        return Vec::new();
    }
    let (lo, hi, reverse) = if d_lo <= d_hi {
        (d_lo, d_hi, false)
    } else {
        (d_hi, d_lo, true)
    };
    if lo == hi {
        return vec![lo];
    }
    let step = nice_step(lo, hi, count);
    if !step.is_finite() || step == 0.0 {
        return vec![lo];
    }
    let start = (lo / step).ceil() * step;
    let end = (hi / step).floor() * step;
    let n_steps = ((end - start) / step).round() as i64;
    if n_steps < 0 {
        return Vec::new();
    }
    let n = (n_steps + 1) as usize;
    let mut out: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();
    if reverse {
        out.reverse();
    }
    out
}

pub(crate) fn nice_time_interval_ms(span_ms: f64, count: usize) -> f64 {
    const SECOND: f64 = 1_000.0;
    const MINUTE: f64 = 60.0 * SECOND;
    const HOUR:   f64 = 60.0 * MINUTE;
    const DAY:    f64 = 24.0 * HOUR;
    const WEEK:   f64 = 7.0 * DAY;
    const MONTH:  f64 = 30.0 * DAY;
    const YEAR:   f64 = 365.0 * DAY;

    if count == 0 || !span_ms.is_finite() || span_ms <= 0.0 {
        return f64::NAN;
    }
    let target = span_ms / count as f64;
    let candidates: [f64; 19] = [
        SECOND, 5.0 * SECOND, 15.0 * SECOND, 30.0 * SECOND,
        MINUTE, 5.0 * MINUTE, 15.0 * MINUTE, 30.0 * MINUTE,
        HOUR, 3.0 * HOUR, 6.0 * HOUR, 12.0 * HOUR,
        DAY, 2.0 * DAY,
        WEEK,
        MONTH, 3.0 * MONTH, 6.0 * MONTH,
        YEAR,
    ];
    // Pick the largest candidate ≤ target; if none, return the smallest.
    let mut chosen = candidates[0];
    for &c in candidates.iter() {
        if c <= target {
            chosen = c;
        } else {
            break;
        }
    }
    chosen
}

// ── Calendar-aware tick generation ──────────────────────────────────────────
//
// For month/year spans, `nice_time_interval_ms` returns approximate 30-day or
// 365-day intervals. `calendar_ticks` instead snaps tick positions to real
// calendar boundaries: the 1st of each month, or Jan 1 of each year.

use chrono::{Datelike, TimeZone, Timelike, Utc};

/// Categorise a millisecond span into a calendar tick interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CalendarInterval {
    SubSecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

const _SECOND: f64 = 1_000.0;
const _MINUTE: f64 = 60.0 * _SECOND;
const _HOUR:   f64 = 60.0 * _MINUTE;
const _DAY:    f64 = 24.0 * _HOUR;
const _WEEK:   f64 = 7.0  * _DAY;
const _MONTH:  f64 = 30.0 * _DAY;
const _YEAR:   f64 = 365.0 * _DAY;

pub(crate) fn nice_calendar_interval(span_ms: f64, count: usize) -> CalendarInterval {
    if count == 0 || !span_ms.is_finite() || span_ms <= 0.0 {
        return CalendarInterval::Second;
    }
    let target = span_ms / count as f64;
    match target {
        t if t < _SECOND  => CalendarInterval::SubSecond,
        t if t < _MINUTE  => CalendarInterval::Second,
        t if t < _HOUR    => CalendarInterval::Minute,
        t if t < _DAY     => CalendarInterval::Hour,
        t if t < _WEEK    => CalendarInterval::Day,
        t if t < _MONTH   => CalendarInterval::Week,
        t if t < _YEAR    => CalendarInterval::Month,
        _                  => CalendarInterval::Year,
    }
}

/// Pick a "nice" month stride (1, 2, 3, 6, or 12) so that `span_months`
/// subdivided by the stride lands near `count` ticks. Keeps month labels evenly
/// spaced (quarters, half-years, years) instead of one tick per month.
fn nice_month_step(span_months: i64, count: usize) -> i64 {
    if count == 0 {
        return 1;
    }
    let target = (span_months as f64 / count as f64).max(1.0);
    for &s in &[1, 2, 3, 6, 12] {
        if (s as f64) >= target {
            return s;
        }
    }
    12
}

/// Pick a "nice" year stride (1, 2, 5, 10, 20, 50, ... — the standard 1/2/5
/// decade progression) so `span_years / step` lands near `count`.
fn nice_year_step(span_years: i64, count: usize) -> i64 {
    if count == 0 {
        return 1;
    }
    let target = (span_years as f64 / count as f64).max(1.0);
    let mut step = 1i64;
    let mults = [1i64, 2, 5];
    let mut pow = 1i64;
    loop {
        for &m in &mults {
            step = m * pow;
            if step as f64 >= target {
                return step;
            }
        }
        pow *= 10;
        if pow > 1_000_000 {
            return step;
        }
    }
}

/// Generate calendar-snapped tick positions (ms since Unix epoch) for a time axis.
///
/// Month ticks snap to the 1st of each month at 00:00 UTC; year ticks snap to
/// Jan 1 of each year. Both strides are widened toward `count` (via
/// [`nice_month_step`] / [`nice_year_step`]) so a long span does not emit one
/// tick per calendar unit. Sub-month intervals fall back to the approximate
/// math in `nice_time_interval_ms`.
pub(crate) fn calendar_ticks(lo_ms: f64, hi_ms: f64, count: usize) -> Vec<f64> {
    if count == 0 || !lo_ms.is_finite() || !hi_ms.is_finite() {
        return Vec::new();
    }
    let (lo, hi, reversed) = if lo_ms <= hi_ms {
        (lo_ms, hi_ms, false)
    } else {
        (hi_ms, lo_ms, true)
    };
    let span = hi - lo;

    let interval = nice_calendar_interval(span, count);
    let mut ticks: Vec<f64> = match interval {
        CalendarInterval::Month => {
            let Some(start_dt) = Utc.timestamp_millis_opt(lo as i64).single() else {
                return Vec::new(); // out-of-range timestamp
            };
            // Stride by a "nice" number of months so a long span (e.g. 72 months)
            // approximates `count` ticks instead of emitting one per month.
            let span_months = (span / _MONTH).round().max(1.0) as i64;
            let step = nice_month_step(span_months, count);
            let mut year = start_dt.year();
            let mut month = start_dt.month();
            if start_dt.day() > 1 || start_dt.hour() > 0 || start_dt.minute() > 0 {
                month += 1;
                if month > 12 { month = 1; year += 1; }
            }
            // Snap the first tick to a stride-aligned month so the labels read
            // evenly (e.g. Jan/Apr/Jul/Oct for a 3-month stride within a year).
            if step > 1 {
                let from_jan = (month as i64 - 1).rem_euclid(step);
                if from_jan != 0 {
                    let advance = step - from_jan;
                    let m0 = month as i64 - 1 + advance;
                    year += (m0 / 12) as i32;
                    month = (m0 % 12) as u32 + 1;
                }
            }
            let mut out = Vec::new();
            while let Some(t) = Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).single() {
                let ms = t.timestamp_millis() as f64;
                if ms > hi { break; }
                out.push(ms);
                let m0 = month as i64 - 1 + step;
                year += (m0 / 12) as i32;
                month = (m0 % 12) as u32 + 1;
            }
            out
        }
        CalendarInterval::Year => {
            let Some(start_dt) = Utc.timestamp_millis_opt(lo as i64).single() else {
                return Vec::new();
            };
            // Stride by a "nice" number of years (1, 2, 5, 10, ...) toward `count`.
            let span_years = (span / _YEAR).round().max(1.0) as i64;
            let step = nice_year_step(span_years, count);
            let mut year = start_dt.year();
            if start_dt.month() > 1 || start_dt.day() > 1 { year += 1; }
            // Snap to a stride-aligned year so labels land on round multiples.
            if step > 1 {
                let rem = (year as i64).rem_euclid(step);
                if rem != 0 {
                    year += (step - rem) as i32;
                }
            }
            let mut out = Vec::new();
            while let Some(t) = Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0).single() {
                let ms = t.timestamp_millis() as f64;
                if ms > hi { break; }
                out.push(ms);
                year += step as i32;
            }
            out
        }
        _ => {
            // Sub-month intervals: use the approximate math.
            let iv = nice_time_interval_ms(span, count);
            if !iv.is_finite() || iv <= 0.0 { return Vec::new(); }
            let start = (lo / iv).ceil() * iv;
            let n = ((hi - start) / iv).floor() as usize + 1;
            (0..n).map(|i| start + i as f64 * iv).collect()
        }
    };

    if reversed { ticks.reverse(); }
    ticks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sturges_floor_known_values() {
        // ceil(log2(n) + 1)
        assert_eq!(sturges_floor(0), 1);
        assert_eq!(sturges_floor(1), 1);
        assert_eq!(sturges_floor(2), 2);
        assert_eq!(sturges_floor(8), 4);
        assert_eq!(sturges_floor(10), 5);    // ceil(log2(10)+1) = ceil(4.32) = 5
        assert_eq!(sturges_floor(100), 8);   // ceil(log2(100)+1) = ceil(7.64) = 8
        assert_eq!(sturges_floor(1024), 11);
    }

    #[test]
    fn test_sturges_floor_returns_at_least_one() {
        assert!(sturges_floor(0) >= 1);
        assert!(sturges_floor(1) >= 1);
    }

    #[test]
    fn test_nice_step_simple_decades() {
        let s = nice_step(0.0, 10.0, 10);
        assert!((s - 1.0).abs() < 1e-12, "got {s}");

        let s = nice_step(0.0, 100.0, 10);
        assert!((s - 10.0).abs() < 1e-12, "got {s}");

        let s = nice_step(0.0, 1.0, 5);
        assert!((s - 0.2).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn test_nice_step_handles_zero_span() {
        assert_eq!(nice_step(5.0, 5.0, 10), 0.0);
    }

    #[test]
    fn test_nice_step_handles_invalid_inputs() {
        assert!(nice_step(0.0, 10.0, 0).is_nan());
        assert!(nice_step(f64::NAN, 10.0, 5).is_nan());
        assert!(nice_step(0.0, f64::INFINITY, 5).is_nan());
    }

    #[test]
    fn test_nice_month_step_progression() {
        // 72 months over ~6 ticks → 12-month stride.
        assert_eq!(nice_month_step(72, 6), 12);
        // 12 months over 6 ticks → ~2-month stride.
        assert_eq!(nice_month_step(12, 6), 2);
        // 6 months over 6 ticks → monthly.
        assert_eq!(nice_month_step(6, 6), 1);
    }

    #[test]
    fn test_nice_year_step_progression() {
        assert_eq!(nice_year_step(10, 10), 1);
        assert_eq!(nice_year_step(20, 5), 5);
        assert_eq!(nice_year_step(100, 5), 20);
    }

    #[test]
    fn test_calendar_ticks_month_count_limits_dense_span() {
        // A 72-month span (2020-01 .. 2025-12) must NOT emit ~72 ticks; with a
        // target count of 6 it should subsample to roughly that many.
        // 2020-01-01 = 1577836800000 ms; +72 months ≈ 2026-01-01.
        let lo = 1_577_836_800_000.0;
        let hi = lo + 72.0 * 30.0 * 86_400_000.0;
        let dense = calendar_ticks(lo, hi, 6);
        assert!(
            dense.len() <= 12,
            "tick_count=6 over 72 months should subsample, got {} ticks",
            dense.len()
        );
        assert!(!dense.is_empty());
    }

    #[test]
    fn test_calendar_ticks_more_count_more_ticks() {
        let lo = 1_577_836_800_000.0;
        let hi = lo + 72.0 * 30.0 * 86_400_000.0;
        let few = calendar_ticks(lo, hi, 4);
        let many = calendar_ticks(lo, hi, 24);
        assert!(
            many.len() >= few.len(),
            "higher count should not produce fewer ticks: few={} many={}",
            few.len(),
            many.len()
        );
    }

    #[test]
    fn test_nice_ticks_inclusive_endpoints() {
        let ticks = nice_ticks(0.0, 10.0, 10);
        assert_eq!(ticks.first().copied(), Some(0.0));
        assert_eq!(ticks.last().copied(), Some(10.0));
        assert_eq!(ticks.len(), 11);
    }

    #[test]
    fn test_nice_ticks_count_approx() {
        let ticks = nice_ticks(0.0, 100.0, 10);
        assert!(ticks.len() >= 5 && ticks.len() <= 15, "got {} ticks: {ticks:?}", ticks.len());
    }

    #[test]
    fn test_nice_ticks_descending_input_descending_output() {
        let ticks = nice_ticks(10.0, 0.0, 10);
        assert!(ticks.first().copied().unwrap() > ticks.last().copied().unwrap());
    }

    #[test]
    fn test_nice_ticks_zero_span_returns_singleton() {
        let ticks = nice_ticks(5.0, 5.0, 10);
        assert_eq!(ticks, vec![5.0]);
    }

    #[test]
    fn test_nice_time_interval_returns_second_for_small_spans() {
        let iv = nice_time_interval_ms(10_000.0, 10);
        assert_eq!(iv, 1_000.0);
    }

    #[test]
    fn test_nice_time_interval_returns_day_for_week_span() {
        let iv = nice_time_interval_ms(7.0 * 24.0 * 3600_000.0, 7);
        assert_eq!(iv, 24.0 * 3600_000.0);
    }

    #[test]
    fn test_nice_time_interval_invalid_inputs() {
        assert!(nice_time_interval_ms(0.0, 5).is_nan());
        assert!(nice_time_interval_ms(-1.0, 5).is_nan());
        assert!(nice_time_interval_ms(1000.0, 0).is_nan());
        assert!(nice_time_interval_ms(f64::NAN, 5).is_nan());
    }

    // ── minor_ticks_default tests ────────────────────────────────────────────

    /// Linear domain [0,10]: major ticks at 0,1,2,...,10 (step=1).
    /// Default_subdivisions=5 → 4 interior minors per interval.
    /// Minors: 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, ... (9 intervals × 4 = 36 minors).
    #[test]
    fn test_minor_ticks_default_count_per_interval() {
        let majors = nice_ticks(0.0, 10.0, 10);
        assert_eq!(majors.len(), 11, "majors: {majors:?}"); // 0..=10

        let minors = minor_ticks_default(&majors, |x| x);
        // 10 intervals × (5-1)=4 interior minors = 40
        assert_eq!(
            minors.len(),
            (majors.len() - 1) * (DEFAULT_MINOR_SUBDIVISIONS - 1),
            "expected {} minors, got {}: {:?}",
            (majors.len() - 1) * (DEFAULT_MINOR_SUBDIVISIONS - 1),
            minors.len(),
            minors,
        );
        // All minors must be non-major (is_major=false).
        assert!(minors.iter().all(|t| !t.is_major));
    }

    /// Minors must lie strictly between consecutive major ticks.
    #[test]
    fn test_minor_ticks_default_strictly_between_majors() {
        let majors = nice_ticks(0.0, 10.0, 10); // 0,1,2,...,10
        let minors = minor_ticks_default(&majors, |x| x);
        // No minor should coincide with a major.
        let major_set: std::collections::HashSet<u64> =
            majors.iter().map(|&v| v.to_bits()).collect();
        for m in &minors {
            assert!(
                !major_set.contains(&m.position.to_bits()),
                "minor at {} coincides with a major",
                m.position,
            );
            assert!(m.position.is_finite(), "minor position is not finite");
        }
    }

    /// Empty majors or single major → no minors.
    #[test]
    fn test_minor_ticks_default_empty_and_singleton() {
        assert!(minor_ticks_default(&[], |x| x).is_empty());
        assert!(minor_ticks_default(&[5.0], |x| x).is_empty());
    }

    // ── minor_ticks_log tests ────────────────────────────────────────────────

    /// Domain [1, 100]: major ticks at 1, 10, 100.
    /// Minors between 1-10: 2,3,4,5,6,7,8,9 (8 ticks).
    /// Minors between 10-100: 20,30,40,50,60,70,80,90 (8 ticks).
    /// Total: 16 minors.
    #[test]
    fn test_minor_ticks_log_two_decades_base10() {
        // Major ticks for [1,100]: 1, 10, 100
        let majors = vec![1.0_f64, 10.0, 100.0];
        let minors = minor_ticks_log(1.0, 100.0, 10.0, &majors);

        assert_eq!(minors.len(), 16, "expected 16 log minors, got {}: {:?}", minors.len(), minors);

        // Check first decade multiples are present.
        let positions: Vec<f64> = minors.iter().map(|t| t.position).collect();
        for m in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] {
            assert!(
                positions.iter().any(|&p| (p - m).abs() < 1e-9),
                "{m} not found in {positions:?}",
            );
        }
        // Second decade multiples.
        for m in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0] {
            assert!(
                positions.iter().any(|&p| (p - m).abs() < 1e-9),
                "{m} not found in {positions:?}",
            );
        }
        // All must be is_major=false.
        assert!(minors.iter().all(|t| !t.is_major));
    }

    /// Log minors are non-uniform in linear space (they crowd towards the top
    /// of each decade).  Verify: 2-9 within [1,10] are non-uniformly spaced.
    #[test]
    fn test_minor_ticks_log_nonuniform_spacing() {
        let majors = vec![1.0_f64, 10.0, 100.0];
        let minors = minor_ticks_log(1.0, 100.0, 10.0, &majors);
        // Filter first-decade minors (2..9).
        let first_decade: Vec<f64> = minors
            .iter()
            .filter(|t| t.position >= 2.0 && t.position <= 9.0)
            .map(|t| t.position)
            .collect();
        assert_eq!(first_decade.len(), 8, "expected 8 first-decade minors");
        // Gaps: 2→3=1, 3→4=1, ..., 8→9=1 (all equal in log minor terms, but
        // the actual data-space gaps are: 2-1=1, 3-2=1, 4-3=1,...).
        // What we're checking is non-uniformity relative to log-uniform ticks:
        // if we had log-uniform ticks at equal log-space intervals between 1 and 10,
        // they'd be at ~1.78, ~3.16, ~5.62, ~10.  Our 2-9 multiples are NOT
        // at those positions, so they are log-non-uniform.
        // Just confirm spacing in linear space is not constant.
        let diffs: Vec<f64> = first_decade
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        // All diffs should be 1.0 (2,3,4,...,9 are evenly-spaced integers) which IS
        // uniform in linear space but is NOT uniform in log space.
        // The key property is these are NOT evenly spaced in log space.
        let log_diffs: Vec<f64> = first_decade
            .windows(2)
            .map(|w| w[1].ln() - w[0].ln())
            .collect();
        // ln(3/2) ≠ ln(8/7): check that not all log gaps are equal.
        let first = log_diffs[0];
        let all_equal = log_diffs.iter().all(|&d| (d - first).abs() < 1e-9);
        assert!(!all_equal, "log minors should be non-uniform in log space: {log_diffs:?}");
        _ = diffs; // suppress unused warning
    }

    /// Log minors must not duplicate major positions.
    #[test]
    fn test_minor_ticks_log_no_coincidence_with_majors() {
        let majors = vec![1.0_f64, 10.0, 100.0, 1000.0];
        let minors = minor_ticks_log(1.0, 1000.0, 10.0, &majors);
        let major_set: std::collections::HashSet<u64> =
            majors.iter().map(|&v| v.to_bits()).collect();
        for m in &minors {
            assert!(
                !major_set.contains(&m.position.to_bits()),
                "minor at {} coincides with major",
                m.position,
            );
        }
    }

    /// Non-base-10 log scales produce no minor ticks.
    #[test]
    fn test_minor_ticks_log_non_base10_returns_empty() {
        let majors = vec![1.0, 2.0, 4.0, 8.0];
        let minors = minor_ticks_log(1.0, 8.0, 2.0, &majors);
        assert!(minors.is_empty(), "base-2 log minors should be empty, got {minors:?}");
    }

    /// Empty major list → empty minors.
    #[test]
    fn test_minor_ticks_log_empty_majors() {
        let minors = minor_ticks_log(1.0, 100.0, 10.0, &[]);
        assert!(minors.is_empty());
    }
}
