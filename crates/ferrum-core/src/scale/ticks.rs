//! Shared tick-generation and binning helpers.

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

/// Generate calendar-snapped tick positions (ms since Unix epoch) for a time axis.
///
/// Month ticks snap to the 1st of each month at 00:00 UTC; year ticks snap to
/// Jan 1 of each year. Sub-month intervals fall back to the approximate math in
/// `nice_time_interval_ms`.
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
            let mut year = start_dt.year();
            let mut month = start_dt.month();
            if start_dt.day() > 1 || start_dt.hour() > 0 || start_dt.minute() > 0 {
                month += 1;
                if month > 12 { month = 1; year += 1; }
            }
            let mut out = Vec::new();
            while let Some(t) = Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).single() {
                let ms = t.timestamp_millis() as f64;
                if ms > hi { break; }
                out.push(ms);
                month += 1;
                if month > 12 { month = 1; year += 1; }
            }
            out
        }
        CalendarInterval::Year => {
            let Some(start_dt) = Utc.timestamp_millis_opt(lo as i64).single() else {
                return Vec::new();
            };
            let mut year = start_dt.year();
            if start_dt.month() > 1 || start_dt.day() > 1 { year += 1; }
            let mut out = Vec::new();
            while let Some(t) = Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0).single() {
                let ms = t.timestamp_millis() as f64;
                if ms > hi { break; }
                out.push(ms);
                year += 1;
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
}
