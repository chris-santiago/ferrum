//! Shared tick-generation and binning helpers.

#![allow(dead_code)]

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
    const MONTH:  f64 = 30.0 * DAY;   // approximate; calendar-aware deferred
    const YEAR:   f64 = 365.0 * DAY;  // approximate

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
