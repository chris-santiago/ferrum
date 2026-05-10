//! Tick label formatters: numeric, time, ordinal. Hardcoded defaults — no
//! per-axis FormatSpec yet (deferred to Phase 8 per locked decision §11 row 8).

/// Format a numeric tick value:
/// - Integer-valued in normal range: drop decimal ("0", "5", "100").
/// - Decimal with ≤ 4 sig figs: drop trailing zeros ("1.5", "0.25").
/// - |x| >= 1e6 or (0 < |x| < 1e-3): scientific notation ("1.5e6", "1e-4").
pub fn format_numeric(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let abs = x.abs();
    if abs >= 1e6 || abs < 1e-3 {
        let formatted = format!("{x:.3e}");
        trim_scientific(&formatted)
    } else if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        let s = format!("{x:.4}");
        trim_trailing_zeros(&s)
    }
}

fn trim_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

fn trim_scientific(s: &str) -> String {
    let (mantissa, exp) = s.split_once('e').unwrap_or((s, ""));
    let mantissa = trim_trailing_zeros(mantissa);
    if exp.is_empty() { mantissa } else { format!("{mantissa}e{exp}") }
}

/// Time-tick formatter. Picks granularity from inter-tick spacing in milliseconds.
/// - >= 1 year: "2026"
/// - >= 1 month: "Mar 2026"
/// - >= 1 day: "2026-03-15"
/// - >= 1 hour: "15:00"
/// - else: "15:30:45"
pub fn format_time(epoch_ms: i64, spacing_ms: i64) -> String {
    let (y, mo, d, h, mi, s) = epoch_ms_to_ymdhms(epoch_ms);
    const DAY: i64 = 86_400_000;
    const HOUR: i64 = 3_600_000;
    if spacing_ms >= 365 * DAY {
        format!("{y}")
    } else if spacing_ms >= 28 * DAY {
        format!("{} {y}", month_short(mo))
    } else if spacing_ms >= DAY {
        format!("{y:04}-{mo:02}-{d:02}")
    } else if spacing_ms >= HOUR {
        format!("{h:02}:{mi:02}")
    } else {
        format!("{h:02}:{mi:02}:{s:02}")
    }
}

fn month_short(m: u32) -> &'static str {
    const NAMES: [&str; 12] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    NAMES[(m - 1) as usize % 12]
}

/// Convert epoch milliseconds to Gregorian (Y, M, D, h, m, s) in UTC.
/// Self-contained — no chrono dep. Civil-from-days algorithm by Howard Hinnant.
fn epoch_ms_to_ymdhms(epoch_ms: i64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = epoch_ms.div_euclid(1000);
    let ms_part = epoch_ms.rem_euclid(1000);
    let _ = ms_part;
    let days = secs.div_euclid(86_400);
    let time = secs.rem_euclid(86_400);
    let h = (time / 3600) as u32;
    let mi = ((time % 3600) / 60) as u32;
    let s = (time % 60) as u32;
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, h, mi, s)
}

/// Ordinal/threshold passthrough — caller already has a string.
pub fn format_ordinal(value: &str) -> String {
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_zero() { assert_eq!(format_numeric(0.0), "0"); }
    #[test]
    fn numeric_integer() { assert_eq!(format_numeric(5.0), "5"); }
    #[test]
    fn numeric_decimal_trims() { assert_eq!(format_numeric(1.5), "1.5"); }
    #[test]
    fn numeric_large_uses_scientific() { assert_eq!(format_numeric(1_500_000.0), "1.5e6"); }
    #[test]
    fn numeric_tiny_uses_scientific() {
        let s = format_numeric(0.0001);
        assert!(s.starts_with("1") && s.contains("e-4"), "got: {s}");
    }
    #[test]
    fn numeric_near_one_trims_to_one() { assert_eq!(format_numeric(1.000001), "1"); }

    #[test]
    fn time_year_spacing() {
        let s = format_time(1767225600000, 365 * 86_400_000);
        assert_eq!(s, "2026");
    }
    #[test]
    fn time_day_spacing() {
        let s = format_time(1767225600000, 86_400_000);
        assert_eq!(s, "2026-01-01");
    }
    #[test]
    fn time_hour_spacing() {
        let s = format_time(1767279600000, 3_600_000);
        assert_eq!(s, "15:00");
    }

    #[test]
    fn ordinal_passthrough() {
        assert_eq!(format_ordinal("setosa"), "setosa");
    }
}
