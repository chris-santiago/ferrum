//! Tick label formatters: numeric, time, ordinal.
//!
//! This module is the single source of truth for number and time formatting.
//! It feeds SVG, PNG, and interactive renders identically (axis tick labels,
//! text-mark labels, tooltips, colorbars). Two layers:
//!
//! - **Defaults** ([`format_numeric`], [`format_time`]) — used when no explicit
//!   format spec is supplied. Their output is byte-stable; do not change it.
//! - **Explicit specs** ([`format_with_spec`], [`format_time_spec`]) — a full
//!   d3-format grammar for numbers and `chrono` strftime patterns for time,
//!   applied only when the encoding/axis carries a format string.
//!
//! The d3-format grammar implemented here (native, no external crate) is:
//! `[[fill]align][sign][symbol][0][width][,][.precision][~][type]`. See
//! [`parse_format_spec`] for the parser and [`format_with_spec`] for the
//! application. The `format_num` crate is a useful reference for the SI/rounding
//! algorithms but is intentionally not a dependency (it lacks `~`/`g`/`r`/`p`
//! and pulls `regex`).

use chrono::{DateTime, Datelike, Timelike, Utc};

/// Format a numeric tick value with the engine's *default* (no-spec) rules:
/// - NaN or Infinity: returns empty string (not suitable for SVG text/tooltip).
/// - Integer-valued in normal range: drop decimal ("0", "5", "100").
/// - Decimal with ≤ 4 sig figs: drop trailing zeros ("1.5", "0.25").
/// - |x| >= 1e6 or (0 < |x| < 1e-3): scientific notation ("1.5e6", "1e-4").
///
/// This is the behavior callers get when no format string is supplied; it must
/// stay byte-stable.
pub fn format_numeric(x: f64) -> String {
    if x.is_nan() || x.is_infinite() {
        return String::new();
    }
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

// ---------------------------------------------------------------------------
// Time formatting (chrono-backed)
// ---------------------------------------------------------------------------

/// Default time-tick formatter. Picks granularity from inter-tick spacing in
/// milliseconds. Output is byte-stable (used when no explicit time format is
/// supplied):
/// - >= 1 year: "2026"
/// - >= 1 month: "Mar 2026"
/// - >= 1 day: "2026-03-15"
/// - >= 1 hour: "15:00"
/// - else: "15:30:45"
pub fn format_time(epoch_ms: i64, spacing_ms: i64) -> String {
    let Some(dt) = DateTime::<Utc>::from_timestamp_millis(epoch_ms) else {
        return String::new();
    };
    const DAY: i64 = 86_400_000;
    const HOUR: i64 = 3_600_000;
    if spacing_ms >= 365 * DAY {
        format!("{}", dt.year())
    } else if spacing_ms >= 28 * DAY {
        format!("{} {}", month_short(dt.month()), dt.year())
    } else if spacing_ms >= DAY {
        format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day())
    } else if spacing_ms >= HOUR {
        format!("{:02}:{:02}", dt.hour(), dt.minute())
    } else {
        format!("{:02}:{:02}:{:02}", dt.hour(), dt.minute(), dt.second())
    }
}

fn month_short(m: u32) -> &'static str {
    const NAMES: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    NAMES[(m - 1) as usize % 12]
}

/// Format an epoch-millisecond timestamp with an explicit `chrono` strftime
/// pattern (e.g. `"%b %Y"`, `"%Y-%m-%d"`, `"%H:%M"`). Returns the empty string
/// for out-of-range timestamps. Used when an axis/encoding carries an explicit
/// time format string (`format_type == "time"`).
pub fn format_time_spec(epoch_ms: i64, pattern: &str) -> String {
    match DateTime::<Utc>::from_timestamp_millis(epoch_ms) {
        Some(dt) => dt.format(pattern).to_string(),
        None => String::new(),
    }
}

/// Ordinal/threshold passthrough — caller already has a string.
pub fn format_ordinal(value: &str) -> String {
    value.to_string()
}

// ---------------------------------------------------------------------------
// d3-format grammar
// ---------------------------------------------------------------------------

/// Alignment of the formatted value within its `width`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Align {
    /// `>` — right (the default).
    Right,
    /// `<` — left.
    Left,
    /// `^` — centered.
    Center,
    /// `=` — like `>`, but pad after the sign/symbol.
    SignAware,
}

/// How the sign is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sign {
    /// `-` — sign only for negatives (default).
    Negative,
    /// `+` — always show a sign.
    Always,
    /// ` ` (space) — space for positives, `-` for negatives.
    Space,
}

/// A parsed d3-format spec:
/// `[[fill]align][sign][symbol][0][width][,][.precision][~][type]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct D3Spec {
    fill: char,
    align: Align,
    sign: Sign,
    /// `$` currency or `#` (alternate form / radix prefix) symbol.
    symbol: Option<char>,
    /// `0` zero-pad flag (implies `=` align + `0` fill unless overridden).
    zero: bool,
    width: Option<usize>,
    /// `,` grouping flag.
    comma: bool,
    precision: Option<usize>,
    /// `~` trim-insignificant-trailing-zeros flag.
    trim: bool,
    /// Type char, `'\0'` if absent.
    ty: char,
}

impl Default for D3Spec {
    fn default() -> Self {
        D3Spec {
            fill: ' ',
            align: Align::Right,
            sign: Sign::Negative,
            symbol: None,
            zero: false,
            width: None,
            comma: false,
            precision: None,
            trim: false,
            ty: '\0',
        }
    }
}

/// Parse a d3-format string into a [`D3Spec`].
///
/// Grammar: `[[fill]align][sign][symbol][0][width][,][.precision][~][type]`.
/// Unrecognized leading bytes are tolerated (best-effort): the parser only
/// consumes tokens it understands and treats the rest as the type/precision
/// tail, matching d3's permissive behavior.
pub(crate) fn parse_format_spec(spec: &str) -> D3Spec {
    let mut out = D3Spec::default();
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let n = chars.len();

    // [[fill]align] — fill is any char immediately followed by an align char.
    let is_align = |c: char| matches!(c, '<' | '>' | '^' | '=');
    if i + 1 < n && is_align(chars[i + 1]) {
        out.fill = chars[i];
        out.align = align_from(chars[i + 1]);
        i += 2;
    } else if i < n && is_align(chars[i]) {
        out.align = align_from(chars[i]);
        i += 1;
    }

    // [sign]
    if i < n {
        match chars[i] {
            '-' => {
                out.sign = Sign::Negative;
                i += 1;
            }
            '+' => {
                out.sign = Sign::Always;
                i += 1;
            }
            ' ' => {
                out.sign = Sign::Space;
                i += 1;
            }
            _ => {}
        }
    }

    // [symbol]
    if i < n && (chars[i] == '$' || chars[i] == '#') {
        out.symbol = Some(chars[i]);
        i += 1;
    }

    // [0] zero-pad
    if i < n && chars[i] == '0' {
        out.zero = true;
        out.fill = '0';
        out.align = Align::SignAware;
        i += 1;
    }

    // [width]
    let start = i;
    while i < n && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i > start {
        out.width = chars[start..i].iter().collect::<String>().parse().ok();
    }

    // [,] grouping
    if i < n && chars[i] == ',' {
        out.comma = true;
        i += 1;
    }

    // [.precision]
    if i < n && chars[i] == '.' {
        i += 1;
        let pstart = i;
        while i < n && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i > pstart {
            out.precision = chars[pstart..i].iter().collect::<String>().parse().ok();
        } else {
            out.precision = Some(0);
        }
    }

    // [~] trim
    if i < n && chars[i] == '~' {
        out.trim = true;
        i += 1;
    }

    // [type]
    if i < n {
        out.ty = chars[i];
    }

    out
}

fn align_from(c: char) -> Align {
    match c {
        '<' => Align::Left,
        '^' => Align::Center,
        '=' => Align::SignAware,
        _ => Align::Right,
    }
}

/// Format a numeric value per a d3-format spec.
///
/// When `spec` is `None`, falls back to [`format_numeric`] (default behavior).
/// When a spec string is present, the full grammar is applied.
///
/// Supported type chars: `e f g s % p r b o d x X c n` plus the empty type.
/// Supported flags: fill/align, sign (`+`/`-`/space), symbol (`$`/`#`),
/// zero-pad, width, grouping `,`, `.precision`, and `~` (trim trailing zeros).
pub fn format_with_spec(v: f64, spec: Option<&str>) -> String {
    let Some(s) = spec else { return format_numeric(v) };
    if s.is_empty() {
        return format_numeric(v);
    }
    let parsed = parse_format_spec(s);
    format_parsed(v, &parsed)
}

/// Apply an already-parsed [`D3Spec`] to a value. Shared entry point so
/// `apply_tick_format` can parse once and reuse across a column of labels.
pub(crate) fn format_parsed(v: f64, spec: &D3Spec) -> String {
    if v.is_nan() || v.is_infinite() {
        return String::new();
    }

    // The `%`/`p` types scale by 100 and append a literal percent sign.
    // The `c` type emits the value as a Unicode code point with no numeric
    // formatting. Everything else formats the magnitude, then re-attaches the
    // sign, symbol prefix, and percent suffix.
    if spec.ty == 'c' {
        let cp = v as u32;
        let body = char::from_u32(cp).map(String::from).unwrap_or_default();
        return pad_with_lead(&body, spec, 0);
    }

    let negative = v.is_sign_negative() && v != 0.0;
    let mut value = v.abs();
    let mut suffix = String::new();
    let mut si_suffix = String::new();

    let precision = spec.precision;
    let mut body = match spec.ty {
        'f' | 'F' => fixed(value, precision.unwrap_or(6), spec.comma),
        'e' | 'E' => {
            let p = precision.unwrap_or(6);
            let s = format!("{value:.p$e}");
            normalize_exp(&s, spec.ty == 'E')
        }
        'g' | 'G' => general(value, precision.unwrap_or(6), spec.comma, spec.ty == 'G'),
        'r' => rounded_significant(value, precision.unwrap_or(6), spec.comma),
        's' => {
            let (scaled, suf) = si_scale(value, precision.unwrap_or(6));
            si_suffix = suf;
            scaled
        }
        '%' => {
            value *= 100.0;
            suffix.push('%');
            fixed(value, precision.unwrap_or(6), spec.comma)
        }
        'p' => {
            value *= 100.0;
            suffix.push('%');
            rounded_significant(value, precision.unwrap_or(6), spec.comma)
        }
        'b' => format!("{:b}", value.round() as i64),
        'o' => format!("{:o}", value.round() as i64),
        'd' | 'n' => integer(value, spec.comma || spec.ty == 'n'),
        'x' => format!("{:x}", value.round() as i64),
        'X' => format!("{:X}", value.round() as i64),
        '\0' => {
            // No type: d3's default behaves like `g` but with the value's own
            // precision when none given. We mirror `format_numeric` for the
            // no-precision case so default tick labels are unchanged; with an
            // explicit precision, use general.
            match precision {
                Some(p) => general(value, p.max(1), spec.comma, false),
                None if spec.comma => {
                    // d3 `,` with no type groups the value. Integer-valued inputs
                    // group as plain integers (no scientific cutover), so a large
                    // count like 1234567 renders "1,234,567" rather than "1.235e6".
                    if value.fract() == 0.0 {
                        group_int(&(value as u64).to_string())
                    } else {
                        regroup_numeric(&format_numeric(value))
                    }
                }
                None => format_numeric(value),
            }
        }
        _ => format_numeric(value),
    };

    if spec.trim {
        // Exponential types carry an exponent suffix in the body (e.g. "3.140e3"),
        // so they must trim only the mantissa via the exp-aware trimmer. SI (`s`)
        // also routes through it: its body has no exponent, so the exp trimmer
        // falls back to the plain trimmer, but this keeps the e/s split aligned
        // with the `g`-format split at lines ~477/482.
        body = match spec.ty {
            'e' | 'E' | 's' => trim_insignificant_exp(&body),
            _ => trim_insignificant(&body),
        };
    }

    // Symbol prefix (`$`); `#` alternate-form prefix for radix types.
    let mut prefix = String::new();
    match spec.symbol {
        Some('$') => prefix.push('$'),
        Some('#') => match spec.ty {
            'b' => prefix.push_str("0b"),
            'o' => prefix.push_str("0o"),
            'x' => prefix.push_str("0x"),
            'X' => prefix.push_str("0X"),
            _ => {}
        },
        _ => {}
    }

    let sign_str = match (negative, spec.sign) {
        (true, _) => "-",
        (false, Sign::Always) => "+",
        (false, Sign::Space) => " ",
        (false, Sign::Negative) => "",
    };

    let combined = format!("{sign_str}{prefix}{body}{si_suffix}{suffix}");
    // Sign-aware padding inserts fill between the sign/prefix and the digits.
    let padless_lead = sign_str.len() + prefix.len();
    pad_with_lead(&combined, spec, padless_lead)
}

/// Fixed-point with optional grouping. `value` is already non-negative.
fn fixed(value: f64, precision: usize, comma: bool) -> String {
    let s = format!("{value:.precision$}");
    if comma { regroup(&s) } else { s }
}

/// Integer (rounded), optional grouping.
fn integer(value: f64, comma: bool) -> String {
    let i = value.round() as u64;
    let s = i.to_string();
    if comma { group_int(&s) } else { s }
}

/// Round to `precision` significant digits, fixed-point output (d3 `r`).
fn rounded_significant(value: f64, precision: usize, comma: bool) -> String {
    if value == 0.0 {
        return if comma { regroup("0") } else { "0".to_string() };
    }
    let p = precision.max(1) as i32;
    let exp = value.abs().log10().floor() as i32;
    // Quantize to `p` significant digits: round(value / 10^q) * 10^q where
    // q = exp - p + 1. When q > 0 (integer rounding above the ones place) the
    // result has no fractional digits; when q < 0 we keep `-q` decimals.
    let q = exp - p + 1;
    let scale = 10f64.powi(q);
    let rounded = (value / scale).round() * scale;
    let decimals = (-q).max(0) as usize;
    let s = format!("{rounded:.decimals$}");
    if comma { regroup(&s) } else { s }
}

/// d3 `g` general format: `precision` significant digits, switching between
/// fixed and scientific based on the exponent.
fn general(value: f64, precision: usize, comma: bool, upper: bool) -> String {
    let p = precision.max(1);
    if value == 0.0 {
        return "0".to_string();
    }
    let exp = value.abs().log10().floor() as i32;
    // d3/printf %g rule: use scientific if exp < -4 or exp >= precision.
    if exp < -4 || exp >= p as i32 {
        let s = format!("{value:.*e}", p.saturating_sub(1));
        let s = trim_insignificant_exp(&s);
        normalize_exp(&s, upper)
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        let s = format!("{value:.decimals$}");
        let s = trim_insignificant(&s);
        if comma { regroup(&s) } else { s }
    }
}

/// SI-prefix scaling (d3 `s`). Returns the scaled mantissa string and the
/// SI suffix (e.g. `"k"`, `"M"`, `"µ"`). Mantissa kept to `precision`
/// significant digits.
fn si_scale(value: f64, precision: usize) -> (String, String) {
    const PREFIXES: [(i32, &str); 17] = [
        (24, "Y"),
        (21, "Z"),
        (18, "E"),
        (15, "P"),
        (12, "T"),
        (9, "G"),
        (6, "M"),
        (3, "k"),
        (0, ""),
        (-3, "m"),
        (-6, "µ"),
        (-9, "n"),
        (-12, "p"),
        (-15, "f"),
        (-18, "a"),
        (-21, "z"),
        (-24, "y"),
    ];
    if value == 0.0 {
        let p = precision.max(1);
        return (format!("{:.*}", p.saturating_sub(1), 0.0), String::new());
    }
    let exp = value.abs().log10().floor() as i32;
    // Group exponent into multiples of 3 within [-24, 24].
    let group = (exp.div_euclid(3) * 3).clamp(-24, 24);
    let (_, suffix) = PREFIXES
        .iter()
        .find(|(e, _)| *e == group)
        .copied()
        .unwrap_or((0, ""));
    let scaled = value / 10f64.powi(group);
    // d3 `s` precision counts significant digits of the scaled mantissa.
    let p = precision.max(1);
    let mant_exp = scaled.abs().log10().floor() as i32;
    let decimals = (p as i32 - 1 - mant_exp).max(0) as usize;
    (format!("{scaled:.decimals$}"), suffix.to_string())
}

/// Normalize Rust's `1.5e6` exponent form to d3's `1.5e+6` style (sign-padded
/// exponent, no leading zeros beyond one digit), upper-casing `E` if requested.
fn normalize_exp(s: &str, upper: bool) -> String {
    let e = if upper { 'E' } else { 'e' };
    let Some((mant, exp)) = s.split_once(['e', 'E']) else {
        return s.to_string();
    };
    let (sign, digits) = if let Some(rest) = exp.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exp.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exp)
    };
    let n: i64 = digits.parse().unwrap_or(0);
    format!("{mant}{e}{sign}{n}")
}

/// Trim insignificant trailing zeros from a plain decimal string (`~` flag /
/// `g` cleanup). `"1.5000" -> "1.5"`, `"3.000" -> "3"`, `"1,234.0" -> "1,234"`.
fn trim_insignificant(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0');
    let trimmed = trimmed.strip_suffix('.').unwrap_or(trimmed);
    trimmed.to_string()
}

/// Trim insignificant zeros from the mantissa of a scientific string,
/// preserving the exponent. `"1.500e6" -> "1.5e6"`.
fn trim_insignificant_exp(s: &str) -> String {
    match s.split_once(['e', 'E']) {
        Some((mant, exp)) => {
            let m = trim_insignificant(mant);
            let sep = if s.contains('E') { 'E' } else { 'e' };
            format!("{m}{sep}{exp}")
        }
        None => trim_insignificant(s),
    }
}

/// Apply thousands grouping to a non-negative decimal string (integer part
/// only). `"1234.50" -> "1,234.50"`.
fn regroup(s: &str) -> String {
    match s.split_once('.') {
        Some((int, frac)) => format!("{}.{}", group_int(int), frac),
        None => group_int(s),
    }
}

/// Apply grouping to a (possibly signed, possibly fractional) numeric string —
/// used when re-grouping the output of `format_numeric` for the no-type `,`
/// case. Preserves a leading sign and any `eN` exponent untouched.
fn regroup_numeric(s: &str) -> String {
    if s.contains('e') || s.contains('E') {
        return s.to_string();
    }
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => ("-", r),
        None => ("", s),
    };
    format!("{sign}{}", regroup(rest))
}

/// Insert thousands separators into a plain non-negative integer digit string.
fn group_int(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len <= 3 {
        return s.to_string();
    }
    let mut out = String::with_capacity(len + len / 3);
    let first = len % 3;
    for (i, ch) in s.chars().enumerate() {
        if i >= first && i != 0 && (i - first).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Pad to width. `lead` is the number of leading chars (sign + prefix) that
/// sign-aware (`=`) alignment keeps in front of the fill. `body` already
/// includes any sign/prefix.
fn pad_with_lead(body: &str, spec: &D3Spec, lead: usize) -> String {
    let Some(width) = spec.width else {
        return body.to_string();
    };
    let cur = body.chars().count();
    if cur >= width {
        return body.to_string();
    }
    let deficit = width - cur;
    let fill: String = std::iter::repeat_n(spec.fill, deficit).collect();
    match spec.align {
        Align::Right => format!("{fill}{body}"),
        Align::Left => format!("{body}{fill}"),
        Align::Center => {
            let left = deficit / 2;
            let right = deficit - left;
            let lf: String = std::iter::repeat_n(spec.fill, left).collect();
            let rf: String = std::iter::repeat_n(spec.fill, right).collect();
            format!("{lf}{body}{rf}")
        }
        Align::SignAware => {
            // Pad between the leading sign/prefix and the magnitude.
            let lead = lead.min(body.len());
            let (head, tail) = body.split_at(lead);
            format!("{head}{fill}{tail}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- default numeric (must stay byte-stable) ----
    #[test]
    fn numeric_zero() {
        assert_eq!(format_numeric(0.0), "0");
    }
    #[test]
    fn numeric_integer() {
        assert_eq!(format_numeric(5.0), "5");
    }
    #[test]
    fn numeric_decimal_trims() {
        assert_eq!(format_numeric(1.5), "1.5");
    }
    #[test]
    fn numeric_large_uses_scientific() {
        assert_eq!(format_numeric(1_500_000.0), "1.5e6");
    }
    #[test]
    fn numeric_tiny_uses_scientific() {
        let s = format_numeric(0.0001);
        assert!(s.starts_with('1') && s.contains("e-4"), "got: {s}");
    }
    #[test]
    fn numeric_near_one_trims_to_one() {
        assert_eq!(format_numeric(1.000001), "1");
    }
    #[test]
    fn with_spec_none_matches_default() {
        assert_eq!(format_with_spec(1_500_000.0, None), "1.5e6");
        assert_eq!(format_with_spec(1.5, None), "1.5");
        assert_eq!(format_with_spec(1_500_000.0, Some("")), "1.5e6");
    }

    // ---- d3 grammar: required cases from the task ----
    #[test]
    fn comma_fixed_zero_precision() {
        assert_eq!(format_with_spec(1234.0, Some(",.0f")), "1,234");
    }
    #[test]
    fn si_trim_megabyte() {
        assert_eq!(format_with_spec(1.5e6, Some("~s")), "1.5M");
    }
    #[test]
    fn si_significant_digits() {
        // d3 `s` precision counts SIGNIFICANT digits of the mantissa, so
        // ".2s" of 1.5e6 → "1.5M" (two sig figs: 1 and 5).
        assert_eq!(format_with_spec(1.5e6, Some(".2s")), "1.5M");
        // ".3s" keeps three sig figs → "1.50M".
        assert_eq!(format_with_spec(1.5e6, Some(".3s")), "1.50M");
    }
    #[test]
    fn percent_one_decimal() {
        assert_eq!(format_with_spec(0.25, Some(".1%")), "25.0%");
    }
    #[test]
    fn currency_grouped_two_decimals() {
        assert_eq!(format_with_spec(1234.5, Some("$,.2f")), "$1,234.50");
    }
    #[test]
    fn plus_sign_one_decimal() {
        assert_eq!(format_with_spec(3.0, Some("+.1f")), "+3.0");
        assert_eq!(format_with_spec(-3.0, Some("+.1f")), "-3.0");
    }
    #[test]
    fn space_sign() {
        assert_eq!(format_with_spec(3.0, Some(" .1f")), " 3.0");
        assert_eq!(format_with_spec(-3.0, Some(" .1f")), "-3.0");
    }
    #[test]
    fn trim_flag_exponential() {
        // "~e" must trim only the mantissa, preserving the exponent suffix.
        // Default precision (6) on 3140 -> "3.140000e+3", trimmed -> "3.14e+3".
        assert_eq!(format_with_spec(3140.0, Some("~e")), "3.14e+3");
        // No-trim ".2e" stays fully zero-padded.
        assert_eq!(format_with_spec(1500.0, Some(".2e")), "1.50e+3");
    }
    #[test]
    fn trim_flag_si_keeps_suffix() {
        // "~s" already worked; this guards against the e/s split regressing it.
        assert_eq!(format_with_spec(1_500_000.0, Some("~s")), "1.5M");
    }
    #[test]
    fn trim_flag_fixed() {
        // "~f" with default precision should drop the trailing zeros.
        assert_eq!(format_with_spec(1.5, Some("~f")), "1.5");
        assert_eq!(format_with_spec(3.0, Some("~f")), "3");
    }
    #[test]
    fn rounded_significant_three() {
        assert_eq!(format_with_spec(1234.567, Some(".3r")), "1230");
        assert_eq!(format_with_spec(0.0123456, Some(".3r")), "0.0123");
    }
    #[test]
    fn general_three() {
        assert_eq!(format_with_spec(1234.567, Some(".3g")), "1.23e+3");
        assert_eq!(format_with_spec(12.3456, Some(".3g")), "12.3");
        // exp == -4 is the printf %g fixed/scientific boundary: -4 is NOT < -4,
        // so it stays fixed-point.
        assert_eq!(format_with_spec(0.000123456, Some(".3g")), "0.000123");
        // exp == -5 crosses into scientific.
        assert_eq!(format_with_spec(0.0000123456, Some(".3g")), "1.23e-5");
    }
    #[test]
    fn percent_with_rounding_p() {
        // ".1p" rounds to 1 significant digit of the percent value.
        assert_eq!(format_with_spec(0.0123, Some(".1p")), "1%");
    }
    #[test]
    fn scientific_normalized_exp() {
        assert_eq!(format_with_spec(1500.0, Some(".2e")), "1.50e+3");
    }
    #[test]
    fn fixed_with_grouping_large() {
        assert_eq!(format_with_spec(1234567.0, Some(",.0f")), "1,234,567");
    }
    #[test]
    fn comma_only_no_type() {
        assert_eq!(format_with_spec(1234.0, Some(",")), "1,234");
        assert_eq!(format_with_spec(1234567.0, Some(",")), "1,234,567");
    }
    #[test]
    fn integer_d() {
        assert_eq!(format_with_spec(42.0, Some("d")), "42");
        assert_eq!(format_with_spec(3.7, Some("d")), "4");
        assert_eq!(format_with_spec(1234.0, Some(",d")), "1,234");
    }
    #[test]
    fn width_and_zero_pad() {
        assert_eq!(format_with_spec(7.0, Some("05d")), "00007");
        assert_eq!(format_with_spec(-7.0, Some("05d")), "-0007");
    }
    #[test]
    fn width_align_left() {
        assert_eq!(format_with_spec(7.0, Some("<5d")), "7    ");
    }
    #[test]
    fn width_align_center() {
        assert_eq!(format_with_spec(7.0, Some("^5d")), "  7  ");
    }
    #[test]
    fn fill_char_align() {
        assert_eq!(format_with_spec(7.0, Some("*>5d")), "****7");
    }
    #[test]
    fn hex_and_alt_form() {
        assert_eq!(format_with_spec(255.0, Some("x")), "ff");
        assert_eq!(format_with_spec(255.0, Some("#x")), "0xff");
        assert_eq!(format_with_spec(255.0, Some("X")), "FF");
    }
    #[test]
    fn binary_octal() {
        assert_eq!(format_with_spec(5.0, Some("b")), "101");
        assert_eq!(format_with_spec(8.0, Some("o")), "10");
    }
    #[test]
    fn currency_si_trim() {
        // The audit motivating case: "$~s" → "$1.5M".
        assert_eq!(format_with_spec(1.5e6, Some("$~s")), "$1.5M");
    }
    #[test]
    fn negative_si() {
        assert_eq!(format_with_spec(-1.5e6, Some("~s")), "-1.5M");
    }
    #[test]
    fn nan_inf_empty() {
        assert_eq!(format_with_spec(f64::NAN, Some(".2f")), "");
        assert_eq!(format_with_spec(f64::INFINITY, Some(".2f")), "");
    }

    // ---- parser unit checks ----
    #[test]
    fn parse_currency_comma_fixed() {
        let s = parse_format_spec("$,.2f");
        assert_eq!(s.symbol, Some('$'));
        assert!(s.comma);
        assert_eq!(s.precision, Some(2));
        assert_eq!(s.ty, 'f');
    }
    #[test]
    fn parse_trim_si() {
        let s = parse_format_spec("~s");
        assert!(s.trim);
        assert_eq!(s.ty, 's');
    }
    #[test]
    fn parse_zero_width() {
        let s = parse_format_spec("08.2f");
        assert!(s.zero);
        assert_eq!(s.width, Some(8));
        assert_eq!(s.precision, Some(2));
        assert_eq!(s.fill, '0');
        assert_eq!(s.align, Align::SignAware);
    }

    // ---- time (chrono) ----
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
    fn time_month_spacing() {
        // 2026-01-01 with monthly spacing → "Jan 2026".
        let s = format_time(1767225600000, 30 * 86_400_000);
        assert_eq!(s, "Jan 2026");
    }
    #[test]
    fn time_spec_month_year() {
        // 2020-01-01T00:00:00Z = 1577836800000 ms.
        assert_eq!(format_time_spec(1_577_836_800_000, "%b %Y"), "Jan 2020");
    }
    #[test]
    fn time_spec_iso_date() {
        assert_eq!(format_time_spec(1_577_836_800_000, "%Y-%m-%d"), "2020-01-01");
    }
    #[test]
    fn time_spec_hour_minute() {
        // 2020-01-01T15:30:00Z.
        let epoch = 1_577_836_800_000 + (15 * 3600 + 30 * 60) * 1000;
        assert_eq!(format_time_spec(epoch, "%H:%M"), "15:30");
    }

    // ---- ordinal ----
    #[test]
    fn ordinal_passthrough() {
        assert_eq!(format_ordinal("setosa"), "setosa");
    }
}
