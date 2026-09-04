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
//! `[[fill]align][sign][symbol][0][width][,][.precision][~][type]`, where
//! `type` is one of `e f g s % p r b o d x X c n` (or absent). See
//! [`parse_format_spec`] for the parser and [`format_with_spec`] for the
//! application. The `format_num` crate is a useful reference for the SI/rounding
//! algorithms but is intentionally not a dependency (it lacks `~`/`g`/`r`/`p`
//! and pulls `regex`).
//!
//! **`"__ordinal__"`** is a reserved sentinel (not real d3 grammar) that
//! [`format_presets::NUMERIC_PRESETS`](../../../../src/ferrum/format_presets.py)'s
//! `"ordinal"` preset resolves to; [`parse_format_spec`]/[`format_parsed`]
//! recognize it up front and dispatch to [`format_ordinal_number`] (1st, 2nd,
//! 3rd, …) instead of the d3 tokenizer (D8, spec §4.5).
//!
//! **Malformed-spec refusal** ([`validate_d3_format_spec`], NF-B1 residual,
//! 2026-09-02): a raw spec that is not a recognized preset passes through
//! Python's `resolve_format_or_raw` unresolved per spec §4.5's
//! unknown-name-passes-raw contract (a typo like `"curency"` is honest raw
//! input, not an error, at that layer) — but the lenient tokenizer below
//! silently drops any trailing characters it doesn't recognize once it has
//! consumed an optional leading type char (`"curency"` tokenizes as
//! `type='c'` — "format as Unicode code point" — with `"urency"` discarded),
//! which previously emitted raw control characters into rendered SVG text for
//! small tick values. [`validate_d3_format_spec`] performs the SAME tokenize
//! pass but requires the whole string to be consumed; callers run it ONCE per
//! resolved (format, format_type) pair before any per-value formatting
//! begins (`render::config_apply::validate_chart_format_specs`), not per tick/label.

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
///
/// **UTC by contract (F-L04-06):** `epoch_ms` is always formatted as UTC —
/// `DateTime::<Utc>::from_timestamp_millis` below, never a local-timezone
/// conversion. This holds for every temporal rendering path in the crate
/// ([`format_time_spec`]'s explicit-pattern formatter included), regardless
/// of a `TimeScale`'s `utc` flag, which is a wire-tag distinction only (see
/// [`crate::scale::time::TimeScale`]'s struct doc) — there is no
/// local-time rendering anywhere, ever (barred by the byte-determinism hard
/// constraint).
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
/// pattern (e.g. `"%b %Y"`, `"%Y-%m-%d"`, `"%H:%M"`). Returns the empty
/// string for an out-of-range TIMESTAMP (a legitimate, un-formattable-data
/// case — matches every other formatter in this module's NaN/Inf/out-of-
/// range convention). An invalid PATTERN is a different case entirely: see
/// below.
///
/// **Non-panicking by construction** (quality-review S4, 2026-09-03):
/// `chrono`'s `DelayedFormat` `Display` impl returns `Err` for an invalid
/// pattern, and `ToString::to_string()`'s blanket impl `.expect()`s that
/// `Display::fmt` never errors — calling it on a malformed pattern panics
/// with `"a Display implementation returned an error unexpectedly"`. Every
/// caller reaching here goes through `validate_chart_format_specs`'s
/// `validate_strftime_spec` gate first (which refuses a malformed pattern
/// with a typed `RenderError` before any formatting is attempted), so this
/// fallback is defense in depth, not the primary guard.
///
/// **The invalid-pattern fallback is deliberately NOT a blank string**
/// (quality-review cycle-4 correction: an unvalidated call site returning
/// `""` would render an axis of BLANK tick labels — the exact silent-empty
/// anti-pattern this batch's malformed-spec-refusal work exists to close,
/// just relocated one layer down instead of eliminated). Two lines of
/// defense instead: a `debug_assert!` makes an unvalidated call site loud
/// in test builds (every current caller is downstream of the validation
/// chokepoint, confirmed by walking the call graph — this should be
/// unreachable), and the release fallback returns the PATTERN ITSELF,
/// unformatted, rather than blank — `"%Y-%m-%d"` appearing verbatim as a
/// tick label is an obviously-wrong, debuggable signal; a blank label is
/// indistinguishable from "no label was ever meant to be here".
pub fn format_time_spec(epoch_ms: i64, pattern: &str) -> String {
    let Some(dt) = DateTime::<Utc>::from_timestamp_millis(epoch_ms) else {
        return String::new();
    };
    match try_format_time_spec(dt, pattern) {
        Some(out) => out,
        None => {
            debug_assert!(
                validate_strftime_spec(pattern).is_ok(),
                "format_time_spec received a pattern that failed to validate: {pattern:?} \
                 (every caller must go through validate_chart_format_specs first — this \
                 is a validation-coverage bug, not a user error)"
            );
            pattern.to_string()
        }
    }
}

/// The fallible core of [`format_time_spec`]: `None` when `pattern` is
/// invalid (`chrono`'s `DelayedFormat` `Display` returned an error), never
/// panicking. Split out so the detection itself (does `chrono` actually
/// reject this pattern) is directly testable without going through
/// `format_time_spec`'s `debug_assert!`, which is deliberately loud (panics
/// in test/debug builds) for the case this fn returns `None`.
fn try_format_time_spec(dt: DateTime<Utc>, pattern: &str) -> Option<String> {
    use std::fmt::Write as _;
    let mut out = String::new();
    write!(out, "{}", dt.format(pattern)).ok()?;
    Some(out)
}

/// Validate a `chrono` strftime pattern against the grammar
/// `chrono::format::StrftimeItems` implements. Returns `Err` naming the
/// unrecognized specifier for a malformed pattern (e.g. the typo'd preset
/// class `"curency%"`, or a genuinely dangling `"%"`/unknown specifier like
/// `"%J"`) — the strftime-grammar sibling of [`validate_d3_format_spec`],
/// closing the panic class quality-review found in cycle-2's
/// `is_time_format_spec`-gated exemption: a `%`-bearing spec with no
/// explicit `format_type` (every raw-accepting surface's default) was
/// skipped from d3-grammar validation on the premise `chrono` handles it
/// "leniently" — it does not; `format_time_spec` panics on exactly this
/// input. Called once per resolved (format, format_type) pair by
/// `render::config_apply::validate_chart_format_specs`, the same chokepoint
/// `validate_d3_format_spec` uses.
pub(crate) fn validate_strftime_spec(pattern: &str) -> Result<(), String> {
    chrono::format::StrftimeItems::new(pattern)
        .parse()
        .map(|_| ())
        .map_err(|e| format!("invalid strftime pattern {pattern:?}: {e}"))
}

/// Ordinal/threshold passthrough — caller already has a string.
///
/// NOT the `"ordinal"` FORMAT PRESET ([`format_ordinal_number`] below) —
/// this is the categorical `ScaleKind::Ordinal` tick-label passthrough
/// (`scale_resolve::mod.rs`'s `tick_labels()`), an unrelated feature that
/// happens to share the "ordinal" name.
pub fn format_ordinal(value: &str) -> String {
    value.to_string()
}

/// The `"ordinal"` format preset's sentinel spec string
/// (`format_presets.NUMERIC_PRESETS["ordinal"]`). [`parse_format_spec`]
/// recognizes it before running the d3 tokenizer.
const ORDINAL_SENTINEL: &str = "__ordinal__";

/// Marks a parsed [`D3Spec`] as the ordinal-suffix formatter. `'\u{1}'` (a
/// C0 control char) can never appear as a real d3 type char — d3 type chars
/// are always ASCII printable — so it is a safe sentinel value for
/// [`D3Spec::ty`] that [`format_parsed`] dispatches on before its normal
/// per-type match.
const ORDINAL_TYPE_MARKER: char = '\u{1}';

/// Format `v` as an integer with its English ordinal suffix: `1st`, `2nd`,
/// `3rd`, `4th`, …, `11th`, `12th`, `13th`, `21st`, … (D8, spec §4.5,
/// F-L07-05). Non-integer values fall back to [`format_numeric`] (the spec's
/// documented behavior: "non-integers fall back to plain formatting").
/// NaN/infinite values return the empty string, matching every other
/// formatter in this module.
pub fn format_ordinal_number(v: f64) -> String {
    if v.is_nan() || v.is_infinite() {
        return String::new();
    }
    if v.fract() != 0.0 {
        return format_numeric(v);
    }
    let n = v as i64;
    format!("{n}{}", ordinal_suffix(n))
}

/// The English ordinal suffix for an integer: `"th"` for the 11–13 exception
/// range (11th, 12th, 13th, 111th, 112th, 113th, …), else keyed off the last
/// digit (`1→st`, `2→nd`, `3→rd`, else `th`). Sign-agnostic (`-1` → `"st"`,
/// matching `-1st`).
fn ordinal_suffix(n: i64) -> &'static str {
    let abs = n.unsigned_abs();
    let last_two = abs % 100;
    if (11..=13).contains(&last_two) {
        return "th";
    }
    match abs % 10 {
        1 => "st",
        2 => "nd",
        3 => "rd",
        _ => "th",
    }
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

/// Parse a d3-format string into a [`D3Spec`], for a caller formatting an
/// ALREADY-validated spec value-by-value (per-tick, per-legend-entry, …).
/// Lenient: unrecognized trailing bytes are silently ignored (see
/// [`parse_format_spec_impl`]'s doc for why leniency here is safe — every
/// caller of this fn is downstream of [`validate_d3_format_spec`] having
/// already run once for the same spec string). To validate a spec BEFORE
/// formatting, use [`validate_d3_format_spec`] instead.
pub(crate) fn parse_format_spec(spec: &str) -> D3Spec {
    parse_format_spec_impl(spec).0
}

/// Tokenize a d3-format string, returning the parsed [`D3Spec`] and the
/// number of characters consumed. A well-formed spec consumes every
/// character (the optional trailing type char is the last one);
/// [`validate_d3_format_spec`] compares `consumed` against the spec's total
/// length to detect trailing garbage the tokenizer couldn't place — the
/// malformed-spec guard this module's doc comment describes.
///
/// Grammar: `[[fill]align][sign][symbol][0][width][,][.precision][~][type]`.
/// The `"__ordinal__"` sentinel (see [`ORDINAL_SENTINEL`]) is recognized up
/// front and treated as fully consumed, dispatching [`format_parsed`] to
/// [`format_ordinal_number`] instead of the type-char match below.
fn parse_format_spec_impl(spec: &str) -> (D3Spec, usize) {
    if spec == ORDINAL_SENTINEL {
        return (D3Spec { ty: ORDINAL_TYPE_MARKER, ..D3Spec::default() }, spec.chars().count());
    }
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
        i += 1;
    }

    (out, i)
}

/// Validate a raw d3-format spec against the grammar this module implements
/// (see the module doc's "Malformed-spec refusal" section). Returns `Err`
/// naming the first unrecognized trailing token when the tokenizer could not
/// place every character — genuinely malformed input like the typo'd preset
/// name `"curency"` (tokenizes as `type='c'` with `"urency"` left over).
/// Every syntactically valid-but-unusual d3 spec (fill/align/sign/symbol/
/// zero-pad/width/comma/precision/trim combinations, any single trailing
/// type char whether or not this module's [`format_parsed`] recognizes it)
/// consumes its whole string and passes. The empty string and the
/// `"__ordinal__"` sentinel are always valid.
pub(crate) fn validate_d3_format_spec(spec: &str) -> Result<(), String> {
    if spec.is_empty() {
        return Ok(());
    }
    let (_, consumed) = parse_format_spec_impl(spec);
    let total = spec.chars().count();
    if consumed < total {
        let tail: String = spec.chars().skip(consumed).collect();
        return Err(format!(
            "unrecognized token {tail:?} at position {consumed} in format spec {spec:?} \
             (grammar: [[fill]align][sign][symbol][0][width][,][.precision][~][type])"
        ));
    }
    Ok(())
}

/// Whether `fmt` should be treated as a `chrono` strftime TIME pattern rather
/// than a d3 numeric spec: an explicit `format_type == "time"`, or an
/// unset `format_type` paired with a `%`-bearing spec (the pre-existing
/// per-channel heuristic — a caller with no format_type at all still gets
/// `%b %Y`-style raw specs auto-detected as time). An explicit non-`"time"`
/// `format_type` (e.g. `"number"`, set by every preset resolution) always
/// wins — this is what keeps a `%`-bearing NUMERIC spec like the `"percent"`
/// preset's `.1%` from being misclassified as strftime (NF-B2, spec §4.5).
pub(crate) fn is_time_format_spec(fmt: &str, format_type: Option<&str>) -> bool {
    fmt.contains('%') && (format_type == Some("time") || format_type.is_none())
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

/// The value-level (not pre-formatted-string) sibling of
/// `prepare::apply_tick_format` — used where the caller already holds the
/// raw domain value `v` directly (colorbar/legend tick labels: D8's
/// `LegendStyleSpec.format_type` threading) rather than an
/// already-string-formatted axis label. `format_type == Some("time")` treats
/// `v` as an epoch-millisecond timestamp: an explicit `spec` is applied as a
/// `chrono` strftime pattern via [`format_time_spec`]; no spec falls back to
/// the default spacing-keyed [`format_time`] (day granularity, matching
/// `apply_tick_format`'s own no-pattern default). Any other `format_type`
/// (including `None`) formats `v` as a plain number via [`format_with_spec`]
/// (which also handles the `"__ordinal__"` sentinel).
pub fn format_value_with_spec(v: f64, spec: Option<&str>, format_type: Option<&str>) -> String {
    if format_type == Some("time") {
        let epoch_ms = v as i64;
        return match spec {
            Some(pattern) => format_time_spec(epoch_ms, pattern),
            None => format_time(epoch_ms, 86_400_000),
        };
    }
    format_with_spec(v, spec)
}

/// Apply an already-parsed [`D3Spec`] to a value. Shared entry point so
/// `apply_tick_format` can parse once and reuse across a column of labels.
pub(crate) fn format_parsed(v: f64, spec: &D3Spec) -> String {
    if v.is_nan() || v.is_infinite() {
        return String::new();
    }

    // The `"__ordinal__"` sentinel (see `ORDINAL_TYPE_MARKER`'s doc):
    // dispatch to the ordinal-suffix formatter, bypassing the rest of the
    // d3-grammar match entirely (width/fill/precision/etc. are meaningless
    // for it, matching how the `c`/type-char formatters below also skip the
    // general numeric pipeline).
    if spec.ty == ORDINAL_TYPE_MARKER {
        return format_ordinal_number(v);
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

    // ---- ordinal (categorical scale passthrough — unrelated to the format
    // preset below; see `format_ordinal`'s doc) ----
    #[test]
    fn ordinal_passthrough() {
        assert_eq!(format_ordinal("setosa"), "setosa");
    }

    // ---- "ordinal" format preset (D8, F-L07-05): real 1st/2nd/3rd suffixes ----
    #[test]
    fn ordinal_number_basic_suffixes() {
        assert_eq!(format_ordinal_number(1.0), "1st");
        assert_eq!(format_ordinal_number(2.0), "2nd");
        assert_eq!(format_ordinal_number(3.0), "3rd");
        assert_eq!(format_ordinal_number(4.0), "4th");
        assert_eq!(format_ordinal_number(0.0), "0th");
    }
    #[test]
    fn ordinal_number_teens_exception() {
        // 11/12/13 (and their higher-decade repeats) are always "th", not
        // the last-digit-keyed st/nd/rd.
        assert_eq!(format_ordinal_number(11.0), "11th");
        assert_eq!(format_ordinal_number(12.0), "12th");
        assert_eq!(format_ordinal_number(13.0), "13th");
        assert_eq!(format_ordinal_number(111.0), "111th");
        assert_eq!(format_ordinal_number(112.0), "112th");
        assert_eq!(format_ordinal_number(113.0), "113th");
    }
    #[test]
    fn ordinal_number_higher_decades() {
        assert_eq!(format_ordinal_number(21.0), "21st");
        assert_eq!(format_ordinal_number(22.0), "22nd");
        assert_eq!(format_ordinal_number(23.0), "23rd");
        assert_eq!(format_ordinal_number(101.0), "101st");
    }
    #[test]
    fn ordinal_number_non_integer_falls_back_to_plain() {
        // Spec §4.5: "non-integers fall back to plain formatting".
        assert_eq!(format_ordinal_number(1.5), format_numeric(1.5));
    }
    #[test]
    fn ordinal_number_nan_inf_empty() {
        assert_eq!(format_ordinal_number(f64::NAN), "");
        assert_eq!(format_ordinal_number(f64::INFINITY), "");
    }
    #[test]
    fn ordinal_sentinel_dispatches_via_format_with_spec() {
        // The sentinel a resolved "ordinal" preset carries end to end.
        assert_eq!(format_with_spec(2.0, Some("__ordinal__")), "2nd");
        assert_eq!(format_with_spec(13.0, Some("__ordinal__")), "13th");
    }

    // ---- malformed-spec refusal (NF-B1 residual, 2026-09-02) ----
    #[test]
    fn validate_rejects_typo_preset_name() {
        // The audit repro: "curency" (typo of "currency") tokenizes as
        // type='c' with "urency" left over — must be refused, not silently
        // truncated into the codepoint formatter.
        let err = validate_d3_format_spec("curency").unwrap_err();
        assert!(err.contains("curency"), "{err}");
        assert!(err.contains("urency"), "{err}");
    }
    #[test]
    fn validate_accepts_empty_and_ordinal_sentinel() {
        assert!(validate_d3_format_spec("").is_ok());
        assert!(validate_d3_format_spec("__ordinal__").is_ok());
    }
    #[test]
    fn validate_accepts_every_valid_but_unusual_spec_already_pinned_above() {
        // Every spec string exercised by the `format_with_spec` tests above
        // must validate clean — the malformed-spec guard must never flag a
        // genuinely valid d3 spec, however exotic.
        for spec in [
            ",.0f", "~s", ".2s", ".3s", ".1%", "$,.2f", "+.1f", " .1f", "~e", ".2e", "~f", ".3r",
            ".3g", ".1p", ",.0f", ",", "d", ",d", "05d", "<5d", "^5d", "*>5d", "x", "#x", "X",
            "b", "o", "$~s",
        ] {
            assert!(validate_d3_format_spec(spec).is_ok(), "expected {spec:?} to validate clean");
        }
    }
    #[test]
    fn validate_rejects_trailing_garbage_after_valid_prefix() {
        // A well-formed prefix followed by unrecognized characters — not
        // just a bare typo'd word.
        assert!(validate_d3_format_spec(",.2fxyz").is_err());
    }

    // ---- is_time_format_spec (NF-B2: %-bearing numeric spec vs strftime) ----
    #[test]
    fn is_time_format_spec_explicit_time_wins() {
        assert!(is_time_format_spec("%Y-%m-%d", Some("time")));
    }
    #[test]
    fn is_time_format_spec_unset_type_with_percent_defaults_time() {
        assert!(is_time_format_spec("%b %Y", None));
    }
    #[test]
    fn is_time_format_spec_explicit_number_never_misclassified() {
        // NF-B2's exact motivating case: the "percent" preset (".1%") always
        // carries an explicit format_type="number" — it must never be
        // treated as strftime just because it contains '%'.
        assert!(!is_time_format_spec(".1%", Some("number")));
    }
    #[test]
    fn is_time_format_spec_no_percent_never_time() {
        assert!(!is_time_format_spec(",.0f", None));
    }

    // ---- validate_strftime_spec (quality-review S4, 2026-09-03: the
    // %-bearing-spec panic class) ----
    #[test]
    fn validate_strftime_rejects_typo_preset_name_with_percent() {
        // The exact repro: fm.X("t:T", axis=fm.Axis(label_format="curency%")).
        assert!(validate_strftime_spec("curency%").is_err());
    }
    #[test]
    fn validate_strftime_rejects_unknown_specifier() {
        assert!(validate_strftime_spec("%J").is_err());
    }
    #[test]
    fn validate_strftime_rejects_dangling_percent() {
        assert!(validate_strftime_spec("%").is_err());
        assert!(validate_strftime_spec(".1%").is_err());
    }
    #[test]
    fn validate_strftime_accepts_every_time_preset() {
        // format_presets.py's TIME_PRESETS values -- every one must validate
        // clean, including the GNU no-pad `%-d`/`%-I` modifiers.
        for spec in [
            "%b %-d", "%B %-d, %Y", "%Y-%m-%d", "%b", "%b %Y", "%Y", "%H:%M", "%-I:%M %p",
            "%b %-d, %H:%M",
        ] {
            assert!(validate_strftime_spec(spec).is_ok(), "expected {spec:?} to validate clean");
        }
    }
    #[test]
    fn validate_strftime_accepts_every_pinned_test_spec() {
        // Every %-bearing spec already exercised by this module's own or
        // mod.rs's pinned tests must stay valid.
        for spec in ["%m/%d", "%b %d", "%b %Y", "%Y-%m-%d"] {
            assert!(validate_strftime_spec(spec).is_ok(), "expected {spec:?} to validate clean");
        }
    }

    // ---- format_time_spec: non-panicking core, loud-in-debug wrapper
    // (quality-review cycle-4 correction: the fallback is no longer a
    // blank string — see try_format_time_spec/format_time_spec's docs) ----
    #[test]
    fn try_format_time_spec_detects_every_malformed_pattern() {
        // The fallible core: proves chrono actually rejects each pattern,
        // without going through format_time_spec's debug_assert (which
        // deliberately panics in test builds for exactly this input).
        let dt = DateTime::<Utc>::from_timestamp_millis(0).unwrap();
        assert_eq!(try_format_time_spec(dt, "curency%"), None);
        assert_eq!(try_format_time_spec(dt, "%J"), None);
        assert_eq!(try_format_time_spec(dt, "%"), None);
        assert_eq!(try_format_time_spec(dt, ".1%"), None);
    }
    #[test]
    fn try_format_time_spec_valid_pattern_formats() {
        let dt = DateTime::<Utc>::from_timestamp_millis(1_577_836_800_000).unwrap();
        assert_eq!(try_format_time_spec(dt, "%Y-%m-%d").as_deref(), Some("2020-01-01"));
    }
    #[test]
    #[should_panic(expected = "format_time_spec received a pattern that failed to validate")]
    fn format_time_spec_malformed_pattern_panics_loudly_in_debug() {
        // The debug_assert is the primary signal a call site skipped
        // validation — every current caller goes through
        // validate_chart_format_specs first, so this should be unreachable
        // in production; test builds must not let it slide by silently.
        format_time_spec(0, "curency%");
    }
    #[test]
    fn format_time_spec_valid_pattern_unaffected() {
        assert_eq!(format_time_spec(1_577_836_800_000, "%Y-%m-%d"), "2020-01-01");
    }
    #[test]
    fn format_time_spec_out_of_range_timestamp_returns_empty() {
        // A legitimate, un-formattable-DATA case (not a pattern problem) —
        // matches every other formatter's NaN/Inf/out-of-range convention.
        // A valid pattern is still supplied so this cannot trip the
        // debug_assert (which only fires for an INVALID pattern).
        assert_eq!(format_time_spec(i64::MAX, "%Y-%m-%d"), "");
    }

    // ---- format_value_with_spec (LegendStyleSpec.format_type threading) ----
    #[test]
    fn format_value_with_spec_time_uses_pattern() {
        assert_eq!(
            format_value_with_spec(1_577_836_800_000.0, Some("%Y-%m-%d"), Some("time")),
            "2020-01-01"
        );
    }
    #[test]
    fn format_value_with_spec_time_no_pattern_uses_default() {
        assert_eq!(
            format_value_with_spec(1_577_836_800_000.0, None, Some("time")),
            format_time(1_577_836_800_000, 86_400_000)
        );
    }
    #[test]
    fn format_value_with_spec_numeric_matches_format_with_spec() {
        assert_eq!(format_value_with_spec(1234.0, Some(",.0f"), None), "1,234");
        assert_eq!(format_value_with_spec(1234.0, Some(",.0f"), Some("number")), "1,234");
    }
    #[test]
    fn format_value_with_spec_ordinal() {
        assert_eq!(format_value_with_spec(2.0, Some("__ordinal__"), Some("number")), "2nd");
    }
}
