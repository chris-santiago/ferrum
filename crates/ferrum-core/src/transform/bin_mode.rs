//! Shared bin-count selection mode for the 1-D `Bin` and 2-D `Bin2D` transforms.
//!
//! `BinMode` makes the "which rule fires" decision a typed enum — illegal states
//! (e.g. both a fixed count and a fixed width) unrepresentable — and is reused by
//! both binning transforms with one shared bin-edge resolver. `Bin2D` constructs
//! it directly from its `bins_x`/`bins_y` Python args via [`parse_bin_axis`]; the
//! 1-D `Bin` accepts the same rule-bearing spelling through its own `bins` param
//! (#193) and additionally keeps its flat `bin_count`/`bin_width` params, which
//! it resolves to a `BinMode` at the boundary (see `bin.rs`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::scale::ticks::sturges_floor;

/// How a binning transform chooses its bin count.
///
/// Serde shape is a tagged enum (`{"kind":"sturges"}`, `{"kind":"fixed","n":10}`,
/// `{"kind":"width","w":2.0}`, `{"kind":"freedman_diaconis"}`) — this is the wire
/// `Bin2D` already emits via its `bins_x`/`bins_y` fields. The 1-D `Bin` does NOT
/// serialize `BinMode` directly; it projects to/from the flat
/// `bin_count`/`bin_width`/`bin_rule` keys via a serde shim (see
/// `bin.rs::BinWire`), which carries every mode — including
/// `FreedmanDiaconis` — round-trip intact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BinMode {
    Sturges,
    FreedmanDiaconis,
    Fixed { n: usize },
    Width { w: f64 },
}

impl BinMode {
    /// The canonical wire/refusal name of the modes that derive their bin count
    /// from a *rule* rather than from a number the caller supplied.
    ///
    /// One table, read by both [`parse_bin_axis`] (which accepts every spelling
    /// that maps onto a name here) and the 1-D `Bin` wire shim (`bin.rs`'s
    /// `BinWire::bin_rule`), so a rule cannot be parseable but unserializable —
    /// the #193 wire hazard, where `FreedmanDiaconis` parsed fine and then
    /// serialized as Sturges.
    ///
    /// `Fixed`/`Width` return `None`: they are carried by the numeric wire keys
    /// (`bin_count`/`bin_width`) and by the numeric constructor params.
    pub(crate) fn rule_name(&self) -> Option<&'static str> {
        match self {
            BinMode::Sturges => Some("sturges"),
            BinMode::FreedmanDiaconis => Some("freedman_diaconis"),
            BinMode::Fixed { .. } | BinMode::Width { .. } => None,
        }
    }

    /// The rule name a mode must be spelled out under because nothing else in
    /// the 1-D projection can express it — `None` when the mode is already
    /// fully carried by something else.
    ///
    /// `Fixed`/`Width` are carried by the numeric `bin_count`/`bin_width`
    /// spellings, and `Sturges` is the documented default that both the wire
    /// and `__repr__` express as the *absence* of any mode key, so every
    /// pre-#193 payload and repr keeps its exact bytes. Only Freedman-Diaconis
    /// is left with nothing to say it. Shared by the wire shim
    /// (`bin.rs::BinWire::bin_rule`) and `PyBin::__repr__` so the two cannot
    /// disagree about which modes need spelling out.
    pub(crate) fn explicit_rule_name(&self) -> Option<&'static str> {
        match self {
            BinMode::FreedmanDiaconis => self.rule_name(),
            BinMode::Sturges | BinMode::Fixed { .. } | BinMode::Width { .. } => None,
        }
    }

    /// Inverse of [`rule_name`](Self::rule_name), also accepting the `"fd"`
    /// shorthand. `None` for anything outside the rule vocabulary.
    pub(crate) fn from_rule_name(name: &str) -> Option<BinMode> {
        match name {
            "sturges" => Some(BinMode::Sturges),
            "fd" | "freedman_diaconis" => Some(BinMode::FreedmanDiaconis),
            _ => None,
        }
    }

    /// The bin count this mode knows *without* an extent — `Some` only for
    /// `Fixed`.
    ///
    /// NICING-TARGET CONTRACT (#193): nicing rounds the extent against a target
    /// bin count, but every rule-derived mode (`Sturges`, `Width`,
    /// `FreedmanDiaconis`) computes its count *from* the extent, so using its
    /// own count as the nicing target would be circular. All three therefore
    /// nice against `sturges_floor(n)` — a stable, extent-independent target —
    /// and only then resolve their real count over the niced range. That is
    /// exactly what `Width` has always done; FD joins it rather than inventing a
    /// second rule. Both the per-group path (`bin.rs`'s `resolve_bin_extent`,
    /// shared with `Kde`'s count-scale derivation) and the faceted pin
    /// (`bin.rs`'s `global_extent`) ask this one question, so a panel and its
    /// facet-wide pin cannot disagree about where the edges fall.
    pub(crate) fn nice_target_count(&self) -> Option<usize> {
        match self {
            BinMode::Fixed { n } => Some(*n),
            BinMode::Sturges | BinMode::FreedmanDiaconis | BinMode::Width { .. } => None,
        }
    }
}

/// Compute IQR for a sorted slice. Used only by the Freedman-Diaconis arm.
fn iqr_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 4 {
        return 0.0;
    }
    let q1 = percentile_sorted(sorted, 0.25);
    let q3 = percentile_sorted(sorted, 0.75);
    q3 - q1
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (hi as f64 - h) + sorted[hi] * (h - lo as f64)
    }
}

/// Resolve a [`BinMode`] to a bin count given the (clean) values and data range.
///
/// `ctx` prefixes the error messages so each caller surfaces its own transform
/// name (`"Bin2D"` for the 2-D transform, `"stat_bin"` for the 1-D one).
pub(crate) fn resolve_bin_count(
    mode: &BinMode,
    vals: &[f64],
    lo: f64,
    hi: f64,
    ctx: &str,
) -> PyResult<usize> {
    match mode {
        BinMode::Sturges => Ok(sturges_floor(vals.len())),
        BinMode::FreedmanDiaconis => {
            let mut sorted = vals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let iqr_val = iqr_sorted(&sorted);
            let h = 2.0 * iqr_val * (vals.len() as f64).powf(-1.0 / 3.0);
            if h > 0.0 && h.is_finite() {
                Ok(((hi - lo) / h).ceil().max(1.0) as usize)
            } else {
                // IQR == 0 or degenerate → fall back to Sturges
                Ok(sturges_floor(vals.len()))
            }
        }
        BinMode::Fixed { n } => {
            if *n == 0 {
                return Err(PyValueError::new_err(format!(
                    "{ctx}: Fixed bin count must be >= 1"
                )));
            }
            Ok(*n)
        }
        BinMode::Width { w } => {
            if !w.is_finite() || *w <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{ctx}: Width must be a positive finite number"
                )));
            }
            Ok(((hi - lo) / w).ceil().max(1.0) as usize)
        }
    }
}

/// Parse a Python `bins`-axis value into a [`BinMode`].
///
/// Accepts a rule string (`"sturges"` / `"fd"` / `"freedman_diaconis"`), an int
/// (fixed count), or a float (fixed width), and refuses everything else by name.
/// This is the single owner of the `bins` vocabulary for all three binning
/// entries: `Bin2D`'s constructor (`ctx = "Bin2D"`), `Kde`'s `count_scale`
/// (`ctx = "stat_kde"`, batch F) and — since #193 — the 1-D `Bin`'s `bins` param
/// and the figure layer's [`validate_bin_rule`]. `ctx` prefixes every error
/// message so each caller's error names the entry the user actually invoked,
/// mirroring `resolve_bin_count`'s own `ctx` parameter.
///
/// Two refusals the arms below make *before* falling through, because falling
/// through silently changed the caller's meaning (spec §6, #193):
///
/// - a Python `bool` is not an integer bin count. `bool` subclasses `int`, so
///   `True` used to satisfy the integer arm and request a one-bin histogram.
/// - a non-positive integer is not a bin width. `-3` used to fail the `usize`
///   arm and land in the float arm as `Width { w: -3.0 }`, i.e. the caller's
///   *count* silently became a (nonsensical) *width*.
pub(crate) fn parse_bin_axis(obj: &Bound<'_, PyAny>, ctx: &str) -> PyResult<BinMode> {
    if let Ok(s) = obj.extract::<&str>() {
        return BinMode::from_rule_name(s).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{ctx}: unknown bins value '{s}'; expected 'sturges'|'fd'|int|float"
            ))
        });
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "{ctx}: bins must not be a bool ({obj}); expected 'sturges'|'fd'|int|float"
        )));
    }
    // Integer arm, gated on Python's integer protocol (`__index__`) rather than
    // on a successful `usize` extraction: a numpy integer is accepted exactly
    // like a builtin `int`, a `float` is not, and a negative or out-of-range
    // integer refuses HERE instead of falling through to the width arm.
    if let Ok(index) = obj.call_method0("__index__") {
        return match index.extract::<usize>() {
            Ok(n) if n >= 1 => Ok(BinMode::Fixed { n }),
            _ => Err(PyValueError::new_err(format!(
                "{ctx}: bins as an int must be a bin count >= 1 within machine range; got {obj}"
            ))),
        };
    }
    if let Ok(w) = obj.extract::<f64>() {
        if !w.is_finite() || w <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "{ctx}: bins as a float must be a positive finite bin width; got {w}"
            )));
        }
        return Ok(BinMode::Width { w });
    }
    Err(PyValueError::new_err(format!(
        "{ctx}: bins must be 'sturges'|'fd'|int|float",
    )))
}

/// Replay [`parse_bin_axis`] under a caller-supplied `ctx`, for its refusal only.
///
/// Exists so a figure-level entry can refuse a malformed `bins=` **in its own
/// voice** — `displot: unknown bins value 'auto'; …` rather than naming the
/// private `Bin` transform the figure happens to build — without restating the
/// accepted vocabulary in Python, where it would immediately be free to drift
/// from the parser above. Same shape as `validate_scale_dict`
/// (`spec/encoding.rs`): ask the real gate directly instead of constructing a
/// throwaway production object to catch its constructor's exception.
///
/// `ctx` is the public entry name the caller invoked (`"displot"`), matching
/// Python's own `validate_choice(func, …)` convention.
#[pyfunction]
pub fn validate_bin_rule(bins: &Bound<'_, PyAny>, ctx: &str) -> PyResult<()> {
    parse_bin_axis(bins, ctx).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sturges resolves to `sturges_floor(n)` regardless of range.
    #[test]
    fn resolve_sturges_matches_sturges_floor() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let got = resolve_bin_count(&BinMode::Sturges, &vals, 0.0, 99.0, "stat_bin").unwrap();
        assert_eq!(got, sturges_floor(100));
    }

    /// Fixed{n} returns n verbatim for n >= 1.
    #[test]
    fn resolve_fixed_returns_n() {
        let vals = vec![1.0, 2.0, 3.0];
        let got = resolve_bin_count(&BinMode::Fixed { n: 7 }, &vals, 0.0, 10.0, "stat_bin").unwrap();
        assert_eq!(got, 7);
    }

    /// Fixed{0} errors with the ctx-prefixed message.
    #[test]
    fn resolve_fixed_zero_errors_with_ctx() {
        pyo3::Python::initialize();
        let vals = vec![1.0];
        let err =
            resolve_bin_count(&BinMode::Fixed { n: 0 }, &vals, 0.0, 1.0, "stat_bin").unwrap_err();
        assert!(err.to_string().contains("stat_bin: Fixed bin count must be >= 1"));
    }

    /// Width{w} → ceil((hi-lo)/w).max(1) — matches the pre-refactor inline match.
    #[test]
    fn resolve_width_matches_inline_formula() {
        // (10 - 0) / 2.0 = 5.0 → ceil = 5
        let vals = vec![0.0, 5.0, 10.0];
        let got = resolve_bin_count(&BinMode::Width { w: 2.0 }, &vals, 0.0, 10.0, "stat_bin").unwrap();
        assert_eq!(got, 5);
        // Non-integer division rounds up: (10 - 0) / 3.0 = 3.33 → ceil = 4
        let got4 = resolve_bin_count(&BinMode::Width { w: 3.0 }, &vals, 0.0, 10.0, "stat_bin").unwrap();
        assert_eq!(got4, 4);
    }

    /// Width{<=0} or non-finite errors with the ctx-prefixed message.
    #[test]
    fn resolve_width_nonpositive_errors_with_ctx() {
        pyo3::Python::initialize();
        let vals = vec![1.0];
        let err =
            resolve_bin_count(&BinMode::Width { w: 0.0 }, &vals, 0.0, 1.0, "Bin2D").unwrap_err();
        assert!(err.to_string().contains("Bin2D: Width must be a positive finite number"));
    }

    /// `parse_bin_axis`'s unknown-string-value error must be prefixed by
    /// whatever `ctx` the caller passes — regression pin for the batch-F
    /// cycle-2 fix (a `Kde(count_scale='bogus')` error previously named
    /// `Bin2D`, a transform the user never called, because `parse_bin_axis`
    /// hardcoded that prefix from its sole prior caller).
    #[test]
    fn parse_bin_axis_unknown_string_errors_with_kde_ctx() {
        use pyo3::types::PyString;
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let obj = PyString::new(py, "bogus").into_any();
            let err = parse_bin_axis(&obj, "stat_kde").unwrap_err();
            assert!(
                err.to_string().contains("stat_kde: unknown bins value 'bogus'"),
                "err: {err}"
            );
        });
    }

    /// `Bin2D`'s own unknown-string-value error text must stay byte-identical
    /// after threading `ctx` through — same call, same `ctx` it always
    /// passed, so this is the "no behavior change for the existing caller"
    /// half of the cycle-2 fix.
    #[test]
    fn parse_bin_axis_unknown_string_errors_with_bin2d_ctx_byte_identical() {
        use pyo3::types::PyString;
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let obj = PyString::new(py, "bogus").into_any();
            let err = parse_bin_axis(&obj, "Bin2D").unwrap_err();
            assert_eq!(
                err.to_string(),
                "ValueError: Bin2D: unknown bins value 'bogus'; expected 'sturges'|'fd'|int|float"
            );
        });
    }

    /// The second `parse_bin_axis` error arm (wrong Python type entirely —
    /// not a string, int, or float) must ALSO carry the caller's `ctx`, not
    /// just the string-value arm.
    #[test]
    fn parse_bin_axis_wrong_type_errors_with_ctx() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let obj = py.None().into_bound(py);
            let err = parse_bin_axis(&obj, "stat_kde").unwrap_err();
            assert_eq!(
                err.to_string(),
                "ValueError: stat_kde: bins must be 'sturges'|'fd'|int|float"
            );
        });
    }

    // ── #193: the two properties the parser did not previously guarantee ───────

    /// Evaluate a Python expression and parse it as a `bins` value.
    fn parse_py(py: Python<'_>, expr: &str, ctx: &str) -> PyResult<BinMode> {
        let obj = py
            .eval(std::ffi::CString::new(expr).unwrap().as_c_str(), None, None)
            .unwrap();
        parse_bin_axis(&obj, ctx)
    }

    /// A Python `bool` must NOT satisfy the integer arm. `bool` subclasses
    /// `int`, so `True` used to parse as `Fixed{1}` — a one-bin histogram from
    /// a caller who never asked for a count at all.
    #[test]
    fn parse_bin_axis_refuses_bool_instead_of_reading_it_as_a_count() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            for expr in ["True", "False"] {
                let err = parse_py(py, expr, "displot").unwrap_err().to_string();
                assert!(
                    err.contains("displot: bins must not be a bool"),
                    "{expr} must refuse as a bool, got: {err}"
                );
            }
        });
    }

    /// A negative integer must refuse in the INTEGER arm. It used to fail the
    /// `usize` extraction and fall through to the float arm, where the caller's
    /// bin *count* silently became a bin *width* of -3.
    #[test]
    fn parse_bin_axis_refuses_negative_int_instead_of_reading_it_as_a_width() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let err = parse_py(py, "-3", "displot").unwrap_err().to_string();
            assert!(
                err.contains("bins as an int must be a bin count >= 1"),
                "a negative int must refuse as a count, got: {err}"
            );
            assert!(err.contains("got -3"), "the refusal must name the value: {err}");
            // Zero refuses at the same boundary, rather than constructing
            // `Fixed{0}` and erroring later at apply time.
            let zero = parse_py(py, "0", "displot").unwrap_err().to_string();
            assert!(zero.contains("bins as an int must be a bin count >= 1"), "got: {zero}");
            // An integer too large for `usize` refuses here too — it must not
            // reach the float arm and become a ~1e30 bin width.
            let huge = parse_py(py, "10**30", "displot").unwrap_err().to_string();
            assert!(huge.contains("within machine range"), "got: {huge}");
        });
    }

    /// A positive integer still parses as a fixed count, and an object that is
    /// integral by `__index__` (a numpy integer is the real case) parses the
    /// same way rather than degrading.
    #[test]
    fn parse_bin_axis_accepts_index_protocol_integers() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            assert_eq!(parse_py(py, "9", "displot").unwrap(), BinMode::Fixed { n: 9 });
            let expr = "type('I', (), {'__index__': lambda self: 9})()";
            assert_eq!(parse_py(py, expr, "displot").unwrap(), BinMode::Fixed { n: 9 });
        });
    }

    /// A float is a bin WIDTH, and a width that cannot bin anything refuses at
    /// the boundary instead of constructing a spec that explodes at apply time.
    #[test]
    fn parse_bin_axis_float_is_a_width_and_refuses_nonpositive_or_nonfinite() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            assert_eq!(parse_py(py, "0.5", "displot").unwrap(), BinMode::Width { w: 0.5 });
            for expr in ["0.0", "-0.5", "float('nan')", "float('inf')"] {
                let err = parse_py(py, expr, "displot").unwrap_err().to_string();
                assert!(
                    err.contains("bins as a float must be a positive finite bin width"),
                    "{expr} must refuse, got: {err}"
                );
            }
        });
    }

    /// `validate_bin_rule` is the same gate under the caller's own name: it
    /// accepts exactly what `parse_bin_axis` accepts and refuses with the
    /// caller's `ctx` (the public entry) as the prefix, so a figure-level
    /// refusal never names a private transform the user did not invoke.
    #[test]
    fn validate_bin_rule_mirrors_the_parser_under_the_callers_name() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            for expr in ["'sturges'", "'fd'", "'freedman_diaconis'", "7", "0.5"] {
                let obj = py
                    .eval(std::ffi::CString::new(expr).unwrap().as_c_str(), None, None)
                    .unwrap();
                assert!(validate_bin_rule(&obj, "displot").is_ok(), "{expr} must validate");
            }
            let obj = py.eval(c"'auto'", None, None).unwrap();
            let err = validate_bin_rule(&obj, "displot").unwrap_err().to_string();
            assert_eq!(
                err,
                "ValueError: displot: unknown bins value 'auto'; expected 'sturges'|'fd'|int|float"
            );
        });
    }

    /// The rule table is shared: every name `from_rule_name` accepts is a name
    /// `parse_bin_axis` accepts, and every rule mode can spell itself back out
    /// via `rule_name` — the property that makes the wire carrier safe.
    #[test]
    fn rule_name_and_from_rule_name_are_inverses() {
        for mode in [BinMode::Sturges, BinMode::FreedmanDiaconis] {
            let name = mode.rule_name().expect("rule modes must have a name");
            assert_eq!(BinMode::from_rule_name(name), Some(mode));
        }
        assert_eq!(BinMode::Fixed { n: 3 }.rule_name(), None);
        assert_eq!(BinMode::Width { w: 1.0 }.rule_name(), None);
        assert_eq!(BinMode::from_rule_name("auto"), None);
    }

    /// `explicit_rule_name` is `Some` only for the mode the 1-D projection
    /// cannot otherwise express — the property that keeps every pre-#193 wire
    /// payload and `Bin(...)` repr byte-identical while FD gains a spelling.
    #[test]
    fn explicit_rule_name_is_some_only_for_freedman_diaconis() {
        assert_eq!(
            BinMode::FreedmanDiaconis.explicit_rule_name(),
            Some("freedman_diaconis")
        );
        assert_eq!(BinMode::Sturges.explicit_rule_name(), None);
        assert_eq!(BinMode::Fixed { n: 3 }.explicit_rule_name(), None);
        assert_eq!(BinMode::Width { w: 1.0 }.explicit_rule_name(), None);
    }

    /// Only `Fixed` knows its bin count before an extent exists; every
    /// rule-derived mode answers `None` and therefore nices against the Sturges
    /// count (NICING-TARGET CONTRACT — see `nice_target_count`'s doc).
    #[test]
    fn nice_target_count_is_some_only_for_fixed() {
        assert_eq!(BinMode::Fixed { n: 12 }.nice_target_count(), Some(12));
        assert_eq!(BinMode::Sturges.nice_target_count(), None);
        assert_eq!(BinMode::Width { w: 2.0 }.nice_target_count(), None);
        assert_eq!(BinMode::FreedmanDiaconis.nice_target_count(), None);
    }

    // ── BUG-HUNT (step4): degenerate / boundary resolve_bin_count inputs ────────

    /// Width{w} must reject a NaN width — a hand-authored / JSON-`1e400`-coerced
    /// width must error, not silently produce a bin count.
    #[test]
    fn bughunt_resolve_width_nan_errors() {
        pyo3::Python::initialize();
        let vals = vec![1.0, 2.0, 3.0];
        let err = resolve_bin_count(&BinMode::Width { w: f64::NAN }, &vals, 0.0, 10.0, "stat_bin")
            .unwrap_err();
        assert!(
            err.to_string().contains("positive finite"),
            "NaN width must error: {err}"
        );
    }

    /// Width{w} must reject +inf (the value a JSON `1e400` literal coerces to via
    /// serde_json) instead of computing `ceil((hi-lo)/inf)=0 → max(1)=1`.
    #[test]
    fn bughunt_resolve_width_infinity_errors() {
        pyo3::Python::initialize();
        let vals = vec![1.0, 2.0, 3.0];
        let err =
            resolve_bin_count(&BinMode::Width { w: f64::INFINITY }, &vals, 0.0, 10.0, "stat_bin")
                .unwrap_err();
        assert!(
            err.to_string().contains("positive finite"),
            "inf width must error: {err}"
        );
    }

    /// Width{w} must reject a negative width.
    #[test]
    fn bughunt_resolve_width_negative_errors() {
        pyo3::Python::initialize();
        let vals = vec![1.0, 2.0, 3.0];
        let err = resolve_bin_count(&BinMode::Width { w: -1.0 }, &vals, 0.0, 10.0, "stat_bin")
            .unwrap_err();
        assert!(err.to_string().contains("positive finite"), "neg width must error: {err}");
    }

    /// Width over a degenerate (lo==hi) range: `(hi-lo)/w = 0 → ceil → 0 → max(1)`.
    /// The bin count must clamp to at least 1; never 0 (which would later cause a
    /// div-by-zero in the bin-edge stride `(hi-lo)/n_bins`).
    #[test]
    fn bughunt_resolve_width_zero_range_clamps_to_one() {
        pyo3::Python::initialize();
        let vals = vec![5.0, 5.0, 5.0];
        let n = resolve_bin_count(&BinMode::Width { w: 2.0 }, &vals, 5.0, 5.0, "stat_bin").unwrap();
        assert_eq!(n, 1, "zero-range width must clamp to >=1 bin, got {n}");
    }

    /// Freedman-Diaconis on a degenerate all-equal slice has IQR==0, so h==0 and
    /// the arm must fall back to Sturges rather than divide by zero.
    #[test]
    fn bughunt_resolve_fd_zero_iqr_falls_back_to_sturges() {
        pyo3::Python::initialize();
        let vals = vec![5.0; 16];
        let n = resolve_bin_count(&BinMode::FreedmanDiaconis, &vals, 5.0, 5.0, "stat_bin").unwrap();
        assert_eq!(n, sturges_floor(16), "FD with IQR=0 must fall back to Sturges, got {n}");
    }

    /// FD on fewer than 4 points: `iqr_sorted` returns 0 for n<4, so the arm
    /// must fall back to Sturges and never panic on the `sorted[lo]` indexing.
    #[test]
    fn bughunt_resolve_fd_small_n_falls_back_without_panic() {
        pyo3::Python::initialize();
        for n_pts in 1..=3 {
            let vals: Vec<f64> = (0..n_pts).map(|i| i as f64).collect();
            let n = resolve_bin_count(&BinMode::FreedmanDiaconis, &vals, 0.0, 2.0, "stat_bin")
                .unwrap();
            assert_eq!(
                n,
                sturges_floor(n_pts),
                "FD on n={n_pts} must fall back to Sturges"
            );
        }
    }

    /// Sturges over an empty value slice: `sturges_floor(0)` is defined as 1, so
    /// the resolver must return 1 (a later div-by-n_bins is then safe).
    #[test]
    fn bughunt_resolve_sturges_empty_slice_returns_one() {
        let vals: Vec<f64> = vec![];
        let n = resolve_bin_count(&BinMode::Sturges, &vals, 0.0, 1.0, "stat_bin").unwrap();
        assert_eq!(n, 1, "Sturges on empty slice must be 1, got {n}");
    }
}
