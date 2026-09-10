//! Shared bin-count selection mode for the 1-D `Bin` and 2-D `Bin2D` transforms.
//!
//! `BinMode` makes the "which rule fires" decision a typed enum — illegal states
//! (e.g. both a fixed count and a fixed width) unrepresentable — and is reused by
//! both binning transforms with one shared bin-edge resolver. `Bin2D` constructs
//! it directly from its `bins_x`/`bins_y` Python args via [`parse_bin_axis`]; the
//! 1-D `Bin` keeps its flat `bin_count`/`bin_width` constructor params and resolves
//! them to a `BinMode` at the boundary (see `bin.rs`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::scale::ticks::sturges_floor;

/// How a binning transform chooses its bin count.
///
/// Serde shape is a tagged enum (`{"kind":"sturges"}`, `{"kind":"fixed","n":10}`,
/// `{"kind":"width","w":2.0}`, `{"kind":"freedman_diaconis"}`) — this is the wire
/// `Bin2D` already emits via its `bins_x`/`bins_y` fields. The 1-D `Bin` does NOT
/// serialize `BinMode` directly; it projects to/from flat `bin_count`/`bin_width`
/// keys via a serde shim (see `bin.rs::BinWire`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BinMode {
    Sturges,
    FreedmanDiaconis,
    Fixed { n: usize },
    Width { w: f64 },
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
/// (fixed count), or a float (fixed width). Used by `Bin2D`'s constructor
/// (`ctx = "Bin2D"`) and `Kde`'s `count_scale` (`ctx = "stat_kde"`, batch F);
/// the 1-D `Bin` keeps its separate `bin_count`/`bin_width` params instead.
/// `ctx` prefixes both error messages so each caller's error names the
/// transform the user actually invoked, mirroring `resolve_bin_count`'s own
/// `ctx` parameter.
pub(crate) fn parse_bin_axis(obj: &Bound<'_, PyAny>, ctx: &str) -> PyResult<BinMode> {
    if let Ok(s) = obj.extract::<&str>() {
        return match s {
            "sturges" => Ok(BinMode::Sturges),
            "fd" | "freedman_diaconis" => Ok(BinMode::FreedmanDiaconis),
            _ => Err(PyValueError::new_err(format!(
                "{ctx}: unknown bins value '{s}'; expected 'sturges'|'fd'|int|float"
            ))),
        };
    }
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(BinMode::Fixed { n });
    }
    if let Ok(w) = obj.extract::<f64>() {
        return Ok(BinMode::Width { w });
    }
    Err(PyValueError::new_err(format!(
        "{ctx}: bins must be 'sturges'|'fd'|int|float",
    )))
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
