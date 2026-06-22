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
/// (fixed count), or a float (fixed width). Used by `Bin2D`'s constructor; the
/// 1-D `Bin` keeps its separate `bin_count`/`bin_width` params instead.
pub(crate) fn parse_bin_axis(obj: &Bound<'_, PyAny>) -> PyResult<BinMode> {
    if let Ok(s) = obj.extract::<&str>() {
        return match s {
            "sturges" => Ok(BinMode::Sturges),
            "fd" | "freedman_diaconis" => Ok(BinMode::FreedmanDiaconis),
            _ => Err(PyValueError::new_err(format!(
                "Bin2D: unknown bins value '{s}'; expected 'sturges'|'fd'|int|float"
            ))),
        };
    }
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(BinMode::Fixed { n });
    }
    if let Ok(w) = obj.extract::<f64>() {
        return Ok(BinMode::Width { w });
    }
    Err(PyValueError::new_err(
        "Bin2D: bins must be 'sturges'|'fd'|int|float",
    ))
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
}
