//! Shared scale-construction helpers.
//!
//! Pre-F2b this module hosted a sealed `Scale` enum that dispatched math
//! across every scale variant. As of F2b that enum is gone — each variant's
//! data + math lives in its own per-file module (`linear::LinearScaleData`,
//! `log::LogScaleData`, etc.). What remains here:
//!
//! - `compute_quantile_cuts` — R-7 / numpy-default quantile linear
//!   interpolation, shared between `QuantileScale::new` and any future
//!   binning helper that needs equal-frequency cut points.
//! - `validate_*` functions — argument validators called from each
//!   PyO3 constructor; centralised so the error messages stay uniform.
//! - `degenerate_ratio` — the shared 0/0 guard for a zero-span domain,
//!   used by every affine-continuous scale's `scale()` (GH #104).

use pyo3::exceptions::PyValueError;
use pyo3::{Py, PyAny, PyResult, Python};

use crate::spec::encoding::{encode_serde_value_for_py, ContinuousScaleCommon, ScaleSpec};

/// Build the `ContinuousScaleCommon` payload shared by the seven affine
/// continuous `ScaleSpec` variants (Linear, Log, Time, Symlog, Pow, Sqrt, Utc).
///
/// Centralises the `domain`/`range` "emit only when user-set" guards (mirroring
/// each pyclass's `domain()`/`range()` getter) plus the `scheme`/`domain_param`
/// fields, which are always `None` on a freshly-constructed `*Scale` (those wire
/// keys originate from the dict-form scale path, never from a pyclass instance).
///
/// `reverse` is domain-swap sugar (F-L04-07): it is carried through to the wire
/// unswapped — the actual domain-pair swap happens later, at the resolver's
/// continuous chokepoint (`apply_domain_reverse` in
/// `render::scale_resolve::positional`), not here and not at pyclass
/// construction. `LinearScale::scale`/`invert`/`ticks` and its five siblings
/// therefore never see the swapped domain; only the rendered/resolved scale
/// does.
pub(crate) fn continuous_common(
    domain: [f64; 2],
    domain_user_set: bool,
    range: [f64; 2],
    range_user_set: bool,
    clamp: bool,
    padding: Option<f64>,
    reverse: bool,
) -> ContinuousScaleCommon {
    ContinuousScaleCommon {
        domain: domain_user_set.then(|| domain.to_vec()),
        range: range_user_set.then(|| range.to_vec()),
        clamp,
        padding,
        scheme: None,
        domain_param: None,
        reverse,
    }
}

/// Serialize a canonical `ScaleSpec` to its Python wire dict.
///
/// Each `*Scale` pyclass's `_to_scale_spec_dict` delegates here so the
/// serialization path is single-sourced through the existing
/// `encode_serde_value_for_py` helper (SPEC-04). `to_scale_spec()` always yields
/// a value, so the `None` arm is unreachable in practice; it surfaces as an error
/// rather than a panic to keep the pyclass boundary panic-free.
pub(crate) fn scale_spec_to_py_dict(py: Python<'_>, spec: ScaleSpec) -> PyResult<Py<PyAny>> {
    encode_serde_value_for_py(py, &Some(spec))?
        .ok_or_else(|| PyValueError::new_err("scale spec failed to serialize"))
}

/// The rejection sentence for a degenerate `[lo, hi]` domain whose endpoints
/// coincide.
///
/// One vocabulary, exactly like [`not_strictly_ascending_message`]: every scale
/// constructor that rejects a zero-width domain (`QuantizeScale`,
/// `DivergingScale`, and [`validate_continuous_domain`]'s continuous family)
/// raises this text, and the render-side discretizing color resolver quotes the
/// same sentence when a raw-dict scale reaches it having bypassed them.
pub(crate) const DEGENERATE_DOMAIN_MESSAGE: &str = "domain endpoints must differ (lo != hi)";

/// `true` when `values` is strictly ascending (every element greater than its
/// predecessor). Empty and single-element slices are trivially ascending.
///
/// The shared predicate behind every "boundary list must be sorted" check:
/// `ThresholdScale`'s and `BinOrdinalScale`'s constructors and the render-side
/// discretizing color resolver all ask this, so a list one of them accepts is
/// exactly the set the others accept.
pub(crate) fn is_strictly_ascending(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[0] < w[1])
}

/// The rejection sentence for a boundary list that is not strictly ascending,
/// naming `field` (`"domain"`, `"bins"`, …).
///
/// One vocabulary: the `ThresholdScale`/`BinOrdinalScale` constructors raise
/// this text as a `ValueError`, and the render-side resolver quotes the same
/// sentence when a raw-dict scale reaches it having bypassed those constructors.
/// A user who gets the message from either path reads identical words.
pub(crate) fn not_strictly_ascending_message(field: &str) -> String {
    format!("{field} must be strictly sorted ascending")
}

/// The `n - 1` interior boundaries of `n` equal-width bins spanning
/// `[lo, hi]` — quantize bin geometry, defined once.
///
/// Companion to [`compute_quantile_cuts`]: same shape (`n` bins → `n - 1`
/// interior cuts, empty for `n <= 1`), different rule (equal *width* rather
/// than equal *probability*). Shared by `QuantizeScale`'s own `thresholds()`
/// and by the render-side discretizing color resolver
/// (`render::scale_resolve::color`), so the two cannot drift.
///
/// `lo > hi` is not rejected here: it yields descending boundaries, which is
/// what `QuantizeScale::thresholds()` has always returned for a descending
/// domain. Callers that need ascending output normalize the endpoints first.
pub(crate) fn uniform_bin_thresholds(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return Vec::new();
    }
    let step = (hi - lo) / n as f64;
    (1..n).map(|i| lo + i as f64 * step).collect()
}

/// R-7 / numpy default quantile cut-points: linear interpolation between
/// order statistics. Returns `k-1` cut points dividing the sorted sample
/// into `k` equal-probability bins.
pub(crate) fn compute_quantile_cuts(sorted_sample: &[f64], k: usize) -> Vec<f64> {
    if k <= 1 || sorted_sample.is_empty() { return Vec::new(); }
    let n = sorted_sample.len();
    let mut cuts = Vec::with_capacity(k - 1);
    for i in 1..k {
        let p = (i as f64) / (k as f64);
        let h = p * (n as f64 - 1.0);
        let lo = h.floor() as usize;
        let hi = (h.ceil() as usize).min(n - 1);
        let frac = h - h.floor();
        let v = sorted_sample[lo] * (1.0 - frac) + sorted_sample[hi] * frac;
        cuts.push(v);
    }
    cuts
}

/// Shared degenerate-domain guard for every affine-continuous scale's
/// `t = (numerator) / (denom)` ratio (Linear/Time, Log, Symlog, Pow/Sqrt).
///
/// `denom` is the domain-space span (e.g. `d1 - d0`, or the same after a
/// scale-specific forward transform — `pow_fwd(d1) - pow_fwd(d0)`,
/// `ld1 - ld0`, `symlog_fwd(d1) - symlog_fwd(d0)`); `numerator` is the same
/// transform applied to `(x, d0)` (e.g. `x - d0`, `pow_fwd(x) - pow_fwd(d0)`).
/// This only overrides the ratio when the domain is **actually** degenerate:
/// `denom == 0.0` (`d0 == d1`, e.g. every value in a data column is
/// identical) *and* `numerator == 0.0`, which happens precisely when `x` is
/// the one point inside the collapsed domain (`x == d0 == d1`, so the
/// scale-specific transform of `x` and `d0` are the exact same
/// floating-point computation twice — bit-identical, not merely close).
/// That is `0/0 = NaN`, which then survives both the `clamp` arm
/// (`NaN.clamp(lo, hi)` returns `NaN` unchanged in Rust's stdlib
/// implementation) and the out-of-domain arm (`x == d0 == d1` is never "out
/// of domain", so that branch never fires to rescue it) — a value that
/// should render at a deterministic pixel silently disappears instead.
/// Returning `0.5` (the range midpoint) instead is finite by construction
/// and matches the "center a degenerate domain rather than drop it"
/// convention already used for the auto-inferred-domain path
/// (`render::scale_resolve::domain::numeric_domain_union`'s symmetric-band
/// expansion) — this is the same convention applied at the scale-formula
/// level, so it holds for every construction path, not only the one that
/// pre-expands the domain upstream. GH #104.
///
/// A `denom == 0.0` with a **nonzero** `numerator` is a different case —
/// `x` outside the collapsed domain (`x != d0` while `d0 == d1`) — and this
/// function deliberately does *not* touch it: `k/0 = ±inf` propagates
/// through unchanged, exactly as it did before this guard existed. That
/// keeps the `clamp == true` arm's pre-existing behavior for that input
/// (`(+inf).clamp(lo, hi) == hi`, `(-inf).clamp(lo, hi) == lo` — the
/// range-endpoint result every affine-continuous scale already gave for an
/// out-of-domain value on a degenerate domain), rather than silently
/// widening this guard's scope to override a case it was never meant to
/// touch.
#[inline]
pub(super) fn degenerate_ratio(numerator: f64, denom: f64) -> f64 {
    if denom == 0.0 && numerator == 0.0 { 0.5 } else { numerator / denom }
}

// ---------- validators (used by pyclass facades) ----------

pub(crate) fn validate_finite(name: &str, values: &[f64]) -> PyResult<()> {
    for v in values {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} must contain only finite values; found {v}"
            )));
        }
    }
    Ok(())
}

/// Validate domain non-emptiness, duplicate categories, and padding bounds.
///
/// Called independently of range validation so that string-only color ranges
/// (which have no numeric extent to check) still get domain and padding
/// validated.
pub(crate) fn validate_ordinal_domain(domain: &[String], padding: f64) -> PyResult<()> {
    if domain.is_empty() {
        return Err(PyValueError::new_err("domain must be non-empty"));
    }
    if !padding.is_finite() || !(0.0..=1.0).contains(&padding) {
        return Err(PyValueError::new_err(format!(
            "padding must be in [0, 1]; got {padding}"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for c in domain {
        if !seen.insert(c.as_str()) {
            return Err(PyValueError::new_err(format!(
                "duplicate category in domain: '{c}'"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_ordinal(domain: &[String], range: &[f64], padding: f64) -> PyResult<()> {
    validate_ordinal_domain(domain, padding)?;
    validate_band_point_range(range)?;
    Ok(())
}

/// Validate a Band/Point pixel range: at least 2 entries (extent endpoints)
/// and every entry finite. Mirrors the numeric-range check `validate_ordinal`
/// applies, so `BandScale`/`PointScale` reject the same malformed `range=`
/// OrdinalScale already rejects, instead of each silently substituting its
/// own `[0.0, 1.0]` placeholder as if it were the user's explicit intent
/// (GH #69 sibling-drift fix). Finiteness is checked on EVERY entry (via
/// `validate_finite`), even though a `range` with more than 2 entries is
/// later truncated to its first two by the resolver (see
/// `band_point_pixel_range` in `render::scale_resolve::positional`) — a
/// non-finite value anywhere in the input is user error worth rejecting.
pub(crate) fn validate_band_point_range(range: &[f64]) -> PyResult<()> {
    if range.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "range must have length >= 2 (extent endpoints); got {}",
            range.len()
        )));
    }
    validate_finite("range", range)?;
    Ok(())
}

pub(crate) fn validate_threshold(domain: &[f64], range: &[f64]) -> PyResult<()> {
    if range.is_empty() {
        return Err(PyValueError::new_err("range must be non-empty"));
    }
    if domain.len() + 1 != range.len() {
        return Err(PyValueError::new_err(format!(
            "range length must equal domain length + 1; got domain={}, range={}",
            domain.len(),
            range.len()
        )));
    }
    validate_finite("domain", domain)?;
    validate_finite("range", range)?;
    if !is_strictly_ascending(domain) {
        return Err(PyValueError::new_err(not_strictly_ascending_message("domain")));
    }
    Ok(())
}

pub(crate) fn validate_quantile(domain: &[f64], range: &[f64]) -> PyResult<()> {
    if range.is_empty() {
        return Err(PyValueError::new_err("range must be non-empty"));
    }
    if domain.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "domain (sample) must have length >= 2; got {}",
            domain.len()
        )));
    }
    validate_finite("domain", domain)?;
    validate_finite("range", range)?;
    Ok(())
}

/// Validate a continuous-scale domain pair: exactly 2 entries, both finite,
/// and endpoints that differ (a degenerate `[c, c]` domain divides by zero
/// downstream). Called from [`resolve_continuous`] (GH #69 cohesion fix —
/// `resolve_continuous` used to hand-duplicate this exact check inline).
pub(crate) fn validate_continuous_domain(domain: &[f64]) -> PyResult<()> {
    if domain.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "domain must have length 2; got {}",
            domain.len()
        )));
    }
    validate_finite("domain", domain)?;
    if domain[0] == domain[1] {
        return Err(PyValueError::new_err(DEGENERATE_DOMAIN_MESSAGE));
    }
    Ok(())
}

/// Validate a continuous-scale range pair: exactly 2 entries, both finite.
/// Called from [`resolve_continuous`] (see [`validate_continuous_domain`]'s
/// doc for why this was extracted).
pub(crate) fn validate_continuous_range(range: &[f64]) -> PyResult<()> {
    if range.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "range must have length 2; got {}",
            range.len()
        )));
    }
    validate_finite("range", range)?;
    Ok(())
}

/// Resolved inputs shared by every affine-continuous PyO3 scale constructor
/// (`LinearScale`, `LogScale`, `PowScale`, `SqrtScale`, `SymlogScale`).
///
/// Carries the user-set flags alongside the materialised `[lo, hi]` pairs so
/// each scale's `#[new]` can construct its own data struct (which differs by
/// the extra per-scale field) without re-implementing the shared prelude.
pub(crate) struct ResolvedContinuous {
    pub(crate) domain: [f64; 2],
    pub(crate) range: [f64; 2],
    pub(crate) range_user_set: bool,
    pub(crate) domain_user_set: bool,
}

/// Shared prelude for the affine-continuous `#[new]` constructors.
///
/// Captures `range_user_set`/`domain_user_set`, unwraps `range` to the
/// `[0, 1]` default, and substitutes the per-scale `domain_sentinel` when no
/// domain is supplied (the sentinel is never validated — render-time
/// inference replaces it before any scale computation). Per-scale validation
/// (Log's base/sign checks, Pow's exponent, Symlog's constant) and `nice`
/// remain in each scale because they depend on fields this helper does not
/// know about.
///
/// `domain` and `range` are each validated independently via
/// [`validate_continuous_domain`] / [`validate_continuous_range`], whenever
/// the user actually supplied that argument (GH #69 sibling fix): the
/// previous `if domain_user_set { validate_continuous_pair(...) }` gate (a
/// combined domain+range validator, since removed — `TimeScale::new` was its
/// last production caller before switching to this helper, F-L04-10) skipped
/// ALL range validation whenever `domain` was left unset, so e.g.
/// `LinearScale(range=[5.0])` indexed past the end of a 1-element `Vec` at
/// `range: [r[0], r[1]]` below and **panicked** (`index out of bounds`)
/// instead of raising a typed `ValueError`; a non-finite `range` with no
/// `domain` slipped through silently for the same reason. This helper calls
/// the same per-field validators the old combined validator composed from
/// (rather than hand-duplicating their checks inline), which keeps every
/// call path from drifting on error messages or thresholds.
pub(crate) fn resolve_continuous(
    domain: Option<Vec<f64>>,
    range: Option<Vec<f64>>,
    domain_sentinel: [f64; 2],
) -> PyResult<ResolvedContinuous> {
    let range_user_set = range.is_some();
    let domain_user_set = domain.is_some();

    if let Some(r) = range.as_deref() {
        validate_continuous_range(r)?;
    }
    if let Some(d) = domain.as_deref() {
        validate_continuous_domain(d)?;
    }

    let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
    let dom = domain.unwrap_or_else(|| domain_sentinel.to_vec());
    Ok(ResolvedContinuous {
        domain: [dom[0], dom[1]],
        range: [r[0], r[1]],
        range_user_set,
        domain_user_set,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // `validate_continuous_pair` (the combined domain+range validator these
    // tests used to pin) was deleted as dead production code (F-L04-10
    // remediation): `TimeScale::new` was its last caller before switching to
    // `resolve_continuous`. Its coverage is not lost — `validate_continuous_domain`/
    // `validate_continuous_range` (the two validators it composed, unchanged)
    // are still exercised, wrong-length/degenerate/non-finite included, via
    // `resolve_continuous`'s own test suite immediately below and via every
    // continuous-scale pyclass constructor test that resolves through it.

    // ── resolve_continuous: range validated even when domain is unset (GH #69) ──

    /// A short `range` with `domain` unset used to skip validation entirely
    /// (gated behind `domain_user_set`) and then panic on `range: [r[0],
    /// r[1]]` with an out-of-bounds index. Must now raise a typed error.
    #[test]
    fn test_resolve_continuous_rejects_short_range_with_no_domain() {
        let r = resolve_continuous(None, Some(vec![5.0]), [0.0, 1.0]);
        assert!(r.is_err(), "1-element range with no domain must be rejected, not panic");
    }

    /// A non-finite `range` with `domain` unset used to slip through silently
    /// (same validation gate). Must now raise.
    #[test]
    fn test_resolve_continuous_rejects_non_finite_range_with_no_domain() {
        let r = resolve_continuous(None, Some(vec![0.0, f64::NAN]), [0.0, 1.0]);
        assert!(r.is_err(), "non-finite range with no domain must be rejected");
    }

    /// A short/non-finite `domain` with `range` unset must also be rejected
    /// independently (the domain-side mirror of the two tests above).
    #[test]
    fn test_resolve_continuous_rejects_bad_domain_with_no_range() {
        assert!(resolve_continuous(Some(vec![5.0]), None, [0.0, 1.0]).is_err());
        assert!(resolve_continuous(Some(vec![0.0, f64::INFINITY]), None, [0.0, 1.0]).is_err());
        assert!(resolve_continuous(Some(vec![5.0, 5.0]), None, [0.0, 1.0]).is_err());
    }

    /// Neither argument supplied: both fall back to their defaults/sentinel
    /// untouched (the sentinel itself is never validated).
    #[test]
    fn test_resolve_continuous_defaults_when_neither_set() {
        let resolved = resolve_continuous(None, None, [1.0, 10.0]).unwrap();
        assert_eq!(resolved.domain, [1.0, 10.0]);
        assert_eq!(resolved.range, [0.0, 1.0]);
        assert!(!resolved.domain_user_set);
        assert!(!resolved.range_user_set);
    }

    /// A well-formed `range` with `domain` unset resolves cleanly (the
    /// non-buggy path this fix must not regress).
    #[test]
    fn test_resolve_continuous_accepts_valid_range_with_no_domain() {
        let resolved = resolve_continuous(None, Some(vec![10.0, 200.0]), [0.0, 1.0]).unwrap();
        assert_eq!(resolved.range, [10.0, 200.0]);
        assert_eq!(resolved.domain, [0.0, 1.0]);
        assert!(resolved.range_user_set);
        assert!(!resolved.domain_user_set);
    }

    #[test]
    fn test_validate_ordinal_rejects_empty_domain() {
        let r = validate_ordinal(&[], &[0.0, 10.0], 0.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_ordinal_rejects_duplicates() {
        let r = validate_ordinal(
            &["a".to_string(), "a".to_string()],
            &[0.0, 10.0],
            0.0,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_ordinal_rejects_bad_padding() {
        let r = validate_ordinal(&["a".to_string()], &[0.0, 10.0], 1.5);
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_threshold_rejects_arity_mismatch() {
        let r = validate_threshold(&[0.0, 10.0], &[1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_threshold_rejects_unsorted_domain() {
        let r = validate_threshold(&[10.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_quantile_cuts_known_values() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = compute_quantile_cuts(&sample, 3);
        assert_eq!(cuts.len(), 2);
        assert!((cuts[0] - 7.0/3.0).abs() < 1e-9, "got {}", cuts[0]);
        assert!((cuts[1] - 11.0/3.0).abs() < 1e-9, "got {}", cuts[1]);
    }

    #[test]
    fn test_validate_quantile_rejects_short_domain() {
        assert!(validate_quantile(&[1.0], &[0.0, 1.0]).is_err());
    }
}
