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
pub(crate) fn continuous_common(
    domain: [f64; 2],
    domain_user_set: bool,
    range: [f64; 2],
    range_user_set: bool,
    clamp: bool,
    padding: Option<f64>,
) -> ContinuousScaleCommon {
    ContinuousScaleCommon {
        domain: domain_user_set.then(|| domain.to_vec()),
        range: range_user_set.then(|| range.to_vec()),
        clamp,
        padding,
        scheme: None,
        domain_param: None,
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
    for w in domain.windows(2) {
        if w[0] >= w[1] {
            return Err(PyValueError::new_err(
                "domain must be strictly sorted ascending",
            ));
        }
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

pub(crate) fn validate_continuous_pair(domain: &[f64], range: &[f64]) -> PyResult<()> {
    if domain.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "domain must have length 2; got {}",
            domain.len()
        )));
    }
    if range.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "range must have length 2; got {}",
            range.len()
        )));
    }
    validate_finite("domain", domain)?;
    validate_finite("range", range)?;
    if domain[0] == domain[1] {
        return Err(PyValueError::new_err(
            "domain endpoints must differ (lo != hi)",
        ));
    }
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
/// `[0, 1]` default, substitutes the per-scale `domain_sentinel` when no
/// domain is supplied, and runs `validate_continuous_pair` only when the user
/// set a domain (the sentinel is never validated — render-time inference
/// replaces it before any scale computation). Per-scale validation (Log's
/// base/sign checks, Pow's exponent, Symlog's constant) and `nice` remain in
/// each scale because they depend on fields this helper does not know about.
pub(crate) fn resolve_continuous(
    domain: Option<Vec<f64>>,
    range: Option<Vec<f64>>,
    domain_sentinel: [f64; 2],
) -> PyResult<ResolvedContinuous> {
    let range_user_set = range.is_some();
    let domain_user_set = domain.is_some();
    let r = range.unwrap_or_else(|| vec![0.0, 1.0]);
    let dom = domain.unwrap_or_else(|| domain_sentinel.to_vec());
    if domain_user_set {
        validate_continuous_pair(&dom, &r)?;
    }
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

    #[test]
    fn test_validate_continuous_pair_rejects_wrong_length() {
        assert!(validate_continuous_pair(&[0.0], &[0.0, 1.0]).is_err());
        assert!(validate_continuous_pair(&[0.0, 1.0], &[]).is_err());
    }

    #[test]
    fn test_validate_continuous_pair_rejects_degenerate_domain() {
        assert!(validate_continuous_pair(&[5.0, 5.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn test_validate_continuous_pair_rejects_non_finite() {
        assert!(validate_continuous_pair(&[0.0, f64::NAN], &[0.0, 1.0]).is_err());
        assert!(validate_continuous_pair(&[0.0, 10.0], &[f64::INFINITY, 1.0]).is_err());
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
