//! Sealed `Scale` enum that centralises math for every scale variant.
//! Each scale-task in section C/D extends this with a new variant.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use super::ticks::{nice_step, nice_ticks};

/// Legacy sealed enum that previously centralised math for every scale
/// variant. As of F2b, each variant's math has migrated into its own
/// per-file module (see `linear::LinearScaleData`, etc.). Variants are
/// being removed one at a time; this enum will be deleted entirely once
/// the last variant has migrated.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Scale {
    Threshold  { domain: Vec<f64>, range: Vec<f64> },
    Quantile   { domain: Vec<f64>, range: Vec<f64>, quantiles: Vec<f64> },
}

impl Scale {
    pub(crate) fn compute_quantile_cuts(sorted_sample: &[f64], k: usize) -> Vec<f64> {
        // R-7 / numpy default: linear interpolation between order statistics.
        // Returns k-1 cut points dividing the sample into k bins.
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

    pub(crate) fn scale_f64(&self, x: f64) -> f64 {
        match self {
            Scale::Threshold { domain, range } => {
                if x.is_nan() { return f64::NAN; }
                let idx = domain.partition_point(|t| *t <= x);
                range[idx]
            }
            Scale::Quantile { range, quantiles, .. } => {
                if x.is_nan() { return f64::NAN; }
                let idx = quantiles.partition_point(|q| *q <= x);
                range[idx]
            }
        }
    }

    pub(crate) fn invert_f64(&self, _y: f64) -> f64 {
        // None of the remaining variants (Ordinal/Threshold/Quantile)
        // support continuous inversion — Ordinal uses invert_band,
        // Threshold/Quantile use invert_extent.
        f64::NAN
    }

    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        match self {
            Scale::Threshold { domain, .. } => domain.clone(),
            Scale::Quantile { domain, quantiles, .. } => {
                let target = count.unwrap_or_else(|| crate::scale::ticks::sturges_floor(domain.len()));
                if target >= quantiles.len() {
                    quantiles.clone()
                } else {
                    let step = quantiles.len() as f64 / target as f64;
                    (0..target)
                        .map(|i| quantiles[((i as f64 + 0.5) * step).floor() as usize])
                        .collect()
                }
            }
        }
    }

    pub(crate) fn nice(self) -> Self {
        match self {
            Scale::Threshold { domain, range } => Scale::Threshold { domain, range },
            Scale::Quantile { domain, range, quantiles } => {
                Scale::Quantile { domain, range, quantiles }
            }
        }
    }

    pub(crate) fn invert_extent(&self, y: f64) -> (f64, f64) {
        match self {
            Scale::Threshold { domain, range } => {
                if y.is_nan() { return (f64::NAN, f64::NAN); }
                let idx = match range.iter().position(|r| *r == y) {
                    Some(i) => i,
                    None => return (f64::NAN, f64::NAN),
                };
                let lo = if idx == 0 { f64::NEG_INFINITY } else { domain[idx - 1] };
                let hi = if idx >= domain.len() { f64::INFINITY } else { domain[idx] };
                (lo, hi)
            }
            Scale::Quantile { range, quantiles, .. } => {
                if y.is_nan() { return (f64::NAN, f64::NAN); }
                let idx = match range.iter().position(|r| *r == y) {
                    Some(i) => i,
                    None => return (f64::NAN, f64::NAN),
                };
                let lo = if idx == 0 { f64::NEG_INFINITY } else { quantiles[idx - 1] };
                let hi = if idx >= quantiles.len() { f64::INFINITY } else { quantiles[idx] };
                (lo, hi)
            }
            _ => (f64::NAN, f64::NAN),
        }
    }
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

pub(crate) fn validate_ordinal(domain: &[String], range: &[f64], padding: f64) -> PyResult<()> {
    if domain.is_empty() {
        return Err(PyValueError::new_err("domain must be non-empty"));
    }
    if range.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "range must have length >= 2 (extent endpoints); got {}",
            range.len()
        )));
    }
    validate_finite("range", range)?;
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
    fn test_threshold_scale_basic() {
        let s = Scale::Threshold {
            domain: vec![0.0, 10.0],
            range: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(s.scale_f64(-1.0), 1.0);
        assert_eq!(s.scale_f64(0.0), 2.0);   // partition_point with <= places 0.0 into bin 1
        assert_eq!(s.scale_f64(5.0), 2.0);
        assert_eq!(s.scale_f64(10.0), 3.0);
        assert_eq!(s.scale_f64(20.0), 3.0);
    }

    #[test]
    fn test_threshold_invert_extent_round_trip() {
        let s = Scale::Threshold {
            domain: vec![0.0, 10.0],
            range: vec![1.0, 2.0, 3.0],
        };
        let (lo, hi) = s.invert_extent(2.0);
        assert_eq!((lo, hi), (0.0, 10.0));
        let (lo, hi) = s.invert_extent(1.0);
        assert!(lo.is_infinite() && lo.is_sign_negative());
        assert_eq!(hi, 0.0);
        let (lo, hi) = s.invert_extent(3.0);
        assert_eq!(lo, 10.0);
        assert!(hi.is_infinite() && hi.is_sign_positive());
    }

    #[test]
    fn test_threshold_invert_extent_unknown_returns_nan() {
        let s = Scale::Threshold {
            domain: vec![0.0],
            range: vec![1.0, 2.0],
        };
        let (lo, hi) = s.invert_extent(99.0);
        assert!(lo.is_nan() && hi.is_nan());
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
        let cuts = Scale::compute_quantile_cuts(&sample, 3);
        assert_eq!(cuts.len(), 2);
        assert!((cuts[0] - 7.0/3.0).abs() < 1e-9, "got {}", cuts[0]);
        assert!((cuts[1] - 11.0/3.0).abs() < 1e-9, "got {}", cuts[1]);
    }

    #[test]
    fn test_quantile_scale_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sorted, 3);
        let s = Scale::Quantile {
            domain: sorted.clone(),
            range: vec![10.0, 20.0, 30.0],
            quantiles: cuts,
        };
        assert_eq!(s.scale_f64(0.0), 10.0);
        assert_eq!(s.scale_f64(2.5), 20.0);
        assert_eq!(s.scale_f64(10.0), 30.0);
    }

    #[test]
    fn test_quantile_invert_extent_round_trip() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sorted, 3);
        let s = Scale::Quantile {
            domain: sorted,
            range: vec![10.0, 20.0, 30.0],
            quantiles: cuts.clone(),
        };
        let (lo, hi) = s.invert_extent(20.0);
        assert!((lo - cuts[0]).abs() < 1e-9);
        assert!((hi - cuts[1]).abs() < 1e-9);
    }

    #[test]
    fn test_quantile_ticks_default_uses_sturges_floor() {
        // domain length = 5, sturges_floor(5) = ceil(log2(5)+1) = ceil(3.32) = 4
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cuts = Scale::compute_quantile_cuts(&sorted, 10); // 9 cuts
        let s = Scale::Quantile {
            domain: sorted,
            range: vec![0.0; 10],
            quantiles: cuts,
        };
        let t = s.ticks(None);
        assert_eq!(t.len(), 4, "expected 4 ticks, got {}: {t:?}", t.len());
    }

    #[test]
    fn test_validate_quantile_rejects_short_domain() {
        assert!(validate_quantile(&[1.0], &[0.0, 1.0]).is_err());
    }
}
