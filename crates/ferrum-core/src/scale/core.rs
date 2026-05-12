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
    Log     { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Symlog  { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
    Ordinal    { domain: Vec<String>, range: Vec<f64>, padding: f64 },
    Threshold  { domain: Vec<f64>, range: Vec<f64> },
    Quantile   { domain: Vec<f64>, range: Vec<f64>, quantiles: Vec<f64> },
}

impl Scale {
    fn ordinal_layout(domain: &[String], range: &[f64], padding: f64) -> (f64, f64, f64) {
        let r_lo = *range.first().unwrap();
        let r_hi = *range.last().unwrap();
        let n = domain.len() as f64;
        let step = (r_hi - r_lo) / n;
        let half_band = step.abs() * (1.0 - padding) / 2.0;
        let first_center = r_lo + step / 2.0;
        (first_center, step, half_band)
    }

    fn symlog_fwd(x: f64, c: f64) -> f64 {
        x.signum() * (x.abs() / c).ln_1p()
    }

    fn symlog_inv(y: f64, c: f64) -> f64 {
        y.signum() * c * (y.abs().exp() - 1.0)
    }

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
            Scale::Log { domain, range, base, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let neg = d0 < 0.0;
                let sign = if neg { -1.0 } else { 1.0 };
                if (x * sign) <= 0.0 && !*clamp { return f64::NAN; }
                let log_base = base.ln();
                let lx = (x * sign).max(f64::MIN_POSITIVE).ln() / log_base;
                let ld0 = (d0 * sign).ln() / log_base;
                let ld1 = (d1 * sign).ln() / log_base;
                let t = (lx - ld0) / (ld1 - ld0);
                let mapped = r0 + t * (r1 - r0);
                if *clamp {
                    let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                    mapped.clamp(lo, hi)
                } else if (x * sign) < (d0 * sign).min(d1 * sign) || (x * sign) > (d0 * sign).max(d1 * sign) {
                    f64::NAN
                } else {
                    mapped
                }
            }
            Scale::Symlog { domain, range, constant, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let f = |v: f64| Self::symlog_fwd(v, *constant);
                let t = (f(x) - f(d0)) / (f(d1) - f(d0));
                let mapped = r0 + t * (r1 - r0);
                if *clamp {
                    let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                    mapped.clamp(lo, hi)
                } else if x < d0.min(d1) || x > d0.max(d1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
            Scale::Ordinal { .. } => f64::NAN,
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

    pub(crate) fn invert_f64(&self, y: f64) -> f64 {
        match self {
            Scale::Log { domain, range, base, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let neg = d0 < 0.0;
                let sign = if neg { -1.0 } else { 1.0 };
                let log_base = base.ln();
                let ld0 = (d0 * sign).ln() / log_base;
                let ld1 = (d1 * sign).ln() / log_base;
                let t = (y - r0) / (r1 - r0);
                let lmapped = ld0 + t * (ld1 - ld0);
                let mapped = sign * base.powf(lmapped);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
            Scale::Symlog { domain, range, constant, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let f = |v: f64| Self::symlog_fwd(v, *constant);
                let t = (y - r0) / (r1 - r0);
                let lmapped = f(d0) + t * (f(d1) - f(d0));
                let mapped = Self::symlog_inv(lmapped, *constant);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
            Scale::Ordinal { .. } => f64::NAN,
            Scale::Threshold { .. } => f64::NAN,
            Scale::Quantile { .. } => f64::NAN,
        }
    }

    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        match self {
            Scale::Log { domain, base, .. } => {
                let n = count.unwrap_or(10);
                let neg = domain[0] < 0.0;
                let sign: f64 = if neg { -1.0 } else { 1.0 };
                let lo = (domain[0] * sign).min(domain[1] * sign);
                let hi = (domain[0] * sign).max(domain[1] * sign);
                let log_base = base.ln();
                let lo_exp = (lo.ln() / log_base).floor() as i64;
                let hi_exp = (hi.ln() / log_base).ceil() as i64;
                let span_decades = (hi_exp - lo_exp).max(1) as usize;
                if span_decades >= n {
                    let mut out: Vec<f64> = (lo_exp..=hi_exp)
                        .map(|e| sign * base.powi(e as i32))
                        .filter(|t| (t.abs() >= lo) && (t.abs() <= hi))
                        .collect();
                    if domain[0] > domain[1] { out.reverse(); }
                    out
                } else {
                    let lvals = nice_ticks(lo.ln() / log_base, hi.ln() / log_base, n);
                    let mut out: Vec<f64> = lvals.into_iter().map(|lv| sign * base.powf(lv)).collect();
                    if domain[0] > domain[1] { out.reverse(); }
                    out
                }
            }
            Scale::Symlog { domain, .. } => {
                nice_ticks(domain[0], domain[1], count.unwrap_or(10))
            }
            Scale::Ordinal { range, .. } => range.clone(),
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
            Scale::Log { domain, range, base, clamp } => {
                let neg = domain[0] < 0.0;
                let sign: f64 = if neg { -1.0 } else { 1.0 };
                let log_base = base.ln();
                let lo = (domain[0] * sign).min(domain[1] * sign);
                let hi = (domain[0] * sign).max(domain[1] * sign);
                let lo_exp = (lo.ln() / log_base).floor();
                let hi_exp = (hi.ln() / log_base).ceil();
                let new_lo = sign * base.powf(lo_exp);
                let new_hi = sign * base.powf(hi_exp);
                let new_domain = if domain[0] <= domain[1] {
                    [new_lo, new_hi]
                } else {
                    [new_hi, new_lo]
                };
                Scale::Log { domain: new_domain, range, base, clamp }
            }
            Scale::Symlog { domain, range, constant, clamp } => {
                let step = nice_step(domain[0], domain[1], 10);
                if !step.is_finite() || step == 0.0 {
                    return Scale::Symlog { domain, range, constant, clamp };
                }
                let lo_min = domain[0].min(domain[1]);
                let hi_max = domain[0].max(domain[1]);
                let nice_lo = (lo_min / step).floor() * step;
                let nice_hi = (hi_max / step).ceil() * step;
                let new_domain = if domain[0] <= domain[1] {
                    [nice_lo, nice_hi]
                } else {
                    [nice_hi, nice_lo]
                };
                Scale::Symlog { domain: new_domain, range, constant, clamp }
            }
            Scale::Ordinal { domain, range, padding } => {
                Scale::Ordinal { domain, range, padding }
            }
            Scale::Threshold { domain, range } => Scale::Threshold { domain, range },
            Scale::Quantile { domain, range, quantiles } => {
                Scale::Quantile { domain, range, quantiles }
            }
        }
    }

    pub(crate) fn scale_str(&self, s: &str) -> f64 {
        match self {
            Scale::Ordinal { domain, range, padding } => {
                let idx = match domain.iter().position(|c| c == s) {
                    Some(i) => i,
                    None => return f64::NAN,
                };
                let (first_center, step, _half_band) = Self::ordinal_layout(domain, range, *padding);
                first_center + (idx as f64) * step
            }
            _ => f64::NAN,
        }
    }

    pub(crate) fn invert_band(&self, y: f64) -> Option<String> {
        match self {
            Scale::Ordinal { domain, range, padding } => {
                if y.is_nan() { return None; }
                let (first_center, step, half_band) = Self::ordinal_layout(domain, range, *padding);
                if step == 0.0 { return None; }
                let raw = (y - first_center) / step;
                let idx = raw.round() as i64;
                if idx < 0 || idx as usize >= domain.len() { return None; }
                let center = first_center + (idx as f64) * step;
                if (y - center).abs() <= half_band {
                    Some(domain[idx as usize].clone())
                } else {
                    None
                }
            }
            _ => None,
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
    fn test_log_scale_basic_decades() {
        let s = Scale::Log { domain: [1.0, 1000.0], range: [0.0, 3.0], base: 10.0, clamp: false };
        assert!((s.scale_f64(1.0) - 0.0).abs() < 1e-12);
        assert!((s.scale_f64(10.0) - 1.0).abs() < 1e-12);
        assert!((s.scale_f64(1000.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_log_inversion_round_trip() {
        let s = Scale::Log { domain: [1.0, 1_000_000.0], range: [0.0, 6.0], base: 10.0, clamp: false };
        for x in [1.0, 10.0, 100.0, 12345.0, 999999.0] {
            let y = s.scale_f64(x);
            let back = s.invert_f64(y);
            assert!((back / x - 1.0).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn test_log_negative_domain_supported() {
        let s = Scale::Log { domain: [-1000.0, -1.0], range: [0.0, 3.0], base: 10.0, clamp: false };
        let y = s.scale_f64(-10.0);
        let back = s.invert_f64(y);
        assert!((back / -10.0 - 1.0).abs() < 1e-9, "negative round-trip failed: got {back}");
    }

    #[test]
    fn test_log_ticks_one_per_decade() {
        let s = Scale::Log { domain: [1.0, 1000.0], range: [0.0, 3.0], base: 10.0, clamp: false };
        let t = s.ticks(Some(4));
        assert!(t.len() >= 3, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn test_log_nice_rounds_to_decades() {
        let s = Scale::Log { domain: [3.0, 700.0], range: [0.0, 1.0], base: 10.0, clamp: false };
        let n = s.nice();
        match n {
            Scale::Log { domain, .. } => {
                assert!((domain[0] - 1.0).abs() < 1e-9);
                assert!((domain[1] - 1000.0).abs() < 1e-9);
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_symlog_scale_handles_zero() {
        let s = Scale::Symlog { domain: [-100.0, 100.0], range: [0.0, 1.0], constant: 1.0, clamp: false };
        let y = s.scale_f64(0.0);
        assert!(y.is_finite(), "scale(0) returned {y}");
        assert!((y - 0.5).abs() < 1e-12, "expected 0.5, got {y}");
    }

    #[test]
    fn test_symlog_inversion_round_trip_across_zero() {
        let s = Scale::Symlog { domain: [-1000.0, 1000.0], range: [0.0, 1.0], constant: 1.0, clamp: false };
        for x in [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0] {
            let y = s.scale_f64(x);
            let back = s.invert_f64(y);
            assert!((back - x).abs() < 1e-6, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn test_symlog_constant_changes_curvature() {
        // Larger constant → more linear (less compression) → x=50 on [-100,100] maps closer to 0.5
        // Smaller constant → more log-like → x=50 maps closer to 1.0 (compressed toward endpoint)
        let s1 = Scale::Symlog { domain: [-100.0, 100.0], range: [0.0, 1.0], constant: 1.0,   clamp: false };
        let s2 = Scale::Symlog { domain: [-100.0, 100.0], range: [0.0, 1.0], constant: 100.0, clamp: false };
        let y1 = s1.scale_f64(50.0);
        let y2 = s2.scale_f64(50.0);
        assert!(y1 > y2, "expected y1={y1} > y2={y2}: smaller constant compresses more");
        // Both must be in (0.5, 1.0) since 50 is in the positive half of [-100, 100]
        assert!(y1 > 0.5 && y1 < 1.0, "y1={y1} out of (0.5,1.0)");
        assert!(y2 > 0.5 && y2 < 1.0, "y2={y2} out of (0.5,1.0)");
    }

    #[test]
    fn test_ordinal_band_centers_no_padding() {
        let s = Scale::Ordinal {
            domain: vec!["a".into(), "b".into(), "c".into()],
            range: vec![0.0, 30.0],
            padding: 0.0,
        };
        assert!((s.scale_str("a") - 5.0).abs() < 1e-12);
        assert!((s.scale_str("b") - 15.0).abs() < 1e-12);
        assert!((s.scale_str("c") - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_ordinal_invert_round_trip() {
        let s = Scale::Ordinal {
            domain: vec!["a".into(), "b".into(), "c".into()],
            range: vec![0.0, 30.0],
            padding: 0.0,
        };
        for cat in ["a", "b", "c"] {
            let y = s.scale_str(cat);
            let back = s.invert_band(y);
            assert_eq!(back.as_deref(), Some(cat), "round-trip failed for {cat}");
        }
    }

    #[test]
    fn test_ordinal_invert_outside_band_returns_none() {
        let s = Scale::Ordinal {
            domain: vec!["a".into(), "b".into(), "c".into()],
            range: vec![0.0, 30.0],
            padding: 0.5,
        };
        assert!(s.invert_band(10.0).is_none());
    }

    #[test]
    fn test_ordinal_unknown_category_returns_nan() {
        let s = Scale::Ordinal {
            domain: vec!["a".into()],
            range: vec![0.0, 10.0],
            padding: 0.0,
        };
        assert!(s.scale_str("z").is_nan());
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
