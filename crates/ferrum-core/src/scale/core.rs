//! Sealed `Scale` enum that centralises math for every scale variant.
//! Each scale-task in section C/D extends this with a new variant.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use super::ticks::{nice_step, nice_ticks};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Scale {
    Linear { domain: [f64; 2], range: [f64; 2], clamp: bool },
    Log    { domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool },
    Symlog { domain: [f64; 2], range: [f64; 2], constant: f64, clamp: bool },
}

impl Scale {
    fn symlog_fwd(x: f64, c: f64) -> f64 {
        x.signum() * (x.abs() / c).ln_1p()
    }

    fn symlog_inv(y: f64, c: f64) -> f64 {
        y.signum() * c * (y.abs().exp() - 1.0)
    }

    pub(crate) fn scale_f64(&self, x: f64) -> f64 {
        match self {
            Scale::Linear { domain, range, clamp } => {
                if x.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let t = (x - d0) / (d1 - d0);
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
        }
    }

    pub(crate) fn invert_f64(&self, y: f64) -> f64 {
        match self {
            Scale::Linear { domain, range, clamp } => {
                if y.is_nan() { return f64::NAN; }
                let [d0, d1] = *domain;
                let [r0, r1] = *range;
                let t = (y - r0) / (r1 - r0);
                let mapped = d0 + t * (d1 - d0);
                if *clamp {
                    let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                    mapped.clamp(lo, hi)
                } else if y < r0.min(r1) || y > r0.max(r1) {
                    f64::NAN
                } else {
                    mapped
                }
            }
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
        }
    }

    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        match self {
            Scale::Linear { domain, .. } => {
                nice_ticks(domain[0], domain[1], count.unwrap_or(10))
            }
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
        }
    }

    pub(crate) fn nice(self) -> Self {
        match self {
            Scale::Linear { domain, range, clamp } => {
                let step = nice_step(domain[0], domain[1], 10);
                if !step.is_finite() || step == 0.0 {
                    return Scale::Linear { domain, range, clamp };
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
                Scale::Linear { domain: new_domain, range, clamp }
            }
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
    fn test_linear_scale_basic() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        assert!((s.scale_f64(5.0) - 0.5).abs() < 1e-12);
        assert!((s.scale_f64(0.0) - 0.0).abs() < 1e-12);
        assert!((s.scale_f64(10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_inversion_round_trip() {
        let s = Scale::Linear { domain: [-50.0, 50.0], range: [0.0, 100.0], clamp: false };
        for x in [-50.0, -25.0, 0.0, 17.5, 50.0] {
            let y = s.scale_f64(x);
            let back = s.invert_f64(y);
            assert!((back - x).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn test_linear_out_of_domain_returns_nan_when_unclamped() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        assert!(s.scale_f64(-1.0).is_nan());
        assert!(s.scale_f64(11.0).is_nan());
    }

    #[test]
    fn test_linear_clamp_clamps_output() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: true };
        assert_eq!(s.scale_f64(-1.0), 0.0);
        assert_eq!(s.scale_f64(11.0), 1.0);
    }

    #[test]
    fn test_linear_nan_propagates() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        assert!(s.scale_f64(f64::NAN).is_nan());
        assert!(s.invert_f64(f64::NAN).is_nan());
    }

    #[test]
    fn test_linear_ticks_default_count() {
        let s = Scale::Linear { domain: [0.0, 10.0], range: [0.0, 1.0], clamp: false };
        let t = s.ticks(None);
        assert!(t.len() >= 5, "got {} ticks: {t:?}", t.len());
    }

    #[test]
    fn test_linear_nice_idempotent() {
        let s = Scale::Linear { domain: [0.13, 9.7], range: [0.0, 1.0], clamp: false };
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1, n2);
    }

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
}
