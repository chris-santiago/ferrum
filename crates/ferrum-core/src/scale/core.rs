//! Sealed `Scale` enum that centralises math for every scale variant.
//! Each scale-task in section C/D extends this with a new variant.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use super::ticks::{nice_step, nice_ticks};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Scale {
    Linear { domain: [f64; 2], range: [f64; 2], clamp: bool },
}

impl Scale {
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
        }
    }

    pub(crate) fn ticks(&self, count: Option<usize>) -> Vec<f64> {
        match self {
            Scale::Linear { domain, .. } => {
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
}
