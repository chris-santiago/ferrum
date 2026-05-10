use pyo3::prelude::*;

use super::core::{validate_continuous_pair, Scale};
use super::ticks::nice_time_interval_ms;

#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct TimeScale(Scale);

impl TimeScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `TimeScale` wraps `Scale::Linear` (epoch-ms in domain), matching `TimeScale::new`.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let inner = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale(inner);
        if nice { s.time_nice() } else { s }
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale_f64(x)
    }

    /// Crate-internal tick call (uses time-aware nice intervals).
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.time_ticks(count)
    }

    /// Pixel-range pair `[lo, hi]` of the underlying scale.
    pub(crate) fn range_pair(&self) -> [f64; 2] {
        match &self.0 {
            Scale::Linear { range, .. } => *range,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub(crate) fn repr_string(&self) -> String {
        match &self.0 {
            Scale::Linear { domain, range, clamp } => format!(
                "TimeScale(domain=[{}, {}], range=[{}, {}], clamp={})",
                domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn time_ticks(&self, count: usize) -> Vec<f64> {
        let (d0, d1) = match &self.0 {
            Scale::Linear { domain, .. } => (domain[0], domain[1]),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let span = hi - lo;
        let interval = nice_time_interval_ms(span, count);
        if !interval.is_finite() || interval <= 0.0 {
            return Vec::new();
        }
        let start = (lo / interval).ceil() * interval;
        let end = (hi / interval).floor() * interval;
        let n_steps = ((end - start) / interval).round() as i64;
        if n_steps < 0 {
            return Vec::new();
        }
        let n = (n_steps + 1) as usize;
        let mut out: Vec<f64> = (0..n).map(|i| start + (i as f64) * interval).collect();
        if d0 > d1 {
            out.reverse();
        }
        out
    }

    fn time_nice(&self) -> Self {
        let (d0, d1, range, clamp) = match &self.0 {
            Scale::Linear { domain, range, clamp } => (domain[0], domain[1], *range, *clamp),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let interval = nice_time_interval_ms(hi - lo, 10);
        if !interval.is_finite() || interval <= 0.0 {
            return self.clone();
        }
        let new_lo = (lo / interval).floor() * interval;
        let new_hi = (hi / interval).ceil() * interval;
        let new_domain = if d0 <= d1 { [new_lo, new_hi] } else { [new_hi, new_lo] };
        TimeScale(Scale::Linear { domain: new_domain, range, clamp })
    }
}

#[pymethods]
impl TimeScale {
    #[new]
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false))]
    fn new(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let inner = Scale::Linear {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale(inner);
        if nice {
            Ok(s.time_nice())
        } else {
            Ok(s)
        }
    }

    fn scale(&self, x: f64) -> f64 { self.0.scale_f64(x) }
    fn invert(&self, y: f64) -> f64 { self.0.invert_f64(y) }

    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.time_ticks(count) }

    fn nice(&self) -> Self { self.time_nice() }

    #[getter]
    fn domain(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { domain, .. } => domain.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn range(&self) -> Vec<f64> {
        match &self.0 {
            Scale::Linear { range, .. } => range.to_vec(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    #[getter]
    fn clamp(&self) -> bool {
        match &self.0 {
            Scale::Linear { clamp, .. } => *clamp,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_scale_round_trip_ms() {
        // 2026-01-01 00:00:00 UTC = 1767225600000.0 ms
        // 2026-12-31 23:59:59 UTC ≈ 1798761599000.0 ms
        let t = TimeScale::new(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            vec![0.0, 1000.0],
            false,
            false,
        ).unwrap();
        let mid = (1_767_225_600_000.0 + 1_798_761_599_000.0) / 2.0;
        let y = t.scale(mid);
        let back = t.invert(y);
        assert!((back - mid).abs() < 1e-3, "round-trip failed: got {back}");
    }

    #[test]
    fn test_time_ticks_returns_some_ticks_for_year_span() {
        let t = TimeScale::new(
            vec![1_767_225_600_000.0, 1_798_761_599_000.0],
            vec![0.0, 1000.0],
            false,
            false,
        ).unwrap();
        let ticks = t.ticks(10);
        assert!(!ticks.is_empty(), "expected non-empty ticks");
    }
}
