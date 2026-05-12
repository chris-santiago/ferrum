use pyo3::prelude::*;

use super::core::validate_continuous_pair;
use super::linear::LinearScaleData;
use super::ticks::nice_time_interval_ms;

/// Continuous temporal scale backed by Unix epoch milliseconds.
///
/// Maps an epoch-millisecond domain to a numeric range. Tick generation
/// uses time-aware "nice" intervals (seconds, minutes, hours, days, months,
/// years) rather than purely numeric rounding. Domain values are
/// floating-point epoch milliseconds (UTC).
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[t_min, t_max]`` in epoch milliseconds (UTC).
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Extend domain endpoints to the nearest calendar interval boundary.
///
/// Examples
/// --------
/// Ferrum converts datetime columns automatically; a ``TimeScale`` is
/// constructed implicitly when the channel data type is temporal::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(x=fr.X("date:T"))
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct TimeScale(LinearScaleData, Option<f64>);

impl TimeScale {
    /// Crate-internal constructor (no PyO3, no validation), for render-side use.
    /// `TimeScale` reuses [`LinearScaleData`]: the domain-to-range mapping is
    /// affine over epoch milliseconds, only tick/nice behavior is time-aware.
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, clamp: bool, nice: bool) -> Self {
        let inner = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale(inner, None);
        if nice { s.time_nice() } else { s }
    }

    /// Crate-internal scale call (no PyO3 boundary).
    pub(crate) fn scale_internal(&self, x: f64) -> f64 {
        self.0.scale(x)
    }

    /// Crate-internal tick call (uses time-aware nice intervals).
    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> {
        self.time_ticks(count)
    }

    /// Pixel-range pair `[lo, hi]` of the underlying scale.
    pub(crate) fn range_pair(&self) -> [f64; 2] {
        self.0.range
    }

    pub(crate) fn repr_string(&self) -> String {
        let LinearScaleData { domain, range, clamp } = &self.0;
        format!(
            "TimeScale(domain=[{}, {}], range=[{}, {}], clamp={})",
            domain[0], domain[1], range[0], range[1], if *clamp { "True" } else { "False" }
        )
    }

    fn time_ticks(&self, count: usize) -> Vec<f64> {
        let [d0, d1] = self.0.domain;
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
        let [d0, d1] = self.0.domain;
        let lo = d0.min(d1);
        let hi = d0.max(d1);
        let interval = nice_time_interval_ms(hi - lo, 10);
        if !interval.is_finite() || interval <= 0.0 {
            return self.clone();
        }
        let new_lo = (lo / interval).floor() * interval;
        let new_hi = (hi / interval).ceil() * interval;
        let new_domain = if d0 <= d1 { [new_lo, new_hi] } else { [new_hi, new_lo] };
        TimeScale(
            LinearScaleData { domain: new_domain, range: self.0.range, clamp: self.0.clamp },
            self.1,
        )
    }
}

#[pymethods]
impl TimeScale {
    #[new]
    #[pyo3(signature = (*, domain, range, clamp = false, nice = false, padding = None))]
    fn new(
        domain: Vec<f64>,
        range: Vec<f64>,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
    ) -> PyResult<Self> {
        validate_continuous_pair(&domain, &range)?;
        let inner = LinearScaleData {
            domain: [domain[0], domain[1]],
            range:  [range[0],  range[1]],
            clamp,
        };
        let s = TimeScale(inner, padding);
        if nice {
            Ok(s.time_nice())
        } else {
            Ok(s)
        }
    }

    /// Map an epoch-millisecond value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.0.scale(x) }
    /// Invert a range coordinate ``y`` back to an epoch-millisecond value.
    fn invert(&self, y: f64) -> f64 { self.0.invert(y) }

    /// Return approximately ``count`` time-aligned tick values within the domain.
    ///
    /// Tick granularity snaps to calendar intervals (seconds, minutes, hours,
    /// days, months, or years) based on the domain span.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.time_ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest calendar interval.
    fn nice(&self) -> Self { self.time_nice() }

    /// Input domain as ``[t_min, t_max]`` in epoch milliseconds.
    #[getter]
    fn domain(&self) -> Vec<f64> { self.0.domain.to_vec() }

    /// Output range as ``[lo, hi]`` pixel coordinates.
    #[getter]
    fn range(&self) -> Vec<f64> { self.0.range.to_vec() }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.0.clamp }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.1 }

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
            None,
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
            None,
        ).unwrap();
        let ticks = t.ticks(10);
        assert!(!ticks.is_empty(), "expected non-empty ticks");
    }
}
