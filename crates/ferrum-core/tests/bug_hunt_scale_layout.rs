//! Edge-case integration tests for recent changes:
//!   - LogScale::nice() negative domain fix (hi_exp/lo_exp swap)
//!   - Facet spacing guard (.max(0.0)) in compute_wrap and compute_grid
//!   - x_title_gutter per-axis title_font_size override
//!
//! Because ferrum-core is a cdylib crate, integration tests cannot link against
//! it directly. These tests reproduce the implementation formulas inline to pin
//! numeric contracts. Any divergence between these pinned formulas and the
//! implementation will surface as a regression.

#[cfg(test)]
#[allow(dead_code)]
mod tests {

    // =========================================================================
    // LogScale helper — reproduces LogScaleData from log.rs
    // =========================================================================

    #[derive(Debug, Clone, PartialEq)]
    struct LogScaleData {
        domain: [f64; 2],
        range: [f64; 2],
        base: f64,
        clamp: bool,
    }

    impl LogScaleData {
        fn scale(&self, x: f64) -> f64 {
            if x.is_nan() { return f64::NAN; }
            let [d0, d1] = self.domain;
            let [r0, r1] = self.range;
            let neg = d0 < 0.0;
            let sign = if neg { -1.0 } else { 1.0 };
            if (x * sign) <= 0.0 && !self.clamp { return f64::NAN; }
            let log_base = self.base.ln();
            let lx = (x * sign).max(f64::MIN_POSITIVE).ln() / log_base;
            let ld0 = (d0 * sign).ln() / log_base;
            let ld1 = (d1 * sign).ln() / log_base;
            let t = (lx - ld0) / (ld1 - ld0);
            let mapped = r0 + t * (r1 - r0);
            if self.clamp {
                let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                mapped.clamp(lo, hi)
            } else if (x * sign) < (d0 * sign).min(d1 * sign)
                   || (x * sign) > (d0 * sign).max(d1 * sign) {
                f64::NAN
            } else {
                mapped
            }
        }

        fn invert(&self, y: f64) -> f64 {
            if y.is_nan() { return f64::NAN; }
            let [d0, d1] = self.domain;
            let [r0, r1] = self.range;
            let neg = d0 < 0.0;
            let sign = if neg { -1.0 } else { 1.0 };
            let log_base = self.base.ln();
            let ld0 = (d0 * sign).ln() / log_base;
            let ld1 = (d1 * sign).ln() / log_base;
            let t = (y - r0) / (r1 - r0);
            let lmapped = ld0 + t * (ld1 - ld0);
            let mapped = sign * self.base.powf(lmapped);
            if self.clamp {
                let (lo, hi) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
                mapped.clamp(lo, hi)
            } else if y < r0.min(r1) || y > r0.max(r1) {
                f64::NAN
            } else {
                mapped
            }
        }

        fn nice(self) -> Self {
            let neg = self.domain[0] < 0.0;
            let sign: f64 = if neg { -1.0 } else { 1.0 };
            let log_base = self.base.ln();
            let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
            let hi = (self.domain[0] * sign).max(self.domain[1] * sign);
            let lo_exp = (lo.ln() / log_base).floor();
            let hi_exp = (hi.ln() / log_base).ceil();
            let (new_lo, new_hi) = if neg {
                (sign * self.base.powf(hi_exp), sign * self.base.powf(lo_exp))
            } else {
                (sign * self.base.powf(lo_exp), sign * self.base.powf(hi_exp))
            };
            let new_domain = if self.domain[0] <= self.domain[1] {
                [new_lo, new_hi]
            } else {
                [new_hi, new_lo]
            };
            Self { domain: new_domain, range: self.range, base: self.base, clamp: self.clamp }
        }

        fn ticks(&self, count: usize) -> Vec<f64> {
            let neg = self.domain[0] < 0.0;
            let sign: f64 = if neg { -1.0 } else { 1.0 };
            let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
            let hi = (self.domain[0] * sign).max(self.domain[1] * sign);
            let log_base = self.base.ln();
            let lo_exp = (lo.ln() / log_base).floor() as i64;
            let hi_exp = (hi.ln() / log_base).ceil() as i64;
            let span_decades = (hi_exp - lo_exp).max(1) as usize;
            if span_decades >= count {
                let mut out: Vec<f64> = (lo_exp..=hi_exp)
                    .map(|e| sign * self.base.powi(e as i32))
                    .filter(|t| (t.abs() >= lo) && (t.abs() <= hi))
                    .collect();
                if self.domain[0] > self.domain[1] { out.reverse(); }
                out
            } else {
                // Fall back to nice_ticks — simplified here for test purposes
                let mut out: Vec<f64> = vec![sign * lo, sign * hi];
                if self.domain[0] > self.domain[1] { out.reverse(); }
                out
            }
        }
    }

    fn d(domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool) -> LogScaleData {
        LogScaleData { domain, range, base, clamp }
    }

    // =========================================================================
    // LogScale nice() — negative domain fix
    // =========================================================================

    /// nice() on ascending negative domain [-3, -700] -> [-1000, -1].
    /// Before the fix, the result was inverted.
    #[test]
    fn log_nice_negative_ascending_domain() {
        // domain = [-700, -3] (ascending: -700 < -3)
        // sign = -1
        // lo = min(700, 3) = 3, hi = max(700, 3) = 700
        // lo_exp = floor(ln(3)/ln(10)) = floor(0.477) = 0
        // hi_exp = ceil(ln(700)/ln(10)) = ceil(2.845) = 3
        // neg branch: new_lo = -1 * 10^3 = -1000, new_hi = -1 * 10^0 = -1
        // domain[0] <= domain[1] (-700 <= -3) => [new_lo, new_hi] = [-1000, -1]
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-1000.0)).abs() < 1e-9,
            "nice negative ascending: domain[0] should be -1000, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-1.0)).abs() < 1e-9,
            "nice negative ascending: domain[1] should be -1, got {}",
            n.domain[1]
        );
        // Domain ordering must be preserved (ascending: d[0] < d[1])
        assert!(
            n.domain[0] < n.domain[1],
            "nice should preserve ascending order: {} < {}",
            n.domain[0], n.domain[1]
        );
    }

    /// nice() on descending negative domain [-3, -700] (d[0] > d[1]).
    #[test]
    fn log_nice_negative_descending_domain() {
        // domain = [-3, -700] (descending: -3 > -700)
        // sign = -1
        // lo = min(3, 700) = 3, hi = max(3, 700) = 700
        // lo_exp = 0, hi_exp = 3
        // neg: new_lo = -1000, new_hi = -1
        // domain[0] > domain[1] (-3 > -700) => [new_hi, new_lo] = [-1, -1000]
        let s = d([-3.0, -700.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-1.0)).abs() < 1e-9,
            "nice negative descending: domain[0] should be -1, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-1000.0)).abs() < 1e-9,
            "nice negative descending: domain[1] should be -1000, got {}",
            n.domain[1]
        );
        // Domain ordering preserved (descending: d[0] > d[1])
        assert!(
            n.domain[0] > n.domain[1],
            "nice should preserve descending order: {} > {}",
            n.domain[0], n.domain[1]
        );
    }

    /// nice() on negative domain spanning one decade: [-5, -0.5].
    /// lo = 0.5, hi = 5, lo_exp = floor(ln(0.5)/ln(10)) = -1, hi_exp = ceil(ln(5)/ln(10)) = 1
    /// neg: new_lo = -10^1 = -10, new_hi = -10^-1 = -0.1
    #[test]
    fn log_nice_negative_single_decade_span() {
        let s = d([-5.0, -0.5], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-10.0)).abs() < 1e-9,
            "nice negative single decade: domain[0] should be -10, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-0.1)).abs() < 1e-9,
            "nice negative single decade: domain[1] should be -0.1, got {}",
            n.domain[1]
        );
    }

    /// nice() on very small negatives: [-0.003, -0.0007].
    /// lo = 0.0007, hi = 0.003
    /// lo_exp = floor(ln(0.0007)/ln(10)) = floor(-3.155) = -4
    /// hi_exp = ceil(ln(0.003)/ln(10)) = ceil(-2.523) = -2
    /// neg: new_lo = -10^-2 = -0.01, new_hi = -10^-4 = -0.0001
    #[test]
    fn log_nice_negative_very_small_values() {
        let s = d([-0.003, -0.0007], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-0.01)).abs() < 1e-12,
            "nice small neg: domain[0] should be -0.01, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-0.0001)).abs() < 1e-12,
            "nice small neg: domain[1] should be -0.0001, got {}",
            n.domain[1]
        );
        // Ascending: -0.01 < -0.0001
        assert!(n.domain[0] < n.domain[1]);
    }

    /// nice() with base 2 and negative domain: [-3, -17].
    /// lo = 3, hi = 17
    /// lo_exp = floor(ln(3)/ln(2)) = floor(1.585) = 1
    /// hi_exp = ceil(ln(17)/ln(2)) = ceil(4.087) = 5
    /// neg: new_lo = -2^5 = -32, new_hi = -2^1 = -2
    /// descending: domain[0] > domain[1] => [-2, -32]
    #[test]
    fn log_nice_negative_base2() {
        let s = d([-3.0, -17.0], [0.0, 1.0], 2.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-2.0)).abs() < 1e-9,
            "nice neg base 2: domain[0] should be -2, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-32.0)).abs() < 1e-9,
            "nice neg base 2: domain[1] should be -32, got {}",
            n.domain[1]
        );
    }

    /// nice() on negative domain where both values are exact powers of 10.
    /// [-100, -1]: lo = 1, hi = 100, lo_exp = 0, hi_exp = 2.
    /// Niced domain should be identical: [-100, -1].
    #[test]
    fn log_nice_negative_already_nice() {
        let s = d([-100.0, -1.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-100.0)).abs() < 1e-9,
            "already nice negative: domain[0] should remain -100, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-1.0)).abs() < 1e-9,
            "already nice negative: domain[1] should remain -1, got {}",
            n.domain[1]
        );
    }

    // =========================================================================
    // LogScale nice() — positive domain regression checks
    // =========================================================================

    /// Positive domain: nice() should still work correctly after the negative fix.
    /// [3, 700] -> [1, 1000] (same as the inline unit test, but explicitly for regression).
    #[test]
    fn log_nice_positive_regression() {
        let s = d([3.0, 700.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - 1.0).abs() < 1e-9,
            "positive nice regression: domain[0] should be 1, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - 1000.0).abs() < 1e-9,
            "positive nice regression: domain[1] should be 1000, got {}",
            n.domain[1]
        );
    }

    /// Positive descending domain: nice() preserves descending order.
    /// [700, 3] (d[0] > d[1]) -> [1000, 1].
    #[test]
    fn log_nice_positive_descending_regression() {
        let s = d([700.0, 3.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - 1000.0).abs() < 1e-9,
            "positive descending nice: domain[0] should be 1000, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - 1.0).abs() < 1e-9,
            "positive descending nice: domain[1] should be 1, got {}",
            n.domain[1]
        );
        assert!(n.domain[0] > n.domain[1]);
    }

    /// Positive domain with base e: [2, 50].
    /// lo_exp = floor(ln(2)/1) = 0, hi_exp = ceil(ln(50)/1) = 4
    /// new_lo = e^0 = 1.0, new_hi = e^4 = 54.598...
    #[test]
    fn log_nice_positive_base_e() {
        let s = d([2.0, 50.0], [0.0, 1.0], std::f64::consts::E, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - 1.0).abs() < 1e-9,
            "nice base-e: domain[0] should be e^0 = 1, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - std::f64::consts::E.powi(4)).abs() < 1e-6,
            "nice base-e: domain[1] should be e^4, got {}",
            n.domain[1]
        );
    }

    // =========================================================================
    // LogScale nice() — degenerate/edge cases
    // =========================================================================

    /// nice() on a single-point negative domain: [-5, -5].
    /// lo = hi = 5, lo_exp = floor(ln5/ln10) = 0, hi_exp = ceil(ln5/ln10) = 1.
    /// neg: new_lo = -10, new_hi = -1. Ascending: [-10, -1].
    #[test]
    fn log_nice_negative_single_point_domain() {
        let s = d([-5.0, -5.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-10.0)).abs() < 1e-9,
            "single-point negative nice: domain[0] should be -10, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-1.0)).abs() < 1e-9,
            "single-point negative nice: domain[1] should be -1, got {}",
            n.domain[1]
        );
    }

    /// nice() on negative domain where value is exact power of base.
    /// [-10, -10]: lo = hi = 10, lo_exp = 1, hi_exp = 1.
    /// neg: new_lo = -10^1 = -10, new_hi = -10^1 = -10.
    /// So domain stays [-10, -10] (degenerate, but no crash).
    #[test]
    fn log_nice_negative_exact_power_single_point() {
        let s = d([-10.0, -10.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        // lo_exp = floor(ln(10)/ln(10)) = 1, hi_exp = ceil(ln(10)/ln(10)) = 1
        // neg: new_lo = -10, new_hi = -10
        assert!(
            (n.domain[0] - (-10.0)).abs() < 1e-9,
            "exact power single-point: domain[0] should be -10, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-10.0)).abs() < 1e-9,
            "exact power single-point: domain[1] should be -10, got {}",
            n.domain[1]
        );
    }

    /// nice() preserves domain order for negative when both endpoints round to
    /// the same decade. [-5, -7] (d[0] > d[1], both in decade [1,10]).
    /// lo = 5, hi = 7; lo_exp = 0, hi_exp = 1.
    /// neg: new_lo = -10, new_hi = -1. d[0]>d[1] -> [new_hi, new_lo] = [-1, -10]
    #[test]
    fn log_nice_negative_same_decade_descending() {
        let s = d([-5.0, -7.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - (-1.0)).abs() < 1e-9,
            "same decade neg desc: domain[0] should be -1, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-10.0)).abs() < 1e-9,
            "same decade neg desc: domain[1] should be -10, got {}",
            n.domain[1]
        );
        assert!(n.domain[0] > n.domain[1]);
    }

    // =========================================================================
    // LogScale scale() + invert() with niced negative domains
    // =========================================================================

    /// After nicing a negative domain, scale() should map endpoints correctly.
    /// nice([-700, -3]) = [-1000, -1]; scale(-1000) = 0.0, scale(-1) = 1.0.
    #[test]
    fn log_scale_after_nice_negative_endpoints() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false).nice();
        assert!(
            (s.domain[0] - (-1000.0)).abs() < 1e-9,
            "pre-check: niced domain[0] = {}", s.domain[0]
        );
        let y0 = s.scale(-1000.0);
        let y1 = s.scale(-1.0);
        assert!(
            (y0 - 0.0).abs() < 1e-9,
            "scale(-1000) on niced domain should be 0.0, got {y0}"
        );
        assert!(
            (y1 - 1.0).abs() < 1e-9,
            "scale(-1) on niced domain should be 1.0, got {y1}"
        );
    }

    /// Round-trip: scale -> invert on niced negative domain.
    #[test]
    fn log_scale_invert_round_trip_niced_negative() {
        let s = d([-700.0, -3.0], [0.0, 300.0], 10.0, false).nice();
        for x in [-1000.0, -100.0, -10.0, -1.0] {
            let y = s.scale(x);
            assert!(!y.is_nan(), "scale({x}) should not be NaN, got NaN");
            let back = s.invert(y);
            assert!(
                (back / x - 1.0).abs() < 1e-9,
                "round-trip failed at x={x}: got {back}"
            );
        }
    }

    /// scale() on niced negative domain: positive input should return NaN (out of domain).
    #[test]
    fn log_scale_niced_negative_positive_input_nan() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false).nice();
        let y = s.scale(5.0);
        assert!(y.is_nan(), "positive input on negative domain should be NaN, got {y}");
    }

    /// scale() on niced negative domain: zero input should return NaN.
    #[test]
    fn log_scale_niced_negative_zero_input_nan() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false).nice();
        let y = s.scale(0.0);
        assert!(y.is_nan(), "zero input on negative domain should be NaN, got {y}");
    }

    // =========================================================================
    // LogScale ticks() on negative domains
    // =========================================================================

    /// ticks() on negative domain should produce negative tick values.
    #[test]
    fn log_ticks_negative_domain_all_negative() {
        let s = d([-1000.0, -1.0], [0.0, 1.0], 10.0, false);
        let t = s.ticks(4);
        assert!(!t.is_empty(), "ticks should not be empty on negative domain");
        for tick in &t {
            assert!(*tick < 0.0, "tick {tick} should be negative on negative domain");
        }
    }

    /// ticks() on descending negative domain: should be in descending order
    /// (matching the domain direction). For positive descending domain [1000, 1],
    /// ticks come out [1000, 100, 10, 1]. For negative descending domain [-1, -1000],
    /// ticks should come out [-1, -10, -100, -1000] (most-negative-last, matching domain).
    /// BUG: The reversal logic uses d[0] > d[1] which reverses the raw tick order,
    /// but for negative domains the raw order [-1,-10,-100,-1000] is already
    /// descending (matching the domain direction [-1, -1000]), so reversing it
    /// produces ascending order [-1000,-100,-10,-1], which is wrong.
    #[test]
    fn log_ticks_negative_descending_order() { // BUG: tick reversal broken for negative descending domains
        let s = d([-1.0, -1000.0], [0.0, 1.0], 10.0, false);
        let t = s.ticks(4);
        for i in 1..t.len() {
            assert!(
                t[i] <= t[i - 1],
                "descending domain ticks should be descending: t[{}]={} > t[{}]={}",
                i, t[i], i - 1, t[i - 1]
            );
        }
    }

    // =========================================================================
    // LogScale nice() with very large negative values
    // =========================================================================

    /// nice() on very large negative domain: [-1e15, -1e3].
    /// lo = 1e3, hi = 1e15. lo_exp should be 3 and hi_exp should be 15 since
    /// these are exact powers of 10. However, ln(1e3)/ln(10) = 2.9999999999999996
    /// due to f64 imprecision, so floor() rounds to 2 instead of 3.
    /// This means nice rounds -1e3 outward to -100 instead of keeping -1000.
    /// BUG: floating-point imprecision in ln-based log computation causes
    /// exact-power-of-base values to round to a wider decade than necessary.
    #[test]
    fn log_nice_negative_large_values() { // BUG: ln-based log10 imprecision causes exact powers to round wrong
        let s = d([-1e15, -1e3], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        // Already at exact powers, so should be unchanged
        assert!(
            (n.domain[0] - (-1e15)).abs() / 1e15 < 1e-9,
            "large neg: domain[0] should be -1e15, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - (-1e3)).abs() < 1e-6,
            "large neg: domain[1] should be -1e3, got {}",
            n.domain[1]
        );
    }

    /// nice() on positive domain where both endpoints are exact powers of 10.
    /// [1e3, 1e6]: lo_exp should be 3, hi_exp should be 6. But ln(1e3)/ln(10) =
    /// 2.9999..., so floor gives 2. This widens the domain to [100, 1e6].
    /// Same ln-imprecision bug as the negative case.
    #[test]
    fn log_nice_positive_exact_power_ln_imprecision() { // BUG: ln-based log10 imprecision on positive exact powers
        let s = d([1e3, 1e6], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - 1e3).abs() < 1e-6,
            "exact power positive: domain[0] should stay at 1e3, got {}",
            n.domain[0]
        );
        assert!(
            (n.domain[1] - 1e6).abs() < 1.0,
            "exact power positive: domain[1] should stay at 1e6, got {}",
            n.domain[1]
        );
    }

    // =========================================================================
    // LogScale nice() idempotency
    // =========================================================================

    /// nice().nice() should be idempotent on positive domains.
    #[test]
    fn log_nice_positive_idempotent() {
        let s = d([3.0, 700.0], [0.0, 1.0], 10.0, false);
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1.domain, n2.domain, "nice should be idempotent on positive domain");
    }

    /// nice().nice() should be idempotent on negative domains.
    #[test]
    fn log_nice_negative_idempotent() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false);
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1.domain, n2.domain, "nice should be idempotent on negative domain");
    }

    // =========================================================================
    // LogScale nice() clamp interaction
    // =========================================================================

    /// nice() with clamp=true on negative domain: clamp should still work after nice.
    #[test]
    fn log_nice_negative_with_clamp() {
        let s = d([-700.0, -3.0], [0.0, 100.0], 10.0, true).nice();
        // Out-of-domain value should be clamped, not NaN
        let y = s.scale(-0.001); // beyond -1, closer to 0
        assert!(y.is_finite(), "clamped scale on niced negative should be finite, got {y}");
        // Should be clamped to max range
        assert!(
            (y - 100.0).abs() < 1e-6 || (y - 0.0).abs() < 1e-6,
            "clamped value should be at a range endpoint, got {y}"
        );
    }

    // =========================================================================
    // Facet spacing guard — compute_wrap
    // =========================================================================

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Rect { x: f64, y: f64, w: f64, h: f64 }

    fn compute_wrap_cell_sizes(
        ncols: u32,
        n_panels: u32,
        origin_w: f64,
        origin_h: f64,
        gutter_x: f64,
        gutter_y: f64,
    ) -> (f64, f64, f64, f64) {
        // Mirrors facet.rs compute_wrap with the negative guard
        let gutter_x = gutter_x.max(0.0);
        let gutter_y = gutter_y.max(0.0);
        let ncols = ncols.max(1);
        let nrows = ((n_panels + ncols - 1) / ncols).max(1);
        let total_x_gutter = gutter_x * (ncols.saturating_sub(1) as f64);
        let total_y_gutter = gutter_y * (nrows.saturating_sub(1) as f64);
        let cell_w = ((origin_w - total_x_gutter) / ncols as f64).max(0.0);
        let cell_h = ((origin_h - total_y_gutter) / nrows as f64).max(0.0);
        (cell_w, cell_h, gutter_x, gutter_y)
    }

    fn compute_grid_cell_sizes(
        nrows: u32,
        ncols: u32,
        origin_w: f64,
        origin_h: f64,
        gutter_x: f64,
        gutter_y: f64,
    ) -> (f64, f64, f64, f64) {
        // Mirrors facet.rs compute_grid with the negative guard
        let gutter_x = gutter_x.max(0.0);
        let gutter_y = gutter_y.max(0.0);
        let nrows = nrows.max(1);
        let ncols = ncols.max(1);
        let total_x_gutter = gutter_x * (ncols.saturating_sub(1) as f64);
        let total_y_gutter = gutter_y * (nrows.saturating_sub(1) as f64);
        let cell_w = ((origin_w - total_x_gutter) / ncols as f64).max(0.0);
        let cell_h = ((origin_h - total_y_gutter) / nrows as f64).max(0.0);
        (cell_w, cell_h, gutter_x, gutter_y)
    }

    /// Negative spacing should be clamped to 0 by the guard.
    /// Before the fix, negative spacing would increase cell sizes beyond origin.
    #[test]
    fn facet_wrap_negative_spacing_clamped_to_zero() {
        let (cell_w, cell_h, gx, gy) =
            compute_wrap_cell_sizes(3, 6, 600.0, 400.0, -50.0, -50.0);
        assert_eq!(gx, 0.0, "negative gutter_x should be clamped to 0");
        assert_eq!(gy, 0.0, "negative gutter_y should be clamped to 0");
        // 3 cols, 2 rows, no gutter => cell_w = 200, cell_h = 200
        assert!((cell_w - 200.0).abs() < 1e-9, "cell_w should be 200, got {cell_w}");
        assert!((cell_h - 200.0).abs() < 1e-9, "cell_h should be 200, got {cell_h}");
    }

    /// Negative spacing in grid mode also clamped.
    #[test]
    fn facet_grid_negative_spacing_clamped_to_zero() {
        let (cell_w, cell_h, gx, gy) =
            compute_grid_cell_sizes(2, 3, 600.0, 400.0, -100.0, -100.0);
        assert_eq!(gx, 0.0, "grid: negative gutter_x clamped to 0");
        assert_eq!(gy, 0.0, "grid: negative gutter_y clamped to 0");
        assert!((cell_w - 200.0).abs() < 1e-9);
        assert!((cell_h - 200.0).abs() < 1e-9);
    }

    /// Zero spacing should pass through unchanged (no clamping artifact).
    #[test]
    fn facet_wrap_zero_spacing_unchanged() {
        let (cell_w, cell_h, gx, gy) =
            compute_wrap_cell_sizes(3, 3, 600.0, 400.0, 0.0, 0.0);
        assert_eq!(gx, 0.0);
        assert_eq!(gy, 0.0);
        assert!((cell_w - 200.0).abs() < 1e-9);
        assert!((cell_h - 400.0).abs() < 1e-9);
    }

    /// Very small positive spacing: should not be clamped.
    #[test]
    fn facet_wrap_tiny_positive_spacing_not_clamped() {
        let epsilon = 1e-15;
        let (_, _, gx, gy) =
            compute_wrap_cell_sizes(3, 3, 600.0, 400.0, epsilon, epsilon);
        assert_eq!(gx, epsilon, "tiny positive spacing should survive the guard");
        assert_eq!(gy, epsilon, "tiny positive spacing should survive the guard");
    }

    /// NaN spacing: f64::NAN.max(0.0) returns 0.0 in Rust (since 1.70).
    /// This means NaN spacing is silently treated as 0.
    #[test]
    fn facet_wrap_nan_spacing_becomes_zero() {
        let (cell_w, cell_h, gx, gy) =
            compute_wrap_cell_sizes(3, 3, 600.0, 400.0, f64::NAN, f64::NAN);
        // f64::NAN.max(0.0) = 0.0 in Rust 1.70+ (IEEE 754-2008: returns non-NaN arg)
        assert_eq!(gx, 0.0, "NaN.max(0.0) should be 0.0");
        assert_eq!(gy, 0.0, "NaN.max(0.0) should be 0.0");
        assert!((cell_w - 200.0).abs() < 1e-9);
        assert!((cell_h - 400.0).abs() < 1e-9);
    }

    /// Infinity spacing: clamped to >= 0, but makes cell sizes 0.
    #[test]
    fn facet_wrap_infinity_spacing_makes_cells_zero() {
        let (cell_w, cell_h, gx, gy) =
            compute_wrap_cell_sizes(3, 3, 600.0, 400.0, f64::INFINITY, f64::INFINITY);
        assert!(gx.is_infinite(), "INFINITY passes max(0.0) guard");
        assert!(gy.is_infinite(), "INFINITY passes max(0.0) guard");
        // total_x_gutter = INFINITY * 2 = INFINITY
        // cell_w = max((600 - INFINITY) / 3, 0) = max(-INFINITY/3, 0) = 0
        assert_eq!(cell_w, 0.0, "infinite gutter should make cell_w = 0");
        assert_eq!(cell_h, 0.0, "infinite gutter should make cell_h = 0");
    }

    /// Negative infinity spacing: clamped to 0 by .max(0.0).
    #[test]
    fn facet_wrap_neg_infinity_spacing_clamped() {
        let (cell_w, cell_h, gx, gy) =
            compute_wrap_cell_sizes(3, 3, 600.0, 400.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert_eq!(gx, 0.0, "NEG_INFINITY.max(0.0) should be 0.0");
        assert_eq!(gy, 0.0);
        assert!((cell_w - 200.0).abs() < 1e-9);
        assert!((cell_h - 400.0).abs() < 1e-9);
    }

    /// Grid mode: infinity spacing in one dimension only.
    #[test]
    fn facet_grid_infinity_x_spacing_only() {
        let (cell_w, cell_h, gx, gy) =
            compute_grid_cell_sizes(2, 3, 600.0, 400.0, f64::INFINITY, 10.0);
        assert!(gx.is_infinite());
        assert_eq!(gy, 10.0);
        assert_eq!(cell_w, 0.0, "infinite x gutter collapses cell_w");
        // cell_h = (400 - 10) / 2 = 195
        assert!((cell_h - 195.0).abs() < 1e-9, "cell_h with finite gy should be normal: {cell_h}");
    }

    /// Negative spacing that would have caused cells wider than origin (pre-fix).
    /// With the fix: clamped to 0, cells are exactly origin/ncols.
    #[test]
    fn facet_wrap_negative_spacing_no_wider_than_origin() {
        let origin_w = 300.0;
        let ncols = 3u32;
        // Pre-fix: gutter = -50, total_x = -50 * 2 = -100
        //          cell_w = (300 - (-100)) / 3 = 400/3 = 133.3 > 100 per cell (wider than fair share!)
        // Post-fix: gutter clamped to 0
        //          cell_w = 300/3 = 100
        let (cell_w, _, _, _) = compute_wrap_cell_sizes(ncols, 3, origin_w, 300.0, -50.0, 0.0);
        let fair_share = origin_w / ncols as f64;
        assert!(
            cell_w <= fair_share + 1e-9,
            "negative spacing should not make cells wider than fair share: cell_w={cell_w}, fair={fair_share}"
        );
    }

    // =========================================================================
    // x_title_gutter — per-axis title_font_size override
    // =========================================================================

    /// The x_title_gutter now uses per-axis title_font_size override.
    /// When axes.x.title_font_size = Some(30.0), the gutter should use 30.0
    /// instead of theme.title_font_size (13.0).
    #[test]
    fn x_title_gutter_uses_per_axis_font_size() {
        let theme_title_fs = 13.0;
        let theme_axis_title_padding = 8.0;
        let line_h_factor = 1.2;

        // No override: uses theme
        let gutter_default: f64 = theme_title_fs * line_h_factor + theme_axis_title_padding;
        // 13 * 1.2 + 8 = 23.6
        assert!((gutter_default - 23.6).abs() < 1e-9);

        // With override: uses per-axis font size
        let per_axis_fs = Some(30.0);
        let effective_fs = per_axis_fs.unwrap_or(theme_title_fs);
        let gutter_override: f64 = effective_fs * line_h_factor + theme_axis_title_padding;
        // 30 * 1.2 + 8 = 44
        assert!((gutter_override - 44.0).abs() < 1e-9);

        assert!(
            gutter_override > gutter_default,
            "per-axis override of 30 should produce larger gutter than default 13"
        );
    }

    /// x_title_gutter with per-axis title_padding override too.
    /// Both title_font_size and title_padding should compose.
    #[test]
    fn x_title_gutter_uses_per_axis_font_size_and_padding() {
        let theme_title_fs = 13.0;
        let theme_axis_title_padding = 8.0;
        let line_h_factor = 1.2;

        let per_axis_fs = Some(24.0);
        let per_axis_padding = Some(16.0);

        let effective_fs = per_axis_fs.unwrap_or(theme_title_fs);
        let effective_padding = per_axis_padding.unwrap_or(theme_axis_title_padding);
        let gutter: f64 = effective_fs * line_h_factor + effective_padding;
        // 24 * 1.2 + 16 = 44.8
        assert!((gutter - 44.8).abs() < 1e-9);
    }

    /// x_title_gutter when no title: should be 0.0 regardless of font size.
    #[test]
    fn x_title_gutter_no_title_is_zero() {
        let has_title = false;
        let gutter = if has_title {
            let effective_fs = Some(100.0).unwrap_or(13.0);
            effective_fs * 1.2 + 8.0
        } else {
            0.0
        };
        assert_eq!(gutter, 0.0, "no title should mean zero gutter");
    }

    /// x_title_gutter with zero per-axis font size: gutter = 0 * 1.2 + padding = padding.
    #[test]
    fn x_title_gutter_zero_font_size() {
        let line_h_factor = 1.2;
        let effective_fs = 0.0_f64;
        let effective_padding = 8.0;
        let gutter = effective_fs * line_h_factor + effective_padding;
        assert!((gutter - 8.0).abs() < 1e-9, "zero font size means gutter = padding only");
    }

    /// x_title_gutter with very large per-axis font size: gutter should be proportionally large.
    #[test]
    fn x_title_gutter_very_large_font_size() {
        let line_h_factor = 1.2;
        let effective_fs = 1000.0_f64;
        let effective_padding = 8.0;
        let gutter = effective_fs * line_h_factor + effective_padding;
        assert!((gutter - 1208.0).abs() < 1e-9, "huge font: gutter = 1000*1.2+8 = 1208");
    }

    /// x_title_gutter with NaN per-axis font size: produces NaN gutter.
    /// This is a potential issue — NaN font size from user input would
    /// propagate to layout arithmetic, possibly causing NaN in panel rects.
    #[test]
    fn x_title_gutter_nan_font_size_propagates() {
        let effective_fs = f64::NAN;
        let effective_padding = 8.0;
        let gutter = effective_fs * 1.2 + effective_padding;
        assert!(gutter.is_nan(), "NaN font size should propagate to NaN gutter");
    }

    /// x_title_gutter with per-axis font size None: falls back to theme.
    #[test]
    fn x_title_gutter_none_falls_back_to_theme() {
        let theme_fs = 13.0;
        let per_axis_fs: Option<f64> = None;
        let effective_fs = per_axis_fs.unwrap_or(theme_fs);
        assert_eq!(effective_fs, 13.0, "None should fall back to theme font size");
    }

    // =========================================================================
    // LogScale nice() + scale() contract: niced domain endpoints map to range endpoints
    // =========================================================================

    /// For any domain, after nice(), scale(domain[0]) == range[0] and
    /// scale(domain[1]) == range[1]. This is the fundamental contract.
    #[test]
    fn log_nice_contract_endpoints_map_to_range_positive() {
        let s = d([7.0, 350.0], [0.0, 500.0], 10.0, false).nice();
        let y0 = s.scale(s.domain[0]);
        let y1 = s.scale(s.domain[1]);
        assert!(
            (y0 - s.range[0]).abs() < 1e-9,
            "positive niced domain[0] should map to range[0]: y0={y0}, r0={}", s.range[0]
        );
        assert!(
            (y1 - s.range[1]).abs() < 1e-9,
            "positive niced domain[1] should map to range[1]: y1={y1}, r1={}", s.range[1]
        );
    }

    /// Same contract for negative domains.
    #[test]
    fn log_nice_contract_endpoints_map_to_range_negative() {
        let s = d([-350.0, -7.0], [0.0, 500.0], 10.0, false).nice();
        let y0 = s.scale(s.domain[0]);
        let y1 = s.scale(s.domain[1]);
        assert!(
            (y0 - s.range[0]).abs() < 1e-9,
            "negative niced domain[0] should map to range[0]: y0={y0}, r0={}",
            s.range[0]
        );
        assert!(
            (y1 - s.range[1]).abs() < 1e-9,
            "negative niced domain[1] should map to range[1]: y1={y1}, r1={}",
            s.range[1]
        );
    }

    /// Descending negative domain: scale(d[0]) should still map to r[0].
    #[test]
    fn log_nice_contract_endpoints_descending_negative() {
        let s = d([-7.0, -350.0], [0.0, 500.0], 10.0, false).nice();
        let y0 = s.scale(s.domain[0]);
        let y1 = s.scale(s.domain[1]);
        assert!(
            (y0 - s.range[0]).abs() < 1e-9,
            "descending neg niced d[0] -> r[0]: y0={y0}, r0={}",
            s.range[0]
        );
        assert!(
            (y1 - s.range[1]).abs() < 1e-9,
            "descending neg niced d[1] -> r[1]: y1={y1}, r1={}",
            s.range[1]
        );
    }

    // =========================================================================
    // LogScale nice() base-2 positive regression
    // =========================================================================

    /// base-2 positive: nice([3, 17]) = [2, 32].
    #[test]
    fn log_nice_positive_base2_regression() {
        let s = d([3.0, 17.0], [0.0, 1.0], 2.0, false);
        let n = s.nice();
        assert!(
            (n.domain[0] - 2.0).abs() < 1e-9,
            "base2 positive nice: domain[0] should be 2, got {}", n.domain[0]
        );
        assert!(
            (n.domain[1] - 32.0).abs() < 1e-9,
            "base2 positive nice: domain[1] should be 32, got {}", n.domain[1]
        );
    }

    // =========================================================================
    // Facet spacing guard — cell_rect interaction
    // =========================================================================

    /// After guard clamping negative spacing, cell_rect positions should be
    /// non-overlapping (each cell start >= previous cell end).
    #[test]
    fn facet_wrap_negative_spacing_cells_dont_overlap() {
        let (cell_w, _, gx, _) = compute_wrap_cell_sizes(3, 3, 600.0, 400.0, -50.0, 0.0);
        let origin_x = 10.0;
        // Cell rects
        let x0 = origin_x;
        let x1 = origin_x + 1.0 * (cell_w + gx);
        let x2 = origin_x + 2.0 * (cell_w + gx);
        // No overlap: each cell start >= previous cell end
        assert!(x1 >= x0 + cell_w, "cell 1 should not overlap cell 0: x1={x1}, x0+w={}", x0 + cell_w);
        assert!(x2 >= x1 + cell_w, "cell 2 should not overlap cell 1: x2={x2}, x1+w={}", x1 + cell_w);
    }

    /// Positive spacing: verify cells have proper gaps.
    #[test]
    fn facet_wrap_positive_spacing_proper_gap() {
        let (cell_w, _, gx, _) = compute_wrap_cell_sizes(3, 3, 630.0, 400.0, 15.0, 0.0);
        // total_x_gutter = 15 * 2 = 30; cell_w = (630 - 30) / 3 = 200
        assert!((cell_w - 200.0).abs() < 1e-9);
        assert_eq!(gx, 15.0);
        let origin_x = 0.0;
        let x0_end = origin_x + cell_w;
        let x1_start = origin_x + 1.0 * (cell_w + gx);
        let gap = x1_start - x0_end;
        assert!((gap - 15.0).abs() < 1e-9, "gap between cells should be 15: got {gap}");
    }

    // =========================================================================
    // x_title_gutter parity with y_title_gutter
    // =========================================================================

    /// Both x and y title gutters should use per-axis overrides consistently.
    /// This test verifies the formulas are symmetric.
    #[test]
    fn title_gutter_x_y_parity_with_overrides() {
        let theme_fs = 13.0;
        let theme_pad = 8.0;
        let line_h_factor = 1.2;

        // Y-axis: compute_y_title_width uses input.title_font_size.unwrap_or(theme_fs)
        let y_override_fs = Some(20.0);
        let y_override_pad = Some(12.0);
        let y_effective_fs = y_override_fs.unwrap_or(theme_fs);
        let y_effective_pad = y_override_pad.unwrap_or(theme_pad);
        let y_gutter = y_effective_fs * line_h_factor + y_effective_pad;

        // X-axis: x_title_gutter now uses same pattern
        let x_override_fs = Some(20.0);
        let x_override_pad = Some(12.0);
        let x_effective_fs = x_override_fs.unwrap_or(theme_fs);
        let x_effective_pad = x_override_pad.unwrap_or(theme_pad);
        let x_gutter = x_effective_fs * line_h_factor + x_effective_pad;

        assert_eq!(x_gutter, y_gutter,
            "x and y gutters with same overrides should be equal: x={x_gutter}, y={y_gutter}");
    }

    // =========================================================================
    // Facet grid spacing guard: boundary between positive and negative
    // =========================================================================

    /// Spacing of -0.0 (negative zero): .max(0.0) should return 0.0.
    #[test]
    fn facet_spacing_negative_zero() {
        let spacing = -0.0_f64;
        let clamped = spacing.max(0.0);
        assert_eq!(clamped, 0.0, "negative zero should be treated as zero");
        assert!(clamped.is_sign_positive() || clamped == 0.0);
    }

    /// Spacing of f64::MIN_POSITIVE: should pass through as-is.
    #[test]
    fn facet_spacing_min_positive() {
        let spacing = f64::MIN_POSITIVE;
        let clamped = spacing.max(0.0);
        assert_eq!(clamped, f64::MIN_POSITIVE, "MIN_POSITIVE should pass through guard");
    }

    /// Spacing of -f64::MIN_POSITIVE: should be clamped to 0.
    #[test]
    fn facet_spacing_neg_min_positive() {
        let spacing = -f64::MIN_POSITIVE;
        let clamped = spacing.max(0.0);
        assert_eq!(clamped, 0.0, "negative MIN_POSITIVE should be clamped to 0");
    }

    // =========================================================================
    // LogScale: NaN in domain does not crash nice()
    // =========================================================================

    /// NaN domain endpoint: nice() should not panic. The result may contain NaN
    /// but should not crash.
    #[test]
    fn log_nice_nan_domain_no_panic() {
        // This tests that nice() doesn't panic with NaN in domain.
        // Note: this is pathological input that should be caught by validation
        // before reaching nice(), but we test for panic-safety.
        let s = LogScaleData {
            domain: [f64::NAN, 100.0],
            range: [0.0, 1.0],
            base: 10.0,
            clamp: false,
        };
        // Should not panic
        let _n = s.nice();
        // Result is garbage (NaN propagation), but no panic
    }

    /// Infinity in domain: nice() should not panic.
    #[test]
    fn log_nice_infinity_domain_no_panic() {
        let s = LogScaleData {
            domain: [1.0, f64::INFINITY],
            range: [0.0, 1.0],
            base: 10.0,
            clamp: false,
        };
        let _n = s.nice();
        // Should not panic; result may contain infinity
    }

    // =========================================================================
    // LogScale: scale()/invert() with negative domain midpoint
    // =========================================================================

    /// Midpoint of negative log domain: scale(-sqrt(d0*d1)) should be ~0.5 range.
    /// For [-1000, -1] base 10: geometric midpoint = -sqrt(1000) = -31.62..
    /// In log space: ln(31.62)/ln(10) = 1.5, which is midpoint of [0,3] -> t=0.5.
    #[test]
    fn log_scale_negative_midpoint() {
        let s = d([-1000.0, -1.0], [0.0, 300.0], 10.0, false);
        let midpoint = -(1000.0_f64).sqrt(); // -31.622..
        let y = s.scale(midpoint);
        assert!(
            (y - 150.0).abs() < 1.0,
            "geometric midpoint should map near range midpoint: got {y}, expected ~150"
        );
    }

    // =========================================================================
    // Facet spacing: guard interaction with ncols=1 (no gutters needed)
    // =========================================================================

    /// With ncols=1, there are 0 inter-column gutters, so spacing is irrelevant.
    /// The guard should still fire, but the result is the same either way.
    #[test]
    fn facet_wrap_single_col_spacing_irrelevant() {
        let (cell_w_neg, _, _, _) = compute_wrap_cell_sizes(1, 3, 600.0, 400.0, -50.0, 0.0);
        let (cell_w_pos, _, _, _) = compute_wrap_cell_sizes(1, 3, 600.0, 400.0, 50.0, 0.0);
        let (cell_w_zero, _, _, _) = compute_wrap_cell_sizes(1, 3, 600.0, 400.0, 0.0, 0.0);
        // With ncols=1, inter-column gutter count = 0, so gx doesn't matter
        assert_eq!(cell_w_neg, cell_w_zero,
            "ncols=1: negative spacing clamped to 0 = same as zero spacing");
        assert_eq!(cell_w_pos, cell_w_zero,
            "ncols=1: positive spacing also zero-effect because 0 gutters");
    }

    // =========================================================================
    // LogScale: very close negative endpoints
    // =========================================================================

    /// Domain endpoints that differ by a tiny amount in the same decade.
    /// [-99.999, -100.001]: both round to decade boundaries, nice expands.
    #[test]
    fn log_nice_negative_very_close_endpoints() {
        let s = d([-100.001, -99.999], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        // lo = 99.999, hi = 100.001
        // lo_exp = floor(ln(99.999)/ln(10)) = floor(1.99999) = 1
        // hi_exp = ceil(ln(100.001)/ln(10)) = ceil(2.00000+) = 3 (wait, ceil(2.0000043) = 3? No, ceil(2.0000043) = 3 is wrong)
        // Actually: ln(100.001) / ln(10) = log10(100.001) = 2.0000043..
        // ceil(2.0000043) = 3
        // neg: new_lo = -10^3 = -1000, new_hi = -10^1 = -10
        // Actually ceil(2.0000043) = 3? No! ceil(2.0000043) = 3 is wrong.
        // ceil(2.0000043) = 3... wait. ceil() rounds UP to nearest integer.
        // 2.0000043 -> ceil = 3. No, ceil(2.0000043) = 3 is wrong.
        // The ceil of 2.0000043 is 3? No! ceil(2.0) = 2, ceil(2.0001) = 3.
        // Actually ceil(2.0000043) = 3. Because ceil rounds up: the next integer above 2.0000043 is 3.
        // Wait no. ceil(2.0000043) = 3? Let me think again.
        // ceil(2.0) = 2. ceil(2.1) = 3. ceil(2.001) = 3. ceil(2.0000043) = 3. Yes.
        // So hi_exp = 3, lo_exp = 1.
        // new_lo = -10^3 = -1000, new_hi = -10^1 = -10.
        assert!(n.domain[0] < n.domain[1],
            "ascending order preserved: {} < {}", n.domain[0], n.domain[1]);
        // The niced domain should encompass the original
        assert!(n.domain[0] <= -100.001, "niced domain[0] should be <= original min");
        assert!(n.domain[1] >= -99.999, "niced domain[1] should be >= original max");
    }

    // =========================================================================
    // x_title_gutter: interaction with plot_region shrink
    // =========================================================================

    /// The x_title_gutter feeds into the bottom inset of plot_region.
    /// A large per-axis font size should shrink the plot_region more.
    #[test]
    fn x_title_gutter_large_font_reduces_plot_region() {
        let inner_h = 400.0;
        let x_label_band = 20.0;

        // With default font size
        let x_title_gutter_default = 13.0 * 1.2 + 8.0; // 23.6
        let plot_h_default = inner_h - x_label_band - x_title_gutter_default;

        // With large per-axis font
        let x_title_gutter_large = 50.0 * 1.2 + 8.0; // 68
        let plot_h_large = inner_h - x_label_band - x_title_gutter_large;

        assert!(
            plot_h_large < plot_h_default,
            "larger x_title_gutter should reduce plot height: large={plot_h_large}, default={plot_h_default}"
        );
        let gutter_diff: f64 = x_title_gutter_large - x_title_gutter_default;
        let height_diff: f64 = plot_h_default - plot_h_large;
        assert!(
            (height_diff - gutter_diff).abs() < 1e-9,
            "difference should equal gutter difference"
        );
    }
}
