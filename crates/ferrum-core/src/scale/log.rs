use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::core::{
    continuous_common, degenerate_ratio, resolve_continuous, scale_spec_to_py_dict,
    ContinuousCommonParts,
};
use super::ticks::{minor_ticks_log, nice_ticks, Tick};
use crate::spec::encoding::ScaleSpec;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LogScaleData {
    domain: [f64; 2],
    range: [f64; 2],
    base: f64,
    clamp: bool,
}

/// The two constraints `LogScale::new` enforces on a USER-SUPPLIED domain.
///
/// Consts, not inline literals, for the same reason
/// [`DEGENERATE_DOMAIN_MESSAGE`](crate::scale::core::DEGENERATE_DOMAIN_MESSAGE)
/// is: two surfaces refuse these domains — the `fm.LogScale(domain=...)`
/// constructor and the chart-level scale-domain config
/// (`configure_axis(domain_min=...)` on a log axis) — and a user who meets the
/// contract from either must read identical words.
pub(crate) const LOG_DOMAIN_CONTAINS_ZERO: &str = "log scale domain must not contain 0";
pub(crate) const LOG_DOMAIN_SIGN_MISMATCH: &str =
    "log scale domain endpoints must have the same sign";

/// Which side of zero a Log scale's domain lives on. A Log scale maps the
/// values on one side; the other side is left out of the channel
/// rather than mirrored, clamped, or silently folded in.
///
/// The side is never inferred from data: with no stated `scale=` domain it
/// is always [`Positive`](Self::Positive) ([`LogSide::of`]`(None)`), even
/// over an all-negative column — an all-negative data-derived extent refuses
/// instead ([`crate::render::RenderError::LogScaleNoValueOnSide`]), naming an
/// explicit negative `domain=` as the remedy. [`Negative`](Self::Negative) is
/// reached only by stating that domain, which
/// [`LogScaleData::validate_user_domain`] already requires to be one-signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LogSide {
    Positive,
    Negative,
}

impl LogSide {
    /// The side a stated `scale=` domain names (its first endpoint's sign),
    /// or [`Positive`](Self::Positive) when no domain is stated. `domain` is
    /// read before validation, but every reachable caller has already passed
    /// it through [`LogScaleData::validate_user_domain`] (directly, or via
    /// `render::scale_resolve::positional::validate_explicit_log_domain`),
    /// so a `Some` domain here is always nonzero and one-signed.
    pub(crate) fn of(domain: Option<&[f64]>) -> LogSide {
        match domain {
            Some([d0, ..]) if *d0 < 0.0 => LogSide::Negative,
            _ => LogSide::Positive,
        }
    }

    /// The remedy word this side's off-side rows are: the side a value must
    /// NOT be to map, in [`RenderWarning::LogScaleRowsLeftOut`] and
    /// [`RenderError::LogScaleNoValueOnSide`]'s `Display` text.
    pub(crate) fn excluded_words(self) -> &'static str {
        match self {
            LogSide::Positive => "zero or negative",
            LogSide::Negative => "zero or positive",
        }
    }
}

impl LogScaleData {
    /// Number of decades below the reference endpoint used as a floor when
    /// [`sanitize_domain`](Self::sanitize_domain) replaces a zero (or
    /// non-finite) domain endpoint. Six decades keeps the floor comfortably
    /// below any realistic data value while staying far from `f64`
    /// underflow.
    const ZERO_FLOOR_DECADES: f64 = 6.0;

    /// Construct a [`LogScaleData`], sanitizing the domain so it can never
    /// store an endpoint that `ln()` cannot handle (see
    /// [`sanitize_domain`](Self::sanitize_domain)).
    fn new(domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool) -> Self {
        Self { domain: Self::sanitize_domain(domain), range, base, clamp }
    }

    /// Whether a USER-SUPPLIED domain is one this scale kind can represent.
    ///
    /// The counterpart of [`sanitize_domain`](Self::sanitize_domain), and the
    /// distinction between them is the whole point: sanitize is the *fallback*
    /// for an AUTO-INFERRED domain, which arrives with no user input to reject
    /// (a `nice`-extended bin edge can legitimately land on 0, GH #49). A
    /// domain the caller actually wrote gets REJECTED instead — silently
    /// flooring someone's explicit `0` to `hi / 1e6` answers a different
    /// question than the one they asked.
    ///
    /// One body, two callers: [`LogScale::new`] (the `domain_user_set` arm)
    /// and `ScaleKind::validate_user_domain`, which is how the chart-level
    /// config surface asks the SCALE what it would have refused at
    /// construction rather than re-deriving log's rules a third time.
    pub(crate) fn validate_user_domain(domain: [f64; 2]) -> Result<(), &'static str> {
        let [d0, d1] = domain;
        if d0 == 0.0 || d1 == 0.0 {
            return Err(LOG_DOMAIN_CONTAINS_ZERO);
        }
        if d0.signum() != d1.signum() {
            return Err(LOG_DOMAIN_SIGN_MISMATCH);
        }
        Ok(())
    }

    /// Whether `x` can stand as a log-domain endpoint: nonzero and finite.
    fn is_loggable(x: f64) -> bool {
        x != 0.0 && x.is_finite()
    }

    /// Replace a domain endpoint that cannot be logarithmed (exactly zero,
    /// or non-finite) with a small floor value derived from the other,
    /// valid endpoint — the d3/vega-lite convention for a log domain that
    /// touches zero, rather than rejecting it outright.
    ///
    /// This matters for auto-inferred domains: e.g. a `nice`-extended
    /// histogram bin edge can legitimately land exactly on `0` even though
    /// every raw data value is strictly positive (GH #49). Without this,
    /// `ln(0) = -inf` propagates into the `as i64` exponent cast in
    /// [`ticks`](Self::ticks) and [`minor_ticks_log`](super::ticks::minor_ticks_log),
    /// which saturates to `i64::MIN`/`i64::MAX` and then panics on the
    /// subsequent `i64` subtraction (debug builds) or silently produces a
    /// bogus tick set (release builds).
    ///
    /// Well-formed domains — both endpoints finite, nonzero, same sign —
    /// pass through completely unchanged; this must never alter behavior
    /// for a valid log domain.
    fn sanitize_domain(domain: [f64; 2]) -> [f64; 2] {
        let [d0, d1] = domain;
        let d0_ok = Self::is_loggable(d0);
        let d1_ok = Self::is_loggable(d1);
        if d0_ok && d1_ok {
            return domain;
        }
        let reference = if d0_ok {
            d0
        } else if d1_ok {
            d1
        } else {
            // Neither endpoint carries a usable sign/magnitude signal (e.g.
            // both exactly 0, or both non-finite). Fall back to the same
            // [1, 10] sentinel `LogScale::new` uses when no domain is
            // supplied at all.
            return [1.0, 10.0];
        };
        let sign = reference.signum();
        let floor = sign * (reference.abs() / 10f64.powf(Self::ZERO_FLOOR_DECADES)).max(f64::MIN_POSITIVE);
        if d0_ok { [d0, floor] } else { [floor, d1] }
    }

    /// Snap a raw exponent to the nearest integer when within floating-point epsilon.
    /// Prevents `ln(1000)/ln(10) = 2.9999999999999996` from flooring to 2 instead of 3.
    fn snap_exp(raw: f64) -> f64 {
        let rounded = raw.round();
        if (raw - rounded).abs() < 1e-10 {
            rounded
        } else {
            raw
        }
    }

    /// Whether `x` lies on this scale's domain side: non-NaN, nonzero, and
    /// signed like the stored domain — an infinity matching the domain's
    /// sign is on-side too, since the side is just the sign, so a value this
    /// gate admits is clamped to the range end downstream like any other
    /// continuous scale's aux reader, never left out for being non-finite.
    /// The gate
    /// `ScaleKind::to_pixel_f64`/`to_pixel_f64_extrapolated` apply before
    /// dispatching to a `Log` scale, so a value failing this never reaches
    /// [`scale`](Self::scale) through the render path at all (left out of
    /// the channel, not clamped or NaN-mapped). `scale`/`invert`/`ticks`
    /// keep their own standalone contract unchanged — this is a render-only
    /// gate, not a new rule on the scale object itself.
    fn on_side(&self, x: f64) -> bool {
        !x.is_nan() && x != 0.0 && (self.domain[0] < 0.0) == (x < 0.0)
    }

    /// The proportional domain→range mapping, with no bounds check at
    /// either end: the arithmetic [`scale`](Self::scale) applies its own
    /// NaN/clamp bounds to, and [`place_tick`](Self::place_tick) applies
    /// its tolerance check to instead. Degenerate domain (d0 == d1, e.g. a
    /// constant-valued data column): see `degenerate_ratio`'s doc comment
    /// (GH #104) for why this must resolve to the range midpoint rather
    /// than 0/0 = NaN.
    fn mapped(&self, x: f64) -> f64 {
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let neg = d0 < 0.0;
        let sign = if neg { -1.0 } else { 1.0 };
        let log_base = self.base.ln();
        let lx = (x * sign).max(f64::MIN_POSITIVE).ln() / log_base;
        let ld0 = (d0 * sign).ln() / log_base;
        let ld1 = (d1 * sign).ln() / log_base;
        let t = degenerate_ratio(lx - ld0, ld1 - ld0);
        r0 + t * (r1 - r0)
    }

    fn scale(&self, x: f64) -> f64 {
        if x.is_nan() { return f64::NAN; }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let neg = d0 < 0.0;
        let sign = if neg { -1.0 } else { 1.0 };
        if (x * sign) <= 0.0 && !self.clamp { return f64::NAN; }
        let mapped = self.mapped(x);
        if self.clamp {
            let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
            mapped.clamp(lo, hi)
        } else if (x * sign) < (d0 * sign).min(d1 * sign) || (x * sign) > (d0 * sign).max(d1 * sign) {
            f64::NAN
        } else {
            mapped
        }
    }

    /// Places a GENERATED tick [`breaks_log`](Self::breaks_log) admits
    /// within [`TICK_DOMAIN_TOLERANCE`](Self::TICK_DOMAIN_TOLERANCE) of the
    /// domain, even when it sits a float hair outside the exact domain
    /// [`scale`](Self::scale) bounds-checks against — the one place that
    /// admission tolerance and this scale's own placement are made to
    /// agree, so a tick the generator emits is never the one value
    /// [`scale`](Self::scale) cannot place.
    ///
    /// `scale(v)` when that is finite; otherwise, for `v` within
    /// `TICK_DOMAIN_TOLERANCE` of `[lo, hi]`, placed EXACTLY at the
    /// domain's own end — the range value paired with whichever of `d0`/
    /// `d1` sits nearer `v`, not `mapped(v)`'s own extrapolated value (a
    /// float hair past that end): a generated tick this close to the
    /// boundary reads as sitting ON it, never visibly outside the plot
    /// area. A `v` outside the domain AND outside that tolerance stays
    /// unplaceable (`NaN`), exactly as `scale` already leaves it: a
    /// caller-written tick value or a data row just past an explicit
    /// domain is still left out and warned, not swept in by this.
    fn place_tick(&self, v: f64) -> f64 {
        let placed = self.scale(v);
        if placed.is_finite() {
            return placed;
        }
        let [d0, d1] = self.domain;
        let [r0, r1] = self.range;
        let neg = d0 < 0.0;
        let sign = if neg { -1.0 } else { 1.0 };
        if (v * sign) <= 0.0 {
            return f64::NAN;
        }
        let lo = (d0 * sign).min(d1 * sign);
        let hi = (d0 * sign).max(d1 * sign);
        if !Self::within_tick_tolerance(v.abs(), lo, hi) {
            return f64::NAN;
        }
        if (v * sign - d0 * sign).abs() <= (v * sign - d1 * sign).abs() {
            r0
        } else {
            r1
        }
    }

    /// The extrapolating twin of [`scale`](Self::scale) (batch-E T13,
    /// `mark_function(clip=False)`'s draw-time seam). The wrong-sign guard
    /// stays — it is a mathematical necessity (`ln` of a non-positive value
    /// under this domain's sign is undefined, not a "this value is merely
    /// out of range" refusal that extrapolation could ever override) — but
    /// the final magnitude-range branch that turns `mapped` into `NaN` is
    /// removed, matching `LinearScaleData::scale_extrapolated`'s contract.
    fn scale_extrapolated(&self, x: f64) -> f64 {
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
        let t = degenerate_ratio(lx - ld0, ld1 - ld0);
        let mapped = r0 + t * (r1 - r0);
        if self.clamp {
            let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
            mapped.clamp(lo, hi)
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

    /// Relative tolerance a candidate tick's magnitude is kept within `[lo,
    /// hi]` by: `ln(1000)/ln(10) = 2.9999999999999996`, so a power or
    /// multiple landing exactly on a snapped exponent must not be floated
    /// just outside the domain it names.
    const TICK_DOMAIN_TOLERANCE: f64 = 1e-9;

    /// `scales::breaks_log` (ggplot2): round values a log axis or legend
    /// labels, instead of the fractional exponents (`1, 1.5849, 2.5119, ...`)
    /// a linear subdivision of log-space gives. Over the domain's
    /// magnitudes `[lo, hi]` with `base`-exponents snapped by
    /// [`snap_exp`](Self::snap_exp) at both ends:
    ///
    /// 1. A degenerate domain (`lo == hi`, e.g. a constant-valued data
    ///    column) returns its one value.
    /// 2. A domain spanning less than one `base`-decade (`ln(hi/lo) <
    ///    ln(base)`, e.g. `[3, 7]` under base 10) has no decade structure to
    ///    round to — falls straight to step 6.
    /// 3. More powers of `base` within `[lo, hi]` than `count` allows
    ///    thinned to every Nth power at a
    ///    nice integer-exponent step (`nice_ticks` over the exponents —
    ///    d3's own `log.ticks` does exactly `ticks(i, j, min(j - i, n)).map(pows)`
    ///    past `n` decades), so a span of many decades stays within budget
    ///    instead of overshooting it.
    /// 4. The integer powers of `base` within `[lo, hi]`, kept with a
    ///    tolerance at the domain ends, when there are at least `count - 2`
    ///    of them (a wide domain's powers alone clear that bar).
    /// 5. Otherwise, for an integer `base`, multiples `k * base^e` with `k`
    ///    drawn from increasingly dense step sets — `{1, 3}`, `{1, 2, 5}`,
    ///    `{1..9}` for base 10, `{1..base-1}` for another integer base —
    ///    taking the DENSEST set whose tick count is AT MOST `count` —
    ///    not the first set reaching `count - 2`, which can skip past a
    ///    set that fits the budget: over `[1, 100]` at `count` 10,
    ///    `{1, 2, 5}` (7 ticks) fits, while a first-reaching rule would
    ///    pass over it for `{1..9}` (19 ticks) because 7 falls short of
    ///    `count - 2 = 8`. When no multiple set fits the budget either,
    ///    the powers themselves, which
    ///    always fit (step 4 already returned when they numbered at least
    ///    `count - 2`, so reaching here they number fewer than that, hence
    ///    fewer than `count`).
    /// 6. Otherwise (step 2's narrow domain, or a non-integer `base`):
    ///    `nice_ticks(lo, hi, count)` in value space, the same fallback d3's
    ///    `log.ticks` and ggplot2's `breaks_log` both take.
    ///
    /// Sign mirroring (a negative domain) and `should_reverse` (a
    /// descending domain) are applied last, to whichever of 1-6 produced the
    /// magnitudes.
    fn ticks(&self, count: usize) -> Vec<f64> {
        let neg = self.domain[0] < 0.0;
        let sign: f64 = if neg { -1.0 } else { 1.0 };
        let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
        let hi = (self.domain[0] * sign).max(self.domain[1] * sign);

        // Ticks are generated in ascending-exponent order.
        // For positive domains, ascending exponent = ascending value.
        // For negative domains, ascending exponent = descending value (more negative).
        // Reversal condition must account for the sign flip.
        let should_reverse = if neg {
            // Negative: ticks come out descending on the number line.
            // Reverse only if the domain is ascending (d[0] < d[1]).
            self.domain[0] < self.domain[1]
        } else {
            // Positive: ticks come out ascending on the number line.
            // Reverse only if the domain is descending (d[0] > d[1]).
            self.domain[0] > self.domain[1]
        };

        let magnitudes = Self::breaks_log(lo, hi, self.base, count);
        let mut out: Vec<f64> = magnitudes.into_iter().map(|v| sign * v).collect();
        if should_reverse { out.reverse(); }
        out
    }

    /// Whether `v` is an integer power of `base`, within
    /// [`TICK_DOMAIN_TOLERANCE`](Self::TICK_DOMAIN_TOLERANCE) — a decade, the
    /// same tolerance [`breaks_log`](Self::breaks_log) keeps a candidate
    /// power inside `[lo, hi]` with. `ScaleKind::tick_anchors` uses this to
    /// mark which of this scale's ticks are decades,
    /// so `layout::axis::anchored_visibility` can keep them through a stride
    /// cull — ggplot2's/d3's "always label the powers" rule, enforced at the
    /// cull instead of only at tick generation.
    fn is_decade(&self, v: f64) -> bool {
        let v = v.abs();
        if v <= 0.0 || !v.is_finite() {
            return false;
        }
        let log_base = self.base.ln();
        let exp = Self::snap_exp(v.ln() / log_base).round();
        let power = self.base.powi(exp as i32);
        Self::within_tick_tolerance(power, v, v)
    }

    /// Whether `v` is within [`TICK_DOMAIN_TOLERANCE`](Self::TICK_DOMAIN_TOLERANCE)
    /// of `[lo, hi]` — both strictly positive magnitudes.
    fn within_tick_tolerance(v: f64, lo: f64, hi: f64) -> bool {
        let tol = Self::TICK_DOMAIN_TOLERANCE;
        v >= lo * (1.0 - tol) && v <= hi * (1.0 + tol)
    }

    /// [`ticks`](Self::ticks)'s rule, over positive magnitudes, with no sign
    /// or reversal applied — see that method's doc for the four steps.
    fn breaks_log(lo: f64, hi: f64, base: f64, count: usize) -> Vec<f64> {
        if lo == hi {
            return vec![lo];
        }
        let log_base = base.ln();
        // Step 2: a sub-decade domain has no powers/multiples structure —
        // straight to the linear fallback.
        if hi.ln() - lo.ln() < log_base {
            return nice_ticks(lo, hi, count);
        }
        let lo_exp = Self::snap_exp(lo.ln() / log_base).floor() as i64;
        let hi_exp = Self::snap_exp(hi.ln() / log_base).ceil() as i64;
        let target = count.saturating_sub(2).max(1);

        let powers: Vec<f64> = (lo_exp..=hi_exp)
            .map(|e| base.powi(e as i32))
            .filter(|&v| Self::within_tick_tolerance(v, lo, hi))
            .collect();
        // Step 3 more powers than the budget
        // allows — thin to every Nth power at a nice integer-exponent step,
        // so a very wide span stays within `count` instead of overshooting
        // it.
        //
        // Thinned over the FIRST and LAST IN-DOMAIN exponents (`powers`'
        // own first/last entries), not the floor/ceil-extended `lo_exp`/
        // `hi_exp` above: those intentionally extend past `[lo, hi]` so the
        // `within_tick_tolerance` filter above can admit a power right at
        // the domain's edge, but `nice_ticks` has no such filter of its
        // own, so thinning over them could map an out-of-domain exponent
        // (e.g. `lo_exp` itself) straight back to a tick with nothing to
        // catch it — a non-power domain end exposes this: `[3, 7e12]`
        // floors `lo_exp` to `0`, one exponent below the first in-domain
        // power `10^1`, so thinning over `lo_exp`/`hi_exp` directly would
        // emit `10^0 = 1`, outside the domain. The final
        // `within_tick_tolerance` filter
        // below is a second, independent backstop in case `nice_ticks`
        // ever rounds past even the in-domain exponents' own span.
        if powers.len() > count {
            let first_exp = Self::snap_exp(powers[0].ln() / log_base).round();
            let last_exp = Self::snap_exp(powers[powers.len() - 1].ln() / log_base).round();
            let exps = nice_ticks(first_exp, last_exp, count);
            return exps
                .into_iter()
                .map(|e| base.powi(e.round() as i32))
                .filter(|&v| Self::within_tick_tolerance(v, lo, hi))
                .collect();
        }
        if powers.len() >= target {
            return powers;
        }

        if base.fract() == 0.0 && base >= 2.0 {
            let b = base as i64;
            let step_sets: Vec<Vec<i64>> = if b == 10 {
                vec![vec![1, 3], vec![1, 2, 5], (1..=9).collect()]
            } else {
                vec![(1..b).collect()]
            };
            // The DENSEST set whose count still fits the `count` budget —
            // scanned over every set rather than stopping at the first
            // dense-enough one, so a set that overshoots `count` never wins
            // over a sparser one that fits, even if density is not perfectly
            // monotonic at the domain's tolerance-clipped edges.
            let mut best: Option<Vec<f64>> = None;
            for steps in step_sets {
                let mut candidate: Vec<f64> = (lo_exp..=hi_exp)
                    .flat_map(|e| steps.iter().map(move |&k| k as f64 * base.powi(e as i32)))
                    .filter(|&v| Self::within_tick_tolerance(v, lo, hi))
                    .collect();
                candidate.sort_by(f64::total_cmp);
                candidate.dedup();
                if candidate.len() <= count && best.as_ref().is_none_or(|b| candidate.len() > b.len()) {
                    best = Some(candidate);
                }
            }
            if let Some(best) = best {
                return best;
            }
            // Step 5's tail no multiple set
            // fits the budget either — fall back to the powers themselves,
            // which always fit (fewer than `count - 2`, or step 4 above
            // would already have returned).
            return powers;
        }

        nice_ticks(lo, hi, count)
    }

    fn nice(self) -> Self {
        // A degenerate domain (both endpoints equal, e.g. a constant-valued
        // data column) has no span to round outward — mirrors
        // `LinearScaleData::nice`'s zero-step early return (`scale/linear.rs`),
        // so a constant column keeps the range midpoint `scale()` already
        // gives it (`degenerate_ratio`) instead of this decade-floor/ceil
        // widening it to a multi-decade domain with nothing in it.
        if self.domain[0] == self.domain[1] {
            return self;
        }
        let neg = self.domain[0] < 0.0;
        let sign: f64 = if neg { -1.0 } else { 1.0 };
        let log_base = self.base.ln();
        let lo = (self.domain[0] * sign).min(self.domain[1] * sign);
        let hi = (self.domain[0] * sign).max(self.domain[1] * sign);
        let lo_exp = Self::snap_exp(lo.ln() / log_base).floor();
        let hi_exp = Self::snap_exp(hi.ln() / log_base).ceil();
        // For positive domains: new_lo < new_hi (smallest to largest absolute value)
        // For negative domains: new_lo = sign * base^lo_exp is closest to zero (e.g. -1),
        // and new_hi = sign * base^hi_exp is most negative (e.g. -1000). Swap them so
        // the magnitude ordering matches the number-line ordering before applying the
        // ascending/descending domain logic below.
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
}

/// Continuous logarithmic scale.
///
/// Maps a numeric domain to a numeric range via a logarithmic transformation.
/// Useful for data spanning several orders of magnitude. Domain must not
/// contain zero and both endpoints must share the same sign.
///
/// Parameters
/// ----------
/// domain : tuple[float, float]
///     Input domain as ``[min, max]``. Neither endpoint may be 0 and both
///     must have the same sign.
/// range : tuple[float, float]
///     Output range as ``[lo, hi]`` pixel coordinates.
/// base : float, default 10.0
///     Logarithm base. Must be finite, positive, and not equal to 1.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
/// nice : bool, default False
///     Round domain endpoints to the nearest power of ``base``.
/// reverse : bool, default False
///     Swap the resolved domain endpoints when this scale resolves inside a
///     chart render, producing a descending axis — equivalent, AT RENDER
///     TIME, to writing ``domain=[hi, lo]`` for an explicit domain (an
///     auto-inferred domain keeps its usual padding before the swap). The
///     swap applies only at render resolution: this object's own
///     ``scale()``/``invert()``/``ticks()`` and its ``domain`` getter keep
///     reporting the constructor's domain unchanged. This diverges from
///     ``PointScale``'s identically-named ``reverse``, which DOES apply
///     inside ``PointScale.scale()``.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     chart = fr.Chart(df).encode(
///         x=fr.X("value", scale=fr.LogScale(domain=[1, 10_000], range=[0, 400]))
///     )
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, PartialEq)]
pub struct LogScale {
    data: LogScaleData,
    padding: Option<f64>,
    range_user_set: bool,
    domain_user_set: bool,
    reverse: bool,
    /// The constructor's `nice=` kwarg, carried verbatim onto the wire by
    /// `to_scale_spec` (`#153`) — see `LinearScale`'s identically-shaped
    /// field for the full rationale (data-derived domain nicing cannot
    /// happen at construction, so this flag is what tells the render path
    /// to nice the domain it infers from the column).
    nice: bool,
}

impl LogScale {
    /// Rust-side constructor (no Python validation overhead).
    pub(crate) fn new_internal(domain: Vec<f64>, range: Vec<f64>, base: f64, clamp: bool, nice: bool) -> Self {
        let mut d = LogScaleData::new([domain[0], domain[1]], [range[0], range[1]], base, clamp);
        if nice { d = d.nice(); }
        LogScale { data: d, padding: None, range_user_set: true, domain_user_set: true, reverse: false, nice }
    }

    pub(crate) fn scale_internal(&self, x: f64) -> f64 { self.data.scale(x) }

    /// `ScaleKind::tick_position_internal`'s `Log` arm — see
    /// [`LogScaleData::place_tick`].
    pub(crate) fn place_tick_internal(&self, v: f64) -> f64 { self.data.place_tick(v) }

    /// `ScaleKind::to_pixel_f64`/`_extrapolated`'s gate — see
    /// [`LogScaleData::on_side`].
    pub(crate) fn on_side(&self, x: f64) -> bool { self.data.on_side(x) }

    /// `ScaleKind::tick_anchors`'s per-value question — see
    /// [`LogScaleData::is_decade`].
    pub(crate) fn is_decade(&self, x: f64) -> bool { self.data.is_decade(x) }

    /// Crate-internal extrapolating scale call — see
    /// [`LogScaleData::scale_extrapolated`].
    pub(crate) fn scale_internal_extrapolated(&self, x: f64) -> f64 { self.data.scale_extrapolated(x) }

    pub(crate) fn ticks_internal(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return minor ticks using the standard 2-9 intra-decade multiples.
    ///
    /// For base-10 log scales this produces the visually-standard minor ticks
    /// that crowd towards the top of each decade.  For other bases the minor
    /// tick list is empty (2-9 multiples are not meaningful for non-decimal
    /// logarithm bases).
    ///
    /// The major tick count is fixed at 10 (the conventional default) to
    /// determine which decade boundaries act as major ticks.  The minor tick
    /// *positions* are always the 2-9 intra-decade multiples, independent of
    /// any count parameter.
    // Wired to the render layer via `ScaleKind::minor_tick_fractions`
    // (`render/scale_resolve/mod.rs`, dispatched through `dispatch_continuous!`).
    pub(crate) fn minor_ticks_internal(&self) -> Vec<Tick> {
        let majors = self.data.ticks(10);
        let [lo, hi] = self.data.domain;
        minor_ticks_log(lo, hi, self.data.base, &majors)
    }

    pub(crate) fn range_pair(&self) -> [f64; 2] { self.data.range }

    pub(crate) fn domain_pair(&self) -> [f64; 2] { self.data.domain }

    /// This scale's logarithm base — `render::position`'s continuous
    /// `Dodge` span split reads it to split a row's span in this scale's
    /// own units rather than data units.
    pub(crate) fn base_internal(&self) -> f64 { self.data.base }

    /// This scale's domain endpoints rounded outward to the nearest power of
    /// `base` — the exact rounding `LogScale(nice=True)` applies at
    /// construction (`LogScaleData::nice`) — without mutating `self`.
    /// `ScaleKind::niced_domain` dispatches here so the chart-level
    /// `configure_axis(nice=True)` cascade rounds a log axis in LOG space
    /// rather than the count-10 linear rounding `nice_step` would give (the
    /// S4 bug this method exists to fix: linear rounding can drive a log
    /// axis's bound to 0, which every log-scale constructor refuses).
    pub(crate) fn nice_domain_pair(&self) -> [f64; 2] { self.data.clone().nice().domain }

    /// Replace this scale's data-space domain in place, keeping its range and
    /// every kind-specific parameter.
    ///
    /// The sibling of [`domain_pair`](Self::domain_pair), added for the
    /// chart-level scale-domain config (D3, spec §4.2), which adjusts a
    /// RESOLVED domain rather than building a new scale — reconstructing via
    /// `new_internal` would have to re-supply parameters the caller cannot
    /// see. Because it is a second way into the domain field, the domain it
    /// stores must have met whatever validation this kind's own constructor
    /// applies — and for `Log` that validation is REJECTION, not repair.
    ///
    /// So the write here is deliberately raw, and the validation lives
    /// upstream at the caller: `render::scale_resolve::positional::
    /// apply_axis_domain_config` runs `ScaleKind::validate_user_domain`,
    /// which delegates to the same `LogScaleData::validate_user_domain`
    /// that `LogScale::new`'s `domain_user_set` arm calls, and refuses a
    /// zero-containing or sign-crossing pair before this is ever reached.
    /// That mirrors the constructor exactly: there too, `LogScaleData::new`'s
    /// `sanitize_domain` is a no-op pass-through for a user-set domain
    /// because the rejection already happened.
    ///
    /// Routing a USER-set domain through `sanitize_domain` here instead — as
    /// an earlier pass did — would apply the AUTO-domain fallback to caller
    /// input, silently flooring an explicit `0` to `hi / 1e6`. See
    /// `LogScaleData::sanitize_domain`'s doc for why the two are not
    /// interchangeable.
    ///
    /// `domain_user_set` flips to `true`: the domain now IS explicitly set (by
    /// the chart config), so `repr_string`/the `domain` getter must stop
    /// reporting it as data-derived.
    pub(crate) fn set_domain_pair(&mut self, domain: [f64; 2]) {
        self.data.domain = domain;
        self.domain_user_set = true;
    }

    pub(crate) fn repr_string(&self) -> String {
        let LogScaleData { domain, range, base, clamp } = &self.data;
        let domain_s = if self.domain_user_set {
            format!("[{}, {}]", domain[0], domain[1])
        } else {
            "None".to_string()
        };
        let range_s = if self.range_user_set {
            format!("[{}, {}]", range[0], range[1])
        } else {
            "None".to_string()
        };
        // `reverse` only appears when non-default (mirrors `TimeScale::repr_string`'s
        // `utc` prefix), so the default-shaped repr stays byte-identical to before.
        let reverse_s = if self.reverse { ", reverse=True" } else { "" };
        format!(
            "LogScale(domain={}, range={}, base={}, clamp={}{})",
            domain_s, range_s, base, if *clamp { "True" } else { "False" }, reverse_s
        )
    }

    /// Canonical `ScaleSpec` for this scale (SPEC-04 single-source bridge).
    /// `nice` carries the constructor's `nice=` kwarg verbatim (`#153`) —
    /// see `LinearScale::to_scale_spec`'s doc for why this does not
    /// double-apply the rounding for an explicit domain.
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec {
        ScaleSpec::Log {
            base: self.data.base,
            common: continuous_common(ContinuousCommonParts {
                domain: self.data.domain,
                domain_user_set: self.domain_user_set,
                range: self.data.range,
                range_user_set: self.range_user_set,
                clamp: self.data.clamp,
                padding: self.padding,
                reverse: self.reverse,
            }),
            nice: self.nice,
        }
    }
}

#[pymethods]
impl LogScale {
    #[new]
    #[pyo3(signature = (*, domain = None, range = None, base = 10.0, clamp = false, nice = false, padding = None, reverse = false))]
    fn new(
        domain: Option<Vec<f64>>,
        range: Option<Vec<f64>>,
        base: f64,
        clamp: bool,
        nice: bool,
        padding: Option<f64>,
        reverse: bool,
    ) -> PyResult<Self> {
        if !base.is_finite() || base <= 0.0 || base == 1.0 {
            return Err(PyValueError::new_err(format!(
                "base must be finite, > 0, and != 1; got {base}"
            )));
        }
        // Sentinel [1.0, 10.0] when no domain supplied; render-time inference
        // replaces it before any scale computation occurs.
        let resolved = resolve_continuous(domain, range, [1.0, 10.0])?;
        if resolved.domain_user_set {
            LogScaleData::validate_user_domain(resolved.domain)
                .map_err(PyValueError::new_err)?;
        }
        // `LogScaleData::new` re-runs `sanitize_domain` unconditionally, but a
        // user-set domain was already rejected above if it touched zero or
        // crossed sign — so for `domain_user_set` this is a no-op pass-through
        // (locked by `log_sanitize_domain_noop_for_well_formed_domains`), not a
        // second, looser validation. The floor/fallback behavior it performs
        // only actually fires for an *auto-inferred* domain, which reaches
        // this constructor unvalidated (there is no user input to reject).
        let mut d = LogScaleData::new(resolved.domain, resolved.range, base, clamp);
        if nice && resolved.domain_user_set {
            d = d.nice();
        }
        Ok(LogScale {
            data: d,
            padding,
            range_user_set: resolved.range_user_set,
            domain_user_set: resolved.domain_user_set,
            reverse,
            nice,
        })
    }

    /// Map a single input value ``x`` to its output range coordinate.
    fn scale(&self, x: f64) -> f64 { self.data.scale(x) }
    /// Invert a range coordinate ``y`` back to the domain.
    fn invert(&self, y: f64) -> f64 { self.data.invert(y) }

    /// Return approximately ``count`` tick values spaced logarithmically within the domain.
    #[pyo3(signature = (count = 10))]
    fn ticks(&self, count: usize) -> Vec<f64> { self.data.ticks(count) }

    /// Return a copy of this scale with domain endpoints rounded to the nearest power of ``base``.
    fn nice(&self) -> Self {
        LogScale {
            data: self.data.clone().nice(),
            padding: self.padding,
            range_user_set: self.range_user_set,
            domain_user_set: self.domain_user_set,
            reverse: self.reverse,
            nice: self.nice,
        }
    }

    /// Fractional inward pixel padding (themes-T4). ``None`` lets the renderer
    /// apply the 5% default when ``domain`` is unset.
    #[getter]
    fn padding(&self) -> Option<f64> { self.padding }

    /// Input domain as ``[min, max]``, or ``None`` when data-derived.
    #[getter]
    fn domain(&self) -> Option<Vec<f64>> {
        if self.domain_user_set { Some(self.data.domain.to_vec()) } else { None }
    }

    /// Output range as ``[lo, hi]`` pixel coordinates, or ``None`` when
    /// the renderer should auto-fill from the plot-area dimensions.
    #[getter]
    fn range(&self) -> Option<Vec<f64>> {
        if self.range_user_set { Some(self.data.range.to_vec()) } else { None }
    }

    /// Logarithm base (default 10.0).
    #[getter]
    fn base(&self) -> f64 { self.data.base }

    /// Whether out-of-domain inputs are clamped to the range endpoints.
    #[getter]
    fn clamp(&self) -> bool { self.data.clamp }

    /// Whether this scale's domain is swapped when it resolves inside a
    /// chart render (descending axis). Does not affect this object's own
    /// `scale`/`invert`/`ticks`/`domain` — unlike `PointScale::reverse`,
    /// which DOES apply inside `PointScale::scale`.
    #[getter]
    fn reverse(&self) -> bool { self.reverse }

    /// Emit this scale's canonical `ScaleSpec` as a wire dict (SPEC-04 bridge).
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scale_spec_to_py_dict(py, self.to_scale_spec())
    }

    fn __repr__(&self) -> String { self.repr_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: [f64; 2], range: [f64; 2], base: f64, clamp: bool) -> LogScaleData {
        LogScaleData::new(domain, range, base, clamp)
    }

    /// A GENERATED tick within `TICK_DOMAIN_TOLERANCE` of the domain's own
    /// end places EXACTLY at the corresponding range end (`r0` or `r1`),
    /// not `mapped(v)`'s own extrapolated value a float hair past it; a
    /// tick further out than that tolerance stays unplaceable (`NaN`), the
    /// same as `scale` itself already refuses it. An in-domain `v` is
    /// byte-identical to `scale`.
    #[test]
    fn place_tick_lands_exactly_on_the_nearer_range_end_within_tolerance() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert_eq!(s.place_tick(1000.0 * (1.0 + 1e-10)), 3.0, "just past hi, within tolerance: lands at r1 exactly");
        assert_eq!(s.place_tick(1.0 * (1.0 - 1e-10)), 0.0, "just under lo, within tolerance: lands at r0 exactly");
        assert!(s.place_tick(1000.0 * (1.0 + 1e-6)).is_nan(), "past the tolerance above hi: NaN");
        assert!(s.place_tick(1.0 * (1.0 - 1e-6)).is_nan(), "past the tolerance below lo: NaN");
        assert_eq!(s.place_tick(10.0), s.scale(10.0), "in-domain: byte-identical to scale");
    }

    /// The mirrored negative-domain side places the same way: `hi` in
    /// magnitude is the MORE negative domain end, so a tick just past it
    /// lands at the range end its signed domain endpoint pairs with.
    #[test]
    fn place_tick_lands_exactly_on_the_nearer_range_end_on_the_negative_side() {
        let s = d([-1000.0, -1.0], [0.0, 3.0], 10.0, false);
        assert_eq!(s.place_tick(-1000.0 * (1.0 + 1e-10)), 0.0, "just past the -1000 end, within tolerance: lands at r0 (d0's own range end)");
        assert_eq!(s.place_tick(-1.0 * (1.0 - 1e-10)), 3.0, "just past the -1 end, within tolerance: lands at r1 (d1's own range end)");
        assert!(s.place_tick(-1000.0 * (1.0 + 1e-6)).is_nan(), "past the tolerance: NaN");
    }

    #[test]
    fn log_scale_basic_decades() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!((s.scale(1.0) - 0.0).abs() < 1e-12);
        assert!((s.scale(10.0) - 1.0).abs() < 1e-12);
        assert!((s.scale(1000.0) - 3.0).abs() < 1e-12);
    }

    /// The extrapolating twin (batch-E T13, `mark_function(clip=False)`'s
    /// draw-time seam): an out-of-magnitude-range input continues the
    /// log-space affine mapping past the domain instead of dropping to `NaN`.
    /// Domain [1,1000] maps decade-for-decade onto [0,3]; x=10000 is one more
    /// decade past 1000, so the extrapolated value must be 4.0.
    #[test]
    fn log_scale_extrapolated_continues_past_domain() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(s.scale(10000.0).is_nan(), "the NaN-dropping sibling must still drop this");
        let extrapolated = s.scale_extrapolated(10000.0);
        assert!((extrapolated - 4.0).abs() < 1e-9, "expected 4.0, got {extrapolated}");
    }

    /// In-domain, `scale_extrapolated` is bit-identical to `scale` — only the
    /// out-of-domain branch differs.
    ///
    /// The twin is a hand-copy of `scale` minus one branch, so the standing
    /// risk is a fix landing on one and not the other. That has happened once
    /// already in this family (`symlog`'s `degenerate_ratio` fix), and only
    /// `linear.rs` guarded against it; this is that guard for `log`. Sampled
    /// across the domain, including both endpoints and a degenerate domain,
    /// since the endpoints are where the dropped branch's predicate turns over.
    #[test]
    fn log_scale_extrapolated_matches_scale_in_domain() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        for x in [1.0, 2.0, 10.0, 250.0, 999.999, 1000.0] {
            assert_eq!(
                s.scale(x).to_bits(),
                s.scale_extrapolated(x).to_bits(),
                "mismatch at x={x}: {} vs {}",
                s.scale(x),
                s.scale_extrapolated(x)
            );
        }
        // Negative domain and the degenerate equal-endpoint domain take
        // different arms of the shared body; both must agree too.
        let neg = d([-1000.0, -1.0], [0.0, 3.0], 10.0, false);
        for x in [-1000.0, -100.0, -1.0] {
            assert_eq!(neg.scale(x).to_bits(), neg.scale_extrapolated(x).to_bits(), "mismatch at x={x}");
        }
        let degenerate = d([5.0, 5.0], [0.0, 100.0], 10.0, false);
        assert_eq!(
            degenerate.scale(5.0).to_bits(),
            degenerate.scale_extrapolated(5.0).to_bits(),
            "the degenerate-domain midpoint must agree too"
        );
    }

    /// The wrong-sign guard is a mathematical necessity, not a "value is out
    /// of range" refusal extrapolation could override: a positive-domain log
    /// scale queried at a non-positive `x` must still return `NaN` under
    /// `scale_extrapolated`, exactly as `scale` does.
    #[test]
    fn log_scale_extrapolated_still_refuses_wrong_sign() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(s.scale_extrapolated(0.0).is_nan());
        assert!(s.scale_extrapolated(-5.0).is_nan());
    }

    #[test]
    fn log_inversion_round_trip() {
        let s = d([1.0, 1_000_000.0], [0.0, 6.0], 10.0, false);
        for x in [1.0, 10.0, 100.0, 12345.0, 999999.0] {
            let y = s.scale(x);
            let back = s.invert(y);
            assert!((back / x - 1.0).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    #[test]
    fn log_negative_domain_supported() {
        let s = d([-1000.0, -1.0], [0.0, 3.0], 10.0, false);
        let y = s.scale(-10.0);
        let back = s.invert(y);
        assert!((back / -10.0 - 1.0).abs() < 1e-9, "negative round-trip failed: got {back}");
    }

    /// #99/#104 residue: a degenerate equal-endpoint domain (`d0 == d1`,
    /// e.g. a constant-valued data column) used to divide by zero
    /// (`0/0 = NaN`) in the `t` ratio (`ld1 - ld0` collapses to 0 when both
    /// endpoints log to the same value). It must instead resolve to the
    /// range midpoint — finite, never NaN — on both `clamp` arms.
    /// Positive-domain edge (log's own defining constraint — domains must
    /// stay strictly one-signed) plus a negative-domain shape to confirm
    /// the sign-flip path is unaffected.
    #[test]
    fn log_scale_degenerate_positive_domain_returns_range_midpoint_not_nan() {
        let unclamped = d([5.0, 5.0], [0.0, 100.0], 10.0, false);
        let mapped = unclamped.scale(5.0);
        assert!(mapped.is_finite(), "degenerate positive log domain must be finite, got NaN");
        assert_eq!(mapped, 50.0, "degenerate positive log domain must map to the range midpoint");

        let clamped = d([5.0, 5.0], [0.0, 100.0], 10.0, true);
        let mapped_clamped = clamped.scale(5.0);
        assert!(mapped_clamped.is_finite(), "clamp=true must also be finite for a degenerate log domain");
        assert_eq!(mapped_clamped, 50.0, "clamp=true degenerate log domain must also map to the midpoint");
    }

    #[test]
    fn log_scale_degenerate_negative_domain_returns_range_midpoint_not_nan() {
        let s = d([-5.0, -5.0], [0.0, 100.0], 10.0, false);
        let mapped = s.scale(-5.0);
        assert!(mapped.is_finite(), "degenerate negative log domain must be finite, got NaN");
        assert_eq!(mapped, 50.0, "degenerate negative log domain must map to the range midpoint");
    }

    #[test]
    fn log_ticks_one_per_decade() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        let t = s.ticks(4);
        assert!(t.len() >= 3, "got {} ticks: {t:?}", t.len());
    }

    // ── `breaks_log` exact vectors ──────────────────────────────────────────

    #[test]
    fn log_breaks_powers_cover_a_wide_domain() {
        let s = d([1.0, 1000.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(5), vec![1.0, 10.0, 100.0, 1000.0]);
    }

    #[test]
    fn log_breaks_fall_to_1_2_5_multiples_when_powers_alone_fall_short() {
        let s = d([1.0, 1000.0], [0.0, 1.0], 10.0, false);
        assert_eq!(
            s.ticks(10),
            vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
        );
    }

    #[test]
    fn log_breaks_fall_to_1_3_multiples_for_a_sub_two_decade_domain() {
        let s = d([1.0, 50.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(5), vec![1.0, 3.0, 10.0, 30.0]);
    }

    #[test]
    fn log_breaks_fall_to_linear_nice_ticks_under_one_decade() {
        let s = d([3.0, 7.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(5), nice_ticks(3.0, 7.0, 5));
    }

    #[test]
    fn log_breaks_non_decimal_base_uses_every_integer_multiple() {
        let s = d([3.0, 243.0], [0.0, 1.0], 3.0, false);
        assert_eq!(s.ticks(5), vec![3.0, 9.0, 27.0, 81.0, 243.0]);
    }

    #[test]
    fn log_breaks_mirror_on_a_negative_domain() {
        // Domain `[-1000, -1]` is ASCENDING (-1000 < -1), so the ticks come
        // out ascending too (matching the domain's own order, the same
        // `should_reverse` rule every other `ticks()` case already follows).
        let s = d([-1000.0, -1.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(5), vec![-1000.0, -100.0, -10.0, -1.0]);
    }

    #[test]
    fn log_breaks_degenerate_domain_returns_its_one_value() {
        let s = d([5.0, 5.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(5), vec![5.0]);
    }

    /// 2 decades at count 10 — powers alone
    /// (3: 1, 10, 100) fall short of `count - 2 = 8`; `{1, 3}` (5 ticks) and
    /// `{1, 2, 5}` (7 ticks) both fit the `count` budget, `{1..9}` (19 ticks)
    /// does not. The densest-fit rule picks `{1, 2, 5}`, not `{1..9}`, which
    /// no stride cull from index 0 keeps every decade of.
    #[test]
    fn log_breaks_picks_the_densest_set_within_the_count_budget() {
        let s = d([1.0, 100.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(10), vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]);
    }

    /// A domain not starting on a decade boundary: `{1, 2, 5}` (`[2, 5, 10,
    /// 20]`, 4 ticks) is the densest set that fits `count` 10 — `{1..9}`
    /// would give 11 ticks (`[2..9, 10, 20, 30]`), over budget.
    #[test]
    fn log_breaks_densest_set_off_decade_boundary() {
        let s = d([2.0, 30.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(10), vec![2.0, 5.0, 10.0, 20.0]);
    }

    /// A wide domain (12 decades) stays on the unchanged powers-only branch:
    /// 13 powers already clear `count - 2 = 8`, so the densest-multiple-set
    /// rule never runs.
    #[test]
    fn log_breaks_wide_domain_keeps_every_power() {
        let s = d([1.0, 1e12], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=12).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(10), expected);
    }

    /// A domain whose powers (fewer than `count - 2`) have no dense
    /// multiple set that fits `count` either (`{1, 3}` over `[1, 1e5]` at
    /// `count` 10 is already 11 ticks) labels every power instead of
    /// falling back to linear `nice_ticks` (which would label `[1, 1e5]`
    /// at `count` 10 as `10000..100000`), so a log axis over 5-6 decades
    /// still labels only powers of the base.
    #[test]
    fn log_breaks_fall_back_to_the_powers_when_no_multiple_set_fits() {
        let s = d([1.0, 1e5], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(10), vec![1.0, 10.0, 100.0, 1e3, 1e4, 1e5]);

        let s = d([1.0, 1e6], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=6).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(10), expected);

        let s = d([1.0, 1e9], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=9).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(16), expected);

        let s = d([1.0, 1e12], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=12).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(16), expected);
    }

    /// Past the budget, the powers thin to every Nth decade at a nice integer-
    /// exponent step (d3's `log.ticks` over `i, j, min(j-i, n)`), not an
    /// unbounded list: `count` acts as a real cap on every side of the
    /// domain, wide or narrow.
    #[test]
    fn log_breaks_thin_to_every_nth_power_past_the_count() {
        let s = d([1.0, 1e20], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=20).step_by(2).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(10), expected);

        let s = d([1.0, 1e12], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=12).step_by(2).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(4), expected.clone());
        assert_eq!(s.ticks(5), expected);

        let s = d([1.0, 1e9], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(4), vec![1.0, 1e2, 1e4, 1e6, 1e8]);

        let s = d([1e-3, 1e9], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(4), vec![1e-2, 1.0, 1e2, 1e4, 1e6, 1e8]);

        let s = d([-1e12, -1.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(4), vec![-1e12, -1e10, -1e8, -1e6, -1e4, -1e2, -1.0]);
    }

    /// Every case here but one picks the densest multiple set that fits
    /// `count`, not merely the first set reaching `count - 2`. The thin
    /// branch (`[1, 1e12]` at `count` 10, the one exception: 13 powers)
    /// fires only when the in-domain powers outnumber `count`, and here it
    /// keeps every power at nice step 1. The fall-back branch, which
    /// returns the powers themselves rather than a linear fallback, fires
    /// only when the powers number fewer than `count - 2` and no dense set
    /// fits `count` either —
    /// `log_breaks_fall_back_to_the_powers_when_no_multiple_set_fits`
    /// above pins that case.
    #[test]
    fn log_breaks_keep_the_fitting_sets_unchanged() {
        let s = d([1.0, 100.0], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(10), vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]);

        let s = d([1.0, 1e4], [0.0, 1.0], 10.0, false);
        assert_eq!(
            s.ticks(10),
            vec![1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4],
        );

        let s = d([1.0, 1e5], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(16).len(), 16, "the {{1,2,5}} ladder over 1..1e5 is 16 values");

        let s = d([1.0, 1e12], [0.0, 1.0], 10.0, false);
        let expected: Vec<f64> = (0..=12).map(|e| 10f64.powi(e)).collect();
        assert_eq!(s.ticks(10), expected, "13 powers, nice step 1");

        let s = d([1.0, 1e9], [0.0, 1.0], 10.0, false);
        assert_eq!(s.ticks(32).len(), 28, "the {{1,2,5}} ladder over 1..1e9 is 28 values");
    }

    /// Step-3 thinning must run over the first and last
    /// IN-DOMAIN exponents, not the floor/ceil-extended `lo_exp`/`hi_exp`
    /// the `powers` list is built from (those extend past the domain on
    /// purpose, to catch every power `within_tick_tolerance` admits) --
    /// otherwise `nice_ticks` thins over a wider exponent span than the
    /// domain actually has, and maps an out-of-domain exponent straight
    /// back to a tick with no filter. A non-power domain end (`[3, 7e12]`:
    /// `lo_exp` floors to `0`, one exponent below the first in-domain power
    /// `10^1`; `hi_exp` ceils to `13`, one above the last in-domain power
    /// `10^12`) is exactly the case that exposes it: every tick must stay
    /// within `[lo, hi]` (`LogScaleData::within_tick_tolerance`'s own
    /// tolerance), on both signs.
    #[test]
    fn log_breaks_stay_within_the_domain_on_non_power_ends() {
        let cases: [(f64, f64, usize); 3] = [(3.0, 7e12, 10), (0.5, 3e9, 4), (20.0, 5e15, 10)];
        for (lo, hi, count) in cases {
            // (domain, the positive magnitude bounds every tick must respect)
            for (dlo, dhi) in [(lo, hi), (-hi, -lo)] {
                let s = d([dlo, dhi], [0.0, 1.0], 10.0, false);
                let ticks = s.ticks(count);
                assert!(!ticks.is_empty(), "domain [{dlo}, {dhi}] count {count}: no ticks at all");
                for &t in &ticks {
                    assert!(
                        LogScaleData::within_tick_tolerance(t.abs(), lo, hi),
                        "domain [{dlo}, {dhi}] count {count}: tick {t} falls outside [{lo}, {hi}]: {ticks:?}"
                    );
                }
                // No budget assertion here: `nice_ticks`' own step
                // rounding (mirroring d3's `tickIncrement`, which the
                // thinning step above is built on) can legitimately keep every
                // in-domain exponent when the span-to-count ratio rounds to
                // a step of 1 (`[20, 5e15]` at count 10 spans 13 exponents,
                // and 13/10 = 1.3 rounds to step 1 under the same <1.5
                // threshold `log_breaks_thin_to_every_nth_power_past_the_count`'s
                // own `[1, 1e20]`-at-10 case already exercises) -- that is
                // accepted, pre-existing thinning behavior, not the bug
                // this test pins. The bug is a tick escaping the DOMAIN,
                // checked above.
            }
        }
    }

    #[test]
    fn log_nice_rounds_to_decades() {
        let s = d([3.0, 700.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - 1.0).abs() < 1e-9);
        assert!((n.domain[1] - 1000.0).abs() < 1e-9);
    }

    /// A degenerate positive domain (a constant-valued data column) has no
    /// span to round outward: `nice()` returns it unchanged, mirroring the
    /// negative case (`log_nice_negative_single_point_domain`). Without this
    /// guard, `[5, 5]` would widen to `[1, 10]`, moving a constant column's
    /// `scale()` off the range midpoint once niced (size and
    /// stroke-width's data-derived domain is niced by default).
    #[test]
    fn log_nice_degenerate_positive_domain_returns_it_unchanged() {
        let s = d([5.0, 5.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert_eq!(n.domain, [5.0, 5.0]);
        assert_eq!(n.scale(5.0), 0.5, "the midpoint of [0, 1], not a decade-widened fraction");
    }

    /// Named-field conversion (T2.5): user-set domain/range/base round-trip,
    /// and an unset domain reports `None` while keeping the [1, 10] sentinel.
    #[test]
    fn log_named_fields_round_trip() {
        let with_domain = LogScale::new(
            Some(vec![1.0, 1000.0]), Some(vec![0.0, 300.0]), 10.0, false, false, Some(0.1), false,
        ).unwrap();
        assert_eq!(with_domain.domain(), Some(vec![1.0, 1000.0]));
        assert_eq!(with_domain.range(), Some(vec![0.0, 300.0]));
        assert_eq!(with_domain.base(), 10.0);
        assert_eq!(with_domain.padding(), Some(0.1));
        assert!(!with_domain.reverse());

        let no_domain = LogScale::new(None, None, 2.0, false, false, None, false).unwrap();
        assert_eq!(no_domain.domain(), None);
        assert_eq!(no_domain.range(), None);
        assert_eq!(no_domain.domain_pair(), [1.0, 10.0]);
        assert_eq!(no_domain.base(), 2.0);
    }

    // ── `reverse` kwarg (F-L04-07, batch-C task 2) ──────────────────────────

    #[test]
    fn log_reverse_round_trips_through_to_scale_spec() {
        let s = LogScale::new(Some(vec![1.0, 1000.0]), None, 10.0, false, false, None, true).unwrap();
        assert!(s.reverse());
        match s.to_scale_spec() {
            ScaleSpec::Log { common, .. } => {
                assert!(common.reverse, "reverse=True must survive to the wire spec");
                assert_eq!(common.domain, Some(vec![1.0, 1000.0]));
            }
            other => panic!("expected ScaleSpec::Log, got {other:?}"),
        }
    }

    /// Default (`reverse` unset) omits the wire key entirely, matching every
    /// pre-existing baseline in `test_scale_spec_parity`.
    #[test]
    fn log_reverse_default_emits_no_reverse_key() {
        let s = LogScale::new(None, None, 10.0, false, false, None, false).unwrap();
        assert!(!s.reverse());
        let json = serde_json::to_string(&s.to_scale_spec()).unwrap();
        assert!(!json.contains("reverse"), "default reverse must not appear on the wire: {json}");
    }

    // ── `nice` wire provenance (#153) ───────────────────────────────────────

    /// `LogScale(nice=True)` with NO domain used to discard `nice` at
    /// construction (`to_scale_spec` hardcoded `nice: false`) — see
    /// `LinearScale`'s identically-shaped tests for the full rationale.
    /// `nice` now carries the constructor kwarg straight through, so a
    /// render-side rebuild (which infers the domain from the encoded
    /// column) nices it, matching the raw-dict equivalent.
    #[test]
    fn log_nice_wire_flag_survives_data_derived_domain() {
        let s = LogScale::new(None, None, 10.0, false, true, None, false).unwrap();
        assert_eq!(s.domain(), None, "no domain to nice at construction — must stay unset");
        match s.to_scale_spec() {
            ScaleSpec::Log { nice, .. } => assert!(nice, "nice=True on a data-derived domain must survive to the wire spec"),
            other => panic!("expected ScaleSpec::Log, got {other:?}"),
        }
    }

    /// The explicit-domain half of `#153`'s fix is a non-change: `nice=True`
    /// with an explicit `domain=` still nices EAGERLY at construction (this
    /// object's own `.domain()` stays rounded to decades, byte-stable), and
    /// now ALSO carries `nice: true` onto the wire — not a double-nice,
    /// since `LogScaleData::nice` is idempotent on an already-niced domain
    /// (`log_nice_positive_idempotent` above).
    #[test]
    fn log_nice_wire_flag_survives_explicit_domain_already_niced() {
        let s = LogScale::new(Some(vec![3.0, 700.0]), None, 10.0, false, true, None, false).unwrap();
        assert_eq!(s.domain(), Some(vec![1.0, 1000.0]), "explicit-domain nicing must stay eager/byte-stable");
        match s.to_scale_spec() {
            ScaleSpec::Log { common, nice, .. } => {
                assert!(nice, "nice=True with an explicit domain must also survive to the wire spec");
                assert_eq!(common.domain, Some(vec![1.0, 1000.0]), "wire domain must be the already-niced value");
            }
            other => panic!("expected ScaleSpec::Log, got {other:?}"),
        }
    }

    /// Default (`nice` unset) still emits `nice: false`.
    #[test]
    fn log_nice_default_false_emits_false() {
        let s = LogScale::new(None, None, 10.0, false, false, None, false).unwrap();
        match s.to_scale_spec() {
            ScaleSpec::Log { nice, .. } => assert!(!nice),
            other => panic!("expected ScaleSpec::Log, got {other:?}"),
        }
    }

    /// `repr_string()` pinned in both directions (quality-review F2): the
    /// default-shaped repr is byte-identical to before this change, and
    /// `reverse=True` appends the exact `, reverse=True)` suffix.
    #[test]
    fn log_repr_pins_both_reverse_branches() {
        let default_scale = LogScale::new(None, None, 10.0, false, false, None, false).unwrap();
        assert_eq!(
            default_scale.repr_string(),
            "LogScale(domain=None, range=None, base=10, clamp=False)",
        );

        let reversed_scale = LogScale::new(None, None, 10.0, false, false, None, true).unwrap();
        assert_eq!(
            reversed_scale.repr_string(),
            "LogScale(domain=None, range=None, base=10, clamp=False, reverse=True)",
        );
    }

    // F17 — Underflow contract:
    //
    // For a positive domain, inputs ≤ 0 are out-of-domain. Without clamp,
    // the scale returns NaN; ScaleKind::to_pixel_f64 maps NaN to None so
    // mark renderers drop the row, matching Linear's out-of-domain
    // behavior. With clamp=true, out-of-domain inputs map to the range
    // endpoint (clamped) without producing NaN.
    //
    // This is intentional: callers either pass clamp=true (to render
    // off-axis points at the boundary) or drop the row. The earlier
    // F17 finding flagged a potential silent-NaN-leak path; these
    // tests are the contract pin.

    #[test]
    fn log_zero_input_returns_nan_unclamped() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(s.scale(0.0).is_nan(), "x=0 must be NaN on positive log domain");
    }

    #[test]
    fn log_negative_input_returns_nan_unclamped() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(s.scale(-1.0).is_nan(), "x=-1 must be NaN on positive log domain");
    }

    #[test]
    fn log_zero_input_clamps_to_range_when_clamp_enabled() {
        let s = d([1.0, 1000.0], [0.0, 3.0], 10.0, true);
        let y = s.scale(0.0);
        assert!(y.is_finite(), "x=0 with clamp must be finite, got {y}");
        // 0 < domain[0], so the clamped result is the lower range bound.
        assert_eq!(y, 0.0);
    }

    // ── `LogSide` / `on_side`: an off-side value is left out, not clamped ──

    #[test]
    fn log_side_of_no_domain_is_always_positive() {
        assert_eq!(LogSide::of(None), LogSide::Positive);
    }

    #[test]
    fn log_side_of_a_positive_domain_is_positive() {
        assert_eq!(LogSide::of(Some(&[1.0, 1000.0])), LogSide::Positive);
    }

    #[test]
    fn log_side_of_a_negative_domain_is_negative() {
        assert_eq!(LogSide::of(Some(&[-1000.0, -1.0])), LogSide::Negative);
    }

    /// `on_side` is the sign alone: an infinity matching the domain's sign
    /// is on-side (admitted), the opposite one is not; only `NaN` and `0.0`
    /// are rejected regardless of sign. Being non-finite never makes a
    /// value off-side: an off-side infinity is left out and counted like
    /// any off-side value.
    #[test]
    fn on_side_is_the_sign() {
        let positive = d([1.0, 1000.0], [0.0, 3.0], 10.0, false);
        assert!(positive.on_side(1.0) && positive.on_side(500.0) && positive.on_side(1000.0));
        assert!(positive.on_side(f64::INFINITY));
        assert!(!positive.on_side(f64::NEG_INFINITY));
        assert!(!positive.on_side(f64::NAN));
        assert!(!positive.on_side(0.0));
        assert!(!positive.on_side(-5.0));

        let negative = d([-1000.0, -1.0], [0.0, 3.0], 10.0, false);
        assert!(negative.on_side(-500.0));
        assert!(negative.on_side(f64::NEG_INFINITY));
        assert!(!negative.on_side(f64::INFINITY));
        assert!(!negative.on_side(5.0));
        assert!(!negative.on_side(0.0));
    }

    /// `on_side` is unaffected by `clamp`: clamp decides what an ADMITTED,
    /// out-of-domain-bound value maps to, not which values are admitted.
    #[test]
    fn on_side_is_unaffected_by_clamp() {
        let clamped = d([1.0, 1000.0], [0.0, 3.0], 10.0, true);
        assert!(!clamped.on_side(0.0));
        assert!(!clamped.on_side(-5.0));
        assert!(clamped.on_side(1.0));
    }

    // ── GH #49 — zero-touching domain regression ────────────────────────────
    //
    // Auto-inferred log domains can legitimately touch 0 (e.g. a
    // `nice`-extended histogram bin edge), even though the constructor
    // rejects a user-supplied domain containing 0. `sanitize_domain` clamps
    // the zero endpoint to a small floor rather than letting `ln(0) = -inf`
    // reach the `as i64` exponent cast in `ticks()`, which previously
    // saturated to `i64::MIN` and panicked on the next subtraction.

    #[test]
    fn log_ticks_finite_for_positive_domain_touching_zero() {
        let s = d([0.0, 1000.0], [0.0, 3.0], 10.0, false);
        let t = s.ticks(10);
        assert!(!t.is_empty(), "expected a non-empty tick set, got {t:?}");
        assert!(t.iter().all(|v| v.is_finite()), "all ticks must be finite: {t:?}");
        // The floored zero endpoint must not spuriously introduce a decade
        // above the real, finite endpoint.
        assert!(t.iter().all(|&v| v <= 1000.0 + 1e-6), "tick exceeds domain max: {t:?}");
    }

    #[test]
    fn log_ticks_finite_for_negative_domain_touching_zero() {
        let s = d([-1000.0, 0.0], [0.0, 3.0], 10.0, false);
        let t = s.ticks(10);
        assert!(!t.is_empty(), "expected a non-empty tick set, got {t:?}");
        assert!(t.iter().all(|v| v.is_finite()), "all ticks must be finite: {t:?}");
        assert!(t.iter().all(|&v| v >= -1000.0 - 1e-6), "tick exceeds domain min: {t:?}");
    }

    #[test]
    fn log_scale_finite_for_positive_domain_touching_zero() {
        let s = d([0.0, 1000.0], [0.0, 3.0], 10.0, false);
        // Interior, in-domain samples must map to finite pixel coordinates.
        for x in [1.0, 10.0, 500.0, 1000.0] {
            let y = s.scale(x);
            assert!(y.is_finite(), "scale({x}) must be finite, got {y}");
        }
    }

    #[test]
    fn log_scale_finite_for_negative_domain_touching_zero() {
        let s = d([-1000.0, 0.0], [0.0, 3.0], 10.0, false);
        for x in [-1.0, -10.0, -500.0, -1000.0] {
            let y = s.scale(x);
            assert!(y.is_finite(), "scale({x}) must be finite, got {y}");
        }
    }

    #[test]
    fn log_minor_ticks_finite_for_domain_touching_zero() {
        // `minor_ticks_internal` reads the stored domain directly (not
        // through `ticks()`), so it needs its own coverage: the sanitized
        // domain must also keep `minor_ticks_log` (ticks.rs) from hitting
        // the same `ln(0) = -inf` → `as i64` saturation.
        let scale = LogScale::new_internal(vec![0.0, 1000.0], vec![0.0, 600.0], 10.0, false, false);
        let minors = scale.minor_ticks_internal();
        assert!(
            minors.iter().all(|t| t.position.is_finite()),
            "all minor ticks must be finite: {minors:?}",
        );
    }

    #[test]
    fn log_sanitize_domain_noop_for_well_formed_domains() {
        // Existing, valid log domains must be returned byte-identical —
        // sanitization must never alter behavior for well-formed input.
        assert_eq!(LogScaleData::sanitize_domain([1.0, 1000.0]), [1.0, 1000.0]);
        assert_eq!(LogScaleData::sanitize_domain([-1000.0, -1.0]), [-1000.0, -1.0]);
    }

    #[test]
    fn log_sanitize_domain_both_zero_falls_back_to_sentinel() {
        assert_eq!(LogScaleData::sanitize_domain([0.0, 0.0]), [1.0, 10.0]);
    }

    // ── Minor tick tests ─────────────────────────────────────────────────────

    /// Regression: major tick positions must be stable (not changed by adding minor support).
    ///
    /// `minor_ticks_internal()` uses the fixed major count of 10 internally.
    /// The stability check compares `ticks_internal(10)` before and after — the
    /// same count that `minor_ticks_internal()` uses — so the comparison is exact.
    ///
    /// For [1, 1000] with count=10, span_decades=3 < 10 so the decade-per-tick path
    /// is NOT triggered; but for count=3 (span_decades >= count) it would be.  We
    /// also verify the count=3 path produces the expected integer-exponent ticks.
    #[test]
    fn log_major_positions_unchanged() {
        let scale = LogScale::new_internal(
            vec![1.0, 1000.0], vec![0.0, 600.0], 10.0, false, false,
        );
        // Verify count=3 triggers integer-exponent path (3 decades >= count=3).
        let majors3 = scale.ticks_internal(3);
        assert!(majors3.iter().any(|&v| (v - 1.0).abs() < 1e-9), "1 not in {majors3:?}");
        assert!(majors3.iter().any(|&v| (v - 10.0).abs() < 1e-9), "10 not in {majors3:?}");
        assert!(majors3.iter().any(|&v| (v - 100.0).abs() < 1e-9), "100 not in {majors3:?}");
        assert!(majors3.iter().any(|&v| (v - 1000.0).abs() < 1e-9), "1000 not in {majors3:?}");

        // Stability: calling minor_ticks_internal() does not mutate the scale.
        // Compare at count=10 (the count minor_ticks_internal uses internally).
        let before = scale.ticks_internal(10);
        let _ = scale.minor_ticks_internal();
        let after = scale.ticks_internal(10);
        assert_eq!(before, after);
    }

    /// Log minors for [1, 1000] land on the multiples `breaks_log`'s own
    /// majors (the 1-2-5 step set at this span/count) leave out.
    ///
    /// `minor_ticks_internal()` uses count=10 for majors internally. For
    /// [1,1000] (3 decades) at count 10, the integer powers of 10 alone
    /// (4 of them) fall short of `breaks_log`'s `count - 2 = 8` target, so
    /// majors are the 1-2-5 multiples `[1, 2, 5, 10, 20, 50, 100, 200, 500,
    /// 1000]` (10 of them) — `minor_ticks_log`'s usual 2-9-per-decade
    /// candidates dedup against THOSE, so `2`/`5` (and their per-decade
    /// multiples) are majors here, not minors, unlike a span where the
    /// majors are bare powers of 10.
    #[test]
    fn log_minors_at_standard_multiples() {
        let scale = LogScale::new_internal(
            vec![1.0, 1000.0], vec![0.0, 600.0], 10.0, false, false,
        );
        let majors = scale.ticks_internal(10);
        assert_eq!(majors, vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]);

        let minors = scale.minor_ticks_internal();
        let expected = [
            3.0, 4.0, 6.0, 7.0, 8.0, 9.0,
            30.0, 40.0, 60.0, 70.0, 80.0, 90.0,
            300.0, 400.0, 600.0, 700.0, 800.0, 900.0,
        ];
        assert_eq!(minors.len(), expected.len(), "expected {} minors, got {}: {:?}", expected.len(), minors.len(), minors);

        let positions: Vec<f64> = minors.iter().map(|t| t.position).collect();
        for m in expected {
            assert!(
                positions.iter().any(|&p| (p - m).abs() < 1e-9),
                "{m} not in {positions:?}",
            );
        }
        assert!(minors.iter().all(|t| !t.is_major));
    }

    /// Log minors have non-uniform spacing in log space (crowding at top of decade).
    ///
    /// `minor_ticks_internal()` uses count=10 for majors internally.
    #[test]
    fn log_minors_nonuniform_in_linear_space() {
        let scale = LogScale::new_internal(
            vec![1.0, 1000.0], vec![0.0, 600.0], 10.0, false, false,
        );
        let minors = scale.minor_ticks_internal();
        // First decade: 2,3,...,9 — uniform in LINEAR space but NOT in LOG space.
        let first_decade: Vec<f64> = minors
            .iter()
            .filter(|t| t.position >= 2.0 && t.position < 10.0)
            .map(|t| t.position)
            .collect();
        let log_gaps: Vec<f64> = first_decade
            .windows(2)
            .map(|w| w[1].ln() - w[0].ln())
            .collect();
        let first_gap = log_gaps[0];
        let all_equal = log_gaps.iter().all(|&g| (g - first_gap).abs() < 1e-9);
        assert!(!all_equal, "log-space gaps should be non-uniform: {log_gaps:?}");
    }

    /// Log major positions are unchanged by the introduction of minor ticks.
    ///
    /// Both calls use count=10 (the fixed internal count for `minor_ticks_internal`).
    #[test]
    fn log_majors_are_preserved_after_minor_call() {
        let scale = LogScale::new_internal(
            vec![1.0, 1000.0], vec![0.0, 600.0], 10.0, false, false,
        );
        let before = scale.ticks_internal(10);
        let _ = scale.minor_ticks_internal(); // side-effect free
        let after = scale.ticks_internal(10);
        assert_eq!(before, after);
    }

    // ── nice() negative-domain fix (ported from tests/bug_hunt_scale_layout.rs, R1) ──
    //
    // Prior in-src coverage only exercised `nice()` on a positive domain
    // (`log_nice_rounds_to_decades`); none of these negative-domain, sign, or
    // ordering cases had any coverage before this port.

    /// nice() on ascending negative domain [-700, -3] -> [-1000, -1], and
    /// ascending order is preserved.
    #[test]
    fn log_nice_negative_ascending_domain() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-1000.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-1.0)).abs() < 1e-9, "got {}", n.domain[1]);
        assert!(n.domain[0] < n.domain[1], "ascending order must be preserved");
    }

    /// nice() on descending negative domain [-3, -700] (d[0] > d[1]) preserves
    /// descending order: [-1, -1000].
    #[test]
    fn log_nice_negative_descending_domain() {
        let s = d([-3.0, -700.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-1.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-1000.0)).abs() < 1e-9, "got {}", n.domain[1]);
        assert!(n.domain[0] > n.domain[1], "descending order must be preserved");
    }

    /// nice() on negative domain spanning one decade: [-5, -0.5] -> [-10, -0.1].
    #[test]
    fn log_nice_negative_single_decade_span() {
        let s = d([-5.0, -0.5], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-10.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-0.1)).abs() < 1e-9, "got {}", n.domain[1]);
    }

    /// nice() on very small negatives: [-0.003, -0.0007] -> [-0.01, -0.0001].
    #[test]
    fn log_nice_negative_very_small_values() {
        let s = d([-0.003, -0.0007], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-0.01)).abs() < 1e-12, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-0.0001)).abs() < 1e-12, "got {}", n.domain[1]);
        assert!(n.domain[0] < n.domain[1]);
    }

    /// nice() with base 2 and negative domain: [-3, -17] -> [-2, -32], preserving
    /// the descending order the ascending-magnitude/negative-sign combination produces.
    #[test]
    fn log_nice_negative_base2() {
        let s = d([-3.0, -17.0], [0.0, 1.0], 2.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-2.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-32.0)).abs() < 1e-9, "got {}", n.domain[1]);
    }

    /// nice() on negative domain where both values are exact powers of 10:
    /// [-100, -1] is already nice and must round-trip unchanged.
    #[test]
    fn log_nice_negative_already_nice() {
        let s = d([-100.0, -1.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-100.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-1.0)).abs() < 1e-9, "got {}", n.domain[1]);
    }

    /// Positive descending domain [700, 3] -> [1000, 1]: nice() preserves
    /// descending order on the positive side too (the ascending case is
    /// already covered by `log_nice_rounds_to_decades`).
    #[test]
    fn log_nice_positive_descending_regression() {
        let s = d([700.0, 3.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - 1000.0).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - 1.0).abs() < 1e-9, "got {}", n.domain[1]);
        assert!(n.domain[0] > n.domain[1]);
    }

    /// Positive domain with a non-decimal base (e): [2, 50] -> [e^0, e^4].
    #[test]
    fn log_nice_positive_base_e() {
        let s = d([2.0, 50.0], [0.0, 1.0], std::f64::consts::E, false);
        let n = s.nice();
        assert!((n.domain[0] - 1.0).abs() < 1e-9, "domain[0] should be e^0 = 1, got {}", n.domain[0]);
        assert!(
            (n.domain[1] - std::f64::consts::E.powi(4)).abs() < 1e-6,
            "domain[1] should be e^4, got {}", n.domain[1]
        );
    }

    /// nice() on a single-point (degenerate) negative domain returns the
    /// domain unchanged: [-5, -5] stays [-5, -5], the midpoint `scale()`
    /// already maps every value to, not the decade-widened [-10, -1] a
    /// floor/ceil rounding would give a domain with no span.
    #[test]
    fn log_nice_negative_single_point_domain() {
        let s = d([-5.0, -5.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert_eq!(n.domain, [-5.0, -5.0]);
    }

    /// nice() on a degenerate negative domain that's already an exact power:
    /// [-10, -10] stays [-10, -10] (no crash, no spurious decade widening).
    #[test]
    fn log_nice_negative_exact_power_single_point() {
        let s = d([-10.0, -10.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-10.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-10.0)).abs() < 1e-9, "got {}", n.domain[1]);
    }

    /// nice() preserves domain order for negative when both endpoints round
    /// to the same decade: [-5, -7] (d[0] > d[1]) -> [-1, -10].
    #[test]
    fn log_nice_negative_same_decade_descending() {
        let s = d([-5.0, -7.0], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-1.0)).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-10.0)).abs() < 1e-9, "got {}", n.domain[1]);
        assert!(n.domain[0] > n.domain[1]);
    }

    /// After nicing a negative domain, `scale()` maps the niced endpoints to
    /// the range endpoints: nice([-700, -3]) = [-1000, -1]; scale(-1000) = 0.0,
    /// scale(-1) = 1.0.
    #[test]
    fn log_scale_after_nice_negative_endpoints() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false).nice();
        assert!((s.domain[0] - (-1000.0)).abs() < 1e-9, "pre-check: niced domain[0] = {}", s.domain[0]);
        let y0 = s.scale(-1000.0);
        let y1 = s.scale(-1.0);
        assert!((y0 - 0.0).abs() < 1e-9, "scale(-1000) on niced domain should be 0.0, got {y0}");
        assert!((y1 - 1.0).abs() < 1e-9, "scale(-1) on niced domain should be 1.0, got {y1}");
    }

    /// Round-trip: `scale` -> `invert` on a niced negative domain.
    #[test]
    fn log_scale_invert_round_trip_niced_negative() {
        let s = d([-700.0, -3.0], [0.0, 300.0], 10.0, false).nice();
        for x in [-1000.0, -100.0, -10.0, -1.0] {
            let y = s.scale(x);
            assert!(!y.is_nan(), "scale({x}) should not be NaN, got NaN");
            let back = s.invert(y);
            assert!((back / x - 1.0).abs() < 1e-9, "round-trip failed at x={x}: got {back}");
        }
    }

    /// `scale()` on a niced negative domain: a positive input is out of
    /// domain and returns NaN (unclamped) — the mirror image of
    /// `log_zero_input_returns_nan_unclamped` on a positive domain.
    #[test]
    fn log_scale_niced_negative_positive_input_nan() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false).nice();
        let y = s.scale(5.0);
        assert!(y.is_nan(), "positive input on negative domain should be NaN, got {y}");
    }

    /// `scale()` on a niced negative domain: zero input is out of domain and
    /// returns NaN (unclamped).
    #[test]
    fn log_scale_niced_negative_zero_input_nan() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false).nice();
        let y = s.scale(0.0);
        assert!(y.is_nan(), "zero input on negative domain should be NaN, got {y}");
    }

    /// `ticks()` on a negative domain produces only negative tick values.
    #[test]
    fn log_ticks_negative_domain_all_negative() {
        let s = d([-1000.0, -1.0], [0.0, 1.0], 10.0, false);
        let t = s.ticks(4);
        assert!(!t.is_empty(), "ticks should not be empty on negative domain");
        for tick in &t {
            assert!(*tick < 0.0, "tick {tick} should be negative on negative domain");
        }
    }

    /// `ticks()` on a descending negative domain [-1, -1000] come out in
    /// descending order (matching the domain direction), exercising the
    /// `should_reverse` sign-aware branch for negative domains.
    #[test]
    fn log_ticks_negative_descending_order() {
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

    /// nice() on a very large negative domain of exact powers of 10:
    /// [-1e15, -1e3] must stay unchanged — `snap_exp`'s epsilon tolerance
    /// absorbs the `ln`-based imprecision (`ln(1e3)/ln(10)` is not exactly
    /// 3.0 in f64) so the domain doesn't spuriously widen a decade.
    #[test]
    fn log_nice_negative_large_values() {
        let s = d([-1e15, -1e3], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - (-1e15)).abs() / 1e15 < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - (-1e3)).abs() < 1e-6, "got {}", n.domain[1]);
    }

    /// nice() on a positive domain of exact powers of 10 (same `snap_exp`
    /// contract as the negative case above): [1e3, 1e6] stays unchanged.
    #[test]
    fn log_nice_positive_exact_power_ln_imprecision() {
        let s = d([1e3, 1e6], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!((n.domain[0] - 1e3).abs() < 1e-6, "got {}", n.domain[0]);
        assert!((n.domain[1] - 1e6).abs() < 1.0, "got {}", n.domain[1]);
    }

    /// nice().nice() is idempotent on positive domains.
    #[test]
    fn log_nice_positive_idempotent() {
        let s = d([3.0, 700.0], [0.0, 1.0], 10.0, false);
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1.domain, n2.domain, "nice should be idempotent on positive domain");
    }

    /// nice().nice() is idempotent on negative domains.
    #[test]
    fn log_nice_negative_idempotent() {
        let s = d([-700.0, -3.0], [0.0, 1.0], 10.0, false);
        let n1 = s.clone().nice();
        let n2 = n1.clone().nice();
        assert_eq!(n1.domain, n2.domain, "nice should be idempotent on negative domain");
    }

    /// nice() with clamp=true on a negative domain: clamp still works after
    /// nicing (out-of-domain input is clamped, not NaN).
    #[test]
    fn log_nice_negative_with_clamp() {
        let s = d([-700.0, -3.0], [0.0, 100.0], 10.0, true).nice();
        let y = s.scale(-0.001);
        assert!(y.is_finite(), "clamped scale on niced negative should be finite, got {y}");
        assert!(
            (y - 100.0).abs() < 1e-6 || (y - 0.0).abs() < 1e-6,
            "clamped value should be at a range endpoint, got {y}"
        );
    }

    /// For any domain, after nice(), scale(domain[0]) == range[0] and
    /// scale(domain[1]) == range[1] — the fundamental contract, on a positive
    /// domain.
    #[test]
    fn log_nice_contract_endpoints_map_to_range_positive() {
        let s = d([7.0, 350.0], [0.0, 500.0], 10.0, false).nice();
        let y0 = s.scale(s.domain[0]);
        let y1 = s.scale(s.domain[1]);
        assert!((y0 - s.range[0]).abs() < 1e-9, "y0={y0}, r0={}", s.range[0]);
        assert!((y1 - s.range[1]).abs() < 1e-9, "y1={y1}, r1={}", s.range[1]);
    }

    /// Same endpoint-to-range contract for negative domains.
    #[test]
    fn log_nice_contract_endpoints_map_to_range_negative() {
        let s = d([-350.0, -7.0], [0.0, 500.0], 10.0, false).nice();
        let y0 = s.scale(s.domain[0]);
        let y1 = s.scale(s.domain[1]);
        assert!((y0 - s.range[0]).abs() < 1e-9, "y0={y0}, r0={}", s.range[0]);
        assert!((y1 - s.range[1]).abs() < 1e-9, "y1={y1}, r1={}", s.range[1]);
    }

    /// Same endpoint-to-range contract for a descending negative domain.
    #[test]
    fn log_nice_contract_endpoints_descending_negative() {
        let s = d([-7.0, -350.0], [0.0, 500.0], 10.0, false).nice();
        let y0 = s.scale(s.domain[0]);
        let y1 = s.scale(s.domain[1]);
        assert!((y0 - s.range[0]).abs() < 1e-9, "y0={y0}, r0={}", s.range[0]);
        assert!((y1 - s.range[1]).abs() < 1e-9, "y1={y1}, r1={}", s.range[1]);
    }

    /// base-2 positive regression: nice([3, 17]) = [2, 32].
    #[test]
    fn log_nice_positive_base2_regression() {
        let s = d([3.0, 17.0], [0.0, 1.0], 2.0, false);
        let n = s.nice();
        assert!((n.domain[0] - 2.0).abs() < 1e-9, "got {}", n.domain[0]);
        assert!((n.domain[1] - 32.0).abs() < 1e-9, "got {}", n.domain[1]);
    }

    /// A NaN domain endpoint must not panic `nice()`. This is pathological
    /// input that the constructor's `sanitize_domain` would normally
    /// intercept — bypassing `new()` via a direct struct literal exercises
    /// `nice()`'s own panic-safety as a defensive backstop.
    #[test]
    fn log_nice_nan_domain_no_panic() {
        let s = LogScaleData { domain: [f64::NAN, 100.0], range: [0.0, 1.0], base: 10.0, clamp: false };
        let _n = s.nice(); // must not panic; result may contain NaN
    }

    /// An infinite domain endpoint must not panic `nice()` (same
    /// sanitize-bypass rationale as the NaN case above).
    #[test]
    fn log_nice_infinity_domain_no_panic() {
        let s = LogScaleData { domain: [1.0, f64::INFINITY], range: [0.0, 1.0], base: 10.0, clamp: false };
        let _n = s.nice(); // must not panic; result may contain infinity
    }

    /// The geometric midpoint of a negative log domain maps near the range
    /// midpoint: for [-1000, -1] base 10, geometric midpoint = -sqrt(1000) ≈
    /// -31.62, whose log-space position is 1.5 (midpoint of [0,3]) -> t=0.5.
    #[test]
    fn log_scale_negative_midpoint() {
        let s = d([-1000.0, -1.0], [0.0, 300.0], 10.0, false);
        let midpoint = -(1000.0_f64).sqrt();
        let y = s.scale(midpoint);
        assert!((y - 150.0).abs() < 1.0, "geometric midpoint should map near range midpoint: got {y}, expected ~150");
    }

    /// Domain endpoints that differ by a tiny amount within the same decade:
    /// [-100.001, -99.999] — nice() must widen to encompass the original
    /// span and preserve ascending order.
    #[test]
    fn log_nice_negative_very_close_endpoints() {
        let s = d([-100.001, -99.999], [0.0, 1.0], 10.0, false);
        let n = s.nice();
        assert!(n.domain[0] < n.domain[1], "ascending order preserved: {} < {}", n.domain[0], n.domain[1]);
        assert!(n.domain[0] <= -100.001, "niced domain[0] should be <= original min");
        assert!(n.domain[1] >= -99.999, "niced domain[1] should be >= original max");
    }
}
