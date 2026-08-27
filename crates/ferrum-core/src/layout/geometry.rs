//! Pixel-space geometry primitives. Coordinates are f64; positive-y is downward
//! (consistent with SVG/screen conventions). All `shrink`/`split_*` operations
//! return new values; `Rect` and `Inset` are `Copy`.

use serde::{Deserialize, Serialize};

/// Themes-T4 default inward padding fraction for quantitative / temporal
/// scales when neither `scale.padding` nor `scale.domain` is set.
pub(crate) const DEFAULT_SCALE_PADDING_FRAC: f64 = 0.05;

/// Themes-T4 maximum inward padding in pixels. The padding band is the
/// smaller of `fraction * span` and this cap so small panels don't lose
/// too much plot area.
pub(crate) const SCALE_PADDING_MAX_PX: f64 = 8.0;

/// Inset a pixel range by the resolved padding band (capped at
/// `SCALE_PADDING_MAX_PX`). Handles inverted ranges (the y-axis range is
/// `(high_px, low_px)` for quantitative scales).
///
/// This is the single source of truth for the positional-scale padding inset.
/// `scale_resolve::positional` re-exports it so that data marks and
/// scale-projected axis ticks share *exactly* the same inset — a tick at value
/// `v` and a data mark at value `v` land on the same pixel.
pub(crate) fn inset_pixel_range(pr: (f64, f64), padding_frac: f64) -> (f64, f64) {
    if padding_frac == 0.0 {
        return pr;
    }
    let span = (pr.1 - pr.0).abs();
    let pad = (span * padding_frac).min(SCALE_PADDING_MAX_PX);
    let sign = if pr.1 >= pr.0 { 1.0 } else { -1.0 };
    (pr.0 + sign * pad, pr.1 - sign * pad)
}

/// A 1-D pixel interval `[lo, hi]` (orientation-bearing: `lo`/`hi` are the *start*
/// and *end* pixels in placement order, so an inverted range — e.g. the y-axis
/// `(bottom, top)` or the colorbar's `bar_bottom → bar_top` — is expressed by
/// passing the start as `lo` and the end as `hi`, not by normalizing). It is the
/// single placement core for the layout's "map a fraction / distribute N items
/// over a 1-D extent" arithmetic, replacing the hand-rolled `lo + t*span` lerps
/// that were copied across the two tick projectors, the uniform-slot fallbacks,
/// and the colorbar tick distribution (cohesion finding LAYOUT-850).
///
/// `lerp` is the only arithmetic primitive (`lo + t*(hi - lo)`); `uniform_center`
/// takes a *precomputed* slot size rather than recomputing `(hi - lo)/n` so it is
/// byte-identical to the original `origin + (i + 0.5)*slot` call sites (a
/// recomputed span is not bit-for-bit equal to a separately-computed slot).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Axis1D {
    pub lo: f64,
    pub hi: f64,
}

impl Axis1D {
    /// Linear interpolation: `lo + t*(hi - lo)`. With `t` in `[0, 1]`, `lerp(0) == lo`
    /// and `lerp(1) == hi`; for an inverted interval (`lo > hi`) increasing `t`
    /// moves from `lo` toward `hi`. This is exactly the per-call lerp the tick
    /// projectors and colorbar previously hand-rolled.
    pub(crate) fn lerp(&self, t: f64) -> f64 {
        self.lo + t * (self.hi - self.lo)
    }

    /// Center pixel of uniform slot `i` (0-based) given a *precomputed* `slot`
    /// size: `lo + (i + 0.5)*slot`. The caller supplies `slot` (typically
    /// `extent / n`, guarded for `n == 0`) so the result is byte-identical to the
    /// original inline uniform-slot fallbacks; this method intentionally does not
    /// recompute `(hi - lo)/n` (which would not be bit-for-bit equal).
    pub(crate) fn uniform_center(&self, i: usize, slot: f64) -> f64 {
        self.lo + (i as f64 + 0.5) * slot
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub const ZERO: Rect = Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };

    /// Shrink by an inset on each side. Returns `Rect::ZERO` if the inset
    /// would collapse either dimension to ≤ 0.
    pub fn shrink(&self, inset: Inset) -> Rect {
        let w = self.w - inset.left - inset.right;
        let h = self.h - inset.top - inset.bottom;
        if w <= 0.0 || h <= 0.0 {
            return Rect::ZERO;
        }
        Rect { x: self.x + inset.left, y: self.y + inset.top, w, h }
    }

    /// Split off a strip of height `h` from the top. Returns `(strip, remainder)`.
    /// If `h >= self.h`, strip == self and remainder == ZERO.
    pub fn split_top(&self, h: f64) -> (Rect, Rect) {
        let h = h.min(self.h).max(0.0);
        let strip = Rect { x: self.x, y: self.y, w: self.w, h };
        let remainder = Rect {
            x: self.x,
            y: self.y + h,
            w: self.w,
            h: self.h - h,
        };
        (strip, remainder)
    }

    pub fn split_bottom(&self, h: f64) -> (Rect, Rect) {
        let h = h.min(self.h).max(0.0);
        let remainder = Rect { x: self.x, y: self.y, w: self.w, h: self.h - h };
        let strip = Rect {
            x: self.x,
            y: self.y + self.h - h,
            w: self.w,
            h,
        };
        (strip, remainder)
    }

    pub fn split_left(&self, w: f64) -> (Rect, Rect) {
        let w = w.min(self.w).max(0.0);
        let strip = Rect { x: self.x, y: self.y, w, h: self.h };
        let remainder = Rect {
            x: self.x + w,
            y: self.y,
            w: self.w - w,
            h: self.h,
        };
        (strip, remainder)
    }

    pub fn split_right(&self, w: f64) -> (Rect, Rect) {
        let w = w.min(self.w).max(0.0);
        let remainder = Rect { x: self.x, y: self.y, w: self.w - w, h: self.h };
        let strip = Rect {
            x: self.x + self.w - w,
            y: self.y,
            w,
            h: self.h,
        };
        (strip, remainder)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Inset {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Inset {
    pub const fn uniform(v: f64) -> Inset {
        Inset { top: v, right: v, bottom: v, left: v }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

impl Viewport {
    pub fn into_rect(self) -> Rect {
        Rect { x: 0.0, y: 0.0, w: self.width, h: self.height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn rect_shrink_normal() {
        let r0 = r(0.0, 0.0, 100.0, 50.0);
        let r1 = r0.shrink(Inset::uniform(5.0));
        assert_eq!(r1, r(5.0, 5.0, 90.0, 40.0));
    }

    #[test]
    fn rect_shrink_collapses_to_zero() {
        let r0 = r(0.0, 0.0, 10.0, 10.0);
        let r1 = r0.shrink(Inset::uniform(10.0));
        assert_eq!(r1, Rect::ZERO);
    }

    #[test]
    fn rect_shrink_collapses_one_dim_to_zero() {
        // Left+right > w but top+bottom fits.
        let r0 = r(0.0, 0.0, 10.0, 100.0);
        let r1 = r0.shrink(Inset { top: 5.0, right: 6.0, bottom: 5.0, left: 6.0 });
        assert_eq!(r1, Rect::ZERO);
    }

    #[test]
    fn split_top_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (top, rest) = r0.split_top(30.0);
        assert_eq!(top, r(10.0, 20.0, 100.0, 30.0));
        assert_eq!(rest, r(10.0, 50.0, 100.0, 50.0));
        // No gap, no overlap.
        assert_eq!(top.y + top.h, rest.y);
    }

    #[test]
    fn split_bottom_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (bottom, rest) = r0.split_bottom(30.0);
        assert_eq!(rest, r(10.0, 20.0, 100.0, 50.0));
        assert_eq!(bottom, r(10.0, 70.0, 100.0, 30.0));
    }

    #[test]
    fn split_left_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (left, rest) = r0.split_left(40.0);
        assert_eq!(left, r(10.0, 20.0, 40.0, 80.0));
        assert_eq!(rest, r(50.0, 20.0, 60.0, 80.0));
    }

    #[test]
    fn split_right_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (right, rest) = r0.split_right(40.0);
        assert_eq!(rest, r(10.0, 20.0, 60.0, 80.0));
        assert_eq!(right, r(70.0, 20.0, 40.0, 80.0));
    }

    #[test]
    fn viewport_into_rect() {
        let v = Viewport { width: 600.0, height: 400.0 };
        assert_eq!(v.into_rect(), r(0.0, 0.0, 600.0, 400.0));
    }

    // ── R1 port (bug_hunt_layout.rs): NaN / Infinity / negative edge cases.
    // The mirror this ported from called `shrink(top, right, bottom, left)` —
    // a stale 4-arg signature; the real `Rect::shrink` takes an `Inset`. That
    // drift is exactly the mirror-decay case finding R1 describes.

    #[test]
    fn split_top_nan_height_is_absorbed_by_min_max_clamp() {
        // f64::min/max return the non-NaN argument, so NaN behaves like "no limit".
        let r0 = r(0.0, 0.0, 100.0, 80.0);
        let (strip, rest) = r0.split_top(f64::NAN);
        assert_eq!(strip.h, 80.0, "NaN height clamps to self.h via min/max");
        assert_eq!(rest.h, 0.0);
    }

    #[test]
    fn split_left_nan_width_is_absorbed_by_min_max_clamp() {
        let r0 = r(0.0, 0.0, 300.0, 200.0);
        let (strip, rest) = r0.split_left(f64::NAN);
        assert_eq!(strip.w, 300.0);
        assert_eq!(rest.w, 0.0);
        assert_eq!(rest.x, 300.0);
    }

    #[test]
    fn split_top_negative_height_clamps_to_zero() {
        let r0 = r(0.0, 0.0, 100.0, 80.0);
        let (strip, rest) = r0.split_top(-10.0);
        assert_eq!(strip.h, 0.0, "negative height must clamp to 0, not go negative");
        assert_eq!(rest, r0, "remainder must equal the original rect");
    }

    #[test]
    fn split_right_infinity_width_clamps_to_self() {
        let r0 = r(0.0, 0.0, 300.0, 200.0);
        let (strip, rest) = r0.split_right(f64::INFINITY);
        assert_eq!(strip.w, 300.0, "INFINITY width must clamp to self.w");
        assert_eq!(rest.w, 0.0);
    }

    #[test]
    fn shrink_nan_inset_fails_open_not_zero() {
        // `w <= 0.0 || h <= 0.0` is false for NaN (NaN comparisons are always
        // false), so a NaN inset does NOT collapse to Rect::ZERO — it silently
        // propagates NaN into the result instead. Pinning this "fails open"
        // shape, not asserting it is desirable, is the point of the test.
        let r0 = r(0.0, 0.0, 100.0, 100.0);
        let result = r0.shrink(Inset { top: f64::NAN, right: 0.0, bottom: 0.0, left: 0.0 });
        assert!(result.h.is_nan(), "NaN top inset must produce a NaN height, not 0");
        assert!(result.y.is_nan(), "NaN top inset must also poison y (self.y + inset.top)");
        assert_ne!(result, Rect::ZERO, "NaN never compares equal to ZERO — the guard fails open");
    }

    #[test]
    fn shrink_infinity_inset_collapses_to_zero() {
        let r0 = r(0.0, 0.0, 100.0, 100.0);
        let result = r0.shrink(Inset { top: 0.0, right: f64::INFINITY, bottom: 0.0, left: 0.0 });
        assert_eq!(result, Rect::ZERO, "an infinite inset must collapse to ZERO, not -Infinity dims");
    }

    #[test]
    fn shrink_negative_width_rect_is_already_zero_even_with_no_inset() {
        let r0 = r(0.0, 0.0, -50.0, 100.0);
        let result = r0.shrink(Inset::uniform(0.0));
        assert_eq!(result, Rect::ZERO, "a rect with negative width collapses to ZERO on shrink(0)");
    }

    #[test]
    fn rect_serde_round_trip() {
        let r0 = r(1.0, 2.0, 3.0, 4.0);
        let json = serde_json::to_string(&r0).unwrap();
        let r1: Rect = serde_json::from_str(&json).unwrap();
        assert_eq!(r0, r1);
    }

    #[test]
    fn axis1d_lerp_endpoints_and_midpoint() {
        let a = Axis1D { lo: 10.0, hi: 110.0 };
        assert_eq!(a.lerp(0.0), 10.0);
        assert_eq!(a.lerp(1.0), 110.0);
        assert_eq!(a.lerp(0.5), 60.0);
        // Byte-identical to the hand-rolled `lo + t*(hi - lo)` form.
        let (lo, hi) = (a.lo, a.hi);
        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0, 0.123] {
            assert_eq!(a.lerp(t).to_bits(), (lo + t * (hi - lo)).to_bits());
        }
    }

    #[test]
    fn axis1d_lerp_inverted_interval_moves_lo_to_hi() {
        // The y-axis range `(bottom, top)` and the colorbar `bar_bottom → bar_top`
        // are inverted (lo > hi): increasing `t` moves from lo toward hi.
        let a = Axis1D { lo: 400.0, hi: 0.0 };
        assert_eq!(a.lerp(0.0), 400.0);
        assert_eq!(a.lerp(1.0), 0.0);
        assert_eq!(a.lerp(0.25), 300.0);
        // Colorbar exact-formula equivalence: `bar_bottom - t*(bar_bottom - bar_top)`.
        let (bar_bottom, bar_top) = (400.0_f64, 0.0_f64);
        let a = Axis1D { lo: bar_bottom, hi: bar_top };
        for &t in &[0.0, 0.5, 1.0, 0.3] {
            let direct = bar_bottom - t * (bar_bottom - bar_top);
            assert_eq!(a.lerp(t).to_bits(), direct.to_bits());
        }
    }

    #[test]
    fn axis1d_uniform_center_matches_inline_slot_formula() {
        // `uniform_center` reproduces `origin + (i + 0.5)*slot` bit-for-bit using
        // the *precomputed* slot (the inline uniform-slot fallback's arithmetic).
        let (origin, h, n) = (37.5_f64, 562.3_f64, 8usize);
        let slot = h / n as f64;
        let a = Axis1D { lo: origin, hi: origin + h };
        for i in 0..n {
            let inline = origin + (i as f64 + 0.5) * slot;
            assert_eq!(a.uniform_center(i, slot).to_bits(), inline.to_bits());
        }
    }
}
