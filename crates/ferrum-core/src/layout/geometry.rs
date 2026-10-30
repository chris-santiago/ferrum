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

/// The minimum ink clearance a `mark_text` glyph on a continuous positional
/// scale keeps from the plot frame, at default padding. Folded
/// into each row's own [`EdgeNeed`] by
/// `render::scale_resolve::domain::text_edge_rows`.
pub(crate) const TEXT_INSET_FLOOR_PX: f64 = 4.0;

/// The share of a continuous axis's inner (unpadded) length
/// [`inset_pixel_range`] keeps even when its `Mark::Text` rows cannot all
/// clear the frame by [`TEXT_INSET_FLOOR_PX`] (labels wider than the panel
/// can hold) — the same floor `layout::mod`'s `AUTO_PADDING_MIN_PLOT_SHARE`
/// keeps for the auto-padding settle.
pub(crate) const TEXT_INSET_MIN_SPAN_SHARE: f64 = 0.5;

/// One `Mark::Text` row's ink requirement on a continuous axis, in the same
/// per-tick domain-fraction convention `TickProjection.major` uses (`frac`
/// is `0` at the resolved range's own start pixel, `1` at its end — see
/// [`ScaleKind::value_fractions`](crate::render::scale_resolve::ScaleKind::value_fractions)).
/// `start`/`end` are how far the row's ink needs the range's start/end pixel
/// to be from it, with [`TEXT_INSET_FLOOR_PX`] and the mark's `dx`/`dy`
/// already folded in; either may be negative (the ink is shifted away from
/// that side, so it poses no constraint there). Built by
/// `ScaleKind::text_edge_demand` from the scale-independent rows
/// `render::scale_resolve::domain::text_edge_rows` measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EdgeNeed {
    pub(crate) frac: f64,
    pub(crate) start: f64,
    pub(crate) end: f64,
    /// Whether the scene currently resolving this axis is the one that
    /// draws this row's mark: true for every row the measuring chart or
    /// composite leaf's own layers contribute, false for a sibling leaf's
    /// row carried along only because the two share this axis's domain
    /// (`render::composite::assign_union`). The inset solve
    /// ([`solve_text_edge_inset`]) reads every row regardless, so a
    /// sibling's wider label still widens the shared inset; only the
    /// overrun count ([`text_edge_overrun_rows`]) reads this flag, so a
    /// render's `TextInkPastFrame` warning counts each overrunning mark
    /// once, from the one axis that actually draws it, never once per
    /// member that merely shares its domain.
    pub(crate) own: bool,
}

/// Every [`EdgeNeed`] a continuous axis's `Mark::Text` layers place on it,
/// shared verbatim by [`inset_pixel_range`] (marks) and
/// [`crate::layout::axis::TickProjection`] (ticks) so the two can never
/// diverge. Empty is the byte-identical default: no `Mark::Text` layer
/// draws the channel, an Ordinal or degenerate scale, or non-default
/// padding.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TextEdgeDemand {
    pub(crate) rows: Vec<EdgeNeed>,
}

impl TextEdgeDemand {
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Keep only the rows no other row with `t' <= t` and `need' >= need`
/// already covers: as `t` increases the survivors' needs strictly
/// increase, so each can be the binding constraint for some inner length
/// — the reduction to non-dominated extremes plotly's autorange applies to
/// per-point pixel padding.
fn non_dominated(mut rows: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    rows.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut kept = Vec::with_capacity(rows.len());
    let mut max_need = f64::NEG_INFINITY;
    for (t, need) in rows {
        if need > max_need {
            kept.push((t, need));
            max_need = need;
        }
    }
    kept
}

/// The per-end inset `(a, c)` that satisfies every row's `start`/`end` need
/// at its own fraction, for an axis of inner span `span` with base pad `b`
/// (today's plain padding) reserved at both virtual ends `t=0`/`t=1`.
///
/// A row at fraction `t` sits at pixel `a + t*L` (`L = span - a - c`, the
/// resulting inner length) and needs `a + t*L >= start` and
/// `span - (a + t*L) >= end`. For a start row `i` and an end row `j` with
/// `t_j > t_i`, both hold at the LARGEST `L` exactly when
/// `L <= (span - start_i - end_j) / (t_j - t_i)`; the minimum of that bound
/// over every such pair is the largest `L` feasible for ALL rows (1-D
/// Helly), and `a` is then the smallest value satisfying every start row at
/// that `L`. Infeasible demand (labels wider than the panel can hold) is
/// clamped to keep at least [`TEXT_INSET_MIN_SPAN_SHARE`] of the plain inner
/// length `span - 2*b`.
fn solve_text_edge_inset(span: f64, base_pad: f64, rows: &[EdgeNeed]) -> (f64, f64) {
    let mut starts: Vec<(f64, f64)> = vec![(0.0, base_pad)];
    let mut ends: Vec<(f64, f64)> = vec![(1.0, base_pad)];
    for r in rows {
        if !r.frac.is_finite() {
            continue;
        }
        if r.start.is_finite() {
            starts.push((r.frac, r.start));
        }
        if r.end.is_finite() {
            ends.push((r.frac, r.end));
        }
    }
    let starts = non_dominated(starts);
    // Ends are non-dominated by the mirrored rule (`t' >= t`, `need' >=
    // need`): negate `t` first so the same ascending reduction applies,
    // then negate it back.
    let ends = non_dominated(ends.into_iter().map(|(t, need)| (-t, need)).collect())
        .into_iter()
        .map(|(t, need)| (-t, need))
        .collect::<Vec<_>>();

    let mut l_star = span;
    let mut binding: Option<(f64, f64)> = None;
    for &(ti, si) in &starts {
        for &(tj, ej) in &ends {
            if tj > ti {
                let candidate = (span - si - ej) / (tj - ti);
                if candidate.is_finite() && candidate < l_star {
                    l_star = candidate;
                    binding = Some((si, ej));
                }
            }
        }
    }
    let min_inner = TEXT_INSET_MIN_SPAN_SHARE * (span - 2.0 * base_pad).max(0.0);
    if l_star >= min_inner {
        let a = starts
            .iter()
            .map(|&(t, s)| s - t * l_star)
            .fold(f64::NEG_INFINITY, f64::max);
        return (a, span - a - l_star);
    }
    // Infeasible: the demand cannot be met without collapsing the inner
    // length past its floor. `base_pad` is reserved on BOTH ends first (the
    // same reservation the feasible branch's virtual (0, base_pad) /
    // (1, base_pad) rows encode), so this path can never inset an end below
    // today's plain pad; only the EXCESS above that reservation is split,
    // between the binding pair's own excess needs, rather than the per-row
    // max formula above, which would otherwise hand the whole oversized need
    // to one side.
    let total_pad = (span - min_inner).max(0.0);
    let (si, ej) = binding.unwrap_or((base_pad, base_pad));
    let excess_pad = (total_pad - 2.0 * base_pad).max(0.0);
    let excess_si = (si - base_pad).max(0.0);
    let excess_ej = (ej - base_pad).max(0.0);
    let denom = excess_si + excess_ej;
    let a = base_pad + if denom > 0.0 { excess_pad * excess_si / denom } else { excess_pad / 2.0 };
    (a, total_pad - a)
}

/// The indices (into `demand.rows`) of this axis's OWN rows
/// ([`EdgeNeed::own`]) that `inset_pixel_range`'s per-end solve, at a real
/// pixel span of `span`, still leaves unmet (the `TEXT_INSET_MIN_SPAN_SHARE`
/// clamp bound) — read by a per-render ledger that decides whether to warn,
/// never by the tick projection, which evaluates
/// `inset_pixel_range` alone. A sibling's row (`own: false`) still shapes
/// the solve (so its wider label still widens a shared inset) but is never
/// counted here, so a channel shared by several axes or composite leaves
/// reports each overrunning mark from the one axis that draws it.
pub(crate) fn text_edge_overrun_rows(span: f64, padding_frac: f64, demand: &TextEdgeDemand) -> Vec<usize> {
    if padding_frac == 0.0 || demand.is_empty() {
        return Vec::new();
    }
    if !span.is_finite() || span == 0.0 {
        return Vec::new();
    }
    let base_pad = (span * padding_frac).min(SCALE_PADDING_MAX_PX);
    let (a, c) = solve_text_edge_inset(span, base_pad, &demand.rows);
    let l = span - a - c;
    const EPS: f64 = 1e-6;
    demand
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.own && r.frac.is_finite())
        .filter(|(_, r)| {
            let pos = a + r.frac * l;
            (r.start.is_finite() && pos < r.start - EPS) || (r.end.is_finite() && (span - pos) < r.end - EPS)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Inset a pixel range by the resolved padding band (capped at
/// `SCALE_PADDING_MAX_PX`), widened per end to satisfy `demand`'s own
/// `Mark::Text` ink requirements (see [`solve_text_edge_inset`]).
/// Handles inverted ranges (the y-axis range is `(high_px, low_px)` for
/// quantitative scales). An empty `demand` returns exactly the plain
/// `min(span*frac, cap)` pair, so a chart with no `Mark::Text` layer on this
/// channel resolves the same inset it always has.
///
/// This is the single source of truth for the positional-scale padding inset.
/// `scale_resolve::positional` re-exports it so that data marks and
/// scale-projected axis ticks share *exactly* the same inset — a tick at value
/// `v` and a data mark at value `v` land on the same pixel.
pub(crate) fn inset_pixel_range(pr: (f64, f64), padding_frac: f64, demand: &TextEdgeDemand) -> (f64, f64) {
    if padding_frac == 0.0 {
        return pr;
    }
    let span = (pr.1 - pr.0).abs();
    let base_pad = (span * padding_frac).min(SCALE_PADDING_MAX_PX);
    let sign = if pr.1 >= pr.0 { 1.0 } else { -1.0 };
    if demand.is_empty() {
        return (pr.0 + sign * base_pad, pr.1 - sign * base_pad);
    }
    let (a, c) = solve_text_edge_inset(span, base_pad, &demand.rows);
    (pr.0 + sign * a, pr.1 - sign * c)
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
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

    /// The rectangle common to `self` and `other` — per side, the larger
    /// gutter of the two. Returns `Rect::ZERO` when they do not overlap.
    ///
    /// Used by the overlay shared-rect pre-pass (GH #89A,
    /// `render::composite_render`): folding a group's leaves through this
    /// yields the one plot region every leaf can lay out against without any
    /// leaf's marks spilling across another leaf's reserved legend or axis
    /// band.
    ///
    /// Two identical rects return the receiver unchanged rather than going
    /// through the edge arithmetic below, which reconstructs `w` as
    /// `(x + w) - x` and so may perturb it by an ulp. That keeps an overlay
    /// whose leaves all reserve the same gutters byte-identical to a
    /// pre-#89A render (design §7).
    pub(crate) fn intersect(&self, other: Rect) -> Rect {
        if *self == other {
            return *self;
        }
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let w = (self.x + self.w).min(other.x + other.w) - x;
        let h = (self.y + self.h).min(other.y + other.h) - y;
        if w <= 0.0 || h <= 0.0 {
            return Rect::ZERO;
        }
        Rect { x, y, w, h }
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
    /// Whether both dimensions are finite and positive: the only viewports
    /// a chart can be laid out in.
    pub fn is_drawable(&self) -> bool {
        self.width.is_finite() && self.height.is_finite() && self.width > 0.0 && self.height > 0.0
    }

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

    // ── inset_pixel_range's text-edge demand solve ──

    fn need(frac: f64, start: f64, end: f64) -> EdgeNeed {
        EdgeNeed { frac, start, end, own: true }
    }

    fn sibling_need(frac: f64, start: f64, end: f64) -> EdgeNeed {
        EdgeNeed { frac, start, end, own: false }
    }

    fn demand(rows: Vec<EdgeNeed>) -> TextEdgeDemand {
        TextEdgeDemand { rows }
    }

    #[test]
    fn inset_pixel_range_with_empty_demand_is_byte_identical_to_the_plain_pad() {
        let with_demand = inset_pixel_range((0.0, 400.0), 0.05, &TextEdgeDemand::default());
        assert_eq!(with_demand, (8.0, 392.0), "0.05 * 400 = 20, capped at SCALE_PADDING_MAX_PX = 8");
    }

    #[test]
    fn inset_pixel_range_endpoint_rows_set_the_inset_to_their_own_need() {
        // A row exactly at each end binds that end's inset to its own need
        // (above the plain 8px pad), matching the floor-only behavior at the
        // domain's own boundary.
        let d = demand(vec![need(0.0, 12.0, 0.0), need(1.0, 0.0, 15.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 400.0), 0.05, &d);
        assert_eq!(lo, 12.0);
        assert_eq!(hi, 400.0 - 15.0, "hi is the resulting pixel bound, not the raw end-inset amount");
    }

    #[test]
    fn inset_pixel_range_a_wide_label_just_inside_the_end_raises_that_end() {
        // A row at t=0.95 (not the exact end) with a large end need: its own
        // position (lerped across the drawn inner range) must still clear
        // the frame by its own 50px need, which requires widening the end
        // inset past the plain 8px pad.
        let d = demand(vec![need(0.95, 0.0, 50.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 400.0), 0.05, &d);
        assert_eq!(lo, 8.0, "no start need, so the plain pad stands");
        let pos = lo + 0.95 * (hi - lo);
        assert!(400.0 - pos >= 50.0 - 1e-9, "row's own clearance {} must reach its 50px need", 400.0 - pos);
        assert!(hi < 392.0, "the row's need must widen the end past the plain 8px pad");
    }

    #[test]
    fn inset_pixel_range_a_need_below_the_plain_pad_changes_nothing() {
        let d = demand(vec![need(0.0, 2.0, 0.0), need(1.0, 0.0, 3.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 400.0), 0.05, &d);
        assert_eq!((lo, hi), (8.0, 392.0));
    }

    #[test]
    fn inset_pixel_range_demand_applies_on_an_inverted_y_range_too() {
        // The y-axis range is `(bottom_px, top_px)`: `.0` still insets the
        // end `pr.0` names (the bottom, `frac=0`), `.1` the top (`frac=1`).
        let d = demand(vec![need(0.0, 15.0, 0.0)]);
        let (lo, hi) = inset_pixel_range((390.0, 10.0), 0.05, &d);
        assert_eq!(lo, 390.0 - 15.0, "the bottom need moves the bottom edge up by 15");
        assert_eq!(hi, 10.0 + 8.0, "the top end keeps the plain 8px pad");
    }

    #[test]
    fn inset_pixel_range_infeasible_demand_clamps_to_half_the_plain_inner_length() {
        // A need far wider than the panel can hold: the inner length clamps
        // to TEXT_INSET_MIN_SPAN_SHARE of the plain inner length rather than
        // collapsing or going negative.
        let d = demand(vec![need(0.0, 1000.0, 0.0), need(1.0, 0.0, 1000.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 100.0), 0.05, &d);
        let plain_inner = 100.0 - 2.0 * 5.0; // span*frac=5, under the 8px cap
        assert!(hi > lo, "lo {lo} hi {hi} must still order start before end");
        assert!((hi - lo) >= TEXT_INSET_MIN_SPAN_SHARE * plain_inner - 1e-9);
    }

    #[test]
    fn inset_pixel_range_explicit_zero_padding_ignores_demand() {
        // `padding_frac == 0.0` is the explicit-domain/explicit-padding early
        // return: the text floor never reaches an axis the user pinned.
        let d = demand(vec![need(0.0, 50.0, 0.0), need(1.0, 0.0, 50.0)]);
        assert_eq!(inset_pixel_range((0.0, 400.0), 0.0, &d), (0.0, 400.0));
    }

    #[test]
    fn inset_pixel_range_nan_need_drops_that_row_not_the_whole_demand() {
        let d = demand(vec![need(0.0, f64::NAN, 0.0), need(1.0, 0.0, 20.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 400.0), 0.05, &d);
        assert_eq!(lo, 8.0, "the NaN start row drops out, leaving the plain pad");
        assert_eq!(hi, 400.0 - 20.0, "hi is the resulting pixel bound, not the raw end-inset amount");
    }

    #[test]
    fn text_edge_overrun_rows_is_zero_for_a_feasible_demand() {
        let d = demand(vec![need(0.0, 12.0, 0.0), need(1.0, 0.0, 15.0)]);
        assert!(text_edge_overrun_rows(400.0, 0.05, &d).is_empty());
    }

    #[test]
    fn text_edge_overrun_rows_counts_rows_still_unmet_after_the_clamp() {
        let d = demand(vec![need(0.0, 1000.0, 0.0), need(1.0, 0.0, 1000.0)]);
        assert!(!text_edge_overrun_rows(100.0, 0.05, &d).is_empty());
    }

    #[test]
    fn text_edge_overrun_rows_is_zero_for_an_empty_demand() {
        assert!(text_edge_overrun_rows(400.0, 0.05, &TextEdgeDemand::default()).is_empty());
    }

    /// A sibling's row (`own: false`) still shapes the solve — it still
    /// needs to clear, so the inset widens for it — but is never counted
    /// as an overrun: only the measuring axis's own marks are.
    #[test]
    fn text_edge_overrun_rows_never_counts_a_sibling_row() {
        let d = demand(vec![sibling_need(0.0, 1000.0, 0.0), sibling_need(1.0, 0.0, 1000.0)]);
        assert!(text_edge_overrun_rows(100.0, 0.05, &d).is_empty(), "no own row, so nothing to count even though the demand is infeasible");
    }

    /// On the infeasible-clamp path, both ends keep at least the plain
    /// pad — only the EXCESS above it is split by the binding pair's own
    /// excess needs, never the raw needs (which could drive one end below
    /// the plain pad when the other's need is huge).
    #[test]
    fn inset_pixel_range_infeasible_split_never_insets_an_end_below_the_plain_pad() {
        let d = demand(vec![need(0.0, 5.0, 0.0), need(1.0, 0.0, 1000.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 100.0), 0.05, &d);
        let base_pad = 5.0; // span(100)*frac(0.05) = 5, under the 8px cap
        assert!(lo >= base_pad - 1e-9, "the start's own need ({base_pad}) must not fall below the plain pad, got lo={lo}");
        assert!(100.0 - hi > base_pad, "the end's huge need should still take the rest of the excess, got hi={hi}");
    }

    /// Symmetric to the start-below-pad case above: a huge start need beside
    /// a plain end leaves the end at least the plain pad too.
    #[test]
    fn inset_pixel_range_infeasible_split_never_insets_the_other_end_below_the_plain_pad() {
        let d = demand(vec![need(0.0, 1000.0, 0.0), need(1.0, 0.0, 5.0)]);
        let (lo, hi) = inset_pixel_range((0.0, 100.0), 0.05, &d);
        let base_pad = 5.0;
        assert!(100.0 - hi >= base_pad - 1e-9, "the end's own need ({base_pad}) must not fall below the plain pad, got hi={hi}");
        assert!(lo > base_pad, "the start's huge need should still take the rest of the excess, got lo={lo}");
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
    fn rect_intersect_takes_the_larger_gutter_per_side() {
        // Two overlay leaves' natural plot regions: the first reserves a
        // wider left gutter, the second a taller bottom gutter. The shared
        // rect must keep BOTH reservations (GH #89A).
        let wide_left = r(60.0, 10.0, 200.0, 180.0);
        let tall_bottom = r(40.0, 10.0, 220.0, 150.0);
        assert_eq!(
            wide_left.intersect(tall_bottom),
            r(60.0, 10.0, 200.0, 150.0)
        );
        // Commutative.
        assert_eq!(
            tall_bottom.intersect(wide_left),
            wide_left.intersect(tall_bottom)
        );
    }

    #[test]
    fn rect_intersect_disjoint_is_zero() {
        assert_eq!(r(0.0, 0.0, 10.0, 10.0).intersect(r(20.0, 0.0, 10.0, 10.0)), Rect::ZERO);
        // Touching edges share no area either.
        assert_eq!(r(0.0, 0.0, 10.0, 10.0).intersect(r(10.0, 0.0, 10.0, 10.0)), Rect::ZERO);
    }

    #[test]
    fn rect_intersect_with_self_is_bit_for_bit_identity() {
        // Byte-stability rail (GH #89A): a homogeneous overlay group folds
        // identical regions through `intersect`, and must come out with the
        // members' own rect unchanged — including coordinates whose
        // `(x + w) - w` reconstruction is not exact.
        for r0 in [
            r(12.5, 7.25, 300.0, 180.5),
            r(59.5439453125, 16.0, 262.052734375, 224.9609375),
            r(0.1, 0.2, 0.3, 0.7000000000000001),
        ] {
            let hit = r0.intersect(r0);
            assert_eq!(hit, r0);
            assert_eq!(hit.w.to_bits(), r0.w.to_bits());
            assert_eq!(hit.h.to_bits(), r0.h.to_bits());
        }
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
