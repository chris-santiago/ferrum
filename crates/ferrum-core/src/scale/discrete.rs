//! The d3 discrete-scale layout model — one implementation, shared by the
//! compute facades and the render path.
//!
//! Before this module existed the band/point formulas lived twice: the compute
//! facades (`BandScale`/`PointScale`, user-query-only) carried a d3-shaped
//! model, while the render-side `OrdinalScale` carried a symmetric
//! `(i + 0.5)·step` model that ignored `padding_inner`, `padding_outer` and
//! `align` entirely. That is finding F-L04-03: those four parameters were
//! documented, validated and settable but moved no geometry. The facades'
//! formulas are what this module started from, **not** what it preserves: the
//! facade placed each band half an inner gap into its slot, which is neither
//! upstream d3 nor range-respecting (see the fidelity note), so unification
//! adopted upstream's placement and **changed the facade's own `scale()`
//! output** for `padding_inner > 0`. Both sides now call this module, so the
//! two models cannot drift again.
//!
//! # The model
//!
//! For `n` categories over a signed pixel extent `extent = range_hi − range_lo`:
//!
//! ```text
//! band:  denom = max(1, n − padding_inner + 2·padding_outer)
//!        step  = extent / denom
//!        bandwidth = |step| · (1 − padding_inner)
//!        start = range_lo + padding_outer·step + align·leftover
//! point: denom = (n − 1) + 2·padding
//!        step  = extent / denom
//!        start = range_lo + padding·step + align·leftover
//! ```
//!
//! A category's *lead* is `start + i·step` under both models — d3's `band(x)`.
//! Its *position*, the pixel a mark is drawn at, is the middle of the drawn
//! band: `lead + step·(1 − padding_inner)/2` for the band model, and the lead
//! itself for the point model, whose positions are zero-width by construction.
//! The inner gap follows each band rather than straddling it, so the `n` bands
//! plus their gaps span exactly `step·(n − padding_inner)` from `start` and
//! **never leave the range** — with `padding_outer = 0` the last band's
//! trailing edge lands on `range_hi` exactly. All three parameters reach the
//! positions through `step` and `start`: `padding_inner` and `padding_outer`
//! both enter the denominator (so both rescale the step), `padding_inner` also
//! narrows the band inside its slot, `padding_outer` insets `start`, and
//! `align` moves `start` only when the denominator clamp leaves a leftover.
//!
//! # Why the zero-padding reduction is bit-exact
//!
//! `padding_inner = 0, padding_outer = 0, align = 0.5` must reproduce the
//! former render model's `range_lo + step/2 + i·step` **in bytes**, not merely
//! in exact arithmetic — every default (auto-inferred) ordinal axis in the
//! golden corpus depends on it. Two details carry that guarantee:
//!
//! - [`clamped_step`] returns the `align` leftover in closed form instead of as
//!   the cancelling difference `extent − denom·step`, which is exactly zero
//!   only in exact arithmetic (`n·(extent/n)` can miss `extent` by an ulp) and
//!   would otherwise leak an `align`-scaled residue into every band center.
//! - [`DiscreteGeometry`] caches `first` as `start + step·(1 − padding_inner)/2.0`
//!   and evaluates `position(i)` as `first + i·step`, preserving the former
//!   model's exact association. At `padding_inner = 0` the cached term is
//!   `step * 1.0 / 2.0`, bit-identical to the old `step/2.0` (multiplying by
//!   `1.0` is exact); `(a + h) + i·s` and `(a + i·s) + h` are not the same
//!   double, so the association matters as much as the value.
//!
//! `crate::scale::discrete::tests::unpadded_band_reduces_to_symmetric_model`
//! pins the reduction against the old expression with `assert_eq!`.
//!
//! # Fidelity to upstream d3
//!
//! Placement follows upstream `scaleBand`/`scalePoint`: `band(i) = start + i·step`
//! with the inner gap after each band. The compute facade's pre-unification
//! math had a `+ step·padding_inner/2` inset (bands centered in their slots),
//! which pushed the last band past `range_hi` whenever
//! `padding_inner > 2·padding_outer` — far enough, at
//! `padding_inner = 0.4, padding_outer = 0` on a 600px canvas, to render a bar
//! off the canvas entirely. The first revision of this module ported that
//! inset along with the rest and turned it into rendered geometry; it is now
//! gone from **both** sides, and `band_never_escapes_the_range` pins its
//! absence. `BandScale.scale()` moves accordingly for a padded scale — a
//! disclosed, corrective change to a public compute API.
//!
//! Two documented divergences from upstream remain, both narrow:
//!
//! - Upstream folds outer padding into the `align` term
//!   (`start += (extent − step·(n − padding_inner))·align`, i.e.
//!   `start = range_lo + 2·padding_outer·align·step`); this model applies
//!   `padding_outer·step` directly and lets `align` distribute only the
//!   clamp's leftover. The two agree whenever `align = 0.5` **or**
//!   `padding_outer = 0`, and differ only for a non-default `align` combined
//!   with outer padding.
//! - Upstream normalizes an inverted range (it lays bands out over the sorted
//!   range and reverses the resulting array), so its `band(x)` is always the
//!   low-coordinate edge. This model keeps the range's sign, so `step` is
//!   negative and `lead(i)` is the leading edge *in domain order* — the
//!   high-coordinate edge of the band. Positions agree; the leads are the two
//!   ends of the same band. The `.max(0.0)` on the leftover then suppresses
//!   `align` for a negative extent under a clamping denominator (the facade's
//!   behavior, pinned by `inverted_range_under_clamp_suppresses_align`).
//!
//! The batch's docs task carries the dated reconciliation note in
//! `ferrum-spec.md`.

use std::borrow::Cow;

/// d3's default `align` for band and point scales: leftover pixels (when the
/// denominator clamp creates any) are split evenly before and after.
pub(crate) const DEFAULT_ALIGN: f64 = 0.5;

/// Which d3 model a discrete scale's pixel geometry follows.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DiscreteModel {
    /// d3 `scaleBand`: bands of non-zero width, `padding_inner` between them
    /// and `padding_outer` before the first and after the last.
    Band { padding_inner: f64, padding_outer: f64 },
    /// d3 `scalePoint`: zero-width positions with `padding` at both ends.
    Point { padding: f64 },
}

/// The layout parameters of a discrete positional scale: which d3 model it
/// follows plus its `align`.
///
/// Constructed at the wire boundary (`ScaleSpec::Band`/`Point`/`Ordinal`) and
/// by the compute facades; consumed via [`DiscreteLayout::geometry`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DiscreteLayout {
    model: DiscreteModel,
    align: f64,
}

impl DiscreteLayout {
    /// The zero-padding band model: `padding_inner = 0`, `padding_outer = 0`,
    /// `align = 0.5`. Every auto-inferred ordinal axis uses it, and it reduces
    /// to the pre-d3 symmetric model bit-exactly (see the module docs).
    pub(crate) const UNPADDED: DiscreteLayout = DiscreteLayout {
        model: DiscreteModel::Band { padding_inner: 0.0, padding_outer: 0.0 },
        align: DEFAULT_ALIGN,
    };

    /// d3 `scaleBand` parameters.
    pub(crate) fn band(padding_inner: f64, padding_outer: f64, align: f64) -> Self {
        DiscreteLayout { model: DiscreteModel::Band { padding_inner, padding_outer }, align }
    }

    /// d3 `scalePoint` parameters. `padding` is an end padding — the point
    /// model has no inner gap, its positions having no width.
    pub(crate) fn point(padding: f64, align: f64) -> Self {
        DiscreteLayout { model: DiscreteModel::Point { padding }, align }
    }

    /// The band model's `padding_inner`, for the facades that report it back
    /// to Python (`BandScale.padding_inner`, `OrdinalScale.padding`). Both
    /// callers are band-model by construction; a point scale has no inner
    /// padding and reports `0.0`.
    pub(crate) fn padding_inner(&self) -> f64 {
        match self.model {
            DiscreteModel::Band { padding_inner, .. } => padding_inner,
            DiscreteModel::Point { .. } => 0.0,
        }
    }

    /// Resolve pixel geometry for `n` categories over `[range_lo, range_hi]`.
    ///
    /// `range_hi < range_lo` (an inverted explicit range) is legal and yields a
    /// negative `step`, placing categories in descending order; the reported
    /// bandwidth stays non-negative (GH #69).
    pub(crate) fn geometry(&self, n: usize, range_lo: f64, range_hi: f64) -> DiscreteGeometry {
        let extent = range_hi - range_lo;
        if n == 0 {
            return DiscreteGeometry::empty(range_lo);
        }
        match self.model {
            DiscreteModel::Band { padding_inner, padding_outer } => {
                let denom_raw = n as f64 - padding_inner + padding_outer * 2.0;
                let (step, leftover) = clamped_step(denom_raw, extent);
                let start = range_lo + padding_outer * step + self.align * leftover;
                DiscreteGeometry {
                    step,
                    bandwidth: (step * (1.0 - padding_inner)).abs(),
                    start,
                    // The middle of the drawn band, not of the slot: the inner
                    // gap follows the band (upstream `scaleBand`). At
                    // `padding_inner = 0` this is `step * 1.0 / 2.0`,
                    // bit-identical to the pre-F-L04-03 `step / 2.0`.
                    first: start + step * (1.0 - padding_inner) / 2.0,
                }
            }
            DiscreteModel::Point { padding } => {
                if n == 1 {
                    // A lone point sits at the range midpoint: `padding` and
                    // `align` have nothing to distribute, and the general
                    // formula would divide by `denom = 2·padding` (zero at
                    // `padding = 0`). Written as `range_lo + extent/2` so a
                    // one-category point scale and a one-category band scale
                    // agree bit-for-bit.
                    let mid = range_lo + extent / 2.0;
                    return DiscreteGeometry {
                        step: extent,
                        bandwidth: extent.abs(),
                        start: mid,
                        first: mid,
                    };
                }
                // `n >= 2` keeps `denom >= 1`, so the band model's clamp cannot
                // fire here and the leftover is always exactly zero: `align` is
                // algebraically inert on a point scale in this model. (Upstream
                // d3 routes the end padding through `align`, where a non-default
                // value does shift the points; the two agree at `align = 0.5`.)
                let denom = (n as f64 - 1.0) + padding * 2.0;
                let (step, leftover) = clamped_step(denom, extent);
                let start = range_lo + padding * step + self.align * leftover;
                DiscreteGeometry {
                    step,
                    // A point has no drawn band (d3 reports `bandwidth() == 0`),
                    // but ferrum's render consumers size dodge sub-bands, jitter
                    // spread and mark widths against this value, so a point scale
                    // reports its slot — d3's `point.step()`. Reporting zero would
                    // collapse every mark drawn against an explicit `PointScale`.
                    bandwidth: step.abs(),
                    start,
                    first: start,
                }
            }
        }
    }
}

/// How a categorical axis learns the pixels its categories are drawn at
/// (F-L04-03, GH #67).
///
/// A discrete scale's centers are always [`DiscreteGeometry::position`] — one
/// model, shared with the marks. What differs is *which pixel interval* that
/// geometry resolves over, and that is decided by where the scale's range came
/// from:
///
/// - **[`Absolute`](Self::Absolute)** — the user supplied `range=`. That range
///   is chart-absolute by design (#39 phase 2), so the centers are already
///   resolved when the scale is built and travel to layout as pixels.
/// - **[`PanelExtent`](Self::PanelExtent)** — the range is the plot area, which
///   is not known when the axis input is built: the provisional scale pass
///   resolves against `[0, 1]` (`prepare::prepare_render_inputs`) precisely
///   because layout has not run yet. So the *model* travels instead, and the
///   axis resolves it against the panel rect. That rect is bit-for-bit the
///   interval `scene_build::resolve_panel_scales` hands the final mark scale
///   (`(plot_area.x, plot_area.x + plot_area.w)`), so labels land on mark
///   centers exactly, not approximately.
///
/// This enum *is* the collapse of #67's centers gate. Before it, only the first
/// case had a carrier; the second fell back to layout's `(i + 0.5)·slot`, a
/// symmetric model blind to `padding_inner`/`padding_outer`/`align` — so a
/// padded no-range `BandScale` drew its bars from this module and its tick
/// labels from a different model, and the two disagreed by pixels (~4.8px at
/// `padding = 0.1` over four categories on a 600px canvas).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CategoricalPlacement {
    /// Chart-absolute band centers, one per domain category in order.
    Absolute(Vec<f64>),
    /// The scale's d3 model plus its domain size, to be resolved against the
    /// panel's pixel interval at layout time.
    PanelExtent { layout: DiscreteLayout, categories: usize },
}

impl CategoricalPlacement {
    /// The absolute center pixel of every category, in domain order, over the
    /// axis's pixel interval `[lo, hi]`.
    ///
    /// `lo`/`hi` are ignored by [`Absolute`](Self::Absolute), whose pixels are
    /// already chart-absolute — borrowed rather than rebuilt, so an
    /// explicit-range axis allocates nothing here.
    pub(crate) fn centers(&self, lo: f64, hi: f64) -> Cow<'_, [f64]> {
        match self {
            CategoricalPlacement::Absolute(centers) => Cow::Borrowed(centers),
            CategoricalPlacement::PanelExtent { layout, categories } => {
                let g = layout.geometry(*categories, lo, hi);
                Cow::Owned((0..*categories).map(|i| g.position(i)).collect())
            }
        }
    }

    /// How many centers [`centers`](Self::centers) yields — the scale's domain
    /// size, known without a pixel interval. Lets the placement/label pairing
    /// invariant be checked before any geometry is resolved
    /// (`AxisInput::debug_assert_placement_invariants`).
    pub(crate) fn len(&self) -> usize {
        match self {
            CategoricalPlacement::Absolute(centers) => centers.len(),
            CategoricalPlacement::PanelExtent { categories, .. } => *categories,
        }
    }
}

/// d3's denominator clamp, and the leftover that clamp creates.
///
/// `step = extent / max(1, denom_raw)`; the leftover `align` distributes is
/// `extent − denom_raw·step`, clamped at zero (a negative extent distributes
/// nothing, matching the facade's `.max(0.0)`).
///
/// The leftover is returned in **closed form** rather than as that difference:
/// when the clamp does not fire, `step` is exactly `extent/denom_raw` and the
/// difference is zero in exact arithmetic — but `denom·(extent/denom)` can miss
/// `extent` by an ulp, and an `align`-scaled ulp on every band center would
/// break the byte-identity of the default (zero-padding) render path. When the
/// clamp does fire, `step` is `extent` and the leftover is `extent·(1 − denom_raw)`.
fn clamped_step(denom_raw: f64, extent: f64) -> (f64, f64) {
    if denom_raw >= 1.0 {
        (extent / denom_raw, 0.0)
    } else {
        (extent, (extent * (1.0 - denom_raw)).max(0.0))
    }
}

/// Resolved pixel geometry of a discrete scale over a concrete range.
///
/// Every field is derived once in [`DiscreteLayout::geometry`] so that the
/// accessors evaluate in a fixed association (see the module docs on
/// bit-exactness).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DiscreteGeometry {
    /// Signed pixel distance between consecutive categories.
    step: f64,
    /// Width a category is drawn at: `|step|·(1 − padding_inner)` for the band
    /// model, `|step|` for the point model (see the `Point` arm's comment).
    /// Always non-negative, including for an inverted range.
    bandwidth: f64,
    /// Leading edge of category 0's band, after outer padding and `align`.
    start: f64,
    /// Pixel of category 0 — the middle of its drawn band (band model) or the
    /// point itself (point model).
    first: f64,
}

impl DiscreteGeometry {
    /// Geometry of an empty domain: no step, no width, nothing to place.
    fn empty(range_lo: f64) -> Self {
        DiscreteGeometry { step: 0.0, bandwidth: 0.0, start: range_lo, first: range_lo }
    }

    /// The pixel category `idx` is drawn at: the middle of its drawn band
    /// under the band model, its point under the point model.
    pub(crate) fn position(&self, idx: usize) -> f64 {
        self.first + idx as f64 * self.step
    }

    /// The leading edge of category `idx`'s drawn band — d3's `band(x)`, and
    /// what the `BandScale`/`PointScale` compute facades return from
    /// `scale()`. "Leading" is in domain order: for an inverted range (negative
    /// `step`) this is the band's high-coordinate edge, and the band runs back
    /// towards `lead + step·(1 − padding_inner)`. Equals
    /// [`position`](Self::position) under the point model, whose positions have
    /// no width.
    pub(crate) fn lead(&self, idx: usize) -> f64 {
        self.start + idx as f64 * self.step
    }

    /// The drawn width of one category. Non-negative.
    pub(crate) fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Signed pixel distance between consecutive categories.
    pub(crate) fn step(&self) -> f64 {
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE byte-identity proof for the whole batch-C geometry change: with
    /// `padding_inner = padding_outer = 0` and `align = 0.5`, the d3 model must
    /// reproduce the former render model — `first_center = range_lo + step/2`,
    /// `center_i = first_center + i·step`, `bandwidth = |step|` — with *bit*
    /// equality, over ranges and category counts chosen so `n·(extent/n)`
    /// misses `extent` where it can. `assert_eq!` on f64 is deliberate here:
    /// a tolerance would not prove what the golden corpus depends on.
    #[test]
    fn unpadded_band_reduces_to_symmetric_model() {
        let ranges = [
            (0.0, 300.0),
            (40.0, 260.0),
            (260.0, 40.0),
            (0.1, 0.3),
            (37.3, 611.7),
            (-1e300, 1e300),
            (150.0, 150.0),
            (12.345_678_9, 987.654_321),
        ];
        for (lo, hi) in ranges {
            for n in 1..=17usize {
                let g = DiscreteLayout::UNPADDED.geometry(n, lo, hi);
                // The pre-d3 render model, verbatim (scale/ordinal.rs).
                let step = (hi - lo) / n as f64;
                let first_center = lo + step / 2.0;
                assert_eq!(g.step(), step, "step for n={n} over [{lo}, {hi}]");
                assert_eq!(
                    g.bandwidth(),
                    step.abs(),
                    "bandwidth for n={n} over [{lo}, {hi}]"
                );
                for i in 0..n {
                    assert_eq!(
                        g.position(i),
                        first_center + i as f64 * step,
                        "center {i} of {n} over [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    /// The d3 band oracle (hand-computed): n=4 over [0, 400] with
    /// `padding_inner = 0.2`, `padding_outer = 0.1`, `align = 0.5`.
    /// denom = 4 − 0.2 + 0.2 = 4.0 → step = 100; bandwidth = 100·0.8 = 80;
    /// start = 0 + 0.1·100 = 10 → leads 10/110/210/310, band middles
    /// 50/150/250/350, and the last band's trailing edge at 310 + 80 = 390,
    /// a full `padding_outer·step` inside `range_hi`.
    #[test]
    fn band_oracle_padded() {
        let g = DiscreteLayout::band(0.2, 0.1, 0.5).geometry(4, 0.0, 400.0);
        assert_eq!(g.step(), 100.0);
        assert_eq!(g.bandwidth(), 80.0);
        assert_eq!(
            (0..4).map(|i| g.lead(i)).collect::<Vec<_>>(),
            vec![10.0, 110.0, 210.0, 310.0]
        );
        assert_eq!(
            (0..4).map(|i| g.position(i)).collect::<Vec<_>>(),
            vec![50.0, 150.0, 250.0, 350.0]
        );
        // The position is the middle of the DRAWN band, not of the slot: the
        // inner gap follows the band.
        for i in 0..4 {
            assert_eq!(g.lead(i) + g.bandwidth() / 2.0, g.position(i));
        }
        assert_eq!(g.lead(3) + g.bandwidth(), 390.0, "last band stays inside [0, 400]");
    }

    /// Regression pin for the placement bug the first revision of this module
    /// shipped: it carried the compute facade's `+ step·padding_inner/2` inset,
    /// which pushed the whole band block half an inner gap along, so the last
    /// band's trailing edge landed at `range_hi + step·(padding_inner/2 −
    /// padding_outer)` — outside the scale range for any
    /// `padding_inner > 2·padding_outer`, and (at these parameters, on a 600px
    /// canvas) far enough to render a bar off the canvas entirely.
    ///
    /// The band block spans exactly `step·(n − padding_inner)` from `start`,
    /// so no legal parameter combination can escape. Swept rather than
    /// spot-checked, since the old bug was invisible at the default
    /// `padding_inner == padding_outer` — and swept over **both range
    /// directions**, since the sign is what decides which edge of a band is
    /// its lead and where the block grows from. Every band's two edges are
    /// checked against the sorted range bounds, not just the first and last
    /// band's, so a sign error cannot hide in the middle of the domain.
    #[test]
    fn band_never_escapes_the_range() {
        for (lo, hi) in [
            (0.0, 400.0),
            (40.0, 260.0),
            (59.775, 584.0),
            // Descending (an inverted explicit `range=[hi, lo]`).
            (400.0, 0.0),
            (260.0, 40.0),
            (584.0, 59.775),
        ] {
            let (low, high) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            for pi in [0.0, 0.1, 0.4, 0.9] {
                for po in [0.0, 0.05, 0.1, 0.5] {
                    for align in [0.0, 0.5, 1.0] {
                        for n in [1usize, 2, 4, 7] {
                            let g = DiscreteLayout::band(pi, po, align).geometry(n, lo, hi);
                            let ctx = format!(
                                "[{lo}, {hi}] n={n} pi={pi} po={po} align={align}"
                            );
                            for i in 0..n {
                                let lead = g.lead(i);
                                let trailing = lead + g.step() * (1.0 - pi);
                                // A 1e-9 slack absorbs the float residue of
                                // `start + i·step + step·(1 − pi)` reassociating
                                // `extent`; the mathematical bound is exact.
                                for (name, edge) in [("lead", lead), ("trailing", trailing)] {
                                    assert!(
                                        edge >= low - 1e-9 && edge <= high + 1e-9,
                                        "band {i}'s {name} edge escapes [{low}, {high}] ({ctx}): {edge}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// `padding_inner` and `padding_outer` are not the same knob: inner
    /// padding is the only one that narrows a band below its slot, outer
    /// padding is the only one that insets the first center away from the
    /// range edge by more than half a slot.
    #[test]
    fn inner_and_outer_padding_have_distinct_effects() {
        let plain = DiscreteLayout::band(0.0, 0.0, 0.5).geometry(4, 0.0, 400.0);
        let inner = DiscreteLayout::band(0.5, 0.0, 0.5).geometry(4, 0.0, 400.0);
        let outer = DiscreteLayout::band(0.0, 0.5, 0.5).geometry(4, 0.0, 400.0);

        assert_eq!(plain.bandwidth(), plain.step().abs(), "no inner padding → full slot");
        assert_eq!(
            inner.bandwidth(),
            inner.step().abs() / 2.0,
            "inner padding halves the drawn band inside its slot"
        );
        assert_eq!(outer.bandwidth(), outer.step().abs(), "outer padding leaves bands full-slot");

        // Both parameters enter the denominator, so both rescale the step; only
        // outer padding also insets `start`, pushing the first center past the
        // half-slot offset a zero-padding scale has.
        // A category sits at the middle of its own drawn band, so a narrower
        // band pulls the first position back towards `range_lo`; only outer
        // padding pushes it past a half-band.
        assert_eq!(plain.position(0), plain.bandwidth() / 2.0);
        assert_eq!(inner.position(0), inner.bandwidth() / 2.0);
        assert!(
            outer.position(0) > outer.bandwidth() / 2.0,
            "outer padding must inset the first band: {} vs half-band {}",
            outer.position(0),
            outer.bandwidth() / 2.0
        );
    }

    /// The denominator clamp is the only source of `align` leftover: n=1,
    /// `padding_inner = 0.5`, `padding_outer = 0` over [0, 100] gives
    /// denom_raw = 0.5 → step = 100 (clamped), leftover = 50. `align = 0`
    /// leaves the band against the low edge, `align = 1` pushes it by the full
    /// leftover — the band is 50 wide, so it lands flush against `range_hi`.
    #[test]
    fn align_moves_only_the_clamped_leftover() {
        let at = |align: f64| DiscreteLayout::band(0.5, 0.0, align).geometry(1, 0.0, 100.0);
        assert_eq!(at(0.0).step(), 100.0, "denominator must clamp to 1");
        assert_eq!(at(0.0).bandwidth(), 50.0);
        assert_eq!(at(0.0).lead(0), 0.0);
        assert_eq!(at(0.5).lead(0), 25.0);
        assert_eq!(at(1.0).lead(0), 50.0);
        assert_eq!(at(1.0).lead(0) + at(1.0).bandwidth(), 100.0, "align=1 sits flush at range_hi");
        // Without a clamp there is no leftover, so align is inert.
        let unclamped = |align: f64| DiscreteLayout::band(0.1, 0.0, align).geometry(4, 0.0, 400.0);
        assert_eq!(unclamped(0.0).position(0), unclamped(1.0).position(0));
    }

    /// The `.max(0.0)` on the leftover: for an INVERTED range the clamped
    /// leftover is negative, and this model suppresses it rather than
    /// distributing it, so `align` moves nothing there. Ported from the compute
    /// facade's own `.max(0.0)` and kept deliberately (a negative extent has no
    /// slack to distribute in the direction `align` names); upstream d3 never
    /// meets the case because it normalizes the range before laying bands out.
    /// Nothing else pins this: `align_moves_only_the_clamped_leftover` is
    /// ascending, and the facade oracle's inverted range is paired only with
    /// non-clamping padding trios.
    #[test]
    fn inverted_range_under_clamp_suppresses_align() {
        let at = |align: f64| DiscreteLayout::band(0.5, 0.0, align).geometry(1, 100.0, 0.0);
        assert_eq!(at(0.0).step(), -100.0, "denominator must clamp to 1, step stays signed");
        assert_eq!(at(0.0).bandwidth(), 50.0);
        assert_eq!(at(0.0).lead(0), 100.0);
        assert_eq!(at(1.0).lead(0), 100.0, "no leftover to distribute for a negative extent");
        assert_eq!(at(0.0).position(0), at(1.0).position(0));
        // The ascending twin of the same parameters DOES move — this is a
        // property of the sign, not of the parameters.
        let ascending = |align: f64| DiscreteLayout::band(0.5, 0.0, align).geometry(1, 0.0, 100.0);
        assert_ne!(ascending(0.0).lead(0), ascending(1.0).lead(0));
    }

    /// The point oracle: n=3 over [0, 300] with the d3 default `padding = 0.5`
    /// gives step = 300/(2 + 1) = 100 and positions 50/150/250 — identical to
    /// the zero-padding band model's centers, bit-for-bit. This coincidence is
    /// why adopting the point formula moves no default `PointScale` output.
    #[test]
    fn point_default_padding_equals_unpadded_band_centers() {
        let point = DiscreteLayout::point(0.5, DEFAULT_ALIGN).geometry(3, 0.0, 300.0);
        let band = DiscreteLayout::UNPADDED.geometry(3, 0.0, 300.0);
        for i in 0..3 {
            assert_eq!(point.position(i), band.position(i), "position {i}");
        }
        assert_eq!(point.step(), band.step());
        assert_eq!(point.bandwidth(), band.bandwidth());
    }

    /// Point padding is real: `padding = 0` puts the first and last categories
    /// on the range endpoints (d3's defining property of a point scale).
    #[test]
    fn point_zero_padding_lands_on_endpoints() {
        let g = DiscreteLayout::point(0.0, DEFAULT_ALIGN).geometry(3, 0.0, 300.0);
        assert_eq!(g.position(0), 0.0);
        assert_eq!(g.position(1), 150.0);
        assert_eq!(g.position(2), 300.0);
        // Lead == position: a point has no band to inset into.
        assert_eq!(g.lead(2), g.position(2));
    }

    /// A one-category point scale sits at the range midpoint for every padding
    /// and align, and agrees with the one-category band scale bit-for-bit.
    #[test]
    fn single_category_point_matches_band_midpoint() {
        let band = DiscreteLayout::UNPADDED.geometry(1, 40.0, 260.0);
        for padding in [0.0, 0.5, 10.0] {
            for align in [0.0, 0.5, 1.0] {
                let point = DiscreteLayout::point(padding, align).geometry(1, 40.0, 260.0);
                assert_eq!(point.position(0), 150.0, "padding={padding} align={align}");
                assert_eq!(point.position(0), band.position(0));
            }
        }
    }

    /// An empty domain yields no geometry at all — no `extent/0` infinity can
    /// escape into a position or a width, under either model.
    #[test]
    fn empty_domain_geometry_is_inert() {
        for layout in [
            DiscreteLayout::UNPADDED,
            DiscreteLayout::band(0.3, 0.2, 0.5),
            DiscreteLayout::point(0.5, 0.5),
        ] {
            let g = layout.geometry(0, 40.0, 260.0);
            assert_eq!(g.step(), 0.0);
            assert_eq!(g.bandwidth(), 0.0);
            assert_eq!(g.position(0), 40.0);
        }
    }

    /// An inverted range places categories in descending order (negative step)
    /// but never reports a negative width (GH #69).
    #[test]
    fn inverted_range_descends_with_non_negative_bandwidth() {
        let g = DiscreteLayout::band(0.2, 0.0, 0.5).geometry(4, 260.0, 40.0);
        assert!(g.step() < 0.0, "step must be signed: {}", g.step());
        assert!(g.bandwidth() > 0.0, "bandwidth must be |step|·(1−pi): {}", g.bandwidth());
        assert!(g.position(0) > g.position(3), "categories must descend");
        let point = DiscreteLayout::point(0.5, 0.5).geometry(4, 260.0, 40.0);
        assert!(point.bandwidth() > 0.0);
        assert!(point.position(0) > point.position(3));
    }

    /// `padding_inner()` — the one parameter read back out of a layout, for
    /// the `OrdinalScale.padding` getter and its wire form — answers per
    /// model, and reports zero for the point model, which has no inner gap.
    #[test]
    fn padding_inner_reports_per_model() {
        assert_eq!(DiscreteLayout::band(0.3, 0.2, 0.25).padding_inner(), 0.3);
        assert_eq!(DiscreteLayout::point(0.4, 0.5).padding_inner(), 0.0);
        assert_eq!(DiscreteLayout::UNPADDED.padding_inner(), 0.0);
    }
}
