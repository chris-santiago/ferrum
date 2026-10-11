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

use serde::{Deserialize, Serialize};

/// d3's default `align` for band and point scales: leftover pixels (when the
/// denominator clamp creates any) are split evenly before and after.
pub(crate) const DEFAULT_ALIGN: f64 = 0.5;

/// A point scale's end padding when the user sets none: half a step at each
/// end, which places `n` points where an unpadded band scale centers its `n`
/// bands. The resolver and the `PointScale` getter both read it; the wire
/// carries a point scale's padding only when the user set one.
pub(crate) const DEFAULT_POINT_PADDING: f64 = 0.5;

/// d3's bound on a band or point scale's `align`: a number in `[0, 1]`.
///
/// The one definition of the bound, shared by the `BandScale` and
/// `PointScale` constructors and by `validate_scale_values`. The error is the
/// refusal's clause alone (`must be a number in [0, 1]; received 1.5`); each
/// door words it around its own name.
pub(crate) fn check_align(align: f64) -> Result<(), String> {
    if align.is_finite() && (0.0..=1.0).contains(&align) {
        Ok(())
    } else {
        Err(format!("must be a number in [0, 1]; received {align}"))
    }
}

/// d3's bound on a point scale's end `padding`: a finite number `>= 0`.
/// Shared by the `PointScale` constructor and `validate_scale_values`; the
/// error is the clause alone, as for [`check_align`].
pub(crate) fn check_point_padding(padding: f64) -> Result<(), String> {
    if padding.is_finite() && padding >= 0.0 {
        Ok(())
    } else {
        Err(format!("must be a finite number >= 0; received {padding}"))
    }
}

/// The bound on an ordinal scale's inner `padding`: a number in `[0, 1]`.
/// Shared by the `OrdinalScale` constructor and `validate_scale_values`; the
/// error is the clause alone, as for [`check_align`].
pub(crate) fn check_ordinal_padding(padding: f64) -> Result<(), String> {
    if padding.is_finite() && (0.0..=1.0).contains(&padding) {
        Ok(())
    } else {
        Err(format!("must be in [0, 1]; got {padding}"))
    }
}

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

    /// The `align` this layout distributes leftover space by.
    pub(crate) fn align(&self) -> f64 {
        self.align
    }

    /// The band model's `padding_inner`, for the facade that reports it back
    /// to Python (`OrdinalScale.padding`), which is band-model by
    /// construction; a point scale has no inner padding and reports `0.0`.
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

/// One of the three band-padding arguments, as a [`BandPaddingError`] names
/// it to whoever wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BandPaddingKey {
    /// d3's `padding(p)`: sets both sides.
    Padding,
    /// d3's `paddingInner`.
    Inner,
    /// d3's `paddingOuter`.
    Outer,
}

impl BandPaddingKey {
    /// d3's precedence for the inner side: `paddingInner`, else `padding`.
    /// The one statement of it, read by [`BandPadding::inner`] and by a type
    /// switch that carries a band's inner gap to an ordinal scale.
    const INNER_PRECEDENCE: [BandPaddingKey; 2] = [BandPaddingKey::Inner, BandPaddingKey::Padding];

    /// d3's precedence for the outer side: `paddingOuter`, else `padding`.
    /// Read by [`BandPadding::outer`] and by a type switch that carries a
    /// band's outer padding to a point scale's end padding.
    const OUTER_PRECEDENCE: [BandPaddingKey; 2] = [BandPaddingKey::Outer, BandPaddingKey::Padding];

    /// The argument's name on the `BandScale` constructor.
    pub(crate) fn argument(self) -> &'static str {
        match self {
            BandPaddingKey::Padding => "padding",
            BandPaddingKey::Inner => "padding_inner",
            BandPaddingKey::Outer => "padding_outer",
        }
    }

    /// The key's spelling in a `{"type": "band"}` scale dict.
    pub(crate) fn wire_key(self) -> &'static str {
        match self {
            BandPaddingKey::Padding => "padding",
            BandPaddingKey::Inner => "paddingInner",
            BandPaddingKey::Outer => "paddingOuter",
        }
    }

    /// d3's bounds: `paddingInner` is clamped to at most 1 (a band can shrink
    /// to nothing but not below), `paddingOuter` is unbounded above, and
    /// `padding` sets both sides so it takes the tighter of the two.
    fn accepts(self, value: f64) -> bool {
        match self {
            BandPaddingKey::Padding | BandPaddingKey::Inner => (0.0..=1.0).contains(&value),
            BandPaddingKey::Outer => value.is_finite() && value >= 0.0,
        }
    }

    fn requirement(self) -> &'static str {
        match self {
            BandPaddingKey::Padding | BandPaddingKey::Inner => "must be a number in [0, 1]",
            BandPaddingKey::Outer => "must be a finite number >= 0",
        }
    }
}

/// A band-padding value outside d3's bounds. Each door words the refusal in
/// its own voice around [`reason`](Self::reason): the `BandScale` constructor
/// names its argument, the render path names the scale dict's key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BandPaddingError {
    key: BandPaddingKey,
    value: f64,
}

impl BandPaddingError {
    pub(crate) fn key(&self) -> BandPaddingKey {
        self.key
    }

    /// `must be a number in [0, 1]; received 1.5`.
    pub(crate) fn reason(&self) -> String {
        format!("{}; received {}", self.key.requirement(), self.value)
    }
}

/// A band scale's padding as the user stated it: d3's `padding`,
/// `paddingInner` and `paddingOuter`, each either set or unset.
///
/// The one owner of band padding. Every way a band scale is declared (the
/// `BandScale` class, a `{"type": "band"}` dict, an override type switch)
/// carries this value; the positional resolver, the compute facade and the
/// class getters all read it. It owns three things:
///
/// - **The unset state.** A side nobody set falls back to `padding`, then to
///   0, exactly as d3 `scaleBand` (whose `paddingInner` and `paddingOuter`
///   both start at 0, with `padding(p)` setting both). With nothing set the
///   layout is [`DiscreteLayout::UNPADDED`], the auto-inferred categorical
///   axis. An explicit 0 is a different value from unset: it stays on the
///   wire and in the getters.
/// - **Validation** ([`BandPaddingKey::accepts`]). Every set value is
///   checked, including a `padding` that explicit inner and outer values
///   override, so a malformed argument is refused even when it moves nothing.
/// - **Resolution** ([`layout`](Self::layout)), the one way to turn this
///   value into geometry. It validates first, so no unvalidated value can
///   become geometry: there is no unchecked resolve for a caller to reach.
///
/// Deserialized without validation: a scale dict's malformed padding is
/// refused at render, where the refusal can name the channel.
// `pub` (not `pub(crate)`) only so it can appear in the `pub`
// `ScaleSpec::Band` variant without tripping `private_interfaces`, the same
// arrangement as `scale::ordinal::OrdinalRangeValue`; the `scale` module is
// private, so this type stays crate-internal.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BandPadding {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    padding: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "paddingInner")]
    inner: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "paddingOuter")]
    outer: Option<f64>,
}

impl BandPadding {
    /// The values a caller passed, refused if any set value is out of bounds.
    pub(crate) fn new(
        padding: Option<f64>,
        inner: Option<f64>,
        outer: Option<f64>,
    ) -> Result<Self, BandPaddingError> {
        let value = BandPadding { padding, inner, outer };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), BandPaddingError> {
        let sides = [
            (BandPaddingKey::Padding, self.padding),
            (BandPaddingKey::Inner, self.inner),
            (BandPaddingKey::Outer, self.outer),
        ];
        for (key, value) in sides {
            if let Some(value) = value {
                if !key.accepts(value) {
                    return Err(BandPaddingError { key, value });
                }
            }
        }
        Ok(())
    }

    /// The `padding` the user set, or `None`.
    pub(crate) fn padding(&self) -> Option<f64> {
        self.padding
    }

    /// The inner padding the user's arguments state: `padding_inner`, else
    /// `padding`, else `None` (inherits the auto-inferred layout's 0).
    pub(crate) fn inner(&self) -> Option<f64> {
        self.first_set(BandPaddingKey::INNER_PRECEDENCE)
    }

    /// The outer padding the user's arguments state; mirrors [`inner`](Self::inner).
    pub(crate) fn outer(&self) -> Option<f64> {
        self.first_set(BandPaddingKey::OUTER_PRECEDENCE)
    }

    /// The first value set among `precedence`, or `None`.
    fn first_set(&self, precedence: [BandPaddingKey; 2]) -> Option<f64> {
        precedence.into_iter().find_map(|key| match key {
            BandPaddingKey::Padding => self.padding,
            BandPaddingKey::Inner => self.inner,
            BandPaddingKey::Outer => self.outer,
        })
    }

    /// The `padding_inner` argument exactly as passed, for the wire and repr.
    pub(crate) fn inner_as_set(&self) -> Option<f64> {
        self.inner
    }

    /// The `padding_outer` argument exactly as passed, for the wire and repr.
    pub(crate) fn outer_as_set(&self) -> Option<f64> {
        self.outer
    }

    /// The band layout this padding resolves to at `align`, after validating
    /// every set value. An unset side is 0, so an all-unset value yields
    /// [`DiscreteLayout::UNPADDED`] bit-for-bit at the default align.
    ///
    /// The only way out of a `BandPadding` into geometry: every caller, the
    /// `BandScale` constructor included, goes through this validation.
    pub(crate) fn layout(&self, align: f64) -> Result<DiscreteLayout, BandPaddingError> {
        self.validate()?;
        Ok(DiscreteLayout::band(self.inner().unwrap_or(0.0), self.outer().unwrap_or(0.0), align))
    }
}

/// What a scale type's padding-family keys (`padding`, `paddingInner`,
/// `paddingOuter`, `align`) mean, for translating them across a scale-type
/// switch.
///
/// `padding` alone has four meanings across the scale types, so a switch
/// that carried it by name would silently reinterpret it. A switch instead
/// reads what the old type's keys meant and writes the new type's keys that
/// mean the same thing ([`switch_carries`](Self::switch_carries)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaddingMeaning {
    /// `ordinal`: `padding` is the inner gap between bands; there is no outer
    /// padding and no `align`.
    OrdinalInner,
    /// `band`: d3's `padding` shorthand plus `paddingInner`, `paddingOuter`
    /// and `align` ([`BandPadding`]).
    Band,
    /// `point`: `padding` is the end padding before the first and after the
    /// last point, plus `align`.
    Point,
    /// A continuous scale: `padding` is a fraction of the pixel span held at
    /// each end.
    SpanFraction,
    /// A scale type with no padding-family key.
    Unpadded,
}

/// One padding-family key a type switch writes on the new scale, and the old
/// scale's keys it reads. The first of `from` the user set wins; when none is
/// set, the key is not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PaddingCarry {
    /// The new scale's wire key.
    pub(crate) to: &'static str,
    /// The old scale's wire keys, in precedence order.
    pub(crate) from: Vec<&'static str>,
}

impl PaddingCarry {
    fn same(key: &'static str) -> Self {
        PaddingCarry { to: key, from: vec![key] }
    }

    fn band_side(to: &'static str, precedence: [BandPaddingKey; 2]) -> Self {
        PaddingCarry { to, from: precedence.map(BandPaddingKey::wire_key).to_vec() }
    }
}

impl PaddingMeaning {
    /// The padding-family wire keys this meaning gives a scale.
    pub(crate) fn keys(self) -> &'static [&'static str] {
        match self {
            PaddingMeaning::OrdinalInner | PaddingMeaning::SpanFraction => &["padding"],
            PaddingMeaning::Band => &["padding", "paddingInner", "paddingOuter", "align"],
            PaddingMeaning::Point => &["padding", "align"],
            PaddingMeaning::Unpadded => &[],
        }
    }

    /// The keys a switch from a scale with this meaning to one with `to`
    /// writes, each with the old keys it reads. A value with no meaning under
    /// the new type has no carry, so it is dropped:
    ///
    /// | From → To | Carried |
    /// |---|---|
    /// | same meaning | every key, unchanged |
    /// | ordinal → band | `padding` → `paddingInner` |
    /// | point → band | `padding` → `paddingOuter`; `align` → `align` |
    /// | band → ordinal | the inner side (`paddingInner`, else `padding`) → `padding` |
    /// | band → point | the outer side (`paddingOuter`, else `padding`) → `padding`; `align` → `align` |
    /// | ordinal ↔ point | nothing: ordinal has no end padding, point no inner gap |
    /// | continuous ↔ discrete | nothing: a span fraction is not a step fraction |
    /// | to or from a type with no padding | nothing |
    ///
    /// The band sides read [`BandPadding`]'s own precedence, so a switch from
    /// band carries exactly what [`BandPadding::inner`] and
    /// [`BandPadding::outer`] report. Values are carried, never checked here:
    /// one that is out of bounds for the new type is refused by that type's
    /// own validation.
    pub(crate) fn switch_carries(self, to: PaddingMeaning) -> Vec<PaddingCarry> {
        use PaddingMeaning::{Band, OrdinalInner, Point, SpanFraction, Unpadded};
        match (self, to) {
            (OrdinalInner, OrdinalInner)
            | (Band, Band)
            | (Point, Point)
            | (SpanFraction, SpanFraction)
            | (Unpadded, Unpadded) => self.keys().iter().copied().map(PaddingCarry::same).collect(),
            (OrdinalInner, Band) => {
                vec![PaddingCarry { to: BandPaddingKey::Inner.wire_key(), from: vec!["padding"] }]
            }
            (Point, Band) => vec![
                PaddingCarry { to: BandPaddingKey::Outer.wire_key(), from: vec!["padding"] },
                PaddingCarry::same("align"),
            ],
            (Band, OrdinalInner) => vec![PaddingCarry::band_side("padding", BandPaddingKey::INNER_PRECEDENCE)],
            (Band, Point) => vec![
                PaddingCarry::band_side("padding", BandPaddingKey::OUTER_PRECEDENCE),
                PaddingCarry::same("align"),
            ],
            (OrdinalInner, Point) | (Point, OrdinalInner) => Vec::new(),
            (SpanFraction, OrdinalInner | Band | Point) | (OrdinalInner | Band | Point, SpanFraction) => {
                Vec::new()
            }
            (Unpadded, _) | (_, Unpadded) => Vec::new(),
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
///   is chart-absolute by design (#39 phase 2), so it travels to layout as the
///   range's own endpoints, already in pixels, together with the model. The
///   centers are resolved from those endpoints with the arithmetic the scale
///   uses for its marks, so a facet panel's re-anchored range
///   ([`translated`](Self::translated)) lands its labels on the panel's marks
///   bit for bit (GH #70): the offset moves the endpoints, never the finished
///   centers, which would round differently.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CategoricalPlacement {
    /// The scale's d3 model and domain size over its chart-absolute pixel
    /// range, given as the range's first and last endpoints.
    Absolute { layout: DiscreteLayout, categories: usize, range: (f64, f64) },
    /// The scale's d3 model plus its domain size, to be resolved against the
    /// panel's pixel interval at layout time.
    PanelExtent { layout: DiscreteLayout, categories: usize },
}

impl CategoricalPlacement {
    /// The absolute center pixel of every category, in domain order, over the
    /// axis's pixel interval `[lo, hi]`.
    ///
    /// `lo`/`hi` are ignored by [`Absolute`](Self::Absolute), which resolves
    /// over its own chart-absolute range.
    pub(crate) fn centers(&self, lo: f64, hi: f64) -> Vec<f64> {
        let (layout, categories, (lo, hi)) = match *self {
            CategoricalPlacement::Absolute { layout, categories, range } => (layout, categories, range),
            CategoricalPlacement::PanelExtent { layout, categories } => (layout, categories, (lo, hi)),
        };
        let g = layout.geometry(categories, lo, hi);
        (0..categories).map(|i| g.position(i)).collect()
    }

    /// How many centers [`centers`](Self::centers) yields — the scale's domain
    /// size, known without a pixel interval. Lets the placement/label pairing
    /// invariant be checked before any geometry is resolved
    /// (`AxisInput::debug_assert_placement_invariants`).
    pub(crate) fn len(&self) -> usize {
        match self {
            CategoricalPlacement::Absolute { categories, .. }
            | CategoricalPlacement::PanelExtent { categories, .. } => *categories,
        }
    }

    /// This placement on a facet panel `offset` pixels from the reference
    /// panel along the axis: an [`Absolute`](Self::Absolute) range has both
    /// endpoints shifted by `offset`, as the panel's mark scale is
    /// (`OrdinalScale::translate_explicit_range`), so the axis and the marks
    /// resolve the same geometry over the same range. A
    /// [`PanelExtent`](Self::PanelExtent) placement already resolves against
    /// each panel's own rect and is returned as is, and so is any placement
    /// at `offset == 0.0`.
    pub(crate) fn translated(self, offset: f64) -> Self {
        match self {
            CategoricalPlacement::Absolute { layout, categories, range: (lo, hi) } if offset != 0.0 => {
                CategoricalPlacement::Absolute { layout, categories, range: (lo + offset, hi + offset) }
            }
            _ => self,
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

    // ── BandPadding: unset state, d3 bounds, d3 resolution (#157) ───────────

    fn pad(padding: Option<f64>, inner: Option<f64>, outer: Option<f64>) -> BandPadding {
        BandPadding::new(padding, inner, outer).expect("in-bounds band padding")
    }

    fn refusal(padding: Option<f64>, inner: Option<f64>, outer: Option<f64>) -> BandPaddingError {
        BandPadding::new(padding, inner, outer).expect_err("out-of-bounds band padding")
    }

    /// Nothing set resolves to the auto-inferred categorical layout, compared
    /// as the layout value itself (bit-for-bit, `assert_eq!` on f64) and as the
    /// geometry it produces.
    #[test]
    fn unset_band_padding_is_the_unpadded_layout() {
        let layout = BandPadding::default().layout(DEFAULT_ALIGN).unwrap();
        assert_eq!(layout, DiscreteLayout::UNPADDED);
        assert_eq!(
            layout.geometry(4, 0.0, 561.636),
            DiscreteLayout::UNPADDED.geometry(4, 0.0, 561.636)
        );
        assert_eq!(BandPadding::default(), pad(None, None, None));
    }

    /// `padding` alone sets both sides, exactly as d3's `padding(p)`.
    #[test]
    fn padding_alone_sets_both_sides() {
        let layout = pad(Some(0.1), None, None).layout(0.5).unwrap();
        assert_eq!(layout, DiscreteLayout::band(0.1, 0.1, 0.5));
        // 4 bars over the 561.636px panel: the disclosed 0.8·0.9·w/4.1.
        let bar = 0.8 * layout.geometry(4, 0.0, 561.636).bandwidth();
        assert!((bar - 98.629).abs() < 1e-3, "padded bar width {bar}");
        let unpadded = 0.8 * DiscreteLayout::UNPADDED.geometry(4, 0.0, 561.636).bandwidth();
        assert!((unpadded - 112.327).abs() < 1e-3, "unpadded bar width {unpadded}");
    }

    /// One side set: the other falls back to `padding`, then to 0 (d3's own
    /// `paddingInner = paddingOuter = 0` defaults), never to a hidden 0.1.
    #[test]
    fn one_side_set_leaves_the_other_at_zero() {
        assert_eq!(pad(None, Some(0.2), None).layout(0.5).unwrap(), DiscreteLayout::band(0.2, 0.0, 0.5));
        assert_eq!(pad(None, None, Some(0.3)).layout(0.5).unwrap(), DiscreteLayout::band(0.0, 0.3, 0.5));
    }

    /// Both sides set win over nothing; an explicit side beats `padding`
    /// while the unset side still takes `padding`.
    #[test]
    fn explicit_sides_take_precedence_over_padding() {
        assert_eq!(pad(None, Some(0.2), Some(0.3)).layout(0.5).unwrap(), DiscreteLayout::band(0.2, 0.3, 0.5));
        assert_eq!(pad(Some(0.4), Some(0.2), None).layout(0.5).unwrap(), DiscreteLayout::band(0.2, 0.4, 0.5));
        assert_eq!(pad(Some(0.4), None, Some(0.2)).layout(0.5).unwrap(), DiscreteLayout::band(0.4, 0.2, 0.5));
    }

    /// The getters report what the user's arguments state: `padding` as
    /// passed, each side as `side ?? padding`, `None` when neither is set.
    #[test]
    fn getters_report_the_stated_padding() {
        let unset = BandPadding::default();
        assert_eq!((unset.padding(), unset.inner(), unset.outer()), (None, None, None));
        let shorthand = pad(Some(0.1), None, None);
        assert_eq!((shorthand.padding(), shorthand.inner(), shorthand.outer()), (Some(0.1), Some(0.1), Some(0.1)));
        let mixed = pad(Some(0.4), Some(0.2), None);
        assert_eq!((mixed.padding(), mixed.inner(), mixed.outer()), (Some(0.4), Some(0.2), Some(0.4)));
        assert_eq!((mixed.inner_as_set(), mixed.outer_as_set()), (Some(0.2), None));
        let inner_only = pad(None, Some(0.2), None);
        assert_eq!((inner_only.padding(), inner_only.inner(), inner_only.outer()), (None, Some(0.2), None));
    }

    /// An explicit 0 is a value, not "unset": it differs from the unset value in
    /// equality, the getters and the wire, while resolving to the same layout.
    #[test]
    fn explicit_zero_is_distinct_from_unset() {
        let zero = pad(Some(0.0), None, None);
        assert_ne!(zero, BandPadding::default());
        assert_eq!(zero.padding(), Some(0.0));
        assert_eq!(zero.inner(), Some(0.0));
        assert_eq!(serde_json::to_value(zero).unwrap(), serde_json::json!({"padding": 0.0}));
        assert_eq!(serde_json::to_value(BandPadding::default()).unwrap(), serde_json::json!({}));
        assert_eq!(zero.layout(DEFAULT_ALIGN).unwrap(), DiscreteLayout::UNPADDED);
    }

    /// The wire carries only the keys the user set, under d3's spellings, and
    /// reads an absent or `null` key as unset.
    #[test]
    fn wire_form_carries_only_set_keys() {
        assert_eq!(
            serde_json::to_value(pad(None, Some(0.2), None)).unwrap(),
            serde_json::json!({"paddingInner": 0.2})
        );
        assert_eq!(
            serde_json::to_value(pad(Some(0.1), Some(0.2), Some(0.3))).unwrap(),
            serde_json::json!({"padding": 0.1, "paddingInner": 0.2, "paddingOuter": 0.3})
        );
        let parsed: BandPadding =
            serde_json::from_value(serde_json::json!({"padding": null, "paddingOuter": 0.3})).unwrap();
        assert_eq!(parsed, pad(None, None, Some(0.3)));
        let empty: BandPadding = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, BandPadding::default());
    }

    /// d3's bounds: `padding` and `paddingInner` in [0, 1] inclusive (an inner
    /// padding of 1 is a zero-width band, which d3 allows), `paddingOuter`
    /// finite and >= 0 with no upper bound. NaN and infinities are refused.
    #[test]
    fn bounds_follow_d3() {
        for v in [0.0, 0.5, 1.0] {
            assert!(BandPadding::new(Some(v), None, None).is_ok(), "padding={v}");
            assert!(BandPadding::new(None, Some(v), None).is_ok(), "inner={v}");
        }
        for v in [0.0, 1.0, 1.5, 10.0] {
            assert!(BandPadding::new(None, None, Some(v)).is_ok(), "outer={v}");
        }
        for v in [-0.1, 1.000_001, 2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(refusal(Some(v), None, None).key(), BandPaddingKey::Padding, "padding={v}");
            assert_eq!(refusal(None, Some(v), None).key(), BandPaddingKey::Inner, "inner={v}");
        }
        for v in [-0.1, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(refusal(None, None, Some(v)).key(), BandPaddingKey::Outer, "outer={v}");
        }
    }

    /// A malformed `padding` is refused even when explicit inner and outer
    /// values mean it would move nothing.
    #[test]
    fn malformed_padding_masked_by_both_sides_is_still_refused() {
        let error = refusal(Some(1.5), Some(0.2), Some(0.3));
        assert_eq!(error.key(), BandPaddingKey::Padding);
        assert_eq!(error.reason(), "must be a number in [0, 1]; received 1.5");
    }

    /// The refusal names the requirement and the value received; each key
    /// knows its constructor argument and its wire spelling.
    #[test]
    fn refusal_reason_and_key_names() {
        assert_eq!(refusal(None, Some(f64::NAN), None).reason(), "must be a number in [0, 1]; received NaN");
        assert_eq!(refusal(None, None, Some(-1.0)).reason(), "must be a finite number >= 0; received -1");
        assert_eq!(refusal(None, None, Some(f64::INFINITY)).reason(), "must be a finite number >= 0; received inf");
        let names = [BandPaddingKey::Padding, BandPaddingKey::Inner, BandPaddingKey::Outer]
            .map(|k| (k.argument(), k.wire_key()));
        assert_eq!(
            names,
            [("padding", "padding"), ("padding_inner", "paddingInner"), ("padding_outer", "paddingOuter")]
        );
    }

    // ── type-switch translation ──────────────────────────────────────────────

    fn carries(from: PaddingMeaning, to: PaddingMeaning) -> Vec<(&'static str, Vec<&'static str>)> {
        from.switch_carries(to).into_iter().map(|c| (c.to, c.from)).collect()
    }

    /// The pairs that translate: each writes the new type's key that means
    /// what the old type's key meant.
    #[test]
    fn switch_translates_padding_by_meaning() {
        use PaddingMeaning::{Band, OrdinalInner, Point};
        assert_eq!(carries(OrdinalInner, Band), [("paddingInner", vec!["padding"])]);
        assert_eq!(carries(Point, Band), [("paddingOuter", vec!["padding"]), ("align", vec!["align"])]);
        assert_eq!(carries(Band, OrdinalInner), [("padding", vec!["paddingInner", "padding"])]);
        assert_eq!(carries(Band, Point), [("padding", vec!["paddingOuter", "padding"]), ("align", vec!["align"])]);
    }

    /// The pairs with no shared meaning carry nothing, so their padding
    /// family is dropped.
    #[test]
    fn switch_drops_padding_with_no_meaning_under_the_new_type() {
        use PaddingMeaning::{Band, OrdinalInner, Point, SpanFraction, Unpadded};
        let mut dropped = vec![(OrdinalInner, Point), (Point, OrdinalInner)];
        for discrete in [OrdinalInner, Band, Point] {
            dropped.extend([(SpanFraction, discrete), (discrete, SpanFraction)]);
        }
        for padded in [OrdinalInner, Band, Point, SpanFraction] {
            dropped.extend([(Unpadded, padded), (padded, Unpadded)]);
        }
        for (from, to) in dropped {
            assert_eq!(carries(from, to), [], "{from:?} -> {to:?}");
        }
    }

    /// A switch between two types of the same meaning (linear → log) carries
    /// every key of that meaning unchanged.
    #[test]
    fn switch_within_one_meaning_carries_every_key_unchanged() {
        use PaddingMeaning::{Band, OrdinalInner, Point, SpanFraction, Unpadded};
        for meaning in [OrdinalInner, Band, Point, SpanFraction, Unpadded] {
            let expected: Vec<_> = meaning.keys().iter().map(|&k| (k, vec![k])).collect();
            assert_eq!(carries(meaning, meaning), expected, "{meaning:?}");
        }
        assert_eq!(carries(SpanFraction, SpanFraction), [("padding", vec!["padding"])]);
    }

    /// A switch from band reads the same precedence `BandPadding` resolves
    /// by: over every set/unset combination, the first set source of the
    /// carry to ordinal is `inner()`, and of the carry to point is `outer()`.
    #[test]
    fn switch_from_band_reads_band_paddings_own_precedence() {
        for padding in [None, Some(0.1)] {
            for inner in [None, Some(0.2)] {
                for outer in [None, Some(0.3)] {
                    let band = BandPadding::new(padding, inner, outer).unwrap();
                    let read = |key: &str| match key {
                        "padding" => padding,
                        "paddingInner" => inner,
                        "paddingOuter" => outer,
                        other => panic!("unexpected source {other}"),
                    };
                    let first_set =
                        |to: PaddingMeaning| carries(PaddingMeaning::Band, to)[0].1.iter().find_map(|k| read(k));
                    assert_eq!(first_set(PaddingMeaning::OrdinalInner), band.inner(), "{band:?}");
                    assert_eq!(first_set(PaddingMeaning::Point), band.outer(), "{band:?}");
                }
            }
        }
    }

    /// The shared bound predicates refuse non-finite values, which a scale
    /// dict cannot carry (JSON has no NaN or infinity) but the class doors
    /// can, and accept their closed edges.
    #[test]
    fn bound_predicates_refuse_non_finite_and_accept_edges() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(check_align(bad).is_err(), "{bad}");
            assert!(check_point_padding(bad).is_err(), "{bad}");
            assert!(check_ordinal_padding(bad).is_err(), "{bad}");
        }
        assert_eq!(check_align(f64::NAN).unwrap_err(), "must be a number in [0, 1]; received NaN");
        assert_eq!(check_point_padding(f64::INFINITY).unwrap_err(), "must be a finite number >= 0; received inf");
        assert_eq!(check_ordinal_padding(f64::NAN).unwrap_err(), "must be in [0, 1]; got NaN");
        for ok in [0.0, 1.0] {
            assert_eq!(check_align(ok), Ok(()));
            assert_eq!(check_ordinal_padding(ok), Ok(()));
        }
        assert_eq!(check_point_padding(0.0), Ok(()));
        assert_eq!(check_point_padding(1.0e6), Ok(()));
    }

    /// A deserialized value is not validated until it resolves, and resolving
    /// refuses it: no unvalidated padding can become geometry.
    #[test]
    fn a_deserialized_malformed_value_cannot_resolve() {
        let parsed: BandPadding = serde_json::from_value(serde_json::json!({"paddingInner": 1.5})).unwrap();
        let error = parsed.layout(DEFAULT_ALIGN).expect_err("1.5 is outside [0, 1]");
        assert_eq!(error.key(), BandPaddingKey::Inner);
    }
}
