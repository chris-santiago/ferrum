//! Composite-resolve ↔ scale-engine seam vocabulary.
//!
//! [`Channel`], [`SharedDomain`], and [`LeafScaleContext`] are consumed by
//! BOTH `render::composite`'s resolve pass (which builds a `LeafScaleContext`
//! per leaf, unioning one `SharedDomain` per shared channel across a
//! composite tree's leaves — `resolve_composite_scales`) and this module's
//! own scale builders (`build_axis_scale`, `build_color_scale`,
//! `build_size_scale`/`build_opacity_scale` via `build_auxiliary_scales`),
//! which consult a leaf's shared domain on the auto-scale path (D4b).
//!
//! These types used to live in `render::composite`, which meant the general
//! scale engine (`scale_resolve`) reached *up* into a consumer feature for
//! its own seam vocabulary — `scale_resolve/{positional,color,auxiliary}.rs`
//! imported `SharedDomain` from `composite`, while `composite` imported
//! `scale_resolve`'s domain-union helpers back, an inverted, circular
//! dependency. Homing the seam types here — the lower layer both the resolve
//! pass and the scale builders sit on top of — makes the dependency point
//! one way: `composite` depends on `scale_resolve`, never the reverse. Pure
//! move (72h findings burndown item 1, following the design/quality review
//! recommendation); no behavior change.

/// Shared-channel selector used to resolve one composite-shared domain at a
/// time. Positional `x`/`y` union through the facet mechanism; non-positional
/// `color`/`size` union through the simpler per-column extent/category
/// helpers (10-pre-b). Moved alongside [`LeafScaleContext`] (whose `set`
/// method is keyed on it) so the seam vocabulary stays together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    X,
    Y,
    Color,
    Size,
}

impl Channel {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Channel::X => "x",
            Channel::Y => "y",
            Channel::Color => "color",
            Channel::Size => "size",
        }
    }
}

/// A positional channel resolved across a composite group. Numeric/temporal
/// channels union to a single `[lo, hi]` extent; ordinal channels union to an
/// order-preserving category vector (semantics locked by #35).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SharedDomain {
    /// Quantitative or temporal extent (temporal is epoch-ms, same as the scale
    /// resolver treats it). The consuming auto path decides Linear vs Time from
    /// the leaf's own column dtype; this only supplies the shared extent.
    Numeric { lo: f64, hi: f64 },
    /// Ordinal/nominal domain in first-appearance order across the group.
    Ordinal(Vec<String>),
}

/// The resolved shared domains for one leaf's shared channels (positional x/y
/// plus non-positional color/size), plus the composite-shared-legend
/// suppression signal (design §6 seam contract, 2026-07-12). `None` on a
/// domain field means "no composite sharing applies" — the leaf resolves that
/// channel exactly as it would standalone (its own data, its own explicit
/// scale, or its own internal facet resolution).
///
/// For `color`, a [`SharedDomain::Numeric`] is a continuous (colorbar) extent and
/// a [`SharedDomain::Ordinal`] is the categorical (swatch) domain; for `size`,
/// only [`SharedDomain::Numeric`] is produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LeafScaleContext {
    pub(crate) x: Option<SharedDomain>,
    pub(crate) y: Option<SharedDomain>,
    pub(crate) color: Option<SharedDomain>,
    pub(crate) size: Option<SharedDomain>,
    /// Layout-stage-only signal (never affects `prepare_render_inputs`, which
    /// always builds the channel's legend bundle in full) that the compositor
    /// is rendering one figure-level legend for this channel and this leaf's
    /// own panel legend must reserve no gutter and draw nothing —
    /// `render::mod::prepare_and_layout` reads these two flags into a
    /// [`crate::layout::LegendSuppression`] for `compute_layout`. Independent
    /// of the `color`/`size` domain fields above (a leaf can carry a shared
    /// domain without being suppressed, e.g. `legend={"color": "independent"}`
    /// over a shared scale); set by the compositor, never derived here.
    /// `false` (the default) reproduces today's per-panel legend rendering.
    pub(crate) suppress_color_legend: bool,
    pub(crate) suppress_size_legend: bool,
}

impl LeafScaleContext {
    /// True when no channel carries a shared domain and no legend is
    /// suppressed — the leaf renders exactly as it would standalone.
    /// Compares against `Default` so a future field is covered automatically.
    pub(crate) fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub(crate) fn set(&mut self, channel: Channel, domain: SharedDomain) {
        match channel {
            Channel::X => self.x = Some(domain),
            Channel::Y => self.y = Some(domain),
            Channel::Color => self.color = Some(domain),
            Channel::Size => self.size = Some(domain),
        }
    }
}
