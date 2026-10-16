//! Composite-resolve ↔ scale-engine seam vocabulary.
//!
//! [`Channel`], [`SharedDomain`], and [`LeafScaleContext`] are consumed by
//! BOTH `render::composite`'s resolve pass (which builds a `LeafScaleContext`
//! per leaf, unioning one `SharedDomain` per shared channel across a
//! composite tree's leaves — `resolve_composite_scales`) and this module's
//! own scale builders (`build_axis_scale`, `build_color_scale`,
//! `build_size_scale`/`build_opacity_scale` via `build_auxiliary_scales`),
//! which consult a leaf's shared domain on the auto-scale path (D4b).
//! [`LeafResolveInput`] is the per-leaf view both passes read, and
//! [`OverlayAuxiliary`] the auxiliary scales an overlay group of leaves
//! resolves once for all its members, carried on each member's
//! `LeafScaleContext`.
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

/// One composite leaf as the composite passes read it after its transforms
/// ran: its spec (layers, facet), its rendering (post-`CoordFlip`,
/// establishing) encoding, its final batch and its named transform outputs.
/// The composite resolve pass unions shared domains over these, and an
/// overlay group's auxiliary scales are built over its members' ones
/// ([`super::build_overlay_auxiliary_scales`]).
pub(crate) struct LeafResolveInput<'a> {
    pub(crate) spec: &'a crate::spec::chart::ChartSpec,
    pub(crate) encoding: &'a crate::spec::encoding::Encoding,
    pub(crate) final_batch: &'a arrow::record_batch::RecordBatch,
    pub(crate) transform_outputs: &'a std::collections::HashMap<String, arrow::record_batch::RecordBatch>,
}

/// The auxiliary channels an overlay group leaves to its members: those an
/// explicit composite resolve names (`resolve.color`/`resolve.size`, or a
/// `legend` override of either) on the overlay or an enclosing composite,
/// which keep the meaning that resolve gives them. No resolve can name
/// `shape`, the opacity family or `stroke_dash`, so those are always grouped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemberChannels {
    pub(crate) color: bool,
    pub(crate) size: bool,
}

impl MemberChannels {
    /// Whether the channel named `channel` (an
    /// [`AUXILIARY_SCALE_CHANNELS`](crate::spec::encoding::AUXILIARY_SCALE_CHANNELS)
    /// name) is left to the members.
    pub(crate) fn holds(self, channel: &str) -> bool {
        match channel {
            "color" => self.color,
            "size" => self.size,
            _ => false,
        }
    }

    /// Whether any channel is left to the members.
    pub(crate) fn any(self) -> bool {
        self.color || self.size
    }

    /// Clear on `encoding` every auxiliary-scale channel for which `keep`
    /// answers `false`.
    fn retain_on(encoding: &mut crate::spec::encoding::Encoding, keep: impl Fn(&str) -> bool) {
        for channel in crate::spec::encoding::AUXILIARY_SCALE_CHANNELS {
            if !keep(channel.name) {
                *channel.binding_mut(encoding) = None;
            }
        }
    }

    /// `encoding` without the channels left to the members: the group's own.
    pub(crate) fn grouped_only(self, encoding: &mut crate::spec::encoding::Encoding) {
        Self::retain_on(encoding, |channel| !self.holds(channel));
    }

    /// `encoding` with only the channels left to the members among its
    /// auxiliary-scale channels: a member's own.
    pub(crate) fn members_only(self, encoding: &mut crate::spec::encoding::Encoding) {
        Self::retain_on(encoding, |channel| self.holds(channel));
    }

    /// `scales` without the group's auxiliary scales, keeping those of the
    /// channels left to the members.
    pub(in crate::render) fn members_only_scales(self, scales: &mut super::ResolvedScales) {
        if !self.color {
            scales.color = None;
        }
        if !self.size {
            scales.size = None;
        }
        scales.shape = None;
        scales.opacity = None;
        scales.fill_opacity = None;
        scales.stroke_opacity = None;
        scales.stroke_dash = None;
    }
}

/// The auxiliary scales of an `Overlay` group of leaves, resolved once over
/// every member as one layered chart resolves them over its layers: each
/// grouped channel's binding comes from the first member drawing it, and its
/// domain spans every admitted member's column. Every member draws through
/// these scales, and the group's first member draws its one legend per
/// grouped channel. The channels left to the members
/// ([`MemberChannels`]) each member resolves and draws itself.
///
/// A leaf resolves its auxiliary scales under two themes: its own, for the
/// legends `prepare` builds, and that theme under its chart config
/// (`configure_color(scheme=...)` and the like), for the marks and the legend
/// colors the scene draws. The group is built under both from its first
/// member's theme and chart config, and a build reads the one under the theme
/// it is given ([`Self::scales_for`]). Every member shares that theme and
/// chart config because `LayerChart` lowers each overlay member with the
/// primary member's theme and the one configure list its members merge
/// into, as `+` resolves them; a member's own theme and config are never
/// read for the group.
#[derive(Debug)]
pub(crate) struct OverlayAuxiliary {
    /// The first member's rendering encoding, with each grouped channel it
    /// leaves unbound taken from the first member that draws it, and no
    /// channel left to the members.
    pub(crate) encoding: crate::spec::encoding::Encoding,
    pub(in crate::render) scales: super::AuxiliaryScales,
    /// The same scales built under the members' theme with their chart
    /// config applied, and the palette that theme carries.
    pub(in crate::render) configured: ConfiguredAuxiliary,
    /// Every member's layers, in member order: the marks the group's
    /// legends describe.
    pub(crate) layers: Vec<crate::render::prepare::LayerPrepared>,
    pub(crate) left_to_members: MemberChannels,
}

impl OverlayAuxiliary {
    /// A group whose scales are not resolved yet: its members prepare their
    /// transforms and layers, which the group's resolution reads, resolving
    /// only the channels left to them.
    pub(crate) fn unresolved(left_to_members: MemberChannels) -> Self {
        Self {
            encoding: crate::spec::encoding::Encoding::default(),
            scales: super::AuxiliaryScales::default(),
            configured: ConfiguredAuxiliary::default(),
            layers: Vec::new(),
            left_to_members,
        }
    }
}

impl OverlayAuxiliary {
    /// The group's scales as a build under `theme` reads them: the
    /// configured ones under the palette the members' chart config gives,
    /// else those under the members' own theme. The palette is the only
    /// theme input a chart config changes that an auxiliary scale reads.
    pub(in crate::render) fn scales_for(&self, theme: &crate::layout::ThemeInputs) -> &super::AuxiliaryScales {
        if theme.palette == self.configured.palette {
            &self.configured.scales
        } else {
            &self.scales
        }
    }
}

/// An overlay group's auxiliary scales built under its members' chart
/// config, and the palette they were built under.
#[derive(Debug, Default)]
pub(crate) struct ConfiguredAuxiliary {
    pub(in crate::render) palette: crate::layout::ThemePalette,
    pub(in crate::render) scales: super::AuxiliaryScales,
}

/// A leaf's place in an [`OverlayAuxiliary`] group: the group, shared by its
/// members, and this leaf's position among them.
#[derive(Debug, Clone)]
pub(crate) struct OverlayMembership {
    pub(crate) group: std::sync::Arc<OverlayAuxiliary>,
    pub(crate) member: usize,
}

/// Two memberships are equal when they name the same member of the same
/// group (the same resolution, not two equal-looking ones).
impl PartialEq for OverlayMembership {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.group, &other.group) && self.member == other.member
    }
}

/// The legends an overlay member builds.
pub(crate) enum OverlayLegends<'a> {
    /// The first member: one legend set over `encoding` (the group's
    /// bindings, plus its own for the channels left to the members) and
    /// every member's layers.
    Composed {
        encoding: crate::spec::encoding::Encoding,
        layers: &'a [crate::render::prepare::LayerPrepared],
    },
    /// A later member: its own legends of the channels left to the members,
    /// over `encoding` (its own bindings of those channels) and its own
    /// layers, drawn against its own scales of them.
    LeftToMembers {
        encoding: crate::spec::encoding::Encoding,
        channels: MemberChannels,
    },
    /// A later member of a group that leaves no channel to its members:
    /// it draws no legend.
    None,
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

/// Everything a composite parent imposes on ONE leaf: the resolved shared
/// domains for its shared channels (positional x/y plus non-positional
/// color/size), the composite-shared-legend suppression signal (design §6
/// seam contract, 2026-07-12), and the overlay group's shared plot region
/// (GH #89A). `None` on a domain field means "no composite sharing applies" —
/// the leaf resolves that channel exactly as it would standalone (its own
/// data, its own explicit scale, or its own internal facet resolution).
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
    /// The caller's positional scale this leaf's `x`/`y` axis resolves with:
    /// set for a leaf that declares no scale on the channel (or only the
    /// injected bar anchor), or an equal copy of it, and draws in an
    /// `Overlay` group whose axis a caller's continuous scale establishes
    /// (see `domain::establishing_caller_scale`). The leaf then resolves that
    /// scale, over the group's shared extent when it states no `domain`, the
    /// same axis the scale's own leaf draws. `None` (the default) resolves
    /// the leaf's own encoding.
    pub(crate) x_scale: Option<crate::spec::encoding::ScaleSpec>,
    pub(crate) y_scale: Option<crate::spec::encoding::ScaleSpec>,
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
    /// Layout-stage-only signal (GH #89A): the one plot region every leaf of
    /// an all-leaves `Overlay` group lays out against, computed by the
    /// compositor's shared-rect pre-pass as the intersection of the group
    /// leaves' natural regions. `render::mod::prepare_and_layout` forwards it
    /// into [`crate::layout::CompositeLayoutSeam::plot_region`], which
    /// replaces the region this leaf's own axis-band reservation produced —
    /// so the leaf's panels, tick pixel positions, and axis titles all
    /// describe the group's rect, and the compositor can drop the duplicate
    /// chrome of every non-primary leaf without leaving a stale layout
    /// product behind. `None` (the default) leaves the leaf laying out
    /// entirely on its own terms.
    pub(crate) imposed_plot_region: Option<crate::layout::Rect>,
    /// Layout-stage-only signal (GH #89A): this leaf's chart-title band must
    /// not be reserved, because the compositor clears its scene title at the
    /// merge seam. Set for exactly the non-primary overlay leaves whose
    /// chrome is dropped, so a title that is never drawn cannot reserve a
    /// phantom top gutter in the group's shared rect. `false` (the default)
    /// reserves the band exactly as a standalone chart does.
    pub(crate) suppress_chart_title: bool,
    /// Layout-stage-only signal (T5b static-composite fix, spec §4.0's second
    /// bullet): this leaf belongs to an all-leaf `Overlay` group (the same
    /// #89A groups `plan_overlay_groups` names) that ALSO contains a sibling
    /// leaf binding `color` to a mark other than line/ribbon — that sibling
    /// genuinely renders the group's shared color mapping. `render::prepare::
    /// legend::build_color_legend`'s inert-color-on-line-or-ribbon check sees
    /// only its OWN leaf's per-panel mark set under the composite path (each
    /// leaf renders through its own standalone `prepare_render_inputs`, never
    /// the whole group), so it cannot see a sibling leaf on its own — this
    /// bit is how the compositor's group-wide view reaches it. Set by
    /// `composite_render::plan_line_ribbon_color_group_exemptions`, never
    /// derived here. `false` (the default) leaves the check exactly as a
    /// standalone chart's.
    pub(crate) color_scale_has_non_line_ribbon_sibling: bool,
    /// Layout-stage-only signal: this leaf belongs to an all-leaf `Overlay`
    /// group in which a sibling leaf draws this leaf's size field with a mark
    /// other than a line, so the group's size scale is drawn as dots and this
    /// leaf's size legend keeps circle swatches even when its own layers are
    /// all lines (`render::prepare::legend::size_legend_symbol`). Set by
    /// `composite_render::plan_size_dot_siblings`. `false` (the default)
    /// decides from this leaf's own layers, as a standalone chart does.
    pub(crate) size_scale_has_non_line_sibling: bool,
    /// The all-leaf `Overlay` group this leaf belongs to, when the group
    /// resolves its auxiliary scales as one chart (set by
    /// `composite_render::plan_overlay_auxiliary`): the leaf draws through
    /// the group's scales, and only the group's first member draws legends.
    /// `None` (the default) resolves and draws the leaf's own.
    pub(crate) overlay: Option<OverlayMembership>,
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

    /// The overlay group this leaf belongs to, when the group resolves its
    /// auxiliary scales as one chart.
    pub(in crate::render) fn overlay_group(&self) -> Option<&OverlayAuxiliary> {
        self.overlay.as_ref().map(|m| &*m.group)
    }

    /// Whether this leaf's overlay group resolves the auxiliary channel named
    /// `channel` for all its members (rather than leaving it to each).
    pub(crate) fn overlay_groups_channel(&self, channel: &str) -> bool {
        self.overlay_group().is_some_and(|g| !g.left_to_members.holds(channel))
    }

    /// The legends this leaf builds as a member of its overlay group, given
    /// its own rendering encoding; `None` when it is in no group and builds
    /// its own.
    pub(crate) fn overlay_legends(&self, rendering: &crate::spec::encoding::Encoding) -> Option<OverlayLegends<'_>> {
        let membership = self.overlay.as_ref()?;
        let group = &*membership.group;
        let left = group.left_to_members;
        Some(if membership.member == 0 {
            let mut encoding = group.encoding.clone();
            for channel in crate::spec::encoding::AUXILIARY_SCALE_CHANNELS {
                if left.holds(channel.name) {
                    channel.binding_mut(&mut encoding).clone_from(channel.binding(rendering));
                }
            }
            OverlayLegends::Composed { encoding, layers: &group.layers }
        } else if left.any() {
            let mut encoding = rendering.clone();
            left.members_only(&mut encoding);
            OverlayLegends::LeftToMembers { encoding, channels: left }
        } else {
            OverlayLegends::None
        })
    }

    /// Record the caller scale this leaf's positional `channel` resolves with
    /// (see [`Self::x_scale`]). Color and size carry no such slot.
    pub(crate) fn set_scale(&mut self, channel: Channel, scale: crate::spec::encoding::ScaleSpec) {
        match channel {
            Channel::X => self.x_scale = Some(scale),
            Channel::Y => self.y_scale = Some(scale),
            Channel::Color | Channel::Size => {}
        }
    }
}
