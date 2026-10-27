//! Axis input (caller-supplied) and axis layout output (engine-computed).
//! Per spec §14.1: tick labels are caller-pre-computed via Phase 4 scales;
//! Phase 6 never touches scale internals.

use serde::{Deserialize, Serialize};

use super::geometry::{Axis1D, Rect};
use crate::scale::discrete::CategoricalPlacement;
use palette::Srgba;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisOrient {
    Top,
    Bottom,
    Left,
    Right,
}

/// The channel dimension an axis belongs to: `X` (horizontal, Top/Bottom edges)
/// or `Y` (vertical, Left/Right edges). 860: this names the x-vs-y distinction
/// that was previously recovered ad hoc from a concrete [`AxisOrient`] via
/// `matches!(.. Top | Bottom)`. The dimension is a property of the orient (orients
/// never cross dimensions — validated upstream in `prepare.rs`), so
/// [`AxisOrient::dimension`] is the single source for it and
/// [`AxisDimension::default_orient`] the single source for each dimension's
/// default edge. Byte-identical: the derived booleans/defaults are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisDimension {
    X,
    Y,
}

impl AxisDimension {
    /// The default axis edge for this dimension when no `orient` override is set:
    /// `Bottom` for x, `Left` for y (matching the historical `resolve_orient`
    /// default).
    pub(crate) fn default_orient(self) -> AxisOrient {
        match self {
            AxisDimension::X => AxisOrient::Bottom,
            AxisDimension::Y => AxisOrient::Left,
        }
    }

    /// This dimension's encoding-channel token, `"x"` or `"y"`.
    ///
    /// The one home for the token the axis-orient validators
    /// (`prepare::parse_axis_orient` / `parse_title_orient`) take, so a
    /// chart-level `orient`/`title_orient` can be checked against the axis's
    /// own dimension. #143 remediation: this replaces `config_apply`'s private
    /// `axis_channel`, which re-derived x-vs-y with the same open-coded
    /// `matches!(.. Top | Bottom)` that [`AxisOrient::dimension`] exists to
    /// own. Reached as `orient.dimension().channel_token()`.
    ///
    /// Note this is the axis's PHYSICAL dimension, never a user-written
    /// encoding channel that may have traveled through `CoordFlip`'s x/y swap
    /// — see the R3 exemption on `prepare::axis_style_fill_from`.
    pub(crate) fn channel_token(self) -> &'static str {
        match self {
            AxisDimension::X => "x",
            AxisDimension::Y => "y",
        }
    }
}

impl AxisOrient {
    /// The channel dimension this orient belongs to: Top/Bottom → X,
    /// Left/Right → Y. The single home for the x-vs-y inference that was open-coded
    /// as `matches!(.. Top | Bottom)` across the layout (860).
    pub(crate) fn dimension(self) -> AxisDimension {
        match self {
            AxisOrient::Top | AxisOrient::Bottom => AxisDimension::X,
            AxisOrient::Left | AxisOrient::Right => AxisDimension::Y,
        }
    }
}

/// Tick-label overlap strategy (B5 unit 6b: `fm.Axis(label_overlap=...)` /
/// `configure_axis(label_overlap=...)`). Maps the Vega-style values onto the
/// existing collision cascade (`cascade_collision_recovery`) primitives rather
/// than introducing a new collision engine.
///
/// `None` on [`AxisInput`] (the default) runs the unmodified cascade, so default
/// output is byte-identical. Only an explicit value changes behavior.
///
/// # Wire vocabulary
/// The wire token comes in as `chart_config::AxisStyleSpec::label_overlap: Option<String>`
/// and is mapped to this enum by the hand-written
/// [`parse_label_overlap`](crate::render::prepare::parse_label_overlap), whose
/// vocabulary is `"true"` → [`ShowAll`](Self::ShowAll), `"false"`/`"greedy"` →
/// [`Greedy`](Self::Greedy), `"parity"` → [`Parity`](Self::Parity), `"rotate"` →
/// [`Rotate`](Self::Rotate). That parser, NOT serde, is the entry point.
///
/// The `Serialize`/`Deserialize` derive with `rename_all = "lowercase"` therefore
/// uses a *different* vocabulary (`"showall"`/`"greedy"`/`"parity"`/`"rotate"`,
/// with no `"true"`/`"false"`) and is inert today: nothing on the wire path
/// (de)serializes this enum directly. Do not assume the serde names match the
/// parser tokens; if a future serde wire path is added, reconcile via per-variant
/// `#[serde(rename = "...")]` against the `parse_label_overlap` tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelOverlap {
    /// `true`: show ALL labels, skipping the overlap cull/elide stages. Labels
    /// may visibly overlap — the user's explicit choice.
    ShowAll,
    /// `false` / `"greedy"`: keep as many labels as fit without overlap (the
    /// cascade's default cull/decimation behavior, unchanged).
    Greedy,
    /// `"parity"`: show every other label (stride-2 decimation), reusing the
    /// cascade's culling with a fixed parity stride.
    Parity,
    /// `"rotate"`: force the cascade's rotate stage (steepest cascade angle).
    Rotate,
}

/// How much horizontal room the x tick-label cascade must leave between
/// adjacent VISIBLE labels, and whether it is allowed to drop labels to get
/// it (`#146`, spec §4.4).
///
/// Before this type the two questions were one `u32`
/// (`Theme(cull_threshold=N)`): `0` meant both "require no gap" and "never
/// cull", and any positive `N` meant both "require `N` px" and "culling
/// allowed". That conflation made "keep an `N` px gap driving the flat /
/// wrap / font-shrink / rotate stages, but never drop a label" unspellable.
/// The two fields are now independent, and all four combinations are
/// meaningful:
///
/// | `gap_px` | `cull` | Meaning |
/// |---|---|---|
/// | `0` | `false` | No gap requirement, never drop a label (legacy `cull_threshold=0`) |
/// | `N` | `true` | Require `N` px, drop labels if that is the only way (legacy `cull_threshold=N`) |
/// | `N` | `false` | **New:** require `N` px of every presentation stage (S0-S3), then keep every label regardless — WHICH stage they land in depends on density: sparse enough and the gap is satisfiable early (flat / shallow rotation); dense enough and the cascade falls through to rotated -90°, possibly overlapping (elision at that angle almost never has enough width to truncate, so it essentially never fires). Both arms pinned by `tests/test_label_gap_policy.py::test_cull_labels_false_stage_depends_on_density` |
/// | `0` | `true` | **New:** drop labels only where they genuinely overlap, with no extra gap |
///
/// # Public spelling
/// `Theme(cull_threshold=N)` sets the magnitude and, unless `cull_labels=` is
/// also given, the legacy mode along with it — see
/// [`from_cull_threshold`](Self::from_cull_threshold) and
/// [`with_overrides`](Self::with_overrides), the two places that mapping
/// lives. `Theme(cull_labels=False)` sets the mode alone.
///
/// # The default is deliberately two independent constants
/// [`DEFAULT_GAP_PX`](Self::DEFAULT_GAP_PX) and
/// [`DEFAULT_CULL`](Self::DEFAULT_CULL) are separate, so the open question of
/// what the default *magnitude* should be (`#134`) can be settled by editing
/// one number here, with no second change to the Theme key surface and no
/// coupled flip of the default mode. Deriving the default mode from the
/// default magnitude (`DEFAULT_GAP_PX > 0`) would have re-introduced exactly
/// the conflation this type removes, one level up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelGapPolicy {
    /// Minimum pixel gap required between adjacent visible x tick labels.
    gap_px: u32,
    /// Whether the cascade may drop labels (stride-cull) to satisfy the gap.
    cull: bool,
}

impl LabelGapPolicy {
    /// Default minimum gap, in pixels. `#134` owns this value and is still
    /// open; it can move without touching [`DEFAULT_CULL`](Self::DEFAULT_CULL),
    /// the Theme keys, or any call site.
    pub const DEFAULT_GAP_PX: u32 = 8;
    /// Whether the default policy may drop labels. Independent of
    /// [`DEFAULT_GAP_PX`](Self::DEFAULT_GAP_PX) on purpose.
    pub const DEFAULT_CULL: bool = true;

    /// The legacy `Theme(cull_threshold=N)` mapping, and the ONLY place the
    /// old coupling of magnitude to mode is written: `0` forbids culling,
    /// any positive value allows it. Keeping it in one named function is what
    /// makes "no existing spelling moves" checkable.
    pub const fn from_cull_threshold(gap_px: u32) -> Self {
        Self {
            gap_px,
            cull: gap_px > 0,
        }
    }

    /// Resolve the two independent `Theme` wire keys against `self` (the
    /// theme's current policy). `cull_threshold=` sets the magnitude;
    /// `cull_labels=` sets the mode and always wins. With `cull_threshold=`
    /// alone the mode follows [`from_cull_threshold`](Self::from_cull_threshold),
    /// so that spelling keeps its pre-split meaning exactly; with neither, the
    /// policy is unchanged.
    pub fn with_overrides(self, cull_threshold: Option<u32>, cull_labels: Option<bool>) -> Self {
        let legacy = cull_threshold
            .map(Self::from_cull_threshold)
            .unwrap_or(self);
        Self {
            gap_px: legacy.gap_px,
            cull: cull_labels.unwrap_or(legacy.cull),
        }
    }

    /// Minimum pixel gap the cascade must leave between adjacent visible
    /// labels. S0-S2 (flat, wrap, font-reduced) compare it against the
    /// overlap tolerance and keep whichever is larger instead of stacking
    /// both (`collision_need`); S3 (rotate) subtracts it directly. S5's
    /// elision fallback is deliberately exempt (see its fit test's own
    /// comment). See [`cascade_collision_recovery`]'s doc.
    pub const fn gap_px(&self) -> u32 {
        self.gap_px
    }

    /// The gap as a raw `f64` pixel value, read by `collision_need` (S0-S2)
    /// and subtracted directly by S3.
    fn gap(&self) -> f64 {
        self.gap_px as f64
    }

    /// Whether the cascade may drop labels (S4) to satisfy the gap. `false`
    /// sends a still-colliding axis to S5 (elision, which leaves an
    /// infeasible truncation intact) instead.
    pub const fn may_cull(&self) -> bool {
        self.cull
    }
}

impl Default for LabelGapPolicy {
    fn default() -> Self {
        Self {
            gap_px: Self::DEFAULT_GAP_PX,
            cull: Self::DEFAULT_CULL,
        }
    }
}

/// The three values [`cascade_collision_recovery`]'s cull-by-stride paths
/// read off `AxisInput` and the axis's own projection, bundled so they
/// travel together into one parameter instead of three positional ones a
/// caller could transpose.
#[derive(Debug, Clone, Copy)]
struct TickCullFrame<'a> {
    /// Which tick positions sit on an explicit value anchor
    /// (`AxisInput.tick_anchors`): `anchored_visibility`/`cull_by_stride`
    /// phase a stride's kept ticks onto those anchors rather than an
    /// arbitrary offset.
    anchors: &'a [bool],
    /// The stride phase ticks are counted from (`AxisInput.tick_origin`, or
    /// `0` when the axis has none).
    origin: usize,
    /// Whether a scale projects this axis at all: gates stage D's
    /// continuous-x thinning, which a categorical or scale-less axis skips.
    continuous: bool,
}

/// A cascade slot paired with the bit that records whether a higher-precedence
/// source has already spoken for it.
///
/// Two axis slots need this, for the identical reason: their EMPTY value is
/// ambiguous. [`AxisInput::title`]'s `None` means BOTH "never set" and
/// "explicitly suppressed via the `title=\"\"` sentinel", and
/// [`LabelFormat::spec`]'s `None` means BOTH "no per-channel format" and "a
/// per-channel temporal format already fully baked into the tick labels". A
/// chart-level fill that tests only for emptiness therefore cannot tell
/// "unclaimed" from "claimed and deliberately empty", and resurrects a value
/// the caller removed — inverting the per-channel-wins cascade (D8/D12, spec
/// §4.9). Both bugs were live and both were fixed by a companion `bool`.
///
/// The bit now travels INSIDE this type rather than as a sibling
/// `*_claimed: bool` on [`AxisStyleOverrides`], which is what the two slots
/// used to carry. What that buys, stated as what the type actually enforces
/// rather than as impossibility: the bit is **private**, so it can only be read
/// through [`is_claimed`](Self::is_claimed) and only written through
/// [`claim`](Self::claim) or one of the two fill methods, each of which owns one
/// precedence discipline end to end; and it **travels with the value it
/// guards**, which the old `AxisStyleOverrides::title_claimed` did not — it sat
/// in a different struct from `AxisInput::title`.
///
/// It does not make a third parallel boolean unrepresentable: nothing stops a
/// future `pub foo_claimed: bool` on a cluster struct. What it does is make the
/// disciplined shape the obvious one and the ad-hoc one visibly ad-hoc, by
/// giving the pattern a name and one place to put the predicate.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Claimed<T> {
    value: T,
    claimed: bool,
}

impl<T> Claimed<T> {
    /// An unclaimed slot holding `value`.
    pub(crate) const fn new(value: T) -> Self {
        Self {
            value,
            claimed: false,
        }
    }

    /// Read the slot.
    pub(crate) const fn get(&self) -> &T {
        &self.value
    }

    /// Overwrite the slot, leaving the claim bit alone. Two production
    /// disciplines use it, and neither wants the bit touched:
    ///
    /// - **Fresh build.** The per-channel writer
    ///   (`render::prepare::build_axis_input`) starts from `Default` and owns
    ///   every slot it writes, recording the claim separately via
    ///   [`claim`](Self::claim) because "a per-channel source spoke" and "the
    ///   value it resolved to" are independent facts.
    /// - **Layout-time suppression on a throwaway clone.**
    ///   `layout::compute_layout` strips the axis title from a *cloned*
    ///   `AxisInput` for non-bottom-row / non-leftmost-column facet panels. The
    ///   bit is left alone because nothing will ever fill that clone again, so
    ///   the claim is meaningless on it — and carrying the original's bit
    ///   across is exactly what the pre-`Claimed` `modified.title = None` did
    ///   by construction. Numerically this is the majority: two of the three
    ///   production call sites.
    ///
    /// A **chart-level** writer must not reach for this. Those go through
    /// [`fill_and_claim`](Self::fill_and_claim) or
    /// [`fill_without_claiming`](Self::fill_without_claiming), which own the
    /// two precedence disciplines.
    pub(crate) fn set(&mut self, value: T) {
        self.value = value;
    }

    /// Record that a per-channel source (or an earlier, more specific
    /// chart-level layer) owns this slot, whatever value it resolved to —
    /// including an explicitly empty one, which is the whole point.
    pub(crate) fn claim(&mut self) {
        self.claimed = true;
    }

    /// Whether a higher-precedence source has spoken for this slot. The one
    /// read of the bit; both fill methods below gate on it through here rather
    /// than touching the field, so "is this slot claimed" has a single spelling.
    pub(crate) const fn is_claimed(&self) -> bool {
        self.claimed
    }

    /// Chart-level write, **claiming** discipline: `fill` sees the slot only
    /// while it is unclaimed, and the slot becomes claimed exactly when `fill`
    /// reports that it took it. Claiming on the way out is what stops the
    /// next, less specific chart-level layer from overwriting the value this
    /// one resolved.
    ///
    /// Used by [`AxisInput::fill_chart_level_title`]. Deliberately distinct
    /// from [`fill_without_claiming`](Self::fill_without_claiming): the two
    /// slots' disciplines genuinely differ, and unifying them would move
    /// output. See that method's doc for the other half.
    pub(crate) fn fill_and_claim(&mut self, fill: impl FnOnce(&mut T) -> bool) {
        if self.is_claimed() {
            return;
        }
        if fill(&mut self.value) {
            self.claim();
        }
    }

    /// Chart-level write, **non-claiming** discipline: `fill` sees the slot
    /// only while it is unclaimed, and the bit is never set.
    ///
    /// Used by [`AxisStyleOverrides::fill_chart_level_label_format`]. The
    /// format slot needs no claim on the way out because its own non-emptiness
    /// already stops the next layer — that fill's predicate is "empty AND
    /// unclaimed", so a filled slot is as good as a claimed one.
    ///
    /// **Claiming here would move output today, on a reachable public
    /// spelling** — the hazard to weigh before "simplifying" the two
    /// disciplines into one.
    /// `config_apply::fill_axis_slots_specific_before_shared` runs
    /// `apply_axis_config_to_axis_input` TWICE per axis: the specific
    /// `axis_x`/`axis_y` layer, then the shared `axis` layer. That function
    /// calls the format fill unconditionally for every config section present,
    /// with a format that may be `None`. So
    /// `configure(axis_x=AxisConfig(label_angle=45))` plus
    /// `configure_axis(label_format=".2f")` reaches this method first with
    /// `(None, None)`: a claim taken on that empty fill would block the shared
    /// layer and drop the `.2f` silently. Claiming would ALSO be a lie —
    /// asserting a per-channel claim where there was only a chart-level fill —
    /// but the dropped format is the load-bearing reason, not the lie.
    ///
    /// Which is to say the separation is right in BOTH directions: claiming on
    /// the title slot is what stops a resurrected suppressed title
    /// ([`fill_and_claim`](Self::fill_and_claim)), and NOT claiming on the
    /// format slot is what stops a dropped chart-level format.
    ///
    /// Pinned by
    /// `render::config_apply::tests::chart_config_application_tests::shared_axis_label_format_survives_a_specific_axis_x_layer_without_one`,
    /// which drives the real two-layer cascade over the literal wire dict that
    /// spelling emits. Adding the claim here fails that test **and nothing else
    /// in the crate or the Python suite** — which is why the pin exists rather
    /// than this comment alone. (The measured counts behind that sentence are
    /// dated in the task report; they are deliberately not repeated here,
    /// because a test count embedded in a doc comment is wrong by the next
    /// commit.)
    pub(crate) fn fill_without_claiming(&mut self, fill: impl FnOnce(&mut T)) {
        if !self.is_claimed() {
            fill(&mut self.value);
        }
    }
}

/// The tick-label format slot: the d3/strftime spec string and its format-type
/// tag, which only ever move together (every writer sets both, and the two are
/// meaningless apart).
///
/// Both halves stay independently `Option` rather than collapsing into one
/// `Option<(String, Option<String>)>`, because `spec: None` with
/// `format_type: Some("time")` is a LIVE state, not a nonsense one:
/// `render::config_apply::apply_label_format_to_axis`'s `tick_values` branch
/// passes the type through to `prepare::apply_tick_format` with no spec, which
/// formats the epoch-ms tick values with the default time formatter
/// (`format_time(ms, 86_400_000)`) rather than passing them through.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct LabelFormat {
    /// d3-format string for tick labels (per-channel `label_format` /
    /// chart-level `label_format_raw`). Applied after
    /// [`AxisTickOverrides::values`].
    pub spec: Option<String>,
    /// [`spec`](Self::spec)'s format-type tag (`"time"`/`"number"`/`None`),
    /// CHART-LEVEL ONLY (D8, spec §4.5) — mirrors `spec`'s own
    /// chart-level-only ownership: the per-channel path resolves and applies
    /// its own format_type immediately during
    /// `prepare::build_axis_tick_inputs`'s `Thread` mode (a genuinely
    /// temporal per-channel format is applied there via raw epoch-ms values
    /// before this bundle is ever consulted, so a per-channel THREADED
    /// `spec` — the only kind that reaches this bundle — is always
    /// numeric by construction and never needs this field). Read by
    /// `render::config_apply::apply_label_format_to_axis` to decide whether the deferred
    /// `spec` string is a `chrono` strftime pattern or a d3 spec.
    pub format_type: Option<String>,
}

/// Tick marks and the values they sit on.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisTickOverrides {
    /// Explicit tick values (`fm.Axis(values=[...])` / `configure_axis(tick_values=[...])`).
    /// When set, tick labels are replaced with formatted versions of these values.
    pub values: Option<Vec<f64>>,
    /// Append a tick at each domain boundary (scale min/max) if not already
    /// present. `None`/`false` → no boundary ticks.
    pub extra: Option<bool>,
    /// Minimum step (data units) between generated ticks; ticks closer than this
    /// in data space are dropped. `None` → no thinning.
    pub min_step: Option<f64>,
    /// Tick-mark length (px) for this axis. `None` → `theme.sizes.tick_size`.
    /// Per-axis rather than theme-global as of D12 (spec §4.9): it was the
    /// LAST `AxisStyleSpec` field whose only carrier was the shared theme, so
    /// `configure(axis_x=AxisConfig(tick_size=…))` had nowhere to land but a
    /// global slot the y axis reads too — the gap Python's
    /// `_redistribute_general_axis` was papering over.
    pub size: Option<f64>,
    /// Whether tick marks render. Same cascade as
    /// [`AxisLabelOverrides::show`]. `None` → `true`.
    pub show: Option<bool>,
}

/// Tick labels: their text, their formatting, and how the collision cascade
/// may treat them.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisLabelOverrides {
    /// Tick-label rotation angle override (`fm.Axis(label_angle=...)`). Bypasses
    /// the collision cascade when `Some`. `None` → cascade-resolved angle.
    pub angle: Option<f64>,
    /// The tick-label format and its claim bit. See [`LabelFormat`] for the
    /// two halves and [`Claimed`] for why the claim cannot be dropped:
    /// `spec.is_none()` alone CANNOT distinguish "no per-channel format
    /// claimed this axis" from "a per-channel format already fully applied",
    /// and missing that distinction (D8) let a chart-level time preset win the
    /// cascade over an explicit per-channel one
    /// (`configure_axis(label_format="date_iso")` re-deriving RAW temporal
    /// values and overwriting an already-formatted `fm.Axis(label_format="%b
    /// %d")` axis — the batch's hard per-channel-wins cascade constraint,
    /// violated). The claim is taken by the per-channel prepare path
    /// (`prepare::build_axis_input`) whenever `resolve_axis_label_format` found
    /// an explicit per-channel format, regardless of whether that format ended
    /// up THREADED onto `spec` (the numeric case) or applied EAGERLY to
    /// `AxisInput.tick_labels` and then discarded (the temporal case).
    pub format: Claimed<LabelFormat>,
    /// Pixel gap between the end of a tick mark and the tick label baseline.
    /// `None` → the renderer's hardcoded per-orient gaps.
    pub padding: Option<f64>,
    /// Tick-label color override. `None` → `theme.colors.label_color`.
    pub color: Option<Srgba<u8>>,
    /// Tick-label font-size override. `None` → `theme.typography.label_font_size`.
    pub font_size: Option<f64>,
    /// Flush the first/last tick labels at the axis ends so edge labels align
    /// within the plot bounds instead of overflowing (Vega `labelFlush`).
    /// `None`/`false` → default anchors (byte-identical).
    pub flush: Option<bool>,
    /// Tick-label overlap strategy override. `None` → the unmodified collision
    /// cascade on x, and no overlap policy at all on y (see
    /// [`resolve_y_overlap`](crate::layout::axis::resolve_y_overlap) for y's
    /// transpose of the four primitives).
    pub overlap: Option<LabelOverlap>,
    /// Whether tick labels render. `None` → `true`. Per-channel
    /// `fm.Axis(labels=)` writes it unconditionally; chart-level
    /// `configure_axis(labels=)` fills it only when still `None`, so
    /// per-channel wins (D12, spec §4.9).
    pub show: Option<bool>,
}

/// Axis-title styling. The title TEXT is not here: it is the resolved render
/// input [`AxisInput::title`], mirroring how the resolved
/// [`AxisInput::orient`] sits apart from its [`AxisPlacementOverrides::orient`]
/// override input. Its claim bit rides with the text, in the `Claimed` wrapper
/// on that field.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisTitleOverrides {
    /// Axis title font size. `None` → `theme.title_font_size`.
    pub font_size: Option<f64>,
    /// Axis title color. `None` → `theme.title_color`.
    pub color: Option<Srgba<u8>>,
    /// Padding between axis title and tick labels. `None` → `theme.axis_title_padding`.
    pub padding: Option<f64>,
    /// Side/orientation of the axis title relative to its axis (e.g. a horizontal
    /// title on a left axis). `None` → the orient-default rotation.
    pub orient: Option<AxisOrient>,
}

/// Gridlines emitted by this axis.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisGridOverrides {
    /// Gridline color override. `None` → `theme.colors.grid_color`.
    pub color: Option<Srgba<u8>>,
    /// Gridline dash override. `None` → `theme.grid.grid_dash`.
    pub dash: Option<Vec<f64>>,
    /// Gridline width override. `None` → `theme.sizes.grid_width`.
    pub width: Option<f64>,
    /// Per-axis grid-line opacity override `[0, 1]`. `None` → `theme.grid.grid_opacity`.
    pub opacity: Option<f64>,
    /// Whether gridlines from this axis render. `None` means "no per-axis
    /// opinion" and is resolved against `theme.grid.grid` by
    /// [`AxesInput::apply_show_defaults`] — the per-axis half of D4/F-L07-01
    /// (spec §4.3). Precedence, not AND: a per-channel `Axis(grid=True)` on a
    /// grid-less theme draws gridlines for that axis only.
    pub show: Option<bool>,
}

/// The axis domain line.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisDomainOverrides {
    /// Domain-line color override. `None` → `theme.colors.axis_line_color`.
    pub color: Option<Srgba<u8>>,
    /// Domain-line width override. `None` → `theme.sizes.axis_line_width`.
    pub width: Option<f64>,
    /// Whether the axis domain line renders. `None` means "no per-axis
    /// opinion" and is resolved against `theme.axis.axis_line` by
    /// [`AxesInput::apply_show_defaults`], so by layout time this is always
    /// `Some`. Precedence (not AND): a per-channel `Axis(domain=True)` wins
    /// over a theme that disables domain lines.
    pub show: Option<bool>,
}

/// Where the axis group sits and in what order it paints.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisPlacementOverrides {
    /// Axis side override, validated against the channel dimension (x→top/bottom,
    /// y→left/right). `None` → the channel's default side (Bottom for x, Left for
    /// y). Resolved into the concrete [`AxisInput::orient`] at layout-build time.
    pub orient: Option<AxisOrient>,
    /// Shift the axis group perpendicular to its line by N px (outward positive),
    /// composing additively with the renderer's `offset` handling. `None` → no shift.
    pub translate: Option<f64>,
    /// Shift the axis perpendicular AWAY from the plot edge by N px (Vega axis
    /// `offset`). Composes **additively** with [`translate`](Self::translate):
    /// the renderer applies `translate + offset` as a single outward shift.
    /// `None`/`0` → no shift.
    pub offset: Option<f64>,
    /// Lower bound (px) for the reserved axis margin band — reserve at least this
    /// much. `None` → dynamic band only.
    pub min_band: Option<f64>,
    /// Upper bound (px) for the reserved axis margin band — cap at this much
    /// (labels may clip past it). `None` → no cap.
    pub max_band: Option<f64>,
    /// Coarse draw order relative to marks: `>= 1` → axis + grid drawn above
    /// marks; `<= 0` (default) → below marks. `None` → default (below).
    pub zindex: Option<i64>,
}

/// Per-axis style/positioning overrides, bundled so a future field is a one-line
/// struct addition instead of a five-site thread (mirrors `LegendStyleOpts` on
/// the legend side). Every leaf is `Option` and falls back to the shared theme
/// value (or a layout default) when `None`, so default output stays byte-identical
/// and only a per-channel `fm.Axis(...)` / chart-level `configure_axis(...)` spec
/// lights one up.
///
/// The leaves live in six clusters named for the axis part they style — ticks,
/// labels, title, grid, domain, placement — rather than as one flat run of 34
/// prefix-named fields. The prefixes WERE the clustering, spelled in a
/// convention the compiler could not hold anyone to; a new field's home is now
/// a decision the type makes.
///
/// Both the per-channel parse path (`render::prepare::encoding_axis_style_overrides`)
/// and the chart-level apply path (`render::config_apply::apply_axis_style_to_axis_input`) write
/// here uniformly via the `is_none()` fill-only pattern, so a higher-precedence
/// source (per-channel, or an earlier config layer) always wins. The resolved
/// concrete axis side is computed from [`AxisPlacementOverrides::orient`] into
/// [`AxisInput::orient`] at layout-build time (x→Bottom / y→Left default).
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisStyleOverrides {
    /// Tick marks and the values they sit on.
    pub ticks: AxisTickOverrides,
    /// Tick labels: text, formatting, collision policy.
    pub labels: AxisLabelOverrides,
    /// Axis-title styling (the title TEXT is [`AxisInput::title`]).
    pub title: AxisTitleOverrides,
    /// Gridlines emitted by this axis.
    pub grid: AxisGridOverrides,
    /// The axis domain line.
    pub domain: AxisDomainOverrides,
    /// Where the axis group sits and in what order it paints.
    pub placement: AxisPlacementOverrides,
}

impl AxisStyleOverrides {
    /// The SOLE way a chart-level source may fill the tick-label format
    /// ([`AxisLabelOverrides::format`]): no-ops when the slot is already
    /// claimed (by a per-channel format — see [`Claimed`]'s doc — or by an
    /// earlier chart-level layer that already filled the spec).
    ///
    /// The `spec.is_none() && !claimed` predicate this method owns used to be
    /// hand-copied at its two call sites
    /// (`render::config_apply::apply_axis_config_to_axis_input`, `prepare::axis_style_fill_from`'s
    /// own defensive fallback write) — the SAME "is this slot actually
    /// unclaimed" question, expressed twice, on a `pub(crate)` field with no
    /// accessor discipline. A fix for the cascade-inversion bug this
    /// predicate exists to enforce was itself found missing the second
    /// writer, proving the hand-copy was already load-bearing and already
    /// fragile. Mechanizing it here means a THIRD writer of the format slot
    /// has exactly one correct way to write it, not a predicate to
    /// remember and re-derive.
    pub(crate) fn fill_chart_level_label_format(
        &mut self,
        fmt: Option<String>,
        format_type: Option<String>,
    ) {
        self.labels.format.fill_without_claiming(|slot| {
            if slot.spec.is_none() {
                slot.spec = fmt;
                slot.format_type = format_type;
            }
        });
    }
}

/// The axis-title 3-way resolution shared by the per-channel builder
/// (`prepare::resolve_axis_title`) and the chart-level fill
/// ([`AxisInput::fill_chart_level_title`]):
///   - absent (`None`)             → keep `fallback`;
///   - present + empty/whitespace  → explicit suppress, yields `None`;
///   - present + non-empty         → use it verbatim.
///
/// The empty-string suppression sentinel comes from Python forwarding
/// `title = ""` only when `title=None` was explicitly passed.
pub(crate) fn resolve_axis_title(value: Option<&str>, fallback: Option<String>) -> Option<String> {
    match value {
        Some(s) if s.trim().is_empty() => None,
        Some(s) => Some(s.to_owned()),
        None => fallback,
    }
}

/// Caller-supplied per-axis input. Phase 6 takes both x and y always.
///
/// The render-input data (orient, title, labels, show toggles, formats, tick
/// projection) lives flat; the per-axis style/positioning overrides are bundled
/// in [`overrides`](Self::overrides) (see [`AxisStyleOverrides`]).
#[derive(Debug, Clone, PartialEq)]
pub struct AxisInput {
    /// Resolved concrete axis side, used by layout (plot-rect sizing,
    /// `layout_*_axis`) and the scene assembler. Resolved from
    /// `overrides.placement.orient` (default Bottom for x, Left for y) via
    /// [`resolve_orient`](Self::resolve_orient) after all override layers merge.
    pub orient: AxisOrient,
    /// Resolved axis title, or `None` for "no title" (either never set, or
    /// explicitly suppressed via the `title=""` sentinel). Which of the two it
    /// is matters for the chart-level cascade, so the slot carries its own
    /// claim bit — see [`Claimed`] and [`AxisInput::fill_chart_level_title`].
    /// The claim used to live across the struct boundary as
    /// `AxisStyleOverrides::title_claimed`, a bit guarding a value it could not
    /// see; the wrapper puts the two back together.
    pub title: Claimed<Option<String>>,
    pub tick_labels: Vec<String>,
    /// Which of `tick_labels` are `Log` decades — in the
    /// same index order, from `ScaleKind::tick_anchors`. Empty for every
    /// non-`Log` axis (and for a `Log` axis whose breaks are all decades,
    /// no multiples in play): `layout_x_axis`'s collision cascade reads it
    /// through `anchored_visibility`, which reduces to plain
    /// `stride_visibility` when empty, so every existing caller that leaves
    /// this at its `AxisInput::new` default (`Vec::new()`) culls
    /// byte-identically to before this field existed.
    pub tick_anchors: Vec<bool>,
    /// The index, among `tick_labels`, of this axis's nicest tick — from
    /// `ScaleKind::tick_origin`, index-aligned with `tick_labels` the same
    /// way `tick_anchors` is. `None` for a categorical/scale-less axis, a
    /// temporal or `Log` scale (see `tick_origin`'s doc), and for fewer
    /// than two ticks. The collision cascade's continuous-x stride culls
    /// (`cascade_collision_recovery`'s stage D, S4, and the band-budget
    /// cull) count strides from this index in both directions instead of
    /// always from index 0, so a thinning keeps the tick a reader expects
    /// (zero when the axis has one, else the coarsest round number)
    /// present, rather than whichever tick index 0 happens to be. `None`
    /// falls back to counting from index 0, exactly
    /// as an empty `tick_anchors` falls back to plain `stride_visibility`.
    pub tick_origin: Option<usize>,
    /// Optional d3-format string applied to each tick label before layout
    /// (D12: `encoding.format` on x/y axes). `None` → use the scale's own
    /// default formatter (existing behavior).
    pub tick_format: Option<String>,
    /// When `Some("time")`, `tick_format` is a time format spec (D12:
    /// `encoding.format_type`). Currently unused by `layout_x_axis` /
    /// `layout_y_axis` — tick strings are already pre-formatted before this
    /// struct is built. Reserved for future granularity hints.
    pub tick_format_type: Option<String>,
    /// Continuous-axis scale projection (continuous-axis tick design,
    /// 2026-05-30). `Some` for continuous (linear/log/pow/symlog/time) axes;
    /// `None` for categorical/discretizing (ordinal) axes, which keep the
    /// uniform-slot placement byte-identically. Presence of this field — not the
    /// scale type — drives the placement branch. See [`TickProjection`].
    pub tick_projection: Option<TickProjection>,
    /// How a **categorical** axis places its categories: the resolved ordinal
    /// scale's own band geometry, so tick labels and grid lines land on the
    /// pixels the marks land on (spec §7, F-L04-03/GH #67). `Some` for every
    /// ordinal positional scale — carrying the d3 model with either its
    /// chart-absolute pixel range (a user-supplied `range=`, re-anchored onto
    /// each facet panel by layout) or no range, to resolve against this
    /// panel's rect (the panel-extent fallback); see [`CategoricalPlacement`].
    ///
    /// `None` for continuous axes, which place through
    /// [`tick_projection`](Self::tick_projection), and for the scale-less
    /// [`compute_layout`](crate::layout::binding::compute_layout) binding, which
    /// is handed bare tick-label strings and has no scale to ask — the one
    /// remaining consumer of the uniform-slot fallback. Mutually exclusive with
    /// `tick_projection`: an ordinal scale yields no projection. Unlike
    /// projected fractions, the resolved centers are absolute pixels used
    /// **directly** by `layout_*_axis`, not mapped through the panel-extent
    /// padding inset.
    ///
    /// The mutual exclusivity with `tick_projection`, and this vec's 1:1
    /// pairing with `tick_labels`, are enforced by
    /// [`debug_assert_placement_invariants`](Self::debug_assert_placement_invariants)
    /// at both `layout_*_axis` entry points (archaeology BR-3 asked for exactly
    /// this hardening).
    pub(crate) categorical_placement: Option<CategoricalPlacement>,
    /// Per-axis style/positioning overrides (B5). See [`AxisStyleOverrides`].
    pub overrides: AxisStyleOverrides,
}

/// Continuous-axis scale projection carried by [`AxisInput`]. Groups the three
/// projection inputs that share the `padding_frac` inset invariant: a tick at
/// domain value `v` and a data mark at value `v` land on the same pixel because
/// both are mapped through `inset_pixel_range(base_range, padding_frac)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TickProjection {
    /// The padding fraction the resolved positional scale used. Layout insets
    /// the panel mark range by this fraction (capped at `SCALE_PADDING_MAX_PX`)
    /// before interpolating `major`/`minor`, reproducing the mark inset exactly.
    pub padding_frac: f64,
    /// One **domain fraction** `t ∈ [0, 1]` per major tick label, in the same
    /// index order as `AxisInput.tick_labels`: the scale's normalized projection
    /// of the tick value, independent of the pixel range.
    pub major: Vec<f64>,
    /// Minor tick positions as per-minor **domain fractions in `[0, 1]`** — the
    /// same projection that produces `major`, applied to the scale's minor
    /// ticks. Layout maps each onto the panel via the *same* padding inset used
    /// for majors and data marks, so a minor at domain `v` coincides with the
    /// major projection of `v`. Empty when minor gridlines are disabled
    /// (`theme.grid.minor` off) — `prepare.rs` only populates it when the gate is
    /// on, and "minor enabled" is derived from this vec being non-empty.
    pub minor: Vec<f64>,
    /// The `Mark::Text` ink demand this axis carries (spec §6.13), threaded
    /// through so `inset_pixel_range(base_range, padding_frac, text_demand)`
    /// builds the SAME inset here that places the marks. Empty for every
    /// axis no `Mark::Text` layer draws, which keeps `padding_frac`/`major`/
    /// `minor`'s plain inset byte-identical on every chart without text.
    pub text_demand: crate::layout::geometry::TextEdgeDemand,
}

impl AxisInput {
    /// Construct an `AxisInput` with all overrides empty (so every show toggle
    /// reads its default: `true` for labels/ticks, the theme's value for
    /// domain/grid once [`AxesInput::apply_show_defaults`] has run).
    /// `label_angle` seeds `overrides.labels.angle`.
    pub fn new(
        orient: AxisOrient,
        title: Option<String>,
        tick_labels: Vec<String>,
        label_angle: Option<f64>,
    ) -> Self {
        Self {
            orient,
            title: Claimed::new(title),
            tick_labels,
            tick_anchors: Vec::new(),
            tick_origin: None,
            tick_format: None,
            tick_format_type: None,
            tick_projection: None,
            categorical_placement: None,
            overrides: AxisStyleOverrides {
                labels: AxisLabelOverrides {
                    angle: label_angle,
                    ..AxisLabelOverrides::default()
                },
                ..AxisStyleOverrides::default()
            },
        }
    }

    /// The one place `tick_labels`, `tick_anchors` and `tick_origin` are
    /// replaced together: a producer that builds a fresh label
    /// ladder for an axis calls this instead of assigning the fields
    /// separately, so the three can never drift out of lockstep by one
    /// assignment being added without the others. No panic/`debug_assert` on
    /// a mismatch — `anchors` degrades to empty
    /// when its length disagrees with `labels`', the same "no anchors"
    /// sentinel [`anchored_visibility`] already treats as "fall back to
    /// plain `stride_visibility`", so a producer bug here is silently safe
    /// rather than ever observably misaligned; `origin` degrades to `None`
    /// (fall back to counting from index 0) the same way when it names an
    /// out-of-range index.
    ///
    /// `adjust_axis_ticks`'s `tick_min_step`/`tick_extra` passes and
    /// `config_apply::sync_tick_placement_to_tick_values`'s `tick_values=`
    /// resync are CORRECTIVE updates to an already-paired axis (thinning or
    /// re-deriving `tick_anchors`/`tick_origin` from the SAME values that
    /// just changed `tick_labels`), not a fresh replacement, so they keep
    /// mutating the fields directly in their own lockstep.
    pub(crate) fn set_ticks(&mut self, labels: Vec<String>, anchors: Vec<bool>, origin: Option<usize>) {
        let anchors = if anchors.is_empty() || anchors.len() == labels.len() { anchors } else { Vec::new() };
        let origin = origin.filter(|&o| o < labels.len());
        self.tick_labels = labels;
        self.tick_anchors = anchors;
        self.tick_origin = origin;
    }

    /// Assert the two tick-placement invariants `layout_*_axis` relies on.
    ///
    /// **One carrier.** An axis carries a continuous projection or a
    /// categorical placement, never both. The two describe the same thing
    /// (where tick `i` goes) in incompatible terms, and `layout_*_axis`
    /// resolves the conflict by silently preferring the projection — so a
    /// producer that set both would get a plausible layout with the wrong
    /// ticks. The exclusivity is structural (a resolved scale is continuous or
    /// ordinal, and `tick_fractions` is empty for the ordinal kinds).
    ///
    /// **Paired lengths.** The categorical carrier is indexed by tick position,
    /// so it must hold exactly one center per entry in `tick_labels`. It is
    /// sized by the scale's DOMAIN, while `tick_labels` can be replaced
    /// wholesale by an explicit `tick_values` override of any length — the
    /// staleness `config_apply::sync_tick_placement_to_tick_values` clears the
    /// carrier for. Asserting it here is the canary for a future producer that
    /// forgets to (a stale carrier used to index out of bounds and abort the
    /// render across the PyO3 boundary; `paired_band_centers` is the release-
    /// build half of the same guard).
    ///
    /// Both are bug canaries, not runtime branches: debug builds trip, release
    /// builds pay nothing.
    fn debug_assert_placement_invariants(&self) {
        debug_assert!(
            !(self.tick_projection.is_some() && self.categorical_placement.is_some()),
            "an axis cannot carry both a continuous tick projection and a categorical placement"
        );
        debug_assert!(
            self.categorical_placement
                .as_ref()
                .is_none_or(|p| p.len() == self.tick_labels.len()),
            "categorical placement holds {:?} centers for {} tick labels — layout indexes the \
             centers by tick position, so they must be paired",
            self.categorical_placement.as_ref().map(|p| p.len()),
            self.tick_labels.len()
        );
    }

    /// This axis's tick-mark length: its own override, else the theme's.
    /// One accessor rather than an `unwrap_or` repeated at each of the six
    /// consumers (band reservation, the two `layout_*_axis` calls, the
    /// secondary-y call, and `build_axis`).
    pub fn tick_size(&self, theme: &super::ThemeInputs) -> f64 {
        self.overrides.ticks.size.unwrap_or(theme.sizes.tick_size)
    }

    /// Whether tick labels render (`overrides.labels.show`, default `true`).
    pub fn show_labels(&self) -> bool {
        self.overrides.labels.show.unwrap_or(true)
    }

    /// Whether tick marks render (`overrides.ticks.show`, default `true`).
    pub fn show_ticks(&self) -> bool {
        self.overrides.ticks.show.unwrap_or(true)
    }

    /// Whether the axis domain line renders. `None` here means the theme
    /// default has not been folded in yet ([`AxesInput::apply_show_defaults`]);
    /// `true` is that theme field's own default, so an un-resolved axis reads
    /// the same as an unconfigured one.
    ///
    /// The `None` fallback below is a real, load-bearing runtime behavior
    /// (an axis this function is called on before `apply_show_defaults` has
    /// run must still return something), so the precondition is enforced
    /// only in debug builds rather than folded into the fallback's meaning:
    /// `AxesInput::apply_show_defaults` is the sole production caller
    /// obligated to fill this slot before `AxisLayout::from_input` reads it
    /// (in turn consumed by `render::marks::axis::build_axis`/`build_grid`),
    /// and any call site that reaches here with the slot still unresolved is
    /// a bug the `true` default was silently masking.
    pub fn show_domain(&self) -> bool {
        debug_assert!(
            self.overrides.domain.show.is_some(),
            "AxisInput::show_domain() read before AxesInput::apply_show_defaults() \
             folded in the theme default"
        );
        self.overrides.domain.show.unwrap_or(true)
    }

    /// Whether gridlines from this axis render. Same "resolved by
    /// [`AxesInput::apply_show_defaults`], defaults to the theme field's own
    /// default" contract — and the same debug-only precondition assert — as
    /// [`show_domain`](Self::show_domain).
    pub fn show_grid(&self) -> bool {
        debug_assert!(
            self.overrides.grid.show.is_some(),
            "AxisInput::show_grid() read before AxesInput::apply_show_defaults() \
             folded in the theme default"
        );
        self.overrides.grid.show.unwrap_or(true)
    }

    /// The SOLE way a chart-level source may fill [`title`](Self::title):
    /// no-ops once the slot is claimed (by a per-channel title — including an
    /// explicit `title=""` suppression — or by an earlier, more specific
    /// chart-level layer), and claims it on the way out so the next, less
    /// specific layer cannot overwrite it.
    ///
    /// `spec_title` carries the raw wire value, so the same empty-string
    /// suppress sentinel the per-channel path honors works at chart level too:
    /// `configure_axis(title="")` removes the field-name default rather than
    /// rendering an empty title. Mirrors
    /// [`AxisStyleOverrides::fill_chart_level_label_format`] field for field —
    /// same hazard (a slot whose `None` is ambiguous), same one-owner fix. The
    /// two disciplines still differ in one respect, and deliberately so: this
    /// one CLAIMS on the way out (see
    /// [`Claimed::fill_and_claim`] versus
    /// [`Claimed::fill_without_claiming`]).
    ///
    /// An absent `spec_title` declines the slot rather than claiming it empty:
    /// `configure_axis()` with no `title=` must leave the next, less specific
    /// layer free to fill it.
    pub(crate) fn fill_chart_level_title(&mut self, spec_title: Option<&str>) {
        self.title.fill_and_claim(|slot| match spec_title {
            Some(raw) => {
                *slot = resolve_axis_title(Some(raw), slot.take());
                true
            }
            None => false,
        });
    }

    /// Re-resolve the concrete [`orient`](Self::orient) from the
    /// `overrides.placement.orient` override after all override layers (per-channel,
    /// chart-level config) have merged. The override (when `Some`) is already
    /// validated against the channel dimension, so the channel is inferred from
    /// the current concrete orient and the default is its dimension edge (Bottom
    /// for x, Left for y). Idempotent.
    pub(crate) fn resolve_orient(&mut self) {
        // 860: the channel dimension is carried by the orient itself
        // (`AxisOrient::dimension`); its default edge is the single source in
        // `AxisDimension::default_orient` (Bottom for x, Left for y) — no inline
        // `matches!(.. Top | Bottom)` discipline.
        let default = self.orient.dimension().default_orient();
        self.orient = self.overrides.placement.orient.unwrap_or(default);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxesInput {
    pub x: AxisInput,
    pub y: AxisInput,
    /// Whether the Cartesian x axis is DRAWN — the one question every
    /// reservation and every emission in this module asks. False suppresses the
    /// axis line + ticks + labels + title at layout time AND reserves no band
    /// for any of them (`#182`, closes `#94`: a gutter exists if and only if
    /// something is drawn in it).
    ///
    /// Its producer (`render::prepare::build_axes`) resolves BOTH reasons an
    /// axis may not be drawn into this one flag, so no consumer needs a second
    /// gate — the same discipline [`AxesInput::apply_show_defaults`] applies to
    /// `grid`/`domain`:
    ///
    /// 1. `ChartSpec.axis_x = Some(false)` (i.e. `Chart.axis(x=False)`), used on
    ///    clustermap dendrogram panels and JointChart marginal panels.
    /// 2. A projected coordinate system that replaces the Cartesian axes
    ///    outright ([`crate::spec::coord::suppresses_cartesian_axes`]):
    ///    `CoordPolar` draws a circular boundary with radial ticks, `CoordGeo`
    ///    draws no axis at all, and `render::scene_build` emits no Cartesian
    ///    axis or gridline node for either. This is `#94` — a polar chart used
    ///    to reserve a full Cartesian left gutter and bottom band for chrome
    ///    that never existed.
    ///
    /// Default `true`.
    pub show_x: bool,
    /// Y-axis variant of `show_x`, with the same two-reason fold. Default `true`.
    pub show_y: bool,
    /// Secondary y-axis inputs, one per `independent_y` layer, in layer order
    /// (secondary-y-axis, GH #52). Each renders on the right, stacked outward
    /// beyond `y`'s band (slot 0 stays `y`, driving the left axis and its
    /// gridlines — see spec §6 slot contract; a secondary draws its own
    /// gridlines only when asked, see [`AxesInput::apply_show_defaults`]).
    /// Empty (the default) means the
    /// chart has no independent-y layer, so layout reserves no extra band and
    /// emits no extra axis — byte-identical to the pre-#52 shared path.
    pub secondary_y: Vec<AxisInput>,
    /// For a faceted chart whose x or y scale resolves independently per
    /// panel: by panel index, the axis inputs each panel actually draws,
    /// resolved from that panel's own scale (see [`PanelAxisInputs`]). Empty
    /// when no channel is independent, so every panel draws `x` and `y`.
    pub panels: Vec<PanelAxisInputs>,
}

/// One facet panel's own axis inputs, for a channel whose scale resolves
/// independently per panel; `None` for a shared channel (the panel draws the
/// chart's `x` or `y`) and for a panel that draws nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PanelAxisInputs {
    pub x: Option<AxisInput>,
    pub y: Option<AxisInput>,
}

impl AxesInput {
    /// The x axis input panel `panel_index` draws: its own when its x scale
    /// is independent, else the chart's.
    pub fn x_for_panel(&self, panel_index: usize) -> &AxisInput {
        self.panels.get(panel_index).and_then(|p| p.x.as_ref()).unwrap_or(&self.x)
    }

    /// The y axis input panel `panel_index` draws: its own when its y scale
    /// is independent, else the chart's.
    pub fn y_for_panel(&self, panel_index: usize) -> &AxisInput {
        self.panels.get(panel_index).and_then(|p| p.y.as_ref()).unwrap_or(&self.y)
    }

    /// Every x axis input some panel draws: the panels' own when x is
    /// independent, else just `x`. The band each axis reserves is the largest
    /// any of them needs.
    pub(crate) fn drawn_x_inputs(&self) -> Vec<&AxisInput> {
        let own: Vec<&AxisInput> = self.panels.iter().filter_map(|p| p.x.as_ref()).collect();
        if own.is_empty() { vec![&self.x] } else { own }
    }

    /// The y twin of [`drawn_x_inputs`](Self::drawn_x_inputs).
    pub(crate) fn drawn_y_inputs(&self) -> Vec<&AxisInput> {
        let own: Vec<&AxisInput> = self.panels.iter().filter_map(|p| p.y.as_ref()).collect();
        if own.is_empty() { vec![&self.y] } else { own }
    }

    /// Fold the two theme-defaulted show toggles (`grid`, `domain`) into every
    /// axis's own override slot, so from here on the per-axis value is the
    /// FINAL answer and no consumer needs a second, theme-level gate.
    ///
    /// This is what makes the grid/domain cascade a PRECEDENCE chain rather
    /// than the AND it used to be (D4/F-L07-01, spec §4.3): `build_grid` and
    /// `build_axis` previously each re-consulted `theme.grid.grid` /
    /// `theme.axis.axis_line` alongside the per-axis flag, so a per-channel
    /// `Axis(grid=True)` could never light up an axis on a grid-less theme,
    /// and a per-axis `configure(axis_x=AxisConfig(grid=False))` had to be
    /// expressed as a GLOBAL theme write that took the y gridlines down with
    /// it. Resolving once, here, removes both the second gate and the global
    /// write; the same-answer conditional is gone from every consumer.
    ///
    /// Idempotent: a slot that is already `Some` is left alone, so running it
    /// twice (or after a per-channel/chart-level fill) changes nothing.
    ///
    /// Applies to the secondary y axes too — the `axis_y2` position's own
    /// `grid`/`domain` values reach the same slots via the shared fill path —
    /// with one difference: a secondary's grid defaults OFF, not to
    /// `theme.grid.grid`. A secondary y axis draws gridlines only when the
    /// layer's own `Axis(grid=True)` or `axis_y2=AxisConfig(grid=True)` asks,
    /// so a dual-axis chart never gains a second, misaligned set of horizontal
    /// lines from the theme default.
    pub(crate) fn apply_show_defaults(&mut self, theme: &super::ThemeInputs) {
        let per_panel = self
            .panels
            .iter_mut()
            .flat_map(|p| [p.x.as_mut(), p.y.as_mut()])
            .flatten();
        let primary = [&mut self.x, &mut self.y]
            .into_iter()
            .chain(per_panel)
            .map(|axis| (axis, theme.grid.grid));
        let secondary = self.secondary_y.iter_mut().map(|axis| (axis, false));
        for (axis, grid_default) in primary.chain(secondary) {
            axis.overrides.grid.show.get_or_insert(grid_default);
            axis.overrides
                .domain
                .show
                .get_or_insert(theme.axis.axis_line);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisLayout {
    pub orient: AxisOrient,
    pub panel_index: usize,
    pub axis_line: Rect,
    pub ticks: Vec<TickLayout>,
    /// Grid item 18: minor (unlabeled) tick positions, kept separate from
    /// `ticks` so the major label/culling path is untouched. Empty unless minor
    /// rendering is enabled (`AxisInput.tick_projection`'s non-empty `minor`).
    /// Each entry has
    /// `is_major == false`, an empty label, and `culled == false`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub minor_ticks: Vec<TickLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<AxisTitleLayout>,
    /// D7: whether to render tick labels. Default `true`.
    #[serde(default = "default_true")]
    pub show_labels: bool,
    /// D7: whether to render tick marks. Default `true`.
    #[serde(default = "default_true")]
    pub show_ticks: bool,
    /// D7: whether to render the axis domain line. Default `true`.
    #[serde(default = "default_true")]
    pub show_domain: bool,
    /// D7: whether to render gridlines from this axis. Default `true`.
    #[serde(default = "default_true")]
    pub show_grid: bool,
    /// Per-axis tick-mark length override (`configure_axis(tick_size=...)` /
    /// `fm.Axis(tick_size=...)`). `None` means use the theme default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tick_size: Option<f64>,
    /// Per-axis title font size override from `configure_axis(title_font_size=...)`.
    /// `None` means use the theme default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_font_size: Option<f64>,
    /// Per-axis title color override from `configure_axis(title_color="...")`.
    /// Stored as [R, G, B, A]. `None` means use the theme default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_color_rgba: Option<[u8; 4]>,
    /// Pixel gap between tick mark end and label baseline from
    /// `configure_axis(label_padding=...)`. `None` means use the renderer default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_padding: Option<f64>,
    // ── Per-axis style overrides (B5). Stored as `[R, G, B, A]` for colors so the
    //    layout serializes without a palette dependency. `None` → theme default. ──
    /// Tick-label color override. `None` → `theme.colors.label_color`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_color_rgba: Option<[u8; 4]>,
    /// Tick-label font-size override. `None` → `theme.typography.label_font_size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_font_size: Option<f64>,
    /// Gridline color override. `None` → `theme.colors.grid_color`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid_color_rgba: Option<[u8; 4]>,
    /// Gridline dash override. `None` → `theme.grid.grid_dash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid_dash: Option<Vec<f64>>,
    /// Gridline width override. `None` → `theme.sizes.grid_width`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid_width: Option<f64>,
    /// Domain-line color override. `None` → `theme.colors.axis_line_color`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_color_rgba: Option<[u8; 4]>,
    /// Domain-line width override. `None` → `theme.sizes.axis_line_width`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_width: Option<f64>,
    // ── Orphan positioning/draw-order overrides (B5 unit 2) ──────────────────
    /// Per-axis grid-line opacity override `[0, 1]`. `None` →
    /// `theme.grid.grid_opacity`. Consumed by `build_grid` for this axis's
    /// gridlines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid_opacity: Option<f64>,
    /// Perpendicular shift (px, outward positive) applied to every axis scene
    /// node (line/ticks/labels/title) at render time. `None`/`0` → no shift.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translate: Option<f64>,
    /// Coarse draw order relative to marks: `>= 1` → this axis + its gridlines
    /// drawn above marks; `<= 0`/`None` (default) → below marks (current
    /// behavior). Consumed by the scene assembler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zindex: Option<i64>,
    /// Perpendicular shift (px, outward positive) from the plot edge (Vega axis
    /// `offset`). Applied **additively with `translate`** at render time as a
    /// single outward shift. `None`/`0` → no shift. (B5 unit 6b)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    /// Flush the first/last tick labels at the axis ends so edge labels align
    /// within the plot bounds (Vega `labelFlush`). `None`/`false` → default
    /// anchors (byte-identical). Consumed by `build_axis`. (B5 unit 6b)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_flush: Option<bool>,
}

fn default_true() -> bool {
    true
}

impl AxisLayout {
    /// Whether this axis (and its gridlines) should be drawn above the data
    /// marks. Maps the bounded `zindex` semantic (B5): `>= 1` → above,
    /// `<= 0`/absent → below (the historical default).
    pub fn draws_above_marks(&self) -> bool {
        self.zindex.is_some_and(|z| z >= 1)
    }

    /// Build an `AxisLayout` from the resolved geometry (`axis_line`, `ticks`,
    /// `minor_ticks`, `title`, `panel_index`) plus the per-axis `input`, owning the
    /// single copy of the four `show_*` toggles and the sixteen
    /// `input.overrides.<field>` / `.map(rgba_array)` threads (385). Both
    /// `layout_x_axis` and `layout_y_axis` end with one call to this instead of a
    /// 22-field copy-paste literal, so a new per-axis override is a one-line change
    /// here. Byte-identical: each field carries the same value as the prior literal.
    fn from_input(
        input: &AxisInput,
        panel_index: usize,
        axis_line: Rect,
        ticks: Vec<TickLayout>,
        minor_ticks: Vec<TickLayout>,
        title: Option<AxisTitleLayout>,
    ) -> AxisLayout {
        AxisLayout {
            orient: input.orient,
            panel_index,
            axis_line,
            ticks,
            minor_ticks,
            title,
            tick_size: input.overrides.ticks.size,
            show_labels: input.show_labels(),
            show_ticks: input.show_ticks(),
            show_domain: input.show_domain(),
            show_grid: input.show_grid(),
            title_font_size: input.overrides.title.font_size,
            title_color_rgba: input.overrides.title.color.map(rgba_array),
            label_padding: input.overrides.labels.padding,
            label_color_rgba: input.overrides.labels.color.map(rgba_array),
            label_font_size: input.overrides.labels.font_size,
            grid_color_rgba: input.overrides.grid.color.map(rgba_array),
            grid_dash: input.overrides.grid.dash.clone(),
            grid_width: input.overrides.grid.width,
            domain_color_rgba: input.overrides.domain.color.map(rgba_array),
            domain_width: input.overrides.domain.width,
            grid_opacity: input.overrides.grid.opacity,
            translate: input.overrides.placement.translate,
            zindex: input.overrides.placement.zindex,
            offset: input.overrides.placement.offset,
            label_flush: input.overrides.labels.flush,
        }
    }
}

/// Convert an `AxisInput` color override (`Srgba<u8>`) to the `[R, G, B, A]`
/// array form stored on `AxisLayout`. Mirrors the existing `title_color`
/// mapping at the two `layout_*_axis` constructors.
fn rgba_array(c: Srgba<u8>) -> [u8; 4] {
    [c.red, c.green, c.blue, c.alpha]
}

/// How [`project_fractions`] handles a non-finite projected pixel. The two tick
/// projectors guard against non-finite output differently, and the difference is
/// *intentional* — this enum makes it a single named policy instead of two
/// hand-copied finiteness loops (cohesion finding LAYOUT-845; the layout-side
/// instance of archaeology R1, "major path is all-or-nothing on non-finite, minor
/// path drops per-element... should be a named policy, not a copied loop").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonFinitePolicy {
    /// All-or-nothing: if *any* projected pixel is non-finite, discard the whole
    /// projection (`None`). Used by the major-tick projector so a single bad major
    /// drops the projection and the caller falls back to uniform slots — keeping
    /// the labeled majors uniformly spaced rather than partially mis-placed.
    DropAll,
    /// Per-element: silently drop only the non-finite pixels and keep the rest.
    /// Used by the minor-tick projector — minors carry no label, so dropping one
    /// does not misalign anything.
    DropEach,
}

/// Project per-tick domain fractions onto `base_range` through the *same* padding
/// inset that places data marks (`crate::layout::geometry::inset_pixel_range`),
/// then apply `policy` to any non-finite pixel. The inset range becomes an
/// [`Axis1D`] and each fraction maps via [`Axis1D::lerp`] (`lo + t*(hi - lo)`),
/// so a tick at value `v` lands on the same pixel a data mark at `v` would.
///
/// A non-finite fraction (or base range) would yield a NaN/±inf pixel that the
/// SVG renderer rejects (`svg.rs` non-finite guard); `ScaleKind::project_values_to_fractions`
/// already drops the carrier for degenerate/zero-span domains, but guard here too.
///
/// Returns `None` when there are no fractions (categorical axes / empty carrier),
/// or — under [`NonFinitePolicy::DropAll`] — when any pixel is non-finite, so the
/// caller falls back to the uniform-slot formula.
fn project_fractions(
    fractions: &[f64],
    base_range: (f64, f64),
    padding_frac: f64,
    text_demand: &crate::layout::geometry::TextEdgeDemand,
    policy: NonFinitePolicy,
) -> Option<Vec<f64>> {
    if fractions.is_empty() {
        return None;
    }
    let (lo, hi) = crate::layout::geometry::inset_pixel_range(base_range, padding_frac, text_demand);
    let axis = Axis1D { lo, hi };
    match policy {
        NonFinitePolicy::DropAll => {
            let positions: Vec<f64> = fractions.iter().map(|&t| axis.lerp(t)).collect();
            if positions.iter().all(|p| p.is_finite()) {
                Some(positions)
            } else {
                None
            }
        }
        NonFinitePolicy::DropEach => Some(
            fractions
                .iter()
                .map(|&t| axis.lerp(t))
                .filter(|p| p.is_finite())
                .collect(),
        ),
    }
}

/// Continuous-axis scale projection: map each per-tick domain fraction onto the
/// panel's mark pixel range, applying the *same* padding inset that places data
/// marks (`crate::layout::geometry::inset_pixel_range`). The base range is the
/// panel extent oriented exactly as the resolved positional scale: `(x, x+w)`
/// for the x axis and the inverted `(y+h, y)` for the y axis (high data → top
/// pixel). Returns one pixel per fraction, in the supplied order. Returns
/// `None` when the carrier is absent (categorical axes), letting callers fall
/// back to the uniform-slot formula.
///
/// Majors use [`NonFinitePolicy::DropAll`]: a single non-finite major drops the
/// whole projection so labeled ticks stay uniformly spaced (uniform-slot fallback)
/// rather than partially mis-placed.
fn project_tick_positions(input: &AxisInput, base_range: (f64, f64)) -> Option<Vec<f64>> {
    let proj = input.tick_projection.as_ref()?;
    // An empty major vec carries no per-label projection (e.g. a minor-only
    // fixture); fall back to uniform-slot placement for the major ticks.
    project_fractions(
        &proj.major,
        base_range,
        proj.padding_frac,
        &proj.text_demand,
        NonFinitePolicy::DropAll,
    )
}

/// Categorical band centers accepted for index-addressed placement: `Some` only
/// when the carrier holds exactly one center per tick label.
///
/// `layout_*_axis` reads `centers[i]` for every `i` over `tick_labels`, and the
/// carrier is sized by the scale's DOMAIN, so a producer that replaced the
/// labels without re-pairing the carrier (an explicit `tick_values` override is
/// the reachable spelling — see
/// `config_apply::sync_tick_placement_to_tick_values`) would index out of
/// bounds and abort the render across the PyO3 boundary. Debug builds trip
/// [`AxisInput::debug_assert_placement_invariants`] first; this is the release
/// half, and it degrades to the uniform-slot placement — the same outcome
/// clearing the carrier produces — rather than placing labels on centers that
/// are not theirs.
fn paired_band_centers(centers: Option<&[f64]>, tick_count: usize) -> Option<&[f64]> {
    centers.filter(|c| c.len() == tick_count)
}

/// Smallest absolute gap between consecutive positions, or `None` when there are
/// fewer than two. Used by the x-axis collision cascade so non-uniform
/// (continuous) tick spacing is judged by its tightest pair, not an average.
fn min_adjacent_gap(positions: &[f64]) -> Option<f64> {
    if positions.len() < 2 {
        return None;
    }
    positions
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::INFINITY, f64::min)
        .into()
}

/// Build `minor_ticks` from per-minor **domain fractions** when the gate is on.
/// Each fraction is mapped onto the panel via the *same* `inset_pixel_range`
/// padding inset that places majors (`project_tick_positions`) and data marks —
/// **not** the naive `origin + frac * extent`. This guarantees a minor at
/// domain value `v` lands at the identical pixel the major projection of `v`
/// would give, so minor and major gridlines coincide.
///
/// `base_range` must be oriented exactly like the resolved positional scale:
/// `(x, x+w)` for the x axis and the inverted `(y+h, y)` for the y axis. Minors
/// carry no label, are never elided/culled, and are tagged `is_major == false`.
/// Returns an empty vec when there is no `tick_projection` or its `minor` is empty.
fn build_minor_ticks(input: &AxisInput, base_range: (f64, f64)) -> Vec<TickLayout> {
    // "Minor enabled" is derived from `minor` being non-empty: `prepare.rs` only
    // populates it when the `theme.grid.minor` gate is on. `None` (categorical)
    // and an empty `minor` both yield no minor ticks.
    let Some(proj) = input.tick_projection.as_ref() else {
        return Vec::new();
    };
    // Minors share the major projector's inset + lerp via `project_fractions`, but
    // use `NonFinitePolicy::DropEach`: a non-finite minor is dropped per-element
    // (minors carry no label, so dropping one does not misalign anything) instead
    // of discarding the whole projection. `None` (empty `minor`) yields no ticks.
    let Some(positions) = project_fractions(
        &proj.minor,
        base_range,
        proj.padding_frac,
        &proj.text_demand,
        NonFinitePolicy::DropEach,
    ) else {
        return Vec::new();
    };
    positions
        .into_iter()
        .map(|position| TickLayout {
            position,
            label: String::new(),
            label_angle: 0.0,
            elided: false,
            culled: false,
            label_font_size: None,
            is_major: false,
            anchor: false,
            origin: false,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickLayout {
    pub position: f64,
    pub label: String, // may contain '\n' for multi-line labels (future task)
    pub label_angle: f64,
    pub elided: bool,
    /// Tick mark is shown but its label is hidden (label density culling).
    #[serde(default)]
    pub culled: bool,
    /// Per-tick font-size override. `None` means use the theme default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_font_size: Option<f64>,
    /// Grid item 18: `true` for major ticks (all of `AxisLayout.ticks`),
    /// `false` for minor ticks (only present in `AxisLayout.minor_ticks`).
    /// Defaults to `true` so previously serialized layouts deserialize as majors.
    #[serde(default = "default_true")]
    pub is_major: bool,
    /// Whether `AxisInput.tick_anchors` protects this tick (a `Log` decade)
    /// — layout-internal only: its one
    /// consumer is `layout::thins_x_labels_evenly`'s even-cull classifier,
    /// which needs to tell an anchored cull apart from an irregular one.
    /// `#[serde(skip)]` so no scene JSON ever carries it; every `TickLayout`
    /// literal defaults it to `false` and `layout_x_axis`/`layout_y_axis`
    /// stamp the real value onto every MAJOR tick once, at the end
    /// (`stamp_tick_anchors`), after whichever presentation path (the flat
    /// cascade, an explicit-angle stamp, or a stride-culled survivor
    /// expansion) built it — so no builder needs to carry the flag through
    /// itself.
    #[serde(skip)]
    pub anchor: bool,
    /// Whether `AxisInput.tick_origin` names this tick as the axis's nicest
    /// one — layout-internal only, same standing as [`anchor`](Self::anchor):
    /// `#[serde(skip)]` so no scene JSON ever carries it, and every
    /// `TickLayout` literal above defaults it to `false` until
    /// `stamp_tick_anchors` corrects it once, at the end.
    #[serde(skip)]
    pub origin: bool,
}

/// [`TickLayout::anchor`]/[`TickLayout::origin`]'s one producer: tag each of
/// `ticks` with whether `input.tick_anchors` protects the label at its
/// index, and whether `origin` names it as the axis's nicest tick — both
/// read off the position the generated label occupies within
/// `AxisInput.tick_labels`, the same index space the cull already reads
/// both in. One seam after every presentation path (the flat cascade, an
/// explicit-angle stamp, a stride-culled survivor expansion) corrects the
/// placeholder `anchor: false`/`origin: false` every `TickLayout` literal
/// above is built with, instead of threading either flag through each of
/// them.
fn stamp_tick_anchors(ticks: &mut [TickLayout], anchors: &[bool], origin: Option<usize>) {
    for (i, tick) in ticks.iter_mut().enumerate() {
        tick.anchor = anchors.get(i).copied().unwrap_or(false);
        tick.origin = origin == Some(i);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisTitleLayout {
    pub text: String,
    pub anchor_x: f64,
    pub anchor_y: f64,
    pub angle: f64,
}

use super::panel::TextAnchor;
use super::text_metrics::TextMetrics;

/// Vertical extent (px below the axis line) occupied by an x-axis tick label
/// rotated by `angle` (degrees, may be negative). Mirrors the end-anchored
/// pivot geometry in `render/marks/axis.rs`: the pivot sits
/// `tick_size + label_pad + sin(|angle|)·font_size` below the axis line, then the
/// rotated label drops a further `sin(|angle|)·max_label_w + cos(|angle|)·line_h`.
///
/// At -90 this collapses to `tick_size + label_pad + font_size + max_label_w`
/// (sin=1, cos=0); at angle 0 it collapses to `tick_size + label_pad + line_h`
/// (sin=0, cos=1) — the SAME quantity `drawn_x_label_band`'s flat branch
/// computes for one line (#97, spec §4.1). The formula is continuous in
/// `angle`: callers no longer need to special-case `angle == 0`, they can
/// call this unconditionally.
///
/// SYNC: render (`render/marks/axis.rs`) and [`drawn_x_label_band`] depend on
/// this. `drawn_x_label_band` is the one band owner: `estimate_x_label_band`
/// reserves from it and `layout_x_axis` places the x title from it, so the
/// reserved band and the title placement agree for flat, wrapped and rotated
/// labels.
fn rotated_x_label_extent(
    angle: f64,
    max_label_w: f64,
    font_size: f64,
    line_h: f64,
    tick_size: f64,
    label_pad: f64,
) -> f64 {
    let rad = angle.to_radians();
    let sin_abs = rad.sin().abs();
    let cos_abs = rad.cos().abs();
    tick_size + label_pad + sin_abs * font_size + sin_abs * max_label_w + cos_abs * line_h
}

/// Horizontal footprint (px, in the slot direction) that a tick label rotated
/// by `angle` occupies, for judging whether a rotated label still collides
/// with its neighbor. TRANSPOSE of `rotated_y_label_extent`'s horizontal
/// term (`cos_abs * max_label_w + sin_abs * line_h`) — the same relationship
/// that formula already encodes for y labels: as `angle` rotates a label
/// toward vertical, its footprint shrinks from the full label width toward
/// the line height (the text's own stroke "thickness"), never to zero.
///
/// This corrects F-L07-04 (spec §4.6): `cos(±90°)` in IEEE-754 `f64` is not
/// exactly zero (~6.12e-17), so a bare `label_w * cos(angle)` collision check
/// (the pre-fix formula) treats a vertical label as having ~0 horizontal
/// footprint, which auto-passes the rotation stage for any real label width —
/// making the cull (S4) and elide (S5) stages of the cascade unreachable.
/// Adding the `sin(angle) * line_h` term gives -90° a real fit test: a
/// vertical label genuinely occupies about one line-height of horizontal
/// space, so a candidate angle only "passes" when it truly fits (a
/// genuinely-fitting rotation still wins), and the cascade falls through to
/// culling when even -90° doesn't fit the slot.
fn rotated_x_label_footprint(angle: f64, label_w: f64, line_h: f64) -> f64 {
    let rad = angle.to_radians();
    let sin_abs = rad.sin().abs();
    let cos_abs = rad.cos().abs();
    cos_abs * label_w + sin_abs * line_h
}

/// How far an x-axis tick label reaches LEFT and RIGHT of its pivot
/// (`(left, right)`, px, both measured outward from the pivot), when drawn
/// with `anchor` and rotated by `angle` degrees about that pivot.
///
/// The label's unrotated box spans `[x0, x1]` horizontally (`anchor`
/// decides where the pivot sits on the text: `Start` = `[0, w]`, `Middle` =
/// `[-w/2, w/2]`, `End` = `[-w, 0]`) and `[-line_h, line_h]` vertically: a
/// [`TextMetrics`] has no ascent or descent, so both use the line height,
/// the same conservative proxy [`rotated_x_label_footprint`] uses for the
/// text's thickness. Rotating the box's corners gives
/// `x' = x·cos(angle) - y·sin(angle)`, whose extremes are the reach here.
/// At `angle == 0` the vertical term vanishes and this is the exact flat
/// reach (`w/2` each side for `Middle`).
///
/// SYNC: mirrors the tick-label placement in `render/marks/axis.rs`'s
/// `build_axis` for `Bottom`/`Top`: the pivot is `x = tick.position`, a flat
/// label is `Middle`-anchored (or `Start`/`End` at a `label_flush` edge), a
/// rotated `Bottom` label is `End`-anchored and a rotated `Top` label is
/// `Start`-anchored, and SVG rotates about the pivot. [`drawn_x_labels`]
/// picks the anchor the same way. Change them together, or `padding.auto`
/// measures labels where the renderer does not draw them.
pub(super) fn rotated_x_label_horizontal_extent(
    angle: f64,
    anchor: TextAnchor,
    label_w: f64,
    line_h: f64,
) -> (f64, f64) {
    let (x0, x1) = match anchor {
        TextAnchor::Start => (0.0, label_w),
        TextAnchor::Middle => (-label_w / 2.0, label_w / 2.0),
        TextAnchor::End => (-label_w, 0.0),
    };
    let rad = angle.to_radians();
    let cos = rad.cos();
    let vertical_reach = line_h * rad.sin().abs();
    let (a, b) = (x0 * cos, x1 * cos);
    (vertical_reach - a.min(b), a.max(b) + vertical_reach)
}

/// One tick label an x axis draws: its tick's index in `AxisLayout::ticks`,
/// the tick's pixel, and how far its ink reaches left and right (canvas x),
/// as [`drawn_x_labels`] reads them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DrawnXLabel {
    pub(super) index: usize,
    pub(super) tick: f64,
    pub(super) left_ink: f64,
    pub(super) right_ink: f64,
}

/// Every tick label one laid-out x axis (`Bottom` or `Top`) draws, with
/// its ink extent; empty when the axis draws no label text (a y axis,
/// `show_labels == false`, or every tick culled or blank).
///
/// Reads the labels the axis actually draws: culled and blank ticks are
/// skipped, a label is measured at its own (possibly cascade-shrunk) font
/// size, a wrapped label at its widest line, and at its drawn angle and
/// anchor via [`rotated_x_label_horizontal_extent`] (see its SYNC note for
/// the renderer placement both mirror). `default_font_size` is the theme's
/// label size, the renderer's last fallback after the tick's and the
/// axis's own sizes.
pub(super) fn drawn_x_labels(
    axis: &AxisLayout,
    default_font_size: f64,
    metrics: &dyn TextMetrics,
) -> Vec<DrawnXLabel> {
    let anchor_for_angle = match axis.orient {
        AxisOrient::Bottom => TextAnchor::End,
        AxisOrient::Top => TextAnchor::Start,
        AxisOrient::Left | AxisOrient::Right => return Vec::new(),
    };
    if !axis.show_labels {
        return Vec::new();
    }
    // `label_flush` re-anchors the first and last NON-CULLED ticks, blank
    // or not, exactly as the renderer does.
    let flush = axis.label_flush.unwrap_or(false);
    let flush_first = flush.then(|| axis.ticks.iter().position(|t| !t.culled)).flatten();
    let flush_last = flush.then(|| axis.ticks.iter().rposition(|t| !t.culled)).flatten();
    let axis_font_size = axis.label_font_size.unwrap_or(default_font_size);
    axis.ticks
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.culled && !t.label.is_empty())
        .map(|(i, t)| {
            let anchor = if t.label_angle != 0.0 {
                anchor_for_angle
            } else if Some(i) == flush_first {
                TextAnchor::Start
            } else if Some(i) == flush_last {
                TextAnchor::End
            } else {
                TextAnchor::Middle
            };
            let font_size = t.label_font_size.unwrap_or(axis_font_size);
            let (left, right) = rotated_x_label_horizontal_extent(
                t.label_angle,
                anchor,
                measure_multiline_width(&t.label, font_size, metrics),
                metrics.line_height(font_size),
            );
            DrawnXLabel {
                index: i,
                tick: t.position,
                left_ink: t.position - left,
                right_ink: t.position + right,
            }
        })
        .collect()
}

/// The four measurement inputs [`estimate_x_label_band`] and [`layout_x_axis`]
/// must agree on, in one value.
///
/// The band predictor and the real cascade are deliberate mirrors: the
/// predictor reserves the gutter before the plot rect exists, and the cascade
/// then lays out into that gutter. If they disagree on the label font size, the
/// gap policy, the tick length or the text metrics, the reserved band no longer
/// matches the rendered extent and labels clip. That agreement used to be a
/// four-argument SYNC obligation re-satisfied by hand at four call sites (two
/// per function, in `layout::compute_layout` and `render::scene_build`), and
/// each function carried a `clippy::too_many_arguments` allow for the nine
/// positional parameters it took to express it.
///
/// [`for_axis`](Self::for_axis) is the point: all four values derive from the
/// same `(AxisInput, ThemeInputs, TextMetrics)` triple at every production call
/// site, identically, so the derivation has ONE home for a fifth call site to
/// reach for. That is a convention with an obvious default, not an enforced
/// invariant — this struct's fields are `pub`, so a hand-assembled context with
/// a hardcoded gap is still spellable (see [`new`](Self::new)'s doc). What the
/// type removes is the four-value positional thread that made divergence the
/// easy outcome.
#[derive(Clone, Copy)]
pub struct LabelBandCtx<'m> {
    /// Tick-label font size, already resolved against the theme.
    pub label_font_size: f64,
    /// The gap/cull policy every cascade stage's fit budget reads: S0-S2
    /// take `max(label_gap.gap(), 10% of the pitch)` via `collision_need`,
    /// S3 subtracts `label_gap.gap()` directly.
    pub label_gap: LabelGapPolicy,
    /// Tick-mark length, already resolved against the theme.
    pub tick_size: f64,
    /// Text measurement, which must be the SAME implementation on both sides:
    /// facet panels measure through their own metrics instance.
    pub metrics: &'m dyn TextMetrics,
    /// The tallest x label band the axis may draw, inclusive (see
    /// [`lay_out_x_tick_labels`]); `f64::INFINITY`, the default, when the
    /// band never bounds the labels.
    pub band_budget: f64,
}

impl<'m> LabelBandCtx<'m> {
    /// Resolve the context for `input` against `theme`, measured by `metrics` —
    /// the one derivation every production call site shares.
    pub fn for_axis(
        input: &AxisInput,
        theme: &super::ThemeInputs,
        metrics: &'m dyn TextMetrics,
    ) -> Self {
        Self::new(
            input
                .overrides
                .labels
                .font_size
                .unwrap_or(theme.typography.label_font_size),
            theme.label_gap,
            input.tick_size(theme),
            metrics,
        )
    }

    /// Construct a context from values resolved by the caller. Production code
    /// uses [`for_axis`](Self::for_axis); this exists for the unit tests that
    /// exercise one measurement input at a time against a mock metric, and is
    /// `pub(crate)` to say so — `for_axis` is the only derivation production
    /// should reach for. Note this does NOT make a hand-assembled context
    /// unspellable: the struct's fields are `pub`, so a struct literal spells it
    /// too. The single production derivation is a convention with one obvious
    /// home, not an enforced invariant.
    pub(crate) fn new(
        label_font_size: f64,
        label_gap: LabelGapPolicy,
        tick_size: f64,
        metrics: &'m dyn TextMetrics,
    ) -> Self {
        Self {
            label_font_size,
            label_gap,
            tick_size,
            metrics,
            band_budget: f64::INFINITY,
        }
    }

    /// This context with the x label band bounded by `band_budget` (see
    /// [`lay_out_x_tick_labels`]).
    pub fn with_band_budget(self, band_budget: f64) -> Self {
        Self { band_budget, ..self }
    }
}

/// The vertical space (in pixels) the x axis's tick labels need beside the
/// axis line, for an axis laid out across a panel `panel_w` wide. Called by
/// `reserve_axis_bands` **before** the plot rect is finalized, with the
/// width the panel will have.
///
/// It reserves exactly what the axis draws: the labels come from
/// [`lay_out_x_tick_labels`], the same decision [`layout_x_axis`] lays the
/// axis out from (same tick pixels, same per-tick budget, the whole
/// collision cascade including the font shrink, wrapping, culling, elision
/// and `label_overlap`, or an explicit `label_angle`), so the band and the
/// drawn labels cannot disagree on the angle, the font or which labels draw.
/// The four inputs it shares with [`layout_x_axis`] (label font size, gap
/// policy, tick size, metrics) arrive as one [`LabelBandCtx`].
///
/// The band, from the drawn (non-culled) labels:
/// - rotated: the full geometric extent of the widest drawn label at the
///   drawn angle ([`rotated_x_label_extent`]);
/// - flat or wrapped: `tick_size + label_pad_eff + lines * line_height`,
///   `lines` being the most lines any drawn label wraps to (1 when flat).
///
/// SYNC: every branch mirrors the label geometry in
/// `crate::render::marks::axis::build_axis`. A flat label's baseline sits at
/// `r.y + tick_size + label_pad + font_size`, and a rotated label is
/// end-anchored at the pivot `(tick.position, label_y)` where
/// `label_y = r.y + tick_size + label_pad + sin(|angle|)·font_size`, then rotated
/// about that pivot. In both cases the full extent from the axis line (`r.y`)
/// down to the label's lowest point is `tick_size + label_pad +
/// sin(|angle|)·(font_size + max_label_w) + cos(|angle|)·descent`, which
/// collapses at `angle = 0` to `tick_size + label_pad + descent`. The band uses
/// `line_h` at the axis's label font size for the descent term (a small safety
/// margin, kept for a cascade-shrunk font too), so the flat and wrapped
/// branches reserve `tick_size + label_pad_eff + n·line_h` (#97, spec §4.1).
/// `label_padding` is honored absolutely (not as a delta from the 2.0
/// default) and clamped at 0 like the renderer's L-2 guard. Changing this
/// formula requires changing the render geometry to match; the title
/// placement in `layout_x_axis` reads the same [`drawn_x_label_band`].
///
/// **Drawn gate (spec §4.1 amended 2026-08-27, `#182`, which closes `#94`):**
/// mirrors `compute_y_label_band_width`'s gate: an axis that draws no labels
/// (`fm.Axis(labels=False)`, `AxisInput.show_labels`) reserves `0.0`.
/// `axes.show_x` (`.axis(x=False)`, a `Polar`/`Geo` coord) already
/// short-circuits the caller in `reserve_axis_bands` before this is invoked.
///
/// An axis with no tick labels at all but `show_labels: true` keeps its own
/// `line_h + padding_delta`: it still draws its tick marks, and JointChart
/// marginals and ClusterMap dendrograms rely on that gutter (spec §4.1, §7).
pub(crate) fn estimate_x_label_band(
    input: &AxisInput,
    panel_w: f64,
    ctx: &LabelBandCtx<'_>,
) -> f64 {
    x_label_band_without_ticks(input, ctx).unwrap_or_else(|| {
        let (ticks, _) = lay_out_x_tick_labels(input, 0.0, panel_w, ctx);
        drawn_x_label_band(&ticks, input, ctx)
    })
}

/// The x label band of an axis that lays out no tick labels, the two early
/// branches of [`estimate_x_label_band`]: `0.0` for an axis whose labels are
/// off, and `line_h + padding_delta` for one with no tick labels at all;
/// `None` for an axis whose drawn labels decide the band
/// ([`drawn_x_label_band`]). [`layout_x_axis`] places the x title from the
/// same value, except for a labels-off axis (see there).
fn x_label_band_without_ticks(input: &AxisInput, ctx: &LabelBandCtx<'_>) -> Option<f64> {
    if !input.show_labels() {
        return Some(0.0);
    }
    if input.tick_labels.is_empty() {
        let line_h = ctx.metrics.line_height(ctx.label_font_size);
        return Some(line_h + input.overrides.labels.padding.map(|lp| lp - 2.0).unwrap_or(0.0));
    }
    None
}

/// The pitch between the lines of a wrapped x tick label, as a multiple of
/// its font size: the renderer steps each line by this, and
/// [`drawn_x_label_band`] reserves it per line after the first.
pub(crate) const X_LABEL_LINE_PITCH_EM: f64 = 1.2;

/// The band the drawn (non-culled) labels in `ticks` need below the axis
/// line ([`estimate_x_label_band`] states the geometry): a rotated label's
/// full extent for the widest drawn label at the drawn angle; for flat (or
/// wrapped) labels, the standoff plus one line height at the axis's label
/// font (kept when the cascade shrinks the font, a small margin) and one
/// [`X_LABEL_LINE_PITCH_EM`] step at the drawn font per further line, which
/// is where the renderer puts them; and the tick marks alone when no label
/// is drawn.
fn drawn_x_label_band(ticks: &[TickLayout], input: &AxisInput, ctx: &LabelBandCtx<'_>) -> f64 {
    let LabelBandCtx {
        label_font_size,
        tick_size,
        metrics,
        ..
    } = *ctx;
    let line_h = metrics.line_height(label_font_size);
    let label_pad_eff = input.overrides.labels.padding.unwrap_or(2.0).max(0.0);
    let drawn = || ticks.iter().filter(|t| !t.culled);
    let Some(first) = drawn().next() else {
        return tick_size;
    };
    let angle = first.label_angle;
    if angle != 0.0 {
        let max_label_w = drawn()
            .map(|t| {
                metrics.measure_width(&t.label, t.label_font_size.unwrap_or(label_font_size))
            })
            .fold(0.0_f64, f64::max);
        return rotated_x_label_extent(
            angle,
            max_label_w,
            label_font_size,
            line_h,
            tick_size,
            label_pad_eff,
        );
    }
    let font_size = first.label_font_size.unwrap_or(label_font_size);
    let lines = drawn().map(|t| t.label.split('\n').count()).max().unwrap_or(1);
    tick_size + label_pad_eff + line_h + (lines - 1) as f64 * X_LABEL_LINE_PITCH_EM * font_size
}

/// Horizontal extent (px away from the y-axis line) occupied by a y-axis tick
/// label rotated by `angle` (degrees, may be negative). TRANSPOSE of
/// `rotated_x_label_extent`: rotating a y label toward vertical SHRINKS its
/// horizontal footprint (the label's own width stops projecting fully onto the
/// horizontal axis), where rotating an x label toward vertical GROWS its
/// vertical footprint.
///
/// Mirrors the render pivot in `render/marks/axis.rs`'s `Left`/`Right` arms:
/// that pivot is the SAME point as the flat (unrotated) label's anchor point
/// — `TextAnchor::End`/`Start` already sit on the axis-facing edge for y, so
/// (unlike x's `Bottom`/`Top`, whose flat anchor is `Middle`) rotation needs no
/// anchor flip and no extra font-size clearance term to keep the label
/// swinging away from the plot as it rotates. That is why this helper, unlike
/// `rotated_x_label_extent`, has no `font_size` parameter.
///
/// **Caveat (θ ≤ 0 only):** the "no clearance term needed" claim above holds
/// for the documented convention of negative override angles (everything the
/// test suite exercises). For a large POSITIVE `angle`, rotating the
/// `End`-anchored bbox about the flat pivot moves its near edge INWARD by up
/// to `sin|angle|·ascent`, which can exceed the `tick_size + label_pad`
/// standoff (e.g. at θ = +90°, ascent ≈ 0.8·font_size can be a few px larger
/// than a small `tick_size + label_pad`) — a few pixels of the label would
/// cross the axis line into the plot. This is a known, low-impact gap in the
/// pivot geometry, not covered by this formula; a future fix would either
/// clamp/mirror the pivot for `angle > 0` or add the missing clearance term.
///
/// SYNC: render (`render/marks/axis.rs` `Left`/`Right` arms), the whole of
/// `compute_y_label_band_width` (a VISIBLE axis routes through this formula
/// at every angle, including θ=0 — there is no separate flat branch), and
/// `layout_y_axis`'s override-angle branch and title placement all depend on
/// this. Change all four together or the y-axis title / plot region will
/// overlap (or float away from) the rendered labels.
fn rotated_y_label_extent(
    angle: f64,
    max_label_w: f64,
    line_h: f64,
    tick_size: f64,
    label_pad: f64,
) -> f64 {
    let rad = angle.to_radians();
    let sin_abs = rad.sin().abs();
    let cos_abs = rad.cos().abs();
    tick_size + label_pad + cos_abs * max_label_w + sin_abs * line_h
}

/// Returns the pixel width the y-axis gutter must reserve for tick labels.
/// Used by the orchestrator to reserve a left (or right) gutter before
/// computing the plot rect, and by `layout_y_axis` to place the title beyond
/// the labels.
///
/// A non-empty label set on a **visible** axis (`visible = true`) goes through
/// `rotated_y_label_extent` at every angle, including θ=0 (no override, or an
/// explicit `0.0`), which collapses to `tick_size + label_pad_eff + max_label_w`
/// (sin=0, cos=1) — matching exactly what `render/marks/axis.rs`'s
/// `Left`/`Right` arms draw: the label's edge sits `tick_size + label_pad`
/// beyond the axis line (#97, spec §4.1; this retires the pre-R2 quirk where
/// the flat band omitted `tick_size`/`label_pad` entirely, under-reserving
/// relative to the render). The formula is continuous in the override angle
/// (SYNC with `rotated_y_label_extent`).
///
/// An empty tick-label set reserves 0.0 regardless of angle or visibility —
/// this deliberately supersedes the pre-#97 rotated-empty formula (which
/// returned a nonzero `tick_size + label_pad + sin|θ|·line_h` phantom
/// reservation for an angle override with zero labels; spec §4.1 amended
/// 2026-08-27 to retire that specific corner as a reservation for labels that
/// don't exist).
///
/// **Drawn gate (`visible: bool`, spec §4.1 amended 2026-08-27; widened by
/// `#182`, which closes `#94`):** the band exists only for an axis that
/// actually DRAWS its labels — visible AND labels shown. Anything else
/// reserves `0.0`, at any angle. Two independent knobs feed this:
///
/// 1. `.axis(show=False)` (`AxesInput.show_x`/`show_y`, passed in as
///    `visible`) does NOT empty `tick_labels` — `build_axis_input` populates
///    them regardless, and `layout::mod`'s panel loop only skips *emitting*
///    the axis layout. So a hidden y axis (JointChart marginals, ClusterMap
///    dendrograms use `.axis(show=False)`) still reaches this function with
///    real labels. `visible` also carries the projected-coord case: a
///    `Polar`/`Geo` chart has no Cartesian axes at all
///    ([`crate::spec::coord::suppresses_cartesian_axes`], folded into the two
///    show flags by `render::prepare::build_axes`), which is `#94`'s half of
///    the defect.
/// 2. `fm.Axis(labels=False)` (`AxisInput.show_labels`, read directly off
///    `input` — no separate parameter needed) draws NO tick label text
///    either (`render/marks/axis.rs`'s tick loop: `if axis.show_labels &&
///    !tick.culled`), on an otherwise fully visible axis. Its title is
///    unaffected, so this collapses the label band alone.
///
/// `#97`'s standoff question (should `tick_size + label_pad_eff` be added on
/// top of the text extent?) is subsumed: an undrawn axis reserves nothing at
/// all, so there is nothing for the standoff to stack onto. What this replaced
/// was the pre-`#94` phantom — bare `max_label_w` at θ=0, and the full
/// `rotated_y_label_extent` at any other angle — i.e. space set aside for
/// label text that never renders.
pub(crate) fn compute_y_label_band_width(
    input: &AxisInput,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
    tick_size: f64,
    visible: bool,
) -> f64 {
    if input.tick_labels.is_empty() {
        return 0.0;
    }
    let max_label_w = input
        .tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .fold(0.0_f64, f64::max);
    let angle = input.overrides.labels.angle.unwrap_or(0.0);
    let labels_drawn = visible && input.show_labels();
    if !labels_drawn {
        // #182 (closes #94): a band exists if and only if something is drawn
        // in it. An axis whose labels never draw — hidden via
        // `.axis(show=False)`, replaced by a projected coord, or shown with
        // `Axis(labels=False)` — reserves NOTHING, at any angle. This
        // supersedes the pre-#97 flat `max_label_w` reservation kept here
        // while #94 was open (and, for a rotated hidden axis, the full
        // `rotated_y_label_extent` the angle arm below used to return): both
        // were space set aside for label text that never renders, which is
        // what made a `.axis(show=False)` chart sit off-centre in its own
        // frame.
        return 0.0;
    }
    let line_h = metrics.line_height(label_font_size);
    let label_pad = input.overrides.labels.padding.unwrap_or(2.0).max(0.0);
    rotated_y_label_extent(angle, max_label_w, line_h, tick_size, label_pad)
}

/// Returns the title-row width contribution: title text height (rotated 90°,
/// so its "width" along the x-axis is its line height) plus axis_title_padding.
/// Returns 0 when this axis draws no title — either because it has none, or
/// because the axis itself is not drawn.
///
/// **Drawn gate (`visible: bool`, `#182`, closes `#94`):** a gutter exists if
/// and only if something is drawn in it. `visible` is the same
/// `AxesInput::show_x`/`show_y` answer [`compute_y_label_band_width`] already
/// takes — "is this Cartesian axis drawn at all", folding both
/// `.axis(show=False)` and a projected coord (`Polar`/`Geo`) that replaces the
/// Cartesian axes outright (see
/// [`crate::spec::coord::suppresses_cartesian_axes`]). `AxisInput::title` is
/// populated regardless of either, so before this gate a hidden axis with a
/// title still reserved the full gutter and the plot sat off-centre in its own
/// frame. Unlike the label band, the title is NOT gated on
/// `AxisInput::show_labels` — `fm.Axis(labels=False)` drops the tick labels and
/// keeps the title, so it collapses the label band alone.
pub fn compute_y_title_width(
    input: &AxisInput,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
    visible: bool,
) -> f64 {
    if visible && input.title.get().is_some() {
        let effective_title_font_size = input.overrides.title.font_size.unwrap_or(title_font_size);
        let effective_title_padding = input.overrides.title.padding.unwrap_or(axis_title_padding);
        metrics.line_height(effective_title_font_size) + effective_title_padding
    } else {
        0.0
    }
}

/// Returns the x-axis title-gutter height contribution: title text line height
/// (the title is unrotated on the x axis, so its band along the y-axis is its line
/// height) plus axis_title_padding. Returns 0 when this axis draws no title —
/// see [`compute_y_title_width`] for the `visible` drawn gate (`#182`/`#94`).
/// The body is byte-identical to [`compute_y_title_width`] (cohesion finding
/// LAYOUT-855: `compute_layout` previously inlined this formula "mirroring the
/// y-axis pattern" while y was a named helper, leaving the axis family
/// asymmetric).
pub fn compute_x_title_width(
    input: &AxisInput,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
    visible: bool,
) -> f64 {
    if visible && input.title.get().is_some() {
        let effective_title_font_size = input.overrides.title.font_size.unwrap_or(title_font_size);
        let effective_title_padding = input.overrides.title.padding.unwrap_or(axis_title_padding);
        metrics.line_height(effective_title_font_size) + effective_title_padding
    } else {
        0.0
    }
}

/// Build the AxisLayout for the y-axis (Left orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.h`; no graduated
/// collision cascade applies to the y-axis (spec §14.4). `input.overrides.labels.angle`
/// (R2) is the one exception: when set, it bypasses the (absent) cascade the
/// same way it bypasses x's cascade, rotating every tick and eliding any label
/// whose rotated footprint still collides with its neighbor (no cull recovery
/// on y — see `AxisLabelWarning`).
///
/// # Why this one is NOT on [`LabelBandCtx`], and keeps its 8 flat args
/// Deliberate, not an unfinished migration — do not "complete" the bundle by
/// handing this function a ctx. It shares only THREE of the four bundled values
/// (label font size, tick size, metrics) and takes no [`LabelGapPolicy`]:
/// stride-culling is an x-label concern, so a ctx here would carry a gap policy
/// this function never reads. More decisive, the bundle exists to delete a
/// *two-function sync obligation* — [`estimate_x_label_band`] reserves the band
/// that [`layout_x_axis`] then lays out into, and they must agree or labels clip.
/// y has no such pair: its band predictor is [`compute_y_label_band_width`], a
/// separate 5-argument function with its own contract, so there is no agreement
/// for a ctx to mechanise. Mechanising the y pair is a follow-up with its own
/// design question (which three values, and whether `compute_y_label_band_width`
/// joins), not a loose end here. Recorded because `compute_x_title_width`'s doc
/// above cites cohesion finding LAYOUT-855 for penalising exactly this kind of
/// axis-family asymmetry: this one is chosen and reasoned, not drift.
#[allow(clippy::too_many_arguments)]
pub fn layout_y_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    label_font_size: f64,
    title_font_size: f64,
    axis_title_padding: f64,
    tick_size: f64,
    metrics: &dyn TextMetrics,
) -> (AxisLayout, Option<AxisLabelWarning>) {
    input.debug_assert_placement_invariants();
    let n = input.tick_labels.len();
    let slot_h = if n > 0 { panel_area.h / n as f64 } else { 0.0 };
    // Uniform-slot fallback range (top → bottom, in pixel order): slot `i`'s center
    // is `panel_area.y + (i + 0.5)*slot_h` via `Axis1D::uniform_center`.
    let slot_axis = Axis1D {
        lo: panel_area.y,
        hi: panel_area.y + panel_area.h,
    };
    // Continuous axes: place each tick at its scale-projected pixel (mark range
    // is the inverted y range `(bottom, top)`, inset exactly like data marks).
    // Categorical axes (no projected fractions): keep the uniform-slot formula.
    let projected = project_tick_positions(input, (panel_area.y + panel_area.h, panel_area.y));
    // Categorical axes (F-L04-03, GH #67): place each tick at the scale's own
    // band center — the same pixel its mark gets — so labels and grid lines agree
    // with the marks. `slot_axis` is bit-for-bit the pixel interval
    // `scene_build::resolve_panel_scales` gives this panel's mark scales (ordinal
    // y is NOT inverted — see `scale_resolve::axis_pixel_range`), so a
    // panel-extent placement resolves to those same centers exactly. Absent
    // (`None`) → the uniform-slot formula, for the scale-less `compute_layout`
    // binding — and so does a carrier that is not paired 1:1 with the labels
    // (see `paired_band_centers`).
    let resolved_centers = input
        .categorical_placement
        .as_ref()
        .map(|p| p.centers(slot_axis.lo, slot_axis.hi));
    let band_centers = paired_band_centers(resolved_centers.as_deref(), n);
    let tick_position = |i: usize| -> f64 {
        match (&projected, band_centers) {
            (Some(px), _) => px[i],
            (None, Some(centers)) => centers[i],
            (None, None) => slot_axis.uniform_center(i, slot_h),
        }
    };
    // Per-tick vertical budget used to judge whether a rotated label still
    // collides with its neighbor — the y-dimension transpose of
    // `layout_x_axis`'s `cascade_slot_w`. Continuous and categorical axes use
    // the minimum adjacent gap between projected/band-center positions (the
    // tightest pair, worst case); the scale-less binding uses `slot_h`.
    let cascade_slot_h = match (&projected, band_centers) {
        (Some(px), _) => min_adjacent_gap(px).unwrap_or(slot_h),
        (None, Some(centers)) => min_adjacent_gap(centers).unwrap_or(slot_h),
        (None, None) => slot_h,
    };

    // `label_overlap` on y (D12, spec §4.9): the y-dimension transpose of the
    // policy `cascade_collision_recovery` applies on x, reading the SAME
    // `LabelOverlap` object off the same override slot — so
    // `configure_axis(label_overlap=…)` and `fm.Axis(label_overlap=…)` mean
    // one thing on both axes instead of silently applying to x only.
    //
    // `None` stays byte-identical: y has never run a collision cascade, so
    // "no policy" must keep meaning "draw every label". That is why `None`
    // and `Greedy` differ HERE and not on x — on y, `Greedy` is an explicit
    // request for the fit-based culling y otherwise never does.
    // Greedy judges distance on the real
    // positions labels draw at, not `cascade_slot_h` slots (which
    // understates every gap but a Log axis's tightest, the 9-to-10 one).
    let tick_positions: Vec<f64> = (0..n).map(tick_position).collect();
    let overlap_visible = resolve_y_overlap(
        input.overrides.labels.overlap,
        &input.tick_labels,
        &tick_positions,
        label_font_size,
        metrics,
        &input.tick_anchors,
    );
    // `label_overlap = "rotate"` supplies the angle when the caller set none,
    // so the two knobs compose instead of one silently shadowing the other:
    // an explicit `label_angle` still wins (it is the more specific request).
    let effective_angle =
        input
            .overrides
            .labels
            .angle
            .or_else(|| match input.overrides.labels.overlap {
                Some(LabelOverlap::Rotate) => Some(*ANGLE_CASCADE.last().unwrap()),
                _ => None,
            });
    let (mut ticks, warning) = if let Some(override_angle) = effective_angle {
        // R2: label_angle_override always applies (there is no cascade to
        // bypass on y). Shared body: see
        // `stamp_override_angle_with_elide`'s doc for the x/y transpose
        // (y's projection factor is `sin|angle|` against `cascade_slot_h`).
        let sin_factor = override_angle.to_radians().sin().abs();
        stamp_override_angle_with_elide(
            &input.tick_labels,
            override_angle,
            sin_factor,
            cascade_slot_h,
            label_font_size,
            metrics,
            tick_position,
        )
    } else {
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: tick_position(i),
                label: label.clone(),
                label_angle: 0.0,
                elided: false,
                culled: false,
                label_font_size: None,
                is_major: true,
                anchor: false,
                origin: false,
            })
            .collect();
        (ticks, None)
    };
    if let Some(visible) = overlap_visible {
        for (tick, keep) in ticks.iter_mut().zip(visible) {
            tick.culled = !keep;
        }
    }
    // Minors use the SAME inverted base range + padding inset as the major
    // projection above, so a minor at domain `v` coincides with the major
    // projection of `v`.
    let minor_ticks = build_minor_ticks(input, (panel_area.y + panel_area.h, panel_area.y));

    // Orient: Left (default) places the axis on the panel's left edge; Right on
    // the right edge. Any other orient is rejected upstream (`prepare.rs`
    // validates x→{top,bottom}, y→{left,right}); default to Left defensively.
    let on_right = matches!(input.orient, AxisOrient::Right);
    let axis_x = if on_right {
        panel_area.x + panel_area.w
    } else {
        panel_area.x
    };
    let axis_line = Rect {
        x: axis_x,
        y: panel_area.y,
        w: 1.0,
        h: panel_area.h,
    };

    let effective_title_font_size = input.overrides.title.font_size.unwrap_or(title_font_size);
    let effective_title_padding = input.overrides.title.padding.unwrap_or(axis_title_padding);

    let title = input.title.get().as_ref().map(|text| {
        // `visible: true` — `layout_y_axis` itself is only ever called for a
        // chart-level-visible axis (the panel loop gates the call on
        // `axes.show_y`/every `axes.secondary_y` entry is always emitted), so
        // the `.axis(show=False)` half of `compute_y_label_band_width`'s gate
        // is unreachable-but-inert here. The `Axis(labels=False)` half is
        // NOT inert, though — a titled axis can still have `show_labels ==
        // false` (a title with no tick labels), and `compute_y_label_band_width`
        // reads `input.show_labels` directly, so that case correctly keeps
        // the title flush against the bare `max_label_w` reservation instead
        // of the new standoff.
        let label_band =
            compute_y_label_band_width(input, label_font_size, metrics, tick_size, true);
        let title_h = metrics.line_height(effective_title_font_size);
        // The title sits beyond the tick labels, on the same side as the axis.
        let title_x = if on_right {
            axis_x + label_band + effective_title_padding + title_h / 2.0
        } else {
            axis_x - label_band - effective_title_padding - title_h / 2.0
        };
        // `title_orient` overrides the title's own rotation/anchor. The default
        // for a vertical (left/right) axis is a 90°-rotated title; a horizontal
        // `title_orient` (top/bottom) renders the title flat (e.g. a horizontal
        // caption above a left axis). The default orient (no override) keeps the
        // historical rotation: `-90` on the left, `+90` on the right.
        let angle = match input.overrides.title.orient {
            Some(AxisOrient::Top) | Some(AxisOrient::Bottom) => 0.0,
            Some(AxisOrient::Right) => 90.0,
            Some(AxisOrient::Left) => -90.0,
            None => {
                if on_right {
                    90.0
                } else {
                    -90.0
                }
            }
        };
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: title_x,
            anchor_y: panel_area.y + panel_area.h / 2.0,
            angle,
        }
    });

    // stamp `anchor` once, after whichever
    // presentation path above built `ticks`, instead of threading the flag
    // through each of them.
    stamp_tick_anchors(&mut ticks, &input.tick_anchors, input.tick_origin);
    // 385: single construction site for the 22-field AxisLayout (shared with
    // layout_x_axis via `AxisLayout::from_input`).
    let layout = AxisLayout::from_input(input, panel_index, axis_line, ticks, minor_ticks, title);
    (layout, warning)
}

/// The y-axis transpose of `cascade_collision_recovery`'s `label_overlap`
/// block (D12, spec §4.9): which tick labels survive, given the policy.
///
/// `None` (the return) means "no policy applied — leave every label as the
/// caller built it", which is what keeps a y axis with no `label_overlap`
/// byte-identical. Per policy:
/// - [`LabelOverlap::ShowAll`] — every label kept, explicitly (the same
///   observable as no policy today, but a promise rather than an accident:
///   a future y-side cascade must not cull under `ShowAll`).
/// - [`LabelOverlap::Parity`] — stride-2 decimation, no measurement, exactly
///   as on x.
/// - [`LabelOverlap::Greedy`] — keep a label only if its line box clears the
///   last kept one's; on x this is the default cascade's behavior, on y it is
///   an explicit request, since y has never culled on its own. Decades (an
///   anchored mask) are greedy among THEMSELVES first, then every
///   non-anchor is kept only when it clears both its nearest kept neighbour
///   (anchor or non-anchor) to each side, judged on the
///   real pixel positions labels draw at, not slot counts: a Log axis's
///   tightest adjacent gap, the 9-to-10 one, understated every other gap
///   when distance was counted in `cascade_slot_h` slots.
/// - [`LabelOverlap::Rotate`] — no culling here; the caller turns this into
///   the steepest cascade angle, mirroring x's rotate stage.
///
/// `positions` are the real pixel position `layout_y_axis` places each
/// label at (`tick_position(i)`), collected once into a slice in label order.
fn resolve_y_overlap(
    policy: Option<LabelOverlap>,
    labels: &[String],
    positions: &[f64],
    label_font_size: f64,
    metrics: &dyn TextMetrics,
    anchors: &[bool],
) -> Option<Vec<bool>> {
    match policy? {
        LabelOverlap::ShowAll | LabelOverlap::Rotate => Some(vec![true; labels.len()]),
        LabelOverlap::Parity => Some(anchored_visibility(labels.len(), 2, anchors, 0)),
        LabelOverlap::Greedy => {
            let line_h = metrics.line_height(label_font_size);
            Some(greedy_y_visibility(positions, line_h, anchors))
        }
    }
}

/// [`resolve_y_overlap`]'s `Greedy` policy:
/// two passes over the real tick positions. Pass 1 keeps
/// each anchor (a Log decade) greedily among the anchors alone, in index
/// order. Pass 2 keeps each non-anchor only when it clears BOTH the nearest
/// already-kept label to its left (an anchor or an earlier-kept non-anchor)
/// and the nearest upcoming KEPT anchor to its right — a later non-anchor's
/// own clearance from this one is checked symmetrically when ITS turn comes,
/// via the same left-hand test, so checking only the upcoming anchors on the
/// right is enough: no two survivors, anchor or non-anchor, ever end up
/// closer than `line_h`. With no anchors (or every position an anchor),
/// pass 1 runs the classic single-pass greedy and pass 2 is a no-op (or vice
/// versa), which is exactly today's slot-count greedy on a uniform axis.
fn greedy_y_visibility(positions: &[f64], line_h: f64, anchors: &[bool]) -> Vec<bool> {
    let n = positions.len();
    let mut keep = vec![false; n];

    // Pass 1: anchors greedy among themselves.
    let mut last_anchor: Option<f64> = None;
    for i in 0..n {
        if anchors.get(i).copied().unwrap_or(false) {
            let clears = last_anchor.is_none_or(|p| (positions[i] - p).abs() >= line_h);
            if clears {
                keep[i] = true;
                last_anchor = Some(positions[i]);
            }
        }
    }

    // The nearest KEPT anchor strictly after each index, scanned backward
    // once, so pass 2 can test a non-anchor against an upcoming anchor
    // before that anchor's own (already-decided) turn arrives.
    let mut next_kept_anchor: Vec<Option<f64>> = vec![None; n];
    let mut running: Option<f64> = None;
    for i in (0..n).rev() {
        next_kept_anchor[i] = running;
        if keep[i] && anchors.get(i).copied().unwrap_or(false) {
            running = Some(positions[i]);
        }
    }

    // Pass 2: non-anchors, in index order.
    let mut last_kept: Option<f64> = None;
    for i in 0..n {
        if anchors.get(i).copied().unwrap_or(false) {
            if keep[i] {
                last_kept = Some(positions[i]);
            }
            continue;
        }
        let clears_left = last_kept.is_none_or(|p| (positions[i] - p).abs() >= line_h);
        let clears_right = next_kept_anchor[i].is_none_or(|p| (p - positions[i]).abs() >= line_h);
        if clears_left && clears_right {
            keep[i] = true;
            last_kept = Some(positions[i]);
        }
    }

    keep
}

use crate::layout::text_metrics::measure_multiline_width;
use crate::layout::{ANGLE_CASCADE, FONT_SHRINK_FACTOR, LABEL_OVERLAP_TOLERANCE};

/// Per-axis tick-label warning the orchestrator may emit (x's collision
/// cascade and y's override-angle elide-to-fit recovery both surface through
/// this one type — see R2). Internal — consumers translate to `LayoutWarning`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisLabelWarning {
    LabelsElided { count: u32 },
}

// --- Collision cascade types (private to axis.rs) ---

/// Diagnostic tag indicating which cascade stage resolved the collision.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CascadeStrategy {
    Flat,
    Wrapped,
    FontReduced,
    Rotated { angle: f64 },
    Culled { stride: u32 },
    Elided { count: u32 },
}

/// Output of `cascade_collision_recovery()`. Consumed by `layout_x_axis()` to
/// build `TickLayout` entries.
struct CascadeResult {
    labels: Vec<String>,
    angle: f64,
    font_size: Option<f64>,
    visible: Vec<bool>,
    strategy: CascadeStrategy,
}

/// Truncate `label` by char prefix until the measured width plus the ellipsis
/// width fits in `max_width`. Returns the truncated label with "…" appended.
/// If even "…" alone exceeds max_width, returns "…" anyway (caller is already
/// in a degenerate state).
fn elide_to_fit(label: &str, max_width: f64, font_size: f64, metrics: &dyn TextMetrics) -> String {
    let ellipsis = '\u{2026}';
    let ellipsis_w = metrics.measure_width(&ellipsis.to_string(), font_size);
    if ellipsis_w >= max_width {
        return ellipsis.to_string();
    }
    let budget = max_width - ellipsis_w;
    let mut out = String::new();
    for ch in label.chars() {
        let mut tentative = out.clone();
        tentative.push(ch);
        if metrics.measure_width(&tentative, font_size) > budget {
            break;
        }
        out = tentative;
    }
    out.push(ellipsis);
    out
}

/// Try to wrap `label` into multiple lines so each line's measured width fits
/// within `max_width`. Returns `Some("\n"-joined string)` if wrapping succeeded
/// (at least one break point was found and all resulting lines fit), or `None`
/// if the label has no applicable break points or any single segment exceeds
/// `max_width`.
///
/// Split strategy — first applicable rule wins:
/// 1. Underscore: split on `_` boundaries (unconditional — every segment
///    gets its own line regardless of whether a shorter combined form
///    would fit).
/// 2. Space: greedy line-fill — pack as many words per line as fit under
///    `max_width` (the label's own NATURAL, minimal-line wrap; pre-task
///    behavior — see `wrap_label_to_line_count`
///    below for the axis-uniform variant).
/// 3. camelCase: split at lowercase->uppercase transitions (unconditional,
///    like Rule 1).
///
/// This is the label's OWN natural wrap in isolation. `cascade_collision_recovery`'s
/// S1 stage additionally enforces that every sibling label in an axis
/// resolves to the SAME line count once a positive `cull_threshold` is in
/// effect (spec §4.6's "the cascade degrades uniformly") — see
/// `wrap_label_to_line_count`'s doc for why that's a SEPARATE function
/// rather than a change to this one: `cull_threshold == 0` must reproduce
/// this function's un-adjusted, potentially non-uniform, pre-task output
/// exactly (byte-identity requirement).
///
/// Rule 2 (space) is the VERBATIM pre-task `String`-accumulator loop: two
/// prior attempts to replicate this algorithm through a `Vec<&str>`
/// accumulator each introduced a NEW divergence for labels with an EMPTY
/// split-word (a leading, trailing, or doubled space) — an un-filtered
/// `Vec` let an empty word survive into the joined line as a stray space or
/// phantom line; a filtered `Vec` deleted empty words outright, which
/// matches pre-task only for a LEADING empty
/// word (`String::push_str("")` on an empty string is a true no-op) and
/// diverges for a trailing or interior one (pre-task's
/// `format!("{} {}", current, word)` inserts a joining space and MEASURES
/// it, which can change which line a word lands on and hence the label's
/// LINE COUNT — not just cosmetic whitespace). Rather than a third attempt
/// to reproduce the original's quirks through a different data structure,
/// this is the original structure: fidelity here is exact by construction.
/// The `Vec`-based `greedy_pack_words` (below) still exists, but only as
/// the uniformization path's (`pack_words_to_line_count`) OWN starting
/// point — a context where the contract is axis-wide uniform line counts,
/// not byte-identity to this function.
fn wrap_label(
    label: &str,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> Option<String> {
    // Rule 1: underscore split.
    if label.contains('_') {
        let segments: Vec<&str> = label.split('_').collect();
        if segments
            .iter()
            .any(|s| metrics.measure_width(s, font_size) > max_width)
        {
            return None;
        }
        return Some(segments.join("\n"));
    }

    // Rule 2: space — greedy line-fill (pack as many words per line as fit).
    if label.contains(' ') {
        let words: Vec<&str> = label.split(' ').collect();
        // Any single word that exceeds max_width makes wrapping impossible.
        if words
            .iter()
            .any(|w| metrics.measure_width(w, font_size) > max_width)
        {
            return None;
        }
        let mut lines: Vec<String> = Vec::new();
        let mut current = String::new();
        for &word in &words {
            if current.is_empty() {
                current.push_str(word);
            } else {
                let candidate = format!("{} {}", current, word);
                if metrics.measure_width(&candidate, font_size) > max_width {
                    lines.push(current);
                    current = word.to_string();
                } else {
                    current = candidate;
                }
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
        return Some(lines.join("\n"));
    }

    // Rule 3: camelCase — split at lowercase->uppercase transitions.
    let chars: Vec<char> = label.chars().collect();
    let has_camel = chars
        .windows(2)
        .any(|w| w[0].is_lowercase() && w[1].is_uppercase());
    if has_camel {
        let mut segments: Vec<String> = Vec::new();
        let mut current = String::new();
        for window_start in 0..chars.len() {
            let ch = chars[window_start];
            let next = chars.get(window_start + 1);
            current.push(ch);
            if let Some(&next_ch) = next {
                if ch.is_lowercase() && next_ch.is_uppercase() {
                    segments.push(current.clone());
                    current.clear();
                }
            }
        }
        if !current.is_empty() {
            segments.push(current);
        }
        if segments
            .iter()
            .any(|s| metrics.measure_width(s, font_size) > max_width)
        {
            return None;
        }
        return Some(segments.join("\n"));
    }

    // No break points found.
    None
}

/// Greedily pack `words` onto as few lines as fit within `max_width` — the
/// `Vec<&str>`-group-producing counterpart to `wrap_label`'s Rule 2 loop,
/// used ONLY as `pack_words_to_line_count`'s (and hence
/// `wrap_label_to_line_count`'s) own starting point before forcing
/// additional lines via `split_balanced`. Returns word groups (not yet
/// joined) rather than strings, since the uniformization path needs real
/// word boundaries to split further.
///
/// This is a DELIBERATELY separate implementation from `wrap_label`'s Rule
/// 2, not a shared extraction of it: the two
/// functions serve different contracts. `wrap_label` (`cull_threshold == 0`)
/// must be byte-identical to pre-task, including pre-task's own quirky
/// treatment of empty split-words (see `wrap_label`'s doc); this function
/// feeds ONLY the uniform-degradation path (`cull_threshold > 0`), where the
/// contract is "every sibling label resolves to the same line count," not
/// byte-identity to anything. Empty words (from a leading/trailing/doubled
/// space, e.g. `" Foo"` splitting to `["", "Foo"]`) are simply dropped
/// before packing here — a valid, simpler choice for a path that owns its
/// own semantics rather than needing to reproduce another function's.
fn greedy_pack_words<'a>(
    words: &[&'a str],
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> Vec<Vec<&'a str>> {
    let mut lines: Vec<Vec<&'a str>> = Vec::new();
    let mut current: Vec<&'a str> = Vec::new();
    for &word in words.iter().filter(|w| !w.is_empty()) {
        if current.is_empty() {
            current.push(word);
        } else {
            let mut candidate = current.clone();
            candidate.push(word);
            let candidate_str = candidate.join(" ");
            if metrics.measure_width(&candidate_str, font_size) > max_width {
                lines.push(current);
                current = vec![word];
            } else {
                current = candidate;
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Split `words` (already known to jointly fit on one line) into two
/// non-empty groups at whichever boundary minimizes the WIDER of the two
/// resulting lines — the most width-balanced 2-way split. Used by
/// `pack_words_to_line_count` to grow a group into more lines without ever
/// producing an unbalanced (e.g. "one word alone") split when a balanced
/// one is available.
fn split_balanced<'a>(
    words: &[&'a str],
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> (Vec<&'a str>, Vec<&'a str>) {
    let k = words.len();
    let mut best_split = 1;
    let mut best_max_w = f64::INFINITY;
    for i in 1..k {
        let left = words[..i].join(" ");
        let right = words[i..].join(" ");
        let m = metrics
            .measure_width(&left, font_size)
            .max(metrics.measure_width(&right, font_size));
        if m < best_max_w {
            best_max_w = m;
            best_split = i;
        }
    }
    (words[..best_split].to_vec(), words[best_split..].to_vec())
}

/// Force `words`' natural (minimal) greedy pack (`greedy_pack_words`) up to
/// exactly `target_lines` lines, by repeatedly splitting the group with the
/// most words into two width-balanced halves (`split_balanced`) until the
/// line count matches. Splitting only ever shrinks a line's width, so the
/// `max_width` fit `greedy_pack_words` already established is preserved
/// throughout — this never degrades to one-word-per-line unless `words`
/// itself has too few words to reach `target_lines` any other way. Returns
/// fewer than `target_lines` groups only in that case (nothing left to
/// split).
fn pack_words_to_line_count<'a>(
    words: &[&'a str],
    target_lines: usize,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> Vec<Vec<&'a str>> {
    let mut groups = greedy_pack_words(words, max_width, font_size, metrics);
    while groups.len() < target_lines {
        let widest = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.len() > 1)
            .max_by_key(|(_, g)| g.len());
        let Some((idx, _)) = widest else { break };
        let group = groups.remove(idx);
        let (left, right) = split_balanced(&group, font_size, metrics);
        groups.insert(idx, right);
        groups.insert(idx, left);
    }
    groups
}

/// Like `wrap_label`, but forces the result onto exactly `target_lines`
/// lines wherever the label's own natural (minimal) wrap would use fewer —
/// the uniform-degradation fix. `cascade_collision_recovery`'s
/// S1 stage calls this (instead of re-deriving uniformity by hand) once
/// `cull_threshold > 0`, so every sibling label in the axis resolves to the
/// SAME line count without ever degrading to one-word-per-line packing —
/// `"United States of America"` still packs 2+2 (`"United States"` /
/// `"of America"`), never four stacked single words.
///
/// Only the space rule (Rule 2) is retargeted this way: underscore (Rule 1)
/// and camelCase (Rule 3) already split unconditionally at every one of
/// their boundaries, so their line count is fixed by the label's own
/// structure, not by width — forcing MORE lines than a label has break
/// points is undefined, so those rules fall through to `wrap_label`
/// unchanged, as does the empty/unsplittable case and any label whose
/// natural line count already meets or exceeds `target_lines`.
fn wrap_label_to_line_count(
    label: &str,
    target_lines: usize,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> Option<String> {
    if !label.contains('_') && label.contains(' ') {
        let words: Vec<&str> = label.split(' ').collect();
        if words
            .iter()
            .any(|w| metrics.measure_width(w, font_size) > max_width)
        {
            return None;
        }
        let groups = pack_words_to_line_count(&words, target_lines, max_width, font_size, metrics);
        let lines: Vec<String> = groups.into_iter().map(|g| g.join(" ")).collect();
        return Some(lines.join("\n"));
    }
    wrap_label(label, max_width, font_size, metrics)
}

/// Uniform-degradation post-process for a wrap stage's natural (per-label)
/// result — spec §4.6. Shared by `cascade_collision_recovery`'s
/// S1 and S2b, the two wrap-invoking stages of the cascade (`estimate_x_label_band`'s
/// own S1 wrap call is the only other `wrap_label` call site, and needs no
/// equivalent treatment: its band-height reservation only ever depends on
/// the maximum line count across labels, unaffected by whether shorter
/// siblings are later uniformized up to it).
///
/// Keyed on the requested gap MAGNITUDE alone, not on
/// [`LabelGapPolicy::may_cull`]: uniform degradation is what a gap
/// requirement asks for, and is unrelated to whether the axis is permitted
/// to drop labels further down the cascade. So the parameter is the bare
/// `gap_px`.
///
/// At `gap_px == 0`, returns `natural_labels` unchanged — this is
/// the pre-task, per-label wrap result byte-for-byte (the byte-identity
/// requirement; pre-task never uniformized either, so
/// preserving whatever raggedness it could produce at `0` IS the contract,
/// not a violation of it). At `gap_px > 0`, computes
/// `target_lines` as the max natural line count across the axis's sibling
/// labels, then re-packs (via `wrap_label_to_line_count`) any label whose
/// natural count falls short — every sibling resolves to the SAME line
/// count without ever collapsing past what's structurally necessary to
/// reach it (multi-word labels keep packing multiple words per line).
fn uniformize_wrapped_labels(
    raw_labels: &[String],
    natural_labels: Vec<String>,
    gap_px: u32,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> Vec<String> {
    if gap_px == 0 {
        return natural_labels;
    }
    let target_lines = natural_labels
        .iter()
        .map(|w| w.matches('\n').count() + 1)
        .max()
        .unwrap_or(1);
    raw_labels
        .iter()
        .zip(natural_labels.iter())
        .map(|(raw, natural)| {
            let natural_lines = natural.matches('\n').count() + 1;
            if natural_lines >= target_lines {
                natural.clone()
            } else {
                wrap_label_to_line_count(raw, target_lines, max_width, font_size, metrics)
                    .unwrap_or_else(|| natural.clone())
            }
        })
        .collect()
}

/// Shared body for `layout_x_axis`'s and `layout_y_axis`'s `label_angle`
/// override branch (R2): stamp `override_angle` onto
/// every tick, then elide-to-fit any label whose rotated footprint still
/// collides with its neighbor's per-tick budget — no cull recovery, on
/// either axis (spec SS7 for x; R2 for y).
///
/// The two axes differ only in WHICH trig factor turns a label's flat width
/// into its footprint along the collision dimension, and what that
/// dimension's per-tick budget is:
/// - x judges HORIZONTAL collision via `cos|angle|` (rotation SHRINKS the
///   horizontal footprint as a label tips toward vertical) against
///   `cascade_slot_w`.
/// - y judges VERTICAL collision via `sin|angle|` (rotation GROWS the
///   vertical footprint — the transpose of x) against `cascade_slot_h`.
///
/// Callers pass their own `projection_factor` and `budget` for this; see
/// `rotated_x_label_extent`'s and `rotated_y_label_extent`'s docs for why the
/// two factors are transposed. `position_fn` resolves each tick's pixel
/// position (already orient-specific — each caller's own
/// uniform-slot/projected/band-center closure).
fn stamp_override_angle_with_elide(
    tick_labels: &[String],
    override_angle: f64,
    projection_factor: f64,
    budget: f64,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
    position_fn: impl Fn(usize) -> f64,
) -> (Vec<TickLayout>, Option<AxisLabelWarning>) {
    let widths: Vec<f64> = tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();
    let any_still_colliding = widths.iter().any(|w| *w * projection_factor > budget);
    let mut elided_count: u32 = 0;
    let ticks: Vec<TickLayout> = tick_labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let w = widths[i];
            let needs_elide = any_still_colliding && (w * projection_factor > budget);
            let final_label = if needs_elide {
                elided_count += 1;
                let elide_budget = budget / projection_factor.max(1e-6);
                elide_to_fit(label, elide_budget, label_font_size, metrics)
            } else {
                label.clone()
            };
            TickLayout {
                position: position_fn(i),
                label: final_label,
                label_angle: override_angle,
                elided: needs_elide,
                culled: false,
                label_font_size: None,
                is_major: true,
                anchor: false,
                origin: false,
            }
        })
        .collect();
    let warning = if elided_count > 0 {
        Some(AxisLabelWarning::LabelsElided {
            count: elided_count,
        })
    } else {
        None
    };
    (ticks, warning)
}

/// The extra pitch budget (beyond the raw adjacent gap) a presentation
/// stage must leave at a pitch of `slot_w`: the LARGER of the requested
/// pixel gap ([`LabelGapPolicy::gap_px`]) and [`LABEL_OVERLAP_TOLERANCE`]'s
/// own fraction of the pitch. The gap REPLACES the tolerance once it asks
/// for more room, rather than the two stacking. S3 (rotate) does not call
/// this: it keeps its own `slot_w - gap` budget.
fn collision_need(slot_w: f64, label_gap: LabelGapPolicy) -> f64 {
    label_gap.gap().max(LABEL_OVERLAP_TOLERANCE * slot_w)
}

/// Whether every adjacent PAIR of `widths` leaves at least `need` px of the
/// shared pitch `slot_w` beyond their own combined half-widths: `slot_w -
/// (w_i + w_{i+1}) / 2 >= need` for every `i`. Two labels only ever collide
/// with each other, so the fit question is each pair's own average width
/// against the pitch between them, not the axis's single widest label
/// tested against every pair.
///
/// A lone label has no real neighbor to pair with, but still has to fit in
/// its own pitch: it degrades to a self-pair (`w` on both sides of the
/// average, i.e. `slot_w - w >= need`). An empty slice has nothing to
/// collide and always fits.
fn fits_adjacent_pairs(widths: &[f64], slot_w: f64, need: f64) -> bool {
    if widths.len() < 2 {
        return widths.iter().all(|&w| slot_w - w >= need);
    }
    widths.windows(2).all(|w| slot_w - (w[0] + w[1]) / 2.0 >= need)
}

/// S0 (flat) and S1 (wrap) alone, at full font size — the two stages
/// [`cascade_collision_recovery`]'s continuous-x thinning stage must see
/// fail before it tries a stride, and the same two stages each candidate
/// stride's survivors are re-judged against. Extracted out of
/// [`resolve_presentation`] so both call sites share one body instead of a
/// thinning-specific copy drifting from the cascade's own S0/S1. See
/// [`resolve_presentation`]'s doc for the full stage order.
fn resolve_presentation_unshrunk(
    labels: &[String],
    slot_w: f64,
    label_font_size: f64,
    label_gap: LabelGapPolicy,
    metrics: &dyn TextMetrics,
) -> Option<CascadeResult> {
    let all_visible = vec![true; labels.len()];
    let widths: Vec<f64> = labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();
    let need = collision_need(slot_w, label_gap);

    // S0 — Flat: see `fits_adjacent_pairs`'s doc for why this judges
    // adjacent pairs instead of every label against one uniform threshold.
    if fits_adjacent_pairs(&widths, slot_w, need) {
        return Some(CascadeResult {
            labels: labels.to_vec(),
            angle: 0.0,
            font_size: None,
            visible: all_visible,
            strategy: CascadeStrategy::Flat,
        });
    }

    // S1 — Wrap: try wrapping all labels within the per-label budget
    // `slot_w - need`. All must successfully wrap AND fit.
    let threshold = slot_w - need;
    let wrapped: Vec<Option<String>> = labels
        .iter()
        .map(|l| wrap_label(l, threshold, label_font_size, metrics))
        .collect();
    let all_wrap_ok = wrapped.iter().all(|w| w.is_some());
    if all_wrap_ok {
        let wrapped_labels: Vec<String> = wrapped.into_iter().flatten().collect();
        let all_fit = wrapped_labels
            .iter()
            .all(|w| measure_multiline_width(w, label_font_size, metrics) <= threshold);
        if all_fit {
            // Uniform-degradation requirement (spec §4.6) — see
            // `uniformize_wrapped_labels`'s doc.
            let final_labels = uniformize_wrapped_labels(
                labels,
                wrapped_labels,
                label_gap.gap_px(),
                threshold,
                label_font_size,
                metrics,
            );
            return Some(CascadeResult {
                labels: final_labels,
                angle: 0.0,
                font_size: None,
                visible: all_visible,
                strategy: CascadeStrategy::Wrapped,
            });
        }
    }
    None
}

/// Try, in cascade order, S0 (flat) -> S1 (wrap) -> S2a (font-reduced flat)
/// -> S2b (font-reduced wrap) -> S3 (rotate) against `labels` within
/// `slot_w` (spec §4.6, D9). S0-S2 budget against `collision_need(slot_w,
/// label_gap)` (the larger of `label_gap`'s pixel gap and
/// `LABEL_OVERLAP_TOLERANCE`'s own share of the pitch); S3 subtracts
/// `label_gap.gap()` directly. S0 and S1 are [`resolve_presentation_unshrunk`]'s body, tried
/// first; S2/S3 below only run once that returns `None`. Returns `None` if
/// no stage resolves collision, i.e. even -90° rotation doesn't fit
/// `labels` within `slot_w`.
///
/// Shared by `cascade_collision_recovery`'s initial attempt (all N tick
/// labels, `slot_w` = the per-tick budget) AND its S4 cull stage's
/// RE-ATTEMPT at the culled stride: S4 computes a stride from -90°'s footprint alone — the
/// SMALLEST possible footprint of any presentation — so a stride wide
/// enough for rotated survivors to fit says nothing about whether a
/// LESS-degraded presentation (flat, wrapped, font-reduced) would ALSO now
/// fit the wider per-surviving-label budget the stride opens up. Without
/// this re-attempt, culled survivors always inherited the rotated
/// presentation the stride's own fit test happened to use, even when e.g. a
/// handful of short numeric labels at a wide culled pitch trivially fit
/// flat — rotated sparse labels are strictly worse than flat ones. Calling
/// this function again at the culled stride, against survivor labels only,
/// re-derives the least-degraded presentation that ACTUALLY fits the
/// density the cull just produced; a genuinely wide label set that still
/// doesn't fit flat/wrapped/font-reduced at the wider pitch legitimately
/// keeps -90° rotation, since S3's loop inside this second call re-tries
/// every angle again.
fn resolve_presentation(
    labels: &[String],
    slot_w: f64,
    label_font_size: f64,
    label_gap: LabelGapPolicy,
    metrics: &dyn TextMetrics,
) -> Option<CascadeResult> {
    if let Some(result) =
        resolve_presentation_unshrunk(labels, slot_w, label_font_size, label_gap, metrics)
    {
        return Some(result);
    }

    let all_visible = vec![true; labels.len()];
    let line_h = metrics.line_height(label_font_size);
    let need = collision_need(slot_w, label_gap);

    // S2 — Font shrink: try at reduced font size. If it doesn't help, proceed
    // at ORIGINAL font size (rotation at smaller fonts is hard to read).
    let reduced_fs = label_font_size * FONT_SHRINK_FACTOR;
    let reduced_widths: Vec<f64> = labels
        .iter()
        .map(|s| metrics.measure_width(s, reduced_fs))
        .collect();

    // S2a: reduced font, flat. Same adjacent-pair test as S0, at the
    // smaller widths.
    if fits_adjacent_pairs(&reduced_widths, slot_w, need) {
        return Some(CascadeResult {
            labels: labels.to_vec(),
            angle: 0.0,
            font_size: Some(reduced_fs),
            visible: all_visible,
            strategy: CascadeStrategy::FontReduced,
        });
    }

    // S2b: reduced font + wrapping, same per-label budget as S1. Same
    // uniform-degradation treatment as S1.
    let threshold = slot_w - need;
    let wrapped_reduced: Vec<Option<String>> = labels
        .iter()
        .map(|l| wrap_label(l, threshold, reduced_fs, metrics))
        .collect();
    let all_wrap_reduced_ok = wrapped_reduced.iter().all(|w| w.is_some());
    if all_wrap_reduced_ok {
        let wrapped_labels: Vec<String> = wrapped_reduced.into_iter().flatten().collect();
        let all_fit = wrapped_labels
            .iter()
            .all(|w| measure_multiline_width(w, reduced_fs, metrics) <= threshold);
        if all_fit {
            let final_labels = uniformize_wrapped_labels(
                labels,
                wrapped_labels,
                label_gap.gap_px(),
                threshold,
                reduced_fs,
                metrics,
            );
            return Some(CascadeResult {
                labels: final_labels,
                angle: 0.0,
                font_size: Some(reduced_fs),
                visible: all_visible,
                strategy: CascadeStrategy::FontReduced,
            });
        }
    }

    // S3 — Graduated rotation: try each angle from ANGLE_CASCADE (skip 0.0, already tried).
    // Use ORIGINAL labels and ORIGINAL font size. `rotated_x_label_footprint`
    // (not a bare `cos(angle)` projection) gives -90° a real fit test — see
    // its doc for why the naive projection made this stage auto-pass and
    // culling (S4) unreachable (spec §4.6, F-L07-04). The budget is
    // `slot_w - cull_gap` (no `LABEL_OVERLAP_TOLERANCE`, matching this
    // stage's pre-existing convention) so a positive `cull_threshold`
    // tightens rotation too — a genuinely-fitting rotation only wins when it
    // ALSO satisfies the requested gap.
    let widths: Vec<f64> = labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();
    let cull_gap = label_gap.gap();
    let s3_budget = slot_w - cull_gap;
    for &angle in &ANGLE_CASCADE[1..] {
        let all_fit = widths
            .iter()
            .all(|w| rotated_x_label_footprint(angle, *w, line_h) <= s3_budget);
        if all_fit {
            return Some(CascadeResult {
                labels: labels.to_vec(),
                angle,
                font_size: None,
                visible: all_visible,
                strategy: CascadeStrategy::Rotated { angle },
            });
        }
    }

    None
}

/// Merge a presentation's SURVIVOR-only label text back into the full,
/// original-length label vector `layout_x_axis` expects (S4 cull
/// re-selection): visible indices consume the next resolved
/// survivor label IN ORDER (both `visible` and `survivor_labels` were built
/// by iterating the same original index order, so they stay aligned);
/// culled indices keep their untouched original text, since a culled tick
/// never renders (`TickLayout.culled`) and its text is therefore inert.
fn merge_survivor_labels(
    original: &[String],
    visible: &[bool],
    survivor_labels: Vec<String>,
) -> Vec<String> {
    let mut survivors = survivor_labels.into_iter();
    original
        .iter()
        .zip(visible)
        .map(|(orig, vis)| {
            if *vis {
                survivors.next().unwrap_or_else(|| orig.clone())
            } else {
                orig.clone()
            }
        })
        .collect()
}

/// Which of `n` labels a stride cull keeps: every `stride`-th, counted from
/// `origin` in both directions (`origin: 0` starts from the first).
/// `origin` need not itself survive the stride — the count simply starts
/// there.
pub(super) fn stride_visibility(n: usize, stride: usize, origin: usize) -> Vec<bool> {
    (0..n).map(|i| (i + stride - origin % stride).is_multiple_of(stride)).collect()
}

/// [`stride_visibility`], but a `Log` axis's decade ticks (`anchors[i] ==
/// true`) always survive — ggplot2's/d3's "always label the powers" rule,
/// enforced at the cull so no width drops a decade in
/// favor of a non-decade multiple. `anchors` is `ScaleKind::tick_anchors`'s
/// mask, in the same index order as the labels being culled; empty (every
/// non-`Log` axis, and a `Log` axis whose powers alone already met its tick
/// budget with no multiples in play) or all-`true` reduces to plain
/// `stride_visibility` (with `origin` passed through), so this is
/// byte-identical there. A length mismatch (defensive only — the caller
/// always pairs `anchors` with its own label count) also falls back to
/// `stride_visibility`. The anchored branch below (non-empty, non-all-true
/// `anchors`) always counts from index `0`: `origin` only ever reaches the
/// `stride_visibility` fallback, since every caller that carries a
/// meaningful `tick_origin` is non-`Log` (`ScaleKind::tick_origin` is
/// `None` for `Log`, whose decades already anchor here).
///
/// With `p` the smallest index gap between consecutive anchors:
/// - `stride >= p`: keep only anchors, every `ceil(stride / p)`-th from the
///   first — consecutive survivors are `ceil(stride / p) * p >= stride`
///   slots apart, so the caller's `stride * slot_w` budget still holds.
/// - `stride < p`: keep every anchor, plus each non-anchor whose offset
///   from the preceding anchor (or from index `0`, for one before the
///   first) is a multiple of `stride` AND whose gap to the next anchor is
///   at least `stride` — the same "every stride-th slot survives" rule
///   `stride_visibility` applies, restarted at each anchor so an anchor is
///   never skipped waiting for its own turn.
pub(super) fn anchored_visibility(n: usize, stride: usize, anchors: &[bool], origin: usize) -> Vec<bool> {
    if anchors.is_empty() || anchors.len() != n || anchors.iter().all(|&a| a) {
        return stride_visibility(n, stride, origin);
    }
    let anchor_idx: Vec<usize> = anchors.iter().enumerate().filter(|&(_, &a)| a).map(|(i, _)| i).collect();
    if anchor_idx.is_empty() {
        return stride_visibility(n, stride, origin);
    }
    let p = anchor_idx.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(n.max(1));
    let mut keep = vec![false; n];
    if stride >= p {
        let k = stride.div_ceil(p);
        for (count, &idx) in anchor_idx.iter().enumerate() {
            if count % k == 0 {
                keep[idx] = true;
            }
        }
    } else {
        for &idx in &anchor_idx {
            keep[idx] = true;
        }
        let non_anchor_survivors: Vec<usize> = (0..n)
            .filter(|&i| {
                if keep[i] {
                    return false;
                }
                let preceding = anchor_idx.iter().rev().find(|&&a| a < i).copied().unwrap_or(0);
                let next = anchor_idx.iter().find(|&&a| a > i).copied();
                let offset_ok = (i - preceding) % stride == 0;
                let gap_ok = next.is_none_or(|na| na - i >= stride);
                offset_ok && gap_ok
            })
            .collect();
        for i in non_anchor_survivors {
            keep[i] = true;
        }
    }
    keep
}

/// The x tick labels' cull stage: for each stride in `strides`, in order,
/// keep every `stride`-th label, decades of a `Log` axis always surviving
/// ([`anchored_visibility`]), lay the survivors out with `present` against
/// the `stride · slot_w` each survivor now spans, and accept the first
/// presentation that `fits`. Returns the accepted stride, its visibility
/// mask and its presentation, or `None` when no stride is both presentable
/// and fitting.
///
/// The collision cascade's S4 tries the one stride it sized from the -90°
/// footprint and accepts it unconditionally; an axis whose drawn band is over
/// its budget tries every stride from 2 up and accepts the first whose band
/// is within it. The two differ only in their strides and `fits`. `origin`
/// is forwarded to [`anchored_visibility`] unchanged — the caller's own
/// axis origin (`AxisInput.tick_origin`) on a continuous-x stride cull, `0`
/// everywhere a stride cull must keep counting from the first label
/// (parity, y, an explicit `label_angle`).
fn cull_by_stride<P>(
    labels: &[String],
    slot_w: f64,
    strides: impl IntoIterator<Item = usize>,
    anchors: &[bool],
    origin: usize,
    mut present: impl FnMut(&[String], f64) -> Option<P>,
    fits: impl Fn(&P) -> bool,
) -> Option<(usize, Vec<bool>, P)> {
    strides.into_iter().find_map(|stride| {
        let visible = anchored_visibility(labels.len(), stride, anchors, origin);
        let survivor_labels: Vec<String> = labels
            .iter()
            .zip(&visible)
            .filter(|(_, v)| **v)
            .map(|(l, _)| l.clone())
            .collect();
        present(&survivor_labels, stride as f64 * slot_w)
            .filter(|p| fits(p))
            .map(|p| (stride, visible, p))
    })
}

/// Run the graduated collision cascade (spec SS4.1). Tries recovery strategies in
/// order (S0 flat -> S1 wrap -> D continuous-x thinning -> S2 font shrink ->
/// S3 rotate -> S4 cull -> S5 elide), returning as soon as one resolves all
/// collisions.
///
/// **D — continuous-x thinning.** Once S0 and S1 both fail at full font,
/// and only on a continuous `x` axis (`is_continuous`; a categorical or
/// scale-less axis skips straight to S2, unthinned) whose policy
/// `may_cull()`, try evenly spaced strides — the SAME `anchored_visibility`
/// a `Log` axis's decade cull uses, so temporal, log and linear x all thin
/// through one primitive — before ever shrinking the font or rotating. The
/// first stride whose survivors present Flat or Wrapped at FULL font with
/// at least [`super::MIN_EVENLY_THINNED_X_LABELS`] of them wins, read as
/// `Culled`. Otherwise this stage changes nothing and the cascade falls
/// through to S2, S3, S4, S5. A sparse continuous axis therefore keeps
/// full-size flat labels over a thinned subset rather than shrinking or
/// rotating every one of them; a genuinely dense axis still rotates or
/// shrinks with every label kept, since D only accepts a stride that
/// resolves flat or wrapped.
///
/// [`LabelGapPolicy`] (spec §4.6, F-L07-04 / D9; `#146`) carries the two
/// independent things the old `cull_threshold: u32` conflated.
///
/// [`gap_px`](LabelGapPolicy::gap_px) is the minimum pixel gap
/// required between adjacent VISIBLE tick labels — `Theme(cull_threshold=)`'s
/// docstring — not a label-count gate, and not merely an S4 stride floor:
/// S0-S2 fold it into `collision_need` (the larger of `gap_px` and
/// `LABEL_OVERLAP_TOLERANCE`'s own share of the pitch), and S3 subtracts it
/// directly, since "does this arrangement leave at least `gap_px`
/// px between labels" is a stronger question than "do labels overlap at
/// all." A stage only resolves collision when it ALSO satisfies the gap —
/// so a chart whose labels already fit flat (or rotate cleanly) with plenty
/// of raw overlap headroom can still need culling once a large gap is
/// requested. At `0`, S3 and S4 collapse to the pure-overlap check (their
/// own gap term is zero); S0-S2 do not, since `collision_need`'s `max`
/// against `LABEL_OVERLAP_TOLERANCE`'s pitch share still applies regardless
/// of `gap_px`. See
/// `rotated_x_label_footprint`'s doc for why -90° rotation (S3's last angle)
/// is a real fit test rather than an automatic pass.
///
/// [`may_cull`](LabelGapPolicy::may_cull) gates S4 alone. `false` sends a
/// still-collided axis to S5 (elision, which leaves an infeasible truncation
/// intact) rather than ever dropping a label; `true` lets S4 size its
/// stride from the widest rotated footprint plus `gap_px` directly (the
/// raw pixel gap, not `collision_need`'s tolerance-widened budget — see
/// the S4 stride comment below) so enough labels drop to clear it. The
/// legacy `Theme(cull_threshold=N)` spelling
/// couples the two ([`LabelGapPolicy::from_cull_threshold`]), so `0` still
/// means "no gap AND never cull" and `N` still means "`N` px AND may cull" —
/// what the split adds is the other two corners of that square.
///
/// `label_overlap` (B5 unit 6b) biases the cascade onto an explicit primitive
/// instead of running the graduated flow:
/// - [`LabelOverlap::ShowAll`] short-circuits to S0-Flat with every label
///   visible (no cull/elide), so labels may overlap — the user's choice.
/// - [`LabelOverlap::Parity`] short-circuits to a stride-2 cull (S4 with a fixed
///   parity stride), independent of measured width.
/// - [`LabelOverlap::Rotate`] short-circuits to S3-rotate at the steepest cascade
///   angle.
/// - `None` / [`LabelOverlap::Greedy`] runs the graduated S0-S5 cascade
///   (default): no forced primitive.
///
/// S4 (cull) re-selects survivors' presentation at the culled density via
/// `resolve_presentation` — see that function's doc for why a stride computed against
/// -90°'s footprint doesn't by itself justify keeping -90° rotation once
/// the stride opens up more room per surviving label.
fn cascade_collision_recovery(
    labels: &[String],
    slot_w: f64,
    label_font_size: f64,
    label_gap: LabelGapPolicy,
    label_overlap: Option<LabelOverlap>,
    metrics: &dyn TextMetrics,
    frame: TickCullFrame<'_>,
) -> CascadeResult {
    let n = labels.len();
    let all_visible = vec![true; n];

    // ── label_overlap override (B5 unit 6b) ─────────────────────────────────
    // Bias the cascade onto an explicit primitive. `None`/`Greedy` skip this and
    // fall through to the graduated cascade below.
    match label_overlap {
        Some(LabelOverlap::ShowAll) => {
            // Show ALL labels flat: skip the cull/elide stages entirely.
            return CascadeResult {
                labels: labels.to_vec(),
                angle: 0.0,
                font_size: None,
                visible: all_visible,
                strategy: CascadeStrategy::Flat,
            };
        }
        Some(LabelOverlap::Parity) => {
            // Stride-2 decimation: show every other label, counted from the
            // first — Vega-Lite's `labelOverlap: "parity"` is index parity,
            // not the axis's nicest tick, so this always passes origin 0
            // regardless of `AxisInput.tick_origin`.
            return CascadeResult {
                labels: labels.to_vec(),
                angle: 0.0,
                font_size: None,
                visible: anchored_visibility(n, 2, frame.anchors, 0),
                strategy: CascadeStrategy::Culled { stride: 2 },
            };
        }
        Some(LabelOverlap::Rotate) => {
            // Force the cascade's rotate stage at the steepest angle.
            let angle = *ANGLE_CASCADE.last().unwrap(); // -90.0
            return CascadeResult {
                labels: labels.to_vec(),
                angle,
                font_size: None,
                visible: all_visible,
                strategy: CascadeStrategy::Rotated { angle },
            };
        }
        // Greedy is the default cascade behavior — fall through.
        None | Some(LabelOverlap::Greedy) => {}
    }

    // S0/S1 — flat / wrap at full font. Tried standalone (not via the full
    // `resolve_presentation`) only when D's thinning gate below might apply;
    // a categorical or non-culling axis goes straight to the S0-S3 cascade
    // a few lines down (D is skipped, so these axes never reach a stride).
    if frame.continuous && label_gap.may_cull() {
        if let Some(result) =
            resolve_presentation_unshrunk(labels, slot_w, label_font_size, label_gap, metrics)
        {
            return result;
        }

        // D — continuous-x thinning. See this function's doc for the
        // stage's placement and rationale. `present` folds the "at least
        // MIN_EVENLY_THINNED_X_LABELS survivors" requirement into itself
        // (declining a stride outright when too few labels would survive
        // it) rather than threading the count through `cull_by_stride`'s
        // separate `fits` predicate, since `fits` only ever sees the
        // presentation, not the stride's mask.
        if let Some((stride, visible, presentation)) = cull_by_stride(
            labels,
            slot_w,
            2..=n,
            frame.anchors,
            frame.origin,
            |survivors, survivor_w| {
                if survivors.len() < super::MIN_EVENLY_THINNED_X_LABELS {
                    return None;
                }
                resolve_presentation_unshrunk(survivors, survivor_w, label_font_size, label_gap, metrics)
            },
            |_| true,
        ) {
            return CascadeResult {
                labels: merge_survivor_labels(labels, &visible, presentation.labels),
                angle: presentation.angle,
                font_size: presentation.font_size,
                visible,
                strategy: CascadeStrategy::Culled {
                    stride: stride as u32,
                },
            };
        }
    }

    // S0 through S3 — flat / wrap / font-reduced / rotate, all against the
    // full N-label set at `slot_w`. See `resolve_presentation`'s doc; it is
    // the extracted, reusable body of these four stages, shared with S4's
    // post-cull re-attempt below. When D's gate above already tried and
    // failed S0/S1, this redoes that pure, deterministic check once more
    // before proceeding to S2/S3 — cheap, and it keeps this call the single
    // shared entry both the categorical and the post-D-failure path use.
    if let Some(result) = resolve_presentation(labels, slot_w, label_font_size, label_gap, metrics)
    {
        return result;
    }

    // Neither flat, wrap, font-reduced, nor even -90° rotation fit every
    // label within `slot_w` — proceed to S4 (cull) / S5 (elide). Both need
    // the full-label widths/line-height `resolve_presentation` already
    // computed and discarded internally; recompute once here.
    let widths: Vec<f64> = labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();
    let line_h = metrics.line_height(label_font_size);
    let cull_gap = label_gap.gap();

    // S4 — Tick culling: PIXEL-GAP semantic (spec §4.6, F-L07-04 / D9).
    // The gap is a required pixel distance between adjacent VISIBLE tick
    // labels, never a label COUNT gate; whether this stage runs at all is
    // the policy's own `may_cull` (`#146`), so a policy that forbids culling
    // falls through to S5 elision instead of ever dropping a label,
    // regardless of how many labels there are or how wide the gap is.
    // Use -90 degrees (last/steepest angle in cascade) for the STRIDE
    // computation below — the smallest possible footprint of any
    // presentation, so it gives the minimum stride any presentation could
    // need. The PRESENTATION actually used for the survivors is re-derived
    // afterward (see below).
    let best_angle = *ANGLE_CASCADE.last().unwrap(); // -90.0
    let cos_best = best_angle.to_radians().cos().abs();

    if label_gap.may_cull() {
        // Find max footprint at the best angle (same real-fit formula as S3 —
        // a bare `w * cos_best` projection is ~0 at -90° and would never
        // trigger culling; see `rotated_x_label_footprint`'s doc).
        let max_projected = widths
            .iter()
            .map(|w| rotated_x_label_footprint(best_angle, *w, line_h))
            .fold(0.0_f64, f64::max);

        // Minimum stride N such that the widest rotated label's footprint
        // PLUS the required gap fits within N slots — S3's own
        // `footprint <= effective_slot_w - gap_px` budget, just solved
        // for stride instead of a fixed angle:
        //   footprint + gap_px <= N * slot_w
        //   N >= (footprint + gap_px) / slot_w
        // A zero-width label set still culls down to a spaced subset when
        // `slot_w` alone can't satisfy the gap
        // ("threshold larger than any gap culls down to a spaced subset").
        let stride = if slot_w <= 0.0 {
            1_u32
        } else {
            ((max_projected + cull_gap) / slot_w).ceil().max(1.0) as u32
        };

        if stride > 1 {
            // `stride` was sized against -90°'s footprint alone, the SMALLEST footprint
            // of any presentation — it says nothing about whether a
            // LESS-degraded presentation now fits the wider
            // per-surviving-label budget the stride opens up. Re-run the
            // presentation cascade (flat -> wrap -> font-reduced -> rotate,
            // same stage order) at the culled density, against only the
            // labels that will actually render, so e.g. a handful of short
            // numeric labels culled to a wide pitch render FLAT instead of
            // inheriting the -90° rotation that made culling necessary at
            // the ORIGINAL, narrower pitch. `resolve_presentation` is
            // guaranteed to return `Some` here: the stride was sized so
            // -90° fits every survivor's (a subset of all N labels')
            // footprint at `stride · slot_w`, and S3's own loop inside
            // this second call re-tries -90° again — the `None` arm is a
            // defensive fallback only, kept in case of a floating-point
            // boundary surprise, and reproduces the pre-fix outcome
            // (rotated survivors) if it were ever hit. The one stride is
            // accepted whatever it presents, and `present` never declines,
            // so the cull always keeps it; the `None` arm below is that same
            // rotated fallback at the stride's own mask, kept so a declining
            // presentation step could only ever yield a culled result.
            let culled = cull_by_stride(
                labels,
                slot_w,
                [stride as usize],
                frame.anchors,
                frame.origin,
                |survivors, survivor_w| {
                    Some(resolve_presentation(
                        survivors,
                        survivor_w,
                        label_font_size,
                        label_gap,
                        metrics,
                    ))
                },
                |_| true,
            );
            let (visible, presentation) = match culled {
                Some((_, visible, presentation)) => (visible, presentation),
                None => (anchored_visibility(n, stride as usize, frame.anchors, frame.origin), None),
            };
            let (final_labels, angle, font_size) = match presentation {
                Some(presentation) => (
                    merge_survivor_labels(labels, &visible, presentation.labels),
                    presentation.angle,
                    presentation.font_size,
                ),
                None => (labels.to_vec(), best_angle, None),
            };

            return CascadeResult {
                labels: final_labels,
                angle,
                font_size,
                visible,
                strategy: CascadeStrategy::Culled { stride },
            };
        }

        // stride == 1 means labels already sit >= `gap_px` apart at
        // -90 without dropping any — return as rotated, not culled.
        return CascadeResult {
            labels: labels.to_vec(),
            angle: best_angle,
            font_size: None,
            visible: all_visible,
            strategy: CascadeStrategy::Rotated { angle: best_angle },
        };
    }

    // S5 — Elision: last resort. Use -90 degrees. Elide labels that still collide.
    // Per-label fit uses the same real footprint as S3/S4 (`rotated_x_label_footprint`)
    // rather than a bare `w * cos_best` projection — reaching S5 means at least one
    // label's TRUE footprint exceeded `slot_w` at every cascade angle (the S3 loop
    // above already tried -90°), so the elide decision must agree with that, or
    // S5 silently elides nothing (the same auto-pass bug one stage further downstream).
    //
    // BUT: the elide REMEDY (`elide_to_fit`, which shortens the label's TEXT)
    // can only ever shrink the footprint's WIDTH term (`cos_best * w`) — it
    // cannot touch the WIDTH-INDEPENDENT `sin_best * line_h` term (quality-
    // review finding S3). At -90°, `cos_best` is ~6.12e-17 (not exactly 0 in
    // IEEE-754 `f64`), so for any REALISTIC label width that term is
    // negligible: truncating the text changes the rendered footprint by a
    // fraction of a pixel, never enough to matter, and `elide_to_fit`
    // degenerates to its own `ellipsis_w >= max_width` early return — a bare
    // "…", zero information, for a remedy that could never have satisfied
    // the constraint it exists to answer. (The only case where the width
    // term is NOT negligible is an astronomically large mocked width, e.g.
    // `cascade_s5_elision`'s `1e18`-px test double: `cos_best * 1e18 ≈ 61px`,
    // a real, non-negligible reduction elision can actually deliver.)
    //
    // Elision is therefore gated on whether truncation could plausibly move
    // the needle at all: if even the WIDEST label's own width contributes
    // less than `ELIDE_MIN_WIDTH_CONTRIBUTION_PX` to its footprint, no
    // amount of truncation across the axis can help, and eliding anyway
    // would only destroy content for nothing — fall back to `Rotated` with
    // every label intact at `best_angle`, the outcome for a policy whose
    // `may_cull` is false (reached by `cull_threshold=0` and by
    // `cull_labels=False` at any gap), rather than collapsing a legible axis
    // into an axis of bare ellipses.
    const ELIDE_MIN_WIDTH_CONTRIBUTION_PX: f64 = 1.0;
    let max_width_contribution = widths.iter().fold(0.0_f64, |acc, w| acc.max(cos_best * w));
    if max_width_contribution < ELIDE_MIN_WIDTH_CONTRIBUTION_PX {
        return CascadeResult {
            labels: labels.to_vec(),
            angle: best_angle,
            font_size: None,
            visible: all_visible,
            strategy: CascadeStrategy::Rotated { angle: best_angle },
        };
    }

    // NOTE (`#146`): this fit test is `projected > slot_w`, with no
    // `cull_gap` term — unlike S0-S4 above, which all charge the gap
    // against the available width (via `collision_need` on S0-S2, directly
    // on S3-S4). That is deliberate, not a missed spot:
    // S5 is reached only because truncation is already the last resort, and
    // charging it the gap on top of the raw slot width would shrink an
    // already-minimal budget further, elide MORE text than a genuine
    // overlap requires, for a gap that a `-90°` truncation could never buy
    // back anyway (the label's own width barely contributes to its
    // footprint at that angle — see `ELIDE_MIN_WIDTH_CONTRIBUTION_PX`
    // above). Before `#146` this asymmetry was unreachable: S5 was only
    // ever entered at `gap_px == 0`, where subtracting the gap would have
    // been a no-op either way. The `{N, false}` state now reaches S5 with a
    // nonzero gap and exposes it for the first time. Do not "fix" this by
    // adding a gap term here — that would destroy text for no gain.
    let mut elided_count: u32 = 0;
    let elided_labels: Vec<String> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let projected = rotated_x_label_footprint(best_angle, widths[i], line_h);
            if projected > slot_w {
                elided_count += 1;
                let budget = if cos_best > 1e-6 {
                    slot_w / cos_best
                } else {
                    slot_w
                };
                elide_to_fit(label, budget, label_font_size, metrics)
            } else {
                label.clone()
            }
        })
        .collect();

    CascadeResult {
        labels: elided_labels,
        angle: best_angle,
        font_size: None,
        visible: all_visible,
        strategy: CascadeStrategy::Elided {
            count: elided_count,
        },
    }
}

/// The tick labels one x axis draws across a panel spanning `panel_x` to
/// `panel_x + panel_w`: each tick's pixel and the label the collision cascade
/// (or an explicit `label_angle`) resolves for it, with its angle, font size,
/// text and cull state. The one place that decision is made:
/// [`layout_x_axis`] lays the axis out from it, and
/// [`estimate_x_label_band`] reserves the gutter for exactly what it draws.
pub(super) fn lay_out_x_tick_labels(
    input: &AxisInput,
    panel_x: f64,
    panel_w: f64,
    ctx: &LabelBandCtx<'_>,
) -> (Vec<TickLayout>, Option<AxisLabelWarning>) {
    let LabelBandCtx {
        label_font_size,
        label_gap,
        metrics,
        ..
    } = *ctx;
    let n = input.tick_labels.len();
    let slot_w = if n > 0 { panel_w / n as f64 } else { 0.0 };
    // Uniform-slot fallback range (left → right, in pixel order): slot `i`'s center
    // is `panel_x + (i + 0.5)*slot_w` via `Axis1D::uniform_center`.
    let slot_axis = Axis1D {
        lo: panel_x,
        hi: panel_x + panel_w,
    };

    // Continuous axes: place each tick at its scale-projected pixel; categorical
    // axes (no projected fractions) keep the uniform-slot center. The closure
    // resolves a tick's position by index against whichever placement applies.
    let projected = project_tick_positions(input, (slot_axis.lo, slot_axis.hi));
    // Categorical axes (F-L04-03, GH #67): place each tick at the scale's own
    // band center — the same pixel its mark gets — so labels and grid lines
    // agree with the marks. `slot_axis` is bit-for-bit the pixel interval
    // `scene_build::resolve_panel_scales` gives this panel's mark scales, so a
    // panel-extent placement resolves to those same centers exactly. Absent
    // (`None`) → the uniform-slot formula, for the scale-less `compute_layout`
    // binding — and so does a carrier that is not paired 1:1 with the labels
    // (see `paired_band_centers`).
    let resolved_centers = input
        .categorical_placement
        .as_ref()
        .map(|p| p.centers(slot_axis.lo, slot_axis.hi));
    let band_centers = paired_band_centers(resolved_centers.as_deref(), n);
    let tick_position = |i: usize| -> f64 {
        match (&projected, band_centers) {
            (Some(px), _) => px[i],
            (None, Some(centers)) => centers[i],
            (None, None) => slot_axis.uniform_center(i, slot_w),
        }
    };
    // The collision cascade judges label fit against the available horizontal
    // budget per tick. Without a scale to ask (`compute_layout`) that is `slot_w`.
    // For continuous axes the spacing is non-uniform (log/pow/symlog), so use the
    // *minimum* adjacent gap between projected positions — the worst case — to
    // avoid under-counting collisions where ticks bunch toward one end.
    // A categorical axis's bands may pack into a sub-interval (an explicit
    // `range=`) or spread by padding, so the per-tick budget is the band step,
    // not the full-panel `slot_w` — use the min adjacent
    // gap between band centers so label collisions are judged against the true
    // (tighter) spacing.
    let cascade_slot_w = match (&projected, band_centers) {
        (Some(px), _) => min_adjacent_gap(px).unwrap_or(slot_w),
        (None, Some(centers)) => min_adjacent_gap(centers).unwrap_or(slot_w),
        (None, None) => slot_w,
    };
    // Continuous-x thinning (stage D, below) gates on there being a scale
    // to project through at all: a categorical axis's `band_centers` and
    // the scale-less `compute_layout` carrier (neither `projected`) both
    // skip it.
    let frame = TickCullFrame {
        anchors: &input.tick_anchors,
        origin: input.tick_origin.unwrap_or(0),
        continuous: projected.is_some(),
    };

    let (ticks, warning) = if let Some(override_angle) = input.overrides.labels.angle {
        // label_angle_override always bypasses the cascade (spec SS7). Shared
        // body: see `stamp_override_angle_with_elide`'s
        // doc for the x/y transpose (x's projection factor is `cos|angle|`
        // against `cascade_slot_w`).
        let cos_factor = override_angle.to_radians().cos().abs();
        stamp_override_angle_with_elide(
            &input.tick_labels,
            override_angle,
            cos_factor,
            cascade_slot_w,
            label_font_size,
            metrics,
            tick_position,
        )
    } else {
        // Run the graduated collision cascade, biased by any `label_overlap`
        // override (B5 unit 6b).
        let cascade = cascade_collision_recovery(
            &input.tick_labels,
            cascade_slot_w,
            label_font_size,
            label_gap,
            input.overrides.labels.overlap,
            metrics,
            frame,
        );
        let is_elision_strategy = matches!(cascade.strategy, CascadeStrategy::Elided { .. });
        let ticks: Vec<TickLayout> = cascade
            .labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: tick_position(i),
                label: label.clone(),
                label_angle: cascade.angle,
                elided: is_elision_strategy && label != &input.tick_labels[i],
                culled: !cascade.visible[i],
                label_font_size: cascade.font_size,
                is_major: true,
                anchor: false,
                origin: false,
            })
            .collect();
        let warning = match cascade.strategy {
            CascadeStrategy::Elided { count } => Some(AxisLabelWarning::LabelsElided { count }),
            _ => None,
        };
        (ticks, warning)
    };
    if drawn_x_label_band(&ticks, input, ctx) <= ctx.band_budget {
        return (ticks, warning);
    }

    // Over budget: the labels as laid out would need a band that leaves the
    // panel too short to draw. Cull further by stride, smallest stride first,
    // laying the survivors out the way the unculled axis was (the explicit
    // angle, or the cascade's flat / wrap / font shrink / rotate stages)
    // within the `stride` slots each now spans; the first whose drawn band is
    // within the budget wins. Fewer labels open a wider per-label budget, so
    // the survivors can sit flatter and the band shrinks.
    //
    // A budget below the least band any drawn label needs
    // (`tick + pad + min(font size, line height)`: at angle θ a label reaches
    // `sin θ·(font size + width) + cos θ·line height` past the standoff, which
    // is concave in θ, so its least is at 0° or 90°) draws no label at once.
    // When no stride gets under the budget, not even a single label, the axis
    // draws its tick marks and no labels: a chart that still draws a plot
    // keeps its data, and a label band it cannot hold is never drawn into the
    // next row or past the canvas. Drawing no label at all is reported as
    // every label culled.
    let labels = &input.tick_labels;
    let label_pad_eff = input.overrides.labels.padding.unwrap_or(2.0).max(0.0);
    let least_label_band =
        ctx.tick_size + label_pad_eff + label_font_size.min(metrics.line_height(label_font_size));
    let all_culled = || (expand_survivor_ticks(labels, &vec![false; n], &[], &tick_position), None);
    if ctx.band_budget < least_label_band {
        return all_culled();
    }
    let present = |survivor_labels: &[String], survivor_w: f64| match input.overrides.labels.angle {
        Some(angle) => {
            let cos_factor = angle.to_radians().cos().abs();
            Some(stamp_override_angle_with_elide(
                survivor_labels,
                angle,
                cos_factor,
                survivor_w,
                label_font_size,
                metrics,
                |_| 0.0,
            ))
        }
        None => {
            let p = resolve_presentation(survivor_labels, survivor_w, label_font_size, label_gap, metrics)?;
            let survivors = p
                .labels
                .into_iter()
                .map(|label| TickLayout {
                    position: 0.0,
                    label,
                    label_angle: p.angle,
                    elided: false,
                    culled: false,
                    label_font_size: p.font_size,
                    is_major: true,
                    anchor: false,
                    origin: false,
                })
                .collect();
            Some((survivors, None))
        }
    };
    let fits_budget = |(survivors, _): &(Vec<TickLayout>, Option<AxisLabelWarning>)| {
        drawn_x_label_band(survivors, input, ctx) <= ctx.band_budget
    };
    // An explicit `label_angle` is the user's own envelope (every stride
    // keeps counting from the first label, as it always has); otherwise
    // count from the axis's own nicest tick, same as the collision cascade.
    let origin = if input.overrides.labels.angle.is_some() { 0 } else { input.tick_origin.unwrap_or(0) };
    match cull_by_stride(labels, cascade_slot_w, 2..=n, &input.tick_anchors, origin, present, fits_budget) {
        Some((_, visible, (survivors, warning))) => {
            (expand_survivor_ticks(labels, &visible, &survivors, &tick_position), warning)
        }
        None => all_culled(),
    }
}

/// One `TickLayout` per tick label, placed at `tick_position`: each visible
/// tick takes the next laid-out survivor in order; every other tick is a
/// culled, flat copy of its own label.
fn expand_survivor_ticks(
    labels: &[String],
    visible: &[bool],
    survivors: &[TickLayout],
    tick_position: &dyn Fn(usize) -> f64,
) -> Vec<TickLayout> {
    let mut survivors = survivors.iter();
    labels
        .iter()
        .enumerate()
        .map(|(i, label)| match visible[i].then(|| survivors.next()).flatten() {
            Some(s) => TickLayout {
                position: tick_position(i),
                ..s.clone()
            },
            None => TickLayout {
                position: tick_position(i),
                label: label.clone(),
                label_angle: 0.0,
                elided: false,
                culled: true,
                label_font_size: None,
                is_major: true,
                anchor: false,
                origin: false,
            },
        })
        .collect()
}

/// Build the AxisLayout for the x-axis (Bottom orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.w` (spec SS14.3 step 7a).
/// Collision policy: graduated cascade (wrap -> shrink -> rotate -> cull -> elide).
///
/// Takes the same [`LabelBandCtx`] as [`estimate_x_label_band`], which reserved
/// the gutter this lays out into — see that type's doc for why the two must not
/// be handed the four measurement inputs separately.
pub fn layout_x_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    title_font_size: f64,
    axis_title_padding: f64,
    ctx: &LabelBandCtx<'_>,
) -> (AxisLayout, Option<AxisLabelWarning>) {
    let LabelBandCtx {
        label_font_size,
        metrics,
        ..
    } = *ctx;
    input.debug_assert_placement_invariants();
    let (mut ticks, warning) = lay_out_x_tick_labels(input, panel_area.x, panel_area.w, ctx);

    // Orient: Bottom (default) places the axis at the panel's bottom edge; Top
    // at the top edge. Any other orient is rejected upstream (`prepare.rs`
    // validates x→{top,bottom}); default to Bottom defensively.
    let on_top = matches!(input.orient, AxisOrient::Top);
    let axis_y = if on_top {
        panel_area.y
    } else {
        panel_area.y + panel_area.h
    };
    let axis_line = Rect {
        x: panel_area.x,
        y: axis_y,
        w: panel_area.w,
        h: 1.0,
    };

    let effective_title_font_size = input.overrides.title.font_size.unwrap_or(title_font_size);
    let effective_title_padding = input.overrides.title.padding.unwrap_or(axis_title_padding);

    // Resolved tick angle: all non-culled ticks share `label_angle` (the cascade
    // and the override path both apply a single angle). 0.0 means flat. Use any
    // non-culled tick; default to flat when every tick is culled or absent.
    let resolved_angle = ticks
        .iter()
        .find(|t| !t.culled)
        .map(|t| t.label_angle)
        .unwrap_or(0.0);

    let title = input.title.get().as_ref().map(|text| {
        let title_h = metrics.line_height(effective_title_font_size);
        // The vertical drop from the axis line to the labels' lowest point:
        // the band `estimate_x_label_band` reserves, from the same owners
        // (`x_label_band_without_ticks` for an axis with no tick labels,
        // `drawn_x_label_band` otherwise), so the title sits below every
        // drawn label line, flat, wrapped or rotated, below the tick marks
        // alone when every label is culled, and where the reserved band ends
        // when there are no tick labels. The one exception is #94's
        // labels-off axis (`fm.Axis(labels=False)`): it reserves no label
        // band, but its title keeps one label line's offset when flat and
        // the extent of its undrawn labels when rotated.
        let label_extent = if !input.show_labels() {
            if resolved_angle == 0.0 {
                metrics.line_height(label_font_size)
            } else {
                drawn_x_label_band(&ticks, input, ctx)
            }
        } else {
            x_label_band_without_ticks(input, ctx).unwrap_or_else(|| drawn_x_label_band(&ticks, input, ctx))
        };
        // Title sits beyond the tick labels, on the same side as the axis. For a
        // Top axis the band extends upward (subtract); for Bottom, downward (add).
        let band = label_extent + effective_title_padding + title_h / 2.0;
        let anchor_y = if on_top { axis_y - band } else { axis_y + band };
        // `title_orient` overrides the title rotation. The default for a
        // horizontal (top/bottom) axis is a flat title (`0`); a vertical
        // `title_orient` (left/right) rotates it (e.g. a vertical caption beside a
        // bottom axis).
        let angle = match input.overrides.title.orient {
            Some(AxisOrient::Left) => -90.0,
            Some(AxisOrient::Right) => 90.0,
            _ => 0.0,
        };
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: panel_area.x + panel_area.w / 2.0,
            anchor_y,
            angle,
        }
    });

    // Minors use the SAME base range + padding inset as the major projection
    // (`(x, x+w)`), so a minor at domain `v` coincides with the major
    // projection of `v`.
    let minor_ticks = build_minor_ticks(input, (panel_area.x, panel_area.x + panel_area.w));

    // stamp `anchor` once, after whichever
    // presentation path above built `ticks`, instead of threading the flag
    // through each of them.
    stamp_tick_anchors(&mut ticks, &input.tick_anchors, input.tick_origin);
    // 385: single construction site for the 22-field AxisLayout (shared with
    // layout_y_axis via `AxisLayout::from_input`).
    let layout = AxisLayout::from_input(input, panel_index, axis_line, ticks, minor_ticks, title);
    (layout, warning)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── `anchored_visibility`: a `Log` axis keeps its decades through a cull ──

    /// Empty anchors reduce to plain `stride_visibility`, for every stride —
    /// every non-`Log` axis culls byte-identically to before `anchored_visibility`
    /// existed.
    #[test]
    fn anchored_visibility_empty_anchors_is_the_stride_identity() {
        for stride in [1, 2, 3, 5, 9] {
            assert_eq!(
                anchored_visibility(7, stride, &[], 0),
                stride_visibility(7, stride, 0),
                "stride {stride}"
            );
        }
    }

    /// Every tick an anchor (a wide-domain ladder of pure powers) also
    /// reduces to plain `stride_visibility`.
    #[test]
    fn anchored_visibility_all_anchors_is_the_stride_identity() {
        let anchors = [true; 7];
        for stride in [1, 2, 3, 5, 9] {
            assert_eq!(
                anchored_visibility(7, stride, &anchors, 0),
                stride_visibility(7, stride, 0),
                "stride {stride}"
            );
        }
    }

    /// The `{1, 2, 5}` ladder over `[1, 100]`: `[1, 2, 5, 10, 20, 50, 100]`,
    /// decades (powers of 10) at indices 0, 3, 6 — `p = 3`. At every tested
    /// stride, every surviving decade is kept, and `1` (index 0) always
    /// survives.
    #[test]
    fn anchored_visibility_1_2_5_ladder_keeps_decades() {
        let anchors = [true, false, false, true, false, false, true];
        let n = anchors.len();
        // stride 2 < p(3): every anchor, plus in-stride non-anchors whose
        // gap to the next anchor is still >= stride.
        assert_eq!(
            anchored_visibility(n, 2, &anchors, 0),
            vec![true, false, false, true, false, false, true]
        );
        // stride 3 == p: every anchor (ceil(3/3) = every 1st).
        assert_eq!(
            anchored_visibility(n, 3, &anchors, 0),
            vec![true, false, false, true, false, false, true]
        );
        // stride 5 > p: ceil(5/3) = 2nd anchor kept — indices 0 and 6 (10 is
        // the 2nd of 3 anchors, dropped; 100 is the 3rd, kept).
        assert_eq!(
            anchored_visibility(n, 5, &anchors, 0),
            vec![true, false, false, false, false, false, true]
        );
        // stride 9 > p: ceil(9/3) = 3rd anchor kept — only index 0.
        assert_eq!(
            anchored_visibility(n, 9, &anchors, 0),
            vec![true, false, false, false, false, false, false]
        );
    }

    /// Decades at indices 0, 9, 19 of a 20-entry `{1..9}` ladder (`p = 9`
    /// and `10`, min `9`) — the DENSEST ladder, so the widest range of
    /// strides to probe the `stride < p` non-anchor-offset rule against.
    #[test]
    fn anchored_visibility_dense_ladder_keeps_decades_at_every_stride() {
        let n = 20;
        let mut anchors = vec![false; n];
        anchors[0] = true;
        anchors[9] = true;
        anchors[19] = true;
        for stride in [1, 2, 3, 5, 9] {
            let mask = anchored_visibility(n, stride, &anchors, 0);
            assert!(mask[0] && mask[9] && mask[19], "stride {stride}: decades dropped: {mask:?}");
        }
    }

    // ── `stride_visibility`/`anchored_visibility`: origin-phased thinning ──

    /// `stride_visibility` counts from `origin` in both directions, not
    /// always from index 0: at `n=14, origin=7`, stride 2 keeps every OTHER
    /// index from 7 (`1, 3, 5, 7, 9, 11, 13`), and stride 3 keeps every
    /// THIRD (`1, 4, 7, 10, 13`) — `origin` itself need not divide evenly
    /// into the stride from 0, it is simply where counting starts.
    #[test]
    fn stride_visibility_counts_from_the_origin_in_both_directions() {
        let kept = |mask: &[bool]| mask.iter().enumerate().filter(|(_, &v)| v).map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(kept(&stride_visibility(14, 2, 7)), vec![1, 3, 5, 7, 9, 11, 13]);
        assert_eq!(kept(&stride_visibility(14, 3, 7)), vec![1, 4, 7, 10, 13]);
    }

    /// `origin: 0` starts the count from the first index.
    #[test]
    fn stride_visibility_origin_zero_starts_from_the_first_index() {
        let kept = |mask: &[bool]| mask.iter().enumerate().filter(|(_, &v)| v).map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(kept(&stride_visibility(9, 2, 0)), vec![0, 2, 4, 6, 8]);
        assert_eq!(kept(&stride_visibility(9, 3, 0)), vec![0, 3, 6]);
    }

    /// `LabelOverlap::Parity` and the y overlap paths always pass `origin:
    /// 0` (Vega-Lite parity is index parity, not the axis's nicest tick),
    /// so an off-phase zero (zero not at an even index) does not shift
    /// which labels parity keeps.
    #[test]
    fn parity_stays_index_phased_regardless_of_origin() {
        let labels: Vec<String> = (0..6).map(|i| i.to_string()).collect();
        let m = mock(10.0);
        // The off-phase-zero axis: zero sits at index 1, not an even one,
        // but Parity must still keep indices 0, 2, 4 — the SAME mask
        // `anchored_visibility(.., 2, &[], 0)` gives regardless.
        let result = cascade_collision_recovery(
            &labels,
            10.0,
            11.0,
            legacy_gap(8),
            Some(LabelOverlap::Parity),
            &m,
            TickCullFrame {
                anchors: &[],
                origin: 1, // a non-zero origin, which Parity must ignore
                continuous: true,
            },
        );
        assert_eq!(result.visible, vec![true, false, true, false, true, false]);
    }

    /// `set_ticks` is the one place that replaces `tick_labels`,
    /// `tick_anchors` and `tick_origin` together. It degrades a length
    /// mismatch in `anchors` to empty, and an out-of-range `origin` to
    /// `None`, rather than panicking — the same fallback
    /// `anchored_visibility`/the collision cascade use for "no anchors"/
    /// "no origin" (counting from index 0). A matching length or a valid
    /// index is carried through untouched.
    #[test]
    fn axis_input_set_ticks_degrades_a_length_mismatch_to_empty_anchors() {
        let mut input = axis_input(AxisOrient::Bottom, None, vec!["a".into(), "b".into()], None);
        input.set_ticks(
            vec!["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
            vec![true, false],
            Some(10),
        );
        assert_eq!(input.tick_labels.len(), 5);
        assert_eq!(input.tick_anchors, Vec::<bool>::new(), "a mismatched length degrades to empty, not a panic");
        assert_eq!(input.tick_origin, None, "an out-of-range origin degrades to None, not a panic");

        input.set_ticks(vec!["1".into(), "2".into(), "3".into()], vec![true, false, true], Some(1));
        assert_eq!(input.tick_anchors, vec![true, false, true], "a matching length is carried through");
        assert_eq!(input.tick_origin, Some(1), "a valid index is carried through");
    }

    /// [`estimate_x_label_band`] for a scale-less (uniform-slot) x axis with
    /// `labels`, an optional explicit `angle` and `label_padding`, and its
    /// labels shown or not, across a panel `slot_w` wide per label: the
    /// per-tick budget the cascade judges is exactly `slot_w`.
    fn estimate_band(
        labels: &[String],
        angle: Option<f64>,
        slot_w: f64,
        label_padding: Option<f64>,
        show_labels: bool,
        ctx: &LabelBandCtx<'_>,
    ) -> f64 {
        let mut input = axis_input(AxisOrient::Bottom, None, labels.to_vec(), angle);
        input.overrides.labels.padding = label_padding;
        input.overrides.labels.show = Some(show_labels);
        estimate_x_label_band(&input, slot_w * labels.len().max(1) as f64, ctx)
    }

    use crate::scale::discrete::DiscreteLayout;

    /// The pre-`#146` `cull_threshold: u32` argument, as the policy it maps
    /// to. Every call site reading `legacy_gap(N)` asserts the legacy
    /// `Theme(cull_threshold=N)` behavior, which the mode/magnitude split
    /// leaves unchanged.
    fn legacy_gap(cull_threshold: u32) -> LabelGapPolicy {
        LabelGapPolicy::from_cull_threshold(cull_threshold)
    }

    /// The frame most cascade unit tests build against: no scale projecting
    /// the axis, no anchors, no stride phase.
    fn scale_less_frame() -> TickCullFrame<'static> {
        TickCullFrame {
            anchors: &[],
            origin: 0,
            continuous: false,
        }
    }

    /// Test helper: `AxisInput::new` plus the theme-default fold
    /// `AxesInput::apply_show_defaults` performs in production, so per-axis
    /// layout tests that don't care about domain/grid visibility (the
    /// overwhelming majority — the theme's own `true` default is what every
    /// unconfigured axis renders) don't have to know about the
    /// `show_domain()`/`show_grid()` precondition those accessors now assert
    /// in debug builds. A test that DOES care overwrites `input.overrides
    /// .domain.show`/`.grid.show` afterward, same as before.
    fn axis_input(
        orient: AxisOrient,
        title: Option<String>,
        tick_labels: Vec<String>,
        label_angle: Option<f64>,
    ) -> AxisInput {
        let new = AxisInput::new;
        let mut input = new(orient, title, tick_labels, label_angle);
        input.overrides.domain.show = Some(true);
        input.overrides.grid.show = Some(true);
        input
    }

    // ── S3 hardening (batch B design review): apply_show_defaults precondition ──

    /// A default-constructed `AxisInput` (via the real `AxisInput::new`, not
    /// the `axis_input()` test helper above, which resolves the defaults
    /// itself) has never had `AxesInput::apply_show_defaults` run against it,
    /// so `overrides.domain.show` is still `None`. Reading `show_domain()` in
    /// that state must trip the debug-only precondition assert rather than
    /// silently returning the non-conservative `true` fallback.
    #[test]
    #[should_panic(expected = "AxesInput::apply_show_defaults")]
    #[cfg(debug_assertions)]
    fn axis_input_show_domain_asserts_before_apply_show_defaults() {
        let input = AxisInput::new(AxisOrient::Bottom, None, vec![], None);
        let _ = input.show_domain();
    }

    /// Same precondition, `show_grid()` twin.
    #[test]
    #[should_panic(expected = "AxesInput::apply_show_defaults")]
    #[cfg(debug_assertions)]
    fn axis_input_show_grid_asserts_before_apply_show_defaults() {
        let input = AxisInput::new(AxisOrient::Bottom, None, vec![], None);
        let _ = input.show_grid();
    }

    // ── 385: AxisLayout::from_input field parity ─────────────────────────────

    /// `AxisLayout::from_input` threads every per-axis override onto the layout.
    /// Set distinct values for the 16 override threads + the geometry/show fields
    /// and assert each lands, so a dropped thread is caught.
    #[test]
    fn axis_layout_from_input_threads_all_overrides() {
        let red = Srgba::new(255u8, 0, 0, 255);
        let blue = Srgba::new(0u8, 0, 255, 255);
        let green = Srgba::new(0u8, 255, 0, 255);
        let mut input = axis_input(AxisOrient::Bottom, Some("T".into()), vec!["a".into()], None);
        input.overrides = AxisStyleOverrides {
            ticks: AxisTickOverrides {
                show: Some(false),
                ..AxisTickOverrides::default()
            },
            labels: AxisLabelOverrides {
                show: Some(false),
                padding: Some(3.0),
                color: Some(blue),
                font_size: Some(9.0),
                flush: Some(true),
                ..AxisLabelOverrides::default()
            },
            title: AxisTitleOverrides {
                font_size: Some(14.0),
                color: Some(red),
                ..AxisTitleOverrides::default()
            },
            grid: AxisGridOverrides {
                show: Some(false),
                color: Some(green),
                dash: Some(vec![2.0, 1.0]),
                width: Some(0.7),
                opacity: Some(0.5),
            },
            domain: AxisDomainOverrides {
                show: Some(false),
                color: Some(red),
                width: Some(1.3),
            },
            placement: AxisPlacementOverrides {
                translate: Some(4.0),
                zindex: Some(2),
                offset: Some(6.0),
                ..AxisPlacementOverrides::default()
            },
        };
        let axis_line = Rect {
            x: 1.0,
            y: 2.0,
            w: 3.0,
            h: 1.0,
        };
        let layout = AxisLayout::from_input(&input, 7, axis_line, vec![], vec![], None);

        assert_eq!(layout.orient, AxisOrient::Bottom);
        assert_eq!(layout.panel_index, 7);
        assert_eq!(layout.axis_line, axis_line);
        assert!(!layout.show_labels);
        assert!(!layout.show_ticks);
        assert!(!layout.show_domain);
        assert!(!layout.show_grid);
        assert_eq!(layout.title_font_size, Some(14.0));
        assert_eq!(layout.title_color_rgba, Some([255, 0, 0, 255]));
        assert_eq!(layout.label_padding, Some(3.0));
        assert_eq!(layout.label_color_rgba, Some([0, 0, 255, 255]));
        assert_eq!(layout.label_font_size, Some(9.0));
        assert_eq!(layout.grid_color_rgba, Some([0, 255, 0, 255]));
        assert_eq!(layout.grid_dash, Some(vec![2.0, 1.0]));
        assert_eq!(layout.grid_width, Some(0.7));
        assert_eq!(layout.domain_color_rgba, Some([255, 0, 0, 255]));
        assert_eq!(layout.domain_width, Some(1.3));
        assert_eq!(layout.grid_opacity, Some(0.5));
        assert_eq!(layout.translate, Some(4.0));
        assert_eq!(layout.zindex, Some(2));
        assert_eq!(layout.offset, Some(6.0));
        assert_eq!(layout.label_flush, Some(true));
    }

    /// 860: the channel dimension is derived from the orient (Top/Bottom → X,
    /// Left/Right → Y) and each dimension's default edge is single-sourced.
    #[test]
    fn axis_orient_dimension_and_default_edge() {
        assert_eq!(AxisOrient::Top.dimension(), AxisDimension::X);
        assert_eq!(AxisOrient::Bottom.dimension(), AxisDimension::X);
        assert_eq!(AxisOrient::Left.dimension(), AxisDimension::Y);
        assert_eq!(AxisOrient::Right.dimension(), AxisDimension::Y);
        assert_eq!(AxisDimension::X.default_orient(), AxisOrient::Bottom);
        assert_eq!(AxisDimension::Y.default_orient(), AxisOrient::Left);
    }

    /// 860: `resolve_orient` defaults to the dimension edge and honors an override.
    #[test]
    fn axis_resolve_orient_defaults_and_override() {
        // x axis, no override → Bottom.
        let mut x = axis_input(AxisOrient::Bottom, None, vec![], None);
        x.resolve_orient();
        assert_eq!(x.orient, AxisOrient::Bottom);
        // y axis, no override → Left.
        let mut y = axis_input(AxisOrient::Left, None, vec![], None);
        y.resolve_orient();
        assert_eq!(y.orient, AxisOrient::Left);
        // x axis with explicit Top override wins.
        let mut xt = axis_input(AxisOrient::Bottom, None, vec![], None);
        xt.overrides.placement.orient = Some(AxisOrient::Top);
        xt.resolve_orient();
        assert_eq!(xt.orient, AxisOrient::Top);
    }

    #[test]
    fn axis_layout_round_trip() {
        let a = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect {
                x: 50.0,
                y: 350.0,
                w: 500.0,
                h: 1.0,
            },
            ticks: vec![TickLayout {
                position: 100.0,
                label: "0".into(),
                label_angle: 0.0,
                elided: false,
                culled: false,
                label_font_size: None,
                is_major: true,
                anchor: false,
                origin: false,
            }],
            minor_ticks: vec![],
            title: Some(AxisTitleLayout {
                text: "Price".into(),
                anchor_x: 300.0,
                anchor_y: 380.0,
                angle: 0.0,
            }),
            tick_size: None,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let json = serde_json::to_string(&a).unwrap();
        let parsed: AxisLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, a);
    }

    #[test]
    fn axis_layout_serde_lowercases_orient() {
        let a = AxisLayout {
            orient: AxisOrient::Left,
            panel_index: 0,
            axis_line: Rect::ZERO,
            ticks: vec![],
            minor_ticks: vec![],
            title: None,
            tick_size: None,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains(r#""orient":"left""#));
        assert!(!json.contains("title"));
    }

    use crate::layout::text_metrics::{fixed_width, measure_multiline_width, MockMetrics};

    fn mock(per_char_px: f64) -> MockMetrics<impl Fn(&str, f64) -> f64> {
        MockMetrics {
            measure: fixed_width(per_char_px),
            line_h_factor: 1.2,
        }
    }

    // ── 395: x-label cascade prediction (flat / wrap / rotate) ───────────────
    // The predictor `estimate_x_label_band` walks the same cascade order the real
    // collision recovery uses; these guard each branch's reserved band so a
    // cascade-policy drift between the predictor and the renderer is caught. (The
    // three encoding sites are NOT byte-safely unifiable — see the task report —
    // so these lock the predictor's three outcomes in place instead.)

    /// S0 flat: a label that fits the estimated slot reserves a single line band
    /// plus the tick_size + label_pad_eff standoff (#97, spec §4.1).
    #[test]
    fn estimate_x_band_flat_when_label_fits() {
        let m = mock(10.0); // 10 px/char
        let labels = vec!["ab".into(), "cd".into()]; // 20 px each
                                                     // Generous slot: 100 px → threshold 90 px ≥ 20 px → flat.
        let band = estimate_band(
            &labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        );
        // Flat band == tick_size(4.0) + label_pad_eff(2.0) + line_height(11) = 19.2.
        let expected = 4.0 + 2.0 + m.line_height(11.0);
        assert!(
            (band - expected).abs() < 1e-9,
            "band={band}, expected={expected}"
        );
    }

    /// S1 wrap: an underscore-splittable label that won't fit flat but whose
    /// segments fit wraps to N lines → band == tick_size + label_pad_eff +
    /// max_lines * line_height (#97, spec §4.1).
    #[test]
    fn estimate_x_band_wraps_underscore_label() {
        let m = mock(10.0);
        // "aa_bb_cc": flat width 80; segments "aa"/"bb"/"cc" are 20 px each.
        let labels = vec!["aa_bb_cc".into()];
        // Slot 50 → threshold 45: flat (80) fails, segments (20) fit → 3 lines.
        let band = estimate_band(
            &labels,
            None,
            50.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        );
        let expected = 4.0 + 2.0 + 3.0 * m.line_height(11.0);
        assert!(
            (band - expected).abs() < 1e-9,
            "band={band}, expected={expected}"
        );
    }

    /// S2/S3 rotate: a single long unsplittable label that cannot fit flat or wrap
    /// falls to the rotate branch, reserving the full rotated extent (> flat band).
    #[test]
    fn estimate_x_band_rotates_long_unsplittable_label() {
        let m = mock(10.0);
        // 12-char label, no break points → 120 px, cannot wrap.
        let labels = vec!["abcdefghijkl".into()];
        let slot = 30.0; // threshold 27 < 120 → not flat, no wrap → rotate/vertical.
        let band = estimate_band(
            &labels,
            None,
            slot,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        );
        // Must exceed the flat single-line band (a rotated/vertical reservation).
        assert!(
            band > m.line_height(11.0),
            "rotated band must exceed flat: {band}"
        );
    }

    /// An explicit `label_angle` override bypasses the cascade and reserves the
    /// rotated extent for that exact angle.
    #[test]
    fn estimate_x_band_honors_explicit_angle() {
        let m = mock(10.0);
        let labels = vec!["abc".into()];
        // label_padding=None → label_pad_eff defaults to 2.0; tick_size=4.0.
        let band = estimate_band(
            &labels,
            Some(-45.0),
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        );
        let expected = rotated_x_label_extent(-45.0, 30.0, 11.0, m.line_height(11.0), 4.0, 2.0);
        assert!(
            (band - expected).abs() < 1e-9,
            "band={band}, expected={expected}"
        );
    }

    #[test]
    fn y_axis_label_band_uses_longest_label() {
        let input = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "100".into(), "10000".into()],
            None,
        );
        let m = mock(10.0);
        let band = compute_y_label_band_width(&input, 11.0, &m, 4.0, true);
        // #97 (spec §4.1): the band now includes the tick_size + label_pad_eff
        // standoff the renderer actually draws (tick_size=4.0 + label_pad
        // default 2.0 + widest label "10000" = 50.0px).
        assert_eq!(band, 4.0 + 2.0 + 50.0);
    }

    #[test]
    fn y_axis_label_band_empty_labels_returns_zero() {
        let input = axis_input(AxisOrient::Left, None, vec![], None);
        let m = mock(10.0);
        assert_eq!(compute_y_label_band_width(&input, 11.0, &m, 4.0, true), 0.0);
        // Byte-identical regardless of visibility (spec §4.1: empty label
        // sets return 0.0 on y at every angle).
        assert_eq!(
            compute_y_label_band_width(&input, 11.0, &m, 4.0, false),
            0.0
        );
    }

    /// #97 spec §4.1 (amended 2026-08-27, gap 2): an empty label set with a
    /// non-zero `label_angle` override must ALSO return 0.0 — deliberately
    /// superseding the pre-#97 rotated-empty formula, which returned a
    /// nonzero `tick_size + label_pad + sin|θ|·line_h` phantom reservation
    /// for labels that don't exist. Pinned explicitly (not just implied by
    /// the θ=0 empty case above) so this corner is a recorded decision, not
    /// an accident of the early-return's placement before the angle branch.
    #[test]
    fn y_axis_label_band_empty_labels_with_rotated_override_returns_zero() {
        let input = axis_input(AxisOrient::Left, None, vec![], Some(-45.0));
        let m = mock(10.0);
        assert_eq!(compute_y_label_band_width(&input, 11.0, &m, 4.0, true), 0.0);
        assert_eq!(
            compute_y_label_band_width(&input, 11.0, &m, 4.0, false),
            0.0
        );
    }

    /// `#182` (closes `#94`), superseding this test's own pre-fix contract: a
    /// HIDDEN y axis (`visible: false` — `.axis(show=False)` on JointChart
    /// marginals / ClusterMap dendrograms, or a projected coord with no
    /// Cartesian axes at all) reserves NOTHING, at any angle. It previously
    /// pinned the pre-`#97` bare `max_label_w` here, which `#97` left standing
    /// as a knowingly-deferred phantom ("the #94 family"); the rule is now that
    /// a band exists if and only if label text is drawn in it, so 50.0 of
    /// reservation for labels the renderer never emits is exactly the defect.
    /// The angle sweep is the part `#182` widened beyond the θ=0 case `#97`
    /// gated: a hidden ROTATED axis used to reserve the full
    /// `rotated_y_label_extent`.
    #[test]
    fn y_axis_label_band_hidden_axis_reserves_nothing() {
        let input = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "100".into(), "10000".into()],
            None,
        );
        let m = mock(10.0);
        let hidden = compute_y_label_band_width(&input, 11.0, &m, 4.0, false);
        assert_eq!(
            hidden, 0.0,
            "a hidden axis draws no labels, so it reserves no band"
        );

        // No override moves it, because there is nothing to move: neither the
        // standoff term `label_padding` feeds nor the rotated extent an angle
        // override selects applies to an axis that draws no labels.
        let mut padded = input.clone();
        padded.overrides.labels.padding = Some(40.0);
        assert_eq!(
            compute_y_label_band_width(&padded, 11.0, &m, 4.0, false),
            0.0
        );
        let mut rotated = input.clone();
        rotated.overrides.labels.angle = Some(-45.0);
        assert_eq!(
            compute_y_label_band_width(&rotated, 11.0, &m, 4.0, false),
            0.0,
            "#182 widened the gate past θ=0: a hidden rotated axis reserves nothing either"
        );

        // Discriminating control: the SAME inputs, VISIBLE, all reserve a real
        // band — so the zeros above come from the drawn gate, not from a
        // degenerate fixture that would measure 0.0 either way.
        assert!(compute_y_label_band_width(&input, 11.0, &m, 4.0, true) >= 50.0);
        assert!(compute_y_label_band_width(&padded, 11.0, &m, 4.0, true) > 50.0);
        assert!(compute_y_label_band_width(&rotated, 11.0, &m, 4.0, true) > 0.0);
    }

    /// `#182` (closes `#94`), superseding this test's own pre-fix contract: the
    /// drawn gate keys off "labels actually drawn", not just "axis shown" — a
    /// SHOWN axis (`visible: true`) with `fm.Axis(labels=False)`
    /// (`AxisInput.show_labels = false`) draws no tick label text
    /// (`render/marks/axis.rs`'s `if axis.show_labels && !tick.culled` guard),
    /// so it reserves no label band at all. It previously pinned the pre-`#97`
    /// bare `max_label_w` here; that was a reservation for text the renderer
    /// never emits. Its axis TITLE gutter is untouched by this — the two are
    /// separately gated, which is what makes `fm.Axis(labels=False)` collapse
    /// the label band alone (pinned end-to-end in
    /// `tests/test_axis_gutter_gating.py`).
    #[test]
    fn y_axis_label_band_labels_off_on_shown_axis_reserves_nothing() {
        let mut input = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "100".into(), "10000".into()],
            None,
        );
        input.overrides.labels.show = Some(false);
        let m = mock(10.0);

        // `visible: true` — the axis itself is shown; only its labels are off.
        let labels_off = compute_y_label_band_width(&input, 11.0, &m, 4.0, true);
        assert_eq!(
            labels_off, 0.0,
            "a shown axis whose labels are off draws no label text, so it reserves no band"
        );

        // Nothing to move: `label_padding` feeds only the standoff of a band
        // that no longer exists.
        let mut padded = input.clone();
        padded.overrides.labels.padding = Some(40.0);
        let labels_off_padded = compute_y_label_band_width(&padded, 11.0, &m, 4.0, true);
        assert_eq!(
            labels_off_padded, labels_off,
            "labels-off axis must be insensitive to label_padding (no band to pad)"
        );

        // Contrast: the SAME input with labels back on DOES grow with
        // label_padding — proving the gate keys on `show_labels`, not some
        // unrelated code path that always returns zero.
        let mut labels_on_padded = padded.clone();
        labels_on_padded.overrides.labels.show = Some(true);
        let labels_on_padded_band =
            compute_y_label_band_width(&labels_on_padded, 11.0, &m, 4.0, true);
        assert!(
            labels_on_padded_band > labels_off_padded,
            "restoring show_labels must restore the band"
        );
    }

    // ── R2: y-axis `label_angle` (transpose of the x rotated-extent geometry) ──

    /// TRANSPOSE of `rotated_x_label_extent_hand_computed_values`: rotation
    /// SHRINKS the y band (the `cos` term carries `max_label_w`, `sin` carries
    /// `line_h` — the opposite weighting from x, where `sin` carries
    /// `max_label_w` and rotation GROWS the band).
    #[test]
    fn rotated_y_label_extent_hand_computed_values() {
        // -45°: sin=cos=√2/2≈0.70710678. With max_w=100, line_h=13.2,
        // tick_size=4, label_pad=2: 4 + 2 + 0.7071·100 + 0.7071·13.2
        let sin45 = (-45.0_f64).to_radians().sin().abs();
        let cos45 = (-45.0_f64).to_radians().cos().abs();
        let expected_45 = 4.0 + 2.0 + cos45 * 100.0 + sin45 * 13.2;
        let got_45 = rotated_y_label_extent(-45.0, 100.0, 13.2, 4.0, 2.0);
        assert!(
            (got_45 - expected_45).abs() < 1e-9,
            "-45° extent should be {expected_45}, got {got_45}",
        );

        // -90°: sin=1, cos≈0 → tick_size + label_pad + line_h (the
        // `cos·max_label_w` term vanishes — the label is fully vertical, so its
        // own width no longer projects onto the horizontal gutter at all; the
        // TRANSPOSE of x's -90° case, where it's the `sin·max_label_w` term
        // that survives and `cos·line_h` that vanishes).
        let got_90 = rotated_y_label_extent(-90.0, 100.0, 13.2, 4.0, 2.0);
        let expected_90 = 4.0 + 2.0 + 13.2;
        assert!(
            (got_90 - expected_90).abs() < 1e-6,
            "-90° extent should be ~{expected_90}, got {got_90}",
        );
    }

    /// `compute_y_label_band_width` with an explicit override angle routes
    /// through `rotated_y_label_extent` instead of the flat max-width formula.
    #[test]
    fn y_axis_label_band_width_honors_explicit_angle() {
        let input = axis_input(
            AxisOrient::Left,
            None,
            vec!["abcdefghij".into()], // 10 chars
            Some(-45.0),
        );
        let m = mock(10.0); // 10 px/char → width 100
        let band = compute_y_label_band_width(&input, 11.0, &m, 4.0, true);
        let expected = rotated_y_label_extent(-45.0, 100.0, m.line_height(11.0), 4.0, 2.0);
        assert!(
            (band - expected).abs() < 1e-9,
            "band={band}, expected={expected}"
        );
    }

    /// #97 continuity gate (spec §4.1, §9.1), replacing the old pre-R2
    /// θ=0-bit-parity gate: the y band is now ONE formula
    /// (`rotated_y_label_extent`) at every angle, so `band(θ)` must approach
    /// `band(0)` continuously as `θ→0` — no jump at exactly zero. `None` and an
    /// explicit `Some(0.0)` override both resolve to angle 0.0 and must be
    /// identical; a tiny non-zero angle must land within float epsilon of the
    /// θ=0 value (not the old, under-reserving `max_label_w`-only quantity).
    #[test]
    fn y_axis_label_band_width_continuous_at_theta_zero() {
        let labels = vec!["0".into(), "100".into(), "10000".into()];
        let m = mock(10.0);
        let no_override = axis_input(AxisOrient::Left, None, labels.clone(), None);
        let explicit_zero = axis_input(AxisOrient::Left, None, labels.clone(), Some(0.0));
        let tiny_angle = axis_input(AxisOrient::Left, None, labels, Some(1e-6));

        let band_none = compute_y_label_band_width(&no_override, 11.0, &m, 4.0, true);
        let band_zero = compute_y_label_band_width(&explicit_zero, 11.0, &m, 4.0, true);
        let band_tiny = compute_y_label_band_width(&tiny_angle, 11.0, &m, 4.0, true);

        assert_eq!(
            band_none, band_zero,
            "no override must equal an explicit 0.0 override"
        );
        assert!(
            (band_tiny - band_zero).abs() < 1e-3,
            "band at θ→0 ({band_tiny}) must approach band(0) ({band_zero}) continuously"
        );
        // tick_size(4.0) + label_pad_eff(2.0) + widest label (50.0) — the
        // renderer's actual standoff, not the retired flat-only quirk.
        assert_eq!(band_zero, 4.0 + 2.0 + 50.0);
    }

    /// `layout_y_axis` stamps the override angle onto every tick (mirrors
    /// `layout_x_axis`'s override branch); with ample vertical spacing no
    /// label collides, so nothing is elided.
    #[test]
    fn layout_y_axis_stamps_override_angle_on_every_tick() {
        let input = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            Some(-45.0),
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 400.0,
        };
        let m = mock(10.0);
        let (axis, warning) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(!t.culled, "y-axis has no cull recovery (spec §4-R2)");
            assert!(
                !t.elided,
                "short single-char labels should not need elision"
            );
        }
        assert!(warning.is_none());
    }

    /// R2 acceptance #2 / decision-record R2: rotated y labels that still
    /// collide use elide-to-fit, never cull. Mirrors
    /// `x_axis_elides_via_override_when_angle_forced`, transposed: y judges
    /// collision by the rotated label's VERTICAL projection (`sin`) against
    /// the per-tick vertical budget.
    #[test]
    fn layout_y_axis_elides_via_override_when_angle_forced() {
        let input = axis_input(
            AxisOrient::Left,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            Some(-45.0),
        );
        // 20 ticks packed into a 200px-tall panel → slot_h = 10px. Each ~7-8
        // char label is 70-80px wide; at -45° the vertical projection
        // (w * sin(45°) ≈ 49-56px) far exceeds the 10px budget.
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let (axis, warning) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(!t.culled, "y-axis has no cull recovery (spec §4-R2)");
            assert!(
                t.elided,
                "expected all 20 labels to be elided with override"
            );
            assert!(
                t.label.ends_with('\u{2026}'),
                "expected ellipsis suffix; got {:?}",
                t.label
            );
        }
        match warning {
            Some(AxisLabelWarning::LabelsElided { count }) => assert_eq!(count, 20),
            other => panic!("expected LabelsElided{{count: 20}}, got {:?}", other),
        }
    }

    /// #97 continuity gate (spec §4.1, §9.1) at the `layout_y_axis` level,
    /// replacing the old θ=0-bit-parity gate. `None` and an explicit
    /// `Some(0.0)` override take DIFFERENT code branches (the override branch
    /// vs. the flat branch), so `y_axis_label_band_width_continuous_at_theta_zero`
    /// (which only pins `compute_y_label_band_width`) does not cover this site.
    /// A dense, long-label fixture is used deliberately: at θ=0 the override
    /// branch's own collision check (`sin(0) = 0` ⇒ `w * 0 > budget` is always
    /// false) must produce the SAME "never elides" outcome as the flat
    /// branch's unconditional "never checks" — proving the equivalence holds
    /// even in a case that WOULD collide at any other angle, not just a
    /// trivially-non-colliding one. A tiny non-zero angle must land on the
    /// same "never elides" outcome too, confirming the collision judgment is
    /// continuous approaching θ=0, not just exact at it.
    #[test]
    fn layout_y_axis_continuous_at_theta_zero() {
        let labels: Vec<String> = (0..20).map(|i| format!("Label_{}", i)).collect();
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        let no_override = axis_input(AxisOrient::Left, None, labels.clone(), None);
        let explicit_zero = axis_input(AxisOrient::Left, None, labels.clone(), Some(0.0));
        let tiny_angle = axis_input(AxisOrient::Left, None, labels, Some(1e-6));

        let (axis_none, warn_none) =
            layout_y_axis(&no_override, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        let (axis_zero, warn_zero) =
            layout_y_axis(&explicit_zero, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        let (_, warn_tiny) = layout_y_axis(&tiny_angle, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);

        assert_eq!(
            axis_zero.ticks, axis_none.ticks,
            "Some(0.0) must produce a field-for-field identical tick vector to None"
        );
        assert_eq!(warn_zero, None, "θ=0 override must never elide");
        assert_eq!(warn_none, None, "no-override flat path must never elide");
        assert_eq!(
            warn_tiny, None,
            "θ→0 must continuously never elide, not jump at exactly zero"
        );
    }

    // ── #97 acceptance (spec §9.1): reservation covers render, both axes,
    // flat and rotated. The flat cases are render-integration tests: they run
    // the SAME `AxisLayout` production code builds through the actual
    // `render::marks::axis::build_axis`, and check the reserved band against
    // the real emitted `SceneNode::Text` position. The rotated cases mirror
    // the pre-existing `estimate_rotated_band_covers_true_label_extent`
    // pattern (an independently re-derived analytic extent), since a rotated
    // glyph's true screen-space bounding box is a renderer/SVG-level property
    // the scene node's pre-rotation pivot coordinates alone don't carry.

    #[test]
    fn y_band_flat_reservation_covers_rendered_label_extent() {
        let labels: Vec<String> = vec!["0".into(), "100".into(), "10000".into()];
        let m = mock(10.0);
        let tick_size = 4.0;
        let font_size = 11.0;
        let input = axis_input(AxisOrient::Left, None, labels, None);
        let panel_area = Rect {
            x: 100.0,
            y: 0.0,
            w: 300.0,
            h: 200.0,
        };
        let (axis, _) = layout_y_axis(&input, panel_area, 0, font_size, 13.0, 4.0, tick_size, &m);

        let band = compute_y_label_band_width(&input, font_size, &m, tick_size, true);

        let theme = crate::layout::ThemeInputs::default();
        let nodes = crate::render::marks::axis::build_axis(&axis, &theme, None).into_nodes();
        // Left orient anchors text at its right edge (`TextAnchor::End`), so the
        // label's leftward reach from the axis line is `(axis_line.x - node.x)
        // + measured_width`.
        let max_left_reach = nodes
            .iter()
            .filter_map(|n| match n {
                ferrum_scene::SceneNode::Text { x, content, .. } => {
                    Some((axis.axis_line.x - x) + m.measure_width(content, font_size))
                }
                _ => None,
            })
            .fold(0.0_f64, f64::max);
        assert!(
            band >= max_left_reach,
            "band ({band}) must cover the rendered label's leftward reach ({max_left_reach})"
        );
    }

    #[test]
    fn y_band_rotated_covers_true_label_extent() {
        // Transpose of `estimate_rotated_band_covers_true_label_extent`: y
        // rotation SHRINKS the horizontal footprint, so the `cos` term carries
        // `max_label_w` and `sin` carries the small descent term (the opposite
        // weighting from x).
        let labels: Vec<String> = vec!["0".into(), "100".into(), "10000".into()];
        let m = mock(10.0);
        let tick_size = 4.0;
        let font_size = 11.0;
        let max_label_w = 50.0; // "10000" = 5 chars * 10px/char
        let input = axis_input(AxisOrient::Left, None, labels, Some(-45.0));
        let band = compute_y_label_band_width(&input, font_size, &m, tick_size, true);

        let angle_rad = (-45.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();
        let descent = font_size * 0.3; // conservative estimate, well under line_h
        let label_pad = 2.0; // default
        let true_extent = tick_size + label_pad + cos_abs * max_label_w + sin_abs * descent;

        assert!(
            band >= true_extent,
            "band ({band}) must cover the true rotated label extent ({true_extent})"
        );
    }

    #[test]
    fn x_band_flat_reservation_covers_rendered_label_extent() {
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let m = mock(10.0);
        let tick_size = 4.0;
        let font_size = 11.0;
        let input = axis_input(AxisOrient::Bottom, None, labels, None);
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(font_size, legacy_gap(8), tick_size, &m),
        );

        let band = estimate_band(
            &input.tick_labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(font_size, legacy_gap(0), tick_size, &m),
        );

        let theme = crate::layout::ThemeInputs::default();
        let nodes = crate::render::marks::axis::build_axis(&axis, &theme, None).into_nodes();
        // Bottom orient's flat baseline is `y`; the vertical drop from the axis
        // line to that baseline is the render's own ground truth.
        let max_drop = nodes
            .iter()
            .filter_map(|n| match n {
                ferrum_scene::SceneNode::Text { y, .. } => Some(y - axis.axis_line.y),
                _ => None,
            })
            .fold(0.0_f64, f64::max);
        assert!(
            band >= max_drop,
            "band ({band}) must cover the rendered label's vertical drop ({max_drop}) from the axis line"
        );
    }

    #[test]
    fn y_axis_layout_uniform_tick_positions() {
        let input = axis_input(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            None,
        );
        let panel_area = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        assert_eq!(axis.orient, AxisOrient::Left);
        assert_eq!(axis.panel_index, 0);
        assert_eq!(axis.ticks.len(), 4);
        assert!((axis.ticks[0].position - (50.0 + 25.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (50.0 + 175.0)).abs() < 1e-9);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        let title = axis.title.unwrap();
        assert_eq!(title.text, "Price");
        assert!((title.angle - (-90.0)).abs() < 1e-9);
    }

    // ── categorical band placement (GH #39 phase 2, GH #67) ─────────────────

    /// The four categories every placement row below lays out.
    fn four_cat_labels() -> Vec<String> {
        vec!["a".into(), "b".into(), "c".into(), "d".into()]
    }

    /// The panel every placement row lays out against: `x = 100, w = 300`
    /// (uniform slots 137.5 / 212.5 / 287.5 / 362.5) and `y = 50, h = 200`
    /// (uniform slots 75 / 125 / 175 / 225). Both dimensions are deliberately
    /// unlike the absolute centers below, so an axis that ignored its placement
    /// could not accidentally agree.
    const PLACEMENT_PANEL: Rect = Rect {
        x: 100.0,
        y: 50.0,
        w: 300.0,
        h: 200.0,
    };

    /// An `Absolute` placement (an explicit-range ordinal axis) puts each tick
    /// on the scale's own band center, NOT on the panel-uniform slot center.
    #[test]
    fn x_axis_uses_absolute_placement_when_present() {
        let mut input = axis_input(AxisOrient::Bottom, None, four_cat_labels(), None);
        input.categorical_placement = Some(CategoricalPlacement::Absolute {
            layout: crate::scale::discrete::DiscreteLayout::UNPADDED,
            categories: 4,
            range: (40.0, 260.0),
        });
        let m = MockMetrics {
            measure: fixed_width(4.0),
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            PLACEMENT_PANEL,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![67.5, 122.5, 177.5, 232.5]);
    }

    /// A `PanelExtent` placement resolves its d3 model against THIS panel — the
    /// half of GH #67 the axis owns. Padded, the ticks land off the uniform
    /// slots (`padding_inner = padding_outer = 0.1` over 300px: denom 4.1,
    /// step 73.17…, first center 140.24… vs the uniform 137.5); unpadded, the
    /// same code path must reproduce the uniform slots, which is what keeps
    /// every default categorical golden byte-identical.
    #[test]
    fn x_axis_resolves_panel_extent_placement_against_this_panel() {
        let m = MockMetrics {
            measure: fixed_width(4.0),
            line_h_factor: 1.2,
        };
        let positions = |layout| {
            let mut input = axis_input(AxisOrient::Bottom, None, four_cat_labels(), None);
            input.categorical_placement = Some(CategoricalPlacement::PanelExtent {
                layout,
                categories: 4,
            });
            let (axis, _) = layout_x_axis(
                &input,
                PLACEMENT_PANEL,
                0,
                13.0,
                4.0,
                &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
            );
            axis.ticks.iter().map(|t| t.position).collect::<Vec<f64>>()
        };

        let step = 300.0 / 4.1;
        let padded_expected: Vec<f64> = (0..4)
            .map(|i| 100.0 + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step)
            .collect();
        assert_eq!(
            positions(DiscreteLayout::band(0.1, 0.1, 0.5)),
            padded_expected
        );

        assert_eq!(
            positions(DiscreteLayout::UNPADDED),
            vec![137.5, 212.5, 287.5, 362.5],
            "the unpadded model must still reduce to the uniform slots"
        );
    }

    /// Without a placement (the scale-less `compute_layout` binding, which is
    /// handed bare label strings), the x-axis keeps the uniform-slot formula.
    #[test]
    fn x_axis_falls_back_to_uniform_center_without_a_placement() {
        let input = axis_input(AxisOrient::Bottom, None, four_cat_labels(), None);
        assert!(input.categorical_placement.is_none());
        let m = MockMetrics {
            measure: fixed_width(4.0),
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            PLACEMENT_PANEL,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![137.5, 212.5, 287.5, 362.5]);
    }

    /// The y-axis honors an `Absolute` placement too — ordinal y is NOT
    /// reversed, so band centers map to labels in domain order (top → bottom).
    #[test]
    fn y_axis_uses_absolute_placement_when_present() {
        let mut input = axis_input(AxisOrient::Left, None, four_cat_labels(), None);
        input.categorical_placement = Some(CategoricalPlacement::Absolute {
            layout: crate::scale::discrete::DiscreteLayout::UNPADDED,
            categories: 4,
            range: (40.0, 260.0),
        });
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, PLACEMENT_PANEL, 0, 11.0, 13.0, 4.0, 4.0, &m);
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![67.5, 122.5, 177.5, 232.5]);
    }

    /// The y transpose of the panel-extent row: the model resolves against the
    /// panel's vertical interval, top → bottom, and reduces to the uniform
    /// slots when unpadded.
    #[test]
    fn y_axis_resolves_panel_extent_placement_against_this_panel() {
        let m = mock(10.0);
        let positions = |layout| {
            let mut input = axis_input(AxisOrient::Left, None, four_cat_labels(), None);
            input.categorical_placement = Some(CategoricalPlacement::PanelExtent {
                layout,
                categories: 4,
            });
            let (axis, _) = layout_y_axis(&input, PLACEMENT_PANEL, 0, 11.0, 13.0, 4.0, 4.0, &m);
            axis.ticks.iter().map(|t| t.position).collect::<Vec<f64>>()
        };

        let step = 200.0 / 4.1;
        let padded_expected: Vec<f64> = (0..4)
            .map(|i| 50.0 + 0.1 * step + step * 0.9 / 2.0 + i as f64 * step)
            .collect();
        assert_eq!(
            positions(DiscreteLayout::band(0.1, 0.1, 0.5)),
            padded_expected
        );

        assert_eq!(
            positions(DiscreteLayout::UNPADDED),
            vec![75.0, 125.0, 175.0, 225.0],
            "the unpadded model must still reduce to the uniform slots"
        );
    }

    /// The collision cascade judges label fit against the PLACEMENT's band
    /// step, not the panel's uniform slot — so a scale whose categories are
    /// spaced wider than `panel/n` gets a looser budget, and one spaced tighter
    /// gets a tighter one.
    ///
    /// This is the behavior change F-L04-03 carried beyond tick placement, and
    /// it is percent-scale on a non-default scale (batch-C T7 review): over four
    /// categories on this 300px panel the budget is 75px with no placement,
    /// 73.17px under `padding = 0.1` (denom 4.1) and 100px under a
    /// `PointScale(padding = 0)` (denom 3) — `w/4` → `w/4.1` → `w/3`. Seventeen
    /// characters at 4px each measure 68px, which sits between the point
    /// scale's threshold (`100·0.9 − 8 = 82`) and the other two (`67.5 − 8 =
    /// 59.5` and `65.85 − 8 = 57.85`), so the same labels on the same panel stay
    /// flat under one placement and rotate under the others. The labels carry no
    /// spaces or hyphens, so the wrap stage cannot absorb the difference.
    #[test]
    fn the_cascade_budget_follows_the_placement_step_not_the_panel_slot() {
        let labels: Vec<String> = ["a", "b", "c", "d"].iter().map(|c| c.repeat(17)).collect();
        let m = MockMetrics {
            measure: fixed_width(4.0),
            line_h_factor: 1.2,
        };
        let angle = |placement: Option<CategoricalPlacement>| {
            let mut input = axis_input(AxisOrient::Bottom, None, labels.clone(), None);
            input.categorical_placement = placement;
            let (axis, _) = layout_x_axis(
                &input,
                PLACEMENT_PANEL,
                0,
                13.0,
                4.0,
                &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
            );
            axis.ticks[0].label_angle
        };
        let panel_extent = |layout| {
            Some(CategoricalPlacement::PanelExtent {
                layout,
                categories: 4,
            })
        };

        assert_eq!(
            angle(panel_extent(DiscreteLayout::point(0.0, 0.5))),
            0.0,
            "a point scale spaces categories 100px apart, so 68px labels fit flat"
        );
        assert_ne!(
            angle(None),
            0.0,
            "the 75px uniform slot cannot fit the same labels — the budget, not the labels, is what changed"
        );
        assert_ne!(
            angle(panel_extent(DiscreteLayout::band(0.1, 0.1, 0.5))),
            0.0,
            "padding tightens the step to 73.17px, tighter still than the uniform slot"
        );
    }

    /// `paired_band_centers` is the release-side half of the length-pairing
    /// invariant `debug_assert_placement_invariants` asserts in debug builds;
    /// nothing in `layout_*_axis` can reach its rejecting branch (the debug
    /// assertion fires first), so this pins it directly (batch-C T7 review,
    /// Bug Risk 3).
    #[test]
    fn paired_band_centers_accepts_matched_length_and_rejects_mismatched() {
        assert_eq!(
            paired_band_centers(Some(&[1.0, 2.0]), 2),
            Some(&[1.0, 2.0][..])
        );
        assert_eq!(paired_band_centers(Some(&[1.0, 2.0]), 3), None);
        // Mutation-tester (batch-C close remediation, fix 1): the carrier
        // LONGER than `tick_count` must also be refused, not just the
        // shorter-carrier case above — a `==` guard relaxed to `>=` would
        // still pass both assertions above (both carriers there are shorter
        // than or equal to `tick_count`) but would let a longer carrier
        // (e.g. a `tick_values` override with fewer labels than categories)
        // pair label *i* onto category *i*'s center, which is not that
        // label's category — exactly what this function's own doc forbids.
        assert_eq!(paired_band_centers(Some(&[1.0, 2.0, 3.0]), 2), None);
    }

    /// Without a placement, the y-axis keeps uniform-slot placement.
    #[test]
    fn y_axis_falls_back_to_uniform_center_without_a_placement() {
        let input = axis_input(AxisOrient::Left, None, four_cat_labels(), None);
        assert!(input.categorical_placement.is_none());
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, PLACEMENT_PANEL, 0, 11.0, 13.0, 4.0, 4.0, &m);
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![75.0, 125.0, 175.0, 225.0]);
    }

    // ── B5 unit 2: orient-aware axis_line + title_orient ────────────────────

    #[test]
    fn draws_above_marks_zindex_threshold() {
        let mut a = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect::ZERO,
            ticks: vec![],
            minor_ticks: vec![],
            title: None,
            tick_size: None,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            title_font_size: None,
            title_color_rgba: None,
            label_padding: None,
            label_color_rgba: None,
            label_font_size: None,
            grid_color_rgba: None,
            grid_dash: None,
            grid_width: None,
            domain_color_rgba: None,
            domain_width: None,
            grid_opacity: None,
            translate: None,
            zindex: None,
            offset: None,
            label_flush: None,
        };
        assert!(!a.draws_above_marks(), "None zindex draws below (default)");
        a.zindex = Some(0);
        assert!(!a.draws_above_marks(), "zindex 0 draws below");
        a.zindex = Some(-2);
        assert!(!a.draws_above_marks(), "negative zindex draws below");
        a.zindex = Some(1);
        assert!(a.draws_above_marks(), "zindex >= 1 draws above");
        a.zindex = Some(99);
        assert!(a.draws_above_marks(), "any zindex >= 1 draws above");
    }

    #[test]
    fn y_axis_right_orient_places_line_on_right_edge() {
        let mut input = axis_input(
            AxisOrient::Right,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into()],
            None,
        );
        input.orient = AxisOrient::Right;
        let panel_area = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        assert_eq!(axis.orient, AxisOrient::Right);
        // Axis line on the right edge (x + w), not the left.
        assert!((axis.axis_line.x - (100.0 + 300.0)).abs() < 1e-9);
        // Default title rotation on a right axis is +90 (reads bottom-to-top).
        let title = axis.title.unwrap();
        assert!((title.angle - 90.0).abs() < 1e-9);
        // Title is to the right of the axis line.
        assert!(title.anchor_x > axis.axis_line.x);
    }

    #[test]
    fn y_axis_horizontal_title_orient_renders_flat() {
        let mut input = axis_input(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into()],
            None,
        );
        input.overrides.title.orient = Some(AxisOrient::Top);
        let panel_area = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        // A horizontal title_orient renders the y-axis title flat (angle 0).
        assert_eq!(axis.title.unwrap().angle, 0.0);
    }

    #[test]
    fn x_axis_top_orient_places_line_on_top_edge() {
        let mut input = axis_input(
            AxisOrient::Top,
            Some("Feature".into()),
            vec!["a".into(), "b".into(), "c".into()],
            None,
        );
        input.orient = AxisOrient::Top;
        let panel_area = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(30), 4.0, &m),
        );
        assert_eq!(axis.orient, AxisOrient::Top);
        // Axis line at the panel top, not the bottom.
        assert!((axis.axis_line.y - 50.0).abs() < 1e-9);
        // Title is above the axis line (smaller y).
        let title = axis.title.unwrap();
        assert!(title.anchor_y < axis.axis_line.y);
    }

    #[test]
    fn x_axis_default_orient_unchanged() {
        // Regression guard: a Bottom x-axis still places the line at the panel
        // bottom and a flat title below it (byte-identity sentinel).
        let input = axis_input(
            AxisOrient::Bottom,
            Some("Feature".into()),
            vec!["a".into(), "b".into()],
            None,
        );
        let panel_area = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(30), 4.0, &m),
        );
        assert_eq!(axis.orient, AxisOrient::Bottom);
        assert!((axis.axis_line.y - (50.0 + 200.0)).abs() < 1e-9);
        let title = axis.title.unwrap();
        assert_eq!(title.angle, 0.0);
        assert!(title.anchor_y > axis.axis_line.y);
    }

    #[test]
    fn x_axis_no_collision_keeps_labels_flat() {
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 50.0,
            line_h_factor: 1.2,
        };
        let (axis, warning) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert_eq!(axis.ticks.len(), 4);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        assert!(warning.is_none());
    }

    #[test]
    fn x_axis_uniform_tick_positions_along_axis() {
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect {
            x: 100.0,
            y: 50.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 10.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert!((axis.ticks[0].position - (100.0 + 50.0)).abs() < 1e-9);
        assert!((axis.ticks[1].position - (100.0 + 150.0)).abs() < 1e-9);
        assert!((axis.ticks[2].position - (100.0 + 250.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (100.0 + 350.0)).abs() < 1e-9);
    }

    #[test]
    fn x_axis_collision_triggers_graduated_rotation() {
        // 8 labels of 80px each in 400px panel. slot_w=50, threshold=45.
        // No break points (L0..L7), so wrapping/shrink fail.
        // Cascade tries rotation using the real footprint (cos*w + sin*line_h,
        // line_h = 11*1.2 = 13.2; spec §4.6 / F-L07-04 fix):
        //   -30: cos(30)*80 + sin(30)*13.2 = 69.28 + 6.6 = 75.88 > 50 -> fail.
        //   -45: cos(45)*80 + sin(45)*13.2 = 56.57 + 9.34 = 65.91 > 50 -> fail.
        //   -60: cos(60)*80 + sin(60)*13.2 = 40.00 + 11.43 = 51.43 > 50 -> fail
        //        (the pre-fix bare-cos check alone, 40<=50, used to pass here).
        //   -90: cos(90)*80 + sin(90)*13.2 ~= 0 + 13.2 = 13.2 <= 50 -> pass at -90!
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
            assert!(!t.elided);
        }
    }

    #[test]
    fn x_axis_rotates_at_custom_angle_override() {
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            Some(-90.0),
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
        }
    }

    #[test]
    fn x_axis_rotation_only_no_elision_when_rotated_fits() {
        // 6 labels of 95px each in 600px panel. slot_w=100, threshold=90.
        // 95>90 -> collision. No break points -> S1/S2 fail.
        // S3: -30 -> cos(30)*95=82.3<=100 -> passes at -30.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 95.0,
            line_h_factor: 1.2,
        };
        let (axis, warning) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -30.0);
            assert!(!t.elided, "rotated projection should fit; no elision");
        }
        assert!(warning.is_none());
    }

    #[test]
    fn x_axis_elides_via_override_when_angle_forced() {
        // With label_angle_override, bypass cascade. 20 labels of 7+ chars each
        // in 200px panel. Override at -45, some labels will need elision.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            Some(-45.0),
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let (axis, warning) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(
                t.elided,
                "expected all 20 labels to be elided with override"
            );
            assert!(
                t.label.ends_with('\u{2026}'),
                "expected ellipsis suffix; got {:?}",
                t.label
            );
        }
        match warning {
            Some(AxisLabelWarning::LabelsElided { count }) => assert_eq!(count, 20),
            other => panic!("expected LabelsElided{{count: 20}}, got {:?}", other),
        }
    }

    #[test]
    fn x_axis_cascade_resolves_dense_labels_without_elision() {
        // 20 labels in 200px panel. slot_w=10. Labels are "Label_0" etc.
        // S1-S2 fail (segments too wide for 9px threshold).
        // S3: at -90, cos(90)~0, projected~0 <= slot_w -> passes at -90.
        // No elision needed.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let (axis, warning) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
            assert!(!t.elided, "cascade should resolve at -90 without elision");
        }
        assert!(warning.is_none(), "no LabelsElided warning expected");
    }

    #[test]
    fn x_axis_elision_unicode_safe() {
        // Use label_angle_override to bypass the cascade and force elision.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["héllo wörld".into(); 20],
            Some(-45.0),
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert!(t.elided);
            assert!(t.label.is_char_boundary(t.label.len()));
        }
    }

    // --- minor-tick threading tests (Grid item 18, Task 2) ---

    /// Build an AxisInput with the minor gate and positions set explicitly.
    fn axis_input_with_minor(
        orient: AxisOrient,
        labels: Vec<String>,
        include_minor: bool,
        minor_positions: Vec<f64>,
    ) -> AxisInput {
        let mut input = axis_input(orient, None, labels, None);
        // Mirror prepare.rs: the gate empties `minor` when off. With no major
        // fractions supplied, presence of the projection is driven by the
        // minors here (these fixtures pre-date the continuous-major path).
        let minor = if include_minor {
            minor_positions
        } else {
            Vec::new()
        };
        if !minor.is_empty() {
            input.tick_projection = Some(TickProjection {
                padding_frac: 0.0,
                major: Vec::new(),
                minor,
                text_demand: Default::default(),
            });
        }
        input
    }

    // --- project_fractions NonFinitePolicy tests (LAYOUT-845) ---

    #[test]
    fn project_fractions_dropall_finite_maps_via_lerp() {
        // Finite fractions over `(0, 100)` with no padding lerp to `lo + t*(hi-lo)`.
        let got = project_fractions(
            &[0.0, 0.5, 1.0],
            (0.0, 100.0),
            0.0,
            &Default::default(),
            NonFinitePolicy::DropAll,
        );
        assert_eq!(got, Some(vec![0.0, 50.0, 100.0]));
    }

    #[test]
    fn project_fractions_dropall_discards_whole_projection_on_nonfinite() {
        // One non-finite fraction → the entire (major) projection is dropped so the
        // caller falls back to uniform slots. All-or-nothing.
        let got = project_fractions(
            &[0.0, f64::NAN, 1.0],
            (0.0, 100.0),
            0.0,
            &Default::default(),
            NonFinitePolicy::DropAll,
        );
        assert_eq!(got, None);
    }

    #[test]
    fn project_fractions_dropeach_filters_only_nonfinite() {
        // The same non-finite input under DropEach keeps the finite pixels and
        // drops only the bad one — the minor-tick policy.
        let got = project_fractions(
            &[0.0, f64::INFINITY, 1.0],
            (0.0, 100.0),
            0.0,
            &Default::default(),
            NonFinitePolicy::DropEach,
        );
        assert_eq!(got, Some(vec![0.0, 100.0]));
    }

    #[test]
    fn project_fractions_empty_is_none_for_both_policies() {
        assert_eq!(
            project_fractions(&[], (0.0, 100.0), 0.0, &Default::default(), NonFinitePolicy::DropAll),
            None
        );
        assert_eq!(
            project_fractions(&[], (0.0, 100.0), 0.0, &Default::default(), NonFinitePolicy::DropEach),
            None
        );
    }

    #[test]
    fn minor_gate_off_emits_only_majors() {
        // Even with minor fractions supplied, the gate being off must yield an
        // empty minor_ticks and major-only `ticks` (each tagged is_major=true).
        let input = axis_input_with_minor(
            AxisOrient::Bottom,
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            false,
            vec![0.1, 0.2, 0.3],
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 10.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );

        assert!(
            axis.minor_ticks.is_empty(),
            "gate off must produce no minors"
        );
        assert_eq!(axis.ticks.len(), 4);
        for t in &axis.ticks {
            assert!(t.is_major, "all `ticks` entries must be majors");
        }
    }

    #[test]
    fn minor_gate_off_y_axis_emits_only_majors() {
        let input = axis_input_with_minor(
            AxisOrient::Left,
            vec!["0".into(), "1".into(), "2".into()],
            false,
            vec![0.2, 0.6],
        );
        let panel = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel, 0, 11.0, 13.0, 4.0, 4.0, &m);

        assert!(axis.minor_ticks.is_empty());
        assert_eq!(axis.ticks.len(), 3);
        for t in &axis.ticks {
            assert!(t.is_major);
        }
    }

    #[test]
    fn minor_gate_on_continuous_threads_minors_between_majors() {
        // 2 majors in a 0..400 panel with 2 labels → slot centers 100 and 300.
        // Supply normalized minor fractions; with extent=400 they map to pixels
        // 50, 150, 200, 250, 350 — interior ones strictly between the majors.
        // All must appear in minor_ticks: unlabeled, is_major=false, culled=false.
        // Majors stay labeled and is_major=true.
        let input = axis_input_with_minor(
            AxisOrient::Bottom,
            vec!["0".into(), "10".into()],
            true,
            vec![0.125, 0.375, 0.5, 0.625, 0.875],
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 10.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );

        // Majors unchanged: 2 labeled ticks at 100 and 300.
        assert_eq!(axis.ticks.len(), 2);
        assert!((axis.ticks[0].position - 100.0).abs() < 1e-9);
        assert!((axis.ticks[1].position - 300.0).abs() < 1e-9);
        for t in &axis.ticks {
            assert!(t.is_major);
            assert!(!t.label.is_empty());
        }

        // Minors: 5 of them, fraction * 400 + panel.x (0.0 here).
        assert_eq!(axis.minor_ticks.len(), 5);
        let expected = [50.0, 150.0, 200.0, 250.0, 350.0];
        for (mt, &exp) in axis.minor_ticks.iter().zip(expected.iter()) {
            assert!((mt.position - exp).abs() < 1e-9);
            assert!(!mt.is_major, "minor must be is_major=false");
            assert_eq!(mt.label, "", "minor must carry no label");
            assert!(!mt.culled, "minors are never culled");
            assert!(!mt.elided, "minors are never elided");
        }

        // The three interior minors fall strictly between the two majors.
        let m0 = axis.ticks[0].position;
        let m1 = axis.ticks[1].position;
        for interior in [150.0, 200.0, 250.0] {
            assert!(
                interior > m0 && interior < m1,
                "minor {interior} must lie strictly between majors {m0} and {m1}"
            );
        }
    }

    #[test]
    fn minor_positions_use_inverted_inset_projection_on_y() {
        // Minors are domain fractions; on the y axis layout maps them through the
        // SAME inverted base range `(y+h, y)` + inset that places majors. With
        // scale_padding_frac=0 the inset is a no-op, so frac f → (y+h) - f*h.
        let input = axis_input_with_minor(
            AxisOrient::Left,
            vec!["0".into(), "1".into()],
            true,
            vec![0.125, 0.375],
        );
        let panel = Rect {
            x: 100.0,
            y: 40.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel, 0, 11.0, 13.0, 4.0, 4.0, &m);

        assert_eq!(axis.minor_ticks.len(), 2);
        // base_range = (240, 40), span = -200: 240 - 0.125*200 = 215, 240 - 0.375*200 = 165.
        assert!((axis.minor_ticks[0].position - 215.0).abs() < 1e-9);
        assert!((axis.minor_ticks[1].position - 165.0).abs() < 1e-9);
        for mt in &axis.minor_ticks {
            assert!(!mt.is_major);
            assert_eq!(mt.label, "");
        }
    }

    #[test]
    fn minor_gate_on_categorical_empty_positions_yields_no_minors() {
        // Categorical/band scales return empty minors at the engine boundary;
        // with the gate on but no positions, minor_ticks stays empty and majors
        // are unchanged. (Mirrors the band-scale case.)
        let input = axis_input_with_minor(
            AxisOrient::Bottom,
            vec!["a".into(), "b".into(), "c".into()],
            true,
            vec![],
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 10.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );

        assert!(axis.minor_ticks.is_empty(), "no positions → no minors");
        assert_eq!(axis.ticks.len(), 3);
        for t in &axis.ticks {
            assert!(t.is_major);
        }
    }

    #[test]
    fn minor_pixel_matches_inset_projection_of_its_domain_value() {
        // Alignment fix: a minor at domain fraction f must land at the SAME pixel
        // the major projection of f gives — the inset projection
        // (inset_pixel_range + lerp), NOT the naive origin + f*extent.
        //
        // Padding_frac=0.05 in a 0..600 panel: cap binds (600*0.05=30 > 8), so
        // inset is 8px → band (8, 592), span 584. A minor at f=0.5 lands at
        // 8 + 0.5*584 = 300 — coinciding with a major at f=0.5. The naive
        // origin+f*extent would give 0.5*600 = 300 here only because the panel
        // origin is 0 and f=0.5 is the midpoint; use f=0.25 to separate them.
        let mut input = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["lo".into(), "hi".into()],
            None,
        );
        input.tick_projection = Some(TickProjection {
            padding_frac: 0.05,
            major: vec![0.0, 1.0],
            minor: vec![0.25],
            text_demand: Default::default(),
        });

        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );

        // Inset band (8, 592), span 584 → minor at f=0.25: 8 + 0.25*584 = 154.
        assert_eq!(axis.minor_ticks.len(), 1);
        let minor_px = axis.minor_ticks[0].position;
        assert!(
            (minor_px - 154.0).abs() < 1e-9,
            "inset projection expected 154, got {minor_px}"
        );

        // The minor must NOT equal the naive origin + f*extent (0.25*600 = 150).
        assert!(
            (minor_px - 150.0).abs() > 1.0,
            "minor must use inset projection (154), not naive linear interp (150); got {minor_px}"
        );

        // Cross-check: a MAJOR projected at the same fraction 0.25 lands at the
        // same pixel. Reuse the same inset path via projected_tick_fractions.
        let mut major_input = axis_input(AxisOrient::Bottom, None, vec!["q".into()], None);
        major_input.tick_projection = Some(TickProjection {
            padding_frac: 0.05,
            major: vec![0.25],
            minor: Vec::new(),
            text_demand: Default::default(),
        });
        let (major_axis, _) = layout_x_axis(
            &major_input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        let major_px = major_axis.ticks[0].position;
        assert!(
            (minor_px - major_px).abs() < 1e-9,
            "minor at f=0.25 ({minor_px}) must coincide with major projection of f=0.25 ({major_px})"
        );
    }

    // --- wrap_label tests ---

    #[test]
    fn wrap_underscore() {
        // "trivial" = 7 chars * 10 = 70, "baseline" = 8 chars * 10 = 80 — both ≤ 80
        let m = mock(10.0);
        let result = wrap_label("trivial_baseline", 80.0, 11.0, &m);
        assert_eq!(result, Some("trivial\nbaseline".to_string()));
    }

    #[test]
    fn wrap_underscore_four_segments() {
        // "very"(4), "long"(4), "snake"(5), "case"(4), "name"(4) * 10 = 40/40/50/40/40
        // All segments fit within 80. Result should have 5 lines joined by \n.
        let m = mock(10.0);
        let result = wrap_label("very_long_snake_case_name", 80.0, 11.0, &m);
        let s = result.expect("should wrap");
        let lines: Vec<&str> = s.split('\n').collect();
        assert!(lines.len() >= 4, "expected 4+ lines, got {}", lines.len());
        assert_eq!(lines, vec!["very", "long", "snake", "case", "name"]);
    }

    #[test]
    fn wrap_space_greedy() {
        // "long"(4)*10=40, "category"(8)*10=80, "name"(4)*10=40
        // max_width = 120: "long category" = 13 chars + 1 space = 14*10 = 140 > 120
        // Greedy: "long" fits (40), "long category" = 40+1+80 = word-sep logic:
        //   candidate = "long category" = measure("long category", 11) = 14*10 = 140 > 120 → wrap
        //   so line1 = "long", then "category" = 80 ≤ 120, "category name" = 14*10=140 > 120 → wrap
        //   line2 = "category", line3 = "name"
        // Expected: "long\ncategory\nname"
        let m = mock(10.0);
        let result = wrap_label("long category name", 120.0, 11.0, &m);
        assert_eq!(result, Some("long\ncategory\nname".to_string()));
    }

    #[test]
    fn wrap_camel_case() {
        // "feature"(7)*10=70 ≤ 100, "Importance"(10)*10=100 ≤ 100
        let m = mock(10.0);
        let result = wrap_label("featureImportance", 100.0, 11.0, &m);
        assert_eq!(result, Some("feature\nImportance".to_string()));
    }

    #[test]
    fn wrap_no_break_point() {
        // "abcdefghij" has no _, no space, no camelCase boundary — no break point
        let m = mock(10.0);
        let result = wrap_label("abcdefghij", 50.0, 11.0, &m);
        assert_eq!(result, None);
    }

    #[test]
    fn wrap_segment_too_wide() {
        // "a"(1)*10=10 ≤ 30, but "verylongword"(12)*10=120 > 30 → None
        let m = mock(10.0);
        let result = wrap_label("a_verylongword", 30.0, 11.0, &m);
        assert_eq!(result, None);
    }

    #[test]
    fn wrap_single_word_no_breaks() {
        // "hello" fits flat (5*10=50 ≤ 100), but has no break points → None
        let m = mock(10.0);
        let result = wrap_label("hello", 100.0, 11.0, &m);
        assert_eq!(result, None);
    }

    /// A leading space splits to an empty
    /// first word (`" AB CD".split(' ')` == `["", "AB", "CD"]`);
    /// `wrap_label`'s `String` accumulator must drop that empty word exactly
    /// like pre-task did (`String::push_str("")` on an empty string is a
    /// no-op, so `current` stays empty and the NEXT real word starts fresh
    /// with no separator), not carry it into the joined line as a stray
    /// leading space. "AB"(2*10=20px) and "CD"(20px) each fit alone but
    /// "AB CD" combined (5*10=50px) exceeds max_width=30, so this forces an
    /// actual 2-line split — the case the original bug corrupted.
    #[test]
    fn wrap_leading_space_drops_empty_word_not_stray_space() {
        let m = mock(10.0);
        let result = wrap_label(" AB CD", 30.0, 11.0, &m);
        assert_eq!(
            result,
            Some("AB\nCD".to_string()),
            "leading space must not survive into the wrapped line"
        );
    }

    /// A trailing empty word that FITS is
    /// pre-task-genuine kept content, not dropped -- `format!("{} {}",
    /// current, word)` inserts a joining space and MEASURES it, so a
    /// trailing space that's within budget stays on the line. "AB CD "
    /// splits to `["AB", "CD", ""]`; at max_width=30 (`mock(10.0)`,
    /// `fixed_width` counts every char INCLUDING spaces), the candidate for
    /// the trailing "" is `"CD "` == 3 chars == 30.0px, and `30.0 > 30.0` is
    /// FALSE, so pre-task keeps it: `current = "CD "`.
    ///
    /// Oracle derived on paper by hand-tracing the removed `String`-loop
    /// (`git show` of the pre-task body) against `fixed_width(10.0)`'s
    /// exact per-character arithmetic (not a rebuilt base-commit extension).
    #[test]
    fn wrap_trailing_empty_word_that_fits_keeps_pretask_trailing_space() {
        let m = mock(10.0);
        let result = wrap_label("AB CD ", 30.0, 11.0, &m);
        assert_eq!(
            result,
            Some("AB\nCD ".to_string()),
            "a trailing empty word whose joining space still fits must be \
             kept, matching pre-task's format!-and-measure behavior"
        );
    }

    /// Companion: a trailing empty word that OVERFLOWS its line IS dropped
    /// (pre-task's final `if !current.is_empty()` guard: the word becomes
    /// `current = "".to_string()` after the overflow push, and an empty
    /// `current` is never pushed as its own line). Same fixture, tighter
    /// budget (25 instead of 30) so `"CD "` (30px) no longer fits.
    #[test]
    fn wrap_trailing_empty_word_that_overflows_is_dropped() {
        let m = mock(10.0);
        let result = wrap_label("AB CD ", 25.0, 11.0, &m);
        assert_eq!(
            result,
            Some("AB\nCD".to_string()),
            "a trailing empty word that overflows its line must be dropped, \
             not pushed as a phantom extra (empty) line"
        );
    }

    /// Headline case: a DOUBLED interior
    /// space is a LINE-COUNT divergence, not just whitespace cosmetics.
    /// "AB  CD" (two spaces) splits to `["AB", "", "CD"]`. At max_width=60,
    /// pre-task keeps everything on one line (the doubled space measures
    /// into the combined candidate, which still fits): `"AB  CD"`. At
    /// max_width=50, the SAME candidate (`"AB  CD"` == 60px) no longer
    /// fits, so pre-task splits — but the interior empty word already
    /// landed on the FIRST line's `current` (`"AB "`, 30px, fit) before the
    /// real second word arrived, so the split lands as `"AB "` / `"CD"`,
    /// TWO lines with a trailing space on the first — not `"AB CD"` on one
    /// (what a same-space-collapsing implementation would produce).
    #[test]
    fn wrap_doubled_interior_space_keeps_pretask_line_count() {
        let m = mock(10.0);
        let fits_one_line = wrap_label("AB  CD", 60.0, 11.0, &m);
        assert_eq!(
            fits_one_line,
            Some("AB  CD".to_string()),
            "a doubled space that still fits combined must stay on one line, \
             double space preserved"
        );

        let splits_two_lines = wrap_label("AB  CD", 50.0, 11.0, &m);
        assert_eq!(
            splits_two_lines,
            Some("AB \nCD".to_string()),
            "a doubled space that no longer fits combined must split into \
             pre-task's exact two lines (\"AB \" / \"CD\"), not collapse the \
             doubled space away"
        );
    }

    // --- measure_multiline_width test (via axis.rs import) ---

    #[test]
    fn multiline_width_returns_max_line() {
        // "trivial"(7)*10=70, "baseline"(8)*10=80 → max=80
        let m = mock(10.0);
        let w = measure_multiline_width("trivial\nbaseline", 11.0, &m);
        assert!((w - 80.0).abs() < 1e-12);
    }

    // --- cascade_collision_recovery tests ---

    #[test]
    fn cascade_s0_flat() {
        // 4 short labels in 400px panel. slot_w=100, threshold=90.
        // "AAAA"=4*10=40 <= 90 -> no collision -> S0 flat.
        let labels: Vec<String> = vec!["AAAA".into(), "BBBB".into(), "CCCC".into(), "DDDD".into()];
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 100.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(result.angle, 0.0);
        assert!(result.font_size.is_none());
        assert_eq!(result.strategy, CascadeStrategy::Flat);
        assert!(result.visible.iter().all(|v| *v));
        assert_eq!(result.labels, labels);
    }

    /// S0 judges each adjacent PAIR against the pitch, not every label
    /// against the axis's single widest member. Five alternating
    /// narrow/wide labels whose widest member (90px) alone would overflow
    /// a uniform per-label threshold at this pitch, but every adjacent
    /// pair averages to 55px, comfortably within it, since the 90px label
    /// never sits next to another 90px label. `fits_adjacent_pairs` passes
    /// it flat.
    #[test]
    fn cascade_s0_flat_judges_adjacent_pairs_not_the_widest_label_alone() {
        let labels: Vec<String> = vec![
            "AB".into(),
            "ABCDEFGHI".into(),
            "CD".into(),
            "ABCDEFGHI".into(),
            "EF".into(),
        ];
        let m = mock(10.0);
        let result =
            cascade_collision_recovery(&labels, 100.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(
            result.strategy,
            CascadeStrategy::Flat,
            "every adjacent pair fits the pitch even though one label alone exceeds \
             a uniform per-label threshold; got {:?}",
            result.strategy
        );
        assert_eq!(result.labels, labels);
        assert!(result.visible.iter().all(|v| *v));
    }

    // --- D: continuous-x thinning --------------------------------------

    /// 9 equal-width labels that collide flat and wrap (no break points) at
    /// `slot_w = 50`, where S3 (rotate) genuinely fits at -60° keeping every
    /// label — the exact shape `cascade_genuinely_fitting_rotation_wins_over_culling`
    /// pins for a CATEGORICAL axis. On a CONTINUOUS axis (`is_continuous:
    /// true`) with culling permitted, D must preempt that rotation: stride 2
    /// opens a pitch of 100px, wide enough for the 5 survivors to sit flat at
    /// full font, and D accepts the first such stride before S2/S3 ever run.
    /// The adjacent-pair test alone (an identical width set has no asymmetry
    /// for it to exploit) still leaves this axis rotated at -60° with every
    /// label kept — only D thins it.
    #[test]
    fn cascade_d_thinning_wins_over_rotation_on_continuous_x() {
        let labels: Vec<String> = (0..9).map(|_| "AAAAA".to_string()).collect();
        let m = mock(10.0);
        let result =
            cascade_collision_recovery(&labels, 50.0, 11.0, legacy_gap(8), None, &m, TickCullFrame { anchors: &[], origin: 0, continuous: true });
        assert_eq!(
            result.strategy,
            CascadeStrategy::Culled { stride: 2 },
            "a continuous axis must thin before rotating; got {:?}",
            result.strategy
        );
        assert_eq!(result.angle, 0.0);
        assert!(result.font_size.is_none());
        assert_eq!(result.visible, stride_visibility(9, 2, 0));
        let shown = result.visible.iter().filter(|v| **v).count();
        assert_eq!(shown, 5, "stride 2 over 9 labels keeps 5");
    }

    /// Companion pin: the SAME labels, slot and gap policy as
    /// `cascade_d_thinning_wins_over_rotation_on_continuous_x`, but
    /// `is_continuous: false` — a categorical or scale-less axis skips
    /// thinning and rotates every label instead.
    #[test]
    fn cascade_d_thinning_skipped_on_categorical_x() {
        let labels: Vec<String> = (0..9).map(|_| "AAAAA".to_string()).collect();
        let m = mock(10.0);
        let result =
            cascade_collision_recovery(&labels, 50.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(
            result.strategy,
            CascadeStrategy::Rotated { angle: -60.0 },
            "a categorical axis must rotate, not thin; got {:?}",
            result.strategy
        );
        assert!(result.visible.iter().all(|v| *v), "no label is dropped");
    }

    /// No stride over 4 labels ever reaches [`MIN_EVENLY_THINNED_X_LABELS`]
    /// (stride 2 keeps 2, stride 3 keeps 2, stride 4 keeps 1), so D declines
    /// every candidate and the axis falls through to S2 -> S3 -> S4 -> S5 —
    /// here, S3's genuinely-fitting -60° rotation, with every label kept,
    /// same as the categorical companion above.
    #[test]
    fn cascade_d_thinning_requires_at_least_min_evenly_thinned_labels() {
        let labels: Vec<String> = (0..4).map(|_| "AAAA".to_string()).collect();
        let m = mock(10.0);
        let result =
            cascade_collision_recovery(&labels, 40.0, 11.0, legacy_gap(8), None, &m, TickCullFrame { anchors: &[], origin: 0, continuous: true });
        assert_eq!(
            result.strategy,
            CascadeStrategy::Rotated { angle: -60.0 },
            "too few survivors at every stride must fall through to rotation, \
             not an under-thinned cull; got {:?}",
            result.strategy
        );
        assert!(result.visible.iter().all(|v| *v), "no label is dropped");
    }

    #[test]
    fn cascade_s1_wrap() {
        // 4 snake_case labels that collide flat but wrap fits.
        // slot_w = 100, threshold = 90.
        // "trivial_baseline" = 16 chars * 5 = 80 (flat) -> 80 <= 90? Yes!
        // Wait, we need them to collide flat. Use per_char_px=6.
        // "trivial_baseline" = 16 * 6 = 96 > 90 -> collision.
        // Wrap: "trivial" = 7*6=42, "baseline" = 8*6=48.
        // measure_multiline_width = max(42, 48) = 48 <= 90 -> wrapping resolves.
        let labels: Vec<String> = vec![
            "trivial_baseline".into(),
            "negative_prompt".into(),
            "persona_limited".into(),
            "minimal_context".into(),
        ];
        let m = mock(6.0);
        let result = cascade_collision_recovery(&labels, 100.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(result.angle, 0.0);
        assert!(result.font_size.is_none());
        assert_eq!(result.strategy, CascadeStrategy::Wrapped);
        // All labels should contain \n
        for lbl in &result.labels {
            assert!(lbl.contains('\n'), "expected wrapped label, got {:?}", lbl);
        }
        assert!(result.visible.iter().all(|v| *v));
    }

    #[test]
    fn cascade_s2_font_shrink() {
        // Labels that collide at the original font size but fit at reduced.
        // We need a mock that IS sensitive to font_size so the shrink matters.
        // Use a closure: width = chars * font_size * 0.5
        // "ABCDEFGHIJ" = 10 chars. At fs=11: 10*11*0.5=55. At fs=11*0.82=9.02: 10*9.02*0.5=45.1
        // slot_w=60, threshold=54. 55>54 -> collision. 45.1<=54 -> reduced fits.
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
        ];
        let m = MockMetrics {
            measure: |text: &str, font_size: f64| text.chars().count() as f64 * font_size * 0.5,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(&labels, 60.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(result.angle, 0.0);
        assert_eq!(result.strategy, CascadeStrategy::FontReduced);
        let expected_fs = 11.0 * 0.82;
        assert!((result.font_size.unwrap() - expected_fs).abs() < 1e-6);
        assert!(result.visible.iter().all(|v| *v));
    }

    #[test]
    fn cascade_s3_rotation() {
        // Labels without break points that collide flat.
        // "ABCDEFGHIJ" = 10*10 = 100px. slot_w=80, threshold=72. line_h = 11*1.2 = 13.2.
        // S1: no break points -> fails.
        // S2: fixed_width ignores font_size -> still 100 -> fails.
        // S3 (real footprint = cos*w + sin*line_h, spec §4.6 / F-L07-04 fix):
        //   -30: cos(30)*100 + sin(30)*13.2 = 86.60 + 6.6 = 93.20 > 80 -> fail.
        //   -45: cos(45)*100 + sin(45)*13.2 = 70.71 + 9.34 = 80.05 > 80 -> fail
        //        (barely — this is the exact margin the pre-fix bare-cos check
        //        missed: 70.71<=80 alone used to auto-pass here).
        //   -60: cos(60)*100 + sin(60)*13.2 = 50.00 + 11.43 = 61.43 <= 80 -> pass!
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "UVWXYZABCD".into(),
            "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 80.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(result.strategy, CascadeStrategy::Rotated { angle: -60.0 });
        assert_eq!(result.angle, -60.0);
        assert!(result.font_size.is_none());
        assert!(result.visible.iter().all(|v| *v));
        // Labels unchanged (not wrapped, not elided).
        assert_eq!(result.labels, labels);
    }

    #[test]
    fn cascade_s3_picks_shallowest_angle() {
        // Pure angle-selection test (isolated from the cull_threshold gap
        // feature via cull_threshold=0 — see `cascade_gap_requirement_*`
        // below for that interaction). Labels that fit at -45 (not the
        // shallower -30, once the real footprint formula is used).
        // 6 chars * 10 = 60px. slot_w=55. line_h = 11*1.2 = 13.2.
        // threshold=49.5. 60>49.5 -> collision.
        // S1: no break points -> fails.
        // S2: fixed_width ignores font_size -> fails.
        // S3 (real footprint, spec §4.6 / F-L07-04 fix; cull_threshold=0 so
        // the budget is plain `slot_w`, no gap term):
        //   -30: cos(30)*60 + sin(30)*13.2 = 51.96 + 6.6 = 58.56 > 55 -> fail
        //        (the pre-fix bare-cos check alone, 51.96<=55, used to pass here).
        //   -45: cos(45)*60 + sin(45)*13.2 = 42.43 + 9.34 = 51.76 <= 55 -> pass at -45!
        let labels: Vec<String> = vec![
            "ABCDEF".into(),
            "GHIJKL".into(),
            "MNOPQR".into(),
            "STUVWX".into(),
        ];
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 55.0, 11.0, legacy_gap(0), None, &m, scale_less_frame());
        assert_eq!(result.strategy, CascadeStrategy::Rotated { angle: -45.0 });
        assert_eq!(result.angle, -45.0);
    }

    /// F-L07-04 / spec §4.6 RED proof: culling is reachable at REALISTIC label
    /// widths, not just via an unrealistic mocked width (see
    /// `cascade_s4_culling_direct` below, which predates this fix and used a
    /// 1e18-px mock specifically because the old bare-`cos(angle)` fit check
    /// made -90° auto-pass for any real label width — the bug this test now
    /// pins the fix for). 20 labels of realistic width, no break points (so
    /// wrap/shrink can't resolve them), and a slot narrower than one line
    /// height (`line_h = 11*1.2 = 13.2`, `slot_w = 10.0`) so even the
    /// steepest rotation (-90°) genuinely doesn't fit — its true footprint is
    /// ~`line_h` (13.2), not ~0. With `n(20) > cull_threshold(8)`, this must
    /// resolve to `Culled`, not an always-passing `Rotated`.
    #[test]
    fn cascade_s4_culling_at_realistic_widths() {
        // "LONGCATEGORY00".."LONGCATEGORY19" = 14 chars * 10px/char = 140px each.
        let labels: Vec<String> = (0..20).map(|i| format!("LONGCATEGORY{:02}", i)).collect();
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        match result.strategy {
            CascadeStrategy::Culled { stride } => {
                // max footprint at -90 ~= line_h = 13.2; stride is the additive
                // gap-inclusive formula:
                // ceil((13.2 + cull_threshold(8)) / slot_w(10)) = ceil(2.12) = 3.
                assert_eq!(stride, 3, "expected stride 3, got {stride}");
                let visible_count = result.visible.iter().filter(|v| **v).count();
                assert!(visible_count < 20, "some labels should be culled");
                assert!(result.visible[0], "first label should be visible");
            }
            other => panic!(
                "expected Culled at a slot narrower than one line height; got {:?}",
                other
            ),
        }
        assert_eq!(result.angle, -90.0);
    }

    /// F-L07-04 pin: `cull_threshold = 0` disables culling entirely (spec
    /// §4.6), even for the same dense/narrow scenario that culls above —
    /// falls through to S5 elision instead.
    #[test]
    fn cascade_cull_threshold_zero_never_culls() {
        let labels: Vec<String> = (0..20).map(|i| format!("LONGCATEGORY{:02}", i)).collect();
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(0), None, &m, scale_less_frame());
        assert!(
            !matches!(result.strategy, CascadeStrategy::Culled { .. }),
            "cull_threshold=0 must never cull; got {:?}",
            result.strategy
        );
    }

    /// F-L07-04 pin: a rotation that genuinely fits still wins over culling,
    /// even when `n > cull_threshold` — culling is a fallback for labels that
    /// truly don't fit at any angle (INCLUDING the requested gap), not a
    /// blanket density trigger. Same 20 labels as
    /// `cascade_s4_culling_at_realistic_widths`, but `slot_w = 22.0`: with
    /// `cull_threshold = 8`, S3's budget is `22 - 8 = 14`, just above -90°'s
    /// footprint (`line_h = 13.2`) — the exact boundary the gap-aware S3
    /// budget makes meaningful (at `slot_w = 20.0`, a gap-unaware budget,
    /// `20 - 8 = 12 < 13.2` would fail and cull instead — the gap
    /// requirement, once folded into S3, changes which slot widths count as
    /// "genuinely fits").
    #[test]
    fn cascade_genuinely_fitting_rotation_wins_over_culling() {
        let labels: Vec<String> = (0..20).map(|i| format!("LONGCATEGORY{:02}", i)).collect();
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 22.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(
            result.strategy,
            CascadeStrategy::Rotated { angle: -90.0 },
            "a genuinely-fitting rotation must win over culling"
        );
        assert!(
            result.visible.iter().all(|v| *v),
            "no labels should be culled"
        );
    }

    #[test]
    fn cascade_s4_culling_direct() {
        // Direct test of cascade_collision_recovery with a mock that makes
        // S3 fail for all angles. We make the mock return a width that
        // depends on whether we're in the "cos" check path by using a very
        // large width where even cos(-90) * width > slot_w.
        //
        // cos(-90 deg) in f64: (-90.0_f64).to_radians().cos().abs() ≈ 6.12e-17
        // For width = 1e18: 1e18 * 6.12e-17 ≈ 61.2 > slot_w=10
        // This forces S3 to fail for all angles.
        let labels: Vec<String> = (0..20).map(|i| format!("X{}", i)).collect();
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        // 20 > 8 (cull_threshold) -> culling eligible.
        match result.strategy {
            CascadeStrategy::Culled { stride } => {
                assert!(stride > 1, "expected stride > 1");
                // Verify some labels are hidden.
                let visible_count = result.visible.iter().filter(|v| **v).count();
                assert!(visible_count < 20, "some labels should be culled");
                assert!(result.visible[0], "first label should be visible");
            }
            other => panic!("expected Culled, got {:?}", other),
        }
        assert_eq!(result.angle, -90.0);
    }

    /// F-L07-04 / D9 boundary pin: `cull_threshold` gates culling as an
    /// all-or-nothing switch on the PIXEL-GAP semantic (spec §4.6), not a
    /// label-count predicate — `0` disables culling entirely regardless of
    /// how badly labels collide, and any positive value re-enables it.
    /// Same labels/slot in both branches, only `cull_threshold` differs, so
    /// this isolates exactly the switch (supersedes the pre-fix count-gate
    /// boundary this test used to pin: `n as u32 > cull_threshold`).
    #[test]
    fn cascade_cull_threshold_zero_vs_one_boundary() {
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let labels: Vec<String> = (0..8).map(|i| format!("VeryLongLabel{i}")).collect();

        let disabled = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(0), None, &m, scale_less_frame());
        assert!(
            matches!(disabled.strategy, CascadeStrategy::Elided { .. }),
            "cull_threshold=0 must disable culling entirely; expected Elided, got {:?}",
            disabled.strategy
        );

        let enabled = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(1), None, &m, scale_less_frame());
        assert!(
            matches!(enabled.strategy, CascadeStrategy::Culled { .. }),
            "cull_threshold=1 must enable culling; expected Culled, got {:?}",
            enabled.strategy
        );
    }

    /// `#146` / spec §4.4 — mode and magnitude are now independent, and the
    /// legacy integer spelling still maps onto exactly the two corners it
    /// always meant.
    ///
    /// The last two assertions below are this test's actual guarantee: the
    /// default's `cull` mode survives a change to the default's `gap_px`.
    /// They do NOT prove `Default` is independently *implemented* today —
    /// at the current constants (`DEFAULT_GAP_PX = 8`, `DEFAULT_CULL =
    /// true`) a `Default` rewritten as `Self::from_cull_threshold(Self::
    /// DEFAULT_GAP_PX)` would compute `cull: 8 > 0 == true`, which is
    /// bit-identical to the real `DEFAULT_CULL`. That rewrite is an
    /// equivalent mutant *right now* — it changes the implementation
    /// without changing any observable value, so no assertion on `Default`'s
    /// output can distinguish it, and none should try to (chasing an
    /// equivalent mutant is how the source-grep version of this test got
    /// written and then reverted). The rewrite stops being equivalent the
    /// moment `#134` moves `DEFAULT_GAP_PX` to `0`: a derived `Default`
    /// would then silently compute `cull: 0 > 0 == false`, while the real
    /// (independent) one still yields `DEFAULT_CULL`. The `d.may_cull() ==
    /// LabelGapPolicy::DEFAULT_CULL` assertion is exactly what catches that
    /// — it fails the instant the coincidence breaks, which is the moment
    /// this guard is actually needed.
    #[test]
    fn label_gap_policy_default_mode_survives_default_gap_change() {
        assert_eq!(LabelGapPolicy::from_cull_threshold(0).gap_px(), 0);
        assert!(!LabelGapPolicy::from_cull_threshold(0).may_cull());
        assert_eq!(LabelGapPolicy::from_cull_threshold(30).gap_px(), 30);
        assert!(LabelGapPolicy::from_cull_threshold(30).may_cull());

        // The default is the legacy default gap plus an INDEPENDENTLY
        // declared mode, so `#134` can move `DEFAULT_GAP_PX` alone — even to
        // `0` — without silently flipping culling off with it.
        let d = LabelGapPolicy::default();
        assert_eq!(d.gap_px(), LabelGapPolicy::DEFAULT_GAP_PX);
        assert_eq!(d.may_cull(), LabelGapPolicy::DEFAULT_CULL);
    }

    /// `#146` — the two Theme keys resolve in one place, and `cull_labels=`
    /// wins over the magnitude's legacy implication without disturbing it.
    #[test]
    fn label_gap_policy_with_overrides_resolves_both_theme_keys() {
        let base = LabelGapPolicy::default();

        // Neither key: unchanged.
        assert_eq!(base.with_overrides(None, None), base);

        // `cull_threshold=` alone keeps its pre-split meaning on BOTH axes.
        assert_eq!(
            base.with_overrides(Some(0), None),
            LabelGapPolicy::from_cull_threshold(0)
        );
        assert_eq!(
            base.with_overrides(Some(30), None),
            LabelGapPolicy::from_cull_threshold(30)
        );

        // `cull_labels=` alone changes only the mode.
        let no_cull = base.with_overrides(None, Some(false));
        assert_eq!(no_cull.gap_px(), LabelGapPolicy::DEFAULT_GAP_PX);
        assert!(!no_cull.may_cull());

        // Both: the explicit mode wins over the magnitude's implication, in
        // both directions — these are the two states the old `u32` could not
        // spell.
        let gap_no_cull = base.with_overrides(Some(30), Some(false));
        assert_eq!(gap_no_cull.gap_px(), 30);
        assert!(!gap_no_cull.may_cull());
        let no_gap_cull = base.with_overrides(Some(0), Some(true));
        assert_eq!(no_gap_cull.gap_px(), 0);
        assert!(no_gap_cull.may_cull());
    }

    /// `#146` / spec §4.4 — "require a gap of N px AND never drop a label" is
    /// a distinct cascade outcome from both legacy spellings at the same
    /// fixture. `{gap: 30, cull: false}` must keep every label (S5's
    /// infeasible-truncation guard leaves them intact at -90°) where
    /// `{gap: 30, cull: true}` drops some, and must NOT collapse to
    /// `{gap: 0, cull: false}`, whose wider S0-S3 budget resolves flat.
    #[test]
    fn cascade_gap_without_culling_is_distinct_from_both_legacy_spellings() {
        let m = MockMetrics {
            measure: |_t: &str, _fs: f64| 20.0,
            line_h_factor: 1.2,
        };
        let labels: Vec<String> = (0..12).map(|i| format!("L{i}")).collect();
        let slot_w = 40.0;

        // `{gap: 0, cull: false}`: 36px of budget for a 20px label -> S0 flat.
        let no_gap = cascade_collision_recovery(&labels, slot_w, 11.0, legacy_gap(0), None, &m, scale_less_frame());
        assert!(
            matches!(no_gap.strategy, CascadeStrategy::Flat),
            "no gap requirement must resolve flat here; got {:?}",
            no_gap.strategy
        );
        assert!(no_gap.visible.iter().all(|v| *v));

        // `{gap: 30, cull: true}`: 6px of budget -> every stage fails -> S4 culls.
        let gap_cull = cascade_collision_recovery(&labels, slot_w, 11.0, legacy_gap(30), None, &m, scale_less_frame());
        assert!(
            matches!(gap_cull.strategy, CascadeStrategy::Culled { .. }),
            "a 30px gap with culling allowed must cull; got {:?}",
            gap_cull.strategy
        );
        assert!(
            gap_cull.visible.iter().any(|v| !*v),
            "culling must actually drop labels"
        );

        // `{gap: 30, cull: false}`: same failed stages, but S4 is closed, so
        // every label survives at the steepest angle. Distinct from BOTH.
        let gap_no_cull = cascade_collision_recovery(
            &labels,
            slot_w,
            11.0,
            LabelGapPolicy::from_cull_threshold(30).with_overrides(None, Some(false)),
            None,
            &m,
            scale_less_frame(),
        );
        assert!(
            gap_no_cull.visible.iter().all(|v| *v),
            "a policy that forbids culling must never drop a label; got {:?}",
            gap_no_cull.strategy
        );
        assert_eq!(
            gap_no_cull.angle, -90.0,
            "the gap still drove the cascade past flat/wrap/font-shrink/rotate"
        );
        assert_ne!(
            gap_no_cull.angle, no_gap.angle,
            "gap-30-never-cull must not collapse onto gap-0-never-cull"
        );
    }

    /// `#146` — the fourth corner: no gap requirement, but culling permitted.
    /// Unreachable from the old integer (`0` forbade culling too), so this
    /// pins that it culls on genuine overlap alone, without the extra gap.
    #[test]
    fn cascade_culling_without_a_gap_requirement_culls_only_on_overlap() {
        let m = MockMetrics {
            measure: |_t: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let labels: Vec<String> = (0..8).map(|i| format!("VeryLongLabel{i}")).collect();
        let policy = LabelGapPolicy::from_cull_threshold(0).with_overrides(None, Some(true));
        assert_eq!(policy.gap_px(), 0);

        let result = cascade_collision_recovery(&labels, 10.0, 11.0, policy, None, &m, scale_less_frame());
        assert!(
            matches!(result.strategy, CascadeStrategy::Culled { .. }),
            "culling permitted at gap 0 must still cull a genuinely overlapping axis; \
             got {:?}",
            result.strategy
        );
    }

    /// `#146` — the gap MAGNITUDE alone drives the wrap-uniformization
    /// post-process, independent of whether culling is permitted, so the
    /// mode split cannot silently change wrapped label text.
    #[test]
    fn uniformize_keys_on_gap_magnitude_not_cull_mode() {
        let m = MockMetrics {
            measure: |t: &str, _fs: f64| t.len() as f64 * 6.0,
            line_h_factor: 1.2,
        };
        let raw = vec!["One Two Three Four".to_string(), "Two Words".to_string()];
        let natural: Vec<String> = vec!["One Two\nThree Four".to_string(), "Two Words".to_string()];

        let at_zero = uniformize_wrapped_labels(&raw, natural.clone(), 0, 60.0, 11.0, &m);
        assert_eq!(
            at_zero, natural,
            "gap 0 must leave the natural wrap untouched"
        );

        let at_eight = uniformize_wrapped_labels(&raw, natural.clone(), 8, 60.0, 11.0, &m);
        assert_ne!(
            at_eight, natural,
            "a positive gap must uniformize the sibling line counts"
        );
    }

    #[test]
    fn cascade_s5_elision() {
        // cull_threshold=0 (disabled, spec §4.6 / F-L07-04), so culling is
        // skipped entirely and elision fires as the last resort.
        // 6 labels with enormous (1e18) widths -> S3 fails for all angles ->
        // S4 skipped (cull_threshold=0) -> S5 elision. The 1e18 mock is
        // deliberate, not incidental: at -90°,
        // `cos_best ≈ 6.12e-17` makes the footprint's width term negligible
        // for any REALISTIC label width, so elision can only ever fire
        // where truncating the text delivers a REAL reduction --
        // `cos_best * 1e18 ≈ 61px`, well above `ELIDE_MIN_WIDTH_CONTRIBUTION_PX`.
        // See `cascade_s5_realistic_widths_falls_back_to_rotated_not_elision`
        // for the realistic-width companion, where elision is a no-op remedy
        // and the cascade must NOT destroy the labels' content.
        let labels: Vec<String> = (0..6).map(|i| format!("VeryLongLabel{}", i)).collect();
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(0), None, &m, scale_less_frame());
        match result.strategy {
            CascadeStrategy::Elided { count } => {
                assert!(count > 0, "expected some labels elided");
            }
            other => panic!("expected Elided, got {:?}", other),
        }
        assert_eq!(result.angle, -90.0);
        // All labels should end with ellipsis.
        for lbl in &result.labels {
            assert!(
                lbl.ends_with('\u{2026}'),
                "expected ellipsis suffix; got {:?}",
                lbl
            );
        }
    }

    /// RED proof: reaching S5 with REALISTIC label
    /// widths (not an astronomical mock) must NOT destroy content. 20 labels
    /// of realistic width (140px, `cascade_s4_culling_at_realistic_widths`'s
    /// fixture) in a slot narrower than one line height (`slot_w=10 <
    /// line_h=13.2`), `cull_threshold=0` (disabled, so S4 is skipped and S5
    /// would be reached) — the elide predicate (`~line_h`, width-independent
    /// at -90°) can never be satisfied by the elide remedy (text truncation,
    /// which only shrinks the width term) for widths this small
    /// (`cos_best * 140 ≈ 8.6e-15`, far below the 1px feasibility floor), so
    /// the cascade must fall back to `Rotated` with every label INTACT --
    /// the pre-task outcome for a threshold the user explicitly disabled --
    /// rather than collapse all 20 labels to a bare, information-free "…".
    #[test]
    fn cascade_s5_realistic_widths_falls_back_to_rotated_not_elision() {
        let labels: Vec<String> = (0..20).map(|i| format!("LONGCATEGORY{:02}", i)).collect();
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(0), None, &m, scale_less_frame());
        assert_eq!(
            result.strategy,
            CascadeStrategy::Rotated { angle: -90.0 },
            "elision that cannot reduce the footprint must fall back to Rotated, not Elided"
        );
        assert_eq!(
            result.labels, labels,
            "labels must render INTACT (no truncation) when elision cannot help"
        );
        assert!(
            result.visible.iter().all(|v| *v),
            "no labels should be culled"
        );
    }

    /// RED proof (`configure/density_styled.svg`): culling opens up the per-surviving-label pitch,
    /// and a presentation earlier than -90° rotation must be re-tried at
    /// that WIDER pitch rather than blindly inheriting the angle that made
    /// the original, narrower pitch's stride computation pass. 13 short
    /// numeric labels ("1", "1.5", ..., "7", widest = "1.5"/"2.5"/etc at 3
    /// chars = 30px) at `slot_w=12` genuinely don't fit at any angle
    /// (`s3_budget = 12 - cull_threshold(8) = 4`, far below -90°'s
    /// ~line_h=13.2 footprint) -> S4 culls with
    /// `stride = ceil((13.2 + 8) / 12) = 2`. The 7 stride-2 survivors
    /// ("1".."7") are single digits (10px each); at the CULLED pitch
    /// (`effective_slot_w = 24`), S0's flat threshold is
    /// `24*0.9 - 8 = 13.6 >= 10` — they fit FLAT. Pre-fix, this resolved to
    /// `Culled { stride: 2 }` at `angle: -90.0` unconditionally (rotated
    /// sparse single digits, strictly worse than flat); RED-proven in place
    /// (temporarily forced the post-cull branch back to
    /// `(labels.to_vec(), best_angle, None)`, rebuilt, confirmed this test
    /// failed with `angle == -90.0`; reverted, rebuilt, confirmed it passes).
    #[test]
    fn cascade_culled_survivors_reselect_flat_when_it_now_fits() {
        let labels: Vec<String> = vec![
            "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6", "6.5", "7",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 12.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        match result.strategy {
            CascadeStrategy::Culled { stride } => {
                assert_eq!(stride, 2, "expected stride 2, got {stride}");
            }
            other => panic!("expected Culled at slot_w=12; got {:?}", other),
        }
        assert_eq!(
            result.angle, 0.0,
            "culled survivors that fit flat at the culled pitch must render flat, not inherit -90° rotation"
        );
        assert!(
            result.font_size.is_none(),
            "flat presentation uses the original font size"
        );
        // Survivors keep their original (unwrapped, unelided) text.
        let survivor_texts: Vec<&str> = result
            .labels
            .iter()
            .zip(&result.visible)
            .filter(|(_, v)| **v)
            .map(|(l, _)| l.as_str())
            .collect();
        assert_eq!(survivor_texts, vec!["1", "2", "3", "4", "5", "6", "7"]);
    }

    /// Companion pin: the re-selection at the culled pitch must NOT
    /// force flat where flat genuinely still doesn't fit — a wide label set
    /// culled to a wider (but still insufficient for flat/wrap) pitch
    /// legitimately keeps -90° rotation. Same fixture as
    /// `cascade_s4_culling_at_realistic_widths` (140px labels, no break
    /// points, so wrap/font-shrink can never resolve them regardless of
    /// pitch): stride 3 widens the pitch to 30px, still far short of the
    /// 140px flat requirement, so -90° (whose footprint is ~line_h
    /// regardless of label width, per `rotated_x_label_footprint`'s doc)
    /// remains the only presentation that fits.
    #[test]
    fn cascade_culled_survivors_stay_rotated_when_still_too_wide() {
        let labels: Vec<String> = (0..20).map(|i| format!("LONGCATEGORY{:02}", i)).collect();
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        match result.strategy {
            CascadeStrategy::Culled { stride } => {
                assert_eq!(stride, 3, "expected stride 3, got {stride}");
            }
            other => panic!("expected Culled; got {:?}", other),
        }
        assert_eq!(
            result.angle, -90.0,
            "wide labels that still don't fit flat/wrapped at the culled pitch must legitimately stay rotated"
        );
        assert!(result.font_size.is_none());
    }

    /// The band estimate reserves what the cascade draws, culling included:
    /// at this fixture the real cascade culls to every other label and draws
    /// the survivors flat (see the pin above), so the band is exactly the
    /// flat band, not the full -90° extent a cascade-blind estimate would
    /// reserve for labels that never rotate.
    #[test]
    fn estimate_x_label_band_reserves_the_flat_band_for_culled_to_flat_survivors() {
        let labels: Vec<String> = vec![
            "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6", "6.5", "7",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let m = mock(10.0);
        let label_font_size = 11.0;
        let tick_size = 4.0;
        let label_pad_eff = 2.0;

        let cascade_result =
            cascade_collision_recovery(&labels, 12.0, label_font_size, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(cascade_result.angle, 0.0);
        assert!(cascade_result.visible.iter().any(|v| !v), "fixture assumption: culled");

        let flat_need = tick_size + label_pad_eff + m.line_height(label_font_size);
        let estimated = estimate_band(
            &labels,
            None,
            12.0,
            None,
            true,
            &LabelBandCtx::new(label_font_size, legacy_gap(8), tick_size, &m),
        );
        assert!(
            (estimated - flat_need).abs() < 1e-9,
            "the estimate ({estimated}) is the flat band the culled survivors draw ({flat_need})"
        );
    }

    #[test]
    fn cascade_9_snake_case_600px() {
        // Acceptance test: 9 snake_case labels in a 600px panel -> NO elision.
        // slot_w = 600/9 ≈ 66.7, threshold = 66.7*0.9 ≈ 60.
        // With per_char_px=6: longest label "persona_constrained" = 19*6 = 114 > 60.
        // S1 wrap: "persona"=7*6=42, "constrained"=11*6=66 > 60 -> wrap fails
        //   for "persona_constrained" (segment too wide).
        //
        // Let me use per_char_px=5:
        // "persona_constrained" = 19*5=95 > 60 -> collision.
        // S1 wrap: "persona"=7*5=35, "constrained"=11*5=55 <= 60 -> ok.
        //   But "real_agent_config" = segments ["real"(4*5=20), "agent"(5*5=25), "config"(6*5=30)].
        //   max_line = 30 <= 60 -> ok. All labels wrap? Let me check each:
        //   "trivial_baseline" -> "trivial"(7*5=35), "baseline"(8*5=40) -> max=40 <= 60
        //   "negative_prompt" -> "negative"(8*5=40), "prompt"(6*5=30) -> max=40 <= 60
        //   "persona_constrained" -> "persona"(7*5=35), "constrained"(11*5=55) -> max=55 <= 60
        //   "minimal_context" -> "minimal"(7*5=35), "context"(7*5=35) -> max=35 <= 60
        //   "none" -> no underscore, no space, no camelCase -> wrap returns None!
        //
        // "none" has no break points, so S1 fails (not ALL labels wrap).
        // S2: reduced_fs=11*0.82=9.02. fixed_width ignores fs -> still fails.
        // S3: -30: cos(30)*95=82.3>66.7 -> fail. -45: cos(45)*95=67.2>66.7 -> fail (barely).
        //   -60: cos(60)*95=47.5<=66.7 -> pass!
        //
        // No elision, angle=-60. This is acceptable behavior.
        //
        // For a better test with HeuristicMetrics-like behavior (width depends on fs):
        let labels: Vec<String> = vec![
            "trivial_baseline".into(),
            "negative_prompt".into(),
            "persona_constrained".into(),
            "minimal_context".into(),
            "none".into(),
            "generic_coder".into(),
            "real_agent_config".into(),
            "python_coder".into(),
            "long_directive".into(),
        ];
        let m = mock(5.0); // fixed_width: chars * 5
        let slot_w = 600.0 / 9.0; // ~66.67
        let result = cascade_collision_recovery(&labels, slot_w, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        // Verify: no elision.
        assert!(
            !matches!(result.strategy, CascadeStrategy::Elided { .. }),
            "expected NO elision for 9 snake_case labels in 600px; got {:?}",
            result.strategy
        );
        // All labels should be visible.
        assert!(result.visible.iter().all(|v| *v));
        // Labels should not contain ellipsis.
        for lbl in &result.labels {
            assert!(
                !lbl.ends_with('\u{2026}'),
                "label should not be elided: {:?}",
                lbl
            );
        }
    }

    #[test]
    fn cascade_override_bypasses() {
        // label_angle_override = Some(-90.0) -> cascade not called.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            vec![
                "trivial_baseline".into(),
                "negative_prompt".into(),
                "persona_constrained".into(),
                "minimal_context".into(),
            ],
            Some(-90.0),
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0, "override should force -90");
            // Labels should not be wrapped (override bypasses cascade).
            assert!(!t.label.contains('\n'), "override should not wrap labels");
        }
    }

    #[test]
    fn cascade_s5_elision_fires_labels_elided_warning() {
        // Verify that the LabelsElided warning fires only for S5 (elision),
        // not for S3 (rotation) or other stages.
        // cull_threshold=0 (disabled) + 6 labels with enormous widths ->
        // elision (spec §4.6 / F-L07-04: cull_threshold gates S4, not a
        // label count).
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("VeryLongLabel{}", i)).collect(),
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 60.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let (_, warning) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        );
        assert!(
            matches!(warning, Some(AxisLabelWarning::LabelsElided { .. })),
            "expected LabelsElided warning; got {:?}",
            warning,
        );
    }

    #[test]
    fn cascade_rotation_no_warning() {
        // When rotation resolves collision, no LabelsElided warning should fire.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let (_, warning) = layout_x_axis(
            &input,
            panel_area,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert!(
            warning.is_none(),
            "rotation should not produce LabelsElided warning"
        );
    }

    // --- estimate_x_label_band tests ---

    #[test]
    fn estimate_flat_labels() {
        // Short labels that fit within slot_w flat should return the tick_size +
        // label_pad_eff standoff plus one line_height (#97, spec §4.1).
        // "A" = 1 char * 10 = 10px. slot_w = 100. threshold = 90. 10 <= 90 -> flat.
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let m = mock(10.0); // fixed_width: chars * 10
        let line_h = m.line_height(11.0); // 11.0 * 1.2 = 13.2
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        let expected = tick_size + 2.0 + line_h; // label_pad_eff default = 2.0
        assert!(
            (band - expected).abs() < 1e-9,
            "flat labels should return tick_size + label_pad_eff + line_height={expected}, got {band}"
        );
    }

    #[test]
    fn estimate_wrapped_labels() {
        // snake_case labels that collide flat but wrap successfully.
        // "trivial_baseline" = 16 * 6 = 96 > threshold = 90.
        // After wrap: max("trivial"=42, "baseline"=48) = 48 <= 90 -> wraps to 2 lines.
        // Expected: tick_size + label_pad_eff + 2*line_height (#97, spec §4.1).
        let labels: Vec<String> = vec![
            "trivial_baseline".into(),
            "negative_prompt".into(),
            "persona_limited".into(),
            "minimal_context".into(),
        ];
        let m = mock(6.0); // per_char * 6; "trivial_baseline" = 16*6 = 96 > 90
        let line_h = m.line_height(11.0); // 11.0 * 1.2 = 13.2
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        let expected = tick_size + 2.0 + 2.0 * line_h;
        assert!(
            (band - expected).abs() < 1e-9,
            "wrapped labels should return tick_size + label_pad_eff + 2*line_height={expected}, got {band}"
        );
    }

    /// `#182` (closes `#94`), superseding this test's own pre-fix contract: a
    /// SHOWN x axis whose labels never draw (`fm.Axis(labels=False)` →
    /// `AxisInput.show_labels = false`, `render/marks/axis.rs`'s `if
    /// axis.show_labels && !tick.culled` guard) reserves NO band — not the
    /// pre-`#97` `line_h + padding_delta` this test used to pin, which `#97`
    /// kept as a knowingly-deferred phantom ("the #94 family") and which
    /// `label_padding` could even drive NEGATIVE. The gate now sits ahead of the
    /// whole cascade, so it covers the rotated branches the earlier flat/wrapped
    /// gate left reserving a full rotated extent for invisible text.
    #[test]
    fn x_axis_label_band_labels_off_on_shown_axis_reserves_nothing() {
        let m = mock(10.0);
        let line_h = m.line_height(11.0);
        let tick_size = 4.0;

        // Flat case (same fixture as `estimate_flat_labels`).
        let flat_labels: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let flat_off = estimate_band(
            &flat_labels,
            None,
            100.0,
            None,
            false,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        assert_eq!(flat_off, 0.0, "labels-off flat band reserves nothing");
        // Contrast: labels drawn DOES reserve the #97 standoff for the same input.
        let flat_on = estimate_band(
            &flat_labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        assert!(
            (flat_on - (tick_size + 2.0 + line_h)).abs() < 1e-9,
            "labels-on flat band should still be tick_size + label_pad_eff + line_h"
        );

        // No `label_padding` can move it: the pre-fix value was `line_h +
        // padding_delta`, i.e. a phantom band that a large negative padding
        // could even make negative.
        for padding in [Some(40.0), Some(-100.0)] {
            assert_eq!(
                estimate_band(
                    &flat_labels,
                    None,
                    100.0,
                    padding,
                    false,
                    &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m)
                ),
                0.0,
                "labels-off band must be zero for label_padding={padding:?}"
            );
        }

        // Rotated: an explicit angle override used to bypass the gate entirely
        // and reserve the full `rotated_x_label_extent`.
        assert_eq!(
            estimate_band(
                &flat_labels,
                Some(-45.0),
                100.0,
                None,
                false,
                &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m)
            ),
            0.0,
            "#182 widened the gate past flat/wrapped: a rotated labels-off axis reserves nothing"
        );
        assert!(
            estimate_band(
                &flat_labels,
                Some(-45.0),
                100.0,
                None,
                true,
                &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m)
            ) > 0.0,
            "the same rotated axis WITH labels drawn still reserves its extent"
        );

        // Wrapped case (same fixture as `estimate_wrapped_labels`).
        let wrapped_labels: Vec<String> = vec![
            "trivial_baseline".into(),
            "negative_prompt".into(),
            "persona_limited".into(),
            "minimal_context".into(),
        ];
        let m6 = mock(6.0);
        let line_h6 = m6.line_height(11.0);
        let wrapped_off = estimate_band(
            &wrapped_labels,
            None,
            100.0,
            None,
            false,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m6),
        );
        assert_eq!(wrapped_off, 0.0, "labels-off wrapped band reserves nothing");
        let wrapped_on = estimate_band(
            &wrapped_labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m6),
        );
        assert!(
            (wrapped_on - (tick_size + 2.0 + 2.0 * line_h6)).abs() < 1e-9,
            "labels-on wrapped band should still be tick_size + label_pad_eff + 2*line_h"
        );
    }

    #[test]
    fn estimate_rotated_labels() {
        // Labels with no break points that collide flat and can't wrap, but fit at -60°
        // once the real footprint formula is used (spec §4.6 / F-L07-04 fix).
        // "ABCDEFGHIJ" = 10 * 10 = 100px. estimated_slot_w = 80. line_h = 11*1.2 = 13.2.
        // threshold = 80 * 0.9 = 72. 100 > 72 -> collision.
        // S1: no break points -> wrap fails for all.
        // S2/S3 (real footprint = cos*w + sin*line_h):
        //   -30: cos(30)*100 + sin(30)*13.2 = 86.60 + 6.6 = 93.20 > 80 -> fail.
        //   -45: cos(45)*100 + sin(45)*13.2 = 70.71 + 9.34 = 80.05 > 80 -> fail
        //        (the pre-fix bare-cos check alone, 70.7<=80, used to pass here).
        //   -60: cos(60)*100 + sin(60)*13.2 = 50.00 + 11.43 = 61.43 <= 80 -> pass!
        // Expected margin = 100 * sin(60°) + line_h * cos(60°).
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "UVWXYZABCD".into(),
            "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let line_h = m.line_height(11.0); // 13.2
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            None,
            80.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        // Full geometric extent (mirrors the render pivot): the old too-tight
        // formula was `sin·max_w + cos·line_h`; the band now adds the pivot
        // offset `tick_size + label_pad + sin·font_size` on top of it.
        let angle_rad = (-60.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();
        let label_pad = 2.0; // default
        let expected = tick_size + label_pad + sin_abs * 11.0 + sin_abs * 100.0 + cos_abs * line_h;
        assert!(
            (band - expected).abs() < 1e-6,
            "rotated -60° band should be {expected}, got {band}"
        );
    }

    #[test]
    fn estimate_override_angle_minus_90() {
        // label_angle_override = -90 → sin(90°)=1, cos(90°)=0 →
        // margin = max_label_w * 1 + line_h * 0 = max_label_w.
        // "ABCDEFGHIJ" = 10 * 10 = 100px.
        let labels: Vec<String> = vec!["ABCDEFGHIJ".into(), "KLMNOPQRST".into()];
        let m = mock(10.0);
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            Some(-90.0),
            80.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        // At -90°: sin=1, cos=0 → full vertical extent =
        // tick_size + label_pad + font_size + max_label_w.
        let label_pad = 2.0; // default
        let expected = tick_size + label_pad + 11.0 + 100.0;
        assert!(
            (band - expected).abs() < 1e-6,
            "override -90° band should be ~{expected}, got {band}"
        );
    }

    #[test]
    fn estimate_override_angle_minus_45() {
        // label_angle_override = -45 → margin = max_w * sin(45°) + line_h * cos(45°).
        let labels: Vec<String> = vec!["ABCDEFGHIJ".into()];
        let m = mock(10.0);
        let line_h = m.line_height(11.0);
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            Some(-45.0),
            200.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        // Full geometric extent at -45° (mirrors the render pivot).
        let angle_rad = (-45.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();
        let label_pad = 2.0; // default
        let expected = tick_size + label_pad + sin_abs * 11.0 + sin_abs * 100.0 + cos_abs * line_h;
        assert!(
            (band - expected).abs() < 1e-6,
            "override -45° band should be {expected}, got {band}"
        );
    }

    #[test]
    fn estimate_empty_labels_returns_line_height() {
        let m = mock(10.0);
        let line_h = m.line_height(11.0);
        let band = estimate_band(
            &[],
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        );
        assert!(
            (band - line_h).abs() < 1e-9,
            "empty labels should return line_height={line_h}, got {band}"
        );
    }

    #[test]
    fn estimate_fallback_for_extreme_widths() {
        // Labels so wide that even -90° doesn't help — fallback to max_label_w + 2.
        // Use a mock that returns 1e18 for all labels so even cos(-90)*1e18 > slot_w.
        let labels: Vec<String> = vec!["X".into(), "Y".into()];
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            None,
            10.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        // Vertical fallback: tick_size + label_pad + font_size + max_label_w.
        // At 1e18 the additive terms vanish into float epsilon, so the band is
        // dominated by max_label_w (1e18). Compare with a generous tolerance.
        let label_pad = 2.0;
        let expected = tick_size + label_pad + 11.0 + 1e18;
        assert!(
            (band - expected).abs() < 1.0,
            "fallback path should return tick_size + label_pad + font_size + max_label_w, got {band}"
        );
    }

    // --- estimate_x_label_band: x-title overlap fix (rotated band reservation) ---
    //
    // These guard the fix where the reserved x-axis label band under-counted the
    // true vertical extent of rotated tick labels, letting the x-axis title ride
    // into the longest labels. The rotated branches must now reserve the full
    // geometric extent that `render::marks::axis::build_axis` draws.

    #[test]
    fn estimate_rotated_band_grew_by_pivot_offset() {
        // Labels that collide flat and resolve at -60° (same setup as
        // `estimate_rotated_labels`; spec §4.6 / F-L07-04 fix moved the
        // resolved angle from -45° to -60° once the real footprint formula
        // is used). The NEW band must exceed the OLD too-tight formula
        // (`sin·max_label_w + cos·line_h`) by exactly the pivot offset
        // `tick_size + label_pad + sin·font_size`. Both sides computed explicitly.
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "UVWXYZABCD".into(),
            "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let font_size = 11.0;
        let tick_size = 4.0;
        let line_h = m.line_height(font_size);
        let max_label_w = 100.0; // 10 chars * 10px

        let band = estimate_band(
            &labels,
            None,
            80.0,
            None,
            true,
            &LabelBandCtx::new(font_size, legacy_gap(0), tick_size, &m),
        );

        // Cascade resolves at -60° here.
        let angle_rad = (-60.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();

        let old_formula = sin_abs * max_label_w + cos_abs * line_h;
        let label_pad = 2.0; // default
        let pivot_offset = tick_size + label_pad + sin_abs * font_size;

        assert!(
            (band - (old_formula + pivot_offset)).abs() < 1e-6,
            "new band ({band}) must equal old formula ({old_formula}) + pivot offset ({pivot_offset})",
        );
    }

    #[test]
    fn estimate_vertical_fallback_full_extent() {
        // The -90 / S4-S5 vertical path returns the full vertical extent at
        // sin=1, cos=0: tick_size + label_pad + font_size + max_label_w.
        // Force the fallback with labels too wide for any cascade angle.
        let labels: Vec<String> = vec!["X".into(), "Y".into()];
        let m = MockMetrics {
            measure: |_, _| 5_000.0,
            line_h_factor: 1.2,
        };
        let font_size = 11.0;
        let tick_size = 4.0;
        let band = estimate_band(
            &labels,
            None,
            10.0,
            None,
            true,
            &LabelBandCtx::new(font_size, legacy_gap(0), tick_size, &m),
        );
        let label_pad = 2.0; // default
        let expected = tick_size + label_pad + font_size + 5_000.0;
        assert!(
            (band - expected).abs() < 1e-6,
            "vertical fallback should return {expected}, got {band}",
        );
    }

    #[test]
    fn estimate_flat_band_scales_linearly_with_tick_size() {
        // #97 (spec §4.1): the flat band now honors tick_size (it is the SAME
        // standoff the renderer draws), so two different tick_size values must
        // differ by exactly the tick_size delta — the flat band is no longer
        // insensitive to tick_size.
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let m = mock(10.0);
        let line_h = m.line_height(11.0);
        let band_ts0 = estimate_band(
            &labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 0.0, &m),
        );
        let band_ts8 = estimate_band(
            &labels,
            None,
            100.0,
            None,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 8.0, &m),
        );
        let expected_ts0 = 0.0 + 2.0 + line_h;
        assert!(
            (band_ts0 - expected_ts0).abs() < 1e-9,
            "flat band at tick_size=0 should be {expected_ts0}, got {band_ts0}",
        );
        assert!(
            (band_ts8 - band_ts0 - 8.0).abs() < 1e-9,
            "flat band must scale linearly with tick_size: band(8)-band(0) should be 8.0, \
             got {} (band_ts0={band_ts0}, band_ts8={band_ts8})",
            band_ts8 - band_ts0,
        );
    }

    #[test]
    fn estimate_label_padding_widens_rotated_band() {
        // Increasing label_padding must widen the rotated band by the same delta.
        // Rotated branch folds `label_pad_eff` directly into the pivot offset, so
        // band(lp = a + Δ) - band(lp = a) == Δ.
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "UVWXYZABCD".into(),
            "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let tick_size = 4.0;
        let band_default = estimate_band(
            &labels,
            None,
            80.0,
            Some(2.0),
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        let band_wider = estimate_band(
            &labels,
            None,
            80.0,
            Some(12.0),
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), tick_size, &m),
        );
        let delta = 12.0 - 2.0;
        assert!(
            (band_wider - band_default - delta).abs() < 1e-6,
            "label_padding +{delta} must widen rotated band by {delta}; \
             got {band_default} -> {band_wider}",
        );
    }

    #[test]
    fn estimate_rotated_band_covers_true_label_extent() {
        // Integration-style guard without wiring a full compute_layout: the
        // returned band must be >= the analytically computed true label extent
        // that the render draws below the axis line, namely
        //   tick_size + label_pad + sin·(font_size + max_label_w) + cos·descent
        // for a representative resolved angle. Because the band uses line_h (>=
        // descent) for the cos term, it clears the labels with a small margin,
        // so the x-axis title placed just below the band cannot overlap them.
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "UVWXYZABCD".into(),
            "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let font_size = 11.0;
        let tick_size = 4.0;
        let max_label_w = 100.0;
        // This setup resolves at -45° in the cascade.
        let angle_rad = (-45.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();
        // Conservative descent estimate (well under line_h ~= 13.2).
        let descent = font_size * 0.3;
        let label_pad = 2.0; // default
        let true_extent =
            tick_size + label_pad + sin_abs * (font_size + max_label_w) + cos_abs * descent;

        let band = estimate_band(
            &labels,
            None,
            80.0,
            None,
            true,
            &LabelBandCtx::new(font_size, legacy_gap(0), tick_size, &m),
        );
        assert!(
            band >= true_extent,
            "band ({band}) must cover the true label extent ({true_extent}) so the \
             x-axis title placed below the band clears the rotated labels",
        );
    }

    // --- rotated_x_label_extent helper + x-title placement tests ---
    //
    // These guard the second half of the overlap fix: the x-axis title's
    // `anchor_y` is now derived from `rotated_x_label_extent` (the same helper the
    // band uses) when labels rotate, so the title drops below the true rotated
    // extent instead of a flat line height. Flat labels keep the old formula.

    #[test]
    fn rotated_x_label_extent_hand_computed_values() {
        // -45°: sin=cos=√2/2≈0.70710678. With font_size=11, max_w=100,
        // line_h=13.2, tick_size=4, label_pad=2:
        //   4 + 2 + 0.7071·11 + 0.7071·100 + 0.7071·13.2
        let sin45 = (-45.0_f64).to_radians().sin().abs();
        let cos45 = (-45.0_f64).to_radians().cos().abs();
        let expected_45 = 4.0 + 2.0 + sin45 * 11.0 + sin45 * 100.0 + cos45 * 13.2;
        let got_45 = rotated_x_label_extent(-45.0, 100.0, 11.0, 13.2, 4.0, 2.0);
        assert!(
            (got_45 - expected_45).abs() < 1e-9,
            "-45° extent should be {expected_45}, got {got_45}",
        );

        // -90°: sin=1, cos≈0 → tick_size + label_pad + font_size + max_w
        // (the cos·line_h term is a sub-femtopixel epsilon, well under 1e-6).
        let got_90 = rotated_x_label_extent(-90.0, 100.0, 11.0, 13.2, 4.0, 2.0);
        let expected_90 = 4.0 + 2.0 + 11.0 + 100.0;
        assert!(
            (got_90 - expected_90).abs() < 1e-6,
            "-90° extent should be ~{expected_90}, got {got_90}",
        );
    }

    #[test]
    fn x_axis_title_clears_rotated_labels() {
        // Long labels in a narrow panel force rotation; with a title present the
        // title `anchor_y` must sit at or below the full rotated-label extent so
        // it cannot overlap the longest label.
        let input = axis_input(
            AxisOrient::Bottom,
            Some("Feature".into()),
            // No underscores/spaces/camelCase boundaries → wrap is impossible, so
            // the cascade resolves by rotation rather than wrapping.
            (0..6).map(|i| format!("aaaaaaaaaaaaaaaaa{i}")).collect(),
            None,
        );
        // 18-char labels at 10px/char = 180px in a 300px panel (slot 50) force the
        // cascade well past flat into rotation.
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let label_font_size = 11.0;
        let title_font_size = 13.0;
        let title_padding = 4.0;
        let tick_size = 4.0;
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            title_font_size,
            title_padding,
            &LabelBandCtx::new(label_font_size, legacy_gap(8), tick_size, &m),
        );

        // The labels must have rotated (non-zero angle) for this test to be meaningful.
        let resolved_angle = axis
            .ticks
            .iter()
            .find(|t| !t.culled)
            .map(|t| t.label_angle)
            .unwrap();
        assert!(
            resolved_angle != 0.0,
            "labels should rotate; got angle {resolved_angle}"
        );

        // Recompute the expected rotated extent from the FINAL non-culled labels.
        let max_label_w = axis
            .ticks
            .iter()
            .filter(|t| !t.culled)
            .map(|t| m.measure_width(&t.label, label_font_size))
            .fold(0.0_f64, f64::max);
        let line_h = m.line_height(label_font_size);
        let extent = rotated_x_label_extent(
            resolved_angle,
            max_label_w,
            label_font_size,
            line_h,
            tick_size,
            2.0,
        );

        let title = axis.title.expect("title present");
        let min_anchor_y = panel_area.y + panel_area.h + extent;
        assert!(
            title.anchor_y >= min_anchor_y,
            "title anchor_y ({}) must be >= panel bottom + rotated extent ({min_anchor_y})",
            title.anchor_y,
        );
        // Exact placement: extent + title_padding + title_h/2.
        let title_h = m.line_height(title_font_size);
        let expected = panel_area.y + panel_area.h + extent + title_padding + title_h / 2.0;
        assert!(
            (title.anchor_y - expected).abs() < 1e-9,
            "title anchor_y ({}) should equal extent-based formula ({expected})",
            title.anchor_y,
        );
    }

    #[test]
    fn x_axis_title_flat_matches_unified_band_formula() {
        // #97 (spec §4.1): flat labels (angle 0) with a title — the title
        // anchor_y now goes through the SAME `rotated_x_label_extent` call as
        // rotated labels (continuous at θ=0), so it includes the
        // tick_size + label_pad_eff standoff instead of a bare line_height.
        let input = axis_input(
            AxisOrient::Bottom,
            Some("Price".into()),
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 20.0,
            line_h_factor: 1.2,
        };
        let label_font_size = 11.0;
        let title_font_size = 13.0;
        let title_padding = 4.0;
        let tick_size = 4.0;
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            title_font_size,
            title_padding,
            &LabelBandCtx::new(label_font_size, legacy_gap(8), tick_size, &m),
        );
        // All flat.
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
        }
        let title = axis.title.expect("title present");
        let line_h = m.line_height(label_font_size);
        let title_h = m.line_height(title_font_size);
        // rotated_x_label_extent(0.0, ..) collapses to tick_size + label_pad_eff + line_h.
        let label_extent = tick_size + 2.0 + line_h;
        let expected = panel_area.y + panel_area.h + label_extent + title_padding + title_h / 2.0;
        assert!(
            (title.anchor_y - expected).abs() < 1e-12,
            "flat title anchor_y ({}) must equal the unified formula ({expected})",
            title.anchor_y,
        );
    }

    /// #97 spec §4.1 x-side extension (issue #97 gap): a titled x
    /// axis with `fm.Axis(labels=False)` (`AxisInput.show_labels = false`)
    /// must place its title at the SAME `anchor_y` the pre-#97 flat-only
    /// special case gave (`git show HEAD`: bare `line_h`, no
    /// `tick_size`/`label_pad_eff` standoff), matching the gated
    /// `estimate_x_label_band` reservation (`line_h + padding_delta`) for
    /// the same axis, so band and title cannot drift. RED-proven: with the
    /// θ=0/`show_labels` gate removed, the title anchor is `tick_size +
    /// label_pad_eff = 6.0` px further out than the gated band reserves,
    /// same fixture as `x_axis_title_flat_matches_unified_band_formula`.
    #[test]
    fn x_axis_title_labels_off_matches_pre_97_flat_value() {
        let mut input = axis_input(
            AxisOrient::Bottom,
            Some("Price".into()),
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        input.overrides.labels.show = Some(false);
        let panel_area = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 20.0,
            line_h_factor: 1.2,
        };
        let label_font_size = 11.0;
        let title_font_size = 13.0;
        let title_padding = 4.0;
        let tick_size = 4.0;
        let (axis, _) = layout_x_axis(
            &input,
            panel_area,
            0,
            title_font_size,
            title_padding,
            &LabelBandCtx::new(label_font_size, legacy_gap(8), tick_size, &m),
        );
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
        }
        let title = axis.title.expect("title present");
        let line_h = m.line_height(label_font_size);
        let title_h = m.line_height(title_font_size);
        // Pre-#97 flat-only special case: bare line_h, NOT
        // tick_size + label_pad_eff + line_h. This is the value the
        // reserved band also collapses to for a labels-off axis
        // (`estimate_x_label_band`'s S0 gate returns `line_h +
        // padding_delta`; with no override `label_padding`, `padding_delta
        // == 0.0`, so the two match exactly here).
        let label_extent = line_h;
        let expected = panel_area.y + panel_area.h + label_extent + title_padding + title_h / 2.0;
        assert!(
            (title.anchor_y - expected).abs() < 1e-12,
            "labels-off flat title anchor_y ({}) must equal the pre-#97 bare line_h formula ({expected})",
            title.anchor_y,
        );

        // Contrast: the SAME input with labels drawn gets the #97 standoff
        // and sits strictly further from the axis line.
        let mut labels_on = input.clone();
        labels_on.overrides.labels.show = Some(true);
        let (axis_on, _) = layout_x_axis(
            &labels_on,
            panel_area,
            0,
            title_font_size,
            title_padding,
            &LabelBandCtx::new(label_font_size, legacy_gap(8), tick_size, &m),
        );
        let title_on = axis_on.title.expect("title present");
        assert!(
            title_on.anchor_y > title.anchor_y,
            "labels-drawn axis's title must sit further out than a labels-off axis's"
        );
    }

    // --- continuous-axis scale-projection tests (2026-05-30) ---

    /// Build an AxisInput carrying projected tick fractions (continuous axis).
    fn axis_input_projected(
        orient: AxisOrient,
        labels: Vec<String>,
        fractions: Vec<f64>,
        padding_frac: f64,
    ) -> AxisInput {
        let mut input = axis_input(orient, None, labels, None);
        input.tick_projection = Some(TickProjection {
            padding_frac,
            major: fractions,
            minor: Vec::new(),
            text_demand: Default::default(),
        });
        input
    }

    #[test]
    fn x_axis_continuous_places_ticks_at_projected_pixels_not_slots() {
        // 3 ticks (lo/mid/hi) with padding_frac=0.05 in a 0..600 panel. The cap
        // binds (600*0.05=30 > 8), so the inset is 8px → range (8, 592). Ticks at
        // fractions 0.0/0.5/1.0 land at 8 / 300 / 592 — NOT the uniform-slot
        // centers (100 / 300 / 500) for n=3.
        let input = axis_input_projected(
            AxisOrient::Bottom,
            vec!["0".into(), "50".into(), "100".into()],
            vec![0.0, 0.5, 1.0],
            0.05,
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert_eq!(axis.ticks.len(), 3);
        assert!(
            (axis.ticks[0].position - 8.0).abs() < 1e-9,
            "got {}",
            axis.ticks[0].position
        );
        assert!(
            (axis.ticks[1].position - 300.0).abs() < 1e-9,
            "got {}",
            axis.ticks[1].position
        );
        assert!(
            (axis.ticks[2].position - 592.0).abs() < 1e-9,
            "got {}",
            axis.ticks[2].position
        );
        // Not at the n=3 uniform slot centers (100/300/500).
        assert!((axis.ticks[0].position - 100.0).abs() > 1.0);
        assert!((axis.ticks[2].position - 500.0).abs() > 1.0);
    }

    #[test]
    fn x_axis_continuous_panel_origin_offsets_projection() {
        // Same as above but a non-zero panel origin; inset is applied to the
        // panel's own (x, x+w) range.
        let input = axis_input_projected(
            AxisOrient::Bottom,
            vec!["0".into(), "100".into()],
            vec![0.0, 1.0],
            0.05,
        );
        let panel = Rect {
            x: 100.0,
            y: 0.0,
            w: 600.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert!((axis.ticks[0].position - 108.0).abs() < 1e-9);
        assert!((axis.ticks[1].position - 692.0).abs() < 1e-9);
    }

    #[test]
    fn x_axis_categorical_keeps_uniform_slot_centers() {
        // No projected fractions → byte-identical uniform-slot placement.
        // This reproduces the pre-change positions for n=4 in a 100..500 panel.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        assert!(input.tick_projection.is_none());
        let panel = Rect {
            x: 100.0,
            y: 50.0,
            w: 400.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert!((axis.ticks[0].position - 150.0).abs() < 1e-9);
        assert!((axis.ticks[1].position - 250.0).abs() < 1e-9);
        assert!((axis.ticks[2].position - 350.0).abs() < 1e-9);
        assert!((axis.ticks[3].position - 450.0).abs() < 1e-9);
    }

    #[test]
    fn y_axis_continuous_places_ticks_at_projected_pixels() {
        // y mark range is the inverted (bottom, top) = (panel.y+h, panel.y),
        // inset by the cap → (panel.y+h-8, panel.y+8). Fraction 0.0 → bottom,
        // 1.0 → top. With reversed labels (high first) the carrier holds
        // [1.0, 0.5, 0.0] so label[0]=high sits at the top.
        let input = axis_input_projected(
            AxisOrient::Left,
            vec!["100".into(), "50".into(), "0".into()],
            vec![1.0, 0.5, 0.0],
            0.05,
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 600.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel, 0, 11.0, 13.0, 4.0, 4.0, &m);
        // inset range (592, 8): t=1 → 8 (top), t=0.5 → 300, t=0 → 592 (bottom).
        assert!(
            (axis.ticks[0].position - 8.0).abs() < 1e-9,
            "top tick; got {}",
            axis.ticks[0].position
        );
        assert!(
            (axis.ticks[1].position - 300.0).abs() < 1e-9,
            "got {}",
            axis.ticks[1].position
        );
        assert!(
            (axis.ticks[2].position - 592.0).abs() < 1e-9,
            "bottom tick; got {}",
            axis.ticks[2].position
        );
        // Not the n=3 uniform slot centers (100/300/500).
        assert!((axis.ticks[0].position - 100.0).abs() > 1.0);
    }

    #[test]
    fn y_axis_categorical_keeps_uniform_slot_centers() {
        // No projected fractions → uniform-slot placement (pre-change positions).
        let input = axis_input(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            None,
        );
        assert!(input.tick_projection.is_none());
        let panel = Rect {
            x: 100.0,
            y: 50.0,
            w: 300.0,
            h: 200.0,
        };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel, 0, 11.0, 13.0, 4.0, 4.0, &m);
        assert!((axis.ticks[0].position - 75.0).abs() < 1e-9);
        assert!((axis.ticks[3].position - 225.0).abs() < 1e-9);
    }

    #[test]
    fn cascade_uses_min_gap_on_nonuniform_continuous_spacing() {
        // Simulate a log axis: ticks bunch toward the right so the min adjacent
        // gap is far smaller than the average slot. 6 labels of width 70px in a
        // 600px panel. The uniform slot would be 100px (labels fit flat), but the
        // tightest projected gap is ~21px — labels must NOT stay flat; the
        // cascade rotates because the real min gap is too small.
        // Fractions chosen so consecutive pixel gaps shrink: cumulative spread
        // with the last pair ~21px apart after the 8px-capped inset on 600px.
        let fractions = vec![0.0, 0.45, 0.72, 0.88, 0.965, 1.0];
        let input = axis_input_projected(
            AxisOrient::Bottom,
            (0..6).map(|i| format!("L{i}")).collect(),
            fractions,
            0.05,
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 200.0,
        };
        // Each label is 70px wide. Uniform slot=100 → would stay flat (70<90).
        let m = MockMetrics {
            measure: |_, _| 70.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        // The tightest gap is between fractions 0.965 and 1.0 over inset span 584:
        // 0.035 * 584 ≈ 20.4px. 70px labels cannot stay flat at that gap, so the
        // cascade must rotate (non-zero angle) — proving it used the real min gap,
        // not the uniform slot (which would have kept them flat at angle 0).
        assert!(
            axis.ticks.iter().all(|t| t.label_angle != 0.0),
            "cascade should rotate when the real min gap is too tight; \
             angles={:?}",
            axis.ticks.iter().map(|t| t.label_angle).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cascade_uniform_slot_would_keep_flat_baseline() {
        // Control for the previous test: with the SAME labels and panel but NO
        // projected fractions (categorical), the uniform slot (100px) easily fits
        // 70px labels, so the cascade stays flat. This pins that the rotation in
        // the continuous case is caused by the min-gap logic, not the labels.
        let input = axis_input(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("L{i}")).collect(),
            None,
        );
        let panel = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 200.0,
        };
        let m = MockMetrics {
            measure: |_, _| 70.0,
            line_h_factor: 1.2,
        };
        let (axis, _) = layout_x_axis(
            &input,
            panel,
            0,
            13.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert!(axis.ticks.iter().all(|t| t.label_angle == 0.0));
    }

    // --- label_padding clamp invariant (band >= render extent) ---

    /// Helper: compute `estimate_x_label_band` for a set of wide labels that
    /// force rotation, with the given `label_padding`.
    fn rotated_band_with_padding(label_padding: Option<f64>) -> f64 {
        // 6 labels of 80px each in a 240px panel.
        // slot_w = 40, threshold = 36.  80 > 36 → S0/S1 fail.
        // S3: cos(-30)*80 ≈ 69.3 > 40, cos(-45)*80 ≈ 56.6 > 40,
        //     cos(-60)*80 = 40.0 ≤ 40 → passes at -60.
        let labels: Vec<String> = (0..6).map(|i| format!("L{i}")).collect();
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        estimate_band(
            &labels,
            None,
            40.0,
            label_padding,
            true,
            &LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m),
        )
    }

    #[test]
    fn negative_label_padding_rotated_band_no_less_than_zero_padding() {
        // Invariant: band(negative_pad) >= band(zero_pad).
        // The renderer clamps label_padding to 0 for negative values; the layout
        // must do the same so the reserved band cannot fall below the actual
        // render extent (which would cause title-vs-label overlap).
        let band_zero = rotated_band_with_padding(Some(0.0));
        let band_neg = rotated_band_with_padding(Some(-10.0));
        assert!(
            band_neg >= band_zero,
            "negative label_padding must not shrink the rotated band below \
             the band computed with label_padding=0: band(-10)={band_neg} < band(0)={band_zero}"
        );
    }

    // --- label_overlap override (B5 unit 6b) ------------------------------------

    /// Six dense colliding labels (each 80px) in a 40px slot — the default
    /// cascade would cull/rotate. Used to exercise the overlap overrides.
    fn dense_labels() -> Vec<String> {
        (0..6).map(|i| format!("L{i}")).collect()
    }

    #[test]
    fn label_overlap_show_all_keeps_every_label_visible() {
        // ShowAll short-circuits to flat with all ticks visible — labels may
        // overlap, but none are culled or elided.
        let labels = dense_labels();
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(
            &labels,
            40.0,
            11.0,
            legacy_gap(8),
            Some(LabelOverlap::ShowAll),
            &m,
            scale_less_frame(),
        );
        assert_eq!(result.strategy, CascadeStrategy::Flat);
        assert_eq!(result.angle, 0.0);
        assert!(
            result.visible.iter().all(|v| *v),
            "ShowAll must keep all labels visible"
        );
        assert_eq!(result.labels, labels, "ShowAll must not elide any label");
    }

    #[test]
    fn label_overlap_parity_shows_every_other_label() {
        // Parity short-circuits to a stride-2 cull regardless of width.
        let labels = dense_labels();
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(
            &labels,
            40.0,
            11.0,
            legacy_gap(8),
            Some(LabelOverlap::Parity),
            &m,
            scale_less_frame(),
        );
        assert_eq!(result.strategy, CascadeStrategy::Culled { stride: 2 });
        let visible: Vec<bool> = result.visible;
        assert_eq!(visible, vec![true, false, true, false, true, false]);
        let shown = visible.iter().filter(|v| **v).count();
        assert_eq!(shown, 3, "parity on 6 labels shows 3");
    }

    #[test]
    fn label_overlap_rotate_forces_steepest_angle_all_visible() {
        // Rotate short-circuits to the steepest cascade angle with all visible.
        let labels = dense_labels();
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(
            &labels,
            40.0,
            11.0,
            legacy_gap(8),
            Some(LabelOverlap::Rotate),
            &m,
            scale_less_frame(),
        );
        let steepest = *ANGLE_CASCADE.last().unwrap();
        assert_eq!(
            result.strategy,
            CascadeStrategy::Rotated { angle: steepest }
        );
        assert_eq!(result.angle, steepest);
        assert!(result.visible.iter().all(|v| *v), "rotate keeps all labels");
    }

    #[test]
    fn label_overlap_greedy_matches_default_cascade() {
        // Greedy must run the unmodified cascade — identical to `None`.
        let labels = dense_labels();
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let greedy = cascade_collision_recovery(
            &labels,
            40.0,
            11.0,
            legacy_gap(8),
            Some(LabelOverlap::Greedy),
            &m,
            scale_less_frame(),
        );
        let default = cascade_collision_recovery(&labels, 40.0, 11.0, legacy_gap(8), None, &m, scale_less_frame());
        assert_eq!(greedy.strategy, default.strategy);
        assert_eq!(greedy.angle, default.angle);
        assert_eq!(greedy.visible, default.visible);
        assert_eq!(greedy.labels, default.labels);
    }

    /// A zero-height slot no longer makes the Greedy arm drop the first
    /// label: positions never multiply by a slot height, so the first
    /// label's "clears the last kept one" test is `last_kept.is_none()`,
    /// true regardless of position — three equal (zero-height-slot)
    /// positions still keep the first.
    #[test]
    fn resolve_y_overlap_greedy_zero_slot_height_keeps_first_label() {
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let m = MockMetrics {
            measure: |_, _| 10.0,
            line_h_factor: 1.2,
        };
        let positions = [0.0, 0.0, 0.0];
        let visible = resolve_y_overlap(Some(LabelOverlap::Greedy), &labels, &positions, 11.0, &m, &[])
            .expect("Greedy always returns Some");
        assert!(visible[0], "the first label must always be kept");
    }

    /// Anchors must be kept greedy among THEMSELVES, then each non-anchor
    /// kept only when it clears both its nearest kept neighbours. A naive
    /// single-pass cascade that keeps every anchor unconditionally and
    /// tests a non-anchor only against the previous KEPT label lets a
    /// non-anchor just before an anchor overlap it.
    #[test]
    fn resolve_y_overlap_greedy_keeps_decades_greedily_then_clears_both_neighbours() {
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let l = m.line_height(11.0);
        let positions = [0.0, 0.5 * l, 1.0 * l, 1.6 * l, 2.1 * l, 2.6 * l, 3.2 * l];
        let anchors = [true, false, false, true, false, false, true];
        let labels: Vec<String> = (0..7).map(|i| i.to_string()).collect();
        let visible = resolve_y_overlap(Some(LabelOverlap::Greedy), &labels, &positions, 11.0, &m, &anchors)
            .expect("Greedy always returns Some");
        assert_eq!(visible, vec![true, false, false, true, false, false, true]);
    }

    /// Anchors greedy among themselves: a decade too close to the last KEPT
    /// decade is culled, even though it is itself an anchor.
    #[test]
    fn resolve_y_overlap_greedy_culls_a_clashing_decade() {
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let l = m.line_height(11.0);
        let positions = [0.0, 0.5 * l, 1.2 * l];
        let anchors = [true, true, true];
        let labels: Vec<String> = (0..3).map(|i| i.to_string()).collect();
        let visible = resolve_y_overlap(Some(LabelOverlap::Greedy), &labels, &positions, 11.0, &m, &anchors)
            .expect("Greedy always returns Some");
        assert_eq!(visible, vec![true, false, true]);
    }

    /// A non-decade that clears both of its neighbouring decades is kept
    /// (guard: this shape was already correct before and after the fix).
    #[test]
    fn resolve_y_overlap_greedy_keeps_non_decades_that_clear_both_decades() {
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let l = m.line_height(11.0);
        let positions = [0.0, 1.2 * l, 2.4 * l, 3.6 * l];
        let anchors = [true, false, false, true];
        let labels: Vec<String> = (0..4).map(|i| i.to_string()).collect();
        let visible = resolve_y_overlap(Some(LabelOverlap::Greedy), &labels, &positions, 11.0, &m, &anchors)
            .expect("Greedy always returns Some");
        assert_eq!(visible, vec![true, true, true, true]);
    }

    /// With no anchors, the two-pass rule reduces to the classic single-
    /// pass greedy over positions (guard: byte-identical to the pre-fix
    /// slot-count greedy on a uniform axis).
    #[test]
    fn resolve_y_overlap_greedy_without_anchors_matches_the_slot_greedy() {
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let l = m.line_height(11.0);
        let positions = [0.0, 0.6 * l, 1.2 * l, 1.8 * l, 2.4 * l];
        let labels: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        let visible = resolve_y_overlap(Some(LabelOverlap::Greedy), &labels, &positions, 11.0, &m, &[])
            .expect("Greedy always returns Some");
        assert_eq!(visible, vec![true, false, true, false, true]);
    }

    #[test]
    fn layout_x_axis_parity_culls_alternating_ticks() {
        // End-to-end through layout_x_axis: a parity override on a dense axis
        // marks alternating ticks culled.
        let mut input = axis_input(AxisOrient::Bottom, None, dense_labels(), None);
        input.overrides.labels.overlap = Some(LabelOverlap::Parity);
        let panel = Rect {
            x: 0.0,
            y: 100.0,
            w: 240.0,
            h: 0.0,
        };
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let (axis, _warn) = layout_x_axis(
            &input,
            panel,
            0,
            11.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        let culled: Vec<bool> = axis.ticks.iter().map(|t| t.culled).collect();
        assert_eq!(culled, vec![false, true, false, true, false, true]);
    }

    #[test]
    fn layout_x_axis_show_all_culls_nothing_on_dense_axis() {
        // A dense axis that the default cascade would cull renders every label
        // when label_overlap = ShowAll.
        let mut input = axis_input(AxisOrient::Bottom, None, dense_labels(), None);
        input.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        let panel = Rect {
            x: 0.0,
            y: 100.0,
            w: 240.0,
            h: 0.0,
        };
        let m = MockMetrics {
            measure: |_, _| 80.0,
            line_h_factor: 1.2,
        };
        let (axis, _warn) = layout_x_axis(
            &input,
            panel,
            0,
            11.0,
            4.0,
            &LabelBandCtx::new(11.0, legacy_gap(8), 4.0, &m),
        );
        assert!(
            axis.ticks.iter().all(|t| !t.culled),
            "ShowAll must cull nothing"
        );
        assert!(
            axis.ticks.iter().all(|t| t.label_angle == 0.0),
            "ShowAll stays flat"
        );
    }

    // ── compute_x_title_width per-axis title_font_size/title_padding override
    // (ported from tests/bug_hunt_scale_layout.rs, R1) ───────────────────────
    //
    // `compute_x_title_width` and `compute_y_title_width` had zero direct
    // coverage before this port — every assertion below calls the real
    // functions instead of the mirror's inlined `fs * 1.2 + padding` formula.

    fn titled_input(title_font_size: Option<f64>, title_padding: Option<f64>) -> AxisInput {
        let mut input = axis_input(
            AxisOrient::Bottom,
            Some("Title".into()),
            vec!["a".into()],
            None,
        );
        input.overrides.title.font_size = title_font_size;
        input.overrides.title.padding = title_padding;
        input
    }

    /// No per-axis override: `compute_x_title_width` uses the theme font size.
    /// With an override, it uses the per-axis font size instead, producing a
    /// larger gutter for a larger override.
    #[test]
    fn x_title_gutter_uses_per_axis_font_size() {
        let m = mock(0.0);
        let theme_fs = 13.0;
        let theme_pad = 8.0;

        let default_input = titled_input(None, None);
        let gutter_default = compute_x_title_width(&default_input, theme_fs, theme_pad, &m, true);
        assert!((gutter_default - 23.6).abs() < 1e-9, "got {gutter_default}");

        let override_input = titled_input(Some(30.0), None);
        let gutter_override = compute_x_title_width(&override_input, theme_fs, theme_pad, &m, true);
        assert!(
            (gutter_override - 44.0).abs() < 1e-9,
            "got {gutter_override}"
        );

        assert!(
            gutter_override > gutter_default,
            "per-axis override of 30 should produce a larger gutter than the theme default 13"
        );
    }

    /// Both `title_font_size` and `title_padding` overrides compose additively.
    #[test]
    fn x_title_gutter_uses_per_axis_font_size_and_padding() {
        let m = mock(0.0);
        let input = titled_input(Some(24.0), Some(16.0));
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        assert!(
            (gutter - 44.8).abs() < 1e-9,
            "24*1.2+16 = 44.8, got {gutter}"
        );
    }

    /// No title: `compute_x_title_width` returns 0.0 regardless of font size.
    #[test]
    fn x_title_gutter_no_title_is_zero() {
        let m = mock(0.0);
        let mut input = axis_input(AxisOrient::Bottom, None, vec!["a".into()], None);
        input.overrides.title.font_size = Some(100.0);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        assert_eq!(gutter, 0.0, "no title should mean zero gutter");
    }

    /// Zero per-axis font size: gutter reduces to the padding term alone.
    #[test]
    fn x_title_gutter_zero_font_size() {
        let m = mock(0.0);
        let input = titled_input(Some(0.0), None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        assert!(
            (gutter - 8.0).abs() < 1e-9,
            "zero font size means gutter = padding only, got {gutter}"
        );
    }

    /// Very large per-axis font size scales the gutter proportionally.
    #[test]
    fn x_title_gutter_very_large_font_size() {
        let m = mock(0.0);
        let input = titled_input(Some(1000.0), None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        assert!(
            (gutter - 1208.0).abs() < 1e-9,
            "huge font: gutter = 1000*1.2+8 = 1208, got {gutter}"
        );
    }

    /// A NaN per-axis font size propagates to a NaN gutter — `compute_x_title_width`
    /// applies no defensive `is_finite` guard, so NaN user input reaches layout
    /// arithmetic unchanged.
    #[test]
    fn x_title_gutter_nan_font_size_propagates() {
        let m = mock(0.0);
        let input = titled_input(Some(f64::NAN), None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        assert!(
            gutter.is_nan(),
            "NaN font size should propagate to a NaN gutter"
        );
    }

    /// An unset per-axis font size falls back to the theme font size passed in.
    #[test]
    fn x_title_gutter_none_falls_back_to_theme() {
        let m = mock(0.0);
        let input = titled_input(None, None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        assert!(
            (gutter - 23.6).abs() < 1e-9,
            "13*1.2+8 = 23.6, got {gutter}"
        );
    }

    /// `#182` (closes `#94`): the title gutter is reserved only for an axis that
    /// DRAWS its title. `visible: false` is both `.axis(show=False)` and a
    /// projected coord with no Cartesian axes (see
    /// `crate::spec::coord::suppresses_cartesian_axes`, which
    /// `render::prepare::build_axes` folds into `AxesInput::show_x`/`show_y`).
    /// Pinned on both axes, and with an override large enough that the pre-fix
    /// phantom would be unmissable.
    #[test]
    fn title_gutter_undrawn_axis_is_zero_on_both_axes() {
        let m = mock(0.0);
        let input = titled_input(Some(50.0), Some(12.0));

        assert_eq!(
            compute_x_title_width(&input, 13.0, 8.0, &m, false),
            0.0,
            "an x axis that is not drawn reserves no title gutter"
        );
        assert_eq!(
            compute_y_title_width(&input, 13.0, 8.0, &m, false),
            0.0,
            "a y axis that is not drawn reserves no title gutter"
        );

        // Discriminating control: the same titled input, drawn, reserves the
        // full 50*1.2 + 12 gutter on both axes — so the zeros above come from
        // the gate, not from a fixture that has no title to measure.
        assert!((compute_x_title_width(&input, 13.0, 8.0, &m, true) - 72.0).abs() < 1e-9);
        assert!((compute_y_title_width(&input, 13.0, 8.0, &m, true) - 72.0).abs() < 1e-9);
    }

    /// `#182`: the title gutter and the label band are gated SEPARATELY, which
    /// is what makes `fm.Axis(labels=False)` collapse the label band alone. The
    /// title is drawn by `layout_x_axis`/`layout_y_axis` whatever `show_labels`
    /// says, so it keeps its gutter; only the labels' band goes to zero.
    #[test]
    fn title_gutter_survives_labels_off_while_the_label_band_collapses() {
        let m = mock(10.0);
        let mut input = axis_input(
            AxisOrient::Left,
            Some("Title".into()),
            vec!["0".into(), "100".into(), "10000".into()],
            None,
        );
        input.overrides.labels.show = Some(false);

        assert!(
            compute_y_title_width(&input, 13.0, 8.0, &m, true) > 0.0,
            "labels=False must not take the axis title's gutter with it"
        );
        assert_eq!(
            compute_y_label_band_width(&input, 11.0, &m, 4.0, true),
            0.0,
            "labels=False collapses the label band alone"
        );
    }

    /// `compute_x_title_width` and `compute_y_title_width` are parity-equal
    /// for the same input and overrides — the doc comment's stated
    /// "byte-identical to compute_y_title_width" claim, pinned against the
    /// real functions rather than two copies of the same local formula.
    #[test]
    fn title_gutter_x_y_parity_with_overrides() {
        let m = mock(0.0);
        let input = titled_input(Some(20.0), Some(12.0));
        let x_gutter = compute_x_title_width(&input, 13.0, 8.0, &m, true);
        let y_gutter = compute_y_title_width(&input, 13.0, 8.0, &m, true);
        assert_eq!(
            x_gutter, y_gutter,
            "x and y title gutters must match for identical overrides"
        );
    }

    /// A larger per-axis title font size drives a strictly larger real
    /// `compute_x_title_width` gutter, monotonically shrinking a downstream
    /// plot-region-height computation by exactly the gutter's growth.
    #[test]
    fn x_title_gutter_large_font_reduces_plot_region() {
        let m = mock(0.0);
        let inner_h = 400.0;
        let x_label_band = 20.0;

        let gutter_default = compute_x_title_width(&titled_input(None, None), 13.0, 8.0, &m, true);
        let gutter_large =
            compute_x_title_width(&titled_input(Some(50.0), None), 13.0, 8.0, &m, true);
        assert!(
            gutter_large > gutter_default,
            "a larger font must produce a larger gutter"
        );

        let plot_h_default = inner_h - x_label_band - gutter_default;
        let plot_h_large = inner_h - x_label_band - gutter_large;
        assert!(
            plot_h_large < plot_h_default,
            "larger x_title_gutter should reduce plot height: large={plot_h_large}, default={plot_h_default}"
        );
        let gutter_diff = gutter_large - gutter_default;
        let height_diff = plot_h_default - plot_h_large;
        assert!(
            (height_diff - gutter_diff).abs() < 1e-9,
            "the plot-region shrink must equal the gutter's growth exactly"
        );
    }

    /// An explicit `label_angle=` axis whose survivors never fit the band
    /// budget — even once culled by stride, since the stride-cull elides
    /// each survivor against its own widened slot, and a fixed angle's
    /// drawn band never shrinks just because more room opened up — draws
    /// tick marks only. Proportional metrics: twelve labels alternating a
    /// cheap (`'i'`) and an expensive (`'W'`) glyph at 45°; index 0 is
    /// always a stride survivor, so every stride's drawn band is anchored
    /// by that `'i'`-run's own width (elided only to the pre-cull slot,
    /// never wider), and the lone `'W'` surviving a given stride (never
    /// narrower than the `'i'`-run once un-elided) never brings the band
    /// under budget either.
    #[test]
    fn an_explicit_angle_axis_whose_survivors_never_fit_draws_tick_marks_only() {
        let m = MockMetrics {
            measure: |s: &str, _fs: f64| {
                s.chars()
                    .map(|c| match c {
                        'i' => 7.0,
                        'W' => 42.0,
                        '\u{2026}' => 10.0,
                        _ => 8.0,
                    })
                    .sum::<f64>()
            },
            line_h_factor: 1.2,
        };
        let labels: Vec<String> = (0..12)
            .map(|i| if i % 2 == 0 { "iiiiiiiiii".to_string() } else { "W".to_string() })
            .collect();
        let mut input = axis_input(AxisOrient::Bottom, None, labels, None);
        input.overrides.labels.angle = Some(45.0);
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let panel_w = 360.0;
        let (free, _) = lay_out_x_tick_labels(&input, 0.0, panel_w, &ctx);
        let band_initial = band_of(&free, &input, &ctx);
        assert!(
            (band_initial - 52.81).abs() < 0.01,
            "fixture assumption: the unbudgeted band is 52.81, got {band_initial}"
        );
        // Midway between the all-'i…'-survivors band (49.98, what a stride-2
        // cull alone would need) and the unbudgeted band (52.81): a tight
        // enough budget that no stride's drawn band can meet it.
        let budget = (49.98 + 52.81) / 2.0;
        let (ticks, _) = lay_out_x_tick_labels(&input, 0.0, panel_w, &ctx.with_band_budget(budget));
        assert!(
            ticks.iter().all(|t| t.culled),
            "no stride's survivors fit under budget, so none are drawn: {ticks:?}"
        );
    }

    /// An axis whose over-budget band is driven by a wide label that a
    /// stride's cull removes draws its narrower survivors, unelided, at
    /// that very stride. Four labels alternating a narrow `"ab"` and a wide
    /// `"W"`; stride 2 keeps only the `"ab"`s, and their own (much smaller)
    /// band fits a budget the full, `"W"`-driven density does not.
    #[test]
    fn an_explicit_angle_axis_thins_when_the_wide_label_is_culled_away() {
        let m = MockMetrics {
            measure: |s: &str, _fs: f64| {
                s.chars()
                    .map(|c| match c {
                        'W' => 42.0,
                        '\u{2026}' => 10.0,
                        _ => 8.0,
                    })
                    .sum::<f64>()
            },
            line_h_factor: 1.2,
        };
        let labels: Vec<String> = (0..4)
            .map(|i| if i % 2 == 0 { "ab".to_string() } else { "W".to_string() })
            .collect();
        let mut input = axis_input(AxisOrient::Bottom, None, labels, None);
        input.overrides.labels.angle = Some(45.0);
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let panel_w = 120.0;
        let (free, _) = lay_out_x_tick_labels(&input, 0.0, panel_w, &ctx);
        let band_initial = band_of(&free, &input, &ctx);
        assert!(band_initial > 43.0, "fixture assumption: full density needs culling, got {band_initial}");
        let (ticks, _) = lay_out_x_tick_labels(&input, 0.0, panel_w, &ctx.with_band_budget(43.0));
        let drawn: Vec<&TickLayout> = ticks.iter().filter(|t| !t.culled).collect();
        assert_eq!(drawn.len(), 2, "stride 2, every other label drawn");
        assert!(
            drawn.iter().all(|t| !t.elided && t.label == "ab"),
            "the narrow survivors draw in full once the wide one is culled away: {drawn:?}"
        );
    }

    // ── the x label band budget ──────────────────────────────────────────────

    /// Twelve 60px labels on a 360px axis: the cascade rotates them to -90
    /// (each slot is 30px), a band of `tick + pad + font + 60`.
    fn twelve_long_labels() -> AxisInput {
        let labels = (0..12).map(|i| format!("{i:06}")).collect();
        axis_input(AxisOrient::Bottom, Some("title".into()), labels, None)
    }

    fn band_of(ticks: &[TickLayout], input: &AxisInput, ctx: &LabelBandCtx<'_>) -> f64 {
        drawn_x_label_band(ticks, input, ctx)
    }

    /// Under a budget the labels already fit, nothing changes: the same
    /// ticks as with no budget at all.
    #[test]
    fn a_band_budget_the_labels_fit_changes_nothing() {
        let m = mock(10.0);
        let input = twelve_long_labels();
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let (free, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx);
        let band = band_of(&free, &input, &ctx);
        assert!(free.iter().all(|t| !t.culled && t.label_angle == -90.0), "fixture assumption");
        let (bounded, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx.with_band_budget(band + 0.5));
        assert_eq!(bounded, free);
    }

    /// Under a budget the rotated band breaks, the cascade culls by stride
    /// until the survivors fit, smallest stride first. Stride 2 gives each
    /// survivor a 60px slot, too narrow (with the overlap tolerance) for a
    /// flat 60px label, so they rotate to -30 (band about 53), still over a
    /// 25px budget. Stride 3 gives 90px slots: every third label draws flat,
    /// a band of `tick + pad + line_h` (19.2).
    #[test]
    fn a_band_budget_culls_by_stride_until_the_labels_fit() {
        let m = mock(10.0);
        let input = twelve_long_labels();
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m).with_band_budget(25.0);
        let (ticks, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx);
        let culled: Vec<bool> = ticks.iter().map(|t| t.culled).collect();
        assert_eq!(culled, (0..12).map(|i| i % 3 != 0).collect::<Vec<_>>(), "every third label kept");
        assert!(ticks.iter().filter(|t| !t.culled).all(|t| t.label_angle == 0.0), "survivors flat");
        let band = band_of(&ticks, &input, &ctx);
        assert!((band - (4.0 + 2.0 + m.line_height(11.0))).abs() < 1e-9 && band < 25.0);
    }

    /// When not even one label fits the budget, the axis draws its tick
    /// marks and no labels: every label culled, a band of the tick length
    /// alone, and the title sits just below the ticks.
    #[test]
    fn a_band_budget_no_label_fits_draws_tick_marks_only() {
        let m = mock(10.0);
        let input = twelve_long_labels();
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m).with_band_budget(10.0);
        let (ticks, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx);
        assert!(ticks.iter().all(|t| t.culled), "no label drawn");
        assert_eq!(band_of(&ticks, &input, &ctx), 4.0, "the band is the ticks alone");
        let panel = Rect { x: 0.0, y: 0.0, w: 360.0, h: 100.0 };
        let (axis, _) = layout_x_axis(&input, panel, 0, 13.0, 4.0, &ctx);
        let title = axis.title.expect("title");
        let want = 100.0 + 4.0 + 4.0 + m.line_height(13.0) / 2.0;
        assert!((title.anchor_y - want).abs() < 1e-9, "title just below the ticks: {} vs {want}", title.anchor_y);
    }

    /// The budget is inclusive: a budget exactly the band the labels draw
    /// keeps them as drawn with no budget at all.
    #[test]
    fn a_band_budget_equal_to_the_drawn_band_changes_nothing() {
        let m = mock(10.0);
        let input = twelve_long_labels();
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let (free, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx);
        let band = band_of(&free, &input, &ctx);
        let (bounded, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx.with_band_budget(band));
        assert_eq!(bounded, free);
    }

    /// Twelve labels pinned at -90: `"."` (1px) at even indices, `"....."`
    /// (5px) at odd ones. At -90 a label's band is `tick + pad + font +
    /// width`, so the `"."` labels need less than the flat band
    /// (`tick + pad + line_h`) would.
    fn dotted_labels_at_minus_90() -> AxisInput {
        let labels = (0..12).map(|i| if i % 2 == 0 { "." } else { "....." }.to_string()).collect();
        let mut input = axis_input(AxisOrient::Bottom, None, labels, None);
        input.overrides.labels.angle = Some(-90.0);
        input
    }

    /// No drawn presentation has a band below `tick + pad + min(font,
    /// line_h)`: a budget just under it draws tick marks only. At exactly
    /// the band a stride's survivors draw at -90 (the `"."` labels, below
    /// the flat band), they are drawn.
    #[test]
    fn a_band_budget_under_the_least_label_band_draws_tick_marks_only() {
        let m = mock(1.0);
        let input = dotted_labels_at_minus_90();
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let least = 4.0 + 2.0 + 11.0_f64.min(m.line_height(11.0));
        let (ticks, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx.with_band_budget(least - 1e-9));
        assert!(ticks.iter().all(|t| t.culled), "no label drawn under the least band");

        let mut dots = input.clone();
        dots.tick_labels = vec![".".to_string(); 6];
        let (dots_free, _) = lay_out_x_tick_labels(&dots, 0.0, 360.0, &ctx);
        let dot_band = band_of(&dots_free, &dots, &ctx);
        assert!(
            dot_band >= least && dot_band < 4.0 + 2.0 + m.line_height(11.0),
            "fixture assumption: a `.` at -90 draws below the flat band ({dot_band})",
        );
        let (ticks, _) = lay_out_x_tick_labels(&input, 0.0, 360.0, &ctx.with_band_budget(dot_band));
        let culled: Vec<bool> = ticks.iter().map(|t| t.culled).collect();
        assert_eq!(culled, (0..12).map(|i| i % 2 != 0).collect::<Vec<_>>(), "every `.` drawn");
        assert!(ticks.iter().filter(|t| !t.culled).all(|t| t.label_angle == -90.0));
        assert!((band_of(&ticks, &input, &ctx) - dot_band).abs() < 1e-9);
    }

    /// The x title is placed from the band the drawn labels need, so a title
    /// below (or, on a top axis, above) labels wrapped onto several lines
    /// clears the last line by exactly the title padding: no overlap and no
    /// blank strip. Three `_`-separated labels, swept over panel widths,
    /// draw flat on one line when wide and wrap onto several when narrower
    /// (rotated widths are left to the rotated-title tests).
    #[test]
    fn the_x_title_clears_every_wrapped_label_line_by_its_padding() {
        let m = mock(7.0);
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let labels: Vec<String> = (0..3).map(|i| format!("cat{i:02}_wwwwwwww_{i}")).collect();
        let (title_fs, title_pad) = (13.0, 4.0);
        let mut wrapped = 0;
        for orient in [AxisOrient::Bottom, AxisOrient::Top] {
            for width in (100..=700).step_by(10).map(f64::from) {
                let input = axis_input(orient, Some("Short".into()), labels.clone(), None);
                let panel = Rect { x: 0.0, y: 100.0, w: width, h: 200.0 };
                let (axis, _) = layout_x_axis(&input, panel, 0, title_fs, title_pad, &ctx);
                if axis.ticks.iter().any(|t| !t.culled && t.label_angle != 0.0) {
                    continue;
                }
                let lines = axis.ticks.iter().filter(|t| !t.culled).map(|t| t.label.split('\n').count()).max().unwrap_or(1);
                wrapped += usize::from(lines > 1);
                let label_reach = 4.0 + 2.0 + lines as f64 * m.line_height(11.0);
                let title = axis.title.expect("title");
                let half = m.line_height(title_fs) / 2.0;
                let gap = match orient {
                    AxisOrient::Top => (panel.y - label_reach) - (title.anchor_y + half),
                    _ => (title.anchor_y - half) - (panel.y + panel.h + label_reach),
                };
                assert!(
                    (gap - title_pad).abs() < 1e-9,
                    "{orient:?} at {width}px, {lines} line(s): the title clears the labels by {gap}, not {title_pad}",
                );
            }
        }
        assert!(wrapped > 0, "the sweep must reach wrapped labels");
    }

    /// An axis that draws no labels keeps its title offset: one label line
    /// below the axis when flat, and the extent of its (undrawn) labels at an
    /// explicit angle, as before the title was placed from the drawn band.
    #[test]
    fn a_labels_off_x_title_keeps_its_offset() {
        let m = mock(7.0);
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let labels: Vec<String> = vec!["alpha".into(), "beta".into(), "gamma".into()];
        let panel = Rect { x: 0.0, y: 100.0, w: 600.0, h: 200.0 };
        for angle in [None, Some(-45.0)] {
            let mut input = axis_input(AxisOrient::Bottom, Some("Title".into()), labels.clone(), None);
            input.overrides.labels.show = Some(false);
            input.overrides.labels.angle = angle;
            let (axis, _) = layout_x_axis(&input, panel, 0, 13.0, 4.0, &ctx);
            let title = axis.title.expect("title");
            let line_h = m.line_height(11.0);
            let extent = match angle {
                None => line_h,
                Some(a) => {
                    let w = labels.iter().map(|l| m.measure_width(l, 11.0)).fold(0.0, f64::max);
                    rotated_x_label_extent(a, w, 11.0, line_h, 4.0, 2.0)
                }
            };
            let want = panel.y + panel.h + extent + 4.0 + m.line_height(13.0) / 2.0;
            assert!((title.anchor_y - want).abs() < 1e-9, "{angle:?}: title at {}, want {want}", title.anchor_y);
        }
    }


    /// A titled axis with no tick labels places its title where the band it
    /// reserves ends, on bottom and top axes and with an explicit
    /// `label_padding`: the band is one label line plus the padding's delta
    /// from 2px, and the title clears it by the title padding.
    #[test]
    fn a_title_over_no_tick_labels_sits_where_the_reserved_band_ends() {
        let m = mock(7.0);
        let ctx = LabelBandCtx::new(11.0, legacy_gap(0), 4.0, &m);
        let panel = Rect { x: 0.0, y: 100.0, w: 400.0, h: 200.0 };
        for orient in [AxisOrient::Bottom, AxisOrient::Top] {
            for padding in [None, Some(6.0)] {
                let mut input = axis_input(orient, Some("Title".into()), Vec::new(), None);
                input.overrides.labels.padding = padding;
                let band = estimate_x_label_band(&input, panel.w, &ctx);
                assert_eq!(band, m.line_height(11.0) + padding.map_or(0.0, |p| p - 2.0), "the one-line gutter");
                let (axis, _) = layout_x_axis(&input, panel, 0, 13.0, 4.0, &ctx);
                let title = axis.title.expect("title");
                let offset = band + 4.0 + m.line_height(13.0) / 2.0;
                let want = match orient {
                    AxisOrient::Top => panel.y - offset,
                    _ => panel.y + panel.h + offset,
                };
                assert!(
                    (title.anchor_y - want).abs() < 1e-9,
                    "{orient:?} with padding {padding:?}: title at {}, the band ends at {want}",
                    title.anchor_y,
                );
            }
        }
    }
}
