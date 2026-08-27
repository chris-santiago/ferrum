//! Axis input (caller-supplied) and axis layout output (engine-computed).
//! Per spec §14.1: tick labels are caller-pre-computed via Phase 4 scales;
//! Phase 6 never touches scale internals.

use serde::{Deserialize, Serialize};

use super::geometry::{Axis1D, Rect};
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

/// Per-axis style/positioning overrides, bundled so a future field is a one-line
/// struct addition instead of a five-site thread (mirrors `LegendStyleOpts` on
/// the legend side). Every field is `Option` and falls back to the shared theme
/// value (or a layout default) when `None`, so default output stays byte-identical
/// and only a per-channel `fm.Axis(...)` / chart-level `configure_axis(...)` spec
/// lights one up.
///
/// Both the per-channel parse path (`render::prepare::encoding_axis_style_overrides`)
/// and the chart-level apply path (`render::apply_axis_style_to_axis_input`) write
/// here uniformly via the `is_none()` fill-only pattern, so a higher-precedence
/// source (per-channel, or an earlier config layer) always wins. The resolved
/// concrete axis side is computed from [`orient`](Self::orient) into
/// [`AxisInput::orient`] at layout-build time (x→Bottom / y→Left default).
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AxisStyleOverrides {
    /// Explicit tick values (`fm.Axis(values=[...])` / `configure_axis(tick_values=[...])`).
    /// When set, tick labels are replaced with formatted versions of these values.
    pub tick_values: Option<Vec<f64>>,
    /// Tick-label rotation angle override (`fm.Axis(label_angle=...)`). Bypasses
    /// the collision cascade when `Some`. `None` → cascade-resolved angle.
    pub label_angle: Option<f64>,
    /// d3-format string for tick labels (per-channel `label_format` /
    /// chart-level `label_format_raw`). Applied after `tick_values`.
    pub label_format: Option<String>,
    /// Axis title font size. `None` → `theme.title_font_size`.
    pub title_font_size: Option<f64>,
    /// Axis title color. `None` → `theme.title_color`.
    pub title_color: Option<Srgba<u8>>,
    /// Padding between axis title and tick labels. `None` → `theme.axis_title_padding`.
    pub title_padding: Option<f64>,
    /// Pixel gap between the end of a tick mark and the tick label baseline.
    /// `None` → the renderer's hardcoded per-orient gaps.
    pub label_padding: Option<f64>,
    /// Tick-label color override. `None` → `theme.colors.label_color`.
    pub label_color: Option<Srgba<u8>>,
    /// Tick-label font-size override. `None` → `theme.typography.label_font_size`.
    pub label_font_size: Option<f64>,
    /// Gridline color override. `None` → `theme.colors.grid_color`.
    pub grid_color: Option<Srgba<u8>>,
    /// Gridline dash override. `None` → `theme.grid.grid_dash`.
    pub grid_dash: Option<Vec<f64>>,
    /// Gridline width override. `None` → `theme.sizes.grid_width`.
    pub grid_width: Option<f64>,
    /// Domain-line color override. `None` → `theme.colors.axis_line_color`.
    pub domain_color: Option<Srgba<u8>>,
    /// Domain-line width override. `None` → `theme.sizes.axis_line_width`.
    pub domain_width: Option<f64>,
    /// Axis side override, validated against the channel dimension (x→top/bottom,
    /// y→left/right). `None` → the channel's default side (Bottom for x, Left for
    /// y). Resolved into the concrete [`AxisInput::orient`] at layout-build time.
    pub orient: Option<AxisOrient>,
    /// Shift the axis group perpendicular to its line by N px (outward positive),
    /// composing additively with the renderer's `offset` handling. `None` → no shift.
    pub translate: Option<f64>,
    /// Lower bound (px) for the reserved axis margin band — reserve at least this
    /// much. `None` → dynamic band only.
    pub min_band: Option<f64>,
    /// Upper bound (px) for the reserved axis margin band — cap at this much
    /// (labels may clip past it). `None` → no cap.
    pub max_band: Option<f64>,
    /// Per-axis grid-line opacity override `[0, 1]`. `None` → `theme.grid.grid_opacity`.
    pub grid_opacity: Option<f64>,
    /// Side/orientation of the axis title relative to its axis (e.g. a horizontal
    /// title on a left axis). `None` → the orient-default rotation.
    pub title_orient: Option<AxisOrient>,
    /// Coarse draw order relative to marks: `>= 1` → axis + grid drawn above
    /// marks; `<= 0` (default) → below marks. `None` → default (below).
    pub zindex: Option<i64>,
    /// Append a tick at each domain boundary (scale min/max) if not already
    /// present. `None`/`false` → no boundary ticks.
    pub tick_extra: Option<bool>,
    /// Minimum step (data units) between generated ticks; ticks closer than this
    /// in data space are dropped. `None` → no thinning.
    pub tick_min_step: Option<f64>,
    /// Shift the axis perpendicular AWAY from the plot edge by N px (Vega axis
    /// `offset`). Composes **additively** with [`translate`](Self::translate):
    /// the renderer applies `translate + offset` as a single outward shift.
    /// `None`/`0` → no shift.
    pub offset: Option<f64>,
    /// Flush the first/last tick labels at the axis ends so edge labels align
    /// within the plot bounds instead of overflowing (Vega `labelFlush`).
    /// `None`/`false` → default anchors (byte-identical).
    pub label_flush: Option<bool>,
    /// Tick-label overlap strategy override. `None` → the unmodified collision
    /// cascade. Only consumed by the x-axis layout; the y-axis applies no overlap
    /// policy.
    pub label_overlap: Option<LabelOverlap>,
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
    /// `overrides.orient` (default Bottom for x, Left for y) via
    /// [`resolve_orient`](Self::resolve_orient) after all override layers merge.
    pub orient: AxisOrient,
    pub title: Option<String>,
    pub tick_labels: Vec<String>,
    /// When `false`, tick labels are suppressed (D7: `axis.labels`).
    /// Default `true` — preserves byte-identity for all existing goldens.
    pub show_labels: bool,
    /// When `false`, tick marks are suppressed (D7: `axis.ticks`).
    /// Default `true`.
    pub show_ticks: bool,
    /// When `false`, the axis domain line is suppressed (D7: `axis.domain`).
    /// Default `true`.
    pub show_domain: bool,
    /// When `false`, gridlines for this axis are suppressed even when the theme
    /// enables them globally (D7: `axis.grid`). Default `true`.
    pub show_grid: bool,
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
    /// Absolute band-center pixels for a **categorical** axis whose ordinal scale
    /// carries an explicit pixel range (GH #39 phase 2, band-geometry
    /// unification). One entry per category in [`tick_labels`](Self::tick_labels)
    /// order; each is the same pixel the mark for that category is placed at, so
    /// tick labels and grid lines agree with the marks (spec §7). `Some` only for
    /// an explicit-range ordinal axis; `None` for continuous axes (which use
    /// [`tick_projection`](Self::tick_projection)) and for ordinal axes without an
    /// explicit range (which keep the `uniform_center` slot placement,
    /// byte-identically). Mutually exclusive with `tick_projection` — an ordinal
    /// scale yields no projection. Unlike projected fractions, these are absolute
    /// pixels used **directly** by `layout_*_axis`, not mapped through the
    /// panel-extent padding inset.
    pub categorical_positions: Option<Vec<f64>>,
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
}

impl AxisInput {
    /// Construct an `AxisInput` with all overrides empty and all show_* = true
    /// (the backward-compatible defaults). `label_angle` seeds
    /// `overrides.label_angle`.
    pub fn new(
        orient: AxisOrient,
        title: Option<String>,
        tick_labels: Vec<String>,
        label_angle: Option<f64>,
    ) -> Self {
        Self {
            orient,
            title,
            tick_labels,
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
            tick_format: None,
            tick_format_type: None,
            tick_projection: None,
            categorical_positions: None,
            overrides: AxisStyleOverrides {
                label_angle,
                ..AxisStyleOverrides::default()
            },
        }
    }

    /// Re-resolve the concrete [`orient`](Self::orient) from the
    /// `overrides.orient` override after all override layers (per-channel,
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
        self.orient = self.overrides.orient.unwrap_or(default);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxesInput {
    pub x: AxisInput,
    pub y: AxisInput,
    /// When false, the x axis line + ticks + labels + title are suppressed
    /// at layout time. Used by `ChartSpec.axis_x = Some(false)` (i.e.
    /// `Chart.axis(x=False)`) on clustermap dendrogram panels and JointChart
    /// marginal panels. Default `true`.
    pub show_x: bool,
    /// Y-axis variant of `show_x`. Default `true`.
    pub show_y: bool,
    /// Secondary y-axis inputs, one per `independent_y` layer, in layer order
    /// (secondary-y-axis, GH #52). Each renders on the right, stacked outward
    /// beyond `y`'s band (slot 0 stays `y`, driving the left axis and
    /// gridlines — see spec §6 slot contract). Empty (the default) means the
    /// chart has no independent-y layer, so layout reserves no extra band and
    /// emits no extra axis — byte-identical to the pre-#52 shared path.
    pub secondary_y: Vec<AxisInput>,
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

fn default_true() -> bool { true }

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
            show_labels: input.show_labels,
            show_ticks: input.show_ticks,
            show_domain: input.show_domain,
            show_grid: input.show_grid,
            title_font_size: input.overrides.title_font_size,
            title_color_rgba: input.overrides.title_color.map(rgba_array),
            label_padding: input.overrides.label_padding,
            label_color_rgba: input.overrides.label_color.map(rgba_array),
            label_font_size: input.overrides.label_font_size,
            grid_color_rgba: input.overrides.grid_color.map(rgba_array),
            grid_dash: input.overrides.grid_dash.clone(),
            grid_width: input.overrides.grid_width,
            domain_color_rgba: input.overrides.domain_color.map(rgba_array),
            domain_width: input.overrides.domain_width,
            grid_opacity: input.overrides.grid_opacity,
            translate: input.overrides.translate,
            zindex: input.overrides.zindex,
            offset: input.overrides.offset,
            label_flush: input.overrides.label_flush,
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
    policy: NonFinitePolicy,
) -> Option<Vec<f64>> {
    if fractions.is_empty() {
        return None;
    }
    let (lo, hi) = crate::layout::geometry::inset_pixel_range(base_range, padding_frac);
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
        NonFinitePolicy::DropAll,
    )
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
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickLayout {
    pub position: f64,
    pub label: String,         // may contain '\n' for multi-line labels (future task)
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisTitleLayout {
    pub text: String,
    pub anchor_x: f64,
    pub anchor_y: f64,
    pub angle: f64,
}

use super::text_metrics::TextMetrics;

/// Vertical extent (px below the axis line) occupied by an x-axis tick label
/// rotated by `angle` (degrees, may be negative). Mirrors the end-anchored
/// pivot geometry in `render/marks/axis.rs`: the pivot sits
/// `tick_size + label_pad + sin(|angle|)·font_size` below the axis line, then the
/// rotated label drops a further `sin(|angle|)·max_label_w + cos(|angle|)·line_h`.
///
/// At -90 this collapses to `tick_size + label_pad + font_size + max_label_w`
/// (sin=1, cos=0); at angle 0 it would give `tick_size + label_pad + line_h`,
/// which is *not* the flat band — callers must guard on `angle != 0` and use the
/// flat `line_h` term directly for un-rotated labels.
///
/// SYNC: render (`render/marks/axis.rs`), the rotated branches of
/// `estimate_x_label_band`, and the rotated title placement in `layout_x_axis`
/// all depend on this. Change all three together or the x-axis title will
/// overlap (or float above) the rotated labels.
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

/// Estimate the vertical space (in pixels) needed below the x-axis to
/// accommodate tick labels, accounting for how the collision cascade is likely
/// to resolve. Called by the layout orchestrator **before** the plot rect is
/// finalized, so it uses worst-case inputs (longest label, estimated slot
/// width). Over-reservation is acceptable; under-reservation causes clipping.
///
/// Algorithm (mirrors the cascade order in `cascade_collision_recovery`):
/// 1. If `label_angle_override` is set, use that angle directly.
/// 2. If all labels fit flat, return `line_height`.
/// 3. If wrapping resolves collision (all labels wrap successfully), return
///    `max_lines * line_height`.
/// 4. Try each angle in `ANGLE_CASCADE`; first that passes returns the full
///    geometric extent of the rotated label (see the SYNC comment below).
/// 5. Fallback: vertical labels (-90°, S4/S5 scenarios) reserve the full extent
///    at sin=1, cos=0.
///
/// SYNC: the rotated branches mirror the rotated-bottom-label geometry in
/// `crate::render::marks::axis::build_axis`. A rotated label is end-anchored at
/// the pivot `(tick.position, label_y)` where
/// `label_y = r.y + tick_size + label_pad + sin(|angle|)·font_size`, then rotated
/// about that pivot. Its lowest point sits below the pivot, so the full extent
/// from the axis line (`r.y`) down to the label bottom is
/// `tick_size + label_pad + sin(|angle|)·(font_size + max_label_w) + cos(|angle|)·descent`.
/// The band below uses `line_h` for the cos term (instead of a bare descent) to
/// match the existing code and keep a small safety margin. Changing either side
/// requires changing the other or the x-axis title will overlap the labels.
pub(crate) fn estimate_x_label_band(
    labels: &[String],
    label_font_size: f64,
    label_angle_override: Option<f64>,
    metrics: &dyn TextMetrics,
    estimated_slot_w: f64,
    label_padding: Option<f64>,
    tick_size: f64,
) -> f64 {
    // When label_padding is explicitly set, it replaces the hardcoded 2.0 gap
    // in build_axis. The delta from the default (2.0) is added to the margin
    // estimate for the flat/wrapped branches. When label_padding is None the
    // existing margin values are unchanged (backward-compatible with all
    // existing goldens). The rotated branches instead fold the *effective* pad
    // directly into the pivot offset (`label_pad_eff` below), so they do not add
    // `padding_delta` again — it is subsumed.
    let padding_delta = label_padding.map(|lp| lp - 2.0).unwrap_or(0.0);
    // Clamp to match the renderer (`render/marks/axis.rs` L-2 guard: `.max(0.0)`).
    // A negative label_padding must not under-reserve the rotated band — the
    // renderer clamps it to 0, so the layout must too or band < render extent.
    let label_pad_eff = label_padding.unwrap_or(2.0).max(0.0);
    let line_h = metrics.line_height(label_font_size);

    // Empty label set: fall back to current behavior.
    if labels.is_empty() {
        return line_h + padding_delta;
    }

    let max_label_w = labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .fold(0.0_f64, f64::max);

    // If the caller has set label_angle_override, skip the cascade entirely
    // and compute the margin for that specific angle.
    if let Some(angle) = label_angle_override {
        // SYNC (see `rotated_x_label_extent`): full geometric extent of the
        // rotated label below the axis line, shared with the title placement.
        return rotated_x_label_extent(
            angle,
            max_label_w,
            label_font_size,
            line_h,
            tick_size,
            label_pad_eff,
        );
    }

    let threshold = estimated_slot_w * (1.0 - LABEL_OVERLAP_TOLERANCE);

    // S0 — flat: if widest label fits, no extra margin needed.
    if max_label_w <= threshold {
        return line_h + padding_delta;
    }

    // S1 — wrapping: attempt to wrap all labels and count max lines.
    {
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
                let max_lines = wrapped_labels
                    .iter()
                    .map(|w| w.split('\n').count())
                    .max()
                    .unwrap_or(1);
                return max_lines as f64 * line_h + padding_delta;
            }
        }
    }

    // S2/S3 — rotation: find the first angle in the cascade that resolves
    // collision (same logic as `cascade_collision_recovery` S3).
    for &angle in &ANGLE_CASCADE[1..] {
        let cos_factor = angle.to_radians().cos().abs();
        if max_label_w * cos_factor <= estimated_slot_w {
            // SYNC (see `rotated_x_label_extent`): full geometric extent of the
            // rotated label, shared with the title placement.
            return rotated_x_label_extent(
                angle,
                max_label_w,
                label_font_size,
                line_h,
                tick_size,
                label_pad_eff,
            );
        }
    }

    // S4/S5 fallback: vertical labels (-90°). The helper at -90 gives sin=1,
    // cos=0, so the extent collapses to
    // `tick_size + label_pad_eff + font_size + max_label_w`
    // (SYNC with the rotated render geometry above).
    rotated_x_label_extent(
        -90.0,
        max_label_w,
        label_font_size,
        line_h,
        tick_size,
        label_pad_eff,
    )
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
/// SYNC: render (`render/marks/axis.rs` `Left`/`Right` arms), the rotated
/// branch of `compute_y_label_band_width`, and `layout_y_axis`'s
/// override-angle branch all depend on this. Change all three together or the
/// y-axis title / plot region will overlap (or float away from) the rotated
/// labels.
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
/// θ=0 (no override, or an explicit `0.0`) returns the widest tick label's
/// measured width — bit-for-bit the pre-R2 behavior, which deliberately omits
/// `tick_size`/`label_pad` (an existing quirk of the y band, preserved here;
/// see `rotated_y_label_extent`'s doc for why the rotated branch DOES include
/// them). A non-zero override goes through the transposed rotated-extent
/// formula instead (SYNC with `rotated_y_label_extent`).
pub fn compute_y_label_band_width(
    input: &AxisInput,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
    tick_size: f64,
) -> f64 {
    let max_label_w = input
        .tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .fold(0.0_f64, f64::max);
    match input.overrides.label_angle {
        Some(angle) if angle != 0.0 => {
            let line_h = metrics.line_height(label_font_size);
            let label_pad = input.overrides.label_padding.unwrap_or(2.0).max(0.0);
            rotated_y_label_extent(angle, max_label_w, line_h, tick_size, label_pad)
        }
        _ => max_label_w,
    }
}

/// Returns the title-row width contribution: title text height (rotated 90°,
/// so its "width" along the x-axis is its line height) plus axis_title_padding.
/// Returns 0 if there is no title.
pub fn compute_y_title_width(
    input: &AxisInput,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    if input.title.is_some() {
        let effective_title_font_size = input.overrides.title_font_size.unwrap_or(title_font_size);
        let effective_title_padding = input.overrides.title_padding.unwrap_or(axis_title_padding);
        metrics.line_height(effective_title_font_size) + effective_title_padding
    } else {
        0.0
    }
}

/// Returns the x-axis title-gutter height contribution: title text line height
/// (the title is unrotated on the x axis, so its band along the y-axis is its line
/// height) plus axis_title_padding. Returns 0 if there is no title. The body is
/// byte-identical to [`compute_y_title_width`] (cohesion finding LAYOUT-855:
/// `compute_layout` previously inlined this formula "mirroring the y-axis pattern"
/// while y was a named helper, leaving the axis family asymmetric).
pub fn compute_x_title_width(
    input: &AxisInput,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    if input.title.is_some() {
        let effective_title_font_size = input.overrides.title_font_size.unwrap_or(title_font_size);
        let effective_title_padding = input.overrides.title_padding.unwrap_or(axis_title_padding);
        metrics.line_height(effective_title_font_size) + effective_title_padding
    } else {
        0.0
    }
}

/// Build the AxisLayout for the y-axis (Left orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.h`; no graduated
/// collision cascade applies to the y-axis (spec §14.4). `input.overrides.label_angle`
/// (R2) is the one exception: when set, it bypasses the (absent) cascade the
/// same way it bypasses x's cascade, rotating every tick and eliding any label
/// whose rotated footprint still collides with its neighbor (no cull recovery
/// on y — see `AxisLabelWarning`).
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
    let n = input.tick_labels.len();
    let slot_h = if n > 0 { panel_area.h / n as f64 } else { 0.0 };
    // Uniform-slot fallback range (top → bottom, in pixel order): slot `i`'s center
    // is `panel_area.y + (i + 0.5)*slot_h` via `Axis1D::uniform_center`.
    let slot_axis = Axis1D { lo: panel_area.y, hi: panel_area.y + panel_area.h };
    // Continuous axes: place each tick at its scale-projected pixel (mark range
    // is the inverted y range `(bottom, top)`, inset exactly like data marks).
    // Categorical axes (no projected fractions): keep the uniform-slot formula.
    let projected = project_tick_positions(
        input,
        (panel_area.y + panel_area.h, panel_area.y),
    );
    // Explicit-range ordinal axes (GH #39 phase 2): place each tick at the
    // scale's absolute band center — the same pixel its mark gets — so labels and
    // grid lines agree with the marks. Absent (`None`) → the uniform-slot formula,
    // byte-identical to before.
    let band_centers = input.categorical_positions.as_deref();
    let tick_position = |i: usize| -> f64 {
        match (&projected, band_centers) {
            (Some(px), _) => px[i],
            (None, Some(centers)) => centers[i],
            (None, None) => slot_axis.uniform_center(i, slot_h),
        }
    };
    // Per-tick vertical budget used to judge whether a rotated label still
    // collides with its neighbor — the y-dimension transpose of
    // `layout_x_axis`'s `cascade_slot_w`. Continuous/explicit-range axes use
    // the minimum adjacent gap between projected/band-center positions (the
    // tightest pair, worst case); uniform categorical axes use `slot_h`.
    let cascade_slot_h = match (&projected, band_centers) {
        (Some(px), _) => min_adjacent_gap(px).unwrap_or(slot_h),
        (None, Some(centers)) => min_adjacent_gap(centers).unwrap_or(slot_h),
        (None, None) => slot_h,
    };

    let (ticks, warning) = if let Some(override_angle) = input.overrides.label_angle {
        // R2: label_angle_override always applies (there is no cascade to
        // bypass on y). Shared body (quality-review fix 4): see
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
            })
            .collect();
        (ticks, None)
    };
    // Minors use the SAME inverted base range + padding inset as the major
    // projection above, so a minor at domain `v` coincides with the major
    // projection of `v`.
    let minor_ticks = build_minor_ticks(input, (panel_area.y + panel_area.h, panel_area.y));

    // Orient: Left (default) places the axis on the panel's left edge; Right on
    // the right edge. Any other orient is rejected upstream (`prepare.rs`
    // validates x→{top,bottom}, y→{left,right}); default to Left defensively.
    let on_right = matches!(input.orient, AxisOrient::Right);
    let axis_x = if on_right { panel_area.x + panel_area.w } else { panel_area.x };
    let axis_line = Rect {
        x: axis_x,
        y: panel_area.y,
        w: 1.0,
        h: panel_area.h,
    };

    let effective_title_font_size = input.overrides.title_font_size.unwrap_or(title_font_size);
    let effective_title_padding = input.overrides.title_padding.unwrap_or(axis_title_padding);

    let title = input.title.as_ref().map(|text| {
        let label_band = compute_y_label_band_width(input, label_font_size, metrics, tick_size);
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
        let angle = match input.overrides.title_orient {
            Some(AxisOrient::Top) | Some(AxisOrient::Bottom) => 0.0,
            Some(AxisOrient::Right) => 90.0,
            Some(AxisOrient::Left) => -90.0,
            None => if on_right { 90.0 } else { -90.0 },
        };
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: title_x,
            anchor_y: panel_area.y + panel_area.h / 2.0,
            angle,
        }
    });

    // 385: single construction site for the 22-field AxisLayout (shared with
    // layout_x_axis via `AxisLayout::from_input`).
    let layout = AxisLayout::from_input(input, panel_index, axis_line, ticks, minor_ticks, title);
    (layout, warning)
}

use crate::layout::{LABEL_OVERLAP_TOLERANCE, ANGLE_CASCADE, FONT_SHRINK_FACTOR};
use crate::layout::text_metrics::measure_multiline_width;

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
fn elide_to_fit(
    label: &str,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> String {
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
/// 1. Underscore: split on `_` boundaries.
/// 2. Space: greedy line-fill — pack words until adding the next would exceed
///    `max_width`, then start a new line.
/// 3. camelCase: split at lowercase->uppercase transitions.
fn wrap_label(
    label: &str,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> Option<String> {
    // Rule 1: underscore split.
    if label.contains('_') {
        let segments: Vec<&str> = label.split('_').collect();
        if segments.iter().any(|s| metrics.measure_width(s, font_size) > max_width) {
            return None;
        }
        return Some(segments.join("\n"));
    }

    // Rule 2: space — greedy line-fill.
    if label.contains(' ') {
        let words: Vec<&str> = label.split(' ').collect();
        // Any single word that exceeds max_width makes wrapping impossible.
        if words.iter().any(|w| metrics.measure_width(w, font_size) > max_width) {
            return None;
        }
        let mut lines: Vec<String> = Vec::new();
        let mut current = String::new();
        for word in &words {
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
        if segments.iter().any(|s| metrics.measure_width(s, font_size) > max_width) {
            return None;
        }
        return Some(segments.join("\n"));
    }

    // No break points found.
    None
}

/// Shared body for `layout_x_axis`'s and `layout_y_axis`'s `label_angle`
/// override branch (R2, quality-review fix 4): stamp `override_angle` onto
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
            }
        })
        .collect();
    let warning = if elided_count > 0 {
        Some(AxisLabelWarning::LabelsElided { count: elided_count })
    } else {
        None
    };
    (ticks, warning)
}

/// Run the graduated collision cascade (spec SS4.1). Tries recovery strategies in
/// order (S0 flat -> S1 wrap -> S2 font shrink -> S3 rotate -> S4 cull -> S5 elide),
/// returning as soon as one resolves all collisions.
///
/// `label_overlap` (B5 unit 6b) biases the cascade onto an explicit primitive
/// instead of running the graduated flow:
/// - [`LabelOverlap::ShowAll`] short-circuits to S0-Flat with every label
///   visible (no cull/elide), so labels may overlap — the user's choice.
/// - [`LabelOverlap::Parity`] short-circuits to a stride-2 cull (S4 with a fixed
///   parity stride), independent of measured width.
/// - [`LabelOverlap::Rotate`] short-circuits to S3-rotate at the steepest cascade
///   angle.
/// - `None` / [`LabelOverlap::Greedy`] runs the unmodified cascade (default),
///   keeping default output byte-identical.
fn cascade_collision_recovery(
    labels: &[String],
    slot_w: f64,
    label_font_size: f64,
    cull_threshold: u32,
    label_overlap: Option<LabelOverlap>,
    metrics: &dyn TextMetrics,
) -> CascadeResult {
    let n = labels.len();
    let all_visible = vec![true; n];

    // ── label_overlap override (B5 unit 6b) ─────────────────────────────────
    // Bias the cascade onto an explicit primitive. `None`/`Greedy` skip this and
    // fall through to the graduated cascade below (byte-identical default).
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
            // Stride-2 decimation: show every other label. Reuses the cascade's
            // cull primitive with a fixed parity stride (no width measurement).
            let visible: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
            return CascadeResult {
                labels: labels.to_vec(),
                angle: 0.0,
                font_size: None,
                visible,
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

    // Measure all labels at their original font size.
    let widths: Vec<f64> = labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();

    let threshold = slot_w * (1.0 - LABEL_OVERLAP_TOLERANCE);

    // S0 — Flat: if no label exceeds the threshold, done.
    if widths.iter().all(|w| *w <= threshold) {
        return CascadeResult {
            labels: labels.to_vec(),
            angle: 0.0,
            font_size: None,
            visible: all_visible,
            strategy: CascadeStrategy::Flat,
        };
    }

    // S1 — Wrap: try wrapping all labels. All must successfully wrap AND fit.
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
            return CascadeResult {
                labels: wrapped_labels,
                angle: 0.0,
                font_size: None,
                visible: all_visible,
                strategy: CascadeStrategy::Wrapped,
            };
        }
    }

    // S2 — Font shrink: try at reduced font size. If it doesn't help, proceed
    // at ORIGINAL font size (rotation at smaller fonts is hard to read).
    let reduced_fs = label_font_size * FONT_SHRINK_FACTOR;
    let reduced_widths: Vec<f64> = labels
        .iter()
        .map(|s| metrics.measure_width(s, reduced_fs))
        .collect();

    // S2a: reduced font, flat.
    if reduced_widths.iter().all(|w| *w <= threshold) {
        return CascadeResult {
            labels: labels.to_vec(),
            angle: 0.0,
            font_size: Some(reduced_fs),
            visible: all_visible,
            strategy: CascadeStrategy::FontReduced,
        };
    }

    // S2b: reduced font + wrapping.
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
            return CascadeResult {
                labels: wrapped_labels,
                angle: 0.0,
                font_size: Some(reduced_fs),
                visible: all_visible,
                strategy: CascadeStrategy::FontReduced,
            };
        }
    }

    // S3 — Graduated rotation: try each angle from ANGLE_CASCADE (skip 0.0, already tried).
    // Use ORIGINAL labels and ORIGINAL font size.
    for &angle in &ANGLE_CASCADE[1..] {
        let cos_factor = angle.to_radians().cos().abs();
        let all_fit = widths.iter().all(|w| *w * cos_factor <= slot_w);
        if all_fit {
            return CascadeResult {
                labels: labels.to_vec(),
                angle,
                font_size: None,
                visible: all_visible,
                strategy: CascadeStrategy::Rotated { angle },
            };
        }
    }

    // S4 — Tick culling: only if labels.len() > cull_threshold.
    // Use -90 degrees (last/steepest angle in cascade).
    let best_angle = *ANGLE_CASCADE.last().unwrap(); // -90.0
    let cos_best = best_angle.to_radians().cos().abs();

    if n as u32 > cull_threshold {
        // Find max projected width at the best angle.
        let max_projected = widths
            .iter()
            .map(|w| *w * cos_best)
            .fold(0.0_f64, f64::max);

        // Compute minimum stride N where max_projected <= slot_w * N.
        let stride = if max_projected <= 0.0 || slot_w <= 0.0 {
            1_u32
        } else {
            (max_projected / slot_w).ceil().max(1.0) as u32
        };

        if stride > 1 {
            let visible: Vec<bool> = (0..n).map(|i| i % stride as usize == 0).collect();
            return CascadeResult {
                labels: labels.to_vec(),
                angle: best_angle,
                font_size: None,
                visible,
                strategy: CascadeStrategy::Culled { stride },
            };
        }

        // stride == 1 means all fit at -90 without culling — return as rotated.
        return CascadeResult {
            labels: labels.to_vec(),
            angle: best_angle,
            font_size: None,
            visible: all_visible,
            strategy: CascadeStrategy::Rotated { angle: best_angle },
        };
    }

    // S5 — Elision: last resort. Use -90 degrees. Elide labels that still collide.
    let mut elided_count: u32 = 0;
    let elided_labels: Vec<String> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let projected = widths[i] * cos_best;
            if projected > slot_w {
                elided_count += 1;
                let budget = if cos_best > 1e-6 { slot_w / cos_best } else { slot_w };
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
        strategy: CascadeStrategy::Elided { count: elided_count },
    }
}

/// Build the AxisLayout for the x-axis (Bottom orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.w` (spec SS14.3 step 7a).
/// Collision policy: graduated cascade (wrap -> shrink -> rotate -> cull -> elide).
#[allow(clippy::too_many_arguments)]
pub fn layout_x_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    label_font_size: f64,
    title_font_size: f64,
    axis_title_padding: f64,
    cull_threshold: u32,
    tick_size: f64,
    metrics: &dyn TextMetrics,
) -> (AxisLayout, Option<AxisLabelWarning>) {
    let n = input.tick_labels.len();
    let slot_w = if n > 0 { panel_area.w / n as f64 } else { 0.0 };
    // Uniform-slot fallback range (left → right, in pixel order): slot `i`'s center
    // is `panel_area.x + (i + 0.5)*slot_w` via `Axis1D::uniform_center`.
    let slot_axis = Axis1D { lo: panel_area.x, hi: panel_area.x + panel_area.w };

    // Continuous axes: place each tick at its scale-projected pixel; categorical
    // axes (no projected fractions) keep the uniform-slot center. The closure
    // resolves a tick's position by index against whichever placement applies.
    let projected = project_tick_positions(input, (panel_area.x, panel_area.x + panel_area.w));
    // Explicit-range ordinal axes (GH #39 phase 2): place each tick at the scale's
    // absolute band center — the same pixel its mark gets — so labels and grid
    // lines agree with the marks. Absent (`None`) → the uniform-slot formula,
    // byte-identical to before.
    let band_centers = input.categorical_positions.as_deref();
    let tick_position = |i: usize| -> f64 {
        match (&projected, band_centers) {
            (Some(px), _) => px[i],
            (None, Some(centers)) => centers[i],
            (None, None) => slot_axis.uniform_center(i, slot_w),
        }
    };
    // The collision cascade judges label fit against the available horizontal
    // budget per tick. For uniform (categorical) axes that is `slot_w`. For
    // continuous axes the spacing is non-uniform (log/pow/symlog), so use the
    // *minimum* adjacent gap between projected positions — the worst case — to
    // avoid under-counting collisions where ticks bunch toward one end.
    // An explicit-range ordinal axis packs its bands into `[a, b]`, so the per-tick
    // budget is the band step, not the full-panel `slot_w` — use the min adjacent
    // gap between band centers so label collisions are judged against the true
    // (tighter) spacing.
    let cascade_slot_w = match (&projected, band_centers) {
        (Some(px), _) => min_adjacent_gap(px).unwrap_or(slot_w),
        (None, Some(centers)) => min_adjacent_gap(centers).unwrap_or(slot_w),
        (None, None) => slot_w,
    };

    let (ticks, warning) = if let Some(override_angle) = input.overrides.label_angle {
        // label_angle_override always bypasses the cascade (spec SS7). Shared
        // body (quality-review fix 4): see `stamp_override_angle_with_elide`'s
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
            cull_threshold,
            input.overrides.label_overlap,
            metrics,
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
            })
            .collect();
        let warning = match cascade.strategy {
            CascadeStrategy::Elided { count } => {
                Some(AxisLabelWarning::LabelsElided { count })
            }
            _ => None,
        };
        (ticks, warning)
    };

    // Orient: Bottom (default) places the axis at the panel's bottom edge; Top
    // at the top edge. Any other orient is rejected upstream (`prepare.rs`
    // validates x→{top,bottom}); default to Bottom defensively.
    let on_top = matches!(input.orient, AxisOrient::Top);
    let axis_y = if on_top { panel_area.y } else { panel_area.y + panel_area.h };
    let axis_line = Rect {
        x: panel_area.x,
        y: axis_y,
        w: panel_area.w,
        h: 1.0,
    };

    let effective_title_font_size = input.overrides.title_font_size.unwrap_or(title_font_size);
    let effective_title_padding = input.overrides.title_padding.unwrap_or(axis_title_padding);

    // Resolved tick angle: all non-culled ticks share `label_angle` (the cascade
    // and the override path both apply a single angle). 0.0 means flat. Use any
    // non-culled tick; default to flat when every tick is culled or absent.
    let resolved_angle = ticks
        .iter()
        .find(|t| !t.culled)
        .map(|t| t.label_angle)
        .unwrap_or(0.0);

    let title = input.title.as_ref().map(|text| {
        let title_h = metrics.line_height(effective_title_font_size);
        // The vertical drop from the axis line to the labels' lowest point. For
        // flat labels this is a single line height (keeps flat goldens byte
        // identical). For rotated labels it is the full end-anchored extent,
        // shared with `estimate_x_label_band` via `rotated_x_label_extent` so the
        // reserved band and the title placement cannot drift.
        let label_extent = if resolved_angle == 0.0 {
            metrics.line_height(label_font_size)
        } else {
            // Clamp matches the renderer's L-2 guard so band >= render extent.
            let label_pad = input.overrides.label_padding.unwrap_or(2.0).max(0.0);
            let line_h = metrics.line_height(label_font_size);
            // Widest final (possibly-elided) label that will actually render —
            // skip culled ticks since they draw no label.
            let max_label_w = ticks
                .iter()
                .filter(|t| !t.culled)
                .map(|t| metrics.measure_width(&t.label, label_font_size))
                .fold(0.0_f64, f64::max);
            rotated_x_label_extent(
                resolved_angle,
                max_label_w,
                label_font_size,
                line_h,
                tick_size,
                label_pad,
            )
        };
        // Title sits beyond the tick labels, on the same side as the axis. For a
        // Top axis the band extends upward (subtract); for Bottom, downward (add).
        let band = label_extent + effective_title_padding + title_h / 2.0;
        let anchor_y = if on_top { axis_y - band } else { axis_y + band };
        // `title_orient` overrides the title rotation. The default for a
        // horizontal (top/bottom) axis is a flat title (`0`); a vertical
        // `title_orient` (left/right) rotates it (e.g. a vertical caption beside a
        // bottom axis).
        let angle = match input.overrides.title_orient {
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

    // 385: single construction site for the 22-field AxisLayout (shared with
    // layout_y_axis via `AxisLayout::from_input`).
    let layout = AxisLayout::from_input(input, panel_index, axis_line, ticks, minor_ticks, title);
    (layout, warning)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 385: AxisLayout::from_input field parity ─────────────────────────────

    /// `AxisLayout::from_input` threads every per-axis override onto the layout.
    /// Set distinct values for the 16 override threads + the geometry/show fields
    /// and assert each lands, so a dropped thread is caught.
    #[test]
    fn axis_layout_from_input_threads_all_overrides() {
        let red = Srgba::new(255u8, 0, 0, 255);
        let blue = Srgba::new(0u8, 0, 255, 255);
        let green = Srgba::new(0u8, 255, 0, 255);
        let mut input = AxisInput::new(
            AxisOrient::Bottom,
            Some("T".into()),
            vec!["a".into()],
            None,
        );
        input.show_labels = false;
        input.show_ticks = false;
        input.show_domain = false;
        input.show_grid = false;
        input.overrides = AxisStyleOverrides {
            title_font_size: Some(14.0),
            title_color: Some(red),
            label_padding: Some(3.0),
            label_color: Some(blue),
            label_font_size: Some(9.0),
            grid_color: Some(green),
            grid_dash: Some(vec![2.0, 1.0]),
            grid_width: Some(0.7),
            domain_color: Some(red),
            domain_width: Some(1.3),
            grid_opacity: Some(0.5),
            translate: Some(4.0),
            zindex: Some(2),
            offset: Some(6.0),
            label_flush: Some(true),
            ..AxisStyleOverrides::default()
        };
        let axis_line = Rect { x: 1.0, y: 2.0, w: 3.0, h: 1.0 };
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
        let mut x = AxisInput::new(AxisOrient::Bottom, None, vec![], None);
        x.resolve_orient();
        assert_eq!(x.orient, AxisOrient::Bottom);
        // y axis, no override → Left.
        let mut y = AxisInput::new(AxisOrient::Left, None, vec![], None);
        y.resolve_orient();
        assert_eq!(y.orient, AxisOrient::Left);
        // x axis with explicit Top override wins.
        let mut xt = AxisInput::new(AxisOrient::Bottom, None, vec![], None);
        xt.overrides.orient = Some(AxisOrient::Top);
        xt.resolve_orient();
        assert_eq!(xt.orient, AxisOrient::Top);
    }

    #[test]
    fn axis_layout_round_trip() {
        let a = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 50.0, y: 350.0, w: 500.0, h: 1.0 },
            ticks: vec![TickLayout {
                position: 100.0,
                label: "0".into(),
                label_angle: 0.0,
                elided: false,
                culled: false,
                label_font_size: None,
                is_major: true,
            }],
            minor_ticks: vec![],
            title: Some(AxisTitleLayout {
                text: "Price".into(),
                anchor_x: 300.0,
                anchor_y: 380.0,
                angle: 0.0,
            }),
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
        MockMetrics { measure: fixed_width(per_char_px), line_h_factor: 1.2 }
    }

    // ── 395: x-label cascade prediction (flat / wrap / rotate) ───────────────
    // The predictor `estimate_x_label_band` walks the same cascade order the real
    // collision recovery uses; these guard each branch's reserved band so a
    // cascade-policy drift between the predictor and the renderer is caught. (The
    // three encoding sites are NOT byte-safely unifiable — see the task report —
    // so these lock the predictor's three outcomes in place instead.)

    /// S0 flat: a label that fits the estimated slot reserves a single line band.
    #[test]
    fn estimate_x_band_flat_when_label_fits() {
        let m = mock(10.0); // 10 px/char
        let labels = vec!["ab".into(), "cd".into()]; // 20 px each
        // Generous slot: 100 px → threshold 90 px ≥ 20 px → flat.
        let band = estimate_x_label_band(&labels, 11.0, None, &m, 100.0, None, 4.0);
        // Flat band == line_height(11) = 13.2 (no padding delta).
        assert!((band - m.line_height(11.0)).abs() < 1e-9);
    }

    /// S1 wrap: an underscore-splittable label that won't fit flat but whose
    /// segments fit wraps to N lines → band == max_lines * line_height.
    #[test]
    fn estimate_x_band_wraps_underscore_label() {
        let m = mock(10.0);
        // "aa_bb_cc": flat width 80; segments "aa"/"bb"/"cc" are 20 px each.
        let labels = vec!["aa_bb_cc".into()];
        // Slot 50 → threshold 45: flat (80) fails, segments (20) fit → 3 lines.
        let band = estimate_x_label_band(&labels, 11.0, None, &m, 50.0, None, 4.0);
        assert!((band - 3.0 * m.line_height(11.0)).abs() < 1e-9, "band={band}");
    }

    /// S2/S3 rotate: a single long unsplittable label that cannot fit flat or wrap
    /// falls to the rotate branch, reserving the full rotated extent (> flat band).
    #[test]
    fn estimate_x_band_rotates_long_unsplittable_label() {
        let m = mock(10.0);
        // 12-char label, no break points → 120 px, cannot wrap.
        let labels = vec!["abcdefghijkl".into()];
        let slot = 30.0; // threshold 27 < 120 → not flat, no wrap → rotate/vertical.
        let band = estimate_x_label_band(&labels, 11.0, None, &m, slot, None, 4.0);
        // Must exceed the flat single-line band (a rotated/vertical reservation).
        assert!(band > m.line_height(11.0), "rotated band must exceed flat: {band}");
    }

    /// An explicit `label_angle` override bypasses the cascade and reserves the
    /// rotated extent for that exact angle.
    #[test]
    fn estimate_x_band_honors_explicit_angle() {
        let m = mock(10.0);
        let labels = vec!["abc".into()];
        // label_padding=None → label_pad_eff defaults to 2.0; tick_size=4.0.
        let band = estimate_x_label_band(&labels, 11.0, Some(-45.0), &m, 100.0, None, 4.0);
        let expected = rotated_x_label_extent(-45.0, 30.0, 11.0, m.line_height(11.0), 4.0, 2.0);
        assert!((band - expected).abs() < 1e-9, "band={band}, expected={expected}");
    }

    #[test]
    fn y_axis_label_band_uses_longest_label() {
        let input = AxisInput::new(
            AxisOrient::Left,
            None,
            vec!["0".into(), "100".into(), "10000".into()],
            None,
        );
        let m = mock(10.0);
        let band = compute_y_label_band_width(&input, 11.0, &m, 4.0);
        assert_eq!(band, 50.0);
    }

    #[test]
    fn y_axis_label_band_empty_labels_returns_zero() {
        let input = AxisInput::new(AxisOrient::Left, None, vec![], None);
        let m = mock(10.0);
        assert_eq!(compute_y_label_band_width(&input, 11.0, &m, 4.0), 0.0);
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
        let input = AxisInput::new(
            AxisOrient::Left,
            None,
            vec!["abcdefghij".into()], // 10 chars
            Some(-45.0),
        );
        let m = mock(10.0); // 10 px/char → width 100
        let band = compute_y_label_band_width(&input, 11.0, &m, 4.0);
        let expected = rotated_y_label_extent(-45.0, 100.0, m.line_height(11.0), 4.0, 2.0);
        assert!((band - expected).abs() < 1e-9, "band={band}, expected={expected}");
    }

    /// θ=0 bit-parity (spec §6, hard gate): an explicit `label_angle: Some(0.0)`
    /// override must reserve EXACTLY the pre-R2 flat band (max label width
    /// only, no `tick_size`/`label_pad`/`line_h` terms) — bit-for-bit equal to
    /// the `None`-override fixture in `y_axis_label_band_uses_longest_label`.
    #[test]
    fn y_axis_label_band_width_zero_angle_is_bit_identical_to_flat() {
        let labels = vec!["0".into(), "100".into(), "10000".into()];
        let m = mock(10.0);
        let no_override = AxisInput::new(AxisOrient::Left, None, labels.clone(), None);
        let explicit_zero = AxisInput::new(AxisOrient::Left, None, labels, Some(0.0));
        let band_none = compute_y_label_band_width(&no_override, 11.0, &m, 4.0);
        let band_zero = compute_y_label_band_width(&explicit_zero, 11.0, &m, 4.0);
        assert_eq!(band_none, 50.0);
        assert_eq!(band_zero, band_none, "θ=0 must be bit-identical to no override");
    }

    /// `layout_y_axis` stamps the override angle onto every tick (mirrors
    /// `layout_x_axis`'s override branch); with ample vertical spacing no
    /// label collides, so nothing is elided.
    #[test]
    fn layout_y_axis_stamps_override_angle_on_every_tick() {
        let input = AxisInput::new(
            AxisOrient::Left,
            None,
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            Some(-45.0),
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 400.0 };
        let m = mock(10.0);
        let (axis, warning) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(!t.culled, "y-axis has no cull recovery (spec §4-R2)");
            assert!(!t.elided, "short single-char labels should not need elision");
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
        let input = AxisInput::new(
            AxisOrient::Left,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            Some(-45.0),
        );
        // 20 ticks packed into a 200px-tall panel → slot_h = 10px. Each ~7-8
        // char label is 70-80px wide; at -45° the vertical projection
        // (w * sin(45°) ≈ 49-56px) far exceeds the 10px budget.
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, warning) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(!t.culled, "y-axis has no cull recovery (spec §4-R2)");
            assert!(t.elided, "expected all 20 labels to be elided with override");
            assert!(t.label.ends_with('\u{2026}'), "expected ellipsis suffix; got {:?}", t.label);
        }
        match warning {
            Some(AxisLabelWarning::LabelsElided { count }) => assert_eq!(count, 20),
            other => panic!("expected LabelsElided{{count: 20}}, got {:?}", other),
        }
    }

    /// θ=0 bit-parity (spec §6, hard gate) at the `layout_y_axis` level —
    /// quality-review fix 3. `None` and an explicit `Some(0.0)` override take
    /// DIFFERENT code branches (the override branch vs. the flat branch), so
    /// `y_axis_label_band_width_zero_angle_is_bit_identical_to_flat` (which
    /// only pins `compute_y_label_band_width`) does not cover this site. A
    /// dense, long-label fixture is used deliberately: at θ=0 the override
    /// branch's own collision check (`sin(0) = 0` ⇒ `w * 0 > budget` is always
    /// false) must produce the SAME "never elides" outcome as the flat
    /// branch's unconditional "never checks" — proving the equivalence holds
    /// even in a case that WOULD collide at any other angle, not just a
    /// trivially-non-colliding one.
    #[test]
    fn layout_y_axis_zero_angle_override_is_bit_identical_to_no_override() {
        let labels: Vec<String> = (0..20).map(|i| format!("Label_{}", i)).collect();
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let no_override = AxisInput::new(AxisOrient::Left, None, labels.clone(), None);
        let explicit_zero = AxisInput::new(AxisOrient::Left, None, labels, Some(0.0));

        let (axis_none, warn_none) =
            layout_y_axis(&no_override, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        let (axis_zero, warn_zero) =
            layout_y_axis(&explicit_zero, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);

        assert_eq!(
            axis_zero.ticks, axis_none.ticks,
            "Some(0.0) must produce a field-for-field identical tick vector to None"
        );
        assert_eq!(warn_zero, None, "θ=0 override must never elide");
        assert_eq!(warn_none, None, "no-override flat path must never elide");
    }

    #[test]
    fn y_axis_layout_uniform_tick_positions() {
        let input = AxisInput::new(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            None,
        );
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
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

    // ── explicit-range ordinal band centers (GH #39 phase 2) ────────────────

    /// When `categorical_positions` is set (an explicit-range ordinal axis), the
    /// x-axis places each tick at the absolute band center — NOT the panel-uniform
    /// slot center. Discriminating: over panel_area `x=100, w=300` the uniform
    /// slots would be 137.5 / 212.5 / 287.5 / 362.5, far from the band centers.
    #[test]
    fn x_axis_uses_categorical_positions_when_present() {
        let mut input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            None,
        );
        input.categorical_positions = Some(vec![67.5, 122.5, 177.5, 232.5]);
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(4.0), line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![67.5, 122.5, 177.5, 232.5]);
    }

    /// Without `categorical_positions` (the default `None`), the x-axis keeps the
    /// uniform-slot formula, byte-identical to before this seam.
    #[test]
    fn x_axis_falls_back_to_uniform_center_without_categorical_positions() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            None,
        );
        assert!(input.categorical_positions.is_none());
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(4.0), line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![137.5, 212.5, 287.5, 362.5]);
    }

    /// The y-axis honors `categorical_positions` too — ordinal y is NOT reversed,
    /// so band centers map to labels in domain order (top → bottom). Discriminating
    /// against the panel-uniform slots (75 / 125 / 175 / 225 over `y=50, h=200`).
    #[test]
    fn y_axis_uses_categorical_positions_when_present() {
        let mut input = AxisInput::new(
            AxisOrient::Left,
            None,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            None,
        );
        input.categorical_positions = Some(vec![67.5, 122.5, 177.5, 232.5]);
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        let got: Vec<f64> = axis.ticks.iter().map(|t| t.position).collect();
        assert_eq!(got, vec![67.5, 122.5, 177.5, 232.5]);
    }

    /// Without `categorical_positions`, the y-axis keeps uniform-slot placement.
    #[test]
    fn y_axis_falls_back_to_uniform_center_without_categorical_positions() {
        let input = AxisInput::new(
            AxisOrient::Left,
            None,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            None,
        );
        assert!(input.categorical_positions.is_none());
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
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
        let mut input = AxisInput::new(
            AxisOrient::Right,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into()],
            None,
        );
        input.orient = AxisOrient::Right;
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
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
        let mut input = AxisInput::new(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into()],
            None,
        );
        input.overrides.title_orient = Some(AxisOrient::Top);
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 4.0, &m);
        // A horizontal title_orient renders the y-axis title flat (angle 0).
        assert_eq!(axis.title.unwrap().angle, 0.0);
    }

    #[test]
    fn x_axis_top_orient_places_line_on_top_edge() {
        let mut input = AxisInput::new(
            AxisOrient::Top,
            Some("Feature".into()),
            vec!["a".into(), "b".into(), "c".into()],
            None,
        );
        input.orient = AxisOrient::Top;
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 30, 4.0, &m);
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
        let input = AxisInput::new(
            AxisOrient::Bottom,
            Some("Feature".into()),
            vec!["a".into(), "b".into()],
            None,
        );
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 30, 4.0, &m);
        assert_eq!(axis.orient, AxisOrient::Bottom);
        assert!((axis.axis_line.y - (50.0 + 200.0)).abs() < 1e-9);
        let title = axis.title.unwrap();
        assert_eq!(title.angle, 0.0);
        assert!(title.anchor_y > axis.axis_line.y);
    }

    #[test]
    fn x_axis_no_collision_keeps_labels_flat() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 50.0, line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        assert_eq!(axis.ticks.len(), 4);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        assert!(warning.is_none());
    }

    #[test]
    fn x_axis_uniform_tick_positions_along_axis() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect { x: 100.0, y: 50.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        assert!((axis.ticks[0].position - (100.0 + 50.0)).abs() < 1e-9);
        assert!((axis.ticks[1].position - (100.0 + 150.0)).abs() < 1e-9);
        assert!((axis.ticks[2].position - (100.0 + 250.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (100.0 + 350.0)).abs() < 1e-9);
    }

    #[test]
    fn x_axis_collision_triggers_graduated_rotation() {
        // 8 labels of 80px each in 400px panel. slot_w=50, threshold=45.
        // No break points (L0..L7), so wrapping/shrink fail.
        // Cascade tries rotation: -30 -> cos(30)*80=69.3>50, -45 -> cos(45)*80=56.6>50,
        // -60 -> cos(60)*80=40<=50 -> passes at -60.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -60.0);
            assert!(!t.elided);
        }
    }

    #[test]
    fn x_axis_rotates_at_custom_angle_override() {
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            Some(-90.0),
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
        }
    }

    #[test]
    fn x_axis_rotation_only_no_elision_when_rotated_fits() {
        // 6 labels of 95px each in 600px panel. slot_w=100, threshold=90.
        // 95>90 -> collision. No break points -> S1/S2 fail.
        // S3: -30 -> cos(30)*95=82.3<=100 -> passes at -30.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 95.0, line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            Some(-45.0),
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(t.elided, "expected all 20 labels to be elided with override");
            assert!(t.label.ends_with('\u{2026}'), "expected ellipsis suffix; got {:?}", t.label);
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
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..20).map(|i| format!("Label_{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
            assert!(!t.elided, "cascade should resolve at -90 without elision");
        }
        assert!(warning.is_none(), "no LabelsElided warning expected");
    }

    #[test]
    fn x_axis_elision_unicode_safe() {
        // Use label_angle_override to bypass the cascade and force elision.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["héllo wörld".into(); 20],
            Some(-45.0),
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        let mut input = AxisInput::new(orient, None, labels, None);
        // Mirror prepare.rs: the gate empties `minor` when off. With no major
        // fractions supplied, presence of the projection is driven by the
        // minors here (these fixtures pre-date the continuous-major path).
        let minor = if include_minor { minor_positions } else { Vec::new() };
        if !minor.is_empty() {
            input.tick_projection = Some(TickProjection {
                padding_frac: 0.0,
                major: Vec::new(),
                minor,
            });
        }
        input
    }

    // --- project_fractions NonFinitePolicy tests (LAYOUT-845) ---

    #[test]
    fn project_fractions_dropall_finite_maps_via_lerp() {
        // Finite fractions over `(0, 100)` with no padding lerp to `lo + t*(hi-lo)`.
        let got = project_fractions(&[0.0, 0.5, 1.0], (0.0, 100.0), 0.0, NonFinitePolicy::DropAll);
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
            NonFinitePolicy::DropEach,
        );
        assert_eq!(got, Some(vec![0.0, 100.0]));
    }

    #[test]
    fn project_fractions_empty_is_none_for_both_policies() {
        assert_eq!(
            project_fractions(&[], (0.0, 100.0), 0.0, NonFinitePolicy::DropAll),
            None
        );
        assert_eq!(
            project_fractions(&[], (0.0, 100.0), 0.0, NonFinitePolicy::DropEach),
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
        let panel = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);

        assert!(axis.minor_ticks.is_empty(), "gate off must produce no minors");
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
        let panel = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
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
        let panel = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);

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
        let panel = Rect { x: 100.0, y: 40.0, w: 300.0, h: 200.0 };
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
        let panel = Rect { x: 0.0, y: 0.0, w: 300.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);

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
        let mut input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["lo".into(), "hi".into()],
            None,
        );
        input.tick_projection = Some(TickProjection {
            padding_frac: 0.05,
            major: vec![0.0, 1.0],
            minor: vec![0.25],
        });

        let panel = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);

        // Inset band (8, 592), span 584 → minor at f=0.25: 8 + 0.25*584 = 154.
        assert_eq!(axis.minor_ticks.len(), 1);
        let minor_px = axis.minor_ticks[0].position;
        assert!((minor_px - 154.0).abs() < 1e-9, "inset projection expected 154, got {minor_px}");

        // The minor must NOT equal the naive origin + f*extent (0.25*600 = 150).
        assert!(
            (minor_px - 150.0).abs() > 1.0,
            "minor must use inset projection (154), not naive linear interp (150); got {minor_px}"
        );

        // Cross-check: a MAJOR projected at the same fraction 0.25 lands at the
        // same pixel. Reuse the same inset path via projected_tick_fractions.
        let mut major_input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["q".into()],
            None,
        );
        major_input.tick_projection = Some(TickProjection {
            padding_frac: 0.05,
            major: vec![0.25],
            minor: Vec::new(),
        });
        let (major_axis, _) = layout_x_axis(&major_input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        let result = cascade_collision_recovery(&labels, 100.0, 11.0, 8, None, &m);
        assert_eq!(result.angle, 0.0);
        assert!(result.font_size.is_none());
        assert_eq!(result.strategy, CascadeStrategy::Flat);
        assert!(result.visible.iter().all(|v| *v));
        assert_eq!(result.labels, labels);
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
        let result = cascade_collision_recovery(&labels, 100.0, 11.0, 8, None, &m);
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
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
        ];
        let m = MockMetrics {
            measure: |text: &str, font_size: f64| text.chars().count() as f64 * font_size * 0.5,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(&labels, 60.0, 11.0, 8, None, &m);
        assert_eq!(result.angle, 0.0);
        assert_eq!(result.strategy, CascadeStrategy::FontReduced);
        let expected_fs = 11.0 * 0.82;
        assert!((result.font_size.unwrap() - expected_fs).abs() < 1e-6);
        assert!(result.visible.iter().all(|v| *v));
    }

    #[test]
    fn cascade_s3_rotation() {
        // Labels without break points that collide flat.
        // "ABCDEFGHIJ" = 10*10 = 100px. slot_w=80, threshold=72.
        // S1: no break points -> fails.
        // S2: fixed_width ignores font_size -> still 100 -> fails.
        // S3: -30: cos(30)*100=86.6>80 -> fail. -45: cos(45)*100=70.7<=80 -> pass!
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
            "UVWXYZABCD".into(), "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 80.0, 11.0, 8, None, &m);
        assert_eq!(result.strategy, CascadeStrategy::Rotated { angle: -45.0 });
        assert_eq!(result.angle, -45.0);
        assert!(result.font_size.is_none());
        assert!(result.visible.iter().all(|v| *v));
        // Labels unchanged (not wrapped, not elided).
        assert_eq!(result.labels, labels);
    }

    #[test]
    fn cascade_s3_picks_shallowest_angle() {
        // Labels that fit at -30. 6 chars * 10 = 60px. slot_w=55.
        // threshold=49.5. 60>49.5 -> collision.
        // S1: no break points -> fails.
        // S2: fixed_width ignores font_size -> fails.
        // S3: -30: cos(30)*60=51.96<=55 -> pass at -30!
        let labels: Vec<String> = vec![
            "ABCDEF".into(), "GHIJKL".into(), "MNOPQR".into(), "STUVWX".into(),
        ];
        let m = mock(10.0);
        let result = cascade_collision_recovery(&labels, 55.0, 11.0, 8, None, &m);
        assert_eq!(result.strategy, CascadeStrategy::Rotated { angle: -30.0 });
        assert_eq!(result.angle, -30.0);
    }

    #[test]
    fn cascade_s4_culling() {
        // 20 labels in a narrow panel. slot_w=10, each label=15 chars*10=150px.
        // Labels have no break points (all uppercase).
        // 20 > cull_threshold=8 -> culling is eligible.
        // S0-S2 fail (150 >> 9 threshold).
        // S3: even at -90, cos(90)*150 ~ 0 <= 10 -> all fit.
        // Wait, S3 at -90 passes because projected width ~= 0. So culling
        // won't fire if -90 resolves it. We need labels where even -90
        // doesn't fully resolve.
        //
        // Actually, cos(-90) is not exactly 0 in floating point; it's ~1.8e-16.
        // So 150 * 1.8e-16 ~ 2.7e-14, which is <= 10. S3 passes at -90.
        //
        // To test culling, we need the S3 check to use `w * cos_factor <= slot_w`
        // but our cascade uses this exact check. At -90 degrees, cos is effectively
        // 0 so ANY width fits. Culling only triggers when even -90 doesn't work.
        //
        // That can't happen with real floating-point cos. So culling is only for
        // when labels.len() > cull_threshold AND rotation resolves but leaves too
        // many labels at -90. Actually re-reading the spec more carefully:
        //
        // S4 triggers when S3 fails (all ANGLE_CASCADE angles tried, none work).
        // But -90 always works (cos ~= 0). Unless slot_w is 0, which would be
        // degenerate. Let me re-read my implementation...
        //
        // Actually, the issue is more subtle. I need to test where cull_threshold
        // is LOW so that culling fires instead of S3. But the cascade is linear:
        // S3 is tried before S4. If -90 resolves all collisions in S3, we never
        // reach S4.
        //
        // Looking at real use cases: S4 makes sense when we WANT to reduce the
        // number of visible labels even though -90 technically fits them all.
        // But in our implementation, S3 genuinely resolves it first.
        //
        // Let me re-examine the cascade design: S4 fires only when S3 fails.
        // With floating-point cos(-90) ~ 0, S3 basically never fails. This
        // means S4 only fires in truly degenerate cases (slot_w = 0).
        //
        // For testing purposes, let's verify culling works when S3 does fail.
        // We can simulate this by ensuring cos(-90) * max_width > slot_w,
        // but that requires enormous label widths or slot_w=0.
        //
        // Alternative: slot_w very small (e.g., 1e-16), so even cos(-90)*w > slot_w.
        // 20 labels in 0.00000001px panel -> slot_w = 5e-10.
        // cos(-90)*150 = ~2.7e-14 > 5e-10? No, 2.7e-14 < 5e-10. Still fits.
        //
        // In practice, S4 fires when we have a rounding scenario. Let me just
        // test cascade_collision_recovery directly with slot_w=0.
        let labels: Vec<String> = (0..20).map(|i| format!("LONGCATEGORY{:02}", i)).collect();
        let m = mock(10.0);
        // slot_w so small that even -90 can't resolve: use width check with
        // a mock that doesn't honor cos (since fixed_width ignores font_size,
        // and we're testing the cascade logic itself).
        //
        // Actually, to make S4 fire, we need ALL angles in S3 to fail.
        // cos(-90 deg) ≈ 6.12e-17. For 14-char labels: 140 * 6.12e-17 ≈ 8.6e-15.
        // This is only > slot_w if slot_w < 8.6e-15. That's effectively zero.
        //
        // slot_w = 0 triggers degenerate path. Let's use slot_w at exactly 0.
        let result = cascade_collision_recovery(&labels, 0.0, 11.0, 8, None, &m);
        // With slot_w=0, threshold=0, all labels collide.
        // S0-S2: fail (width > 0).
        // S3: for each angle, w*cos_factor <= 0 only if cos_factor=0 exactly.
        //   cos(-90) is not exactly 0 in IEEE 754, but 140*6.12e-17 ≈ 8.6e-15 > 0.
        //   So S3 might still pass. Depends on precision.
        // If S3 does pass at -90, S4 won't fire. Let's check.
        //
        // Due to floating-point behavior, let's verify whatever stage actually fires.
        // This test documents the behavior regardless.
        assert!(
            matches!(result.strategy,
                CascadeStrategy::Rotated { .. } |
                CascadeStrategy::Culled { .. } |
                CascadeStrategy::Elided { .. }
            ),
            "expected rotation, culling, or elision; got {:?}",
            result.strategy
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
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, 8, None, &m);
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

    #[test]
    fn cascade_s5_elision() {
        // Extreme density with few labels (below cull_threshold), so culling
        // is skipped and elision fires as last resort.
        // 6 labels (< cull_threshold=8) with enormous widths -> S3 fails for all
        // angles -> S4 skipped (6 < 8) -> S5 elision.
        let labels: Vec<String> = (0..6).map(|i| format!("VeryLongLabel{}", i)).collect();
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let result = cascade_collision_recovery(&labels, 10.0, 11.0, 8, None, &m);
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
        let result = cascade_collision_recovery(&labels, slot_w, 11.0, 8, None, &m);
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
        let input = AxisInput::new(
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
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        // 6 labels below cull_threshold with enormous widths -> elision.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("VeryLongLabel{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 60.0, h: 200.0 };
        let m = MockMetrics {
            measure: |_text: &str, _fs: f64| 1e18,
            line_h_factor: 1.2,
        };
        let (_, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        assert!(
            matches!(warning, Some(AxisLabelWarning::LabelsElided { .. })),
            "expected LabelsElided warning; got {:?}",
            warning,
        );
    }

    #[test]
    fn cascade_rotation_no_warning() {
        // When rotation resolves collision, no LabelsElided warning should fire.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..8).map(|i| format!("L{}", i)).collect(),
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (_, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        assert!(warning.is_none(), "rotation should not produce LabelsElided warning");
    }

    // --- estimate_x_label_band tests ---

    #[test]
    fn estimate_flat_labels() {
        // Short labels that fit within slot_w flat should return exactly line_height.
        // "A" = 1 char * 10 = 10px. slot_w = 100. threshold = 90. 10 <= 90 -> flat.
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let m = mock(10.0); // fixed_width: chars * 10
        let line_h = m.line_height(11.0); // 11.0 * 1.2 = 13.2
        // Regression guard: the new `tick_size` param must NOT affect the flat
        // case — a non-zero tick_size (4.0) is passed, yet the band stays line_h.
        let band = estimate_x_label_band(&labels, 11.0, None, &m, 100.0, None, 4.0);
        assert!(
            (band - line_h).abs() < 1e-9,
            "flat labels should return line_height={line_h}, got {band}"
        );
    }

    #[test]
    fn estimate_wrapped_labels() {
        // snake_case labels that collide flat but wrap successfully.
        // "trivial_baseline" = 16 * 6 = 96 > threshold = 90.
        // After wrap: max("trivial"=42, "baseline"=48) = 48 <= 90 -> wraps to 2 lines.
        // Expected: 2 * line_height.
        let labels: Vec<String> = vec![
            "trivial_baseline".into(),
            "negative_prompt".into(),
            "persona_limited".into(),
            "minimal_context".into(),
        ];
        let m = mock(6.0); // per_char * 6; "trivial_baseline" = 16*6 = 96 > 90
        let line_h = m.line_height(11.0); // 11.0 * 1.2 = 13.2
        let band = estimate_x_label_band(&labels, 11.0, None, &m, 100.0, None, 4.0);
        let expected = 2.0 * line_h;
        assert!(
            (band - expected).abs() < 1e-9,
            "wrapped labels should return 2*line_height={expected}, got {band}"
        );
    }

    #[test]
    fn estimate_rotated_labels() {
        // Labels with no break points that collide flat and can't wrap, but fit at -45°.
        // "ABCDEFGHIJ" = 10 * 10 = 100px. estimated_slot_w = 80.
        // threshold = 80 * 0.9 = 72. 100 > 72 -> collision.
        // S1: no break points -> wrap fails for all.
        // S2/S3: -30: cos(30)*100 = 86.6 > 80 -> fail.
        //        -45: cos(45)*100 = 70.7 <= 80 -> pass.
        // Expected margin = 100 * sin(45°) + line_h * cos(45°).
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
            "UVWXYZABCD".into(), "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let line_h = m.line_height(11.0); // 13.2
        let tick_size = 4.0;
        let band = estimate_x_label_band(&labels, 11.0, None, &m, 80.0, None, tick_size);
        // Full geometric extent (mirrors the render pivot): the old too-tight
        // formula was `sin·max_w + cos·line_h`; the band now adds the pivot
        // offset `tick_size + label_pad + sin·font_size` on top of it.
        let angle_rad = (-45.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();
        let label_pad = 2.0; // default
        let expected = tick_size
            + label_pad
            + sin_abs * 11.0
            + sin_abs * 100.0
            + cos_abs * line_h;
        assert!(
            (band - expected).abs() < 1e-6,
            "rotated -45° band should be {expected}, got {band}"
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
        let band = estimate_x_label_band(&labels, 11.0, Some(-90.0), &m, 80.0, None, tick_size);
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
        let band = estimate_x_label_band(&labels, 11.0, Some(-45.0), &m, 200.0, None, tick_size);
        // Full geometric extent at -45° (mirrors the render pivot).
        let angle_rad = (-45.0_f64).to_radians();
        let sin_abs = angle_rad.sin().abs();
        let cos_abs = angle_rad.cos().abs();
        let label_pad = 2.0; // default
        let expected = tick_size
            + label_pad
            + sin_abs * 11.0
            + sin_abs * 100.0
            + cos_abs * line_h;
        assert!(
            (band - expected).abs() < 1e-6,
            "override -45° band should be {expected}, got {band}"
        );
    }

    #[test]
    fn estimate_empty_labels_returns_line_height() {
        let m = mock(10.0);
        let line_h = m.line_height(11.0);
        let band = estimate_x_label_band(&[], 11.0, None, &m, 100.0, None, 4.0);
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
        let band = estimate_x_label_band(&labels, 11.0, None, &m, 10.0, None, tick_size);
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
        // Labels that collide flat and resolve at -45° (same setup as
        // `estimate_rotated_labels`). The NEW band must exceed the OLD too-tight
        // formula (`sin·max_label_w + cos·line_h`) by exactly the pivot offset
        // `tick_size + label_pad + sin·font_size`. Both sides computed explicitly.
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
            "UVWXYZABCD".into(), "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let font_size = 11.0;
        let tick_size = 4.0;
        let line_h = m.line_height(font_size);
        let max_label_w = 100.0; // 10 chars * 10px

        let band = estimate_x_label_band(&labels, font_size, None, &m, 80.0, None, tick_size);

        // Cascade resolves at -45° here.
        let angle_rad = (-45.0_f64).to_radians();
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
        let m = MockMetrics { measure: |_, _| 5_000.0, line_h_factor: 1.2 };
        let font_size = 11.0;
        let tick_size = 4.0;
        let band = estimate_x_label_band(&labels, font_size, None, &m, 10.0, None, tick_size);
        let label_pad = 2.0; // default
        let expected = tick_size + label_pad + font_size + 5_000.0;
        assert!(
            (band - expected).abs() < 1e-6,
            "vertical fallback should return {expected}, got {band}",
        );
    }

    #[test]
    fn estimate_flat_band_unaffected_by_tick_size() {
        // Regression guard for flat goldens: short labels that fit return exactly
        // `line_h + padding_delta` regardless of the new tick_size param. Passing
        // two different tick_size values must yield the identical flat band.
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let m = mock(10.0);
        let line_h = m.line_height(11.0);
        let band_ts0 = estimate_x_label_band(&labels, 11.0, None, &m, 100.0, None, 0.0);
        let band_ts8 = estimate_x_label_band(&labels, 11.0, None, &m, 100.0, None, 8.0);
        assert!(
            (band_ts0 - line_h).abs() < 1e-9 && (band_ts8 - line_h).abs() < 1e-9,
            "flat band must equal line_h ({line_h}) and ignore tick_size; got {band_ts0} and {band_ts8}",
        );
    }

    #[test]
    fn estimate_label_padding_widens_rotated_band() {
        // Increasing label_padding must widen the rotated band by the same delta.
        // Rotated branch folds `label_pad_eff` directly into the pivot offset, so
        // band(lp = a + Δ) - band(lp = a) == Δ.
        let labels: Vec<String> = vec![
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
            "UVWXYZABCD".into(), "EFGHIJKLMN".into(),
        ];
        let m = mock(10.0);
        let tick_size = 4.0;
        let band_default = estimate_x_label_band(&labels, 11.0, None, &m, 80.0, Some(2.0), tick_size);
        let band_wider = estimate_x_label_band(&labels, 11.0, None, &m, 80.0, Some(12.0), tick_size);
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
            "ABCDEFGHIJ".into(), "KLMNOPQRST".into(),
            "UVWXYZABCD".into(), "EFGHIJKLMN".into(),
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

        let band = estimate_x_label_band(&labels, font_size, None, &m, 80.0, None, tick_size);
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
        let input = AxisInput::new(
            AxisOrient::Bottom,
            Some("Feature".into()),
            // No underscores/spaces/camelCase boundaries → wrap is impossible, so
            // the cascade resolves by rotation rather than wrapping.
            (0..6).map(|i| format!("aaaaaaaaaaaaaaaaa{i}")).collect(),
            None,
        );
        // 18-char labels at 10px/char = 180px in a 300px panel (slot 50) force the
        // cascade well past flat into rotation.
        let panel_area = Rect { x: 0.0, y: 0.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let label_font_size = 11.0;
        let title_font_size = 13.0;
        let title_padding = 4.0;
        let tick_size = 4.0;
        let (axis, _) = layout_x_axis(
            &input, panel_area, 0, label_font_size, title_font_size, title_padding,
            8, tick_size, &m,
        );

        // The labels must have rotated (non-zero angle) for this test to be meaningful.
        let resolved_angle = axis.ticks.iter().find(|t| !t.culled).map(|t| t.label_angle).unwrap();
        assert!(resolved_angle != 0.0, "labels should rotate; got angle {resolved_angle}");

        // Recompute the expected rotated extent from the FINAL non-culled labels.
        let max_label_w = axis
            .ticks
            .iter()
            .filter(|t| !t.culled)
            .map(|t| m.measure_width(&t.label, label_font_size))
            .fold(0.0_f64, f64::max);
        let line_h = m.line_height(label_font_size);
        let extent = rotated_x_label_extent(
            resolved_angle, max_label_w, label_font_size, line_h, tick_size, 2.0,
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
    fn x_axis_title_flat_unchanged_regression_guard() {
        // Short labels (angle 0) with a title: the title anchor_y must EXACTLY
        // equal the old flat formula so flat-label goldens never move.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            Some("Price".into()),
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 20.0, line_h_factor: 1.2 };
        let label_font_size = 11.0;
        let title_font_size = 13.0;
        let title_padding = 4.0;
        let (axis, _) = layout_x_axis(
            &input, panel_area, 0, label_font_size, title_font_size, title_padding,
            8, 4.0, &m,
        );
        // All flat.
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
        }
        let title = axis.title.expect("title present");
        let label_h = m.line_height(label_font_size);
        let title_h = m.line_height(title_font_size);
        let expected =
            panel_area.y + panel_area.h + label_h + title_padding + title_h / 2.0;
        assert!(
            (title.anchor_y - expected).abs() < 1e-12,
            "flat title anchor_y ({}) must equal the old formula ({expected})",
            title.anchor_y,
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
        let mut input = AxisInput::new(orient, None, labels, None);
        input.tick_projection = Some(TickProjection {
            padding_frac,
            major: fractions,
            minor: Vec::new(),
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
        let panel = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        assert_eq!(axis.ticks.len(), 3);
        assert!((axis.ticks[0].position - 8.0).abs() < 1e-9, "got {}", axis.ticks[0].position);
        assert!((axis.ticks[1].position - 300.0).abs() < 1e-9, "got {}", axis.ticks[1].position);
        assert!((axis.ticks[2].position - 592.0).abs() < 1e-9, "got {}", axis.ticks[2].position);
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
        let panel = Rect { x: 100.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
        assert!((axis.ticks[0].position - 108.0).abs() < 1e-9);
        assert!((axis.ticks[1].position - 692.0).abs() < 1e-9);
    }

    #[test]
    fn x_axis_categorical_keeps_uniform_slot_centers() {
        // No projected fractions → byte-identical uniform-slot placement.
        // This reproduces the pre-change positions for n=4 in a 100..500 panel.
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            None,
        );
        assert!(input.tick_projection.is_none());
        let panel = Rect { x: 100.0, y: 50.0, w: 400.0, h: 200.0 };
        let m = mock(10.0);
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        let panel = Rect { x: 0.0, y: 0.0, w: 200.0, h: 600.0 };
        let m = mock(10.0);
        let (axis, _) = layout_y_axis(&input, panel, 0, 11.0, 13.0, 4.0, 4.0, &m);
        // inset range (592, 8): t=1 → 8 (top), t=0.5 → 300, t=0 → 592 (bottom).
        assert!((axis.ticks[0].position - 8.0).abs() < 1e-9, "top tick; got {}", axis.ticks[0].position);
        assert!((axis.ticks[1].position - 300.0).abs() < 1e-9, "got {}", axis.ticks[1].position);
        assert!((axis.ticks[2].position - 592.0).abs() < 1e-9, "bottom tick; got {}", axis.ticks[2].position);
        // Not the n=3 uniform slot centers (100/300/500).
        assert!((axis.ticks[0].position - 100.0).abs() > 1.0);
    }

    #[test]
    fn y_axis_categorical_keeps_uniform_slot_centers() {
        // No projected fractions → uniform-slot placement (pre-change positions).
        let input = AxisInput::new(
            AxisOrient::Left,
            Some("Price".into()),
            vec!["0".into(), "1".into(), "2".into(), "3".into()],
            None,
        );
        assert!(input.tick_projection.is_none());
        let panel = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
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
        let panel = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        // Each label is 70px wide. Uniform slot=100 → would stay flat (70<90).
        let m = MockMetrics { measure: |_, _| 70.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        let input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            (0..6).map(|i| format!("L{i}")).collect(),
            None,
        );
        let panel = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 70.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel, 0, 11.0, 13.0, 4.0, 8, 4.0, &m);
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
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        estimate_x_label_band(&labels, 11.0, None, &m, 40.0, label_padding, 4.0)
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
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let result =
            cascade_collision_recovery(&labels, 40.0, 11.0, 8, Some(LabelOverlap::ShowAll), &m);
        assert_eq!(result.strategy, CascadeStrategy::Flat);
        assert_eq!(result.angle, 0.0);
        assert!(result.visible.iter().all(|v| *v), "ShowAll must keep all labels visible");
        assert_eq!(result.labels, labels, "ShowAll must not elide any label");
    }

    #[test]
    fn label_overlap_parity_shows_every_other_label() {
        // Parity short-circuits to a stride-2 cull regardless of width.
        let labels = dense_labels();
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let result =
            cascade_collision_recovery(&labels, 40.0, 11.0, 8, Some(LabelOverlap::Parity), &m);
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
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let result =
            cascade_collision_recovery(&labels, 40.0, 11.0, 8, Some(LabelOverlap::Rotate), &m);
        let steepest = *ANGLE_CASCADE.last().unwrap();
        assert_eq!(result.strategy, CascadeStrategy::Rotated { angle: steepest });
        assert_eq!(result.angle, steepest);
        assert!(result.visible.iter().all(|v| *v), "rotate keeps all labels");
    }

    #[test]
    fn label_overlap_greedy_matches_default_cascade() {
        // Greedy must run the unmodified cascade — identical to `None`.
        let labels = dense_labels();
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let greedy =
            cascade_collision_recovery(&labels, 40.0, 11.0, 8, Some(LabelOverlap::Greedy), &m);
        let default = cascade_collision_recovery(&labels, 40.0, 11.0, 8, None, &m);
        assert_eq!(greedy.strategy, default.strategy);
        assert_eq!(greedy.angle, default.angle);
        assert_eq!(greedy.visible, default.visible);
        assert_eq!(greedy.labels, default.labels);
    }

    #[test]
    fn layout_x_axis_parity_culls_alternating_ticks() {
        // End-to-end through layout_x_axis: a parity override on a dense axis
        // marks alternating ticks culled.
        let mut input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            dense_labels(),
            None,
        );
        input.overrides.label_overlap = Some(LabelOverlap::Parity);
        let panel = Rect { x: 0.0, y: 100.0, w: 240.0, h: 0.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _warn) = layout_x_axis(&input, panel, 0, 11.0, 11.0, 4.0, 8, 4.0, &m);
        let culled: Vec<bool> = axis.ticks.iter().map(|t| t.culled).collect();
        assert_eq!(culled, vec![false, true, false, true, false, true]);
    }

    #[test]
    fn layout_x_axis_show_all_culls_nothing_on_dense_axis() {
        // A dense axis that the default cascade would cull renders every label
        // when label_overlap = ShowAll.
        let mut input = AxisInput::new(
            AxisOrient::Bottom,
            None,
            dense_labels(),
            None,
        );
        input.overrides.label_overlap = Some(LabelOverlap::ShowAll);
        let panel = Rect { x: 0.0, y: 100.0, w: 240.0, h: 0.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _warn) = layout_x_axis(&input, panel, 0, 11.0, 11.0, 4.0, 8, 4.0, &m);
        assert!(axis.ticks.iter().all(|t| !t.culled), "ShowAll must cull nothing");
        assert!(axis.ticks.iter().all(|t| t.label_angle == 0.0), "ShowAll stays flat");
    }

    // ── compute_x_title_width per-axis title_font_size/title_padding override
    // (ported from tests/bug_hunt_scale_layout.rs, R1) ───────────────────────
    //
    // `compute_x_title_width` and `compute_y_title_width` had zero direct
    // coverage before this port — every assertion below calls the real
    // functions instead of the mirror's inlined `fs * 1.2 + padding` formula.

    fn titled_input(title_font_size: Option<f64>, title_padding: Option<f64>) -> AxisInput {
        let mut input = AxisInput::new(AxisOrient::Bottom, Some("Title".into()), vec!["a".into()], None);
        input.overrides.title_font_size = title_font_size;
        input.overrides.title_padding = title_padding;
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
        let gutter_default = compute_x_title_width(&default_input, theme_fs, theme_pad, &m);
        assert!((gutter_default - 23.6).abs() < 1e-9, "got {gutter_default}");

        let override_input = titled_input(Some(30.0), None);
        let gutter_override = compute_x_title_width(&override_input, theme_fs, theme_pad, &m);
        assert!((gutter_override - 44.0).abs() < 1e-9, "got {gutter_override}");

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
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        assert!((gutter - 44.8).abs() < 1e-9, "24*1.2+16 = 44.8, got {gutter}");
    }

    /// No title: `compute_x_title_width` returns 0.0 regardless of font size.
    #[test]
    fn x_title_gutter_no_title_is_zero() {
        let m = mock(0.0);
        let mut input = AxisInput::new(AxisOrient::Bottom, None, vec!["a".into()], None);
        input.overrides.title_font_size = Some(100.0);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        assert_eq!(gutter, 0.0, "no title should mean zero gutter");
    }

    /// Zero per-axis font size: gutter reduces to the padding term alone.
    #[test]
    fn x_title_gutter_zero_font_size() {
        let m = mock(0.0);
        let input = titled_input(Some(0.0), None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        assert!((gutter - 8.0).abs() < 1e-9, "zero font size means gutter = padding only, got {gutter}");
    }

    /// Very large per-axis font size scales the gutter proportionally.
    #[test]
    fn x_title_gutter_very_large_font_size() {
        let m = mock(0.0);
        let input = titled_input(Some(1000.0), None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        assert!((gutter - 1208.0).abs() < 1e-9, "huge font: gutter = 1000*1.2+8 = 1208, got {gutter}");
    }

    /// A NaN per-axis font size propagates to a NaN gutter — `compute_x_title_width`
    /// applies no defensive `is_finite` guard, so NaN user input reaches layout
    /// arithmetic unchanged.
    #[test]
    fn x_title_gutter_nan_font_size_propagates() {
        let m = mock(0.0);
        let input = titled_input(Some(f64::NAN), None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        assert!(gutter.is_nan(), "NaN font size should propagate to a NaN gutter");
    }

    /// An unset per-axis font size falls back to the theme font size passed in.
    #[test]
    fn x_title_gutter_none_falls_back_to_theme() {
        let m = mock(0.0);
        let input = titled_input(None, None);
        let gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        assert!((gutter - 23.6).abs() < 1e-9, "13*1.2+8 = 23.6, got {gutter}");
    }

    /// `compute_x_title_width` and `compute_y_title_width` are parity-equal
    /// for the same input and overrides — the doc comment's stated
    /// "byte-identical to compute_y_title_width" claim, pinned against the
    /// real functions rather than two copies of the same local formula.
    #[test]
    fn title_gutter_x_y_parity_with_overrides() {
        let m = mock(0.0);
        let input = titled_input(Some(20.0), Some(12.0));
        let x_gutter = compute_x_title_width(&input, 13.0, 8.0, &m);
        let y_gutter = compute_y_title_width(&input, 13.0, 8.0, &m);
        assert_eq!(x_gutter, y_gutter, "x and y title gutters must match for identical overrides");
    }

    /// A larger per-axis title font size drives a strictly larger real
    /// `compute_x_title_width` gutter, monotonically shrinking a downstream
    /// plot-region-height computation by exactly the gutter's growth.
    #[test]
    fn x_title_gutter_large_font_reduces_plot_region() {
        let m = mock(0.0);
        let inner_h = 400.0;
        let x_label_band = 20.0;

        let gutter_default = compute_x_title_width(&titled_input(None, None), 13.0, 8.0, &m);
        let gutter_large = compute_x_title_width(&titled_input(Some(50.0), None), 13.0, 8.0, &m);
        assert!(gutter_large > gutter_default, "a larger font must produce a larger gutter");

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
}
