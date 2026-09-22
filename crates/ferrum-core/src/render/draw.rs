//! Per-panel draw context + mark dispatch. Spec §4.5 / §4.6.

use arrow::record_batch::RecordBatch;

use crate::layout::{PanelLayout, ThemeInputs};
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;

use ferrum_scene::{
    MarkBatchKind, SceneNode as FsSceneNode, TooltipContent as FsTooltipContent,
    TooltipField as FsTooltipField,
};

use super::color::{parse_color, with_opacity, Color};
use super::marks::curve::Interpolate;
use super::scale_resolve::{ColorInput, ColorScale, ResolvedScales, ScaleKind};
use super::RenderError;

impl DrawCtx<'_> {
    /// Whether this chart's coordinate system is flipped (`fm.CoordFlip()`).
    ///
    /// Mirrors `marks::rule`'s read of the same flag off the same `spec`.
    pub(crate) fn coord_flipped(&self) -> bool {
        matches!(self.spec.coord, Some(crate::spec::coord::CoordKind::Flip))
    }

    /// Which screen coordinate a series drawn from this batch advances along —
    /// the independent variable [`marks::curve`](super::marks::curve) fits the
    /// dependent one against.
    ///
    /// Normally a series advances along x with its value on y. THREE things put
    /// the value on x and the position/category on y instead:
    ///
    /// - **`CoordFlip`** — [`Encoding::flip_positional`](crate::spec::encoding::Encoding::flip_positional)
    ///   swaps `x`/`y` on the *rendering* encoding, so the drawer receives the
    ///   value on x and the category on y.
    /// - **A horizontal stack** (`Stack(value_axis=X)`, or a horizontal
    ///   composite-mark desugar) — `apply_stack` cumulates the x column and
    ///   stamps `__stack_value_on_x__`.
    /// - **A producer's own declaration** ([`MarkKwargsSpec::value_axis`]) —
    ///   `#172`(2). A desugar that remaps its value onto `x` and emits neither
    ///   a flip nor a stack says so here. `mark_density(orientation=
    ///   "horizontal")` at the default `multiple="layer"` is exactly that
    ///   shape: its spec carries no `coord` and no `position`, so the first two
    ///   producers see nothing, and its rendered spec is indistinguishable from
    ///   an ordinary vertical area whose rows arrived unsorted.
    ///
    /// # The three are NOT peers, and combining them with `||` is a bug
    ///
    /// They answer different questions, in different spaces:
    ///
    /// - The **declaration** is a statement about the RENDERED series — which
    ///   screen axis carries the value, full stop. It is authored pre-flip and
    ///   moved into post-flip space by
    ///   [`LayerPrepared::flip_coords`](crate::render::prepare) alongside the
    ///   channel swap, so by the time it is read here it already accounts for
    ///   the flip.
    /// - **`CoordFlip` and the stack stamp** are statements about
    ///   TRANSFORMATIONS applied to a series whose authored orientation is the
    ///   default, vertical. They say what happened to that default; they do not
    ///   describe the result of a series that was never vertical to begin with.
    ///
    /// So the composition is a PRECEDENCE, not a disjunction: a declaration
    /// answers outright, and the transformations only speak for an undeclared
    /// series.
    ///
    /// A precedence is correct *because this declaration is already post-flip*:
    /// [`LayerPrepared::flip_coords`](crate::render::prepare) swapped it with the
    /// channels, so the flip has been spent by the time it is read here and
    /// `Some(Y)` genuinely means "vertical as rendered". Its sibling
    /// [`resolve_stack_value_on_x`](crate::render::position) answers the same
    /// question for `PositionAdjust::Stack::value_axis`, which arrives AUTHORED
    /// (nothing swaps `position`), and so composes the two as an XOR instead.
    /// The two are one rule — "reconcile a channel-naming declaration with the
    /// flip, exactly once" — applied in the two spaces their inputs live in.
    /// Copying this precedence onto the authored field was the close-review F3
    /// defect: it spends the flip zero times, and a flipped horizontal stack
    /// cumulated its bin column. Read that function's doc before changing either.
    ///
    /// A disjunction shipped briefly and was wrong for exactly one input:
    /// `mark_density(orientation="horizontal").coord(CoordFlip())`. Flipping the
    /// declaration turns its disjunct off, but `coord_flipped()` stays true and
    /// carries the `||` on its own, so the chart rendered horizontal while its
    /// post-flip encoding (`x`=observation, `y`=density) is an ordinary vertical
    /// area — measured, it closed its fill on the panel's LEFT edge where the
    /// plain vertical density with the identical post-flip encoding closes on the
    /// bottom. Under precedence the flipped declaration says `Y` and that is the
    /// answer.
    ///
    /// Any future producer of a horizontal series belongs HERE. Deriving this
    /// per-mark is what went wrong once already: the first attempt enumerated
    /// call sites, knew about the stamp, and did not know about the flip, so
    /// every flipped curve was fitted against the wrong axis while each mark's
    /// const asserted it could not happen. *Inferring* it per-mark is what went
    /// wrong the second time: `#172`(2) first read the answer off the traced
    /// edge's monotonicity, which cannot tell a horizontal density from an
    /// unsorted vertical area — measured, it turned the latter's fill inside
    /// out. A declaration cannot be misread; an inference can.
    ///
    /// A mark whose points are NOT built from the positional encodings this
    /// describes (a `y2` band, whose bottom edge rides the row's own x) must
    /// override this — see `marks::area::BottomEdge::axis`.
    pub(crate) fn series_curve_axis(&self) -> super::marks::curve::CurveAxis {
        use super::marks::curve::CurveAxis;
        use crate::spec::position::StackValueAxis;
        let value_on_x = match self.mark_style.misc.value_axis {
            // Declared, in post-flip space: the whole answer.
            Some(StackValueAxis::X) => true,
            Some(StackValueAxis::Y) => false,
            // Undeclared, so the authored orientation is the default (vertical)
            // and the series is horizontal exactly if something turned it: the
            // user's flip, or a stack that cumulated the x column.
            None => {
                self.coord_flipped()
                    || super::position::BatchPositionMeta::from_batch(self.batch)
                        .stack_value_on_x()
            }
        };
        if value_on_x { CurveAxis::Y } else { CurveAxis::X }
    }

    /// Project a quantitative value to a pixel coordinate, honoring this mark's
    /// `clip=` — the draw-time half of the escape (batch-E T13).
    ///
    /// `clip=False` at the SVG-group level only stops the group from being
    /// clipped; a vertex the scale had already turned into `None` never reaches
    /// the node list for that escape to apply to. So an unclipped batch must
    /// also read its positions through
    /// [`ScaleKind::to_pixel_f64_extrapolated`], which keeps an out-of-domain
    /// point at the pixel its scale's own formula computes rather than dropping
    /// it. The default (`clip` unset or `Some(true)`) calls
    /// [`ScaleKind::to_pixel_f64`], so every clipped mark is byte-identical.
    ///
    /// This lives on `DrawCtx` rather than in each mark because
    /// [`MarkKwargsSpec::clip`](crate::spec::mark_style::MarkKwargsSpec) is a
    /// general mark-style field with no per-mark allowlist on the Rust side:
    /// any spec that sets it — including a layer's own `mark_style` on the wire
    /// — reaches `resolve_mark_style` whatever the mark is. So every mark that
    /// can receive it must mean the same thing by it. When only `line.rs`
    /// carried the gate, an `area` or `ribbon` layer carrying `clip: false` got
    /// the SVG-group escape and kept dropping the very vertices that escape
    /// existed to reveal. (Python's own mark allowlist is narrower today —
    /// `clip=` is surfaced through `mark_function`, which desugars to a *line*
    /// layer — so this closes the seam ahead of the API rather than behind it.)
    ///
    /// A mark that reads positions through this method inherits the whole
    /// feature; one that calls `to_pixel_f64` directly opts out, deliberately.
    /// `ScaleKind::to_pixel_str` has no continuous mapping to extrapolate and
    /// is unaffected either way.
    pub(crate) fn to_pixel(&self, scale: &ScaleKind, value: f64) -> Option<f64> {
        if self.mark_style.misc.clip == Some(false) {
            scale.to_pixel_f64_extrapolated(value)
        } else {
            scale.to_pixel_f64(value)
        }
    }
}

pub struct DrawCtx<'a> {
    pub spec: &'a crate::spec::chart::ChartSpec,
    pub panel: &'a PanelLayout,
    pub theme: &'a ThemeInputs,
    pub scales: &'a ResolvedScales,
    pub batch: &'a RecordBatch,
    pub mark_style: &'a MarkStyle,
}

/// Where one of [`MarkPaint`]'s two constant paints came from.
///
/// This three-state provenance replaced the `{fill,stroke}_is_user_set` +
/// `{fill,stroke}_cleared` boolean pairs (wave 6A T7, `#110`). The pairs could
/// spell a fourth state the domain does not have — "inherited but cleared" —
/// and every reader had to know that `cleared` is only meaningful when
/// `is_user_set` holds; the enum makes the illegal combination unrepresentable
/// and turns the four-field read at `scene_build.rs`'s literal-paint gate into
/// one comparison.
///
/// "User-set" means the value resolved from an explicit `mark_kwargs` override
/// (spec §4.4, 2026-08-28 T4 amendment), never merely inherited from the mark's
/// theme-aware default. An *unparseable* override is neither variant: it fails
/// the whole render with [`RenderError::InvalidColor`] (batch-A Task 8), so the
/// "resolved or refused" reading holds with no third outcome.
///
/// The source is recorded on the *layer-resolved* [`MarkPaint`] (each layer
/// calls [`resolve_mark_style`] independently), so it is correct in both flat
/// and layered charts — unlike reading the raw chart-level `mark_kwargs` off
/// `ChartSpec.mark_style`, which a layered chart's synthetic per-layer
/// `ChartSpec` does NOT override (it stays the chart-level value via
/// `..spec.clone()` in `scene_build.rs::build_panel_mark_batches`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintSource {
    /// Inherited from the mark-aware theme default: the user named no color for
    /// this slot.
    Inherited,
    /// Resolved from an explicit user override that named a color — anything
    /// [`parse_color`] accepts, or the `"theme:label"` theme-lookup sentinel.
    UserColor,
    /// Resolved from the explicit `"none"`/`"transparent"` paint clear.
    ///
    /// The slot's *value* is still
    /// [`color::TRANSPARENT`](super::color::TRANSPARENT) (that is what
    /// [`resolve_paint_color`] resolves the sentinel to, so the clear travels as
    /// an ordinary color no fill fallback can resurrect); this variant is the
    /// *provenance* that [`ClearablePaint::scene_paint`] gates the omitted
    /// attribute on. A paint that merely equals `TRANSPARENT` by value — an
    /// explicit `fill="#00000000"`, or `opacity=0` composed onto black — is
    /// [`UserColor`](PaintSource::UserColor) and keeps serializing
    /// byte-identically (T8 quality-review c2, spec §4.1 2026-09-01 amendment,
    /// spec §7).
    UserClear,
}

impl PaintSource {
    /// The provenance of one `fill=`/`stroke=` override string. Called beside
    /// the [`resolve_paint_color`] that resolves the same string's *value*, so
    /// value and provenance are written in one place and cannot drift.
    fn from_override(value: &str) -> Self {
        if is_paint_clear_sentinel(value) { Self::UserClear } else { Self::UserColor }
    }

    /// `true` for either explicit-override variant. Rule/segment renderers read
    /// this to enforce the precedence "explicit constant stroke > per-row color
    /// encoding > fill fallback" ([`resolve_stroke_color`]), and `text.rs` reads
    /// it to decide whether its constant color fallback is the resolved `fill`
    /// or the theme font color.
    pub fn is_user_set(self) -> bool {
        !matches!(self, Self::Inherited)
    }
}

/// Cross-cutting paint applied to **every** mark family: fill/stroke colors,
/// their provenance, stroke width, mark opacity, dash pattern, and rect/bar
/// corner radius. These are the only style fields read by more than one mark
/// family, so they form the shared base of [`MarkStyle`].
///
/// `fill`/`stroke` carry the resolved *value*; `fill_source`/`stroke_source`
/// carry where it came from. Marks that build a scene paint out of a slot go
/// through [`fill_paint`](MarkPaint::fill_paint) /
/// [`stroke_paint`](MarkPaint::stroke_paint), which pair the two into one
/// [`ClearablePaint`] rather than letting a call site re-pair them by hand.
#[derive(Debug, Clone)]
pub struct MarkPaint {
    pub fill: Color,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub corner_radius: f64,
    pub stroke_dash: Option<Vec<f64>>,
    pub stroke_source: PaintSource,
    pub fill_source: PaintSource,
}

impl MarkPaint {
    /// The constant `fill` slot as one paint: [`ClearablePaint::Cleared`] when
    /// the user cleared it, else the resolved color. Never
    /// [`Absent`](ClearablePaint::Absent) — every mark style resolves a fill.
    ///
    /// **The source wins over the value.** See [`stroke_paint`](Self::stroke_paint)
    /// for why that is safe rather than lossy; the fill case is the simpler one,
    /// since `fill` is a plain `Color` with no absent spelling to discard.
    pub fn fill_paint(&self) -> ClearablePaint {
        match self.fill_source {
            PaintSource::UserClear => ClearablePaint::Cleared,
            _ => ClearablePaint::Paint(self.fill),
        }
    }

    /// The constant `stroke` slot as one paint: [`ClearablePaint::Cleared`] when
    /// the user cleared it, [`Absent`](ClearablePaint::Absent) when this mark's
    /// style resolves no stroke at all, else the resolved color. The
    /// cleared/absent split is load-bearing, not cosmetic — see
    /// [`resolve_effective_stroke`].
    ///
    /// **The source wins over the value**, and that is the one pairing this type
    /// still leaves representable: `stroke` and `stroke_source` are independent
    /// fields, so `UserClear` alongside `stroke: None` can be written by hand
    /// (the crate's own tests assign these fields directly), and this projection
    /// answers `Cleared` for it — shutting the RMARK-04 rescue gate that the
    /// bare `None` would have opened. That is the deliberate reading, not an
    /// oversight: a clear is a *decision about this slot*, and it outranks the
    /// absence of a value under it. It is also lossless, because a cleared
    /// paint carries no color at all — the `None` the projection discards would
    /// have resolved to exactly nothing a reader could have used.
    ///
    /// [`resolve_mark_style`] never produces that pairing: it writes
    /// `Some(TRANSPARENT)` and `UserClear` together, in one place, from one
    /// override string.
    pub fn stroke_paint(&self) -> ClearablePaint {
        match self.stroke_source {
            PaintSource::UserClear => ClearablePaint::Cleared,
            // The one caller of `uncleared` that is NOT `#110` migration debt:
            // the clear is already honored by the arm above, so what reaches
            // here has no clear to drop.
            _ => ClearablePaint::uncleared(self.stroke),
        }
    }
}

/// Text/label-mark overrides (`mark_text`, `mark_label`). `None` = fall back to
/// the theme/hardcoded default at the call site. `rect.rs` also reads
/// `font_size` for its in-cell value labels.
#[derive(Debug, Clone, Default)]
pub struct TextStyle {
    pub font_size: Option<f64>,
    pub font_weight: Option<String>,
    pub align: Option<String>,
    pub baseline: Option<String>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub angle: Option<f64>,
    /// S7: max label length before truncation (text).
    pub limit: Option<usize>,
    /// S11: draw a leader line from label to its anchor (label).
    pub leader_line: Option<bool>,
}

/// Path-shaping overrides shared by `mark_line` and (for `interpolate`) by
/// `mark_area` and `mark_ribbon`. `interpolate` controls curve interpolation;
/// `stroke_cap` / `stroke_join` shape line endpoints/corners. Area and ribbon
/// reuse `interpolate` from this struct (all three marks trace their edges
/// through [`marks::curve`](super::marks::curve)); area's own overrides live in
/// [`AreaStyle`].
#[derive(Debug, Clone, Default)]
pub struct LineStyle {
    /// S1: interpolate (line/area/ribbon). Parsed — and refused — once in
    /// [`resolve_mark_style`], so every drawer downstream holds a method that
    /// is valid by construction. `None` means the user set nothing and the
    /// drawer applies [`Interpolate::default`] (linear).
    pub interpolate: Option<Interpolate>,
    /// S2: stroke_cap (line).
    pub stroke_cap: Option<String>,
    /// S3: stroke_join (line/area).
    pub stroke_join: Option<String>,
}

/// `mark_area`-only overrides: whether to draw the area's top border line and
/// per-segment side borders. Area's `interpolate` lives in [`LineStyle`].
#[derive(Debug, Clone, Default)]
pub struct AreaStyle {
    /// S9: draw the area's top border line.
    pub line_border: Option<bool>,
    /// S10: draw per-segment side borders.
    pub borders: Option<bool>,
}

/// `mark_point`-only overrides: marker area, fill mode, and shape glyph.
#[derive(Debug, Clone)]
pub struct PointStyle {
    /// Marker area in px² (radius derived as `sqrt(point_size / PI)`).
    pub point_size: f64,
    /// S5: `false` = unfilled (stroke-only) marker.
    pub filled: Option<bool>,
    /// S6: constant shape glyph name.
    pub shape: Option<String>,
}

/// Data-grouping overrides. `detail` is a grouping column shared by
/// line/area/polygon (split a single mark into one path per detail value);
/// `cmap` is the polygon-mark continuous colormap name.
#[derive(Debug, Clone, Default)]
pub struct GroupStyle {
    pub detail: Option<String>,
    pub cmap: Option<String>,
}

/// `mark_segment`-only overrides: whether to draw a filled triangle
/// arrowhead at each segment's `(x2, y2)` end, and its size (spec §4 H1,
/// `ferrum-spec.md:690`).
#[derive(Debug, Clone, Default)]
pub struct SegmentStyle {
    /// `None`/unset resolves to `false` at the draw site (`segment.rs`) —
    /// byte-stable, matching every segment rendered before this field
    /// existed.
    pub arrow: Option<bool>,
    /// Arrowhead size in pixels. `None`/unset resolves to `8.0` at the draw
    /// site, the same default `annotate_arrow`'s unlabelled overlay uses.
    pub arrow_size: Option<f64>,
}

/// Per-mark overrides that don't cluster with a single style family:
/// `band_size` (tick/rect band fraction), image-tile `width`/`height`, the
/// `clip` escape, and the declared `value_axis`.
#[derive(Debug, Clone, Default)]
pub struct MiscStyle {
    /// S8: band fraction for tick/rect category bands.
    pub band_size: Option<f64>,
    /// `mark_image` URL-tile width/height (px).
    pub width: Option<f64>,
    pub height: Option<f64>,
    /// Batch E Task 9: `Some(false)` clip-escapes the batch out of the
    /// panel's clip region. `None` resolves to `true` (clipped) at the
    /// `MarkBatch` construction site, matching every mark before this field
    /// existed.
    pub clip: Option<bool>,
    /// `#172`(2): the producer's declaration of which positional channel
    /// carries this mark's value. Resolved straight through from
    /// [`MarkKwargsSpec::value_axis`], whose doc carries the full rationale,
    /// and read by [`DrawCtx::series_curve_axis`]. `None` — every spec written
    /// before this field existed — leaves that answer exactly as it was.
    ///
    /// It rides `MarkStyle` rather than being read off `ctx.spec` because a
    /// layer's `DrawCtx` spec is `ChartSpec { mark, encoding, ..spec.clone() }`
    /// (`render/scene_build.rs`), so `ctx.spec.mark_style` is the CHART's, not
    /// the layer's. `resolve_mark_style` is the per-layer resolution, which
    /// makes this the per-layer answer.
    pub value_axis: Option<crate::spec::position::StackValueAxis>,
}

/// Per-mark resolved style. Fields are populated from theme defaults (mark-aware)
/// and then overridden by any `MarkKwargsSpec` present on the layer or chart.
///
/// The cross-cutting paint every mark reads lives in [`paint`](MarkStyle::paint);
/// each mark family's single-family overrides live in a typed sub-struct
/// ([`text`](MarkStyle::text), [`line`](MarkStyle::line),
/// [`area`](MarkStyle::area), [`point`](MarkStyle::point),
/// [`group`](MarkStyle::group), [`misc`](MarkStyle::misc),
/// [`segment`](MarkStyle::segment)). Per-mark draw
/// functions read only the sub-structs relevant to their family; the other
/// sub-structs carry their (`None`/default) baseline and are ignored.
#[derive(Debug, Clone)]
pub struct MarkStyle {
    pub paint: MarkPaint,
    pub text: TextStyle,
    pub line: LineStyle,
    pub area: AreaStyle,
    pub point: PointStyle,
    pub group: GroupStyle,
    pub misc: MiscStyle,
    pub segment: SegmentStyle,
}

impl MarkStyle {
    /// Theme-driven base style with `fill = mark_color × default_opacity`,
    /// no stroke, no stroke width, and every text/group/area-only override unset.
    /// All per-mark variants in `resolve_mark_style` are 1-5 field overrides
    /// applied on top of this baseline. Matches the prior Tick/Text/Image
    /// arm byte-for-byte.
    fn theme_base(theme: &ThemeInputs) -> Self {
        MarkStyle {
            paint: MarkPaint {
                fill: with_opacity(theme.colors.mark_color, theme.sizes.default_opacity),
                stroke: None,
                stroke_width: 0.0,
                opacity: theme.sizes.default_opacity,
                corner_radius: 0.0,
                stroke_dash: None,
                stroke_source: PaintSource::Inherited,
                fill_source: PaintSource::Inherited,
            },
            text: TextStyle::default(),
            line: LineStyle::default(),
            area: AreaStyle::default(),
            point: PointStyle {
                point_size: theme.sizes.point_size,
                filled: None,
                shape: None,
            },
            group: GroupStyle::default(),
            misc: MiscStyle::default(),
            segment: SegmentStyle::default(),
        }
    }
}

/// Resolve one user-supplied `fill=` / `stroke=` string to a paint color.
///
/// Three shapes are accepted, checked in this order:
/// 1. the `"theme:label"` sentinel — a theme lookup token, not a color, so it
///    is matched *before* parsing (`parse_color("theme:label")` is an error).
///    Matched exactly: it is an internal token emitted verbatim by ferrum's own
///    desugars, never something a user types;
/// 2. the literal `"none"` or `"transparent"` — an explicit paint clear,
///    resolved to [`color::TRANSPARENT`](super::color::TRANSPARENT) so the
///    clear travels as an ordinary color that no downstream fill fallback can
///    resurrect. `"transparent"` is a real CSS Color 4 keyword, so refusing it
///    while advertising CSS-name support would be self-contradicting; both
///    spellings clear paint identically rather than one clearing and the other
///    refusing. Neither enters `parse_color`'s 148-name vocabulary — the clear
///    is a sentinel, not a parsed color. Trimmed and case-folded like every
///    other color spelling, since users type it;
/// 3. anything else — the full CSS vocabulary via [`parse_color`].
///
/// Anything `parse_color` rejects is a typed [`RenderError::InvalidColor`]
/// naming the accepted forms, never a silent skip: an unparseable `fill=` used
/// to leave the mark painted in its theme default with no diagnostic at all.
fn resolve_paint_color(value: &str, theme: &ThemeInputs) -> Result<Color, RenderError> {
    if value == "theme:label" {
        return Ok(theme.colors.label_color);
    }
    if is_paint_clear_sentinel(value) {
        return Ok(super::color::TRANSPARENT);
    }
    parse_color(value).map_err(RenderError::InvalidColor)
}

/// `true` when `value` is the `"none"`/`"transparent"` paint-clear sentinel
/// (trimmed, case-folded) — the same predicate [`resolve_paint_color`] checks
/// ahead of parsing. Exposed separately so [`PaintSource::from_override`] can
/// record *why* a paint resolved to
/// [`color::TRANSPARENT`](super::color::TRANSPARENT) (a genuine user clear) as
/// opposed to merely resolving to that value (a real color that happens to be
/// zero-alpha, e.g. `"#00000000"`), rather than re-deriving it from the resolved
/// `Color` after the fact — which is exactly the by-value ambiguity
/// [`PaintSource::UserClear`] exists to avoid.
fn is_paint_clear_sentinel(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.eq_ignore_ascii_case("none") || trimmed.eq_ignore_ascii_case("transparent")
}

/// Build the mark-aware theme base and then apply any `MarkKwargsSpec` overrides.
///
/// When `overrides` is `None`, the result is identical to the Phase 7 path
/// (pure theme defaults, mark-aware) — goldens remain byte-identical, and the
/// call is infallible in practice since only the override block can refuse.
///
/// String color fields (stroke, fill) resolve through [`resolve_paint_color`]:
/// full CSS vocabulary, the `"theme:label"` sentinel, and the two clearing
/// spellings `"none"`/`"transparent"` (spec §4.1, 2026-09-01 T8 quality-review
/// supersession — `"transparent"` joined `"none"` as a clear, superseding the
/// earlier none-only shape this comment used to describe) honored, and an
/// unparseable value refused as [`RenderError::InvalidColor`]. The refusal
/// reaches the user at the `build_panel_mark_batches` seam (the sole production
/// caller), alongside `validate_mark_encoding`'s.
pub fn resolve_mark_style(
    overrides: Option<&MarkKwargsSpec>,
    theme: &ThemeInputs,
    mark: &Mark,
) -> Result<MarkStyle, RenderError> {
    // Mark-aware deltas from the theme baseline. Only fields that differ
    // from `MarkStyle::theme_base` are written; everything else falls
    // through to the baseline value.
    let mut style = MarkStyle::theme_base(theme);
    match mark {
        Mark::Area | Mark::Ribbon | Mark::Polygon => {
            style.paint.fill = theme.colors.mark_color;
            style.paint.stroke = None;
            style.paint.stroke_width = 0.0;
            style.paint.opacity = theme.sizes.area_opacity;
        }
        Mark::Line => {
            style.paint.fill = theme.colors.mark_color;
            style.paint.stroke = Some(theme.colors.mark_color);
            style.paint.stroke_width = theme.sizes.line_stroke_width;
        }
        Mark::Bar | Mark::Rect => {
            style.paint.corner_radius = theme.sizes.bar_corner_radius;
        }
        Mark::Rule => {
            // Reference-line defaults from theme; non-reference rules
            // (boxplot whiskers, error bars) override via mark_kwargs.
            style.paint.fill = theme.colors.reference_line_color;
            style.paint.stroke = Some(theme.colors.reference_line_color);
            style.paint.stroke_width = theme.sizes.line_stroke_width;
            style.paint.stroke_dash = theme.reference_line.reference_line_dash.clone();
        }
        Mark::Segment => {
            style.paint.fill = theme.colors.mark_color;
            style.paint.stroke = Some(theme.colors.mark_color);
            style.paint.stroke_width = theme.sizes.line_stroke_width;
        }
        Mark::Point => {
            style.paint.opacity = theme.sizes.point_opacity;
        }
        Mark::Tick | Mark::Text | Mark::Image | Mark::Label => {
            // Baseline applies as-is.
        }
        Mark::Arc => {
            style.paint.stroke_width = 0.0;
        }
        Mark::Geoshape => {
            style.paint.fill = theme.colors.mark_color;
            style.paint.stroke = Some(theme.colors.mark_color);
            style.paint.stroke_width = 0.5;
        }
    }

    // --- Apply MarkKwargsSpec overrides (if any) ---
    let Some(o) = overrides else { return Ok(style) };

    if let Some(size) = o.size { style.point.point_size = size; }
    if let Some(opacity) = o.opacity { style.paint.opacity = opacity; }
    if let Some(cr) = o.corner_radius { style.paint.corner_radius = cr; }
    if let Some(sw) = o.stroke_width { style.paint.stroke_width = sw; }
    // Empty vec = clear the dash (solid line); non-empty = set explicitly.
    if let Some(ref dash) = o.stroke_dash {
        style.paint.stroke_dash = if dash.is_empty() { None } else { Some(dash.clone()) };
    }

    // Both paints resolve or refuse — there is no third "silently keep the
    // theme default" outcome, so `*_source` is written unconditionally
    // alongside the value it describes.
    if let Some(ref value) = o.stroke {
        style.paint.stroke = Some(resolve_paint_color(value, theme)?);
        style.paint.stroke_source = PaintSource::from_override(value);
    }
    if let Some(ref value) = o.fill {
        style.paint.fill = resolve_paint_color(value, theme)?;
        style.paint.fill_source = PaintSource::from_override(value);
    }

    // Text-mark-specific fields
    if let Some(fs) = o.font_size { style.text.font_size = Some(fs); }
    if let Some(ref fw) = o.font_weight { style.text.font_weight = Some(fw.clone()); }
    if let Some(ref al) = o.align { style.text.align = Some(al.clone()); }
    if let Some(ref bl) = o.baseline { style.text.baseline = Some(bl.clone()); }
    if let Some(dx) = o.dx { style.text.dx = Some(dx); }
    if let Some(dy) = o.dy { style.text.dy = Some(dy); }
    if let Some(ang) = o.angle { style.text.angle = Some(ang); }

    // Data-grouping fields (detail: line/area/polygon; cmap: polygon)
    if let Some(ref d) = o.detail { style.group.detail = Some(d.clone()); }
    if let Some(ref c) = o.cmap { style.group.cmap = Some(c.clone()); }

    // S1: interpolate — resolves or refuses; there is no third "unrecognized
    // method silently renders straight" outcome (spec §7, no silent fallbacks).
    if let Some(ref i) = o.interpolate { style.line.interpolate = Some(Interpolate::parse(i)?); }
    // S2: stroke_cap
    if let Some(ref sc) = o.stroke_cap { style.line.stroke_cap = Some(sc.clone()); }
    // S3: stroke_join
    if let Some(ref sj) = o.stroke_join { style.line.stroke_join = Some(sj.clone()); }
    // S5: filled
    if let Some(f) = o.filled { style.point.filled = Some(f); }
    // S6: shape (constant)
    if let Some(ref sh) = o.shape { style.point.shape = Some(sh.clone()); }
    // S7: limit
    if let Some(l) = o.limit { style.text.limit = Some(l); }
    // S8: band_size
    if let Some(bs) = o.band_size { style.misc.band_size = Some(bs); }
    // S9: line border on area
    if let Some(lb) = o.line { style.area.line_border = Some(lb); }
    // S10: borders on area
    if let Some(b) = o.borders { style.area.borders = Some(b); }
    // mark_image URL-tile sizing
    if let Some(w) = o.width { style.misc.width = Some(w); }
    if let Some(h) = o.height { style.misc.height = Some(h); }
    // S11: leader_line (label)
    if let Some(ll) = o.leader_line { style.text.leader_line = Some(ll); }
    // Batch E Task 9: clip escape
    if let Some(c) = o.clip { style.misc.clip = Some(c); }
    // Batch H Task 1: arrow / arrow_size (segment)
    if let Some(a) = o.arrow { style.segment.arrow = Some(a); }
    if let Some(sz) = o.arrow_size { style.segment.arrow_size = Some(sz); }
    // `#172`(2): the producer's declared value axis (line family)
    if let Some(va) = o.value_axis { style.misc.value_axis = Some(va); }

    Ok(style)
}

pub(crate) use super::arrow_cast::{
    col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, col_as_str,
};

/// Pre-read per-row SVG metadata columns (tooltip, href, description).
/// Constructed once per draw call; individual mark renderers call
/// `open_metadata`/`close_metadata` around each mark element.
pub struct MetadataColumns {
    pub tooltip_cols: Vec<(String, Vec<Option<String>>)>,
    pub href: Option<Vec<Option<String>>>,
    pub description: Option<Vec<Option<String>>>,
}

impl MetadataColumns {
    /// Read tooltip/href/description columns from the RecordBatch when the
    /// corresponding encoding is present. Falls back to f64 → string
    /// conversion when the column is numeric.
    ///
    /// For numeric columns, the `format` and `format_type` fields on each
    /// tooltip `EncodingSpec` are honored (matching the same logic as text
    /// mark labels). If no format is specified, the default behavior trims
    /// trailing zeros (e.g. `1.5` not `1.5000`).
    ///
    /// `ctx.spec.encoding` reads as "the chart-level encoding" only for a
    /// single-layer (unlayered) chart. For a multi-layer chart, the per-layer
    /// draw loop (`build_panel_mark_batches` in `scene_build.rs`) builds a
    /// **synthetic per-layer `ChartSpec`** whose `encoding` is that layer's
    /// own merged encoding (`LayerPrepared.encoding`, already chart-merged via
    /// `Encoding::inherit_from` — a layer's own `tooltip_fields` wins when
    /// set; only an unset layer inherits the chart-level fallback). So this
    /// function does not need its own per-layer branch: reading
    /// `ctx.spec.encoding.tooltip_fields` here already resolves to the
    /// correct layer's tooltip fields, by construction of the caller's `ctx`
    /// (GH #52 Task 10f).
    pub fn from_ctx(ctx: &DrawCtx) -> Self {
        use crate::render::format::{format_numeric, format_time, format_with_spec};

        // Collect tooltip columns: use tooltip_fields if present, fall back to single tooltip.
        let tooltip_specs: Vec<&crate::spec::encoding::EncodingSpec> =
            if let Some(fields) = ctx.spec.encoding.tooltip_fields.as_ref() {
                fields.iter().collect()
            } else if let Some(t) = ctx.spec.encoding.tooltip.as_ref() {
                vec![t]
            } else {
                vec![]
            };

        // Read a single tooltip column, applying format/format_type from the spec.
        let read_col_with_spec = |field: &str, fmt: Option<&str>, fmt_type: Option<&str>| -> Option<Vec<Option<String>>> {
            // ONE resolution per column (`#147`), `TimeAutodetect::Denied`:
            // a tooltip column holds no scale to fall back on, so an untyped
            // `%`-bearing spec stays a d3 numeric spec. A resolved strftime
            // PATTERN is applied rather than dropped in favor of the
            // spacing-keyed default, matching the colorbar and text-mark
            // surfaces for the identical pair.
            let kind = crate::render::format::FormatKind::resolve(
                fmt,
                fmt_type,
                crate::render::format::TimeAutodetect::Denied,
            );
            col_as_str(ctx.batch, field)
                .ok()
                .or_else(|| {
                    col_as_f64(ctx.batch, field).ok().map(|vals| {
                        // For time formatting, compute spacing from actual data range
                        // rather than hardcoding 1 day, so sub-day precision is preserved.
                        let spacing_ms: i64 = if matches!(kind, Some(crate::render::format::FormatKind::TemporalDefault)) {
                            let finite: Vec<f64> = vals.iter()
                                .filter_map(|v| v.filter(|f| f.is_finite()))
                                .collect();
                            if finite.len() >= 2 {
                                let lo = finite.iter().cloned().fold(f64::INFINITY, f64::min);
                                let hi = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let range_ms = (hi - lo).abs();
                                // Estimate spacing as range / (n - 1) for n points,
                                // clamped to at least 1 ms.
                                ((range_ms / (finite.len() - 1) as f64).round() as i64).max(1)
                            } else {
                                crate::render::format::DAY_MS // fallback: 1 day
                            }
                        } else {
                            0 // unused: only `TemporalDefault` reads the spacing
                        };
                        vals.into_iter()
                            .map(|v| v.map(|f| {
                                match &kind {
                                    Some(crate::render::format::FormatKind::Strftime(pattern)) => {
                                        crate::render::format::format_time_spec(f as i64, pattern)
                                    }
                                    // Time-typed but naming no pattern: the
                                    // data-derived spacing default, which is
                                    // why this column cannot simply call
                                    // `format_value_with_spec` (that fixes the
                                    // spacing at one day).
                                    Some(crate::render::format::FormatKind::TemporalDefault) => {
                                        format_time(f as i64, spacing_ms)
                                    }
                                    // D3, or the ordinal preset whatever
                                    // `fmt_type` says (`#147`).
                                    Some(
                                        crate::render::format::FormatKind::D3(_)
                                        | crate::render::format::FormatKind::Ordinal,
                                    ) => format_with_spec(f, fmt),
                                    None => format_numeric(f),
                                }
                            }))
                            .collect()
                    })
                })
        };

        let tooltip_cols: Vec<(String, Vec<Option<String>>)> = tooltip_specs
            .iter()
            .filter_map(|e| {
                let fmt = e.format.as_deref();
                let fmt_type = e.format_type.as_deref();
                // The tooltip's DISPLAYED name prefers the channel's `title`
                // over its `field`. This matters when `field` is an internal
                // rename sentinel (e.g. `Chart.__add__`'s `__rhs_<n>` collision
                // suffix, GH #71): the original column name is stamped onto
                // `title` for presentation, while `field` must stay the
                // renamed column so the read below still finds the right data.
                let name = e.title.clone().unwrap_or_else(|| e.field.clone());
                read_col_with_spec(&e.field, fmt, fmt_type).map(|col| (name, col))
            })
            .collect();

        let href = ctx.spec.encoding.href.as_ref().and_then(|e| {
            col_as_str(ctx.batch, &e.field).ok()
        });
        let description = ctx.spec.encoding.description.as_ref().and_then(|e| {
            col_as_str(ctx.batch, &e.field)
                .ok()
                .or_else(|| {
                    col_as_f64(ctx.batch, &e.field).ok().map(|vals| {
                        vals.into_iter()
                            .map(|v| v.map(format_numeric))
                            .collect()
                    })
                })
        });
        MetadataColumns { tooltip_cols, href, description }
    }

}

pub fn x_field<'a>(_ctx: &'a DrawCtx, spec: &'a crate::spec::chart::ChartSpec) -> Option<&'a str> {
    spec.encoding.x.as_ref().map(|e| e.field.as_str())
}
pub fn y_field<'a>(_ctx: &'a DrawCtx, spec: &'a crate::spec::chart::ChartSpec) -> Option<&'a str> {
    spec.encoding.y.as_ref().map(|e| e.field.as_str())
}
pub fn color_field<'a>(_ctx: &'a DrawCtx, spec: &'a crate::spec::chart::ChartSpec) -> Option<&'a str> {
    spec.encoding.color.as_ref().map(|e| e.field.as_str())
}

// ── Scene-graph path (11a) ──────────────────────────────────────────

pub struct MarkBuildResult {
    pub kind: MarkBatchKind,
    pub nodes: Vec<FsSceneNode>,
    pub data_indices: Option<Vec<usize>>,
    pub tooltips: Option<Vec<FsTooltipContent>>,
    pub hrefs: Option<Vec<Option<String>>>,
    pub descriptions: Option<Vec<Option<String>>>,
    /// Per-node `(angle_start, angle_end)` ground truth for a polar wedge
    /// (`MarkBatchKind::Arc` only — `arc.rs`/`bar.rs::build_polar`), threaded
    /// straight onto `MarkBatch::polar_spans` (batch E Task 2, cycle 5).
    /// `None` for every other mark.
    pub polar_spans: Option<Vec<(f64, f64)>>,
    /// Per-node `(x, y)` exact data anchor, threaded straight onto
    /// `MarkBatch::glyph_anchors` (#230-3) — see that field's doc for why it
    /// exists. Populated only by the `BreakPlacement::MoveGlyph`/
    /// `MoveGlyphAndConnector` builders (`point.rs`, `tick.rs`, `text.rs`,
    /// `image.rs`, `label.rs`); `None` for every other mark.
    pub glyph_anchors: Option<Vec<(f64, f64)>>,
    /// Why `nodes` is empty, when it is (#118) — spec §4's silent-empty
    /// policy. `None` covers every LEGITIMATE empty a mark builder can
    /// produce: a zero-row input batch, a documented per-row null/non-finite/
    /// out-of-domain skip, or a post-filter facet panel (none of which reach
    /// a builder with rows to lose in the first place, or degrade the same
    /// way regardless of input) — every builder that constructs its own
    /// `nodes: vec![]` on a non-empty, presence-legal input still passes
    /// `None` here unless it explicitly names a reason via
    /// [`empty_because`](Self::empty_because). `Some` names the specific
    /// channel, column, or scale-kind mismatch responsible for a GENUINE
    /// defect — a presence-legal channel combination (it already passed
    /// `validate_mark_encoding`) that still can't draw anything, which is
    /// otherwise indistinguishable from a rendering bug. Read ONLY at
    /// `scene_build`'s `dispatch_mark_build` call site, the sole place a
    /// `MarkBuildResult`'s emptiness becomes a `RenderWarning` — see
    /// `RenderWarning::MarkBuildEmpty`.
    pub empty_reason: Option<EmptyReason>,
}

/// A ready-to-surface sentence fragment naming why a mark builder produced
/// zero nodes — see [`MarkBuildResult::empty_reason`]. A plain `String`
/// wrapper, not a variant enum: the handful of genuine cases found so far
/// (a segment missing `x2`/`y2`, an image with neither a `url` channel nor
/// Raster-transform data, a rect needing an ordinal axis it didn't get) are
/// heterogeneous enough that a shared field shape (`channel`/`column`/`kind`)
/// would either force an awkward common denominator or grow a variant per
/// mark; each call site already knows the exact true sentence to write.
#[derive(Debug, Clone, PartialEq)]
pub struct EmptyReason(pub String);

impl MarkBuildResult {
    pub fn empty(kind: MarkBatchKind) -> Self {
        Self { kind, nodes: vec![], data_indices: Some(vec![]), tooltips: None, hrefs: None, descriptions: None, polar_spans: None, glyph_anchors: None, empty_reason: None }
    }

    /// Like [`empty`](Self::empty), but names why — for a builder that
    /// discovers a genuine defect (see [`EmptyReason`]'s doc) rather than a
    /// legitimate exemption.
    pub fn empty_because(kind: MarkBatchKind, reason: impl Into<String>) -> Self {
        Self { empty_reason: Some(EmptyReason(reason.into())), ..Self::empty(kind) }
    }
}

/// Return type for metadata build helpers: (tooltips, hrefs, descriptions).
///
/// The type alias avoids the "very complex type" clippy lint on every helper
/// that returns the three optional per-node metadata vectors.
pub type MetadataOutput = (
    Option<Vec<FsTooltipContent>>,
    Option<Vec<Option<String>>>,
    Option<Vec<Option<String>>>,
);

impl MetadataColumns {
    /// Build tooltip/href/description vectors **aligned to the emitted nodes**.
    ///
    /// `data_indices` is the source-row index for each emitted node, in node
    /// order.  This method gathers exactly those rows from each metadata column,
    /// producing output vectors of the same length as `data_indices`.  The result
    /// is node-order aligned: node `j` receives `data_indices[j]`'s metadata.
    ///
    /// This is the single metadata entry point for all mark builders.
    /// Builders that emit every row in order pass `0..n_rows`; builders that
    /// skip rows, emit multiple nodes per row, or group rows pass a custom
    /// index vector from their [`MarkNodes`](crate::render::mark_nodes::MarkNodes)
    /// accumulator.
    pub fn build_metadata_for_indices(
        &self,
        data_indices: &[usize],
    ) -> MetadataOutput {
        let tooltips = if self.tooltip_cols.is_empty() {
            None
        } else {
            Some(
                data_indices.iter().map(|&row_idx| FsTooltipContent {
                    fields: self.tooltip_cols.iter()
                        .map(|(name, col)| FsTooltipField {
                            name: name.clone(),
                            value: col.get(row_idx).and_then(|v| v.clone()).unwrap_or_default(),
                        })
                        .collect(),
                }).collect(),
            )
        };
        let hrefs = self.href.as_ref().map(|col| {
            data_indices.iter().map(|&row_idx| col.get(row_idx).and_then(|v| v.clone())).collect()
        });
        let descriptions = self.description.as_ref().map(|col| {
            data_indices.iter().map(|&row_idx| col.get(row_idx).and_then(|v| v.clone())).collect()
        });
        (tooltips, hrefs, descriptions)
    }
}

pub fn to_scene_color(c: Color) -> ferrum_scene::Color {
    ferrum_scene::Color { r: c.red, g: c.green, b: c.blue, a: c.alpha }
}

/// One paint slot on its way into a scene node: the color to paint, or the
/// reason there is none.
///
/// This is the typed replacement (wave 6A T7, `#110`) for the `(Color, bool)` /
/// `(Option<Color>, bool)` pairs that [`resolve_fill_color`],
/// [`resolve_effective_stroke`] and the `to_scene_fill_stroke*` constructors
/// threaded as unlabeled positional booleans, and for the `normalize_scene_paint`
/// gate they fed. A slot's value and its cleared-ness travel together, so a
/// fallback can no longer substitute one paint's color while keeping the other's
/// provenance.
///
/// [`Cleared`](Self::Cleared) and [`Absent`](Self::Absent) both serialize to no
/// attribute, which is the *whole* observable difference between them today.
/// They stay distinct for two reasons: they are different facts at the call site
/// (this mark has no stroke at all, versus the user asked for none), and
/// [`resolve_effective_stroke`]'s stroke-width rescue fires on `Absent` ONLY —
/// NF-A7 (spec §4.1), an explicit `stroke="none"` must beat the rescue rather
/// than be repainted in the fill color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClearablePaint {
    /// Paint this color.
    Paint(Color),
    /// The user cleared this slot with the `"none"`/`"transparent"` sentinel
    /// ([`PaintSource::UserClear`]). It carries no color and serializes as
    /// nothing; see [`scene_paint`](Self::scene_paint).
    Cleared,
    /// This slot has no paint at all: a mark style that resolves no stroke, a
    /// hollow point's fill, a line's absent fill.
    Absent,
}

impl ClearablePaint {
    /// A never-cleared paint from an `Option<Color>`: `Some` paints, `None` is
    /// [`Absent`](Self::Absent).
    ///
    /// This is the projection for a slot that carries no provenance of its own,
    /// and it has exactly **one** caller: [`MarkPaint::stroke_paint`], which
    /// reaches it only after answering the clear. T7 left it as the marker for
    /// the five mark builders (`area`/`arc`/`geoshape`/`polygon`/`line`) that had
    /// not yet threaded provenance — a cleared constant still serialized as
    /// `rgba(0,0,0,0.000)` there; T12 migrated all five, so a NEW call here is a
    /// site dropping a clear on the floor rather than a known-pending one. Prefer
    /// [`MarkPaint::stroke_paint`] / [`MarkPaint::fill_paint`], which pair a slot
    /// with its own provenance.
    pub fn uncleared(c: Option<Color>) -> Self {
        match c {
            Some(c) => Self::Paint(c),
            None => Self::Absent,
        }
    }

    /// The scene color to serialize, or `None` to omit the attribute entirely.
    ///
    /// A **user-cleared** paint collapses to `None` here so it never serializes
    /// as an explicit `rgba(0,0,0,0.000)` attribute in SVG or as a zero-alpha
    /// color in interactive scene JSON — both outputs walk the same
    /// [`ferrum_scene::SceneGraph`], so this has to happen at construction,
    /// before a `FillStroke`/paint field is built, not at either output's
    /// emission step.
    ///
    /// The gate is *provenance*, never a value test on the color (T8
    /// quality-review c2, spec §4.1 2026-09-01 amendment). Gating on
    /// `c == TRANSPARENT` instead — the pre-c2 shape of this code — aliased any
    /// resolved paint that merely *equals* `color::TRANSPARENT` by value
    /// (`fill="#00000000"`, or `opacity=0` composed onto black) with a genuine
    /// `"none"`/`"transparent"` clear, silently moving those charts' SVG bytes
    /// and violating spec §7's hex-only byte-identity invariant. Only the
    /// literal sentinel matched at the mark-style boundary produces
    /// [`Cleared`](Self::Cleared), so a same-valued but never-cleared paint keeps
    /// serializing exactly as before.
    pub fn scene_paint(self) -> Option<ferrum_scene::Color> {
        match self {
            Self::Paint(c) => Some(to_scene_color(c)),
            Self::Cleared | Self::Absent => None,
        }
    }

    /// Substitute `fallback` for [`Absent`](Self::Absent), leaving both
    /// [`Paint`](Self::Paint) and [`Cleared`](Self::Cleared) as they are.
    ///
    /// For a slot whose fallback is another slot: `geoshape`'s line stroke
    /// falls back to its fill, and if *that* slot was cleared the line paints
    /// nothing (`#110`, spec §4.1).
    pub fn or(self, fallback: Self) -> Self {
        match self {
            Self::Absent => fallback,
            other => other,
        }
    }

    /// [`or`](Self::or) with a plain color as the fallback.
    ///
    /// For a caller whose slot has a hardcoded default but must still honor a
    /// clear: point's `cross`/`vline`/`hline` glyphs paint their lines in the
    /// *fill* slot and default to black when a mark resolves no fill, but a fill
    /// the user cleared must paint nothing at all rather than black or a
    /// zero-alpha stroke.
    pub fn or_paint(self, fallback: Color) -> Self {
        self.or(Self::Paint(fallback))
    }

    /// `resolved` as this slot's paint — unless the slot was **cleared**, which
    /// outranks it.
    ///
    /// This is the one precedence rule `#110` turns on (spec §4.1): a color
    /// resolved from somewhere other than the constant mark style — a color
    /// channel's scale lookup, a per-row column, a group key — paints, but a
    /// paint the user explicitly cleared with `"none"` paints nothing, however
    /// the other source resolved. Before this, every such site rebuilt the
    /// resolved color as a bare `Color` and the clear was simply lost.
    ///
    /// [`Absent`](Self::Absent) paints `resolved` like [`Paint`](Self::Paint)
    /// does: an absent slot is not a decision against painting, and each
    /// caller's own fallback chain has already accounted for it.
    pub fn cleared_or(self, resolved: Color) -> Self {
        match self {
            Self::Cleared => Self::Cleared,
            Self::Paint(_) | Self::Absent => Self::Paint(resolved),
        }
    }

    /// Apply `f` to a [`Paint`](Self::Paint)'s color — composing per-row opacity,
    /// typically. [`Cleared`](Self::Cleared) and [`Absent`](Self::Absent) pass
    /// through: neither has a color to compose onto, and both serialize to
    /// nothing either way.
    pub fn map(self, f: impl FnOnce(Color) -> Color) -> Self {
        match self {
            Self::Paint(c) => Self::Paint(f(c)),
            other => other,
        }
    }
}

/// The scalar paint channels of a [`FillStroke`](ferrum_scene::FillStroke),
/// as one named value.
///
/// These six travelled as bare positionals through
/// [`to_scene_fill_stroke`] — five of them `f64`, four of them adjacent — where
/// nothing but argument order distinguished `opacity` from `fill_opacity` from
/// `angle`. Naming them at the call site is the point (wave 6A T12, carried from
/// T7's quality review); it is also what let the constructor drop from 8
/// arguments to 3 and shed its `clippy::too_many_arguments` suppression.
///
/// Build with [`new`](Self::new) for the common case — a mark that binds none of
/// the per-row opacity/angle channels — or a struct literal to set them.
#[derive(Debug, Clone, Copy)]
pub struct PaintChannels<'d> {
    pub stroke_width: f64,
    /// Element opacity, composed over both paints by the output walkers.
    pub opacity: f64,
    pub stroke_dash: Option<&'d [f64]>,
    /// Per-element `fill-opacity`, distinct from `opacity`. Neutral: `1.0`.
    pub fill_opacity: f64,
    /// Per-element `stroke-opacity`, distinct from `opacity`. Neutral: `1.0`.
    pub stroke_opacity: f64,
    /// Rotation in degrees about the element's anchor. Neutral: `0.0`.
    pub angle: f64,
}

impl<'d> PaintChannels<'d> {
    /// The width/opacity/dash a mark always has, with `fill_opacity`,
    /// `stroke_opacity` and `angle` at their neutral values (`1.0`, `1.0`,
    /// `0.0`).
    ///
    /// This is the shape of every mark that does not bind those three channels
    /// (legend swatches, axis fills, strip backgrounds, area borders,
    /// arc/geoshape/polygon fills), and replaces the former
    /// `to_scene_fill_stroke` / `to_scene_fill_stroke_full` pair: the
    /// convenience wrapper's whole job was defaulting those three fields, which
    /// is now this constructor's job, so one constructor serves both call
    /// shapes.
    pub fn new(stroke_width: f64, opacity: f64, stroke_dash: Option<&'d [f64]>) -> Self {
        Self {
            stroke_width,
            opacity,
            stroke_dash,
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            angle: 0.0,
        }
    }
}

/// Build a [`FillStroke`](ferrum_scene::FillStroke) from two paints and their
/// scalar channels.
///
/// Each paint arrives as one [`ClearablePaint`] rather than as a color plus an
/// unlabeled trailing boolean, and the channels arrive as one
/// [`PaintChannels`] rather than as five bare `f64` positionals.
///
/// A [`ClearablePaint::Cleared`] paint (provenance from [`MarkPaint`], never a
/// value test) serializes as nothing via [`ClearablePaint::scene_paint`], so it
/// never emits an explicit `rgba(0,0,0,0.000)`. A paint that merely *resolves
/// to* `color::TRANSPARENT` without having been cleared (`fill="#00000000"`, or
/// `opacity=0` composed onto black) keeps serializing byte-identically, per
/// spec §7.
///
/// Note the fill and stroke asymmetry in the *emitted* SVG, which lives in
/// `svg.rs::push_fill_stroke` rather than here: an absent stroke omits the
/// attribute, while an absent fill emits an explicit `fill="none"`, because
/// SVG's default fill is black and omitting the attribute would paint it.
pub fn to_scene_fill_stroke(
    fill: ClearablePaint,
    stroke: ClearablePaint,
    channels: PaintChannels<'_>,
) -> ferrum_scene::FillStroke {
    ferrum_scene::FillStroke {
        fill: fill.scene_paint(),
        stroke: stroke.scene_paint(),
        stroke_width: channels.stroke_width,
        opacity: channels.opacity,
        stroke_dash: channels.stroke_dash.map(|d| d.to_vec()),
        stroke_opacity: channels.stroke_opacity,
        fill_opacity: channels.fill_opacity,
        angle: channels.angle,
    }
}

/// Build a [`StrokeStyle`](ferrum_scene::StrokeStyle) for a stroke-only node
/// (`Line`, `Polyline`, gridlines, axis rules).
///
/// `paint` is a [`ClearablePaint`] for the same reason
/// [`to_scene_fill_stroke`]'s two paints are: a user-cleared stroke must reach
/// the scene as *no* color, not as a zero-alpha one. Before `#110` this
/// constructor took a bare `Color` and `StrokeStyle.color` was non-optional, so
/// every stroke-only mark (line/rule/tick/segment, point's cross/vline/hline
/// glyphs) serialized a cleared paint as `rgba(0,0,0,0.000)`; chrome callers
/// that always have a color pass [`ClearablePaint::Paint`] explicitly.
pub fn to_scene_stroke(
    paint: ClearablePaint,
    width: f64,
    opacity: f64,
    dash: Option<&[f64]>,
    cap: Option<&str>,
    join: Option<&str>,
) -> ferrum_scene::StrokeStyle {
    ferrum_scene::StrokeStyle {
        color: paint.scene_paint(),
        width,
        opacity,
        dash: dash.map(|d| d.to_vec()),
        stroke_opacity: opacity,
        stroke_cap: cap.and_then(parse_stroke_cap),
        stroke_join: join.and_then(parse_stroke_join),
    }
}

/// Resolve a single row's per-row stroke color from the constant mark style and
/// an optional color-encoding color for that row.
///
/// Precedence (preserved byte-for-byte from commit a450ef1 — explicit constant
/// stroke beats inherited color encoding):
/// - `stroke_source.is_user_set()` → the explicit user `stroke=` wins; fall back
///   to fill when `stroke` is somehow `None`.
/// - otherwise → per-row color encoding wins, then the theme stroke, then fill.
///
/// This resolves the stroke's *value* only. Mark renderers call
/// [`resolve_stroke_paint`], which pairs that value with the slot's provenance
/// so a cleared stroke reaches the scene as a clear rather than as a present,
/// invisible color.
///
/// The `unwrap_or(fill)` rescue is for a user-set stroke that is somehow
/// `None`; it does NOT fire for an explicit `stroke="none"`, which
/// `resolve_mark_style` resolves to `Some(`[`color::TRANSPARENT`](super::color::TRANSPARENT)`)`
/// — a present, invisible stroke. A cleared stroke therefore stays cleared here
/// too, with no extra case.
pub(crate) fn resolve_stroke_color(ms: &MarkStyle, row_color: Option<Color>) -> Color {
    let p = &ms.paint;
    if p.stroke_source.is_user_set() {
        p.stroke.unwrap_or(p.fill)
    } else {
        row_color.or(p.stroke).unwrap_or(p.fill)
    }
}

/// One row's stroke **paint** for a stroke-only mark: [`resolve_stroke_color`]'s
/// value under the slot's own provenance.
///
/// Used by rule/tick/segment (and any axis-aligned/diagonal line mark) so the
/// precedence lives in one place instead of being copy-pasted per renderer. A
/// `stroke="none"` returns [`ClearablePaint::Cleared`], which serializes as no
/// `stroke` attribute at all — where before `#110` the same clear travelled as
/// `Some(TRANSPARENT)` and serialized as `rgba(0,0,0,0.000)`.
pub(crate) fn resolve_stroke_paint(ms: &MarkStyle, row_color: Option<Color>) -> ClearablePaint {
    ms.paint.stroke_paint().cleared_or(resolve_stroke_color(ms, row_color))
}

/// Resolve a single row's fill color from the color scale + that row's encoded
/// value, falling back to the constant mark-style `fill` when no color encoding
/// applies.
///
/// The [`ColorInput::Numeric`] branch (continuous and discretizing scales)
/// consumes the numeric value directly via `lookup_f64` (correct-by-
/// construction: no `f64 → String → f64` round-trip), matching the point
/// renderer. The [`ColorInput::Category`] branch maps the row's category string
/// via `lookup`. `scale` is `None` when the chart has no color scale.
///
/// A scale lookup that hits paints the looked-up color, which is never itself a
/// user paint-clear however it happens to render (T8 quality-review c2) — but a
/// **cleared** fallback outranks the lookup and stays cleared
/// ([`ClearablePaint::cleared_or`], spec §4.1): `fill="none"` under a color
/// channel paints nothing rather than the channel's palette color. A miss, or an
/// absent scale, returns `fallback` unchanged, clear and all, so the constant
/// paint's provenance survives the lookup either way.
pub(crate) fn resolve_fill_color(
    scale: Option<&ColorScale>,
    cat_value: Option<&str>,
    num_value: Option<f64>,
    fallback: ClearablePaint,
) -> ClearablePaint {
    let Some(s) = scale else { return fallback };
    let looked_up = match s.input() {
        ColorInput::Numeric => num_value.filter(|v| v.is_finite()).and_then(|v| s.lookup_f64(v)),
        ColorInput::Category => cat_value.and_then(|v| s.lookup(v)),
    };
    match looked_up {
        Some(c) => fallback.cleared_or(c),
        None => fallback,
    }
}

/// [`resolve_fill_color`] for a mark that resolves one paint per **group**
/// rather than per row: the group key is a single string, so the numeric branch
/// reads that same string through `str::parse`.
///
/// **Callers: `area`, and `polygon`'s categorical branch.** `line` is
/// group-keyed too and rolled the same chain, but deliberately stayed OFF this
/// helper: its fallback differs on a scale *miss* (it falls back to `fill`, not
/// to the theme stroke), so moving it would change output for charts carrying no
/// clear at all. `polygon`'s *quantitative* branch is likewise still off it, for
/// a different reason — it never consults `ctx.scales.color` at all, which is a
/// separate live defect logged at that branch.
///
/// The chain this replaces is exactly what `ColorScale::lookup(&str)` does
/// internally, which is what those marks called when they each rolled their own
/// `scale.lookup(key).unwrap_or(paint.fill)` fallback — so this is the
/// group-keyed *spelling* of the one resolution, not a second implementation of
/// it, and a group-keyed mark's numeric-scale handling is unchanged by the move.
///
/// Two things do change for a caller that moves onto this helper: it gains the
/// cleared-fallback precedence (`#110` — a `fill="none"` outranks the scale), and
/// it inherits [`resolve_fill_color`]'s `is_finite` filter, so a group key that
/// parses to `inf`/`NaN` falls back to the constant paint instead of sampling the
/// scheme at a non-finite position.
pub(crate) fn resolve_group_fill_color(
    scale: Option<&ColorScale>,
    group_key: Option<&str>,
    fallback: ClearablePaint,
) -> ClearablePaint {
    resolve_fill_color(scale, group_key, group_key.and_then(|k| k.parse().ok()), fallback)
}

/// Resolve the effective per-row stroke color for a *filled* mark
/// (bar/rect/point), honoring the "visible stroke-width needs a stroke color"
/// rule (RMARK-04).
///
/// SVG only emits `stroke-width` when a stroke color is present. So when a
/// per-row `stroke_width` channel produces a positive width but the mark has no
/// explicit stroke color, fall back to the fill color so the width is actually
/// visible. The gate is intentionally narrow — it triggers only when a
/// `stroke_width` *column* is bound (`stroke_width_col_present`) — so a constant
/// `stroke_width` with no stroke color stays strokeless, exactly as before.
///
/// The rescue returns `fill` *whole*, not just its color: it paints the stroke
/// in the fill color, so a cleared fill rescued into the stroke slot is still a
/// cleared paint (T8 quality-review c2).
///
/// NF-A7 (spec §4.1): an explicit `stroke="none"` beats this rescue, because
/// only [`ClearablePaint::Absent`] opens the gate — a cleared stroke is
/// [`Cleared`](ClearablePaint::Cleared), a distinct state, and the row paints no
/// visible stroke however wide the bound width column says. (Before the paint
/// typing this held indirectly: the clear resolved to `Some(transparent)`, so the
/// `is_none()` gate never opened.)
///
/// This is the one rescue for the whole family. `bar`/`point` each carried a
/// hand-rolled copy of the same three-line block until wave 6A T7 folded them
/// onto this helper; `rect` already called it.
#[inline]
pub(crate) fn resolve_effective_stroke(
    row_stroke_width: f64,
    base_stroke: ClearablePaint,
    fill: ClearablePaint,
    stroke_width_col_present: bool,
) -> ClearablePaint {
    if row_stroke_width > 0.0
        && matches!(base_stroke, ClearablePaint::Absent)
        && stroke_width_col_present
    {
        fill
    } else {
        base_stroke
    }
}

/// Canonical parser: text-baseline string → `TextBaseline` enum.
///
/// Accepts the full union vocabulary of SVG/CSS `dominant-baseline` values and
/// the shorthand aliases used at Python-layer call sites. Returns `None` for
/// unrecognized strings so each call site can apply its own fallback.
///
/// Recognized spellings and their canonical enum values:
/// - `"top"` / `"hanging"` / `"text-before-edge"` → `Top`
/// - `"middle"` / `"central"` → `Middle`
/// - `"bottom"` / `"text-after-edge"` / `"ideographic"` → `Bottom`
/// - `"alphabetic"` → `Alphabetic`
///
/// `svg_walk.rs` renders the enum variants as:
/// - `Top` → `dominant-baseline="hanging"` (valid SVG)
/// - `Middle` → `dominant-baseline="central"` (valid SVG)
/// - `Bottom` → `dominant-baseline="text-after-edge"` (valid SVG)
/// - `Alphabetic` → attribute omitted (SVG default)
/// - `Custom(_)` → attribute emitted verbatim
pub(crate) fn parse_text_baseline(s: &str) -> Option<ferrum_scene::TextBaseline> {
    match s {
        "top" | "hanging" | "text-before-edge" => Some(ferrum_scene::TextBaseline::Top),
        "middle" | "central" => Some(ferrum_scene::TextBaseline::Middle),
        "bottom" | "text-after-edge" | "ideographic" => Some(ferrum_scene::TextBaseline::Bottom),
        "alphabetic" => Some(ferrum_scene::TextBaseline::Alphabetic),
        _ => None,
    }
}

/// Build a scene [`TextStyle`](ferrum_scene::TextStyle).
///
/// `paint` carries the text fill's provenance, exactly as
/// [`to_scene_stroke`]'s does: a [`ClearablePaint::Cleared`] fill reaches the
/// scene as `color: None` (no glyphs painted) rather than as a zero-alpha
/// color. Chrome text (axis, legend, title, strip) has no clear spelling and
/// passes [`ClearablePaint::Paint`].
pub fn to_scene_text_style(
    paint: ClearablePaint,
    font_size: f64,
    anchor: crate::layout::TextAnchor,
    angle: f64,
    font_family: &str,
    font_weight: Option<&str>,
    dominant_baseline: Option<&str>,
    opacity: f64,
) -> ferrum_scene::TextStyle {
    ferrum_scene::TextStyle {
        font_size,
        font_weight: match font_weight {
            Some("bold") => ferrum_scene::FontWeight::Bold,
            Some(w) if w != "normal" => ferrum_scene::FontWeight::Custom(w.to_string()),
            _ => ferrum_scene::FontWeight::Normal,
        },
        anchor: match anchor {
            crate::layout::TextAnchor::Start => ferrum_scene::TextAnchor::Start,
            crate::layout::TextAnchor::Middle => ferrum_scene::TextAnchor::Middle,
            crate::layout::TextAnchor::End => ferrum_scene::TextAnchor::End,
        },
        baseline: match dominant_baseline {
            Some(s) => parse_text_baseline(s)
                .unwrap_or_else(|| ferrum_scene::TextBaseline::Custom(s.to_string())),
            None => ferrum_scene::TextBaseline::Alphabetic,
        },
        angle,
        color: paint.scene_paint(),
        opacity,
        font_family: font_family.to_string(),
    }
}

/// Canonical stroke-dash palette: index → dasharray pattern.
///
/// Index 0 is solid (no entry here — `resolve_stroke_dash` returns `None` for 0).
/// Indices 1–3 map to the pattern at `DASH_PALETTE[index - 1]`:
///   - index 1 → `[6, 3]`   long dash
///   - index 2 → `[2, 3]`   short dash / dot
///   - index 3 → `[6, 3, 2, 3]`  long-short dash
///
/// **Both** `resolve_stroke_dash` (index→pattern, SVG/scene path) and
/// `pack_instances::stroke_dash_index` (pattern→index, GPU packed path) derive
/// from this single source so the two directions cannot silently drift.
///
/// The WASM shader (`crates/ferrum-wasm/src/shaders/`) encodes the same palette
/// to interpret the `stroke_dash` f32 field packed by `stroke_dash_index`. Any
/// change to this table **must** be reflected there too — that shader is not
/// edited by the Rust cohesion refactor; its comment notes the dependency.
pub(crate) const DASH_PALETTE: &[&[f64]] = &[
    &[6.0, 3.0],           // index 1
    &[2.0, 3.0],           // index 2
    &[6.0, 3.0, 2.0, 3.0], // index 3
];

/// Map a stroke-dash palette index to its canonical SVG dasharray pattern.
///
/// The index is rounded and clamped to `[0, 3]` before lookup:
/// - `0` → solid (returns `None`)
/// - `1` → long dash `[6, 3]`
/// - `2` → short dash / dot `[2, 3]`
/// - `3` → long-short dash `[6, 3, 2, 3]`
///
/// Derives from [`DASH_PALETTE`] so it cannot drift from the GPU-path inverse
/// in `pack_instances::stroke_dash_index`.
pub(crate) fn resolve_stroke_dash(idx: f64) -> Option<Vec<f64>> {
    let idx = (idx.round() as i64).clamp(0, 3) as usize;
    if idx == 0 {
        None
    } else {
        DASH_PALETTE.get(idx - 1).map(|pat| pat.to_vec())
    }
}

pub(crate) fn parse_stroke_cap(s: &str) -> Option<ferrum_scene::StrokeCap> {
    match s {
        "round" => Some(ferrum_scene::StrokeCap::Round),
        "square" => Some(ferrum_scene::StrokeCap::Square),
        "butt" => Some(ferrum_scene::StrokeCap::Butt),
        _ => None,
    }
}

pub(crate) fn parse_stroke_join(s: &str) -> Option<ferrum_scene::StrokeJoin> {
    match s {
        "round" => Some(ferrum_scene::StrokeJoin::Round),
        "bevel" => Some(ferrum_scene::StrokeJoin::Bevel),
        "miter" => Some(ferrum_scene::StrokeJoin::Miter),
        _ => None,
    }
}

/// Dispatch a mark's `build` fn. `Mark::Rule` and `Mark::Segment` are
/// fallible mark builders (batch-A Task 13 spec c2 — the totality invariant,
/// `marks/rule.rs`'s module doc): rule's anchor-channel dtype dispatch can
/// raise a typed `RenderError` when a presence-legal shape's anchor column is
/// neither ordinal nor numeric-parseable, and both marks propagate the typed
/// error from the shared per-row color reader
/// (`marks::channels::row_colors_from_scale`) rather than discarding it into a
/// silently theme-colored render (NF-A3 sweep completion, 2026-09-02).
/// `Mark::Tick` joined them in batch-E T13 cycle 3, fallible for the same
/// per-row color reader — see `marks/tick.rs`'s module doc for the full
/// per-row property enumeration that cycle audited against `rule.rs`.
/// `Mark::Arc` and `Mark::Geoshape` are fallible too (batch E Task 6): each
/// raises `RenderError::MarkRequiresCoord` from its own non-matching-coord
/// arm, the path-independent backstop for a caller that reaches the mark
/// builder without going through `scale_resolve`'s own (earlier-firing, same
/// error) guard — see `marks::arc::requires_polar_coord`'s doc comment for
/// why arc needs both. `Mark::Text`, `Mark::Rect`, and `Mark::Label` joined
/// them in wave-1 hardening #166: each propagates the typed error from its
/// own `text`-channel reader (`col_as_str` plus its `col_as_f64` fallback)
/// instead of discarding it into a render with silently missing/blank
/// labels — see each module's doc comment. `Mark::Point`'s sibling swallow
/// (its `shape`-channel reader) is a **disclosed reader-switch, not a
/// refusal** (`marks::point`'s doc comment): it now reads through
/// `col_as_ordinal_category_str` so an `Int*`/`Float*`/`Bool` shape column
/// resolves its real shapes, and stays infallible. Every other mark's
/// `build` stays infallible (unchanged) and is wrapped in `Ok` here. This is
/// every mark's ONLY `build` call site, so widening this function's return
/// type needed no downstream signature changes beyond it and its two
/// callers (`scene_build::build_panel_mark_batches`, and a `position.rs`
/// test helper). One exhaustive match (not the shared
/// `for_each_mark!` macro the sibling `Mark::as_str`/`FromStr` impls use)
/// because the fallible arms have a different type than the other 7 — the
/// macro's single expansion can't express that asymmetry. Every arm is written
/// out, and no wildcard, so a future `Mark` variant is a compile error here
/// rather than a silent miss; the fallible arms return their `Result` directly
/// and the infallible ones wrap in `Ok`, which needs no early return and leaves
/// no unreachable arm to panic from at this `pub` boundary.
pub fn dispatch_mark_build(mark: &Mark, ctx: &DrawCtx) -> Result<MarkBuildResult, RenderError> {
    match mark {
        Mark::Rule => super::marks::rule::build(ctx),
        Mark::Segment => super::marks::segment::build(ctx),
        Mark::Point => Ok(super::marks::point::build(ctx)),
        Mark::Line => Ok(super::marks::line::build(ctx)),
        Mark::Bar => Ok(super::marks::bar::build(ctx)),
        Mark::Area => Ok(super::marks::area::build(ctx)),
        Mark::Text => super::marks::text::build(ctx),
        Mark::Tick => super::marks::tick::build(ctx),
        Mark::Rect => super::marks::rect::build(ctx),
        Mark::Polygon => Ok(super::marks::polygon::build(ctx)),
        Mark::Image => Ok(super::marks::image::build(ctx)),
        Mark::Ribbon => Ok(super::marks::ribbon::build(ctx)),
        Mark::Arc => super::marks::arc::build(ctx),
        Mark::Label => super::marks::label::build(ctx),
        Mark::Geoshape => super::marks::geoshape::build(ctx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RSUP-01: DASH_PALETTE round-trip (index→pattern→index and pattern→index→pattern) ---

    /// Verify that `resolve_stroke_dash` and `stroke_dash_index` are exact mutual inverses
    /// for every entry in `DASH_PALETTE`.  A drift between the two functions silently renders
    /// dashes correctly in SVG but wrong in the GPU/interactive path with no compile error.
    ///
    /// This test exercises both directions of the round-trip:
    ///   1. index → pattern (`resolve_stroke_dash`) → index (`stroke_dash_index`) == original index
    ///   2. pattern → index (`stroke_dash_index`) → pattern (`resolve_stroke_dash`) == original pattern
    #[test]
    fn dash_palette_round_trip_index_to_pattern_to_index() {
        use crate::render::pack_instances::stroke_dash_index;

        // Index 0 is solid.
        assert_eq!(resolve_stroke_dash(0.0), None, "index 0 must be solid (None)");
        let solid_idx = stroke_dash_index(&None);
        assert!((solid_idx - 0.0).abs() < 1e-6, "None pattern → index 0");

        // Indices 1..=N: round-trip in both directions.
        for palette_idx in 1..=(DASH_PALETTE.len()) {
            let pattern = resolve_stroke_dash(palette_idx as f64)
                .expect("non-zero index must produce a pattern");

            // Direction 1: index → pattern → index must recover the original index.
            let recovered_idx = stroke_dash_index(&Some(pattern.clone()));
            assert!(
                (recovered_idx - palette_idx as f32).abs() < 1e-6,
                "round-trip index {palette_idx}: pattern={pattern:?} → index={recovered_idx} (expected {palette_idx})"
            );

            // Direction 2: pattern → index → pattern must recover the original pattern.
            let recovered_pattern = resolve_stroke_dash(recovered_idx as f64)
                .expect("recovered index must produce a pattern");
            assert_eq!(
                pattern, recovered_pattern,
                "round-trip pattern {pattern:?}: index={recovered_idx} → pattern={recovered_pattern:?}"
            );
        }
    }

    /// Solid and unknown patterns always map to index 0 (the solid sentinel).
    #[test]
    fn dash_palette_solid_and_unknown_map_to_zero() {
        use crate::render::pack_instances::stroke_dash_index;

        assert!((stroke_dash_index(&None) - 0.0).abs() < 1e-6, "None → 0");
        assert!((stroke_dash_index(&Some(vec![])) - 0.0).abs() < 1e-6, "empty → 0");
        assert!((stroke_dash_index(&Some(vec![5.0, 5.0])) - 0.0).abs() < 1e-6,
            "unknown pattern → 0");
    }

    // --- Phase 7 baseline tests (updated to 3-arg signature; None overrides = same result) ---

    #[test]
    fn resolve_style_for_area_uses_area_opacity() {
        // Area fill is opaque; area_opacity is carried in style.opacity so the
        // renderer can apply it (and user opacity kwarg can override it).
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Area).unwrap();
        assert_eq!(style.paint.fill.alpha, 0xFF, "area fill should be opaque");
        assert!((style.paint.opacity - theme.sizes.area_opacity).abs() < 1e-6,
            "area opacity should default to theme.sizes.area_opacity");
    }

    #[test]
    fn resolve_style_for_bar_has_corner_radius_from_theme() {
        let mut theme = ThemeInputs::default();
        theme.sizes.bar_corner_radius = 4.0;
        let style = resolve_mark_style(None, &theme, &Mark::Bar).unwrap();
        assert_eq!(style.paint.corner_radius, 4.0);
    }

    #[test]
    fn resolve_style_for_point_is_opaque_by_default() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        assert_eq!(style.paint.fill.alpha, 0xFF);
    }

    // --- Phase 8a Task 7 tests ---

    #[test]
    fn resolve_mark_style_with_no_overrides_returns_theme_defaults() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        assert_eq!(style.point.point_size, theme.sizes.point_size);
    }

    #[test]
    fn resolve_mark_style_overrides_point_size() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { size: Some(100.0), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap();
        assert_eq!(style.point.point_size, 100.0);
    }

    #[test]
    fn resolve_mark_style_overrides_stroke_color() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("#ff0000".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap();
        let stroke = style.paint.stroke.expect("stroke should be set");
        assert_eq!(stroke.red, 0xff);
        assert_eq!(stroke.green, 0x00);
        assert_eq!(stroke.blue, 0x00);
    }

    // --- Batch A Task 8: strict color parsing at the mark-style boundary ---
    //
    // These replace `resolve_mark_style_invalid_color_silently_skipped` and
    // `resolve_mark_style_invalid_fill_leaves_fill_is_user_set_false` (both
    // long deleted, and named for a boolean wave 6A T7 replaced with
    // `PaintSource`), which pinned the pre-batch-A contract: an unparseable `fill=`/`stroke=` left
    // the mark painted in its theme default with no diagnostic. Both inputs
    // are now typed refusals (spec §4.1/§6), so the old assertions are the
    // deliberate contract change, not a regression.

    /// A full CSS color name resolves at the mark-style boundary (F-L02-01):
    /// `steelblue` is CSS Color 4 `(70, 130, 180)`, not a silent no-op.
    #[test]
    fn resolve_mark_style_accepts_css_named_fill() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("steelblue".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap();
        assert_eq!(
            (style.paint.fill.red, style.paint.fill.green, style.paint.fill.blue),
            (70, 130, 180)
        );
        assert_eq!(style.paint.fill_source, PaintSource::UserColor);
    }

    /// The `rgb()` functional form resolves too, and the resolved paint is
    /// byte-identical to the equivalent hex — one parser, one answer.
    #[test]
    fn resolve_mark_style_accepts_rgb_function_stroke_identically_to_hex() {
        let theme = ThemeInputs::default();
        let functional = MarkKwargsSpec {
            stroke: Some("rgb(255, 99, 71)".into()),
            ..Default::default()
        };
        let hex = MarkKwargsSpec { stroke: Some("#ff6347".into()), ..Default::default() };
        let named = MarkKwargsSpec { stroke: Some("tomato".into()), ..Default::default() };
        let from_fn = resolve_mark_style(Some(&functional), &theme, &Mark::Point).unwrap();
        let from_hex = resolve_mark_style(Some(&hex), &theme, &Mark::Point).unwrap();
        let from_name = resolve_mark_style(Some(&named), &theme, &Mark::Point).unwrap();
        assert_eq!(from_fn.paint.stroke, Some(crate::render::color::from_rgb(0xff, 0x63, 0x47)));
        assert_eq!(from_fn.paint.stroke, from_hex.paint.stroke);
        assert_eq!(from_name.paint.stroke, from_hex.paint.stroke);
    }

    /// An unparseable `stroke=` is a typed refusal naming the accepted forms,
    /// never a silent skip.
    #[test]
    fn resolve_mark_style_invalid_stroke_is_typed_error() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("not-a-color".into()), ..Default::default() };
        let err = resolve_mark_style(Some(&overrides), &theme, &Mark::Point)
            .expect_err("an unparseable stroke must refuse, not fall back to the theme default");
        assert_eq!(
            err,
            RenderError::InvalidColor(crate::render::color::primitive::ColorParseError("not-a-color".into()))
        );
        let msg = err.to_string();
        assert!(
            msg.contains(crate::render::color::primitive::ACCEPTED_COLOR_FORMS),
            "message must quote ACCEPTED_COLOR_FORMS verbatim; got {msg:?}"
        );
        assert!(msg.contains("not-a-color"), "message must echo the offending value; got {msg:?}");
    }

    /// The `fill=` arm refuses on the same terms as `stroke=`.
    #[test]
    fn resolve_mark_style_invalid_fill_is_typed_error() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("notacolor".into()), ..Default::default() };
        let err = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).expect_err("must refuse");
        assert_eq!(
            err,
            RenderError::InvalidColor(crate::render::color::primitive::ColorParseError("notacolor".into()))
        );
    }

    // --- "none" clears paint (spec §4.1) ---

    /// `fill="none"` is an explicit clear, never a parse error: the paint
    /// becomes fully transparent AND its source is `UserClear` (a user-set
    /// state), so `text.rs`'s constant-color fallback uses the cleared paint
    /// instead of silently painting the theme font color.
    #[test]
    fn resolve_mark_style_none_fill_clears_paint_and_sets_fill_source_to_user_clear() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("none".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        assert_eq!(style.paint.fill.alpha, 0, "'none' must clear the fill");
        assert_eq!(
            style.paint.fill_source,
            PaintSource::UserClear,
            "'none' is a user-set clear, not an absent fill",
        );
    }

    /// `stroke="none"` likewise, and the cleared stroke is *present* (a
    /// transparent color) rather than `None`, so no fill fallback resurrects it.
    #[test]
    fn resolve_mark_style_none_stroke_clears_paint_and_sets_stroke_source_to_user_clear() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("none".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule).unwrap();
        let stroke = style.paint.stroke.expect("a cleared stroke is present-but-transparent");
        assert_eq!(stroke.alpha, 0, "'none' must clear the stroke");
        assert_eq!(style.paint.stroke_source, PaintSource::UserClear);
        assert_eq!(
            resolve_stroke_color(&style, None).alpha,
            0,
            "the `unwrap_or(fill)` rescue must not resurrect a cleared stroke"
        );
    }

    /// The clear is matched case-insensitively after trimming, like every
    /// other color spelling the parser accepts.
    #[test]
    fn resolve_mark_style_none_clear_is_trimmed_and_case_insensitive() {
        let theme = ThemeInputs::default();
        for spelling in ["NONE", " none ", "None"] {
            let overrides =
                MarkKwargsSpec { fill: Some(spelling.into()), ..Default::default() };
            let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point)
                .unwrap_or_else(|e| panic!("{spelling:?} must clear, not refuse: {e}"));
            assert_eq!(style.paint.fill.alpha, 0, "{spelling:?} must clear the fill");
        }
    }

    /// `"transparent"` is a real CSS Color 4 keyword, so refusing it while
    /// `InvalidColor`'s message promises CSS-name support is
    /// self-contradicting (T8 quality-review finding 1, spec §4.1 2026-09-01
    /// supersession). It joins `"none"` as a clearing spelling: same trimmed,
    /// case-folded branch, same outcome, on both `fill=` and `stroke=`.
    #[test]
    fn resolve_mark_style_transparent_clears_paint_like_none() {
        let theme = ThemeInputs::default();
        let fill_overrides = MarkKwargsSpec { fill: Some("transparent".into()), ..Default::default() };
        let fill_style = resolve_mark_style(Some(&fill_overrides), &theme, &Mark::Text)
            .unwrap_or_else(|e| panic!("'transparent' must clear the fill, not refuse: {e}"));
        assert_eq!(fill_style.paint.fill.alpha, 0, "'transparent' must clear the fill");
        assert_eq!(fill_style.paint.fill_source, PaintSource::UserClear);

        let stroke_overrides = MarkKwargsSpec { stroke: Some("transparent".into()), ..Default::default() };
        let stroke_style = resolve_mark_style(Some(&stroke_overrides), &theme, &Mark::Rule)
            .unwrap_or_else(|e| panic!("'transparent' must clear the stroke, not refuse: {e}"));
        let stroke = stroke_style.paint.stroke.expect("a cleared stroke is present-but-transparent");
        assert_eq!(stroke.alpha, 0, "'transparent' must clear the stroke");
        assert_eq!(stroke_style.paint.stroke_source, PaintSource::UserClear);
    }

    /// The `"transparent"` clear is matched case-insensitively after
    /// trimming, exactly like `"none"`.
    #[test]
    fn resolve_mark_style_transparent_clear_is_trimmed_and_case_insensitive() {
        let theme = ThemeInputs::default();
        for spelling in ["TRANSPARENT", " transparent ", "Transparent"] {
            let overrides = MarkKwargsSpec { fill: Some(spelling.into()), ..Default::default() };
            let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point)
                .unwrap_or_else(|e| panic!("{spelling:?} must clear, not refuse: {e}"));
            assert_eq!(style.paint.fill.alpha, 0, "{spelling:?} must clear the fill");
        }
    }

    /// Neither clearing spelling enters `parse_color`'s vocabulary — the
    /// clear is a sentinel handled ahead of parsing, not a parsed color.
    #[test]
    fn none_and_transparent_are_not_parseable_colors() {
        assert!(crate::render::color::parse_color("none").is_err());
        assert!(crate::render::color::parse_color("transparent").is_err());
    }

    // --- theme:label sentinel tests ---

    /// The two sentinels are checked BEFORE parsing, and that ordering is
    /// load-bearing: neither is a color the parser accepts, so a dispatch-order
    /// regression turns both into `InvalidColor` refusals.
    #[test]
    fn mark_style_sentinels_are_not_parseable_colors() {
        assert!(crate::render::color::parse_color("theme:label").is_err());
        assert!(crate::render::color::parse_color("none").is_err());
    }

    #[test]
    fn resolve_mark_style_stroke_theme_label_sentinel_uses_label_color() {
        let mut theme = ThemeInputs::default();
        theme.colors.label_color = palette::Srgba::new(0x11, 0x22, 0x33, 0xFF);
        let overrides = MarkKwargsSpec { stroke: Some("theme:label".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule).unwrap();
        let stroke = style.paint.stroke.expect("stroke must be set by sentinel");
        assert_eq!(stroke.red,   0x11, "sentinel stroke.red must be label_color.red");
        assert_eq!(stroke.green, 0x22, "sentinel stroke.green must be label_color.green");
        assert_eq!(stroke.blue,  0x33, "sentinel stroke.blue must be label_color.blue");
    }

    #[test]
    fn resolve_mark_style_fill_theme_label_sentinel_uses_label_color() {
        let mut theme = ThemeInputs::default();
        theme.colors.label_color = palette::Srgba::new(0x44, 0x55, 0x66, 0xFF);
        let overrides = MarkKwargsSpec { fill: Some("theme:label".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Tick).unwrap();
        assert_eq!(style.paint.fill.red,   0x44, "sentinel fill.red must be label_color.red");
        assert_eq!(style.paint.fill.green, 0x55, "sentinel fill.green must be label_color.green");
        assert_eq!(style.paint.fill.blue,  0x66, "sentinel fill.blue must be label_color.blue");
    }

    // --- `MarkPaint::fill_source` (Task 4 remediation, spec §4.4,
    // 2026-08-28 T4 amendment; the `fill_is_user_set` boolean it began as was
    // typed into `PaintSource` by wave 6A T7) ---

    #[test]
    fn resolve_mark_style_no_fill_override_leaves_fill_source_inherited() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        assert_eq!(style.paint.fill_source, PaintSource::Inherited);
    }

    #[test]
    fn resolve_mark_style_hex_fill_sets_fill_source_to_user_color() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("#ff0000".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        assert_eq!(style.paint.fill_source, PaintSource::UserColor);
        assert_eq!(style.paint.fill.red, 0xff);
    }

    #[test]
    fn resolve_mark_style_theme_label_fill_sentinel_sets_fill_source_to_user_color() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("theme:label".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        assert_eq!(style.paint.fill_source, PaintSource::UserColor);
    }

    /// An unparseable `fill=` value never leaves `fill_source` user-set — a
    /// downstream constant-fallback consumer (e.g. `text.rs`) must not be told
    /// the user successfully set a fill. Since batch-A Task 8 that guarantee
    /// holds by refusal rather than by a silently-unwritten source: the render
    /// stops before any consumer can read the source at all. (The refusal
    /// itself is asserted by `resolve_mark_style_invalid_fill_is_typed_error`.)
    #[test]
    fn resolve_mark_style_invalid_fill_refuses_before_writing_a_fill_source() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("notacolor".into()), ..Default::default() };
        assert!(resolve_mark_style(Some(&overrides), &theme, &Mark::Text).is_err());
    }

    #[test]
    fn resolve_mark_style_empty_stroke_dash_clears_reference_line_dash() {
        // Rule mark picks up reference_line_dash from theme by default.
        // Passing stroke_dash: [] should clear it (solid line for composite structural rules).
        let theme = ThemeInputs::default(); // reference_line_dash = Some([4.0, 4.0])
        assert!(theme.reference_line.reference_line_dash.is_some(), "test requires non-None reference_line_dash");
        let overrides = MarkKwargsSpec { stroke_dash: Some(vec![]), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule).unwrap();
        assert!(style.paint.stroke_dash.is_none(), "empty stroke_dash override must clear the dash");
    }

    // --- B6: tooltip format/format_type must be honored ---

    /// B6: When a tooltip encoding has `format: ".2f"`, numeric values must be
    /// formatted with 2 decimal places, not the hardcoded `"{:.4}"` fallback.
    #[test]
    fn b6_tooltip_format_spec_applied_to_numeric_column() {
        use crate::layout::{PanelLayout, Rect};
        use crate::render::scale_resolve::resolve_scales;
        use crate::spec::chart::ChartSpec;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec {
                    field: "val".into(),
                    type_: Some(SDT::Quantitative),
                    format: Some(".2f".to_string()),
                    format_type: Some("number".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![3.0, 4.0])),
            Arc::new(Float64Array::from(vec![3.14159, 2.71828])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let meta = MetadataColumns::from_ctx(&ctx);
        assert_eq!(meta.tooltip_cols.len(), 1, "expected 1 tooltip column");
        let (field_name, values) = &meta.tooltip_cols[0];
        assert_eq!(field_name, "val");

        // With format ".2f", 3.14159 should appear as "3.14", NOT "3.1416" (the old hardcoded .4 pattern).
        let formatted_pi = values[0].as_deref().expect("value[0] must be Some");
        assert_eq!(
            formatted_pi, "3.14",
            "tooltip format '.2f' must produce 2 decimal places; got: '{formatted_pi}'"
        );
        let formatted_e = values[1].as_deref().expect("value[1] must be Some");
        assert_eq!(
            formatted_e, "2.72",
            "tooltip format '.2f' must produce 2 decimal places; got: '{formatted_e}'"
        );
    }

    // --- F4: resolve_stroke_color precedence truth table ---

    #[test]
    fn resolve_stroke_color_user_set_ignores_row_color() {
        // A user-set stroke source → explicit stroke wins regardless of row color.
        let mut ms = resolve_mark_style(None, &ThemeInputs::default(), &Mark::Rule).unwrap();
        ms.paint.stroke = Some(crate::render::color::from_rgb(0x11, 0x22, 0x33));
        ms.paint.stroke_source = PaintSource::UserColor;
        let row_color = Some(crate::render::color::from_rgb(0xAA, 0xBB, 0xCC));
        // Present row color is ignored.
        let c = resolve_stroke_color(&ms, row_color);
        assert_eq!((c.red, c.green, c.blue), (0x11, 0x22, 0x33));
        // Absent row color → still the explicit stroke.
        let c2 = resolve_stroke_color(&ms, None);
        assert_eq!((c2.red, c2.green, c2.blue), (0x11, 0x22, 0x33));
    }

    #[test]
    fn resolve_stroke_color_user_set_falls_back_to_fill_when_stroke_none() {
        let mut ms = resolve_mark_style(None, &ThemeInputs::default(), &Mark::Rule).unwrap();
        ms.paint.stroke = None;
        ms.paint.fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        ms.paint.stroke_source = PaintSource::UserColor;
        let c = resolve_stroke_color(&ms, Some(crate::render::color::from_rgb(0xAA, 0xBB, 0xCC)));
        assert_eq!((c.red, c.green, c.blue), (0x44, 0x55, 0x66),
            "user-set stroke with None stroke must fall back to fill");
    }

    #[test]
    fn resolve_stroke_color_not_user_set_prefers_row_color() {
        let mut ms = resolve_mark_style(None, &ThemeInputs::default(), &Mark::Rule).unwrap();
        ms.paint.stroke = Some(crate::render::color::from_rgb(0x11, 0x22, 0x33));
        ms.paint.fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        ms.paint.stroke_source = PaintSource::Inherited;
        // Row color present → it wins over theme stroke and fill.
        let row = crate::render::color::from_rgb(0xAA, 0xBB, 0xCC);
        let c = resolve_stroke_color(&ms, Some(row));
        assert_eq!((c.red, c.green, c.blue), (0xAA, 0xBB, 0xCC));
    }

    #[test]
    fn resolve_stroke_color_not_user_set_no_row_color_uses_stroke_then_fill() {
        let mut ms = resolve_mark_style(None, &ThemeInputs::default(), &Mark::Rule).unwrap();
        ms.paint.stroke = Some(crate::render::color::from_rgb(0x11, 0x22, 0x33));
        ms.paint.fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        ms.paint.stroke_source = PaintSource::Inherited;
        // No row color → theme stroke wins.
        let c = resolve_stroke_color(&ms, None);
        assert_eq!((c.red, c.green, c.blue), (0x11, 0x22, 0x33));
        // No row color, no theme stroke → fill.
        ms.paint.stroke = None;
        let c2 = resolve_stroke_color(&ms, None);
        assert_eq!((c2.red, c2.green, c2.blue), (0x44, 0x55, 0x66));
    }

    // --- NF-A7: resolve_effective_stroke's stroke-width rescue vs. an
    // explicit `stroke="none"` clear (spec §4.1) ---

    /// The rescue itself, pinned: a bound `stroke_width` column with a positive
    /// width and NO stroke color paints the stroke in the fill color, so the
    /// width the user asked for is actually visible (RMARK-04).
    #[test]
    fn resolve_effective_stroke_rescues_unset_stroke_with_fill() {
        let fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        let fill_paint = ClearablePaint::Paint(fill);
        assert_eq!(
            resolve_effective_stroke(2.0, ClearablePaint::Absent, fill_paint, true),
            fill_paint,
            "positive width + bound width column + no stroke → stroke painted in the fill color"
        );
        // The gate stays narrow: a constant width (no bound column) does not rescue.
        assert_eq!(
            resolve_effective_stroke(2.0, ClearablePaint::Absent, fill_paint, false),
            ClearablePaint::Absent,
        );
        // Nor does a zero-width row.
        assert_eq!(
            resolve_effective_stroke(0.0, ClearablePaint::Absent, fill_paint, true),
            ClearablePaint::Absent,
        );
    }

    /// A rescued *cleared* fill stays cleared in the stroke slot: the rescue
    /// paints the stroke in the fill's paint, provenance included, so the
    /// cleared-ness cannot be dropped on the way across (T8 quality-review c2).
    #[test]
    fn resolve_effective_stroke_rescue_carries_the_fills_cleared_ness() {
        assert_eq!(
            resolve_effective_stroke(2.0, ClearablePaint::Absent, ClearablePaint::Cleared, true),
            ClearablePaint::Cleared,
        );
    }

    /// NF-A7: `stroke="none"` beats the rescue. A user who cleared the stroke
    /// and bound a `stroke_width` column gets NO visible stroke — not a stroke
    /// silently painted in the fill color. The clear is a state of its own
    /// (`Cleared`, not `Absent`), which is what keeps the rescue gate shut.
    #[test]
    fn resolve_effective_stroke_cleared_stroke_beats_the_width_rescue() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("none".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rect).unwrap();
        let fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        let effective = resolve_effective_stroke(
            2.0,
            style.paint.stroke_paint(),
            ClearablePaint::Paint(fill),
            true,
        );
        assert_eq!(
            effective,
            ClearablePaint::Cleared,
            "the rescue did not fire, so the cleared stroke carries through",
        );
        assert_eq!(effective.scene_paint(), None, "and it must not serialize at all");
    }

    /// Full-pipeline proof of the same NF-A7 invariant, at the `resolve_effective_stroke`
    /// call site the mutation report names (`rect.rs`'s `build_heatmap`, reached via
    /// `render_svg`): `fm.Chart(df).mark_rect(fill="none").encode(x="xc:N", y="yc:N",
    /// stroke_width="w:Q")` must emit NO `stroke` attribute on any cell.
    ///
    /// The unit tests above pin `resolve_effective_stroke`'s return value directly
    /// (including the cleared-fill rescue), but only at the helper. This test drives
    /// the same branch through the real mark builder: a bound `stroke_width` column,
    /// a positive per-row width, and no explicit stroke color opens the rescue gate,
    /// while `fill="none"` makes the rescued paint a cleared one — the case the
    /// mutation report's mutant (returning the base stroke's provenance instead of
    /// the fill's) desyncs.
    #[test]
    fn fill_none_rect_with_bound_stroke_width_emits_no_stroke_attribute() {
        use crate::layout::Viewport;
        use crate::render::chart_config::ChartConfig;
        use crate::render::config::RenderConfig;
        use crate::spec::chart::ChartSpec;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as SpecDataType, Encoding, EncodingSpec};
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType as ArrowType, Field, Schema};
        use std::sync::Arc;

        let field = |name: &str, t: SpecDataType| {
            Some(EncodingSpec { field: name.into(), type_: Some(t), ..Default::default() })
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Rect,
            encoding: Encoding {
                x: field("xc", SpecDataType::Nominal),
                y: field("yc", SpecDataType::Nominal),
                stroke_width: field("w", SpecDataType::Quantitative),
                ..Default::default()
            },
            mark_style: Some(MarkKwargsSpec { fill: Some("none".into()), ..Default::default() }),
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("xc", ArrowType::Utf8, false),
            Field::new("yc", ArrowType::Utf8, false),
            Field::new("w", ArrowType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(StringArray::from(vec!["p", "q"])),
                Arc::new(Float64Array::from(vec![2.0, 3.0])),
            ],
        )
        .unwrap();
        let theme = ThemeInputs::default();
        let result = crate::render::render_svg(
            &spec,
            &batch,
            &theme,
            Viewport { width: 600.0, height: 400.0 },
            &RenderConfig::default(),
            &ChartConfig::default(),
        )
        .expect("a fill=\"none\" heatmap-style rect with a bound stroke_width column must render");
        let svg = result.bytes;
        // Filter to the heatmap cells specifically (`fill="none"`), not the
        // panel background/clip rects `render_svg` also emits, neither of
        // which carries this mark's paint at all.
        let cells: Vec<&str> = svg
            .split("<rect ")
            .skip(1)
            .map(|rest| rest.split('>').next().unwrap())
            .filter(|el| el.contains("fill=\"none\""))
            .collect();
        assert_eq!(cells.len(), 2, "expected 2 heatmap cells with a cleared fill; got {cells:?}");
        for element in cells {
            assert!(
                !element.contains("stroke"),
                "a cleared fill rescued into the stroke slot must stay a cleared paint — no \
                 `stroke` attribute may appear, got: {element}"
            );
        }
    }

    // --- F3: resolve_fill_color categorical + continuous ---

    #[test]
    fn resolve_fill_color_categorical_lookup() {
        use crate::render::scale_resolve::ColorScale;
        let red = crate::render::color::from_rgb(0xFF, 0x00, 0x00);
        let blue = crate::render::color::from_rgb(0x00, 0x00, 0xFF);
        let scale = ColorScale::Categorical {
            domain: vec!["a".into(), "b".into()],
            palette: std::borrow::Cow::Owned(vec![red, blue]),
        };
        let fallback = crate::render::color::from_rgb(0x77, 0x77, 0x77);
        let paint = ClearablePaint::Paint(fallback);
        // "a" → palette[0] = red; num_value is ignored for categorical.
        assert_eq!(resolve_fill_color(Some(&scale), Some("a"), Some(99.0), paint), ClearablePaint::Paint(red));
        // A hit over a CLEARED fallback stays cleared: an explicit `fill="none"`
        // outranks the color channel (`#110`, spec §4.1). Before wave 6A T12 this
        // arm answered `Paint(red)`, which is how a cleared paint under a color
        // scale rendered the palette color anyway.
        assert_eq!(
            resolve_fill_color(Some(&scale), Some("a"), Some(99.0), ClearablePaint::Cleared),
            ClearablePaint::Cleared,
            "a user paint-clear outranks a scale-lookup hit",
        );
        // "b" → palette[1] = blue.
        assert_eq!(
            resolve_fill_color(Some(&scale), Some("b"), None, paint),
            ClearablePaint::Paint(blue),
        );
        // Unknown category → the fallback paint, cleared-ness and all.
        assert_eq!(resolve_fill_color(Some(&scale), Some("z"), None, paint), paint);
        assert_eq!(
            resolve_fill_color(Some(&scale), Some("z"), None, ClearablePaint::Cleared),
            ClearablePaint::Cleared,
            "a scale-lookup miss keeps the fallback's clear",
        );
        // No category string → fallback.
        assert_eq!(resolve_fill_color(Some(&scale), None, None, paint), paint);
    }

    #[test]
    fn resolve_fill_color_continuous_uses_lookup_f64() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use crate::render::scale_resolve::ColorScale;
        let scheme = ContinuousScheme::Named(NamedContinuous::Viridis);
        let scale = ColorScale::Continuous { domain: (0.0, 10.0), scheme, midpoint: None };
        let paint = ClearablePaint::Paint(crate::render::color::from_rgb(0x77, 0x77, 0x77));
        // The continuous branch must consume the f64 directly (no round-trip):
        // the resolved color equals scale.lookup_f64(value).
        let v = 3.0;
        assert_eq!(
            resolve_fill_color(Some(&scale), Some("ignored"), Some(v), paint),
            ClearablePaint::Paint(scale.lookup_f64(v).unwrap()),
        );
        assert_eq!(
            resolve_fill_color(Some(&scale), Some("ignored"), Some(v), ClearablePaint::Cleared),
            ClearablePaint::Cleared,
            "a user paint-clear outranks a continuous-scale hit too",
        );
        // Non-finite / absent numeric → fallback.
        assert_eq!(resolve_fill_color(Some(&scale), None, Some(f64::NAN), paint), paint);
        assert_eq!(resolve_fill_color(Some(&scale), None, None, paint), paint);
    }

    #[test]
    fn resolve_fill_color_no_scale_returns_fallback() {
        let paint = ClearablePaint::Paint(crate::render::color::from_rgb(0x12, 0x34, 0x56));
        assert_eq!(resolve_fill_color(None, Some("a"), Some(1.0), paint), paint);
        assert_eq!(
            resolve_fill_color(None, Some("a"), Some(1.0), ClearablePaint::Cleared),
            ClearablePaint::Cleared,
            "no scale → the fallback paint passes through unchanged",
        );
    }

    /// B6: When no format is specified, the existing default behavior (trim trailing zeros)
    /// must be preserved — this verifies we do not break the fallback path.
    #[test]
    fn b6_tooltip_default_format_trims_trailing_zeros() {
        use crate::layout::{PanelLayout, Rect};
        use crate::render::scale_resolve::resolve_scales;
        use crate::spec::chart::ChartSpec;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec {
                    field: "val".into(),
                    type_: Some(SDT::Quantitative),
                    // No format or format_type set.
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![2.0])),
            Arc::new(Float64Array::from(vec![1.5])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let meta = MetadataColumns::from_ctx(&ctx);
        let (_, values) = &meta.tooltip_cols[0];
        // Default: 1.5 → "1.5" (trimmed, not "1.5000")
        let formatted = values[0].as_deref().expect("must have value");
        assert!(
            !formatted.contains("0000"),
            "default tooltip format must trim trailing zeros; got: '{formatted}'"
        );
    }

    /// GH #71 residual: `MetadataColumns::from_ctx` must build each tooltip's
    /// DISPLAYED name from the channel's `title` when one is set, not always
    /// from `field`. This matters for internal rename sentinels: `field` must
    /// stay the actual (renamed) data column so the value lookup still works,
    /// but the name shown in the rendered tooltip must be the original column
    /// name stamped onto `title` (`chart._rename_encoding_fields`). Covers both
    /// the `tooltip_fields` (auto-injected multi-field) path and the single
    /// `tooltip` (explicit `Tooltip(...)`) path, since both feed the same
    /// `tooltip_specs` collection above.
    #[test]
    fn tooltip_name_prefers_title_over_field_when_title_set() {
        use crate::layout::{PanelLayout, Rect};
        use crate::render::scale_resolve::resolve_scales;
        use crate::spec::chart::ChartSpec;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip_fields: Some(vec![
                    EncodingSpec {
                        field: "y__rhs_0001".into(),
                        type_: Some(SDT::Quantitative),
                        title: Some("y".into()),
                        ..Default::default()
                    },
                    EncodingSpec {
                        field: "x".into(),
                        type_: Some(SDT::Quantitative),
                        ..Default::default()
                    },
                ]),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None, params: Vec::new(),
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y__rhs_0001", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![2.0])),
            Arc::new(Float64Array::from(vec![10.0])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let meta = MetadataColumns::from_ctx(&ctx);
        assert_eq!(meta.tooltip_cols.len(), 2, "expected 2 tooltip columns");

        let (renamed_name, _) = &meta.tooltip_cols[0];
        assert_eq!(
            renamed_name, "y",
            "tooltip name must prefer title ('y') over the internal rename sentinel field ('y__rhs_0001'); got: '{renamed_name}'"
        );
        assert!(
            !renamed_name.contains("__rhs_"),
            "internal rename sentinel must never appear in the tooltip's displayed name; got: '{renamed_name}'"
        );

        let (untitled_name, _) = &meta.tooltip_cols[1];
        assert_eq!(
            untitled_name, "x",
            "when no title is set, tooltip name must fall back to field, unchanged from prior behavior"
        );
    }

    // --- RSUP-04: parse_text_baseline + to_scene_text_style corrected SVG output ---

    /// RSUP-04 regression: baseline="bottom" must resolve to TextBaseline::Bottom,
    /// which svg_walk renders as dominant-baseline="text-after-edge" (valid SVG).
    /// The old verbatim-passthrough code would have produced Custom("bottom"),
    /// rendering as the invalid attribute dominant-baseline="bottom".
    #[test]
    fn parse_text_baseline_bottom_resolves_to_bottom_not_custom() {
        let result = parse_text_baseline("bottom");
        assert_eq!(result, Some(ferrum_scene::TextBaseline::Bottom),
            "\"bottom\" must resolve to TextBaseline::Bottom (SVG: text-after-edge), not Custom(\"bottom\")");
    }

    /// RSUP-04 regression: baseline="top" must resolve to TextBaseline::Top,
    /// which svg_walk renders as dominant-baseline="hanging" (valid SVG).
    /// The old verbatim-passthrough code would have produced Custom("top"),
    /// rendering as the invalid attribute dominant-baseline="top".
    #[test]
    fn parse_text_baseline_top_resolves_to_top_not_custom() {
        let result = parse_text_baseline("top");
        assert_eq!(result, Some(ferrum_scene::TextBaseline::Top),
            "\"top\" must resolve to TextBaseline::Top (SVG: hanging), not Custom(\"top\")");
    }

    /// RSUP-04 regression: baseline="alphabetic" must resolve to TextBaseline::Alphabetic,
    /// which svg_walk renders as omitting the attribute (SVG default, correct behavior).
    /// The old verbatim-passthrough code would have produced Custom("alphabetic"),
    /// rendering as the redundant but technically valid attribute dominant-baseline="alphabetic".
    #[test]
    fn parse_text_baseline_alphabetic_resolves_correctly() {
        let result = parse_text_baseline("alphabetic");
        assert_eq!(result, Some(ferrum_scene::TextBaseline::Alphabetic),
            "\"alphabetic\" must resolve to TextBaseline::Alphabetic (attribute omitted in SVG)");
    }

    /// Full union vocabulary: every known alias must resolve to the correct variant.
    #[test]
    fn parse_text_baseline_full_vocabulary() {
        use ferrum_scene::TextBaseline::*;
        // Top aliases
        assert_eq!(parse_text_baseline("top"),              Some(Top));
        assert_eq!(parse_text_baseline("hanging"),          Some(Top));
        assert_eq!(parse_text_baseline("text-before-edge"), Some(Top));
        // Middle aliases
        assert_eq!(parse_text_baseline("middle"),           Some(Middle));
        assert_eq!(parse_text_baseline("central"),          Some(Middle));
        // Bottom aliases
        assert_eq!(parse_text_baseline("bottom"),           Some(Bottom));
        assert_eq!(parse_text_baseline("text-after-edge"),  Some(Bottom));
        assert_eq!(parse_text_baseline("ideographic"),      Some(Bottom));
        // Alphabetic
        assert_eq!(parse_text_baseline("alphabetic"),       Some(Alphabetic));
        // Unrecognized → None
        assert_eq!(parse_text_baseline("auto"),             None);
        assert_eq!(parse_text_baseline(""),                 None);
        assert_eq!(parse_text_baseline("Top"),              None, "case-sensitive");
    }

    /// to_scene_text_style with baseline="bottom" must produce TextBaseline::Bottom
    /// in the returned TextStyle. This pins the corrected behavior end-to-end
    /// (MarkStyle → to_scene_text_style → TextBaseline → svg_walk → DOM attribute).
    #[test]
    fn to_scene_text_style_bottom_emits_text_after_edge_baseline() {
        use crate::layout::TextAnchor;
        use crate::render::color::from_rgb;
        let style = to_scene_text_style(
            ClearablePaint::Paint(from_rgb(0, 0, 0)),
            12.0,
            TextAnchor::Middle,
            0.0,
            "Inter",
            None,
            Some("bottom"),
            1.0,
        );
        assert_eq!(style.baseline, ferrum_scene::TextBaseline::Bottom,
            "baseline=\"bottom\" must produce TextBaseline::Bottom (svg_walk: dominant-baseline=\"text-after-edge\")");
    }

    /// to_scene_text_style with baseline="top" must produce TextBaseline::Top.
    #[test]
    fn to_scene_text_style_top_emits_hanging_baseline() {
        use crate::layout::TextAnchor;
        use crate::render::color::from_rgb;
        let style = to_scene_text_style(
            ClearablePaint::Paint(from_rgb(0, 0, 0)),
            12.0,
            TextAnchor::Middle,
            0.0,
            "Inter",
            None,
            Some("top"),
            1.0,
        );
        assert_eq!(style.baseline, ferrum_scene::TextBaseline::Top,
            "baseline=\"top\" must produce TextBaseline::Top (svg_walk: dominant-baseline=\"hanging\")");
    }

    // --- RSUP-07: parse_stroke_cap / parse_stroke_join string→enum contracts ---

    #[test]
    fn parse_stroke_cap_maps_known_strings() {
        assert_eq!(parse_stroke_cap("round"),  Some(ferrum_scene::StrokeCap::Round));
        assert_eq!(parse_stroke_cap("square"), Some(ferrum_scene::StrokeCap::Square));
        assert_eq!(parse_stroke_cap("butt"),   Some(ferrum_scene::StrokeCap::Butt));
    }

    #[test]
    fn parse_stroke_cap_returns_none_for_unknown() {
        assert_eq!(parse_stroke_cap("flat"),   None);
        assert_eq!(parse_stroke_cap(""),        None);
        assert_eq!(parse_stroke_cap("Round"),  None, "matching must be case-sensitive");
    }

    #[test]
    fn parse_stroke_join_maps_known_strings() {
        assert_eq!(parse_stroke_join("round"), Some(ferrum_scene::StrokeJoin::Round));
        assert_eq!(parse_stroke_join("bevel"), Some(ferrum_scene::StrokeJoin::Bevel));
        assert_eq!(parse_stroke_join("miter"), Some(ferrum_scene::StrokeJoin::Miter));
    }

    #[test]
    fn parse_stroke_join_returns_none_for_unknown() {
        assert_eq!(parse_stroke_join("arcs"),  None);
        assert_eq!(parse_stroke_join(""),       None);
        assert_eq!(parse_stroke_join("Miter"), None, "matching must be case-sensitive");
    }

    // ── Batch E Task 9: clip escape (`mark_function(clip=False)`) ──────────

    /// No `clip` override leaves `misc.clip` at its `None` baseline — the
    /// byte-stable default every layer resolves before this field existed.
    /// The `MarkBatch` construction site (`scene_build.rs`) treats `None` as
    /// `true` (clipped).
    #[test]
    fn resolve_mark_style_with_no_clip_override_leaves_misc_clip_none() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        assert_eq!(style.misc.clip, None);
    }

    #[test]
    fn resolve_mark_style_clip_override_false_sets_misc_clip() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { clip: Some(false), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        assert_eq!(style.misc.clip, Some(false));
    }

    #[test]
    fn resolve_mark_style_clip_override_true_sets_misc_clip() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { clip: Some(true), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        assert_eq!(style.misc.clip, Some(true));
    }

    // ── Batch H Task 1: arrow / arrow_size (segment) ────────────────────────

    /// No `arrow`/`arrow_size` override leaves `segment` at its `None`
    /// baseline — the byte-stable default every segment resolves before
    /// this field existed. `segment.rs` treats `None` arrow as `false`.
    #[test]
    fn resolve_mark_style_with_no_arrow_override_leaves_segment_none() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Segment).unwrap();
        assert_eq!(style.segment.arrow, None);
        assert_eq!(style.segment.arrow_size, None);
    }

    #[test]
    fn resolve_mark_style_arrow_override_threads_arrow_and_size() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { arrow: Some(true), arrow_size: Some(12.0), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Segment).unwrap();
        assert_eq!(style.segment.arrow, Some(true));
        assert_eq!(style.segment.arrow_size, Some(12.0));
    }

}
