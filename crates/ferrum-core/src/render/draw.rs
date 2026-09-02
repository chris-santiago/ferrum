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
use super::scale_resolve::{ColorInput, ColorScale, ResolvedScales};
use super::RenderError;

pub struct DrawCtx<'a> {
    pub spec: &'a crate::spec::chart::ChartSpec,
    pub panel: &'a PanelLayout,
    pub theme: &'a ThemeInputs,
    pub scales: &'a ResolvedScales,
    pub batch: &'a RecordBatch,
    pub mark_style: &'a MarkStyle,
}

/// Cross-cutting paint applied to **every** mark family: fill/stroke colors,
/// stroke width, mark opacity, dash pattern, and rect/bar corner radius. These
/// are the only style fields read by more than one mark family, so they form the
/// shared base of [`MarkStyle`].
///
/// `stroke_is_user_set` is `true` when `stroke` was set by an explicit user
/// `mark_kwargs` override (including the `"theme:label"` sentinel), not merely
/// inherited from the mark's theme default. Used by rule/segment renderers to
/// enforce the precedence: explicit constant stroke > per-row color encoding >
/// fill fallback.
///
/// `fill_is_user_set` is the `fill` analog (spec §4.4, 2026-08-28 T4
/// amendment): `true` only when `fill` actually resolved from an explicit user
/// `mark_kwargs` override — the `"theme:label"` sentinel, the explicit `"none"`
/// clear, or a color string [`parse_color`] accepts — never merely inherited
/// from the mark's theme-aware default. An *unparseable* string sets neither the
/// flag nor the paint: it fails the whole render with
/// [`RenderError::InvalidColor`] (batch-A Task 8), so the "resolved or refused"
/// reading of the flag holds with no third outcome. Set on
/// the *layer-resolved* `MarkPaint` (each layer calls `resolve_mark_style`
/// independently), so it is correct in both flat and layered charts — unlike
/// reading the raw chart-level `mark_kwargs` off `ChartSpec.mark_style`, which
/// a layered chart's synthetic per-layer `ChartSpec` does NOT override (it
/// stays the chart-level value via `..spec.clone()` in
/// `scene_build.rs::build_panel_mark_batches`). Used by `text.rs` to decide
/// whether its constant color fallback is the resolved `fill` or the theme
/// font color.
#[derive(Debug, Clone)]
pub struct MarkPaint {
    pub fill: Color,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub corner_radius: f64,
    pub stroke_dash: Option<Vec<f64>>,
    pub stroke_is_user_set: bool,
    pub fill_is_user_set: bool,
    /// `true` only when `fill` resolved via the `"none"`/`"transparent"` clear
    /// sentinel in [`resolve_paint_color`] — never when it merely *equals*
    /// [`color::TRANSPARENT`](super::color::TRANSPARENT) by value (an explicit
    /// `fill="#00000000"`, or `opacity=0` composed onto black, both resolve a
    /// real color that happens to be zero-alpha). This is the provenance flag
    /// [`normalize_scene_paint`] gates on (T8 quality-review c2, spec §4.1
    /// 2026-09-01 amendment): a cleared paint omits its SVG/scene-JSON
    /// attribute, but a same-valued-but-not-cleared paint must keep
    /// serializing byte-identically (spec §7).
    pub fill_cleared: bool,
    /// The `stroke` analog of [`fill_cleared`](MarkPaint::fill_cleared).
    pub stroke_cleared: bool,
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
/// `mark_area`. `interpolate` controls curve interpolation; `stroke_cap` /
/// `stroke_join` shape line endpoints/corners. Area reuses `interpolate` from
/// this struct (the line and area renderers share the same path builder); area's
/// own area-only overrides live in [`AreaStyle`].
#[derive(Debug, Clone, Default)]
pub struct LineStyle {
    /// S1: interpolate (line/area).
    pub interpolate: Option<String>,
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

/// Geometry-sizing overrides that don't cluster with a single style family:
/// `band_size` (tick/rect band fraction) and image-tile `width`/`height`.
#[derive(Debug, Clone, Default)]
pub struct MiscStyle {
    /// S8: band fraction for tick/rect category bands.
    pub band_size: Option<f64>,
    /// `mark_image` URL-tile width/height (px).
    pub width: Option<f64>,
    pub height: Option<f64>,
}

/// Per-mark resolved style. Fields are populated from theme defaults (mark-aware)
/// and then overridden by any `MarkKwargsSpec` present on the layer or chart.
///
/// The cross-cutting paint every mark reads lives in [`paint`](MarkStyle::paint);
/// each mark family's single-family overrides live in a typed sub-struct
/// ([`text`](MarkStyle::text), [`line`](MarkStyle::line),
/// [`area`](MarkStyle::area), [`point`](MarkStyle::point),
/// [`group`](MarkStyle::group), [`misc`](MarkStyle::misc)). Per-mark draw
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
                stroke_is_user_set: false,
                fill_is_user_set: false,
                fill_cleared: false,
                stroke_cleared: false,
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
/// ahead of parsing. Exposed separately so [`resolve_mark_style`] can record
/// *why* a paint resolved to [`color::TRANSPARENT`](super::color::TRANSPARENT)
/// (a genuine user clear) as opposed to merely resolving to that value (a real
/// color that happens to be zero-alpha, e.g. `"#00000000"`), rather than
/// re-deriving it from the resolved `Color` after the fact — which is exactly
/// the by-value ambiguity `fill_cleared`/`stroke_cleared` exist to avoid.
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
    // theme default" outcome, so `*_is_user_set` is set unconditionally
    // alongside the value it describes.
    if let Some(ref value) = o.stroke {
        style.paint.stroke = Some(resolve_paint_color(value, theme)?);
        style.paint.stroke_is_user_set = true;
        style.paint.stroke_cleared = is_paint_clear_sentinel(value);
    }
    if let Some(ref value) = o.fill {
        style.paint.fill = resolve_paint_color(value, theme)?;
        style.paint.fill_is_user_set = true;
        style.paint.fill_cleared = is_paint_clear_sentinel(value);
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

    // S1: interpolate
    if let Some(ref i) = o.interpolate { style.line.interpolate = Some(i.clone()); }
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
            col_as_str(ctx.batch, field)
                .ok()
                .or_else(|| {
                    col_as_f64(ctx.batch, field).ok().map(|vals| {
                        // For time formatting, compute spacing from actual data range
                        // rather than hardcoding 1 day, so sub-day precision is preserved.
                        let spacing_ms: i64 = if fmt_type == Some("time") {
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
                                86_400_000 // fallback: 1 day
                            }
                        } else {
                            0 // unused for non-time formatting
                        };
                        vals.into_iter()
                            .map(|v| v.map(|f| {
                                if fmt_type == Some("time") {
                                    format_time(f as i64, spacing_ms)
                                } else if fmt.is_some() {
                                    format_with_spec(f, fmt)
                                } else {
                                    format_numeric(f)
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
}

impl MarkBuildResult {
    pub fn empty(kind: MarkBatchKind) -> Self {
        Self { kind, nodes: vec![], data_indices: Some(vec![]), tooltips: None, hrefs: None, descriptions: None }
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

/// Normalize a resolved scene paint, collapsing a **user-cleared** paint to
/// `None` so it never serializes as an explicit `rgba(0,0,0,0.000)` attribute
/// in SVG or as a zero-alpha color in interactive scene JSON — both outputs
/// walk the same [`ferrum_scene::SceneGraph`], so this has to happen at
/// construction, before a `FillStroke`/paint field is built, not at either
/// output's emission step.
///
/// `cleared` is the caller's *provenance* signal (from
/// [`MarkPaint::fill_cleared`]/[`stroke_cleared`](MarkPaint::stroke_cleared)),
/// never a value test on `c` (T8 quality-review c2, spec §4.1 2026-09-01
/// amendment). Gating on `c == TRANSPARENT` instead — the pre-c2 shape of
/// this function — aliased any resolved paint that merely *equals*
/// `color::TRANSPARENT` by value (`fill="#00000000"`, or `opacity=0`
/// composed onto black) with a genuine `"none"`/`"transparent"` clear,
/// silently moving those charts' SVG bytes and violating spec §7's
/// hex-only byte-identity invariant. `cleared` is `true` only when the
/// literal sentinel matched at the mark-style boundary, so a same-valued
/// but never-cleared paint keeps serializing exactly as before.
///
/// Callers must derive `cleared` from state captured *before* any
/// `Option`-based fallback decision (`unwrap_or`, [`resolve_effective_stroke`]'s
/// stroke-width rescue, a color-scale lookup) has run — a scale-resolved
/// per-row color is never "cleared" even if the constant paint underneath it
/// was, and a fallback that substitutes the *other* paint (e.g. stroke
/// rescued from fill) must carry that other paint's own cleared-ness, not
/// this one's.
pub fn normalize_scene_paint(
    c: Option<ferrum_scene::Color>,
    cleared: bool,
) -> Option<ferrum_scene::Color> {
    if cleared { None } else { c }
}

/// Build a [`FillStroke`](ferrum_scene::FillStroke) with `fill_opacity`,
/// `stroke_opacity`, and `angle` left at their neutral defaults (`1.0`, `1.0`,
/// `0.0`).
///
/// This is the common case for marks that do not bind those channels (legend
/// swatches, axis fills, strip backgrounds, area/line group styles). Marks that
/// *do* resolve per-row `fill_opacity`/`stroke_opacity`/`angle` should call
/// [`to_scene_fill_stroke_full`] instead of patching the returned struct
/// afterward — that variant takes all ten fields so the caller receives a
/// finished `FillStroke` in one call.
///
/// `fill_cleared`/`stroke_cleared` sit directly beside the paint they
/// describe (not appended at the end) so a call site reads as "this value,
/// and whether it was cleared" rather than two unlabeled trailing booleans —
/// see [`normalize_scene_paint`] for what they gate. Most callers of this
/// mark family never resolve a user paint-clear at this call site (legend
/// swatches, axis bands, strip backgrounds, area borders, arc/geoshape/polygon
/// fills derived from a color scale) and pass `false, false`.
pub fn to_scene_fill_stroke(
    fill: Option<Color>,
    fill_cleared: bool,
    stroke: Option<Color>,
    stroke_cleared: bool,
    stroke_width: f64,
    opacity: f64,
    stroke_dash: Option<&[f64]>,
) -> ferrum_scene::FillStroke {
    to_scene_fill_stroke_full(
        fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, stroke_dash, 1.0, 1.0,
        0.0,
    )
}

/// Build a complete [`FillStroke`](ferrum_scene::FillStroke), including the
/// `fill_opacity` / `stroke_opacity` / `angle` channels.
///
/// [`to_scene_fill_stroke`] is the convenience wrapper that defaults the last
/// three fields to their neutral values. Marks that resolve those channels
/// per-row (point, rect, area, line) call this directly so they no longer build
/// a partial `FillStroke` and patch the three fields by hand at the call site.
///
/// A **user-cleared** paint (`fill_cleared`/`stroke_cleared` — provenance from
/// [`MarkPaint`], never a value test) is normalized to `None` here via
/// [`normalize_scene_paint`], so it never serializes as an explicit
/// `rgba(0,0,0,0.000)`. A paint that merely *resolves to* `color::TRANSPARENT`
/// without having been cleared (`fill="#00000000"`, or `opacity=0` composed
/// onto black) keeps serializing byte-identically, per spec §7.
#[allow(clippy::too_many_arguments)]
pub fn to_scene_fill_stroke_full(
    fill: Option<Color>,
    fill_cleared: bool,
    stroke: Option<Color>,
    stroke_cleared: bool,
    stroke_width: f64,
    opacity: f64,
    stroke_dash: Option<&[f64]>,
    fill_opacity: f64,
    stroke_opacity: f64,
    angle: f64,
) -> ferrum_scene::FillStroke {
    ferrum_scene::FillStroke {
        fill: normalize_scene_paint(fill.map(to_scene_color), fill_cleared),
        stroke: normalize_scene_paint(stroke.map(to_scene_color), stroke_cleared),
        stroke_width,
        opacity,
        stroke_dash: stroke_dash.map(|d| d.to_vec()),
        stroke_opacity,
        fill_opacity,
        angle,
    }
}

pub fn to_scene_stroke(
    color: Color,
    width: f64,
    opacity: f64,
    dash: Option<&[f64]>,
    cap: Option<&str>,
    join: Option<&str>,
) -> ferrum_scene::StrokeStyle {
    ferrum_scene::StrokeStyle {
        color: to_scene_color(color),
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
/// - `stroke_is_user_set` → the explicit user `stroke=` wins; fall back to fill
///   when `stroke` is somehow `None`.
/// - otherwise → per-row color encoding wins, then the theme stroke, then fill.
///
/// Used by rule/segment (and any axis-aligned/diagonal line mark) so the
/// precedence lives in one place instead of being copy-pasted per renderer.
///
/// The `unwrap_or(fill)` rescue is for a user-set stroke that is somehow
/// `None`; it does NOT fire for an explicit `stroke="none"`, which
/// `resolve_mark_style` resolves to `Some(`[`color::TRANSPARENT`](super::color::TRANSPARENT)`)`
/// — a present, invisible stroke. A cleared stroke therefore stays cleared here
/// too, with no extra case.
pub(crate) fn resolve_stroke_color(ms: &MarkStyle, row_color: Option<Color>) -> Color {
    let p = &ms.paint;
    if p.stroke_is_user_set {
        p.stroke.unwrap_or(p.fill)
    } else {
        row_color.or(p.stroke).unwrap_or(p.fill)
    }
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
/// Returns `(color, cleared)`: `cleared` mirrors `fallback_cleared` whenever
/// the fallback is actually used (no scale, or the scale lookup misses) and
/// is always `false` when a scale lookup succeeds — a color-scale value is
/// never a user paint-clear, however it happens to render (T8 quality-review
/// c2). Callers that don't need per-row cleared provenance may discard the
/// second element.
pub(crate) fn resolve_fill_color(
    scale: Option<&ColorScale>,
    cat_value: Option<&str>,
    num_value: Option<f64>,
    fallback: Color,
    fallback_cleared: bool,
) -> (Color, bool) {
    let Some(s) = scale else { return (fallback, fallback_cleared) };
    match s.input() {
        ColorInput::Numeric => match num_value.filter(|v| v.is_finite()).and_then(|v| s.lookup_f64(v)) {
            Some(c) => (c, false),
            None => (fallback, fallback_cleared),
        },
        ColorInput::Category => match cat_value.and_then(|v| s.lookup(v)) {
            Some(c) => (c, false),
            None => (fallback, fallback_cleared),
        },
    }
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
/// Byte-identical to the inline `if row_stroke_width > 0.0 && base_stroke
/// .is_none() && stroke_width_values.is_some() { Some(fill) } else { base_stroke }`
/// blocks that bar/rect/point each carried.
///
/// NF-A7 (spec §4.1): an explicit `stroke="none"` beats this rescue. It does so
/// structurally, without a case here — `resolve_mark_style` resolves the clear
/// to [`color::TRANSPARENT`](super::color::TRANSPARENT), so `base_stroke` is
/// `Some(transparent)` rather than `None`, the `is_none()` gate never opens, and
/// the row paints no visible stroke however wide the bound width column says.
///
/// Returns `(stroke, cleared)`: `cleared` is `base_stroke_cleared` when the
/// rescue does not fire, and `fill_cleared` when it does (the rescue paints
/// the stroke *in the fill color*, so a cleared fill rescued into the stroke
/// slot is still a cleared paint — T8 quality-review c2). Callers that don't
/// need per-row cleared provenance may discard the second element.
#[inline]
pub(crate) fn resolve_effective_stroke(
    row_stroke_width: f64,
    base_stroke: Option<Color>,
    base_stroke_cleared: bool,
    fill: Color,
    fill_cleared: bool,
    stroke_width_col_present: bool,
) -> (Option<Color>, bool) {
    if row_stroke_width > 0.0 && base_stroke.is_none() && stroke_width_col_present {
        (Some(fill), fill_cleared)
    } else {
        (base_stroke, base_stroke_cleared)
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

pub fn to_scene_text_style(
    color: Color,
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
        color: to_scene_color(color),
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

pub fn dispatch_mark_build(mark: &Mark, ctx: &DrawCtx) -> MarkBuildResult {
    use crate::spec::mark::for_each_mark;
    macro_rules! arm {
        ($($V:ident => $m:ident,)*) => {
            match mark { $( Mark::$V => super::marks::$m::build(ctx), )* }
        };
    }
    for_each_mark!(arm)
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
    // `resolve_mark_style_invalid_fill_leaves_fill_is_user_set_false`, which
    // pinned the pre-batch-A contract: an unparseable `fill=`/`stroke=` left
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
        assert!(style.paint.fill_is_user_set);
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
    /// becomes fully transparent AND `fill_is_user_set` is set, so
    /// `text.rs`'s constant-color fallback uses the cleared paint instead of
    /// silently painting the theme font color.
    #[test]
    fn resolve_mark_style_none_fill_clears_paint_and_marks_user_set() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("none".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        assert_eq!(style.paint.fill.alpha, 0, "'none' must clear the fill");
        assert!(style.paint.fill_is_user_set, "'none' is a user-set clear, not an absent fill");
    }

    /// `stroke="none"` likewise, and the cleared stroke is *present* (a
    /// transparent color) rather than `None`, so no fill fallback resurrects it.
    #[test]
    fn resolve_mark_style_none_stroke_clears_paint_and_marks_user_set() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("none".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule).unwrap();
        let stroke = style.paint.stroke.expect("a cleared stroke is present-but-transparent");
        assert_eq!(stroke.alpha, 0, "'none' must clear the stroke");
        assert!(style.paint.stroke_is_user_set);
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
        assert!(fill_style.paint.fill_is_user_set);

        let stroke_overrides = MarkKwargsSpec { stroke: Some("transparent".into()), ..Default::default() };
        let stroke_style = resolve_mark_style(Some(&stroke_overrides), &theme, &Mark::Rule)
            .unwrap_or_else(|e| panic!("'transparent' must clear the stroke, not refuse: {e}"));
        let stroke = stroke_style.paint.stroke.expect("a cleared stroke is present-but-transparent");
        assert_eq!(stroke.alpha, 0, "'transparent' must clear the stroke");
        assert!(stroke_style.paint.stroke_is_user_set);
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

    // --- fill_is_user_set (Task 4 remediation, spec §4.4, 2026-08-28 T4
    // amendment) ---

    #[test]
    fn resolve_mark_style_no_fill_override_leaves_fill_is_user_set_false() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Text).unwrap();
        assert!(!style.paint.fill_is_user_set);
    }

    #[test]
    fn resolve_mark_style_hex_fill_sets_fill_is_user_set_true() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("#ff0000".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        assert!(style.paint.fill_is_user_set);
        assert_eq!(style.paint.fill.red, 0xff);
    }

    #[test]
    fn resolve_mark_style_theme_label_fill_sentinel_sets_fill_is_user_set_true() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { fill: Some("theme:label".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Text).unwrap();
        assert!(style.paint.fill_is_user_set);
    }

    /// An unparseable `fill=` value never sets `fill_is_user_set` — a
    /// downstream constant-fallback consumer (e.g. `text.rs`) must not be told
    /// the user successfully set a fill. Since batch-A Task 8 that guarantee
    /// holds by refusal rather than by a silently-unset flag: the render stops
    /// before any consumer can read the flag at all. (The refusal itself is
    /// asserted by `resolve_mark_style_invalid_fill_is_typed_error`.)
    #[test]
    fn resolve_mark_style_invalid_fill_never_reports_a_user_set_fill() {
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
        // stroke_is_user_set = true → explicit stroke wins regardless of row color.
        let mut ms = resolve_mark_style(None, &ThemeInputs::default(), &Mark::Rule).unwrap();
        ms.paint.stroke = Some(crate::render::color::from_rgb(0x11, 0x22, 0x33));
        ms.paint.stroke_is_user_set = true;
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
        ms.paint.stroke_is_user_set = true;
        let c = resolve_stroke_color(&ms, Some(crate::render::color::from_rgb(0xAA, 0xBB, 0xCC)));
        assert_eq!((c.red, c.green, c.blue), (0x44, 0x55, 0x66),
            "user-set stroke with None stroke must fall back to fill");
    }

    #[test]
    fn resolve_stroke_color_not_user_set_prefers_row_color() {
        let mut ms = resolve_mark_style(None, &ThemeInputs::default(), &Mark::Rule).unwrap();
        ms.paint.stroke = Some(crate::render::color::from_rgb(0x11, 0x22, 0x33));
        ms.paint.fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        ms.paint.stroke_is_user_set = false;
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
        ms.paint.stroke_is_user_set = false;
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
        assert_eq!(
            resolve_effective_stroke(2.0, None, false, fill, false, true),
            (Some(fill), false),
            "positive width + bound width column + no stroke → stroke painted in the fill color"
        );
        // The gate stays narrow: a constant width (no bound column) does not rescue.
        assert_eq!(resolve_effective_stroke(2.0, None, false, fill, false, false), (None, false));
        // Nor does a zero-width row.
        assert_eq!(resolve_effective_stroke(0.0, None, false, fill, false, true), (None, false));
    }

    /// NF-A7: `stroke="none"` beats the rescue. A user who cleared the stroke
    /// and bound a `stroke_width` column gets NO visible stroke — not a stroke
    /// silently painted in the fill color.
    #[test]
    fn resolve_effective_stroke_cleared_stroke_beats_the_width_rescue() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("none".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rect).unwrap();
        let fill = crate::render::color::from_rgb(0x44, 0x55, 0x66);
        let (effective, cleared) = resolve_effective_stroke(
            2.0, style.paint.stroke, style.paint.stroke_cleared, fill, false, true,
        );
        let effective = effective.expect("a cleared stroke is present-but-transparent");
        assert_eq!(effective.alpha, 0, "the cleared stroke must stay invisible");
        assert_ne!(effective, fill, "the fill must not be resurrected as the stroke");
        assert!(cleared, "the rescue did not fire (base_stroke was Some), so cleared carries through from base_stroke_cleared");
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
        // "a" → palette[0] = red; num_value is ignored for categorical.
        let (c, cleared) = resolve_fill_color(Some(&scale), Some("a"), Some(99.0), fallback, false);
        assert_eq!((c.red, c.green, c.blue), (0xFF, 0x00, 0x00));
        assert!(!cleared, "a scale lookup hit is never cleared");
        // "b" → palette[1] = blue.
        let (c2, _) = resolve_fill_color(Some(&scale), Some("b"), None, fallback, false);
        assert_eq!((c2.red, c2.green, c2.blue), (0x00, 0x00, 0xFF));
        // Unknown category → fallback.
        let (c3, cleared3) = resolve_fill_color(Some(&scale), Some("z"), None, fallback, true);
        assert_eq!((c3.red, c3.green, c3.blue), (0x77, 0x77, 0x77));
        assert!(cleared3, "a scale-lookup miss falls back to fallback_cleared");
        // No category string → fallback.
        let (c4, _) = resolve_fill_color(Some(&scale), None, None, fallback, false);
        assert_eq!((c4.red, c4.green, c4.blue), (0x77, 0x77, 0x77));
    }

    #[test]
    fn resolve_fill_color_continuous_uses_lookup_f64() {
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        use crate::render::scale_resolve::ColorScale;
        let scheme = ContinuousScheme::Named(NamedContinuous::Viridis);
        let scale = ColorScale::Continuous { domain: (0.0, 10.0), scheme, midpoint: None };
        let fallback = crate::render::color::from_rgb(0x77, 0x77, 0x77);
        // The continuous branch must consume the f64 directly (no round-trip):
        // the resolved color equals scale.lookup_f64(value).
        let v = 3.0;
        let (c, cleared) = resolve_fill_color(Some(&scale), Some("ignored"), Some(v), fallback, false);
        let expected = scale.lookup_f64(v).unwrap();
        assert_eq!((c.red, c.green, c.blue), (expected.red, expected.green, expected.blue));
        assert!(!cleared, "a scale lookup hit is never cleared");
        // Non-finite / absent numeric → fallback.
        let (c2, _) = resolve_fill_color(Some(&scale), None, Some(f64::NAN), fallback, false);
        assert_eq!((c2.red, c2.green, c2.blue), (0x77, 0x77, 0x77));
        let (c3, _) = resolve_fill_color(Some(&scale), None, None, fallback, false);
        assert_eq!((c3.red, c3.green, c3.blue), (0x77, 0x77, 0x77));
    }

    #[test]
    fn resolve_fill_color_no_scale_returns_fallback() {
        let fallback = crate::render::color::from_rgb(0x12, 0x34, 0x56);
        let (c, cleared) = resolve_fill_color(None, Some("a"), Some(1.0), fallback, true);
        assert_eq!((c.red, c.green, c.blue), (0x12, 0x34, 0x56));
        assert!(cleared, "no scale → fallback_cleared passes through unchanged");
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
            from_rgb(0, 0, 0),
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
            from_rgb(0, 0, 0),
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
}
