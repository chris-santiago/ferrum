//! Per-panel draw context + mark dispatch. Spec §4.5 / §4.6.

use arrow::record_batch::RecordBatch;

use crate::layout::{PanelLayout, ThemeInputs};
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;

use ferrum_scene::{
    MarkBatchKind, SceneNode as FsSceneNode, TooltipContent as FsTooltipContent,
    TooltipField as FsTooltipField,
};

use super::color::{from_hex_str, with_opacity, Color};
use super::scale_resolve::ResolvedScales;

pub struct DrawCtx<'a> {
    pub spec: &'a crate::spec::chart::ChartSpec,
    pub panel: &'a PanelLayout,
    pub theme: &'a ThemeInputs,
    pub scales: &'a ResolvedScales,
    pub batch: &'a RecordBatch,
    pub mark_style: &'a MarkStyle,
}

/// Per-mark resolved style. Fields are populated from theme defaults (mark-aware)
/// and then overridden by any `MarkKwargsSpec` present on the layer or chart.
///
/// Text-mark-specific fields (`font_size`, `font_weight`, `align`, `baseline`,
/// `dx`, `dy`, `angle`) are stored here as `Option<>` and default to `None`.
/// Per-mark draw functions for text marks read them; non-text marks ignore them.
#[derive(Debug, Clone)]
pub struct MarkStyle {
    pub fill: Color,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub point_size: f64,
    pub corner_radius: f64,
    pub stroke_dash: Option<Vec<f64>>,
    // Text-mark-only fields (None = fall back to theme/hardcoded defaults).
    pub font_size: Option<f64>,
    pub font_weight: Option<String>,
    pub align: Option<String>,
    pub baseline: Option<String>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub angle: Option<f64>,
    // Polygon-mark-only fields (None = no detail grouping / default cmap)
    pub detail: Option<String>,
    pub cmap: Option<String>,
    // ── S1: interpolate (line/area) ──
    pub interpolate: Option<String>,
    // ── S2: stroke_cap (line) ────────
    pub stroke_cap: Option<String>,
    // ── S3: stroke_join (line/area) ──
    pub stroke_join: Option<String>,
    // ── S5: filled (point) ───────────
    pub filled: Option<bool>,
    // ── S6: shape (point, constant) ──
    pub shape: Option<String>,
    // ── S7: limit (text) ─────────────
    pub limit: Option<usize>,
    // ── S8: band_size (tick/rect) ────
    pub band_size: Option<f64>,
    // ── S9: line (area) ──────────────
    pub line_border: Option<bool>,
    // ── S10: borders (area) ──────────
    pub borders: Option<bool>,
    // ── mark_image URL-tile sizing ───
    pub width: Option<f64>,
    pub height: Option<f64>,
    // ── S11: leader_line (label) ─────
    pub leader_line: Option<bool>,
}

impl MarkStyle {
    /// Theme-driven base style with `fill = mark_color × default_opacity`,
    /// no stroke, no stroke width, and every text/polygon-only field unset.
    /// All per-mark variants in `resolve_mark_style` are 1-5 field overrides
    /// applied on top of this baseline. Matches the prior Tick/Text/Image
    /// arm byte-for-byte.
    fn theme_base(theme: &ThemeInputs) -> Self {
        MarkStyle {
            fill: with_opacity(theme.mark_color, theme.default_opacity),
            stroke: None,
            stroke_width: 0.0,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
            font_size: None,
            font_weight: None,
            align: None,
            baseline: None,
            dx: None,
            dy: None,
            angle: None,
            detail: None,
            cmap: None,
            interpolate: None,
            stroke_cap: None,
            stroke_join: None,
            filled: None,
            shape: None,
            limit: None,
            band_size: None,
            line_border: None,
            borders: None,
            width: None,
            height: None,
            leader_line: None,
        }
    }
}

/// Build the mark-aware theme base and then apply any `MarkKwargsSpec` overrides.
///
/// When `overrides` is `None`, the result is identical to the Phase 7 path
/// (pure theme defaults, mark-aware) — goldens remain byte-identical.
///
/// String color fields (stroke, fill) are parsed via `from_hex_str`; parse
/// failures are silently skipped (warn at the Python layer per spec).
pub fn resolve_mark_style(
    overrides: Option<&MarkKwargsSpec>,
    theme: &ThemeInputs,
    mark: &Mark,
) -> MarkStyle {
    // Mark-aware deltas from the theme baseline. Only fields that differ
    // from `MarkStyle::theme_base` are written; everything else falls
    // through to the baseline value.
    let mut style = MarkStyle::theme_base(theme);
    match mark {
        Mark::Area | Mark::Ribbon | Mark::Polygon => {
            style.fill = theme.mark_color;
            style.stroke = None;
            style.stroke_width = 0.0;
            style.opacity = theme.area_opacity;
        }
        Mark::Line => {
            style.fill = theme.mark_color;
            style.stroke = Some(theme.mark_color);
            style.stroke_width = theme.line_stroke_width;
        }
        Mark::Bar | Mark::Rect => {
            style.corner_radius = theme.bar_corner_radius;
        }
        Mark::Rule => {
            // Reference-line defaults from theme; non-reference rules
            // (boxplot whiskers, error bars) override via mark_kwargs.
            style.fill = theme.reference_line_color;
            style.stroke = Some(theme.reference_line_color);
            style.stroke_width = theme.line_stroke_width;
            style.stroke_dash = theme.reference_line_dash.clone();
        }
        Mark::Segment => {
            style.fill = theme.mark_color;
            style.stroke = Some(theme.mark_color);
            style.stroke_width = theme.line_stroke_width;
        }
        Mark::Point => {
            style.opacity = theme.point_opacity;
        }
        Mark::Tick | Mark::Text | Mark::Image | Mark::Label => {
            // Baseline applies as-is.
        }
        Mark::Arc => {
            style.stroke_width = 0.0;
        }
        Mark::Geoshape => {
            style.fill = theme.mark_color;
            style.stroke = Some(theme.mark_color);
            style.stroke_width = 0.5;
        }
    }

    // --- Apply MarkKwargsSpec overrides (if any) ---
    let Some(o) = overrides else { return style };

    if let Some(size) = o.size { style.point_size = size; }
    if let Some(opacity) = o.opacity { style.opacity = opacity; }
    if let Some(cr) = o.corner_radius { style.corner_radius = cr; }
    if let Some(sw) = o.stroke_width { style.stroke_width = sw; }
    // Empty vec = clear the dash (solid line); non-empty = set explicitly.
    if let Some(ref dash) = o.stroke_dash {
        style.stroke_dash = if dash.is_empty() { None } else { Some(dash.clone()) };
    }

    if let Some(ref hex) = o.stroke {
        if hex == "theme:label" {
            style.stroke = Some(theme.label_color);
        } else if let Ok(c) = from_hex_str(hex) {
            style.stroke = Some(c);
        }
        // other parse failure: silently skip; warn at Python layer
    }
    if let Some(ref hex) = o.fill {
        if hex == "theme:label" {
            style.fill = theme.label_color;
        } else if let Ok(c) = from_hex_str(hex) {
            style.fill = c;
        }
    }

    // Text-mark-specific fields
    if let Some(fs) = o.font_size { style.font_size = Some(fs); }
    if let Some(ref fw) = o.font_weight { style.font_weight = Some(fw.clone()); }
    if let Some(ref al) = o.align { style.align = Some(al.clone()); }
    if let Some(ref bl) = o.baseline { style.baseline = Some(bl.clone()); }
    if let Some(dx) = o.dx { style.dx = Some(dx); }
    if let Some(dy) = o.dy { style.dy = Some(dy); }
    if let Some(ang) = o.angle { style.angle = Some(ang); }

    // Polygon-mark-only fields
    if let Some(ref d) = o.detail { style.detail = Some(d.clone()); }
    if let Some(ref c) = o.cmap { style.cmap = Some(c.clone()); }

    // S1: interpolate
    if let Some(ref i) = o.interpolate { style.interpolate = Some(i.clone()); }
    // S2: stroke_cap
    if let Some(ref sc) = o.stroke_cap { style.stroke_cap = Some(sc.clone()); }
    // S3: stroke_join
    if let Some(ref sj) = o.stroke_join { style.stroke_join = Some(sj.clone()); }
    // S5: filled
    if let Some(f) = o.filled { style.filled = Some(f); }
    // S6: shape (constant)
    if let Some(ref sh) = o.shape { style.shape = Some(sh.clone()); }
    // S7: limit
    if let Some(l) = o.limit { style.limit = Some(l); }
    // S8: band_size
    if let Some(bs) = o.band_size { style.band_size = Some(bs); }
    // S9: line border on area
    if let Some(lb) = o.line { style.line_border = Some(lb); }
    // S10: borders on area
    if let Some(b) = o.borders { style.borders = Some(b); }
    // mark_image URL-tile sizing
    if let Some(w) = o.width { style.width = Some(w); }
    if let Some(h) = o.height { style.height = Some(h); }
    // S11: leader_line (label)
    if let Some(ll) = o.leader_line { style.leader_line = Some(ll); }

    style
}

pub(crate) use super::arrow_cast::{col_as_f64, col_as_str};

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
                read_col_with_spec(&e.field, fmt, fmt_type).map(|col| (e.field.clone(), col))
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

impl MetadataColumns {
    pub fn build_metadata(
        &self,
        _ctx: &DrawCtx,
    ) -> (Option<Vec<FsTooltipContent>>, Option<Vec<Option<String>>>, Option<Vec<Option<String>>>) {
        let tooltips = if self.tooltip_cols.is_empty() {
            None
        } else {
            // Use the first col's length as the row count; all cols should have the same length.
            let n = self.tooltip_cols.first().map(|(_, c)| c.len()).unwrap_or(0);
            Some((0..n).map(|i| {
                FsTooltipContent {
                    fields: self.tooltip_cols.iter()
                        .map(|(name, col)| FsTooltipField {
                            name: name.clone(),
                            value: col.get(i).and_then(|v| v.clone()).unwrap_or_default(),
                        })
                        .collect(),
                }
            }).collect())
        };
        let hrefs = self.href.clone();
        let descriptions = self.description.clone();
        (tooltips, hrefs, descriptions)
    }
}

pub fn to_scene_color(c: Color) -> ferrum_scene::Color {
    ferrum_scene::Color { r: c.red, g: c.green, b: c.blue, a: c.alpha }
}

pub fn to_scene_fill_stroke(
    fill: Option<Color>,
    stroke: Option<Color>,
    stroke_width: f64,
    opacity: f64,
    stroke_dash: Option<&[f64]>,
) -> ferrum_scene::FillStroke {
    ferrum_scene::FillStroke {
        fill: fill.map(to_scene_color),
        stroke: stroke.map(to_scene_color),
        stroke_width,
        opacity,
        stroke_dash: stroke_dash.map(|d| d.to_vec()),
        stroke_opacity: 1.0,
        fill_opacity: 1.0,
        angle: 0.0,
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
        // Unknown strings return None so SVG uses its default (butt/miter),
        // consistent with parse_stroke_cap / parse_stroke_join.
        stroke_cap: cap.and_then(|s| match s {
            "round" => Some(ferrum_scene::StrokeCap::Round),
            "square" => Some(ferrum_scene::StrokeCap::Square),
            "butt" => Some(ferrum_scene::StrokeCap::Butt),
            _ => None,
        }),
        stroke_join: join.and_then(|s| match s {
            "round" => Some(ferrum_scene::StrokeJoin::Round),
            "bevel" => Some(ferrum_scene::StrokeJoin::Bevel),
            "miter" => Some(ferrum_scene::StrokeJoin::Miter),
            _ => None,
        }),
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
            Some("hanging") | Some("text-before-edge") => ferrum_scene::TextBaseline::Top,
            Some("central") | Some("middle") => ferrum_scene::TextBaseline::Middle,
            Some("ideographic") | Some("text-after-edge") => ferrum_scene::TextBaseline::Bottom,
            Some(other) => ferrum_scene::TextBaseline::Custom(other.to_string()),
            None => ferrum_scene::TextBaseline::Alphabetic,
        },
        angle,
        color: to_scene_color(color),
        opacity,
        font_family: font_family.to_string(),
    }
}

/// Map a stroke-dash palette index to its canonical SVG dasharray pattern.
///
/// The index is rounded and clamped to `[0, 3]` before lookup:
/// - `0` → solid (returns `None`)
/// - `1` → long dash `[6, 3]`
/// - `2` → short dash / dot `[2, 3]`
/// - `3` → long-short dash `[6, 3, 2, 3]`
pub(crate) fn resolve_stroke_dash(idx: f64) -> Option<Vec<f64>> {
    let idx = (idx.round() as i64).clamp(0, 3);
    match idx {
        1 => Some(vec![6.0, 3.0]),
        2 => Some(vec![2.0, 3.0]),
        3 => Some(vec![6.0, 3.0, 2.0, 3.0]),
        _ => None,
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

    // --- Phase 7 baseline tests (updated to 3-arg signature; None overrides = same result) ---

    #[test]
    fn resolve_style_for_area_uses_area_opacity() {
        // Area fill is opaque; area_opacity is carried in style.opacity so the
        // renderer can apply it (and user opacity kwarg can override it).
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Area);
        assert_eq!(style.fill.alpha, 0xFF, "area fill should be opaque");
        assert!((style.opacity - theme.area_opacity).abs() < 1e-6,
            "area opacity should default to theme.area_opacity");
    }

    #[test]
    fn resolve_style_for_bar_has_corner_radius_from_theme() {
        let mut theme = ThemeInputs::default();
        theme.bar_corner_radius = 4.0;
        let style = resolve_mark_style(None, &theme, &Mark::Bar);
        assert_eq!(style.corner_radius, 4.0);
    }

    #[test]
    fn resolve_style_for_point_is_opaque_by_default() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Point);
        assert_eq!(style.fill.alpha, 0xFF);
    }

    // --- Phase 8a Task 7 tests ---

    #[test]
    fn resolve_mark_style_with_no_overrides_returns_theme_defaults() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Point);
        assert_eq!(style.point_size, theme.point_size);
    }

    #[test]
    fn resolve_mark_style_overrides_point_size() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { size: Some(100.0), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point);
        assert_eq!(style.point_size, 100.0);
    }

    #[test]
    fn resolve_mark_style_overrides_stroke_color() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("#ff0000".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point);
        let stroke = style.stroke.expect("stroke should be set");
        assert_eq!(stroke.red, 0xff);
        assert_eq!(stroke.green, 0x00);
        assert_eq!(stroke.blue, 0x00);
    }

    #[test]
    fn resolve_mark_style_invalid_color_silently_skipped() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { stroke: Some("not-a-color".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point);
        // Mark::Point theme default stroke is None; invalid color does NOT set it
        let baseline = resolve_mark_style(None, &theme, &Mark::Point);
        assert_eq!(style.stroke, baseline.stroke);
    }

    // --- theme:label sentinel tests ---

    #[test]
    fn resolve_mark_style_stroke_theme_label_sentinel_uses_label_color() {
        let mut theme = ThemeInputs::default();
        theme.label_color = palette::Srgba::new(0x11, 0x22, 0x33, 0xFF);
        let overrides = MarkKwargsSpec { stroke: Some("theme:label".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule);
        let stroke = style.stroke.expect("stroke must be set by sentinel");
        assert_eq!(stroke.red,   0x11, "sentinel stroke.red must be label_color.red");
        assert_eq!(stroke.green, 0x22, "sentinel stroke.green must be label_color.green");
        assert_eq!(stroke.blue,  0x33, "sentinel stroke.blue must be label_color.blue");
    }

    #[test]
    fn resolve_mark_style_fill_theme_label_sentinel_uses_label_color() {
        let mut theme = ThemeInputs::default();
        theme.label_color = palette::Srgba::new(0x44, 0x55, 0x66, 0xFF);
        let overrides = MarkKwargsSpec { fill: Some("theme:label".into()), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Tick);
        assert_eq!(style.fill.red,   0x44, "sentinel fill.red must be label_color.red");
        assert_eq!(style.fill.green, 0x55, "sentinel fill.green must be label_color.green");
        assert_eq!(style.fill.blue,  0x66, "sentinel fill.blue must be label_color.blue");
    }

    #[test]
    fn resolve_mark_style_empty_stroke_dash_clears_reference_line_dash() {
        // Rule mark picks up reference_line_dash from theme by default.
        // Passing stroke_dash: [] should clear it (solid line for composite structural rules).
        let theme = ThemeInputs::default(); // reference_line_dash = Some([4.0, 4.0])
        assert!(theme.reference_line_dash.is_some(), "test requires non-None reference_line_dash");
        let overrides = MarkKwargsSpec { stroke_dash: Some(vec![]), ..Default::default() };
        let style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule);
        assert!(style.stroke_dash.is_none(), "empty stroke_dash override must clear the dash");
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
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
            selections: Vec::new(), conditionals: Vec::new(), chart_description: None,
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
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
}
