//! Per-panel draw context + mark dispatch. Spec §4.5 / §4.6.

use arrow::array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, StringArray,
    TimestampMillisecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::record_batch::RecordBatch;

use crate::layout::{PanelLayout, ThemeInputs};
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;

use super::color::{from_hex_str, with_opacity, Color};
use super::scale_resolve::ResolvedScales;
use super::svg::SvgBuffer;

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
    // --- Mark-aware theme base (preserves Phase 7 behaviour exactly) ---
    let base_fill = with_opacity(theme.mark_color, theme.default_opacity);
    let mut style = match mark {
        Mark::Area | Mark::Ribbon => MarkStyle {
            fill: with_opacity(theme.mark_color, theme.area_opacity),
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: 1.0,
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
        },
        Mark::Line => MarkStyle {
            fill: theme.mark_color,
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
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
        },
        Mark::Bar | Mark::Rect => MarkStyle {
            fill: base_fill,
            stroke: None,
            stroke_width: 0.0,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: theme.bar_corner_radius,
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
        },
        Mark::Rule | Mark::Segment => MarkStyle {
            fill: theme.mark_color,
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
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
        },
        Mark::Polygon => MarkStyle {
            fill: with_opacity(theme.mark_color, theme.area_opacity),
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: 1.0,
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
        },
        Mark::Tick | Mark::Point | Mark::Text | Mark::Image => MarkStyle {
            fill: base_fill,
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
        },
    };

    // --- Apply MarkKwargsSpec overrides (if any) ---
    let Some(o) = overrides else { return style };

    if let Some(size) = o.size { style.point_size = size; }
    if let Some(opacity) = o.opacity { style.opacity = opacity; }
    if let Some(cr) = o.corner_radius { style.corner_radius = cr; }
    if let Some(sw) = o.stroke_width { style.stroke_width = sw; }
    if let Some(ref dash) = o.stroke_dash { style.stroke_dash = Some(dash.clone()); }

    if let Some(ref hex) = o.stroke {
        if let Ok(c) = from_hex_str(hex) {
            style.stroke = Some(c);
        }
        // parse failure: silently skip; warn at Python layer
    }
    if let Some(ref hex) = o.fill {
        if let Ok(c) = from_hex_str(hex) {
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

    style
}

/// Try to read a column as `f64`, regardless of the underlying numeric type.
///
/// Polars emits `UInt32` from `.len()` (group_by counts), `Int32`/`Int16`/`Int8`
/// from many builders, and `Float32` from some compute paths — all of these
/// must coerce cleanly so renderers see a uniform `f64` view rather than
/// silently aborting via `Err -> early return`. Returns `None` for null rows.
pub fn col_as_f64(batch: &RecordBatch, field: &str) -> Result<Vec<Option<f64>>, super::RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| super::RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        Ok(a.iter().collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int16Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int8Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<UInt64Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<UInt32Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<UInt16Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<UInt8Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<TimestampMillisecondArray>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else {
        Err(super::RenderError::ScaleResolutionFailed(
            format!("column '{field}' has unsupported dtype for f64 read: {:?}", col.data_type())
        ))
    }
}

/// Read a column as Vec<Option<String>>.
pub fn col_as_str(batch: &RecordBatch, field: &str) -> Result<Vec<Option<String>>, super::RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| super::RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect())
    } else {
        Err(super::RenderError::ScaleResolutionFailed(
            format!("column '{field}' must be Utf8 to read as strings: {:?}", col.data_type())
        ))
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

pub fn dispatch_mark(mark: &Mark, ctx: &DrawCtx, out: &mut SvgBuffer) {
    match mark {
        Mark::Point => super::marks::point::draw(ctx, out),
        Mark::Line  => super::marks::line::draw(ctx, out),
        Mark::Area  => super::marks::area::draw(ctx, out),
        Mark::Bar   => super::marks::bar::draw(ctx, out),
        Mark::Rect  => super::marks::rect::draw(ctx, out),
        Mark::Rule  => super::marks::rule::draw(ctx, out),
        Mark::Text  => super::marks::text::draw(ctx, out),
        Mark::Tick  => super::marks::tick::draw(ctx, out),
        Mark::Polygon => super::marks::polygon::draw(ctx, out),
        Mark::Image => super::marks::image::draw(ctx, out),
        Mark::Ribbon => super::marks::ribbon::draw(ctx, out),
        Mark::Segment => super::marks::segment::draw(ctx, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Phase 7 baseline tests (updated to 3-arg signature; None overrides = same result) ---

    #[test]
    fn resolve_style_for_area_uses_area_opacity() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(None, &theme, &Mark::Area);
        assert!((style.fill.alpha as i32 - 102).abs() <= 1);
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
}
