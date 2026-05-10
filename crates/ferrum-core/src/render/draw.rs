//! Per-panel draw context + mark dispatch. Spec §4.5 / §4.6.

use arrow::array::{Array, Float64Array, Int64Array, StringArray, TimestampMillisecondArray};
use arrow::record_batch::RecordBatch;

use crate::layout::{PanelLayout, ThemeInputs};
use crate::spec::mark::Mark;

use super::color::{with_opacity, Color};
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

#[derive(Debug, Clone)]
pub struct MarkStyle {
    pub fill: Color,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub point_size: f64,
    pub corner_radius: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

pub fn resolve_mark_style(theme: &ThemeInputs, mark: &Mark) -> MarkStyle {
    let base_fill = with_opacity(theme.mark_color, theme.default_opacity);
    match mark {
        Mark::Area => MarkStyle {
            fill: with_opacity(theme.mark_color, theme.area_opacity),
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: 1.0,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
        Mark::Line => MarkStyle {
            fill: theme.mark_color,
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
        Mark::Bar | Mark::Rect => MarkStyle {
            fill: base_fill,
            stroke: None,
            stroke_width: 0.0,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: theme.bar_corner_radius,
            stroke_dash: None,
        },
        Mark::Rule => MarkStyle {
            fill: theme.mark_color,
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
        Mark::Tick | Mark::Point | Mark::Text => MarkStyle {
            fill: base_fill,
            stroke: None,
            stroke_width: 0.0,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
    }
}

/// Try to read a column as `f64`, regardless of whether the underlying type
/// is Float64 / Int64 / Timestamp(ms). Returns None for null rows.
pub fn col_as_f64(batch: &RecordBatch, field: &str) -> Result<Vec<Option<f64>>, super::RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| super::RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        Ok(a.iter().collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_style_for_area_uses_area_opacity() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(&theme, &Mark::Area);
        assert!((style.fill.alpha as i32 - 102).abs() <= 1);
    }

    #[test]
    fn resolve_style_for_bar_has_corner_radius_from_theme() {
        let mut theme = ThemeInputs::default();
        theme.bar_corner_radius = 4.0;
        let style = resolve_mark_style(&theme, &Mark::Bar);
        assert_eq!(style.corner_radius, 4.0);
    }

    #[test]
    fn resolve_style_for_point_is_opaque_by_default() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(&theme, &Mark::Point);
        assert_eq!(style.fill.alpha, 0xFF);
    }
}
