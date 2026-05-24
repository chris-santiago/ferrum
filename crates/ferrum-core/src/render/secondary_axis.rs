//! Secondary Y axis — independent right-side Y scale and marks.
//!
//! Dual-axis charts show two quantitative ranges on the same panel: the primary
//! Y scale on the left (built by `scale_resolve`) and a secondary Y scale on the
//! right driven by a different field. This module resolves the Y2 scale from the
//! field's own min/max and builds the right-side axis nodes and the marks that
//! should be rendered against it.

use arrow::record_batch::RecordBatch;
use ferrum_scene::{FillStroke, SceneNode};

use crate::layout::{Rect, TextAnchor as LayoutAnchor, ThemeInputs};
use crate::render::arrow_cast;
use crate::render::chart_config::SecondaryYSpec;
use crate::render::color::{parse_color, with_opacity};
use crate::render::draw::{to_scene_color, to_scene_stroke, to_scene_text_style};
use crate::render::format::format_numeric;
use crate::render::scale_resolve::ScaleKind;
use crate::render::RenderError;
use crate::scale::linear::LinearScale;

// ── Scale resolution ─────────────────────────────────────────────────────────

/// Compute an independent linear Y scale from `field` in `batch`.
///
/// The scale maps the field's `[min, max]` to the same pixel range as the
/// primary Y axis (`y_pixel_range` is `(plot_top, plot_bottom)` in screen
/// pixel coordinates, where the top pixel value is smaller than the bottom).
///
/// Returns `UnknownColumn` when the column is absent.
pub fn resolve_secondary_y_scale(
    field: &str,
    batch: &RecordBatch,
    y_pixel_range: (f64, f64),
) -> Result<ScaleKind, RenderError> {
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;

    let (mut mn, mut mx) = arrow_cast::finite_min_max_f64(col.as_ref())
        .unwrap_or((0.0, 1.0));

    if mn == mx {
        mn -= 0.5;
        mx += 0.5;
    }

    // Y scale is inverted: high data value → small pixel (top of plot).
    Ok(ScaleKind::Linear(LinearScale::new_internal(
        vec![mn, mx],
        vec![y_pixel_range.1, y_pixel_range.0],
        false,
        false,
    )))
}

// ── Axis rendering ────────────────────────────────────────────────────────────

/// Build right-side axis nodes (domain line, ticks, labels) for the secondary Y scale.
pub fn build_secondary_axis(
    scale: &ScaleKind,
    plot_area: &Rect,
    theme: &ThemeInputs,
    _spec: &SecondaryYSpec,
) -> Vec<SceneNode> {
    let mut nodes = Vec::new();

    let axis_x = plot_area.x + plot_area.w;
    let plot_top = plot_area.y;
    let plot_bottom = plot_area.y + plot_area.h;

    let axis_stroke = to_scene_stroke(
        theme.axis_line_color,
        theme.axis_line_width,
        1.0,
        None,
        None,
        None,
    );
    let tick_stroke = to_scene_stroke(theme.tick_color, theme.tick_width, 1.0, None, None, None);

    // Domain line on the right side of the plot area.
    if theme.axis_line {
        nodes.push(SceneNode::Line {
            x1: axis_x,
            y1: plot_top,
            x2: axis_x,
            y2: plot_bottom,
            style: axis_stroke,
        });
    }

    let (data_lo, data_hi) = match scale.data_domain() {
        Some(d) => d,
        None => return nodes, // ordinal — shouldn't happen for secondary Y
    };

    let tick_count = 5usize;
    let step = (data_hi - data_lo) / (tick_count as f64);
    let tick_values: Vec<f64> = (0..=tick_count)
        .map(|i| data_lo + step * (i as f64))
        .collect();

    for v in &tick_values {
        let py = match scale.to_pixel_f64(*v) {
            Some(p) => p,
            None => continue,
        };
        if py < plot_top - 1.0 || py > plot_bottom + 1.0 {
            continue;
        }

        // Tick mark pointing right.
        nodes.push(SceneNode::Line {
            x1: axis_x,
            y1: py,
            x2: axis_x + theme.tick_size,
            y2: py,
            style: tick_stroke.clone(),
        });

        // Label to the right of the tick.
        nodes.push(SceneNode::Text {
            x: axis_x + theme.tick_size + 2.0,
            y: py + theme.label_font_size / 3.0,
            content: format_numeric(*v),
            style: to_scene_text_style(
                theme.label_color,
                theme.label_font_size,
                LayoutAnchor::Start,
                0.0,
                &theme.label_font_family,
                None,
                None,
                1.0,
            ),
        });
    }

    nodes
}

// ── Mark rendering ────────────────────────────────────────────────────────────

/// Build marks (line or point) against the secondary Y scale.
///
/// `x_field` is the column name used by the primary X encoding; `x_scale` maps
/// its values to pixel positions. `spec.field` drives the secondary Y axis.
pub fn build_secondary_marks(
    batch: &RecordBatch,
    x_field: &str,
    x_scale: &ScaleKind,
    y2_scale: &ScaleKind,
    spec: &SecondaryYSpec,
    _plot_area: &Rect,
) -> Vec<SceneNode> {
    let mark_color = spec
        .color
        .as_deref()
        .and_then(|hex| parse_color(hex).ok())
        .unwrap_or_else(|| crate::render::color::from_rgb(0xE4, 0x57, 0x56));
    let opacity = spec.opacity.unwrap_or(1.0);
    let color_with_alpha = with_opacity(mark_color, opacity);

    let x_vals: Vec<Option<f64>> = match arrow_cast::col_as_f64(batch, x_field) {
        Ok(vs) => vs.into_iter().map(|v| v.and_then(|x| x_scale.to_pixel_f64(x))).collect(),
        Err(_) => return Vec::new(),
    };
    let y2_vals: Vec<Option<f64>> = match arrow_cast::col_as_f64(batch, &spec.field) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let n = x_vals.len().min(y2_vals.len());
    if n == 0 {
        return Vec::new();
    }

    match spec.mark.as_str() {
        "point" | "circle" => {
            build_point_marks(&x_vals[..n], &y2_vals[..n], y2_scale, color_with_alpha)
        }
        _ => build_line_mark(&x_vals[..n], &y2_vals[..n], y2_scale, color_with_alpha),
    }
}

fn build_line_mark(
    x_px: &[Option<f64>],
    y_data: &[Option<f64>],
    y2_scale: &ScaleKind,
    color: crate::render::color::Color,
) -> Vec<SceneNode> {
    use ferrum_scene::PathCmd;

    let mut commands = Vec::new();
    let mut started = false;

    for (xv, yv) in x_px.iter().zip(y_data.iter()) {
        let (Some(px), Some(raw_y)) = (xv, yv) else {
            started = false;
            continue;
        };
        let Some(py) = y2_scale.to_pixel_f64(*raw_y) else {
            started = false;
            continue;
        };
        if !started {
            commands.push(PathCmd::MoveTo { x: *px, y: py });
            started = true;
        } else {
            commands.push(PathCmd::LineTo { x: *px, y: py });
        }
    }

    if commands.is_empty() {
        return Vec::new();
    }

    let stroke_color = to_scene_color(color);
    vec![SceneNode::Path {
        commands,
        style: FillStroke {
            fill: None,
            stroke: Some(stroke_color),
            stroke_width: 1.5,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        closed: false,
    }]
}

fn build_point_marks(
    x_px: &[Option<f64>],
    y_data: &[Option<f64>],
    y2_scale: &ScaleKind,
    color: crate::render::color::Color,
) -> Vec<SceneNode> {
    let fill_color = to_scene_color(color);
    let mut nodes = Vec::new();
    for (xv, yv) in x_px.iter().zip(y_data.iter()) {
        let (Some(px), Some(raw_y)) = (xv, yv) else { continue };
        let Some(py) = y2_scale.to_pixel_f64(*raw_y) else { continue };
        nodes.push(SceneNode::Circle {
            cx: *px,
            cy: py,
            r: 3.0,
            style: FillStroke {
                fill: Some(fill_color),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
        });
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("revenue", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
                Arc::new(Float64Array::from(vec![100.0, 150.0, 200.0, 250.0, 300.0])),
            ],
        )
        .unwrap()
    }

    fn linear_scale(lo: f64, hi: f64, px_lo: f64, px_hi: f64) -> ScaleKind {
        ScaleKind::Linear(LinearScale::new_internal(
            vec![lo, hi],
            vec![px_lo, px_hi],
            false,
            false,
        ))
    }

    #[test]
    fn resolve_secondary_y_scale_basic() {
        let batch = make_batch();
        let scale = resolve_secondary_y_scale("revenue", &batch, (50.0, 350.0)).unwrap();
        // min=100 → pixel 350 (bottom), max=300 → pixel 50 (top)
        let px_min = scale.to_pixel_f64(100.0).unwrap();
        let px_max = scale.to_pixel_f64(300.0).unwrap();
        assert!(
            px_min > px_max,
            "min data value should map to lower pixel (bottom); got min_px={px_min}, max_px={px_max}"
        );
    }

    #[test]
    fn resolve_secondary_y_scale_missing_column_errors() {
        let batch = make_batch();
        let err = resolve_secondary_y_scale("missing", &batch, (50.0, 350.0)).unwrap_err();
        assert!(matches!(err, RenderError::UnknownColumn { .. }));
    }

    #[test]
    fn resolve_secondary_y_scale_degenerate_domain() {
        let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0]))],
        )
        .unwrap();
        let scale = resolve_secondary_y_scale("y", &batch, (50.0, 350.0)).unwrap();
        // Should not panic; degenerate domain expanded ±0.5.
        let px = scale.to_pixel_f64(5.0);
        assert!(px.is_some(), "expected finite pixel for midpoint of degenerate domain");
    }

    #[test]
    fn build_secondary_axis_returns_nodes() {
        let scale = linear_scale(0.0, 100.0, 350.0, 50.0);
        let plot_area = Rect { x: 50.0, y: 50.0, w: 300.0, h: 300.0 };
        let theme = ThemeInputs::default();
        let spec = SecondaryYSpec::default();
        let nodes = build_secondary_axis(&scale, &plot_area, &theme, &spec);
        assert!(!nodes.is_empty(), "expected at least axis domain line");
        let has_line = nodes.iter().any(|n| matches!(n, SceneNode::Line { .. }));
        assert!(has_line, "expected at least one line node");
    }

    #[test]
    fn build_secondary_marks_line_returns_path() {
        let batch = make_batch();
        let x_scale = linear_scale(1.0, 5.0, 50.0, 350.0);
        let y2_scale = linear_scale(0.0, 400.0, 350.0, 50.0);
        let spec = SecondaryYSpec {
            field: "revenue".to_string(),
            mark: "line".to_string(),
            color: Some("#e45756".to_string()),
            opacity: Some(0.9),
        };
        let plot_area = Rect { x: 50.0, y: 50.0, w: 300.0, h: 300.0 };
        let nodes = build_secondary_marks(&batch, "x", &x_scale, &y2_scale, &spec, &plot_area);
        assert!(!nodes.is_empty(), "expected mark nodes for line secondary");
        let has_path = nodes.iter().any(|n| matches!(n, SceneNode::Path { .. }));
        assert!(has_path, "expected a path node for line mark");
    }

    #[test]
    fn build_secondary_marks_point_returns_circles() {
        let batch = make_batch();
        let x_scale = linear_scale(1.0, 5.0, 50.0, 350.0);
        let y2_scale = linear_scale(0.0, 400.0, 350.0, 50.0);
        let spec = SecondaryYSpec {
            field: "revenue".to_string(),
            mark: "point".to_string(),
            color: None,
            opacity: None,
        };
        let plot_area = Rect { x: 50.0, y: 50.0, w: 300.0, h: 300.0 };
        let nodes = build_secondary_marks(&batch, "x", &x_scale, &y2_scale, &spec, &plot_area);
        let circle_count = nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count();
        assert_eq!(circle_count, 5, "expected one circle per data point");
    }

    #[test]
    fn build_secondary_marks_missing_x_field_returns_empty() {
        let batch = make_batch();
        let x_scale = linear_scale(1.0, 5.0, 50.0, 350.0);
        let y2_scale = linear_scale(0.0, 400.0, 350.0, 50.0);
        let spec = SecondaryYSpec {
            field: "revenue".to_string(),
            mark: "line".to_string(),
            color: None,
            opacity: None,
        };
        let plot_area = Rect { x: 50.0, y: 50.0, w: 300.0, h: 300.0 };
        // Pass a nonexistent x field — should gracefully return empty.
        let nodes =
            build_secondary_marks(&batch, "nonexistent", &x_scale, &y2_scale, &spec, &plot_area);
        assert!(nodes.is_empty(), "expected empty nodes for missing x field");
    }
}
