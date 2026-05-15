//! Internal: build axis and grid scene nodes from an AxisLayout.

use crate::layout::{AxisLayout, AxisOrient, Rect, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_stroke, to_scene_text_style};
use ferrum_scene::SceneNode;

pub fn build_axis(axis: &AxisLayout, theme: &ThemeInputs) -> Vec<SceneNode> {
    let mut nodes = Vec::new();
    let r = axis.axis_line;

    // Domain line.
    if theme.axis_line && axis.show_domain {
        nodes.push(SceneNode::Line {
            x1: r.x,
            y1: r.y,
            x2: r.x + r.w,
            y2: r.y + r.h,
            style: to_scene_stroke(theme.axis_line_color, theme.axis_line_width, 1.0, None, None, None),
        });
    }

    let tick_stroke = to_scene_stroke(theme.tick_color, theme.tick_width, 1.0, None, None, None);

    let label_fw: Option<&str> = if theme.font_weight == "normal" {
        None
    } else {
        Some(&theme.font_weight)
    };

    for tick in &axis.ticks {
        let (tx1, ty1, tx2, ty2, label_x, label_y, anchor, angle) = match axis.orient {
            AxisOrient::Bottom => (
                tick.position, r.y, tick.position, r.y + theme.tick_size,
                tick.position, r.y + theme.tick_size + theme.label_font_size + 2.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Top => (
                tick.position, r.y, tick.position, r.y - theme.tick_size,
                tick.position, r.y - theme.tick_size - 4.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Left => (
                r.x, tick.position, r.x - theme.tick_size, tick.position,
                r.x - theme.tick_size - 2.0, tick.position + theme.label_font_size / 3.0,
                TextAnchor::End, 0.0,
            ),
            AxisOrient::Right => (
                r.x, tick.position, r.x + theme.tick_size, tick.position,
                r.x + theme.tick_size + 2.0, tick.position + theme.label_font_size / 3.0,
                TextAnchor::Start, 0.0,
            ),
        };
        if axis.show_ticks {
            nodes.push(SceneNode::Line {
                x1: tx1,
                y1: ty1,
                x2: tx2,
                y2: ty2,
                style: tick_stroke.clone(),
            });
        }
        if axis.show_labels {
            nodes.push(SceneNode::Text {
                x: label_x,
                y: label_y,
                content: tick.label.clone(),
                style: to_scene_text_style(
                    theme.label_color,
                    theme.label_font_size,
                    anchor,
                    angle,
                    &theme.label_font_family,
                    label_fw,
                    None,
                    1.0,
                ),
            });
        }
    }

    if let Some(t) = &axis.title {
        let title_fw: Option<&str> = if theme.title_font_weight == "normal" {
            None
        } else {
            Some(&theme.title_font_weight)
        };
        nodes.push(SceneNode::Text {
            x: t.anchor_x,
            y: t.anchor_y,
            content: t.text.clone(),
            style: to_scene_text_style(
                theme.title_color,
                theme.title_font_size,
                TextAnchor::Middle,
                t.angle,
                &theme.title_font_family,
                title_fw,
                None,
                1.0,
            ),
        });
    }

    nodes
}

pub fn build_grid(
    plot_area: Rect,
    x_axis: Option<&AxisLayout>,
    y_axis: Option<&AxisLayout>,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    if !theme.grid {
        return Vec::new();
    }
    let mut nodes = Vec::new();
    let color = theme.grid_color;
    let width = theme.grid_width;
    let dash: Option<&[f64]> = theme.grid_dash.as_deref();
    let opacity = theme.grid_opacity;

    let y_baseline_x = y_axis.map(|a| a.axis_line.x).unwrap_or(plot_area.x);
    let x_baseline_y = x_axis
        .map(|a| a.axis_line.y)
        .unwrap_or(plot_area.y + plot_area.h);

    if let Some(ax) = x_axis.filter(|a| a.show_grid) {
        for tick in &ax.ticks {
            if (tick.position - y_baseline_x).abs() < 0.5 {
                continue;
            }
            nodes.push(SceneNode::Line {
                x1: tick.position,
                y1: plot_area.y,
                x2: tick.position,
                y2: plot_area.y + plot_area.h,
                style: to_scene_stroke(color, width, opacity, dash, None, None),
            });
        }
    }

    if let Some(ay) = y_axis.filter(|a| a.show_grid) {
        for tick in &ay.ticks {
            if (tick.position - x_baseline_y).abs() < 0.5 {
                continue;
            }
            nodes.push(SceneNode::Line {
                x1: plot_area.x,
                y1: tick.position,
                x2: plot_area.x + plot_area.w,
                y2: tick.position,
                style: to_scene_stroke(color, width, opacity, dash, None, None),
            });
        }
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AxisLayout, AxisOrient, AxisTitleLayout, Rect, TickLayout};

    #[test]
    fn axis_builds_line_ticks_and_title() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout { position: 25.0, label: "0".into(), label_angle: 0.0, elided: false },
                TickLayout { position: 75.0, label: "1".into(), label_angle: 0.0, elided: false },
            ],
            title: Some(AxisTitleLayout {
                text: "x".into(),
                anchor_x: 50.0,
                anchor_y: 95.0,
                angle: 0.0,
            }),
            show_labels: true,
            show_ticks: true,
            show_domain: true,
            show_grid: true,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);
        // 1 domain line + 2 tick lines + 2 tick labels + 1 title = 6 nodes
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert!(line_count >= 3, "expected >=3 lines (domain + ticks), got {line_count}");
        assert!(text_count >= 3, "expected >=3 texts (2 labels + title), got {text_count}");
    }
}
