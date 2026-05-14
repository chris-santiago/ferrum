//! Internal: draw axis line, ticks, tick labels, and axis title from an AxisLayout.

use crate::layout::{AxisLayout, AxisOrient, Rect, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_stroke, to_scene_text_style};
use crate::render::svg::{Stroke, SvgBuffer, TextStyle};
use ferrum_scene::SceneNode;

pub fn draw(axis: &AxisLayout, theme: &ThemeInputs, out: &mut SvgBuffer) {
    let r = axis.axis_line;
    // D7: show_domain controls the axis domain line.
    if theme.axis_line && axis.show_domain {
        let line_style = Stroke {
            stroke: theme.axis_line_color,
            stroke_width: theme.axis_line_width,
            stroke_dash: None,
        };
        out.line(r.x, r.y, r.x + r.w, r.y + r.h, &line_style);
    }

    let tick_style = Stroke {
        stroke: theme.tick_color,
        stroke_width: theme.tick_width,
        stroke_dash: None,
    };
    let label_style_base = TextStyle {
        fill: theme.label_color,
        font_size: theme.label_font_size,
        anchor: TextAnchor::Middle,
        angle: 0.0,
        font_family: &theme.label_font_family,
        font_weight: if theme.font_weight == "normal" {
            None
        } else {
            Some(&theme.font_weight)
        },
        dominant_baseline: None,
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
        // D7: show_ticks controls tick marks.
        if axis.show_ticks {
            out.line(tx1, ty1, tx2, ty2, &tick_style);
        }
        // D7: show_labels controls tick label text.
        if axis.show_labels {
            let mut style = label_style_base.clone();
            style.anchor = anchor;
            style.angle = angle;
            out.text(label_x, label_y, &tick.label, &style);
        }
    }

    if let Some(t) = &axis.title {
        let title_style = TextStyle {
            fill: theme.title_color,
            font_size: theme.title_font_size,
            anchor: TextAnchor::Middle,
            angle: t.angle,
            font_family: &theme.title_font_family,
            font_weight: if theme.title_font_weight == "normal" {
                None
            } else {
                Some(&theme.title_font_weight)
            },
            dominant_baseline: None,
        };
        out.text(t.anchor_x, t.anchor_y, &t.text, &title_style);
    }
}

/// Draw the gridlines for a panel — vertical lines from x-axis tick positions
/// spanning the plot height, horizontal lines from y-axis tick positions
/// spanning the plot width. Called once per panel from the renderer's panel
/// loop *before* `axis::draw` so the axis line + ticks render on top.
///
/// Skips any gridline whose position coincides with an axis baseline (within
/// 0.5 px) to avoid a double-strokes at the plot edge. Returns early when
/// `theme.grid` is false.
pub fn draw_grid(
    plot: Rect,
    x_axis: Option<&AxisLayout>,
    y_axis: Option<&AxisLayout>,
    theme: &ThemeInputs,
    out: &mut SvgBuffer,
) {
    if !theme.grid {
        return;
    }
    let color = theme.grid_color;
    let width = theme.grid_width;
    let dash: Option<&[f64]> = theme.grid_dash.as_deref();
    let opacity = theme.grid_opacity;

    // y-axis baseline x-coord — vertical gridlines coinciding with it are skipped.
    let y_baseline_x = y_axis.map(|a| a.axis_line.x).unwrap_or(plot.x);
    // x-axis baseline y-coord — horizontal gridlines coinciding with it are skipped.
    let x_baseline_y = x_axis
        .map(|a| a.axis_line.y)
        .unwrap_or(plot.y + plot.h);

    // D7: per-axis show_grid gate (true by default → backward-compat).
    if let Some(ax) = x_axis.filter(|a| a.show_grid) {
        for tick in &ax.ticks {
            if (tick.position - y_baseline_x).abs() < 0.5 {
                continue;
            }
            out.gridline(
                tick.position,
                plot.y,
                tick.position,
                plot.y + plot.h,
                color,
                width,
                dash,
                opacity,
            );
        }
    }

    if let Some(ay) = y_axis.filter(|a| a.show_grid) {
        for tick in &ay.ticks {
            if (tick.position - x_baseline_y).abs() < 0.5 {
                continue;
            }
            out.gridline(
                plot.x,
                tick.position,
                plot.x + plot.w,
                tick.position,
                color,
                width,
                dash,
                opacity,
            );
        }
    }
}

// ── Scene-graph path ────────────────────────────────────────────────

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
    fn axis_draws_line_ticks_and_title() {
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
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&axis, &theme, &mut out);
        let s = out.finish();
        assert!(s.contains("<line "));
        assert!(s.matches("<line ").count() >= 3);
        assert!(s.contains(">x</text>") || s.contains(">x<"));
    }
}
