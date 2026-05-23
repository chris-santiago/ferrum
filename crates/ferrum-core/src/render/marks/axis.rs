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
        let effective_font_size = tick.label_font_size.unwrap_or(theme.label_font_size);

        let (tx1, ty1, tx2, ty2, label_x, label_y, anchor, angle) = match axis.orient {
            AxisOrient::Bottom => (
                tick.position, r.y, tick.position, r.y + theme.tick_size,
                tick.position, r.y + theme.tick_size + effective_font_size + 2.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Top => (
                tick.position, r.y, tick.position, r.y - theme.tick_size,
                tick.position, r.y - theme.tick_size - 4.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Left => (
                r.x, tick.position, r.x - theme.tick_size, tick.position,
                r.x - theme.tick_size - 2.0, tick.position + effective_font_size / 3.0,
                TextAnchor::End, 0.0,
            ),
            AxisOrient::Right => (
                r.x, tick.position, r.x + theme.tick_size, tick.position,
                r.x + theme.tick_size + 2.0, tick.position + effective_font_size / 3.0,
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
        if axis.show_labels && !tick.culled {
            let lines: Vec<&str> = tick.label.split('\n').collect();
            if lines.len() == 1 || axis.orient != AxisOrient::Bottom {
                // Single-line label, or non-Bottom orient (no multi-line for those).
                nodes.push(SceneNode::Text {
                    x: label_x,
                    y: label_y,
                    content: tick.label.clone(),
                    style: to_scene_text_style(
                        theme.label_color,
                        effective_font_size,
                        anchor,
                        angle,
                        &theme.label_font_family,
                        label_fw,
                        None,
                        1.0,
                    ),
                });
            } else {
                // Multi-line label (Bottom orient only): emit one text node per line.
                let line_height = effective_font_size * 1.2;
                for (line_idx, line) in lines.iter().enumerate() {
                    let line_y = label_y + (line_idx as f64) * line_height;
                    nodes.push(SceneNode::Text {
                        x: label_x,
                        y: line_y,
                        content: line.to_string(),
                        style: to_scene_text_style(
                            theme.label_color,
                            effective_font_size,
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
                TickLayout { position: 25.0, label: "0".into(), label_angle: 0.0, elided: false, culled: false, label_font_size: None },
                TickLayout { position: 75.0, label: "1".into(), label_angle: 0.0, elided: false, culled: false, label_font_size: None },
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

    #[test]
    fn culled_tick_skips_label() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout { position: 25.0, label: "visible".into(), label_angle: 0.0, elided: false, culled: false, label_font_size: None },
                TickLayout { position: 75.0, label: "culled".into(), label_angle: 0.0, elided: false, culled: true, label_font_size: None },
            ],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);

        // Both ticks should emit a tick mark line.
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 2, "expected 2 tick lines, got {line_count}");

        // Only the non-culled tick emits a text label.
        let texts: Vec<_> = nodes.iter().filter_map(|n| {
            if let SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert_eq!(texts, vec!["visible"], "culled tick must not emit a label; got {texts:?}");
    }

    #[test]
    fn multiline_label_emits_stacked_text() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout {
                    position: 50.0,
                    label: "trivial\nbaseline".into(),
                    label_angle: 0.0,
                    elided: false,
                    culled: false,
                    label_font_size: None,
                },
            ],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
        };
        let theme = ThemeInputs::default();
        let nodes = build_axis(&axis, &theme);

        // One tick line + two text nodes (one per line).
        let text_contents: Vec<_> = nodes.iter().filter_map(|n| {
            if let SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None }
        }).collect();
        assert_eq!(
            text_contents, vec!["trivial", "baseline"],
            "multi-line label should emit one text node per line; got {text_contents:?}",
        );

        // The second line should be offset downward from the first.
        let text_ys: Vec<f64> = nodes.iter().filter_map(|n| {
            if let SceneNode::Text { y, .. } = n { Some(*y) } else { None }
        }).collect();
        assert_eq!(text_ys.len(), 2);
        assert!(
            text_ys[1] > text_ys[0],
            "second line y ({}) should be below first line y ({})",
            text_ys[1], text_ys[0],
        );
    }

    #[test]
    fn per_tick_font_size_override() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout {
                    position: 50.0,
                    label: "small".into(),
                    label_angle: 0.0,
                    elided: false,
                    culled: false,
                    label_font_size: Some(9.0),
                },
            ],
            title: None,
            show_labels: true,
            show_ticks: true,
            show_domain: false,
            show_grid: false,
        };
        let theme = ThemeInputs::default(); // theme.label_font_size == 11.0
        let nodes = build_axis(&axis, &theme);

        // The text node should use font_size 9.0, not the theme default.
        let text_node = nodes.iter().find(|n| matches!(n, SceneNode::Text { .. }));
        assert!(text_node.is_some(), "expected a text node");
        if let Some(SceneNode::Text { style, y, .. }) = text_node {
            assert_eq!(
                style.font_size, 9.0,
                "expected font_size 9.0 from per-tick override, got {}",
                style.font_size,
            );
            // label_y = r.y + tick_size + effective_font_size + 2.0
            // With r.y=80, tick_size=4 (default), effective_font_size=9: 80 + 4 + 9 + 2 = 95
            let expected_y = 80.0 + theme.tick_size + 9.0 + 2.0;
            assert!(
                (y - expected_y).abs() < 0.01,
                "label_y should use per-tick font size for positioning: expected {expected_y}, got {y}",
            );
        }
    }
}
