//! Internal: build legend scene nodes from a LegendLayout.

use crate::layout::{LegendLayout, SymbolKind, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_fill_stroke, to_scene_stroke, to_scene_text_style};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::fmt_f;
use ferrum_scene::SceneNode;

pub fn build_legend(
    legend: &LegendLayout,
    color_scale: Option<&ColorScale>,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let mut nodes = Vec::new();

    let label_fw: Option<&str> = if theme.typography.font_weight == "normal" {
        None
    } else {
        Some(&theme.typography.font_weight)
    };
    let label_text_style = to_scene_text_style(
        theme.colors.font_color,
        theme.typography.label_font_size,
        TextAnchor::Start,
        0.0,
        &theme.typography.label_font_family,
        label_fw,
        None,
        1.0,
    );

    // Legend title.
    if let Some(title) = &legend.title {
        let title_fw: Option<&str> = if theme.typography.title_font_weight == "normal" {
            None
        } else {
            Some(&theme.typography.title_font_weight)
        };
        nodes.push(SceneNode::Text {
            x: title.x,
            y: title.y,
            content: title.text.clone(),
            style: to_scene_text_style(
                theme.colors.title_color,
                theme.typography.legend_title_font_size,
                TextAnchor::Start,
                0.0,
                &theme.typography.title_font_family,
                title_fw,
                None,
                1.0,
            ),
        });
    }

    // Continuous colorbar.
    if let Some(cb) = &legend.colorbar {
        let grad_id = "ferrum-colorbar-0".to_string();
        let mut stops_xml = String::new();
        for (pos, color) in &cb.stops {
            stops_xml.push_str(&format!(
                "<stop offset=\"{:.4}\" stop-color=\"{}\"/>",
                pos, color,
            ));
        }
        // Gradient defs as raw SVG.
        nodes.push(SceneNode::Raw {
            svg: format!(
                "<defs><linearGradient id=\"{grad_id}\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">{stops_xml}</linearGradient></defs>"
            ),
        });
        // Gradient-filled rect as raw SVG (FillStroke cannot express url(#…) fills).
        nodes.push(SceneNode::Raw {
            svg: format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"url(#{grad_id})\" stroke=\"{}\" stroke-width=\"0.5\"/>",
                fmt_f(cb.bar_rect.x),
                fmt_f(cb.bar_rect.y),
                fmt_f(cb.bar_rect.w),
                fmt_f(cb.bar_rect.h),
                crate::render::color::fmt_svg(theme.colors.axis_line_color),
            ),
        });

        // Tick marks + labels on the right edge.
        let tick_x_start = cb.bar_rect.x + cb.bar_rect.w;
        let tick_x_end = tick_x_start + 4.0;
        let label_x = tick_x_end + 4.0;
        for tick in &cb.ticks {
            nodes.push(SceneNode::Line {
                x1: tick_x_start,
                y1: tick.y,
                x2: tick_x_end,
                y2: tick.y,
                style: to_scene_stroke(
                    theme.colors.axis_line_color,
                    theme.sizes.axis_line_width,
                    1.0,
                    None,
                    None,
                    None,
                ),
            });
            nodes.push(SceneNode::Text {
                x: label_x,
                y: tick.y + theme.typography.label_font_size * 0.35,
                content: tick.label.clone(),
                style: label_text_style.clone(),
            });
        }
    }

    // Categorical entries.
    for entry in &legend.entries {
        let color = color_scale
            .and_then(|s| match s {
                ColorScale::Categorical { .. } => s.lookup(&entry.label),
                ColorScale::Continuous { .. } => None,
            })
            .unwrap_or(theme.colors.mark_color);
        let sx = entry.symbol_anchor_x;
        let sy = entry.symbol_anchor_y;
        match entry.symbol_kind {
            SymbolKind::Circle => {
                nodes.push(SceneNode::Circle {
                    cx: sx,
                    cy: sy,
                    r: 4.0,
                    style: to_scene_fill_stroke(Some(color), None, 0.0, 1.0, None),
                });
            }
            SymbolKind::Square => {
                nodes.push(SceneNode::Rect {
                    x: sx - 4.0,
                    y: sy - 4.0,
                    w: 8.0,
                    h: 8.0,
                    style: to_scene_fill_stroke(Some(color), None, 0.0, 1.0, None),
                    corner_radius: 0.0,
                });
            }
            SymbolKind::Line => {
                nodes.push(SceneNode::Line {
                    x1: sx - 6.0,
                    y1: sy,
                    x2: sx + 6.0,
                    y2: sy,
                    style: to_scene_stroke(color, theme.sizes.line_stroke_width, 1.0, None, None, None),
                });
            }
        }
        nodes.push(SceneNode::Text {
            x: entry.label_anchor_x,
            y: entry.label_anchor_y,
            content: entry.label.clone(),
            style: label_text_style.clone(),
        });
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LegendDirection, LegendEntryLayout, LegendOrient, Rect};

    #[test]
    fn legend_builds_circle_swatch_per_entry() {
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 20.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![
                LegendEntryLayout {
                    label: "a".into(),
                    label_anchor_x: 88.0,
                    label_anchor_y: 10.0,
                    symbol_anchor_x: 84.0,
                    symbol_anchor_y: 10.0,
                    symbol_kind: SymbolKind::Circle,
                },
                LegendEntryLayout {
                    label: "b".into(),
                    label_anchor_x: 88.0,
                    label_anchor_y: 24.0,
                    symbol_anchor_x: 84.0,
                    symbol_anchor_y: 24.0,
                    symbol_kind: SymbolKind::Circle,
                },
            ],
            title: None,
            colorbar: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_legend(&legend, None, &theme);
        let circle_count = nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert_eq!(circle_count, 2);
        assert_eq!(text_count, 2);
    }
}
