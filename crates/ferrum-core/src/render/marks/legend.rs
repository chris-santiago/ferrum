//! Internal: draw legend swatches + labels from a LegendLayout.

use crate::layout::{LegendLayout, SymbolKind, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_fill_stroke, to_scene_stroke, to_scene_text_style};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{fmt_f, FillStroke, Stroke, SvgBuffer, TextStyle};
use ferrum_scene::SceneNode;

pub fn draw(
    legend: &LegendLayout,
    color_scale: Option<&ColorScale>,
    theme: &ThemeInputs,
    out: &mut SvgBuffer,
) {
    let label_style = TextStyle {
        fill: theme.font_color,
        font_size: theme.label_font_size,
        anchor: TextAnchor::Start,
        angle: 0.0,
        font_family: &theme.label_font_family,
        font_weight: if theme.font_weight == "normal" {
            None
        } else {
            Some(&theme.font_weight)
        },
        dominant_baseline: None,
    };

    // Legend title (Themes-T2.5b). Drawn before entries so it stays on top
    // visually if there's any overlap during layout shrink.
    if let Some(title) = &legend.title {
        let title_style = TextStyle {
            fill: theme.title_color,
            font_size: theme.legend_title_font_size,
            anchor: TextAnchor::Start,
            angle: 0.0,
            font_family: &theme.title_font_family,
            font_weight: if theme.title_font_weight == "normal" {
                None
            } else {
                Some(&theme.title_font_weight)
            },
            dominant_baseline: None,
        };
        out.text(title.x, title.y, &title.text, &title_style);
    }

    // Continuous colorbar (added 2026-05-11). Renders a vertical gradient
    // rect with per-tick ticks + labels on its right edge. Mutually
    // exclusive with categorical entries.
    if let Some(cb) = &legend.colorbar {
        // Deterministic gradient ID — one colorbar per chart, so `…-0` is
        // unique within the chart's own SVG. The grid compositor calls
        // `uniquify_clip_ids` to prefix `ferrum-colorbar-` (alongside
        // `ferrum-clip-`) with a per-cell key, so multi-cell composites
        // (clustermap, JointChart) stay collision-free.
        let grad_id = "ferrum-colorbar-0".to_string();
        let mut stops_xml = String::new();
        // Bottom = min (offset 0%, low position), top = max (offset 100%).
        // The y1=100%, y2=0% gradient maps offset 0 → bottom of bar.
        for (pos, color) in &cb.stops {
            stops_xml.push_str(&format!(
                "<stop offset=\"{:.4}\" stop-color=\"{}\"/>",
                pos, color,
            ));
        }
        out.raw(&format!(
            "<defs><linearGradient id=\"{grad_id}\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">{stops_xml}</linearGradient></defs>"
        ));
        out.raw(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"url(#{grad_id})\" stroke=\"{}\" stroke-width=\"0.5\"/>",
            fmt_f(cb.bar_rect.x),
            fmt_f(cb.bar_rect.y),
            fmt_f(cb.bar_rect.w),
            fmt_f(cb.bar_rect.h),
            crate::render::color::fmt_svg(theme.axis_line_color),
        ));

        // Tick marks + labels on the right edge of the bar.
        let tick_x_start = cb.bar_rect.x + cb.bar_rect.w;
        let tick_x_end = tick_x_start + 4.0;
        let label_x = tick_x_end + 4.0;
        for tick in &cb.ticks {
            out.line(
                tick_x_start, tick.y, tick_x_end, tick.y,
                &Stroke {
                    stroke: theme.axis_line_color,
                    stroke_width: theme.axis_line_width,
                    stroke_dash: None,
                },
            );
            out.text(label_x, tick.y + theme.label_font_size * 0.35,
                &tick.label, &label_style);
        }
    }

    for entry in &legend.entries {
        let color = color_scale
            .and_then(|s| match s {
                ColorScale::Categorical { .. } => s.lookup(&entry.label),
                // Continuous scales don't render legend swatches in 8a/9 — the
                // categorical legend builder only emits entries for categorical
                // scales, so this arm is unreachable in practice.
                ColorScale::Continuous { .. } => None,
            })
            .unwrap_or(theme.mark_color);
        let sx = entry.symbol_anchor_x;
        let sy = entry.symbol_anchor_y;
        match entry.symbol_kind {
            SymbolKind::Circle => out.circle(sx, sy, 4.0, &FillStroke {
                fill: Some(color), stroke: None, stroke_width: 0.0,
            }),
            SymbolKind::Square => out.rect(
                crate::layout::Rect { x: sx - 4.0, y: sy - 4.0, w: 8.0, h: 8.0 },
                &FillStroke { fill: Some(color), stroke: None, stroke_width: 0.0 },
                None,
            ),
            SymbolKind::Line => out.line(sx - 6.0, sy, sx + 6.0, sy, &Stroke {
                stroke: color, stroke_width: theme.line_stroke_width, stroke_dash: None,
            }),
        }
        out.text(entry.label_anchor_x, entry.label_anchor_y, &entry.label, &label_style);
    }
}

// ── Scene-graph path ────────────────────────────────────────────────

pub fn build_legend(
    legend: &LegendLayout,
    color_scale: Option<&ColorScale>,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let mut nodes = Vec::new();

    let label_fw: Option<&str> = if theme.font_weight == "normal" {
        None
    } else {
        Some(&theme.font_weight)
    };
    let label_text_style = to_scene_text_style(
        theme.font_color,
        theme.label_font_size,
        TextAnchor::Start,
        0.0,
        &theme.label_font_family,
        label_fw,
        None,
        1.0,
    );

    // Legend title.
    if let Some(title) = &legend.title {
        let title_fw: Option<&str> = if theme.title_font_weight == "normal" {
            None
        } else {
            Some(&theme.title_font_weight)
        };
        nodes.push(SceneNode::Text {
            x: title.x,
            y: title.y,
            content: title.text.clone(),
            style: to_scene_text_style(
                theme.title_color,
                theme.legend_title_font_size,
                TextAnchor::Start,
                0.0,
                &theme.title_font_family,
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
                crate::render::color::fmt_svg(theme.axis_line_color),
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
                    theme.axis_line_color,
                    theme.axis_line_width,
                    1.0,
                    None,
                    None,
                    None,
                ),
            });
            nodes.push(SceneNode::Text {
                x: label_x,
                y: tick.y + theme.label_font_size * 0.35,
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
            .unwrap_or(theme.mark_color);
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
                    style: to_scene_stroke(color, theme.line_stroke_width, 1.0, None, None, None),
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
    fn legend_emits_circle_swatch_per_entry() {
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
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&legend, None, &theme, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 2);
        assert_eq!(s.matches("<text ").count(), 2);
    }
}
