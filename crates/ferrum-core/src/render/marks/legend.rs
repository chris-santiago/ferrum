//! Internal: build legend scene nodes from a LegendLayout.

use crate::layout::{LegendLayout, SymbolKind, TextAnchor, ThemeInputs};
use crate::render::draw::{to_scene_fill_stroke, to_scene_stroke, to_scene_text_style};
use crate::render::scale_resolve::{ColorInput, ColorScale};
use crate::render::svg::fmt_f;
use ferrum_scene::{RawAnchor, SceneNode};

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
    // B5 unit 6a: `label_color` overrides the entry-label text fill. Absent → the
    // theme `font_color` default (byte-identical). An unparsable hex falls back to
    // the theme color rather than failing the render.
    let label_fill = legend
        .label_color
        .as_deref()
        .and_then(|s| crate::render::color::parse_color(s).ok())
        .unwrap_or(theme.colors.font_color);
    // Per-legend `label_font_size` override sizes both entry labels and colorbar
    // tick labels. The layout already reserved space at this size; drawing at the
    // raw theme size (the prior behavior) under-/over-flowed. Absent → the theme
    // default (byte-identical).
    let label_font_size = legend
        .label_font_size
        .unwrap_or(theme.typography.label_font_size);
    let label_text_style = to_scene_text_style(
        label_fill,
        label_font_size,
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
            slot: None,
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
        // Gradient defs + the gradient-filled rect that consumes them, emitted
        // as a SINGLE self-contained Raw fragment. Keeping the `id="{grad_id}"`
        // definition and its only `url(#{grad_id})` consumer co-located in one
        // fragment means the WASM overlay can namespace each Raw fragment
        // independently without dangling the cross-reference (R5). Chrome: the
        // colorbar is fixed UI chrome, not positioned in data space — it does
        // not track pan/zoom. (FillStroke cannot express url(#…) fills, so the
        // rect is raw too.) Byte order: defs then rect, identical to the prior
        // two consecutive Raw nodes, so static SVG output is unchanged.
        nodes.push(SceneNode::Raw {
            svg: format!(
                "<defs><linearGradient id=\"{grad_id}\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">{stops_xml}</linearGradient></defs>\
                 <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"url(#{grad_id})\" stroke=\"{}\" stroke-width=\"0.5\"/>",
                fmt_f(cb.bar_rect.x),
                fmt_f(cb.bar_rect.y),
                fmt_f(cb.bar_rect.w),
                fmt_f(cb.bar_rect.h),
                crate::render::color::fmt_svg(theme.colors.axis_line_color),
            ),
            anchor: RawAnchor::Chrome,
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
                y: tick.y + label_font_size * 0.35,
                content: tick.label.clone(),
                slot: None,
                style: label_text_style.clone(),
            });
        }
    }

    // B5 unit 3: `symbol_stroke_width` strokes each symbol swatch. The stroke
    // reuses the swatch's own fill color (a common legend convention), so a
    // single positive width thickens the outline without introducing a second
    // color knob. `None`/0 → no stroke (byte-identical default).
    let symbol_stroke_w = legend.symbol_stroke_width.filter(|w| *w > 0.0);

    // B5 unit 6a: `symbol_size` (area px²) scales the swatch geometry. Reuse the
    // mark `size` channel's area→radius mapping (`radius = sqrt(area / PI)`, see
    // `render::marks::point`) so a legend swatch and a same-area data point share
    // a radius. The default swatch geometry (circle r=4, square side=8, line
    // half-length=6, shape r=5) corresponds to r=4, so each dimension scales by
    // `r / 4` to keep the historical proportions. Absent / non-positive → the
    // fixed defaults (byte-identical). Negative/NaN areas are ignored.
    const DEFAULT_SWATCH_R: f64 = 4.0;
    let swatch_r = legend
        .symbol_size
        .filter(|a| *a > 0.0 && a.is_finite())
        .map(|area| (area / std::f64::consts::PI).sqrt())
        .unwrap_or(DEFAULT_SWATCH_R);
    let swatch_scale = swatch_r / DEFAULT_SWATCH_R;

    // Categorical entries (color), graduated size symbols, and shape glyphs.
    // Per-entry precedence for the swatch color: an explicit `color_hex`
    // (merged color+size legend) wins; else the categorical color-scale
    // lookup; else the theme mark color.
    for entry in &legend.entries {
        let color = entry
            .color_hex
            .as_deref()
            .and_then(|s| crate::render::color::parse_color(s).ok())
            .or_else(|| {
                color_scale.and_then(|s| match s.input() {
                    ColorInput::Category => s.lookup(&entry.label),
                    // Numeric scales render as a colorbar, never as entries.
                    ColorInput::Numeric => None,
                })
            })
            .unwrap_or(theme.colors.mark_color);
        let sx = entry.symbol_anchor_x;
        let sy = entry.symbol_anchor_y;

        if let Some(shape_name) = &entry.shape_name {
            // Shape legend: draw the category's point glyph. `symbol_size` scales
            // its fixed 5px radius (default scale = 1.0 → byte-identical).
            let kind = crate::render::marks::point::shape_from_str(shape_name);
            let style = crate::render::marks::point::ShapeStyle {
                fill: Some(color),
                fill_cleared: false,
                stroke: None,
                stroke_cleared: false,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                stroke_dash_idx: None,
                angle: 0.0,
            };
            nodes.extend(crate::render::marks::point::emit_shape_nodes(kind, sx, sy, 5.0 * swatch_scale, style));
        } else if let Some(radius) = entry.symbol_radius {
            // Graduated size legend: a filled circle at the scaled radius.
            nodes.push(SceneNode::Circle {
                cx: sx,
                cy: sy,
                r: radius,
                style: to_scene_fill_stroke(Some(color), false, None, false, 0.0, 1.0, None),
            });
        } else {
            // A positive `symbol_stroke_width` outlines the swatch in its own
            // color; otherwise no stroke (the historical fill-only swatch).
            let (swatch_stroke, swatch_stroke_w) = match symbol_stroke_w {
                Some(w) => (Some(color), w),
                None => (None, 0.0),
            };
            // `symbol_size` scales each swatch dimension by `swatch_scale`
            // (default 1.0 → byte-identical). Circle r=4, square side=8 (half
            // 4), line half-length=6 are the defaults at scale 1.0.
            match entry.symbol_kind {
                SymbolKind::Circle => {
                    nodes.push(SceneNode::Circle {
                        cx: sx,
                        cy: sy,
                        r: swatch_r,
                        style: to_scene_fill_stroke(Some(color), false, swatch_stroke, false, swatch_stroke_w, 1.0, None),
                    });
                }
                SymbolKind::Square => {
                    let half = 4.0 * swatch_scale;
                    nodes.push(SceneNode::Rect {
                        x: sx - half,
                        y: sy - half,
                        w: 2.0 * half,
                        h: 2.0 * half,
                        style: to_scene_fill_stroke(Some(color), false, swatch_stroke, false, swatch_stroke_w, 1.0, None),
                        corner_radius: 0.0,
                    });
                }
                SymbolKind::Line => {
                    // A line swatch has no fill; `symbol_stroke_width` overrides the
                    // theme line width when set so the legend line thickens too.
                    let line_w = symbol_stroke_w.unwrap_or(theme.sizes.line_stroke_width);
                    let half_len = 6.0 * swatch_scale;
                    nodes.push(SceneNode::Line {
                        x1: sx - half_len,
                        y1: sy,
                        x2: sx + half_len,
                        y2: sy,
                        style: to_scene_stroke(color, line_w, 1.0, None, None, None),
                    });
                }
            }
        }
        nodes.push(SceneNode::Text {
            x: entry.label_anchor_x,
            y: entry.label_anchor_y,
            content: entry.label.clone(),
            slot: None,
            style: label_text_style.clone(),
        });
    }

    // B5 unit 3: `clip_height` hard-clips the legend group to the requested
    // height. Wrap every legend node in a `<g clip-path>` whose `<clipPath>`
    // rect caps the legend rect at `clip_height`. The def and its consuming
    // group are emitted together (one Raw def + one Group) so overflowing
    // entries are clipped in the static SVG. This is static-SVG-only: the WASM
    // scene loader flattens `SceneNode::Group` and discards its `attrs`, so the
    // `clip-path` reference is dropped and the interactive render does not clip
    // overflowing legend entries (consistent with the W4/W5 interactive-export
    // limitations). Absent → nodes pass through unwrapped (byte-identical
    // default).
    if let Some(clip_h) = legend.clip_height.filter(|h| *h > 0.0) {
        return clip_legend_nodes(legend, clip_h, nodes);
    }

    nodes
}

/// Wrap `nodes` in an SVG `clipPath` capping the legend at `clip_h` pixels tall.
/// Emits a single `<clipPath>` def (Chrome-anchored `Raw`) followed by a
/// `<g clip-path="url(#…)">` group (`SceneNode::Group`) that contains the legend
/// content, so overflow past `clip_h` is hard-clipped (B5 unit 3 `clip_height`).
///
/// Static-SVG-only: the WASM scene loader flattens `SceneNode::Group` and drops
/// its `attrs`, so the `clip-path` reference is discarded in the interactive
/// render (the `<clipPath>` def still lands but nothing consumes it) and
/// overflowing legend entries are not clipped there. This matches the known
/// W4/W5 interactive-export limitations and is intentionally not fixed here.
fn clip_legend_nodes(legend: &LegendLayout, clip_h: f64, nodes: Vec<SceneNode>) -> Vec<SceneNode> {
    const CLIP_ID: &str = "ferrum-legend-clip-0";
    let r = &legend.rect;
    let def = SceneNode::Raw {
        svg: format!(
            "<defs><clipPath id=\"{CLIP_ID}\">\
             <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>\
             </clipPath></defs>",
            fmt_f(r.x),
            fmt_f(r.y),
            fmt_f(r.w),
            fmt_f(clip_h),
        ),
        anchor: RawAnchor::Chrome,
    };
    let group = SceneNode::Group {
        attrs: vec![("clip-path".to_string(), format!("url(#{CLIP_ID})"))],
        children: nodes,
    };
    vec![def, group]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{ColorbarLayout, LegendDirection, LegendEntryLayout, LegendOrient, Rect};
    use ferrum_scene::RawAnchor;

    /// Colorbar Raw nodes must carry `anchor == Chrome` (they are legend chrome,
    /// not positioned in data space).
    #[test]
    fn colorbar_raw_nodes_have_chrome_anchor() {
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 20.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![],
            title: None,
            colorbar: Some(ColorbarLayout {
                bar_rect: Rect { x: 85.0, y: 10.0, w: 10.0, h: 80.0 },
                stops: vec![(0.0, "#0000ff".to_string()), (1.0, "#ff0000".to_string())],
                ticks: vec![],
            }),
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_legend(&legend, None, &theme);
        let raw_nodes: Vec<_> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Raw { anchor, .. } = n { Some(*anchor) } else { None })
            .collect();
        assert!(!raw_nodes.is_empty(), "expected at least one Raw node for colorbar");
        for anchor in raw_nodes {
            assert_eq!(anchor, RawAnchor::Chrome, "all colorbar Raw nodes must be Chrome");
        }
    }

    /// The colorbar must emit its gradient `<defs>` and the consuming
    /// `<rect fill="url(#…)">` as ONE self-contained Raw fragment, with the
    /// defs preceding the rect (so static SVG bytes are unchanged from the
    /// prior two consecutive Raw nodes). This co-location is what lets the
    /// WASM overlay namespace each fragment independently without dangling
    /// the cross-reference (R5).
    #[test]
    fn colorbar_emits_single_self_contained_raw_fragment() {
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 20.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![],
            title: None,
            colorbar: Some(ColorbarLayout {
                bar_rect: Rect { x: 85.0, y: 10.0, w: 10.0, h: 80.0 },
                stops: vec![(0.0, "#0000ff".to_string()), (1.0, "#ff0000".to_string())],
                ticks: vec![],
            }),
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_legend(&legend, None, &theme);
        let raw_svgs: Vec<&str> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Raw { svg, .. } = n { Some(svg.as_str()) } else { None })
            .collect();
        assert_eq!(raw_svgs.len(), 1, "colorbar must emit exactly one Raw fragment");
        let frag = raw_svgs[0];
        // Both the gradient def and its only consumer live in this one fragment.
        assert!(frag.contains("<linearGradient id=\"ferrum-colorbar-0\""));
        assert!(frag.contains("fill=\"url(#ferrum-colorbar-0)\""));
        // defs precede the rect, with no whitespace between (byte-order preserved).
        let defs_end = frag.find("</linearGradient></defs>").expect("defs block present");
        let rect_start = frag.find("<rect").expect("rect present");
        assert!(defs_end < rect_start, "defs must precede rect");
        assert!(
            frag.contains("</defs><rect "),
            "rect must immediately follow defs (no intervening bytes): {frag}"
        );
    }

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
                    symbol_radius: None,
                    shape_name: None,
                    color_hex: None,
                },
                LegendEntryLayout {
                    label: "b".into(),
                    label_anchor_x: 88.0,
                    label_anchor_y: 24.0,
                    symbol_anchor_x: 84.0,
                    symbol_anchor_y: 24.0,
                    symbol_kind: SymbolKind::Circle,
                    symbol_radius: None,
                    shape_name: None,
                    color_hex: None,
                },
            ],
            title: None,
            colorbar: None,
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_legend(&legend, None, &theme);
        let circle_count = nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert_eq!(circle_count, 2);
        assert_eq!(text_count, 2);
    }

    /// A graduated size legend draws one circle per entry at the entry's
    /// `symbol_radius`, plus a numeric label per entry.
    #[test]
    fn size_legend_draws_graduated_circles_with_labels() {
        let entry = |label: &str, r: f64, y: f64| LegendEntryLayout {
            label: label.into(),
            label_anchor_x: 100.0,
            label_anchor_y: y,
            symbol_anchor_x: 90.0,
            symbol_anchor_y: y,
            symbol_kind: SymbolKind::Circle,
            symbol_radius: Some(r),
            shape_name: None,
            color_hex: None,
        };
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 40.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![entry("20", 3.0, 10.0), entry("60", 6.0, 30.0), entry("100", 10.0, 55.0)],
            title: None,
            colorbar: None,
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_legend(&legend, None, &theme);
        let radii: Vec<f64> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Circle { r, .. } = n { Some(*r) } else { None })
            .collect();
        assert_eq!(radii, vec![3.0, 6.0, 10.0], "graduated radii drawn in order");
        let labels: Vec<&str> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Text { content, .. } = n { Some(content.as_str()) } else { None })
            .collect();
        assert_eq!(labels, vec!["20", "60", "100"], "numeric size labels present");
    }

    /// A shape legend draws the category's point glyph (here a diamond → a
    /// `Path` node) plus a label, honoring an explicit `color_hex`.
    #[test]
    fn shape_legend_draws_glyph_per_category() {
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 40.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![LegendEntryLayout {
                label: "cat".into(),
                label_anchor_x: 100.0,
                label_anchor_y: 10.0,
                symbol_anchor_x: 90.0,
                symbol_anchor_y: 10.0,
                symbol_kind: SymbolKind::Circle,
                symbol_radius: None,
                shape_name: Some("diamond".into()),
                color_hex: None,
            }],
            title: None,
            colorbar: None,
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        };
        let theme = ThemeInputs::default();
        let nodes = build_legend(&legend, None, &theme);
        assert_eq!(
            nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count(),
            1,
            "diamond glyph emits one Path node"
        );
        assert_eq!(
            nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count(),
            1,
            "one shape label"
        );
    }

    // ── B5 unit 3: symbol_stroke_width + clip_height render ───────────────

    fn two_circle_legend(symbol_stroke_width: Option<f64>, clip_height: Option<f64>) -> LegendLayout {
        two_circle_legend_styled(symbol_stroke_width, clip_height, None, None, None)
    }

    fn two_circle_legend_styled(
        symbol_stroke_width: Option<f64>,
        clip_height: Option<f64>,
        symbol_size: Option<f64>,
        label_color: Option<String>,
        label_font_size: Option<f64>,
    ) -> LegendLayout {
        let entry = |label: &str, y: f64| LegendEntryLayout {
            label: label.into(),
            label_anchor_x: 100.0,
            label_anchor_y: y,
            symbol_anchor_x: 90.0,
            symbol_anchor_y: y,
            symbol_kind: SymbolKind::Circle,
            symbol_radius: None,
            shape_name: None,
            color_hex: Some("#1f77b4".into()),
        };
        LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 40.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![entry("a", 10.0), entry("b", 30.0)],
            title: None,
            colorbar: None,
            symbol_stroke_width,
            clip_height,
            symbol_size,
            label_color,
            label_font_size,
        }
    }

    /// `symbol_stroke_width=3` strokes each swatch circle with width 3.
    #[test]
    fn symbol_stroke_width_strokes_swatches() {
        let theme = ThemeInputs::default();
        let nodes = build_legend(&two_circle_legend(Some(3.0), None), None, &theme);
        let stroke_widths: Vec<f64> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Circle { style, .. } = n {
                Some(style.stroke_width)
            } else {
                None
            })
            .collect();
        assert_eq!(stroke_widths, vec![3.0, 3.0], "both swatches stroked at width 3");
        // The stroke is present (Some) on each swatch.
        for n in &nodes {
            if let SceneNode::Circle { style, .. } = n {
                assert!(style.stroke.is_some(), "stroked swatch must carry a stroke color");
            }
        }
    }

    /// With no `symbol_stroke_width`, swatches keep the historical fill-only
    /// shape (stroke_width 0, no stroke color).
    #[test]
    fn symbol_stroke_width_absent_no_stroke() {
        let theme = ThemeInputs::default();
        let nodes = build_legend(&two_circle_legend(None, None), None, &theme);
        for n in &nodes {
            if let SceneNode::Circle { style, .. } = n {
                assert_eq!(style.stroke_width, 0.0);
                assert!(style.stroke.is_none(), "default swatch has no stroke");
            }
        }
    }

    /// `clip_height` wraps the legend in a `<clipPath>` def + a clipping group,
    /// hard-clipping overflow.
    #[test]
    fn clip_height_wraps_legend_in_clip_group() {
        let theme = ThemeInputs::default();
        let nodes = build_legend(&two_circle_legend(None, Some(50.0)), None, &theme);
        // Exactly one Raw clipPath def + one clipping Group.
        let raw_defs: Vec<&str> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Raw { svg, .. } = n { Some(svg.as_str()) } else { None })
            .collect();
        assert_eq!(raw_defs.len(), 1, "one clipPath def emitted");
        let def = raw_defs[0];
        assert!(def.contains("<clipPath id=\"ferrum-legend-clip-0\""), "clipPath def present: {def}");
        // The clip rect height equals clip_height (50), not the rect's full 100.
        assert!(def.contains("height=\"50\""), "clip rect height = clip_height: {def}");

        let groups: Vec<&Vec<SceneNode>> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Group { attrs, children } = n {
                assert!(attrs.iter().any(|(k, v)| k == "clip-path"
                    && v == "url(#ferrum-legend-clip-0)"));
                Some(children)
            } else {
                None
            })
            .collect();
        assert_eq!(groups.len(), 1, "one clipping group wraps the content");
        // The clipped group holds the actual swatches/labels.
        assert!(
            groups[0].iter().any(|n| matches!(n, SceneNode::Circle { .. })),
            "clipped group contains the swatch circles",
        );
    }

    /// Without `clip_height`, the legend nodes are emitted unwrapped (no
    /// clipPath, no clipping group) — byte-identical to the historical shape.
    #[test]
    fn clip_height_absent_no_clip_group() {
        let theme = ThemeInputs::default();
        let nodes = build_legend(&two_circle_legend(None, None), None, &theme);
        assert!(
            !nodes.iter().any(|n| matches!(n, SceneNode::Group { .. })),
            "no clip group when clip_height absent",
        );
        assert!(
            !nodes.iter().any(|n| matches!(n, SceneNode::Raw { .. })),
            "no clipPath def when clip_height absent",
        );
    }

    // ── B5 unit 6a: symbol_size + label_color render ──────────────────────

    fn swatch_radii(nodes: &[SceneNode]) -> Vec<f64> {
        nodes
            .iter()
            .filter_map(|n| if let SceneNode::Circle { r, .. } = n { Some(*r) } else { None })
            .collect()
    }

    /// `symbol_size` enlarges the circle swatch via `radius = sqrt(area / PI)`,
    /// the same mapping the mark `size` channel uses. Default (None) → r=4.
    #[test]
    fn symbol_size_scales_circle_swatch_radius() {
        let theme = ThemeInputs::default();
        let base = build_legend(&two_circle_legend_styled(None, None, None, None, None), None, &theme);
        assert_eq!(swatch_radii(&base), vec![4.0, 4.0], "default swatch r=4");

        // area = 400 → r = sqrt(400 / PI) ≈ 11.28, larger than the default 4.
        let big = build_legend(
            &two_circle_legend_styled(None, None, Some(400.0), None, None), None, &theme,
        );
        let expected_r = (400.0_f64 / std::f64::consts::PI).sqrt();
        for r in swatch_radii(&big) {
            assert!((r - expected_r).abs() < 1e-9, "swatch r = sqrt(area/PI): got {r}");
            assert!(r > 4.0, "symbol_size=400 must enlarge the swatch beyond the default");
        }
    }

    /// `symbol_size` scales the square swatch side proportionally (default side 8
    /// at r=4, so side = 2 * r).
    #[test]
    fn symbol_size_scales_square_swatch_side() {
        let theme = ThemeInputs::default();
        let mut legend = two_circle_legend_styled(None, None, Some(400.0), None, None);
        for e in &mut legend.entries {
            e.symbol_kind = SymbolKind::Square;
        }
        let nodes = build_legend(&legend, None, &theme);
        let expected_side = 2.0 * (400.0_f64 / std::f64::consts::PI).sqrt();
        for n in &nodes {
            if let SceneNode::Rect { w, h, .. } = n {
                assert!((w - expected_side).abs() < 1e-9, "square side = 2r: got {w}");
                assert!((h - expected_side).abs() < 1e-9);
                assert!(*w > 8.0, "symbol_size=400 must enlarge the square beyond side 8");
            }
        }
    }

    /// Absent `symbol_size` keeps the historical fixed geometry (byte-identical).
    #[test]
    fn symbol_size_absent_keeps_default_geometry() {
        let theme = ThemeInputs::default();
        let nodes = build_legend(&two_circle_legend_styled(None, None, None, None, None), None, &theme);
        assert_eq!(swatch_radii(&nodes), vec![4.0, 4.0], "default r=4 unchanged");
    }

    /// `label_color` overrides the entry-label text fill with the given hex.
    #[test]
    fn label_color_overrides_entry_label_fill() {
        use crate::render::draw::to_scene_color;
        let theme = ThemeInputs::default();
        let red = to_scene_color(crate::render::color::parse_color("#ff0000").unwrap());
        let nodes = build_legend(
            &two_circle_legend_styled(None, None, None, Some("#ff0000".into()), None), None, &theme,
        );
        let label_colors: Vec<_> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Text { style, .. } = n { Some(style.color) } else { None })
            .collect();
        assert!(!label_colors.is_empty(), "legend must emit label text nodes");
        for color in label_colors {
            assert_eq!(color, red, "entry label fill must be the label_color override");
        }
    }

    /// Batch A Task 8 sweep: `label_color` was hex-only, so a CSS name or an
    /// `rgb()` string silently fell back to the theme font color. All three
    /// spellings of one color now produce the same label fill.
    #[test]
    fn label_color_accepts_named_and_rgb_forms_identically_to_hex() {
        use crate::render::draw::to_scene_color;
        let theme = ThemeInputs::default();
        let label_fills = |spelling: &str| -> Vec<_> {
            build_legend(
                &two_circle_legend_styled(None, None, None, Some(spelling.into()), None),
                None,
                &theme,
            )
            .iter()
            .filter_map(|n| if let SceneNode::Text { style, .. } = n { Some(style.color) } else { None })
            .collect()
        };
        let expected = to_scene_color(crate::render::color::parse_color("#4682b4").unwrap());
        for spelling in ["steelblue", "rgb(70, 130, 180)", "#4682b4"] {
            let fills = label_fills(spelling);
            assert!(!fills.is_empty(), "legend must emit label text nodes");
            for fill in fills {
                assert_eq!(fill, expected, "{spelling:?} must resolve to the same label fill");
            }
        }
        // Unparseable input keeps the theme font color, as before the sweep.
        for fill in label_fills("not-a-color") {
            assert_eq!(fill, to_scene_color(theme.colors.font_color));
        }
    }

    /// Absent `label_color` uses the theme `font_color` (byte-identical default).
    #[test]
    fn label_color_absent_uses_theme_font_color() {
        use crate::render::draw::to_scene_color;
        let theme = ThemeInputs::default();
        let expected = to_scene_color(theme.colors.font_color);
        let nodes = build_legend(&two_circle_legend_styled(None, None, None, None, None), None, &theme);
        for n in &nodes {
            if let SceneNode::Text { style, .. } = n {
                assert_eq!(style.color, expected);
            }
        }
    }

    /// Per-legend `label_font_size` override sizes the rendered entry-label text
    /// (the layout already reserves space at this size). Absent → theme default.
    #[test]
    fn label_font_size_override_sizes_entry_label_text() {
        let theme = ThemeInputs::default();
        // Override to 18; entry-label text nodes must be drawn at 18, not theme.
        let nodes = build_legend(
            &two_circle_legend_styled(None, None, None, None, Some(18.0)), None, &theme,
        );
        let sizes: Vec<f64> = nodes
            .iter()
            .filter_map(|n| if let SceneNode::Text { style, .. } = n { Some(style.font_size) } else { None })
            .collect();
        assert!(!sizes.is_empty(), "legend must emit label text nodes");
        for s in sizes {
            assert_eq!(s, 18.0, "entry label must render at the label_font_size override");
        }
        // Absent → theme label_font_size (byte-identical default).
        let nodes_default =
            build_legend(&two_circle_legend_styled(None, None, None, None, None), None, &theme);
        for n in &nodes_default {
            if let SceneNode::Text { style, .. } = n {
                assert_eq!(style.font_size, theme.typography.label_font_size);
            }
        }
    }
}
