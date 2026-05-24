//! Inset chart embedding — positions a pre-rendered SVG at normalized bounds
//! within the plot area, with optional border, background, shadow, and connector.
//!
//! Insets use SVG's native nested `<svg>` element support to embed the
//! pre-rendered content inside a positioned, sized viewport.

use ferrum_scene::{FillStroke, SceneNode, StrokeStyle};

use crate::layout::Rect;
use crate::render::chart_config::InsetSpec;
use crate::render::color::parse_color;
use crate::render::draw::to_scene_color;

/// Build scene nodes that embed the inset SVG at `spec.bounds` within `plot_area`.
///
/// Rendering order (back to front):
/// 1. Drop shadow (if `spec.shadow`)
/// 2. Background fill rect (if `spec.background`)
/// 3. The SVG content as a `SceneNode::Raw`
/// 4. Border rect (if `spec.border`)
/// 5. Connector lines from `spec.connect_to` to inset corners (if set)
pub fn build_inset_nodes(spec: &InsetSpec, plot_area: &Rect) -> Vec<SceneNode> {
    let [b_left, b_top, b_right, b_bottom] = spec.bounds;

    // Convert normalized [0,1] bounds to pixel coordinates relative to plot_area.
    let px_left = plot_area.x + b_left * plot_area.w;
    let px_top = plot_area.y + b_top * plot_area.h;
    let px_right = plot_area.x + b_right * plot_area.w;
    let px_bottom = plot_area.y + b_bottom * plot_area.h;

    let px_w = (px_right - px_left).max(1.0);
    let px_h = (px_bottom - px_top).max(1.0);

    let mut nodes: Vec<SceneNode> = Vec::new();

    // 1. Drop shadow: an offset, slightly blurred gray rectangle behind the inset.
    if spec.shadow {
        let shadow_offset = 3.0;
        let shadow_color = ferrum_scene::Color::rgba(0, 0, 0, 40);
        nodes.push(SceneNode::Rect {
            x: px_left + shadow_offset,
            y: px_top + shadow_offset,
            w: px_w,
            h: px_h,
            style: FillStroke {
                fill: Some(shadow_color),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
            corner_radius: 0.0,
        });
    }

    // 2. Background fill.
    if let Some(ref bg_hex) = spec.background {
        if let Ok(bg_color) = parse_color(bg_hex) {
            let bg_scene = to_scene_color(bg_color);
            nodes.push(SceneNode::Rect {
                x: px_left,
                y: px_top,
                w: px_w,
                h: px_h,
                style: FillStroke {
                    fill: Some(bg_scene),
                    stroke: None,
                    stroke_width: 0.0,
                    opacity: 1.0,
                    stroke_dash: None,
                    stroke_opacity: 1.0,
                    fill_opacity: 1.0,
                    angle: 0.0,
                },
                corner_radius: 0.0,
            });
        }
    }

    // 3. The SVG content embedded as a nested <svg> element.
    //
    // SVG supports nested <svg> natively. We strip the outer <svg ...> wrapper
    // from the pre-rendered string and wrap it in a positioned <svg> element
    // with the correct x/y/width/height so it clips to the inset bounds.
    let inner_content = strip_svg_wrapper(&spec.svg);
    let svg_raw = format!(
        "<svg x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\" overflow=\"hidden\">{}</svg>",
        px_left, px_top, px_w, px_h, inner_content
    );
    nodes.push(SceneNode::Raw { svg: svg_raw });

    // 4. Border rect drawn on top of the content.
    if spec.border {
        let border_color = parse_color(&spec.border_color)
            .unwrap_or_else(|_| crate::render::color::from_rgb(0x99, 0x99, 0x99));
        let border_scene = to_scene_color(border_color);
        let border_dash = spec.border_dash.clone();
        nodes.push(SceneNode::Rect {
            x: px_left,
            y: px_top,
            w: px_w,
            h: px_h,
            style: FillStroke {
                fill: None,
                stroke: Some(border_scene),
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_dash: border_dash,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
            corner_radius: 0.0,
        });
    }

    // 5. Connector lines from a data-space point to the nearest inset corner.
    //    `spec.connect_to` holds a pixel-space point [x, y] that has already
    //    been resolved by the caller (scene_build passes the data-space coords
    //    through the primary scales before calling here — or simply uses the
    //    raw values if they're already in pixel space).
    if let Some([cx, cy]) = spec.connect_to {
        if spec.connect_style == "lines" {
            let connector_stroke = build_connector_stroke();
            // Find the two nearest corners of the inset to the connect_to point.
            let corners = [
                (px_left, px_top),
                (px_right, px_top),
                (px_left, px_bottom),
                (px_right, px_bottom),
            ];
            let mut sorted_corners = corners;
            sorted_corners.sort_by(|a, b| {
                let da = (a.0 - cx).hypot(a.1 - cy);
                let db = (b.0 - cx).hypot(b.1 - cy);
                // hypot of finite coords is always finite; Equal is the safe fallback.
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Draw lines to the two closest corners.
            for (ix, iy) in sorted_corners.iter().take(2) {
                nodes.push(SceneNode::Line {
                    x1: cx,
                    y1: cy,
                    x2: *ix,
                    y2: *iy,
                    style: connector_stroke.clone(),
                });
            }
        }
        // Other connector styles: nothing rendered for unknown styles.
    }

    nodes
}

/// Strip the outer `<svg ...>...</svg>` wrapper from a pre-rendered SVG string.
///
/// Returns the inner content (everything between the first `>` and the last
/// `</svg>`). Returns an empty string for SVGs with no inner content (e.g.
/// `<svg></svg>`). If the string does not start with `<svg`, returns the full
/// string unchanged so non-SVG content is passed through.
fn strip_svg_wrapper(svg: &str) -> &str {
    let s = svg.trim();
    if !s.starts_with("<svg") {
        return s;
    }
    // Find end of the opening <svg ...> tag.
    let open_end = match s.find('>') {
        Some(i) => i + 1,
        None => return s,
    };
    // Find the last </svg>.
    let close_start = s.rfind("</svg>").unwrap_or(s.len());
    if open_end <= close_start {
        &s[open_end..close_start]
    } else {
        ""
    }
}

fn build_connector_stroke() -> ferrum_scene::StrokeStyle {
    StrokeStyle {
        color: ferrum_scene::Color::rgba(150, 150, 150, 200),
        width: 0.75,
        opacity: 0.8,
        dash: Some(vec![3.0, 3.0]),
        stroke_opacity: 0.8,
        stroke_cap: None,
        stroke_join: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plot() -> Rect {
        Rect { x: 50.0, y: 50.0, w: 300.0, h: 200.0 }
    }

    fn basic_spec() -> InsetSpec {
        InsetSpec {
            svg: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"150\"><circle cx=\"100\" cy=\"75\" r=\"50\"/></svg>".to_string(),
            bounds: [0.6, 0.1, 0.95, 0.55],
            border: true,
            border_color: "#999999".to_string(),
            border_dash: None,
            background: None,
            shadow: false,
            connect_to: None,
            connect_style: "lines".to_string(),
        }
    }

    // ── strip_svg_wrapper ────────────────────────────────────────────────────

    #[test]
    fn strip_svg_wrapper_extracts_inner() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="10"/></svg>"#;
        let inner = strip_svg_wrapper(svg);
        assert!(inner.contains("<circle"), "inner should contain circle element");
        assert!(!inner.contains("<svg "), "inner should not contain opening svg tag");
        assert!(!inner.contains("</svg>"), "inner should not contain closing svg tag");
    }

    #[test]
    fn strip_svg_wrapper_empty_svg() {
        let svg = "<svg></svg>";
        let inner = strip_svg_wrapper(svg);
        // An empty SVG wrapper should produce empty inner content.
        assert!(inner.is_empty(), "empty svg should produce empty inner; got: {inner:?}");
    }

    // ── build_inset_nodes: border ────────────────────────────────────────────

    #[test]
    fn inset_with_border_includes_rect_node() {
        let spec = basic_spec();
        let nodes = build_inset_nodes(&spec, &plot());
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert!(rect_count >= 1, "expected at least one rect node for border");
    }

    #[test]
    fn inset_without_border_has_no_border_rect() {
        let spec = InsetSpec { border: false, ..basic_spec() };
        let nodes = build_inset_nodes(&spec, &plot());
        // With no background and no shadow, there should be no Rect nodes.
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert_eq!(rect_count, 0, "no border or shadow should mean no rect nodes");
    }

    // ── build_inset_nodes: background ───────────────────────────────────────

    #[test]
    fn inset_with_background_includes_extra_rect() {
        let spec = InsetSpec {
            border: false,
            background: Some("#ffffff".to_string()),
            ..basic_spec()
        };
        let nodes = build_inset_nodes(&spec, &plot());
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert_eq!(rect_count, 1, "expected exactly one rect for background fill");
    }

    // ── build_inset_nodes: shadow ────────────────────────────────────────────

    #[test]
    fn inset_with_shadow_has_additional_rect() {
        let spec_no_shadow = basic_spec(); // border=true, shadow=false
        let spec_shadow = InsetSpec { shadow: true, ..basic_spec() }; // border=true, shadow=true

        let nodes_no_shadow = build_inset_nodes(&spec_no_shadow, &plot());
        let nodes_shadow = build_inset_nodes(&spec_shadow, &plot());

        let rects_no_shadow = nodes_no_shadow.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        let rects_shadow = nodes_shadow.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();

        assert_eq!(
            rects_shadow,
            rects_no_shadow + 1,
            "shadow should add one extra rect node"
        );
    }

    // ── build_inset_nodes: SVG embedding ─────────────────────────────────────

    #[test]
    fn inset_always_includes_raw_svg_node() {
        let spec = basic_spec();
        let nodes = build_inset_nodes(&spec, &plot());
        let raw_count = nodes.iter().filter(|n| matches!(n, SceneNode::Raw { .. })).count();
        assert_eq!(raw_count, 1, "expected exactly one Raw SVG node");
    }

    #[test]
    fn inset_raw_svg_is_nested_svg_element() {
        let spec = basic_spec();
        let nodes = build_inset_nodes(&spec, &plot());
        if let Some(SceneNode::Raw { svg }) = nodes.iter().find(|n| matches!(n, SceneNode::Raw { .. })) {
            assert!(svg.starts_with("<svg "), "Raw node should be a nested <svg> element");
            assert!(svg.contains("overflow=\"hidden\""));
        }
    }

    #[test]
    fn inset_bounds_map_correctly() {
        let spec = InsetSpec {
            bounds: [0.0, 0.0, 1.0, 1.0], // full plot area
            border: true,
            ..basic_spec()
        };
        let nodes = build_inset_nodes(&spec, &plot());
        // The border rect should cover the full plot area.
        let border_rect = nodes.iter().find(|n| {
            if let SceneNode::Rect { style, .. } = n {
                style.stroke.is_some() && style.fill.is_none()
            } else {
                false
            }
        });
        if let Some(SceneNode::Rect { x, y, w, h, .. }) = border_rect {
            assert!((x - 50.0).abs() < 0.1, "rect x should be plot_area.x");
            assert!((y - 50.0).abs() < 0.1, "rect y should be plot_area.y");
            assert!((w - 300.0).abs() < 0.1, "rect w should match plot width");
            assert!((h - 200.0).abs() < 0.1, "rect h should match plot height");
        } else {
            panic!("expected a border rect node");
        }
    }

    // ── build_inset_nodes: connector ─────────────────────────────────────────

    #[test]
    fn inset_with_connect_to_includes_line_nodes() {
        let spec = InsetSpec {
            connect_to: Some([100.0, 100.0]),
            connect_style: "lines".to_string(),
            ..basic_spec()
        };
        let nodes = build_inset_nodes(&spec, &plot());
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 2, "expected 2 connector lines (to 2 nearest corners)");
    }

    #[test]
    fn inset_without_connect_to_has_no_line_nodes() {
        let spec = InsetSpec { connect_to: None, ..basic_spec() };
        let nodes = build_inset_nodes(&spec, &plot());
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 0, "no connector lines without connect_to");
    }
}
