//! Inset chart embedding — positions a pre-rendered SVG at normalized bounds
//! within the plot area, with optional border, background, shadow, and connector.
//!
//! Insets use SVG's native nested `<svg>` element support to embed the
//! pre-rendered content inside a positioned, sized viewport. Each inset's
//! `clipPath`/colorbar/legend-clip ids are namespaced via
//! [`uniquify_clip_ids_with_prefix`](crate::render::svg::uniquify_clip_ids_with_prefix)
//! before embedding, since the pre-rendered inset body numbers its own ids
//! from zero independently of the host chart.

use ferrum_scene::{FillStroke, RawAnchor, SceneNode, StrokeStyle};

use crate::layout::Rect;
use crate::render::chart_config::InsetSpec;
use crate::render::color::parse_color;
use crate::render::draw::to_scene_color;
use crate::render::svg::uniquify_clip_ids_with_prefix;

/// Build scene nodes that embed the inset SVG at `spec.bounds` within `plot_area`.
///
/// `inset_idx` namespaces this inset's clip/colorbar/legend-clip ids
/// (`inset{inset_idx}-ferrum-clip-…`) so they stay disjoint from both the
/// host chart's own (unprefixed) ids and any other inset embedded in the
/// same chart. `spec.svg` is a fully independent pre-rendered document that
/// numbers its own ids from zero, so without this the host and inset (or two
/// insets) can define the same `id="ferrum-clip-0"`; SVG ids are
/// document-scoped, so `url(#ferrum-clip-0)` then resolves to whichever
/// definition comes first and the other clips its content by the wrong
/// rect. Caller passes a distinct `inset_idx` per `StructuralSpec::Inset`
/// processed for this chart (see `scene_build::build_structural_nodes`).
///
/// Rendering order (back to front):
/// 1. Drop shadow (if `spec.shadow`)
/// 2. Background fill rect (if `spec.background`)
/// 3. The SVG content as a `SceneNode::Raw`
/// 4. Border rect (if `spec.border`)
/// 5. Connector lines from `spec.connect_to` to inset corners (if set)
pub fn build_inset_nodes(spec: &InsetSpec, plot_area: &Rect, inset_idx: usize) -> Vec<SceneNode> {
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
    // with x/y/width/height plus the original viewBox so the content scales
    // to fit the inset bounds (without viewBox it just clips at 1:1).
    let inner_content =
        uniquify_clip_ids_with_prefix(strip_svg_wrapper(&spec.svg), &format!("inset{inset_idx}"));
    let viewbox = extract_viewbox(&spec.svg);
    let svg_raw = format!(
        "<svg x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\" viewBox=\"{}\" overflow=\"hidden\">{}</svg>",
        px_left, px_top, px_w, px_h, viewbox, inner_content
    );
    // Chrome: insets are fixed overlays positioned in normalized plot-area space,
    // not anchored to data coordinates. They do not track pan/zoom.
    nodes.push(SceneNode::Raw { svg: svg_raw, anchor: RawAnchor::Chrome });

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

/// Extract the `viewBox` attribute value from the outer `<svg>` tag.
/// Returns `"0 0 640 480"` as fallback if not found (ferrum default).
fn extract_viewbox(svg: &str) -> String {
    let s = svg.trim();
    if let Some(start) = s.find("viewBox=\"") {
        let after = &s[start + 9..];
        if let Some(end) = after.find('"') {
            return after[..end].to_string();
        }
    }
    "0 0 640 480".to_string()
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
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert!(rect_count >= 1, "expected at least one rect node for border");
    }

    #[test]
    fn inset_without_border_has_no_border_rect() {
        let spec = InsetSpec { border: false, ..basic_spec() };
        let nodes = build_inset_nodes(&spec, &plot(), 0);
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
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        assert_eq!(rect_count, 1, "expected exactly one rect for background fill");
    }

    // ── build_inset_nodes: shadow ────────────────────────────────────────────

    #[test]
    fn inset_with_shadow_has_additional_rect() {
        let spec_no_shadow = basic_spec(); // border=true, shadow=false
        let spec_shadow = InsetSpec { shadow: true, ..basic_spec() }; // border=true, shadow=true

        let nodes_no_shadow = build_inset_nodes(&spec_no_shadow, &plot(), 0);
        let nodes_shadow = build_inset_nodes(&spec_shadow, &plot(), 0);

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
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let raw_count = nodes.iter().filter(|n| matches!(n, SceneNode::Raw { .. })).count();
        assert_eq!(raw_count, 1, "expected exactly one Raw SVG node");
    }

    #[test]
    fn inset_raw_svg_is_nested_svg_element() {
        let spec = basic_spec();
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        if let Some(SceneNode::Raw { svg, .. }) = nodes.iter().find(|n| matches!(n, SceneNode::Raw { .. })) {
            assert!(svg.starts_with("<svg "), "Raw node should be a nested <svg> element");
            assert!(svg.contains("overflow=\"hidden\""));
        }
    }

    /// Inset Raw nodes must carry `anchor == Chrome` — insets are fixed overlays
    /// in normalized plot-area space, not anchored to data coordinates.
    #[test]
    fn inset_raw_node_has_chrome_anchor() {
        use ferrum_scene::RawAnchor;
        let spec = basic_spec();
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let raw = nodes.iter().find_map(|n| {
            if let SceneNode::Raw { anchor, .. } = n { Some(*anchor) } else { None }
        });
        assert_eq!(raw, Some(RawAnchor::Chrome), "inset Raw node must have Chrome anchor");
    }

    /// Regression for the corpus `duplicate_id` finding (#98 finding 1): the
    /// inset body is a fully independent pre-rendered SVG document that
    /// numbers its own clipPath ids from zero, exactly like the host chart
    /// does. Embedding it verbatim collides `id="ferrum-clip-0"` in the host
    /// with the inset's own `id="ferrum-clip-0"` — both `url(#ferrum-clip-0)`
    /// refs then resolve to the host's rect (SVG ids are document-scoped),
    /// clipping the inset content by the wrong region. The embedded body
    /// must carry a namespaced id disjoint from the host's un-prefixed one.
    #[test]
    fn inset_embed_namespaces_clip_ids_disjoint_from_a_colliding_host_id() {
        let spec = InsetSpec {
            svg: concat!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150">"#,
                r#"<defs><clipPath id="ferrum-clip-0"><rect width="200" height="150"/></clipPath></defs>"#,
                r#"<g clip-path="url(#ferrum-clip-0)"><circle cx="100" cy="75" r="50"/></g>"#,
                r#"</svg>"#,
            )
            .to_string(),
            ..basic_spec()
        };
        let host_id = r#"id="ferrum-clip-0""#;

        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let embedded = nodes
            .iter()
            .find_map(|n| if let SceneNode::Raw { svg, .. } = n { Some(svg.as_str()) } else { None })
            .expect("expected a Raw svg node");

        // The embedded body must no longer carry the bare host-colliding id...
        assert!(
            !embedded.contains(host_id),
            "embedded inset body still carries the un-namespaced host id: {embedded}"
        );
        // ...and its def + reference must both have been renamed to the same
        // namespaced id, so the clip-path reference still resolves internally.
        assert!(
            embedded.contains(r#"id="inset0-ferrum-clip-0""#),
            "expected namespaced clip def: {embedded}"
        );
        assert!(
            embedded.contains("url(#inset0-ferrum-clip-0)"),
            "expected namespaced clip reference: {embedded}"
        );
    }

    /// Two insets embedded into the same host chart each pre-render starting
    /// their own clip numbering at zero, so without a per-inset namespace a
    /// second inset would collide with the first, not just with the host.
    #[test]
    fn inset_embed_namespaces_two_insets_disjoint_from_each_other() {
        let two_inset_svg = || {
            concat!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150">"#,
                r#"<defs><clipPath id="ferrum-clip-0"><rect width="200" height="150"/></clipPath></defs>"#,
                r#"<g clip-path="url(#ferrum-clip-0)"><circle cx="100" cy="75" r="50"/></g>"#,
                r#"</svg>"#,
            )
            .to_string()
        };
        let spec_a = InsetSpec { svg: two_inset_svg(), ..basic_spec() };
        let spec_b = InsetSpec { svg: two_inset_svg(), ..basic_spec() };

        let nodes_a = build_inset_nodes(&spec_a, &plot(), 0);
        let nodes_b = build_inset_nodes(&spec_b, &plot(), 1);
        let raw_of = |nodes: &[SceneNode]| {
            nodes
                .iter()
                .find_map(|n| if let SceneNode::Raw { svg, .. } = n { Some(svg.clone()) } else { None })
                .expect("expected a Raw svg node")
        };
        let embedded_a = raw_of(&nodes_a);
        let embedded_b = raw_of(&nodes_b);

        assert!(embedded_a.contains(r#"id="inset0-ferrum-clip-0""#));
        assert!(embedded_b.contains(r#"id="inset1-ferrum-clip-0""#));
        assert!(
            !embedded_b.contains(r#"id="inset0-ferrum-clip-0""#),
            "second inset must not collide with the first inset's namespaced id"
        );
    }

    #[test]
    fn inset_bounds_map_correctly() {
        let spec = InsetSpec {
            bounds: [0.0, 0.0, 1.0, 1.0], // full plot area
            border: true,
            ..basic_spec()
        };
        let nodes = build_inset_nodes(&spec, &plot(), 0);
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
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 2, "expected 2 connector lines (to 2 nearest corners)");
    }

    #[test]
    fn inset_without_connect_to_has_no_line_nodes() {
        let spec = InsetSpec { connect_to: None, ..basic_spec() };
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 0, "no connector lines without connect_to");
    }

    // ── extract_viewbox: R1 port (bug_hunt_render_pipeline.rs) ───────────────
    // Zero prior in-src coverage of this function.

    #[test]
    fn extract_viewbox_standard() {
        let svg = r#"<svg viewBox="0 0 200 150" width="200" height="150"><circle/></svg>"#;
        assert_eq!(extract_viewbox(svg), "0 0 200 150");
    }

    #[test]
    fn extract_viewbox_missing_returns_default() {
        let svg = r#"<svg width="200" height="150"><circle/></svg>"#;
        assert_eq!(extract_viewbox(svg), "0 0 640 480");
    }

    #[test]
    fn extract_viewbox_empty_and_non_svg_input_return_default() {
        assert_eq!(extract_viewbox(""), "0 0 640 480");
        assert_eq!(extract_viewbox("<div>hello</div>"), "0 0 640 480");
    }

    #[test]
    fn extract_viewbox_malformed_no_closing_quote_returns_default() {
        // No closing quote after the viewBox value: after.find('"') is None,
        // so the parser falls through to the default rather than panicking
        // or returning a truncated fragment.
        let svg = r#"<svg viewBox="0 0 200 150><circle/></svg>"#;
        assert_eq!(extract_viewbox(svg), "0 0 640 480");
    }

    #[test]
    fn extract_viewbox_single_quotes_not_detected() {
        // Only the double-quoted `viewBox="..."` form is recognized; this is a
        // documented gap, not a panic risk.
        let svg = "<svg viewBox='0 0 100 100'><rect/></svg>";
        assert_eq!(extract_viewbox(svg), "0 0 640 480");
    }

    #[test]
    fn extract_viewbox_negative_and_decimal_values_preserved() {
        assert_eq!(
            extract_viewbox(r#"<svg viewBox="-50 -25 200 150"><rect/></svg>"#),
            "-50 -25 200 150"
        );
        assert_eq!(
            extract_viewbox(r#"<svg viewBox="0.5 1.5 199.5 148.5"><rect/></svg>"#),
            "0.5 1.5 199.5 148.5"
        );
    }

    // ── strip_svg_wrapper: additional edges (R1 port) ─────────────────────────

    #[test]
    fn strip_svg_wrapper_no_closing_tag_returns_rest_of_content() {
        let inner = strip_svg_wrapper("<svg><circle/>");
        assert_eq!(inner, "<circle/>");
    }

    #[test]
    fn strip_svg_wrapper_no_closing_bracket_returns_input_unchanged() {
        let svg = "<svg malformed";
        assert_eq!(strip_svg_wrapper(svg), svg);
    }

    #[test]
    fn strip_svg_wrapper_self_closing_produces_empty_inner() {
        assert_eq!(strip_svg_wrapper("<svg/>"), "");
    }

    // ── build_inset_nodes: bounds/connector edge cases (R1 port) ──────────────

    #[test]
    fn inset_inverted_bounds_clamp_to_1px_width() {
        // left > right in normalized bounds produces a negative pixel span,
        // which `.max(1.0)` clamps to a 1px-wide inset rather than a
        // negative-width Rect.
        let spec = InsetSpec { bounds: [0.9, 0.1, 0.1, 0.9], ..basic_spec() };
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let raw = nodes.iter().find_map(|n| {
            if let SceneNode::Raw { svg, .. } = n { Some(svg) } else { None }
        }).expect("expected a Raw svg node");
        assert!(raw.contains("width=\"1.000\""), "expected 1px-clamped width; got: {raw}");
    }

    #[test]
    fn inset_nan_connect_to_does_not_panic_and_draws_two_lines() {
        // Corner-distance sorting uses `.partial_cmp(...).unwrap_or(Equal)` so a
        // NaN connect_to point (e.g. from an unresolved scale) must not panic;
        // it degrades to an arbitrary (but stable) choice of the first two
        // corners in sort order.
        let spec = InsetSpec {
            connect_to: Some([f64::NAN, f64::NAN]),
            connect_style: "lines".to_string(),
            ..basic_spec()
        };
        let nodes = build_inset_nodes(&spec, &plot(), 0);
        let line_count = nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(line_count, 2, "NaN connect_to must still draw exactly 2 connector lines");
    }
}
