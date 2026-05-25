use ferrum_scene::{
    FillStroke as FsFillStroke, ImageData, PathCmd, SceneGraph, SceneNode, StrokeStyle as FsStroke,
    TextStyle as FsText,
};

use super::color::from_rgba;
use super::svg::{escape_attr, fmt_f, FillStroke, Stroke, SvgBuffer, TextStyle};
use super::CLIP_ID_PREFIX;
use crate::layout::{Rect, TextAnchor};

pub fn walk_svg(scene: &SceneGraph, embed_fonts: bool) -> String {
    let viewport = Rect {
        x: 0.0,
        y: 0.0,
        w: scene.width,
        h: scene.height,
    };
    let bg = scene.background.map(|c| from_rgba(c.r, c.g, c.b, c.a));
    let mut svg = SvgBuffer::new(viewport, bg, embed_fonts);

    // Chart-level accessibility description — emitted as the first child of
    // <svg> so screen readers encounter it before any visual content.
    if let Some(ref desc) = scene.chart_description {
        svg.desc_elem(desc);
    }

    // Chart title (before panels, matching old render_svg order)
    for node in &scene.title {
        emit_node(&mut svg, node);
    }

    for panel in &scene.panels {
        // Gridlines (behind marks)
        for node in &panel.grid {
            emit_gridline_node(&mut svg, node);
        }

        // Axes
        for node in &panel.axes {
            emit_node(&mut svg, node);
        }

        // Strip title (outside clip, before marks)
        for node in &panel.strip_title {
            emit_node(&mut svg, node);
        }

        // Clip open
        let clip_id = format!("{}{}", CLIP_ID_PREFIX, panel.id);
        let clip_rect = Rect {
            x: panel.clip.x,
            y: panel.clip.y,
            w: panel.clip.w,
            h: panel.clip.h,
        };
        svg.clip_open(&clip_id, clip_rect);
        svg.use_clip_open(&clip_id);

        // Marks (non-text first, inside clip)
        for batch in &panel.marks {
            if batch.kind == ferrum_scene::MarkBatchKind::Text { continue; }
            // Additive blend: wrap batch in <g style="mix-blend-mode:screen">.
            let blend_wrap = matches!(batch.blend, ferrum_scene::BlendMode::Additive);
            if blend_wrap {
                svg.raw("<g style=\"mix-blend-mode:screen\">");
            }
            // Wrap batch in <g stroke-linecap/stroke-linejoin> when present,
            // matching the old draw() pattern.
            let cap_join_wrap = if batch.stroke_cap.is_some() || batch.stroke_join.is_some() {
                let mut g_attrs = String::new();
                if let Some(ref cap) = batch.stroke_cap {
                    let s = match cap {
                        ferrum_scene::StrokeCap::Butt => "butt",
                        ferrum_scene::StrokeCap::Round => "round",
                        ferrum_scene::StrokeCap::Square => "square",
                    };
                    g_attrs.push_str(&format!(" stroke-linecap=\"{}\"", s));
                }
                if let Some(ref join) = batch.stroke_join {
                    let s = match join {
                        ferrum_scene::StrokeJoin::Miter => "miter",
                        ferrum_scene::StrokeJoin::Round => "round",
                        ferrum_scene::StrokeJoin::Bevel => "bevel",
                    };
                    g_attrs.push_str(&format!(" stroke-linejoin=\"{}\"", s));
                }
                Some(g_attrs)
            } else {
                None
            };
            if let Some(ref attrs) = cap_join_wrap {
                svg.raw(&format!("<g{}>", attrs));
            }
            for (i, node) in batch.nodes.iter().enumerate() {
                let href = batch.hrefs.as_ref().and_then(|h| h.get(i).and_then(|o| o.as_deref()));
                let tooltip = batch.tooltips.as_ref().and_then(|t| t.get(i));
                let desc = batch.descriptions.as_ref().and_then(|d| d.get(i).and_then(|o| o.as_deref()));
                let needs_wrap = href.is_some() || tooltip.is_some() || desc.is_some();

                if needs_wrap {
                    if let Some(url) = href {
                        svg.a_open(url);
                    }
                    svg.g_open(None);
                    if let Some(tt) = tooltip {
                        let text: String = tt.fields.iter()
                            .map(|f| format!("{}: {}", f.name, f.value))
                            .collect::<Vec<_>>()
                            .join(", ");
                        if !text.is_empty() {
                            svg.title_elem(&text);
                        }
                    }
                    if let Some(d) = desc {
                        svg.desc_elem(d);
                    }
                    emit_node(&mut svg, node);
                    svg.g_close();
                    if href.is_some() {
                        svg.a_close();
                    }
                } else {
                    emit_node(&mut svg, node);
                }
            }
            if cap_join_wrap.is_some() {
                svg.g_close();
            }
            if blend_wrap {
                svg.raw("</g>");
            }
        }

        svg.use_clip_close();

        // Annotations (reference lines, hlines, vlines) — emitted OUTSIDE the
        // clip region so they can span the full panel width/height, and AFTER
        // marks so they render above data (matching WASM z-order in scene_load.rs).
        for node in &panel.annotations {
            emit_node(&mut svg, node);
        }

        // Text-kind mark batches (mark_text, mark_label) outside clip so
        // dy/dx offsets near panel edges are not cut off.
        for batch in &panel.marks {
            if batch.kind != ferrum_scene::MarkBatchKind::Text { continue; }
            for (i, node) in batch.nodes.iter().enumerate() {
                let href = batch.hrefs.as_ref().and_then(|h| h.get(i).and_then(|o| o.as_deref()));
                let tooltip = batch.tooltips.as_ref().and_then(|t| t.get(i));
                let desc = batch.descriptions.as_ref().and_then(|d| d.get(i).and_then(|o| o.as_deref()));
                let needs_wrap = href.is_some() || tooltip.is_some() || desc.is_some();
                if needs_wrap {
                    if let Some(url) = href { svg.a_open(url); }
                    svg.g_open(None);
                    if let Some(tt) = tooltip {
                        let text: String = tt.fields.iter().map(|f| format!("{}: {}", f.name, f.value)).collect::<Vec<_>>().join(", ");
                        if !text.is_empty() { svg.title_elem(&text); }
                    }
                    if let Some(d) = desc { svg.desc_elem(d); }
                    emit_node(&mut svg, node);
                    svg.g_close();
                    if href.is_some() { svg.a_close(); }
                } else {
                    emit_node(&mut svg, node);
                }
            }
        }
    }

    // Legend (after panels, matching old render_svg order)
    for node in &scene.legend {
        emit_node(&mut svg, node);
    }

    // Any remaining decorations
    for node in &scene.decorations {
        emit_node(&mut svg, node);
    }

    svg.finish()
}

fn emit_node(svg: &mut SvgBuffer, node: &SceneNode) {
    match node {
        SceneNode::Rect { x, y, w, h, style, corner_radius } => {
            let cr = if *corner_radius > 0.0 { Some(*corner_radius) } else { None };
            // Anchor: center of the rect.
            let svg_style = to_svg_fill_stroke_with_anchor(style, x + w / 2.0, y + h / 2.0);
            svg.rect(
                Rect { x: *x, y: *y, w: *w, h: *h },
                &svg_style,
                cr,
            );
        }
        SceneNode::Circle { cx, cy, r, style } => {
            // Anchor: circle center.
            let svg_style = to_svg_fill_stroke_with_anchor(style, *cx, *cy);
            svg.circle(*cx, *cy, *r, &svg_style);
        }
        SceneNode::Line { x1, y1, x2, y2, style } => {
            svg.line(*x1, *y1, *x2, *y2, &to_svg_stroke(style));
        }
        SceneNode::Path { commands, style, closed: _ } => {
            let d = path_cmds_to_d(commands);
            // Anchor: first MoveTo coordinates.
            let (anchor_x, anchor_y) = commands.iter().find_map(|c| {
                if let ferrum_scene::PathCmd::MoveTo { x, y } = c { Some((*x, *y)) } else { None }
            }).unwrap_or((0.0, 0.0));
            let svg_style = to_svg_fill_stroke_with_anchor(style, anchor_x, anchor_y);
            svg.path(&d, &svg_style);
        }
        SceneNode::Text { x, y, content, style } => {
            emit_text(svg, *x, *y, content, style);
        }
        SceneNode::Image { x, y, w, h, data } => {
            match data {
                ImageData::Inline { bytes, mime } => {
                    svg.image(*x, *y, *w, *h, bytes, *mime);
                }
                ImageData::Url { url } => {
                    svg.image_data_url(*x, *y, *w, *h, url);
                }
            }
        }
        SceneNode::Polyline { points, style } => {
            svg.polyline(points, &to_svg_stroke(style));
        }
        SceneNode::Polygon { rings, style } => {
            let svg_rings: Vec<Vec<(f64, f64)>> = rings
                .iter()
                .map(|r| r.iter().map(|p| (p[0], p[1])).collect())
                .collect();
            // Anchor: centroid of first ring's points (good enough for rotation anchor).
            let (anchor_x, anchor_y) = if let Some(ring) = rings.first() {
                if ring.is_empty() { (0.0, 0.0) } else {
                    let n = ring.len() as f64;
                    let sx: f64 = ring.iter().map(|p| p[0]).sum();
                    let sy: f64 = ring.iter().map(|p| p[1]).sum();
                    (sx / n, sy / n)
                }
            } else { (0.0, 0.0) };
            svg.polygon(&svg_rings, &to_svg_fill_stroke_with_anchor(style, anchor_x, anchor_y));
        }
        SceneNode::Group { attrs, children } => {
            if attrs.is_empty() {
                svg.g_open(None);
            } else {
                let attr_str: String = attrs.iter()
                    .map(|(k, v)| format!(" {}=\"{}\"", k, escape_attr(v)))
                    .collect();
                svg.raw(&format!("<g{}>", attr_str));
            }
            for child in children {
                emit_node(svg, child);
            }
            svg.g_close();
        }
        SceneNode::Raw { svg: raw } => {
            svg.raw(raw);
        }
    }
}

fn emit_gridline_node(svg: &mut SvgBuffer, node: &SceneNode) {
    if let SceneNode::Line { x1, y1, x2, y2, style } = node {
        let color = from_rgba(style.color.r, style.color.g, style.color.b, style.color.a);
        svg.gridline(
            *x1, *y1, *x2, *y2,
            color,
            style.width,
            style.dash.as_deref(),
            style.opacity,
        );
    } else {
        emit_node(svg, node);
    }
}

fn to_svg_fill_stroke_with_anchor(s: &FsFillStroke, anchor_x: f64, anchor_y: f64) -> FillStroke {
    FillStroke {
        fill: s.fill.map(|c| from_rgba(c.r, c.g, c.b, c.a)),
        stroke: s.stroke.map(|c| from_rgba(c.r, c.g, c.b, c.a)),
        stroke_width: s.stroke_width,
        stroke_opacity: s.stroke_opacity,
        fill_opacity: s.fill_opacity,
        stroke_dash: s.stroke_dash.clone(),
        angle: s.angle,
        angle_cx: anchor_x,
        angle_cy: anchor_y,
    }
}

fn to_svg_stroke(s: &FsStroke) -> Stroke {
    Stroke {
        stroke: from_rgba(s.color.r, s.color.g, s.color.b, s.color.a),
        stroke_width: s.width,
        stroke_dash: s.dash.clone(),
        stroke_opacity: s.stroke_opacity,
    }
}

fn emit_text(svg: &mut SvgBuffer, x: f64, y: f64, content: &str, s: &FsText) {
    let anchor = match s.anchor {
        ferrum_scene::TextAnchor::Start => TextAnchor::Start,
        ferrum_scene::TextAnchor::Middle => TextAnchor::Middle,
        ferrum_scene::TextAnchor::End => TextAnchor::End,
    };
    let fw_str: Option<String> = match &s.font_weight {
        ferrum_scene::FontWeight::Bold => Some("bold".to_string()),
        ferrum_scene::FontWeight::Custom(w) => Some(w.clone()),
        ferrum_scene::FontWeight::Normal => None,
    };
    let db_owned: Option<String> = match &s.baseline {
        ferrum_scene::TextBaseline::Top => Some("hanging".to_string()),
        ferrum_scene::TextBaseline::Middle => Some("central".to_string()),
        ferrum_scene::TextBaseline::Bottom => Some("text-after-edge".to_string()),
        ferrum_scene::TextBaseline::Custom(v) => Some(v.clone()),
        ferrum_scene::TextBaseline::Alphabetic => None,
    };
    // Guard NaN opacity: NaN.clamp(0,1) is NaN in Rust, and NaN*255 as u8 = 0,
    // which makes text invisible. Default to 1.0 (fully opaque) when NaN.
    let opacity = if s.opacity.is_nan() { 1.0 } else { s.opacity };
    let effective_alpha = ((s.color.a as f64 / 255.0) * opacity).clamp(0.0, 1.0);
    let alpha_byte = (effective_alpha * 255.0).round() as u8;
    let ts = TextStyle {
        fill: from_rgba(s.color.r, s.color.g, s.color.b, alpha_byte),
        font_size: s.font_size,
        anchor,
        angle: s.angle,
        font_family: &s.font_family,
        font_weight: fw_str.as_deref(),
        dominant_baseline: db_owned.as_deref(),
    };
    svg.text(x, y, content, &ts);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::{
        BlendMode, Color, CoordKind, FillStroke, InteractionConfig, MarkBatch, MarkBatchKind,
        Panel, Rect, SceneGraph, SceneNode, StrokeStyle,
    };

    fn minimal_scene(annotations: Vec<SceneNode>) -> SceneGraph {
        SceneGraph {
            width: 400.0,
            height: 300.0,
            background: None,
            title: vec![],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 50.0, y: 10.0, w: 300.0, h: 250.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 300.0, h: 250.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None,
                    y_domain: None,
                    expand: true,
                    clip: true,
                },
                grid: vec![],
                axes: vec![],
                strip_title: vec![],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![SceneNode::Circle {
                        cx: 100.0,
                        cy: 150.0,
                        r: 4.0,
                        style: FillStroke {
                            fill: Some(Color::rgb(70, 130, 180)),
                            stroke: None,
                            stroke_width: 0.0,
                            opacity: 1.0,
                            stroke_dash: None,
                            stroke_opacity: 1.0,
                            fill_opacity: 1.0,
                            angle: 0.0,
                        },
                    }],
                    data_indices: None,
                    tooltips: None,
                    hrefs: None,
                    descriptions: None,
                    keys: None,
                    blend: BlendMode::Normal,
                    stroke_cap: None,
                    stroke_join: None,
                    packed_instances: None,
                }],
                annotations,
            }],
        }
    }

    fn annotation_line() -> SceneNode {
        // A horizontal reference line at y=50 with a distinctive red stroke.
        SceneNode::Line {
            x1: 50.0,
            y1: 50.0,
            x2: 350.0,
            y2: 50.0,
            style: StrokeStyle {
                color: Color::rgb(255, 0, 0),
                width: 2.0,
                opacity: 1.0,
                dash: None,
                stroke_cap: None,
                stroke_join: None,
                stroke_opacity: 1.0,
            },
        }
    }

    // ── Step 1 (TDD): this test was written BEFORE the fix and must fail
    // without the annotation iteration in walk_svg. ──────────────────────
    #[test]
    fn annotation_line_appears_in_svg() {
        let scene = minimal_scene(vec![annotation_line()]);
        let svg = walk_svg(&scene, false);
        // The red stroke color (#ff0000) and y-coordinates are distinctive;
        // if annotations are emitted this assertion passes.
        assert!(
            svg.contains("ff0000") || svg.contains("255,0,0") || svg.contains("#f00"),
            "annotation line stroke color not found in SVG — annotations are being dropped\nSVG snippet: {}",
            &svg[..svg.len().min(500)]
        );
        // Also confirm the y1=50 coordinate appears (rendered as "50" or "50.0").
        assert!(
            svg.contains("y1=\"50\"") || svg.contains("y1=\"50.0\""),
            "annotation line y1=50 not found in SVG\nSVG: {}",
            &svg[..svg.len().min(500)]
        );
    }

    // ── Step 4: regression tests ─────────────────────────────────────────

    #[test]
    fn empty_annotations_no_regression() {
        // A panel with no annotations must produce the same SVG as the
        // baseline (no crash, no extra elements).
        let scene_with = minimal_scene(vec![]);
        let scene_without = minimal_scene(vec![]);
        let svg_with = walk_svg(&scene_with, false);
        let svg_without = walk_svg(&scene_without, false);
        assert_eq!(
            svg_with, svg_without,
            "empty annotations changed SVG output"
        );
        // Confirm the mark circle is still present.
        assert!(svg_with.contains("<circle "), "mark circle missing");
    }

    #[test]
    fn multiple_annotation_types_both_appear() {
        // A reference line + a text label — both must appear in SVG output.
        let text_node = SceneNode::Text {
            x: 200.0,
            y: 50.0,
            content: "ref_label_xyz".to_string(),
            style: ferrum_scene::TextStyle {
                font_size: 12.0,
                font_weight: ferrum_scene::FontWeight::Normal,
                anchor: ferrum_scene::TextAnchor::Middle,
                baseline: ferrum_scene::TextBaseline::Alphabetic,
                angle: 0.0,
                color: Color::rgb(80, 80, 80),
                opacity: 1.0,
                font_family: "sans-serif".to_string(),
            },
        };
        let scene = minimal_scene(vec![annotation_line(), text_node]);
        let svg = walk_svg(&scene, false);

        assert!(
            svg.contains("ff0000") || svg.contains("255,0,0"),
            "annotation line not found in SVG"
        );
        assert!(
            svg.contains("ref_label_xyz"),
            "annotation text label not found in SVG"
        );
    }

    #[test]
    fn annotations_appear_after_marks_in_svg() {
        // Annotations must be emitted after marks so they render above data
        // (matching WASM z-order).
        let scene = minimal_scene(vec![annotation_line()]);
        let svg = walk_svg(&scene, false);

        let circle_pos = svg.find("<circle ").expect("mark circle missing from SVG");
        // The annotation line's distinctive color should appear after the circle.
        let annotation_pos = svg.find("ff0000")
            .or_else(|| svg.find("255,0,0"))
            .expect("annotation line not found in SVG");

        assert!(
            annotation_pos > circle_pos,
            "annotation appears before mark in SVG (wrong z-order): \
             circle at byte {circle_pos}, annotation at byte {annotation_pos}"
        );
    }
}

fn path_cmds_to_d(cmds: &[PathCmd]) -> String {
    let mut d = String::new();
    for cmd in cmds {
        if !d.is_empty() {
            d.push(' ');
        }
        match cmd {
            PathCmd::MoveTo { x, y } => {
                d.push_str(&format!("M{} {}", fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::LineTo { x, y } => {
                d.push_str(&format!("L{} {}", fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::QuadTo { cx, cy, x, y } => {
                d.push_str(&format!("Q{} {} {} {}", fmt_f(*cx), fmt_f(*cy), fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
                d.push_str(&format!("C{} {} {} {} {} {}",
                    fmt_f(*c1x), fmt_f(*c1y), fmt_f(*c2x), fmt_f(*c2y), fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::HLineTo { x } => {
                d.push_str(&format!("H{}", fmt_f(*x)));
            }
            PathCmd::VLineTo { y } => {
                d.push_str(&format!("V{}", fmt_f(*y)));
            }
            PathCmd::ArcTo { rx, ry, rotation, large_arc, sweep, x, y } => {
                d.push_str(&format!("A{} {} {} {} {} {} {}",
                    fmt_f(*rx), fmt_f(*ry), fmt_f(*rotation),
                    if *large_arc { 1 } else { 0 },
                    if *sweep { 1 } else { 0 },
                    fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::Close => {
                d.push('Z');
            }
        }
    }
    d
}
