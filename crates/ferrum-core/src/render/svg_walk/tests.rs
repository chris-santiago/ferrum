//! Static SVG walker test coverage.
//!
//! This module carries two kinds of tests:
//! - the scene-walk ordering, layout-scale and text-opacity suite that the
//!   walker has always carried, and
//! - the tapered-stroke coverage: the outline geometry of a node with
//!   `widths` (half-thickness per vertex, caps, joins, dashes, paint, winding)
//!   asserted on parsed `<path>` rings, plus the pipeline matrix that renders a
//!   sized line through `render_svg` under every interpolate the vocabulary
//!   publishes.
//!
//! The tapered coverage parses rendered geometry and drives whole charts, so
//! inlined it outgrew the walker it tests; it lives here rather than in
//! `tests/` because it calls the crate-private `walk_svg` and render entry
//! points directly.

use super::*;
use ferrum_scene::{
    Color, CoordKind, FillStroke, InteractionConfig, LayoutScale, MarkBatch,
    MarkBatchKind, Panel, Rect, SceneGraph, SceneNode, StrokeStyle,
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
            plot_area: Rect {
                x: 50.0,
                y: 10.0,
                w: 300.0,
                h: 250.0,
            },
            clip: Rect {
                x: 50.0,
                y: 10.0,
                w: 300.0,
                h: 250.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            axes: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
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
                ..MarkBatch::default()
            }],
            annotations,
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
            color: Some(Color::rgb(255, 0, 0)),
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
    let svg = walk_svg(&scene, false).unwrap();
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
    let svg_with = walk_svg(&scene_with, false).unwrap();
    let svg_without = walk_svg(&scene_without, false).unwrap();
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
        slot: None,
        style: ferrum_scene::TextStyle {
            font_size: 12.0,
            font_weight: ferrum_scene::FontWeight::Normal,
            anchor: ferrum_scene::TextAnchor::Middle,
            baseline: ferrum_scene::TextBaseline::Alphabetic,
            angle: 0.0,
            color: Some(Color::rgb(80, 80, 80)),
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        },
    };
    let scene = minimal_scene(vec![annotation_line(), text_node]);
    let svg = walk_svg(&scene, false).unwrap();

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
    let svg = walk_svg(&scene, false).unwrap();

    let circle_pos = svg.find("<circle ").expect("mark circle missing from SVG");
    // The annotation line's distinctive color should appear after the circle.
    let annotation_pos = svg
        .find("ff0000")
        .or_else(|| svg.find("255,0,0"))
        .expect("annotation line not found in SVG");

    assert!(
        annotation_pos > circle_pos,
        "annotation appears before mark in SVG (wrong z-order): \
         circle at byte {circle_pos}, annotation at byte {annotation_pos}"
    );
}

// ── GH #89B: Panel::below_marks / chrome_above paint order ──────────────

/// `below_marks` (typed sibling of `grid`) must paint after `grid` and
/// before marks — the same visual position these nodes held when they
/// were commingled into `grid` itself.
#[test]
fn below_marks_appears_after_grid_and_before_marks_in_svg() {
    let mut scene = minimal_scene(vec![]);
    scene.panels[0].grid = vec![SceneNode::Line {
        x1: 0.0,
        y1: 0.0,
        x2: 100.0,
        y2: 0.0,
        style: StrokeStyle {
            color: Some(Color::rgb(0, 255, 0)),
            width: 1.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        },
    }];
    scene.panels[0].below_marks = vec![SceneNode::Text {
        x: 60.0,
        y: 70.0,
        content: "BELOWMARKXYZ".to_string(),
        slot: None,
        style: ferrum_scene::TextStyle {
            font_size: 12.0,
            font_weight: ferrum_scene::FontWeight::Normal,
            anchor: ferrum_scene::TextAnchor::Middle,
            baseline: ferrum_scene::TextBaseline::Alphabetic,
            angle: 0.0,
            color: Some(Color::rgb(80, 80, 80)),
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        },
    }];
    let svg = walk_svg(&scene, false).unwrap();

    let grid_pos = svg
        .find("00ff00")
        .or_else(|| svg.find("0,255,0"))
        .expect("grid line stroke color not found in SVG");
    let below_marks_pos = svg
        .find("BELOWMARKXYZ")
        .expect("below_marks text not found in SVG");
    let circle_pos = svg.find("<circle ").expect("mark circle missing from SVG");

    assert!(
        grid_pos < below_marks_pos,
        "below_marks must appear after grid: grid at {grid_pos}, below_marks at {below_marks_pos}"
    );
    assert!(
        below_marks_pos < circle_pos,
        "below_marks must appear before marks: below_marks at {below_marks_pos}, mark at {circle_pos}"
    );
}

/// `chrome_above` (typed sibling of `annotations`) must paint after marks
/// and before `annotations` — the deliberate z-order refinement so
/// above-marks user annotations always paint above above-marks axis
/// chrome.
#[test]
fn chrome_above_appears_after_marks_and_before_annotations_in_svg() {
    let mut scene = minimal_scene(vec![SceneNode::Text {
        x: 200.0,
        y: 50.0,
        content: "ANNOTATIONXYZ".to_string(),
        slot: None,
        style: ferrum_scene::TextStyle {
            font_size: 12.0,
            font_weight: ferrum_scene::FontWeight::Normal,
            anchor: ferrum_scene::TextAnchor::Middle,
            baseline: ferrum_scene::TextBaseline::Alphabetic,
            angle: 0.0,
            color: Some(Color::rgb(80, 80, 80)),
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        },
    }]);
    scene.panels[0].chrome_above = vec![annotation_line()];
    let svg = walk_svg(&scene, false).unwrap();

    let circle_pos = svg.find("<circle ").expect("mark circle missing from SVG");
    let chrome_pos = svg
        .find("ff0000")
        .or_else(|| svg.find("255,0,0"))
        .expect("chrome_above line not found in SVG");
    let annotation_pos = svg
        .find("ANNOTATIONXYZ")
        .expect("annotations text not found in SVG");

    assert!(
        circle_pos < chrome_pos,
        "chrome_above must appear after marks: mark at {circle_pos}, chrome_above at {chrome_pos}"
    );
    assert!(
        chrome_pos < annotation_pos,
        "chrome_above must appear before annotations (z-order refinement): \
         chrome_above at {chrome_pos}, annotations at {annotation_pos}"
    );
}

// ── MarkBatch::clip escape paint order (batch E Task 9, quality review RF1) ──

/// Full order-pin fixture: one panel carrying the baseline clip=true
/// Point batch (from `minimal_scene`), a clip=false Rect batch (the
/// escape under test, distinctive magenta fill), a `chrome_above` Line
/// (green), an annotation Line (`annotation_line()`, red), and a
/// Text-kind mark batch (distinctive content) — the four buckets whose
/// relative order after the clip closes is the actual open design
/// choice this test pins.
fn scene_with_clip_escape_order_fixture() -> SceneGraph {
    let mut scene = minimal_scene(vec![annotation_line()]);
    scene.panels[0].chrome_above = vec![SceneNode::Line {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 0.0,
        style: StrokeStyle {
            color: Some(Color::rgb(0, 255, 0)),
            width: 1.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        },
    }];
    scene.panels[0].marks.push(MarkBatch {
        clip: false,
        kind: MarkBatchKind::Rect,
        nodes: vec![SceneNode::Rect {
            x: 0.0,
            y: 0.0,
            w: 5.0,
            h: 5.0,
            style: FillStroke {
                fill: Some(Color::rgb(255, 0, 255)),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
            corner_radius: 0.0,
        }],
        ..MarkBatch::default()
    });
    scene.panels[0].marks.push(MarkBatch {
        kind: MarkBatchKind::Text,
        nodes: vec![SceneNode::Text {
            x: 1.0,
            y: 1.0,
            content: "TEXTLABELXYZ".to_string(),
            slot: None,
            style: ferrum_scene::TextStyle {
                font_size: 12.0,
                font_weight: ferrum_scene::FontWeight::Normal,
                anchor: ferrum_scene::TextAnchor::Middle,
                baseline: ferrum_scene::TextBaseline::Alphabetic,
                angle: 0.0,
                color: Some(Color::rgb(80, 80, 80)),
                opacity: 1.0,
                font_family: "sans-serif".to_string(),
            },
        }],
        ..MarkBatch::default()
    });
    scene
}

/// The real order pin the brief required: a `clip=false` batch must
/// paint AFTER the panel clip group closes AND before `chrome_above`,
/// `annotations`, and Text/Label batches — not merely "somewhere after
/// the clip closes" (which the two `render_svg_clip_false_*` tests in
/// `render/mod.rs` already covered and which relocating this loop to
/// the very end of `walk_svg` would still satisfy). Moving the
/// clip-exempt loop to after `chrome_above`/`annotations`/Text-Label
/// must fail this test.
#[test]
fn clip_exempt_batch_paints_after_clipped_marks_and_before_chrome_annotations_and_text() {
    let scene = scene_with_clip_escape_order_fixture();
    let svg = walk_svg(&scene, false).unwrap();

    let clip_open = svg.find("<g clip-path=").expect("clip group must be present");
    let clip_close = svg[clip_open..]
        .find("</g>")
        .map(|i| i + clip_open)
        .expect("clip group must close");

    let exempt_pos = svg.find("ff00ff").expect("clip-exempt rect fill not found in SVG");
    let chrome_pos = svg.find("00ff00").expect("chrome_above marker not found in SVG");
    let annotation_pos = svg
        .find("ff0000")
        .or_else(|| svg.find("255,0,0"))
        .expect("annotation marker not found in SVG");
    let text_pos = svg.find("TEXTLABELXYZ").expect("text/label marker not found in SVG");

    assert!(
        exempt_pos > clip_close,
        "clip-exempt batch must paint after the panel clip group closes: \
         exempt at {exempt_pos}, clip group closes at {clip_close}"
    );
    assert!(
        exempt_pos < chrome_pos,
        "clip-exempt batch must paint before chrome_above: \
         exempt at {exempt_pos}, chrome_above at {chrome_pos}"
    );
    assert!(
        exempt_pos < annotation_pos,
        "clip-exempt batch must paint before annotations: \
         exempt at {exempt_pos}, annotation at {annotation_pos}"
    );
    assert!(
        exempt_pos < text_pos,
        "clip-exempt batch must paint before Text/Label batches: \
         exempt at {exempt_pos}, text at {text_pos}"
    );
}

// ── per-panel `layout_scale` (ratio-fitted cells / W5 foundation) ────────

/// A scene whose only panel carries `ls` as its `layout_scale`, otherwise
/// identical to [`minimal_scene`]. Used to isolate the wrapper's effect
/// from everything else `walk_svg` emits.
fn scene_with_layout_scale(ls: LayoutScale) -> SceneGraph {
    let mut scene = minimal_scene(vec![]);
    scene.panels[0].layout_scale = ls;
    scene
}

#[test]
fn identity_layout_scale_emits_byte_identical_svg() {
    // The byte-stability anchor: a scene at the default (identity)
    // `layout_scale` must render exactly like a scene with the field
    // absent — no `<g transform>` wrapper, no other change — so every
    // flat/faceted golden stays byte-identical.
    let baseline = walk_svg(&minimal_scene(vec![]), false).unwrap();
    let explicit_identity = walk_svg(&scene_with_layout_scale(LayoutScale::identity()), false).unwrap();
    assert_eq!(
        baseline, explicit_identity,
        "explicit identity layout_scale must not change SVG output"
    );
    assert!(
        !explicit_identity.contains("<g transform="),
        "identity layout_scale must not emit a transform wrapper"
    );
}

#[test]
fn non_identity_layout_scale_wraps_panel_in_transform_group() {
    // A ratio-fitted cell (sx != sy, as JointChart marginals require)
    // must wrap the panel's content in a <g transform="translate(...)
    // scale(...)"> group carrying exactly those values.
    let ls = LayoutScale {
        sx: 0.5,
        sy: 0.2,
        tx: 10.0,
        ty: 20.0,
    };
    let svg = walk_svg(&scene_with_layout_scale(ls), false).unwrap();
    assert!(
        svg.contains(r#"<g transform="translate(10,20) scale(0.5,0.2)">"#),
        "expected non-uniform layout_scale transform wrapper not found\nSVG: {}",
        &svg[..svg.len().min(800)]
    );
    // The mark circle must still be present (inside the wrapper).
    assert!(
        svg.contains("<circle "),
        "mark circle missing under layout_scale wrapper"
    );
}

#[test]
fn non_identity_layout_scale_wrapper_closes() {
    // The transform group opened for a non-identity layout_scale must be
    // closed with a matching </g> (paired open/close, no dangling group).
    let ls = LayoutScale {
        sx: 2.0,
        sy: 2.0,
        tx: 0.0,
        ty: 0.0,
    };
    let svg = walk_svg(&scene_with_layout_scale(ls), false).unwrap();
    let opens = svg
        .matches("<g transform=\"translate(0,0) scale(2,2)\">")
        .count();
    assert_eq!(opens, 1, "expected exactly one layout_scale wrapper open");
}

/// R1 port (bug_hunt_draw.rs / bug_hunt_render_pipeline.rs): `NaN.clamp(0,1)`
/// is NaN in Rust, which would otherwise round-trip to `alpha_byte = 0`
/// (invisible text). `emit_text` guards NaN opacity to render fully opaque.
#[test]
fn emit_text_nan_opacity_renders_opaque_not_invisible() {
    let mut svg = SvgBuffer::new(
        crate::layout::Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 80.0,
        },
        None,
        false,
    );
    let style = FsText {
        font_size: 11.0,
        font_weight: ferrum_scene::FontWeight::Normal,
        anchor: ferrum_scene::TextAnchor::Start,
        baseline: ferrum_scene::TextBaseline::Alphabetic,
        angle: 0.0,
        color: Some(Color::rgb(0, 0, 0)),
        opacity: f64::NAN,
        font_family: "Inter".to_string(),
    };
    emit_text(&mut svg, 10.0, 10.0, "hi", &style);
    let out = svg.finish();
    assert!(
        out.contains("fill=\"#000000\""),
        "NaN opacity must render fully opaque (alpha=255), not invisible; got: {out}"
    );
}

// #99 residue: `emit_text`'s alpha-compositing formula
// (`effective_alpha = (color.a/255 * opacity).clamp(0,1)`, then
// `* 255).round()`) must be pinned by tests that call the real function,
// at the boundary and endpoint values spec §4.6 names: opacity 0 → alpha
// 0, opacity 1 → alpha 255, `color.a=128, opacity=0.5` → alpha 64, and
// opacity outside `[0, 1]` clamps to the same 255/0 endpoints. Fully
// opaque colors render as `fill="#rrggbb"`; any other alpha renders as
// `fill="rgba(r,g,b,a)"` (see `fmt_svg`). `text_style`/`emit_text_svg`
// below are shared construction helpers for the test group that follows.

/// Build an `FsText` style with the given color/opacity, all other
/// fields fixed at a plain default (used by the `emit_text_*` group).
fn text_style(color: Color, opacity: f64) -> FsText {
    FsText {
        font_size: 11.0,
        font_weight: ferrum_scene::FontWeight::Normal,
        anchor: ferrum_scene::TextAnchor::Start,
        baseline: ferrum_scene::TextBaseline::Alphabetic,
        angle: 0.0,
        color: Some(color),
        opacity,
        font_family: "Inter".to_string(),
    }
}

fn emit_text_svg(style: &FsText) -> String {
    let mut svg = SvgBuffer::new(
        crate::layout::Rect { x: 0.0, y: 0.0, w: 100.0, h: 80.0 },
        None,
        false,
    );
    emit_text(&mut svg, 10.0, 10.0, "hi", style);
    svg.finish()
}

#[test]
fn emit_text_opacity_zero_yields_zero_alpha() {
    let style = text_style(Color::rgb(10, 20, 30), 0.0);
    let out = emit_text_svg(&style);
    assert!(
        out.contains("fill=\"rgba(10,20,30,0.000)\""),
        "opacity=0 must composite to alpha=0; got: {out}"
    );
}

#[test]
fn emit_text_opacity_one_yields_full_alpha() {
    let style = text_style(Color::rgb(10, 20, 30), 1.0);
    let out = emit_text_svg(&style);
    assert!(
        out.contains("fill=\"#0a141e\""),
        "opacity=1 must composite to alpha=255 (opaque hex fill); got: {out}"
    );
}

#[test]
fn emit_text_color_alpha_128_opacity_half_yields_alpha_64() {
    let style = text_style(Color::rgba(10, 20, 30, 128), 0.5);
    let out = emit_text_svg(&style);
    // (128/255) * 0.5 * 255 = 64.0 exactly. 64/255 = 0.250980... -> "0.251".
    assert!(
        out.contains("fill=\"rgba(10,20,30,0.251)\""),
        "color.a=128, opacity=0.5 must composite to alpha=64; got: {out}"
    );
}

#[test]
fn emit_text_opacity_above_one_clamps_to_full_alpha() {
    let style = text_style(Color::rgb(10, 20, 30), 2.0);
    let out = emit_text_svg(&style);
    assert!(
        out.contains("fill=\"#0a141e\""),
        "opacity > 1 must clamp to alpha=255, not overflow; got: {out}"
    );
}

#[test]
fn emit_text_negative_opacity_clamps_to_zero_alpha() {
    let style = text_style(Color::rgb(10, 20, 30), -0.5);
    let out = emit_text_svg(&style);
    assert!(
        out.contains("fill=\"rgba(10,20,30,0.000)\""),
        "negative opacity must clamp to alpha=0, not underflow; got: {out}"
    );
}

// ── Tapered strokes: a node with `widths` paints one filled outline ──

use crate::spec::chart::ChartSpec;
use crate::spec::encoding::{Encoding, EncodingSpec};
use crate::spec::mark::Mark;
use crate::spec::mark_style::MarkKwargsSpec;
use ferrum_scene::{StrokeCap, StrokeJoin};

const INK: Color = Color { r: 10, g: 20, b: 30, a: 255 };

/// One line batch carrying `nodes`, with the batch-level cap and join a
/// line mark's `stroke_cap=`/`stroke_join=` kwargs write.
fn line_scene(nodes: Vec<SceneNode>, cap: Option<StrokeCap>, join: Option<StrokeJoin>) -> SceneGraph {
    let mut scene = minimal_scene(vec![]);
    scene.panels[0].marks = vec![MarkBatch {
        kind: MarkBatchKind::Line,
        nodes,
        stroke_cap: cap,
        stroke_join: join,
        ..MarkBatch::default()
    }];
    scene
}

fn walk(scene: &SceneGraph) -> String {
    walk_svg(scene, false).expect("the scene walks")
}

fn polyline(points: &[(f64, f64)], widths: Option<&[f64]>, dash: Option<Vec<f64>>) -> SceneNode {
    SceneNode::Polyline {
        points: points.to_vec(),
        style: StrokeStyle {
            color: Some(INK),
            width: 1.5,
            opacity: 1.0,
            dash,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        },
        widths: widths.map(<[f64]>::to_vec),
    }
}

fn path(commands: Vec<PathCmd>, widths: Option<&[f64]>) -> SceneNode {
    SceneNode::Path {
        commands,
        style: FillStroke {
            fill: None,
            stroke: Some(INK),
            stroke_width: 1.5,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        closed: false,
        widths: widths.map(<[f64]>::to_vec),
    }
}

/// Every `<path>` painted as a filled outline (`stroke="none"`): its
/// element text and its rings, parsed from `d` (`M x y L x y ... Z`).
fn outlines(svg: &str) -> Vec<(String, Vec<Vec<(f64, f64)>>)> {
    svg.split("<path").skip(1)
        .map(|el| el[..el.find("/>").expect("a self-closed path")].to_string())
        .filter(|el| el.contains(" stroke=\"none\""))
        .map(|el| {
            let at = el.find(" d=\"").expect("a d attribute") + " d=\"".len();
            let d = &el[at..at + el[at..].find('"').expect("a closed d")];
            let rings = d.split('M').filter(|r| !r.trim().is_empty())
                .map(|ring| ring.trim().trim_end_matches('Z').split('L')
                    .map(|pt| {
                        let xy: Vec<f64> = pt.split_whitespace().map(|v| v.parse().expect("a number")).collect();
                        assert_eq!(xy.len(), 2, "vertex {pt:?} in {d}");
                        (xy[0], xy[1])
                    })
                    .collect())
                .collect();
            (el, rings)
        })
        .collect()
}

/// The one outline a single tapered node paints.
fn the_outline(svg: &str) -> (String, Vec<Vec<(f64, f64)>>) {
    let mut all = outlines(svg);
    assert_eq!(all.len(), 1, "one outline <path> per tapered node:\n{svg}");
    all.remove(0)
}

fn vertices(rings: &[Vec<(f64, f64)>]) -> Vec<(f64, f64)> {
    rings.iter().flatten().copied().collect()
}

/// SVG writes 3 decimals, so a vertex is "at" a point within 1e-3.
fn has_vertex(rings: &[Vec<(f64, f64)>], p: (f64, f64)) -> bool {
    vertices(rings).iter().any(|v| (v.0 - p.0).abs() < 1e-3 && (v.1 - p.1).abs() < 1e-3)
}

fn signed_area(ring: &[(f64, f64)]) -> f64 {
    let n = ring.len();
    (0..n).map(|i| {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        a.0 * b.1 - b.0 * a.1
    }).sum::<f64>() / 2.0
}

/// The summed area of an outline's rings (overlapping join pieces count
/// once per ring).
fn painted_area(rings: &[Vec<(f64, f64)>]) -> f64 {
    rings.iter().map(|ring| signed_area(ring).abs()).sum()
}

fn unit_normal(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let len = (b.0 - a.0).hypot(b.1 - a.1);
    (-(b.1 - a.1) / len, (b.0 - a.0) / len)
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// At every vertex, the outline's edge sits `w/2` either side of the
/// centerline, perpendicular to each segment meeting there.
fn assert_half_thickness(rings: &[Vec<(f64, f64)>], points: &[(f64, f64)], widths: &[f64]) {
    for (i, (&p, &w)) in points.iter().zip(widths).enumerate() {
        let segments = [i.checked_sub(1).map(|j| (j, i)), (i + 1 < points.len()).then_some((i, i + 1))];
        for (a, b) in segments.into_iter().flatten() {
            let n = unit_normal(points[a], points[b]);
            for side in [1.0, -1.0] {
                let edge = (p.0 + side * n.0 * w / 2.0, p.1 + side * n.1 * w / 2.0);
                assert!(has_vertex(rings, edge), "vertex {i}: no outline edge at {edge:?} (w/2 = {})", w / 2.0);
            }
        }
    }
}

#[test]
fn a_tapered_polyline_paints_one_outline_whose_half_thickness_is_half_its_width() {
    let points = [(0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (35.0, 4.0)];
    let widths = [2.0, 6.0, 4.0, 9.0];
    let svg = walk(&line_scene(vec![polyline(&points, Some(&widths), None)], None, None));
    assert!(!svg.contains("<polyline"), "a tapered node draws no stroked polyline:\n{svg}");
    let (_, rings) = the_outline(&svg);
    assert_half_thickness(&rings, &points, &widths);
    // Nothing reaches further from the centerline than the widest half-width.
    let reach = vertices(&rings).iter()
        .map(|v| points.iter().map(|p| distance(*v, *p)).fold(f64::INFINITY, f64::min))
        .fold(0.0, f64::max);
    assert!(reach <= 4.5 + 1e-3, "outline reaches {reach} from the centerline");
}

#[test]
fn a_tapered_path_outlines_every_command_family() {
    // Straight runs, the H/V shorthand the step interpolations write, and
    // the curve and arc commands, each flattened before outlining.
    let step = vec![
        PathCmd::MoveTo { x: 0.0, y: 0.0 },
        PathCmd::HLineTo { x: 10.0 },
        PathCmd::VLineTo { y: 10.0 },
        PathCmd::LineTo { x: 20.0, y: 10.0 },
    ];
    let svg = walk(&line_scene(vec![path(step, Some(&[2.0, 2.0, 4.0, 4.0]))], None, None));
    let (_, rings) = the_outline(&svg);
    assert_half_thickness(&rings, &[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0)], &[2.0, 2.0, 4.0, 4.0]);

    // A cubic from (0,0) to (30,10) with horizontal end tangents: the
    // endpoints carry their command's widths and the curve midpoint, at
    // parameter 1/2, the mean of the two.
    let cubic = vec![
        PathCmd::MoveTo { x: 0.0, y: 0.0 },
        PathCmd::CubicTo { c1x: 10.0, c1y: 0.0, c2x: 20.0, c2y: 10.0, x: 30.0, y: 10.0 },
    ];
    let svg = walk(&line_scene(vec![path(cubic, Some(&[2.0, 6.0]))], None, None));
    let (_, rings) = the_outline(&svg);
    // The flattened chords are not tangents, so each edge sits w/2 from
    // its vertex along the chord's normal: count the vertices at w/2.
    let at = |p: (f64, f64), half: f64| {
        vertices(&rings).iter().filter(|v| (distance(**v, p) - half).abs() < 1e-3).count()
    };
    assert!(at((0.0, 0.0), 1.0) >= 2, "start edge at w/2 = 1");
    assert!(at((30.0, 10.0), 3.0) >= 2, "end edge at w/2 = 3");
    assert!(at((15.0, 5.0), 2.0) >= 4, "the curve midpoint is 2 (= 4/2) from both chords' edges");
    assert!(rings.len() > 8, "the curve is flattened into many segments, got {} rings", rings.len());

    let quad = vec![
        PathCmd::MoveTo { x: 0.0, y: 0.0 },
        PathCmd::QuadTo { cx: 10.0, cy: 20.0, x: 20.0, y: 0.0 },
    ];
    let svg = walk(&line_scene(vec![path(quad, Some(&[2.0, 2.0]))], None, None));
    let (_, rings) = the_outline(&svg);
    // Its parameter-1/2 point is (10, 10), the apex.
    let at_apex = vertices(&rings).iter().filter(|v| (distance(**v, (10.0, 10.0)) - 1.0).abs() < 1e-3).count();
    assert!(at_apex >= 4, "quad apex edges at w/2 = 1: {at_apex}");

    // A half circle of radius 10 about (10, 0): every outline vertex is
    // 10 ± w/2 from the centre.
    let arc = vec![
        PathCmd::MoveTo { x: 0.0, y: 0.0 },
        PathCmd::ArcTo { rx: 10.0, ry: 10.0, rotation: 0.0, large_arc: false, sweep: true, x: 20.0, y: 0.0 },
    ];
    let svg = walk(&line_scene(vec![path(arc, Some(&[2.0, 2.0]))], None, None));
    let (_, rings) = the_outline(&svg);
    for v in vertices(&rings) {
        let r = distance(v, (10.0, 0.0));
        assert!(r > 9.0 - 2e-3 && r < 11.0 + 2e-3, "arc vertex {v:?} at {r}");
    }
    assert!(vertices(&rings).iter().any(|v| v.1 < -10.5), "the arc bulges to y = -11 (sweep toward -y)");
}

#[test]
fn tapered_caps_follow_the_stroke_cap() {
    let points = [(0.0, 0.0), (10.0, 0.0)];
    let widths = [2.0, 4.0];
    let extent = |cap: Option<StrokeCap>| {
        let svg = walk(&line_scene(vec![polyline(&points, Some(&widths), None)], cap, None));
        let vs = vertices(&the_outline(&svg).1);
        let min = vs.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
        let max = vs.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
        (min, max, vs)
    };
    for cap in [None, Some(StrokeCap::Butt)] {
        let (min, max, _) = extent(cap);
        assert!((min - 0.0).abs() < 1e-3 && (max - 10.0).abs() < 1e-3, "{cap:?}: butt ends at the vertices");
    }
    let (min, max, vs) = extent(Some(StrokeCap::Square));
    assert!((min + 1.0).abs() < 1e-3 && (max - 12.0).abs() < 1e-3, "square extends by w/2: {min} {max}");
    assert!(vs.iter().any(|v| (v.0 - 12.0).abs() < 1e-3 && (v.1 - 2.0).abs() < 1e-3), "square corner");
    let (min, max, vs) = extent(Some(StrokeCap::Round));
    assert!((min + 1.0).abs() < 1e-3 && (max - 12.0).abs() < 1e-3, "round reaches w/2: {min} {max}");
    let cap_points: Vec<_> = vs.iter().filter(|v| v.0 > 10.0 + 1e-3).collect();
    assert!(cap_points.len() >= 4, "a round cap is an arc: {cap_points:?}");
    for v in cap_points {
        assert!((distance(*v, (10.0, 0.0)) - 2.0).abs() < 1e-3, "round cap vertex {v:?} off radius 2");
    }
    // The node's own cap outranks the batch's.
    let mut node = polyline(&points, Some(&widths), None);
    if let SceneNode::Polyline { style, .. } = &mut node {
        style.stroke_cap = Some(StrokeCap::Square);
    }
    let svg = walk(&line_scene(vec![node], Some(StrokeCap::Butt), None));
    let max = vertices(&the_outline(&svg).1).iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
    assert!((max - 12.0).abs() < 1e-3, "the node's square cap wins over the batch's butt: {max}");
}

#[test]
fn tapered_joins_follow_the_stroke_join() {
    // A right-angle corner at (10, 0), width 4: the outer corner lies
    // toward (+x, -y).
    let corner = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
    let widths = [4.0, 4.0, 4.0];
    let reach = |points: &[(f64, f64)], join: Option<StrokeJoin>| {
        let svg = walk(&line_scene(vec![polyline(points, Some(&widths), None)], None, join));
        let rings = the_outline(&svg).1;
        // How far the outline reaches from the corner, over the vertices
        // that are not the far ends' own.
        let far = vertices(&rings).iter()
            .filter(|v| distance(**v, points[0]) > 3.0 && distance(**v, points[2]) > 3.0)
            .map(|v| distance(*v, points[1]))
            .fold(0.0, f64::max);
        (far, rings)
    };
    for join in [None, Some(StrokeJoin::Bevel)] {
        let (far, rings) = reach(&corner, join);
        assert!((far - 2.0).abs() < 1e-3, "{join:?}: a bevel stays within w/2 of the vertex, reached {far}");
        assert!(!has_vertex(&rings, (12.0, -2.0)), "{join:?}: no miter tip");
    }
    let (far, rings) = reach(&corner, Some(StrokeJoin::Miter));
    assert!(has_vertex(&rings, (12.0, -2.0)), "miter tip at the offset lines' meet");
    assert!((far - 8f64.sqrt()).abs() < 1e-3, "miter reaches w/2·√2, got {far}");
    let (far, rings) = reach(&corner, Some(StrokeJoin::Round));
    assert!((far - 2.0).abs() < 1e-3, "a round join stays on radius w/2, reached {far}");
    let h = 2.0 / 2f64.sqrt();
    assert!(has_vertex(&rings, (10.0 + h, -h)), "a round join passes through the outer bisector");
    // Past SVG's default miter limit 4 a miter bevels: a turn back of
    // about 174° would put the tip ~38 px out.
    let sharp = [(0.0, 0.0), (10.0, 0.0), (0.0, 1.0)];
    let (far, _) = reach(&sharp, Some(StrokeJoin::Miter));
    assert!(far < 2.0 + 1e-3, "a miter past limit 4 bevels, reached {far}");
    // A full reversal rounds ahead of the arriving segment.
    let reversal = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0)];
    let svg = walk(&line_scene(vec![polyline(&reversal, Some(&widths), None)], None, Some(StrokeJoin::Round)));
    assert!(has_vertex(&the_outline(&svg).1, (12.0, 0.0)), "a reversal's round join bulges along +x");
}

#[test]
fn a_malformed_tapered_node_refuses_with_a_typed_error() {
    let refusal = |node: SceneNode| match walk_svg(&line_scene(vec![node], None, None), false) {
        Err(RenderError::TaperedStrokeMalformed { source }) => source,
        other => panic!("expected a tapered-stroke refusal, got {other:?}"),
    };
    let line = [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0)];
    assert_eq!(
        refusal(polyline(&line, Some(&[1.0, 2.0]), None)),
        TaperError::Misaligned { points: 3, widths: 2 }
    );
    assert_eq!(
        refusal(polyline(&line, Some(&[1.0, 0.0, 2.0]), None)),
        TaperError::InvalidWidth { index: 1, value: 0.0 }
    );
    let straight = vec![PathCmd::MoveTo { x: 0.0, y: 0.0 }, PathCmd::LineTo { x: 10.0, y: 0.0 }];
    assert_eq!(
        refusal(path(straight.clone(), Some(&[1.0, 2.0, 3.0]))),
        TaperError::Misaligned { points: 2, widths: 3 }
    );
    let mut closed = straight.clone();
    closed.push(PathCmd::Close);
    assert_eq!(refusal(path(closed, Some(&[1.0, 2.0, 2.0]))), TaperError::Closed);
    let mut flagged = path(straight, Some(&[1.0, 2.0]));
    if let SceneNode::Path { closed, .. } = &mut flagged {
        *closed = true;
    }
    assert_eq!(refusal(flagged), TaperError::Closed);
    let err = walk_svg(&line_scene(vec![polyline(&line, Some(&[1.0, 2.0]), None)], None, None), false).unwrap_err();
    assert_eq!(
        err.to_string(),
        "the scene's tapered stroke cannot be drawn: tapered stroke has 2 widths for 3 vertices; \
         it needs one width per vertex"
    );
}

#[test]
fn tapered_dashes_alternate_along_the_stroke_with_interpolated_widths() {
    // Length 10, widths 2 → 12 (w = 2 + x), dashes [2, 3]: on at 0-2 and
    // 5-7, each a separately capped piece.
    let points = [(0.0, 0.0), (10.0, 0.0)];
    let widths = [2.0, 12.0];
    let svg = walk(&line_scene(vec![polyline(&points, Some(&widths), Some(vec![2.0, 3.0]))], None, None));
    let (el, rings) = the_outline(&svg);
    assert!(!el.contains("stroke-dasharray"), "dashes are geometry, not an attribute: {el}");
    for p in [(0.0, 1.0), (2.0, 2.0), (2.0, -2.0), (5.0, 3.5), (5.0, -3.5), (7.0, 4.5), (7.0, -4.5)] {
        assert!(has_vertex(&rings, p), "dash edge {p:?} missing");
    }
    for v in vertices(&rings) {
        assert!(!(v.0 > 2.0 + 1e-3 && v.0 < 5.0 - 1e-3) && v.0 < 7.0 + 1e-3, "off-dash vertex {v:?}");
    }
    // Each dash is capped as SVG caps dashes: square extends each by its
    // cut width / 2.
    let svg = walk(&line_scene(
        vec![polyline(&points, Some(&widths), Some(vec![2.0, 3.0]))],
        Some(StrokeCap::Square),
        None,
    ));
    let rings = the_outline(&svg).1;
    assert!(has_vertex(&rings, (4.0, 2.0)) && has_vertex(&rings, (1.5, 3.5)), "square dash caps");
    // A tapered path dashes from its own `stroke_dash`.
    let mut node = path(vec![PathCmd::MoveTo { x: 0.0, y: 0.0 }, PathCmd::LineTo { x: 10.0, y: 0.0 }], Some(&widths));
    if let SceneNode::Path { style, .. } = &mut node {
        style.stroke_dash = Some(vec![2.0, 3.0]);
    }
    let rings = the_outline(&walk(&line_scene(vec![node], None, None))).1;
    assert!(has_vertex(&rings, (5.0, 3.5)) && !vertices(&rings).iter().any(|v| v.0 > 3.0 && v.0 < 4.0));
}

#[test]
fn a_tapered_outline_fills_with_the_stroke_paint_and_no_stroke() {
    let points = [(0.0, 0.0), (10.0, 0.0)];
    let mut node = polyline(&points, Some(&[2.0, 4.0]), None);
    if let SceneNode::Polyline { style, .. } = &mut node {
        style.stroke_opacity = 0.5;
        style.color = Some(Color { r: 10, g: 20, b: 30, a: 128 });
    }
    let (el, _) = the_outline(&walk(&line_scene(vec![node], None, None)));
    assert!(el.contains(" fill=\"rgba(10,20,30,0.502)\""), "{el}");
    assert!(el.contains(" fill-opacity=\"0.5\""), "{el}");
    assert!(el.contains(" stroke=\"none\""), "{el}");
    assert!(!el.contains("stroke-width") && !el.contains("stroke-opacity"), "{el}");

    let mut node = path(vec![PathCmd::MoveTo { x: 0.0, y: 0.0 }, PathCmd::LineTo { x: 10.0, y: 0.0 }], Some(&[2.0, 4.0]));
    if let SceneNode::Path { style, .. } = &mut node {
        style.stroke_opacity = 0.25;
        style.fill = Some(Color::rgb(200, 0, 0));
        style.fill_opacity = 0.75;
    }
    let (el, _) = the_outline(&walk(&line_scene(vec![node], None, None)));
    assert!(el.contains(" fill=\"#0a141e\"") && el.contains(" fill-opacity=\"0.25\""), "the stroke paints the outline: {el}");

    // A cleared stroke paints nothing.
    let mut node = polyline(&points, Some(&[2.0, 4.0]), None);
    if let SceneNode::Polyline { style, .. } = &mut node {
        style.color = None;
    }
    let (el, _) = the_outline(&walk(&line_scene(vec![node], None, None)));
    assert!(el.contains(" fill=\"none\""), "{el}");
}

#[test]
fn every_outline_piece_winds_the_same_way_so_overlaps_paint_once() {
    // A hairpin folds back over itself: under nonzero fill, overlapping
    // pieces paint once only if none winds against another.
    let points = [(0.0, 0.0), (20.0, 0.0), (0.0, 2.0)];
    for join in [StrokeJoin::Bevel, StrokeJoin::Round, StrokeJoin::Miter] {
        for cap in [StrokeCap::Butt, StrokeCap::Round, StrokeCap::Square] {
            let svg = walk(&line_scene(vec![polyline(&points, Some(&[4.0, 4.0, 4.0]), Some(vec![5.0, 1.0]))], Some(cap), Some(join)));
            let rings = the_outline(&svg).1;
            assert!(!rings.is_empty());
            for ring in &rings {
                assert!(signed_area(ring) > 0.0, "{cap:?}/{join:?}: ring {ring:?} winds the other way");
            }
        }
    }
}

/// The pre-fix walker's bytes for an unsized polyline and path under a
/// cap/join batch wrapper.
const UNSIZED_LINE_SVG: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400\" height=\"300\" viewBox=\"0 0 400 300\"><defs><clipPath id=\"ferrum-clip-0\"><rect x=\"50\" y=\"10\" width=\"300\" height=\"250\"/></clipPath></defs><g clip-path=\"url(#ferrum-clip-0)\"><g stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"0,0 10,5\" fill=\"none\" stroke=\"#0a141e\" stroke-width=\"1.5\" stroke-dasharray=\"4,2\"/><path d=\"M0 0 C1 2 3 4 5 6\" fill=\"none\" stroke=\"#0a141e\" stroke-width=\"1.5\"/></g></g></svg>";

#[test]
fn untapered_strokes_are_byte_identical() {
    let nodes = vec![
        polyline(&[(0.0, 0.0), (10.0, 5.0)], None, Some(vec![4.0, 2.0])),
        path(vec![PathCmd::MoveTo { x: 0.0, y: 0.0 }, PathCmd::CubicTo { c1x: 1.0, c1y: 2.0, c2x: 3.0, c2y: 4.0, x: 5.0, y: 6.0 }], None),
    ];
    let svg = walk(&line_scene(nodes, Some(StrokeCap::Round), Some(StrokeJoin::Round)));
    assert_eq!(svg, UNSIZED_LINE_SVG);
    assert!(outlines(&svg).is_empty());
}

/// x/y over eight rows, two groups `g`, and the size column `v`.
fn sized_batch() -> arrow::record_batch::RecordBatch {
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("v", DataType::Float64, false),
        Field::new("g", DataType::Utf8, false),
    ]));
    arrow::record_batch::RecordBatch::try_new(schema, vec![
        Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])),
        Arc::new(Float64Array::from(vec![0.0, 3.0, 1.0, 2.0, 4.0, 6.0, 5.0, 7.0])),
        Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])),
        Arc::new(StringArray::from(vec!["a", "a", "a", "a", "b", "b", "b", "b"])),
    ]).unwrap()
}

fn line_chart(size: bool, grouped: bool, kwargs: MarkKwargsSpec) -> ChartSpec {
    let field = |f: &str| Some(EncodingSpec { field: f.into(), ..Default::default() });
    ChartSpec {
        mark: Mark::Line,
        encoding: Encoding {
            x: field("x"),
            y: field("y"),
            size: if size { field("v") } else { None },
            color: if grouped { field("g") } else { None },
            ..Default::default()
        }, mark_style: Some(kwargs),
        ..Default::default()
    }
}

fn render_chart(spec: &ChartSpec) -> String {
    crate::render::render_svg(
        spec,
        &sized_batch(),
        &crate::layout::ThemeInputs::default(),
        crate::layout::Viewport { width: 400.0, height: 300.0 },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("the chart renders")
    .bytes
}

/// Every interpolate the vocabulary publishes × {solid, dashed} ×
/// {opaque, translucent} × {one series, two}: a sized line paints one
/// outline per series and no stroked line; the unsized twin paints none.
#[test]
fn sized_lines_outline_under_every_interpolate_dash_opacity_and_grouping() {
    let mut cells = 0;
    for method in crate::render::marks::curve::interpolate_accepted_values() {
        for dash in [None, Some(vec![6.0, 3.0])] {
            for opacity in [None, Some(0.5)] {
                for grouped in [false, true] {
                    let kwargs = || MarkKwargsSpec {
                        interpolate: Some(method.clone()),
                        stroke_dash: dash.clone(),
                        opacity,
                        ..Default::default()
                    };
                    let cell = format!("{method}/{dash:?}/{opacity:?}/grouped={grouped}");
                    let series = if grouped { 2 } else { 1 };
                    let svg = render_chart(&line_chart(true, grouped, kwargs()));
                    let found = outlines(&svg);
                    assert_eq!(found.len(), series, "{cell}: one outline per series");
                    assert!(!svg.contains("<polyline"), "{cell}: no stroked polyline");
                    for (el, rings) in &found {
                        assert!(!el.contains("stroke-dasharray"), "{cell}: {el}");
                        assert_eq!(el.contains("rgba("), opacity.is_some(), "{cell}: opacity reaches the fill: {el}");
                        assert!(rings.len() >= 3, "{cell}: {} rings", rings.len());
                    }
                    // Dashing keeps only the on-pieces: under butt caps a
                    // dashed series paints the pattern's on-fraction (6 of
                    // every 9 px) of its undashed twin's area. Ring counts do
                    // not discriminate here: a gap drops whole flattened
                    // curve segments while each cut adds one.
                    if let Some(pattern) = &dash {
                        let on_fraction = pattern[0] / pattern.iter().sum::<f64>();
                        let solid = MarkKwargsSpec { stroke_dash: None, ..kwargs() };
                        let twin = outlines(&render_chart(&line_chart(true, grouped, solid)));
                        assert_eq!(twin.len(), series, "{cell}: the undashed twin");
                        for ((_, dashed), (_, undashed)) in found.iter().zip(&twin) {
                            let ratio = painted_area(dashed) / painted_area(undashed);
                            assert!(
                                (ratio - on_fraction).abs() < 0.02,
                                "{cell}: dashed paints {ratio:.3} of the undashed area, want {on_fraction:.3}"
                            );
                        }
                    }
                    // Vacuity control: the same chart without `size` paints
                    // stroked lines and no outline.
                    let control = render_chart(&line_chart(false, grouped, kwargs()));
                    assert!(outlines(&control).is_empty(), "{cell}: the unsized twin outlines nothing");
                    assert_ne!(svg, control, "{cell}");
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, crate::render::marks::curve::interpolate_accepted_values().len() * 8);
}

/// A zero-length on-dash on a tapered line draws a dot at each on-position,
/// as a constant-width SVG stroke with the same dash and cap does: a disk of
/// radius `w/2` under a round cap, a `w`-wide square turned to the line under
/// a square cap, and nothing under a butt cap. The line runs diagonally, 45
/// px long, so `[0, 10]` puts a dot at 0, 10, 20, 30 and 40 px along it.
#[test]
fn a_zero_length_dash_draws_a_dot_at_each_on_position_by_its_cap() {
    let (end, widths) = ((27.0, 36.0), [4.0, 4.0]);
    let (h, n, r) = ((0.6, 0.8), (-0.8, 0.6), 2.0);
    let dots: Vec<(f64, f64)> = (0..5).map(|k| (h.0 * 10.0 * k as f64, h.1 * 10.0 * k as f64)).collect();
    let rings = |cap: StrokeCap| {
        let node = polyline(&[(0.0, 0.0), end], Some(&widths), Some(vec![0.0, 10.0]));
        the_outline(&walk(&line_scene(vec![node], Some(cap), None))).1
    };
    let nearest = |v: (f64, f64)| dots.iter().map(|d| distance(v, *d)).fold(f64::INFINITY, f64::min);

    let round = rings(StrokeCap::Round);
    for dot in &dots {
        assert!(has_vertex(&round, (dot.0 + r * n.0, dot.1 + r * n.1)), "round: a disk about {dot:?}");
        assert!(has_vertex(&round, (dot.0 - r * n.0, dot.1 - r * n.1)), "round: a disk about {dot:?}");
    }
    assert!(vertices(&round).iter().all(|v| (nearest(*v) - r).abs() < 2e-3), "round: only disks of radius w/2");

    let square = rings(StrokeCap::Square);
    for dot in &dots {
        for (a, b) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let corner = (dot.0 + r * (a * h.0 + b * n.0), dot.1 + r * (a * h.1 + b * n.1));
            assert!(has_vertex(&square, corner), "square: corner {corner:?} of the dot at {dot:?}");
        }
    }
    assert!(
        vertices(&square).iter().all(|v| nearest(*v) < r * 2f64.sqrt() + 2e-3),
        "square: only the dots' squares"
    );

    assert!(rings(StrokeCap::Butt).is_empty(), "butt: a zero-length dash draws nothing");
}
