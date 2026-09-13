//! Annotation test coverage.
//!
//! Promoted out of an inline `#[cfg(test)] mod tests { ... }` block in
//! `annotation.rs` (this crate's Rust test-module convention: promote when
//! the test body exceeds the source module rather than on a line-count
//! threshold). At the time of promotion `annotation.rs` was 3554 lines with
//! the inline test module starting at line 1534 — the test body outsized
//! the ~1529-line source above it. Precedent: `render/scale_resolve/tests.rs`,
//! `projection/tests.rs`, `transform/stats/tests.rs`. Pure code motion: every
//! test below is unchanged from its inline form, still exercising the real
//! functions in `super::*`.

use super::*;
use crate::render::scale_resolve::ScaleKind;
use crate::scale::linear::LinearScale;
// Batch H Task 1 fix: `PathCmd` is only referenced by test assertions
// (production code now goes through `draw::arrowhead_path`, which builds
// its own `PathCmd` variants internally) — scoped here rather than at
// module scope so a module-wide `use` doesn't go stale/unused again.
use ferrum_scene::PathCmd;

// ── Per-variant wire key drift tests (#92/#148) ──────────────────────────
//
// Mines each variant's REAL accepted-field set out of its own
// `deny_unknown_fields` error (a minimal valid `type` tag plus one bogus
// key — the tag is required to select the variant, unlike
// `chart_config.rs`'s plain-struct `mined_fields_of`, which has no tag to
// supply), and asserts it equals the corresponding const minus `type`
// (the tag itself is consumed by serde before the variant's own
// `deny_unknown_fields` check runs, so it never appears in the mined set
// — see `AnnotationSpec`'s own doc comment).

fn mined_annotation_variant_fields(type_tag: &str) -> std::collections::BTreeSet<String> {
    let json = format!(r#"{{"type": "{type_tag}", "__definitely_not_a_real_field__": 1}}"#);
    let res: Result<AnnotationSpec, _> = serde_json::from_str(&json);
    let err = res.expect_err("bogus key must be rejected by deny_unknown_fields");
    super::super::chart_config::accepted_fields_from_deny_unknown_error(&err.to_string())
}

/// Shared assertion body for every `annotation_*_keys_match_serde` test
/// below — a plain function rather than a macro (no behavior worth
/// obscuring, per this crate's macro-avoidance convention).
fn assert_annotation_variant_keys_match_serde(type_tag: &str, keys_const: &[&str]) {
    let mined = mined_annotation_variant_fields(type_tag);
    let expected = keys_minus_type(keys_const);
    assert_eq!(
        mined, expected,
        "annotation `{type_tag}` variant's key const drifted from AnnotationSpec.\n\
         in serde but not const: {:?}\nin const but not serde: {:?}",
        mined.difference(&expected).collect::<Vec<_>>(),
        expected.difference(&mined).collect::<Vec<_>>(),
    );
}

fn keys_minus_type(keys: &[&str]) -> std::collections::BTreeSet<String> {
    keys.iter().filter(|&&k| k != "type").map(|s| s.to_string()).collect()
}

#[test]
fn annotation_text_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("text", ANNOTATION_TEXT_KEYS);
}
#[test]
fn annotation_arrow_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("arrow", ANNOTATION_ARROW_KEYS);
}
#[test]
fn annotation_rect_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("rect", ANNOTATION_RECT_KEYS);
}
#[test]
fn annotation_line_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("line", ANNOTATION_LINE_KEYS);
}
#[test]
fn annotation_span_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("span", ANNOTATION_SPAN_KEYS);
}
#[test]
fn annotation_bracket_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("bracket", ANNOTATION_BRACKET_KEYS);
}
#[test]
fn annotation_callout_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("callout", ANNOTATION_CALLOUT_KEYS);
}
#[test]
fn annotation_image_keys_match_serde() {
    assert_annotation_variant_keys_match_serde("image", ANNOTATION_IMAGE_KEYS);
}

/// `accepted_keys_for_annotation_variant` covers every real `AnnotationSpec`
/// tag and only those — the completeness half of closing #148's array
/// carve-out (the inverse of the pre-fix state, where every tag returned
/// no per-item key set at all because the gate never looked).
#[test]
fn accepted_keys_for_annotation_variant_covers_every_tag() {
    for tag in ["text", "arrow", "rect", "line", "span", "bracket", "callout", "image"] {
        assert!(
            accepted_keys_for_annotation_variant(tag).is_some(),
            "tag `{tag}` must have an accepted-key set"
        );
    }
    assert!(accepted_keys_for_annotation_variant("not_a_real_type").is_none());
}

/// Create a simple linear scale context for testing.
/// Domain [0, 100], x pixel range [50, 550], y pixel range [20, 320].
fn test_ctx() -> (ScaleKind, ScaleKind, Rect) {
    let plot_area = Rect {
        x: 50.0,
        y: 20.0,
        w: 500.0,
        h: 300.0,
    };
    let x_scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 100.0],
        vec![50.0, 550.0],
        false,
        false,
    ));
    let y_scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 100.0],
        vec![20.0, 320.0],
        false,
        false,
    ));
    (x_scale, y_scale, plot_area)
}

#[test]
fn resolve_data_coord_x() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    // Data value 50 should map to the center of the x pixel range.
    let px = ctx.resolve_x(&CoordValue::Data(50.0));
    assert!((px - 300.0).abs() < 1.0, "expected ~300, got {px}");
}

#[test]
fn resolve_pixel_coord() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let px = ctx.resolve_x(&CoordValue::Pixel { px: 100.0 });
    assert!((px - 150.0).abs() < f64::EPSILON, "expected 150, got {px}");
}

#[test]
fn resolve_norm_coord() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let px = ctx.resolve_x(&CoordValue::Norm { norm: 0.5 });
    assert!((px - 300.0).abs() < f64::EPSILON, "expected 300, got {px}");
}

#[test]
fn build_text_annotation() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let specs = vec![AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        text: "hello".to_string(),
        font_size: 14.0,
        color: "#ff0000".to_string(),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".to_string(),
    }];
    let ann = build_annotations(&specs, &ctx);
    // Default z="above_marks" lands in above_marks, not below_marks.
    assert!(ann.below_marks.is_empty());
    assert_eq!(ann.above_marks.len(), 1);
    match &ann.above_marks[0] {
        SceneNode::Text {
            x,
            y,
            content,
            style,
            ..
        } => {
            assert!((x - 300.0).abs() < f64::EPSILON);
            assert!((y - 170.0).abs() < f64::EPSILON);
            assert_eq!(content, "hello");
            assert_eq!(style.font_size, 14.0);
            assert_eq!(style.color, Color::rgb(255, 0, 0));
        }
        _ => panic!("expected Text node"),
    }
}

#[test]
fn build_line_annotation() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let specs = vec![AnnotationSpec::Line {
        x1: CoordValue::Norm { norm: 0.0 },
        y1: CoordValue::Norm { norm: 0.5 },
        x2: CoordValue::Norm { norm: 1.0 },
        y2: CoordValue::Norm { norm: 0.5 },
        stroke: "#000000".to_string(),
        stroke_width: 2.0,
        dash: Some(vec![4.0, 2.0]),
    }];
    let ann = build_annotations(&specs, &ctx);
    // Non-Text specs always land in above_marks.
    assert!(ann.below_marks.is_empty());
    assert_eq!(ann.above_marks.len(), 1);
    match &ann.above_marks[0] {
        SceneNode::Line {
            x1,
            y1,
            x2,
            y2,
            style,
        } => {
            assert!((x1 - 50.0).abs() < f64::EPSILON);
            assert!((x2 - 550.0).abs() < f64::EPSILON);
            assert!((y1 - 170.0).abs() < f64::EPSILON);
            assert!((y2 - 170.0).abs() < f64::EPSILON);
            assert_eq!(style.width, 2.0);
            assert_eq!(style.dash.as_deref(), Some(&[4.0, 2.0][..]));
        }
        _ => panic!("expected Line node"),
    }
}

#[test]
fn build_arrow_annotation() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    // `curve` is no longer a field on AnnotationSpec::Arrow (dropped as dead code).
    let specs = vec![AnnotationSpec::Arrow {
        x: CoordValue::Norm { norm: 0.0 },
        y: CoordValue::Norm { norm: 0.0 },
        x2: CoordValue::Norm { norm: 1.0 },
        y2: CoordValue::Norm { norm: 1.0 },
        stroke: "#333333".to_string(),
        stroke_width: 1.5,
        head_size: 8.0,
    }];
    let ann = build_annotations(&specs, &ctx);
    // Arrow (non-Text) always lands in above_marks.
    assert!(ann.below_marks.is_empty());
    // Arrow produces a line + arrowhead path.
    assert_eq!(ann.above_marks.len(), 2);
    assert!(matches!(&ann.above_marks[0], SceneNode::Line { .. }));
    assert!(matches!(&ann.above_marks[1], SceneNode::Path { .. }));
}

#[test]
fn build_span_annotation() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let specs = vec![AnnotationSpec::Span {
        axis: "x".to_string(),
        start: CoordValue::Norm { norm: 0.2 },
        end: CoordValue::Norm { norm: 0.8 },
        fill: "#aaaaaa".to_string(),
        opacity: 0.3,
        label_position: "center".to_string(),
        label: Some("span".to_string()),
    }];
    let ann = build_annotations(&specs, &ctx);
    // Non-Text specs always land in above_marks.
    assert!(ann.below_marks.is_empty());
    // Span produces a rect + label text.
    assert_eq!(ann.above_marks.len(), 2);
    assert!(matches!(&ann.above_marks[0], SceneNode::Rect { .. }));
    assert!(matches!(&ann.above_marks[1], SceneNode::Text { .. }));
}

#[test]
fn deser_annotation_spec_text() {
    let json = r#"{"type": "text", "x": 50.0, "y": {"norm": 0.5}, "text": "hi"}"#;
    let spec: AnnotationSpec = serde_json::from_str(json).unwrap();
    assert!(matches!(spec, AnnotationSpec::Text { .. }));
}

#[test]
fn deser_annotation_spec_line() {
    let json = r##"{"type": "line", "x1": {"px": 0}, "y1": {"norm": 0.5}, "x2": {"px": 100}, "y2": {"norm": 0.5}, "stroke": "#000"}"##;
    let spec: AnnotationSpec = serde_json::from_str(json).unwrap();
    assert!(matches!(spec, AnnotationSpec::Line { .. }));
}

#[test]
fn color_fallback_on_invalid() {
    let c = resolve_color("not-a-valid-color-xyz");
    assert_eq!(c, Color::rgb(51, 51, 51));
}

#[test]
fn empty_specs_returns_empty_vecs() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let ann = build_annotations(&[], &ctx);
    assert!(ann.below_marks.is_empty());
    assert!(ann.above_marks.is_empty());
}

// ── Regression tests for z-routing (T2.1b / XDEAD-03) ───────────────────
//
// These tests must fail on the pre-wiring code where build_annotations
// returned a flat Vec<SceneNode> with no z-based partition.

/// A Text spec with z="below_marks" routes to `below_marks`; a Text spec
/// with z="above_marks" (or default) routes to `above_marks`.
#[test]
fn text_z_routing_below_and_above() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };

    let mk_text = |z: &str| AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        text: z.to_string(),
        font_size: 12.0,
        color: "#000000".to_string(),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: z.to_string(),
    };

    let specs = vec![mk_text("below_marks"), mk_text("above_marks")];
    let ann = build_annotations(&specs, &ctx);

    // Exactly one node in each bucket.
    assert_eq!(
        ann.below_marks.len(),
        1,
        "below_marks bucket must have the below_marks text"
    );
    assert_eq!(
        ann.above_marks.len(),
        1,
        "above_marks bucket must have the above_marks text"
    );

    // Confirm the content so we know which went where.
    match &ann.below_marks[0] {
        SceneNode::Text { content, .. } => assert_eq!(content, "below_marks"),
        _ => panic!("expected Text in below_marks"),
    }
    match &ann.above_marks[0] {
        SceneNode::Text { content, .. } => assert_eq!(content, "above_marks"),
        _ => panic!("expected Text in above_marks"),
    }
}

/// An unknown/unrecognized z value falls through to the above-marks bucket
/// (not below_marks).  Only the exact string "below_marks" selects the
/// below bucket; everything else is above (fail-safe default).
#[test]
fn text_unknown_z_falls_to_above_marks() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };

    let specs = vec![AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        text: "unknown".to_string(),
        font_size: 12.0,
        color: "#000000".to_string(),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "front".to_string(), // old default — must still go above
    }];
    let ann = build_annotations(&specs, &ctx);
    assert!(
        ann.below_marks.is_empty(),
        "unrecognized z must not go to below_marks"
    );
    assert_eq!(ann.above_marks.len(), 1);
}

/// Non-Text specs (Arrow, Line, Rect, Span, Bracket, Callout, Image) have
/// no z field and always land in above_marks.
#[test]
fn non_text_specs_always_above_marks() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };

    let specs = vec![
        AnnotationSpec::Line {
            x1: CoordValue::Norm { norm: 0.0 },
            y1: CoordValue::Norm { norm: 0.5 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 0.5 },
            stroke: "#000000".to_string(),
            stroke_width: 1.0,
            dash: None,
        },
        AnnotationSpec::Arrow {
            x: CoordValue::Norm { norm: 0.0 },
            y: CoordValue::Norm { norm: 0.0 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 1.0 },
            stroke: "#000000".to_string(),
            stroke_width: 1.5,
            head_size: 0.0, // no arrowhead → 1 node only
        },
    ];
    let ann = build_annotations(&specs, &ctx);
    assert!(
        ann.below_marks.is_empty(),
        "non-Text annotations must never go to below_marks"
    );
    // Line = 1 node; Arrow with head_size=0 = 1 node (no path triangle).
    assert_eq!(ann.above_marks.len(), 2);
}

/// Image annotation Raw nodes must carry `anchor == Data` — they are
/// positioned via the panel's coordinate system (data/pixel/norm → plot-area
/// pixels) and must track the canvas transform during pan/zoom.
#[test]
fn image_annotation_raw_node_has_data_anchor() {
    use ferrum_scene::RawAnchor;
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let specs = vec![AnnotationSpec::Image {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        src: "https://example.com/logo.png".to_string(),
        width: 40.0,
        height: 30.0,
        anchor: "middle".to_string(),
    }];
    let ann = build_annotations(&specs, &ctx);
    assert!(
        ann.below_marks.is_empty(),
        "image annotation should not go to below_marks"
    );
    assert_eq!(
        ann.above_marks.len(),
        1,
        "expected exactly one node for image annotation"
    );
    match &ann.above_marks[0] {
        SceneNode::Raw { anchor, .. } => {
            assert_eq!(
                *anchor,
                RawAnchor::Data,
                "image annotation Raw node must have Data anchor"
            );
        }
        _ => panic!("expected SceneNode::Raw for image annotation"),
    }
}

/// R1 port (bug_hunt_draw.rs / bug_hunt_render_pipeline.rs): `parse_baseline`
/// now routes through the single canonical `draw::parse_text_baseline`
/// vocabulary instead of an independently hand-rolled match, so the
/// previously-divergent aliases (`hanging`, `text-before-edge`,
/// `ideographic`) resolve the same way here as they do for mark text.
#[test]
fn parse_baseline_delegates_to_canonical_vocabulary() {
    assert_eq!(parse_baseline("hanging"), TextBaseline::Top);
    assert_eq!(parse_baseline("text-before-edge"), TextBaseline::Top);
    assert_eq!(parse_baseline("ideographic"), TextBaseline::Bottom);
    assert_eq!(parse_baseline("unknown-value"), TextBaseline::Middle);
}

/// R1 port: `emit_span` skips emission entirely when either resolved
/// coordinate is non-finite (e.g. an out-of-domain scale mapping), rather
/// than emitting a rect with a NaN/zero-width dimension.
#[test]
fn emit_span_skips_on_non_finite_coordinate() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let mut out = Vec::new();
    emit_span(
        "x",
        &CoordValue::Pixel { px: f64::NAN },
        &CoordValue::Data(50.0),
        "#ff0000",
        0.3,
        None,
        "middle",
        &ctx,
        &mut out,
    );
    assert!(
        out.is_empty(),
        "non-finite span coordinate must skip emission entirely"
    );

    // A fully finite span does emit.
    let mut out2 = Vec::new();
    emit_span(
        "x",
        &CoordValue::Data(10.0),
        &CoordValue::Data(50.0),
        "#ff0000",
        0.3,
        None,
        "middle",
        &ctx,
        &mut out2,
    );
    assert_eq!(out2.len(), 1, "finite span must emit exactly one rect");
}

/// R1 port: `emit_arrow` clamps `head_size` to the shaft length so an
/// epsilon-length arrow does not produce an arrowhead triangle that
/// vastly overshoots the line's start point.
#[test]
fn emit_arrow_clamps_head_size_to_epsilon_shaft() {
    let mut out = Vec::new();
    emit_arrow(100.0, 100.0, 100.001, 100.0, "#000000", 1.0, 8.0, &mut out);
    assert_eq!(out.len(), 2, "expected shaft line + arrowhead path");
    let SceneNode::Path { commands, .. } = &out[1] else {
        panic!("expected arrowhead Path node");
    };
    // The arrowhead's base points must not extend past the shaft's start (x1=100.0).
    for cmd in commands {
        if let ferrum_scene::PathCmd::LineTo { x, .. } = cmd {
            assert!(
                *x >= 100.0 - 1e-9,
                "clamped arrowhead must not extend past the shaft start; got x={x}"
            );
        }
    }
}

/// Batch H Task 1: `emit_arrow`'s arrowhead Path must carry EXACTLY the
/// vertices `draw::arrowhead_triangle` computes for the same
/// endpoints/head_size — proving the annotation path and
/// `marks::segment::build`'s new `arrow=true` path (segment.rs) share one
/// geometry function rather than two independently-typed copies that
/// could silently drift. Fails to compile against the pre-extraction
/// surface: `draw::arrowhead_triangle` did not exist before this task.
#[test]
fn emit_arrow_triangle_matches_shared_arrowhead_triangle_helper() {
    let (x1, y1, x2, y2, head_size) = (10.0, 20.0, 130.0, 65.0, 8.0);
    let mut out = Vec::new();
    emit_arrow(x1, y1, x2, y2, "#123456", 1.0, head_size, &mut out);
    let SceneNode::Path { commands, .. } = &out[1] else {
        panic!("expected arrowhead Path node");
    };
    let emitted: Vec<(f64, f64)> = commands
        .iter()
        .filter_map(|c| match c {
            PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => Some((*x, *y)),
            _ => None,
        })
        .collect();

    let expected = arrowhead_triangle(x1, y1, x2, y2, head_size)
        .expect("non-degenerate segment must yield a triangle");

    assert_eq!(
        emitted,
        vec![expected[0], expected[1], expected[2]],
        "emit_arrow's triangle vertices must equal arrowhead_triangle's output exactly"
    );
}

/// R1 port: an unrecognized bracket `direction` defaults to "up" (matching
/// the fallthrough arm), rather than silently producing a zero-length tip.
#[test]
fn emit_bracket_unknown_direction_defaults_to_up() {
    let mut out_up = Vec::new();
    emit_bracket(
        0.0,
        100.0,
        100.0,
        100.0,
        "",
        "up",
        "#000000",
        6.0,
        &mut out_up,
    );
    let mut out_unknown = Vec::new();
    emit_bracket(
        0.0,
        100.0,
        100.0,
        100.0,
        "",
        "diagonal",
        "#000000",
        6.0,
        &mut out_unknown,
    );
    assert_eq!(out_up.len(), out_unknown.len());
    // Both must produce identical tip geometry (same Line endpoints).
    for (a, b) in out_up.iter().zip(out_unknown.iter()) {
        match (a, b) {
            (
                SceneNode::Line {
                    x1: ax1,
                    y1: ay1,
                    x2: ax2,
                    y2: ay2,
                    ..
                },
                SceneNode::Line {
                    x1: bx1,
                    y1: by1,
                    x2: bx2,
                    y2: by2,
                    ..
                },
            ) => {
                assert_eq!((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2));
            }
            _ => panic!("expected Line nodes for both directions"),
        }
    }
}

/// R1 port: the callout background box width uses Unicode character count
/// (`chars().count()`), not UTF-8 byte length, so multi-byte labels are not
/// over-estimated.
#[test]
fn emit_callout_width_uses_char_count_not_byte_len() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let ascii = "resume"; // 6 chars, 6 bytes
    let unicode = "r\u{00e9}sum\u{00e9}"; // 6 chars, 8 bytes
    assert_eq!(ascii.chars().count(), unicode.chars().count());

    let build = |text: &str| {
        let mut out = Vec::new();
        emit_callout(
            &ctx,
            &CoordValue::Data(50.0),
            &CoordValue::Data(50.0),
            text,
            "curved",
            4.0,
            "#fff",
            "#333",
            0.0,
            None,
            None,
            &mut out,
        );
        let SceneNode::Rect { w, .. } = out[0] else {
            panic!("expected background Rect")
        };
        w
    };
    assert_eq!(
        build(ascii),
        build(unicode),
        "equal char count must produce equal box width"
    );
}

/// R1 port: `arrow == "none"` suppresses the callout's leader line; any
/// other value (including an empty string) still draws it.
#[test]
fn emit_callout_arrow_none_suppresses_leader_line() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let count_lines = |arrow: &str| {
        let mut out = Vec::new();
        emit_callout(
            &ctx,
            &CoordValue::Data(50.0),
            &CoordValue::Data(50.0),
            "hi",
            arrow,
            4.0,
            "#fff",
            "#333",
            0.0,
            None,
            None,
            &mut out,
        );
        out.iter()
            .filter(|n| matches!(n, SceneNode::Line { .. }))
            .count()
    };
    assert_eq!(
        count_lines("none"),
        0,
        "arrow='none' must suppress the leader line"
    );
    assert_eq!(
        count_lines(""),
        1,
        "empty string is not 'none' — leader line still drawn"
    );
    assert_eq!(count_lines("curved"), 1);
}

// ── kind_name / data_coords_on_axis (spec §4 H5, F-L08-07) ──────────────

#[test]
fn kind_name_covers_every_shape() {
    let cases: Vec<(AnnotationSpec, &str)> = vec![
        (
            AnnotationSpec::Text {
                x: CoordValue::Data(1.0),
                y: CoordValue::Data(2.0),
                text: "t".into(),
                font_size: 12.0,
                color: "#000".into(),
                anchor: "middle".into(),
                baseline: "middle".into(),
                angle: 0.0,
                dx: 0.0,
                dy: 0.0,
                z: "above_marks".into(),
            },
            "text",
        ),
        (
            AnnotationSpec::Arrow {
                x: CoordValue::Data(1.0),
                y: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                stroke: "#000".into(),
                stroke_width: 1.5,
                head_size: 8.0,
            },
            "arrow",
        ),
        (
            AnnotationSpec::Rect {
                x1: CoordValue::Data(1.0),
                y1: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                fill: "#ccc".into(),
                opacity: 0.3,
                corner_radius: 0.0,
                stroke: None,
            },
            "rect",
        ),
        (
            AnnotationSpec::Line {
                x1: CoordValue::Data(1.0),
                y1: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                stroke: "#000".into(),
                stroke_width: 1.5,
                dash: None,
            },
            "line",
        ),
        (
            AnnotationSpec::Span {
                axis: "x".into(),
                start: CoordValue::Data(1.0),
                end: CoordValue::Data(2.0),
                fill: "#ccc".into(),
                opacity: 0.2,
                label_position: "center".into(),
                label: None,
            },
            "span",
        ),
        (
            AnnotationSpec::Bracket {
                x1: CoordValue::Data(1.0),
                y1: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                label: "".into(),
                direction: "up".into(),
                stroke: "#000".into(),
                tip_length: 6.0,
            },
            "bracket",
        ),
        (
            AnnotationSpec::Callout {
                x: CoordValue::Data(1.0),
                y: CoordValue::Data(2.0),
                text: "c".into(),
                arrow: "curved".into(),
                padding: 4.0,
                background: "#fff".into(),
                border_color: "#333".into(),
                border_radius: 0.0,
                text_x: None,
                text_y: None,
            },
            "callout",
        ),
        (
            AnnotationSpec::Image {
                x: CoordValue::Data(1.0),
                y: CoordValue::Data(2.0),
                src: "s".into(),
                width: 10.0,
                height: 10.0,
                anchor: "middle".into(),
            },
            "image",
        ),
    ];
    for (spec, expected) in cases {
        assert_eq!(kind_name(&spec), expected);
    }
}

#[test]
fn data_coords_on_axis_text() {
    let spec = AnnotationSpec::Text {
        x: CoordValue::Data(10.0),
        y: CoordValue::Data(20.0),
        text: "t".into(),
        font_size: 12.0,
        color: "#000".into(),
        anchor: "middle".into(),
        baseline: "middle".into(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".into(),
    };
    assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Points(vec![10.0]));
    assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Points(vec![20.0]));
}

#[test]
fn data_coords_on_axis_excludes_px_and_norm() {
    let spec = AnnotationSpec::Text {
        x: CoordValue::Pixel { px: 5.0 },
        y: CoordValue::Norm { norm: 0.5 },
        text: "t".into(),
        font_size: 12.0,
        color: "#000".into(),
        anchor: "middle".into(),
        baseline: "middle".into(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".into(),
    };
    assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Points(Vec::new()));
    assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Points(Vec::new()));
}

#[test]
fn data_coords_on_axis_arrow_both_endpoints() {
    let spec = AnnotationSpec::Arrow {
        x: CoordValue::Data(1.0),
        y: CoordValue::Data(2.0),
        x2: CoordValue::Data(3.0),
        y2: CoordValue::Data(4.0),
        stroke: "#000".into(),
        stroke_width: 1.5,
        head_size: 8.0,
    };
    assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Extent(1.0, 3.0));
    assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Extent(2.0, 4.0));
}

#[test]
fn data_coords_on_axis_rect_line_bracket_share_shape() {
    let specs = [
        AnnotationSpec::Rect {
            x1: CoordValue::Data(1.0),
            y1: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            fill: "#ccc".into(),
            opacity: 0.3,
            corner_radius: 0.0,
            stroke: None,
        },
        AnnotationSpec::Line {
            x1: CoordValue::Data(1.0),
            y1: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            stroke: "#000".into(),
            stroke_width: 1.5,
            dash: None,
        },
        AnnotationSpec::Bracket {
            x1: CoordValue::Data(1.0),
            y1: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            label: "".into(),
            direction: "up".into(),
            stroke: "#000".into(),
            tip_length: 6.0,
        },
    ];
    for spec in &specs {
        assert_eq!(data_coords_on_axis(spec, "x"), AxisCoords::Extent(1.0, 3.0));
        assert_eq!(data_coords_on_axis(spec, "y"), AxisCoords::Extent(2.0, 4.0));
    }
}

#[test]
fn data_coords_on_axis_span_gated_by_its_own_axis_field() {
    let x_span = AnnotationSpec::Span {
        axis: "x".into(),
        start: CoordValue::Data(1.0),
        end: CoordValue::Data(2.0),
        fill: "#ccc".into(),
        opacity: 0.2,
        label_position: "center".into(),
        label: None,
    };
    assert_eq!(data_coords_on_axis(&x_span, "x"), AxisCoords::Extent(1.0, 2.0));
    assert_eq!(data_coords_on_axis(&x_span, "y"), AxisCoords::Points(Vec::new()));

    let y_span = AnnotationSpec::Span {
        axis: "y".into(),
        start: CoordValue::Data(5.0),
        end: CoordValue::Data(6.0),
        fill: "#ccc".into(),
        opacity: 0.2,
        label_position: "center".into(),
        label: None,
    };
    assert_eq!(data_coords_on_axis(&y_span, "y"), AxisCoords::Extent(5.0, 6.0));
    assert_eq!(data_coords_on_axis(&y_span, "x"), AxisCoords::Points(Vec::new()));
}

#[test]
fn data_coords_on_axis_callout_includes_optional_text_position() {
    let with_text_pos = AnnotationSpec::Callout {
        x: CoordValue::Data(1.0),
        y: CoordValue::Data(2.0),
        text: "c".into(),
        arrow: "curved".into(),
        padding: 4.0,
        background: "#fff".into(),
        border_color: "#333".into(),
        border_radius: 0.0,
        text_x: Some(CoordValue::Data(11.0)),
        text_y: Some(CoordValue::Data(12.0)),
    };
    assert_eq!(
        data_coords_on_axis(&with_text_pos, "x"),
        AxisCoords::Points(vec![1.0, 11.0]),
        "a callout's anchor and text anchor are two points, never one interval"
    );
    assert_eq!(
        data_coords_on_axis(&with_text_pos, "y"),
        AxisCoords::Points(vec![2.0, 12.0])
    );

    // text_x/text_y omitted: only the anchor coordinate is reported.
    let no_text_pos = AnnotationSpec::Callout {
        x: CoordValue::Data(1.0),
        y: CoordValue::Data(2.0),
        text: "c".into(),
        arrow: "curved".into(),
        padding: 4.0,
        background: "#fff".into(),
        border_color: "#333".into(),
        border_radius: 0.0,
        text_x: None,
        text_y: None,
    };
    assert_eq!(data_coords_on_axis(&no_text_pos, "x"), AxisCoords::Points(vec![1.0]));
    assert_eq!(data_coords_on_axis(&no_text_pos, "y"), AxisCoords::Points(vec![2.0]));
}

#[test]
fn data_coords_on_axis_image_single_anchor() {
    let spec = AnnotationSpec::Image {
        x: CoordValue::Data(7.0),
        y: CoordValue::Data(8.0),
        src: "s".into(),
        width: 10.0,
        height: 10.0,
        anchor: "middle".into(),
    };
    assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Points(vec![7.0]));
    assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Points(vec![8.0]));
}

// ── break-aware resolve_x/resolve_y (spec §4 H5, F-L08-07) ──────────────

#[test]
fn resolve_x_data_coord_routes_through_break_when_present() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let result = crate::render::break_axis::apply_break_to_scale(
        (0.0, 100.0),
        &[[40.0, 60.0]],
        (50.0, 550.0),
        10.0,
    );
    let remap = crate::render::break_axis::BreakRemap::new(
        "x",
        (0.0, 100.0),
        (50.0, 550.0),
        &result,
    );
    let breaks = [remap];
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &breaks,
    };
    let broken_px = ctx.resolve_x(&CoordValue::Data(20.0));
    let expected = crate::render::break_axis::broken_scale_map(20.0, &result).unwrap();
    assert!((broken_px - expected).abs() < f64::EPSILON);

    // Without the break, the same value maps to the plain linear
    // position — proving the break actually moved something.
    let unbroken_ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    let unbroken_px = unbroken_ctx.resolve_x(&CoordValue::Data(20.0));
    assert!(
        (broken_px - unbroken_px).abs() > 1.0,
        "the break must move the compressed position away from the unbroken one"
    );
}

// `resolve_x_folds_through_multiple_same_axis_breaks` and
// `resolve_x_folds_same_axis_breaks_in_declaration_order` (quality-review
// cycle-2 finding #2) used to live here, each hand-building a
// `ScaleContext` whose `breaks` list held TWO same-axis `BreakRemap`s to
// pin that `fold_breaks` composes every same-axis break in declaration
// order rather than taking only the first.
//
// #230-4's same-axis merge (`scene_build::merge_break_specs`) makes
// that construction unreachable from production: N same-axis `BreakAxis`
// specs now merge into ONE `BreakResult` — one gap list, run through
// `normalize_gaps` once — before `break_remaps` is ever built, so
// `ScaleContext::breaks` carries at most one remap per axis. Composing
// two independently-built same-axis `BreakRemap`s (each resolved against
// the ORIGINAL, unbroken domain) was itself the numerically-wrong
// behavior #230-4 replaces — sequential composition re-interprets the
// first break's already-broken pixel as an unbroken-domain data value,
// and (via `BreakRemap::data_at`'s domain clamp) could resurrect a
// coordinate the first break had hidden. Keeping these two tests green
// would pin that replaced behavior as if it were still a supported
// shape, so they are retired rather than kept passing.
//
// `fold_breaks` itself is unchanged and stays exercised: with the merge
// upstream, its loop now always executes 0 or 1 times per axis, and the
// cross-axis case (independent x and y folds) is what
// `resolve_y_pixel_and_norm_coords_are_unaffected_by_a_break` below and
// `test_two_same_axis_breaks_fold_annotations_and_marks_to_the_same_position`
// (`tests/test_break_axis_geometry.py`) pin end to end. The same-axis
// merge and the `BREAK_HIDDEN` poison guard are pinned directly in
// `scene_build.rs` and `break_axis.rs`, closer to where they are now
// implemented.

#[test]
fn resolve_y_pixel_and_norm_coords_are_unaffected_by_a_break() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let result = crate::render::break_axis::apply_break_to_scale(
        (0.0, 100.0),
        &[[40.0, 60.0]],
        (20.0, 320.0),
        10.0,
    );
    let remap = crate::render::break_axis::BreakRemap::new(
        "y",
        (0.0, 100.0),
        (20.0, 320.0),
        &result,
    );
    let broken_ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[remap],
    };
    let plain_ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };
    // Plot-area-relative coordinates are identical with or without an
    // active break on this axis — only `CoordValue::Data` routes through
    // the piecewise mapping.
    assert_eq!(
        broken_ctx.resolve_y(&CoordValue::Pixel { px: 42.0 }),
        plain_ctx.resolve_y(&CoordValue::Pixel { px: 42.0 }),
    );
    assert_eq!(
        broken_ctx.resolve_y(&CoordValue::Norm { norm: 0.25 }),
        plain_ctx.resolve_y(&CoordValue::Norm { norm: 0.25 }),
    );
}

/// quality-review cycle-3: `resolve_y`'s out-of-domain fallback used
/// `frac` directly (no flip), which is backwards relative to how every
/// non-ordinal y axis actually maps its domain onto pixels
/// (`scale_resolve::positional::axis_pixel_range` flips domain[0] to the
/// BOTTOM pixel and domain[1] to the TOP, unconditionally). This test
/// builds a `y_scale` with that SAME flipped range (`[320.0, 20.0]` —
/// domain min -> bottom, domain max -> top), the shape a real chart's y
/// scale actually has, rather than `test_ctx`'s simplified direct range —
/// so it exercises the fallback against the real invariant it depends on,
/// not just the formula in isolation.
#[test]
fn resolve_y_out_of_domain_value_clamps_to_the_correct_side() {
    use crate::scale::linear::LinearScale;

    let (x_scale, _, plot_area) = test_ctx();
    let y_scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 100.0],
        vec![320.0, 20.0],
        false,
        false,
    ));
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
    };

    // Domain is [0, 100]; both values are out of domain, so `to_pixel_f64`
    // returns `None` and the fallback fires (verified separately below).
    let above_max = ctx.resolve_y(&CoordValue::Data(150.0));
    let below_min = ctx.resolve_y(&CoordValue::Data(-50.0));

    assert!(
        y_scale.to_pixel_f64(150.0).is_none() && y_scale.to_pixel_f64(-50.0).is_none(),
        "this test must exercise the out-of-domain fallback, not to_pixel_f64's happy path"
    );

    // A value ABOVE the domain max must land on the TOP side (smaller
    // pixel) of a value BELOW the domain min (BOTTOM side, larger
    // pixel) — matching the real y scale's domain-max-at-top convention.
    // Pre-fix this assertion failed: above_max (470) was LARGER than
    // below_min (-130), the sides swapped.
    assert!(
        above_max < below_min,
        "above_max={above_max} must be nearer the top (smaller) than \
         below_min={below_min}"
    );
    // Exact values, hand-computed from the flipped formula:
    // plot_area.y=20, h=300. above_max: frac=(150-0)/100=1.5 ->
    // 20 + (1-1.5)*300 = -130. below_min: frac=(-50-0)/100=-0.5 ->
    // 20 + (1-(-0.5))*300 = 470.
    assert!((above_max - (-130.0)).abs() < 1e-6, "got {above_max}");
    assert!((below_min - 470.0).abs() < 1e-6, "got {below_min}");
}

// ── Break routing (#228 stratified clip, D1) ─────────────────────────────
//
// These pins moved here from `scene_build.rs`, where they tested the
// `suppress_annotations_for_break` filter this routing replaces. Three of
// them were REWRITTEN rather than moved, because #228 retires their old
// answers: an extent that spans a gap is no longer dropped (it clips), N
// spanning rects no longer roll up as suppressed (nothing is suppressed),
// and an extent with an in-gap ENDPOINT is no longer dropped whole (it
// clips to whatever survives). The suppression they pinned is now the
// narrower fact it always should have been: nothing survives, or the kind
// admits no honest split.
//
// All geometry below is hand-checkable against one fixture: domain
// [0, 1000] on y pixels 320 (bottom) → 20 (top), 300 pixels total, 12 of
// them reserved per break indicator.

/// A spec from its wire JSON — shorter than an 8-field literal, and it
/// exercises the same `serde` defaults a real chart's annotations arrive
/// with.
fn spec(json: &str) -> AnnotationSpec {
    serde_json::from_str(json).expect("test annotation spec must deserialize")
}

/// A y-`Rect` spec with the given data bounds; every other coordinate is
/// norm, so only the y interval is under test.
fn y_rect(y1: f64, y2: f64) -> AnnotationSpec {
    spec(&format!(
        r#"{{"type":"rect","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2}}}"#
    ))
}

/// Like [`y_rect`], with an explicit `corner_radius` — for #230-2
/// coverage of `annotate_rect(corner_radius=...)` straddling a gap (T7
/// routed `rect`/`span` annotations through the same clip machinery as
/// marks, so they inherited the mark-side defect; this is the annotation
/// half of that fix's regression coverage).
fn y_rect_with_radius(y1: f64, y2: f64, corner_radius: f64) -> AnnotationSpec {
    spec(&format!(
        r#"{{"type":"rect","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2},"corner_radius":{corner_radius}}}"#
    ))
}

/// A y-`Bracket` spec with the given data bounds — the kind that is
/// PERMANENTLY suppress-whole.
fn y_bracket(y1: f64, y2: f64) -> AnnotationSpec {
    spec(&format!(
        r#"{{"type":"bracket","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2},"label":"b"}}"#
    ))
}

/// A `Text` spec anchored at a y data value; its x is norm, so an x break
/// never reaches it.
fn y_text(y: f64, text: &str) -> AnnotationSpec {
    spec(&format!(
        r#"{{"type":"text","x":{{"norm":1.0}},"y":{y},"text":"{text}"}}"#
    ))
}

/// The broken panel every routing test shares: domain [0, 1000] on both
/// axes, y mapped 320 (bottom) → 20 (top) over 300 pixels and x 50 → 550
/// over 500, each broken by `gaps` with a 12px indicator.
///
/// One `BreakResult` per axis, because a result's segments carry PIXEL
/// bounds: sharing one across both axes would place x geometry on the y
/// axis's pixels.
struct BrokenPanel {
    x_scale: ScaleKind,
    y_scale: ScaleKind,
    plot_area: Rect,
    x_result: break_axis::BreakResult,
    y_result: break_axis::BreakResult,
}

const X_PIXELS: (f64, f64) = (50.0, 550.0);
const Y_PIXELS: (f64, f64) = (320.0, 20.0);

fn broken_panel(gaps: &[[f64; 2]]) -> BrokenPanel {
    use crate::scale::linear::LinearScale;
    BrokenPanel {
        x_scale: ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 1000.0],
            vec![X_PIXELS.0, X_PIXELS.1],
            false,
            false,
        )),
        y_scale: ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 1000.0],
            vec![Y_PIXELS.0, Y_PIXELS.1],
            false,
            false,
        )),
        plot_area: Rect { x: 50.0, y: 20.0, w: 500.0, h: 300.0 },
        x_result: break_axis::apply_break_to_scale((0.0, 1000.0), gaps, X_PIXELS, 12.0),
        y_result: break_axis::apply_break_to_scale((0.0, 1000.0), gaps, Y_PIXELS, 12.0),
    }
}

impl BrokenPanel {
    /// The remap for one axis, built exactly as `scene_build` builds it:
    /// the scale's own domain and pixel range.
    fn remap(&self, axis: &'static str) -> break_axis::BreakRemap<'_> {
        let (pixels, result) = if axis == "y" {
            (Y_PIXELS, &self.y_result)
        } else {
            (X_PIXELS, &self.x_result)
        };
        break_axis::BreakRemap::new(axis, (0.0, 1000.0), pixels, result)
    }

    /// Build `specs` through `breaks`, returning the above-marks nodes and
    /// the warnings pushed.
    fn build(
        &self,
        specs: &[AnnotationSpec],
        breaks: &[break_axis::BreakRemap<'_>],
    ) -> (Vec<SceneNode>, Vec<RenderWarning>) {
        let ctx = ScaleContext {
            plot_area: self.plot_area,
            x_scale: &self.x_scale,
            y_scale: &self.y_scale,
            breaks,
        };
        let mut warnings = Vec::new();
        let nodes = build_annotations_through_breaks(specs, &ctx, &mut warnings);
        assert!(
            nodes.below_marks.is_empty(),
            "these fixtures use above-marks specs only"
        );
        (nodes.above_marks, warnings)
    }
}

/// `(y, h)` of every `Rect` node, in paint order — how an extent's clipped
/// pieces are read.
fn rect_bands(nodes: &[SceneNode]) -> Vec<(f64, f64)> {
    nodes.iter().filter_map(rect_extent).map(|(_, y, _, h)| (y, h)).collect()
}

/// The intent-ledger headline: `annotate_rect(y1=0, y2=1000)` on a
/// `[0, 1000]` axis broken at `(150, 850)` renders TWO pieces, silently.
///
/// Pre-#228 the same rect was dropped whole with an
/// `AnnotationSpansBreakGap` warning, while identical MARK geometry (a bar
/// of the same extent) clipped per segment — the same chart answering the
/// same question two ways.
#[test]
fn a_straddling_rect_clips_into_one_piece_per_visible_segment() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_rect(0.0, 1000.0)], &[y]);

    // 288 data pixels split 150:150 → 144 each. Segment 0 (data 0–150)
    // runs 320 → 176; the indicator takes 176 → 164; segment 1 (data
    // 850–1000) runs 164 → 20.
    assert_eq!(
        rect_bands(&nodes),
        vec![(176.0, 144.0), (20.0, 144.0)],
        "one piece per visible segment, cut at the gap edges"
    );
    assert!(
        warnings.is_empty(),
        "a successful clip is SILENT, like a clipped mark; got {warnings:?}"
    );
}

/// #230-2, the annotation half: `annotate_rect(corner_radius=8)`
/// straddling a gap must square the cut edge on each piece, not carry
/// `corner_radius` onto it verbatim. `annotate_rect` clips through the
/// exact same [`break_axis::BreakRemap::clip_nodes`] a mark batch does
/// (see [`a_straddling_rect_clips_into_one_piece_per_visible_segment`]),
/// so a fix scoped only to the mark-facing call site would leave this
/// broken — this proves the annotation path gets it too.
#[test]
fn a_straddling_rounded_rect_squares_only_its_cut_edge() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_rect_with_radius(0.0, 1000.0, 8.0)], &[y]);

    assert_eq!(nodes.len(), 2, "same two pieces as the unrounded case; got {nodes:?}");
    assert!(warnings.is_empty(), "a successful clip is SILENT; got {warnings:?}");

    // Piece 0 (pixel 176..320, data 0..150): y=320 is the real baseline
    // (data 0) — uncut, rounded. y=176 sits against the gap — cut,
    // squared.
    let lower = path_cmds(&nodes[0]);
    assert_eq!(arc_count(&lower), 2, "only the two uncut (bottom) corners get an arc; got {lower:?}");
    for cmd in &lower {
        if let PathCmd::ArcTo { y, .. } = cmd {
            assert!(*y > 176.0 + 1e-6, "an arc must not sit on the cut (top) edge; got {cmd:?}");
        }
    }

    // Piece 1 (pixel 20..164, data 850..1000): y=20 is the real top
    // (data 1000) — uncut, rounded. y=164 sits against the gap — cut,
    // squared.
    let upper = path_cmds(&nodes[1]);
    assert_eq!(arc_count(&upper), 2, "only the two uncut (top) corners get an arc; got {upper:?}");
    for cmd in &upper {
        if let PathCmd::ArcTo { y, .. } = cmd {
            assert!(*y < 164.0 - 1e-6, "an arc must not sit on the cut (bottom) edge; got {cmd:?}");
        }
    }
}

/// The commands of a `Path` node, panicking on anything else — the
/// squared-corner pieces #230-2 produces for a cut, rounded rect.
fn path_cmds(node: &SceneNode) -> Vec<PathCmd> {
    let SceneNode::Path { commands, .. } = node else {
        panic!("expected a Path (rounded, cut piece); got {node:?}")
    };
    commands.clone()
}

/// The number of `ArcTo` commands in a path — how many of a clipped
/// rect piece's four corners kept their rounding.
fn arc_count(cmds: &[PathCmd]) -> usize {
    cmds.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count()
}

/// The other half of that contract: a rect with nothing outside the gap is
/// still dropped whole, and still rolls up as an in-gap suppression (both
/// of its endpoints are real coordinates inside the gap).
#[test]
fn a_rect_wholly_inside_a_gap_is_suppressed_and_rolled_up() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_rect(300.0, 700.0)], &[y]);

    assert!(nodes.is_empty(), "nothing of it is visible, so nothing renders");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["rect".to_string()]);
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }
}

/// The retired endpoint-first shortcut (rewrite of
/// `suppress_annotations_for_break_keeps_the_endpoint_answer_for_extents`):
/// an extent with an in-gap ENDPOINT whose surviving portion is non-empty
/// CLIPS to that portion. Pre-#228 the in-gap endpoint suppressed the whole
/// rect, so `annotate_rect(y1=100, y2=400)` under a `(150, 850)` gap
/// vanished instead of rendering its visible 100–150 band.
#[test]
fn an_extent_with_an_in_gap_endpoint_clips_to_its_surviving_portion() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_rect(100.0, 400.0)], &[y]);

    // Data 100 → 320 − (100/150)·144 = 224; data 150 → 176.
    assert_eq!(
        rect_bands(&nodes),
        vec![(176.0, 48.0)],
        "the visible 100–150 portion renders, cut at the gap edge"
    );
    assert!(warnings.is_empty(), "clipping is silent; got {warnings:?}");
}

/// The unchanged half of that same pin: an extent wholly on one side of the
/// gap is untouched — no clip, no warning, one node.
#[test]
fn an_extent_clear_of_every_gap_is_untouched() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_rect(0.0, 140.0)], &[y]);

    // Data 140 → 320 − (140/150)·144 = 185.6; data 0 → 320.
    assert_eq!(rect_bands(&nodes), vec![(185.6, 134.4)]);
    assert!(warnings.is_empty());
}

/// The knife edge where a clip-routed extent STILL suppresses as spanning:
/// a rect whose bounds are exactly the gap's bounds reaches across the gap
/// (neither endpoint is inside it — gaps are open intervals) yet contributes
/// no length to any segment, so nothing survives and the drop rolls into
/// `AnnotationSpansBreakGap`.
#[test]
fn a_rect_exactly_covering_the_gap_rolls_up_as_spanning() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_rect(150.0, 850.0)], &[y]);

    assert!(nodes.is_empty());
    match warnings.as_slice() {
        [RenderWarning::AnnotationSpansBreakGap { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["rect".to_string()]);
        }
        other => panic!("expected one AnnotationSpansBreakGap, got {other:?}"),
    }
}

/// `bracket` is suppress-whole PERMANENTLY (spec §4 #228, D1 — not a
/// residual): its two endpoint tips are what say where the bracketed range
/// begins and ends, so a cut at a gap edge would either strand them at
/// positions the data never had or invent new tips at the cut. A bracket
/// straddling a gap is therefore dropped where a rect of the same extent
/// clips.
#[test]
fn a_bracket_straddling_a_gap_is_dropped_whole_not_split() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let (nodes, warnings) = panel.build(&[y_bracket(100.0, 900.0)], &[y]);

    assert!(nodes.is_empty(), "a bracket admits no honest split");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSpansBreakGap { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["bracket".to_string()]);
        }
        other => panic!("expected one AnnotationSpansBreakGap, got {other:?}"),
    }
}

/// #240-3's rollup, re-aimed at the population that still suppresses
/// (rewrite of
/// `suppress_annotations_for_break_rolls_up_n_spanning_annotations_into_one_warning`,
/// whose three straddling RECTS now clip instead): N brackets spanning the
/// same axis's gap roll up into exactly ONE warning carrying every kind.
#[test]
fn n_spanning_annotations_roll_up_into_one_warning() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let specs = vec![
        y_bracket(100.0, 900.0),
        y_bracket(120.0, 880.0),
        y_bracket(140.0, 860.0),
    ];
    let (nodes, warnings) = panel.build(&specs, &[y]);

    assert!(nodes.is_empty());
    assert_eq!(warnings.len(), 1, "N drops, one warning; got {warnings:?}");
    match &warnings[0] {
        RenderWarning::AnnotationSpansBreakGap { axis, kinds } => {
            assert_eq!(axis, "y");
            assert_eq!(kinds.len(), 3, "the count of dropped instances survives");
            assert!(kinds.iter().all(|k| k == "bracket"));
        }
        other => panic!("expected AnnotationSpansBreakGap, got {other:?}"),
    }
}

/// A span's band clips like a rect, and its label anchors on the midpoint of
/// the LARGEST surviving piece — not on the pre-clip band's midpoint, which
/// belongs to no piece at all.
#[test]
fn a_span_clips_and_its_label_rides_the_largest_surviving_piece() {
    // Gap (200, 600): 288 data pixels split 200:400 → 96 and 192. Segment 0
    // (data 0–200) runs 320 → 224, the indicator 224 → 212, segment 1 (data
    // 600–1000) runs 212 → 20. The larger piece is segment 1's.
    let panel = broken_panel(&[[200.0, 600.0]]);
    let y = panel.remap("y");
    let span = spec(
        r#"{"type":"span","axis":"y","start":0.0,"end":1000.0,"label":"band",
                "label_position":"middle"}"#,
    );
    let (nodes, warnings) = panel.build(&[span], &[y]);

    assert_eq!(rect_bands(&nodes), vec![(224.0, 96.0), (20.0, 192.0)]);
    assert!(warnings.is_empty(), "clipping is silent; got {warnings:?}");

    let labels: Vec<(f64, f64)> = nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Text { x, y, content, .. } if content == "band" => Some((*x, *y)),
            _ => None,
        })
        .collect();
    assert_eq!(
        labels,
        vec![(300.0, 116.0)],
        "one label, on the larger piece's midpoint (20 + 192/2), centered in \
         the plot (50 + 500/2)"
    );
}

/// The same rule's discriminating case: with two EQUAL pieces the label
/// takes the first, and either way it can never land in the gap band — where
/// the pre-clip midpoint of a full-domain span falls when the gap is centred.
#[test]
fn a_clipped_span_label_never_lands_inside_the_gap_band() {
    // Gap (400, 600) splits the 288 data pixels 144:144: segment 0 runs
    // 320 → 176, the indicator 176 → 164, segment 1 runs 164 → 20. The
    // pre-clip band is the whole range 20…320, whose midpoint is 170 —
    // inside the indicator band, which is exactly where the old pre-clip
    // placement would have drawn the label.
    let panel = broken_panel(&[[400.0, 600.0]]);
    let y = panel.remap("y");
    let span = spec(
        r#"{"type":"span","axis":"y","start":0.0,"end":1000.0,"label":"band",
                "label_position":"middle"}"#,
    );
    let (nodes, _) = panel.build(&[span], &[y]);

    let label_y = nodes
        .iter()
        .find_map(|n| match n {
            SceneNode::Text { y, .. } => Some(*y),
            _ => None,
        })
        .expect("the span's label must render");
    assert_eq!(label_y, 248.0, "first of two equal pieces: 176 + 144/2");
    assert!(
        !(164.0..=176.0).contains(&label_y),
        "the label must not land in the invisible gap band; got {label_y}"
    );
}

/// An annotation `line` splits through the vertex machinery (#230-1's
/// splitter, one mechanism for marks and annotations alike): one sub-line
/// per visible segment, cut at the gap edges, with the CROSS-axis
/// coordinate interpolated at each crossing rather than dropped back to the
/// last visible vertex.
#[test]
fn an_annotation_line_splits_at_the_gap_with_its_crossing_interpolated() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    // A diagonal from (norm 0, data 0) to (norm 1, data 1000): x runs
    // 50 → 550 as y runs 320 → 20.
    let line = spec(
        r#"{"type":"line","x1":{"norm":0.0},"y1":0.0,"x2":{"norm":1.0},"y2":1000.0}"#,
    );
    let (nodes, warnings) = panel.build(&[line], &[y]);

    let segments: Vec<(f64, f64, f64, f64)> = nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        })
        .collect();
    // The crossings are at data 150 (15% along the line → x = 125) and data
    // 850 (85% → x = 475), placed at the gap-edge pixels 176 and 164.
    assert_eq!(
        segments,
        vec![(50.0, 320.0, 125.0, 176.0), (475.0, 164.0, 550.0, 20.0)],
        "two pieces, each ending ON the gap edge at the x it actually crosses"
    );
    assert!(warnings.is_empty(), "splitting is silent; got {warnings:?}");
}

/// An `arrow`'s shaft splits like a line, and its head rides the piece that
/// reaches the TIP — rebuilt there, keeping its drawn size and pointing
/// along the piece a reader sees it attached to.
#[test]
fn an_arrow_head_rides_the_tip_piece_with_its_size_kept() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let arrow = spec(
        r#"{"type":"arrow","x":{"norm":0.0},"y":0.0,"x2":{"norm":1.0},"y2":1000.0,
                "head_size":10.0}"#,
    );
    let (nodes, warnings) = panel.build(&[arrow], &[y]);
    assert!(warnings.is_empty(), "splitting is silent; got {warnings:?}");

    let shafts: Vec<(f64, f64)> = nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Line { x2, y2, .. } => Some((*x2, *y2)),
            _ => None,
        })
        .collect();
    assert_eq!(
        shafts,
        vec![(125.0, 176.0), (550.0, 20.0)],
        "the shaft splits in two; the second piece reaches the tip"
    );

    let heads: Vec<&SceneNode> = nodes
        .iter()
        .filter(|n| matches!(n, SceneNode::Path { closed: true, .. }))
        .collect();
    assert_eq!(heads.len(), 1, "exactly one head, on the tip piece");
    let (tip, size) = head_tip_and_size(heads[0]);
    assert_eq!(tip, (550.0, 20.0), "the head sits on the placed tip");
    assert!(
        (size - 10.0).abs() < 1e-9,
        "the head keeps its drawn size; got {size}"
    );
}

/// `(tip, head_size)` of an arrowhead path — `draw::arrowhead_triangle`
/// writes `[tip, left, right]`, with the tip `head_size` from the base
/// midpoint.
fn head_tip_and_size(node: &SceneNode) -> ((f64, f64), f64) {
    let SceneNode::Path { commands, .. } = node else {
        panic!("expected the head Path, got {node:?}");
    };
    let points: Vec<(f64, f64)> = commands
        .iter()
        .filter_map(|cmd| match cmd {
            PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => Some((*x, *y)),
            _ => None,
        })
        .collect();
    assert_eq!(points.len(), 3, "an arrowhead is three vertices");
    let (tip, left, right) = (points[0], points[1], points[2]);
    let base_mid = ((left.0 + right.0) / 2.0, (left.1 + right.1) / 2.0);
    (tip, (tip.0 - base_mid.0).hypot(tip.1 - base_mid.1))
}

/// An arrow with nothing outside the gap is dropped whole and warns — the
/// suppress path is still reachable for a cut-routed kind.
#[test]
fn an_arrow_wholly_inside_a_gap_is_suppressed_and_warns() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let arrow = spec(
        r#"{"type":"arrow","x":{"norm":0.0},"y":300.0,"x2":{"norm":1.0},"y2":700.0}"#,
    );
    let (nodes, warnings) = panel.build(&[arrow], &[y]);

    assert!(nodes.is_empty(), "no piece of the shaft survives, so no head either");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["arrow".to_string()]);
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }
}

/// An arrow whose TAIL is visible but whose TIP lies inside a gap: the tip
/// has no piece to terminate the head on, so only the HEAD drops. The
/// shaft's own visible piece still renders, truncated at the gap edge it
/// ends on — clip-fate geometry (a faithful rendering of what is visible),
/// not a silent drop of the annotation, the same fate a
/// `mark_segment(arrow=True)` shaft gives an in-gap tip (#230-1). This is
/// the case neither `an_arrow_head_rides_the_tip_piece_with_its_size_kept`
/// (visible tip) nor `an_arrow_wholly_inside_a_gap_is_suppressed_and_warns`
/// (nothing survives) covers — the only one where head and shaft disagree.
#[test]
fn an_arrow_with_an_in_gap_tip_drops_only_the_head() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let arrow = spec(
        r#"{"type":"arrow","x":{"norm":0.0},"y":0.0,"x2":{"norm":1.0},"y2":500.0,
                "head_size":10.0}"#,
    );
    let (nodes, warnings) = panel.build(&[arrow], &[y]);
    assert!(warnings.is_empty(), "the drop is silent; got {warnings:?}");

    let shafts: Vec<&SceneNode> =
        nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).collect();
    assert_eq!(
        shafts.len(),
        1,
        "the visible tail survives as one shaft piece; got {shafts:?}"
    );

    let heads: Vec<&SceneNode> = nodes
        .iter()
        .filter(|n| matches!(n, SceneNode::Path { closed: true, .. }))
        .collect();
    assert!(heads.is_empty(), "the head drops with its hidden tip; got {heads:?}");
}

/// Point-anchored kinds keep the plain membership test: in the gap → gone,
/// outside it → built, whatever lies between them.
#[test]
fn point_kinds_keep_the_membership_test() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let specs = vec![
        y_text(500.0, "inside the break"),
        y_text(100.0, "outside the break"),
        spec(r#"{"type":"callout","x":{"norm":0.5},"y":500.0,"text":"in"}"#),
        spec(r#"{"type":"image","x":{"norm":0.5},"y":500.0,"src":"a.png"}"#),
    ];
    let (nodes, warnings) = panel.build(&specs, &[y]);

    let texts: Vec<&str> = nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Text { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["outside the break"]);
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["text", "callout", "image"]);
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }
}

/// A callout's background rect is a TEXT BOX — drawn size around an anchor,
/// never data — so it is never clipped, even when it straddles the gap band
/// in pixels. The negative control for the clip's reach.
#[test]
fn a_callout_box_is_never_clipped_even_across_the_gap_band() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    // Anchor at data 125 places at pixel 200; the box centres 30px above
    // that (ty = 170) and is 22px tall (14 + 2·4 padding), so it covers
    // 159…181 — straight across the 164…176 indicator band.
    let callout = spec(r#"{"type":"callout","x":{"norm":0.5},"y":125.0,"text":"note"}"#);
    let (nodes, warnings) = panel.build(&[callout], &[y]);

    assert_eq!(
        rect_bands(&nodes),
        vec![(159.0, 22.0)],
        "ONE box, uncut, spanning the gap band"
    );
    assert!(warnings.is_empty());
}

/// The desugar fan-out, honestly: one `annotate_hline(label=...)` emits a
/// `line` plus its own `text` at the same y, so under a y break the pair
/// shares one fate and rolls into one warning naming both kinds.
#[test]
fn an_hline_and_its_label_share_one_fate_under_a_y_break() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let hline = |value: f64| {
        vec![
            spec(&format!(
                r#"{{"type":"line","x1":{{"norm":0.0}},"y1":{value},"x2":{{"norm":1.0}},
                        "y2":{value}}}"#
            )),
            y_text(value, "label"),
        ]
    };

    let (dropped, warnings) = panel.build(&hline(500.0), &[y]);
    assert!(dropped.is_empty(), "an hline in the gap takes its label with it");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["line", "text"], "both primitives, one warning");
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }

    let (kept, warnings) = panel.build(&hline(100.0), &[y]);
    assert_eq!(kept.len(), 2, "a visible hline keeps both primitives");
    assert!(warnings.is_empty());
}

/// The mixed-fate half of that fan-out: on an x break, a full-width `line`
/// CLIPS (its x extent is data and part of it survives) while a `text`
/// anchored inside the gap point-suppresses. One annotation's pieces render;
/// the other's warning names only the kind that was actually dropped.
#[test]
fn a_clipped_line_and_a_suppressed_label_report_only_the_drop() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let x = panel.remap("x");
    let specs = vec![
        spec(
            r#"{"type":"line","x1":0.0,"y1":{"norm":0.0},"x2":1000.0,"y2":{"norm":0.0}}"#,
        ),
        spec(r#"{"type":"text","x":500.0,"y":{"norm":0.5},"text":"mid"}"#),
    ];
    let (nodes, warnings) = panel.build(&specs, &[x]);

    let xs: Vec<(f64, f64)> = nodes
        .iter()
        .filter_map(|n| match n {
            SceneNode::Line { x1, x2, .. } => Some((*x1, *x2)),
            _ => None,
        })
        .collect();
    // x pixels 50 → 550: 488 data pixels split 150:150 → 244 each, so
    // data 150 lands at 294 and data 850 at 306 across the indicator.
    assert_eq!(
        xs,
        vec![(50.0, 294.0), (306.0, 550.0)],
        "the line clips into two visible pieces"
    );
    assert!(
        !nodes.iter().any(|n| matches!(n, SceneNode::Text { .. })),
        "the in-gap label is gone"
    );
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(axis, "x");
            assert_eq!(kinds, &["text".to_string()], "only the DROP is reported");
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }
}

/// A drop on one axis beats a clip on the other: an annotation with nothing
/// left on y is gone, not rendered in pieces along x.
#[test]
fn a_drop_on_one_axis_wins_over_a_cut_on_the_other() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let (x, y) = (panel.remap("x"), panel.remap("y"));
    // x spans the whole domain (clippable); y sits entirely in the gap.
    let rect = spec(
        r#"{"type":"rect","x1":0.0,"y1":300.0,"x2":1000.0,"y2":700.0}"#,
    );
    let (nodes, warnings) = panel.build(&[rect], &[x, y]);

    assert!(nodes.is_empty(), "nothing survives on y, so nothing renders");
    assert_eq!(warnings.len(), 1, "one drop, one warning; got {warnings:?}");
}

/// With no breaks the router IS the plain build: same nodes, no warnings.
/// This is what keeps every un-broken chart byte-identical.
#[test]
fn no_breaks_routes_straight_to_the_plain_build() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let specs = vec![y_rect(0.0, 1000.0), y_text(500.0, "t")];
    let ctx = ScaleContext {
        plot_area: panel.plot_area,
        x_scale: &panel.x_scale,
        y_scale: &panel.y_scale,
        breaks: &[],
    };
    let mut warnings = Vec::new();
    let routed = build_annotations_through_breaks(&specs, &ctx, &mut warnings);
    let plain = build_annotations(&specs, &ctx);

    assert_eq!(routed.above_marks, plain.above_marks);
    assert_eq!(routed.below_marks, plain.below_marks);
    assert!(warnings.is_empty());
}

/// `z_bucket`'s second call site — the routed build, not just the plain
/// one — is exercised too: a `z="below_marks"` Text untouched by the break
/// still lands in `below_marks`, not `above_marks`. Every other fixture in
/// this module goes through `BrokenPanel::build`, which asserts
/// `nodes.below_marks.is_empty()` for every case, so without this pin that
/// assertion is never falsified by a real below_marks case reaching the
/// routed build.
#[test]
fn a_below_marks_text_untouched_by_the_break_still_routes_to_below_marks() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let text = AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Data(100.0),
        text: "caption".to_string(),
        font_size: 12.0,
        color: "#000000".to_string(),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "below_marks".to_string(),
    };
    let ctx = ScaleContext {
        plot_area: panel.plot_area,
        x_scale: &panel.x_scale,
        y_scale: &panel.y_scale,
        breaks: &[y],
    };
    let mut warnings = Vec::new();
    let nodes = build_annotations_through_breaks(&[text], &ctx, &mut warnings);

    assert!(warnings.is_empty());
    assert!(
        nodes.above_marks.is_empty(),
        "a below_marks Text must not land above marks; got {:?}",
        nodes.above_marks
    );
    assert_eq!(
        nodes.below_marks.len(),
        1,
        "z_bucket must still route it through the router"
    );
    match &nodes.below_marks[0] {
        SceneNode::Text { content, .. } => assert_eq!(content, "caption"),
        other => panic!("expected the below_marks Text, got {other:?}"),
    }
}

/// `RenderWarning::AnnotationSuppressedByBreak`'s third named population: a
/// `Cut`-routed extent kind (`rect`) whose two bounds on the broken axis
/// are NOT both data — one is pixel-space — degenerates to
/// `AxisCoords::Points` on that axis. With only one data bound there is no
/// interval to clip, so an in-gap value there drops the whole rect even
/// though `rect`'s own routing is `Cut` — enumerated in the warning's own
/// doc (`render/mod.rs`), not merely asserted here.
#[test]
fn a_mixed_coordinate_rect_drops_whole_when_its_data_bound_is_in_the_gap() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap("y");
    let rect = spec(
        r#"{"type":"rect","x1":{"norm":0.0},"y1":500.0,"x2":{"norm":1.0},"y2":{"px":50.0}}"#,
    );
    let (nodes, warnings) = panel.build(&[rect], &[y]);

    assert!(nodes.is_empty(), "an unreadable interval cannot be clipped; got {nodes:?}");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(axis, "y");
            assert_eq!(kinds, &["rect".to_string()]);
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }
}
