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
use crate::spec::coord::CoordKind;
use crate::scale::linear::LinearScale;
// Batch H Task 1 fix: `PathCmd` is only referenced by test assertions
// (production code now goes through `marks::arrowhead::arrowhead_path`,
// which builds its own `PathCmd` variants internally) — scoped here rather
// than at module scope so a module-wide `use` doesn't go stale/unused again.
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
        theme: &ThemeInputs::default(),
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
        theme: &ThemeInputs::default(),
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
        theme: &ThemeInputs::default(),
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
        theme: &ThemeInputs::default(),
    };
    let specs = vec![AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        text: "hello".to_string(),
        font_size: 14.0,
        color: Some("#ff0000".to_string()),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".to_string(),
    }];
    let ann = build_annotations(&specs, &ctx).unwrap();
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
            assert_eq!(style.color, Some(Color::rgb(255, 0, 0)));
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
        theme: &ThemeInputs::default(),
    };
    let specs = vec![AnnotationSpec::Line {
        x1: CoordValue::Norm { norm: 0.0 },
        y1: CoordValue::Norm { norm: 0.5 },
        x2: CoordValue::Norm { norm: 1.0 },
        y2: CoordValue::Norm { norm: 0.5 },
        stroke: Some("#000000".to_string()),
        stroke_width: 2.0,
        dash: Some(vec![4.0, 2.0]),
        entry: None,
    }];
    let ann = build_annotations(&specs, &ctx).unwrap();
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
        theme: &ThemeInputs::default(),
    };
    // `curve` is no longer a field on AnnotationSpec::Arrow (dropped as dead code).
    let specs = vec![AnnotationSpec::Arrow {
        x: CoordValue::Norm { norm: 0.0 },
        y: CoordValue::Norm { norm: 0.0 },
        x2: CoordValue::Norm { norm: 1.0 },
        y2: CoordValue::Norm { norm: 1.0 },
        stroke: Some("#333333".to_string()),
        stroke_width: 1.5,
        head_size: 8.0,
    }];
    let ann = build_annotations(&specs, &ctx).unwrap();
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
        theme: &ThemeInputs::default(),
    };
    let specs = vec![AnnotationSpec::Span {
        axis: "x".to_string(),
        start: CoordValue::Norm { norm: 0.2 },
        end: CoordValue::Norm { norm: 0.8 },
        fill: Some("#aaaaaa".to_string()),
        opacity: 0.3,
        label_position: "center".to_string(),
        label: Some("span".to_string()),
    }];
    let ann = build_annotations(&specs, &ctx).unwrap();
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

/// A parsed colour as the renderer draws it; the string must be a colour.
fn colour(s: &str) -> Color {
    resolve_color("test", "color", s).unwrap().expect("a colour, not a paint word")
}

/// The refusal (or nodes) of building one annotation spec on a plain panel.
fn build_on_plain_panel(json: &str) -> Result<AnnotationNodes, AnnotationRefusal> {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
        theme: &ThemeInputs::default(),
    };
    let mut warnings = Vec::new();
    build_annotations_through_breaks(&[spec(json)], &ctx, None, false, &mut warnings)
}

const POINT: &str = r#""x":{"px":10},"y":{"px":20}"#;
const SEGMENT: &str = r#""x1":{"px":10},"y1":{"px":20},"x2":{"px":60},"y2":{"px":70}"#;

/// Every colour field of every spec that draws one, as `(spec json, message)`
/// with `notacolor` in that field.
fn bad_colour_specs() -> Vec<(String, &'static str)> {
    let bad = r#""notacolor""#;
    vec![
        (format!(r#"{{"type":"text",{POINT},"text":"t","color":{bad}}}"#), "text: color"),
        (format!(r#"{{"type":"arrow","x":{{"px":1}},"y":{{"px":2}},"x2":{{"px":9}},"y2":{{"px":9}},"stroke":{bad}}}"#), "arrow: stroke"),
        (format!(r#"{{"type":"rect",{SEGMENT},"fill":{bad}}}"#), "rect: fill"),
        (format!(r#"{{"type":"rect",{SEGMENT},"stroke":{bad}}}"#), "rect: stroke"),
        (format!(r#"{{"type":"line",{SEGMENT},"stroke":{bad}}}"#), "line: stroke"),
        (format!(r#"{{"type":"span","axis":"x","start":{{"px":1}},"end":{{"px":9}},"fill":{bad}}}"#), "span: fill"),
        (format!(r#"{{"type":"bracket",{SEGMENT},"label":"b","stroke":{bad}}}"#), "bracket: stroke"),
        (format!(r#"{{"type":"callout",{POINT},"text":"t","background":{bad}}}"#), "callout: background"),
        (format!(r#"{{"type":"callout",{POINT},"text":"t","border_color":{bad}}}"#), "callout: border_color"),
    ]
}

#[test]
fn an_unparseable_colour_on_any_field_is_refused_naming_the_kind_field_and_value() {
    for (json, field) in bad_colour_specs() {
        let refused = build_on_plain_panel(&json).expect_err(&json);
        let parser = parse_color_str("notacolor").unwrap_err();
        assert_eq!(
            refused.to_string(),
            format!("annotation {field}='notacolor' is not a valid color ({parser})"),
            "{json}"
        );
    }
}

/// `none` and `transparent` are accepted paint words, in any case and padding.
/// They draw no paint: a cleared fill and a cleared stroke.
#[test]
fn the_paint_sentinels_are_accepted_and_draw_no_paint() {
    for word in ["none", "transparent", "None", " TRANSPARENT "] {
        let json = format!(r#"{{"type":"rect",{SEGMENT},"fill":"{word}","stroke":"{word}"}}"#);
        let nodes = build_on_plain_panel(&json).expect(word);
        match nodes.above_marks.as_slice() {
            [SceneNode::Rect { style, .. }] => {
                assert_eq!(style.fill, None, "{word}");
                assert_eq!(style.stroke, None, "{word}");
            }
            other => panic!("expected one rect, got {other:?}"),
        }
    }
}

#[test]
fn a_parseable_colour_draws_as_parsed() {
    let json = format!(r##"{{"type":"rect",{SEGMENT},"fill":"#123"}}"##);
    let nodes = build_on_plain_panel(&json).unwrap();
    match nodes.above_marks.as_slice() {
        [SceneNode::Rect { style, .. }] => assert_eq!(style.fill, Some(Color::rgb(0x11, 0x22, 0x33))),
        other => panic!("expected one rect, got {other:?}"),
    }
}

/// Under a caller's `CoordFlip` the break remap measures the exchanged
/// (drawn) axis; both rollups name the caller's letter for it.
#[test]
fn break_rollups_name_the_callers_axis_under_a_flip() {
    for (lo, hi, expect_suppressed) in [(300.0, 700.0, true), (150.0, 850.0, false)] {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap(AlignAxis::Y);
        let breaks = [y];
        let ctx = ScaleContext {
            plot_area: panel.plot_area,
            x_scale: &panel.x_scale,
            y_scale: &panel.y_scale,
            breaks: &breaks,
            theme: &ThemeInputs::default(),
        };
        // The caller wrote an x extent; the flip draws it on the broken y axis.
        let callers_spec = y_rect(lo, hi).exchanged();
        let mut warnings = Vec::new();
        build_annotations_through_breaks(&[callers_spec], &ctx, None, true, &mut warnings).unwrap();
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, .. }] if expect_suppressed => {
                assert_eq!(*axis, AlignAxis::X)
            }
            [RenderWarning::AnnotationSpansBreakGap { axis, .. }] if !expect_suppressed => {
                assert_eq!(*axis, AlignAxis::X)
            }
            other => panic!("expected one rollup naming x, got {other:?}"),
        }
    }
}

#[test]
fn empty_specs_returns_empty_vecs() {
    let (x_scale, y_scale, plot_area) = test_ctx();
    let ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
        theme: &ThemeInputs::default(),
    };
    let ann = build_annotations(&[], &ctx).unwrap();
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
        theme: &ThemeInputs::default(),
    };

    let mk_text = |z: &str| AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        text: z.to_string(),
        font_size: 12.0,
        color: Some("#000000".to_string()),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: z.to_string(),
    };

    let specs = vec![mk_text("below_marks"), mk_text("above_marks")];
    let ann = build_annotations(&specs, &ctx).unwrap();

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
        theme: &ThemeInputs::default(),
    };

    let specs = vec![AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        text: "unknown".to_string(),
        font_size: 12.0,
        color: Some("#000000".to_string()),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "front".to_string(), // old default — must still go above
    }];
    let ann = build_annotations(&specs, &ctx).unwrap();
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
        theme: &ThemeInputs::default(),
    };

    let specs = vec![
        AnnotationSpec::Line {
            x1: CoordValue::Norm { norm: 0.0 },
            y1: CoordValue::Norm { norm: 0.5 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 0.5 },
            stroke: Some("#000000".to_string()),
            stroke_width: 1.0,
            dash: None,
            entry: None,
        },
        AnnotationSpec::Arrow {
            x: CoordValue::Norm { norm: 0.0 },
            y: CoordValue::Norm { norm: 0.0 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 1.0 },
            stroke: Some("#000000".to_string()),
            stroke_width: 1.5,
            head_size: 0.0, // no arrowhead → 1 node only
        },
    ];
    let ann = build_annotations(&specs, &ctx).unwrap();
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
        theme: &ThemeInputs::default(),
    };
    let specs = vec![AnnotationSpec::Image {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Norm { norm: 0.5 },
        src: "https://example.com/logo.png".to_string(),
        width: 40.0,
        height: 30.0,
        anchor: "middle".to_string(),
    }];
    let ann = build_annotations(&specs, &ctx).unwrap();
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
        theme: &ThemeInputs::default(),
    };
    let mut out = Vec::new();
    emit_span(
        "x",
        &CoordValue::Pixel { px: f64::NAN },
        &CoordValue::Data(50.0),
        Some("#ff0000"),
        0.3,
        None,
        "middle",
        &ctx,
        &mut out,
    ).unwrap();
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
        Some("#ff0000"),
        0.3,
        None,
        "middle",
        &ctx,
        &mut out2,
    ).unwrap();
    assert_eq!(out2.len(), 1, "finite span must emit exactly one rect");
}

/// R1 port: `emit_arrow` clamps `head_size` to the shaft length so an
/// epsilon-length arrow does not produce an arrowhead triangle that
/// vastly overshoots the line's start point.
#[test]
fn emit_arrow_clamps_head_size_to_epsilon_shaft() {
    let mut out = Vec::new();
    emit_arrow(100.0, 100.0, 100.001, 100.0, Some(colour("#000000")), 1.0, 8.0, &mut out);
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
/// vertices `marks::arrowhead::arrowhead_triangle` computes for the same
/// endpoints/head_size — proving the annotation path and
/// `marks::segment::build`'s new `arrow=true` path (segment.rs) share one
/// geometry function rather than two independently-typed copies that
/// could silently drift. Fails to compile against the pre-extraction
/// surface: `marks::arrowhead::arrowhead_triangle` did not exist before
/// this task.
#[test]
fn emit_arrow_triangle_matches_shared_arrowhead_triangle_helper() {
    let (x1, y1, x2, y2, head_size) = (10.0, 20.0, 130.0, 65.0, 8.0);
    let mut out = Vec::new();
    emit_arrow(x1, y1, x2, y2, Some(colour("#123456")), 1.0, head_size, &mut out);
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

/// "below" draws as "down" and "above" as "up", tips and label alike.
#[test]
fn emit_bracket_below_draws_like_down_and_above_like_up() {
    let drawn = |direction: &str| {
        let mut out = Vec::new();
        emit_bracket(0.0, 100.0, 100.0, 100.0, "n", direction, Some(colour("#000000")), colour("#111111"), 6.0, &mut out);
        format!("{out:?}")
    };
    assert_eq!(drawn("below"), drawn("down"));
    assert_eq!(drawn("above"), drawn("up"));
    assert_ne!(drawn("below"), drawn("up"));
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
        Some(colour("#000000")),
        colour("#000000"),
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
        Some(colour("#000000")),
        colour("#000000"),
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
        theme: &ThemeInputs::default(),
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
            Some("#fff"),
            Some("#333"),
            0.0,
            None,
            None,
            &mut out,
        ).unwrap();
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
        theme: &ThemeInputs::default(),
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
            Some("#fff"),
            Some("#333"),
            0.0,
            None,
            None,
            &mut out,
        ).unwrap();
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
                color: Some("#000".into()),
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
                stroke: Some("#000".into()),
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
                fill: Some("#ccc".into()),
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
                stroke: Some("#000".into()),
                stroke_width: 1.5,
                dash: None,
                entry: None,
            },
            "line",
        ),
        (
            AnnotationSpec::Span {
                axis: "x".into(),
                start: CoordValue::Data(1.0),
                end: CoordValue::Data(2.0),
                fill: Some("#ccc".into()),
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
                stroke: Some("#000".into()),
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
                background: Some("#fff".into()),
                border_color: Some("#333".into()),
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
        color: Some("#000".into()),
        anchor: "middle".into(),
        baseline: "middle".into(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".into(),
    };
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::X), AxisCoords::Points(vec![10.0]));
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::Y), AxisCoords::Points(vec![20.0]));
}

#[test]
fn data_coords_on_axis_excludes_px_and_norm() {
    let spec = AnnotationSpec::Text {
        x: CoordValue::Pixel { px: 5.0 },
        y: CoordValue::Norm { norm: 0.5 },
        text: "t".into(),
        font_size: 12.0,
        color: Some("#000".into()),
        anchor: "middle".into(),
        baseline: "middle".into(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".into(),
    };
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::X), AxisCoords::Points(Vec::new()));
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::Y), AxisCoords::Points(Vec::new()));
}

#[test]
fn data_coords_on_axis_arrow_both_endpoints() {
    let spec = AnnotationSpec::Arrow {
        x: CoordValue::Data(1.0),
        y: CoordValue::Data(2.0),
        x2: CoordValue::Data(3.0),
        y2: CoordValue::Data(4.0),
        stroke: Some("#000".into()),
        stroke_width: 1.5,
        head_size: 8.0,
    };
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::X), AxisCoords::Extent(1.0, 3.0));
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::Y), AxisCoords::Extent(2.0, 4.0));
}

#[test]
fn data_coords_on_axis_rect_line_bracket_share_shape() {
    let specs = [
        AnnotationSpec::Rect {
            x1: CoordValue::Data(1.0),
            y1: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            fill: Some("#ccc".into()),
            opacity: 0.3,
            corner_radius: 0.0,
            stroke: None,
        },
        AnnotationSpec::Line {
            x1: CoordValue::Data(1.0),
            y1: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            stroke: Some("#000".into()),
            stroke_width: 1.5,
            dash: None,
            entry: None,
        },
        AnnotationSpec::Bracket {
            x1: CoordValue::Data(1.0),
            y1: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            label: "".into(),
            direction: "up".into(),
            stroke: Some("#000".into()),
            tip_length: 6.0,
        },
    ];
    for spec in &specs {
        assert_eq!(data_coords_on_axis(spec, AlignAxis::X), AxisCoords::Extent(1.0, 3.0));
        assert_eq!(data_coords_on_axis(spec, AlignAxis::Y), AxisCoords::Extent(2.0, 4.0));
    }
}

#[test]
fn data_coords_on_axis_span_gated_by_its_own_axis_field() {
    let x_span = AnnotationSpec::Span {
        axis: "x".into(),
        start: CoordValue::Data(1.0),
        end: CoordValue::Data(2.0),
        fill: Some("#ccc".into()),
        opacity: 0.2,
        label_position: "center".into(),
        label: None,
    };
    assert_eq!(data_coords_on_axis(&x_span, AlignAxis::X), AxisCoords::Extent(1.0, 2.0));
    assert_eq!(data_coords_on_axis(&x_span, AlignAxis::Y), AxisCoords::Points(Vec::new()));

    let y_span = AnnotationSpec::Span {
        axis: "y".into(),
        start: CoordValue::Data(5.0),
        end: CoordValue::Data(6.0),
        fill: Some("#ccc".into()),
        opacity: 0.2,
        label_position: "center".into(),
        label: None,
    };
    assert_eq!(data_coords_on_axis(&y_span, AlignAxis::Y), AxisCoords::Extent(5.0, 6.0));
    assert_eq!(data_coords_on_axis(&y_span, AlignAxis::X), AxisCoords::Points(Vec::new()));
}

#[test]
fn data_coords_on_axis_callout_includes_optional_text_position() {
    let with_text_pos = AnnotationSpec::Callout {
        x: CoordValue::Data(1.0),
        y: CoordValue::Data(2.0),
        text: "c".into(),
        arrow: "curved".into(),
        padding: 4.0,
        background: Some("#fff".into()),
        border_color: Some("#333".into()),
        border_radius: 0.0,
        text_x: Some(CoordValue::Data(11.0)),
        text_y: Some(CoordValue::Data(12.0)),
    };
    assert_eq!(
        data_coords_on_axis(&with_text_pos, AlignAxis::X),
        AxisCoords::Points(vec![1.0, 11.0]),
        "a callout's anchor and text anchor are two points, never one interval"
    );
    assert_eq!(
        data_coords_on_axis(&with_text_pos, AlignAxis::Y),
        AxisCoords::Points(vec![2.0, 12.0])
    );

    // text_x/text_y omitted: only the anchor coordinate is reported.
    let no_text_pos = AnnotationSpec::Callout {
        x: CoordValue::Data(1.0),
        y: CoordValue::Data(2.0),
        text: "c".into(),
        arrow: "curved".into(),
        padding: 4.0,
        background: Some("#fff".into()),
        border_color: Some("#333".into()),
        border_radius: 0.0,
        text_x: None,
        text_y: None,
    };
    assert_eq!(data_coords_on_axis(&no_text_pos, AlignAxis::X), AxisCoords::Points(vec![1.0]));
    assert_eq!(data_coords_on_axis(&no_text_pos, AlignAxis::Y), AxisCoords::Points(vec![2.0]));
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
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::X), AxisCoords::Points(vec![7.0]));
    assert_eq!(data_coords_on_axis(&spec, AlignAxis::Y), AxisCoords::Points(vec![8.0]));
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
        AlignAxis::X,
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
        theme: &ThemeInputs::default(),
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
        theme: &ThemeInputs::default(),
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
        AlignAxis::Y,
        (0.0, 100.0),
        (20.0, 320.0),
        &result,
    );
    let broken_ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[remap],
        theme: &ThemeInputs::default(),
    };
    let plain_ctx = ScaleContext {
        plot_area,
        x_scale: &x_scale,
        y_scale: &y_scale,
        breaks: &[],
        theme: &ThemeInputs::default(),
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
        theme: &ThemeInputs::default(),
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
    fn remap(&self, axis: AlignAxis) -> break_axis::BreakRemap<'_> {
        let (pixels, result) = if axis.is_y() {
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
            theme: &ThemeInputs::default(),
        };
        let mut warnings = Vec::new();
        let nodes = build_annotations_through_breaks(specs, &ctx, None, false, &mut warnings).unwrap();
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
    let (nodes, warnings) = panel.build(&[y_rect(300.0, 700.0)], &[y]);

    assert!(nodes.is_empty(), "nothing of it is visible, so nothing renders");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(*axis, AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
    let (nodes, warnings) = panel.build(&[y_rect(150.0, 850.0)], &[y]);

    assert!(nodes.is_empty());
    match warnings.as_slice() {
        [RenderWarning::AnnotationSpansBreakGap { axis, kinds }] => {
            assert_eq!(*axis, AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
    let (nodes, warnings) = panel.build(&[y_bracket(100.0, 900.0)], &[y]);

    assert!(nodes.is_empty(), "a bracket admits no honest split");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSpansBreakGap { axis, kinds }] => {
            assert_eq!(*axis, AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
            assert_eq!(*axis, AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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

/// `(tip, head_size)` of an arrowhead path — `marks::arrowhead::arrowhead_triangle`
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
    let y = panel.remap(AlignAxis::Y);
    let arrow = spec(
        r#"{"type":"arrow","x":{"norm":0.0},"y":300.0,"x2":{"norm":1.0},"y2":700.0}"#,
    );
    let (nodes, warnings) = panel.build(&[arrow], &[y]);

    assert!(nodes.is_empty(), "no piece of the shaft survives, so no head either");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(*axis, AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
            assert_eq!(*axis, AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
    let y = panel.remap(AlignAxis::Y);
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
            assert_eq!(*axis, AlignAxis::Y);
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
    let x = panel.remap(AlignAxis::X);
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
            assert_eq!(*axis, AlignAxis::X);
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
    let (x, y) = (panel.remap(AlignAxis::X), panel.remap(AlignAxis::Y));
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
        theme: &ThemeInputs::default(),
    };
    let mut warnings = Vec::new();
    let routed = build_annotations_through_breaks(&specs, &ctx, None, false, &mut warnings).unwrap();
    let plain = build_annotations(&specs, &ctx).unwrap();

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
    let y = panel.remap(AlignAxis::Y);
    let text = AnnotationSpec::Text {
        x: CoordValue::Norm { norm: 0.5 },
        y: CoordValue::Data(100.0),
        text: "caption".to_string(),
        font_size: 12.0,
        color: Some("#000000".to_string()),
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
        theme: &ThemeInputs::default(),
    };
    let mut warnings = Vec::new();
    let nodes = build_annotations_through_breaks(&[text], &ctx, None, false, &mut warnings).unwrap();

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
    let y = panel.remap(AlignAxis::Y);
    let rect = spec(
        r#"{"type":"rect","x1":{"norm":0.0},"y1":500.0,"x2":{"norm":1.0},"y2":{"px":50.0}}"#,
    );
    let (nodes, warnings) = panel.build(&[rect], &[y]);

    assert!(nodes.is_empty(), "an unreadable interval cannot be clipped; got {nodes:?}");
    match warnings.as_slice() {
        [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
            assert_eq!(*axis, AlignAxis::Y);
            assert_eq!(kinds, &["rect".to_string()]);
        }
        other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
    }
}

/// An instant fares exactly as its epoch number does on a broken continuous
/// axis: suppressed inside a gap, cut across one, placed past one.
#[test]
fn an_instant_fares_as_its_number_through_a_break() {
    let panel = broken_panel(&[[300.0, 400.0]]);
    let x = panel.remap(AlignAxis::X);
    let text_at = |x: CoordValue| AnnotationSpec::Text {
        x,
        y: CoordValue::Norm { norm: 0.5 },
        text: "t".to_string(),
        font_size: 12.0,
        color: Some("#333333".to_string()),
        anchor: "middle".to_string(),
        baseline: "middle".to_string(),
        angle: 0.0,
        dx: 0.0,
        dy: 0.0,
        z: "above_marks".to_string(),
    };
    let span_of = |start: CoordValue, end: CoordValue| AnnotationSpec::Span {
        axis: "x".to_string(),
        start,
        end,
        fill: Some("#cccccc".to_string()),
        opacity: 0.2,
        label_position: "center".to_string(),
        label: None,
    };
    let inst = |ms: f64| CoordValue::Instant { instant: ms, text: None };
    let cases = [
        (text_at(inst(350.0)), text_at(CoordValue::Data(350.0)), 0, 1),
        (span_of(inst(200.0), inst(600.0)), span_of(CoordValue::Data(200.0), CoordValue::Data(600.0)), 2, 0),
        (text_at(inst(700.0)), text_at(CoordValue::Data(700.0)), 1, 0),
    ];
    for (instant_spec, number_spec, drawn, warned) in cases {
        let (a, wa) = panel.build(&[instant_spec], std::slice::from_ref(&x));
        let (b, wb) = panel.build(&[number_spec], std::slice::from_ref(&x));
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(format!("{wa:?}"), format!("{wb:?}"));
        assert_eq!(a.len(), drawn, "{a:?}");
        assert_eq!(wa.len(), warned, "{wa:?}");
    }
}

// ── Category coordinates ────────────────────────────────────────────────────

mod categories {
    use super::*;
    use crate::scale::discrete::DiscreteLayout;
    use crate::scale::ordinal::OrdinalScale;

    const PLOT: Rect = Rect { x: 50.0, y: 20.0, w: 400.0, h: 300.0 };

    /// A band axis over `domain` spanning `range` with `padding_inner`.
    fn band(domain: &[&str], range: [f64; 2], padding_inner: f64) -> ScaleKind {
        ScaleKind::Ordinal(OrdinalScale::new_internal(
            domain.iter().map(|c| c.to_string()).collect(),
            range.to_vec(),
            DiscreteLayout::band(padding_inner, 0.0, 0.5),
        ))
    }

    fn linear() -> ScaleKind {
        let (x, _, _) = test_ctx();
        x
    }

    fn category(label: &str) -> CoordValue {
        CoordValue::Category { category: label.to_string() }
    }

    fn span(axis: &str, start: CoordValue, end: CoordValue) -> AnnotationSpec {
        AnnotationSpec::Span {
            axis: axis.to_string(),
            start,
            end,
            fill: Some("#cccccc".to_string()),
            opacity: 0.2,
            label_position: "center".to_string(),
            label: None,
        }
    }

    fn text(x: CoordValue, y: CoordValue) -> AnnotationSpec {
        AnnotationSpec::Text {
            x,
            y,
            text: "t".to_string(),
            font_size: 12.0,
            color: Some("#333333".to_string()),
            anchor: "middle".to_string(),
            baseline: "middle".to_string(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "above_marks".to_string(),
        }
    }

    fn run(
        specs: &[AnnotationSpec],
        x: &ScaleKind,
        y: &ScaleKind,
        projected: Option<&CoordKind>,
    ) -> (Result<AnnotationNodes, AnnotationRefusal>, Vec<RenderWarning>) {
        let ctx = ScaleContext { plot_area: PLOT, x_scale: x, y_scale: y, breaks: &[], theme: &ThemeInputs::default() };
        let mut warnings = Vec::new();
        let built = build_annotations_through_breaks(specs, &ctx, projected, false, &mut warnings);
        (built, warnings)
    }

    fn rect_extent_of(nodes: &AnnotationNodes) -> (f64, f64, f64, f64) {
        match nodes.above_marks.first() {
            Some(SceneNode::Rect { x, y, w, h, .. }) => (*x, *y, *w, *h),
            other => panic!("expected a rect, got {other:?}"),
        }
    }

    fn text_position(nodes: &AnnotationNodes) -> (f64, f64) {
        match nodes.above_marks.first() {
            Some(SceneNode::Text { x, y, .. }) => (*x, *y),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn wire_form_is_a_category_object() {
        let v: CoordValue = serde_json::from_str(r#"{"category": "b"}"#).unwrap();
        assert_eq!(v.category(), Some("b"));
        let n: CoordValue = serde_json::from_str("2.5").unwrap();
        assert!(matches!(n, CoordValue::Data(d) if d == 2.5));
    }

    #[test]
    fn a_point_lands_on_the_band_centre_the_scale_gives_it() {
        // Domain order c, a, b, d (a sorted axis): `b` is the third band.
        let x = band(&["c", "a", "b", "d"], [50.0, 450.0], 0.0);
        let (built, warnings) = run(&[text(category("b"), CoordValue::Data(0.0))], &x, &linear(), None);
        let (px, _) = text_position(&built.unwrap());
        assert!((px - 300.0).abs() < 1e-9, "band b centre is 300, got {px}");
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_point_on_the_vertical_axis_lands_on_its_band() {
        let y = band(&["a", "b", "c", "d"], [20.0, 320.0], 0.0);
        let (built, _) = run(&[text(CoordValue::Data(0.0), category("c"))], &linear(), &y, None);
        let (_, py) = text_position(&built.unwrap());
        assert!((py - 207.5).abs() < 1e-9, "band c centre is 207.5, got {py}");
    }

    #[test]
    fn a_span_covers_whole_bands_in_either_order() {
        let x = band(&["a", "b", "c", "d"], [50.0, 450.0], 0.0);
        for (start, end) in [("b", "c"), ("c", "b")] {
            let (built, _) = run(&[span("x", category(start), category(end))], &x, &linear(), None);
            let (rx, ry, rw, rh) = rect_extent_of(&built.unwrap());
            assert!((rx - 150.0).abs() < 1e-9 && (rw - 200.0).abs() < 1e-9, "{rx} {rw}");
            assert!((ry - PLOT.y).abs() < 1e-9 && (rh - PLOT.h).abs() < 1e-9);
        }
    }

    #[test]
    fn a_span_over_one_category_shades_that_whole_band() {
        let x = band(&["a", "b", "c", "d"], [50.0, 450.0], 0.0);
        let (built, _) = run(&[span("x", category("b"), category("b"))], &x, &linear(), None);
        let (rx, _, rw, _) = rect_extent_of(&built.unwrap());
        assert!((rx - 150.0).abs() < 1e-9 && (rw - 100.0).abs() < 1e-9, "{rx} {rw}");
    }

    #[test]
    fn a_span_on_the_vertical_axis_covers_whole_bands() {
        let y = band(&["a", "b", "c", "d"], [20.0, 320.0], 0.0);
        let (built, _) = run(&[span("y", category("b"), category("c"))], &linear(), &y, None);
        let (_, ry, _, rh) = rect_extent_of(&built.unwrap());
        assert!((ry - 95.0).abs() < 1e-9 && (rh - 150.0).abs() < 1e-9, "{ry} {rh}");
    }

    #[test]
    fn a_span_band_is_the_scales_bandwidth_under_padding() {
        let x = band(&["a", "b", "c", "d"], [50.0, 450.0], 0.5);
        let ScaleKind::Ordinal(scale) = &x else { return };
        let centre = scale.scale_internal("b").unwrap_or(f64::NAN);
        let half = scale.bandwidth() / 2.0;
        let (built, _) = run(&[span("x", category("b"), category("b"))], &x, &linear(), None);
        let (rx, _, rw, _) = rect_extent_of(&built.unwrap());
        assert!((rx - (centre - half)).abs() < 1e-9 && (rw - 2.0 * half).abs() < 1e-9);
        assert!(rw < 100.0, "padding narrows the band: {rw}");
    }

    #[test]
    fn a_span_follows_a_reversed_range() {
        let x = band(&["a", "b", "c", "d"], [450.0, 50.0], 0.0);
        let (built, _) = run(&[span("x", category("a"), category("a"))], &x, &linear(), None);
        let (rx, _, rw, _) = rect_extent_of(&built.unwrap());
        assert!((rx - 350.0).abs() < 1e-9 && (rw - 100.0).abs() < 1e-9, "{rx} {rw}");
    }

    #[test]
    fn a_category_on_a_continuous_axis_is_refused_naming_the_axis() {
        let y = band(&["a", "b"], [20.0, 320.0], 0.0);
        let (x_case, _) = run(&[text(category("b"), CoordValue::Data(1.0))], &linear(), &y, None);
        assert_eq!(
            x_case.unwrap_err().to_string(),
            "annotation category 'b' on the x axis: the x scale is continuous, so it holds no \
             categories; pass a number, a date, an ISO-8601 string, fm.NormCoord or fm.PixelCoord"
        );
        let (y_case, _) = run(&[text(CoordValue::Data(1.0), category("b"))], &linear(), &linear(), None);
        assert!(y_case.unwrap_err().to_string().starts_with("annotation category 'b' on the y axis:"));
    }

    #[test]
    fn a_span_category_on_a_continuous_axis_is_refused() {
        let (built, _) = run(&[span("x", category("b"), category("c"))], &linear(), &linear(), None);
        assert!(matches!(
            built,
            Err(AnnotationRefusal::CategoryOnContinuousAxis { axis: AlignAxis::X, .. })
        ));
    }

    #[test]
    fn a_category_under_a_projected_coord_is_refused_before_the_axis_kind_is_read() {
        // A linear axis on both x and y: were the axis-kind check first, the
        // continuous-axis text would win here.
        for (json, coord) in [(r#"{"kind": "polar"}"#, "CoordPolar"), (r#"{"kind": "geo"}"#, "CoordGeo")] {
            let kind = coord_kind(json);
            for (x, y, axis) in [
                (category("b"), CoordValue::Data(1.0), "x"),
                (CoordValue::Data(1.0), category("b"), "y"),
            ] {
                let (built, _) = run(&[text(x, y)], &linear(), &linear(), Some(&kind));
                assert_eq!(
                    built.unwrap_err().to_string(),
                    format!(
                        "annotation category 'b' on the {axis} axis: a category cannot be placed \
                         under {coord}; pass fm.NormCoord or fm.PixelCoord"
                    )
                );
            }
        }
        // The named remedy renders under the same coord.
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let (built, _) = run(
            &[text(CoordValue::Norm { norm: 0.5 }, CoordValue::Pixel { px: 40.0 })],
            &x,
            &linear(),
            Some(&coord_kind(r#"{"kind": "polar"}"#)),
        );
        assert!(built.is_ok());
    }

    #[test]
    fn a_mixed_span_on_a_category_axis_is_refused_naming_its_category_end() {
        let x = band(&["a", "b", "c"], [50.0, 450.0], 0.0);
        for (start, end, named, other) in [
            (category("b"), CoordValue::Norm { norm: 0.5 }, "start", "end"),
            (CoordValue::Pixel { px: 40.0 }, category("b"), "end", "start"),
        ] {
            let (built, _) = run(&[span("x", start, end)], &x, &linear(), None);
            assert_eq!(
                built.unwrap_err().to_string(),
                format!(
                    "annotation span on the x axis: {named} 'b' is a category and {other} is not; \
                     pass categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
                )
            );
        }
        let (y_case, _) = run(
            &[span("y", category("b"), CoordValue::Norm { norm: 0.5 })],
            &linear(),
            &band(&["a", "b"], [20.0, 320.0], 0.0),
            None,
        );
        assert_eq!(
            y_case.unwrap_err().to_string(),
            "annotation span on the y axis: start 'b' is a category and end is not; pass \
             categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
        );
    }

    #[test]
    fn a_mixed_span_is_refused_for_its_coord_or_axis_before_its_shape() {
        let (built, _) = run(&[span("x", category("b"), CoordValue::Data(2.5))], &linear(), &linear(), None);
        assert!(matches!(
            built,
            Err(AnnotationRefusal::CategoryOnContinuousAxis { axis: AlignAxis::X, .. })
        ));
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let (built, _) = run(&[span("x", category("b"), CoordValue::Data(2.5))], &x, &linear(), Some(&coord_kind(r#"{"kind": "polar"}"#)));
        assert!(matches!(built, Err(AnnotationRefusal::CategoryUnderCoord { .. })));
    }

    #[test]
    fn an_unknown_category_warns_once_with_the_axis_letter_and_is_not_drawn() {
        let x = band(&["b", "a", "c", "d"], [50.0, 450.0], 0.0);
        let specs = [
            text(category("zz"), CoordValue::Data(1.0)),
            text(category("zz"), CoordValue::Data(2.0)),
        ];
        let (built, warnings) = run(&specs, &x, &linear(), None);
        let built = built.unwrap();
        assert!(built.above_marks.is_empty() && built.below_marks.is_empty(), "{built:?}");
        let shown: Vec<String> = warnings.iter().map(ToString::to_string).collect();
        assert_eq!(
            shown,
            ["annotation category 'zz' not found in x-axis domain ['a', 'b', 'c', 'd']; the annotation is not drawn"]
        );
    }

    #[test]
    fn an_unknown_span_or_callout_position_names_the_axis_not_the_field() {
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let callout = AnnotationSpec::Callout {
            x: category("a"),
            y: CoordValue::Data(1.0),
            text: "n".to_string(),
            arrow: "straight".to_string(),
            padding: 4.0,
            background: Some("#ffffff".to_string()),
            border_color: Some("#333333".to_string()),
            border_radius: 0.0,
            text_x: Some(category("zz")),
            text_y: None,
        };
        for spec in [span("x", category("a"), category("zz")), callout] {
            let (_, warnings) = run(&[spec], &x, &linear(), None);
            assert_eq!(warnings.len(), 1);
            assert!(warnings[0].to_string().contains("not found in x-axis domain ['a', 'b']"));
        }
    }

    #[test]
    fn a_span_with_an_unknown_end_is_not_drawn_and_a_known_neighbour_still_draws() {
        let x = band(&["a", "b", "c", "d"], [50.0, 450.0], 0.0);
        let specs = [
            span("x", category("a"), category("zz")),
            span("x", category("b"), category("b")),
            span("x", category("yy"), category("zz")),
        ];
        let (built, warnings) = run(&specs, &x, &linear(), None);
        let built = built.unwrap();
        assert_eq!(built.above_marks.len(), 1, "only the known span draws: {built:?}");
        let (rx, _, rw, _) = rect_extent_of(&built);
        assert!((rx - 150.0).abs() < 1e-9 && (rw - 100.0).abs() < 1e-9, "{rx} {rw}");
        let shown: Vec<String> = warnings.iter().map(ToString::to_string).collect();
        assert_eq!(shown.len(), 2, "one warning per label, 'zz' once: {shown:?}");
    }

    #[test]
    fn a_known_category_warns_nothing() {
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let (_, warnings) = run(&[text(category("a"), CoordValue::Data(1.0))], &x, &linear(), None);
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_norm_coordinate_on_a_band_axis_and_a_number_on_a_continuous_axis_keep_their_placement() {
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let (built, warnings) = run(&[text(CoordValue::Norm { norm: 0.5 }, CoordValue::Data(50.0))], &x, &linear(), None);
        let (px, py) = text_position(&built.unwrap());
        // Norm 0.5 on the band axis is the plot centre; the linear y maps 50 of [0, 100].
        assert!((px - 250.0).abs() < 1e-9, "x {px}");
        assert!((py - 300.0).abs() < 1e-9, "y {py}");
        assert!(warnings.is_empty());
    }

    fn value_message(value: &str, axis: &str, label: Option<&str>) -> String {
        let clause = label.map_or("a category label".to_string(), |l| format!("the category label '{l}'"));
        format!(
            "annotation value {value} on the {axis} axis: the {axis} scale is categorical, so it places \
             categories, not numbers or dates; pass {clause}, fm.NormCoord or fm.PixelCoord"
        )
    }

    fn instant(ms: f64) -> CoordValue {
        CoordValue::Instant { instant: ms, text: None }
    }

    const JAN_1: f64 = 1_577_836_800_000.0;
    const JAN_2: f64 = 1_577_923_200_000.0;
    const NOON: f64 = 1_577_880_000_000.0;

    fn day_axis(labels: &[&str]) -> ScaleKind {
        match band(labels, [50.0, 450.0], 0.0) {
            ScaleKind::Ordinal(ordinal) => ScaleKind::Ordinal(ordinal.with_of_dates(true)),
            other => other,
        }
    }

    #[test]
    fn a_number_on_a_band_axis_is_refused_naming_the_label_it_spells() {
        let x = band(&["a", "b", "c", "d"], [50.0, 450.0], 0.0);
        let years = band(&["2020", "2021"], [50.0, 450.0], 0.0);
        for (axis, value, label, expected) in [
            (&x, 2.0, None, "2"),
            (&x, 2.5, None, "2.5"),
            (&years, 2021.0, Some("2021"), "2021"),
        ] {
            let (built, _) = run(&[text(CoordValue::Data(value), CoordValue::Data(1.0))], axis, &linear(), None);
            assert_eq!(built.unwrap_err().to_string(), value_message(expected, "x", label));
        }
        let (built, _) = run(&[text(CoordValue::Data(1.0), CoordValue::Data(2.0))], &linear(), &x, None);
        assert_eq!(built.unwrap_err().to_string(), value_message("2", "y", None));
    }

    #[test]
    fn a_number_on_a_band_axis_is_refused_for_every_kind_and_end() {
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        for spec in [
            span("x", category("a"), CoordValue::Data(1.0)),
            span("x", CoordValue::Data(1.0), CoordValue::Data(2.0)),
            self::spec(r#"{"type":"rect","x1":{"category":"a"},"y1":1.0,"x2":9.0,"y2":2.0}"#),
            self::spec(r#"{"type":"line","x1":3.0,"y1":1.0,"x2":{"category":"a"},"y2":2.0}"#),
        ] {
            let (built, _) = run(std::slice::from_ref(&spec), &x, &linear(), None);
            assert!(
                matches!(built, Err(AnnotationRefusal::ValueOnCategoricalAxis { axis: AlignAxis::X, .. })),
                "{spec:?}"
            );
        }
    }

    #[test]
    fn a_flipped_number_refusal_names_the_callers_axis_letter() {
        let y = band(&["a", "b"], [20.0, 320.0], 0.0);
        let (built, _) = run_flipped(&[text(CoordValue::Data(1.0), CoordValue::Data(1.0))], &linear(), &y);
        assert_eq!(built.unwrap_err().to_string(), value_message("1", "x", None));
    }

    #[test]
    fn the_instant_wire_form_parses_and_a_bare_number_stays_a_number() {
        let v: CoordValue = serde_json::from_str(r#"{"instant": 1577836800000}"#).unwrap();
        assert!(matches!(v, CoordValue::Instant { instant, text: None } if instant == JAN_1));
        let t: CoordValue = serde_json::from_str(r#"{"instant": 1577923200000, "text": "2020-01-02"}"#).unwrap();
        assert!(matches!(&t, CoordValue::Instant { text: Some(text), .. } if text == "2020-01-02"));
        let n: CoordValue = serde_json::from_str("1577836800000.0").unwrap();
        assert!(matches!(n, CoordValue::Data(_)));
    }

    #[test]
    fn an_instant_on_a_continuous_axis_places_as_the_number() {
        let (built, _) = run(&[text(instant(30.0), instant(60.0))], &linear(), &linear(), None);
        let (twin, _) = run(&[text(CoordValue::Data(30.0), CoordValue::Data(60.0))], &linear(), &linear(), None);
        assert_eq!(text_position(&built.unwrap()), text_position(&twin.unwrap()));
    }

    #[test]
    fn an_instant_on_a_date_axis_lands_on_its_band_and_spans_whole_bands() {
        let x = day_axis(&["2020-01-01", "2020-01-02", "2020-01-03"]);
        let (built, warnings) = run(&[text(instant(JAN_2), CoordValue::Data(1.0))], &x, &linear(), None);
        let (px, _) = text_position(&built.unwrap());
        let band_w = 400.0 / 3.0;
        assert!((px - (50.0 + 1.5 * band_w)).abs() < 1e-6, "second of three bands, got {px}");
        assert!(warnings.is_empty());
        let (built, _) = run(&[span("x", instant(JAN_1), instant(JAN_2))], &x, &linear(), None);
        let (rx, _, rw, _) = rect_extent_of(&built.unwrap());
        assert!((rx - 50.0).abs() < 1e-6 && (rw - 2.0 * band_w).abs() < 1e-6, "{rx} {rw}");
    }

    #[test]
    fn an_instant_absent_from_a_date_axis_is_not_drawn_and_warns_with_the_date_labels() {
        let x = day_axis(&["2020-01-01", "2020-01-02"]);
        let (built, warnings) = run(&[text(instant(NOON), CoordValue::Data(1.0))], &x, &linear(), None);
        assert!(built.unwrap().above_marks.is_empty());
        let shown: Vec<String> = warnings.iter().map(ToString::to_string).collect();
        assert_eq!(
            shown,
            ["annotation category '2020-01-01T12:00:00' not found in x-axis domain ['2020-01-01', '2020-01-02']; \
              the annotation is not drawn"]
        );
        let noon = day_axis(&["2020-01-01T12:00:00"]);
        let (built, _) = run(&[text(instant(NOON), CoordValue::Data(1.0))], &noon, &linear(), None);
        assert_eq!(built.unwrap().above_marks.len(), 1);
    }

    #[test]
    fn a_number_on_a_date_axis_and_an_instant_on_a_plain_band_axis_are_refused_without_a_label() {
        let dates = day_axis(&["2020-01-01", "2020-01-02"]);
        let (built, _) = run(&[text(CoordValue::Data(3.0), CoordValue::Data(1.0))], &dates, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("3", "x", None));
        let plain = band(&["a", "b"], [50.0, 450.0], 0.0);
        let (built, _) = run(&[text(instant(JAN_2), CoordValue::Data(1.0))], &plain, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("2020-01-02", "x", None));
        let (built, _) = run(&[text(instant(NOON), CoordValue::Data(1.0))], &plain, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("2020-01-01T12:00:00", "x", None));
    }

    #[test]
    fn an_unknown_category_is_skipped_on_the_break_path_too() {
        let panel = broken_panel(&[[300.0, 400.0]]);
        let x = band(&["a", "b"], [50.0, 550.0], 0.0);
        let y = panel.remap(AlignAxis::Y);
        let ctx = ScaleContext {
            plot_area: panel.plot_area,
            x_scale: &x,
            y_scale: &panel.y_scale,
            breaks: std::slice::from_ref(&y),
            theme: &ThemeInputs::default(),
        };
        let mut warnings = Vec::new();
        let specs = [text(category("zz"), CoordValue::Data(100.0)), text(category("a"), CoordValue::Data(100.0))];
        let nodes = build_annotations_through_breaks(&specs, &ctx, None, false, &mut warnings).unwrap();
        assert_eq!(nodes.above_marks.len(), 1, "{nodes:?}");
        assert_eq!(warnings.len(), 1);
        let refused = build_annotations_through_breaks(
            &[text(CoordValue::Data(1.0), CoordValue::Data(100.0))],
            &ctx,
            None,
            false,
            &mut warnings,
        );
        assert!(matches!(refused, Err(AnnotationRefusal::ValueOnCategoricalAxis { .. })));
    }

    fn polar() -> CoordKind {
        coord_kind(r#"{"kind": "polar"}"#)
    }

    #[test]
    fn an_instant_on_a_date_axis_is_its_category_for_the_coord_and_mixed_span_checks() {
        let x = day_axis(&["2020-01-01", "2020-01-02", "2020-01-03"]);
        // Under a projected coord: present and absent dates are refused as categories.
        for (ms, label) in [(JAN_2, "2020-01-02"), (JAN_2 + 7.0 * 86_400_000.0, "2020-01-09")] {
            let (built, _) = run(&[text(instant(ms), CoordValue::Data(1.0))], &x, &linear(), Some(&polar()));
            assert_eq!(
                built.unwrap_err().to_string(),
                format!(
                    "annotation category '{label}' on the x axis: a category cannot be placed under \
                     CoordPolar; pass fm.NormCoord or fm.PixelCoord"
                )
            );
        }
        // Mixed span: an instant end beside a position end is a category end.
        let (built, _) = run(&[span("x", instant(JAN_2), CoordValue::Norm { norm: 1.0 })], &x, &linear(), None);
        assert_eq!(
            built.unwrap_err().to_string(),
            "annotation span on the x axis: start '2020-01-02' is a category and end is not; pass \
             categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
        );
        let (built, _) = run(&[span("x", CoordValue::Pixel { px: 5.0 }, instant(JAN_2))], &x, &linear(), None);
        assert_eq!(
            built.unwrap_err().to_string(),
            "annotation span on the x axis: end '2020-01-02' is a category and start is not; pass \
             categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
        );
        // An instant beside an unknown category: skipped and warned, not refused.
        let (built, warnings) = run(&[span("x", instant(JAN_2), category("zz"))], &x, &linear(), None);
        assert!(built.unwrap().above_marks.is_empty());
        let shown: Vec<String> = warnings.iter().map(ToString::to_string).collect();
        assert_eq!(
            shown,
            ["annotation category 'zz' not found in x-axis domain ['2020-01-01', '2020-01-02', '2020-01-03']; \
              the annotation is not drawn"]
        );
        // A category beside an instant: whole bands of both.
        let (built, warnings) = run(&[span("x", category("2020-01-01"), instant(JAN_2))], &x, &linear(), None);
        let (rx, _, rw, _) = rect_extent_of(&built.unwrap());
        assert!(warnings.is_empty());
        assert!((rx - 50.0).abs() < 1e-6 && (rw - 2.0 * 400.0 / 3.0).abs() < 1e-6, "{rx} {rw}");
    }

    #[test]
    fn a_value_under_a_projected_coord_names_only_the_position_remedies() {
        let years = band(&["2020", "2021"], [50.0, 450.0], 0.0);
        let (built, _) = run(&[text(CoordValue::Data(2021.0), CoordValue::Data(1.0))], &years, &linear(), Some(&polar()));
        let err = built.unwrap_err();
        assert!(matches!(err, AnnotationRefusal::ValueUnderCoord { axis: AlignAxis::X, coord: "CoordPolar", .. }));
        assert_eq!(
            err.to_string(),
            "annotation value 2021 on the x axis: the x scale is categorical, so it places categories, \
             not numbers or dates, and a category cannot be placed under CoordPolar; pass fm.NormCoord \
             or fm.PixelCoord"
        );
        let (built, _) = run(&[text(CoordValue::Data(2021.0), CoordValue::Data(1.0))], &years, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("2021", "x", Some("2021")));
        let plain = band(&["a", "b"], [50.0, 450.0], 0.0);
        let (built, _) = run(&[text(instant(JAN_2), CoordValue::Data(1.0))], &plain, &linear(), Some(&polar()));
        assert_eq!(
            built.unwrap_err().to_string(),
            "annotation value 2020-01-02 on the x axis: the x scale is categorical, so it places \
             categories, not numbers or dates, and a category cannot be placed under CoordPolar; pass \
             fm.NormCoord or fm.PixelCoord"
        );
        // A category written after a number is still named first.
        let (built, _) = run(&[span("x", CoordValue::Data(2.5), category("b"))], &plain, &linear(), Some(&polar()));
        assert!(matches!(built, Err(AnnotationRefusal::CategoryUnderCoord { .. })));
    }

    #[test]
    fn a_line_its_helper_wrote_is_refused_in_the_helpers_voice() {
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let line = |entry: Option<&str>| AnnotationSpec::Line {
            x1: CoordValue::Data(-1_000_000.0),
            y1: CoordValue::Data(0.0),
            x2: CoordValue::Data(1_000_000.0),
            y2: CoordValue::Data(1.0),
            stroke: None,
            stroke_width: 1.5,
            dash: None,
            entry: entry.map(str::to_string),
        };
        let (built, _) = run(&[line(Some("annotate_abline"))], &x, &linear(), None);
        assert_eq!(
            built.unwrap_err().to_string(),
            "annotate_abline: the x scale is categorical, so it places categories, not numbers or dates, \
             and annotate_abline takes no category; draw it on a chart whose x and y scales are continuous"
        );
        // Under a projected coord the helper's voice still wins.
        let (built, _) = run(&[line(Some("annotate_abline"))], &x, &linear(), Some(&polar()));
        assert!(matches!(built, Err(AnnotationRefusal::HelperOnCategoricalAxis { .. })));
        // The caller's letter under a flip.
        let y = band(&["a", "b"], [20.0, 320.0], 0.0);
        let (built, _) = run_flipped(&[line(Some("annotate_abline"))], &linear(), &y);
        assert!(matches!(built, Err(AnnotationRefusal::HelperOnCategoricalAxis { axis: AlignAxis::X, .. })));
        // Without an entry the value voice stays.
        let (built, _) = run(&[line(None)], &x, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("-1000000", "x", None));
    }

    fn text_instant(ms: f64, text: &str) -> CoordValue {
        CoordValue::Instant { instant: ms, text: Some(text.to_string()) }
    }

    #[test]
    fn a_text_instant_names_the_string_category_it_spells() {
        let x = band(&["2020-01-01", "2020-01-02", "2020-01-03"], [50.0, 450.0], 0.0);
        let band_w = 400.0 / 3.0;
        // The written text lands on its band, as the category does.
        let (built, warnings) = run(&[text(text_instant(JAN_2, "2020-01-02"), CoordValue::Data(1.0))], &x, &linear(), None);
        let (px, _) = text_position(&built.unwrap());
        let (twin, _) = run(&[text(category("2020-01-02"), CoordValue::Data(1.0))], &x, &linear(), None);
        assert_eq!(px, text_position(&twin.unwrap()).0);
        assert!((px - (50.0 + 1.5 * band_w)).abs() < 1e-6, "{px}");
        assert!(warnings.is_empty());
        // A whole-band span between two written texts.
        let (built, _) = run(&[span("x", text_instant(JAN_1, "2020-01-01"), text_instant(JAN_2, "2020-01-02"))], &x, &linear(), None);
        let (rx, _, rw, _) = rect_extent_of(&built.unwrap());
        assert!((rx - 50.0).abs() < 1e-6 && (rw - 2.0 * band_w).abs() < 1e-6, "{rx} {rw}");
        // Text that is not a label is refused quoting the text as written.
        let (built, _) = run(&[text(text_instant(JAN_2, "2020-01-02T00:00:00"), CoordValue::Data(1.0))], &x, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("2020-01-02T00:00:00", "x", None));
        // The same instant without text is a date object: refused as a date.
        let (built, _) = run(&[text(instant(JAN_2), CoordValue::Data(1.0))], &x, &linear(), None);
        assert_eq!(built.unwrap_err().to_string(), value_message("2020-01-02", "x", None));
        // Under a projected coord the text is the category refused.
        let (built, _) = run(&[text(text_instant(JAN_2, "2020-01-02"), CoordValue::Data(1.0))], &x, &linear(), Some(&polar()));
        assert!(matches!(built, Err(AnnotationRefusal::CategoryUnderCoord { ref label, .. }) if label == "2020-01-02"));
        // Beside a position end it is a category end of a mixed span.
        let (built, _) = run(&[span("x", text_instant(JAN_2, "2020-01-02"), CoordValue::Norm { norm: 0.9 })], &x, &linear(), None);
        assert_eq!(
            built.unwrap_err().to_string(),
            "annotation span on the x axis: start '2020-01-02' is a category and end is not; pass \
             categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
        );
    }

    #[test]
    fn on_a_date_axis_the_instant_wins_over_its_text_and_continuous_axes_ignore_it() {
        let dates = day_axis(&["2020-01-01", "2020-01-02"]);
        let (a, _) = run(&[text(text_instant(JAN_2, "whatever"), CoordValue::Data(1.0))], &dates, &linear(), None);
        let (b, _) = run(&[text(instant(JAN_2), CoordValue::Data(1.0))], &dates, &linear(), None);
        assert_eq!(text_position(&a.unwrap()), text_position(&b.unwrap()));
        let (a, _) = run(&[text(text_instant(30.0, "x"), CoordValue::Data(1.0))], &linear(), &linear(), None);
        let (b, _) = run(&[text(instant(30.0), CoordValue::Data(1.0))], &linear(), &linear(), None);
        assert_eq!(text_position(&a.unwrap()), text_position(&b.unwrap()));
    }

    #[test]
    fn a_flipped_instant_on_a_date_axis_lands_on_the_drawn_vertical_band() {
        let y = day_axis(&["2020-01-01", "2020-01-02"]);
        let (built, _) = run_flipped(&[text(instant(JAN_2), CoordValue::Data(1.0))], &linear(), &y);
        let (_, py) = text_position(&built.unwrap());
        // The y band axis here spans [50, 450] with two bands: the second centres at 350.
        assert!((py - 350.0).abs() < 1e-9, "got {py}");
    }

    #[test]
    fn a_category_is_refused_only_under_a_projected_coord() {
        let x = band(&["a", "b"], [50.0, 450.0], 0.0);
        let spec = [text(category("b"), CoordValue::Data(1.0))];
        for json in [r#"{"kind": "cartesian"}"#, r#"{"kind": "fixed"}"#, r#"{"kind": "flip"}"#] {
            let (built, _) = run(&spec, &x, &linear(), Some(&coord_kind(json)));
            assert!(built.is_ok(), "{json} must place a category");
        }
        let (built, _) = run(&spec, &x, &linear(), None);
        assert!(built.is_ok());
        for json in [r#"{"kind": "polar"}"#, r#"{"kind": "geo"}"#] {
            let (built, _) = run(&spec, &x, &linear(), Some(&coord_kind(json)));
            assert!(
                matches!(built, Err(AnnotationRefusal::CategoryUnderCoord { .. })),
                "{json} must refuse a category"
            );
        }
    }

    fn coord_kind(json: &str) -> CoordKind {
        serde_json::from_str(json).unwrap()
    }

    fn run_flipped(
        specs: &[AnnotationSpec],
        x: &ScaleKind,
        y: &ScaleKind,
    ) -> (Result<AnnotationNodes, AnnotationRefusal>, Vec<RenderWarning>) {
        let ctx = ScaleContext { plot_area: PLOT, x_scale: x, y_scale: y, breaks: &[], theme: &ThemeInputs::default() };
        let mut warnings = Vec::new();
        let built = build_annotations_through_breaks(specs, &ctx, None, true, &mut warnings);
        (built, warnings)
    }

    fn data(v: f64) -> CoordValue {
        CoordValue::Data(v)
    }

    fn shown(nodes: &AnnotationNodes) -> String {
        format!("{:?}", nodes.above_marks)
    }

    /// Both axes linear and distinct, so an exchange is visible in the pixels.
    fn unequal_axes() -> (ScaleKind, ScaleKind) {
        let (x, y, _) = test_ctx();
        (x, y)
    }

    #[test]
    fn a_flipped_text_places_the_callers_x_on_the_drawn_vertical_axis() {
        let (x, y) = unequal_axes();
        let ctx = ScaleContext { plot_area: PLOT, x_scale: &x, y_scale: &y, breaks: &[], theme: &ThemeInputs::default() };
        let (built, _) = run_flipped(&[text(data(30.0), data(0.5))], &x, &y);
        let (px, py) = text_position(&built.unwrap());
        assert!((px - ctx.resolve_x(&data(0.5))).abs() < 1e-9, "x {px}");
        assert!((py - ctx.resolve_y(&data(30.0))).abs() < 1e-9, "y {py}");
        let (plain, _) = run(&[text(data(30.0), data(0.5))], &x, &y, None);
        assert_ne!(text_position(&plain.unwrap()), (px, py), "the exchange moves the text");
    }

    #[test]
    fn a_flipped_text_keeps_its_offsets_and_anchor_in_drawn_space() {
        let (x, y) = unequal_axes();
        let mut spec = text(data(30.0), data(0.5));
        if let AnnotationSpec::Text { dx, dy, anchor, .. } = &mut spec {
            *dx = 7.0;
            *dy = -5.0;
            *anchor = "end".to_string();
        }
        let exchanged = spec.exchanged();
        let AnnotationSpec::Text { dx, dy, anchor, .. } = &exchanged else {
            panic!("a text spec stays a text spec under the exchange, got {exchanged:?}");
        };
        assert_eq!((*dx, *dy, anchor.as_str()), (7.0, -5.0, "end"));
        let (flipped, _) = run_flipped(&[spec], &x, &y);
        let (px, py) = text_position(&flipped.unwrap());
        let ctx = ScaleContext { plot_area: PLOT, x_scale: &x, y_scale: &y, breaks: &[], theme: &ThemeInputs::default() };
        assert!((px - (ctx.resolve_x(&data(0.5)) + 7.0)).abs() < 1e-9);
        assert!((py - (ctx.resolve_y(&data(30.0)) - 5.0)).abs() < 1e-9);
    }

    #[test]
    fn a_flipped_category_x_lands_on_the_band_of_the_drawn_vertical_axis() {
        let y = band(&["a", "b", "c", "d"], [20.0, 320.0], 0.0);
        let (built, warnings) = run_flipped(&[text(category("b"), data(2.0))], &linear(), &y);
        let (_, py) = text_position(&built.unwrap());
        assert!((py - 132.5).abs() < 1e-9, "band b centre is 132.5, got {py}");
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_flipped_category_span_shades_whole_bands_across_the_plot_width() {
        let y = band(&["a", "b", "c", "d"], [20.0, 320.0], 0.0);
        let (built, _) = run_flipped(&[span("x", category("b"), category("c"))], &linear(), &y);
        let (rx, ry, rw, rh) = rect_extent_of(&built.unwrap());
        assert!((ry - 95.0).abs() < 1e-9 && (rh - 150.0).abs() < 1e-9, "{ry} {rh}");
        assert!((rx - PLOT.x).abs() < 1e-9 && (rw - PLOT.w).abs() < 1e-9, "{rx} {rw}");
    }

    #[test]
    fn a_flipped_numeric_span_runs_along_the_drawn_vertical_axis() {
        let (x, y) = unequal_axes();
        let ctx = ScaleContext { plot_area: PLOT, x_scale: &x, y_scale: &y, breaks: &[], theme: &ThemeInputs::default() };
        let (built, _) = run_flipped(&[span("x", data(20.0), data(40.0))], &x, &y);
        let (rx, ry, rw, rh) = rect_extent_of(&built.unwrap());
        let (a, b) = (ctx.resolve_y(&data(20.0)), ctx.resolve_y(&data(40.0)));
        assert!((ry - a.min(b)).abs() < 1e-9 && (rh - (a - b).abs()).abs() < 1e-9, "{ry} {rh}");
        assert!((rx - PLOT.x).abs() < 1e-9 && (rw - PLOT.w).abs() < 1e-9, "{rx} {rw}");
    }

    #[test]
    fn a_flipped_norm_and_pixel_measure_along_the_drawn_axis_their_field_lands_on() {
        let (x, y) = unequal_axes();
        let ctx = ScaleContext { plot_area: PLOT, x_scale: &x, y_scale: &y, breaks: &[], theme: &ThemeInputs::default() };
        let (built, _) = run_flipped(&[text(CoordValue::Norm { norm: 0.1 }, data(0.5))], &x, &y);
        let (px, py) = text_position(&built.unwrap());
        assert!((py - (PLOT.y + 0.1 * PLOT.h)).abs() < 1e-9, "10% down from the top, got {py}");
        assert!((px - ctx.resolve_x(&data(0.5))).abs() < 1e-9, "caller's y on the horizontal axis");
        let (built, _) = run_flipped(&[text(CoordValue::Pixel { px: 25.0 }, data(0.5))], &x, &y);
        let (_, py) = text_position(&built.unwrap());
        assert!((py - (PLOT.y + 25.0)).abs() < 1e-9, "25 px below the top, got {py}");
    }

    fn callout(x: f64, y: f64, text_x: Option<f64>, text_y: Option<f64>) -> AnnotationSpec {
        AnnotationSpec::Callout {
            x: data(x),
            y: data(y),
            text: "n".to_string(),
            arrow: "straight".to_string(),
            padding: 4.0,
            background: Some("#ffffff".to_string()),
            border_color: Some("#333333".to_string()),
            border_radius: 0.0,
            text_x: text_x.map(data),
            text_y: text_y.map(data),
        }
    }

    fn arrow(x: f64, y: f64, x2: f64, y2: f64) -> AnnotationSpec {
        AnnotationSpec::Arrow {
            x: data(x),
            y: data(y),
            x2: data(x2),
            y2: data(y2),
            stroke: Some("#000000".to_string()),
            stroke_width: 1.0,
            head_size: 6.0,
        }
    }

    fn rect(x1: f64, y1: f64, x2: f64, y2: f64) -> AnnotationSpec {
        AnnotationSpec::Rect {
            x1: data(x1),
            y1: data(y1),
            x2: data(x2),
            y2: data(y2),
            fill: Some("#cccccc".to_string()),
            opacity: 0.5,
            corner_radius: 0.0,
            stroke: None,
        }
    }

    fn line(x1: f64, y1: f64, x2: f64, y2: f64) -> AnnotationSpec {
        AnnotationSpec::Line {
            x1: data(x1),
            y1: data(y1),
            x2: data(x2),
            y2: data(y2),
            stroke: Some("#000000".to_string()),
            stroke_width: 1.0,
            dash: None,
            entry: None,
        }
    }

    fn bracket(x1: f64, y1: f64, x2: f64, y2: f64, direction: &str) -> AnnotationSpec {
        AnnotationSpec::Bracket {
            x1: data(x1),
            y1: data(y1),
            x2: data(x2),
            y2: data(y2),
            label: "b".to_string(),
            direction: direction.to_string(),
            stroke: Some("#000000".to_string()),
            tip_length: 4.0,
        }
    }

    fn image(x: f64, y: f64) -> AnnotationSpec {
        AnnotationSpec::Image {
            x: data(x),
            y: data(y),
            src: "data:,".to_string(),
            width: 10.0,
            height: 10.0,
            anchor: "top-right".to_string(),
        }
    }

    /// Each kind written by the caller beside the spec a hand exchange of its
    /// pairs gives, with four distinct values so one missed swap shows.
    fn caller_and_exchanged() -> Vec<(AnnotationSpec, AnnotationSpec)> {
        vec![
            (text(data(0.2), data(10.0)), text(data(10.0), data(0.2))),
            (image(0.2, 10.0), image(10.0, 0.2)),
            (arrow(0.2, 10.0, 0.8, 70.0), arrow(10.0, 0.2, 70.0, 0.8)),
            (rect(0.2, 10.0, 0.8, 70.0), rect(10.0, 0.2, 70.0, 0.8)),
            (line(0.2, 10.0, 0.8, 70.0), line(10.0, 0.2, 70.0, 0.8)),
            (bracket(0.2, 10.0, 0.8, 70.0, "up"), bracket(10.0, 0.2, 70.0, 0.8, "right")),
            (callout(0.2, 10.0, Some(0.5), Some(60.0)), callout(10.0, 0.2, Some(60.0), Some(0.5))),
            (callout(0.2, 10.0, Some(0.5), None), callout(10.0, 0.2, None, Some(0.5))),
            (span("x", data(10.0), data(70.0)), span("y", data(10.0), data(70.0))),
            (span("y", data(10.0), data(70.0)), span("x", data(10.0), data(70.0))),
        ]
    }

    #[test]
    fn an_exchange_swaps_each_field_pair_of_every_kind_and_nothing_else() {
        for (caller, expected) in caller_and_exchanged() {
            assert_eq!(format!("{:?}", caller.exchanged()), format!("{expected:?}"));
        }
    }

    #[test]
    fn a_bracket_direction_is_transposed_with_its_pair() {
        for (written, drawn) in [
            ("up", "right"),
            ("down", "left"),
            ("left", "down"),
            ("right", "up"),
            ("above", "right"),
            ("below", "left"),
        ] {
            assert_eq!(
                format!("{:?}", bracket(0.2, 10.0, 0.8, 70.0, written).exchanged()),
                format!("{:?}", bracket(10.0, 0.2, 70.0, 0.8, drawn)),
                "direction {written}"
            );
        }
    }

    #[test]
    fn a_flipped_bracket_equals_the_unflipped_chart_with_axes_exchanged() {
        // The default bracket's tips and label sit off its baseline, never on it.
        let (x, y) = unequal_axes();
        let caller = bracket(0.2, 10.0, 0.8, 70.0, "up");
        let (flipped, _) = run_flipped(std::slice::from_ref(&caller), &x, &y);
        let (twin, _) = run(&[bracket(10.0, 0.2, 70.0, 0.8, "right")], &x, &y, None);
        assert_eq!(shown(&flipped.unwrap()), shown(&twin.unwrap()));
    }

    #[test]
    fn a_flipped_spec_of_every_kind_draws_the_hand_exchanged_spec_unflipped() {
        let (x, y) = unequal_axes();
        for (caller, expected) in caller_and_exchanged() {
            let (flipped, _) = run_flipped(std::slice::from_ref(&caller), &x, &y);
            let (hand, _) = run(&[expected], &x, &y, None);
            let (plain, _) = run(std::slice::from_ref(&caller), &x, &y, None);
            let (flipped, hand, plain) = (shown(&flipped.unwrap()), shown(&hand.unwrap()), shown(&plain.unwrap()));
            assert_eq!(flipped, hand);
            assert_ne!(flipped, plain, "{caller:?} must move under the exchange");
        }
    }

    #[test]
    fn an_unexchanged_build_is_the_same_nodes_as_before() {
        let (x, y) = unequal_axes();
        let specs = [text(data(30.0), data(0.5)), span("x", data(0.2), data(0.6))];
        let (a, _) = run(&specs, &x, &y, None);
        let ctx = ScaleContext { plot_area: PLOT, x_scale: &x, y_scale: &y, breaks: &[], theme: &ThemeInputs::default() };
        assert_eq!(shown(&a.unwrap()), shown(&build_annotations(&specs, &ctx).unwrap()));
    }

    #[test]
    fn a_flipped_refusal_names_the_callers_axis_letter() {
        // Caller's x is a category but the drawn vertical axis is continuous.
        let (built, _) = run_flipped(&[text(category("b"), data(1.0))], &linear(), &linear());
        assert!(built.unwrap_err().to_string().starts_with("annotation category 'b' on the x axis:"));
        // Caller's y is a category and the drawn horizontal axis is continuous.
        let (built, _) = run_flipped(&[text(data(1.0), category("b"))], &linear(), &linear());
        assert!(built.unwrap_err().to_string().starts_with("annotation category 'b' on the y axis:"));
        // Caller's y is a category and the drawn horizontal axis is continuous; the drawn vertical axis is a band.
        let band_y = band(&["a", "b"], [20.0, 320.0], 0.0);
        let (built, _) = run_flipped(&[text(data(1.0), category("b"))], &linear(), &band_y);
        assert!(built.unwrap_err().to_string().starts_with("annotation category 'b' on the y axis:"));
        let (built, _) = run_flipped(&[span("x", category("b"), CoordValue::Norm { norm: 0.5 })], &linear(), &band_y);
        assert_eq!(
            built.unwrap_err().to_string(),
            "annotation span on the x axis: start 'b' is a category and end is not; pass \
             categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
        );
        let (built, _) = run_flipped(&[text(category("b"), data(1.0))], &linear(), &linear());
        assert!(matches!(
            built,
            Err(AnnotationRefusal::CategoryOnContinuousAxis { axis: AlignAxis::X, .. })
        ));
    }

    #[test]
    fn a_flipped_projected_refusal_names_the_callers_axis_letter() {
        let ctx_x = linear();
        let ctx = ScaleContext { plot_area: PLOT, x_scale: &ctx_x, y_scale: &ctx_x, breaks: &[], theme: &ThemeInputs::default() };
        let mut warnings = Vec::new();
        let built = build_annotations_through_breaks(
            &[text(category("b"), data(1.0))],
            &ctx,
            Some(&coord_kind(r#"{"kind": "polar"}"#)),
            true,
            &mut warnings,
        );
        assert!(matches!(built, Err(AnnotationRefusal::CategoryUnderCoord { axis: AlignAxis::X, .. })));
    }

    #[test]
    fn a_flipped_unknown_category_warns_naming_the_callers_axis() {
        // The caller's x is categorical: it is drawn on the vertical axis.
        let y = band(&["b", "a"], [20.0, 320.0], 0.0);
        let (built, warnings) = run_flipped(&[text(category("zz"), data(1.0))], &linear(), &y);
        assert!(built.unwrap().above_marks.is_empty());
        let shown: Vec<String> = warnings.iter().map(ToString::to_string).collect();
        assert_eq!(
            shown,
            ["annotation category 'zz' not found in x-axis domain ['a', 'b']; the annotation is not drawn"]
        );
    }
}

// ── Unset colour is the theme's text colour ─────────────────────────────────

mod theme_ink {
    use super::*;

    const FONT: (u8, u8, u8) = (0x0a, 0x0b, 0x0c);

    fn themed() -> ThemeInputs {
        let mut theme = ThemeInputs::default();
        theme.colors.font_color = palette::Srgba::new(FONT.0, FONT.1, FONT.2, 0xFF);
        theme
    }

    fn build(json: &str, theme: &ThemeInputs) -> AnnotationNodes {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext { plot_area, x_scale: &x_scale, y_scale: &y_scale, breaks: &[], theme };
        build_annotations(&[spec(json)], &ctx).unwrap()
    }

    /// `json` with one more `"key":value` member appended to its outer object.
    fn extended(json: &str, member: &str) -> String {
        let body = json.strip_suffix('}').expect("an object");
        format!("{body},{member}}}")
    }

    /// Every stroke or text colour the nodes carry, in order.
    fn inks(nodes: &[SceneNode]) -> Vec<(u8, u8, u8)> {
        let rgb = |c: &Option<Color>| {
            let c = c.expect("every annotation node is coloured");
            (c.r, c.g, c.b)
        };
        nodes
            .iter()
            .filter_map(|n| match n {
                SceneNode::Line { style, .. } => Some(rgb(&style.color)),
                SceneNode::Text { style, .. } => Some(rgb(&style.color)),
                _ => None,
            })
            .collect()
    }

    const TEXT: &str = r#"{"type":"text","x":{"px":10},"y":{"px":10},"text":"t"}"#;
    const LINE: &str = r#"{"type":"line","x1":{"px":0},"y1":{"px":0},"x2":{"px":9},"y2":{"px":9}}"#;
    const ARROW: &str = r#"{"type":"arrow","x":{"px":0},"y":{"px":0},"x2":{"px":90},"y2":{"px":9}}"#;
    const BRACKET: &str =
        r#"{"type":"bracket","x1":{"px":0},"y1":{"px":50},"x2":{"px":90},"y2":{"px":50},"label":"n"}"#;
    const SPAN: &str = r#"{"type":"span","axis":"x","start":10.0,"end":20.0,"label":"s"}"#;

    #[test]
    fn an_absent_colour_deserializes_to_none_for_every_primitive() {
        for json in [TEXT, LINE, ARROW, BRACKET] {
            match spec(json) {
                AnnotationSpec::Text { color, .. } => assert_eq!(color, None),
                AnnotationSpec::Line { stroke, .. }
                | AnnotationSpec::Arrow { stroke, .. }
                | AnnotationSpec::Bracket { stroke, .. } => assert_eq!(stroke, None),
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn an_unset_colour_draws_the_theme_text_colour_on_every_primitive() {
        for json in [TEXT, LINE, ARROW, BRACKET, SPAN] {
            let nodes = build(json, &themed());
            let drawn = inks(&nodes.above_marks);
            assert!(!drawn.is_empty(), "{json} drew nothing coloured");
            assert!(drawn.iter().all(|c| *c == FONT), "{json} drew {drawn:?}, not the theme font colour");
        }
    }

    #[test]
    fn an_unset_colour_follows_the_theme_it_is_drawn_under() {
        let mut dark = ThemeInputs::default();
        dark.colors.font_color = palette::Srgba::new(0xe6, 0xe6, 0xe6, 0xFF);
        let light = inks(&build(LINE, &ThemeInputs::default()).above_marks);
        let on_dark = inks(&build(LINE, &dark).above_marks);
        assert_eq!(light, vec![(0x1f, 0x29, 0x37)]);
        assert_eq!(on_dark, vec![(0xe6, 0xe6, 0xe6)]);
    }

    #[test]
    fn an_explicit_colour_wins_over_the_theme() {
        let with = |json: &str, key: &str| extended(json, &format!(r##""{key}":"#ff0000""##));
        let text = build(&with(TEXT, "color"), &themed());
        assert_eq!(inks(&text.above_marks), vec![(255, 0, 0)]);
        for json in [LINE, ARROW] {
            let nodes = build(&with(json, "stroke"), &themed());
            assert!(inks(&nodes.above_marks).iter().all(|c| *c == (255, 0, 0)), "{json}");
        }
    }

    #[test]
    fn a_bracket_label_is_theme_ink_even_with_an_explicit_stroke() {
        let json = extended(BRACKET, r##""stroke":"#ff0000""##);
        let drawn = inks(&build(&json, &themed()).above_marks);
        // Baseline and two tips in the stroke, then the label.
        assert_eq!(drawn, vec![(255, 0, 0), (255, 0, 0), (255, 0, 0), FONT]);
    }

    #[test]
    fn a_below_marks_text_is_theme_ink_in_the_below_marks_slot() {
        let json = extended(TEXT, r#""z":"below_marks""#);
        let nodes = build(&json, &themed());
        assert!(nodes.above_marks.is_empty());
        assert_eq!(inks(&nodes.below_marks), vec![FONT]);
    }

    const BG: (u8, u8, u8) = (0x0d, 0x0e, 0x0f);
    const RECT: &str = r#"{"type":"rect","x1":{"px":10},"y1":{"px":10},"x2":{"px":60},"y2":{"px":40}}"#;
    const CALLOUT: &str = r#"{"type":"callout","x":{"px":10},"y":{"px":10},"text":"c"}"#;

    fn themed_with_background() -> ThemeInputs {
        let mut theme = themed();
        theme.colors.background_color = palette::Srgba::new(BG.0, BG.1, BG.2, 0xFF);
        theme
    }

    fn rgb(c: Option<Color>) -> Option<(u8, u8, u8)> {
        c.map(|c| (c.r, c.g, c.b))
    }

    /// The box's `(fill, stroke)`, the leader's colour and the text's colour.
    fn callout_paints(
        json: &str,
    ) -> (Option<(u8, u8, u8)>, Option<(u8, u8, u8)>, Option<(u8, u8, u8)>, Option<(u8, u8, u8)>) {
        let nodes = build(json, &themed_with_background()).above_marks;
        match nodes.as_slice() {
            [SceneNode::Rect { style, .. }, SceneNode::Line { style: leader, .. }, SceneNode::Text { style: text, .. }] => {
                (rgb(style.fill), rgb(style.stroke), rgb(leader.color), rgb(text.color))
            }
            other => panic!("expected box, leader, text; got {other:?}"),
        }
    }

    fn rect_style(nodes: &AnnotationNodes) -> &FillStroke {
        match nodes.above_marks.first() {
            Some(SceneNode::Rect { style, .. }) => style,
            other => panic!("expected a rect, got {other:?}"),
        }
    }

    #[test]
    fn an_unset_rect_and_span_fill_draw_the_theme_text_colour_at_their_default_opacity() {
        let rect = build(RECT, &themed());
        assert_eq!(rgb(rect_style(&rect).fill), Some(FONT));
        assert_eq!(rect_style(&rect).fill_opacity, 0.3);
        let span = build(SPAN, &themed());
        assert_eq!(rgb(rect_style(&span).fill), Some(FONT));
        assert_eq!(rect_style(&span).fill_opacity, 0.2);
    }

    #[test]
    fn an_unset_rect_fill_follows_the_theme_it_is_drawn_under() {
        let mut dark = ThemeInputs::default();
        dark.colors.font_color = palette::Srgba::new(0xe6, 0xe6, 0xe6, 0xFF);
        assert_eq!(rgb(rect_style(&build(RECT, &dark)).fill), Some((0xe6, 0xe6, 0xe6)));
        assert_eq!(rgb(rect_style(&build(RECT, &ThemeInputs::default())).fill), Some((0x1f, 0x29, 0x37)));
    }

    #[test]
    fn an_explicit_rect_and_span_fill_wins_over_the_theme() {
        for (json, opacity) in [(RECT, 0.3), (SPAN, 0.2)] {
            let nodes = build(&extended(json, r##""fill":"#ff0000""##), &themed());
            assert_eq!(rgb(rect_style(&nodes).fill), Some((255, 0, 0)), "{json}");
            assert_eq!(rect_style(&nodes).fill_opacity, opacity, "{json}");
        }
    }

    #[test]
    fn an_unstyled_callout_draws_the_theme_colours() {
        assert_eq!(callout_paints(CALLOUT), (Some(BG), Some(FONT), Some(FONT), Some(FONT)));
    }

    #[test]
    fn an_unstyled_callout_on_a_dark_theme_is_legible() {
        let mut dark = ThemeInputs::default();
        dark.colors.font_color = palette::Srgba::new(0xe6, 0xe6, 0xe6, 0xFF);
        dark.colors.background_color = palette::Srgba::new(0x11, 0x11, 0x11, 0xFF);
        let nodes = build(CALLOUT, &dark).above_marks;
        match nodes.as_slice() {
            [SceneNode::Rect { style, .. }, SceneNode::Line { .. }, SceneNode::Text { style: text, .. }] => {
                assert_eq!(rgb(style.fill), Some((0x11, 0x11, 0x11)));
                assert_eq!(rgb(text.color), Some((0xe6, 0xe6, 0xe6)));
            }
            other => panic!("expected box, leader, text; got {other:?}"),
        }
    }

    #[test]
    fn an_explicit_callout_colour_wins_on_its_own_field_only() {
        let border = extended(CALLOUT, r##""border_color":"#1a56db""##);
        assert_eq!(callout_paints(&border), (Some(BG), Some((0x1a, 0x56, 0xdb)), Some((0x1a, 0x56, 0xdb)), Some(FONT)));
        let background = extended(CALLOUT, r##""background":"#ffeecc""##);
        assert_eq!(callout_paints(&background), (Some((0xff, 0xee, 0xcc)), Some(FONT), Some(FONT), Some(FONT)));
        let both = extended(&border, r##""background":"#ffeecc""##);
        assert_eq!(
            callout_paints(&both),
            (Some((0xff, 0xee, 0xcc)), Some((0x1a, 0x56, 0xdb)), Some((0x1a, 0x56, 0xdb)), Some(FONT))
        );
    }

    #[test]
    fn a_paint_word_draws_no_paint_on_any_colour_field() {
        for word in ["none", "transparent", " None ", "TRANSPARENT"] {
            let with = |json: &str, key: &str| extended(json, &format!(r#""{key}":"{word}""#));
            let rect = build(&extended(&with(RECT, "fill"), &format!(r#""stroke":"{word}""#)), &themed());
            assert_eq!((rect_style(&rect).fill, rect_style(&rect).stroke), (None, None), "{word}");
            assert_eq!(rect_style(&build(&with(SPAN, "fill"), &themed())).fill, None, "{word}");
            let callout = extended(&with(CALLOUT, "background"), &format!(r#""border_color":"{word}""#));
            assert_eq!(callout_paints(&callout), (None, None, None, Some(FONT)), "{word}");
            let text = build(&with(TEXT, "color"), &themed()).above_marks;
            assert!(matches!(text.as_slice(), [SceneNode::Text { style, .. }] if style.color.is_none()), "{word}");
            for json in [LINE, ARROW] {
                let nodes = build(&with(json, "stroke"), &themed()).above_marks;
                for node in &nodes {
                    match node {
                        SceneNode::Line { style, .. } => assert_eq!(style.color, None, "{word} {json}"),
                        SceneNode::Path { style, .. } => assert_eq!(style.fill, None, "{word} {json}"),
                        other => panic!("unexpected {other:?}"),
                    }
                }
            }
            let bracket = build(&with(BRACKET, "stroke"), &themed()).above_marks;
            assert_eq!(
                bracket.iter().map(|n| match n {
                    SceneNode::Line { style, .. } => rgb(style.color),
                    SceneNode::Text { style, .. } => rgb(style.color),
                    other => panic!("unexpected {other:?}"),
                }).collect::<Vec<_>>(),
                vec![None, None, None, Some(FONT)],
                "{word}"
            );
        }
    }

    #[test]
    fn a_clipped_span_label_and_cut_pieces_are_theme_ink() {
        let panel = broken_panel(&[[400.0, 600.0]]);
        let y = panel.remap(AlignAxis::Y);
        let span = spec(r#"{"type":"span","axis":"y","start":0.0,"end":1000.0,"label":"band"}"#);
        let line = spec(r#"{"type":"line","x1":{"norm":0.0},"y1":0.0,"x2":{"norm":1.0},"y2":1000.0}"#);
        let (nodes, _) = panel.build(&[span, line], &[y]);
        let drawn = inks(&nodes);
        assert!(drawn.len() >= 2, "a label and at least one line piece: {drawn:?}");
        assert!(drawn.iter().all(|c| *c == (0x1f, 0x29, 0x37)), "{drawn:?}");
    }
}

/// A bad colour is refused whatever a break would do with the annotation: one
/// the gap drops whole and one it clips are both refused, as on a plain panel.
#[test]
fn an_unparseable_colour_is_refused_on_a_panel_with_breaks() {
    let panel = broken_panel(&[[150.0, 850.0]]);
    let y = panel.remap(AlignAxis::Y);
    let breaks = [y];
    let ctx = ScaleContext {
        plot_area: panel.plot_area,
        x_scale: &panel.x_scale,
        y_scale: &panel.y_scale,
        breaks: &breaks,
        theme: &ThemeInputs::default(),
    };
    for (y1, y2, how) in [(300.0, 700.0, "dropped whole"), (100.0, 400.0, "clipped")] {
        let json = format!(
            r#"{{"type":"rect","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2},"fill":"notacolor"}}"#
        );
        let mut warnings = Vec::new();
        let refused = build_annotations_through_breaks(&[spec(&json)], &ctx, None, false, &mut warnings)
            .expect_err(how);
        assert!(
            matches!(&refused, AnnotationRefusal::InvalidColor(c) if c.kind == "rect" && c.field == "fill"),
            "{how}: {refused:?}"
        );
    }
}
