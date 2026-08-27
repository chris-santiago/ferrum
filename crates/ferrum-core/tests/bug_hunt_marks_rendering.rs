//! Integration tests for marks-rendering scene-node contracts.
//!
//! Every test here exercises real `ferrum_scene` crate behavior — a `serde`
//! round-trip (including `skip_serializing_if` default-omission and enum
//! variant tagging) — never a struct literal echoed straight back to itself.
//! This is the one integration-test pattern that observes real crate code;
//! mark-geometry contracts belong as `#[cfg(test)]` modules next to the code
//! they exercise (see `crates/ferrum-core/src/render/marks/{arc,bar,point,
//! line,area,text}.rs`, where the formula-mirror and literal-echo tests that
//! used to live in this file and its now-deleted `_r2` sibling were ported
//! against the real mark-builder functions, or deleted outright as
//! duplicates/tautologies — R1 remediation, 2026-08-27). No test in this
//! file claims to detect divergence from a mirrored implementation — there
//! is no mirror; every assertion runs against real `ferrum_scene` `serde`
//! output.

#[cfg(test)]
mod tests {
    use ferrum_scene::{
        Color, FillStroke, FontWeight, ImageData, ImageMime, PathCmd, SceneNode, StrokeCap,
        StrokeJoin, StrokeStyle, TextAnchor, TextBaseline, TextStyle,
    };

    fn default_fill_stroke() -> FillStroke {
        FillStroke {
            fill: Some(Color::rgb(70, 130, 180)),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    // ── FillStroke serde: skip_serializing_if defaults ────────────────────

    #[test]
    fn fill_stroke_default_fields_omitted_in_json() {
        // When stroke_opacity=1.0, fill_opacity=1.0, angle=0.0, they should be
        // omitted from JSON (skip_serializing_if).
        let style = FillStroke {
            fill: Some(Color::rgb(100, 100, 100)),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let json = serde_json::to_string(&style).unwrap();
        assert!(!json.contains("stroke_opacity"),
            "stroke_opacity=1.0 should be omitted, got: {json}");
        assert!(!json.contains("fill_opacity"),
            "fill_opacity=1.0 should be omitted, got: {json}");
        assert!(!json.contains("angle"),
            "angle=0.0 should be omitted, got: {json}");
    }

    #[test]
    fn fill_stroke_non_default_fields_present_in_json() {
        let style = FillStroke {
            fill: Some(Color::rgb(100, 100, 100)),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 0.5,
            fill_opacity: 0.3,
            angle: 45.0,
        };
        let json = serde_json::to_string(&style).unwrap();
        assert!(json.contains("stroke_opacity"),
            "stroke_opacity=0.5 must be present, got: {json}");
        assert!(json.contains("fill_opacity"),
            "fill_opacity=0.3 must be present, got: {json}");
        assert!(json.contains("angle"),
            "angle=45.0 must be present, got: {json}");
    }

    #[test]
    fn bug_hunt_empty_stroke_dash_serde() {
        let style = FillStroke {
            fill: Some(Color::rgb(0, 0, 0)),
            stroke: Some(Color::rgb(255, 0, 0)),
            stroke_width: 2.0,
            opacity: 1.0,
            stroke_dash: Some(vec![]),
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let json = serde_json::to_string(&style).unwrap();
        let back: FillStroke = serde_json::from_str(&json).unwrap();
        // Empty vec round-trips as Some([]) -- renderer treats it as solid.
        assert_eq!(back.stroke_dash, Some(vec![]));
    }

    #[test]
    fn bug_hunt_r2_transparent_color_serde() {
        // Color with alpha=0 must round-trip (fully transparent is valid).
        let color = Color::rgba(255, 0, 0, 0); // red, fully transparent
        let style = FillStroke {
            fill: Some(color),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let json = serde_json::to_string(&style).unwrap();
        let back: FillStroke = serde_json::from_str(&json).unwrap();
        let c = back.fill.unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 0, "alpha=0 must be preserved through serde");
    }

    // ── SceneNode serde round-trips ─────────────────────────────────────

    #[test]
    fn circle_serde_round_trip_preserves_all_fields() {
        let node = SceneNode::Circle {
            cx: 123.456,
            cy: 789.012,
            r: 5.5,
            style: FillStroke {
                fill: Some(Color::rgba(255, 0, 0, 128)),
                stroke: Some(Color::rgb(0, 0, 0)),
                stroke_width: 1.5,
                opacity: 0.8,
                stroke_dash: Some(vec![4.0, 2.0]),
                stroke_opacity: 0.6,
                fill_opacity: 0.4,
                angle: 45.0,
            },
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Circle { cx, cy, r, style } = &back {
            assert!((cx - 123.456).abs() < 1e-9);
            assert!((cy - 789.012).abs() < 1e-9);
            assert!((r - 5.5).abs() < 1e-9);
            assert!((style.stroke_opacity - 0.6).abs() < 1e-9);
            assert!((style.fill_opacity - 0.4).abs() < 1e-9);
            assert!((style.angle - 45.0).abs() < 1e-9);
            assert_eq!(style.stroke_dash.as_ref().unwrap(), &[4.0, 2.0]);
        } else {
            panic!("expected Circle after round-trip");
        }
    }

    #[test]
    fn rect_serde_round_trip_preserves_corner_radius() {
        let node = SceneNode::Rect {
            x: 10.0,
            y: 20.0,
            w: 80.0,
            h: 60.0,
            style: FillStroke {
                fill: Some(Color::rgb(50, 100, 200)),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 0.9,
                angle: 0.0,
            },
            corner_radius: 5.0,
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Rect { corner_radius, style, .. } = &back {
            assert!((corner_radius - 5.0).abs() < 1e-9);
            assert!((style.fill_opacity - 0.9).abs() < 1e-9);
        } else {
            panic!("expected Rect after round-trip");
        }
    }

    #[test]
    fn bug_hunt_r2_rect_corner_radius_larger_than_dims() {
        // Corner radius larger than rect dimensions should still serialize.
        // Browsers clamp this, but the scene node must be valid.
        let node = SceneNode::Rect {
            x: 10.0,
            y: 10.0,
            w: 20.0,
            h: 30.0,
            style: default_fill_stroke(),
            corner_radius: 100.0, // larger than both w and h
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Rect { corner_radius, .. } = &back {
            assert!((corner_radius - 100.0).abs() < 1e-9);
        } else {
            panic!("expected Rect after round-trip");
        }
    }

    #[test]
    fn polyline_serde_round_trip_preserves_points() {
        let points = vec![(0.0, 0.0), (50.0, 100.0), (100.0, 50.0), (150.0, 75.0)];
        let node = SceneNode::Polyline {
            points: points.clone(),
            style: StrokeStyle {
                color: Color::rgb(0, 128, 255),
                width: 2.0,
                opacity: 1.0,
                dash: None,
                stroke_cap: Some(StrokeCap::Round),
                stroke_join: Some(StrokeJoin::Round),
                stroke_opacity: 0.8,
            },
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Polyline { points: pts, style } = &back {
            assert_eq!(pts.len(), 4);
            assert!((pts[0].0 - 0.0).abs() < 1e-9);
            assert!((pts[3].1 - 75.0).abs() < 1e-9);
            assert!((style.stroke_opacity - 0.8).abs() < 1e-9);
            assert_eq!(style.stroke_cap, Some(StrokeCap::Round));
            assert_eq!(style.stroke_join, Some(StrokeJoin::Round));
        } else {
            panic!("expected Polyline after round-trip");
        }
    }

    #[test]
    fn path_close_cmd_serde_round_trip() {
        let node = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 0.0 },
                PathCmd::LineTo { x: 100.0, y: 0.0 },
                PathCmd::LineTo { x: 50.0, y: 100.0 },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Path { commands, closed, .. } = &back {
            assert_eq!(commands.len(), 4);
            assert!(matches!(commands[3], PathCmd::Close));
            assert!(*closed);
        } else {
            panic!("expected Path after round-trip");
        }
    }

    #[test]
    fn bug_hunt_degenerate_path_moveto_close() {
        // A path with only a MoveTo and Close (no LineTo) is degenerate but
        // must be valid (e.g., area with 1 point that somehow generates a path).
        let node = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 50.0, y: 50.0 },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Path { commands, .. } = &back {
            assert_eq!(commands.len(), 2);
        } else {
            panic!("expected Path after round-trip");
        }
    }

    #[test]
    fn bug_hunt_arcto_zero_radii() {
        // ArcTo with rx=0, ry=0 (degenerate donut inner ring) must serialize
        // without error and round-trip successfully with finite coordinates.
        let cmd = PathCmd::ArcTo {
            rx: 0.0, ry: 0.0, rotation: 0.0,
            large_arc: false, sweep: true,
            x: 50.0, y: 50.0,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        // Serde should not panic on zero radii
        assert!(!json.is_empty(), "ArcTo must serialize to non-empty JSON");
        let back: PathCmd = serde_json::from_str(&json).unwrap();
        if let PathCmd::ArcTo { rx, ry, x, y, .. } = &back {
            assert!(rx.is_finite());
            assert!(ry.is_finite());
            assert!(x.is_finite());
            assert!(y.is_finite());
            assert_eq!(*rx, 0.0);
            assert_eq!(*ry, 0.0);
        } else {
            panic!("expected ArcTo after round-trip, got: {:?}", back);
        }
    }

    #[test]
    fn bug_hunt_r2_hlineto_vlineto_serde() {
        // HLineTo and VLineTo path commands must round-trip through JSON.
        // These are used by step/step-before/step-after interpolation.
        let cmds = vec![
            PathCmd::MoveTo { x: 0.0, y: 0.0 },
            PathCmd::HLineTo { x: 50.0 },
            PathCmd::VLineTo { y: 100.0 },
            PathCmd::HLineTo { x: 100.0 },
            PathCmd::Close,
        ];
        let json = serde_json::to_string(&cmds).unwrap();
        let back: Vec<PathCmd> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 5);
        assert!(matches!(back[1], PathCmd::HLineTo { x } if (x - 50.0).abs() < 1e-9));
        assert!(matches!(back[2], PathCmd::VLineTo { y } if (y - 100.0).abs() < 1e-9));
    }

    // ── SceneNode serde: Polygon variant ──────────────────────────────────

    #[test]
    fn bug_hunt_r2_polygon_serde_with_hole() {
        // Polygon SceneNode with 2 rings (exterior + hole) must round-trip via JSON.
        let node = SceneNode::Polygon {
            rings: vec![
                vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
                vec![[25.0, 25.0], [75.0, 25.0], [75.0, 75.0], [25.0, 75.0]],
            ],
            style: default_fill_stroke(),
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Polygon { rings, .. } = &back {
            assert_eq!(rings.len(), 2, "expected 2 rings (exterior + hole)");
            assert_eq!(rings[0].len(), 4, "exterior ring should have 4 vertices");
            assert_eq!(rings[1].len(), 4, "hole ring should have 4 vertices");
        } else {
            panic!("expected Polygon after round-trip");
        }
    }

    #[test]
    fn bug_hunt_r2_polygon_empty_rings_serde() {
        // Empty polygon rings should serialize/deserialize without panicking.
        let node = SceneNode::Polygon {
            rings: vec![],
            style: default_fill_stroke(),
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Polygon { rings, .. } = &back {
            assert!(rings.is_empty());
        } else {
            panic!("expected Polygon after round-trip");
        }
    }

    // ── SceneNode serde: Image variant ────────────────────────────────────

    #[test]
    fn bug_hunt_r2_image_url_serde() {
        // Image SceneNode with Url data should round-trip.
        let node = SceneNode::Image {
            x: 10.0,
            y: 20.0,
            w: 100.0,
            h: 50.0,
            data: ImageData::Url {
                url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
            },
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Image { x, y, w, h, data } = &back {
            assert!((x - 10.0).abs() < 1e-9);
            assert!((y - 20.0).abs() < 1e-9);
            assert!((w - 100.0).abs() < 1e-9);
            assert!((h - 50.0).abs() < 1e-9);
            assert!(matches!(data, ImageData::Url { .. }));
        } else {
            panic!("expected Image after round-trip");
        }
    }

    #[test]
    fn bug_hunt_r2_image_inline_serde() {
        // Image SceneNode with Inline PNG data should round-trip.
        let node = SceneNode::Image {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
            data: ImageData::Inline {
                bytes: vec![137, 80, 78, 71, 13, 10, 26, 10], // PNG magic
                mime: ImageMime::Png,
            },
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Image { data, .. } = &back {
            if let ImageData::Inline { bytes, mime } = data {
                assert_eq!(bytes[0..4], [137, 80, 78, 71], "PNG magic header preserved");
                assert!(matches!(mime, ImageMime::Png));
            } else {
                panic!("expected ImageData::Inline after round-trip");
            }
        } else {
            panic!("expected Image after round-trip");
        }
    }

    // ── SceneNode serde: Text variant ─────────────────────────────────────

    #[test]
    fn bug_hunt_r2_text_node_unicode_serde() {
        // Text SceneNode must round-trip, including unicode content.
        let node = SceneNode::Text {
            x: 50.0,
            y: 50.0,
            content: "\u{2026} \u{03B1}\u{03B2}\u{03B3}".to_string(), // ellipsis + greek
            slot: None,
            style: TextStyle {
                color: Color::rgb(0, 0, 0),
                font_size: 12.0,
                anchor: TextAnchor::Middle,
                angle: 0.0,
                font_family: "Arial".to_string(),
                font_weight: FontWeight::Normal,
                baseline: TextBaseline::Alphabetic,
                opacity: 1.0,
            },
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        if let SceneNode::Text { content, .. } = &back {
            assert!(content.contains("\u{2026}"), "ellipsis must survive serde");
            assert!(content.contains("\u{03B1}"), "alpha must survive serde");
        } else {
            panic!("expected Text after round-trip");
        }
    }
}
