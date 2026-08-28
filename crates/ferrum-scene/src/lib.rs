pub mod error;
pub mod parameter;
pub mod selection;
pub mod types;

pub use error::*;
pub use parameter::*;
pub use selection::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    // ── RawAnchor serde round-trip ────────────────────────────────────────────

    #[test]
    fn raw_anchor_chrome_serializes_to_snake_case() {
        let json = serde_json::to_string(&RawAnchor::Chrome).unwrap();
        assert_eq!(json, "\"chrome\"");
    }

    #[test]
    fn raw_anchor_data_serializes_to_snake_case() {
        let json = serde_json::to_string(&RawAnchor::Data).unwrap();
        assert_eq!(json, "\"data\"");
    }

    #[test]
    fn raw_anchor_round_trips() {
        for anchor in [RawAnchor::Chrome, RawAnchor::Data] {
            let json = serde_json::to_string(&anchor).unwrap();
            let back: RawAnchor = serde_json::from_str(&json).unwrap();
            assert_eq!(back, anchor);
        }
    }

    /// A `SceneNode::Raw` JSON WITHOUT an `anchor` key must deserialize with
    /// `anchor == Chrome` (serde back-compat default).
    #[test]
    fn raw_node_without_anchor_field_defaults_to_chrome() {
        let json = r#"{"type":"raw","svg":"<rect/>"}"#;
        let node: SceneNode = serde_json::from_str(json).unwrap();
        match node {
            SceneNode::Raw { anchor, .. } => {
                assert_eq!(
                    anchor,
                    RawAnchor::Chrome,
                    "missing anchor should default to Chrome"
                );
            }
            _ => panic!("expected SceneNode::Raw"),
        }
    }

    /// A `SceneNode::Raw` with explicit `anchor: "data"` deserializes correctly.
    #[test]
    fn raw_node_with_data_anchor_deserializes() {
        let json = r#"{"type":"raw","svg":"<image/>","anchor":"data"}"#;
        let node: SceneNode = serde_json::from_str(json).unwrap();
        match node {
            SceneNode::Raw { anchor, .. } => {
                assert_eq!(anchor, RawAnchor::Data);
            }
            _ => panic!("expected SceneNode::Raw"),
        }
    }

    /// Round-trip a `SceneNode::Raw` with `Chrome` anchor through JSON.
    #[test]
    fn raw_node_chrome_serde_round_trip() {
        let node = SceneNode::Raw {
            svg: "<defs/>".to_string(),
            anchor: RawAnchor::Chrome,
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, node);
    }

    /// Round-trip a `SceneNode::Raw` with `Data` anchor through JSON.
    #[test]
    fn raw_node_data_serde_round_trip() {
        let node = SceneNode::Raw {
            svg: "<image/>".to_string(),
            anchor: RawAnchor::Data,
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, node);
    }

    #[test]
    fn scene_graph_serde_round_trip() {
        let scene = SceneGraph {
            width: 400.0,
            height: 300.0,
            background: Some(Color::rgb(255, 255, 255)),
            title: vec![],
            legend: vec![],
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
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![
                        SceneNode::Circle {
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
                        },
                        SceneNode::Rect {
                            x: 200.0,
                            y: 100.0,
                            w: 20.0,
                            h: 40.0,
                            style: FillStroke {
                                fill: Some(Color::rgba(255, 0, 0, 128)),
                                stroke: Some(Color::rgb(0, 0, 0)),
                                stroke_width: 1.0,
                                opacity: 0.8,
                                stroke_dash: None,
                                stroke_opacity: 1.0,
                                fill_opacity: 1.0,
                                angle: 0.0,
                            },
                            corner_radius: 2.0,
                        },
                    ],
                    data_indices: Some(vec![0, 1]),
                    tooltips: None,
                    hrefs: None,
                    descriptions: None,
                    keys: None,
                    blend: BlendMode::Normal,
                    stroke_cap: None,
                    stroke_join: None,
                    packed_instances: None,
                    y_slot: 0,
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
                below_marks: Vec::new(),
                chrome_above: Vec::new(),
            }],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let json = serde_json::to_string(&scene).expect("serialize");
        let deserialized: SceneGraph = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(scene, deserialized);
    }

    // ── Panel::layout_scale (ratio-fitted cells / W5 foundation) ──────────────

    #[test]
    fn layout_scale_identity_apply_is_noop() {
        let ls = LayoutScale::identity();
        assert_eq!(ls.apply(10.0, 20.0), (10.0, 20.0));
    }

    #[test]
    fn layout_scale_apply_scales_and_translates() {
        let ls = LayoutScale {
            sx: 0.5,
            sy: 2.0,
            tx: 10.0,
            ty: -5.0,
        };
        assert_eq!(ls.apply(4.0, 3.0), (12.0, 1.0));
    }

    #[test]
    fn layout_scale_default_is_identity() {
        assert_eq!(LayoutScale::default(), LayoutScale::identity());
        assert!(LayoutScale::default().is_identity());
    }

    #[test]
    fn layout_scale_non_identity_is_not_identity() {
        assert!(!LayoutScale {
            sx: 1.5,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0
        }
        .is_identity());
        assert!(!LayoutScale {
            sx: 1.0,
            sy: 1.0,
            tx: 1.0,
            ty: 0.0
        }
        .is_identity());
    }

    /// Identity `layout_scale` must be skipped by serde (`skip_serializing_if`)
    /// so scenes serialized before this field existed stay byte-identical.
    #[test]
    fn layout_scale_identity_is_not_serialized() {
        let panel = Panel {
            id: 0,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        };
        let json = serde_json::to_string(&panel).expect("serialize");
        assert!(
            !json.contains("layout_scale"),
            "identity layout_scale must be omitted from serialized JSON, got: {json}"
        );
    }

    /// A `Panel` JSON payload without a `layout_scale` key (e.g. a scene
    /// serialized before this field existed) must deserialize with
    /// `layout_scale == LayoutScale::identity()` — the serde back-compat
    /// contract (`#[serde(default)]`).
    #[test]
    fn panel_without_layout_scale_field_defaults_to_identity() {
        let json = r#"{
            "id": 0,
            "plot_area": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
            "clip": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
            "coord": {"kind": "cartesian", "x_domain": null, "y_domain": null, "expand": true, "clip": true},
            "grid": [],
            "marks": [],
            "axes": [],
            "annotations": [],
            "strip_title": []
        }"#;
        let panel: Panel = serde_json::from_str(json).expect("deserialize");
        assert_eq!(
            panel.layout_scale,
            LayoutScale::identity(),
            "missing layout_scale field must default to identity"
        );
    }

    /// A non-identity `layout_scale` must round-trip through JSON, and the
    /// serialized form must actually carry the field (proving the skip only
    /// applies at identity).
    #[test]
    fn layout_scale_non_identity_round_trips_through_json() {
        let ls = LayoutScale {
            sx: 0.5,
            sy: 0.25,
            tx: 12.0,
            ty: -8.0,
        };
        let panel = Panel {
            id: 1,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: ls,
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        };
        let json = serde_json::to_string(&panel).expect("serialize");
        assert!(
            json.contains("layout_scale"),
            "non-identity layout_scale must be serialized, got: {json}"
        );
        let deserialized: Panel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.layout_scale, ls);
        assert_eq!(panel, deserialized);
    }

    // ── Panel::below_marks / chrome_above (GH #89B typed chrome/content slots) ──

    /// Empty `below_marks`/`chrome_above` — the overwhelming majority of
    /// scenes — must be omitted from serialized JSON (`skip_serializing_if`),
    /// matching the `layout_scale` byte-stability precedent above.
    #[test]
    fn below_marks_and_chrome_above_empty_are_not_serialized() {
        let panel = Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        };
        let json = serde_json::to_string(&panel).expect("serialize");
        assert!(
            !json.contains("below_marks"),
            "empty below_marks must be omitted from serialized JSON, got: {json}"
        );
        assert!(
            !json.contains("chrome_above"),
            "empty chrome_above must be omitted from serialized JSON, got: {json}"
        );
    }

    /// A `Panel` JSON payload without `below_marks`/`chrome_above` keys (every
    /// scene serialized before these fields existed) must deserialize with
    /// both empty — the serde back-compat contract (`#[serde(default)]`).
    #[test]
    fn panel_without_below_marks_or_chrome_above_fields_defaults_to_empty() {
        let json = r#"{
            "id": 0,
            "plot_area": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
            "clip": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
            "coord": {"kind": "cartesian", "x_domain": null, "y_domain": null, "expand": true, "clip": true},
            "grid": [],
            "marks": [],
            "axes": [],
            "annotations": [],
            "strip_title": []
        }"#;
        let panel: Panel = serde_json::from_str(json).expect("deserialize");
        assert!(
            panel.below_marks.is_empty(),
            "missing below_marks field must default to empty"
        );
        assert!(
            panel.chrome_above.is_empty(),
            "missing chrome_above field must default to empty"
        );
    }

    /// Non-empty `below_marks`/`chrome_above` must round-trip through JSON,
    /// and the serialized form must actually carry the keys (proving the
    /// skip only applies when empty).
    #[test]
    fn below_marks_and_chrome_above_non_empty_round_trip_through_json() {
        let below_marks_node = SceneNode::Text {
            x: 10.0,
            y: 20.0,
            content: "below-marks label".to_string(),
            slot: None,
            style: TextStyle {
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                anchor: TextAnchor::Middle,
                baseline: TextBaseline::Alphabetic,
                angle: 0.0,
                color: Color::rgb(50, 50, 50),
                opacity: 1.0,
                font_family: "sans-serif".to_string(),
            },
        };
        let chrome_above_node = SceneNode::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 0.0,
            style: StrokeStyle {
                color: Color::rgb(0, 0, 0),
                width: 1.0,
                opacity: 1.0,
                dash: None,
                stroke_cap: None,
                stroke_join: None,
                stroke_opacity: 1.0,
            },
        };
        let panel = Panel {
            id: 2,
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: vec![below_marks_node.clone()],
            chrome_above: vec![chrome_above_node.clone()],
        };
        let json = serde_json::to_string(&panel).expect("serialize");
        assert!(
            json.contains("below_marks"),
            "non-empty below_marks must be serialized, got: {json}"
        );
        assert!(
            json.contains("chrome_above"),
            "non-empty chrome_above must be serialized, got: {json}"
        );
        let deserialized: Panel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.below_marks, vec![below_marks_node]);
        assert_eq!(deserialized.chrome_above, vec![chrome_above_node]);
        assert_eq!(panel, deserialized);
    }
}
