pub mod error;
pub mod types;
pub mod selection;

pub use error::*;
pub use types::*;
pub use selection::*;

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
                assert_eq!(anchor, RawAnchor::Chrome, "missing anchor should default to Chrome");
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
        let node = SceneNode::Raw { svg: "<defs/>".to_string(), anchor: RawAnchor::Chrome };
        let json = serde_json::to_string(&node).unwrap();
        let back: SceneNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, node);
    }

    /// Round-trip a `SceneNode::Raw` with `Data` anchor through JSON.
    #[test]
    fn raw_node_data_serde_round_trip() {
        let node = SceneNode::Raw { svg: "<image/>".to_string(), anchor: RawAnchor::Data };
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
                plot_area: Rect { x: 50.0, y: 10.0, w: 300.0, h: 250.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 300.0, h: 250.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None,
                    y_domain: None,
                    expand: true,
                    clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
}
