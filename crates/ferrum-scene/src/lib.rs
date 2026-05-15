pub mod error;
pub mod types;
pub mod selection;

pub use error::*;
pub use types::*;
pub use selection::*;

#[cfg(test)]
mod tests {
    use super::*;

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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
            }],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
        };

        let json = serde_json::to_string(&scene).expect("serialize");
        let deserialized: SceneGraph = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(scene, deserialized);
    }
}
