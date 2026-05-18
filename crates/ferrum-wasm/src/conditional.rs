use ferrum_scene::{
    ChannelName, ConditionalEncoding, EncodingValue, MarkBatch, Panel, SceneNode,
};

use crate::scene_load::{CircleInstance, RectInstance};
use crate::selection_state::SelectionState;

use std::collections::HashMap;

pub struct ConditionalUpdates {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
}

pub fn resolve_conditionals(
    panels: &[Panel],
    conditionals: &[ConditionalEncoding],
    selections: &HashMap<String, SelectionState>,
    base_circles: &[CircleInstance],
    base_rects: &[RectInstance],
) -> ConditionalUpdates {
    let mut circles = base_circles.to_vec();
    let mut rects = base_rects.to_vec();

    let mut circle_offset = 0usize;
    let mut rect_offset = 0usize;

    for panel in panels {
        for batch in &panel.marks {
            let (n_circles, n_rects) = count_instances(batch);

            for cond in conditionals {
                let Some(sel) = selections.get(&cond.selection_name) else {
                    continue;
                };
                if matches!(sel, SelectionState::Empty) {
                    continue;
                }
                if let Some(indices) = &batch.data_indices {
                    apply_conditional_to_batch(
                        &cond.channel,
                        &cond.if_selected,
                        &cond.if_not,
                        sel,
                        indices,
                        batch,
                        &mut circles,
                        circle_offset,
                        &mut rects,
                        rect_offset,
                    );
                }
            }

            circle_offset += n_circles;
            rect_offset += n_rects;
        }
    }

    ConditionalUpdates {
        circle_instances: circles,
        rect_instances: rects,
    }
}

fn count_instances(batch: &MarkBatch) -> (usize, usize) {
    let mut nc = 0usize;
    let mut nr = 0usize;
    for node in &batch.nodes {
        match node {
            SceneNode::Circle { .. } => nc += 1,
            SceneNode::Rect { .. } => nr += 1,
            _ => {}
        }
    }
    (nc, nr)
}

#[allow(clippy::too_many_arguments)]
fn apply_conditional_to_batch(
    channel: &ChannelName,
    if_selected: &EncodingValue,
    if_not: &EncodingValue,
    sel: &SelectionState,
    data_indices: &[usize],
    batch: &MarkBatch,
    circles: &mut [CircleInstance],
    circle_offset: usize,
    rects: &mut [RectInstance],
    rect_offset: usize,
) {
    let mut ci = 0usize;
    let mut ri = 0usize;

    for (node_idx, node) in batch.nodes.iter().enumerate() {
        let data_idx = data_indices.get(node_idx).copied();

        let selected = match sel {
            SelectionState::Interval { .. } => {
                // Spatial containment: check if mark center is inside brush.
                let pos = match node {
                    SceneNode::Circle { cx, cy, .. } => Some((*cx, *cy)),
                    SceneNode::Rect { x, y, w, h, .. } => Some((*x + *w / 2.0, *y + *h / 2.0)),
                    _ => None,
                };
                pos.is_some_and(|(mx, my)| sel.contains_point(mx, my))
            }
            _ => data_idx.is_some_and(|di| sel.contains(di)),
        };

        let value = if selected { if_selected } else { if_not };

        match node {
            SceneNode::Circle { .. } => {
                if let Some(inst) = circles.get_mut(circle_offset + ci) {
                    apply_value_to_circle(inst, channel, value);
                }
                ci += 1;
            }
            SceneNode::Rect { .. } => {
                if let Some(inst) = rects.get_mut(rect_offset + ri) {
                    apply_value_to_rect(inst, channel, value);
                }
                ri += 1;
            }
            _ => {}
        }
    }
}

fn apply_value_to_circle(inst: &mut CircleInstance, channel: &ChannelName, value: &EncodingValue) {
    match (channel, value) {
        (ChannelName::Color, EncodingValue::Color { value: c }) => {
            inst.fill_color = [
                c.r as f32 / 255.0,
                c.g as f32 / 255.0,
                c.b as f32 / 255.0,
                c.a as f32 / 255.0,
            ];
        }
        (ChannelName::Opacity, EncodingValue::Opacity { value: o }) => {
            inst.opacity = *o as f32;
        }
        (ChannelName::Size, EncodingValue::Size { value: s }) => {
            inst.radius = (*s as f32 / std::f32::consts::PI).sqrt();
        }
        _ => {}
    }
}

fn apply_value_to_rect(inst: &mut RectInstance, channel: &ChannelName, value: &EncodingValue) {
    match (channel, value) {
        (ChannelName::Color, EncodingValue::Color { value: c }) => {
            inst.fill_color = [
                c.r as f32 / 255.0,
                c.g as f32 / 255.0,
                c.b as f32 / 255.0,
                c.a as f32 / 255.0,
            ];
        }
        (ChannelName::Opacity, EncodingValue::Opacity { value: o }) => {
            inst.opacity = *o as f32;
        }
        (ChannelName::Size, EncodingValue::Size { value: s }) => {
            // Without orientation context (h-bar vs v-bar) we apply the size to
            // both width and height so the conditional always has a visible effect.
            // The rendering layer controls which dimension is the data extent; the
            // other is the band width set by the scale.  Applying to both is the
            // conservative fallback that matches the circle counterpart's intent
            // (scale the mark proportionally to the encoded value).
            inst.size = [*s as f32, *s as f32];
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::Color;

    #[test]
    fn apply_color_to_circle() {
        let mut inst = CircleInstance {
            center: [0.0, 0.0],
            radius: 5.0,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 0.0],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        apply_value_to_circle(
            &mut inst,
            &ChannelName::Color,
            &EncodingValue::Color { value: red },
        );
        assert!((inst.fill_color[0] - 1.0).abs() < 0.01);
        assert!(inst.fill_color[1] < 0.01);
    }

    #[test]
    fn apply_opacity_to_rect() {
        let mut inst = RectInstance {
            position: [0.0, 0.0],
            size: [10.0, 10.0],
            corner_radius: 0.0,
            fill_color: [0.5, 0.5, 0.5, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 0.0],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        apply_value_to_rect(
            &mut inst,
            &ChannelName::Opacity,
            &EncodingValue::Opacity { value: 0.3 },
        );
        assert!((inst.opacity - 0.3).abs() < 0.01);
    }

    #[test]
    fn apply_size_to_rect() {
        let mut inst = RectInstance {
            position: [0.0, 0.0],
            size: [10.0, 10.0],
            corner_radius: 0.0,
            fill_color: [0.5, 0.5, 0.5, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 0.0],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        apply_value_to_rect(
            &mut inst,
            &ChannelName::Size,
            &EncodingValue::Size { value: 20.0 },
        );
        assert!((inst.size[0] - 20.0).abs() < 0.01);
        assert!((inst.size[1] - 20.0).abs() < 0.01);
    }

    // ── R3: Interval conditional encoding applies ───────────────────────────
    //
    // resolve_conditionals currently only uses SelectionState::contains(data_idx)
    // for membership, which always returns false for Interval selections. After
    // implementing spatial containment (contains_point), this test should pass.
    // The test documents the EXPECTED behavior: marks inside the brush rectangle
    // get the if_selected color, marks outside get if_not.

    #[test]
    fn r3_interval_conditional_encoding_applies() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };

        // Three circles: (20,30) inside, (100,100) outside, (30,40) inside brush.
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
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
                    SceneNode::Circle { cx: 20.0, cy: 30.0, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: 100.0, cy: 100.0, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: 30.0, cy: 40.0, r: 5.0, style: style.clone() },
                ],
                data_indices: Some(vec![0, 1, 2]),
                tooltips: None,
                hrefs: None,
                keys: None,
                blend: BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
                packed_instances: None,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];

        let red = Color { r: 255, g: 0, b: 0, a: 255 };
        let grey = Color { r: 128, g: 128, b: 128, a: 255 };

        let conditionals = vec![ConditionalEncoding {
            selection_name: "brush".to_string(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Color { value: red },
            if_not: EncodingValue::Color { value: grey },
        }];

        let mut selections = HashMap::new();
        selections.insert(
            "brush".to_string(),
            SelectionState::Interval {
                x_range: Some((10.0, 50.0)),
                y_range: Some((20.0, 60.0)),
            },
        );

        // Base circle instances — neutral fill so we can detect changes.
        let neutral = [0.0_f32, 0.0, 0.0, 1.0];
        let base_circles = vec![
            CircleInstance {
                center: [20.0, 30.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [100.0, 100.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [30.0, 40.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];

        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);

        let red_fill = [1.0_f32, 0.0, 0.0, 1.0];
        let grey_fill = [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0];

        // Circle 0 at (20,30) is inside brush (10..50, 20..60) — should be red.
        assert!(
            (result.circle_instances[0].fill_color[0] - red_fill[0]).abs() < 0.01,
            "circle at (20,30) should be red (inside brush), got {:?}",
            result.circle_instances[0].fill_color
        );

        // Circle 1 at (100,100) is outside brush — should be grey.
        assert!(
            (result.circle_instances[1].fill_color[0] - grey_fill[0]).abs() < 0.01,
            "circle at (100,100) should be grey (outside brush), got {:?}",
            result.circle_instances[1].fill_color
        );

        // Circle 2 at (30,40) is inside brush — should be red.
        assert!(
            (result.circle_instances[2].fill_color[0] - red_fill[0]).abs() < 0.01,
            "circle at (30,40) should be red (inside brush), got {:?}",
            result.circle_instances[2].fill_color
        );
    }
}
