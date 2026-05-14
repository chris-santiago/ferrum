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
        let selected = data_idx.is_some_and(|di| sel.contains(di));
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
        };
        apply_value_to_rect(
            &mut inst,
            &ChannelName::Opacity,
            &EncodingValue::Opacity { value: 0.3 },
        );
        assert!((inst.opacity - 0.3).abs() < 0.01);
    }
}
