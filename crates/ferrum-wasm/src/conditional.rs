use ferrum_scene::{
    ChannelName, ConditionalEncoding, EncodingValue, FieldValue, MarkBatch, Panel, SceneNode,
};

use crate::scene_load::{color_to_linear, CircleInstance, RectInstance};
use crate::selection_state::SelectionState;

use std::collections::HashMap;

pub struct ConditionalUpdates {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
}

/// Mutable references to the circle and rect instance buffers together with
/// the current write offsets. Groups the four mutable-state parameters that
/// `apply_conditional_to_batch` needs, keeping its signature manageable.
struct InstanceBuffers<'a> {
    circles: &'a mut [CircleInstance],
    circle_offset: usize,
    rects: &'a mut [RectInstance],
    rect_offset: usize,
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
                    let mut bufs = InstanceBuffers {
                        circles: &mut circles,
                        circle_offset,
                        rects: &mut rects,
                        rect_offset,
                    };
                    apply_conditional_to_batch(
                        &cond.channel,
                        &cond.if_selected,
                        &cond.if_not,
                        sel,
                        indices,
                        batch,
                        &mut bufs,
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

fn apply_conditional_to_batch(
    channel: &ChannelName,
    if_selected: &EncodingValue,
    if_not: &EncodingValue,
    sel: &SelectionState,
    data_indices: &[usize],
    batch: &MarkBatch,
    bufs: &mut InstanceBuffers<'_>,
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
            SelectionState::Point { field_values, .. } if !field_values.is_empty() => {
                // Field-value matching: check if this mark's tooltip contains
                // matching values for all selection fields. This enables
                // cross-panel linked selection where panels have different
                // datasets and data indices cannot be shared.
                batch
                    .tooltips
                    .as_ref()
                    .and_then(|tips| tips.get(node_idx))
                    .map(|tip| {
                        field_values.iter().all(|(fname, fval)| {
                            tip.fields
                                .iter()
                                .any(|f| f.name == *fname && field_value_matches_tooltip(&f.value, fval))
                        })
                    })
                    .unwrap_or(false)
            }
            _ => data_idx.is_some_and(|di| sel.contains(di)),
        };

        let value = if selected { if_selected } else { if_not };

        match node {
            SceneNode::Circle { .. } => {
                if let Some(inst) = bufs.circles.get_mut(bufs.circle_offset + ci) {
                    apply_value_to_circle(inst, channel, value);
                }
                ci += 1;
            }
            SceneNode::Rect { .. } => {
                if let Some(inst) = bufs.rects.get_mut(bufs.rect_offset + ri) {
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
            inst.fill_color = color_to_linear(c, 1.0);
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

/// Compare a tooltip string value against a typed `FieldValue`.
///
/// Tooltip fields are always stored as strings in the scene graph. This
/// function bridges the gap by parsing the string according to the
/// `FieldValue` variant so cross-panel matching works even when one panel
/// stores `"42"` and the selection carries `FieldValue::Number { value: 42.0 }`.
fn field_value_matches_tooltip(tooltip_value: &str, field_value: &FieldValue) -> bool {
    match field_value {
        FieldValue::String { value } => tooltip_value == value,
        FieldValue::Number { value } => tooltip_value
            .parse::<f64>()
            .ok()
            .is_some_and(|n| (n - *value).abs() < 1e-10),
        FieldValue::Bool { value } => tooltip_value
            .parse::<bool>()
            .ok()
            .is_some_and(|b| b == *value),
        FieldValue::Null => tooltip_value.is_empty() || tooltip_value == "null",
    }
}

fn apply_value_to_rect(inst: &mut RectInstance, channel: &ChannelName, value: &EncodingValue) {
    match (channel, value) {
        (ChannelName::Color, EncodingValue::Color { value: c }) => {
            inst.fill_color = color_to_linear(c, 1.0);
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
    use crate::scene_load::srgb_to_linear;
    use ferrum_scene::{Color, FieldValue};

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

    // ── Linked-selection conditional tests ────────────────────────────

    // Test 6: Field-value matching selects correct marks.
    #[test]
    fn field_value_matching_selects_correct_marks() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
            TooltipContent, TooltipField,
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

        // Three circles with tooltips: mark 0 group="a", mark 1 group="b", mark 2 group="a".
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
                    SceneNode::Circle { cx: 50.0, cy: 50.0, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: 150.0, cy: 50.0, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: 250.0, cy: 50.0, r: 5.0, style: style.clone() },
                ],
                data_indices: Some(vec![0, 1, 2]),
                tooltips: Some(vec![
                    TooltipContent {
                        fields: vec![TooltipField {
                            name: "group".to_string(),
                            value: "a".to_string(),
                        }],
                    },
                    TooltipContent {
                        fields: vec![TooltipField {
                            name: "group".to_string(),
                            value: "b".to_string(),
                        }],
                    },
                    TooltipContent {
                        fields: vec![TooltipField {
                            name: "group".to_string(),
                            value: "a".to_string(),
                        }],
                    },
                ]),
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
            selection_name: "sel".to_string(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Color { value: red },
            if_not: EncodingValue::Color { value: grey },
        }];

        // Point selection with field_values indicating group="a".
        let mut selections = HashMap::new();
        selections.insert(
            "sel".to_string(),
            SelectionState::Point {
                indices: vec![0, 2],
                field_values: vec![(
                    "group".to_string(),
                    FieldValue::String { value: "a".to_string() },
                )],
            },
        );

        let neutral = [0.0_f32, 0.0, 0.0, 1.0];
        let base_circles: Vec<CircleInstance> = (0..3)
            .map(|i| CircleInstance {
                center: [50.0 + i as f32 * 100.0, 50.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            })
            .collect();

        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);

        let red_r = srgb_to_linear(1.0); // sRGB 255/255 = 1.0 → linear 1.0
        let grey_linear = srgb_to_linear(128.0 / 255.0);

        // Mark 0 (group="a") should be red.
        assert!(
            (result.circle_instances[0].fill_color[0] - red_r).abs() < 0.01,
            "mark 0 (group=a) should be red, got {:?}",
            result.circle_instances[0].fill_color
        );
        // Mark 1 (group="b") should be grey.
        assert!(
            (result.circle_instances[1].fill_color[0] - grey_linear).abs() < 0.01,
            "mark 1 (group=b) should be grey, got {:?}",
            result.circle_instances[1].fill_color
        );
        // Mark 2 (group="a") should be red.
        assert!(
            (result.circle_instances[2].fill_color[0] - red_r).abs() < 0.01,
            "mark 2 (group=a) should be red, got {:?}",
            result.circle_instances[2].fill_color
        );
    }

    // Test 7: Empty selection skips conditional (all marks retain original colors).
    #[test]
    fn empty_selection_skips_conditional() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
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
                    SceneNode::Circle { cx: 50.0, cy: 50.0, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: 150.0, cy: 50.0, r: 5.0, style: style.clone() },
                ],
                data_indices: Some(vec![0, 1]),
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
            selection_name: "sel".to_string(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Color { value: red },
            if_not: EncodingValue::Color { value: grey },
        }];

        // Selection is Empty — conditional should be skipped entirely.
        let mut selections = HashMap::new();
        selections.insert("sel".to_string(), SelectionState::Empty);

        let original_fill = [0.5_f32, 0.3, 0.1, 1.0];
        let base_circles = vec![
            CircleInstance {
                center: [50.0, 50.0],
                radius: 5.0,
                fill_color: original_fill,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [150.0, 50.0],
                radius: 5.0,
                fill_color: original_fill,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];

        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);

        // Both marks should retain their original colors unchanged.
        assert_eq!(
            result.circle_instances[0].fill_color, original_fill,
            "empty selection must not alter mark 0 fill color"
        );
        assert_eq!(
            result.circle_instances[1].fill_color, original_fill,
            "empty selection must not alter mark 1 fill color"
        );
    }

    // Test 8: Mixed point + interval selections — two conditionals apply independently.
    #[test]
    fn mixed_point_and_interval_selections_apply_independently() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
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

        // Two circles: (50,50) and (200,200).
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
                    SceneNode::Circle { cx: 50.0, cy: 50.0, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: 200.0, cy: 200.0, r: 5.0, style: style.clone() },
                ],
                data_indices: Some(vec![0, 1]),
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

        // Point selection selects index 0 only.
        // Interval selection covers (100..300, 100..300) — only circle at (200,200) is inside.
        let mut selections = HashMap::new();
        selections.insert(
            "point_sel".to_string(),
            SelectionState::Point {
                indices: vec![0],
                field_values: Vec::new(),
            },
        );
        selections.insert(
            "brush_sel".to_string(),
            SelectionState::Interval {
                x_range: Some((100.0, 300.0)),
                y_range: Some((100.0, 300.0)),
            },
        );

        // Conditional 1: point_sel → opacity 1.0 if selected, 0.2 if not.
        // Conditional 2: brush_sel → color red if selected, grey if not.
        let conditionals = vec![
            ConditionalEncoding {
                selection_name: "point_sel".to_string(),
                channel: ChannelName::Opacity,
                if_selected: EncodingValue::Opacity { value: 1.0 },
                if_not: EncodingValue::Opacity { value: 0.2 },
            },
            ConditionalEncoding {
                selection_name: "brush_sel".to_string(),
                channel: ChannelName::Color,
                if_selected: EncodingValue::Color {
                    value: Color { r: 255, g: 0, b: 0, a: 255 },
                },
                if_not: EncodingValue::Color {
                    value: Color { r: 128, g: 128, b: 128, a: 255 },
                },
            },
        ];

        let neutral = [0.0_f32, 0.0, 0.0, 1.0];
        let base_circles = vec![
            CircleInstance {
                center: [50.0, 50.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [200.0, 200.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];

        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);

        // Circle 0 at (50,50):
        //   - point_sel: index 0 is selected → opacity = 1.0
        //   - brush_sel: (50,50) is outside (100..300) → grey color
        assert!(
            (result.circle_instances[0].opacity - 1.0).abs() < 0.01,
            "circle 0: point_sel selected → opacity=1.0, got {}",
            result.circle_instances[0].opacity
        );
        let grey_linear = srgb_to_linear(128.0 / 255.0);
        assert!(
            (result.circle_instances[0].fill_color[0] - grey_linear).abs() < 0.01,
            "circle 0: brush_sel not selected → grey, got {:?}",
            result.circle_instances[0].fill_color
        );

        // Circle 1 at (200,200):
        //   - point_sel: index 1 is NOT selected → opacity = 0.2
        //   - brush_sel: (200,200) is inside (100..300) → red color
        assert!(
            (result.circle_instances[1].opacity - 0.2).abs() < 0.01,
            "circle 1: point_sel not selected → opacity=0.2, got {}",
            result.circle_instances[1].opacity
        );
        assert!(
            (result.circle_instances[1].fill_color[0] - 1.0).abs() < 0.01,
            "circle 1: brush_sel selected → red, got {:?}",
            result.circle_instances[1].fill_color
        );
    }

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
        // Grey (128,128,128) in linear space after srgb_to_linear conversion.
        let grey_linear = srgb_to_linear(128.0 / 255.0);
        let grey_fill = [grey_linear, grey_linear, grey_linear, 1.0];

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

    // ── f64 epsilon comparison in field_value_matches_tooltip ─────────

    #[test]
    fn field_value_matches_number_with_epsilon() {
        // "0.1" + "0.2" = 0.30000000000000004 in f64, which != 0.3 exactly.
        // The epsilon comparison should still match.
        let sum = 0.1_f64 + 0.2_f64;
        assert!(
            field_value_matches_tooltip("0.3", &FieldValue::Number { value: sum }),
            "0.3 should match 0.1+0.2 ({sum}) via epsilon comparison"
        );
    }

    #[test]
    fn field_value_does_not_match_distant_number() {
        // Numbers far apart should not match.
        assert!(
            !field_value_matches_tooltip("1.0", &FieldValue::Number { value: 2.0 }),
            "1.0 should not match 2.0"
        );
    }
}
