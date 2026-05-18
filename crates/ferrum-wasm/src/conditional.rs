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
            .is_some_and(|n| n == *value || (n - *value).abs() < 1e-10),
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

    // ── bug_hunt: field_value_matches_tooltip edge cases ─────────────────

    #[test]
    fn bug_hunt_field_value_matches_nan() {
        // "NaN" tooltip vs FieldValue::Number{NaN}: parse("NaN") returns NaN,
        // then (NaN - NaN).abs() is NaN which is NOT < 1e-10 -> false.
        // This means NaN field values will never match, which is arguably
        // correct (NaN != NaN). Verify that behavior.
        assert!(
            !field_value_matches_tooltip("NaN", &FieldValue::Number { value: f64::NAN }),
            "NaN must not match NaN (NaN != NaN)"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_infinity() { // BUG: (inf - inf).abs() is NaN, not < 1e-10, so equal infinities don't match
        // "Infinity" parses to f64::INFINITY.
        assert!(
            field_value_matches_tooltip("inf", &FieldValue::Number { value: f64::INFINITY }),
            "inf tooltip must match INFINITY field value"
        );
        assert!(
            field_value_matches_tooltip("-inf", &FieldValue::Number { value: f64::NEG_INFINITY }),
            "-inf tooltip must match NEG_INFINITY field value"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_negative_zero() {
        // -0.0 == 0.0 in f64, so they should match.
        assert!(
            field_value_matches_tooltip("0", &FieldValue::Number { value: -0.0_f64 }),
            "0 tooltip must match -0.0 field value (they're equal in f64)"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_empty_string() {
        // Empty tooltip string vs empty FieldValue::String.
        assert!(
            field_value_matches_tooltip("", &FieldValue::String { value: String::new() }),
            "empty tooltip must match empty string field value"
        );
        // Empty tooltip vs Null: "".is_empty() is true so should match.
        assert!(
            field_value_matches_tooltip("", &FieldValue::Null),
            "empty tooltip must match Null field value"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_null_keyword() {
        // "null" string as tooltip vs FieldValue::Null.
        assert!(
            field_value_matches_tooltip("null", &FieldValue::Null),
            "'null' tooltip must match Null field value"
        );
        // "null" vs String("null") should also match.
        assert!(
            field_value_matches_tooltip("null", &FieldValue::String { value: "null".to_string() }),
            "'null' tooltip must match String('null') field value"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_bool() {
        assert!(
            field_value_matches_tooltip("true", &FieldValue::Bool { value: true }),
            "'true' tooltip must match Bool(true)"
        );
        assert!(
            field_value_matches_tooltip("false", &FieldValue::Bool { value: false }),
            "'false' tooltip must match Bool(false)"
        );
        assert!(
            !field_value_matches_tooltip("True", &FieldValue::Bool { value: true }),
            "'True' (capitalized) must not match Bool(true) -- Rust parse::<bool> is case-sensitive"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_unicode() {
        // Unicode strings must match exactly.
        let emoji = "\u{1F600}";
        assert!(
            field_value_matches_tooltip(emoji, &FieldValue::String { value: emoji.to_string() }),
            "unicode emoji must match identical string field value"
        );
        assert!(
            !field_value_matches_tooltip(emoji, &FieldValue::String { value: "smiley".to_string() }),
            "emoji must not match non-emoji string"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_numeric_string_as_string() {
        // A tooltip value "42" compared against FieldValue::String{"42"} must
        // match (direct string equality).
        assert!(
            field_value_matches_tooltip("42", &FieldValue::String { value: "42".to_string() }),
            "'42' tooltip must match String('42') field value"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_very_long_string() {
        let long = "x".repeat(100_000);
        assert!(
            field_value_matches_tooltip(&long, &FieldValue::String { value: long.clone() }),
            "very long strings must still match"
        );
    }

    // ── bug_hunt: resolve_conditionals edge cases ────────────────────────

    #[test]
    fn bug_hunt_resolve_conditionals_nonexistent_selection_name() {
        // A conditional referencing a selection name that doesn't exist in
        // the selections map should be silently skipped (no panic).
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
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![SceneNode::Circle {
                    cx: 50.0,
                    cy: 50.0,
                    r: 5.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
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

        let conditionals = vec![ConditionalEncoding {
            selection_name: "does_not_exist".to_string(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Color {
                value: Color { r: 255, g: 0, b: 0, a: 255 },
            },
            if_not: EncodingValue::Color {
                value: Color { r: 128, g: 128, b: 128, a: 255 },
            },
        }];

        let selections = HashMap::new(); // empty -- no matching selection

        let original_fill = [0.5_f32, 0.5, 0.5, 1.0];
        let base_circles = vec![CircleInstance {
            center: [50.0, 50.0],
            radius: 5.0,
            fill_color: original_fill,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];

        // Must not panic and must leave instances unchanged.
        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);
        assert_eq!(
            result.circle_instances[0].fill_color, original_fill,
            "non-existent selection must not alter instances"
        );
    }

    #[test]
    fn bug_hunt_resolve_conditionals_no_data_indices_skips() {
        // Batch without data_indices should be skipped (no panic).
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
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![SceneNode::Circle {
                    cx: 50.0,
                    cy: 50.0,
                    r: 5.0,
                    style: style.clone(),
                }],
                data_indices: None, // <-- no data indices
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

        let conditionals = vec![ConditionalEncoding {
            selection_name: "sel".to_string(),
            channel: ChannelName::Opacity,
            if_selected: EncodingValue::Opacity { value: 1.0 },
            if_not: EncodingValue::Opacity { value: 0.2 },
        }];

        let mut selections = HashMap::new();
        selections.insert(
            "sel".to_string(),
            SelectionState::Point {
                indices: vec![0],
                field_values: Vec::new(),
            },
        );

        let base_circles = vec![CircleInstance {
            center: [50.0, 50.0],
            radius: 5.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 0.8,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];

        // Must not panic -- batch without data_indices is skipped.
        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);
        assert!(
            (result.circle_instances[0].opacity - 0.8).abs() < 0.01,
            "batch without data_indices must not be modified"
        );
    }

    #[test]
    fn bug_hunt_apply_size_to_circle_computes_radius() {
        // Size -> radius conversion: radius = sqrt(size / PI).
        let mut inst = CircleInstance {
            center: [0.0, 0.0],
            radius: 5.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        apply_value_to_circle(
            &mut inst,
            &ChannelName::Size,
            &EncodingValue::Size { value: std::f64::consts::PI * 100.0 },
        );
        // Expected: sqrt(PI*100 / PI) = sqrt(100) = 10
        assert!(
            (inst.radius - 10.0).abs() < 0.1,
            "size PI*100 should give radius ~10, got {}",
            inst.radius
        );
    }

    #[test]
    fn bug_hunt_apply_unknown_channel_is_noop() {
        // Applying an X-channel value (not Color/Opacity/Size) to a circle
        // should be a no-op (falls through the match).
        let mut inst = CircleInstance {
            center: [0.0, 0.0],
            radius: 5.0,
            fill_color: [1.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 0.7,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let original_fill = inst.fill_color;
        let original_opacity = inst.opacity;
        let original_radius = inst.radius;

        apply_value_to_circle(
            &mut inst,
            &ChannelName::X,
            &EncodingValue::Opacity { value: 0.1 },
        );
        assert_eq!(inst.fill_color, original_fill, "X channel must not change fill");
        assert!((inst.opacity - original_opacity).abs() < 1e-10, "X channel must not change opacity");
        assert!((inst.radius - original_radius).abs() < 1e-10, "X channel must not change radius");
    }

    #[test]
    fn bug_hunt_interval_selection_rect_center_containment() {
        // Rect mark containment uses center (x + w/2, y + h/2).
        // Verify that a rect whose corner is outside but center is inside is selected.
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

        // Rect at x=90, y=90, w=20, h=20 -> center=(100, 100).
        // Brush from (95, 95) to (105, 105) -- contains center (100,100).
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
                nodes: vec![SceneNode::Rect {
                    x: 90.0,
                    y: 90.0,
                    w: 20.0,
                    h: 20.0,
                    corner_radius: 0.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
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

        let conditionals = vec![ConditionalEncoding {
            selection_name: "brush".to_string(),
            channel: ChannelName::Opacity,
            if_selected: EncodingValue::Opacity { value: 1.0 },
            if_not: EncodingValue::Opacity { value: 0.1 },
        }];

        let mut selections = HashMap::new();
        selections.insert(
            "brush".to_string(),
            SelectionState::Interval {
                x_range: Some((95.0, 105.0)),
                y_range: Some((95.0, 105.0)),
            },
        );

        let base_rects = vec![RectInstance {
            position: [90.0, 90.0],
            size: [20.0, 20.0],
            corner_radius: 0.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 0.5,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];

        let result = resolve_conditionals(&panels, &conditionals, &selections, &[], &base_rects);
        assert!(
            (result.rect_instances[0].opacity - 1.0).abs() < 0.01,
            "rect whose center is inside brush must be selected (opacity=1.0), got {}",
            result.rect_instances[0].opacity
        );
    }

    #[test]
    fn bug_hunt_point_selection_no_tooltips_falls_back_to_index() {
        // Point selection with field_values but no tooltips on the batch
        // must fall back to index-based matching (the `unwrap_or(false)` path).
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
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 },
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
                tooltips: None, // <-- no tooltips
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

        let conditionals = vec![ConditionalEncoding {
            selection_name: "sel".to_string(),
            channel: ChannelName::Opacity,
            if_selected: EncodingValue::Opacity { value: 1.0 },
            if_not: EncodingValue::Opacity { value: 0.2 },
        }];

        // Point selection with field_values but batch has no tooltips.
        // The code checks field_values.is_empty() first; since it's not empty,
        // it tries tooltip matching which returns false (no tooltips). So the
        // mark should be treated as NOT selected and get if_not opacity.
        let mut selections = HashMap::new();
        selections.insert(
            "sel".to_string(),
            SelectionState::Point {
                indices: vec![0],
                field_values: vec![
                    ("group".to_string(), FieldValue::String { value: "a".to_string() }),
                ],
            },
        );

        let base_circles = vec![
            CircleInstance {
                center: [50.0, 50.0],
                radius: 5.0,
                fill_color: [0.0; 4],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [150.0, 50.0],
                radius: 5.0,
                fill_color: [0.0; 4],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];

        let result = resolve_conditionals(&panels, &conditionals, &selections, &base_circles, &[]);
        // When field_values are present but tooltips are missing, the mark
        // is NOT matched by field values. So both marks get if_not = 0.2.
        assert!(
            (result.circle_instances[0].opacity - 0.2).abs() < 0.01,
            "mark 0 with no tooltip must get if_not opacity (0.2), got {}",
            result.circle_instances[0].opacity
        );
        assert!(
            (result.circle_instances[1].opacity - 0.2).abs() < 0.01,
            "mark 1 with no tooltip must get if_not opacity (0.2), got {}",
            result.circle_instances[1].opacity
        );
    }
}
