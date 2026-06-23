use ferrum_scene::{ConditionalEncoding, EncodingValue, FieldValue, MarkBatch, Panel, SceneNode};

use crate::scene_load::{
    color_to_linear, stroke_dash_index, CircleInstance, DrawKind, PackedBatchMeta, RectInstance,
};
use crate::selection_state::SelectionState;

use std::collections::HashMap;

/// Opacity applied to marks that fall *outside* a D6 crossfilter brush.
///
/// Shared by the crossfilter dimming path here and the `WasmRenderer`
/// call site in `lib.rs` so the "filtered out" appearance is defined once.
pub(crate) const CROSSFILTER_DIM_OPACITY: f64 = 0.15;

// ── wasm32-only: apply conditionals and re-render ────────────────────────────

/// Resolve conditional encodings against the current selection state,
/// rebuild GPU buffers with updated instance colors, render a frame, and
/// return the serialized selection state JSON.
///
/// Called by both `handle_click` and `handle_drag` after they update the
/// selection state. Centralising this ensures that if `SceneData` gains a
/// field, only one site needs updating.
#[cfg(target_arch = "wasm32")]
pub(crate) fn apply_conditionals_and_render(
    loaded_scene: &ferrum_scene::SceneGraph,
    loaded_data: &crate::scene_load::SceneData,
    loaded_buffers: &mut crate::render::GpuBuffers,
    interaction_state: &crate::selection_state::InteractionState,
    gpu: &crate::gpu::GpuContext,
    pipelines: &crate::pipelines::RenderPipelines,
) -> Result<String, wasm_bindgen::JsValue> {
    // Apply conditional encodings to produce dimmed/highlighted instance colors.
    let conditionals = &loaded_scene.interaction.conditionals;
    let updates = resolve_conditionals_with_packed(
        &loaded_scene.panels,
        conditionals,
        &interaction_state.selections,
        &loaded_data.circle_instances,
        &loaded_data.rect_instances,
        &loaded_data.packed_batch_meta,
    );

    // Re-upload only the instance buffers that changed (circle and rect
    // colors). Mesh, static mesh, annotation mesh, images, and uniforms are
    // unaffected by selection changes and do not need to be re-uploaded.
    loaded_buffers.update_instances(gpu, &updates.circle_instances, &updates.rect_instances);
    crate::render::render_frame(gpu, pipelines, loaded_buffers, loaded_data.background)
        .map_err(wasm_bindgen::JsValue::from)?;

    // Serialize current selection state for Python sync.
    let state_json = interaction_state.to_json();
    Ok(state_json)
}

pub struct ConditionalUpdates {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
}

/// Compute the total number of packed circle and rect instances.
///
/// The loader (load_scene_with_packed) builds the flat instance arrays
/// packed-FIRST: all packed instances are prepended, then non-packed instances
/// are appended in scene-graph walk order. Non-packed batches therefore start
/// at the total packed count. Both `apply_crossfilter_to_panel` and
/// `resolve_conditionals_with_packed` need this same basis; centralising it
/// here keeps the two paths in lock-step and removes a copy-paste drift risk.
fn packed_base_offsets(packed_batch_meta: &HashMap<(u32, u32), PackedBatchMeta>) -> (usize, usize) {
    let total_packed_circles: usize = packed_batch_meta
        .values()
        .filter(|m| m.kind == DrawKind::Circle)
        .map(|m| m.instance_count)
        .sum();
    let total_packed_rects: usize = packed_batch_meta
        .values()
        .filter(|m| m.kind == DrawKind::Rect)
        .map(|m| m.instance_count)
        .sum();
    (total_packed_circles, total_packed_rects)
}

/// Apply a crossfilter dim to a single target panel's marks (D6 `Filter`
/// binding).
///
/// A `transform_filter(param)` declares that marks outside the brushed interval
/// should be removed from the bound panel. We reuse the exact spatial-
/// containment dimming selections already use: the brushed extent is supplied
/// as an `Interval` selection in the *target panel's pixel space* (re-projected
/// through the shared data domain by the caller), and a synthesized opacity
/// conditional dims marks whose centers fall outside it.
///
/// `circles`/`rects` are mutated in place over the full instance buffers; only
/// the instances belonging to `target_panel` are touched (offsets are advanced
/// across earlier panels exactly as `resolve_conditionals_with_packed` does, so
/// the two paths stay in lock-step on the packed/non-packed layout).
///
/// Consumed by the wasm32 `WasmRenderer`; gated to wasm32 + test so the
/// non-test host build does not flag it as dead code (host coverage is via the
/// `#[cfg(test)]` unit test below).
#[cfg(any(target_arch = "wasm32", test))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_crossfilter_to_panel(
    panels: &[Panel],
    target_panel: usize,
    selection: &SelectionState,
    dimmed_opacity: f64,
    circles: &mut [CircleInstance],
    rects: &mut [RectInstance],
    packed_batch_meta: &HashMap<(u32, u32), PackedBatchMeta>,
) {
    // A filtered-in mark keeps its opacity; a filtered-out mark is dimmed.
    // The targeted channel is implied by the value variant (`Opacity`); the
    // apply path dispatches on the value alone.
    let if_selected = EncodingValue::Opacity { value: 1.0 };
    let if_not = EncodingValue::Opacity { value: dimmed_opacity };

    let (total_packed_circles, total_packed_rects) = packed_base_offsets(packed_batch_meta);
    let mut circle_offset = total_packed_circles;
    let mut rect_offset = total_packed_rects;

    for (panel_idx, panel) in panels.iter().enumerate() {
        for (batch_idx, batch) in panel.marks.iter().enumerate() {
            let key = (panel_idx as u32, batch_idx as u32);

            if let Some(meta) = packed_batch_meta.get(&key) {
                // Packed batch: use meta.instance_start (absolute index).
                // Do NOT advance circle_offset/rect_offset for packed batches.
                if panel_idx == target_panel && !matches!(selection, SelectionState::Empty) {
                    apply_conditional_to_packed(
                        &if_selected, &if_not, selection, meta, circles, rects,
                    );
                }
            } else {
                // Non-packed batch: instances live at circle_offset/rect_offset
                // (after all packed instances in the flat array).
                let (n_circles, n_rects) = count_instances(batch);
                if panel_idx == target_panel
                    && !matches!(selection, SelectionState::Empty)
                {
                    if let Some(indices) = &batch.data_indices {
                        let mut bufs = InstanceBuffers {
                            circles,
                            circle_offset,
                            rects,
                            rect_offset,
                        };
                        apply_conditional_to_batch(
                            &if_selected, &if_not, selection, indices, batch, &mut bufs,
                        );
                    }
                }
                circle_offset += n_circles;
                rect_offset += n_rects;
            }
        }
    }
}

/// Mutable references to the circle and rect instance buffers together with
/// the current write offsets. Groups the four mutable-state parameters that
/// `apply_conditional_to_batch` needs, keeping its signature manageable.
pub(crate) struct InstanceBuffers<'a> {
    pub(crate) circles: &'a mut [CircleInstance],
    pub(crate) circle_offset: usize,
    pub(crate) rects: &'a mut [RectInstance],
    pub(crate) rect_offset: usize,
}

pub fn resolve_conditionals(
    panels: &[Panel],
    conditionals: &[ConditionalEncoding],
    selections: &HashMap<String, SelectionState>,
    base_circles: &[CircleInstance],
    base_rects: &[RectInstance],
) -> ConditionalUpdates {
    resolve_conditionals_with_packed(
        panels,
        conditionals,
        selections,
        base_circles,
        base_rects,
        &HashMap::new(),
    )
}

pub fn resolve_conditionals_with_packed(
    panels: &[Panel],
    conditionals: &[ConditionalEncoding],
    selections: &HashMap<String, SelectionState>,
    base_circles: &[CircleInstance],
    base_rects: &[RectInstance],
    packed_batch_meta: &HashMap<(u32, u32), PackedBatchMeta>,
) -> ConditionalUpdates {
    let mut circles = base_circles.to_vec();
    let mut rects = base_rects.to_vec();

    // The loader (load_scene_with_packed) builds the flat instance arrays
    // packed-FIRST: all packed instances are prepended by
    // `unpack_binary_instances`, then non-packed instances are appended in
    // scene-graph walk order. Non-packed batches therefore start at the total
    // packed count, not at 0. Initialise the non-packed running offsets here
    // so they address the correct flat-array positions regardless of the scene
    // order of packed vs. non-packed batches.
    //
    // Packed batches use meta.instance_start (an absolute index, already
    // correct) and do not advance these offsets.
    let (total_packed_circles, total_packed_rects) = packed_base_offsets(packed_batch_meta);
    let mut circle_offset = total_packed_circles;
    let mut rect_offset = total_packed_rects;

    for (panel_idx, panel) in panels.iter().enumerate() {
        for (batch_idx, batch) in panel.marks.iter().enumerate() {
            let key = (panel_idx as u32, batch_idx as u32);

            if let Some(meta) = packed_batch_meta.get(&key) {
                // Packed batch: instances are in circle_instances or
                // rect_instances at meta.instance_start..+instance_count.
                // batch.nodes is empty for packed batches, so we must use
                // the packed metadata to locate and process instances.
                // Note: packed batches do NOT advance circle_offset/rect_offset
                // because those accumulators track non-packed positions only.
                for cond in conditionals {
                    let Some(sel) = selections.get(&cond.selection_name) else {
                        continue;
                    };
                    if matches!(sel, SelectionState::Empty) {
                        continue;
                    }
                    apply_conditional_to_packed(
                        &cond.if_selected,
                        &cond.if_not,
                        sel,
                        meta,
                        &mut circles,
                        &mut rects,
                    );
                }
            } else {
                // Non-packed batch: instances live at circle_offset/rect_offset
                // in the flat arrays (after all packed instances).
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
    }

    ConditionalUpdates {
        circle_instances: circles,
        rect_instances: rects,
    }
}

/// Decide whether the `i`-th instance of a packed batch is selected.
///
/// Mirrors the unpacked [`apply_conditional_to_batch`] membership rules so the
/// two paths stay in lock-step:
/// - `Point` with non-empty `field_values` → match against the packed tooltip
///   string table (the field-projected legend / linked-selection path);
/// - `Point` with `indices` only → data-index membership;
/// - `Interval` → spatial containment of the instance center (via `with_pos`).
///
/// `with_pos` supplies the instance center lazily so the caller reads it from
/// whichever buffer (circle or rect) holds this packed batch.
fn packed_instance_selected(
    sel: &SelectionState,
    meta: &PackedBatchMeta,
    i: usize,
    with_pos: impl Fn() -> Option<(f64, f64)>,
) -> bool {
    match sel {
        SelectionState::Empty => false,
        SelectionState::Interval { .. } => {
            with_pos().is_some_and(|(x, y)| sel.contains_point(x, y))
        }
        SelectionState::Point { field_values, .. } if !field_values.is_empty() => {
            // Field-value matching against the packed tooltip table — the same
            // semantics the unpacked path applies via `batch.tooltips`. This is
            // what makes legend toggle and field-projected linked selection work
            // on packed (>= 1000-mark) batches.
            let Some(bytes) = meta.tooltip_bytes.as_deref() else {
                return false;
            };
            tooltip_matches(field_values, |fname| {
                crate::scene_load::tooltip_field_value(bytes, i, fname)
            })
        }
        SelectionState::Point { indices, .. } => meta
            .data_indices
            .as_ref()
            .and_then(|dis| dis.get(i))
            .is_some_and(|&di| indices.contains(&(di as usize))),
    }
}

/// Apply conditional encoding to a packed batch's instances.
///
/// For Interval selections, checks spatial containment of each instance's
/// center against the selection rectangle. For Point selections this matches
/// either by `field_values` (against the packed tooltip table) or by data
/// index — the same rules the unpacked path uses.
///
/// The visual effect is determined by the [`EncodingValue`] variant alone
/// (`apply_value_to_circle`/`apply_value_to_rect`); the targeted channel is
/// `value.channel()`. There is no separate channel argument to disagree.
pub(crate) fn apply_conditional_to_packed(
    if_selected: &EncodingValue,
    if_not: &EncodingValue,
    sel: &SelectionState,
    meta: &PackedBatchMeta,
    circles: &mut [CircleInstance],
    rects: &mut [RectInstance],
) {
    let start = meta.instance_start;
    let count = meta.instance_count;

    match meta.kind {
        DrawKind::Circle => {
            for i in 0..count {
                let idx = start + i;
                let selected = packed_instance_selected(sel, meta, i, || {
                    circles
                        .get(idx)
                        .map(|ci| (ci.center[0] as f64, ci.center[1] as f64))
                });
                let value = if selected { if_selected } else { if_not };
                if let Some(inst) = circles.get_mut(idx) {
                    apply_value_to_circle(inst, value);
                }
            }
        }
        DrawKind::Rect => {
            for i in 0..count {
                let idx = start + i;
                let selected = packed_instance_selected(sel, meta, i, || {
                    rects.get(idx).map(|ri| {
                        let cx = ri.position[0] as f64 + ri.size[0] as f64 / 2.0;
                        let cy = ri.position[1] as f64 + ri.size[1] as f64 / 2.0;
                        (cx, cy)
                    })
                });
                let value = if selected { if_selected } else { if_not };
                if let Some(inst) = rects.get_mut(idx) {
                    apply_value_to_rect(inst, value);
                }
            }
        }
    }
}

pub(crate) fn count_instances(batch: &MarkBatch) -> (usize, usize) {
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

pub(crate) fn apply_conditional_to_batch(
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
                        tooltip_matches(field_values, |fname| {
                            tip.fields
                                .iter()
                                .find(|f| f.name == fname)
                                .map(|f| f.value.clone())
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
                    apply_value_to_circle(inst, value);
                }
                ci += 1;
            }
            SceneNode::Rect { .. } => {
                if let Some(inst) = bufs.rects.get_mut(bufs.rect_offset + ri) {
                    apply_value_to_rect(inst, value);
                }
                ri += 1;
            }
            _ => {}
        }
    }
}

/// Mutable view over the fill/stroke fields shared by [`CircleInstance`] and
/// [`RectInstance`].
///
/// The seven conditional-encoding arms that are identical for both geometries
/// (Color, Opacity, StrokeWidth, StrokeDash, StrokeOpacity, FillOpacity, Angle)
/// mutate exactly these fields. Routing both instance types through one view
/// means [`apply_value_common`] has a single copy of those arms, so a new
/// shared channel cannot be added to circles and silently forgotten for rects.
struct FillStrokeView<'a> {
    fill_color: &'a mut [f32; 4],
    opacity: &'a mut f32,
    stroke_width: &'a mut f32,
    stroke_dash: &'a mut f32,
    stroke_opacity: &'a mut f32,
    angle: &'a mut f32,
}

impl<'a> FillStrokeView<'a> {
    fn of_circle(inst: &'a mut CircleInstance) -> Self {
        Self {
            fill_color: &mut inst.fill_color,
            opacity: &mut inst.opacity,
            stroke_width: &mut inst.stroke_width,
            stroke_dash: &mut inst.stroke_dash,
            stroke_opacity: &mut inst.stroke_opacity,
            angle: &mut inst.angle,
        }
    }

    fn of_rect(inst: &'a mut RectInstance) -> Self {
        Self {
            fill_color: &mut inst.fill_color,
            opacity: &mut inst.opacity,
            stroke_width: &mut inst.stroke_width,
            stroke_dash: &mut inst.stroke_dash,
            stroke_opacity: &mut inst.stroke_opacity,
            angle: &mut inst.angle,
        }
    }
}

/// Apply the conditional-encoding arms that are identical for circles and rects,
/// dispatching on the [`EncodingValue`] variant ALONE.
///
/// Each `EncodingValue` variant targets exactly one channel
/// (`EncodingValue::channel()` is the source of truth), so the value uniquely
/// determines the effect — there is no separate `channel` argument to disagree
/// with it. Returns `true` when a shared arm handled the value; returns `false`
/// for `Size` (the caller applies per-geometry size logic) and for `Field` /
/// any value with no rendered effect on these fields.
fn apply_value_common(value: &EncodingValue, view: &mut FillStrokeView<'_>) -> bool {
    match value {
        EncodingValue::Color { value: c } => {
            *view.fill_color = color_to_linear(c, 1.0);
            true
        }
        EncodingValue::Opacity { value: o } => {
            *view.opacity = *o as f32;
            true
        }
        EncodingValue::StrokeWidth { value: w } => {
            *view.stroke_width = *w as f32;
            true
        }
        EncodingValue::StrokeDash { value: pattern } => {
            *view.stroke_dash = stroke_dash_index(&Some(pattern.clone()));
            true
        }
        EncodingValue::StrokeOpacity { value: o } => {
            *view.stroke_opacity = *o as f32;
            true
        }
        EncodingValue::FillOpacity { value: o } => {
            view.fill_color[3] = *o as f32;
            true
        }
        EncodingValue::Angle { value: a } => {
            *view.angle = *a as f32;
            true
        }
        // Size is per-geometry (radius vs width/height); Field/any other value
        // has no shared-field effect. Both are left to the caller.
        EncodingValue::Size { .. } | EncodingValue::Field { .. } => false,
    }
}

fn apply_value_to_circle(inst: &mut CircleInstance, value: &EncodingValue) {
    // Only the Size arm differs per geometry; everything else is shared.
    if let EncodingValue::Size { value: s } = value {
        inst.radius = (*s as f32 / std::f32::consts::PI).sqrt();
        return;
    }
    let mut view = FillStrokeView::of_circle(inst);
    apply_value_common(value, &mut view);
}

/// Test whether a mark's tooltip satisfies all (field, value) constraints,
/// using a caller-supplied comparison predicate.
///
/// This is the shared loop helper for BOTH the conditional-highlight path and
/// the click-selection membership scan (`collect_matching_indices` in
/// `selection_state.rs`).  Both call sites use the typed comparison via
/// [`tooltip_matches`], so a clicked "42.0" co-selects a candidate "42"
/// consistently across highlighting and selection.
///
/// `lookup` is a closure that, given a field name, returns the tooltip string
/// for that field (or `None` if absent).
pub(crate) fn tooltip_matches_with_cmp(
    field_values: &[(String, FieldValue)],
    lookup: impl Fn(&str) -> Option<String>,
    cmp: impl Fn(&str, &FieldValue) -> bool,
) -> bool {
    field_values.iter().all(|(fname, fval)| {
        lookup(fname.as_str())
            .as_deref()
            .is_some_and(|tv| cmp(tv, fval))
    })
}

/// Convenience wrapper: test membership using typed comparison
/// (`field_value_matches_tooltip`).
///
/// Used by the conditional-highlight path (`packed_instance_selected`,
/// `apply_conditional_to_batch`) and the click-selection membership scan
/// (`collect_matching_indices`).  Typed comparison means "42" matches
/// `FieldValue::Number { value: 42.0 }` because both parse to the same f64.
pub(crate) fn tooltip_matches(
    field_values: &[(String, FieldValue)],
    lookup: impl Fn(&str) -> Option<String>,
) -> bool {
    tooltip_matches_with_cmp(field_values, lookup, field_value_matches_tooltip)
}

/// Compare a tooltip string value against a typed `FieldValue` using typed,
/// epsilon-tolerant comparison.
///
/// Tooltip fields are always stored as strings in the scene graph. This
/// function bridges the gap by parsing the string according to the
/// `FieldValue` variant so cross-panel matching works even when one panel
/// stores `"42"` and the selection carries `FieldValue::Number { value: 42.0 }`.
///
/// Used by the conditional-highlight path via [`tooltip_matches`].
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

fn apply_value_to_rect(inst: &mut RectInstance, value: &EncodingValue) {
    // Only the Size arm differs per geometry.
    // Without orientation context (h-bar vs v-bar) we apply the size to both
    // width and height so the conditional always has a visible effect. The
    // rendering layer controls which dimension is the data extent; the other is
    // the band width set by the scale. Applying to both is the conservative
    // fallback that matches the circle counterpart's intent (scale the mark
    // proportionally to the encoded value).
    if let EncodingValue::Size { value: s } = value {
        inst.size = [*s as f32, *s as f32];
        return;
    }
    let mut view = FillStrokeView::of_rect(inst);
    apply_value_common(value, &mut view);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_load::srgb_to_linear;
    // `ChannelName` is still a field on `ConditionalEncoding` (kept for wire
    // compatibility), so test fixtures construct it; the runtime apply path no
    // longer dispatches on it (WASM-05: value is the source of truth).
    use ferrum_scene::{ChannelName, Color, FieldValue};

    // ── D6 crossfilter: apply_crossfilter_to_panel dims only the target panel ─

    #[test]
    fn crossfilter_dims_only_target_panel_outside_interval() {
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
        let mk_panel = |cx: f64| Panel {
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
                nodes: vec![SceneNode::Circle { cx, cy: 50.0, r: 5.0, style: style.clone() }],
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
        };
        // Panel 0 (source) mark at x=10 (inside the brush); panel 1 (target)
        // mark at x=400 (outside the re-projected interval).
        let panels = vec![mk_panel(10.0), mk_panel(400.0)];

        let mut circles = vec![
            CircleInstance {
                center: [10.0, 50.0], radius: 5.0,
                fill_color: [0.0; 4], stroke_color: [0.0; 4],
                stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                stroke_dash: 0.0, angle: 0.0,
            },
            CircleInstance {
                center: [400.0, 50.0], radius: 5.0,
                fill_color: [0.0; 4], stroke_color: [0.0; 4],
                stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                stroke_dash: 0.0, angle: 0.0,
            },
        ];
        let mut rects: Vec<RectInstance> = vec![];

        // Brush covers x in [0, 100] on the target panel's pixel space.
        let sel = SelectionState::Interval {
            x_range: Some((0.0, 100.0)),
            y_range: None,
        };

        apply_crossfilter_to_panel(
            &panels, 1, &sel, 0.15, &mut circles, &mut rects, &HashMap::new(),
        );

        // Source panel (index 0) mark must be untouched (full opacity).
        assert!(
            (circles[0].opacity - 1.0).abs() < 1e-6,
            "source-panel mark must not be dimmed by crossfilter, got {}",
            circles[0].opacity
        );
        // Target panel mark at x=400 is outside [0,100] → dimmed.
        assert!(
            (circles[1].opacity - 0.15).abs() < 1e-6,
            "target-panel mark outside interval must be dimmed, got {}",
            circles[1].opacity
        );
    }

    #[test]
    fn crossfilter_keeps_target_marks_inside_interval() {
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
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![SceneNode::Circle { cx: 50.0, cy: 50.0, r: 5.0, style }],
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
        let mut circles = vec![CircleInstance {
            center: [50.0, 50.0], radius: 5.0,
            fill_color: [0.0; 4], stroke_color: [0.0; 4],
            stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
            stroke_dash: 0.0, angle: 0.0,
        }];
        let mut rects: Vec<RectInstance> = vec![];
        // Brush [0, 100] contains the mark at x=50.
        let sel = SelectionState::Interval { x_range: Some((0.0, 100.0)), y_range: None };
        apply_crossfilter_to_panel(
            &panels, 0, &sel, 0.15, &mut circles, &mut rects, &HashMap::new(),
        );
        assert!(
            (circles[0].opacity - 1.0).abs() < 1e-6,
            "mark inside interval must stay at full opacity, got {}",
            circles[0].opacity
        );
    }

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
    fn bug_hunt_apply_field_value_is_noop() {
        // Value-is-source-of-truth: the only value with no rendered effect on a
        // circle's shared fill/stroke fields (and no per-geometry size effect)
        // is `Field` — it names a data field rather than a literal value, so the
        // apply path leaves the instance untouched.
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
            &EncodingValue::Field { name: "category".to_string() },
        );
        assert_eq!(inst.fill_color, original_fill, "Field value must not change fill");
        assert!((inst.opacity - original_opacity).abs() < 1e-10, "Field value must not change opacity");
        assert!((inst.radius - original_radius).abs() < 1e-10, "Field value must not change radius");
    }

    // ── W7: StrokeWidth and StrokeDash conditional values ────────────────
    //
    // WASM-05: `apply_value_*` now dispatches on the `EncodingValue` variant
    // alone (the value is the single source of truth for the targeted channel —
    // `EncodingValue::channel()` proves the 1:1 mapping). There is no separate
    // channel argument that could be ignored or disagree, so a `StrokeWidth`
    // value sets stroke width unconditionally.

    /// apply_value_to_circle must set stroke_width when value is StrokeWidth.
    #[test]
    fn w7_stroke_width_value_applied_to_circle() {
        let mut inst = CircleInstance {
            center: [0.0, 0.0],
            radius: 5.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        apply_value_to_circle(
            &mut inst,
            &EncodingValue::StrokeWidth { value: 3.5 },
        );
        assert!(
            (inst.stroke_width - 3.5).abs() < 0.01,
            "StrokeWidth value must set stroke_width, got {}",
            inst.stroke_width
        );
    }

    /// apply_value_to_circle must set stroke_dash when value is StrokeDash.
    #[test]
    fn w7_stroke_dash_value_applied_to_circle() {
        let mut inst = CircleInstance {
            center: [0.0, 0.0],
            radius: 5.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        // [6.0, 3.0] maps to dash index 1.0
        apply_value_to_circle(
            &mut inst,
            &EncodingValue::StrokeDash { value: vec![6.0, 3.0] },
        );
        assert!(
            (inst.stroke_dash - 1.0).abs() < 0.01,
            "StrokeDash [6,3] must set stroke_dash index to 1.0, got {}",
            inst.stroke_dash
        );
    }

    /// apply_value_to_rect must set stroke_width when value is StrokeWidth.
    #[test]
    fn w7_stroke_width_value_applied_to_rect() {
        let mut inst = RectInstance {
            position: [0.0, 0.0],
            size: [10.0, 10.0],
            corner_radius: 0.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        apply_value_to_rect(
            &mut inst,
            &EncodingValue::StrokeWidth { value: 2.0 },
        );
        assert!(
            (inst.stroke_width - 2.0).abs() < 0.01,
            "StrokeWidth value must set rect stroke_width, got {}",
            inst.stroke_width
        );
    }

    /// apply_value_to_rect must set stroke_dash when value is StrokeDash.
    #[test]
    fn w7_stroke_dash_value_applied_to_rect() {
        let mut inst = RectInstance {
            position: [0.0, 0.0],
            size: [10.0, 10.0],
            corner_radius: 0.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        // [2.0, 3.0] maps to dash index 2.0
        apply_value_to_rect(
            &mut inst,
            &EncodingValue::StrokeDash { value: vec![2.0, 3.0] },
        );
        assert!(
            (inst.stroke_dash - 2.0).abs() < 0.01,
            "StrokeDash [2,3] must set rect stroke_dash index to 2.0, got {}",
            inst.stroke_dash
        );
    }

    /// WASM-05 parity: every shared arm produces the SAME result on a circle and
    /// a rect (so the shared logic cannot drift between the two geometries).
    #[test]
    fn w7_shared_arms_circle_rect_parity() {
        use ferrum_scene::Color;

        let circle_base = CircleInstance {
            center: [0.0, 0.0],
            radius: 5.0,
            fill_color: [0.1, 0.2, 0.3, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let rect_base = RectInstance {
            position: [0.0, 0.0],
            size: [10.0, 10.0],
            corner_radius: 0.0,
            fill_color: [0.1, 0.2, 0.3, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };

        let shared_values = [
            EncodingValue::Color { value: Color { r: 200, g: 100, b: 50, a: 255 } },
            EncodingValue::Opacity { value: 0.4 },
            EncodingValue::StrokeWidth { value: 2.5 },
            EncodingValue::StrokeDash { value: vec![6.0, 3.0] },
            EncodingValue::StrokeOpacity { value: 0.6 },
            EncodingValue::FillOpacity { value: 0.7 },
            EncodingValue::Angle { value: 30.0 },
        ];

        for value in &shared_values {
            let mut c = circle_base;
            let mut r = rect_base;
            apply_value_to_circle(&mut c, value);
            apply_value_to_rect(&mut r, value);
            assert_eq!(
                c.fill_color, r.fill_color,
                "fill_color must match across geometries for {value:?}"
            );
            assert!((c.opacity - r.opacity).abs() < 1e-6, "opacity parity for {value:?}");
            assert!(
                (c.stroke_width - r.stroke_width).abs() < 1e-6,
                "stroke_width parity for {value:?}"
            );
            assert!(
                (c.stroke_dash - r.stroke_dash).abs() < 1e-6,
                "stroke_dash parity for {value:?}"
            );
            assert!(
                (c.stroke_opacity - r.stroke_opacity).abs() < 1e-6,
                "stroke_opacity parity for {value:?}"
            );
            assert!((c.angle - r.angle).abs() < 1e-6, "angle parity for {value:?}");
        }
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

    // ── Bug 3: resolve_conditionals_with_packed handles packed batches ──

    #[test]
    fn packed_circles_interval_conditional_applies() {
        use ferrum_scene::{
            BlendMode, CoordKind, MarkBatchKind, Panel, Rect,
        };

        // Panel with one batch that has empty nodes (packed batch).
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
                nodes: vec![], // empty — packed batch
                data_indices: None,
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

        // Three packed circle instances:
        //   (20, 30) inside brush, (100, 100) outside, (30, 40) inside.
        let neutral = [0.0_f32, 0.0, 0.0, 1.0];
        let base_circles = vec![
            CircleInstance {
                center: [20.0, 30.0], radius: 5.0, fill_color: neutral,
                stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 1.0,
                stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
            CircleInstance {
                center: [100.0, 100.0], radius: 5.0, fill_color: neutral,
                stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 1.0,
                stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
            CircleInstance {
                center: [30.0, 40.0], radius: 5.0, fill_color: neutral,
                stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 1.0,
                stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
        ];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: None,
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        let result = resolve_conditionals_with_packed(
            &panels, &conditionals, &selections, &base_circles, &[], &packed_meta,
        );

        let red_r = srgb_to_linear(1.0);
        let grey_linear = srgb_to_linear(128.0 / 255.0);

        // Circle 0 at (20,30) is inside brush — should be red.
        assert!(
            (result.circle_instances[0].fill_color[0] - red_r).abs() < 0.01,
            "packed circle at (20,30) should be red (inside brush), got {:?}",
            result.circle_instances[0].fill_color
        );

        // Circle 1 at (100,100) is outside brush — should be grey.
        assert!(
            (result.circle_instances[1].fill_color[0] - grey_linear).abs() < 0.01,
            "packed circle at (100,100) should be grey (outside brush), got {:?}",
            result.circle_instances[1].fill_color
        );

        // Circle 2 at (30,40) is inside brush — should be red.
        assert!(
            (result.circle_instances[2].fill_color[0] - red_r).abs() < 0.01,
            "packed circle at (30,40) should be red (inside brush), got {:?}",
            result.circle_instances[2].fill_color
        );
    }

    #[test]
    fn packed_rects_interval_conditional_applies() {
        use ferrum_scene::{
            BlendMode, CoordKind, MarkBatchKind, Panel, Rect,
        };

        // Panel with one batch that has empty nodes (packed batch).
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
                kind: MarkBatchKind::Bar,
                nodes: vec![], // empty — packed batch
                data_indices: None,
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
                x_range: Some((10.0, 50.0)),
                y_range: Some((10.0, 50.0)),
            },
        );

        // Two packed rect instances:
        //   rect 0: pos=(15,15) size=(20,20) -> center=(25,25) inside brush
        //   rect 1: pos=(100,100) size=(30,30) -> center=(115,115) outside brush
        let base_rects = vec![
            RectInstance {
                position: [15.0, 15.0], size: [20.0, 20.0], corner_radius: 0.0,
                fill_color: [0.0; 4], stroke_color: [0.0; 4], stroke_width: 0.0,
                opacity: 0.5, stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
            RectInstance {
                position: [100.0, 100.0], size: [30.0, 30.0], corner_radius: 0.0,
                fill_color: [0.0; 4], stroke_color: [0.0; 4], stroke_width: 0.0,
                opacity: 0.5, stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
        ];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: None,
                tooltip_bytes: None,
                kind: DrawKind::Rect,
                instance_start: 0,
                instance_count: 2,
            },
        );

        let result = resolve_conditionals_with_packed(
            &panels, &conditionals, &selections, &[], &base_rects, &packed_meta,
        );

        // Rect 0 center (25,25) is inside brush — opacity should be 1.0.
        assert!(
            (result.rect_instances[0].opacity - 1.0).abs() < 0.01,
            "packed rect at (25,25) should be selected (opacity=1.0), got {}",
            result.rect_instances[0].opacity
        );

        // Rect 1 center (115,115) is outside brush — opacity should be 0.1.
        assert!(
            (result.rect_instances[1].opacity - 0.1).abs() < 0.01,
            "packed rect at (115,115) should not be selected (opacity=0.1), got {}",
            result.rect_instances[1].opacity
        );
    }

    #[test]
    fn packed_batch_offset_does_not_corrupt_subsequent_nonpacked_batch() {
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

        // Panel with two batches: first is packed (2 circles), second is non-packed (1 circle).
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![
                // Batch 0: packed (empty nodes).
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![],
                    data_indices: None,
                    tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal,
                    descriptions: None, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                },
                // Batch 1: non-packed (1 circle in nodes).
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![
                        SceneNode::Circle { cx: 200.0, cy: 200.0, r: 5.0, style: style.clone() },
                    ],
                    data_indices: Some(vec![99]),
                    tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal,
                    descriptions: None, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                },
            ],
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
                indices: vec![99],
                field_values: Vec::new(),
            },
        );

        // 2 packed circles + 1 non-packed circle = 3 total.
        let neutral = [0.0_f32, 0.0, 0.0, 1.0];
        let base_circles = vec![
            // Packed batch 0, instance 0:
            CircleInstance {
                center: [50.0, 50.0], radius: 5.0, fill_color: neutral,
                stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 0.5,
                stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
            // Packed batch 0, instance 1:
            CircleInstance {
                center: [80.0, 80.0], radius: 5.0, fill_color: neutral,
                stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 0.5,
                stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
            // Non-packed batch 1, instance 0:
            CircleInstance {
                center: [200.0, 200.0], radius: 5.0, fill_color: neutral,
                stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 0.5,
                stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
            },
        ];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: Some(vec![0, 1]),
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 2,
            },
        );

        let result = resolve_conditionals_with_packed(
            &panels, &conditionals, &selections, &base_circles, &[], &packed_meta,
        );

        // Non-packed circle (index 2) has data_idx=99 which IS selected.
        // If packed offset tracking is broken, the offset would be wrong and
        // this circle would either not be processed or the wrong instance modified.
        assert!(
            (result.circle_instances[2].opacity - 1.0).abs() < 0.01,
            "non-packed circle after packed batch must be selected (opacity=1.0), got {}",
            result.circle_instances[2].opacity
        );
    }

    // ── T3: non-packed-BEFORE-packed ordering regression ─────────────────────
    //
    // The loader builds the flat array packed-FIRST: all packed instances are
    // prepended by `unpack_binary_instances`, then the scene-graph walk appends
    // non-packed instances. When a non-packed batch appears BEFORE a same-kind
    // packed batch in scene order, the running `circle_offset` (= 0 at that
    // point in scene order) is wrong — the non-packed instance actually lives at
    // `total_packed` in the flat array. This test captures the correct behavior
    // after the fix: the non-packed instance at flat index 3 (after 3 packed
    // circles) must be dimmed by the conditional, and the packed region must not
    // be corrupted.
    #[test]
    fn nonpacked_before_packed_same_kind_dims_correct_instances() {
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

        // Panel with two batches in SCENE order:
        //   batch 0: non-packed, 1 circle (data_idx=50) — should be dimmed
        //   batch 1: packed,     3 circles               — should be untouched
        //
        // LOADER flat layout (packed-FIRST):
        //   circles[0..3] = packed batch 1 instances
        //   circles[3]    = non-packed batch 0 instance
        //
        // With the bug: circle_offset=0 when batch 0 is processed → writes into
        // circles[0], corrupting the packed region.
        // After fix: circle_offset starts at total_packed_circles=3, so batch 0
        // correctly addresses circles[3].
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![
                // Batch 0 (scene order 0): non-packed, 1 circle.
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![
                        SceneNode::Circle { cx: 100.0, cy: 100.0, r: 5.0, style: style.clone() },
                    ],
                    data_indices: Some(vec![50]),
                    tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal,
                    descriptions: None, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                },
                // Batch 1 (scene order 1): packed, 3 circles.
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![], // empty — packed
                    data_indices: None,
                    tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal,
                    descriptions: None, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                },
            ],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];

        // Select data index 50 — that is the non-packed circle.
        let conditionals = vec![ConditionalEncoding {
            selection_name: "sel".to_string(),
            channel: ChannelName::Opacity,
            if_selected: EncodingValue::Opacity { value: 1.0 },
            if_not: EncodingValue::Opacity { value: 0.15 },
        }];
        let mut selections = HashMap::new();
        selections.insert(
            "sel".to_string(),
            SelectionState::Point {
                indices: vec![50],
                field_values: Vec::new(),
            },
        );

        // Flat array — loader layout (packed-FIRST):
        //   [0] packed circle 0  (data_idx=0)
        //   [1] packed circle 1  (data_idx=1)
        //   [2] packed circle 2  (data_idx=2)
        //   [3] non-packed circle (data_idx=50, should be selected)
        let neutral = [0.0_f32, 0.0, 0.0, 1.0];
        let mk_circle = |cx: f32| CircleInstance {
            center: [cx, 50.0], radius: 5.0, fill_color: neutral,
            stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 0.5,
            stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
        };
        let base_circles = vec![
            mk_circle(10.0), // packed [0]
            mk_circle(20.0), // packed [1]
            mk_circle(30.0), // packed [2]
            mk_circle(100.0), // non-packed [3]
        ];

        // Packed meta: batch (0,1) = 3 circles starting at index 0.
        // batch (0,0) is non-packed so it has no entry.
        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 1u32),
            PackedBatchMeta {
                data_indices: Some(vec![0, 1, 2]),
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        let result = resolve_conditionals_with_packed(
            &panels, &conditionals, &selections, &base_circles, &[], &packed_meta,
        );

        // The packed region (circles[0..3]) should receive if_not = 0.15 from the
        // packed conditional arm (none of data_idx 0,1,2 matches the selected 50).
        // NOTE: this assertion is documentation-only and does NOT discriminate the
        // bug — with the old buggy offset (circle_offset=0) the packed conditional
        // arm still runs and overwrites circles[0] back to 0.15, so this loop
        // passes even without the fix. The load-bearing discriminator is the
        // circles[3] assertion below.
        for i in 0..3 {
            assert!(
                (result.circle_instances[i].opacity - 0.15).abs() < 0.01,
                "packed circle[{}] must be dimmed (opacity=0.15, not selected), got {}",
                i, result.circle_instances[i].opacity
            );
        }

        // DISCRIMINATOR: the non-packed circle at flat index 3 has data_idx=50
        // which IS selected. With the old bug (circle_offset=0) the non-packed
        // write hits circles[0] instead of circles[3], so circles[3] stays at
        // the neutral opacity 0.5. After the fix it must be 1.0 (if_selected).
        assert!(
            (result.circle_instances[3].opacity - 1.0).abs() < 0.01,
            "non-packed circle[3] (data_idx=50) must be selected (opacity=1.0), got {}",
            result.circle_instances[3].opacity
        );
    }

    // Crossfilter variant of the same bug: apply_crossfilter_to_panel must also
    // use the packed-FIRST basis for non-packed batch offsets.
    #[test]
    fn crossfilter_nonpacked_before_packed_dims_correct_instances() {
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

        // Same panel layout: non-packed batch BEFORE packed batch in scene order.
        // Flat layout (loader): [packed x3][non-packed x1]
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![
                // Batch 0 (scene order): non-packed, 1 circle at (400, 50).
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![
                        SceneNode::Circle { cx: 400.0, cy: 50.0, r: 5.0, style: style.clone() },
                    ],
                    data_indices: Some(vec![0]),
                    tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal,
                    descriptions: None, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                },
                // Batch 1 (scene order): packed, 3 circles.
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![],
                    data_indices: None,
                    tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal,
                    descriptions: None, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                },
            ],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];

        // Crossfilter brush covers x in [0, 100]: packed circles at x=10,20,30
        // are inside (kept full opacity), non-packed at x=400 is outside (dimmed).
        let sel = SelectionState::Interval {
            x_range: Some((0.0, 100.0)),
            y_range: None,
        };

        let mk_circle = |cx: f32| CircleInstance {
            center: [cx, 50.0], radius: 5.0, fill_color: [0.0; 4],
            stroke_color: [0.0; 4], stroke_width: 0.0, opacity: 1.0,
            stroke_opacity: 0.0, stroke_dash: 0.0, angle: 0.0,
        };
        // Flat layout (loader): [packed x3][non-packed x1]
        let mut circles = vec![
            mk_circle(10.0), // packed [0], inside brush
            mk_circle(20.0), // packed [1], inside brush
            mk_circle(30.0), // packed [2], inside brush
            mk_circle(400.0), // non-packed [3], outside brush
        ];
        let mut rects: Vec<RectInstance> = vec![];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 1u32),
            PackedBatchMeta {
                data_indices: None,
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        apply_crossfilter_to_panel(
            &panels, 0, &sel, 0.15, &mut circles, &mut rects, &packed_meta,
        );

        // Packed circles (inside brush) must keep full opacity.
        for i in 0..3 {
            assert!(
                (circles[i].opacity - 1.0).abs() < 0.01,
                "packed circle[{}] inside brush must stay at opacity=1.0, got {}",
                i, circles[i].opacity
            );
        }

        // Non-packed circle at [3] (x=400, outside brush) must be dimmed.
        assert!(
            (circles[3].opacity - 0.15).abs() < 0.01,
            "non-packed circle[3] outside brush must be dimmed (opacity=0.15), got {}",
            circles[3].opacity
        );
    }

    // ── Fix 3: packed-batch field-value point selection (legend toggle) ──────

    /// Build a packed tooltip string table (matches `parse_tooltip_json`'s
    /// layout): `[num_fields][names…][rows × values…]`, row-major.
    fn build_tooltip_bytes(field_names: &[&str], rows: &[Vec<&str>]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(field_names.len() as u32).to_le_bytes());
        for name in field_names {
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
        }
        for row in rows {
            for val in row {
                buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
                buf.extend_from_slice(val.as_bytes());
            }
        }
        buf
    }

    /// A legend toggle builds a `Point` selection with non-empty `field_values`
    /// and empty `indices`. On a packed batch this must match marks whose packed
    /// tooltip carries the selected category — not silently match nothing.
    #[test]
    fn packed_field_value_point_selection_matches_via_tooltip() {
        use ferrum_scene::{
            BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect,
        };

        // Single packed batch with three circles, categories a / b / a.
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![], // empty — packed batch
                data_indices: None,
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
            selection_name: "leg".to_string(),
            channel: ChannelName::Opacity,
            if_selected: EncodingValue::Opacity { value: 1.0 },
            if_not: EncodingValue::Opacity { value: 0.2 },
        }];
        // Colors kept distinct so a future colour conditional would still read
        // sensibly; this test exercises the opacity dim that legend toggle uses.
        let _ = (red, grey);

        // Legend toggle on category "a": Point selection, empty indices.
        let mut selections = HashMap::new();
        selections.insert(
            "leg".to_string(),
            SelectionState::Point {
                indices: Vec::new(),
                field_values: vec![(
                    "cat".to_string(),
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
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            })
            .collect();

        let tooltip_bytes = build_tooltip_bytes(
            &["cat"],
            &[vec!["a"], vec!["b"], vec!["a"]],
        );
        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: Some(vec![0, 1, 2]),
                tooltip_bytes: Some(tooltip_bytes),
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        let result = resolve_conditionals_with_packed(
            &panels, &conditionals, &selections, &base_circles, &[], &packed_meta,
        );

        // Marks 0 and 2 carry cat="a" → selected → opacity 1.0.
        // Mark 1 carries cat="b" → not selected → dimmed to 0.2.
        assert!(
            (result.circle_instances[0].opacity - 1.0).abs() < 0.01,
            "packed mark 0 (cat=a) must be selected, got {}",
            result.circle_instances[0].opacity
        );
        assert!(
            (result.circle_instances[1].opacity - 0.2).abs() < 0.01,
            "packed mark 1 (cat=b) must be dimmed, got {}",
            result.circle_instances[1].opacity
        );
        assert!(
            (result.circle_instances[2].opacity - 1.0).abs() < 0.01,
            "packed mark 2 (cat=a) must be selected, got {}",
            result.circle_instances[2].opacity
        );
    }

    // ── WASM-04: tooltip_matches ─────────────────────────────────────────────

    /// All field_values must match for `tooltip_matches` to return true.
    #[test]
    fn tooltip_matches_all_fields_match_returns_true() {
        let field_values = vec![
            ("cat".to_string(), FieldValue::String { value: "a".to_string() }),
            ("val".to_string(), FieldValue::Number { value: 42.0 }),
        ];
        let result = tooltip_matches(&field_values, |fname| match fname {
            "cat" => Some("a".to_string()),
            "val" => Some("42".to_string()),
            _ => None,
        });
        assert!(result, "all fields matching must return true");
    }

    /// If any field does not match, `tooltip_matches` returns false.
    #[test]
    fn tooltip_matches_one_mismatch_returns_false() {
        let field_values = vec![
            ("cat".to_string(), FieldValue::String { value: "a".to_string() }),
            ("val".to_string(), FieldValue::Number { value: 42.0 }),
        ];
        let result = tooltip_matches(&field_values, |fname| match fname {
            "cat" => Some("a".to_string()),
            "val" => Some("99".to_string()), // mismatch
            _ => None,
        });
        assert!(!result, "one mismatching field must return false");
    }

    /// If the lookup returns `None` for a field, it counts as a mismatch.
    #[test]
    fn tooltip_matches_missing_field_returns_false() {
        let field_values = vec![
            ("cat".to_string(), FieldValue::String { value: "a".to_string() }),
        ];
        let result = tooltip_matches(&field_values, |_fname| None);
        assert!(!result, "missing field (lookup returns None) must return false");
    }

    /// Empty `field_values` slice — vacuously true (all zero constraints pass).
    #[test]
    fn tooltip_matches_empty_field_values_returns_true() {
        let field_values: Vec<(String, FieldValue)> = vec![];
        let result = tooltip_matches(&field_values, |_| None);
        assert!(result, "empty field_values must return true (vacuous truth)");
    }

    /// Typed comparison: numeric tooltip string "42" matches `FieldValue::Number{42.0}`.
    #[test]
    fn tooltip_matches_numeric_field_value_typed_comparison() {
        let field_values = vec![
            ("x".to_string(), FieldValue::Number { value: 42.0 }),
        ];
        // The string "42" must match Number{42.0} via field_value_matches_tooltip.
        let result = tooltip_matches(&field_values, |_| Some("42".to_string()));
        assert!(result, "string '42' must match Number(42.0) via typed comparison");
    }

    /// Bool field value: "true" matches `FieldValue::Bool{true}`.
    #[test]
    fn tooltip_matches_bool_field_value() {
        let field_values = vec![
            ("flag".to_string(), FieldValue::Bool { value: true }),
        ];
        let result = tooltip_matches(&field_values, |_| Some("true".to_string()));
        assert!(result, "'true' string must match Bool(true)");

        let result_false = tooltip_matches(&field_values, |_| Some("false".to_string()));
        assert!(!result_false, "'false' string must not match Bool(true)");
    }
}
