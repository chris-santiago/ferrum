use ferrum_scene::{
    stroke_dash_index, ConditionalEncoding, EncodingValue, FieldValue, MarkBatch, SceneNode,
};
// `Panel` is named only by the wasm32 render entry and this module's host tests
// — every other signature here takes the panels inside an `InstanceLayout`.
#[cfg(any(target_arch = "wasm32", test))]
use ferrum_scene::Panel;

use crate::scene_load::{
    batch_instance_spans, color_to_linear, CircleInstance, DrawKind, InstanceLayout,
    PackedBatchMeta, RectInstance,
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
/// return the serialized selection state JSON, plus the restyled panels when
/// a line was redrawn ([`redraw_conditional_lines`]).
///
/// Called by both `handle_click` and `handle_drag` after they update the
/// selection state. Centralising this ensures that if `SceneData` gains a
/// field, only one site needs updating.
///
/// `panels` is the caller's BAKED panel geometry (`scene_load::bake_panels`),
/// not the raw `SceneGraph::panels` — `resolve_conditionals_with_packed`
/// reads non-packed mark node coordinates (`SceneNode::Circle`/`Rect`
/// centers) for containment, which must be at final on-screen position for a
/// ratio-fitted panel (D4a amendment addendum). `conditionals` is passed
/// separately (rather than the whole `SceneGraph`) so this function never
/// has access to the raw, un-baked panel list.
#[cfg(target_arch = "wasm32")]
pub(crate) fn apply_conditionals_and_render(
    panels: &[Panel],
    conditionals: &[ConditionalEncoding],
    loaded_data: &crate::scene_load::SceneData,
    loaded_buffers: &mut crate::render::GpuBuffers,
    interaction_state: &crate::selection_state::InteractionState,
    gpu: &crate::gpu::GpuContext,
    pipelines: &crate::pipelines::RenderPipelines,
) -> Result<(String, Option<Vec<Panel>>), wasm_bindgen::JsValue> {
    // Apply conditional encodings to produce dimmed/highlighted instance colors.
    let updates = resolve_conditionals_with_packed(
        InstanceLayout {
            panels,
            packed_batch_meta: &loaded_data.packed_batch_meta,
            draw_commands: &loaded_data.draw_commands,
        },
        conditionals,
        &interaction_state.selections,
        &loaded_data.circle_instances,
        &loaded_data.rect_instances,
    );

    // Re-upload the instance buffers (circle and rect styles), and the mark
    // mesh only when a size conditional restyles a line. Static mesh,
    // annotation mesh, images, and uniforms are unaffected by selection
    // changes and are not re-uploaded.
    loaded_buffers.update_instances(gpu, &updates.circle_instances, &updates.rect_instances);
    let restyled = redraw_conditional_lines(
        panels,
        conditionals,
        &interaction_state.selections,
        loaded_data,
        loaded_buffers,
        gpu,
    )?;
    crate::render::render_frame(gpu, pipelines, loaded_buffers, loaded_data.background)
        .map_err(wasm_bindgen::JsValue::from)?;

    // Serialize current selection state for Python sync.
    let state_json = interaction_state.to_json();
    Ok((state_json, restyled))
}

pub struct ConditionalUpdates {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
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
/// the instances belonging to `target_panel` are touched. Instance addresses
/// come from the shared `scene_load::batch_instance_spans` walk, the one
/// definition of the packed/non-packed flat-array layout, which
/// `resolve_conditionals_with_packed` reads too.
///
/// Consumed by the wasm32 `WasmRenderer`; gated to wasm32 + test so the
/// non-test host build does not flag it as dead code (host coverage is via the
/// `#[cfg(test)]` unit test below).
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn apply_crossfilter_to_panel(
    layout: InstanceLayout<'_>,
    target_panel: usize,
    selection: &SelectionState,
    dimmed_opacity: f64,
    circles: &mut [CircleInstance],
    rects: &mut [RectInstance],
) {
    // A filtered-in mark keeps its opacity; a filtered-out mark is dimmed.
    // The targeted channel is implied by the value variant (`Opacity`); the
    // apply path dispatches on the value alone.
    let if_selected = EncodingValue::Opacity { value: 1.0 };
    let if_not = EncodingValue::Opacity {
        value: dimmed_opacity,
    };

    if matches!(selection, SelectionState::Empty) {
        return;
    }

    for span in batch_instance_spans(layout) {
        if span.panel_idx != target_panel {
            continue;
        }
        match span.packed {
            Some(meta) => {
                apply_conditional_to_packed(&if_selected, &if_not, selection, meta, circles, rects);
            }
            None => {
                if let Some(indices) = &span.batch.data_indices {
                    let mut bufs = InstanceBuffers {
                        circles,
                        circle_offset: span.circles.start,
                        rects,
                        rect_offset: span.rects.start,
                    };
                    apply_conditional_to_batch(
                        &if_selected,
                        &if_not,
                        selection,
                        indices,
                        span.batch,
                        &mut bufs,
                    );
                }
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

/// Apply every conditional encoding to a scene's instance buffers.
///
/// `layout` locates each mark batch's instances, and must come from the same
/// loader pass that produced `base_circles`/`base_rects` (see
/// [`InstanceLayout`]). The returned buffers are copies of those, with the
/// selected/unselected value written onto the instances each conditional claims.
pub fn resolve_conditionals_with_packed(
    layout: InstanceLayout<'_>,
    conditionals: &[ConditionalEncoding],
    selections: &HashMap<String, SelectionState>,
    base_circles: &[CircleInstance],
    base_rects: &[RectInstance],
) -> ConditionalUpdates {
    let mut circles = base_circles.to_vec();
    let mut rects = base_rects.to_vec();

    // Instance addresses come from the shared `batch_instance_spans` walk:
    // packed batches sit at their absolute sidecar `instance_start`, non-packed
    // batches at the offsets the loader recorded on their own draw commands.
    for span in batch_instance_spans(layout) {
        for (index, cond) in conditionals.iter().enumerate() {
            // A `size` conditional reaches only the batches whose layer
            // declared it, when the batch names its own.
            let out_of_scope = cond.sets_size()
                && span.batch.size_conditionals.as_ref().is_some_and(|own| !own.contains(&index));
            if out_of_scope {
                continue;
            }
            let Some(sel) = selections.get(&cond.selection_name) else {
                continue;
            };
            if matches!(sel, SelectionState::Empty) {
                continue;
            }
            match span.packed {
                // Packed batch: `batch.nodes` is empty, so the sidecar
                // metadata is the only way to locate and process instances.
                Some(meta) => apply_conditional_to_packed(
                    &cond.if_selected,
                    &cond.if_not,
                    sel,
                    meta,
                    &mut circles,
                    &mut rects,
                ),
                None => {
                    if let Some(indices) = &span.batch.data_indices {
                        let mut bufs = InstanceBuffers {
                            circles: &mut circles,
                            circle_offset: span.circles.start,
                            rects: &mut rects,
                            rect_offset: span.rects.start,
                        };
                        apply_conditional_to_batch(
                            &cond.if_selected,
                            &cond.if_not,
                            sel,
                            indices,
                            span.batch,
                            &mut bufs,
                        );
                    }
                }
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
        // Spatial containment reads the mark's center.
        let center = match node {
            SceneNode::Circle { cx, cy, .. } => Some((*cx, *cy)),
            SceneNode::Rect { x, y, w, h, .. } => Some((*x + *w / 2.0, *y + *h / 2.0)),
            _ => None,
        };
        let selected = mark_selected(sel, batch, node_idx, data_idx, center);

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

/// Whether the mark at `node_idx` of a non-packed `batch` is in `sel`, the
/// unpacked membership rules [`packed_instance_selected`] mirrors:
/// - `Interval` → spatial containment of `pos` (`None` is never inside);
/// - `Point` with non-empty `field_values` → the node's tooltip carries every
///   selected field value, which is what links panels whose datasets (and so
///   data indices) differ;
/// - `Point` with `indices` only → `data_idx` membership.
///
/// A circle or rect asks once per node (its center, its row); a line asks once
/// per width vertex (that vertex's point and source row), so the field-value
/// rule reads the series' own tooltip for every vertex.
fn mark_selected(
    sel: &SelectionState,
    batch: &MarkBatch,
    node_idx: usize,
    data_idx: Option<usize>,
    pos: Option<(f64, f64)>,
) -> bool {
    match sel {
        SelectionState::Interval { .. } => pos.is_some_and(|(x, y)| sel.contains_point(x, y)),
        SelectionState::Point { field_values, .. } if !field_values.is_empty() => batch
            .tooltips
            .as_ref()
            .and_then(|tips| tips.get(node_idx))
            .is_some_and(|tip| {
                tooltip_matches(field_values, |fname| {
                    tip.fields
                        .iter()
                        .find(|f| f.name == fname)
                        .map(|f| f.value.clone())
                })
            }),
        _ => data_idx.is_some_and(|di| sel.contains(di)),
    }
}

/// `panels` (BAKED, [`crate::scene_load::bake_panels`]) with every line that
/// carries `vertex_rows` redrawn under the active selections: each width
/// vertex takes the dot diameter ([`ferrum_scene::dot_diameter`]) of the area
/// its own row's branch of each `size` conditional the batch names in
/// `size_conditionals` (its own layer's) picks, a later conditional
/// outranking an earlier one as [`resolve_conditionals_with_packed`] layers
/// them, baked by the panel's layout scale as every stroke width is
/// ([`crate::scene_load::baked_stroke_width`]). A vertex a break cut between
/// two rows takes the two rows' widths interpolated at the cut
/// ([`ferrum_scene::width_at`]). A vertex no active `size` conditional
/// reaches keeps its loaded width, and a line none reaches keeps its loaded
/// `widths`, so with every selection empty the panels tessellate exactly as
/// loaded. `None` when no line carries `vertex_rows`: nothing to redraw. The
/// loader has already refused rows that do not align with their nodes
/// ([`ferrum_scene::validate_vertex_rows`]).
///
/// Every panel is cloned, not only those holding a restyled line: the mark
/// mesh is one buffer whose per-panel index ranges [`crate::scene_load::mark_mesh`]
/// rebuilds from all panels at once, and the result is kept whole as the
/// panel slice hit-testing reads.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn conditional_line_panels(
    panels: &[Panel],
    conditionals: &[ConditionalEncoding],
    selections: &HashMap<String, SelectionState>,
) -> Option<Vec<Panel>> {
    if !panels.iter().flat_map(|p| &p.marks).any(|b| b.vertex_rows.is_some()) {
        return None;
    }
    let mut panels = panels.to_vec();
    for panel in &mut panels {
        let ls = panel.layout_scale;
        for batch in &mut panel.marks {
            let Some(rows) = batch.vertex_rows.as_ref() else { continue };
            // Only the conditionals this line's own layer declared, in scene
            // order, each with a non-empty selection.
            let active: Vec<(&ConditionalEncoding, &SelectionState)> = batch
                .size_conditionals
                .iter()
                .flatten()
                .filter_map(|&i| {
                    let cond = conditionals.get(i)?;
                    Some((cond, selections.get(&cond.selection_name)?))
                })
                .filter(|(_, sel)| !matches!(sel, SelectionState::Empty))
                .collect();
            let widths: Vec<Option<Vec<f64>>> = rows
                .iter()
                .enumerate()
                .map(|(node_idx, rows)| vertex_widths(batch, node_idx, rows, &active, &ls))
                .collect();
            for (node, widths) in batch.nodes.iter_mut().zip(widths) {
                if let (SceneNode::Polyline { widths: slot, .. } | SceneNode::Path { widths: slot, .. }, Some(w)) =
                    (node, widths)
                {
                    *slot = Some(w);
                }
            }
        }
    }
    Some(panels)
}

/// One line node's widths under the `active` conditionals (see
/// [`conditional_line_panels`]), or `None` to keep it as loaded: no active
/// `size` conditional reaches it.
#[cfg(any(target_arch = "wasm32", test))]
fn vertex_widths(
    batch: &MarkBatch,
    node_idx: usize,
    rows: &[ferrum_scene::VertexRow],
    active: &[(&ConditionalEncoding, &SelectionState)],
    ls: &ferrum_scene::LayoutScale,
) -> Option<Vec<f64>> {
    let (points, base, loaded) = match batch.nodes.get(node_idx)? {
        SceneNode::Polyline { points, style, widths } => (points.clone(), style.width, widths),
        SceneNode::Path { commands, style, widths, .. } => {
            (ferrum_scene::command_endpoints(commands), style.stroke_width, widths)
        }
        _ => return None,
    };
    // The branch width (baked px) a row draws at `point`, or `None` when no
    // active conditional's branch for that row is a size.
    let row_width = |row: usize, point: (f64, f64)| {
        let area = active.iter().fold(None, |area, (cond, sel)| {
            let selected = mark_selected(sel, batch, node_idx, Some(row), Some(point));
            match if selected { &cond.if_selected } else { &cond.if_not } {
                EncodingValue::Size { value } => Some(*value),
                _ => area,
            }
        });
        area.map(|area| crate::scene_load::baked_stroke_width(ls, ferrum_scene::dot_diameter(area)))
    };
    let mut reached = false;
    let widths = points
        .iter()
        .zip(rows)
        .enumerate()
        .map(|(i, (&point, row))| {
            let base = loaded.as_ref().and_then(|w| w.get(i).copied()).unwrap_or(base);
            let width = resolve_vertex_width(row, base, &|r| row_width(r, point));
            reached |= width.is_some();
            width.unwrap_or(base)
        })
        .collect();
    reached.then_some(widths)
}

/// The width of one vertex at `row`: its row's branch width, or at a cut the
/// two ends' widths interpolated at `t`, an end no conditional reaches
/// drawing `base`. `None` when no conditional reaches any row of it.
#[cfg(any(target_arch = "wasm32", test))]
fn resolve_vertex_width(
    row: &ferrum_scene::VertexRow,
    base: f64,
    row_width: &dyn Fn(usize) -> Option<f64>,
) -> Option<f64> {
    match row {
        ferrum_scene::VertexRow::Row(r) => row_width(*r),
        ferrum_scene::VertexRow::Cut { from, to, t } => {
            let (from, to) = (resolve_vertex_width(from, base, row_width), resolve_vertex_width(to, base, row_width));
            if from.is_none() && to.is_none() {
                return None;
            }
            ferrum_scene::width_at(&[from.unwrap_or(base), to.unwrap_or(base)], 0, *t)
        }
    }
}

/// Re-tessellate the mark mesh when a selection restyles a line's width (see
/// [`conditional_line_panels`]) and upload it, returning the restyled BAKED
/// panels for hit-testing (a restyled line's hover tolerance is its restyled
/// width); `None` for a scene with no such line. A restyled width the shared
/// taper validation refuses is a typed error.
#[cfg(target_arch = "wasm32")]
pub(crate) fn redraw_conditional_lines(
    panels: &[Panel],
    conditionals: &[ConditionalEncoding],
    selections: &HashMap<String, SelectionState>,
    loaded_data: &crate::scene_load::SceneData,
    loaded_buffers: &mut crate::render::GpuBuffers,
    gpu: &crate::gpu::GpuContext,
) -> Result<Option<Vec<Panel>>, wasm_bindgen::JsValue> {
    let Some(lines) = conditional_line_panels(panels, conditionals, selections) else {
        return Ok(None);
    };
    let extent = [0.0, 0.0, loaded_data.width, loaded_data.height];
    let (mesh, mesh_panels) = crate::scene_load::mark_mesh(&lines, extent)
        .map_err(|e| wasm_bindgen::JsValue::from(crate::error::WasmRenderError::TaperedStroke(e)))?;
    loaded_buffers.update_mark_mesh(gpu, &mesh, mesh_panels);
    Ok(Some(lines))
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
            *view.stroke_dash = stroke_dash_index(Some(pattern.as_slice())) as f32;
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
    use ferrum_scene::{ChannelName, Color, FieldValue, LayoutScale};

    use crate::scene_load::{load_scene, DrawCommand, MarkDrawTarget};

    /// One mark draw command in the shape `SceneCollector::emit` records it:
    /// `count` instances of `kind` at `start`, owned by `(panel_idx,
    /// batch_idx)`.
    ///
    /// Spelled out by hand only for the packed fixtures below, whose absolute
    /// sidecar offsets are asserted in the test's own layout comment; building
    /// the binary sidecar instead would add a fourth copy of the packer's byte
    /// format to this crate. Non-packed fixtures use [`loader_draw_commands`].
    fn mark_cmd(
        kind: DrawKind,
        start: usize,
        count: usize,
        panel_idx: usize,
        batch_idx: usize,
    ) -> DrawCommand {
        DrawCommand::new(
            kind,
            start,
            count,
            Some(MarkDrawTarget {
                panel_id: panel_idx,
                batch_idx,
                y_slot: 0,
                plot_area: Some([0.0, 0.0, 500.0, 500.0]),
                additive: false,
            }),
        )
    }

    /// The draw commands the REAL loader records for these panels.
    ///
    /// `batch_instance_spans` reads a non-packed batch's instance offsets off
    /// these commands (GH #105), so a hand-assembled list would pin the very
    /// assumption the fix removed. Running `load_scene` instead means the
    /// offsets under test are the ones production assigns.
    fn loader_draw_commands(panels: &[Panel]) -> Vec<DrawCommand> {
        // The draw commands do not depend on conditional scopes, which index
        // conditionals this scene does not carry.
        let mut panels = panels.to_vec();
        for batch in panels.iter_mut().flat_map(|p| &mut p.marks) {
            batch.size_conditionals = None;
        }
        let scene = ferrum_scene::SceneGraph {
            width: 500.0,
            height: 500.0,
            background: None,
            title: vec![],
            panels,
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: ferrum_scene::InteractionConfig::default(),
            chart_description: None,
        };
        load_scene(&scene).unwrap().draw_commands
    }

    /// [`resolve_conditionals_with_packed`] for a scene with no packed batches:
    /// an empty sidecar map, and the loader's own draw commands.
    fn resolve_conditionals(
        panels: &[Panel],
        conditionals: &[ConditionalEncoding],
        selections: &HashMap<String, SelectionState>,
        base_circles: &[CircleInstance],
        base_rects: &[RectInstance],
    ) -> ConditionalUpdates {
        let draw_commands = loader_draw_commands(panels);
        resolve_conditionals_with_packed(
            InstanceLayout {
                panels,
                packed_batch_meta: &HashMap::new(),
                draw_commands: &draw_commands,
            },
            conditionals,
            selections,
            base_circles,
            base_rects,
        )
    }

    // ── D6 crossfilter: apply_crossfilter_to_panel dims only the target panel ─

    #[test]
    fn crossfilter_dims_only_target_panel_outside_interval() {
        use ferrum_scene::{
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                nodes: vec![SceneNode::Circle {
                    cx,
                    cy: 50.0,
                    r: 5.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        };
        // Panel 0 (source) mark at x=10 (inside the brush); panel 1 (target)
        // mark at x=400 (outside the re-projected interval).
        let panels = vec![mk_panel(10.0), mk_panel(400.0)];

        let mut circles = vec![
            CircleInstance {
                center: [10.0, 50.0],
                radius: 5.0,
                fill_color: [0.0; 4],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [400.0, 50.0],
                radius: 5.0,
                fill_color: [0.0; 4],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];
        let mut rects: Vec<RectInstance> = vec![];

        // Brush covers x in [0, 100] on the target panel's pixel space.
        let sel = SelectionState::Interval {
            x_range: Some((0.0, 100.0)),
            y_range: None,
        };

        apply_crossfilter_to_panel(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &HashMap::new(),
                draw_commands: &loader_draw_commands(&panels),
            },
            1,
            &sel,
            0.15,
            &mut circles,
            &mut rects,
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                nodes: vec![SceneNode::Circle {
                    cx: 50.0,
                    cy: 50.0,
                    r: 5.0,
                    style,
                }],
                data_indices: Some(vec![0]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];
        let mut circles = vec![CircleInstance {
            center: [50.0, 50.0],
            radius: 5.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];
        let mut rects: Vec<RectInstance> = vec![];
        // Brush [0, 100] contains the mark at x=50.
        let sel = SelectionState::Interval {
            x_range: Some((0.0, 100.0)),
            y_range: None,
        };
        apply_crossfilter_to_panel(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &HashMap::new(),
                draw_commands: &loader_draw_commands(&panels),
            },
            0,
            &sel,
            0.15,
            &mut circles,
            &mut rects,
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
        apply_value_to_circle(&mut inst, &EncodingValue::Color { value: red });
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
        apply_value_to_rect(&mut inst, &EncodingValue::Opacity { value: 0.3 });
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
        apply_value_to_rect(&mut inst, &EncodingValue::Size { value: 20.0 });
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
            TooltipContent, TooltipField,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                        cx: 50.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 150.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 250.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
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
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let grey = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };

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
                    FieldValue::String {
                        value: "a".to_string(),
                    },
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                        cx: 50.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 150.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                ],
                data_indices: Some(vec![0, 1]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let grey = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };

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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                        cx: 50.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 200.0,
                        cy: 200.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                ],
                data_indices: Some(vec![0, 1]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
                    value: Color {
                        r: 255,
                        g: 0,
                        b: 0,
                        a: 255,
                    },
                },
                if_not: EncodingValue::Color {
                    value: Color {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 255,
                    },
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
            CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                        cx: 20.0,
                        cy: 30.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 100.0,
                        cy: 100.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 30.0,
                        cy: 40.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                ],
                data_indices: Some(vec![0, 1, 2]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let grey = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };

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
    fn bug_hunt_field_value_matches_infinity() {
        // BUG: (inf - inf).abs() is NaN, not < 1e-10, so equal infinities don't match
        // "Infinity" parses to f64::INFINITY.
        assert!(
            field_value_matches_tooltip(
                "inf",
                &FieldValue::Number {
                    value: f64::INFINITY
                }
            ),
            "inf tooltip must match INFINITY field value"
        );
        assert!(
            field_value_matches_tooltip(
                "-inf",
                &FieldValue::Number {
                    value: f64::NEG_INFINITY
                }
            ),
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
            field_value_matches_tooltip(
                "",
                &FieldValue::String {
                    value: String::new()
                }
            ),
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
            field_value_matches_tooltip(
                "null",
                &FieldValue::String {
                    value: "null".to_string()
                }
            ),
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
            field_value_matches_tooltip(
                emoji,
                &FieldValue::String {
                    value: emoji.to_string()
                }
            ),
            "unicode emoji must match identical string field value"
        );
        assert!(
            !field_value_matches_tooltip(
                emoji,
                &FieldValue::String {
                    value: "smiley".to_string()
                }
            ),
            "emoji must not match non-emoji string"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_numeric_string_as_string() {
        // A tooltip value "42" compared against FieldValue::String{"42"} must
        // match (direct string equality).
        assert!(
            field_value_matches_tooltip(
                "42",
                &FieldValue::String {
                    value: "42".to_string()
                }
            ),
            "'42' tooltip must match String('42') field value"
        );
    }

    #[test]
    fn bug_hunt_field_value_matches_very_long_string() {
        let long = "x".repeat(100_000);
        assert!(
            field_value_matches_tooltip(
                &long,
                &FieldValue::String {
                    value: long.clone()
                }
            ),
            "very long strings must still match"
        );
    }

    // ── bug_hunt: resolve_conditionals edge cases ────────────────────────

    #[test]
    fn bug_hunt_resolve_conditionals_nonexistent_selection_name() {
        // A conditional referencing a selection name that doesn't exist in
        // the selections map should be silently skipped (no panic).
        use ferrum_scene::{
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
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
                nodes: vec![SceneNode::Circle {
                    cx: 50.0,
                    cy: 50.0,
                    r: 5.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let conditionals = vec![ConditionalEncoding {
            selection_name: "does_not_exist".to_string(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Color {
                value: Color {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255,
                },
            },
            if_not: EncodingValue::Color {
                value: Color {
                    r: 128,
                    g: 128,
                    b: 128,
                    a: 255,
                },
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
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
                nodes: vec![SceneNode::Circle {
                    cx: 50.0,
                    cy: 50.0,
                    r: 5.0,
                    style: style.clone(),
                }],
                data_indices: None, // <-- no data indices
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
            &EncodingValue::Size {
                value: std::f64::consts::PI * 100.0,
            },
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
            &EncodingValue::Field {
                name: "category".to_string(),
            },
        );
        assert_eq!(
            inst.fill_color, original_fill,
            "Field value must not change fill"
        );
        assert!(
            (inst.opacity - original_opacity).abs() < 1e-10,
            "Field value must not change opacity"
        );
        assert!(
            (inst.radius - original_radius).abs() < 1e-10,
            "Field value must not change radius"
        );
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
        apply_value_to_circle(&mut inst, &EncodingValue::StrokeWidth { value: 3.5 });
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
            &EncodingValue::StrokeDash {
                value: vec![6.0, 3.0],
            },
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
        apply_value_to_rect(&mut inst, &EncodingValue::StrokeWidth { value: 2.0 });
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
            &EncodingValue::StrokeDash {
                value: vec![2.0, 3.0],
            },
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
            EncodingValue::Color {
                value: Color {
                    r: 200,
                    g: 100,
                    b: 50,
                    a: 255,
                },
            },
            EncodingValue::Opacity { value: 0.4 },
            EncodingValue::StrokeWidth { value: 2.5 },
            EncodingValue::StrokeDash {
                value: vec![6.0, 3.0],
            },
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
            assert!(
                (c.opacity - r.opacity).abs() < 1e-6,
                "opacity parity for {value:?}"
            );
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
            assert!(
                (c.angle - r.angle).abs() < 1e-6,
                "angle parity for {value:?}"
            );
        }
    }

    /// Field names where `before` and `after` differ, in struct declaration
    /// order. Every untouched field is copied verbatim rather than
    /// recomputed, so a bit-exact comparison needs no tolerance. Both sides
    /// are destructured with every field named (no `..`), so a field added
    /// to `CircleInstance` is a compile error here until this diff names it.
    fn changed_circle_fields(before: &CircleInstance, after: &CircleInstance) -> Vec<&'static str> {
        let CircleInstance {
            center: before_center,
            radius: before_radius,
            fill_color: before_fill_color,
            stroke_color: before_stroke_color,
            stroke_width: before_stroke_width,
            opacity: before_opacity,
            stroke_opacity: before_stroke_opacity,
            stroke_dash: before_stroke_dash,
            angle: before_angle,
        } = *before;
        let CircleInstance {
            center: after_center,
            radius: after_radius,
            fill_color: after_fill_color,
            stroke_color: after_stroke_color,
            stroke_width: after_stroke_width,
            opacity: after_opacity,
            stroke_opacity: after_stroke_opacity,
            stroke_dash: after_stroke_dash,
            angle: after_angle,
        } = *after;

        let mut changed = Vec::new();
        if before_center != after_center {
            changed.push("center");
        }
        if before_radius != after_radius {
            changed.push("radius");
        }
        if before_fill_color != after_fill_color {
            changed.push("fill_color");
        }
        if before_stroke_color != after_stroke_color {
            changed.push("stroke_color");
        }
        if before_stroke_width != after_stroke_width {
            changed.push("stroke_width");
        }
        if before_opacity != after_opacity {
            changed.push("opacity");
        }
        if before_stroke_opacity != after_stroke_opacity {
            changed.push("stroke_opacity");
        }
        if before_stroke_dash != after_stroke_dash {
            changed.push("stroke_dash");
        }
        if before_angle != after_angle {
            changed.push("angle");
        }
        changed
    }

    /// The rect sibling of [`changed_circle_fields`], same field-order and
    /// exhaustive-destructure convention.
    fn changed_rect_fields(before: &RectInstance, after: &RectInstance) -> Vec<&'static str> {
        let RectInstance {
            position: before_position,
            size: before_size,
            corner_radius: before_corner_radius,
            fill_color: before_fill_color,
            stroke_color: before_stroke_color,
            stroke_width: before_stroke_width,
            opacity: before_opacity,
            stroke_opacity: before_stroke_opacity,
            stroke_dash: before_stroke_dash,
            angle: before_angle,
        } = *before;
        let RectInstance {
            position: after_position,
            size: after_size,
            corner_radius: after_corner_radius,
            fill_color: after_fill_color,
            stroke_color: after_stroke_color,
            stroke_width: after_stroke_width,
            opacity: after_opacity,
            stroke_opacity: after_stroke_opacity,
            stroke_dash: after_stroke_dash,
            angle: after_angle,
        } = *after;

        let mut changed = Vec::new();
        if before_position != after_position {
            changed.push("position");
        }
        if before_size != after_size {
            changed.push("size");
        }
        if before_corner_radius != after_corner_radius {
            changed.push("corner_radius");
        }
        if before_fill_color != after_fill_color {
            changed.push("fill_color");
        }
        if before_stroke_color != after_stroke_color {
            changed.push("stroke_color");
        }
        if before_stroke_width != after_stroke_width {
            changed.push("stroke_width");
        }
        if before_opacity != after_opacity {
            changed.push("opacity");
        }
        if before_stroke_opacity != after_stroke_opacity {
            changed.push("stroke_opacity");
        }
        if before_stroke_dash != after_stroke_dash {
            changed.push("stroke_dash");
        }
        if before_angle != after_angle {
            changed.push("angle");
        }
        changed
    }

    /// The value-to-field table: every `EncodingValue` variant, applied to a
    /// default circle and rect instance, changes exactly the field set the
    /// pixel-harness oracle predicts. The `match` choosing each variant's
    /// expected fields carries no wildcard arm, so a variant added to
    /// `EncodingValue` fails to compile here rather than reaching the
    /// shaders with no mapped field — the same guard the codebase already
    /// uses for `MarkBatchKind::ALL`.
    #[test]
    fn value_to_field_table_covers_every_encoding_value_variant() {
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

        let samples = [
            EncodingValue::Color {
                value: Color {
                    r: 200,
                    g: 100,
                    b: 50,
                    a: 255,
                },
            },
            EncodingValue::Opacity { value: 0.42 },
            EncodingValue::Size { value: 40.0 },
            EncodingValue::StrokeWidth { value: 3.0 },
            EncodingValue::StrokeDash {
                value: vec![6.0, 3.0],
            },
            EncodingValue::StrokeOpacity { value: 0.33 },
            EncodingValue::FillOpacity { value: 0.66 },
            EncodingValue::Angle { value: 30.0 },
            EncodingValue::Field {
                name: "x".to_string(),
            },
        ];

        for value in &samples {
            let (expected_circle, expected_rect): (&[&str], &[&str]) = match value {
                EncodingValue::Color { .. } => (&["fill_color"], &["fill_color"]),
                EncodingValue::Opacity { .. } => (&["opacity"], &["opacity"]),
                EncodingValue::Size { .. } => (&["radius"], &["size"]),
                EncodingValue::StrokeWidth { .. } => (&["stroke_width"], &["stroke_width"]),
                EncodingValue::StrokeDash { .. } => (&["stroke_dash"], &["stroke_dash"]),
                EncodingValue::StrokeOpacity { .. } => (&["stroke_opacity"], &["stroke_opacity"]),
                EncodingValue::FillOpacity { .. } => (&["fill_color"], &["fill_color"]),
                EncodingValue::Angle { .. } => (&["angle"], &["angle"]),
                // Field carries no rendered field: the vacuity control that
                // every other row's non-empty set is compared against.
                EncodingValue::Field { .. } => (&[], &[]),
            };

            let mut circle = circle_base;
            apply_value_to_circle(&mut circle, value);
            assert_eq!(
                changed_circle_fields(&circle_base, &circle),
                expected_circle,
                "circle fields changed by {value:?}"
            );

            let mut rect = rect_base;
            apply_value_to_rect(&mut rect, value);
            assert_eq!(
                changed_rect_fields(&rect_base, &rect),
                expected_rect,
                "rect fields changed by {value:?}"
            );
        }
    }

    #[test]
    fn bug_hunt_interval_selection_rect_center_containment() {
        // Rect mark containment uses center (x + w/2, y + h/2).
        // Verify that a rect whose corner is outside but center is inside is selected.
        use ferrum_scene::{
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                nodes: vec![SceneNode::Rect {
                    x: 90.0,
                    y: 90.0,
                    w: 20.0,
                    h: 20.0,
                    corner_radius: 0.0,
                    style: style.clone(),
                }],
                data_indices: Some(vec![0]),
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
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
                        cx: 50.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                    SceneNode::Circle {
                        cx: 150.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    },
                ],
                data_indices: Some(vec![0, 1]),
                tooltips: None, // <-- no tooltips
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
                field_values: vec![(
                    "group".to_string(),
                    FieldValue::String {
                        value: "a".to_string(),
                    },
                )],
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
        use ferrum_scene::{CoordKind, MarkBatchKind, Panel, Rect};

        // Panel with one batch that has empty nodes (packed batch).
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                nodes: vec![], // empty — packed batch
                data_indices: None,
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let grey = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };

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

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                keys: None,
                data_indices: None,
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        // The sole batch is packed: its instances sit at the absolute sidecar
        // range, which is what the loader stamps on its one draw command.
        let result = resolve_conditionals_with_packed(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &packed_meta,
                draw_commands: &[mark_cmd(DrawKind::Circle, 0, 3, 0, 0)],
            },
            &conditionals,
            &selections,
            &base_circles,
            &[],
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
        use ferrum_scene::{CoordKind, MarkBatchKind, Panel, Rect};

        // Panel with one batch that has empty nodes (packed batch).
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                kind: MarkBatchKind::Bar,
                nodes: vec![], // empty — packed batch
                data_indices: None,
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
                position: [15.0, 15.0],
                size: [20.0, 20.0],
                corner_radius: 0.0,
                fill_color: [0.0; 4],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            RectInstance {
                position: [100.0, 100.0],
                size: [30.0, 30.0],
                corner_radius: 0.0,
                fill_color: [0.0; 4],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                keys: None,
                data_indices: None,
                tooltip_bytes: None,
                kind: DrawKind::Rect,
                instance_start: 0,
                instance_count: 2,
            },
        );

        // The sole batch is packed: its instances sit at the absolute sidecar
        // range, which is what the loader stamps on its one draw command.
        let result = resolve_conditionals_with_packed(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &packed_meta,
                draw_commands: &[mark_cmd(DrawKind::Rect, 0, 2, 0, 0)],
            },
            &conditionals,
            &selections,
            &[],
            &base_rects,
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![
                // Batch 0: packed (empty nodes).
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![],
                    ..MarkBatch::default()
                },
                // Batch 1: non-packed (1 circle in nodes).
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![SceneNode::Circle {
                        cx: 200.0,
                        cy: 200.0,
                        r: 5.0,
                        style: style.clone(),
                    }],
                    data_indices: Some(vec![99]),
                    ..MarkBatch::default()
                },
            ],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
            // Packed batch 0, instance 1:
            CircleInstance {
                center: [80.0, 80.0],
                radius: 5.0,
                fill_color: neutral,
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 0.5,
                stroke_opacity: 0.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            // Non-packed batch 1, instance 0:
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

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                keys: None,
                data_indices: Some(vec![0, 1]),
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 2,
            },
        );

        // Loader layout: the 2 packed circles are prepended, so batch 1's own
        // (non-packed) instance is appended at flat index 2 — the offset the
        // loader stamps on batch 1's draw command, and the one the span walk
        // reads back off it (GH #105).
        let result = resolve_conditionals_with_packed(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &packed_meta,
                draw_commands: &[
                    mark_cmd(DrawKind::Circle, 0, 2, 0, 0),
                    mark_cmd(DrawKind::Circle, 2, 1, 0, 1),
                ],
            },
            &conditionals,
            &selections,
            &base_circles,
            &[],
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![
                // Batch 0 (scene order 0): non-packed, 1 circle.
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![SceneNode::Circle {
                        cx: 100.0,
                        cy: 100.0,
                        r: 5.0,
                        style: style.clone(),
                    }],
                    data_indices: Some(vec![50]),
                    ..MarkBatch::default()
                },
                // Batch 1 (scene order 1): packed, 3 circles.
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![], // empty — packed
                    data_indices: None,
                    ..MarkBatch::default()
                },
            ],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
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
            center: [cx, 50.0],
            radius: 5.0,
            fill_color: neutral,
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 0.5,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let base_circles = vec![
            mk_circle(10.0),  // packed [0]
            mk_circle(20.0),  // packed [1]
            mk_circle(30.0),  // packed [2]
            mk_circle(100.0), // non-packed [3]
        ];

        // Packed meta: batch (0,1) = 3 circles starting at index 0.
        // batch (0,0) is non-packed so it has no entry.
        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 1u32),
            PackedBatchMeta {
                keys: None,
                data_indices: Some(vec![0, 1, 2]),
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        // Loader layout, as documented above: packed batch 1 at [0..3], then
        // non-packed batch 0 appended at [3]. Those are the offsets the loader
        // stamps on each batch's own draw command.
        let result = resolve_conditionals_with_packed(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &packed_meta,
                draw_commands: &[
                    mark_cmd(DrawKind::Circle, 0, 3, 0, 1),
                    mark_cmd(DrawKind::Circle, 3, 1, 0, 0),
                ],
            },
            &conditionals,
            &selections,
            &base_circles,
            &[],
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
                i,
                result.circle_instances[i].opacity
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
            CoordKind, FillStroke, MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
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
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![
                // Batch 0 (scene order): non-packed, 1 circle at (400, 50).
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![SceneNode::Circle {
                        cx: 400.0,
                        cy: 50.0,
                        r: 5.0,
                        style: style.clone(),
                    }],
                    data_indices: Some(vec![0]),
                    ..MarkBatch::default()
                },
                // Batch 1 (scene order): packed, 3 circles.
                MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![],
                    ..MarkBatch::default()
                },
            ],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        // Crossfilter brush covers x in [0, 100]: packed circles at x=10,20,30
        // are inside (kept full opacity), non-packed at x=400 is outside (dimmed).
        let sel = SelectionState::Interval {
            x_range: Some((0.0, 100.0)),
            y_range: None,
        };

        let mk_circle = |cx: f32| CircleInstance {
            center: [cx, 50.0],
            radius: 5.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 0.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        // Flat layout (loader): [packed x3][non-packed x1]
        let mut circles = vec![
            mk_circle(10.0),  // packed [0], inside brush
            mk_circle(20.0),  // packed [1], inside brush
            mk_circle(30.0),  // packed [2], inside brush
            mk_circle(400.0), // non-packed [3], outside brush
        ];
        let mut rects: Vec<RectInstance> = vec![];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 1u32),
            PackedBatchMeta {
                keys: None,
                data_indices: None,
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        // Loader layout, as documented above: packed batch 1 at [0..3], then
        // non-packed batch 0 appended at [3]. Those are the offsets the loader
        // stamps on each batch's own draw command.
        apply_crossfilter_to_panel(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &packed_meta,
                draw_commands: &[
                    mark_cmd(DrawKind::Circle, 0, 3, 0, 1),
                    mark_cmd(DrawKind::Circle, 3, 1, 0, 0),
                ],
            },
            0,
            &sel,
            0.15,
            &mut circles,
            &mut rects,
        );

        // Packed circles (inside brush) must keep full opacity.
        for i in 0..3 {
            assert!(
                (circles[i].opacity - 1.0).abs() < 0.01,
                "packed circle[{}] inside brush must stay at opacity=1.0, got {}",
                i,
                circles[i].opacity
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
        use ferrum_scene::{CoordKind, MarkBatch, MarkBatchKind, Panel, Rect};

        // Single packed batch with three circles, categories a / b / a.
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
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
                nodes: vec![], // empty — packed batch
                data_indices: None,
                ..MarkBatch::default()
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let grey = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };
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
                    FieldValue::String {
                        value: "a".to_string(),
                    },
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

        let tooltip_bytes = build_tooltip_bytes(&["cat"], &[vec!["a"], vec!["b"], vec!["a"]]);
        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                keys: None,
                data_indices: Some(vec![0, 1, 2]),
                tooltip_bytes: Some(tooltip_bytes),
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        // The sole batch is packed: its instances sit at the absolute sidecar
        // range, which is what the loader stamps on its one draw command.
        let result = resolve_conditionals_with_packed(
            InstanceLayout {
                panels: &panels,
                packed_batch_meta: &packed_meta,
                draw_commands: &[mark_cmd(DrawKind::Circle, 0, 3, 0, 0)],
            },
            &conditionals,
            &selections,
            &base_circles,
            &[],
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
            (
                "cat".to_string(),
                FieldValue::String {
                    value: "a".to_string(),
                },
            ),
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
            (
                "cat".to_string(),
                FieldValue::String {
                    value: "a".to_string(),
                },
            ),
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
        let field_values = vec![(
            "cat".to_string(),
            FieldValue::String {
                value: "a".to_string(),
            },
        )];
        let result = tooltip_matches(&field_values, |_fname| None);
        assert!(
            !result,
            "missing field (lookup returns None) must return false"
        );
    }

    /// Empty `field_values` slice — vacuously true (all zero constraints pass).
    #[test]
    fn tooltip_matches_empty_field_values_returns_true() {
        let field_values: Vec<(String, FieldValue)> = vec![];
        let result = tooltip_matches(&field_values, |_| None);
        assert!(
            result,
            "empty field_values must return true (vacuous truth)"
        );
    }

    /// Typed comparison: numeric tooltip string "42" matches `FieldValue::Number{42.0}`.
    #[test]
    fn tooltip_matches_numeric_field_value_typed_comparison() {
        let field_values = vec![("x".to_string(), FieldValue::Number { value: 42.0 })];
        // The string "42" must match Number{42.0} via field_value_matches_tooltip.
        let result = tooltip_matches(&field_values, |_| Some("42".to_string()));
        assert!(
            result,
            "string '42' must match Number(42.0) via typed comparison"
        );
    }

    /// Bool field value: "true" matches `FieldValue::Bool{true}`.
    #[test]
    fn tooltip_matches_bool_field_value() {
        let field_values = vec![("flag".to_string(), FieldValue::Bool { value: true })];
        let result = tooltip_matches(&field_values, |_| Some("true".to_string()));
        assert!(result, "'true' string must match Bool(true)");

        let result_false = tooltip_matches(&field_values, |_| Some("false".to_string()));
        assert!(!result_false, "'false' string must not match Bool(true)");
    }

    // ── a size conditional on a line ────────────────────────────────────────

    fn size_conditional() -> ConditionalEncoding {
        ConditionalEncoding {
            selection_name: "s".to_string(),
            channel: ChannelName::Size,
            if_selected: EncodingValue::Size { value: 100.0 },
            if_not: EncodingValue::Size { value: 20.0 },
        }
    }

    fn selections_of(sel: SelectionState) -> HashMap<String, SelectionState> {
        HashMap::from([("s".to_string(), sel)])
    }

    fn point_selection(indices: &[usize]) -> HashMap<String, SelectionState> {
        selections_of(SelectionState::Point { indices: indices.to_vec(), field_values: Vec::new() })
    }

    fn line_stroke() -> ferrum_scene::StrokeStyle {
        ferrum_scene::StrokeStyle {
            color: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            width: 1.5,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        }
    }

    /// A one-panel scene drawing `node` as its only line batch.
    fn line_scene(node: SceneNode) -> ferrum_scene::SceneGraph {
        use ferrum_scene::{CoordKind, Rect};
        let area = Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 };
        let panel = Panel {
            id: 0,
            plot_area: area.clone(),
            clip: area,
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![MarkBatch { kind: ferrum_scene::MarkBatchKind::Line, nodes: vec![node], ..MarkBatch::default() }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        };
        ferrum_scene::SceneGraph {
            width: EXTENT[2] as f64,
            height: EXTENT[3] as f64,
            background: None,
            title: vec![],
            panels: vec![panel],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: ferrum_scene::InteractionConfig::default(),
            chart_description: None,
        }
    }

    /// One line node whose width vertices come from `rows`, answering to the
    /// scene's first conditional: the shape `mark_line` emits under a size
    /// conditional.
    fn conditional_line_scene(node: SceneNode, rows: Vec<usize>) -> ferrum_scene::SceneGraph {
        conditional_line_scene_rows(node, rows.into_iter().map(ferrum_scene::VertexRow::Row).collect())
    }

    fn conditional_line_scene_rows(node: SceneNode, rows: Vec<ferrum_scene::VertexRow>) -> ferrum_scene::SceneGraph {
        let mut scene = line_scene(node);
        let batch = &mut scene.panels[0].marks[0];
        batch.data_indices = match rows.first() {
            Some(ferrum_scene::VertexRow::Row(r)) => Some(vec![*r]),
            _ => Some(vec![0]),
        };
        batch.vertex_rows = Some(vec![rows]);
        batch.size_conditionals = Some(vec![0]);
        scene.interaction.conditionals = vec![size_conditional()];
        scene
    }

    fn zigzag() -> SceneNode {
        SceneNode::Polyline {
            points: vec![(60.0, 100.0), (160.0, 100.0), (260.0, 200.0), (360.0, 300.0)],
            style: line_stroke(),
            widths: None,
        }
    }

    fn node_widths(panels: &[Panel]) -> Option<Vec<f64>> {
        match &panels[0].marks[0].nodes[0] {
            SceneNode::Polyline { widths, .. } | SceneNode::Path { widths, .. } => widths.clone(),
            other => panic!("not a line node: {other:?}"),
        }
    }

    fn mesh_bytes(mesh: &lyon::tessellation::VertexBuffers<crate::tessellate::MeshVertex, u32>) -> (Vec<u8>, Vec<u32>) {
        (bytemuck::cast_slice(&mesh.vertices).to_vec(), mesh.indices.clone())
    }

    const EXTENT: [f32; 4] = [0.0, 0.0, 500.0, 400.0];

    /// A point selection redraws each vertex at the dot diameter of its own
    /// row's branch, and the redrawn mesh is the tapered tessellation of
    /// exactly those widths: the same bytes the loader produces for a line
    /// that carried them.
    #[test]
    fn a_selection_redraws_each_line_vertex_at_its_own_rows_branch() {
        let scene = conditional_line_scene(zigzag(), vec![10, 11, 12, 13]);
        let conditionals = [size_conditional()];
        let lines = conditional_line_panels(&scene.panels, &conditionals, &point_selection(&[11, 13]))
            .expect("a line carrying vertex_rows is redrawn");

        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        assert_eq!(node_widths(&lines), Some(vec![off, on, off, on]));

        let (mesh, ranges) = crate::scene_load::mark_mesh(&lines, EXTENT).unwrap();
        let mut tapered = scene.clone();
        if let SceneNode::Polyline { widths, .. } = &mut tapered.panels[0].marks[0].nodes[0] {
            *widths = Some(vec![off, on, off, on]);
        }
        let want = load_scene(&tapered).unwrap();
        assert_eq!(mesh_bytes(&mesh), mesh_bytes(&want.mesh_buffers));
        assert_eq!(format!("{ranges:?}"), format!("{:?}", want.mark_mesh_panels));

        // Every vertex sits between the two branches' half-widths (a turn's
        // bracketing vertices take the interpolated width), reaching both.
        let half_widths: Vec<f32> = mesh.vertices.iter().map(|v| v.half_width).collect();
        let (lo, hi) = half_widths.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &w| (lo.min(w), hi.max(w)));
        assert!((lo - (off / 2.0) as f32).abs() < 1e-4 && (hi - (on / 2.0) as f32).abs() < 1e-4, "{lo}..{hi}");

        let base = load_scene(&scene).unwrap();
        assert_ne!(mesh_bytes(&mesh), mesh_bytes(&base.mesh_buffers), "the redraw must change the loaded mesh");
    }

    /// Clicking a mark outside the line (its row is no vertex's) puts every
    /// vertex on the unselected branch: one constant width along the line.
    #[test]
    fn a_selection_missing_every_row_draws_the_unselected_branch_throughout() {
        let scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        let lines = conditional_line_panels(&scene.panels, &[size_conditional()], &point_selection(&[4])).unwrap();
        assert_eq!(node_widths(&lines), Some(vec![ferrum_scene::dot_diameter(20.0); 4]));
    }

    /// With no active selection the line keeps no `widths` and tessellates
    /// byte for byte as it loaded, so clearing a selection restores it.
    #[test]
    fn an_empty_selection_tessellates_the_line_as_loaded() {
        let scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        let base = load_scene(&scene).unwrap();
        for selections in [selections_of(SelectionState::Empty), HashMap::new()] {
            let lines = conditional_line_panels(&scene.panels, &[size_conditional()], &selections).unwrap();
            assert_eq!(node_widths(&lines), None);
            let (mesh, _) = crate::scene_load::mark_mesh(&lines, EXTENT).unwrap();
            assert_eq!(mesh_bytes(&mesh), mesh_bytes(&base.mesh_buffers));
        }
    }

    /// A brush selects a stepped path's vertices by containment of each
    /// command's endpoint, the `H`/`V` shorthand included.
    #[test]
    fn a_brush_selects_each_step_vertex_by_its_endpoint() {
        use ferrum_scene::{FillStroke, PathCmd};
        let step = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 60.0, y: 100.0 },
                PathCmd::HLineTo { x: 160.0 },
                PathCmd::VLineTo { y: 200.0 },
                PathCmd::HLineTo { x: 260.0 },
            ],
            style: FillStroke {
                fill: None,
                stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
                stroke_width: 1.5,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
            closed: false,
            widths: None,
        };
        let scene = conditional_line_scene(step, vec![0, 0, 1, 1]);
        let brush = SelectionState::Interval { x_range: Some((150.0, 300.0)), y_range: Some((150.0, 250.0)) };
        let lines = conditional_line_panels(&scene.panels, &[size_conditional()], &selections_of(brush)).unwrap();
        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        // Endpoints (60,100) (160,100) (160,200) (260,200): only the last two
        // fall inside the brush's y band.
        assert_eq!(node_widths(&lines), Some(vec![off, off, on, on]));
        assert!(crate::scene_load::mark_mesh(&lines, EXTENT).is_ok());
    }

    /// Restyled widths are baked like every other stroke width: times the
    /// panel's layout `mag`.
    #[test]
    fn restyled_widths_scale_by_the_layout_mag() {
        let mut scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        scene.panels[0].layout_scale = LayoutScale { sx: 2.0, sy: 2.0, tx: 0.0, ty: 0.0 };
        let baked = crate::scene_load::bake_panels(&scene);
        let lines = conditional_line_panels(&baked, &[size_conditional()], &point_selection(&[0])).unwrap();
        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        assert_eq!(node_widths(&lines), Some(vec![2.0 * on, 2.0 * off, 2.0 * off, 2.0 * off]));
    }

    /// Under a non-identity layout (`sx != sy`, translated) the redrawn mesh
    /// of the baked panels is byte for byte the loader's mesh, ranges and plot
    /// area for the same widths: at rest, and restyled against a line that
    /// loaded with the restyled widths in its own unbaked space.
    #[test]
    fn a_non_identity_layout_redraw_matches_the_loaded_mesh() {
        let layout = LayoutScale { sx: 2.0, sy: 1.5, tx: 10.0, ty: 20.0 };
        let mut scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        scene.panels[0].layout_scale = layout;
        let baked = crate::scene_load::bake_panels(&scene);
        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        let mut tapered = scene.clone();
        if let SceneNode::Polyline { widths, .. } = &mut tapered.panels[0].marks[0].nodes[0] {
            *widths = Some(vec![on, off, off, off]);
        }
        for (selections, loaded) in [(HashMap::new(), &scene), (point_selection(&[0]), &tapered)] {
            let lines = conditional_line_panels(&baked, &[size_conditional()], &selections).unwrap();
            let (mesh, ranges) = crate::scene_load::mark_mesh(&lines, EXTENT).unwrap();
            let want = load_scene(loaded).unwrap();
            assert_eq!(mesh_bytes(&mesh), mesh_bytes(&want.mesh_buffers));
            assert_eq!(format!("{ranges:?}"), format!("{:?}", want.mark_mesh_panels));
        }

        let mut identity = tapered.clone();
        identity.panels[0].layout_scale = LayoutScale::identity();
        let at_identity = load_scene(&identity).unwrap();
        assert_ne!(
            mesh_bytes(&load_scene(&tapered).unwrap().mesh_buffers),
            mesh_bytes(&at_identity.mesh_buffers),
            "the layout must move the mesh, or the equality above proves nothing"
        );
    }

    /// A loaded scene whose rows do not carry one row per vertex, or one list
    /// per line, is refused when it loads, in the voice of a misaligned
    /// `widths`, and never drawn at a guessed width; a scene with no such
    /// line needs no redraw.
    #[test]
    fn misaligned_vertex_rows_refuse_at_load() {
        use ferrum_scene::TaperError;
        let short = conditional_line_scene(zigzag(), vec![0, 1, 2]);
        let err = load_scene(&short).err();
        assert_eq!(err, Some(TaperError::RowsMisaligned { points: 4, rows: 3 }));
        assert_eq!(
            crate::error::WasmRenderError::from(err.unwrap()).to_string(),
            "mark_line: the tapered line cannot be drawn: tapered stroke has 3 vertex rows for 4 vertices; it \
             needs one row per vertex"
        );
        let mut two_lists = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        two_lists.panels[0].marks[0].vertex_rows.as_mut().unwrap().push(vec![]);
        assert_eq!(load_scene(&two_lists).err(), Some(TaperError::RowListsMisaligned { nodes: 1, lists: 2 }));
        assert!(load_scene(&conditional_line_scene(zigzag(), vec![0, 1, 2, 3])).is_ok(), "aligned rows load");

        let plain = line_scene(zigzag());
        assert!(conditional_line_panels(&plain.panels, &[size_conditional()], &point_selection(&[0])).is_none());
    }

    /// A vertex a break cut between rows on different branches draws the two
    /// branch widths interpolated at the cut, by the taper helper's rule.
    #[test]
    fn a_cut_vertex_interpolates_its_two_rows_branch_widths() {
        use ferrum_scene::VertexRow;
        let cut = |from, to, t| VertexRow::Cut { from: Box::new(VertexRow::Row(from)), to: Box::new(VertexRow::Row(to)), t };
        // Rows 1 (unselected) and 2 (selected) meet at a cut a quarter along.
        let rows = vec![VertexRow::Row(1), cut(1, 2, 0.25), VertexRow::Row(2), VertexRow::Row(2)];
        let scene = conditional_line_scene_rows(zigzag(), rows);
        let lines = conditional_line_panels(&scene.panels, &[size_conditional()], &point_selection(&[2])).unwrap();
        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        let at_cut = off + (on - off) * 0.25;
        assert_eq!(node_widths(&lines), Some(vec![off, at_cut, on, on]));
        // A cut nested in a cut interpolates its inner end first.
        let nested = VertexRow::Cut { from: Box::new(cut(1, 2, 0.5)), to: Box::new(VertexRow::Row(2)), t: 0.5 };
        let scene = conditional_line_scene_rows(zigzag(), vec![VertexRow::Row(1), nested, VertexRow::Row(2), VertexRow::Row(2)]);
        let lines = conditional_line_panels(&scene.panels, &[size_conditional()], &point_selection(&[2])).unwrap();
        let inner = off + (on - off) * 0.5;
        assert_eq!(node_widths(&lines).unwrap()[1], inner + (on - inner) * 0.5);
    }

    /// A restyled line is hit-tested at the width it draws: a point beyond
    /// its loaded width but within the selected branch's misses on the loaded
    /// panels and hits on the restyled ones.
    #[test]
    fn a_restyled_line_hits_at_its_restyled_width() {
        let scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        let lines = conditional_line_panels(&scene.panels, &[size_conditional()], &point_selection(&[0, 1, 2, 3])).unwrap();
        let zoom = crate::zoom_pan::ZoomPanState::new(1, &ferrum_scene::InteractionConfig::default());
        let plan = crate::hit_test::SlotRescalePlan { slot_rescales: &[], panel_slot_counts: &[1] };
        // 8 px below the first segment: beyond the loaded 1.5 px, within the
        // selected branch's ~11.3 px.
        let (x, y) = (110.0, 108.0);
        assert!(crate::hit_test::hit_test(&scene.panels, x, y, &zoom, &plan).is_none());
        assert!(crate::hit_test::hit_test(&lines, x, y, &zoom, &plan).is_some());
    }

    /// A line answers only to the conditionals its batch names. A composite
    /// sibling leaf's `size` conditional (a point's, with a size-0 branch a
    /// point may draw and a line may not) neither restyles the line nor
    /// leaves it a width the mesh refuses, alone or beside the line's own.
    #[test]
    fn a_line_folds_only_the_size_conditionals_its_batch_names() {
        let sibling = ConditionalEncoding {
            selection_name: "p".to_string(),
            if_not: EncodingValue::Size { value: 0.0 },
            ..size_conditional()
        };
        let mut scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        scene.interaction.conditionals.push(sibling);
        let conditionals = scene.interaction.conditionals.clone();
        let sibling_only =
            HashMap::from([("p".to_string(), SelectionState::Point { indices: vec![9], field_values: Vec::new() })]);
        let lines = conditional_line_panels(&scene.panels, &conditionals, &sibling_only).unwrap();
        assert_eq!(node_widths(&lines), None, "the sibling's conditional does not reach the line");
        assert!(crate::scene_load::mark_mesh(&lines, EXTENT).is_ok());

        let mut both = sibling_only;
        both.extend(point_selection(&[1]));
        let lines = conditional_line_panels(&scene.panels, &conditionals, &both).unwrap();
        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        assert_eq!(node_widths(&lines), Some(vec![off, on, off, off]), "its own conditional still restyles it");
    }

    /// A `size` conditional reaches a point batch that names its own
    /// `size_conditionals` only when it is named there; a batch that names
    /// none answers to every one. A line layer's conditional therefore never
    /// resizes a sibling point layer's dots.
    #[test]
    fn a_size_conditional_resizes_only_the_point_batches_in_its_scope() {
        use ferrum_scene::{FillStroke, MarkBatchKind};
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
        let mut scene = line_scene(zigzag());
        scene.panels[0].marks[0] = MarkBatch {
            kind: MarkBatchKind::Point,
            nodes: vec![SceneNode::Circle { cx: 100.0, cy: 100.0, r: 2.0, style }],
            data_indices: Some(vec![0]),
            ..MarkBatch::default()
        };
        scene.interaction.conditionals = vec![size_conditional()];
        let radius = |scope: Option<Vec<usize>>| {
            let mut scene = scene.clone();
            scene.panels[0].marks[0].size_conditionals = scope;
            let data = load_scene(&scene).unwrap();
            let base = data.circle_instances[0].radius;
            let out = resolve_conditionals(
                &scene.panels,
                &scene.interaction.conditionals,
                &point_selection(&[0]),
                &data.circle_instances,
                &data.rect_instances,
            );
            (base, out.circle_instances[0].radius)
        };
        let (base, out_of_scope) = radius(Some(vec![]));
        assert_eq!(out_of_scope, base, "a conditional outside the batch's scope leaves the dot as loaded");
        let selected = (100.0_f64 / std::f64::consts::PI).sqrt() as f32;
        assert!((radius(Some(vec![0])).1 - selected).abs() < 1e-4, "a named conditional resizes the dot");
        // Vacuity control: an unscoped batch answers to every conditional.
        assert!((radius(None).1 - selected).abs() < 1e-4);
        assert_ne!(base, selected);
    }

    /// A non-size conditional beside the line's `size` conditional does not
    /// unset the size branch each vertex picked.
    #[test]
    fn a_later_non_size_conditional_keeps_each_vertex_on_its_size_branch() {
        let color = |r| EncodingValue::Color { value: Color { r, g: 0, b: 0, a: 255 } };
        let mut scene = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        scene.interaction.conditionals.push(ConditionalEncoding {
            selection_name: "s".to_string(),
            channel: ChannelName::Color,
            if_selected: color(255),
            if_not: color(0),
        });
        let lines =
            conditional_line_panels(&scene.panels, &scene.interaction.conditionals, &point_selection(&[1])).unwrap();
        let (on, off) = (ferrum_scene::dot_diameter(100.0), ferrum_scene::dot_diameter(20.0));
        assert_eq!(node_widths(&lines), Some(vec![off, on, off, off]));
    }

    /// The loader refuses a line whose `vertex_rows` come without
    /// `size_conditionals`, and a batch naming a conditional the scene lacks
    /// or one that sets no `size`.
    #[test]
    fn unpaired_or_out_of_range_size_conditionals_refuse_at_load() {
        use ferrum_scene::TaperError;
        let mut unpaired = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        unpaired.panels[0].marks[0].size_conditionals = None;
        assert_eq!(load_scene(&unpaired).err(), Some(TaperError::SizeConditionalsUnpaired));
        let mut not_size = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        not_size.interaction.conditionals.push(ConditionalEncoding {
            selection_name: "s".to_string(),
            channel: ChannelName::Color,
            if_selected: EncodingValue::Color { value: Color { r: 255, g: 0, b: 0, a: 255 } },
            if_not: EncodingValue::Color { value: Color { r: 0, g: 0, b: 0, a: 255 } },
        });
        not_size.panels[0].marks[0].size_conditionals = Some(vec![0, 1]);
        assert_eq!(load_scene(&not_size).err(), Some(TaperError::NotASizeConditional { index: 1 }));
        // A scope alone, on a batch with no rows, loads.
        let mut scoped = line_scene(zigzag());
        scoped.panels[0].marks[0].size_conditionals = Some(vec![]);
        assert!(load_scene(&scoped).is_ok());
        let mut out_of_range = conditional_line_scene(zigzag(), vec![0, 1, 2, 3]);
        out_of_range.panels[0].marks[0].size_conditionals = Some(vec![0, 1]);
        assert_eq!(
            load_scene(&out_of_range).err(),
            Some(TaperError::SizeConditionalOutOfRange { index: 1, conditionals: 1 })
        );
    }
}
