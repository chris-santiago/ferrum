//! Scene-to-scene transitions.
//!
//! Two pairing rules decide which old mark becomes which new mark (spec §4.3 /
//! GH #93):
//!
//! * **Index-zip** — the historical rule: old instance `i` becomes new instance
//!   `i` over the whole flat instance array. Used whenever no batch in the
//!   scene qualifies for keyed pairing, and byte-identical to the pre-#93
//!   behavior for those scenes.
//! * **Keyed** — object constancy from `encode(key=...)`: within a batch, old
//!   and new instances pair by key, so an insert/delete/reorder moves each mark
//!   to *its own* new position instead of morphing into its neighbour. Keys
//!   only in the new batch **enter** (final geometry, opacity 0 → target); keys
//!   only in the old batch **exit** (old geometry, opacity target → 0, dropped
//!   at `t = 1`).
//!
//! Keyed pairing applies per batch, and only when BOTH sides of the pair carry
//! keys that are aligned and unique; anything else falls back to index-zip for
//! that batch as a whole (no partial keying).

use std::collections::{HashMap, HashSet};

use ferrum_scene::Panel;

use crate::error::WasmRenderError;
use crate::scene_load::{
    batch_instance_spans, BatchInstanceSpan, CircleInstance, DrawCommand, DrawKind,
    PackedBatchMeta, RectInstance, SceneData,
};

/// Field-by-field interpolation for one GPU instance record.
///
/// Implemented by the two instance types the transition machinery moves
/// ([`CircleInstance`] and [`RectInstance`]) so the pairing logic — index-zip,
/// keyed matching, enter/exit fades — is written once instead of once per
/// instance kind.
trait InstanceLerp: Copy {
    /// Interpolate every continuous field from `self` toward `other` at `t`.
    fn lerp(&self, other: &Self, t: f32) -> Self;

    /// Scale both opacity channels by `factor`, keeping geometry and color.
    /// This is the enter/exit fade: `factor` ramps 0 → 1 for an entering mark
    /// and 1 → 0 for an exiting one.
    fn scale_alpha(&self, factor: f32) -> Self;
}

impl InstanceLerp for CircleInstance {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        CircleInstance {
            center: [
                self.center[0] + (other.center[0] - self.center[0]) * t,
                self.center[1] + (other.center[1] - self.center[1]) * t,
            ],
            radius: self.radius + (other.radius - self.radius) * t,
            fill_color: lerp_color(self.fill_color, other.fill_color, t),
            stroke_color: lerp_color(self.stroke_color, other.stroke_color, t),
            stroke_width: self.stroke_width + (other.stroke_width - self.stroke_width) * t,
            opacity: self.opacity + (other.opacity - self.opacity) * t,
            stroke_opacity: self.stroke_opacity
                + (other.stroke_opacity - self.stroke_opacity) * t,
            stroke_dash: self.stroke_dash, // dash palette index: use old value (no lerp for discrete)
            angle: self.angle + (other.angle - self.angle) * t,
        }
    }

    fn scale_alpha(&self, factor: f32) -> Self {
        CircleInstance {
            opacity: self.opacity * factor,
            stroke_opacity: self.stroke_opacity * factor,
            ..*self
        }
    }
}

impl InstanceLerp for RectInstance {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        RectInstance {
            position: [
                self.position[0] + (other.position[0] - self.position[0]) * t,
                self.position[1] + (other.position[1] - self.position[1]) * t,
            ],
            size: [
                self.size[0] + (other.size[0] - self.size[0]) * t,
                self.size[1] + (other.size[1] - self.size[1]) * t,
            ],
            corner_radius: self.corner_radius + (other.corner_radius - self.corner_radius) * t,
            fill_color: lerp_color(self.fill_color, other.fill_color, t),
            stroke_color: lerp_color(self.stroke_color, other.stroke_color, t),
            stroke_width: self.stroke_width + (other.stroke_width - self.stroke_width) * t,
            opacity: self.opacity + (other.opacity - self.opacity) * t,
            stroke_opacity: self.stroke_opacity
                + (other.stroke_opacity - self.stroke_opacity) * t,
            stroke_dash: self.stroke_dash, // palette index: use old value (discrete, no lerp)
            angle: self.angle + (other.angle - self.angle) * t,
        }
    }

    fn scale_alpha(&self, factor: f32) -> Self {
        RectInstance {
            opacity: self.opacity * factor,
            stroke_opacity: self.stroke_opacity * factor,
            ..*self
        }
    }
}

/// Interpolate two flat instance arrays index-to-index, truncating to the
/// shorter one — the historical whole-scene pairing.
fn zip_lerp<T: InstanceLerp>(old: &[T], new: &[T], t: f32) -> Vec<T> {
    old.iter()
        .zip(new.iter())
        .map(|(a, b)| a.lerp(b, t))
        .collect()
}

// Typed [`zip_lerp`] aliases, kept only so the pre-#93 test corpus below still
// reads in terms of the two instance kinds. They have no production caller:
// the runtime enters through `plan_transition` / `interpolate`, and the
// whole-plan `TransitionPlan::IndexZip` arm calls `zip_lerp` directly.
#[cfg(test)]
fn lerp_circles(old: &[CircleInstance], new: &[CircleInstance], t: f32) -> Vec<CircleInstance> {
    zip_lerp(old, new, t)
}

#[cfg(test)]
fn lerp_rects(old: &[RectInstance], new: &[RectInstance], t: f32) -> Vec<RectInstance> {
    zip_lerp(old, new, t)
}

pub fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0_f32).powi(3) / 2.0
    }
}

fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

// ── Keyed pairing (spec §4.3 / GH #93) ──────────────────────────────────────

/// One side of a transition: everything the matcher needs to locate a scene's
/// mark instances and read their keys.
///
/// `panels` may be raw `SceneGraph::panels` or baked geometry — see
/// [`batch_instance_spans`].
#[derive(Clone, Copy)]
pub(crate) struct TransitionSide<'a> {
    pub(crate) panels: &'a [Panel],
    pub(crate) packed_batch_meta: &'a HashMap<(u32, u32), PackedBatchMeta>,
    pub(crate) draw_commands: &'a [DrawCommand],
}

/// The flat GPU instance arrays of one scene side.
#[derive(Clone, Copy)]
pub(crate) struct Instances<'a> {
    pub(crate) circles: &'a [CircleInstance],
    pub(crate) rects: &'a [RectInstance],
}

/// Everything a transition needs from the scene it is moving away FROM, and
/// nothing else: the two instance arrays it interpolates, plus the batch
/// structure the matcher pairs on.
///
/// **Old-side identity (spec §4.3).** The renderer snapshots the outgoing
/// scene in `loadScene`, before the incoming one replaces it, and pairs
/// against that snapshot. Re-parsing the previous scene's JSON cannot
/// substitute for it: the packer moves a batch above the pack threshold
/// entirely into the binary sidecar, clearing `nodes` AND `keys`
/// (`pack_instances::extract_packed_bytes`), so the JSON of a packed batch has
/// no instances and no identity at all. The in-memory form is the only place a
/// large batch's keys exist on the old side, which is why
/// [`SceneSnapshot::from_scene_data`] is the primary path and
/// [`SceneSnapshot::from_scene_json`] is the fallback for the first load,
/// where there is no predecessor to snapshot.
pub(crate) struct SceneSnapshot {
    circles: Vec<CircleInstance>,
    rects: Vec<RectInstance>,
    packed_batch_meta: HashMap<(u32, u32), PackedBatchMeta>,
    draw_commands: Vec<DrawCommand>,
    panels: Vec<Panel>,
}

impl SceneSnapshot {
    /// Snapshot a loaded scene: the instances the GPU actually drew, together
    /// with the panels they came from. Takes ownership so the caller's
    /// `SceneData` (and the GPU buffers alongside it) can be dropped.
    pub(crate) fn from_scene_data(data: SceneData, panels: Vec<Panel>) -> Self {
        SceneSnapshot {
            circles: data.circle_instances,
            rects: data.rect_instances,
            packed_batch_meta: data.packed_batch_meta,
            draw_commands: data.draw_commands,
            panels,
        }
    }

    /// Rebuild a snapshot by parsing a scene's JSON — the fallback when no
    /// in-memory predecessor exists. Carries no packed instances and no packed
    /// keys (see the type docs), so batches above the pack threshold pair by
    /// index through this path.
    pub(crate) fn from_scene_json(scene_json: &str) -> Result<Self, WasmRenderError> {
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(scene_json)
            .map_err(|e| WasmRenderError::SceneDeserialization(e.to_string()))?;
        let data = crate::scene_load::load_scene(&scene);
        Ok(Self::from_scene_data(data, scene.panels))
    }

    /// This snapshot as the old side of a pairing.
    pub(crate) fn side(&self) -> TransitionSide<'_> {
        TransitionSide {
            panels: &self.panels,
            packed_batch_meta: &self.packed_batch_meta,
            draw_commands: &self.draw_commands,
        }
    }

    /// This snapshot's instance arrays.
    pub(crate) fn instances(&self) -> Instances<'_> {
        Instances {
            circles: &self.circles,
            rects: &self.rects,
        }
    }
}

/// How one instance run's old instances map onto its new ones.
#[derive(Debug, PartialEq)]
pub(crate) enum RunPairing {
    /// Positional: new instance `i` interpolates from old instance `i` while
    /// both exist, within this run's own span. Instances past the shorter side
    /// simply appear or disappear, with no fade: a new instance beyond the old
    /// side's end is written at its final geometry, and an old instance beyond
    /// the new side's end is not drawn.
    ///
    /// No fade, deliberately. A fade is the visible consequence of identity —
    /// spec §4.3 grants enter/exit only to a key present on one side. An
    /// unkeyed batch has no identity, so "the 3rd instance" is not a mark that
    /// arrived; treating it as one would claim knowledge the runtime does not
    /// have. This is also the pre-#93 behavior for those slots, now applied
    /// per batch instead of across the whole flat array.
    IndexZip,
    /// Keyed: `sources[i]` is the ABSOLUTE old-array index that new instance
    /// `i` transitions from, or `None` when that key is new (it enters).
    /// `exits` are absolute old-array indices whose keys are gone.
    Keyed {
        sources: Vec<Option<usize>>,
        exits: Vec<usize>,
    },
}

/// One mark batch's instances of a single [`DrawKind`], paired old-to-new.
pub(crate) struct InstanceRun {
    kind: DrawKind,
    old_start: usize,
    old_len: usize,
    new_start: usize,
    new_len: usize,
    pairing: RunPairing,
    /// Draw-command template for exiting instances, retargeted at apply time
    /// onto the appended exit block. `None` when neither scene has a draw
    /// command for this run, in which case exits cannot be drawn at all.
    exit_draw: Option<DrawCommand>,
}

/// A resolved pairing for a whole scene transition, computed once when the
/// transition starts and reused for every frame.
pub(crate) enum TransitionPlan {
    /// No batch pairs by key: interpolate the flat arrays whole, exactly as
    /// the runtime did before #93. This is the byte-identity guarantee for
    /// every unkeyed chart — the keyed machinery is never entered.
    IndexZip,
    /// At least one batch pairs by key. Every positionally-paired batch gets
    /// its own run (keyed or index-zip); non-mark chrome instances, which
    /// carry no keys and are index-aligned by construction, keep the
    /// index-wise interpolation the whole-array path gave them.
    Keyed { runs: Vec<InstanceRun> },
}

/// One interpolated frame: the instance arrays to upload, plus any extra draw
/// commands needed to paint exiting marks.
pub(crate) struct TransitionFrame {
    pub(crate) circles: Vec<CircleInstance>,
    pub(crate) rects: Vec<RectInstance>,
    /// Draw commands for exiting instances, appended past the new scene's own
    /// instance ranges so every existing command still addresses the same
    /// slots. Empty unless a keyed batch lost instances. Callers append these
    /// to the new scene's `draw_commands`, which paints fading-out marks above
    /// the rest of the frame for the duration of the transition — a deliberate
    /// simplification over threading each exit block back to its batch's
    /// position in the command list.
    pub(crate) exit_draws: Vec<DrawCommand>,
}

/// Resolve the pairing between two scenes.
///
/// Batches pair positionally by `(panel, batch)` index. Within a paired batch,
/// instances of each kind pair by key when [`aligned_keys`] yields unique keys
/// for both sides, and positionally otherwise — the fallback is per batch and
/// total, never partial.
pub(crate) fn plan_transition(old: TransitionSide<'_>, new: TransitionSide<'_>) -> TransitionPlan {
    let old_spans = batch_instance_spans(old.panels, old.packed_batch_meta);
    let new_spans = batch_instance_spans(new.panels, new.packed_batch_meta);
    let old_by_id: HashMap<(usize, usize), &BatchInstanceSpan<'_>> = old_spans
        .iter()
        .map(|s| ((s.panel_idx, s.batch_idx), s))
        .collect();

    let mut runs = Vec::new();
    let mut any_keyed = false;

    // Iterating the NEW spans decides both asymmetric cases:
    //
    // * A batch that vanished from the scene is never visited, so it gets no
    //   run and its instances are not drawn during the transition. (Pre-#93
    //   the whole-array zip still consumed them at their flat indices,
    //   morphing them into whichever new marks occupied those slots — so this
    //   is a change, not a preservation, and only in a keyed scene: an unkeyed
    //   one takes the whole-plan `IndexZip` path and never reaches here.)
    // * A batch that is NEW in the scene has `old_span == None`, hence an
    //   empty old range and `RunPairing::IndexZip`, which writes every one of
    //   its slots at final geometry — the run still owns its span, so those
    //   marks appear rather than morphing out of unrelated old instances.
    for new_span in &new_spans {
        let old_span = old_by_id
            .get(&(new_span.panel_idx, new_span.batch_idx))
            .copied();
        for kind in [DrawKind::Circle, DrawKind::Rect] {
            let new_range = instance_range(new_span, kind);
            let old_range = old_span.map_or(0..0, |s| instance_range(s, kind));
            if new_range.is_empty() && old_range.is_empty() {
                continue;
            }

            let pairing = old_span
                .and_then(|old_span| keyed_pairing(old_span, new_span, kind, old_range.start))
                .unwrap_or(RunPairing::IndexZip);
            any_keyed |= matches!(pairing, RunPairing::Keyed { .. });

            let has_exits = matches!(&pairing, RunPairing::Keyed { exits, .. } if !exits.is_empty());
            let exit_draw = if has_exits {
                find_draw_command(new.draw_commands, kind, &new_range)
                    .or_else(|| find_draw_command(old.draw_commands, kind, &old_range))
                    .cloned()
            } else {
                None
            };

            runs.push(InstanceRun {
                kind,
                old_start: old_range.start,
                old_len: old_range.len(),
                new_start: new_range.start,
                new_len: new_range.len(),
                pairing,
                exit_draw,
            });
        }
    }

    if any_keyed {
        TransitionPlan::Keyed { runs }
    } else {
        TransitionPlan::IndexZip
    }
}

/// Interpolate one frame at eased progress `t` ∈ [0, 1].
pub(crate) fn interpolate(
    plan: &TransitionPlan,
    old: Instances<'_>,
    new: Instances<'_>,
    t: f32,
) -> TransitionFrame {
    match plan {
        TransitionPlan::IndexZip => TransitionFrame {
            circles: zip_lerp(old.circles, new.circles, t),
            rects: zip_lerp(old.rects, new.rects, t),
            exit_draws: Vec::new(),
        },
        TransitionPlan::Keyed { runs } => {
            let mut circles = chrome_fill(old.circles, new.circles, t);
            let mut rects = chrome_fill(old.rects, new.rects, t);
            let mut exit_draws = Vec::new();
            for run in runs {
                match run.kind {
                    DrawKind::Circle => {
                        apply_run(run, old.circles, new.circles, &mut circles, t, &mut exit_draws)
                    }
                    DrawKind::Rect => {
                        apply_run(run, old.rects, new.rects, &mut rects, t, &mut exit_draws)
                    }
                }
            }
            TransitionFrame {
                circles,
                rects,
                exit_draws,
            }
        }
    }
}

/// This batch's instance range for `kind`.
fn instance_range(span: &BatchInstanceSpan<'_>, kind: DrawKind) -> std::ops::Range<usize> {
    match kind {
        DrawKind::Circle => span.circles.clone(),
        DrawKind::Rect => span.rects.clone(),
    }
}

/// Keys aligned one-to-one with this batch's instances of `kind`.
///
/// Requires the batch to carry keys, to contribute instances of exactly one
/// kind, and to have one key per instance. `MarkBatch::keys` is aligned to
/// `nodes`, which may include nodes that produce no instance at all (lines,
/// text, …), so a mixed or partially-tessellated batch has no unambiguous
/// key-to-instance mapping and is not eligible for keyed pairing. The packed
/// carrier is always homogeneous, so this only ever rejects JSON-node batches.
///
/// The count check is the runtime end of spec §4.3's alignment invariant. The
/// loud failure lives at the producer, on both emission paths (`mark_nodes`'s
/// node/metadata guard and its packed `HAS_KEYS` sibling); a renderer that
/// panicked here would take down the widget over a defect the producer
/// already refuses to emit, so a desynced batch falls back to index-zip
/// instead of pairing marks against the wrong keys.
fn aligned_keys<'a>(span: &'a BatchInstanceSpan<'_>, kind: DrawKind) -> Option<&'a [String]> {
    let keys = span.keys()?;
    let (this, other) = match kind {
        DrawKind::Circle => (span.circles.len(), span.rects.len()),
        DrawKind::Rect => (span.rects.len(), span.circles.len()),
    };
    if other != 0 || keys.len() != this {
        return None;
    }
    Some(keys)
}

/// Pair one run's instances by key, or `None` to fall back to index-zip.
///
/// Falls back when either side lacks aligned keys, or when either side
/// contains a duplicate key — a non-injective key column (a Boolean or
/// coarse-grained field) cannot define object constancy, and pairing part of
/// such a batch would be worse than pairing none of it.
fn keyed_pairing(
    old_span: &BatchInstanceSpan<'_>,
    new_span: &BatchInstanceSpan<'_>,
    kind: DrawKind,
    old_start: usize,
) -> Option<RunPairing> {
    let old_keys = aligned_keys(old_span, kind)?;
    let new_keys = aligned_keys(new_span, kind)?;

    let mut old_index: HashMap<&str, usize> = HashMap::with_capacity(old_keys.len());
    for (i, key) in old_keys.iter().enumerate() {
        if old_index.insert(key.as_str(), i).is_some() {
            return None;
        }
    }
    let mut new_set: HashSet<&str> = HashSet::with_capacity(new_keys.len());
    for key in new_keys {
        if !new_set.insert(key.as_str()) {
            return None;
        }
    }

    let sources = new_keys
        .iter()
        .map(|key| old_index.get(key.as_str()).map(|i| old_start + i))
        .collect();
    let exits = old_keys
        .iter()
        .enumerate()
        .filter(|(_, key)| !new_set.contains(key.as_str()))
        .map(|(i, _)| old_start + i)
        .collect();
    Some(RunPairing::Keyed { sources, exits })
}

/// The draw command covering exactly `range` in the `kind` instance array.
///
/// The loader emits at most one command per batch per kind, spanning that
/// batch's contiguous instances, so an exact `(kind, start, count)` match
/// identifies it unambiguously.
fn find_draw_command<'a>(
    commands: &'a [DrawCommand],
    kind: DrawKind,
    range: &std::ops::Range<usize>,
) -> Option<&'a DrawCommand> {
    commands.iter().find(|c| {
        c.kind == kind
            && c.instance_start as usize == range.start
            && c.instance_count as usize == range.len()
    })
}

/// Seed the output array with the new scene's instances, then interpolate the
/// indices both scenes share.
///
/// Starting from the new array is what keeps every existing draw command
/// valid: each instance stays in its own slot.
///
/// Every run then overwrites its ENTIRE span — both pairing arms write all of
/// `new_start..new_start + new_len` — so what this pass leaves standing is
/// exactly the instances no mark batch claims: the chrome (grid, axes, legend,
/// title). Chrome is unkeyed and index-aligned by construction, and this
/// interpolates it exactly as the pre-#93 whole-array path did.
fn chrome_fill<T: InstanceLerp>(old: &[T], new: &[T], t: f32) -> Vec<T> {
    let mut out = new.to_vec();
    for (i, o) in old.iter().enumerate().take(new.len()) {
        out[i] = o.lerp(&new[i], t);
    }
    out
}

/// Write one run's interpolated instances into `out`, appending any exiting
/// instances (and their draw command) at the end of the array.
fn apply_run<T: InstanceLerp>(
    run: &InstanceRun,
    old: &[T],
    new: &[T],
    out: &mut Vec<T>,
    t: f32,
    exit_draws: &mut Vec<DrawCommand>,
) {
    // The plan is built against these very arrays; the guard keeps a
    // mismatched caller from panicking mid-frame in the browser.
    if run.new_start + run.new_len > new.len().min(out.len()) {
        return;
    }

    match &run.pairing {
        RunPairing::IndexZip => {
            // Every slot in the run is written, not just the paired prefix: a
            // run owns its whole span. Leaving the tail to `chrome_fill` would
            // interpolate it from whatever unrelated instance happens to sit
            // at that FLAT index in the old array — in a keyed scene that
            // makes one old mark the source for two new marks at once.
            for i in 0..run.new_len {
                let target = &new[run.new_start + i];
                let source = if i < run.old_len {
                    old.get(run.old_start + i)
                } else {
                    None
                };
                out[run.new_start + i] = match source {
                    Some(from) => from.lerp(target, t),
                    None => *target,
                };
            }
        }
        RunPairing::Keyed { sources, exits } => {
            for (i, source) in sources.iter().enumerate().take(run.new_len) {
                let target = &new[run.new_start + i];
                out[run.new_start + i] = match source.and_then(|o| old.get(o)) {
                    Some(from) => from.lerp(target, t),
                    // Enter: final geometry, opacity ramping 0 → target.
                    None => target.scale_alpha(t),
                };
            }
            // Exit: old geometry, opacity ramping target → 0, dropped at t = 1.
            if t < 1.0 && !exits.is_empty() {
                if let Some(template) = &run.exit_draw {
                    let start = out.len();
                    out.extend(
                        exits
                            .iter()
                            .filter_map(|&o| old.get(o))
                            .map(|inst| inst.scale_alpha(1.0 - t)),
                    );
                    let appended = out.len() - start;
                    if appended > 0 {
                        exit_draws.push(DrawCommand {
                            instance_start: start as u32,
                            instance_count: appended as u32,
                            ..template.clone()
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod bug_hunt_tests {
    use super::*;

    fn make_circle(cx: f32, cy: f32, r: f32) -> CircleInstance {
        CircleInstance {
            center: [cx, cy],
            radius: r,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }
    }

    fn make_rect(x: f32, y: f32, w: f32, h: f32) -> RectInstance {
        RectInstance {
            position: [x, y],
            size: [w, h],
            corner_radius: 0.0,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }
    }

    #[test]
    fn bug_hunt_lerp_circles_empty_slices_returns_empty() {
        let result = lerp_circles(&[], &[], 0.5);
        assert!(
            result.is_empty(),
            "lerp_circles on empty slices must return empty vec"
        );
    }

    #[test]
    fn bug_hunt_lerp_rects_empty_slices_returns_empty() {
        let result = lerp_rects(&[], &[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn bug_hunt_lerp_circles_t_zero_returns_old() {
        let old = vec![make_circle(0.0, 0.0, 10.0)];
        let new = vec![make_circle(100.0, 200.0, 20.0)];
        let result = lerp_circles(&old, &new, 0.0);
        assert!(
            (result[0].center[0] - 0.0).abs() < 0.001,
            "t=0 must return old center.x"
        );
        assert!(
            (result[0].radius - 10.0).abs() < 0.001,
            "t=0 must return old radius"
        );
    }

    #[test]
    fn bug_hunt_lerp_circles_t_one_returns_new() {
        let old = vec![make_circle(0.0, 0.0, 10.0)];
        let new = vec![make_circle(100.0, 200.0, 20.0)];
        let result = lerp_circles(&old, &new, 1.0);
        assert!(
            (result[0].center[0] - 100.0).abs() < 0.001,
            "t=1 must return new center.x"
        );
        assert!(
            (result[0].radius - 20.0).abs() < 0.001,
            "t=1 must return new radius"
        );
    }

    #[test]
    fn bug_hunt_lerp_rects_t_zero_returns_old() {
        let old = vec![make_rect(0.0, 0.0, 50.0, 30.0)];
        let new = vec![make_rect(100.0, 100.0, 200.0, 150.0)];
        let result = lerp_rects(&old, &new, 0.0);
        assert!((result[0].position[0] - 0.0).abs() < 0.001);
        assert!((result[0].size[0] - 50.0).abs() < 0.001);
    }

    #[test]
    fn bug_hunt_lerp_rects_t_one_returns_new() {
        let old = vec![make_rect(0.0, 0.0, 50.0, 30.0)];
        let new = vec![make_rect(100.0, 100.0, 200.0, 150.0)];
        let result = lerp_rects(&old, &new, 1.0);
        assert!((result[0].position[0] - 100.0).abs() < 0.001);
        assert!((result[0].size[0] - 200.0).abs() < 0.001);
    }

    #[test]
    fn bug_hunt_lerp_circles_mismatched_lengths_truncates_to_shorter() {
        // When old is longer than new, zip() truncates to min(old.len(), new.len())
        let old = vec![make_circle(0.0, 0.0, 5.0), make_circle(100.0, 100.0, 10.0)];
        let new = vec![make_circle(50.0, 50.0, 8.0)];
        let result = lerp_circles(&old, &new, 0.5);
        // Only 1 result — second old element is dropped by zip
        assert_eq!(
            result.len(),
            1,
            "mismatched lengths must truncate to shorter (zip behaviour)"
        );
    }

    #[test]
    fn bug_hunt_ease_in_out_is_monotone_over_uniform_samples() {
        // ease_in_out_cubic must be monotonically non-decreasing on [0, 1]
        let samples = 100;
        let mut prev = 0.0f32;
        for i in 0..=samples {
            let t = i as f32 / samples as f32;
            let v = ease_in_out_cubic(t);
            assert!(
                v >= prev - 1e-6,
                "ease_in_out_cubic is not monotone at t={t}: prev={prev}, current={v}"
            );
            prev = v;
        }
    }

    #[test]
    fn bug_hunt_ease_in_out_midpoint_continuity() {
        // The function is defined piecewise at t=0.5; both branches must agree.
        let left = ease_in_out_cubic(0.5 - 1e-6);
        let right = ease_in_out_cubic(0.5 + 1e-6);
        assert!(
            (right - left).abs() < 0.001,
            "ease_in_out_cubic discontinuity at 0.5: left={left}, right={right}"
        );
    }

    #[test]
    fn bug_hunt_ease_in_out_values_in_zero_one_range() {
        // All outputs must be in [0, 1] for all inputs in [0, 1].
        for i in 0..=200 {
            let t = i as f32 / 200.0;
            let v = ease_in_out_cubic(t);
            assert!(
                v >= 0.0 && v <= 1.0,
                "ease_in_out_cubic({t}) = {v}, out of [0, 1]"
            );
        }
    }

    #[test]
    fn bug_hunt_lerp_circles_color_clamps_or_remains_in_range() {
        // Verify that lerp of two colors (white/black) at midpoint produces mid-grey.
        let white = CircleInstance {
            center: [0.0, 0.0],
            radius: 1.0,
            fill_color: [1.0, 1.0, 1.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let black = CircleInstance {
            center: [0.0, 0.0],
            radius: 1.0,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let mid = lerp_circles(&[white], &[black], 0.5);
        assert!(
            (mid[0].fill_color[0] - 0.5).abs() < 0.01,
            "mid-grey R channel"
        );
        assert!(
            (mid[0].fill_color[1] - 0.5).abs() < 0.01,
            "mid-grey G channel"
        );
        assert!(
            (mid[0].fill_color[2] - 0.5).abs() < 0.01,
            "mid-grey B channel"
        );
    }

    #[test]
    fn bug_hunt_lerp_rects_corner_radius_interpolates() {
        // corner_radius must be linearly interpolated between old and new.
        let old = make_rect(0.0, 0.0, 100.0, 50.0);
        let mut new_r = make_rect(0.0, 0.0, 100.0, 50.0);
        new_r.corner_radius = 20.0;
        let mid = lerp_rects(&[old], &[new_r], 0.5);
        assert!(
            (mid[0].corner_radius - 10.0).abs() < 0.01,
            "corner_radius must lerp to 10.0 at t=0.5, got {}",
            mid[0].corner_radius
        );
    }

    #[test]
    fn bug_hunt_lerp_circles_new_longer_than_old_truncates() {
        // zip() truncates to min(old.len, new.len). When new is longer, extra new elements dropped.
        let old = vec![make_circle(0.0, 0.0, 5.0)];
        let new = vec![
            make_circle(50.0, 50.0, 8.0),
            make_circle(100.0, 100.0, 10.0),
        ];
        let result = lerp_circles(&old, &new, 0.5);
        assert_eq!(result.len(), 1, "zip truncates to min(1, 2) = 1");
    }

    #[test]
    fn bug_hunt_lerp_circles_nan_in_new_propagates_nan() {
        // NaN in the new circle's center must propagate to the result.
        let old = vec![make_circle(100.0, 200.0, 10.0)];
        let mut nan_circle = make_circle(f32::NAN, 300.0, 20.0);
        let result = lerp_circles(&old, &[nan_circle], 0.5);
        // center[0] = 100.0 + (NaN - 100.0) * 0.5 = NaN
        assert!(
            result[0].center[0].is_nan(),
            "NaN in new.center[0] must propagate: got {}",
            result[0].center[0]
        );
    }

    #[test]
    fn bug_hunt_lerp_circles_infinity_in_radius() {
        // Infinity in new.radius: lerp should propagate infinity.
        let old = vec![make_circle(0.0, 0.0, 10.0)];
        let mut inf_circle = make_circle(0.0, 0.0, f32::INFINITY);
        let result = lerp_circles(&old, &[inf_circle], 0.5);
        assert!(
            result[0].radius.is_infinite(),
            "infinity in new.radius must propagate: got {}",
            result[0].radius
        );
    }

    #[test]
    fn bug_hunt_lerp_rects_nan_in_position_propagates() {
        // NaN in new rect position must propagate.
        let old = vec![make_rect(10.0, 20.0, 100.0, 50.0)];
        let mut nan_rect = make_rect(f32::NAN, 30.0, 200.0, 60.0);
        let result = lerp_rects(&old, &[nan_rect], 0.5);
        assert!(
            result[0].position[0].is_nan(),
            "NaN in new.position[0] must propagate"
        );
    }

    #[test]
    fn bug_hunt_lerp_circles_negative_t() {
        // t < 0 should extrapolate (not clamp). The function doesn't clamp t.
        let old = vec![make_circle(0.0, 0.0, 10.0)];
        let new = vec![make_circle(100.0, 0.0, 10.0)];
        let result = lerp_circles(&old, &new, -0.5);
        // center[0] = 0.0 + (100.0 - 0.0) * (-0.5) = -50.0
        assert!(
            (result[0].center[0] - (-50.0)).abs() < 0.01,
            "negative t must extrapolate; got {}",
            result[0].center[0]
        );
    }

    #[test]
    fn bug_hunt_lerp_circles_t_greater_than_one() {
        // t > 1 should extrapolate.
        let old = vec![make_circle(0.0, 0.0, 10.0)];
        let new = vec![make_circle(100.0, 0.0, 10.0)];
        let result = lerp_circles(&old, &new, 1.5);
        // center[0] = 0.0 + (100.0 - 0.0) * 1.5 = 150.0
        assert!(
            (result[0].center[0] - 150.0).abs() < 0.01,
            "t > 1 must extrapolate; got {}",
            result[0].center[0]
        );
    }

    #[test]
    fn bug_hunt_ease_in_out_cubic_negative_t() {
        // Negative t should produce negative output (not clamped to 0).
        let result = ease_in_out_cubic(-0.5);
        assert!(
            result < 0.0,
            "ease_in_out_cubic(-0.5) should be negative; got {result}"
        );
    }

    #[test]
    fn bug_hunt_ease_in_out_cubic_t_greater_than_one() {
        // t > 1 should produce > 1.0 (not clamped).
        let result = ease_in_out_cubic(1.5);
        assert!(
            result > 1.0,
            "ease_in_out_cubic(1.5) should be > 1.0; got {result}"
        );
    }

    #[test]
    fn bug_hunt_lerp_circles_stroke_dash_uses_old_value() {
        // stroke_dash is discrete (palette index): always uses old value, never interpolated.
        let mut old_c = make_circle(0.0, 0.0, 5.0);
        old_c.stroke_dash = 1.0; // dashed
        let mut new_c = make_circle(0.0, 0.0, 5.0);
        new_c.stroke_dash = 3.0; // dash-dot

        let result = lerp_circles(&[old_c], &[new_c], 0.5);
        assert!(
            (result[0].stroke_dash - 1.0).abs() < 1e-6,
            "stroke_dash must use old value (1.0), not interpolate; got {}",
            result[0].stroke_dash
        );
    }

    #[test]
    fn bug_hunt_lerp_rects_stroke_dash_uses_old_value() {
        let mut old_r = make_rect(0.0, 0.0, 100.0, 50.0);
        old_r.stroke_dash = 2.0; // dotted
        let mut new_r = make_rect(0.0, 0.0, 100.0, 50.0);
        new_r.stroke_dash = 0.0; // solid

        let result = lerp_rects(&[old_r], &[new_r], 0.99);
        assert!(
            (result[0].stroke_dash - 2.0).abs() < 1e-6,
            "stroke_dash must use old value (2.0) at any t; got {}",
            result[0].stroke_dash
        );
    }
}

/// B4 regression: when old_data == new_data, lerp produces no visible change.
/// This demonstrates the symptom: transition from a scene to itself is a no-op.
/// The root cause is that start_transition cloned loaded.data (already the new
/// scene) as old_data, then parsed the same JSON as new_data.
#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod b4_transition_tests {
    use super::*;

    fn make_circle(cx: f32, cy: f32, r: f32, fill_r: f32) -> CircleInstance {
        CircleInstance {
            center: [cx, cy],
            radius: r,
            fill_color: [fill_r, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }
    }

    fn make_rect(x: f32, y: f32, w: f32, h: f32) -> RectInstance {
        RectInstance {
            position: [x, y],
            size: [w, h],
            corner_radius: 0.0,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }
    }

    /// Self-to-self interpolation produces no visible change at any t.
    /// This is the symptom of B4: when old == new, the animation is a no-op.
    #[test]
    fn self_to_self_lerp_circles_produces_no_change() {
        let scene = vec![make_circle(100.0, 200.0, 10.0, 0.5)];
        let mid = lerp_circles(&scene, &scene, 0.5);
        // At t=0.5 of a self-transition, the result is identical to the input.
        assert!(
            (mid[0].center[0] - 100.0).abs() < 1e-6,
            "self-lerp must not move the circle"
        );
    }

    /// When old != new, lerp_circles at t=0.5 MUST produce midpoint values.
    /// This is the correct behavior after B4 is fixed: start_transition
    /// passes different old/new data to the interpolation.
    #[test]
    fn different_scenes_lerp_circles_produces_midpoint() {
        let old = vec![make_circle(0.0, 0.0, 10.0, 0.0)];
        let new = vec![make_circle(200.0, 400.0, 30.0, 1.0)];
        let mid = lerp_circles(&old, &new, 0.5);
        assert!(
            (mid[0].center[0] - 100.0).abs() < 0.01,
            "midpoint x should be 100.0, got {}",
            mid[0].center[0]
        );
        assert!(
            (mid[0].center[1] - 200.0).abs() < 0.01,
            "midpoint y should be 200.0, got {}",
            mid[0].center[1]
        );
        assert!(
            (mid[0].radius - 20.0).abs() < 0.01,
            "midpoint radius should be 20.0, got {}",
            mid[0].radius
        );
        assert!(
            (mid[0].fill_color[0] - 0.5).abs() < 0.01,
            "midpoint fill_color[0] should be 0.5, got {}",
            mid[0].fill_color[0]
        );
    }

    /// Same test for rects: self-interpolation is a no-op.
    #[test]
    fn self_to_self_lerp_rects_produces_no_change() {
        let scene = vec![make_rect(50.0, 60.0, 100.0, 80.0)];
        let mid = lerp_rects(&scene, &scene, 0.5);
        assert!(
            (mid[0].position[0] - 50.0).abs() < 1e-6,
            "self-lerp must not move the rect"
        );
    }

    /// When old != new, lerp_rects at t=0.5 MUST produce midpoint values.
    #[test]
    fn different_scenes_lerp_rects_produces_midpoint() {
        let old = vec![make_rect(0.0, 0.0, 100.0, 50.0)];
        let new = vec![make_rect(200.0, 300.0, 400.0, 250.0)];
        let mid = lerp_rects(&old, &new, 0.5);
        assert!(
            (mid[0].position[0] - 100.0).abs() < 0.01,
            "midpoint x should be 100.0, got {}",
            mid[0].position[0]
        );
        assert!(
            (mid[0].size[0] - 250.0).abs() < 0.01,
            "midpoint width should be 250.0, got {}",
            mid[0].size[0]
        );
    }
}

/// Keyed pairing / object constancy (spec §4.3 / GH #93).
///
/// Every test here fails against the pre-#93 index-zip runtime: with
/// `plan_transition` stubbed to `TransitionPlan::IndexZip`, the reorder test
/// lands each mark on its neighbour's position, the insert/delete tests find
/// no fade, and the fallback tests lose their discriminating power.
#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod keyed_pairing_tests {
    use super::*;
    use ferrum_scene::{
        BlendMode, Color, CoordKind, FillStroke, LayoutScale, MarkBatch, MarkBatchKind, Rect,
        SceneNode,
    };

    fn style() -> FillStroke {
        FillStroke {
            fill: Some(Color::rgba(10, 20, 30, 255)),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    fn circle_node() -> SceneNode {
        SceneNode::Circle {
            cx: 0.0,
            cy: 0.0,
            r: 1.0,
            style: style(),
        }
    }

    /// A circle mark batch of `n` marks, optionally keyed. Geometry lives in
    /// the instance arrays, not the nodes, so the nodes only need to exist in
    /// the right quantity.
    fn circle_batch(n: usize, keys: Option<&[&str]>) -> MarkBatch {
        MarkBatch {
            kind: MarkBatchKind::Point,
            nodes: (0..n).map(|_| circle_node()).collect(),
            data_indices: None,
            tooltips: None,
            hrefs: None,
            descriptions: None,
            keys: keys.map(|k| k.iter().map(|s| s.to_string()).collect()),
            blend: BlendMode::Normal,
            stroke_cap: None,
            stroke_join: None,
            packed_instances: None,
            y_slot: 0,
        }
    }

    /// A single panel carrying one circle batch per `(marks, keys)` spec, in
    /// order — the multi-batch shape `batch_instance_spans` exists for, and
    /// the shape in which one batch's pairing can corrupt another's slots.
    fn panel_with_batches(specs: &[(usize, Option<&[&str]>)]) -> Panel {
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        Panel {
            id: 0,
            plot_area: area,
            clip: area,
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: specs.iter().map(|(n, k)| circle_batch(*n, *k)).collect(),
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }
    }

    /// The single-batch case, by far the most common in these tests.
    fn panel(n: usize, keys: Option<&[&str]>) -> Panel {
        panel_with_batches(&[(n, keys)])
    }

    fn circle_at(x: f32) -> CircleInstance {
        CircleInstance {
            center: [x, 0.0],
            radius: 5.0,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }
    }

    fn circles(xs: &[f32]) -> Vec<CircleInstance> {
        xs.iter().copied().map(circle_at).collect()
    }

    /// A single-panel scene with one keyed rect batch of `n` bars — the other
    /// instance kind the matcher pairs.
    fn rect_panel(n: usize, keys: Option<&[&str]>) -> Panel {
        let mut p = panel(0, keys);
        p.marks[0].kind = MarkBatchKind::Bar;
        p.marks[0].nodes = (0..n)
            .map(|_| SceneNode::Rect {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
                style: style(),
                corner_radius: 0.0,
            })
            .collect();
        p
    }

    fn rects(xs: &[f32]) -> Vec<RectInstance> {
        xs.iter()
            .map(|&x| RectInstance {
                position: [x, 0.0],
                size: [4.0, 10.0],
                corner_radius: 0.0,
                fill_color: [0.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0; 4],
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            })
            .collect()
    }

    /// One circle draw command per `(instance_start, instance_count)` span —
    /// the loader emits exactly one per batch per kind, covering that batch's
    /// contiguous run.
    fn circle_draws(spans: &[(u32, u32)]) -> Vec<DrawCommand> {
        spans
            .iter()
            .map(|&(instance_start, instance_count)| DrawCommand {
                kind: DrawKind::Circle,
                instance_start,
                instance_count,
                additive: false,
                is_mark: true,
                plot_area: Some([0.0, 0.0, 100.0, 100.0]),
                panel_id: 0,
                y_slot: 0,
            })
            .collect()
    }

    /// The one circle draw command a batch of `n` marks produces.
    fn draws(n: usize) -> Vec<DrawCommand> {
        circle_draws(&[(0, n as u32)])
    }

    /// Plan a transition between two scenes.
    fn plan(
        old_panels: &[Panel],
        old_draws: &[DrawCommand],
        new_panels: &[Panel],
        new_draws: &[DrawCommand],
    ) -> TransitionPlan {
        let empty = HashMap::new();
        plan_transition(
            TransitionSide {
                panels: old_panels,
                packed_batch_meta: &empty,
                draw_commands: old_draws,
            },
            TransitionSide {
                panels: new_panels,
                packed_batch_meta: &empty,
                draw_commands: new_draws,
            },
        )
    }

    fn frame(
        plan: &TransitionPlan,
        old: &[CircleInstance],
        new: &[CircleInstance],
        t: f32,
    ) -> TransitionFrame {
        interpolate(
            plan,
            Instances {
                circles: old,
                rects: &[],
            },
            Instances {
                circles: new,
                rects: &[],
            },
            t,
        )
    }

    #[test]
    fn reorder_pairs_by_key_not_index() {
        // Same three marks, reversed in the new scene. Each key must travel to
        // ITS OWN new x, so at the midpoint every mark sits halfway between its
        // own endpoints — a→2->... index-zip would instead drag "a" toward
        // "c"'s slot and land the midpoints on entirely different values.
        let old_panels = vec![panel(3, Some(&["a", "b", "c"]))];
        let new_panels = vec![panel(3, Some(&["c", "b", "a"]))];
        let old = circles(&[0.0, 10.0, 20.0]);
        let new = circles(&[20.0, 10.0, 0.0]);

        let plan = plan(&old_panels, &draws(3), &new_panels, &draws(3));
        let mid = frame(&plan, &old, &new, 0.5);

        // New slot 0 holds key "c" (old index 2, x=20) targeting x=20 → stays.
        assert_eq!(mid.circles[0].center[0], 20.0);
        // New slot 1 holds key "b" (old index 1, x=10) targeting x=10 → stays.
        assert_eq!(mid.circles[1].center[0], 10.0);
        // New slot 2 holds key "a" (old index 0, x=0) targeting x=0 → stays.
        assert_eq!(mid.circles[2].center[0], 0.0);
        assert!(mid.exit_draws.is_empty(), "a pure reorder has no exits");
    }

    #[test]
    fn insert_enters_with_opacity_ramp() {
        // "b" is new: it must be drawn at its FINAL geometry with opacity
        // ramping 0 → target, not interpolated from whatever mark index 1 held.
        let old_panels = vec![panel(2, Some(&["a", "c"]))];
        let new_panels = vec![panel(3, Some(&["a", "b", "c"]))];
        let old = circles(&[0.0, 20.0]);
        let new = circles(&[0.0, 10.0, 20.0]);

        let plan = plan(&old_panels, &draws(2), &new_panels, &draws(3));
        let mid = frame(&plan, &old, &new, 0.5);

        assert_eq!(mid.circles.len(), 3, "output keeps the new scene's layout");
        assert_eq!(mid.circles[1].center[0], 10.0, "enter at final geometry");
        assert!(
            (mid.circles[1].opacity - 0.5).abs() < 1e-6,
            "entering mark fades in: expected 0.5, got {}",
            mid.circles[1].opacity
        );
        assert!(
            (mid.circles[1].stroke_opacity - 0.5).abs() < 1e-6,
            "both opacity channels ramp together"
        );
        // "c" survives at index 2 and does not move.
        assert_eq!(mid.circles[2].center[0], 20.0);
        assert!((mid.circles[2].opacity - 1.0).abs() < 1e-6, "survivor is opaque");
    }

    #[test]
    fn enter_is_fully_opaque_at_t_one() {
        let old_panels = vec![panel(1, Some(&["a"]))];
        let new_panels = vec![panel(2, Some(&["a", "b"]))];
        let old = circles(&[0.0]);
        let new = circles(&[0.0, 10.0]);

        let plan = plan(&old_panels, &draws(1), &new_panels, &draws(2));
        let end = frame(&plan, &old, &new, 1.0);
        assert!((end.circles[1].opacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn delete_exits_with_fade_and_its_own_draw_command() {
        // "b" is gone. It must still be painted at its OLD geometry, fading
        // out, via an appended draw command — index-zip would simply drop the
        // tail instance and pair "c" with "b"'s old position.
        let old_panels = vec![panel(3, Some(&["a", "b", "c"]))];
        let new_panels = vec![panel(2, Some(&["a", "c"]))];
        let old = circles(&[0.0, 10.0, 20.0]);
        let new = circles(&[0.0, 20.0]);

        let plan = plan(&old_panels, &draws(3), &new_panels, &draws(2));
        let mid = frame(&plan, &old, &new, 0.5);

        // "c" pairs with its own old instance (x=20 → x=20), not with "b".
        assert_eq!(mid.circles[1].center[0], 20.0);
        // The exiting "b" is appended past the new scene's own instances.
        assert_eq!(mid.circles.len(), 3);
        assert_eq!(mid.circles[2].center[0], 10.0, "exit holds old geometry");
        assert!(
            (mid.circles[2].opacity - 0.5).abs() < 1e-6,
            "exiting mark fades out: expected 0.5, got {}",
            mid.circles[2].opacity
        );

        assert_eq!(mid.exit_draws.len(), 1);
        let cmd = &mid.exit_draws[0];
        assert_eq!(cmd.instance_start, 2);
        assert_eq!(cmd.instance_count, 1);
        assert_eq!(cmd.kind, DrawKind::Circle);
        assert!(cmd.is_mark, "exits inherit the batch's draw settings");
        assert_eq!(cmd.plot_area, Some([0.0, 0.0, 100.0, 100.0]));
    }

    #[test]
    fn exits_are_dropped_at_t_one() {
        let old_panels = vec![panel(2, Some(&["a", "b"]))];
        let new_panels = vec![panel(1, Some(&["a"]))];
        let old = circles(&[0.0, 10.0]);
        let new = circles(&[0.0]);

        let plan = plan(&old_panels, &draws(2), &new_panels, &draws(1));
        let end = frame(&plan, &old, &new, 1.0);
        assert_eq!(end.circles.len(), 1, "exiting instances are gone at t=1");
        assert!(end.exit_draws.is_empty());
    }

    #[test]
    fn duplicate_key_falls_back_to_index_zip() {
        // A non-injective key column cannot define object constancy, so the
        // WHOLE batch reverts to positional pairing — no partial keying.
        let old_panels = vec![panel(3, Some(&["a", "a", "c"]))];
        let new_panels = vec![panel(3, Some(&["c", "a", "a"]))];
        assert!(matches!(
            plan(&old_panels, &draws(3), &new_panels, &draws(3)),
            TransitionPlan::IndexZip
        ));
    }

    #[test]
    fn duplicate_key_on_the_new_side_alone_falls_back() {
        let old_panels = vec![panel(2, Some(&["a", "b"]))];
        let new_panels = vec![panel(2, Some(&["a", "a"]))];
        assert!(matches!(
            plan(&old_panels, &draws(2), &new_panels, &draws(2)),
            TransitionPlan::IndexZip
        ));
    }

    #[test]
    fn one_sided_keys_fall_back_to_index_zip() {
        let keyed = vec![panel(2, Some(&["a", "b"]))];
        let unkeyed = vec![panel(2, None)];
        assert!(matches!(
            plan(&keyed, &draws(2), &unkeyed, &draws(2)),
            TransitionPlan::IndexZip
        ));
        assert!(matches!(
            plan(&unkeyed, &draws(2), &keyed, &draws(2)),
            TransitionPlan::IndexZip
        ));
    }

    #[test]
    fn misaligned_key_count_falls_back_to_index_zip() {
        // Three marks, two keys: the producer's alignment guard should have
        // caught this, so the matcher refuses to pair rather than mapping
        // marks onto the wrong keys.
        let old_panels = vec![panel(3, Some(&["a", "b"]))];
        let new_panels = vec![panel(3, Some(&["a", "b", "c"]))];
        assert!(matches!(
            plan(&old_panels, &draws(3), &new_panels, &draws(3)),
            TransitionPlan::IndexZip
        ));
    }

    #[test]
    fn unkeyed_scene_is_byte_identical_to_the_flat_lerp() {
        // The byte-identity invariant (spec §7): an unkeyed scene never enters
        // the keyed machinery, and its frame equals the pre-#93 whole-array
        // interpolation exactly.
        let old_panels = vec![panel(3, None)];
        let new_panels = vec![panel(3, None)];
        let old = circles(&[0.0, 10.0, 20.0]);
        let new = circles(&[5.0, 25.0, 45.0]);

        let plan = plan(&old_panels, &draws(3), &new_panels, &draws(3));
        assert!(matches!(plan, TransitionPlan::IndexZip));

        for t in [0.0f32, 0.25, 0.5, 1.0] {
            let got = frame(&plan, &old, &new, t);
            let want = lerp_circles(&old, &new, t);
            assert_eq!(got.circles.len(), want.len());
            for (a, b) in got.circles.iter().zip(want.iter()) {
                assert_eq!(a.center, b.center);
                assert_eq!(a.radius, b.radius);
                assert_eq!(a.opacity, b.opacity);
            }
            assert!(got.exit_draws.is_empty());
        }
    }

    #[test]
    fn keyed_batch_leaves_unkeyed_chrome_interpolating() {
        // Chrome instances (grid/axes/legend) live outside every mark batch and
        // carry no keys; they must keep interpolating index-wise while the
        // keyed batch pairs by key.
        let old_panels = vec![panel(2, Some(&["a", "b"]))];
        let new_panels = vec![panel(2, Some(&["b", "a"]))];
        // Two batch instances followed by one chrome instance.
        let old = circles(&[0.0, 10.0, 100.0]);
        let new = circles(&[10.0, 0.0, 200.0]);

        let plan = plan(&old_panels, &draws(2), &new_panels, &draws(2));
        let mid = frame(&plan, &old, &new, 0.5);
        assert_eq!(mid.circles[0].center[0], 10.0, "key b stays put");
        assert_eq!(mid.circles[1].center[0], 0.0, "key a stays put");
        assert_eq!(mid.circles[2].center[0], 150.0, "chrome lerps index-wise");
    }

    // ── Multi-batch panels: each run owns its own span ───────────────────
    //
    // A keyed plan applies per batch, so an index-zip batch sharing a panel
    // with a keyed one must still resolve entirely within its own span. The
    // failure these pin is a run leaving its unpaired slots to the whole-array
    // seed pass, which interpolates them from whatever sits at that FLAT index
    // in the old array — a different batch's mark, or chrome.

    #[test]
    fn unkeyed_batch_growing_beside_a_keyed_one_lands_at_final_geometry() {
        // batch0 keyed 4 → 2 (two keys exit), batch1 unkeyed 2 → 3.
        // old flat: [b0: 0,10,20,30][b1: 100,101]
        // new flat: [b0: 0,10]      [b1: 200,201,202]
        // batch1's third mark has no old counterpart in ITS OWN span. Leaving
        // it to the seed pass would lerp it from old[4] = 100 — the same old
        // mark already feeding new slot 2 — landing it at 151 instead of 202.
        let old_panels = vec![panel_with_batches(&[(4, Some(&["a", "b", "c", "d"])), (2, None)])];
        let new_panels = vec![panel_with_batches(&[(2, Some(&["a", "b"])), (3, None)])];
        let old = circles(&[0.0, 10.0, 20.0, 30.0, 100.0, 101.0]);
        let new = circles(&[0.0, 10.0, 200.0, 201.0, 202.0]);

        let plan = plan(
            &old_panels,
            &circle_draws(&[(0, 4), (4, 2)]),
            &new_panels,
            &circle_draws(&[(0, 2), (2, 3)]),
        );
        let mid = frame(&plan, &old, &new, 0.5);

        assert_eq!(mid.circles[0].center[0], 0.0, "key a holds its position");
        assert_eq!(mid.circles[1].center[0], 10.0, "key b holds its position");
        assert_eq!(mid.circles[2].center[0], 150.0, "batch1[0] pairs positionally");
        assert_eq!(mid.circles[3].center[0], 151.0, "batch1[1] pairs positionally");
        assert_eq!(
            mid.circles[4].center[0], 202.0,
            "batch1's unpaired tail must appear at its final geometry, not \
             interpolate from another batch's instance"
        );
        assert!(
            (mid.circles[4].opacity - 1.0).abs() < 1e-6,
            "an unkeyed arrival does not fade in — no identity, no enter"
        );
        // The keyed batch's two exits are appended after the new layout.
        assert_eq!(mid.circles.len(), 7);
        assert_eq!(mid.circles[5].center[0], 20.0, "key c exits at old geometry");
        assert_eq!(mid.circles[6].center[0], 30.0, "key d exits at old geometry");
    }

    #[test]
    fn brand_new_batch_appears_instead_of_morphing_from_chrome() {
        // batch1 exists only in the new scene, and the old array's slot at
        // that flat index is a chrome instance at 900. With no run of its own
        // covering the span, the mark would land at 700 (halfway from the
        // chrome) instead of at its final 500.
        let old_panels = vec![panel_with_batches(&[(2, Some(&["a", "b"]))])];
        let new_panels = vec![panel_with_batches(&[(2, Some(&["a", "b"])), (1, None)])];
        // Old flat index 2 belongs to no batch — it is chrome.
        let old = circles(&[0.0, 10.0, 900.0]);
        let new = circles(&[0.0, 10.0, 500.0]);

        let plan = plan(
            &old_panels,
            &circle_draws(&[(0, 2)]),
            &new_panels,
            &circle_draws(&[(0, 2), (2, 1)]),
        );
        let mid = frame(&plan, &old, &new, 0.5);

        assert_eq!(
            mid.circles[2].center[0], 500.0,
            "a batch new to the scene appears at its final geometry"
        );
        assert_eq!(mid.circles.len(), 3, "nothing exits");
    }

    #[test]
    fn unkeyed_batch_shrinking_beside_a_keyed_one_pairs_within_its_span() {
        // The companion arm: batch1 unkeyed 3 → 2 while batch0 keyed grows
        // 1 → 2, so batch1's span MOVES between the two scenes. Every new slot
        // is paired, so this arm was already correct before the fix — it is
        // here to pin that the fix did not disturb it, and that the run reads
        // its own old span rather than its new flat offset.
        // old flat: [b0: 0][b1: 100,101,102]
        // new flat: [b0: 0,50][b1: 200,201]
        let old_panels = vec![panel_with_batches(&[(1, Some(&["a"])), (3, None)])];
        let new_panels = vec![panel_with_batches(&[(2, Some(&["a", "z"])), (2, None)])];
        let old = circles(&[0.0, 100.0, 101.0, 102.0]);
        let new = circles(&[0.0, 50.0, 200.0, 201.0]);

        let plan = plan(
            &old_panels,
            &circle_draws(&[(0, 1), (1, 3)]),
            &new_panels,
            &circle_draws(&[(0, 2), (2, 2)]),
        );
        let mid = frame(&plan, &old, &new, 0.5);

        assert_eq!(mid.circles[0].center[0], 0.0, "key a holds its position");
        assert_eq!(mid.circles[1].center[0], 50.0, "key z enters at final geometry");
        assert!(
            (mid.circles[1].opacity - 0.5).abs() < 1e-6,
            "a KEYED arrival does fade in"
        );
        assert_eq!(
            mid.circles[2].center[0], 150.0,
            "batch1[0] pairs with old batch1[0] = 100, not with old flat index 2"
        );
        assert_eq!(mid.circles[3].center[0], 151.0, "batch1[1] pairs with 101");
        assert_eq!(mid.circles.len(), 4, "the dropped old tail is simply not drawn");
    }

    #[test]
    fn rect_batch_pairs_by_key_and_exits_fade() {
        // The rect arm of the matcher: bars key exactly as points do, into
        // the rect instance array and its own draw command.
        let old_panels = vec![rect_panel(3, Some(&["a", "b", "c"]))];
        let new_panels = vec![rect_panel(2, Some(&["c", "a"]))];
        let old = rects(&[0.0, 10.0, 20.0]);
        let new = rects(&[20.0, 0.0]);
        let rect_draws = |n: usize| {
            let mut cmds = draws(n);
            cmds[0].kind = DrawKind::Rect;
            cmds
        };

        let plan = plan(&old_panels, &rect_draws(3), &new_panels, &rect_draws(2));
        let mid = interpolate(
            &plan,
            Instances {
                circles: &[],
                rects: &old,
            },
            Instances {
                circles: &[],
                rects: &new,
            },
            0.5,
        );

        assert_eq!(mid.rects[0].position[0], 20.0, "key c pairs with its own bar");
        assert_eq!(mid.rects[1].position[0], 0.0, "key a pairs with its own bar");
        // "b" exits: old geometry, half opacity, appended with its own command.
        assert_eq!(mid.rects.len(), 3);
        assert_eq!(mid.rects[2].position[0], 10.0);
        assert!((mid.rects[2].opacity - 0.5).abs() < 1e-6);
        assert_eq!(mid.exit_draws.len(), 1);
        assert_eq!(mid.exit_draws[0].kind, DrawKind::Rect);
        assert!(mid.circles.is_empty(), "a rect batch touches no circles");
    }

    // ── Old-side identity: snapshot vs. JSON re-parse (spec §4.3) ────────
    //
    // A batch above the 1000-mark pack threshold ships with empty `nodes` AND
    // with `keys` cleared into the binary sidecar. Its identity therefore
    // exists only in the loaded, in-memory scene — which is why `loadScene`
    // snapshots the outgoing frame instead of letting `startTransition`
    // re-parse the previous JSON.

    /// Wrap `panel` in a scene graph so it can round-trip through
    /// `load_scene_with_packed` / JSON.
    fn scene_of(panel: Panel) -> ferrum_scene::SceneGraph {
        ferrum_scene::SceneGraph {
            width: 200.0,
            height: 200.0,
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

    /// The scene graph a packer leaves behind for a packed batch: no nodes, no
    /// keys — everything moved into the binary sidecar.
    fn packed_scene() -> ferrum_scene::SceneGraph {
        let mut p = panel(0, None);
        p.marks[0].nodes.clear();
        scene_of(p)
    }

    /// One packed circle batch for panel 0 / batch 0, with a `HAS_KEYS`
    /// section. Mirrors `pack_instances::extract_packed_bytes`'s layout (20-byte
    /// header, instances, then `[len u32][utf8]` per key); the byte format
    /// itself is pinned by `scene_load`'s decoder tests.
    fn packed_bytes_with_keys(instances: &[CircleInstance], keys: &[&str]) -> Vec<u8> {
        // Third and last copy of this flag value, and the only test-local one.
        // The producer owns it (`ferrum-core`'s `pack_instances::HAS_KEYS`) and
        // the reader mirrors it (`scene_load`'s private `HAS_KEYS`); a renumber
        // has to touch all three.
        const HAS_KEYS: u32 = 0x4;
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes()); // panel_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // batch_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // kind = circle
        buf.extend_from_slice(&(instances.len() as u32).to_le_bytes());
        buf.extend_from_slice(&HAS_KEYS.to_le_bytes());
        for inst in instances {
            buf.extend_from_slice(bytemuck::bytes_of(inst));
        }
        for key in keys {
            buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
        }
        buf
    }

    /// The snapshot `loadScene` would hold for a packed scene.
    fn packed_snapshot(instances: &[CircleInstance], keys: &[&str]) -> SceneSnapshot {
        let scene = packed_scene();
        let data = crate::scene_load::load_scene_with_packed(
            &scene,
            &packed_bytes_with_keys(instances, keys),
        );
        SceneSnapshot::from_scene_data(data, scene.panels)
    }

    #[test]
    fn packed_old_side_keys_through_the_in_memory_snapshot() {
        // Both sides above the pack threshold: keys live only in the sidecar,
        // and the snapshot is what carries them into the matcher.
        let old_instances = circles(&[0.0, 10.0, 20.0]);
        let new_instances = circles(&[20.0, 10.0, 0.0]);
        let old = packed_snapshot(&old_instances, &["a", "b", "c"]);
        let new = packed_snapshot(&new_instances, &["c", "b", "a"]);

        let plan = plan_transition(old.side(), new.side());
        assert!(
            matches!(plan, TransitionPlan::Keyed { .. }),
            "a packed old side must key against a packed new side"
        );

        let mid = interpolate(&plan, old.instances(), new.instances(), 0.5);
        // Each key transitions to its own new slot, so nothing moves.
        assert_eq!(mid.circles[0].center[0], 20.0, "key c");
        assert_eq!(mid.circles[1].center[0], 10.0, "key b");
        assert_eq!(mid.circles[2].center[0], 0.0, "key a");
    }

    #[test]
    fn packed_old_side_cannot_key_when_rebuilt_from_json() {
        // The gap the snapshot closes: the same packed scene, re-parsed from
        // its JSON, has no instances and no keys at all, so the run falls back
        // to index-zip. This is why `from_scene_json` is the fallback and not
        // the primary path.
        //
        // Its remaining reach, after `ferrum-anywidget.js`'s `_reload` learned
        // to load into the live renderer: a first render (nothing to snapshot),
        // and a reload the widget declines to serve in place (canvas size or
        // handler config changed, or the load failed) and therefore rebuilds.
        // A steady-state data update no longer lands here.
        let scene_json = serde_json::to_string(&packed_scene()).expect("serialize");
        let old = SceneSnapshot::from_scene_json(&scene_json).expect("parse");
        let new = packed_snapshot(&circles(&[20.0, 10.0, 0.0]), &["c", "b", "a"]);

        assert!(
            old.instances().circles.is_empty(),
            "a packed batch's JSON carries no instances"
        );
        assert!(matches!(
            plan_transition(old.side(), new.side()),
            TransitionPlan::IndexZip
        ));
    }

    #[test]
    fn packed_old_side_keys_against_a_json_node_new_side() {
        // The crossing case: a large (packed) scene shrinking below the pack
        // threshold. Old keys come from the sidecar, new keys from
        // `MarkBatch::keys`, and the two pair.
        let old = packed_snapshot(&circles(&[0.0, 10.0]), &["a", "b"]);
        let new_scene = scene_of(panel(2, Some(&["b", "a"])));
        let new_data = crate::scene_load::load_scene(&new_scene);
        let new = SceneSnapshot::from_scene_data(new_data, new_scene.panels);

        let plan = plan_transition(old.side(), new.side());
        assert!(matches!(plan, TransitionPlan::Keyed { .. }));

        let new_instances = circles(&[10.0, 0.0]);
        let mid = interpolate(
            &plan,
            old.instances(),
            Instances {
                circles: &new_instances,
                rects: &[],
            },
            0.5,
        );
        assert_eq!(mid.circles[0].center[0], 10.0, "key b");
        assert_eq!(mid.circles[1].center[0], 0.0, "key a");
    }

    #[test]
    fn packed_batch_pairs_on_sidecar_keys() {
        // A packed batch carries its keys in `PackedBatchMeta`, not in
        // `MarkBatch`; the matcher must read whichever carrier is populated.
        let mut panels = vec![panel(0, None)];
        panels[0].marks[0].nodes.clear();
        let meta_for = |keys: &[&str]| {
            let mut m = HashMap::new();
            m.insert(
                (0u32, 0u32),
                PackedBatchMeta {
                    data_indices: None,
                    tooltip_bytes: None,
                    keys: Some(keys.iter().map(|s| s.to_string()).collect()),
                    kind: DrawKind::Circle,
                    instance_start: 0,
                    instance_count: keys.len(),
                },
            );
            m
        };
        let old_meta = meta_for(&["a", "b"]);
        let new_meta = meta_for(&["b", "a"]);
        let plan = plan_transition(
            TransitionSide {
                panels: &panels,
                packed_batch_meta: &old_meta,
                draw_commands: &draws(2),
            },
            TransitionSide {
                panels: &panels,
                packed_batch_meta: &new_meta,
                draw_commands: &draws(2),
            },
        );

        let old = circles(&[0.0, 10.0]);
        let new = circles(&[10.0, 0.0]);
        let mid = frame(&plan, &old, &new, 0.5);
        assert_eq!(mid.circles[0].center[0], 10.0);
        assert_eq!(mid.circles[1].center[0], 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ease_in_out_at_boundaries() {
        assert!((ease_in_out_cubic(0.0)).abs() < 1e-6);
        assert!((ease_in_out_cubic(1.0) - 1.0).abs() < 1e-6);
        assert!((ease_in_out_cubic(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_circles_midpoint() {
        let old = vec![CircleInstance {
            center: [0.0, 0.0],
            radius: 10.0,
            fill_color: [0.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];
        let new = vec![CircleInstance {
            center: [100.0, 200.0],
            radius: 20.0,
            fill_color: [1.0, 1.0, 1.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 0.5,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];
        let mid = lerp_circles(&old, &new, 0.5);
        assert!((mid[0].center[0] - 50.0).abs() < 0.01);
        assert!((mid[0].radius - 15.0).abs() < 0.01);
        assert!((mid[0].opacity - 0.75).abs() < 0.01);
    }

    #[test]
    fn lerp_color_interpolates() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [1.0, 1.0, 1.0, 0.0];
        let mid = lerp_color(a, b, 0.5);
        assert!((mid[0] - 0.5).abs() < 0.01);
        assert!((mid[3] - 0.5).abs() < 0.01);
    }
}
