use std::collections::HashMap;

use ferrum_scene::*;
use lyon::tessellation::VertexBuffers;

use crate::tessellate::{self, MeshVertex};

/// Convert a single sRGB channel value (0.0..1.0) to linear light.
///
/// WebGPU with an sRGB surface format automatically applies linear-to-sRGB
/// conversion on output. If we feed sRGB values directly, the gamma curve
/// is applied twice and colors appear washed-out. This function undoes the
/// sRGB encoding so the GPU's automatic conversion produces correct output.
///
/// The alpha channel is NOT converted — alpha is linear in both spaces.
pub(crate) fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert the RGB channels of a `[r, g, b, a]` color array from sRGB to
/// linear. Alpha is left untouched (it is already linear in both spaces).
fn linearize_color_channels(color: &mut [f32; 4]) {
    color[0] = srgb_to_linear(color[0]);
    color[1] = srgb_to_linear(color[1]);
    color[2] = srgb_to_linear(color[2]);
    // color[3] (alpha) stays as-is.
}

/// Convert a scene `Color` + opacity to a linearized `[f32; 4]` suitable for
/// GPU upload. RGB channels are converted from sRGB to linear; the alpha
/// channel combines the color's own alpha with the provided opacity.
///
/// This is the single canonical path for "Color → GPU color". All call sites
/// that previously inlined `srgb_to_linear(c.r as f32 / 255.0)` should use
/// this instead.
pub(crate) fn color_to_linear(c: &Color, opacity: f64) -> [f32; 4] {
    [
        srgb_to_linear(c.r as f32 / 255.0),
        srgb_to_linear(c.g as f32 / 255.0),
        srgb_to_linear(c.b as f32 / 255.0),
        (c.a as f32 / 255.0) * opacity as f32,
    ]
}

/// Stroke-dash palette: maps palette index (0–3) to a `stroke-dasharray`
/// pattern string, shared with the SVG renderer for cross-renderer consistency.
///
/// Index mapping:
/// - 0 → solid (no dash)
/// - 1 → dashed  "6,3"
/// - 2 → dotted  "2,3"
/// - 3 → dash-dot "6,3,2,3"
///
/// Out-of-range float values in instance data are clamped to `[0, 3]` by
/// the shader's `clamp(floor(stroke_dash + 0.5), 0, 3)` idiom.
pub const STROKE_DASH_PALETTE: [&str; 4] = ["", "6,3", "2,3", "6,3,2,3"];

/// GPU instance record for a single circle mark.
///
/// Layout (16 floats = 64 bytes):
///   center(2) + radius(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1)
///   + stroke_opacity(1) + stroke_dash(1) + angle(1)
///
/// `stroke_dash` is stored as an f32 palette index (0.0–3.0) for Pod
/// compatibility; the vertex shader casts it to a uint index with
/// `clamp(floor(dash + 0.5), 0, 3)`.
///
/// `angle` is in screen-space degrees, applied as a rotation around the
/// instance anchor (circle center / rect center).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleInstance {
    pub center: [f32; 2],
    pub radius: f32,
    pub fill_color: [f32; 4],
    pub stroke_color: [f32; 4],
    pub stroke_width: f32,
    pub opacity: f32,
    /// Stroke opacity in [0, 1]. Default: 1.0 (fully opaque).
    pub stroke_opacity: f32,
    /// Palette index as f32 (0.0 = solid, 1.0 = dashed, 2.0 = dotted,
    /// 3.0 = dash-dot). Values outside [0, 3] are clamped. Default: 0.0.
    pub stroke_dash: f32,
    /// Rotation in screen-space degrees around the circle center. Default: 0.0.
    pub angle: f32,
}

/// GPU instance record for a single rect/bar mark.
///
/// Layout (18 floats = 72 bytes):
///   pos(2) + size(2) + corner_r(1) + fill(4) + stroke(4) + stroke_w(1) + opacity(1)
///   + stroke_opacity(1) + stroke_dash(1) + angle(1)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectInstance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub corner_radius: f32,
    pub fill_color: [f32; 4],
    pub stroke_color: [f32; 4],
    pub stroke_width: f32,
    pub opacity: f32,
    /// Stroke opacity in [0, 1]. Default: 1.0 (fully opaque).
    pub stroke_opacity: f32,
    /// Palette index as f32 (0.0 = solid, 1.0 = dashed, 2.0 = dotted,
    /// 3.0 = dash-dot). Values outside [0, 3] are clamped. Default: 0.0.
    pub stroke_dash: f32,
    /// Rotation in screen-space degrees around the rect center. Default: 0.0.
    pub angle: f32,
}

// ── Wire-format compile-time assertions ────────────────────────────────────
//
// The packed binary sidecar produced by `ferrum-core/src/render/pack_instances.rs`
// must be byte-compatible with these structs.  The canonical stride values are:
//   CircleInstance: 16 × f32 = 64 bytes  (CIRCLE_STRIDE in ferrum-core)
//   RectInstance:   18 × f32 = 72 bytes  (RECT_STRIDE in ferrum-core)
// The consumer's stride computation (`std::mem::size_of::<…>()`) is already
// correct and is NOT changed here — these asserts only add enforcement.
// Any layout drift fails the build immediately rather than silently corrupting
// the interactive render.
const _: () = assert!(std::mem::size_of::<CircleInstance>() == 64);
const _: () = assert!(std::mem::size_of::<RectInstance>() == 72);
// X/Y fields sit at byte 0 (center/position are the first fields in each struct).
// Toolchain is Rust ≥1.77 so offset_of! is available in core.
const _: () = assert!(core::mem::offset_of!(CircleInstance, center) == 0);
const _: () = assert!(core::mem::offset_of!(RectInstance, position) == 0);

#[derive(Clone)]
pub struct SceneData {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
    /// Mesh vertices/indices for mark batches (lines, areas, paths,
    /// polygons, polylines). Drawn with the zoom/pan transform.
    pub mesh_buffers: VertexBuffers<MeshVertex, u32>,
    /// Mesh vertices/indices for non-mark elements (grid lines, axis
    /// ticks, legend lines, title decorations, etc.).
    /// Drawn with the identity transform so they stay fixed during
    /// zoom/pan.
    pub static_mesh_buffers: VertexBuffers<MeshVertex, u32>,
    /// Mesh vertices/indices for annotation Line/Path nodes
    /// (from `annotate_hline` / `annotate_vline` etc.).  Drawn with the
    /// identity transform AFTER mark mesh so reference lines appear above
    /// data marks, matching SVG painter order.
    pub annotation_mesh_buffers: VertexBuffers<MeshVertex, u32>,
    pub text_elements: Vec<TextElementData>,
    pub image_quads: Vec<ImageQuad>,
    /// Verbatim SVG fragments from `SceneNode::Raw` nodes (colorbar gradients,
    /// insets, annotation images). Exported to JS for DOM injection into the
    /// text-overlay `<svg>`. Mirrors the `text_elements` export path.
    pub raw_fragments: Vec<RawFragmentData>,
    pub background: Option<[f32; 4]>,
    pub width: f32,
    pub height: f32,
    /// Per-batch metadata from the packed binary sidecar, keyed by
    /// `(panel_idx, batch_idx)`.
    pub packed_batch_meta: HashMap<(u32, u32), PackedBatchMeta>,
    /// Ordered draw commands for circle/rect batches. The render loop
    /// iterates these to select the correct blend pipeline per batch.
    /// Non-mark instances (grid, axes, legend, title) are emitted as
    /// normal-blend commands; mark batches carry their scene-graph
    /// `BlendMode`.
    pub draw_commands: Vec<DrawCommand>,
    /// Per-panel mark-mesh index ranges + plot areas.
    ///
    /// Each entry corresponds to one panel whose mark batches contributed
    /// mesh geometry (lines, areas, paths). `render_frame` uses this list to
    /// draw each panel's mesh slice with a scissor rect clamped to its
    /// `plot_area`, preventing zoomed/panned geometry from bleeding outside
    /// the plot boundary into axis margins or adjacent panels.
    ///
    /// Panels that contributed no mesh geometry (e.g. only packed
    /// circle/rect instances) are absent from this list.
    pub mark_mesh_panels: Vec<MarkMeshPanel>,
    /// Number of panels in the source scene graph. Always `>= 1`.
    pub panel_count: usize,
    /// Per-panel y-scale slot count (secondary-y-axis, GH #52), indexed by
    /// `panel_id`. `1` for every single-y panel; `n` for a panel resolving `n`
    /// independent-y layers. `GpuBuffers` allocates one mark-transform slot per
    /// (panel, slot) pair — `sum(panel_slot_counts)` total — and the render loop
    /// binds each mark draw's `(panel_id, y_slot)` composed affine via
    /// [`transform_slot_index`]. A single-y scene has every count `== 1`, so the
    /// mapping is `index == panel_id` and the allocation is byte-identical to
    /// the former per-panel one.
    pub panel_slot_counts: Vec<usize>,
}

#[derive(Clone)]
pub struct TextElementData {
    pub x: f64,
    pub y: f64,
    pub content: String,
    pub style: TextStyle,
}

/// A verbatim SVG fragment collected from `SceneNode::Raw`.
///
/// Exported to JS as a JSON array alongside text elements.  The JS overlay
/// injects each fragment into the existing text-overlay `<svg>` — chrome
/// fragments in a fixed `<g>` (stays put during pan/zoom), data-anchored
/// fragments in a transform-tracking `<g>`.
#[derive(Clone)]
pub struct RawFragmentData {
    /// The verbatim SVG markup to inject.
    pub svg: String,
    /// `"chrome"` (fixed) or `"data"` (tracks pan/zoom transform).
    pub anchor: String,
}

#[derive(Clone)]
pub struct ImageQuad {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub data: Vec<u8>,
    pub img_width: u32,
    pub img_height: u32,
}

/// Per-batch metadata extracted from the packed binary sidecar.
///
/// Stored in `SceneData::packed_batch_meta` keyed by `(panel_idx, batch_idx)`.
/// Enables lazy tooltip decoding: the raw bytes are kept until `getTooltip` is
/// called, avoiding upfront string-table parsing for every batch.
#[derive(Clone, Debug)]
pub struct PackedBatchMeta {
    pub data_indices: Option<Vec<u32>>,
    pub tooltip_bytes: Option<Vec<u8>>,
    /// Which GPU instance buffer this batch's instances live in. Decoded once
    /// from the packed `kind: u32` header in [`unpack_binary_instances`] via
    /// [`DrawKind::try_from`], so all downstream sites match on the typed enum
    /// rather than re-interpreting the magic `0`/`1` values.
    pub kind: DrawKind,
    pub instance_start: usize,
    pub instance_count: usize,
}

/// Which GPU instance buffer a draw command targets.
///
/// This is the single typed representation of the circle-vs-rect distinction.
/// The packed binary sidecar encodes it as a `u32` (`0` = circle, `1` = rect);
/// [`DrawKind::try_from`] decodes that wire value once at unpack time so no
/// downstream site re-interprets the raw magic numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawKind {
    Circle,
    Rect,
}

/// An unrecognized packed `kind` discriminant.
///
/// The packed sidecar uses `0` for circles and `1` for rects; any other value
/// is a malformed/unsupported batch header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownDrawKind(pub u32);

impl TryFrom<u32> for DrawKind {
    type Error = UnknownDrawKind;

    /// Decode the packed `kind: u32` header. `0` → [`DrawKind::Circle`],
    /// `1` → [`DrawKind::Rect`]; any other value is rejected.
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DrawKind::Circle),
            1 => Ok(DrawKind::Rect),
            other => Err(UnknownDrawKind(other)),
        }
    }
}

/// A single instanced draw call for circle or rect batches.
///
/// `render_frame` iterates these in order to select the correct pipeline
/// (normal vs. additive blend) and draw the correct instance slice. The
/// ordering preserves the painter's algorithm: batches appear in panel order,
/// then batch order within each panel, matching the scene-graph walk in
/// `load_scene_with_packed`.
#[derive(Clone, Debug)]
pub struct DrawCommand {
    pub kind: DrawKind,
    /// First instance index in the corresponding flat array
    /// (`circle_instances` or `rect_instances`).
    pub instance_start: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// When `true`, the additive-blend pipeline is used instead of the
    /// normal alpha-blend pipeline.
    pub additive: bool,
    /// When `true`, the zoom/pan affine transform is applied to this
    /// command's instances. When `false` (axes, gridlines, legend, title,
    /// etc.), the identity transform is used so these elements stay fixed
    /// during zoom.
    pub is_mark: bool,
    /// For mark draw commands (`is_mark == true`), the panel's plot area
    /// `[x, y, w, h]` in canvas pixels. The render loop uses this to
    /// restrict the GPU clip region so zoomed marks do not bleed into axis
    /// margins. `None` for non-mark commands (axes, grid, legend, title).
    pub plot_area: Option<[f32; 4]>,
    /// For mark draw commands (`is_mark == true`), the index of the panel that
    /// owns these instances. The render loop binds that panel's own affine
    /// transform, so a non-uniform domain-rescale on one panel does not shear
    /// or translate sibling panels' marks. Meaningless (always 0) for non-mark
    /// commands, which always draw with the identity transform.
    pub panel_id: usize,
    /// Y-scale slot these instances map through (secondary-y-axis, GH #52),
    /// copied from the owning `MarkBatch::y_slot`. `0` = the primary/left-axis
    /// scale — the byte-stable default for every single-y chart. Lets the render
    /// loop compose this layer's per-slot rescale affine with the panel affine.
    /// Meaningless for non-mark commands (always 0).
    pub y_slot: usize,
}

/// The number of distinct y-scale slots a panel's marks map through
/// (secondary-y-axis, GH #52): the length of the per-slot y-domain list, or `1`
/// for every single-y panel (empty `y_domains` — the byte-stable default).
pub fn panel_slot_count(coord: &ferrum_scene::CoordKind) -> usize {
    match coord {
        ferrum_scene::CoordKind::Cartesian { y_domains, .. } => y_domains.len().max(1),
        _ => 1,
    }
}

/// Flat transform-slot index for one `(panel, y_slot)` pair, given each panel's
/// slot count. Panels' slots are laid out consecutively: slot `s` of panel `p`
/// sits at `sum(counts[..p]) + min(s, counts[p] - 1)`. A single-y scene (every
/// count `== 1`) yields `index == panel`, so the transform-slot vector matches
/// the former per-panel one exactly — the byte-stability anchor. An out-of-range
/// `y_slot` clamps to the panel's last slot rather than indexing past it.
pub fn transform_slot_index(panel_slot_counts: &[usize], panel: usize, y_slot: usize) -> usize {
    let base: usize = panel_slot_counts.iter().take(panel).sum();
    let count = panel_slot_counts.get(panel).copied().unwrap_or(1).max(1);
    base + y_slot.min(count - 1)
}

/// Total number of transform slots for a scene = the sum of per-panel slot
/// counts (always `>= 1`). `GpuBuffers` allocates this many mark-transform
/// uniform buffers.
pub fn total_transform_slots(panel_slot_counts: &[usize]) -> usize {
    panel_slot_counts.iter().sum::<usize>().max(1)
}

/// The contiguous transform-slot range one panel owns (secondary-y-axis,
/// GH #52): `[base, base + count)` where `base = sum(counts[..panel])` and
/// `count = counts[panel]` (`1` when `panel` is out of range, mirroring
/// [`transform_slot_index`]'s clamp-safe default). A single-y panel's range
/// is exactly `panel..panel + 1`, matching the flat per-panel index.
///
/// Used to reset every slot a panel owns together when its zoom/pan
/// transform resets (`WasmRenderer::set_transform`), so a per-layer
/// domainParam/brush rescale parked in `slot_rescales` does not survive a
/// view reset for that panel.
pub fn panel_slot_range(panel_slot_counts: &[usize], panel: usize) -> std::ops::Range<usize> {
    let base: usize = panel_slot_counts.iter().take(panel).sum();
    let count = panel_slot_counts.get(panel).copied().unwrap_or(1).max(1);
    base..(base + count)
}

/// Per-(panel, y-slot) mark-mesh draw range.
///
/// Captures the contiguous slice of the mark-mesh index buffer contributed by
/// one mesh-bearing mark batch, together with the owning panel's plot area and
/// the batch's y-scale slot. `render_frame` iterates this list to scissor each
/// slice to its panel's plot area (preventing zoomed/panned geometry from
/// bleeding into axis margins or adjacent panels) and to bind that (panel, slot)
/// pair's affine. Single-y charts record one slot-0 entry per mesh batch, so
/// the rendered geometry is byte-identical to the pre-#52 per-panel recording.
#[derive(Clone, Debug)]
pub struct MarkMeshPanel {
    /// First index in the flat mark-mesh index buffer for this slice.
    pub index_start: u32,
    /// Number of indices (triangle soup) belonging to this slice.
    pub index_count: u32,
    /// Plot area `[x, y, w, h]` in canvas pixels.
    pub plot_area: [f32; 4],
    /// Index of the panel that owns this mesh slice. The render loop binds
    /// this panel's own affine transform so a non-uniform domain-rescale on a
    /// sibling panel does not shear or translate this panel's mesh.
    pub panel_id: usize,
    /// Y-scale slot this mesh slice maps through (secondary-y-axis, GH #52),
    /// copied from the owning `MarkBatch::y_slot`. `0` = the primary/left-axis
    /// scale — the byte-stable default for every single-y chart.
    pub y_slot: usize,
}

/// Accumulator for one full scene-load pass.
///
/// Replaces the 8 loose `let mut` variables previously threaded through every
/// `collect_nodes` / `emit_draw_commands` call. Callers use the three surface
/// methods (`collect_static` / `collect_mark` / `collect_annotation`) which
/// handle mesh routing and draw-command emission internally.
pub struct SceneCollector {
    pub circles: Vec<CircleInstance>,
    pub rects: Vec<RectInstance>,
    /// Mark mesh: lines, areas, paths, polygons, polylines from mark batches.
    /// Drawn with the zoom/pan transform.
    pub mesh: VertexBuffers<MeshVertex, u32>,
    /// Static mesh: grid lines, axis ticks, legend lines, title decorations,
    /// etc. Drawn with the identity transform so they stay fixed during
    /// zoom/pan.
    pub static_mesh: VertexBuffers<MeshVertex, u32>,
    /// Annotation mesh: Line/Path nodes from `panel.annotations`
    /// (e.g. `annotate_hline` / `annotate_vline`). Drawn with the identity
    /// transform AFTER mark mesh so reference lines appear above data marks.
    pub annotation_mesh: VertexBuffers<MeshVertex, u32>,
    pub texts: Vec<TextElementData>,
    pub images: Vec<ImageQuad>,
    /// Verbatim SVG fragments from `SceneNode::Raw` nodes.
    /// Collected alongside text elements and exported to JS.
    pub raws: Vec<RawFragmentData>,
    pub draw_commands: Vec<DrawCommand>,
    /// Per-panel mark-mesh index ranges. Each entry records the contiguous
    /// slice of the mark-mesh index buffer contributed by one panel and that
    /// panel's plot area, so `render_frame` can scissor each panel's mesh
    /// draw independently.
    pub mark_mesh_panels: Vec<MarkMeshPanel>,
    /// Snapshot of `circles.len()` at the last `emit` call.
    prev_c: usize,
    /// Snapshot of `rects.len()` at the last `emit` call.
    prev_r: usize,
}

impl Default for SceneCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCollector {
    pub fn new() -> Self {
        Self {
            circles: Vec::new(),
            rects: Vec::new(),
            mesh: VertexBuffers::new(),
            static_mesh: VertexBuffers::new(),
            annotation_mesh: VertexBuffers::new(),
            texts: Vec::new(),
            images: Vec::new(),
            raws: Vec::new(),
            draw_commands: Vec::new(),
            mark_mesh_panels: Vec::new(),
            prev_c: 0,
            prev_r: 0,
        }
    }

    /// Collect `nodes` into the static mesh (non-mark elements: grid, axes,
    /// annotations, legend, title, decorations) and immediately emit draw
    /// commands for any new circle/rect instances.
    pub fn collect_static(
        &mut self,
        nodes: &[SceneNode],
        batch_cap: Option<StrokeCap>,
        batch_join: Option<StrokeJoin>,
    ) {
        collect_nodes(
            nodes,
            &mut self.circles,
            &mut self.rects,
            &mut self.static_mesh,
            &mut self.texts,
            &mut self.images,
            &mut self.raws,
            batch_cap,
            batch_join,
        );
        self.emit(false, false, None, 0, 0);
    }

    /// Collect `nodes` into the mark mesh (mark batches: lines, areas, paths,
    /// polygons, polylines) and immediately emit draw commands for any new
    /// circle/rect instances with the given blend mode, plot area, owning panel
    /// index, and y-scale slot.
    ///
    /// The trailing parameters (`batch_cap`/`batch_join`/`y_slot`) are per-batch
    /// attributes forwarded straight from the source `MarkBatch`; grouping them
    /// would only add an intermediate struct without reducing the fan-out at the
    /// single call site, so the arg count is allowed here (matching
    /// `render::upload_transform_and_render`).
    #[allow(clippy::too_many_arguments)]
    pub fn collect_mark(
        &mut self,
        nodes: &[SceneNode],
        additive: bool,
        plot_area: Option<[f32; 4]>,
        panel_id: usize,
        batch_cap: Option<StrokeCap>,
        batch_join: Option<StrokeJoin>,
        y_slot: usize,
    ) {
        collect_nodes(
            nodes,
            &mut self.circles,
            &mut self.rects,
            &mut self.mesh,
            &mut self.texts,
            &mut self.images,
            &mut self.raws,
            batch_cap,
            batch_join,
        );
        self.emit(additive, true, plot_area, panel_id, y_slot);
    }

    /// Collect `nodes` into the annotation mesh.
    ///
    /// Annotation Line/Path nodes (from `annotate_hline` / `annotate_vline`
    /// etc.) are stored separately so they can be drawn AFTER mark mesh in
    /// `render_frame`, giving them the correct z-order above data marks.
    /// Circle/Rect annotation nodes are still emitted as draw commands via
    /// `emit` (they already appear at the correct z-order because they are
    /// emitted after mark batches).
    pub fn collect_annotation(
        &mut self,
        nodes: &[SceneNode],
        batch_cap: Option<StrokeCap>,
        batch_join: Option<StrokeJoin>,
    ) {
        collect_nodes(
            nodes,
            &mut self.circles,
            &mut self.rects,
            &mut self.annotation_mesh,
            &mut self.texts,
            &mut self.images,
            &mut self.raws,
            batch_cap,
            batch_join,
        );
        // Emit draw commands for any Circle/Rect annotation nodes so they
        // appear at the correct z-order (after mark batches).
        self.emit(false, false, None, 0, 0);
    }

    /// Emit draw commands for any circles/rects added since the last snapshot.
    ///
    /// `panel_id`/`y_slot` are only meaningful for mark commands
    /// (`is_mark == true`); the render loop composes that (panel, slot) pair's
    /// affine. Non-mark commands always draw with the identity transform, so
    /// callers pass `0` for both.
    fn emit(
        &mut self,
        additive: bool,
        is_mark: bool,
        plot_area: Option<[f32; 4]>,
        panel_id: usize,
        y_slot: usize,
    ) {
        let new_c = self.circles.len();
        if new_c > self.prev_c {
            self.draw_commands.push(DrawCommand {
                kind: DrawKind::Circle,
                instance_start: self.prev_c as u32,
                instance_count: (new_c - self.prev_c) as u32,
                additive,
                is_mark,
                plot_area,
                panel_id,
                y_slot,
            });
        }
        self.prev_c = new_c;

        let new_r = self.rects.len();
        if new_r > self.prev_r {
            self.draw_commands.push(DrawCommand {
                kind: DrawKind::Rect,
                instance_start: self.prev_r as u32,
                instance_count: (new_r - self.prev_r) as u32,
                additive,
                is_mark,
                plot_area,
                panel_id,
                y_slot,
            });
        }
        self.prev_r = new_r;
    }

    /// Record the mark-mesh index range for one panel.
    ///
    /// Call this once per panel in the scene-graph walk: pass the mesh index
    /// count *before* collecting any mark batches for the panel as
    /// `index_start_before`, and the count *after* all batches have been
    /// collected as `index_end_after`. If the panel contributed no mesh
    /// geometry (e.g. a panel with only packed circle/rect batches), no entry
    /// is recorded.
    pub fn record_mark_mesh_panel(
        &mut self,
        index_start_before: u32,
        index_end_after: u32,
        plot_area: [f32; 4],
        panel_id: usize,
        y_slot: usize,
    ) {
        let index_count = index_end_after - index_start_before;
        if index_count > 0 {
            self.mark_mesh_panels.push(MarkMeshPanel {
                index_start: index_start_before,
                index_count,
                plot_area,
                panel_id,
                y_slot,
            });
        }
    }

    /// Snapshot the current circle and rect counts so a subsequent `emit` only
    /// covers instances appended after this point. Used when instances are
    /// pre-populated from a binary sidecar (packed batches) rather than from
    /// `collect_nodes`.
    pub fn snapshot(&mut self) {
        self.prev_c = self.circles.len();
        self.prev_r = self.rects.len();
    }
}

// ── `Panel::layout_scale` application (ratio-fitted cells / W5 foundation) ──
//
// `walk_svg` applies this same field as a `<g transform="translate(tx,ty)
// scale(sx,sy)">` wrapper around a panel's emitted content. The WASM GPU
// pipeline has no group-transform equivalent for baked mesh vertices and
// instanced circles/rects, so here the transform is baked directly into each
// panel's geometry (node coordinates before tessellation, plus the packed
// instance sidecar) at scene-load time — before any zoom/pan state exists.
// At identity (every flat/faceted panel today) every function below is a
// cheap no-op check, so nothing changes for existing scenes.
//
// Direction-dependent extents (rect/image width & height) scale
// independently by `sx`/`sy` — an exact match for the SVG behavior, since
// width tracks the x-axis and height the y-axis. Direction-*independent*
// scalars that SVG's true 2D transform would otherwise skew into an ellipse
// or a stretched stroke (circle/arc radius, rect corner radius, stroke
// width, dash-pattern lengths, font size) scale by the geometric mean
// `sqrt(sx * sy)`: exact when `sx == sy` (the uniform case), a documented
// approximation otherwise.
// Rotation angles are left unchanged — representing a rotated shape under a
// non-uniform scale requires a shear decomposition this schema does not
// carry, a narrow gap in the same spirit as `SceneNode::Raw`'s existing W4
// baked-coordinate limitation.

/// The scale factor applied to direction-independent scalar magnitudes
/// (radius, stroke width, font size) — the geometric mean of `sx`/`sy`.
fn scalar_scale_factor(ls: &LayoutScale) -> f64 {
    (ls.sx * ls.sy).abs().sqrt()
}

fn transform_fill_stroke(style: &FillStroke, mag: f64) -> FillStroke {
    let mut style = style.clone();
    style.stroke_width *= mag;
    style.stroke_dash = style
        .stroke_dash
        .map(|d| d.iter().map(|v| v * mag).collect());
    style
}

fn transform_stroke_style(style: &StrokeStyle, mag: f64) -> StrokeStyle {
    let mut style = style.clone();
    style.width *= mag;
    style.dash = style.dash.map(|d| d.iter().map(|v| v * mag).collect());
    style
}

fn transform_text_style(style: &TextStyle, mag: f64) -> TextStyle {
    let mut style = style.clone();
    style.font_size *= mag;
    style
}

fn transform_path_cmd(cmd: &PathCmd, ls: &LayoutScale) -> PathCmd {
    match *cmd {
        PathCmd::MoveTo { x, y } => {
            let (x, y) = ls.apply(x, y);
            PathCmd::MoveTo { x, y }
        }
        PathCmd::LineTo { x, y } => {
            let (x, y) = ls.apply(x, y);
            PathCmd::LineTo { x, y }
        }
        PathCmd::QuadTo { cx, cy, x, y } => {
            let (cx, cy) = ls.apply(cx, cy);
            let (x, y) = ls.apply(x, y);
            PathCmd::QuadTo { cx, cy, x, y }
        }
        PathCmd::CubicTo {
            c1x,
            c1y,
            c2x,
            c2y,
            x,
            y,
        } => {
            let (c1x, c1y) = ls.apply(c1x, c1y);
            let (c2x, c2y) = ls.apply(c2x, c2y);
            let (x, y) = ls.apply(x, y);
            PathCmd::CubicTo {
                c1x,
                c1y,
                c2x,
                c2y,
                x,
                y,
            }
        }
        PathCmd::HLineTo { x } => PathCmd::HLineTo {
            x: ls.sx * x + ls.tx,
        },
        PathCmd::VLineTo { y } => PathCmd::VLineTo {
            y: ls.sy * y + ls.ty,
        },
        PathCmd::ArcTo {
            rx,
            ry,
            rotation,
            large_arc,
            sweep,
            x,
            y,
        } => {
            let (x, y) = ls.apply(x, y);
            PathCmd::ArcTo {
                rx: rx * ls.sx,
                ry: ry * ls.sy,
                rotation,
                large_arc,
                sweep,
                x,
                y,
            }
        }
        PathCmd::Close => PathCmd::Close,
    }
}

/// Apply `ls` to every point-like coordinate (and, per the module doc above,
/// direction-independent scalar) in a single [`SceneNode`].
fn transform_node(node: &SceneNode, ls: &LayoutScale) -> SceneNode {
    let mag = scalar_scale_factor(ls);
    match node {
        SceneNode::Rect {
            x,
            y,
            w,
            h,
            style,
            corner_radius,
        } => {
            let (x, y) = ls.apply(*x, *y);
            SceneNode::Rect {
                x,
                y,
                w: w * ls.sx,
                h: h * ls.sy,
                style: transform_fill_stroke(style, mag),
                corner_radius: corner_radius * mag,
            }
        }
        SceneNode::Circle { cx, cy, r, style } => {
            let (cx, cy) = ls.apply(*cx, *cy);
            SceneNode::Circle {
                cx,
                cy,
                r: r * mag,
                style: transform_fill_stroke(style, mag),
            }
        }
        SceneNode::Line {
            x1,
            y1,
            x2,
            y2,
            style,
        } => {
            let (x1, y1) = ls.apply(*x1, *y1);
            let (x2, y2) = ls.apply(*x2, *y2);
            SceneNode::Line {
                x1,
                y1,
                x2,
                y2,
                style: transform_stroke_style(style, mag),
            }
        }
        SceneNode::Path {
            commands,
            style,
            closed,
        } => SceneNode::Path {
            commands: commands.iter().map(|c| transform_path_cmd(c, ls)).collect(),
            style: transform_fill_stroke(style, mag),
            closed: *closed,
        },
        SceneNode::Text {
            x,
            y,
            content,
            style,
        } => {
            let (x, y) = ls.apply(*x, *y);
            SceneNode::Text {
                x,
                y,
                content: content.clone(),
                style: transform_text_style(style, mag),
            }
        }
        SceneNode::Image { x, y, w, h, data } => {
            let (x, y) = ls.apply(*x, *y);
            SceneNode::Image {
                x,
                y,
                w: w * ls.sx,
                h: h * ls.sy,
                data: data.clone(),
            }
        }
        SceneNode::Polygon { rings, style } => SceneNode::Polygon {
            rings: rings
                .iter()
                .map(|ring| {
                    ring.iter()
                        .map(|[x, y]| {
                            let (x, y) = ls.apply(*x, *y);
                            [x, y]
                        })
                        .collect()
                })
                .collect(),
            style: transform_fill_stroke(style, mag),
        },
        SceneNode::Polyline { points, style } => SceneNode::Polyline {
            points: points.iter().map(|(x, y)| ls.apply(*x, *y)).collect(),
            style: transform_stroke_style(style, mag),
        },
        SceneNode::Group { attrs, children } => SceneNode::Group {
            attrs: attrs.clone(),
            children: children.iter().map(|c| transform_node(c, ls)).collect(),
        },
        // `Raw` fragments bake absolute coordinates into an opaque SVG
        // string; rewriting them is the existing W4 gap (documented in
        // CLAUDE.md), out of scope here — pass through unchanged.
        SceneNode::Raw { .. } => node.clone(),
    }
}

/// Apply `ls` to every node in `nodes`. Returns a fresh `Vec` even at
/// identity (callers only invoke this behind an `is_identity()` guard).
fn transform_nodes(nodes: &[SceneNode], ls: &LayoutScale) -> Vec<SceneNode> {
    nodes.iter().map(|n| transform_node(n, ls)).collect()
}

/// Borrow `nodes` unchanged at identity (the byte/pixel-stability anchor for
/// every flat and faceted panel today — no allocation on that hot path),
/// or return a freshly transformed owned copy otherwise.
fn maybe_transform_nodes<'a>(
    nodes: &'a [SceneNode],
    ls: &LayoutScale,
) -> std::borrow::Cow<'a, [SceneNode]> {
    if ls.is_identity() {
        std::borrow::Cow::Borrowed(nodes)
    } else {
        std::borrow::Cow::Owned(transform_nodes(nodes, ls))
    }
}

/// Apply `ls` to a panel's `plot_area`/`clip` rect (used for the WASM
/// scissor rect and packed-batch metadata), matching the node-level
/// transform above.
fn transform_rect(rect: &ferrum_scene::Rect, ls: &LayoutScale) -> ferrum_scene::Rect {
    let (x, y) = ls.apply(rect.x, rect.y);
    ferrum_scene::Rect {
        x,
        y,
        w: rect.w * ls.sx,
        h: rect.h * ls.sy,
    }
}

/// Bake each panel's `layout_scale` into its packed circle/rect instances.
///
/// Packed instances bypass `collect_nodes` (they are pre-populated from the
/// binary sidecar before the scene-graph walk begins), so they need their
/// own transform pass keyed by `batch_meta`'s `(panel_idx, batch_idx) ->
/// instance range` mapping.
fn apply_layout_scale_to_packed_instances(
    scene: &SceneGraph,
    circles: &mut [CircleInstance],
    rects: &mut [RectInstance],
    batch_meta: &HashMap<(u32, u32), PackedBatchMeta>,
) {
    for (&(panel_idx, _batch_idx), meta) in batch_meta.iter() {
        let Some(panel) = scene.panels.get(panel_idx as usize) else {
            continue;
        };
        let ls = &panel.layout_scale;
        if ls.is_identity() {
            continue;
        }
        let mag = scalar_scale_factor(ls);
        let range = meta.instance_start..(meta.instance_start + meta.instance_count);
        match meta.kind {
            DrawKind::Circle => {
                if let Some(slice) = circles.get_mut(range) {
                    for c in slice {
                        let (x, y) = ls.apply(c.center[0] as f64, c.center[1] as f64);
                        c.center = [x as f32, y as f32];
                        c.radius = (c.radius as f64 * mag) as f32;
                        c.stroke_width = (c.stroke_width as f64 * mag) as f32;
                    }
                }
            }
            DrawKind::Rect => {
                if let Some(slice) = rects.get_mut(range) {
                    for r in slice {
                        let (x, y) = ls.apply(r.position[0] as f64, r.position[1] as f64);
                        r.position = [x as f32, y as f32];
                        r.size = [
                            (r.size[0] as f64 * ls.sx) as f32,
                            (r.size[1] as f64 * ls.sy) as f32,
                        ];
                        r.corner_radius = (r.corner_radius as f64 * mag) as f32;
                        r.stroke_width = (r.stroke_width as f64 * mag) as f32;
                    }
                }
            }
        }
    }
}

// ── Interaction-geometry single source of truth (D4a amendment addendum) ───
//
// This scene loader bakes each panel's `layout_scale` into GPU mesh vertices
// and packed instances (the block above) at load time, so the RENDERED frame
// is always at final on-screen coordinates. But `hit_test.rs`, `lib.rs`
// (brush/crossfilter), `spatial_index.rs`, and
// `render.rs::upload_transform_and_render` historically read `scene.panels`
// directly — the RAW, un-baked geometry the core emits for ratio-fitted
// panels (native coordinates + a non-identity `layout_scale`, per
// `composite_render.rs`'s placement contract). For every panel today
// (identity `layout_scale`) raw and baked are numerically the same, so this
// was invisible; a non-identity `layout_scale` panel would silently hit-test,
// brush, and scissor against the WRONG (native, not on-screen) rectangle.
//
// `bake_panels` is the single place that produces the corrected geometry:
// every one of the four consumers above now takes a `&[Panel]` that is
// EITHER `&scene.panels` (identity case, unchanged) OR the output of this
// function — never raw panels directly when a non-identity `layout_scale`
// might be present. `WasmRenderer::load_scene` (`lib.rs`) computes this once
// per scene load and stores it on `LoadedScene`, so no consumer re-derives it.
//
// No double-apply: this only rewrites `plot_area`/`clip`/mark-batch `nodes`.
// Packed batches ship with already-EMPTY `nodes` (packed instance bytes are
// cleared server-side and baked separately by
// `apply_layout_scale_to_packed_instances` above), so transforming an empty
// node list here is a no-op — packed geometry is baked exactly once, not
// twice.
/// Bake every panel's `layout_scale` into its `plot_area`, `clip`, and
/// (non-packed) mark-batch node coordinates.
///
/// At identity `layout_scale` (every flat/faceted panel today, and every
/// composite panel placed by pure translation — see `composite_render.rs`'s
/// `place_panel`) this is a plain `Panel::clone()`, byte/pixel-identical to
/// reading `scene.panels` directly. Only a ratio-fitted panel (non-identity
/// `layout_scale`) is actually rewritten.
pub fn bake_panels(scene: &SceneGraph) -> Vec<Panel> {
    scene
        .panels
        .iter()
        .map(|panel| {
            let ls = &panel.layout_scale;
            if ls.is_identity() {
                return panel.clone();
            }
            let mut baked = panel.clone();
            baked.plot_area = transform_rect(&panel.plot_area, ls);
            baked.clip = transform_rect(&panel.clip, ls);
            for batch in &mut baked.marks {
                batch.nodes = transform_nodes(&batch.nodes, ls);
            }
            baked
        })
        .collect()
}

pub fn load_scene(scene: &SceneGraph) -> SceneData {
    load_scene_with_packed(scene, &[])
}

pub fn load_scene_with_packed(scene: &SceneGraph, packed_data: &[u8]) -> SceneData {
    let mut collector = SceneCollector::new();
    let mut batch_meta = HashMap::new();

    // Unpack binary instance data (passed as raw bytes, not base64).
    // Draw commands for packed batches are emitted in the scene-graph walk
    // below, where the MarkBatch.blend mode is available.
    unpack_binary_instances(
        packed_data,
        &mut collector.circles,
        &mut collector.rects,
        &mut batch_meta,
    );
    // Bake each panel's `layout_scale` into its packed instances (a no-op
    // scan at identity — every panel today).
    apply_layout_scale_to_packed_instances(
        scene,
        &mut collector.circles,
        &mut collector.rects,
        &batch_meta,
    );
    // Sync snapshot counters after pre-populating from packed data.
    collector.snapshot();

    let background = scene.background.as_ref().map(|c| color_to_linear(c, 1.0));

    // Title: non-mark → static mesh
    collector.collect_static(&scene.title, None, None);

    for (panel_idx, panel) in scene.panels.iter().enumerate() {
        // Ratio-fitted cells (JointChart/ClusterMap marginals) carry a
        // non-identity `layout_scale`. `walk_svg` applies it as a `<g
        // transform>` wrapper; the GPU pipeline has no such wrapper for
        // baked mesh/instance geometry, so it is baked into every node,
        // rect, and packed instance for this panel instead (see the
        // "`Panel::layout_scale` application" doc block above). At identity
        // (every panel today) `maybe_transform_nodes`/`transform_rect` are
        // no-ops, so nothing changes for existing scenes.
        let ls = &panel.layout_scale;

        // Grid: non-mark → static mesh. Snap Line nodes to pixel centers to
        // avoid sub-pixel aliasing in the GPU rasterizer (the SVG renderer
        // handles this natively; WASM needs explicit snapping). Snapping
        // runs on the already layout-scaled coordinates, since it must
        // operate in final scene-pixel space.
        let scaled_grid = maybe_transform_nodes(&panel.grid, ls);
        let snapped_grid: Vec<SceneNode> = scaled_grid
            .iter()
            .map(|node| {
                if let SceneNode::Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    style,
                } = node
                {
                    let (sx1, sy1, sx2, sy2) = if (x1 - x2).abs() < 0.5 {
                        // Vertical line: snap x to pixel center (round + 0.5)
                        let snapped_x = x1.round() + 0.5;
                        (snapped_x, *y1, snapped_x, *y2)
                    } else if (y1 - y2).abs() < 0.5 {
                        // Horizontal line: snap y to pixel center (round + 0.5)
                        let snapped_y = y1.round() + 0.5;
                        (*x1, snapped_y, *x2, snapped_y)
                    } else {
                        // Diagonal line: pass through unchanged
                        (*x1, *y1, *x2, *y2)
                    };
                    SceneNode::Line {
                        x1: sx1,
                        y1: sy1,
                        x2: sx2,
                        y2: sy2,
                        style: style.clone(),
                    }
                } else {
                    node.clone()
                }
            })
            .collect();
        collector.collect_static(&snapped_grid, None, None);

        let effective_plot_area = if ls.is_identity() {
            panel.plot_area
        } else {
            transform_rect(&panel.plot_area, ls)
        };
        let panel_plot_area_arr = [
            effective_plot_area.x as f32,
            effective_plot_area.y as f32,
            effective_plot_area.w as f32,
            effective_plot_area.h as f32,
        ];
        let panel_plot_area = Some(panel_plot_area_arr);

        for (batch_idx, batch) in panel.marks.iter().enumerate() {
            let additive = batch_uses_additive_blend(batch.blend);

            // If this batch has packed binary instances, emit a draw command
            // from the packed metadata (the instances were already added by
            // unpack_binary_instances above, and already layout-scaled by
            // `apply_layout_scale_to_packed_instances`). Otherwise, collect
            // from nodes.
            let key = (panel_idx as u32, batch_idx as u32);
            if let Some(meta) = batch_meta.get(&key) {
                collector.draw_commands.push(DrawCommand {
                    kind: meta.kind,
                    instance_start: meta.instance_start as u32,
                    instance_count: meta.instance_count as u32,
                    additive,
                    is_mark: true,
                    plot_area: panel_plot_area,
                    panel_id: panel_idx,
                    y_slot: batch.y_slot,
                });
            } else {
                // Mark batches → mark mesh (zoom transform). Record this
                // batch's mesh index range tagged with its y-slot so the render
                // loop can bind the (panel, slot) affine (secondary-y, #52).
                // Single-y charts emit one slot-0 range per mesh batch, which
                // draws byte-identically to the former per-panel recording.
                let mesh_start = collector.mesh.indices.len() as u32;
                let scaled_nodes = maybe_transform_nodes(&batch.nodes, ls);
                collector.collect_mark(
                    &scaled_nodes,
                    additive,
                    panel_plot_area,
                    panel_idx,
                    batch.stroke_cap,
                    batch.stroke_join,
                    batch.y_slot,
                );
                let mesh_end = collector.mesh.indices.len() as u32;
                collector.record_mark_mesh_panel(
                    mesh_start,
                    mesh_end,
                    panel_plot_area_arr,
                    panel_idx,
                    batch.y_slot,
                );
            }
        }

        // Axes, strip titles: non-mark → static mesh
        collector.collect_static(&maybe_transform_nodes(&panel.axes, ls), None, None);
        collector.collect_static(&maybe_transform_nodes(&panel.strip_title, ls), None, None);
        // Annotations: route to annotation_mesh so they appear above data
        // marks in WASM (matching SVG painter order).
        collector.collect_annotation(&maybe_transform_nodes(&panel.annotations, ls), None, None);
    }

    // Legend, decorations: non-mark → static mesh
    collector.collect_static(&scene.legend, None, None);
    collector.collect_static(&scene.decorations, None, None);

    SceneData {
        circle_instances: collector.circles,
        rect_instances: collector.rects,
        mesh_buffers: collector.mesh,
        static_mesh_buffers: collector.static_mesh,
        annotation_mesh_buffers: collector.annotation_mesh,
        text_elements: collector.texts,
        image_quads: collector.images,
        raw_fragments: collector.raws,
        background,
        width: scene.width as f32,
        height: scene.height as f32,
        packed_batch_meta: batch_meta,
        draw_commands: collector.draw_commands,
        mark_mesh_panels: collector.mark_mesh_panels,
        panel_count: scene.panels.len().max(1),
        panel_slot_counts: scene
            .panels
            .iter()
            .map(|p| panel_slot_count(&p.coord))
            .collect(),
    }
}

/// Flag: tooltips string table follows instance data (+ optional data_indices).
const HAS_TOOLTIPS: u32 = 0x1;
/// Flag: `count × u32` data-index array follows instance data.
const HAS_DATA_INDICES: u32 = 0x2;

/// Read a little-endian `u32` from `data[offset..offset+4]`.
///
/// The caller must guarantee `offset + 4 <= data.len()`.
#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    u32::from_le_bytes(bytes)
}

/// Unpack raw binary instance data into circle/rect buffers.
///
/// Format (v2, 20-byte header): repeated
/// `[panel_idx: u32][batch_idx: u32][kind: u32][count: u32][flags: u32]`
/// followed by instance data, then optional data_indices and tooltip bytes
/// based on `flags`.
///
/// The `kind: u32` header is decoded once into [`DrawKind`] via
/// [`DrawKind::try_from`] (`0` → Circle, `1` → Rect); an unrecognized value
/// halts parsing.
///
fn unpack_binary_instances(
    data: &[u8],
    circles: &mut Vec<CircleInstance>,
    rects: &mut Vec<RectInstance>,
    meta: &mut HashMap<(u32, u32), PackedBatchMeta>,
) {
    let mut offset = 0;
    while offset + 20 <= data.len() {
        let panel_idx = read_u32_le(data, offset);
        let batch_idx = read_u32_le(data, offset + 4);
        // Decode the packed kind discriminant ONCE into the typed `DrawKind`.
        // An unrecognized value is a malformed batch header; stop parsing (the
        // same effect the old `_ => break` fallthrough had for unknown `u32`).
        let Ok(kind) = DrawKind::try_from(read_u32_le(data, offset + 8)) else {
            break;
        };
        let count = read_u32_le(data, offset + 12) as usize;
        let flags = read_u32_le(data, offset + 16);
        offset += 20;

        // Read instance data, tracking the start index for hit-testing.
        // Returns (instance_byte_len, instance_start, loaded_count) where
        // loaded_count reflects what was ACTUALLY pushed (0 on bytemuck failure)
        // so PackedBatchMeta.instance_count never points at phantom instances.
        let (instance_byte_len, instance_start, loaded_count) = match kind {
            DrawKind::Circle => {
                let byte_len = count * std::mem::size_of::<CircleInstance>();
                if offset + byte_len > data.len() {
                    break;
                }
                let start = circles.len();
                if let Ok(instances) =
                    bytemuck::try_cast_slice::<_, CircleInstance>(&data[offset..offset + byte_len])
                {
                    circles.extend_from_slice(instances);
                    // Packed instance color channels are sRGB — convert to linear.
                    for ci in &mut circles[start..] {
                        linearize_color_channels(&mut ci.fill_color);
                        linearize_color_channels(&mut ci.stroke_color);
                    }
                }
                let loaded = circles.len() - start;
                (byte_len, start, loaded)
            }
            DrawKind::Rect => {
                let byte_len = count * std::mem::size_of::<RectInstance>();
                if offset + byte_len > data.len() {
                    break;
                }
                let start = rects.len();
                if let Ok(instances) =
                    bytemuck::try_cast_slice::<_, RectInstance>(&data[offset..offset + byte_len])
                {
                    rects.extend_from_slice(instances);
                    // Packed instance color channels are sRGB — convert to linear.
                    for ri in &mut rects[start..] {
                        linearize_color_channels(&mut ri.fill_color);
                        linearize_color_channels(&mut ri.stroke_color);
                    }
                }
                let loaded = rects.len() - start;
                (byte_len, start, loaded)
            }
        };
        offset += instance_byte_len;

        // Read data_indices if flagged.
        let data_indices = if flags & HAS_DATA_INDICES != 0 {
            let indices_byte_len = count * 4;
            if offset + indices_byte_len > data.len() {
                break;
            }
            let indices: Vec<u32> = (0..count)
                .map(|i| read_u32_le(data, offset + i * 4))
                .collect();
            offset += indices_byte_len;
            Some(indices)
        } else {
            None
        };

        // Read tooltip bytes if flagged.
        let tooltip_bytes = if flags & HAS_TOOLTIPS != 0 {
            // Delegate to PackedTooltipTable to measure the table's byte length.
            // Pass `count` (the batch's instance count from the header) so the
            // walker stops after exactly `count × num_fields` value entries rather
            // than running to buffer-end, which would over-run into subsequent
            // batches in the concatenated sidecar. (WASM-03 fix)
            if offset > data.len() {
                break;
            }
            let table_slice = &data[offset..];
            let table_len = PackedTooltipTable::total_byte_length(table_slice, count);
            let end = (offset + table_len).min(data.len());
            let bytes = data[offset..end].to_vec();
            offset = end;
            Some(bytes)
        } else {
            None
        };

        meta.insert(
            (panel_idx, batch_idx),
            PackedBatchMeta {
                data_indices,
                tooltip_bytes,
                kind,
                instance_start,
                // Record the count ACTUALLY loaded, not the header `count`.
                // On a bytemuck cast failure the `if let Ok` block is skipped
                // so loaded_count is 0, preventing phantom instance references
                // in hit-testing. On the success path loaded_count == count. (F3)
                instance_count: loaded_count,
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_nodes(
    nodes: &[SceneNode],
    circles: &mut Vec<CircleInstance>,
    rects: &mut Vec<RectInstance>,
    mesh: &mut VertexBuffers<MeshVertex, u32>,
    texts: &mut Vec<TextElementData>,
    images: &mut Vec<ImageQuad>,
    raws: &mut Vec<RawFragmentData>,
    batch_cap: Option<StrokeCap>,
    batch_join: Option<StrokeJoin>,
) {
    for node in nodes {
        match node {
            SceneNode::Circle { cx, cy, r, style } => {
                circles.push(CircleInstance {
                    center: [*cx as f32, *cy as f32],
                    radius: *r as f32,
                    fill_color: opt_color_to_f32(style.fill.as_ref(), style.fill_opacity),
                    stroke_color: opt_color_to_f32(style.stroke.as_ref(), 1.0),
                    stroke_width: style.stroke_width as f32,
                    opacity: style.opacity as f32,
                    stroke_opacity: style.stroke_opacity as f32,
                    stroke_dash: stroke_dash_index(&style.stroke_dash),
                    angle: style.angle as f32,
                });
            }
            SceneNode::Rect {
                x,
                y,
                w,
                h,
                style,
                corner_radius,
            } => {
                rects.push(RectInstance {
                    position: [*x as f32, *y as f32],
                    size: [*w as f32, *h as f32],
                    corner_radius: *corner_radius as f32,
                    fill_color: opt_color_to_f32(style.fill.as_ref(), style.fill_opacity),
                    stroke_color: opt_color_to_f32(style.stroke.as_ref(), 1.0),
                    stroke_width: style.stroke_width as f32,
                    opacity: style.opacity as f32,
                    stroke_opacity: style.stroke_opacity as f32,
                    stroke_dash: stroke_dash_index(&style.stroke_dash),
                    angle: style.angle as f32,
                });
            }
            SceneNode::Line {
                x1,
                y1,
                x2,
                y2,
                style,
            } => {
                let mut s = style.clone();
                if s.stroke_cap.is_none() {
                    s.stroke_cap = batch_cap;
                }
                if s.stroke_join.is_none() {
                    s.stroke_join = batch_join;
                }
                tessellate::tessellate_line(*x1, *y1, *x2, *y2, &s, mesh);
            }
            SceneNode::Path {
                commands,
                style,
                closed,
            } => {
                tessellate::tessellate_path(commands, style, *closed, batch_cap, batch_join, mesh);
            }
            SceneNode::Polyline { points, style } => {
                let mut s = style.clone();
                if s.stroke_cap.is_none() {
                    s.stroke_cap = batch_cap;
                }
                if s.stroke_join.is_none() {
                    s.stroke_join = batch_join;
                }
                tessellate::tessellate_polyline(points, &s, mesh);
            }
            SceneNode::Polygon { rings, style } => {
                tessellate::tessellate_polygon(rings, style, mesh);
            }
            SceneNode::Text {
                x,
                y,
                content,
                style,
            } => {
                texts.push(TextElementData {
                    x: *x,
                    y: *y,
                    content: content.clone(),
                    style: style.clone(),
                });
            }
            SceneNode::Image { x, y, w, h, data } => match data {
                ImageData::Inline { bytes, .. } => {
                    if let Some(quad) = decode_image_quad(*x, *y, *w, *h, bytes) {
                        images.push(quad);
                    }
                }
                ImageData::Url { .. } => {
                    web_sys::console::warn_1(
                        &"ferrum: ImageData::Url not supported in WASM renderer".into(),
                    );
                }
            },
            SceneNode::Group { children, .. } => {
                collect_nodes(
                    children, circles, rects, mesh, texts, images, raws, batch_cap, batch_join,
                );
            }
            SceneNode::Raw { svg, anchor } => {
                let anchor_str = match anchor {
                    ferrum_scene::RawAnchor::Chrome => "chrome",
                    ferrum_scene::RawAnchor::Data => "data",
                };
                raws.push(RawFragmentData {
                    svg: svg.clone(),
                    anchor: anchor_str.to_string(),
                });
            }
        }
    }
}

fn decode_image_quad(x: f64, y: f64, w: f64, h: f64, png_bytes: &[u8]) -> Option<ImageQuad> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(buf.len() / 3 * 4);
            for chunk in buf.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgba = Vec::with_capacity(buf.len() / 2 * 4);
            for chunk in buf.chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            rgba
        }
        png::ColorType::Grayscale => {
            let mut rgba = Vec::with_capacity(buf.len() * 4);
            for &g in &buf {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
            rgba
        }
        _ => return None,
    };

    Some(ImageQuad {
        x: x as f32,
        y: y as f32,
        w: w as f32,
        h: h as f32,
        data: rgba,
        img_width: info.width,
        img_height: info.height,
    })
}

// ── PackedTooltipTable ────────────────────────────────────────────────────────

/// A zero-copy cursor over the packed tooltip string-table format.
///
/// Binary layout (written by `ferrum-core/src/render/pack_instances.rs`):
/// ```text
/// [num_fields: u32]
/// num_fields × { [name_len: u32] [name bytes] }
/// total_rows × num_fields × { [value_len: u32] [value bytes] }
/// ```
///
/// `parse()` reads the header (field count + names) once.  Callers then use
/// the returned table to walk individual rows without re-parsing the header.
/// The four former hand-rolled walkers (`unpack_binary_instances` skip-scan,
/// `parse_tooltip_json`, `tooltip_field_value`, and the `format_tooltip_content`
/// non-packed consumer) are all driven through this single canonical layout
/// description.
pub(crate) struct PackedTooltipTable<'a> {
    bytes: &'a [u8],
    /// Names of the `num_fields` columns, in order.
    field_names: Vec<&'a str>,
    /// Byte offset of the first value entry (start of row 0, column 0).
    values_start: usize,
}

impl<'a> PackedTooltipTable<'a> {
    /// Parse the header (field count + names) of a tooltip byte slice.
    ///
    /// Returns `None` when the slice is empty, the `num_fields` header is
    /// missing or zero, or a name entry is truncated.  The returned table is
    /// cheap to construct — it borrows `bytes` without copying.
    pub(crate) fn parse(bytes: &'a [u8]) -> Option<Self> {
        let mut offset = 0usize;
        if offset + 4 > bytes.len() {
            return None;
        }
        let num_fields = read_u32_le(bytes, offset) as usize;
        offset += 4;
        if num_fields == 0 {
            return None;
        }

        let mut field_names = Vec::with_capacity(num_fields);
        for _ in 0..num_fields {
            if offset + 4 > bytes.len() {
                return None;
            }
            let slen = read_u32_le(bytes, offset) as usize;
            offset += 4;
            if offset + slen > bytes.len() {
                return None;
            }
            let name = std::str::from_utf8(&bytes[offset..offset + slen]).unwrap_or("");
            field_names.push(name);
            offset += slen;
        }

        Some(PackedTooltipTable {
            bytes,
            field_names,
            values_start: offset,
        })
    }

    /// Number of columns in this table.
    fn num_fields(&self) -> usize {
        self.field_names.len()
    }

    /// Advance `offset` past `count` length-prefixed string entries.
    ///
    /// Returns `None` (bounds violation) if the data is truncated; updates
    /// `offset` in place on success.
    fn skip_entries(bytes: &[u8], offset: &mut usize, count: usize) -> Option<()> {
        for _ in 0..count {
            if *offset + 4 > bytes.len() {
                return None;
            }
            let slen = read_u32_le(bytes, *offset) as usize;
            *offset += 4 + slen;
            if *offset > bytes.len() {
                return None;
            }
        }
        Some(())
    }

    /// Read the next length-prefixed string at `offset`, advancing past it.
    ///
    /// Returns `None` on truncation.
    fn read_str<'b>(bytes: &'b [u8], offset: &mut usize) -> Option<&'b str> {
        if *offset + 4 > bytes.len() {
            return None;
        }
        let slen = read_u32_le(bytes, *offset) as usize;
        *offset += 4;
        if *offset + slen > bytes.len() {
            return None;
        }
        let s = std::str::from_utf8(&bytes[*offset..*offset + slen]).ok()?;
        *offset += slen;
        Some(s)
    }

    /// Seek to the start of `row_idx` in the values section.
    ///
    /// Returns `None` when the row is out of range or the data is truncated.
    fn seek_to_row(&self, row_idx: usize) -> Option<usize> {
        let mut offset = self.values_start;
        Self::skip_entries(self.bytes, &mut offset, row_idx * self.num_fields())?;
        Some(offset)
    }

    /// Total byte length of the table (header + exactly `count` value rows).
    ///
    /// Used by `unpack_binary_instances` to measure how many bytes this batch's
    /// tooltip table occupies in the concatenated sidecar buffer so the outer
    /// batch-load cursor can advance to the next batch's 20-byte header.
    ///
    /// The walk is bounded to exactly `count × num_fields` value entries —
    /// mirroring the producer's layout in `pack_instances.rs::pack_tooltips`.
    /// Walking to buffer-end instead would over-run this batch's table into the
    /// next batch's header/instance bytes when multiple batches are concatenated,
    /// corrupting the first batch's tooltip slice and causing the outer
    /// `while offset+20 <= data.len()` loop to terminate early so every
    /// subsequent batch would silently never load.
    ///
    /// Returns `bytes.len()` (clamped) when the table header is malformed or the
    /// data is truncated — the caller's `.min(data.len())` guard handles it.
    pub(crate) fn total_byte_length(bytes: &'_ [u8], count: usize) -> usize {
        let Some(table) = PackedTooltipTable::parse(bytes) else {
            // If parse fails (empty or malformed header), return the full slice
            // length so the caller's `.min(data.len())` guard clamps safely.
            return bytes.len();
        };
        let mut offset = table.values_start;
        // Walk exactly count × num_fields value entries — NOT to buffer end.
        // This is the critical boundary: the old "walk until buffer runs out"
        // loop over-ran this batch's table into subsequent batches' bytes.
        let total_value_entries = count * table.num_fields();
        if Self::skip_entries(bytes, &mut offset, total_value_entries).is_none() {
            // Truncated table — clamp to whatever offset we reached.
            return bytes.len();
        }
        offset
    }

    /// Return all field values for `row_idx` as `Vec<(name, value)>` pairs.
    ///
    /// Returns `None` when the row is out of range or data is malformed.
    pub(crate) fn row_fields(&self, row_idx: usize) -> Option<Vec<(&str, &str)>> {
        let mut offset = self.seek_to_row(row_idx)?;
        let mut result = Vec::with_capacity(self.num_fields());
        for &name in &self.field_names {
            let value = Self::read_str(self.bytes, &mut offset)?;
            result.push((name, value));
        }
        Some(result)
    }

    /// Return the value for `field` in `row_idx`, or `None` if absent.
    pub(crate) fn field_value(&self, row_idx: usize, field: &str) -> Option<&str> {
        let field_col = self.field_names.iter().position(|&n| n == field)?;
        let mut offset = self.seek_to_row(row_idx)?;
        for col in 0..self.num_fields() {
            let value = Self::read_str(self.bytes, &mut offset)?;
            if col == field_col {
                return Some(value);
            }
        }
        None
    }
}

// ── Public tooltip helpers ────────────────────────────────────────────────────

/// Parse a tooltip string table and return a JSON string for one row.
///
/// The `tooltip_bytes` slice uses the [`PackedTooltipTable`] format:
/// `[num_fields: u32]`, followed by `num_fields` length-prefixed field name
/// strings, then `total_rows × num_fields` length-prefixed value strings
/// (row-major).
///
/// Returns `{"fields":[{"name":"x","value":"1.23"},…]}` for the requested
/// `row_idx`, or `"{}"` if the index is out of range or the data is malformed.
pub fn parse_tooltip_json(tooltip_bytes: &[u8], row_idx: usize) -> String {
    let Some(table) = PackedTooltipTable::parse(tooltip_bytes) else {
        return "{}".to_string();
    };
    let Some(fields) = table.row_fields(row_idx) else {
        return "{}".to_string();
    };

    let fields_json: Vec<String> = fields
        .iter()
        .map(|(name, value)| {
            let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
            let escaped_value = value.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                r#"{{"name":"{}","value":"{}"}}"#,
                escaped_name, escaped_value
            )
        })
        .collect();

    format!(r#"{{"fields":[{}]}}"#, fields_json.join(","))
}

/// Read a single tooltip field value (as a string) for one row of a packed
/// batch's tooltip string table.
///
/// The layout matches [`parse_tooltip_json`] (see [`PackedTooltipTable`]).
/// Returns `None` when the field is absent, the row is out of range, or the
/// data is malformed — packed legend/field-value matching uses this to mirror
/// the unpacked tooltip-field path on `< 1000`-mark batches.
pub fn tooltip_field_value(tooltip_bytes: &[u8], row_idx: usize, field: &str) -> Option<String> {
    let table = PackedTooltipTable::parse(tooltip_bytes)?;
    table.field_value(row_idx, field).map(|s| s.to_string())
}

/// Returns `true` when a mark batch should use the additive blend pipeline.
///
/// Only `BlendMode::Additive` raster batches use additive compositing; all
/// other batches use the standard alpha pipeline. This pure-logic helper is
/// tested on the host; the actual pipeline selection in `render.rs` calls it.
pub fn batch_uses_additive_blend(blend: BlendMode) -> bool {
    matches!(blend, BlendMode::Additive)
}

fn opt_color_to_f32(color: Option<&Color>, opacity: f64) -> [f32; 4] {
    match color {
        Some(c) => color_to_linear(c, opacity),
        None => [0.0, 0.0, 0.0, 0.0],
    }
}

/// Map a `FillStroke.stroke_dash` pattern vector to the closest palette index.
///
/// `None` / empty vec → 0.0 (solid). Otherwise matches the first non-empty
/// palette pattern by round-trip string comparison of the vec joined with ",".
/// Falls back to 0.0 (solid) for unrecognised patterns.
pub(crate) fn stroke_dash_index(dash: &Option<Vec<f64>>) -> f32 {
    match dash {
        None => 0.0,
        Some(v) if v.is_empty() => 0.0,
        Some(v) => {
            let joined = v
                .iter()
                .map(|x| {
                    // Format without trailing ".0" for integer-valued floats so
                    // "6,3" matches rather than "6.0,3.0".
                    if x.fract() == 0.0 {
                        format!("{}", *x as i64)
                    } else {
                        format!("{x}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            STROKE_DASH_PALETTE
                .iter()
                .enumerate()
                .skip(1) // index 0 is always solid / empty
                .find(|(_, pat)| **pat == joined)
                .map(|(i, _)| i as f32)
                .unwrap_or(0.0) // unrecognised pattern → solid
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── #52: per-(panel, y-slot) transform-slot index mapping ────────────

    #[test]
    fn panel_slot_count_single_y_is_one() {
        let coord = ferrum_scene::CoordKind::Cartesian {
            x_domain: None,
            y_domain: Some((0.0, 1.0)),
            expand: true,
            clip: true,
            y_domains: Vec::new(),
        };
        assert_eq!(panel_slot_count(&coord), 1, "empty y_domains → 1 slot");
    }

    #[test]
    fn panel_slot_count_dual_axis_matches_domain_list() {
        let coord = ferrum_scene::CoordKind::Cartesian {
            x_domain: None,
            y_domain: Some((0.0, 1.0)),
            expand: true,
            clip: true,
            y_domains: vec![Some((0.0, 1.0)), Some((0.0, 9.0)), None],
        };
        assert_eq!(panel_slot_count(&coord), 3, "one slot per y_domains entry");
    }

    #[test]
    fn transform_slot_index_single_y_is_panel_id() {
        // Every panel single-y (count == 1): the flat index is the panel id,
        // so the transform-slot vector matches the former per-panel one.
        let counts = vec![1, 1, 1];
        assert_eq!(transform_slot_index(&counts, 0, 0), 0);
        assert_eq!(transform_slot_index(&counts, 1, 0), 1);
        assert_eq!(transform_slot_index(&counts, 2, 0), 2);
        assert_eq!(total_transform_slots(&counts), 3);
    }

    #[test]
    fn transform_slot_index_lays_slots_consecutively() {
        // Panel 0: 1 slot (idx 0). Panel 1: 2 slots (idx 1,2). Panel 2: 1 (idx 3).
        let counts = vec![1, 2, 1];
        assert_eq!(transform_slot_index(&counts, 0, 0), 0);
        assert_eq!(transform_slot_index(&counts, 1, 0), 1);
        assert_eq!(transform_slot_index(&counts, 1, 1), 2);
        assert_eq!(transform_slot_index(&counts, 2, 0), 3);
        assert_eq!(total_transform_slots(&counts), 4);
    }

    #[test]
    fn transform_slot_index_clamps_out_of_range_slot() {
        // An out-of-range y_slot clamps to the panel's last slot rather than
        // indexing past the panel's block.
        let counts = vec![2, 2];
        assert_eq!(transform_slot_index(&counts, 0, 5), 1, "clamp to panel 0's last slot");
        assert_eq!(transform_slot_index(&counts, 1, 9), 3, "clamp to panel 1's last slot");
    }

    #[test]
    fn panel_slot_range_single_y_is_one_slot_per_panel() {
        let counts = vec![1, 1, 1];
        assert_eq!(panel_slot_range(&counts, 0), 0..1);
        assert_eq!(panel_slot_range(&counts, 1), 1..2);
        assert_eq!(panel_slot_range(&counts, 2), 2..3);
    }

    #[test]
    fn panel_slot_range_covers_every_slot_of_a_multi_slot_panel() {
        // Panel 0: 1 slot (0..1). Panel 1: 2 slots (1..3). Panel 2: 1 (3..4).
        let counts = vec![1, 2, 1];
        assert_eq!(panel_slot_range(&counts, 0), 0..1);
        assert_eq!(panel_slot_range(&counts, 1), 1..3);
        assert_eq!(panel_slot_range(&counts, 2), 3..4);
    }

    #[test]
    fn panel_slot_range_out_of_range_panel_defaults_to_one_slot() {
        let counts = vec![2, 2];
        assert_eq!(panel_slot_range(&counts, 5), 4..5);
    }

    /// Discriminating test for the `set_transform` reset fix (#52 Task 9c):
    /// mirrors `WasmRenderer::set_transform`'s reset loop
    /// (`for slot in &mut self.slot_rescales[panel_slot_range(...)] { *slot =
    /// Affine2::identity(); }`) against a 3-panel scene where panel 1 has two
    /// independent-y slots. A domainParam rescale on panel 1's secondary
    /// layer (slot 1) must return to identity when panel 1's view resets,
    /// while a sibling panel's own rescaled slot is left untouched.
    #[test]
    fn panel_slot_range_reset_clears_only_the_owning_panels_slots() {
        use crate::zoom_pan::Affine2;

        // panel 0: 1 slot (idx 0). panel 1: 2 slots (idx 1,2). panel 2: 1 (idx 3).
        let counts = vec![1, 2, 1];
        let mut slot_rescales = vec![Affine2::identity(); total_transform_slots(&counts)];

        // Simulate two independent rescales: panel 1's secondary-y layer
        // (slot idx 2) via a domainParam brush, and panel 2 (slot idx 3) via
        // its own independent rescale.
        slot_rescales[2] = Affine2 { sx: 1.0, sy: 2.5, tx: 0.0, ty: -10.0 };
        slot_rescales[3] = Affine2 { sx: 1.0, sy: 4.0, tx: 0.0, ty: 5.0 };

        // Reset panel 1 (as `set_transform(1, ...)` would): only its owned
        // range (1..3) resets to identity.
        for slot in &mut slot_rescales[panel_slot_range(&counts, 1)] {
            *slot = Affine2::identity();
        }

        assert_eq!(slot_rescales[0].sy, 1.0, "panel 0's untouched slot stays identity");
        assert_eq!(slot_rescales[1].sy, 1.0, "panel 1's primary slot was already identity");
        assert_eq!(
            slot_rescales[2].sy, 1.0,
            "panel 1's rescaled secondary-y slot must reset to identity"
        );
        assert_eq!(
            slot_rescales[2].ty, 0.0,
            "panel 1's rescaled secondary-y slot must reset to identity"
        );
        assert_eq!(
            slot_rescales[3].sy, 4.0,
            "panel 2's own rescale must survive panel 1's reset"
        );
    }

    /// Discriminating test for the out-of-range `panel_id` fix (#52 Task 9e):
    /// `WasmRenderer::set_transform` now bounds the reset loop via
    /// `self.slot_rescales.get_mut(slot_range)` instead of indexing the
    /// range directly. `panel_slot_range` returns `total..total+1` for a
    /// panel past the end of `panel_slot_counts` (see
    /// `panel_slot_range_out_of_range_panel_defaults_to_one_slot` above),
    /// which is out of bounds for a `slot_rescales` of length `total` — a
    /// direct index there panics and aborts the WASM module. This mirrors
    /// that lookup pattern and asserts it must not panic and must leave
    /// every in-range slot untouched.
    #[test]
    fn panel_slot_range_reset_out_of_range_panel_is_noop() {
        use crate::zoom_pan::Affine2;

        let counts = vec![1, 2, 1];
        let mut slot_rescales = vec![Affine2::identity(); total_transform_slots(&counts)];
        slot_rescales[2] = Affine2 { sx: 1.0, sy: 2.5, tx: 0.0, ty: -10.0 };
        slot_rescales[3] = Affine2 { sx: 1.0, sy: 4.0, tx: 0.0, ty: 5.0 };

        let out_of_range_slot_range = panel_slot_range(&counts, 5);
        assert_eq!(out_of_range_slot_range, 4..5, "sanity: past the end of slot_rescales (len 4)");

        // Bounded reset: must not panic, and must leave every existing slot
        // untouched since panel 5 owns none of them.
        if let Some(slots) = slot_rescales.get_mut(out_of_range_slot_range) {
            for slot in slots {
                *slot = Affine2::identity();
            }
        }

        assert_eq!(slot_rescales[0].sy, 1.0);
        assert_eq!(slot_rescales[1].sy, 1.0);
        assert_eq!(
            slot_rescales[2].sy, 2.5,
            "panel 1's rescaled secondary-y slot must survive an out-of-range reset"
        );
        assert_eq!(
            slot_rescales[3].sy, 4.0,
            "panel 2's own rescale must survive an out-of-range reset"
        );
    }

    // ── sRGB-to-linear conversion ────────────────────────────────────

    #[test]
    fn srgb_to_linear_boundary_values() {
        // 0.0 maps to 0.0 (black stays black).
        assert!((srgb_to_linear(0.0)).abs() < 1e-7);
        // 1.0 maps to 1.0 (white stays white).
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn srgb_to_linear_mid_grey() {
        // sRGB 0.5 ≈ linear 0.214
        let linear = srgb_to_linear(0.5);
        assert!(
            (linear - 0.214).abs() < 0.002,
            "sRGB 0.5 → linear ~0.214, got {linear}"
        );
    }

    #[test]
    fn srgb_to_linear_low_range_uses_linear_segment() {
        // Below 0.04045 the function is s/12.92 (linear segment).
        let s = 0.03;
        let expected = 0.03 / 12.92;
        assert!((srgb_to_linear(s) - expected).abs() < 1e-7);
    }

    #[test]
    fn srgb_to_linear_monotonic() {
        // The conversion must be monotonically increasing.
        let mut prev = 0.0_f32;
        for i in 1..=100 {
            let s = i as f32 / 100.0;
            let l = srgb_to_linear(s);
            assert!(
                l >= prev,
                "srgb_to_linear must be monotonic: f({s}) = {l} < f({}) = {prev}",
                (i - 1) as f32 / 100.0
            );
            prev = l;
        }
    }

    // ── Task 1: instance struct layout and round-trip ─────────────────

    #[test]
    fn circle_instance_has_new_stroke_fields() {
        let inst = CircleInstance {
            center: [10.0, 20.0],
            radius: 5.0,
            fill_color: [1.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 2.0,
            opacity: 0.8,
            stroke_opacity: 0.5,
            stroke_dash: 1.0, // "dashed"
            angle: 45.0,
        };
        assert!((inst.stroke_opacity - 0.5).abs() < 1e-6);
        assert!((inst.stroke_dash - 1.0).abs() < 1e-6);
        assert!((inst.angle - 45.0).abs() < 1e-6);
    }

    #[test]
    fn rect_instance_has_new_stroke_fields() {
        let inst = RectInstance {
            position: [0.0, 0.0],
            size: [100.0, 50.0],
            corner_radius: 4.0,
            fill_color: [0.0, 1.0, 0.0, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 0.75,
            stroke_dash: 2.0, // "dotted"
            angle: 90.0,
        };
        assert!((inst.stroke_opacity - 0.75).abs() < 1e-6);
        assert!((inst.stroke_dash - 2.0).abs() < 1e-6);
        assert!((inst.angle - 90.0).abs() < 1e-6);
    }

    #[test]
    fn circle_instance_pod_roundtrip() {
        // bytemuck::Pod requires no padding — confirm the struct round-trips
        // through byte representation without panicking.
        let inst = CircleInstance {
            center: [1.0, 2.0],
            radius: 3.0,
            fill_color: [0.1, 0.2, 0.3, 1.0],
            stroke_color: [0.4, 0.5, 0.6, 0.9],
            stroke_width: 1.5,
            opacity: 0.9,
            stroke_opacity: 0.6,
            stroke_dash: 3.0,
            angle: 180.0,
        };
        let bytes = bytemuck::bytes_of(&inst);
        // 16 floats × 4 bytes = 64 bytes
        assert_eq!(
            bytes.len(),
            16 * 4,
            "CircleInstance must be exactly 16 floats"
        );
        let back: &CircleInstance = bytemuck::from_bytes(bytes);
        assert!((back.stroke_opacity - 0.6).abs() < 1e-6);
        assert!((back.stroke_dash - 3.0).abs() < 1e-6);
        assert!((back.angle - 180.0).abs() < 1e-6);
    }

    #[test]
    fn rect_instance_pod_roundtrip() {
        let inst = RectInstance {
            position: [5.0, 10.0],
            size: [40.0, 20.0],
            corner_radius: 2.0,
            fill_color: [0.9, 0.8, 0.7, 1.0],
            stroke_color: [0.1, 0.2, 0.3, 0.5],
            stroke_width: 2.0,
            opacity: 0.85,
            stroke_opacity: 0.4,
            stroke_dash: 1.0,
            angle: 30.0,
        };
        let bytes = bytemuck::bytes_of(&inst);
        // 18 floats × 4 bytes = 72 bytes
        assert_eq!(
            bytes.len(),
            18 * 4,
            "RectInstance must be exactly 18 floats"
        );
        let back: &RectInstance = bytemuck::from_bytes(bytes);
        assert!((back.stroke_opacity - 0.4).abs() < 1e-6);
        assert!((back.stroke_dash - 1.0).abs() < 1e-6);
        assert!((back.angle - 30.0).abs() < 1e-6);
    }

    // ── stroke_dash_index helper ──────────────────────────────────────

    #[test]
    fn stroke_dash_index_solid_on_none() {
        assert!((stroke_dash_index(&None) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stroke_dash_index_solid_on_empty_vec() {
        assert!((stroke_dash_index(&Some(vec![])) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stroke_dash_index_dashed() {
        // "6,3" → index 1
        assert!((stroke_dash_index(&Some(vec![6.0, 3.0])) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stroke_dash_index_dotted() {
        // "2,3" → index 2
        assert!((stroke_dash_index(&Some(vec![2.0, 3.0])) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn stroke_dash_index_dash_dot() {
        // "6,3,2,3" → index 3
        assert!((stroke_dash_index(&Some(vec![6.0, 3.0, 2.0, 3.0])) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn stroke_dash_index_unknown_pattern_gives_solid() {
        // unrecognised → solid (0.0)
        assert!((stroke_dash_index(&Some(vec![5.0, 5.0])) - 0.0).abs() < 1e-6);
    }

    // ── Task 3: scene_load builds correct instance fields ────────────

    /// Build a minimal SceneGraph with one Circle and one Rect; confirm the
    /// scene loader populates the new stroke fields from style.
    #[test]
    fn load_scene_populates_stroke_opacity_and_angle_for_circle() {
        use ferrum_scene::{BlendMode, FillStroke, MarkBatch, MarkBatchKind, Panel, SceneNode};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect, SceneGraph};

        let style = FillStroke {
            fill: Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke_width: 2.0,
            opacity: 1.0,
            stroke_opacity: 0.5,
            fill_opacity: 1.0,
            stroke_dash: Some(vec![6.0, 3.0]), // index 1 = dashed
            angle: 45.0,
        };

        let node = SceneNode::Circle {
            cx: 50.0,
            cy: 50.0,
            r: 10.0,
            style,
        };

        let scene = SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
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
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![node],
                    data_indices: None,
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
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(data.circle_instances.len(), 1);
        let ci = &data.circle_instances[0];
        assert!(
            (ci.stroke_opacity - 0.5).abs() < 1e-6,
            "stroke_opacity should be 0.5, got {}",
            ci.stroke_opacity
        );
        assert!(
            (ci.stroke_dash - 1.0).abs() < 1e-6,
            "stroke_dash index should be 1 (dashed), got {}",
            ci.stroke_dash
        );
        assert!(
            (ci.angle - 45.0).abs() < 1e-6,
            "angle should be 45.0, got {}",
            ci.angle
        );
    }

    #[test]
    fn load_scene_populates_stroke_opacity_and_angle_for_rect() {
        use ferrum_scene::{BlendMode, FillStroke, MarkBatch, MarkBatchKind, Panel, SceneNode};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect, SceneGraph};

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 128,
                b: 255,
                a: 255,
            }),
            stroke: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke_width: 1.0,
            opacity: 0.9,
            stroke_opacity: 0.75,
            fill_opacity: 1.0,
            stroke_dash: Some(vec![2.0, 3.0]), // index 2 = dotted
            angle: 30.0,
        };

        let node = SceneNode::Rect {
            x: 10.0,
            y: 20.0,
            w: 40.0,
            h: 30.0,
            style,
            corner_radius: 0.0,
        };

        let scene = SceneGraph {
            width: 200.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 200.0,
                    h: 100.0,
                },
                clip: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 200.0,
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
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Bar,
                    nodes: vec![node],
                    data_indices: None,
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
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(data.rect_instances.len(), 1);
        let ri = &data.rect_instances[0];
        assert!(
            (ri.stroke_opacity - 0.75).abs() < 1e-6,
            "stroke_opacity should be 0.75, got {}",
            ri.stroke_opacity
        );
        assert!(
            (ri.stroke_dash - 2.0).abs() < 1e-6,
            "stroke_dash index should be 2 (dotted), got {}",
            ri.stroke_dash
        );
        assert!(
            (ri.angle - 30.0).abs() < 1e-6,
            "angle should be 30.0, got {}",
            ri.angle
        );
    }

    // ── Task 6: blend mode selection ────────────────────────────────

    /// Assert that `batch_uses_additive_blend` correctly identifies additive
    /// batches: the function is the single testable decision point for
    /// per-batch pipeline selection.
    #[test]
    fn additive_blend_mode_selects_additive_pipeline() {
        assert!(
            batch_uses_additive_blend(BlendMode::Additive),
            "BlendMode::Additive must select the additive pipeline"
        );
    }

    #[test]
    fn normal_blend_mode_does_not_select_additive_pipeline() {
        assert!(
            !batch_uses_additive_blend(BlendMode::Normal),
            "BlendMode::Normal must NOT select the additive pipeline"
        );
    }

    // ── Defaults: missing stroke_opacity/angle use defaults ─────────

    #[test]
    fn load_scene_uses_defaults_when_stroke_fields_absent() {
        use ferrum_scene::{BlendMode, FillStroke, MarkBatch, MarkBatchKind, Panel, SceneNode};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect, SceneGraph};

        // FillStroke with default stroke_opacity (1.0) and angle (0.0)
        let style = FillStroke {
            fill: Some(Color {
                r: 100,
                g: 100,
                b: 100,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0, // default
            fill_opacity: 1.0,   // default
            stroke_dash: None,   // solid → 0.0
            angle: 0.0,          // default
        };

        let node = SceneNode::Circle {
            cx: 25.0,
            cy: 25.0,
            r: 5.0,
            style,
        };

        let scene = SceneGraph {
            width: 50.0,
            height: 50.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 50.0,
                    h: 50.0,
                },
                clip: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 50.0,
                    h: 50.0,
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
                    nodes: vec![node],
                    data_indices: None,
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
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        let ci = &data.circle_instances[0];
        assert!(
            (ci.stroke_opacity - 1.0).abs() < 1e-6,
            "default stroke_opacity is 1.0"
        );
        assert!(
            (ci.stroke_dash - 0.0).abs() < 1e-6,
            "default stroke_dash is 0.0 (solid)"
        );
        assert!((ci.angle - 0.0).abs() < 1e-6, "default angle is 0.0");
    }

    // ── Polygon tessellation regression tests ─────────────────────────
    //
    // These test the exact code path that the WASM interactive renderer
    // uses for hexbin and geoshape marks: SceneNode::Polygon → lyon
    // tessellation → mesh vertex/index buffers. If tessellation produces
    // zero vertices, the GPU renders nothing (the "empty hex" bug).

    fn make_scene_with_polygons(nodes: Vec<SceneNode>) -> SceneGraph {
        use ferrum_scene::{BlendMode, MarkBatch, MarkBatchKind, Panel};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect};
        SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
                },
                clip: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
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
                    kind: MarkBatchKind::Polygon,
                    nodes,
                    data_indices: None,
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
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        }
    }

    fn hex_polygon(cx: f64, cy: f64, r: f64) -> SceneNode {
        let ring: Vec<[f64; 2]> = (0..6)
            .map(|i| {
                let angle = std::f64::consts::FRAC_PI_3 * i as f64;
                [cx + r * angle.cos(), cy + r * angle.sin()]
            })
            .collect();
        SceneNode::Polygon {
            rings: vec![ring],
            style: FillStroke {
                fill: Some(Color {
                    r: 100,
                    g: 150,
                    b: 200,
                    a: 255,
                }),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
        }
    }

    #[test]
    fn polygon_tessellation_produces_nonzero_mesh() {
        let scene = make_scene_with_polygons(vec![hex_polygon(200.0, 200.0, 20.0)]);
        let data = load_scene(&scene);
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "polygon tessellation must produce vertices"
        );
        assert!(
            !data.mesh_buffers.indices.is_empty(),
            "polygon tessellation must produce indices"
        );
        assert!(
            data.mesh_buffers.indices.len() >= 3,
            "polygon must tessellate to at least one triangle"
        );
    }

    #[test]
    fn multiple_hex_polygons_all_tessellate() {
        let nodes: Vec<SceneNode> = (0..20)
            .map(|i| {
                let row = i / 5;
                let col = i % 5;
                let cx = 100.0 + col as f64 * 40.0;
                let cy = 100.0 + row as f64 * 40.0;
                hex_polygon(cx, cy, 15.0)
            })
            .collect();
        let scene = make_scene_with_polygons(nodes);
        let data = load_scene(&scene);
        // 20 hexagons × 4 triangles each (fan tessellation of a 6-gon) = 80
        // triangles minimum. Each triangle = 3 indices.
        assert!(
            data.mesh_buffers.indices.len() >= 20 * 3 * 3,
            "20 hex polygons must produce ≥180 indices; got {}",
            data.mesh_buffers.indices.len()
        );
    }

    #[test]
    fn polygon_with_hole_tessellates() {
        let exterior = vec![
            [100.0, 100.0],
            [300.0, 100.0],
            [300.0, 300.0],
            [100.0, 300.0],
        ];
        let hole = vec![
            [150.0, 150.0],
            [250.0, 150.0],
            [250.0, 250.0],
            [150.0, 250.0],
        ];
        let node = SceneNode::Polygon {
            rings: vec![exterior, hole],
            style: FillStroke {
                fill: Some(Color {
                    r: 50,
                    g: 100,
                    b: 150,
                    a: 255,
                }),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
        };
        let scene = make_scene_with_polygons(vec![node]);
        let data = load_scene(&scene);
        assert!(
            data.mesh_buffers.vertices.len() >= 8,
            "polygon with hole must tessellate to ≥8 vertices; got {}",
            data.mesh_buffers.vertices.len()
        );
    }

    #[test]
    fn polygon_no_fill_produces_no_fill_triangles() {
        let node = SceneNode::Polygon {
            rings: vec![vec![
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 200.0],
                [100.0, 200.0],
            ]],
            style: FillStroke {
                fill: None,
                stroke: Some(Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                }),
                stroke_width: 2.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
        };
        let scene = make_scene_with_polygons(vec![node]);
        let data = load_scene(&scene);
        // Stroke-only polygon still produces mesh vertices (stroke tessellation),
        // but we mainly want to ensure it doesn't panic.
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "stroke-only polygon must still produce stroke mesh vertices"
        );
    }

    #[test]
    fn polygon_vertices_are_finite() {
        let nodes: Vec<SceneNode> = (0..5)
            .map(|i| hex_polygon(100.0 + i as f64 * 60.0, 200.0, 20.0))
            .collect();
        let scene = make_scene_with_polygons(nodes);
        let data = load_scene(&scene);
        for (i, v) in data.mesh_buffers.vertices.iter().enumerate() {
            assert!(
                v.position[0].is_finite() && v.position[1].is_finite(),
                "mesh vertex {i} has non-finite position: {:?}",
                v.position
            );
            for c in &v.color {
                assert!(
                    c.is_finite(),
                    "mesh vertex {i} has non-finite color component"
                );
            }
        }
    }

    #[test]
    fn degenerate_polygon_does_not_panic() {
        // 2-point "polygon" — should be skipped gracefully, not panic.
        let node = SceneNode::Polygon {
            rings: vec![vec![[100.0, 100.0], [200.0, 200.0]]],
            style: FillStroke {
                fill: Some(Color {
                    r: 255,
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
            },
        };
        let scene = make_scene_with_polygons(vec![node]);
        let data = load_scene(&scene);
        // Degenerate polygon with <3 points should be skipped.
        assert!(
            data.mesh_buffers.vertices.is_empty(),
            "degenerate 2-point polygon should produce zero mesh vertices"
        );
    }

    // ── Shared helpers for all mark-type tests ────────────────────────

    fn default_fill_stroke() -> FillStroke {
        FillStroke {
            fill: Some(Color {
                r: 70,
                g: 130,
                b: 180,
                a: 255,
            }),
            stroke: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    fn default_stroke_style() -> ferrum_scene::StrokeStyle {
        ferrum_scene::StrokeStyle {
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            width: 1.5,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        }
    }

    fn make_scene_with_nodes(
        kind: ferrum_scene::MarkBatchKind,
        nodes: Vec<SceneNode>,
    ) -> SceneGraph {
        use ferrum_scene::{BlendMode, MarkBatch, Panel};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect};
        SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
                },
                clip: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
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
                    kind,
                    nodes,
                    data_indices: None,
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
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        }
    }

    // ── Circle (point marks) ──────────────────────────────────────────

    #[test]
    fn circle_node_produces_instance() {
        let nodes = vec![
            SceneNode::Circle {
                cx: 100.0,
                cy: 100.0,
                r: 5.0,
                style: default_fill_stroke(),
            },
            SceneNode::Circle {
                cx: 200.0,
                cy: 150.0,
                r: 8.0,
                style: default_fill_stroke(),
            },
            SceneNode::Circle {
                cx: 300.0,
                cy: 200.0,
                r: 3.0,
                style: default_fill_stroke(),
            },
        ];
        let scene = make_scene_with_nodes(MarkBatchKind::Point, nodes);
        let data = load_scene(&scene);
        assert_eq!(data.circle_instances.len(), 3, "3 circles → 3 instances");
        let c = &data.circle_instances[0];
        assert!((c.center[0] - 100.0).abs() < 1e-3);
        assert!((c.center[1] - 100.0).abs() < 1e-3);
        assert!((c.radius - 5.0).abs() < 1e-3);
        assert!(c.fill_color[3] > 0.0, "fill alpha must be nonzero");
    }

    #[test]
    fn circle_stroke_fields_populated() {
        let mut style = default_fill_stroke();
        style.stroke_opacity = 0.7;
        style.stroke_dash = Some(vec![6.0, 3.0]);
        style.angle = 30.0;
        let scene = make_scene_with_nodes(
            MarkBatchKind::Point,
            vec![SceneNode::Circle {
                cx: 50.0,
                cy: 50.0,
                r: 10.0,
                style,
            }],
        );
        let data = load_scene(&scene);
        let c = &data.circle_instances[0];
        assert!((c.stroke_opacity - 0.7).abs() < 1e-3);
        assert!((c.stroke_dash - 1.0).abs() < 1e-3, "6,3 → dashed index 1");
        assert!((c.angle - 30.0).abs() < 1e-3);
    }

    // ── Rect (bar marks) ──────────────────────────────────────────────

    #[test]
    fn rect_node_produces_instance() {
        let nodes = vec![
            SceneNode::Rect {
                x: 60.0,
                y: 50.0,
                w: 30.0,
                h: 200.0,
                style: default_fill_stroke(),
                corner_radius: 0.0,
            },
            SceneNode::Rect {
                x: 100.0,
                y: 80.0,
                w: 30.0,
                h: 170.0,
                style: default_fill_stroke(),
                corner_radius: 2.0,
            },
            SceneNode::Rect {
                x: 140.0,
                y: 30.0,
                w: 30.0,
                h: 220.0,
                style: default_fill_stroke(),
                corner_radius: 0.0,
            },
        ];
        let scene = make_scene_with_nodes(MarkBatchKind::Bar, nodes);
        let data = load_scene(&scene);
        assert_eq!(data.rect_instances.len(), 3, "3 rects → 3 instances");
        let r = &data.rect_instances[0];
        assert!((r.position[0] - 60.0).abs() < 1e-3);
        assert!((r.size[0] - 30.0).abs() < 1e-3);
        assert!((r.size[1] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn rect_corner_radius_preserved() {
        let scene = make_scene_with_nodes(
            MarkBatchKind::Rect,
            vec![SceneNode::Rect {
                x: 10.0,
                y: 20.0,
                w: 80.0,
                h: 60.0,
                style: default_fill_stroke(),
                corner_radius: 5.5,
            }],
        );
        let data = load_scene(&scene);
        let r = &data.rect_instances[0];
        assert!((r.corner_radius - 5.5).abs() < 1e-3);
    }

    // ── Line (rule / tick / segment marks) ────────────────────────────

    #[test]
    fn line_node_tessellates_to_mesh() {
        let nodes = vec![SceneNode::Line {
            x1: 50.0,
            y1: 50.0,
            x2: 300.0,
            y2: 200.0,
            style: default_stroke_style(),
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Rule, nodes);
        let data = load_scene(&scene);
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "Line node must produce mesh vertices"
        );
        assert!(
            data.mesh_buffers.indices.len() >= 6,
            "Line tessellates to ≥2 triangles (6 indices); got {}",
            data.mesh_buffers.indices.len()
        );
    }

    #[test]
    fn multiple_lines_all_tessellate() {
        let nodes: Vec<SceneNode> = (0..10)
            .map(|i| SceneNode::Line {
                x1: 50.0,
                y1: 30.0 + i as f64 * 30.0,
                x2: 400.0,
                y2: 30.0 + i as f64 * 30.0,
                style: default_stroke_style(),
            })
            .collect();
        let scene = make_scene_with_nodes(MarkBatchKind::Rule, nodes);
        let data = load_scene(&scene);
        assert!(
            data.mesh_buffers.indices.len() >= 10 * 6,
            "10 lines must produce ≥60 indices; got {}",
            data.mesh_buffers.indices.len()
        );
    }

    // ── Polyline (line marks) ─────────────────────────────────────────

    #[test]
    fn polyline_node_tessellates_to_mesh() {
        let points = vec![(50.0, 300.0), (150.0, 100.0), (250.0, 250.0), (350.0, 50.0)];
        let nodes = vec![SceneNode::Polyline {
            points,
            style: default_stroke_style(),
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Line, nodes);
        let data = load_scene(&scene);
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "Polyline must produce mesh vertices"
        );
        assert!(
            data.mesh_buffers.indices.len() >= 3 * 6,
            "4-point polyline (3 segments) → ≥18 indices; got {}",
            data.mesh_buffers.indices.len()
        );
    }

    #[test]
    fn polyline_single_segment() {
        let nodes = vec![SceneNode::Polyline {
            points: vec![(100.0, 100.0), (400.0, 300.0)],
            style: default_stroke_style(),
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Line, nodes);
        let data = load_scene(&scene);
        assert!(!data.mesh_buffers.vertices.is_empty());
    }

    // ── Path (arc / area / ribbon marks) ──────────────────────────────

    #[test]
    fn closed_path_tessellates_fill_and_stroke() {
        use ferrum_scene::PathCmd;
        let commands = vec![
            PathCmd::MoveTo { x: 100.0, y: 100.0 },
            PathCmd::LineTo { x: 300.0, y: 100.0 },
            PathCmd::LineTo { x: 300.0, y: 300.0 },
            PathCmd::LineTo { x: 100.0, y: 300.0 },
            PathCmd::Close,
        ];
        let nodes = vec![SceneNode::Path {
            commands,
            style: default_fill_stroke(),
            closed: true,
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Area, nodes);
        let data = load_scene(&scene);
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "closed Path must produce mesh vertices"
        );
        assert!(
            data.mesh_buffers.indices.len() >= 6,
            "closed square path → ≥2 fill triangles; got {}",
            data.mesh_buffers.indices.len()
        );
    }

    #[test]
    fn arc_path_with_arc_to_tessellates() {
        use ferrum_scene::PathCmd;
        // Simulate a pie wedge: move to center, line to edge, arc, close.
        let commands = vec![
            PathCmd::MoveTo { x: 200.0, y: 200.0 },
            PathCmd::LineTo { x: 200.0, y: 100.0 },
            PathCmd::ArcTo {
                rx: 100.0,
                ry: 100.0,
                rotation: 0.0,
                large_arc: false,
                sweep: true,
                x: 300.0,
                y: 200.0,
            },
            PathCmd::Close,
        ];
        let nodes = vec![SceneNode::Path {
            commands,
            style: default_fill_stroke(),
            closed: true,
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Arc, nodes);
        let data = load_scene(&scene);
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "arc wedge Path must produce mesh vertices"
        );
        assert!(
            data.mesh_buffers.indices.len() >= 3,
            "arc wedge must tessellate to ≥1 triangle"
        );
    }

    #[test]
    fn open_path_stroke_only() {
        use ferrum_scene::PathCmd;
        let commands = vec![
            PathCmd::MoveTo { x: 50.0, y: 200.0 },
            PathCmd::CubicTo {
                c1x: 150.0,
                c1y: 50.0,
                c2x: 250.0,
                c2y: 350.0,
                x: 350.0,
                y: 200.0,
            },
        ];
        let mut style = default_fill_stroke();
        style.fill = None;
        let nodes = vec![SceneNode::Path {
            commands,
            style,
            closed: false,
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Ribbon, nodes);
        let data = load_scene(&scene);
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "open stroke-only cubic Path must produce mesh vertices"
        );
    }

    #[test]
    fn multiple_arc_wedges_tessellate() {
        use ferrum_scene::PathCmd;
        // 3 pie wedges
        let wedges: Vec<SceneNode> = (0..3)
            .map(|i| {
                let angle_start = i as f64 * 120.0_f64.to_radians();
                let angle_end = (i + 1) as f64 * 120.0_f64.to_radians();
                let cx = 200.0;
                let cy = 200.0;
                let r = 100.0;
                let commands = vec![
                    PathCmd::MoveTo { x: cx, y: cy },
                    PathCmd::LineTo {
                        x: cx + r * angle_start.cos(),
                        y: cy + r * angle_start.sin(),
                    },
                    PathCmd::ArcTo {
                        rx: r,
                        ry: r,
                        rotation: 0.0,
                        large_arc: false,
                        sweep: true,
                        x: cx + r * angle_end.cos(),
                        y: cy + r * angle_end.sin(),
                    },
                    PathCmd::Close,
                ];
                SceneNode::Path {
                    commands,
                    style: default_fill_stroke(),
                    closed: true,
                }
            })
            .collect();
        let scene = make_scene_with_nodes(MarkBatchKind::Arc, wedges);
        let data = load_scene(&scene);
        assert!(
            data.mesh_buffers.indices.len() >= 3 * 3,
            "3 arc wedges must produce ≥9 indices; got {}",
            data.mesh_buffers.indices.len()
        );
    }

    // ── Text ──────────────────────────────────────────────────────────

    #[test]
    fn text_node_produces_text_element() {
        use ferrum_scene::{FontWeight, TextAnchor, TextBaseline, TextStyle};
        let style = TextStyle {
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        };
        let nodes = vec![
            SceneNode::Text {
                x: 100.0,
                y: 50.0,
                content: "Hello".to_string(),
                style: style.clone(),
            },
            SceneNode::Text {
                x: 200.0,
                y: 80.0,
                content: "World".to_string(),
                style,
            },
        ];
        let scene = make_scene_with_nodes(MarkBatchKind::Text, nodes);
        let data = load_scene(&scene);
        assert_eq!(
            data.text_elements.len(),
            2,
            "2 Text nodes → 2 text elements"
        );
        assert_eq!(data.text_elements[0].content, "Hello");
        assert!((data.text_elements[0].x - 100.0).abs() < 1e-3);
        assert_eq!(data.text_elements[1].content, "World");
    }

    #[test]
    fn text_does_not_produce_mesh_or_instances() {
        use ferrum_scene::{FontWeight, TextAnchor, TextBaseline, TextStyle};
        let style = TextStyle {
            font_size: 14.0,
            font_weight: FontWeight::Bold,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Middle,
            angle: 0.0,
            color: Color {
                r: 50,
                g: 50,
                b: 50,
                a: 255,
            },
            opacity: 1.0,
            font_family: "serif".to_string(),
        };
        let nodes = vec![SceneNode::Text {
            x: 100.0,
            y: 100.0,
            content: "Label".to_string(),
            style,
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Text, nodes);
        let data = load_scene(&scene);
        assert!(
            data.circle_instances.is_empty(),
            "text must not produce circles"
        );
        assert!(
            data.rect_instances.is_empty(),
            "text must not produce rects"
        );
        assert!(
            data.mesh_buffers.vertices.is_empty(),
            "text must not produce mesh"
        );
    }

    // ── Group (recursive) ─────────────────────────────────────────────

    #[test]
    fn group_node_recurses_into_children() {
        let children = vec![
            SceneNode::Circle {
                cx: 100.0,
                cy: 100.0,
                r: 5.0,
                style: default_fill_stroke(),
            },
            SceneNode::Rect {
                x: 200.0,
                y: 50.0,
                w: 40.0,
                h: 30.0,
                style: default_fill_stroke(),
                corner_radius: 0.0,
            },
        ];
        let nodes = vec![SceneNode::Group {
            attrs: vec![],
            children,
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Point, nodes);
        let data = load_scene(&scene);
        assert_eq!(
            data.circle_instances.len(),
            1,
            "group child circle must be collected"
        );
        assert_eq!(
            data.rect_instances.len(),
            1,
            "group child rect must be collected"
        );
    }

    // ── Mixed scene (all node types at once) ──────────────────────────

    #[test]
    fn mixed_scene_all_buffers_populated() {
        use ferrum_scene::{BlendMode, MarkBatch, MarkBatchKind, Panel, PathCmd};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect};
        use ferrum_scene::{FontWeight, TextAnchor, TextBaseline, TextStyle};

        let circle = SceneNode::Circle {
            cx: 100.0,
            cy: 100.0,
            r: 8.0,
            style: default_fill_stroke(),
        };
        let rect = SceneNode::Rect {
            x: 200.0,
            y: 50.0,
            w: 50.0,
            h: 100.0,
            style: default_fill_stroke(),
            corner_radius: 0.0,
        };
        let line = SceneNode::Line {
            x1: 50.0,
            y1: 300.0,
            x2: 400.0,
            y2: 300.0,
            style: default_stroke_style(),
        };
        let path = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 100.0, y: 200.0 },
                PathCmd::LineTo { x: 200.0, y: 100.0 },
                PathCmd::LineTo { x: 300.0, y: 200.0 },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        };
        let polygon = hex_polygon(350.0, 200.0, 20.0);
        let polyline = SceneNode::Polyline {
            points: vec![(50.0, 350.0), (150.0, 320.0), (250.0, 340.0)],
            style: default_stroke_style(),
        };
        let text = SceneNode::Text {
            x: 250.0,
            y: 30.0,
            content: "Title".to_string(),
            style: TextStyle {
                font_size: 16.0,
                font_weight: FontWeight::Bold,
                anchor: TextAnchor::Middle,
                baseline: TextBaseline::Top,
                angle: 0.0,
                color: Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                },
                opacity: 1.0,
                font_family: "sans-serif".to_string(),
            },
        };

        let scene = SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![text],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
                },
                clip: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
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
                    MarkBatch {
                        kind: MarkBatchKind::Point,
                        nodes: vec![circle],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Bar,
                        nodes: vec![rect],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Rule,
                        nodes: vec![line],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Area,
                        nodes: vec![path],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Polygon,
                        nodes: vec![polygon],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Line,
                        nodes: vec![polyline],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                ],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(data.circle_instances.len(), 1, "1 circle from point batch");
        assert_eq!(data.rect_instances.len(), 1, "1 rect from bar batch");
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "mesh from line+path+polygon+polyline"
        );
        assert!(
            !data.mesh_buffers.indices.is_empty(),
            "mesh indices from tessellation"
        );
        assert_eq!(data.text_elements.len(), 1, "1 text from title");

        // Verify all mesh vertices are finite
        for (i, v) in data.mesh_buffers.vertices.iter().enumerate() {
            assert!(
                v.position[0].is_finite() && v.position[1].is_finite(),
                "mixed scene: mesh vertex {i} has non-finite position"
            );
        }
    }

    // ── Binary packed instance round-trip ────────────────────────────

    fn build_packed_circle_stream(instances: &[CircleInstance]) -> Vec<u8> {
        build_packed_circle_stream_ex(0, 0, instances, 0, &[])
    }

    fn build_packed_rect_stream(instances: &[RectInstance]) -> Vec<u8> {
        build_packed_rect_stream_ex(0, 0, instances, 0, &[])
    }

    /// Build a packed circle stream with a 20-byte header and optional trailing data.
    fn build_packed_circle_stream_ex(
        panel_idx: u32,
        batch_idx: u32,
        instances: &[CircleInstance],
        flags: u32,
        trailing: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&panel_idx.to_le_bytes());
        buf.extend_from_slice(&batch_idx.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // kind=0 circle
        buf.extend_from_slice(&(instances.len() as u32).to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        for inst in instances {
            buf.extend_from_slice(bytemuck::bytes_of(inst));
        }
        buf.extend_from_slice(trailing);
        buf
    }

    /// Build a packed rect stream with a 20-byte header and optional trailing data.
    fn build_packed_rect_stream_ex(
        panel_idx: u32,
        batch_idx: u32,
        instances: &[RectInstance],
        flags: u32,
        trailing: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&panel_idx.to_le_bytes());
        buf.extend_from_slice(&batch_idx.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // kind=1 rect
        buf.extend_from_slice(&(instances.len() as u32).to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        for inst in instances {
            buf.extend_from_slice(bytemuck::bytes_of(inst));
        }
        buf.extend_from_slice(trailing);
        buf
    }

    #[test]
    fn binary_unpack_circle_round_trip() {
        let inst = CircleInstance {
            center: [100.0, 200.0],
            radius: 5.0,
            fill_color: [1.0, 0.0, 0.0, 0.8],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.5,
            opacity: 0.8,
            stroke_opacity: 0.6,
            stroke_dash: 1.0,
            angle: 45.0,
        };
        let packed = build_packed_circle_stream(&[inst]);
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);
        assert_eq!(circles.len(), 1);
        assert!(rects.is_empty());
        let c = &circles[0];
        assert!((c.center[0] - 100.0).abs() < 1e-6);
        assert!((c.radius - 5.0).abs() < 1e-6);
        assert!((c.opacity - 0.8).abs() < 1e-6);
        assert!((c.angle - 45.0).abs() < 1e-6);
        // flags=0 → no data_indices, no tooltip_bytes
        let m = meta.get(&(0, 0)).expect("meta entry should exist");
        assert!(m.data_indices.is_none());
        assert!(m.tooltip_bytes.is_none());
    }

    #[test]
    fn binary_unpack_rect_round_trip() {
        let instances = vec![
            RectInstance {
                position: [10.0, 20.0],
                size: [100.0, 50.0],
                corner_radius: 3.0,
                fill_color: [0.0, 1.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 0.5],
                stroke_width: 2.0,
                opacity: 0.9,
                stroke_opacity: 0.7,
                stroke_dash: 2.0,
                angle: 0.0,
            },
            RectInstance {
                position: [200.0, 30.0],
                size: [80.0, 60.0],
                corner_radius: 0.0,
                fill_color: [0.0, 0.0, 1.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 90.0,
            },
        ];
        let packed = build_packed_rect_stream(&instances);
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);
        assert_eq!(rects.len(), 2);
        assert!(circles.is_empty());
        assert!((rects[0].position[0] - 10.0).abs() < 1e-6);
        assert!((rects[1].angle - 90.0).abs() < 1e-6);
        let m = meta.get(&(0, 0)).expect("meta entry should exist");
        assert!(m.data_indices.is_none());
        assert!(m.tooltip_bytes.is_none());
    }

    #[test]
    fn binary_unpack_malformed_data_does_not_panic() {
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        // Truncated header (less than 20 bytes)
        unpack_binary_instances(&[0u8; 8], &mut circles, &mut rects, &mut meta);
        assert!(circles.is_empty());
        // Valid header but truncated instance data
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes()); // panel_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // batch_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // kind
        buf.extend_from_slice(&100u32.to_le_bytes()); // count
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        unpack_binary_instances(&buf, &mut circles, &mut rects, &mut meta);
        assert!(circles.is_empty());
    }

    #[test]
    fn load_scene_with_packed_uses_binary_sidecar() {
        use ferrum_scene::{BlendMode, MarkBatch, MarkBatchKind, Panel};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect, SceneGraph};

        let instances: Vec<CircleInstance> = (0..3)
            .map(|i| CircleInstance {
                center: [i as f32 * 100.0, 50.0],
                radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            })
            .collect();
        let packed = build_packed_circle_stream(&instances);

        let scene = SceneGraph {
            width: 400.0,
            height: 200.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 400.0,
                    h: 200.0,
                },
                clip: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 400.0,
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
                    nodes: vec![],
                    data_indices: None,
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
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene_with_packed(&scene, &packed);
        assert_eq!(data.circle_instances.len(), 3);
        assert!((data.circle_instances[0].fill_color[0] - 1.0).abs() < 1e-6);
        assert!((data.circle_instances[1].center[0] - 100.0).abs() < 1e-6);
        assert!((data.circle_instances[2].center[0] - 200.0).abs() < 1e-6);
    }

    // ── v2 header: data_indices and tooltips ─────────────────────────

    /// Helper: build a tooltip string table byte slice.
    fn build_tooltip_bytes(field_names: &[&str], rows: &[Vec<&str>]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(field_names.len() as u32).to_le_bytes());
        for name in field_names {
            let bytes = name.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        for row in rows {
            for val in row {
                let bytes = val.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
        }
        buf
    }

    /// Helper: build data_indices trailing bytes.
    fn build_data_indices_bytes(indices: &[u32]) -> Vec<u8> {
        let mut buf = Vec::new();
        for &idx in indices {
            buf.extend_from_slice(&idx.to_le_bytes());
        }
        buf
    }

    #[test]
    fn binary_unpack_with_data_indices() {
        let instances = vec![
            CircleInstance {
                center: [10.0, 20.0],
                radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [30.0, 40.0],
                radius: 7.0,
                fill_color: [0.0, 1.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];
        let trailing = build_data_indices_bytes(&[42, 99]);
        let packed = build_packed_circle_stream_ex(1, 2, &instances, HAS_DATA_INDICES, &trailing);

        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 2);
        let m = meta.get(&(1, 2)).expect("meta for (1,2) should exist");
        let indices = m
            .data_indices
            .as_ref()
            .expect("data_indices should be Some");
        assert_eq!(indices, &[42, 99]);
        assert!(m.tooltip_bytes.is_none());
    }

    #[test]
    fn binary_unpack_with_tooltips() {
        let instances = vec![CircleInstance {
            center: [10.0, 20.0],
            radius: 5.0,
            fill_color: [1.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];
        let tooltip_data = build_tooltip_bytes(&["x", "y"], &[vec!["1.23", "4.56"]]);
        let packed = build_packed_circle_stream_ex(0, 0, &instances, HAS_TOOLTIPS, &tooltip_data);

        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 1);
        let m = meta.get(&(0, 0)).expect("meta for (0,0) should exist");
        assert!(m.data_indices.is_none());
        let tb = m
            .tooltip_bytes
            .as_ref()
            .expect("tooltip_bytes should be Some");
        assert_eq!(tb, &tooltip_data, "tooltip bytes should match input");
    }

    #[test]
    fn binary_unpack_with_data_indices_and_tooltips() {
        let instances = vec![
            CircleInstance {
                center: [10.0, 20.0],
                radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
            CircleInstance {
                center: [30.0, 40.0],
                radius: 7.0,
                fill_color: [0.0, 1.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            },
        ];
        let mut trailing = build_data_indices_bytes(&[10, 20]);
        let tooltip_data = build_tooltip_bytes(
            &["name", "value"],
            &[vec!["alpha", "100"], vec!["beta", "200"]],
        );
        trailing.extend_from_slice(&tooltip_data);

        let packed = build_packed_circle_stream_ex(
            0,
            0,
            &instances,
            HAS_DATA_INDICES | HAS_TOOLTIPS,
            &trailing,
        );

        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 2);
        let m = meta.get(&(0, 0)).expect("meta should exist");
        let indices = m.data_indices.as_ref().expect("data_indices");
        assert_eq!(indices, &[10, 20]);
        let tb = m.tooltip_bytes.as_ref().expect("tooltip_bytes");
        assert_eq!(tb, &tooltip_data);
    }

    // ── F2 regression: truncated tooltip table must not panic ─────────
    //
    // Pre-fix: the scan loop advances `offset += 4 + slen` without clamping,
    // so `data[scan_start..offset]` panics when offset > data.len().
    // Post-fix: `offset.min(data.len())` clamps the slice end.

    /// A malformed/truncated tooltip section (slen overruns the buffer) must
    /// not panic — `unpack_binary_instances` must return gracefully with the
    /// tooltip bytes bounded by the buffer length.
    #[test]
    fn binary_unpack_truncated_tooltip_does_not_panic() {
        // Build a well-formed header + circle instances.
        let inst = CircleInstance {
            center: [1.0, 2.0],
            radius: 3.0,
            fill_color: [1.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };

        // Build a tooltip section that claims slen=9999 but the buffer ends immediately.
        // Format: [num_fields=1 u32] [field_name_len=9999 u32] <-- then EOF.
        let mut truncated_tooltip: Vec<u8> = Vec::new();
        truncated_tooltip.extend_from_slice(&1u32.to_le_bytes()); // num_fields = 1
        truncated_tooltip.extend_from_slice(&9999u32.to_le_bytes()); // slen = 9999 (overruns)
                                                                     // No actual string bytes — the buffer is truncated here.

        let packed = build_packed_circle_stream_ex(0, 0, &[inst], HAS_TOOLTIPS, &truncated_tooltip);

        // This must not panic. Pre-fix: offset overruns, slice panics.
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        // The circle was loaded before the tooltip scan; it should be present.
        assert_eq!(
            circles.len(),
            1,
            "circle must be loaded before tooltip scan"
        );
        // The meta was inserted; tooltip_bytes (if Some) must be bounded by data.len().
        if let Some(m) = meta.get(&(0, 0)) {
            if let Some(tb) = &m.tooltip_bytes {
                assert!(
                    tb.len() <= packed.len(),
                    "tooltip_bytes len {} must not exceed buffer len {}",
                    tb.len(),
                    packed.len()
                );
            }
        }
    }

    // ── F3 regression: instance_count reflects actual loaded count ─────
    //
    // When bytemuck::try_cast_slice fails (e.g. misalignment), the `if let Ok`
    // block is skipped but pre-fix code still recorded instance_count: count
    // (the header value), creating phantom references in hit-testing.
    // Post-fix: instance_count = loaded_count (circles.len() - start), which
    // is 0 when the cast failed.
    //
    // Note on constructing a genuine cast-failure: `bytemuck::try_cast_slice`
    // for CircleInstance (align=4) fails when the slice start is not 4-byte
    // aligned. We cannot force this via safe Rust's `Vec<u8>` allocation (which
    // always aligns to the element's alignment requirements). The alternative
    // approach — constructing a `&[u8]` offset by 1 byte — requires unsafe.
    // Instead we assert the success-path invariant: on a well-formed buffer,
    // instance_count == count == circles.len() - start. This is the critical
    // property that both paths must satisfy; the cast-failure path cannot be
    // constructed without unsafe and the host guarantee is alignment always holds
    // for Vec-allocated buffers.

    /// On the success path, recorded instance_count equals the number of
    /// instances actually pushed into the circles buffer.
    #[test]
    fn binary_unpack_instance_count_matches_loaded() {
        let instances: Vec<CircleInstance> = (0..5)
            .map(|i| CircleInstance {
                center: [i as f32 * 10.0, 0.0],
                radius: 2.0,
                fill_color: [1.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_opacity: 1.0,
                stroke_dash: 0.0,
                angle: 0.0,
            })
            .collect();

        let packed = build_packed_circle_stream_ex(0, 0, &instances, 0, &[]);
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 5);
        let m = meta.get(&(0, 0)).expect("meta must exist");
        assert_eq!(
            m.instance_count,
            circles.len() - m.instance_start,
            "instance_count must equal the number of instances actually loaded"
        );
        assert_eq!(
            m.instance_count, 5,
            "instance_count must equal header count on success"
        );
    }

    // ── parse_tooltip_json ──────────────────────────────────────────────

    #[test]
    fn parse_tooltip_json_single_row() {
        let bytes = build_tooltip_bytes(&["x", "y"], &[vec!["1.23", "4.56"]]);
        let json = parse_tooltip_json(&bytes, 0);
        assert_eq!(
            json,
            r#"{"fields":[{"name":"x","value":"1.23"},{"name":"y","value":"4.56"}]}"#,
        );
    }

    #[test]
    fn parse_tooltip_json_second_row() {
        let bytes = build_tooltip_bytes(
            &["a", "b"],
            &[vec!["10", "20"], vec!["30", "40"], vec!["50", "60"]],
        );
        let json = parse_tooltip_json(&bytes, 1);
        assert_eq!(
            json,
            r#"{"fields":[{"name":"a","value":"30"},{"name":"b","value":"40"}]}"#,
        );
    }

    #[test]
    fn parse_tooltip_json_last_row() {
        let bytes = build_tooltip_bytes(&["col"], &[vec!["first"], vec!["second"], vec!["third"]]);
        let json = parse_tooltip_json(&bytes, 2);
        assert_eq!(json, r#"{"fields":[{"name":"col","value":"third"}]}"#,);
    }

    #[test]
    fn parse_tooltip_json_out_of_range_returns_empty() {
        let bytes = build_tooltip_bytes(&["x"], &[vec!["1"]]);
        let json = parse_tooltip_json(&bytes, 5);
        assert_eq!(json, "{}");
    }

    #[test]
    fn parse_tooltip_json_empty_bytes_returns_empty() {
        let json = parse_tooltip_json(&[], 0);
        assert_eq!(json, "{}");
    }

    // ── sRGB round-trip and opt_color_to_f32 regression tests ──────────

    /// Inverse of srgb_to_linear: convert linear light back to sRGB.
    /// Used only in tests to verify round-trip accuracy.
    fn linear_to_srgb(l: f32) -> f32 {
        if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        }
    }

    // Test 9: srgb_to_linear round-trip accuracy for known sRGB values.
    #[test]
    fn srgb_to_linear_round_trip_accuracy() {
        let test_values: [u8; 5] = [0, 64, 128, 192, 255];
        for &srgb_byte in &test_values {
            let s = srgb_byte as f32 / 255.0;
            let linear = srgb_to_linear(s);
            let back = linear_to_srgb(linear);
            assert!(
                (back - s).abs() < 1e-5,
                "round-trip failed for sRGB byte {srgb_byte}: {s} → linear {linear} → {back}, delta {}",
                (back - s).abs()
            );
        }
    }

    // Test 10: opt_color_to_f32 produces linear values (catches double-gamma bug).
    #[test]
    fn opt_color_to_f32_produces_linear_mid_grey() {
        let mid_grey = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };
        let result = opt_color_to_f32(Some(&mid_grey), 1.0);
        // sRGB 128/255 ≈ 0.502 → linear ≈ 0.216.
        // If double-gamma were applied, the value would be ~0.0397 (way too dark).
        // If sRGB were passed through raw, the value would be ~0.502 (way too bright).
        // The correct linear value is ~0.216.
        assert!(
            result[0] < 0.5,
            "opt_color_to_f32 r=128 must produce linear < 0.5 (not raw sRGB), got {}",
            result[0]
        );
        assert!(
            result[0] > 0.1,
            "opt_color_to_f32 r=128 must produce linear > 0.1 (not double-gamma), got {}",
            result[0]
        );
        assert!(
            (result[0] - 0.216).abs() < 0.01,
            "opt_color_to_f32 r=128 should produce ~0.216 (linear mid-grey), got {}",
            result[0]
        );
        // All RGB channels should be equal for a neutral grey.
        assert!(
            (result[0] - result[1]).abs() < 1e-6 && (result[1] - result[2]).abs() < 1e-6,
            "neutral grey must have equal RGB channels, got {:?}",
            result
        );
        // Alpha should be 1.0 (fully opaque) and NOT gamma-corrected.
        assert!(
            (result[3] - 1.0).abs() < 1e-6,
            "alpha must be 1.0 for a=255, opacity=1.0, got {}",
            result[3]
        );
    }

    #[test]
    fn parse_tooltip_json_escapes_quotes() {
        let bytes = build_tooltip_bytes(&["label"], &[vec![r#"say "hello""#]]);
        let json = parse_tooltip_json(&bytes, 0);
        assert_eq!(
            json,
            r#"{"fields":[{"name":"label","value":"say \"hello\""}]}"#,
        );
    }

    // ── bug_hunt: srgb_to_linear boundary and extreme values ─────────────

    #[test]
    fn bug_hunt_srgb_to_linear_negative_input() {
        // Negative input: the sRGB spec doesn't define this, but the function
        // should not panic and should return a negative or zero value.
        let result = srgb_to_linear(-0.5);
        assert!(result.is_finite(), "srgb_to_linear(-0.5) must be finite");
    }

    #[test]
    fn bug_hunt_srgb_to_linear_above_one() {
        // Input > 1.0: out-of-spec but should not panic.
        let result = srgb_to_linear(1.5);
        assert!(result.is_finite(), "srgb_to_linear(1.5) must be finite");
        assert!(result > 1.0, "srgb_to_linear(1.5) must be > 1.0");
    }

    #[test]
    fn bug_hunt_srgb_to_linear_at_knee_point() {
        // The knee point is at 0.04045 where the two branches meet.
        // Both branches should produce the same value (continuity).
        let below = srgb_to_linear(0.04045);
        let above = srgb_to_linear(0.04046);
        // The two branches should produce nearly the same value at the knee.
        assert!(
            (below - above).abs() < 0.001,
            "srgb_to_linear must be continuous at knee: below={below}, above={above}"
        );
    }

    #[test]
    fn bug_hunt_color_to_linear_full_white() {
        let white = Color {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        let result = color_to_linear(&white, 1.0);
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "white r must be ~1.0 linear"
        );
        assert!(
            (result[1] - 1.0).abs() < 1e-5,
            "white g must be ~1.0 linear"
        );
        assert!(
            (result[2] - 1.0).abs() < 1e-5,
            "white b must be ~1.0 linear"
        );
        assert!((result[3] - 1.0).abs() < 1e-5, "white a must be ~1.0");
    }

    #[test]
    fn bug_hunt_color_to_linear_full_black() {
        let black = Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        };
        let result = color_to_linear(&black, 1.0);
        assert!(result[0].abs() < 1e-7, "black r must be ~0.0 linear");
        assert!(result[1].abs() < 1e-7, "black g must be ~0.0 linear");
        assert!(result[2].abs() < 1e-7, "black b must be ~0.0 linear");
    }

    #[test]
    fn bug_hunt_color_to_linear_opacity_scales_alpha() {
        let color = Color {
            r: 128,
            g: 128,
            b: 128,
            a: 128,
        };
        let result = color_to_linear(&color, 0.5);
        // a = (128/255) * 0.5 = ~0.251
        let expected_a = (128.0 / 255.0) * 0.5;
        assert!(
            (result[3] - expected_a).abs() < 0.01,
            "alpha must combine color alpha with opacity: expected ~{expected_a}, got {}",
            result[3]
        );
    }

    // ── bug_hunt: parse_tooltip_json edge cases ─────────────────────────────

    #[test]
    fn bug_hunt_parse_tooltip_json_truncated_field_name() {
        // Header says 1 field, but the string length exceeds the buffer.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_fields = 1
        bytes.extend_from_slice(&100u32.to_le_bytes()); // string length = 100 (but buffer is short)
        bytes.extend_from_slice(b"short");
        let result = parse_tooltip_json(&bytes, 0);
        assert_eq!(result, "{}", "truncated field name must return empty JSON");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_zero_fields() {
        // num_fields = 0 must return empty JSON.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let result = parse_tooltip_json(&bytes, 0);
        assert_eq!(result, "{}", "zero fields must return empty JSON");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_row_idx_exactly_at_count() {
        // 2 rows, requesting row 2 (out of bounds)
        let bytes = build_tooltip_bytes(&["x"], &[vec!["1"], vec!["2"]]);
        let result = parse_tooltip_json(&bytes, 2);
        assert_eq!(result, "{}", "row_idx == count must return empty JSON");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_large_row_idx() {
        let bytes = build_tooltip_bytes(&["x"], &[vec!["val"]]);
        let result = parse_tooltip_json(&bytes, 999999);
        assert_eq!(result, "{}", "very large row_idx must return empty JSON");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_utf8_content() {
        // Unicode content in field names and values
        let bytes = build_tooltip_bytes(&["name"], &[vec!["hello world"]]);
        let result = parse_tooltip_json(&bytes, 0);
        assert!(
            result.contains("hello world"),
            "ASCII content must appear in JSON"
        );
    }

    // ── tooltip_field_value: single-field lookup for packed legend matching ──

    #[test]
    fn tooltip_field_value_reads_named_column() {
        // Two fields, three rows; pick the second column on each row.
        let bytes = build_tooltip_bytes(
            &["x", "cat"],
            &[vec!["1", "a"], vec!["2", "b"], vec!["3", "a"]],
        );
        assert_eq!(tooltip_field_value(&bytes, 0, "cat").as_deref(), Some("a"));
        assert_eq!(tooltip_field_value(&bytes, 1, "cat").as_deref(), Some("b"));
        assert_eq!(tooltip_field_value(&bytes, 2, "cat").as_deref(), Some("a"));
        // First column still reachable.
        assert_eq!(tooltip_field_value(&bytes, 1, "x").as_deref(), Some("2"));
    }

    #[test]
    fn tooltip_field_value_missing_field_is_none() {
        let bytes = build_tooltip_bytes(&["x"], &[vec!["1"]]);
        assert_eq!(tooltip_field_value(&bytes, 0, "cat"), None);
    }

    #[test]
    fn tooltip_field_value_out_of_range_row_is_none() {
        let bytes = build_tooltip_bytes(&["cat"], &[vec!["a"]]);
        assert_eq!(tooltip_field_value(&bytes, 5, "cat"), None);
    }

    #[test]
    fn tooltip_field_value_empty_bytes_is_none() {
        assert_eq!(tooltip_field_value(&[], 0, "cat"), None);
    }

    // ── bug_hunt: unpack_binary_instances edge cases ─────────────────────────

    #[test]
    fn bug_hunt_unpack_empty_data_is_noop() {
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&[], &mut circles, &mut rects, &mut meta);
        assert!(circles.is_empty());
        assert!(rects.is_empty());
        assert!(meta.is_empty());
    }

    #[test]
    fn bug_hunt_unpack_unknown_kind_stops_parsing() {
        // kind=99 is unknown; parser should stop at this batch.
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes()); // panel_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // batch_idx
        buf.extend_from_slice(&99u32.to_le_bytes()); // kind = unknown
        buf.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&buf, &mut circles, &mut rects, &mut meta);
        assert!(circles.is_empty(), "unknown kind must not produce circles");
        assert!(rects.is_empty(), "unknown kind must not produce rects");
    }

    #[test]
    fn bug_hunt_load_scene_empty_scenegraph() {
        use ferrum_scene::{InteractionConfig, SceneGraph};
        let scene = SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };
        let data = load_scene(&scene);
        assert!(data.circle_instances.is_empty());
        assert!(data.rect_instances.is_empty());
        assert!(data.text_elements.is_empty());
        assert!((data.width - 100.0).abs() < 1e-3);
        assert!((data.height - 100.0).abs() < 1e-3);
    }

    // ── B2: fill_opacity is used for fill color alpha, not overall opacity ──

    /// A circle with fill_opacity=0.5 and opacity=1.0 must produce a fill
    /// color with alpha ≈ 0.5, not 1.0 (the overall opacity value).
    #[test]
    fn b2_circle_fill_color_uses_fill_opacity_not_opacity() {
        use ferrum_scene::{Color, FillStroke, SceneNode};

        let style = FillStroke {
            fill: Some(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 0.5, // half-transparent fill
            angle: 0.0,
        };

        let nodes = vec![SceneNode::Circle {
            cx: 10.0,
            cy: 10.0,
            r: 5.0,
            style,
        }];

        let mut collector = SceneCollector::new();
        collector.collect_mark(&nodes, false, None, 0, None, None, 0);

        assert_eq!(collector.circles.len(), 1);
        // fill_color alpha must reflect fill_opacity (0.5), not overall opacity (1.0).
        assert!(
            (collector.circles[0].fill_color[3] - 0.5).abs() < 0.02,
            "fill_color alpha must use fill_opacity (0.5), got {}",
            collector.circles[0].fill_color[3]
        );
        // overall opacity field must reflect the opacity field (1.0).
        assert!(
            (collector.circles[0].opacity - 1.0).abs() < 0.01,
            "instance opacity must reflect style.opacity (1.0), got {}",
            collector.circles[0].opacity
        );
    }

    /// Same check for Rect nodes.
    #[test]
    fn b2_rect_fill_color_uses_fill_opacity_not_opacity() {
        use ferrum_scene::{Color, FillStroke, SceneNode};

        let style = FillStroke {
            fill: Some(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 0.25, // quarter-transparent fill
            angle: 0.0,
        };

        let nodes = vec![SceneNode::Rect {
            x: 0.0,
            y: 0.0,
            w: 20.0,
            h: 10.0,
            corner_radius: 0.0,
            style,
        }];

        let mut collector = SceneCollector::new();
        collector.collect_mark(&nodes, false, None, 0, None, None, 0);

        assert_eq!(collector.rects.len(), 1);
        assert!(
            (collector.rects[0].fill_color[3] - 0.25).abs() < 0.02,
            "rect fill_color alpha must use fill_opacity (0.25), got {}",
            collector.rects[0].fill_color[3]
        );
        assert!(
            (collector.rects[0].opacity - 1.0).abs() < 0.01,
            "rect instance opacity must reflect style.opacity (1.0), got {}",
            collector.rects[0].opacity
        );
    }

    // ── B3: draw commands populated with correct blend mode ──────────

    /// Helper: build a scene with two panels, each having one normal and one
    /// additive circle batch, to test that draw_commands correctly records
    /// per-batch blend mode and instance ranges.
    fn make_blend_test_scene() -> SceneGraph {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, InteractionConfig, MarkBatch, MarkBatchKind, Panel,
            Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };

        let circle_a = SceneNode::Circle {
            cx: 10.0,
            cy: 10.0,
            r: 5.0,
            style: style.clone(),
        };
        let circle_b = SceneNode::Circle {
            cx: 20.0,
            cy: 20.0,
            r: 5.0,
            style: style.clone(),
        };
        let circle_c = SceneNode::Circle {
            cx: 30.0,
            cy: 30.0,
            r: 5.0,
            style,
        };

        SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
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
                marks: vec![
                    MarkBatch {
                        kind: MarkBatchKind::Point,
                        nodes: vec![circle_a],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Point,
                        nodes: vec![circle_b, circle_c],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Additive,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                ],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        }
    }

    #[test]
    fn draw_commands_created_for_each_batch() {
        let scene = make_blend_test_scene();
        let data = load_scene(&scene);

        // 3 circles total across 2 batches.
        assert_eq!(data.circle_instances.len(), 3);

        // Should have at least 2 draw commands for the 2 mark batches.
        let circle_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .collect();
        assert_eq!(
            circle_cmds.len(),
            2,
            "expected 2 circle draw commands, got {}",
            circle_cmds.len()
        );
    }

    #[test]
    fn draw_commands_normal_batch_not_additive() {
        let scene = make_blend_test_scene();
        let data = load_scene(&scene);

        let circle_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .collect();

        // First circle batch: 1 circle, normal blend.
        assert_eq!(circle_cmds[0].instance_count, 1);
        assert!(
            !circle_cmds[0].additive,
            "first batch should use normal blend"
        );
    }

    #[test]
    fn draw_commands_additive_batch_flagged() {
        let scene = make_blend_test_scene();
        let data = load_scene(&scene);

        let circle_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .collect();

        // Second circle batch: 2 circles, additive blend.
        assert_eq!(circle_cmds[1].instance_count, 2);
        assert!(
            circle_cmds[1].additive,
            "second batch should use additive blend"
        );
    }

    #[test]
    fn draw_commands_instance_ranges_are_contiguous() {
        let scene = make_blend_test_scene();
        let data = load_scene(&scene);

        let circle_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .collect();

        // First batch starts at 0, second batch starts where first ended.
        assert_eq!(circle_cmds[0].instance_start, 0);
        assert_eq!(
            circle_cmds[1].instance_start,
            circle_cmds[0].instance_start + circle_cmds[0].instance_count,
            "second batch should start where first batch ended"
        );
    }

    #[test]
    fn draw_commands_cover_all_instances() {
        let scene = make_blend_test_scene();
        let data = load_scene(&scene);

        let total_circles: u32 = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .map(|c| c.instance_count)
            .sum();
        assert_eq!(
            total_circles,
            data.circle_instances.len() as u32,
            "draw commands must cover all circle instances"
        );
    }

    #[test]
    fn draw_commands_rect_batch_with_blend() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, InteractionConfig, MarkBatch, MarkBatchKind, Panel,
            Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };

        let rect_node = SceneNode::Rect {
            x: 10.0,
            y: 10.0,
            w: 20.0,
            h: 30.0,
            style,
            corner_radius: 0.0,
        };

        let scene = SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
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
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Bar,
                    nodes: vec![rect_node],
                    data_indices: None,
                    tooltips: None,
                    hrefs: None,
                    descriptions: None,
                    keys: None,
                    blend: BlendMode::Additive,
                    stroke_cap: None,
                    stroke_join: None,
                    packed_instances: None,
                    y_slot: 0,
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        let rect_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Rect)
            .collect();
        assert_eq!(rect_cmds.len(), 1);
        assert!(rect_cmds[0].additive, "rect batch should be additive");
        assert_eq!(rect_cmds[0].instance_count, 1);
    }

    // ── B3: stroke color uses raw alpha, no opacity baked ─────────────

    #[test]
    fn circle_stroke_color_uses_raw_alpha_not_opacity() {
        // opacity=0.5, stroke_opacity=0.8
        // Stroke color alpha must be raw (color.a/255 * 1.0), NOT baked with opacity.
        // The shader applies stroke_opacity and opacity independently.
        let style = FillStroke {
            fill: Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke_width: 2.0,
            opacity: 0.5,
            stroke_opacity: 0.8,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };
        let node = SceneNode::Circle {
            cx: 50.0,
            cy: 50.0,
            r: 10.0,
            style,
        };
        let scene = make_scene_with_nodes(MarkBatchKind::Point, vec![node]);
        let data = load_scene(&scene);
        let ci = &data.circle_instances[0];
        // stroke_color.a should be linearized raw alpha ≈ color_to_linear(black, 1.0)[3] = 1.0
        // NOT color_to_linear(black, 0.5)[3] = 0.5 (the old buggy behavior)
        assert!(
            (ci.stroke_color[3] - 1.0).abs() < 1e-5,
            "circle stroke alpha must be raw (1.0), not opacity-baked; got {}",
            ci.stroke_color[3]
        );
    }

    #[test]
    fn rect_stroke_color_uses_raw_alpha_not_opacity() {
        let style = FillStroke {
            fill: Some(Color {
                r: 0,
                g: 128,
                b: 255,
                a: 255,
            }),
            stroke: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 200,
            }),
            stroke_width: 1.0,
            opacity: 0.5,
            stroke_opacity: 0.8,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };
        let node = SceneNode::Rect {
            x: 10.0,
            y: 20.0,
            w: 40.0,
            h: 30.0,
            style,
            corner_radius: 0.0,
        };
        let scene = make_scene_with_nodes(MarkBatchKind::Bar, vec![node]);
        let data = load_scene(&scene);
        let ri = &data.rect_instances[0];
        // stroke_color.a = (200/255)*1.0 (raw), linearized alpha is unchanged
        let expected_raw = 200.0_f32 / 255.0;
        assert!(
            (ri.stroke_color[3] - expected_raw).abs() < 1e-5,
            "rect stroke alpha must be raw ({expected_raw}), not opacity-baked; got {}",
            ri.stroke_color[3]
        );
    }

    // ── M3: SceneCollector produces the same output as load_scene ────────

    /// Verify that SceneCollector via collect_mark produces the same instance
    /// counts and per-field values as the full load_scene path for a mixed-mark
    /// scene (circles + rects + mesh nodes + text).
    #[test]
    fn test_scene_collector_produces_same_output() {
        use ferrum_scene::{FontWeight, PathCmd, TextAnchor, TextBaseline, TextStyle};

        // Build a scene with one circle batch (Normal), one rect batch (Additive),
        // and one path batch (Normal), plus a title text node.
        let circle_style = FillStroke {
            fill: Some(Color {
                r: 200,
                g: 100,
                b: 50,
                a: 255,
            }),
            stroke: Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke_width: 1.5,
            opacity: 0.9,
            stroke_opacity: 0.7,
            fill_opacity: 0.8,
            stroke_dash: Some(vec![6.0, 3.0]),
            angle: 15.0,
        };
        let rect_style = FillStroke {
            fill: Some(Color {
                r: 50,
                g: 150,
                b: 200,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };
        let circle_node = SceneNode::Circle {
            cx: 80.0,
            cy: 80.0,
            r: 12.0,
            style: circle_style,
        };
        let rect_node = SceneNode::Rect {
            x: 50.0,
            y: 50.0,
            w: 80.0,
            h: 40.0,
            style: rect_style,
            corner_radius: 3.0,
        };
        let path_node = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 10.0, y: 200.0 },
                PathCmd::LineTo { x: 100.0, y: 100.0 },
                PathCmd::LineTo { x: 200.0, y: 200.0 },
                PathCmd::Close,
            ],
            style: default_fill_stroke(),
            closed: true,
        };
        let text_style = TextStyle {
            font_size: 14.0,
            font_weight: FontWeight::Normal,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        };
        let title_node = SceneNode::Text {
            x: 150.0,
            y: 20.0,
            content: "Title".to_string(),
            style: text_style,
        };

        // Build the scene graph.
        use ferrum_scene::{
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect,
        };
        let scene = SceneGraph {
            width: 300.0,
            height: 250.0,
            background: None,
            title: vec![title_node],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 30.0,
                    y: 20.0,
                    w: 240.0,
                    h: 200.0,
                },
                clip: Rect {
                    x: 30.0,
                    y: 20.0,
                    w: 240.0,
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
                marks: vec![
                    MarkBatch {
                        kind: MarkBatchKind::Point,
                        nodes: vec![circle_node],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Bar,
                        nodes: vec![rect_node],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Additive,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Area,
                        nodes: vec![path_node],
                        data_indices: None,
                        tooltips: None,
                        hrefs: None,
                        descriptions: None,
                        keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None,
                        stroke_join: None,
                        packed_instances: None,
                        y_slot: 0,
                    },
                ],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        // Load via the full load_scene path (which now uses SceneCollector internally).
        let data = load_scene(&scene);

        // Verify instance counts.
        assert_eq!(data.circle_instances.len(), 1, "exactly 1 circle");
        assert_eq!(data.rect_instances.len(), 1, "exactly 1 rect");
        assert_eq!(data.text_elements.len(), 1, "exactly 1 text (title)");
        assert!(
            !data.mesh_buffers.vertices.is_empty(),
            "path batch produces mesh vertices"
        );
        assert!(
            !data.mesh_buffers.indices.is_empty(),
            "path batch produces mesh indices"
        );

        // Circle instance fields are correctly transferred.
        let ci = &data.circle_instances[0];
        assert!((ci.center[0] - 80.0).abs() < 1e-3, "circle cx");
        assert!((ci.center[1] - 80.0).abs() < 1e-3, "circle cy");
        assert!((ci.radius - 12.0).abs() < 1e-3, "circle radius");
        assert!((ci.opacity - 0.9).abs() < 1e-3, "circle opacity");
        assert!(
            (ci.stroke_opacity - 0.7).abs() < 1e-3,
            "circle stroke_opacity"
        );
        assert!(
            (ci.stroke_dash - 1.0).abs() < 1e-3,
            "circle stroke_dash index 1 (dashed)"
        );
        assert!((ci.angle - 15.0).abs() < 1e-3, "circle angle");

        // Rect instance fields are correctly transferred.
        let ri = &data.rect_instances[0];
        assert!((ri.position[0] - 50.0).abs() < 1e-3, "rect x");
        assert!((ri.corner_radius - 3.0).abs() < 1e-3, "rect corner_radius");

        // Draw commands: Normal circle (is_mark, not additive), Additive rect (is_mark, additive).
        let circle_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle && c.is_mark)
            .collect();
        let rect_cmds: Vec<_> = data
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Rect && c.is_mark)
            .collect();
        assert_eq!(circle_cmds.len(), 1, "one circle draw command");
        assert!(!circle_cmds[0].additive, "circle batch is Normal blend");
        assert_eq!(rect_cmds.len(), 1, "one rect draw command");
        assert!(rect_cmds[0].additive, "rect batch is Additive blend");

        // Text element content is preserved.
        assert_eq!(data.text_elements[0].content, "Title");
        assert!((data.text_elements[0].x - 150.0).abs() < 1e-3);
    }

    // ── M9: annotation mesh is separate from static mesh ─────────────

    /// Annotation Line nodes must land in `annotation_mesh_buffers`, not
    /// `static_mesh_buffers`.  Grid Line nodes (non-annotation) must land in
    /// `static_mesh_buffers`, not `annotation_mesh_buffers`.
    ///
    /// This regression test guards the z-order fix: annotation lines from
    /// `annotate_hline`/`annotate_vline` should appear above data marks in
    /// WASM, matching SVG painter order.  The separation into a distinct mesh
    /// buffer is the mechanism that enables `render_frame` to draw annotation
    /// lines after mark batches.
    #[test]
    fn test_annotation_mesh_separate_from_static_mesh() {
        use ferrum_scene::{
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect,
            SceneGraph, SceneNode, StrokeStyle,
        };

        let stroke = StrokeStyle {
            color: Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
            width: 2.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        };

        // A grid line (non-annotation) and an annotation line.
        let grid_line = SceneNode::Line {
            x1: 50.0,
            y1: 0.0,
            x2: 50.0,
            y2: 400.0,
            style: stroke.clone(),
        };
        let annotation_line = SceneNode::Line {
            x1: 0.0,
            y1: 200.0,
            x2: 500.0,
            y2: 200.0,
            style: stroke.clone(),
        };

        let scene = SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
                },
                clip: Rect {
                    x: 50.0,
                    y: 10.0,
                    w: 400.0,
                    h: 350.0,
                },
                coord: CoordKind::Cartesian {
                    x_domain: None,
                    y_domain: None,
                    expand: true,
                    clip: true,
                    y_domains: Vec::new(),
                },
                grid: vec![grid_line],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Rule,
                    nodes: vec![],
                    data_indices: None,
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
                annotations: vec![annotation_line],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);

        // Grid line → static_mesh (not annotation_mesh).
        assert!(
            !data.static_mesh_buffers.vertices.is_empty(),
            "grid Line must produce static_mesh vertices"
        );

        // Annotation line → annotation_mesh (not static_mesh).
        assert!(
            !data.annotation_mesh_buffers.vertices.is_empty(),
            "annotation Line must produce annotation_mesh vertices"
        );
        assert!(
            !data.annotation_mesh_buffers.indices.is_empty(),
            "annotation Line must produce annotation_mesh indices"
        );

        // Verify that the static_mesh and annotation_mesh have different vertex
        // counts (both contribute but are in different buffers).  A grid
        // vertical line and an annotation horizontal line tessellate to the
        // same number of vertices, so we verify both buffers are non-empty and
        // that each contains exactly the geometry for one line.
        assert!(
            data.static_mesh_buffers.indices.len() >= 6,
            "grid line must tessellate to ≥6 indices in static_mesh; got {}",
            data.static_mesh_buffers.indices.len()
        );
        assert!(
            data.annotation_mesh_buffers.indices.len() >= 6,
            "annotation line must tessellate to ≥6 indices in annotation_mesh; got {}",
            data.annotation_mesh_buffers.indices.len()
        );

        // Mark mesh must be empty (no mark nodes were added).
        assert!(
            data.mesh_buffers.vertices.is_empty(),
            "mark mesh must be empty when no mark nodes are present"
        );
    }

    // ── Per-panel mark-mesh scissor ranges ───────────────────────────
    //
    // These tests cover the `MarkMeshPanel` mechanism added to fix the
    // zoom/pan geometry bleed bug: mark mesh geometry must be partitioned
    // by panel so render_frame can scissor each panel's draw to its own
    // plot area.

    /// A single-panel scene with mesh marks must produce exactly one
    /// `MarkMeshPanel` entry that spans the entire mark mesh index buffer.
    #[test]
    fn single_panel_mesh_scene_produces_one_mark_mesh_panel() {
        let nodes = vec![SceneNode::Line {
            x1: 50.0,
            y1: 50.0,
            x2: 300.0,
            y2: 200.0,
            style: default_stroke_style(),
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Rule, nodes);
        let data = load_scene(&scene);

        // At least some mesh indices must have been produced for the assertion
        // to be meaningful.
        assert!(
            !data.mesh_buffers.indices.is_empty(),
            "prerequisite: line must tessellate to non-empty mesh"
        );

        assert_eq!(
            data.mark_mesh_panels.len(),
            1,
            "one panel with mesh marks → one MarkMeshPanel entry"
        );

        let panel = &data.mark_mesh_panels[0];
        assert_eq!(
            panel.index_start, 0,
            "single panel: index_start must be 0 (buffer starts fresh)"
        );
        assert_eq!(
            panel.index_count,
            data.mesh_buffers.indices.len() as u32,
            "single panel: index_count must span the entire mesh index buffer"
        );
        // Plot area must match the panel's plot_area from make_scene_with_nodes.
        assert!((panel.plot_area[0] - 50.0).abs() < 1e-3, "plot_area x");
        assert!((panel.plot_area[1] - 10.0).abs() < 1e-3, "plot_area y");
        assert!((panel.plot_area[2] - 400.0).abs() < 1e-3, "plot_area w");
        assert!((panel.plot_area[3] - 350.0).abs() < 1e-3, "plot_area h");
    }

    /// A two-panel scene each with mesh marks must produce two `MarkMeshPanel`
    /// entries whose index ranges are contiguous, non-overlapping, and together
    /// cover the entire mark mesh index buffer.
    #[test]
    fn two_panel_mesh_scene_produces_two_mark_mesh_panels_covering_full_buffer() {
        use ferrum_scene::{BlendMode, MarkBatch, Panel};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect};

        let line_node = || SceneNode::Line {
            x1: 50.0,
            y1: 50.0,
            x2: 300.0,
            y2: 200.0,
            style: default_stroke_style(),
        };

        let scene = SceneGraph {
            width: 600.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![
                Panel {
                    id: 0,
                    plot_area: Rect {
                        x: 50.0,
                        y: 10.0,
                        w: 200.0,
                        h: 150.0,
                    },
                    clip: Rect {
                        x: 50.0,
                        y: 10.0,
                        w: 200.0,
                        h: 150.0,
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
                        kind: MarkBatchKind::Rule,
                        nodes: vec![line_node()],
                        data_indices: None,
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
                },
                Panel {
                    id: 1,
                    plot_area: Rect {
                        x: 310.0,
                        y: 10.0,
                        w: 200.0,
                        h: 150.0,
                    },
                    clip: Rect {
                        x: 310.0,
                        y: 10.0,
                        w: 200.0,
                        h: 150.0,
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
                        kind: MarkBatchKind::Rule,
                        nodes: vec![line_node()],
                        data_indices: None,
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
                },
            ],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);

        assert_eq!(
            data.mark_mesh_panels.len(),
            2,
            "two panels with mesh marks → two MarkMeshPanel entries"
        );

        let p0 = &data.mark_mesh_panels[0];
        let p1 = &data.mark_mesh_panels[1];

        // Ranges must be contiguous: panel 1 starts where panel 0 ends.
        assert_eq!(
            p1.index_start,
            p0.index_start + p0.index_count,
            "panel 1 index_start must immediately follow panel 0 range"
        );

        // Together they must cover the entire mesh index buffer.
        assert_eq!(
            p0.index_count + p1.index_count,
            data.mesh_buffers.indices.len() as u32,
            "combined panel ranges must cover the full mesh index buffer"
        );

        // Plot areas must match each panel.
        assert!((p0.plot_area[0] - 50.0).abs() < 1e-3, "panel 0 x");
        assert!((p1.plot_area[0] - 310.0).abs() < 1e-3, "panel 1 x");
    }

    /// A panel with only circle instances (no mesh nodes) must NOT appear in
    /// `mark_mesh_panels` — zero-count ranges are dropped.
    #[test]
    fn panel_with_no_mesh_marks_produces_no_mark_mesh_panel_entry() {
        // Circle nodes go to the instance buffer, not the mesh buffer.
        let circle_node = SceneNode::Circle {
            cx: 100.0,
            cy: 100.0,
            r: 5.0,
            style: default_fill_stroke(),
        };
        let scene = make_scene_with_nodes(MarkBatchKind::Point, vec![circle_node]);
        let data = load_scene(&scene);

        assert!(
            data.mark_mesh_panels.is_empty(),
            "circle-only panel must produce no MarkMeshPanel entries \
             (circles go to the instance buffer, not the mesh)"
        );
        // Sanity: the circle did land in instances.
        assert_eq!(data.circle_instances.len(), 1);
    }

    /// `SceneCollector::record_mark_mesh_panel` is the core primitive.
    /// Verify it correctly accumulates entries and drops zero-count panels.
    #[test]
    fn record_mark_mesh_panel_accumulates_and_drops_zeros() {
        let mut collector = SceneCollector::new();

        // Record a non-zero range for panel 0.
        collector.record_mark_mesh_panel(0, 30, [10.0, 20.0, 200.0, 150.0], 0, 0);
        assert_eq!(collector.mark_mesh_panels.len(), 1);
        assert_eq!(collector.mark_mesh_panels[0].index_start, 0);
        assert_eq!(collector.mark_mesh_panels[0].index_count, 30);
        assert_eq!(collector.mark_mesh_panels[0].panel_id, 0);

        // Record a zero-count range — must be dropped.
        collector.record_mark_mesh_panel(30, 30, [10.0, 200.0, 200.0, 150.0], 1, 0);
        assert_eq!(
            collector.mark_mesh_panels.len(),
            1,
            "zero-count panel must not be appended"
        );

        // Record a second non-zero range that follows the first (panel 2).
        collector.record_mark_mesh_panel(30, 55, [220.0, 20.0, 200.0, 150.0], 2, 0);
        assert_eq!(collector.mark_mesh_panels.len(), 2);
        assert_eq!(collector.mark_mesh_panels[1].index_start, 30);
        assert_eq!(collector.mark_mesh_panels[1].index_count, 25);
        assert_eq!(
            collector.mark_mesh_panels[1].panel_id, 2,
            "the recorded panel_id must be preserved (the zero-count panel 1 was dropped)"
        );

        // Verify plot areas are stored correctly.
        let pa0 = collector.mark_mesh_panels[0].plot_area;
        assert!((pa0[0] - 10.0).abs() < 1e-6, "panel 0 x");
        assert!((pa0[2] - 200.0).abs() < 1e-6, "panel 0 w");
        let pa1 = collector.mark_mesh_panels[1].plot_area;
        assert!((pa1[0] - 220.0).abs() < 1e-6, "panel 2 x");
    }

    /// FA-18: a two-panel scene must record distinct `panel_id`s on the
    /// mesh-panel entries AND on the mark draw commands, so the render loop can
    /// bind each panel's own transform slot. Without this, a non-uniform
    /// domain-rescale on one panel sheared its siblings.
    #[test]
    fn two_panel_scene_records_distinct_panel_ids() {
        use ferrum_scene::{
            BlendMode, CoordKind, FillStroke, InteractionConfig, MarkBatch, MarkBatchKind, Panel,
            Rect, SceneGraph, SceneNode, StrokeStyle,
        };

        let line_style = StrokeStyle {
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            width: 1.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        };
        let circle_style = FillStroke {
            fill: Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };

        // Each panel carries one Line (→ mark mesh) and one Circle (→ mark
        // instance draw command) so both per-panel tracks get populated.
        let mk_panel = |x_off: f64| Panel {
            id: 0,
            plot_area: Rect {
                x: x_off,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
            clip: Rect {
                x: x_off,
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
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Line,
                nodes: vec![
                    SceneNode::Line {
                        x1: x_off,
                        y1: 10.0,
                        x2: x_off + 50.0,
                        y2: 80.0,
                        style: line_style.clone(),
                    },
                    SceneNode::Circle {
                        cx: x_off + 25.0,
                        cy: 50.0,
                        r: 4.0,
                        style: circle_style.clone(),
                    },
                ],
                data_indices: None,
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
        };

        let scene = SceneGraph {
            width: 200.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![mk_panel(0.0), mk_panel(100.0)],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);

        assert_eq!(
            data.panel_count, 2,
            "two-panel scene must report panel_count == 2"
        );

        // One mesh slice per panel, carrying distinct panel_ids 0 and 1.
        assert_eq!(data.mark_mesh_panels.len(), 2, "one mesh slice per panel");
        assert_eq!(data.mark_mesh_panels[0].panel_id, 0);
        assert_eq!(data.mark_mesh_panels[1].panel_id, 1);

        // The mark circle draw commands must carry distinct panel_ids too.
        let mark_panel_ids: Vec<usize> = data
            .draw_commands
            .iter()
            .filter(|c| c.is_mark)
            .map(|c| c.panel_id)
            .collect();
        assert!(
            mark_panel_ids.contains(&0),
            "a mark command must belong to panel 0"
        );
        assert!(
            mark_panel_ids.contains(&1),
            "a mark command must belong to panel 1"
        );
    }

    // ── Task 3 gap-fix: non-identity `layout_scale` bake correctness ────
    //
    // Spec-review gap: no test previously exercised the bake path with a
    // non-uniform (sx != sy) `LayoutScale`. All tests below use
    // `sx = 2.0, sy = 8.0, tx = 10.0, ty = -5.0`, so the geometric-mean
    // scalar factor `sqrt(sx*sy) = 4.0` is distinguishable from both `sx`
    // and `sy` individually — a uniform scale (or `sx*sy == 1`) would hide
    // bugs where the wrong factor is applied to direction-independent
    // scalars (radius, corner_radius, stroke_width, font_size).

    /// A non-identity, non-uniform `LayoutScale` shared by the gap-fix tests
    /// below: `sx=2.0, sy=8.0, tx=10.0, ty=-5.0` → geometric mean `4.0`.
    fn gap_fix_layout_scale() -> LayoutScale {
        LayoutScale {
            sx: 2.0,
            sy: 8.0,
            tx: 10.0,
            ty: -5.0,
        }
    }

    fn gap_fix_fill_stroke(stroke_width: f64, angle: f64) -> ferrum_scene::FillStroke {
        ferrum_scene::FillStroke {
            fill: None,
            stroke: None,
            stroke_width,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle,
        }
    }

    fn gap_fix_stroke_style(width: f64) -> ferrum_scene::StrokeStyle {
        ferrum_scene::StrokeStyle {
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            width,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        }
    }

    fn gap_fix_text_style(font_size: f64, angle: f64) -> ferrum_scene::TextStyle {
        ferrum_scene::TextStyle {
            font_size,
            font_weight: ferrum_scene::FontWeight::Normal,
            anchor: ferrum_scene::TextAnchor::Start,
            baseline: ferrum_scene::TextBaseline::Alphabetic,
            angle,
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        }
    }

    #[test]
    fn scalar_scale_factor_is_geometric_mean_of_sx_sy() {
        let ls = gap_fix_layout_scale();
        assert!(
            (scalar_scale_factor(&ls) - 4.0).abs() < 1e-9,
            "got {}",
            scalar_scale_factor(&ls)
        );

        // Uniform scale: geometric mean equals the shared factor exactly.
        let uniform = LayoutScale {
            sx: 3.0,
            sy: 3.0,
            tx: 0.0,
            ty: 0.0,
        };
        assert!((scalar_scale_factor(&uniform) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn transform_node_rect_scales_axes_independently_and_scalar_by_geometric_mean() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Rect {
            x: 3.0,
            y: 4.0,
            w: 5.0,
            h: 6.0,
            style: gap_fix_fill_stroke(1.5, 30.0),
            corner_radius: 2.0,
        };
        match transform_node(&node, &ls) {
            SceneNode::Rect {
                x,
                y,
                w,
                h,
                style,
                corner_radius,
            } => {
                assert!((x - 16.0).abs() < 1e-9, "x: got {x}"); // 2*3+10
                assert!((y - 27.0).abs() < 1e-9, "y: got {y}"); // 8*4-5
                assert!((w - 10.0).abs() < 1e-9, "w: got {w}"); // 5*2 (sx)
                assert!((h - 48.0).abs() < 1e-9, "h: got {h}"); // 6*8 (sy)
                assert!(
                    (corner_radius - 8.0).abs() < 1e-9,
                    "corner_radius should scale by geometric mean 4.0, got {corner_radius}"
                );
                assert!(
                    (style.stroke_width - 6.0).abs() < 1e-9,
                    "stroke_width should scale by geometric mean 4.0, got {}",
                    style.stroke_width
                );
                assert_eq!(
                    style.angle, 30.0,
                    "rotation angle must pass through UNCHANGED (documented gap)"
                );
            }
            other => panic!("expected Rect, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_circle_radius_uses_geometric_mean_scale() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Circle {
            cx: 3.0,
            cy: 4.0,
            r: 2.0,
            style: gap_fix_fill_stroke(1.0, 0.0),
        };
        match transform_node(&node, &ls) {
            SceneNode::Circle { cx, cy, r, style } => {
                assert!((cx - 16.0).abs() < 1e-9, "cx: got {cx}");
                assert!((cy - 27.0).abs() < 1e-9, "cy: got {cy}");
                assert!(
                    (r - 8.0).abs() < 1e-9,
                    "radius should scale by sqrt(sx*sy)=4.0, got {r}"
                );
                assert!((style.stroke_width - 4.0).abs() < 1e-9);
            }
            other => panic!("expected Circle, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_line_scales_dash_pattern_by_geometric_mean() {
        // Regression (Task 3 quality review): dash-pattern pixel lengths must
        // scale with the same geometric-mean factor as stroke width — the SVG
        // walker's <g transform> scales dasharray automatically, so leaving
        // them unscaled would misrender dashed marks on ratio-fitted panels.
        let ls = gap_fix_layout_scale();
        let mut style = gap_fix_stroke_style(1.0);
        style.dash = Some(vec![4.0, 2.0]);
        let node = SceneNode::Line { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0, style };
        match transform_node(&node, &ls) {
            SceneNode::Line { style, .. } => {
                let dash = style.dash.expect("dash must survive the bake");
                // geometric mean of (2, 8) = 4 → [16, 8]
                assert!((dash[0] - 16.0).abs() < 1e-9, "dash[0]: got {}", dash[0]);
                assert!((dash[1] - 8.0).abs() < 1e-9, "dash[1]: got {}", dash[1]);
                assert!((style.width - 4.0).abs() < 1e-9, "width must still scale");
            }
            other => panic!("expected Line, got {other:?}"),
        }
    }

    #[test]
    fn transform_fill_stroke_scales_stroke_dash_by_geometric_mean() {
        let mut style = gap_fix_fill_stroke(1.0, 0.0);
        style.stroke_dash = Some(vec![3.0, 1.5]);
        let out = transform_fill_stroke(&style, 4.0);
        let dash = out.stroke_dash.expect("stroke_dash must survive the bake");
        assert!((dash[0] - 12.0).abs() < 1e-9, "dash[0]: got {}", dash[0]);
        assert!((dash[1] - 6.0).abs() < 1e-9, "dash[1]: got {}", dash[1]);
        assert!((out.stroke_width - 4.0).abs() < 1e-9, "stroke_width must still scale");
    }

    #[test]
    fn transform_node_line_scales_endpoints_and_stroke_width() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Line {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            style: gap_fix_stroke_style(1.0),
        };
        match transform_node(&node, &ls) {
            SceneNode::Line {
                x1,
                y1,
                x2,
                y2,
                style,
            } => {
                assert!((x1 - 12.0).abs() < 1e-9, "x1: got {x1}");
                assert!((y1 - 11.0).abs() < 1e-9, "y1: got {y1}");
                assert!((x2 - 16.0).abs() < 1e-9, "x2: got {x2}");
                assert!((y2 - 27.0).abs() < 1e-9, "y2: got {y2}");
                assert!(
                    (style.width - 4.0).abs() < 1e-9,
                    "stroke width: got {}",
                    style.width
                );
            }
            other => panic!("expected Line, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_text_scales_position_and_font_size_leaves_angle() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Text {
            x: 5.0,
            y: 6.0,
            content: "hi".to_string(),
            style: gap_fix_text_style(12.0, 45.0),
        };
        match transform_node(&node, &ls) {
            SceneNode::Text {
                x,
                y,
                content,
                style,
            } => {
                assert!((x - 20.0).abs() < 1e-9, "x: got {x}");
                assert!((y - 43.0).abs() < 1e-9, "y: got {y}");
                assert_eq!(content, "hi");
                assert!(
                    (style.font_size - 48.0).abs() < 1e-9,
                    "font_size should scale by sqrt(sx*sy)=4.0, got {}",
                    style.font_size
                );
                assert_eq!(
                    style.angle, 45.0,
                    "text rotation must pass through UNCHANGED (documented gap)"
                );
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_image_scales_position_and_size_per_axis() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Image {
            x: 1.0,
            y: 2.0,
            w: 3.0,
            h: 4.0,
            data: ferrum_scene::ImageData::Url {
                url: "x".to_string(),
            },
        };
        match transform_node(&node, &ls) {
            SceneNode::Image { x, y, w, h, .. } => {
                assert!((x - 12.0).abs() < 1e-9, "x: got {x}");
                assert!((y - 11.0).abs() < 1e-9, "y: got {y}");
                assert!((w - 6.0).abs() < 1e-9, "w should scale by sx=2.0, got {w}");
                assert!((h - 32.0).abs() < 1e-9, "h should scale by sy=8.0, got {h}");
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_polygon_scales_every_ring_point() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Polygon {
            rings: vec![vec![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]],
            style: gap_fix_fill_stroke(1.0, 0.0),
        };
        match transform_node(&node, &ls) {
            SceneNode::Polygon { rings, .. } => {
                assert_eq!(rings.len(), 1);
                let ring = &rings[0];
                assert!((ring[0][0] - 12.0).abs() < 1e-9);
                assert!((ring[0][1] - 3.0).abs() < 1e-9);
                assert!((ring[1][0] - 14.0).abs() < 1e-9);
                assert!((ring[1][1] - 11.0).abs() < 1e-9);
                assert!((ring[2][0] - 16.0).abs() < 1e-9);
                assert!((ring[2][1] - 19.0).abs() < 1e-9);
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_polyline_scales_points_and_stroke_width() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Polyline {
            points: vec![(1.0, 1.0), (2.0, 2.0)],
            style: gap_fix_stroke_style(1.0),
        };
        match transform_node(&node, &ls) {
            SceneNode::Polyline { points, style } => {
                assert!((points[0].0 - 12.0).abs() < 1e-9);
                assert!((points[0].1 - 3.0).abs() < 1e-9);
                assert!((points[1].0 - 14.0).abs() < 1e-9);
                assert!((points[1].1 - 11.0).abs() < 1e-9);
                assert!((style.width - 4.0).abs() < 1e-9);
            }
            other => panic!("expected Polyline, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_path_scales_commands_and_stroke_width() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 1.0, y: 1.0 },
                PathCmd::LineTo { x: 2.0, y: 2.0 },
            ],
            style: gap_fix_fill_stroke(1.0, 0.0),
            closed: true,
        };
        match transform_node(&node, &ls) {
            SceneNode::Path {
                commands,
                style,
                closed,
            } => {
                assert!(closed);
                assert!((style.stroke_width - 4.0).abs() < 1e-9);
                match commands[0] {
                    PathCmd::MoveTo { x, y } => {
                        assert!((x - 12.0).abs() < 1e-9);
                        assert!((y - 3.0).abs() < 1e-9);
                    }
                    ref other => panic!("expected MoveTo, got {other:?}"),
                }
                match commands[1] {
                    PathCmd::LineTo { x, y } => {
                        assert!((x - 14.0).abs() < 1e-9);
                        assert!((y - 11.0).abs() < 1e-9);
                    }
                    ref other => panic!("expected LineTo, got {other:?}"),
                }
            }
            other => panic!("expected Path, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_group_recurses_into_children() {
        let ls = gap_fix_layout_scale();
        let child = SceneNode::Circle {
            cx: 3.0,
            cy: 4.0,
            r: 2.0,
            style: gap_fix_fill_stroke(1.0, 0.0),
        };
        let node = SceneNode::Group {
            attrs: vec![],
            children: vec![child],
        };
        match transform_node(&node, &ls) {
            SceneNode::Group { children, .. } => {
                assert_eq!(children.len(), 1);
                match &children[0] {
                    SceneNode::Circle { cx, cy, r, .. } => {
                        assert!((cx - 16.0).abs() < 1e-9);
                        assert!((cy - 27.0).abs() < 1e-9);
                        assert!((r - 8.0).abs() < 1e-9);
                    }
                    other => panic!("expected Circle child, got {other:?}"),
                }
            }
            other => panic!("expected Group, got {other:?}"),
        }
    }

    #[test]
    fn transform_node_raw_passes_through_unchanged_documented_gap() {
        let ls = gap_fix_layout_scale();
        let node = SceneNode::Raw {
            svg: "<rect x=\"5\" y=\"6\" width=\"1\" height=\"1\"/>".to_string(),
            anchor: ferrum_scene::RawAnchor::Chrome,
        };
        let out = transform_node(&node, &ls);
        assert_eq!(
            out, node,
            "Raw fragments bake absolute coordinates into an opaque SVG string; the W4 \
             gap means they pass through the bake pass unchanged"
        );
    }

    #[test]
    fn transform_path_cmd_covers_all_variants_exact_coordinates() {
        let ls = gap_fix_layout_scale();

        match transform_path_cmd(&PathCmd::MoveTo { x: 1.0, y: 1.0 }, &ls) {
            PathCmd::MoveTo { x, y } => {
                assert!((x - 12.0).abs() < 1e-9);
                assert!((y - 3.0).abs() < 1e-9);
            }
            other => panic!("expected MoveTo, got {other:?}"),
        }

        match transform_path_cmd(&PathCmd::LineTo { x: 2.0, y: 2.0 }, &ls) {
            PathCmd::LineTo { x, y } => {
                assert!((x - 14.0).abs() < 1e-9);
                assert!((y - 11.0).abs() < 1e-9);
            }
            other => panic!("expected LineTo, got {other:?}"),
        }

        match transform_path_cmd(
            &PathCmd::QuadTo {
                cx: 3.0,
                cy: 3.0,
                x: 4.0,
                y: 4.0,
            },
            &ls,
        ) {
            PathCmd::QuadTo { cx, cy, x, y } => {
                assert!((cx - 16.0).abs() < 1e-9);
                assert!((cy - 19.0).abs() < 1e-9);
                assert!((x - 18.0).abs() < 1e-9);
                assert!((y - 27.0).abs() < 1e-9);
            }
            other => panic!("expected QuadTo, got {other:?}"),
        }

        match transform_path_cmd(
            &PathCmd::CubicTo {
                c1x: 1.0,
                c1y: 1.0,
                c2x: 2.0,
                c2y: 2.0,
                x: 3.0,
                y: 3.0,
            },
            &ls,
        ) {
            PathCmd::CubicTo {
                c1x,
                c1y,
                c2x,
                c2y,
                x,
                y,
            } => {
                assert!((c1x - 12.0).abs() < 1e-9);
                assert!((c1y - 3.0).abs() < 1e-9);
                assert!((c2x - 14.0).abs() < 1e-9);
                assert!((c2y - 11.0).abs() < 1e-9);
                assert!((x - 16.0).abs() < 1e-9);
                assert!((y - 19.0).abs() < 1e-9);
            }
            other => panic!("expected CubicTo, got {other:?}"),
        }

        // HLineTo carries only an x-coordinate: scaled by sx/tx; sy/ty must
        // not leak in (this is what a copy-paste bug in `ls.apply` would hide).
        match transform_path_cmd(&PathCmd::HLineTo { x: 5.0 }, &ls) {
            PathCmd::HLineTo { x } => assert!((x - 20.0).abs() < 1e-9, "got {x}"),
            other => panic!("expected HLineTo, got {other:?}"),
        }

        // VLineTo carries only a y-coordinate: scaled by sy/ty; sx/tx must not leak in.
        match transform_path_cmd(&PathCmd::VLineTo { y: 5.0 }, &ls) {
            PathCmd::VLineTo { y } => assert!((y - 35.0).abs() < 1e-9, "got {y}"),
            other => panic!("expected VLineTo, got {other:?}"),
        }

        match transform_path_cmd(
            &PathCmd::ArcTo {
                rx: 2.0,
                ry: 3.0,
                rotation: 15.0,
                large_arc: false,
                sweep: true,
                x: 6.0,
                y: 7.0,
            },
            &ls,
        ) {
            PathCmd::ArcTo {
                rx,
                ry,
                rotation,
                large_arc,
                sweep,
                x,
                y,
            } => {
                assert!(
                    (rx - 4.0).abs() < 1e-9,
                    "rx should scale by sx=2.0, got {rx}"
                );
                assert!(
                    (ry - 24.0).abs() < 1e-9,
                    "ry should scale by sy=8.0, got {ry}"
                );
                assert_eq!(
                    rotation, 15.0,
                    "arc rotation must pass through UNCHANGED (documented gap)"
                );
                assert!(!large_arc);
                assert!(sweep);
                assert!((x - 22.0).abs() < 1e-9);
                assert!((y - 51.0).abs() < 1e-9);
            }
            other => panic!("expected ArcTo, got {other:?}"),
        }

        assert_eq!(transform_path_cmd(&PathCmd::Close, &ls), PathCmd::Close);
    }

    #[test]
    fn transform_rect_bakes_exact_plot_area_coordinates() {
        let ls = gap_fix_layout_scale();
        let rect = ferrum_scene::Rect {
            x: 3.0,
            y: 4.0,
            w: 5.0,
            h: 6.0,
        };
        let out = transform_rect(&rect, &ls);
        assert!((out.x - 16.0).abs() < 1e-9, "x: got {}", out.x);
        assert!((out.y - 27.0).abs() < 1e-9, "y: got {}", out.y);
        assert!(
            (out.w - 10.0).abs() < 1e-9,
            "w should scale by sx=2.0, got {}",
            out.w
        );
        assert!(
            (out.h - 48.0).abs() < 1e-9,
            "h should scale by sy=8.0, got {}",
            out.h
        );
    }

    #[test]
    fn apply_layout_scale_to_packed_instances_bakes_exact_circle_and_rect_geometry() {
        use ferrum_scene::{
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect,
            SceneGraph,
        };

        let ls = gap_fix_layout_scale();

        let scene = SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
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
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![],
                    data_indices: None,
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
                layout_scale: ls,
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let mut circles = vec![CircleInstance {
            center: [3.0, 4.0],
            radius: 2.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];
        let mut rects = vec![RectInstance {
            position: [3.0, 4.0],
            size: [5.0, 6.0],
            corner_radius: 2.0,
            fill_color: [0.0; 4],
            stroke_color: [0.0; 4],
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        }];

        let mut batch_meta = HashMap::new();
        batch_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 1,
                data_indices: None,
                tooltip_bytes: None,
            },
        );
        batch_meta.insert(
            (0u32, 1u32),
            PackedBatchMeta {
                kind: DrawKind::Rect,
                instance_start: 0,
                instance_count: 1,
                data_indices: None,
                tooltip_bytes: None,
            },
        );

        apply_layout_scale_to_packed_instances(&scene, &mut circles, &mut rects, &batch_meta);

        let c = &circles[0];
        assert!(
            (c.center[0] - 16.0).abs() < 1e-4,
            "circle x: got {}",
            c.center[0]
        );
        assert!(
            (c.center[1] - 27.0).abs() < 1e-4,
            "circle y: got {}",
            c.center[1]
        );
        assert!(
            (c.radius - 8.0).abs() < 1e-4,
            "circle radius should scale by geometric mean 4.0, got {}",
            c.radius
        );
        assert!(
            (c.stroke_width - 4.0).abs() < 1e-4,
            "circle stroke_width: got {}",
            c.stroke_width
        );

        let r = &rects[0];
        assert!(
            (r.position[0] - 16.0).abs() < 1e-4,
            "rect x: got {}",
            r.position[0]
        );
        assert!(
            (r.position[1] - 27.0).abs() < 1e-4,
            "rect y: got {}",
            r.position[1]
        );
        assert!(
            (r.size[0] - 10.0).abs() < 1e-4,
            "rect w should scale by sx=2.0, got {}",
            r.size[0]
        );
        assert!(
            (r.size[1] - 48.0).abs() < 1e-4,
            "rect h should scale by sy=8.0, got {}",
            r.size[1]
        );
        assert!(
            (r.corner_radius - 8.0).abs() < 1e-4,
            "rect corner_radius: got {}",
            r.corner_radius
        );
        assert!(
            (r.stroke_width - 4.0).abs() < 1e-4,
            "rect stroke_width: got {}",
            r.stroke_width
        );
    }

    #[test]
    fn load_scene_bakes_non_identity_layout_scale_only_for_the_carrying_panel() {
        use ferrum_scene::{
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect,
            SceneGraph,
        };

        let ls = gap_fix_layout_scale();
        let circle_style = gap_fix_fill_stroke(1.0, 0.0);

        let mk_panel = |layout_scale: LayoutScale| Panel {
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
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![SceneNode::Circle {
                    cx: 3.0,
                    cy: 4.0,
                    r: 2.0,
                    style: circle_style.clone(),
                }],
                data_indices: None,
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
            layout_scale,
        };

        let scene = SceneGraph {
            width: 200.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![mk_panel(LayoutScale::identity()), mk_panel(ls)],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(
            data.circle_instances.len(),
            2,
            "one circle instance per panel"
        );

        // Panel 0 carries identity layout_scale: geometry must be untouched
        // (this is the identity no-op contract the bake path must preserve).
        let c0 = &data.circle_instances[0];
        assert!(
            (c0.center[0] - 3.0).abs() < 1e-4,
            "panel 0 x must be untouched, got {}",
            c0.center[0]
        );
        assert!(
            (c0.center[1] - 4.0).abs() < 1e-4,
            "panel 0 y must be untouched, got {}",
            c0.center[1]
        );
        assert!(
            (c0.radius - 2.0).abs() < 1e-4,
            "panel 0 radius must be untouched, got {}",
            c0.radius
        );

        // Panel 1 carries the non-identity layout_scale: geometry must be
        // baked exactly, matching the same formula verified in isolation
        // above (2*3+10, 8*4-5, radius * sqrt(2*8)).
        let c1 = &data.circle_instances[1];
        assert!(
            (c1.center[0] - 16.0).abs() < 1e-4,
            "panel 1 x: got {}",
            c1.center[0]
        );
        assert!(
            (c1.center[1] - 27.0).abs() < 1e-4,
            "panel 1 y: got {}",
            c1.center[1]
        );
        assert!(
            (c1.radius - 8.0).abs() < 1e-4,
            "panel 1 radius should scale by geometric mean 4.0, got {}",
            c1.radius
        );

        // Panel 1's plot_area (carried on its mark draw command) must also
        // be baked exactly, matching `transform_rect_bakes_exact_plot_area_coordinates`.
        let panel1_plot_area = data
            .draw_commands
            .iter()
            .find(|cmd| cmd.is_mark && cmd.panel_id == 1)
            .and_then(|cmd| cmd.plot_area)
            .expect("panel 1's mark draw command should carry a plot_area");
        assert!(
            (panel1_plot_area[0] - 10.0).abs() < 1e-4,
            "plot_area.x: got {}",
            panel1_plot_area[0]
        );
        assert!(
            (panel1_plot_area[1] - (-5.0)).abs() < 1e-4,
            "plot_area.y: got {}",
            panel1_plot_area[1]
        );
        assert!(
            (panel1_plot_area[2] - 200.0).abs() < 1e-4,
            "plot_area.w should scale by sx=2.0, got {}",
            panel1_plot_area[2]
        );
        assert!(
            (panel1_plot_area[3] - 800.0).abs() < 1e-4,
            "plot_area.h should scale by sy=8.0, got {}",
            panel1_plot_area[3]
        );

        // Panel 0's plot_area must remain untouched (identity no-op).
        let panel0_plot_area = data
            .draw_commands
            .iter()
            .find(|cmd| cmd.is_mark && cmd.panel_id == 0)
            .and_then(|cmd| cmd.plot_area)
            .expect("panel 0's mark draw command should carry a plot_area");
        assert_eq!(panel0_plot_area, [0.0, 0.0, 100.0, 100.0]);
    }

    // ── Task 5c: interaction-geometry single source of truth (D4a amendment
    // addendum) — `bake_panels` must produce the SAME baked geometry the GPU
    // load path already bakes internally, for every consumer (hit_test.rs,
    // lib.rs brush/crossfilter, spatial_index.rs,
    // render.rs::upload_transform_and_render) that reads panel geometry
    // outside the GPU mesh/instance pipeline.

    #[test]
    fn bake_panels_matches_gpu_bake_for_non_identity_and_is_noop_at_identity() {
        use ferrum_scene::{
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect,
            SceneGraph,
        };

        let ls = gap_fix_layout_scale();
        let circle_style = gap_fix_fill_stroke(1.0, 0.0);

        let mk_panel = |id: usize, layout_scale: LayoutScale| Panel {
            id,
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
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![SceneNode::Circle {
                    cx: 3.0,
                    cy: 4.0,
                    r: 2.0,
                    style: circle_style.clone(),
                }],
                data_indices: None,
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
            layout_scale,
        };

        let scene = SceneGraph {
            width: 200.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![mk_panel(0, LayoutScale::identity()), mk_panel(1, ls)],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let baked = bake_panels(&scene);
        assert_eq!(baked.len(), 2);

        // Panel 0 (identity): numerically untouched — the byte/pixel
        // stability anchor for every flat/faceted (and pure-translate
        // composite) panel today.
        assert_eq!(baked[0].plot_area, scene.panels[0].plot_area);
        assert_eq!(baked[0].marks[0].nodes, scene.panels[0].marks[0].nodes);

        // Panel 1 (non-identity): plot_area and mark-node coordinates must
        // match the SAME values the GPU-mesh bake path produces (pinned just
        // above by `load_scene_bakes_non_identity_layout_scale_only_for_the_
        // carrying_panel`: plot_area (10, -5, 200, 800), circle at
        // (16, 27, r=8)) — one bake implementation shared by both paths, not
        // two that could drift.
        assert_eq!(
            baked[1].plot_area,
            Rect { x: 10.0, y: -5.0, w: 200.0, h: 800.0 }
        );
        match &baked[1].marks[0].nodes[0] {
            SceneNode::Circle { cx, cy, r, .. } => {
                assert!((cx - 16.0).abs() < 1e-9, "cx: got {cx}");
                assert!((cy - 27.0).abs() < 1e-9, "cy: got {cy}");
                assert!((r - 8.0).abs() < 1e-9, "r: got {r}");
            }
            other => panic!("expected Circle, got {other:?}"),
        }

        // `bake_panels` must not mutate the source scene in place.
        assert_eq!(
            scene.panels[1].plot_area,
            Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }
        );
    }

    #[test]
    fn bake_panels_preserves_panel_count_and_order() {
        use ferrum_scene::{CoordKind, InteractionConfig, Panel, Rect, SceneGraph};

        let mk_panel = |id: usize| Panel {
            id,
            plot_area: Rect { x: id as f64 * 10.0, y: 0.0, w: 50.0, h: 50.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 50.0, h: 50.0 },
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
        };

        let scene = SceneGraph {
            width: 400.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![mk_panel(0), mk_panel(1), mk_panel(2)],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let baked = bake_panels(&scene);
        assert_eq!(baked.len(), 3);
        for (i, panel) in baked.iter().enumerate() {
            assert_eq!(panel.id, i, "bake_panels must preserve id/order at index {i}");
            assert_eq!(panel.plot_area.x, i as f64 * 10.0);
        }
    }

    // ── Task 5c gap-fix (spec-review item 3): FA-18 per-panel transform
    // composes on baked coords, not a second layout_scale application ───────
    //
    // `zoom_pan::select_panel_transform` (FA-18) selects the reactive-rescale
    // affine the render loop applies per panel at draw time, on top of the
    // ALREADY-baked mesh/instance geometry `bake_panels`/`load_scene`
    // produce. This test proves the composition is single-bake:
    // `screen_pos == zoom_affine(baked_position)`, not
    // `zoom_affine(layout_scale(baked_position))` (a double-apply bug).
    #[test]
    fn fa18_per_panel_transform_composes_on_baked_coords_not_double_baked() {
        use crate::zoom_pan::{select_panel_transform, Affine2};
        use ferrum_scene::{
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect,
            SceneGraph,
        };

        let ls = gap_fix_layout_scale(); // sx=2, sy=8, tx=10, ty=-5
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
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![SceneNode::Circle {
                    cx: 3.0,
                    cy: 4.0,
                    r: 2.0,
                    style: gap_fix_fill_stroke(1.0, 0.0),
                }],
                data_indices: None,
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
            layout_scale: ls,
        };
        let scene = SceneGraph {
            width: 200.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![panel],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        // Bake once, as `load_scene`/`bake_panels` do at scene-load time.
        let baked = bake_panels(&scene);
        let baked_circle = match &baked[0].marks[0].nodes[0] {
            SceneNode::Circle { cx, cy, .. } => (*cx, *cy),
            other => panic!("expected Circle, got {other:?}"),
        };
        assert!((baked_circle.0 - 16.0).abs() < 1e-9, "got {}", baked_circle.0);
        assert!((baked_circle.1 - 27.0).abs() < 1e-9, "got {}", baked_circle.1);

        // A reactive rescale on panel 0 (matching `apply_reactive_rescale`'s
        // output shape, e.g. an x-only domain rescale): sx=3, tx=100; sy/ty
        // are also non-identity here to exercise both axes.
        let transforms = vec![Affine2 { sx: 3.0, sy: 3.0, tx: 100.0, ty: 50.0 }];
        let t = select_panel_transform(&transforms, 0);

        // Correct composition: `t` applies to the ALREADY-baked position
        // (this is what the GPU vertex shader and `hit_test`'s inverse-apply
        // both assume) — the single source of truth this task establishes.
        let correct_screen_pos = t.apply(baked_circle.0, baked_circle.1);
        assert!((correct_screen_pos.0 - 148.0).abs() < 1e-9, "got {}", correct_screen_pos.0);
        assert!((correct_screen_pos.1 - 131.0).abs() < 1e-9, "got {}", correct_screen_pos.1);

        // A double-bake bug would instead apply `layout_scale` a SECOND time
        // on the already-baked position before the zoom affine.
        let double_baked = ls.apply(baked_circle.0, baked_circle.1);
        let double_baked_screen_pos = t.apply(double_baked.0, double_baked.1);

        assert!(
            (correct_screen_pos.0 - double_baked_screen_pos.0).abs() > 1.0
                || (correct_screen_pos.1 - double_baked_screen_pos.1).abs() > 1.0,
            "fixture must be discriminating: single-bake and double-bake screen \
             positions must differ substantially"
        );
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod bug_hunt_tests {
    use super::*;
    use std::collections::HashMap;

    // ── parse_tooltip_json: NaN / special characters ────────────────────

    #[test]
    fn bug_hunt_parse_tooltip_json_with_nan_value() {
        // "NaN" as a tooltip value must be properly escaped in JSON output.
        let bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&1u32.to_le_bytes()); // 1 field
            let name = b"val";
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name);
            let val = b"NaN";
            buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
            buf.extend_from_slice(val);
            buf
        };
        let json = parse_tooltip_json(&bytes, 0);
        // Must be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("parse_tooltip_json with NaN value must produce valid JSON");
        assert_eq!(parsed["fields"][0]["value"], "NaN");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_with_backslash_in_value() {
        // Backslash in value must be properly escaped for JSON.
        let bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&1u32.to_le_bytes());
            let name = b"path";
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name);
            let val = br"C:\Users\test";
            buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
            buf.extend_from_slice(val);
            buf
        };
        let json = parse_tooltip_json(&bytes, 0);
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("backslash in tooltip value must produce valid JSON");
        assert_eq!(parsed["fields"][0]["value"], r"C:\Users\test");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_with_empty_field_name() {
        // Empty field name must produce valid JSON.
        let bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&1u32.to_le_bytes());
            let name = b"";
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name);
            let val = b"value";
            buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
            buf.extend_from_slice(val);
            buf
        };
        let json = parse_tooltip_json(&bytes, 0);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("empty field name must produce valid JSON");
        assert_eq!(parsed["fields"][0]["name"], "");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_with_empty_value() {
        // Empty value must produce valid JSON.
        let bytes = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&1u32.to_le_bytes());
            let name = b"x";
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name);
            let val = b"";
            buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
            buf.extend_from_slice(val);
            buf
        };
        let json = parse_tooltip_json(&bytes, 0);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("empty value must produce valid JSON");
        assert_eq!(parsed["fields"][0]["value"], "");
    }

    #[test]
    fn bug_hunt_parse_tooltip_json_many_fields() {
        // 100 fields: must produce valid JSON without panic.
        let num_fields = 100usize;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(num_fields as u32).to_le_bytes());
        for i in 0..num_fields {
            let name = format!("field_{i}");
            bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
        }
        // 1 row of data
        for i in 0..num_fields {
            let val = format!("val_{i}");
            bytes.extend_from_slice(&(val.len() as u32).to_le_bytes());
            bytes.extend_from_slice(val.as_bytes());
        }
        let json = parse_tooltip_json(&bytes, 0);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("100 fields must produce valid JSON");
        let fields = parsed["fields"].as_array().expect("fields array");
        assert_eq!(fields.len(), 100);
    }

    // ── srgb_to_linear edge cases ──────────────────────────────────────

    #[test]
    fn bug_hunt_srgb_to_linear_nan_input_produces_nan() {
        // NaN input should propagate NaN (not panic).
        let result = srgb_to_linear(f32::NAN);
        assert!(result.is_nan(), "srgb_to_linear(NaN) must produce NaN");
    }

    #[test]
    fn bug_hunt_srgb_to_linear_infinity_does_not_panic() {
        // Infinity input should not panic.
        let result = srgb_to_linear(f32::INFINITY);
        assert!(
            result.is_infinite() || result.is_finite(),
            "srgb_to_linear(inf) must not panic"
        );
    }

    // ── linearize_color_channels: alpha channel untouched ──────────────

    #[test]
    fn bug_hunt_linearize_color_channels_preserves_alpha() {
        // Alpha channel must NOT be linearized.
        let mut color = [0.5_f32, 0.5, 0.5, 0.7];
        linearize_color_channels(&mut color);
        // Alpha must be exactly 0.7 (untouched).
        assert!(
            (color[3] - 0.7).abs() < 1e-7,
            "linearize must not modify alpha; got {}",
            color[3]
        );
        // RGB channels must be modified (linearized).
        assert!(
            color[0] < 0.5,
            "RGB channels must be linearized (smaller than sRGB input); got {}",
            color[0]
        );
    }

    // ── opt_color_to_f32: None color → transparent ─────────────────────

    #[test]
    fn bug_hunt_opt_color_none_produces_transparent() {
        let result = opt_color_to_f32(None, 1.0);
        assert_eq!(
            result,
            [0.0, 0.0, 0.0, 0.0],
            "None color must produce fully transparent [0,0,0,0]"
        );
    }

    // ── stroke_dash_index: float edge values ───────────────────────────

    #[test]
    fn bug_hunt_stroke_dash_index_non_integer_floats() {
        // Non-integer-valued floats (e.g. 6.5, 3.5) should not match "6,3".
        let result = stroke_dash_index(&Some(vec![6.5, 3.5]));
        assert!(
            (result - 0.0).abs() < 1e-6,
            "non-integer pattern must fall back to solid (0.0); got {result}"
        );
    }

    #[test]
    fn bug_hunt_stroke_dash_index_single_element_vec() {
        // Single-element vec must not match any palette entry.
        let result = stroke_dash_index(&Some(vec![6.0]));
        assert!(
            (result - 0.0).abs() < 1e-6,
            "single-element vec must fall back to solid; got {result}"
        );
    }

    // ── SceneCollector: collect_annotation vs collect_static routing ────

    #[test]
    fn bug_hunt_collect_annotation_circle_emits_draw_command() {
        // A Circle annotation node must produce a draw command via collect_annotation.
        let style = FillStroke {
            fill: Some(Color {
                r: 255,
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
        let nodes = vec![SceneNode::Circle {
            cx: 100.0,
            cy: 100.0,
            r: 10.0,
            style,
        }];
        let mut collector = SceneCollector::new();
        collector.collect_annotation(&nodes, None, None);
        assert_eq!(
            collector.circles.len(),
            1,
            "circle annotation must be collected"
        );
        // Must have a draw command for the circle
        let circle_cmds: Vec<_> = collector
            .draw_commands
            .iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .collect();
        assert_eq!(
            circle_cmds.len(),
            1,
            "circle annotation must emit a draw command"
        );
        // The draw command must be non-mark (is_mark=false) since annotations use identity transform
        assert!(
            !circle_cmds[0].is_mark,
            "annotation circle draw command must not be is_mark"
        );
    }

    #[test]
    fn bug_hunt_collect_annotation_line_goes_to_annotation_mesh() {
        // A Line annotation node must go to annotation_mesh, not static_mesh.
        let style = ferrum_scene::StrokeStyle {
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            width: 2.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
        };
        let nodes = vec![SceneNode::Line {
            x1: 0.0,
            y1: 50.0,
            x2: 500.0,
            y2: 50.0,
            style,
        }];
        let mut collector = SceneCollector::new();
        collector.collect_annotation(&nodes, None, None);
        assert!(
            !collector.annotation_mesh.vertices.is_empty(),
            "Line annotation must go to annotation_mesh"
        );
        assert!(
            collector.static_mesh.vertices.is_empty(),
            "Line annotation must NOT go to static_mesh"
        );
        assert!(
            collector.mesh.vertices.is_empty(),
            "Line annotation must NOT go to mark mesh"
        );
    }

    #[test]
    fn bug_hunt_collect_static_line_goes_to_static_mesh() {
        // A Line node via collect_static must go to static_mesh.
        let style = ferrum_scene::StrokeStyle {
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
        };
        let nodes = vec![SceneNode::Line {
            x1: 0.0,
            y1: 100.0,
            x2: 500.0,
            y2: 100.0,
            style,
        }];
        let mut collector = SceneCollector::new();
        collector.collect_static(&nodes, None, None);
        assert!(
            !collector.static_mesh.vertices.is_empty(),
            "Line via collect_static must go to static_mesh"
        );
        assert!(
            collector.annotation_mesh.vertices.is_empty(),
            "Line via collect_static must NOT go to annotation_mesh"
        );
    }

    // ── batch_uses_additive_blend: exhaustive ──────────────────────────

    #[test]
    fn bug_hunt_batch_additive_is_exhaustive() {
        // Only Additive returns true; Normal returns false.
        assert!(batch_uses_additive_blend(BlendMode::Additive));
        assert!(!batch_uses_additive_blend(BlendMode::Normal));
    }

    // ── unpack_binary_instances: multi-batch stream ────────────────────

    #[test]
    fn bug_hunt_unpack_two_batches_in_single_stream() {
        // Two batches (circles then rects) concatenated in a single stream.
        let ci = CircleInstance {
            center: [10.0, 20.0],
            radius: 3.0,
            fill_color: [1.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };
        let ri = RectInstance {
            position: [50.0, 60.0],
            size: [20.0, 30.0],
            corner_radius: 0.0,
            fill_color: [0.0, 1.0, 0.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dash: 0.0,
            angle: 0.0,
        };

        let mut buf = Vec::new();
        // Batch 0: panel=0, batch=0, kind=0 (circle), count=1, flags=0
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(bytemuck::bytes_of(&ci));
        // Batch 1: panel=0, batch=1, kind=1 (rect), count=1, flags=0
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(bytemuck::bytes_of(&ri));

        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&buf, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 1, "must unpack 1 circle from first batch");
        assert_eq!(rects.len(), 1, "must unpack 1 rect from second batch");
        assert!(meta.contains_key(&(0, 0)), "meta for (0,0) must exist");
        assert!(meta.contains_key(&(0, 1)), "meta for (0,1) must exist");
    }

    // ── color_to_linear: zero opacity ──────────────────────────────────

    #[test]
    fn bug_hunt_color_to_linear_zero_opacity_produces_transparent() {
        let c = Color {
            r: 255,
            g: 128,
            b: 0,
            a: 255,
        };
        let result = color_to_linear(&c, 0.0);
        assert!(
            result[3].abs() < 1e-7,
            "zero opacity must produce alpha=0.0; got {}",
            result[3]
        );
        // RGB channels are still linearized (opacity only affects alpha).
        assert!(result[0] > 0.0, "R channel must still be linearized");
    }

    // ── Task 6 (item 19): SceneNode::Raw collection ───────────────────
    //
    // Verifies that Raw nodes are collected into `SceneData::raw_fragments`
    // instead of being dropped with a console.warn. This is the Rust-side
    // regression test against the silent drop.

    /// A scene with a single chrome Raw node must yield exactly one raw fragment
    /// with anchor = "chrome".
    #[test]
    fn raw_chrome_node_collected_with_correct_anchor() {
        use ferrum_scene::{BlendMode, MarkBatch, MarkBatchKind, Panel, RawAnchor};
        use ferrum_scene::{CoordKind, InteractionConfig, Rect, SceneGraph};

        let svg_content =
            r#"<linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient>"#;
        let node = SceneNode::Raw {
            svg: svg_content.to_string(),
            anchor: RawAnchor::Chrome,
        };

        let scene = SceneGraph {
            width: 400.0,
            height: 300.0,
            background: None,
            title: vec![],
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
                    nodes: vec![],
                    data_indices: None,
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
                annotations: vec![node],
                strip_title: vec![],
                layout_scale: LayoutScale::identity(),
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(
            data.raw_fragments.len(),
            1,
            "SceneNode::Raw must be collected into raw_fragments, not dropped"
        );
        assert_eq!(
            data.raw_fragments[0].anchor, "chrome",
            "chrome anchor must serialize as 'chrome'"
        );
        assert!(
            data.raw_fragments[0].svg.contains("linearGradient"),
            "raw fragment must preserve svg content"
        );
    }

    /// A scene with a data-anchored Raw node must yield anchor = "data".
    #[test]
    fn raw_data_node_collected_with_correct_anchor() {
        use ferrum_scene::{InteractionConfig, RawAnchor, SceneGraph};

        let node = SceneNode::Raw {
            svg: r#"<image href="data:image/png;base64,abc" x="0" y="0"/>"#.to_string(),
            anchor: RawAnchor::Data,
        };

        let scene = SceneGraph {
            width: 400.0,
            height: 300.0,
            background: None,
            title: vec![],
            panels: vec![],
            legend: vec![],
            decorations: vec![node],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(
            data.raw_fragments.len(),
            1,
            "data-anchored Raw node must be collected"
        );
        assert_eq!(
            data.raw_fragments[0].anchor, "data",
            "data anchor must serialize as 'data'"
        );
    }

    /// Two Raw nodes in different scene regions are both collected.
    #[test]
    fn multiple_raw_nodes_all_collected() {
        use ferrum_scene::{InteractionConfig, RawAnchor, SceneGraph};

        let chrome_node = SceneNode::Raw {
            svg: "<g id='chrome-raw'/>".to_string(),
            anchor: RawAnchor::Chrome,
        };
        let data_node = SceneNode::Raw {
            svg: "<g id='data-raw'/>".to_string(),
            anchor: RawAnchor::Data,
        };

        let scene = SceneGraph {
            width: 400.0,
            height: 300.0,
            background: None,
            title: vec![],
            panels: vec![],
            legend: vec![chrome_node],
            decorations: vec![data_node],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(
            data.raw_fragments.len(),
            2,
            "both Raw nodes must be collected"
        );
        let anchors: Vec<&str> = data
            .raw_fragments
            .iter()
            .map(|r| r.anchor.as_str())
            .collect();
        assert!(anchors.contains(&"chrome"), "must have a chrome fragment");
        assert!(anchors.contains(&"data"), "must have a data fragment");
    }

    /// A Raw node nested inside a Group must also be collected (recursive walk).
    #[test]
    fn raw_node_inside_group_is_collected() {
        use ferrum_scene::{InteractionConfig, RawAnchor, SceneGraph};

        let raw = SceneNode::Raw {
            svg: "<rect id='nested'/>".to_string(),
            anchor: RawAnchor::Chrome,
        };
        let group = SceneNode::Group {
            attrs: vec![],
            children: vec![raw],
        };

        let scene = SceneGraph {
            width: 400.0,
            height: 300.0,
            background: None,
            title: vec![],
            panels: vec![],
            legend: vec![group],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(
            data.raw_fragments.len(),
            1,
            "Raw node nested inside Group must be collected via recursive walk"
        );
        assert_eq!(data.raw_fragments[0].anchor, "chrome");
        assert!(data.raw_fragments[0].svg.contains("nested"));
    }

    // ── WASM-03: PackedTooltipTable ───────────────────────────────────────────

    /// Build a well-formed packed tooltip byte slice with the given fields and rows.
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

    /// `parse` returns `None` on empty input.
    #[test]
    fn packed_tooltip_table_parse_empty_returns_none() {
        assert!(PackedTooltipTable::parse(&[]).is_none());
    }

    /// `parse` returns `None` when num_fields is zero.
    #[test]
    fn packed_tooltip_table_parse_zero_fields_returns_none() {
        let bytes: Vec<u8> = 0u32.to_le_bytes().to_vec();
        assert!(PackedTooltipTable::parse(&bytes).is_none());
    }

    /// `parse` returns `None` when the header is truncated (only 2 bytes).
    #[test]
    fn packed_tooltip_table_parse_truncated_header_returns_none() {
        assert!(PackedTooltipTable::parse(&[0x01, 0x00]).is_none());
    }

    /// `parse` succeeds on a minimal valid table (1 field, 0 rows).
    #[test]
    fn packed_tooltip_table_parse_single_field_no_rows() {
        let bytes = build_tooltip_bytes(&["x"], &[]);
        let table = PackedTooltipTable::parse(&bytes).expect("must parse single field");
        assert_eq!(table.field_names, vec!["x"]);
    }

    /// `row_fields` returns the correct pairs for a single row.
    #[test]
    fn packed_tooltip_table_row_fields_single_row() {
        let bytes = build_tooltip_bytes(&["name", "val"], &[vec!["alice", "42"]]);
        let table = PackedTooltipTable::parse(&bytes).unwrap();
        let fields = table.row_fields(0).expect("must return row 0");
        assert_eq!(fields, vec![("name", "alice"), ("val", "42")]);
    }

    /// `row_fields` works correctly for multiple rows.
    #[test]
    fn packed_tooltip_table_row_fields_multiple_rows() {
        let bytes = build_tooltip_bytes(&["cat"], &[vec!["a"], vec!["b"], vec!["c"]]);
        let table = PackedTooltipTable::parse(&bytes).unwrap();
        assert_eq!(table.row_fields(0).unwrap(), vec![("cat", "a")]);
        assert_eq!(table.row_fields(1).unwrap(), vec![("cat", "b")]);
        assert_eq!(table.row_fields(2).unwrap(), vec![("cat", "c")]);
    }

    /// `row_fields` returns `None` when the row is out of range.
    #[test]
    fn packed_tooltip_table_row_fields_out_of_range_returns_none() {
        let bytes = build_tooltip_bytes(&["x"], &[vec!["1"]]);
        let table = PackedTooltipTable::parse(&bytes).unwrap();
        assert!(
            table.row_fields(1).is_none(),
            "row 1 does not exist — must return None"
        );
    }

    /// `field_value` returns the correct value by column name.
    #[test]
    fn packed_tooltip_table_field_value_correct_column() {
        let bytes = build_tooltip_bytes(&["a", "b", "c"], &[vec!["x", "y", "z"]]);
        let table = PackedTooltipTable::parse(&bytes).unwrap();
        assert_eq!(table.field_value(0, "a"), Some("x"));
        assert_eq!(table.field_value(0, "b"), Some("y"));
        assert_eq!(table.field_value(0, "c"), Some("z"));
    }

    /// `field_value` returns `None` for an unknown field name.
    #[test]
    fn packed_tooltip_table_field_value_unknown_field_returns_none() {
        let bytes = build_tooltip_bytes(&["x"], &[vec!["1"]]);
        let table = PackedTooltipTable::parse(&bytes).unwrap();
        assert!(table.field_value(0, "unknown").is_none());
    }

    /// `field_value` returns `None` when the row is out of range.
    #[test]
    fn packed_tooltip_table_field_value_out_of_range_row_returns_none() {
        let bytes = build_tooltip_bytes(&["x"], &[vec!["1"]]);
        let table = PackedTooltipTable::parse(&bytes).unwrap();
        assert!(table.field_value(99, "x").is_none());
    }

    /// `total_byte_length` with the correct count equals the whole byte slice.
    #[test]
    fn packed_tooltip_table_total_byte_length_matches_buffer() {
        // 2 rows, 2 fields each → count=2
        let bytes = build_tooltip_bytes(&["cat", "val"], &[vec!["a", "1"], vec!["b", "2"]]);
        let len = PackedTooltipTable::total_byte_length(&bytes, 2);
        assert_eq!(
            len,
            bytes.len(),
            "total_byte_length with count=2 must equal the full buffer length"
        );
    }

    /// `total_byte_length` on an empty slice returns `bytes.len()` (0) without panicking.
    #[test]
    fn packed_tooltip_table_total_byte_length_empty_returns_zero() {
        let len = PackedTooltipTable::total_byte_length(&[], 0);
        assert_eq!(len, 0, "empty input must return 0");
    }

    /// `parse_tooltip_json` produces valid JSON with correctly escaped special chars.
    ///
    /// The `name` column in each fields entry is the column name (e.g. "col_name");
    /// the `value` column is the cell value (which may contain special chars).
    #[test]
    fn packed_tooltip_table_parse_tooltip_json_produces_valid_json() {
        // Field names: "col_name" and "col_val" (plain strings).
        // Row values: one contains a double-quote, one contains a backslash.
        let bytes = build_tooltip_bytes(
            &["col_name", "col_val"],
            &[vec![r#"key"with"quotes"#, r"val\backslash"]],
        );
        let json = parse_tooltip_json(&bytes, 0);
        // Must parse as valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("parse_tooltip_json must produce valid JSON for special chars");
        // The "name" JSON key holds the column name; the "value" key holds the cell value.
        assert_eq!(parsed["fields"][0]["name"], "col_name", "first column name");
        assert_eq!(
            parsed["fields"][0]["value"], r#"key"with"quotes"#,
            "value with embedded quotes must round-trip correctly"
        );
        assert_eq!(parsed["fields"][1]["name"], "col_val", "second column name");
        assert_eq!(
            parsed["fields"][1]["value"], r"val\backslash",
            "value with backslash must round-trip correctly"
        );
    }

    /// `tooltip_field_value` (the public entry point) delegates correctly to
    /// `PackedTooltipTable::field_value`.
    #[test]
    fn packed_tooltip_table_tooltip_field_value_correct() {
        let bytes = build_tooltip_bytes(&["x", "y"], &[vec!["10", "20"]]);
        assert_eq!(tooltip_field_value(&bytes, 0, "x"), Some("10".to_string()));
        assert_eq!(tooltip_field_value(&bytes, 0, "y"), Some("20".to_string()));
        assert_eq!(tooltip_field_value(&bytes, 0, "z"), None);
        assert_eq!(tooltip_field_value(&bytes, 1, "x"), None);
    }

    // ── WASM-03 regression: two-batch concatenated packed stream ────────────
    //
    // This test guards against the exact regression fixed by the WASM-03 patch:
    // `total_byte_length` previously walked to buffer-end instead of stopping
    // after `count × num_fields` value entries, so for a concatenated two-batch
    // sidecar where batch 0 has HAS_TOOLTIPS:
    //   (a) batch 0's tooltip slice was over-run into batch 1's header/instance bytes
    //   (b) the outer `while offset+20 <= data.len()` loop saw a bad offset and
    //       terminated early, so batch 1 silently never loaded.
    //
    // This test MUST FAIL on the old `total_byte_length(bytes)` (no count param)
    // and MUST PASS on the fixed `total_byte_length(bytes, count)`.

    /// Build a single packed-batch byte vector from its components.
    ///
    /// `panel_idx`, `batch_idx`, `kind` (0=circle, 1=rect), `count` form the
    /// 20-byte header.  `instance_bytes` is the raw instance data.
    /// `tooltip_rows` is packed into a HAS_TOOLTIPS table appended after.
    fn build_two_batch_packed(
        // batch 0
        b0_count: usize,
        b0_instance_bytes: &[u8],
        b0_tooltip_names: &[&str],
        b0_tooltip_rows: &[Vec<&str>],
        // batch 1
        b1_count: usize,
        b1_instance_bytes: &[u8],
    ) -> Vec<u8> {
        let has_tooltips: u32 = 0x1;

        let mut buf = Vec::new();

        // ── batch 0 header (panel=0, batch=0, kind=0/circle, count, flags=HAS_TOOLTIPS)
        buf.extend_from_slice(&0u32.to_le_bytes()); // panel_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // batch_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // kind = circle
        buf.extend_from_slice(&(b0_count as u32).to_le_bytes());
        buf.extend_from_slice(&has_tooltips.to_le_bytes());
        // batch 0 instance data
        buf.extend_from_slice(b0_instance_bytes);
        // batch 0 tooltip table
        buf.extend_from_slice(&build_tooltip_bytes(b0_tooltip_names, b0_tooltip_rows));

        // ── batch 1 header (panel=0, batch=1, kind=0/circle, count, flags=0)
        buf.extend_from_slice(&0u32.to_le_bytes()); // panel_idx
        buf.extend_from_slice(&1u32.to_le_bytes()); // batch_idx
        buf.extend_from_slice(&0u32.to_le_bytes()); // kind = circle
        buf.extend_from_slice(&(b1_count as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags = 0 (no tooltips)
                                                    // batch 1 instance data
        buf.extend_from_slice(b1_instance_bytes);

        buf
    }

    /// Two-batch packed stream: batch 0 has HAS_TOOLTIPS, batch 1 follows.
    ///
    /// Asserts:
    /// - BOTH batches load (meta contains (0,0) and (0,1))
    /// - batch 0's tooltip slice decodes correctly (not over-run into batch 1)
    /// - `parse_tooltip_json` on batch 0's tooltip bytes returns valid data
    ///
    /// This test fails on the old `total_byte_length` (no count param) because
    /// the walk runs to buffer-end, consuming batch 1's header bytes as if they
    /// were tooltip values; the outer loop then sees `offset > data.len() - 20`
    /// and batch 1 is never added to `meta`.
    ///
    /// Tooltip field/value strings are chosen so the tooltip table's byte length
    /// is divisible by 4, keeping subsequent batch instance data 4-byte aligned
    /// for `bytemuck::try_cast_slice` on native (non-WASM) platforms.
    ///   field "xyz" (3 chars) → 4+3 = 7 bytes
    ///   3 rows, value "abc"/"def"/"ghi" (3 chars each) → 3 × (4+3) = 21 bytes
    ///   + num_fields header (4 bytes) = 4+7+21 = 32 bytes (32 % 4 = 0 ✓)
    #[test]
    fn wasm03_two_batch_packed_stream_both_batches_load_and_tooltip_slice_correct() {
        // Build minimal valid CircleInstance bytes (64 bytes each, all zeros is
        // a valid bytemuck cast since CircleInstance: Zeroable).
        let mk_circle_bytes =
            |n: usize| -> Vec<u8> { vec![0u8; n * std::mem::size_of::<CircleInstance>()] };

        // 3 circles in batch 0, 3 tooltip rows (one per instance).
        // Tooltip table is 32 bytes (divisible by 4) so batch 1's header and
        // instance data land on 4-byte-aligned offsets for bytemuck.
        let b0_count = 3usize;
        let b1_count = 2usize;

        let b0_tooltip_names = &["xyz"];
        let b0_tooltip_rows: Vec<Vec<&str>> = vec![vec!["abc"], vec!["def"], vec!["ghi"]];

        let packed = build_two_batch_packed(
            b0_count,
            &mk_circle_bytes(b0_count),
            b0_tooltip_names,
            &b0_tooltip_rows,
            b1_count,
            &mk_circle_bytes(b1_count),
        );

        // Parse via unpack_binary_instances.
        let mut circles: Vec<CircleInstance> = Vec::new();
        let mut rects: Vec<RectInstance> = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        // BOTH batches must load.
        assert!(
            meta.contains_key(&(0, 0)),
            "batch (0,0) must be present in meta after loading two-batch stream"
        );
        assert!(
            meta.contains_key(&(0, 1)),
            "batch (0,1) must be present in meta — old bug caused batch 1 to be silently dropped"
        );

        // Batch 0 must have loaded b0_count instances.
        let m0 = &meta[&(0, 0)];
        assert_eq!(
            m0.instance_count, b0_count,
            "batch 0 must have {} instances",
            b0_count
        );

        // Batch 1 must have loaded b1_count instances.
        let m1 = &meta[&(0, 1)];
        assert_eq!(
            m1.instance_count, b1_count,
            "batch 1 must have {} instances",
            b1_count
        );

        // Batch 0's tooltip slice must decode to the correct values (not garbled
        // by over-running into batch 1's bytes).
        let tip0 = m0
            .tooltip_bytes
            .as_deref()
            .expect("batch 0 must have tooltip_bytes");
        assert_eq!(
            parse_tooltip_json(tip0, 0),
            r#"{"fields":[{"name":"xyz","value":"abc"}]}"#,
            "batch 0 tooltip row 0 must decode to xyz=abc"
        );
        assert_eq!(
            parse_tooltip_json(tip0, 1),
            r#"{"fields":[{"name":"xyz","value":"def"}]}"#,
            "batch 0 tooltip row 1 must decode to xyz=def"
        );
        assert_eq!(
            parse_tooltip_json(tip0, 2),
            r#"{"fields":[{"name":"xyz","value":"ghi"}]}"#,
            "batch 0 tooltip row 2 must decode to xyz=ghi"
        );
        // Row 3 is out of range for batch 0 (only 3 rows).
        assert_eq!(
            parse_tooltip_json(tip0, 3),
            "{}",
            "batch 0 tooltip row 3 must return empty JSON (only 3 rows in this batch)"
        );

        // Batch 1 has no tooltips.
        assert!(
            m1.tooltip_bytes.is_none(),
            "batch 1 must have no tooltip_bytes"
        );
    }

    // ── WASM-02: DrawKind::try_from decodes the packed kind discriminant ─────

    /// `0` → Circle, `1` → Rect; any other value is rejected. This is the single
    /// decode point for the packed `kind: u32` header.
    #[test]
    fn draw_kind_try_from_maps_known_values() {
        assert_eq!(DrawKind::try_from(0), Ok(DrawKind::Circle));
        assert_eq!(DrawKind::try_from(1), Ok(DrawKind::Rect));
        assert_eq!(DrawKind::try_from(2), Err(UnknownDrawKind(2)));
        assert_eq!(DrawKind::try_from(u32::MAX), Err(UnknownDrawKind(u32::MAX)));
    }
}
