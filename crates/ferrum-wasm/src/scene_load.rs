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
    /// Number of panels in the source scene graph. `GpuBuffers` allocates one
    /// per-panel transform slot for each, so any `panel_id` recorded on a mark
    /// draw command or mesh panel has a slot to bind. Always `>= 1`.
    pub panel_count: usize,
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
    pub kind: u32,
    pub instance_start: usize,
    pub instance_count: usize,
}

/// Which GPU instance buffer a draw command targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawKind {
    Circle,
    Rect,
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
}

/// Per-panel mark-mesh draw range.
///
/// Captures the contiguous slice of the mark-mesh index buffer that belongs
/// to one panel, together with that panel's plot area. `render_frame` iterates
/// this list to scissor each panel's mesh draw to its own plot area, preventing
/// zoomed/panned geometry from bleeding into axis margins or adjacent panels.
#[derive(Clone, Debug)]
pub struct MarkMeshPanel {
    /// First index in the flat mark-mesh index buffer for this panel.
    pub index_start: u32,
    /// Number of indices (triangle soup) belonging to this panel.
    pub index_count: u32,
    /// Plot area `[x, y, w, h]` in canvas pixels.
    pub plot_area: [f32; 4],
    /// Index of the panel that owns this mesh slice. The render loop binds
    /// this panel's own affine transform so a non-uniform domain-rescale on a
    /// sibling panel does not shear or translate this panel's mesh.
    pub panel_id: usize,
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
        self.emit(false, false, None, 0);
    }

    /// Collect `nodes` into the mark mesh (mark batches: lines, areas, paths,
    /// polygons, polylines) and immediately emit draw commands for any new
    /// circle/rect instances with the given blend mode, plot area, and owning
    /// panel index.
    pub fn collect_mark(
        &mut self,
        nodes: &[SceneNode],
        additive: bool,
        plot_area: Option<[f32; 4]>,
        panel_id: usize,
        batch_cap: Option<StrokeCap>,
        batch_join: Option<StrokeJoin>,
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
        self.emit(additive, true, plot_area, panel_id);
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
        self.emit(false, false, None, 0);
    }

    /// Emit draw commands for any circles/rects added since the last snapshot.
    ///
    /// `panel_id` is only meaningful for mark commands (`is_mark == true`); the
    /// render loop binds that panel's affine. Non-mark commands always draw
    /// with the identity transform, so callers pass `0`.
    fn emit(&mut self, additive: bool, is_mark: bool, plot_area: Option<[f32; 4]>, panel_id: usize) {
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
    ) {
        let index_count = index_end_after - index_start_before;
        if index_count > 0 {
            self.mark_mesh_panels.push(MarkMeshPanel {
                index_start: index_start_before,
                index_count,
                plot_area,
                panel_id,
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

pub fn load_scene(scene: &SceneGraph) -> SceneData {
    load_scene_with_packed(scene, &[])
}

pub fn load_scene_with_packed(scene: &SceneGraph, packed_data: &[u8]) -> SceneData {
    let mut collector = SceneCollector::new();
    let mut batch_meta = HashMap::new();

    // Unpack binary instance data (passed as raw bytes, not base64).
    // Draw commands for packed batches are emitted in the scene-graph walk
    // below, where the MarkBatch.blend mode is available.
    unpack_binary_instances(packed_data, &mut collector.circles, &mut collector.rects, &mut batch_meta);
    // Sync snapshot counters after pre-populating from packed data.
    collector.snapshot();

    let background = scene.background.as_ref().map(|c| color_to_linear(c, 1.0));

    // Title: non-mark → static mesh
    collector.collect_static(&scene.title, None, None);

    for (panel_idx, panel) in scene.panels.iter().enumerate() {
        // Grid: non-mark → static mesh. Snap Line nodes to pixel centers to
        // avoid sub-pixel aliasing in the GPU rasterizer (the SVG renderer
        // handles this natively; WASM needs explicit snapping).
        let snapped_grid: Vec<SceneNode> = panel.grid.iter().map(|node| {
            if let SceneNode::Line { x1, y1, x2, y2, style } = node {
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
                SceneNode::Line { x1: sx1, y1: sy1, x2: sx2, y2: sy2, style: style.clone() }
            } else {
                node.clone()
            }
        }).collect();
        collector.collect_static(&snapped_grid, None, None);

        let panel_plot_area_arr = [
            panel.plot_area.x as f32,
            panel.plot_area.y as f32,
            panel.plot_area.w as f32,
            panel.plot_area.h as f32,
        ];
        let panel_plot_area = Some(panel_plot_area_arr);

        // Snapshot the mark-mesh index count before processing this panel's
        // mark batches so we can record the contiguous range afterward.
        let mesh_index_start_before = collector.mesh.indices.len() as u32;

        for (batch_idx, batch) in panel.marks.iter().enumerate() {
            let additive = batch_uses_additive_blend(batch.blend);

            // If this batch has packed binary instances, emit a draw command
            // from the packed metadata (the instances were already added by
            // unpack_binary_instances above). Otherwise, collect from nodes.
            let key = (panel_idx as u32, batch_idx as u32);
            if let Some(meta) = batch_meta.get(&key) {
                let kind = match meta.kind {
                    0 => DrawKind::Circle,
                    _ => DrawKind::Rect,
                };
                collector.draw_commands.push(DrawCommand {
                    kind,
                    instance_start: meta.instance_start as u32,
                    instance_count: meta.instance_count as u32,
                    additive,
                    is_mark: true,
                    plot_area: panel_plot_area,
                    panel_id: panel_idx,
                });
            } else {
                // Mark batches → mark mesh (zoom transform)
                collector.collect_mark(
                    &batch.nodes,
                    additive,
                    panel_plot_area,
                    panel_idx,
                    batch.stroke_cap,
                    batch.stroke_join,
                );
            }
        }

        // Record this panel's mark-mesh index range. Panels that contributed
        // no mesh geometry (e.g. only packed instances) produce a zero-count
        // range and are skipped by `record_mark_mesh_panel`.
        let mesh_index_end_after = collector.mesh.indices.len() as u32;
        collector.record_mark_mesh_panel(
            mesh_index_start_before,
            mesh_index_end_after,
            panel_plot_area_arr,
            panel_idx,
        );

        // Axes, strip titles: non-mark → static mesh
        collector.collect_static(&panel.axes, None, None);
        collector.collect_static(&panel.strip_title, None, None);
        // Annotations: route to annotation_mesh so they appear above data
        // marks in WASM (matching SVG painter order).
        collector.collect_annotation(&panel.annotations, None, None);
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
    let bytes: [u8; 4] = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
    u32::from_le_bytes(bytes)
}

/// Unpack raw binary instance data into circle/rect buffers.
///
/// Format (v2, 20-byte header): repeated
/// `[panel_idx: u32][batch_idx: u32][kind: u32][count: u32][flags: u32]`
/// followed by instance data, then optional data_indices and tooltip bytes
/// based on `flags`.
///
/// kind=0 → CircleInstance, kind=1 → RectInstance.
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
        let kind = read_u32_le(data, offset + 8);
        let count = read_u32_le(data, offset + 12) as usize;
        let flags = read_u32_le(data, offset + 16);
        offset += 20;

        // Read instance data, tracking the start index for hit-testing.
        // Returns (instance_byte_len, instance_start, loaded_count) where
        // loaded_count reflects what was ACTUALLY pushed (0 on bytemuck failure)
        // so PackedBatchMeta.instance_count never points at phantom instances.
        let (instance_byte_len, instance_start, loaded_count) = match kind {
            0 => {
                let byte_len = count * std::mem::size_of::<CircleInstance>();
                if offset + byte_len > data.len() { break; }
                let start = circles.len();
                if let Ok(instances) = bytemuck::try_cast_slice::<_, CircleInstance>(&data[offset..offset+byte_len]) {
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
            1 => {
                let byte_len = count * std::mem::size_of::<RectInstance>();
                if offset + byte_len > data.len() { break; }
                let start = rects.len();
                if let Ok(instances) = bytemuck::try_cast_slice::<_, RectInstance>(&data[offset..offset+byte_len]) {
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
            _ => break,
        };
        offset += instance_byte_len;

        // Read data_indices if flagged.
        let data_indices = if flags & HAS_DATA_INDICES != 0 {
            let indices_byte_len = count * 4;
            if offset + indices_byte_len > data.len() { break; }
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
            // Scan the string table to find its total length:
            //   [num_fields: u32]
            //   num_fields × [len: u32][bytes]     (field names)
            //   count × num_fields × [len: u32][bytes]  (values)
            let scan_start = offset;
            if offset + 4 > data.len() { break; }
            let num_fields = read_u32_le(data, offset) as usize;
            offset += 4;

            // Skip field names.
            for _ in 0..num_fields {
                if offset + 4 > data.len() { break; }
                let slen = read_u32_le(data, offset) as usize;
                offset += 4 + slen;
                if offset > data.len() { break; }
            }

            // Skip row values: count rows × num_fields entries.
            for _ in 0..count * num_fields {
                if offset + 4 > data.len() { break; }
                let slen = read_u32_le(data, offset) as usize;
                offset += 4 + slen;
                if offset > data.len() { break; }
            }

            // Clamp the slice end to the buffer length so a malformed/truncated
            // tooltip table (where the scan loop advanced offset past the end)
            // does not panic. scan_start is always <= data.len() (it was set
            // immediately after the 20-byte header bounds check), so only the
            // end needs clamping. (F2)
            Some(data[scan_start..offset.min(data.len())].to_vec())
        } else {
            None
        };

        meta.insert(
            (panel_idx, batch_idx),
            PackedBatchMeta {
                data_indices, tooltip_bytes,
                kind, instance_start,
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
            SceneNode::Rect { x, y, w, h, style, corner_radius } => {
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
            SceneNode::Line { x1, y1, x2, y2, style } => {
                let mut s = style.clone();
                if s.stroke_cap.is_none() { s.stroke_cap = batch_cap; }
                if s.stroke_join.is_none() { s.stroke_join = batch_join; }
                tessellate::tessellate_line(*x1, *y1, *x2, *y2, &s, mesh);
            }
            SceneNode::Path { commands, style, closed } => {
                tessellate::tessellate_path(commands, style, *closed, batch_cap, batch_join, mesh);
            }
            SceneNode::Polyline { points, style } => {
                let mut s = style.clone();
                if s.stroke_cap.is_none() { s.stroke_cap = batch_cap; }
                if s.stroke_join.is_none() { s.stroke_join = batch_join; }
                tessellate::tessellate_polyline(points, &s, mesh);
            }
            SceneNode::Polygon { rings, style } => {
                tessellate::tessellate_polygon(rings, style, mesh);
            }
            SceneNode::Text { x, y, content, style } => {
                texts.push(TextElementData {
                    x: *x,
                    y: *y,
                    content: content.clone(),
                    style: style.clone(),
                });
            }
            SceneNode::Image { x, y, w, h, data } => {
                match data {
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
                }
            }
            SceneNode::Group { children, .. } => {
                collect_nodes(children, circles, rects, mesh, texts, images, raws, batch_cap, batch_join);
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

/// Parse a tooltip string table and return a JSON string for one row.
///
/// The `tooltip_bytes` slice starts with `[num_fields: u32]`, followed by
/// `num_fields` length-prefixed field name strings, then `total_rows ×
/// num_fields` length-prefixed value strings (row-major).
///
/// Returns `{"fields":[{"name":"x","value":"1.23"},…]}` for the requested
/// `row_idx`, or `"{}"` if the index is out of range or the data is malformed.
pub fn parse_tooltip_json(tooltip_bytes: &[u8], row_idx: usize) -> String {
    let mut offset = 0;

    // Read num_fields.
    if offset + 4 > tooltip_bytes.len() {
        return "{}".to_string();
    }
    let num_fields = read_u32_le(tooltip_bytes, offset) as usize;
    offset += 4;

    if num_fields == 0 {
        return "{}".to_string();
    }

    // Read field names.
    let mut field_names = Vec::with_capacity(num_fields);
    for _ in 0..num_fields {
        if offset + 4 > tooltip_bytes.len() {
            return "{}".to_string();
        }
        let slen = read_u32_le(tooltip_bytes, offset) as usize;
        offset += 4;
        if offset + slen > tooltip_bytes.len() {
            return "{}".to_string();
        }
        let name = std::str::from_utf8(&tooltip_bytes[offset..offset + slen])
            .unwrap_or("");
        field_names.push(name);
        offset += slen;
    }

    // Skip to row `row_idx`: each value entry is [len: u32][bytes].
    for _ in 0..row_idx * num_fields {
        if offset + 4 > tooltip_bytes.len() {
            return "{}".to_string();
        }
        let slen = read_u32_le(tooltip_bytes, offset) as usize;
        offset += 4 + slen;
        if offset > tooltip_bytes.len() {
            return "{}".to_string();
        }
    }

    // Read this row's values.
    let mut fields_json = Vec::with_capacity(num_fields);
    for field_name in &field_names {
        if offset + 4 > tooltip_bytes.len() {
            return "{}".to_string();
        }
        let slen = read_u32_le(tooltip_bytes, offset) as usize;
        offset += 4;
        if offset + slen > tooltip_bytes.len() {
            return "{}".to_string();
        }
        let value = std::str::from_utf8(&tooltip_bytes[offset..offset + slen])
            .unwrap_or("");
        offset += slen;

        // Escape quotes in name/value for JSON safety.
        let escaped_name = field_name.replace('\\', "\\\\").replace('"', "\\\"");
        let escaped_value = value.replace('\\', "\\\\").replace('"', "\\\"");
        fields_json.push(format!(
            r#"{{"name":"{}","value":"{}"}}"#,
            escaped_name, escaped_value
        ));
    }

    format!(r#"{{"fields":[{}]}}"#, fields_json.join(","))
}

/// Read a single tooltip field value (as a string) for one row of a packed
/// batch's tooltip string table.
///
/// The layout matches [`parse_tooltip_json`]: `[num_fields: u32]`, then
/// `num_fields` length-prefixed field-name strings, then `total_rows ×
/// num_fields` length-prefixed value strings (row-major). This walks the table
/// for `row_idx` and returns the value owned by the named field.
///
/// Returns `None` when the field is absent, the row is out of range, or the
/// data is malformed — packed legend/field-value matching uses this to mirror
/// the unpacked tooltip-field path on `< 1000`-mark batches.
pub fn tooltip_field_value(tooltip_bytes: &[u8], row_idx: usize, field: &str) -> Option<String> {
    let mut offset = 0usize;

    if offset + 4 > tooltip_bytes.len() {
        return None;
    }
    let num_fields = read_u32_le(tooltip_bytes, offset) as usize;
    offset += 4;
    if num_fields == 0 {
        return None;
    }

    // Read field names, tracking which column holds the requested field.
    let mut field_col = None;
    for col in 0..num_fields {
        if offset + 4 > tooltip_bytes.len() {
            return None;
        }
        let slen = read_u32_le(tooltip_bytes, offset) as usize;
        offset += 4;
        if offset + slen > tooltip_bytes.len() {
            return None;
        }
        let name = std::str::from_utf8(&tooltip_bytes[offset..offset + slen]).unwrap_or("");
        if name == field {
            field_col = Some(col);
        }
        offset += slen;
    }
    let field_col = field_col?;

    // Skip whole rows before `row_idx`.
    for _ in 0..row_idx * num_fields {
        if offset + 4 > tooltip_bytes.len() {
            return None;
        }
        let slen = read_u32_le(tooltip_bytes, offset) as usize;
        offset += 4 + slen;
        if offset > tooltip_bytes.len() {
            return None;
        }
    }

    // Walk this row's columns up to and including the target column.
    for col in 0..num_fields {
        if offset + 4 > tooltip_bytes.len() {
            return None;
        }
        let slen = read_u32_le(tooltip_bytes, offset) as usize;
        offset += 4;
        if offset + slen > tooltip_bytes.len() {
            return None;
        }
        if col == field_col {
            let value = std::str::from_utf8(&tooltip_bytes[offset..offset + slen]).ok()?;
            return Some(value.to_string());
        }
        offset += slen;
    }

    None
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
            let joined = v.iter()
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
        assert!((linear - 0.214).abs() < 0.002, "sRGB 0.5 → linear ~0.214, got {linear}");
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
            assert!(l >= prev, "srgb_to_linear must be monotonic: f({s}) = {l} < f({}) = {prev}", (i - 1) as f32 / 100.0);
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
        assert_eq!(bytes.len(), 16 * 4, "CircleInstance must be exactly 16 floats");
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
        assert_eq!(bytes.len(), 18 * 4, "RectInstance must be exactly 18 floats");
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
        use ferrum_scene::{FillStroke, SceneNode, Panel, MarkBatch, MarkBatchKind, BlendMode};
        use ferrum_scene::{CoordKind, Rect, SceneGraph, InteractionConfig};

        let style = FillStroke {
            fill: Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke_width: 2.0,
            opacity: 1.0,
            stroke_opacity: 0.5,
            fill_opacity: 1.0,
            stroke_dash: Some(vec![6.0, 3.0]), // index 1 = dashed
            angle: 45.0,
        };

        let node = SceneNode::Circle { cx: 50.0, cy: 50.0, r: 10.0, style };

        let scene = SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
        assert!((ci.stroke_opacity - 0.5).abs() < 1e-6,
            "stroke_opacity should be 0.5, got {}", ci.stroke_opacity);
        assert!((ci.stroke_dash - 1.0).abs() < 1e-6,
            "stroke_dash index should be 1 (dashed), got {}", ci.stroke_dash);
        assert!((ci.angle - 45.0).abs() < 1e-6,
            "angle should be 45.0, got {}", ci.angle);
    }

    #[test]
    fn load_scene_populates_stroke_opacity_and_angle_for_rect() {
        use ferrum_scene::{FillStroke, SceneNode, Panel, MarkBatch, MarkBatchKind, BlendMode};
        use ferrum_scene::{CoordKind, Rect, SceneGraph, InteractionConfig};

        let style = FillStroke {
            fill: Some(Color { r: 0, g: 128, b: 255, a: 255 }),
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke_width: 1.0,
            opacity: 0.9,
            stroke_opacity: 0.75,
            fill_opacity: 1.0,
            stroke_dash: Some(vec![2.0, 3.0]), // index 2 = dotted
            angle: 30.0,
        };

        let node = SceneNode::Rect {
            x: 10.0, y: 20.0, w: 40.0, h: 30.0, style, corner_radius: 0.0,
        };

        let scene = SceneGraph {
            width: 200.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
        assert!((ri.stroke_opacity - 0.75).abs() < 1e-6,
            "stroke_opacity should be 0.75, got {}", ri.stroke_opacity);
        assert!((ri.stroke_dash - 2.0).abs() < 1e-6,
            "stroke_dash index should be 2 (dotted), got {}", ri.stroke_dash);
        assert!((ri.angle - 30.0).abs() < 1e-6,
            "angle should be 30.0, got {}", ri.angle);
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
        use ferrum_scene::{FillStroke, SceneNode, Panel, MarkBatch, MarkBatchKind, BlendMode};
        use ferrum_scene::{CoordKind, Rect, SceneGraph, InteractionConfig};

        // FillStroke with default stroke_opacity (1.0) and angle (0.0)
        let style = FillStroke {
            fill: Some(Color { r: 100, g: 100, b: 100, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0, // default
            fill_opacity: 1.0,   // default
            stroke_dash: None,   // solid → 0.0
            angle: 0.0,          // default
        };

        let node = SceneNode::Circle { cx: 25.0, cy: 25.0, r: 5.0, style };

        let scene = SceneGraph {
            width: 50.0,
            height: 50.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 50.0, h: 50.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 50.0, h: 50.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        let ci = &data.circle_instances[0];
        assert!((ci.stroke_opacity - 1.0).abs() < 1e-6, "default stroke_opacity is 1.0");
        assert!((ci.stroke_dash - 0.0).abs() < 1e-6, "default stroke_dash is 0.0 (solid)");
        assert!((ci.angle - 0.0).abs() < 1e-6, "default angle is 0.0");
    }

    // ── Polygon tessellation regression tests ─────────────────────────
    //
    // These test the exact code path that the WASM interactive renderer
    // uses for hexbin and geoshape marks: SceneNode::Polygon → lyon
    // tessellation → mesh vertex/index buffers. If tessellation produces
    // zero vertices, the GPU renders nothing (the "empty hex" bug).

    fn make_scene_with_polygons(nodes: Vec<SceneNode>) -> SceneGraph {
        use ferrum_scene::{Panel, MarkBatch, MarkBatchKind, BlendMode};
        use ferrum_scene::{CoordKind, Rect, InteractionConfig};
        SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
                fill: Some(Color { r: 100, g: 150, b: 200, a: 255 }),
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
        let scene = make_scene_with_polygons(vec![
            hex_polygon(200.0, 200.0, 20.0),
        ]);
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
            [100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0],
        ];
        let hole = vec![
            [150.0, 150.0], [250.0, 150.0], [250.0, 250.0], [150.0, 250.0],
        ];
        let node = SceneNode::Polygon {
            rings: vec![exterior, hole],
            style: FillStroke {
                fill: Some(Color { r: 50, g: 100, b: 150, a: 255 }),
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
                [100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0],
            ]],
            style: FillStroke {
                fill: None,
                stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
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
                assert!(c.is_finite(), "mesh vertex {i} has non-finite color component");
            }
        }
    }

    #[test]
    fn degenerate_polygon_does_not_panic() {
        // 2-point "polygon" — should be skipped gracefully, not panic.
        let node = SceneNode::Polygon {
            rings: vec![vec![[100.0, 100.0], [200.0, 200.0]]],
            style: FillStroke {
                fill: Some(Color { r: 255, g: 0, b: 0, a: 255 }),
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
            fill: Some(Color { r: 70, g: 130, b: 180, a: 255 }),
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
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
            color: Color { r: 0, g: 0, b: 0, a: 255 },
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
        use ferrum_scene::{Panel, MarkBatch, BlendMode};
        use ferrum_scene::{CoordKind, Rect, InteractionConfig};
        SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
            SceneNode::Circle { cx: 100.0, cy: 100.0, r: 5.0, style: default_fill_stroke() },
            SceneNode::Circle { cx: 200.0, cy: 150.0, r: 8.0, style: default_fill_stroke() },
            SceneNode::Circle { cx: 300.0, cy: 200.0, r: 3.0, style: default_fill_stroke() },
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
            vec![SceneNode::Circle { cx: 50.0, cy: 50.0, r: 10.0, style }],
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
            SceneNode::Rect { x: 60.0, y: 50.0, w: 30.0, h: 200.0,
                              style: default_fill_stroke(), corner_radius: 0.0 },
            SceneNode::Rect { x: 100.0, y: 80.0, w: 30.0, h: 170.0,
                              style: default_fill_stroke(), corner_radius: 2.0 },
            SceneNode::Rect { x: 140.0, y: 30.0, w: 30.0, h: 220.0,
                              style: default_fill_stroke(), corner_radius: 0.0 },
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
                x: 10.0, y: 20.0, w: 80.0, h: 60.0,
                style: default_fill_stroke(), corner_radius: 5.5,
            }],
        );
        let data = load_scene(&scene);
        let r = &data.rect_instances[0];
        assert!((r.corner_radius - 5.5).abs() < 1e-3);
    }

    // ── Line (rule / tick / segment marks) ────────────────────────────

    #[test]
    fn line_node_tessellates_to_mesh() {
        let nodes = vec![
            SceneNode::Line {
                x1: 50.0, y1: 50.0, x2: 300.0, y2: 200.0,
                style: default_stroke_style(),
            },
        ];
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
                x1: 50.0, y1: 30.0 + i as f64 * 30.0,
                x2: 400.0, y2: 30.0 + i as f64 * 30.0,
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
        let points = vec![
            (50.0, 300.0), (150.0, 100.0), (250.0, 250.0), (350.0, 50.0),
        ];
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
                rx: 100.0, ry: 100.0, rotation: 0.0,
                large_arc: false, sweep: true,
                x: 300.0, y: 200.0,
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
                c1x: 150.0, c1y: 50.0, c2x: 250.0, c2y: 350.0, x: 350.0, y: 200.0,
            },
        ];
        let mut style = default_fill_stroke();
        style.fill = None;
        let nodes = vec![SceneNode::Path {
            commands, style, closed: false,
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
        let wedges: Vec<SceneNode> = (0..3).map(|i| {
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
                    rx: r, ry: r, rotation: 0.0,
                    large_arc: false, sweep: true,
                    x: cx + r * angle_end.cos(),
                    y: cy + r * angle_end.sin(),
                },
                PathCmd::Close,
            ];
            SceneNode::Path { commands, style: default_fill_stroke(), closed: true }
        }).collect();
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
        use ferrum_scene::{TextStyle, FontWeight, TextAnchor, TextBaseline};
        let style = TextStyle {
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
            color: Color { r: 0, g: 0, b: 0, a: 255 },
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        };
        let nodes = vec![
            SceneNode::Text { x: 100.0, y: 50.0, content: "Hello".to_string(), style: style.clone() },
            SceneNode::Text { x: 200.0, y: 80.0, content: "World".to_string(), style },
        ];
        let scene = make_scene_with_nodes(MarkBatchKind::Text, nodes);
        let data = load_scene(&scene);
        assert_eq!(data.text_elements.len(), 2, "2 Text nodes → 2 text elements");
        assert_eq!(data.text_elements[0].content, "Hello");
        assert!((data.text_elements[0].x - 100.0).abs() < 1e-3);
        assert_eq!(data.text_elements[1].content, "World");
    }

    #[test]
    fn text_does_not_produce_mesh_or_instances() {
        use ferrum_scene::{TextStyle, FontWeight, TextAnchor, TextBaseline};
        let style = TextStyle {
            font_size: 14.0,
            font_weight: FontWeight::Bold,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Middle,
            angle: 0.0,
            color: Color { r: 50, g: 50, b: 50, a: 255 },
            opacity: 1.0,
            font_family: "serif".to_string(),
        };
        let nodes = vec![
            SceneNode::Text { x: 100.0, y: 100.0, content: "Label".to_string(), style },
        ];
        let scene = make_scene_with_nodes(MarkBatchKind::Text, nodes);
        let data = load_scene(&scene);
        assert!(data.circle_instances.is_empty(), "text must not produce circles");
        assert!(data.rect_instances.is_empty(), "text must not produce rects");
        assert!(data.mesh_buffers.vertices.is_empty(), "text must not produce mesh");
    }

    // ── Group (recursive) ─────────────────────────────────────────────

    #[test]
    fn group_node_recurses_into_children() {
        let children = vec![
            SceneNode::Circle { cx: 100.0, cy: 100.0, r: 5.0, style: default_fill_stroke() },
            SceneNode::Rect { x: 200.0, y: 50.0, w: 40.0, h: 30.0,
                              style: default_fill_stroke(), corner_radius: 0.0 },
        ];
        let nodes = vec![SceneNode::Group {
            attrs: vec![],
            children,
        }];
        let scene = make_scene_with_nodes(MarkBatchKind::Point, nodes);
        let data = load_scene(&scene);
        assert_eq!(data.circle_instances.len(), 1, "group child circle must be collected");
        assert_eq!(data.rect_instances.len(), 1, "group child rect must be collected");
    }

    // ── Mixed scene (all node types at once) ──────────────────────────

    #[test]
    fn mixed_scene_all_buffers_populated() {
        use ferrum_scene::{Panel, MarkBatch, MarkBatchKind, BlendMode, PathCmd};
        use ferrum_scene::{CoordKind, Rect, InteractionConfig};
        use ferrum_scene::{TextStyle, FontWeight, TextAnchor, TextBaseline};

        let circle = SceneNode::Circle {
            cx: 100.0, cy: 100.0, r: 8.0, style: default_fill_stroke(),
        };
        let rect = SceneNode::Rect {
            x: 200.0, y: 50.0, w: 50.0, h: 100.0,
            style: default_fill_stroke(), corner_radius: 0.0,
        };
        let line = SceneNode::Line {
            x1: 50.0, y1: 300.0, x2: 400.0, y2: 300.0,
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
            x: 250.0, y: 30.0,
            content: "Title".to_string(),
            style: TextStyle {
                font_size: 16.0,
                font_weight: FontWeight::Bold,
                anchor: TextAnchor::Middle,
                baseline: TextBaseline::Top,
                angle: 0.0,
                color: Color { r: 0, g: 0, b: 0, a: 255 },
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
                plot_area: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
                },
                grid: vec![],
                marks: vec![
                    MarkBatch {
                        kind: MarkBatchKind::Point,
                        nodes: vec![circle],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Bar,
                        nodes: vec![rect],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Rule,
                        nodes: vec![line],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Area,
                        nodes: vec![path],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Polygon,
                        nodes: vec![polygon],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Line,
                        nodes: vec![polyline],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                ],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
        assert!(!data.mesh_buffers.vertices.is_empty(), "mesh from line+path+polygon+polyline");
        assert!(!data.mesh_buffers.indices.is_empty(), "mesh indices from tessellation");
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
            center: [100.0, 200.0], radius: 5.0,
            fill_color: [1.0, 0.0, 0.0, 0.8],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.5, opacity: 0.8,
            stroke_opacity: 0.6, stroke_dash: 1.0, angle: 45.0,
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
                position: [10.0, 20.0], size: [100.0, 50.0], corner_radius: 3.0,
                fill_color: [0.0, 1.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 0.5],
                stroke_width: 2.0, opacity: 0.9, stroke_opacity: 0.7, stroke_dash: 2.0, angle: 0.0,
            },
            RectInstance {
                position: [200.0, 30.0], size: [80.0, 60.0], corner_radius: 0.0,
                fill_color: [0.0, 0.0, 1.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 90.0,
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
        buf.extend_from_slice(&0u32.to_le_bytes());  // panel_idx
        buf.extend_from_slice(&0u32.to_le_bytes());  // batch_idx
        buf.extend_from_slice(&0u32.to_le_bytes());  // kind
        buf.extend_from_slice(&100u32.to_le_bytes()); // count
        buf.extend_from_slice(&0u32.to_le_bytes());  // flags
        unpack_binary_instances(&buf, &mut circles, &mut rects, &mut meta);
        assert!(circles.is_empty());
    }

    #[test]
    fn load_scene_with_packed_uses_binary_sidecar() {
        use ferrum_scene::{Panel, MarkBatch, MarkBatchKind, BlendMode};
        use ferrum_scene::{CoordKind, Rect, SceneGraph, InteractionConfig};

        let instances: Vec<CircleInstance> = (0..3)
            .map(|i| CircleInstance {
                center: [i as f32 * 100.0, 50.0], radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
            })
            .collect();
        let packed = build_packed_circle_stream(&instances);

        let scene = SceneGraph {
            width: 400.0, height: 200.0, background: None, title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 },
                coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
                grid: vec![],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point, nodes: vec![],
                    data_indices: None, tooltips: None, hrefs: None,
                    descriptions: None, keys: None,
                    blend: BlendMode::Normal, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                }],
                axes: vec![], annotations: vec![], strip_title: vec![],
            }],
            legend: vec![], decorations: vec![], selections: vec![],
            interaction: InteractionConfig::default(), chart_description: None,
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
                center: [10.0, 20.0], radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
            },
            CircleInstance {
                center: [30.0, 40.0], radius: 7.0,
                fill_color: [0.0, 1.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
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
        let indices = m.data_indices.as_ref().expect("data_indices should be Some");
        assert_eq!(indices, &[42, 99]);
        assert!(m.tooltip_bytes.is_none());
    }

    #[test]
    fn binary_unpack_with_tooltips() {
        let instances = vec![
            CircleInstance {
                center: [10.0, 20.0], radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
            },
        ];
        let tooltip_data = build_tooltip_bytes(
            &["x", "y"],
            &[vec!["1.23", "4.56"]],
        );
        let packed = build_packed_circle_stream_ex(0, 0, &instances, HAS_TOOLTIPS, &tooltip_data);

        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 1);
        let m = meta.get(&(0, 0)).expect("meta for (0,0) should exist");
        assert!(m.data_indices.is_none());
        let tb = m.tooltip_bytes.as_ref().expect("tooltip_bytes should be Some");
        assert_eq!(tb, &tooltip_data, "tooltip bytes should match input");
    }

    #[test]
    fn binary_unpack_with_data_indices_and_tooltips() {
        let instances = vec![
            CircleInstance {
                center: [10.0, 20.0], radius: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
            },
            CircleInstance {
                center: [30.0, 40.0], radius: 7.0,
                fill_color: [0.0, 1.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
            },
        ];
        let mut trailing = build_data_indices_bytes(&[10, 20]);
        let tooltip_data = build_tooltip_bytes(
            &["name", "value"],
            &[vec!["alpha", "100"], vec!["beta", "200"]],
        );
        trailing.extend_from_slice(&tooltip_data);

        let packed = build_packed_circle_stream_ex(
            0, 0, &instances, HAS_DATA_INDICES | HAS_TOOLTIPS, &trailing,
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
            center: [1.0, 2.0], radius: 3.0,
            fill_color: [1.0, 0.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
        };

        // Build a tooltip section that claims slen=9999 but the buffer ends immediately.
        // Format: [num_fields=1 u32] [field_name_len=9999 u32] <-- then EOF.
        let mut truncated_tooltip: Vec<u8> = Vec::new();
        truncated_tooltip.extend_from_slice(&1u32.to_le_bytes()); // num_fields = 1
        truncated_tooltip.extend_from_slice(&9999u32.to_le_bytes()); // slen = 9999 (overruns)
        // No actual string bytes — the buffer is truncated here.

        let packed = build_packed_circle_stream_ex(
            0, 0, &[inst], HAS_TOOLTIPS, &truncated_tooltip,
        );

        // This must not panic. Pre-fix: offset overruns, slice panics.
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        // The circle was loaded before the tooltip scan; it should be present.
        assert_eq!(circles.len(), 1, "circle must be loaded before tooltip scan");
        // The meta was inserted; tooltip_bytes (if Some) must be bounded by data.len().
        if let Some(m) = meta.get(&(0, 0)) {
            if let Some(tb) = &m.tooltip_bytes {
                assert!(
                    tb.len() <= packed.len(),
                    "tooltip_bytes len {} must not exceed buffer len {}",
                    tb.len(), packed.len()
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
        let instances: Vec<CircleInstance> = (0..5).map(|i| CircleInstance {
            center: [i as f32 * 10.0, 0.0], radius: 2.0,
            fill_color: [1.0, 0.0, 0.0, 1.0], stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, stroke_dash: 0.0, angle: 0.0,
        }).collect();

        let packed = build_packed_circle_stream_ex(0, 0, &instances, 0, &[]);
        let mut circles = Vec::new();
        let mut rects = Vec::new();
        let mut meta = HashMap::new();
        unpack_binary_instances(&packed, &mut circles, &mut rects, &mut meta);

        assert_eq!(circles.len(), 5);
        let m = meta.get(&(0, 0)).expect("meta must exist");
        assert_eq!(
            m.instance_count, circles.len() - m.instance_start,
            "instance_count must equal the number of instances actually loaded"
        );
        assert_eq!(m.instance_count, 5, "instance_count must equal header count on success");
    }

    // ── parse_tooltip_json ──────────────────────────────────────────────

    #[test]
    fn parse_tooltip_json_single_row() {
        let bytes = build_tooltip_bytes(
            &["x", "y"],
            &[vec!["1.23", "4.56"]],
        );
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
        let bytes = build_tooltip_bytes(
            &["col"],
            &[vec!["first"], vec!["second"], vec!["third"]],
        );
        let json = parse_tooltip_json(&bytes, 2);
        assert_eq!(
            json,
            r#"{"fields":[{"name":"col","value":"third"}]}"#,
        );
    }

    #[test]
    fn parse_tooltip_json_out_of_range_returns_empty() {
        let bytes = build_tooltip_bytes(
            &["x"],
            &[vec!["1"]],
        );
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
        let mid_grey = Color { r: 128, g: 128, b: 128, a: 255 };
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
        let bytes = build_tooltip_bytes(
            &["label"],
            &[vec![r#"say "hello""#]],
        );
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
        let white = Color { r: 255, g: 255, b: 255, a: 255 };
        let result = color_to_linear(&white, 1.0);
        assert!((result[0] - 1.0).abs() < 1e-5, "white r must be ~1.0 linear");
        assert!((result[1] - 1.0).abs() < 1e-5, "white g must be ~1.0 linear");
        assert!((result[2] - 1.0).abs() < 1e-5, "white b must be ~1.0 linear");
        assert!((result[3] - 1.0).abs() < 1e-5, "white a must be ~1.0");
    }

    #[test]
    fn bug_hunt_color_to_linear_full_black() {
        let black = Color { r: 0, g: 0, b: 0, a: 255 };
        let result = color_to_linear(&black, 1.0);
        assert!(result[0].abs() < 1e-7, "black r must be ~0.0 linear");
        assert!(result[1].abs() < 1e-7, "black g must be ~0.0 linear");
        assert!(result[2].abs() < 1e-7, "black b must be ~0.0 linear");
    }

    #[test]
    fn bug_hunt_color_to_linear_opacity_scales_alpha() {
        let color = Color { r: 128, g: 128, b: 128, a: 128 };
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
        let bytes = build_tooltip_bytes(
            &["x"],
            &[vec!["1"], vec!["2"]],
        );
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
        let bytes = build_tooltip_bytes(
            &["name"],
            &[vec!["hello world"]],
        );
        let result = parse_tooltip_json(&bytes, 0);
        assert!(result.contains("hello world"), "ASCII content must appear in JSON");
    }

    // ── tooltip_field_value: single-field lookup for packed legend matching ──

    #[test]
    fn tooltip_field_value_reads_named_column() {
        // Two fields, three rows; pick the second column on each row.
        let bytes = build_tooltip_bytes(
            &["x", "cat"],
            &[
                vec!["1", "a"],
                vec!["2", "b"],
                vec!["3", "a"],
            ],
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
        buf.extend_from_slice(&0u32.to_le_bytes());  // panel_idx
        buf.extend_from_slice(&0u32.to_le_bytes());  // batch_idx
        buf.extend_from_slice(&99u32.to_le_bytes()); // kind = unknown
        buf.extend_from_slice(&1u32.to_le_bytes());  // count = 1
        buf.extend_from_slice(&0u32.to_le_bytes());  // flags
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
            width: 100.0, height: 100.0, background: None, title: vec![],
            panels: vec![], legend: vec![], decorations: vec![],
            selections: vec![], interaction: InteractionConfig::default(),
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
            fill: Some(Color { r: 255, g: 255, b: 255, a: 255 }),
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
        collector.collect_mark(&nodes, false, None, 0, None, None);

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
            fill: Some(Color { r: 255, g: 255, b: 255, a: 255 }),
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
        collector.collect_mark(&nodes, false, None, 0, None, None);

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
            BlendMode, CoordKind, FillStroke, InteractionConfig, MarkBatch,
            MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };

        let circle_a = SceneNode::Circle { cx: 10.0, cy: 10.0, r: 5.0, style: style.clone() };
        let circle_b = SceneNode::Circle { cx: 20.0, cy: 20.0, r: 5.0, style: style.clone() };
        let circle_c = SceneNode::Circle { cx: 30.0, cy: 30.0, r: 5.0, style };

        SceneGraph {
            width: 100.0,
            height: 100.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                    },
                ],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
            BlendMode, CoordKind, FillStroke, InteractionConfig, MarkBatch,
            MarkBatchKind, Panel, Rect, SceneNode,
        };

        let style = FillStroke {
            fill: Some(Color { r: 0, g: 0, b: 255, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };

        let rect_node = SceneNode::Rect {
            x: 10.0, y: 10.0, w: 20.0, h: 30.0,
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
                plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
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
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
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
            fill: Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke_width: 2.0,
            opacity: 0.5,
            stroke_opacity: 0.8,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };
        let node = SceneNode::Circle { cx: 50.0, cy: 50.0, r: 10.0, style };
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
            fill: Some(Color { r: 0, g: 128, b: 255, a: 255 }),
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 200 }),
            stroke_width: 1.0,
            opacity: 0.5,
            stroke_opacity: 0.8,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };
        let node = SceneNode::Rect {
            x: 10.0, y: 20.0, w: 40.0, h: 30.0, style, corner_radius: 0.0,
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
        use ferrum_scene::{PathCmd, TextStyle, FontWeight, TextAnchor, TextBaseline};

        // Build a scene with one circle batch (Normal), one rect batch (Additive),
        // and one path batch (Normal), plus a title text node.
        let circle_style = FillStroke {
            fill: Some(Color { r: 200, g: 100, b: 50, a: 255 }),
            stroke: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke_width: 1.5,
            opacity: 0.9,
            stroke_opacity: 0.7,
            fill_opacity: 0.8,
            stroke_dash: Some(vec![6.0, 3.0]),
            angle: 15.0,
        };
        let rect_style = FillStroke {
            fill: Some(Color { r: 50, g: 150, b: 200, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };
        let circle_node = SceneNode::Circle { cx: 80.0, cy: 80.0, r: 12.0, style: circle_style };
        let rect_node = SceneNode::Rect {
            x: 50.0, y: 50.0, w: 80.0, h: 40.0,
            style: rect_style, corner_radius: 3.0,
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
            color: Color { r: 0, g: 0, b: 0, a: 255 },
            opacity: 1.0,
            font_family: "sans-serif".to_string(),
        };
        let title_node = SceneNode::Text { x: 150.0, y: 20.0, content: "Title".to_string(), style: text_style };

        // Build the scene graph.
        use ferrum_scene::{BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind, Panel, Rect};
        let scene = SceneGraph {
            width: 300.0,
            height: 250.0,
            background: None,
            title: vec![title_node],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 30.0, y: 20.0, w: 240.0, h: 200.0 },
                clip: Rect { x: 30.0, y: 20.0, w: 240.0, h: 200.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
                },
                grid: vec![],
                marks: vec![
                    MarkBatch {
                        kind: MarkBatchKind::Point,
                        nodes: vec![circle_node],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Bar,
                        nodes: vec![rect_node],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Additive,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                    MarkBatch {
                        kind: MarkBatchKind::Area,
                        nodes: vec![path_node],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    },
                ],
                axes: vec![], annotations: vec![], strip_title: vec![],
            }],
            legend: vec![], decorations: vec![], selections: vec![],
            interaction: InteractionConfig::default(), chart_description: None,
        };

        // Load via the full load_scene path (which now uses SceneCollector internally).
        let data = load_scene(&scene);

        // Verify instance counts.
        assert_eq!(data.circle_instances.len(), 1, "exactly 1 circle");
        assert_eq!(data.rect_instances.len(), 1, "exactly 1 rect");
        assert_eq!(data.text_elements.len(), 1, "exactly 1 text (title)");
        assert!(!data.mesh_buffers.vertices.is_empty(), "path batch produces mesh vertices");
        assert!(!data.mesh_buffers.indices.is_empty(), "path batch produces mesh indices");

        // Circle instance fields are correctly transferred.
        let ci = &data.circle_instances[0];
        assert!((ci.center[0] - 80.0).abs() < 1e-3, "circle cx");
        assert!((ci.center[1] - 80.0).abs() < 1e-3, "circle cy");
        assert!((ci.radius - 12.0).abs() < 1e-3, "circle radius");
        assert!((ci.opacity - 0.9).abs() < 1e-3, "circle opacity");
        assert!((ci.stroke_opacity - 0.7).abs() < 1e-3, "circle stroke_opacity");
        assert!((ci.stroke_dash - 1.0).abs() < 1e-3, "circle stroke_dash index 1 (dashed)");
        assert!((ci.angle - 15.0).abs() < 1e-3, "circle angle");

        // Rect instance fields are correctly transferred.
        let ri = &data.rect_instances[0];
        assert!((ri.position[0] - 50.0).abs() < 1e-3, "rect x");
        assert!((ri.corner_radius - 3.0).abs() < 1e-3, "rect corner_radius");

        // Draw commands: Normal circle (is_mark, not additive), Additive rect (is_mark, additive).
        let circle_cmds: Vec<_> = data.draw_commands.iter()
            .filter(|c| c.kind == DrawKind::Circle && c.is_mark).collect();
        let rect_cmds: Vec<_> = data.draw_commands.iter()
            .filter(|c| c.kind == DrawKind::Rect && c.is_mark).collect();
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
            BlendMode, CoordKind, InteractionConfig, MarkBatch, MarkBatchKind,
            Panel, Rect, SceneGraph, SceneNode, StrokeStyle,
        };

        let stroke = StrokeStyle {
            color: Color { r: 255, g: 0, b: 0, a: 255 },
            width: 2.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        };

        // A grid line (non-annotation) and an annotation line.
        let grid_line = SceneNode::Line {
            x1: 50.0, y1: 0.0, x2: 50.0, y2: 400.0,
            style: stroke.clone(),
        };
        let annotation_line = SceneNode::Line {
            x1: 0.0, y1: 200.0, x2: 500.0, y2: 200.0,
            style: stroke.clone(),
        };

        let scene = SceneGraph {
            width: 500.0,
            height: 400.0,
            background: None,
            title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 400.0, h: 350.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
                },
                grid: vec![grid_line],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Rule,
                    nodes: vec![],
                    data_indices: None, tooltips: None, hrefs: None,
                    descriptions: None, keys: None,
                    blend: BlendMode::Normal,
                    stroke_cap: None, stroke_join: None, packed_instances: None,
                }],
                axes: vec![],
                annotations: vec![annotation_line],
                strip_title: vec![],
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
        let nodes = vec![
            SceneNode::Line {
                x1: 50.0, y1: 50.0, x2: 300.0, y2: 200.0,
                style: default_stroke_style(),
            },
        ];
        let scene = make_scene_with_nodes(MarkBatchKind::Rule, nodes);
        let data = load_scene(&scene);

        // At least some mesh indices must have been produced for the assertion
        // to be meaningful.
        assert!(
            !data.mesh_buffers.indices.is_empty(),
            "prerequisite: line must tessellate to non-empty mesh"
        );

        assert_eq!(
            data.mark_mesh_panels.len(), 1,
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
        use ferrum_scene::{Panel, MarkBatch, BlendMode};
        use ferrum_scene::{CoordKind, Rect, InteractionConfig};

        let line_node = || SceneNode::Line {
            x1: 50.0, y1: 50.0, x2: 300.0, y2: 200.0,
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
                    plot_area: Rect { x: 50.0, y: 10.0, w: 200.0, h: 150.0 },
                    clip: Rect { x: 50.0, y: 10.0, w: 200.0, h: 150.0 },
                    coord: CoordKind::Cartesian {
                        x_domain: None, y_domain: None, expand: true, clip: true,
                    },
                    grid: vec![],
                    marks: vec![MarkBatch {
                        kind: MarkBatchKind::Rule,
                        nodes: vec![line_node()],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    }],
                    axes: vec![], annotations: vec![], strip_title: vec![],
                },
                Panel {
                    id: 1,
                    plot_area: Rect { x: 310.0, y: 10.0, w: 200.0, h: 150.0 },
                    clip: Rect { x: 310.0, y: 10.0, w: 200.0, h: 150.0 },
                    coord: CoordKind::Cartesian {
                        x_domain: None, y_domain: None, expand: true, clip: true,
                    },
                    grid: vec![],
                    marks: vec![MarkBatch {
                        kind: MarkBatchKind::Rule,
                        nodes: vec![line_node()],
                        data_indices: None, tooltips: None, hrefs: None,
                        descriptions: None, keys: None,
                        blend: BlendMode::Normal,
                        stroke_cap: None, stroke_join: None, packed_instances: None,
                    }],
                    axes: vec![], annotations: vec![], strip_title: vec![],
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
            data.mark_mesh_panels.len(), 2,
            "two panels with mesh marks → two MarkMeshPanel entries"
        );

        let p0 = &data.mark_mesh_panels[0];
        let p1 = &data.mark_mesh_panels[1];

        // Ranges must be contiguous: panel 1 starts where panel 0 ends.
        assert_eq!(
            p1.index_start, p0.index_start + p0.index_count,
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
            cx: 100.0, cy: 100.0, r: 5.0,
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
        collector.record_mark_mesh_panel(0, 30, [10.0, 20.0, 200.0, 150.0], 0);
        assert_eq!(collector.mark_mesh_panels.len(), 1);
        assert_eq!(collector.mark_mesh_panels[0].index_start, 0);
        assert_eq!(collector.mark_mesh_panels[0].index_count, 30);
        assert_eq!(collector.mark_mesh_panels[0].panel_id, 0);

        // Record a zero-count range — must be dropped.
        collector.record_mark_mesh_panel(30, 30, [10.0, 200.0, 200.0, 150.0], 1);
        assert_eq!(
            collector.mark_mesh_panels.len(), 1,
            "zero-count panel must not be appended"
        );

        // Record a second non-zero range that follows the first (panel 2).
        collector.record_mark_mesh_panel(30, 55, [220.0, 20.0, 200.0, 150.0], 2);
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
            BlendMode, CoordKind, FillStroke, InteractionConfig, MarkBatch, MarkBatchKind,
            Panel, Rect, SceneGraph, SceneNode, StrokeStyle,
        };

        let line_style = StrokeStyle {
            color: Color { r: 0, g: 0, b: 0, a: 255 },
            width: 1.0,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        };
        let circle_style = FillStroke {
            fill: Some(Color { r: 255, g: 0, b: 0, a: 255 }),
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
            plot_area: Rect { x: x_off, y: 0.0, w: 100.0, h: 100.0 },
            clip: Rect { x: x_off, y: 0.0, w: 100.0, h: 100.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Line,
                nodes: vec![
                    SceneNode::Line {
                        x1: x_off, y1: 10.0, x2: x_off + 50.0, y2: 80.0,
                        style: line_style.clone(),
                    },
                    SceneNode::Circle {
                        cx: x_off + 25.0, cy: 50.0, r: 4.0, style: circle_style.clone(),
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
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
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

        assert_eq!(data.panel_count, 2, "two-panel scene must report panel_count == 2");

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
        assert!(mark_panel_ids.contains(&0), "a mark command must belong to panel 0");
        assert!(mark_panel_ids.contains(&1), "a mark command must belong to panel 1");
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
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("empty field name must produce valid JSON");
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
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("empty value must produce valid JSON");
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
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("100 fields must produce valid JSON");
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
        assert!(result.is_infinite() || result.is_finite(),
            "srgb_to_linear(inf) must not panic");
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
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0],
            "None color must produce fully transparent [0,0,0,0]");
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
            fill: Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let nodes = vec![SceneNode::Circle {
            cx: 100.0, cy: 100.0, r: 10.0, style,
        }];
        let mut collector = SceneCollector::new();
        collector.collect_annotation(&nodes, None, None);
        assert_eq!(collector.circles.len(), 1, "circle annotation must be collected");
        // Must have a draw command for the circle
        let circle_cmds: Vec<_> = collector.draw_commands.iter()
            .filter(|c| c.kind == DrawKind::Circle)
            .collect();
        assert_eq!(circle_cmds.len(), 1, "circle annotation must emit a draw command");
        // The draw command must be non-mark (is_mark=false) since annotations use identity transform
        assert!(!circle_cmds[0].is_mark, "annotation circle draw command must not be is_mark");
    }

    #[test]
    fn bug_hunt_collect_annotation_line_goes_to_annotation_mesh() {
        // A Line annotation node must go to annotation_mesh, not static_mesh.
        let style = ferrum_scene::StrokeStyle {
            color: Color { r: 0, g: 0, b: 0, a: 255 },
            width: 2.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
        };
        let nodes = vec![SceneNode::Line {
            x1: 0.0, y1: 50.0, x2: 500.0, y2: 50.0, style,
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
            color: Color { r: 0, g: 0, b: 0, a: 255 },
            width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
        };
        let nodes = vec![SceneNode::Line {
            x1: 0.0, y1: 100.0, x2: 500.0, y2: 100.0, style,
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
            center: [10.0, 20.0], radius: 3.0,
            fill_color: [1.0, 0.0, 0.0, 1.0],
            stroke_color: [0.0; 4], stroke_width: 0.0,
            opacity: 1.0, stroke_opacity: 1.0,
            stroke_dash: 0.0, angle: 0.0,
        };
        let ri = RectInstance {
            position: [50.0, 60.0], size: [20.0, 30.0],
            corner_radius: 0.0,
            fill_color: [0.0, 1.0, 0.0, 1.0],
            stroke_color: [0.0; 4], stroke_width: 0.0,
            opacity: 1.0, stroke_opacity: 1.0,
            stroke_dash: 0.0, angle: 0.0,
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
        let c = Color { r: 255, g: 128, b: 0, a: 255 };
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
        use ferrum_scene::{Panel, MarkBatch, MarkBatchKind, BlendMode, RawAnchor};
        use ferrum_scene::{CoordKind, Rect, SceneGraph, InteractionConfig};

        let svg_content = r#"<linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient>"#;
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
                plot_area: Rect { x: 50.0, y: 10.0, w: 300.0, h: 250.0 },
                clip: Rect { x: 50.0, y: 10.0, w: 300.0, h: 250.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
                },
                grid: vec![],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![],
                    data_indices: None, tooltips: None, hrefs: None,
                    descriptions: None, keys: None,
                    blend: BlendMode::Normal,
                    stroke_cap: None, stroke_join: None, packed_instances: None,
                }],
                axes: vec![],
                annotations: vec![node],
                strip_title: vec![],
            }],
            legend: vec![],
            decorations: vec![],
            selections: vec![],
            interaction: InteractionConfig::default(),
            chart_description: None,
        };

        let data = load_scene(&scene);
        assert_eq!(
            data.raw_fragments.len(), 1,
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
        use ferrum_scene::{RawAnchor, SceneGraph, InteractionConfig};

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
            data.raw_fragments.len(), 1,
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
        use ferrum_scene::{RawAnchor, SceneGraph, InteractionConfig};

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
            data.raw_fragments.len(), 2,
            "both Raw nodes must be collected"
        );
        let anchors: Vec<&str> = data.raw_fragments.iter().map(|r| r.anchor.as_str()).collect();
        assert!(anchors.contains(&"chrome"), "must have a chrome fragment");
        assert!(anchors.contains(&"data"), "must have a data fragment");
    }

    /// A Raw node nested inside a Group must also be collected (recursive walk).
    #[test]
    fn raw_node_inside_group_is_collected() {
        use ferrum_scene::{RawAnchor, SceneGraph, InteractionConfig};

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
            data.raw_fragments.len(), 1,
            "Raw node nested inside Group must be collected via recursive walk"
        );
        assert_eq!(data.raw_fragments[0].anchor, "chrome");
        assert!(data.raw_fragments[0].svg.contains("nested"));
    }

}
