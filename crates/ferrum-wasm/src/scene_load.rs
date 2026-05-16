use std::collections::HashMap;

use ferrum_scene::*;
use lyon::tessellation::VertexBuffers;

use crate::tessellate::{self, MeshVertex};

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

#[derive(Clone)]
pub struct SceneData {
    pub circle_instances: Vec<CircleInstance>,
    pub rect_instances: Vec<RectInstance>,
    pub mesh_buffers: VertexBuffers<MeshVertex, u32>,
    pub text_elements: Vec<TextElementData>,
    pub image_quads: Vec<ImageQuad>,
    pub background: Option<[f32; 4]>,
    pub width: f32,
    pub height: f32,
    /// Per-batch metadata from the packed binary sidecar, keyed by
    /// `(panel_idx, batch_idx)`.
    pub packed_batch_meta: HashMap<(u32, u32), PackedBatchMeta>,
}

#[derive(Clone)]
pub struct TextElementData {
    pub x: f64,
    pub y: f64,
    pub content: String,
    pub style: TextStyle,
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

pub fn load_scene(scene: &SceneGraph) -> SceneData {
    load_scene_with_packed(scene, &[])
}

pub fn load_scene_with_packed(scene: &SceneGraph, packed_data: &[u8]) -> SceneData {
    let mut circles = Vec::new();
    let mut rects = Vec::new();
    let mut mesh = VertexBuffers::new();
    let mut images = Vec::new();
    let mut texts = Vec::new();
    let mut batch_meta = HashMap::new();

    // Unpack binary instance data (passed as raw bytes, not base64).
    unpack_binary_instances(packed_data, &mut circles, &mut rects, &mut batch_meta);

    let background = scene.background.as_ref().map(|c| {
        [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        ]
    });

    collect_nodes(&scene.title, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);

    for panel in &scene.panels {
        collect_nodes(&panel.grid, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);

        for batch in &panel.marks {
            collect_nodes(
                &batch.nodes,
                &mut circles, &mut rects, &mut mesh, &mut texts, &mut images,
                batch.stroke_cap, batch.stroke_join,
            );
        }

        collect_nodes(&panel.axes, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
        collect_nodes(&panel.strip_title, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
        collect_nodes(&panel.annotations, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
    }

    collect_nodes(&scene.legend, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);
    collect_nodes(&scene.decorations, &mut circles, &mut rects, &mut mesh, &mut texts, &mut images, None, None);

    SceneData {
        circle_instances: circles,
        rect_instances: rects,
        mesh_buffers: mesh,
        text_elements: texts,
        image_quads: images,
        background,
        width: scene.width as f32,
        height: scene.height as f32,
        packed_batch_meta: batch_meta,
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
        let (instance_byte_len, instance_start) = match kind {
            0 => {
                let byte_len = count * std::mem::size_of::<CircleInstance>();
                if offset + byte_len > data.len() { break; }
                let start = circles.len();
                if let Ok(instances) = bytemuck::try_cast_slice(&data[offset..offset+byte_len]) {
                    circles.extend_from_slice(instances);
                }
                (byte_len, start)
            }
            1 => {
                let byte_len = count * std::mem::size_of::<RectInstance>();
                if offset + byte_len > data.len() { break; }
                let start = rects.len();
                if let Ok(instances) = bytemuck::try_cast_slice(&data[offset..offset+byte_len]) {
                    rects.extend_from_slice(instances);
                }
                (byte_len, start)
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

            Some(data[scan_start..offset].to_vec())
        } else {
            None
        };

        meta.insert(
            (panel_idx, batch_idx),
            PackedBatchMeta {
                data_indices, tooltip_bytes,
                kind, instance_start, instance_count: count,
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
    batch_cap: Option<StrokeCap>,
    batch_join: Option<StrokeJoin>,
) {
    for node in nodes {
        match node {
            SceneNode::Circle { cx, cy, r, style } => {
                circles.push(CircleInstance {
                    center: [*cx as f32, *cy as f32],
                    radius: *r as f32,
                    fill_color: opt_color_to_f32(style.fill.as_ref(), style.opacity),
                    stroke_color: opt_color_to_f32(style.stroke.as_ref(), style.opacity),
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
                    fill_color: opt_color_to_f32(style.fill.as_ref(), style.opacity),
                    stroke_color: opt_color_to_f32(style.stroke.as_ref(), style.opacity),
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
                if let ImageData::Inline { bytes, .. } = data {
                    if let Some(quad) = decode_image_quad(*x, *y, *w, *h, bytes) {
                        images.push(quad);
                    }
                }
            }
            SceneNode::Group { children, .. } => {
                collect_nodes(children, circles, rects, mesh, texts, images, batch_cap, batch_join);
            }
            SceneNode::Raw { .. } => {
                web_sys::console::warn_1(
                    &"ferrum: Raw SVG node skipped in WASM renderer".into(),
                );
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
        Some(c) => [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            (c.a as f32 / 255.0) * opacity as f32,
        ],
        None => [0.0, 0.0, 0.0, 0.0],
    }
}

/// Map a `FillStroke.stroke_dash` pattern vector to the closest palette index.
///
/// `None` / empty vec → 0.0 (solid). Otherwise matches the first non-empty
/// palette pattern by round-trip string comparison of the vec joined with ",".
/// Falls back to 0.0 (solid) for unrecognised patterns.
fn stroke_dash_index(dash: &Option<Vec<f64>>) -> f32 {
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
}
