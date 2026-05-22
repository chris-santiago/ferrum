//! Packed-instance extraction for high-cardinality mark batches.
//!
//! When a `MarkBatch` contains >1000 homogeneous circle or rect nodes,
//! `extract_packed_bytes` packs their GPU instance data as raw bytes.
//! The bytes travel to the WASM renderer as a separate `Uint8Array`,
//! completely bypassing JSON serialization and parsing.
//!
//! The byte layout matches `CircleInstance` (16 × f32 = 64 bytes) and
//! `RectInstance` (18 × f32 = 72 bytes) defined in `ferrum-wasm/src/scene_load.rs`.
//!
//! ## Extended header (v2)
//!
//! Each packed batch now uses a 20-byte header:
//! ```text
//! [panel_idx: u32][batch_idx: u32][kind: u32][count: u32][flags: u32]
//! ```
//! After instance data, optional sections follow based on `flags`:
//! - `HAS_DATA_INDICES (0x2)`: `count × u32` little-endian data indices
//! - `HAS_TOOLTIPS (0x1)`: length-prefixed string table (field names then row values)

use ferrum_scene::{Color, FillStroke, MarkBatchKind, SceneGraph, SceneNode};

/// Minimum node count for a batch to qualify for packing.
const PACK_THRESHOLD: usize = 1000;

/// Flag: tooltips string table follows instance data (+ optional data_indices).
const HAS_TOOLTIPS: u32 = 0x1;
/// Flag: `count × u32` data-index array follows instance data.
const HAS_DATA_INDICES: u32 = 0x2;

/// Extract packed instance bytes from large homogeneous mark batches.
///
/// Returns the concatenated raw bytes for all qualifying batches. The nodes
/// in each qualifying batch are cleared — the packed binary data is the sole
/// representation. The WASM renderer receives these bytes as a separate
/// `Uint8Array`, bypassing JSON entirely.
///
/// Byte layout per batch:
///   `[header 20B][instance_data][data_indices?][tooltips?]`
///
/// Header (20 bytes):
///   `[panel_idx: u32][batch_idx: u32][kind: u32][count: u32][flags: u32]`
///
/// After instance data, optional trailing sections appear in flag order:
///   - `HAS_DATA_INDICES`: `count × u32` LE data indices
///   - `HAS_TOOLTIPS`: length-prefixed string table (see `pack_tooltips`)
pub fn extract_packed_bytes(scene: &mut SceneGraph) -> Vec<u8> {
    let mut packed = Vec::new();
    for (pi, panel) in scene.panels.iter_mut().enumerate() {
        for (bi, batch) in panel.marks.iter_mut().enumerate() {
            let n = batch.nodes.len();
            if n < PACK_THRESHOLD {
                continue;
            }

            let instance_bytes = match batch.kind {
                MarkBatchKind::Point if all_circles(&batch.nodes) => {
                    pack_circle_batch(&batch.nodes)
                }
                MarkBatchKind::Bar | MarkBatchKind::Rect if all_rects(&batch.nodes) => {
                    pack_rect_batch(&batch.nodes)
                }
                _ => continue,
            };

            let kind_tag: u32 = match batch.kind {
                MarkBatchKind::Point => 0,
                MarkBatchKind::Bar | MarkBatchKind::Rect => 1,
                _ => continue,
            };

            // Build flags and optional trailing sections.
            let mut flags: u32 = 0;
            let mut trailing = Vec::new();

            // data_indices — written first after instance bytes.
            if let Some(ref indices) = batch.data_indices {
                flags |= HAS_DATA_INDICES;
                for &idx in indices {
                    trailing.extend_from_slice(&(idx as u32).to_le_bytes());
                }
            }

            // tooltips — written after data_indices (if any).
            if let Some(ref tooltips) = batch.tooltips {
                if !tooltips.is_empty() {
                    flags |= HAS_TOOLTIPS;
                    pack_tooltips(tooltips, &mut trailing);
                }
            }

            // Header: panel_idx, batch_idx, kind, count, flags (20 bytes)
            packed.extend_from_slice(&(pi as u32).to_le_bytes());
            packed.extend_from_slice(&(bi as u32).to_le_bytes());
            packed.extend_from_slice(&kind_tag.to_le_bytes());
            packed.extend_from_slice(&(n as u32).to_le_bytes());
            packed.extend_from_slice(&flags.to_le_bytes());
            packed.extend_from_slice(&instance_bytes);
            packed.extend_from_slice(&trailing);

            batch.nodes.clear();

            // Clear the fields we just packed — they are now in the binary sidecar.
            if flags & HAS_DATA_INDICES != 0 {
                batch.data_indices = None;
            }
            if flags & HAS_TOOLTIPS != 0 {
                batch.tooltips = None;
            }
        }
    }
    packed
}

// ── Tooltip string-table packing ────────────────────────────────────

/// Append a tooltip string table to `buf`.
///
/// Layout:
/// ```text
/// [num_fields: u32]
/// [field_name_0_len: u32][field_name_0 UTF-8 bytes]
/// …
/// [row_0_field_0_value_len: u32][row_0_field_0_value UTF-8 bytes]
/// [row_0_field_1_value_len: u32][row_0_field_1_value UTF-8 bytes]
/// …
/// [row_N-1_field_M-1_value_len: u32][row_N-1_field_M-1 UTF-8 bytes]
/// ```
fn pack_tooltips(tooltips: &[ferrum_scene::TooltipContent], buf: &mut Vec<u8>) {
    // Field names come from the first row.
    let num_fields = tooltips[0].fields.len() as u32;
    buf.extend_from_slice(&num_fields.to_le_bytes());

    // Write field names.
    for field in &tooltips[0].fields {
        let name_bytes = field.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
    }

    // Write per-row values in field order.
    for row in tooltips {
        for field in &row.fields {
            let val_bytes = field.value.as_bytes();
            buf.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(val_bytes);
        }
    }
}

// ── Predicate helpers ────────────────────────────────────────────────

fn all_circles(nodes: &[SceneNode]) -> bool {
    nodes.iter().all(|n| matches!(n, SceneNode::Circle { .. }))
}

fn all_rects(nodes: &[SceneNode]) -> bool {
    nodes.iter().all(|n| matches!(n, SceneNode::Rect { .. }))
}

// ── Circle packing ──────────────────────────────────────────────────

/// Pack circle nodes into raw bytes matching `CircleInstance` layout.
///
/// Layout per instance (16 × f32 = 64 bytes):
///   center_x, center_y, radius,
///   fill_r, fill_g, fill_b, fill_a,
///   stroke_r, stroke_g, stroke_b, stroke_a,
///   stroke_width, opacity, stroke_opacity, stroke_dash, angle
pub fn pack_circle_batch(nodes: &[SceneNode]) -> Vec<u8> {
    const FLOATS_PER_CIRCLE: usize = 16;
    let mut buf = Vec::with_capacity(nodes.len() * FLOATS_PER_CIRCLE * 4);

    for node in nodes {
        if let SceneNode::Circle { cx, cy, r, style } = node {
            push_f32(&mut buf, *cx as f32);
            push_f32(&mut buf, *cy as f32);
            push_f32(&mut buf, *r as f32);
            push_color(&mut buf, style.fill.as_ref(), style.fill_opacity);
            push_color(&mut buf, style.stroke.as_ref(), 1.0);
            push_f32(&mut buf, style.stroke_width as f32);
            push_f32(&mut buf, style.opacity as f32);
            push_f32(&mut buf, style.stroke_opacity as f32);
            push_f32(&mut buf, stroke_dash_index(&style.stroke_dash));
            push_f32(&mut buf, style.angle as f32);
        }
    }

    buf
}

// ── Rect packing ────────────────────────────────────────────────────

/// Pack rect nodes into raw bytes matching `RectInstance` layout.
///
/// Layout per instance (18 × f32 = 72 bytes):
///   position_x, position_y, size_w, size_h, corner_radius,
///   fill_r, fill_g, fill_b, fill_a,
///   stroke_r, stroke_g, stroke_b, stroke_a,
///   stroke_width, opacity, stroke_opacity, stroke_dash, angle
pub fn pack_rect_batch(nodes: &[SceneNode]) -> Vec<u8> {
    const FLOATS_PER_RECT: usize = 18;
    let mut buf = Vec::with_capacity(nodes.len() * FLOATS_PER_RECT * 4);

    for node in nodes {
        if let SceneNode::Rect {
            x,
            y,
            w,
            h,
            style,
            corner_radius,
        } = node
        {
            push_f32(&mut buf, *x as f32);
            push_f32(&mut buf, *y as f32);
            push_f32(&mut buf, *w as f32);
            push_f32(&mut buf, *h as f32);
            push_f32(&mut buf, *corner_radius as f32);
            push_color(&mut buf, style.fill.as_ref(), style.fill_opacity);
            push_color(&mut buf, style.stroke.as_ref(), 1.0);
            push_f32(&mut buf, style.stroke_width as f32);
            push_f32(&mut buf, style.opacity as f32);
            push_f32(&mut buf, style.stroke_opacity as f32);
            push_f32(&mut buf, stroke_dash_index(&style.stroke_dash));
            push_f32(&mut buf, style.angle as f32);
        }
    }

    buf
}

// ── Low-level byte helpers ──────────────────────────────────────────

#[inline]
fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Convert an optional Color to [f32; 4] and push all four components.
/// Matches the WASM renderer's `opt_color_to_f32` exactly:
///   rgb → [0..1], alpha → (color.a / 255) * alpha_factor
///
/// For fill colors, `alpha_factor` is `fill_opacity`.
/// For stroke colors, `alpha_factor` is `1.0` (raw color; the shader
/// applies `stroke_opacity` and `opacity` independently).
#[inline]
fn push_color(buf: &mut Vec<u8>, color: Option<&Color>, alpha_factor: f64) {
    match color {
        Some(c) => {
            push_f32(buf, c.r as f32 / 255.0);
            push_f32(buf, c.g as f32 / 255.0);
            push_f32(buf, c.b as f32 / 255.0);
            push_f32(buf, (c.a as f32 / 255.0) * alpha_factor as f32);
        }
        None => {
            push_f32(buf, 0.0);
            push_f32(buf, 0.0);
            push_f32(buf, 0.0);
            push_f32(buf, 0.0);
        }
    }
}

/// Map a stroke-dash pattern to the same palette index used by the WASM
/// renderer: 0 = solid, 1 = dashed [6,3], 2 = dotted [2,3], 3 = dash-dot [6,3,2,3].
fn stroke_dash_index(dash: &Option<Vec<f64>>) -> f32 {
    match dash {
        None => 0.0,
        Some(v) if v.is_empty() => 0.0,
        Some(v) => {
            let pattern: Vec<u64> = v.iter().map(|&x| x as u64).collect();
            match pattern.as_slice() {
                [6, 3] => 1.0,
                [2, 3] => 2.0,
                [6, 3, 2, 3] => 3.0,
                _ => 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::Color;

    fn test_style(fill_r: u8, opacity: f64) -> FillStroke {
        FillStroke {
            fill: Some(Color::rgba(fill_r, 100, 200, 255)),
            stroke: Some(Color::rgb(0, 0, 0)),
            stroke_width: 1.5,
            opacity,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    #[test]
    fn pack_circle_batch_produces_correct_byte_count() {
        let nodes: Vec<SceneNode> = (0..100)
            .map(|i| SceneNode::Circle {
                cx: i as f64,
                cy: i as f64 * 2.0,
                r: 5.0,
                style: test_style(70, 0.8),
            })
            .collect();
        let bytes = pack_circle_batch(&nodes);
        // 16 floats × 4 bytes × 100 instances = 6400 bytes
        assert_eq!(bytes.len(), 100 * 16 * 4);
    }

    #[test]
    fn pack_rect_batch_produces_correct_byte_count() {
        let nodes: Vec<SceneNode> = (0..50)
            .map(|i| SceneNode::Rect {
                x: i as f64 * 10.0,
                y: 0.0,
                w: 8.0,
                h: 100.0,
                style: test_style(128, 1.0),
                corner_radius: 2.0,
            })
            .collect();
        let bytes = pack_rect_batch(&nodes);
        // 18 floats × 4 bytes × 50 instances = 3600 bytes
        assert_eq!(bytes.len(), 50 * 18 * 4);
    }

    #[test]
    fn pack_circle_matches_wasm_layout() {
        // Single circle — verify the exact byte layout matches CircleInstance.
        let style = FillStroke {
            fill: Some(Color::rgba(255, 0, 0, 255)),
            stroke: Some(Color::rgba(0, 0, 0, 128)),
            stroke_width: 2.0,
            opacity: 0.5,
            stroke_dash: Some(vec![6.0, 3.0]), // dashed → index 1
            stroke_opacity: 0.75,
            fill_opacity: 1.0,
            angle: 45.0,
        };
        let nodes = vec![SceneNode::Circle {
            cx: 100.0,
            cy: 200.0,
            r: 10.0,
            style,
        }];
        let bytes = pack_circle_batch(&nodes);
        assert_eq!(bytes.len(), 64); // 16 × 4

        // Parse back as f32 values
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(floats.len(), 16);
        // center
        assert!((floats[0] - 100.0).abs() < 1e-6, "center_x");
        assert!((floats[1] - 200.0).abs() < 1e-6, "center_y");
        // radius
        assert!((floats[2] - 10.0).abs() < 1e-6, "radius");
        // fill_color: (255/255, 0, 0, (255/255)*fill_opacity) = (1.0, 0.0, 0.0, 1.0)
        // fill uses fill_opacity (1.0), not opacity (0.5)
        assert!((floats[3] - 1.0).abs() < 1e-5, "fill_r");
        assert!((floats[4] - 0.0).abs() < 1e-5, "fill_g");
        assert!((floats[5] - 0.0).abs() < 1e-5, "fill_b");
        assert!((floats[6] - 1.0).abs() < 1e-5, "fill_a = (255/255)*fill_opacity(1.0)");
        // stroke_color: raw color, no opacity baked in — shader applies stroke_opacity and opacity
        // (0, 0, 0, (128/255)*1.0)
        assert!((floats[7] - 0.0).abs() < 1e-5, "stroke_r");
        assert!((floats[8] - 0.0).abs() < 1e-5, "stroke_g");
        assert!((floats[9] - 0.0).abs() < 1e-5, "stroke_b");
        let expected_stroke_a = 128.0_f32 / 255.0;
        assert!(
            (floats[10] - expected_stroke_a).abs() < 1e-5,
            "stroke_a = (128/255)*1.0 (raw, no opacity baked)"
        );
        // stroke_width
        assert!((floats[11] - 2.0).abs() < 1e-6, "stroke_width");
        // opacity
        assert!((floats[12] - 0.5).abs() < 1e-6, "opacity");
        // stroke_opacity
        assert!((floats[13] - 0.75).abs() < 1e-6, "stroke_opacity");
        // stroke_dash (dashed → 1.0)
        assert!((floats[14] - 1.0).abs() < 1e-6, "stroke_dash index");
        // angle
        assert!((floats[15] - 45.0).abs() < 1e-6, "angle");
    }

    #[test]
    fn extract_packed_bytes_packs_circles_above_threshold() {
        use ferrum_scene::*;

        let n = PACK_THRESHOLD + 10;
        let nodes: Vec<SceneNode> = (0..n)
            .map(|i| SceneNode::Circle {
                cx: i as f64, cy: i as f64, r: 3.0,
                style: test_style(70, 1.0),
            })
            .collect();

        let mut scene = test_scene(MarkBatchKind::Point, nodes);
        let bytes = extract_packed_bytes(&mut scene);

        // Header: 20 bytes + instance data: n × 16 f32 × 4 bytes
        let expected = 20 + n * 16 * 4;
        assert_eq!(bytes.len(), expected, "packed byte count");

        // Verify header (20 bytes: panel_idx, batch_idx, kind, count, flags)
        let panel_idx = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let batch_idx = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let kind = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(panel_idx, 0);
        assert_eq!(batch_idx, 0);
        assert_eq!(kind, 0); // circle
        assert_eq!(count, n as u32);
        assert_eq!(flags, 0, "no data_indices or tooltips → flags = 0");

        // Nodes should be cleared after extraction
        assert!(scene.panels[0].marks[0].nodes.is_empty(), "nodes must be cleared");
    }

    #[test]
    fn extract_packed_bytes_skips_small_batches() {
        use ferrum_scene::*;

        let nodes: Vec<SceneNode> = (0..100)
            .map(|i| SceneNode::Circle {
                cx: i as f64, cy: i as f64, r: 3.0,
                style: test_style(70, 1.0),
            })
            .collect();

        let mut scene = test_scene(MarkBatchKind::Point, nodes);
        let bytes = extract_packed_bytes(&mut scene);

        assert!(bytes.is_empty(), "small batch should produce no packed bytes");
        assert_eq!(scene.panels[0].marks[0].nodes.len(), 100, "nodes should be preserved");
    }

    fn test_scene(kind: MarkBatchKind, nodes: Vec<SceneNode>) -> ferrum_scene::SceneGraph {
        test_scene_with_metadata(kind, nodes, None, None)
    }

    fn test_scene_with_metadata(
        kind: MarkBatchKind,
        nodes: Vec<SceneNode>,
        data_indices: Option<Vec<usize>>,
        tooltips: Option<Vec<ferrum_scene::TooltipContent>>,
    ) -> ferrum_scene::SceneGraph {
        use ferrum_scene::*;
        SceneGraph {
            width: 500.0, height: 400.0, background: None, title: vec![],
            panels: vec![Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 400.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 400.0 },
                coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
                grid: vec![],
                marks: vec![MarkBatch {
                    kind, nodes,
                    data_indices, tooltips, hrefs: None,
                    descriptions: None, keys: None,
                    blend: BlendMode::Normal, stroke_cap: None, stroke_join: None,
                    packed_instances: None,
                }],
                axes: vec![], annotations: vec![], strip_title: vec![],
            }],
            legend: vec![], decorations: vec![], selections: vec![],
            interaction: InteractionConfig::default(), chart_description: None,
        }
    }

    #[test]
    fn extract_packed_bytes_packs_data_indices_and_tooltips() {
        use ferrum_scene::*;

        let n = PACK_THRESHOLD + 5;
        let nodes: Vec<SceneNode> = (0..n)
            .map(|i| SceneNode::Circle {
                cx: i as f64, cy: i as f64, r: 3.0,
                style: test_style(70, 1.0),
            })
            .collect();

        let data_indices: Vec<usize> = (0..n).collect();
        let tooltips: Vec<TooltipContent> = (0..n)
            .map(|i| TooltipContent {
                fields: vec![
                    TooltipField { name: "x".into(), value: format!("{}.0", i) },
                    TooltipField { name: "y".into(), value: format!("{}.0", i * 2) },
                ],
            })
            .collect();

        let mut scene = test_scene_with_metadata(
            MarkBatchKind::Point,
            nodes,
            Some(data_indices),
            Some(tooltips),
        );
        let bytes = extract_packed_bytes(&mut scene);

        // ── Parse header ──
        assert!(bytes.len() >= 20, "must have at least a 20-byte header");
        let panel_idx = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let batch_idx = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let kind = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[16..20].try_into().unwrap());

        assert_eq!(panel_idx, 0);
        assert_eq!(batch_idx, 0);
        assert_eq!(kind, 0);
        assert_eq!(count, n as u32);
        assert_eq!(flags, HAS_DATA_INDICES | HAS_TOOLTIPS, "both flags set");

        // ── Skip instance data ──
        let instance_size = n * 16 * 4; // 16 f32 per circle
        let mut cursor = 20 + instance_size;

        // ── Verify data_indices section ──
        for i in 0..n {
            let idx = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            assert_eq!(idx, i as u32, "data_index[{i}]");
            cursor += 4;
        }

        // ── Verify tooltips section ──
        let num_fields = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(num_fields, 2, "two tooltip fields: x, y");
        cursor += 4;

        // Field names
        let read_str = |cur: &mut usize| -> String {
            let len = u32::from_le_bytes(bytes[*cur..*cur + 4].try_into().unwrap()) as usize;
            *cur += 4;
            let s = std::str::from_utf8(&bytes[*cur..*cur + len]).unwrap().to_string();
            *cur += len;
            s
        };
        assert_eq!(read_str(&mut cursor), "x");
        assert_eq!(read_str(&mut cursor), "y");

        // Spot-check first and last row values.
        // Row 0: "0.0", "0.0"
        assert_eq!(read_str(&mut cursor), "0.0");
        assert_eq!(read_str(&mut cursor), "0.0");

        // Skip to last row: each row is 2 string-table entries.
        // We already consumed row 0 (2 entries). Skip rows 1..n-1.
        for _ in 1..n - 1 {
            // Skip 2 fields per row.
            for _ in 0..2 {
                let len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4 + len;
            }
        }

        // Last row: format!("{}.0", n-1), format!("{}.0", (n-1)*2)
        let last = n - 1;
        assert_eq!(read_str(&mut cursor), format!("{last}.0"));
        assert_eq!(read_str(&mut cursor), format!("{}.0", last * 2));

        // cursor should be at the end of bytes
        assert_eq!(cursor, bytes.len(), "all bytes consumed");

        // ── Verify fields cleared on batch ──
        let batch = &scene.panels[0].marks[0];
        assert!(batch.nodes.is_empty(), "nodes cleared");
        assert!(batch.data_indices.is_none(), "data_indices cleared");
        assert!(batch.tooltips.is_none(), "tooltips cleared");
    }

    #[test]
    fn stroke_dash_index_maps_correctly() {
        assert!((stroke_dash_index(&None) - 0.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![])) - 0.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![6.0, 3.0])) - 1.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![2.0, 3.0])) - 2.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![6.0, 3.0, 2.0, 3.0])) - 3.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![5.0, 5.0])) - 0.0).abs() < 1e-6);
    }

    // ── B1/B3: opacity semantics regression tests ─────────────────────

    #[test]
    fn packed_circle_fill_uses_fill_opacity_not_opacity() {
        // fill_opacity=0.5, opacity=1.0 → fill_a = (255/255)*0.5 = 0.5
        let style = FillStroke {
            fill: Some(Color::rgba(255, 255, 255, 255)),
            stroke: Some(Color::rgb(0, 0, 0)),
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 0.5,
            angle: 0.0,
        };
        let nodes = vec![SceneNode::Circle {
            cx: 50.0, cy: 50.0, r: 5.0, style,
        }];
        let bytes = pack_circle_batch(&nodes);
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // fill_a at index 6: (255/255)*fill_opacity(0.5) = 0.5
        assert!(
            (floats[6] - 0.5).abs() < 1e-5,
            "fill alpha must use fill_opacity (0.5), got {}",
            floats[6]
        );
    }

    #[test]
    fn packed_circle_stroke_uses_raw_color_no_opacity_baked() {
        // stroke_opacity=0.75, opacity=0.5 → stroke_a should be raw: (255/255)*1.0
        // (shader applies stroke_opacity and opacity separately)
        let style = FillStroke {
            fill: Some(Color::rgb(255, 0, 0)),
            stroke: Some(Color::rgba(0, 0, 0, 255)),
            stroke_width: 2.0,
            opacity: 0.5,
            stroke_dash: None,
            stroke_opacity: 0.75,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let nodes = vec![SceneNode::Circle {
            cx: 50.0, cy: 50.0, r: 5.0, style,
        }];
        let bytes = pack_circle_batch(&nodes);
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // stroke_a at index 10: (255/255)*1.0 = 1.0 (raw, no opacity baked)
        assert!(
            (floats[10] - 1.0).abs() < 1e-5,
            "stroke alpha must be raw (1.0), got {} — shader handles opacity/stroke_opacity",
            floats[10]
        );
    }

    #[test]
    fn packed_rect_fill_uses_fill_opacity_not_opacity() {
        let style = FillStroke {
            fill: Some(Color::rgba(100, 200, 50, 255)),
            stroke: Some(Color::rgb(0, 0, 0)),
            stroke_width: 1.0,
            opacity: 0.8,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 0.6,
            angle: 0.0,
        };
        let nodes = vec![SceneNode::Rect {
            x: 10.0, y: 20.0, w: 30.0, h: 40.0, style, corner_radius: 0.0,
        }];
        let bytes = pack_rect_batch(&nodes);
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // fill_a at index 8 (after pos(2)+size(2)+corner_radius(1)+fill_rgb(3))
        // = (255/255)*fill_opacity(0.6) = 0.6
        assert!(
            (floats[8] - 0.6).abs() < 1e-5,
            "rect fill alpha must use fill_opacity (0.6), got {}",
            floats[8]
        );
    }

    #[test]
    fn packed_rect_stroke_uses_raw_color_no_opacity_baked() {
        let style = FillStroke {
            fill: Some(Color::rgb(100, 200, 50)),
            stroke: Some(Color::rgba(0, 0, 0, 200)),
            stroke_width: 2.0,
            opacity: 0.5,
            stroke_dash: None,
            stroke_opacity: 0.75,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        let nodes = vec![SceneNode::Rect {
            x: 10.0, y: 20.0, w: 30.0, h: 40.0, style, corner_radius: 0.0,
        }];
        let bytes = pack_rect_batch(&nodes);
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // stroke_a at index 12 (pos(2)+size(2)+corner(1)+fill(4)+stroke_rgb(3))
        // = (200/255)*1.0 (raw, no opacity baked)
        let expected = 200.0_f32 / 255.0;
        assert!(
            (floats[12] - expected).abs() < 1e-5,
            "rect stroke alpha must be raw ({expected}), got {} — shader handles opacity/stroke_opacity",
            floats[12]
        );
    }
}
