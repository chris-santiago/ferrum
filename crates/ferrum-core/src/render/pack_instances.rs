//! Packed-instance extraction for high-cardinality mark batches.
//!
//! When a `MarkBatch` contains >1000 homogeneous circle or rect nodes,
//! `extract_packed_bytes` packs their GPU instance data as raw bytes.
//! The bytes travel to the WASM renderer as a separate `Uint8Array`,
//! completely bypassing JSON serialization and parsing.
//!
//! The byte layout matches `CircleInstance` (16 × f32 = 64 bytes) and
//! `RectInstance` (18 × f32 = 72 bytes) defined in `ferrum-wasm/src/scene_load.rs`.

use ferrum_scene::{Color, FillStroke, MarkBatchKind, SceneGraph, SceneNode};

/// Minimum node count for a batch to qualify for packing.
const PACK_THRESHOLD: usize = 1000;

/// Extract packed instance bytes from large homogeneous mark batches.
///
/// Returns the concatenated raw bytes for all qualifying batches. The nodes
/// in each qualifying batch are cleared — the packed binary data is the sole
/// representation. The WASM renderer receives these bytes as a separate
/// `Uint8Array`, bypassing JSON entirely.
///
/// Byte layout: `[batch_header][instance_data]` repeated per batch.
/// Batch header: `[panel_idx: u32][batch_idx: u32][kind: u32][count: u32]` (16 bytes).
/// Instance data: `count × sizeof(CircleInstance|RectInstance)` bytes.
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

            // Header: panel_idx, batch_idx, kind (0=circle, 1=rect), count
            packed.extend_from_slice(&(pi as u32).to_le_bytes());
            packed.extend_from_slice(&(bi as u32).to_le_bytes());
            packed.extend_from_slice(&kind_tag.to_le_bytes());
            packed.extend_from_slice(&(n as u32).to_le_bytes());
            packed.extend_from_slice(&instance_bytes);

            batch.nodes.clear();
        }
    }
    packed
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
            push_color(&mut buf, style.fill.as_ref(), style.opacity);
            push_color(&mut buf, style.stroke.as_ref(), style.opacity);
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
            push_color(&mut buf, style.fill.as_ref(), style.opacity);
            push_color(&mut buf, style.stroke.as_ref(), style.opacity);
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
///   rgb → [0..1], alpha → (color.a / 255) * opacity
#[inline]
fn push_color(buf: &mut Vec<u8>, color: Option<&Color>, opacity: f64) {
    match color {
        Some(c) => {
            push_f32(buf, c.r as f32 / 255.0);
            push_f32(buf, c.g as f32 / 255.0);
            push_f32(buf, c.b as f32 / 255.0);
            push_f32(buf, (c.a as f32 / 255.0) * opacity as f32);
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
        // fill_color: (255/255, 0, 0, (255/255)*0.5) = (1.0, 0.0, 0.0, 0.5)
        assert!((floats[3] - 1.0).abs() < 1e-5, "fill_r");
        assert!((floats[4] - 0.0).abs() < 1e-5, "fill_g");
        assert!((floats[5] - 0.0).abs() < 1e-5, "fill_b");
        assert!((floats[6] - 0.5).abs() < 1e-5, "fill_a = (255/255)*0.5");
        // stroke_color: (0, 0, 0, (128/255)*0.5)
        assert!((floats[7] - 0.0).abs() < 1e-5, "stroke_r");
        assert!((floats[8] - 0.0).abs() < 1e-5, "stroke_g");
        assert!((floats[9] - 0.0).abs() < 1e-5, "stroke_b");
        let expected_stroke_a = (128.0_f32 / 255.0) * 0.5;
        assert!(
            (floats[10] - expected_stroke_a).abs() < 1e-5,
            "stroke_a = (128/255)*0.5"
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

        // Verify header
        let panel_idx = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let batch_idx = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let kind = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(panel_idx, 0);
        assert_eq!(batch_idx, 0);
        assert_eq!(kind, 0); // circle
        assert_eq!(count, n as u32);

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
                    data_indices: None, tooltips: None, hrefs: None,
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
    fn stroke_dash_index_maps_correctly() {
        assert!((stroke_dash_index(&None) - 0.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![])) - 0.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![6.0, 3.0])) - 1.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![2.0, 3.0])) - 2.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![6.0, 3.0, 2.0, 3.0])) - 3.0).abs() < 1e-6);
        assert!((stroke_dash_index(&Some(vec![5.0, 5.0])) - 0.0).abs() < 1e-6);
    }
}
