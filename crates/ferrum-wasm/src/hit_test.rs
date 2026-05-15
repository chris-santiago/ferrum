use ferrum_scene::{MarkBatch, MarkBatchKind, SceneNode};

pub struct HitResult {
    pub panel_id: usize,
    pub batch_idx: usize,
    pub node_idx: usize,
    pub data_idx: Option<usize>,
}

pub fn hit_test(
    panels: &[ferrum_scene::Panel],
    x: f64,
    y: f64,
    zoom: &crate::zoom_pan::ZoomPanState,
) -> Option<HitResult> {
    for (panel_pos, panel) in panels.iter().enumerate().rev() {
        // Map the click from visual (post-zoom) pixel space back to scene pixel space.
        let (px, py) = zoom.transforms
            .get(panel_pos)
            .map(|t| t.inverse_apply(x, y))
            .unwrap_or((x, y));
        if !rect_contains(&panel.plot_area, px, py) {
            continue;
        }
        for (bi, batch) in panel.marks.iter().enumerate().rev() {
            if let Some(ni) = hit_test_batch(batch, panel, px, py) {
                let data_idx = batch
                    .data_indices
                    .as_ref()
                    .and_then(|ids| ids.get(ni).copied());
                return Some(HitResult {
                    panel_id: panel.id,
                    batch_idx: bi,
                    node_idx: ni,
                    data_idx,
                });
            }
        }
    }
    None
}

pub fn hit_test_nearest(
    panels: &[ferrum_scene::Panel],
    x: f64,
    y: f64,
    zoom: &crate::zoom_pan::ZoomPanState,
) -> Option<HitResult> {
    let mut best: Option<(f64, HitResult)> = None;
    for (panel_pos, panel) in panels.iter().enumerate() {
        // Map the click from visual (post-zoom) pixel space back to scene pixel space.
        let (px, py) = zoom.transforms
            .get(panel_pos)
            .map(|t| t.inverse_apply(x, y))
            .unwrap_or((x, y));
        if !rect_contains(&panel.plot_area, px, py) {
            continue;
        }
        for (bi, batch) in panel.marks.iter().enumerate() {
            if let Some((ni, dist)) = nearest_in_batch(batch, px, py) {
                let is_closer = best.as_ref().is_none_or(|(d, _)| dist < *d);
                if is_closer {
                    let data_idx = batch
                        .data_indices
                        .as_ref()
                        .and_then(|ids| ids.get(ni).copied());
                    best = Some((
                        dist,
                        HitResult {
                            panel_id: panel.id,
                            batch_idx: bi,
                            node_idx: ni,
                            data_idx,
                        },
                    ));
                }
            }
        }
    }
    best.map(|(_, r)| r)
}

fn nearest_in_batch(batch: &MarkBatch, x: f64, y: f64) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, node) in batch.nodes.iter().enumerate() {
        let dist = match node {
            SceneNode::Circle { cx, cy, .. } => {
                ((x - cx).powi(2) + (y - cy).powi(2)).sqrt()
            }
            SceneNode::Rect {
                x: rx, y: ry, w, h, ..
            } => {
                let cx = rx + w / 2.0;
                let cy = ry + h / 2.0;
                ((x - cx).powi(2) + (y - cy).powi(2)).sqrt()
            }
            _ => continue,
        };
        if best.as_ref().is_none_or(|(_, d)| dist < *d) {
            best = Some((i, dist));
        }
    }
    best
}

fn hit_test_batch(batch: &MarkBatch, panel: &ferrum_scene::Panel, x: f64, y: f64) -> Option<usize> {
    match batch.kind {
        MarkBatchKind::Point => hit_test_circles(&batch.nodes, x, y),
        MarkBatchKind::Bar | MarkBatchKind::Rect => hit_test_rects(&batch.nodes, x, y),
        MarkBatchKind::Line | MarkBatchKind::Area => hit_test_lines(&batch.nodes, x, y),
        MarkBatchKind::Rule => hit_test_lines(&batch.nodes, x, y),
        // Geoshape polygons: use existing polygon point-in-polygon test.
        MarkBatchKind::Polygon => hit_test_lines(&batch.nodes, x, y),
        // Arc wedges (polar pie/donut): polar coordinate hit-test.
        MarkBatchKind::Arc => hit_test_polar_arcs(&batch.nodes, panel, x, y),
        _ => None,
    }
}

fn hit_test_circles(nodes: &[SceneNode], x: f64, y: f64) -> Option<usize> {
    for (i, node) in nodes.iter().enumerate().rev() {
        if let SceneNode::Circle { cx, cy, r, .. } = node {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r * r {
                return Some(i);
            }
        }
    }
    None
}

fn hit_test_rects(nodes: &[SceneNode], x: f64, y: f64) -> Option<usize> {
    for (i, node) in nodes.iter().enumerate().rev() {
        if let SceneNode::Rect {
            x: rx,
            y: ry,
            w,
            h,
            ..
        } = node
        {
            if x >= *rx && x <= rx + w && y >= *ry && y <= ry + h {
                return Some(i);
            }
        }
    }
    None
}

fn hit_test_lines(nodes: &[SceneNode], x: f64, y: f64) -> Option<usize> {
    const MIN_TOLERANCE: f64 = 3.0;
    for (i, node) in nodes.iter().enumerate().rev() {
        match node {
            SceneNode::Line {
                x1,
                y1,
                x2,
                y2,
                style,
            } => {
                let tol = style.width.max(MIN_TOLERANCE);
                if dist_to_segment(x, y, *x1, *y1, *x2, *y2) <= tol {
                    return Some(i);
                }
            }
            SceneNode::Polyline { points, style } => {
                let tol = style.width.max(MIN_TOLERANCE);
                for pair in points.windows(2) {
                    if dist_to_segment(x, y, pair[0].0, pair[0].1, pair[1].0, pair[1].1) <= tol {
                        return Some(i);
                    }
                }
            }
            SceneNode::Polygon { points, style }
                if style.fill.is_some() && point_in_polygon(x, y, points) =>
            {
                return Some(i);
            }
            _ => {}
        }
    }
    None
}

/// Hit-test polar arc wedge nodes by converting pixel coords to (θ, r) and checking
/// each wedge's angular range (extracted from the first MoveTo/ArcTo commands).
fn hit_test_polar_arcs(
    nodes: &[SceneNode],
    panel: &ferrum_scene::Panel,
    x: f64,
    y: f64,
) -> Option<usize> {
    use std::f64::consts::TAU;
    use ferrum_scene::CoordKind;

    let (inner_r, outer_r) = match &panel.coord {
        CoordKind::Polar { inner_radius, outer_radius, .. } => (*inner_radius, *outer_radius),
        _ => return None,
    };
    let cx = panel.plot_area.x + panel.plot_area.w / 2.0;
    let cy = panel.plot_area.y + panel.plot_area.h / 2.0;
    let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
    if r < inner_r || r > outer_r {
        return None;
    }
    let mut theta = (x - cx).atan2(-(y - cy)); // clockwise from top, [-π, π]
    if theta < 0.0 { theta += TAU; }

    for (i, node) in nodes.iter().enumerate().rev() {
        let SceneNode::Path { commands, .. } = node else { continue };
        // Extract start angle from first MoveTo, end angle from first ArcTo.
        let mut start_theta: Option<f64> = None;
        let mut end_theta: Option<f64> = None;
        for cmd in commands {
            match cmd {
                ferrum_scene::PathCmd::MoveTo { x: mx, y: my } if start_theta.is_none() => {
                    let mut t = (*mx - cx).atan2(-(*my - cy));
                    if t < 0.0 { t += TAU; }
                    start_theta = Some(t);
                }
                ferrum_scene::PathCmd::ArcTo { x: ax, y: ay, .. } if end_theta.is_none() => {
                    let mut t = (*ax - cx).atan2(-(*ay - cy));
                    if t < 0.0 { t += TAU; }
                    end_theta = Some(t);
                    break;
                }
                _ => {}
            }
        }
        if let (Some(t0), Some(t1)) = (start_theta, end_theta) {
            let inside = if t1 >= t0 {
                theta >= t0 && theta <= t1
            } else {
                // Wedge wraps around 0 (e.g. last slice covers 330°–360°+0°–30°)
                theta >= t0 || theta <= t1
            };
            if inside {
                return Some(i);
            }
        }
    }
    None
}

fn dist_to_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = x1 + t * dx;
    let proj_y = y1 + t * dy;
    ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
}

fn point_in_polygon(px: f64, py: f64, vertices: &[[f64; 2]]) -> bool {
    let n = vertices.len();
    if n == 0 { return false; }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (vertices[i][0], vertices[i][1]);
        let (xj, yj) = (vertices[j][0], vertices[j][1]);
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn rect_contains(r: &ferrum_scene::Rect, x: f64, y: f64) -> bool {
    x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod bug_hunt_tests {
    use super::*;

    // ── hit_test_circles: boundary / edge cases ────────────────────────────

    fn make_circle(cx: f64, cy: f64, r: f64) -> SceneNode {
        SceneNode::Circle {
            cx,
            cy,
            r,
            style: ferrum_scene::FillStroke {
                fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
            },
        }
    }

    #[test]
    fn bug_hunt_circle_hit_exactly_on_boundary() {
        // Point exactly at radius distance must count as a hit (<=, not <)
        let nodes = vec![make_circle(0.0, 0.0, 5.0)];
        // Exactly at r=5: dx=5, dy=0, dx*dx+dy*dy = 25 == r*r = 25
        assert!(hit_test_circles(&nodes, 5.0, 0.0).is_some());
    }

    #[test]
    fn bug_hunt_circle_zero_radius_hit_only_at_center() {
        // Zero-radius circle: only a hit at the exact center pixel
        let nodes = vec![make_circle(10.0, 10.0, 0.0)];
        assert!(hit_test_circles(&nodes, 10.0, 10.0).is_some());
        assert!(hit_test_circles(&nodes, 10.001, 10.0).is_none());
    }

    #[test]
    fn bug_hunt_rect_boundary_corners() {
        // Rect corners must be included in the hit region
        let nodes = vec![SceneNode::Rect {
            x: 10.0,
            y: 20.0,
            w: 50.0,
            h: 30.0,
            style: ferrum_scene::FillStroke {
                fill: None,
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
            },
            corner_radius: 0.0,
        }];
        // All four corners
        assert!(hit_test_rects(&nodes, 10.0, 20.0).is_some(), "top-left corner");
        assert!(hit_test_rects(&nodes, 60.0, 20.0).is_some(), "top-right corner");
        assert!(hit_test_rects(&nodes, 10.0, 50.0).is_some(), "bottom-left corner");
        assert!(hit_test_rects(&nodes, 60.0, 50.0).is_some(), "bottom-right corner");
    }

    #[test]
    fn bug_hunt_rect_just_outside_boundary_is_miss() {
        let nodes = vec![SceneNode::Rect {
            x: 10.0,
            y: 20.0,
            w: 50.0,
            h: 30.0,
            style: ferrum_scene::FillStroke {
                fill: None,
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
            },
            corner_radius: 0.0,
        }];
        assert!(hit_test_rects(&nodes, 9.999, 25.0).is_none(), "just left");
        assert!(hit_test_rects(&nodes, 60.001, 25.0).is_none(), "just right");
    }

    #[test]
    fn bug_hunt_segment_distance_degenerate_zero_length() {
        // A zero-length segment: distance should be point-to-point
        let d = dist_to_segment(5.0, 3.0, 5.0, 3.0, 5.0, 3.0);
        assert!(d < 1e-10, "zero-length segment: dist should be 0, got {d}");
    }

    #[test]
    fn bug_hunt_segment_distance_perpendicular_projection_off_end() {
        // Point projects beyond the segment end — dist should be to the endpoint
        let d = dist_to_segment(20.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        assert!((d - 10.0).abs() < 1e-6, "expected 10.0, got {d}");
    }

    #[test]
    fn bug_hunt_polygon_point_on_edge_is_ambiguous_but_no_panic() {
        // A point exactly on the polygon edge must not panic.
        let square = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        // This just checks it doesn't panic; edge behaviour is implementation-defined.
        let _ = point_in_polygon(5.0, 0.0, &square);
    }

    #[test]
    fn bug_hunt_polygon_empty_vertices_no_panic() {
        // Empty polygon must not panic
        let empty: Vec<[f64; 2]> = vec![];
        let result = point_in_polygon(5.0, 5.0, &empty);
        assert!(!result, "empty polygon must never contain a point");
    }

    #[test]
    fn bug_hunt_hit_test_circles_last_in_list_wins() {
        // Overlapping circles: the later one in the list (rendered on top) should win
        let nodes = vec![
            make_circle(50.0, 50.0, 10.0),
            make_circle(50.0, 50.0, 10.0),
        ];
        // hit_test_circles iterates in reverse, so the LAST node (idx 1) should win
        let result = hit_test_circles(&nodes, 50.0, 50.0);
        assert_eq!(result, Some(1), "last (topmost) overlapping circle must win");
    }

    #[test]
    fn bug_hunt_rect_contains_and_rect_miss_with_negative_coords() {
        // Rect at negative coordinates must still be hit-testable
        let nodes = vec![SceneNode::Rect {
            x: -50.0,
            y: -30.0,
            w: 20.0,
            h: 15.0,
            style: ferrum_scene::FillStroke {
                fill: None,
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
            },
            corner_radius: 0.0,
        }];
        assert!(hit_test_rects(&nodes, -40.0, -25.0).is_some());
        assert!(hit_test_rects(&nodes, 0.0, 0.0).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoom_pan::{Affine2, ZoomPanState};
    use ferrum_scene::{
        CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_style() -> FillStroke {
        FillStroke {
            fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
        }
    }

    fn circle_node(cx: f64, cy: f64, r: f64) -> SceneNode {
        SceneNode::Circle { cx, cy, r, style: default_style() }
    }

    /// Build a single-panel scene with one circle at (cx, cy, r).
    /// The plot_area is large enough to contain the circle.
    fn single_circle_panel(cx: f64, cy: f64, r: f64) -> Vec<Panel> {
        vec![Panel {
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
                nodes: vec![circle_node(cx, cy, r)],
                data_indices: Some(vec![0]),
                tooltips: None,
                hrefs: None,
                keys: None,
                blend: ferrum_scene::BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }]
    }

    fn identity_zoom() -> ZoomPanState {
        let config = ferrum_scene::InteractionConfig::default();
        ZoomPanState::new(1, &config)
    }

    fn zoom_with(sx: f64, sy: f64, tx: f64, ty: f64) -> ZoomPanState {
        let config = ferrum_scene::InteractionConfig::default();
        let mut z = ZoomPanState::new(1, &config);
        z.transforms[0] = Affine2 { sx, sy, tx, ty };
        z
    }

    // ── inverse-transform hit-test tests ─────────────────────────────────────

    #[test]
    fn hit_test_identity_zoom_finds_circle() {
        // Baseline: identity transform — click at circle center hits.
        let panels = single_circle_panel(100.0, 100.0, 10.0);
        let zoom = identity_zoom();
        assert!(hit_test(&panels, 100.0, 100.0, &zoom).is_some());
        assert!(hit_test(&panels, 200.0, 200.0, &zoom).is_none());
    }

    #[test]
    fn hit_test_zoom_2x_click_at_visual_position_hits() {
        // Circle at (100, 100). After 2× zoom (sx=2, sy=2, tx=0, ty=0) the
        // circle visually appears at (200, 200). Clicking at (200, 200) must hit.
        let panels = single_circle_panel(100.0, 100.0, 10.0);
        let zoom = zoom_with(2.0, 2.0, 0.0, 0.0);
        let result = hit_test(&panels, 200.0, 200.0, &zoom);
        assert!(result.is_some(), "click at zoomed visual position must hit");
    }

    #[test]
    fn hit_test_zoom_2x_click_at_original_position_misses() {
        // Same setup: after zoom the circle moved to (200, 200).
        // Clicking at (100, 100) — the OLD position — must miss.
        let panels = single_circle_panel(100.0, 100.0, 10.0);
        let zoom = zoom_with(2.0, 2.0, 0.0, 0.0);
        let result = hit_test(&panels, 100.0, 100.0, &zoom);
        // Inverse maps (100,100) → (50,50), which is not within radius 10 of (100,100).
        assert!(result.is_none(), "click at pre-zoom position must miss after zoom");
    }

    #[test]
    fn hit_test_pan_only_click_at_panned_position_hits() {
        // Circle at (100, 100). Pan right by 50 px (tx=50, sy/sx unchanged).
        // Visual position = (150, 100). Click at (150, 100) must hit.
        let panels = single_circle_panel(100.0, 100.0, 10.0);
        let zoom = zoom_with(1.0, 1.0, 50.0, 0.0);
        assert!(hit_test(&panels, 150.0, 100.0, &zoom).is_some());
        assert!(hit_test(&panels, 100.0, 100.0, &zoom).is_none());
    }

    #[test]
    fn hit_test_zoom_returns_correct_data_index() {
        // Verify that data_idx is threaded through correctly after zoom.
        let panels = single_circle_panel(100.0, 100.0, 10.0);
        let zoom = zoom_with(2.0, 2.0, 0.0, 0.0);
        let result = hit_test(&panels, 200.0, 200.0, &zoom).expect("must hit");
        assert_eq!(result.data_idx, Some(0));
    }

    #[test]
    fn hit_test_zoom_out_half_click_at_visual_position_hits() {
        // Zoom out 0.5× — circle at (100,100) visually appears at (50,50).
        // Clicking at (50,50) must hit.
        let panels = single_circle_panel(100.0, 100.0, 5.0);
        let zoom = zoom_with(0.5, 0.5, 0.0, 0.0);
        // Inverse: (50,50) → (100,100) → within r=5 ✓
        assert!(hit_test(&panels, 50.0, 50.0, &zoom).is_some());
        // Clicking at original (100,100) → inverse → (200,200) → miss
        assert!(hit_test(&panels, 100.0, 100.0, &zoom).is_none());
    }

    #[test]
    fn circle_hit_inside() {
        let nodes = vec![SceneNode::Circle {
            cx: 100.0,
            cy: 100.0,
            r: 10.0,
            style: ferrum_scene::FillStroke {
                fill: Some(ferrum_scene::Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                }),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
            },
        }];
        assert!(hit_test_circles(&nodes, 105.0, 100.0).is_some());
        assert!(hit_test_circles(&nodes, 200.0, 200.0).is_none());
    }

    #[test]
    fn rect_hit_inside() {
        let nodes = vec![SceneNode::Rect {
            x: 10.0,
            y: 10.0,
            w: 50.0,
            h: 30.0,
            style: ferrum_scene::FillStroke {
                fill: None,
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
            },
            corner_radius: 0.0,
        }];
        assert!(hit_test_rects(&nodes, 35.0, 25.0).is_some());
        assert!(hit_test_rects(&nodes, 5.0, 5.0).is_none());
    }

    #[test]
    fn segment_distance() {
        let d = dist_to_segment(5.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        assert!(d < 0.001);
        let d2 = dist_to_segment(5.0, 3.0, 0.0, 0.0, 10.0, 0.0);
        assert!((d2 - 3.0).abs() < 0.001);
    }

    #[test]
    fn winding_number_polygon() {
        let square = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        assert!(point_in_polygon(5.0, 5.0, &square));
        assert!(!point_in_polygon(15.0, 5.0, &square));
    }
}
