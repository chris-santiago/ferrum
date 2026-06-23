use ferrum_scene::{MarkBatch, MarkBatchKind, Panel, SceneNode};

use crate::spatial_index::{is_indexed_kind, SpatialIndex};

pub struct HitResult {
    pub panel_id: usize,
    pub batch_idx: usize,
    pub node_idx: usize,
    pub data_idx: Option<usize>,
}

/// Resolve the `data_idx` for a spatial-index hit result.
///
/// The spatial index stores a `data_idx` in each `MarkEntry` (populated from
/// `PackedBatchMeta` for packed batches).  For non-packed batches the entry's
/// `data_idx` may be `None`; fall back to the scene-graph's
/// `batch.data_indices[node_idx]` in that case.
///
/// This rule is the same at all four call sites inside `hit_test_with_index`
/// and `hit_test_nearest_with_index`, so a single helper eliminates the
/// four-way copy-paste and ensures the precedence order cannot drift.
pub(crate) fn resolve_data_idx(
    panel: &Panel,
    batch_idx: usize,
    node_idx: usize,
    entry_data_idx: Option<usize>,
) -> Option<usize> {
    entry_data_idx.or_else(|| {
        panel
            .marks
            .get(batch_idx)
            .and_then(|b| b.data_indices.as_ref())
            .and_then(|ids| ids.get(node_idx).copied())
    })
}

pub fn hit_test(
    panels: &[ferrum_scene::Panel],
    x: f64,
    y: f64,
    zoom: &crate::zoom_pan::ZoomPanState,
) -> Option<HitResult> {
    hit_test_with_index(panels, x, y, zoom, None)
}

/// Exact hit-test with an optional spatial index for O(log n) circle/rect lookups.
///
/// When `spatial_index` is provided, indexed batch kinds (Point, Bar, Rect) use
/// the R-tree for candidate lookup.  All other batch kinds always use the linear
/// scan.  The R-tree result takes priority over the linear scan result.
pub fn hit_test_with_index(
    panels: &[ferrum_scene::Panel],
    x: f64,
    y: f64,
    zoom: &crate::zoom_pan::ZoomPanState,
    spatial_index: Option<&SpatialIndex>,
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

        // R-tree path for indexed batch kinds (Point, Bar, Rect).
        if let Some(idx) = spatial_index {
            if let Some(entry) = idx.hit_test(panel_pos, px, py, 0.0) {
                let data_idx = resolve_data_idx(panel, entry.batch_idx, entry.node_idx, entry.data_idx);
                return Some(HitResult {
                    panel_id: panel_pos,
                    batch_idx: entry.batch_idx,
                    node_idx: entry.node_idx,
                    data_idx,
                });
            }
        }

        // Linear scan for non-indexed batch kinds (Line, Area, Arc, Tick, Text, etc.)
        // and for the full linear path when no spatial index is available.
        for (bi, batch) in panel.marks.iter().enumerate().rev() {
            // When a spatial index is available, skip indexed kinds — already handled above.
            if spatial_index.is_some() && is_indexed_kind(batch.kind) {
                continue;
            }
            if let Some(ni) = hit_test_batch(batch, panel, px, py) {
                let data_idx = batch
                    .data_indices
                    .as_ref()
                    .and_then(|ids| ids.get(ni).copied());
                return Some(HitResult {
                    panel_id: panel_pos,
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
    hit_test_nearest_with_index(panels, x, y, zoom, None)
}

/// Nearest-mark hit-test with an optional spatial index for O(log n) circle/rect lookups.
///
/// Strategy:
/// 1. If a `SpatialIndex` is provided, query it for the nearest indexed mark (circle/rect)
///    within `SNAP_DISTANCE`.
/// 2. Perform a linear scan over *non-indexed* batch kinds (Line, Area, Arc, Tick, Text, etc.).
/// 3. Return the closer of the two results.
///
/// When no index is provided, behaves identically to the original linear-only path.
pub fn hit_test_nearest_with_index(
    panels: &[ferrum_scene::Panel],
    x: f64,
    y: f64,
    zoom: &crate::zoom_pan::ZoomPanState,
    spatial_index: Option<&SpatialIndex>,
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

        // R-tree nearest for indexed batch kinds.
        if let Some(idx) = spatial_index {
            if let Some((entry, dist)) = idx.nearest(panel_pos, px, py) {
                let data_idx = resolve_data_idx(panel, entry.batch_idx, entry.node_idx, entry.data_idx);
                let is_closer = best.as_ref().is_none_or(|(d, _)| dist < *d);
                if is_closer {
                    best = Some((dist, HitResult {
                        panel_id: panel_pos,
                        batch_idx: entry.batch_idx,
                        node_idx: entry.node_idx,
                        data_idx,
                    }));
                }
            }
        }

        // Linear scan for non-indexed batch kinds (and the full path when no index given).
        for (bi, batch) in panel.marks.iter().enumerate() {
            // When a spatial index is available, skip indexed kinds — already covered above.
            if spatial_index.is_some() && is_indexed_kind(batch.kind) {
                continue;
            }
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
                            panel_id: panel_pos,
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
        // Short line segments (rug ticks, boxplot caps, etc.) — emit Line nodes.
        MarkBatchKind::Tick => hit_test_lines(&batch.nodes, x, y),
        // Text and Label marks — point proximity using font_size as tolerance.
        // Label routes identically to Text: same nodes, same hit-test policy.
        MarkBatchKind::Text | MarkBatchKind::Label => hit_test_texts(&batch.nodes, x, y),
        // Confidence band / filled area between two lines — emit closed Path nodes.
        MarkBatchKind::Ribbon => hit_test_lines(&batch.nodes, x, y),
        // Diagonal line segments between two (x,y) / (x2,y2) pairs — emit Line nodes.
        MarkBatchKind::Segment => hit_test_lines(&batch.nodes, x, y),
        // Rectangular raster tiles — emit Image nodes with x/y/w/h bounds.
        MarkBatchKind::Image => hit_test_images(&batch.nodes, x, y),
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
            SceneNode::Polygon { rings, style }
                if style.fill.is_some()
                    && rings.first().is_some_and(|r| point_in_polygon(x, y, r)) =>
            {
                return Some(i);
            }
            // Closed filled Path (Ribbon, Area with path geometry): extract
            // MoveTo/LineTo vertices and run point-in-polygon. Curves are
            // approximated by their endpoints, which is sufficient for tooltip
            // hit-testing on smooth CI bands.
            SceneNode::Path { commands, style, closed }
                if *closed && style.fill.is_some() =>
            {
                let vertices: Vec<[f64; 2]> = commands
                    .iter()
                    .filter_map(|cmd| match cmd {
                        ferrum_scene::PathCmd::MoveTo { x: px, y: py } => Some([*px, *py]),
                        ferrum_scene::PathCmd::LineTo { x: px, y: py } => Some([*px, *py]),
                        _ => None,
                    })
                    .collect();
                if point_in_polygon(x, y, &vertices) {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Hit-test text label nodes by proximity. `SceneNode::Text` carries only the
/// anchor point (x, y) — no pre-computed bounding box is available in the scene
/// graph. We treat the font_size as a square half-width tolerance so that a
/// click anywhere within the approximate glyph cell counts as a hit.
fn hit_test_texts(nodes: &[SceneNode], x: f64, y: f64) -> Option<usize> {
    for (i, node) in nodes.iter().enumerate().rev() {
        if let SceneNode::Text { x: tx, y: ty, style, .. } = node {
            let half = style.font_size.max(4.0);
            if (x - tx).abs() <= half && (y - ty).abs() <= half {
                return Some(i);
            }
        }
    }
    None
}

/// Hit-test rectangular image nodes using the stored `x / y / w / h` bounds.
/// `SceneNode::Image` is distinct from `SceneNode::Rect` — `hit_test_rects`
/// cannot be reused here.
fn hit_test_images(nodes: &[SceneNode], x: f64, y: f64) -> Option<usize> {
    for (i, node) in nodes.iter().enumerate().rev() {
        if let SceneNode::Image { x: ix, y: iy, w, h, .. } = node {
            if x >= *ix && x <= ix + w && y >= *iy && y <= iy + h {
                return Some(i);
            }
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
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
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
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
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
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
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
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
            corner_radius: 0.0,
        }];
        assert!(hit_test_rects(&nodes, -40.0, -25.0).is_some());
        assert!(hit_test_rects(&nodes, 0.0, 0.0).is_none());
    }

    // ── hit_test_nearest: new coverage ─────────────────────────────────────────

    fn make_panel_two_circles(c1: (f64, f64), c2: (f64, f64)) -> Vec<ferrum_scene::Panel> {
        use ferrum_scene::{BlendMode, CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect};
        let style = FillStroke {
            fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        };
        vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![
                    SceneNode::Circle { cx: c1.0, cy: c1.1, r: 5.0, style: style.clone() },
                    SceneNode::Circle { cx: c2.0, cy: c2.1, r: 5.0, style: style.clone() },
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
        }]
    }

    fn identity_zoom_n(n: usize) -> crate::zoom_pan::ZoomPanState {
        let config = ferrum_scene::InteractionConfig::default();
        crate::zoom_pan::ZoomPanState::new(n, &config)
    }

    #[test]
    fn bug_hunt_hit_test_nearest_empty_panels_returns_none() {
        // No panels: nearest must return None, not panic
        let zoom = identity_zoom_n(0);
        let result = hit_test_nearest(&[], 100.0, 100.0, &zoom);
        assert!(result.is_none(), "nearest on empty panels must return None");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_returns_closest_circle() {
        // Two circles: (100, 100) and (300, 300). Click near (110, 100) — closest is idx 0.
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 300.0));
        let zoom = identity_zoom_n(1);
        let result = hit_test_nearest(&panels, 110.0, 100.0, &zoom);
        let r = result.expect("must find nearest mark");
        assert_eq!(r.data_idx, Some(0), "circle at (100,100) must be nearest to (110,100)");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_returns_second_circle_when_closer() {
        // Two circles: (100, 100) and (300, 300). Click near (295, 295) — closest is idx 1.
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 300.0));
        let zoom = identity_zoom_n(1);
        let result = hit_test_nearest(&panels, 295.0, 295.0, &zoom);
        let r = result.expect("must find nearest mark");
        assert_eq!(r.data_idx, Some(1), "circle at (300,300) must be nearest to (295,295)");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_click_outside_plot_area_returns_none() {
        // plot_area is (0..500, 0..500). Click at (-50, 250) is outside.
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 300.0));
        let zoom = identity_zoom_n(1);
        let result = hit_test_nearest(&panels, -50.0, 250.0, &zoom);
        assert!(result.is_none(), "click outside plot_area must miss even for nearest");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_equidistant_returns_first_found() {
        // Two circles exactly equidistant from click at (200, 100):
        // circle A at (100, 100) → dist = 100, circle B at (300, 100) → dist = 100.
        // Whichever batch is first in the iteration wins (no panic, deterministic).
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 100.0));
        let zoom = identity_zoom_n(1);
        let result = hit_test_nearest(&panels, 200.0, 100.0, &zoom);
        assert!(result.is_some(), "equidistant case must not return None");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_panel_with_no_marks() {
        // Panel with no mark batches: nearest must return None, not panic.
        use ferrum_scene::{CoordKind, Panel, Rect};
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![],  // no marks
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];
        let zoom = identity_zoom_n(1);
        let result = hit_test_nearest(&panels, 250.0, 250.0, &zoom);
        assert!(result.is_none(), "panel with no marks must yield None for nearest");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_data_idx_threaded_correctly() {
        // Verify data_indices are read correctly for nearest result
        let panels = make_panel_two_circles((200.0, 200.0), (400.0, 200.0));
        let zoom = identity_zoom_n(1);
        // Click just next to first circle
        let result = hit_test_nearest(&panels, 200.0, 200.0, &zoom).expect("must hit");
        assert_eq!(result.data_idx, Some(0));
    }

    // ── hit_test with zoom transform: coordinate-space correctness ──────────

    fn zoom_with_n(n: usize, sx: f64, sy: f64, tx: f64, ty: f64) -> crate::zoom_pan::ZoomPanState {
        let config = ferrum_scene::InteractionConfig::default();
        let mut z = crate::zoom_pan::ZoomPanState::new(n, &config);
        if n > 0 {
            z.transforms[0] = crate::zoom_pan::Affine2 { sx, sy, tx, ty };
        }
        z
    }

    #[test]
    fn bug_hunt_hit_test_with_zoom_2x_and_translation() {
        // Circle at scene-space (100, 100). Zoom 2x with tx=50, ty=30.
        // Visual position = 2*100+50 = 250, 2*100+30 = 230.
        // Click at (250, 230) must hit; click at (100, 100) must miss.
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 300.0));
        let zoom = zoom_with_n(1, 2.0, 2.0, 50.0, 30.0);
        let hit_at_visual = hit_test(&panels, 250.0, 230.0, &zoom);
        assert!(hit_at_visual.is_some(), "click at zoomed+translated visual pos must hit");
        assert_eq!(hit_at_visual.as_ref().map(|h| h.data_idx), Some(Some(0)));

        let miss_at_original = hit_test(&panels, 100.0, 100.0, &zoom);
        assert!(miss_at_original.is_none(), "click at pre-zoom pos must miss");
    }

    #[test]
    fn bug_hunt_hit_test_nearest_with_zoom_returns_closest_in_scene_space() {
        // Two circles at (100, 100) and (300, 300). Zoom 2x, no translation.
        // Click at canvas (200, 200) maps to scene (100, 100) — must select first circle.
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 300.0));
        let zoom = zoom_with_n(1, 2.0, 2.0, 0.0, 0.0);
        let result = hit_test_nearest(&panels, 200.0, 200.0, &zoom);
        let r = result.expect("must find nearest mark under zoom");
        assert_eq!(r.data_idx, Some(0), "canvas (200,200) / scene (100,100) must be nearest to circle 0");
    }

    #[test]
    fn bug_hunt_hit_test_zoom_out_half_nearest_picks_correct_circle() {
        // Zoom out 0.5x: circle at scene (100, 100) appears at canvas (50, 50).
        // Circle at scene (300, 300) appears at canvas (150, 150).
        // Click at canvas (145, 145) maps to scene (290, 290) — nearest to (300, 300).
        let panels = make_panel_two_circles((100.0, 100.0), (300.0, 300.0));
        let zoom = zoom_with_n(1, 0.5, 0.5, 0.0, 0.0);
        let result = hit_test_nearest(&panels, 145.0, 145.0, &zoom);
        let r = result.expect("must find nearest mark");
        assert_eq!(r.data_idx, Some(1), "canvas (145,145) / scene (290,290) nearest to circle 1");
    }

    #[test]
    fn bug_hunt_hit_test_pan_only_nearest() {
        // Pan tx=100, ty=0. Circle at scene (50, 50) appears at canvas (150, 50).
        // Click at canvas (150, 50) maps to scene (50, 50) — must hit circle 0.
        let panels = make_panel_two_circles((50.0, 50.0), (400.0, 400.0));
        let zoom = zoom_with_n(1, 1.0, 1.0, 100.0, 0.0);
        let result = hit_test_nearest(&panels, 150.0, 50.0, &zoom);
        let r = result.expect("must find nearest mark under pan");
        assert_eq!(r.data_idx, Some(0));
    }

    #[test]
    fn bug_hunt_hit_test_multi_panel_zoom_independence() {
        // Two panels, only panel 0 zoomed. Click should use correct zoom per panel.
        use ferrum_scene::{BlendMode, CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect};
        let style = FillStroke {
            fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None, stroke_width: 0.0, opacity: 1.0,
            stroke_dash: None, stroke_opacity: 1.0, fill_opacity: 1.0, angle: 0.0,
        };
        let panels = vec![
            Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 500.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 200.0, h: 500.0 },
                coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
                grid: vec![], axes: vec![], annotations: vec![], strip_title: vec![],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![SceneNode::Circle { cx: 100.0, cy: 100.0, r: 5.0, style: style.clone() }],
                    data_indices: Some(vec![0]), tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal, descriptions: None,
                    stroke_cap: None, stroke_join: None, packed_instances: None,
                }],
            },
            Panel {
                id: 1,
                plot_area: Rect { x: 250.0, y: 0.0, w: 200.0, h: 500.0 },
                clip: Rect { x: 250.0, y: 0.0, w: 200.0, h: 500.0 },
                coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
                grid: vec![], axes: vec![], annotations: vec![], strip_title: vec![],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![SceneNode::Circle { cx: 350.0, cy: 100.0, r: 5.0, style: style.clone() }],
                    data_indices: Some(vec![10]), tooltips: None, hrefs: None, keys: None,
                    blend: BlendMode::Normal, descriptions: None,
                    stroke_cap: None, stroke_join: None, packed_instances: None,
                }],
            },
        ];
        let config = ferrum_scene::InteractionConfig::default();
        let mut zoom = crate::zoom_pan::ZoomPanState::new(2, &config);
        // Zoom panel 0 to 2x; panel 1 stays identity
        zoom.transforms[0] = crate::zoom_pan::Affine2 { sx: 2.0, sy: 2.0, tx: 0.0, ty: 0.0 };
        // Click at panel 1's circle at identity position (350, 100)
        let result = hit_test(&panels, 350.0, 100.0, &zoom);
        assert!(result.is_some(), "panel 1 click at identity position must hit");
        assert_eq!(result.as_ref().map(|h| h.panel_id), Some(1));
    }

    // ── newly wired mark kinds ──────────────────────────────────────────────

    fn stroke_style() -> ferrum_scene::StrokeStyle {
        ferrum_scene::StrokeStyle {
            color: ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 },
            width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
        }
    }

    fn fill_stroke_with_fill() -> ferrum_scene::FillStroke {
        ferrum_scene::FillStroke {
            fill: Some(ferrum_scene::Color { r: 100, g: 100, b: 100, a: 200 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    fn text_style(font_size: f64) -> ferrum_scene::TextStyle {
        ferrum_scene::TextStyle {
            font_size,
            font_weight: ferrum_scene::FontWeight::Normal,
            anchor: ferrum_scene::TextAnchor::Middle,
            baseline: ferrum_scene::TextBaseline::Alphabetic,
            angle: 0.0,
            color: ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 },
            opacity: 1.0,
            font_family: "sans-serif".into(),
        }
    }

    // Tick — emits Line nodes; hit_test_batch must route to hit_test_lines.
    #[test]
    fn bug_hunt_tick_hit_test_hits_line() {
        let nodes = vec![SceneNode::Line {
            x1: 50.0, y1: 90.0, x2: 50.0, y2: 100.0,
            style: stroke_style(),
        }];
        // Click directly on the tick line — should hit.
        assert!(hit_test_lines(&nodes, 50.0, 95.0).is_some(), "tick line must be hit");
        // Click far away — should miss.
        assert!(hit_test_lines(&nodes, 200.0, 200.0).is_none(), "miss expected");
    }

    #[test]
    fn bug_hunt_tick_batch_routes_correctly() {
        use ferrum_scene::{BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect};
        let panel = Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        };
        let batch = MarkBatch {
            kind: MarkBatchKind::Tick,
            nodes: vec![SceneNode::Line {
                x1: 50.0, y1: 90.0, x2: 50.0, y2: 100.0,
                style: stroke_style(),
            }],
            data_indices: Some(vec![0]),
            tooltips: None, hrefs: None, keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None, stroke_join: None, packed_instances: None,
        };
        assert!(hit_test_batch(&batch, &panel, 50.0, 95.0).is_some(),
            "Tick batch must route to hit_test_lines and return a hit");
    }

    // Text — emits Text nodes; proximity check using font_size as tolerance.
    #[test]
    fn bug_hunt_text_hit_inside_font_box() {
        let nodes = vec![SceneNode::Text {
            x: 100.0, y: 100.0,
            content: "hello".into(),
            style: text_style(12.0),
        }];
        // Click within font_size radius of anchor — should hit.
        assert!(hit_test_texts(&nodes, 105.0, 105.0).is_some(), "text hit inside box expected");
        // Click far outside — should miss.
        assert!(hit_test_texts(&nodes, 200.0, 200.0).is_none(), "miss expected");
    }

    #[test]
    fn bug_hunt_text_hit_on_anchor_exact() {
        let nodes = vec![SceneNode::Text {
            x: 50.0, y: 50.0,
            content: "label".into(),
            style: text_style(16.0),
        }];
        assert!(hit_test_texts(&nodes, 50.0, 50.0).is_some(), "click exactly on anchor must hit");
    }

    #[test]
    fn bug_hunt_text_batch_routes_correctly() {
        use ferrum_scene::{BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect};
        let panel = Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        };
        let batch = MarkBatch {
            kind: MarkBatchKind::Text,
            nodes: vec![SceneNode::Text {
                x: 100.0, y: 100.0,
                content: "label".into(),
                style: text_style(14.0),
            }],
            data_indices: Some(vec![0]),
            tooltips: None, hrefs: None, keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None, stroke_join: None, packed_instances: None,
        };
        assert!(hit_test_batch(&batch, &panel, 100.0, 100.0).is_some(),
            "Text batch must route to hit_test_texts and return a hit on anchor");
    }

    // Ribbon — emits closed filled Path nodes; point-in-polygon via hit_test_lines.
    #[test]
    fn bug_hunt_ribbon_hit_inside_closed_path() {
        // Build a simple rectangular closed path: (0,0)→(100,0)→(100,50)→(0,50)→Close.
        let commands = vec![
            ferrum_scene::PathCmd::MoveTo { x: 0.0,   y: 0.0  },
            ferrum_scene::PathCmd::LineTo { x: 100.0, y: 0.0  },
            ferrum_scene::PathCmd::LineTo { x: 100.0, y: 50.0 },
            ferrum_scene::PathCmd::LineTo { x: 0.0,   y: 50.0 },
            ferrum_scene::PathCmd::Close,
        ];
        let nodes = vec![SceneNode::Path {
            commands,
            style: fill_stroke_with_fill(),
            closed: true,
        }];
        // Click inside the band.
        assert!(hit_test_lines(&nodes, 50.0, 25.0).is_some(), "click inside ribbon band must hit");
        // Click outside.
        assert!(hit_test_lines(&nodes, 200.0, 200.0).is_none(), "click outside ribbon must miss");
    }

    #[test]
    fn bug_hunt_ribbon_unfilled_path_not_hit_by_interior_click() {
        // An unfilled (stroke-only) closed path should NOT hit on interior click —
        // only the outline would be tested, and a click far from the border misses.
        let commands = vec![
            ferrum_scene::PathCmd::MoveTo { x: 0.0,   y: 0.0  },
            ferrum_scene::PathCmd::LineTo { x: 100.0, y: 0.0  },
            ferrum_scene::PathCmd::LineTo { x: 100.0, y: 50.0 },
            ferrum_scene::PathCmd::LineTo { x: 0.0,   y: 50.0 },
            ferrum_scene::PathCmd::Close,
        ];
        let nodes = vec![SceneNode::Path {
            commands,
            style: ferrum_scene::FillStroke {
                fill: None,  // stroke-only
                stroke: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
                stroke_width: 1.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
            closed: true,
        }];
        // Interior click should miss an unfilled path (no point-in-polygon).
        assert!(hit_test_lines(&nodes, 50.0, 25.0).is_none(),
            "interior click on unfilled path must miss");
    }

    #[test]
    fn bug_hunt_ribbon_batch_routes_correctly() {
        use ferrum_scene::{BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect};
        let panel = Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        };
        let commands = vec![
            ferrum_scene::PathCmd::MoveTo { x: 10.0, y: 10.0 },
            ferrum_scene::PathCmd::LineTo { x: 200.0, y: 10.0 },
            ferrum_scene::PathCmd::LineTo { x: 200.0, y: 100.0 },
            ferrum_scene::PathCmd::LineTo { x: 10.0, y: 100.0 },
            ferrum_scene::PathCmd::Close,
        ];
        let batch = MarkBatch {
            kind: MarkBatchKind::Ribbon,
            nodes: vec![SceneNode::Path {
                commands,
                style: fill_stroke_with_fill(),
                closed: true,
            }],
            data_indices: Some(vec![0]),
            tooltips: None, hrefs: None, keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None, stroke_join: None, packed_instances: None,
        };
        assert!(hit_test_batch(&batch, &panel, 100.0, 50.0).is_some(),
            "Ribbon batch must hit inside filled closed path");
    }

    // Segment — emits Line nodes (same as Rule/Tick); routes to hit_test_lines.
    #[test]
    fn bug_hunt_segment_hit_on_diagonal_line() {
        let nodes = vec![SceneNode::Line {
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            style: stroke_style(),
        }];
        // Click very close to the midpoint of the diagonal.
        assert!(hit_test_lines(&nodes, 50.0, 50.0).is_some(), "midpoint click on diagonal must hit");
        // Click well off the line.
        assert!(hit_test_lines(&nodes, 50.0, 150.0).is_none(), "off-diagonal miss expected");
    }

    #[test]
    fn bug_hunt_segment_batch_routes_correctly() {
        use ferrum_scene::{BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect};
        let panel = Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        };
        let batch = MarkBatch {
            kind: MarkBatchKind::Segment,
            nodes: vec![SceneNode::Line {
                x1: 10.0, y1: 10.0, x2: 200.0, y2: 200.0,
                style: stroke_style(),
            }],
            data_indices: Some(vec![0]),
            tooltips: None, hrefs: None, keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None, stroke_join: None, packed_instances: None,
        };
        assert!(hit_test_batch(&batch, &panel, 105.0, 105.0).is_some(),
            "Segment batch must route to hit_test_lines and return a hit near midpoint");
    }

    // Image — emits Image nodes with x/y/w/h bbox; routes to hit_test_images.
    #[test]
    fn bug_hunt_image_hit_inside_bbox() {
        let nodes = vec![SceneNode::Image {
            x: 10.0, y: 20.0, w: 80.0, h: 60.0,
            data: ferrum_scene::ImageData::Url { url: "data:image/png;base64,abc".into() },
        }];
        // Click inside the image rect.
        assert!(hit_test_images(&nodes, 50.0, 50.0).is_some(), "click inside image must hit");
        // Click outside.
        assert!(hit_test_images(&nodes, 200.0, 200.0).is_none(), "click outside image must miss");
    }

    #[test]
    fn bug_hunt_image_hit_on_boundary() {
        let nodes = vec![SceneNode::Image {
            x: 0.0, y: 0.0, w: 100.0, h: 100.0,
            data: ferrum_scene::ImageData::Url { url: "data:image/png;base64,abc".into() },
        }];
        // Corners should be included (<=, not <).
        assert!(hit_test_images(&nodes, 0.0, 0.0).is_some(), "top-left corner must hit");
        assert!(hit_test_images(&nodes, 100.0, 100.0).is_some(), "bottom-right corner must hit");
        assert!(hit_test_images(&nodes, 100.001, 50.0).is_none(), "just outside right edge must miss");
    }

    #[test]
    fn bug_hunt_image_batch_routes_correctly() {
        use ferrum_scene::{BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect};
        let panel = Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![],
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        };
        let batch = MarkBatch {
            kind: MarkBatchKind::Image,
            nodes: vec![SceneNode::Image {
                x: 50.0, y: 50.0, w: 200.0, h: 150.0,
                data: ferrum_scene::ImageData::Url {
                    url: "data:image/png;base64,abc".into()
                },
            }],
            data_indices: Some(vec![0]),
            tooltips: None, hrefs: None, keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None, stroke_join: None, packed_instances: None,
        };
        assert!(hit_test_batch(&batch, &panel, 150.0, 125.0).is_some(),
            "Image batch must route to hit_test_images and return a hit inside bbox");
        assert!(hit_test_batch(&batch, &panel, 10.0, 10.0).is_none(),
            "Image batch must miss outside bbox");
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
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
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
                packed_instances: None,
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

    // ── B6: panel_id must be array position, not panel.id ─────────────────

    /// Build a two-panel scene where panel.id diverges from array position.
    /// Panel 0 (array pos 0) has id=0, panel 1 (array pos 1) has id=5.
    /// A circle is placed only in panel 1 (id=5).
    fn two_panels_with_id_gap() -> Vec<Panel> {
        vec![
            Panel {
                id: 0,
                plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 500.0 },
                clip: Rect { x: 0.0, y: 0.0, w: 200.0, h: 500.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
                },
                grid: vec![],
                marks: vec![],  // no marks in panel 0
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
            },
            Panel {
                id: 5,  // logical id diverges from array position (1)
                plot_area: Rect { x: 250.0, y: 0.0, w: 200.0, h: 500.0 },
                clip: Rect { x: 250.0, y: 0.0, w: 200.0, h: 500.0 },
                coord: CoordKind::Cartesian {
                    x_domain: None, y_domain: None, expand: true, clip: true,
                },
                grid: vec![],
                marks: vec![MarkBatch {
                    kind: MarkBatchKind::Point,
                    nodes: vec![circle_node(350.0, 250.0, 10.0)],
                    data_indices: Some(vec![42]),
                    tooltips: None,
                    hrefs: None,
                    keys: None,
                    blend: ferrum_scene::BlendMode::Normal,
                    descriptions: None,
                    stroke_cap: None,
                    stroke_join: None,
                    packed_instances: None,
                }],
                axes: vec![],
                annotations: vec![],
                strip_title: vec![],
            },
        ]
    }

    #[test]
    fn hit_test_returns_array_position_not_panel_id() {
        // B6: When panel.id diverges from array index (e.g. skipped panels),
        // hit_test must return the array position (1) not the logical id (5),
        // because consumers (tooltip_for_hit, zoom_pan) index panels by position.
        let panels = two_panels_with_id_gap();
        let config = ferrum_scene::InteractionConfig::default();
        let zoom = ZoomPanState::new(2, &config);
        // Click at circle center in panel 1 (array pos 1, id 5)
        let result = hit_test(&panels, 350.0, 250.0, &zoom)
            .expect("must hit circle in panel 1");
        assert_eq!(
            result.panel_id, 1,
            "panel_id must be array position (1), not panel.id (5)"
        );
        assert_eq!(result.data_idx, Some(42));
    }

    #[test]
    fn hit_test_nearest_returns_array_position_not_panel_id() {
        // Same as above but for hit_test_nearest.
        let panels = two_panels_with_id_gap();
        let config = ferrum_scene::InteractionConfig::default();
        let zoom = ZoomPanState::new(2, &config);
        // Click at circle center in panel 1 (array pos 1, id 5)
        let result = hit_test_nearest(&panels, 350.0, 250.0, &zoom)
            .expect("must find nearest in panel 1");
        assert_eq!(
            result.panel_id, 1,
            "panel_id must be array position (1), not panel.id (5)"
        );
        assert_eq!(result.data_idx, Some(42));
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
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
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
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
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

    // ── Task 2: Label kind hit-test routing ──────────────────────────────────

    fn text_style_small() -> ferrum_scene::TextStyle {
        ferrum_scene::TextStyle {
            font_size: 12.0,
            font_weight: ferrum_scene::FontWeight::Normal,
            anchor: ferrum_scene::TextAnchor::Middle,
            baseline: ferrum_scene::TextBaseline::Alphabetic,
            angle: 0.0,
            color: ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 },
            opacity: 1.0,
            font_family: "sans-serif".into(),
        }
    }

    /// A Label batch routes to hit_test_texts (same as Text).
    #[test]
    fn label_batch_routes_to_hit_test_texts() {
        let panel = Panel {
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
            marks: vec![],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        };
        let batch = MarkBatch {
            kind: MarkBatchKind::Label,
            nodes: vec![SceneNode::Text {
                x: 100.0,
                y: 100.0,
                content: "label-node".into(),
                style: text_style_small(),
            }],
            data_indices: Some(vec![0]),
            tooltips: None,
            hrefs: None,
            keys: None,
            blend: ferrum_scene::BlendMode::Normal,
            descriptions: None,
            stroke_cap: None,
            stroke_join: None,
            packed_instances: None,
        };
        // Click on the label anchor — must hit (routes to hit_test_texts).
        assert!(
            hit_test_batch(&batch, &panel, 100.0, 100.0).is_some(),
            "Label batch must hit on anchor point via hit_test_texts"
        );
        // Click far away — must miss.
        assert!(
            hit_test_batch(&batch, &panel, 400.0, 400.0).is_none(),
            "Label batch must miss far from anchor"
        );
    }

    // ── WASM-10: resolve_data_idx ─────────────────────────────────────────────

    fn make_panel_with_data_indices(data_indices: Option<Vec<usize>>) -> Panel {
        Panel {
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
                nodes: vec![circle_node(50.0, 50.0, 5.0)],
                data_indices,
                tooltips: None,
                hrefs: None,
                keys: None,
                blend: ferrum_scene::BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
                packed_instances: None,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }
    }

    /// When `entry_data_idx` is `Some`, it is returned directly (packed-batch
    /// precedence — the spatial index stores it from `PackedBatchMeta`).
    #[test]
    fn resolve_data_idx_returns_entry_data_idx_when_some() {
        let panel = make_panel_with_data_indices(Some(vec![42]));
        // entry_data_idx = Some(99) takes priority over batch.data_indices[0] = 42.
        let result = resolve_data_idx(&panel, 0, 0, Some(99));
        assert_eq!(result, Some(99), "entry_data_idx must take priority over batch.data_indices");
    }

    /// When `entry_data_idx` is `None`, fall back to `batch.data_indices[node_idx]`.
    #[test]
    fn resolve_data_idx_falls_back_to_batch_data_indices() {
        let panel = make_panel_with_data_indices(Some(vec![42]));
        let result = resolve_data_idx(&panel, 0, 0, None);
        assert_eq!(result, Some(42), "must fall back to batch.data_indices[node_idx]");
    }

    /// When both `entry_data_idx` and `batch.data_indices` are `None`, returns `None`.
    #[test]
    fn resolve_data_idx_returns_none_when_both_absent() {
        let panel = make_panel_with_data_indices(None);
        let result = resolve_data_idx(&panel, 0, 0, None);
        assert_eq!(result, None, "must return None when both sources are absent");
    }

    /// When `batch_idx` is out of range, returns `None` gracefully.
    #[test]
    fn resolve_data_idx_out_of_range_batch_idx_returns_none() {
        let panel = make_panel_with_data_indices(Some(vec![7]));
        // batch_idx = 99 does not exist.
        let result = resolve_data_idx(&panel, 99, 0, None);
        assert_eq!(result, None, "out-of-range batch_idx must return None");
    }

    /// When `node_idx` is out of range for `batch.data_indices`, returns `None`.
    #[test]
    fn resolve_data_idx_out_of_range_node_idx_returns_none() {
        let panel = make_panel_with_data_indices(Some(vec![7]));
        // node_idx = 99 is beyond the single-element data_indices.
        let result = resolve_data_idx(&panel, 0, 99, None);
        assert_eq!(result, None, "out-of-range node_idx must return None");
    }
}
