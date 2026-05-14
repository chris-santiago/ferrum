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
) -> Option<HitResult> {
    for panel in panels.iter().rev() {
        if !rect_contains(&panel.plot_area, x, y) {
            continue;
        }
        for (bi, batch) in panel.marks.iter().enumerate().rev() {
            if let Some(ni) = hit_test_batch(batch, panel, x, y) {
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
) -> Option<HitResult> {
    let mut best: Option<(f64, HitResult)> = None;
    for panel in panels.iter() {
        if !rect_contains(&panel.plot_area, x, y) {
            continue;
        }
        for (bi, batch) in panel.marks.iter().enumerate() {
            if let Some((ni, dist)) = nearest_in_batch(batch, x, y) {
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
    let mut inside = false;
    let n = vertices.len();
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
mod tests {
    use super::*;

    #[test]
    fn circle_hit_inside() {
        let nodes = vec![SceneNode::Circle {
            cx: 100.0,
            cy: 100.0,
            r: 10.0,
            style: ferrum_scene::MarkStyle {
                fill: Some(ferrum_scene::Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                }),
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
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
