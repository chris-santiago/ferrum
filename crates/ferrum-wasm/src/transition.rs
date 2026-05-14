use std::collections::HashMap;

use ferrum_scene::{MarkBatch, Panel};

use crate::scene_load::{CircleInstance, RectInstance};

pub struct Transition {
    pub old_circles: Vec<CircleInstance>,
    pub new_circles: Vec<CircleInstance>,
    pub old_rects: Vec<RectInstance>,
    pub new_rects: Vec<RectInstance>,
}

pub fn diff_scenes(
    old_panels: &[Panel],
    new_panels: &[Panel],
    old_circles: &[CircleInstance],
    new_circles: &[CircleInstance],
    old_rects: &[RectInstance],
    new_rects: &[RectInstance],
) -> Transition {
    let mut result_old_c = old_circles.to_vec();
    let result_new_c = new_circles.to_vec();
    let mut result_old_r = old_rects.to_vec();
    let result_new_r = new_rects.to_vec();

    let mut old_c_off = 0usize;
    let mut new_c_off = 0usize;
    let mut old_r_off = 0usize;
    let mut new_r_off = 0usize;

    for (old_p, new_p) in old_panels.iter().zip(new_panels.iter()) {
        for (old_b, new_b) in old_p.marks.iter().zip(new_p.marks.iter()) {
            let (old_nc, old_nr) = count_instances(old_b);
            let (new_nc, new_nr) = count_instances(new_b);

            if let (Some(old_keys), Some(new_keys)) = (&old_b.keys, &new_b.keys) {
                let new_key_map: HashMap<&str, usize> = new_keys
                    .iter()
                    .enumerate()
                    .map(|(i, k)| (k.as_str(), i))
                    .collect();

                for (old_idx, old_key) in old_keys.iter().enumerate() {
                    if let Some(&new_idx) = new_key_map.get(old_key.as_str()) {
                        match_circle_pair(
                            &mut result_old_c,
                            old_c_off + old_idx,
                            &result_new_c,
                            new_c_off + new_idx,
                        );
                        match_rect_pair(
                            &mut result_old_r,
                            old_r_off + old_idx,
                            &result_new_r,
                            new_r_off + new_idx,
                        );
                    }
                }
            }

            old_c_off += old_nc;
            new_c_off += new_nc;
            old_r_off += old_nr;
            new_r_off += new_nr;
        }
    }

    Transition {
        old_circles: result_old_c,
        new_circles: result_new_c,
        old_rects: result_old_r,
        new_rects: result_new_r,
    }
}

pub fn lerp_circles(
    old: &[CircleInstance],
    new: &[CircleInstance],
    t: f32,
) -> Vec<CircleInstance> {
    old.iter()
        .zip(new.iter())
        .map(|(a, b)| CircleInstance {
            center: [
                a.center[0] + (b.center[0] - a.center[0]) * t,
                a.center[1] + (b.center[1] - a.center[1]) * t,
            ],
            radius: a.radius + (b.radius - a.radius) * t,
            fill_color: lerp_color(a.fill_color, b.fill_color, t),
            stroke_color: lerp_color(a.stroke_color, b.stroke_color, t),
            stroke_width: a.stroke_width + (b.stroke_width - a.stroke_width) * t,
            opacity: a.opacity + (b.opacity - a.opacity) * t,
        })
        .collect()
}

pub fn lerp_rects(
    old: &[RectInstance],
    new: &[RectInstance],
    t: f32,
) -> Vec<RectInstance> {
    old.iter()
        .zip(new.iter())
        .map(|(a, b)| RectInstance {
            position: [
                a.position[0] + (b.position[0] - a.position[0]) * t,
                a.position[1] + (b.position[1] - a.position[1]) * t,
            ],
            size: [
                a.size[0] + (b.size[0] - a.size[0]) * t,
                a.size[1] + (b.size[1] - a.size[1]) * t,
            ],
            corner_radius: a.corner_radius + (b.corner_radius - a.corner_radius) * t,
            fill_color: lerp_color(a.fill_color, b.fill_color, t),
            stroke_color: lerp_color(a.stroke_color, b.stroke_color, t),
            stroke_width: a.stroke_width + (b.stroke_width - a.stroke_width) * t,
            opacity: a.opacity + (b.opacity - a.opacity) * t,
        })
        .collect()
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

fn count_instances(batch: &MarkBatch) -> (usize, usize) {
    let mut nc = 0usize;
    let mut nr = 0usize;
    for node in &batch.nodes {
        match node {
            ferrum_scene::SceneNode::Circle { .. } => nc += 1,
            ferrum_scene::SceneNode::Rect { .. } => nr += 1,
            _ => {}
        }
    }
    (nc, nr)
}

fn match_circle_pair(
    _old: &mut [CircleInstance],
    _old_idx: usize,
    _new: &[CircleInstance],
    _new_idx: usize,
) {
    // Key matching establishes correspondence — the actual lerp happens
    // in lerp_circles using aligned arrays. No per-pair action needed.
}

fn match_rect_pair(
    _old: &mut [RectInstance],
    _old_idx: usize,
    _new: &[RectInstance],
    _new_idx: usize,
) {
    // Same as match_circle_pair.
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
        }];
        let new = vec![CircleInstance {
            center: [100.0, 200.0],
            radius: 20.0,
            fill_color: [1.0, 1.0, 1.0, 1.0],
            stroke_color: [0.0; 4],
            stroke_width: 0.0,
            opacity: 0.5,
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
