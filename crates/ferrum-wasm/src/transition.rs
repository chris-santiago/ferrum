use crate::scene_load::{CircleInstance, RectInstance};

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
            stroke_opacity: a.stroke_opacity + (b.stroke_opacity - a.stroke_opacity) * t,
            stroke_dash: a.stroke_dash, // dash palette index: use old value (no lerp for discrete)
            angle: a.angle + (b.angle - a.angle) * t,
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
            stroke_opacity: a.stroke_opacity + (b.stroke_opacity - a.stroke_opacity) * t,
            stroke_dash: a.stroke_dash, // palette index: use old value (discrete, no lerp)
            angle: a.angle + (b.angle - a.angle) * t,
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
        assert!(result.is_empty(), "lerp_circles on empty slices must return empty vec");
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
        assert!((result[0].center[0] - 0.0).abs() < 0.001, "t=0 must return old center.x");
        assert!((result[0].radius - 10.0).abs() < 0.001, "t=0 must return old radius");
    }

    #[test]
    fn bug_hunt_lerp_circles_t_one_returns_new() {
        let old = vec![make_circle(0.0, 0.0, 10.0)];
        let new = vec![make_circle(100.0, 200.0, 20.0)];
        let result = lerp_circles(&old, &new, 1.0);
        assert!((result[0].center[0] - 100.0).abs() < 0.001, "t=1 must return new center.x");
        assert!((result[0].radius - 20.0).abs() < 0.001, "t=1 must return new radius");
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
        assert_eq!(result.len(), 1, "mismatched lengths must truncate to shorter (zip behaviour)");
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
        assert!((mid[0].fill_color[0] - 0.5).abs() < 0.01, "mid-grey R channel");
        assert!((mid[0].fill_color[1] - 0.5).abs() < 0.01, "mid-grey G channel");
        assert!((mid[0].fill_color[2] - 0.5).abs() < 0.01, "mid-grey B channel");
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
            "corner_radius must lerp to 10.0 at t=0.5, got {}", mid[0].corner_radius
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
