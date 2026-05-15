use ferrum_scene::InteractionConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    Independent,
    Uniform,
}

#[derive(Debug, Clone, Copy)]
pub struct Affine2 {
    pub sx: f64,
    pub sy: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Affine2 {
    pub fn identity() -> Self {
        Self {
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn zoom_factor(&self) -> f64 {
        self.sx.abs().max(self.sy.abs())
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.sx * x + self.tx, self.sy * y + self.ty)
    }

    pub fn inverse_apply(&self, x: f64, y: f64) -> (f64, f64) {
        if self.sx.abs() < 1e-12 || self.sy.abs() < 1e-12 {
            return (x, y);
        }
        ((x - self.tx) / self.sx, (y - self.ty) / self.sy)
    }
}

pub struct ZoomPanState {
    pub transforms: Vec<Affine2>,
    pub zoom_range: (f64, f64),
}

impl ZoomPanState {
    pub fn new(n_panels: usize, _config: &InteractionConfig) -> Self {
        ZoomPanState {
            transforms: vec![Affine2::identity(); n_panels],
            zoom_range: (0.1, 50.0),
        }
    }

    pub fn on_wheel(
        &mut self,
        panel_id: usize,
        delta: f64,
        cursor_x: f64,
        cursor_y: f64,
        scale_mode: ScaleMode,
    ) {
        let Some(t) = self.transforms.get_mut(panel_id) else {
            return;
        };
        let factor = 1.0 + delta * 0.001;
        let new_sx = (t.sx * factor).clamp(self.zoom_range.0, self.zoom_range.1);
        // Uniform scaling: sy always equals sx (CoordFixed / square-pixel panels).
        let new_sy = if scale_mode == ScaleMode::Uniform { new_sx } else {
            (t.sy * factor).clamp(self.zoom_range.0, self.zoom_range.1)
        };

        t.tx = cursor_x - new_sx * ((cursor_x - t.tx) / t.sx);
        t.ty = cursor_y - new_sy * ((cursor_y - t.ty) / t.sy);
        t.sx = new_sx;
        t.sy = new_sy;
    }

    pub fn on_pan(&mut self, panel_id: usize, dx: f64, dy: f64) {
        let Some(t) = self.transforms.get_mut(panel_id) else {
            return;
        };
        t.tx += dx;
        t.ty += dy;
    }

    pub fn reset(&mut self, panel_id: usize) {
        if let Some(t) = self.transforms.get_mut(panel_id) {
            *t = Affine2::identity();
        }
    }

    pub fn current_tick_level_idx(&self, panel_id: usize) -> usize {
        let zoom = self
            .transforms
            .get(panel_id)
            .map(|t| t.zoom_factor())
            .unwrap_or(1.0);
        tick_level_for_zoom(zoom)
    }
}

pub(crate) fn tick_level_for_zoom(zoom: f64) -> usize {
    if zoom < 0.5 {
        0
    } else if zoom < 2.0 {
        1
    } else if zoom < 4.0 {
        2
    } else {
        3
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod bug_hunt_tests {
    use super::*;

    // ── ZoomPanState / Affine2: edge and boundary cases ───────────────────

    #[test]
    fn bug_hunt_on_wheel_out_of_bounds_panel_id_is_noop() {
        // Calling on_wheel with a panel_id >= transforms.len() must not panic
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(2, &config);
        state.on_wheel(99, 500.0, 100.0, 100.0, ScaleMode::Independent);
        // transforms for valid panels must be unchanged
        assert!((state.transforms[0].sx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_on_pan_out_of_bounds_panel_id_is_noop() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_pan(99, 10.0, 20.0);
        assert!((state.transforms[0].tx).abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_reset_out_of_bounds_panel_id_is_noop() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.transforms[0].sx = 3.0;
        state.reset(99);
        // panel 0 should be unchanged
        assert!((state.transforms[0].sx - 3.0).abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_negative_wheel_delta_zooms_out() {
        // Negative delta should reduce scale (zoom out)
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_wheel(0, -500.0, 0.0, 0.0, ScaleMode::Independent);
        assert!(state.transforms[0].sx < 1.0, "negative delta must zoom out");
    }

    #[test]
    fn bug_hunt_zoom_clamps_to_min() {
        // Repeated large negative deltas must clamp to zoom_range.0 (0.1)
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        for _ in 0..200 {
            state.on_wheel(0, -5000.0, 0.0, 0.0, ScaleMode::Independent);
        }
        assert!(state.transforms[0].sx >= 0.1, "scale must not go below min");
    }

    #[test]
    fn bug_hunt_inverse_apply_near_zero_scale_does_not_divide_by_zero() {
        // When scale is effectively zero, inverse_apply must not panic or produce inf/nan
        let t = Affine2 {
            sx: 1e-15,
            sy: 1e-15,
            tx: 0.0,
            ty: 0.0,
        };
        // The guard is 1e-12; 1e-15 < 1e-12, so it should return (x, y) unchanged
        let (rx, ry) = t.inverse_apply(5.0, 10.0);
        assert!(rx.is_finite(), "inverse_apply must not produce inf/nan for near-zero scale");
        assert!(ry.is_finite());
    }

    #[test]
    fn bug_hunt_zoom_factor_returns_max_of_abs_sx_sy() {
        let t = Affine2 { sx: -3.0, sy: 2.0, tx: 0.0, ty: 0.0 };
        // zoom_factor should be max(|sx|, |sy|) = max(3.0, 2.0) = 3.0
        assert!((t.zoom_factor() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_tick_level_exactly_at_breakpoint() {
        // Values exactly at breakpoints should fall into the LOWER bucket (strict <)
        assert_eq!(tick_level_for_zoom(0.5), 1, "0.5 is at the >=0.5 boundary");
        assert_eq!(tick_level_for_zoom(2.0), 2, "2.0 is at the >=2.0 boundary");
        assert_eq!(tick_level_for_zoom(4.0), 3, "4.0 is at the >=4.0 boundary");
    }

    #[test]
    fn bug_hunt_current_tick_level_for_invalid_panel_returns_default() {
        // out-of-bounds panel_id must not panic; fallback zoom is 1.0 → level 1
        let config = InteractionConfig::default();
        let state = ZoomPanState::new(1, &config);
        let level = state.current_tick_level_idx(99);
        assert_eq!(level, 1, "fallback zoom=1.0 maps to tick level 1");
    }

    #[test]
    fn bug_hunt_zero_panel_count_is_safe() {
        // ZoomPanState with 0 panels must be constructible and tolerate queries
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(0, &config);
        state.on_wheel(0, 500.0, 100.0, 100.0, ScaleMode::Independent); // must not panic
        state.on_pan(0, 10.0, 20.0); // must not panic
        state.reset(0); // must not panic
        let level = state.current_tick_level_idx(0);
        assert_eq!(level, 1);
    }

    #[test]
    fn bug_hunt_accumulated_wheel_then_pan_translation_is_finite() {
        // After many wheel + pan operations the translation must remain finite.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        for i in 0..50 {
            let delta = if i % 2 == 0 { 300.0 } else { -200.0 };
            state.on_wheel(0, delta, 150.0, 120.0, ScaleMode::Independent);
            state.on_pan(0, 5.0 * (i as f64), -3.0 * (i as f64));
        }
        let t = &state.transforms[0];
        assert!(t.tx.is_finite(), "tx must remain finite after accumulated ops");
        assert!(t.ty.is_finite(), "ty must remain finite after accumulated ops");
        assert!(t.sx.is_finite(), "sx must remain finite");
        assert!(t.sy.is_finite(), "sy must remain finite");
    }

    #[test]
    fn bug_hunt_wheel_cursor_invariant_preserved_through_sequence() {
        // The cursor point should not move visually under repeated wheel zoom.
        // This mirrors the JS property: forward(cursor) == cursor after each wheel call.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        let (cx, cy) = (200.0, 150.0);
        for delta in [200.0_f64, 500.0, -300.0, 100.0] {
            state.on_wheel(0, delta, cx, cy, ScaleMode::Independent);
            let t = &state.transforms[0];
            let (fx, fy) = t.apply(cx, cy);
            assert!(
                (fx - cx).abs() < 1e-6,
                "cursor x must not move after wheel delta={delta}: expected={cx}, got={fx}"
            );
            assert!(
                (fy - cy).abs() < 1e-6,
                "cursor y must not move after wheel delta={delta}: expected={cy}, got={fy}"
            );
        }
    }

    #[test]
    fn bug_hunt_uniform_mode_sx_equals_sy_after_asymmetric_initial_state() {
        // Even if sx != sy initially, a Uniform wheel event must make them equal.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.transforms[0].sx = 1.5;
        state.transforms[0].sy = 3.0;
        state.on_wheel(0, 100.0, 50.0, 50.0, ScaleMode::Uniform);
        let t = &state.transforms[0];
        assert!(
            (t.sx - t.sy).abs() < 1e-10,
            "Uniform mode must produce sx == sy; got sx={}, sy={}", t.sx, t.sy
        );
    }

    #[test]
    fn bug_hunt_pan_then_reset_restores_identity() {
        // reset() must undo any pan + zoom combination.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_wheel(0, 1000.0, 100.0, 100.0, ScaleMode::Independent);
        state.on_pan(0, 300.0, -200.0);
        state.reset(0);
        let t = &state.transforms[0];
        assert!((t.sx - 1.0).abs() < 1e-10);
        assert!((t.sy - 1.0).abs() < 1e-10);
        assert!(t.tx.abs() < 1e-10);
        assert!(t.ty.abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_apply_inverse_apply_roundtrip_after_wheel_and_pan() {
        // After a wheel+pan sequence, forward then inverse must recover the original point.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_wheel(0, 800.0, 200.0, 150.0, ScaleMode::Independent);
        state.on_pan(0, 40.0, -25.0);
        let t = state.transforms[0];
        let orig = (77.0_f64, 123.0_f64);
        let (fx, fy) = t.apply(orig.0, orig.1);
        let (bx, by) = t.inverse_apply(fx, fy);
        assert!((bx - orig.0).abs() < 1e-8, "roundtrip x failed: got {bx}");
        assert!((by - orig.1).abs() < 1e-8, "roundtrip y failed: got {by}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_preserves_point() {
        let t = Affine2::identity();
        let (x, y) = t.apply(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn zoom_increases_scale() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_wheel(0, 500.0, 100.0, 100.0, ScaleMode::Independent);
        assert!(state.transforms[0].sx > 1.0);
    }

    #[test]
    fn pan_shifts_translation() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_pan(0, 10.0, 20.0);
        assert!((state.transforms[0].tx - 10.0).abs() < 1e-10);
        assert!((state.transforms[0].ty - 20.0).abs() < 1e-10);
    }

    #[test]
    fn double_click_resets() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.on_wheel(0, 1000.0, 50.0, 50.0, ScaleMode::Independent);
        state.reset(0);
        assert!((state.transforms[0].sx - 1.0).abs() < 1e-10);
        assert!((state.transforms[0].tx).abs() < 1e-10);
    }

    #[test]
    fn inverse_round_trips() {
        let t = Affine2 {
            sx: 2.0,
            sy: 3.0,
            tx: 10.0,
            ty: -5.0,
        };
        let (fx, fy) = t.apply(7.0, 11.0);
        let (bx, by) = t.inverse_apply(fx, fy);
        assert!((bx - 7.0).abs() < 1e-10);
        assert!((by - 11.0).abs() < 1e-10);
    }

    #[test]
    fn zoom_clamps() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        for _ in 0..100 {
            state.on_wheel(0, 5000.0, 0.0, 0.0, ScaleMode::Independent);
        }
        assert!(state.transforms[0].sx <= 50.0);
    }

    #[test]
    fn tick_level_breakpoints() {
        assert_eq!(tick_level_for_zoom(0.3), 0);
        assert_eq!(tick_level_for_zoom(1.0), 1);
        assert_eq!(tick_level_for_zoom(3.0), 2);
        assert_eq!(tick_level_for_zoom(10.0), 3);
    }

    #[test]
    fn coord_fixed_enforces_uniform_scale() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        // Initial sx/sy are different: simulate a non-square zoom then fix it.
        state.transforms[0].sx = 1.5;
        state.transforms[0].sy = 2.0;
        state.on_wheel(0, 100.0, 50.0, 50.0, ScaleMode::Uniform);
        let t = &state.transforms[0];
        assert!((t.sx - t.sy).abs() < 1e-10, "sx={} sy={} must be equal for CoordFixed", t.sx, t.sy);
    }
}
