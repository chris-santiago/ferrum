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

    /// Set the transform for the given panel to absolute values.
    ///
    /// `k` is the uniform scale factor; `tx`/`ty` are the translation offsets.
    /// This replaces whatever accumulated state `on_wheel`/`on_pan` built up,
    /// and is the entry point for D3-zoom which provides absolute transforms.
    pub fn set_absolute(&mut self, panel_id: usize, k: f64, tx: f64, ty: f64) {
        let Some(t) = self.transforms.get_mut(panel_id) else {
            return;
        };
        t.sx = k.clamp(self.zoom_range.0, self.zoom_range.1);
        t.sy = t.sx; // D3-zoom uses uniform scaling
        t.tx = tx;
        t.ty = ty;
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

/// Select the affine transform a given panel's marks should be drawn with.
///
/// Returns `transforms[panel_id]` when present, otherwise the identity
/// transform. This is the per-panel render path's byte-stability anchor: at
/// rest every panel maps to identity, so the per-panel uniform path produces
/// the same affine the former single mark uniform did. Out-of-range ids never
/// panic — they fall back to identity. Factored out here (host-compiled) so the
/// panel → affine selection is unit-testable without a GPU device.
pub(crate) fn select_panel_transform(transforms: &[Affine2], panel_id: usize) -> Affine2 {
    transforms
        .get(panel_id)
        .copied()
        .unwrap_or_else(Affine2::identity)
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

    // ── set_absolute → inverse_apply coordinate-space correctness ───────────

    #[test]
    fn bug_hunt_set_absolute_then_inverse_apply_recovers_scene_space() {
        // This is the exact coordinate-space transform that handleClick
        // must use: D3 calls setTransform(k, tx, ty), then hit_test_at
        // uses inverse_apply to convert canvas coords to scene coords.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, 3.0, 120.0, -50.0);
        let t = &state.transforms[0];
        // Scene point (80, 200): canvas = (3*80+120, 3*200-50) = (360, 550)
        let (canvas_x, canvas_y) = t.apply(80.0, 200.0);
        assert!((canvas_x - 360.0).abs() < 1e-8);
        assert!((canvas_y - 550.0).abs() < 1e-8);
        // Inverse must recover scene point
        let (scene_x, scene_y) = t.inverse_apply(canvas_x, canvas_y);
        assert!((scene_x - 80.0).abs() < 1e-8, "inverse must recover x=80");
        assert!((scene_y - 200.0).abs() < 1e-8, "inverse must recover y=200");
    }

    #[test]
    fn bug_hunt_set_absolute_negative_translation() {
        // Negative translation: user pans left/up.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, 1.5, -100.0, -75.0);
        let t = &state.transforms[0];
        let (fx, fy) = t.apply(0.0, 0.0);
        assert!((fx - (-100.0)).abs() < 1e-8, "origin maps to tx");
        assert!((fy - (-75.0)).abs() < 1e-8, "origin maps to ty");
        let (bx, by) = t.inverse_apply(fx, fy);
        assert!((bx).abs() < 1e-8, "roundtrip x must be 0");
        assert!((by).abs() < 1e-8, "roundtrip y must be 0");
    }

    #[test]
    fn bug_hunt_inverse_apply_with_identity_is_noop() {
        let t = Affine2::identity();
        let (x, y) = t.inverse_apply(42.0, 99.0);
        assert!((x - 42.0).abs() < 1e-10, "identity inverse must not change x");
        assert!((y - 99.0).abs() < 1e-10, "identity inverse must not change y");
    }

    #[test]
    fn bug_hunt_set_absolute_zero_scale_clamps_to_min() {
        // Zero scale should be clamped to zoom_range.0 (0.1).
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, 0.0, 0.0, 0.0);
        let t = &state.transforms[0];
        assert!(t.sx >= 0.1, "zero scale must clamp to min 0.1, got {}", t.sx);
        assert!(t.sy >= 0.1, "zero scale must clamp to min 0.1, got {}", t.sy);
    }

    #[test]
    fn bug_hunt_set_absolute_negative_scale_clamps_to_min() {
        // Negative scale should be clamped to zoom_range.0 (0.1).
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, -5.0, 0.0, 0.0);
        let t = &state.transforms[0];
        assert!(t.sx >= 0.1, "negative scale must clamp to min 0.1, got {}", t.sx);
    }

    #[test]
    fn bug_hunt_multiple_set_absolute_last_wins() {
        // Multiple set_absolute calls: the last one should overwrite.
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, 2.0, 10.0, 20.0);
        state.set_absolute(0, 5.0, 100.0, 200.0);
        let t = &state.transforms[0];
        assert!((t.sx - 5.0).abs() < 1e-10);
        assert!((t.tx - 100.0).abs() < 1e-10);
        assert!((t.ty - 200.0).abs() < 1e-10);
    }

    #[test]
    fn bug_hunt_tick_level_for_very_small_zoom() {
        // Very small zoom factor (below 0.5) should map to level 0.
        assert_eq!(tick_level_for_zoom(0.01), 0);
        assert_eq!(tick_level_for_zoom(0.0), 0);
    }

    #[test]
    fn bug_hunt_tick_level_for_very_large_zoom() {
        // Very large zoom factor should map to the highest level (3).
        assert_eq!(tick_level_for_zoom(100.0), 3);
        assert_eq!(tick_level_for_zoom(1000.0), 3);
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

    // ── FA-18: per-panel mark transform selection ────────────────────────────

    /// The selector must pick each panel's OWN affine: a non-uniform rescale
    /// (sx ≠ sy) on one panel must not leak into a sibling that is at identity.
    /// This is the core invariant that stops sibling shear during a reactive
    /// domain rescale.
    #[test]
    fn select_panel_transform_picks_per_panel_affine() {
        // Panel 0: identity. Panel 1: non-uniform rescale (sx != sy), like an
        // x-only domain rescale written by `apply_reactive_rescale`.
        let rescaled = Affine2 { sx: 3.0, sy: 1.0, tx: -40.0, ty: 0.0 };
        let transforms = vec![Affine2::identity(), rescaled];

        let t0 = select_panel_transform(&transforms, 0);
        assert!(
            (t0.sx - 1.0).abs() < 1e-9 && (t0.sy - 1.0).abs() < 1e-9,
            "panel 0 must stay identity, got sx={} sy={}", t0.sx, t0.sy
        );

        let t1 = select_panel_transform(&transforms, 1);
        assert!((t1.sx - 3.0).abs() < 1e-9, "panel 1 sx must be the rescale, got {}", t1.sx);
        assert!((t1.sy - 1.0).abs() < 1e-9, "panel 1 sy must be 1.0 (x-only rescale), got {}", t1.sy);
        assert!((t1.tx - (-40.0)).abs() < 1e-9, "panel 1 tx must be the rescale offset, got {}", t1.tx);

        // Out-of-range panel_id falls back to identity (never panics).
        let t_oob = select_panel_transform(&transforms, 9);
        assert!(
            (t_oob.sx - 1.0).abs() < 1e-9 && (t_oob.sy - 1.0).abs() < 1e-9,
            "out-of-range panel must fall back to identity"
        );
    }

    /// At rest (all panels identity) the per-panel selection yields identity
    /// for every slot, so the rendered affine matches the former single-uniform
    /// path exactly — the byte-stability anchor.
    #[test]
    fn select_panel_transform_identity_at_rest() {
        let transforms = vec![Affine2::identity(); 3];
        for panel_id in 0..3 {
            let t = select_panel_transform(&transforms, panel_id);
            assert!((t.sx - 1.0).abs() < 1e-9, "slot {panel_id} sx");
            assert!((t.sy - 1.0).abs() < 1e-9, "slot {panel_id} sy");
            assert!(t.tx.abs() < 1e-9 && t.ty.abs() < 1e-9, "slot {panel_id} translation");
        }
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

    #[test]
    fn set_absolute_sets_scale_and_translation() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, 2.0, 100.0, 50.0);
        let t = &state.transforms[0];
        assert!((t.sx - 2.0).abs() < 1e-10, "sx should be 2.0, got {}", t.sx);
        assert!((t.sy - 2.0).abs() < 1e-10, "sy should equal sx (uniform), got {}", t.sy);
        assert!((t.tx - 100.0).abs() < 1e-10, "tx should be 100.0, got {}", t.tx);
        assert!((t.ty - 50.0).abs() < 1e-10, "ty should be 50.0, got {}", t.ty);
    }

    #[test]
    fn set_absolute_then_reset_returns_to_identity() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(0, 3.0, 10.0, 20.0);
        state.reset(0);
        let t = &state.transforms[0];
        assert!((t.sx - 1.0).abs() < 1e-10, "sx should be 1.0 after reset");
        assert!((t.sy - 1.0).abs() < 1e-10, "sy should be 1.0 after reset");
        assert!(t.tx.abs() < 1e-10, "tx should be 0.0 after reset");
        assert!(t.ty.abs() < 1e-10, "ty should be 0.0 after reset");
    }

    #[test]
    fn set_absolute_out_of_bounds_panel_is_noop() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        state.set_absolute(99, 5.0, 300.0, 400.0);
        // Panel 0 should be untouched
        let t = &state.transforms[0];
        assert!((t.sx - 1.0).abs() < 1e-10);
        assert!((t.sy - 1.0).abs() < 1e-10);
        assert!(t.tx.abs() < 1e-10);
        assert!(t.ty.abs() < 1e-10);
    }

    #[test]
    fn set_absolute_clamps_scale_to_zoom_range() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        // Scale below minimum (0.1)
        state.set_absolute(0, 0.01, 0.0, 0.0);
        assert!((state.transforms[0].sx - 0.1).abs() < 1e-10, "scale should clamp to min");
        // Scale above maximum (50.0)
        state.set_absolute(0, 100.0, 0.0, 0.0);
        assert!((state.transforms[0].sx - 50.0).abs() < 1e-10, "scale should clamp to max");
    }

    #[test]
    fn set_absolute_overwrites_accumulated_state() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        // Accumulate some zoom and pan state
        state.on_wheel(0, 500.0, 100.0, 100.0, ScaleMode::Independent);
        state.on_pan(0, 30.0, -20.0);
        // Now set_absolute should overwrite everything
        state.set_absolute(0, 1.5, 42.0, -17.0);
        let t = &state.transforms[0];
        assert!((t.sx - 1.5).abs() < 1e-10);
        assert!((t.sy - 1.5).abs() < 1e-10);
        assert!((t.tx - 42.0).abs() < 1e-10);
        assert!((t.ty - (-17.0)).abs() < 1e-10);
    }

    /// B1 regression: When zoomed 2x with translation, canvas-space coordinates
    /// must be converted to scene-space via inverse_apply before comparing against
    /// mark positions. This test exercises the exact math that hit_test_at must use
    /// for packed-circle fallback: set_absolute(2x, +50, +30) then inverse_apply
    /// a canvas-space point to recover the scene-space position.
    #[test]
    fn inverse_apply_converts_canvas_to_scene_space_under_zoom() {
        let config = InteractionConfig::default();
        let mut state = ZoomPanState::new(1, &config);
        // Simulate D3-zoom: 2x scale with translation (50, 30)
        state.set_absolute(0, 2.0, 50.0, 30.0);
        let t = &state.transforms[0];

        // A scene-space point at (100, 200) maps to canvas-space:
        //   canvas_x = 2.0 * 100 + 50 = 250
        //   canvas_y = 2.0 * 200 + 30 = 430
        let (canvas_x, canvas_y) = t.apply(100.0, 200.0);
        assert!((canvas_x - 250.0).abs() < 1e-10);
        assert!((canvas_y - 430.0).abs() < 1e-10);

        // inverse_apply must recover the scene-space point from canvas-space
        let (scene_x, scene_y) = t.inverse_apply(canvas_x, canvas_y);
        assert!(
            (scene_x - 100.0).abs() < 1e-10,
            "inverse_apply must recover scene x=100.0, got {scene_x}"
        );
        assert!(
            (scene_y - 200.0).abs() < 1e-10,
            "inverse_apply must recover scene y=200.0, got {scene_y}"
        );

        // Critically: the raw canvas coords (250, 430) differ from scene coords (100, 200).
        // If hit_test_at compares canvas coords directly against scene-space mark positions,
        // it will miss hits that are actually under the cursor.
        assert!(
            (canvas_x - 100.0).abs() > 1.0,
            "canvas x must differ from scene x to demonstrate the bug"
        );
    }
}
