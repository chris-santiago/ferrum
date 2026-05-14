use ferrum_scene::InteractionConfig;

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

    pub fn on_wheel(&mut self, panel_id: usize, delta: f64, cursor_x: f64, cursor_y: f64) {
        let Some(t) = self.transforms.get_mut(panel_id) else {
            return;
        };
        let factor = 1.0 + delta * 0.001;
        let new_sx = (t.sx * factor).clamp(self.zoom_range.0, self.zoom_range.1);
        let new_sy = (t.sy * factor).clamp(self.zoom_range.0, self.zoom_range.1);

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

fn tick_level_for_zoom(zoom: f64) -> usize {
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
        state.on_wheel(0, 500.0, 100.0, 100.0);
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
        state.on_wheel(0, 1000.0, 50.0, 50.0);
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
            state.on_wheel(0, 5000.0, 0.0, 0.0);
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
}
