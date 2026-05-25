//! Facet input/output types and grid arithmetic. Phase 6 supports two modes:
//! Wrap (ncols set, nrows derived from n_panels) and Grid (both explicit;
//! panels beyond nrows*ncols are dropped with a warning).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;
use super::panel::FacetKey;

/// Spec-side facet declaration. Carried by `ChartSpec.facet`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetSpec {
    /// Primary facet field (column dimension in grid mode, or the sole field in wrap mode).
    pub field: String,
    /// Row dimension field for grid-mode facets. Stored for wire-format round-trip;
    /// the renderer distributes panels by `mode.nrows`/`mode.ncols` independently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row: Option<String>,
    pub mode: FacetMode,
    /// If set, overrides `theme.column_padding` / `theme.row_padding` symmetrically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum FacetMode {
    Wrap { ncols: u32 },
    Grid { nrows: u32, ncols: u32 },
}

/// Caller-supplied per-panel input. `n_rows` is informational only — Phase 6
/// does not use it for layout decisions but Phase 7+ may.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetGroup {
    pub key: FacetKey,
    pub n_rows: u64,
}

/// Computed grid sizing. `cell_rect(row, col, origin)` returns the panel rect.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetGrid {
    pub mode: FacetMode,
    pub n_panels: u32,
    pub cell_w: f64,
    pub cell_h: f64,
    pub gutter_x: f64,
    pub gutter_y: f64,
    pub origin: Rect,
}

impl FacetGrid {
    /// Wrap mode: ncols is fixed; nrows = ceil(n_panels / ncols).
    pub fn compute_wrap(
        ncols: u32,
        n_panels: u32,
        origin: Rect,
        gutter_x: f64,
        gutter_y: f64,
    ) -> FacetGrid {
        // L-1: guard negative spacing — negative gutters cause panels to overlap.
        let gutter_x = gutter_x.max(0.0);
        let gutter_y = gutter_y.max(0.0);
        let ncols = ncols.max(1);
        let nrows = (n_panels + ncols - 1) / ncols;
        let nrows = nrows.max(1);
        let total_x_gutter = gutter_x * (ncols.saturating_sub(1) as f64);
        let total_y_gutter = gutter_y * (nrows.saturating_sub(1) as f64);
        let cell_w = ((origin.w - total_x_gutter) / ncols as f64).max(0.0);
        let cell_h = ((origin.h - total_y_gutter) / nrows as f64).max(0.0);
        FacetGrid {
            mode: FacetMode::Wrap { ncols },
            n_panels,
            cell_w,
            cell_h,
            gutter_x,
            gutter_y,
            origin,
        }
    }

    pub fn cell_rect(&self, row: u32, col: u32) -> Rect {
        Rect {
            x: self.origin.x + col as f64 * (self.cell_w + self.gutter_x),
            y: self.origin.y + row as f64 * (self.cell_h + self.gutter_y),
            w: self.cell_w,
            h: self.cell_h,
        }
    }

    /// Returns the (row, col) for each of the `n_panels` panels, in panel-index
    /// order. For wrap mode: row-major. For grid mode (Task 7): same row-major
    /// order, capped at nrows*ncols.
    pub fn panel_positions(&self) -> Vec<(u32, u32)> {
        let ncols = match self.mode {
            FacetMode::Wrap { ncols } => ncols,
            FacetMode::Grid { ncols, .. } => ncols,
        };
        let cap = match self.mode {
            FacetMode::Wrap { .. } => self.n_panels,
            FacetMode::Grid { nrows, ncols } => self.n_panels.min(nrows * ncols),
        };
        (0..cap)
            .map(|i| (i / ncols, i % ncols))
            .collect()
    }

    /// Grid mode: nrows and ncols both fixed. Panels beyond nrows*ncols are
    /// dropped (caller should emit `LayoutWarning::PanelsDropped`).
    pub fn compute_grid(
        nrows: u32,
        ncols: u32,
        n_panels: u32,
        origin: Rect,
        gutter_x: f64,
        gutter_y: f64,
    ) -> FacetGrid {
        // L-1: guard negative spacing — negative gutters cause panels to overlap.
        let gutter_x = gutter_x.max(0.0);
        let gutter_y = gutter_y.max(0.0);
        let nrows = nrows.max(1);
        let ncols = ncols.max(1);
        let total_x_gutter = gutter_x * (ncols.saturating_sub(1) as f64);
        let total_y_gutter = gutter_y * (nrows.saturating_sub(1) as f64);
        let cell_w = ((origin.w - total_x_gutter) / ncols as f64).max(0.0);
        let cell_h = ((origin.h - total_y_gutter) / nrows as f64).max(0.0);
        FacetGrid {
            mode: FacetMode::Grid { nrows, ncols },
            n_panels,
            cell_w,
            cell_h,
            gutter_x,
            gutter_y,
            origin,
        }
    }

    /// In grid mode, returns max(0, n_panels - nrows*ncols). Always 0 in wrap mode.
    pub fn dropped_count(&self) -> u32 {
        match self.mode {
            FacetMode::Grid { nrows, ncols } => {
                let cap = nrows * ncols;
                if self.n_panels > cap { self.n_panels - cap } else { 0 }
            }
            FacetMode::Wrap { .. } => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facet_spec_round_trip_wrap() {
        let s = FacetSpec {
            field: "species".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 3 },
            spacing: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: FacetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
        assert!(json.contains(r#""kind":"wrap""#));
        assert!(json.contains(r#""ncols":3"#));
    }

    #[test]
    fn facet_spec_round_trip_grid() {
        let s = FacetSpec {
            field: "year".into(),
            row: None,
            mode: FacetMode::Grid { nrows: 2, ncols: 3 },
            spacing: Some(12.0),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: FacetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
        assert!(json.contains(r#""kind":"grid""#));
    }

    #[test]
    fn facet_spec_omits_spacing_when_none() {
        let s = FacetSpec {
            field: "f".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("spacing"));
    }

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn facet_grid_wrap_exact_fit() {
        // viewport 600x400, ncols=3, n_panels=3, gutter=0
        // → 1 row of 3 cells, each 200x400.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_wrap(3, 3, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 400.0);
        assert_eq!(grid.cell_rect(0, 0), rect(0.0, 0.0, 200.0, 400.0));
        assert_eq!(grid.cell_rect(0, 1), rect(200.0, 0.0, 200.0, 400.0));
        assert_eq!(grid.cell_rect(0, 2), rect(400.0, 0.0, 200.0, 400.0));
    }

    #[test]
    fn facet_grid_wrap_ragged() {
        // ncols=3, n_panels=5, gutter=0 → 2 rows; last row has 2 cells.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_wrap(3, 5, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 200.0);
        assert_eq!(grid.cell_rect(0, 0), rect(0.0, 0.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(0, 1), rect(200.0, 0.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(0, 2), rect(400.0, 0.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(1, 0), rect(0.0, 200.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(1, 1), rect(200.0, 200.0, 200.0, 200.0));
    }

    #[test]
    fn facet_grid_wrap_with_gutters() {
        // viewport 620x420, ncols=3, n_panels=5, gutter_x=10, gutter_y=10
        // cell_w = (620 - 2*10) / 3 = 200
        // cell_h = (420 - 1*10) / 2 = 205
        let origin = rect(0.0, 0.0, 620.0, 420.0);
        let grid = FacetGrid::compute_wrap(3, 5, origin, 10.0, 10.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 205.0);
        assert_eq!(grid.cell_rect(0, 0), rect(0.0, 0.0, 200.0, 205.0));
        assert_eq!(grid.cell_rect(0, 1), rect(210.0, 0.0, 200.0, 205.0));
        assert_eq!(grid.cell_rect(0, 2), rect(420.0, 0.0, 200.0, 205.0));
        assert_eq!(grid.cell_rect(1, 0), rect(0.0, 215.0, 200.0, 205.0));
    }

    #[test]
    fn facet_grid_wrap_panel_index_to_row_col() {
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_wrap(3, 5, origin, 0.0, 0.0);
        let panels = grid.panel_positions();
        assert_eq!(panels, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]);
    }

    #[test]
    fn facet_grid_mode_exact_fit() {
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_grid(2, 3, 6, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 200.0);
        assert_eq!(grid.panel_positions().len(), 6);
        assert_eq!(grid.cell_rect(1, 2), rect(400.0, 200.0, 200.0, 200.0));
        assert_eq!(grid.dropped_count(), 0);
    }

    #[test]
    fn facet_grid_mode_overflow_drops_panels() {
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_grid(2, 3, 8, origin, 0.0, 0.0);
        assert_eq!(grid.panel_positions().len(), 6);
        assert_eq!(grid.dropped_count(), 2);
    }

    #[test]
    fn facet_grid_mode_underfill_does_not_panic() {
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_grid(2, 3, 4, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 200.0);
        assert_eq!(grid.panel_positions().len(), 4);
        assert_eq!(grid.dropped_count(), 0);
    }
}
