//! Phase 6 — layout engine. Pure function: ChartSpec + Theme + Viewport ->
//! pixel rectangles for panels, axes, legend. No I/O, no rendering, no data
//! values touched. See docs/superpowers/specs/2026-05-09-layout-engine-design.md.

pub(crate) mod geometry;
pub(crate) mod text_metrics;
pub(crate) mod panel;
pub(crate) mod facet;
pub(crate) mod axis;
pub(crate) mod legend;
pub(crate) mod binding;

// Spec §6.1 constants.
pub const LABEL_OVERLAP_TOLERANCE: f64 = 0.10;
pub const DEFAULT_LABEL_ANGLE: f64 = -45.0;
pub const DEFAULT_HEURISTIC_K: f64 = 0.6;
pub const MIN_PANEL_DIM: f64 = 1.0;
pub const DEFAULT_PADDING: f64 = 8.0;
pub const DEFAULT_LABEL_FONT_SIZE: f64 = 11.0;
pub const DEFAULT_TITLE_FONT_SIZE: f64 = 13.0;
pub const DEFAULT_AXIS_TITLE_PADDING: f64 = 4.0;
