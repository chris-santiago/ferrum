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

use serde::{Deserialize, Serialize};

pub use self::axis::{
    AxesInput, AxisInput, AxisLayout, AxisOrient, AxisTitleLayout, TickLayout,
};
pub use self::facet::{FacetGroup, FacetMode, FacetSpec};
pub use self::geometry::{Inset, Rect, Viewport};
pub use self::legend::{
    LegendDirection, LegendEntry, LegendEntryLayout, LegendLayout, LegendOrient, SymbolKind,
};
pub use self::panel::{FacetKey, PanelLayout};
pub use self::text_metrics::{HeuristicMetrics, TextMetrics};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutResult {
    pub viewport: Rect,
    pub panels: Vec<PanelLayout>,
    pub axes: Vec<AxisLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub legend: Option<LegendLayout>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<LayoutWarning>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayoutWarning {
    PanelCollapsed { panel_index: usize },
    LabelsElided { axis: usize, count: u32 },
    LegendOverflowed { entries_dropped: u32 },
    PanelsDropped { count: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutError {
    InvalidViewport { width: f64, height: f64 },
    InvalidFacetSpec(String),
    PaddingExceedsViewport { padding: f64, viewport_dim: f64 },
    EmptyFacetGroups,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::InvalidViewport { width, height } =>
                write!(f, "invalid viewport: width={width}, height={height} (both must be > 0)"),
            LayoutError::InvalidFacetSpec(s) =>
                write!(f, "invalid facet spec: {s}"),
            LayoutError::PaddingExceedsViewport { padding, viewport_dim } =>
                write!(f, "padding {padding} exceeds viewport dimension {viewport_dim}"),
            LayoutError::EmptyFacetGroups =>
                write!(f, "facet specified but facet_groups input is empty"),
        }
    }
}

impl std::error::Error for LayoutError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_result_round_trip_empty() {
        let r = LayoutResult {
            viewport: Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 },
            panels: vec![],
            axes: vec![],
            legend: None,
            warnings: vec![],
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
        assert!(!json.contains("legend"));
        assert!(!json.contains("warnings"));
    }

    #[test]
    fn layout_warning_round_trip_each_variant() {
        for w in [
            LayoutWarning::PanelCollapsed { panel_index: 2 },
            LayoutWarning::LabelsElided { axis: 0, count: 5 },
            LayoutWarning::LegendOverflowed { entries_dropped: 3 },
            LayoutWarning::PanelsDropped { count: 1 },
        ] {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: LayoutWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w);
        }
    }
}
