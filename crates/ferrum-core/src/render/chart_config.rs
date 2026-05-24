//! Chart-level configuration (axis, legend, grid, padding, color).
//!
//! `ChartConfig` is the Rust mirror of the `chart_config` dict passed from
//! Python's `Chart.configure(...)`. It sits between per-channel encoding
//! overrides (highest precedence) and theme defaults (lowest precedence).
//!
//! All fields are `Option<_>` with `#[serde(default)]` so missing keys are
//! silently accepted; unknown keys produce a serde error for fast feedback.

use serde::Deserialize;

/// Top-level chart configuration passed from Python via the `chart_config` dict.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct ChartConfig {
    pub axis: Option<AxisConfigSpec>,
    pub axis_x: Option<AxisConfigSpec>,
    pub axis_y: Option<AxisConfigSpec>,
    pub legend: Option<LegendConfigSpec>,
    pub grid: Option<GridConfigSpec>,
    pub padding: Option<PaddingConfigSpec>,
    pub color: Option<ColorConfigSpec>,
}

/// Per-axis configuration. Applied after per-channel values but before theme.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct AxisConfigSpec {
    pub label_angle: Option<f64>,
    pub label_font_size: Option<f64>,
    pub label_color: Option<String>,
    pub label_format: Option<String>,
    pub label_overlap: Option<String>,
    pub tick_count: Option<u32>,
    pub tick_size: Option<f64>,
    pub domain: Option<bool>,
    pub domain_color: Option<String>,
    pub domain_width: Option<f64>,
    pub grid: Option<bool>,
    pub grid_color: Option<String>,
    pub grid_dash: Option<Vec<f64>>,
    pub grid_width: Option<f64>,
    pub domain_min: Option<f64>,
    pub domain_max: Option<f64>,
    pub nice: Option<bool>,
    pub zero: Option<bool>,
}

/// Legend configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct LegendConfigSpec {
    pub orient: Option<String>,
    pub direction: Option<String>,
    pub columns: Option<u32>,
    pub title_font_size: Option<f64>,
    pub label_font_size: Option<f64>,
    pub symbol_size: Option<f64>,
    pub offset: Option<f64>,
    pub padding: Option<f64>,
}

/// Grid configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct GridConfigSpec {
    pub x: Option<bool>,
    pub y: Option<bool>,
    pub color: Option<String>,
    pub width: Option<f64>,
    pub dash: Option<Vec<f64>>,
    pub opacity: Option<f64>,
}

/// Padding configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct PaddingConfigSpec {
    pub top: Option<f64>,
    pub right: Option<f64>,
    pub bottom: Option<f64>,
    pub left: Option<f64>,
    pub auto: Option<bool>,
}

/// Color configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct ColorConfigSpec {
    pub scheme: Option<String>,
    pub sequential_scheme: Option<String>,
    pub diverging_scheme: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_json_deserializes_to_defaults() {
        let cfg: ChartConfig = serde_json::from_str("{}").unwrap();
        assert!(cfg.axis.is_none());
        assert!(cfg.axis_x.is_none());
        assert!(cfg.axis_y.is_none());
        assert!(cfg.legend.is_none());
        assert!(cfg.grid.is_none());
        assert!(cfg.padding.is_none());
        assert!(cfg.color.is_none());
    }

    #[test]
    fn partial_axis_config_deserializes() {
        let json = r#"{"axis": {"label_angle": -45, "grid": true}}"#;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let axis = cfg.axis.unwrap();
        assert_eq!(axis.label_angle, Some(-45.0));
        assert_eq!(axis.grid, Some(true));
        assert!(axis.label_font_size.is_none());
    }

    #[test]
    fn full_config_round_trip() {
        let json = r##"{
            "axis": {"label_angle": -30, "label_format": ",.0f"},
            "axis_x": {"domain_min": 0, "domain_max": 100},
            "axis_y": {"zero": true},
            "legend": {"orient": "bottom", "columns": 3},
            "grid": {"x": true, "y": false, "color": "#eee"},
            "padding": {"top": 20, "right": 20, "bottom": 40, "left": 50},
            "color": {"scheme": "tableau10", "sequential_scheme": "viridis"}
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.axis.as_ref().unwrap().label_angle, Some(-30.0));
        assert_eq!(cfg.axis_x.as_ref().unwrap().domain_min, Some(0.0));
        assert_eq!(cfg.axis_x.as_ref().unwrap().domain_max, Some(100.0));
        assert_eq!(cfg.axis_y.as_ref().unwrap().zero, Some(true));
        assert_eq!(cfg.legend.as_ref().unwrap().orient.as_deref(), Some("bottom"));
        assert_eq!(cfg.legend.as_ref().unwrap().columns, Some(3));
        assert_eq!(cfg.grid.as_ref().unwrap().x, Some(true));
        assert_eq!(cfg.grid.as_ref().unwrap().y, Some(false));
        assert_eq!(cfg.grid.as_ref().unwrap().color.as_deref(), Some("#eee"));
        assert_eq!(cfg.padding.as_ref().unwrap().top, Some(20.0));
        assert_eq!(cfg.color.as_ref().unwrap().scheme.as_deref(), Some("tableau10"));
    }
}
