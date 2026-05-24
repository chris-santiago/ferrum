//! Chart-level configuration (axis, legend, grid, padding, color, annotations).
//!
//! `ChartConfig` is the Rust mirror of the `chart_config` dict passed from
//! Python's `Chart.configure(...)`. It sits between per-channel encoding
//! overrides (highest precedence) and theme defaults (lowest precedence).
//!
//! All fields are `Option<_>` with `#[serde(default)]` so missing keys are
//! silently accepted; unknown keys produce a serde error for fast feedback.

use serde::Deserialize;

use super::annotation::AnnotationSpec;

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
    /// Title-level theme overrides (font size, weight, anchor, color, offset).
    pub title: Option<TitleConfigSpec>,
    /// Annotation layer: positioned text, lines, arrows, etc. overlaid on the plot.
    #[serde(default)]
    pub annotations: Vec<AnnotationSpec>,
    /// Structural features: secondary Y axis, axis breaks, inset charts.
    #[serde(default)]
    pub structural: Vec<StructuralSpec>,
}

// ── Structural feature specs ────────────────────────────────────────────────

/// One structural feature descriptor, deserialized from the `structural` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuralSpec {
    SecondaryY(SecondaryYSpec),
    BreakAxis(BreakAxisSpec),
    Inset(InsetSpec),
}

/// Secondary Y axis — independent right-side scale rendered over a named field.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SecondaryYSpec {
    /// Column name whose range drives the secondary scale.
    pub field: String,
    /// Mark type to render against the secondary scale (`"line"`, `"point"`, etc.).
    pub mark: String,
    /// Optional fixed fill/stroke color (hex string). Defaults to a contrasting
    /// color from the theme when absent.
    pub color: Option<String>,
    /// Overall opacity [0, 1].
    pub opacity: Option<f64>,
}

impl Default for SecondaryYSpec {
    fn default() -> Self {
        Self { field: String::new(), mark: "line".to_string(), color: None, opacity: None }
    }
}

/// Axis break — removes a range from the data domain and adds visual indicators.
#[derive(Debug, Clone, Deserialize)]
pub struct BreakAxisSpec {
    /// Which axis to break (`"x"` or `"y"`).
    pub axis: String,
    /// List of `[start, end]` data-value pairs that are excluded from the scale.
    pub gaps: Vec<[f64; 2]>,
    /// Pixel width/height of the break indicator region (default 12).
    #[serde(default = "default_break_size")]
    pub break_size: f64,
    /// Visual style: `"slash"`, `"zigzag"`, `"wave"`, or `"gap"` (default `"slash"`).
    #[serde(default = "default_break_style")]
    pub break_style: String,
}

/// Inset chart — embeds a pre-rendered SVG at normalized plot-area bounds.
#[derive(Debug, Clone, Deserialize)]
pub struct InsetSpec {
    /// Pre-rendered SVG string to embed.
    pub svg: String,
    /// `[left, top, right, bottom]` in normalized coordinates [0, 1] relative
    /// to the plot area.
    pub bounds: [f64; 4],
    /// Whether to draw a border rect around the inset (default `true`).
    #[serde(default = "default_true")]
    pub border: bool,
    /// Border stroke color (default `"#999"`).
    #[serde(default = "default_border_color")]
    pub border_color: String,
    /// Optional dash pattern for the border.
    pub border_dash: Option<Vec<f64>>,
    /// Optional background fill color.
    pub background: Option<String>,
    /// Whether to render a drop shadow (default `false`).
    #[serde(default)]
    pub shadow: bool,
    /// Optional data-space point `[x, y]` to connect to the inset bounds.
    pub connect_to: Option<[f64; 2]>,
    /// Connector style: `"lines"` (default).
    #[serde(default = "default_connect_style")]
    pub connect_style: String,
}

fn default_break_size() -> f64 { 12.0 }
fn default_break_style() -> String { "slash".to_string() }
fn default_true() -> bool { true }
fn default_border_color() -> String { "#999999".to_string() }
fn default_connect_style() -> String { "lines".to_string() }

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
    /// Explicit tick values for the axis scale.
    pub tick_values: Option<Vec<f64>>,
    /// Font size for axis title text.
    pub title_font_size: Option<f64>,
    /// Color of the axis title text (hex string).
    pub title_color: Option<String>,
    /// Padding between axis title and tick labels (pixels).
    pub title_padding: Option<f64>,
    /// d3-format string applied to tick labels.
    pub label_format_raw: Option<String>,
    /// Pixel gap between the end of a tick mark and the baseline of the tick label.
    /// Overrides the per-orient hardcoded gap in the axis renderer.
    pub label_padding: Option<f64>,
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
    /// Shape of legend symbols (e.g. `"circle"`, `"square"`, `"diamond"`).
    pub symbol_type: Option<String>,
    /// Length of a continuous gradient bar in pixels.
    pub gradient_length: Option<f64>,
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
    /// Alternating band fill colors for categorical axes (e.g. `["#f0f0f0", "transparent"]`).
    pub band_colors: Option<Vec<String>>,
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
    /// Explicit numeric domain bounds for continuous color scales.
    pub domain: Option<Vec<f64>>,
    /// Explicit hex-string color range for continuous color scales.
    pub range: Option<Vec<String>>,
}

/// Chart title configuration (controls title-level theme overrides).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct TitleConfigSpec {
    pub font_size: Option<f64>,
    pub font_weight: Option<String>,
    pub anchor: Option<String>,
    pub color: Option<String>,
    pub offset: Option<f64>,
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
        assert!(cfg.annotations.is_empty());
        assert!(cfg.structural.is_empty());
    }

    #[test]
    fn structural_secondary_y_deserializes() {
        let json = r##"{
            "structural": [
                {
                    "type": "secondary_y",
                    "field": "revenue",
                    "mark": "line",
                    "color": "#e45756",
                    "opacity": 0.7
                }
            ]
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.structural.len(), 1);
        match &cfg.structural[0] {
            StructuralSpec::SecondaryY(spec) => {
                assert_eq!(spec.field, "revenue");
                assert_eq!(spec.mark, "line");
                assert_eq!(spec.color.as_deref(), Some("#e45756"));
                assert_eq!(spec.opacity, Some(0.7));
            }
            other => panic!("expected SecondaryY, got {other:?}"),
        }
    }

    #[test]
    fn structural_break_axis_deserializes() {
        let json = r##"{
            "structural": [
                {
                    "type": "break_axis",
                    "axis": "y",
                    "gaps": [[50.0, 200.0]],
                    "break_size": 12.0,
                    "break_style": "slash"
                }
            ]
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.structural.len(), 1);
        match &cfg.structural[0] {
            StructuralSpec::BreakAxis(spec) => {
                assert_eq!(spec.axis, "y");
                assert_eq!(spec.gaps.len(), 1);
                assert_eq!(spec.gaps[0], [50.0, 200.0]);
                assert_eq!(spec.break_size, 12.0);
                assert_eq!(spec.break_style, "slash");
            }
            other => panic!("expected BreakAxis, got {other:?}"),
        }
    }

    #[test]
    fn structural_inset_deserializes() {
        let json = r##"{
            "structural": [
                {
                    "type": "inset",
                    "svg": "<svg></svg>",
                    "bounds": [0.6, 0.1, 0.95, 0.45],
                    "border": true,
                    "border_color": "#999",
                    "shadow": false
                }
            ]
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.structural.len(), 1);
        match &cfg.structural[0] {
            StructuralSpec::Inset(spec) => {
                assert_eq!(spec.bounds, [0.6, 0.1, 0.95, 0.45]);
                assert!(spec.border);
                assert!(!spec.shadow);
                assert_eq!(spec.border_color, "#999");
            }
            other => panic!("expected Inset, got {other:?}"),
        }
    }

    #[test]
    fn structural_break_axis_defaults() {
        let json = r##"{
            "structural": [
                {
                    "type": "break_axis",
                    "axis": "x",
                    "gaps": [[10.0, 50.0]]
                }
            ]
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        match &cfg.structural[0] {
            StructuralSpec::BreakAxis(spec) => {
                assert_eq!(spec.break_size, 12.0);
                assert_eq!(spec.break_style, "slash");
            }
            other => panic!("expected BreakAxis, got {other:?}"),
        }
    }

    #[test]
    fn structural_inset_defaults() {
        let json = r##"{
            "structural": [
                {
                    "type": "inset",
                    "svg": "<svg></svg>",
                    "bounds": [0.0, 0.0, 0.5, 0.5]
                }
            ]
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        match &cfg.structural[0] {
            StructuralSpec::Inset(spec) => {
                assert!(spec.border);
                assert!(!spec.shadow);
                assert_eq!(spec.border_color, "#999999");
                assert_eq!(spec.connect_style, "lines");
                assert!(spec.background.is_none());
                assert!(spec.connect_to.is_none());
            }
            other => panic!("expected Inset, got {other:?}"),
        }
    }

    #[test]
    fn annotations_deserialize_from_chart_config() {
        let json = r##"{
            "annotations": [
                {"type": "text", "x": 50.0, "y": {"norm": 0.5}, "text": "hello"},
                {"type": "line", "x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0, "stroke": "#ff0000"}
            ]
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.annotations.len(), 2);
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

    #[test]
    fn title_config_deserializes() {
        let json = r##"{
            "title": {
                "font_size": 18.0,
                "font_weight": "bold",
                "anchor": "middle",
                "color": "#333333",
                "offset": 10.0
            }
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let title = cfg.title.unwrap();
        assert_eq!(title.font_size, Some(18.0));
        assert_eq!(title.font_weight.as_deref(), Some("bold"));
        assert_eq!(title.anchor.as_deref(), Some("middle"));
        assert_eq!(title.color.as_deref(), Some("#333333"));
        assert_eq!(title.offset, Some(10.0));
    }

    #[test]
    fn title_config_absent_means_none() {
        let cfg: ChartConfig = serde_json::from_str("{}").unwrap();
        assert!(cfg.title.is_none());
    }

    #[test]
    fn axis_config_new_fields_deserialize() {
        let json = r##"{
            "axis": {
                "tick_values": [0.0, 1.0, 2.0],
                "title_font_size": 14.0,
                "title_color": "#555555",
                "title_padding": 4.0,
                "label_format_raw": ",.2f"
            }
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let axis = cfg.axis.unwrap();
        assert_eq!(axis.tick_values, Some(vec![0.0, 1.0, 2.0]));
        assert_eq!(axis.title_font_size, Some(14.0));
        assert_eq!(axis.title_color.as_deref(), Some("#555555"));
        assert_eq!(axis.title_padding, Some(4.0));
        assert_eq!(axis.label_format_raw.as_deref(), Some(",.2f"));
    }

    #[test]
    fn legend_config_new_fields_deserialize() {
        let json = r##"{
            "legend": {
                "symbol_type": "square",
                "gradient_length": 200.0
            }
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let legend = cfg.legend.unwrap();
        assert_eq!(legend.symbol_type.as_deref(), Some("square"));
        assert_eq!(legend.gradient_length, Some(200.0));
    }

    #[test]
    fn grid_config_band_colors_deserializes() {
        let json = r##"{
            "grid": {"band_colors": ["#f0f0f0", "transparent"]}
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let grid = cfg.grid.unwrap();
        assert_eq!(
            grid.band_colors.as_deref(),
            Some(["#f0f0f0".to_string(), "transparent".to_string()].as_slice())
        );
    }

    #[test]
    fn color_config_domain_and_range_deserialize() {
        let json = r##"{
            "color": {
                "domain": [0.0, 100.0],
                "range": ["#ffffff", "#000000"]
            }
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let color = cfg.color.unwrap();
        assert_eq!(color.domain, Some(vec![0.0, 100.0]));
        assert_eq!(
            color.range.as_deref(),
            Some(["#ffffff".to_string(), "#000000".to_string()].as_slice())
        );
    }
}
