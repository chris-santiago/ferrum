//! Chart-level configuration (axis, legend, grid, padding, color, annotations).
//!
//! `ChartConfig` is the Rust mirror of the `chart_config` dict passed from
//! Python's `Chart.configure(...)`. It sits between per-channel encoding
//! overrides (highest precedence) and theme defaults (lowest precedence).
//!
//! All fields are `Option<_>` with `#[serde(default)]` so missing keys are
//! silently accepted; unknown keys produce a serde error for fast feedback.

use serde::{Deserialize, Serialize};

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
    /// Structural features: axis breaks, inset charts.
    #[serde(default)]
    pub structural: Vec<StructuralSpec>,
}

// ── Structural feature specs ────────────────────────────────────────────────

/// One structural feature descriptor, deserialized from the `structural` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuralSpec {
    BreakAxis(BreakAxisSpec),
    Inset(InsetSpec),
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

/// Shared axis **styling + positioning** fields, mirroring the snake_case keys
/// `fm.Axis.to_dict()` emits. This is the single schema used by BOTH the
/// per-channel `EncodingSpec.axis` (directly) and the chart-level
/// [`AxisConfigSpec`] (via `#[serde(flatten)]`). Factoring it out (B5 fix,
/// 2026-06-14) guarantees per-channel and chart-level honor the same field set,
/// closing the silent-drop gap where ~22 advertised `fm.Axis` fields reached the
/// per-channel path as an opaque map and were never read.
///
/// `#[serde(deny_unknown_fields)]` makes a misspelled per-channel key fail loud
/// (a serde error surfaced as `ValueError`) instead of dropping silently. Note:
/// serde does not enforce `deny_unknown_fields` through a `#[serde(flatten)]`
/// container, so the chart-level `AxisConfigSpec` (which flattens this) keeps its
/// historical lenient behavior; the deny only bites on the standalone
/// per-channel `EncodingSpec.axis` path, which is exactly where fail-loud matters.
///
/// Camel-case `#[serde(alias = ...)]`es preserve back-compat with raw-dict
/// callers and the keys the old `prepare.rs` hand-reader accepted.
///
/// Some fields are **orphans** (no renderer honors them yet) — `orient`,
/// `translate`, `min_band`/`max_band`, `tick_extra`, `tick_min_step`,
/// `grid_opacity`, `title_orient`, `zindex`. They deserialize and round-trip now;
/// their render support lands in later units.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AxisStyleSpec {
    // ── Tick labels ──────────────────────────────────────────────────────────
    #[serde(alias = "labelAngle", skip_serializing_if = "Option::is_none")]
    pub label_angle: Option<f64>,
    #[serde(alias = "labelFontSize", skip_serializing_if = "Option::is_none")]
    pub label_font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_color: Option<String>,
    /// d3-format string for tick labels. The chart-level path calls its sibling
    /// key `label_format_raw`; the two are reconciled in
    /// [`AxisConfigSpec::effective_label_format`].
    #[serde(alias = "labelFormat", skip_serializing_if = "Option::is_none")]
    pub label_format: Option<String>,
    #[serde(alias = "labelFormatType", skip_serializing_if = "Option::is_none")]
    pub label_format_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_overlap: Option<String>,
    /// Orphan (no renderer yet): flush labels at axis boundaries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_flush: Option<bool>,
    /// Whether to show tick labels (`false` suppresses them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<bool>,
    // ── Ticks ────────────────────────────────────────────────────────────────
    /// Whether to show tick marks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ticks: Option<bool>,
    #[serde(alias = "tickCount", skip_serializing_if = "Option::is_none")]
    pub tick_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_size: Option<f64>,
    /// Orphan: append a tick at each domain boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_extra: Option<bool>,
    /// Orphan: minimum step between ticks in data space.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_min_step: Option<f64>,
    /// Explicit tick values. The per-channel `fm.Axis` spelling is `values`; the
    /// chart-level `AxisConfig` spelling is `tick_values` — both map here.
    #[serde(alias = "tick_values", skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    // ── Grid ─────────────────────────────────────────────────────────────────
    /// Whether to show gridlines for this axis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_dash: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_width: Option<f64>,
    /// Orphan: per-axis grid-line opacity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_opacity: Option<f64>,
    // ── Domain line ──────────────────────────────────────────────────────────
    /// Whether to show the axis domain line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_width: Option<f64>,
    // ── Title ────────────────────────────────────────────────────────────────
    /// Axis title text. `Some("")` suppresses the title (the `title=None`
    /// contract); absent means use the field-name default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(alias = "titleFontSize", skip_serializing_if = "Option::is_none")]
    pub title_font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_padding: Option<f64>,
    /// Orphan: side/orientation of the axis title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_orient: Option<String>,
    // ── Positioning ──────────────────────────────────────────────────────────
    /// Pixel gap between the end of a tick mark and the tick-label baseline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_padding: Option<f64>,
    /// Orphan: place the axis on the named side (top/bottom/left/right).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orient: Option<String>,
    /// Orphan: shift the axis group perpendicular to its line by N px.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translate: Option<f64>,
    /// Orphan: lower bound for the reserved axis margin band.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_band: Option<f64>,
    /// Orphan: upper bound for the reserved axis margin band.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_band: Option<f64>,
    /// Orphan: offset of the axis from the plot area.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    /// Orphan: coarse draw order relative to marks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zindex: Option<i64>,
}

/// Per-axis chart-level configuration (`configure_axis` / `axis_x` / `axis_y`).
/// Embeds [`AxisStyleSpec`] (the styling/positioning fields shared with the
/// per-channel path) and adds the **chart-only** fields that are meaningless
/// per-channel: the scale-domain controls (`domain_min`/`domain_max`/`nice`/
/// `zero`) and the d3-format alias `label_format_raw`.
///
/// Applied after per-channel values but before theme.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct AxisConfigSpec {
    #[serde(flatten)]
    pub style: AxisStyleSpec,
    // ── Chart-only scale-domain fields (never per-channel) ───────────────────
    pub domain_min: Option<f64>,
    pub domain_max: Option<f64>,
    pub nice: Option<bool>,
    pub zero: Option<bool>,
    /// d3-format string applied to tick labels. Chart-level callers historically
    /// used this name; the per-channel path uses `label_format`. Both are honored
    /// via [`AxisConfigSpec::effective_label_format`] (`label_format_raw` wins,
    /// then `label_format`).
    pub label_format_raw: Option<String>,
}

impl AxisConfigSpec {
    /// Reconcile the two d3-format key names: chart callers pass
    /// `label_format_raw`, the per-channel struct carries `label_format`. The
    /// raw key wins when both are set (it is the chart-level-specific spelling).
    pub fn effective_label_format(&self) -> Option<&str> {
        self.label_format_raw
            .as_deref()
            .or(self.style.label_format.as_deref())
    }
}

/// Shared legend **styling + positioning** fields, mirroring the snake_case keys
/// `fm.Legend.to_dict()` emits, plus the `disabled` suppression key that
/// `_normalize_legend` (`legend=None`/`False`) produces and the internal
/// `tickLabels` key used by SHAP beeswarm charts.
///
/// Single schema for BOTH per-channel `EncodingSpec.legend` (directly, with
/// fail-loud `deny_unknown_fields`) and chart-level [`LegendConfigSpec`] (via
/// `flatten`). Camel-case aliases preserve the keys the old `prepare.rs` D13
/// reader accepted (`titleFontSize`, `labelFontSize`, `gradientLength`,
/// `gradientThickness`, `tickCount`).
///
/// Formerly-orphan fields (`clip_height`, `row_padding`, `column_padding`,
/// `symbol_stroke_width`, `label_limit`, `tick_min_step`, `zindex`) now render
/// at both chart and per-channel level (B5 unit 3). `label_limit` truncates with
/// an ellipsis, `clip_height` hard-clips via an SVG `clipPath`, and `zindex`
/// maps to coarse below/above-marks ordering (legends sit outside the plot, so
/// it is usually a visual no-op).
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LegendStyleSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orient: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    /// Legend kind override (`"symbol"`/`"gradient"`). The old reader keyed this
    /// as `type`; `fm.Legend` also emits `type`.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub legend_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(alias = "titleFontSize", skip_serializing_if = "Option::is_none")]
    pub title_font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_padding: Option<f64>,
    #[serde(alias = "labelFontSize", skip_serializing_if = "Option::is_none")]
    pub label_font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_color: Option<String>,
    /// Maximum legend-label pixel width; labels wider than this are truncated
    /// with an ellipsis (B5 unit 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_limit: Option<f64>,
    #[serde(alias = "tickCount", skip_serializing_if = "Option::is_none")]
    pub tick_count: Option<u32>,
    /// Minimum step between colorbar ticks in data units (B5 unit 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_min_step: Option<f64>,
    /// Explicit tick/entry labels (`fm.Legend(values=[...])`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<serde_json::Value>>,
    /// d3-format string for colorbar tick labels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_size: Option<f64>,
    /// Stroke width (px) of legend symbol swatches (B5 unit 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_stroke_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_type: Option<String>,
    #[serde(alias = "gradientLength", skip_serializing_if = "Option::is_none")]
    pub gradient_length: Option<f64>,
    #[serde(alias = "gradientThickness", skip_serializing_if = "Option::is_none")]
    pub gradient_thickness: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub columns: Option<u32>,
    /// Horizontal entry spacing (px) for horizontal-direction legends (B5 unit 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column_padding: Option<f64>,
    /// Vertical entry spacing (px) for vertical-direction legends (B5 unit 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_padding: Option<f64>,
    /// Cap on the legend group height (px); overflow is hard-clipped (B5 unit 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_height: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub padding: Option<f64>,
    /// Coarse draw order relative to marks (B5 unit 3): `>= 1` → above, `<= 0`
    /// (default) → below. Legends sit outside the plot, so usually a no-op.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zindex: Option<i64>,
    /// Suppress the legend entirely. Produced by `_normalize_legend` for
    /// `legend=None` / `legend=False`; not a user-facing `fm.Legend` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disabled: Option<bool>,
    /// Internal: explicit colorbar tick labels (e.g. SHAP beeswarm `["Low",
    /// "High"]`). Not a user-facing `fm.Legend` field.
    #[serde(rename = "tickLabels", skip_serializing_if = "Option::is_none")]
    pub tick_labels: Option<Vec<String>>,
}

/// Chart-level legend configuration (`configure_legend`). Legend has no
/// chart-only-extra fields, so this is `LegendStyleSpec` verbatim (flattened).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct LegendConfigSpec {
    #[serde(flatten)]
    pub style: LegendStyleSpec,
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
    /// Explicit domain for color scales. Numeric values for continuous scales,
    /// string values for categorical scales.
    pub domain: Option<Vec<serde_json::Value>>,
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
    pub subtitle_font_size: Option<f64>,
    pub subtitle_color: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

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
        assert_eq!(axis.style.label_angle, Some(-45.0));
        assert_eq!(axis.style.grid, Some(true));
        assert!(axis.style.label_font_size.is_none());
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
        assert_eq!(cfg.axis.as_ref().unwrap().style.label_angle, Some(-30.0));
        assert_eq!(cfg.axis_x.as_ref().unwrap().domain_min, Some(0.0));
        assert_eq!(cfg.axis_x.as_ref().unwrap().domain_max, Some(100.0));
        assert_eq!(cfg.axis_y.as_ref().unwrap().zero, Some(true));
        assert_eq!(cfg.legend.as_ref().unwrap().style.orient.as_deref(), Some("bottom"));
        assert_eq!(cfg.legend.as_ref().unwrap().style.columns, Some(3));
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
        // Chart-level `AxisConfig` emits `tick_values` (aliased to `values`) and
        // the chart-only `label_format_raw`.
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
        assert_eq!(axis.style.values, Some(vec![0.0, 1.0, 2.0]));
        assert_eq!(axis.style.title_font_size, Some(14.0));
        assert_eq!(axis.style.title_color.as_deref(), Some("#555555"));
        assert_eq!(axis.style.title_padding, Some(4.0));
        assert_eq!(axis.label_format_raw.as_deref(), Some(",.2f"));
        assert_eq!(axis.effective_label_format(), Some(",.2f"));
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
        assert_eq!(legend.style.symbol_type.as_deref(), Some("square"));
        assert_eq!(legend.style.gradient_length, Some(200.0));
    }

    /// B5 unit 6a: the six residual render fields deserialize from the legend
    /// JSON contract Python emits.
    #[test]
    fn legend_config_6a_render_fields_deserialize() {
        let json = r##"{
            "legend": {
                "symbol_size": 400.0,
                "label_color": "#ff0000",
                "offset": 50.0,
                "padding": 30.0,
                "title_padding": 25.0,
                "column_padding": 40.0
            }
        }"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let s = cfg.legend.unwrap().style;
        assert_eq!(s.symbol_size, Some(400.0));
        assert_eq!(s.label_color.as_deref(), Some("#ff0000"));
        assert_eq!(s.offset, Some(50.0));
        assert_eq!(s.padding, Some(30.0));
        assert_eq!(s.title_padding, Some(25.0));
        assert_eq!(s.column_padding, Some(40.0));
    }

    #[test]
    fn axis_style_deny_unknown_fields_rejects_typo() {
        // Standalone per-channel deserialization (not via the flatten container)
        // must reject an unknown key — the B5 fail-loud guard.
        let bad: Result<AxisStyleSpec, _> = serde_json::from_str(r##"{"grid_colr":"#f00"}"##);
        assert!(bad.is_err(), "unknown axis style key must fail to deserialize");
        let good: Result<AxisStyleSpec, _> = serde_json::from_str(r##"{"grid_color":"#f00"}"##);
        assert!(good.is_ok());
    }

    #[test]
    fn axis_style_camel_case_alias_deserializes() {
        let s: AxisStyleSpec = serde_json::from_str(r#"{"labelAngle":-30,"tickCount":4}"#).unwrap();
        assert_eq!(s.label_angle, Some(-30.0));
        assert_eq!(s.tick_count, Some(4));
    }

    #[test]
    fn legend_style_deny_unknown_fields_rejects_typo() {
        let bad: Result<LegendStyleSpec, _> =
            serde_json::from_str(r##"{"symbol_sze":10}"##);
        assert!(bad.is_err(), "unknown legend style key must fail to deserialize");
    }

    #[test]
    fn axis_style_carries_orphan_fields() {
        // Orphan fields deserialize and round-trip even though no renderer honors
        // them yet (their render lands in later units).
        let s: AxisStyleSpec = serde_json::from_str(
            r#"{"orient":"bottom","translate":3.0,"min_band":10.0,"max_band":40.0,"zindex":1}"#,
        )
        .unwrap();
        assert_eq!(s.orient.as_deref(), Some("bottom"));
        assert_eq!(s.translate, Some(3.0));
        assert_eq!(s.min_band, Some(10.0));
        assert_eq!(s.max_band, Some(40.0));
        assert_eq!(s.zindex, Some(1));
        let back = serde_json::to_string(&s).unwrap();
        let reparsed: AxisStyleSpec = serde_json::from_str(&back).unwrap();
        assert_eq!(reparsed, s);
    }

    /// Wire-contract regression for D-EXTENT-1: the serde key is now `min_band`/
    /// `max_band`. A stray `min_extent` key must be silently ignored (the struct
    /// uses `deny_unknown_fields` only on the standalone per-channel path; the
    /// chart-level flatten path is lenient). A `min_extent` key must NOT populate
    /// `min_band`.
    #[test]
    fn axis_style_min_band_max_band_wire_keys() {
        // New wire key is accepted.
        let s: AxisStyleSpec =
            serde_json::from_str(r#"{"min_band":10.0,"max_band":40.0}"#).unwrap();
        assert_eq!(s.min_band, Some(10.0));
        assert_eq!(s.max_band, Some(40.0));

        // Old wire key (`min_extent`) is no longer a field — a stray key is
        // silently ignored under the lenient `#[serde(default)]` path used by
        // ChartConfig (via flatten).  We verify it does NOT populate `min_band`.
        let via_config: crate::render::chart_config::ChartConfig =
            serde_json::from_str(r#"{"axis_x": {"min_extent": 99.0}}"#).unwrap();
        let axis_x = via_config.axis_x.unwrap_or_default();
        assert_eq!(
            axis_x.style.min_band, None,
            "stray `min_extent` key must not populate `min_band`"
        );
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
        assert_eq!(color.domain, Some(vec![Value::from(0.0), Value::from(100.0)]));
        assert_eq!(
            color.range.as_deref(),
            Some(["#ffffff".to_string(), "#000000".to_string()].as_slice())
        );
    }
}
