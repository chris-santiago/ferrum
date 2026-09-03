//! Chart-level configuration (axis, legend, grid, padding, color, annotations).
//!
//! `ChartConfig` is the Rust mirror of the `chart_config` dict passed from
//! Python's `Chart.configure(...)`. It sits between per-channel encoding
//! overrides (highest precedence) and theme defaults (lowest precedence).
//!
//! All fields are `Option<_>` with `#[serde(default)]` so missing keys are
//! silently accepted. Unknown keys are refused, but NOT primarily by serde's
//! own `deny_unknown_fields`: `AxisConfigSpec`/`LegendConfigSpec` flatten a
//! shared style struct (see `AxisStyleSpec`'s doc), and serde's per-field
//! `deny_unknown_fields` diagnostic (naming the accepted set) does not survive
//! that flatten. The real, pinned-text gate is the wire chokepoint —
//! `chart_config_from_dict` in `binding.rs` — which validates every top-level
//! section and every section's keys against the schema-derived consts in this
//! module (`CHART_CONFIG_SECTIONS`, `AXIS_STYLE_CANONICAL_KEYS` +
//! `AXIS_STYLE_ALIAS_KEYS` + `AXIS_CONFIG_EXTRA_KEYS`,
//! `LEGEND_STYLE_CANONICAL_KEYS` + `LEGEND_STYLE_ALIAS_KEYS`,
//! `GRID_CONFIG_KEYS`, `PADDING_CONFIG_KEYS`, `COLOR_CONFIG_KEYS`,
//! `TITLE_CONFIG_KEYS`) BEFORE deserializing, refusing with `chart config:
//! unknown key '<k>' in <section>; accepted: <sorted list>` (spec
//! 2026-09-02 batch-b-config-plumbing §4.1/§6, D1). Every struct in this
//! module that CAN carry `#[serde(deny_unknown_fields)]` without a flatten
//! conflict does too (`ChartConfig` itself, `AxisConfigSpec`,
//! `LegendConfigSpec`, `GridConfigSpec`, `PaddingConfigSpec`,
//! `ColorConfigSpec`, `TitleConfigSpec`) — genuine defense in depth (and the
//! reflection source the drift tests below mine), but the wire chokepoint's
//! manual gate always runs first and owns the pinned error text.
//!
//! `chart_config_manifest.json` (this directory) is the completeness
//! instrument for NF-B11/NF-B12, and the SINGLE cross-language source for
//! it — not a Rust table mirrored by a separate Python table (that shape
//! lets a field missing from BOTH sides pass vacuously). Every serde field
//! reachable from `ChartConfig` has one entry, `{"honored": bool, "reason":
//! "..."}`: `honored: true` names the real consumer, `honored: false` names
//! why it isn't wired yet — a field with neither fails
//! `chart_config_field_disposition_manifest_is_complete` (this crate, which
//! verifies the JSON against the schema-derived consts above) AND is read by
//! `tests/test_config_manifest.py` (the Python twin, cross-checking
//! `configure.py`/`_configure_mixin.py`'s surfaces against the same file).

use serde::{Deserialize, Serialize};

use super::annotation::AnnotationSpec;

/// Top-level chart configuration passed from Python via the `chart_config` dict.
///
/// `deny_unknown_fields` is safe here: none of `ChartConfig`'s OWN fields are
/// `#[serde(flatten)]` (the flatten lives one level down, inside
/// `AxisConfigSpec`/`LegendConfigSpec`), so this struct's own accepted-field
/// enumeration is real and un-defeated — it is the reflection source the
/// `chart_config_top_level_sections_match_serde` drift test mines. The wire
/// chokepoint's manual gate (`binding.rs::validate_chart_config_keys`) still
/// owns the pinned refusal text and runs first; this is the defense-in-depth
/// second line described in the module doc.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChartConfig {
    pub axis: Option<AxisConfigSpec>,
    pub axis_x: Option<AxisConfigSpec>,
    pub axis_y: Option<AxisConfigSpec>,
    /// Applies only to the secondary y axis (an `independent_y` layer's own
    /// axis input, `AxesInput.secondary_y`) — D2/F-L07-06. Fed through the
    /// same fill-only per-axis path `axis`/`axis_x`/`axis_y` use
    /// (`apply_axis_config_to_axis_input`); a chart with no secondary y axis
    /// warns (`RenderWarning::ConfigSurfaceNotPresent`) rather than silently
    /// dropping the override.
    pub axis_y2: Option<AxisConfigSpec>,
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
/// None of `orient`, `translate`, `min_band`/`max_band`, `tick_extra`,
/// `tick_min_step`, `grid_opacity`, `title_orient`, `zindex` are orphans (a
/// prior version of this doc claimed otherwise — stale as of the 2026-09-02
/// batch-b-config-plumbing disposition audit): every one of them is merged
/// into `AxisStyleOverrides` by `axis_style_fill_from` and consumed by
/// `layout/axis.rs` (`resolve_orient`, the collision cascade, `clamp_axis_band`,
/// `build_grid`) or `render/marks/axis.rs`. See `chart_config_manifest.json` at the
/// bottom of this file for the per-field consumer citations.
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
    /// Flush the first/last rendered tick label at the axis boundary.
    /// Consumed by `render/marks/axis.rs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_flush: Option<bool>,
    /// Whether to show tick labels (`false` suppresses them). Per-channel
    /// only (`EncodingSpec.axis`, `prepare::build_axes`) — the chart-level
    /// `axis`/`axis_x`/`axis_y`/`axis_y2` position has NO consumer today:
    /// `AxisInput.show_labels` is a plain `bool` (not `Option<bool>`),
    /// exclusively owned by the per-channel prepare path (see the doc note
    /// on `apply_axis_style_to_axis_input`). See `chart_config_manifest.json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<bool>,
    // ── Ticks ────────────────────────────────────────────────────────────────
    /// Whether to show tick marks. Same per-channel-only disposition as
    /// `labels` above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ticks: Option<bool>,
    #[serde(alias = "tickCount", skip_serializing_if = "Option::is_none")]
    pub tick_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_size: Option<f64>,
    /// Append a tick at each domain boundary. Consumed by
    /// `prepare::adjust_axis_ticks`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_extra: Option<bool>,
    /// Minimum step between ticks in data space. Consumed by
    /// `prepare::adjust_axis_ticks`.
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
    /// Per-axis grid-line opacity. Consumed by `build_grid` (`layout/axis.rs`).
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
    /// contract); absent means use the field-name default. Consumed at the
    /// PER-CHANNEL position only (`prepare::resolve_axis_title`, reading
    /// `EncodingSpec.axis.title`); the chart-level `axis`/`axis_x`/`axis_y`/
    /// `axis_y2` position has NO consumer. Not a naive gap: `AxisInput.title`
    /// conflates "unset" with "per-channel explicitly suppressed" (the
    /// `title=""` -> `None` contract above), so a fill-only-if-`None`
    /// chart-level fill would resurrect an explicitly suppressed per-channel
    /// title — inverting the per-channel-wins cascade. Wiring it safely needs
    /// a tri-state model first; see `chart_config_manifest.json`. Python's
    /// `AxisConfig` dataclass does not expose a `title` parameter today
    /// either (chart-level `title` is reachable only via raw-dict callers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(alias = "titleFontSize", skip_serializing_if = "Option::is_none")]
    pub title_font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_padding: Option<f64>,
    /// Side/orientation of the axis title. Consumed by `layout/axis.rs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_orient: Option<String>,
    // ── Positioning ──────────────────────────────────────────────────────────
    /// Pixel gap between the end of a tick mark and the tick-label baseline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_padding: Option<f64>,
    /// Place the axis on the named side (top/bottom/left/right). Consumed by
    /// `AxisInput::resolve_orient`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orient: Option<String>,
    /// Shift the axis group perpendicular to its line by N px. Consumed by
    /// `layout/axis.rs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translate: Option<f64>,
    /// Lower bound for the reserved axis margin band. Consumed by
    /// `clamp_axis_band` (`layout/mod.rs`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_band: Option<f64>,
    /// Upper bound for the reserved axis margin band. Consumed by
    /// `clamp_axis_band` (`layout/mod.rs`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_band: Option<f64>,
    /// Offset of the axis from the plot area. Consumed by
    /// `render/marks/axis.rs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    /// Coarse draw order relative to marks. Consumed by `layout/axis.rs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zindex: Option<i64>,
}

/// Per-axis chart-level configuration (`configure_axis` / `axis_x` / `axis_y`
/// / `axis_y2`). Embeds [`AxisStyleSpec`] (the styling/positioning fields
/// shared with the per-channel path) and adds the **chart-only** fields that
/// are meaningless per-channel: the scale-domain controls
/// (`domain_min`/`domain_max`/`nice`/`zero` — currently unread; scale-domain
/// resolution lands in a later batch task, see `chart_config_manifest.json`) and the
/// d3-format alias `label_format_raw`.
///
/// Applied after per-channel values but before theme.
///
/// `deny_unknown_fields` on THIS struct (not on the flattened `style` field)
/// correctly rejects a bogus key even with a `#[serde(flatten)]` field
/// present — verified empirically; only `deny_unknown_fields` on the
/// flattened INNER struct is defeated by flatten (see the crate-level note on
/// [`AxisStyleSpec`]). The rejection here carries no accepted-field list
/// (serde cannot enumerate across a flatten boundary), which is exactly why
/// the wire chokepoint's manual gate — not this attribute — owns the pinned
/// refusal text; this is the reflection source `AXIS_CONFIG_EXTRA_KEYS`'
/// doc explains it can't mine from.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

    /// The format-type tag (`"time"`/`"number"`) accompanying
    /// [`effective_label_format`](Self::effective_label_format) (D8, spec
    /// §4.5). `label_format_raw` carries no sibling type field at the Python
    /// boundary (`AxisConfig` has no `label_format_raw`-paired type
    /// parameter — a raw spec's time-vs-numeric classification is decided by
    /// the `%`-containment heuristic at format-application time, matching
    /// every other raw-accepting surface), so this returns `None` whenever
    /// `label_format_raw` won the resolution above, even if `style
    /// .label_format_type` happens to be set (a caller mixing the two keys
    /// is already refused at the Python boundary as mutually exclusive).
    /// When `style.label_format` won instead, this is that field's own type
    /// — set by `AxisConfig.to_dict()`'s preset resolution
    /// (`resolve_format_field`) whenever the resolved preset carries one.
    pub fn effective_label_format_type(&self) -> Option<&str> {
        if self.label_format_raw.is_some() {
            None
        } else {
            self.style.label_format_type.as_deref()
        }
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

/// The `orient` token that means "draw no legend" rather than naming an edge.
/// Deliberately absent from [`crate::layout::LegendOrient::parse`]'s placement
/// vocabulary — it is consumed by [`LegendStyleSpec::suppressed_by`] instead.
pub(crate) const LEGEND_ORIENT_NONE: &str = "none";

impl LegendStyleSpec {
    /// Whether the legend addressed by the per-channel precedence chain
    /// `specs` is suppressed. `specs` is ordered highest-precedence first
    /// (color > x > y for the color legend); a single-channel asker — a
    /// size/shape aux block, or chart-level `configure_legend` — passes a
    /// one-element slice, which is why this is the only expression of the
    /// rule anywhere.
    ///
    /// Two spellings, one meaning: `legend=None` / `legend=False` (which
    /// Python's `_normalize_legend` turns into `disabled: true`) and
    /// `orient="none"` — the per-channel mirror of chart-level
    /// `configure_legend(orient="none")`, which `_resolve_chart_config` also
    /// resolves to `disabled` before it reaches the wire (spec §4.4:
    /// "`fm.Legend(orient="none")` disables that channel's legend (parity
    /// with chart-level)").
    ///
    /// Each spelling resolves **field by field** with the same first-`Some`
    /// rule every other per-channel legend field uses (quality review cycle 1,
    /// S3). Reading the chain with "any channel suppresses" instead let a
    /// lower-precedence `Y(legend=Legend(orient="none"))` blank a legend that
    /// `Color(legend=Legend(orient="right"))` had explicitly placed — the same
    /// `orient` field answering at two different precedences inside one
    /// function.
    pub fn suppressed_by(specs: &[&LegendStyleSpec]) -> bool {
        let disabled = specs.iter().find_map(|s| s.disabled);
        let orient = specs.iter().find_map(|s| s.orient.as_deref());
        disabled.unwrap_or(false) || orient == Some(LEGEND_ORIENT_NONE)
    }
}

/// Chart-level legend configuration (`configure_legend`). Legend has no
/// chart-only-extra fields, so this is `LegendStyleSpec` verbatim (flattened).
/// `deny_unknown_fields` here works the same way as on `AxisConfigSpec` (see
/// its doc) — real rejection, no accepted-field list in the error text.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LegendConfigSpec {
    #[serde(flatten)]
    pub style: LegendStyleSpec,
}

/// Grid configuration. No `#[serde(flatten)]` field, so `deny_unknown_fields`
/// gives a real accepted-field-list error (mined by `grid_config_keys_match_serde`).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

/// Padding configuration. No `#[serde(flatten)]` field, so `deny_unknown_fields`
/// gives a real accepted-field-list error (mined by `padding_config_keys_match_serde`).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PaddingConfigSpec {
    pub top: Option<f64>,
    pub right: Option<f64>,
    pub bottom: Option<f64>,
    pub left: Option<f64>,
    /// Reserved: intended to auto-expand margins to fit measured labels when
    /// no explicit side is set. Currently unread — `apply_chart_config`'s own
    /// comment records that it "does not disable explicit values" but does
    /// nothing else either. See `chart_config_manifest.json`.
    pub auto: Option<bool>,
}

/// Color configuration. No `#[serde(flatten)]` field, so `deny_unknown_fields`
/// gives a real accepted-field-list error (mined by `color_config_keys_match_serde`).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

/// Chart title configuration (controls title-level theme overrides). No
/// `#[serde(flatten)]` field, so `deny_unknown_fields` gives a real
/// accepted-field-list error (mined by `title_config_keys_match_serde`).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TitleConfigSpec {
    pub font_size: Option<f64>,
    pub font_weight: Option<String>,
    pub anchor: Option<String>,
    pub color: Option<String>,
    pub offset: Option<f64>,
    pub subtitle_font_size: Option<f64>,
    pub subtitle_color: Option<String>,
}

// ── Wire-key gate: schema-derived accepted-key consts (D1, spec §4.1/§6) ───
//
// Single source of truth for `binding.rs::validate_chart_config_keys`'s pinned
// refusal text AND for `chart_config_manifest.json`' completeness enumeration below.
// Each const is drift-tested against the struct it describes in `mod tests`.

/// `ChartConfig`'s own top-level field names — the accepted `chart_config`
/// dict keys. Drift-tested against `ChartConfig` itself
/// (`chart_config_top_level_sections_match_serde`).
pub(crate) const CHART_CONFIG_SECTIONS: &[&str] = &[
    "axis",
    "axis_x",
    "axis_y",
    "axis_y2",
    "legend",
    "grid",
    "padding",
    "color",
    "title",
    "annotations",
    "structural",
];

/// `AxisStyleSpec`'s canonical (non-alias) field names. Drift-tested (jointly
/// with `AXIS_STYLE_ALIAS_KEYS`) against `AxisStyleSpec`
/// (`axis_style_keys_match_serde`). Also the canonical enumeration
/// `chart_config_manifest.json` uses for this struct.
pub(crate) const AXIS_STYLE_CANONICAL_KEYS: &[&str] = &[
    "label_angle",
    "label_font_size",
    "label_color",
    "label_format",
    "label_format_type",
    "label_overlap",
    "label_flush",
    "labels",
    "ticks",
    "tick_count",
    "tick_size",
    "tick_extra",
    "tick_min_step",
    "values",
    "grid",
    "grid_color",
    "grid_dash",
    "grid_width",
    "grid_opacity",
    "domain",
    "domain_color",
    "domain_width",
    "title",
    "title_font_size",
    "title_color",
    "title_padding",
    "title_orient",
    "label_padding",
    "orient",
    "translate",
    "min_band",
    "max_band",
    "offset",
    "zindex",
];

/// `AxisStyleSpec`'s `#[serde(alias = ...)]` spellings — accepted by serde but
/// not a distinct schema field (each aliases one of
/// [`AXIS_STYLE_CANONICAL_KEYS`]). Needed in the wire gate's accepted set (a
/// raw-dict caller may legitimately use these) but NOT in the disposition
/// manifest (aliases aren't separate fields).
pub(crate) const AXIS_STYLE_ALIAS_KEYS: &[&str] = &[
    "labelAngle",
    "labelFontSize",
    "labelFormat",
    "labelFormatType",
    "tickCount",
    "titleFontSize",
    "tick_values",
];

/// `AxisConfigSpec`'s chart-only extras (never per-channel): the fields
/// declared directly on `AxisConfigSpec` alongside its flattened `style`.
/// NOT reflectively mined — `#[serde(flatten)]` defeats
/// `deny_unknown_fields`'s "expected one of" enumeration for a flattened
/// struct's OWN fields the same way it defeats the inner struct's (see the
/// doc on `AxisConfigSpec`); this is a small (5-field), stable, hand-verified
/// set instead. A genuinely new extra field needs this list — and a
/// `chart_config_manifest.json` entry — updated by whoever adds it.
pub(crate) const AXIS_CONFIG_EXTRA_KEYS: &[&str] =
    &["domain_min", "domain_max", "nice", "zero", "label_format_raw"];

/// `AxisStyleSpec` fields whose only real consumer today is the GLOBAL theme
/// path (`apply_axis_config_to_theme`): `axis`/`axis_x`/`axis_y` route
/// through it (so these ARE `honored: true` at the `AxisStyleSpec.*`
/// manifest entry), but `axis_y2` deliberately does NOT — that path writes
/// genuinely shared `ThemeInputs` fields, and routing `axis_y2` through it
/// would leak the "secondary y only" override onto the primary x/y axes'
/// theme fallback (see the doc on `ChartConfig::axis_y2`). Used only to
/// namespace this scope-specific gap as its own manifest entries
/// (`AxisConfigSpec.axis_y2.*`, spec §4.9 extended 2026-09-02) — Task 8
/// (D12) owns giving `axis_y2` its own theme-scoped consumer. `#[cfg(test)]`:
/// pure manifest-completeness instrumentation, no production reader.
#[cfg(test)]
pub(crate) const AXIS_Y2_THEME_SCOPED_CAVEAT_FIELDS: &[&str] = &["grid", "domain", "tick_size"];

/// `AxisStyleSpec` fields whose consumer (`apply_label_format_to_axis` /
/// `prepare::adjust_axis_ticks` / `sync_projected_fractions_to_tick_values`,
/// all in `render/mod.rs::prepare_and_layout`) runs on `prep.axes.x` /
/// `prep.axes.y` only — never `prep.axes.secondary_y` — so `axis_y2`'s own
/// copy of these fields reaches `AxisStyleOverrides` via the same fill-only
/// path `axis`/`axis_x`/`axis_y` use, but has no secondary-axis consumer to
/// read it (spec §4.9, extended 2026-09-02: T1's manifest sweep). Task 8
/// (D12) owns the secondary-axis consumers. `label_format_type` joined this
/// set in Task 4 (D8): it now has a real `axis`/`axis_x`/`axis_y` consumer
/// (`render::apply_axis_config_to_axis_input` →
/// `render::apply_label_format_to_axis`), which — like `label_format`
/// alongside it — runs on `prep.axes.x`/`prep.axes.y` only. `#[cfg(test)]`:
/// pure manifest-completeness instrumentation, no production reader.
#[cfg(test)]
pub(crate) const AXIS_Y2_PREP_SCOPE_CAVEAT_FIELDS: &[&str] =
    &["label_format", "label_format_type", "tick_extra", "tick_min_step", "values"];

/// `LegendStyleSpec`'s canonical (non-alias) field names, using each field's
/// WIRE spelling (`#[serde(rename = ...)]` where present: `type` for
/// `legend_type`, `tickLabels` for `tick_labels` — the Rust identifier is
/// never a valid wire key for those two). Drift-tested (jointly with
/// [`LEGEND_STYLE_ALIAS_KEYS`]) against `LegendStyleSpec`
/// (`legend_style_keys_match_serde`).
pub(crate) const LEGEND_STYLE_CANONICAL_KEYS: &[&str] = &[
    "orient",
    "direction",
    "type",
    "title",
    "title_font_size",
    "title_padding",
    "label_font_size",
    "label_color",
    "label_limit",
    "tick_count",
    "tick_min_step",
    "values",
    "format",
    "format_type",
    "symbol_size",
    "symbol_stroke_width",
    "symbol_type",
    "gradient_length",
    "gradient_thickness",
    "columns",
    "column_padding",
    "row_padding",
    "clip_height",
    "offset",
    "padding",
    "zindex",
    "disabled",
    "tickLabels",
];

/// `LegendStyleSpec`'s `#[serde(alias = ...)]` spellings.
pub(crate) const LEGEND_STYLE_ALIAS_KEYS: &[&str] =
    &["titleFontSize", "labelFontSize", "tickCount", "gradientLength", "gradientThickness"];

/// `GridConfigSpec`'s field names (no aliases). Drift-tested against
/// `GridConfigSpec` (`grid_config_keys_match_serde`).
pub(crate) const GRID_CONFIG_KEYS: &[&str] =
    &["x", "y", "color", "width", "dash", "opacity", "band_colors"];

/// `PaddingConfigSpec`'s field names (no aliases). Drift-tested against
/// `PaddingConfigSpec` (`padding_config_keys_match_serde`).
pub(crate) const PADDING_CONFIG_KEYS: &[&str] = &["top", "right", "bottom", "left", "auto"];

/// `ColorConfigSpec`'s field names (no aliases). Drift-tested against
/// `ColorConfigSpec` (`color_config_keys_match_serde`).
pub(crate) const COLOR_CONFIG_KEYS: &[&str] =
    &["scheme", "sequential_scheme", "diverging_scheme", "domain", "range"];

/// `TitleConfigSpec`'s field names (no aliases). Drift-tested against
/// `TitleConfigSpec` (`title_config_keys_match_serde`).
pub(crate) const TITLE_CONFIG_KEYS: &[&str] = &[
    "font_size",
    "font_weight",
    "anchor",
    "color",
    "offset",
    "subtitle_font_size",
    "subtitle_color",
];

/// The full accepted-key set for a `chart_config` section, or `None` when
/// `section` isn't a gated (single-object) section — `annotations`/
/// `structural` are arrays of internally-tagged variant structs, each with
/// its own field set; per-item key gating for those is out of this task's
/// scope (Grid's OWN per-axis sub-structs are Task 8's, by the same
/// carve-out). Called by `binding.rs::validate_chart_config_keys`, the wire
/// chokepoint gate — this is the single place a new section's key set is
/// wired into the gate.
pub(crate) fn accepted_keys_for_section(section: &str) -> Option<Vec<&'static str>> {
    let keys: Vec<&'static str> = match section {
        "axis" | "axis_x" | "axis_y" | "axis_y2" => AXIS_STYLE_CANONICAL_KEYS
            .iter()
            .chain(AXIS_STYLE_ALIAS_KEYS.iter())
            .chain(AXIS_CONFIG_EXTRA_KEYS.iter())
            .copied()
            .collect(),
        "legend" => LEGEND_STYLE_CANONICAL_KEYS
            .iter()
            .chain(LEGEND_STYLE_ALIAS_KEYS.iter())
            .copied()
            .collect(),
        "grid" => GRID_CONFIG_KEYS.to_vec(),
        "padding" => PADDING_CONFIG_KEYS.to_vec(),
        "color" => COLOR_CONFIG_KEYS.to_vec(),
        "title" => TITLE_CONFIG_KEYS.to_vec(),
        _ => return None,
    };
    Some(keys)
}

// ── Disposition manifest (spec §6, NF-B11/NF-B12) ──────────────────────────
//
// The checked-in `chart_config_manifest.json` (same directory) is the SINGLE
// source for every field's disposition — not a Rust const mirrored by a
// separate Python table. A prior draft of this module kept the data as a
// hand-written Rust array (`FIELD_DISPOSITIONS`); that risked exactly the
// "two owners, no independent ground truth" shape the spec's completeness
// requirement exists to prevent (a field missing from BOTH sides would pass
// a Rust-vs-Rust or Python-vs-Python check vacuously). One JSON file, two
// consumers instead: `chart_config_field_disposition_manifest_is_complete`
// below loads it and asserts its key set equals the schema-derived
// `expected` set (this crate remains the schema's source of truth — the
// JSON is verified against it, never hand-trusted); the Python twin
// (`tests/test_config_manifest.py`) reads the identical file to cross-check
// `configure.py`/`_configure_mixin.py`'s surfaces against the SAME data,
// not just against each other.
//
// Entry shape: `{"Struct.field": {"honored": bool, "reason": "..."}}`. Keys
// are namespaced `"Struct.field"` — the same field NAME recurs across
// unrelated structs with different meanings (e.g. `title` is ChartConfig's
// own section, AND AxisStyleSpec's axis-title-text field, AND
// LegendStyleSpec's legend-title field). `honored: true` names the real
// consumer in `reason`; `honored: false` names why not (and, where known,
// which follow-up task lands it). No third state: a field with neither is
// what the completeness test makes structurally impossible.
//
// Evidence for each disposition was traced against the render pipeline
// (2026-09-02, this task) by grep-following each field from its struct
// definition to its consumer (or confirming none exists). A field honored at
// one schema position but not another (e.g. `labels`/`ticks`, live
// per-channel, dead at chart level) is disposed by its MOST consequential
// gap — `honored: false` when the chart-level position (the one this
// batch's findings target) has no consumer, with the working position named
// in `reason`.
//
// `honored` is not just parsed — `chart_config_manifest_honored_flags_match_
// known_dispositions` (below) reads it against `KNOWN_UNHONORED_FIELDS` so a
// silent `honored` flip on any entry fails a test, not just `reason`'s
// non-empty check above.
#[cfg(test)]
const CHART_CONFIG_MANIFEST_JSON: &str = include_str!("chart_config_manifest.json");

/// Test-only: parse the accepted-field set out of a serde `deny_unknown_fields`
/// error produced by deserializing with ONE bogus key. serde_json renders the
/// message as `unknown field \`x\`, expected one of \`a\`, \`b\`, … \`z\` at
/// line 1 column N` — this slices after the `expected one of ` marker, drops
/// the trailing ` at line …` position suffix, then collects every
/// backtick-delimited token. Shared by every wire-schema drift test in this
/// crate (hoisted here from `binding.rs`'s prior private copy, per the
/// mirror-by-reference rule — both files are in this task's footprint):
/// `THEME_KNOWN_KEYS`'s `serde_accepted_fields_match_manifest`
/// (`binding.rs`) is the pattern this generalizes, and every `*_match_serde`
/// test below reuses it.
#[cfg(test)]
pub(crate) fn accepted_fields_from_deny_unknown_error(msg: &str) -> std::collections::BTreeSet<String> {
    const MARKER: &str = "expected one of ";
    let after = msg
        .find(MARKER)
        .map(|i| &msg[i + MARKER.len()..])
        .unwrap_or_else(|| panic!("error missing `{MARKER}` marker: {msg}"));
    let list = after.split(" at line ").next().unwrap_or(after);
    let mut fields = std::collections::BTreeSet::new();
    let mut rest = list;
    while let Some(open) = rest.find('`') {
        let tail = &rest[open + 1..];
        let close = tail.find('`').unwrap_or_else(|| panic!("unbalanced backticks in: {msg}"));
        fields.insert(tail[..close].to_string());
        rest = &tail[close + 1..];
    }
    fields
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

        // Old wire key (`min_extent`) is no longer a field. Prior to this task
        // `ChartConfig`/`AxisConfigSpec` were lenient (no `deny_unknown_fields`
        // on the flatten-containing struct itself), so a stray `min_extent`
        // key was silently swallowed rather than rejected — the exact
        // NF-B11/NF-B12 class this task fixes. `AxisConfigSpec` now carries
        // `deny_unknown_fields` on itself (works even with a flattened field
        // present — verified empirically, see its doc), so the stray key is
        // refused instead of silently ignored.
        let via_config: Result<crate::render::chart_config::ChartConfig, _> =
            serde_json::from_str(r#"{"axis_x": {"min_extent": 99.0}}"#);
        let err = via_config.expect_err("stray `min_extent` key must now be refused, not silently ignored");
        assert!(
            err.to_string().contains("min_extent"),
            "error must name the offending key: {err}"
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

    /// R1 port (bug_hunt_render_pipeline.rs): `domain` is `Vec<serde_json::Value>`
    /// specifically so categorical (string) and mixed-type domains deserialize,
    /// not just the numeric case the sibling test above covers.
    #[test]
    fn color_config_domain_accepts_string_and_mixed_values() {
        let strings: ChartConfig =
            serde_json::from_str(r#"{"color": {"domain": ["low", "medium", "high"]}}"#).unwrap();
        let domain = strings.color.unwrap().domain.unwrap();
        assert_eq!(domain, vec![Value::from("low"), Value::from("medium"), Value::from("high")]);

        let mixed: ChartConfig =
            serde_json::from_str(r#"{"color": {"domain": [0, "mid", 100]}}"#).unwrap();
        let domain = mixed.color.unwrap().domain.unwrap();
        assert!(domain[0].is_number());
        assert!(domain[1].is_string());
        assert!(domain[2].is_number());
    }

    // ── axis_y2 (D2/F-L07-06) ────────────────────────────────────────────────

    #[test]
    fn axis_y2_deserializes_like_axis_x_and_axis_y() {
        let json = r##"{"axis_y2": {"label_color": "#654321", "tick_count": 3}}"##;
        let cfg: ChartConfig = serde_json::from_str(json).unwrap();
        let axis_y2 = cfg.axis_y2.expect("axis_y2 must deserialize");
        assert_eq!(axis_y2.style.label_color.as_deref(), Some("#654321"));
        assert_eq!(axis_y2.style.tick_count, Some(3));
    }

    #[test]
    fn axis_y2_absent_by_default() {
        let cfg: ChartConfig = serde_json::from_str("{}").unwrap();
        assert!(cfg.axis_y2.is_none());
    }

    // ── Wire-key gate: const <-> schema drift tests ─────────────────────────
    //
    // Each test mines the REAL accepted-field set out of the struct's own
    // `deny_unknown_fields` error (via `accepted_fields_from_deny_unknown_error`)
    // and asserts it equals the corresponding const, so a field added to the
    // struct without updating the const (or vice versa) fails here rather
    // than silently drifting the wire gate away from the schema.

    fn mined_fields_of<T: serde::de::DeserializeOwned + std::fmt::Debug>() -> std::collections::BTreeSet<String> {
        let res: Result<T, _> =
            serde_json::from_str(r#"{"__definitely_not_a_real_field__": 1}"#);
        let err = res.expect_err("bogus key must be rejected by deny_unknown_fields");
        accepted_fields_from_deny_unknown_error(&err.to_string())
    }

    #[test]
    fn chart_config_top_level_sections_match_serde() {
        let mined = mined_fields_of::<ChartConfig>();
        let manifest: std::collections::BTreeSet<String> =
            CHART_CONFIG_SECTIONS.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            mined, manifest,
            "CHART_CONFIG_SECTIONS drifted from ChartConfig's own fields.\n\
             in serde but not const: {:?}\nin const but not serde: {:?}",
            mined.difference(&manifest).collect::<Vec<_>>(),
            manifest.difference(&mined).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn axis_style_keys_match_serde() {
        let mined = mined_fields_of::<AxisStyleSpec>();
        let manifest: std::collections::BTreeSet<String> = AXIS_STYLE_CANONICAL_KEYS
            .iter()
            .chain(AXIS_STYLE_ALIAS_KEYS.iter())
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            mined, manifest,
            "AXIS_STYLE_CANONICAL_KEYS/AXIS_STYLE_ALIAS_KEYS drifted from AxisStyleSpec.\n\
             in serde but not const: {:?}\nin const but not serde: {:?}",
            mined.difference(&manifest).collect::<Vec<_>>(),
            manifest.difference(&mined).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn legend_style_keys_match_serde() {
        let mined = mined_fields_of::<LegendStyleSpec>();
        let manifest: std::collections::BTreeSet<String> = LEGEND_STYLE_CANONICAL_KEYS
            .iter()
            .chain(LEGEND_STYLE_ALIAS_KEYS.iter())
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            mined, manifest,
            "LEGEND_STYLE_CANONICAL_KEYS/LEGEND_STYLE_ALIAS_KEYS drifted from LegendStyleSpec.\n\
             in serde but not const: {:?}\nin const but not serde: {:?}",
            mined.difference(&manifest).collect::<Vec<_>>(),
            manifest.difference(&mined).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn grid_config_keys_match_serde() {
        let mined = mined_fields_of::<GridConfigSpec>();
        let manifest: std::collections::BTreeSet<String> =
            GRID_CONFIG_KEYS.iter().map(|s| s.to_string()).collect();
        assert_eq!(mined, manifest, "GRID_CONFIG_KEYS drifted from GridConfigSpec");
    }

    #[test]
    fn padding_config_keys_match_serde() {
        let mined = mined_fields_of::<PaddingConfigSpec>();
        let manifest: std::collections::BTreeSet<String> =
            PADDING_CONFIG_KEYS.iter().map(|s| s.to_string()).collect();
        assert_eq!(mined, manifest, "PADDING_CONFIG_KEYS drifted from PaddingConfigSpec");
    }

    #[test]
    fn color_config_keys_match_serde() {
        let mined = mined_fields_of::<ColorConfigSpec>();
        let manifest: std::collections::BTreeSet<String> =
            COLOR_CONFIG_KEYS.iter().map(|s| s.to_string()).collect();
        assert_eq!(mined, manifest, "COLOR_CONFIG_KEYS drifted from ColorConfigSpec");
    }

    #[test]
    fn title_config_keys_match_serde() {
        let mined = mined_fields_of::<TitleConfigSpec>();
        let manifest: std::collections::BTreeSet<String> =
            TITLE_CONFIG_KEYS.iter().map(|s| s.to_string()).collect();
        assert_eq!(mined, manifest, "TITLE_CONFIG_KEYS drifted from TitleConfigSpec");
    }

    /// `AxisConfigSpec`'s chart-only extras deserialize correctly through the
    /// REAL struct (not a hand-rolled mirror) — the positive half of the
    /// hand-verified guarantee `AXIS_CONFIG_EXTRA_KEYS`' doc describes (the
    /// `deny_unknown_fields`-mining technique can't reach these, since
    /// `#[serde(flatten)]` defeats the "expected one of" enumeration for a
    /// flatten-containing struct's own extra fields the same way it defeats
    /// the inner struct's).
    #[test]
    fn axis_config_extra_keys_all_deserialize() {
        let json = r#"{"domain_min": 0.0, "domain_max": 100.0, "nice": true, "zero": true, "label_format_raw": ",.2f"}"#;
        let spec: AxisConfigSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.domain_min, Some(0.0));
        assert_eq!(spec.domain_max, Some(100.0));
        assert_eq!(spec.nice, Some(true));
        assert_eq!(spec.zero, Some(true));
        assert_eq!(spec.label_format_raw.as_deref(), Some(",.2f"));
    }

    // ── Disposition manifest completeness (spec §6, NF-B11/NF-B12) ──────────

    /// One field entry in `chart_config_manifest.json`: `honored` names
    /// whether the field is consumed; `reason` is required non-empty prose
    /// either way (the consumer citation, or why not — see the module doc on
    /// `CHART_CONFIG_MANIFEST_JSON`).
    #[cfg(test)]
    #[derive(serde::Deserialize)]
    struct ManifestEntry {
        honored: bool,
        reason: String,
    }

    /// Every schema field enumerated by the accepted-key consts above must
    /// have exactly one entry in `chart_config_manifest.json` — no fewer (an
    /// undispositioned field is exactly what NF-B11/NF-B12 makes structurally
    /// impossible), no extra (a stale entry for a field that no longer
    /// exists). This is the RED-provable completeness gate: add a new serde
    /// field to any struct in this module without a matching JSON entry, and
    /// this test fails. It is also the reflection half of the "one artifact,
    /// two consumers" contract: this crate remains the schema's source of
    /// truth (the JSON is checked against the schema-derived consts here,
    /// never hand-trusted), and the Python twin reads the SAME file to
    /// cross-check its own surfaces.
    #[test]
    fn chart_config_field_disposition_manifest_is_complete() {
        let manifest: std::collections::BTreeMap<String, ManifestEntry> =
            serde_json::from_str(CHART_CONFIG_MANIFEST_JSON)
                .expect("chart_config_manifest.json must parse as {field: {honored, reason}}");
        for (field, entry) in &manifest {
            assert!(
                !entry.reason.trim().is_empty(),
                "manifest entry `{field}` has an empty reason"
            );
        }
        let dispositioned: std::collections::BTreeSet<String> = manifest.keys().cloned().collect();
        assert_eq!(
            dispositioned.len(),
            manifest.len(),
            "duplicate field name in chart_config_manifest.json"
        );

        // Keys are namespaced `"Struct.field"` — the same field NAME recurs
        // across unrelated structs with different meanings and different
        // dispositions (e.g. `title` is ChartConfig's own section, AND
        // AxisStyleSpec's axis-title-text field, AND LegendStyleSpec's
        // legend-title field — three distinct entries). A bare field-name key
        // would collide across these; the namespace prefix is what keeps the
        // manifest one entry per real schema field.
        fn namespaced(prefix: &str, keys: &[&str]) -> Vec<String> {
            keys.iter().map(|k| format!("{prefix}.{k}")).collect()
        }

        // CONTRACT for whoever adds a new nested config struct (e.g. Task 8's
        // `GridConfigSpec` x/y sub-structs): this manifest-completeness gate
        // only sees a struct's fields if they are BOTH (1) reflected into
        // their own accepted-key const above (drift-tested against the real
        // struct, the same way `AXIS_STYLE_CANONICAL_KEYS`/`GRID_CONFIG_KEYS`
        // etc. are) and (2) added here as a new
        // `expected.extend(namespaced("NewStruct", NEW_STRUCT_KEYS));` line.
        // Adding the const alone is not enough — an const that exists but is
        // never namespaced into `expected` contributes zero expected keys,
        // and the new struct's fields get no manifest entry and no failing
        // test, which is exactly the parsed-but-never-read class NF-B11/B12
        // exist to make impossible (see `chart_config_manifest.json`'s own
        // module doc for the two-consumer contract this feeds).
        let mut expected: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        expected.extend(namespaced("ChartConfig", CHART_CONFIG_SECTIONS));
        expected.extend(namespaced("AxisStyleSpec", AXIS_STYLE_CANONICAL_KEYS));
        expected.extend(namespaced("AxisConfigSpec", AXIS_CONFIG_EXTRA_KEYS));
        expected.extend(namespaced("LegendStyleSpec", LEGEND_STYLE_CANONICAL_KEYS));
        expected.extend(namespaced("GridConfigSpec", GRID_CONFIG_KEYS));
        expected.extend(namespaced("PaddingConfigSpec", PADDING_CONFIG_KEYS));
        expected.extend(namespaced("ColorConfigSpec", COLOR_CONFIG_KEYS));
        expected.extend(namespaced("TitleConfigSpec", TITLE_CONFIG_KEYS));
        // axis_y2 scope caveats (spec §4.9 extended 2026-09-02): the SAME
        // `AxisStyleSpec` fields as above, but disposed separately for the
        // `axis_y2` position specifically, since that position's consumer
        // gap is real even where `axis`/`axis_x`/`axis_y`'s is not (see the
        // two consts' docs). Distinct namespace (`AxisConfigSpec.axis_y2.*`)
        // so these don't collide with the shared `AxisStyleSpec.*` entries.
        expected.extend(namespaced("AxisConfigSpec.axis_y2", AXIS_Y2_THEME_SCOPED_CAVEAT_FIELDS));
        expected.extend(namespaced("AxisConfigSpec.axis_y2", AXIS_Y2_PREP_SCOPE_CAVEAT_FIELDS));

        let missing: Vec<&String> = expected.difference(&dispositioned).collect();
        assert!(
            missing.is_empty(),
            "schema fields with no chart_config_manifest.json entry: {missing:?}"
        );
        let extra: Vec<&String> = dispositioned.difference(&expected).collect();
        assert!(
            extra.is_empty(),
            "chart_config_manifest.json entries for fields not in the schema \
             (stale after a field rename/removal): {extra:?}"
        );
    }

    /// The manifest's `honored: false` field set, exactly. Read by
    /// `chart_config_manifest_honored_flags_match_known_dispositions` below —
    /// a silent `honored` flip on ANY manifest entry (a currently-honored
    /// field flipping to `false` gains a member not in this set; a
    /// currently-unhonored field flipping to `true` drops a member this set
    /// still expects) fails that test by naming the offending field. This is
    /// what makes `ManifestEntry::honored` load-bearing instead of a
    /// parsed-but-never-read struct field: the completeness test above only
    /// ever reads `entry.reason`, so `honored` needs its own consumer, not a
    /// wider net cast over `reason`.
    #[cfg(test)]
    const KNOWN_UNHONORED_FIELDS: &[&str] = &[
        "AxisConfigSpec.axis_y2.domain",
        "AxisConfigSpec.axis_y2.grid",
        "AxisConfigSpec.axis_y2.label_format",
        "AxisConfigSpec.axis_y2.label_format_type",
        "AxisConfigSpec.axis_y2.tick_extra",
        "AxisConfigSpec.axis_y2.tick_min_step",
        "AxisConfigSpec.axis_y2.tick_size",
        "AxisConfigSpec.axis_y2.values",
        "AxisConfigSpec.domain_max",
        "AxisConfigSpec.domain_min",
        "AxisConfigSpec.nice",
        "AxisConfigSpec.zero",
        "AxisStyleSpec.labels",
        "AxisStyleSpec.tick_count",
        "AxisStyleSpec.ticks",
        "AxisStyleSpec.title",
        "PaddingConfigSpec.auto",
    ];

    /// Gives `ManifestEntry::honored` teeth (spec review, T1 cycle 2): reads
    /// the flag out of every manifest entry and asserts the `honored: false`
    /// subset equals `KNOWN_UNHONORED_FIELDS` exactly. RED-provable the same
    /// way as the completeness test: flip any entry's `honored` value in
    /// `chart_config_manifest.json` without updating this const, and this
    /// test fails naming the field.
    #[test]
    fn chart_config_manifest_honored_flags_match_known_dispositions() {
        let manifest: std::collections::BTreeMap<String, ManifestEntry> =
            serde_json::from_str(CHART_CONFIG_MANIFEST_JSON)
                .expect("chart_config_manifest.json must parse as {field: {honored, reason}}");
        let unhonored: std::collections::BTreeSet<String> = manifest
            .iter()
            .filter(|(_, entry)| !entry.honored)
            .map(|(field, _)| field.clone())
            .collect();
        let known: std::collections::BTreeSet<String> =
            KNOWN_UNHONORED_FIELDS.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            unhonored, known,
            "chart_config_manifest.json's `honored: false` field set drifted from \
             KNOWN_UNHONORED_FIELDS — a field's `honored` flag changed without this \
             const being updated (the exact silent-flip class this test exists to catch).\n\
             newly false, not in const: {:?}\n\
             in const but now honored true (or entry removed): {:?}",
            unhonored.difference(&known).collect::<Vec<_>>(),
            known.difference(&unhonored).collect::<Vec<_>>(),
        );
    }

    /// `AxisConfigSpec`'s chart-only extras, reflected via `Serialize` rather
    /// than hand-verified (spec §6: an unlisted field must fail a test, not
    /// just documentation). Reflective mining (`mined_fields_of`, used by
    /// every other `*_match_serde` test above) can't reach these: `#[serde(
    /// flatten)]` defeats `deny_unknown_fields`'s "expected one of"
    /// enumeration for a flatten-containing struct's OWN fields the same way
    /// it defeats the inner struct's (see `AxisConfigSpec`'s doc).
    ///
    /// Reflection instead: a FULLY explicit struct literal (no `..Default`
    /// shorthand for `AxisConfigSpec`'s own fields) serialized to JSON. Two
    /// independent guarantees fall out of the same literal: (1) a field
    /// added to `AxisConfigSpec` without updating this literal fails to
    /// COMPILE ("missing field in initializer"), not just fails an
    /// assertion; (2) every own field here has no `skip_serializing_if` (see
    /// the struct definition), so it always appears in the output, while
    /// `style`'s flattened fields all DO have `skip_serializing_if =
    /// "Option::is_none"` and a `Default` style has none set — so the
    /// serialized key set is exactly `AxisConfigSpec`'s own extras, never
    /// polluted by the flatten.
    #[test]
    fn axis_config_extra_keys_match_serde() {
        let spec = AxisConfigSpec {
            style: AxisStyleSpec::default(),
            domain_min: Some(0.0),
            domain_max: Some(100.0),
            nice: Some(true),
            zero: Some(true),
            label_format_raw: Some(",.2f".to_string()),
        };
        let value = serde_json::to_value(&spec).expect("AxisConfigSpec must serialize");
        let mined: std::collections::BTreeSet<String> = value
            .as_object()
            .expect("AxisConfigSpec serializes to a JSON object")
            .keys()
            .cloned()
            .collect();
        let expected: std::collections::BTreeSet<String> =
            AXIS_CONFIG_EXTRA_KEYS.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            mined, expected,
            "AXIS_CONFIG_EXTRA_KEYS drifted from AxisConfigSpec's own (non-flattened) fields.\n\
             in serde but not const: {:?}\nin const but not serde: {:?}",
            mined.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&mined).collect::<Vec<_>>(),
        );
    }

    /// `accepted_keys_for_section` covers every gated ChartConfig section and
    /// returns `None` only for the two array sections (out of this task's
    /// per-item-gating scope) — derived from `CHART_CONFIG_SECTIONS` itself
    /// (spec review, cycle 2) rather than a hand-copied list, so a future
    /// section added to `ChartConfig` without a matching
    /// `accepted_keys_for_section` arm fails THIS test instead of silently
    /// falling into the ungated `annotations`/`structural` bucket.
    #[test]
    fn accepted_keys_for_section_covers_every_gated_section() {
        const ARRAY_SECTIONS: &[&str] = &["annotations", "structural"];
        for &section in CHART_CONFIG_SECTIONS {
            let is_array_section = ARRAY_SECTIONS.contains(&section);
            assert_eq!(
                accepted_keys_for_section(section).is_some(),
                !is_array_section,
                "section `{section}`: accepted_keys_for_section must return {} (array sections \
                 are exactly {ARRAY_SECTIONS:?})",
                if is_array_section { "None" } else { "Some(...)" }
            );
        }
        assert!(accepted_keys_for_section("not_a_real_section").is_none());
    }
}
