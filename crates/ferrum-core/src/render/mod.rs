//! Phase 7 — static renderer. Pure functions: ChartSpec + RecordBatch + ThemeInputs +
//! Viewport -> deterministic SVG/PNG. See docs/superpowers/specs/2026-05-09-static-renderer-design.md.

pub(crate) mod annotation;
pub(crate) mod arrow_cast;
pub(crate) mod chart_config;
pub(crate) mod config;
pub(crate) mod color;
pub(crate) mod palette;
pub(crate) mod font;
pub(crate) mod format;
pub(crate) mod svg;
pub(crate) mod embed_font;
pub(crate) mod scale_resolve;
pub(crate) mod prepare;
pub(crate) mod rasterize;
pub(crate) mod draw;
pub(crate) mod png;
pub(crate) mod binding;
pub(crate) mod marks;
pub(crate) mod position;
pub mod compositor;
pub(crate) mod grid_compose;
pub(crate) mod pack_instances;
pub(crate) mod scene_build;
pub(crate) mod svg_walk;
pub(crate) mod secondary_axis;
pub(crate) mod break_axis;
pub(crate) mod inset;
pub use compositor::{
    compose_svg_horizontal, compose_svg_vertical, CompositorError, HorizontalAlign, VerticalAlign,
};

// Constants (spec §6.1).
pub const FLOAT_PRECISION: usize = 3;
pub const CLIP_ID_PREFIX: &str = "ferrum-clip-";

use serde::{Deserialize, Serialize};

use crate::layout::LayoutWarning;

#[derive(Debug, Clone, PartialEq)]
pub enum RenderError {
    InvalidViewport { width: f64, height: f64 },
    EmptyBatch,
    UnknownColumn { name: String },
    InvalidColor(String),
    EncodingTypeMismatch { channel: &'static str, expected: &'static str, got: String },
    TransformFailed(String),
    ScaleResolutionFailed(String),
    LayoutFailed(String),
    ResvgFailed(String),
    /// A position-adjustment pass (Dodge/Jitter/Stack) rejected its inputs.
    /// `adjustment` names the adjustment; `reason` is the user-facing detail.
    PositionAdjustFailed { adjustment: &'static str, reason: String },
    /// A column carried an Arrow dtype the renderer cannot interpret.
    /// `field` is the column name; `context` is an optional channel /
    /// scale tag (e.g. `"size"`, `"opacity"`, `"scale"`) used to
    /// disambiguate when the same column feeds multiple resolution
    /// passes. Display: `"column '<field>' has unsupported dtype: <dtype>"`
    /// or `"<context>: column '<field>' has unsupported dtype: <dtype>"`.
    UnsupportedDtype { field: String, dtype: String, context: Option<&'static str> },
    /// The unioned numeric/temporal extent for an axis or color channel
    /// produced no finite values (all rows null/NaN or empty after filter).
    EmptyDomain { channel: String, field: String },
    SceneConstruction(String),
    HtmlBundleAssembly(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidViewport { width, height } =>
                write!(f, "invalid viewport: width={width}, height={height} (both must be > 0)"),
            Self::EmptyBatch =>
                write!(f, "input batch is empty (num_rows == 0)"),
            Self::UnknownColumn { name } =>
                write!(f, "unknown column '{name}' referenced by an encoding"),
            Self::InvalidColor(s) =>
                write!(f, "invalid color string: '{s}' (expected #rrggbb or #rrggbbaa)"),
            Self::EncodingTypeMismatch { channel, expected, got } =>
                write!(f, "encoding '{channel}' expected {expected}, got {got}"),
            Self::TransformFailed(s) =>
                write!(f, "transform failed: {s}"),
            Self::ScaleResolutionFailed(s) =>
                write!(f, "scale resolution failed: {s}"),
            Self::LayoutFailed(s) =>
                write!(f, "layout failed: {s}"),
            Self::ResvgFailed(s) =>
                write!(f, "PNG rasterization failed: {s}"),
            Self::PositionAdjustFailed { adjustment, reason } =>
                write!(f, "{adjustment}: {reason}"),
            Self::UnsupportedDtype { field, dtype, context } => match context {
                Some(ctx) => write!(f, "{ctx}: column '{field}' has unsupported dtype: {dtype}"),
                None => write!(f, "column '{field}' has unsupported dtype: {dtype}"),
            },
            Self::EmptyDomain { channel, field } =>
                write!(f, "{channel}: no usable values found for field '{field}'"),
            Self::SceneConstruction(msg) =>
                write!(f, "scene construction failed: {msg}"),
            Self::HtmlBundleAssembly(msg) =>
                write!(f, "HTML bundle assembly failed: {msg}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Warnings emitted during render. Geometric edge cases or wrapped layout warnings.
///
/// Note (2026-05-09): spec §11 (line ~556) used `#[serde(tag = "kind", ...)]` but
/// that collides with `LayoutWarning`'s own `kind` tag when wrapped via
/// `RenderWarning::Layout(LayoutWarning)` (serde flattens newtype-around-struct
/// variants). Outer tag renamed to `type` to disambiguate; `LayoutWarning`'s
/// `kind` tag is preserved (already pinned by Phase 6 round-trip tests).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RenderWarning {
    Layout(LayoutWarning),
    OutOfDomainRows { mark: String, count: u64 },
    ColorPaletteOverflowed { categories: u32 },
    ShapePaletteOverflowed { categories: u32 },
    EmptyPanel { panel_index: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderOutput<T> {
    pub bytes: T,
    pub layout: crate::layout::LayoutResult,
    pub warnings: Vec<RenderWarning>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_config_default_values() {
        let c = config::RenderConfig::default();
        assert_eq!(c.scale, 2.0);
        assert!(c.embed_fonts);
        assert!(c.background.is_none());
        assert!(c.width.is_none());
        assert!(c.height.is_none());
    }

    #[test]
    fn render_warning_round_trip_each_variant() {
        use crate::layout::LayoutWarning;
        for w in [
            RenderWarning::Layout(LayoutWarning::PanelCollapsed { panel_index: 0 }),
            RenderWarning::OutOfDomainRows { mark: "point".into(), count: 3 },
            RenderWarning::ColorPaletteOverflowed { categories: 12 },
            RenderWarning::ShapePaletteOverflowed { categories: 7 },
            RenderWarning::EmptyPanel { panel_index: 1 },
        ] {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: RenderWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w);
        }
    }

    #[test]
    fn render_error_display_messages_are_meaningful() {
        let err = RenderError::InvalidViewport { width: 0.0, height: 100.0 };
        let msg = format!("{err}");
        assert!(msg.contains("invalid viewport"), "msg: {msg}");
        assert!(msg.contains("0"), "msg: {msg}");

        let err = RenderError::UnknownColumn { name: "missing".into() };
        let msg = format!("{err}");
        assert!(msg.contains("unknown column"), "msg: {msg}");
        assert!(msg.contains("missing"), "msg: {msg}");
    }
}

// ---------------------------------------------------------------------------
// Task 20 — render_svg full pipeline orchestration (spec §6).
// ---------------------------------------------------------------------------

use crate::layout::{compute_layout, LegendDirection, LegendOrient, LegendOverrides, TextAnchor, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;
use chart_config::{AxisConfigSpec, ChartConfig};

/// Apply [`ChartConfig`] overrides to a [`ThemeInputs`] clone.
///
/// This implements "configure > theme" precedence (level 3 > level 4–5). It is
/// called in both `render_svg` and `render_scene_json` after the per-encoding
/// legend overrides have been merged into `effective_theme` but before
/// `compute_layout` and `build_scene` are invoked.
///
/// Per-channel `axis=Axis(...)` overrides live in `AxisInput` and are resolved
/// by `prepare_render_inputs` — they take effect at layout time (level 2) and
/// are never touched here.
fn apply_chart_config(theme: &mut ThemeInputs, config: &ChartConfig) {
    // ── Grid overrides ────────────────────────────────────────────────────────
    if let Some(ref grid) = config.grid {
        if let Some(enabled) = grid.x {
            // x-grid is controlled globally via theme.grid; a dedicated x-only
            // flag does not exist in ThemeInputs. When both x and y are present
            // we only flip the global flag when they agree.
            if grid.y.unwrap_or(enabled) == enabled {
                theme.grid = enabled;
            }
        } else if let Some(enabled) = grid.y {
            theme.grid = enabled;
        }
        if let Some(ref color) = grid.color {
            if let Ok(c) = color::from_hex_str(color) {
                theme.grid_color = c;
            }
        }
        if let Some(w) = grid.width {
            theme.grid_width = w;
        }
        if let Some(ref d) = grid.dash {
            theme.grid_dash = Some(d.clone());
        }
        if let Some(o) = grid.opacity {
            theme.grid_opacity = o;
        }
    }

    // ── Padding overrides ─────────────────────────────────────────────────────
    // Per-side padding: map each supplied side directly to ThemeInputs.padding_*.
    // `auto=true` (the Python default) is no longer a guard — it was previously
    // blocking all padding overrides. The `auto` field is reserved for a future
    // "auto-expand to fit labels" semantic; it does not disable explicit values.
    if let Some(ref padding) = config.padding {
        if let Some(top) = padding.top {
            theme.padding_top = Some(top);
        }
        if let Some(right) = padding.right {
            theme.padding_right = Some(right);
        }
        if let Some(bottom) = padding.bottom {
            theme.padding_bottom = Some(bottom);
        }
        if let Some(left) = padding.left {
            theme.padding_left = Some(left);
        }
    }

    // ── Legend overrides ──────────────────────────────────────────────────────
    if let Some(ref legend) = config.legend {
        if let Some(ref orient) = legend.orient {
            theme.legend_orient = match orient.as_str() {
                "right"  => LegendOrient::Right,
                "left"   => LegendOrient::Left,
                "top"    => LegendOrient::Top,
                "bottom" => LegendOrient::Bottom,
                _ => theme.legend_orient,
            };
        }
        if let Some(ref dir) = legend.direction {
            theme.legend_direction = match dir.as_str() {
                "vertical"   => Some(LegendDirection::Vertical),
                "horizontal" => Some(LegendDirection::Horizontal),
                _ => theme.legend_direction,
            };
        }
        if let Some(cols) = legend.columns {
            theme.legend_columns = Some(cols);
        }
        if let Some(fs) = legend.title_font_size {
            theme.legend_title_font_size = fs;
        }
        // legend.label_font_size maps to the shared theme.label_font_size,
        // which controls both axis and legend label sizing.
        if let Some(fs) = legend.label_font_size {
            theme.label_font_size = fs;
        }
    }

    // ── Color scheme overrides ────────────────────────────────────────────────
    if let Some(ref color) = config.color {
        if let Some(ref scheme) = color.scheme {
            theme.color_scheme = scheme.clone();
        }
        if let Some(ref seq) = color.sequential_scheme {
            theme.sequential_scheme = seq.clone();
        }
        if let Some(ref div) = color.diverging_scheme {
            theme.diverging_scheme = div.clone();
        }
    }

    // ── Axis overrides (applied to both axes simultaneously) ──────────────────
    if let Some(ref axis) = config.axis {
        apply_axis_config_to_theme(theme, axis);
    }
    // Per-axis overrides run after the combined override so axis_x / axis_y
    // wins over axis when both are specified.
    if let Some(ref axis_x) = config.axis_x {
        apply_axis_x_config_to_theme(theme, axis_x);
    }
    if let Some(ref axis_y) = config.axis_y {
        apply_axis_y_config_to_theme(theme, axis_y);
    }

    // ── Title overrides ───────────────────────────────────────────────────────
    if let Some(ref title) = config.title {
        if let Some(fs) = title.font_size {
            theme.title_font_size = fs;
        }
        if let Some(ref fw) = title.font_weight {
            theme.title_font_weight = fw.clone();
        }
        if let Some(ref anchor) = title.anchor {
            theme.title_anchor = match anchor.as_str() {
                "middle" => TextAnchor::Middle,
                "end"    => TextAnchor::End,
                _        => TextAnchor::Start,
            };
        }
        if let Some(ref c) = title.color {
            if let Ok(parsed) = color::from_hex_str(c) {
                theme.title_color = parsed;
            }
        }
        if let Some(o) = title.offset {
            theme.title_offset = o;
        }
    }
}

/// Apply axis config fields that affect shared theme state (both axes).
///
/// Fields like `label_font_size`, `tick_size`, and grid color are global in
/// `ThemeInputs` and are applied here regardless of x/y axis distinction.
/// Per-axis-specific overrides (x vs y) are handled in the sibling functions.
fn apply_axis_config_to_theme(theme: &mut ThemeInputs, axis: &AxisConfigSpec) {
    if let Some(fs) = axis.label_font_size {
        theme.label_font_size = fs;
    }
    if let Some(ref c) = axis.label_color {
        if let Ok(parsed) = color::from_hex_str(c) {
            theme.label_color = parsed;
        }
    }
    if let Some(ts) = axis.tick_size {
        theme.tick_size = ts;
    }
    if let Some(enabled) = axis.domain {
        theme.axis_line = enabled;
    }
    if let Some(ref c) = axis.domain_color {
        if let Ok(parsed) = color::from_hex_str(c) {
            theme.axis_line_color = parsed;
        }
    }
    if let Some(w) = axis.domain_width {
        theme.axis_line_width = w;
    }
    if let Some(enabled) = axis.grid {
        theme.grid = enabled;
    }
    if let Some(ref c) = axis.grid_color {
        if let Ok(parsed) = color::from_hex_str(c) {
            theme.grid_color = parsed;
        }
    }
    if let Some(ref d) = axis.grid_dash {
        theme.grid_dash = Some(d.clone());
    }
    if let Some(w) = axis.grid_width {
        theme.grid_width = w;
    }
}

/// Apply `axis_x`-specific config fields that have per-axis theme equivalents.
///
/// Currently `ThemeInputs` has no separate x/y axis theme fields, so x-specific
/// overrides that conflict with y settings cannot be expressed at this level.
/// When axis_x and axis_y specify the same field with different values, the last
/// one applied wins (axis_y runs after axis_x in the caller).
fn apply_axis_x_config_to_theme(theme: &mut ThemeInputs, axis: &AxisConfigSpec) {
    // x-axis-specific overrides reuse the same shared theme fields.
    // Per-channel `axis=Axis(...)` on the x encoding remains the highest-priority
    // override and is handled separately in `prepare_render_inputs`.
    apply_axis_config_to_theme(theme, axis);
}

/// Apply `axis_y`-specific config fields that have per-axis theme equivalents.
fn apply_axis_y_config_to_theme(theme: &mut ThemeInputs, axis: &AxisConfigSpec) {
    apply_axis_config_to_theme(theme, axis);
}

/// Build a [`LegendOverrides`] from a [`prepare::PreparedInputs`].
fn legend_overrides_from_prep(prep: &prepare::PreparedInputs) -> LegendOverrides {
    LegendOverrides {
        tick_count:         prep.legend_tick_count_override,
        label_font_size:    prep.legend_label_font_size_override,
        gradient_length:    prep.legend_gradient_length_override,
        gradient_thickness: prep.legend_gradient_thickness_override,
        direction:          prep.legend_direction_override,
        values:             prep.legend_values_override.clone(),
        legend_type:        prep.legend_type_override.clone(),
        symbol_type:        None, // filled by apply_chart_config_to_legend_overrides
    }
}

/// Apply `ChartConfig.legend` fields to a `LegendOverrides`.
///
/// Per-encoding `legend=Legend(...)` overrides (level 2) are already in the
/// `LegendOverrides` built by `legend_overrides_from_prep`. This function
/// fills in the `configure_legend(...)` overrides (level 3) only when the
/// per-encoding value is absent (level 2 wins over level 3).
fn apply_chart_config_to_legend_overrides(
    overrides: &mut LegendOverrides,
    config: &ChartConfig,
) {
    let Some(ref legend) = config.legend else { return };
    if overrides.gradient_length.is_none() {
        overrides.gradient_length = legend.gradient_length;
    }
    if overrides.symbol_type.is_none() {
        overrides.symbol_type = legend.symbol_type.clone();
    }
}

/// Apply `ChartConfig.axis` / `axis_x` / `axis_y` fields that are not expressed
/// in `ThemeInputs` to the per-axis `AxisInput`.
///
/// These are the per-axis fields that cannot be represented in the shared theme:
/// `title_font_size`, `title_color`, `title_padding`, `label_format_raw`,
/// and `tick_values` (stored for layout-time use).
///
/// Called with `config_both` (the `axis` key that applies to both x and y),
/// then with `config_specific` (the `axis_x` or `axis_y` key). The specific
/// key wins because it runs last.
fn apply_axis_config_to_axis_input(
    axis: &mut crate::layout::AxisInput,
    config: Option<&chart_config::AxisConfigSpec>,
) {
    let Some(cfg) = config else { return };
    // label_angle_override: per-encoding Axis() wins (already set); only fill if not set.
    if axis.label_angle_override.is_none() {
        axis.label_angle_override = cfg.label_angle;
    }
    // label_format_raw: d3-format string for tick labels.
    if axis.label_format_override.is_none() {
        axis.label_format_override = cfg.label_format_raw.clone()
            .or_else(|| cfg.label_format.clone());
    }
    // tick_values: explicit tick positions in data space.
    if axis.tick_values_override.is_none() {
        axis.tick_values_override = cfg.tick_values.clone();
    }
    // title_font_size / title_color / title_padding.
    if axis.title_font_size.is_none() {
        axis.title_font_size = cfg.title_font_size;
    }
    if axis.title_color.is_none() {
        if let Some(ref c) = cfg.title_color {
            if let Ok(parsed) = color::from_hex_str(c) {
                axis.title_color = Some(parsed);
            }
        }
    }
    if axis.title_padding.is_none() {
        axis.title_padding = cfg.title_padding;
    }
    if axis.label_padding.is_none() {
        axis.label_padding = cfg.label_padding;
    }
}

/// Re-format tick label strings using a d3-format string override.
///
/// When `tick_values_override` is set, tick_labels are replaced entirely
/// with formatted versions of the explicit tick values. If a
/// `label_format_override` is also provided the values are formatted using
/// that d3-format spec; otherwise they are converted to plain decimal strings.
///
/// When only `label_format_override` is set (no explicit tick values), each
/// existing tick label is parsed as a float and reformatted via the d3-format
/// spec. Non-numeric labels (category names, time strings) are passed through
/// unchanged.
fn apply_label_format_to_axis(axis: &mut crate::layout::AxisInput) {
    if let Some(tick_vals) = axis.tick_values_override.clone() {
        // Replace tick_labels with formatted versions of the explicit tick_values.
        let numeric_strings: Vec<String> = tick_vals.iter().map(|v| v.to_string()).collect();
        let fmt = axis.label_format_override.as_deref();
        axis.tick_labels = prepare::apply_tick_format(numeric_strings, fmt, None);
    } else if let Some(ref fmt_str) = axis.label_format_override.clone() {
        // Re-format existing labels using the d3-format spec.
        axis.tick_labels = prepare::apply_tick_format(
            std::mem::take(&mut axis.tick_labels),
            Some(fmt_str),
            None,
        );
    }
}

/// Apply `ChartConfig.color.domain` and `color.range` overrides to the
/// resolved color scale. `pub(crate)` so `scene_build` can call it after
/// per-panel scale resolution (which re-resolves the scale independently of
/// the provisional_scales copy patched in `render_svg`).
///
/// Per-encoding `Color(scale=...)` overrides (level 2) have already been
/// applied during scale resolution. This function applies `configure_color(...)`
/// overrides (level 3) only when the per-encoding value is absent (level 2
/// wins over level 3).
///
/// Continuous scales: `domain` pins the data extent; `range` builds an evenly
/// spaced gradient from the provided hex stops.
///
/// Categorical scales: `range` replaces the palette with a heap-allocated
/// `Cow::Owned` slice of parsed hex colors. Invalid hex strings are silently
/// skipped; the override is a no-op when no valid colors are parsed.
pub(crate) fn apply_color_config_to_color_scale(
    color_scale: &mut Option<scale_resolve::ColorScale>,
    cfg: &chart_config::ColorConfigSpec,
) {
    let Some(ref mut scale) = color_scale else { return };
    match scale {
        scale_resolve::ColorScale::Continuous { ref mut domain, ref mut scheme } => {
            if let Some(ref d) = cfg.domain {
                let floats: Vec<f64> = d.iter().filter_map(|v| v.as_f64()).collect();
                if floats.len() == 2 {
                    *domain = (floats[0], floats[1]);
                }
            }
            if let Some(ref r) = cfg.range {
                if r.len() >= 2 {
                    let parsed: Vec<color::Color> = r
                        .iter()
                        .filter_map(|s| color::from_hex_str(s).ok())
                        .collect();
                    if parsed.len() >= 2 {
                        // Build a gradient from the parsed stops, evenly spaced.
                        let stops: Vec<(f64, color::Color)> = parsed
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| {
                                let t = i as f64 / (parsed.len() - 1) as f64;
                                (t, c)
                            })
                            .collect();
                        *scheme = color::ContinuousScheme::Gradient(stops);
                    }
                }
            }
        }
        scale_resolve::ColorScale::Categorical { ref mut palette, .. } => {
            if let Some(ref range) = cfg.range {
                let colors: Vec<color::Color> = range
                    .iter()
                    .filter_map(|hex| color::from_hex_str(hex).ok())
                    .collect();
                if !colors.is_empty() {
                    *palette = std::borrow::Cow::Owned(colors);
                }
            }
        }
    }
}

pub fn render_svg(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<RenderOutput<String>, RenderError> {
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(RenderError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }

    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };

    let mut prep = prepare::prepare_render_inputs(spec, batch, theme)?;
    let mut warnings = prep.warnings.clone();

    // Apply ChartConfig axis overrides (level 3) to AxisInput (level 2 wins when already set).
    // Order: shared `axis` key first, then axis-specific keys (axis_x / axis_y) win.
    apply_axis_config_to_axis_input(&mut prep.axes.x, chart_config.axis.as_ref());
    apply_axis_config_to_axis_input(&mut prep.axes.y, chart_config.axis.as_ref());
    apply_axis_config_to_axis_input(&mut prep.axes.x, chart_config.axis_x.as_ref());
    apply_axis_config_to_axis_input(&mut prep.axes.y, chart_config.axis_y.as_ref());
    // Apply label_format_override to tick labels (requires axis config to be set first).
    apply_label_format_to_axis(&mut prep.axes.x);
    apply_label_format_to_axis(&mut prep.axes.y);
    // Apply color domain/range overrides (level 3) to resolved color scale.
    if let Some(ref cfg) = chart_config.color {
        apply_color_config_to_color_scale(&mut prep.provisional_scales.color, cfg);
    }

    // Build effective_theme: start from the caller-supplied theme, then layer in
    // overrides from lowest to highest priority within the "render" concern:
    //   1. D13 per-encoding legend overrides (from encoding.color.legend.*).
    //   2. ChartConfig overrides (configure_*() calls, level 3 > theme level 4-5).
    // Per-channel axis overrides (level 2) are in AxisInput and applied at layout time.
    let mut effective_theme = theme.clone();
    // D13: per-encoding legend overrides.
    if let Some(orient) = prep.legend_orient_override {
        effective_theme.legend_orient = orient;
    }
    if let Some(fs) = prep.legend_title_font_size_override {
        effective_theme.legend_title_font_size = fs;
    }
    if let Some(cols) = prep.legend_columns_override {
        effective_theme.legend_columns = Some(cols);
    }
    // ChartConfig overrides (configure > theme).
    apply_chart_config(&mut effective_theme, chart_config);

    // Reserve right-side padding for secondary Y axis labels when present.
    for structural in &chart_config.structural {
        if let chart_config::StructuralSpec::SecondaryY(y2_spec) = structural {
            use crate::layout::TextMetrics;
            let m = font::FontdueMetrics::new();
            let y2_label_width = if let Ok(vals) = crate::render::arrow_cast::col_as_f64(prep.final_batch(), &y2_spec.field) {
                let finite: Vec<f64> = vals.into_iter().filter_map(|v| v.filter(|f| f.is_finite())).collect();
                if let (Some(&lo), Some(&hi)) = (finite.iter().min_by(|a,b| a.partial_cmp(b).unwrap()), finite.iter().max_by(|a,b| a.partial_cmp(b).unwrap())) {
                    let step = (hi - lo) / 5.0;
                    (0..=5).map(|i| {
                        let v = lo + step * i as f64;
                        let label = crate::render::format::format_numeric(v);
                        m.measure_width(&label, effective_theme.label_font_size)
                    }).fold(0.0_f64, f64::max)
                } else { 30.0 }
            } else { 30.0 };
            let needed = y2_label_width + effective_theme.tick_size + 8.0;
            let current = effective_theme.padding_right.unwrap_or(effective_theme.padding);
            if current < needed {
                effective_theme.padding_right = Some(needed);
            }
            break;
        }
    }

    let theme_ref = &effective_theme;

    // D13: legend title override (replaces the default field-name title when Some).
    let effective_legend_title = prep
        .legend_title_override
        .clone()
        .or_else(|| prep.legend_title.clone());

    let mut legend_overrides = legend_overrides_from_prep(&prep);
    // Apply configure_legend overrides (level 3) — only fills in None fields.
    apply_chart_config_to_legend_overrides(&mut legend_overrides, chart_config);
    let metrics = font::FontdueMetrics::new();
    let layout = compute_layout(
        spec,
        theme_ref,
        viewport,
        &prep.axes,
        &prep.facet_groups,
        &prep.legend_entries,
        effective_legend_title,
        prep.colorbar.as_ref(),
        &metrics,
        &legend_overrides,
    )
    .map_err(|e| RenderError::LayoutFailed(e.to_string()))?;
    for w in &layout.warnings {
        warnings.push(RenderWarning::Layout(w.clone()));
    }

    let scene = scene_build::build_scene(
        spec, &prep, &layout, theme_ref, config, &mut warnings, chart_config,
    )?;
    let svg_string = svg_walk::walk_svg(&scene, config.embed_fonts);

    Ok(RenderOutput { bytes: svg_string, layout, warnings })
}

pub fn render_png(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<RenderOutput<Vec<u8>>, RenderError> {
    let svg_out = render_svg(spec, batch, theme, viewport, config, chart_config)?;
    let w = (svg_out.layout.viewport.w * config.scale).round() as u32;
    let h = (svg_out.layout.viewport.h * config.scale).round() as u32;
    let bytes = png::svg_string_to_png_bytes(&svg_out.bytes, w, h, config.scale)?;
    Ok(RenderOutput { bytes, layout: svg_out.layout, warnings: svg_out.warnings })
}

pub fn render_scene_json(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
    chart_config: &ChartConfig,
) -> Result<(String, Vec<u8>), RenderError> {
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(RenderError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }

    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };

    let mut prep = prepare::prepare_render_inputs(spec, batch, theme)?;
    let mut warnings = prep.warnings.clone();

    // Apply ChartConfig axis overrides (level 3) — same logic as render_svg.
    apply_axis_config_to_axis_input(&mut prep.axes.x, chart_config.axis.as_ref());
    apply_axis_config_to_axis_input(&mut prep.axes.y, chart_config.axis.as_ref());
    apply_axis_config_to_axis_input(&mut prep.axes.x, chart_config.axis_x.as_ref());
    apply_axis_config_to_axis_input(&mut prep.axes.y, chart_config.axis_y.as_ref());
    apply_label_format_to_axis(&mut prep.axes.x);
    apply_label_format_to_axis(&mut prep.axes.y);
    if let Some(ref cfg) = chart_config.color {
        apply_color_config_to_color_scale(&mut prep.provisional_scales.color, cfg);
    }

    // Build effective_theme with the same layered override logic as render_svg.
    let mut effective_theme = theme.clone();
    if let Some(orient) = prep.legend_orient_override {
        effective_theme.legend_orient = orient;
    }
    if let Some(fs) = prep.legend_title_font_size_override {
        effective_theme.legend_title_font_size = fs;
    }
    if let Some(cols) = prep.legend_columns_override {
        effective_theme.legend_columns = Some(cols);
    }
    apply_chart_config(&mut effective_theme, chart_config);
    let theme_ref = &effective_theme;

    let effective_legend_title = prep
        .legend_title_override
        .clone()
        .or_else(|| prep.legend_title.clone());

    let mut legend_overrides = legend_overrides_from_prep(&prep);
    apply_chart_config_to_legend_overrides(&mut legend_overrides, chart_config);
    let metrics = font::FontdueMetrics::new();
    let layout = compute_layout(
        spec,
        theme_ref,
        viewport,
        &prep.axes,
        &prep.facet_groups,
        &prep.legend_entries,
        effective_legend_title,
        prep.colorbar.as_ref(),
        &metrics,
        &legend_overrides,
    )
    .map_err(|e| RenderError::LayoutFailed(e.to_string()))?;
    for w in &layout.warnings {
        warnings.push(RenderWarning::Layout(w.clone()));
    }

    let mut scene = scene_build::build_scene(
        spec, &prep, &layout, theme_ref, config, &mut warnings, chart_config,
    )?;

    // Extract large homogeneous mark batches as raw packed bytes, clearing
    // their nodes from the scene graph. The JSON stays lightweight; the
    // packed bytes travel as a separate Uint8Array to the WASM renderer.
    let packed_bytes = pack_instances::extract_packed_bytes(&mut scene);

    let json = serde_json::to_string(&scene)
        .map_err(|e| RenderError::LayoutFailed(format!("scene serialization: {e}")))?;
    Ok((json, packed_bytes))
}

pub(crate) fn filter_batch_by_facet(
    batch: &RecordBatch,
    field: &str,
    value: &str,
) -> Result<RecordBatch, RenderError> {
    use arrow::array::{Array, BooleanArray, StringArray};
    use arrow::compute::filter_record_batch;
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            RenderError::ScaleResolutionFailed(format!("facet field '{field}' must be Utf8"))
        })?;
    let mask: BooleanArray = arr
        .iter()
        .map(|v| Some(v.map(|s| s == value).unwrap_or(false)))
        .collect();
    filter_record_batch(batch, &mask)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("filter: {e}")))
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn scatter_3() -> (ChartSpec, RecordBatch) {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    #[test]
    fn render_svg_minimal_scatter() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let svg = result.bytes;
        assert!(svg.starts_with("<svg "));
        assert!(svg.ends_with("</svg>"));
        assert_eq!(svg.matches("<circle ").count(), 3);
        assert!(svg.contains("@font-face"));
    }

    #[test]
    fn render_svg_invalid_viewport_errors() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let result = render_svg(
            &spec,
            &batch,
            &theme,
            Viewport { width: 0.0, height: 100.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        );
        assert!(matches!(result.unwrap_err(), RenderError::InvalidViewport { .. }));
    }

    #[test]
    fn render_svg_unknown_column_errors() {
        let (mut spec, batch) = scatter_3();
        spec.encoding.x = Some(EncodingSpec { field: "missing".into(), type_: None, ..Default::default() });
        let result = render_svg(
            &spec,
            &batch,
            &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        );
        assert!(matches!(result.unwrap_err(), RenderError::UnknownColumn { .. }));
    }

    #[test]
    fn render_svg_faceted_emits_strip_titles() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "c", "b", "c"])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let result = render_svg(
            &spec,
            &batch,
            &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        )
        .unwrap();
        let svg = result.bytes;
        assert!(svg.contains(">a<") || svg.contains(">a</text>"));
        assert!(svg.contains(">b<") || svg.contains(">b</text>"));
        assert!(svg.contains(">c<") || svg.contains(">c</text>"));
    }

    #[test]
    fn render_svg_determinism_two_calls_byte_identical() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let a = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let b = render_svg(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }

    #[test]
    fn scene_graph_path_matches_old_path_scatter() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let cfg = config::RenderConfig::default();
        let old_svg = render_svg(&spec, &batch, &theme, viewport, &cfg, &ChartConfig::default()).unwrap().bytes;

        let prep = prepare::prepare_render_inputs(&spec, &batch, &theme).unwrap();
        let mut warnings = prep.warnings.clone();

        let mut effective_theme = theme.clone();
        if let Some(orient) = prep.legend_orient_override {
            effective_theme.legend_orient = orient;
        }
        if let Some(fs) = prep.legend_title_font_size_override {
            effective_theme.legend_title_font_size = fs;
        }
        if let Some(cols) = prep.legend_columns_override {
            effective_theme.legend_columns = Some(cols);
        }
        apply_chart_config(&mut effective_theme, &ChartConfig::default());
        let theme_ref = &effective_theme;
        let effective_legend_title = prep
            .legend_title_override
            .clone()
            .or_else(|| prep.legend_title.clone());

        let legend_overrides = legend_overrides_from_prep(&prep);
        let metrics = font::FontdueMetrics::new();
        let vp2 = Viewport {
            width: cfg.width.unwrap_or(viewport.width),
            height: cfg.height.unwrap_or(viewport.height),
        };
        let layout = compute_layout(
            &spec, theme_ref, vp2,
            &prep.axes, &prep.facet_groups, &prep.legend_entries,
            effective_legend_title, prep.colorbar.as_ref(), &metrics,
            &legend_overrides,
        ).unwrap();
        for w in &layout.warnings {
            warnings.push(RenderWarning::Layout(w.clone()));
        }

        let scene = scene_build::build_scene(
            &spec, &prep, &layout, theme_ref, &cfg, &mut warnings,
            &ChartConfig::default(),
        ).unwrap();
        let new_svg = svg_walk::walk_svg(&scene, cfg.embed_fonts);

        if old_svg != new_svg {
            let old_chars: Vec<char> = old_svg.chars().collect();
            let new_chars: Vec<char> = new_svg.chars().collect();
            let first_diff = old_chars.iter().zip(new_chars.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(old_chars.len().min(new_chars.len()));
            let context_start = first_diff.saturating_sub(80);
            let context_end = (first_diff + 80).min(old_svg.len()).min(new_svg.len());
            panic!(
                "Scene graph SVG differs from old path at byte {}.\n\
                 OLD[{}..{}]: {:?}\n\
                 NEW[{}..{}]: {:?}\n\
                 old len={}, new len={}",
                first_diff,
                context_start, context_end, &old_svg[context_start..context_end.min(old_svg.len())],
                context_start, context_end, &new_svg[context_start..context_end.min(new_svg.len())],
                old_svg.len(), new_svg.len(),
            );
        }
    }
}

#[cfg(test)]
mod png_tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn render_png_produces_png_magic_bytes() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        let result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 100.0, height: 80.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        assert_eq!(&result.bytes[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    #[test]
    fn render_png_determinism_two_calls_byte_identical() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 100.0, height: 80.0 };
        let config = config::RenderConfig::default();
        let a = render_png(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        let b = render_png(&spec, &batch, &theme, viewport, &config, &ChartConfig::default()).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }
}

#[cfg(test)]
mod golden_tests {
    //! End-to-end goldens. Refresh via `FERRUM_UPDATE_GOLDENS=1 cargo test`.
    //! See spec §9.4 for refresh discipline.

    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn check_golden(name: &str, svg: &str) {
        let path = format!("tests/golden/{name}.svg");
        let abs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&path);
        if std::env::var("FERRUM_UPDATE_GOLDENS").is_ok() {
            std::fs::create_dir_all(abs_path.parent().unwrap()).unwrap();
            std::fs::write(&abs_path, svg).expect("write golden");
            return;
        }
        let expected = std::fs::read_to_string(&abs_path)
            .unwrap_or_else(|e| panic!("read golden {path}: {e} — run FERRUM_UPDATE_GOLDENS=1 to create"));
        assert_eq!(svg, expected, "golden mismatch for {name} — run FERRUM_UPDATE_GOLDENS=1 to refresh");
    }

    fn check_png_hash(name: &str, png: &[u8]) {
        use sha2::Digest;
        use std::io::Write;
        let path = format!("tests/golden/{name}.sha256");
        let abs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&path);
        let mut hasher = sha2::Sha256::new();
        hasher.update(png);
        let hash = format!("{:x}", hasher.finalize());
        if std::env::var("FERRUM_UPDATE_GOLDENS").is_ok() {
            std::fs::create_dir_all(abs_path.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&abs_path).unwrap();
            f.write_all(hash.as_bytes()).unwrap();
            return;
        }
        let expected = std::fs::read_to_string(&abs_path)
            .unwrap_or_else(|e| panic!("read png hash {path}: {e}"));
        assert_eq!(hash.trim(), expected.trim(), "PNG hash mismatch for {name}");
    }

    #[test]
    fn scatter_minimal_golden() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("scatter_minimal", &result.bytes);

        let png_result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_png_hash("scatter_minimal.png", &png_result.bytes);
    }

    #[test]
    fn scatter_color_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            Arc::new(StringArray::from(vec!["a","b","c","a","b","c"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("scatter_color", &result.bytes);
    }

    #[test]
    fn bar_grouped_golden() {
        use crate::spec::encoding::DataType as SDT;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","b","c","d"])),
            Arc::new(Float64Array::from(vec![3.0, 1.0, 4.0, 1.5])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("bar_grouped", &result.bytes);
    }

    #[test]
    fn line_simple_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0, 30.0, 80.0, 60.0])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("line_simple", &result.bytes);
    }

    #[test]
    fn area_filled_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0, 30.0, 80.0, 60.0])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("area_filled", &result.bytes);
    }

    #[test]
    fn faceted_scatter_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0])),
            Arc::new(StringArray::from(vec!["setosa","setosa","setosa","versicolor","versicolor","versicolor","virginica","virginica","virginica"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                row: None,
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
            &ChartConfig::default(),
        ).unwrap();
        check_golden("faceted_scatter", &result.bytes);
    }
}

#[cfg(test)]
mod chart_config_application_tests {
    //! Unit tests for `apply_chart_config` — verify that ChartConfig overrides
    //! are correctly applied to ThemeInputs.

    use super::*;
    use chart_config::{
        AxisConfigSpec, ChartConfig, ColorConfigSpec, GridConfigSpec, LegendConfigSpec,
        PaddingConfigSpec, TitleConfigSpec,
    };

    #[test]
    fn apply_chart_config_noop_on_empty_config() {
        let default_theme = ThemeInputs::default();
        let mut theme = default_theme.clone();
        apply_chart_config(&mut theme, &ChartConfig::default());
        // Empty config must not change anything.
        assert_eq!(theme.grid, default_theme.grid);
        assert_eq!(theme.grid_width, default_theme.grid_width);
        assert_eq!(theme.grid_color, default_theme.grid_color);
        assert_eq!(theme.padding, default_theme.padding);
        assert_eq!(theme.legend_orient, default_theme.legend_orient);
        assert_eq!(theme.color_scheme, default_theme.color_scheme);
        assert_eq!(theme.label_font_size, default_theme.label_font_size);
        assert_eq!(theme.title_font_size, default_theme.title_font_size);
    }

    #[test]
    fn apply_chart_config_grid_color_and_width() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                color: Some("#ff0000".to_string()),
                width: Some(2.0),
                opacity: Some(0.5),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.grid_color, color::from_hex_str("#ff0000").unwrap());
        assert_eq!(theme.grid_width, 2.0);
        assert_eq!(theme.grid_opacity, 0.5);
    }

    #[test]
    fn apply_chart_config_grid_disabled_via_grid_config() {
        let mut theme = ThemeInputs::default();
        assert!(theme.grid); // default is on
        let config = ChartConfig {
            grid: Some(GridConfigSpec { x: Some(false), y: Some(false), ..Default::default() }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert!(!theme.grid);
    }

    #[test]
    fn apply_chart_config_grid_enabled_via_axis_config() {
        let mut theme = ThemeInputs::default();
        theme.grid = false;
        let config = ChartConfig {
            axis: Some(AxisConfigSpec { grid: Some(true), ..Default::default() }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert!(theme.grid);
    }

    #[test]
    fn apply_chart_config_padding_per_side() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            padding: Some(PaddingConfigSpec {
                top: Some(10.0),
                right: Some(20.0),
                bottom: Some(30.0),
                left: Some(40.0),
                auto: None,
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        // Each side is set independently.
        assert_eq!(theme.padding_top, Some(10.0));
        assert_eq!(theme.padding_right, Some(20.0));
        assert_eq!(theme.padding_bottom, Some(30.0));
        assert_eq!(theme.padding_left, Some(40.0));
        // Uniform fallback padding is unchanged.
        assert_eq!(theme.padding, 16.0);
    }

    #[test]
    fn apply_chart_config_padding_auto_does_not_block_explicit_sides() {
        // auto=true (the Python default) must NOT block explicit side values.
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            padding: Some(PaddingConfigSpec {
                top: Some(5.0),
                auto: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        // The explicit top value must be applied even when auto=true.
        assert_eq!(theme.padding_top, Some(5.0));
        // Sides not specified remain None.
        assert!(theme.padding_right.is_none());
        assert!(theme.padding_bottom.is_none());
        assert!(theme.padding_left.is_none());
    }

    #[test]
    fn apply_chart_config_legend_orient_and_direction() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            legend: Some(LegendConfigSpec {
                orient: Some("bottom".to_string()),
                direction: Some("horizontal".to_string()),
                columns: Some(4),
                title_font_size: Some(16.0),
                label_font_size: Some(9.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.legend_orient, LegendOrient::Bottom);
        assert_eq!(theme.legend_direction, Some(LegendDirection::Horizontal));
        assert_eq!(theme.legend_columns, Some(4));
        assert_eq!(theme.legend_title_font_size, 16.0);
        assert_eq!(theme.label_font_size, 9.0);
    }

    #[test]
    fn apply_chart_config_color_scheme_override() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            color: Some(ColorConfigSpec {
                scheme: Some("tableau10".to_string()),
                sequential_scheme: Some("viridis".to_string()),
                diverging_scheme: Some("rdbu".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.color_scheme, "tableau10");
        assert_eq!(theme.sequential_scheme, "viridis");
        assert_eq!(theme.diverging_scheme, "rdbu");
    }

    #[test]
    fn apply_chart_config_axis_label_font_size() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec { label_font_size: Some(14.0), ..Default::default() }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.label_font_size, 14.0);
    }

    #[test]
    fn apply_chart_config_axis_tick_size_and_domain_visibility() {
        let mut theme = ThemeInputs::default();
        assert!(theme.axis_line); // default on
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                tick_size: Some(6.0),
                domain: Some(false),
                domain_width: Some(2.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.tick_size, 6.0);
        assert!(!theme.axis_line);
        assert_eq!(theme.axis_line_width, 2.0);
    }

    #[test]
    fn apply_chart_config_title_overrides() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            title: Some(TitleConfigSpec {
                font_size: Some(22.0),
                font_weight: Some("700".to_string()),
                anchor: Some("end".to_string()),
                color: Some("#123456".to_string()),
                offset: Some(8.0),
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.title_font_size, 22.0);
        assert_eq!(theme.title_font_weight, "700");
        assert_eq!(theme.title_anchor, TextAnchor::End);
        assert_eq!(theme.title_color, color::from_hex_str("#123456").unwrap());
        assert_eq!(theme.title_offset, 8.0);
    }

    #[test]
    fn apply_chart_config_invalid_hex_color_silently_ignored() {
        let mut theme = ThemeInputs::default();
        let original_grid_color = theme.grid_color;
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                color: Some("not-a-hex-color".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        // Bad color must not change the existing value.
        assert_eq!(theme.grid_color, original_grid_color);
    }

    #[test]
    fn apply_chart_config_axis_x_wins_over_axis_for_same_field() {
        // When both `axis` and `axis_x` set label_font_size, axis_x wins
        // (applied last). This is the documented behavior for per-axis overrides.
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec { label_font_size: Some(10.0), ..Default::default() }),
            axis_x: Some(AxisConfigSpec { label_font_size: Some(14.0), ..Default::default() }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.label_font_size, 14.0);
    }

    #[test]
    fn apply_chart_config_grid_dash() {
        let mut theme = ThemeInputs::default();
        assert!(theme.grid_dash.is_none());
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                dash: Some(vec![4.0, 4.0]),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config(&mut theme, &config);
        assert_eq!(theme.grid_dash, Some(vec![4.0, 4.0]));
    }

    // ── New wired fields (configure-punchlist, 2026-05-24) ───────────────────

    #[test]
    fn axis_config_title_font_size_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            Some("X Axis".to_string()),
            vec!["0".to_string(), "50".to_string(), "100".to_string()],
            None,
        );
        let cfg = AxisConfigSpec { title_font_size: Some(16.0), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        assert_eq!(axis.title_font_size, Some(16.0));
    }

    #[test]
    fn axis_config_title_color_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Left,
            Some("Y".to_string()),
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec { title_color: Some("#ff0000".to_string()), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        let c = axis.title_color.expect("title_color should be Some");
        assert_eq!(c.red, 0xff);
        assert_eq!(c.green, 0x00);
        assert_eq!(c.blue, 0x00);
    }

    #[test]
    fn axis_config_title_padding_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        let cfg = AxisConfigSpec { title_padding: Some(12.0), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        assert_eq!(axis.title_padding, Some(12.0));
    }

    #[test]
    fn axis_config_label_format_raw_reformats_numeric_labels() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["1000".to_string(), "2000".to_string(), "3000".to_string()],
            None,
        );
        let cfg = AxisConfigSpec { label_format_raw: Some(",.0f".to_string()), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        apply_label_format_to_axis(&mut axis);
        // ",.0f" formats with thousands separator and 0 decimal places.
        assert_eq!(axis.tick_labels, vec!["1,000", "2,000", "3,000"]);
    }

    #[test]
    fn axis_config_label_format_raw_with_tick_values_replaces_labels() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0.0".to_string(), "0.5".to_string(), "1.0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            label_format_raw: Some(".1%".to_string()),
            tick_values: Some(vec![0.0, 0.5, 1.0]),
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        apply_label_format_to_axis(&mut axis);
        assert_eq!(axis.tick_labels, vec!["0.0%", "50.0%", "100.0%"]);
    }

    #[test]
    fn axis_config_label_padding_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string(), "50".to_string(), "100".to_string()],
            None,
        );
        let cfg = AxisConfigSpec { label_padding: Some(6.0), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        assert_eq!(axis.label_padding, Some(6.0));
    }

    #[test]
    fn axis_config_per_channel_wins_over_configure() {
        // Per-channel label_angle_override (level 2) wins over configure (level 3).
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            Some(-45.0), // per-channel override already set
        );
        let cfg = AxisConfigSpec { label_angle: Some(-90.0), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg));
        // -45.0 should win because it was already set (Some).
        assert_eq!(axis.label_angle_override, Some(-45.0));
    }

    #[test]
    fn legend_config_gradient_length_fills_legend_overrides() {
        let prep_overrides = legend_overrides_from_prep_default();
        // When prep has no gradient_length, configure() should fill it.
        let mut overrides = prep_overrides;
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                gradient_length: Some(300.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.gradient_length, Some(300.0));
    }

    #[test]
    fn legend_config_symbol_type_fills_legend_overrides() {
        let mut overrides = legend_overrides_from_prep_default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                symbol_type: Some("square".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.symbol_type.as_deref(), Some("square"));
    }

    #[test]
    fn legend_config_per_encoding_wins_over_configure() {
        // Per-encoding gradient_length (already set) must not be overwritten by configure.
        let mut overrides = LegendOverrides {
            gradient_length: Some(150.0), // already set at level 2
            ..Default::default()
        };
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                gradient_length: Some(300.0), // configure (level 3) tries to override
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.gradient_length, Some(150.0)); // level 2 wins
    }

    #[test]
    fn color_config_domain_override_updates_continuous_scale() {
        use scale_resolve::ColorScale;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let mut color_scale: Option<ColorScale> = Some(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
        });
        let cfg = ColorConfigSpec { domain: Some(vec![serde_json::json!(10.0), serde_json::json!(90.0)]), ..Default::default() };
        apply_color_config_to_color_scale(&mut color_scale, &cfg);
        if let Some(ColorScale::Continuous { domain, .. }) = color_scale {
            assert_eq!(domain, (10.0, 90.0));
        } else {
            panic!("expected Continuous color scale");
        }
    }

    #[test]
    fn color_config_range_override_builds_gradient() {
        use scale_resolve::ColorScale;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let mut color_scale: Option<ColorScale> = Some(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
        });
        let cfg = ColorConfigSpec {
            range: Some(vec!["#ffffff".to_string(), "#000000".to_string()]),
            ..Default::default()
        };
        apply_color_config_to_color_scale(&mut color_scale, &cfg);
        if let Some(ColorScale::Continuous { scheme: ContinuousScheme::Gradient(stops), .. }) = color_scale {
            assert_eq!(stops.len(), 2);
            // First stop at t=0, last at t=1.
            assert!((stops[0].0 - 0.0).abs() < 1e-9);
            assert!((stops[1].0 - 1.0).abs() < 1e-9);
        } else {
            panic!("expected Gradient color scheme after range override");
        }
    }

    /// Helper: build a zero-filled LegendOverrides (as if no per-encoding overrides existed).
    fn legend_overrides_from_prep_default() -> LegendOverrides {
        LegendOverrides::default()
    }
}
