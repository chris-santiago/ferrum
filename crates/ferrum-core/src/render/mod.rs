//! Phase 7 — static renderer. Pure functions: ChartSpec + RecordBatch + ThemeInputs +
//! Viewport -> deterministic SVG/PNG. See docs/superpowers/specs/2026-05-09-static-renderer-design.md.

pub(crate) mod arrow_cast;
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
pub(crate) mod scene_build;
pub(crate) mod svg_walk;
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

use crate::layout::{compute_layout, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;

pub fn render_svg(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
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
    let background = config.background.or(Some(theme.background_color));

    let prep = prepare::prepare_render_inputs(spec, batch, theme)?;
    let mut warnings = prep.warnings.clone();

    // D13: apply per-chart legend overrides from encoding.color.legend extra fields.
    // Clone theme and patch the relevant fields so existing golden tests are unaffected
    // when no legend overrides are present (the clone is zero-cost when unneeded because
    // all fields are Copy or Clone — the legend_orient_override path is uncommon).
    let mut effective_theme;
    let theme_ref: &ThemeInputs = if prep.legend_orient_override.is_some()
        || prep.legend_title_font_size_override.is_some()
    {
        effective_theme = theme.clone();
        if let Some(orient) = prep.legend_orient_override {
            effective_theme.legend_orient = orient;
        }
        if let Some(fs) = prep.legend_title_font_size_override {
            effective_theme.legend_title_font_size = fs;
        }
        &effective_theme
    } else {
        theme
    };
    // D13: legend title override (replaces the default field-name title when Some).
    let effective_legend_title = prep
        .legend_title_override
        .clone()
        .or_else(|| prep.legend_title.clone());

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
    )
    .map_err(|e| RenderError::LayoutFailed(e.to_string()))?;
    for w in &layout.warnings {
        warnings.push(RenderWarning::Layout(w.clone()));
    }

    let scene = scene_build::build_scene(
        spec, &prep, &layout, theme_ref, config, &mut warnings,
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
) -> Result<RenderOutput<Vec<u8>>, RenderError> {
    let svg_out = render_svg(spec, batch, theme, viewport, config)?;
    let w = (svg_out.layout.viewport.w * config.scale).round() as u32;
    let h = (svg_out.layout.viewport.h * config.scale).round() as u32;
    let bytes = png::svg_string_to_png_bytes(&svg_out.bytes, w, h, config.scale)?;
    Ok(RenderOutput { bytes, layout: svg_out.layout, warnings: svg_out.warnings })
}

/// Emit the chart-level title and optional subtitle into the SVG buffer.
///
/// Resolves per-chart `TitleSpec` overrides (font size, weight, color,
/// subtitle color/size) falling back to `theme` defaults. Pure output —
/// no state escapes beyond what is written to `out`.
fn render_title(
    layout: &crate::layout::LayoutResult,
    spec: &ChartSpec,
    theme: &ThemeInputs,
    out: &mut svg::SvgBuffer,
) {
    let Some(title) = &layout.chart_title else { return };
    let title_spec = spec.title.as_ref();
    let resolved_font_size = title_spec
        .and_then(|t| t.font_size)
        .unwrap_or(theme.title_font_size);
    let resolved_font_weight: String = title_spec
        .and_then(|t| t.font_weight.clone())
        .unwrap_or_else(|| theme.title_font_weight.clone());
    let resolved_color = title_spec
        .and_then(|t| t.color.as_deref())
        .and_then(|hex| color::from_hex_str(hex).ok())
        .unwrap_or(theme.title_color);
    let style = svg::TextStyle {
        fill: resolved_color,
        font_size: resolved_font_size,
        anchor: title.anchor,
        angle: 0.0,
        font_family: &theme.title_font_family,
        font_weight: if resolved_font_weight == "normal" {
            None
        } else {
            Some(&resolved_font_weight)
        },
        dominant_baseline: None,
    };
    out.text(title.x, title.y, &title.text, &style);
    if let (Some(subtitle), Some(sy)) = (&title.subtitle, title.subtitle_y) {
        let resolved_sub_color = title_spec
            .and_then(|t| t.subtitle_color.as_deref())
            .and_then(|hex| color::from_hex_str(hex).ok())
            .unwrap_or(theme.font_color);
        let resolved_sub_font_size = title_spec
            .and_then(|t| t.subtitle_font_size)
            .unwrap_or(resolved_font_size * 0.85);
        let sub_style = svg::TextStyle {
            fill: resolved_sub_color,
            font_size: resolved_sub_font_size,
            anchor: title.anchor,
            angle: 0.0,
            font_family: &theme.font_family,
            font_weight: None,
            dominant_baseline: None,
        };
        out.text(title.x, sy, subtitle, &sub_style);
    }
}

/// Emit the legend (categorical or colorbar) into the SVG buffer.
///
/// Builds a rendering spec from the first layer's encoding (accounts for
/// CoordFlip), re-resolves the color scale for the legend palette, and
/// dispatches to `marks::legend::draw`. Returns `Err` only if scale
/// resolution fails.
fn render_legend(
    layout: &crate::layout::LayoutResult,
    spec: &ChartSpec,
    prep: &prepare::PreparedInputs,
    theme: &ThemeInputs,
    out: &mut svg::SvgBuffer,
) -> Result<(), RenderError> {
    let Some(legend) = &layout.legend else { return Ok(()) };
    let rendering_spec_for_legend = ChartSpec {
        encoding: prep.layers[0].encoding.clone(),
        ..spec.clone()
    };
    let color_scale = if rendering_spec_for_legend.encoding.color.is_some() {
        let (gs, _) = scale_resolve::resolve_scales_with_outputs(
            &rendering_spec_for_legend,
            prep.final_batch(),
            &prep.transform_outputs,
            (0.0, 1.0),
            (0.0, 1.0),
            theme,
        )?;
        gs.color
    } else {
        None
    };
    marks::legend::draw(legend, color_scale.as_ref(), theme, out);
    Ok(())
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
        let result = render_svg(&spec, &batch, &theme, viewport, &config).unwrap();
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
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let result = render_svg(
            &spec,
            &batch,
            &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
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
        let a = render_svg(&spec, &batch, &theme, viewport, &config).unwrap();
        let b = render_svg(&spec, &batch, &theme, viewport, &config).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }

    #[test]
    fn scene_graph_path_matches_old_path_scatter() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let cfg = config::RenderConfig::default();
        let old_svg = render_svg(&spec, &batch, &theme, viewport, &cfg).unwrap().bytes;

        let prep = prepare::prepare_render_inputs(&spec, &batch, &theme).unwrap();
        let mut warnings = prep.warnings.clone();

        let mut effective_theme;
        let theme_ref: &ThemeInputs = if prep.legend_orient_override.is_some()
            || prep.legend_title_font_size_override.is_some()
        {
            effective_theme = theme.clone();
            if let Some(orient) = prep.legend_orient_override {
                effective_theme.legend_orient = orient;
            }
            if let Some(fs) = prep.legend_title_font_size_override {
                effective_theme.legend_title_font_size = fs;
            }
            &effective_theme
        } else {
            &theme
        };
        let effective_legend_title = prep
            .legend_title_override
            .clone()
            .or_else(|| prep.legend_title.clone());

        let metrics = font::FontdueMetrics::new();
        let vp2 = Viewport {
            width: cfg.width.unwrap_or(viewport.width),
            height: cfg.height.unwrap_or(viewport.height),
        };
        let layout = compute_layout(
            &spec, theme_ref, vp2,
            &prep.axes, &prep.facet_groups, &prep.legend_entries,
            effective_legend_title, prep.colorbar.as_ref(), &metrics,
        ).unwrap();
        for w in &layout.warnings {
            warnings.push(RenderWarning::Layout(w.clone()));
        }

        let scene = scene_build::build_scene(
            &spec, &prep, &layout, theme_ref, &cfg, &mut warnings,
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
        let a = render_png(&spec, &batch, &theme, viewport, &config).unwrap();
        let b = render_png(&spec, &batch, &theme, viewport, &config).unwrap();
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
        ).unwrap();
        check_golden("scatter_minimal", &result.bytes);

        let png_result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
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
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
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
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
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
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
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
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
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
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
            }),
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("faceted_scatter", &result.bytes);
    }
}
