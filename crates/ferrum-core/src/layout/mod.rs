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
pub use self::panel::{FacetKey, PanelLayout, StripTitleLayout, TextAnchor};
pub use self::text_metrics::{HeuristicMetrics, TextMetrics};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutResult {
    pub viewport: Rect,
    pub panels: Vec<PanelLayout>,
    pub axes: Vec<AxisLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub legend: Option<LegendLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub chart_title: Option<ChartTitleLayout>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<LayoutWarning>,
}

/// Chart-level (top-of-SVG) title placement. Positioned in the band reserved
/// at the top of the inner rect by `compute_layout`. The renderer reads
/// `theme.title_color`, `theme.title_font_family`, `theme.title_font_size`,
/// and `theme.title_font_weight` for styling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartTitleLayout {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub anchor: TextAnchor,
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

/// Theme fields actually read by Phase 6 + Phase 7. Kept decoupled from a full
/// Theme type — Phase 8 grammar will translate ferrum.Theme into this shape.
///
/// Color fields use palette::Srgba<u8>. Task 6 will add a `Color` type alias
/// and `from_hex_str` helper; for now we construct directly via Srgba::new.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeInputs {
    // Phase 6 layout fields.
    pub padding: f64,
    pub column_padding: f64,
    pub row_padding: f64,
    pub axis_title_padding: f64,
    pub label_font_size: f64,
    pub title_font_size: f64,
    pub legend_orient: LegendOrient,

    // Phase 7 render fields — sizes/widths/opacities.
    pub point_size: f64,
    pub line_stroke_width: f64,
    pub bar_corner_radius: f64,
    pub area_opacity: f64,
    pub default_opacity: f64,
    pub axis_line_width: f64,
    pub tick_size: f64,
    pub grid_width: f64,
    pub grid: bool,
    pub strip_text_size: f64,
    pub strip_padding: f64,

    // Phase 7 render fields — colors.
    pub mark_color: palette::Srgba<u8>,
    pub axis_line_color: palette::Srgba<u8>,
    pub tick_color: palette::Srgba<u8>,
    pub grid_color: palette::Srgba<u8>,
    pub font_color: palette::Srgba<u8>,
    pub background_color: palette::Srgba<u8>,
    pub strip_background_color: palette::Srgba<u8>,

    // Phase 8a size/opacity range fields.
    pub point_size_min: f64,  // default 3.0
    pub point_size_max: f64,  // default 30.0
    pub opacity_min: f64,     // default 0.1
    pub opacity_max: f64,     // default 1.0

    // Themes-T1 additions (ferrum-spec.md §3.13).

    // Typography
    pub font_family: String,
    pub font_weight: String,
    pub title_font_family: String,
    pub title_font_weight: String,
    pub title_color: palette::Srgba<u8>,
    pub title_anchor: TextAnchor,
    pub title_offset: f64,
    pub label_font_family: String,
    pub label_color: palette::Srgba<u8>,

    // Axes
    pub axis_line: bool,
    pub tick_width: f64,

    // Grid
    pub grid_dash: Option<Vec<f64>>,
    pub grid_opacity: f64,

    // Marks
    pub point_opacity: f64,

    // Palette
    pub color_scheme: String,

    // Legend
    pub legend_direction: Option<LegendDirection>,
    pub legend_title_font_size: f64,
}

impl Default for ThemeInputs {
    fn default() -> Self {
        // OKABE_ITO[0] = #E69F00 = (230, 159, 0).
        let okabe_orange = palette::Srgba::new(0xE6, 0x9F, 0x00, 0xFF);
        let neutral_888  = palette::Srgba::new(0x88, 0x88, 0x88, 0xFF);
        let neutral_eee  = palette::Srgba::new(0xEE, 0xEE, 0xEE, 0xFF);
        let text_222     = palette::Srgba::new(0x22, 0x22, 0x22, 0xFF);
        let bg_white     = palette::Srgba::new(0xFF, 0xFF, 0xFF, 0xFF);
        let strip_bg     = palette::Srgba::new(0xF0, 0xF0, 0xF0, 0xFF);

        Self {
            // Phase 6.
            padding: DEFAULT_PADDING,
            column_padding: DEFAULT_PADDING,
            row_padding: DEFAULT_PADDING,
            axis_title_padding: DEFAULT_AXIS_TITLE_PADDING,
            label_font_size: DEFAULT_LABEL_FONT_SIZE,
            title_font_size: DEFAULT_TITLE_FONT_SIZE,
            legend_orient: LegendOrient::Right,

            // Phase 7 sizes / widths / opacities.
            point_size: 30.0,
            line_stroke_width: 1.5,
            bar_corner_radius: 0.0,
            area_opacity: 0.4,
            default_opacity: 1.0,
            axis_line_width: 1.0,
            tick_size: 4.0,
            grid_width: 1.0,
            grid: true,
            strip_text_size: 13.0,
            strip_padding: 4.0,

            // Phase 7 colors.
            mark_color: okabe_orange,
            axis_line_color: neutral_888,
            tick_color: neutral_888,
            grid_color: neutral_eee,
            font_color: text_222,
            background_color: bg_white,
            strip_background_color: strip_bg,

            // Phase 8a size/opacity ranges.
            point_size_min: 3.0,
            point_size_max: 30.0,
            opacity_min: 0.1,
            opacity_max: 1.0,

            // Themes-T1 — values match current visual identity. T4 will flip these.
            font_family: "Inter".into(),       // resvg default; T4 → "DejaVu Sans"
            font_weight: "normal".into(),
            title_font_family: "Inter".into(),
            title_font_weight: "normal".into(),       // T4 → "600"; "normal" suppresses attr emission to preserve current SVG
            title_color: text_222,
            title_anchor: TextAnchor::Middle,         // T4 → Start
            title_offset: 4.0,                        // T4 → 6.0
            label_font_family: "Inter".into(),
            label_color: text_222,                    // T4 → label_555 = #555555

            axis_line: true,
            tick_width: 1.0,

            grid_dash: None,
            grid_opacity: 1.0,

            point_opacity: 1.0,

            color_scheme: "okabe_ito".into(),         // T4 → "tableau10"

            legend_direction: None,                   // None = auto-derive from legend_orient
            legend_title_font_size: 13.0,             // matches title_font_size
        }
    }
}

pub fn compute_layout(
    spec: &crate::spec::chart::ChartSpec,
    theme: &ThemeInputs,
    viewport: Viewport,
    axes: &AxesInput,
    facet_groups: &[FacetGroup],
    legend_entries: &[LegendEntry],
    legend_title: Option<String>,
    metrics: &dyn TextMetrics,
) -> Result<LayoutResult, LayoutError> {
    // 1. Validate inputs.
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(LayoutError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }
    if let Some(facet) = &spec.facet {
        match &facet.mode {
            FacetMode::Wrap { ncols } if *ncols == 0 => {
                return Err(LayoutError::InvalidFacetSpec("ncols must be > 0".into()));
            }
            FacetMode::Grid { nrows, ncols } if *nrows == 0 || *ncols == 0 => {
                return Err(LayoutError::InvalidFacetSpec("nrows and ncols must be > 0".into()));
            }
            _ => {}
        }
        if facet_groups.is_empty() {
            return Err(LayoutError::EmptyFacetGroups);
        }
    }

    // 2. Apply outer padding.
    let viewport_rect = viewport.into_rect();
    let inner = viewport_rect.shrink(Inset::uniform(theme.padding));
    if inner.w <= 0.0 || inner.h <= 0.0 {
        let dim = viewport.width.min(viewport.height);
        return Err(LayoutError::PaddingExceedsViewport {
            padding: theme.padding,
            viewport_dim: dim,
        });
    }

    // 2b. Reserve chart-level title band (Themes-T2.5a) above plot region.
    // Band height ≈ title_font_size * 1.4 + title_offset. Stored absolute x/y for render emission.
    let (chart_title_layout, inner) = if let Some(title_text) = spec.title.as_ref() {
        let title_line_h = metrics.line_height(theme.title_font_size);
        let band_h = title_line_h + theme.title_offset;
        let (band, rest) = inner.split_top(band_h);
        // x position derived from anchor; y baseline lands at top of remaining inner (just above plot region)
        let x = match theme.title_anchor {
            TextAnchor::Start => band.x,
            TextAnchor::Middle => band.x + band.w / 2.0,
            TextAnchor::End => band.x + band.w,
        };
        let y = band.y + title_line_h;
        let chart_title = ChartTitleLayout {
            text: title_text.clone(),
            x,
            y,
            anchor: theme.title_anchor,
        };
        (Some(chart_title), rest)
    } else {
        (None, inner)
    };

    // 3. Reserve legend strip.
    let (legend_layout, inner_after_legend) = legend::layout_legend(
        legend_entries,
        theme.legend_orient,
        inner,
        theme.label_font_size,
        metrics,
        theme.legend_direction,
        legend_title.as_deref(),
        theme.legend_title_font_size,
    );
    let legend_dropped = legend_entries
        .len()
        .saturating_sub(legend_layout.as_ref().map_or(0, |l| l.entries.len()))
        as u32;

    // 4 + 5. Reserve y-axis title gutter + label band; reserve x-axis label band.
    let y_title_gutter = axis::compute_y_title_width(
        &axes.y,
        theme.title_font_size,
        theme.axis_title_padding,
        metrics,
    );
    let y_label_band = axis::compute_y_label_band_width(&axes.y, theme.label_font_size, metrics);
    let x_label_band = metrics.line_height(theme.label_font_size);
    let x_title_gutter = if axes.x.title.is_some() {
        metrics.line_height(theme.title_font_size) + theme.axis_title_padding
    } else {
        0.0
    };

    let plot_region = inner_after_legend.shrink(Inset {
        top: 0.0,
        right: 0.0,
        bottom: x_label_band + x_title_gutter,
        left: y_label_band + y_title_gutter,
    });

    // 6. Split into facet cells (or a single panel).
    let mut panels: Vec<PanelLayout> = Vec::new();
    let mut warnings: Vec<LayoutWarning> = Vec::new();
    if legend_dropped > 0 {
        warnings.push(LayoutWarning::LegendOverflowed { entries_dropped: legend_dropped });
    }

    let panel_rects: Vec<(u32, u32, Rect, Option<FacetKey>)> = if let Some(facet) = &spec.facet {
        let n_panels = facet_groups.len() as u32;
        let (gx, gy) = facet
            .spacing
            .map(|s| (s, s))
            .unwrap_or((theme.column_padding, theme.row_padding));
        let grid = match facet.mode {
            FacetMode::Wrap { ncols } => {
                facet::FacetGrid::compute_wrap(ncols, n_panels, plot_region, gx, gy)
            }
            FacetMode::Grid { nrows, ncols } => {
                facet::FacetGrid::compute_grid(nrows, ncols, n_panels, plot_region, gx, gy)
            }
        };
        if grid.dropped_count() > 0 {
            warnings.push(LayoutWarning::PanelsDropped { count: grid.dropped_count() });
        }
        grid.panel_positions()
            .into_iter()
            .enumerate()
            .map(|(i, (row, col))| {
                let rect = grid.cell_rect(row, col);
                let key = facet_groups.get(i).map(|g| g.key.clone());
                (row, col, rect, key)
            })
            .collect()
    } else {
        vec![(0, 0, plot_region, None)]
    };

    let strip_band_height = if spec.facet.is_some() {
        metrics.line_height(theme.strip_text_size) + 2.0 * theme.strip_padding
    } else {
        0.0
    };

    // 7. Per-panel: clamp degenerate rects, collect axes.
    let mut axis_layouts: Vec<AxisLayout> = Vec::new();
    for (panel_index, (row, col, mut rect, facet_key)) in panel_rects.into_iter().enumerate() {
        if rect.w <= MIN_PANEL_DIM || rect.h <= MIN_PANEL_DIM {
            warnings.push(LayoutWarning::PanelCollapsed { panel_index });
            rect = Rect::ZERO;
        }

        let strip_title = if let Some(key) = &facet_key {
            if rect != Rect::ZERO {
                let strip_rect = Rect {
                    x: rect.x,
                    y: rect.y,
                    w: rect.w,
                    h: strip_band_height,
                };
                let new_panel_rect = Rect {
                    x: rect.x,
                    y: rect.y + strip_band_height,
                    w: rect.w,
                    h: (rect.h - strip_band_height).max(0.0),
                };
                rect = new_panel_rect;
                Some(StripTitleLayout {
                    text: key.value.clone(),
                    anchor: (
                        strip_rect.x + strip_rect.w / 2.0,
                        strip_rect.y + theme.strip_padding + theme.strip_text_size,
                    ),
                    align: TextAnchor::Middle,
                    font_size: theme.strip_text_size,
                })
            } else {
                None
            }
        } else {
            None
        };

        panels.push(PanelLayout {
            plot_area: rect,
            facet_key,
            row,
            col,
            strip_title,
        });

        if rect != Rect::ZERO {
            let y_axis = axis::layout_y_axis(
                &axes.y,
                rect,
                panel_index,
                theme.label_font_size,
                theme.title_font_size,
                theme.axis_title_padding,
                metrics,
            );
            axis_layouts.push(y_axis);

            let (x_axis, xwarn) = axis::layout_x_axis(
                &axes.x,
                rect,
                panel_index,
                theme.label_font_size,
                theme.title_font_size,
                theme.axis_title_padding,
                metrics,
            );
            if let Some(axis::XAxisWarning::LabelsElided { count }) = xwarn {
                warnings.push(LayoutWarning::LabelsElided {
                    axis: axis_layouts.len(),
                    count,
                });
            }
            axis_layouts.push(x_axis);
        }
    }

    Ok(LayoutResult {
        viewport: viewport_rect,
        panels,
        axes: axis_layouts,
        legend: legend_layout,
        chart_title: chart_title_layout,
        warnings,
    })
}

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
            chart_title: None,
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

    use crate::layout::axis::{AxesInput, AxisInput, AxisOrient};
    use crate::layout::facet::FacetGroup;
    use crate::layout::text_metrics::{fixed_width, MockMetrics};
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    fn minimal_chart_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
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
        }
    }

    fn dummy_axes() -> AxesInput {
        AxesInput {
            x: AxisInput {
                orient: AxisOrient::Bottom,
                title: None,
                tick_labels: vec!["0".into(), "1".into(), "2".into(), "3".into()],
                label_angle_override: None,
            },
            y: AxisInput {
                orient: AxisOrient::Left,
                title: None,
                tick_labels: vec!["0".into(), "5".into(), "10".into()],
                label_angle_override: None,
            },
        }
    }

    fn default_theme_inputs() -> ThemeInputs {
        ThemeInputs::default()
    }

    #[test]
    fn compute_layout_single_chart_no_facet_no_legend() {
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes,
            &[],
            &[],
            None,
            &m,
        )
        .expect("layout should succeed on minimal spec");

        assert_eq!(result.viewport, viewport.into_rect());
        assert_eq!(result.panels.len(), 1);
        assert_eq!(result.axes.len(), 2);
        assert!(result.legend.is_none());
        assert!(result.warnings.is_empty());

        let panel = &result.panels[0];
        assert!(panel.plot_area.w > 0.0 && panel.plot_area.h > 0.0);
        assert_eq!(panel.row, 0);
        assert_eq!(panel.col, 0);
        assert!(panel.facet_key.is_none());
    }

    #[test]
    fn compute_layout_invalid_viewport_errors() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let err = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 0.0, height: 400.0 },
            &axes,
            &[],
            &[],
            None,
            &m,
        )
        .unwrap_err();
        match err {
            LayoutError::InvalidViewport { width, .. } => assert_eq!(width, 0.0),
            other => panic!("expected InvalidViewport, got {:?}", other),
        }
    }

    #[test]
    fn compute_layout_padding_exceeds_viewport_errors() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let theme = ThemeInputs { padding: 100.0, ..ThemeInputs::default() };
        let err = compute_layout(
            &spec,
            &theme,
            Viewport { width: 50.0, height: 50.0 },
            &axes,
            &[],
            &[],
            None,
            &m,
        )
        .unwrap_err();
        match err {
            LayoutError::PaddingExceedsViewport { .. } => {}
            other => panic!("expected PaddingExceedsViewport, got {:?}", other),
        }
    }

    #[test]
    fn compute_layout_serde_round_trip() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 600.0, height: 400.0 },
            &axes,
            &[],
            &[],
            None,
            &m,
        )
        .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, result);
    }

    use crate::layout::facet::{FacetMode, FacetSpec};
    use crate::layout::panel::FacetKey;

    fn faceted_spec(ncols: u32) -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "species".into(),
            mode: FacetMode::Wrap { ncols },
            spacing: None,
        });
        s
    }

    fn three_groups() -> Vec<FacetGroup> {
        vec![
            FacetGroup { key: FacetKey { field: "species".into(), value: "setosa".into() }, n_rows: 50 },
            FacetGroup { key: FacetKey { field: "species".into(), value: "versicolor".into() }, n_rows: 50 },
            FacetGroup { key: FacetKey { field: "species".into(), value: "virginica".into() }, n_rows: 50 },
        ]
    }

    #[test]
    fn compute_layout_faceted_three_panels_one_legend() {
        let spec = faceted_spec(3);
        let groups = three_groups();
        let legend = vec![
            LegendEntry { label: "setosa".into(), symbol: SymbolKind::Circle },
            LegendEntry { label: "versicolor".into(), symbol: SymbolKind::Circle },
            LegendEntry { label: "virginica".into(), symbol: SymbolKind::Circle },
        ];
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 400.0 },
            &axes,
            &groups,
            &legend,
            None,
            &m,
        )
        .unwrap();

        assert_eq!(result.panels.len(), 3);
        assert_eq!(result.axes.len(), 6);
        assert!(result.legend.is_some());
        assert!(result.warnings.is_empty(), "unexpected warnings: {:?}", result.warnings);

        assert_eq!(
            result.panels[0].facet_key.as_ref().unwrap().value,
            "setosa"
        );
        assert_eq!(
            result.panels[2].facet_key.as_ref().unwrap().value,
            "virginica"
        );
    }

    #[test]
    fn compute_layout_facet_grid_overflow_warns() {
        let mut spec = minimal_chart_spec();
        spec.facet = Some(FacetSpec {
            field: "species".into(),
            mode: FacetMode::Grid { nrows: 1, ncols: 2 },
            spacing: None,
        });
        let groups = three_groups();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 400.0 },
            &axes,
            &groups,
            &[],
            None,
            &m,
        )
        .unwrap();

        assert_eq!(result.panels.len(), 2);
        let dropped = result.warnings.iter().any(|w| matches!(
            w,
            LayoutWarning::PanelsDropped { count: 1 }
        ));
        assert!(dropped, "expected PanelsDropped(1); got {:?}", result.warnings);
    }

    #[test]
    fn compute_layout_faceted_emits_strip_titles() {
        let spec = faceted_spec(3);
        let groups = three_groups();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 400.0 },
            &axes,
            &groups,
            &[],
            None,
            &m,
        ).unwrap();

        assert_eq!(result.panels.len(), 3);
        for (i, panel) in result.panels.iter().enumerate() {
            let strip = panel.strip_title.as_ref()
                .unwrap_or_else(|| panic!("panel {i} missing strip_title"));
            assert!(!strip.text.is_empty());
            assert_eq!(strip.font_size, 13.0);
            assert!(strip.anchor.0 >= panel.plot_area.x);
            assert!(strip.anchor.0 <= panel.plot_area.x + panel.plot_area.w);
        }
    }

    #[test]
    fn compute_layout_unfaceted_omits_strip_titles() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 600.0, height: 400.0 },
            &axes,
            &[],
            &[],
            None,
            &m,
        ).unwrap();
        assert!(result.panels[0].strip_title.is_none());
    }

    #[test]
    fn theme_inputs_default_includes_render_fields() {
        let t = ThemeInputs::default();
        // Phase 6 fields preserved.
        assert_eq!(t.padding, DEFAULT_PADDING);
        assert_eq!(t.label_font_size, DEFAULT_LABEL_FONT_SIZE);
        // Phase 7 additions.
        assert_eq!(t.point_size, 30.0);
        assert_eq!(t.line_stroke_width, 1.5);
        assert_eq!(t.bar_corner_radius, 0.0);
        assert_eq!(t.area_opacity, 0.4);
        assert_eq!(t.default_opacity, 1.0);
        assert_eq!(t.axis_line_width, 1.0);
        assert_eq!(t.tick_size, 4.0);
        assert_eq!(t.grid_width, 1.0);
        assert_eq!(t.grid, true);
        assert_eq!(t.strip_text_size, 13.0);
        assert_eq!(t.strip_padding, 4.0);
    }
}
