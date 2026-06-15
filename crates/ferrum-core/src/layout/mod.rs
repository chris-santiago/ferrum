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
pub const MIN_PANEL_DIM: f64 = 1.0;
pub const DEFAULT_LABEL_FONT_SIZE: f64 = 11.0;
pub const DEFAULT_TITLE_FONT_SIZE: f64 = 13.0;

// Axis label overhaul constants.
/// Cascade of angles tried in order before falling back to elision or culling.
pub(crate) const ANGLE_CASCADE: [f64; 5] = [0.0, -30.0, -45.0, -60.0, -90.0];
/// Multiplicative factor applied to font size on each shrink step.
pub(crate) const FONT_SHRINK_FACTOR: f64 = 0.82;
/// Default maximum number of visible tick labels before culling kicks in.
pub(crate) const DEFAULT_CULL_THRESHOLD: u32 = 8;

use serde::{Deserialize, Serialize};

pub use self::axis::{
    AxesInput, AxisInput, AxisLayout, AxisOrient, AxisTitleLayout, LabelOverlap, TickLayout,
    TickProjection,
};
pub(crate) use self::axis::AxisStyleOverrides;
pub use self::facet::{FacetGroup, FacetMode, FacetResolve, FacetSpec, ResolveMode};
pub use self::geometry::{Inset, Rect, Viewport};
pub use self::legend::{
    ColorbarInput, ColorbarLayout, ColorbarTick, LegendDirection, LegendEntry,
    AuxLegendInput, LegendEntryLayout, LegendLayout, LegendOrient, LegendOverrides,
    ShapeLegendEntry, SizeLegendEntry, SymbolKind,
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
    /// Auxiliary (size / shape) legend blocks stacked beneath the color
    /// legend. Empty for color-only charts (byte-identical serialization to
    /// the pre-size-legend shape via `skip_serializing_if`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aux_legends: Vec<LegendLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub chart_title: Option<ChartTitleLayout>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<LayoutWarning>,
}

/// Clamp a dynamically-estimated axis margin band to the per-axis
/// `min_extent`/`max_extent` overrides (B5). `min` reserves at least that many
/// px; `max` caps the reservation (labels may clip past the cap — allowed). When
/// both are `None` the dynamic value passes through unchanged, preserving
/// byte-identical default output. A `min > max` (user contradiction) resolves to
/// `max` (the cap wins), matching the "max is a hard ceiling" semantic.
fn clamp_axis_extent(dynamic: f64, min_extent: Option<f64>, max_extent: Option<f64>) -> f64 {
    let mut band = dynamic;
    if let Some(min) = min_extent {
        band = band.max(min);
    }
    if let Some(max) = max_extent {
        band = band.min(max);
    }
    band
}

/// Chart-level (top-of-SVG) title placement. Positioned in the band reserved
/// at the top of the inner rect by `compute_layout`. The renderer reads
/// `theme.title_color`, `theme.title_font_family`, `theme.title_font_size`,
/// and `theme.title_font_weight` for styling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartTitleLayout {
    pub text: String,
    /// Schwabish SB1: optional subtitle drawn as a second line below the title.
    /// When `Some`, the title band reserves an extra line height; when `None`,
    /// layout is byte-identical to Themes-T2.5a.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    pub x: f64,
    pub y: f64,
    /// Schwabish SB1: y baseline for the subtitle line. Only meaningful when
    /// `subtitle` is `Some`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_y: Option<f64>,
    pub anchor: TextAnchor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayoutWarning {
    PanelCollapsed { panel_index: usize },
    LabelsElided { axis: usize, count: u32 },
    LegendOverflowed { entries_dropped: u32 },
    /// One or more facet panels were dropped because the explicit grid
    /// `nrows × ncols` is smaller than the number of facet groups.
    /// `count` is the number of dropped panels; `keys` identifies them by
    /// their facet-group key string (e.g. `"col_cat=c2"`).
    PanelsDropped { count: u32, keys: Vec<String> },
    /// One or more grid-facet cells in the observed cartesian product of
    /// distinct(row) × distinct(col) values contain no data rows.
    /// `keys` is one entry per empty cell, formatted as
    /// `"<col_field>=<col_val>, <row_field>=<row_val>"` so the user can
    /// identify the missing data combination.  One aggregated warning is
    /// emitted listing all empty cells (spec §4/§11).
    EmptyPartitions { keys: Vec<String> },
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

// ── ThemeInputs sub-structs ──────────────────────────────────────────────────
//
// The flat ~42-field ThemeInputs is decomposed into logical sub-structs.
// Each group is Clone + Debug + PartialEq + Default so ThemeInputs retains
// those derives. Serde is not derived on the sub-structs because
// ThemeInputs itself is not `Serialize`/`Deserialize` — it is populated
// from Python dicts via `render/binding.rs` (ThemeOverridesSpec).

/// Outer and inter-cell padding values.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemePadding {
    pub padding: f64,
    /// Per-side padding overrides. When `Some`, these win over `padding` for
    /// that side. `configure_padding(top=N)` sets `padding_top = Some(N)`.
    pub padding_top: Option<f64>,
    pub padding_right: Option<f64>,
    pub padding_bottom: Option<f64>,
    pub padding_left: Option<f64>,
    pub column_padding: f64,
    pub row_padding: f64,
    pub axis_title_padding: f64,
    pub strip_padding: f64,
}

impl Default for ThemePadding {
    fn default() -> Self {
        Self {
            padding: 16.0,
            padding_top: None,
            padding_right: None,
            padding_bottom: None,
            padding_left: None,
            column_padding: 12.0,
            row_padding: 12.0,
            axis_title_padding: 8.0,
            strip_padding: 6.0,
        }
    }
}

/// Numeric sizes, widths, opacities, and range extremes for marks.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeRenderSizes {
    pub point_size: f64,
    pub line_stroke_width: f64,
    pub bar_corner_radius: f64,
    pub area_opacity: f64,
    pub default_opacity: f64,
    pub point_opacity: f64,
    pub axis_line_width: f64,
    pub tick_size: f64,
    pub tick_width: f64,
    /// Major gridline stroke width. The legacy `grid_width` theme key maps here.
    pub grid_width: f64,
    /// Minor gridline stroke width. Derived lighter/thinner default (see
    /// `ThemeRenderSizes::default`); overridable via the `minor_grid_width` key.
    pub minor_grid_width: f64,
    pub strip_text_size: f64,
    pub point_size_min: f64,
    pub point_size_max: f64,
    pub opacity_min: f64,
    pub opacity_max: f64,
}

impl Default for ThemeRenderSizes {
    fn default() -> Self {
        Self {
            point_size: 36.0,
            line_stroke_width: 1.5,
            bar_corner_radius: 0.0,
            area_opacity: 0.35,
            default_opacity: 1.0,
            point_opacity: 1.0,
            axis_line_width: 1.0,
            tick_size: 4.0,
            tick_width: 1.0,
            grid_width: 0.5,
            // Minor gridlines are thinner than major by default (matplotlib/
            // seaborn convention). Overridable via `minor_grid_width`.
            minor_grid_width: 0.3,
            strip_text_size: 12.0,
            point_size_min: 4.0,
            point_size_max: 36.0,
            opacity_min: 0.1,
            opacity_max: 1.0,
        }
    }
}

/// Color values for marks, axes, grid, text, background, and strips.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeColors {
    pub mark_color: palette::Srgba<u8>,
    pub axis_line_color: palette::Srgba<u8>,
    pub tick_color: palette::Srgba<u8>,
    /// Major gridline color. The legacy `grid_color` theme key maps here.
    pub grid_color: palette::Srgba<u8>,
    /// Minor gridline color. Derived lighter default (see `ThemeColors::default`);
    /// overridable via the `minor_grid_color` key.
    pub minor_grid_color: palette::Srgba<u8>,
    pub font_color: palette::Srgba<u8>,
    pub background_color: palette::Srgba<u8>,
    pub strip_background_color: palette::Srgba<u8>,
    pub title_color: palette::Srgba<u8>,
    pub label_color: palette::Srgba<u8>,
    pub reference_line_color: palette::Srgba<u8>,
}

impl Default for ThemeColors {
    fn default() -> Self {
        let mark_blue  = palette::Srgba::new(0x25, 0x63, 0xEB, 0xFF);
        let text_fg    = palette::Srgba::new(0x1F, 0x29, 0x37, 0xFF);
        let label_gray = palette::Srgba::new(0x6B, 0x72, 0x80, 0xFF);
        let grid_warm  = palette::Srgba::new(0xD6, 0xD3, 0xD1, 0xFF);
        // Minor gridlines are lighter than major by default (derived tint of the
        // warm grid color). Overridable via `minor_grid_color`.
        let grid_minor = palette::Srgba::new(0xE8, 0xE6, 0xE4, 0xFF);
        let bg_cream   = palette::Srgba::new(0xFA, 0xF7, 0xF2, 0xFF);
        let strip_bg   = palette::Srgba::new(0xED, 0xE9, 0xE3, 0xFF);
        Self {
            mark_color: mark_blue,
            axis_line_color: label_gray,
            tick_color: label_gray,
            grid_color: grid_warm,
            minor_grid_color: grid_minor,
            font_color: text_fg,
            background_color: bg_cream,
            strip_background_color: strip_bg,
            title_color: text_fg,
            label_color: label_gray,
            reference_line_color: palette::Srgba::new(0x9C, 0xA3, 0xAF, 0xFF),
        }
    }
}

/// Font family, weight, and size fields for body, title, and label text.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeTypography {
    pub font_family: String,
    pub font_weight: String,
    pub label_font_size: f64,
    pub label_font_family: String,
    pub title_font_family: String,
    pub title_font_size: f64,
    pub title_font_weight: String,
    pub title_anchor: TextAnchor,
    pub title_offset: f64,
    pub legend_title_font_size: f64,
}

impl Default for ThemeTypography {
    fn default() -> Self {
        Self {
            font_family: "Inter".into(),
            font_weight: "normal".into(),
            label_font_size: DEFAULT_LABEL_FONT_SIZE,
            label_font_family: "Inter".into(),
            title_font_family: "Inter".into(),
            title_font_size: DEFAULT_TITLE_FONT_SIZE,
            title_font_weight: "600".into(),
            title_anchor: TextAnchor::Start,
            title_offset: 6.0,
            legend_title_font_size: DEFAULT_LABEL_FONT_SIZE,
        }
    }
}

/// Legend placement and layout.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeLegend {
    pub legend_orient: LegendOrient,
    pub legend_direction: Option<LegendDirection>,
    /// Number of columns for categorical legend entries. `None` (default) means
    /// a single vertical column (Right/Left orient) or a single horizontal row
    /// (Top/Bottom orient). `Some(N)` arranges entries left-to-right, top-to-bottom
    /// in N columns; only meaningful for vertical-direction legends.
    pub legend_columns: Option<u32>,
}

impl Default for ThemeLegend {
    fn default() -> Self {
        Self {
            legend_orient: LegendOrient::Right,
            legend_direction: None,
            legend_columns: None,
        }
    }
}

/// Grid visibility and styling, split into a major and a minor level.
///
/// The legacy single-level fields (`grid`, `grid_dash`, `grid_opacity`) are the
/// **major** level — existing themes/goldens are unchanged. The `minor` enable
/// flag defaults `false`, so minor gridlines are not emitted unless a theme opts
/// in; when off, `build_grid` output is byte-identical to before. Per-level
/// color/width live on `ThemeColors`/`ThemeRenderSizes`.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeGrid {
    /// Major gridline enable (legacy `grid` key). Default `true`.
    pub grid: bool,
    /// Major gridline dash (legacy `grid_dash` key).
    pub grid_dash: Option<Vec<f64>>,
    /// Major gridline opacity (legacy `grid_opacity` key).
    pub grid_opacity: f64,
    /// Minor gridline enable (`minor` key). Default `false` — minor gridlines
    /// are emitted only when a theme opts in, keeping default output unchanged.
    pub minor: bool,
    /// Minor gridline dash (`minor_grid_dash` key). Defaults to no dash.
    pub minor_grid_dash: Option<Vec<f64>>,
    /// Minor gridline opacity (`minor_grid_opacity` key). Derived lighter default.
    pub minor_grid_opacity: f64,
}

impl Default for ThemeGrid {
    fn default() -> Self {
        Self {
            grid: true,
            grid_dash: None,
            grid_opacity: 1.0,
            // Minor disabled by default → default output byte-identical.
            minor: false,
            minor_grid_dash: None,
            // Minor gridlines are fainter than major by default.
            minor_grid_opacity: 0.6,
        }
    }
}

/// Axis domain line visibility.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeAxis {
    pub axis_line: bool,
}

impl Default for ThemeAxis {
    fn default() -> Self {
        Self { axis_line: true }
    }
}

/// Palette scheme names.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemePalette {
    pub color_scheme: String,
    pub sequential_scheme: String,
    pub diverging_scheme: String,
}

impl Default for ThemePalette {
    fn default() -> Self {
        Self {
            color_scheme: "paper_ink".into(),
            sequential_scheme: "cool_blue".into(),
            diverging_scheme: "blue_to_red".into(),
        }
    }
}

/// Reference-line defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeReferenceLine {
    pub reference_line_dash: Option<Vec<f64>>,
}

impl Default for ThemeReferenceLine {
    fn default() -> Self {
        Self {
            reference_line_dash: Some(vec![4.0, 4.0]),
        }
    }
}

/// Theme fields actually read by Phase 6 + Phase 7. Kept decoupled from a full
/// Theme type — Phase 8 grammar will translate ferrum.Theme into this shape.
///
/// Color fields use palette::Srgba<u8>. Task 6 will add a `Color` type alias
/// and `from_hex_str` helper; for now we construct directly via Srgba::new.
///
/// Fields are organized into logical sub-structs. Accessor methods provide
/// backward-compatible flat access for consumers that prefer `theme.field_name`
/// style.
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeInputs {
    pub padding: ThemePadding,
    pub sizes: ThemeRenderSizes,
    pub colors: ThemeColors,
    pub typography: ThemeTypography,
    pub legend: ThemeLegend,
    pub grid: ThemeGrid,
    pub axis: ThemeAxis,
    pub palette: ThemePalette,
    pub reference_line: ThemeReferenceLine,

    // Axis label overhaul
    /// Maximum number of visible tick labels before culling kicks in.
    /// Values above this threshold trigger label density reduction.
    pub cull_threshold: u32,
}

impl Default for ThemeInputs {
    fn default() -> Self {
        // Paper Ink default identity (2026-05-12).
        // Warm cream background, blue lead mark, warm-tinted grid.
        // See docs/superpowers/specs/2026-05-12-custom-themes-design.md §4.
        Self {
            padding: ThemePadding::default(),
            sizes: ThemeRenderSizes::default(),
            colors: ThemeColors::default(),
            typography: ThemeTypography::default(),
            legend: ThemeLegend::default(),
            grid: ThemeGrid::default(),
            axis: ThemeAxis::default(),
            palette: ThemePalette::default(),
            reference_line: ThemeReferenceLine::default(),
            cull_threshold: DEFAULT_CULL_THRESHOLD,
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
    colorbar: Option<&ColorbarInput>,
    metrics: &dyn TextMetrics,
    legend_overrides: &legend::LegendOverrides,
    aux_legend_inputs: &[legend::AuxLegendInput],
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
    let inset = Inset {
        top:    theme.padding.padding_top.unwrap_or(theme.padding.padding),
        right:  theme.padding.padding_right.unwrap_or(theme.padding.padding),
        bottom: theme.padding.padding_bottom.unwrap_or(theme.padding.padding),
        left:   theme.padding.padding_left.unwrap_or(theme.padding.padding),
    };
    let inner = viewport_rect.shrink(inset);
    if inner.w <= 0.0 || inner.h <= 0.0 {
        let dim = viewport.width.min(viewport.height);
        return Err(LayoutError::PaddingExceedsViewport {
            padding: theme.padding.padding,
            viewport_dim: dim,
        });
    }

    // 2b. Reserve chart-level title band (Themes-T2.5a; Schwabish SB1 adds subtitle).
    // Band height ≈ title_font_size * 1.4 + (subtitle_font_size * 1.4 if subtitle)
    //   + title_offset. Without a subtitle, layout is byte-identical to T2.5a.
    let (chart_title_layout, inner) = if let Some(title_spec) = spec.title.as_ref() {
        // D1-D6: per-chart TitleSpec overrides for layout geometry.
        let resolved_font_size = title_spec
            .font_size
            .unwrap_or(theme.typography.title_font_size);
        let resolved_offset = title_spec
            .offset
            .unwrap_or(theme.typography.title_offset);
        let resolved_anchor = match title_spec.anchor.as_deref() {
            Some("middle") => TextAnchor::Middle,
            Some("end")    => TextAnchor::End,
            Some(_)        => TextAnchor::Start,
            None           => theme.typography.title_anchor,
        };
        let title_line_h = metrics.line_height(resolved_font_size);
        let subtitle_font_size = title_spec
            .subtitle_font_size
            .unwrap_or(resolved_font_size * 0.85);
        let subtitle_line_h = if title_spec.subtitle.is_some() {
            metrics.line_height(subtitle_font_size)
        } else {
            0.0
        };
        let band_h = title_line_h + subtitle_line_h + resolved_offset;
        let (band, rest) = inner.split_top(band_h);
        let x = match resolved_anchor {
            TextAnchor::Start => band.x,
            TextAnchor::Middle => band.x + band.w / 2.0,
            TextAnchor::End => band.x + band.w,
        };
        let y = band.y + title_line_h;
        let subtitle_y = if title_spec.subtitle.is_some() {
            Some(y + subtitle_line_h)
        } else {
            None
        };
        let chart_title = ChartTitleLayout {
            text: title_spec.text.clone(),
            subtitle: title_spec.subtitle.clone(),
            x,
            y,
            subtitle_y,
            anchor: resolved_anchor,
        };
        (Some(chart_title), rest)
    } else {
        (None, inner)
    };

    // 3. Reserve legend strip. Continuous color scales emit a colorbar
    //    instead of categorical entries; both paths consume the same
    //    legend gutter.
    //
    // D13+: Per-chart overrides from `encoding.color.legend` extra fields:
    //   labelFontSize  → overrides theme.label_font_size for entries/ticks
    //   direction      → overrides theme.legend_direction for categorical layout
    //   type="gradient"→ force colorbar even when categorical entries exist
    //   type="symbol"  → force categorical legend even when colorbar available
    //   tickCount      → subsample colorbar ticks to at most N
    //   values         → replace auto-generated tick labels
    //   gradientLength / gradientThickness → colorbar bar dimensions
    let effective_label_font_size = legend_overrides.label_font_size.unwrap_or(theme.typography.label_font_size);
    let effective_direction = legend_overrides.direction.or(theme.legend.legend_direction);
    let force_colorbar = legend_overrides.legend_type.as_deref() == Some("gradient");
    let force_symbol   = legend_overrides.legend_type.as_deref() == Some("symbol");
    let use_colorbar   = (colorbar.is_some() && legend_entries.is_empty() && !force_symbol)
        || (colorbar.is_some() && force_colorbar);

    let (legend_layout, inner_after_legend) = if use_colorbar {
        let cb = colorbar.unwrap();
        let tick_labels: Vec<String> =
            legend_overrides.values.clone().unwrap_or_else(|| cb.tick_labels.clone());
        let tick_labels = if let Some(n) = legend_overrides.tick_count {
            legend::subsample_tick_labels(tick_labels, n)
        } else {
            tick_labels
        };
        // B5 unit 3: thin colorbar ticks to honor `tick_min_step`. Needs the
        // numeric domain the labels span (carried on `ColorbarInput.domain`);
        // absent for explicit non-numeric label overrides → no-op.
        let tick_labels = match (legend_overrides.tick_min_step, cb.domain) {
            (Some(min_step), Some(domain)) =>
                legend::thin_colorbar_ticks_by_min_step(tick_labels, domain, min_step),
            _ => tick_labels,
        };
        legend::layout_colorbar(
            inner,
            theme.legend.legend_orient,
            legend_title.clone(),
            cb.stops.clone(),
            tick_labels,
            effective_label_font_size,
            theme.typography.legend_title_font_size,
            metrics,
            theme.padding.column_padding,
            legend_overrides.gradient_length,
            legend_overrides.gradient_thickness,
            legend_overrides.clip_height,
            legend_overrides.label_color.clone(),
            legend_overrides.label_font_size,
        )
    } else {
        legend::layout_legend(
            legend_entries,
            theme.legend.legend_orient,
            inner,
            effective_label_font_size,
            metrics,
            effective_direction,
            legend_title.as_deref(),
            theme.typography.legend_title_font_size,
            theme.legend.legend_columns,
            legend_overrides.symbol_type.as_deref(),
            legend::LegendStyleOpts {
                symbol_stroke_width: legend_overrides.symbol_stroke_width,
                row_padding: legend_overrides.row_padding,
                column_padding: legend_overrides.column_padding,
                label_limit: legend_overrides.label_limit,
                clip_height: legend_overrides.clip_height,
                padding: legend_overrides.padding,
                title_padding: legend_overrides.title_padding,
                offset: legend_overrides.offset,
                symbol_size: legend_overrides.symbol_size,
                label_color: legend_overrides.label_color.clone(),
                label_font_size: legend_overrides.label_font_size,
            },
        )
    };
    let legend_dropped = legend_entries
        .len()
        .saturating_sub(legend_layout.as_ref().map_or(0, |l| l.entries.len()))
        as u32;

    // 3b. Auxiliary (size / shape) legends, stacked beneath the color legend in
    //     the same gutter. When the widest aux block exceeds the color block,
    //     `layout_aux_legends` shrinks `inner_after_legend` further. Empty for
    //     color-only charts, so this is a no-op (plot region unchanged).
    let (aux_legends, inner_after_legend) = legend::layout_aux_legends(
        aux_legend_inputs,
        legend_layout.as_ref(),
        theme.legend.legend_orient,
        inner,
        inner_after_legend,
        effective_label_font_size,
        theme.typography.legend_title_font_size,
        metrics,
        theme.padding.column_padding,
    );

    // 4 + 5. Reserve y-axis title gutter + label band; reserve x-axis label band.
    let y_title_gutter = axis::compute_y_title_width(
        &axes.y,
        theme.typography.title_font_size,
        theme.padding.axis_title_padding,
        metrics,
    );
    // Per-axis label_font_size override must drive the band reservation too (it
    // already drives the rendered label size in marks/axis.rs); otherwise the
    // gutter is sized at the theme value but drawn at the override → mis-sized.
    let y_label_font_size = axes
        .y
        .overrides
        .label_font_size
        .unwrap_or(theme.typography.label_font_size);
    let y_label_band = axis::compute_y_label_band_width(&axes.y, y_label_font_size, metrics);

    // Rotation-aware bottom margin estimate (spec §4.8). Compute the probable
    // angle the cascade will choose by running a lightweight worst-case check
    // against an estimated slot width, before the plot rect is finalized.
    // Over-reservation is preferable to under-reservation (label clipping).
    let x_label_band = if axes.show_x {
        let estimated_plot_w = inner_after_legend.w - y_label_band - y_title_gutter;
        let n_labels = axes.x.tick_labels.len().max(1);
        let estimated_slot_w = estimated_plot_w / n_labels as f64;
        let x_label_font_size = axes
            .x
            .overrides
            .label_font_size
            .unwrap_or(theme.typography.label_font_size);
        axis::estimate_x_label_band(
            &axes.x.tick_labels,
            x_label_font_size,
            axes.x.overrides.label_angle,
            metrics,
            estimated_slot_w,
            axes.x.overrides.label_padding,
            theme.sizes.tick_size,
        )
    } else {
        0.0
    };

    // L-3: use per-axis title_font_size/title_padding overrides for gutter
    // reservation, matching the y-axis pattern in compute_y_title_width.
    // Previously used theme.title_font_size unconditionally, which caused the
    // gutter to be undersized when axes.x.title_font_size was larger than theme.
    let x_title_gutter = if axes.x.title.is_some() {
        let effective_title_font_size = axes.x.overrides.title_font_size.unwrap_or(theme.typography.title_font_size);
        let effective_title_padding = axes.x.overrides.title_padding.unwrap_or(theme.padding.axis_title_padding);
        metrics.line_height(effective_title_font_size) + effective_title_padding
    } else {
        0.0
    };

    // Reserved band totals per axis (label band + title gutter). The orphan
    // `min_extent`/`max_extent` overrides (B5) clamp each total to `[min, max]`
    // after the dynamic estimate: `min` reserves at least that much, `max` caps
    // it (labels may clip past the cap — allowed). `None`/unset leaves the
    // dynamic value unchanged, so default output is byte-identical.
    let x_band = clamp_axis_extent(
        x_label_band + x_title_gutter,
        axes.x.overrides.min_extent,
        axes.x.overrides.max_extent,
    );
    let y_band = clamp_axis_extent(
        y_label_band + y_title_gutter,
        axes.y.overrides.min_extent,
        axes.y.overrides.max_extent,
    );

    // Orient (B5): reserve each axis's band on its chosen side. x defaults to the
    // bottom (Bottom orient) but moves to the top for `orient="top"`; y defaults
    // to the left (Left orient) but moves to the right for `orient="right"`. The
    // cross-dimension case is rejected upstream (`prepare.rs`), so only the two
    // valid sides per dimension occur here.
    let x_on_top = matches!(axes.x.orient, AxisOrient::Top);
    let y_on_right = matches!(axes.y.orient, AxisOrient::Right);
    let plot_region = inner_after_legend.shrink(Inset {
        top: if x_on_top { x_band } else { 0.0 },
        right: if y_on_right { y_band } else { 0.0 },
        bottom: if x_on_top { 0.0 } else { x_band },
        left: if y_on_right { 0.0 } else { y_band },
    });

    // 6. Split into facet cells (or a single panel).
    let mut panels: Vec<PanelLayout> = Vec::new();
    let mut warnings: Vec<LayoutWarning> = Vec::new();
    if legend_dropped > 0 {
        warnings.push(LayoutWarning::LegendOverflowed { entries_dropped: legend_dropped });
    }

    // panel_rects: (grid_row, grid_col, cell_rect, col_facet_key, row_facet_key)
    // row_facet_key is Some only in grid mode (FacetSpec.row is set).
    let panel_rects: Vec<(u32, u32, Rect, Option<FacetKey>, Option<FacetKey>)> = if let Some(facet) = &spec.facet {
        let n_panels = facet_groups.len() as u32;
        let (gx, gy) = facet
            .spacing
            .map(|s| (s, s))
            .unwrap_or((theme.padding.column_padding, theme.padding.row_padding));
        // When there are multiple rows of facet panels and the x-axis is
        // visible, the inter-row gutter must accommodate x-axis tick labels
        // so non-bottom-row labels are not clipped by the next row's panel.
        let effective_nrows = match facet.mode {
            FacetMode::Wrap { ncols } => {
                let nc = ncols.max(1);
                (n_panels + nc - 1) / nc
            }
            FacetMode::Grid { nrows, .. } => nrows.max(1),
        };
        let gy = if effective_nrows > 1 && axes.show_x {
            gy + x_label_band
        } else {
            gy
        };
        // Grid mode with a row dimension: reserve a right-side strip band for
        // row headers (ggplot2 / Altair convention — row strips on the right).
        // Reserving on the right keeps the y-axis title and tick labels
        // unobstructed on the left. The band width equals one line of
        // strip text plus vertical padding on each side.
        let row_strip_width = if facet.row.is_some() {
            let text_h = metrics.line_height(theme.sizes.strip_text_size);
            text_h + 2.0 * theme.padding.strip_padding
        } else {
            0.0
        };
        let grid_region = if row_strip_width > 0.0 {
            Rect {
                x: plot_region.x,
                y: plot_region.y,
                w: (plot_region.w - row_strip_width).max(0.0),
                h: plot_region.h,
            }
        } else {
            plot_region
        };
        let grid = match facet.mode {
            FacetMode::Wrap { ncols } => {
                facet::FacetGrid::compute_wrap(ncols, n_panels, grid_region, gx, gy)
            }
            FacetMode::Grid { nrows, ncols } => {
                facet::FacetGrid::compute_grid(nrows, ncols, n_panels, grid_region, gx, gy)
            }
        };
        if grid.dropped_count() > 0 {
            let dropped = grid.dropped_count();
            // Collect the key strings for the panels that were cut off.
            // `panel_positions()` returns only the first `nrows*ncols` panels;
            // the remaining `facet_groups` entries (beyond that cap) are dropped.
            let cap = facet_groups.len().saturating_sub(dropped as usize);
            let dropped_keys: Vec<String> = facet_groups[cap..]
                .iter()
                .map(|g| format!("{}={}", g.key.field, g.key.value))
                .collect();
            warnings.push(LayoutWarning::PanelsDropped { count: dropped, keys: dropped_keys });
        }
        // Detect empty cells in the observed cartesian product of distinct
        // row × col values (two-way grid mode only). `group_rows_by_two_fields`
        // emits one `FacetGroup` per (row_val, col_val) pair — including pairs
        // with no data rows (`n_rows == 0`). Those are genuine empty cells that
        // the user may not have intended. Emit one aggregated warning listing
        // all of them so the user can diagnose the data gap.
        //
        // Scope: only fires when `facet.row` is set (two-way grid mode) and
        // at least one group has `n_rows == 0`. Wrap mode and single-field
        // facets never produce empty groups (partition_batch_by_field only
        // yields observed values), so this check is harmless for them but the
        // `facet.row.is_some()` guard makes the intent explicit.
        if facet.row.is_some() {
            let empty_keys: Vec<String> = facet_groups
                .iter()
                .filter(|g| g.n_rows == 0)
                .map(|g| {
                    // Format: "<col_field>=<col_val>, <row_field>=<row_val>"
                    // Both field names and values are present so the user can
                    // identify which data combination is missing.
                    let row_part = g.row_key.as_ref()
                        .map(|rk| format!(", {}={}", rk.field, rk.value))
                        .unwrap_or_default();
                    format!("{}={}{}", g.key.field, g.key.value, row_part)
                })
                .collect();
            if !empty_keys.is_empty() {
                warnings.push(LayoutWarning::EmptyPartitions { keys: empty_keys });
            }
        }
        grid.panel_positions()
            .into_iter()
            .enumerate()
            .map(|(i, (row, col))| {
                let rect = grid.cell_rect(row, col);
                let key = facet_groups.get(i).map(|g| g.key.clone());
                let row_key = facet_groups.get(i).and_then(|g| g.row_key.clone());
                (row, col, rect, key, row_key)
            })
            .collect()
    } else {
        vec![(0, 0, plot_region, None, None)]
    };

    let strip_band_height = if spec.facet.is_some() {
        metrics.line_height(theme.sizes.strip_text_size) + 2.0 * theme.padding.strip_padding
    } else {
        0.0
    };

    // Compute the maximum row index so non-bottom panels can suppress the
    // x-axis title (only the bottom row needs it — duplicating it in every
    // inter-row gutter is visually noisy).
    let max_row = panel_rects.iter().map(|(r, _, _, _, _)| *r).max().unwrap_or(0);
    // Compute the minimum column index so non-leftmost panels can suppress the
    // y-axis title (only the leftmost column needs it — duplicating it on every
    // panel in a multi-column faceted chart causes visual overlap).
    let min_col = panel_rects.iter().map(|(_, c, _, _, _)| *c).min().unwrap_or(0);

    // Width of the row-header strip on the right side of the grid region (grid
    // mode only). Mirrored from the reservation made when building panel_rects.
    let row_strip_band_width = if spec.facet.as_ref().is_some_and(|f| f.row.is_some()) {
        let text_h = metrics.line_height(theme.sizes.strip_text_size);
        text_h + 2.0 * theme.padding.strip_padding
    } else {
        0.0
    };
    // Maximum column index across all panels. Used to identify the rightmost
    // column so each row-header strip is emitted once, on the right edge.
    // Two-way grid facets always form a proper rectangular grid, so max_col is
    // uniform across all rows.
    let max_col = panel_rects.iter().map(|(_, c, _, _, _)| *c).max().unwrap_or(0);
    // Track which grid rows have already had their row-header strip emitted. In
    // a 3×3 grid, row_val "r1" appears for three panels (cols c1, c2, c3); we
    // only want one row-header strip per grid row (the rightmost col).
    let mut emitted_row_strips: std::collections::HashSet<String> = std::collections::HashSet::new();

    // 7. Per-panel: clamp degenerate rects, collect axes.
    let mut axis_layouts: Vec<AxisLayout> = Vec::new();
    for (panel_index, (row, col, mut rect, facet_key, row_facet_key)) in panel_rects.into_iter().enumerate() {
        if rect.w <= MIN_PANEL_DIM || rect.h <= MIN_PANEL_DIM {
            warnings.push(LayoutWarning::PanelCollapsed { panel_index });
            rect = Rect::ZERO;
        }

        // Column header strip (top of each panel). Reserved from the top of
        // the cell rect — behavior unchanged from single-field faceting.
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
                        strip_rect.y + theme.padding.strip_padding + theme.sizes.strip_text_size,
                    ),
                    align: TextAnchor::Middle,
                    font_size: theme.sizes.strip_text_size,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Row header strip (right side, grid mode only). One strip per grid row,
        // emitted on the rightmost panel of each row. The strip sits in the
        // row_strip_band_width reservation made when building panel_rects,
        // which was carved from the right side of the grid region so it never
        // overlaps the y-axis title or tick labels on the left.
        let row_strip_title = if let Some(rk) = &row_facet_key {
            // Emit on the rightmost column only (col == max_col). Use the
            // HashSet to guard against degenerate cases where max_col panels
            // are repeated (shouldn't happen in a proper grid, but be safe).
            let is_rightmost = col == max_col;
            let not_yet_emitted = is_rightmost && emitted_row_strips.insert(rk.value.clone());
            if not_yet_emitted && rect != Rect::ZERO && row_strip_band_width > 0.0 {
                // Place the row-header strip to the right of the rightmost panel,
                // centered within the reserved band.
                let strip_center_x = rect.x + rect.w + row_strip_band_width / 2.0;
                let strip_center_y = rect.y + rect.h / 2.0;
                Some(StripTitleLayout {
                    text: rk.value.clone(),
                    anchor: (strip_center_x, strip_center_y),
                    align: TextAnchor::Middle,
                    font_size: theme.sizes.strip_text_size,
                })
            } else {
                None
            }
        } else {
            None
        };

        // CoordFixed: shrink the binding dimension so w/h == ratio, center.
        if let Some(crate::spec::coord::CoordKind::Fixed { ratio, .. }) = &spec.coord {
            if rect != Rect::ZERO && *ratio > 0.0 {
                let current_ratio = rect.w / rect.h;
                if current_ratio > *ratio {
                    // Too wide — shrink width.
                    let new_w = rect.h * ratio;
                    let dx = (rect.w - new_w) / 2.0;
                    rect = Rect { x: rect.x + dx, y: rect.y, w: new_w, h: rect.h };
                } else {
                    // Too tall — shrink height.
                    let new_h = rect.w / ratio;
                    let dy = (rect.h - new_h) / 2.0;
                    rect = Rect { x: rect.x, y: rect.y + dy, w: rect.w, h: new_h };
                }
            }
        }

        panels.push(PanelLayout {
            plot_area: rect,
            facet_key,
            row,
            col,
            strip_title,
            row_strip_title,
            row_facet_key,
        });

        if rect != Rect::ZERO {
            // Spec-level axis suppression: when `chart.axis(y=False)` is
            // active, skip emitting the y axis layout entirely. The plot
            // area is unchanged (gutters remain reserved upstream); this
            // simply omits axis line + ticks + labels + title.
            if axes.show_y {
                // Suppress y-axis title on non-leftmost-column facet panels to
                // avoid duplicating the title in every inter-column gutter —
                // only the leftmost column's title is needed.
                let y_input = if col > min_col && spec.facet.is_some() {
                    let mut modified = axes.y.clone();
                    modified.title = None;
                    modified
                } else {
                    axes.y.clone()
                };
                let y_label_fs = y_input
                    .overrides
                    .label_font_size
                    .unwrap_or(theme.typography.label_font_size);
                let y_axis = axis::layout_y_axis(
                    &y_input,
                    rect,
                    panel_index,
                    y_label_fs,
                    theme.typography.title_font_size,
                    theme.padding.axis_title_padding,
                    metrics,
                );
                axis_layouts.push(y_axis);
            }

            if axes.show_x {
                // Suppress x-axis title on non-bottom-row facet panels to
                // avoid duplicating "Feature value" (or similar) in every
                // inter-row gutter — only the bottom row's title is needed.
                let x_input = if row < max_row && spec.facet.is_some() {
                    let mut modified = axes.x.clone();
                    modified.title = None;
                    modified
                } else {
                    axes.x.clone()
                };
                let x_label_fs = x_input
                    .overrides
                    .label_font_size
                    .unwrap_or(theme.typography.label_font_size);
                let (x_axis, xwarn) = axis::layout_x_axis(
                    &x_input,
                    rect,
                    panel_index,
                    x_label_fs,
                    theme.typography.title_font_size,
                    theme.padding.axis_title_padding,
                    theme.cull_threshold,
                    theme.sizes.tick_size,
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
    }

    Ok(LayoutResult {
        viewport: viewport_rect,
        panels,
        axes: axis_layouts,
        legend: legend_layout,
        aux_legends,
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
            aux_legends: vec![],
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
            LayoutWarning::PanelsDropped { count: 1, keys: vec!["col_cat=c2".into()] },
            LayoutWarning::EmptyPartitions { keys: vec!["col_cat=c2, row_cat=r2".into()] },
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
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        }
    }

    fn dummy_axes() -> AxesInput {
        AxesInput {
            x: AxisInput::new(
                AxisOrient::Bottom,
                None,
                vec!["0".into(), "1".into(), "2".into(), "3".into()],
                None,
            ),
            y: AxisInput::new(
                AxisOrient::Left,
                None,
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            show_x: true,
            show_y: true,
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
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
    fn per_axis_label_font_size_override_widens_reserved_y_band() {
        // A large per-axis `label_font_size` override must drive the y-label band
        // reservation (not just the rendered text). With metrics whose width
        // scales with font size, a bigger override → wider band → narrower plot.
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        // Font-size-sensitive metrics: width scales with the supplied font size.
        let m = MockMetrics {
            measure: |t: &str, fs: f64| t.chars().count() as f64 * fs * 0.6,
            line_h_factor: 1.2,
        };

        let baseline = compute_layout(
            &spec, &default_theme_inputs(), viewport, &dummy_axes(),
            &[], &[], None, None, &m,
            &legend::LegendOverrides::default(), &[],
        )
        .expect("baseline layout");

        let mut axes_big = dummy_axes();
        axes_big.y.overrides.label_font_size = Some(40.0);
        let widened = compute_layout(
            &spec, &default_theme_inputs(), viewport, &axes_big,
            &[], &[], None, None, &m,
            &legend::LegendOverrides::default(), &[],
        )
        .expect("widened layout");

        let base_w = baseline.panels[0].plot_area.w;
        let wide_w = widened.panels[0].plot_area.w;
        assert!(
            wide_w < base_w,
            "large per-axis label_font_size must widen the y-label band, shrinking \
             the plot ({wide_w} should be < {base_w})",
        );
    }

    // ── B5 unit 2: clamp_axis_extent + orient band reservation ──────────────

    #[test]
    fn clamp_axis_extent_passthrough_when_unset() {
        // Default path (both None) must return the dynamic value unchanged so
        // existing layouts stay byte-identical.
        assert_eq!(clamp_axis_extent(37.5, None, None), 37.5);
    }

    #[test]
    fn clamp_axis_extent_min_reserves_at_least() {
        assert_eq!(clamp_axis_extent(20.0, Some(80.0), None), 80.0);
        // Already above min: unchanged.
        assert_eq!(clamp_axis_extent(120.0, Some(80.0), None), 120.0);
    }

    #[test]
    fn clamp_axis_extent_max_caps() {
        assert_eq!(clamp_axis_extent(120.0, None, Some(40.0)), 40.0);
        // Already below max: unchanged.
        assert_eq!(clamp_axis_extent(20.0, None, Some(40.0)), 20.0);
    }

    #[test]
    fn clamp_axis_extent_max_wins_over_contradictory_min() {
        // min > max is a user contradiction; the cap (max) wins.
        assert_eq!(clamp_axis_extent(50.0, Some(90.0), Some(40.0)), 40.0);
    }

    #[test]
    fn min_extent_reserves_larger_left_band() {
        // A y-axis min_extent of 200px must push the plot area right by at least
        // that much vs. the unset baseline.
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let baseline = compute_layout(
            &spec, &default_theme_inputs(), viewport, &dummy_axes(),
            &[], &[], None, None, &m, &legend::LegendOverrides::default(), &[],
        ).unwrap();

        let mut axes = dummy_axes();
        axes.y.overrides.min_extent = Some(200.0);
        let widened = compute_layout(
            &spec, &default_theme_inputs(), viewport, &axes,
            &[], &[], None, None, &m, &legend::LegendOverrides::default(), &[],
        ).unwrap();

        let base_x = baseline.panels[0].plot_area.x;
        let wide_x = widened.panels[0].plot_area.x;
        assert!(
            wide_x >= base_x + 100.0,
            "min_extent=200 must reserve a much larger left band: base x={base_x}, widened x={wide_x}"
        );
        // The reserved left band is at least min_extent.
        assert!(wide_x - viewport.into_rect().x >= 200.0);
    }

    #[test]
    fn x_orient_top_reserves_band_above_plot() {
        // orient="top" must reserve the x band on the TOP and free up the bottom,
        // mirroring the default Bottom layout. Compare against the Bottom baseline.
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let mut bottom_axes = dummy_axes();
        bottom_axes.x.title = Some("x title".into());
        let baseline = compute_layout(
            &spec, &default_theme_inputs(), viewport, &bottom_axes,
            &[], &[], None, None, &m, &legend::LegendOverrides::default(), &[],
        ).unwrap();
        let base_plot = baseline.panels[0].plot_area;

        let mut top_axes = dummy_axes();
        top_axes.x.title = Some("x title".into());
        top_axes.x.orient = AxisOrient::Top;
        let result = compute_layout(
            &spec, &default_theme_inputs(), viewport, &top_axes,
            &[], &[], None, None, &m, &legend::LegendOverrides::default(), &[],
        ).unwrap();
        let plot = result.panels[0].plot_area;

        // The reserved band moved from the bottom to the top: the top axis layout
        // has a larger plot.y (band reserved above) and its plot bottom reaches
        // lower than the bottom-oriented baseline's plot bottom (no bottom band).
        assert!(
            plot.y > base_plot.y + 5.0,
            "top-oriented x axis must reserve a top band: base y={}, top y={}",
            base_plot.y, plot.y
        );
        assert!(
            plot.y + plot.h > base_plot.y + base_plot.h + 5.0,
            "top-oriented x axis must free the bottom: base bottom={}, top bottom={}",
            base_plot.y + base_plot.h, plot.y + plot.h
        );
        // The emitted x axis line sits at the plot top.
        let x_axis = result.axes.iter().find(|a| a.orient == AxisOrient::Top).unwrap();
        assert!((x_axis.axis_line.y - plot.y).abs() < 0.01);
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
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
        let theme = ThemeInputs { padding: ThemePadding { padding: 100.0, ..ThemePadding::default() }, ..ThemeInputs::default() };
        let err = compute_layout(
            &spec,
            &theme,
            Viewport { width: 50.0, height: 50.0 },
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, result);
    }

    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::layout::panel::FacetKey;

    fn faceted_spec(ncols: u32) -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "species".into(),
            row: None,
            mode: FacetMode::Wrap { ncols },
            spacing: None,
            resolve: FacetResolve::default(),
        });
        s
    }

    fn three_groups() -> Vec<FacetGroup> {
        vec![
            FacetGroup { key: FacetKey { field: "species".into(), value: "setosa".into() }, n_rows: 50, row_key: None },
            FacetGroup { key: FacetKey { field: "species".into(), value: "versicolor".into() }, n_rows: 50, row_key: None },
            FacetGroup { key: FacetKey { field: "species".into(), value: "virginica".into() }, n_rows: 50, row_key: None },
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
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
            row: None,
            mode: FacetMode::Grid { nrows: 1, ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        assert_eq!(result.panels.len(), 2);
        let dropped = result.warnings.iter().any(|w| matches!(
            w,
            LayoutWarning::PanelsDropped { count: 1, .. }
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        ).unwrap();

        assert_eq!(result.panels.len(), 3);
        for (i, panel) in result.panels.iter().enumerate() {
            let strip = panel.strip_title.as_ref()
                .unwrap_or_else(|| panic!("panel {i} missing strip_title"));
            assert!(!strip.text.is_empty());
            // Themes-T4: strip_text_size default flipped 13.0 → 12.0.
            assert_eq!(strip.font_size, 12.0);
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        ).unwrap();
        assert!(result.panels[0].strip_title.is_none());
    }

    #[test]
    fn faceted_2x2_emits_one_y_title_per_row() {
        // 2x2 Grid facet with 4 groups. Axes have titles on both x and y.
        // Expected: 4 panels, 8 total axis layouts (4x y + 4x x).
        // - y-axis title suppressed on non-leftmost columns → 2 y titles (col 0 only).
        // - x-axis title suppressed on non-bottom rows → 2 x titles (row 1 only).
        // Total titled axes = 4 out of 8.
        let mut spec = minimal_chart_spec();
        spec.facet = Some(FacetSpec {
            field: "group".into(),
            row: None,
            mode: FacetMode::Grid { nrows: 2, ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        });

        let groups: Vec<FacetGroup> = ["a", "b", "c", "d"]
            .iter()
            .map(|v| FacetGroup {
                key: FacetKey { field: "group".into(), value: v.to_string() },
                n_rows: 10,
                row_key: None,
            })
            .collect();

        let axes = AxesInput {
            x: AxisInput::new(
                AxisOrient::Bottom,
                Some("x_title".into()),
                vec!["0".into(), "1".into()],
                None,
            ),
            y: AxisInput::new(
                AxisOrient::Left,
                Some("y_title".into()),
                vec!["0".into(), "5".into()],
                None,
            ),
            show_x: true,
            show_y: true,
        };
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 600.0 },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        assert_eq!(result.panels.len(), 4, "expected 4 panels in a 2x2 grid");
        assert_eq!(result.axes.len(), 8, "expected 8 axis layouts (2 per panel)");

        let y_axes: Vec<&AxisLayout> = result
            .axes
            .iter()
            .filter(|a| a.orient == AxisOrient::Left)
            .collect();
        let y_with_title = y_axes.iter().filter(|a| a.title.is_some()).count();
        assert_eq!(
            y_with_title, 2,
            "expected 2 y-axis titles (one per row, leftmost column only); got {y_with_title}"
        );

        let x_axes: Vec<&AxisLayout> = result
            .axes
            .iter()
            .filter(|a| a.orient == AxisOrient::Bottom)
            .collect();
        let x_with_title = x_axes.iter().filter(|a| a.title.is_some()).count();
        assert_eq!(
            x_with_title, 2,
            "expected 2 x-axis titles (one per column, bottom row only); got {x_with_title}"
        );
    }

    #[test]
    fn compute_layout_rotated_labels_have_larger_bottom_margin_than_flat() {
        // A chart with long labels (forced to rotate via override) must reserve
        // a taller bottom band than a chart with short flat labels. We verify
        // this by comparing the bottom edge of the plot area — a larger bottom
        // margin pushes the plot area top-of-bottom-gutter upward, meaning
        // plot_area.y + plot_area.h is smaller relative to the viewport height.
        //
        // Short-label chart: 4 short labels ("A".."D"), no angle override.
        // Long-label chart: same geometry but labels forced to -45° via override.
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        // Short flat labels: "A"=10px fits in slot_w=600/4=150.
        let short_axes = AxesInput {
            x: AxisInput::new(
                AxisOrient::Bottom,
                None,
                vec!["A".into(), "B".into(), "C".into(), "D".into()],
                None,
            ),
            y: AxisInput::new(
                AxisOrient::Left,
                None,
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            show_x: true,
            show_y: true,
        };

        // Long labels with -45° override: "ABCDEFGHIJ"=100px. Angle override forces
        // margin = 100*sin(45°) + line_h*cos(45°) ≈ 70.7 + 9.3 = 80.
        let long_axes = AxesInput {
            x: AxisInput::new(
                AxisOrient::Bottom,
                None,
                vec!["ABCDEFGHIJ".into(), "KLMNOPQRST".into(), "UVWXYZABCD".into(), "EFGHIJKLMN".into()],
                Some(-45.0),
            ),
            y: AxisInput::new(
                AxisOrient::Left,
                None,
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            show_x: true,
            show_y: true,
        };

        let short_result = compute_layout(
            &spec, &default_theme_inputs(), viewport,
            &short_axes, &[], &[], None, None, &m,
            &legend::LegendOverrides::default(),
            &[],
        ).unwrap();

        let long_result = compute_layout(
            &spec, &default_theme_inputs(), viewport,
            &long_axes, &[], &[], None, None, &m,
            &legend::LegendOverrides::default(),
            &[],
        ).unwrap();

        let short_bottom = short_result.panels[0].plot_area.y + short_result.panels[0].plot_area.h;
        let long_bottom = long_result.panels[0].plot_area.y + long_result.panels[0].plot_area.h;

        assert!(
            long_bottom < short_bottom,
            "rotated labels (-45°) must consume more bottom margin than flat labels; \
             long_bottom={long_bottom:.1} should be less than short_bottom={short_bottom:.1}"
        );
    }

    #[test]
    fn compute_layout_show_x_false_bottom_margin_is_zero() {
        // When show_x=false, the x_label_band must be 0 so no bottom space
        // is reserved for labels. We verify by checking that the plot area
        // bottom edge equals the inner_after_legend bottom edge (no gutter).
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let axes_no_x = AxesInput {
            x: AxisInput::new(
                AxisOrient::Bottom,
                None,
                vec!["ABCDEFGHIJ".into(), "KLMNOPQRST".into()],
                None,
            ),
            y: AxisInput::new(AxisOrient::Left, None, vec!["0".into(), "5".into()], None),
            show_x: false,
            show_y: false,
        };
        let axes_with_x = AxesInput {
            show_x: true,
            ..axes_no_x.clone()
        };

        let no_x_result = compute_layout(
            &spec, &default_theme_inputs(), viewport,
            &axes_no_x, &[], &[], None, None, &m,
            &legend::LegendOverrides::default(),
            &[],
        ).unwrap();
        let with_x_result = compute_layout(
            &spec, &default_theme_inputs(), viewport,
            &axes_with_x, &[], &[], None, None, &m,
            &legend::LegendOverrides::default(),
            &[],
        ).unwrap();

        let no_x_bottom = no_x_result.panels[0].plot_area.y + no_x_result.panels[0].plot_area.h;
        let with_x_bottom = with_x_result.panels[0].plot_area.y + with_x_result.panels[0].plot_area.h;

        assert!(
            no_x_bottom > with_x_bottom,
            "show_x=false should reserve less bottom space (larger bottom edge); \
             no_x={no_x_bottom:.1}, with_x={with_x_bottom:.1}"
        );
    }

    #[test]
    fn theme_inputs_default_includes_render_fields() {
        // Paper Ink default identity (2026-05-12).
        let t = ThemeInputs::default();
        assert_eq!(t.padding.padding, 16.0);
        assert_eq!(t.padding.column_padding, 12.0);
        assert_eq!(t.padding.row_padding, 12.0);
        assert_eq!(t.typography.label_font_size, DEFAULT_LABEL_FONT_SIZE);
        assert_eq!(t.sizes.point_size, 36.0);
        assert_eq!(t.sizes.point_size_min, 4.0);
        assert_eq!(t.sizes.point_size_max, 36.0);
        assert_eq!(t.sizes.line_stroke_width, 1.5);
        assert_eq!(t.sizes.bar_corner_radius, 0.0);
        assert_eq!(t.sizes.area_opacity, 0.35);
        assert_eq!(t.sizes.default_opacity, 1.0);
        assert_eq!(t.sizes.axis_line_width, 1.0);
        assert_eq!(t.sizes.tick_size, 4.0);
        assert_eq!(t.sizes.grid_width, 0.5);
        assert_eq!(t.grid.grid, true);
        // Minor level: disabled by default (byte-identical output) with derived
        // lighter/thinner styling so unstyled minors look right when enabled.
        assert_eq!(t.grid.minor, false);
        assert!(t.sizes.minor_grid_width < t.sizes.grid_width);
        assert!(t.grid.minor_grid_opacity < t.grid.grid_opacity);
        assert_eq!(t.sizes.strip_text_size, 12.0);
        assert_eq!(t.padding.strip_padding, 6.0);
        assert_eq!(t.padding.axis_title_padding, 8.0);
        assert_eq!(t.palette.color_scheme, "paper_ink");
        assert_eq!(t.palette.sequential_scheme, "cool_blue");
        assert_eq!(t.palette.diverging_scheme, "blue_to_red");
        assert_eq!(t.typography.title_font_weight, "600");
        assert_eq!(t.typography.title_anchor, TextAnchor::Start);
        assert_eq!(t.typography.title_offset, 6.0);
        assert_eq!(t.colors.background_color, palette::Srgba::new(0xFA, 0xF7, 0xF2, 0xFF));
        assert_eq!(t.colors.mark_color, palette::Srgba::new(0x25, 0x63, 0xEB, 0xFF));
    }

    // ── D5c: empty-partition warning for sparse two-way grid facets ──────────

    /// Build a two-way grid FacetSpec with explicit nrows/ncols that matches
    /// the number of distinct row × col values.
    fn sparse_grid_spec(nrows: u32, ncols: u32) -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "col_cat".into(),
            row: Some("row_cat".into()),
            mode: FacetMode::Grid { nrows, ncols },
            spacing: None,
            resolve: FacetResolve::default(),
        });
        s
    }

    /// Build `FacetGroup` entries for a 2×2 sparse grid where (r2, c2) is empty.
    ///
    /// `group_rows_by_two_fields` produces all four cartesian-product entries
    /// in row-major order, with `n_rows == 0` for the missing (r2, c2) cell.
    fn sparse_2x2_groups() -> Vec<FacetGroup> {
        vec![
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c1".into() },
                n_rows: 1,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r1".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c2".into() },
                n_rows: 1,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r1".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c1".into() },
                n_rows: 1,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r2".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c2".into() },
                n_rows: 0,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r2".into() }),
            },
        ]
    }

    #[test]
    fn compute_layout_sparse_grid_emits_empty_partitions_warning() {
        // A 2×2 two-way grid facet where (r2, c2) has no data must emit
        // exactly one EmptyPartitions warning identifying the missing cell.
        let spec = sparse_grid_spec(2, 2);
        let groups = sparse_2x2_groups();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 600.0 },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        // Exactly one EmptyPartitions warning.
        let empty_warns: Vec<&LayoutWarning> = result
            .warnings
            .iter()
            .filter(|w| matches!(w, LayoutWarning::EmptyPartitions { .. }))
            .collect();
        assert_eq!(
            empty_warns.len(), 1,
            "expected exactly one EmptyPartitions warning; got {:?}",
            result.warnings
        );

        // The warning identifies both the col and row key values.
        if let LayoutWarning::EmptyPartitions { keys } = empty_warns[0] {
            assert_eq!(keys.len(), 1, "expected one empty key entry");
            let key = &keys[0];
            assert!(
                key.contains("c2"),
                "empty-partition key must contain col value 'c2'; got '{key}'"
            );
            assert!(
                key.contains("r2"),
                "empty-partition key must contain row value 'r2'; got '{key}'"
            );
        }

        // No PanelsDropped (the 2×2 grid fits all 4 groups — including the empty one).
        let dropped = result.warnings.iter().any(|w| matches!(w, LayoutWarning::PanelsDropped { .. }));
        assert!(!dropped, "sparse grid must not emit PanelsDropped when grid is correctly sized");
    }

    #[test]
    fn compute_layout_complete_grid_emits_no_empty_partitions_warning() {
        // A 2×2 two-way grid facet where all four cells have data must not
        // emit any EmptyPartitions warning.
        let spec = sparse_grid_spec(2, 2);
        // All four groups have n_rows > 0.
        let complete_groups: Vec<FacetGroup> = vec![
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c1".into() },
                n_rows: 2,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r1".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c2".into() },
                n_rows: 2,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r1".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c1".into() },
                n_rows: 2,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r2".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "c2".into() },
                n_rows: 2,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "r2".into() }),
            },
        ];
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 600.0 },
            &axes,
            &complete_groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        let empty_warns: Vec<&LayoutWarning> = result
            .warnings
            .iter()
            .filter(|w| matches!(w, LayoutWarning::EmptyPartitions { .. }))
            .collect();
        assert!(
            empty_warns.is_empty(),
            "complete grid must emit no EmptyPartitions warning; got {:?}",
            result.warnings
        );
    }

    #[test]
    fn compute_layout_wrap_mode_emits_no_empty_partitions_warning() {
        // Wrap-mode facets never have empty cells (they only group observed values),
        // so no EmptyPartitions warning must be emitted even when n_rows differs.
        let spec = faceted_spec(3); // wrap mode, ncols=3
        let groups = three_groups(); // all three groups have n_rows > 0
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
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        let empty_warns: Vec<&LayoutWarning> = result
            .warnings
            .iter()
            .filter(|w| matches!(w, LayoutWarning::EmptyPartitions { .. }))
            .collect();
        assert!(
            empty_warns.is_empty(),
            "wrap-mode facet must never emit EmptyPartitions; got {:?}",
            result.warnings
        );
    }

    // ── Row-strip right-side placement (layout regression) ───────────────────

    /// Helper: two-way grid spec for row/col faceting.
    fn two_way_grid_spec() -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "col_cat".into(),
            row: Some("row_cat".into()),
            mode: FacetMode::Grid { nrows: 2, ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        });
        s
    }

    /// Helper: 2×2 two-way grid facet groups with data in every cell.
    fn two_way_2x2_groups() -> Vec<FacetGroup> {
        vec![
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "A".into() },
                n_rows: 10,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "High".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "B".into() },
                n_rows: 10,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "High".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "A".into() },
                n_rows: 10,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "Low".into() }),
            },
            FacetGroup {
                key: FacetKey { field: "col_cat".into(), value: "B".into() },
                n_rows: 10,
                row_key: Some(FacetKey { field: "row_cat".into(), value: "Low".into() }),
            },
        ]
    }

    /// The row-strip label must be placed to the RIGHT of the rightmost panel,
    /// not to the left where it would collide with y-axis ticks/title.
    ///
    /// Specifically:
    ///  - Each row produces exactly one `row_strip_title` (on the rightmost col panel).
    ///  - The strip anchor x must be strictly greater than all panel plot_area right edges.
    ///  - The strip anchor x must be strictly greater than the y-axis label band right edge
    ///    (i.e. `plot_region.x`), confirming no left-side placement.
    #[test]
    fn row_strip_anchor_is_right_of_panels_not_left() {
        let spec = two_way_grid_spec();
        let groups = two_way_2x2_groups();
        let axes = AxesInput {
            x: AxisInput::new(AxisOrient::Bottom, Some("x".into()), vec!["0".into(), "5".into()], None),
            y: AxisInput::new(AxisOrient::Left, Some("y".into()), vec!["0".into(), "5".into()], None),
            show_x: true,
            show_y: true,
        };
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 600.0 },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        assert_eq!(result.panels.len(), 4, "expected 4 panels in 2×2 grid");

        // Collect panels that have a row_strip_title.
        let stripped_panels: Vec<&PanelLayout> = result
            .panels
            .iter()
            .filter(|p| p.row_strip_title.is_some())
            .collect();
        assert_eq!(stripped_panels.len(), 2, "expected one row-strip per row (2 rows)");

        for panel in &stripped_panels {
            let strip = panel.row_strip_title.as_ref().unwrap();
            let panel_right = panel.plot_area.x + panel.plot_area.w;

            // Strip anchor must be to the RIGHT of the panel's right edge.
            assert!(
                strip.anchor.0 > panel_right,
                "row strip anchor.x ({:.1}) must be right of panel right edge ({:.1})",
                strip.anchor.0,
                panel_right
            );

            // Strip anchor must not be to the left of any panel's right edge
            // (verifies it's not placed in the left margin near the y-axis).
            let min_panel_right = result
                .panels
                .iter()
                .map(|p| p.plot_area.x + p.plot_area.w)
                .fold(f64::INFINITY, f64::min);
            assert!(
                strip.anchor.0 >= min_panel_right,
                "row strip anchor.x ({:.1}) must not be left of any panel right edge ({:.1})",
                strip.anchor.0,
                min_panel_right
            );
        }

        // Verify no row_strip label is emitted on the leftmost column (col 0 panels
        // should have row_strip_title == None in a 2-column right-side layout).
        let left_col_panels: Vec<&PanelLayout> = result.panels.iter().filter(|p| p.col == 0).collect();
        for p in left_col_panels {
            assert!(
                p.row_strip_title.is_none(),
                "col-0 panel should have no row_strip_title (strip emitted on rightmost col only)"
            );
        }
    }

    /// Smoke test: a two-way grid facet chart renders without panicking and
    /// produces valid strip titles (both col strips on top and row strips on right).
    #[test]
    fn two_way_facet_grid_renders_both_strip_kinds() {
        let spec = two_way_grid_spec();
        let groups = two_way_2x2_groups();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 600.0 },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
        )
        .unwrap();

        // Column strips: all 4 panels have a col strip_title.
        assert!(
            result.panels.iter().all(|p| p.strip_title.is_some()),
            "all panels should have a column strip_title"
        );

        // Row strips: exactly 2 panels (rightmost column) have row_strip_title.
        let row_strip_count = result.panels.iter().filter(|p| p.row_strip_title.is_some()).count();
        assert_eq!(row_strip_count, 2, "expected 2 row-strip panels (one per row, rightmost col)");

        // Row strip text values must match the row_cat values.
        let row_strip_texts: Vec<&str> = result
            .panels
            .iter()
            .filter_map(|p| p.row_strip_title.as_ref())
            .map(|s| s.text.as_str())
            .collect();
        assert!(row_strip_texts.contains(&"High"), "expected 'High' row strip");
        assert!(row_strip_texts.contains(&"Low"), "expected 'Low' row strip");
    }
}
