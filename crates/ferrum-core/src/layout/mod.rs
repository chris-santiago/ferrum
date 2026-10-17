//! Phase 6 — layout engine. Pure function: ChartSpec + Theme + Viewport ->
//! pixel rectangles for panels, axes, legend. No I/O, no rendering, no data
//! values touched. See docs/superpowers/specs/2026-05-09-layout-engine-design.md.

pub(crate) mod axis;
pub(crate) mod binding;
pub(crate) mod facet;
pub(crate) mod geometry;
pub(crate) mod legend;
pub(crate) mod panel;
pub(crate) mod text_metrics;

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

use serde::{Deserialize, Serialize};

pub use self::axis::{
    AxesInput, AxisInput, AxisLayout, AxisOrient, AxisTitleLayout, LabelBandCtx, LabelGapPolicy,
    LabelOverlap, PanelAxisInputs, TickLayout, TickProjection,
};
pub(crate) use self::axis::{AxisStyleOverrides, Claimed, LabelFormat};
pub use self::facet::{FacetGroup, FacetMode, FacetResolve, FacetSpec, ResolveMode};
pub use self::geometry::{Inset, Rect, Viewport};
pub use self::legend::{
    AuxLegendInput, ColorbarInput, ColorbarLayout, ColorbarTick, LegendDirection, LegendEntry,
    LegendEntryLayout, LegendLayout, LegendOrient, LegendOverrides, LegendStyleOpts,
    LegendSuppression, OpacityLegendEntry, OpacityPaint, ShapeLegendEntry, SizeLegendEntry, StrokeDashLegendEntry,
    StrokeWidthLegendEntry, SwatchOpacity, SymbolKind,
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
    /// Secondary y-axis layouts, one per `independent_y` layer per panel,
    /// orient `Right` and stacked outward beyond the primary's band
    /// (secondary-y-axis, GH #52 — see `AxesInput.secondary_y`). Flat across
    /// all panels like `axes`, filtered by `panel_index` at consumption time.
    /// Empty (the default) when the chart has no independent-y layer —
    /// byte-identical to the pre-#52 shared path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub secondary_y_axes: Vec<AxisLayout>,
    /// The region `panels` were split from — `inner` less the chart-title
    /// band, the legend strip, and the axis margin bands (or the region a
    /// composite parent imposed, see [`CompositeLayoutSeam::plot_region`]).
    ///
    /// In-process only: `#[serde(skip)]` keeps `fm.compute_layout`'s returned
    /// dict byte-identical, and no consumer reads this off a *deserialized*
    /// `LayoutResult` — its one reader, the overlay shared-rect pre-pass
    /// (`render::composite_render`), always reads it from a layout it just
    /// computed. A deserialized value is therefore `Rect::ZERO`.
    #[serde(skip)]
    pub(crate) plot_region: Rect,
    /// `viewport` less the theme's padding, BEFORE any chrome band — the box
    /// the chart-title, legend and axis reservations were taken out of.
    ///
    /// Its one reader is the declared-extent-alignment pass
    /// (`render::composite_render`), which needs to tell the two kinds of
    /// inset apart: the gap between `padded_region` and `plot_region` on a
    /// side is chrome that gets DRAWN there, while the gap between
    /// `padded_region` and `viewport` is blank margin. Same in-process-only
    /// contract as `plot_region` above — `#[serde(skip)]` keeps
    /// `fm.compute_layout`'s returned dict byte-identical, and a deserialized
    /// value is `Rect::ZERO`.
    #[serde(skip)]
    pub(crate) padded_region: Rect,
}

/// What a composite parent imposes on ONE leaf's own layout: legend
/// suppression, an imposed plot region, and chart-title suppression. Bundled
/// so a new composite-driven constraint adds a field here instead of another
/// positional parameter on [`compute_layout`], and so all three travel
/// together through the same seam
/// (`render::scale_resolve::LeafScaleContext` → `render::prepare_and_layout`
/// → here).
///
/// `Default` — no suppression of either kind, no imposed region — reproduces
/// a standalone chart's layout byte-for-byte, and is what every non-composite
/// render passes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompositeLayoutSeam {
    /// Per-channel legend suppression: this leaf's color/size legend reserves
    /// no gutter and draws nothing because the composite renders one
    /// figure-level legend for the channel (GH #16, design §6). See
    /// [`LegendSuppression`].
    pub legend: LegendSuppression,
    /// The one plot region every leaf of an all-leaves `Overlay` group lays
    /// out against (GH #89A). `Some` replaces the region this leaf's own
    /// axis-band reservation produced, so every later stage — facet split,
    /// panel clamp, axis tick pixel positions, strip bands — is computed
    /// against the group's shared rect instead of the leaf's own. The
    /// compositor derives it as the intersection of the group leaves'
    /// natural regions (per side, the largest gutter any leaf reserves), so
    /// no leaf's marks can spill across another leaf's legend or axis band.
    pub plot_region: Option<Rect>,
    /// Reserve no chart-title band and lay out no chart title, because the
    /// composite clears this leaf's scene title when it merges (GH #89A). Set
    /// for exactly the non-primary overlay leaves whose chrome is dropped:
    /// their title is never drawn, so its band must not narrow the group's
    /// shared rect. Legends are NOT suppressed this way — they render, so
    /// their gutters keep participating.
    pub suppress_chart_title: bool,
}

/// Clamp a dynamically-estimated axis margin band to the per-axis
/// `min_band`/`max_band` overrides (B5). `min` reserves at least that many
/// px; `max` caps the reservation (labels may clip past the cap — allowed). When
/// both are `None` the dynamic value passes through unchanged, preserving
/// byte-identical default output. A `min > max` (user contradiction) resolves to
/// `max` (the cap wins), matching the "max is a hard ceiling" semantic.
fn clamp_axis_band(dynamic: f64, min_band: Option<f64>, max_band: Option<f64>) -> f64 {
    let mut band = dynamic;
    if let Some(min) = min_band {
        band = band.max(min);
    }
    if let Some(max) = max_band {
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
    PanelCollapsed {
        panel_index: usize,
    },
    /// `axis` is a truthful identity, but its NAMESPACE depends on
    /// `secondary_slot`: when `None`, `axis` is the index this axis was (or
    /// will be) pushed to in the shared `axis_layouts` stream (x and primary-y
    /// axes across all panels share that one vec, so the index is unique
    /// there). When `Some(slot)`, `axis` is instead the 0-based rank of this
    /// secondary-y axis in `secondary_y_axis_layouts` — GH #52's secondary
    /// axes live in a SEPARATE vec, so an `axis_layouts`-namespace index would
    /// be fabricated and could collide with an unrelated x/primary-y warning
    /// in the same panel. `secondary_slot` disambiguates the namespace so the
    /// two integer spaces are never conflated, in the field and in the
    /// rendered message.
    ///
    /// `secondary_slot`'s value is the per-panel y-SLOT index (the same
    /// numbering `build_axis`'s `tick_slot`, `route_panel_axes_and_grid`'s
    /// `y_slot`, `MarkBatch::y_slot`, and `SceneNode::Text.slot` all use, and
    /// the loop-local `slot_idx` this axis was built from) — NOT
    /// `secondary_y_axis_layouts.len()`. In a single-panel chart the two
    /// happen to coincide (each panel's secondary loop runs once), but in a
    /// faceted chart with N panels × 1 secondary axis each, panel *k*'s
    /// secondary axis is slot 0 while its vec rank is *k* — using the vec
    /// rank here would silently misreport the slot to any consumer that reads
    /// the field as what its name says.
    LabelsElided {
        axis: usize,
        count: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        secondary_slot: Option<usize>,
    },
    LegendOverflowed {
        entries_dropped: u32,
    },
    /// One or more facet panels were dropped because the explicit grid
    /// `nrows × ncols` is smaller than the number of facet groups.
    /// `count` is the number of dropped panels; `keys` identifies them by
    /// their facet-group key string (e.g. `"col_cat=c2"`).
    PanelsDropped {
        count: u32,
        keys: Vec<String>,
    },
    /// One or more grid-facet cells in the observed cartesian product of
    /// distinct(row) × distinct(col) values contain no data rows.
    /// `keys` is one entry per empty cell, formatted as
    /// `"<col_field>=<col_val>, <row_field>=<row_val>"` so the user can
    /// identify the missing data combination.  One aggregated warning is
    /// emitted listing all empty cells (spec §4/§11).
    EmptyPartitions {
        keys: Vec<String>,
    },
}

impl std::fmt::Display for LayoutWarning {
    /// User-facing warning text forwarded to Python's ``warnings.warn``.
    ///
    /// These messages are an intentional, stable Display contract — not the
    /// derived Debug of the enum's internal fields. Callers (and the test
    /// suite) may match on the wording below; restructuring a variant's fields
    /// must not change the sentence a user sees unless deliberately revised.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutWarning::PanelCollapsed { panel_index } => write!(
                f,
                "panel {panel_index} collapsed to zero size and was not drawn"
            ),
            LayoutWarning::LabelsElided {
                axis,
                count,
                secondary_slot: None,
            } => write!(
                f,
                "{count} tick label(s) on axis {axis} were elided to avoid overlap"
            ),
            LayoutWarning::LabelsElided {
                axis,
                count,
                secondary_slot: Some(_),
            } => write!(
                f,
                "{count} tick label(s) on secondary y-axis {axis} were elided to avoid overlap"
            ),
            LayoutWarning::LegendOverflowed { entries_dropped } => write!(
                f,
                "legend overflowed; {entries_dropped} entry(ies) were dropped"
            ),
            LayoutWarning::PanelsDropped { count, keys } => write!(
                f,
                "{count} facet panel(s) were dropped because the grid is smaller \
                 than the number of facet groups: {}",
                keys.join("; ")
            ),
            LayoutWarning::EmptyPartitions { keys } => write!(
                f,
                "facet grid has empty cell(s) with no data: {}",
                keys.join("; ")
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutError {
    InvalidViewport {
        width: f64,
        height: f64,
    },
    InvalidFacetSpec(String),
    /// `padding` and `side` are the CALLER'S actual value and the specific
    /// side that overflowed (D10, spec §4.7 — repro:
    /// `configure_padding(top=1e9)` previously reported the theme's default
    /// 16 instead of the caller's 1e9). `side` names one of
    /// `"top"|"right"|"bottom"|"left"` — the one whose resolved inset is
    /// largest on the violated axis (width for left/right, height for
    /// top/bottom), so a caller who only overrode one side always sees that
    /// side named even when the other side of the same axis still carries
    /// the theme default.
    PaddingExceedsViewport {
        padding: f64,
        side: &'static str,
        viewport_dim: f64,
    },
    EmptyFacetGroups,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::InvalidViewport { width, height } => write!(
                f,
                "invalid viewport: width={width}, height={height} (both must be finite and > 0)"
            ),
            LayoutError::InvalidFacetSpec(s) => write!(f, "invalid facet spec: {s}"),
            LayoutError::PaddingExceedsViewport {
                padding,
                side,
                viewport_dim,
            } => write!(
                f,
                "padding {side}={padding} exceeds viewport dimension {viewport_dim}"
            ),
            LayoutError::EmptyFacetGroups => {
                write!(f, "facet specified but facet_groups input is empty")
            }
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
    /// `padding.auto` (D10, spec §4.7, F-L07-08): when `true`, a side left
    /// unset above (`padding_top`/etc. all `None`) is expanded — instead of
    /// falling back to the flat [`padding`](Self::padding) default — to
    /// contain a continuous axis's edge-tick-label overhang and/or recenter
    /// an axis title that would otherwise render past the viewport edge.
    /// Read by [`compute_layout`](super::compute_layout), which composes
    /// every reader's correction SEQUENTIALLY and sums the result per side,
    /// then caps it against the viewport (an earlier
    /// elementwise-max merge was unsound; see the composition note on
    /// `compute_layout`'s own padding step and each reader's own doc for
    /// why); a side the caller DID set is never touched (explicit always
    /// wins — spec §4.7). Built-in default `true` (#135):
    /// a chart that never mentions padding still gets the rescue, so it is
    /// byte-identical to today only for the charts that don't exercise it
    /// (no continuous axis ending in a long tick label or an overflowing
    /// axis title). There is no theme key for this field — it is a Rust
    /// built-in default, not a theme default — so `fm.compute_layout`
    /// (which uses `ThemeInputs::default()`) also runs with auto on.
    pub padding_auto: bool,
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
            // #135: the single owner of the `padding.auto` default.
            // Built-in, not a theme key — `true` so a chart that never
            // mentions padding still gets the edge-padding rescue.
            padding_auto: true,
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
    /// Narrowest width (px) a `stroke_width` bound to a field maps its
    /// smallest value to: the low end of the stroke-width scale's default
    /// range, as `point_size_min` is the size scale's. A built-in default
    /// with no theme key; a `scale=` range on the channel overrides it.
    pub line_stroke_width_min: f64,
    /// Widest width (px) a `stroke_width` bound to a field maps its largest
    /// value to; see [`Self::line_stroke_width_min`].
    pub line_stroke_width_max: f64,
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
            // Vega-Lite's default strokeWidth range, 1 to 4 px.
            line_stroke_width_min: 1.0,
            line_stroke_width_max: 4.0,
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
    /// Subtitle text color. `None` falls back to `font_color` at render time,
    /// preserving the pre-subtitle-config default output.
    pub subtitle_color: Option<palette::Srgba<u8>>,
    pub label_color: palette::Srgba<u8>,
    pub reference_line_color: palette::Srgba<u8>,
}

impl Default for ThemeColors {
    fn default() -> Self {
        let mark_blue = palette::Srgba::new(0x25, 0x63, 0xEB, 0xFF);
        let text_fg = palette::Srgba::new(0x1F, 0x29, 0x37, 0xFF);
        let label_gray = palette::Srgba::new(0x6B, 0x72, 0x80, 0xFF);
        let grid_warm = palette::Srgba::new(0xD6, 0xD3, 0xD1, 0xFF);
        // Minor gridlines are lighter than major by default (derived tint of the
        // warm grid color). Overridable via `minor_grid_color`.
        let grid_minor = palette::Srgba::new(0xE8, 0xE6, 0xE4, 0xFF);
        let bg_cream = palette::Srgba::new(0xFA, 0xF7, 0xF2, 0xFF);
        let strip_bg = palette::Srgba::new(0xED, 0xE9, 0xE3, 0xFF);
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
            subtitle_color: None,
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
    /// Subtitle font size. `None` falls back to `title_font_size * 0.85`,
    /// preserving the pre-subtitle-config default output.
    pub subtitle_font_size: Option<f64>,
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
            subtitle_font_size: None,
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
/// Color fields use palette::Srgba<u8> (the `render::color::Color` alias);
/// values are constructed directly via Srgba::new, or parsed from user strings
/// by `render::color::parse_color` at the binding/config boundaries.
///
/// Fields are organized into logical sub-structs. Accessor methods provide
/// backward-compatible flat access for consumers that prefer `theme.field_name`
/// style.
///
/// `Default` is derived: the Paper Ink default identity (2026-05-12; warm
/// cream background, blue lead mark, warm-tinted grid — see
/// `docs/superpowers/specs/2026-05-12-custom-themes-design.md` §4) lives in
/// each sub-struct's own `Default`, which is where a default value should be
/// read and changed.
#[derive(Debug, Clone, PartialEq, Default)]
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
    /// Minimum pixel gap the x tick-label cascade keeps between adjacent
    /// visible labels, and whether it may drop labels to get it. Set by
    /// `Theme(cull_threshold=)` / `Theme(cull_labels=)`; see
    /// [`LabelGapPolicy`] for the four combinations and for where the
    /// default's value lives.
    pub label_gap: LabelGapPolicy,
}

/// 840: the strip-band size shared by the facet reservation, the per-panel column
/// strip, and the row-header strip — one line of strip text plus vertical padding
/// on each side. Computed once and reused at every strip site.
fn strip_band_size(theme: &ThemeInputs, metrics: &dyn TextMetrics) -> f64 {
    metrics.line_height(theme.sizes.strip_text_size) + 2.0 * theme.padding.strip_padding
}

/// One panel's grid placement before per-panel layout:
/// `(grid_row, grid_col, cell_rect, col_facet_key, row_facet_key)`. `row_facet_key`
/// is `Some` only in two-way grid mode (`FacetSpec.row` set). 400: names the tuple
/// threaded between `split_panels` and `layout_panel_axes`.
type PanelRect = (u32, u32, Rect, Option<FacetKey>, Option<FacetKey>);

/// The chart-level title band's height: title line height, plus a subtitle
/// line height when present, plus the resolved offset. Shared by every
/// consumer that needs this quantity — `reserve_chart_title` (the actual
/// reservation), and `auto_padding_for_edge_ticks`'s and
/// `auto_padding_for_axis_titles`'s top-cushion estimates (padding.auto,
/// D10, spec §4.7) — so hand-copying it at each call site can never let an
/// estimate silently drift from what `reserve_chart_title` actually
/// reserves. Returns `0.0` when the chart has no title.
fn chart_title_band_height(
    spec: &crate::spec::chart::ChartSpec,
    theme: &ThemeInputs,
    metrics: &dyn TextMetrics,
) -> f64 {
    let Some(title_spec) = spec.title.as_ref() else {
        return 0.0;
    };
    let resolved_font_size = title_spec
        .font_size
        .unwrap_or(theme.typography.title_font_size);
    let resolved_offset = title_spec.offset.unwrap_or(theme.typography.title_offset);
    let title_line_h = metrics.line_height(resolved_font_size);
    let subtitle_font_size = title_spec
        .subtitle_font_size
        .or(theme.typography.subtitle_font_size)
        .unwrap_or(resolved_font_size * 0.85);
    let subtitle_line_h = if title_spec.subtitle.is_some() {
        metrics.line_height(subtitle_font_size)
    } else {
        0.0
    };
    title_line_h + subtitle_line_h + resolved_offset
}

/// 400 stage 1 — reserve the chart-level (top-of-SVG) title band off the top of
/// `inner` (Themes-T2.5a; Schwabish SB1 adds the subtitle line). Returns the
/// placed title (or `None`) and the remaining rect. Pure extraction of the
/// former inline block; arithmetic unchanged.
fn reserve_chart_title(
    inner: Rect,
    spec: &crate::spec::chart::ChartSpec,
    theme: &ThemeInputs,
    metrics: &dyn TextMetrics,
) -> (Option<ChartTitleLayout>, Rect) {
    let Some(title_spec) = spec.title.as_ref() else {
        return (None, inner);
    };
    // D1-D6: per-chart TitleSpec overrides for layout geometry.
    let resolved_font_size = title_spec
        .font_size
        .unwrap_or(theme.typography.title_font_size);
    let resolved_anchor = match title_spec.anchor.as_deref() {
        Some("middle") => TextAnchor::Middle,
        Some("end") => TextAnchor::End,
        Some(_) => TextAnchor::Start,
        None => theme.typography.title_anchor,
    };
    let title_line_h = metrics.line_height(resolved_font_size);
    let subtitle_font_size = title_spec
        .subtitle_font_size
        .or(theme.typography.subtitle_font_size)
        .unwrap_or(resolved_font_size * 0.85);
    let subtitle_line_h = if title_spec.subtitle.is_some() {
        metrics.line_height(subtitle_font_size)
    } else {
        0.0
    };
    let band_h = chart_title_band_height(spec, theme, metrics);
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
}

/// 400 stage 2 — reserve the color legend strip (categorical entries or
/// continuous colorbar) plus any stacked auxiliary (size / shape) legend blocks,
/// off `inner`. Returns the color legend, the aux blocks, the plot rect remaining
/// after both reservations, and the number of categorical entries dropped on
/// overflow. Pure extraction of the former inline blocks; arithmetic unchanged.
///
/// `suppression` is the composite-shared-legend seam (design §6): a `true`
/// flag skips that channel's reservation entirely (no gutter, no draw) even
/// though `legend_entries`/`colorbar`/`aux_legend_inputs` are still populated
/// by the caller — the compositor reads them from `PreparedInputs` to build
/// its own figure-level legend. `LegendSuppression::default()` (both `false`)
/// is byte-identical to the pre-suppression behavior.
#[allow(clippy::too_many_arguments)]
fn reserve_legends(
    inner: Rect,
    theme: &ThemeInputs,
    legend_entries: &[LegendEntry],
    legend_title: Option<&str>,
    colorbar: Option<&ColorbarInput>,
    metrics: &dyn TextMetrics,
    legend_overrides: &legend::LegendOverrides,
    aux_legend_inputs: &[legend::AuxLegendInput],
    suppression: legend::LegendSuppression,
) -> (Option<LegendLayout>, Vec<LegendLayout>, Rect, u32) {
    // D13+: Per-chart overrides from `encoding.color.legend` extra fields:
    //   labelFontSize  → overrides theme.label_font_size for entries/ticks
    //   direction      → overrides theme.legend_direction; sizes + arranges a
    //                    categorical legend and transposes a colorbar (D5)
    //   type="gradient"→ force colorbar even when categorical entries exist
    //   type="symbol"  → force categorical legend even when colorbar available
    //   tickCount      → subsample colorbar ticks to at most N
    //   values         → replace auto-generated tick labels
    //   gradientLength / gradientThickness → colorbar bar dimensions
    // `suppression.color`: an empty-entries + no-colorbar input makes the
    // shared dispatch a no-op `(None, inner, ..)` via `layout_legend`'s
    // empty-entries early return — matching the pre-extraction "never
    // attempted layout" skip (a suppressed channel raises no overflow warning
    // below either).
    let (color_entries, color_colorbar): (&[LegendEntry], Option<&ColorbarInput>) =
        if suppression.color {
            (&[], None)
        } else {
            (legend_entries, colorbar)
        };
    let (legend_layout, inner_after_legend, effective_label_font_size) =
        legend::layout_color_legend(
            inner,
            theme.legend.legend_orient,
            theme.typography.label_font_size,
            theme.legend.legend_direction,
            theme.typography.legend_title_font_size,
            theme.legend.legend_columns,
            theme.padding.column_padding,
            color_entries,
            legend_title,
            color_colorbar,
            metrics,
            legend_overrides,
        );
    // A suppressed color legend never attempted layout, so nothing "dropped"
    // — a suppressed channel must not raise the unrelated overflow warning.
    let legend_dropped = if suppression.color {
        0
    } else {
        legend_entries
            .len()
            .saturating_sub(legend_layout.as_ref().map_or(0, |l| l.entries.len())) as u32
    };

    // 3b. Auxiliary (size / shape) legends, stacked beneath the color legend in
    //     the same gutter. When the widest aux block exceeds the color block,
    //     `layout_aux_legends` shrinks `inner_after_legend` further. Empty for
    //     color-only charts, so this is a no-op (plot region unchanged).
    //
    // Suppressed size (design §6 seam): drop the `Size` aux block before
    // layout so it reserves no gutter and draws nothing, while `Shape` blocks
    // (never compositor-suppressed) pass through unchanged.
    let filtered_aux_inputs: Vec<legend::AuxLegendInput>;
    let aux_legend_inputs: &[legend::AuxLegendInput] = if suppression.size {
        filtered_aux_inputs = aux_legend_inputs
            .iter()
            .filter(|a| !matches!(a, legend::AuxLegendInput::Size { .. }))
            .cloned()
            .collect();
        &filtered_aux_inputs
    } else {
        aux_legend_inputs
    };
    let (aux_legends, inner_after_legend) = legend::layout_aux_legends(
        aux_legend_inputs,
        legend_layout.as_ref(),
        theme.legend.legend_orient,
        inner,
        inner_after_legend,
        effective_label_font_size,
        legend_overrides.style.label_font_size,
        theme.typography.legend_title_font_size,
        metrics,
        theme.padding.column_padding,
    );

    (
        legend_layout,
        aux_legends,
        inner_after_legend,
        legend_dropped,
    )
}

/// Stages 2b+3 together: the chart-level title band, then the color legend
/// plus any aux (size/shape) blocks, off `inner`. One function with two
/// callers — the real layout and the chrome measurement below — so a probe
/// of "what these two stages reserve" can never drift from what they
/// actually reserve: both run the identical sequence against the identical
/// stages ([`reserve_chart_title`], [`reserve_legends`]).
#[allow(clippy::too_many_arguments)]
fn reserve_title_and_legends(
    inner: Rect,
    spec: &crate::spec::chart::ChartSpec,
    theme: &ThemeInputs,
    legend_entries: &[LegendEntry],
    legend_title: Option<&str>,
    colorbar: Option<&ColorbarInput>,
    metrics: &dyn TextMetrics,
    legend_overrides: &legend::LegendOverrides,
    aux_legend_inputs: &[legend::AuxLegendInput],
    seam: CompositeLayoutSeam,
) -> (Option<ChartTitleLayout>, Option<LegendLayout>, Vec<LegendLayout>, Rect, u32) {
    let (chart_title_layout, inner) = if seam.suppress_chart_title {
        (None, inner)
    } else {
        reserve_chart_title(inner, spec, theme, metrics)
    };
    let (legend_layout, aux_legends, inner_after_legend, legend_dropped) = reserve_legends(
        inner,
        theme,
        legend_entries,
        legend_title,
        colorbar,
        metrics,
        legend_overrides,
        aux_legend_inputs,
        seam.legend,
    );
    (
        chart_title_layout,
        legend_layout,
        aux_legends,
        inner_after_legend,
        legend_dropped,
    )
}

/// One secondary y-axis's own band width (secondary-y-axis, GH #52): its
/// label band plus title gutter, clamped to its own `min_band`/`max_band`
/// overrides. Extracted out of [`reserve_axis_bands`]'s per-axis map so
/// [`probe_auto_padding`] (#135) can total the same per-axis widths
/// independent of the padding inset, before `reserve_axis_bands` itself
/// ever runs, and the two stages agree because they call this exact same
/// function.
///
/// Drawn gate (#97/#182, same discipline as the primary y band): secondary
/// y-axes have no `.axis(show=False)`-style suppression toggle anywhere in
/// `AxesInput` — the panel loop always emits every `axes.secondary_y` entry
/// ("Independent of axes.show_y" — that toggle only suppresses the
/// primary/left axis). `visible: true` is therefore correct for both the
/// label band and the title gutter under every Cartesian coord, not a
/// placeholder; if a per-secondary-axis show toggle is ever added, thread it
/// through here. `Axis(labels=False)` on a secondary axis IS already
/// honored — `compute_y_label_band_width` reads `a.show_labels` directly off
/// this `AxisInput`.
///
/// Named residual (#182): under a PROJECTED coord (`Polar`/`Geo`)
/// `render::scene_build` suppresses secondary y-axes along with the
/// primaries, so these bands are reserved for nothing. The primaries close
/// via `axes.show_x`/`show_y` (which `render::prepare::build_axes` folds the
/// coord into); the secondary slots cannot, because the coord fact is not
/// per-axis and the honest fix is for the #52 slot plan to produce NO
/// secondary axes under a projected coord — its three index-correlated
/// vectors (inputs/scales/tick_counts) must be emptied together, which is
/// that plan's call, not this reservation's. Reachable only by combining
/// `CoordPolar`/`CoordGeo` with an independent-y layer.
fn secondary_y_axis_band_width(
    a: &AxisInput,
    theme: &ThemeInputs,
    metrics: &dyn TextMetrics,
) -> f64 {
    let label_font_size = a
        .overrides
        .labels
        .font_size
        .unwrap_or(theme.typography.label_font_size);
    let label_band =
        axis::compute_y_label_band_width(a, label_font_size, metrics, a.tick_size(theme), true);
    let title_gutter = axis::compute_y_title_width(
        a,
        theme.typography.title_font_size,
        theme.padding.axis_title_padding,
        metrics,
        true,
    );
    clamp_axis_band(
        label_band + title_gutter,
        a.overrides.placement.min_band,
        a.overrides.placement.max_band,
    )
}

/// The y label band (spec §4.1): the widest any panel's drawn y axis needs
/// ([`AxesInput::drawn_y_inputs`]: the chart's y axis, or each panel's own
/// when y resolves independently per panel), each at its own label font size
/// (a per-axis `label_font_size` drives the drawn size, so it drives the
/// band too).
///
/// Drawn gate (#97, spec §4.1 amended 2026-08-27; widened by #182, which
/// closes #94): `axes.show_y` answers "is this Cartesian y axis drawn" —
/// `.axis(y=False)`'s chart-level toggle (JointChart marginals, ClusterMap
/// dendrograms) AND a projected coord that replaces the Cartesian axes
/// (`render::prepare::build_axes` folds both into the flag). It does NOT
/// empty the tick labels, so the reservation asks the flag itself.
/// `compute_y_label_band_width` also reads `show_labels()` off each input
/// (`fm.Axis(labels=False)` draws no label text either), so only an axis
/// that both is shown AND draws its labels reserves a band at all.
fn y_label_band(axes: &AxesInput, theme: &ThemeInputs, metrics: &dyn TextMetrics) -> f64 {
    axes.drawn_y_inputs()
        .into_iter()
        .map(|y| {
            axis::compute_y_label_band_width(
                y,
                y.overrides.labels.font_size.unwrap_or(theme.typography.label_font_size),
                metrics,
                y.tick_size(theme),
                axes.show_y,
            )
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The x label band for panels `panel_w` wide: the tallest any panel's drawn
/// x axis needs ([`AxesInput::drawn_x_inputs`]: the chart's x axis, or each
/// panel's own when x resolves independently per panel), each reserved for
/// the labels it draws ([`axis::estimate_x_label_band`]) within
/// `band_budget` (see [`LabelBandCtx::band_budget`]).
fn x_label_band(
    axes: &AxesInput,
    panel_w: f64,
    band_budget: f64,
    theme: &ThemeInputs,
    metrics: &dyn TextMetrics,
) -> f64 {
    axes.drawn_x_inputs()
        .into_iter()
        .map(|x| {
            let ctx = LabelBandCtx::for_axis(x, theme, metrics).with_band_budget(band_budget);
            axis::estimate_x_label_band(x, panel_w, &ctx)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The tallest x label band that still leaves every panel drawable, when the
/// band the labels would otherwise draw (`unbounded`) does not; `f64::INFINITY`
/// when the labels never bound the panels (they fit, or the panels collapse
/// even with no band at all). `drawable(b)` says whether a band `b` tall
/// leaves every panel larger than [`MIN_PANEL_DIM`]; it is monotone (a taller
/// band leaves less of the panel), so the bound is found by bisection between a
/// band that is drawable (none) and one that is not (`unbounded`).
fn x_label_band_budget(unbounded: f64, drawable: &dyn Fn(f64) -> bool) -> f64 {
    if drawable(unbounded) || !drawable(0.0) {
        return f64::INFINITY;
    }
    let (mut lo, mut hi) = (0.0, unbounded);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if drawable(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// 400 stage 3 — reserve the x/y axis margin bands off `inner_after_legend`,
/// returning the shrunk plot region, the `x_label_band` (also needed for the
/// inter-row facet gutter), and the per-secondary-y-axis band widths
/// (secondary-y-axis, GH #52; empty when `axes.secondary_y` is empty, the
/// pre-#52 default).
///
/// The x label band and its bound come from [`reserve_x_label_band`], over
/// the panels `geometry` lays out.
///
/// Returns the plot region, the x label band, the secondary-y band widths
/// and the x label band budget (`f64::INFINITY` when unbounded).
fn reserve_axis_bands(
    inner_after_legend: Rect,
    geometry: &PanelGeometry,
    metrics: &dyn TextMetrics,
) -> (Rect, f64, Vec<f64>, f64) {
    let (axes, theme) = (geometry.axes, geometry.theme);
    // Drawn gate (#182, closes #94): the title gutter is reserved only for an
    // axis that draws its title — the same `axes.show_y` answer the label band
    // three lines below already gates on, which also carries the projected-coord
    // case (`Polar`/`Geo` have no Cartesian axes at all; see
    // `spec::coord::suppresses_cartesian_axes`).
    let y_title_gutter = axis::compute_y_title_width(
        &axes.y,
        theme.typography.title_font_size,
        theme.padding.axis_title_padding,
        metrics,
        axes.show_y,
    );
    let y_label_band = y_label_band(axes, theme, metrics);

    // L-3: use per-axis title_font_size/title_padding overrides for gutter
    // reservation, via the `compute_x_title_width` sibling of `compute_y_title_width`
    // (cohesion finding LAYOUT-855 — was inlined here "mirroring the y-axis pattern"
    // to fix an undersizing bug, leaving the axis family asymmetric). Both axes now
    // reserve the gutter through a named helper, so the formula has a single home.
    let x_title_gutter = axis::compute_x_title_width(
        &axes.x,
        theme.typography.title_font_size,
        theme.padding.axis_title_padding,
        metrics,
        axes.show_x,
    );

    // Reserved band totals per axis (label band + title gutter). The orphan
    // `min_band`/`max_band` overrides (B5) clamp each total to `[min, max]`
    // after the dynamic estimate: `min` reserves at least that much, `max` caps
    // it (labels may clip past the cap — allowed). `None`/unset leaves the
    // dynamic value unchanged, so default output is byte-identical.
    let y_band = clamp_axis_band(
        y_label_band + y_title_gutter,
        axes.y.overrides.placement.min_band,
        axes.y.overrides.placement.max_band,
    );

    // Secondary y-axis margin bands (secondary-y-axis, GH #52): one right-side
    // band per `independent_y` layer, stacked outward beyond the primary's own
    // band (spec §6 slot contract — slot 0 stays the primary/left axis
    // regardless of `y_on_right`; slots 1..n always render right). Each band
    // is that axis's own label band + title gutter, honoring its own
    // `label_font_size`/`title_font_size`/`title_padding` overrides exactly
    // like the primary's reservation above, and — mirroring the primary
    // `y_band` clamp above — its own `min_band`/`max_band` overrides too
    // (these were previously silently dropped for secondary
    // axes even though `build_axis_input` populates them per layer). Empty
    // `axes.secondary_y` (the pre-#52 default) makes `secondary_y_total`
    // zero, so the shrink below is a no-op and default output stays
    // byte-identical.
    let secondary_y_bands: Vec<f64> = axes
        .secondary_y
        .iter()
        .map(|a| secondary_y_axis_band_width(a, theme, metrics))
        .collect();
    let secondary_y_total: f64 = secondary_y_bands.iter().sum();

    // The x label band (spec §4.8), for the labels the x axis draws across
    // the panel it gets. The y bands never depend on the x band, so the
    // region's width is known before the x band is.
    let (x_label_band, x_band_budget) = reserve_x_label_band(
        geometry,
        inner_after_legend,
        y_band + secondary_y_total,
        x_title_gutter,
        metrics,
    );
    // Orient (B5): reserve each axis's band on its chosen side. x defaults to the
    // bottom (Bottom orient) but moves to the top for `orient="top"`; y defaults
    // to the left (Left orient) but moves to the right for `orient="right"`. The
    // cross-dimension case is rejected upstream (`prepare.rs`), so only the two
    // valid sides per dimension occur here.
    let x_on_top = matches!(axes.x.orient, AxisOrient::Top);
    let x_band = clamp_axis_band(
        x_label_band + x_title_gutter,
        axes.x.overrides.placement.min_band,
        axes.x.overrides.placement.max_band,
    );
    let y_on_right = matches!(axes.y.orient, AxisOrient::Right);
    let plot_region = inner_after_legend.shrink(Inset {
        top: if x_on_top { x_band } else { 0.0 },
        right: (if y_on_right { y_band } else { 0.0 }) + secondary_y_total,
        bottom: if x_on_top { 0.0 } else { x_band },
        left: if y_on_right { 0.0 } else { y_band },
    });

    (plot_region, x_label_band, secondary_y_bands, x_band_budget)
}

/// The four sides' non-auto fallback padding — an explicit side if the
/// caller set one, else the theme's flat default. This is the "floor"
/// `padding.auto` (D10) never lowers and never needs to reduce: every
/// auto-padding reader starts from it, and `compute_layout`'s own feasibility
/// cap treats it as the space auto must
/// never encroach on, so all three read the identical quantity instead of
/// each re-deriving `padding_<side>.unwrap_or(padding)` by hand.
pub(crate) fn base_padding_insets(theme: &ThemeInputs) -> Inset {
    Inset {
        top: theme.padding.padding_top.unwrap_or(theme.padding.padding),
        right: theme.padding.padding_right.unwrap_or(theme.padding.padding),
        bottom: theme
            .padding
            .padding_bottom
            .unwrap_or(theme.padding.padding),
        left: theme.padding.padding_left.unwrap_or(theme.padding.padding),
    }
}

/// Pre-layout best-estimate of one facet panel's pixel size (M2, spec §4.2 of
/// `2026-09-05-batch-d-transform-engine-design.md`; decision doc's same-dated
/// M2 entry). Computed as the viewport minus [`base_padding_insets`] — the
/// same theme-padding resolution `compute_layout` itself uses — divided
/// per-axis by `grid_dims` (`(cols, rows)`; pass `(1, 1)` for an unfaceted
/// chart).
///
/// This is an ESTIMATE, not the true rendered panel size: true panel size is
/// circular pre-layout (layout measures a panel's plot rect only after axis
/// bands, legends, and facet strip bands are reserved, all of which are
/// unknowable before this pass runs). It deliberately excludes axis/legend/
/// facet-strip gutters. It exists solely to seed
/// [`crate::transform::context::TransformContext::panel_pixel_size`] before
/// transforms run — `compute_layout` never reads it, and it must not be
/// confused with any later, exact panel rect. A second full layout pass could
/// compute the true value but would double the render cost and risk
/// byte-determinism drift; the north star (a post-layout transform phase) is
/// filed, not built (see the decision doc's M2 "Rejected" note).
///
/// **Contract: always returns a dimension ≥ 1 on each axis.** When the
/// theme's resolved padding meets or exceeds the viewport, the true inner
/// extent is zero or negative; rather than propagate that degenerate `0`
/// to callers (a bare zero-sized estimate is a hazard no consumer of
/// `panel_pixel_size` is built to survive — see the `transform::raster`
/// `"screen"` arm's panic before this floor was added), this clamps to a
/// floor of `(1, 1)`. The floor is a deliberately unusable placeholder, not
/// a real panel size: `compute_layout` runs downstream of this estimate on
/// the SAME viewport/theme and independently detects the same
/// padding-exceeds-viewport condition, raising its own typed
/// `LayoutError::PaddingExceedsViewport` before the transform's `(1, 1)`
/// output would ever reach a renderer. The floor exists only to let the
/// transform pass complete without panicking; layout's refusal remains the
/// one and only observable error for this condition.
pub(crate) fn estimate_panel_pixel_size(
    viewport: Viewport,
    theme: &ThemeInputs,
    grid_dims: (u32, u32),
) -> (u32, u32) {
    let pad = base_padding_insets(theme);
    let inner_w = (viewport.width - pad.left - pad.right).max(0.0);
    let inner_h = (viewport.height - pad.top - pad.bottom).max(0.0);
    let (cols, rows) = (grid_dims.0.max(1) as f64, grid_dims.1.max(1) as f64);
    (
        ((inner_w / cols).round() as u32).max(1),
        ((inner_h / rows).round() as u32).max(1),
    )
}

/// The factor in `[0, 1]` that scales one axis's TOTAL `padding.auto`
/// contribution (`desired`, both sides together, after every auto-padding
/// reader has been combined) down to what that axis can spend. Both sides
/// are scaled by the same factor, preserving which side needed more, just
/// less of it. Two bounds apply, and the tighter wins:
/// - the plot: `room`, the margin this axis may add before a panel falls
///   below the share of its `auto=false` size `padding.auto` must leave it
///   ([`AutoPlotRoom`]). A single label or title that needs more than that
///   alone was already dropped as infeasible, so what reaches here is
///   corrections that each fit but together do not.
/// - the viewport: applying the extra can never convert a chart that
///   renders at `auto=false` into a `LayoutError::PaddingExceedsViewport`
///   refusal. `recentering_padding`'s feasible branch is unbounded in
///   isolation (it solves for whatever shift the title needs, with no
///   notion of a shared budget), so without this a large enough extra
///   could collapse the inner rect, and the error would then name a padding
///   value the caller never set, undercutting
///   [`LayoutError::PaddingExceedsViewport`]'s own "report the caller's
///   actual value" contract. The extra must leave [`MIN_PANEL_DIM`] of
///   `span` past `fixed_low`/`fixed_high`, the axis's
///   [`base_padding_insets`] (its `auto=false` inset, which auto never
///   touches). If that floor ALONE already fails, the failure is
///   pre-existing and auto cannot fix it: the factor is `0.0`, so auto
///   contributes nothing and the render fails, or doesn't, exactly as it
///   would without auto, with the same message.
///
/// When `desired` fits both, the factor is exactly `1.0`. A factor below
/// `1.0` is the settle's continuous relaxation, so the fixed point stays
/// within the budget while it iterates; it never ships. It marks the axis
/// ([`AutoPaddingExtra::x_scaled`]), and [`settle_auto_padding`] closes that
/// axis and settles again, so a shipped extra is either the full correction
/// or nothing on that axis.
fn cap_auto_scale(fixed_low: f64, fixed_high: f64, desired: f64, span: f64, room: f64) -> f64 {
    let viewport_budget = span - fixed_low - fixed_high - MIN_PANEL_DIM;
    if viewport_budget <= 0.0 {
        return 0.0;
    }
    let budget = viewport_budget.min(room);
    if desired <= budget {
        return 1.0;
    }
    budget / desired
}

/// The y-axis gutter width `padding.auto` (D10) treats as the "cushion"
/// already available on whichever side the y-axis sits: label band + title
/// gutter, clamped through `clamp_axis_band` exactly like
/// `reserve_axis_bands` clamps its own `y_band` (a fix for the
/// cushion estimate previously using the UNCLAMPED sum, so
/// `fm.Axis(max_band=...)` capping the real gutter smaller than this
/// estimate made the cushion look bigger than it really is, silently
/// under-reserving `extra_left`/`extra_right` below). Shared by
/// `auto_padding_for_edge_ticks` and `auto_padding_for_axis_titles` so the
/// two auto-padding passes can never compute two different notions of "how
/// much of the left/right margin is already spoken for by the y-axis."
fn clamped_y_gutter(theme: &ThemeInputs, axes: &AxesInput, metrics: &dyn TextMetrics) -> f64 {
    let raw = y_label_band(axes, theme, metrics) + axis::compute_y_title_width(
        &axes.y,
        theme.typography.title_font_size,
        theme.padding.axis_title_padding,
        metrics,
        axes.show_y,
    );
    clamp_axis_band(
        raw,
        axes.y.overrides.placement.min_band,
        axes.y.overrides.placement.max_band,
    )
}

/// Everything one leaf's layout is computed from, borrowed once by
/// [`compute_layout`] so the `padding.auto` fixed point
/// ([`settle_auto_padding`], [`probe_auto_padding`]) lays out candidate
/// insets from exactly the inputs the real layout uses.
#[derive(Clone, Copy)]
struct LayoutInputs<'a> {
    spec: &'a crate::spec::chart::ChartSpec,
    theme: &'a ThemeInputs,
    viewport: Viewport,
    axes: &'a AxesInput,
    facet_groups: &'a [FacetGroup],
    legend_entries: &'a [LegendEntry],
    legend_title: Option<&'a str>,
    colorbar: Option<&'a ColorbarInput>,
    metrics: &'a dyn TextMetrics,
    legend_overrides: &'a legend::LegendOverrides,
    aux_legend_inputs: &'a [legend::AuxLegendInput],
    seam: CompositeLayoutSeam,
}

/// How far the drawn x-axis tick labels reach past the plot region's own
/// left and right edges, in the units [`auto_padding_for_edge_ticks`]
/// charges (px past the edge, so the extra is this less the cushion;
/// negative = inside), read off a real layout by [`probe_auto_padding`]:
/// the labels the axis actually draws, after the collision cascade's cull
/// and font shrink, at their drawn angle and anchor, at the tick's real
/// (inset) pixel.
///
/// Every drawn label counts, weighted by its LEVER on each side: how far
/// its tick actually moves per pixel that side's margin widens. A label
/// whose ink overhangs the left canvas edge by `o` needs `o / lever` more
/// left margin, and one past the right edge `o / lever` more right margin.
/// The lever is measured, not assumed: [`probe_auto_padding`] lays the
/// plot out again with each margin [`LEVER_PROBE_PX`] wider and reads how
/// far each tick moved. So it carries the scale's inset arithmetic
/// (`inset_pixel_range` insets the ticks by `padding_frac` of the width,
/// capped at `SCALE_PADDING_MAX_PX`: on an axis wide enough for the cap,
/// the end tick sits a fixed distance from the far edge and the near
/// margin does not move it at all) and the facet split (each panel's tick
/// moves by its share of the region). A side's value is the largest need
/// over every drawn label of every panel.
///
/// Two cases contribute nothing to a side:
/// - the limit, a lever below [`MIN_LABEL_LEVER`]: that margin cannot move
///   the label (a tick pinned at the far end);
/// - an infeasible charge, `o / lever` larger than that side can widen
///   before the label's panel falls below the width [`AutoPlotRoom`] says
///   `padding.auto` must leave it: containing the label would trade the
///   data area for it (or, through [`cap_auto_scale`], squeeze the other
///   side's real extra), so it clips as with `auto=false`. This is the
///   per-label twin of an infeasible axis title (see
///   [`auto_padding_extra`]). It judges the label's own panel width only; a
///   charge that fits there can still leave the plot too small another way
///   (a narrower plot can rotate the labels into a taller band), which
///   [`settle_auto_padding`] catches by never accepting an unusable pass.
#[derive(Debug, Clone, Copy, PartialEq)]
struct XLabelOverhang {
    left: f64,
    right: f64,
}

/// The smallest tick movement per pixel of margin for that margin to
/// rescue a label (see [`XLabelOverhang`]); below it the label sits at the
/// far end of the axis from that margin.
const MIN_LABEL_LEVER: f64 = 1e-9;

/// How much wider [`probe_auto_padding`] lays each margin out to measure
/// the ticks' levers. Tick positions are piecewise linear in the margins,
/// so one pixel reads the slope exactly away from the inset cap's kink.
const LEVER_PROBE_PX: f64 = 1.0;

impl XLabelOverhang {
    /// Nothing measured, or no label drawn: reaches no distance past
    /// either edge, so it charges no edge-label extra.
    const NONE: Self = Self {
        left: f64::NEG_INFINITY,
        right: f64::NEG_INFINITY,
    };

    /// The weighted overhang of the x axes `plot` lays out (one per panel
    /// when faceted) past its plot region, on a `viewport_width`-wide
    /// canvas. `wider_left` / `wider_right` are the same plot laid out with
    /// that margin [`LEVER_PROBE_PX`] wider; a tick missing from one (its
    /// panel collapsed there) has no lever on that side. `room` says how
    /// narrow each panel may get (see [`MarginResponse`]).
    ///
    /// Written in the cushion form [`auto_padding_for_edge_ticks`] expects:
    /// the plot edge's current distance from the canvas edge plus the
    /// label's weighted canvas overhang. With a lever of 1 this is the
    /// label's plain reach past the plot edge.
    fn of_drawn_labels(
        plot: &PlotLayout,
        wider_left: &PlotLayout,
        wider_right: &PlotLayout,
        viewport_width: f64,
        room: &AutoPlotRoom,
        theme: &ThemeInputs,
        metrics: &dyn TextMetrics,
    ) -> Self {
        let region = plot.plot_region;
        let mut overhang = Self::NONE;
        for axis in &plot.axis_layouts {
            let panel_w = plot
                .panels
                .get(axis.panel_index)
                .map_or(0.0, |p| p.plot_area.w);
            let (width_floor, _) = room.floor(axis.panel_index);
            // Widening the left margin moves a tick right (+1), the right
            // margin moves it left (-1).
            let left = MarginResponse::of(axis, panel_w, width_floor, wider_left, 1.0);
            let right = MarginResponse::of(axis, panel_w, width_floor, wider_right, -1.0);
            for label in axis::drawn_x_labels(axis, theme.typography.label_font_size, metrics) {
                if let Some(charge) = left.charge(label.index, label.tick, -label.left_ink) {
                    overhang.left = overhang.left.max(region.x + charge);
                }
                if let Some(charge) =
                    right.charge(label.index, label.tick, label.right_ink - viewport_width)
                {
                    let gap = viewport_width - (region.x + region.w);
                    overhang.right = overhang.right.max(gap + charge);
                }
            }
        }
        overhang
    }

    /// Whether `self` and `other` agree within [`AUTO_PADDING_EPS`] on both
    /// sides (two unmeasured sides agree with each other).
    fn settled_with(self, other: Self) -> bool {
        let agree = |a: f64, b: f64| a == b || (a - b).abs() <= AUTO_PADDING_EPS;
        agree(self.left, other.left) && agree(self.right, other.right)
    }
}

/// How one x axis's ticks and panel respond to one margin widening by
/// [`LEVER_PROBE_PX`], read by comparing the axis with its counterpart in
/// the wider layout (same panel, same orient).
struct MarginResponse<'a> {
    /// The counterpart axis's ticks, or `None` when it has none (its panel
    /// collapsed in the wider layout).
    wider_ticks: Option<&'a [TickLayout]>,
    /// How far the panel can still shrink before it falls below
    /// `width_floor` (or collapses), in pixels of this margin
    /// (`f64::INFINITY` when the margin does not shrink it).
    room: f64,
    /// The direction this margin moves a tick as it widens: `1.0` for the
    /// left margin, `-1.0` for the right.
    direction: f64,
}

impl<'a> MarginResponse<'a> {
    /// `panel_w` is the axis's panel width at the measured inset and
    /// `width_floor` the narrowest [`AutoPlotRoom`] lets that panel get.
    fn of(
        axis: &AxisLayout,
        panel_w: f64,
        width_floor: f64,
        wider: &'a PlotLayout,
        direction: f64,
    ) -> Self {
        let counterpart = wider
            .axis_layouts
            .iter()
            .find(|a| a.panel_index == axis.panel_index && a.orient == axis.orient);
        let wider_w = wider
            .panels
            .get(axis.panel_index)
            .map_or(0.0, |p| p.plot_area.w);
        let shrink_per_px = (panel_w - wider_w) / LEVER_PROBE_PX;
        let room = if shrink_per_px > 0.0 {
            (panel_w - width_floor.max(MIN_PANEL_DIM)).max(0.0) / shrink_per_px
        } else {
            f64::INFINITY
        };
        Self {
            wider_ticks: counterpart.map(|a| a.ticks.as_slice()),
            room,
            direction,
        }
    }

    /// The margin a label needs on this side: its canvas overhang divided
    /// by its tick's lever, or `None` when it contributes nothing (no
    /// counterpart tick, a lever below [`MIN_LABEL_LEVER`], or an
    /// infeasible charge; see [`XLabelOverhang`]). `tick` is the tick's
    /// pixel at the measured inset and `overhang` how far the label's ink
    /// is past this side's canvas edge (negative = inside).
    fn charge(&self, index: usize, tick: f64, overhang: f64) -> Option<f64> {
        let wider = self.wider_ticks?.get(index)?;
        let lever = self.direction * (wider.position - tick) / LEVER_PROBE_PX;
        if lever < MIN_LABEL_LEVER {
            return None;
        }
        let charge = overhang / lever;
        (charge <= self.room).then_some(charge)
    }
}

/// What a candidate `padding.auto` inset actually lays out, measured by
/// running the real layout stages against it ([`reserve_title_and_legends`]
/// then [`lay_out_plot`], the same two calls [`compute_layout`] makes):
///
/// - `chrome`: what the chart title, the color/aux legend gutter (every
///   orient, including an aux block the color legend itself doesn't reach)
///   and the secondary-y bands (always the right side, see
///   [`secondary_y_axis_band_width`]) reserve on each side of the padded
///   rect, BEFORE `padding.auto`'s own tick/title corrections.
///   [`auto_padding_for_edge_ticks`] and [`auto_padding_for_axis_titles`]
///   fold it into their cushion/baseline-center math so a correction never
///   pads MORE than a label's or title's own overhang past everything
///   already reserved.
/// - `x_overhang`: how far the drawn x tick labels reach past the plot
///   region's edges (see [`XLabelOverhang`]).
/// - `unusable`: whether the inset leaves a panel smaller than `room`
///   lets `padding.auto` leave it, or collapsed, or loses more of an x
///   axis's own tick labels than `auto=false` does (see [`AutoPlotRoom`]).
///
/// The first two are only exact at the `inset` that produced them: every
/// categorical legend strip is capped at half of whichever rect it is
/// carved from, and the x-axis collision cascade picks its labels, angle
/// and font size from the plot width. [`settle_auto_padding`] calls this in
/// a loop until what it credits and what this measures agree.
fn probe_auto_padding(
    inset: Inset,
    inputs: &LayoutInputs,
    room: &AutoPlotRoom,
) -> AutoPaddingProbe {
    let LayoutInputs {
        theme,
        viewport,
        axes,
        metrics,
        ..
    } = *inputs;
    let InsetLayout {
        padded,
        after,
        plot,
    } = lay_out_at(inset, inputs);
    let secondary_y_total: f64 = axes
        .secondary_y
        .iter()
        .map(|a| secondary_y_axis_band_width(a, theme, metrics))
        .sum();
    // Clamped at 0 per side: a carve can advance an edge past the padded
    // rect while clamping the OTHER dimension to 0 (a colorbar or aux
    // block wider/taller than the rect it is carved from — see
    // `layout_colorbar`'s and `layout_aux_legends`' own `.max(0.0)`
    // floors), which would otherwise make this side's delta negative. A
    // negative credit is never a real reservation, and the fixed point in
    // `settle_auto_padding` depends on every credit being >= 0.
    let chrome = Inset {
        top: (after.y - padded.y).max(0.0),
        right: ((padded.x + padded.w) - (after.x + after.w)).max(0.0) + secondary_y_total,
        bottom: ((padded.y + padded.h) - (after.y + after.h)).max(0.0),
        left: (after.x - padded.x).max(0.0),
    };
    let unusable = !room.leaves_usable(&plot);
    let x_overhang = if charges_x_edge_labels(axes) {
        let wider_left = lay_out_at(
            Inset {
                left: inset.left + LEVER_PROBE_PX,
                ..inset
            },
            inputs,
        )
        .plot;
        let wider_right = lay_out_at(
            Inset {
                right: inset.right + LEVER_PROBE_PX,
                ..inset
            },
            inputs,
        )
        .plot;
        XLabelOverhang::of_drawn_labels(
            &plot,
            &wider_left,
            &wider_right,
            viewport.width,
            room,
            theme,
            metrics,
        )
    } else {
        XLabelOverhang::NONE
    };
    AutoPaddingProbe {
        chrome,
        x_overhang,
        unusable,
    }
}

/// What [`compute_layout`] lays out at one candidate `inset`: the padded
/// rect, what is left of it after the chart title and legends
/// ([`reserve_title_and_legends`]), and the plot laid out into that
/// ([`lay_out_plot`]).
struct InsetLayout {
    padded: Rect,
    after: Rect,
    plot: PlotLayout,
}

fn lay_out_at(inset: Inset, inputs: &LayoutInputs) -> InsetLayout {
    let padded = inputs.viewport.into_rect().shrink(inset);
    let (_, _, _, after, _) = reserve_title_and_legends(
        padded,
        inputs.spec,
        inputs.theme,
        inputs.legend_entries,
        inputs.legend_title,
        inputs.colorbar,
        inputs.metrics,
        inputs.legend_overrides,
        inputs.aux_legend_inputs,
        inputs.seam,
    );
    let plot = lay_out_plot(
        after,
        inputs.spec,
        inputs.theme,
        inputs.axes,
        inputs.facet_groups,
        inputs.metrics,
        inputs.seam,
    );
    InsetLayout { padded, after, plot }
}

/// Whether `padding.auto` charges for the x axis's edge tick labels at all:
/// only a drawn, labeled, CONTINUOUS x axis (see
/// [`auto_padding_for_edge_ticks`]). [`probe_auto_padding`] skips measuring
/// the labels of a chart this is false for, since nothing would read it.
fn charges_x_edge_labels(axes: &AxesInput) -> bool {
    axes.show_x && axes.x.show_labels() && axes.x.tick_projection.is_some()
}

/// [`probe_auto_padding`]'s measurements at one candidate inset.
/// `unusable` is whether any panel came out at or below [`MIN_PANEL_DIM`]
/// on either dimension (so nothing in it is drawn), or below the share of
/// its `auto=false` size [`AutoPlotRoom`] says `padding.auto` must leave.
#[derive(Debug, Clone, Copy)]
struct AutoPaddingProbe {
    chrome: Inset,
    x_overhang: XLabelOverhang,
    unusable: bool,
}

/// How much of the plot `padding.auto` may spend on its margins: every
/// panel keeps at least [`AUTO_PADDING_MIN_PLOT_SHARE`] of the width and of
/// the height it has with `auto=false`, so auto takes at most that much of
/// the `auto=false` plot across both x sides together, and across both y
/// sides together. A label or title whose rescue would take more is not
/// rescued: that would trade the data area for it, so auto adds nothing for
/// it and it clips exactly as with `auto=false`. Rescues on one axis that
/// each fit but together take more are not rescued either: auto adds
/// nothing on that axis ([`settle_auto_padding`] closes it), so no rescue
/// ships at a size that still clips. The chart is never refused or
/// blanked by this rule, and `auto=false` rendering keeps [`MIN_PANEL_DIM`]
/// as its only floor.
///
/// Measured once per chart from the `auto=false` layout, it is the one rule
/// every place `padding.auto` judges room reads:
/// - [`probe_auto_padding`]: an inset that leaves a panel below its floor
///   (or collapsed), or whose x axes lose more of their own generated tick
///   labels (culled or elided) than the `auto=false` layout loses on the
///   same axis ([`AutoPlotRoom::culls_x_labels`]), is unusable, and
///   [`settle_auto_padding`] never accepts it. The exception is a
///   continuous x axis that labels every `s`-th generated tick from the
///   first, at least [`MIN_EVENLY_THINNED_X_LABELS`] of them and none
///   elided: a sparser, fully labelled axis is not a loss, even where it
///   loses more than `auto=false`;
/// - [`MarginResponse`]: one x label whose charge would push its panel
///   below its width floor is infeasible and contributes nothing;
/// - [`axis_title_infeasibility`]: an axis title whose rescue needs more
///   margin than the axis's budget is infeasible, like one wider than the
///   viewport;
/// - [`cap_auto_scale`]: an axis's combined extra is scaled to its budget,
///   and an axis it scales is closed (its budget 0).
#[derive(Debug, Clone)]
struct AutoPlotRoom {
    /// Per panel, by index: the smallest `(width, height)` auto may leave.
    floors: Vec<(f64, f64)>,
    /// How many of its own tick labels each x axis of the `auto=false`
    /// layout loses ([`x_label_losses`]), in `axis_layouts` order: a pass's
    /// x axis may lose no more.
    x_label_losses: Vec<usize>,
    /// Whether the x axis is continuous, so an even stride thinning of its
    /// labels ([`thins_x_labels_evenly`]) is not a loss.
    x_continuous: bool,
    /// The margin auto may add across both x sides together before a panel
    /// reaches its width floor (`f64::INFINITY` when the margins do not
    /// narrow any panel).
    width_budget: f64,
    /// The same across both y sides, against the height floors.
    height_budget: f64,
}

impl AutoPlotRoom {
    /// Measures the floors and budgets on `inputs` laid out at
    /// `auto_off_inset`, the inset `auto=false` lays the chart out at. Each
    /// budget is read off one more layout with that axis's low margin
    /// [`LEVER_PROBE_PX`] wider: how much a pixel of margin narrows each
    /// panel (all of it on a plain chart, its share of the region when
    /// faceted, half of it beside a capped legend), so the budget is the
    /// margin that takes a panel from its `auto=false` size to its floor.
    fn of_auto_off(inputs: &LayoutInputs, auto_off_inset: Inset) -> Self {
        let off = lay_out_at(auto_off_inset, inputs).plot;
        let wider = lay_out_at(
            Inset {
                left: auto_off_inset.left + LEVER_PROBE_PX,
                ..auto_off_inset
            },
            inputs,
        )
        .plot;
        let taller = lay_out_at(
            Inset {
                top: auto_off_inset.top + LEVER_PROBE_PX,
                ..auto_off_inset
            },
            inputs,
        )
        .plot;
        let floors: Vec<(f64, f64)> = off
            .panels
            .iter()
            .map(|p| {
                (
                    AUTO_PADDING_MIN_PLOT_SHARE * p.plot_area.w,
                    AUTO_PADDING_MIN_PLOT_SHARE * p.plot_area.h,
                )
            })
            .collect();
        // The margin that takes each panel from its `auto=false` extent to
        // its floor, at the measured narrowing per pixel; the first panel
        // to reach its floor bounds the axis.
        let budget = |probe: &PlotLayout, extent: fn(&Rect) -> f64| {
            off.panels
                .iter()
                .enumerate()
                .map(|(i, panel)| {
                    let now = extent(&panel.plot_area);
                    let after = probe.panels.get(i).map_or(0.0, |p| extent(&p.plot_area));
                    let shrink_per_px = (now - after) / LEVER_PROBE_PX;
                    if shrink_per_px > 0.0 {
                        (1.0 - AUTO_PADDING_MIN_PLOT_SHARE) * now / shrink_per_px
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(f64::INFINITY, f64::min)
        };
        let width_budget = budget(&wider, |r| r.w);
        let height_budget = budget(&taller, |r| r.h);
        Self {
            floors,
            x_label_losses: x_label_losses(&off),
            x_continuous: inputs.axes.x.tick_projection.is_some(),
            width_budget,
            height_budget,
        }
    }

    /// This room with the x axis closed when `x` (its budget 0) and the y
    /// axis closed when `y`.
    fn closing(&self, x: bool, y: bool) -> Self {
        Self {
            width_budget: if x { 0.0 } else { self.width_budget },
            height_budget: if y { 0.0 } else { self.height_budget },
            ..self.clone()
        }
    }

    /// Whether the x axis has no room: auto adds nothing left or right, and
    /// [`auto_padding_extra`] drops that axis's tick and title extra before
    /// solving the other axis, as for an infeasible title, so the other
    /// axis is solved against the x margins that ship.
    fn closes_x(&self) -> bool {
        self.width_budget <= 0.0
    }

    /// The same for the y axis.
    fn closes_y(&self) -> bool {
        self.height_budget <= 0.0
    }

    /// Panel `panel_index`'s `(width, height)` floor; `(0.0, 0.0)` for a
    /// panel the `auto=false` layout did not have.
    fn floor(&self, panel_index: usize) -> (f64, f64) {
        self.floors.get(panel_index).copied().unwrap_or((0.0, 0.0))
    }

    /// Whether every panel `plot` lays out is drawable (above
    /// [`MIN_PANEL_DIM`] on both dimensions) and at least its floor, within
    /// [`AUTO_PADDING_EPS`], and its x axes lose no more of their own tick
    /// labels than the `auto=false` layout's ([`Self::culls_x_labels`]).
    fn leaves_usable(&self, plot: &PlotLayout) -> bool {
        let panels_usable = plot.panels.iter().enumerate().all(|(i, p)| {
            let (w_floor, h_floor) = self.floor(i);
            let area = p.plot_area;
            area.w > MIN_PANEL_DIM
                && area.h > MIN_PANEL_DIM
                && area.w >= w_floor - AUTO_PADDING_EPS
                && area.h >= h_floor - AUTO_PADDING_EPS
        });
        panels_usable && !self.culls_x_labels(plot)
    }

    /// Whether `plot` loses more x tick labels than `auto=false` allows: it
    /// lays out a different number of x axes, or one of its x axes loses
    /// more of its own tick labels ([`lost_x_tick_labels`]) than the same
    /// axis of the `auto=false` layout does, unless that axis is continuous
    /// and thins its labels evenly ([`thins_x_labels_evenly`]). The tick
    /// labels do not depend on the inset, so every layout of one chart
    /// generates the same ticks; what a pass can change is which of them it
    /// labels.
    fn culls_x_labels(&self, plot: &PlotLayout) -> bool {
        let axes: Vec<&AxisLayout> = x_axis_layouts(plot).collect();
        axes.len() != self.x_label_losses.len()
            || axes.iter().zip(&self.x_label_losses).any(|(axis, &auto_off)| {
                lost_x_tick_labels(axis) > auto_off && !(self.x_continuous && thins_x_labels_evenly(axis))
            })
    }
}

/// The x axes `plot` lays out (`Bottom` or `Top`), in `axis_layouts` order.
fn x_axis_layouts(plot: &PlotLayout) -> impl Iterator<Item = &AxisLayout> {
    plot.axis_layouts
        .iter()
        .filter(|a| matches!(a.orient, AxisOrient::Bottom | AxisOrient::Top))
}

/// How many of its own tick labels each x axis of `plot` loses
/// ([`lost_x_tick_labels`]), in `axis_layouts` order.
fn x_label_losses(plot: &PlotLayout) -> Vec<usize> {
    x_axis_layouts(plot).map(lost_x_tick_labels).collect()
}

/// How many of `axis`'s ticks with a generated label do not draw it as
/// generated: the label is culled (the tick mark stays) or elided to `"…"`.
/// An axis that draws no labels (`show_labels == false`) loses none.
fn lost_x_tick_labels(axis: &AxisLayout) -> usize {
    if !axis.show_labels {
        return 0;
    }
    axis.ticks
        .iter()
        .filter(|t| !t.label.is_empty() && (t.culled || t.elided))
        .count()
}

/// The fewest labels an evenly thinned x axis draws for
/// [`thins_x_labels_evenly`] to count it as a sparser, fully labelled axis;
/// one thinned to its two ends alone is a loss.
const MIN_EVENLY_THINNED_X_LABELS: usize = 3;

/// Whether `axis` draws its labels as one even stride thinning of its
/// generated ticks (those with non-empty label text, the ticks
/// [`lost_x_tick_labels`] counts): no label is elided, the generated ticks
/// whose label is drawn are exactly the ones [`axis::stride_visibility`]
/// keeps for one stride (every `s`-th, from the first), and at least
/// [`MIN_EVENLY_THINNED_X_LABELS`] labels are drawn. An irregular cull, an
/// elided label or an axis that draws tick marks only is not one.
fn thins_x_labels_evenly(axis: &AxisLayout) -> bool {
    let generated = || axis.ticks.iter().filter(|t| !t.label.is_empty());
    if generated().any(|t| t.elided) {
        return false;
    }
    let drawn: Vec<bool> = generated().map(|t| !t.culled).collect();
    let labelled = drawn.iter().filter(|&&d| d).count();
    let Some(stride) = drawn.iter().skip(1).position(|&d| d).map(|i| i + 1) else {
        return false;
    };
    labelled >= MIN_EVENLY_THINNED_X_LABELS && drawn == axis::stride_visibility(drawn.len(), stride)
}

/// `padding.auto` (D10, spec §4.7, F-L07-08): the extra outer margin needed,
/// per side, to contain a CONTINUOUS axis's drawn tick-label OVERHANG
/// past the plot's own edge. (Axis TITLE overflow is a separate, sibling
/// correction — see [`auto_padding_for_axis_titles`]. `compute_layout`
/// solves THIS pass first, against the untouched base, then solves the
/// title pass against the base this pass leaves behind and sums the two —
/// see the composition note on `compute_layout`'s own padding step for why
/// an earlier elementwise-max merge was unsound.)
///
/// A continuous scale places its ticks across the plot rect's own extent
/// (`AxisInput::tick_projection`), so the end ticks sit at or just inside
/// the plot's own edges, and drawn labels can reach PAST those edges: a
/// flat label by half its width, a rotated one by its rotated reach on the
/// anchored side. The theme's flat default (16px) is
/// not necessarily enough for a long numeric label (`"1,000,000"`), so the
/// label clips past the canvas edge. The same happens in the perpendicular
/// direction for a continuous y-axis's topmost tick (using half a
/// line-height as a conservative, font-agnostic proxy for a label's
/// vertical half-extent, since text baselines aren't exactly centered).
///
/// `x_overhang` is how far the labels the x axis actually DRAWS reach past
/// the plot's edges ([`XLabelOverhang`], measured by
/// [`probe_auto_padding`] on a real layout): the collision cascade's
/// surviving labels at its chosen font size, angle and anchor (or the
/// explicit `label_angle`, on a `Bottom` or `Top` axis alike), at the
/// tick's real pixel, which a scale's `padding_frac` insets from the edge,
/// each label weighted by how far this side's margin moves its tick (see
/// [`XLabelOverhang`] for the weighted rule and its limit). The extra is
/// that reach less the cushion already between the plot edge and the
/// canvas edge, so labels that already fit charge nothing. There is
/// no separate prediction of the angle here: [`settle_auto_padding`]
/// re-measures at each candidate width, because the cascade's choice
/// depends on the width this extra produces.
///
/// Padding is the right lever here specifically because the overhang is
/// anchored relative to the PLOT's own edge, which padding controls
/// directly (shrinking the plot moves its edge — and therefore the tick's
/// pixel position — away from the canvas edge, opening exactly the room
/// the overhang needs).
///
/// Scoped to continuous axes only (`tick_projection.is_some()`): an
/// ordinal/nominal axis places ticks at BAND-SLOT centers, inset from the
/// plot edges by construction, so its labels keep their non-auto layout.
///
/// `chrome` is whatever the chart title, a legend on the same side (left,
/// right or top), or the secondary-y bands already reserve on that side
/// (see [`probe_auto_padding`]), folded into each cushion below exactly
/// like `y_gutter` is: it is space the plot's edge has already moved
/// behind, so the overhang this function corrects for is already that
/// much smaller. `chrome.bottom` is unused here: there is no bottom
/// correction to adjust (see below).
fn auto_padding_for_edge_ticks(
    theme: &ThemeInputs,
    axes: &AxesInput,
    metrics: &dyn TextMetrics,
    chrome: Inset,
    x_overhang: XLabelOverhang,
) -> Inset {
    // The non-auto fallback for each side — mirrors `compute_layout`'s own
    // `inset` computation exactly (an explicit side stays exactly this;
    // auto only ever adds on top of it below).
    let base = base_padding_insets(theme);
    let (base_left, base_right, base_top) = (base.left, base.right, base.top);

    let y_on_right = matches!(axes.y.orient, AxisOrient::Right);
    let y_label_font_size = axes
        .y
        .overrides
        .labels
        .font_size
        .unwrap_or(theme.typography.label_font_size);
    let y_gutter = clamped_y_gutter(theme, axes, metrics);

    let (extra_left, extra_right) = if charges_x_edge_labels(axes) {
        let left_cushion = base_left + chrome.left + if y_on_right { 0.0 } else { y_gutter };
        let right_cushion = base_right + chrome.right + if y_on_right { y_gutter } else { 0.0 };
        (
            (x_overhang.left - left_cushion).max(0.0),
            (x_overhang.right - right_cushion).max(0.0),
        )
    } else {
        (0.0, 0.0)
    };

    // Top only. The bottom cushion is `base_bottom + chrome.bottom`, plus
    // the x band where the x axis sits at the bottom. That is at least half
    // a y label line whenever `base_bottom` is (the theme's default padding
    // exceeds half a line at every default label size), and whenever the
    // bottom x band draws labels (its band holds a full label line). Not
    // covered: a theme `padding` below half a label line with no labels on
    // the bottom band (a tick-only band-budget cull, `orient="top"`, or a
    // hidden x axis), where the lowest y label can overhang the bottom
    // edge. Covering it would need the probe to measure the y labels'
    // vertical overhang the way it measures x's. The top cushion has no
    // such floor when there's no chart title and no top-side chrome, so it
    // is corrected here. `chrome.top` already covers the chart-title band
    // (see `probe_auto_padding`), so it is not added again.
    let y_flat = axes.y.overrides.labels.angle.is_none_or(|a| a == 0.0);
    let y_continuous = axes.y.tick_projection.is_some();
    let extra_top = if axes.show_y
        && axes.y.show_labels()
        && y_continuous
        && y_flat
        && !axes.y.tick_labels.is_empty()
    {
        let half_line = metrics.line_height(y_label_font_size) / 2.0;
        let top_cushion = base_top + chrome.top;
        (half_line - top_cushion).max(0.0)
    } else {
        0.0
    };

    Inset {
        top: extra_top,
        right: extra_right,
        bottom: 0.0,
        left: extra_left,
    }
}

/// The minimal `(extra_low, extra_high)` pair of ADDITIONAL padding needed
/// on the low-coordinate side (left/top) and high-coordinate side
/// (right/bottom) of one axis so a CENTER-anchored element of half-extent
/// `half_extent`, whose baseline center sits at `baseline_center` along a
/// `span`-long viewport axis, no longer overflows either edge.
///
/// Derivation (F-L07-08 — the symmetric-only
/// derivation this replaces incorrectly concluded a centered element could
/// NEVER be rescued by padding; that holds only when both sides expand by
/// the SAME amount): expanding the low side's padding by `x` shifts the
/// element's center by `+x/2`; expanding the high side's by `y` shifts it
/// by `-y/2` (`layout_x_axis`/`layout_y_axis` anchor a title at
/// `panel_area.{x,y} + panel_area.{w,h}/2`, and each unit of low-side
/// padding both pushes the panel's low edge out by 1 and leaves its high
/// edge fixed, so the center moves half as much). Requiring both edges to
/// land within `[0, span]` after a shift `s = (extra_low - extra_high)/2`
/// gives:
///
/// - low edge:  `baseline_center + s - half_extent >= 0`  →  `s >= half_extent - baseline_center`
/// - high edge: `baseline_center + s + half_extent <= span` → `s <= span - baseline_center - half_extent`
///
/// This pair is satisfiable (some shift fixes BOTH edges at once) exactly
/// when the lower bound does not exceed the upper bound, which simplifies
/// to `2*half_extent <= span` — the element is no wider than the viewport
/// axis itself. When satisfiable, the MINIMAL sufficient shift (added to
/// only one side, leaving the other side's `extra` at `0`) is used, so an
/// already-fitting element gets `(0.0, 0.0)`. When not satisfiable — the
/// element is wider than the viewport itself — no padding arrangement can
/// eliminate all clipping (this is the genuine, narrow residual: NOT "any
/// centered element," only one wider than the canvas), and this returns
/// `(0.0, 0.0)` rather than adding unbounded padding that cannot help and
/// would risk turning a "renders with a clipped element" chart into a
/// `PaddingExceedsViewport` refusal.
/// Whether [`recentering_padding`] can eliminate ALL clipping for a
/// center-anchored element of half-extent `half_extent` somewhere within a
/// `span`-long viewport axis — independent of `baseline_center` (the
/// `2*half_extent <= span` bound below the derivation cancels it out; see
/// `recentering_padding`'s own doc for the full derivation). Split out
/// because `recentering_padding`'s own
/// `(extra_low, extra_high)` output is `(0.0, 0.0)` in TWO different
/// situations a caller composing it with another auto-padding pass must
/// tell apart: "already centered, nothing to do" and "genuinely too wide
/// to ever fully fit" (the documented residual). Only the second is a
/// signal that a same-axis pass's own contribution must be suppressed too
/// — see [`axis_title_infeasibility`], the one place that verdict is
/// decided, and `compute_layout`'s use of it.
fn recentering_is_feasible(half_extent: f64, span: f64) -> bool {
    2.0 * half_extent <= span
}

fn recentering_padding(baseline_center: f64, half_extent: f64, span: f64) -> (f64, f64) {
    if !recentering_is_feasible(half_extent, span) {
        return (0.0, 0.0);
    }
    let low_bound = half_extent - baseline_center;
    let high_bound = span - baseline_center - half_extent;
    let shift = if low_bound > 0.0 {
        low_bound
    } else if high_bound < 0.0 {
        high_bound
    } else {
        0.0
    };
    if shift > 0.0 {
        (2.0 * shift, 0.0)
    } else {
        (0.0, -2.0 * shift)
    }
}

/// Half the extent a drawn axis title occupies along its own axis, or `None`
/// when the axis is hidden or untitled: the one measurement both the
/// feasibility verdict ([`axis_title_infeasibility`]) and the recentering
/// solve ([`auto_padding_for_axis_titles`]) read. A y title renders rotated
/// -90°, so its measured (unrotated) width is its vertical extent.
fn title_half_extent(
    axis: &AxisInput,
    shown: bool,
    theme: &ThemeInputs,
    metrics: &dyn TextMetrics,
) -> Option<f64> {
    if !shown {
        return None;
    }
    let text = axis.title.get().as_deref()?;
    let font_size = axis.overrides.title.font_size.unwrap_or(theme.typography.title_font_size);
    Some(metrics.measure_width(text, font_size) / 2.0)
}

/// Whether x's/y's axis-title rescue is infeasible, in either of two ways:
/// - the genuine residual: the title is wider than the viewport axis
///   itself ([`recentering_is_feasible`]), so no padding contains it;
/// - the rescue costs the plot too much: the recentering margin it needs
///   (both sides together) is more than `room`'s budget for that axis, so
///   it would leave a panel below the share of its `auto=false` size
///   `padding.auto` must keep (see [`AutoPlotRoom`]).
///
/// The rescue is solved against `base` and `chrome` alone, without
/// [`auto_padding_for_edge_ticks`]'s contribution: the same title, chrome
/// and `auto=false` geometry always yield the same verdict, whatever padding
/// the tick pass wants. [`auto_padding_extra`] calls this BEFORE composing
/// the tick pass's contribution, so a same-axis tick correction can be
/// suppressed on an infeasible axis before it ever reaches the title solve,
/// not after, once that solve has already run against the unsuppressed
/// geometry.
fn axis_title_infeasibility(
    inputs: &LayoutInputs,
    base: Inset,
    chrome: Inset,
    room: &AutoPlotRoom,
) -> (bool, bool) {
    let LayoutInputs {
        theme,
        viewport,
        axes,
        metrics,
        ..
    } = *inputs;
    let too_wide = |half_extent: Option<f64>, span: f64| {
        half_extent.is_some_and(|half| !recentering_is_feasible(half, span))
    };
    let rescue = auto_padding_for_axis_titles(inputs, base, chrome);
    let x_infeasible = too_wide(
        title_half_extent(&axes.x, axes.show_x, theme, metrics),
        viewport.width,
    ) || rescue.left + rescue.right > room.width_budget;
    let y_infeasible = too_wide(
        title_half_extent(&axes.y, axes.show_y, theme, metrics),
        viewport.height,
    ) || rescue.top + rescue.bottom > room.height_budget;
    (x_infeasible, y_infeasible)
}

/// `padding.auto` (D10, spec §4.7, F-L07-08): the extra outer margin needed,
/// per side, to RECENTER an axis title that would otherwise clip past the
/// viewport edge — the sibling correction to
/// [`auto_padding_for_edge_ticks`] (tick-label overhang). Both axis titles
/// are center-anchored on the plot rect (`layout_x_axis`/`layout_y_axis`'s
/// `anchor_x`/`anchor_y`), which is why a naive SYMMETRIC padding expansion
/// cannot help (it shrinks the plot without moving a centered title's
/// on-screen position — see [`recentering_padding`]'s doc for the full
/// derivation) — but an ASYMMETRIC one can, and does, by shifting the
/// plot's (and therefore the title's) center away from the edge it would
/// otherwise overflow. Only a title wider than the viewport itself is a
/// genuine residual (see [`recentering_padding`]).
///
/// `base` is the outer padding to solve against — NOT re-derived from
/// `theme.padding` here: `compute_layout`
/// passes the padding AFTER `auto_padding_for_edge_ticks`'s own correction
/// is folded in, so this function's solve — the last word on outer
/// padding, since nothing after it adds more — sees the geometry the tick
/// pass actually leaves behind, instead of the tick-naive base. Merging
/// two independently-against-the-untouched-base solutions by elementwise
/// max was unsound here specifically because [`recentering_padding`]
/// solves for a DIFFERENCE between two sides, not an independent minimum
/// per side like the tick pass: a tick-only expansion on one side silently
/// shifts the center this function would otherwise have recentered
/// correctly, re-introducing the exact clip this function exists to
/// remove. Sequential composition (ticks, then titles against the
/// tick-adjusted base) is sound because the tick pass's own guarantee is
/// monotone under later additions on either side — solving titles last,
/// against the settled result, is the only order nothing downstream can
/// invalidate.
///
/// Gated on `axes.show_x`/`axes.show_y` — mirrors
/// `auto_padding_for_edge_ticks`'s own gate: `AxisInput.title` is populated independently of the chart-level
/// show flag, so without this gate a `.axis(x=False)` chart with a title
/// set would still pay for recentering a title `layout_panel_axes` never
/// draws.
///
/// Reuses [`clamped_y_gutter`] (same `max_band`
/// gap as `auto_padding_for_edge_ticks`) for the x-title's left/right
/// baseline-center estimate, and the x-axis's own clamped band (its label
/// band plus title gutter, via the same `clamp_axis_band` `reserve_axis_bands`
/// applies) for the y-title's top/bottom baseline-center estimate. Both
/// estimates fold in `chrome` (see [`probe_auto_padding`]): the chart-title
/// band, the legend gutter and the secondary-y bands, on whichever side
/// they actually land. Ignoring a real right-side legend or secondary-y
/// band would make `baseline_center` LARGER than the true plot center,
/// asking the solver for MORE right padding than actually needed and
/// pushing the title further past its LEFT edge than intended — an
/// over-correction risk on the title's opposite side.
/// `auto_padding_for_edge_ticks`'s own `orient="top"` gap does NOT apply
/// here: this function's y-title estimate computes `x_on_top` and folds
/// the x-band into `plot_y`/`plot_h` on both orientations (see the
/// `x_on_top` branch below), so it is correct for `orient="top"` as
/// written.
///
/// Whether an axis's title is in the genuine residual regime (title wider
/// than the viewport itself) has exactly one home:
/// [`axis_title_infeasibility`]. This function's own `(extra_low,
/// extra_high)` for that axis is `(0.0, 0.0)` either way in that regime
/// ([`recentering_padding`] returns that pair itself when infeasible), so
/// this function has no need to also decide and report the verdict — a
/// caller that needs it calls [`axis_title_infeasibility`] directly.
/// `compute_layout` does exactly that, before calling this function, and
/// zeroes the same-axis tick-pass contribution before `base` ever reaches
/// here.
fn auto_padding_for_axis_titles(inputs: &LayoutInputs, base: Inset, chrome: Inset) -> Inset {
    let LayoutInputs {
        theme,
        viewport,
        axes,
        metrics,
        ..
    } = *inputs;
    let (base_left, base_right, base_top, base_bottom) =
        (base.left, base.right, base.top, base.bottom);

    let y_on_right = matches!(axes.y.orient, AxisOrient::Right);
    let y_gutter = clamped_y_gutter(theme, axes, metrics);
    let plot_w =
        (viewport.width - base_left - base_right - chrome.left - chrome.right - y_gutter).max(0.0);

    let (extra_left, extra_right) = match title_half_extent(&axes.x, axes.show_x, theme, metrics) {
        Some(half_w) => {
            let plot_x = base_left + chrome.left + if y_on_right { 0.0 } else { y_gutter };
            let baseline_center = plot_x + plot_w / 2.0;
            recentering_padding(baseline_center, half_w, viewport.width)
        }
        None => (0.0, 0.0),
    };

    let (extra_top, extra_bottom) = match title_half_extent(&axes.y, axes.show_y, theme, metrics) {
        Some(half_h) => {
            let x_title_gutter = axis::compute_x_title_width(
                &axes.x,
                theme.typography.title_font_size,
                theme.padding.axis_title_padding,
                metrics,
                axes.show_x,
            );
            let (x_label_band, _) = reserve_x_label_band(
                &PanelGeometry::of(inputs),
                viewport.into_rect().shrink(per_side_sum(base, chrome)),
                y_gutter,
                x_title_gutter,
                metrics,
            );
            let x_band = clamp_axis_band(
                x_label_band + x_title_gutter,
                axes.x.overrides.placement.min_band,
                axes.x.overrides.placement.max_band,
            );
            let x_on_top = matches!(axes.x.orient, AxisOrient::Top);
            let plot_y = base_top + chrome.top + if x_on_top { x_band } else { 0.0 };
            let plot_h = (viewport.height
                - base_top
                - base_bottom
                - chrome.top
                - chrome.bottom
                - x_band)
                .max(0.0);
            let baseline_center = plot_y + plot_h / 2.0;
            recentering_padding(baseline_center, half_h, viewport.height)
        }
        None => (0.0, 0.0),
    };

    Inset {
        top: extra_top,
        right: extra_right,
        bottom: extra_bottom,
        left: extra_left,
    }
}

/// Resolves one side's final padding inset: the caller's explicit value if
/// set, else `base` plus whatever `padding.auto` contributed on that side.
/// Shared by `compute_layout`'s own inset and by `settle_auto_padding`'s
/// per-iteration probe, so the two can never land on two different insets
/// for what is supposed to be the same value.
fn resolve_padding_inset(theme: &ThemeInputs, base: Inset, extra: Inset) -> Inset {
    Inset {
        top: theme.padding.padding_top.unwrap_or(base.top + extra.top),
        right: theme
            .padding
            .padding_right
            .unwrap_or(base.right + extra.right),
        bottom: theme
            .padding
            .padding_bottom
            .unwrap_or(base.bottom + extra.bottom),
        left: theme.padding.padding_left.unwrap_or(base.left + extra.left),
    }
}

/// One solve of the tick-then-title composition against a fixed `chrome`
/// credit and a fixed measured `x_overhang` — see `settle_auto_padding`'s
/// own doc for why both are parameters here rather than computed inside.
///
/// The tick pass is solved first, against the untouched `base`, and its
/// own guarantee is monotone under later additions on either side; the
/// title pass is solved LAST, against the tick-adjusted base
/// (`auto_padding_for_axis_titles`'s `base` parameter), so its recentering
/// solve sees the geometry the tick pass actually leaves behind and is the
/// final word nothing downstream perturbs. Merging two independently-
/// against-the-untouched-base solutions by elementwise max — composing
/// SEQUENTIALLY instead — was unsound specifically because the title
/// solver is a function of the DIFFERENCE between its two sides
/// (`recentering_padding`), not an independent per-side minimum like the
/// tick pass: a tick-only expansion on one side silently moved the center
/// the title pass had solved for, re-clipping a title auto claims to
/// protect.
///
/// Per-axis infeasibility (#139(2)) is decided FIRST, via
/// `axis_title_infeasibility`, independent of the tick pass, and the tick
/// pass's own same-axis contribution is zeroed before the title pass ever
/// solves against it. When a title's rescue is INFEASIBLE (the title is
/// wider than the viewport itself, the documented genuine residual, or its
/// rescue would take more of the plot than [`AutoPlotRoom`] allows), its
/// own contribution is `(0.0, 0.0)`, but the tick pass's contribution on
/// that SAME axis is not: being asymmetric by construction, it still
/// shifts the plot's (and title's) center, and the title pass has no way
/// to counteract a shift it declined to solve for — and, left unsuppressed
/// until AFTER the title pass runs, it also leaks into the OTHER axis's
/// own title solve (a continuous x-tick correction moves `plot_w`, which
/// moves the x-axis label band, which moves the y-title's own baseline
/// center). Measured live (pre-fix): a same-axis tick correction alone
/// turned a 9.51px title clip into a 26.53px one — `auto=true` strictly
/// worse than `auto=false`, the exact regression `cap_auto_scale`'s
/// best-effort contract forbids. Suppressing that axis's tick contribution
/// BEFORE the title pass solves — instead of after, against geometry the
/// solve has already consumed — restores `auto=false`'s exact geometry on
/// that axis and stops the leak into a sibling axis's own solve.
///
/// A single x tick label can be infeasible the same way: when the margin
/// it would need on a side is more than that side can widen before the
/// label's panel falls below its [`AutoPlotRoom`] floor, it contributes
/// nothing to that side's tick extra (see [`XLabelOverhang`]), rather than
/// being charged and then capped, which would shrink the plot past that
/// floor or squeeze the other side's extra.
///
/// An explicit side's own extra is zeroed before [`cap_auto_scale`] too
/// (#139(3)): `compute_layout`'s final inset
/// (`theme.padding.padding_<side>.unwrap_or(...)`) discards `extra`
/// entirely on an explicit side, so a real but doomed-to-be-discarded
/// extra must not compete for the cap's shared budget, which would throttle
/// the OTHER (auto) side for a contribution that was never going to ship.
/// For the same reason an explicit side's tick extra is zeroed before the
/// title pass solves: the title is recentred on the plot that ships, not on
/// one narrowed by a tick extra the explicit side discards.
///
/// Returns the extra as laid out together with its tick part as shipped:
/// after the infeasible-title zeroing, the explicit-side zeroing and the
/// cap's scale, the same three steps the whole extra goes through.
/// [`settle_auto_padding`] judges whether a pass's labels are contained by
/// that shipped tick part, so it never counts as granted a tick extra the
/// layout discards.
fn auto_padding_extra(
    inputs: &LayoutInputs,
    chrome: Inset,
    x_overhang: XLabelOverhang,
    room: &AutoPlotRoom,
) -> AutoPaddingExtra {
    let LayoutInputs {
        theme,
        viewport,
        axes,
        metrics,
        ..
    } = *inputs;
    let base = base_padding_insets(theme);
    let (x_infeasible, y_infeasible) = axis_title_infeasibility(inputs, base, chrome, room);
    let x_dropped = x_infeasible || room.closes_x();
    let y_dropped = y_infeasible || room.closes_y();
    let feasible_only = |extra: Inset| Inset {
        top: if y_dropped { 0.0 } else { extra.top },
        right: if x_dropped { 0.0 } else { extra.right },
        bottom: if y_dropped { 0.0 } else { extra.bottom },
        left: if x_dropped { 0.0 } else { extra.left },
    };
    let padding = &theme.padding;
    let auto_sides_only = |extra: Inset| Inset {
        top: if padding.padding_top.is_some() { 0.0 } else { extra.top },
        right: if padding.padding_right.is_some() { 0.0 } else { extra.right },
        bottom: if padding.padding_bottom.is_some() { 0.0 } else { extra.bottom },
        left: if padding.padding_left.is_some() { 0.0 } else { extra.left },
    };
    let ticks = auto_sides_only(feasible_only(auto_padding_for_edge_ticks(
        theme, axes, metrics, chrome, x_overhang,
    )));
    let titles = auto_sides_only(feasible_only(auto_padding_for_axis_titles(
        inputs,
        per_side_sum(base, ticks),
        chrome,
    )));
    let combined = per_side_sum(ticks, titles);
    let x_scale = cap_auto_scale(
        base.left,
        base.right,
        combined.left + combined.right,
        viewport.width,
        room.width_budget,
    );
    let y_scale = cap_auto_scale(
        base.top,
        base.bottom,
        combined.top + combined.bottom,
        viewport.height,
        room.height_budget,
    );
    let capped = |extra: Inset| Inset {
        top: extra.top * y_scale,
        right: extra.right * x_scale,
        bottom: extra.bottom * y_scale,
        left: extra.left * x_scale,
    };
    AutoPaddingExtra {
        total: capped(combined),
        ticks: capped(ticks),
        x_scaled: combined.left + combined.right > 0.0 && x_scale < 1.0,
        y_scaled: combined.top + combined.bottom > 0.0 && y_scale < 1.0,
        no_usable_pass: false,
    }
}

/// What [`auto_padding_extra`] solves for one candidate credit and
/// overhang.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct AutoPaddingExtra {
    /// The extra per side, as laid out.
    total: Inset,
    /// The edge-tick part of `total`, as shipped.
    ticks: Inset,
    /// Whether [`cap_auto_scale`] scaled the x (left and right) extra down,
    /// so what the rescues on that axis need together is more than it may
    /// take ([`settle_auto_padding`] closes such an axis).
    x_scaled: bool,
    /// The same for the y (top and bottom) extra.
    y_scaled: bool,
    /// Whether this is the no-extra fallback of a settle where no pass was
    /// usable (each collapsed a panel or lost more of an x axis's own tick
    /// labels than `auto=false` does, see [`AutoPlotRoom::culls_x_labels`]):
    /// [`settle_auto_padding`] then judges each axis's rescue on
    /// its own.
    no_usable_pass: bool,
}

impl AutoPaddingExtra {
    /// The per-side larger of `self` and `other`, scaled on an axis when
    /// either was.
    fn per_side_max(self, other: Self) -> Self {
        Self {
            total: per_side_max(self.total, other.total),
            ticks: per_side_max(self.ticks, other.ticks),
            x_scaled: self.x_scaled || other.x_scaled,
            y_scaled: self.y_scaled || other.y_scaled,
            no_usable_pass: self.no_usable_pass || other.no_usable_pass,
        }
    }
}

/// Bounded pass limit for `settle_auto_padding`'s fixed point: enough
/// iterations for the capped-legend contraction to converge well past
/// float precision, with a hard stop for any non-contractive layout.
const AUTO_PADDING_MAX_PASSES: u32 = 64;

/// Convergence tolerance for `settle_auto_padding`: comfortably below
/// `render::FLOAT_PRECISION`'s own rounding, so it never surfaces in
/// rendered output.
const AUTO_PADDING_EPS: f64 = 1e-6;

/// The least share of its `auto=false` width and height every panel keeps
/// under `padding.auto`: auto never takes more than half of the plot for
/// its margins (see [`AutoPlotRoom`]).
const AUTO_PADDING_MIN_PLOT_SHARE: f64 = 0.5;

/// Finds a self-consistent `padding.auto` extra. Two things it solves
/// against are themselves a function of how much padding `padding.auto`
/// ends up adding:
///
/// - the chrome credit: the legend gutter, the chart-title band and the
///   secondary-y bands, because every categorical legend strip caps its
///   own width or height at half of whatever rect it is carved from;
/// - the x tick labels' overhang: the collision cascade picks which labels
///   it draws, at what font size and angle, from the plot width, so the
///   reach of the drawn edge labels changes with the extra (a narrower
///   plot can rotate or cull the very label an extra was sized for).
///
/// A single probe at one fixed rect can over-credit the chrome, which clips
/// a label or title past the viewport edge, or size the edge-label extra
/// for labels the narrower plot no longer draws. So this solves both as one
/// fixed point: solve `padding.auto` against a candidate credit and
/// overhang, lay the resulting inset out for real ([`probe_auto_padding`]),
/// and accept when what it measures agrees with what it was solved against
/// within `AUTO_PADDING_EPS`.
///
/// Starts from `credit = { right: secondary_total }`, the one chrome term
/// that never depends on the rect, and an unmeasured overhang (no edge-label
/// extra), so the first pass lays out the labels exactly as `auto=false`
/// draws them; labels that already fit there keep that layout, and the
/// chart is unchanged by `padding.auto`. Iterates up to
/// `AUTO_PADDING_MAX_PASSES` times. A pass COVERS when its plot is usable
/// (no panel collapsed, none below the share of its `auto=false` size
/// [`AutoPlotRoom`] keeps), its measured chrome covers its credit on all
/// four sides (`reserved >= credit - AUTO_PADDING_EPS`), and the labels it
/// measured need no more edge extra than it granted; the last covering pass
/// is remembered. Both sides of that last comparison are the tick extra as
/// shipped ([`AutoPaddingExtra::ticks`]): the tick part of the pass's own
/// extra, and the tick part the same solve gives for the overhang the pass
/// measured. So a side that ships no tick extra (an explicit side, or the
/// axis of an infeasible title) is covered whatever its labels do, and a
/// capped side is judged at the size the cap lets it reach. A collapsed
/// pass draws no labels, so it would otherwise always look covered, and
/// accepting it would blank a chart `auto=false` draws. The ordinary case
/// (no capped legend, labels that keep their presentation) converges in two
/// passes, or a few more when a scale's `padding_frac` inset shrinks with
/// the plot.
///
/// The bound is reached only when the measurement does not settle, which
/// happens when the cascade alternates between passes: flat labels that
/// need an extra, at a width where they no longer stay flat (rotated, they
/// need none, so the next pass is back at the flat width). Then the last
/// covering pass is returned, so this never credits more chrome than some
/// real reservation measured, and never grants less edge extra than the
/// labels it measured needed. If no pass covered, the result is the
/// per-side maximum of the extra each pass's own measurement needed
/// (solved against the smaller of its credit and its measured chrome), so
/// every label state any pass measured stays contained, unless that extra
/// leaves the plot unusable: then the result is no extra at all, and the
/// chart lays out exactly as with `auto=false`. A pass that settles on an
/// unusable plot ends the search the same way. The rule is the one
/// [`XLabelOverhang`] applies to a single infeasible label: an extra that
/// cannot help without taking more of the plot than [`AutoPlotRoom`]
/// allows is dropped, and `padding.auto` never blanks a chart `auto=false`
/// draws.
///
/// The settle runs within an [`AutoPlotRoom`] ([`settle_within`]). When the
/// extra it settles on was scaled down on an axis by [`cap_auto_scale`], the
/// rescues on that axis together need more than the axis may take, and a
/// scaled rescue still clips. So that axis is closed (its budget set to 0,
/// so auto adds nothing on it) and the settle runs again, since what the
/// other axis needs depends on the plot the first one leaves.
///
/// When no pass was usable ([`AutoPaddingExtra::no_usable_pass`]), each
/// axis is judged on its own, x and y alike: its rescue alone is the settle
/// with the other axis closed. An axis whose rescue alone ships nothing (it
/// collapses a panel, loses more of an x axis's own tick labels than
/// `auto=false` does ([`AutoPlotRoom::culls_x_labels`]), or is not
/// needed) is closed, and the settle runs again, so the other axis's valid
/// rescue still ships. When both rescues are valid alone but fail together,
/// neither is closed and auto adds nothing.
///
/// That repeats until the settled extra asks to close no axis the room
/// still has open; each repeat closes an axis, so there are at most three
/// settles of the whole chart ([`AUTO_PADDING_MAX_SETTLES`]), each judging
/// at most one rescue per open axis alone.
fn settle_auto_padding(inputs: &LayoutInputs, base: Inset) -> Inset {
    let mut room = AutoPlotRoom::of_auto_off(
        inputs,
        resolve_padding_inset(inputs.theme, base, Inset::default()),
    );
    let mut settled = settle_within(inputs, base, &room);
    for _ in 1..AUTO_PADDING_MAX_SETTLES {
        let mut close_x = settled.x_scaled && !room.closes_x();
        let mut close_y = settled.y_scaled && !room.closes_y();
        if settled.no_usable_pass {
            let alone = |closing_x, closing_y| settle_within(inputs, base, &room.closing(closing_x, closing_y)).total;
            if !room.closes_x() {
                let x = alone(false, true);
                close_x |= x.left + x.right <= 0.0;
            }
            if !room.closes_y() {
                let y = alone(true, false);
                close_y |= y.top + y.bottom <= 0.0;
            }
        }
        if !close_x && !close_y {
            break;
        }
        room = room.closing(close_x, close_y);
        settled = settle_within(inputs, base, &room);
    }
    settled.total
}

/// A bound on the settles [`settle_auto_padding`] runs: the first, and one
/// more for each axis it closes. Each settle after the first closes an axis
/// that was still open, so there are at most three.
const AUTO_PADDING_MAX_SETTLES: u32 = 3;

/// The fixed point [`settle_auto_padding`] describes, within one `room`: the
/// extra it ships, and whether the cap scaled either axis's extra to reach
/// it (the converged pass's, the last covering pass's, or, for the no-cover
/// fallback, whether any need that fed it was scaled on that axis).
fn settle_within(inputs: &LayoutInputs, base: Inset, room: &AutoPlotRoom) -> AutoPaddingExtra {
    let LayoutInputs {
        theme,
        axes,
        metrics,
        ..
    } = *inputs;
    let solve = |credit: Inset, overhang: XLabelOverhang| {
        auto_padding_extra(inputs, credit, overhang, room)
    };
    let probe = |extra: Inset| {
        probe_auto_padding(resolve_padding_inset(theme, base, extra), inputs, room)
    };
    let secondary_total: f64 = axes
        .secondary_y
        .iter()
        .map(|a| secondary_y_axis_band_width(a, theme, metrics))
        .sum();
    let mut credit = Inset {
        right: secondary_total,
        ..Inset::default()
    };
    let mut overhang = XLabelOverhang::NONE;
    let mut covering_extra = None;
    let mut most_needed = AutoPaddingExtra::default();
    for _ in 0..AUTO_PADDING_MAX_PASSES {
        let extra = solve(credit, overhang);
        let measured = probe(extra.total);
        let reserved = measured.chrome;
        let needed = solve(credit, measured.x_overhang).ticks;
        let covers = !measured.unusable
            && reserved.top >= credit.top - AUTO_PADDING_EPS
            && reserved.right >= credit.right - AUTO_PADDING_EPS
            && reserved.bottom >= credit.bottom - AUTO_PADDING_EPS
            && reserved.left >= credit.left - AUTO_PADDING_EPS
            && needed.left <= extra.ticks.left + AUTO_PADDING_EPS
            && needed.right <= extra.ticks.right + AUTO_PADDING_EPS;
        if covers {
            covering_extra = Some(extra);
        }
        let measured_credit = per_side_min(credit, reserved);
        most_needed = most_needed.per_side_max(solve(measured_credit, measured.x_overhang));
        let converged = (reserved.top - credit.top).abs() <= AUTO_PADDING_EPS
            && (reserved.right - credit.right).abs() <= AUTO_PADDING_EPS
            && (reserved.bottom - credit.bottom).abs() <= AUTO_PADDING_EPS
            && (reserved.left - credit.left).abs() <= AUTO_PADDING_EPS
            && measured.x_overhang.settled_with(overhang);
        if converged {
            if !measured.unusable {
                return extra;
            }
            // Settled on an unusable plot: every later pass would repeat it.
            break;
        }
        credit = reserved;
        overhang = measured.x_overhang;
    }
    match covering_extra {
        Some(extra) => extra,
        None if !probe(most_needed.total).unusable => most_needed,
        None => AutoPaddingExtra {
            no_usable_pass: true,
            ..AutoPaddingExtra::default()
        },
    }
}

/// The sum of `a` and `b` on each side.
fn per_side_sum(a: Inset, b: Inset) -> Inset {
    Inset {
        top: a.top + b.top,
        right: a.right + b.right,
        bottom: a.bottom + b.bottom,
        left: a.left + b.left,
    }
}

/// The smaller of `a` and `b` on each side.
fn per_side_min(a: Inset, b: Inset) -> Inset {
    Inset {
        top: a.top.min(b.top),
        right: a.right.min(b.right),
        bottom: a.bottom.min(b.bottom),
        left: a.left.min(b.left),
    }
}

/// The larger of `a` and `b` on each side.
fn per_side_max(a: Inset, b: Inset) -> Inset {
    Inset {
        top: a.top.max(b.top),
        right: a.right.max(b.right),
        bottom: a.bottom.max(b.bottom),
        left: a.left.max(b.left),
    }
}

/// Stages 4 to 7 of [`compute_layout`] against one chrome-reserved rect
/// (`inner_after_legend`): the axis margin bands, the composite seam's
/// imposed plot region, the facet split, and every panel's strips and axes.
/// One function with two callers, the real layout and
/// [`probe_auto_padding`]'s measurement, so what `padding.auto` measures is
/// what the chart lays out.
fn lay_out_plot(
    inner_after_legend: Rect,
    spec: &crate::spec::chart::ChartSpec,
    theme: &ThemeInputs,
    axes: &AxesInput,
    facet_groups: &[FacetGroup],
    metrics: &dyn TextMetrics,
    seam: CompositeLayoutSeam,
) -> PlotLayout {
    // 4 + 5. Reserve the x/y axis margin bands, yielding the plot region. The
    //    `x_label_band` is also needed for the inter-row facet gutter below;
    //    `secondary_y_bands` (secondary-y-axis, GH #52) is one band width per
    //    `axes.secondary_y` entry, threaded to per-panel placement below.
    let strip_band = strip_band_size(theme, metrics);
    let geometry = PanelGeometry {
        spec,
        facet_groups,
        theme,
        axes,
        strip_band,
        seam,
    };
    let (natural_plot_region, x_label_band, secondary_y_bands, x_band_budget) =
        reserve_axis_bands(inner_after_legend, &geometry, metrics);

    // 5b. Overlay shared-rect imposition (GH #89A). The ONE point where a
    //     composite parent replaces this leaf's own plot region — placed
    //     exactly where that region is born, so every stage after it (facet
    //     split, degenerate-panel clamp, CoordFixed correction, axis tick
    //     pixel positions, strip bands) is computed against the imposed rect
    //     and no layout product is left describing a rect the marks don't
    //     use. That is the whole point of imposing here rather than
    //     overwriting `panels[0].plot_area` after this function returns, which
    //     is what the pre-#89A compositor did: the overwrite left the axis
    //     tick positions, title, and legend placement stale, so the
    //     compositor could only dedup chrome for leaves it could prove the
    //     staleness invisible for.
    let plot_region = seam.plot_region.unwrap_or(natural_plot_region);

    // 6 + 7. Split into facet cells, then lay out each panel's strips + axes.
    //    840: the strip band size, computed once above, is threaded to both
    //    stages.
    let (panel_rects, mut warnings) = split_panels(
        plot_region,
        spec,
        facet_groups,
        theme,
        x_label_band,
        axes.show_x,
        strip_band,
    );
    let (panels, axis_layouts, secondary_y_axes, panel_warnings) = layout_panel_axes(
        panel_rects,
        spec,
        axes,
        theme,
        metrics,
        strip_band,
        &secondary_y_bands,
        x_band_budget,
    );
    warnings.extend(panel_warnings);
    PlotLayout {
        plot_region,
        panels,
        axis_layouts,
        secondary_y_axes,
        warnings,
    }
}

/// `rect` under the chart's `CoordFixed` aspect: the binding dimension
/// shrunk so `w / h == ratio`, centred. Unchanged for any other coord, a
/// non-positive ratio, or a collapsed (`Rect::ZERO`) panel. The one owner of
/// the correction, read by [`layout_panel_axes`] and by
/// [`PanelGeometry::extent`].
fn fixed_aspect(rect: Rect, spec: &crate::spec::chart::ChartSpec) -> Rect {
    let Some(crate::spec::coord::CoordKind::Fixed { ratio, .. }) = &spec.coord else {
        return rect;
    };
    if rect == Rect::ZERO || *ratio <= 0.0 {
        return rect;
    }
    if rect.w / rect.h > *ratio {
        // Too wide: shrink width.
        let new_w = rect.h * ratio;
        Rect {
            x: rect.x + (rect.w - new_w) / 2.0,
            w: new_w,
            ..rect
        }
    } else {
        // Too tall: shrink height.
        let new_h = rect.w / ratio;
        Rect {
            y: rect.y + (rect.h - new_h) / 2.0,
            h: new_h,
            ..rect
        }
    }
}

/// What the plot's panels are laid out from, for sizing the x label band
/// before the panels exist: the facet split, each panel's strip, the
/// `CoordFixed` aspect and a composite's imposed region, exactly as
/// [`lay_out_plot`] applies them.
#[derive(Clone, Copy)]
struct PanelGeometry<'a> {
    spec: &'a crate::spec::chart::ChartSpec,
    facet_groups: &'a [FacetGroup],
    theme: &'a ThemeInputs,
    axes: &'a AxesInput,
    strip_band: f64,
    seam: CompositeLayoutSeam,
}

/// The panel sizes [`PanelGeometry::extent`] reports: the narrowest panel's
/// width before the `CoordFixed` aspect (`pre_aspect_w`) and after it (`w`),
/// and the shortest panel's height after it (`h`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct PanelExtent {
    pre_aspect_w: f64,
    w: f64,
    h: f64,
}

impl<'a> PanelGeometry<'a> {
    fn of(inputs: &LayoutInputs<'a>) -> Self {
        Self {
            spec: inputs.spec,
            facet_groups: inputs.facet_groups,
            theme: inputs.theme,
            axes: inputs.axes,
            strip_band: strip_band_size(inputs.theme, inputs.metrics),
            seam: inputs.seam,
        }
    }

    /// The panels' sizes when the plot region is `natural_region` (or the
    /// region a composite imposes instead) and the x label band is
    /// `x_label_band` tall (a facet grid's row gutters hold that band): each
    /// facet panel's strip taken off, then the `CoordFixed` aspect
    /// ([`fixed_aspect`]). The widths never depend on the band unless the
    /// aspect binds the width.
    fn extent(&self, natural_region: Rect, x_label_band: f64) -> PanelExtent {
        let region = self.seam.plot_region.unwrap_or(natural_region);
        let cells: Vec<Rect> = if self.spec.facet.is_none() {
            vec![region]
        } else {
            split_panels(
                region,
                self.spec,
                self.facet_groups,
                self.theme,
                x_label_band,
                self.axes.show_x,
                self.strip_band,
            )
            .0
            .into_iter()
            .map(|(_, _, cell, _, _)| Rect {
                h: cell.h - self.strip_band,
                ..cell
            })
            .collect()
        };
        if cells.is_empty() {
            return PanelExtent { pre_aspect_w: region.w, w: region.w, h: region.h };
        }
        let mut extent = PanelExtent {
            pre_aspect_w: f64::INFINITY,
            w: f64::INFINITY,
            h: f64::INFINITY,
        };
        for cell in cells {
            let aspect = fixed_aspect(Rect { h: cell.h.max(0.0), ..cell }, self.spec);
            let h = if aspect.h == cell.h.max(0.0) { cell.h } else { aspect.h };
            extent.pre_aspect_w = extent.pre_aspect_w.min(cell.w);
            extent.w = extent.w.min(aspect.w);
            extent.h = extent.h.min(h);
        }
        extent
    }
}

/// A bound on the passes [`reserve_x_label_band`]'s `CoordFixed` fixed point
/// takes. The pass sequence is non-increasing over the finitely many bands a
/// drawn presentation can have, so it stops long before this; the bound is
/// defensive only.
const X_BAND_FIXED_POINT_PASSES: u32 = 32;

/// The x label band, and the bound the x labels are laid out within, for a
/// plot region taken out of `inner_after_legend` beside `y_bands_w` of y
/// bands (the primary and any secondary), with the x title's gutter
/// `x_title_gutter` beside the labels: the one reservation every reader of
/// the x band makes (`reserve_axis_bands` and the `padding.auto` title pass).
///
/// 1. The labels each x axis draws at the panel width ([`x_label_band`]).
/// 2. When that band leaves no panel drawable, it is bounded
///    ([`x_label_band_budget`]) and the labels are laid out within the bound.
/// 3. When the `CoordFixed` aspect binds the width, the panel the axis draws
///    across is narrower than the region, by an amount that depends on the
///    band. The band is then the fixed point of `b → band at the panel width
///    b leaves, laid out within b`, started from the larger of the
///    unconstrained band and the band at the width it leaves. That start
///    is bounded the way step 2 bounds the unconstrained band, but against
///    the panel the aspect leaves: a band that leaves the aspect's panel no
///    width or height is never started from. Within a bound `b` the drawn
///    band never exceeds `b`, so the passes never increase and every pass
///    leaves a drawable panel; they stop at a band the axis draws exactly,
///    or after [`X_BAND_FIXED_POINT_PASSES`]. The bound returned is that
///    band (or the bound, if smaller), so every later layout of the x axes
///    draws exactly what is reserved.
///
/// Returns `(0.0, f64::INFINITY)` for an undrawn x axis.
fn reserve_x_label_band(
    geometry: &PanelGeometry,
    inner_after_legend: Rect,
    y_bands_w: f64,
    x_title_gutter: f64,
    metrics: &dyn TextMetrics,
) -> (f64, f64) {
    let (axes, theme) = (geometry.axes, geometry.theme);
    if !axes.show_x {
        return (0.0, f64::INFINITY);
    }
    let x_on_top = matches!(axes.x.orient, AxisOrient::Top);
    let region_w = (inner_after_legend.w - y_bands_w).max(0.0);
    let extent = |x_label_band: f64| {
        let x_band = clamp_axis_band(
            x_label_band + x_title_gutter,
            axes.x.overrides.placement.min_band,
            axes.x.overrides.placement.max_band,
        );
        let region = Rect {
            y: inner_after_legend.y + if x_on_top { x_band } else { 0.0 },
            w: region_w,
            h: (inner_after_legend.h - x_band).max(0.0),
            ..inner_after_legend
        };
        geometry.extent(region, x_label_band)
    };
    let band = |panel_w: f64, budget: f64| x_label_band(axes, panel_w, budget, theme, metrics);
    let pre_aspect_w = extent(0.0).pre_aspect_w;
    let unbounded = band(pre_aspect_w, f64::INFINITY);
    let cramped = x_label_band_budget(unbounded, &|b| extent(b).h > MIN_PANEL_DIM);
    let reserved = if cramped.is_finite() { band(pre_aspect_w, cramped) } else { unbounded };
    let corrected_w = |b: f64| extent(b).w;
    if corrected_w(reserved) == pre_aspect_w {
        return (reserved, cramped);
    }
    let start = reserved.max(band(corrected_w(reserved), cramped));
    let drawable = |b: f64| {
        let e = extent(b);
        e.w > MIN_PANEL_DIM && e.h > MIN_PANEL_DIM
    };
    let bound = cramped.min(x_label_band_budget(start, &drawable));
    let mut b = start.min(bound);
    for _ in 0..X_BAND_FIXED_POINT_PASSES {
        let next = band(corrected_w(b), bound.min(b));
        if next == b {
            break;
        }
        b = next;
    }
    (b, bound.min(b))
}

/// [`lay_out_plot`]'s products. `warnings` holds the facet stage's
/// warnings (dropped panels, empty cells) then the per-panel stage's
/// (collapse, x-label elision), in that push order.
#[cfg_attr(test, derive(Clone))]
struct PlotLayout {
    plot_region: Rect,
    panels: Vec<PanelLayout>,
    axis_layouts: Vec<AxisLayout>,
    secondary_y_axes: Vec<AxisLayout>,
    warnings: Vec<LayoutWarning>,
}

/// 400 stage 4 — split `plot_region` into facet cells (or a single panel),
/// returning the per-panel grid placements and any facet warnings (dropped
/// panels, empty cells). 840: the row-header strip band is the shared
/// `strip_band` value. Pure extraction of the former inline `panel_rects` block;
/// arithmetic + warning order unchanged.
fn split_panels(
    plot_region: Rect,
    spec: &crate::spec::chart::ChartSpec,
    facet_groups: &[FacetGroup],
    theme: &ThemeInputs,
    x_label_band: f64,
    show_x: bool,
    strip_band: f64,
) -> (Vec<PanelRect>, Vec<LayoutWarning>) {
    let mut warnings: Vec<LayoutWarning> = Vec::new();

    // panel_rects: (grid_row, grid_col, cell_rect, col_facet_key, row_facet_key)
    // row_facet_key is Some only in grid mode (FacetSpec.row is set).
    let panel_rects: Vec<PanelRect> = if let Some(facet) = &spec.facet {
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
        let gy = if effective_nrows > 1 && show_x {
            gy + x_label_band
        } else {
            gy
        };
        // Grid mode with a row dimension: reserve a right-side strip band for
        // row headers (ggplot2 / Altair convention — row strips on the right).
        // Reserving on the right keeps the y-axis title and tick labels
        // unobstructed on the left. The band width equals one line of
        // strip text plus vertical padding on each side (840: `strip_band`).
        let row_strip_width = if facet.row.is_some() { strip_band } else { 0.0 };
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
            warnings.push(LayoutWarning::PanelsDropped {
                count: dropped,
                keys: dropped_keys,
            });
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
                    let row_part = g
                        .row_key
                        .as_ref()
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

    (panel_rects, warnings)
}

/// Formats facet strip text: `"{title}: {value}"` when a dimension title is
/// set and non-blank, bare `"{value}"` otherwise (spec §6.2). Shared by the
/// column-strip and row-strip builders below so the two dimensions' prefix
/// behavior cannot drift apart. `title` absent or trim-empty reproduces the
/// pre-existing bare value, keeping untitled facets byte-identical. Trimming
/// only decides absence: a non-blank title is rendered verbatim, whitespace
/// and all.
fn facet_strip_text(title: Option<&str>, value: &str) -> String {
    match title.filter(|t| !t.trim().is_empty()) {
        Some(t) => format!("{t}: {value}"),
        None => value.to_string(),
    }
}

/// 400 stage 5 — lay out each panel: clamp degenerate rects, reserve column /
/// row-header strips, apply CoordFixed aspect correction, and build the per-axis
/// layouts (with facet title suppression on non-edge panels). Returns the panels,
/// the flat axis-layout list, and any per-panel warnings (collapse, x-label
/// elision). 840: column + row strip bands are the shared `strip_band`. Pure
/// extraction of the former per-panel loop; arithmetic + warning/axis push order
/// unchanged.
#[allow(clippy::too_many_arguments)]
fn layout_panel_axes(
    panel_rects: Vec<PanelRect>,
    spec: &crate::spec::chart::ChartSpec,
    axes: &AxesInput,
    theme: &ThemeInputs,
    metrics: &dyn TextMetrics,
    strip_band: f64,
    secondary_y_bands: &[f64],
    x_band_budget: f64,
) -> (
    Vec<PanelLayout>,
    Vec<AxisLayout>,
    Vec<AxisLayout>,
    Vec<LayoutWarning>,
) {
    let mut panels: Vec<PanelLayout> = Vec::new();
    let mut axis_layouts: Vec<AxisLayout> = Vec::new();
    // Secondary y-axis layouts, one per `independent_y` layer per panel
    // (secondary-y-axis, GH #52). Kept separate from `axis_layouts` (the
    // primary x/y list every other consumer already filters by orient) so
    // adding slots 1..n cannot perturb any existing `.find()`/`.filter()` over
    // `axis_layouts` — the shared-path byte-stability invariant.
    let mut secondary_y_axis_layouts: Vec<AxisLayout> = Vec::new();
    let mut warnings: Vec<LayoutWarning> = Vec::new();

    // 840: column-strip and row-header-strip bands are the same shared size; the
    // column strip applies whenever faceting, the row strip only in grid mode.
    let strip_band_height = if spec.facet.is_some() {
        strip_band
    } else {
        0.0
    };

    // Compute the maximum row index so non-bottom panels can suppress the
    // x-axis title (only the bottom row needs it — duplicating it in every
    // inter-row gutter is visually noisy).
    let max_row = panel_rects
        .iter()
        .map(|(r, _, _, _, _)| *r)
        .max()
        .unwrap_or(0);
    // Compute the minimum column index so non-leftmost panels can suppress the
    // y-axis title (only the leftmost column needs it — duplicating it on every
    // panel in a multi-column faceted chart causes visual overlap).
    let min_col = panel_rects
        .iter()
        .map(|(_, c, _, _, _)| *c)
        .min()
        .unwrap_or(0);

    // Width of the row-header strip on the right side of the grid region (grid
    // mode only). Mirrored from the reservation made when building panel_rects.
    let row_strip_band_width = if spec.facet.as_ref().is_some_and(|f| f.row.is_some()) {
        strip_band
    } else {
        0.0
    };
    // Maximum column index across all panels. Used to identify the rightmost
    // column so each row-header strip is emitted once, on the right edge.
    // Two-way grid facets always form a proper rectangular grid, so max_col is
    // uniform across all rows.
    let max_col = panel_rects
        .iter()
        .map(|(_, c, _, _, _)| *c)
        .max()
        .unwrap_or(0);
    // Track which grid rows have already had their row-header strip emitted. In
    // a 3×3 grid, row_val "r1" appears for three panels (cols c1, c2, c3); we
    // only want one row-header strip per grid row (the rightmost col).
    let mut emitted_row_strips: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    // 7. Per-panel: clamp degenerate rects, collect axes.
    for (panel_index, (row, col, mut rect, facet_key, row_facet_key)) in
        panel_rects.into_iter().enumerate()
    {
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
                let title = spec.facet.as_ref().and_then(|f| f.column_strip_title());
                Some(StripTitleLayout {
                    text: facet_strip_text(title, &key.value),
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
                let row_title = spec.facet.as_ref().and_then(|f| f.row_strip_title());
                Some(StripTitleLayout {
                    text: facet_strip_text(row_title, &rk.value),
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

        let rect = fixed_aspect(rect, spec);

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
            // This panel's offset from the reference panel's plot area. An
            // explicit Band/Point/positional-Ordinal range is chart-absolute,
            // and each facet panel's mark scales re-anchor it by exactly this
            // offset (GH #70, `render::scene_build::resolve_panel_scales`), so
            // every axis below translates its placement's range by the same
            // amount and its labels land on the panel's marks bit for bit.
            let reference = panels[0].plot_area;
            let (dx, dy) = (rect.x - reference.x, rect.y - reference.y);

            // Axis suppression: when this y axis is not drawn — `chart.axis(y=False)`,
            // or a projected coord that has no Cartesian axes (see `AxesInput::show_y`) —
            // skip emitting the y axis layout entirely. Since #182 the plot area is
            // LARGER in that case rather than unchanged: `reserve_axis_bands` reserves
            // neither the label band nor the title gutter for chrome this branch omits,
            // so the omission and the reservation now read the same flag.
            if axes.show_y {
                let mut y_input = axes.y_for_panel(panel_index).clone();
                y_input.categorical_placement = y_input.categorical_placement.map(|p| p.translated(dy));
                // Suppress y-axis title on non-leftmost-column facet panels to
                // avoid duplicating the title in every inter-column gutter —
                // only the leftmost column's title is needed.
                if col > min_col && spec.facet.is_some() {
                    y_input.title.set(None);
                }
                let y_label_fs = y_input
                    .overrides
                    .labels
                    .font_size
                    .unwrap_or(theme.typography.label_font_size);
                let (y_axis, ywarn) = axis::layout_y_axis(
                    &y_input,
                    rect,
                    panel_index,
                    y_label_fs,
                    theme.typography.title_font_size,
                    theme.padding.axis_title_padding,
                    y_input.tick_size(theme),
                    metrics,
                );
                if let Some(axis::AxisLabelWarning::LabelsElided { count }) = ywarn {
                    warnings.push(LayoutWarning::LabelsElided {
                        axis: axis_layouts.len(),
                        count,
                        secondary_slot: None,
                    });
                }
                axis_layouts.push(y_axis);
            }

            // Secondary y-axes (secondary-y-axis, GH #52): one per
            // `independent_y` layer, orient forced `Right` and stacked
            // outward beyond the primary's own band. Slot k's `translate`
            // offset is the sum of the PRECEDING slots' band widths, so slot 1
            // sits flush against `rect`'s right edge (the plot area is already
            // shrunk to reserve every slot's band) and each later slot sits
            // beyond the previous one's band — reusing the same
            // `translate`-shift render mechanism (B5) axis style overrides
            // already use, rather than a bespoke placement path. A user-set
            // `translate` on that layer's own `Axis(...)` composes additively
            // on top. Independent of `axes.show_y` (that toggle only
            // suppresses the primary/left axis); gated only on the panel rect
            // being non-degenerate, mirroring the primary y/x guards above.
            let mut cumulative_offset = 0.0_f64;
            for (slot_idx, secondary_input) in axes.secondary_y.iter().enumerate() {
                let mut sec_input = secondary_input.clone();
                sec_input.orient = AxisOrient::Right;
                sec_input.categorical_placement = sec_input.categorical_placement.map(|p| p.translated(dy));
                let existing_translate = sec_input.overrides.placement.translate.unwrap_or(0.0);
                sec_input.overrides.placement.translate =
                    Some(cumulative_offset + existing_translate);
                let sec_label_fs = sec_input
                    .overrides
                    .labels
                    .font_size
                    .unwrap_or(theme.typography.label_font_size);
                let (sec_axis, sec_warn) = axis::layout_y_axis(
                    &sec_input,
                    rect,
                    panel_index,
                    sec_label_fs,
                    theme.typography.title_font_size,
                    theme.padding.axis_title_padding,
                    sec_input.tick_size(theme),
                    metrics,
                );
                if let Some(axis::AxisLabelWarning::LabelsElided { count }) = sec_warn {
                    // `axis`: this secondary axis's own 0-based rank in
                    // `secondary_y_axis_layouts` (the vec it is about to be
                    // pushed to), NOT an index into `axis_layouts` — that vec
                    // does not contain secondary axes at all, and a fabricated
                    // combined index can collide with an unrelated x/primary-y
                    // warning emitted later in the same panel.
                    // `secondary_slot`: the loop's own `slot_idx` — the
                    // established per-panel y-SLOT numbering (matches
                    // `tick_slot`/`y_slot` elsewhere), NOT the vec rank above.
                    // The two coincide in a single-panel chart but diverge
                    // across facet panels (panel k's slot-0 axis has vec rank
                    // k), so `secondary_slot` must read `slot_idx`, not `axis`.
                    warnings.push(LayoutWarning::LabelsElided {
                        axis: secondary_y_axis_layouts.len(),
                        count,
                        secondary_slot: Some(slot_idx),
                    });
                }
                secondary_y_axis_layouts.push(sec_axis);
                cumulative_offset += secondary_y_bands.get(slot_idx).copied().unwrap_or(0.0);
            }

            if axes.show_x {
                let mut x_input = axes.x_for_panel(panel_index).clone();
                x_input.categorical_placement = x_input.categorical_placement.map(|p| p.translated(dx));
                // Suppress x-axis title on non-bottom-row facet panels to
                // avoid duplicating "Feature value" (or similar) in every
                // inter-row gutter — only the bottom row's title is needed.
                if row < max_row && spec.facet.is_some() {
                    x_input.title.set(None);
                }
                let (x_axis, xwarn) = axis::layout_x_axis(
                    &x_input,
                    rect,
                    panel_index,
                    theme.typography.title_font_size,
                    theme.padding.axis_title_padding,
                    &LabelBandCtx::for_axis(&x_input, theme, metrics).with_band_budget(x_band_budget),
                );
                if let Some(axis::AxisLabelWarning::LabelsElided { count }) = xwarn {
                    warnings.push(LayoutWarning::LabelsElided {
                        axis: axis_layouts.len(),
                        count,
                        secondary_slot: None,
                    });
                }
                axis_layouts.push(x_axis);
            }
        }
    }

    (panels, axis_layouts, secondary_y_axis_layouts, warnings)
}

/// `seam` is what a composite parent imposes on this leaf (see
/// [`CompositeLayoutSeam`]), all three fields:
///
/// - `legend` — per-channel legend suppression: reserve no gutter and draw
///   nothing for a channel the composite renders one figure-level legend for
///   (design §6).
/// - `plot_region` — the overlay group's shared plot rect, replacing the one
///   this leaf's own axis-band reservation produces (GH #89A).
/// - `suppress_chart_title` — reserve no title band and lay out no title,
///   for a leaf whose scene title the composite clears when it merges
///   (GH #89A), so an undrawn title cannot inflate that shared rect.
///
/// `Default` reproduces a standalone chart's layout byte-for-byte.
#[allow(clippy::too_many_arguments)]
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
    seam: CompositeLayoutSeam,
) -> Result<LayoutResult, LayoutError> {
    // 1. Validate inputs.
    if !viewport.is_drawable() {
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
                return Err(LayoutError::InvalidFacetSpec(
                    "nrows and ncols must be > 0".into(),
                ));
            }
            _ => {}
        }
        if facet_groups.is_empty() {
            return Err(LayoutError::EmptyFacetGroups);
        }
    }

    // 2. Apply outer padding. `padding.auto` (D10, spec §4.7, F-L07-08)
    // expands a side with no explicit override to also contain (a) the
    // overhang of a continuous axis's drawn edge tick labels, and (b) an axis title
    // that would otherwise clip past the viewport edge, plus whatever a
    // legend, the chart title band, or the secondary-y bands already
    // reserve on that side. `auto_padding_extra`'s own doc has the
    // tick-then-title composition rationale; `settle_auto_padding`'s own
    // doc has the fixed-point rationale for the chrome credit the two
    // passes solve against. The extra is a bounded fixed-point solve, run
    // only when requested, so a non-auto chart's inset is unchanged.
    let viewport_rect = viewport.into_rect();
    let base = base_padding_insets(theme);
    let extra = if theme.padding.padding_auto {
        let inputs = LayoutInputs {
            spec,
            theme,
            viewport,
            axes,
            facet_groups,
            legend_entries,
            legend_title: legend_title.as_deref(),
            colorbar,
            metrics,
            legend_overrides,
            aux_legend_inputs,
            seam,
        };
        settle_auto_padding(&inputs, base)
    } else {
        Inset::default()
    };
    let inset = resolve_padding_inset(theme, base, extra);
    let inner = viewport_rect.shrink(inset);
    // Captured before the chrome reservations below shadow `inner`: this is
    // the blank-margin boundary `LayoutResult::padded_region` promises.
    let padded_region = inner;
    if inner.w <= 0.0 || inner.h <= 0.0 {
        // Report the CALLER's actual value and the specific side responsible
        // (D10, spec §4.7 — repro:
        // `configure_padding(top=1e9)` previously reported the theme
        // DEFAULT 16, not the caller's 1e9). `Rect::shrink` collapses both
        // `inner.w` and `inner.h` to 0 together once either axis fails, so
        // the violated axis is re-derived here from the raw (pre-shrink)
        // extents rather than read back off `inner`.
        let raw_w = viewport.width - inset.left - inset.right;
        let raw_h = viewport.height - inset.top - inset.bottom;
        debug_assert!(
            raw_w <= 0.0 || raw_h <= 0.0,
            "inner collapsed to ZERO but neither raw extent is <= 0"
        );
        let (side, value, dim) = if raw_w <= 0.0 {
            if inset.left >= inset.right {
                ("left", inset.left, viewport.width)
            } else {
                ("right", inset.right, viewport.width)
            }
        } else if inset.top >= inset.bottom {
            ("top", inset.top, viewport.height)
        } else {
            ("bottom", inset.bottom, viewport.height)
        };
        return Err(LayoutError::PaddingExceedsViewport {
            padding: value,
            side,
            viewport_dim: dim,
        });
    }

    // 2b + 3 + 3b. Reserve the chart-level title band (Schwabish SB1 adds
    // subtitle; band height ≈ title_font_size * 1.4 +
    // (subtitle_font_size * 1.4 if subtitle) + title_offset; a composite
    // that will clear this leaf's title, GH #89A, reserves nothing and
    // lays out no title, so the band cannot become a gutter nothing is
    // ever drawn in), then the color legend strip (categorical or
    // colorbar) plus any stacked size/shape aux blocks, both off the same
    // gutter.
    let (chart_title_layout, legend_layout, aux_legends, inner_after_legend, legend_dropped) =
        reserve_title_and_legends(
            inner,
            spec,
            theme,
            legend_entries,
            legend_title.as_deref(),
            colorbar,
            metrics,
            legend_overrides,
            aux_legend_inputs,
            seam,
        );

    // 4 to 7. Axis bands, the seam's imposed plot region, the facet split,
    //    and each panel's strips + axes (see `lay_out_plot`).
    let PlotLayout {
        plot_region,
        panels,
        axis_layouts,
        secondary_y_axes,
        warnings: plot_warnings,
    } = lay_out_plot(inner_after_legend, spec, theme, axes, facet_groups, metrics, seam);

    // Assemble warnings in the original push order: legend overflow first, then
    // the facet stage (dropped panels / empty cells), then the per-panel stage
    // (collapse / x-label elision).
    let mut warnings: Vec<LayoutWarning> = Vec::new();
    if legend_dropped > 0 {
        warnings.push(LayoutWarning::LegendOverflowed {
            entries_dropped: legend_dropped,
        });
    }
    warnings.extend(plot_warnings);

    Ok(LayoutResult {
        viewport: viewport_rect,
        panels,
        axes: axis_layouts,
        legend: legend_layout,
        aux_legends,
        chart_title: chart_title_layout,
        warnings,
        secondary_y_axes,
        plot_region,
        padded_region,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── M2 (batch D task 5): `estimate_panel_pixel_size` ──────────────────

    #[test]
    fn estimate_panel_pixel_size_subtracts_theme_padding_unfaceted() {
        let theme = ThemeInputs::default(); // padding.padding == 16.0 on every side
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        assert_eq!(
            estimate_panel_pixel_size(viewport, &theme, (1, 1)),
            (568, 368),
            "600-16-16=568, 400-16-16=368 for the default flat 16px padding"
        );
    }

    #[test]
    fn estimate_panel_pixel_size_divides_by_grid_dims() {
        let theme = ThemeInputs::default();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let (w, h) = estimate_panel_pixel_size(viewport, &theme, (1, 1));
        let (fw, fh) = estimate_panel_pixel_size(viewport, &theme, (3, 2));
        assert_eq!(fw, ((w as f64) / 3.0).round() as u32);
        assert_eq!(fh, ((h as f64) / 2.0).round() as u32);
    }

    #[test]
    fn estimate_panel_pixel_size_floors_at_one_when_padding_exceeds_viewport() {
        // Contract: the estimate never returns a zero dimension. When
        // padding meets or exceeds the viewport, the true inner extent is
        // <= 0, and this floors each axis at 1 rather than propagate a
        // zero-sized estimate to consumers (`transform::raster`'s `"screen"`
        // arm cannot survive a zero — see the module doc on
        // `estimate_panel_pixel_size`). `compute_layout` independently
        // detects this same condition downstream and raises the typed
        // `LayoutError::PaddingExceedsViewport` refusal; this floor is not a
        // real panel size, just a value that lets the transform pass
        // complete without panicking before layout's own refusal fires.
        let mut theme = ThemeInputs::default();
        theme.padding.padding = 1000.0;
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        assert_eq!(estimate_panel_pixel_size(viewport, &theme, (1, 1)), (1, 1));
    }

    #[test]
    fn estimate_panel_pixel_size_honors_per_side_padding_overrides() {
        let mut theme = ThemeInputs::default();
        theme.padding.padding_left = Some(50.0);
        theme.padding.padding_right = Some(10.0);
        theme.padding.padding_top = Some(0.0);
        theme.padding.padding_bottom = Some(0.0);
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        assert_eq!(
            estimate_panel_pixel_size(viewport, &theme, (1, 1)),
            (540, 400),
            "600 - 50 - 10 = 540 width; full 400 height with 0/0 top/bottom"
        );
    }

    /// Test helper: `AxisInput::new` plus the theme-default fold
    /// `AxesInput::apply_show_defaults` performs in production (against
    /// `ThemeInputs::default()`, which every test in this module uses), so
    /// `compute_layout` tests that hand-build an `AxesInput` don't have to
    /// know about the `show_domain()`/`show_grid()` precondition those
    /// accessors now assert in debug builds. A test that DOES care overwrites
    /// `input.overrides.domain.show`/`.grid.show` afterward, same as before.
    fn axis_input(
        orient: AxisOrient,
        title: Option<String>,
        tick_labels: Vec<String>,
        label_angle: Option<f64>,
    ) -> AxisInput {
        let new = AxisInput::new;
        let mut input = new(orient, title, tick_labels, label_angle);
        input.overrides.domain.show = Some(true);
        input.overrides.grid.show = Some(true);
        input
    }

    #[test]
    fn layout_result_round_trip_empty() {
        let r = LayoutResult {
            viewport: Rect {
                x: 0.0,
                y: 0.0,
                w: 600.0,
                h: 400.0,
            },
            panels: vec![],
            axes: vec![],
            legend: None,
            aux_legends: vec![],
            chart_title: None,
            warnings: vec![],
            secondary_y_axes: vec![],
            plot_region: Rect::ZERO,
            padded_region: Rect::ZERO,
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
        assert!(!json.contains("legend"));
        assert!(!json.contains("warnings"));
        assert!(!json.contains("secondary_y_axes"));
        // `plot_region` is in-process only (GH #89A): serializing it would
        // add a key to `fm.compute_layout`'s returned dict.
        assert!(!json.contains("plot_region"));
    }

    #[test]
    fn layout_warning_round_trip_each_variant() {
        for w in [
            LayoutWarning::PanelCollapsed { panel_index: 2 },
            LayoutWarning::LabelsElided {
                axis: 0,
                count: 5,
                secondary_slot: None,
            },
            LayoutWarning::LabelsElided {
                axis: 1,
                count: 2,
                secondary_slot: Some(1),
            },
            LayoutWarning::LegendOverflowed { entries_dropped: 3 },
            LayoutWarning::PanelsDropped {
                count: 1,
                keys: vec!["col_cat=c2".into()],
            },
            LayoutWarning::EmptyPartitions {
                keys: vec!["col_cat=c2, row_cat=r2".into()],
            },
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
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    /// 840: the strip band is one line of strip text plus padding on each side,
    /// computed once via `strip_band_size` and reused at every strip site.
    #[test]
    fn strip_band_size_is_line_height_plus_padding_both_sides() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let mut theme = ThemeInputs::default();
        theme.sizes.strip_text_size = 12.0;
        theme.padding.strip_padding = 6.0;
        // line_height(12) = 12 * 1.2 = 14.4; + 2 * 6 = 26.4.
        assert!((strip_band_size(&theme, &m) - (12.0 * 1.2 + 2.0 * 6.0)).abs() < 1e-9);
    }

    /// F-L01-04: no dimension title -> bare value, byte-identical to the
    /// pre-fix hardwired `key.value.clone()`.
    #[test]
    fn facet_strip_text_is_bare_when_title_absent() {
        assert_eq!(facet_strip_text(None, "setosa"), "setosa");
    }

    /// F-L01-04: a dimension title set -> `"{title}: {value}"` (spec §6.2).
    #[test]
    fn facet_strip_text_is_prefixed_when_title_set() {
        assert_eq!(
            facet_strip_text(Some("Species"), "setosa"),
            "Species: setosa"
        );
    }

    /// #176(10a): an empty or whitespace-only title is treated as absent, on
    /// both the column-strip and row-strip call sites of the shared function.
    #[test]
    fn facet_strip_text_is_bare_when_title_trim_empty() {
        assert_eq!(facet_strip_text(Some(""), "p"), "p");
        assert_eq!(facet_strip_text(Some("   "), "p"), "p");
        assert_eq!(facet_strip_text(Some("\t\n"), "q"), "q");
        // control: a non-empty title still prefixes.
        assert_eq!(facet_strip_text(Some("Species"), "q"), "Species: q");
        // control: trim only decides absence, never alters the rendered
        // text of a non-blank title.
        assert_eq!(facet_strip_text(Some(" Species "), "q"), " Species : q");
    }

    fn minimal_chart_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "a".into(),
                    type_: None,
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "b".into(),
                    type_: None,
                    ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn dummy_axes() -> AxesInput {
        AxesInput {
            x: axis_input(
                AxisOrient::Bottom,
                None,
                vec!["0".into(), "1".into(), "2".into(), "3".into()],
                None,
            ),
            y: axis_input(
                AxisOrient::Left,
                None,
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    fn default_theme_inputs() -> ThemeInputs {
        ThemeInputs::default()
    }

    /// `ThemeInputs::default()`'s auto-off control (#135: the built-in
    /// default flipped `padding_auto` to `true`). A test that demonstrates
    /// the `padding.auto` rescue by contrasting an auto-off baseline against
    /// an explicit auto-on render needs its baseline to actually be off, or
    /// the contrast goes vacuous now that `ThemeInputs::default()` is on.
    fn theme_inputs_with_padding_auto_off() -> ThemeInputs {
        ThemeInputs {
            padding: ThemePadding {
                padding_auto: false,
                ..ThemeInputs::default().padding
            },
            ..ThemeInputs::default()
        }
    }

    #[test]
    fn compute_layout_single_chart_no_facet_no_legend() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

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
            CompositeLayoutSeam::default(),
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

    // ── composite-shared-legend seam (design §6, 2026-07-12): LegendSuppression ──

    /// `LegendSuppression { color: true, .. }` must reserve exactly zero
    /// gutter and draw nothing for the color legend — the layout-stage half
    /// of the seam contract. Proven three ways against the same non-empty
    /// `legend_entries`: (1) unsuppressed lays the legend out and narrows the
    /// plot area vs. a no-legend baseline; (2) suppressed draws no legend
    /// (`result.legend.is_none()`); (3) suppressed's plot area is byte-equal
    /// to the no-legend baseline — proof no gutter was reserved, not just
    /// that entries were dropped. Also checks suppression never raises the
    /// unrelated `LegendOverflowed` warning (it never attempted layout, so
    /// nothing was "dropped").
    #[test]
    fn compute_layout_color_legend_suppression_reserves_no_gutter_and_draws_nothing() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![
            LegendEntry::new("alpha", SymbolKind::Circle),
            LegendEntry::new("beta", SymbolKind::Circle),
        ];

        let baseline = compute_layout(
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
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed with no legend at all");

        let with_legend = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed with an active legend");
        assert!(
            with_legend.legend.is_some(),
            "unsuppressed legend must be laid out"
        );
        assert!(
            with_legend.panels[0].plot_area.w < baseline.panels[0].plot_area.w,
            "an active legend must narrow the plot area vs. the no-legend baseline"
        );

        let suppressed = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam {
                legend: LegendSuppression {
                    color: true,
                    size: false,
                },
                ..Default::default()
            },
        )
        .expect("layout should succeed with a suppressed legend");
        assert!(
            suppressed.legend.is_none(),
            "suppressed color legend must not be laid out/drawn"
        );
        assert!(
            suppressed.warnings.is_empty(),
            "suppression must not raise LegendOverflowed"
        );
        assert_eq!(
            suppressed.panels[0].plot_area, baseline.panels[0].plot_area,
            "a suppressed legend must reserve exactly zero gutter — plot area matches the no-legend baseline"
        );
    }

    /// `LegendSuppression { size: true, .. }` filters only the `Size` aux
    /// block out of the reservation/draw pass; a `Shape` aux block (never
    /// compositor-suppressed) is untouched. Proven by comparing a
    /// size-suppressed [Size, Shape] run against a Shape-only unsuppressed
    /// run: byte-equal plot areas mean the suppressed size block reserved
    /// zero extra gutter beyond what Shape alone would need.
    #[test]
    fn compute_layout_size_legend_suppression_reserves_no_gutter_but_keeps_shape_aux() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let size_entry = AuxLegendInput::Size {
            title: Some("pop".into()),
            entries: vec![SizeLegendEntry {
                label: "10".into(),
                radius: 4.0,
                color_hex: None,
            }],
            symbol: SymbolKind::Circle,
        };
        let shape_entry = AuxLegendInput::Shape {
            title: Some("region".into()),
            entries: vec![ShapeLegendEntry {
                label: "AS".into(),
                shape_name: "circle".into(),
            }],
        };

        let shape_only = compute_layout(
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
            &[shape_entry.clone()],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed with only a shape aux legend");
        assert_eq!(
            shape_only.aux_legends.len(),
            1,
            "shape-only baseline must lay out one block"
        );

        let mixed_suppressed = compute_layout(
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
            &[size_entry, shape_entry],
            CompositeLayoutSeam {
                legend: LegendSuppression {
                    color: false,
                    size: true,
                },
                ..Default::default()
            },
        )
        .expect("layout should succeed with a suppressed size aux legend");

        assert_eq!(
            mixed_suppressed.aux_legends.len(),
            1,
            "only the shape block should survive size suppression"
        );
        assert!(
            mixed_suppressed.aux_legends[0]
                .entries
                .iter()
                .all(|e| e.shape_name.is_some()),
            "surviving aux block must be the shape legend, not size"
        );
        assert_eq!(
            mixed_suppressed.panels[0].plot_area, shape_only.panels[0].plot_area,
            "suppressed size must reserve zero gutter beyond the shape-only baseline"
        );
    }

    // ── #52 Task 3: secondary y-axis layout + axis emission ──────────────────

    /// Build `n` secondary `AxisInput`s, each with the SAME 3-char-max tick
    /// labels (`"0"`, `"50"`, `"100"`) and a title, so every slot reserves an
    /// identical, precisely-computable band under `MockMetrics`'s fixed
    /// per-char width. Distinguishable via `AxisInput.title` (`"Sec1"..`) for
    /// per-axis assertions.
    fn n_secondary_axes(n: usize) -> Vec<AxisInput> {
        (1..=n)
            .map(|i| {
                axis_input(
                    AxisOrient::Right,
                    Some(format!("Sec{i}")),
                    vec!["0".into(), "50".into(), "100".into()],
                    None,
                )
            })
            .collect()
    }

    /// The exact per-secondary-axis band width `MockMetrics { measure:
    /// fixed_width(8.0), line_h_factor: 1.2 }` produces for [`n_secondary_axes`]:
    /// label band (#97, spec §4.1: `tick_size + label_pad_eff` standoff +
    /// `"100"` = 3 chars * 8px) + title gutter (`title_font_size * line_h_factor
    /// + axis_title_padding`), using `ThemeInputs::default()`'s `tick_size`
    /// (4.0), `title_font_size` (13.0), and `axis_title_padding` (8.0).
    fn expected_secondary_band(theme: &ThemeInputs) -> f64 {
        let label_band = theme.sizes.tick_size + 2.0 + 3.0 * 8.0;
        let title_gutter =
            theme.typography.title_font_size * 1.2 + theme.padding.axis_title_padding;
        label_band + title_gutter
    }

    /// Band math for n=1,2,3 secondaries: each additional independent-y layer
    /// narrows the plot area by exactly one more axis's band width, and
    /// `secondary_y_axes.len()` matches the slot count (GH #52 Task 3).
    #[test]
    fn compute_layout_secondary_y_band_math_n1_n2_n3() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 800.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let band = expected_secondary_band(&theme);

        let run = |n: usize| {
            let mut axes = dummy_axes();
            axes.secondary_y = n_secondary_axes(n);
            compute_layout(
                &spec,
                &theme,
                viewport,
                &axes,
                &[],
                &[],
                None,
                None,
                &m,
                &legend::LegendOverrides::default(),
                &[],
                CompositeLayoutSeam::default(),
            )
            .expect("layout should succeed with secondary y axes")
        };

        let r0 = run(0);
        let r1 = run(1);
        let r2 = run(2);
        let r3 = run(3);

        assert!(r0.secondary_y_axes.is_empty(), "n=0 has no secondary axes");
        assert_eq!(r1.secondary_y_axes.len(), 1);
        assert_eq!(r2.secondary_y_axes.len(), 2);
        assert_eq!(r3.secondary_y_axes.len(), 3);

        let w0 = r0.panels[0].plot_area.w;
        let w1 = r1.panels[0].plot_area.w;
        let w2 = r2.panels[0].plot_area.w;
        let w3 = r3.panels[0].plot_area.w;

        // Each additional secondary axis narrows the plot area by exactly one
        // more band width — no overdraw, no double-reservation.
        assert!(
            (w0 - w1 - band).abs() < 1e-6,
            "n=0→1 shrink: {w0} - {w1} should be {band}"
        );
        assert!(
            (w1 - w2 - band).abs() < 1e-6,
            "n=1→2 shrink: {w1} - {w2} should be {band}"
        );
        assert!(
            (w2 - w3 - band).abs() < 1e-6,
            "n=2→3 shrink: {w2} - {w3} should be {band}"
        );
    }

    /// R2 secondary-y: `label_angle` on a secondary-y `Axis(...)` (GH #52) is
    /// honored exactly like the primary y-axis — `layout_y_axis` is shared by
    /// both, so this is a wiring check that the override actually reaches the
    /// secondary-axis call site (`reserve_axis_bands`'s `secondary_y_bands` map
    /// and the panel-loop's secondary-axis emission both thread it through).
    #[test]
    fn compute_layout_secondary_y_axis_honors_label_angle_override() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 800.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();

        let mut axes = dummy_axes();
        let mut sec = axis_input(
            AxisOrient::Right,
            Some("Sec1".into()),
            vec!["0".into(), "50".into(), "100".into()],
            Some(-45.0),
        );
        sec.orient = AxisOrient::Right;
        axes.secondary_y = vec![sec];

        let result = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed with a rotated secondary y axis");

        assert_eq!(result.secondary_y_axes.len(), 1);
        for t in &result.secondary_y_axes[0].ticks {
            assert_eq!(
                t.label_angle, -45.0,
                "secondary-y override angle must reach every tick"
            );
        }
    }

    /// Regression: a secondary-y elision warning and an
    /// x-axis elision warning emitted in the SAME panel must carry distinct,
    /// non-colliding identities. Pre-fix, the secondary-y push used
    /// `axis_layouts.len() + secondary_y_axis_layouts.len()` — an index into
    /// NEITHER real vec — which could numerically collide with the x-axis
    /// warning's `axis_layouts.len()` pushed later in the same iteration
    /// (both read `axis_layouts.len()` before the x push, since secondaries
    /// never append to `axis_layouts`). Force both a secondary-y elision
    /// (20 tightly-packed rotated ticks) and an x-axis elision (20
    /// tightly-packed rotated ticks) in one `compute_layout` call and assert
    /// the two `LabelsElided` warnings are tagged unambiguously.
    #[test]
    fn compute_layout_secondary_y_and_x_elision_warnings_do_not_collide() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 400.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();

        let mut axes = dummy_axes();
        // 20 long unsplittable x labels + a forced override angle: the
        // override branch's own collision check (not the graduated cascade)
        // deterministically elides regardless of the label-gap policy.
        axes.x = axis_input(
            AxisOrient::Bottom,
            None,
            (0..20).map(|i| format!("Label_{i}")).collect(),
            Some(-45.0),
        );
        let sec = axis_input(
            AxisOrient::Right,
            None,
            (0..20).map(|i| format!("Sec_{i}")).collect(),
            Some(-45.0),
        );
        axes.secondary_y = vec![sec];

        let result = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed with both axes forced to elide");

        let elided: Vec<&LayoutWarning> = result
            .warnings
            .iter()
            .filter(|w| matches!(w, LayoutWarning::LabelsElided { .. }))
            .collect();
        assert_eq!(
            elided.len(),
            2,
            "expected exactly one x and one secondary-y elision warning"
        );

        let secondary_warns: Vec<&LayoutWarning> = elided
            .iter()
            .filter(|w| {
                matches!(
                    w,
                    LayoutWarning::LabelsElided {
                        secondary_slot: Some(_),
                        ..
                    }
                )
            })
            .copied()
            .collect();
        let primary_warns: Vec<&LayoutWarning> = elided
            .iter()
            .filter(|w| {
                matches!(
                    w,
                    LayoutWarning::LabelsElided {
                        secondary_slot: None,
                        ..
                    }
                )
            })
            .copied()
            .collect();
        assert_eq!(
            secondary_warns.len(),
            1,
            "exactly one secondary-y elision, tagged secondary_slot: Some"
        );
        assert_eq!(
            primary_warns.len(),
            1,
            "exactly one x-axis elision, tagged secondary_slot: None"
        );

        // The rendered messages must be distinguishable text, not just an
        // internal tag — a user reading two warnings must be able to tell
        // them apart even without inspecting the struct fields.
        let secondary_msg = secondary_warns[0].to_string();
        let primary_msg = primary_warns[0].to_string();
        assert!(
            secondary_msg.contains("secondary y-axis"),
            "got: {secondary_msg}"
        );
        assert!(
            !primary_msg.contains("secondary y-axis"),
            "got: {primary_msg}"
        );
    }

    /// Cycle-2 non-blocking S3 fix: `secondary_slot` must be the per-panel
    /// y-SLOT index (`slot_idx`), not `secondary_y_axis_layouts`' cross-panel
    /// vec rank — the two diverge as soon as more than one panel contributes
    /// a secondary axis. Two facet panels, each with exactly one secondary
    /// y-axis forced to elide: panel 0's secondary axis lands at vec rank 0,
    /// panel 1's at vec rank 1 (the shared vec accumulates across panels) —
    /// but BOTH are slot 0 within their own panel (the inner `slot_idx` loop
    /// resets every panel). Both warnings must report `secondary_slot:
    /// Some(0)`, proving the field reads the slot, not the vec rank.
    #[test]
    fn compute_layout_secondary_slot_is_per_panel_slot_not_vec_rank_across_facets() {
        let spec = faceted_spec(2);
        let groups = vec![
            FacetGroup {
                key: FacetKey {
                    field: "species".into(),
                    value: "a".into(),
                },
                n_rows: 10,
                row_key: None,
            },
            FacetGroup {
                key: FacetKey {
                    field: "species".into(),
                    value: "b".into(),
                },
                n_rows: 10,
                row_key: None,
            },
        ];
        let viewport = Viewport {
            width: 800.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();

        let mut axes = dummy_axes();
        // 20 tightly-packed rotated labels per panel — same forcing recipe as
        // the collision-identity test above, sized to still collide inside a
        // single facet cell (roughly half the single-panel width/height).
        let sec = axis_input(
            AxisOrient::Right,
            None,
            (0..20).map(|i| format!("Sec_{i}")).collect(),
            Some(-45.0),
        );
        axes.secondary_y = vec![sec];

        let result = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("faceted layout should succeed with both panels' secondary axes eliding");

        assert_eq!(result.panels.len(), 2, "expected 2 facet panels");
        let secondary_slots: Vec<Option<usize>> = result
            .warnings
            .iter()
            .filter_map(|w| match w {
                LayoutWarning::LabelsElided {
                    secondary_slot: Some(slot),
                    ..
                } => Some(Some(*slot)),
                _ => None,
            })
            .collect();
        assert_eq!(
            secondary_slots.len(),
            2,
            "expected one secondary-y elision warning per panel, got {secondary_slots:?}"
        );
        for slot in secondary_slots {
            assert_eq!(
                slot, Some(0),
                "every panel's lone secondary axis is slot 0, regardless of its vec rank across panels"
            );
        }
    }

    /// Every secondary y-axis renders `Right`-orient and stacks outward: slot
    /// k's `translate` offset is the sum of the PRECEDING slots' band widths
    /// (0, band, 2*band for three identical-width axes), so consecutive axes
    /// never overlap (no plot-area overdraw) — GH #52 Task 3.
    #[test]
    fn compute_layout_secondary_y_orient_right_and_stacked_translate_offsets() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 800.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let band = expected_secondary_band(&theme);

        let mut axes = dummy_axes();
        axes.secondary_y = n_secondary_axes(3);
        let result = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.secondary_y_axes.len(), 3);
        for axis in &result.secondary_y_axes {
            assert_eq!(
                axis.orient,
                AxisOrient::Right,
                "every secondary axis renders on the right"
            );
        }
        let offsets: Vec<f64> = result
            .secondary_y_axes
            .iter()
            .map(|a| a.translate.unwrap_or(0.0))
            .collect();
        assert!(
            (offsets[0] - 0.0).abs() < 1e-6,
            "slot 1 has no preceding band: {offsets:?}"
        );
        assert!(
            (offsets[1] - band).abs() < 1e-6,
            "slot 2 stacks past slot 1's band: {offsets:?}"
        );
        assert!(
            (offsets[2] - 2.0 * band).abs() < 1e-6,
            "slot 3 stacks past slots 1+2: {offsets:?}"
        );

        // Titles thread through per axis (spec §4: each axis titled from its
        // own layer's y field/title).
        let titles: Vec<&str> = result
            .secondary_y_axes
            .iter()
            .map(|a| a.title.as_ref().unwrap().text.as_str())
            .collect();
        assert_eq!(titles, vec!["Sec1", "Sec2", "Sec3"]);
    }

    /// A secondary axis's `min_band` override must widen its OWN reserved band
    /// (not the primary's), narrowing the plot area and pushing every
    /// subsequent slot's stacked offset outward by the same delta — mirroring
    /// the primary y-axis's `min_band_reserves_larger_left_band` behavior.
    /// Regression test for the silently-dropped-override bug in
    /// secondary-y-axis-design (#52): `clamp_axis_band` was
    /// applied to the primary y band but not per-secondary-axis bands.
    #[test]
    fn compute_layout_secondary_y_min_band_widens_its_own_band_and_shifts_offsets() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 800.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let band = expected_secondary_band(&theme);

        let mut baseline_axes = dummy_axes();
        baseline_axes.secondary_y = n_secondary_axes(2);
        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &baseline_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let widened_min = band + 100.0;
        let mut widened_axes = dummy_axes();
        widened_axes.secondary_y = n_secondary_axes(2);
        widened_axes.secondary_y[0].overrides.placement.min_band = Some(widened_min);
        let widened = compute_layout(
            &spec,
            &theme,
            viewport,
            &widened_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let delta = widened_min - band;
        let base_w = baseline.panels[0].plot_area.w;
        let wide_w = widened.panels[0].plot_area.w;
        assert!(
            (base_w - wide_w - delta).abs() < 1e-6,
            "min_band on slot 0 must narrow the plot area by exactly the delta: base={base_w}, widened={wide_w}, delta={delta}"
        );

        let base_offsets: Vec<f64> = baseline
            .secondary_y_axes
            .iter()
            .map(|a| a.translate.unwrap_or(0.0))
            .collect();
        let wide_offsets: Vec<f64> = widened
            .secondary_y_axes
            .iter()
            .map(|a| a.translate.unwrap_or(0.0))
            .collect();
        assert!(
            (wide_offsets[0] - base_offsets[0]).abs() < 1e-6,
            "slot 0's own offset is unaffected by its own min_band"
        );
        assert!(
            (wide_offsets[1] - base_offsets[1] - delta).abs() < 1e-6,
            "slot 1 must stack past slot 0's WIDENED band: base={base_offsets:?}, widened={wide_offsets:?}, delta={delta}"
        );
    }

    /// The `max_band` mirror: capping a secondary axis's band below its
    /// natural width narrows the reservation (and subsequent offsets shrink
    /// by the same delta), never affecting the primary y-axis's own band.
    #[test]
    fn compute_layout_secondary_y_max_band_caps_its_own_band_and_shifts_offsets() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 800.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let band = expected_secondary_band(&theme);

        let mut baseline_axes = dummy_axes();
        baseline_axes.secondary_y = n_secondary_axes(2);
        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &baseline_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let capped_max = (band - 20.0).max(1.0);
        let mut capped_axes = dummy_axes();
        capped_axes.secondary_y = n_secondary_axes(2);
        capped_axes.secondary_y[0].overrides.placement.max_band = Some(capped_max);
        let capped = compute_layout(
            &spec,
            &theme,
            viewport,
            &capped_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let delta = band - capped_max;
        let base_w = baseline.panels[0].plot_area.w;
        let capped_w = capped.panels[0].plot_area.w;
        assert!(
            (capped_w - base_w - delta).abs() < 1e-6,
            "max_band on slot 0 must widen the plot area by exactly the delta: base={base_w}, capped={capped_w}, delta={delta}"
        );

        let base_offsets: Vec<f64> = baseline
            .secondary_y_axes
            .iter()
            .map(|a| a.translate.unwrap_or(0.0))
            .collect();
        let capped_offsets: Vec<f64> = capped
            .secondary_y_axes
            .iter()
            .map(|a| a.translate.unwrap_or(0.0))
            .collect();
        assert!(
            (base_offsets[1] - capped_offsets[1] - delta).abs() < 1e-6,
            "slot 1 must stack past slot 0's CAPPED (narrower) band: base={base_offsets:?}, capped={capped_offsets:?}, delta={delta}"
        );
    }

    /// Byte-stability: `axes.secondary_y` empty (the pre-#52 wire default) is
    /// the exact same `AxesInput` [`compute_layout_single_chart_no_facet_no_legend`]
    /// exercises, and reproduces every one of its assertions plus an empty
    /// `secondary_y_axes` — the new field is additive, not a behavior change,
    /// on the shared path.
    #[test]
    fn compute_layout_no_secondary_y_is_byte_stable_with_pre_52_shape() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let axes = dummy_axes();
        assert!(
            axes.secondary_y.is_empty(),
            "dummy_axes() default has no secondary axes"
        );
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

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
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed on minimal spec");

        assert_eq!(result.viewport, viewport.into_rect());
        assert_eq!(result.panels.len(), 1);
        assert_eq!(result.axes.len(), 2);
        assert!(result.legend.is_none());
        assert!(result.warnings.is_empty());
        assert!(result.secondary_y_axes.is_empty());

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
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        // Font-size-sensitive metrics: width scales with the supplied font size.
        let m = MockMetrics {
            measure: |t: &str, fs: f64| t.chars().count() as f64 * fs * 0.6,
            line_h_factor: 1.2,
        };

        let baseline = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &dummy_axes(),
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("baseline layout");

        let mut axes_big = dummy_axes();
        axes_big.y.overrides.labels.font_size = Some(40.0);
        let widened = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes_big,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
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

    // ── B5 unit 2: clamp_axis_band + orient band reservation ──────────────

    #[test]
    fn clamp_axis_band_passthrough_when_unset() {
        // Default path (both None) must return the dynamic value unchanged so
        // existing layouts stay byte-identical.
        assert_eq!(clamp_axis_band(37.5, None, None), 37.5);
    }

    #[test]
    fn clamp_axis_band_min_reserves_at_least() {
        assert_eq!(clamp_axis_band(20.0, Some(80.0), None), 80.0);
        // Already above min: unchanged.
        assert_eq!(clamp_axis_band(120.0, Some(80.0), None), 120.0);
    }

    #[test]
    fn clamp_axis_band_max_caps() {
        assert_eq!(clamp_axis_band(120.0, None, Some(40.0)), 40.0);
        // Already below max: unchanged.
        assert_eq!(clamp_axis_band(20.0, None, Some(40.0)), 20.0);
    }

    #[test]
    fn clamp_axis_band_max_wins_over_contradictory_min() {
        // min > max is a user contradiction; the cap (max) wins.
        assert_eq!(clamp_axis_band(50.0, Some(90.0), Some(40.0)), 40.0);
    }

    #[test]
    fn min_band_reserves_larger_left_band() {
        // A y-axis min_band of 200px must push the plot area right by at least
        // that much vs. the unset baseline.
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        let baseline = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &dummy_axes(),
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let mut axes = dummy_axes();
        axes.y.overrides.placement.min_band = Some(200.0);
        let widened = compute_layout(
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
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let base_x = baseline.panels[0].plot_area.x;
        let wide_x = widened.panels[0].plot_area.x;
        assert!(
            wide_x >= base_x + 100.0,
            "min_band=200 must reserve a much larger left band: base x={base_x}, widened x={wide_x}"
        );
        // The reserved left band is at least min_band.
        assert!(wide_x - viewport.into_rect().x >= 200.0);
    }

    #[test]
    fn x_orient_top_reserves_band_above_plot() {
        // orient="top" must reserve the x band on the TOP and free up the bottom,
        // mirroring the default Bottom layout. Compare against the Bottom baseline.
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        let mut bottom_axes = dummy_axes();
        bottom_axes.x.title.set(Some("x title".into()));
        let baseline = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &bottom_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let base_plot = baseline.panels[0].plot_area;

        let mut top_axes = dummy_axes();
        top_axes.x.title.set(Some("x title".into()));
        top_axes.x.orient = AxisOrient::Top;
        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &top_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let plot = result.panels[0].plot_area;

        // The reserved band moved from the bottom to the top: the top axis layout
        // has a larger plot.y (band reserved above) and its plot bottom reaches
        // lower than the bottom-oriented baseline's plot bottom (no bottom band).
        assert!(
            plot.y > base_plot.y + 5.0,
            "top-oriented x axis must reserve a top band: base y={}, top y={}",
            base_plot.y,
            plot.y
        );
        assert!(
            plot.y + plot.h > base_plot.y + base_plot.h + 5.0,
            "top-oriented x axis must free the bottom: base bottom={}, top bottom={}",
            base_plot.y + base_plot.h,
            plot.y + plot.h
        );
        // The emitted x axis line sits at the plot top.
        let x_axis = result
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Top)
            .unwrap();
        assert!((x_axis.axis_line.y - plot.y).abs() < 0.01);
    }

    #[test]
    fn compute_layout_invalid_viewport_errors() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let err = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 0.0,
                height: 400.0,
            },
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap_err();
        match err {
            LayoutError::InvalidViewport { width, .. } => assert_eq!(width, 0.0),
            other => panic!("expected InvalidViewport, got {:?}", other),
        }
    }

    /// `compute_layout` refuses a NaN or infinite viewport dimension with
    /// the same typed error as a zero one, with `padding.auto` on and off.
    #[test]
    fn compute_layout_refuses_a_non_finite_viewport() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        for auto in [true, false] {
            let mut theme = default_theme_inputs();
            theme.padding.padding_auto = auto;
            for (width, height) in [
                (f64::NAN, 400.0),
                (400.0, f64::NAN),
                (f64::INFINITY, 400.0),
                (400.0, f64::INFINITY),
            ] {
                let err = compute_layout(
                    &spec,
                    &theme,
                    Viewport { width, height },
                    &axes,
                    &[],
                    &[],
                    None,
                    None,
                    &m,
                    &legend::LegendOverrides::default(),
                    &[],
                    CompositeLayoutSeam::default(),
                )
                .unwrap_err();
                assert!(matches!(err, LayoutError::InvalidViewport { .. }), "{width}x{height}: {err:?}");
            }
        }
    }

    /// The close-and-resettle step ends by its own structure: each repeat
    /// closes an axis that was still open, so it runs at most three settles
    /// whatever the room's floors and budgets are. Given a NaN or infinite
    /// viewport directly (past `compute_layout`'s refusal), where every
    /// budget and floor is non-finite, it still returns.
    #[test]
    fn settle_auto_padding_ends_on_a_non_finite_viewport() {
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let axes = continuous_x_axes_with_long_first_and_last_labels();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let base = base_padding_insets(&theme);
        for (width, height) in [(f64::NAN, 200.0), (200.0, f64::NAN), (f64::INFINITY, 200.0), (200.0, f64::INFINITY)] {
            let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, Viewport { width, height });
            let _ = settle_auto_padding(&inputs, base);
        }
    }

    #[test]
    fn compute_layout_padding_exceeds_viewport_errors() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = ThemeInputs {
            padding: ThemePadding {
                padding: 100.0,
                ..ThemePadding::default()
            },
            ..ThemeInputs::default()
        };
        let err = compute_layout(
            &spec,
            &theme,
            Viewport {
                width: 50.0,
                height: 50.0,
            },
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap_err();
        match err {
            LayoutError::PaddingExceedsViewport {
                padding,
                side,
                viewport_dim,
            } => {
                assert_eq!(padding, 100.0);
                assert_eq!(side, "left");
                assert_eq!(viewport_dim, 50.0);
            }
            other => panic!("expected PaddingExceedsViewport, got {:?}", other),
        }
    }

    /// D10, spec §4.7 — repro: `configure_padding(top=
    /// 1e9)` previously reported the theme DEFAULT (16), not the caller's
    /// 1e9, and never named which side. `padding_top` is the only side set
    /// (mirroring `configure_padding(top=1e9)` — the other three sides stay
    /// theme-default), so `side` must name `"top"` and `padding` must be the
    /// caller's own 1e9, not the untouched default.
    #[test]
    fn compute_layout_padding_exceeds_viewport_names_the_callers_side_and_value() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = ThemeInputs {
            padding: ThemePadding {
                padding_top: Some(1e9),
                ..ThemePadding::default()
            },
            ..ThemeInputs::default()
        };
        let err = compute_layout(
            &spec,
            &theme,
            Viewport {
                width: 480.0,
                height: 360.0,
            },
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap_err();
        match err {
            LayoutError::PaddingExceedsViewport {
                padding,
                side,
                viewport_dim,
            } => {
                assert_eq!(
                    padding, 1e9,
                    "must report the CALLER's value, not the theme default"
                );
                assert_eq!(side, "top");
                assert_eq!(viewport_dim, 360.0);
            }
            other => panic!("expected PaddingExceedsViewport, got {:?}", other),
        }
        let msg = err.to_string();
        assert!(
            msg.contains("top=1000000000"),
            "message must name the side and caller value: {msg}"
        );
    }

    // ── padding.auto (D10, spec §4.7, F-L07-08) ─────────────────────────────

    /// #135: the Rust built-in default is the single owner of
    /// `padding.auto`, and it is `true` — a chart that never mentions
    /// padding still gets the edge-padding rescue. There is no theme key
    /// for it (the field lives directly on `ThemeInputs::default()`).
    #[test]
    fn theme_padding_default_has_auto_on() {
        assert!(
            ThemeInputs::default().padding.padding_auto,
            "padding.auto's built-in default must be true (#135)",
        );
        assert!(
            ThemePadding::default().padding_auto,
            "ThemePadding::default() must also carry auto on directly",
        );
    }

    /// A continuous x-axis (`tick_projection: Some`) whose last tick's label
    /// is a long number, at a narrow viewport with default (16px) padding.
    fn continuous_x_axes_with_long_last_label() -> AxesInput {
        let mut x = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["0".into(), "999999".into()],
            None,
        );
        x.tick_projection = Some(TickProjection {
            padding_frac: 0.0,
            major: vec![0.0, 1.0],
            minor: vec![],
        });
        let y = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "5".into(), "10".into()],
            None,
        );
        AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// The drawn x-label overhang [`probe_auto_padding`] measures at
    /// `theme`'s `auto=false` inset: the labels exactly as a chart without
    /// `padding.auto` lays them out, for tests that pin
    /// `auto_padding_for_edge_ticks` against a hand-derived reach.
    fn x_overhang_without_auto(
        theme: &ThemeInputs,
        axes: &AxesInput,
        metrics: &dyn TextMetrics,
        viewport: Viewport,
    ) -> XLabelOverhang {
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, theme, axes, metrics, viewport);
        let base = base_padding_insets(theme);
        probe_auto_padding(base, &inputs, &AutoPlotRoom::of_auto_off(&inputs, base)).x_overhang
    }

    /// The [`AutoPlotRoom`] `settle_auto_padding` measures for a chart
    /// with no legend or facets, for tests that solve one auto-padding step
    /// by hand.
    fn plot_room(
        theme: &ThemeInputs,
        axes: &AxesInput,
        metrics: &dyn TextMetrics,
        viewport: Viewport,
    ) -> AutoPlotRoom {
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, theme, axes, metrics, viewport);
        AutoPlotRoom::of_auto_off(&inputs, base_padding_insets(theme))
    }

    /// [`LayoutInputs`] for a chart with no legend or facets.
    fn plain_layout_inputs<'a>(
        spec: &'a crate::spec::chart::ChartSpec,
        legend_overrides: &'a legend::LegendOverrides,
        theme: &'a ThemeInputs,
        axes: &'a AxesInput,
        metrics: &'a dyn TextMetrics,
        viewport: Viewport,
    ) -> LayoutInputs<'a> {
        LayoutInputs {
            spec,
            theme,
            viewport,
            axes,
            facet_groups: &[],
            legend_entries: &[],
            legend_title: None,
            colorbar: None,
            metrics,
            legend_overrides,
            aux_legend_inputs: &[],
            seam: CompositeLayoutSeam::default(),
        }
    }

    /// Unit-level pin of `auto_padding_for_edge_ticks`'s concrete numbers
    /// (hand-derived in the task's own working notes, reproduced here):
    /// last label `"999999"` measures 60px (6 chars * 10px), so its half-
    /// width (30) exceeds the flat 16px right padding by 14 — that shortfall
    /// is exactly what `extra.right` must be. The first label `"0"`
    /// (half-width 5) never exceeds its cushion (16px padding + the 26px
    /// y-gutter), so `extra.left` stays 0 and `extra.top`/`extra.bottom`
    /// are untouched by an x-only correction.
    #[test]
    fn auto_padding_for_edge_ticks_reserves_exactly_the_overhang() {
        let viewport = Viewport {
            width: 400.0,
            height: 200.0,
        };
        let axes = continuous_x_axes_with_long_last_label();
        let theme = default_theme_inputs();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let extra = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(
            (extra.right - 14.0).abs() < 1e-9,
            "extra.right = {}",
            extra.right
        );
        assert_eq!(extra.left, 0.0);
        assert_eq!(extra.bottom, 0.0);
    }

    /// A continuous x-axis whose FIRST tick's label is a long negative
    /// number, paired with a y-axis whose `max_band` caps its reserved
    /// gutter well below its natural (unclamped) width.
    fn continuous_x_axes_with_long_first_label_and_capped_y_band() -> AxesInput {
        let mut x = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["-999999999999999".into(), "0".into()],
            None,
        );
        x.tick_projection = Some(TickProjection {
            padding_frac: 0.0,
            major: vec![0.0, 1.0],
            minor: vec![],
        });
        let mut y = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "5".into(), "10".into()],
            None,
        );
        y.overrides.placement.max_band = Some(10.0);
        AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// The cushion estimate previously used
    /// the UNCLAMPED y label band, so `fm.Axis(max_band=...)` capping the
    /// REAL reserved gutter smaller made the cushion look bigger than it
    /// really is and silently under-reserved `extra.left`. Hand-derived
    /// numbers: raw y label band is 26 (`tick_size 4 + label_pad 2 +
    /// max_label_w 20` for `"10"`), `max_band=10` caps it to 10, so the
    /// left cushion is `16 (default padding) + 10 (clamped y band) = 26` —
    /// NOT `16 + 26 = 42` (the pre-fix, unclamped figure). The first
    /// label (`"-999999999999999"`, 16 chars) has half-width 80, so
    /// `extra.left` must be `80 - 26 = 54`, not the under-reserved
    /// `80 - 42 = 38` the unclamped cushion would have produced.
    #[test]
    fn auto_padding_for_edge_ticks_honors_max_band_when_computing_the_cushion() {
        let viewport = Viewport {
            width: 400.0,
            height: 200.0,
        };
        let axes = continuous_x_axes_with_long_first_label_and_capped_y_band();
        let theme = default_theme_inputs();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let extra = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(
            (extra.left - 54.0).abs() < 1e-9,
            "extra.left = {} (expected 54.0 = 80 - 26; the pre-fix unclamped \
             cushion would have wrongly produced 38 = 80 - 42)",
            extra.left,
        );
    }

    /// Integration-level sibling of
    /// `padding_auto_prevents_last_x_tick_label_from_overhanging_the_viewport`,
    /// through the real `compute_layout` path, pinning the reviewer's live
    /// repro: a long first x tick label
    /// clips past the LEFT canvas edge, identically under auto=True and
    /// auto=False, when the cushion estimate ignores `max_band`. After the
    /// fix, `auto=True` keeps it within bounds.
    #[test]
    fn padding_auto_honors_max_band_and_prevents_left_tick_overhang() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_first_label_and_capped_y_band();
        // Tall enough that the x label band the narrower plot reserves
        // leaves half the plot's height.
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();

        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("baseline layout should succeed");
        let x_axis = baseline
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let first_tick = x_axis.ticks.first().expect("x axis has ticks");
        let half_w = m.measure_width(&first_tick.label, theme.typography.label_font_size) / 2.0;
        assert!(
            first_tick.position - half_w < 0.0,
            "fixture assumption broken: the capped y max_band should already clip the \
             first x tick label past the left edge (position {} - half-width {} should be < 0)",
            first_tick.position,
            half_w,
        );

        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should succeed");
        let x_axis2 = with_auto
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let first_tick2 = x_axis2.ticks.first().expect("x axis has ticks");
        let half_w2 = m.measure_width(&first_tick2.label, theme.typography.label_font_size) / 2.0;
        assert!(
            first_tick2.position - half_w2 >= -1e-9,
            "padding.auto=true should keep the first x tick label within the viewport \
             even with max_band capping the y gutter (position {} - half-width {} should be >= 0)",
            first_tick2.position,
            half_w2,
        );
    }

    /// The pin the brief asks for: "a chart whose labels would clip at
    /// default padding renders unclipped under auto" — proven on CONCRETE
    /// geometry (the real tick's pixel position plus its measured half-
    /// width against the real viewport width), through the actual
    /// `compute_layout` production path, not a hand-built shape.
    #[test]
    fn padding_auto_prevents_last_x_tick_label_from_overhanging_the_viewport() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();

        // Baseline (auto=false, the flat 16px default): the fixture must
        // actually clip, or this test proves nothing about the fix.
        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("baseline layout should succeed");
        let x_axis = baseline
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let last_tick = x_axis.ticks.last().expect("x axis has ticks");
        let half_w = m.measure_width(&last_tick.label, theme.typography.label_font_size) / 2.0;
        assert!(
            last_tick.position + half_w > baseline.viewport.w,
            "fixture assumption broken: default padding should already clip the last x \
             tick label (position {} + half-width {} should exceed viewport width {})",
            last_tick.position,
            half_w,
            baseline.viewport.w,
        );

        // Same chart, `padding.auto = true`: the SAME last tick's label no
        // longer overhangs past the viewport edge.
        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should succeed");
        let x_axis2 = with_auto
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let last_tick2 = x_axis2.ticks.last().expect("x axis has ticks");
        let half_w2 = m.measure_width(&last_tick2.label, theme.typography.label_font_size) / 2.0;
        assert!(
            last_tick2.position + half_w2 <= with_auto.viewport.w + 1e-9,
            "padding.auto=true should keep the last x tick label within the viewport \
             (position {} + half-width {} should not exceed viewport width {})",
            last_tick2.position,
            half_w2,
            with_auto.viewport.w,
        );
    }

    /// An ordinal/nominal axis (`tick_projection: None`) has no edge overhang
    /// to correct — `dummy_axes()` is ordinal-shaped — so `padding.auto`
    /// must leave its layout byte-identical to the non-auto render (claim 3:
    /// byte-identity for charts that don't exercise the touched field).
    #[test]
    fn padding_auto_is_byte_identical_for_a_chart_with_no_continuous_axis_or_title() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();

        let without_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme
        };
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        assert_eq!(without_auto, with_auto);
    }

    // ── padding.auto: axis-title recentering ──

    #[test]
    fn recentering_padding_no_correction_when_already_fits() {
        // Center 100, half-extent 20 -> edges [80, 120], well inside [0, 200].
        assert_eq!(recentering_padding(100.0, 20.0, 200.0), (0.0, 0.0));
    }

    #[test]
    fn recentering_padding_shifts_right_edge_back_when_only_the_right_overflows() {
        // Center 150, half-extent 60 -> edges [90, 210]; only the right (210)
        // overflows a 200-wide span, by 10. The minimal fix adds ONLY to the
        // right side (shifting the center left by 10) — extra_left stays 0.
        assert_eq!(recentering_padding(150.0, 60.0, 200.0), (0.0, 20.0));
    }

    #[test]
    fn recentering_padding_shifts_left_edge_back_when_only_the_left_overflows() {
        // Center 50, half-extent 60 -> edges [-10, 110]; only the left (-10)
        // overflows, by 10. The minimal fix adds ONLY to the left side.
        assert_eq!(recentering_padding(50.0, 60.0, 200.0), (20.0, 0.0));
    }

    #[test]
    fn recentering_padding_is_zero_when_the_element_is_wider_than_the_span() {
        // half-extent 150 -> full width 300 > span 200: no shift can satisfy
        // both edges regardless of baseline_center (the genuine, narrow
        // residual: an element wider than the viewport itself). Returns
        // (0.0, 0.0) rather than adding unbounded, unhelpful padding.
        assert_eq!(recentering_padding(100.0, 150.0, 200.0), (0.0, 0.0));
        // Off-center baseline: still infeasible (2*half_extent > span is the
        // exact feasibility test, independent of where the center starts).
        assert_eq!(recentering_padding(30.0, 150.0, 200.0), (0.0, 0.0));
    }

    /// `recentering_is_feasible` is the SAME `2*half_extent <= span` test
    /// `recentering_padding` uses internally, exposed so `compute_layout`
    /// can tell "no correction needed" apart from "genuinely infeasible" —
    /// both of which `recentering_padding` collapses to `(0.0, 0.0)`.
    #[test]
    fn recentering_is_feasible_matches_recentering_padding_own_bound() {
        assert!(recentering_is_feasible(20.0, 200.0), "well within span");
        assert!(
            recentering_is_feasible(100.0, 200.0),
            "exactly the boundary (2*100 == 200)"
        );
        assert!(
            !recentering_is_feasible(150.0, 200.0),
            "wider than the span"
        );
        // Independent of baseline_center, per recentering_padding's own derivation.
        assert!(recentering_is_feasible(60.0, 200.0));
    }

    /// A short-tick-label chart (so `auto_padding_for_edge_ticks` contributes
    /// nothing, isolating the title-only effect) with a long x-axis title.
    fn axes_with_wide_x_title() -> AxesInput {
        let x = axis_input(
            AxisOrient::Bottom,
            Some("A Wide X Axis Title".into()), // 19 chars
            vec!["0".into(), "1".into()],
            None,
        );
        let y = axis_input(AxisOrient::Left, None, vec!["0".into(), "1".into()], None);
        AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// Unit-level pin of `auto_padding_for_axis_titles`'s concrete numbers
    /// (hand-derived): y gutter is 16 (`tick_size 4 + label_pad 2 +
    /// max_label_w 10` for the single-digit y labels), so at
    /// `viewport.width = 200` the baseline plot spans `[32, 184]`
    /// (`x = 16 + 16`, `w = 200 - 16 - 16 - 16 = 152`), centered at 108. The
    /// title (`"A Wide X Axis Title"`, 19 chars * 10px = 190px, half-width
    /// 95) would span `[13, 203]` around that center — only the right edge
    /// (203) overflows the 200-wide viewport, by 3, so `extra.right` must be
    /// exactly `6` (`2 * 3`) and `extra.left` must stay `0`.
    #[test]
    fn auto_padding_for_axis_titles_recenters_a_right_overflowing_x_title() {
        let axes = axes_with_wide_x_title();
        let theme = default_theme_inputs();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let base = base_padding_insets(&theme);
        let extra =
            auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());
        assert!(
            (extra.right - 6.0).abs() < 1e-9,
            "extra.right = {}",
            extra.right
        );
        assert_eq!(extra.left, 0.0);
        assert_eq!(extra.top, 0.0);
        assert_eq!(extra.bottom, 0.0);

        let (x_infeasible, y_infeasible) = axis_title_infeasibility(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default(), &plot_room(&theme, &axes, &m, viewport));
        assert!(
            !x_infeasible,
            "this title fits within the viewport once recentered"
        );
        assert!(!y_infeasible, "no y title in this fixture");
    }

    /// Finding 1's own repro, proven through the real `compute_layout` path:
    /// a chart whose x-axis title would clip past the viewport's right edge
    /// at default (symmetric) padding renders fully within bounds under
    /// `padding.auto = true` — the ASYMMETRIC recentering case the
    /// symmetric-only derivation this replaces incorrectly ruled out.
    #[test]
    fn padding_auto_recenters_an_overflowing_x_title_within_the_viewport() {
        let spec = minimal_chart_spec();
        let axes = axes_with_wide_x_title();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();

        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("baseline layout should succeed");
        let x_axis = baseline
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let title = x_axis.title.as_ref().expect("x axis has a title");
        let half_w = m.measure_width(&title.text, theme.typography.title_font_size) / 2.0;
        assert!(
            title.anchor_x + half_w > baseline.viewport.w,
            "fixture assumption broken: default padding should already clip the x \
             title (anchor_x {} + half-width {} should exceed viewport width {})",
            title.anchor_x,
            half_w,
            baseline.viewport.w,
        );

        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should succeed");
        let x_axis2 = with_auto
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let title2 = x_axis2.title.as_ref().expect("x axis has a title");
        let half_w2 = m.measure_width(&title2.text, theme.typography.title_font_size) / 2.0;
        assert!(
            title2.anchor_x + half_w2 <= with_auto.viewport.w + 1e-9,
            "padding.auto=true should keep the x title within the viewport \
             (anchor_x {} + half-width {} should not exceed viewport width {})",
            title2.anchor_x,
            half_w2,
            with_auto.viewport.w,
        );
        assert!(
            title2.anchor_x - half_w2 >= -1e-9,
            "padding.auto=true's recentering must not push the title past the LEFT \
             edge while fixing the right (anchor_x {} - half-width {} should be >= 0)",
            title2.anchor_x,
            half_w2,
        );
    }

    /// The genuine residual (a narrowed
    /// claim): a title wider than the viewport ITSELF cannot be fully
    /// rescued by any padding arrangement. `padding.auto=true` must not
    /// turn this into a hard failure (no `PaddingExceedsViewport`), and —
    /// in this ISOLATED fixture, where no other auto-padding pass fires on
    /// the same axis (a short 2-tick label, no `max_band`, no overhang) —
    /// it renders BYTE-IDENTICAL to `auto=false`: the title pass's own
    /// contribution is `(0.0, 0.0)` and there is nothing else to compose it
    /// with. (This is NOT the general
    /// guarantee — see
    /// `padding_auto_never_worsens_an_infeasible_title_when_a_tick_correction_also_fires`
    /// below for the composed case, where a same-axis tick correction used
    /// to make an infeasible title's clip strictly WORSE than `auto=false`
    /// before this fix, contradicting a since-corrected version of this
    /// docstring that claimed byte-identity applied universally.)
    #[test]
    fn padding_auto_does_not_blow_up_on_a_title_wider_than_the_viewport() {
        let spec = minimal_chart_spec();
        let mut axes = axes_with_wide_x_title();
        // A title far wider than any reasonable viewport.
        axes.x.title.set(Some("A".repeat(100)));
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();
        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("baseline (auto=false) must render");
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        );
        assert!(
            with_auto.is_ok(),
            "an unfixably-wide title must still render, not refuse: {with_auto:?}"
        );
        assert_eq!(
            with_auto.unwrap(),
            baseline,
            "with no competing auto-padding pass on this axis, an infeasible \
             title's auto=true render must be byte-identical to auto=false",
        );
    }

    /// The composed sibling of the test above:
    /// `padding_auto_keeps_title_on_canvas_when_a_tick_correction_also_fires`
    /// proved the FEASIBLE case, where the title pass's solve, evaluated
    /// against the tick-adjusted base, fully recenters the title. This is
    /// the INFEASIBLE case — same tick-overhang fixture
    /// (`axes_with_left_tick_overhang_and_a_title_the_shift_would_clip`),
    /// narrowed so the 66-char (660px-wide, half-width 330) title no longer
    /// fits the viewport at all (`2*330 = 660 > 300`). Pre-fix,
    /// `auto_padding_for_axis_titles` still reported `(0.0, 0.0)` for x (it
    /// cannot help), but the tick pass's real, asymmetric left expansion
    /// applied anyway — shifting the plot's (and title's) center and
    /// turning a real but bounded clip into a WORSE one. Reconstructed at
    /// the Rust unit-test level from the quality reviewer's live (real-font)
    /// Python repro (`W=280`: `9.51px -> 26.53px` clip, `auto=True` strictly
    /// worse than `auto=False`) using this file's `MockMetrics`-based
    /// fixture family, since the reviewer's fixture depended on real font
    /// metrics unavailable to a Rust-only unit test.
    #[test]
    fn padding_auto_never_worsens_an_infeasible_title_when_a_tick_correction_also_fires() {
        let spec = minimal_chart_spec();
        let axes = axes_with_left_tick_overhang_and_a_title_the_shift_would_clip();
        let viewport = Viewport {
            width: 300.0,
            height: 300.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();

        // Fixture assumption: the title really is infeasible at this width
        // (unlike the 700px sibling fixture, where it is feasible).
        let title_half_w = m.measure_width(&"a".repeat(66), theme.typography.title_font_size) / 2.0;
        assert!(
            !recentering_is_feasible(title_half_w, viewport.width),
            "fixture assumption broken: title half-width {title_half_w} should \
             exceed half the {}-wide viewport",
            viewport.width,
        );
        // And the tick pass genuinely wants left padding here — this test
        // would be vacuous if the fixture stopped exercising the tick pass.
        let ticks_alone = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(
            ticks_alone.left > 0.0,
            "fixture assumption broken: the tick pass should still fire"
        );

        let baseline = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("baseline (auto=false) layout should succeed");

        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should still render, not refuse");

        // The genuine guarantee: never worse than auto=false.
        // Since this axis's combined auto contribution is suppressed to
        // exactly zero in the infeasible regime, the two renders are
        // byte-identical -- not merely "no worse".
        assert_eq!(
            with_auto, baseline,
            "an infeasible title's auto=true render must be byte-identical to \
             auto=false when a same-axis tick correction cannot be applied \
             without decentering an already-unfixable title",
        );
    }

    // ── padding.auto: composition ─────

    #[test]
    fn cap_auto_scale_passes_through_when_it_already_fits() {
        // fixed 16+16=32, extra 10+10=20, span 200: 32+20=52 << 200, no cap.
        assert_eq!(cap_auto_scale(16.0, 16.0, 20.0, 200.0, f64::INFINITY), 1.0);
    }

    #[test]
    fn cap_auto_scale_scales_proportionally_when_over_budget() {
        // fixed 16+16=32, span 100, MIN_PANEL_DIM 1 -> budget = 100-32-1 = 67.
        // desired 60+30=90 > 67, scale = 67/90, so the sum consumes the budget.
        let scale = cap_auto_scale(16.0, 16.0, 90.0, 100.0, f64::INFINITY);
        assert!((scale - 67.0 / 90.0).abs() < 1e-12, "scale = {scale}");
        assert!((90.0 * scale - 67.0).abs() < 1e-9, "the sum should exactly consume the budget");
    }

    #[test]
    fn cap_auto_scale_is_zero_when_the_fixed_floor_alone_overflows() {
        // fixed 100+100=200 already >= span 150: pre-existing failure, auto
        // must contribute nothing (not attempt a negative-budget scale).
        assert_eq!(cap_auto_scale(100.0, 100.0, 10.0, 150.0, f64::INFINITY), 0.0);
    }

    /// The plot room is the tighter bound when it is smaller than the
    /// viewport's: 90px wanted, the viewport allows 167 but the plot room
    /// 40, so the extra is scaled to 40; 40px, exactly the room, ships
    /// whole. A plot room of 0 still lets nothing through.
    #[test]
    fn cap_auto_scale_honors_the_plot_room_when_it_is_tighter() {
        let scale = cap_auto_scale(16.0, 16.0, 90.0, 200.0, 40.0);
        assert!((90.0 * scale - 40.0).abs() < 1e-9, "scaled to the plot room: {}", 90.0 * scale);
        assert_eq!(cap_auto_scale(16.0, 16.0, 40.0, 200.0, 40.0), 1.0, "exactly the room");
        assert_eq!(cap_auto_scale(16.0, 16.0, 40.0, 200.0, 0.0), 0.0, "no room");
    }

    /// #139(2): `compute_layout` must decide per-axis infeasibility and
    /// zero the same-axis tick contribution BEFORE the title pass solves —
    /// not solve titles first and zero the combined total afterward. The
    /// two orders disagree here because the y-title's baseline center is
    /// computed from `plot_h`, which folds in the x-axis's own label band
    /// (keyed on `plot_w`), which is a function of the SAME `base.left`/
    /// `base.right` the x-tick pass writes into: whether that x-tick
    /// contribution is suppressed before or after the y-title solve changes
    /// the y-title's own correction, even though the x-title's own
    /// correction (already `(0.0, 0.0)` either way in the infeasible
    /// regime) does not care.
    #[test]
    fn compute_layout_solves_titles_against_the_post_suppression_base() {
        let spec = minimal_chart_spec();
        let mut axes = axes_with_left_tick_overhang_and_a_title_the_shift_would_clip();
        axes.y.title.set(Some("a".repeat(18)));
        let theme = theme_inputs_with_padding_auto_off();
        let viewport = Viewport {
            width: 270.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        // Fixture assumptions: x's title is infeasible, and the x-tick pass
        // still reserves real left padding — the same combination
        // `padding_auto_never_worsens_an_infeasible_title_when_a_tick_correction_also_fires`
        // pins for the SAME-axis case; this is its cross-axis sibling.
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(ticks.left > 0.0, "fixture must exercise the tick pass");
        let x_half_w = m.measure_width(&"a".repeat(66), theme.typography.title_font_size) / 2.0;
        assert!(
            !recentering_is_feasible(x_half_w, viewport.width),
            "fixture assumption: x title half-width {x_half_w} should exceed \
             half the {}-wide viewport",
            viewport.width,
        );
        // This fixture's x-axis tick overhang must not itself want any
        // top/bottom padding (only a y-axis tick overhang would), or the
        // "raw base, no y-tick contribution" derivation below is wrong.
        assert_eq!(
            (ticks.top, ticks.bottom),
            (0.0, 0.0),
            "fixture assumption: no y-driven top/bottom tick contribution"
        );

        let base = base_padding_insets(&theme);

        // The CORRECT order (#139(2)): x's tick contribution is suppressed
        // before the title pass solves, since x's own infeasibility is
        // decided from title measurement and viewport alone. Since
        // ticks.top/bottom are asserted zero above, solving against the raw
        // `base` here is equivalent to solving against the suppressed one.
        let titles_against_suppressed =
            auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());

        // The pre-fix order: titles solved against the base with the FULL,
        // unsuppressed x-tick contribution folded in.
        let base_after_unsuppressed_ticks = Inset {
            top: base.top + ticks.top,
            right: base.right + ticks.right,
            bottom: base.bottom + ticks.bottom,
            left: base.left + ticks.left,
        };
        let titles_against_unsuppressed = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base_after_unsuppressed_ticks, Inset::default());

        assert_ne!(
            titles_against_suppressed.top, titles_against_unsuppressed.top,
            "the two orders must disagree on the y title's own correction, \
             or this fixture is vacuous",
        );

        // The top/bottom cap must not fire on this fixture, or the "raw
        // base" derivation below (which applies no cap) is wrong: the
        // top/bottom total `cap_auto_scale` receives must pass through
        // unchanged.
        assert_eq!(
            cap_auto_scale(
                base.top,
                base.bottom,
                titles_against_suppressed.top + titles_against_suppressed.bottom,
                viewport.height,
                plot_room(&theme, &axes, &m, viewport).height_budget,
            ),
            1.0,
            "fixture assumption: the top/bottom cap must not fire",
        );

        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should still render");

        // `compute_layout` must ship the geometry solved against the
        // SUPPRESSED base, not the pre-fix unsuppressed one.
        assert!(
            (with_auto.padded_region.y - (base.top + titles_against_suppressed.top)).abs() < 1e-9,
            "compute_layout's top inset ({}) must match the post-suppression \
             solve ({}), not the pre-fix unsuppressed solve ({})",
            with_auto.padded_region.y,
            base.top + titles_against_suppressed.top,
            base.top + titles_against_unsuppressed.top,
        );
    }

    /// A continuous x-axis whose FIRST and LAST tick labels are both long
    /// enough (19 characters, half-width 95 at 10px/char) to want real
    /// overhang padding on both sides at once, on a 150px viewport.
    /// Each side's need alone fits that side (it stays below the plot's
    /// width, so it is not an infeasible charge), but together they exceed
    /// the viewport's budget, so the cap fires. `label_overlap=True`
    /// (`ShowAll`) keeps them flat on a plot narrower than they are; the
    /// collision cascade would otherwise rotate them, and the drawn labels
    /// would no longer overhang enough to make the cap fire.
    fn continuous_x_axes_with_long_first_and_last_labels() -> AxesInput {
        let mut x = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["-".to_string() + &"9".repeat(18), "9".repeat(19)],
            None,
        );
        x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        x.tick_projection = Some(TickProjection {
            padding_frac: 0.0,
            major: vec![0.0, 1.0],
            minor: vec![],
        });
        let y = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "5".into(), "10".into()],
            None,
        );
        AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// #139(3): the two `cap_auto_scale` call sites must receive an
    /// explicit side's own extra as ZERO, not the real (but doomed-to-be-
    /// discarded) value the tick pass computed — `compute_layout`'s own
    /// final inset assembly (`theme.padding.padding_left.unwrap_or(...)`)
    /// throws an explicit side's `extra` away entirely regardless. Pre-fix,
    /// `cap_auto_scale` shared its scarce budget between that doomed
    /// extra and the auto (right) side's real one, throttling the auto side
    /// more than necessary for a contribution that was never going to ship.
    #[test]
    fn compute_layout_zeroes_an_explicit_sides_extra_before_the_cap() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_first_and_last_labels();
        let mut theme = default_theme_inputs();
        theme.padding.padding_auto = true;
        theme.padding.padding_left = Some(5.0);
        // Tall enough that the x label band (reserved as if the 190px
        // labels rotated to -90) leaves the plot a real height, and wide
        // enough that each label's own extra keeps half the plot.
        let viewport = Viewport {
            width: 250.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        // Fixture assumptions: both sides want real extra, and together
        // they exceed the axis's budget (the cap must actually fire, or
        // this test is vacuous).
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(
            ticks.left > 0.0 && ticks.right > 0.0,
            "fixture must want both sides: left={}, right={}",
            ticks.left,
            ticks.right,
        );
        let base = base_padding_insets(&theme);
        let room = plot_room(&theme, &axes, &m, viewport);
        let budget =
            (viewport.width - base.left - base.right - MIN_PANEL_DIM).min(room.width_budget);
        assert!(
            ticks.left + ticks.right > budget,
            "fixture must exceed budget, or the cap never fires: {} + {} <= {budget}",
            ticks.left,
            ticks.right,
        );

        // The CORRECT cap input: the explicit left side's extra is zero, so
        // only the right side's real extra competes for the budget.
        let expected_right = ticks.right
            * cap_auto_scale(base.left, base.right, ticks.right, viewport.width, room.width_budget);
        assert!(
            (expected_right - ticks.right).abs() < 1e-9,
            "fixture assumption: with the left extra zeroed, the right side \
             should fit uncapped ({expected_right} should equal {})",
            ticks.right,
        );

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should still render");

        // The explicit side ships verbatim either way — this is a control,
        // not the discriminating assertion.
        assert!(
            (with_auto.padded_region.x - 5.0).abs() < 1e-9,
            "explicit padding_left must survive verbatim: {}",
            with_auto.padded_region.x,
        );

        let actual_right_inset =
            viewport.width - with_auto.padded_region.x - with_auto.padded_region.w;
        assert!(
            (actual_right_inset - (base.right + expected_right)).abs() < 1e-9,
            "the right (auto) side's inset ({actual_right_inset}) must be \
             capped as if the explicit left side's extra were zero ({}), \
             not throttled to share budget with a contribution that never \
             ships",
            base.right + expected_right,
        );
    }

    /// Mirror of `compute_layout_zeroes_an_explicit_sides_extra_before_the_cap`
    /// with the explicit/auto sides swapped: an explicit RIGHT side's own
    /// extra must be zeroed before `cap_auto_scale` too, so it cannot
    /// compete for budget with the auto LEFT side's real contribution.
    #[test]
    fn compute_layout_zeroes_an_explicit_right_sides_extra_before_the_cap() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_first_and_last_labels();
        let mut theme = default_theme_inputs();
        theme.padding.padding_auto = true;
        theme.padding.padding_right = Some(5.0);
        // Tall enough that the x label band (reserved as if the 190px
        // labels rotated to -90) leaves the plot a real height, and wide
        // enough that each label's own extra keeps half the plot.
        let viewport = Viewport {
            width: 250.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        // Fixture assumptions: both sides want real extra, and together
        // they exceed the axis's budget (the cap must actually fire, or
        // this test is vacuous).
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(
            ticks.left > 0.0 && ticks.right > 0.0,
            "fixture must want both sides: left={}, right={}",
            ticks.left,
            ticks.right,
        );
        let base = base_padding_insets(&theme);
        let room = plot_room(&theme, &axes, &m, viewport);
        let budget =
            (viewport.width - base.left - base.right - MIN_PANEL_DIM).min(room.width_budget);
        assert!(
            ticks.left + ticks.right > budget,
            "fixture must exceed budget, or the cap never fires: {} + {} <= {budget}",
            ticks.left,
            ticks.right,
        );

        // The CORRECT cap input: the explicit right side's extra is zero, so
        // only the left side's real extra competes for the budget.
        let expected_left = ticks.left
            * cap_auto_scale(base.left, base.right, ticks.left, viewport.width, room.width_budget);
        assert!(
            (expected_left - ticks.left).abs() < 1e-9,
            "fixture assumption: with the right extra zeroed, the left side \
             should fit uncapped ({expected_left} should equal {})",
            ticks.left,
        );

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should still render");

        // The explicit side ships verbatim either way — this is a control,
        // not the discriminating assertion.
        let actual_explicit_right_inset =
            viewport.width - with_auto.padded_region.x - with_auto.padded_region.w;
        assert!(
            (actual_explicit_right_inset - 5.0).abs() < 1e-9,
            "explicit padding_right must survive verbatim: {}",
            actual_explicit_right_inset,
        );

        assert!(
            (with_auto.padded_region.x - (base.left + expected_left)).abs() < 1e-9,
            "the left (auto) side's inset ({}) must be capped as if the \
             explicit right side's extra were zero ({}), not throttled to \
             share budget with a contribution that never ships",
            with_auto.padded_region.x,
            base.left + expected_left,
        );
    }

    /// A continuous y axis titled `y_title` (10px per character, so its
    /// half extent is `5 * len`) beside an x axis on TOP. The 40px label font
    /// makes the y-tick pass want 22px of top padding at a 2px base, and the
    /// tall top x band pushes a feasible y title's center low enough that
    /// its recentering asks for extra BOTTOM padding.
    fn vertical_padding_fixture(y_title: &str) -> (AxesInput, ThemeInputs, impl TextMetrics) {
        let x = axis_input(AxisOrient::Top, None, vec!["0".into(), "1".into()], None);
        let mut y = axis_input(
            AxisOrient::Left,
            Some(y_title.to_string()),
            vec!["0".into(), "5".into(), "10".into()],
            None,
        );
        y.tick_projection = Some(TickProjection {
            padding_frac: 0.0,
            major: vec![0.0, 0.5, 1.0],
            minor: vec![],
        });
        let axes = AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };
        let mut theme = default_theme_inputs();
        theme.padding.padding_auto = true;
        theme.padding.padding = 2.0;
        theme.typography.label_font_size = 40.0;
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        (axes, theme, m)
    }

    fn auto_layout(
        theme: &ThemeInputs,
        viewport: Viewport,
        axes: &AxesInput,
        m: &dyn TextMetrics,
    ) -> LayoutResult {
        compute_layout(
            &minimal_chart_spec(),
            theme,
            viewport,
            axes,
            &[],
            &[],
            None,
            None,
            m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should render")
    }

    /// The y title renders rotated along the vertical axis, so its
    /// feasibility is judged against the viewport HEIGHT. This fixture's y
    /// title (100px) is taller than the 80px viewport but far narrower than
    /// its 400px width, so a width-judged verdict would call it feasible.
    #[test]
    fn axis_title_infeasibility_judges_the_y_title_against_the_viewport_height() {
        let (axes, theme, m) = vertical_padding_fixture(&"a".repeat(10));
        let viewport = Viewport {
            width: 400.0,
            height: 80.0,
        };
        assert_eq!(
            axis_title_infeasibility(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base_padding_insets(&theme), Inset::default(), &plot_room(&theme, &axes, &m, viewport)),
            (false, true),
            "an untitled x axis is feasible; a y title taller than the viewport is not",
        );
    }

    /// An infeasible y title suppresses the y-tick pass's top/bottom
    /// contribution, the vertical sibling of the x title's left/right
    /// suppression: the shipped top inset is the bare base.
    #[test]
    fn compute_layout_suppresses_the_vertical_tick_pass_on_an_infeasible_y_title() {
        let (axes, theme, m) = vertical_padding_fixture(&"a".repeat(10));
        let viewport = Viewport {
            width: 400.0,
            height: 80.0,
        };
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(ticks.top > 0.0, "fixture must exercise the y-tick pass: {ticks:?}");

        let layout = auto_layout(&theme, viewport, &axes, &m);
        let base = base_padding_insets(&theme);
        assert!(
            (layout.padded_region.y - base.top).abs() < 1e-9,
            "the top inset ({}) must be the bare base ({}), not base plus the \
             suppressed tick contribution ({})",
            layout.padded_region.y,
            base.top,
            base.top + ticks.top,
        );
    }

    /// The vertical sibling of
    /// `compute_layout_zeroes_an_explicit_sides_extra_before_the_cap`: an
    /// explicit TOP side's doomed tick extra must not share the top/bottom
    /// budget with the auto bottom side's title recentering, and the title
    /// is recentred on the plot that ships (the explicit top discards the
    /// tick extra, so it does not narrow the plot).
    #[test]
    fn compute_layout_zeroes_an_explicit_tops_extra_before_the_cap() {
        let (axes, mut theme, m) = vertical_padding_fixture(&"a".repeat(6));
        theme.padding.padding_top = Some(2.0);
        let viewport = Viewport {
            width: 400.0,
            height: 100.0,
        };
        let base = base_padding_insets(&theme);
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        let titles = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());
        let budget = (viewport.height - base.top - base.bottom - MIN_PANEL_DIM)
            .min(plot_room(&theme, &axes, &m, viewport).height_budget);
        assert!(
            ticks.top > 0.0 && titles.bottom > 0.0 && titles.bottom <= budget,
            "fixture must want a doomed top extra and a fitting bottom one: \
             ticks.top={}, titles.bottom={}, budget={budget}",
            ticks.top,
            titles.bottom,
        );
        assert!(
            ticks.top + titles.bottom > budget,
            "fixture must exceed the budget, or the cap never fires",
        );

        // `auto_padding_extra` is where an explicit side's extra is zeroed
        // before the cap, so the cap is judged there.
        let extra = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), x_overhang_without_auto(&theme, &axes, &m, viewport), &plot_room(&theme, &axes, &m, viewport))
        .total;
        assert!(
            (extra.bottom - titles.bottom).abs() < 1e-9,
            "the auto bottom extra ({}) must ship uncapped ({}), not throttled \
             to share budget with the explicit top's extra",
            extra.bottom,
            titles.bottom,
        );

        // The full layout ships it: the bottom extra keeps half the plot.
        let layout = auto_layout(&theme, viewport, &axes, &m);
        assert!(
            (layout.padded_region.y - 2.0).abs() < 1e-9,
            "explicit padding_top must survive verbatim: {}",
            layout.padded_region.y,
        );
        let bottom_inset = viewport.height - layout.padded_region.y - layout.padded_region.h;
        assert!(
            (bottom_inset - (base.bottom + titles.bottom)).abs() < 1e-9,
            "the uncapped bottom extra must ship: bottom inset {bottom_inset}",
        );
    }

    /// A tick extra an explicit side discards does not move the title
    /// solve: with the top explicit, the y tick pass's 22px top extra never
    /// ships, so the y title is recentred on the plot that does ship, and
    /// it ends inside the canvas. Solved as if that top extra narrowed the
    /// plot, the plot's centre would sit 11px lower and the bottom extra
    /// would be 22px larger, which ships and pushes the title 11px past the
    /// top edge.
    #[test]
    fn an_explicit_sides_discarded_tick_extra_does_not_move_the_title() {
        let (axes, mut theme, m) = vertical_padding_fixture(&"a".repeat(6));
        theme.padding.padding_top = Some(2.0);
        let viewport = Viewport {
            width: 400.0,
            height: 100.0,
        };
        let base = base_padding_insets(&theme);
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        let titles = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());
        let phantom = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset {
                top: base.top + ticks.top,
                ..base
            }, Inset::default());
        assert!(
            ticks.top == 22.0 && titles.bottom > 0.0 && phantom.bottom - titles.bottom == ticks.top,
            "fixture assumption: the discarded tick extra would change the title solve \
             (ticks {ticks:?}, titles {titles:?}, phantom {phantom:?})",
        );
        let layout = auto_layout(&theme, viewport, &axes, &m);
        let bottom_inset = viewport.height - layout.padded_region.y - layout.padded_region.h;
        assert!(
            (bottom_inset - (base.bottom + titles.bottom)).abs() < 1e-9,
            "the bottom extra is solved on the shipped plot: inset {bottom_inset}, want {}",
            base.bottom + titles.bottom,
        );
        let title = layout
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Left)
            .and_then(|a| a.title.as_ref())
            .expect("y title");
        let half = m.measure_width(&title.text, theme.typography.title_font_size) / 2.0;
        assert!(
            title.anchor_y - half >= -1e-9 && title.anchor_y + half <= viewport.height + 1e-9,
            "the y title ends inside the canvas (anchor {}, half {half})",
            title.anchor_y,
        );
    }

    /// As above, mirrored: an explicit BOTTOM side's doomed title extra must
    /// not share the budget with the auto top side's tick correction.
    #[test]
    fn compute_layout_zeroes_an_explicit_bottoms_extra_before_the_cap() {
        let (axes, mut theme, m) = vertical_padding_fixture(&"a".repeat(6));
        theme.padding.padding = 10.0;
        theme.padding.padding_bottom = Some(2.0);
        let viewport = Viewport {
            width: 400.0,
            height: 110.0,
        };
        let base = base_padding_insets(&theme);
        let ticks = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        let titles = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset {
                top: base.top + ticks.top,
                ..base
            }, Inset::default());
        let budget = (viewport.height - base.top - base.bottom - MIN_PANEL_DIM)
            .min(plot_room(&theme, &axes, &m, viewport).height_budget);
        assert!(
            ticks.top > 0.0 && ticks.top <= budget && titles.bottom > 0.0 && titles.top == 0.0,
            "fixture must want a fitting top extra and a doomed bottom one: \
             ticks.top={}, titles={titles:?}, budget={budget}",
            ticks.top,
        );
        assert!(
            ticks.top + titles.bottom > budget,
            "fixture must exceed the budget, or the cap never fires",
        );

        // `auto_padding_extra` is where an explicit side's extra is zeroed
        // before the cap, so the cap is judged there.
        let extra = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), x_overhang_without_auto(&theme, &axes, &m, viewport), &plot_room(&theme, &axes, &m, viewport))
        .total;
        assert!(
            (extra.top - ticks.top).abs() < 1e-9,
            "the auto top extra ({}) must ship uncapped ({}), not throttled to \
             share budget with the explicit bottom's extra",
            extra.top,
            ticks.top,
        );

        let layout = auto_layout(&theme, viewport, &axes, &m);
        let bottom_inset = viewport.height - layout.padded_region.y - layout.padded_region.h;
        assert!(
            (bottom_inset - 2.0).abs() < 1e-9,
            "explicit padding_bottom must survive verbatim: {bottom_inset}",
        );
        // As in the mirror test, the full layout ships it.
        assert!(
            (layout.padded_region.y - (base.top + ticks.top)).abs() < 1e-9,
            "the uncapped top extra must ship: top inset {}",
            layout.padded_region.y,
        );
    }

    /// The composed fixture behind findings 1 and 2: the LEFT-tick-overhang
    /// fixture (`continuous_x_axes_with_long_first_label_and_capped_y_band`)
    /// PLUS an x-axis title (66 characters, half-width 330 at 10px/char)
    /// wide enough to fit against the UNTOUCHED base but to clip past the
    /// RIGHT edge once the tick pass's own left expansion shifts the plot
    /// rightward — exactly the interference the elementwise-max composition
    /// missed (each pass alone reported "no correction needed" against a
    /// base the OTHER pass was about to change). The viewport (`700x300` in
    /// the test below) is wide enough that the x-axis's own 2-tick label
    /// band stays flat (no rotation cascade) both before and after both
    /// auto-padding corrections apply, isolating the outer-padding
    /// composition question from the unrelated tick-rotation cascade.
    fn axes_with_left_tick_overhang_and_a_title_the_shift_would_clip() -> AxesInput {
        let mut axes = continuous_x_axes_with_long_first_label_and_capped_y_band();
        axes.x.title.set(Some("a".repeat(66)));
        axes
    }

    /// Finding 1's own repro, proven through the real `compute_layout` path
    /// (the same fixture family as the reviewer's live Python repro, scaled
    /// to a viewport wide enough to keep the x-axis's own tick-rotation
    /// cascade out of the picture): the tick pass alone wants real left
    /// padding (the long negative first label, capped y `max_band`); the
    /// title alone (against the untouched base) would have reported "fits,
    /// no correction" — the exact shape that let elementwise-max ship a
    /// title-clipping bug for five cycles. Sequential composition must
    /// still keep the title fully on-canvas.
    #[test]
    fn padding_auto_keeps_title_on_canvas_when_a_tick_correction_also_fires() {
        let spec = minimal_chart_spec();
        let axes = axes_with_left_tick_overhang_and_a_title_the_shift_would_clip();
        let viewport = Viewport {
            width: 700.0,
            height: 300.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();
        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };

        let with_auto = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("auto-padded layout should succeed");
        let x_axis = with_auto
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let title = x_axis.title.as_ref().expect("x axis has a title");
        let half_w = m.measure_width(&title.text, theme.typography.title_font_size) / 2.0;
        assert!(
            title.anchor_x + half_w <= with_auto.viewport.w + 1e-9,
            "the title must not overhang the right edge once the tick pass has \
             also shifted the plot (anchor_x {} + half-width {} vs viewport width {})",
            title.anchor_x,
            half_w,
            with_auto.viewport.w,
        );
        assert!(
            title.anchor_x - half_w >= -1e-9,
            "the title must not overhang the left edge either (anchor_x {} - half-width {})",
            title.anchor_x,
            half_w,
        );

        // Also pin that the tick pass's own correction is genuinely still
        // in effect (this test would be vacuous if the fixture stopped
        // exercising the tick pass at all).
        let last_x_tick = x_axis.ticks.last().expect("x axis has ticks");
        assert!(
            with_auto.viewport.w - last_x_tick.position > 20.0,
            "fixture assumption broken: the tick pass should still be reserving \
             real left padding in this fixture",
        );
    }

    /// Finding 2's own repro, proven through the real `compute_layout` path:
    /// a viewport too narrow for the FULL auto correction a huge first-tick
    /// label + capped `max_band` would otherwise compute must still RENDER
    /// under `auto=True` (best-effort, degraded correction) rather than
    /// raise `PaddingExceedsViewport` — auto is never allowed to convert a
    /// chart that renders at `auto=False` into a refusal, let alone one
    /// citing a padding value the caller never set.
    #[test]
    fn padding_auto_never_converts_a_rendering_chart_into_a_padding_exceeds_viewport_refusal() {
        let spec = minimal_chart_spec();
        let mut axes = continuous_x_axes_with_long_first_label_and_capped_y_band();
        // Far longer than the sibling fixture's label, to force a much
        // larger desired tick-overhang correction than a narrow viewport
        // can possibly absorb.
        axes.x.tick_labels[0] = format!("-{}", "9".repeat(200));
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();

        compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect(
            "baseline (auto=false) must render — the fixture must not be pathological on its own",
        );

        let auto_theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: true,
                ..theme.padding.clone()
            },
            ..theme
        };
        let result = compute_layout(
            &spec,
            &auto_theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        );
        assert!(
            result.is_ok(),
            "padding.auto=true must never convert a rendering chart into \
             PaddingExceedsViewport: {result:?}",
        );
    }

    /// The right-side x-tick overhang correction subtracts whatever a
    /// right-side legend already reserves (via [`probe_auto_padding`])
    /// before deciding how much MORE padding the label's own overhang
    /// needs: on this fixture (a 200px chart), `padding.auto` reserves
    /// exactly the right edge the legend's own gutter already gave it —
    /// 0px of excess — while still keeping the label inside the viewport.
    #[test]
    fn padding_auto_right_legend_adds_zero_excess_padding() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];

        let auto_off = theme_inputs_with_padding_auto_off();
        let auto_on = default_theme_inputs();

        let without_auto = compute_layout(
            &spec,
            &auto_off,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("legend-right layout should succeed");
        let with_auto = compute_layout(
            &spec,
            &auto_on,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("legend-right layout should still succeed under padding.auto");

        let last_tick_gap = |r: &LayoutResult| {
            let x_axis = r
                .axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present");
            let t = x_axis.ticks.last().expect("x axis has ticks");
            let half_w = m.measure_width(&t.label, auto_off.typography.label_font_size) / 2.0;
            r.viewport.w - (t.position + half_w)
        };
        assert!(
            last_tick_gap(&without_auto) >= 0.0,
            "fixture assumption: the legend's own gutter must already keep \
             the last x-tick label inside the viewport with padding.auto off, \
             or this test is not isolating what it measures",
        );
        assert!(
            last_tick_gap(&with_auto) >= 0.0,
            "the #135 rescue: a long last x-tick label must stay inside the \
             viewport with a right-side legend present",
        );

        let plot_right_edge =
            |r: &LayoutResult| r.panels[0].plot_area.x + r.panels[0].plot_area.w;
        let without_auto_edge = plot_right_edge(&without_auto);
        let with_auto_edge = plot_right_edge(&with_auto);
        assert!(
            (without_auto_edge - 94.0).abs() < 1e-9,
            "without_auto_edge = {without_auto_edge}",
        );
        assert_eq!(
            with_auto_edge, without_auto_edge,
            "padding.auto must reserve exactly 0px beyond what the \
             legend's own gutter already provided for this label (with_auto \
             = {with_auto_edge}, without_auto = {without_auto_edge})",
        );
    }

    /// The secondary-y sibling of the legend test above: the same chrome
    /// credit, against a secondary-y axis's own band instead of a legend
    /// gutter. `padding.auto` adds exactly 0px beyond what the secondary-y
    /// band already reserved.
    #[test]
    fn padding_auto_secondary_y_adds_zero_excess_padding() {
        let spec = minimal_chart_spec();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.secondary_y = n_secondary_axes(1);
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        let auto_off = theme_inputs_with_padding_auto_off();
        let auto_on = default_theme_inputs();

        let without_auto = compute_layout(
            &spec,
            &auto_off,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("secondary-y layout should succeed");
        let with_auto = compute_layout(
            &spec,
            &auto_on,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("secondary-y layout should still succeed under padding.auto");

        let last_tick_gap = |r: &LayoutResult| {
            let x_axis = r
                .axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present");
            let t = x_axis.ticks.last().expect("x axis has ticks");
            let half_w = m.measure_width(&t.label, auto_off.typography.label_font_size) / 2.0;
            r.viewport.w - (t.position + half_w)
        };
        assert!(
            last_tick_gap(&without_auto) >= 0.0,
            "fixture assumption: the secondary-y band must already keep the \
             last x-tick label inside the viewport with padding.auto off, \
             or this test is not isolating what it measures",
        );
        assert!(
            last_tick_gap(&with_auto) >= 0.0,
            "the #135 rescue: a long last x-tick label must stay inside the \
             viewport with a secondary-y axis present",
        );

        let plot_right_edge =
            |r: &LayoutResult| r.panels[0].plot_area.x + r.panels[0].plot_area.w;
        let without_auto_edge = plot_right_edge(&without_auto);
        let with_auto_edge = plot_right_edge(&with_auto);
        assert!(
            (without_auto_edge - 124.4).abs() < 1e-9,
            "without_auto_edge = {without_auto_edge}",
        );
        assert_eq!(
            with_auto_edge, without_auto_edge,
            "padding.auto must reserve exactly 0px beyond what the \
             secondary-y band already provided for this label (with_auto = \
             {with_auto_edge}, without_auto = {without_auto_edge})",
        );
    }

    /// A legend on the BOTTOM (not the right) must leave the right-side
    /// x-tick correction exactly as it behaves with no legend at all —
    /// [`probe_auto_padding`] only ever credits a side the legend actually
    /// occupies, so a bottom legend contributes nothing to `chrome.right`.
    /// Pinned against the same 170 the no-legend sibling
    /// (`padding_auto_prevents_last_x_tick_label_from_overhanging_the_viewport`)
    /// computes for this exact fixture.
    #[test]
    fn padding_auto_right_side_unaffected_by_a_legend_on_another_side() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let mut theme = default_theme_inputs();
        theme.legend.legend_orient = LegendOrient::Bottom;

        let with_bottom_legend = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("bottom-legend layout should succeed");

        let plot_right_edge = with_bottom_legend.panels[0].plot_area.x
            + with_bottom_legend.panels[0].plot_area.w;
        assert!(
            (plot_right_edge - 170.0).abs() < 1e-9,
            "a legend on the bottom must not change the right-side tick \
             correction at all (plot_right_edge = {plot_right_edge}, \
             expected the same 170 the no-legend fixture produces)",
        );
    }

    /// When the last label's own overhang exceeds EVEN the legend's
    /// gutter, `padding.auto` pads exactly the remainder — not the full
    /// overhang (over-correcting past the gutter) and not zero
    /// (under-correcting, clipping past the gutter). Uses a realistic
    /// (600px) viewport so the legend strip's own half-viewport safety cap
    /// (`carve_legend_strip`) never engages — seeing the uncapped case is
    /// the point of this test; the capped case gets its own sibling,
    /// `padding_auto_settles_a_capped_legend_to_its_fixed_point`.
    #[test]
    fn padding_auto_pads_exactly_the_remainder_past_the_legend_gutter() {
        let spec = minimal_chart_spec();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels[1] = "9".repeat(23);
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let theme = default_theme_inputs();

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("legend-right layout with an oversized label should still succeed");

        let x_axis = with_auto
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let last_tick = x_axis.ticks.last().expect("x axis has ticks");
        let half_w = m.measure_width(&last_tick.label, theme.typography.label_font_size) / 2.0;
        assert!(
            (half_w - 115.0).abs() < 1e-9,
            "fixture assumption: this label's half-width must exceed the \
             legend gutter (half_w = {half_w})",
        );
        assert!(
            last_tick.position + half_w <= with_auto.viewport.w + 1e-9,
            "the label must stay inside the viewport even though its \
             overhang exceeds the legend's own gutter (position {} + \
             half-width {} vs viewport width {})",
            last_tick.position,
            half_w,
            with_auto.viewport.w,
        );
        let plot_right_edge =
            with_auto.panels[0].plot_area.x + with_auto.panels[0].plot_area.w;
        assert!(
            (plot_right_edge - 485.0).abs() < 1e-9,
            "plot_right_edge = {plot_right_edge} (expected 485: padding.auto \
             pads only the 9px remainder past the legend's own 90px gutter, \
             not the full 115px overhang)",
        );
        assert!(
            (last_tick.position + half_w - with_auto.viewport.w).abs() < 1e-9,
            "the remainder correction should land the label's edge exactly \
             on the viewport boundary, not with extra slack beyond it",
        );
    }

    /// A chart with no legend and no secondary-y axis is byte-identical to
    /// what `padding_auto_prevents_last_x_tick_label_from_overhanging_the_viewport`
    /// already pins — `probe_auto_padding` must contribute nothing when
    /// there is nothing on the right to credit.
    #[test]
    fn padding_auto_with_no_legend_or_secondary_y_is_unchanged() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed");
        let plot_right_edge =
            with_auto.panels[0].plot_area.x + with_auto.panels[0].plot_area.w;
        assert!(
            (plot_right_edge - 170.0).abs() < 1e-9,
            "plot_right_edge = {plot_right_edge}, expected 170 (unchanged \
             from the no-chrome case)",
        );
    }

    /// An explicit right padding still wins over `padding.auto` entirely —
    /// `probe_auto_padding`'s contribution never enters the picture on an
    /// explicit side, mirroring
    /// `compute_layout_zeroes_an_explicit_right_sides_extra_before_the_cap`
    /// but with a legend present, to pin that the chrome plumbing does not
    /// disturb the #139(3) explicit-side-zeroing rule.
    #[test]
    fn padding_auto_with_a_legend_still_honors_explicit_right_padding() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let mut theme = default_theme_inputs();
        theme.padding.padding_right = Some(50.0);

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed");

        // The padding stage's own output (before legend/axis-band
        // reservation) must be exactly the caller's explicit value — proof
        // that `padding.auto`'s chrome-aware correction never competed for
        // this side at all, exactly as `theme.padding.padding_right` being
        // `Some` already zeroes `combined.right` before `cap_auto_scale`.
        let padded_right_inset =
            viewport.width - with_auto.padded_region.w - with_auto.padded_region.x;
        assert!(
            (padded_right_inset - 50.0).abs() < 1e-9,
            "the explicit padding_right=50 must be honored exactly, \
             unaffected by the legend's chrome (padded_right_inset = \
             {padded_right_inset})",
        );
    }

    fn continuous_x_axes_with_long_first_label() -> AxesInput {
        let mut x = axis_input(
            AxisOrient::Bottom,
            None,
            vec!["zzzzzzzzz".into(), "0".into()],
            None,
        );
        x.tick_projection = Some(TickProjection {
            padding_frac: 0.0,
            major: vec![0.0, 1.0],
            minor: vec![],
        });
        let y = axis_input(
            AxisOrient::Left,
            None,
            vec!["0".into(), "5".into(), "10".into()],
            None,
        );
        AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// A left-side legend's own gutter already covers a long first x-tick
    /// label's overhang on this fixture (uncapped): the left plot edge is
    /// identical with `padding.auto` on and off, and the label stays
    /// inside the viewport either way.
    #[test]
    fn padding_auto_left_legend_adds_zero_excess_padding() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_first_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let mut auto_off = theme_inputs_with_padding_auto_off();
        auto_off.legend.legend_orient = LegendOrient::Left;
        let mut auto_on = default_theme_inputs();
        auto_on.legend.legend_orient = LegendOrient::Left;

        let without_auto = compute_layout(
            &spec,
            &auto_off,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("left-legend layout should succeed");
        let with_auto = compute_layout(
            &spec,
            &auto_on,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("left-legend layout should still succeed under padding.auto");

        let first_tick_gap = |r: &LayoutResult| {
            let x_axis = r
                .axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present");
            let t = x_axis.ticks.first().expect("x axis has ticks");
            let half_w = m.measure_width(&t.label, auto_off.typography.label_font_size) / 2.0;
            t.position - half_w
        };
        assert!(
            first_tick_gap(&without_auto) >= 0.0,
            "fixture assumption: the legend's own gutter must already keep \
             the first x-tick label inside the viewport with padding.auto off",
        );
        assert!(
            first_tick_gap(&with_auto) >= 0.0,
            "the long first x-tick label must stay inside the viewport \
             with a left-side legend present",
        );
        let plot_left_edge = |r: &LayoutResult| r.panels[0].plot_area.x;
        let without_auto_edge = plot_left_edge(&without_auto);
        let with_auto_edge = plot_left_edge(&with_auto);
        assert!(
            (without_auto_edge - 132.0).abs() < 1e-9,
            "without_auto_edge = {without_auto_edge}",
        );
        assert_eq!(
            with_auto_edge, without_auto_edge,
            "padding.auto must reserve exactly 0px beyond what the legend's \
             own gutter already provided on the left (with_auto = \
             {with_auto_edge}, without_auto = {without_auto_edge})",
        );
    }

    /// The top-side sibling of the legend tests above: a continuous y axis
    /// whose topmost tick label overhangs, with a top-side legend already
    /// covering the overhang (uncapped). The top plot edge is identical
    /// with `padding.auto` on and off.
    #[test]
    fn padding_auto_top_legend_adds_zero_excess_padding() {
        let (axes, mut theme_on, m) = vertical_padding_fixture(&"a".repeat(6));
        theme_on.legend.legend_orient = LegendOrient::Top;
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let spec = minimal_chart_spec();
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let mut theme_off = theme_inputs_with_padding_auto_off();
        theme_off.padding.padding = theme_on.padding.padding;
        theme_off.typography.label_font_size = theme_on.typography.label_font_size;
        theme_off.legend.legend_orient = LegendOrient::Top;

        let without_auto = compute_layout(
            &spec,
            &theme_off,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("top-legend layout should succeed");
        let with_auto = compute_layout(
            &spec,
            &theme_on,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("top-legend layout should still succeed under padding.auto");

        let without_auto_top = without_auto.panels[0].plot_area.y;
        let with_auto_top = with_auto.panels[0].plot_area.y;
        assert!(
            (without_auto_top - 120.0).abs() < 1e-9,
            "without_auto_top = {without_auto_top}",
        );
        assert_eq!(
            with_auto_top, without_auto_top,
            "padding.auto must reserve exactly 0px beyond what the legend's \
             own gutter already provided on top (with_auto = {with_auto_top}, \
             without_auto = {without_auto_top})",
        );
    }

    /// A bottom-side legend changes how much of the viewport height is
    /// left for the plot, which shifts where a y-axis title's recentering
    /// solve places it. The title (28 chars, half-width 140 against a
    /// 300px/half-150 viewport) sits close enough to the feasibility
    /// boundary that the shift is real, not a no-op: dropping
    /// `chrome.bottom` from `plot_h` moves the recentered anchor from
    /// 156 to 128 (confirmed by mutation), and the measured anchor also
    /// differs from the same fixture with no legend at all. The title
    /// stays within the viewport either way.
    #[test]
    fn padding_auto_bottom_legend_shifts_the_recentered_y_title() {
        let (axes, mut theme, m) = vertical_padding_fixture(&"a".repeat(28));
        theme.legend.legend_orient = LegendOrient::Bottom;
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let spec = minimal_chart_spec();
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];

        let with_legend = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("bottom-legend layout should succeed");
        let no_legend = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("no-legend layout should succeed");

        let y_title_anchor_y = |r: &LayoutResult| {
            r.axes
                .iter()
                .find(|a| a.orient == AxisOrient::Left)
                .and_then(|a| a.title.as_ref())
                .expect("y axis has a title")
                .anchor_y
        };
        let with_legend_anchor = y_title_anchor_y(&with_legend);
        let no_legend_anchor = y_title_anchor_y(&no_legend);
        assert!(
            (with_legend_anchor - 156.0).abs() < 1e-9,
            "with_legend_anchor = {with_legend_anchor}",
        );
        assert_ne!(
            with_legend_anchor, no_legend_anchor,
            "a bottom legend must change the recentered anchor \
             (chrome.bottom must reach plot_h/plot_y), not leave it at the \
             no-legend value {no_legend_anchor}",
        );
        let half_h =
            m.measure_width("a".repeat(28).as_str(), theme.typography.title_font_size) / 2.0;
        assert!(
            with_legend_anchor - half_h >= -1e-9
                && with_legend_anchor + half_h <= viewport.height + 1e-9,
            "the recentered y title must stay inside the viewport \
             (anchor {with_legend_anchor}, half {half_h}, height {})",
            viewport.height,
        );
    }

    /// An x-axis title's recentering solve also folds in `chrome`: with a
    /// right-side legend, and separately with a secondary-y axis, the
    /// title's recentered anchor stays within the viewport on a fixture
    /// whose title would otherwise sit flush against the right edge. Each
    /// chart is wide enough that the rescue keeps half the plot: the legend
    /// takes more of a 200px chart than the secondary-y band does, so its
    /// chart is 250px.
    #[test]
    fn padding_auto_title_recentering_with_a_right_legend_or_secondary_y() {
        let spec = minimal_chart_spec();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let legend_viewport = Viewport {
            width: 250.0,
            ..viewport
        };
        let half_w = m.measure_width("A Wide X Axis Title", 13.0) / 2.0;

        let axes_legend = axes_with_wide_x_title();
        let mut theme_legend = default_theme_inputs();
        theme_legend.legend.legend_orient = LegendOrient::Right;
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let with_legend = compute_layout(
            &spec,
            &theme_legend,
            legend_viewport,
            &axes_legend,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("right-legend layout should succeed");
        let legend_anchor_x = with_legend
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .and_then(|a| a.title.as_ref())
            .expect("x axis has a title")
            .anchor_x;
        assert!(
            (legend_anchor_x - 95.0).abs() < 1e-6,
            "legend_anchor_x = {legend_anchor_x}",
        );
        assert!(
            legend_anchor_x - half_w >= -1e-6
                && legend_anchor_x + half_w <= legend_viewport.width + 1e-6,
            "the recentered x title must stay inside the viewport with a \
             right legend (anchor {legend_anchor_x}, half {half_w})",
        );

        let mut axes_secondary = axes_with_wide_x_title();
        axes_secondary.secondary_y = n_secondary_axes(1);
        let with_secondary = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes_secondary,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("secondary-y layout should succeed");
        let secondary_anchor_x = with_secondary
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .and_then(|a| a.title.as_ref())
            .expect("x axis has a title")
            .anchor_x;
        assert!(
            (secondary_anchor_x - 95.0).abs() < 1e-6,
            "secondary_anchor_x = {secondary_anchor_x}",
        );
        assert!(
            secondary_anchor_x - half_w >= -1e-6
                && secondary_anchor_x + half_w <= viewport.width + 1e-6,
            "the recentered x title must stay inside the viewport with a \
             secondary-y axis (anchor {secondary_anchor_x}, half {half_w})",
        );
    }

    /// The capped-legend fixture: a legend whose natural strip is wider
    /// than half of the final, auto-narrowed plot rect, so
    /// `carve_legend_strip`'s own `inner.w * 0.5` cap applies. A single
    /// probe at an oversized rect would credit the legend's full natural
    /// size (90px including its gap) and clip the label by up to 10px;
    /// the fixed point converges on the self-consistent credit (85px) the
    /// capped legend actually reserves, landing the label's edge exactly
    /// on the viewport boundary with no clip and no excess.
    #[test]
    fn padding_auto_settles_a_capped_legend_to_its_fixed_point() {
        let spec = minimal_chart_spec();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels[1] = "9".repeat(23);
        // Flat on a plot narrower than the label (`label_overlap=True`), so
        // its drawn overhang is its half-width; the cascade would rotate it.
        axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        let viewport = Viewport {
            width: 200.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        // The 230px label is wider than the 200px canvas, so its left half
        // also crosses the left edge, and the left margin does move its
        // tick (through the capped legend). No pair of margins contains
        // both halves; an explicit left side leaves only the right-side
        // credit fixed point under test.
        let mut theme = default_theme_inputs();
        theme.padding.padding_left = Some(16.0);

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("capped-legend layout should succeed");

        let x_axis = with_auto
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let last_tick = x_axis.ticks.last().expect("x axis has ticks");
        let half_w = m.measure_width(&last_tick.label, theme.typography.label_font_size) / 2.0;
        assert!(
            (half_w - 115.0).abs() < 1e-9,
            "fixture assumption: the label's overhang must exceed the \
             legend's natural gutter (half_w = {half_w})",
        );
        // The capped regime's loop gain is 1/2 (one side of one dimension
        // depends on the rect at a time — see `settle_auto_padding`'s own
        // doc), so a converged residual can be amplified by at most
        // 1 / (1 - 1/2) = 2x `AUTO_PADDING_EPS` relative to the true fixed
        // point, not just `AUTO_PADDING_EPS` itself. `FIXED_POINT_TOLERANCE`
        // is a 5x margin on that 2x bound, not an arbitrary loosening.
        const FIXED_POINT_TOLERANCE: f64 = 10.0 * AUTO_PADDING_EPS;
        assert!(
            last_tick.position + half_w <= with_auto.viewport.w + FIXED_POINT_TOLERANCE,
            "the label must land inside (or exactly on) the viewport edge, \
             not clip past it (position {} + half-width {} vs viewport \
             width {})",
            last_tick.position,
            half_w,
            with_auto.viewport.w,
        );
        let plot_right_edge = with_auto.panels[0].plot_area.x + with_auto.panels[0].plot_area.w;
        assert!(
            (plot_right_edge - 85.0).abs() < FIXED_POINT_TOLERANCE,
            "plot_right_edge = {plot_right_edge} (expected the fixed point \
             at 85: the legend's own cap reserves less than its natural \
             size once padding.auto has narrowed the rect, and the credit \
             must settle there rather than at the natural, uncapped size)",
        );
    }

    /// The height-cap sibling of the capped-legend test above: a Top
    /// legend forced to a vertical direction (so its natural HEIGHT scales
    /// with entry count instead of its width) ends up far taller than the
    /// padded rect, so `carve_legend_strip`'s `inner.h * 0.5` cap applies
    /// on this axis. The y-tick pass's own top correction (`extra_top`)
    /// reads `chrome.top`, so the fixed point must settle on the capped
    /// height the legend actually reserves, not its natural, uncapped one.
    #[test]
    fn padding_auto_settles_a_capped_top_legend_to_its_fixed_point() {
        let (axes, mut theme, m) = vertical_padding_fixture(&"a".repeat(6));
        theme.legend.legend_orient = LegendOrient::Top;
        theme.legend.legend_direction = Some(LegendDirection::Vertical);
        let entries: Vec<LegendEntry> = (0..5)
            .map(|i| LegendEntry::new(format!("cat{i}"), SymbolKind::Circle))
            .collect();
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 300.0,
            height: 200.0,
        };
        const FIXED_POINT_TOLERANCE: f64 = 10.0 * AUTO_PADDING_EPS;

        let with_auto = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("capped top-legend layout should succeed");

        let plot_top_edge = with_auto.panels[0].plot_area.y;
        assert!(
            (plot_top_edge - 150.0).abs() < FIXED_POINT_TOLERANCE,
            "plot_top_edge = {plot_top_edge} (expected the fixed point at \
             150: the Top legend's own height-cap reserves less than its \
             natural, 5-entry vertical-stack size once padding.auto has \
             narrowed the rect, and the credit must settle there rather \
             than at the natural, uncapped size)",
        );
    }

    /// A Left colorbar whose natural width exceeds the ENTIRE padded rect
    /// (not just half of it): `layout_colorbar`'s Left-orient carve always
    /// advances `inner.x` by the colorbar's full declared width before
    /// clamping `inner.w` to 0 (`legend.rs`'s own `.max(0.0)` floor on `w`,
    /// with no matching clamp on `x`), so the post-carve rect's right edge
    /// can land to the RIGHT of the padded rect's own right edge. Reading
    /// that as a plain edge difference makes the RIGHT side's credit
    /// negative unless clamped. Calls `probe_auto_padding` directly (the
    /// precise site of the clamp) so the assertion is not diluted by
    /// `Rect::shrink`'s own downstream `.max(0.0)` floors, which would
    /// otherwise mask a negative credit before it reached a visible rect.
    ///
    /// The same condition that produces the negative delta (the colorbar's
    /// declared width exceeding the padded rect) also collapses the real
    /// plot to zero width downstream, at every viewport tried: `inner.w`
    /// is `(plot_inner.w - size.width - gutter).max(0.0)`, which is `0`
    /// whenever `size.width` alone already exceeds `plot_inner.w`, with no
    /// viewport size escaping it (a larger viewport only grows
    /// `plot_inner.w`, and once it is large enough that the colorbar no
    /// longer exceeds it, the credit is no longer negative either). This
    /// test therefore asserts the collapsed result exactly rather than a
    /// surviving label.
    #[test]
    fn padding_auto_clamps_a_left_colorbar_wider_than_the_padded_rect() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_first_label();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let colorbar = legend::ColorbarInput {
            stops: vec![(0.0, "#000000".into()), (1.0, "#ffffff".into())],
            tick_labels: vec!["0".into(), "zzzzzzzzzzzz".into()],
            domain: Some((0.0, 1.0)),
        };
        let mut theme = default_theme_inputs();
        theme.legend.legend_orient = LegendOrient::Left;

        for (width, height) in [(60.0, 80.0), (80.0, 100.0), (100.0, 150.0), (150.0, 200.0)] {
            let viewport = Viewport { width, height };
            let inset = base_padding_insets(&theme);
            let legend_overrides = legend::LegendOverrides::default();
            let inputs = LayoutInputs {
                spec: &spec,
                theme: &theme,
                viewport,
                axes: &axes,
                facet_groups: &[],
                legend_entries: &[],
                legend_title: None,
                colorbar: Some(&colorbar),
                metrics: &m,
                legend_overrides: &legend_overrides,
                aux_legend_inputs: &[],
                seam: CompositeLayoutSeam::default(),
            };
            let credit =
                probe_auto_padding(inset, &inputs, &AutoPlotRoom::of_auto_off(&inputs, inset)).chrome;
            assert!(
                credit.top >= 0.0 && credit.right >= 0.0 && credit.bottom >= 0.0
                    && credit.left >= 0.0,
                "every side of the measured chrome must be >= 0 at \
                 {width}x{height} (credit = {credit:?}); a negative \
                 credit is never a real reservation",
            );

            let layout = compute_layout(
                &spec,
                &theme,
                viewport,
                &axes,
                &[],
                &[],
                None,
                Some(&colorbar),
                &m,
                &legend::LegendOverrides::default(),
                &[],
                CompositeLayoutSeam::default(),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "an oversized Left colorbar must still terminate and \
                     render at {width}x{height}, not panic, hang, or \
                     refuse: {e:?}"
                )
            });
            let plot = layout.panels[0].plot_area;
            assert_eq!(
                plot,
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 0.0,
                },
                "fixture assumption: a colorbar wider than the entire \
                 padded rect must collapse the plot to zero at \
                 {width}x{height} (plot = {plot:?}); a surviving plot \
                 would mean the colorbar no longer exceeds the rect, and \
                 the negative-delta case above is no longer exercised",
            );
            assert!(
                layout.axes.is_empty(),
                "a collapsed plot must lay out no axes at {width}x{height} \
                 (axes = {:?})",
                layout.axes,
            );
        }
    }

    /// A color legend oriented Top or Bottom carves its aux (size/shape)
    /// overflow on the RIGHT side of the plot — a different side than the
    /// legend's own orient — so crediting chrome only on the legend's own
    /// side would miss it. With a Top legend plus a `Size` aux block wide
    /// enough to overflow the color block, the right edge is identical
    /// with `padding.auto` on and off: the aux strip is credited too.
    #[test]
    fn padding_auto_credits_an_aux_strip_carved_on_a_different_side_than_the_legend() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let aux = vec![legend::AuxLegendInput::Size {
            title: None,
            entries: vec![legend::SizeLegendEntry {
                label: "big".into(),
                radius: 50.0,
                color_hex: None,
            }],
            symbol: SymbolKind::Circle,
        }];
        let mut theme_off = theme_inputs_with_padding_auto_off();
        theme_off.legend.legend_orient = LegendOrient::Top;
        let mut theme_on = default_theme_inputs();
        theme_on.legend.legend_orient = LegendOrient::Top;

        let without_auto = compute_layout(
            &spec,
            &theme_off,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &aux,
            CompositeLayoutSeam::default(),
        )
        .expect("top-legend-plus-aux layout should succeed");
        let with_auto = compute_layout(
            &spec,
            &theme_on,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &aux,
            CompositeLayoutSeam::default(),
        )
        .expect("top-legend-plus-aux layout should still succeed under padding.auto");

        let plot_right_edge =
            |r: &LayoutResult| r.panels[0].plot_area.x + r.panels[0].plot_area.w;
        let without_auto_edge = plot_right_edge(&without_auto);
        let with_auto_edge = plot_right_edge(&with_auto);
        assert!(
            (without_auto_edge - 426.0).abs() < 1e-9,
            "without_auto_edge = {without_auto_edge}",
        );
        assert_eq!(
            with_auto_edge, without_auto_edge,
            "the aux strip's right-side reservation must be credited even \
             though the legend itself is oriented Top (with_auto = \
             {with_auto_edge}, without_auto = {without_auto_edge})",
        );
    }

    /// `seam.suppress_chart_title` (the overlay non-primary-leaf path)
    /// means the chart title band is never actually reserved, so the
    /// chrome measurement for that leaf must credit nothing for it. Uses
    /// a continuous y axis with a small (2px) padding floor
    /// (`vertical_padding_fixture`) so the TICK pass's own top correction
    /// has slack to absorb whatever chrome.top credits: a suppressed
    /// title must match a no-title chart exactly (`plot_area` identical).
    /// If `probe_auto_padding` instead always measured as unsuppressed —
    /// crediting a title band the real render never draws — the
    /// suppressed case would reserve LESS of the real top correction (the
    /// wrongly-credited title eats into `extra_top`'s own budget) while
    /// still not drawing the title, moving `plot_area` away from the
    /// no-title baseline. Confirmed by mutation: forcing
    /// `probe_auto_padding` to always pass `suppress_chart_title: false`
    /// changes `suppressed`'s `plot_area.y` from 78.0 to 56.4 against the
    /// unchanged 78.0 no-title baseline.
    #[test]
    fn padding_auto_suppressed_chart_title_credits_nothing() {
        let (axes, theme, m) = vertical_padding_fixture("y");
        let spec = minimal_chart_spec();
        let mut titled_spec = spec.clone();
        titled_spec.title = Some(crate::spec::title::TitleSpec {
            text: "A Title".into(),
            ..Default::default()
        });
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let suppressed_seam = CompositeLayoutSeam {
            suppress_chart_title: true,
            ..CompositeLayoutSeam::default()
        };

        let suppressed = compute_layout(
            &titled_spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            suppressed_seam,
        )
        .expect("suppressed-title layout should succeed");
        let no_title = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("no-title layout should succeed");
        let titled_not_suppressed = compute_layout(
            &titled_spec,
            &theme,
            viewport,
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("titled-not-suppressed layout should succeed");

        assert_eq!(
            suppressed.panels[0].plot_area, no_title.panels[0].plot_area,
            "a suppressed chart title must reserve exactly what a chart \
             with no title at all reserves",
        );
        // Also equal to the actually-drawn title, not merely to the
        // no-title case: with a real (2px) padding floor, the tick pass's
        // own correction has slack to absorb the title band either way,
        // so a correctly-credited chrome makes all three paths agree —
        // the same 0-excess invariant the legend tests above pin. The
        // discriminating assertion is the one above: a wrongly-credited
        // suppressed title (probe_auto_padding ignoring `seam` entirely)
        // breaks exactly that equality, not this one.
        assert_eq!(
            suppressed.panels[0].plot_area, titled_not_suppressed.panels[0].plot_area,
            "a suppressed title and an actually-drawn title must reserve \
             the same amount when chrome is credited correctly",
        );
    }

    /// An aux legend block far wider than the rect it is carved from
    /// overflows `layout_aux_legends`' own `(new_inner.w - extra -
    /// legend_gutter.min(0.0)).max(0.0)` clamp — the same derivation the
    /// colorbar test above states: once the declared block exceeds what
    /// the carve has left, the result is `0` regardless of viewport size,
    /// so a surviving plot cannot be constructed for this premise either.
    /// This test asserts that collapse exactly, plus that `compute_layout`
    /// still terminates (`Ok`) within `settle_auto_padding`'s bounded pass
    /// count rather than panicking or hanging — the overflow itself is
    /// what is under test here, not a surviving label.
    #[test]
    fn padding_auto_terminates_when_an_aux_block_is_wider_than_the_inner_rect() {
        let spec = minimal_chart_spec();
        let axes = continuous_x_axes_with_long_last_label();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let entries = vec![LegendEntry::new("alpha", SymbolKind::Circle)];
        let aux = vec![legend::AuxLegendInput::Size {
            title: None,
            entries: vec![legend::SizeLegendEntry {
                label: "9".repeat(60),
                radius: 5.0,
                color_hex: None,
            }],
            symbol: SymbolKind::Circle,
        }];

        let theme = default_theme_inputs();
        let result = compute_layout(
            &spec,
            &theme,
            viewport,
            &axes,
            &[],
            &entries,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &aux,
            CompositeLayoutSeam::default(),
        );
        let layout = result.expect(
            "an oversized aux block must still terminate and render, not panic or hang",
        );
        let plot = layout.panels.first().expect("one panel").plot_area;
        assert_eq!(
            plot,
            Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            "fixture assumption: an aux block wider than the rect it is \
             carved from must collapse the plot to zero (plot = {plot:?}); \
             a surviving plot would mean the block no longer overflows, \
             and the overflow case under test is no longer exercised",
        );
        assert!(
            layout.axes.is_empty(),
            "a collapsed plot must lay out no axes (axes = {:?})",
            layout.axes,
        );
    }

    /// Finding 3: `AxisInput.title` is populated independently of the
    /// chart-level show flag (`.axis(x=False)`), so without a matching gate
    /// `auto_padding_for_axis_titles` would expand margins to recenter a
    /// title `layout_panel_axes` never draws — mirrors
    /// `auto_padding_for_edge_ticks`'s own `show_x`/`show_y` gate.
    #[test]
    fn auto_padding_for_axis_titles_is_inert_when_the_axis_is_hidden() {
        let theme = default_theme_inputs();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let base = base_padding_insets(&theme);

        let mut axes_x_hidden = axes_with_wide_x_title();
        axes_x_hidden.show_x = false;
        let extra_x = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes_x_hidden, &m, viewport), base, Inset::default());
        assert_eq!(
            extra_x,
            Inset::default(),
            "a hidden x-axis's title must not be recentered"
        );
        let (x_infeasible, _) = axis_title_infeasibility(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes_x_hidden, &m, viewport), base, Inset::default(), &plot_room(&theme, &axes_x_hidden, &m, viewport));
        assert!(
            !x_infeasible,
            "a hidden axis's title is not even considered"
        );

        let mut axes_y_hidden = axes_with_wide_x_title();
        axes_y_hidden
            .y
            .title
            .set(Some("A Wide Y Axis Title".into()));
        axes_y_hidden.x.title.set(None);
        axes_y_hidden.show_y = false;
        let extra_y = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes_y_hidden, &m, viewport), base, Inset::default());
        assert_eq!(
            extra_y,
            Inset::default(),
            "a hidden y-axis's title must not be recentered"
        );
        let (_, y_infeasible) = axis_title_infeasibility(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes_y_hidden, &m, viewport), base, Inset::default(), &plot_room(&theme, &axes_y_hidden, &m, viewport));
        assert!(
            !y_infeasible,
            "a hidden axis's title is not even considered"
        );
    }

    #[test]
    fn compute_layout_serde_round_trip() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 600.0,
                height: 400.0,
            },
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        // `plot_region` (GH #89A) and `padded_region` (#182's extent
        // alignment) are deliberately off the wire: serializing either would
        // add a key to `fm.compute_layout`'s returned dict for a purely
        // in-process reader. They are the ONLY fields a round trip may drop,
        // and this asserts exactly that — every other field survives
        // byte-for-byte. (Previously this named `plot_region` as the one such
        // field; `padded_region` joins it under the same contract, so the
        // claim is now "these two and no others".)
        assert!(
            result.plot_region.w > 0.0,
            "the computed layout has a real plot region"
        );
        assert!(
            result.padded_region.w > result.plot_region.w,
            "the padded box is the plot region plus this chart's axis chrome"
        );
        assert_eq!(
            parsed.plot_region,
            Rect::ZERO,
            "plot_region does not round-trip by design"
        );
        assert_eq!(
            parsed.padded_region,
            Rect::ZERO,
            "padded_region does not round-trip by design"
        );
        assert_eq!(
            parsed,
            LayoutResult {
                plot_region: Rect::ZERO,
                padded_region: Rect::ZERO,
                ..result
            }
        );
    }

    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::layout::panel::FacetKey;

    fn faceted_spec(ncols: u32) -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "species".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            mode: FacetMode::Wrap { ncols },
            spacing: None,
            resolve: FacetResolve::default(),
        });
        s
    }

    fn three_groups() -> Vec<FacetGroup> {
        vec![
            FacetGroup {
                key: FacetKey {
                    field: "species".into(),
                    value: "setosa".into(),
                },
                n_rows: 50,
                row_key: None,
            },
            FacetGroup {
                key: FacetKey {
                    field: "species".into(),
                    value: "versicolor".into(),
                },
                n_rows: 50,
                row_key: None,
            },
            FacetGroup {
                key: FacetKey {
                    field: "species".into(),
                    value: "virginica".into(),
                },
                n_rows: 50,
                row_key: None,
            },
        ]
    }

    #[test]
    fn compute_layout_faceted_three_panels_one_legend() {
        let spec = faceted_spec(3);
        let groups = three_groups();
        let legend = vec![
            LegendEntry::new("setosa", SymbolKind::Circle),
            LegendEntry::new("versicolor", SymbolKind::Circle),
            LegendEntry::new("virginica", SymbolKind::Circle),
        ];
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 400.0,
            },
            &axes,
            &groups,
            &legend,
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 3);
        assert_eq!(result.axes.len(), 6);
        assert!(result.legend.is_some());
        assert!(
            result.warnings.is_empty(),
            "unexpected warnings: {:?}",
            result.warnings
        );

        assert_eq!(result.panels[0].facet_key.as_ref().unwrap().value, "setosa");
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
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            mode: FacetMode::Grid { nrows: 1, ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        });
        let groups = three_groups();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 400.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 2);
        let dropped = result
            .warnings
            .iter()
            .any(|w| matches!(w, LayoutWarning::PanelsDropped { count: 1, .. }));
        assert!(
            dropped,
            "expected PanelsDropped(1); got {:?}",
            result.warnings
        );
    }

    #[test]
    fn compute_layout_faceted_emits_strip_titles() {
        let spec = faceted_spec(3);
        let groups = three_groups();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 400.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 3);
        for (i, panel) in result.panels.iter().enumerate() {
            let strip = panel
                .strip_title
                .as_ref()
                .unwrap_or_else(|| panic!("panel {i} missing strip_title"));
            assert!(!strip.text.is_empty());
            // F-L01-04: no `title` set on the FacetSpec -> bare value, no
            // "{title}: " prefix. This is the byte-identical-when-absent claim.
            let expected = &panel.facet_key.as_ref().unwrap().value;
            assert_eq!(
                &strip.text, expected,
                "absent title must not prefix strip text"
            );
            // Themes-T4: strip_text_size default flipped 13.0 → 12.0.
            assert_eq!(strip.font_size, 12.0);
            assert!(strip.anchor.0 >= panel.plot_area.x);
            assert!(strip.anchor.0 <= panel.plot_area.x + panel.plot_area.w);
        }
    }

    /// F-L01-04: `FacetSpec.title` set -> every column strip renders
    /// `"{title}: {value}"` instead of the bare value.
    #[test]
    fn compute_layout_facet_title_set_prefixes_column_strip_text() {
        let mut spec = faceted_spec(3);
        spec.facet.as_mut().unwrap().title = Some("Species".into());
        let groups = three_groups();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 400.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 3);
        for panel in &result.panels {
            let strip = panel.strip_title.as_ref().unwrap();
            let value = &panel.facet_key.as_ref().unwrap().value;
            assert_eq!(strip.text, format!("Species: {value}"));
        }
    }

    /// F-L01-04 wrap-row trap: `row_title` set on a Wrap-mode spec (which is
    /// exactly what `Chart.facet(row=X)` alone lowers to — `row` left unset,
    /// `field: X`) must be a pure no-op. There is no row-header strip in
    /// wrap mode (no group carries a `row_key`), so `row_title` cannot reach
    /// any strip; a row-only facet's title MUST ride `title` instead (pinned
    /// above by `compute_layout_facet_title_set_prefixes_column_strip_text`).
    /// This is the negative half of that contract: setting the wrong field
    /// changes nothing observable, rather than silently misrouting.
    #[test]
    fn compute_layout_facet_row_title_is_inert_in_wrap_mode() {
        let mut spec = faceted_spec(3);
        spec.facet.as_mut().unwrap().row_title = Some("Region".into());
        let groups = three_groups(); // wrap-mode groups: row_key is None on all.
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 400.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 3);
        for panel in &result.panels {
            // No row-header strip exists at all in wrap mode.
            assert!(
                panel.row_strip_title.is_none(),
                "wrap mode must never emit a row-header strip"
            );
            // The column strip is untouched by row_title — still bare,
            // because `title` (the field wrap mode actually reads) is unset.
            let strip = panel.strip_title.as_ref().unwrap();
            let value = &panel.facet_key.as_ref().unwrap().value;
            assert_eq!(
                &strip.text, value,
                "row_title must not leak into the column/wrap strip"
            );
        }
    }

    #[test]
    fn compute_layout_unfaceted_omits_strip_titles() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 600.0,
                height: 400.0,
            },
            &axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
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
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            mode: FacetMode::Grid { nrows: 2, ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        });

        let groups: Vec<FacetGroup> = ["a", "b", "c", "d"]
            .iter()
            .map(|v| FacetGroup {
                key: FacetKey {
                    field: "group".into(),
                    value: v.to_string(),
                },
                n_rows: 10,
                row_key: None,
            })
            .collect();

        let axes = AxesInput {
            x: axis_input(
                AxisOrient::Bottom,
                Some("x_title".into()),
                vec!["0".into(), "1".into()],
                None,
            ),
            y: axis_input(
                AxisOrient::Left,
                Some("y_title".into()),
                vec!["0".into(), "5".into()],
                None,
            ),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 4, "expected 4 panels in a 2x2 grid");
        assert_eq!(
            result.axes.len(),
            8,
            "expected 8 axis layouts (2 per panel)"
        );

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
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        // Short flat labels: "A"=10px fits in slot_w=600/4=150.
        let short_axes = AxesInput {
            x: axis_input(
                AxisOrient::Bottom,
                None,
                vec!["A".into(), "B".into(), "C".into(), "D".into()],
                None,
            ),
            y: axis_input(
                AxisOrient::Left,
                None,
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };

        // Long labels with -45° override: "ABCDEFGHIJ"=100px. Angle override forces
        // margin = 100*sin(45°) + line_h*cos(45°) ≈ 70.7 + 9.3 = 80.
        let long_axes = AxesInput {
            x: axis_input(
                AxisOrient::Bottom,
                None,
                vec![
                    "ABCDEFGHIJ".into(),
                    "KLMNOPQRST".into(),
                    "UVWXYZABCD".into(),
                    "EFGHIJKLMN".into(),
                ],
                Some(-45.0),
            ),
            y: axis_input(
                AxisOrient::Left,
                None,
                vec!["0".into(), "5".into(), "10".into()],
                None,
            ),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };

        let short_result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &short_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let long_result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &long_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let short_bottom = short_result.panels[0].plot_area.y + short_result.panels[0].plot_area.h;
        let long_bottom = long_result.panels[0].plot_area.y + long_result.panels[0].plot_area.h;

        assert!(
            long_bottom < short_bottom,
            "rotated labels (-45°) must consume more bottom margin than flat labels; \
             long_bottom={long_bottom:.1} should be less than short_bottom={short_bottom:.1}"
        );
    }

    /// R2, TRANSPOSE of the x test above: rotating y tick labels SHRINKS the
    /// reserved left margin (the opposite direction from x), because a y
    /// label's own width stops projecting fully onto the horizontal gutter as
    /// it rotates toward vertical. Same labels, flat vs. `-45°` override; a
    /// smaller left margin means the plot area starts closer to the viewport's
    /// left edge (`plot_area.x` shrinks).
    #[test]
    fn compute_layout_rotated_y_labels_reserve_smaller_left_margin_than_flat() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let y_labels = vec![
            "ABCDEFGHIJ".into(),
            "KLMNOPQRST".into(),
            "UVWXYZABCD".into(),
        ];

        let flat_axes = AxesInput {
            x: axis_input(AxisOrient::Bottom, None, vec!["A".into(), "B".into()], None),
            y: axis_input(AxisOrient::Left, None, y_labels.clone(), None),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };
        let rotated_axes = AxesInput {
            x: axis_input(AxisOrient::Bottom, None, vec!["A".into(), "B".into()], None),
            y: axis_input(AxisOrient::Left, None, y_labels, Some(-45.0)),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };

        let flat_result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &flat_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let rotated_result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &rotated_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let flat_x = flat_result.panels[0].plot_area.x;
        let rotated_x = rotated_result.panels[0].plot_area.x;
        assert!(
            rotated_x < flat_x,
            "rotated y labels (-45°) must reserve a smaller left margin than flat labels; \
             rotated_x={rotated_x:.1} should be less than flat_x={flat_x:.1}"
        );
    }

    #[test]
    fn compute_layout_show_x_false_bottom_margin_is_zero() {
        // When show_x=false, the x_label_band must be 0 so no bottom space
        // is reserved for labels. We verify by checking that the plot area
        // bottom edge equals the inner_after_legend bottom edge (no gutter).
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };

        let axes_no_x = AxesInput {
            x: axis_input(
                AxisOrient::Bottom,
                None,
                vec!["ABCDEFGHIJ".into(), "KLMNOPQRST".into()],
                None,
            ),
            y: axis_input(AxisOrient::Left, None, vec!["0".into(), "5".into()], None),
            show_x: false,
            show_y: false,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };
        let axes_with_x = AxesInput {
            show_x: true,
            ..axes_no_x.clone()
        };

        let no_x_result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes_no_x,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let with_x_result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes_with_x,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let no_x_bottom = no_x_result.panels[0].plot_area.y + no_x_result.panels[0].plot_area.h;
        let with_x_bottom =
            with_x_result.panels[0].plot_area.y + with_x_result.panels[0].plot_area.h;

        assert!(
            no_x_bottom > with_x_bottom,
            "show_x=false should reserve less bottom space (larger bottom edge); \
             no_x={no_x_bottom:.1}, with_x={with_x_bottom:.1}"
        );
    }

    /// `#182` (closes `#94`): with NEITHER Cartesian axis drawn, the plot area
    /// is exactly the viewport minus the theme padding — no label band, no title
    /// gutter, on either side. The fixture carries a title AND labels on both
    /// axes precisely so every phantom the pre-fix code reserved would show up
    /// as a shortfall here: the y band was `max_label_w + line_h(title) +
    /// title_padding` and the x band the title gutter.
    ///
    /// This is the Rust-side twin of `tests/test_axis_gutter_gating.py`'s
    /// measured `(16, 16, 608, 448)` plot rect; it pins the arithmetic at the
    /// layout boundary, where the reservation lives.
    #[test]
    fn compute_layout_undrawn_axes_reserve_no_band_at_all() {
        let spec = minimal_chart_spec();
        let viewport = Viewport {
            width: 600.0,
            height: 400.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let titled_axes = AxesInput {
            x: axis_input(
                AxisOrient::Bottom,
                Some("X title".into()),
                vec!["ABCDEFGHIJ".into(), "KLMNOPQRST".into()],
                None,
            ),
            y: axis_input(
                AxisOrient::Left,
                Some("Y title".into()),
                vec!["0".into(), "10000".into()],
                None,
            ),
            show_x: false,
            show_y: false,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };
        let theme = default_theme_inputs();
        let result = compute_layout(
            &spec,
            &theme,
            viewport,
            &titled_axes,
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let pad = theme.padding.padding;
        let plot = result.panels[0].plot_area;
        assert_eq!(
            (plot.x, plot.y, plot.w, plot.h),
            (
                pad,
                pad,
                viewport.width - 2.0 * pad,
                viewport.height - 2.0 * pad
            ),
            "undrawn axes must leave the padded inner rect untouched"
        );

        // Discriminating control: the SAME axes, drawn, reserve a real band on
        // both sides — so the equality above is the gate's work, not a fixture
        // whose bands happen to measure zero.
        let drawn = compute_layout(
            &spec,
            &theme,
            viewport,
            &AxesInput {
                show_x: true,
                show_y: true,
                ..titled_axes.clone()
            },
            &[],
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();
        let drawn_plot = drawn.panels[0].plot_area;
        assert!(
            drawn_plot.x > plot.x,
            "a drawn y axis must reserve a left band"
        );
        assert!(
            drawn_plot.h < plot.h,
            "a drawn x axis must reserve a bottom band"
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
        assert_eq!(
            t.colors.background_color,
            palette::Srgba::new(0xFA, 0xF7, 0xF2, 0xFF)
        );
        assert_eq!(
            t.colors.mark_color,
            palette::Srgba::new(0x25, 0x63, 0xEB, 0xFF)
        );
    }

    // ── D5c: empty-partition warning for sparse two-way grid facets ──────────

    /// Build a two-way grid FacetSpec with explicit nrows/ncols that matches
    /// the number of distinct row × col values.
    fn sparse_grid_spec(nrows: u32, ncols: u32) -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "col_cat".into(),
            row: Some("row_cat".into()),
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
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
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c1".into(),
                },
                n_rows: 1,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r1".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c2".into(),
                },
                n_rows: 1,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r1".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c1".into(),
                },
                n_rows: 1,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r2".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c2".into(),
                },
                n_rows: 0,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r2".into(),
                }),
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
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        // Exactly one EmptyPartitions warning.
        let empty_warns: Vec<&LayoutWarning> = result
            .warnings
            .iter()
            .filter(|w| matches!(w, LayoutWarning::EmptyPartitions { .. }))
            .collect();
        assert_eq!(
            empty_warns.len(),
            1,
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
        let dropped = result
            .warnings
            .iter()
            .any(|w| matches!(w, LayoutWarning::PanelsDropped { .. }));
        assert!(
            !dropped,
            "sparse grid must not emit PanelsDropped when grid is correctly sized"
        );
    }

    #[test]
    fn compute_layout_complete_grid_emits_no_empty_partitions_warning() {
        // A 2×2 two-way grid facet where all four cells have data must not
        // emit any EmptyPartitions warning.
        let spec = sparse_grid_spec(2, 2);
        // All four groups have n_rows > 0.
        let complete_groups: Vec<FacetGroup> = vec![
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c1".into(),
                },
                n_rows: 2,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r1".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c2".into(),
                },
                n_rows: 2,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r1".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c1".into(),
                },
                n_rows: 2,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r2".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "c2".into(),
                },
                n_rows: 2,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "r2".into(),
                }),
            },
        ];
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
            &axes,
            &complete_groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
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
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 400.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
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
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
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
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "A".into(),
                },
                n_rows: 10,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "High".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "B".into(),
                },
                n_rows: 10,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "High".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "A".into(),
                },
                n_rows: 10,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "Low".into(),
                }),
            },
            FacetGroup {
                key: FacetKey {
                    field: "col_cat".into(),
                    value: "B".into(),
                },
                n_rows: 10,
                row_key: Some(FacetKey {
                    field: "row_cat".into(),
                    value: "Low".into(),
                }),
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
            x: axis_input(
                AxisOrient::Bottom,
                Some("x".into()),
                vec!["0".into(), "5".into()],
                None,
            ),
            y: axis_input(
                AxisOrient::Left,
                Some("y".into()),
                vec!["0".into(), "5".into()],
                None,
            ),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        };
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        assert_eq!(result.panels.len(), 4, "expected 4 panels in 2×2 grid");

        // Collect panels that have a row_strip_title.
        let stripped_panels: Vec<&PanelLayout> = result
            .panels
            .iter()
            .filter(|p| p.row_strip_title.is_some())
            .collect();
        assert_eq!(
            stripped_panels.len(),
            2,
            "expected one row-strip per row (2 rows)"
        );

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
        let left_col_panels: Vec<&PanelLayout> =
            result.panels.iter().filter(|p| p.col == 0).collect();
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
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        // Column strips: all 4 panels have a col strip_title.
        assert!(
            result.panels.iter().all(|p| p.strip_title.is_some()),
            "all panels should have a column strip_title"
        );

        // Row strips: exactly 2 panels (rightmost column) have row_strip_title.
        let row_strip_count = result
            .panels
            .iter()
            .filter(|p| p.row_strip_title.is_some())
            .count();
        assert_eq!(
            row_strip_count, 2,
            "expected 2 row-strip panels (one per row, rightmost col)"
        );

        // Row strip text values must match the row_cat values.
        let row_strip_texts: Vec<&str> = result
            .panels
            .iter()
            .filter_map(|p| p.row_strip_title.as_ref())
            .map(|s| s.text.as_str())
            .collect();
        assert!(
            row_strip_texts.contains(&"High"),
            "expected 'High' row strip"
        );
        assert!(row_strip_texts.contains(&"Low"), "expected 'Low' row strip");
    }

    /// F-L01-04: `FacetSpec.row_title` set -> row-header strips render
    /// `"{row_title}: {value}"`, independently of `title` (column strips stay
    /// bare when `title` is unset — the two dimensions cannot leak into each
    /// other's prefix).
    #[test]
    fn compute_layout_facet_row_title_prefixes_row_strip_only() {
        let mut spec = two_way_grid_spec();
        spec.facet.as_mut().unwrap().row_title = Some("Region".into());
        let groups = two_way_2x2_groups();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
            &axes,
            &groups,
            &[],
            None,
            None,
            &m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .unwrap();

        let row_strip_texts: Vec<&str> = result
            .panels
            .iter()
            .filter_map(|p| p.row_strip_title.as_ref())
            .map(|s| s.text.as_str())
            .collect();
        assert!(
            row_strip_texts.contains(&"Region: High"),
            "expected 'Region: High'; got {row_strip_texts:?}"
        );
        assert!(
            row_strip_texts.contains(&"Region: Low"),
            "expected 'Region: Low'; got {row_strip_texts:?}"
        );

        // Column strips are unaffected — `title` is still unset on this spec.
        for panel in &result.panels {
            let strip = panel.strip_title.as_ref().unwrap();
            let value = &panel.facet_key.as_ref().unwrap().value;
            assert_eq!(
                &strip.text, value,
                "column strip must stay bare when title is unset"
            );
        }
    }

    // ── padding.auto measures the drawn x tick labels ───────────────────────

    /// Independent oracle for where the x tick labels a layout draws put
    /// ink: `(leftmost, rightmost)` canvas x over every drawn label, from
    /// the renderer's placement (`render/marks/axis.rs`): the pivot is the
    /// tick, a flat label is `Middle`-anchored (`Start` / `End` on the first
    /// / last non-culled tick under `label_flush`, blank or not), a rotated
    /// `Bottom` label `End`-anchored and a rotated `Top` label
    /// `Start`-anchored, rotated about the pivot, with the line height
    /// standing in for the glyphs' ascent and descent.
    fn drawn_x_label_ink(layout: &LayoutResult, m: &dyn TextMetrics, font_size: f64) -> (f64, f64) {
        let mut ink = (f64::INFINITY, f64::NEG_INFINITY);
        for axis in layout
            .axes
            .iter()
            .filter(|a| matches!(a.orient, AxisOrient::Bottom | AxisOrient::Top))
        {
            let flush = axis.label_flush == Some(true);
            let first = axis.ticks.iter().position(|t| !t.culled);
            let last = axis.ticks.iter().rposition(|t| !t.culled);
            for (i, t) in axis.ticks.iter().enumerate() {
                if t.culled || t.label.is_empty() {
                    continue;
                }
                let fs = t.label_font_size.unwrap_or(font_size);
                let w = m.measure_width(&t.label, fs);
                let h = m.line_height(fs);
                let (x0, x1) = if t.label_angle != 0.0 {
                    if axis.orient == AxisOrient::Bottom {
                        (-w, 0.0)
                    } else {
                        (0.0, w)
                    }
                } else if flush && Some(i) == first {
                    (0.0, w)
                } else if flush && Some(i) == last {
                    (-w, 0.0)
                } else {
                    (-w / 2.0, w / 2.0)
                };
                let (sin, cos) = t.label_angle.to_radians().sin_cos();
                for x in [x0, x1] {
                    for y in [-h, h] {
                        let rx = t.position + x * cos - y * sin;
                        ink = (ink.0.min(rx), ink.1.max(rx));
                    }
                }
            }
        }
        ink
    }

    /// `compute_layout` for a chart with no legend or facets, with
    /// `padding.auto` set to `auto` on `theme`.
    fn layout_with_auto(
        theme: &ThemeInputs,
        auto: bool,
        viewport: Viewport,
        axes: &AxesInput,
        m: &dyn TextMetrics,
    ) -> LayoutResult {
        let theme = ThemeInputs {
            padding: ThemePadding {
                padding_auto: auto,
                ..theme.padding.clone()
            },
            ..theme.clone()
        };
        compute_layout(
            &minimal_chart_spec(),
            &theme,
            viewport,
            axes,
            &[],
            &[],
            None,
            None,
            m,
            &legend::LegendOverrides::default(),
            &[],
            CompositeLayoutSeam::default(),
        )
        .expect("layout should succeed")
    }

    /// Eleven currency-style ticks `$0 .. $1,000,000` (the last label 100px
    /// at 10px/char) on a continuous x scale with a 5% padding inset: at
    /// most widths the collision cascade rotates and culls them, and the
    /// long `$1,000,000` is often culled.
    fn currency_x_axes() -> AxesInput {
        let labels: Vec<String> = [
            "$0", "$100,000", "$200,000", "$300,000", "$400,000", "$500,000", "$600,000",
            "$700,000", "$800,000", "$900,000", "$1,000,000",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut x = axis_input(AxisOrient::Bottom, None, labels, None);
        x.tick_projection = Some(TickProjection {
            padding_frac: 0.05,
            major: (0..=10).map(|i| f64::from(i) / 10.0).collect(),
            minor: vec![],
        });
        let y = axis_input(
            AxisOrient::Left,
            None,
            vec!["$0".into(), "$1".into(), "$2".into()],
            None,
        );
        AxesInput {
            x,
            y,
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// The default must reserve only what the drawn labels need. At 170px
    /// the cascade rotates the currency labels to -90 and culls to every
    /// third tick, so the 100px `$1,000,000` is never drawn and the
    /// survivors fit inside the canvas with `auto=false`. The default must
    /// then be the `auto=false` layout exactly, not one narrowed for the
    /// culled label's flat half-width.
    #[test]
    fn padding_auto_leaves_rotated_culled_labels_that_fit_untouched() {
        let axes = currency_x_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 170.0,
            height: 200.0,
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let x_axis = off
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        assert!(
            x_axis.ticks[0].label_angle == -90.0
                && x_axis.ticks.last().is_some_and(|t| t.culled),
            "fixture assumption: rotated to -90 with the last label culled ({:?})",
            x_axis.ticks,
        );
        let (lo, hi) = drawn_x_label_ink(&off, &m, theme.typography.label_font_size);
        assert!(
            lo >= 0.0 && hi <= viewport.width,
            "fixture assumption: the drawn labels fit with auto off ({lo}, {hi})",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert_eq!(
            on, off,
            "labels that already fit must leave the layout exactly as auto=false lays it out",
        );
    }

    /// Every width in the cascade's range, derived by sweeping rather than
    /// hand-listed (it crosses the -90 cull strides, -60, -45, -30 and
    /// flat). Where the drawn labels fit with `auto=false`, the default must
    /// lay out exactly the same labels at the same angle (the whole layout
    /// is equal). Where they do not fit (the flat band, whose `$1,000,000`
    /// overhangs the right edge), the default must keep them inside the
    /// canvas. Both populations must be non-empty, so neither arm is
    /// vacuous.
    #[test]
    fn padding_auto_matches_auto_off_wherever_the_drawn_labels_fit() {
        let axes = currency_x_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let fs = theme.typography.label_font_size;
        let mut fitting = 0;
        let mut rescued = 0;
        let mut presentations = std::collections::BTreeSet::new();
        for width in 150..=1400 {
            let viewport = Viewport {
                width: f64::from(width),
                height: 200.0,
            };
            let off = layout_with_auto(&theme, false, viewport, &axes, &m);
            let on = layout_with_auto(&theme, true, viewport, &axes, &m);
            let x_axis = off
                .axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present");
            let drawn = x_axis.ticks.iter().filter(|t| !t.culled).count();
            presentations.insert((x_axis.ticks[0].label_angle as i64, drawn));
            let (lo, hi) = drawn_x_label_ink(&off, &m, fs);
            if lo >= 0.0 && hi <= viewport.width {
                fitting += 1;
                assert_eq!(
                    on, off,
                    "at {width}px the labels fit with auto off, so the default must match it",
                );
            } else {
                rescued += 1;
                let (lo, hi) = drawn_x_label_ink(&on, &m, fs);
                assert!(
                    lo >= -1e-6 && hi <= viewport.width + 1e-6,
                    "at {width}px the default must keep the drawn labels inside the canvas ({lo}, {hi})",
                );
            }
        }
        assert!(
            fitting > 0 && rescued > 0,
            "both arms must be exercised (fitting {fitting}, rescued {rescued})",
        );
        assert!(
            presentations.len() >= 5,
            "the sweep must cross the cascade's presentations: {presentations:?}",
        );
    }

    /// A flat edge label that overhangs is still rescued, and the extra is
    /// what the DRAWN label needs: the tick sits inside the plot's edge by
    /// the scale's padding inset (here capped at 8px), so the extra is the
    /// half-width less that inset less the 16px cushion, `30 - 8 - 16 = 6`,
    /// and the label's right ink lands exactly on the canvas edge. Charging
    /// the half-width as if the tick sat on the edge would reserve 14 and
    /// leave an 8px gap.
    #[test]
    fn padding_auto_flat_edge_label_extra_is_less_the_tick_inset() {
        let mut axes = continuous_x_axes_with_long_last_label();
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.05;
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 200.0,
        };
        let fs = theme.typography.label_font_size;
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let (_, hi_off) = drawn_x_label_ink(&off, &m, fs);
        assert!(
            hi_off > viewport.width,
            "fixture assumption: the flat last label overhangs with auto off ({hi_off})",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        let (_, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!(
            (hi - viewport.width).abs() < 1e-6,
            "the rescued label must end exactly on the canvas edge (right ink {hi})",
        );
        let right_inset = viewport.width - on.padded_region.x - on.padded_region.w;
        assert!(
            (right_inset - 22.0).abs() < 1e-6,
            "right inset = 16 + (30 - 8 - 16) = 22, got {right_inset}",
        );
    }

    /// A continuous x axis with `labels` at an explicit `label_angle`,
    /// with no y axis drawn (so the left cushion is just the padding).
    fn continuous_x_axes_at_angle(
        orient: AxisOrient,
        labels: Vec<String>,
        angle: f64,
    ) -> AxesInput {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.orient = orient;
        axes.x.tick_labels = labels;
        axes.x.overrides.labels.angle = Some(angle);
        axes.show_y = false;
        axes
    }

    /// An explicit `label_angle=-45` whose long first label hangs down and
    /// to the left of its tick (`End` anchor) far past the left canvas
    /// edge: the default must widen the left margin until it is inside.
    #[test]
    fn padding_auto_contains_an_explicit_minus_45_label_on_the_left() {
        let axes = continuous_x_axes_at_angle(
            AxisOrient::Bottom,
            vec!["-999999999".into(), "0".into()],
            -45.0,
        );
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let fs = theme.typography.label_font_size;
        let (lo_off, _) = drawn_x_label_ink(&layout_with_auto(&theme, false, viewport, &axes, &m), &m, fs);
        assert!(lo_off < 0.0, "fixture assumption: clips left with auto off ({lo_off})");
        let (lo, hi) = drawn_x_label_ink(&layout_with_auto(&theme, true, viewport, &axes, &m), &m, fs);
        assert!(
            lo >= -1e-6 && hi <= viewport.width + 1e-6,
            "the default must keep the -45 labels inside the canvas ({lo}, {hi})",
        );
    }

    /// An explicit `label_angle=-90` reaches one line height either side of
    /// its tick; with 4px padding that clips both edges. The default must
    /// widen both margins.
    #[test]
    fn padding_auto_contains_explicit_minus_90_labels_on_both_sides() {
        let axes = continuous_x_axes_at_angle(
            AxisOrient::Bottom,
            vec!["0".into(), "5".into(), "10".into()],
            -90.0,
        );
        let mut axes = axes;
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.5, 1.0];
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let mut theme = default_theme_inputs();
        theme.padding.padding = 4.0;
        let viewport = Viewport {
            width: 300.0,
            height: 300.0,
        };
        let fs = theme.typography.label_font_size;
        let (lo_off, hi_off) =
            drawn_x_label_ink(&layout_with_auto(&theme, false, viewport, &axes, &m), &m, fs);
        assert!(
            lo_off < 0.0 && hi_off > viewport.width,
            "fixture assumption: clips both edges with auto off ({lo_off}, {hi_off})",
        );
        let (lo, hi) = drawn_x_label_ink(&layout_with_auto(&theme, true, viewport, &axes, &m), &m, fs);
        assert!(
            lo >= -1e-6 && hi <= viewport.width + 1e-6,
            "the default must keep the -90 labels inside the canvas ({lo}, {hi})",
        );
    }

    /// A Top-orient x axis anchors a rotated label at its START, so at -45
    /// the long last label rises up and to the RIGHT of its tick, past the
    /// right canvas edge. The default must widen the right margin.
    #[test]
    fn padding_auto_contains_top_orient_rotated_labels_on_the_right() {
        let axes = continuous_x_axes_at_angle(
            AxisOrient::Top,
            vec!["0".into(), "999999999".into()],
            -45.0,
        );
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let fs = theme.typography.label_font_size;
        let (_, hi_off) = drawn_x_label_ink(&layout_with_auto(&theme, false, viewport, &axes, &m), &m, fs);
        assert!(
            hi_off > viewport.width,
            "fixture assumption: clips right with auto off ({hi_off})",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        let (lo, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!(
            lo >= -1e-6 && hi <= viewport.width + 1e-6,
            "the default must keep the Top -45 labels inside the canvas ({lo}, {hi})",
        );
    }

    /// A categorical x axis (no `tick_projection`) places its labels at
    /// band centers, inside the plot edge by construction; its long labels
    /// get no edge-label extra, so the default is the `auto=false` layout.
    #[test]
    fn padding_auto_leaves_a_categorical_x_axis_untouched() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_projection = None;
        axes.x.tick_labels = vec!["alpha".into(), "a much longer category".into()];
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        for width in [120.0, 200.0, 400.0] {
            let viewport = Viewport {
                width,
                height: 300.0,
            };
            assert_eq!(
                layout_with_auto(&theme, true, viewport, &axes, &m),
                layout_with_auto(&theme, false, viewport, &axes, &m),
                "a chart with no continuous axis must lay out identically at {width}px",
            );
        }
    }

    /// The right-side twin of
    /// `padding_auto_never_worsens_an_infeasible_title_when_a_tick_correction_also_fires`:
    /// the tick pass wants RIGHT padding (the flat `"999999"` last label)
    /// while the 66-char x title is wider than the 300px viewport. The right
    /// tick extra must be dropped before the title pass, exactly like the
    /// left one, so the default is byte-identical to `auto=false`.
    #[test]
    fn padding_auto_drops_the_right_tick_extra_on_an_infeasible_x_title() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.title.set(Some("a".repeat(66)));
        let viewport = Viewport {
            width: 300.0,
            height: 300.0,
        };
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_inputs_with_padding_auto_off();
        let title_half_w = m.measure_width(&"a".repeat(66), theme.typography.title_font_size) / 2.0;
        assert!(
            !recentering_is_feasible(title_half_w, viewport.width),
            "fixture assumption: the title is wider than the viewport",
        );
        let ticks_alone = auto_padding_for_edge_ticks(
            &theme,
            &axes,
            &m,
            Inset::default(),
            x_overhang_without_auto(&theme, &axes, &m, viewport),
        );
        assert!(
            ticks_alone.right > 0.0 && ticks_alone.left == 0.0,
            "fixture assumption: only the right tick extra fires ({ticks_alone:?})",
        );
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            layout_with_auto(&theme, false, viewport, &axes, &m),
            "an infeasible x title must drop the right tick extra too",
        );
    }

    /// A chart whose `padding.auto` has no fixed point, so
    /// `settle_auto_padding` runs all `AUTO_PADDING_MAX_PASSES` passes. At
    /// 175px with `auto=false` the flat 80px last label overhangs the right
    /// edge; the extra that rescues it narrows the plot just enough that
    /// the next pass's inset grows the overhang, and the slightly larger
    /// extra after that narrows the plot past the cascade's flat threshold,
    /// so the labels rotate, need no extra, and the next pass is back at
    /// `auto=false`. That three-pass cycle never settles; pass 64 lands on
    /// its no-extra phase. The fallback must return the last COVERING pass
    /// (the rotated labels, which fit), not zero and not the last pass.
    #[test]
    fn settle_auto_padding_returns_the_last_covering_extra_when_the_cascade_cycles() {
        let spec = minimal_chart_spec();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels[1] = "9".repeat(8);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.05;
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 175.0,
            height: 300.0,
        };
        let fs = theme.typography.label_font_size;

        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let off_x = off
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        assert!(
            off_x.ticks[0].label_angle == 0.0
                && drawn_x_label_ink(&off, &m, fs).1 > viewport.width,
            "fixture assumption: flat and overhanging with auto off",
        );

        let legend_overrides = legend::LegendOverrides::default();
        let inputs = LayoutInputs {
            spec: &spec,
            theme: &theme,
            viewport,
            axes: &axes,
            facet_groups: &[],
            legend_entries: &[],
            legend_title: None,
            colorbar: None,
            metrics: &m,
            legend_overrides: &legend_overrides,
            aux_legend_inputs: &[],
            seam: CompositeLayoutSeam::default(),
        };
        let extra = settle_auto_padding(&inputs, base_padding_insets(&theme));
        assert!(
            extra.right > 0.0,
            "the fallback must return the covering pass's extra, not zero ({extra:?})",
        );

        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        let on_x = on
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let (lo, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!(
            on_x.ticks[0].label_angle != 0.0 && lo >= -1e-6 && hi <= viewport.width + 1e-6,
            "the covering pass draws rotated labels that fit ({:?}, ink {lo}..{hi})",
            on_x.ticks,
        );
        let base = base_padding_insets(&theme);
        let probe = probe_auto_padding(
            resolve_padding_inset(&theme, base, extra),
            &inputs,
            &AutoPlotRoom::of_auto_off(&inputs, base),
        );
        let needed = auto_padding_for_edge_ticks(&theme, &axes, &m, Inset::default(), probe.x_overhang);
        assert!(
            needed.right == 0.0,
            "the covering pass's labels need no edge extra ({needed:?}), yet removing it \
             brings back the flat overhang above: no fixed point exists",
        );
    }

    /// `rotated_x_label_horizontal_extent` against hand-computed reaches.
    #[test]
    fn rotated_x_label_horizontal_extent_hand_computed() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let close = |(a, b): (f64, f64), (c, d): (f64, f64)| (a - c).abs() < 1e-9 && (b - d).abs() < 1e-9;
        // Flat Middle: half the width each side, no line-height term.
        assert!(close(
            axis::rotated_x_label_horizontal_extent(0.0, TextAnchor::Middle, 60.0, 13.2),
            (30.0, 30.0),
        ));
        // Bottom -45 (End): cos·w + sin·line_h left, sin·line_h right.
        assert!(close(
            axis::rotated_x_label_horizontal_extent(-45.0, TextAnchor::End, 100.0, 13.2),
            (s * 100.0 + s * 13.2, s * 13.2),
        ));
        // Top -45 (Start): mirrored onto the right.
        assert!(close(
            axis::rotated_x_label_horizontal_extent(-45.0, TextAnchor::Start, 100.0, 13.2),
            (s * 13.2, s * 100.0 + s * 13.2),
        ));
        // -90: one line height either side, whatever the width.
        assert!(close(
            axis::rotated_x_label_horizontal_extent(-90.0, TextAnchor::End, 100.0, 13.2),
            (13.2, 13.2),
        ));
    }

    /// Every drawn label counts, not only the edge ones. With
    /// `label_overlap=True` (`ShowAll`) a long flat label one tick in from
    /// the left end is centred on its tick, so half of it reaches past the
    /// left canvas edge while the short first label `"0"` does not; the
    /// default must widen the left margin for it.
    #[test]
    fn padding_auto_contains_a_long_interior_label_near_the_left_end() {
        let mut axes = continuous_x_axes_at_angle(
            AxisOrient::Bottom,
            vec!["0".into(), "9".repeat(20), "5".into(), "10".into()],
            0.0,
        );
        axes.x.overrides.labels.angle = None;
        axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.1, 0.5, 1.0];
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 600.0,
            height: 300.0,
        };
        let fs = theme.typography.label_font_size;
        let (lo_off, _) =
            drawn_x_label_ink(&layout_with_auto(&theme, false, viewport, &axes, &m), &m, fs);
        assert!(lo_off < 0.0, "fixture assumption: the interior label clips left ({lo_off})");
        let (lo, hi) = drawn_x_label_ink(&layout_with_auto(&theme, true, viewport, &axes, &m), &m, fs);
        assert!(
            lo >= -1e-6 && hi <= viewport.width + 1e-6,
            "the default must keep the interior label inside the canvas ({lo}, {hi})",
        );
    }

    /// `label_flush` anchors the first label at its start and the last at
    /// its end, so neither reaches past the plot: a flat `"999999"` that
    /// overhangs by half its width without flush needs nothing with it,
    /// and the default must be the `auto=false` layout.
    #[test]
    fn padding_auto_charges_nothing_for_flush_edge_labels() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.overrides.labels.flush = Some(true);
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 200.0,
            height: 200.0,
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let (lo, hi) = drawn_x_label_ink(&off, &m, theme.typography.label_font_size);
        assert!(
            lo >= 0.0 && hi <= viewport.width,
            "fixture assumption: flush labels fit with auto off ({lo}, {hi})",
        );
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            off,
            "flush edge labels reach no further than the plot, so auto must add nothing",
        );
    }

    /// `label_flush` re-anchors the last NON-CULLED tick even when its label
    /// is blank, so the last DRAWN label (`"999999"`, just inside the right
    /// end) stays centred and overhangs the right canvas edge. The default
    /// must rescue it rather than treat it as flushed.
    #[test]
    fn padding_auto_rescues_a_centred_label_behind_a_blank_flushed_end() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.overrides.labels.flush = Some(true);
        // Shown flat and unculled despite the 2% gap to the blank end tick.
        axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        axes.x.tick_labels = vec!["0".into(), "999999".into(), String::new()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.98, 1.0];
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 200.0,
        };
        let fs = theme.typography.label_font_size;
        let (_, hi_off) =
            drawn_x_label_ink(&layout_with_auto(&theme, false, viewport, &axes, &m), &m, fs);
        assert!(hi_off > viewport.width, "fixture assumption: clips right with auto off ({hi_off})");
        let (lo, hi) = drawn_x_label_ink(&layout_with_auto(&theme, true, viewport, &axes, &m), &m, fs);
        assert!(
            lo >= -1e-6 && (hi - viewport.width).abs() < 1e-6,
            "the default must end the centred label exactly on the canvas edge ({lo}, {hi})",
        );
    }

    /// Faceted: the reach is taken over every panel's x axis against the
    /// unsplit plot region, so the leftmost panel's first label decides the
    /// left margin and the rightmost panel's last label the right margin.
    /// Both overhang with `auto=false`; the default must contain both.
    #[test]
    fn padding_auto_contains_the_outer_labels_of_a_faceted_x_axis() {
        let spec = faceted_spec(2);
        let groups: Vec<FacetGroup> = three_groups().into_iter().take(2).collect();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["-999999999".into(), "0".into(), "999999999".into()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.5, 1.0];
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let viewport = Viewport {
            width: 700.0,
            height: 300.0,
        };
        let fs = default_theme_inputs().typography.label_font_size;
        let run = |auto: bool| {
            let mut theme = default_theme_inputs();
            theme.padding.padding_auto = auto;
            compute_layout(
                &spec,
                &theme,
                viewport,
                &axes,
                &groups,
                &[],
                None,
                None,
                &m,
                &legend::LegendOverrides::default(),
                &[],
                CompositeLayoutSeam::default(),
            )
            .expect("faceted layout should succeed")
        };
        let off = run(false);
        assert_eq!(off.panels.len(), 2, "fixture assumption: two panels");
        let (lo_off, hi_off) = drawn_x_label_ink(&off, &m, fs);
        assert!(
            lo_off < 0.0 && hi_off > viewport.width,
            "fixture assumption: the outer labels clip both edges with auto off ({lo_off}, {hi_off})",
        );
        let on = run(true);
        let (lo, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!(
            lo >= -1e-6 && hi <= viewport.width + 1e-6,
            "the default must keep both outer labels inside the canvas ({lo}, {hi})",
        );
    }

    /// A run where NO pass covers, and the no-cover fallback would leave
    /// the plot unusable. On a 288x150 chart the flat 100px last label
    /// overhangs the right edge with `auto=false`, by its 50px half-width
    /// past the plot edge less the 16px padding. The 34px extra that
    /// contains it narrows the plot until the labels no longer fit flat in
    /// half its width, so the cascade rotates them, and their band leaves
    /// the plot under half its 9px `auto=false` height: that pass is
    /// unusable, and the next one is back at `auto=false`. The fallback,
    /// the per-side maximum need, is that same 34px extra, so auto adds
    /// nothing and the chart is laid out exactly as `auto=false` lays it.
    #[test]
    fn settle_auto_padding_adds_nothing_when_the_no_cover_fallback_would_collapse() {
        let spec = minimal_chart_spec();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["0".into(), "5".into(), "9".repeat(10)];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.0;
            p.major = vec![0.0, 0.5, 1.0];
        }
        axes.y.tick_labels = vec!["0".into(), "5".into()];
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 288.0,
            height: 150.0,
        };
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let pass_one = probe_auto_padding(base, &inputs, &room);
        let pass_one_need = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), pass_one.x_overhang, &room)
        .total;
        assert_eq!(
            pass_one_need,
            Inset {
                right: 34.0,
                ..Inset::default()
            },
            "fixture assumption: the auto-off label needs 50 - 16 = 34px on the right",
        );
        let pass_two = probe_auto_padding(resolve_padding_inset(&theme, base, pass_one_need), &inputs, &room);
        assert!(
            pass_two.unusable,
            "fixture assumption: that extra leaves the plot unusable ({pass_two:?})",
        );
        assert_eq!(
            settle_auto_padding(&inputs, base),
            Inset::default(),
            "a no-cover fallback that leaves the plot unusable must add nothing",
        );
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            layout_with_auto(&theme, false, viewport, &axes, &m),
            "the default must be the auto=false layout",
        );
    }

    /// A `label_overlap=True` axis whose long flat label sits just right of
    /// the axis centre (`f = 0.55`), on a plot the explicit 250px right
    /// padding pushes to the left of the canvas, so the label's left half
    /// crosses the left canvas edge. The left margin moves its tick by
    /// `0.45·d`, so it can be rescued from the left, and the default must.
    /// Charts using it are 800px tall: the x label band is reserved as if
    /// the long label rotated to -90.
    fn axes_with_a_long_label_right_of_centre() -> AxesInput {
        let mut axes = continuous_x_axes_at_angle(
            AxisOrient::Bottom,
            vec!["0".into(), "9".repeat(44), "10".into()],
            0.0,
        );
        axes.x.overrides.labels.angle = None;
        axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.55, 1.0];
        }
        axes
    }

    fn theme_with_wide_explicit_right_padding() -> ThemeInputs {
        let mut theme = default_theme_inputs();
        theme.padding.padding_right = Some(250.0);
        theme
    }

    #[test]
    fn padding_auto_rescues_a_label_right_of_centre_from_the_left() {
        let axes = axes_with_a_long_label_right_of_centre();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = theme_with_wide_explicit_right_padding();
        let viewport = Viewport {
            width: 600.0,
            height: 800.0,
        };
        let fs = theme.typography.label_font_size;
        let (lo_off, hi_off) =
            drawn_x_label_ink(&layout_with_auto(&theme, false, viewport, &axes, &m), &m, fs);
        assert!(
            lo_off < 0.0 && hi_off <= viewport.width,
            "fixture assumption: only the left edge is crossed with auto off ({lo_off}, {hi_off})",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        let (lo, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!(
            lo.abs() < 1e-6 && hi <= viewport.width,
            "the default must bring the label's left ink exactly to the canvas edge ({lo}, {hi})",
        );
    }

    /// The weighted rule on a mock, at the default `padding_frac` 0.05 on
    /// a plot wider than 160px, so the scale inset is capped at 8px and the
    /// ticks sit at `x + 8 + t·(w - 16)` for domain fraction `t`. Widening
    /// the left margin then moves a tick by `1 - t` per pixel and the right
    /// margin by `t`, NOT by the tick's fraction of the plot width. A label
    /// whose ink overhangs the left canvas edge by `o` is charged
    /// `o / (1 - t)`, one past the right edge `o / t`.
    #[test]
    fn x_label_overhang_weights_each_label_by_its_true_lever() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let viewport = Viewport {
            width: 600.0,
            height: 800.0,
        };
        let fs = default_theme_inputs().typography.label_font_size;
        let check = |theme: &ThemeInputs, axes: &AxesInput, t: f64, left: bool| {
            let off = layout_with_auto(theme, false, viewport, axes, &m);
            let region = off.plot_region;
            assert!(region.w * 0.05 > 8.0, "fixture assumption: the inset is capped");
            let x_axis = off
                .axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present");
            let long = x_axis
                .ticks
                .iter()
                .find(|t| t.label.len() > 10)
                .expect("the long label is drawn");
            let w = m.measure_width(&long.label, fs);
            let region_fraction = (long.position - region.x) / region.w;
            let got = x_overhang_without_auto(theme, axes, &m, viewport);
            if left {
                let o = -(long.position - w / 2.0);
                assert!(o > 0.0, "fixture assumption: crosses left");
                let want = region.x + o / (1.0 - t);
                let region_fraction_rule = region.x + o / (1.0 - region_fraction);
                assert!((want - region_fraction_rule).abs() > 1.0, "the two rules must differ");
                assert!((got.left - want).abs() < 1e-6, "left {} != o/(1-t) = {want}", got.left);
            } else {
                let o = long.position + w / 2.0 - viewport.width;
                assert!(o > 0.0, "fixture assumption: crosses right");
                let gap = viewport.width - (region.x + region.w);
                let want = gap + o / t;
                let region_fraction_rule = gap + o / region_fraction;
                assert!((want - region_fraction_rule).abs() > 1.0, "the two rules must differ");
                assert!((got.right - want).abs() < 1e-6, "right {} != o/t = {want}", got.right);
            }
        };
        // A 184px plot (the other side explicit at 400px), the long label
        // at t = 0.75 crossing the left edge, then at t = 0.25 crossing the
        // right edge: there the capped inset makes the lever 0.25 where the
        // region fraction would say about 0.27.
        let mut axes = axes_with_a_long_label_right_of_centre();
        axes.x.tick_labels[1] = "9".repeat(34);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.05;
            p.major = vec![0.0, 0.75, 1.0];
        }
        let mut theme = default_theme_inputs();
        theme.padding.padding_right = Some(400.0);
        check(&theme, &axes, 0.75, true);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.25, 1.0];
        }
        let mut theme = default_theme_inputs();
        theme.padding.padding_left = Some(400.0);
        check(&theme, &axes, 0.25, false);
    }

    /// At the limit the margin cannot move the label. At the default
    /// `padding_frac` 0.05 on a plot wider than 160px the inset is capped,
    /// so the last tick sits a fixed 8px from the right edge and widening
    /// the LEFT margin does not move it at all. Its 600px flat label crosses
    /// the left canvas edge, and must contribute nothing on the left: the
    /// left value is the first label's alone and the default is exactly the
    /// `auto=false` layout (charging it collapses the plot to nothing).
    #[test]
    fn a_label_pinned_at_the_far_end_contributes_nothing_to_the_near_margin() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels[1] = "9".repeat(60);
        axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.05;
        }
        axes.show_y = false;
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        // The right side is explicit, so only the left side's rule is in play.
        let mut theme = default_theme_inputs();
        theme.padding.padding_right = Some(16.0);
        let viewport = Viewport {
            width: 300.0,
            height: 800.0,
        };
        let fs = theme.typography.label_font_size;
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert!(off.plot_region.w * 0.05 > 8.0, "fixture assumption: the inset is capped");
        let x_axis = off
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let last = x_axis.ticks.last().expect("two ticks");
        assert!(
            last.position - m.measure_width(&last.label, fs) / 2.0 < 0.0,
            "fixture assumption: the far-end label crosses the left canvas edge",
        );
        let first = x_axis.ticks.first().expect("two ticks");
        let got = x_overhang_without_auto(&theme, &axes, &m, viewport);
        let first_only = off.plot_region.x - (first.position - m.measure_width(&first.label, fs) / 2.0);
        assert!(
            (got.left - first_only).abs() < 1e-9,
            "the pinned label must not count on the left ({} vs {first_only})",
            got.left,
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert!(on.panels[0].plot_area.w > 100.0, "the plot must not collapse: {:?}", on.panels[0]);
        assert_eq!(on, off, "a label the left margin cannot move leaves the layout as auto=false");
    }

    /// The right-side twin: a Top-orient axis at -45 anchors its labels at
    /// their START, so the long (elided-to-fit) first label rises up and to
    /// the right of the first tick, which the capped inset pins 8px from the
    /// LEFT edge. With no base padding, its ink crosses the right canvas
    /// edge, yet widening the right margin does not move its tick, so it
    /// contributes nothing on the right and the default is `auto=false`.
    /// (The left side is explicit, and the last label is blank.)
    #[test]
    fn a_label_pinned_at_the_near_end_contributes_nothing_to_the_far_margin() {
        let mut axes = continuous_x_axes_at_angle(
            AxisOrient::Top,
            vec!["9".repeat(50), String::new()],
            -45.0,
        );
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.05;
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let mut theme = default_theme_inputs();
        theme.padding.padding = 0.0;
        theme.padding.padding_left = Some(0.0);
        let viewport = Viewport {
            width: 300.0,
            height: 800.0,
        };
        let fs = theme.typography.label_font_size;
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert!(off.plot_region.w * 0.05 > 8.0, "fixture assumption: the inset is capped");
        let (_, hi_off) = drawn_x_label_ink(&off, &m, fs);
        assert!(
            hi_off > viewport.width,
            "fixture assumption: the first label crosses the right canvas edge ({hi_off})",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert!(on.panels[0].plot_area.w > 100.0, "the plot must not collapse: {:?}", on.panels[0]);
        assert_eq!(on, off, "a label the right margin cannot move leaves the layout as auto=false");
    }

    /// An infeasible charge contributes nothing, and does not squeeze the
    /// other side. No base padding, `padding_frac` 0.001 on a 450px plot
    /// (inset 0.45px, NOT capped), x labels at an explicit -45: the last
    /// label hangs down and to the left of its tick from the right end. Its
    /// ink crosses the left canvas edge, but widening the left margin moves
    /// that tick by only 0.001 per pixel, so containing it there would take
    /// thousands of pixels: more than the left side can widen before the
    /// plot collapses. Its ink also reaches just past the right edge, which
    /// the right margin can fix at full lever. The default must add no left
    /// extra and give the right side exactly the extra it gets when the left
    /// side is explicit (so takes no auto share of the budget). The first
    /// label is blank, so nothing else charges the left.
    #[test]
    fn an_infeasible_charge_leaves_the_other_sides_extra_unchanged() {
        let mut axes = continuous_x_axes_at_angle(
            AxisOrient::Bottom,
            vec![String::new(), "9".repeat(80)],
            -45.0,
        );
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.001;
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let mut theme = default_theme_inputs();
        theme.padding.padding = 0.0;
        let viewport = Viewport {
            width: 450.0,
            height: 800.0,
        };
        let fs = theme.typography.label_font_size;
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert!(off.plot_region.w * 0.001 < 8.0, "fixture assumption: the inset is not capped");
        let (lo_off, hi_off) = drawn_x_label_ink(&off, &m, fs);
        assert!(
            lo_off < 0.0 && hi_off > viewport.width,
            "fixture assumption: the label crosses both canvas edges ({lo_off}, {hi_off})",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        let mut left_explicit = theme.clone();
        left_explicit.padding.padding_left = Some(0.0);
        let reference = layout_with_auto(&left_explicit, true, viewport, &axes, &m);
        let right_inset = |l: &LayoutResult| viewport.width - l.padded_region.x - l.padded_region.w;
        assert_eq!(on.padded_region.x, 0.0, "no left extra for an infeasible charge");
        assert!(right_inset(&on) > 0.0, "fixture assumption: the right side is rescued");
        assert!(
            (right_inset(&on) - right_inset(&reference)).abs() < 1e-9,
            "the right extra must not be squeezed ({} vs {})",
            right_inset(&on),
            right_inset(&reference),
        );
    }

    /// `label_flush` anchors the first label at its START, so a long first
    /// label that would need a left extra centred reaches only rightward;
    /// the default must be the `auto=false` layout.
    #[test]
    fn padding_auto_charges_nothing_for_a_long_flushed_first_label() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["-999999999999999".into(), "0".into()];
        axes.x.overrides.labels.flush = Some(true);
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 600.0,
            height: 300.0,
        };
        let mut centred = axes.clone();
        centred.x.overrides.labels.flush = None;
        let (lo_centred, _) = drawn_x_label_ink(
            &layout_with_auto(&theme, false, viewport, &centred, &m),
            &m,
            theme.typography.label_font_size,
        );
        assert!(lo_centred < 0.0, "fixture assumption: centred, it clips left ({lo_centred})");
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            off,
            "a start-anchored first label reaches no further left than its tick",
        );
    }

    /// Three ticks with a long flat first label, on a continuous x axis with
    /// no scale inset, so each label's budget is half the plot width.
    fn axes_whose_left_rescue_rotates_the_labels() -> AxesInput {
        let mut axes = continuous_x_axes_with_long_first_label_and_capped_y_band();
        axes.y.overrides.placement.max_band = None;
        axes.x.tick_labels = vec![format!("-{}", "9".repeat(11)), "5".into(), "10".into()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.5, 1.0];
        }
        axes
    }

    /// The x tick labels `plot` draws on its first panel, as `(angle,
    /// culled)` per tick, or `None` when the panel draws no x axis.
    fn x_presentation(plot: &PlotLayout) -> Option<Vec<(f64, bool)>> {
        plot.axis_layouts
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .map(|a| a.ticks.iter().map(|t| (t.label_angle, t.culled)).collect())
    }

    /// A rescue that would cull a label is declined. On a 360x110 chart,
    /// `auto=false` draws the three labels flat and the 120px first label
    /// overhangs the left edge. The 18px left extra that contains it
    /// narrows the plot until the three labels no longer fit flat, rotated
    /// they would need a band that leaves no plot height, and the cascade's
    /// band budget would cull the middle label. That pass loses one of its
    /// own ticks where `auto=false` loses none, and its labels (every 2nd
    /// tick) are fewer than [`MIN_EVENLY_THINNED_X_LABELS`], so it is not an
    /// even thinning either. It is declined like one that collapses the
    /// plot, so the chart is the `auto=false` layout: every label it draws,
    /// by text, is drawn, and the first still clips.
    #[test]
    fn a_rescue_that_would_cull_a_label_is_declined() {
        let axes = axes_whose_left_rescue_rotates_the_labels();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 360.0,
            height: 110.0,
        };
        let fs = theme.typography.label_font_size;
        let presentation = |l: &LayoutResult| {
            l.axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present")
                .ticks
                .iter()
                .map(|t| (t.label_angle, t.culled))
                .collect::<Vec<_>>()
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert_eq!(presentation(&off), vec![(0.0, false); 3], "fixture assumption: flat with auto off");
        assert!(drawn_x_label_ink(&off, &m, fs).0 < 0.0, "fixture assumption: the first label clips left");
        let drawn_text = |l: &LayoutResult| {
            l.axes
                .iter()
                .filter(|a| a.orient == AxisOrient::Bottom)
                .flat_map(|a| a.ticks.iter().filter(|t| !t.culled).map(|t| t.label.clone()))
                .collect::<Vec<_>>()
        };
        let base = base_padding_insets(&theme);
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let culling = lay_out_at(Inset { left: base.left + 18.0, ..base }, &inputs).plot;
        assert_eq!(
            x_presentation(&culling),
            Some(vec![(0.0, false), (0.0, true), (0.0, false)]),
            "fixture assumption: the 18px rescue culls the middle label",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert_eq!(drawn_text(&on), axes.x.tick_labels, "every label auto=false draws is drawn");
        assert_eq!(on, off, "the declined rescue leaves the auto=false layout");
        assert!(drawn_x_label_ink(&on, &m, fs).0 < 0.0, "the first label still clips");
    }

    /// Each axis is judged on its own: when the x rescue would cull a label
    /// (the chart above, its long first label overhanging the left edge,
    /// 280-440px wide) and a y title needs a rescue that culls none, x is
    /// closed and
    /// the y rescue still ships. The x margins are the `auto=false` ones,
    /// every label `auto=false` draws is drawn, and the y margins are those
    /// of the same chart with its x sides explicit at their `auto=false`
    /// values. The first label's length, the y title's length and the chart
    /// height are swept; the sweep must reach charts where the y rescue is
    /// real.
    #[test]
    fn a_culling_x_rescue_is_dropped_and_the_y_rescue_still_ships() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let mut shipped = 0;
        for (first, width, title, height) in (8..=16).step_by(4).flat_map(|f| {
            (280..=440).step_by(40).flat_map(move |w| {
                (4..=24).step_by(4).flat_map(move |t| (90..=220).step_by(10).map(move |h| (f, w, t, h)))
            })
        }) {
            {
                let mut axes = axes_whose_left_rescue_rotates_the_labels();
                axes.x.tick_labels[0] = format!("-{}", "9".repeat(first));
                axes.y.title.set(Some("a".repeat(title)));
                let theme = default_theme_inputs();
                let viewport = Viewport {
                    width: f64::from(width),
                    height: f64::from(height),
                };
                let base = base_padding_insets(&theme);
                let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
                let room = AutoPlotRoom::of_auto_off(&inputs, base);
                let probe = probe_auto_padding(base, &inputs, &room);
                let x_rescue = auto_padding_extra(&inputs, probe.chrome, probe.x_overhang, &room).ticks;
                if x_rescue.left + x_rescue.right <= 0.0 {
                    continue;
                }
                let at_x_rescue = Inset {
                    left: base.left + x_rescue.left,
                    right: base.right + x_rescue.right,
                    ..base
                };
                if !room.culls_x_labels(&lay_out_at(at_x_rescue, &inputs).plot) {
                    continue;
                }
                let off = layout_with_auto(&theme, false, viewport, &axes, &m);
                let on = layout_with_auto(&theme, true, viewport, &axes, &m);
                let mut explicit_x = theme.clone();
                explicit_x.padding.padding_left = Some(base.left);
                explicit_x.padding.padding_right = Some(base.right);
                let want = layout_with_auto(&explicit_x, true, viewport, &axes, &m);
                assert_eq!(x_margins(&on), x_margins(&off), "{title}-char y title at {height}px: x adds nothing");
                assert_eq!(y_margins(&on), y_margins(&want), "{title}-char y title at {height}px: the y rescue ships");
                let drawn = |l: &LayoutResult| {
                    l.axes
                        .iter()
                        .filter(|a| a.orient == AxisOrient::Bottom)
                        .flat_map(|a| a.ticks.iter().filter(|t| !t.culled).map(|t| t.label.clone()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(drawn(&on), drawn(&off), "{title}-char y title at {height}px: every label auto=false draws");
                shipped += usize::from(y_margins(&want) != y_margins(&off));
            }
        }
        assert!(shipped > 0, "the sweep must reach a real y rescue");
    }

    /// Eleven ticks `0, 0.1, …, 1` on a continuous x axis (`continuous`)
    /// or a categorical one, as a 700x200 chart lays them out: every label
    /// drawn flat. Returns the room measured on that `auto=false` layout and
    /// the layout itself, with the index of its x axis in `axis_layouts`.
    fn eleven_x_ticks(continuous: bool) -> (AutoPlotRoom, PlotLayout, usize) {
        let mut axes = axes_whose_left_rescue_rotates_the_labels();
        axes.x.tick_labels = (0..=10)
            .map(|i| match i {
                0 => "0".to_string(),
                10 => "1".to_string(),
                i => format!("0.{i}"),
            })
            .collect();
        if continuous {
            if let Some(p) = axes.x.tick_projection.as_mut() {
                p.major = (0..=10).map(|i| f64::from(i) / 10.0).collect();
            }
        } else {
            axes.x.tick_projection = None;
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let viewport = Viewport {
            width: 700.0,
            height: 200.0,
        };
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let plot = lay_out_at(base, &inputs).plot;
        let x = plot
            .axis_layouts
            .iter()
            .position(|a| a.orient == AxisOrient::Bottom)
            .expect("x axis present");
        let ticks = &plot.axis_layouts[x].ticks;
        assert!(
            ticks.len() == 11 && ticks.iter().all(|t| !t.culled && !t.elided),
            "fixture assumption: eleven x labels, all drawn",
        );
        assert_eq!(room.x_label_losses, vec![0], "fixture assumption: auto=false loses no x label");
        (room, plot, x)
    }

    /// `plot` with its x axis (`axis_layouts[x]`) labelling only the ticks
    /// `drawn` keeps, the rest culled.
    fn labelling(plot: &PlotLayout, x: usize, drawn: impl Fn(usize) -> bool) -> PlotLayout {
        let mut plot = plot.clone();
        for (i, t) in plot.axis_layouts[x].ticks.iter_mut().enumerate() {
            t.culled = !drawn(i);
        }
        plot
    }

    /// A pass is judged by how many of its own ticks it labels, not which:
    /// where `auto=false` labels every 2nd tick (`0, 0.2, …, 1`), a
    /// continuous x axis that labels every 3rd (`0, 0.3, 0.6, 0.9`), 4th or
    /// 5th, none elided and at least three labels drawn, is a sparser,
    /// fully labelled axis and the pass is usable, as is the same stride.
    /// Every 10th (two labels), every other tick but one, an elided label
    /// and tick marks only are losses.
    #[test]
    fn an_evenly_thinned_continuous_x_axis_is_not_a_loss() {
        let (mut room, plot, x) = eleven_x_ticks(true);
        let auto_off = labelling(&plot, x, |i| i % 2 == 0);
        room.x_label_losses = x_label_losses(&auto_off);
        assert_eq!(room.x_label_losses, vec![5], "auto=false labels every 2nd tick");
        for stride in [2, 3, 4, 5] {
            let pass = labelling(&plot, x, |i| i % stride == 0);
            assert!(!room.culls_x_labels(&pass), "every {stride}th tick labelled is usable");
        }
        let two_labels = labelling(&plot, x, |i| i % 10 == 0);
        assert!(room.culls_x_labels(&two_labels), "two labels (every 10th tick) is a loss");
        let irregular = labelling(&plot, x, |i| i % 3 == 0 && i != 6);
        assert!(room.culls_x_labels(&irregular), "a stride with one more label culled is a loss");
        let offset = labelling(&plot, x, |i| i % 3 == 1);
        assert!(room.culls_x_labels(&offset), "a stride that does not start at the first tick is a loss");
        let mut elided = labelling(&plot, x, |i| i % 3 == 0);
        elided.axis_layouts[x].ticks[3].elided = true;
        assert!(room.culls_x_labels(&elided), "an elided label is a loss");
        let marks_only = labelling(&plot, x, |_| false);
        assert!(room.culls_x_labels(&marks_only), "tick marks only is a loss");
    }

    /// Where `auto=false` labels every tick, a pass that culls one of its own
    /// ticks, or elides one label to `"…"` while drawing every tick, loses
    /// it and is unusable; the same layout is usable. An even stride
    /// thinning of a continuous axis is still not a loss.
    #[test]
    fn an_x_axis_that_culls_or_elides_its_own_tick_is_a_loss() {
        let (room, plot, x) = eleven_x_ticks(true);
        assert!(!room.culls_x_labels(&plot), "the auto=false layout itself is usable");
        let culled = labelling(&plot, x, |i| i != 4);
        assert!(room.culls_x_labels(&culled), "one tick culled is a loss");
        let mut elided = plot.clone();
        elided.axis_layouts[x].ticks[1].elided = true;
        assert!(room.culls_x_labels(&elided), "one label elided is a loss");
        let thinned = labelling(&plot, x, |i| i % 2 == 0);
        assert!(!room.culls_x_labels(&thinned), "every 2nd tick labelled is a sparser, fully labelled axis");
    }

    /// The stride is read over the generated ticks, those with label text:
    /// a blank-label tick (never a loss) between them does not break an even
    /// thinning. Where `auto=false` labels every generated tick, a pass that
    /// labels every 2nd generated tick around a blank one is usable.
    #[test]
    fn an_even_thinning_skips_blank_label_ticks() {
        let (room, plot, x) = eleven_x_ticks(true);
        let mut pass = plot.clone();
        let ticks = &mut pass.axis_layouts[x].ticks;
        ticks[1].label = String::new();
        let generated: Vec<usize> = (0..ticks.len()).filter(|&i| i != 1).collect();
        for (k, &i) in generated.iter().enumerate() {
            ticks[i].culled = k % 2 != 0;
        }
        assert_eq!(lost_x_tick_labels(&pass.axis_layouts[x]), 5, "fixture assumption: five generated ticks lose their label");
        assert!(!room.culls_x_labels(&pass), "every 2nd generated tick labelled is an even thinning");
    }

    /// A categorical x axis is judged by count alone: where `auto=false`
    /// labels every 2nd category, a pass at the same stride loses as many
    /// and is usable, and one at the next stride loses more and is not.
    #[test]
    fn a_categorical_x_axis_may_not_thin_further_than_auto_off() {
        let (mut room, plot, x) = eleven_x_ticks(false);
        assert!(!room.x_continuous, "fixture assumption: the x axis is categorical");
        room.x_label_losses = x_label_losses(&labelling(&plot, x, |i| i % 2 == 0));
        assert!(!room.culls_x_labels(&labelling(&plot, x, |i| i % 2 == 0)), "the same stride is usable");
        assert!(room.culls_x_labels(&labelling(&plot, x, |i| i % 3 == 0)), "the next stride is a loss");
    }

    /// A culled blank-label tick is not a lost label ([`lost_x_tick_labels`]
    /// filters `!t.label.is_empty()` before counting a cull or elide): on a
    /// categorical axis (no even-thinning exception), a pass that culls only
    /// a blank-label tick, drawing every one of its own generated labels,
    /// draws as many as `auto=false` and is usable; the same pass with one
    /// more, real tick also culled loses one real label and is not.
    #[test]
    fn a_culled_blank_label_tick_on_a_categorical_axis_is_still_usable() {
        let (mut room, plot, x) = eleven_x_ticks(false);
        assert!(!room.x_continuous, "fixture assumption: the x axis is categorical");
        let mut blanked = plot.clone();
        blanked.axis_layouts[x].ticks[1].label = String::new();
        room.x_label_losses = x_label_losses(&blanked);
        assert_eq!(
            room.x_label_losses,
            vec![0],
            "fixture assumption: auto=false draws every generated (non-blank) tick"
        );
        let blank_only_culled = labelling(&blanked, x, |i| i != 1);
        assert_eq!(
            lost_x_tick_labels(&blank_only_culled.axis_layouts[x]),
            0,
            "the culled blank tick is not a lost label"
        );
        assert!(
            !room.culls_x_labels(&blank_only_culled),
            "culling only the blank tick draws every generated label"
        );
        let blank_and_real_culled = labelling(&blanked, x, |i| i != 1 && i != 3);
        assert_eq!(
            lost_x_tick_labels(&blank_and_real_culled.axis_layouts[x]),
            1,
            "the one real tick culled alongside the blank one is a lost label"
        );
        assert!(
            room.culls_x_labels(&blank_and_real_culled),
            "culling a real tick too is a loss"
        );
    }

    /// A pass that lays out a different number of x axes than `auto=false`
    /// is unusable. An x axis that draws no labels (`show_labels == false`)
    /// loses none, however its ticks are flagged.
    #[test]
    fn x_axis_count_and_hidden_labels() {
        let (mut room, plot, x) = eleven_x_ticks(true);
        let mut no_x_axis = plot.clone();
        no_x_axis.axis_layouts.remove(x);
        assert!(room.culls_x_labels(&no_x_axis), "a pass with no x axis is a loss");
        let mut two_x_axes = plot.clone();
        two_x_axes.axis_layouts.push(plot.axis_layouts[x].clone());
        assert!(room.culls_x_labels(&two_x_axes), "a pass with an extra x axis is a loss");
        let mut hidden = labelling(&plot, x, |_| false);
        hidden.axis_layouts[x].show_labels = false;
        hidden.axis_layouts[x].ticks[2].elided = true;
        assert_eq!(x_label_losses(&hidden), vec![0], "an axis that draws no labels loses none");
        room.x_continuous = false;
        assert!(!room.culls_x_labels(&hidden), "so it is usable on any axis");
    }

    /// An x title rescue whose narrower plot labels every 3rd of the x
    /// axis's eleven ticks where `auto=false` labels every 2nd ships: on a
    /// 160x200 chart the 20-character title clips at `auto=false`, and the
    /// default draws it inside the canvas, labels `0, 0.3, 0.6, 0.9` (none
    /// elided), and keeps more than half the plot width.
    #[test]
    fn a_title_rescue_that_thins_the_x_labels_evenly_ships() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let mut axes = axes_whose_left_rescue_rotates_the_labels();
        axes.x.tick_labels = (0..=10)
            .map(|i| match i {
                0 => "0".to_string(),
                10 => "1".to_string(),
                i => format!("0.{i}"),
            })
            .collect();
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = (0..=10).map(|i| f64::from(i) / 10.0).collect();
        }
        axes.x.title.set(Some("a".repeat(20)));
        let viewport = Viewport {
            width: 160.0,
            height: 200.0,
        };
        let x_axis = |l: &LayoutResult| l.axes.iter().find(|a| a.orient == AxisOrient::Bottom).expect("x axis present").clone();
        let title_ink = |l: &LayoutResult| {
            let title = x_axis(l).title.expect("x title");
            let half = m.measure_width(&title.text, theme.typography.title_font_size) / 2.0;
            (title.anchor_x - half, title.anchor_x + half)
        };
        let labelled = |l: &LayoutResult| x_axis(l).ticks.iter().map(|t| !t.culled).collect::<Vec<_>>();
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert_eq!(labelled(&off), axis::stride_visibility(11, 2), "fixture assumption: auto=false labels every 2nd tick");
        assert!(title_ink(&off).1 > viewport.width, "fixture assumption: the title clips at auto=false");
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        let (lo, hi) = title_ink(&on);
        assert!(lo >= -1e-6 && hi <= viewport.width + 1e-6, "the title is rescued inside the canvas ({lo}, {hi})");
        assert_eq!(labelled(&on), axis::stride_visibility(11, 3), "the rescue labels every 3rd tick");
        let drawn: Vec<String> = x_axis(&on).ticks.iter().filter(|t| !t.culled).map(|t| t.label.clone()).collect();
        assert_eq!(drawn, ["0", "0.3", "0.6", "0.9"], "every drawn label is drawn as generated");
        assert!(x_axis(&on).ticks.iter().all(|t| !t.elided), "no label is elided");
        assert!(
            on.panels[0].plot_area.w >= off.panels[0].plot_area.w / 2.0,
            "the rescue keeps half the plot ({} of {})",
            on.panels[0].plot_area.w,
            off.panels[0].plot_area.w,
        );
    }

    /// The x label band budget is unbounded when no band bounds the panels:
    /// when even no band leaves them undrawable, as when the band they
    /// would draw leaves them drawable.
    #[test]
    fn the_x_label_band_budget_is_unbounded_when_no_band_is_drawable() {
        assert_eq!(x_label_band_budget(50.0, &|_| false), f64::INFINITY);
        assert_eq!(x_label_band_budget(50.0, &|_| true), f64::INFINITY);
    }

    /// Between a band that leaves the panels drawable (none) and the one the
    /// labels would draw, which does not, the budget is the tallest drawable
    /// band, found by bisection.
    #[test]
    fn the_x_label_band_budget_bisects_to_the_tallest_drawable_band() {
        let budget = x_label_band_budget(50.0, &|b| b <= 17.25);
        assert!(budget <= 17.25 && 17.25 - budget < 1e-9, "budget {budget}");
    }

    /// A rescue that thins a continuous x axis to its two end labels is
    /// declined. On a 300x100 chart `auto=false` draws all three `",.4f"`
    /// labels (`0.0000`, `617,283.9450`, `1,234,567.8900`) and the last one
    /// overhangs the right edge. The 33px right extra that contains it keeps
    /// the plot above its floors but narrows it until only every 2nd tick is
    /// labelled: an even thinning, but to two labels, fewer than
    /// [`MIN_EVENLY_THINNED_X_LABELS`], and it loses a tick `auto=false`
    /// draws. So the chart ships as the `auto=false` layout.
    #[test]
    fn a_rescue_that_thins_the_x_labels_to_two_is_declined() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["0.0000".into(), "617,283.9450".into(), "1,234,567.8900".into()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.5, 1.0];
        }
        let m = MockMetrics {
            measure: fixed_width(7.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 100.0,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let labelled = |plot: &PlotLayout| {
            x_axis_layouts(plot)
                .next()
                .map(|a| a.ticks.iter().map(|t| !t.culled && !t.elided).collect::<Vec<_>>())
        };
        let auto_off = lay_out_at(base, &inputs).plot;
        assert_eq!(labelled(&auto_off), Some(vec![true; 3]), "fixture assumption: auto=false draws all three labels");
        let probe = probe_auto_padding(base, &inputs, &room);
        let rescue = auto_padding_extra(&inputs, probe.chrome, probe.x_overhang, &room).total;
        assert!(rescue.right > 0.0, "fixture assumption: the last label needs a right rescue ({rescue:?})");
        let candidate = lay_out_at(resolve_padding_inset(&theme, base, rescue), &inputs).plot;
        let area = candidate.panels[0].plot_area;
        let (w_floor, h_floor) = room.floor(0);
        assert!(area.w >= w_floor && area.h >= h_floor, "fixture assumption: the rescue keeps both floors ({area:?})");
        assert_eq!(
            labelled(&candidate),
            Some(vec![true, false, true]),
            "fixture assumption: the rescue labels the two ends only",
        );
        assert!(room.culls_x_labels(&candidate), "two labels is a loss");
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            layout_with_auto(&theme, false, viewport, &axes, &m),
            "the chart ships as auto=false",
        );
    }

    /// The charts of [`a_failing_rescue_on_one_axis_keeps_the_other`]: an
    /// x axis whose long first label (`first` nines) overhangs the left edge
    /// among `labels` ticks, and a y title of `title` characters, on a
    /// `width` x `height` chart.
    fn two_rescue_axes(first: usize, labels: usize, title: usize) -> AxesInput {
        let mut axes = axes_whose_left_rescue_rotates_the_labels();
        let mut texts: Vec<String> = (0..labels).map(|i| format!("{}", i * 5)).collect();
        texts[0] = format!("-{}", "9".repeat(first));
        axes.x.tick_labels = texts;
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = (0..labels).map(|i| i as f64 / (labels - 1) as f64).collect();
        }
        axes.y.title.set(Some("a".repeat(title)));
        axes
    }

    /// Where no pass of the whole chart is usable, each axis is judged on
    /// its own, x and y alike. Where the x rescue
    /// alone collapses the plot (rotating the labels into a band that leaves
    /// under half the height) and the y title's rescue alone is valid, x is
    /// closed and the y rescue ships: the x margins are `auto=false`'s and
    /// the y margins those of the same chart with explicit x sides. Where the
    /// y rescue alone culls an x label (a shorter plot whose band budget
    /// culls) and the x rescue alone is valid, y is closed and the x rescue
    /// ships: the y margins are `auto=false`'s and the x margins those of the
    /// same chart with explicit y sides. The sweep must reach both cases.
    #[test]
    fn a_failing_rescue_on_one_axis_keeps_the_other() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let (mut x_fails, mut y_fails) = (0, 0);
        for (first, width, title, height, labels) in [6, 10]
            .into_iter()
            .flat_map(|f| [240, 320, 400].map(move |w| (f, w)))
            .flat_map(|(f, w)| (4..=16).step_by(4).map(move |t| (f, w, t)))
            .flat_map(|(f, w, t)| (110..=150).step_by(10).map(move |h| (f, w, t, h)))
            .flat_map(|(f, w, t, h)| [6usize, 10].map(move |n| (f, w, t, h, n)))
        {
            let axes = two_rescue_axes(first + 2, labels, title);
            let theme = default_theme_inputs();
            let viewport = Viewport {
                width: f64::from(width),
                height: f64::from(height),
            };
            let base = base_padding_insets(&theme);
            let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
            let room = AutoPlotRoom::of_auto_off(&inputs, base);
            let probe = probe_auto_padding(base, &inputs, &room);
            let need = auto_padding_extra(&inputs, probe.chrome, probe.x_overhang, &room).total;
            let x_only = Inset { top: 0.0, bottom: 0.0, ..need };
            let y_only = Inset { left: 0.0, right: 0.0, ..need };
            if x_only == Inset::default() || y_only == Inset::default() {
                continue;
            }
            let at = |extra: Inset| resolve_padding_inset(&theme, base, extra);
            let x_unusable = probe_auto_padding(at(x_only), &inputs, &room).unusable;
            let y_unusable = probe_auto_padding(at(y_only), &inputs, &room).unusable;
            let x_culls = room.culls_x_labels(&lay_out_at(at(x_only), &inputs).plot);
            let y_culls = room.culls_x_labels(&lay_out_at(at(y_only), &inputs).plot);
            if !settle_within(&inputs, base, &room).no_usable_pass {
                continue;
            }
            let off = layout_with_auto(&theme, false, viewport, &axes, &m);
            let on = layout_with_auto(&theme, true, viewport, &axes, &m);
            if x_unusable && !x_culls && !y_unusable {
                x_fails += 1;
                let mut explicit_x = theme.clone();
                explicit_x.padding.padding_left = Some(base.left);
                explicit_x.padding.padding_right = Some(base.right);
                let want = layout_with_auto(&explicit_x, true, viewport, &axes, &m);
                let case = format!("x collapses: {first}-char first label, {labels} labels, {title}-char y title, {width}x{height}");
                assert_eq!(x_margins(&on), x_margins(&off), "{case}: x adds nothing");
                assert_eq!(y_margins(&on), y_margins(&want), "{case}: the y rescue ships");
            }
            if y_unusable && y_culls && !x_unusable {
                y_fails += 1;
                let mut explicit_y = theme.clone();
                explicit_y.padding.padding_top = Some(base.top);
                explicit_y.padding.padding_bottom = Some(base.bottom);
                let want = layout_with_auto(&explicit_y, true, viewport, &axes, &m);
                let case = format!("y culls: {first}-char first label, {labels} labels, {title}-char y title, {width}x{height}");
                assert_eq!(y_margins(&on), y_margins(&off), "{case}: y adds nothing");
                assert_eq!(x_margins(&on), x_margins(&want), "{case}: the x rescue ships");
            }
        }
        assert!(x_fails > 0 && y_fails > 0, "the sweep must reach both cases (x fails {x_fails}, y fails {y_fails})");
    }

    /// When the x and the y rescue are each valid alone but together leave
    /// the plot unusable, neither axis is closed and the chart ships as
    /// `auto=false`. On a 240x140 chart with an 8-nine first label among 6
    /// x labels and an 8-character y title, the x rescue alone ships 7.37px
    /// on the left and the y rescue alone 7.93px on top, but no settle with
    /// both open has a usable pass.
    #[test]
    fn rescues_valid_alone_but_failing_together_ship_auto_off() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let axes = two_rescue_axes(8, 6, 8);
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 240.0,
            height: 140.0,
        };
        let base = base_padding_insets(&theme);
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        assert!(settle_within(&inputs, base, &room).no_usable_pass, "fixture assumption: no usable pass with both open");
        let x_alone = settle_within(&inputs, base, &room.closing(false, true)).total;
        let y_alone = settle_within(&inputs, base, &room.closing(true, false)).total;
        assert!(
            (x_alone.left - 7.373419818451708).abs() < 1e-9 && x_alone.right == 0.0,
            "fixture assumption: the x rescue alone is valid ({x_alone:?})",
        );
        assert!(
            (y_alone.top - 7.931535329954585).abs() < 1e-9 && y_alone.bottom == 0.0,
            "fixture assumption: the y rescue alone is valid ({y_alone:?})",
        );
        assert_eq!(settle_auto_padding(&inputs, base), Inset::default());
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            layout_with_auto(&theme, false, viewport, &axes, &m),
            "the chart ships as auto=false",
        );
    }

    /// The same rule for an x axis drawn on top: the rescue that would cull
    /// the middle label is declined, every label `auto=false` draws is
    /// drawn, and the chart is the `auto=false` layout.
    #[test]
    fn a_top_axis_rescue_that_would_cull_a_label_is_declined() {
        let mut axes = axes_whose_left_rescue_rotates_the_labels();
        axes.x.orient = AxisOrient::Top;
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 360.0,
            height: 110.0,
        };
        let top_ticks = |axes_out: &[AxisLayout]| {
            axes_out
                .iter()
                .filter(|a| a.orient == AxisOrient::Top)
                .flat_map(|a| a.ticks.iter().map(|t| (t.label.clone(), t.culled)))
                .collect::<Vec<_>>()
        };
        let base = base_padding_insets(&theme);
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let culling = lay_out_at(Inset { left: base.left + 18.0, ..base }, &inputs).plot;
        assert!(
            top_ticks(&culling.axis_layouts).iter().any(|(_, culled)| *culled),
            "fixture assumption: the 18px rescue culls a label",
        );
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert!(
            top_ticks(&off.axes).iter().all(|(_, culled)| !culled),
            "fixture assumption: auto=false draws every label",
        );
        assert!(
            auto_padding_extra(&inputs, Inset::default(), probe_auto_padding(base, &inputs, &AutoPlotRoom::of_auto_off(&inputs, base)).x_overhang, &AutoPlotRoom::of_auto_off(&inputs, base)).total.left > 0.0,
            "fixture assumption: auto wants a left rescue",
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert_eq!(top_ticks(&on.axes), top_ticks(&off.axes), "every label auto=false draws is drawn");
        assert_eq!(on, off, "the declined rescue leaves the auto=false layout");
    }

    /// A 388x300 chart with a right legend and 2px padding, the default
    /// cascade, and an 80px first x label, on which `padding.auto` has no
    /// fixed point and no pass covers.
    fn no_cover_fallback_that_ships() -> (ChartSpec, AxesInput, ThemeInputs, Viewport, Vec<LegendEntry>) {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["9".repeat(8), "5".into(), "0".into()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.0;
            p.major = vec![0.0, 0.5, 1.0];
        }
        axes.y.tick_labels = vec!["0".into(), "5".into()];
        let mut theme = default_theme_inputs();
        theme.padding.padding = 2.0;
        theme.legend.legend_orient = LegendOrient::Right;
        let viewport = Viewport {
            width: 388.0,
            height: 300.0,
        };
        let entries = vec![
            LegendEntry::new("alpha", SymbolKind::Circle),
            LegendEntry::new("beta-long-entry", SymbolKind::Circle),
        ];
        (minimal_chart_spec(), axes, theme, viewport, entries)
    }

    /// A run where no pass covers but the no-cover fallback leaves a
    /// usable plot: it ships the per-side maximum of every pass's need,
    /// not zero and not the last pass. With the labels at -30 (End
    /// anchored, so `cos30°·w + sin30°·line_h` left of the tick and
    /// `sin30°·line_h` right of it), the left need is the 80px first
    /// label's reach less the 2px padding and the 16px y gutter, and the
    /// right need the `"0"` label's `13.2·sin30°` reach less the 2px
    /// padding (see the next test for why the legend gutter does not hold
    /// it). The plot must survive, and the result must not be a fixed
    /// point: re-solving at its own measurement gives another extra.
    #[test]
    fn settle_auto_padding_ships_a_non_collapsing_no_cover_fallback() {
        let (spec, axes, theme, viewport, entries) = no_cover_fallback_that_ships();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = LayoutInputs {
            legend_entries: &entries,
            ..plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport)
        };
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let (sin, cos) = 30f64.to_radians().sin_cos();
        let line_h = 1.2 * theme.typography.label_font_size;
        let want_left = 80.0 * cos + line_h * sin - 2.0 - 16.0;
        let want_right = line_h * sin - 2.0;
        let extra = settle_auto_padding(&inputs, base);
        assert!(
            (extra.left - want_left).abs() < 1e-9
                && (extra.right - want_right).abs() < 1e-9
                && extra.top == 0.0
                && extra.bottom == 0.0,
            "the per-side maximum need must ship: {extra:?}, want left {want_left}, right {want_right}",
        );
        let shipped = probe_auto_padding(resolve_padding_inset(&theme, base, extra), &inputs, &room);
        assert!(!shipped.unusable, "the shipped fallback leaves a plot to draw in");
        let re_solved = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), shipped.chrome, shipped.x_overhang, &room)
        .total;
        assert_ne!(re_solved, extra, "fixture assumption: the fallback is not a fixed point");
    }

    /// The no-cover fallback solves each pass's need against the SMALLER of
    /// the chrome that pass credited and the chrome it measured, never
    /// crediting a reservation one of them did not see. Pass 1 credits no
    /// chrome (nothing is measured yet) and measures the right legend's
    /// gutter: against that gutter the `"0"` label needs nothing on the
    /// right, against the smaller (empty) credit it needs its reach past
    /// the 2px padding, and that is the right extra the fallback ships.
    #[test]
    fn no_cover_fallback_solves_each_need_against_the_smaller_credit() {
        let (spec, axes, theme, viewport, entries) = no_cover_fallback_that_ships();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = LayoutInputs {
            legend_entries: &entries,
            ..plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport)
        };
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let pass_one = probe_auto_padding(base, &inputs, &room);
        let right_need = |credit: Inset| {
            auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), credit, pass_one.x_overhang, &room)
                .total
                .right
        };
        assert!(
            pass_one.chrome.right > 0.0 && right_need(pass_one.chrome) == 0.0,
            "fixture assumption: pass 1's measured legend gutter alone holds the label ({pass_one:?})",
        );
        let want = right_need(per_side_min(Inset::default(), pass_one.chrome));
        assert!(want > 0.0, "fixture assumption: with no credit the label needs a right extra");
        let shipped = settle_auto_padding(&inputs, base).right;
        assert!(
            (shipped - want).abs() < 1e-9,
            "the fallback's right extra ({shipped}) must be pass 1's need against the smaller credit ({want})",
        );
    }

    // ── padding.auto keeps half the auto=false plot ─────────────────────────

    /// The room is measured against the `auto=false` plot: every floor is
    /// half that panel's `auto=false` size, and spending an axis's whole
    /// budget on its low margin leaves the panel exactly at its floor. On a
    /// plain chart a pixel of margin is a pixel of plot, so each budget is
    /// half the plot; split into two facet columns, a pixel of margin
    /// narrows each panel by half a pixel, so the width budget is a whole
    /// panel width.
    #[test]
    fn auto_plot_room_is_half_the_auto_off_plot() {
        let axes = continuous_x_axes_with_long_last_label();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let base = base_padding_insets(&theme);
        let legend_overrides = legend::LegendOverrides::default();
        let plain_spec = minimal_chart_spec();
        let faceted = faceted_spec(2);
        let groups: Vec<FacetGroup> = three_groups().into_iter().take(2).collect();
        let plain = plain_layout_inputs(&plain_spec, &legend_overrides, &theme, &axes, &m, viewport);
        let two_columns = LayoutInputs {
            spec: &faceted,
            facet_groups: &groups,
            ..plain
        };
        for (inputs, panels_per_row) in [(plain, 1.0), (two_columns, 2.0)] {
            let off = lay_out_at(base, &inputs).plot;
            let room = AutoPlotRoom::of_auto_off(&inputs, base);
            assert_eq!(room.floors.len(), off.panels.len());
            for (panel, floor) in off.panels.iter().zip(&room.floors) {
                assert_eq!(*floor, (panel.plot_area.w / 2.0, panel.plot_area.h / 2.0));
            }
            let w0 = off.panels[0].plot_area.w;
            assert!(
                (room.width_budget - panels_per_row * w0 / 2.0).abs() < 1e-9,
                "width budget {} for {panels_per_row} panel(s) per row of width {w0}",
                room.width_budget,
            );
            let spent = lay_out_at(
                Inset {
                    left: base.left + room.width_budget,
                    ..base
                },
                &inputs,
            )
            .plot;
            for (panel, floor) in spent.panels.iter().zip(&room.floors) {
                assert!(
                    (panel.plot_area.w - floor.0).abs() < 1e-9,
                    "spending the width budget leaves the floor: {} vs {}",
                    panel.plot_area.w,
                    floor.0,
                );
            }
            let spent = lay_out_at(
                Inset {
                    top: base.top + room.height_budget,
                    ..base
                },
                &inputs,
            )
            .plot;
            for (panel, floor) in spent.panels.iter().zip(&room.floors) {
                assert!(
                    (panel.plot_area.h - floor.1).abs() < 1e-9,
                    "spending the height budget leaves the floor: {} vs {}",
                    panel.plot_area.h,
                    floor.1,
                );
            }
        }
    }

    /// A 220px chart with a wide y gutter (ten-digit y labels) and a
    /// categorical x axis, so the x title is the only thing auto can pad
    /// for, titled with `n` characters.
    fn narrow_chart_with_an_x_title(n: usize) -> AxesInput {
        let mut axes = axes_with_wide_x_title();
        axes.x.title.set(Some("a".repeat(n)));
        axes.y.tick_labels = vec!["0".into(), "1000000000".into()];
        axes
    }

    /// An x title whose rescue would take more than half the `auto=false`
    /// plot width is not rescued: auto adds nothing for it and the chart is
    /// the `auto=false` layout, its title clipped. One whose rescue keeps
    /// half the plot is rescued, ending inside the canvas. Every title
    /// length is derived by sweeping, and the given-up arm includes titles
    /// narrower than the viewport, which the viewport alone would let auto
    /// rescue by shrinking the plot to a sliver.
    #[test]
    fn padding_auto_gives_up_on_an_x_title_whose_rescue_takes_over_half_the_plot() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 220.0,
            height: 200.0,
        };
        let base = base_padding_insets(&theme);
        let (mut rescued, mut given_up) = (0, 0);
        for n in 1..=40 {
            let axes = narrow_chart_with_an_x_title(n);
            let off = layout_with_auto(&theme, false, viewport, &axes, &m);
            let on = layout_with_auto(&theme, true, viewport, &axes, &m);
            let rescue = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());
            let need = rescue.left + rescue.right;
            let half = off.panels[0].plot_area.w / 2.0;
            if need == 0.0 {
                assert_eq!(on, off, "a {n}-char title that fits needs nothing");
            } else if need <= half {
                rescued += 1;
                let title = on
                    .axes
                    .iter()
                    .find(|a| a.orient == AxisOrient::Bottom)
                    .and_then(|a| a.title.as_ref())
                    .expect("x title");
                let half_w = m.measure_width(&title.text, theme.typography.title_font_size) / 2.0;
                assert!(
                    title.anchor_x - half_w >= -1e-6 && title.anchor_x + half_w <= viewport.width + 1e-6,
                    "a {n}-char title whose rescue keeps half the plot must end inside the canvas",
                );
                assert!(on.panels[0].plot_area.w >= half - 1e-6, "the rescue keeps half the plot");
            } else {
                given_up += 1;
                assert_eq!(on, off, "a {n}-char title whose rescue takes {need}px of a {}px plot", 2.0 * half);
            }
        }
        let viewport_feasible_given_up = (1..=40)
            .filter(|&n| {
                let axes = narrow_chart_with_an_x_title(n);
                let rescue = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());
                let off = layout_with_auto(&theme, false, viewport, &axes, &m);
                rescue.left + rescue.right > off.panels[0].plot_area.w / 2.0
                    && rescue.left + rescue.right < off.panels[0].plot_area.w - MIN_PANEL_DIM
            })
            .count();
        assert!(
            rescued > 0 && given_up > 0 && viewport_feasible_given_up > 0,
            "both arms, and a give-up the viewport alone would have allowed \
             (rescued {rescued}, given up {given_up}, viewport-feasible {viewport_feasible_given_up})",
        );
    }

    /// The vertical twin: a y title whose rescue would take more than half
    /// the `auto=false` plot height is not rescued, and one whose rescue
    /// keeps half the height is. The 54px x label band on top pushes the
    /// plot's centre down, so a long enough y title overflows the bottom;
    /// the y axis is categorical, so the title is the only thing auto pads
    /// for vertically.
    #[test]
    fn padding_auto_gives_up_on_a_y_title_whose_rescue_takes_over_half_the_plot() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let viewport = Viewport {
            width: 300.0,
            height: 120.0,
        };
        let (mut rescued, mut given_up) = (0, 0);
        for n in 1..=25 {
            let (mut axes, theme, _) = vertical_padding_fixture(&"a".repeat(n));
            axes.y.tick_projection = None;
            let base = base_padding_insets(&theme);
            let off = layout_with_auto(&theme, false, viewport, &axes, &m);
            let on = layout_with_auto(&theme, true, viewport, &axes, &m);
            let rescue = auto_padding_for_axis_titles(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), base, Inset::default());
            let need = rescue.top + rescue.bottom;
            let half = off.panels[0].plot_area.h / 2.0;
            if need == 0.0 {
                assert_eq!(on, off, "a {n}-char y title that fits needs nothing");
            } else if need <= half {
                rescued += 1;
                let title = on
                    .axes
                    .iter()
                    .find(|a| a.orient == AxisOrient::Left)
                    .and_then(|a| a.title.as_ref())
                    .expect("y title");
                let half_h = m.measure_width(&title.text, theme.typography.title_font_size) / 2.0;
                assert!(
                    title.anchor_y - half_h >= -1e-6 && title.anchor_y + half_h <= viewport.height + 1e-6,
                    "a {n}-char y title whose rescue keeps half the plot must end inside the canvas",
                );
                assert!(on.panels[0].plot_area.h >= half - 1e-6, "the rescue keeps half the plot");
            } else {
                given_up += 1;
                assert_eq!(on, off, "a {n}-char y title whose rescue takes {need}px of a {}px plot", 2.0 * half);
            }
        }
        assert!(rescued > 0 && given_up > 0, "both arms (rescued {rescued}, given up {given_up})");
    }

    /// One edge label whose containing margin would take more than half
    /// the `auto=false` plot width charges nothing on that side, and the
    /// chart is the `auto=false` layout; a shorter one is rescued to end on
    /// the canvas edge. The flat (`label_overlap=True`) last label sits on
    /// the plot's right edge, so its need is its half-width less the 16px
    /// padding, and the label length is swept across the boundary.
    #[test]
    fn padding_auto_gives_up_on_an_edge_label_whose_charge_takes_over_half_the_plot() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 400.0,
        };
        let fs = theme.typography.label_font_size;
        let (mut rescued, mut given_up) = (0, 0);
        for chars in 1..=40 {
            let mut axes = continuous_x_axes_with_long_last_label();
            axes.x.tick_labels = vec!["0".into(), "9".repeat(chars)];
            axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
            let off = layout_with_auto(&theme, false, viewport, &axes, &m);
            let on = layout_with_auto(&theme, true, viewport, &axes, &m);
            let need = m.measure_width(&"9".repeat(chars), fs) / 2.0 - 16.0;
            let half = off.panels[0].plot_area.w / 2.0;
            if need <= 0.0 {
                assert_eq!(on, off, "a {chars}-char label that fits needs nothing");
            } else if need <= half {
                rescued += 1;
                let (_, hi) = drawn_x_label_ink(&on, &m, fs);
                assert!((hi - viewport.width).abs() < 1e-6, "a {chars}-char label is rescued to the edge ({hi})");
            } else {
                given_up += 1;
                assert_eq!(on, off, "a {chars}-char label needing {need}px of a {}px plot", 2.0 * half);
            }
        }
        assert!(rescued > 0 && given_up > 0, "both arms (rescued {rescued}, given up {given_up})");
    }

    /// A label whose charge would take more than half the plot is dropped
    /// on its own side only: the other side's label still gets its rescue.
    /// The flat first label needs more left margin than half the plot (but
    /// less than the whole plot), and the last label a few pixels on the
    /// right; the default must add nothing on the left and contain the last
    /// label exactly on the right. The chart is 800px tall: the x label band
    /// is reserved as if the long label rotated to -90.
    #[test]
    fn an_over_half_charge_leaves_the_other_sides_rescue() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["9".repeat(36), "9".repeat(5)];
        axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 800.0,
        };
        let fs = theme.typography.label_font_size;
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let plot = off.panels[0].plot_area;
        let left_need = m.measure_width(&"9".repeat(36), fs) / 2.0 - plot.x;
        let right_need = m.measure_width(&"9".repeat(5), fs) / 2.0 - 16.0;
        assert!(
            left_need > plot.w / 2.0 && left_need < plot.w - MIN_PANEL_DIM && right_need > 0.0,
            "fixture assumption: the left need ({left_need}) is over half of a {}px plot but fits it, \
             and the right need ({right_need}) is real",
            plot.w,
        );
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert_eq!(on.padded_region.x, off.padded_region.x, "no left extra for an over-half charge");
        let (_, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!((hi - viewport.width).abs() < 1e-9, "the last label is rescued to the edge ({hi})");
    }

    // ── padding.auto never ships a partial rescue on an axis ─────────────────

    /// What `padding.auto` would add at the `auto=false` layout if no axis
    /// budget bound it (`(extra, room)`, `room` the chart's real one): the
    /// rescues each axis needs together, before [`AutoPlotRoom`] judges them.
    fn unbounded_auto_need(inputs: &LayoutInputs) -> (AutoPaddingExtra, AutoPlotRoom) {
        let base = base_padding_insets(inputs.theme);
        let room = AutoPlotRoom::of_auto_off(inputs, base);
        let off = probe_auto_padding(base, inputs, &room);
        let unbounded = AutoPlotRoom {
            width_budget: f64::INFINITY,
            height_budget: f64::INFINITY,
            ..room.clone()
        };
        (auto_padding_extra(inputs, off.chrome, off.x_overhang, &unbounded), room)
    }

    /// The title part of an [`AutoPaddingExtra`]: its total less its ticks.
    fn title_part(extra: &AutoPaddingExtra) -> Inset {
        Inset {
            top: extra.total.top - extra.ticks.top,
            right: extra.total.right - extra.ticks.right,
            bottom: extra.total.bottom - extra.ticks.bottom,
            left: extra.total.left - extra.ticks.left,
        }
    }

    /// The padded region's left edge and width: what the x margins leave.
    fn x_margins(layout: &LayoutResult) -> (f64, f64) {
        (layout.padded_region.x, layout.padded_region.w)
    }

    /// The padded region's top edge and height: what the y margins leave.
    fn y_margins(layout: &LayoutResult) -> (f64, f64) {
        (layout.padded_region.y, layout.padded_region.h)
    }

    /// Flat first and last x labels whose rescues each fit the width budget
    /// but together take more: auto adds nothing on x, so the x margins and
    /// the plot width are `auto=false`'s and both labels clip as they do
    /// there. Label lengths are swept on a 300px chart.
    #[test]
    fn x_label_rescues_over_the_budget_together_add_nothing_on_x() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 800.0,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let mut combined = 0;
        for first in 1..=30 {
            for last in 1..=30 {
                let mut axes = continuous_x_axes_with_long_last_label();
                axes.x.tick_labels = vec!["9".repeat(first), "9".repeat(last)];
                axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
                let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
                let (need, room) = unbounded_auto_need(&inputs);
                let (left, right) = (need.ticks.left, need.ticks.right);
                if left <= 0.0 || right <= 0.0 || left + right <= room.width_budget {
                    continue;
                }
                combined += 1;
                let off = layout_with_auto(&theme, false, viewport, &axes, &m);
                let on = layout_with_auto(&theme, true, viewport, &axes, &m);
                assert_eq!(
                    x_margins(&on),
                    x_margins(&off),
                    "{first}/{last}-char labels needing {left} + {right} of a {} budget",
                    room.width_budget,
                );
                assert_eq!(on.panels[0].plot_area.w, off.panels[0].plot_area.w);
            }
        }
        assert!(combined > 0, "the sweep must reach a combined over-budget pair");
    }

    /// An x edge-label rescue and an x title rescue that each fit the width
    /// budget but together take more: auto adds nothing on x. The x title
    /// and the last label's length are swept on a 300px chart.
    #[test]
    fn an_x_label_and_x_title_rescue_over_the_budget_together_add_nothing_on_x() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 400.0,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let mut combined = 0;
        for title in 1..=40 {
            for last in 1..=20 {
                let mut axes = continuous_x_axes_with_long_last_label();
                axes.x.tick_labels = vec!["0".into(), "9".repeat(last)];
                axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
                axes.x.title.set(Some("a".repeat(title)));
                let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
                let (need, room) = unbounded_auto_need(&inputs);
                let ticks = need.ticks.left + need.ticks.right;
                let alone = auto_padding_for_axis_titles(&inputs, base_padding_insets(&theme), Inset::default());
                let titles = title_part(&need);
                let budget = room.width_budget;
                if ticks <= 0.0
                    || titles.left + titles.right <= 0.0
                    || ticks > budget
                    || alone.left + alone.right > budget
                    || need.total.left + need.total.right <= budget
                {
                    continue;
                }
                combined += 1;
                let off = layout_with_auto(&theme, false, viewport, &axes, &m);
                let on = layout_with_auto(&theme, true, viewport, &axes, &m);
                assert_eq!(
                    x_margins(&on),
                    x_margins(&off),
                    "a {title}-char title and {last}-char label together over a {budget} budget",
                );
            }
        }
        assert!(combined > 0, "the sweep must reach a combined label-and-title case");
    }

    /// The vertical twin: the top half-line the y labels need and a y title
    /// rescue that each fit the height budget but together take more add
    /// nothing on y. The y title and the chart height are swept.
    #[test]
    fn a_y_label_and_y_title_rescue_over_the_budget_together_add_nothing_on_y() {
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let mut combined = 0;
        for title in 1..=30 {
            for height in (80..=240).step_by(8) {
                let (axes, theme, m) = vertical_padding_fixture(&"a".repeat(title));
                let viewport = Viewport {
                    width: 300.0,
                    height: f64::from(height),
                };
                let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
                let (need, room) = unbounded_auto_need(&inputs);
                let ticks = need.ticks.top + need.ticks.bottom;
                let titles = title_part(&need);
                let alone = auto_padding_for_axis_titles(&inputs, base_padding_insets(&theme), Inset::default());
                let budget = room.height_budget;
                if ticks <= 0.0
                    || titles.top + titles.bottom <= 0.0
                    || ticks > budget
                    || alone.top + alone.bottom > budget
                    || need.total.top + need.total.bottom <= budget
                {
                    continue;
                }
                combined += 1;
                let off = layout_with_auto(&theme, false, viewport, &axes, &m);
                let on = layout_with_auto(&theme, true, viewport, &axes, &m);
                assert_eq!(
                    y_margins(&on),
                    y_margins(&off),
                    "a {title}-char y title at {height}px together over a {budget} budget",
                );
            }
        }
        assert!(combined > 0, "the sweep must reach a combined y case");
    }

    /// Closing x lays y out against the plot x actually ships. A 400-500px
    /// wide, 300-540px tall chart whose two long flat x labels reach past
    /// both edges (the theme padding is 2px) and together need more than
    /// the width budget, with a y title auto rescues: its y margins are
    /// those of the same chart with the x sides explicit at their
    /// `auto=false` values. The sweep must include charts where the y
    /// rescue solved alongside the scaled x extra differs (the narrower plot
    /// rotates the x labels into a taller band without culling one), so the
    /// y extra really is re-solved.
    #[test]
    fn closing_x_re_solves_the_y_rescue_at_the_x_that_ships() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let (mut closed, mut re_solved) = (0, 0);
        for (edge, width) in (18..=30).flat_map(|e| [400, 450, 500].map(move |w| (e, w))) {
            for title in [26, 38, 50] {
                for height in (300..=540).step_by(40) {
                    let mut theme = default_theme_inputs();
                    theme.padding.padding_auto = true;
                    theme.padding.padding = 2.0;
                    let mut axes = continuous_x_axes_with_long_last_label();
                    axes.x.tick_labels = vec!["9".repeat(edge), "9".repeat(edge)];
                    axes.y.title.set(Some("a".repeat(title)));
                    let viewport = Viewport {
                        width: f64::from(width),
                        height: f64::from(height),
                    };
                    let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
                    let base = base_padding_insets(&theme);
                    let room = AutoPlotRoom::of_auto_off(&inputs, base);
                    let first = settle_within(&inputs, base, &room);
                    if !first.x_scaled || first.y_scaled {
                        continue;
                    }
                    closed += 1;
                    let off = layout_with_auto(&theme, false, viewport, &axes, &m);
                    let on = layout_with_auto(&theme, true, viewport, &axes, &m);
                    assert_eq!(x_margins(&on), x_margins(&off), "x is closed");
                    let mut explicit_x = theme.clone();
                    explicit_x.padding.padding_left = Some(base.left);
                    explicit_x.padding.padding_right = Some(base.right);
                    let want = layout_with_auto(&explicit_x, true, viewport, &axes, &m);
                    assert_eq!(
                        y_margins(&on),
                        y_margins(&want),
                        "{edge}-char edge labels, {title}-char y title at {height}px",
                    );
                    let shipped_top = on.padded_region.y - off.padded_region.y;
                    let shipped_bottom = (off.padded_region.y + off.padded_region.h)
                        - (on.padded_region.y + on.padded_region.h);
                    if (first.total.top - shipped_top).abs() > 1e-9
                        || (first.total.bottom - shipped_bottom).abs() > 1e-9
                    {
                        re_solved += 1;
                    }
                }
            }
        }
        assert!(closed > 0 && re_solved > 0, "closed {closed}, re-solved {re_solved}");
    }

    /// The first pass credits no chrome, so beside a one-entry right legend
    /// it sees the last label's whole reach past the plot as need: more than the
    /// width budget, and the cap scales it. Once the legend's gutter is
    /// measured and credited, the label needs only what the gutter does not
    /// hold, which fits; the settle converges there unscaled, so that full
    /// rescue ships and x is not closed. The last label's length is swept.
    #[test]
    fn an_under_credited_first_pass_does_not_close_a_rescue_that_fits() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let (spec, _, theme, _, _) = no_cover_fallback_that_ships();
        let entries = vec![LegendEntry::new("legend", SymbolKind::Circle)];
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let legend_overrides = legend::LegendOverrides::default();
        let base = base_padding_insets(&theme);
        let mut under_credited = 0;
        for chars in 1..=40 {
            let mut axes = continuous_x_axes_with_long_last_label();
            axes.x.tick_labels = vec!["0".into(), "9".repeat(chars)];
            axes.x.overrides.labels.overlap = Some(LabelOverlap::ShowAll);
            let inputs = LayoutInputs {
                legend_entries: &entries,
                ..plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport)
            };
            let room = AutoPlotRoom::of_auto_off(&inputs, base);
            let off = probe_auto_padding(base, &inputs, &room);
            let first = auto_padding_extra(&inputs, Inset::default(), off.x_overhang, &room);
            let credited = auto_padding_extra(&inputs, off.chrome, off.x_overhang, &room);
            if !first.x_scaled || credited.x_scaled || credited.total.right <= 0.0 || credited.total.left > 0.0 {
                continue;
            }
            under_credited += 1;
            let shipped = settle_auto_padding(&inputs, base);
            assert_eq!(
                (shipped.left, shipped.right),
                (credited.total.left, credited.total.right),
                "a {chars}-char last label beside the legend ships its full credited rescue",
            );
        }
        assert!(under_credited > 0, "the sweep must reach an under-credited first pass");
    }

    /// The no-cover fallback reports the axes its needs were scaled on, so
    /// it is closed like any settled extra. On the no-cover chart (where no
    /// pass covers and the per-side maximum of the needs ships), a room whose
    /// width budget holds the left need and the right need each but not
    /// both scales those needs, so the settled extra is flagged on x and
    /// auto ships nothing on x.
    #[test]
    fn a_no_cover_fallback_over_the_budget_is_flagged() {
        let (spec, axes, theme, viewport, entries) = no_cover_fallback_that_ships();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = LayoutInputs {
            legend_entries: &entries,
            ..plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport)
        };
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let fallback = settle_within(&inputs, base, &room);
        assert!(!fallback.x_scaled, "fixture assumption: the fallback fits the real room ({fallback:?})");
        let (left, right) = (fallback.total.left, fallback.total.right);
        assert!(left > 0.0 && right > 0.0, "fixture assumption: the fallback needs both sides");
        let tight = AutoPlotRoom {
            width_budget: left.max(right) + (left.min(right) / 2.0),
            ..room.clone()
        };
        let settled = settle_within(&inputs, base, &tight);
        assert!(settled.x_scaled, "needs over the budget together are flagged ({settled:?})");
        let closed = settle_within(&inputs, base, &tight.closing(true, false));
        assert_eq!((closed.total.left, closed.total.right), (0.0, 0.0), "a closed x ships nothing");
    }

    /// The room judges a candidate's whole plot, in both dimensions,
    /// whichever margin moved. On a 360x150 chart the 18px left extra that
    /// contains the long flat first label keeps most of the plot's width,
    /// but the narrower plot no longer fits the labels flat in half its
    /// width, so the cascade rotates them to -30 and the x axis draws (and
    /// reserves) a taller band. That leaves less than half the `auto=false`
    /// plot height (and more than [`MIN_PANEL_DIM`], so this is not a
    /// collapse). The candidate is unusable, so auto adds nothing.
    #[test]
    fn padding_auto_adds_nothing_when_a_rescue_leaves_under_half_the_plot_height() {
        let spec = minimal_chart_spec();
        let axes = axes_whose_left_rescue_rotates_the_labels();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 360.0,
            height: 150.0,
        };
        let legend_overrides = legend::LegendOverrides::default();
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let auto_off = probe_auto_padding(base, &inputs, &room);
        let rescue = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), auto_off.x_overhang, &room)
        .total;
        assert!((rescue.left - 18.0).abs() < 1e-9, "fixture assumption: the label needs 60 - 42 = 18px ({rescue:?})");
        let candidate = lay_out_at(resolve_padding_inset(&theme, base, rescue), &inputs).plot;
        assert_eq!(
            x_presentation(&candidate).map(|p| p[0]),
            Some((-30.0, false)),
            "fixture assumption: the rescue really rotates the labels",
        );
        let area = candidate.panels[0].plot_area;
        let (w_floor, h_floor) = room.floor(0);
        assert!(
            area.w >= w_floor && area.h < h_floor && area.h > MIN_PANEL_DIM,
            "fixture assumption: the rescue keeps the width but not half the height ({area:?}, floor {w_floor} x {h_floor})",
        );
        assert!(probe_auto_padding(resolve_padding_inset(&theme, base, rescue), &inputs, &room).unusable);
        assert_eq!(
            layout_with_auto(&theme, true, viewport, &axes, &m),
            layout_with_auto(&theme, false, viewport, &axes, &m),
            "a rescue that leaves under half the plot height adds nothing",
        );
    }

    /// The width twin: with no budget to scale or decline it, an x title
    /// rescue that would leave under half the `auto=false` plot width (and
    /// keeps the height) is unusable, and the settle adds nothing; the same
    /// room without floors ships it.
    #[test]
    fn a_rescue_that_leaves_under_half_the_plot_width_adds_nothing() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let base = base_padding_insets(&theme);
        let mut reached = 0;
        for (width, n) in (220..=400).step_by(30).flat_map(|w| (1..=50).map(move |n| (w, n))) {
            let viewport = Viewport {
                width: f64::from(width),
                height: 200.0,
            };
            let axes = narrow_chart_with_an_x_title(n);
            let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
            let room = AutoPlotRoom::of_auto_off(&inputs, base);
            let unbudgeted = AutoPlotRoom {
                width_budget: f64::INFINITY,
                height_budget: f64::INFINITY,
                ..room.clone()
            };
            let no_floors = AutoPlotRoom {
                floors: Vec::new(),
                ..unbudgeted.clone()
            };
            let rescue = settle_within(&inputs, base, &no_floors).total;
            if rescue == Inset::default() {
                continue;
            }
            let area = lay_out_at(resolve_padding_inset(&theme, base, rescue), &inputs).plot.panels[0].plot_area;
            let (w_floor, h_floor) = room.floor(0);
            if !(area.w < w_floor && area.w > MIN_PANEL_DIM && area.h >= h_floor) {
                continue;
            }
            reached += 1;
            assert_eq!(
                settle_within(&inputs, base, &unbudgeted).total,
                Inset::default(),
                "a {n}-char title whose rescue leaves {} of a {}px plot adds nothing",
                area.w,
                2.0 * w_floor,
            );
        }
        assert!(reached > 0, "the sweep must reach a rescue under the width floor and above the height floor");
    }

    /// `leaves_usable` holds a panel to its width floor on its own: a plot
    /// whose panel is below its width floor and above its height floor is
    /// unusable, and the same plot with that floor met is usable.
    #[test]
    fn a_panel_under_its_width_floor_alone_is_unusable() {
        let axes = axes_whose_left_rescue_rotates_the_labels();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let viewport = Viewport {
            width: 360.0,
            height: 200.0,
        };
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let base = base_padding_insets(&theme);
        let room = AutoPlotRoom::of_auto_off(&inputs, base);
        let plot = lay_out_at(base, &inputs).plot;
        let area = plot.panels[0].plot_area;
        assert!(room.leaves_usable(&plot), "fixture assumption: the auto=false plot is usable");
        let narrow = AutoPlotRoom {
            floors: vec![(area.w + 1.0, area.h - 1.0)],
            ..room.clone()
        };
        assert!(!narrow.leaves_usable(&plot), "a panel under its width floor is unusable");
        let met = AutoPlotRoom {
            floors: vec![(area.w - 1.0, area.h - 1.0)],
            ..room
        };
        assert!(met.leaves_usable(&plot), "the same panel at its floors is usable");
    }

    // ── padding.auto judges coverage by the tick extra that ships ───────────

    /// An explicit side ships no tick extra whatever its labels do: the
    /// right side is explicit, and its last label is measured in two states
    /// (reaching 40px and 70px past the plot edge, as a cascade that
    /// re-presents the label between passes would measure it). The raw
    /// tick need on the right differs between them, but what ships is 0 in
    /// both, so a pass is never judged short on that side.
    #[test]
    fn an_explicit_side_ships_no_tick_extra_in_any_label_state() {
        let axes = continuous_x_axes_with_long_last_label();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let mut theme = default_theme_inputs();
        theme.padding.padding_right = Some(4.0);
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let mut raw = Vec::new();
        for reach in [40.0, 70.0] {
            let overhang = XLabelOverhang {
                left: f64::NEG_INFINITY,
                right: reach,
            };
            raw.push(auto_padding_for_edge_ticks(&theme, &axes, &m, Inset::default(), overhang).right);
            let shipped = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), overhang, &plot_room(&theme, &axes, &m, viewport));
            assert_eq!(shipped.ticks.right, 0.0, "an explicit side ships no tick extra at reach {reach}");
            assert_eq!(shipped.total.right, 0.0);
        }
        assert!(
            raw[0] > 0.0 && raw[0] != raw[1],
            "fixture assumption: the raw need is real and changes with the label state ({raw:?})",
        );
    }

    /// The axis of an infeasible title ships no tick extra: the 66-char x
    /// title is wider than the 300px viewport, so the x ticks' raw right
    /// need is dropped before it ships.
    #[test]
    fn an_infeasible_titles_axis_ships_no_tick_extra() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.title.set(Some("a".repeat(66)));
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 300.0,
        };
        let overhang = x_overhang_without_auto(&theme, &axes, &m, viewport);
        let raw = auto_padding_for_edge_ticks(&theme, &axes, &m, Inset::default(), overhang);
        assert!(raw.right > 0.0, "fixture assumption: the raw tick need is real ({raw:?})");
        let shipped = auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), overhang, &plot_room(&theme, &axes, &m, viewport));
        assert_eq!((shipped.ticks.left, shipped.ticks.right), (0.0, 0.0));
    }

    /// A side the viewport cap scales ships its tick extra at the capped
    /// size, the same for the same labels: with a plot room that does not
    /// bind (the margins narrow no plot), a 150px chart whose two 190px
    /// labels want more than the viewport leaves ships each side's raw
    /// need times one shared factor, so a steady pass compares equal.
    #[test]
    fn a_capped_side_ships_its_tick_extra_at_the_capped_size() {
        let axes = continuous_x_axes_with_long_first_and_last_labels();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 150.0,
            height: 400.0,
        };
        let base = base_padding_insets(&theme);
        let unbounded = AutoPlotRoom {
            floors: Vec::new(),
            x_label_losses: Vec::new(),
            x_continuous: true,
            width_budget: f64::INFINITY,
            height_budget: f64::INFINITY,
        };
        let overhang = XLabelOverhang {
            left: 120.0,
            right: 110.0,
        };
        let raw = auto_padding_for_edge_ticks(&theme, &axes, &m, Inset::default(), overhang);
        let budget = viewport.width - base.left - base.right - MIN_PANEL_DIM;
        assert!(raw.left + raw.right > budget, "fixture assumption: the cap fires ({raw:?})");
        let scale = budget / (raw.left + raw.right);
        let solve = || {
            auto_padding_extra(&plain_layout_inputs(&minimal_chart_spec(), &legend::LegendOverrides::default(), &theme, &axes, &m, viewport), Inset::default(), overhang, &unbounded)
        };
        let shipped = solve();
        assert!(
            (shipped.ticks.left - raw.left * scale).abs() < 1e-9
                && (shipped.ticks.right - raw.right * scale).abs() < 1e-9,
            "the shipped tick extra is the capped size ({shipped:?}, scale {scale})",
        );
        assert_eq!(shipped.ticks, shipped.total, "with no title the tick extra is all of it");
        assert_eq!(solve(), shipped, "the same labels ship the same capped extra");
    }

    // ── padding.auto pins: what it measures and what it skips ───────────────

    /// The width a [`MockMetrics`] built on this reports grows with the
    /// font size (0.6 of it per character), so a cascade font shrink
    /// changes what a label measures.
    fn font_scaled_width(text: &str, font_size: f64) -> f64 {
        text.chars().count() as f64 * font_size * 0.6
    }

    /// At 330px the cascade draws these five currency labels flat at the
    /// shrunk font size (`FONT_SHRINK_FACTOR` of 11), with and without
    /// auto. The flat last label sits on the plot's right edge, so the
    /// rescue is exactly its SHRUNK half-width past the edge: the right
    /// inset equals that half-width and the label ends on the canvas edge.
    /// Measured at the unshrunk size it would be `9·11·0.6/2 = 29.7`.
    #[test]
    fn padding_auto_sizes_a_shrunk_font_rescue_by_the_drawn_font() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = ["0", "250,000", "500,000", "750,000", "1,000,000"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.0;
            p.major = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        }
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 330.0,
            height: 300.0,
        };
        let fs = theme.typography.label_font_size;
        let shrunk = fs * FONT_SHRINK_FACTOR;
        let presentation = |l: &LayoutResult| {
            l.axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present")
                .ticks
                .iter()
                .map(|t| (t.label_angle, t.label_font_size, t.culled))
                .collect::<Vec<_>>()
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert_eq!(presentation(&off), vec![(0.0, Some(shrunk), false); 5], "fixture assumption");
        assert_eq!(presentation(&on), presentation(&off), "the rescue keeps the presentation");
        assert!(drawn_x_label_ink(&off, &m, fs).1 > viewport.width, "fixture assumption: overhangs");
        let half_w = font_scaled_width("1,000,000", shrunk) / 2.0;
        let right_inset = viewport.width - on.padded_region.x - on.padded_region.w;
        assert!(
            (right_inset - half_w).abs() < 1e-9,
            "right inset {right_inset} must be the shrunk half-width {half_w}",
        );
        assert!((drawn_x_label_ink(&on, &m, fs).1 - viewport.width).abs() < 1e-9);
    }

    /// A culled label is not drawn, so it charges nothing: with
    /// `label_overlap="parity"` the long last label is culled, and the
    /// default is the `auto=false` layout although that label, drawn, would
    /// overhang the right edge by far less than half the plot.
    #[test]
    fn padding_auto_charges_nothing_for_a_culled_label() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["0".into(), "5".into(), "10".into(), "9".repeat(10)];
        axes.x.overrides.labels.overlap = Some(LabelOverlap::Parity);
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.0;
            p.major = vec![0.0, 0.3, 0.6, 1.0];
        }
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 300.0,
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let last = off
            .axes
            .iter()
            .find(|a| a.orient == AxisOrient::Bottom)
            .and_then(|a| a.ticks.last())
            .expect("x ticks");
        let reach = last.position + m.measure_width(&last.label, theme.typography.label_font_size) / 2.0;
        assert!(
            last.culled && reach > viewport.width,
            "fixture assumption: the culled last label would overhang if drawn ({last:?}, reach {reach})",
        );
        assert_eq!(layout_with_auto(&theme, true, viewport, &axes, &m), off);
    }

    /// A categorical x axis is not charged for its labels, even ones that
    /// reach past the canvas: the long first label at an explicit -45 hangs
    /// down and to the left of its band centre, past the left edge, and the
    /// default is still the `auto=false` layout.
    #[test]
    fn padding_auto_charges_nothing_for_a_categorical_axis_label_past_the_edge() {
        let mut axes = continuous_x_axes_at_angle(
            AxisOrient::Bottom,
            vec!["9".repeat(30), "b".into()],
            -45.0,
        );
        axes.x.tick_projection = None;
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 600.0,
            height: 600.0,
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        let (lo, _) = drawn_x_label_ink(&off, &m, theme.typography.label_font_size);
        assert!(lo < 0.0, "fixture assumption: the first label crosses the left edge ({lo})");
        assert_eq!(layout_with_auto(&theme, true, viewport, &axes, &m), off);
    }

    // ── drawn_x_labels matches the renderer ─────────────────────────────────

    /// Each x tick label `build_axis` emits for `layout`, as the ink span
    /// `(left, right)` its pivot, anchor, angle and font size give, measured
    /// with the extent helper `drawn_x_labels` uses. Lines of one label
    /// (one tick anchor) merge into one span.
    fn rendered_x_label_ink(
        layout: &LayoutResult,
        theme: &ThemeInputs,
        m: &dyn TextMetrics,
    ) -> Vec<(f64, f64)> {
        let mut spans: Vec<(f64, (f64, f64))> = Vec::new();
        for axis in layout
            .axes
            .iter()
            .filter(|a| matches!(a.orient, AxisOrient::Bottom | AxisOrient::Top))
        {
            for tick in crate::render::marks::axis::build_axis(axis, theme, None).data_positioned {
                let ferrum_scene::SceneNode::Text { x, content, style, .. } = tick.node else {
                    continue;
                };
                let anchor = match style.anchor {
                    ferrum_scene::TextAnchor::Start => TextAnchor::Start,
                    ferrum_scene::TextAnchor::Middle => TextAnchor::Middle,
                    ferrum_scene::TextAnchor::End => TextAnchor::End,
                };
                let (left, right) = axis::rotated_x_label_horizontal_extent(
                    style.angle,
                    anchor,
                    m.measure_width(&content, style.font_size),
                    m.line_height(style.font_size),
                );
                let span = (x - left, x + right);
                match spans.last_mut() {
                    Some((anchor_px, merged)) if *anchor_px == tick.anchor => {
                        *merged = (merged.0.min(span.0), merged.1.max(span.1));
                    }
                    _ => spans.push((tick.anchor, span)),
                }
            }
        }
        spans.into_iter().map(|(_, span)| span).collect()
    }

    /// Where `drawn_x_labels` says each x label's ink reaches is where the
    /// renderer draws it: for flat labels, rotated `Bottom` labels at -45
    /// and -90, a rotated `Top` axis, flushed first and last labels, and a
    /// cascade-shrunk font, every label `build_axis` emits spans exactly
    /// the canvas range `drawn_x_labels` reports, in the same order.
    #[test]
    fn drawn_x_labels_match_the_rendered_label_anchors() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 400.0,
            height: 400.0,
        };
        let labels = || vec!["-999".to_string(), "0".into(), "999999".into()];
        let angled = |orient, angle| {
            let mut axes = continuous_x_axes_at_angle(orient, labels(), angle);
            if let Some(p) = axes.x.tick_projection.as_mut() {
                p.major = vec![0.0, 0.5, 1.0];
            }
            axes
        };
        let mut flat = angled(AxisOrient::Bottom, 0.0);
        flat.x.overrides.labels.angle = None;
        let mut flush = flat.clone();
        flush.x.overrides.labels.flush = Some(true);
        let mut shrunk = continuous_x_axes_with_long_last_label();
        shrunk.x.tick_labels = ["0", "250,000", "500,000", "750,000", "1,000,000"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(p) = shrunk.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        }
        let cases = [
            ("flat", flat, viewport),
            ("bottom -45", angled(AxisOrient::Bottom, -45.0), viewport),
            ("bottom -90", angled(AxisOrient::Bottom, -90.0), viewport),
            ("top -45", angled(AxisOrient::Top, -45.0), viewport),
            ("flush", flush, viewport),
            ("shrunk", shrunk, Viewport { width: 330.0, ..viewport }),
        ];
        for (name, axes, viewport) in cases {
            let layout = layout_with_auto(&theme, false, viewport, &axes, &m);
            let drawn: Vec<(f64, f64)> = layout
                .axes
                .iter()
                .filter(|a| matches!(a.orient, AxisOrient::Bottom | AxisOrient::Top))
                .flat_map(|a| axis::drawn_x_labels(a, theme.typography.label_font_size, &m))
                .map(|l| (l.left_ink, l.right_ink))
                .collect();
            let rendered = rendered_x_label_ink(&layout, &theme, &m);
            assert!(!drawn.is_empty(), "{name}: fixture draws labels");
            assert_eq!(drawn.len(), rendered.len(), "{name}: one span per drawn label");
            for (d, r) in drawn.iter().zip(&rendered) {
                assert!(
                    (d.0 - r.0).abs() < 1e-9 && (d.1 - r.1).abs() < 1e-9,
                    "{name}: drawn_x_labels {d:?} vs rendered {r:?}",
                );
            }
        }
    }

    // ── the x label band is what the axis draws ─────────────────────────────

    /// The x label band a laid-out `layout` reserves below its plot: the
    /// padded region's bottom less the plot's, for a chart with no x title
    /// and no legend.
    fn reserved_x_label_band(layout: &LayoutResult) -> f64 {
        let padded = layout.padded_region;
        let plot = layout.plot_region;
        (padded.y + padded.h) - (plot.y + plot.h)
    }

    /// The band the x labels `layout` draws need, read off the drawn ticks
    /// independently of the estimator: a rotated label's full extent at its
    /// drawn angle; below the standoff, one line height at the axis font
    /// and a 1.2em step at the drawn font per further line; or the tick
    /// marks alone when no label is drawn.
    fn drawn_x_label_band(layout: &LayoutResult, theme: &ThemeInputs, m: &dyn TextMetrics) -> f64 {
        let fs = theme.typography.label_font_size;
        let tick = theme.sizes.tick_size;
        let line_h = m.line_height(fs);
        let drawn: Vec<&TickLayout> = layout
            .axes
            .iter()
            .filter(|a| a.orient == AxisOrient::Bottom)
            .flat_map(|a| a.ticks.iter().filter(|t| !t.culled))
            .collect();
        if drawn.is_empty() {
            return tick;
        }
        let angle = drawn.first().map_or(0.0, |t| t.label_angle);
        if angle != 0.0 {
            let w = drawn.iter().map(|t| m.measure_width(&t.label, fs)).fold(0.0, f64::max);
            let (sin, cos) = angle.to_radians().sin_cos();
            return tick + 2.0 + sin.abs() * (fs + w) + cos.abs() * line_h;
        }
        let lines = drawn.iter().map(|t| t.label.split('\n').count()).max().unwrap_or(1);
        let drawn_fs = drawn[0].label_font_size.unwrap_or(fs);
        tick + 2.0 + line_h + (lines - 1) as f64 * 1.2 * drawn_fs
    }

    /// A 272x150 chart whose last x label is `1,234,567,890` (about 86px):
    /// with `auto=false` the labels draw flat and the long one overhangs the
    /// right edge. The rescue narrows the plot to 167.5px, where each label
    /// still has the whole plot width between the two ticks, so the cascade
    /// keeps them flat. The band must be the flat band (the standoff plus one
    /// line), not one reserved for a rotation that is never drawn, so the
    /// rescue holds (the label ends on the canvas edge) and the plot keeps
    /// its full `auto=false` height.
    #[test]
    fn a_flat_rescued_label_reserves_the_flat_band() {
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["0".into(), "1,234,567,890".into()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.padding_frac = 0.0;
        }
        axes.y.tick_labels = vec!["0".into(), "999999".into()];
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 272.0,
            height: 150.0,
        };
        let fs = theme.typography.label_font_size;
        let flat_band = theme.sizes.tick_size + 2.0 + m.line_height(fs);
        let angles = |l: &LayoutResult| {
            l.axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present")
                .ticks
                .iter()
                .map(|t| (t.label_angle, t.culled))
                .collect::<Vec<_>>()
        };
        let off = layout_with_auto(&theme, false, viewport, &axes, &m);
        assert_eq!(angles(&off), vec![(0.0, false); 2], "fixture assumption: flat with auto off");
        assert!(drawn_x_label_ink(&off, &m, fs).1 > viewport.width, "fixture assumption: overhangs");
        let on = layout_with_auto(&theme, true, viewport, &axes, &m);
        assert_eq!(angles(&on), vec![(0.0, false); 2], "the rescued labels are drawn flat");
        assert!(
            (reserved_x_label_band(&on) - flat_band).abs() < 1e-9,
            "the reserved band ({}) is the flat band ({flat_band})",
            reserved_x_label_band(&on),
        );
        let (lo, hi) = drawn_x_label_ink(&on, &m, fs);
        assert!(lo >= 0.0 && (hi - viewport.width).abs() < 1e-9, "the rescue holds ({lo}, {hi})");
        assert!(
            on.panels[0].plot_area.h > off.panels[0].plot_area.h / 2.0,
            "the plot keeps more than half its auto=false height ({} vs {})",
            on.panels[0].plot_area.h,
            off.panels[0].plot_area.h,
        );
    }

    /// Labels that really rotate keep their rotated band: eleven currency
    /// labels on a 300px chart do not fit flat, wrapped or shrunk, so the
    /// cascade rotates them, and the band is that rotation's full extent for
    /// the widest drawn label, larger than the flat band.
    #[test]
    fn labels_that_rotate_keep_the_rotated_band() {
        let axes = currency_x_axes();
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let fs = theme.typography.label_font_size;
        for auto in [false, true] {
            let layout = layout_with_auto(
                &theme,
                auto,
                Viewport {
                    width: 300.0,
                    height: 300.0,
                },
                &axes,
                &m,
            );
            let x = layout
                .axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .expect("x axis present");
            let angle = x.ticks.iter().find(|t| !t.culled).map_or(0.0, |t| t.label_angle);
            assert!(angle != 0.0, "fixture assumption: the labels rotate (auto {auto})");
            let band = reserved_x_label_band(&layout);
            assert!(
                (band - drawn_x_label_band(&layout, &theme, &m)).abs() < 1e-9
                    && band > theme.sizes.tick_size + 2.0 + m.line_height(fs),
                "the rotated band ({band}) is the drawn rotation's extent (auto {auto})",
            );
        }
    }

    /// `spec` under `fm.CoordFixed(ratio=ratio)`, or unchanged for `None`.
    fn with_fixed_aspect(mut spec: ChartSpec, ratio: Option<f64>) -> ChartSpec {
        spec.coord = ratio.map(|ratio| crate::spec::coord::CoordKind::Fixed {
            ratio,
            x_domain: None,
            y_domain: None,
            expand: true,
            clip: true,
        });
        spec
    }

    /// Across every width the cascade's presentations span, label sets that
    /// draw flat, wrapped, shrunk, rotated and culled, `label_overlap`
    /// policies, one or two facet columns, landscape and portrait charts
    /// and a `CoordFixed` aspect of 0.5, 1 or 2 (which narrows the panel the
    /// labels draw across by an amount that depends on the band), with
    /// `padding.auto` on and off, the reserved x label band is exactly the
    /// band the drawn labels need, and a chart this tall always draws a
    /// label. The population must cover flat and rotated presentations, a
    /// faceted one, and a `CoordFixed` one whose aspect narrows the panel.
    #[test]
    fn the_reserved_x_label_band_is_what_the_axis_draws() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let groups: Vec<FacetGroup> = three_groups().into_iter().take(2).collect();
        let label_sets: Vec<Vec<String>> = vec![
            vec!["0".into(), "1,234,567,890".into()],
            vec!["0".into(), "500".into(), "1,000".into(), "1,500".into(), "2,000".into()],
            vec!["alpha beta".into(), "gamma delta".into(), "epsilon zeta".into()],
            (0..=10).map(|i| format!("${},000", i * 100)).collect(),
        ];
        let (mut flat, mut rotated, mut faceted, mut narrowed) = (0, 0, 0, 0);
        for labels in &label_sets {
            for overlap in [None, Some(LabelOverlap::Parity), Some(LabelOverlap::ShowAll)] {
                let mut axes = continuous_x_axes_with_long_last_label();
                let k = labels.len();
                axes.x.tick_labels = labels.clone();
                axes.x.overrides.labels.overlap = overlap;
                if let Some(p) = axes.x.tick_projection.as_mut() {
                    p.major = (0..k).map(|i| i as f64 / (k - 1) as f64).collect();
                }
                for (cols, ratio, portrait) in [None, Some(2)]
                    .into_iter()
                    .flat_map(|c| [None, Some(0.5), Some(1.0), Some(2.0)].map(move |r| (c, r)))
                    .flat_map(|(c, r)| [false, true].map(move |p| (c, r, p)))
                {
                    if ratio.is_none() && portrait {
                        continue;
                    }
                    let spec = with_fixed_aspect(cols.map_or_else(minimal_chart_spec, faceted_spec), ratio);
                    let facet_groups: &[FacetGroup] = if cols.is_some() { &groups } else { &[] };
                    for width in (120..=900).step_by(15) {
                        let height = if portrait { 1.5 * f64::from(width) } else { 400.0 };
                        for auto in [false, true] {
                            let mut t = theme.clone();
                            t.padding.padding_auto = auto;
                            let layout = compute_layout(
                                &spec,
                                &t,
                                Viewport {
                                    width: f64::from(width),
                                    height,
                                },
                                &axes,
                                facet_groups,
                                &[],
                                None,
                                None,
                                &m,
                                &legend::LegendOverrides::default(),
                                &[],
                                CompositeLayoutSeam::default(),
                            )
                            .expect("layout should succeed");
                            if layout.axes.iter().all(|a| a.orient != AxisOrient::Bottom) {
                                continue;
                            }
                            let band = reserved_x_label_band(&layout);
                            let drawn = drawn_x_label_band(&layout, &t, &m);
                            assert!(
                                (band - drawn).abs() < 1e-9,
                                "{labels:?} {overlap:?} cols {cols:?} ratio {ratio:?} portrait {portrait} \
                                 at {width}px auto {auto}: reserved {band}, drawn {drawn}",
                            );
                            let x = layout.axes.iter().find(|a| a.orient == AxisOrient::Bottom);
                            let first_drawn = x.and_then(|a| a.ticks.iter().find(|t| !t.culled));
                            assert!(
                                first_drawn.is_some(),
                                "{labels:?} {overlap:?} cols {cols:?} ratio {ratio:?} at {width}px auto {auto}: no label drawn",
                            );
                            if first_drawn.is_some_and(|t| t.label_angle != 0.0) {
                                rotated += 1;
                            } else {
                                flat += 1;
                            }
                            faceted += usize::from(cols.is_some());
                            narrowed += usize::from(
                                ratio.is_some() && layout.panels.iter().all(|p| p.plot_area.h > 0.0)
                                    && layout.panels[0].plot_area.w / layout.panels[0].plot_area.h > ratio.unwrap_or(0.0) - 1e-9
                                    && layout.panels[0].plot_area.w + 1e-9 < layout.plot_region.w / cols.unwrap_or(1) as f64,
                            );
                        }
                    }
                }
            }
        }
        assert!(
            flat > 0 && rotated > 0 && faceted > 0 && narrowed > 0,
            "flat {flat}, rotated {rotated}, faceted {faceted}, narrowed {narrowed}",
        );
    }

    /// The x labels are judged against the panel the secondary y axes leave:
    /// with none, one and two secondary y axes, at every width of the sweep
    /// the reserved x label band is the band the axis draws, and for one and
    /// for two secondary axes the sweep reaches widths where the secondary
    /// bands change the drawn x label angle against the same chart with
    /// fewer of them.
    #[test]
    fn the_reserved_x_label_band_is_what_the_axis_draws_beside_secondary_y_axes() {
        let m = MockMetrics {
            measure: fixed_width(8.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let label_sets: Vec<Vec<String>> = vec![
            vec!["0".into(), "500".into(), "1,000".into(), "1,500".into(), "2,000".into()],
            (0..=10).map(|i| format!("${},000", i * 100)).collect(),
        ];
        let angle = |l: &LayoutResult| {
            l.axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .and_then(|a| a.ticks.iter().find(|t| !t.culled))
                .map(|t| t.label_angle)
        };
        let drawable = |l: &LayoutResult| l.panels.iter().all(|p| p.plot_area.w > MIN_PANEL_DIM && p.plot_area.h > MIN_PANEL_DIM);
        // The drawn x label angle of each drawable layout, keyed by
        // (label set, width, auto, secondary count).
        let mut angles: std::collections::HashMap<(usize, u32, bool, usize), Option<f64>> = std::collections::HashMap::new();
        for secondaries in [0, 1, 2] {
            for (set, labels) in label_sets.iter().enumerate() {
                let mut axes = continuous_x_axes_with_long_last_label();
                let k = labels.len();
                axes.x.tick_labels = labels.clone();
                if let Some(p) = axes.x.tick_projection.as_mut() {
                    p.major = (0..k).map(|i| i as f64 / (k - 1) as f64).collect();
                }
                let mut beside = axes;
                beside.secondary_y = n_secondary_axes(secondaries);
                for width in (120..=900).step_by(5) {
                    let viewport = Viewport {
                        width: f64::from(width),
                        height: 400.0,
                    };
                    for auto in [false, true] {
                        let layout = layout_with_auto(&theme, auto, viewport, &beside, &m);
                        if !drawable(&layout) {
                            continue;
                        }
                        let band = reserved_x_label_band(&layout);
                        let drawn = drawn_x_label_band(&layout, &theme, &m);
                        assert!(
                            (band - drawn).abs() < 1e-9,
                            "{secondaries} secondary y, {labels:?} at {width}px auto {auto}: reserved {band}, drawn {drawn}",
                        );
                        angles.insert((set, width, auto, secondaries), angle(&layout));
                    }
                }
            }
        }
        for secondaries in [1, 2] {
            let flips = angles
                .iter()
                .filter(|(&(set, width, auto, n), a)| {
                    n == secondaries
                        && a.is_some()
                        && angles.get(&(set, width, auto, n - 1)).is_some_and(|fewer| fewer.is_some() && fewer != *a)
                })
                .count();
            assert!(flips > 0, "{secondaries} secondary y: the sweep must reach a width where they change the x label angle");
        }
    }

    /// Under a `CoordFixed` aspect that binds the width, sizing the band at
    /// the width it leaves can alternate: flat labels at the region width
    /// reserve a short band, the taller plot that leaves is narrower under
    /// the aspect, the labels rotate there and reserve a tall band, and the
    /// shorter plot is wide enough again for flat labels. Across a sweep of
    /// widths and aspects, every chart where that naive sequence cycles
    /// still reserves exactly the band its axis draws, and draws a label.
    /// Where the labels instead rotate at the narrower width and stay
    /// rotated, the axis draws them as it does there with no bound: the band
    /// is sized for the rotation, never culled down to the flat band.
    #[test]
    fn a_coord_fixed_band_that_alternates_settles_on_what_the_axis_draws() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let legend_overrides = legend::LegendOverrides::default();
        let (mut cycles, mut stays_rotated) = (0, 0);
        for chars in 3..=5 {
            let mut axes = continuous_x_axes_with_long_last_label();
            let labels: Vec<String> = (0..6).map(|i| format!("{i}").repeat(chars)).collect();
            axes.x.tick_labels = labels;
            if let Some(p) = axes.x.tick_projection.as_mut() {
                p.major = (0..6).map(|i| f64::from(i) / 5.0).collect();
            }
            for ratio in [0.8, 1.0, 1.25, 2.0] {
                let spec = with_fixed_aspect(minimal_chart_spec(), Some(ratio));
                for width in (150..=700).step_by(10) {
                    for height in (100..=400).step_by(20) {
                        let viewport = Viewport {
                            width: f64::from(width),
                            height: f64::from(height),
                        };
                        let layout = compute_layout(
                            &spec,
                            &theme,
                            viewport,
                            &axes,
                            &[],
                            &[],
                            None,
                            None,
                            &m,
                            &legend_overrides,
                            &[],
                            CompositeLayoutSeam::default(),
                        )
                        .expect("layout should succeed");
                        let (padded, plot) = (layout.padded_region, layout.plot_region);
                        let region_w = padded.x + padded.w - plot.x;
                        let panel_w = |b: f64| region_w.min((padded.h - b).max(0.0) * ratio);
                        let band = |w: f64| x_label_band(&axes, w, f64::INFINITY, &theme, &m);
                        let b1 = band(region_w);
                        let b2 = band(panel_w(b1));
                        let b3 = band(panel_w(b2));
                        if b1 == b2 || panel_w(b1) >= region_w {
                            continue;
                        }
                        if b3 == b2 && b2 > b1 {
                            stays_rotated += 1;
                            let panel_w = layout.panels[0].plot_area.w;
                            assert_eq!(
                                drawn_x_label_band(&layout, &theme, &m),
                                band(panel_w),
                                "{chars}-char labels, ratio {ratio}, {width}x{height}: the labels draw as with no bound",
                            );
                            continue;
                        }
                        if b3 != b1 {
                            continue;
                        }
                        cycles += 1;
                        let reserved = reserved_x_label_band(&layout);
                        let drawn = drawn_x_label_band(&layout, &theme, &m);
                        assert!(
                            (reserved - drawn).abs() < 1e-9,
                            "{chars}-char labels, ratio {ratio}, {width}x{height}: reserved {reserved}, drawn {drawn}",
                        );
                        assert!(
                            layout
                                .axes
                                .iter()
                                .filter(|a| a.orient == AxisOrient::Bottom)
                                .any(|a| a.ticks.iter().any(|t| !t.culled)),
                            "{chars}-char labels, ratio {ratio}, {width}x{height}: a label is drawn",
                        );
                    }
                }
            }
        }
        assert!(
            cycles > 0 && stays_rotated > 0,
            "the sweep must reach both populations (alternating {cycles}, staying rotated {stays_rotated})",
        );
    }

    /// Under a `CoordFixed` aspect, labels that fit flat across the region
    /// can rotate at the narrow panel the aspect leaves into a band so tall
    /// it leaves that panel no height (and so, under the aspect, no width).
    /// The band is bounded so the panel stays drawable, as a cramped band is
    /// without an aspect: every chart in the sweep draws a plot and reserves
    /// the band its axis draws, and the sweep must reach charts whose
    /// unbounded band at the aspect's width would leave no panel.
    #[test]
    fn a_coord_fixed_band_never_leaves_the_panel_undrawable() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = (0..8).map(|i| format!("cxxxxxxxxxx{i:02}")).collect();
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = (0..8).map(|i| f64::from(i) / 7.0).collect();
        }
        let spec = with_fixed_aspect(minimal_chart_spec(), Some(2.0));
        let mut would_collapse = 0;
        for width in (400..=900).step_by(20) {
            for height in (110..=200).step_by(10) {
                let viewport = Viewport {
                    width: f64::from(width),
                    height: f64::from(height),
                };
                let layout = compute_layout(
                    &spec,
                    &theme,
                    viewport,
                    &axes,
                    &[],
                    &[],
                    None,
                    None,
                    &m,
                    &legend::LegendOverrides::default(),
                    &[],
                    CompositeLayoutSeam::default(),
                )
                .expect("layout should succeed");
                let area = layout.panels[0].plot_area;
                assert!(
                    area.w > MIN_PANEL_DIM && area.h > MIN_PANEL_DIM,
                    "{width}x{height}: the panel is drawn ({area:?})",
                );
                let reserved = reserved_x_label_band(&layout);
                let drawn = drawn_x_label_band(&layout, &theme, &m);
                assert!((reserved - drawn).abs() < 1e-9, "{width}x{height}: reserved {reserved}, drawn {drawn}");
                let padded = layout.padded_region;
                let band = |w: f64| x_label_band(&axes, w, f64::INFINITY, &theme, &m);
                let region_w = padded.x + padded.w - layout.plot_region.x;
                let flat = band(region_w);
                let aspect_w = region_w.min((padded.h - flat).max(0.0) * 2.0);
                if padded.h - band(aspect_w) <= MIN_PANEL_DIM {
                    would_collapse += 1;
                }
            }
        }
        assert!(would_collapse > 0, "the sweep must reach a band that would leave no panel");
    }

    /// A `CoordFixed` aspect that only shortens the panel (the region is too
    /// tall for the ratio) leaves the width the labels draw across, so the
    /// band and its labels are those of the same chart without the aspect.
    #[test]
    fn a_coord_fixed_aspect_that_binds_the_height_keeps_the_band() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 300.0,
            height: 900.0,
        };
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = (0..6).map(|i| format!("{i}").repeat(7)).collect();
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = (0..6).map(|i| f64::from(i) / 5.0).collect();
        }
        let lay_out = |spec: &ChartSpec| {
            compute_layout(
                spec,
                &theme,
                viewport,
                &axes,
                &[],
                &[],
                None,
                None,
                &m,
                &legend::LegendOverrides::default(),
                &[],
                CompositeLayoutSeam::default(),
            )
            .expect("layout should succeed")
        };
        let plain = lay_out(&minimal_chart_spec());
        let fixed = lay_out(&with_fixed_aspect(minimal_chart_spec(), Some(1.0)));
        assert!(
            fixed.panels[0].plot_area.h < plain.panels[0].plot_area.h - 1.0
                && fixed.panels[0].plot_area.w == plain.panels[0].plot_area.w,
            "fixture assumption: the aspect shortens the panel only",
        );
        assert_eq!(reserved_x_label_band(&fixed), reserved_x_label_band(&plain));
        let ticks = |l: &LayoutResult| {
            l.axes
                .iter()
                .find(|a| a.orient == AxisOrient::Bottom)
                .map(|a| a.ticks.iter().map(|t| (t.label_angle, t.culled)).collect::<Vec<_>>())
        };
        assert_eq!(ticks(&fixed), ticks(&plain));
    }

    /// An axis auto wants nothing on is never flagged as scaled, even on a
    /// viewport with no room left past the `auto=false` insets (where the
    /// cap's factor is 0 whatever is wanted): a 20x20 chart whose axes draw
    /// nothing auto pads for.
    #[test]
    fn an_axis_wanting_nothing_is_never_flagged_scaled() {
        let spec = minimal_chart_spec();
        let legend_overrides = legend::LegendOverrides::default();
        let axes = dummy_axes();
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let viewport = Viewport {
            width: 20.0,
            height: 20.0,
        };
        let inputs = plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport);
        let room = AutoPlotRoom::of_auto_off(&inputs, base_padding_insets(&theme));
        let extra = auto_padding_extra(&inputs, Inset::default(), XLabelOverhang::NONE, &room);
        assert_eq!(extra.total, Inset::default(), "fixture assumption: nothing is wanted");
        assert!(!extra.x_scaled && !extra.y_scaled, "{extra:?}");
    }

    /// The `padding.auto` y-title pass recentres the title on the plot the
    /// layout lays out: on a 2-column facet grid whose x labels draw flat at
    /// the whole region's width but rotate at each panel's, the band (and
    /// so the plot's vertical centre) is the one the panels draw. The title
    /// extra is the recentering against the laid-out plot region's centre.
    #[test]
    fn the_y_title_rescue_recentres_on_the_faceted_plot_that_lays_out() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let spec = faceted_spec(2);
        let groups: Vec<FacetGroup> = three_groups().into_iter().take(2).collect();
        let legend_overrides = legend::LegendOverrides::default();
        let base = base_padding_insets(&theme);
        let mut differs = 0;
        for title in 10..=30 {
            for width in (240..=600).step_by(20) {
                let mut axes = continuous_x_axes_with_long_last_label();
                axes.x.tick_labels = (0..5).map(|i| format!("{},000", i * 250)).collect();
                if let Some(p) = axes.x.tick_projection.as_mut() {
                    p.major = (0..5).map(|i| f64::from(i) / 4.0).collect();
                }
                axes.y.title.set(Some("a".repeat(title)));
                let viewport = Viewport {
                    width: f64::from(width),
                    height: 160.0,
                };
                let inputs = LayoutInputs {
                    facet_groups: &groups,
                    ..plain_layout_inputs(&spec, &legend_overrides, &theme, &axes, &m, viewport)
                };
                let laid_out = lay_out_at(base, &inputs).plot;
                let region = laid_out.plot_region;
                let half_h = m.measure_width(&"a".repeat(title), theme.typography.title_font_size) / 2.0;
                let (top, bottom) = recentering_padding(region.y + region.h / 2.0, half_h, viewport.height);
                let rescue = auto_padding_for_axis_titles(&inputs, base, Inset::default());
                assert!(
                    (rescue.top - top).abs() < 1e-9 && (rescue.bottom - bottom).abs() < 1e-9,
                    "{title}-char y title at {width}px: rescue {rescue:?}, want top {top}, bottom {bottom}",
                );
                let cell_w = laid_out.panels[0].plot_area.w;
                let band_at = |w: f64| x_label_band(&axes, w, f64::INFINITY, &theme, &m);
                if top + bottom > 0.0 && band_at(region.w) != band_at(cell_w) {
                    differs += 1;
                }
            }
        }
        assert!(differs > 0, "the sweep must reach a rescued title whose region and panel bands differ");
    }

    /// The facet row gutter holds the band the upper row's x labels draw,
    /// sized at the panel width they are drawn at, including when a
    /// composite imposes the plot region: a 3-panel, 2-column chart whose
    /// imposed region is far narrower than its natural one, where the
    /// labels fit flat at the natural width and rotate at the imposed one.
    #[test]
    fn the_facet_row_gutter_holds_the_labels_drawn_at_the_imposed_width() {
        let m = MockMetrics {
            measure: font_scaled_width,
            line_h_factor: 1.2,
        };
        let theme = default_theme_inputs();
        let mut axes = continuous_x_axes_with_long_last_label();
        axes.x.tick_labels = vec!["0".into(), "250,000".into(), "500,000".into()];
        if let Some(p) = axes.x.tick_projection.as_mut() {
            p.major = vec![0.0, 0.5, 1.0];
        }
        let spec = faceted_spec(2);
        let groups = three_groups();
        let viewport = Viewport {
            width: 800.0,
            height: 500.0,
        };
        let lay_out = |plot_region: Option<Rect>| {
            compute_layout(
                &spec,
                &theme,
                viewport,
                &axes,
                &groups,
                &[],
                None,
                None,
                &m,
                &legend::LegendOverrides::default(),
                &[],
                CompositeLayoutSeam {
                    plot_region,
                    ..CompositeLayoutSeam::default()
                },
            )
            .expect("faceted layout should succeed")
        };
        let row_gutter = |l: &LayoutResult| {
            let top = l.panels.iter().find(|p| p.row == 0).expect("row 0").plot_area;
            let below = l.panels.iter().find(|p| p.row == 1).expect("row 1").plot_area;
            let strip = strip_band_size(&theme, &m);
            below.y - strip - (top.y + top.h) - theme.padding.row_padding
        };
        let upper_band = |l: &LayoutResult| {
            let upper = LayoutResult {
                axes: l
                    .axes
                    .iter()
                    .filter(|a| l.panels.get(a.panel_index).is_some_and(|p| p.row == 0))
                    .cloned()
                    .collect(),
                ..l.clone()
            };
            drawn_x_label_band(&upper, &theme, &m)
        };
        let angle = |l: &LayoutResult| l.axes.iter().find(|a| a.orient == AxisOrient::Bottom).map(|a| a.ticks[0].label_angle);
        let natural = lay_out(None);
        assert_eq!(angle(&natural), Some(0.0), "fixture assumption: flat at the natural width");
        let imposed = lay_out(Some(Rect {
            x: 100.0,
            y: 40.0,
            w: 160.0,
            h: 400.0,
        }));
        assert!(angle(&imposed).is_some_and(|a| a != 0.0), "fixture assumption: rotated when imposed");
        for (name, l) in [("natural", &natural), ("imposed", &imposed)] {
            assert!(
                (row_gutter(l) - upper_band(l)).abs() < 1e-9,
                "{name}: the row gutter ({}) holds the upper row's drawn band ({})",
                row_gutter(l),
                upper_band(l),
            );
        }
    }
}
