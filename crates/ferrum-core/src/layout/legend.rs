//! Legend input (caller-supplied entries) and output (engine-computed rects).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LegendOrient {
    Right,
    Left,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LegendDirection {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Square,
    Circle,
    Line,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LegendEntry {
    pub label: String,
    pub symbol: SymbolKind,
}

/// Input shape passed to `compute_layout` for a continuous colorbar legend.
/// Mirrors `LegendEntry` for the categorical path; `compute_layout` calls
/// `layout_colorbar` when this is `Some` and `legend_entries` is empty.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorbarInput {
    /// Gradient stops as (position in [0, 1], css hex color).
    pub stops: Vec<(f64, String)>,
    /// Tick labels low → high; the layout linearly distributes them along
    /// the bar from bottom to top.
    pub tick_labels: Vec<String>,
    /// Numeric `(lo, hi)` data domain the labels span. `None` when the labels are
    /// an explicit non-numeric override (e.g. SHAP `["Low", "High"]`). Carried so
    /// `compute_layout` can apply `tick_min_step` (B5 unit 3) with the data step.
    pub domain: Option<(f64, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendLayout {
    pub rect: Rect,
    pub orient: LegendOrient,
    pub direction: LegendDirection,
    pub entries: Vec<LegendEntryLayout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<LegendTitleLayout>,
    /// Continuous colorbar layout — emitted alongside or in place of
    /// categorical `entries` when the color encoding maps to a numeric
    /// domain. The render layer draws a gradient bar with tick labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colorbar: Option<ColorbarLayout>,
    /// B5 unit 3: stroke width (px) for legend symbol swatches. `None` → no
    /// symbol stroke (historical default; omitted from serialized JSON so
    /// non-styled legends round-trip byte-identically).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol_stroke_width: Option<f64>,
    /// B5 unit 3: cap on the legend group's drawn height (px). When `Some`, the
    /// renderer wraps the legend in an SVG `clipPath` so overflowing entries are
    /// hard-clipped. `None` → no clip (default; omitted from JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_height: Option<f64>,
    /// B5 unit 6a: legend swatch/symbol area in px². The renderer derives the
    /// swatch geometry (circle radius, square side, line length, shape glyph
    /// radius) from this area via the mark `size` channel's `radius = sqrt(area /
    /// PI)` mapping. `None` → the fixed default swatch geometry (byte-identical;
    /// omitted from JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol_size: Option<f64>,
    /// B5 unit 6a: fill color (css hex) of the legend ENTRY label text. `None` →
    /// the theme `font_color` default (byte-identical; omitted from JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_color: Option<String>,
    /// Effective font size (px) for entry/colorbar-tick label text. Carries the
    /// per-legend `label_font_size` override so the renderer draws text at the
    /// same size the layout reserved space for. `None` → the theme
    /// `label_font_size` default (byte-identical; omitted from JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_font_size: Option<f64>,
}

/// Continuous-color colorbar within a LegendLayout. The bar is rendered as
/// a vertical SVG `linearGradient` rect with tick marks + labels on its
/// right edge. Used by `clustermap()` and any chart with a continuous color
/// encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorbarLayout {
    /// Pixel rect of the gradient strip (excluding tick labels and title).
    pub bar_rect: Rect,
    /// Gradient stops as (position in [0, 1], css hex color). Stop count
    /// depends on the source scheme — 2 for simple linear, N for sampled
    /// schemes (viridis etc.).
    pub stops: Vec<(f64, String)>,
    /// Tick anchor (data value, label, y-pixel). The renderer emits a tick
    /// mark + label per entry.
    pub ticks: Vec<ColorbarTick>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorbarTick {
    pub label: String,
    /// Y pixel position (top-of-bar = colorbar's max value if `flipped`,
    /// bottom-of-bar = min value).
    pub y: f64,
}

/// Legend title placement (Themes-T2.5b). Positioned above the entries
/// when direction=Vertical, or to the left when direction=Horizontal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendTitleLayout {
    pub text: String,
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendEntryLayout {
    pub label: String,
    pub label_anchor_x: f64,
    pub label_anchor_y: f64,
    pub symbol_anchor_x: f64,
    pub symbol_anchor_y: f64,
    pub symbol_kind: SymbolKind,
    /// Size-legend graduated symbol radius in pixels. When `Some`, the entry's
    /// circle swatch is drawn at this radius (graduated-symbol size legend)
    /// rather than the fixed legend swatch radius. `None` for color/shape
    /// entries (byte-identical serialization to the pre-size-legend shape).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol_radius: Option<f64>,
    /// Shape-legend point-shape name (e.g. `"diamond"`, `"triangle-up"`). When
    /// `Some`, the renderer draws the matching point glyph instead of the
    /// `symbol_kind` swatch. `None` for color/size entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_name: Option<String>,
    /// Explicit per-entry swatch color (css hex). Set on a merged color+size
    /// legend so each graduated symbol also carries the color the shared field
    /// maps to. `None` falls back to the color-scale lookup (color legend) or
    /// the theme mark color (size/shape legend).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_hex: Option<String>,
}

use super::text_metrics::TextMetrics;

/// Per-chart legend overrides extracted from `encoding.color.legend` dict.
/// These are applied on top of `ThemeInputs` defaults at layout time.
#[derive(Debug, Clone, Default)]
pub struct LegendOverrides {
    /// Max ticks for a continuous colorbar. Subsets `tick_labels` before layout.
    pub tick_count: Option<usize>,
    /// Font size for entry/tick labels (overrides `theme.label_font_size`).
    pub label_font_size: Option<f64>,
    /// Pixel length (height) of the colorbar gradient bar (overrides `COLORBAR_HEIGHT`).
    pub gradient_length: Option<f64>,
    /// Pixel thickness (width) of the colorbar gradient bar (overrides `COLORBAR_WIDTH`).
    pub gradient_thickness: Option<f64>,
    /// Direction override for categorical legend (overrides `theme.legend_direction`).
    pub direction: Option<LegendDirection>,
    /// Explicit tick/entry labels. Replaces the auto-generated `tick_labels`.
    pub values: Option<Vec<String>>,
    /// `"gradient"` → force colorbar; `"symbol"` → force discrete entries.
    pub legend_type: Option<String>,
    /// Override legend symbol shape (e.g. `"circle"`, `"square"`, `"line"`).
    /// Applies to all categorical legend entries when set.
    pub symbol_type: Option<String>,
    /// B5 unit 3: stroke width of legend symbols (px). `None` → no symbol stroke
    /// (the historical default, byte-identical output).
    pub symbol_stroke_width: Option<f64>,
    /// B5 unit 3: vertical entry spacing (px) overriding `LEGEND_ENTRY_ROW_PAD`.
    pub row_padding: Option<f64>,
    /// B5 unit 3: horizontal entry spacing (px) for horizontal-direction legends,
    /// overriding `LEGEND_ENTRY_ROW_PAD`.
    pub column_padding: Option<f64>,
    /// B5 unit 3: max legend-label pixel width. Labels wider than this are
    /// truncated with an ellipsis (`…`) and the legend rect shrinks accordingly.
    pub label_limit: Option<f64>,
    /// B5 unit 3: cap the legend group's total height (px). Overflow is
    /// hard-clipped via an SVG `clipPath` on the legend group.
    pub clip_height: Option<f64>,
    /// B5 unit 3: minimum step between colorbar ticks (data units). Drops ticks
    /// whose labelled value is closer than this to a kept tick.
    pub tick_min_step: Option<f64>,
    /// B5 unit 6a: legend swatch area (px²). The renderer derives the swatch
    /// geometry from this via `radius = sqrt(area / PI)`.
    pub symbol_size: Option<f64>,
    /// B5 unit 6a: fill color (css hex) of the legend entry label text.
    pub label_color: Option<String>,
    /// B5 unit 6a: extra gap (px) between the plot area and the legend strip.
    pub offset: Option<f64>,
    /// B5 unit 6a: internal padding (px) between the legend box edge and its
    /// contents (replaces `LEGEND_OUTER_PAD` for the categorical legend).
    pub padding: Option<f64>,
    /// B5 unit 6a: gap (px) between the legend title and the first entry row
    /// (replaces `LEGEND_TITLE_GAP`).
    pub title_padding: Option<f64>,
}

const SYMBOL_WIDTH: f64 = 12.0;
const SYMBOL_LABEL_GAP: f64 = 4.0;
const LEGEND_OUTER_PAD: f64 = 8.0;
const LEGEND_ENTRY_ROW_PAD: f64 = 4.0;
const LEGEND_TITLE_GAP: f64 = 8.0;
/// Base gap between the plot area and the legend strip so data points near the
/// axis edge don't visually merge with legend text. The per-legend `offset`
/// (B5 unit 6a) adds to this.
const LEGEND_PLOT_GAP: f64 = 8.0;
/// The ellipsis appended to a label truncated by `label_limit`.
const LABEL_ELLIPSIS: &str = "…";

/// B5 unit 3 / 6a: per-legend styling/positioning options that previously had no
/// renderer (`symbol_stroke_width`, `row_padding`/`column_padding`,
/// `label_limit`, `clip_height`, `padding`/`title_padding`/`offset`,
/// `symbol_size`, `label_color`). Bundled into one struct so `layout_legend`
/// keeps a manageable arity. All fields are `Option`; `Default` (all `None`)
/// reproduces the historical layout byte-for-byte. `symbol_size`/`label_color`
/// do not affect layout geometry — they are carried straight onto the
/// `LegendLayout` for the renderer.
#[derive(Debug, Clone, Default)]
pub struct LegendStyleOpts {
    /// Stroke width (px) for legend symbol swatches; carried onto the
    /// `LegendLayout` for the renderer.
    pub symbol_stroke_width: Option<f64>,
    /// Vertical entry spacing (px), replacing `LEGEND_ENTRY_ROW_PAD` for the
    /// vertical (row) direction.
    pub row_padding: Option<f64>,
    /// Horizontal entry spacing (px), replacing `LEGEND_ENTRY_ROW_PAD` for the
    /// horizontal (column) direction. Also sets the inter-column gap in a
    /// multi-column vertical legend (`columns > 1`).
    pub column_padding: Option<f64>,
    /// Max legend-label width (px); labels wider than this are truncated with an
    /// ellipsis and the rect shrinks to the truncated width.
    pub label_limit: Option<f64>,
    /// Cap on the legend group height (px); carried onto the `LegendLayout` for
    /// the renderer to hard-clip.
    pub clip_height: Option<f64>,
    /// B5 unit 6a: internal padding (px) between the legend's bounding-box edge
    /// and its contents, replacing `LEGEND_OUTER_PAD` for the categorical legend.
    /// Affects both size estimation and entry/title placement so the rect grows
    /// with the inset. `None` → `LEGEND_OUTER_PAD` (byte-identical default).
    pub padding: Option<f64>,
    /// B5 unit 6a: gap (px) between the legend title and the first entry row,
    /// replacing `LEGEND_TITLE_GAP`. `None` → `LEGEND_TITLE_GAP`.
    pub title_padding: Option<f64>,
    /// B5 unit 6a: extra gap (px) between the plot area and the legend strip,
    /// added on top of the base `LEGEND_PLOT_GAP`. Shifts the whole legend away
    /// from the plot edge. `None` → no extra offset (byte-identical default).
    pub offset: Option<f64>,
    /// B5 unit 6a: legend swatch area (px²); carried onto the `LegendLayout` so
    /// the renderer scales the swatch geometry. Layout anchors are unaffected.
    pub symbol_size: Option<f64>,
    /// B5 unit 6a: legend entry-label fill color (css hex); carried onto the
    /// `LegendLayout` for the renderer. Layout is unaffected.
    pub label_color: Option<String>,
    /// Per-legend `label_font_size` override (px); carried onto the
    /// `LegendLayout` so the renderer draws entry labels at the same size the
    /// layout reserved space for. `None` → theme default (byte-identical).
    /// NOTE: the layout's own geometry already uses the resolved effective font
    /// size passed as `label_font_size`; this field exists solely to forward the
    /// override to the renderer.
    pub label_font_size: Option<f64>,
}

/// Truncate `label` to at most `limit` pixels wide, appending an ellipsis when
/// truncation occurs. Returns the original label untouched when it already fits
/// (or `limit` is non-positive). Used by `label_limit` (B5 unit 3).
fn truncate_label(label: &str, limit: f64, font_size: f64, metrics: &dyn TextMetrics) -> String {
    if limit <= 0.0 || metrics.measure_width(label, font_size) <= limit {
        return label.to_string();
    }
    let ellipsis_w = metrics.measure_width(LABEL_ELLIPSIS, font_size);
    let budget = (limit - ellipsis_w).max(0.0);
    let mut kept = String::new();
    for ch in label.chars() {
        let mut candidate = kept.clone();
        candidate.push(ch);
        if metrics.measure_width(&candidate, font_size) > budget {
            break;
        }
        kept = candidate;
    }
    if kept.is_empty() {
        // Nothing fits even before the ellipsis — emit just the ellipsis.
        return LABEL_ELLIPSIS.to_string();
    }
    kept.push_str(LABEL_ELLIPSIS);
    kept
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendSize {
    pub width: f64,
    pub height: f64,
}

/// Title-aware version of `estimate_legend_size`. For Vertical-direction
/// legends, adds a title line to the height. For Horizontal, adds title
/// width to the left. Falls through to the no-title shape when title=None.
///
/// `outer_pad` is the effective inner padding (`LEGEND_OUTER_PAD` or the
/// per-legend `padding` override) and `title_gap` the title-to-entry gap
/// (`LEGEND_TITLE_GAP` or the per-legend `title_padding` override); both must
/// match the values `layout_legend` places with so the rect stays sized.
#[allow(clippy::too_many_arguments)]
pub fn estimate_legend_size_with_title(
    entries: &[LegendEntry],
    orient: LegendOrient,
    label_font_size: f64,
    title: Option<&str>,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
    entry_pad: f64,
    outer_pad: f64,
    title_gap: f64,
    symbol_width: f64,
) -> LegendSize {
    let base = estimate_legend_size(
        entries, orient, label_font_size, metrics, entry_pad, outer_pad, symbol_width,
    );
    let Some(title_text) = title else { return base };
    let title_h = metrics.line_height(title_font_size);
    let title_w = metrics.measure_width(title_text, title_font_size);
    match orient {
        LegendOrient::Right | LegendOrient::Left => LegendSize {
            width: base.width.max(title_w + 2.0 * outer_pad),
            height: base.height + title_h + title_gap,
        },
        LegendOrient::Top | LegendOrient::Bottom => LegendSize {
            width: base.width + title_w + title_gap,
            height: base.height.max(title_h + 2.0 * outer_pad),
        },
    }
}

/// `entry_pad` is the inter-entry spacing — `LEGEND_ENTRY_ROW_PAD` by default,
/// or the per-legend `row_padding`/`column_padding` override (B5 unit 3). It
/// applies to the height for vertical legends and the width for horizontal ones.
///
/// `outer_pad` is the effective inner padding (`LEGEND_OUTER_PAD` by default,
/// or the per-legend `padding` override, B5 unit 6a). `symbol_width` is the
/// reserved swatch-column width (`SYMBOL_WIDTH` by default, widened when
/// `symbol_size` enlarges the swatch). Passing `LEGEND_OUTER_PAD` / `SYMBOL_WIDTH`
/// reproduces the historical size byte-for-byte.
pub fn estimate_legend_size(
    entries: &[LegendEntry],
    orient: LegendOrient,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
    entry_pad: f64,
    outer_pad: f64,
    symbol_width: f64,
) -> LegendSize {
    let line_h = metrics.line_height(label_font_size);
    let max_label_w = entries
        .iter()
        .map(|e| metrics.measure_width(&e.label, label_font_size))
        .fold(0.0_f64, f64::max);
    let n = entries.len() as f64;
    // Row pitch is driven by the label line height; the swatch is centered on
    // the row so `symbol_width` widens the column (so labels clear the swatch)
    // but does not change the vertical pitch. Default `symbol_width = 12`
    // reproduces the historical width byte-for-byte.

    match orient {
        LegendOrient::Right | LegendOrient::Left => {
            let entry_w = symbol_width + SYMBOL_LABEL_GAP + max_label_w;
            let width = entry_w + 2.0 * outer_pad;
            let height = if entries.is_empty() {
                0.0
            } else {
                n * line_h + (n - 1.0).max(0.0) * entry_pad + 2.0 * outer_pad
            };
            LegendSize { width, height }
        }
        LegendOrient::Top | LegendOrient::Bottom => {
            let entry_w = symbol_width + SYMBOL_LABEL_GAP + max_label_w;
            let width = if entries.is_empty() {
                0.0
            } else {
                n * entry_w + (n - 1.0).max(0.0) * entry_pad + 2.0 * outer_pad
            };
            let height = line_h + 2.0 * outer_pad;
            LegendSize { width, height }
        }
    }
}

/// Compute the legend rect on the given orient and the remaining inner rect for
/// the rest of the chart. Returns `(None, inner)` for empty entries.
/// Returns `(Some(legend_with_dropped_entries), reduced_inner)` if the legend
/// overflows the available strip.
#[allow(clippy::too_many_arguments)]
pub fn layout_legend(
    entries: &[LegendEntry],
    orient: LegendOrient,
    inner: Rect,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
    direction_override: Option<LegendDirection>,
    title: Option<&str>,
    title_font_size: f64,
    columns: Option<u32>,
    symbol_type_override: Option<&str>,
    opts: LegendStyleOpts,
) -> (Option<LegendLayout>, Rect) {
    if entries.is_empty() {
        return (None, inner);
    }
    let override_symbol: Option<SymbolKind> = symbol_type_override.and_then(|s| match s {
        "circle"         => Some(SymbolKind::Circle),
        "square" | "rect" => Some(SymbolKind::Square),
        "line"           => Some(SymbolKind::Line),
        _                => None,
    });

    let direction = direction_override.unwrap_or(match orient {
        LegendOrient::Right | LegendOrient::Left => LegendDirection::Vertical,
        LegendOrient::Top | LegendOrient::Bottom => LegendDirection::Horizontal,
    });
    // Per-direction entry spacing: vertical legends use `row_padding`, horizontal
    // ones use `column_padding`; either defaults to `LEGEND_ENTRY_ROW_PAD` so the
    // unset path is byte-identical.
    let entry_pad = match direction {
        LegendDirection::Vertical => opts.row_padding,
        LegendDirection::Horizontal => opts.column_padding,
    }
    .unwrap_or(LEGEND_ENTRY_ROW_PAD);

    // B5 unit 6a effective spacing. `padding` replaces `LEGEND_OUTER_PAD` (inner
    // inset), `title_padding` replaces `LEGEND_TITLE_GAP` (title→entry gap), and
    // `offset` adds extra distance from the plot edge. Each defaults to the
    // historical constant / 0 so the unset path is byte-identical.
    let outer_pad = opts.padding.unwrap_or(LEGEND_OUTER_PAD);
    let title_gap = opts.title_padding.unwrap_or(LEGEND_TITLE_GAP);
    let extra_offset = opts.offset.unwrap_or(0.0);
    // Inter-column gap for a multi-column vertical legend. `column_padding`
    // doubles as this horizontal gap; absent → 0 (columns abut evenly, the
    // historical layout).
    let col_gap = opts.column_padding.unwrap_or(0.0);

    // B5 unit 6a: a `symbol_size`-enlarged swatch needs a wider symbol column so
    // the label clears it. The swatch radius is `sqrt(area / PI)` (the mark
    // `size` mapping, mirrored in the renderer); the widest swatch dimension is
    // the line swatch at `3 * r` (= `SYMBOL_WIDTH` at the default `r = 4`).
    // Reserve `max(SYMBOL_WIDTH, 3 * r)` so the default stays byte-identical.
    let symbol_width = match opts.symbol_size.filter(|a| *a > 0.0 && a.is_finite()) {
        Some(area) => {
            let r = (area / std::f64::consts::PI).sqrt();
            SYMBOL_WIDTH.max(3.0 * r)
        }
        None => SYMBOL_WIDTH,
    };

    // `label_limit`: truncate every entry label to the pixel budget up front so
    // size estimation, the rect, and the laid-out anchors all see the shortened
    // text. When unset, the entries pass through untouched (byte-identical).
    let entries_owned: Vec<LegendEntry>;
    let entries: &[LegendEntry] = if let Some(limit) = opts.label_limit {
        entries_owned = entries
            .iter()
            .map(|e| LegendEntry {
                label: truncate_label(&e.label, limit, label_font_size, metrics),
                symbol: e.symbol,
            })
            .collect();
        &entries_owned
    } else {
        entries
    };

    let size = estimate_legend_size_with_title(
        entries, orient, label_font_size, title, title_font_size, metrics, entry_pad,
        outer_pad, title_gap, symbol_width,
    );

    // Add a small gap between the plot area and the legend strip so data
    // points near the axis edge don't visually merge with legend text. The
    // per-legend `offset` (B5 unit 6a) widens this gap further.
    let legend_plot_gap = LEGEND_PLOT_GAP + extra_offset;

    let (legend_rect, plot_inner) = match orient {
        LegendOrient::Right => {
            let w = size.width.min(inner.w * 0.5);
            let (strip_with_gap, rest) = inner.split_right(w + legend_plot_gap);
            // Inset the legend rect by the gap on its left side.
            let legend_rect = crate::layout::Rect {
                x: strip_with_gap.x + legend_plot_gap,
                y: strip_with_gap.y,
                w: strip_with_gap.w - legend_plot_gap,
                h: strip_with_gap.h,
            };
            (legend_rect, rest)
        }
        LegendOrient::Left => {
            let w = size.width.min(inner.w * 0.5);
            let (strip_with_gap, rest) = inner.split_left(w + legend_plot_gap);
            let legend_rect = crate::layout::Rect {
                x: strip_with_gap.x,
                y: strip_with_gap.y,
                w: strip_with_gap.w - legend_plot_gap,
                h: strip_with_gap.h,
            };
            (legend_rect, rest)
        }
        LegendOrient::Top => {
            let h = (size.height + extra_offset).min(inner.h * 0.5);
            let (strip, rest) = inner.split_top(h);
            // Inset the legend rect down by the offset gap from the plot edge.
            let legend_rect = crate::layout::Rect {
                x: strip.x,
                y: strip.y + extra_offset,
                w: strip.w,
                h: (strip.h - extra_offset).max(0.0),
            };
            (legend_rect, rest)
        }
        LegendOrient::Bottom => {
            let h = (size.height + extra_offset).min(inner.h * 0.5);
            let (strip, rest) = inner.split_bottom(h);
            // Inset the legend rect by the offset gap on its top side.
            let legend_rect = crate::layout::Rect {
                x: strip.x,
                y: strip.y + extra_offset,
                w: strip.w,
                h: (strip.h - extra_offset).max(0.0),
            };
            (legend_rect, rest)
        }
    };

    let line_h = metrics.line_height(label_font_size);

    // Title placement: above entries (Vertical) or to the left (Horizontal).
    // y_offset / x_offset values push entry layout to start past the title.
    let (title_layout, title_y_offset, title_x_offset) = if let Some(title_text) = title {
        let title_h = metrics.line_height(title_font_size);
        let title_w = metrics.measure_width(title_text, title_font_size);
        match direction {
            LegendDirection::Vertical => {
                // Title sits at the top, left-aligned with entry symbols.
                let tx = legend_rect.x + outer_pad;
                let ty = legend_rect.y + outer_pad + title_h;
                (
                    Some(LegendTitleLayout { text: title_text.to_string(), x: tx, y: ty }),
                    title_h + title_gap,
                    0.0,
                )
            }
            LegendDirection::Horizontal => {
                // Title sits to the left, vertically centered with entry row.
                let tx = legend_rect.x + outer_pad;
                let ty = legend_rect.y + legend_rect.h / 2.0 + title_h / 3.0;
                (
                    Some(LegendTitleLayout { text: title_text.to_string(), x: tx, y: ty }),
                    0.0,
                    title_w + title_gap,
                )
            }
        }
    } else {
        (None, 0.0, 0.0)
    };

    let entries_laid_out: Vec<LegendEntryLayout> = match direction {
        LegendDirection::Vertical => {
            let avail_h = (legend_rect.h - 2.0 * outer_pad - title_y_offset).max(0.0);
            let avail_w = (legend_rect.w - 2.0 * outer_pad).max(0.0);
            let row_pitch = line_h + entry_pad;
            let n_cols = columns.unwrap_or(1).max(1) as usize;

            // Column width: divide the available width (minus the inter-column
            // gaps) evenly across N columns, then place each column at
            // `col_w + col_gap` pitch. `col_gap` defaults to 0, so without
            // `column_padding` the columns abut exactly as before (byte-identical).
            // Label overflow within a column is acceptable — text truncation is render-side.
            let col_pitch = if n_cols > 1 {
                let gaps_total = (n_cols - 1) as f64 * col_gap;
                let col_w = ((avail_w - gaps_total).max(0.0)) / n_cols as f64;
                col_w + col_gap
            } else {
                avail_w
            };

            let max_rows = if row_pitch > 0.0 {
                ((avail_h + entry_pad) / row_pitch).floor() as usize
            } else {
                0
            };
            // Maximum entries that fit: rows × columns.
            let max_fit = max_rows * n_cols;
            let n_fit = entries.len().min(max_fit.max(1));
            entries
                .iter()
                .take(n_fit)
                .enumerate()
                .map(|(i, e)| {
                    let col = i % n_cols;
                    let row = i / n_cols;
                    let col_origin_x = legend_rect.x + outer_pad + col as f64 * col_pitch;
                    let y = legend_rect.y + outer_pad + title_y_offset
                        + (row as f64) * row_pitch + line_h / 2.0;
                    let symbol_x = col_origin_x + symbol_width / 2.0;
                    let label_x = col_origin_x + symbol_width + SYMBOL_LABEL_GAP;
                    LegendEntryLayout {
                        label: e.label.clone(),
                        label_anchor_x: label_x,
                        label_anchor_y: y,
                        symbol_anchor_x: symbol_x,
                        symbol_anchor_y: y,
                        symbol_kind: override_symbol.unwrap_or(e.symbol),
                        symbol_radius: None,
                        shape_name: None,
                        color_hex: None,
                    }
                })
                .collect()
        }
        LegendDirection::Horizontal => {
            let avail_w = (legend_rect.w - 2.0 * outer_pad - title_x_offset).max(0.0);
            let max_label_w = entries
                .iter()
                .map(|e| metrics.measure_width(&e.label, label_font_size))
                .fold(0.0_f64, f64::max);
            let entry_w = symbol_width + SYMBOL_LABEL_GAP + max_label_w;
            let pitch = entry_w + entry_pad;
            let max_n = if pitch > 0.0 {
                ((avail_w + entry_pad) / pitch).floor() as usize
            } else {
                0
            };
            let n_fit = entries.len().min(max_n);
            entries
                .iter()
                .take(n_fit)
                .enumerate()
                .map(|(i, e)| {
                    let entry_x = legend_rect.x + outer_pad + title_x_offset
                        + (i as f64) * pitch;
                    let cy = legend_rect.y + legend_rect.h / 2.0;
                    let symbol_x = entry_x + symbol_width / 2.0;
                    let label_x = entry_x + symbol_width + SYMBOL_LABEL_GAP;
                    LegendEntryLayout {
                        label: e.label.clone(),
                        label_anchor_x: label_x,
                        label_anchor_y: cy,
                        symbol_anchor_x: symbol_x,
                        symbol_anchor_y: cy,
                        symbol_kind: override_symbol.unwrap_or(e.symbol),
                        symbol_radius: None,
                        shape_name: None,
                        color_hex: None,
                    }
                })
                .collect()
        }
    };

    let legend = LegendLayout {
        rect: legend_rect,
        orient,
        direction,
        entries: entries_laid_out,
        title: title_layout,
        colorbar: None,
        symbol_stroke_width: opts.symbol_stroke_width,
        clip_height: opts.clip_height,
        symbol_size: opts.symbol_size,
        label_color: opts.label_color,
        label_font_size: opts.label_font_size,
    };
    (Some(legend), plot_inner)
}

pub(crate) const COLORBAR_WIDTH: f64 = 14.0;
pub(crate) const COLORBAR_HEIGHT: f64 = 180.0;
const COLORBAR_TICK_GAP: f64 = 4.0;

/// Thin evenly-spaced colorbar `labels` so adjacent kept ticks are at least
/// `min_step` apart in data space (B5 unit 3 `tick_min_step`). The labels span
/// `(lo, hi)` uniformly, so the current step is `span / (n-1)`; the kept count is
/// the largest `n >= 2` whose step is `>= min_step`. Returns `labels` unchanged
/// when `min_step <= 0`, the span is degenerate, or fewer than 3 labels exist.
pub fn thin_colorbar_ticks_by_min_step(
    labels: Vec<String>,
    domain: (f64, f64),
    min_step: f64,
) -> Vec<String> {
    let n = labels.len();
    if min_step <= 0.0 || n < 3 {
        return labels;
    }
    let span = (domain.1 - domain.0).abs();
    if span == 0.0 || !span.is_finite() {
        return labels;
    }
    // Max intervals whose even spacing is still >= min_step, clamped to [1, n-1].
    let max_intervals = ((span / min_step).floor() as usize).clamp(1, n - 1);
    let keep = max_intervals + 1;
    subsample_tick_labels(labels, keep)
}

/// Subsample `labels` to at most `max_count` evenly-spaced entries, always
/// keeping the first and last. Used when `tickCount` is set on a colorbar.
pub fn subsample_tick_labels(labels: Vec<String>, max_count: usize) -> Vec<String> {
    let n = labels.len();
    if max_count == 0 || n == 0 {
        return Vec::new();
    }
    if max_count >= n {
        return labels;
    }
    if max_count == 1 {
        return vec![labels[0].clone()];
    }
    // Always keep first and last; distribute the rest evenly.
    let mut result = Vec::with_capacity(max_count);
    for i in 0..max_count {
        let idx = (i * (n - 1)) / (max_count - 1);
        result.push(labels[idx].clone());
    }
    result
}

/// Estimate the bounding rect of a continuous colorbar (vertical bar +
/// tick labels). Width: bar width + gap + max tick label.
/// Height: pixel range of the bar plus title (when set) and outer padding.
///
/// `bar_w_override` / `bar_h_override` replace the default constants when
/// `Some` (from `gradientThickness` / `gradientLength` legend kwargs).
pub fn estimate_colorbar_size(
    title: Option<&str>,
    tick_labels: &[String],
    label_font_size: f64,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
    bar_w_override: Option<f64>,
    bar_h_override: Option<f64>,
) -> LegendSize {
    let line_h = metrics.line_height(label_font_size);
    let max_tick_w = tick_labels.iter().map(|s|
        metrics.measure_width(s, label_font_size)).fold(0.0_f64, f64::max);
    let bar_h = bar_h_override.unwrap_or(COLORBAR_HEIGHT);
    let bar_w = bar_w_override.unwrap_or(COLORBAR_WIDTH);
    let mut width = bar_w + COLORBAR_TICK_GAP + max_tick_w + 2.0 * LEGEND_OUTER_PAD;
    let title_h = title.map(|t| {
        let tw = metrics.measure_width(t, title_font_size);
        width = width.max(tw + 2.0 * LEGEND_OUTER_PAD);
        metrics.line_height(title_font_size) + LEGEND_TITLE_GAP
    }).unwrap_or(0.0);
    let height = title_h + bar_h.max(line_h) + 2.0 * LEGEND_OUTER_PAD;
    LegendSize { width, height }
}

/// Build a colorbar legend (continuous color scale variant). Mirrors
/// `layout_legend` but for the gradient/colorbar shape — bar geometry,
/// per-tick y-positions, and title placement.
///
/// `stops` is the gradient definition as (position in [0, 1], hex color);
/// `tick_labels` is the data-domain labels (low → high) used to position
/// tick marks on the bar.
///
/// `gradient_length_override` / `gradient_thickness_override` — when `Some`,
/// replace the default `COLORBAR_HEIGHT` / `COLORBAR_WIDTH` constants.
///
/// `clip_height` (B5 unit 3) — when `Some`, carried onto the `LegendLayout` so
/// the renderer hard-clips the colorbar group to that height.
#[allow(clippy::too_many_arguments)]
pub fn layout_colorbar(
    plot_inner: Rect,
    orient: LegendOrient,
    title: Option<String>,
    stops: Vec<(f64, String)>,
    tick_labels: Vec<String>,
    label_font_size: f64,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
    legend_gutter: f64,
    gradient_length_override: Option<f64>,
    gradient_thickness_override: Option<f64>,
    clip_height: Option<f64>,
    // Per-legend `label_color`/`label_font_size` overrides carried onto the
    // emitted `LegendLayout` so the renderer styles the colorbar tick labels to
    // match the categorical legend path. `None` → theme default (byte-identical).
    label_color: Option<String>,
    label_font_size_override: Option<f64>,
) -> (Option<LegendLayout>, Rect) {
    if stops.is_empty() {
        return (None, plot_inner);
    }
    let effective_bar_h = gradient_length_override.unwrap_or(COLORBAR_HEIGHT).max(1.0);
    let effective_bar_w = gradient_thickness_override.unwrap_or(COLORBAR_WIDTH).max(1.0);
    let size = estimate_colorbar_size(
        title.as_deref(), &tick_labels, label_font_size, title_font_size, metrics,
        Some(effective_bar_w), Some(effective_bar_h),
    );

    // Place the colorbar in the same gutter slot a categorical legend would
    // occupy (right of plot by default).
    let (legend_rect, new_inner) = match orient {
        LegendOrient::Right => {
            let lx = plot_inner.x + plot_inner.w - size.width;
            let rect = Rect { x: lx, y: plot_inner.y, w: size.width, h: size.height.min(plot_inner.h) };
            let inner = Rect {
                x: plot_inner.x,
                y: plot_inner.y,
                w: (plot_inner.w - size.width - legend_gutter).max(0.0),
                h: plot_inner.h,
            };
            (rect, inner)
        }
        LegendOrient::Left => {
            let rect = Rect { x: plot_inner.x, y: plot_inner.y, w: size.width, h: size.height.min(plot_inner.h) };
            let inner = Rect {
                x: plot_inner.x + size.width + legend_gutter,
                y: plot_inner.y,
                w: (plot_inner.w - size.width - legend_gutter).max(0.0),
                h: plot_inner.h,
            };
            (rect, inner)
        }
        LegendOrient::Top => {
            let rect = Rect { x: plot_inner.x, y: plot_inner.y, w: size.width.min(plot_inner.w), h: size.height };
            let inner = Rect {
                x: plot_inner.x,
                y: plot_inner.y + size.height + legend_gutter,
                w: plot_inner.w,
                h: (plot_inner.h - size.height - legend_gutter).max(0.0),
            };
            (rect, inner)
        }
        LegendOrient::Bottom => {
            let ly = plot_inner.y + plot_inner.h - size.height;
            let rect = Rect { x: plot_inner.x, y: ly, w: size.width.min(plot_inner.w), h: size.height };
            let inner = Rect {
                x: plot_inner.x,
                y: plot_inner.y,
                w: plot_inner.w,
                h: (plot_inner.h - size.height - legend_gutter).max(0.0),
            };
            (rect, inner)
        }
    };

    let title_h = title.as_ref()
        .map(|_| metrics.line_height(title_font_size) + LEGEND_TITLE_GAP)
        .unwrap_or(0.0);
    let bar_top = legend_rect.y + LEGEND_OUTER_PAD + title_h;
    let bar_bottom = (bar_top + effective_bar_h).min(legend_rect.y + legend_rect.h - LEGEND_OUTER_PAD);
    let bar_left = legend_rect.x + LEGEND_OUTER_PAD;
    let bar_rect = Rect {
        x: bar_left,
        y: bar_top,
        w: effective_bar_w,
        h: (bar_bottom - bar_top).max(0.0),
    };

    let title_layout = title.map(|text| {
        LegendTitleLayout {
            text,
            x: legend_rect.x + LEGEND_OUTER_PAD,
            y: legend_rect.y + LEGEND_OUTER_PAD + metrics.line_height(title_font_size) * 0.8,
        }
    });

    // Distribute tick y-positions linearly from bar_bottom (= min value)
    // to bar_top (= max value). Cartesian convention: high data → top.
    let n_ticks = tick_labels.len().max(1);
    let ticks: Vec<ColorbarTick> = tick_labels.iter().enumerate().map(|(i, label)| {
        let t = if n_ticks <= 1 { 0.0 } else { i as f64 / (n_ticks - 1) as f64 };
        let y = bar_bottom - t * (bar_bottom - bar_top);
        ColorbarTick { label: label.clone(), y }
    }).collect();

    let legend = LegendLayout {
        rect: legend_rect,
        orient,
        direction: LegendDirection::Vertical,
        entries: Vec::new(),
        title: title_layout,
        colorbar: Some(ColorbarLayout { bar_rect, stops, ticks }),
        symbol_stroke_width: None,
        clip_height,
        symbol_size: None,
        label_color,
        label_font_size: label_font_size_override,
    };
    (Some(legend), new_inner)
}

// ── Auxiliary (size / shape) legends ────────────────────────────────────────

/// A graduated size-legend value: the data value's formatted label and the
/// pixel *radius* its size scale maps to. Built in `prepare.rs` from the size
/// scale's nice round values.
#[derive(Debug, Clone, PartialEq)]
pub struct SizeLegendEntry {
    pub label: String,
    /// Symbol radius in pixels (output of the size scale for this value).
    pub radius: f64,
    /// Optional per-entry swatch color (css hex). Set on a merged color+size
    /// legend so the graduated symbol also varies in color. `None` → theme
    /// mark color.
    pub color_hex: Option<String>,
}

/// A shape-legend category: the formatted label and the point-shape name the
/// shape scale assigned to it (e.g. `"diamond"`).
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeLegendEntry {
    pub label: String,
    pub shape_name: String,
}

/// Input for one auxiliary legend block (size or shape) stacked below the
/// color legend. Built in `prepare.rs`; consumed by [`layout_aux_legends`].
#[derive(Debug, Clone, PartialEq)]
pub enum AuxLegendInput {
    Size { title: Option<String>, entries: Vec<SizeLegendEntry> },
    Shape { title: Option<String>, entries: Vec<ShapeLegendEntry> },
}

/// The optional title of an aux legend block, regardless of variant.
fn aux_title(input: &AuxLegendInput) -> Option<&str> {
    match input {
        AuxLegendInput::Size { title, .. } | AuxLegendInput::Shape { title, .. } =>
            title.as_deref(),
    }
}

/// Estimate the (width, height) of one auxiliary legend block.
fn estimate_aux_block_size(
    input: &AuxLegendInput,
    label_font_size: f64,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
) -> LegendSize {
    let line_h = metrics.line_height(label_font_size);
    let (labels, max_symbol_w): (Vec<&str>, f64) = match input {
        AuxLegendInput::Size { entries, .. } => {
            let w = entries.iter().map(|e| 2.0 * e.radius).fold(SYMBOL_WIDTH, f64::max);
            (entries.iter().map(|e| e.label.as_str()).collect(), w)
        }
        AuxLegendInput::Shape { entries, .. } => (
            entries.iter().map(|e| e.label.as_str()).collect(),
            SYMBOL_WIDTH,
        ),
    };
    let max_label_w = labels
        .iter()
        .map(|l| metrics.measure_width(l, label_font_size))
        .fold(0.0_f64, f64::max);
    let n = labels.len() as f64;
    // Per-row pitch must clear the tallest symbol in a size legend.
    let row_h = match input {
        AuxLegendInput::Size { entries, .. } => entries
            .iter()
            .map(|e| 2.0 * e.radius)
            .fold(line_h, f64::max),
        AuxLegendInput::Shape { .. } => line_h,
    };
    let entry_w = max_symbol_w + SYMBOL_LABEL_GAP + max_label_w;
    let width = entry_w + 2.0 * LEGEND_OUTER_PAD;
    let entries_h = if labels.is_empty() {
        0.0
    } else {
        n * row_h + (n - 1.0).max(0.0) * LEGEND_ENTRY_ROW_PAD
    };
    let title_h = aux_title(input)
        .map(|_| metrics.line_height(title_font_size) + LEGEND_TITLE_GAP)
        .unwrap_or(0.0);
    let height = title_h + entries_h + 2.0 * LEGEND_OUTER_PAD;
    LegendSize { width, height }
}

/// The y-pixel at the bottom of a color legend's drawn content (last entry row
/// or colorbar + its lowest tick), used to anchor stacked aux blocks. Falls
/// back to the rect top when the legend has neither entries nor a colorbar.
fn color_legend_content_bottom(
    legend: &LegendLayout,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    let line_h = metrics.line_height(label_font_size);
    let mut bottom = legend.rect.y + LEGEND_OUTER_PAD;
    for e in &legend.entries {
        bottom = bottom.max(e.symbol_anchor_y + line_h / 2.0);
    }
    if let Some(cb) = &legend.colorbar {
        bottom = bottom.max(cb.bar_rect.y + cb.bar_rect.h);
        for t in &cb.ticks {
            bottom = bottom.max(t.y + line_h / 2.0);
        }
    }
    bottom
}

/// Lay out auxiliary (size / shape) legend blocks, stacked vertically beneath
/// the color legend in the same gutter. Returns the laid-out blocks plus a
/// possibly-reduced `inner_after_legend` rect: if the widest aux block exceeds
/// the color block's reserved width, the plot region shrinks to accommodate it.
///
/// `color_legend` is the already-placed color legend (`None` when only
/// size/shape are encoded). `inner_pre_legend` is the rect *before* any legend
/// reservation (used to compute the gutter when there is no color block).
/// `inner_after_legend` is the plot rect remaining after the color reservation.
///
/// Blocks stack from the bottom of the color block (or the top of the gutter
/// when there is no color block) downward. Each block is left-aligned with the
/// color block's gutter `x`.
#[allow(clippy::too_many_arguments)]
pub fn layout_aux_legends(
    inputs: &[AuxLegendInput],
    color_legend: Option<&LegendLayout>,
    orient: LegendOrient,
    inner_pre_legend: Rect,
    inner_after_legend: Rect,
    label_font_size: f64,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
    legend_gutter: f64,
) -> (Vec<LegendLayout>, Rect) {
    if inputs.is_empty() {
        return (Vec::new(), inner_after_legend);
    }

    // Determine the gutter x-origin and the y at which to start stacking.
    // With a color legend present, reuse its rect; otherwise carve a fresh
    // strip from the pre-legend inner on the requested orient.
    let (gutter_x, mut cursor_y, mut new_inner, color_w) =
        if let Some(cl) = color_legend {
            // Stack starts below the color block's *content* (its entries or
            // colorbar), not its full-height gutter rect — otherwise the aux
            // blocks land at the bottom of the viewport and clip.
            let content_bottom = color_legend_content_bottom(cl, label_font_size, metrics);
            (cl.rect.x, content_bottom, inner_after_legend, cl.rect.w)
        } else {
            // No color legend: place the aux stack in the gutter on `orient`.
            // Reserve a provisional width (the widest aux block) from the
            // pre-legend inner.
            let max_w = inputs
                .iter()
                .map(|i| estimate_aux_block_size(i, label_font_size, title_font_size, metrics).width)
                .fold(0.0_f64, f64::max);
            const LEGEND_PLOT_GAP: f64 = 8.0;
            let (gutter_x, new_inner) = match orient {
                LegendOrient::Right | LegendOrient::Top | LegendOrient::Bottom => {
                    let w = max_w.min(inner_pre_legend.w * 0.5);
                    let (_strip, rest) = inner_pre_legend.split_right(w + LEGEND_PLOT_GAP);
                    (inner_pre_legend.x + inner_pre_legend.w - w, rest)
                }
                LegendOrient::Left => {
                    let w = max_w.min(inner_pre_legend.w * 0.5);
                    let (strip, rest) = inner_pre_legend.split_left(w + LEGEND_PLOT_GAP);
                    (strip.x, rest)
                }
            };
            (gutter_x, inner_pre_legend.y, new_inner, max_w)
        };

    let mut blocks: Vec<LegendLayout> = Vec::with_capacity(inputs.len());
    let mut max_block_w = color_w;
    // Gap between stacked legend blocks.
    const AUX_BLOCK_GAP: f64 = 12.0;

    for input in inputs {
        let size = estimate_aux_block_size(input, label_font_size, title_font_size, metrics);
        max_block_w = max_block_w.max(size.width);
        let block_rect = Rect {
            x: gutter_x,
            y: cursor_y + AUX_BLOCK_GAP,
            w: size.width,
            h: size.height,
        };
        let block = layout_aux_block(input, block_rect, label_font_size, title_font_size, metrics);
        cursor_y = block_rect.y + block_rect.h;
        blocks.push(block);
    }

    // If the widest aux block exceeds the color block, shrink the plot region.
    if max_block_w > color_w {
        let extra = max_block_w - color_w;
        new_inner = match orient {
            LegendOrient::Right | LegendOrient::Top | LegendOrient::Bottom => Rect {
                x: new_inner.x,
                y: new_inner.y,
                w: (new_inner.w - extra - legend_gutter.min(0.0)).max(0.0),
                h: new_inner.h,
            },
            LegendOrient::Left => Rect {
                x: new_inner.x + extra,
                y: new_inner.y,
                w: (new_inner.w - extra).max(0.0),
                h: new_inner.h,
            },
        };
    }

    (blocks, new_inner)
}

/// Lay out a single aux block (size or shape) within `block_rect`: title at
/// top, then one row per entry. Returns a `LegendLayout` whose `entries` carry
/// `symbol_radius` (size) or `shape_name` (shape) so the renderer draws the
/// graduated / shaped glyph.
fn layout_aux_block(
    input: &AuxLegendInput,
    block_rect: Rect,
    label_font_size: f64,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
) -> LegendLayout {
    let line_h = metrics.line_height(label_font_size);
    let title_h = aux_title(input)
        .map(|_| metrics.line_height(title_font_size))
        .unwrap_or(0.0);
    let title_layout = aux_title(input).map(|text| LegendTitleLayout {
        text: text.to_string(),
        x: block_rect.x + LEGEND_OUTER_PAD,
        y: block_rect.y + LEGEND_OUTER_PAD + title_h,
    });
    let title_offset = if title_layout.is_some() {
        title_h + LEGEND_TITLE_GAP
    } else {
        0.0
    };

    // The largest symbol radius (size legend) sets the symbol column width so
    // labels clear the biggest circle.
    let max_radius = match input {
        AuxLegendInput::Size { entries, .. } =>
            entries.iter().map(|e| e.radius).fold(SYMBOL_WIDTH / 2.0, f64::max),
        AuxLegendInput::Shape { .. } => SYMBOL_WIDTH / 2.0,
    };
    let symbol_col_x = block_rect.x + LEGEND_OUTER_PAD + max_radius;
    let label_x = block_rect.x + LEGEND_OUTER_PAD + 2.0 * max_radius + SYMBOL_LABEL_GAP;

    let mut entries: Vec<LegendEntryLayout> = Vec::new();
    let mut row_top = block_rect.y + LEGEND_OUTER_PAD + title_offset;

    match input {
        AuxLegendInput::Size { entries: ses, .. } => {
            for e in ses {
                let row_h = (2.0 * e.radius).max(line_h);
                let cy = row_top + row_h / 2.0;
                entries.push(LegendEntryLayout {
                    label: e.label.clone(),
                    label_anchor_x: label_x,
                    label_anchor_y: cy + label_font_size * 0.35,
                    symbol_anchor_x: symbol_col_x,
                    symbol_anchor_y: cy,
                    symbol_kind: SymbolKind::Circle,
                    symbol_radius: Some(e.radius),
                    shape_name: None,
                    color_hex: e.color_hex.clone(),
                });
                row_top += row_h + LEGEND_ENTRY_ROW_PAD;
            }
        }
        AuxLegendInput::Shape { entries: shes, .. } => {
            for e in shes {
                let cy = row_top + line_h / 2.0;
                entries.push(LegendEntryLayout {
                    label: e.label.clone(),
                    label_anchor_x: label_x,
                    label_anchor_y: cy + label_font_size * 0.35,
                    symbol_anchor_x: symbol_col_x,
                    symbol_anchor_y: cy,
                    symbol_kind: SymbolKind::Circle,
                    symbol_radius: None,
                    shape_name: Some(e.shape_name.clone()),
                    color_hex: None,
                });
                row_top += line_h + LEGEND_ENTRY_ROW_PAD;
            }
        }
    }

    LegendLayout {
        rect: block_rect,
        orient: LegendOrient::Right,
        direction: LegendDirection::Vertical,
        entries,
        title: title_layout,
        colorbar: None,
        symbol_stroke_width: None,
        clip_height: None,
        symbol_size: None,
        label_color: None,
        label_font_size: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_layout_round_trip() {
        let l = LegendLayout {
            rect: Rect { x: 500.0, y: 50.0, w: 100.0, h: 300.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![LegendEntryLayout {
                label: "A".into(),
                label_anchor_x: 520.0,
                label_anchor_y: 70.0,
                symbol_anchor_x: 510.0,
                symbol_anchor_y: 70.0,
                symbol_kind: SymbolKind::Circle,
                symbol_radius: None,
                shape_name: None,
                color_hex: None,
            }],
            title: None,
            colorbar: None,
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        };
        let json = serde_json::to_string(&l).unwrap();
        let parsed: LegendLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, l);
        assert!(json.contains(r#""orient":"right""#));
        assert!(json.contains(r#""symbol_kind":"circle""#));
    }

    use crate::layout::text_metrics::{fixed_width, MockMetrics};

    fn mock(per_char_px: f64) -> MockMetrics<impl Fn(&str, f64) -> f64> {
        MockMetrics { measure: fixed_width(per_char_px), line_h_factor: 1.2 }
    }

    fn entries(n: usize, label_chars: usize) -> Vec<LegendEntry> {
        (0..n)
            .map(|i| LegendEntry {
                label: "X".repeat(label_chars).chars().chain(format!("{i}").chars()).collect(),
                symbol: SymbolKind::Circle,
            })
            .collect()
    }

    #[test]
    fn legend_size_right_orient_vertical() {
        let es = vec![
            LegendEntry { label: "abcd".into(), symbol: SymbolKind::Circle },
            LegendEntry { label: "abcdef".into(), symbol: SymbolKind::Circle },
        ];
        let m = mock(10.0);
        let size = estimate_legend_size(
            &es, LegendOrient::Right, 11.0, &m, LEGEND_ENTRY_ROW_PAD, LEGEND_OUTER_PAD, SYMBOL_WIDTH,
        );
        // max_label_w = 6 * 10 = 60; symbol_w + sep = 12 + 4 = 16; outer pad 8+8=16.
        // total width = 16 + 60 + 16 = 92. height = 2*line_h + 1*row_pad + 2*outer_pad
        //   = 2*13.2 + 4.0 + 16.0 = 46.4. line_h = 11*1.2 = 13.2.
        assert!((size.width - 92.0).abs() < 1e-6);
        assert!((size.height - (2.0 * 13.2 + 4.0 + 16.0)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_right_does_not_overlap_inner() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None, LegendStyleOpts::default());
        let legend = legend.expect("legend should be Some");
        assert_eq!(legend.orient, LegendOrient::Right);
        assert_eq!(legend.direction, LegendDirection::Vertical);
        // Legend rect starts LEGEND_PLOT_GAP pixels to the right of the plot area.
        assert!((legend.rect.x - (plot_inner.x + plot_inner.w) - 8.0).abs() < 1e-6);
        assert!(plot_inner.w < inner.w);
        assert_eq!(plot_inner.h, inner.h);
    }

    #[test]
    fn legend_layout_left_orient() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Left, inner, 11.0, &m, None, None, 13.0, None, None, LegendStyleOpts::default());
        let legend = legend.unwrap();
        assert_eq!(legend.rect.x, inner.x);
        // plot area starts LEGEND_PLOT_GAP to the right of the legend rect end.
        assert!((plot_inner.x - (inner.x + legend.rect.w) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_top_orient_horizontal() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Top, inner, 11.0, &m, None, None, 13.0, None, None, LegendStyleOpts::default());
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert_eq!(legend.rect.y, inner.y);
        assert!((plot_inner.y - (inner.y + legend.rect.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_bottom_orient_horizontal() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Bottom, inner, 11.0, &m, None, None, 13.0, None, None, LegendStyleOpts::default());
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert!((legend.rect.y - (inner.y + plot_inner.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_empty_entries_returns_none_and_inner_unchanged() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&[], LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None, LegendStyleOpts::default());
        assert!(legend.is_none());
        assert_eq!(plot_inner, inner);
    }

    #[test]
    fn legend_layout_overflow_drops_entries() {
        let inner = Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 };
        let es = entries(50, 4);
        let m = mock(10.0);
        let (legend, _) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None, LegendStyleOpts::default());
        let legend = legend.unwrap();
        assert!(legend.entries.len() < 50, "expected overflow drop; got {} entries", legend.entries.len());
    }

    #[test]
    fn legend_layout_symbol_type_override_circle() {
        // Default symbol is Circle for all entries — override to Square should change all.
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let mut es = entries(3, 4);
        // Set all to Circle by default.
        for e in &mut es { e.symbol = SymbolKind::Circle; }
        let m = mock(10.0);
        let (legend, _) = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m,
            None, None, 13.0, None,
            Some("square"), // symbol_type_override
            LegendStyleOpts::default(),
        );
        let legend = legend.unwrap();
        for entry in &legend.entries {
            assert_eq!(
                entry.symbol_kind, SymbolKind::Square,
                "expected Square from symbol_type override, got {:?}", entry.symbol_kind,
            );
        }
    }

    #[test]
    fn legend_layout_symbol_type_override_none_preserves_entry_symbol() {
        // When no override, each entry keeps its own symbol.
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let mut es = entries(2, 4);
        es[0].symbol = SymbolKind::Circle;
        es[1].symbol = SymbolKind::Square;
        let m = mock(10.0);
        let (legend, _) = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m,
            None, None, 13.0, None,
            None, // no override
            LegendStyleOpts::default(),
        );
        let legend = legend.unwrap();
        assert_eq!(legend.entries[0].symbol_kind, SymbolKind::Circle);
        assert_eq!(legend.entries[1].symbol_kind, SymbolKind::Square);
    }

    // ── B5 unit 3: orphan legend styling/positioning fields ──────────────

    /// `row_padding` widens the vertical gap between entry rows, so the second
    /// entry's anchor lands lower than with the default pad.
    #[test]
    fn legend_row_padding_increases_vertical_spacing() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let base = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let padded = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { row_padding: Some(20.0), ..Default::default() },
        ).0.unwrap();
        let base_gap = base.entries[1].symbol_anchor_y - base.entries[0].symbol_anchor_y;
        let padded_gap = padded.entries[1].symbol_anchor_y - padded.entries[0].symbol_anchor_y;
        assert!(
            padded_gap > base_gap,
            "row_padding=20 must widen the row gap: base={base_gap}, padded={padded_gap}",
        );
    }

    /// `column_padding` widens the horizontal gap between entries in a
    /// horizontal-direction legend.
    #[test]
    fn legend_column_padding_increases_horizontal_spacing() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let base = layout_legend(
            &es, LegendOrient::Top, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let padded = layout_legend(
            &es, LegendOrient::Top, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { column_padding: Some(30.0), ..Default::default() },
        ).0.unwrap();
        let base_gap = base.entries[1].symbol_anchor_x - base.entries[0].symbol_anchor_x;
        let padded_gap = padded.entries[1].symbol_anchor_x - padded.entries[0].symbol_anchor_x;
        assert!(
            padded_gap > base_gap,
            "column_padding=30 must widen the column gap: base={base_gap}, padded={padded_gap}",
        );
    }

    /// `label_limit` truncates a long label to an ellipsis and shrinks the rect.
    #[test]
    fn legend_label_limit_truncates_with_ellipsis() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = vec![LegendEntry {
            label: "a-very-long-category-label".into(),
            symbol: SymbolKind::Circle,
        }];
        let m = mock(10.0); // 10 px per char
        let base = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let limited = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { label_limit: Some(40.0), ..Default::default() },
        ).0.unwrap();
        assert!(
            limited.entries[0].label.ends_with('…'),
            "truncated label must end with an ellipsis, got {:?}",
            limited.entries[0].label,
        );
        assert!(
            limited.entries[0].label.chars().count() < es[0].label.chars().count(),
            "truncated label must be shorter than the original",
        );
        assert!(
            limited.rect.w < base.rect.w,
            "label_limit must shrink the legend rect width: base={}, limited={}",
            base.rect.w, limited.rect.w,
        );
    }

    /// A label already within `label_limit` is untouched (no ellipsis).
    #[test]
    fn legend_label_limit_keeps_short_label() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = vec![LegendEntry { label: "ab".into(), symbol: SymbolKind::Circle }];
        let m = mock(10.0);
        let limited = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { label_limit: Some(200.0), ..Default::default() },
        ).0.unwrap();
        assert_eq!(limited.entries[0].label, "ab", "short label must be untouched");
    }

    /// `clip_height` is carried onto the produced `LegendLayout` for the renderer.
    #[test]
    fn legend_clip_height_carried_onto_layout() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let legend = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { clip_height: Some(50.0), ..Default::default() },
        ).0.unwrap();
        assert_eq!(legend.clip_height, Some(50.0));
    }

    /// `symbol_stroke_width` is carried onto the produced `LegendLayout`.
    #[test]
    fn legend_symbol_stroke_width_carried_onto_layout() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(2, 4);
        let m = mock(10.0);
        let legend = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { symbol_stroke_width: Some(3.0), ..Default::default() },
        ).0.unwrap();
        assert_eq!(legend.symbol_stroke_width, Some(3.0));
    }

    // ── B5 unit 6a: offset / padding / title_padding / multi-col column_padding ──

    /// `offset` widens the gap between the plot area and the legend strip. For a
    /// right-orient legend (pinned to the viewport's right edge) the legend rect
    /// is unchanged but the plot area shrinks by the extra offset, and the gap
    /// between the plot's right edge and the legend's left edge grows by it.
    #[test]
    fn legend_offset_widens_plot_to_legend_gap() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (base, base_plot) = {
            let (l, p) = layout_legend(
                &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
                LegendStyleOpts::default(),
            );
            (l.unwrap(), p)
        };
        let (offset, offset_plot) = {
            let (l, p) = layout_legend(
                &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
                LegendStyleOpts { offset: Some(50.0), ..Default::default() },
            );
            (l.unwrap(), p)
        };
        // The legend strip stays pinned to the viewport's right edge (same width).
        assert!((offset.rect.w - base.rect.w).abs() < 1e-9, "legend width unchanged by offset");
        // The plot area shrinks by the extra offset.
        assert!(
            (base_plot.w - offset_plot.w - 50.0).abs() < 1e-9,
            "offset=50 must shrink the plot width by 50: base={}, offset={}",
            base_plot.w, offset_plot.w,
        );
        // The gap between the plot's right edge and the legend's left edge grows
        // by the offset (8px base gap → 58px).
        let base_gap = base.rect.x - (base_plot.x + base_plot.w);
        let offset_gap = offset.rect.x - (offset_plot.x + offset_plot.w);
        assert!(
            (offset_gap - base_gap - 50.0).abs() < 1e-9,
            "offset=50 must widen the plot→legend gap by 50: base={base_gap}, offset={offset_gap}",
        );
    }

    /// `padding` widens the inner inset, growing the rect and pushing the first
    /// entry's symbol anchor further from the rect edge.
    #[test]
    fn legend_padding_insets_contents_and_grows_rect() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let base = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let padded = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { padding: Some(30.0), ..Default::default() },
        ).0.unwrap();
        // Larger padding → wider legend rect (entry_w + 2*padding).
        assert!(
            padded.rect.w > base.rect.w,
            "padding=30 must widen the legend rect: base={}, padded={}",
            base.rect.w, padded.rect.w,
        );
        // First entry symbol is inset by `padding` from the rect's left edge.
        let base_inset = base.entries[0].symbol_anchor_x - SYMBOL_WIDTH / 2.0 - base.rect.x;
        let padded_inset = padded.entries[0].symbol_anchor_x - SYMBOL_WIDTH / 2.0 - padded.rect.x;
        assert!((base_inset - LEGEND_OUTER_PAD).abs() < 1e-9, "default inset = LEGEND_OUTER_PAD");
        assert!((padded_inset - 30.0).abs() < 1e-9, "padded inset = padding override");
    }

    /// `title_padding` widens the gap between the title and the first entry row,
    /// pushing the first entry lower (with a title present).
    #[test]
    fn legend_title_padding_increases_title_to_entry_gap() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let base = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, Some("Title"), 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let padded = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, Some("Title"), 13.0, None, None,
            LegendStyleOpts { title_padding: Some(25.0), ..Default::default() },
        ).0.unwrap();
        // The title sits at the same place; the first entry drops by the extra gap.
        assert!(
            padded.entries[0].symbol_anchor_y > base.entries[0].symbol_anchor_y,
            "title_padding=25 must push the first entry lower: base={}, padded={}",
            base.entries[0].symbol_anchor_y, padded.entries[0].symbol_anchor_y,
        );
    }

    /// In a multi-column vertical legend, `column_padding` widens the gap between
    /// columns so the second column's symbol anchor lands further right.
    #[test]
    fn legend_column_padding_widens_multi_column_gap() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(4, 4);
        let m = mock(10.0);
        let base = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, Some(2), None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let padded = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, Some(2), None,
            LegendStyleOpts { column_padding: Some(40.0), ..Default::default() },
        ).0.unwrap();
        // Entry 0 → col 0, entry 1 → col 1 (row-major). The inter-column gap is
        // the difference of their symbol anchor x.
        let base_gap = base.entries[1].symbol_anchor_x - base.entries[0].symbol_anchor_x;
        let padded_gap = padded.entries[1].symbol_anchor_x - padded.entries[0].symbol_anchor_x;
        assert!(
            padded_gap > base_gap,
            "column_padding=40 must widen the inter-column gap: base={base_gap}, padded={padded_gap}",
        );
    }

    /// `symbol_size` widens the reserved symbol column so the swatch clears the
    /// label, pushing the label anchor further right.
    #[test]
    fn legend_symbol_size_widens_symbol_column() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let base = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        let big = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { symbol_size: Some(400.0), ..Default::default() },
        ).0.unwrap();
        let base_col = base.entries[0].label_anchor_x - base.entries[0].symbol_anchor_x;
        let big_col = big.entries[0].label_anchor_x - big.entries[0].symbol_anchor_x;
        assert!(
            big_col > base_col,
            "symbol_size=400 must widen the symbol→label gap: base={base_col}, big={big_col}",
        );
        // symbol_size/label_color are carried onto the layout for the renderer.
        assert_eq!(big.symbol_size, Some(400.0));
    }

    /// `label_color` is carried onto the produced `LegendLayout` for the renderer
    /// (layout geometry unaffected).
    #[test]
    fn legend_label_color_carried_onto_layout() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(2, 4);
        let m = mock(10.0);
        let legend = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts { label_color: Some("#ff0000".into()), ..Default::default() },
        ).0.unwrap();
        assert_eq!(legend.label_color.as_deref(), Some("#ff0000"));
    }

    /// Colorbar legends must carry the per-legend `label_color` /
    /// `label_font_size` overrides onto the emitted `LegendLayout` (mirroring the
    /// categorical path) so the renderer styles the colorbar tick labels. The bug
    /// hardcoded both to `None`, silently dropping them on the continuous path.
    #[test]
    fn colorbar_carries_label_color_and_font_size_onto_layout() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(10.0);
        let stops = vec![(0.0, "#0000ff".to_string()), (1.0, "#ff0000".to_string())];
        let labels = vec!["0".into(), "50".into(), "100".into()];
        let (legend, _inner) = layout_colorbar(
            inner, LegendOrient::Right, None, stops, labels, 11.0, 13.0, &m, 13.0,
            None, None, None,
            Some("#ff0000".into()), Some(18.0),
        );
        let legend = legend.expect("colorbar layout");
        assert_eq!(legend.label_color.as_deref(), Some("#ff0000"));
        assert_eq!(legend.label_font_size, Some(18.0));
        // Defaults still produce None on both fields (byte-identical path).
        let (default_legend, _) = layout_colorbar(
            inner, LegendOrient::Right, None,
            vec![(0.0, "#0000ff".to_string()), (1.0, "#ff0000".to_string())],
            vec!["0".into(), "100".into()], 11.0, 13.0, &m, 13.0,
            None, None, None, None, None,
        );
        let default_legend = default_legend.expect("default colorbar layout");
        assert_eq!(default_legend.label_color, None);
        assert_eq!(default_legend.label_font_size, None);
    }

    /// `LegendStyleOpts::default()` reproduces the historical layout — same rect
    /// and same entry anchors as the pre-orphan code path.
    #[test]
    fn legend_default_opts_unchanged() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let legend = layout_legend(
            &es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None, None,
            LegendStyleOpts::default(),
        ).0.unwrap();
        // No orphan fields set → both new layout fields are None.
        assert_eq!(legend.symbol_stroke_width, None);
        assert_eq!(legend.clip_height, None);
        // The first row anchor matches the historical formula
        // (legend.y + outer_pad + line_h/2).
        let line_h = m.line_height(11.0);
        let expected_y = legend.rect.y + LEGEND_OUTER_PAD + line_h / 2.0;
        assert!((legend.entries[0].symbol_anchor_y - expected_y).abs() < 1e-9);
    }

    /// `thin_colorbar_ticks_by_min_step` drops ticks closer than `min_step`.
    #[test]
    fn colorbar_tick_min_step_thins_ticks() {
        // 5 ticks across [0, 100] → data step 25. A min_step of 40 allows only
        // floor(100/40)=2 intervals → 3 ticks.
        let labels = vec!["0".into(), "25".into(), "50".into(), "75".into(), "100".into()];
        let thinned = thin_colorbar_ticks_by_min_step(labels.clone(), (0.0, 100.0), 40.0);
        assert_eq!(thinned.len(), 3, "min_step=40 over span 100 → 3 ticks");
        assert_eq!(thinned.first().map(String::as_str), Some("0"));
        assert_eq!(thinned.last().map(String::as_str), Some("100"));
        // A min_step smaller than the natural step keeps all 5.
        let kept = thin_colorbar_ticks_by_min_step(labels.clone(), (0.0, 100.0), 10.0);
        assert_eq!(kept.len(), 5, "min_step below natural step keeps all ticks");
    }

    // ── Multivariate B1: auxiliary (size / shape) legend layout ──────────

    fn size_input(n: usize) -> AuxLegendInput {
        AuxLegendInput::Size {
            title: Some("pop".into()),
            entries: (0..n)
                .map(|i| SizeLegendEntry {
                    label: format!("{}", (i + 1) * 20),
                    radius: 3.0 + i as f64 * 4.0,
                    color_hex: None,
                })
                .collect(),
        }
    }

    fn shape_input(n: usize) -> AuxLegendInput {
        AuxLegendInput::Shape {
            title: Some("region".into()),
            entries: (0..n)
                .map(|i| ShapeLegendEntry {
                    label: format!("cat{i}"),
                    shape_name: "circle".into(),
                })
                .collect(),
        }
    }

    /// A color legend rect to anchor aux blocks beneath.
    fn color_legend_rect() -> LegendLayout {
        LegendLayout {
            rect: Rect { x: 500.0, y: 0.0, w: 80.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![],
            title: None,
            colorbar: None,
            symbol_stroke_width: None,
            clip_height: None,
            symbol_size: None,
            label_color: None,
            label_font_size: None,
        }
    }

    #[test]
    fn aux_legend_empty_is_noop() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(8.0);
        let (blocks, new_inner) = layout_aux_legends(
            &[], None, LegendOrient::Right, inner, inner, 11.0, 13.0, &m, 13.0,
        );
        assert!(blocks.is_empty());
        assert_eq!(new_inner, inner, "no aux legends → plot region unchanged");
    }

    #[test]
    fn aux_size_block_has_five_graduated_entries() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let after = Rect { x: 0.0, y: 0.0, w: 480.0, h: 400.0 };
        let m = mock(8.0);
        let cl = color_legend_rect();
        let (blocks, _) = layout_aux_legends(
            &[size_input(5)], Some(&cl), LegendOrient::Right, inner, after, 11.0, 13.0, &m, 13.0,
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].entries.len(), 5, "five graduated size entries");
        for e in &blocks[0].entries {
            assert!(e.symbol_radius.is_some(), "size entry must carry a radius");
        }
    }

    #[test]
    fn aux_blocks_stack_below_color_and_do_not_overlap() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let after = Rect { x: 0.0, y: 0.0, w: 480.0, h: 400.0 };
        let m = mock(8.0);
        let cl = color_legend_rect();
        let (blocks, _) = layout_aux_legends(
            &[size_input(4), shape_input(3)],
            Some(&cl), LegendOrient::Right, inner, after, 11.0, 13.0, &m, 13.0,
        );
        assert_eq!(blocks.len(), 2, "two stacked blocks (size, shape)");
        // First block starts below the color legend's content top.
        assert!(blocks[0].rect.y >= cl.rect.y, "size block below color legend top");
        // Second block starts below the first (no vertical overlap).
        let first_bottom = blocks[0].rect.y + blocks[0].rect.h;
        assert!(
            blocks[1].rect.y >= first_bottom,
            "shape block (y={}) must start below size block bottom ({})",
            blocks[1].rect.y, first_bottom
        );
    }

    #[test]
    fn aux_legend_without_color_reserves_gutter() {
        // Size-only chart: no color legend. The aux stack must carve its own
        // gutter and shrink the plot region.
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(8.0);
        let (blocks, new_inner) = layout_aux_legends(
            &[size_input(5)], None, LegendOrient::Right, inner, inner, 11.0, 13.0, &m, 13.0,
        );
        assert_eq!(blocks.len(), 1);
        assert!(new_inner.w < inner.w, "size-only legend must reserve gutter width");
    }
}
