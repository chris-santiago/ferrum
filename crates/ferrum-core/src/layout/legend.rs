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
}

const SYMBOL_WIDTH: f64 = 12.0;
const SYMBOL_LABEL_GAP: f64 = 4.0;
const LEGEND_OUTER_PAD: f64 = 8.0;
const LEGEND_ENTRY_ROW_PAD: f64 = 4.0;
const LEGEND_TITLE_GAP: f64 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendSize {
    pub width: f64,
    pub height: f64,
}

/// Title-aware version of `estimate_legend_size`. For Vertical-direction
/// legends, adds a title line to the height. For Horizontal, adds title
/// width to the left. Falls through to the no-title shape when title=None.
pub fn estimate_legend_size_with_title(
    entries: &[LegendEntry],
    orient: LegendOrient,
    label_font_size: f64,
    title: Option<&str>,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
) -> LegendSize {
    let base = estimate_legend_size(entries, orient, label_font_size, metrics);
    let Some(title_text) = title else { return base };
    let title_h = metrics.line_height(title_font_size);
    let title_w = metrics.measure_width(title_text, title_font_size);
    match orient {
        LegendOrient::Right | LegendOrient::Left => LegendSize {
            width: base.width.max(title_w + 2.0 * LEGEND_OUTER_PAD),
            height: base.height + title_h + LEGEND_TITLE_GAP,
        },
        LegendOrient::Top | LegendOrient::Bottom => LegendSize {
            width: base.width + title_w + LEGEND_TITLE_GAP,
            height: base.height.max(title_h + 2.0 * LEGEND_OUTER_PAD),
        },
    }
}

pub fn estimate_legend_size(
    entries: &[LegendEntry],
    orient: LegendOrient,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
) -> LegendSize {
    let line_h = metrics.line_height(label_font_size);
    let max_label_w = entries
        .iter()
        .map(|e| metrics.measure_width(&e.label, label_font_size))
        .fold(0.0_f64, f64::max);
    let n = entries.len() as f64;

    match orient {
        LegendOrient::Right | LegendOrient::Left => {
            let entry_w = SYMBOL_WIDTH + SYMBOL_LABEL_GAP + max_label_w;
            let width = entry_w + 2.0 * LEGEND_OUTER_PAD;
            let height = if entries.is_empty() {
                0.0
            } else {
                n * line_h + (n - 1.0).max(0.0) * LEGEND_ENTRY_ROW_PAD + 2.0 * LEGEND_OUTER_PAD
            };
            LegendSize { width, height }
        }
        LegendOrient::Top | LegendOrient::Bottom => {
            let entry_w = SYMBOL_WIDTH + SYMBOL_LABEL_GAP + max_label_w;
            let width = if entries.is_empty() {
                0.0
            } else {
                n * entry_w + (n - 1.0).max(0.0) * LEGEND_ENTRY_ROW_PAD + 2.0 * LEGEND_OUTER_PAD
            };
            let height = line_h + 2.0 * LEGEND_OUTER_PAD;
            LegendSize { width, height }
        }
    }
}

/// Compute the legend rect on the given orient and the remaining inner rect for
/// the rest of the chart. Returns `(None, inner)` for empty entries.
/// Returns `(Some(legend_with_dropped_entries), reduced_inner)` if the legend
/// overflows the available strip.
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
) -> (Option<LegendLayout>, Rect) {
    if entries.is_empty() {
        return (None, inner);
    }
    let size = estimate_legend_size_with_title(
        entries, orient, label_font_size, title, title_font_size, metrics,
    );

    // Add a small gap between the plot area and the legend strip so data
    // points near the axis edge don't visually merge with legend text.
    const LEGEND_PLOT_GAP: f64 = 8.0;

    let (legend_rect, plot_inner) = match orient {
        LegendOrient::Right => {
            let w = size.width.min(inner.w * 0.5);
            let (strip_with_gap, rest) = inner.split_right(w + LEGEND_PLOT_GAP);
            // Inset the legend rect by the gap on its left side.
            let legend_rect = crate::layout::Rect {
                x: strip_with_gap.x + LEGEND_PLOT_GAP,
                y: strip_with_gap.y,
                w: strip_with_gap.w - LEGEND_PLOT_GAP,
                h: strip_with_gap.h,
            };
            (legend_rect, rest)
        }
        LegendOrient::Left => {
            let w = size.width.min(inner.w * 0.5);
            let (strip_with_gap, rest) = inner.split_left(w + LEGEND_PLOT_GAP);
            let legend_rect = crate::layout::Rect {
                x: strip_with_gap.x,
                y: strip_with_gap.y,
                w: strip_with_gap.w - LEGEND_PLOT_GAP,
                h: strip_with_gap.h,
            };
            (legend_rect, rest)
        }
        LegendOrient::Top => {
            let h = size.height.min(inner.h * 0.5);
            let (strip, rest) = inner.split_top(h);
            (strip, rest)
        }
        LegendOrient::Bottom => {
            let h = size.height.min(inner.h * 0.5);
            let (strip, rest) = inner.split_bottom(h);
            (strip, rest)
        }
    };

    let direction = direction_override.unwrap_or_else(|| match orient {
        LegendOrient::Right | LegendOrient::Left => LegendDirection::Vertical,
        LegendOrient::Top | LegendOrient::Bottom => LegendDirection::Horizontal,
    });

    let line_h = metrics.line_height(label_font_size);

    // Title placement: above entries (Vertical) or to the left (Horizontal).
    // y_offset / x_offset values push entry layout to start past the title.
    let (title_layout, title_y_offset, title_x_offset) = if let Some(title_text) = title {
        let title_h = metrics.line_height(title_font_size);
        let title_w = metrics.measure_width(title_text, title_font_size);
        match direction {
            LegendDirection::Vertical => {
                // Title sits at the top, left-aligned with entry symbols.
                let tx = legend_rect.x + LEGEND_OUTER_PAD;
                let ty = legend_rect.y + LEGEND_OUTER_PAD + title_h;
                (
                    Some(LegendTitleLayout { text: title_text.to_string(), x: tx, y: ty }),
                    title_h + LEGEND_TITLE_GAP,
                    0.0,
                )
            }
            LegendDirection::Horizontal => {
                // Title sits to the left, vertically centered with entry row.
                let tx = legend_rect.x + LEGEND_OUTER_PAD;
                let ty = legend_rect.y + legend_rect.h / 2.0 + title_h / 3.0;
                (
                    Some(LegendTitleLayout { text: title_text.to_string(), x: tx, y: ty }),
                    0.0,
                    title_w + LEGEND_TITLE_GAP,
                )
            }
        }
    } else {
        (None, 0.0, 0.0)
    };

    let entries_laid_out: Vec<LegendEntryLayout> = match direction {
        LegendDirection::Vertical => {
            let avail_h = (legend_rect.h - 2.0 * LEGEND_OUTER_PAD - title_y_offset).max(0.0);
            let avail_w = (legend_rect.w - 2.0 * LEGEND_OUTER_PAD).max(0.0);
            let row_pitch = line_h + LEGEND_ENTRY_ROW_PAD;
            let n_cols = columns.unwrap_or(1).max(1) as usize;

            // Column width: divide available width evenly across N columns.
            // Label overflow within a column is acceptable — text truncation is render-side.
            let col_w = if n_cols > 1 { avail_w / n_cols as f64 } else { avail_w };

            let max_rows = if row_pitch > 0.0 {
                ((avail_h + LEGEND_ENTRY_ROW_PAD) / row_pitch).floor() as usize
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
                    let col_origin_x = legend_rect.x + LEGEND_OUTER_PAD + col as f64 * col_w;
                    let y = legend_rect.y + LEGEND_OUTER_PAD + title_y_offset
                        + (row as f64) * row_pitch + line_h / 2.0;
                    let symbol_x = col_origin_x + SYMBOL_WIDTH / 2.0;
                    let label_x = col_origin_x + SYMBOL_WIDTH + SYMBOL_LABEL_GAP;
                    LegendEntryLayout {
                        label: e.label.clone(),
                        label_anchor_x: label_x,
                        label_anchor_y: y,
                        symbol_anchor_x: symbol_x,
                        symbol_anchor_y: y,
                        symbol_kind: e.symbol,
                    }
                })
                .collect()
        }
        LegendDirection::Horizontal => {
            let avail_w = (legend_rect.w - 2.0 * LEGEND_OUTER_PAD - title_x_offset).max(0.0);
            let max_label_w = entries
                .iter()
                .map(|e| metrics.measure_width(&e.label, label_font_size))
                .fold(0.0_f64, f64::max);
            let entry_w = SYMBOL_WIDTH + SYMBOL_LABEL_GAP + max_label_w;
            let pitch = entry_w + LEGEND_ENTRY_ROW_PAD;
            let max_n = if pitch > 0.0 {
                ((avail_w + LEGEND_ENTRY_ROW_PAD) / pitch).floor() as usize
            } else {
                0
            };
            let n_fit = entries.len().min(max_n);
            entries
                .iter()
                .take(n_fit)
                .enumerate()
                .map(|(i, e)| {
                    let entry_x = legend_rect.x + LEGEND_OUTER_PAD + title_x_offset
                        + (i as f64) * pitch;
                    let cy = legend_rect.y + legend_rect.h / 2.0;
                    let symbol_x = entry_x + SYMBOL_WIDTH / 2.0;
                    let label_x = entry_x + SYMBOL_WIDTH + SYMBOL_LABEL_GAP;
                    LegendEntryLayout {
                        label: e.label.clone(),
                        label_anchor_x: label_x,
                        label_anchor_y: cy,
                        symbol_anchor_x: symbol_x,
                        symbol_anchor_y: cy,
                        symbol_kind: e.symbol,
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
    };
    (Some(legend), plot_inner)
}

pub(crate) const COLORBAR_WIDTH: f64 = 14.0;
pub(crate) const COLORBAR_HEIGHT: f64 = 180.0;
const COLORBAR_TICK_GAP: f64 = 4.0;

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
    };
    (Some(legend), new_inner)
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
            }],
            title: None,
            colorbar: None,
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
        let size = estimate_legend_size(&es, LegendOrient::Right, 11.0, &m);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Left, inner, 11.0, &m, None, None, 13.0, None);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Top, inner, 11.0, &m, None, None, 13.0, None);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Bottom, inner, 11.0, &m, None, None, 13.0, None);
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert!((legend.rect.y - (inner.y + plot_inner.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_empty_entries_returns_none_and_inner_unchanged() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&[], LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None);
        assert!(legend.is_none());
        assert_eq!(plot_inner, inner);
    }

    #[test]
    fn legend_layout_overflow_drops_entries() {
        let inner = Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 };
        let es = entries(50, 4);
        let m = mock(10.0);
        let (legend, _) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0, None);
        let legend = legend.unwrap();
        assert!(legend.entries.len() < 50, "expected overflow drop; got {} entries", legend.entries.len());
    }
}
