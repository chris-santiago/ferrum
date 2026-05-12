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

const SYMBOL_WIDTH: f64 = 12.0;
const SYMBOL_LABEL_GAP: f64 = 4.0;
const LEGEND_OUTER_PAD: f64 = 8.0;
const LEGEND_ENTRY_ROW_PAD: f64 = 4.0;
const LEGEND_TITLE_GAP: f64 = 4.0;

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
) -> (Option<LegendLayout>, Rect) {
    if entries.is_empty() {
        return (None, inner);
    }
    let size = estimate_legend_size_with_title(
        entries, orient, label_font_size, title, title_font_size, metrics,
    );

    let (legend_rect, plot_inner) = match orient {
        LegendOrient::Right => {
            let w = size.width.min(inner.w * 0.5);
            let (strip, rest) = inner.split_right(w);
            (strip, rest)
        }
        LegendOrient::Left => {
            let w = size.width.min(inner.w * 0.5);
            let (strip, rest) = inner.split_left(w);
            (strip, rest)
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
            let row_pitch = line_h + LEGEND_ENTRY_ROW_PAD;
            let max_rows = if row_pitch > 0.0 {
                ((avail_h + LEGEND_ENTRY_ROW_PAD) / row_pitch).floor() as usize
            } else {
                0
            };
            let n_fit = entries.len().min(max_rows);
            entries
                .iter()
                .take(n_fit)
                .enumerate()
                .map(|(i, e)| {
                    let y = legend_rect.y + LEGEND_OUTER_PAD + title_y_offset
                        + (i as f64) * row_pitch + line_h / 2.0;
                    let symbol_x = legend_rect.x + LEGEND_OUTER_PAD + SYMBOL_WIDTH / 2.0;
                    let label_x = legend_rect.x + LEGEND_OUTER_PAD + SYMBOL_WIDTH + SYMBOL_LABEL_GAP;
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

/// Estimate the bounding rect of a continuous colorbar (vertical bar +
/// tick labels). Width: bar (`COLORBAR_WIDTH`) + gap + max tick label.
/// Height: pixel range of the bar plus title (when set) and outer padding.
pub fn estimate_colorbar_size(
    title: Option<&str>,
    tick_labels: &[String],
    label_font_size: f64,
    title_font_size: f64,
    metrics: &dyn TextMetrics,
) -> LegendSize {
    let line_h = metrics.line_height(label_font_size);
    let max_tick_w = tick_labels.iter().map(|s|
        metrics.measure_width(s, label_font_size)).fold(0.0_f64, f64::max);
    let bar_h = COLORBAR_HEIGHT;
    let bar_w = COLORBAR_WIDTH;
    let mut width = bar_w + COLORBAR_TICK_GAP + max_tick_w + 2.0 * LEGEND_OUTER_PAD;
    let title_h = title.map(|t| {
        let tw = metrics.measure_width(t, title_font_size);
        width = width.max(tw + 2.0 * LEGEND_OUTER_PAD);
        metrics.line_height(title_font_size) + LEGEND_TITLE_GAP
    }).unwrap_or(0.0);
    let height = title_h + bar_h.max(line_h) + 2.0 * LEGEND_OUTER_PAD;
    LegendSize { width, height }
}

const COLORBAR_WIDTH: f64 = 14.0;
const COLORBAR_HEIGHT: f64 = 180.0;
const COLORBAR_TICK_GAP: f64 = 4.0;

/// Build a colorbar legend (continuous color scale variant). Mirrors
/// `layout_legend` but for the gradient/colorbar shape — bar geometry,
/// per-tick y-positions, and title placement.
///
/// `stops` is the gradient definition as (position in [0, 1], hex color);
/// `tick_labels` is the data-domain labels (low → high) used to position
/// tick marks on the bar.
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
) -> (Option<LegendLayout>, Rect) {
    if stops.is_empty() {
        return (None, plot_inner);
    }
    let size = estimate_colorbar_size(
        title.as_deref(), &tick_labels, label_font_size, title_font_size, metrics,
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
    let bar_bottom = (bar_top + COLORBAR_HEIGHT).min(legend_rect.y + legend_rect.h - LEGEND_OUTER_PAD);
    let bar_left = legend_rect.x + LEGEND_OUTER_PAD;
    let bar_rect = Rect {
        x: bar_left,
        y: bar_top,
        w: COLORBAR_WIDTH,
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0);
        let legend = legend.expect("legend should be Some");
        assert_eq!(legend.orient, LegendOrient::Right);
        assert_eq!(legend.direction, LegendDirection::Vertical);
        assert!((legend.rect.x - (plot_inner.x + plot_inner.w)).abs() < 1e-6);
        assert!(plot_inner.w < inner.w);
        assert_eq!(plot_inner.h, inner.h);
    }

    #[test]
    fn legend_layout_left_orient() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Left, inner, 11.0, &m, None, None, 13.0);
        let legend = legend.unwrap();
        assert_eq!(legend.rect.x, inner.x);
        assert!((plot_inner.x - (inner.x + legend.rect.w)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_top_orient_horizontal() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Top, inner, 11.0, &m, None, None, 13.0);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Bottom, inner, 11.0, &m, None, None, 13.0);
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert!((legend.rect.y - (inner.y + plot_inner.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_empty_entries_returns_none_and_inner_unchanged() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&[], LegendOrient::Right, inner, 11.0, &m, None, None, 13.0);
        assert!(legend.is_none());
        assert_eq!(plot_inner, inner);
    }

    #[test]
    fn legend_layout_overflow_drops_entries() {
        let inner = Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 };
        let es = entries(50, 4);
        let m = mock(10.0);
        let (legend, _) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None, None, 13.0);
        let legend = legend.unwrap();
        assert!(legend.entries.len() < 50, "expected overflow drop; got {} entries", legend.entries.len());
    }
}
