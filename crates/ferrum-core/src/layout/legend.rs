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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendLayout {
    pub rect: Rect,
    pub orient: LegendOrient,
    pub direction: LegendDirection,
    pub entries: Vec<LegendEntryLayout>,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendSize {
    pub width: f64,
    pub height: f64,
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
) -> (Option<LegendLayout>, Rect) {
    if entries.is_empty() {
        return (None, inner);
    }
    let size = estimate_legend_size(entries, orient, label_font_size, metrics);

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
    let entries_laid_out: Vec<LegendEntryLayout> = match direction {
        LegendDirection::Vertical => {
            let avail_h = (legend_rect.h - 2.0 * LEGEND_OUTER_PAD).max(0.0);
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
                    let y = legend_rect.y + LEGEND_OUTER_PAD + (i as f64) * row_pitch + line_h / 2.0;
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
            let avail_w = (legend_rect.w - 2.0 * LEGEND_OUTER_PAD).max(0.0);
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
                    let entry_x = legend_rect.x + LEGEND_OUTER_PAD + (i as f64) * pitch;
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
    };
    (Some(legend), plot_inner)
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Left, inner, 11.0, &m, None);
        let legend = legend.unwrap();
        assert_eq!(legend.rect.x, inner.x);
        assert!((plot_inner.x - (inner.x + legend.rect.w)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_top_orient_horizontal() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Top, inner, 11.0, &m, None);
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
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Bottom, inner, 11.0, &m, None);
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert!((legend.rect.y - (inner.y + plot_inner.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_empty_entries_returns_none_and_inner_unchanged() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&[], LegendOrient::Right, inner, 11.0, &m, None);
        assert!(legend.is_none());
        assert_eq!(plot_inner, inner);
    }

    #[test]
    fn legend_layout_overflow_drops_entries() {
        let inner = Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 };
        let es = entries(50, 4);
        let m = mock(10.0);
        let (legend, _) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m, None);
        let legend = legend.unwrap();
        assert!(legend.entries.len() < 50, "expected overflow drop; got {} entries", legend.entries.len());
    }
}
