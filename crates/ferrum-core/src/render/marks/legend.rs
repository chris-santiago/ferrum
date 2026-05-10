//! Internal: draw legend swatches + labels from a LegendLayout.

use crate::layout::{LegendLayout, SymbolKind, TextAnchor, ThemeInputs};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{FillStroke, Stroke, SvgBuffer, TextStyle};

pub fn draw(
    legend: &LegendLayout,
    color_scale: Option<&ColorScale>,
    theme: &ThemeInputs,
    out: &mut SvgBuffer,
) {
    let label_style = TextStyle {
        fill: theme.font_color,
        font_size: theme.label_font_size,
        anchor: TextAnchor::Start,
        angle: 0.0,
    };

    for entry in &legend.entries {
        let color = color_scale
            .and_then(|s| match s {
                ColorScale::Categorical { .. } => s.lookup(&entry.label),
            })
            .unwrap_or(theme.mark_color);
        let sx = entry.symbol_anchor_x;
        let sy = entry.symbol_anchor_y;
        match entry.symbol_kind {
            SymbolKind::Circle => out.circle(sx, sy, 4.0, &FillStroke {
                fill: Some(color), stroke: None, stroke_width: 0.0,
            }),
            SymbolKind::Square => out.rect(
                crate::layout::Rect { x: sx - 4.0, y: sy - 4.0, w: 8.0, h: 8.0 },
                &FillStroke { fill: Some(color), stroke: None, stroke_width: 0.0 },
                None,
            ),
            SymbolKind::Line => out.line(sx - 6.0, sy, sx + 6.0, sy, &Stroke {
                stroke: color, stroke_width: theme.line_stroke_width, stroke_dash: None,
            }),
        }
        out.text(entry.label_anchor_x, entry.label_anchor_y, &entry.label, &label_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LegendDirection, LegendEntryLayout, LegendOrient, Rect};

    #[test]
    fn legend_emits_circle_swatch_per_entry() {
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 20.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![
                LegendEntryLayout {
                    label: "a".into(),
                    label_anchor_x: 88.0,
                    label_anchor_y: 10.0,
                    symbol_anchor_x: 84.0,
                    symbol_anchor_y: 10.0,
                    symbol_kind: SymbolKind::Circle,
                },
                LegendEntryLayout {
                    label: "b".into(),
                    label_anchor_x: 88.0,
                    label_anchor_y: 24.0,
                    symbol_anchor_x: 84.0,
                    symbol_anchor_y: 24.0,
                    symbol_kind: SymbolKind::Circle,
                },
            ],
        };
        let theme = ThemeInputs::default();
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&legend, None, &theme, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 2);
        assert_eq!(s.matches("<text ").count(), 2);
    }
}
