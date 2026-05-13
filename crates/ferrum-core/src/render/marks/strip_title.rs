//! Internal: draw a per-panel strip-title band (background rect + centered text).

use crate::layout::{Rect, StripTitleLayout, ThemeInputs};
use crate::render::svg::{FillStroke, SvgBuffer, TextStyle};

pub fn draw(
    strip: &StripTitleLayout,
    panel_rect: &Rect,
    theme: &ThemeInputs,
    out: &mut SvgBuffer,
) {
    let band_h = (panel_rect.y - (strip.anchor.1 - strip.font_size - theme.strip_padding))
        .abs()
        .max(strip.font_size + 2.0 * theme.strip_padding);
    let band = Rect {
        x: panel_rect.x,
        y: panel_rect.y - band_h,
        w: panel_rect.w,
        h: band_h,
    };
    out.rect(band, &FillStroke {
        fill: Some(theme.strip_background_color),
        stroke: None,
        stroke_width: 0.0,
    }, None);
    out.text(strip.anchor.0, strip.anchor.1, &strip.text, &TextStyle {
        fill: theme.font_color,
        font_size: strip.font_size,
        anchor: strip.align,
        angle: 0.0,
        font_family: &theme.font_family,
        font_weight: None,
        dominant_baseline: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::TextAnchor;

    #[test]
    fn strip_title_emits_background_and_text() {
        let strip = StripTitleLayout {
            text: "setosa".into(),
            anchor: (50.0, 18.0),
            align: TextAnchor::Middle,
            font_size: 13.0,
        };
        let panel = Rect { x: 0.0, y: 22.0, w: 100.0, h: 78.0 };
        let theme = ThemeInputs::default();
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&strip, &panel, &theme, &mut out);
        let s = out.finish();
        assert!(s.contains("<rect "), "expected strip background rect");
        assert!(s.contains(">setosa</text>") || s.contains(">setosa<"));
    }
}
