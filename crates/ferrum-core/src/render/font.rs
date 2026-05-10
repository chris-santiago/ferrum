//! Bundled Inter Regular + FontdueMetrics impl of TextMetrics.

use crate::layout::TextMetrics;

pub const INTER_REGULAR: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.ttf");

pub struct FontdueMetrics {
    font: fontdue::Font,
}

impl FontdueMetrics {
    pub fn new() -> Self {
        let font = fontdue::Font::from_bytes(INTER_REGULAR, fontdue::FontSettings::default())
            .expect("bundled Inter-Regular.ttf must parse");
        Self { font }
    }

    pub fn font(&self) -> &fontdue::Font {
        &self.font
    }
}

impl Default for FontdueMetrics {
    fn default() -> Self { Self::new() }
}

impl TextMetrics for FontdueMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars()
            .map(|c| self.font.metrics(c, font_size as f32).advance_width as f64)
            .sum()
    }

    fn line_height(&self, font_size: f64) -> f64 {
        let lm = self
            .font
            .horizontal_line_metrics(font_size as f32)
            .expect("Inter has horizontal line metrics");
        (lm.ascent - lm.descent + lm.line_gap) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_parses() {
        let _ = FontdueMetrics::new();
    }

    #[test]
    fn measure_width_is_positive_for_ascii() {
        let m = FontdueMetrics::new();
        let w = m.measure_width("100", 11.0);
        assert!(w > 0.0, "expected positive width, got {w}");
        assert!(w < 50.0, "unexpectedly wide: {w}");
    }

    #[test]
    fn line_height_is_positive_and_proportional() {
        let m = FontdueMetrics::new();
        let lh_11 = m.line_height(11.0);
        let lh_22 = m.line_height(22.0);
        assert!(lh_11 > 0.0);
        assert!((lh_22 / lh_11 - 2.0).abs() < 0.1, "lh_11={lh_11}, lh_22={lh_22}");
    }
}
