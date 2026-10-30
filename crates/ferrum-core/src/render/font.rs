//! Bundled Inter Regular + FontdueMetrics impl of TextMetrics.

use crate::layout::TextMetrics;

pub const INTER_REGULAR: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.ttf");

/// A shaped run's ink, in pixels at the run's own `font_size`, relative to
/// the run's pen start (x) and baseline (y): `advance` is the shaped
/// (kerned) total advance `emit_text_svg` draws the next glyph run from;
/// `xmin`/`xmax` is the horizontal reach of every glyph's own outline, which
/// can fall outside `[0, advance]` for a negative first-glyph bearing or an
/// overhanging last glyph; `ymin`/`ymax` is the vertical reach in the font's
/// up-positive convention (a descender's `ymin` is negative); `ascent`/
/// `descent` are the font-wide positive magnitudes
/// [`FontdueMetrics::shaped_ink`]'s caller unions with the ink so a label
/// with no tall or deep glyphs still reserves the font's own line box.
pub(crate) struct ShapedInk {
    pub(crate) advance: f64,
    pub(crate) xmin: f64,
    pub(crate) xmax: f64,
    pub(crate) ymin: f64,
    pub(crate) ymax: f64,
    pub(crate) ascent: f64,
    pub(crate) descent: f64,
}

pub struct FontdueMetrics {
    font: fontdue::Font,
    // the shaper usvg (and so every browser-equivalent render
    // path ferrum ships) shapes with. Parsed once from the same bytes as
    // `font`, borrowed for `'static` since `INTER_REGULAR` is a `'static`
    // byte slice.
    buzz_face: rustybuzz::Face<'static>,
}

impl FontdueMetrics {
    pub fn new() -> Self {
        let font = fontdue::Font::from_bytes(INTER_REGULAR, fontdue::FontSettings::default())
            .expect("bundled Inter-Regular.ttf must parse");
        // Same build invariant as the fontdue parse above: INTER_REGULAR is a
        // compile-time include_bytes! asset, not user input, so a parse
        // failure here can only mean the bundled font file is corrupt.
        let buzz_face = rustybuzz::Face::from_slice(INTER_REGULAR, 0)
            .expect("bundled Inter-Regular.ttf must parse as a rustybuzz face");
        Self { font, buzz_face }
    }

    pub fn font(&self) -> &fontdue::Font {
        &self.font
    }

    /// Shape `label` at `font_size` with GPOS kerning on, the same shaping
    /// usvg applies when it renders the glyphs `emit_text` writes.
    /// `rustybuzz::shape` returns glyph advances, offsets and ids in
    /// font design units (`units_per_em`); every value here is already
    /// scaled to pixels at `font_size`. A glyph with no outline (whitespace)
    /// contributes to `advance` but not to the ink box, so a whitespace-only
    /// label collapses to the plain advance box, `ink.xmin == ink.xmax == 0.0`
    /// and the same for y — the same degenerate case
    /// [`crate::render::marks::text::glyph_edge_extents`] handled before
    /// this measured ink at all.
    pub(crate) fn shaped_ink(&self, label: &str, font_size: f64) -> ShapedInk {
        use rustybuzz::ttf_parser::GlyphId;

        let upem = self.buzz_face.units_per_em() as f64;
        let scale = if upem > 0.0 { font_size / upem } else { 0.0 };
        // Font-wide ascent/descent, read from the same face `shaped_ink`
        // shapes with (not fontdue's): the content-independent line-box
        // floor a shaped run's ink is unioned with, so a label with no
        // descender still reserves descent clearance. `descender()` is
        // negative in `ttf_parser`'s convention; negated to the same
        // positive-magnitude convention as `ascent`.
        let ascent = f64::from(self.buzz_face.ascender()) * scale;
        let descent = -f64::from(self.buzz_face.descender()) * scale;

        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(label);
        let shaped = rustybuzz::shape(&self.buzz_face, &[], buffer);

        let mut pen_x = 0.0_f64;
        let mut pen_y = 0.0_f64;
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut has_ink = false;
        for (info, pos) in shaped.glyph_infos().iter().zip(shaped.glyph_positions()) {
            let gx = pen_x + pos.x_offset as f64 * scale;
            let gy = pen_y + pos.y_offset as f64 * scale;
            if let Some(bbox) = self.buzz_face.glyph_bounding_box(GlyphId(info.glyph_id as u16)) {
                let (bx0, bx1) = (gx + bbox.x_min as f64 * scale, gx + bbox.x_max as f64 * scale);
                let (by0, by1) = (gy + bbox.y_min as f64 * scale, gy + bbox.y_max as f64 * scale);
                if has_ink {
                    xmin = xmin.min(bx0);
                    xmax = xmax.max(bx1);
                    ymin = ymin.min(by0);
                    ymax = ymax.max(by1);
                } else {
                    (xmin, xmax, ymin, ymax) = (bx0, bx1, by0, by1);
                    has_ink = true;
                }
            }
            pen_x += pos.x_advance as f64 * scale;
            pen_y += pos.y_advance as f64 * scale;
        }
        ShapedInk { advance: pen_x, xmin, xmax, ymin, ymax, ascent, descent }
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
