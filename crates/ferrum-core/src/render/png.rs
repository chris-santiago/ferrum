//! SVG → PNG via usvg + resvg + tiny_skia.

use super::font::INTER_REGULAR;
use super::RenderError;

pub fn svg_string_to_png_bytes(
    svg: &str,
    width_px: u32,
    height_px: u32,
    scale: f64,
) -> Result<Vec<u8>, RenderError> {
    let mut opts = usvg::Options::default();
    opts.fontdb_mut().load_font_data(INTER_REGULAR.to_vec());

    let tree = usvg::Tree::from_str(svg, &opts)
        .map_err(|e| RenderError::ResvgFailed(format!("usvg parse: {e}")))?;

    let mut pixmap = tiny_skia::Pixmap::new(width_px, height_px)
        .ok_or_else(|| RenderError::ResvgFailed("pixmap allocation".into()))?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale as f32, scale as f32),
        &mut pixmap.as_mut(),
    );
    pixmap.encode_png()
        .map_err(|e| RenderError::ResvgFailed(format!("encode_png: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_svg_rasterizes_to_png() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20"><rect x="0" y="0" width="20" height="20" fill="#ff0000"/></svg>"##;
        let png = svg_string_to_png_bytes(svg, 40, 40, 2.0).unwrap();
        assert_eq!(&png[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }
}
