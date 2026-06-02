//! Internal: build strip-title scene nodes (background rect + centered text).

use crate::layout::{Rect, StripTitleLayout, ThemeInputs};
use crate::render::draw::{to_scene_fill_stroke, to_scene_text_style};
use ferrum_scene::SceneNode;

/// Build the row-header strip title nodes for grid-mode two-way faceting.
///
/// The row strip is a vertical band to the left of the panel column.  Its
/// anchor coordinates are pre-computed in `layout/mod.rs` (centred in the
/// reserved `row_strip_band_width` region). Unlike the column-header strip,
/// the row strip text is rotated 270° so it reads top-to-bottom along the left
/// edge.  The background rect spans the full vertical extent of the row's cell
/// using the anchor position as the centre.
pub fn build_row_strip_title(strip: &StripTitleLayout, theme: &ThemeInputs) -> Vec<SceneNode> {
    // Background band: height = font_size + 2×padding; width estimated from font_size.
    let band_h = strip.font_size + 2.0 * theme.padding.strip_padding;
    // Place a tall background behind the strip text. We use the anchor as the
    // top-left of the text's vertical extent, offset up by half the band height.
    let bg = SceneNode::Rect {
        x: strip.anchor.0 - band_h / 2.0,
        y: strip.anchor.1 - strip.font_size,
        w: band_h,
        h: strip.font_size * 2.0, // approximate; real extent controlled by panel height
        style: crate::render::draw::to_scene_fill_stroke(
            Some(theme.colors.strip_background_color), None, 0.0, 1.0, None,
        ),
        corner_radius: 0.0,
    };

    let txt = SceneNode::Text {
        x: strip.anchor.0,
        y: strip.anchor.1,
        content: strip.text.clone(),
        style: crate::render::draw::to_scene_text_style(
            theme.colors.font_color,
            strip.font_size,
            strip.align,
            270.0, // rotated to read along the left edge
            &theme.typography.font_family,
            None,
            None,
            1.0,
        ),
    };

    vec![bg, txt]
}

pub fn build_strip_title(
    strip: &StripTitleLayout,
    panel_rect: &Rect,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let band_h = (panel_rect.y - (strip.anchor.1 - strip.font_size - theme.padding.strip_padding))
        .abs()
        .max(strip.font_size + 2.0 * theme.padding.strip_padding);
    let band = Rect {
        x: panel_rect.x,
        y: panel_rect.y - band_h,
        w: panel_rect.w,
        h: band_h,
    };

    let bg = SceneNode::Rect {
        x: band.x,
        y: band.y,
        w: band.w,
        h: band.h,
        style: to_scene_fill_stroke(Some(theme.colors.strip_background_color), None, 0.0, 1.0, None),
        corner_radius: 0.0,
    };

    let txt = SceneNode::Text {
        x: strip.anchor.0,
        y: strip.anchor.1,
        content: strip.text.clone(),
        style: to_scene_text_style(
            theme.colors.font_color,
            strip.font_size,
            strip.align,
            0.0,
            &theme.typography.font_family,
            None,
            None,
            1.0,
        ),
    };

    vec![bg, txt]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::TextAnchor;

    #[test]
    fn strip_title_builds_background_and_text() {
        let strip = StripTitleLayout {
            text: "setosa".into(),
            anchor: (50.0, 18.0),
            align: TextAnchor::Middle,
            font_size: 13.0,
        };
        let panel = Rect { x: 0.0, y: 22.0, w: 100.0, h: 78.0 };
        let theme = ThemeInputs::default();
        let nodes = build_strip_title(&strip, &panel, &theme);
        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert_eq!(rect_count, 1, "expected strip background rect");
        assert_eq!(text_count, 1, "expected strip title text");
    }
}
