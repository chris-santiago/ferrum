//! Internal: build strip-title scene nodes (background rect + centered text).

use crate::layout::{Rect, StripTitleLayout, ThemeInputs};
use crate::render::draw::{to_scene_fill_stroke, to_scene_text_style};
use ferrum_scene::SceneNode;

pub fn build_strip_title(
    strip: &StripTitleLayout,
    panel_rect: &Rect,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let band_h = (panel_rect.y - (strip.anchor.1 - strip.font_size - theme.strip_padding))
        .abs()
        .max(strip.font_size + 2.0 * theme.strip_padding);
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
        style: to_scene_fill_stroke(Some(theme.strip_background_color), None, 0.0, 1.0, None),
        corner_radius: 0.0,
    };

    let txt = SceneNode::Text {
        x: strip.anchor.0,
        y: strip.anchor.1,
        content: strip.text.clone(),
        style: to_scene_text_style(
            theme.font_color,
            strip.font_size,
            strip.align,
            0.0,
            &theme.font_family,
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
