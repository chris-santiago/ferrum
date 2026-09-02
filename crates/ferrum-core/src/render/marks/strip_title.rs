//! Internal: build strip-title scene nodes (background rect + centered text).

use crate::layout::{Rect, StripTitleLayout, ThemeInputs};
use crate::render::draw::{to_scene_fill_stroke, to_scene_text_style};
use ferrum_scene::SceneNode;

/// Build the row-header strip title nodes for grid-mode two-way faceting.
///
/// The row strip is a vertical band to the RIGHT of the panel column
/// (ggplot2 / Altair convention). Its anchor coordinates are pre-computed in
/// `layout/mod.rs` centred in the reserved `row_strip_band_width` region on the
/// right. The row strip text is rotated 90° so it reads bottom-to-top along the
/// right edge. The background rect spans the full vertical extent of the row's
/// cell using the anchor x as the band centre and `panel_h` as the height.
pub fn build_row_strip_title(
    strip: &StripTitleLayout,
    panel_h: f64,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    // Background band width = font_size + 2×padding (same metric used by layout
    // to reserve the band).
    let band_w = strip.font_size + 2.0 * theme.padding.strip_padding;
    // The anchor y is the vertical centre of the panel; the background covers
    // the full panel height starting from strip.anchor.1 - panel_h/2.
    let bg_y = strip.anchor.1 - panel_h / 2.0;
    let bg = SceneNode::Rect {
        x: strip.anchor.0 - band_w / 2.0,
        y: bg_y,
        w: band_w,
        h: panel_h,
        style: crate::render::draw::to_scene_fill_stroke(
            Some(theme.colors.strip_background_color), false, None, false, 0.0, 1.0, None,
        ),
        corner_radius: 0.0,
    };

    let txt = SceneNode::Text {
        x: strip.anchor.0,
        y: strip.anchor.1,
        content: strip.text.clone(),
        slot: None,
        style: crate::render::draw::to_scene_text_style(
            theme.colors.font_color,
            strip.font_size,
            strip.align,
            90.0, // rotated 90°: reads bottom-to-top along the right edge
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
        style: to_scene_fill_stroke(Some(theme.colors.strip_background_color), false, None, false, 0.0, 1.0, None),
        corner_radius: 0.0,
    };

    let txt = SceneNode::Text {
        x: strip.anchor.0,
        y: strip.anchor.1,
        content: strip.text.clone(),
        slot: None,
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

    /// Row strip title nodes: background rect + text, text rotated 90°, background
    /// spans full panel height and is centred on the anchor x.
    #[test]
    fn row_strip_title_builds_right_side_nodes() {
        let strip = StripTitleLayout {
            text: "High".into(),
            anchor: (320.0, 150.0), // x = right-edge centre, y = panel centre
            align: TextAnchor::Middle,
            font_size: 12.0,
        };
        let panel_h = 200.0_f64;
        let theme = ThemeInputs::default();
        let nodes = build_row_strip_title(&strip, panel_h, &theme);

        let rect_count = nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count();
        let text_count = nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert_eq!(rect_count, 1, "expected one background rect");
        assert_eq!(text_count, 1, "expected one text node");

        // Background rect must span the full panel height.
        if let SceneNode::Rect { h, y, .. } = &nodes[0] {
            assert!(
                (*h - panel_h).abs() < 1e-6,
                "background height must equal panel_h={panel_h}; got {h}"
            );
            // y should be anchor.1 - panel_h/2
            let expected_y = strip.anchor.1 - panel_h / 2.0;
            assert!(
                (*y - expected_y).abs() < 1e-6,
                "background y should be {expected_y}; got {y}"
            );
        } else {
            panic!("first node should be Rect");
        }

        // Text node must be at the anchor position.
        if let SceneNode::Text { x, y, style, .. } = &nodes[1] {
            assert!((*x - strip.anchor.0).abs() < 1e-6, "text x should be anchor.0");
            assert!((*y - strip.anchor.1).abs() < 1e-6, "text y should be anchor.1");
            // Rotation must be 90° for right-side strip.
            assert!(
                (style.angle - 90.0).abs() < 1e-6,
                "row strip text must be rotated 90°; got {}",
                style.angle
            );
        } else {
            panic!("second node should be Text");
        }
    }
}
