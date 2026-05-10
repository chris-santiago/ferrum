//! Internal: draw axis line, ticks, tick labels, and axis title from an AxisLayout.

use crate::layout::{AxisLayout, AxisOrient, TextAnchor, ThemeInputs};
use crate::render::svg::{Stroke, SvgBuffer, TextStyle};

pub fn draw(axis: &AxisLayout, theme: &ThemeInputs, out: &mut SvgBuffer) {
    let line_style = Stroke {
        stroke: theme.axis_line_color,
        stroke_width: theme.axis_line_width,
        stroke_dash: None,
    };
    let r = axis.axis_line;
    out.line(r.x, r.y, r.x + r.w, r.y + r.h, &line_style);

    let tick_style = Stroke {
        stroke: theme.tick_color,
        stroke_width: theme.axis_line_width,
        stroke_dash: None,
    };
    let label_style_base = TextStyle {
        fill: theme.font_color,
        font_size: theme.label_font_size,
        anchor: TextAnchor::Middle,
        angle: 0.0,
    };
    for tick in &axis.ticks {
        let (tx1, ty1, tx2, ty2, label_x, label_y, anchor, angle) = match axis.orient {
            AxisOrient::Bottom => (
                tick.position, r.y, tick.position, r.y + theme.tick_size,
                tick.position, r.y + theme.tick_size + theme.label_font_size + 2.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Top => (
                tick.position, r.y, tick.position, r.y - theme.tick_size,
                tick.position, r.y - theme.tick_size - 4.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Left => (
                r.x, tick.position, r.x - theme.tick_size, tick.position,
                r.x - theme.tick_size - 2.0, tick.position + theme.label_font_size / 3.0,
                TextAnchor::End, 0.0,
            ),
            AxisOrient::Right => (
                r.x, tick.position, r.x + theme.tick_size, tick.position,
                r.x + theme.tick_size + 2.0, tick.position + theme.label_font_size / 3.0,
                TextAnchor::Start, 0.0,
            ),
        };
        out.line(tx1, ty1, tx2, ty2, &tick_style);
        let mut style = label_style_base;
        style.anchor = anchor;
        style.angle = angle;
        out.text(label_x, label_y, &tick.label, &style);
    }

    if let Some(t) = &axis.title {
        let title_style = TextStyle {
            fill: theme.font_color,
            font_size: theme.title_font_size,
            anchor: TextAnchor::Middle,
            angle: t.angle,
        };
        out.text(t.anchor_x, t.anchor_y, &t.text, &title_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AxisLayout, AxisOrient, AxisTitleLayout, Rect, TickLayout};

    #[test]
    fn axis_draws_line_ticks_and_title() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout { position: 25.0, label: "0".into(), label_angle: 0.0, elided: false },
                TickLayout { position: 75.0, label: "1".into(), label_angle: 0.0, elided: false },
            ],
            title: Some(AxisTitleLayout {
                text: "x".into(),
                anchor_x: 50.0,
                anchor_y: 95.0,
                angle: 0.0,
            }),
        };
        let theme = ThemeInputs::default();
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&axis, &theme, &mut out);
        let s = out.finish();
        assert!(s.contains("<line "));
        assert!(s.matches("<line ").count() >= 3);
        assert!(s.contains(">x</text>") || s.contains(">x<"));
    }
}
