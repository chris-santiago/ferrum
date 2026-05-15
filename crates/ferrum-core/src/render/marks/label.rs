use ferrum_scene::{MarkBatchKind, SceneNode, StrokeStyle};

use crate::layout::TextAnchor;
use crate::render::arrow_cast::{col_as_f64, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_color, to_scene_text_style, DrawCtx, MarkBuildResult};

/// Build positioned text label nodes with optional leader lines.
///
/// Each row emits one text label at (x, y) + (dx, dy) offset.  When
/// `mark_style.line = true` and offsets are non-zero a thin leader line
/// is drawn from the label to the datum point.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let xf = match ctx.spec.encoding.x.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Text),
    };
    let yf = match ctx.spec.encoding.y.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Text),
    };

    let Ok(xs) = col_as_f64(ctx.batch, xf) else {
        return MarkBuildResult::empty(MarkBatchKind::Text);
    };
    let Ok(ys) = col_as_f64(ctx.batch, yf) else {
        return MarkBuildResult::empty(MarkBatchKind::Text);
    };

    // Optional text content: `text` encoding field, else format x value.
    let texts: Vec<Option<String>> = if let Some(tf) = ctx.spec.encoding.text.as_ref().map(|e| e.field.as_str()) {
        col_as_str(ctx.batch, tf).unwrap_or_default()
    } else {
        xs.iter().map(|v| v.map(|f| format!("{:.2}", f))).collect()
    };

    let dx = ctx.mark_style.dx.unwrap_or(0.0);
    let dy = ctx.mark_style.dy.unwrap_or(-8.0);
    let font_size = ctx.mark_style.font_size.unwrap_or(11.0);
    let color = with_opacity(ctx.mark_style.fill, ctx.mark_style.opacity);
    // Leader lines require non-trivial offset; no mark_style.line field exists yet.
    let draw_leader = false;

    let n = ctx.batch.num_rows();
    let mut nodes: Vec<SceneNode> = Vec::with_capacity(n);
    let mut data_indices: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        let xv = match xs.get(i).and_then(|v| *v) { Some(v) => v, None => continue };
        let yv = match ys.get(i).and_then(|v| *v) { Some(v) => v, None => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) if p.is_finite() => p, _ => continue };
        let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) if p.is_finite() => p, _ => continue };
        let content = texts.get(i).and_then(|v| v.clone()).unwrap_or_default();

        nodes.push(SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content,
            style: to_scene_text_style(
                color, font_size, TextAnchor::Middle, 0.0,
                ctx.theme.font_family.as_str(),
                ctx.mark_style.font_weight.as_deref(), None, ctx.mark_style.opacity,
            ),
        });

        if draw_leader {
            let lc = with_opacity(ctx.mark_style.fill, ctx.mark_style.opacity * 0.5);
            nodes.push(SceneNode::Polyline {
                points: vec![(px, py), (px + dx * 0.8, py + dy * 0.8)],
                style: StrokeStyle {
                    color: to_scene_color(lc),
                    width: 0.75,
                    opacity: ctx.mark_style.opacity * 0.5,
                    dash: None,
                    stroke_cap: None,
                    stroke_join: None,
                },
            });
        }

        data_indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Text,
        nodes,
        data_indices: Some(data_indices),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}
