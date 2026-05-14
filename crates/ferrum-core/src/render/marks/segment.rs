//! Segment mark — diagonal line from (x, y) to (x2, y2).
//! Distinct from rule (axis-aligned only): segments may go in any direction.

use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (Some(xf), Some(yf)) = (x_field(ctx, spec), y_field(ctx, spec)) else { return; };
    let Some(x2f) = spec.encoding.x2.as_ref().map(|e| e.field.as_str()) else { return; };
    let Some(y2f) = spec.encoding.y2.as_ref().map(|e| e.field.as_str()) else { return; };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return };

    let style = Stroke {
        stroke: ctx.mark_style.fill,
        stroke_width: ctx.mark_style.stroke_width,
        stroke_dash: ctx.mark_style.stroke_dash.clone(),
    };

    // Phase 9c — per-row pixel offsets from a position adjustment.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let n = xs.len().min(ys.len()).min(x2s.len()).min(y2s.len());
    for i in 0..n {
        let (xv, yv, x2v, y2v) = match (xs[i], ys[i], x2s[i], y2s[i]) {
            (Some(a), Some(b), Some(c), Some(d))
                if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() =>
                (a, b, c, d),
            _ => continue,
        };
        let p1x = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let p1y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let p2x = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
        let p2y = match ctx.scales.y.to_pixel_f64(y2v) { Some(p) => p, None => continue };
        let xo = x_offsets.get(i).copied().unwrap_or(0.0);
        let yo = y_offsets.get(i).copied().unwrap_or(0.0);
        out.line(p1x + xo, p1y + yo, p2x + xo, p2y + yo, &style);
    }
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{to_scene_stroke, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Segment,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    };

    let spec = ctx.spec;
    let (Some(xf), Some(yf)) = (x_field(ctx, spec), y_field(ctx, spec)) else { return empty(); };
    let Some(x2f) = spec.encoding.x2.as_ref().map(|e| e.field.as_str()) else { return empty(); };
    let Some(y2f) = spec.encoding.y2.as_ref().map(|e| e.field.as_str()) else { return empty(); };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
    let x2s = match col_as_f64(ctx.batch, x2f) { Ok(v) => v, Err(_) => return empty() };
    let y2s = match col_as_f64(ctx.batch, y2f) { Ok(v) => v, Err(_) => return empty() };

    let stroke_style = to_scene_stroke(
        ctx.mark_style.fill,
        ctx.mark_style.stroke_width,
        ctx.mark_style.opacity,
        ctx.mark_style.stroke_dash.as_deref(),
        None,
        None,
    );

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    let n = xs.len().min(ys.len()).min(x2s.len()).min(y2s.len());
    for i in 0..n {
        let (xv, yv, x2v, y2v) = match (xs[i], ys[i], x2s[i], y2s[i]) {
            (Some(a), Some(b), Some(c), Some(d))
                if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() =>
                (a, b, c, d),
            _ => continue,
        };
        let p1x = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let p1y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let p2x = match ctx.scales.x.to_pixel_f64(x2v) { Some(p) => p, None => continue };
        let p2y = match ctx.scales.y.to_pixel_f64(y2v) { Some(p) => p, None => continue };
        let xo = x_offsets.get(i).copied().unwrap_or(0.0);
        let yo = y_offsets.get(i).copied().unwrap_or(0.0);
        nodes.push(SceneNode::Line {
            x1: p1x + xo,
            y1: p1y + yo,
            x2: p2x + xo,
            y2: p2y + yo,
            style: stroke_style.clone(),
        });
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Segment,
        nodes,
        data_indices: Some(indices),
        tooltips,
        hrefs,
        descriptions,    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn segment_renders_diagonal_lines() {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Segment,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Segment);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<line ").count(), 2);
    }
}
