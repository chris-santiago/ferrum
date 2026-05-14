//! mark_area: filled region between y(x) and the x-axis baseline. Single area
//! over all rows when no color encoding; one area per category otherwise.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{FillStroke, Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };

    let baseline_y = ctx.panel.plot_area.y + ctx.panel.plot_area.h;

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());

    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut g: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let key = v.clone();
                match g.iter().position(|(k, _)| k == &key) {
                    Some(p) => g[p].1.push(i),
                    None => g.push((key, vec![i])),
                }
            }
            g
        }
        _ => vec![(None, (0..xs.len()).collect())],
    };

    // Phase 9c — per-row position-adjustment pixel offsets (Stack).
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Stack writes __stack_y_base__ with per-segment baselines.
    let stack_bases: Option<Vec<Option<f64>>> = ctx.batch.schema()
        .index_of("__stack_y_base__")
        .ok()
        .and_then(|i| {
            ctx.batch.column(i).as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|a| a.iter().collect())
        });
    let is_stacked = stack_bases.is_some();

    // Stacked areas use opaque fills so each band is visually distinct.
    let effective_opacity = if is_stacked { 1.0 } else { ctx.mark_style.opacity };

    // S3: wrap all area paths in a <g> when stroke-linejoin is set.
    let need_join = ctx.mark_style.stroke_join.is_some();
    if need_join {
        let join = ctx.mark_style.stroke_join.as_deref().unwrap_or("miter");
        out.raw(&format!("<g stroke-linejoin=\"{}\">", join));
    }

    let interpolate = ctx.mark_style.interpolate.as_deref();

    for (key, rows) in groups {
        let mut top: Vec<(f64, f64)> = Vec::new();
        let mut bottom: Vec<(f64, f64)> = Vec::new();
        for i in rows {
            let (xv, yv) = match (xs[i], ys[i]) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
                _ => continue,
            };
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cy = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            let xo = x_offsets.get(i).copied().unwrap_or(0.0);
            let yo = y_offsets.get(i).copied().unwrap_or(0.0);
            let cx = cx + xo;
            let cy_top = cy + yo;
            top.push((cx, cy_top));
            if let Some(ref bases) = stack_bases {
                let base_y = bases.get(i)
                    .and_then(|v| *v)
                    .and_then(|b| ctx.scales.y.to_pixel_f64(b))
                    .unwrap_or(baseline_y);
                bottom.push((cx, base_y));
            }
        }
        if top.len() < 2 { continue; }
        let path = if is_stacked {
            build_stacked_area_path(&top, &bottom, interpolate)
        } else {
            build_area_path(&top, baseline_y, interpolate)
        };
        let fill = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) => {
                let base = scale.lookup(v).unwrap_or(ctx.mark_style.fill);
                with_opacity(base, effective_opacity)
            }
            _ => with_opacity(ctx.mark_style.fill, effective_opacity),
        };
        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.fill,
        };
        out.path(&path, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        });

        // S9: draw a line border on top of the area fill.
        if ctx.mark_style.line_border == Some(true) {
            let line_path = build_top_line_path(&top, interpolate);
            out.raw(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"/>",
                line_path,
                crate::render::color::fmt_svg(stroke_color),
                crate::render::svg::fmt_f(ctx.mark_style.stroke_width.max(1.0)),
            ));
        }

        // S10: draw border lines on both top and bottom edges.
        if ctx.mark_style.borders == Some(true) {
            let top_path = build_top_line_path(&top, interpolate);
            let bottom_y = baseline_y;
            // Bottom edge: horizontal line at baseline level.
            let x0 = top[0].0;
            let x_last = top[top.len() - 1].0;
            let bottom_path = format!(
                "M{} {} H{}",
                crate::render::svg::fmt_f(x0),
                crate::render::svg::fmt_f(bottom_y),
                crate::render::svg::fmt_f(x_last),
            );
            let sw = crate::render::svg::fmt_f(ctx.mark_style.stroke_width.max(1.0));
            let stroke_str = crate::render::color::fmt_svg(stroke_color);
            out.raw(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"/>",
                top_path, stroke_str, sw,
            ));
            out.raw(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"/>",
                bottom_path, stroke_str, sw,
            ));
        }
    }

    if need_join {
        out.g_close();
    }
}

/// Build `Vec<PathCmd>` for the top-edge line using the given interpolation
/// method. Mirrors `build_top_line_path` but emits structured commands.
fn build_top_line_cmds(top: &[(f64, f64)], interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    if top.is_empty() { return Vec::new(); }
    let method = interpolate.unwrap_or("linear");
    let mut cmds = Vec::with_capacity(top.len() * 2);
    cmds.push(PathCmd::MoveTo { x: top[0].0, y: top[0].1 });
    for i in 1..top.len() {
        let (px, py) = top[i - 1];
        let (cx, cy) = top[i];
        match method {
            "step" => {
                let mid_x = (px + cx) / 2.0;
                cmds.push(PathCmd::LineTo { x: mid_x, y: py });
                cmds.push(PathCmd::LineTo { x: mid_x, y: cy });
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
            "step-before" => {
                cmds.push(PathCmd::LineTo { x: px, y: cy });
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
            "step-after" => {
                cmds.push(PathCmd::LineTo { x: cx, y: py });
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
            _ => {
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
        }
    }
    cmds
}

/// Build closed area path commands: top edge (with interpolation) + baseline closure.
fn build_area_cmds(top: &[(f64, f64)], baseline: f64, interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    let mut cmds = build_top_line_cmds(top, interpolate);
    let last_x = top[top.len() - 1].0;
    let x0 = top[0].0;
    cmds.push(PathCmd::LineTo { x: last_x, y: baseline });
    cmds.push(PathCmd::LineTo { x: x0, y: baseline });
    cmds.push(PathCmd::Close);
    cmds
}

/// Build closed stacked area path commands: top edge forward, bottom edge reversed.
fn build_stacked_area_cmds(top: &[(f64, f64)], bottom: &[(f64, f64)], interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    let mut cmds = build_top_line_cmds(top, interpolate);
    for &(x, y) in bottom.iter().rev() {
        cmds.push(PathCmd::LineTo { x, y });
    }
    cmds.push(PathCmd::Close);
    cmds
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{
        to_scene_fill_stroke, MarkBuildResult, MetadataColumns,
    };
    use ferrum_scene::MarkBatchKind;

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Area,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    };

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return empty(),
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };

    let baseline_y = ctx.panel.plot_area.y + ctx.panel.plot_area.h;

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());

    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut g: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let key = v.clone();
                match g.iter().position(|(k, _)| k == &key) {
                    Some(p) => g[p].1.push(i),
                    None => g.push((key, vec![i])),
                }
            }
            g
        }
        _ => vec![(None, (0..xs.len()).collect())],
    };

    // Phase 9c — per-row position-adjustment pixel offsets (Stack).
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Stack writes __stack_y_base__ with per-segment baselines.
    let stack_bases: Option<Vec<Option<f64>>> = ctx.batch.schema()
        .index_of("__stack_y_base__")
        .ok()
        .and_then(|i| {
            ctx.batch.column(i).as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|a| a.iter().collect())
        });
    let is_stacked = stack_bases.is_some();

    // Stacked areas use opaque fills so each band is visually distinct.
    let effective_opacity = if is_stacked { 1.0 } else { ctx.mark_style.opacity };

    let interpolate = ctx.mark_style.interpolate.as_deref();

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut data_indices = Vec::new();

    for (key, rows) in groups {
        let mut top: Vec<(f64, f64)> = Vec::new();
        let mut bottom: Vec<(f64, f64)> = Vec::new();
        let mut row_indices: Vec<usize> = Vec::new();
        for i in rows {
            let (xv, yv) = match (xs[i], ys[i]) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
                _ => continue,
            };
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cy = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            let xo = x_offsets.get(i).copied().unwrap_or(0.0);
            let yo = y_offsets.get(i).copied().unwrap_or(0.0);
            let cx = cx + xo;
            let cy_top = cy + yo;
            top.push((cx, cy_top));
            row_indices.push(i);
            if let Some(ref bases) = stack_bases {
                let base_y = bases.get(i)
                    .and_then(|v| *v)
                    .and_then(|b| ctx.scales.y.to_pixel_f64(b))
                    .unwrap_or(baseline_y);
                bottom.push((cx, base_y));
            }
        }
        if top.len() < 2 { continue; }

        let cmds = if is_stacked {
            build_stacked_area_cmds(&top, &bottom, interpolate)
        } else {
            build_area_cmds(&top, baseline_y, interpolate)
        };
        let fill = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) => {
                let base = scale.lookup(v).unwrap_or(ctx.mark_style.fill);
                with_opacity(base, effective_opacity)
            }
            _ => with_opacity(ctx.mark_style.fill, effective_opacity),
        };
        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.fill,
        };
        let style = to_scene_fill_stroke(
            Some(fill),
            ctx.mark_style.stroke,
            ctx.mark_style.stroke_width,
            1.0,
            None,
        );
        nodes.push(ferrum_scene::SceneNode::Path {
            commands: cmds,
            style,
            closed: true,
        });

        // S9: line border on top of the area fill.
        if ctx.mark_style.line_border == Some(true) {
            let line_cmds = build_top_line_cmds(&top, interpolate);
            let border_style = to_scene_fill_stroke(
                None,
                Some(stroke_color),
                ctx.mark_style.stroke_width.max(1.0),
                1.0,
                None,
            );
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: line_cmds,
                style: border_style,
                closed: false,
            });
        }

        // S10: border lines on both top and bottom edges.
        if ctx.mark_style.borders == Some(true) {
            let top_cmds = build_top_line_cmds(&top, interpolate);
            let sw = ctx.mark_style.stroke_width.max(1.0);
            let border_style = to_scene_fill_stroke(
                None,
                Some(stroke_color),
                sw,
                1.0,
                None,
            );
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: top_cmds,
                style: border_style.clone(),
                closed: false,
            });

            // Bottom edge: horizontal line at baseline level.
            let x0 = top[0].0;
            let x_last = top[top.len() - 1].0;
            let bottom_cmds = vec![
                ferrum_scene::PathCmd::MoveTo { x: x0, y: baseline_y },
                ferrum_scene::PathCmd::LineTo { x: x_last, y: baseline_y },
            ];
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: bottom_cmds,
                style: border_style,
                closed: false,
            });
        }

        data_indices.extend(row_indices);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Area,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,    }
}

/// Build the top-edge line path using the given interpolation method.
/// This is the upper boundary of the area shape (before the baseline closure).
fn build_top_line_path(top: &[(f64, f64)], interpolate: Option<&str>) -> String {
    use crate::render::svg::fmt_f;
    if top.is_empty() { return String::new(); }
    let method = interpolate.unwrap_or("linear");
    let mut d = format!("M{} {}", fmt_f(top[0].0), fmt_f(top[0].1));
    for i in 1..top.len() {
        let (px, _py) = top[i - 1];
        let (cx, cy) = top[i];
        match method {
            "step" => {
                let mid_x = (px + cx) / 2.0;
                d.push_str(&format!(" H{} V{} H{}", fmt_f(mid_x), fmt_f(cy), fmt_f(cx)));
            }
            "step-before" => {
                d.push_str(&format!(" V{} H{}", fmt_f(cy), fmt_f(cx)));
            }
            "step-after" => {
                d.push_str(&format!(" H{} V{}", fmt_f(cx), fmt_f(cy)));
            }
            _ => {
                d.push_str(&format!(" L{} {}", fmt_f(cx), fmt_f(cy)));
            }
        }
    }
    d
}

/// Build a closed stacked area path: top edge forward, bottom edge reversed.
fn build_stacked_area_path(top: &[(f64, f64)], bottom: &[(f64, f64)], interpolate: Option<&str>) -> String {
    use crate::render::svg::fmt_f;
    let mut d = build_top_line_path(top, interpolate);
    for &(x, y) in bottom.iter().rev() {
        d.push_str(&format!(" L{} {}", fmt_f(x), fmt_f(y)));
    }
    d.push_str(" Z");
    d
}

/// Build a closed area path: top edge (with interpolation) + baseline closure.
fn build_area_path(top: &[(f64, f64)], baseline: f64, interpolate: Option<&str>) -> String {
    use crate::render::svg::fmt_f;
    let mut d = build_top_line_path(top, interpolate);
    let last_x = top[top.len() - 1].0;
    let x0 = top[0].0;
    d.push_str(&format!(" L{} {}", fmt_f(last_x), fmt_f(baseline)));
    d.push_str(&format!(" L{} {}", fmt_f(x0), fmt_f(baseline)));
    d.push_str(" Z");
    d
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

    fn area_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        }
    }

    #[test]
    fn area_emits_one_path_with_z_close() {
        let spec = area_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<path ").count(), 1);
        assert!(s.contains(" Z\""), "area path must close with Z");
    }

    #[test]
    fn area_uses_translucent_fill() {
        let spec = area_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(out.finish().contains("rgba("));
    }
}
