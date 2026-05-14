//! mark_line: render rows as a polyline.
//!
//! Two axis modes:
//! - Quantitative x: read `x` as `f64`, project via `scales.x.to_pixel_f64`.
//! - Ordinal x: read `x` as `str`, project via `scales.x.to_pixel_str`. Used
//!   by parallel coordinates where the x axis is a sequence of feature
//!   names. Ordinal y is symmetric (read as `str` via `to_pixel_str`).
//!
//! Grouping rules:
//! - Color encoding only: one polyline per color category (rows of the
//!   same color value linked in batch order). Color determines stroke.
//! - `mark_style.detail` only: one polyline per detail value, theme-default
//!   stroke. Polylines are not legendable.
//! - Both: one polyline per (color, detail) pair — color determines stroke,
//!   detail subdivides each color group. Used by parallel_coordinates so
//!   multiple samples within a class each get their own polyline.
//! - Otherwise: one polyline over all rows in batch order.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ScaleKind;
use crate::render::svg::{Stroke, SvgBuffer};

/// Build an SVG path `d` string from a sequence of (x, y) pixel points using
/// the given interpolation method. Supports "linear" (default), "step"
/// (horizontal-then-vertical at midpoint), "step-before" (vertical-first),
/// and "step-after" (horizontal-first). Unknown methods fall back to linear.
fn build_line_path(points: &[(f64, f64)], interpolate: Option<&str>) -> String {
    use crate::render::svg::fmt_f;
    if points.is_empty() { return String::new(); }
    let method = interpolate.unwrap_or("linear");
    let mut out = format!("M{} {}", fmt_f(points[0].0), fmt_f(points[0].1));
    for i in 1..points.len() {
        let (px, _py) = points[i - 1];
        let (cx, cy) = points[i];
        match method {
            "step" => {
                // Go horizontal to the midpoint x, then vertical to target y,
                // then horizontal to target x.
                let mid_x = (px + cx) / 2.0;
                out.push_str(&format!(" H{} V{} H{}", fmt_f(mid_x), fmt_f(cy), fmt_f(cx)));
            }
            "step-before" => {
                // Drop to next y at current x, then go horizontal to next x.
                out.push_str(&format!(" V{} H{}", fmt_f(cy), fmt_f(cx)));
            }
            "step-after" => {
                // Advance to next x at current y, then drop to next y.
                out.push_str(&format!(" H{} V{}", fmt_f(cx), fmt_f(cy)));
            }
            _ => {
                // "linear", "monotone", "basis" — all use linear L segments.
                out.push_str(&format!(" L{} {}", fmt_f(cx), fmt_f(cy)));
            }
        }
    }
    out
}

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };

    // Per-row pixel projections — populated from whichever axis kind matches
    // the column's data type. Ordinal scales use `to_pixel_str`; quantitative
    // scales use `to_pixel_f64`. Either axis may be ordinal independently.
    let n_rows = ctx.batch.num_rows();
    let xs_pix: Vec<Option<f64>> = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => {
            let xs_str = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            xs_str.iter()
                .map(|opt| opt.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
                .collect()
        }
        _ => {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
            xs.iter()
                .map(|opt| opt
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.x.to_pixel_f64(v)))
                .collect()
        }
    };
    let ys_pix: Vec<Option<f64>> = match &ctx.scales.y {
        ScaleKind::Ordinal(_) => {
            let ys_str = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
            ys_str.iter()
                .map(|opt| opt.as_deref().and_then(|s| ctx.scales.y.to_pixel_str(s)))
                .collect()
        }
        _ => {
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
            ys.iter()
                .map(|opt| opt
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.y.to_pixel_f64(v)))
                .collect()
        }
    };
    if xs_pix.len() != n_rows || ys_pix.len() != n_rows {
        return;
    }

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());
    let detail_values = ctx.mark_style.detail.as_deref()
        .and_then(|f| col_as_str(ctx.batch, f).ok());

    // Build groups: each tuple is (color_key_for_lookup, row_indices). When
    // `detail` is set, each color group is partitioned into per-detail
    // sub-groups; all sub-groups within a color class share the same lookup
    // key so they stroke identically.
    let groups: Vec<(Option<String>, Vec<usize>)> = match (
        color_values.as_ref(),
        detail_values.as_ref(),
        &ctx.scales.color,
    ) {
        (Some(cv), None, Some(_)) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in cv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups
        }
        (None, Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in dv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups.into_iter().map(|(_, rows)| (None, rows)).collect()
        }
        (Some(cv), Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for i in 0..n_rows {
                let composite = (cv[i].clone(), dv[i].clone());
                match groups.iter().position(|(_, rows)| {
                    rows.first().map(|&r| (cv[r].clone(), dv[r].clone()) == composite)
                        .unwrap_or(false)
                }) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((cv[i].clone(), vec![i])),
                }
            }
            groups
        }
        _ => vec![(None, (0..n_rows).collect())],
    };

    // S2/S3: wrap all lines in a <g> when stroke-linecap or stroke-linejoin
    // are set so SVG attribute inheritance applies to every path in the group.
    let need_cap_join = ctx.mark_style.stroke_cap.is_some()
        || ctx.mark_style.stroke_join.is_some();
    if need_cap_join {
        let mut g_attrs = String::new();
        if let Some(ref cap) = ctx.mark_style.stroke_cap {
            g_attrs.push_str(&format!(" stroke-linecap=\"{}\"", cap));
        }
        if let Some(ref join) = ctx.mark_style.stroke_join {
            g_attrs.push_str(&format!(" stroke-linejoin=\"{}\"", join));
        }
        out.raw(&format!("<g{}>", g_attrs));
    }

    let interpolate = ctx.mark_style.interpolate.as_deref();
    let use_path = interpolate.is_some() && interpolate != Some("linear");

    for (key, rows) in groups {
        let mut points: Vec<(f64, f64)> = Vec::new();
        for i in rows {
            let (cx, cy) = match (xs_pix[i], ys_pix[i]) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            points.push((cx, cy));
        }
        if points.len() < 2 { continue; }

        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) =>
                scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.stroke.unwrap_or(ctx.mark_style.fill),
        };
        let stroke_color = with_opacity(stroke_color, ctx.mark_style.opacity);
        let stroke = Stroke {
            stroke: stroke_color,
            stroke_width: ctx.mark_style.stroke_width,
            stroke_dash: ctx.mark_style.stroke_dash.clone(),
        };

        if use_path {
            // S1: emit as a <path> element for non-linear interpolation.
            use crate::render::svg::FillStroke;
            use crate::render::color::fmt_svg;
            let d = build_line_path(&points, interpolate);
            let dash_attr = if let Some(ref dash) = stroke.stroke_dash {
                let v: Vec<String> = dash.iter()
                    .map(|x| crate::render::svg::fmt_f(*x))
                    .collect();
                format!(" stroke-dasharray=\"{}\"", v.join(","))
            } else {
                String::new()
            };
            out.raw(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"{}/> ",
                d,
                fmt_svg(stroke.stroke),
                crate::render::svg::fmt_f(stroke.stroke_width),
                dash_attr,
            ));
        } else {
            out.polyline(&points, &stroke);
        }
    }

    if need_cap_join {
        out.g_close();
    }
}

/// Build a `Vec<PathCmd>` from a sequence of (x, y) pixel points using the
/// given interpolation method. Mirrors `build_line_path` but emits structured
/// commands instead of a `d` string.
fn build_line_cmds(points: &[(f64, f64)], interpolate: Option<&str>) -> Vec<ferrum_scene::PathCmd> {
    use ferrum_scene::PathCmd;
    if points.is_empty() {
        return Vec::new();
    }
    let method = interpolate.unwrap_or("linear");
    let mut cmds = Vec::with_capacity(points.len() * 2);
    cmds.push(PathCmd::MoveTo { x: points[0].0, y: points[0].1 });
    for i in 1..points.len() {
        let (px, py) = points[i - 1];
        let (cx, cy) = points[i];
        match method {
            "step" => {
                let mid_x = (px + cx) / 2.0;
                cmds.push(PathCmd::HLineTo { x: mid_x });
                cmds.push(PathCmd::VLineTo { y: cy });
                cmds.push(PathCmd::HLineTo { x: cx });
            }
            "step-before" => {
                cmds.push(PathCmd::VLineTo { y: cy });
                cmds.push(PathCmd::HLineTo { x: cx });
            }
            "step-after" => {
                cmds.push(PathCmd::HLineTo { x: cx });
                cmds.push(PathCmd::VLineTo { y: cy });
            }
            _ => {
                cmds.push(PathCmd::LineTo { x: cx, y: cy });
            }
        }
    }
    cmds
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{
        to_scene_fill_stroke, to_scene_stroke, MarkBuildResult, MetadataColumns,
    };
    use ferrum_scene::MarkBatchKind;

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Line,
        nodes: vec![],
        data_indices: Some(vec![]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    };

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return empty(),
    };

    let n_rows = ctx.batch.num_rows();
    let xs_pix: Vec<Option<f64>> = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => {
            let xs_str = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
            xs_str.iter()
                .map(|opt| opt.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
                .collect()
        }
        _ => {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
            xs.iter()
                .map(|opt| opt
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.x.to_pixel_f64(v)))
                .collect()
        }
    };
    let ys_pix: Vec<Option<f64>> = match &ctx.scales.y {
        ScaleKind::Ordinal(_) => {
            let ys_str = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
            ys_str.iter()
                .map(|opt| opt.as_deref().and_then(|s| ctx.scales.y.to_pixel_str(s)))
                .collect()
        }
        _ => {
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
            ys.iter()
                .map(|opt| opt
                    .filter(|v| v.is_finite())
                    .and_then(|v| ctx.scales.y.to_pixel_f64(v)))
                .collect()
        }
    };
    if xs_pix.len() != n_rows || ys_pix.len() != n_rows {
        return empty();
    }

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());
    let detail_values = ctx.mark_style.detail.as_deref()
        .and_then(|f| col_as_str(ctx.batch, f).ok());

    let groups: Vec<(Option<String>, Vec<usize>)> = match (
        color_values.as_ref(),
        detail_values.as_ref(),
        &ctx.scales.color,
    ) {
        (Some(cv), None, Some(_)) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in cv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups
        }
        (None, Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in dv.iter().enumerate() {
                let key = v.clone();
                match groups.iter().position(|(k, _)| k == &key) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups.into_iter().map(|(_, rows)| (None, rows)).collect()
        }
        (Some(cv), Some(dv), _) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for i in 0..n_rows {
                let composite = (cv[i].clone(), dv[i].clone());
                match groups.iter().position(|(_, rows)| {
                    rows.first().map(|&r| (cv[r].clone(), dv[r].clone()) == composite)
                        .unwrap_or(false)
                }) {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((cv[i].clone(), vec![i])),
                }
            }
            groups
        }
        _ => vec![(None, (0..n_rows).collect())],
    };

    let interpolate = ctx.mark_style.interpolate.as_deref();
    let use_path = interpolate.is_some() && interpolate != Some("linear");

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut data_indices = Vec::new();

    for (key, rows) in groups {
        let mut points: Vec<(f64, f64)> = Vec::new();
        let mut row_indices: Vec<usize> = Vec::new();
        for i in rows {
            let (cx, cy) = match (xs_pix[i], ys_pix[i]) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            points.push((cx, cy));
            row_indices.push(i);
        }
        if points.len() < 2 { continue; }

        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) =>
                scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.stroke.unwrap_or(ctx.mark_style.fill),
        };
        let stroke_color = with_opacity(stroke_color, ctx.mark_style.opacity);

        if use_path {
            let cmds = build_line_cmds(&points, interpolate);
            let style = to_scene_fill_stroke(
                None,
                Some(stroke_color),
                ctx.mark_style.stroke_width,
                1.0,
                ctx.mark_style.stroke_dash.as_deref(),
            );
            nodes.push(ferrum_scene::SceneNode::Path {
                commands: cmds,
                style,
                closed: false,
            });
        } else {
            let stroke_style = to_scene_stroke(
                stroke_color,
                ctx.mark_style.stroke_width,
                1.0,
                ctx.mark_style.stroke_dash.as_deref(),
                ctx.mark_style.stroke_cap.as_deref(),
                ctx.mark_style.stroke_join.as_deref(),
            );
            nodes.push(ferrum_scene::SceneNode::Polyline {
                points: points.clone(),
                style: stroke_style,
            });
        }
        data_indices.extend(row_indices);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Line,
        nodes,
        data_indices: Some(data_indices),
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
    use crate::spec::mark_style::MarkKwargsSpec;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn line_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Line,
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
    fn line_emits_one_polyline_for_5_rows() {
        let spec = line_spec();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<polyline ").count(), 1);
    }

    #[test]
    fn line_skips_when_fewer_than_two_points() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(!out.finish().contains("<polyline"));
    }

    #[test]
    fn line_detail_only_emits_one_polyline_per_sample() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sample_id", DataType::Utf8, false),
        ]));
        let xs: Vec<f64> = (0..3).flat_map(|_| vec![0.0, 1.0, 2.0, 3.0]).collect();
        let ys: Vec<f64> = (0..3).flat_map(|s| vec![s as f64; 4]).collect();
        let sids: Vec<&str> = (0..12).map(|i| match i / 4 {
            0 => "s0", 1 => "s1", _ => "s2",
        }).collect();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(sids)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("sample_id".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert_eq!(out.finish().matches("<polyline ").count(), 3);
    }

    #[test]
    fn line_color_plus_detail_emits_one_polyline_per_pair() {
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec {
            field: "class".into(),
            type_: None,
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("class", DataType::Utf8, false),
            Field::new("sample_id", DataType::Utf8, false),
        ]));
        let n = 24;
        let xs: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i / 4) as f64).collect();
        let classes: Vec<&str> = (0..n).map(|i| if (i / 4) < 3 { "A" } else { "B" }).collect();
        let sids: Vec<String> = (0..n).map(|i| format!("s{}", i / 4)).collect();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(classes)),
            Arc::new(StringArray::from(sids)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("sample_id".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert_eq!(out.finish().matches("<polyline ").count(), 6);
    }

    #[test]
    fn line_ordinal_x_emits_polyline_via_band_centers() {
        // x: Utf8 categorical, y: Float64 quantitative. Used by
        // parallel_coordinates where features form the x axis.
        let mut spec = line_spec();
        use crate::spec::encoding::DataType as EncDataType;
        spec.encoding.x = Some(EncodingSpec {
            field: "feature".into(),
            type_: Some(EncDataType::Nominal),
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["f0", "f1", "f2", "f3"])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<polyline ").count(), 1);
    }
}
