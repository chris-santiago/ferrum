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
//! - `mark_style.group.detail` only: one polyline per detail value, theme-default
//!   stroke. Polylines are not legendable.
//! - Both: one polyline per (color, detail) pair — color determines stroke,
//!   detail subdivides each color group. Used by parallel_coordinates so
//!   multiple samples within a class each get their own polyline.
//! - Otherwise: one polyline over all rows in batch order.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, col_as_str, color_field, resolve_stroke_dash, x_field, y_field, DrawCtx};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::build_color_detail_groups;
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::ScaleKind;

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
        to_scene_fill_stroke_full, to_scene_stroke, MarkBuildResult, MetadataColumns,
    };
    use ferrum_scene::MarkBatchKind;

    let empty = || MarkBuildResult::empty(MarkBatchKind::Line);

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return empty(),
    };

    let n_rows = ctx.batch.num_rows();
    let xs_pix: Vec<Option<f64>> = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => {
            let xs_str = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
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
            let ys_str = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
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
    let detail_values = ctx.mark_style.group.detail.as_deref()
        .and_then(|f| col_as_ordinal_category_str(ctx.batch, f).ok());

    let groups = build_color_detail_groups(
        color_values.as_ref(),
        detail_values.as_ref(),
        ctx.scales.color.is_some(),
        n_rows,
    );

    let interpolate = ctx.mark_style.line.interpolate.as_deref();
    let use_path = interpolate.is_some() && interpolate != Some("linear");

    // Per-row stroke channel vectors — sampled at the first valid row of each group.
    let sw_vals: Option<Vec<Option<f64>>> = spec.encoding.stroke_width.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let sd_vals: Option<Vec<Option<f64>>> = spec.encoding.stroke_dash.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    // opacity / fill_opacity / stroke_opacity resolution (shared resolver, FA-11).
    // Defaults: opacity → mark_style.paint.opacity, fill_opacity → 1.0, stroke_opacity → 1.0.
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));

    let meta = MetadataColumns::from_ctx(ctx);

    // Each group emits exactly one node (a `Path` for non-linear interpolation,
    // else a `Polyline`), attached to the group's representative source row (the
    // first valid point's row). The accumulator keeps `data_indices` in node
    // order so `build_metadata_for_indices` aligns each line's tooltip/href to
    // that row (bug #6). Before the fix, `data_indices` carried one entry per
    // *contributing row* while metadata was full-row, so the lengths diverged.
    let mut acc = MarkNodes::with_capacity(groups.len());

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

        // Sample stroke-channel values from the first valid row of the group;
        // this same row represents the group's node for metadata (bug #6).
        let first = row_indices.first().copied().unwrap_or(0);
        // opacity / fill_opacity / stroke_opacity from the group's first row.
        let (group_opacity, group_fill_opacity, group_stroke_opacity) =
            opacity_res.at_group_first(first);
        let group_stroke_width = sw_vals.as_ref()
            .and_then(|v| v.get(first).copied().flatten())
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(ctx.mark_style.paint.stroke_width);
        let dash_vec: Option<Vec<f64>> = sd_vals.as_ref()
            .and_then(|v| v.get(first).copied().flatten())
            .filter(|v| v.is_finite())
            .and_then(resolve_stroke_dash);
        let effective_dash = dash_vec.as_deref().or(ctx.mark_style.paint.stroke_dash.as_deref());

        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) =>
                scale.lookup(v).unwrap_or(ctx.mark_style.paint.fill),
            _ => ctx.mark_style.paint.stroke.unwrap_or(ctx.mark_style.paint.fill),
        };
        let stroke_color = with_opacity(stroke_color, group_opacity);

        let node = if use_path {
            let cmds = build_line_cmds(&points, interpolate);
            // FA-11: honor the stroke_opacity and fill_opacity channels (the
            // latter was previously dropped) via the complete FillStroke builder.
            let style = to_scene_fill_stroke_full(
                None,
                Some(stroke_color),
                group_stroke_width,
                1.0,
                effective_dash,
                group_fill_opacity,
                group_stroke_opacity,
                0.0,
            );
            ferrum_scene::SceneNode::Path {
                commands: cmds,
                style,
                closed: false,
            }
        } else {
            let mut stroke_style = to_scene_stroke(
                stroke_color,
                group_stroke_width,
                1.0,
                effective_dash,
                ctx.mark_style.line.stroke_cap.as_deref(),
                ctx.mark_style.line.stroke_join.as_deref(),
            );
            stroke_style.stroke_opacity = group_stroke_opacity;
            // Polyline uses `StrokeStyle`, which has no fill channel — the
            // fill_opacity channel only applies to the `Path` (FillStroke)
            // variant emitted under non-linear interpolation.
            ferrum_scene::SceneNode::Polyline {
                points: points.clone(),
                style: stroke_style,
            }
        };
        acc.push(node, first);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Line,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
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
    use ferrum_scene::SceneNode;
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Polyline { .. })).count(), 1);
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert!(result.nodes.is_empty());
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("sample_id".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Polyline { .. })).count(), 3);
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("sample_id".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Polyline { .. })).count(), 6);
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
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Polyline { .. })).count(), 1);
    }

    #[test]
    fn line_integer_ordinal_x_emits_polylines() {
        // D9-B regression: Int64 ordinal x must render the same number of
        // polylines as Utf8 ordinal x. Previously col_as_str failed on Int64
        // and the renderer returned empty().
        use arrow::array::Int64Array;
        use crate::spec::encoding::DataType as EncDataType;
        let mut spec = line_spec();
        spec.encoding.x = Some(EncodingSpec {
            field: "year".into(),
            type_: Some(EncDataType::Ordinal),
            ..Default::default()
        });
        spec.encoding.color = Some(EncodingSpec {
            field: "series".into(),
            type_: Some(EncDataType::Nominal),
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("year", DataType::Int64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("series", DataType::Utf8, false),
        ]));
        // 3 series × 4 years = 12 rows
        let years: Vec<i64> = (0..3).flat_map(|_| vec![2000, 2001, 2002, 2003]).collect();
        let ys: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let series: Vec<&str> = (0..12).map(|i| match i / 4 { 0 => "a", 1 => "b", _ => "c" }).collect();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(years)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(series)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        // 3 series → 3 polylines; none should be missing due to the integer dtype
        let polyline_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Polyline { .. }))
            .count();
        assert_eq!(polyline_count, 3,
            "Int64 ordinal x must emit one polyline per color series; got {polyline_count}");
    }

    // ── #6 metadata/node alignment (group marks) ─────────────────────────────

    /// Bug #6: a multi-group line attaches each group's polyline to that group's
    /// REPRESENTATIVE source row (its first valid point's row), not a per-node
    /// position. Groups are BLOCKED (color "A" rows 0..3, "B" rows 3..6) so the
    /// representative rows are 0 and 3 — distinct from node positions 0,1. Old
    /// code (`build_metadata(ctx)`) attaches row j → node j, so node 1 would get
    /// row 1's tooltip ("A") instead of group B's ("B").
    #[test]
    fn line_group_polyline_aligns_to_representative_row() {
        use crate::spec::encoding::DataType as SDT;
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Nominal), ..Default::default() });
        spec.encoding.tooltip = Some(EncodingSpec { field: "tip".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        // 2 groups × 3 rows, blocked.
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let ys: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let gs: Vec<&str> = vec!["A", "A", "A", "B", "B", "B"];
        let tips: Vec<&str> = gs.clone();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(gs)),
            Arc::new(StringArray::from(tips)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2, "expected one polyline per group");
        let di = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(di.len(), result.nodes.len(), "nodes.len() must equal data_indices.len()");
        assert_eq!(di, &vec![0, 3], "each polyline maps to its group's first valid row");

        let tooltips = result.tooltips.expect("tooltips must be Some");
        let vals: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(vals, vec!["A", "B"],
            "node j tooltip is its group's representative-row label; \
             old full-row code gives node 1 → row 1 → 'A' (the bug).");
    }

    /// Bug #6: alignment holds for the `Path` node variant too (non-linear
    /// interpolation emits a Path instead of a Polyline). The href channel must
    /// align to the representative row even with no tooltip encoded.
    #[test]
    fn line_path_variant_href_aligns_to_representative_row() {
        use crate::spec::encoding::DataType as SDT;
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Nominal), ..Default::default() });
        spec.encoding.href = Some(EncodingSpec { field: "url".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let ys: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let gs: Vec<&str> = vec!["A", "A", "A", "B", "B", "B"];
        let urls: Vec<&str> = vec!["url_A", "url_A", "url_A", "url_B", "url_B", "url_B"];
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(gs)),
            Arc::new(StringArray::from(urls)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        // "basis" interpolation → use_path = true → Path nodes.
        let overrides = MarkKwargsSpec { interpolate: Some("basis".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // Path variant: nodes are Path, not Polyline.
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Path { .. })).count(), 2,
            "non-linear interpolation must emit Path nodes");
        assert!(result.tooltips.is_none(), "no tooltip encoding → tooltips None");
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), result.nodes.len(), "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_A"), "node 0 href = group A representative");
        assert_eq!(hrefs[1].as_deref(), Some("url_B"),
            "node 1 href = group B representative (row 3); old full-row code gives row 1 → 'url_A'.");
    }

    /// No-skip backward-compat: a single-series line (one node, no skipped rows)
    /// keeps its representative row at the first row, and tooltip output matches
    /// the original-row-order semantics for the kept node.
    #[test]
    fn line_no_grouping_tooltip_unchanged() {
        let mut spec = line_spec();
        spec.encoding.tooltip = Some(EncodingSpec { field: "tip".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(StringArray::from(vec!["t0", "t1", "t2", "t3"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 1, "single series → one polyline");
        let di = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(di, &vec![0], "single group's representative is row 0");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 1, "tooltip count == node count");
        assert_eq!(tooltips[0].fields[0].value, "t0", "representative tooltip is row 0's");
    }

    /// D8 regression: Int64 detail column must split line into one polyline per
    /// distinct integer group value.  Previously col_as_str rejected Int64 and
    /// detail_values became None, causing the renderer to fall through to the
    /// single-polyline path and producing one tangled polyline instead of N.
    #[test]
    fn line_int_detail_emits_one_polyline_per_group() {
        use arrow::array::Int64Array;
        let spec = line_spec();
        // 3 groups × 3 x positions = 9 rows, interleaved (group 1, 2, 3, 1, 2, 3, ...)
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Int64, false),
        ]));
        let xs: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let ys: Vec<f64> = vec![0.0, 100.0, 200.0, 1.0, 101.0, 201.0, 2.0, 102.0, 202.0];
        let gs: Vec<i64> = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(Int64Array::from(gs)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec {
            detail: Some("g".into()),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let polyline_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Polyline { .. }))
            .count();
        assert_eq!(polyline_count, 3,
            "Int64 detail column must emit one polyline per group (3 groups); got {polyline_count}");
    }

    // ── FA-11 (#5): mark_line honors the fill_opacity channel ────────────────

    /// FA-11 headline (fail-before / pass-after): a grouped line with a
    /// `fill_opacity` encoding must emit distinct per-group `fill_opacity` on the
    /// `Path` (FillStroke) node it builds. `fill_opacity` only applies to the
    /// non-linear-interpolation `Path` variant (the linear `Polyline` uses
    /// `StrokeStyle`, which has no fill channel), so this uses "basis"
    /// interpolation. Before the fix the channel was dropped → both groups were
    /// `1.0`.
    #[test]
    fn line_path_applies_per_group_fill_opacity() {
        use crate::spec::encoding::DataType as SDT;
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Nominal), ..Default::default() });
        spec.encoding.fill_opacity = Some(EncodingSpec { field: "fo".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("fo", DataType::Float64, false),
        ]));
        // 2 groups × 3 rows, blocked. Group A fill_opacity = 0.25, B = 0.75.
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let ys: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let gs: Vec<&str> = vec!["A", "A", "A", "B", "B", "B"];
        let fos: Vec<f64> = vec![0.25, 0.25, 0.25, 0.75, 0.75, 0.75];
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(gs)),
            Arc::new(Float64Array::from(fos)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        // "basis" → use_path = true → Path nodes carrying a FillStroke.
        let overrides = MarkKwargsSpec { interpolate: Some("basis".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let fill_opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Path { style, .. } = n { Some(style.fill_opacity) } else { None }
        }).collect();
        assert_eq!(fill_opacities.len(), 2, "expected one Path node per group");
        assert!((fill_opacities[0] - 0.25).abs() < 1e-9,
            "group A fill_opacity must be 0.25 (was dropped → 1.0 before FA-11); got {}", fill_opacities[0]);
        assert!((fill_opacities[1] - 0.75).abs() < 1e-9,
            "group B fill_opacity must be 0.75; got {}", fill_opacities[1]);
    }

    /// FA-11 guard: without a `fill_opacity` encoding, a `Path`-variant line
    /// keeps `fill_opacity = 1.0` (the default), so non-fill_opacity charts stay
    /// byte-identical.
    #[test]
    fn line_path_default_fill_opacity_is_one() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { interpolate: Some("basis".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let path = result.nodes.iter().find_map(|n| {
            if let SceneNode::Path { style, .. } = n { Some(style) } else { None }
        }).expect("basis interpolation must emit a Path node");
        assert!((path.fill_opacity - 1.0).abs() < 1e-12,
            "absent fill_opacity encoding must leave fill_opacity = 1.0; got {}", path.fill_opacity);
    }

    // ── Ported from bug_hunt_marks_rendering_r2.rs (R1) ─────────────────────
    // `build_line_cmds` command-count contracts per interpolation method,
    // exercising the real function instead of a hand-copied formula.

    #[test]
    fn build_line_cmds_step_emits_hvh_triplet_per_segment() {
        use ferrum_scene::PathCmd;
        let points = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0), (30.0, 15.0)];
        let cmds = build_line_cmds(&points, Some("step"));
        // MoveTo + 3 segments * (HLineTo, VLineTo, HLineTo) = 10 commands.
        assert_eq!(cmds.len(), 10);
        assert!(matches!(cmds[0], PathCmd::MoveTo { .. }));
        for chunk in cmds[1..].chunks(3) {
            assert!(matches!(chunk[0], PathCmd::HLineTo { .. }));
            assert!(matches!(chunk[1], PathCmd::VLineTo { .. }));
            assert!(matches!(chunk[2], PathCmd::HLineTo { .. }));
        }
    }

    #[test]
    fn build_line_cmds_step_before_emits_vh_pair_per_segment() {
        use ferrum_scene::PathCmd;
        let points = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0), (30.0, 15.0)];
        let cmds = build_line_cmds(&points, Some("step-before"));
        assert_eq!(cmds.len(), 7, "MoveTo + 3 * (VLineTo, HLineTo)");
        for chunk in cmds[1..].chunks(2) {
            assert!(matches!(chunk[0], PathCmd::VLineTo { .. }));
            assert!(matches!(chunk[1], PathCmd::HLineTo { .. }));
        }
    }

    #[test]
    fn build_line_cmds_step_after_emits_hv_pair_per_segment() {
        use ferrum_scene::PathCmd;
        let points = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0), (30.0, 15.0)];
        let cmds = build_line_cmds(&points, Some("step-after"));
        assert_eq!(cmds.len(), 7, "MoveTo + 3 * (HLineTo, VLineTo)");
        for chunk in cmds[1..].chunks(2) {
            assert!(matches!(chunk[0], PathCmd::HLineTo { .. }));
            assert!(matches!(chunk[1], PathCmd::VLineTo { .. }));
        }
    }

    #[test]
    fn build_line_cmds_linear_default_emits_one_lineto_per_segment() {
        use ferrum_scene::PathCmd;
        let points = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0), (30.0, 15.0)];
        let unlabeled = build_line_cmds(&points, None);
        let explicit = build_line_cmds(&points, Some("linear"));
        for cmds in [unlabeled, explicit] {
            assert_eq!(cmds.len(), 4, "MoveTo + 3 LineTo");
            assert!(matches!(cmds[0], PathCmd::MoveTo { .. }));
            for cmd in &cmds[1..] {
                assert!(matches!(cmd, PathCmd::LineTo { .. }));
            }
        }
    }
}
