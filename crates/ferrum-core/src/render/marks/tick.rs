//! mark_tick: six modes —
//!   quantitative x only → x-rug: vertical ticks at panel baseline;
//!   quantitative y only → y-rug: horizontal ticks at left axis edge;
//!   ordinal x + quantitative y → horizontal tick at data y position (boxplot median);
//!   ordinal y + quantitative x → vertical tick at data x position (strip plot);
//!   ordinal y only → horizontal crossbars centered on each category band;
//!   ordinal x only → vertical crossbars centered on each category band.

use crate::render::draw::{col_as_f64, col_as_positional_category_str, x_field, y_field, DrawCtx};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::ScaleKind;

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{MarkBuildResult, to_scene_stroke, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Tick);

    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;

    // Common setup shared by all four tick modes.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    // Under an ordinal-band Dodge, band-fraction ticks (boxplot median line,
    // whisker caps) must shrink to their sub-band so adjacent dodge groups don't
    // overlap. No Dodge → n_groups == 1 → byte-identical to the non-dodged tick.
    let n_groups = crate::render::position::n_dodge_groups(ctx.batch) as f64;
    let stroke_color = ctx.mark_style.paint.stroke.unwrap_or(ctx.mark_style.paint.fill);
    let default_stroke_width = ctx.mark_style.paint.stroke_width.max(1.0);

    // Per-row opacity via the shared OpacityResolver (C7); stroke_width stays
    // local. Tick is a stroke-only mark (no fill / stroke_opacity columns), so
    // only the resolver's `opacity` slot is read.
    let opacity_res =
        OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let row_stroke = |i: usize| -> ferrum_scene::StrokeStyle {
        let (opacity, _, _) = opacity_res.at_row(i);
        let width = stroke_width_values.as_ref()
            .and_then(|v| v.get(i).copied().flatten())
            .unwrap_or(default_stroke_width);
        to_scene_stroke(stroke_color, width, opacity, None, None, None)
    };

    let meta = MetadataColumns::from_ctx(ctx);

    // All modes use the accumulator pattern: push each emitted node with its
    // source row index, finalize, then build metadata for kept indices only
    // (#6 defect class fix — metadata aligned to nodes, not all rows).
    let xf_opt = x_field(ctx, spec);

    // No x field: either ordinal-y-only or quantitative-y-rug.
    if xf_opt.is_none() {
        if let Some(yf) = y_field(ctx, spec) {
            // Ordinal y only → horizontal crossbars centered on each category band.
            if matches!(&ctx.scales.y, ScaleKind::Ordinal(_)) {
                let ys = match col_as_positional_category_str(ctx.batch, yf) {
                    Ok(v) => v,
                    Err(_) => return empty(),
                };
                let n_cats = {
                    let mut set = std::collections::HashSet::<&str>::new();
                    for v in ys.iter().flatten() { set.insert(v.as_str()); }
                    set.len().max(1)
                };
                // Band-geometry unification (#39 phase 2): x is unencoded in
                // this mode, so `ctx.scales.x` is the dummy unit scale, whose
                // `explicit_band_extent()` is always `None` — this always
                // falls through to `panel.w`, matching the site-pairing
                // convention (cross-axis tick length keyed to the panel-w
                // term regardless of which field is ordinal).
                //
                // GH #66 sub-band clamp NOT applicable here: Dodge offsets
                // `cy` (the band axis, y), but this tick's length runs along
                // the CROSS axis (x). Sibling dodge groups land at different
                // `cy` and never share a row, so this tick length cannot
                // collide with a neighbouring group regardless of `padding`
                // — the `n_groups` divisor here is purely a cosmetic
                // width-pairing convention with the band-axis formulas, not
                // a collision-prone one.
                let band_extent = crate::render::marks::channels::band_extent_or(&ctx.scales.x, panel.w);
                let tick_half = (band_extent / n_cats as f64 / n_groups) * ctx.mark_style.misc.band_size.unwrap_or(0.3);
                let baseline_x = panel.x;
                let mut acc = MarkNodes::with_capacity(ys.len());
                for i in 0..ys.len() {
                    let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                    let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                    let cy = cy + y_offsets[i];
                    let bx = baseline_x + x_offsets[i];
                    acc.push(SceneNode::Line {
                        x1: bx,
                        y1: cy,
                        x2: bx + 2.0 * tick_half,
                        y2: cy,
                        style: row_stroke(i),
                    }, i);
                }
                let (nodes, data_indices) = acc.finalize();
                let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
                return MarkBuildResult {
                    kind: MarkBatchKind::Tick, nodes, data_indices: Some(data_indices),
                    tooltips, hrefs, descriptions,
                };
            }

            // Quantitative y only → y-rug: horizontal ticks at left axis edge.
            let tick_len = ctx.theme.sizes.tick_size * 2.0;
            let ys = match col_as_f64(ctx.batch, yf) {
                Ok(v) => v,
                Err(_) => return empty(),
            };
            let baseline_x = panel.x;
            let mut acc = MarkNodes::with_capacity(ys.len());
            for i in 0..ys.len() {
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let py = py + y_offsets[i];
                let bx = baseline_x + x_offsets[i];
                acc.push(SceneNode::Line {
                    x1: bx, y1: py,
                    x2: bx + tick_len, y2: py,
                    style: row_stroke(i),
                }, i);
            }
            let (nodes, data_indices) = acc.finalize();
            let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
            return MarkBuildResult {
                kind: MarkBatchKind::Tick, nodes, data_indices: Some(data_indices),
                tooltips, hrefs, descriptions,
            };
        }
        // No x and no y: return empty.
        return empty();
    }

    let xf = xf_opt.expect("invariant: xf_opt is Some — None case returned above");

    // Ordinal x + quantitative y → horizontal tick at data y position.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_positional_category_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
            let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in xs.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            // Band-geometry unification (#39 phase 2): honor an explicit
            // x-band pixel range when the resolver recorded one; otherwise
            // identical to `panel.w`.
            let band_extent = crate::render::marks::channels::band_extent_or(&ctx.scales.x, panel.w);
            let tick_half_raw = (band_extent / n_cats as f64 / n_groups) * ctx.mark_style.misc.band_size.unwrap_or(0.3);
            // Clamp to the Dodge sub-band (GH #66): this tick spans the SAME
            // axis (x) that Dodge offsets `cx` along, so — unlike the
            // ordinal-only crossbar modes above, whose tick length runs along
            // the cross axis and never collides with a sibling dodge group —
            // a band_size-factor width here can genuinely overlap a
            // neighbouring group at high padding. Byte-identical no-op when
            // undodged or at low/default padding. `*2.0`/`/2.0` is exact in
            // IEEE-754 (power-of-two scaling), so the no-op case round-trips
            // to `tick_half_raw` bit-for-bit.
            let tick_half = crate::render::position::clamp_to_dodge_sub_band(tick_half_raw * 2.0, ctx.batch) / 2.0;
            let mut acc = MarkNodes::with_capacity(xs.len());
            for i in 0..xs.len() {
                let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
                let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
                let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
                let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
                let cx = cx + x_offsets[i];
                let py = py + y_offsets[i];
                acc.push(SceneNode::Line {
                    x1: cx - tick_half,
                    y1: py,
                    x2: cx + tick_half,
                    y2: py,
                    style: row_stroke(i),
                }, i);
            }
            let (nodes, data_indices) = acc.finalize();
            let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
            return MarkBuildResult {
                kind: MarkBatchKind::Tick,
                nodes,
                data_indices: Some(data_indices),
                tooltips,
                hrefs,
                descriptions,
            };
        }
    }

    // Ordinal y + quantitative x → vertical tick at data x position (strip plot).
    if matches!(&ctx.scales.y, ScaleKind::Ordinal(_)) {
        if let Some(yf) = y_field(ctx, spec) {
            let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
            let ys = match col_as_positional_category_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return empty() };
            let n_cats = {
                let mut set = std::collections::HashSet::<&str>::new();
                for v in ys.iter().flatten() { set.insert(v.as_str()); }
                set.len().max(1)
            };
            // Band-geometry unification (#39 phase 2): honor an explicit
            // y-band pixel range when the resolver recorded one; otherwise
            // identical to `panel.h`.
            let band_extent = crate::render::marks::channels::band_extent_or(&ctx.scales.y, panel.h);
            let tick_half_raw = (band_extent / n_cats as f64 / n_groups) * ctx.mark_style.misc.band_size.unwrap_or(0.3);
            // Clamp to the Dodge sub-band (GH #66); see the ordinal-x mode's
            // analogous comment above — this tick spans y, the same axis
            // Dodge offsets `cy` along, so the overlap risk (and the
            // exact-round-trip `*2.0`/`/2.0` no-op) is the same.
            let tick_half = crate::render::position::clamp_to_dodge_sub_band(tick_half_raw * 2.0, ctx.batch) / 2.0;
            let mut acc = MarkNodes::with_capacity(xs.len());
            for i in 0..xs.len() {
                let xv = match xs[i] { Some(v) if v.is_finite() => v, _ => continue };
                let yv = match &ys[i] { Some(s) => s.as_str(), None => continue };
                let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
                let cy = match ctx.scales.y.to_pixel_str(yv) { Some(p) => p, None => continue };
                let px = px + x_offsets[i];
                let cy = cy + y_offsets[i];
                acc.push(SceneNode::Line {
                    x1: px,
                    y1: cy - tick_half,
                    x2: px,
                    y2: cy + tick_half,
                    style: row_stroke(i),
                }, i);
            }
            let (nodes, data_indices) = acc.finalize();
            let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
            return MarkBuildResult {
                kind: MarkBatchKind::Tick,
                nodes,
                data_indices: Some(data_indices),
                tooltips,
                hrefs,
                descriptions,
            };
        }
    }

    // Ordinal x only (no y field) → vertical crossbars at each category, anchored at panel bottom.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) && y_field(ctx, spec).is_none() {
        let xs = match col_as_positional_category_str(ctx.batch, xf) {
            Ok(v) => v,
            Err(_) => return empty(),
        };
        let n_cats = {
            let mut set = std::collections::HashSet::<&str>::new();
            for v in xs.iter().flatten() { set.insert(v.as_str()); }
            set.len().max(1)
        };
        // Band-geometry unification (#39 phase 2): y is unencoded in this
        // mode, so `ctx.scales.y` is the dummy unit scale, whose
        // `explicit_band_extent()` is always `None` — this always falls
        // through to `panel.h`, matching the site-pairing convention
        // (cross-axis tick length keyed to the panel-h term regardless of
        // which field is ordinal).
        //
        // GH #66 sub-band clamp NOT applicable here: Dodge offsets `cx` (the
        // band axis, x), but this tick's length runs along the CROSS axis
        // (y). Sibling dodge groups land at different `cx` and never share a
        // column, so this tick length cannot collide with a neighbouring
        // group regardless of `padding` — see the analogous "ordinal y only"
        // mode above.
        let band_extent = crate::render::marks::channels::band_extent_or(&ctx.scales.y, panel.h);
        let tick_half = (band_extent / n_cats as f64 / n_groups) * ctx.mark_style.misc.band_size.unwrap_or(0.3);
        let baseline_y = panel.y + panel.h;
        let mut acc = MarkNodes::with_capacity(xs.len());
        for i in 0..xs.len() {
            let xv = match &xs[i] { Some(s) => s.as_str(), None => continue };
            let cx = match ctx.scales.x.to_pixel_str(xv) { Some(p) => p, None => continue };
            let cx = cx + x_offsets[i];
            let by = baseline_y + y_offsets[i];
            acc.push(SceneNode::Line {
                x1: cx,
                y1: by,
                x2: cx,
                y2: by - 2.0 * tick_half,
                style: row_stroke(i),
            }, i);
        }
        let (nodes, data_indices) = acc.finalize();
        let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
        return MarkBuildResult {
            kind: MarkBatchKind::Tick, nodes, data_indices: Some(data_indices),
            tooltips, hrefs, descriptions,
        };
    }

    // Quantitative x → rug-style vertical tick at panel baseline.
    let tick_len = ctx.theme.sizes.tick_size * 2.0;
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return empty() };
    let baseline_y = panel.y + panel.h;
    let mut acc = MarkNodes::with_capacity(xs.len());
    for (i, xopt) in xs.iter().enumerate() {
        let xv = match xopt { Some(v) if v.is_finite() => *v, _ => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let px = px + x_offsets[i];
        let by = baseline_y + y_offsets[i];
        acc.push(SceneNode::Line {
            x1: px,
            y1: by,
            x2: px,
            y2: by - tick_len,
            style: row_stroke(i),
        }, i);
    }
    let _ = y_field(ctx, spec);

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Tick,
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
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn tick_ordinal_x_emits_horizontal_tick_at_y() {
        // Phase 10c-pre: ordinal x + quantitative y → horizontal tick (boxplot median).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "median".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat",    arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("median", arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![3.0, 5.0, 7.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        // 3 horizontal ticks — one per (cat, median) row.
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. })).count(), 3, "expected 3 horizontal tick lines");
        // Lines are horizontal: x1 != x2 for at least one line.
        let has_horizontal = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = n { (x1 - x2).abs() > f64::EPSILON } else { false }
        });
        assert!(has_horizontal, "ticks must have different x1 and x2 endpoints");
    }

    #[test]
    fn tick_emits_one_line_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. })).count(), 3);
    }

    #[test]
    fn tick_uses_stroke_color_when_mark_style_stroke_is_set() {
        // Composite structural ticks (boxplot caps, median, errorbar caps) pass
        // stroke: "theme:label" via mark_kwargs. After resolve_mark_style,
        // mark_style.paint.stroke = Some(label_color). tick.rs must use that stroke
        // color, not fall back to mark_style.paint.fill (which is mark_color = blue).
        use arrow::array::StringArray;
        use crate::spec::mark_style::MarkKwargsSpec;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("val", arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Float64Array::from(vec![5.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();

        // Stroke = #aabbcc (distinctive; different from mark_color and fill)
        let overrides = MarkKwargsSpec { stroke: Some("#aabbcc".into()), ..Default::default() };
        let mark_style = crate::render::draw::resolve_mark_style(Some(&overrides), &theme, &Mark::Tick);
        // Confirm fill is NOT #aabbcc (it's still mark_color blue)
        assert_ne!(mark_style.paint.fill.red, 0xAA, "fill should not be the stroke color");
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let line = result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.clone()) } else { None }
        }).expect("expected at least one Line node");
        assert_eq!(line.color.r, 0xAA, "tick stroke must use mark_style.paint.stroke, not mark_style.paint.fill");
        assert_eq!(line.color.g, 0xBB, "tick stroke must use mark_style.paint.stroke, not mark_style.paint.fill");
        assert_eq!(line.color.b, 0xCC, "tick stroke must use mark_style.paint.stroke, not mark_style.paint.fill");
    }

    #[test]
    fn tick_ordinal_y_only_emits_horizontal_lines() {
        // Regression: ordinal-y-only (no x) previously returned empty nodes because
        // col_as_f64 failed on the string column. Now it should emit one horizontal
        // crossbar per row centered on the ordinal band.
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: None,
                y: Some(EncodingSpec {
                    field: "cat".into(),
                    type_: Some(crate::spec::encoding::DataType::Ordinal),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let line_count = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. }))
            .count();
        assert_eq!(line_count, 3, "expected 3 horizontal tick lines for 3 ordinal-y categories, got {line_count}");

        // All lines must be horizontal: y1 == y2.
        for node in &result.nodes {
            if let ferrum_scene::SceneNode::Line { y1, y2, .. } = node {
                assert!(
                    (y1 - y2).abs() < f64::EPSILON,
                    "ordinal-y-only ticks must be horizontal (y1={y1}, y2={y2})"
                );
            }
        }

        // Lines must have non-zero width (x1 != x2).
        let has_width = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = n {
                (x2 - x1).abs() > f64::EPSILON
            } else {
                false
            }
        });
        assert!(has_width, "ordinal-y-only tick lines must have non-zero horizontal extent");
    }

    #[test]
    fn tick_ordinal_x_only_emits_vertical_lines() {
        // Ordinal-x-only (no y field) should emit one vertical crossbar per row
        // anchored at the panel bottom, centered on the ordinal band.
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "cat".into(),
                    type_: Some(crate::spec::encoding::DataType::Ordinal),
                    ..Default::default()
                }),
                y: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["x", "y", "z"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let line_count = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Line { .. }))
            .count();
        assert_eq!(line_count, 3, "expected 3 vertical tick lines for 3 ordinal-x categories, got {line_count}");

        // All lines must be vertical: x1 == x2.
        for node in &result.nodes {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = node {
                assert!(
                    (x1 - x2).abs() < f64::EPSILON,
                    "ordinal-x-only ticks must be vertical (x1={x1}, x2={x2})"
                );
            }
        }

        // Lines must have non-zero height (y1 != y2).
        let has_height = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Line { y1, y2, .. } = n {
                (y2 - y1).abs() > f64::EPSILON
            } else {
                false
            }
        });
        assert!(has_height, "ordinal-x-only tick lines must have non-zero vertical extent");
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Tick has six modes; all now use MarkNodes. Tests cover:
    //   - quantitative-x rug (primary skip path tested below)
    //   - ordinal-x + quantitative-y (second mode, href channel)
    //
    // Fail-before: `build_metadata(ctx)` produced full per-row vectors before any
    // loop. When row 1 was skipped, node 1 received row 1's metadata (the bug).
    //
    // Pass-after: `MarkNodes` + `build_metadata_for_indices` aligns to kept rows.

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Regression: quantitative-x rug tick with a non-finite x skips that row.
    /// The tooltip on each surviving node must point to its true source row.
    ///
    /// Batch: 3 rows, x=[10.0, NaN, 90.0], tooltip=["tip_a","tip_b","tip_c"].
    /// Row 1 (NaN x) is skipped → 2 nodes. Node 1 must have "tip_c", not "tip_b".
    #[test]
    fn tick_quant_x_skipped_nan_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has NaN x → skipped by `v.is_finite()` guard.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, f64::NAN, 90.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 tick nodes after NaN-x skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a' (row 0); got '{t0}'");

        // Node 1 → row 2 → "tip_c". Old code (full-row): "tip_b" (the bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'. \
             This fails on pre-migration code using build_metadata(ctx).");
    }

    /// Href-channel alignment on the ordinal-x + quantitative-y mode.
    /// Row 1 has a null y-value → skipped. Href on node 1 must be "url_c" (row 2),
    /// not "url_b" (row 1, old bug).
    #[test]
    fn tick_ordinal_x_quant_y_skipped_null_y_href_aligned() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "val".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                href: Some(EncodingSpec { field: "url".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("val", DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("url", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![Some(10.0_f64), None, Some(80.0)])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 2,
            "expected 2 tick nodes after null-y skip; got {}", result.nodes.len());

        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_a"), "node 0 href must be 'url_a'");
        assert_eq!(hrefs[1].as_deref(), Some("url_c"),
            "node 1 href must be 'url_c' (row 2), not 'url_b' (row 1); \
             old build_metadata would give 'url_b'");
    }

    /// No-skip backward-compat (quantitative-x rug mode): all rows are finite →
    /// all nodes produced, tooltips in original row order.
    #[test]
    fn tick_no_skip_tooltips_unchanged() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",   DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce tick nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }

    /// C7 regression guard: after migrating tick's per-row opacity to the shared
    /// `OpacityResolver`, each tick line must still carry its own row's opacity
    /// value (per-row sampling, not a single constant). Fails if a dedup ever
    /// collapses the per-row sample to one value or drops the encoding column.
    #[test]
    fn tick_per_row_opacity_is_sampled_per_row() {
        use crate::spec::encoding::DataType as SDT;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![0.2_f64, 0.5, 0.9])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.opacity) } else { None }
        }).collect();
        assert_eq!(opacities, vec![0.2, 0.5, 0.9],
            "each tick must carry its own row's opacity (per-row OpacityResolver sample)");
    }

    /// C7 family-consistency guard: tick now clamps out-of-range opacity to
    /// `[0, 1]` and falls non-finite values back to the default, matching every
    /// other mark via the shared `OpacityResolver`. (Before C7 tick passed these
    /// raw, which could emit SVG with an invalid `opacity` attribute.) No real
    /// chart feeds out-of-range opacity, so goldens are unaffected.
    #[test]
    fn tick_opacity_is_clamped_and_finite_checked() {
        use crate::spec::encoding::DataType as SDT;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        // Row 0: > 1 → clamps to 1.0. Row 1: NaN → falls back to default opacity.
        // Row 2: in-range → passes through.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 90.0])),
            Arc::new(Float64Array::from(vec![1.5_f64, f64::NAN, 0.3])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let default_opacity = mark_style.paint.opacity;
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { style, .. } = n { Some(style.opacity) } else { None }
        }).collect();
        assert_eq!(opacities, vec![1.0, default_opacity, 0.3],
            "tick opacity must clamp >1 to 1.0, fall NaN to default, pass in-range through");
    }

    // ── Dodge band-narrowing (Task 3c remediation) ───────────────────────────

    /// Ordinal-x + quantitative-y tick spec (boxplot median line), plus a batch
    /// that optionally carries the synthetic `__pos_x_offset__` /
    /// `__pos_y_offset__` columns and the `__dodge_n_groups__` schema-metadata
    /// key. Two categories (a, b), two dodge groups per category (4 rows).
    fn ordinal_tick_offset_batch(
        x_offsets: Option<Vec<f64>>,
        n_groups_metadata: Option<usize>,
    ) -> (ChartSpec, arrow::record_batch::RecordBatch) {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::StringArray;
        use std::collections::HashMap;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "median".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let mut fields = vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("median", DataType::Float64, false),
        ];
        let mut cols: Vec<arrow::array::ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
            Arc::new(Float64Array::from(vec![3.0, 4.0, 5.0, 6.0])),
        ];
        if let Some(xo) = x_offsets {
            fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
            fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
            cols.push(Arc::new(Float64Array::from(xo.clone())));
            cols.push(Arc::new(Float64Array::from(vec![0.0; xo.len()])));
        }
        let mut schema = Schema::new(fields);
        if let Some(n) = n_groups_metadata {
            let mut metadata = HashMap::new();
            metadata.insert(crate::render::position::DODGE_N_GROUPS_KEY.to_string(), n.to_string());
            schema = schema.with_metadata(metadata);
        }
        let batch = arrow::record_batch::RecordBatch::try_new(Arc::new(schema), cols).unwrap();
        (spec, batch)
    }

    fn tick_horizontal_extents(spec: &ChartSpec, batch: &arrow::record_batch::RecordBatch) -> Vec<(f64, f64)> {
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(spec, batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec, panel: &panel, theme: &theme, scales: &scales, batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = n { Some((x1.min(*x2), x1.max(*x2))) } else { None }
        }).collect()
    }

    #[test]
    fn tick_ordinal_x_dodge_narrows_extent_by_group_count() {
        // Task 3c: a dodged boxplot median tick must shrink its band-dimension
        // extent (width) by the dodge group count, driven by the explicit
        // `__dodge_n_groups__` metadata Dodge stamps — not by counting distinct
        // offset values. panel.w=100, 2 cats → bandwidth 50, 2 groups →
        // sub_band 25 → offsets -12.5 / +12.5; tick_half = (100/2/2)*0.3 = 7.5,
        // so each tick spans 15.0.
        let (spec, batch) = ordinal_tick_offset_batch(Some(vec![-12.5, 12.5, -12.5, 12.5]), Some(2));
        let extents = tick_horizontal_extents(&spec, &batch);
        assert_eq!(extents.len(), 4, "expected 4 dodged ticks");
        for (lo, hi) in &extents {
            assert!(((hi - lo) - 15.0).abs() < 1e-9, "dodged tick width must be 15.0 (narrowed by 2 groups); got {}", hi - lo);
        }
        // Adjacent dodge groups within a category must not overlap in x.
        let mut sorted = extents.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in sorted.windows(2) {
            assert!(w[0].1 <= w[1].0 + 1e-9, "dodged tick intervals must not overlap: {:?} vs {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn tick_ordinal_x_jitter_offsets_without_dodge_metadata_not_narrowed() {
        // Regression for the quality-review bug: an ordinal-axis Jitter
        // (mark_tick is jitter-eligible per src/ferrum/position.py) writes
        // per-row noise into the SAME __pos_x_offset__/__pos_y_offset__ columns
        // Dodge uses, but never stamps __dodge_n_groups__. Four distinct
        // per-row noise values (which the old to_bits-distinct heuristic would
        // have misread as 4 dodge groups) must NOT narrow the tick: with no
        // metadata, n_dodge_groups returns 1, so tick_half = (100/2/1)*0.3 = 15.0
        // and each tick spans the full un-narrowed 30.0 width.
        let (spec, batch) = ordinal_tick_offset_batch(Some(vec![3.1, -7.4, 0.9, -2.2]), None);
        let extents = tick_horizontal_extents(&spec, &batch);
        assert_eq!(extents.len(), 4);
        for (lo, hi) in &extents {
            assert!(((hi - lo) - 30.0).abs() < 1e-9,
                "jitter-only tick width must stay 30.0 (no dodge narrowing); got {}", hi - lo);
        }
    }

    #[test]
    fn tick_ordinal_x_no_offsets_no_metadata_unchanged() {
        // Baseline regression: no offset columns, no metadata → n_groups == 1 →
        // tick_half = (100/2)*0.3 = 15.0, byte-identical to pre-Task-3c.
        let (spec, batch) = ordinal_tick_offset_batch(None, None);
        let extents = tick_horizontal_extents(&spec, &batch);
        assert_eq!(extents.len(), 4);
        for (lo, hi) in &extents {
            assert!(((hi - lo) - 30.0).abs() < 1e-9, "non-dodged tick width must be 30.0; got {}", hi - lo);
        }
    }

    // ── Band-geometry unification (#39 phase 2) ──────────────────────────────

    /// Ordinal-x + quantitative-y mode (boxplot median line): an explicit
    /// `BandScale` x-axis (extent 220px, distinct from panel.w = 300px) must
    /// drive `tick_half` from the scale's extent, not `panel.w`. Fails on
    /// pre-Task-3 code, which always divides by `panel.w`.
    #[test]
    fn tick_ordinal_x_quant_y_explicit_range_scales_half_extent() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "median".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("median", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![3.0, 5.0])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };

        let x_scale = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![40.0, 260.0], 0.0)
            .with_explicit_range(true);
        let scales = ResolvedScales {
            x: ScaleKind::Ordinal(x_scale),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![3.0, 5.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None,
            x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let widths: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { x1, x2, .. } = n { Some((x1 - x2).abs()) } else { None }
        }).collect();
        assert_eq!(widths.len(), 2);
        // |260 - 40| = 220 extent; 2 categories, 1 group, band_size 0.3 (default)
        // → tick_half = 220 / 2 * 0.3 = 33.0 → full width 66.0.
        for w in widths {
            assert!((w - 66.0).abs() < 1e-9, "expected tick width 66.0 from the 220px explicit extent, got {w}");
        }
    }

    /// Ordinal-y + quantitative-x mode (strip plot): an explicit `BandScale`
    /// y-axis drives `tick_half` from the scale's extent, not `panel.h`.
    #[test]
    fn tick_ordinal_y_quant_x_explicit_range_scales_half_extent() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::linear::LinearScale;
        use crate::scale::ordinal::OrdinalScale;
        use arrow::array::StringArray;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "val".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![3.0, 5.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 300.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };

        let y_scale = OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![40.0, 260.0], 0.0)
            .with_explicit_range(true);
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![3.0, 5.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Ordinal(y_scale),
            color: None, size: None, shape: None, opacity: None,
            x2: None, y2: None, y_slots: Default::default(),
        };

        let mark_style = resolve_mark_style(None, &theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let heights: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Line { y1, y2, .. } = n { Some((y1 - y2).abs()) } else { None }
        }).collect();
        assert_eq!(heights.len(), 2);
        // |260 - 40| = 220 extent; 2 categories, 1 group, band_size 0.3 (default)
        // → tick_half = 220 / 2 * 0.3 = 33.0 → full height 66.0.
        for h in heights {
            assert!((h - 66.0).abs() < 1e-9, "expected tick height 66.0 from the 220px explicit extent, got {h}");
        }
    }
}
