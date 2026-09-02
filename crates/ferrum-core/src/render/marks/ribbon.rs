//! Ribbon mark: closed area between y(x) and y2(x) along a shared x. Used by
//! `mark_errorband`, `mark_smooth(ci=...)`, `mark_learning_curve` /
//! `mark_validation_curve` (with pre-aggregated CI columns), and the QQ
//! confidence-interval region.
//!
//! Path emission (spec §4.2.4), per color group when a color encoding is bound:
//!     M x[0] y[0]
//!     L x[1] y[1] ... L x[n-1] y[n-1]
//!     L x[n-1] y2[n-1] L x[n-2] y2[n-2] ... L x[0] y2[0]
//!     Z
//!
//! Walks x ascending on the top edge (y), then x descending on the bottom edge
//! (y2) to close the polygon. When a color encoding is bound, rows are first
//! partitioned by color category and each group emits its own closed polygon
//! (mirroring `line.rs`); otherwise all rows form one polygon. Without
//! per-color grouping, multi-series CI bands (e.g. train vs test learning
//! curves) collapse into a single zigzag polygon stitching across categories.
//!
//! When `encoding.y2` is unset the drawer silently skips — ribbon requires y2.

use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::{build_color_detail_groups, resolve_row_stroke_dash, stroke_dash_column_loader};
use crate::render::scale_resolve::{ColorInput, ScaleKind};

fn resolve_x_pixels(ctx: &DrawCtx, xf: &str, n: usize) -> Option<(Vec<Option<f64>>, bool)> {
    // For ordinal x scales, stringify the column (works for Utf8 and integer
    // dtypes) and look up band centers. This must come before the f64 path
    // because col_as_f64 succeeds on integer columns, but to_pixel_f64 returns
    // None for ordinal scales, producing all-None pixels.
    if matches!(&ctx.scales.x, ScaleKind::Ordinal(_)) {
        if let Ok(xs_s) = col_as_positional_category_str(ctx.batch, xf) {
            let pixels: Vec<Option<f64>> = xs_s
                .into_iter()
                .take(n)
                .map(|v| v.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
                .collect();
            return Some((pixels, true));
        }
    }
    if let Ok(xs_f) = col_as_f64(ctx.batch, xf) {
        let pixels: Vec<Option<f64>> = xs_f
            .into_iter()
            .take(n)
            .map(|v| v.and_then(|x| ctx.scales.x.to_pixel_f64(x)))
            .collect();
        return Some((pixels, false));
    }
    if let Ok(xs_s) = col_as_str(ctx.batch, xf) {
        let pixels: Vec<Option<f64>> = xs_s
            .into_iter()
            .take(n)
            .map(|v| v.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
            .collect();
        return Some((pixels, true));
    }
    None
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{
        to_scene_fill_stroke, MarkBuildResult, MetadataColumns,
    };
    use ferrum_scene::{MarkBatchKind, PathCmd};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Ribbon);

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return empty(),
    };
    let y2f = match spec.encoding.y2.as_ref() {
        Some(e) => e.field.as_str(),
        None => return empty(),
    };

    let ys = match col_as_f64(ctx.batch, yf) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let y2s = match col_as_f64(ctx.batch, y2f) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let n = ys.len().min(y2s.len());
    let (x_pixels, x_is_ordinal) = match resolve_x_pixels(ctx, xf, n) {
        Some(v) => v,
        None => return empty(),
    };
    let n = n.min(x_pixels.len());

    let cf = color_field(ctx, spec);
    // Ribbon only uses color (no detail channel); the helper's detail=None path
    // resolves to the color-only or fallback arm as appropriate.
    //
    // col_as_ordinal_category_str (not col_as_str), gated on the resolved
    // color scale being category-keyed (Categorical) or absent — mirrors
    // channels.rs::color_column_loader's `ColorInput::Category | None` gate.
    // Unlike `area` (which always force-resolves ColorScale::Categorical,
    // scale_resolve/mod.rs), ribbon does not: a Quantitative-typed/-inferred
    // color column resolves ColorScale::Continuous or ::Discretizing
    // (ColorInput::Numeric), and must NOT be grouped here — grouping by every
    // distinct numeric value fragments the band into (mostly) single-point
    // groups, which the `indices.len() < 2` / `pixels.len() < 2` gates below
    // then drop, silently blanking the plot under a still-drawn colorbar
    // legend (NF-A3 follow-up). For a Category-keyed (or scale-less) color
    // column, Int*/Float*/Bool dtypes still split into per-category bands
    // matching the legend domain built by the dtype-wide
    // distinct_values_in_order — `col_as_str` returns `Err` for non-Utf8,
    // which previously collapsed every category into one merged polygon.
    let color_input = ctx.scales.color.as_ref().map(|s| s.input());
    let color_values = cf.and_then(|f| {
        match color_input {
            Some(ColorInput::Category) | None => col_as_ordinal_category_str(ctx.batch, f).ok(),
            Some(ColorInput::Numeric) => None,
        }
    });
    // T12 fix round (spec §4.3, amended 2026-09-01): a categorical `stroke_dash`
    // field extends ribbon's series partition key from (color) to (color,
    // stroke_dash-field) — mirrors line.rs's `dash_cols.categorical` wiring
    // exactly, so a two-category `stroke_dash` encoding draws two distinct
    // bands instead of one merged polygon under a multi-entry dashed legend.
    // `stroke_dash_column_loader` gates the categorical column on a resolved
    // `StrokeDashScale`; a numeric (or absent) `stroke_dash` field yields
    // `categorical == None`, which skips subdivision entirely — grouping stays
    // byte-identical to pre-fix for that case (ribbon still does not consume a
    // numeric or literal-only `stroke_dash`, only the categorical-scale case
    // this round authorizes).
    let dash_cols = stroke_dash_column_loader(ctx);

    let groups = build_color_detail_groups(
        color_values.as_ref(),
        None,
        dash_cols.categorical.as_ref(),
        ctx.scales.color.is_some(),
        n,
    );

    // Phase 9c — per-row pixel offsets (Stack/Dodge ordinal).
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    let has_color_groups = color_values.is_some() && ctx.scales.color.is_some();

    let meta = MetadataColumns::from_ctx(ctx);

    // One closed polygon per color group, attached to the group's representative
    // source row (the first vertex's row after filtering/sorting). The
    // accumulator keeps `data_indices` in node order so
    // `build_metadata_for_indices` aligns each band's tooltip/href to that row
    // (bug #6). Before the fix, `data_indices` carried one entry per *contributing
    // row* while metadata was full-row, so node and metadata lengths diverged.
    let mut acc = MarkNodes::with_capacity(groups.len());

    for (key, rows) in groups {
        let mut indices: Vec<usize> = rows
            .into_iter()
            .filter(|&i| {
                let xp_ok = x_pixels.get(i).and_then(|v| *v).map(|p| p.is_finite()).unwrap_or(false);
                let yv_ok = ys.get(i).and_then(|v| *v).map(|v| v.is_finite()).unwrap_or(false);
                let y2v_ok = y2s.get(i).and_then(|v| *v).map(|v| v.is_finite()).unwrap_or(false);
                xp_ok && yv_ok && y2v_ok
            })
            .collect();
        if !x_is_ordinal {
            indices.sort_by(|&a, &b| {
                let xa = x_pixels[a].unwrap_or(f64::NAN);
                let xb = x_pixels[b].unwrap_or(f64::NAN);
                xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        if indices.len() < 2 {
            continue;
        }
        // Representative source row for this group's single polygon node: the
        // first kept vertex's row (in render order). Captured before the ring is
        // built so the node maps to a stable, well-defined row.
        let representative_row = indices[0];
        let pixels: Vec<(f64, f64, f64)> = indices
            .iter()
            .filter_map(|&i| {
                let cx = x_pixels.get(i).and_then(|v| *v)?;
                let yv = ys[i]?;
                let y2v = y2s[i]?;
                let cy = ctx.scales.y.to_pixel_f64(yv)?;
                let cy2 = ctx.scales.y.to_pixel_f64(y2v)?;
                let xo = x_offsets.get(i).copied().unwrap_or(0.0);
                let yo = y_offsets.get(i).copied().unwrap_or(0.0);
                Some((cx + xo, cy + yo, cy2 + yo))
            })
            .collect();
        if pixels.len() < 2 {
            continue;
        }

        // Build closed path commands: top edge x ascending, bottom edge x descending, Close.
        let mut cmds = Vec::with_capacity(pixels.len() * 2 + 2);
        let (x0, y0, _) = pixels[0];
        cmds.push(PathCmd::MoveTo { x: x0, y: y0 });
        for &(x, y, _) in &pixels[1..] {
            cmds.push(PathCmd::LineTo { x, y });
        }
        for &(x, _, y2) in pixels.iter().rev() {
            cmds.push(PathCmd::LineTo { x, y: y2 });
        }
        cmds.push(PathCmd::Close);

        // Color resolution — mirrors draw().
        // `fill_cleared` (T8 quality-review c2): true only when the color-scale
        // lookup missed (or there was no group) AND the constant fill was
        // cleared by a literal "none"/"transparent" — a scale-resolved `solid`
        // color is never a user paint-clear. `stroke` is always the literal
        // constant (never scale-driven for ribbon), so `stroke_cleared` follows
        // it unconditionally.
        let (fill, fill_cleared, stroke) = if has_color_groups {
            let looked_up = key
                .as_deref()
                .and_then(|v| ctx.scales.color.as_ref().and_then(|s| s.lookup(v)));
            let fill_cleared = looked_up.is_none() && ctx.mark_style.paint.fill_cleared;
            let solid = looked_up.unwrap_or(ctx.mark_style.paint.fill);
            let fill = crate::render::color::with_opacity(solid, ctx.mark_style.paint.opacity);
            (Some(fill), fill_cleared, ctx.mark_style.paint.stroke)
        } else {
            let fill = crate::render::color::with_opacity(ctx.mark_style.paint.fill, ctx.mark_style.paint.opacity);
            (Some(fill), ctx.mark_style.paint.fill_cleared, ctx.mark_style.paint.stroke)
        };
        // Dash resolution mirrors line.rs: sampled once from the group's
        // representative row. Only a resolved categorical `StrokeDashScale`
        // (dash_cols.categorical == Some) applies a pattern here — a numeric
        // or absent `stroke_dash` field keeps `None`, preserving byte-identity
        // for every case this fix round did not authorize: ribbon still never
        // resolves a numeric-index `stroke_dash`. Within the categorical
        // case, though, `resolve_row_stroke_dash` falls back to the mark
        // literal when the representative row's own dash category is null —
        // so a literal-only `stroke_dash` CAN reach a band's paint here, in
        // that one corner. Harmless (the band's dash still matches what a
        // per-row lookup would draw for that same null category), but real.
        let dash_vec = dash_cols.categorical.as_ref().and_then(|_| {
            resolve_row_stroke_dash(
                &dash_cols,
                ctx.scales.stroke_dash.as_ref(),
                representative_row,
                ctx.mark_style.paint.stroke_dash.as_deref(),
            )
        });
        let style = to_scene_fill_stroke(
            fill,
            fill_cleared,
            stroke,
            ctx.mark_style.paint.stroke_cleared,
            ctx.mark_style.paint.stroke_width,
            1.0,
            dash_vec.as_deref(),
        );
        acc.push(
            ferrum_scene::SceneNode::Path {
                commands: cmds,
                style,
                closed: true,
            },
            representative_row,
        );
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Ribbon,
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
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn ribbon_spec(with_y2: bool) -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                y2: if with_y2 {
                    Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() })
                } else {
                    None
                },
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        }
    }

    fn three_col_batch(xs: Vec<f64>, ys: Vec<f64>, y2s: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(Float64Array::from(y2s)),
            ],
        )
        .unwrap()
    }

    fn two_col_batch(xs: Vec<f64>, ys: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
            ],
        )
        .unwrap()
    }

    fn render(spec: &ChartSpec, batch: &RecordBatch) -> crate::render::draw::MarkBuildResult {
        render_with_overrides(spec, batch, None)
    }

    /// Like [`render`], but resolves `mark_style` with the given
    /// `MarkKwargsSpec` overrides instead of the theme defaults — used to
    /// exercise a cleared (`"none"`) `fill=`/`stroke=` through the real
    /// `resolve_mark_style` -> `build` path.
    fn render_with_overrides(
        spec: &ChartSpec,
        batch: &RecordBatch,
        overrides: Option<&crate::spec::mark_style::MarkKwargsSpec>,
    ) -> crate::render::draw::MarkBuildResult {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            spec,
            batch,
            (0.0, 100.0),
            (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(overrides, &theme, &Mark::Ribbon).unwrap();
        let ctx = DrawCtx {
            spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch,
            mark_style: &mark_style,
        };
        super::build(&ctx)
    }

    #[test]
    fn ribbon_emits_closed_path() {
        let spec = ribbon_spec(true);
        // Auto-domain is computed from `y` only; keep y2 values within [min(y), max(y)]
        // to avoid out-of-range scale projections in this isolated unit test.
        let batch = three_col_batch(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 2.0, 4.0, 6.0, 8.0],
            vec![1.0, 3.0, 5.0, 7.0, 8.0],
        );
        let result = render(&spec, &batch);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).count(), 1);
        // Path must be closed and start with MoveTo.
        let path = result.nodes.iter().find(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).unwrap();
        if let ferrum_scene::SceneNode::Path { commands, closed, .. } = path {
            assert!(*closed, "ribbon path must be closed");
            assert!(matches!(commands.first(), Some(ferrum_scene::PathCmd::MoveTo { .. })),
                "ribbon path must start with MoveTo");
        }
    }

    /// T8 quality-review finding 2: a cleared `stroke=` must not serialize as
    /// an explicit zero-alpha color (`rgba(0,0,0,0.000)` in SVG). Both SVG
    /// and interactive scene JSON are built from the same `FillStroke`
    /// (`crate::render::draw::to_scene_fill_stroke`), so pinning
    /// `style.stroke.is_none()` here covers both outputs — `push_fill_stroke`
    /// (svg.rs) omits the `stroke=` attribute entirely on `None`, and the
    /// interactive scene JSON is a direct serde serialization of this same
    /// node. This is the exact `mark_ribbon`/`mark_errorband` regression the
    /// finding named: composite.py passes `mark_kwargs={"stroke": "none"}`.
    #[test]
    fn ribbon_cleared_stroke_serializes_as_no_stroke_not_zero_alpha() {
        let spec = ribbon_spec(true);
        let batch = three_col_batch(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 2.0, 4.0, 6.0, 8.0],
            vec![1.0, 3.0, 5.0, 7.0, 8.0],
        );
        let overrides = crate::spec::mark_style::MarkKwargsSpec {
            stroke: Some("none".into()),
            ..Default::default()
        };
        let result = render_with_overrides(&spec, &batch, Some(&overrides));
        let path = result.nodes.iter().find(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).unwrap();
        if let ferrum_scene::SceneNode::Path { style, .. } = path {
            assert!(
                style.stroke.is_none(),
                "a cleared stroke must normalize to None, not Some(TRANSPARENT), \
                 so it never serializes as stroke=\"rgba(0,0,0,0.000)\""
            );
        } else {
            panic!("expected a Path node");
        }
    }

    /// Same hazard, fill side: a cleared `fill=` must resolve `None` (SVG's
    /// existing `None -> fill="none"` branch), not `Some(TRANSPARENT)`
    /// (which would serialize as `fill="rgba(0,0,0,0.000)"`).
    #[test]
    fn ribbon_cleared_fill_serializes_as_none_not_zero_alpha() {
        let spec = ribbon_spec(true);
        let batch = three_col_batch(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 2.0, 4.0, 6.0, 8.0],
            vec![1.0, 3.0, 5.0, 7.0, 8.0],
        );
        let overrides = crate::spec::mark_style::MarkKwargsSpec {
            fill: Some("none".into()),
            ..Default::default()
        };
        let result = render_with_overrides(&spec, &batch, Some(&overrides));
        let path = result.nodes.iter().find(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).unwrap();
        if let ferrum_scene::SceneNode::Path { style, .. } = path {
            assert!(
                style.fill.is_none(),
                "a cleared fill must normalize to None, not Some(TRANSPARENT), \
                 so it serializes as fill=\"none\", not fill=\"rgba(0,0,0,0.000)\""
            );
        } else {
            panic!("expected a Path node");
        }
    }

    #[test]
    fn ribbon_walks_x_ascending_then_descending() {
        let spec = ribbon_spec(true);
        // 3 rows: top edge has 3 vertices (M + 2 L), bottom edge reversed adds 3 more L.
        // Expect 5 LineTo commands total in the path (before Close).
        let batch = three_col_batch(
            vec![0.0, 5.0, 10.0],
            vec![0.0, 4.0, 10.0],
            vec![2.0, 6.0, 8.0],
        );
        let result = render(&spec, &batch);
        let path = result.nodes.iter().find(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).unwrap();
        if let ferrum_scene::SceneNode::Path { commands, .. } = path {
            let line_to_count = commands.iter().filter(|c| matches!(c, ferrum_scene::PathCmd::LineTo { .. })).count();
            assert_eq!(line_to_count, 5, "expected 5 LineTo commands, got {line_to_count}");
            // Verify the path structure: MoveTo + 2 top LineTo + 3 bottom-reversed LineTo + Close.
            assert_eq!(commands.len(), 7, "expected 7 path commands (M + 5L + Z), got {}", commands.len());
        }
    }

    #[test]
    fn ribbon_emits_one_polygon_per_color_group() {
        // Two interleaved series (A,B,A,B,A,B) sharing an x axis. Without
        // color grouping the renderer would stitch a single zigzag polygon
        // across both categories — regression fixed by the per-color
        // partitioning in `build`.
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec {
            field: "g".into(),
            type_: None,
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        use arrow::array::StringArray;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 5.0, 2.0, 6.0, 4.0, 7.0])),
                Arc::new(Float64Array::from(vec![1.0, 6.0, 3.0, 7.0, 5.0, 8.0])),
                Arc::new(StringArray::from(vec!["A", "B", "A", "B", "A", "B"])),
            ],
        )
        .unwrap();
        let result = render(&spec, &batch);
        assert_eq!(
            result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).count(),
            2,
            "expected one polygon per color group"
        );
    }

    // ── NF-A3: non-Utf8 color grouping matches the legend (batch-A task 5) ──
    // area.rs already used `col_as_ordinal_category_str`; ribbon.rs used
    // `col_as_str`, which returns `Err` for any non-Utf8 dtype and silently
    // collapsed every color category into one merged zigzag polygon.

    /// Regression (NF-A3): an Int64 color column must split into one closed
    /// polygon per distinct value, and each polygon's fill must be the
    /// legend-resolved color for its group (not the theme-default fallback
    /// every group collapsed to under the old `col_as_str` bug).
    #[test]
    fn ribbon_int64_color_splits_into_n_polygons_with_legend_parity() {
        use crate::spec::encoding::DataType as SDT;
        use arrow::array::Int64Array;
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec {
            field: "g".into(),
            type_: Some(SDT::Ordinal),
            ..Default::default()
        });
        // 3 groups (0,1,2) x 3 rows each, sharing an x axis (like the Utf8
        // 2-group fixture above but with an integer group column).
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Int64, false),
        ]));
        let n = 9usize;
        let xs: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y2s: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let groups: Vec<i64> = (0..n).map(|i| (i / 3) as i64).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(Float64Array::from(y2s)),
                Arc::new(Int64Array::from(groups)),
            ],
        )
        .unwrap();

        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let color_scale = scales.color.clone().expect("Int64 ordinal color must resolve a categorical color scale");
        let domain = match &color_scale {
            crate::render::scale_resolve::ColorScale::Categorical { domain, .. } => domain.clone(),
            other => panic!("expected a Categorical color scale, got {other:?}"),
        };
        assert_eq!(domain.len(), 3,
            "legend domain must carry 3 distinct group values (\"0\", \"1\", \"2\"); got {domain:?}");

        let result = render(&spec, &batch);
        let polygons: Vec<Option<ferrum_scene::Color>> = result.nodes.iter()
            .filter_map(|n| if let ferrum_scene::SceneNode::Path { style, closed: true, .. } = n { Some(style.fill) } else { None })
            .collect();
        assert_eq!(polygons.len(), 3,
            "Int64 ordinal color must emit one polygon per group (3 groups); got {}. \
             Old bug: col_as_str failed on Int64 → color_values=None → single merged polygon.",
            polygons.len());

        // Legend parity: 3 color groups must paint 3 distinct fills (not the
        // theme-default fallback every group collapsed to under the old bug).
        let mut distinct_fills: Vec<Option<ferrum_scene::Color>> = Vec::new();
        for fill in &polygons {
            if !distinct_fills.contains(fill) { distinct_fills.push(*fill); }
        }
        assert_eq!(distinct_fills.len(), 3,
            "3 color groups must paint 3 distinct fills, matching the legend; got {distinct_fills:?}");
    }

    /// Byte-identity guard (NF-A3): Utf8 color columns must keep producing
    /// distinct per-group fills after the `col_as_ordinal_category_str` swap,
    /// since that helper is an identity pass-through for Utf8. Extends
    /// `ribbon_emits_one_polygon_per_color_group` (group *count* only) with a
    /// legend-parity check on the fills themselves.
    #[test]
    fn ribbon_utf8_color_polygons_have_distinct_legend_fills() {
        use arrow::array::StringArray;
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 5.0, 2.0, 6.0, 4.0, 7.0])),
                Arc::new(Float64Array::from(vec![1.0, 6.0, 3.0, 7.0, 5.0, 8.0])),
                Arc::new(StringArray::from(vec!["A", "B", "A", "B", "A", "B"])),
            ],
        )
        .unwrap();
        let result = render(&spec, &batch);
        let polygons: Vec<Option<ferrum_scene::Color>> = result.nodes.iter()
            .filter_map(|n| if let ferrum_scene::SceneNode::Path { style, closed: true, .. } = n { Some(style.fill) } else { None })
            .collect();
        assert_eq!(polygons.len(), 2, "Utf8 color must still emit one polygon per group");
        assert_ne!(polygons[0], polygons[1],
            "2 Utf8 color groups must paint 2 distinct fills, matching the legend");
    }

    /// Shared fixture for the two tests below: 5 rows, all-distinct x/y/y2/color
    /// values (mirrors a realistic `color=<Float64>` quantitative encoding), so
    /// a per-distinct-value grouping bug would fragment into near-all singleton
    /// groups rather than coincidentally re-merging via repeats.
    fn quantitative_color_batch_and_spec(color_type: Option<crate::spec::encoding::DataType>) -> (ChartSpec, RecordBatch) {
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec { field: "c".into(), type_: color_type, ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0, 8.0])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0, 7.0, 8.0])),
                Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    /// Regression: a `color=<Float64>` field explicitly typed Quantitative
    /// resolves a numeric-keyed (`Continuous`) color scale, which must NOT be
    /// grouped by distinct color value — grouping fragments 5 all-distinct
    /// rows into 5 singleton bands, every one of which the `indices.len() < 2`
    /// gate then drops, silently blanking the plot under a still-drawn
    /// colorbar legend. The correct behavior is the pre-NF-A3-fix baseline:
    /// one merged closed polygon over all rows (color grouping is
    /// Categorical-only).
    #[test]
    fn ribbon_quantitative_float_color_explicit_type_merges_into_one_polygon() {
        use crate::spec::encoding::DataType as SDT;
        let (spec, batch) = quantitative_color_batch_and_spec(Some(SDT::Quantitative));
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        assert!(matches!(scales.color, Some(crate::render::scale_resolve::ColorScale::Continuous { .. })),
            "explicit Quantitative color must resolve ColorScale::Continuous; got {:?}", scales.color);

        let result = render(&spec, &batch);
        let polygons = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Path { closed: true, .. }))
            .count();
        assert_eq!(polygons, 1,
            "quantitative color must emit exactly 1 merged polygon over all rows, not fragment per distinct value; got {polygons}");
    }

    /// Same hazard as above, but via the type-inference path: no explicit
    /// `type_` on a Float64 color column. `infer_spec_type` infers
    /// Quantitative for numeric dtypes, so this must resolve the same
    /// numeric-keyed scale and the same single-merged-polygon behavior.
    #[test]
    fn ribbon_quantitative_float_color_inferred_type_merges_into_one_polygon() {
        let (spec, batch) = quantitative_color_batch_and_spec(None);
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        assert!(matches!(scales.color, Some(crate::render::scale_resolve::ColorScale::Continuous { .. })),
            "inferred Float64 color must resolve ColorScale::Continuous; got {:?}", scales.color);

        let result = render(&spec, &batch);
        let polygons = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Path { closed: true, .. }))
            .count();
        assert_eq!(polygons, 1,
            "inferred quantitative color must emit exactly 1 merged polygon over all rows, not fragment per distinct value; got {polygons}");
    }

    // ── #6 metadata/node alignment (group marks) ─────────────────────────────

    /// Bug #6: a multi-group ribbon attaches each band polygon to its group's
    /// REPRESENTATIVE source row (the first kept vertex's row in render order),
    /// not a per-node position. Groups are BLOCKED (A rows 0..3, B rows 3..6) so
    /// representatives are rows 0 and 3. Old code (`build_metadata(ctx)`) returns
    /// full per-row vectors of length 6, while only 2 nodes are emitted, so the
    /// metadata length diverges from the node count and node 1 mis-maps.
    #[test]
    fn ribbon_group_band_aligns_to_representative_row() {
        use crate::spec::encoding::EncodingSpec;
        use arrow::array::StringArray;
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() });
        spec.encoding.tooltip = Some(EncodingSpec { field: "tip".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        // 2 groups × 3 rows, blocked. y2 within domain to keep projections finite.
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0])),
            Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])),
            Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
            Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
        ]).unwrap();
        let result = render(&spec, &batch);

        let path_nodes: Vec<_> = result.nodes.iter()
            .filter(|n| matches!(n, ferrum_scene::SceneNode::Path { .. })).collect();
        assert_eq!(path_nodes.len(), 2, "expected one band per color group");
        let di = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(di.len(), result.nodes.len(), "nodes.len() must equal data_indices.len()");
        assert_eq!(di, &vec![0, 3], "each band maps to its group's first kept vertex row");

        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");
        let vals: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(vals, vec!["A", "B"],
            "node j tooltip is its group's representative-row label; \
             old full-row code (len 6) mis-indexes / mis-sizes the metadata.");
    }

    /// Bug #6 href channel: href aligns to the representative row with no tooltip
    /// encoded (href-without-tooltip soundness hole).
    #[test]
    fn ribbon_group_href_aligns_to_representative_row() {
        use crate::spec::encoding::EncodingSpec;
        use arrow::array::StringArray;
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() });
        spec.encoding.href = Some(EncodingSpec { field: "url".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0])),
            Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])),
            Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
            Arc::new(StringArray::from(vec!["url_A", "url_A", "url_A", "url_B", "url_B", "url_B"])),
        ]).unwrap();
        let result = render(&spec, &batch);

        assert!(result.tooltips.is_none(), "no tooltip encoding → tooltips None");
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), result.nodes.len(), "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_A"), "node 0 href = group A representative");
        assert_eq!(hrefs[1].as_deref(), Some("url_B"),
            "node 1 href = group B representative (row 3); old full-row code gives row 1 → 'url_A'.");
    }

    /// No-skip backward-compat: a single ungrouped ribbon (one node) keeps its
    /// representative at the first kept row and the tooltip output is unchanged.
    #[test]
    fn ribbon_no_grouping_tooltip_unchanged() {
        use crate::spec::encoding::EncodingSpec;
        use arrow::array::StringArray;
        let mut spec = ribbon_spec(true);
        spec.encoding.tooltip = Some(EncodingSpec { field: "tip".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0])),
            Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0, 6.0])),
            Arc::new(StringArray::from(vec!["t0", "t1", "t2", "t3"])),
        ]).unwrap();
        let result = render(&spec, &batch);

        assert_eq!(result.nodes.len(), 1, "single ribbon → one node");
        let di = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(di, &vec![0], "single group's representative is the first kept row (0)");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 1, "tooltip count == node count");
        assert_eq!(tooltips[0].fields[0].value, "t0", "representative tooltip is row 0's");
    }

    #[test]
    fn ribbon_no_y2_silently_skips() {
        let spec = ribbon_spec(false);
        let batch = two_col_batch(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        let result = render(&spec, &batch);
        assert!(result.nodes.iter().all(|n| !matches!(n, ferrum_scene::SceneNode::Path { .. })),
            "no path should be emitted without y2");
    }
}
