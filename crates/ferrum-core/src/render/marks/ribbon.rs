//! Ribbon mark: closed area between a lower and an upper bound along a shared
//! coordinate. Used by `mark_errorband`, `mark_smooth(ci=...)`,
//! `mark_learning_curve` / `mark_validation_curve` (with pre-aggregated CI
//! columns), and the QQ confidence-interval region.
//!
//! Path emission (spec §4.2.4), per color group when a color encoding is bound,
//! shown for the ordinary `y`..`y2` band at a shared x:
//!     M x[0] y[0]
//!     L x[1] y[1] ... L x[n-1] y[n-1]
//!     L x[n-1] y2[n-1] L x[n-2] y2[n-2] ... L x[0] y2[0]
//!     Z
//!
//! Walks the shared coordinate ascending on the lower edge, then descending on
//! the upper edge to close the polygon. Both edges trace under the mark's
//! `interpolate=` method via [`marks::curve`](super::curve), so the `L` runs
//! above become step or cubic commands when a non-linear method is set (spec
//! §6.1); the shape shown is the `linear` case. When a color encoding is bound,
//! rows are first partitioned by color category and each group emits its own
//! closed polygon (mirroring `line.rs`); otherwise all rows form one polygon.
//! Without per-color grouping, multi-series CI bands (e.g. train vs test
//! learning curves) collapse into a single zigzag polygon stitching across
//! categories.
//!
//! When neither `y2` nor `x2` is bound the drawer skips — ribbon requires a
//! band channel.

use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::{build_color_detail_groups, resolve_row_stroke_dash, stroke_dash_column_loader};
use crate::render::marks::curve::{CurveAxis, StepSpelling, Trace};
use crate::render::scale_resolve::{ColorInput, ScaleKind};

/// `mark_ribbon` spells its step risers as explicit `LineTo` vertices, like
/// its closest sibling `mark_area` (both are filled bands). See
/// [`StepSpelling`].
const RIBBON_STEP_SPELLING: StepSpelling = StepSpelling::Vertices;

/// Which channel pair forms a ribbon's band, and therefore which coordinate
/// its edges advance along.
///
/// A ribbon's band spans a lower and an upper bound at a SHARED coordinate,
/// and the pair that carries the bounds *is* the parameterization — the same
/// rule `mark_area`'s `BottomEdge::axis` states. A `y`..`y2` band spans y at a
/// shared x, so x is independent; `CoordFlip`'s `flip_positional` moves the
/// user's `y2` to `x2`, producing the mirror image — an `x`..`x2` band spanning
/// x at a shared y, with y independent.
///
/// This is read off `spec.encoding`, never off `ctx.coord_flipped()`: the flip
/// is one producer of the horizontal band, not its definition. Before `#172`(3)
/// the unbound `y2` made `build` return empty, so every flipped `mark_ribbon`,
/// `mark_errorband`, `mark_smooth(ci=)`, `mark_learning_curve`,
/// `mark_validation_curve` and QQ CI region silently drew nothing.
fn band_curve_axis(spec: &crate::spec::chart::ChartSpec) -> Option<(&str, CurveAxis)> {
    match (spec.encoding.y2.as_ref(), spec.encoding.x2.as_ref()) {
        (Some(e), _) => Some((e.field.as_str(), CurveAxis::X)),
        (None, Some(e)) => Some((e.field.as_str(), CurveAxis::Y)),
        (None, None) => None,
    }
}

/// Place a vertex from its shared coordinate and its value, in screen space.
///
/// One expression for both band orientations, so the top and bottom edges
/// cannot be assembled by two spellings that drift apart.
fn edge_point(axis: CurveAxis, shared: f64, value: f64) -> (f64, f64) {
    match axis {
        CurveAxis::X => (shared, value),
        CurveAxis::Y => (value, shared),
    }
}

/// Pixels for the coordinate both edges share, plus whether that coordinate is
/// ordinal (an ordinal shared coordinate keeps its row order instead of being
/// sorted by pixel).
///
/// `scale`/`field` are the shared axis's, not x's: under a flipped band the
/// shared coordinate is y. Keeping one resolver rather than a y-side copy is
/// what keeps the ordinal/f64/string fallback chain — and `clip=False`'s
/// extrapolating read — identical on both axes.
fn resolve_shared_pixels(
    ctx: &DrawCtx,
    scale: &ScaleKind,
    field: &str,
    n: usize,
) -> Option<(Vec<Option<f64>>, bool)> {
    // For ordinal scales, stringify the column (works for Utf8 and integer
    // dtypes) and look up band centers. This must come before the f64 path
    // because col_as_f64 succeeds on integer columns, but to_pixel_f64 returns
    // None for ordinal scales, producing all-None pixels.
    if matches!(scale, ScaleKind::Ordinal(_)) {
        if let Ok(xs_s) = col_as_positional_category_str(ctx.batch, field) {
            let pixels: Vec<Option<f64>> = xs_s
                .into_iter()
                .take(n)
                .map(|v| v.as_deref().and_then(|s| scale.to_pixel_str(s)))
                .collect();
            return Some((pixels, true));
        }
    }
    if let Ok(xs_f) = col_as_f64(ctx.batch, field) {
        // `DrawCtx::to_pixel`, not `to_pixel_f64` directly, so
        // `mark_ribbon(clip=False)` keeps its out-of-domain vertices like
        // `mark_line` does — one meaning for one public kwarg across the whole
        // line family. Clipped ribbons (the default) are byte-identical.
        let pixels: Vec<Option<f64>> = xs_f
            .into_iter()
            .take(n)
            .map(|v| v.and_then(|x| ctx.to_pixel(scale, x)))
            .collect();
        return Some((pixels, false));
    }
    if let Ok(xs_s) = col_as_str(ctx.batch, field) {
        let pixels: Vec<Option<f64>> = xs_s
            .into_iter()
            .take(n)
            .map(|v| v.as_deref().and_then(|s| scale.to_pixel_str(s)))
            .collect();
        return Some((pixels, true));
    }
    None
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{
        to_scene_fill_stroke, ClearablePaint, MarkBuildResult, MetadataColumns, PaintChannels,
    };
    use ferrum_scene::{MarkBatchKind, PathCmd};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Ribbon);

    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return empty(),
    };
    let (band_field, curve_axis) = match band_curve_axis(spec) {
        Some(v) => v,
        None => return empty(),
    };
    // The lower bound rides the same axis as the band channel; the remaining
    // positional channel is the coordinate both edges share.
    let (lower_field, shared_field, value_scale, shared_scale) = match curve_axis {
        CurveAxis::X => (yf, xf, &ctx.scales.y, &ctx.scales.x),
        CurveAxis::Y => (xf, yf, &ctx.scales.x, &ctx.scales.y),
    };

    let lows = match col_as_f64(ctx.batch, lower_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let highs = match col_as_f64(ctx.batch, band_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let n = lows.len().min(highs.len());
    let (shared_pixels, shared_is_ordinal) =
        match resolve_shared_pixels(ctx, shared_scale, shared_field, n) {
            Some(v) => v,
            None => return empty(),
        };
    let n = n.min(shared_pixels.len());

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

    // F-L02-03: ribbon honors `interpolate=` like line and area do. Until
    // batch E it read the setting not at all and always traced straight
    // edges, which is why the Python desugar refused every non-linear value.
    let interpolate = ctx.mark_style.line.interpolate.unwrap_or_default();
    let trace = Trace {
        method: interpolate,
        step: RIBBON_STEP_SPELLING,
        axis: curve_axis,
    };

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
                let sp_ok = shared_pixels.get(i).and_then(|v| *v).map(|p| p.is_finite()).unwrap_or(false);
                let lo_ok = lows.get(i).and_then(|v| *v).map(|v| v.is_finite()).unwrap_or(false);
                let hi_ok = highs.get(i).and_then(|v| *v).map(|v| v.is_finite()).unwrap_or(false);
                sp_ok && lo_ok && hi_ok
            })
            .collect();
        if !shared_is_ordinal {
            indices.sort_by(|&a, &b| {
                let sa = shared_pixels[a].unwrap_or(f64::NAN);
                let sb = shared_pixels[b].unwrap_or(f64::NAN);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
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
                let shared = shared_pixels.get(i).and_then(|v| *v)?;
                let lo = ctx.to_pixel(value_scale, lows[i]?)?;
                let hi = ctx.to_pixel(value_scale, highs[i]?)?;
                let xo = x_offsets.get(i).copied().unwrap_or(0.0);
                let yo = y_offsets.get(i).copied().unwrap_or(0.0);
                // Each pixel offset belongs to the axis it displaces, so the
                // pair follows the band's orientation along with everything
                // else: a shared-x band takes `xo` on the shared coordinate
                // and `yo` on both bounds, a shared-y band the reverse.
                let (shared_off, value_off) = match curve_axis {
                    CurveAxis::X => (xo, yo),
                    CurveAxis::Y => (yo, xo),
                };
                Some((shared + shared_off, lo + value_off, hi + value_off))
            })
            .collect();
        if pixels.len() < 2 {
            continue;
        }

        // Build closed path commands: lower edge with the shared coordinate
        // ascending, upper edge descending, Close. Both edges describe the SAME
        // function of the shared coordinate under the mark's interpolation
        // method (spec §6.1), so a curved band is bounded by two curves and a
        // stepped one by two matching step functions. Both edges are collected
        // in FORWARD (shared-ascending) order: `extend_reversed` owns walking
        // the upper edge back and mirroring `step-before`/`step-after`, which
        // are direction-dependent.
        let top_edge: Vec<(f64, f64)> =
            pixels.iter().map(|&(s, lo, _)| edge_point(curve_axis, s, lo)).collect();
        let bottom_edge: Vec<(f64, f64)> =
            pixels.iter().map(|&(s, _, hi)| edge_point(curve_axis, s, hi)).collect();
        let mut cmds = trace.subpath(&top_edge);
        trace.extend_reversed(&mut cmds, &bottom_edge);
        cmds.push(PathCmd::Close);

        // Color resolution — mirrors draw().
        // The fill paint is `Cleared` (T8 quality-review c2) only when the
        // color-scale lookup missed (or there was no group) AND the constant
        // fill was cleared by a literal "none"/"transparent" — a scale-resolved
        // `solid` color is never a user paint-clear. `stroke` is always the
        // literal constant (never scale-driven for ribbon), so its own
        // provenance carries through unconditionally.
        let fill = if has_color_groups {
            match key
                .as_deref()
                .and_then(|v| ctx.scales.color.as_ref().and_then(|s| s.lookup(v)))
            {
                Some(solid) => ClearablePaint::Paint(solid),
                None => ctx.mark_style.paint.fill_paint(),
            }
        } else {
            ctx.mark_style.paint.fill_paint()
        }
        .map(|c| crate::render::color::with_opacity(c, ctx.mark_style.paint.opacity));
        let stroke = ctx.mark_style.paint.stroke_paint();
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
            stroke,
            PaintChannels::new(ctx.mark_style.paint.stroke_width, 1.0, dash_vec.as_deref()),
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
        empty_reason: None,
        polar_spans: None,
        glyph_anchors: None,
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
    use ferrum_scene::PathCmd;
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

    /// With NEITHER band channel bound there is no band to draw, so the drawer
    /// still skips. Since `#172`(3) the gate is "no `y2` and no `x2`" rather
    /// than "no `y2`" — this fixture binds neither, which is what keeps it a
    /// statement about a missing band instead of a restatement of the flipped
    /// case below.
    #[test]
    fn ribbon_without_either_band_channel_skips() {
        let spec = ribbon_spec(false);
        assert!(spec.encoding.y2.is_none() && spec.encoding.x2.is_none(),
            "test premise: neither band channel is bound");
        let batch = two_col_batch(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        let result = render(&spec, &batch);
        assert!(result.nodes.iter().all(|n| !matches!(n, ferrum_scene::SceneNode::Path { .. })),
            "no path should be emitted without a band channel");
    }

    // ── Interpolation wiring (F-L02-03, batch E task 1) ──────────────────
    //
    // Exercised at the Rust level: `mark_ribbon`'s Python desugar still
    // refuses every non-linear `interpolate=` value (a later batch-E task
    // lifts that), so these drive `build` through `resolve_mark_style`
    // directly rather than through the public mark API.

    /// A four-point band's path commands under the given interpolation method.
    fn ribbon_cmds(interpolate: Option<&str>) -> Vec<PathCmd> {
        let spec = ribbon_spec(true);
        let batch = three_col_batch(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 2.0, 1.0, 3.0],
            vec![4.0, 6.0, 5.0, 7.0],
        );
        let overrides = interpolate.map(|value| crate::spec::mark_style::MarkKwargsSpec {
            interpolate: Some(value.into()),
            ..Default::default()
        });
        let result = render_with_overrides(&spec, &batch, overrides.as_ref());
        result.nodes.into_iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Path { commands, .. } = n { Some(commands) } else { None }
        }).expect("a y2-bound ribbon must emit a closed Path")
    }

    /// F-L02-03 headline: before batch E, ribbon read `interpolate` not at all,
    /// so EVERY value produced the same all-`LineTo` polygon. The three curve
    /// methods must now emit cubics on both edges of the band.
    #[test]
    fn ribbon_curve_methods_curve_both_edges() {
        for method in ["monotone", "basis", "cardinal"] {
            let cmds = ribbon_cmds(Some(method));
            let cubics = cmds.iter().filter(|c| matches!(c, PathCmd::CubicTo { .. })).count();
            assert!(
                cubics >= 6,
                "ribbon({method:?}) must curve the top edge AND the reversed bottom edge \
                 (>= 3 cubics each); got {cubics} in {cmds:?}"
            );
            assert_eq!(
                cmds.iter().filter(|c| matches!(c, PathCmd::MoveTo { .. })).count(),
                1,
                "the band stays ONE closed subpath"
            );
            assert!(matches!(cmds.last(), Some(PathCmd::Close)), "the ring is closed");
        }
    }

    /// Step methods reach ribbon too, spelled with explicit `LineTo` vertices
    /// like `mark_area`'s (never `mark_line`'s `H`/`V` shorthand).
    #[test]
    fn ribbon_step_uses_the_explicit_lineto_spelling() {
        let cmds = ribbon_cmds(Some("step"));
        assert!(
            !cmds.iter().any(|c| matches!(c, PathCmd::HLineTo { .. } | PathCmd::VLineTo { .. })),
            "ribbon must not emit H/V shorthand; got {cmds:?}"
        );
        assert!(
            cmds.len() > ribbon_cmds(None).len(),
            "a stepped band has more vertices than a linear one"
        );
    }

    /// Byte-identity guard: with `interpolate` unset — every chart that exists
    /// today, since the Python desugar refuses anything else — ribbon emits
    /// exactly the commands it emitted before the curve module landed.
    #[test]
    fn ribbon_linear_output_is_byte_identical_to_pre_change() {
        let spec = ribbon_spec(true);
        let batch = three_col_batch(vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 1.0], vec![4.0, 6.0, 5.0]);
        let result = render(&spec, &batch);
        let cmds = result.nodes.iter().find_map(|n| {
            if let ferrum_scene::SceneNode::Path { commands, .. } = n { Some(commands.clone()) } else { None }
        }).expect("a y2-bound ribbon must emit a closed Path");

        // MoveTo + 2 top LineTo + 3 reversed-bottom LineTo + Close.
        assert_eq!(cmds.len(), 7, "got {cmds:?}");
        assert!(matches!(cmds[0], PathCmd::MoveTo { .. }));
        assert!(cmds[1..6].iter().all(|c| matches!(c, PathCmd::LineTo { .. })),
            "the linear band is all straight segments; got {cmds:?}");
        assert!(matches!(cmds[6], PathCmd::Close));
        // The bottom edge is walked in reverse: cmds[3] rejoins at the LAST x.
        let (last_top_x, first_bottom_x) = match (&cmds[2], &cmds[3]) {
            (PathCmd::LineTo { x: a, .. }, PathCmd::LineTo { x: b, .. }) => (*a, *b),
            other => panic!("expected LineTo pair at the turn, got {other:?}"),
        };
        assert!((last_top_x - first_bottom_x).abs() < 1e-9,
            "the ring turns at the same x it reached; got {last_top_x} then {first_bottom_x}");
    }

    /// `#172`(3), the inversion of `ribbon_under_coord_flip_emits_nothing`.
    ///
    /// That test pinned the premise the old `RIBBON_CURVE_AXIS = CurveAxis::X`
    /// const rested on — that `CoordFlip` could not produce a horizontal
    /// ribbon, because `flip_positional` moves the user's `y2` to `x2` and
    /// ribbon read only `y2` — and instructed whoever taught ribbon to read
    /// `x2` to CHOOSE the curve axis at that read. This is that choice, taken
    /// and pinned: the band channel decides the axis (`band_curve_axis`), so a
    /// post-flip `x`..`x2` band draws, y-parameterized, and the `emits nothing`
    /// assertion is now the bug rather than the contract.
    ///
    /// What it asserts is geometry, not presence: the ring's vertices pair off
    /// into shared y-values, which is what "a band across x at a shared y"
    /// means and what a band mistakenly built against x would not satisfy.
    #[test]
    fn ribbon_under_coord_flip_draws_a_y_parameterized_band() {
        // A faithful POST-FLIP rendering spec. The user wrote
        // `.encode(x=cat, y=lo, y2=hi).coord(CoordFlip())`; `flip_positional`
        // swaps x<->y and x2<->y2, so the drawer receives:
        //   x  = lo   (lower bound, now on the value axis)
        //   x2 = hi   (upper bound — the band the flip MOVED here)
        //   y  = cat  (category)
        //   y2 = unbound
        //
        // Binding `x2` is what makes this a statement about the flip rather
        // than a restatement of "no band channel emits nothing" (covered by
        // `ribbon_without_either_band_channel_skips`): the band IS present in
        // the spec, on the channel only a flipped ribbon receives it on.
        //
        // The category column spans the band's value range, so every row's
        // endpoints are INSIDE both scales' domains and none is dropped as
        // out-of-domain. The command-count assertions below are what pin that:
        // a fixture that lost rows to clipping would emit a shorter ring and
        // fail rather than quietly prove less.
        // No row's bounds coincide under EITHER reading, so every vertex pair
        // below is a real span rather than a degenerate point — the control
        // and the flipped case are held to the same standard.
        let batch = three_col_batch(
            vec![0.0, 2.0, 4.0, 6.0], // "x" column — the category, post-flip on y
            vec![0.0, 2.0, 1.0, 3.0], // "y" column — lower bound, post-flip on x
            vec![4.0, 6.0, 5.0, 7.0], // "y2" column — upper bound, post-flip on x2
        );

        let mut spec = ribbon_spec(false);
        spec.coord = Some(crate::spec::coord::CoordKind::Flip);
        spec.encoding.x = Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() });
        spec.encoding.x2 = Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() });
        spec.encoding.y = Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() });
        assert!(spec.encoding.y2.is_none(), "test premise: the flip left y2 unbound");
        assert!(spec.encoding.x2.is_some(), "test premise: the band sits on x2");

        // The mirror control: the SAME columns bound the unflipped way. Its
        // ring must pair off on x (a band across y at a shared x); the flipped
        // ring below must pair off on y. Asserting both is what separates "the
        // band is y-parameterized" from "a band is drawn at all" — the pre-fix
        // const would have produced an x-paired ring here too, had it drawn
        // anything.
        let mut unflipped = ribbon_spec(true);
        unflipped.encoding.x = Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() });
        unflipped.encoding.y = Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() });
        unflipped.encoding.y2 = Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() });

        /// A linear 4-row band's ring: `M p0, L p1..p3, L q3..q0, Z`. Returns
        /// the eight vertices in emission order.
        fn ring_vertices(result: &crate::render::draw::MarkBuildResult) -> Vec<(f64, f64)> {
            let cmds = result
                .nodes
                .iter()
                .find_map(|n| match n {
                    ferrum_scene::SceneNode::Path { commands, .. } => Some(commands.clone()),
                    _ => None,
                })
                .expect("a band-bound ribbon must emit a closed Path");
            assert_eq!(
                cmds.len(),
                9,
                "a 4-row linear band is M + 7 L + Z — a shorter ring means rows were \
                 dropped and the pairing below would prove less; got {cmds:?}"
            );
            assert!(matches!(cmds[8], PathCmd::Close), "the ring is closed; got {cmds:?}");
            cmds[..8]
                .iter()
                .map(|c| match *c {
                    PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => (x, y),
                    ref other => panic!("expected a straight linear ring, got {other:?}"),
                })
                .collect()
        }

        let unflipped_ring = ring_vertices(&render(&unflipped, &batch));
        for k in 0..4 {
            let (lower, upper) = (unflipped_ring[k], unflipped_ring[7 - k]);
            assert!(
                (lower.0 - upper.0).abs() < 1e-9,
                "control: a y..y2 band spans y at a SHARED x, so vertex {k} and its partner \
                 must share an x; got {lower:?} and {upper:?}"
            );
            assert!(
                (lower.1 - upper.1).abs() > 1e-9,
                "control: and must differ in y — that difference IS the band; got \
                 {lower:?} and {upper:?}"
            );
        }

        let flipped_ring = ring_vertices(&render(&spec, &batch));
        for k in 0..4 {
            let (lower, upper) = (flipped_ring[k], flipped_ring[7 - k]);
            assert!(
                (lower.1 - upper.1).abs() < 1e-9,
                "a post-flip x..x2 band spans x at a SHARED y, so vertex {k} and its partner \
                 must share a y; got {lower:?} and {upper:?} — an x-paired ring here means the \
                 band was built against the wrong coordinate"
            );
            assert!(
                (lower.0 - upper.0).abs() > 1e-9,
                "and must differ in x — that difference IS the flipped band; got {lower:?} \
                 and {upper:?}"
            );
        }
    }

    /// The refusal reaches ribbon through the same chokepoint as line and area.
    #[test]
    fn ribbon_unknown_interpolate_refuses() {
        let overrides = crate::spec::mark_style::MarkKwargsSpec {
            interpolate: Some("smooth".into()),
            ..Default::default()
        };
        let err = resolve_mark_style(Some(&overrides), &ThemeInputs::default(), &Mark::Ribbon)
            .expect_err("an unknown interpolate value must refuse");
        assert_eq!(err, crate::render::RenderError::UnknownInterpolate { value: "smooth".into() });
    }
}
