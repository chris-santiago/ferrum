//! mark_line: render rows as a polyline.
//!
//! Two axis modes:
//! - Quantitative x: read `x` as `f64`, project via `DrawCtx::to_pixel`, which
//!   swaps in the extrapolating scale twin when the resolved mark style's
//!   `clip == Some(false)`. `mark_function(clip=False)` is the producer of that
//!   style on the Python path; `area` and `ribbon` read through the same method
//!   so the field means one thing for any spec that sets it.
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
//! - A binned color (`Color(bin=...)`): one polyline per bin, in its bin's
//!   color, low bin first; detail and a categorical `stroke_dash` subdivide
//!   each bin's series as they do a color category's
//!   ([`BinnedColorSeries`]). Any other numeric color draws one series.
//! - Otherwise: one polyline over all rows in batch order.
//!
//! Every rule above links a group's rows in BATCH order by default. A line
//! that declares `mark_style.sort_vertices` joins each group's rows ascending
//! by its own data value on the axis it advances along instead (stable, so
//! tied rows keep their row order), except an `Ordinal` advancing axis, which
//! has no numeric value to order by and keeps batch order regardless.
//!
//! Width: a `size` field tapers each series along its length, every vertex
//! as wide as a point of the same value is across (see [`dot_diameter`]); a
//! constant `size=` kwarg is the same mapping at one width. `size` and
//! `stroke_width` both bound on one line refuse
//! ([`RenderError::LineWidthConflict`]). A `size` conditional the line's own
//! layer declares (`size=fm.when(sel).then(a).otherwise(b)`) is a size
//! binding too: it draws the series' constant width here and records each
//! width vertex's source row on the batch (`vertex_rows`), so the interactive
//! runtime can redraw every vertex at its own row's active branch. A sibling
//! layer's `size` conditional never reaches the line.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, color_field, stroke_width_column, x_field, y_field, DrawCtx};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::{build_color_detail_groups, grouping_series_values, resolve_row_stroke_dash, stroke_dash_column_loader, BinnedColorSeries};
use crate::render::marks::curve::{CurveAxis, StepSpelling, Trace};
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::{ColorInput, ScaleKind};
use crate::render::{LineWidthSources, NonPositiveLineWidth, RenderError, WidthBinding};

/// `mark_line` spells its step risers with the `HLineTo`/`VLineTo` shorthand.
/// See [`StepSpelling`] for why the two spellings coexist.
const LINE_STEP_SPELLING: StepSpelling = StepSpelling::Shorthand;

use ferrum_scene::dot_diameter;

/// The width of one drawn vertex of a line sized by `field`: the dot diameter
/// of its `(value, area)` on the size scale, or `fallback` (the series'
/// constant width) for a row with no usable size. A width that is not finite
/// and `> 0` refuses ([`RenderError::LineWidthNotPositive`]) rather than
/// reaching the scene's `widths`.
fn vertex_width(field: &str, sample: Option<(f64, f64)>, fallback: f64) -> Result<f64, RenderError> {
    let width = sample.map_or(fallback, |(_, area)| dot_diameter(area));
    if width.is_finite() && width > 0.0 {
        return Ok(width);
    }
    let field = field.to_owned();
    let source = match sample {
        Some((value, area)) => NonPositiveLineWidth::Scaled { field, value, area },
        None => NonPositiveLineWidth::Fallback { field, width },
    };
    Err(RenderError::LineWidthNotPositive { source: Box::new(source) })
}

/// Every `size` conditional this line's layer declared (the per-layer spec
/// carries only its own layer's conditionals). The interactive runtime folds
/// all of them (the batch lists them in `conditionals`), so each one's
/// branches are checked.
fn size_conditionals(spec: &crate::spec::chart::ChartSpec) -> Vec<&ferrum_scene::ConditionalEncoding> {
    spec.conditionals
        .iter()
        .map(|c| &c.encoding)
        .filter(|cond| cond.sets_size())
        .collect()
}

/// Refuse a `size` conditional with a branch whose dot diameter is not a
/// positive width.
fn refuse_non_positive_branch(cond: &ferrum_scene::ConditionalEncoding) -> Result<(), RenderError> {
    for branch in [&cond.if_selected, &cond.if_not] {
        let ferrum_scene::EncodingValue::Size { value: area } = *branch else { continue };
        let width = dot_diameter(area);
        if !(width.is_finite() && width > 0.0) {
            let selection = cond.selection_name.clone();
            let source = NonPositiveLineWidth::Conditional { selection, area };
            return Err(RenderError::LineWidthNotPositive { source: Box::new(source) });
        }
    }
    Ok(())
}

/// Refuse a line whose `size` and `stroke_width` are both bound, each by a
/// field or by a constant kwarg, and `size` also by its own conditional (a
/// theme width is never a binding).
fn refuse_width_conflict(ctx: &DrawCtx, size_conditional: bool) -> Result<(), RenderError> {
    let binding = |field: Option<&crate::spec::encoding::EncodingSpec>, constant: Option<f64>| {
        field
            .map(|e| WidthBinding::Field(e.field.clone()))
            .or(constant.map(WidthBinding::Constant))
    };
    let encoding = &ctx.spec.encoding;
    let size = binding(encoding.size.as_ref(), ctx.mark_style.line.size)
        .or(size_conditional.then_some(WidthBinding::Conditional));
    let stroke_width = binding(encoding.stroke_width.as_ref(), ctx.mark_style.line.stroke_width);
    match (size, stroke_width) {
        (Some(size), Some(stroke_width)) => Err(RenderError::LineWidthConflict {
            sources: Box::new(LineWidthSources { size, stroke_width }),
        }),
        _ => Ok(()),
    }
}

/// This mark's own series partition: `color` (or a binned color's keys),
/// `detail` and a categorical `stroke_dash` field, in that grouping key's
/// own order (a binned color's bin order) — the SAME partition `build`
/// draws one polyline per group from. Read by [`vertex_frame`] too, so the
/// census's vertex-unit count and the real render can never group a
/// statistic layer's rows differently.
pub(crate) fn series_groups(ctx: &DrawCtx) -> Vec<(Option<String>, Vec<usize>)> {
    let spec = ctx.spec;
    let n_rows = ctx.batch.num_rows();
    let cf = color_field(ctx, spec);
    // col_as_ordinal_category_str (not col_as_str), gated on the resolved
    // color scale being category-keyed (Categorical) or absent — mirrors
    // channels.rs::color_column_loader's `ColorInput::Category | None` gate.
    // Unlike `area` (which always force-resolves ColorScale::Categorical,
    // scale_resolve/mod.rs), line does not: a Quantitative-typed/-inferred
    // color column resolves ColorScale::Continuous or ::Discretizing
    // (ColorInput::Numeric), and must NOT be grouped here — grouping by every
    // distinct numeric value fragments the series into (mostly) single-point
    // groups, which the `points.len() < 2` gate below then drops, silently
    // blanking the plot under a still-drawn colorbar legend (NF-A3 follow-up).
    // For a Category-keyed (or scale-less) color column, Int*/Float*/Bool
    // dtypes still split into per-category polylines matching the legend
    // domain built by the dtype-wide distinct_values_in_order — `col_as_str`
    // returns `Err` for non-Utf8, which previously collapsed every category
    // into one merged polyline.
    //
    // A binned color is the one numeric color these marks draw: it splits the
    // series by bin instead, one series per bin in its bin's color, in bin
    // order, as the area's bands do.
    let binned = BinnedColorSeries::of(ctx);
    let color_input = ctx.scales.color.as_ref().map(|s| s.input());
    let category_values = cf.filter(|_| binned.is_none()).and_then(|f| {
        match color_input {
            Some(ColorInput::Category) | None => col_as_ordinal_category_str(ctx.batch, f).ok(),
            Some(ColorInput::Numeric) => None,
        }
    });
    let color_values = binned.as_ref().map(BinnedColorSeries::keys).or(category_values.as_ref());
    let detail_values = ctx.mark_style.group.detail.as_deref()
        .and_then(|f| col_as_ordinal_category_str(ctx.batch, f).ok());

    // A categorical `stroke_dash` field extends the series partition key
    // (color, detail) -> (color, detail, stroke_dash-field), spec §4.3, so
    // distinct style categories draw as distinct polylines rather than all
    // sampling the first row's dash within their color/detail group. A
    // numeric field (or none) yields `dash_cols.categorical == None`, which
    // skips subdivision entirely — byte-identical grouping.
    let dash_cols = stroke_dash_column_loader(ctx);

    let grouping = grouping_series_values(ctx);
    let mut groups = build_color_detail_groups(
        color_values,
        detail_values.as_ref(),
        dash_cols.categorical.iter().chain(&grouping),
        ctx.scales.color.is_some(),
        n_rows,
    );
    if let Some(binned) = &binned {
        binned.order(&mut groups);
    }
    groups
}

/// `x` and `y`: Category on an `Ordinal` scale, [`PlacementRead::ClipAware`]
/// otherwise — [`vertex_pixels`]' two readers, each dispatching on scale
/// kind the same way.
pub(crate) fn placements<'a>(ctx: &'a DrawCtx<'a>) -> Vec<crate::render::draw::Placement<'a>> {
    use crate::render::draw::{clip_aware_or_category, Placement};
    use crate::spec::layer::PositionalChannel;
    let spec = ctx.spec;
    let mut out = Vec::new();
    if let Some(f) = x_field(ctx, spec) { out.push(Placement::new(PositionalChannel::X, f, clip_aware_or_category(&ctx.scales.x))); }
    if let Some(f) = y_field(ctx, spec) { out.push(Placement::new(PositionalChannel::Y, f, clip_aware_or_category(&ctx.scales.y))); }
    out
}

/// Each row's vertex in pixels, `None` for a row `build` cannot place: a
/// null or unplaceable value on either axis, a category an `Ordinal`
/// scale does not map, or a value its continuous scale does not admit or
/// map. `None` overall when `x` or `y` is unbound or unreadable, where
/// `build` draws nothing.
///
/// `clip == Some(false)` (the `mark_function(clip=False)` escape at the
/// SVG-group level) must also stop this read from dropping an
/// out-of-domain vertex — otherwise the point never reaches the node list
/// for that escape to apply to. `DrawCtx::to_pixel` decides this for
/// the whole line family (line/area/ribbon all accept `clip=`), so the
/// kwarg means one thing everywhere; see its doc. Clipped marks are
/// byte-identical.
pub(crate) fn vertex_pixels(ctx: &DrawCtx) -> Option<Vec<Option<(f64, f64)>>> {
    let spec = ctx.spec;
    let (xf, yf) = (x_field(ctx, spec)?, y_field(ctx, spec)?);
    let axis_pixels = |scale: &ScaleKind, field: &str| -> Option<Vec<Option<f64>>> {
        Some(match scale {
            ScaleKind::Ordinal(_) => col_as_positional_category_str(ctx.batch, field)
                .ok()?
                .iter()
                .map(|opt| opt.as_deref().and_then(|s| scale.to_pixel_str(s)))
                .collect(),
            _ => col_as_f64(ctx.batch, field)
                .ok()?
                .iter()
                .map(|opt| opt.filter(|v| scale.admits_for_pixel(*v)).and_then(|v| ctx.to_pixel(scale, v)))
                .collect(),
        })
    };
    let xs_pix = axis_pixels(&ctx.scales.x, xf)?;
    let ys_pix = axis_pixels(&ctx.scales.y, yf)?;
    let n_rows = ctx.batch.num_rows();
    if xs_pix.len() != n_rows || ys_pix.len() != n_rows {
        return None;
    }
    Some(xs_pix.into_iter().zip(ys_pix).map(|(x, y)| Some((x?, y?))).collect())
}

/// This mark's [`VertexFrame`](crate::render::draw::VertexFrame): its own
/// [`series_groups`], [`vertex_pixels`]' placed rows and `build`'s minimum
/// of 2 vertices per polyline.
pub(crate) fn vertex_frame(ctx: &DrawCtx) -> crate::render::draw::VertexFrame {
    let groups = series_groups(ctx).into_iter().map(|(_, rows)| rows).collect();
    crate::render::draw::VertexFrame::from_vertices(groups, vertex_pixels(ctx), ctx.batch.num_rows(), 2)
}

pub fn build(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, RenderError> {
    use crate::render::draw::{
        to_scene_fill_stroke, to_scene_stroke, ClearablePaint, MarkBuildResult, MetadataColumns,
        PaintChannels,
    };
    use ferrum_scene::MarkBatchKind;

    let conditionals = size_conditionals(ctx.spec);
    let size_conditional = !conditionals.is_empty();
    refuse_width_conflict(ctx, size_conditional)?;
    conditionals.into_iter().try_for_each(refuse_non_positive_branch)?;

    let empty = || Ok(MarkBuildResult::empty(MarkBatchKind::Line));

    let spec = ctx.spec;
    let Some(vertices) = vertex_pixels(ctx) else { return empty() };

    let groups = series_groups(ctx);
    // The categorical `stroke_dash` column `series_groups` folds into the
    // series key, read again here for the per-row dash resolution below; a
    // numeric field (or none) yields `dash_cols.categorical == None`.
    let dash_cols = stroke_dash_column_loader(ctx);

    let interpolate = ctx.mark_style.line.interpolate.unwrap_or_default();
    // A straight series stays a `Polyline` (`StrokeStyle`-shaped); anything
    // that bends — steps and the three curves — needs a `Path`.
    let use_path = !interpolate.is_linear();
    // A line's points come straight from the positional encodings, so the
    // series' own axis governs — under `CoordFlip` that is y, and a curve
    // fitted against x there doubles back on itself.
    let curve_axis = ctx.series_curve_axis();
    let trace = Trace {
        method: interpolate,
        step: LINE_STEP_SPELLING,
        axis: curve_axis,
    };

    // A declared series join order (`mark_style.sort_vertices`, set by
    // `displot(kind="ecdf")`'s step line): each group's rows order ascending
    // by their own DATA value on the advancing axis, not its pixel, so a
    // reversed scale still steps after in data order; stable, so tied rows
    // keep their row order. An advancing axis on an `Ordinal` scale has no
    // numeric value to order by and keeps batch order, the rule `ribbon`
    // follows for its own series sort. `None`/unset leaves every group in
    // `series_groups`' own (batch) order, byte-identical.
    let advancing_field = match curve_axis {
        CurveAxis::X => x_field(ctx, spec),
        CurveAxis::Y => y_field(ctx, spec),
    };
    let advancing_scale = match curve_axis {
        CurveAxis::X => &ctx.scales.x,
        CurveAxis::Y => &ctx.scales.y,
    };
    let vertex_sort_values: Option<Vec<Option<f64>>> =
        (ctx.mark_style.line.sort_vertices && !matches!(advancing_scale, ScaleKind::Ordinal(_)))
            .then(|| advancing_field.and_then(|f| col_as_f64(ctx.batch, f).ok()))
            .flatten();

    // Per-row stroke channel vectors — sampled at the first valid row of each group.
    let sw_vals = stroke_width_column(ctx);
    // Per-row `(value, area)` from an encoded `size`, through the shared size
    // scale. A null, non-finite or unmappable row is `None` and takes its
    // series' constant width below (see `vertex_width`).
    let size_areas = match (&spec.encoding.size, &ctx.scales.size) {
        (Some(enc), Some(scale)) => col_as_f64(ctx.batch, &enc.field).ok().map(|values| {
            let areas = values
                .iter()
                .map(|v| {
                    v.filter(|v| v.is_finite())
                        .and_then(|v| scale.inner.to_pixel_f64(v).map(|area| (v, area)))
                })
                .collect::<Vec<Option<(f64, f64)>>>();
            (enc.field.as_str(), areas)
        }),
        _ => None,
    };
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
    // Per node, the source row of each width vertex (see `vertex_rows`).
    let mut vertex_rows: Vec<Vec<ferrum_scene::VertexRow>> = Vec::new();

    for (key, mut rows) in groups {
        if let Some(values) = &vertex_sort_values {
            rows.sort_by(|&a, &b| {
                let (va, vb) = (values.get(a).copied().flatten(), values.get(b).copied().flatten());
                match (va, vb) {
                    (Some(va), Some(vb)) => va.total_cmp(&vb),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                }
            });
        }
        let mut points: Vec<(f64, f64)> = Vec::new();
        let mut row_indices: Vec<usize> = Vec::new();
        for i in rows {
            let Some((cx, cy)) = vertices[i] else { continue };
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
        let dash_vec = resolve_row_stroke_dash(
            &dash_cols,
            ctx.scales.stroke_dash.as_ref(),
            first,
            ctx.mark_style.paint.stroke_dash.as_deref(),
        );
        let effective_dash = dash_vec.as_deref();
        // One width per kept point, aligned with `points`.
        let point_widths: Option<Vec<f64>> = size_areas
            .as_ref()
            .map(|(field, areas)| {
                row_indices
                    .iter()
                    .map(|&i| vertex_width(field, areas.get(i).copied().flatten(), group_stroke_width))
                    .collect::<Result<Vec<f64>, RenderError>>()
            })
            .transpose()?;

        // One series' stroke color: the color channel paints each group its own
        // palette color, and a series with no color channel takes the constant
        // stroke, then the fill. The color-scale arm falls back to `fill` rather
        // than to the theme stroke, which is why this is not
        // `resolve_stroke_paint`'s row-wise chain.
        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale)) =>
                scale.lookup(v).unwrap_or(ctx.mark_style.paint.fill),
            _ => ctx.mark_style.paint.stroke.unwrap_or(ctx.mark_style.paint.fill),
        };
        // The slot's own provenance outranks whatever the chain resolved: an
        // explicit `stroke="none"` paints no line at all. Before `#110` the
        // color-scale arm above overwrote the clear with the palette color, so
        // `mark_line(stroke="none").encode(color=…)` rendered byte-identically
        // to the same chart without the clear (`#120`'s last comment); the arm
        // with no color scale kept the clear's value but serialized it as
        // `stroke="rgba(0,0,0,0.000)"`. Both spellings now emit no stroke.
        let stroke_paint = ctx
            .mark_style
            .paint
            .stroke_paint()
            .cleared_or(with_opacity(stroke_color, group_opacity));

        let (node, vertex_points) = if use_path {
            let (cmds, cmd_points) = trace.subpath_with_rows(&points);
            // Each command takes the width of the point whose run it ends in.
            // `cmd_points` index `points`, which `point_widths` matches one for one.
            let widths = point_widths.map(|widths| cmd_points.iter().map(|&p| widths[p]).collect());
            // FA-11: honor the stroke_opacity and fill_opacity channels (the
            // latter was previously dropped) via the complete FillStroke builder.
            // Both node shapes carry the SAME `stroke_paint`, so a cleared stroke
            // renders identically under every `interpolate` spelling: this
            // `Path` branch (non-linear interpolation) and the `Polyline` branch
            // below (linear) each serialize no stroke attribute for a clear.
            let style = to_scene_fill_stroke(
                ClearablePaint::Absent,
                stroke_paint,
                PaintChannels {
                    stroke_width: group_stroke_width,
                    opacity: 1.0,
                    stroke_dash: effective_dash,
                    fill_opacity: group_fill_opacity,
                    stroke_opacity: group_stroke_opacity,
                    angle: 0.0,
                },
            );
            let node = ferrum_scene::SceneNode::Path {
                commands: cmds,
                style,
                closed: false,
                widths,
            };
            (node, cmd_points)
        } else {
            let mut stroke_style = to_scene_stroke(
                stroke_paint,
                group_stroke_width,
                1.0,
                effective_dash,
                ctx.mark_style.line.stroke_cap,
                ctx.mark_style.line.stroke_join,
            );
            stroke_style.stroke_opacity = group_stroke_opacity;
            // Polyline uses `StrokeStyle`, which has no fill channel — the
            // fill_opacity channel only applies to the `Path` (FillStroke)
            // variant emitted under non-linear interpolation.
            let node = ferrum_scene::SceneNode::Polyline {
                points: points.clone(),
                style: stroke_style,
                widths: point_widths,
            };
            (node, (0..points.len()).collect())
        };
        acc.push(node, first);
        if size_conditional {
            vertex_rows.push(
                // `vertex_points` index `points`, which `row_indices` matches
                // one for one.
                vertex_points.iter().map(|&p| ferrum_scene::VertexRow::Row(row_indices[p])).collect(),
            );
        }
    }

    let (nodes, data_indices) = acc.finalize();

    let result = MarkBuildResult::populated(MarkBatchKind::Line, nodes, data_indices, &meta);
    Ok(if size_conditional { result.with_vertex_rows(vertex_rows) } else { result })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use crate::spec::mark_style::MarkKwargsSpec;
    use ferrum_scene::SceneNode;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn line_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let fill_opacities: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Path { style, .. } = n { Some(style.fill_opacity) } else { None }
        }).collect();
        assert_eq!(fill_opacities.len(), 2, "expected one Path node per group");
        // Batch A §4.3 (sanctioned): the extent [0.25, 0.75] maps onto the theme
        // opacity band [0.1, 1.0], so the two groups take the band endpoints.
        // Before FA-11 the channel was dropped entirely and both were 1.0 — the
        // distinctness this guards is what matters.
        assert!((fill_opacities[0] - 0.1).abs() < 1e-9,
            "group A fill_opacity must be the band min 0.1; got {}", fill_opacities[0]);
        assert!((fill_opacities[1] - 1.0).abs() < 1e-9,
            "group B fill_opacity must be the band max 1.0; got {}", fill_opacities[1]);
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
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let path = result.nodes.iter().find_map(|n| {
            if let SceneNode::Path { style, .. } = n { Some(style) } else { None }
        }).expect("basis interpolation must emit a Path node");
        assert!((path.fill_opacity - 1.0).abs() < 1e-12,
            "absent fill_opacity encoding must leave fill_opacity = 1.0; got {}", path.fill_opacity);
    }

    // ── NF-A3: non-Utf8 color grouping matches the legend (batch-A task 5) ──
    // area.rs already used `col_as_ordinal_category_str`; line.rs used
    // `col_as_str`, which returns `Err` for any non-Utf8 dtype and silently
    // collapsed every color category into a single merged polyline.

    /// Shared 3-group / 4-rows-per-group fixture, mirroring
    /// `area.rs::make_grouped_color_batch_and_spec` so the two marks are
    /// pinned by the same layout: x cycles 0..4, y is the row index, and the
    /// `g` column carries the color group (dtype supplied by the caller).
    fn make_grouped_color_batch_and_spec(
        group_col: Arc<dyn arrow::array::Array>,
        group_dtype: DataType,
    ) -> (arrow::record_batch::RecordBatch, ChartSpec) {
        use crate::spec::encoding::DataType as SDT;
        let n = 12usize;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", group_dtype, false),
        ]));
        let xs: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            group_col,
        ]).unwrap();
        let spec = ChartSpec {
            mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal), ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
        };
        (batch, spec)
    }

    /// Regression (NF-A3): an Int64 color column must split into one
    /// polyline per distinct value, and each polyline's stroke must be the
    /// legend-resolved color for its group (not the theme-default fallback
    /// every group collapsed to under the old `col_as_str` bug).
    #[test]
    fn line_int64_color_splits_into_n_polylines_with_legend_parity() {
        use arrow::array::Int64Array;
        let groups: Vec<i64> = (0..12).map(|i| (i / 4) as i64).collect();
        let group_col: Arc<dyn arrow::array::Array> = Arc::new(Int64Array::from(groups));
        let (batch, spec) = make_grouped_color_batch_and_spec(group_col, DataType::Int64);
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let color_scale = ctx.scales.color.as_ref().expect("Int64 ordinal color must resolve a categorical color scale");
        let domain = match color_scale {
            crate::render::scale_resolve::ColorScale::Categorical { domain, .. } => domain.clone(),
            other => panic!("expected a Categorical color scale, got {other:?}"),
        };
        assert_eq!(domain.len(), 3,
            "legend domain must carry 3 distinct group values (\"0\", \"1\", \"2\"); got {domain:?}");

        let result = super::build(&ctx).unwrap();
        let polylines: Vec<&ferrum_scene::StrokeStyle> = result.nodes.iter()
            .filter_map(|n| if let SceneNode::Polyline { style, .. } = n { Some(style) } else { None })
            .collect();
        assert_eq!(polylines.len(), 3,
            "Int64 ordinal color must emit one polyline per group (3 groups); got {}. \
             Old bug: col_as_str failed on Int64 → color_values=None → single merged polyline.",
            polylines.len());

        // Legend parity: every polyline's stroke color must be one of the
        // legend-resolved colors for the 3 domain keys, not the fallback
        // fill color every group collapsed to under the old bug.
        // (`ferrum_scene::Color` derives `Eq` but not `Hash`, so dedup by
        // linear scan rather than `HashSet`.)
        let expected_colors: Vec<ferrum_scene::Color> = domain.iter()
            .map(|k| crate::render::draw::to_scene_color(color_scale.lookup(k).expect("domain key must resolve via lookup")))
            .collect();
        for style in &polylines {
            let color = style.color.expect("a color-encoded polyline must paint a stroke");
            assert!(expected_colors.contains(&color),
                "polyline stroke {color:?} is not a legend-resolved color for any of {domain:?}");
        }
        let mut distinct_strokes: Vec<ferrum_scene::Color> = Vec::new();
        for style in &polylines {
            let color = style.color.expect("a color-encoded polyline must paint a stroke");
            if !distinct_strokes.contains(&color) { distinct_strokes.push(color); }
        }
        assert_eq!(distinct_strokes.len(), 3,
            "3 color groups must paint 3 distinct stroke colors, matching the legend; got {distinct_strokes:?}");
    }

    /// Byte-identity guard (NF-A3): Utf8 color columns must produce the same
    /// grouping and legend-parity behavior after the `col_as_ordinal_category_str`
    /// swap, since that helper is an identity pass-through for Utf8.
    #[test]
    fn line_utf8_color_splits_into_n_polylines_with_legend_parity() {
        use arrow::array::StringArray;
        let groups: Vec<&str> = (0..12).map(|i| match i / 4 { 0 => "a", 1 => "b", _ => "c" }).collect();
        let group_col: Arc<dyn arrow::array::Array> = Arc::new(StringArray::from(groups));
        let (batch, spec) = make_grouped_color_batch_and_spec(group_col, DataType::Utf8);
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        let color_scale = ctx.scales.color.as_ref().expect("Utf8 color must resolve a categorical color scale");
        let domain = match color_scale {
            crate::render::scale_resolve::ColorScale::Categorical { domain, .. } => domain.clone(),
            other => panic!("expected a Categorical color scale, got {other:?}"),
        };
        assert_eq!(domain, vec!["a".to_string(), "b".to_string(), "c".to_string()]);

        let result = super::build(&ctx).unwrap();
        let polylines: Vec<&ferrum_scene::StrokeStyle> = result.nodes.iter()
            .filter_map(|n| if let SceneNode::Polyline { style, .. } = n { Some(style) } else { None })
            .collect();
        assert_eq!(polylines.len(), 3, "Utf8 color must still emit one polyline per group");

        let mut distinct_strokes: Vec<ferrum_scene::Color> = Vec::new();
        for style in &polylines {
            let color = style.color.expect("a color-encoded polyline must paint a stroke");
            if !distinct_strokes.contains(&color) { distinct_strokes.push(color); }
        }
        assert_eq!(distinct_strokes.len(), 3,
            "Utf8 grouping must remain unaffected: 3 distinct stroke colors matching the legend");
    }

    /// Shared fixture for the two tests below: 5 rows, all-distinct x/y/color
    /// values (mirrors a realistic `color=<Float64>` quantitative encoding),
    /// so a per-distinct-value grouping bug would fragment into near-all
    /// singleton groups rather than coincidentally re-merging via repeats.
    fn quantitative_color_batch_and_spec(color_type: Option<crate::spec::encoding::DataType>) -> (arrow::record_batch::RecordBatch, ChartSpec) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![0.0, 10.0, 20.0, 30.0, 40.0])),
            Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5])),
        ]).unwrap();
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec { field: "c".into(), type_: color_type, ..Default::default() });
        (batch, spec)
    }

    /// Regression: a `color=<Float64>` field explicitly typed Quantitative
    /// resolves a numeric-keyed (`Continuous`) color scale, which must NOT be
    /// grouped by distinct color value — grouping fragments 5 all-distinct
    /// rows into 5 singleton groups, every one of which the `points.len() < 2`
    /// gate then drops, silently blanking the plot under a still-drawn
    /// colorbar legend. The correct behavior is the pre-NF-A3-fix baseline:
    /// one merged polyline over all rows (color grouping is Categorical-only).
    #[test]
    fn line_quantitative_float_color_explicit_type_merges_into_one_polyline() {
        use crate::spec::encoding::DataType as SDT;
        let (batch, spec) = quantitative_color_batch_and_spec(Some(SDT::Quantitative));
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        // Test premise: an explicit :Q color field must resolve a numeric-keyed
        // scale, not Categorical (otherwise this test isn't exercising the hazard).
        assert!(matches!(ctx.scales.color, Some(crate::render::scale_resolve::ColorScale::Continuous { .. })),
            "explicit Quantitative color must resolve ColorScale::Continuous; got {:?}", ctx.scales.color);

        let result = super::build(&ctx).unwrap();
        let polylines = result.nodes.iter().filter(|n| matches!(n, SceneNode::Polyline { .. })).count();
        assert_eq!(polylines, 1,
            "quantitative color must emit exactly 1 merged polyline over all rows, not fragment per distinct value; got {polylines}");
    }

    /// Same hazard as above, but via the type-inference path: no explicit
    /// `type_` on a Float64 color column. `infer_spec_type` infers
    /// Quantitative for numeric dtypes, so this must resolve the same
    /// numeric-keyed scale and the same single-merged-polyline behavior.
    #[test]
    fn line_quantitative_float_color_inferred_type_merges_into_one_polyline() {
        let (batch, spec) = quantitative_color_batch_and_spec(None);
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };

        assert!(matches!(ctx.scales.color, Some(crate::render::scale_resolve::ColorScale::Continuous { .. })),
            "inferred Float64 color must resolve ColorScale::Continuous; got {:?}", ctx.scales.color);

        let result = super::build(&ctx).unwrap();
        let polylines = result.nodes.iter().filter(|n| matches!(n, SceneNode::Polyline { .. })).count();
        assert_eq!(polylines, 1,
            "inferred quantitative color must emit exactly 1 merged polyline over all rows, not fragment per distinct value; got {polylines}");
    }

    // ── Interpolation wiring (batch E task 1) ───────────────────────────────
    //
    // The per-method command contracts live with the math, in
    // `marks::curve`'s own tests (they moved there with the step logic; they
    // came originally from `bug_hunt_marks_rendering_r2.rs`). What has to be
    // pinned HERE is the wiring: which method `build` reads, and which step
    // spelling `mark_line` asks for — the property that keeps every committed
    // step golden byte-identical across the relocation.

    /// Render 4 collinear-in-x points under `interpolate` and return the
    /// single emitted node's path commands.
    fn line_path_cmds(interpolate: &str) -> Vec<ferrum_scene::PathCmd> {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 2.0, 1.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { interpolate: Some(interpolate.into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        result.nodes.into_iter().find_map(|n| {
            if let SceneNode::Path { commands, .. } = n { Some(commands) } else { None }
        }).expect("non-linear interpolation must emit a Path node")
    }

    /// Byte-stability guard for the relocation: `mark_line`'s step risers stay
    /// spelled with the `HLineTo`/`VLineTo` shorthand, which is what every
    /// committed step golden (`H`/`V` in the `d` attribute) was blessed with.
    /// `mark_area` deliberately spells the same geometry with explicit
    /// `LineTo` vertices — see `area.rs`'s twin of this test.
    #[test]
    fn line_step_keeps_the_hv_shorthand_spelling() {
        for method in ["step", "step-before", "step-after"] {
            let cmds = line_path_cmds(method);
            assert!(
                cmds.iter().any(|c| matches!(c, ferrum_scene::PathCmd::HLineTo { .. }))
                    && cmds.iter().any(|c| matches!(c, ferrum_scene::PathCmd::VLineTo { .. })),
                "mark_line(interpolate={method:?}) must keep the H/V shorthand; got {cmds:?}"
            );
        }
    }

    /// F-L02-02: the three curve methods must compute real curves. Before this
    /// fix `mark_line(interpolate="monotone"|"basis"|"cardinal")` fell through
    /// to the linear arm and emitted straight `LineTo` segments under a `Path`
    /// node, so the advertised curves were indistinguishable from `linear`.
    #[test]
    fn line_curve_methods_emit_cubics_not_straight_segments() {
        for method in ["monotone", "basis", "cardinal"] {
            let cmds = line_path_cmds(method);
            let cubics = cmds.iter().filter(|c| matches!(c, ferrum_scene::PathCmd::CubicTo { .. })).count();
            assert!(
                cubics > 0,
                "mark_line(interpolate={method:?}) must emit CubicTo commands; got {cmds:?}"
            );
        }
    }

    // ── CoordFlip: the series advances along y ───────────────────────────
    //
    // The drawer receives an encoding already in RENDERING space — by the time
    // `build` runs, `prepare::build_layers` has applied
    // `Encoding::flip_positional`, so a flipped chart arrives with the VALUE on
    // x and the CATEGORY on y, while `spec.coord` still reads `Flip` (the same
    // flag `marks::rule` reads at draw time). These fixtures are authored that
    // way deliberately.

    /// Spec + batch for a flipped series: x carries a non-monotone value,
    /// y carries the evenly-advancing category.
    fn flipped_line_fixture() -> (ChartSpec, arrow::record_batch::RecordBatch) {
        use crate::spec::encoding::DataType as SDT;
        let mut spec = line_spec();
        spec.coord = Some(crate::spec::coord::CoordKind::Flip);
        spec.encoding.x = Some(EncodingSpec { field: "x".into(), type_: Some(SDT::Quantitative), ..Default::default() });
        spec.encoding.y = Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            // value on x (non-monotone), category on y (steadily advancing)
            Arc::new(Float64Array::from(vec![1.0, 4.0, 2.0, 5.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        ]).unwrap();
        (spec, batch)
    }

    /// A flipped `monotone` line must be fitted against y, the axis its points
    /// actually advance along. Fitted against x it doubles back on itself,
    /// because the flipped x carries the (non-monotone) value.
    ///
    /// Discriminating assertion: every control point's y sits at an exact third
    /// of its segment's y span. An x-fitted curve puts the thirds on x instead.
    #[test]
    fn coord_flip_line_fits_curves_against_y() {
        let (spec, batch) = flipped_line_fixture();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { interpolate: Some("monotone".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let cmds = super::build(&ctx).unwrap().nodes.into_iter().find_map(|n| {
            if let SceneNode::Path { commands, .. } = n { Some(commands) } else { None }
        }).expect("a curved line must emit a Path");

        let mut prev = match cmds[0] {
            ferrum_scene::PathCmd::MoveTo { x, y } => (x, y),
            ref other => panic!("expected a leading MoveTo, got {other:?}"),
        };
        let mut checked = 0;
        for cmd in &cmds[1..] {
            let ferrum_scene::PathCmd::CubicTo { c1y, c2y, x, y, .. } = *cmd else {
                panic!("expected CubicTo, got {cmd:?}");
            };
            let third = (y - prev.1) / 3.0;
            assert!(
                (c1y - (prev.1 + third)).abs() < 1e-9 && (c2y - (y - third)).abs() < 1e-9,
                "under CoordFlip the control points must sit on thirds of the Y span \
                 ({}, {}), got ({c1y}, {c2y}) — an x-fitted curve puts the thirds on x",
                prev.1 + third,
                y - third
            );
            prev = (x, y);
            checked += 1;
        }
        assert_eq!(checked, 3, "one cubic per segment");
    }

    /// The flipped-step case (adjudicated in scope): a step's TREAD advances
    /// the independent coordinate. Under `CoordFlip` that is y, so the tread is
    /// a `VLineTo` and the riser an `HLineTo` — the mirror of the unflipped
    /// spelling. Fitted against x, the treads run along the value axis and
    /// reverse direction instead of stepping down the category axis.
    #[test]
    fn coord_flip_line_steps_tread_along_y() {
        let (spec, batch) = flipped_line_fixture();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { interpolate: Some("step-after".into()), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let cmds = super::build(&ctx).unwrap().nodes.into_iter().find_map(|n| {
            if let SceneNode::Path { commands, .. } = n { Some(commands) } else { None }
        }).expect("a stepped line must emit a Path");

        // step-after: tread first, then riser. Under flip that is V then H.
        for pair in cmds[1..].chunks(2) {
            assert!(
                matches!(pair[0], ferrum_scene::PathCmd::VLineTo { .. }),
                "flipped step-after must tread along y (VLineTo) first; got {:?}",
                pair[0]
            );
            assert!(
                matches!(pair[1], ferrum_scene::PathCmd::HLineTo { .. }),
                "flipped step-after's riser must move x (HLineTo); got {:?}",
                pair[1]
            );
        }
        // The treads must advance monotonically down the category axis rather
        // than reversing direction along the value axis.
        let treads: Vec<f64> = cmds.iter().filter_map(|c| match c {
            ferrum_scene::PathCmd::VLineTo { y } => Some(*y),
            _ => None,
        }).collect();
        assert_eq!(treads.len(), 3, "one tread per segment");
        assert!(
            treads.windows(2).all(|w| w[1] < w[0]) || treads.windows(2).all(|w| w[1] > w[0]),
            "treads must advance monotonically along the category axis; got {treads:?}"
        );
    }

    /// The refusal fires at `resolve_mark_style`, before any geometry is
    /// built, and names the accepted set. Previously an unknown value rendered
    /// as straight segments, indistinguishable from a real `linear` chart.
    #[test]
    fn line_unknown_interpolate_refuses_with_the_accepted_set() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { interpolate: Some("mono".into()), ..Default::default() };
        let err = resolve_mark_style(Some(&overrides), &theme, &Mark::Line)
            .expect_err("an unknown interpolate value must refuse");
        assert_eq!(
            err,
            crate::render::RenderError::UnknownInterpolate { value: "mono".into() }
        );
        let message = err.to_string();
        assert!(message.contains("monotone") && message.contains("cardinal"),
            "the refusal must name the accepted set; got {message}");
    }

    // ── clip=False extrapolation (batch-E T13, mark_function completeness) ──
    //
    // `mark_function` desugars to `mark_line`; `clip=False` on the resulting
    // layer sets `mark_style.misc.clip = Some(false)`. Before this fix, an
    // out-of-domain y (or x) vertex became `None` at the scale-mapping seam
    // regardless of `clip`, so it never reached `points` and T9's SVG-group
    // clip escape had nothing outside the panel to reveal. These tests build
    // `ResolvedScales` by hand (not `resolve_scales`, which infers the domain
    // FROM the data and would make "out of domain" unreachable) so the y
    // scale's domain [0,10] is narrower than the batch's real y range.

    fn narrow_domain_scales() -> crate::render::scale_resolve::ResolvedScales {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::linear::LinearScale;
        ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 10.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 10.0], vec![0.0, 100.0], false, false)),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
        }
    }

    /// RED-proven: with the panel's y domain [0,10] (range [0,100]) and a
    /// batch whose third row's y=20 is out of domain, `clip=Some(false)`
    /// must keep all 3 points, with the third landing OUTSIDE the panel's
    /// pixel range (y=200, past the [0,100] range) instead of being dropped.
    #[test]
    fn line_clip_false_keeps_out_of_domain_vertex_outside_panel() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 5.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let scales = narrow_domain_scales();
        let overrides = MarkKwargsSpec { clip: Some(false), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let polyline_ys: Vec<f64> = result.nodes.iter().find_map(|n| {
            if let SceneNode::Polyline { points, .. } = n { Some(points.iter().map(|p| p.1).collect()) } else { None }
        }).expect("expected a Polyline node");
        assert_eq!(polyline_ys.len(), 3, "clip=False must keep all 3 vertices, including the out-of-domain one");
        let max_y = polyline_ys.iter().cloned().fold(f64::MIN, f64::max);
        assert!((max_y - 200.0).abs() < 1e-9,
            "the out-of-domain vertex must extrapolate to y=200 (outside the [0,100] panel range), got {max_y}");
        assert!(max_y > panel.plot_area.h,
            "clip=False must render a vertex OUTSIDE the panel rect (max_y={max_y} > panel height={})",
            panel.plot_area.h);
    }

    /// Byte-identity guard: the default (`clip` unset) drops the same
    /// out-of-domain vertex exactly as before this fix — 2 surviving points,
    /// neither exceeding the panel's pixel range.
    #[test]
    fn line_clip_unset_drops_out_of_domain_vertex_byte_identical() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 5.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let scales = narrow_domain_scales();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let polyline_ys: Vec<f64> = result.nodes.iter().find_map(|n| {
            if let SceneNode::Polyline { points, .. } = n { Some(points.iter().map(|p| p.1).collect()) } else { None }
        }).expect("expected a Polyline node");
        assert_eq!(polyline_ys.len(), 2, "default (unset clip) must drop the out-of-domain vertex, keeping 2 points");
        let max_y = polyline_ys.iter().cloned().fold(f64::MIN, f64::max);
        assert!(max_y <= panel.plot_area.h,
            "default clip must never exceed the panel's pixel range; got max_y={max_y}");
    }

    /// `clip=Some(true)` (the explicit spelling of the default) must be
    /// bit-identical to leaving `clip` unset — extrapolation is gated
    /// strictly on `Some(false)`, never on "clip is not false".
    #[test]
    fn line_clip_true_is_byte_identical_to_unset() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 5.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let scales = narrow_domain_scales();

        let unset_style = resolve_mark_style(None, &theme, &Mark::Line).unwrap();
        let overrides = MarkKwargsSpec { clip: Some(true), ..Default::default() };
        let true_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();

        let ctx_unset = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &unset_style };
        let ctx_true = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &true_style };

        let ys_unset: Vec<f64> = super::build(&ctx_unset).unwrap().nodes.iter().find_map(|n| {
            if let SceneNode::Polyline { points, .. } = n { Some(points.iter().map(|p| p.1).collect()) } else { None }
        }).unwrap();
        let ys_true: Vec<f64> = super::build(&ctx_true).unwrap().nodes.iter().find_map(|n| {
            if let SceneNode::Polyline { points, .. } = n { Some(points.iter().map(|p| p.1).collect()) } else { None }
        }).unwrap();
        assert_eq!(ys_unset, ys_true, "clip=Some(true) must be byte-identical to clip unset");
    }

    // ── Width: `size` tapers the line, `mark_line(size=)` is a constant ──
    //
    // Expected widths are computed here from the resolved size scale and the
    // dot-diameter rule written out (`2·sqrt(area/π)`), independently of
    // `dot_diameter`, and the series' end widths are also pinned against the
    // theme's default area band.

    /// x/y over four rows plus the size column `v` and a stroke-width column
    /// `w`.
    fn width_batch(v: Vec<Option<f64>>) -> arrow::record_batch::RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 3.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(v)),
            Arc::new(Float64Array::from(vec![2.0, 2.0, 2.0, 2.0])),
        ]).unwrap()
    }

    fn sized_spec(size_field: bool, stroke_width_field: bool) -> ChartSpec {
        let mut spec = line_spec();
        if size_field {
            spec.encoding.size = Some(EncodingSpec { field: "v".into(), ..Default::default() });
        }
        if stroke_width_field {
            spec.encoding.stroke_width = Some(EncodingSpec { field: "w".into(), ..Default::default() });
        }
        spec
    }

    /// Build the line and return its nodes, plus each row's expected tapered
    /// width (`None` where the size value is null).
    fn build_width_case(
        spec: &ChartSpec,
        batch: &arrow::record_batch::RecordBatch,
        overrides: &MarkKwargsSpec,
    ) -> Result<(Vec<SceneNode>, Vec<Option<f64>>), RenderError> {
        build_width_case_themed(spec, batch, overrides, ThemeInputs::default())
    }

    fn build_width_case_themed(
        spec: &ChartSpec,
        batch: &arrow::record_batch::RecordBatch,
        overrides: &MarkKwargsSpec,
        theme: ThemeInputs,
    ) -> Result<(Vec<SceneNode>, Vec<Option<f64>>), RenderError> {
        let (scales, _) = resolve_scales(spec, batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let expected: Vec<Option<f64>> = match &scales.size {
            Some(scale) => col_as_f64(batch, "v").unwrap().iter()
                .map(|v| v.and_then(|v| scale.inner.to_pixel_f64(v))
                    .map(|area| 2.0 * (area / std::f64::consts::PI).sqrt()))
                .collect(),
            None => vec![None; batch.num_rows()],
        };
        build_line(spec, batch, overrides, &theme).map(|r| (r.nodes, expected))
    }

    /// Build the line over a 100×100 panel.
    fn build_line(
        spec: &ChartSpec,
        batch: &arrow::record_batch::RecordBatch,
        overrides: &MarkKwargsSpec,
        theme: &ThemeInputs,
    ) -> Result<crate::render::draw::MarkBuildResult, RenderError> {
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(spec, batch, (0.0, 100.0), (0.0, 100.0), theme).unwrap();
        let mark_style = resolve_mark_style(Some(overrides), theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec, panel: &panel, theme, scales: &scales, batch, mark_style: &mark_style };
        super::build(&ctx)
    }

    fn assert_widths(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "widths {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "widths {got:?} vs {want:?}");
        }
    }

    const ASCENDING: [Option<f64>; 4] = [Some(1.0), Some(2.0), Some(3.0), Some(4.0)];

    #[test]
    fn sized_line_polyline_carries_the_dot_diameter_at_every_vertex() {
        let (nodes, expected) =
            build_width_case(&sized_spec(true, false), &width_batch(ASCENDING.to_vec()), &MarkKwargsSpec::default()).unwrap();
        let [SceneNode::Polyline { points, style, widths: Some(widths) }] = nodes.as_slice() else {
            panic!("expected one tapered polyline, got {nodes:?}");
        };
        let expected: Vec<f64> = expected.into_iter().map(Option::unwrap).collect();
        assert_eq!(widths.len(), points.len());
        assert_widths(widths, &expected);
        assert!(widths.windows(2).all(|p| p[0] < p[1]), "widths must rise with the size column: {widths:?}");
        // The size column spans the theme's whole area band [4, 36] px².
        let theme = ThemeInputs::default();
        let pi = std::f64::consts::PI;
        assert!((widths[0] - 2.0 * (theme.sizes.point_size_min / pi).sqrt()).abs() < 1e-9);
        assert!((widths[3] - 2.0 * (theme.sizes.point_size_max / pi).sqrt()).abs() < 1e-9);
        // The constant fallback width stays on the style for readers that
        // ignore `widths`.
        assert_eq!(style.width, theme.sizes.line_stroke_width);
    }

    #[test]
    fn a_null_size_row_takes_the_series_constant_width() {
        let v = vec![Some(1.0), None, Some(3.0), Some(4.0)];
        let (nodes, expected) =
            build_width_case(&sized_spec(true, false), &width_batch(v), &MarkKwargsSpec::default()).unwrap();
        let [SceneNode::Polyline { widths: Some(widths), .. }] = nodes.as_slice() else {
            panic!("expected one tapered polyline, got {nodes:?}");
        };
        let constant = ThemeInputs::default().sizes.line_stroke_width;
        assert_widths(widths, &[expected[0].unwrap(), constant, expected[2].unwrap(), expected[3].unwrap()]);
    }

    /// Each command of a bent line takes the width of the row whose run it
    /// ends in: a curve endpoint its row's width, a step's tread and riser the
    /// width of the row at whose level they run.
    #[test]
    fn sized_bent_lines_carry_one_width_per_command() {
        let cases: [(&str, &[usize]); 6] = [
            ("monotone", &[0, 1, 2, 3]),
            ("cardinal", &[0, 1, 2, 3]),
            ("basis", &[0, 0, 1, 2, 3, 3]),
            ("step", &[0, 0, 1, 1, 1, 2, 2, 2, 3, 3]),
            ("step-before", &[0, 1, 1, 2, 2, 3, 3]),
            ("step-after", &[0, 0, 1, 1, 2, 2, 3]),
        ];
        // The cases cover every bent method the vocabulary publishes.
        let mut covered: Vec<&str> = cases.iter().map(|(m, _)| *m).collect();
        covered.sort_unstable();
        let mut bent: Vec<String> = crate::render::marks::curve::interpolate_accepted_values()
            .into_iter()
            .filter(|m| m != "linear")
            .collect();
        bent.sort_unstable();
        assert_eq!(covered, bent);

        for (method, rows) in cases {
            let overrides = MarkKwargsSpec { interpolate: Some(method.into()), ..Default::default() };
            let (nodes, expected) =
                build_width_case(&sized_spec(true, false), &width_batch(ASCENDING.to_vec()), &overrides).unwrap();
            let [SceneNode::Path { commands, closed: false, widths: Some(widths), .. }] = nodes.as_slice() else {
                panic!("{method}: expected one open tapered path, got {nodes:?}");
            };
            assert_eq!(widths.len(), commands.len(), "{method}: one width per command");
            let want: Vec<f64> = rows.iter().map(|&r| expected[r].unwrap()).collect();
            assert_widths(widths, &want);
        }
    }

    #[test]
    fn an_unsized_line_carries_no_widths_under_every_interpolate() {
        for method in crate::render::marks::curve::interpolate_accepted_values() {
            let overrides = MarkKwargsSpec { interpolate: Some(method.clone()), ..Default::default() };
            let (nodes, _) =
                build_width_case(&sized_spec(false, false), &width_batch(ASCENDING.to_vec()), &overrides).unwrap();
            assert_eq!(nodes.len(), 1, "{method}");
            assert!(
                matches!(&nodes[0], SceneNode::Polyline { widths: None, .. } | SceneNode::Path { widths: None, .. }),
                "{method}: {nodes:?}"
            );
        }
    }

    #[test]
    fn constant_size_kwarg_is_a_constant_dot_diameter_width() {
        let want = 2.0 * (50.0 / std::f64::consts::PI).sqrt();
        for method in ["linear", "monotone"] {
            let overrides = MarkKwargsSpec { size: Some(50.0), interpolate: Some(method.into()), ..Default::default() };
            let (nodes, _) =
                build_width_case(&sized_spec(false, false), &width_batch(ASCENDING.to_vec()), &overrides).unwrap();
            match nodes.as_slice() {
                [SceneNode::Polyline { style, widths: None, .. }] => assert!((style.width - want).abs() < 1e-12),
                [SceneNode::Path { style, widths: None, .. }] => assert!((style.stroke_width - want).abs() < 1e-12),
                other => panic!("{method}: expected one constant-width line, got {other:?}"),
            }
        }
    }

    /// An encoded `size` outranks the constant `size=`; the constant is the
    /// width a null row falls back to.
    #[test]
    fn encoded_size_outranks_the_constant_size_kwarg() {
        let v = vec![Some(1.0), None, Some(3.0), Some(4.0)];
        let overrides = MarkKwargsSpec { size: Some(50.0), ..Default::default() };
        let (nodes, expected) = build_width_case(&sized_spec(true, false), &width_batch(v), &overrides).unwrap();
        let [SceneNode::Polyline { style, widths: Some(widths), .. }] = nodes.as_slice() else {
            panic!("expected one tapered polyline, got {nodes:?}");
        };
        let constant = 2.0 * (50.0 / std::f64::consts::PI).sqrt();
        assert_widths(widths, &[expected[0].unwrap(), constant, expected[2].unwrap(), expected[3].unwrap()]);
        assert!((style.width - constant).abs() < 1e-12);
    }

    fn conflict_text(size: &str, stroke_width: &str) -> String {
        format!(
            "mark_line: size and stroke_width both set the line's width (size={size}, \
             stroke_width={stroke_width}); bind only one. size= sets a dot-diameter width that \
             can taper along the line; stroke_width= sets a width in pixels."
        )
    }

    /// `size` and `stroke_width` both bound refuses, for every pairing of a
    /// field and a constant kwarg; either one alone does not.
    #[test]
    fn size_and_stroke_width_both_bound_refuse_in_every_pairing() {
        // (size field?, size kwarg, stroke_width field?, stroke_width kwarg, size text, stroke_width text)
        let cells = [
            (true, None, true, None, "'v'", "'w'"),
            (true, None, false, Some(2.5), "'v'", "2.5"),
            (false, Some(10.0), true, None, "10", "'w'"),
            (false, Some(10.0), false, Some(2.5), "10", "2.5"),
        ];
        for (size_field, size, sw_field, stroke_width, s_text, w_text) in cells {
            let overrides = MarkKwargsSpec { size, stroke_width, ..Default::default() };
            let err = build_width_case(&sized_spec(size_field, sw_field), &width_batch(ASCENDING.to_vec()), &overrides)
                .expect_err("size and stroke_width together must refuse");
            assert!(matches!(err, RenderError::LineWidthConflict { .. }), "{err:?}");
            assert_eq!(err.to_string(), conflict_text(s_text, w_text));
        }
        // Vacuity control: each binding alone draws.
        for (size_field, size, sw_field, stroke_width) in
            [(true, None, false, None), (false, Some(10.0), false, None), (false, None, true, None), (false, None, false, Some(2.5))]
        {
            let overrides = MarkKwargsSpec { size, stroke_width, ..Default::default() };
            let (nodes, _) =
                build_width_case(&sized_spec(size_field, sw_field), &width_batch(ASCENDING.to_vec()), &overrides).unwrap();
            assert_eq!(nodes.len(), 1);
        }
    }

    fn test_layer(mark: Mark, encoding: Encoding, mark_style: Option<MarkKwargsSpec>) -> crate::spec::layer::Layer {
        crate::spec::layer::Layer {
            mark,
            encoding,
            mark_style,
            ..Default::default()
        }
    }

    /// `a + b + ...` as `Chart.__add__` writes it to the wire: the leading
    /// layer's encoding is the chart's, and a later layer's `size` promotes
    /// to the chart level when the chart binds none, so every layer that
    /// does not bind `size` inherits it. `stroke_width` is no auxiliary
    /// scale channel and never promotes.
    fn added(layers: Vec<crate::spec::layer::Layer>) -> ChartSpec {
        let mut spec = line_spec();
        spec.mark = layers[0].mark;
        spec.encoding = layers[0].encoding.clone();
        if spec.encoding.size.is_none() {
            spec.encoding.size = layers.iter().find_map(|l| l.encoding.size.clone());
        }
        spec.layers = Some(layers);
        spec
    }

    fn render_static(spec: &ChartSpec) -> Result<String, RenderError> {
        crate::render::render_svg(
            spec,
            &width_batch(ASCENDING.to_vec()),
            &ThemeInputs::default(),
            crate::layout::Viewport { width: 400.0, height: 300.0 },
            &crate::render::config::RenderConfig::default(),
            &crate::render::chart_config::ChartConfig::default(),
        )
        .map(|out| out.bytes)
    }

    fn render_interactive(spec: &ChartSpec) -> Result<ferrum_scene::SceneGraph, RenderError> {
        let (json, _, _) = crate::render::render_scene_json(
            spec,
            &width_batch(ASCENDING.to_vec()),
            &ThemeInputs::default(),
            crate::layout::Viewport { width: 400.0, height: 300.0 },
            &crate::render::config::RenderConfig::default(),
            &crate::render::chart_config::ChartConfig::default(),
        )?;
        Ok(serde_json::from_str(&json).expect("the scene wire JSON parses"))
    }

    /// The one line node layer `li` draws on the first panel: its style
    /// width and its per-vertex widths.
    fn line_of(scene: &ferrum_scene::SceneGraph, li: usize) -> (f64, Option<Vec<f64>>) {
        match scene.panels[0].marks[li].nodes.as_slice() {
            [SceneNode::Polyline { style, widths, .. }] => (style.width, widths.clone()),
            other => panic!("layer {li}: expected one polyline, got {other:?}"),
        }
    }

    /// The `stroke-width` of every `<polyline>` in a static SVG.
    fn svg_polyline_widths(svg: &str) -> Vec<f64> {
        svg.split("<polyline").skip(1)
            .map(|el| {
                let el = &el[..el.find('>').expect("a closed tag")];
                let at = el.find(" stroke-width=\"").expect("a stroke-width attribute") + " stroke-width=\"".len();
                el[at..at + el[at..].find('"').unwrap()].parse().unwrap()
            })
            .collect()
    }

    /// Every pairing of a field and a constant kwarg for a line's two width
    /// channels, with the text the refusal renders for each.
    #[allow(clippy::type_complexity)]
    fn width_conflict_cells() -> [(Encoding, Option<MarkKwargsSpec>, &'static str, &'static str); 4] {
        let kwargs = |size, stroke_width| Some(MarkKwargsSpec { size, stroke_width, ..Default::default() });
        [
            (sized_spec(true, true).encoding, None, "'v'", "'w'"),
            (sized_spec(true, false).encoding, kwargs(None, Some(2.5)), "'v'", "2.5"),
            (sized_spec(false, true).encoding, kwargs(Some(10.0), None), "10", "'w'"),
            (line_spec().encoding, kwargs(Some(10.0), Some(2.5)), "10", "2.5"),
        ]
    }

    /// A line that itself binds both `size` and `stroke_width` refuses in
    /// every field/constant pairing, in either layer position on a layered
    /// chart (naming its layer) and unprefixed on its own.
    #[test]
    fn a_line_binding_both_widths_refuses_flat_and_in_either_layer_position() {
        let point = test_layer(Mark::Point, line_spec().encoding, None);
        for (encoding, mark_style, size, stroke_width) in width_conflict_cells() {
            let text = conflict_text(size, stroke_width);
            let line = test_layer(Mark::Line, encoding.clone(), mark_style.clone());
            for (layers, li) in [(vec![line.clone(), point.clone()], 0), (vec![point.clone(), line.clone()], 1)] {
                let spec = added(layers);
                let want = format!("layer {li}: {text}");
                assert_eq!(render_static(&spec).err().map(|e| e.to_string()), Some(want.clone()));
                assert_eq!(render_interactive(&spec).err().map(|e| e.to_string()), Some(want));
            }
            let mut flat = line_spec();
            flat.encoding = encoding;
            flat.mark_style = mark_style;
            assert_eq!(render_static(&flat).err().map(|e| e.to_string()), Some(text.clone()));
            assert_eq!(render_interactive(&flat).err().map(|e| e.to_string()), Some(text));
        }
    }

    /// A line layer's own `stroke_width` (a kwarg or its own encoding)
    /// outranks a `size` it only inherits from a sibling: the line keeps its
    /// own pixel width, untapered, the sibling's dots keep the size, and
    /// nothing refuses, in either layer order, on both output kinds.
    #[test]
    fn a_line_layers_own_stroke_width_outranks_a_size_it_inherits() {
        let xy = line_spec().encoding;
        let sized_point = test_layer(Mark::Point, sized_spec(true, false).encoding, None);
        let plain_point = test_layer(Mark::Point, xy.clone(), None);
        let own_widths = [
            (test_layer(Mark::Line, xy.clone(), Some(MarkKwargsSpec { stroke_width: Some(3.0), ..Default::default() })), Some(3.0)),
            (test_layer(Mark::Line, sized_spec(false, true).encoding, None), None),
        ];
        for (line, kwarg_width) in own_widths {
            for line_first in [true, false] {
                let order = |partner: &crate::spec::layer::Layer| {
                    if line_first { vec![line.clone(), partner.clone()] } else { vec![partner.clone(), line.clone()] }
                };
                let (li, pi) = if line_first { (0, 1) } else { (1, 0) };
                let spec = added(order(&sized_point));
                assert!(spec.encoding.size.is_some(), "the size reaches the chart level, so the line inherits it");
                let scene = render_interactive(&spec).unwrap_or_else(|e| panic!("line_first={line_first}: {e}"));
                // The line draws exactly the style it draws beside an unsized
                // partner, with no taper.
                let control = render_interactive(&added(order(&plain_point))).unwrap();
                let (width, widths) = line_of(&scene, li);
                assert_eq!((width, widths.clone()), line_of(&control, li), "line_first={line_first}");
                assert_eq!(widths, None);
                if let Some(kwarg_width) = kwarg_width {
                    assert_eq!(width, kwarg_width);
                }
                // The dots stay sized by `v` (ascending 1..4).
                let radii: Vec<f64> = scene.panels[0].marks[pi].nodes.iter()
                    .map(|n| match n { SceneNode::Circle { r, .. } => *r, other => panic!("{other:?}") })
                    .collect();
                assert_eq!(radii.len(), 4);
                assert!(radii.windows(2).all(|w| w[0] < w[1]), "sized dots: {radii:?}");
                let svg = render_static(&spec).unwrap_or_else(|e| panic!("line_first={line_first}: {e}"));
                assert_eq!(svg_polyline_widths(&svg), vec![width]);
            }
        }
        // Vacuity control: without its own stroke_width the same line tapers
        // by the size it inherits, in either order.
        let bare = test_layer(Mark::Line, xy, None);
        for (layers, li) in [(vec![bare.clone(), sized_point.clone()], 0), (vec![sized_point, bare], 1)] {
            let (_, widths) = line_of(&render_interactive(&added(layers)).unwrap(), li);
            assert_eq!(widths.map(|w| w.len()), Some(4));
        }
    }

    /// A line layer's own constant width kwarg outranks an inherited field on
    /// the same channel: beside a sibling's `size='v'` a `size=5` line draws
    /// untapered at `dot_diameter(5)`, and beside a leading line's
    /// `stroke_width='w'` a `stroke_width=3` line draws 3 px, in either layer
    /// order, on both output kinds, while the sibling keeps its own binding.
    #[test]
    fn a_line_layers_own_width_kwarg_outranks_the_same_field_it_inherits() {
        let xy = line_spec().encoding;
        let kwargs = |size, stroke_width| Some(MarkKwargsSpec { size, stroke_width, ..Default::default() });
        // (sibling binding the field, unbound control sibling, own-kwarg line, its width)
        let cases = [
            (
                test_layer(Mark::Point, sized_spec(true, false).encoding, None),
                test_layer(Mark::Point, xy.clone(), None),
                test_layer(Mark::Line, xy.clone(), kwargs(Some(5.0), None)),
                dot_diameter(5.0),
            ),
            (
                test_layer(Mark::Line, sized_spec(false, true).encoding, None),
                test_layer(Mark::Line, xy.clone(), None),
                test_layer(Mark::Line, xy.clone(), kwargs(None, Some(3.0))),
                3.0,
            ),
        ];
        for (sibling, control_sibling, line, want) in cases {
            let sibling_is_line = sibling.mark == Mark::Line;
            for line_first in [true, false] {
                let order = |sibling: &crate::spec::layer::Layer| {
                    if line_first { vec![line.clone(), sibling.clone()] } else { vec![sibling.clone(), line.clone()] }
                };
                let (li, si) = if line_first { (0, 1) } else { (1, 0) };
                let spec = added(order(&sibling));
                let scene = render_interactive(&spec).unwrap_or_else(|e| panic!("line_first={line_first}: {e}"));
                let control = render_interactive(&added(order(&control_sibling))).unwrap();
                assert_eq!(line_of(&scene, li), (want, None), "line_first={line_first}");
                assert_eq!(line_of(&scene, li), line_of(&control, li), "line_first={line_first}");
                let svg = render_static(&spec).unwrap_or_else(|e| panic!("line_first={line_first}: {e}"));
                let svg_widths = svg_polyline_widths(&svg);
                let control_widths = svg_polyline_widths(&render_static(&added(order(&control_sibling))).unwrap());
                if sibling_is_line {
                    // The sibling keeps its own `stroke_width` field: its
                    // constant `w` column maps to the middle of the
                    // stroke-width range.
                    let sizes = ThemeInputs::default().sizes;
                    let mapped = (sizes.line_stroke_width_min + sizes.line_stroke_width_max) / 2.0;
                    assert_eq!(line_of(&scene, si).0, mapped, "line_first={line_first}");
                    assert_eq!(svg_widths[li], control_widths[li], "line_first={line_first}");
                    assert_eq!(svg_widths[si], mapped, "line_first={line_first}");
                } else {
                    // The sibling's dots stay sized by `v` (ascending 1..4).
                    let radii: Vec<f64> = scene.panels[0].marks[si].nodes.iter()
                        .map(|n| match n { SceneNode::Circle { r, .. } => *r, other => panic!("{other:?}") })
                        .collect();
                    assert_eq!(radii.len(), 4);
                    assert!(radii.windows(2).all(|w| w[0] < w[1]), "sized dots: {radii:?}");
                    assert_eq!(svg_widths, control_widths, "line_first={line_first}");
                }
            }
        }
    }

    /// The precedence extends to every way a width can be inherited: a line
    /// that owns `size` outranks a `stroke_width` field it inherits from the
    /// leading layer and tapers, and a line that owns neither but inherits
    /// both keeps the inherited `stroke_width` and draws untapered.
    #[test]
    fn an_inherited_width_never_conflicts_with_another_width() {
        let stroked = test_layer(Mark::Line, sized_spec(false, true).encoding, None);
        let sized = test_layer(Mark::Line, sized_spec(true, false).encoding, None);
        let sized_point = test_layer(Mark::Point, sized_spec(true, false).encoding, None);
        let bare = test_layer(Mark::Line, line_spec().encoding, None);

        let spec = added(vec![stroked.clone(), sized]);
        assert!(spec.encoding.stroke_width.is_some() && spec.encoding.size.is_some());
        let scene = render_interactive(&spec).unwrap();
        assert_eq!(line_of(&scene, 0).1, None, "the stroked line keeps its own pixel width");
        assert_eq!(line_of(&scene, 1).1.map(|w| w.len()), Some(4), "the sized line tapers");
        render_static(&spec).unwrap();

        let spec = added(vec![stroked, sized_point, bare]);
        let scene = render_interactive(&spec).unwrap();
        assert_eq!(line_of(&scene, 2), line_of(&scene, 0), "both inherited: the stroke_width wins");
        assert_eq!(line_of(&scene, 2).1, None);
        render_static(&spec).unwrap();
    }

    /// The size legend asks the width precedence the line draws by: a line
    /// whose own `stroke_width=3` outranks the `size` it inherits draws no
    /// size, so beside a sized rule, or alone under a chart-level size, there
    /// is no size legend; beside a sized point the point's circles. A line
    /// without its own width draws the inherited size, so the same rule
    /// pairing shows line swatches at the dot diameter of each entry.
    #[test]
    fn the_size_legend_follows_the_width_the_line_draws() {
        let xy = line_spec().encoding;
        let sized = |mut encoding: Encoding| {
            encoding.size = Some(EncodingSpec { field: "v".into(), ..Default::default() });
            encoding
        };
        let own_width = Some(MarkKwargsSpec { stroke_width: Some(3.0), ..Default::default() });
        let stroked = test_layer(Mark::Line, xy.clone(), own_width);
        let bare = test_layer(Mark::Line, xy.clone(), None);
        let rule_y = Encoding { y: xy.y.clone(), ..Default::default() };
        let sized_rule = test_layer(Mark::Rule, sized(rule_y), None);
        let sized_point = test_layer(Mark::Point, sized(xy.clone()), None);
        let swatches = |spec: &ChartSpec| {
            let scene = render_interactive(spec).unwrap_or_else(|e| panic!("{e}"));
            assert!(render_static(spec).is_ok());
            crate::render::legend_swatches(&scene)
        };

        let reference = swatches(&added(vec![sized_point.clone()]));
        assert!(!reference.is_empty() && reference.iter().all(|(k, _)| *k == "circle"), "{reference:?}");

        assert_eq!(swatches(&added(vec![stroked.clone(), sized_rule.clone()])), Vec::new());
        let chart_level_only = ChartSpec { encoding: sized(xy.clone()), ..added(vec![stroked.clone()]) };
        assert_eq!(swatches(&chart_level_only), Vec::new());
        assert_eq!(swatches(&added(vec![stroked, sized_point])), reference);

        // Vacuity control: the bare line draws the inherited size.
        let lines = swatches(&added(vec![bare.clone(), sized_rule]));
        let want: Vec<(&str, f64)> = reference.iter().map(|(_, r)| ("line", 2.0 * r)).collect();
        assert_eq!(lines.len(), want.len(), "{lines:?}");
        for ((kind, width), (_, w)) in lines.iter().zip(&want) {
            assert_eq!(*kind, "line", "{lines:?}");
            assert!((width - w).abs() < 1e-9, "{lines:?} vs {want:?}");
        }
        assert_eq!(swatches(&ChartSpec { encoding: sized(xy), ..added(vec![bare]) }), lines);
    }

    // ── Every drawn width is finite and positive, or the line refuses ──

    /// A size scale whose range starts at 0 maps the domain minimum to area
    /// 0, a dot with no diameter: that vertex has no width to draw, and the
    /// line refuses instead of substituting its constant width.
    #[test]
    fn a_size_mapping_to_zero_area_refuses_naming_the_value() {
        let mut spec = sized_spec(true, false);
        let scale = |range: &str| serde_json::from_str(&format!(r#"{{"type":"linear","domain":[1,4],"range":{range}}}"#)).unwrap();
        spec.encoding.size.as_mut().unwrap().scale = Some(scale("[0,36]"));
        let err = build_width_case(&spec, &width_batch(ASCENDING.to_vec()), &MarkKwargsSpec::default())
            .expect_err("a zero-area vertex must refuse");
        assert!(matches!(err, RenderError::LineWidthNotPositive { .. }), "{err:?}");
        assert_eq!(
            err.to_string(),
            "mark_line: size 'v' value 1 maps to area 0 on the size scale, which gives no positive \
             line width. Every width along a line must be positive; give the size scale a range above 0."
        );
        // Vacuity control: the same scale starting above 0 draws every vertex.
        spec.encoding.size.as_mut().unwrap().scale = Some(scale("[4,36]"));
        let (nodes, _) = build_width_case(&spec, &width_batch(ASCENDING.to_vec()), &MarkKwargsSpec::default()).unwrap();
        let [SceneNode::Polyline { widths: Some(widths), .. }] = nodes.as_slice() else {
            panic!("expected one tapered polyline, got {nodes:?}");
        };
        assert!(widths.iter().all(|w| w.is_finite() && *w > 0.0), "{widths:?}");
    }

    /// A row with no usable size takes the series' constant width; when that
    /// width is the theme's `line_stroke_width=0`, the vertex has no width and
    /// the line refuses. Without such a row the zero theme width never reaches
    /// `widths` and the line draws.
    #[test]
    fn a_null_size_row_on_a_zero_default_width_refuses() {
        let mut theme = ThemeInputs::default();
        theme.sizes.line_stroke_width = 0.0;
        let spec = sized_spec(true, false);
        let with_null = vec![Some(1.0), None, Some(3.0), Some(4.0)];
        let err = build_width_case_themed(&spec, &width_batch(with_null), &MarkKwargsSpec::default(), theme.clone())
            .expect_err("a zero fallback width must refuse");
        assert!(matches!(err, RenderError::LineWidthNotPositive { .. }), "{err:?}");
        assert_eq!(
            err.to_string(),
            "mark_line: size 'v' has a row with no usable size (null, non-finite or off the scale), \
             which takes the line's default width 0. Every width along a line must be positive; set \
             a positive Theme(line_stroke_width=)."
        );
        // Vacuity control: every row sized, the zero theme width stays on the
        // style and every vertex width is positive.
        let (nodes, _) =
            build_width_case_themed(&spec, &width_batch(ASCENDING.to_vec()), &MarkKwargsSpec::default(), theme).unwrap();
        let [SceneNode::Polyline { style, widths: Some(widths), .. }] = nodes.as_slice() else {
            panic!("expected one tapered polyline, got {nodes:?}");
        };
        assert_eq!(style.width, 0.0);
        assert!(widths.iter().all(|w| w.is_finite() && *w > 0.0), "{widths:?}");
    }

    /// A hand-built wire spec reaches render without Python's constructor
    /// check, so render runs the one size rule on every layer: a constant
    /// `size=` must be finite and `> 0` on a mark that draws one, and is
    /// refused on a mark that never does, naming that mark's own sizing kwarg
    /// where it has one. The refusal names the entry the layer records, and
    /// its layer on a layered chart.
    #[test]
    fn wire_door_refuses_a_constant_size_the_mark_cannot_draw_and_names_the_layer() {
        let sized = |mark, value| {
            let mut spec = line_spec();
            spec.mark = mark;
            spec.mark_style = Some(MarkKwargsSpec { size: Some(value), ..Default::default() });
            spec
        };
        for mark in [Mark::Line, Mark::Point] {
            for (value, text) in [(-4.0, "-4"), (0.0, "0"), (f64::NAN, "NaN"), (f64::INFINITY, "inf")] {
                let spec = sized(mark, value);
                let want = format!("mark_{mark}(size=...) must be a positive number; received {text}");
                let err = render_static(&spec).err().unwrap_or_else(|| panic!("{mark} size={value} must refuse"));
                assert!(matches!(err.root(), RenderError::MarkSizeOutOfRange { mark: m, .. } if *m == mark), "{err:?}");
                assert_eq!(err.to_string(), want);
                assert_eq!(render_interactive(&spec).err().map(|e| e.to_string()), Some(want));
            }
        }
        for (mark, tail) in [(Mark::Bar, ""), (Mark::Area, ""), (Mark::Rule, " Set stroke_width= instead.")] {
            let spec = sized(mark, 5.0);
            let want = format!("mark_{mark}(size=...) has no effect on mark_{mark}; received 5.{tail}");
            let err = render_static(&spec).err().unwrap_or_else(|| panic!("{mark} size=5 must refuse"));
            assert!(matches!(err.root(), RenderError::MarkSizeNotDrawn { mark: m, .. } if *m == mark), "{err:?}");
            assert_eq!(err.to_string(), want);
            assert_eq!(render_interactive(&spec).err().map(|e| e.to_string()), Some(want));
        }
        // The chart's entry names the public call in place of the mark.
        let stamped = ChartSpec { entry: Some("relplot".into()), ..sized(Mark::Point, 0.0) };
        assert_eq!(
            render_static(&stamped).err().map(|e| e.to_string()),
            Some("relplot(size=...) must be a positive number; received 0".to_string())
        );
        // Vacuity controls: a positive constant draws on both marks that draw one.
        for mark in [Mark::Line, Mark::Point] {
            assert!(render_static(&sized(mark, 10.0)).is_ok(), "{mark:?}");
            assert!(render_interactive(&sized(mark, 10.0)).is_ok(), "{mark:?}");
        }

        // On a layered chart a refused constant names its layer.
        let xy = line_spec().encoding;
        let bad = |mark, value| test_layer(mark, xy.clone(), Some(MarkKwargsSpec { size: Some(value), ..Default::default() }));
        for (mark, value, text) in [
            (Mark::Line, 0.0, "mark_line(size=...) must be a positive number; received 0"),
            (Mark::Point, 0.0, "mark_point(size=...) must be a positive number; received 0"),
            (Mark::Area, 5.0, "mark_area(size=...) has no effect on mark_area; received 5."),
        ] {
            let partner = test_layer(Mark::Point, xy.clone(), None);
            for (layers, li) in [(vec![bad(mark, value), partner.clone()], 0), (vec![partner, bad(mark, value)], 1)] {
                let spec = added(layers);
                let want = format!("layer {li}: {text}");
                assert_eq!(render_static(&spec).err().map(|e| e.to_string()), Some(want.clone()));
                assert_eq!(render_interactive(&spec).err().map(|e| e.to_string()), Some(want));
            }
        }

        // Naming is a property of where the refusal arose, not of its kind: an
        // unparseable color on a layer reads as that layer's entry too. A
        // layer a desugaring call built is named by that call alone, its
        // internal index omitted.
        let mut bad_fill = test_layer(Mark::Point, xy.clone(), Some(MarkKwargsSpec { fill: Some("notacolor".into()), ..Default::default() }));
        bad_fill.entry = Some("relplot".into());
        let spec = added(vec![test_layer(Mark::Point, xy.clone(), None), bad_fill]);
        let err = render_static(&spec).err().expect("an unparseable fill must refuse");
        assert!(matches!(err, RenderError::InLayer { layer: None, .. }), "{err:?}");
        assert!(matches!(err.root(), RenderError::InvalidColor(_)), "{err:?}");
        assert!(err.to_string().starts_with("relplot: "), "{err}");

        // The size field is chart-level (the one size scale), inherited by
        // both layers; the line is layer 1.
        let mut layered = sized_spec(true, false);
        layered.encoding.size.as_mut().unwrap().scale =
            Some(serde_json::from_str(r#"{"type":"linear","domain":[1,4],"range":[0,36]}"#).unwrap());
        layered.layers = Some(vec![test_layer(Mark::Point, xy.clone(), None), test_layer(Mark::Line, xy, None)]);
        let err = render_static(&layered).err().expect("the line layer must refuse");
        assert!(err.to_string().starts_with("layer 1: mark_line: size 'v' value 1 maps to area 0"), "{err}");
    }

    // ── A size conditional: the base width here, each vertex's row recorded ──

    fn size_conditional(selected: f64, unselected: f64) -> ferrum_scene::ConditionalEncoding {
        ferrum_scene::ConditionalEncoding {
            selection_name: "s".into(),
            channel: ferrum_scene::ChannelName::Size,
            if_selected: ferrum_scene::EncodingValue::Size { value: selected },
            if_not: ferrum_scene::EncodingValue::Size { value: unselected },
        }
    }

    fn with_size_conditional(mut spec: ChartSpec, selected: f64, unselected: f64) -> ChartSpec {
        spec.conditionals = vec![size_conditional(selected, unselected).into()];
        spec
    }

    /// `spec` with one `size` conditional declared by `layers` (`None`: the
    /// whole chart).
    fn with_scoped_size_conditional(mut spec: ChartSpec, selected: f64, unselected: f64, layers: Option<Vec<usize>>) -> ChartSpec {
        spec.conditionals = vec![crate::spec::chart::ChartConditional { encoding: size_conditional(selected, unselected), layers }];
        spec
    }

    /// The row numbers of a node's vertex rows, none of them a cut.
    fn row_numbers(rows: &[ferrum_scene::VertexRow]) -> Vec<usize> {
        rows.iter()
            .map(|row| match row {
                ferrum_scene::VertexRow::Row(r) => *r,
                cut => panic!("an unbroken line has no cut vertex, got {cut:?}"),
            })
            .collect()
    }

    fn all_row_numbers(rows: Option<Vec<Vec<ferrum_scene::VertexRow>>>) -> Option<Vec<Vec<usize>>> {
        rows.map(|rows| rows.iter().map(|r| row_numbers(r)).collect())
    }

    #[test]
    fn a_size_conditional_records_each_vertex_row_and_draws_the_base_width() {
        let batch = width_batch(ASCENDING.to_vec());
        let theme = ThemeInputs::default();
        let none = MarkKwargsSpec::default();
        let plain = build_line(&line_spec(), &batch, &none, &theme).unwrap();
        let conditional = build_line(&with_size_conditional(line_spec(), 100.0, 20.0), &batch, &none, &theme).unwrap();
        assert_eq!(conditional.nodes, plain.nodes, "the line draws exactly as an unconditional one");
        assert_eq!(all_row_numbers(conditional.vertex_rows), Some(vec![vec![0, 1, 2, 3]]));
        assert_eq!(plain.vertex_rows, None, "an unconditional line records no rows");
    }

    #[test]
    fn a_size_conditional_step_line_records_one_row_per_command() {
        let batch = width_batch(ASCENDING.to_vec());
        let step = MarkKwargsSpec { interpolate: Some("step".into()), ..Default::default() };
        let result =
            build_line(&with_size_conditional(line_spec(), 100.0, 20.0), &batch, &step, &ThemeInputs::default()).unwrap();
        let [SceneNode::Path { commands, widths: None, .. }] = result.nodes.as_slice() else {
            panic!("expected one untapered path, got {:?}", result.nodes);
        };
        let rows = &all_row_numbers(result.vertex_rows).expect("rows recorded")[0];
        assert_eq!(rows.len(), commands.len(), "one row per command, aligned like `widths`");
        assert_eq!((rows[0], rows[rows.len() - 1]), (0, 3));
        assert!(rows.windows(2).all(|w| w[0] <= w[1]), "a step walks the rows in order: {rows:?}");
    }

    #[test]
    fn a_size_conditional_records_each_series_own_rows() {
        use arrow::array::Int64Array;
        // Three color groups interleaved row by row: group g holds rows g, g+3, g+6, g+9.
        let groups: Vec<i64> = (0..12).map(|i| (i % 3) as i64).collect();
        let (batch, spec) = make_grouped_color_batch_and_spec(Arc::new(Int64Array::from(groups)), DataType::Int64);
        let result = build_line(
            &with_size_conditional(spec, 100.0, 20.0),
            &batch,
            &MarkKwargsSpec::default(),
            &ThemeInputs::default(),
        )
        .unwrap();
        let rows = all_row_numbers(result.vertex_rows).expect("rows recorded");
        let mut sorted = rows.clone();
        sorted.sort();
        assert_eq!(sorted, vec![vec![0, 3, 6, 9], vec![1, 4, 7, 10], vec![2, 5, 8, 11]]);
        let firsts: Vec<usize> = rows.iter().map(|r| r[0]).collect();
        assert_eq!(result.data_indices, Some(firsts), "each node's rows start at its own representative row");
    }

    #[test]
    fn a_size_conditional_branch_with_no_positive_width_refuses() {
        let batch = width_batch(ASCENDING.to_vec());
        for (selected, unselected, bad) in [(0.0, 20.0, "0"), (100.0, -4.0, "-4")] {
            let spec = with_size_conditional(line_spec(), selected, unselected);
            let err = build_line(&spec, &batch, &MarkKwargsSpec::default(), &ThemeInputs::default())
                .err()
                .expect("a non-positive branch must refuse");
            assert_eq!(
                err.to_string(),
                format!(
                    "mark_line: size=fm.when(...) on selection 's' has a branch of size {bad}, which gives \
                     no positive line width. Every width along a line must be positive; give every branch \
                     a size above 0."
                )
            );
        }
    }

    #[test]
    fn every_size_conditional_on_a_line_has_its_branches_checked() {
        // The runtime folds every size conditional a line carries, so a
        // non-positive branch on the second one refuses as on the first.
        let batch = width_batch(ASCENDING.to_vec());
        let mut later = size_conditional(100.0, 0.0);
        later.selection_name = "t".into();
        let mut spec = line_spec();
        spec.conditionals = vec![size_conditional(100.0, 20.0).into(), later.into()];
        let err = build_line(&spec, &batch, &MarkKwargsSpec::default(), &ThemeInputs::default())
            .err()
            .expect("the second conditional's size-0 branch must refuse");
        assert_eq!(
            err.to_string(),
            "mark_line: size=fm.when(...) on selection 't' has a branch of size 0, which gives no positive \
             line width. Every width along a line must be positive; give every branch a size above 0."
        );
    }

    // ── Scope: a size conditional reaches only the line whose layer declared it ──

    /// A layered point (layer 0) + line (layer 1) over the one shared batch.
    fn point_then_line() -> ChartSpec {
        let xy = line_spec().encoding;
        added(vec![test_layer(Mark::Point, xy.clone(), None), test_layer(Mark::Line, xy, None)])
    }

    /// The line batch (layer 1) of a `point_then_line` scene.
    fn line_batch(scene: &ferrum_scene::SceneGraph) -> &ferrum_scene::MarkBatch {
        let batch = &scene.panels[0].marks[1];
        assert_eq!(batch.kind, ferrum_scene::MarkBatchKind::Line);
        batch
    }

    #[test]
    fn a_sibling_layers_size_conditional_neither_restyles_nor_refuses_a_line() {
        // The point layer's own conditional, with a size-0 branch a point
        // accepts and a line could not draw: the plain line renders untouched.
        let point_scoped = with_scoped_size_conditional(point_then_line(), 100.0, 0.0, Some(vec![0]));
        assert!(render_static(&point_scoped).is_ok(), "{:?}", render_static(&point_scoped).err());
        let scene = render_interactive(&point_scoped).unwrap();
        assert_eq!(line_batch(&scene).vertex_rows, None, "the overlay line carries no vertex rows");
        let plain = render_interactive(&point_then_line()).unwrap();
        assert_eq!(line_batch(&scene).nodes, line_batch(&plain).nodes, "the line draws as with no conditional");
        assert_eq!(scene.interaction.conditionals, vec![size_conditional(100.0, 0.0)], "points keep it chart-wide");

        // Controls: the line's own layer, or the whole chart, declaring it
        // records the rows and checks the branches.
        for layers in [Some(vec![1]), Some(vec![0, 1]), None] {
            let spec = with_scoped_size_conditional(point_then_line(), 100.0, 20.0, layers.clone());
            let scene = render_interactive(&spec).unwrap();
            assert_eq!(all_row_numbers(line_batch(&scene).vertex_rows.clone()), Some(vec![vec![0, 1, 2, 3]]), "{layers:?}");
            let refused = with_scoped_size_conditional(point_then_line(), 100.0, 0.0, layers.clone());
            let want = "layer 1: mark_line: size=fm.when(...) on selection 's' has a branch of size 0, which gives no \
                        positive line width. Every width along a line must be positive; give every branch a size above 0.";
            assert_eq!(render_static(&refused).err().map(|e| e.to_string()).as_deref(), Some(want), "{layers:?}");
        }
    }

    /// Every batch lists exactly the conditionals its layer declared, so the
    /// line's own conditional is out of the point layer's list and the
    /// runtime never resizes the dots by it, while a chart-wide one reaches
    /// both layers.
    #[test]
    fn every_batch_lists_the_conditionals_its_layer_declared() {
        for (layers, point, line) in [
            (Some(vec![1]), vec![], vec![0]),
            (Some(vec![0]), vec![0], vec![]),
            (Some(vec![0, 1]), vec![0], vec![0]),
            (None, vec![0], vec![0]),
        ] {
            let scene = render_interactive(&with_scoped_size_conditional(point_then_line(), 100.0, 20.0, layers.clone())).unwrap();
            assert_eq!(scene.panels[0].marks[0].conditionals, point, "point batch, {layers:?}");
            assert_eq!(line_batch(&scene).conditionals, line, "line batch, {layers:?}");
        }
        // Vacuity control: a chart without conditionals lists none.
        let plain = render_interactive(&point_then_line()).unwrap();
        assert!(plain.panels[0].marks.iter().all(|b| b.conditionals.is_empty()));
    }

    /// A host chart's conditional reaches every layer attached to it, through
    /// `Chart.layer(Layer(...))` (an inheriting line layer) and through
    /// `chart + SecondaryY(...)` (an independent-y line layer): Python stamps
    /// it with every layer, `[0, 1]`, on both routes, and both batches list
    /// it. Control: one declared only on the attached layer, `[1]`, stays on
    /// the line batch alone. The two specs mirror the wire `to_json()` emits
    /// for each route, with the base chart a point layer on `x`/`y`.
    #[test]
    fn a_host_charts_conditional_reaches_every_attached_layer_on_both_host_routes() {
        let color = |r| ferrum_scene::EncodingValue::Color { value: ferrum_scene::Color { r, g: 0, b: 0, a: 255 } };
        let declared = |spec: ChartSpec, layers: Vec<usize>| ChartSpec {
            conditionals: vec![crate::spec::chart::ChartConditional {
                encoding: ferrum_scene::ConditionalEncoding {
                    selection_name: "s".into(),
                    channel: ferrum_scene::ChannelName::Color,
                    if_selected: color(255),
                    if_not: color(136),
                },
                layers: Some(layers),
            }],
            ..spec
        };
        let xy = line_spec().encoding;
        let attached = added(vec![test_layer(Mark::Point, xy.clone(), None), test_layer(Mark::Line, Encoding::default(), None)]);
        let secondary_y = Encoding { y: Some(EncodingSpec { field: "w".into(), type_: None, ..Default::default() }), ..xy.clone() };
        let secondary = added(vec![
            test_layer(Mark::Point, xy, None),
            crate::spec::layer::Layer { independent_y: true, ..test_layer(Mark::Line, secondary_y, None) },
        ]);
        for (route, base) in [("layer", attached), ("SecondaryY", secondary)] {
            let host = render_interactive(&declared(base.clone(), vec![0, 1])).unwrap();
            assert_eq!(host.panels[0].marks[0].conditionals, vec![0], "{route}: the host's own point layer");
            assert_eq!(line_batch(&host).conditionals, vec![0], "{route}: the attached line layer");
            let own = render_interactive(&declared(base, vec![1])).unwrap();
            assert_eq!(own.panels[0].marks[0].conditionals, Vec::<usize>::new(), "{route}: control, the host layer");
            assert_eq!(line_batch(&own).conditionals, vec![0], "{route}: control, the attached layer");
        }
    }

    /// A later color conditional beside a line's size conditional is listed on
    /// the batch like every conditional its layer declared, but is not
    /// branch-checked as a width, and the line renders.
    #[test]
    fn a_line_answers_to_a_later_color_conditional_without_width_checking_it() {
        let mut spec = with_size_conditional(line_spec(), 100.0, 20.0);
        let color = |r| ferrum_scene::EncodingValue::Color { value: ferrum_scene::Color { r, g: 0, b: 0, a: 255 } };
        spec.conditionals.push(
            ferrum_scene::ConditionalEncoding {
                selection_name: "s".into(),
                channel: ferrum_scene::ChannelName::Color,
                if_selected: color(255),
                if_not: color(0),
            }
            .into(),
        );
        assert!(render_static(&spec).is_ok(), "{:?}", render_static(&spec).err());
        let scene = render_interactive(&spec).unwrap();
        let batch = &scene.panels[0].marks[0];
        assert_eq!(batch.conditionals, vec![0, 1]);
        assert_eq!(all_row_numbers(batch.vertex_rows.clone()), Some(vec![vec![0, 1, 2, 3]]));
        assert_eq!(scene.interaction.conditionals.len(), 2);
    }

    /// A layer that records the public entry which produced it (a composite
    /// such as `mark_smooth` desugaring into a fit line) names that entry,
    /// not `mark_line`, in every per-layer line refusal, without its internal
    /// layer index; a layer that records none keeps `mark_line` and the index
    /// that locates it, and an unlayered chart is unchanged.
    #[test]
    fn a_line_refusal_names_the_public_entry_its_layer_records() {
        let xy = line_spec().encoding;
        let layered = |entry: Option<&str>, line_style: MarkKwargsSpec| {
            let mut line = test_layer(Mark::Line, xy.clone(), Some(line_style));
            line.entry = entry.map(Into::into);
            added(vec![test_layer(Mark::Point, xy.clone(), None), line])
        };
        let conflict = MarkKwargsSpec { size: Some(40.0), stroke_width: Some(3.0), ..Default::default() };
        let conflict_text = |voice: &str| {
            format!(
                "{voice}: size and stroke_width both set the line's width (size=40, stroke_width=3); bind \
                 only one. size= sets a dot-diameter width that can taper along the line; stroke_width= sets a \
                 width in pixels."
            )
        };
        let bad_size = MarkKwargsSpec { size: Some(-4.0), ..Default::default() };
        let size_text = |voice: &str| format!("{voice}(size=...) must be a positive number; received -4");
        let branch_text = |voice: &str| {
            format!(
                "{voice}: size=fm.when(...) on selection 's' has a branch of size 0, which gives no \
                 positive line width. Every width along a line must be positive; give every branch a size above 0."
            )
        };
        for (entry, named) in [(Some("mark_smooth"), "mark_smooth"), (None, "layer 1: mark_line")] {
            let refusal = |spec: &ChartSpec| render_static(spec).err().map(|e| e.to_string());
            assert_eq!(refusal(&layered(entry, conflict.clone())), Some(conflict_text(named)), "{entry:?}");
            assert_eq!(refusal(&layered(entry, bad_size.clone())), Some(size_text(named)), "{entry:?}");
            let branch = with_scoped_size_conditional(layered(entry, MarkKwargsSpec::default()), 100.0, 0.0, None);
            assert_eq!(refusal(&branch), Some(branch_text(named)), "{entry:?}");
            assert_eq!(render_interactive(&branch).err().map(|e| e.to_string()), Some(branch_text(named)));
        }
    }

    /// A chart that records the public entry which produced it (a composite
    /// such as `mark_smooth` lowering to one unlayered line) names that entry
    /// in every line refusal, without a layer prefix; a chart that records
    /// none keeps `mark_line`. On a layered chart the chart's entry names
    /// every layer that records none, and a layer's own entry outranks it;
    /// either names a layer the call built, so no index precedes it.
    #[test]
    fn a_line_refusal_names_the_public_entry_its_chart_records() {
        let conflict = MarkKwargsSpec { size: Some(40.0), stroke_width: Some(3.0), ..Default::default() };
        let conflict_text = |entry: &str| {
            format!(
                "{entry}: size and stroke_width both set the line's width (size=40, stroke_width=3); bind only \
                 one. size= sets a dot-diameter width that can taper along the line; stroke_width= sets a width in \
                 pixels."
            )
        };
        let bad_size = MarkKwargsSpec { size: Some(-4.0), ..Default::default() };
        let size_text = |entry: &str| format!("{entry}(size=...) must be a positive number; received -4");
        let branch_text = |entry: &str| {
            format!(
                "{entry}: size=fm.when(...) on selection 's' has a branch of size 0, which gives no positive line \
                 width. Every width along a line must be positive; give every branch a size above 0."
            )
        };
        let flat = |entry: Option<&str>, style: Option<MarkKwargsSpec>| ChartSpec {
            mark_style: style,
            entry: entry.map(Into::into),
            ..line_spec()
        };
        let refusal = |spec: &ChartSpec| render_static(spec).err().map(|e| e.to_string());
        for (entry, named) in [(Some("mark_smooth"), "mark_smooth"), (None, "mark_line")] {
            assert_eq!(refusal(&flat(entry, Some(conflict.clone()))), Some(conflict_text(named)), "{entry:?}");
            assert_eq!(refusal(&flat(entry, Some(bad_size.clone()))), Some(size_text(named)), "{entry:?}");
            let branch = with_size_conditional(flat(entry, None), 100.0, 0.0);
            assert_eq!(refusal(&branch), Some(branch_text(named)), "{entry:?}");
            assert_eq!(render_interactive(&branch).err().map(|e| e.to_string()), Some(branch_text(named)));
        }

        let xy = line_spec().encoding;
        let layered = |layer_entry: Option<&str>| {
            let mut line = test_layer(Mark::Line, xy.clone(), Some(conflict.clone()));
            line.entry = layer_entry.map(Into::into);
            ChartSpec { entry: Some("lmplot".into()), ..added(vec![test_layer(Mark::Point, xy.clone(), None), line]) }
        };
        for (layer_entry, named) in [(Some("mark_smooth"), "mark_smooth"), (None, "lmplot")] {
            let want = conflict_text(named);
            assert_eq!(refusal(&layered(layer_entry)), Some(want), "{layer_entry:?}");
        }
    }

    #[test]
    fn a_line_binding_a_conditional_size_and_stroke_width_refuses() {
        let text = |w: &str| {
            format!(
                "mark_line: size and stroke_width both set the line's width (size=fm.when(...), stroke_width={w}); \
                 bind only one. size= sets a dot-diameter width that can taper along the line; stroke_width= sets \
                 a width in pixels."
            )
        };
        let kwarg = || MarkKwargsSpec { stroke_width: Some(3.0), ..Default::default() };
        let mut by_kwarg = with_size_conditional(line_spec(), 100.0, 20.0);
        by_kwarg.mark_style = Some(kwarg());
        let by_field = with_size_conditional(sized_spec(false, true), 100.0, 20.0);
        for (spec, w) in [(by_kwarg, "3"), (by_field, "'w'")] {
            assert_eq!(render_static(&spec).err().map(|e| e.to_string()), Some(text(w)));
            assert_eq!(render_interactive(&spec).err().map(|e| e.to_string()), Some(text(w)));
        }

        // Layered: the line's own conditional refuses naming its layer; the
        // point's leaves the 3 px line alone.
        let xy = line_spec().encoding;
        let layered = added(vec![test_layer(Mark::Point, xy.clone(), None), test_layer(Mark::Line, xy, Some(kwarg()))]);
        let own = with_scoped_size_conditional(layered.clone(), 100.0, 20.0, Some(vec![1]));
        assert_eq!(render_static(&own).err().map(|e| e.to_string()), Some(format!("layer 1: {}", text("3"))));
        let sibling = with_scoped_size_conditional(layered, 100.0, 20.0, Some(vec![0]));
        let scene = render_interactive(&sibling).unwrap();
        assert_eq!(line_of(&scene, 1), (3.0, None));
        assert_eq!(line_batch(&scene).vertex_rows, None);
    }

    #[test]
    fn a_conditional_naming_a_layer_the_chart_lacks_refuses() {
        let layered = with_scoped_size_conditional(point_then_line(), 100.0, 20.0, Some(vec![0, 2]));
        let want = "conditionals[0].layers names layer 2, but the chart has 2 layer(s); name only the layers that \
                    declared the conditional.";
        assert_eq!(render_static(&layered).err().map(|e| e.to_string()).as_deref(), Some(want));
        assert_eq!(render_interactive(&layered).err().map(|e| e.to_string()).as_deref(), Some(want));
        let unlayered = with_scoped_size_conditional(line_spec(), 100.0, 20.0, Some(vec![1]));
        assert_eq!(
            render_static(&unlayered).err().map(|e| e.to_string()).as_deref(),
            Some("conditionals[0].layers names layer 1, but the chart has 1 layer(s); name only the layers that \
                  declared the conditional.")
        );
        // Control: layer 0 is the unlayered chart's one layer.
        assert!(render_static(&with_scoped_size_conditional(line_spec(), 100.0, 20.0, Some(vec![0]))).is_ok());
    }

    /// A point selection holds the clicked mark's `data_indices` entry, and a
    /// line's vertex rows must index the same rows for membership to mean the
    /// same data. Every layer reads the one merged batch (`chart + chart`
    /// diagonal-concats a second frame onto it), so a shared frame gives the
    /// line exactly the point's rows, and a second frame gives it rows no
    /// point owns.
    #[test]
    fn a_lines_vertex_rows_index_the_batch_the_selection_indexes() {
        let spec = with_scoped_size_conditional(point_then_line(), 100.0, 20.0, Some(vec![1]));
        let scene = render_interactive(&spec).unwrap();
        let point = &scene.panels[0].marks[0];
        assert_eq!(point.data_indices, Some(vec![0, 1, 2, 3]));
        assert_eq!(all_row_numbers(line_batch(&scene).vertex_rows.clone()), Some(vec![vec![0, 1, 2, 3]]));

        // Diagonal concat: the point's frame fills rows 0..4, the line's 4..8.
        let half = |lo: bool| -> Arc<dyn arrow::array::Array> {
            Arc::new(Float64Array::from((0..8).map(|i| ((i < 4) == lo).then_some(i as f64)).collect::<Vec<_>>()))
        };
        let schema = Schema::new(["x", "y", "x2", "y2"].map(|n| Field::new(n, DataType::Float64, true)).to_vec());
        let batch = arrow::record_batch::RecordBatch::try_new(Arc::new(schema), vec![half(true), half(true), half(false), half(false)]).unwrap();
        let mut line_xy = line_spec().encoding;
        line_xy.x = Some(EncodingSpec { field: "x2".into(), ..Default::default() });
        line_xy.y = Some(EncodingSpec { field: "y2".into(), ..Default::default() });
        let mut spec = added(vec![test_layer(Mark::Point, line_spec().encoding, None), test_layer(Mark::Line, line_xy, None)]);
        spec = with_scoped_size_conditional(spec, 100.0, 20.0, Some(vec![1]));
        let (json, _, _) = crate::render::render_scene_json(
            &spec,
            &batch,
            &ThemeInputs::default(),
            crate::layout::Viewport { width: 400.0, height: 300.0 },
            &crate::render::config::RenderConfig::default(),
            &crate::render::chart_config::ChartConfig::default(),
        )
        .unwrap();
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(scene.panels[0].marks[0].data_indices, Some(vec![0, 1, 2, 3]));
        assert_eq!(all_row_numbers(line_batch(&scene).vertex_rows.clone()), Some(vec![vec![4, 5, 6, 7]]));
    }

    // ── declared vertex join order (`sort_vertices`) ─────────────────────────

    /// Two color series arrive in unsorted, interleaved row order.
    /// `sort_vertices: Some(true)` joins each series' own vertices ascending
    /// by x, so every polyline's x-pixels come out non-decreasing, though the
    /// batch itself never does.
    #[test]
    fn declared_line_over_unsorted_rows_joins_each_color_series_ascending() {
        use crate::spec::encoding::DataType as SDT;
        let mut spec = line_spec();
        spec.encoding.color = Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Nominal), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        // group A rows (x=3,1,2) and group B rows (x=1,5,3) interleaved, each
        // in unsorted batch order.
        let xs: Vec<f64> = vec![3.0, 1.0, 1.0, 5.0, 2.0, 3.0];
        let ys: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let gs: Vec<&str> = vec!["A", "B", "A", "B", "A", "B"];
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(gs)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let overrides = MarkKwargsSpec { sort_vertices: Some(true), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let polylines: Vec<&Vec<(f64, f64)>> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Polyline { points, .. } = n { Some(points) } else { None }
        }).collect();
        assert_eq!(polylines.len(), 2);
        for points in polylines {
            let xs: Vec<f64> = points.iter().map(|(x, _)| *x).collect();
            let mut sorted = xs.clone();
            sorted.sort_by(f64::total_cmp);
            assert_eq!(xs, sorted, "each series' x-pixels must come out ascending: {xs:?}");
        }
        // The representative row (each node's `data_indices` entry, which
        // drives tooltip/href metadata and the group-first opacity/stroke
        // sample) is the FIRST row in the declared order, not the first row
        // in batch order: A's rows 0,2,4 (x 3,1,2) draw first from row 2;
        // B's rows 1,3,5 (x 1,5,3) draw first from row 1. Batch order would
        // give [0, 1].
        assert_eq!(result.data_indices, Some(vec![2, 1]));
    }

    /// A size conditional records each kept vertex's source row
    /// (`vertex_rows`); under a declared sort those rows follow the same
    /// ascending-x order the points do, not the batch's row order.
    #[test]
    fn a_declared_sorts_vertex_rows_follow_the_same_order_as_its_points() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        // Row i's x is 3, 1, 2, 0 — ascending x visits rows 3, 1, 2, 0.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0, 0.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        ]).unwrap();
        let overrides = MarkKwargsSpec { sort_vertices: Some(true), ..Default::default() };
        let result = build_line(&with_size_conditional(line_spec(), 100.0, 20.0), &batch, &overrides, &ThemeInputs::default()).unwrap();
        let rows = all_row_numbers(result.vertex_rows).expect("rows recorded");
        assert_eq!(rows, vec![vec![3, 1, 2, 0]], "vertex_rows must follow the declared ascending-x order, not batch order [0, 1, 2, 3]");
    }

    /// `sort_vertices` unset (and explicitly `Some(false)`) both keep the
    /// batch's own row order, byte-identically to each other and to every
    /// line rendered before the field existed.
    #[test]
    fn undeclared_line_keeps_batch_order_byte_identically() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let points_with = |style: Option<&MarkKwargsSpec>| {
            let mark_style = resolve_mark_style(style, &theme, &Mark::Line).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            let result = super::build(&ctx).unwrap();
            let SceneNode::Polyline { points, .. } = &result.nodes[0] else { panic!("expected a Polyline") };
            points.clone()
        };
        let unset = points_with(None);
        let explicit_false = points_with(Some(&MarkKwargsSpec { sort_vertices: Some(false), ..Default::default() }));
        assert_eq!(explicit_false, unset, "unset and Some(false) must be byte-identical");
        // Batch order (x = 3, 1, 2) is not ascending; neither spelling sorts it.
        let xs: Vec<f64> = unset.iter().map(|(x, _)| *x).collect();
        let mut sorted = xs.clone();
        sorted.sort_by(f64::total_cmp);
        assert_ne!(xs, sorted, "batch order must stay in row order, not ascending x");
    }

    /// The declared join order reads the advancing field's DATA value, not
    /// its pixel, so a reversed x scale still steps through ascending data
    /// even though its pixels descend.
    #[test]
    fn a_reversed_x_scale_joins_in_ascending_data_order() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::linear::LinearScale;
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![5.0, 1.0, 9.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 0.0])),
        ]).unwrap();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 10.0], vec![300.0, 0.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 10.0], vec![0.0, 100.0], false, false)),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None,
            stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None,
            y_slots: Default::default(), raster_cells: None,
        };
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 300.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let overrides = MarkKwargsSpec { sort_vertices: Some(true), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Line).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let SceneNode::Polyline { points, .. } = &result.nodes[0] else { panic!("expected a Polyline") };
        // Ascending data order 1, 5, 9 maps through the reversed scale
        // (domain [0,10] -> range [300,0]) to descending pixels 270, 150, 30.
        assert_eq!(points, &vec![(270.0, 0.0), (150.0, 0.0), (30.0, 0.0)]);
    }

    /// An `Ordinal` advancing axis has no numeric data value to order by, so
    /// a declared line keeps its batch row order exactly as an undeclared
    /// one does.
    #[test]
    fn an_ordinal_advancing_axis_keeps_row_order_even_when_declared() {
        // A NUMERIC column typed Ordinal: `col_as_f64` reads it fine (unlike
        // a text column, which the Ordinal guard's absence would mask —
        // `vertex_sort_values` would already be `None` for want of a numeric
        // read, not because of the guard), so this is the one fixture that
        // actually exercises the guard rather than vacuously agreeing with
        // it.
        use crate::spec::encoding::DataType as EncDataType;
        use arrow::array::Int64Array;
        let mut spec = line_spec();
        spec.encoding.x = Some(EncodingSpec {
            field: "year".into(),
            type_: Some(EncDataType::Ordinal),
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("year", DataType::Int64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        // Batch order 2, 0, 3, 1 — unsorted, so a sort would visibly reorder.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![2i64, 0, 3, 1])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        assert!(matches!(scales.x, ScaleKind::Ordinal(_)), "the fixture must resolve an Ordinal x scale, not Linear");
        let points_with = |style: Option<&MarkKwargsSpec>| {
            let mark_style = resolve_mark_style(style, &theme, &Mark::Line).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            let result = super::build(&ctx).unwrap();
            let SceneNode::Polyline { points, .. } = &result.nodes[0] else { panic!("expected a Polyline") };
            points.clone()
        };
        let unset = points_with(None);
        let declared = points_with(Some(&MarkKwargsSpec { sort_vertices: Some(true), ..Default::default() }));
        assert_eq!(declared, unset, "an Ordinal advancing axis must ignore sort_vertices and keep batch order");
    }

    /// `sort_vertices` is a `mark_line`-only declaration; any other mark
    /// carrying it refuses through the existing style-table gate, naming the
    /// mark and the field.
    #[test]
    fn a_non_line_mark_carrying_sort_vertices_refuses() {
        let theme = ThemeInputs::default();
        let overrides = MarkKwargsSpec { sort_vertices: Some(true), ..Default::default() };
        let err = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap_err();
        assert_eq!(err, RenderError::MarkStyleNotDrawn { mark: Mark::Point, field: "sort_vertices" });
    }
}
