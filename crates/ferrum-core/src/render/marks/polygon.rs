//! mark_polygon: closed-region drawer. Groups vertices into one polygon per
//! `mark_kwargs.detail` value (e.g. `hex_id`, `violin_id`, `level_id`). When
//! `detail` is unset, all rows form a single polygon.
//!
//! Color resolution:
//! - `color` encoding + Categorical scale → per-group string lookup (mirror area.rs).
//! - `color` encoding + Float64 column → quantitative; sample named cmap (default Viridis)
//!   at `(value - vmin) / (vmax - vmin)`. One polygon per detail group; group color
//!   taken from the first row in the group.
//! - No `color` encoding → single fill from `mark_style.paint.fill`.
//!
//! Used by violin (Task 25), contour-fill (Task 30), and hex (Task 28) composite marks.

use std::collections::HashMap;
use std::hash::Hash;

use arrow::array::{Array, Float64Array, Int64Array, UInt32Array};

use crate::render::color::{with_opacity, ContinuousScheme, NamedContinuous};
use crate::render::draw::{
    col_as_f64, col_as_ordinal_category_str, col_as_str, color_field, x_field, y_field, DrawCtx,
};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::curve::{SeriesNoun, ShortRuns};
use crate::render::marks::opacity::resolve_scaled_opacity;
use crate::render::scale_resolve::ScaleKind;

/// Whether this polygon's `color` channel is quantitative: its column is a
/// float column, which the polygon paints through a colormap (`cmap`) rather
/// than a categorical palette. The style table's
/// [`StyleNeed::QuantitativeColor`](crate::spec::mark::StyleNeed) reads this
/// same fact.
pub(crate) fn color_is_quantitative(ctx: &DrawCtx) -> bool {
    color_field(ctx, ctx.spec)
        .and_then(|f| ctx.batch.column_by_name(f))
        .is_some_and(|a| a.as_any().downcast_ref::<Float64Array>().is_some())
}

/// Strict when `field` reads as `f64`, else Category — `build`'s own
/// `xpx`/`ypx` reader order below (`col_as_f64` tried first, `col_as_str`
/// the fallback), not a static "by scale kind" rule: a polygon's x/y
/// placement depends on the COLUMN's own dtype, since `build` never
/// consults `ctx.scales.{x,y}`'s kind to choose a reader.
fn polygon_read(ctx: &DrawCtx, field: &str) -> crate::render::draw::PlacementRead {
    use crate::render::draw::PlacementRead;
    if col_as_f64(ctx.batch, field).is_ok() { PlacementRead::Strict } else { PlacementRead::Category }
}

/// One first-appearance-order group per distinct non-null `key_at(i)`, over
/// `0..n`: the single ordering rule [`detail_groups`] applies to every
/// `detail` dtype, the same order every series mark's own `detail` groups
/// by ([`build_color_detail_groups`](super::channels::build_color_detail_groups)'s
/// detail-only branch). `O(n)` through a value-to-group-index map, never a
/// per-row linear search — a contour or hex batch runs to tens of thousands
/// of rows.
fn group_first_appearance<K: Eq + Hash>(n: usize, key_at: impl Fn(usize) -> Option<K>) -> Vec<Vec<usize>> {
    let mut index: HashMap<K, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let Some(key) = key_at(i) else { continue };
        match index.get(&key) {
            Some(&p) => groups[p].push(i),
            None => {
                index.insert(key, groups.len());
                groups.push(vec![i]);
            }
        }
    }
    groups
}

/// This mark's own ring partition: one group per `mark_style.group.detail`
/// value, in the order each value first appears — the same rule every
/// series mark's `detail` partition uses, for every dtype alike. Integer
/// and float columns keep typed keys (`i64`, a `Float64`'s `to_bits` as
/// `i64`) for a cheap hash; every other dtype [`col_as_ordinal_category_str`]
/// can read groups the same way, by its string key. A single group of
/// every row when `detail` is unset. Empty (no groups at all) when a bound
/// `detail` names a column this batch does not carry, or a dtype even the
/// category reader cannot place (nested or binary columns, as `line`'s
/// `detail` also leaves ungrouped) — `build` draws nothing for that same
/// reason, so there is nothing for a statistic layer's census to attribute
/// to a domain exclusion either. Null-detail rows are left out of every
/// ring.
/// Read by [`vertex_frame`] too, so the census's vertex-unit count and the
/// real render can never group a statistic layer's rows differently.
pub(crate) fn detail_groups(ctx: &DrawCtx) -> Vec<Vec<usize>> {
    let n_rows = ctx.batch.num_rows();
    let Some(field) = ctx.mark_style.group.detail.as_deref() else {
        return vec![(0..n_rows).collect()];
    };
    let Some(arr) = ctx.batch.column_by_name(field) else { return Vec::new() };
    if let Some(u) = arr.as_any().downcast_ref::<UInt32Array>() {
        return group_first_appearance(u.len(), |i| (!u.is_null(i)).then(|| u.value(i) as i64));
    }
    if let Some(i64a) = arr.as_any().downcast_ref::<Int64Array>() {
        return group_first_appearance(i64a.len(), |i| (!i64a.is_null(i)).then(|| i64a.value(i)));
    }
    if let Some(f) = arr.as_any().downcast_ref::<Float64Array>() {
        return group_first_appearance(f.len(), |i| (!f.is_null(i)).then(|| f.value(i).to_bits() as i64));
    }
    // Every other dtype a bare `fm.Layer(mark="polygon")` might bind
    // `detail` to (text, a narrower int, a bool): one ring per distinct
    // value, first-appearance order, the same rule as the typed branches
    // above. Left unhandled only by a dtype this category reader cannot
    // place (nested or binary), which keeps one ring — a fallback, not a
    // silent misdraw, since every production polygon id is one of the
    // three typed branches.
    if let Ok(categories) = col_as_ordinal_category_str(ctx.batch, field) {
        return group_first_appearance(categories.len(), |i| categories[i].clone());
    }
    vec![(0..n_rows).collect()]
}

/// The caller's own column(s) naming a dropped ring's entry in
/// [`crate::render::RenderWarning::SeriesDropped`]. A ring a CALL built
/// (`entry` is not this mark's own `mark_polygon` primitive —
/// [`built_by_call`](crate::render::prepare::built_by_call) —
/// `violin`/`hex`/`contour`'s own layer) is named by every drawn field
/// among a categorical x, a categorical y and color, in that order: these
/// marks always bind `detail` to an internal grouping id
/// (`violin_id`/`hex_id`/`level_id`) the caller never wrote, so naming them
/// by it would expose an implementation detail instead of the categories
/// and color a reader actually sees. A ring the CALLER placed (a bare
/// `fm.Layer(mark="polygon")`) is named by `detail` alone, the one field
/// its own rings partition by — color may be shared across rings there and
/// would not distinguish the dropped one.
pub(crate) fn naming_columns(ctx: &DrawCtx) -> Vec<String> {
    let spec = ctx.spec;
    let entry = spec.entry.clone().unwrap_or_else(|| format!("mark_{}", spec.mark.as_str()));
    if crate::render::prepare::built_by_call(&entry, spec.mark) {
        let mut cols = Vec::new();
        if let Some(f) = x_field(ctx, spec) {
            if matches!(ctx.scales.x, ScaleKind::Ordinal(_)) {
                cols.push(f.to_string());
            }
        }
        if let Some(f) = y_field(ctx, spec) {
            if matches!(ctx.scales.y, ScaleKind::Ordinal(_)) {
                cols.push(f.to_string());
            }
        }
        if let Some(f) = color_field(ctx, spec) {
            cols.push(f.to_string());
        }
        return cols;
    }
    ctx.mark_style.group.detail.clone().into_iter().collect()
}

/// `x` and `y` per vertex row, each placed Strict or Category by
/// [`polygon_read`].
pub(crate) fn placements<'a>(ctx: &'a DrawCtx<'a>) -> Vec<crate::render::draw::Placement<'a>> {
    use crate::render::draw::Placement;
    use crate::spec::layer::PositionalChannel;
    let spec = ctx.spec;
    let mut out = Vec::new();
    if let Some(f) = x_field(ctx, spec) { out.push(Placement::new(PositionalChannel::X, f, polygon_read(ctx, f))); }
    if let Some(f) = y_field(ctx, spec) { out.push(Placement::new(PositionalChannel::Y, f, polygon_read(ctx, f))); }
    out
}

/// Each row's vertex in pixels (numeric or ordinal on each axis, so
/// `CoordFlip` works when the categorical axis is swapped onto y), before
/// position offsets, `None` for a row `build` cannot place: a null value or
/// one its scale does not map. `None` overall when `x` or `y` is unbound or
/// reads as neither numbers nor strings, where `build` draws nothing.
pub(crate) fn vertex_pixels(ctx: &DrawCtx) -> Option<Vec<Option<(f64, f64)>>> {
    let spec = ctx.spec;
    let (xf, yf) = (x_field(ctx, spec)?, y_field(ctx, spec)?);
    let axis_pixels = |scale: &crate::render::scale_resolve::ScaleKind, field: &str| -> Option<Vec<Option<f64>>> {
        if let Ok(v) = col_as_f64(ctx.batch, field) {
            Some(v.into_iter().map(|opt| opt.and_then(|x| scale.to_pixel_f64(x))).collect())
        } else if let Ok(v) = col_as_str(ctx.batch, field) {
            Some(v.into_iter().map(|opt| opt.as_deref().and_then(|s| scale.to_pixel_str(s))).collect())
        } else {
            None
        }
    };
    let xpx = axis_pixels(&ctx.scales.x, xf)?;
    let ypx = axis_pixels(&ctx.scales.y, yf)?;
    Some(xpx.into_iter().zip(ypx).map(|(x, y)| Some((x?, y?))).collect())
}

/// This mark's [`VertexFrame`](crate::render::draw::VertexFrame): its own
/// [`detail_groups`], [`vertex_pixels`]' placed rows and `build`'s minimum
/// of 3 vertices per ring.
pub(crate) fn vertex_frame(ctx: &DrawCtx) -> crate::render::draw::VertexFrame {
    crate::render::draw::VertexFrame::from_vertices(detail_groups(ctx), vertex_pixels(ctx), ctx.batch.num_rows(), 3)
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{
        resolve_group_fill_color, to_scene_fill_stroke, MarkBuildResult, MetadataColumns,
        PaintChannels,
    };
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Polygon);

    let spec = ctx.spec;
    let Some(vertices) = vertex_pixels(ctx) else { return empty() };

    // --- Group rows by detail column (or single group if unset) ---
    let groups = detail_groups(ctx);

    // --- Resolve color encoding mode ---
    let cf = color_field(ctx, spec);
    let color_arr = cf.and_then(|f| ctx.batch.column_by_name(f));
    let color_is_quantitative = color_is_quantitative(ctx);

    // NF-A3 (spec §4.4, sweep completed 2026-09-02): the category read is
    // `col_as_ordinal_category_str`, never `col_as_str`. `col_as_str` errors on
    // every non-`Utf8` dtype, and this branch is reached for every dtype that
    // is not `Float64` — so an `Int*`/`Bool`/`Float32` color column used to
    // yield `None` here and paint every polygon the constant mark fill while
    // the legend enumerated its categories. Identical on `Utf8`.
    let color_str_values: Option<Vec<Option<String>>> =
        if !color_is_quantitative {
            cf.and_then(|f| col_as_ordinal_category_str(ctx.batch, f).ok())
        } else {
            None
        };

    let scheme = if color_is_quantitative {
        let named = ctx
            .mark_style
            .group
            .cmap
            .as_deref()
            .and_then(NamedContinuous::from_name)
            .unwrap_or_else(|| {
                NamedContinuous::from_name(&ctx.theme.palette.sequential_scheme)
                    .unwrap_or(NamedContinuous::Viridis)
            });
        Some(ContinuousScheme::Named(named))
    } else {
        None
    };
    let (vmin, vmax) = if color_is_quantitative {
        if let Some(a) = color_arr.and_then(|a| a.as_any().downcast_ref::<Float64Array>()) {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for i in 0..a.len() {
                if !a.is_null(i) {
                    let v = a.value(i);
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            (lo, hi)
        } else {
            (0.0, 1.0)
        }
    } else {
        (0.0, 1.0)
    };
    let denom = (vmax - vmin).max(f64::EPSILON);

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Per-row opacity encoding (first row of each group used, mirroring color).
    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let meta = MetadataColumns::from_ctx(ctx);

    // One node per group, attached to the group's representative source row
    // (`group_indices[0]`, the first row of the group). The accumulator keeps
    // `data_indices` in node order so `build_metadata_for_indices` aligns each
    // polygon's tooltip/href/description to that representative row (bug #6).
    let mut acc = MarkNodes::with_capacity(groups.len());
    const RULE: &str = "a polygon joins three or more vertices";
    let mut short_runs = ShortRuns::new(RULE, SeriesNoun::Polygon, groups.len());

    // --- Emit one polygon per group ---
    for (ordinal, group_indices) in groups.iter().enumerate() {
        let ring: Vec<(f64, f64)> = group_indices
            .iter()
            .filter_map(|&i| {
                let (cx, cy) = vertices.get(i).copied().flatten()?;
                let xo = x_offsets.get(i).copied().unwrap_or(0.0);
                let yo = y_offsets.get(i).copied().unwrap_or(0.0);
                Some((cx + xo, cy + yo))
            })
            .collect();
        if ring.len() < 3 {
            short_runs.note(ordinal, group_indices, ring.len());
            continue;
        }

        let first_row = group_indices[0];

        // Resolve fill for this group, carrying the constant fill slot's own
        // provenance so a `fill="none"` clear outranks whichever branch resolves
        // a color (`#110`).
        //
        // The quantitative branch samples polygon's OWN `cmap`/theme scheme
        // rather than `ctx.scales.color`, so unlike `area`'s it cannot fold onto
        // `resolve_fill_color`: doing so would change which colors a
        // scale-carrying polygon chart renders. That divergence is a separate
        // live defect (a `SequentialScale(scheme=…)` on the color channel is
        // silently dropped by the marks while the legend honors it) and is
        // reported as its own follow-up rather than closed here.
        let base = ctx.mark_style.paint.fill_paint();
        let fill = if color_is_quantitative {
            if let Some(a) = color_arr.and_then(|a| a.as_any().downcast_ref::<Float64Array>()) {
                if a.is_null(first_row) {
                    base
                } else {
                    let v = a.value(first_row);
                    let t = ((v - vmin) / denom).clamp(0.0, 1.0);
                    match scheme.as_ref() {
                        Some(s) => base.cleared_or(s.sample(t)),
                        None => base,
                    }
                }
            } else {
                base
            }
        } else {
            // Group-keyed: the representative row's category string, resolved the
            // way `ColorScale::lookup(&str)` resolved it when this branch was a
            // hand-rolled `scale.lookup(v).unwrap_or(paint.fill)` — including the
            // parse a numeric-input scale needs for a non-`Float64` column.
            resolve_group_fill_color(
                ctx.scales.color.as_ref(),
                color_str_values.as_ref().and_then(|v| v.get(first_row)).and_then(|o| o.as_deref()),
                base,
            )
        };

        // Resolve per-group opacity through the scale if present (sampled at the
        // group's representative first row).
        let group_opacity = resolve_scaled_opacity(
            &opacity_values,
            &ctx.scales.opacity,
            first_row,
            ctx.mark_style.paint.opacity,
        );

        let fill = fill.map(|c| with_opacity(c, group_opacity));

        let exterior: Vec<[f64; 2]> = ring.iter().map(|&(x, y)| [x, y]).collect();
        acc.push(
            SceneNode::Polygon {
                rings: vec![exterior],
                style: to_scene_fill_stroke(
                    fill,
                    ctx.mark_style.paint.stroke_paint(),
                    PaintChannels::new(ctx.mark_style.paint.stroke_width, group_opacity, None),
                ),
            },
            first_row,
        );
    }

    let (nodes, data_indices) = acc.finalize();
    let naming_fields = naming_columns(ctx);
    let (empty_reason, dropped_series) = short_runs.finish(nodes.len(), ctx.batch, &naming_fields);

    MarkBuildResult::populated(MarkBatchKind::Polygon, nodes, data_indices, &meta)
        .with_empty_reason(empty_reason)
        .with_dropped_series(dropped_series)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style, DrawCtx};
    use crate::render::scale_resolve::{resolve_scales, OpacityScale, ResolvedScales, ScaleKind};
    use crate::scale::linear::LinearScale;
    use crate::spec::chart::ChartSpec;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn polygon_spec(detail: Option<&str>, color: Option<&str>) -> ChartSpec {
        use crate::spec::mark_style::MarkKwargsSpec;
        let mut kwargs = MarkKwargsSpec::default();
        if let Some(d) = detail {
            kwargs.detail = Some(d.to_string());
        }
        kwargs.cmap = Some("viridis".to_string());
        ChartSpec {
            mark: Mark::Polygon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: color.map(|f| EncodingSpec { field: f.into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(kwargs),
            ..Default::default()
        }
    }

    fn rect_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    #[test]
    fn polygon_single_ring_three_vertices() {
        let spec = polygon_spec(None, None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(), 1, "expected 1 polygon");
        // Polygon must have at least 3 points.
        let has_points = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Polygon { rings, .. } = n { rings.first().map_or(false, |r| r.len() >= 3) } else { false }
        });
        assert!(has_points, "polygon must have at least 3 points");
    }

    #[test]
    fn polygon_multi_ring_via_separate_detail_groups() {
        let spec = polygon_spec(Some("group_id"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,    2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,    0.0, 0.0, 1.0])),
            Arc::new(UInt32Array::from(vec![0u32, 0, 0,        1, 1, 1])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(), 2, "expected 2 paths (one per group)");
    }

    #[test]
    fn polygon_three_groups_three_paths() {
        let spec = polygon_spec(Some("group_id"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 0.5,    2.0, 3.0, 2.5,    4.0, 5.0, 4.5,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,
            ])),
            Arc::new(UInt32Array::from(vec![
                0u32, 0, 0,       1, 1, 1,          2, 2, 2,
            ])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(), 3, "expected 3 paths");
    }

    #[test]
    fn polygon_quantitative_color_yields_distinct_fills() {
        // 3 groups, color column = Float64 with values 0.0, 0.5, 1.0 across groups.
        // Expect 3 distinct fill colors (sampled from viridis).
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;

        let spec = polygon_spec(Some("group_id"), Some("value"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 0.5,    2.0, 3.0, 2.5,    4.0, 5.0, 4.5,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,
            ])),
            Arc::new(UInt32Array::from(vec![
                0u32, 0, 0,       1, 1, 1,          2, 2, 2,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 0.0,    0.5, 0.5, 0.5,    1.0, 1.0, 1.0,
            ])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 5.0], vec![0.0, 100.0], false, false,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0], vec![100.0, 0.0], false, false,
            )),
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
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // Collect distinct fill colors from Polygon nodes.
        let mut fills: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &result.nodes {
            if let ferrum_scene::SceneNode::Polygon { style, .. } = node {
                if let Some(c) = &style.fill {
                    fills.insert(format!("{},{},{},{}", c.r, c.g, c.b, c.a));
                }
            }
        }
        assert_eq!(fills.len(), 3, "expected 3 distinct polygon fills, got {}: {:?}", fills.len(), fills);
    }

    // --- W18: opacity encoding must be applied per-group ---

    /// W18: When an opacity encoding is present, each polygon group must have
    /// a different alpha in its fill color. Previously opacity was always taken
    /// from mark_style.paint.opacity (a single value for all polygons).
    #[test]
    fn w18_polygon_opacity_encoding_applied_per_group() {
        use crate::render::scale_resolve::{OpacityScale, ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use crate::spec::encoding::{DataType as SDT, EncodingSpec};

        let mut spec = polygon_spec(Some("group_id"), None);
        // Add opacity encoding.
        spec.encoding.opacity = Some(EncodingSpec {
            field: "op".into(),
            type_: Some(SDT::Quantitative),
            ..Default::default()
        });

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
            Field::new("op", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,    2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,    0.0, 0.0, 1.0])),
            Arc::new(UInt32Array::from(vec![0u32, 0, 0,        1, 1, 1])),
            Arc::new(Float64Array::from(vec![0.2, 0.2, 0.2,    0.8, 0.8, 0.8])),
        ]).unwrap();

        let theme = crate::layout::ThemeInputs::default();
        let panel = rect_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 3.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![100.0, 0.0], false, false)),
            color: None,
            size: None,
            shape: None,
            opacity: Some(OpacityScale {
                inner: ScaleKind::Linear(LinearScale::new_internal(vec![0.2, 0.8], vec![0.2, 0.8], false, false)),
            }),
            fill_opacity: None,
            stroke_opacity: None,
            stroke_dash: None,
            stroke_width: None,
            x2: None,
            y2: None,
            y_slots: Default::default(),
            raster_cells: None,
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let polys: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let ferrum_scene::SceneNode::Polygon { style, .. } = n { Some(style.clone()) } else { None }
        }).collect();
        assert_eq!(polys.len(), 2, "expected 2 polygon nodes");

        // ferrum_scene::Color uses .a for alpha.
        let alpha0 = polys[0].fill.as_ref().map(|c| c.a).unwrap_or(0);
        let alpha1 = polys[1].fill.as_ref().map(|c| c.a).unwrap_or(0);
        assert_ne!(
            alpha0, alpha1,
            "per-row opacity encoding must produce different alphas on polygons; both were {alpha0}"
        );
    }

    // ── #6 metadata/node alignment (group marks) ─────────────────────────────

    /// Bug #6: a multi-group polygon attaches each polygon to its detail group's
    /// REPRESENTATIVE source row (`group_indices[0]`, the first row of the
    /// group), not a per-node position. 3 detail groups, each 3 rows → first
    /// rows at 0, 3, 6 — distinct from node positions 0,1,2. Old code
    /// (`build_metadata(ctx)`) returns full per-row vectors (length 9) while only
    /// 3 nodes are emitted, so the metadata length diverges and node j mis-maps.
    #[test]
    fn polygon_group_aligns_to_representative_row() {
        use arrow::array::{Int64Array, StringArray};

        let mut spec = polygon_spec(Some("group_id"), None);
        spec.encoding.tooltip = Some(EncodingSpec { field: "tip".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::Int64, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        // group_id sorted-key order (100,200,300) matches block order; each
        // group's first row is at 0, 3, 6. Per-row tip == group label.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 0.5,    2.0, 3.0, 2.5,    4.0, 5.0, 4.5,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,
            ])),
            Arc::new(Int64Array::from(vec![
                100i64, 100, 100,   200, 200, 200,   300, 300, 300,
            ])),
            Arc::new(StringArray::from(vec![
                "g100", "g100", "g100",  "g200", "g200", "g200",  "g300", "g300", "g300",
            ])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 5.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(),
            raster_cells: None,
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 3, "expected one polygon per detail group");
        let di = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(di.len(), result.nodes.len(), "nodes.len() must equal data_indices.len()");
        assert_eq!(di, &vec![0, 3, 6], "each polygon maps to its group's first row");

        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let vals: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(vals, vec!["g100", "g200", "g300"],
            "node j tooltip is its group's representative-row label; \
             old full-row code (len 9) mis-indexes / mis-sizes the metadata.");
    }

    /// Bug #6 href channel: href aligns to the representative row with no tooltip
    /// encoded (href-without-tooltip soundness hole).
    #[test]
    fn polygon_group_href_aligns_to_representative_row() {
        use arrow::array::{Int64Array, StringArray};

        let mut spec = polygon_spec(Some("group_id"), None);
        spec.encoding.href = Some(EncodingSpec { field: "url".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::Int64, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 0.5,    2.0, 3.0, 2.5,    4.0, 5.0, 4.5,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,
            ])),
            Arc::new(Int64Array::from(vec![
                100i64, 100, 100,   200, 200, 200,   300, 300, 300,
            ])),
            Arc::new(StringArray::from(vec![
                "u100", "u100", "u100",  "u200", "u200", "u200",  "u300", "u300", "u300",
            ])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 5.0], vec![0.0, 100.0], false, false)),
            y: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(),
            raster_cells: None,
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert!(result.tooltips.is_none(), "no tooltip encoding → tooltips None");
        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), result.nodes.len(), "href count must equal node count");
        assert_eq!(hrefs.iter().map(|h| h.as_deref()).collect::<Vec<_>>(),
            vec![Some("u100"), Some("u200"), Some("u300")],
            "each polygon's href is its group's representative-row url; \
             old full-row code mis-sizes/mis-indexes the href vector.");
    }

    /// No-skip backward-compat: a single ungrouped polygon (one node) keeps its
    /// representative at row 0 and tooltip output is unchanged.
    #[test]
    fn polygon_no_grouping_tooltip_unchanged() {
        use arrow::array::StringArray;
        let mut spec = polygon_spec(None, None);
        spec.encoding.tooltip = Some(EncodingSpec { field: "tip".into(), ..Default::default() });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("tip", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0])),
            Arc::new(StringArray::from(vec!["t0", "t1", "t2"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        assert_eq!(result.nodes.len(), 1, "single ungrouped polygon → one node");
        let di = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(di, &vec![0], "single group's representative is row 0");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 1, "tooltip count == node count");
        assert_eq!(tooltips[0].fields[0].value, "t0", "representative tooltip is row 0's");
    }

    #[test]
    fn polygon_int64_detail_groups_correctly() {
        // Regression: hex_id is Int64 but the grouping code previously only
        // handled UInt32 and Float64, collapsing all hex vertices into one
        // degenerate polygon. Verify three Int64 groups → three Polygon nodes.
        use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
        use crate::scale::linear::LinearScale;
        use arrow::array::Int64Array;

        let spec = polygon_spec(Some("hex_id"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("hex_id", DataType::Int64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 0.5,    2.0, 3.0, 2.5,    4.0, 5.0, 4.5,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,
            ])),
            Arc::new(Int64Array::from(vec![
                100i64, 100, 100,   200, 200, 200,   300, 300, 300,
            ])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 5.0], vec![0.0, 100.0], false, false,
            )),
            y: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0], vec![100.0, 0.0], false, false,
            )),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(),
            raster_cells: None,
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let n = result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count();
        assert_eq!(n, 3, "Int64 hex_id must produce 3 separate polygons, got {n}");
    }

    /// A text `detail` column (e.g. a bare `fm.Layer(mark="polygon")` ring
    /// naming its own zone directly, never an internal numeric id) splits
    /// into one ring per distinct value, in first-appearance order — the
    /// same grouping a numeric `detail` already gets, rather than every
    /// row merging into a single ring because `detail` reads as neither
    /// `UInt32`, `Int64` nor `Float64`.
    #[test]
    fn a_text_detail_column_splits_into_one_ring_per_value() {
        let spec = polygon_spec(Some("zone"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("zone", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])),
            Arc::new(arrow::array::StringArray::from(vec!["p", "p", "p", "q", "q", "q"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let n = result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count();
        assert_eq!(n, 2, "a text detail column must produce 2 separate rings, got {n}");
    }

    // ── NF-A3: the categorical branch reads every category dtype (§4.4) ──────

    /// An `Int64` nominal color column must fill each group with its category's
    /// color, not leave every group in the constant mark fill.
    ///
    /// The categorical branch here is reached for every dtype that is not
    /// `Float64` (`color_is_quantitative` is a `Float64Array` downcast), so
    /// `Int64` lands in it — and `col_as_str` returned `Err` for exactly that
    /// dtype. RED (verified in place): with `col_as_str`, `color_str_values` is
    /// `None`, the `(values, scale)` guard below the quantitative branch never
    /// opens, and all three polygons take `ctx.mark_style.paint.fill` while the
    /// legend enumerates the three categories. The `Utf8` twin carries the same
    /// three categories in the same row order and is asserted equal.
    #[test]
    fn polygon_fills_groups_from_an_int64_nominal_color() {
        use arrow::array::{Int64Array, StringArray};
        use crate::spec::encoding::DataType as SpecType;

        let group_fills = |color_field: Field, color_col: Arc<dyn arrow::array::Array>| -> Vec<String> {
            let mut spec = polygon_spec(Some("group_id"), Some("grp"));
            spec.encoding.color.as_mut().unwrap().type_ = Some(SpecType::Nominal);
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
                Field::new("group_id", DataType::UInt32, false),
                color_field,
            ]));
            let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,  2.0, 3.0, 2.5,  4.0, 5.0, 4.5])),
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0])),
                Arc::new(UInt32Array::from(vec![0u32, 0, 0,  1, 1, 1,  2, 2, 2])),
                color_col,
            ]).unwrap();
            let theme = ThemeInputs::default();
            let panel = rect_panel();
            let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
            assert!(scales.color.is_some(), "fixture must resolve a color scale");
            let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            super::build(&ctx).nodes.iter().filter_map(|n| match n {
                ferrum_scene::SceneNode::Polygon { style, .. } => {
                    style.fill.map(|c| format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b))
                }
                _ => None,
            }).collect()
        };

        let utf8 = group_fills(
            Field::new("grp", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["1", "1", "1",  "2", "2", "2",  "3", "3", "3"])),
        );
        assert_eq!(utf8.len(), 3, "Utf8 reference must produce 3 polygons");
        assert_eq!(
            utf8.iter().collect::<std::collections::BTreeSet<_>>().len(),
            3,
            "Utf8 reference must give three distinct group fills; got {utf8:?}"
        );

        let int64 = group_fills(
            Field::new("grp", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![1_i64, 1, 1,  2, 2, 2,  3, 3, 3])),
        );
        assert_eq!(
            int64, utf8,
            "an Int64 nominal color column must fill the groups exactly as its \
             Utf8 twin does; got {int64:?}. Three identical fills here is the \
             col_as_str drop."
        );
    }

    /// `build` over nullable `x`/`y` columns, one polygon per `group_id`.
    fn build_rings(xs: Vec<Option<f64>>, ys: Vec<Option<f64>>, groups: Vec<u32>) -> crate::render::draw::MarkBuildResult {
        let spec = polygon_spec(Some("group_id"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("group_id", DataType::UInt32, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(UInt32Array::from(groups)),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        super::build(&ctx)
    }

    /// A polygon mark whose every polygon has one or two drawable vertices
    /// draws nothing, and names why, counting its polygons.
    #[test]
    fn a_polygon_whose_every_ring_is_short_names_its_emptiness() {
        let one_vertex = build_rings(vec![Some(1.0)], vec![Some(2.0)], vec![0]);
        assert!(one_vertex.nodes.is_empty());
        assert_eq!(
            one_vertex.empty_reason.map(|reason| reason.0).as_deref(),
            Some("a polygon joins three or more vertices, and its one polygon has one or two"),
        );

        let three_short = build_rings(
            (0..5).map(|i| Some(f64::from(i))).collect(),
            (0..5).map(|i| Some(f64::from(i))).collect(),
            vec![0, 1, 1, 2, 2],
        );
        assert!(three_short.nodes.is_empty());
        assert_eq!(
            three_short.empty_reason.map(|reason| reason.0).as_deref(),
            Some("a polygon joins three or more vertices, and each of its 3 polygons has one or two"),
        );
    }

    /// A polygon mark that draws any polygon, or had no drawable vertex at
    /// all, carries no `empty_reason`: a short polygon beside a drawn one, a
    /// zero-row batch and an all-null batch. A short ring beside a drawn
    /// one is reported by name.
    #[test]
    fn a_polygon_that_draws_or_had_no_vertex_carries_no_empty_reason() {
        let partial = build_rings(
            vec![Some(0.0), Some(1.0), Some(0.5), Some(2.0), Some(3.0)],
            vec![Some(0.0), Some(0.0), Some(1.0), Some(0.0), Some(1.0)],
            vec![0, 0, 0, 1, 1],
        );
        assert_eq!(partial.nodes.len(), 1);
        assert!(partial.empty_reason.is_none(), "{:?}", partial.empty_reason);
        let dropped = partial.dropped_series.expect("a drawn ring beside a dropped one reports the dropped one");
        assert_eq!(dropped.rule, "a polygon joins three or more vertices");
        assert_eq!(dropped.names(), vec!["group_id=1 (two vertices)"]);

        let zero_rows = build_rings(vec![], vec![], vec![]);
        assert!(zero_rows.nodes.is_empty());
        assert!(zero_rows.empty_reason.is_none(), "{:?}", zero_rows.empty_reason);
        assert!(zero_rows.dropped_series.is_none());

        let all_null = build_rings(vec![None, None, None], vec![None, None, None], vec![0, 0, 0]);
        assert!(all_null.nodes.is_empty());
        assert!(all_null.empty_reason.is_none(), "{:?}", all_null.empty_reason);
        assert!(all_null.dropped_series.is_none());
    }

    /// A polygon with a short ring beside drawn ones reports the dropped
    /// one's name and why: a CALLER-PLACED ring (no call built this layer)
    /// is named by its `detail` field alone, even when a color is also
    /// bound — color does not partition a caller's own polygon rings,
    /// detail does, the field they wrote it by. Three identical vertices
    /// form a drawn (zero-area) ring rather than a dropped one.
    #[test]
    fn a_partial_short_series_drop_reports_the_dropped_series() {
        // A caller-placed ring (no call built this layer, so `entry` is
        // unset) with both `detail` and a color bound is still named by
        // `detail` alone — color never distinguishes a caller's own rings.
        let colored = polygon_spec(Some("group_id"), Some("cat"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 0.0, 1.0])),
            Arc::new(UInt32Array::from(vec![0u32, 0, 0, 1, 1])),
            Arc::new(arrow::array::StringArray::from(vec!["lo", "lo", "lo", "hi", "hi"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&colored, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(colored.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &colored, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.dropped_series.unwrap().names(), vec!["group_id=1 (two vertices)"]);

        // A bare ring with neither color nor a categorical x/y position
        // names the detail field itself — the field the caller DID write.
        let bare = build_rings(
            vec![Some(0.0), Some(1.0), Some(0.5), Some(2.0), Some(3.0)],
            vec![Some(0.0), Some(0.0), Some(1.0), Some(0.0), Some(1.0)],
            vec![0, 0, 0, 1, 1],
        );
        assert_eq!(bare.dropped_series.unwrap().names(), vec!["group_id=1 (two vertices)"]);

        // Three identical vertices form a drawn (zero-area) ring, not a
        // dropped one — the connectivity gate only drops a ring below ITS
        // mark's three-vertex minimum, never a degenerate but
        // fully-placed one.
        let tied = build_rings(
            vec![Some(1.0), Some(1.0), Some(1.0), Some(2.0), Some(3.0)],
            vec![Some(1.0), Some(1.0), Some(1.0), Some(0.0), Some(1.0)],
            vec![0, 0, 0, 1, 1],
        );
        assert_eq!(
            tied.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(),
            1,
            "the tied triple draws a zero-area ring"
        );
        assert_eq!(tied.dropped_series.unwrap().names(), vec!["group_id=1 (two vertices)"]);
    }

    /// A ring a CALL built (`entry` is not `mark_polygon` —
    /// `violin`/`hex`/`contour`'s own layer) is named by every drawn field
    /// among a categorical x, a categorical y and color, in that order —
    /// never by `detail`, always an internal grouping id on those marks, a
    /// null color value renders the way `describe_group_row` already
    /// renders a null. With neither a categorical position nor a color it
    /// falls back to its ordinal. A ring the CALLER placed (no call built
    /// this layer) keeps its own `detail` field even with a color bound
    /// too.
    #[test]
    fn a_call_built_ring_is_named_by_its_drawn_positions_and_color() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::discrete::DiscreteLayout;
        use crate::scale::ordinal::OrdinalScale;

        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let ordinal_x_scales = || ResolvedScales {
            x: ScaleKind::Ordinal(OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![0.0, 100.0], DiscreteLayout::UNPADDED)),
            y: ScaleKind::Linear(crate::scale::linear::LinearScale::new_internal(vec![0.0, 1.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(),
            raster_cells: None,
        };

        // Call-built, categorical x ("g") and color ("h"): group 0 ("b", 3
        // rows) draws, group 1 ("a", "u", 2 rows) is short.
        let call_built = |with_color: bool, color_rows: Vec<Option<&str>>| {
            let mut kwargs = crate::spec::mark_style::MarkKwargsSpec::default();
            kwargs.detail = Some("group_id".to_string());
            let spec = ChartSpec {
                mark: Mark::Polygon,
                entry: Some("mark_violin".to_string()),
                encoding: Encoding {
                    x: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                    y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                    color: with_color.then(|| EncodingSpec { field: "h".into(), type_: None, ..Default::default() }),
                    ..Default::default()
                },
                mark_style: Some(kwargs),
                ..Default::default()
            };
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("y", DataType::Float64, false),
                Field::new("group_id", DataType::UInt32, false),
                Field::new("h", DataType::Utf8, true),
            ]));
            let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
                Arc::new(arrow::array::StringArray::from(vec!["b", "b", "b", "a", "a"])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 0.0, 1.0])),
                Arc::new(UInt32Array::from(vec![0u32, 0, 0, 1, 1])),
                Arc::new(arrow::array::StringArray::from(color_rows)),
            ]).unwrap();
            let scales = ordinal_x_scales();
            let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
            let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
            super::build(&ctx)
        };

        let with_color = call_built(true, vec![Some("u"), Some("u"), Some("u"), Some("u"), Some("u")]);
        assert_eq!(with_color.dropped_series.unwrap().names(), vec!["g=a, h=u (two vertices)"]);

        let without_color = call_built(false, vec![Some("u"), Some("u"), Some("u"), Some("u"), Some("u")]);
        assert_eq!(without_color.dropped_series.unwrap().names(), vec!["g=a (two vertices)"]);

        // A null color value renders the way `describe_group_row` already
        // renders a null.
        let null_color = call_built(true, vec![Some("u"), Some("u"), Some("u"), None, None]);
        assert_eq!(null_color.dropped_series.unwrap().names(), vec!["g=a, h=null (two vertices)"]);

        // Call-built, neither a categorical position nor a color: falls
        // back to its ordinal, under its own mark's noun ("polygon", not
        // the line family's "series").
        let mut kwargs = crate::spec::mark_style::MarkKwargsSpec::default();
        kwargs.detail = Some("group_id".to_string());
        let neither_spec = ChartSpec {
            mark: Mark::Polygon,
            entry: Some("mark_violin".to_string()),
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(kwargs),
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 0.0, 1.0])),
            Arc::new(UInt32Array::from(vec![0u32, 0, 0, 1, 1])),
        ]).unwrap();
        let (scales, _) = resolve_scales(&neither_spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(neither_spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &neither_spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.dropped_series.unwrap().names(), vec!["polygon 2 of 2 (two vertices)"]);

        // A caller-placed ring (no call built this layer) with both
        // `detail` and a color is named by `detail` alone.
        let caller_spec = polygon_spec(Some("ring"), Some("kind"));
        let caller_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("ring", DataType::Utf8, false),
            Field::new("kind", DataType::Utf8, false),
        ]));
        let caller_batch = arrow::record_batch::RecordBatch::try_new(caller_schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 0.0, 1.0])),
            Arc::new(arrow::array::StringArray::from(vec!["p", "p", "p", "q", "q"])),
            Arc::new(arrow::array::StringArray::from(vec!["k1", "k1", "k1", "k1", "k1"])),
        ]).unwrap();
        let (scales, _) = resolve_scales(&caller_spec, &caller_batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(caller_spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &caller_spec, panel: &panel, theme: &theme, scales: &scales, batch: &caller_batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.dropped_series.unwrap().names(), vec!["ring=q (two vertices)"]);
    }

    /// A `UInt32` detail written with the LATER id first paints that id's
    /// ring first: `detail_groups` partitions in first-appearance order,
    /// never ascending id order (which would paint id `1`'s ring first
    /// here).
    #[test]
    fn a_uint32_detail_paints_its_rings_in_first_appearance_order() {
        let spec = polygon_spec(Some("group_id"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,   2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,   0.0, 0.0, 1.0])),
            Arc::new(UInt32Array::from(vec![5u32, 5, 5,   1, 1, 1])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        assert_eq!(super::detail_groups(&ctx), vec![vec![0, 1, 2], vec![3, 4, 5]], "id 5 appears first and must paint first, not id 1 by value order");
        let result = super::build(&ctx);
        assert_eq!(result.data_indices, Some(vec![0, 3]));
    }

    /// An `Int64` detail written with the LATER id first paints that id's
    /// ring first, the same first-appearance rule as `UInt32`.
    #[test]
    fn an_int64_detail_paints_its_rings_in_first_appearance_order() {
        use arrow::array::Int64Array;
        let spec = polygon_spec(Some("hex_id"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("hex_id", DataType::Int64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,   2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,   0.0, 0.0, 1.0])),
            Arc::new(Int64Array::from(vec![300i64, 300, 300,   100, 100, 100])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        assert_eq!(super::detail_groups(&ctx), vec![vec![0, 1, 2], vec![3, 4, 5]], "id 300 appears first and must paint first, not id 100 by value order");
        let result = super::build(&ctx);
        assert_eq!(result.data_indices, Some(vec![0, 3]));
    }

    /// A `Float64` detail (including negative ids) paints in
    /// first-appearance order, which here differs from BOTH ascending
    /// numeric order and `to_bits`-as-`i64` order — the two orders a
    /// `BTreeMap`-keyed implementation could fall back to without any
    /// single test telling them apart from first-appearance order.
    #[test]
    fn a_float64_detail_paints_its_rings_in_first_appearance_order() {
        let spec = polygon_spec(Some("value"), None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![
                0.0, 1.0, 0.5,    2.0, 3.0, 2.5,    4.0, 5.0, 4.5,
            ])),
            Arc::new(Float64Array::from(vec![
                0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,
            ])),
            Arc::new(Float64Array::from(vec![
                3.0, 3.0, 3.0,    -2.0, -2.0, -2.0,    -1.0, -1.0, -1.0,
            ])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        assert_eq!(
            super::detail_groups(&ctx),
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]],
            "value 3.0 appears first, then -2.0, then -1.0 — neither ascending numeric order \
             (-2.0, -1.0, 3.0) nor to_bits-as-i64 order (-1.0, -2.0, 3.0) matches",
        );
        let result = super::build(&ctx);
        assert_eq!(result.data_indices, Some(vec![0, 3, 6]));
    }

    /// An `Int32` detail (the category-reader fallback branch, since
    /// `detail_groups` only special-cases `UInt32`/`Int64`/`Float64`)
    /// groups identically to its `Utf8` twin, in first-appearance order —
    /// never merged into one ring, and never reordered by value.
    #[test]
    fn an_int32_detail_groups_like_its_utf8_twin_in_first_appearance_order() {
        use arrow::array::Int32Array;

        let int32_spec = polygon_spec(Some("zone"), None);
        let int32_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("zone", DataType::Int32, false),
        ]));
        let int32_batch = arrow::record_batch::RecordBatch::try_new(int32_schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,   2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,   0.0, 0.0, 1.0])),
            Arc::new(Int32Array::from(vec![7, 7, 7,   2, 2, 2])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let (int32_scales, _) = resolve_scales(&int32_spec, &int32_batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(int32_spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let int32_ctx = DrawCtx { spec: &int32_spec, panel: &panel, theme: &theme, scales: &int32_scales, batch: &int32_batch, mark_style: &mark_style };
        let int32_groups = super::detail_groups(&int32_ctx);
        assert_eq!(int32_groups, vec![vec![0, 1, 2], vec![3, 4, 5]], "id 7 appears first and must paint first, not id 2 by value order");
        let int32_n = super::build(&int32_ctx).nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count();
        assert_eq!(int32_n, 2, "Int32 detail must split into 2 rings, not merge into 1, got {int32_n}");

        let utf8_spec = polygon_spec(Some("zone"), None);
        let utf8_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("zone", DataType::Utf8, false),
        ]));
        let utf8_batch = arrow::record_batch::RecordBatch::try_new(utf8_schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5,   2.0, 3.0, 2.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0,   0.0, 0.0, 1.0])),
            Arc::new(arrow::array::StringArray::from(vec!["q", "q", "q", "p", "p", "p"])),
        ]).unwrap();
        let (utf8_scales, _) = resolve_scales(&utf8_spec, &utf8_batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let utf8_ctx = DrawCtx { spec: &utf8_spec, panel: &panel, theme: &theme, scales: &utf8_scales, batch: &utf8_batch, mark_style: &mark_style };
        let utf8_groups = super::detail_groups(&utf8_ctx);
        assert_eq!(utf8_groups, int32_groups, "Int32 detail must group identically to its Utf8 twin");
    }

    /// A ring's third vertex being `NaN` or null leaves it with two placed
    /// vertices, below the mark's three-vertex minimum — the same boundary
    /// a null or non-finite coordinate leaves on every vertex-unit mark, so
    /// it is named and counted as unplaced, never as a third row that
    /// wasn't there.
    #[test]
    fn a_nan_or_null_vertex_leaves_a_ring_with_two_placed_vertices() {
        let nan_case = build_rings(
            vec![Some(0.0), Some(1.0), Some(0.5), Some(2.0), Some(3.0), Some(2.5)],
            vec![Some(0.0), Some(0.0), Some(1.0), Some(0.0), Some(0.0), Some(f64::NAN)],
            vec![0, 0, 0, 1, 1, 1],
        );
        assert_eq!(nan_case.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(), 1);
        assert!(nan_case.empty_reason.is_none(), "{:?}", nan_case.empty_reason);
        assert_eq!(nan_case.dropped_series.unwrap().names(), vec!["group_id=1 (two vertices)"]);

        let null_case = build_rings(
            vec![Some(0.0), Some(1.0), Some(0.5), Some(2.0), Some(3.0), Some(2.5)],
            vec![Some(0.0), Some(0.0), Some(1.0), Some(0.0), Some(0.0), None],
            vec![0, 0, 0, 1, 1, 1],
        );
        assert_eq!(null_case.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(), 1);
        assert!(null_case.empty_reason.is_none(), "{:?}", null_case.empty_reason);
        assert_eq!(null_case.dropped_series.unwrap().names(), vec!["group_id=1 (two vertices)"]);
    }

    /// A call-built ring whose categorical x and color are bound to the
    /// SAME field (the "hue = x" idiom) is still named by that field
    /// exactly once, not once per drawn position it also matches.
    #[test]
    fn a_call_built_ring_with_the_same_categorical_x_and_color_field_is_named_once() {
        use crate::render::scale_resolve::ResolvedScales;
        use crate::scale::discrete::DiscreteLayout;
        use crate::scale::ordinal::OrdinalScale;

        let mut kwargs = crate::spec::mark_style::MarkKwargsSpec::default();
        kwargs.detail = Some("group_id".to_string());
        let spec = ChartSpec {
            mark: Mark::Polygon,
            entry: Some("mark_violin".to_string()),
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(kwargs),
            ..Default::default()
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
            Field::new("group_id", DataType::UInt32, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::StringArray::from(vec!["b", "b", "b", "a", "a"])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 0.5, 0.0, 1.0])),
            Arc::new(UInt32Array::from(vec![0u32, 0, 0, 1, 1])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = rect_panel();
        let scales = ResolvedScales {
            x: ScaleKind::Ordinal(OrdinalScale::new_internal(vec!["a".into(), "b".into()], vec![0.0, 100.0], DiscreteLayout::UNPADDED)),
            y: ScaleKind::Linear(crate::scale::linear::LinearScale::new_internal(vec![0.0, 1.0], vec![100.0, 0.0], false, false)),
            color: None, size: None, shape: None, opacity: None, fill_opacity: None, stroke_opacity: None, stroke_dash: None, stroke_width: None, x2: None, y2: None, y_slots: Default::default(),
            raster_cells: None,
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(
            result.dropped_series.unwrap().names(),
            vec!["g=a (two vertices)"],
            "a field bound to both the categorical x position and color must be named once, not 'g=a, g=a'",
        );
    }
}
