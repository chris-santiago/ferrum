//! mark_polygon: closed-region drawer. Groups vertices into one polygon per
//! `mark_kwargs.detail` value (e.g. `hex_id`, `violin_id`, `level_id`). When
//! `detail` is unset, all rows form a single polygon.
//!
//! Color resolution:
//! - `color` encoding + Categorical scale → per-group string lookup (mirror area.rs).
//! - `color` encoding + Float64 column → quantitative; sample named cmap (default Viridis)
//!   at `(value - vmin) / (vmax - vmin)`. One polygon per detail group; group color
//!   taken from the first row in the group.
//! - No `color` encoding → single fill from `mark_style.fill`.
//!
//! Used by violin (Task 25), contour-fill (Task 30), and hex (Task 28) composite marks.

use std::collections::BTreeMap;

use arrow::array::{Array, Float64Array, UInt32Array};

use crate::render::color::{with_opacity, ContinuousScheme, NamedContinuous};
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ColorScale;

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::{to_scene_fill_stroke, MarkBuildResult, MetadataColumns};
    use ferrum_scene::{MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult {
        kind: MarkBatchKind::Polygon,
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

    // Pre-resolve per-row x pixel positions (numeric or ordinal).
    let xpx: Vec<Option<f64>> = if let Ok(v) = col_as_f64(ctx.batch, xf) {
        v.into_iter()
            .map(|opt| opt.and_then(|x| ctx.scales.x.to_pixel_f64(x)))
            .collect()
    } else if let Ok(v) = col_as_str(ctx.batch, xf) {
        v.into_iter()
            .map(|opt| opt.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
            .collect()
    } else {
        return empty();
    };
    let ys = match col_as_f64(ctx.batch, yf) {
        Ok(v) => v,
        Err(_) => return empty(),
    };

    // --- Group rows by detail column (or single group if unset) ---
    let detail_field = ctx.mark_style.detail.as_deref();
    let groups: BTreeMap<i64, Vec<usize>> = match detail_field {
        Some(field) => {
            let arr = match ctx.batch.column_by_name(field) {
                Some(a) => a,
                None => return empty(),
            };
            let mut g: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
            if let Some(u) = arr.as_any().downcast_ref::<UInt32Array>() {
                for i in 0..u.len() {
                    if !u.is_null(i) {
                        g.entry(u.value(i) as i64).or_default().push(i);
                    }
                }
            } else if let Some(f) = arr.as_any().downcast_ref::<Float64Array>() {
                for i in 0..f.len() {
                    if !f.is_null(i) {
                        g.entry(f.value(i).to_bits() as i64).or_default().push(i);
                    }
                }
            } else {
                g.insert(0, (0..xpx.len()).collect());
            }
            g
        }
        None => {
            let mut g = BTreeMap::new();
            g.insert(0, (0..xpx.len()).collect());
            g
        }
    };

    // --- Resolve color encoding mode ---
    let cf = color_field(ctx, spec);
    let color_arr = cf.and_then(|f| ctx.batch.column_by_name(f));
    let color_is_quantitative = color_arr
        .map(|a| a.as_any().downcast_ref::<Float64Array>().is_some())
        .unwrap_or(false);

    let color_str_values: Option<Vec<Option<String>>> =
        if !color_is_quantitative {
            cf.and_then(|f| col_as_str(ctx.batch, f).ok())
        } else {
            None
        };

    let scheme = if color_is_quantitative {
        let named = ctx
            .mark_style
            .cmap
            .as_deref()
            .and_then(NamedContinuous::from_name)
            .unwrap_or_else(|| {
                NamedContinuous::from_name(&ctx.theme.sequential_scheme)
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

    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    // --- Emit one polygon per group ---
    for (_id, group_indices) in &groups {
        let ring: Vec<(f64, f64)> = group_indices
            .iter()
            .filter_map(|&i| {
                let cx = xpx.get(i).copied().flatten()?;
                let yv = ys[i]?;
                if !yv.is_finite() {
                    return None;
                }
                let cy = ctx.scales.y.to_pixel_f64(yv)?;
                let xo = x_offsets.get(i).copied().unwrap_or(0.0);
                let yo = y_offsets.get(i).copied().unwrap_or(0.0);
                Some((cx + xo, cy + yo))
            })
            .collect();
        if ring.len() < 3 {
            continue;
        }

        // Resolve fill for this group.
        let fill = if color_is_quantitative {
            let first_row = group_indices[0];
            if let Some(a) = color_arr.and_then(|a| a.as_any().downcast_ref::<Float64Array>()) {
                if a.is_null(first_row) {
                    ctx.mark_style.fill
                } else {
                    let v = a.value(first_row);
                    let t = ((v - vmin) / denom).clamp(0.0, 1.0);
                    scheme.as_ref().map(|s| s.sample(t)).unwrap_or(ctx.mark_style.fill)
                }
            } else {
                ctx.mark_style.fill
            }
        } else if let (Some(values), Some(scale)) =
            (color_str_values.as_ref(), &ctx.scales.color)
        {
            let first_row = group_indices[0];
            match values.get(first_row).and_then(|v| v.as_deref()) {
                Some(v) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);

        let points: Vec<[f64; 2]> = ring.iter().map(|&(x, y)| [x, y]).collect();
        nodes.push(SceneNode::Polygon {
            points,
            style: to_scene_fill_stroke(
                Some(fill),
                ctx.mark_style.stroke,
                ctx.mark_style.stroke_width,
                ctx.mark_style.opacity,
                None,
            ),
        });
        indices.push(group_indices[0]);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Polygon,
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
            data: DataRef::default(),
            mark: Mark::Polygon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: color.map(|f| EncodingSpec { field: f.into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: Some(kwargs),
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        }
    }

    fn rect_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
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
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Polygon { .. })).count(), 1, "expected 1 polygon");
        // Polygon must have at least 3 points.
        let has_points = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Polygon { points, .. } = n { points.len() >= 3 } else { false }
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
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon);
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
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon);
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
            x2: None,
            y2: None,
        };
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Polygon);
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
}
