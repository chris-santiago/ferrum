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

use crate::render::color::{ContinuousScheme, NamedContinuous};
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) {
        Ok(v) => v,
        Err(_) => return,
    };
    let ys = match col_as_f64(ctx.batch, yf) {
        Ok(v) => v,
        Err(_) => return,
    };

    // --- Group rows by detail column (or single group if unset) ---
    // BTreeMap keeps groups ordered for deterministic SVG emission.
    let detail_field = ctx.mark_style.detail.as_deref();
    let groups: BTreeMap<i64, Vec<usize>> = match detail_field {
        Some(field) => {
            let arr = match ctx.batch.column_by_name(field) {
                Some(a) => a,
                None => return, // detail column missing: nothing to draw
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
                // Unknown dtype: fall back to a single group containing all rows.
                g.insert(0, (0..xs.len()).collect());
            }
            g
        }
        None => {
            let mut g = BTreeMap::new();
            g.insert(0, (0..xs.len()).collect());
            g
        }
    };

    // --- Resolve color encoding mode ---
    let cf = color_field(ctx, spec);
    let color_arr = cf.and_then(|f| ctx.batch.column_by_name(f));
    let color_is_quantitative = color_arr
        .map(|a| a.as_any().downcast_ref::<Float64Array>().is_some())
        .unwrap_or(false);

    // Categorical lookup values (parallel to row index) — only populated when
    // color encoding is set + column is Utf8.
    let color_str_values: Option<Vec<Option<String>>> =
        if !color_is_quantitative {
            cf.and_then(|f| col_as_str(ctx.batch, f).ok())
        } else {
            None
        };

    // For quantitative coloring: compute global vmin/vmax + scheme.
    let scheme = if color_is_quantitative {
        let named = ctx
            .mark_style
            .cmap
            .as_deref()
            .and_then(NamedContinuous::from_name)
            .unwrap_or(NamedContinuous::Viridis);
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

    // --- Emit one polygon per group ---
    for (_id, indices) in &groups {
        let ring: Vec<(f64, f64)> = indices
            .iter()
            .filter_map(|&i| {
                let xv = xs[i]?;
                let yv = ys[i]?;
                if !xv.is_finite() || !yv.is_finite() {
                    return None;
                }
                let cx = ctx.scales.x.to_pixel_f64(xv)?;
                let cy = ctx.scales.y.to_pixel_f64(yv)?;
                Some((cx, cy))
            })
            .collect();
        if ring.len() < 3 {
            continue;
        }

        // Resolve fill for this group.
        let fill = if color_is_quantitative {
            // Quantitative: take first row's color value, normalize, sample cmap.
            let first_row = indices[0];
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
        } else if let (Some(values), Some(scale @ ColorScale::Categorical { .. })) =
            (color_str_values.as_ref(), &ctx.scales.color)
        {
            // Categorical: take first row's category string, look up in scale.
            let first_row = indices[0];
            match values.get(first_row).and_then(|v| v.as_deref()) {
                Some(v) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };

        out.polygon(
            &[ring],
            &FillStroke {
                fill: Some(fill),
                stroke: ctx.mark_style.stroke,
                stroke_width: ctx.mark_style.stroke_width,
            },
        );
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<path ").count(), 1, "expected 1 path: {s}");
        assert!(s.contains(" Z\""), "polygon path must close with Z: {s}");
        assert!(s.contains(r#"d="M "#), "polygon path must start with M: {s}");
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<path ").count(), 2, "expected 2 paths (one per group): {s}");
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<path ").count(), 3, "expected 3 paths: {s}");
    }

    #[test]
    fn polygon_quantitative_color_yields_distinct_fills() {
        // 3 groups, color column = Float64 with values 0.0, 0.5, 1.0 across groups.
        // Expect 3 distinct fill="..." attributes (sampled from viridis).
        //
        // NOTE: `resolve_scales` only supports categorical color (Utf8 column).
        // For quantitative color, the polygon renderer reads the Float64 column
        // directly via `color_field()` and bypasses the scale entirely. To exercise
        // that path in this unit test, we construct ResolvedScales manually with
        // `color: None` while keeping `spec.encoding.color = Some("value")` so the
        // renderer's `color_field()` returns "value".
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();

        // Collect distinct fill="#..." values from the polygon paths.
        let mut fills: Vec<&str> = Vec::new();
        for chunk in s.split("<path ").skip(1) {
            if let Some(rest) = chunk.split_once(r#"fill=""#) {
                if let Some(end) = rest.1.find('"') {
                    fills.push(&rest.1[..end]);
                }
            }
        }
        let mut uniq: Vec<&&str> = fills.iter().collect();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 3, "expected 3 distinct polygon fills, got {fills:?} from svg: {s}");
    }
}
