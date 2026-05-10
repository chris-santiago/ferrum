//! mark_point: render each row as a shape glyph at (scale_x(row.x), scale_y(row.y)).
//! Phase 7: always emits <circle> using ctx.mark_style.point_size.
//! Phase 8a: honors per-row size/shape/opacity from ctx.scales when populated.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind, ShapeKind};
use crate::render::svg::{FillStroke, Stroke, SvgBuffer};

/// Emit one shape glyph centered at (cx, cy) with the given radius and fill/stroke style.
///
/// `ShapeKind::Circle` emits a `<circle>` element — byte-identical to the Phase 7 path.
/// Other shapes emit the corresponding SVG primitive(s).
fn emit_shape(out: &mut SvgBuffer, kind: ShapeKind, cx: f64, cy: f64, r: f64,
              style: &FillStroke) {
    match kind {
        ShapeKind::Circle => out.circle(cx, cy, r, style),
        ShapeKind::Square => {
            let s = r * 1.6; // visual area parity with circle
            out.rect(crate::layout::Rect { x: cx - s / 2.0, y: cy - s / 2.0, w: s, h: s },
                     style, None);
        }
        ShapeKind::Cross => {
            // Two perpendicular stroked lines; stroke color is the fill color.
            let stroke_color = style.fill.unwrap_or(crate::render::color::from_rgb(0, 0, 0));
            let stroke = Stroke {
                stroke: stroke_color,
                stroke_width: r * 0.4,
                stroke_dash: None,
            };
            let arm = r * 0.5;
            out.line(cx - arm, cy, cx + arm, cy, &stroke);
            out.line(cx, cy - arm, cx, cy + arm, &stroke);
        }
        ShapeKind::Diamond => {
            let d = r * 1.4;
            let path = format!("M {} {} L {} {} L {} {} L {} {} Z",
                cx, cy - d, cx + d, cy, cx, cy + d, cx - d, cy);
            out.path(&path, style);
        }
        ShapeKind::TriangleUp => {
            let h = r * 1.4;
            let path = format!("M {} {} L {} {} L {} {} Z",
                cx, cy - h, cx + h * 0.866, cy + h * 0.5, cx - h * 0.866, cy + h * 0.5);
            out.path(&path, style);
        }
        ShapeKind::TriangleDown => {
            let h = r * 1.4;
            let path = format!("M {} {} L {} {} L {} {} Z",
                cx, cy + h, cx + h * 0.866, cy - h * 0.5, cx - h * 0.866, cy - h * 0.5);
            out.path(&path, style);
        }
    }
}

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() { return; }

    // Color encoding (Phase 7 feature, preserved).
    let color_values: Option<Vec<Option<String>>> = color_field(ctx, spec)
        .and_then(|f| col_as_str(ctx.batch, f).ok());

    // Phase 8a: optional per-row size / shape / opacity vectors.
    let size_values: Option<Vec<Option<f64>>> = spec.encoding.size
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let shape_values: Option<Vec<Option<String>>> = spec.encoding.shape
        .as_ref()
        .and_then(|e| col_as_str(ctx.batch, &e.field).ok());

    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Default radius from mark_style (Phase 7 path, area → radius conversion).
    let default_radius = (ctx.mark_style.point_size / std::f64::consts::PI).sqrt();

    for i in 0..xs.len() {
        let (xv, yv) = match (xs[i], ys[i]) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
            _ => continue,
        };
        let cx = match scale_value(&ctx.scales.x, xv, None) { Some(p) => p, None => continue };
        let cy = match scale_value(&ctx.scales.y, yv, None) { Some(p) => p, None => continue };

        // Resolve color (same logic as Phase 7).
        let fill_base = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };

        // Resolve per-row opacity (Phase 8a), falling back to mark_style.opacity.
        let row_opacity = if let (Some(values), Some(scale)) = (&opacity_values, &ctx.scales.opacity) {
            match values[i].and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(op) => op,
                None => ctx.mark_style.opacity,
            }
        } else {
            ctx.mark_style.opacity
        };

        let fill = with_opacity(fill_base, row_opacity);

        let style = FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        };

        // Resolve per-row radius from size encoding (area → radius), falling back to default.
        let radius = if let (Some(values), Some(scale)) = (&size_values, &ctx.scales.size) {
            match values[i].and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(area) => (area / std::f64::consts::PI).sqrt(),
                None => default_radius,
            }
        } else {
            default_radius
        };

        // Resolve per-row shape kind using ShapeScale.lookup(), falling back to Circle.
        let shape_kind = if let (Some(values), Some(scale)) = (&shape_values, &ctx.scales.shape) {
            match values[i].as_deref() {
                Some(v) => scale.lookup(v).unwrap_or(ShapeKind::Circle),
                None => ShapeKind::Circle,
            }
        } else {
            ShapeKind::Circle
        };

        emit_shape(out, shape_kind, cx, cy, radius, &style);
    }
}

fn scale_value(s: &ScaleKind, v: f64, label: Option<&str>) -> Option<f64> {
    match s {
        ScaleKind::Linear(_) | ScaleKind::Time(_) | ScaleKind::Log(_) | ScaleKind::Symlog(_) => {
            s.to_pixel_f64(v)
        }
        ScaleKind::Ordinal(_) => label.and_then(|l| s.to_pixel_str(l)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style};
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    fn three_row_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        }
    }

    fn three_row_batch() -> arrow::record_batch::RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap()
    }

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None,
        }
    }

    #[test]
    fn three_rows_emit_three_circles() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 3);
    }

    #[test]
    fn out_of_domain_rows_are_skipped() {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, f64::NAN, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let spec = three_row_spec();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 2);
    }

    // ── Phase 8a new tests ──────────────────────────────────────────────────

    /// Build a batch with x, y, and a size column [10.0, 20.0, 30.0].
    fn batch_with_size() -> arrow::record_batch::RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sz", DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap()
    }

    fn spec_with_size() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                size: Some(EncodingSpec { field: "sz".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        }
    }

    #[test]
    fn point_with_size_encoding_emits_three_circles() {
        let spec = spec_with_size();
        let batch = batch_with_size();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let svg = out.finish();

        // Three circles emitted (size encoding uses Circle shape by default).
        assert_eq!(svg.matches("<circle ").count(), 3);

        // Extract radii from `r="..."` attributes and verify they are strictly increasing.
        let radii: Vec<f64> = svg.split("<circle ").skip(1).filter_map(|seg| {
            seg.find(" r=\"").map(|pos| {
                let rest = &seg[pos + 4..];
                let end = rest.find('"').unwrap_or(rest.len());
                rest[..end].parse::<f64>().ok()
            }).flatten()
        }).collect();

        assert_eq!(radii.len(), 3, "expected 3 radius values; got: {svg}");
        assert!(radii[0] < radii[1] && radii[1] < radii[2],
            "radii not strictly increasing: {radii:?}");
    }

    /// Build a batch with x, y, and a shape column ["cat", "dog", "bird"].
    fn batch_with_shape() -> arrow::record_batch::RecordBatch {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(StringArray::from(vec!["cat", "dog", "bird"])),
        ]).unwrap()
    }

    fn spec_with_shape() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                shape: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        }
    }

    #[test]
    fn point_with_shape_encoding_emits_3_shape_kinds() {
        // Domain ["cat","dog","bird"] → SHAPE_PALETTE[0..3] = Circle, Square, Cross.
        let spec = spec_with_shape();
        let batch = batch_with_shape();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let svg = out.finish();

        // "cat" → Circle, "dog" → Square, "bird" → Cross (2 × <line>).
        assert_eq!(svg.matches("<circle ").count(), 1, "circle count; svg: {svg}");
        assert_eq!(svg.matches("<rect ").count(), 1, "rect count; svg: {svg}");
        assert_eq!(svg.matches("<line ").count(), 2, "line count (cross); svg: {svg}");
    }

    /// Build a batch with x, y, and an opacity column [0.2, 0.5, 0.9].
    fn batch_with_opacity() -> arrow::record_batch::RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("op", DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.2, 0.5, 0.9])),
        ]).unwrap()
    }

    fn spec_with_opacity() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        }
    }

    #[test]
    fn point_with_opacity_encoding_sets_fill_opacity_per_row() {
        // The default mark color (fully opaque) baked with varying per-row opacity
        // must produce rgba(...) fill strings (alpha < 1.0).
        let spec = spec_with_opacity();
        let batch = batch_with_opacity();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let svg = out.finish();

        // At least one row must have a fractional opacity → rgba(...) fill.
        assert!(svg.contains("rgba("), "expected rgba fill in svg; got: {svg}");
        // All three rows are emitted.
        assert_eq!(svg.matches("<circle ").count(), 3);
    }
}
