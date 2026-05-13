//! mark_point: render each row as a shape glyph at (scale_x(row.x), scale_y(row.y)).
//! Phase 7: always emits <circle> using ctx.mark_style.point_size.
//! Phase 8a: honors per-row size/shape/opacity from ctx.scales when populated.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::scale_resolve::{ColorScale, ScaleKind, ShapeKind};
use crate::render::svg::{FillStroke, Stroke, SvgBuffer};

/// Parse a shape name string to a `ShapeKind`. Unknown values fall back to `Circle`.
fn shape_from_str(s: &str) -> ShapeKind {
    match s {
        "square" => ShapeKind::Square,
        "cross" => ShapeKind::Cross,
        "diamond" => ShapeKind::Diamond,
        "triangle-up" | "triangle_up" => ShapeKind::TriangleUp,
        "triangle-down" | "triangle_down" => ShapeKind::TriangleDown,
        _ => ShapeKind::Circle, // "circle" and unknown values
    }
}

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

    // Read per-axis as f64 OR as string depending on column dtype. Ordinal
    // axes (Utf8 columns) route through `to_pixel_str`; quantitative axes
    // through `to_pixel_f64`. Reading both and dispatching at the loop level
    // lets `mark_point` participate in categorical scatters (e.g. Phase 10d
    // SHAP beeswarm with feature on y-axis).
    let xs_f64 = col_as_f64(ctx.batch, xf).ok();
    let xs_str = col_as_str(ctx.batch, xf).ok();
    let ys_f64 = col_as_f64(ctx.batch, yf).ok();
    let ys_str = col_as_str(ctx.batch, yf).ok();
    let n = xs_f64
        .as_ref().map(|v| v.len())
        .or_else(|| xs_str.as_ref().map(|v| v.len()))
        .unwrap_or(0);
    let n_y = ys_f64
        .as_ref().map(|v| v.len())
        .or_else(|| ys_str.as_ref().map(|v| v.len()))
        .unwrap_or(0);
    if n == 0 || n != n_y { return; }

    // Color encoding. Phase 7 read color as Utf8 only (categorical lookups);
    // Phase 10d adds a Continuous path that reads color as f64 and resolves
    // via `lookup_f64` (mirrors the Phase 10c-pre rect.rs pattern).
    let cfield = color_field(ctx, spec);
    let color_values_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        // No color scale resolved yet → fall back to Utf8 read so legacy
        // single-color charts behave as before.
        (None, Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_values_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };

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

    // Phase 9c — per-row pixel offsets from a position adjustment (e.g. Dodge into
    // an ordinal-x band). Zero-valued when no adjustment was applied.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // SVG metadata channels (tooltip, href, description).
    let meta = MetadataColumns::from_ctx(ctx);

    for i in 0..n {
        // Resolve x-pixel: prefer Utf8 lookup when the scale is ordinal AND a
        // string column is available, falling back to f64.
        let cx = match &ctx.scales.x {
            ScaleKind::Ordinal(_) => match &xs_str {
                Some(v) => match &v[i] {
                    Some(s) => match ctx.scales.x.to_pixel_str(s.as_str()) {
                        Some(p) => p, None => continue,
                    },
                    None => continue,
                },
                None => continue,
            },
            _ => match xs_f64.as_ref().and_then(|v| v[i]) {
                Some(a) if a.is_finite() => match ctx.scales.x.to_pixel_f64(a) {
                    Some(p) => p, None => continue,
                },
                _ => continue,
            },
        };
        let cy = match &ctx.scales.y {
            ScaleKind::Ordinal(_) => match &ys_str {
                Some(v) => match &v[i] {
                    Some(s) => match ctx.scales.y.to_pixel_str(s.as_str()) {
                        Some(p) => p, None => continue,
                    },
                    None => continue,
                },
                None => continue,
            },
            _ => match ys_f64.as_ref().and_then(|v| v[i]) {
                Some(a) if a.is_finite() => match ctx.scales.y.to_pixel_f64(a) {
                    Some(p) => p, None => continue,
                },
                _ => continue,
            },
        };
        let cx = cx + x_offsets[i];
        let cy = cy + y_offsets[i];

        // Resolve color: Continuous → lookup_f64 over the numeric column;
        // Categorical → string lookup; otherwise use the mark style default.
        let fill_base = match (&ctx.scales.color, &color_values_f64, &color_values_str) {
            (Some(scale @ ColorScale::Continuous { .. }), Some(values), _) => {
                match values[i] {
                    Some(v) if v.is_finite() => scale.lookup_f64(v).unwrap_or(ctx.mark_style.fill),
                    _ => ctx.mark_style.fill,
                }
            }
            (Some(scale @ ColorScale::Categorical { .. }), _, Some(values)) => {
                match values[i].as_deref() {
                    Some(v) => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                    None => ctx.mark_style.fill,
                }
            }
            _ => ctx.mark_style.fill,
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

        // S5: filled=false → hollow points: fill="none", color goes to stroke.
        let (effective_fill, effective_stroke, effective_sw) =
            if ctx.mark_style.filled == Some(false) {
                // Hollow: no fill, color applied to stroke with a visible stroke width.
                let sw = if ctx.mark_style.stroke_width > 0.0 {
                    ctx.mark_style.stroke_width
                } else {
                    1.5
                };
                (None, Some(fill_base), sw)
            } else {
                (Some(fill), ctx.mark_style.stroke, ctx.mark_style.stroke_width)
            };

        let style = FillStroke {
            fill: effective_fill,
            stroke: effective_stroke,
            stroke_width: effective_sw,
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

        // Resolve per-row shape kind:
        // 1. Data-driven shape encoding (ShapeScale), if present.
        // 2. S6: constant mark_style.shape, if set and encoding is absent.
        // 3. Default: Circle.
        let shape_kind = if let (Some(values), Some(scale)) = (&shape_values, &ctx.scales.shape) {
            match values[i].as_deref() {
                Some(v) => scale.lookup(v).unwrap_or(ShapeKind::Circle),
                None => ShapeKind::Circle,
            }
        } else if let Some(ref shape_name) = ctx.mark_style.shape {
            shape_from_str(shape_name)
        } else {
            ShapeKind::Circle
        };

        let wrapped = meta.open(i, out);
        emit_shape(out, shape_kind, cx, cy, radius, &style);
        if wrapped {
            meta.close(i, out);
        }
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
        position: None,
        title: None,
        axis_x: None, axis_y: None,
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
        position: None,
        title: None,
        axis_x: None, axis_y: None,
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
        position: None,
        title: None,
        axis_x: None, axis_y: None,
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
        position: None,
        title: None,
        axis_x: None, axis_y: None,
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
