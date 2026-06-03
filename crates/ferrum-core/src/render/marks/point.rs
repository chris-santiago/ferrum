//! mark_point: render each row as a shape glyph at (scale_x(row.x), scale_y(row.y)).
//! Phase 7: always emits <circle> using ctx.mark_style.point_size.
//! Phase 8a: honors per-row size/shape/opacity from ctx.scales when populated.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_positional_category_str, col_as_str, color_field, resolve_stroke_dash, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::scale_resolve::{ColorScale, ScaleKind, ShapeKind};

/// Parse a shape name string to a `ShapeKind`.
///
/// Returns `ShapeKind::Circle` for any unrecognised string. This is the
/// raw-spec fallback: specs that arrive via `ChartSpec::from_json` or other
/// routes that bypass the Python `mark_point(shape=)` boundary can carry
/// arbitrary strings; falling back to a circle avoids a panic on that path.
/// The Python layer (`ferrum.marks.base._VALID_POINT_SHAPES`) is the
/// primary validation gate for user-facing errors — unknown shape names
/// produce a clear `ValueError` there before any JSON is produced.
pub(crate) fn shape_from_str(s: &str) -> ShapeKind {
    match s {
        "circle" => ShapeKind::Circle,
        "square" => ShapeKind::Square,
        "cross" => ShapeKind::Cross,
        "diamond" => ShapeKind::Diamond,
        "triangle-up" | "triangle_up" => ShapeKind::TriangleUp,
        "triangle-down" | "triangle_down" => ShapeKind::TriangleDown,
        "|" | "vline" => ShapeKind::VLine,
        "-" | "hline" => ShapeKind::HLine,
        // Unknown strings from raw/JSON specs fall back to circle.
        // This path is not reachable via the normal Python API because
        // mark_point(shape=) validates against _VALID_POINT_SHAPES first.
        _ => ShapeKind::Circle,
    }
}

// ── Scene-graph build path (11a) ───────────────────────────────────

/// Emit one shape glyph as `SceneNode` variants. Returns a `Vec` because
/// `ShapeKind::Cross` produces two `Line` nodes while all other shapes
/// produce exactly one node.
///
/// `stroke_opacity`, `fill_opacity`, and `angle` flow directly from encoding
/// columns into the emitted `FillStroke` style — no scale transform.
pub(crate) struct ShapeStyle {
    pub fill: Option<crate::render::color::Color>,
    pub stroke: Option<crate::render::color::Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub stroke_opacity: f64,
    pub fill_opacity: f64,
    pub stroke_dash_idx: Option<f64>,
    pub angle: f64,
}

pub(crate) fn emit_shape_nodes(
    kind: ShapeKind,
    cx: f64,
    cy: f64,
    r: f64,
    style: ShapeStyle,
) -> Vec<ferrum_scene::SceneNode> {
    let ShapeStyle { fill, stroke, stroke_width, opacity, stroke_opacity, fill_opacity, stroke_dash_idx, angle } = style;
    use crate::render::draw::{to_scene_fill_stroke, to_scene_stroke};
    use ferrum_scene::{PathCmd, SceneNode};

    let dash_vec: Option<Vec<f64>> = stroke_dash_idx.and_then(resolve_stroke_dash);
    let dash_ref: Option<&[f64]> = dash_vec.as_deref();

    match kind {
        ShapeKind::Circle => {
            let mut style = to_scene_fill_stroke(fill, stroke, stroke_width, opacity, dash_ref);
            style.stroke_opacity = stroke_opacity;
            style.fill_opacity = fill_opacity;
            style.angle = angle;
            vec![SceneNode::Circle { cx, cy, r, style }]
        }
        ShapeKind::Square => {
            let s = r * 1.6;
            let mut style = to_scene_fill_stroke(fill, stroke, stroke_width, opacity, dash_ref);
            style.stroke_opacity = stroke_opacity;
            style.fill_opacity = fill_opacity;
            style.angle = angle;
            vec![SceneNode::Rect {
                x: cx - s / 2.0,
                y: cy - s / 2.0,
                w: s,
                h: s,
                style,
                corner_radius: 0.0,
            }]
        }
        ShapeKind::Cross => {
            let stroke_color =
                fill.unwrap_or(crate::render::color::from_rgb(0, 0, 0));
            let arm = r * 0.5;
            let sw = r * 0.4;
            let s1 = to_scene_stroke(stroke_color, sw, 1.0, None, None, None);
            let s2 = to_scene_stroke(stroke_color, sw, 1.0, None, None, None);
            vec![
                SceneNode::Line {
                    x1: cx - arm,
                    y1: cy,
                    x2: cx + arm,
                    y2: cy,
                    style: s1,
                },
                SceneNode::Line {
                    x1: cx,
                    y1: cy - arm,
                    x2: cx,
                    y2: cy + arm,
                    style: s2,
                },
            ]
        }
        ShapeKind::Diamond => {
            let d = r * 1.4;
            let mut style = to_scene_fill_stroke(fill, stroke, stroke_width, opacity, dash_ref);
            style.stroke_opacity = stroke_opacity;
            style.fill_opacity = fill_opacity;
            style.angle = angle;
            vec![SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: cx, y: cy - d },
                    PathCmd::LineTo { x: cx + d, y: cy },
                    PathCmd::LineTo { x: cx, y: cy + d },
                    PathCmd::LineTo { x: cx - d, y: cy },
                    PathCmd::Close,
                ],
                style,
                closed: true,
            }]
        }
        ShapeKind::TriangleUp => {
            let h = r * 1.4;
            let mut style = to_scene_fill_stroke(fill, stroke, stroke_width, opacity, dash_ref);
            style.stroke_opacity = stroke_opacity;
            style.fill_opacity = fill_opacity;
            style.angle = angle;
            vec![SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: cx, y: cy - h },
                    PathCmd::LineTo { x: cx + h * 0.866, y: cy + h * 0.5 },
                    PathCmd::LineTo { x: cx - h * 0.866, y: cy + h * 0.5 },
                    PathCmd::Close,
                ],
                style,
                closed: true,
            }]
        }
        ShapeKind::TriangleDown => {
            let h = r * 1.4;
            let mut style = to_scene_fill_stroke(fill, stroke, stroke_width, opacity, dash_ref);
            style.stroke_opacity = stroke_opacity;
            style.fill_opacity = fill_opacity;
            style.angle = angle;
            vec![SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: cx, y: cy + h },
                    PathCmd::LineTo { x: cx + h * 0.866, y: cy - h * 0.5 },
                    PathCmd::LineTo { x: cx - h * 0.866, y: cy - h * 0.5 },
                    PathCmd::Close,
                ],
                style,
                closed: true,
            }]
        }
        ShapeKind::VLine => {
            let stroke_color = fill.unwrap_or(crate::render::color::from_rgb(0, 0, 0));
            let arm = r * 0.7;
            let sw = r * 0.35;
            let s = to_scene_stroke(stroke_color, sw, 1.0, None, None, None);
            vec![SceneNode::Line {
                x1: cx,
                y1: cy - arm,
                x2: cx,
                y2: cy + arm,
                style: s,
            }]
        }
        ShapeKind::HLine => {
            let stroke_color = fill.unwrap_or(crate::render::color::from_rgb(0, 0, 0));
            let arm = r * 0.7;
            let sw = r * 0.35;
            let s = to_scene_stroke(stroke_color, sw, 1.0, None, None, None);
            vec![SceneNode::Line {
                x1: cx - arm,
                y1: cy,
                x2: cx + arm,
                y2: cy,
                style: s,
            }]
        }
    }
}

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::MarkBatchKind;

    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) {
        Some(f) => f,
        None => return MarkBuildResult {
            kind: MarkBatchKind::Point,
            nodes: vec![],
            data_indices: Some(vec![]),
            tooltips: None,
            hrefs: None,
            descriptions: None,
        },
    };
    let yf = match y_field(ctx, spec) {
        Some(f) => f,
        None => return MarkBuildResult {
            kind: MarkBatchKind::Point,
            nodes: vec![],
            data_indices: Some(vec![]),
            tooltips: None,
            hrefs: None,
            descriptions: None,
        },
    };

    let xs_f64 = col_as_f64(ctx.batch, xf).ok();
    // Use col_as_positional_category_str so integer-typed ordinal columns (e.g.
    // Int64 year values) stringify the same way the ordinal domain was built, and
    // a null positional category lands in its own band (FA-9).
    let xs_str = col_as_positional_category_str(ctx.batch, xf).ok();
    let ys_f64 = col_as_f64(ctx.batch, yf).ok();
    let ys_str = col_as_positional_category_str(ctx.batch, yf).ok();
    let n = xs_f64
        .as_ref().map(|v| v.len())
        .or_else(|| xs_str.as_ref().map(|v| v.len()))
        .unwrap_or(0);
    let n_y = ys_f64
        .as_ref().map(|v| v.len())
        .or_else(|| ys_str.as_ref().map(|v| v.len()))
        .unwrap_or(0);
    if n == 0 || n != n_y {
        return MarkBuildResult {
            kind: MarkBatchKind::Point,
            nodes: vec![],
            data_indices: Some(vec![]),
            tooltips: None,
            hrefs: None,
            descriptions: None,
        };
    }

    // Color encoding.
    let cfield = color_field(ctx, spec);
    let color_values_str: Option<Vec<Option<String>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Categorical { .. }), Some(f)) => col_as_str(ctx.batch, f).ok(),
        (None, Some(f)) => col_as_str(ctx.batch, f).ok(),
        _ => None,
    };
    let color_values_f64: Option<Vec<Option<f64>>> = match (&ctx.scales.color, cfield) {
        (Some(ColorScale::Continuous { .. }), Some(f)) => col_as_f64(ctx.batch, f).ok(),
        _ => None,
    };

    // Per-row size / shape / opacity vectors.
    let size_values: Option<Vec<Option<f64>>> = spec.encoding.size
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let shape_values: Option<Vec<Option<String>>> = spec.encoding.shape
        .as_ref()
        .and_then(|e| col_as_str(ctx.batch, &e.field).ok());

    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // Per-row stroke/angle channel values (direct passthrough — no scale transform).
    let stroke_opacity_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let stroke_dash_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_dash
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let angle_values: Option<Vec<Option<f64>>> = spec.encoding.angle
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let fill_opacity_values: Option<Vec<Option<f64>>> = spec.encoding.fill_opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let default_radius = (ctx.mark_style.point_size / std::f64::consts::PI).sqrt();

    // Per-row pixel offsets from position adjustment.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Metadata: build tooltips and hrefs as parallel vecs.
    let meta = MetadataColumns::from_ctx(ctx);
    let (tooltips, hrefs, descriptions) = meta.build_metadata(ctx);

    let mut nodes = Vec::new();
    let mut indices = Vec::new();

    for i in 0..n {
        // Resolve x-pixel.
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
        // Resolve y-pixel.
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

        // Resolve color.
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

        // Resolve per-row opacity.
        let row_opacity = if let (Some(values), Some(scale)) = (&opacity_values, &ctx.scales.opacity) {
            match values[i].and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(op) => op,
                None => ctx.mark_style.opacity,
            }
        } else {
            ctx.mark_style.opacity
        };

        let fill = with_opacity(fill_base, row_opacity);

        // filled=false → hollow points.
        let (effective_fill, effective_stroke, effective_sw) =
            if ctx.mark_style.filled == Some(false) {
                let sw = if ctx.mark_style.stroke_width > 0.0 {
                    ctx.mark_style.stroke_width
                } else {
                    1.5
                };
                (None, Some(fill_base), sw)
            } else {
                (Some(fill), ctx.mark_style.stroke, ctx.mark_style.stroke_width)
            };

        // Resolve per-row radius from size encoding.
        let radius = if let (Some(values), Some(scale)) = (&size_values, &ctx.scales.size) {
            match values[i].and_then(|v| scale.inner.to_pixel_f64(v)) {
                Some(area) => (area / std::f64::consts::PI).sqrt(),
                None => default_radius,
            }
        } else {
            default_radius
        };

        // Resolve per-row shape kind.
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

        // Resolve per-row stroke/angle channel values (direct passthrough).
        let row_stroke_opacity = stroke_opacity_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);

        let row_stroke_width = stroke_width_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(effective_sw);

        let row_stroke_dash = stroke_dash_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite());

        let row_angle = angle_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);

        let row_fill_opacity = fill_opacity_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);

        // When stroke_width encoding produces a positive value but no explicit
        // stroke color exists, use the fill color as the stroke so the width is
        // visible in SVG (stroke-width is only emitted when stroke is Some).
        let effective_stroke_for_row = if row_stroke_width > 0.0 && effective_stroke.is_none() && stroke_width_values.is_some() {
            effective_fill
        } else {
            effective_stroke
        };

        let shape_nodes = emit_shape_nodes(
            shape_kind, cx, cy, radius,
            ShapeStyle {
                fill: effective_fill,
                stroke: effective_stroke_for_row,
                stroke_width: row_stroke_width,
                opacity: row_opacity,
                stroke_opacity: row_stroke_opacity,
                fill_opacity: row_fill_opacity,
                stroke_dash_idx: row_stroke_dash,
                angle: row_angle,
            },
        );
        nodes.extend(shape_nodes);
        indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Point,
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
    use ferrum_scene::SceneNode;

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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
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
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count(), 3);
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
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count(), 2);
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
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
        let result = super::build(&ctx);

        // Three circles emitted (size encoding uses Circle shape by default).
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count(), 3);

        // Extract radii from Circle nodes and verify they are strictly increasing.
        let radii: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { r, .. } = n { Some(*r) } else { None }
        }).collect();

        assert_eq!(radii.len(), 3, "expected 3 radius values; got: {radii:?}");
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
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
        let result = super::build(&ctx);

        // "cat" → Circle, "dog" → Square, "bird" → Cross (2 × Line).
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count(), 1, "circle count");
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Rect { .. })).count(), 1, "rect count");
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2, "line count (cross)");
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        }
    }

    #[test]
    fn point_with_opacity_encoding_sets_fill_opacity_per_row() {
        // The default mark color (fully opaque) baked with varying per-row opacity
        // must produce fill colors with alpha < 255.
        let spec = spec_with_opacity();
        let batch = batch_with_opacity();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // At least one row must have a fractional opacity → fill color with alpha < 255.
        let has_translucent = result.nodes.iter().any(|n| {
            if let SceneNode::Circle { style, .. } = n {
                style.fill.map_or(false, |c| c.a < 255)
            } else {
                false
            }
        });
        assert!(has_translucent, "expected at least one circle with translucent fill");
        // All three rows are emitted.
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count(), 3);
    }

    // ── Task 2: stroke/angle encoding channels flow through to FillStroke ──

    fn batch_with_stroke_channels() -> arrow::record_batch::RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("so", DataType::Float64, false),  // stroke_opacity
            Field::new("sw", DataType::Float64, false),  // stroke_width
            Field::new("sd", DataType::Float64, false),  // stroke_dash index
            Field::new("ang", DataType::Float64, false), // angle
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.3, 0.6, 0.9])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])), // solid, dashed, dotted
            Arc::new(Float64Array::from(vec![0.0, 45.0, 90.0])),
        ]).unwrap()
    }

    fn spec_with_stroke_channels() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_opacity: Some(EncodingSpec { field: "so".into(), type_: None, ..Default::default() }),
                stroke_width: Some(EncodingSpec { field: "sw".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                angle: Some(EncodingSpec { field: "ang".into(), type_: None, ..Default::default() }),
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

    #[test]
    fn stroke_opacity_encoding_flows_to_fill_stroke() {
        let spec = spec_with_stroke_channels();
        let batch = batch_with_stroke_channels();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let circles: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style) } else { None }
        }).collect();
        assert_eq!(circles.len(), 3, "expected 3 circle nodes");

        // stroke_opacity values from batch: [0.3, 0.6, 0.9]
        assert!((circles[0].stroke_opacity - 0.3).abs() < 1e-5,
            "row 0 stroke_opacity: expected 0.3, got {}", circles[0].stroke_opacity);
        assert!((circles[1].stroke_opacity - 0.6).abs() < 1e-5,
            "row 1 stroke_opacity: expected 0.6, got {}", circles[1].stroke_opacity);
        assert!((circles[2].stroke_opacity - 0.9).abs() < 1e-5,
            "row 2 stroke_opacity: expected 0.9, got {}", circles[2].stroke_opacity);
    }

    #[test]
    fn angle_encoding_flows_to_fill_stroke() {
        let spec = spec_with_stroke_channels();
        let batch = batch_with_stroke_channels();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let circles: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style) } else { None }
        }).collect();

        // angle values: [0.0, 45.0, 90.0]
        assert!((circles[0].angle - 0.0).abs() < 1e-5,
            "row 0 angle: expected 0.0, got {}", circles[0].angle);
        assert!((circles[1].angle - 45.0).abs() < 1e-5,
            "row 1 angle: expected 45.0, got {}", circles[1].angle);
        assert!((circles[2].angle - 90.0).abs() < 1e-5,
            "row 2 angle: expected 90.0, got {}", circles[2].angle);
    }

    // ── VLine / HLine shape tests ───────────────────────────────────────────

    #[test]
    fn test_shape_from_str_circle() {
        assert_eq!(shape_from_str("circle"), ShapeKind::Circle);
    }

    #[test]
    fn test_shape_from_str_all_valid() {
        // All Python-validated shape names must map to a distinct ShapeKind.
        assert_eq!(shape_from_str("circle"), ShapeKind::Circle);
        assert_eq!(shape_from_str("square"), ShapeKind::Square);
        assert_eq!(shape_from_str("cross"), ShapeKind::Cross);
        assert_eq!(shape_from_str("diamond"), ShapeKind::Diamond);
        assert_eq!(shape_from_str("triangle-up"), ShapeKind::TriangleUp);
        assert_eq!(shape_from_str("triangle_up"), ShapeKind::TriangleUp);
        assert_eq!(shape_from_str("triangle-down"), ShapeKind::TriangleDown);
        assert_eq!(shape_from_str("triangle_down"), ShapeKind::TriangleDown);
        assert_eq!(shape_from_str("|"), ShapeKind::VLine);
        assert_eq!(shape_from_str("vline"), ShapeKind::VLine);
        assert_eq!(shape_from_str("-"), ShapeKind::HLine);
        assert_eq!(shape_from_str("hline"), ShapeKind::HLine);
    }

    #[test]
    fn test_shape_from_str_vline() {
        assert_eq!(shape_from_str("|"), ShapeKind::VLine);
        assert_eq!(shape_from_str("vline"), ShapeKind::VLine);
    }

    #[test]
    fn test_shape_from_str_hline() {
        assert_eq!(shape_from_str("-"), ShapeKind::HLine);
        assert_eq!(shape_from_str("hline"), ShapeKind::HLine);
    }

    /// Unknown shape strings from raw/JSON specs fall back to Circle.
    /// The Python API validates shape names before producing JSON, so this
    /// path is only reachable via hand-crafted or round-tripped specs.
    #[test]
    fn test_shape_from_str_unknown_defaults_to_circle() {
        assert_eq!(shape_from_str("hexagon"), ShapeKind::Circle);
        assert_eq!(shape_from_str("not_a_real_shape"), ShapeKind::Circle);
        assert_eq!(shape_from_str(""), ShapeKind::Circle);
    }

    fn default_shape_style() -> ShapeStyle {
        ShapeStyle {
            fill: Some(crate::render::color::from_rgb(0, 0, 0)),
            stroke: None,
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash_idx: None,
            angle: 0.0,
        }
    }

    #[test]
    fn test_emit_vline_single_vertical_line() {
        use ferrum_scene::SceneNode;
        let nodes = emit_shape_nodes(ShapeKind::VLine, 100.0, 100.0, 5.0, default_shape_style());
        assert_eq!(nodes.len(), 1, "VLine should emit exactly 1 node");
        match &nodes[0] {
            SceneNode::Line { x1, y1: _, x2, y2: _, .. } => {
                assert!((x1 - x2).abs() < 1e-10, "VLine x1 and x2 must be equal (vertical line): x1={x1}, x2={x2}");
            }
            other => panic!("Expected SceneNode::Line, got: {other:?}"),
        }
    }

    #[test]
    fn test_emit_hline_single_horizontal_line() {
        use ferrum_scene::SceneNode;
        let nodes = emit_shape_nodes(ShapeKind::HLine, 100.0, 100.0, 5.0, default_shape_style());
        assert_eq!(nodes.len(), 1, "HLine should emit exactly 1 node");
        match &nodes[0] {
            SceneNode::Line { x1: _, y1, x2: _, y2, .. } => {
                assert!((y1 - y2).abs() < 1e-10, "HLine y1 and y2 must be equal (horizontal line): y1={y1}, y2={y2}");
            }
            other => panic!("Expected SceneNode::Line, got: {other:?}"),
        }
    }

    #[test]
    fn stroke_dash_encoding_maps_index_to_dash_pattern() {
        let spec = spec_with_stroke_channels();
        let batch = batch_with_stroke_channels();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let circles: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style) } else { None }
        }).collect();

        // Index 0 → solid (None), index 1 → "6,3", index 2 → "2,3"
        assert!(circles[0].stroke_dash.is_none(), "index 0 should be solid (None)");
        assert_eq!(circles[1].stroke_dash.as_deref(), Some([6.0, 3.0].as_ref()),
            "index 1 should be dashed [6,3]");
        assert_eq!(circles[2].stroke_dash.as_deref(), Some([2.0, 3.0].as_ref()),
            "index 2 should be dotted [2,3]");
    }

    #[test]
    fn point_integer_ordinal_x_emits_circles() {
        // D9-B regression: ordinal x with Int64 column must emit circles, not
        // skip every row because xs_str is None. Previously col_as_str returned
        // Err for Int64 → xs_str = None → every ordinal arm continued.
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        use crate::layout::{PanelLayout, Rect, ThemeInputs};
        use crate::render::draw::resolve_mark_style;
        use crate::render::scale_resolve::resolve_scales;
        use crate::spec::chart::ChartSpec;
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{DataType as EncDataType, Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "year".into(),
                    type_: Some(EncDataType::Ordinal),
                    ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y".into(),
                    type_: Some(EncDataType::Quantitative),
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
        let schema = Arc::new(Schema::new(vec![
            Field::new("year", DataType::Int64, false),
            Field::new("y",    DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![2000i64, 2001, 2002])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let circle_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Circle { .. }))
            .count();
        assert_eq!(circle_count, 3,
            "Int64 ordinal x must emit one circle per row; got {circle_count}");
    }
}
