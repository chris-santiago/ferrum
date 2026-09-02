//! mark_point: render each row as a shape glyph at (scale_x(row.x), scale_y(row.y)).
//! Phase 7: always emits <circle> using ctx.mark_style.point.point_size.
//! Phase 8a: honors per-row size/shape/opacity from ctx.scales when populated.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_positional_category_str, col_as_str, resolve_fill_color, x_field, y_field, DrawCtx, MetadataColumns};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::{resolve_row_stroke_dash, stroke_dash_column_loader};
use crate::render::marks::opacity::{resolve_scaled_opacity, OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::{ScaleKind, ShapeKind};

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
    /// Provenance for `fill` (T8 quality-review c2): `true` only when `fill`
    /// resolved from a user `"none"`/`"transparent"` clear, never merely
    /// because it equals `color::TRANSPARENT` by value. Gates whether the
    /// Circle/Square/Diamond/TriangleUp/TriangleDown `FillStroke` variants
    /// omit the attribute; unused by the StrokeStyle-shaped Cross/VLine/HLine
    /// variants (out of scope — `StrokeStyle.color` has no absent
    /// representation).
    pub fill_cleared: bool,
    pub stroke: Option<crate::render::color::Color>,
    /// The `stroke` analog of [`fill_cleared`](ShapeStyle::fill_cleared).
    pub stroke_cleared: bool,
    pub stroke_width: f64,
    pub opacity: f64,
    pub stroke_opacity: f64,
    pub fill_opacity: f64,
    /// The resolved dasharray pattern, already looked up from either the
    /// numeric `DASH_PALETTE` index contract or a categorical
    /// `StrokeDashScale` (T12) — callers pass the fully-resolved value
    /// (`channels::resolve_row_stroke_dash`'s output) rather than a raw
    /// index, since a categorical field has no numeric index to pass here.
    pub stroke_dash: Option<Vec<f64>>,
    pub angle: f64,
}

pub(crate) fn emit_shape_nodes(
    kind: ShapeKind,
    cx: f64,
    cy: f64,
    r: f64,
    style: ShapeStyle,
) -> Vec<ferrum_scene::SceneNode> {
    let ShapeStyle {
        fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, stroke_opacity,
        fill_opacity, stroke_dash, angle,
    } = style;
    use crate::render::draw::{to_scene_fill_stroke_full, to_scene_stroke};
    use ferrum_scene::{PathCmd, SceneNode};

    let dash_ref: Option<&[f64]> = stroke_dash.as_deref();

    match kind {
        ShapeKind::Circle => {
            let style = to_scene_fill_stroke_full(
                fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, dash_ref,
                fill_opacity, stroke_opacity, angle,
            );
            vec![SceneNode::Circle { cx, cy, r, style }]
        }
        ShapeKind::Square => {
            let s = r * 1.6;
            let style = to_scene_fill_stroke_full(
                fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, dash_ref,
                fill_opacity, stroke_opacity, angle,
            );
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
            let mut s1 = to_scene_stroke(stroke_color, sw, opacity, None, None, None);
            s1.stroke_opacity = stroke_opacity;
            let mut s2 = to_scene_stroke(stroke_color, sw, opacity, None, None, None);
            s2.stroke_opacity = stroke_opacity;
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
            let style = to_scene_fill_stroke_full(
                fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, dash_ref,
                fill_opacity, stroke_opacity, angle,
            );
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
            let style = to_scene_fill_stroke_full(
                fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, dash_ref,
                fill_opacity, stroke_opacity, angle,
            );
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
            let style = to_scene_fill_stroke_full(
                fill, fill_cleared, stroke, stroke_cleared, stroke_width, opacity, dash_ref,
                fill_opacity, stroke_opacity, angle,
            );
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
            let mut s = to_scene_stroke(stroke_color, sw, opacity, None, None, None);
            s.stroke_opacity = stroke_opacity;
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
            let mut s = to_scene_stroke(stroke_color, sw, opacity, None, None, None);
            s.stroke_opacity = stroke_opacity;
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
        None => return MarkBuildResult::empty(MarkBatchKind::Point),
    };
    let yf = match y_field(ctx, spec) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Point),
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
        return MarkBuildResult::empty(MarkBatchKind::Point);
    }

    // Color encoding (shared loader, C9 — byte-identical to the prior inline
    // (scale_kind, field) → (categorical, numeric) split).
    let (color_values_str, color_values_f64) =
        crate::render::marks::channels::color_column_loader(ctx);

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

    // fill_opacity / stroke_opacity via the shared resolver (FA-11), sampled
    // per-row. Defaults: fill_opacity → 1.0, stroke_opacity → 1.0. The opacity
    // channel is mapped through `ctx.scales.opacity` at the call site below, so
    // the resolver's opacity output is unused here (its default is irrelevant).
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));

    // Per-row stroke/angle channel values (direct passthrough — no scale transform).
    let stroke_width_values: Option<Vec<Option<f64>>> = spec.encoding.stroke_width
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // T12: categorical resolves through ctx.scales.stroke_dash (StrokeDashScale);
    // numeric keeps the DASH_PALETTE index contract byte-identically.
    let dash_cols = stroke_dash_column_loader(ctx);

    let angle_values: Option<Vec<Option<f64>>> = spec.encoding.angle
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    let default_radius = (ctx.mark_style.point.point_size / std::f64::consts::PI).sqrt();

    // Per-row pixel offsets from position adjustment.
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Metadata columns are read here; metadata vectors are built AFTER the loop
    // via build_metadata_for_indices so they are in node order, not row order.
    // This is required because Cross emits 2 nodes per row, so calling
    // build_metadata(ctx) (which returns n_rows entries) would misalign node j
    // with the wrong source row's tooltip (archaeology bug #6).
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep. For single-node shapes
    // (Circle, Square, etc.) push_many is equivalent to push. For Cross, both line
    // nodes are mapped to the SAME source row i, so data_indices carries 2 copies
    // of i — one per emitted node — keeping the alignment invariant.
    let mut acc = MarkNodes::with_capacity(n);

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

        // Resolve color via the shared per-row fill resolver (RMARK-03).
        // `fill_cleared` is provenance (T8 quality-review c2): true only when
        // the constant `mark_style.paint.fill` was cleared by a literal
        // "none"/"transparent", never merely because the resolved color is
        // zero-alpha by value; a color-scale hit is never cleared.
        let (fill_base, fill_cleared) = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_values_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_values_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill,
            ctx.mark_style.paint.fill_cleared,
        );

        // Resolve per-row opacity (through scale if present).
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);

        let fill = with_opacity(fill_base, row_opacity);

        // filled=false → hollow points. The stroke is painted in the fill's
        // color here, so it carries the fill's cleared-ness too.
        let (effective_fill, effective_fill_cleared, effective_stroke, effective_stroke_cleared, effective_sw) =
            if ctx.mark_style.point.filled == Some(false) {
                let sw = if ctx.mark_style.paint.stroke_width > 0.0 {
                    ctx.mark_style.paint.stroke_width
                } else {
                    1.5
                };
                (None, false, Some(fill_base), fill_cleared, sw)
            } else {
                (
                    Some(fill), fill_cleared, ctx.mark_style.paint.stroke,
                    ctx.mark_style.paint.stroke_cleared, ctx.mark_style.paint.stroke_width,
                )
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
        } else if let Some(ref shape_name) = ctx.mark_style.point.shape {
            shape_from_str(shape_name)
        } else {
            ShapeKind::Circle
        };

        // Resolve per-row fill_opacity / stroke_opacity via the shared resolver.
        let (_, row_fill_opacity, row_stroke_opacity) = opacity_res.at_row(i);

        // Resolve per-row stroke/angle channel values (direct passthrough).
        let row_stroke_width = stroke_width_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| *v >= 0.0 && v.is_finite())
            .unwrap_or(effective_sw);

        let row_stroke_dash = resolve_row_stroke_dash(
            &dash_cols,
            ctx.scales.stroke_dash.as_ref(),
            i,
            ctx.mark_style.paint.stroke_dash.as_deref(),
        );

        let row_angle = angle_values
            .as_ref()
            .and_then(|v| v[i])
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);

        // When stroke_width encoding produces a positive value but no explicit
        // stroke color exists, use the fill color as the stroke so the width is
        // visible in SVG (stroke-width is only emitted when stroke is Some).
        // The rescue paints the stroke in the fill color, so cleared-ness
        // follows the fill's, not the (absent) stroke's.
        let (effective_stroke_for_row, effective_stroke_for_row_cleared) =
            if row_stroke_width > 0.0 && effective_stroke.is_none() && stroke_width_values.is_some() {
                (effective_fill, effective_fill_cleared)
            } else {
                (effective_stroke, effective_stroke_cleared)
            };

        let shape_nodes = emit_shape_nodes(
            shape_kind, cx, cy, radius,
            ShapeStyle {
                fill: effective_fill,
                fill_cleared: effective_fill_cleared,
                stroke: effective_stroke_for_row,
                stroke_cleared: effective_stroke_for_row_cleared,
                stroke_width: row_stroke_width,
                opacity: row_opacity,
                stroke_opacity: row_stroke_opacity,
                fill_opacity: row_fill_opacity,
                stroke_dash: row_stroke_dash,
                angle: row_angle,
            },
        );
        // push_many maps EVERY emitted node to source row i. For single-node shapes
        // this is a single push; for Cross it pushes 2 nodes both tagged with row i.
        // This keeps data_indices[k] == source-row-of-nodes[k] for all k.
        acc.push_many(shape_nodes, i);
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Point,
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let circles: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style) } else { None }
        }).collect();
        assert_eq!(circles.len(), 3, "expected 3 circle nodes");

        // stroke_opacity values from batch: [0.3, 0.6, 0.9]. Batch A §4.3
        // (sanctioned): the channel resolves a scale mapping that extent onto
        // the theme opacity band [0.1, 1.0] rather than passing the raw value
        // through, so the endpoints become the band endpoints and the middle row
        // lands halfway. Per-row distinctness is what this guard pins.
        let band = |v: f64| 0.1 + 0.9 * (v - 0.3) / (0.9 - 0.3);
        for (i, circle) in circles.iter().enumerate() {
            let expected = band([0.3, 0.6, 0.9][i]);
            assert!((circle.stroke_opacity - expected).abs() < 1e-5,
                "row {i} stroke_opacity: expected {expected}, got {}", circle.stroke_opacity);
        }
    }

    #[test]
    fn angle_encoding_flows_to_fill_stroke() {
        let spec = spec_with_stroke_channels();
        let batch = batch_with_stroke_channels();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
            fill_cleared: false,
            stroke: None,
            stroke_cleared: false,
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
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

    // ── Ported from bug_hunt_marks_rendering(_r2).rs (R1) ──────────────────
    // Path/rect-shape branches of `emit_shape_nodes` that VLine/HLine/Cross's
    // existing coverage doesn't reach: Diamond, TriangleUp, TriangleDown,
    // Square, and Cross at r=0.

    #[test]
    fn emit_diamond_at_origin_produces_closed_4_point_path() {
        use ferrum_scene::{PathCmd, SceneNode};
        let nodes = emit_shape_nodes(ShapeKind::Diamond, 0.0, 0.0, 5.0, default_shape_style());
        assert_eq!(nodes.len(), 1);
        match &nodes[0] {
            SceneNode::Path { commands, closed, .. } => {
                assert!(*closed);
                assert_eq!(commands.len(), 5, "Diamond: MoveTo + 3 LineTo + Close");
                assert!(matches!(commands[4], PathCmd::Close));
                for cmd in commands {
                    if let PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } = cmd {
                        assert!(x.is_finite() && y.is_finite());
                    }
                }
                // Top vertex (index 0) must be above the center at (0, -d).
                if let PathCmd::MoveTo { x, y } = commands[0] {
                    assert!((x - 0.0).abs() < 1e-9);
                    assert!(y < 0.0, "diamond top vertex must be above center, got y={y}");
                }
            }
            other => panic!("Expected SceneNode::Path, got: {other:?}"),
        }
    }

    #[test]
    fn emit_triangle_up_apex_points_above_center() {
        use ferrum_scene::{PathCmd, SceneNode};
        let nodes = emit_shape_nodes(ShapeKind::TriangleUp, 0.0, 0.0, 5.0, default_shape_style());
        match &nodes[0] {
            SceneNode::Path { commands, .. } => {
                assert_eq!(commands.len(), 4, "TriangleUp: MoveTo + 2 LineTo + Close");
                if let PathCmd::MoveTo { x, y } = commands[0] {
                    assert!((x - 0.0).abs() < 1e-9);
                    assert!(y < 0.0, "triangle-up apex must be above center, got y={y}");
                } else {
                    panic!("expected MoveTo apex");
                }
            }
            other => panic!("Expected SceneNode::Path, got: {other:?}"),
        }
    }

    #[test]
    fn emit_triangle_down_apex_points_below_center() {
        use ferrum_scene::{PathCmd, SceneNode};
        let nodes = emit_shape_nodes(ShapeKind::TriangleDown, 0.0, 0.0, 5.0, default_shape_style());
        match &nodes[0] {
            SceneNode::Path { commands, .. } => {
                assert_eq!(commands.len(), 4, "TriangleDown: MoveTo + 2 LineTo + Close");
                if let PathCmd::MoveTo { x, y } = commands[0] {
                    assert!((x - 0.0).abs() < 1e-9);
                    assert!(y > 0.0, "triangle-down apex must be below center, got y={y}");
                } else {
                    panic!("expected MoveTo apex");
                }
            }
            other => panic!("Expected SceneNode::Path, got: {other:?}"),
        }
    }

    #[test]
    fn emit_square_at_origin_has_side_1_6x_radius_centered() {
        use ferrum_scene::SceneNode;
        let r = 5.0_f64;
        let nodes = emit_shape_nodes(ShapeKind::Square, 0.0, 0.0, r, default_shape_style());
        match &nodes[0] {
            SceneNode::Rect { x, y, w, h, .. } => {
                let s = r * 1.6;
                assert!((w - s).abs() < 1e-9, "square side must be r*1.6, got w={w}");
                assert!((h - s).abs() < 1e-9);
                assert!((x - (-s / 2.0)).abs() < 1e-9, "square must be centered on cx");
                assert!((y - (-s / 2.0)).abs() < 1e-9, "square must be centered on cy");
            }
            other => panic!("Expected SceneNode::Rect, got: {other:?}"),
        }
    }

    #[test]
    fn emit_cross_zero_radius_collapses_both_arms_to_center() {
        use ferrum_scene::SceneNode;
        let nodes = emit_shape_nodes(ShapeKind::Cross, 50.0, 50.0, 0.0, default_shape_style());
        assert_eq!(nodes.len(), 2, "Cross always emits 2 Line nodes, even degenerate");
        for node in &nodes {
            match node {
                SceneNode::Line { x1, y1, x2, y2, .. } => {
                    assert!((x1 - x2).abs() < 1e-12, "zero-radius arm must collapse to a point: x1={x1}, x2={x2}");
                    assert!((y1 - y2).abs() < 1e-12, "zero-radius arm must collapse to a point: y1={y1}, y2={y2}");
                    assert!((x1 - 50.0).abs() < 1e-12 && (y1 - 50.0).abs() < 1e-12,
                        "collapsed arm must sit exactly at center");
                }
                other => panic!("Expected SceneNode::Line, got: {other:?}"),
            }
        }
    }

    /// With no size encoding, `build`'s default radius is derived from
    /// `mark_style.point.point_size` as `sqrt(point_size / PI)` (area-to-radius
    /// conversion, so doubling point_size doesn't double the visible radius).
    /// `resolve_mark_style_overrides_point_size` (draw.rs) covers the override
    /// wiring; this covers the formula actually applied to the emitted circles.
    #[test]
    fn default_radius_from_constant_point_size_matches_area_formula() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = crate::spec::mark_style::MarkKwargsSpec { size: Some(50.0), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let expected_r = (50.0_f64 / std::f64::consts::PI).sqrt();
        let radii: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { r, .. } = n { Some(*r) } else { None }
        }).collect();
        assert_eq!(radii.len(), 3);
        for r in radii {
            assert!((r - expected_r).abs() < 1e-9, "expected radius {expected_r}, got {r}");
        }
    }

    /// point_size = 0 is a valid (if invisible) configuration: the formula
    /// must yield radius 0, not NaN or a panic.
    #[test]
    fn default_radius_zero_point_size_yields_zero_radius() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = crate::spec::mark_style::MarkKwargsSpec { size: Some(0.0), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let radii: Vec<f64> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { r, .. } = n { Some(*r) } else { None }
        }).collect();
        assert_eq!(radii.len(), 3);
        for r in radii {
            assert_eq!(r, 0.0, "zero point_size must yield exactly zero radius");
        }
    }

    /// RMARK-01 regression: line-based shapes (Cross, VLine, HLine) must honor
    /// the opacity channel. Before the fix, to_scene_stroke was called with a
    /// hardcoded 1.0, so mark_point(shape="cross", opacity=0.3) rendered fully
    /// opaque while shape="circle" honored 0.3. This test verifies the fix by
    /// calling emit_shape_nodes directly with a non-1.0 opacity and asserting
    /// that the emitted StrokeStyle carries the expected opacity — NOT 1.0.
    #[test]
    fn line_shapes_honor_opacity_channel_rmark01() {
        use ferrum_scene::SceneNode;

        let opacity_val = 0.3_f64;
        let stroke_opacity_val = 0.5_f64;

        let style = ShapeStyle {
            fill: Some(crate::render::color::from_rgb(100, 150, 200)),
            fill_cleared: false,
            stroke: None,
            stroke_cleared: false,
            stroke_width: 1.0,
            opacity: opacity_val,
            stroke_opacity: stroke_opacity_val,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        };

        // Cross emits 2 Line nodes — both must carry the row opacity.
        let cross_nodes = emit_shape_nodes(ShapeKind::Cross, 50.0, 50.0, 5.0, ShapeStyle {
            fill: style.fill,
            fill_cleared: style.fill_cleared,
            stroke: style.stroke,
            stroke_cleared: style.stroke_cleared,
            stroke_width: style.stroke_width,
            opacity: opacity_val,
            stroke_opacity: stroke_opacity_val,
            fill_opacity: style.fill_opacity,
            stroke_dash: style.stroke_dash.clone(),
            angle: style.angle,
        });
        assert_eq!(cross_nodes.len(), 2, "Cross must emit 2 Line nodes");
        for (i, node) in cross_nodes.iter().enumerate() {
            match node {
                SceneNode::Line { style: s, .. } => {
                    assert!(
                        (s.opacity - opacity_val).abs() < 1e-9,
                        "Cross line[{i}] opacity: expected {opacity_val}, got {}. \
                         RMARK-01 regression — line shapes must honor the opacity channel.",
                        s.opacity
                    );
                    assert!(
                        (s.stroke_opacity - stroke_opacity_val).abs() < 1e-9,
                        "Cross line[{i}] stroke_opacity: expected {stroke_opacity_val}, got {}.",
                        s.stroke_opacity
                    );
                }
                other => panic!("Expected SceneNode::Line for Cross, got: {other:?}"),
            }
        }

        // VLine emits 1 Line node.
        let vline_nodes = emit_shape_nodes(ShapeKind::VLine, 50.0, 50.0, 5.0, ShapeStyle {
            fill: style.fill,
            fill_cleared: style.fill_cleared,
            stroke: style.stroke,
            stroke_cleared: style.stroke_cleared,
            stroke_width: style.stroke_width,
            opacity: opacity_val,
            stroke_opacity: stroke_opacity_val,
            fill_opacity: style.fill_opacity,
            stroke_dash: style.stroke_dash.clone(),
            angle: style.angle,
        });
        assert_eq!(vline_nodes.len(), 1, "VLine must emit 1 Line node");
        match &vline_nodes[0] {
            SceneNode::Line { style: s, .. } => {
                assert!(
                    (s.opacity - opacity_val).abs() < 1e-9,
                    "VLine opacity: expected {opacity_val}, got {}. \
                     RMARK-01 regression — line shapes must honor the opacity channel.",
                    s.opacity
                );
                assert!(
                    (s.stroke_opacity - stroke_opacity_val).abs() < 1e-9,
                    "VLine stroke_opacity: expected {stroke_opacity_val}, got {}.",
                    s.stroke_opacity
                );
            }
            other => panic!("Expected SceneNode::Line for VLine, got: {other:?}"),
        }

        // HLine emits 1 Line node.
        let hline_nodes = emit_shape_nodes(ShapeKind::HLine, 50.0, 50.0, 5.0, ShapeStyle {
            fill: style.fill,
            fill_cleared: style.fill_cleared,
            stroke: style.stroke,
            stroke_cleared: style.stroke_cleared,
            stroke_width: style.stroke_width,
            opacity: opacity_val,
            stroke_opacity: stroke_opacity_val,
            fill_opacity: style.fill_opacity,
            stroke_dash: style.stroke_dash.clone(),
            angle: style.angle,
        });
        assert_eq!(hline_nodes.len(), 1, "HLine must emit 1 Line node");
        match &hline_nodes[0] {
            SceneNode::Line { style: s, .. } => {
                assert!(
                    (s.opacity - opacity_val).abs() < 1e-9,
                    "HLine opacity: expected {opacity_val}, got {}. \
                     RMARK-01 regression — line shapes must honor the opacity channel.",
                    s.opacity
                );
                assert!(
                    (s.stroke_opacity - stroke_opacity_val).abs() < 1e-9,
                    "HLine stroke_opacity: expected {stroke_opacity_val}, got {}.",
                    s.stroke_opacity
                );
            }
            other => panic!("Expected SceneNode::Line for HLine, got: {other:?}"),
        }
    }

    #[test]
    fn stroke_dash_encoding_maps_index_to_dash_pattern() {
        let spec = spec_with_stroke_channels();
        let batch = batch_with_stroke_channels();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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

    /// T12: a categorical `stroke_dash` field resolves through
    /// `ctx.scales.stroke_dash` (`StrokeDashScale::dash_for`) instead of the
    /// numeric palette-index contract — each row's dash matches its category,
    /// not its row position.
    #[test]
    fn stroke_dash_categorical_encoding_resolves_through_scale() {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sd", DataType::Utf8, false),
        ]));
        // First-appearance domain order: solid, dashed, dotted.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(StringArray::from(vec!["solid", "dashed", "dotted"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        assert!(scales.stroke_dash.is_some(), "a categorical stroke_dash field must resolve a StrokeDashScale");
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let circles: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style) } else { None }
        }).collect();
        assert_eq!(circles.len(), 3);
        assert!(circles[0].stroke_dash.is_none(), "'solid' (domain index 0) must be the solid slot");
        assert_eq!(circles[1].stroke_dash.as_deref(), Some([6.0, 3.0].as_ref()),
            "'dashed' (domain index 1) must be the long-dash pattern");
        assert_eq!(circles[2].stroke_dash.as_deref(), Some([2.0, 3.0].as_ref()),
            "'dotted' (domain index 2) must be the short-dash pattern");
    }

    /// T12: a literal `mark_point(stroke_dash=[...])` with NO stroke_dash
    /// encoding now takes effect. Pre-T12 this fell back to nothing —
    /// `MarkStyle::stroke_dash` was never read by point.rs — silently
    /// dropping the literal (a member of the silent-drop class this batch
    /// remediates); other marks (line/rule/bar/rect) already honored it.
    #[test]
    fn stroke_dash_literal_with_no_encoding_now_applies() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let overrides = crate::spec::mark_style::MarkKwargsSpec {
            stroke_dash: Some(vec![6.0, 3.0]),
            ..Default::default()
        };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        let circles: Vec<_> = result.nodes.iter().filter_map(|n| {
            if let SceneNode::Circle { style, .. } = n { Some(style) } else { None }
        }).collect();
        assert_eq!(circles.len(), 3);
        for (i, circle) in circles.iter().enumerate() {
            assert_eq!(circle.stroke_dash.as_deref(), Some([6.0, 3.0].as_ref()),
                "row {i}: literal stroke_dash must apply when no encoding is bound");
        }
    }

    /// T12 fix round (spec §4.3, amended 2026-09-01, Issue 2 pin): the spec
    /// reviewer's missing pin — `mark_point(filled=False, stroke_dash=[6,3])`
    /// with no `stroke_dash` encoding — through the full `render_svg`
    /// pipeline, not just the internal `ShapeStyle` field the test above
    /// pins. `filled=False` is the realistic usage this ruling names: a
    /// hollow marker's visible outline IS its stroke, so the dash pattern is
    /// only meaningful there. Proves the literal reaches the actual
    /// `stroke-dasharray` SVG attribute end to end.
    #[test]
    fn stroke_dash_literal_filled_false_no_encoding_emits_svg_dasharray() {
        let mut spec = three_row_spec();
        spec.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec {
            filled: Some(false),
            stroke_dash: Some(vec![6.0, 3.0]),
            ..Default::default()
        });
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
        let config = crate::render::config::RenderConfig::default();
        let result = crate::render::render_svg(
            &spec, &batch, &theme, viewport, &config, &crate::render::chart_config::ChartConfig::default(),
        ).unwrap();
        let svg = &result.bytes;

        assert_eq!(svg.matches("<circle ").count(), 3, "3 rows must draw 3 hollow circles: {svg}");
        assert_eq!(svg.matches("stroke-dasharray=\"6,3\"").count(), 3,
            "every hollow point must carry the literal stroke_dash as a real SVG dasharray attribute: {svg}");
    }

    /// T12 fix round (spec §4.3, amended 2026-09-01, Issue 2 pin): the
    /// reviewer's other missing pin — a NUMERIC `stroke_dash` encoding whose
    /// per-row index is null falls back to the mark's literal `stroke_dash`,
    /// per `resolve_row_stroke_dash`'s numeric branch
    /// (`channels.rs`: `.or_else(|| base.map(<[f64]>::to_vec))`). Pre-T12,
    /// point.rs read no literal at all, so a null numeric index row drew
    /// solid; this fix round's ruling is to KEEP this fallback (the reviewer
    /// judged it consistent with line/bar's own `.or(base_dash)` idiom) and
    /// pin it. Reproduces the reviewer's live repro exactly:
    /// `mark_point(filled=False, stroke_dash=[9,1])` +
    /// `StrokeDash('sd', type_='quantitative')` with `sd=[1.0, None, 2.0]`
    /// through the full `render_svg` pipeline, asserting the actual SVG
    /// dasharrays in row order: `["6,3", "9,1", "2,3"]` — row 0 resolves
    /// `DASH_PALETTE[0]` via its own index, row 1 (null index) inherits the
    /// literal `[9,1]`, row 2 resolves `DASH_PALETTE[1]`.
    #[test]
    fn stroke_dash_numeric_null_index_row_inherits_literal_fallback() {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("sd", DataType::Float64, true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(2.0)])),
        ]).unwrap();
        let mut spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None, coord: None, mark_style: None,
            position: None, title: None, axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        spec.mark_style = Some(crate::spec::mark_style::MarkKwargsSpec {
            filled: Some(false),
            stroke_dash: Some(vec![9.0, 1.0]),
            ..Default::default()
        });
        let theme = ThemeInputs::default();
        let viewport = crate::layout::Viewport { width: 600.0, height: 400.0 };
        let config = crate::render::config::RenderConfig::default();
        let result = crate::render::render_svg(
            &spec, &batch, &theme, viewport, &config, &crate::render::chart_config::ChartConfig::default(),
        ).unwrap();
        let svg = &result.bytes;

        let circle_tags: Vec<&str> = svg.split("<circle ").skip(1).collect();
        assert_eq!(circle_tags.len(), 3, "expected 3 circle elements: {svg}");
        let extract_dash = |tag: &str| -> Option<String> {
            let attrs = &tag[..tag.find('>').unwrap_or(tag.len())];
            attrs.find("stroke-dasharray=\"").map(|start| {
                let rest = &attrs[start + "stroke-dasharray=\"".len()..];
                rest[..rest.find('"').unwrap()].to_string()
            })
        };
        let dashes: Vec<Option<String>> = circle_tags.iter().map(|t| extract_dash(t)).collect();
        assert_eq!(
            dashes,
            vec![Some("6,3".to_string()), Some("9,1".to_string()), Some("2,3".to_string())],
            "row order must match the reviewer's live repro shape [\"6,3\",\"9,1\",\"2,3\"]: {svg}"
        );
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
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);
        let circle_count = result.nodes.iter()
            .filter(|n| matches!(n, SceneNode::Circle { .. }))
            .count();
        assert_eq!(circle_count, 3,
            "Int64 ordinal x must emit one circle per row; got {circle_count}");
    }

    // ── Task 4: metadata alignment for Cross (multi-node) and skipped rows ────

    /// Build a batch suitable for Cross-shape alignment tests.
    /// Columns: x, y (both quantitative, no skips), tooltip "lbl".
    fn cross_batch(n: usize) -> arrow::record_batch::RecordBatch {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let lbls: Vec<String> = (0..n).map(|i| format!("row{i}")).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("lbl", DataType::Utf8, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(lbls)),
        ]).unwrap()
    }

    fn cross_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "lbl".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                shape: Some("cross".into()),
                ..Default::default()
            }),
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        }
    }

    fn make_ctx<'a>(
        spec: &'a ChartSpec,
        batch: &'a arrow::record_batch::RecordBatch,
        panel: &'a PanelLayout,
        theme: &'a ThemeInputs,
        scales: &'a crate::render::scale_resolve::ResolvedScales,
        mark_style: &'a crate::render::draw::MarkStyle,
    ) -> DrawCtx<'a> {
        DrawCtx { spec, panel, theme, scales, batch, mark_style }
    }

    /// HEADLINE TEST (spec §9): Cross with ZERO skipped rows.
    ///
    /// Cross emits 2 Line nodes per row. With 3 rows and no skips the builder
    /// must produce 6 nodes (3 crosses × 2 lines) and 6 tooltip entries, each
    /// pointing at the correct source row.
    ///
    /// Pre-migration code pushed 1 index per row (indices=[0,1,2]) for 6 nodes,
    /// so node 1 (second line of row 0's cross) received row 1's tooltip instead
    /// of row 0's. This test proves the fix.
    #[test]
    fn cross_zero_skip_tooltips_align_to_source_rows() {
        let n = 3;
        let batch = cross_batch(n);
        let spec = cross_spec();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        // Pass the spec's mark_style so the constant "cross" shape is resolved.
        let mark_style = resolve_mark_style(spec.mark_style.as_ref(), &theme, &Mark::Point).unwrap();
        let ctx = make_ctx(&spec, &batch, &panel, &theme, &scales, &mark_style);
        let result = super::build(&ctx);

        // Each row contributes 2 Line nodes (cross = horizontal + vertical arm).
        assert_eq!(result.nodes.len(), 2 * n,
            "cross: expected {} nodes (2 per row), got {}", 2 * n, result.nodes.len());
        assert!(result.nodes.iter().all(|n| matches!(n, SceneNode::Line { .. })),
            "all cross nodes must be Line variants");

        let tooltips = result.tooltips.as_ref().expect("tooltip encoding must produce tooltips");
        assert_eq!(tooltips.len(), 2 * n,
            "tooltip count must equal node count (2*n_rows); got {}", tooltips.len());

        // Node 0 and node 1 are BOTH the first row's cross arms → tooltip = "row0".
        // Node 2 and node 3 are the second row's cross arms → tooltip = "row1".
        // Node 4 and node 5 are the third row's cross arms → tooltip = "row2".
        // Pre-migration: node 1 would incorrectly carry "row1" (off-by-one).
        for row in 0..n {
            for arm in 0..2 {
                let node_idx = row * 2 + arm;
                let tip = &tooltips[node_idx];
                let val = tip.fields.first().map(|f| f.value.as_str()).unwrap_or("");
                assert_eq!(
                    val, format!("row{row}"),
                    "node {node_idx} (row {row}, arm {arm}) expected tooltip 'row{row}', got '{val}'"
                );
            }
        }

        // data_indices: [..., row, row, ...] — each row appears twice consecutively.
        let indices = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(indices.len(), 2 * n);
        for row in 0..n {
            assert_eq!(indices[row * 2],     row, "data_indices[{}] must be {row}", row * 2);
            assert_eq!(indices[row * 2 + 1], row, "data_indices[{}] must be {row}", row * 2 + 1);
        }
    }

    /// Skipped-row alignment test: single-node shape (Circle) with a null row.
    ///
    /// Row 1 has x=NaN → is skipped by the renderer. The kept nodes are for rows
    /// 0 and 2. Their tooltips must be "row0" and "row2", not "row0" and "row1".
    #[test]
    fn circle_with_null_row_tooltips_align_to_kept_source_rows() {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("lbl", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, f64::NAN, 2.0])), // row 1 skipped (NaN x)
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(StringArray::from(vec!["row0", "row1", "row2"])),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "lbl".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None, layers: None, coord: None,
            mark_style: None, // default shape = Circle
            position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = make_ctx(&spec, &batch, &panel, &theme, &scales, &mark_style);
        let result = super::build(&ctx);

        // 2 kept rows → 2 circle nodes.
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Circle { .. })).count(), 2,
            "expected 2 circles (row 1 skipped)");

        let tooltips = result.tooltips.as_ref().expect("tooltips must be present");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let val0 = tooltips[0].fields.first().map(|f| f.value.as_str()).unwrap_or("");
        let val1 = tooltips[1].fields.first().map(|f| f.value.as_str()).unwrap_or("");
        assert_eq!(val0, "row0", "node 0 tooltip must come from source row 0; got '{val0}'");
        assert_eq!(val1, "row2", "node 1 tooltip must come from source row 2 (row 1 skipped); got '{val1}'");
    }

    /// Href-channel alignment test: point chart with href encoding and a skipped
    /// row. The kept nodes must carry the href of their true source rows.
    #[test]
    fn circle_href_aligns_to_kept_source_rows() {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        // Row 1: y=NaN → skipped. Rows 0 and 2 kept.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, f64::NAN, 2.0])), // row 1 skipped (NaN y)
            Arc::new(StringArray::from(vec![
                "https://example.com/0",
                "https://example.com/1",
                "https://example.com/2",
            ])),
        ]).unwrap();

        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                href: Some(EncodingSpec { field: "url".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None, layers: None, coord: None,
            mark_style: None,
            position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };

        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = make_ctx(&spec, &batch, &panel, &theme, &scales, &mark_style);
        let result = super::build(&ctx);

        // 2 kept nodes.
        assert_eq!(result.nodes.len(), 2, "expected 2 nodes (row 1 skipped)");

        let hrefs = result.hrefs.as_ref().expect("href encoding must produce hrefs");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");

        assert_eq!(hrefs[0].as_deref(), Some("https://example.com/0"),
            "node 0 href must come from source row 0");
        assert_eq!(hrefs[1].as_deref(), Some("https://example.com/2"),
            "node 1 href must come from source row 2 (row 1 skipped)");
    }

    /// Backward-compat test: Circle points with no skipped rows and no metadata
    /// must produce the same geometry as before (byte-stable node count and
    /// positions). This guards that the accumulator path does not alter geometry
    /// for the common single-node case.
    #[test]
    fn circle_no_skip_no_metadata_backward_compat() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // 3 rows → 3 circles (single-node path, no skips).
        assert_eq!(result.nodes.len(), 3);
        assert!(result.nodes.iter().all(|n| matches!(n, SceneNode::Circle { .. })));

        // data_indices must be 1:1 with source rows.
        let indices = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(indices.as_slice(), &[0, 1, 2]);

        // No tooltip encoding → no tooltips.
        assert!(result.tooltips.is_none());
        assert!(result.hrefs.is_none());
        assert!(result.descriptions.is_none());
    }
}
