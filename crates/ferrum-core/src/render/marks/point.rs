//! mark_point: render each row as a shape glyph at (scale_x(row.x), scale_y(row.y)).
//! Phase 7: always emits <circle> using ctx.mark_style.point.point_size.
//! Phase 8a: honors per-row size/shape/opacity from ctx.scales when populated.
//!
//! # The `shape` channel reads through the ordinal-category reader (#166)
//!
//! `shape_values` reads `col_as_ordinal_category_str`, not `col_as_str`
//! (matching `color`'s NF-A3 sweep and `point`'s own positional reads
//! above). `col_as_str` errors on every non-`Utf8` dtype and the `.ok()`
//! swallowed it, so an `Int*`/`Float*`/`Bool` shape column left
//! `shape_values` `None` for every row and every point fell back to the
//! mark-style default (or plain `Circle`) while the legend — built from the
//! dtype-wide `distinct_values_in_order` — enumerated every category. This
//! is a **disclosed reader-switch, not a refusal**: unlike `color`, a shape
//! column this batch does not carry at all is not a distinct case `shape`
//! needs to tell apart from "no shape values" (there is no cross-layer
//! shape-inheritance carve-out to protect), so the read stays infallible —
//! an unreadable dtype (e.g. `Timestamp`) still falls back to the mark-style
//! default exactly as before. `Utf8` shape columns are byte-identical.

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_ordinal_category_str, col_as_positional_category_str, resolve_effective_stroke, resolve_fill_color, stroke_width_column, x_field, y_field, ClearablePaint, DrawCtx, MetadataColumns};
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
///
/// Delegates to [`shape_from_str_strict`], the single owner of the name
/// table, so the circle-fallback callers below and the refusing callers in
/// `scale_resolve::auxiliary::build_shape_scale` (#123) read one table
/// rather than two copies that could drift.
pub(crate) fn shape_from_str(s: &str) -> ShapeKind {
    shape_from_str_strict(s).unwrap_or(ShapeKind::Circle)
}

/// The strict counterpart of [`shape_from_str`]: `None` for any name not in
/// the point-shape table, instead of defaulting to `Circle`.
///
/// Used by `scale_resolve::auxiliary::build_shape_scale` to parse an
/// explicit `Shape(scale=OrdinalScale(range=[...]))` range (#123): there, an
/// unrecognized entry must refuse
/// ([`crate::render::RenderError::ShapeRangeEntryUnrecognized`]) rather
/// than silently draw every unmatched glyph as a circle, so that call site
/// needs to tell "recognized" apart from "defaulted".
pub(crate) fn shape_from_str_strict(s: &str) -> Option<ShapeKind> {
    Some(match s {
        "circle" => ShapeKind::Circle,
        "square" => ShapeKind::Square,
        "cross" => ShapeKind::Cross,
        "diamond" => ShapeKind::Diamond,
        "triangle-up" | "triangle_up" => ShapeKind::TriangleUp,
        "triangle-down" | "triangle_down" => ShapeKind::TriangleDown,
        "|" | "vline" => ShapeKind::VLine,
        "-" | "hline" => ShapeKind::HLine,
        _ => return None,
    })
}

// ── Scene-graph build path (11a) ───────────────────────────────────

/// Emit one shape glyph as `SceneNode` variants. Returns a `Vec` because
/// `ShapeKind::Cross` produces two `Line` nodes while all other shapes
/// produce exactly one node.
///
/// `stroke_opacity`, `fill_opacity`, and `angle` flow directly from encoding
/// columns into the emitted `FillStroke` style — no scale transform.
pub(crate) struct ShapeStyle {
    /// The row's fill paint, carrying its own cleared-ness (T8 quality-review
    /// c2): [`ClearablePaint::Cleared`] only when the constant fill resolved
    /// from a user `"none"`/`"transparent"` clear, never merely because it
    /// equals `color::TRANSPARENT` by value. The
    /// Circle/Square/Diamond/TriangleUp/TriangleDown arms omit the attribute
    /// for a clear, and since `#110` the StrokeStyle-shaped Cross/VLine/HLine
    /// arms do too: they paint their glyph lines in this slot, so a cleared fill
    /// paints no line rather than the zero-alpha one it emitted while
    /// `StrokeStyle.color` had no absent representation.
    pub fill: ClearablePaint,
    pub stroke: ClearablePaint,
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
        fill, stroke, stroke_width, opacity, stroke_opacity, fill_opacity, stroke_dash, angle,
    } = style;
    use crate::render::draw::{to_scene_fill_stroke, to_scene_stroke, PaintChannels};
    use ferrum_scene::{PathCmd, SceneNode};

    let dash_ref: Option<&[f64]> = stroke_dash.as_deref();

    // The five filled glyphs (circle/square/diamond/triangle-up/-down) build the
    // identical `FillStroke`; only their geometry differs. One value, built once,
    // instead of the same six-field literal five times.
    let filled = || {
        to_scene_fill_stroke(
            fill,
            stroke,
            PaintChannels {
                stroke_width,
                opacity,
                stroke_dash: dash_ref,
                fill_opacity,
                stroke_opacity,
                angle,
            },
        )
    };

    // The three `StrokeStyle`-shaped glyphs (cross/vline/hline) have no fill
    // channel: they paint their LINES in the fill slot, defaulting to black for a
    // mark that resolves no fill. A fill the user *cleared* stays cleared, so the
    // glyph paints nothing at all — before `#110` it took the clear's transparent
    // value and serialized `stroke="rgba(0,0,0,0.000)"`.
    let glyph_stroke = |line_width: f64| {
        let mut s = to_scene_stroke(
            fill.or_paint(crate::render::color::from_rgb(0, 0, 0)),
            line_width,
            opacity,
            None,
            None,
            None,
        );
        s.stroke_opacity = stroke_opacity;
        s
    };

    match kind {
        ShapeKind::Circle => {
            vec![SceneNode::Circle { cx, cy, r, style: filled() }]
        }
        ShapeKind::Square => {
            let s = r * 1.6;
            let style = filled();
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
            let arm = r * 0.5;
            let sw = r * 0.4;
            let (s1, s2) = (glyph_stroke(sw), glyph_stroke(sw));
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
            let style = filled();
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
                widths: None,
            }]
        }
        ShapeKind::TriangleUp => {
            let h = r * 1.4;
            let style = filled();
            vec![SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: cx, y: cy - h },
                    PathCmd::LineTo { x: cx + h * 0.866, y: cy + h * 0.5 },
                    PathCmd::LineTo { x: cx - h * 0.866, y: cy + h * 0.5 },
                    PathCmd::Close,
                ],
                style,
                closed: true,
                widths: None,
            }]
        }
        ShapeKind::TriangleDown => {
            let h = r * 1.4;
            let style = filled();
            vec![SceneNode::Path {
                commands: vec![
                    PathCmd::MoveTo { x: cx, y: cy + h },
                    PathCmd::LineTo { x: cx + h * 0.866, y: cy - h * 0.5 },
                    PathCmd::LineTo { x: cx - h * 0.866, y: cy - h * 0.5 },
                    PathCmd::Close,
                ],
                style,
                closed: true,
                widths: None,
            }]
        }
        ShapeKind::VLine => {
            let arm = r * 0.7;
            let s = glyph_stroke(r * 0.35);
            vec![SceneNode::Line {
                x1: cx,
                y1: cy - arm,
                x2: cx,
                y2: cy + arm,
                style: s,
            }]
        }
        ShapeKind::HLine => {
            let arm = r * 0.7;
            let s = glyph_stroke(r * 0.35);
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

/// The dot area (px²) a point layer draws a row at when its size channel
/// does not size it: the layer's own constant `size`, else the theme
/// `point_size`. The point builder reads it through
/// [`resolve_mark_style`](crate::render::draw::resolve_mark_style), and the
/// swarm packing bound through [`fallback_dot`], so the two cannot disagree. Pure:
/// it validates nothing, so a caller reading it ahead of the style's
/// refusals does not refuse in the wrong voice.
pub(crate) fn fallback_dot_area(
    overrides: Option<&crate::spec::mark_style::MarkKwargsSpec>,
    theme: &crate::layout::ThemeInputs,
) -> f64 {
    overrides.and_then(|o| o.size).unwrap_or(theme.sizes.point_size)
}

/// The size a point layer draws a row at when that row does not take its
/// size from the layer's own size channel. Decided once by [`fallback_dot`];
/// every reader projects it through [`area`](Self::area) or
/// [`diameter`](Self::diameter) and none restates which variant applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FallbackDot<'a> {
    /// A swarm feeds the layer and the layer sizes no dot of its own: it
    /// draws at the declared swarm's `point_size`.
    Swarm(&'a crate::transform::swarm::SwarmSpec),
    /// Any other point layer: [`fallback_dot_area`], the layer's own
    /// constant `size`, else the theme `point_size` (px²).
    Area(f64),
}

impl FallbackDot<'_> {
    /// The dot area (px²), which the point builder draws at.
    pub(crate) fn area(self) -> f64 {
        match self {
            Self::Swarm(swarm) => swarm.dot_area(),
            Self::Area(area) => area,
        }
    }

    /// The dot diameter (px), which the swarm packing bound reads. A swarm
    /// dot is its declared `point_size` exactly, not a round trip through
    /// [`area`](Self::area): `dot_diameter(dot_area(x)) != x` at ULP level.
    pub(crate) fn diameter(self) -> f64 {
        match self {
            Self::Swarm(swarm) => swarm.point_size,
            Self::Area(area) => ferrum_scene::dot_diameter(area),
        }
    }
}

/// The one decision of what a point layer's unsized rows draw at: a swarm's
/// own dot when `swarm` feeds the layer and the layer sizes no dot itself
/// ([`sizes_own_dots`](crate::render::position::sizes_own_dots)), else
/// [`fallback_dot_area`].
pub(crate) fn fallback_dot<'a>(
    layer: &crate::render::prepare::LayerPrepared,
    swarm: Option<&'a crate::transform::swarm::SwarmSpec>,
    theme: &crate::layout::ThemeInputs,
) -> FallbackDot<'a> {
    match swarm {
        Some(swarm) if !crate::render::position::sizes_own_dots(layer) => FallbackDot::Swarm(swarm),
        _ => FallbackDot::Area(fallback_dot_area(layer.mark_style.as_ref(), theme)),
    }
}

/// The dot area (px²) each row of a point layer's `batch` takes from its
/// size channel: the row's value through the size scale, or `None` for a row
/// that draws at the layer's resolved `point_size` instead (a null value, or
/// one the scale maps nowhere). `None` overall when every row does: no size
/// channel, no size scale, or a size column this batch cannot read as
/// numeric, which is not refused.
pub(crate) fn scaled_dot_areas(
    batch: &arrow::array::RecordBatch,
    size: Option<&crate::spec::encoding::EncodingSpec>,
    scale: Option<&crate::render::scale_resolve::SizeScale>,
) -> Option<Vec<Option<f64>>> {
    let (size, scale) = size.zip(scale)?;
    let values = col_as_f64(batch, &size.field).ok()?;
    Some(values.into_iter().map(|v| v.and_then(|v| scale.inner.to_pixel_f64(v))).collect())
}

/// `x` and `y`, by scale kind (Category for an `Ordinal` axis, Strict
/// otherwise) — the same dispatch `build`'s own `cx`/`cy` reads take below.
pub(crate) fn placements<'a>(ctx: &'a DrawCtx<'a>) -> Vec<crate::render::draw::Placement<'a>> {
    crate::render::draw::xy_by_scale_kind(ctx)
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
    let size_areas = scaled_dot_areas(ctx.batch, spec.encoding.size.as_ref(), ctx.scales.size.as_ref());

    let shape_values: Option<Vec<Option<String>>> = spec.encoding.shape
        .as_ref()
        .and_then(|e| col_as_ordinal_category_str(ctx.batch, &e.field).ok());

    let opacity_values: Option<Vec<Option<f64>>> = spec.encoding.opacity
        .as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());

    // fill_opacity / stroke_opacity via the shared resolver (FA-11), sampled
    // per-row. Defaults: fill_opacity → 1.0, stroke_opacity → 1.0. The opacity
    // channel is mapped through `ctx.scales.opacity` at the call site below, so
    // the resolver's opacity output is unused here (its default is irrelevant).
    let opacity_res = OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));

    // Per-row stroke/angle channel values (direct passthrough — no scale transform).
    let stroke_width_values = stroke_width_column(ctx);

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
    // #230-3: one exact `(cx, cy)` data anchor per emitted node — see the
    // `MarkBatch::glyph_anchors` doc for why a break's glyph remap needs this
    // instead of measuring a bbox off the shape's own drawn geometry.
    let mut anchors: Vec<(f64, f64)> = Vec::with_capacity(n);

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
            // `ScaleKind::admits_for_pixel` is the ONE shared predicate every
            // mark builder's positional pre-filter reads: only a `Log` scale
            // admits a non-`NaN` infinity, deferring to `to_pixel_f64`'s own
            // on-side gate (clamps when on-side, left out when not); every other
            // scale kind still requires a plain finite value, unchanged.
            _ => match xs_f64.as_ref().and_then(|v| v[i]) {
                Some(a) if ctx.scales.x.admits_for_pixel(a) => match ctx.scales.x.to_pixel_f64(a) {
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
                Some(a) if ctx.scales.y.admits_for_pixel(a) => match ctx.scales.y.to_pixel_f64(a) {
                    Some(p) => p, None => continue,
                },
                _ => continue,
            },
        };
        let cx = cx + x_offsets[i];
        let cy = cy + y_offsets[i];

        // Resolve color via the shared per-row fill resolver (RMARK-03). The
        // resolved paint carries its own provenance (T8 quality-review c2):
        // `Cleared` only when the constant `mark_style.paint.fill` was cleared
        // by a literal "none"/"transparent", never merely because the resolved
        // color is zero-alpha by value. A cleared fill outranks a color-scale
        // hit (`#110`): the clear wins, the hit's palette color is dropped.
        let fill_base = resolve_fill_color(
            ctx.scales.color.as_ref(),
            color_values_str.as_ref().and_then(|v| v.get(i)).and_then(|o| o.as_deref()),
            color_values_f64.as_ref().and_then(|v| v.get(i).copied().flatten()),
            ctx.mark_style.paint.fill_paint(),
        );

        // Resolve per-row opacity (through scale if present).
        let row_opacity =
            resolve_scaled_opacity(&opacity_values, &ctx.scales.opacity, i, ctx.mark_style.paint.opacity);

        let fill = fill_base.map(|c| with_opacity(c, row_opacity));

        // filled=false → hollow points: no fill, and the stroke is painted in
        // the fill's paint (cleared-ness included), at the pre-opacity color.
        let (effective_fill, effective_stroke, effective_sw) =
            if ctx.mark_style.point.filled == Some(false) {
                let sw = if ctx.mark_style.paint.stroke_width > 0.0 {
                    ctx.mark_style.paint.stroke_width
                } else {
                    1.5
                };
                (ClearablePaint::Absent, fill_base, sw)
            } else {
                (fill, ctx.mark_style.paint.stroke_paint(), ctx.mark_style.paint.stroke_width)
            };

        // Resolve per-row radius from size encoding.
        let radius = match size_areas.as_ref().and_then(|areas| areas[i]) {
            Some(area) => (area / std::f64::consts::PI).sqrt(),
            None => default_radius,
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

        // RMARK-04's stroke-width rescue, shared with bar/rect: a positive
        // per-row width over an *absent* stroke paints the stroke in the fill
        // paint (cleared-ness included). Point carried a hand-rolled copy of
        // this block until wave 6A T7 folded it onto the shared helper.
        let effective_stroke_for_row = resolve_effective_stroke(
            row_stroke_width,
            effective_stroke,
            effective_fill,
            stroke_width_values.is_some(),
        );

        let shape_nodes = emit_shape_nodes(
            shape_kind, cx, cy, radius,
            ShapeStyle {
                fill: effective_fill,
                stroke: effective_stroke_for_row,
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
        //
        // #230-3: `(cx, cy)` — the scale-mapped position BEFORE `emit_shape_nodes`
        // drew this row's shape geometry around it — is this row's exact glyph
        // anchor. Repeated once per emitted node (2× for `Cross`, matching
        // `push_many`'s own row-repeat) so `glyph_anchors` stays aligned to nodes,
        // not rows, exactly like `data_indices`.
        anchors.extend(std::iter::repeat_n((cx, cy), shape_nodes.len()));
        acc.push_many(shape_nodes, i);
    }

    let (nodes, data_indices) = acc.finalize();

    MarkBuildResult::populated(MarkBatchKind::Point, nodes, data_indices, &meta)
        .with_glyph_anchors(anchors)
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
    use ferrum_scene::SceneNode;

    /// A row takes its area from the size scale, or reads `None` (the
    /// fallback) when its value is null or maps nowhere; every row falls back
    /// (`None` overall) without a size scale or when the size column cannot
    /// be read as numeric.
    #[test]
    fn scaled_dot_areas_reads_each_row_through_the_size_scale_or_falls_back() {
        use crate::render::scale_resolve::SizeScale;
        use crate::scale::linear::LinearScale;
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("s", DataType::Float64, true),
                Field::new("t", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![Some(0.0), None, Some(1.0)])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        let scale = SizeScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], vec![4.0, 100.0], false, true)),
        };
        let field = |name: &str| EncodingSpec { field: name.into(), ..Default::default() };
        assert_eq!(scaled_dot_areas(&batch, Some(&field("s")), Some(&scale)), Some(vec![Some(4.0), None, Some(100.0)]));
        assert_eq!(scaled_dot_areas(&batch, Some(&field("t")), Some(&scale)), None, "unreadable column");
        assert_eq!(scaled_dot_areas(&batch, Some(&field("s")), None), None, "no size scale");
        assert_eq!(scaled_dot_areas(&batch, None, Some(&scale)), None, "no size channel");
    }

    fn three_row_spec() -> ChartSpec {
        ChartSpec {
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                size: Some(EncodingSpec { field: "sz".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                shape: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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

    /// #166 (disclosed reader-switch): an `Int64` `shape` column must resolve
    /// each row's real shape kind, not default-circle every row.
    ///
    /// RED (verified in place): with `col_as_str` here, the read returns
    /// `Err` for `Int64`, `.ok()` swallows it, `shape_values` stays `None`
    /// for every row, and the per-row match falls to
    /// `ctx.mark_style.point.shape` (unset) → `ShapeKind::Circle` for all
    /// three rows — the legend still lists 3 categories while every glyph on
    /// the panel is a circle. The Utf8 twin
    /// (`point_with_shape_encoding_emits_3_shape_kinds`, immediately above)
    /// is the reference: an Int64 shape column must produce the identical
    /// three shape kinds.
    #[test]
    fn point_with_int64_shape_encoding_emits_3_shape_kinds() {
        use arrow::array::{Float64Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let spec = spec_with_shape();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Int64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        assert!(scales.shape.is_some(), "an Int64 shape column must still resolve a ShapeScale");
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx);

        // Domain [1, 2, 3] (first-appearance order) → SHAPE_PALETTE[0..3] =
        // Circle, Square, Cross — same three kinds as the Utf8 twin.
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                opacity: Some(EncodingSpec { field: "op".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
            ..Default::default()
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
            fill: ClearablePaint::Paint(crate::render::color::from_rgb(0, 0, 0)),
            stroke: ClearablePaint::Absent,
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            stroke_dash: None,
            angle: 0.0,
        }
    }

    /// The three `StrokeStyle`-shaped glyphs paint their lines in the FILL slot,
    /// so a `mark_point(fill="none")` must paint no line at all.
    ///
    /// Wave 6A T12 (`#110`) tightened this from T7's contract: with
    /// `StrokeStyle.color` non-optional, the best a cleared fill could do was
    /// resolve to `TRANSPARENT` and serialize `stroke="rgba(0,0,0,0.000)"`. It is
    /// now `None`, and the attribute is omitted. Both negative controls — an
    /// ABSENT fill still defaulting to black, and an ordinary paint reaching the
    /// glyph unchanged — are what keeps this from passing by simply never
    /// painting.
    #[test]
    fn stroke_shaped_glyphs_omit_a_cleared_fills_paint_entirely() {
        use ferrum_scene::SceneNode;

        let line_colors =
            |kind: ShapeKind, fill: ClearablePaint| -> Vec<Option<ferrum_scene::Color>> {
                let style = ShapeStyle { fill, ..default_shape_style() };
                emit_shape_nodes(kind, 50.0, 50.0, 5.0, style)
                    .iter()
                    .map(|n| match n {
                        SceneNode::Line { style, .. } => style.color,
                        other => panic!("{kind:?} must emit only Line nodes, got: {other:?}"),
                    })
                    .collect()
            };

        let blue = crate::render::color::from_rgb(0x11, 0x77, 0xEE);
        for kind in [ShapeKind::Cross, ShapeKind::VLine, ShapeKind::HLine] {
            let expected_nodes = if matches!(kind, ShapeKind::Cross) { 2 } else { 1 };

            let cleared = line_colors(kind, ClearablePaint::Cleared);
            assert_eq!(cleared.len(), expected_nodes);
            for c in &cleared {
                assert_eq!(
                    *c, None,
                    "{kind:?} with a CLEARED fill must paint NO glyph line — any color here \
                     serializes a stroke attribute for a `mark_point(fill=\"none\")`"
                );
            }

            // Negative control: an ABSENT fill keeps the historical black default.
            let absent = line_colors(kind, ClearablePaint::Absent);
            assert_eq!(absent.len(), expected_nodes);
            for c in &absent {
                let c = c.expect("{kind:?} with an ABSENT fill must still paint its glyph");
                assert_eq!(
                    (c.r, c.g, c.b, c.a),
                    (0, 0, 0, 255),
                    "{kind:?} with an ABSENT fill must keep painting opaque black",
                );
            }

            // And an ordinary paint still reaches the glyph unchanged.
            for c in line_colors(kind, ClearablePaint::Paint(blue)) {
                let c = c.expect("an ordinary paint must reach the glyph");
                assert_eq!((c.r, c.g, c.b, c.a), (0x11, 0x77, 0xEE, 255), "{kind:?}");
            }
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

    /// A theme `point_size` of 0 is a valid (if invisible) configuration:
    /// the formula must yield radius 0, not NaN or a panic. (A constant
    /// `size=0` is refused by the size rule instead.)
    #[test]
    fn default_radius_zero_point_size_yields_zero_radius() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let mut theme = ThemeInputs::default();
        theme.sizes.point_size = 0.0;
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Point).unwrap();
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
            fill: ClearablePaint::Paint(crate::render::color::from_rgb(100, 150, 200)),
            stroke: ClearablePaint::Absent,
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
            stroke: style.stroke,
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
            stroke: style.stroke,
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
            stroke: style.stroke,
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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
        use crate::spec::encoding::{DataType as EncDataType, Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        use arrow::array::Float64Array;

        let spec = ChartSpec {
            mark: Mark::Point,
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
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "lbl".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
                shape: Some("cross".into()),
                ..Default::default()
            }),
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "lbl".into(), type_: None, ..Default::default() }),
                ..Default::default()
            }, // default shape = Circle
            position: None,
            ..Default::default()
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
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                href: Some(EncodingSpec { field: "url".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            ..Default::default()
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

    /// One decision, two exact projections, over {unsized, constant-sized,
    /// field-sized, both} × {swarm-fed, unfed}: only an unsized fed layer
    /// draws the swarm's own dot, whose diameter is the declared
    /// `point_size` bit-exact; every other layer draws its own constant,
    /// else the theme's.
    #[test]
    fn fallback_dot_is_the_swarm_dot_only_for_an_unsized_fed_layer() {
        use crate::render::prepare::LayerPrepared;
        use crate::spec::mark_style::MarkKwargsSpec;
        use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};
        let theme = ThemeInputs::default();
        let swarm = SwarmSpec {
            category: "g".into(),
            value: "v".into(),
            point_size: 3.3,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Vertical,
            dodge: None,
            name: None,
        };
        // Vacuity: this declared size does not survive an area round trip,
        // so only a projection that reads it directly is bit-exact.
        assert_ne!(ferrum_scene::dot_diameter(swarm.dot_area()), swarm.point_size);
        let constant = 200.0;
        assert_ne!(constant, theme.sizes.point_size);
        let layer = |own: Option<f64>, field: bool| {
            LayerPrepared::from_chart_only(&ChartSpec {
                mark: Mark::Point,
                mark_style: own.map(|size| MarkKwargsSpec { size: Some(size), ..Default::default() }),
                encoding: Encoding {
                    size: field.then(|| EncodingSpec { field: "s".into(), ..Default::default() }),
                    ..Default::default()
                },
                ..Default::default()
            })
        };
        let theme_dot = FallbackDot::Area(theme.sizes.point_size);
        let own_dot = FallbackDot::Area(constant);
        let cells = [
            ("unsized", layer(None, false), FallbackDot::Swarm(&swarm), theme_dot),
            ("constant", layer(Some(constant), false), own_dot, own_dot),
            ("field", layer(None, true), theme_dot, theme_dot),
            ("constant and field", layer(Some(constant), true), own_dot, own_dot),
        ];
        for (name, layer, fed, unfed) in cells {
            for (feed, expected) in [(Some(&swarm), fed), (None, unfed)] {
                let dot = fallback_dot(&layer, feed, &theme);
                let fed = feed.is_some();
                assert_eq!(dot, expected, "{name} fed={fed}");
                let (area, diameter) = match expected {
                    FallbackDot::Swarm(s) => (s.dot_area(), s.point_size),
                    FallbackDot::Area(a) => (a, ferrum_scene::dot_diameter(a)),
                };
                assert_eq!(dot.area().to_bits(), area.to_bits(), "{name} fed={fed} area");
                assert_eq!(dot.diameter().to_bits(), diameter.to_bits(), "{name} fed={fed} diameter");
            }
        }
    }
}
