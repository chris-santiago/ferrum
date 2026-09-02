//! mark_rule: reference lines and axis-aligned segments.
//!
//! # Shape derivation (batch-A Task 13, spec c3, ruled 2026-09-01)
//!
//! Rule's geometry is chosen ONCE, by [`RuleShape::resolve`], from the bound/
//! absent pattern of the four positional channels. Every arm of that match
//! names all four channels' presence explicitly, so the arms are exhaustive
//! (the compiler checks it — there is no wildcard) and mutually exclusive by
//! construction. No shape can be hijacked into another shape's mode by a
//! channel that mode does not consume, which is exactly the defect this
//! derivation replaced: a `y`-matching branch that ignored a bound `x` turned
//! `shap_chart(kind="beeswarm")`'s vertical zero-line layer into 1000
//! full-width horizontal spans.
//!
//! | `x` | `y` | `x2` | `y2` | shape |
//! |---|---|---|---|---|
//! | ✓ | ✓ | ✓ | ✓ | [diagonal segment](RuleShape::Diagonal) — mirrors `segment.rs`; `mark_qq(line=True)`'s reference layer |
//! | ✓ | ✓ | — | ✓ | [vertical segment](RuleShape::VerticalSegment) at `x`, spanning `y`..`y2` (boxplot whisker) |
//! | ✓ | ✓ | ✓ | — | [horizontal segment](RuleShape::HorizontalSegment) at `y`, spanning `x`..`x2` (feature-importance error bar) |
//! | ✓ | — | — | — | [vertical span](RuleShape::VerticalSpan) at `x`, full panel height |
//! | — | ✓ | — | — | [horizontal span](RuleShape::HorizontalSpan) at `y`, full panel width |
//! | ✓ | ✓ | — | — | horizontal span at `y` — the documented tie-break, see [`RuleShape::resolve`] |
//!
//! Every other pattern is a typed `RenderError::UnsupportedChannelCombination`
//! naming the supported set. `scene_build.rs`'s `validate_mark_encoding`
//! refuses it up front by calling this same [`RuleShape::resolve`], so the
//! gate and the geometry can never drift apart — they are one function.
//!
//! # Positional reads key off the resolved scale, never the column dtype
//!
//! Every positional read — the anchor channel of each mode AND each mode's
//! value channels — goes through the one shared [`positional_pixels`], which
//! matches on the RESOLVED [`ScaleKind`] exactly as `point.rs`/`bar.rs` do:
//! an `Ordinal` scale takes the categorical reading
//! (`col_as_positional_category_str` + [`ScaleKind::to_pixel_str`]), every
//! continuous scale (Linear/Log/Symlog/Pow/Time) the numeric reading
//! (`col_as_f64` + [`ScaleKind::to_pixel_f64`], non-finite rows skipped).
//! Consequences, matching what `mark_point`/`mark_bar` already give the same
//! columns:
//!
//! - An Int64/Float64/Boolean column on an ordinal scale renders at its
//!   category band — a numeric column is not forced down the continuous path
//!   (where `to_pixel_f64` returns `None` for every row, the silent-empty bug
//!   a dtype-keyed dispatch caused).
//! - A Timestamp column on a time scale renders numerically; nothing about
//!   Timestamp raises.
//! - A dtype the chosen reading genuinely cannot read (e.g. Timestamp forced
//!   onto an ordinal scale) raises the typed `RenderError::UnsupportedDtype`
//!   that reader already constructs — propagated with `?`, never discarded.
//!
//! # Totality invariant (spec c2)
//!
//! No presence-legal channel combination produces empty output silently.
//! `build` resolves a shape (or raises), then renders that shape; there is no
//! terminal fall-through. Empty output is reachable only where it is the
//! honest answer: zero rows, or every row skipped by the documented
//! null/non-finite/out-of-domain per-row checks.

use crate::render::color::Color;
use crate::render::draw::{
    col_as_f64, col_as_positional_category_str, col_as_str, color_field, resolve_stroke_color,
    DrawCtx,
};
use crate::render::mark_nodes::MarkNodes;
use crate::render::marks::channels::{resolve_row_stroke_dash, stroke_dash_column_loader, DashColumns};
use crate::render::marks::opacity::{OpacityFallback, OpacityResolver};
use crate::render::scale_resolve::ScaleKind;
use crate::render::RenderError;
use crate::spec::encoding::Encoding;

/// The geometry a rule layer renders, derived from its bound positional
/// channels by [`RuleShape::resolve`]. Field names are the resolved (post
/// coord-flip) channel columns.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RuleShape<'a> {
    /// `x` + `y` + `x2` + `y2` → one segment per row from `(x, y)` to
    /// `(x2, y2)`. Mirrors `segment.rs`'s geometry exactly; rule keeps its own
    /// copy so its per-row stroke resolution (`rule_stroke_style`) applies
    /// uniformly across all of rule's shapes.
    Diagonal { x: &'a str, y: &'a str, x2: &'a str, y2: &'a str },
    /// `x` + `y` + `y2` (no `x2`) → vertical segment at `x` from `y` to `y2`.
    VerticalSegment { x: &'a str, y: &'a str, y2: &'a str },
    /// `y` + `x` + `x2` (no `y2`) → horizontal segment at `y` from `x` to `x2`.
    HorizontalSegment { y: &'a str, x: &'a str, x2: &'a str },
    /// `x` alone → full-panel-height vertical reference line at `x`.
    VerticalSpan { x: &'a str },
    /// `y` alone → full-panel-width horizontal reference line at `y`.
    HorizontalSpan { y: &'a str },
}

/// Build the typed refusal for a channel pattern rule cannot draw. Shared by
/// [`RuleShape::resolve`]'s unsupported arms so the message (which enumerates
/// the supported set) has exactly one definition.
fn unsupported_rule_shape(coord_flipped: bool) -> RenderError {
    RenderError::UnsupportedChannelCombination {
        mark: "mark_rule",
        channel: "positional",
        hint: "mark_rule supports: y= alone (horizontal span), x= alone (vertical span), \
               x=+y=+y2= (ranged vertical segment), y=+x=+x2= (ranged horizontal segment), \
               or x=+y=+x2=+y2= (diagonal segment)",
        hint_alt_channel: None,
        coord_flipped,
    }
}

impl<'a> RuleShape<'a> {
    /// Derive the shape from `encoding`'s bound positional channels.
    ///
    /// The match below lists all sixteen presence patterns with no wildcard:
    /// adding a channel, or changing which channels a shape consumes, is a
    /// compile error until every pattern is re-decided. That is the point —
    /// the three cycles of "another shape got swallowed by a branch that
    /// ignored a channel it doesn't consume" all came from ordered `if let`
    /// gates that each named only the channels they wanted.
    ///
    /// **The `x` + `y` tie-break (both bound, no `x2`/`y2`).** Neither channel
    /// has a second endpoint, so the shape is genuinely ambiguous. It resolves
    /// to a horizontal span at `y`, which is what this shape has always
    /// rendered (`mark_rule().encode(x=…, y=…)`); `x` still contributes to the
    /// x-scale domain. This case is reachable only when a layer DECLARES both
    /// channels: `scene_build.rs`'s `build_panel_mark_batches` clears a rule
    /// layer's inherited-only opposite span channel before the shape is
    /// derived, precisely so chart-level inheritance cannot decide a reference
    /// line's axis (see the comment at that call site — it is the other half
    /// of this ruling).
    pub(crate) fn resolve(
        encoding: &'a Encoding,
        coord_flipped: bool,
    ) -> Result<Self, RenderError> {
        let field = |e: &'a Option<crate::spec::encoding::EncodingSpec>| {
            e.as_ref().map(|s| s.field.as_str())
        };
        match (
            field(&encoding.x),
            field(&encoding.y),
            field(&encoding.x2),
            field(&encoding.y2),
        ) {
            (Some(x), Some(y), Some(x2), Some(y2)) => Ok(Self::Diagonal { x, y, x2, y2 }),
            (Some(x), Some(y), None, Some(y2)) => Ok(Self::VerticalSegment { x, y, y2 }),
            (Some(x), Some(y), Some(x2), None) => Ok(Self::HorizontalSegment { y, x, x2 }),
            // Tie-break — see this fn's doc comment.
            (Some(_), Some(y), None, None) => Ok(Self::HorizontalSpan { y }),
            (Some(x), None, None, None) => Ok(Self::VerticalSpan { x }),
            (None, Some(y), None, None) => Ok(Self::HorizontalSpan { y }),
            // A second endpoint with no anchor to pair it with (`x2` without
            // `y`, `y2` without `y`, `x2` without `x`), or no positional
            // channel at all: nothing to draw, refused by name rather than
            // drawn as some other shape that ignores the extra channel.
            (Some(_), None, Some(_), None)
            | (Some(_), None, None, Some(_))
            | (Some(_), None, Some(_), Some(_))
            | (None, Some(_), Some(_), None)
            | (None, Some(_), None, Some(_))
            | (None, Some(_), Some(_), Some(_))
            | (None, None, None, None)
            | (None, None, Some(_), None)
            | (None, None, None, Some(_))
            | (None, None, Some(_), Some(_)) => Err(unsupported_rule_shape(coord_flipped)),
        }
    }
}

/// Per-row pixel positions for one positional channel, keyed off the RESOLVED
/// [`ScaleKind`] — never off the column's Arrow dtype (batch-A Task 13 spec
/// c3). This is the single positional read every rule shape uses, for anchor
/// and value channels alike; see this module's doc comment for the dtype
/// consequences and why the scale, not the dtype, is the discriminant.
///
/// `None` at row `i` means "this row draws nothing": a null category, a null
/// or non-finite number, or a value the scale maps outside its range. An
/// `Err` means the column cannot be read at all under the chosen reading —
/// the typed `RenderError::UnsupportedDtype` the reader constructs, returned
/// as-is.
fn positional_pixels(
    ctx: &DrawCtx,
    field: &str,
    scale: &ScaleKind,
) -> Result<Vec<Option<f64>>, RenderError> {
    match scale {
        // Same reader `point.rs`/`bar.rs`/`tick.rs` use for an ordinal
        // positional channel: Int/Float/Boolean/Utf8 all stringify exactly the
        // way the ordinal domain was built, and a null row lands in the null
        // band (FA-9).
        ScaleKind::Ordinal(_) => {
            let cats = col_as_positional_category_str(ctx.batch, field)?;
            Ok(cats
                .iter()
                .map(|c| c.as_deref().and_then(|v| scale.to_pixel_str(v)))
                .collect())
        }
        _ => {
            let vals = col_as_f64(ctx.batch, field)?;
            Ok(vals
                .into_iter()
                .map(|v| v.filter(|x| x.is_finite()).and_then(|x| scale.to_pixel_f64(x)))
                .collect())
        }
    }
}

/// Resolve a per-row stroke color from the color encoding + color scale, if both
/// are present. Each row's category value is mapped through `ctx.scales.color`
/// (the same path `line.rs`/`point.rs` use). Returns `None` when there is no
/// color encoding, so callers fall back to the constant mark-style stroke.
fn rule_color_values(ctx: &DrawCtx) -> Option<Vec<Option<Color>>> {
    let field = color_field(ctx, ctx.spec)?;
    let scale = ctx.scales.color.as_ref()?;
    let cats = col_as_str(ctx.batch, field).ok()?;
    Some(
        cats.iter()
            .map(|c| c.as_deref().and_then(|v| scale.lookup(v)))
            .collect(),
    )
}

/// Build a per-row stroke style for rule segments, applying encoding column values.
///
/// `opacity` / `stroke_opacity` are resolved via the shared [`OpacityResolver`]
/// (C7) — byte-identical to the prior inline finite-check + clamp + default,
/// which already matched the resolver's contract exactly. Rule is a stroke-only
/// mark, so the resolver's `fill_opacity` slot is unused.
///
/// `stroke_dash` (batch-A Task 13) resolves through the shared T12 helpers
/// (`resolve_row_stroke_dash`): a categorical `stroke_dash` field maps each
/// row through `ctx.scales.stroke_dash` (`StrokeDashScale::dash_for`); a
/// numeric field keeps the pre-existing `DASH_PALETTE` index contract
/// byte-identically. Both fall back to the mark-style literal for a null/
/// out-of-domain row, matching every other stroke-consuming mark.
fn rule_stroke_style(
    ctx: &DrawCtx,
    i: usize,
    opacity_res: &OpacityResolver,
    sw_vals: &Option<Vec<Option<f64>>>,
    dash_cols: &DashColumns,
    color_vals: &Option<Vec<Option<Color>>>,
) -> ferrum_scene::StrokeStyle {
    use crate::render::color::with_opacity;
    use crate::render::draw::to_scene_stroke;

    let (opacity, _, stroke_opacity) = opacity_res.at_row(i);
    let stroke_width = sw_vals.as_ref()
        .and_then(|v| v.get(i).copied().flatten())
        .filter(|v| *v >= 0.0 && v.is_finite())
        .unwrap_or(ctx.mark_style.paint.stroke_width);
    let dash_vec = resolve_row_stroke_dash(
        dash_cols,
        ctx.scales.stroke_dash.as_ref(),
        i,
        ctx.mark_style.paint.stroke_dash.as_deref(),
    );
    let effective_dash = dash_vec.as_deref();
    // Precedence (explicit constant stroke > per-row color > theme > fill) lives
    // in `resolve_stroke_color`. An explicit `stroke=` in mark_kwargs must not be
    // overridden by a color encoding inherited from a parent chart (e.g. boxplot
    // whiskers keep their neutral gray even when the chart encodes `hue`).
    let row_color = color_vals
        .as_ref()
        .and_then(|v| v.get(i).copied().flatten());
    let base_color = resolve_stroke_color(ctx.mark_style, row_color);
    let stroke_color = with_opacity(base_color, opacity);
    let mut style = to_scene_stroke(stroke_color, stroke_width, 1.0, effective_dash, None, None);
    style.stroke_opacity = stroke_opacity;
    style
}

/// Finalize a rule branch's accumulated nodes into a `MarkBuildResult`,
/// aligning metadata (tooltip/href/description) to the kept node indices —
/// the row-skip/metadata-alignment pattern (#6 defect class) every one of
/// rule's modes repeats identically; hoisted here (batch-A Task 13 spec
/// review) once the fix round added two more modes that needed it.
fn finalize_rule_build(
    acc: MarkNodes,
    meta: &crate::render::draw::MetadataColumns,
) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::MarkBatchKind;
    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
    MarkBuildResult {
        kind: MarkBatchKind::Rule,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
    }
}

/// Emit one `Line` node per row whose `endpoints` resolve, keeping node and
/// source-row indices in lockstep (#6 defect class). Every rule shape shares
/// this loop: the shape-specific part is the `endpoints` closure, which
/// returns each row's final pixel coordinates (position offsets already
/// applied by the caller, since the span shapes deliberately anchor their
/// non-positional ends to the panel edges) or `None` to skip the row.
fn rule_segments(
    n: usize,
    endpoints: impl Fn(usize) -> Option<(f64, f64, f64, f64)>,
    style: impl Fn(usize) -> ferrum_scene::StrokeStyle,
) -> MarkNodes {
    use ferrum_scene::SceneNode;
    let mut acc = MarkNodes::with_capacity(n);
    for i in 0..n {
        if let Some((x1, y1, x2, y2)) = endpoints(i) {
            acc.push(SceneNode::Line { x1, y1, x2, y2, style: style(i) }, i);
        }
    }
    acc
}

pub fn build(ctx: &DrawCtx) -> Result<crate::render::draw::MarkBuildResult, RenderError> {
    use crate::render::draw::MetadataColumns;

    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;

    // Per-row stroke channel vectors. `opacity` / `stroke_opacity` are resolved
    // by the shared OpacityResolver (C7); `stroke_width` stays local. Rule has
    // no fill, so the resolver's fill default is unused. `stroke_dash` (T13)
    // loads via the shared T12 `stroke_dash_column_loader`, mirroring point.rs/
    // line.rs — see `rule_stroke_style`'s doc for the categorical/numeric split.
    let opacity_res =
        OpacityResolver::load(ctx, OpacityFallback::Standard, (ctx.mark_style.paint.opacity, 1.0, 1.0));
    let sw_vals: Option<Vec<Option<f64>>> = spec.encoding.stroke_width.as_ref()
        .and_then(|e| col_as_f64(ctx.batch, &e.field).ok());
    let dash_cols = stroke_dash_column_loader(ctx);
    let color_vals = rule_color_values(ctx);

    let meta = MetadataColumns::from_ctx(ctx);

    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);
    let style = |i: usize| rule_stroke_style(ctx, i, &opacity_res, &sw_vals, &dash_cols, &color_vals);

    // The shape is derived once, from the bound-channel pattern
    // (batch-A Task 13 spec c3 — see [`RuleShape::resolve`]). Every positional
    // read below goes through [`positional_pixels`], keyed off that channel's
    // RESOLVED scale kind, so an ordinal scale reads categories and a
    // continuous scale reads numbers whatever the column's Arrow dtype is.
    //
    // `scene_build.rs`'s `validate_mark_encoding` has already run this exact
    // derivation and refused every unsupported pattern before `build` is
    // called, so the `?` below cannot fire on a chart that renders — the gate
    // and the geometry are one function, not two that could drift.
    let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));
    let acc = match RuleShape::resolve(&spec.encoding, coord_flipped)? {
        RuleShape::Diagonal { x, y, x2, y2 } => {
            let xs = positional_pixels(ctx, x, &ctx.scales.x)?;
            let ys = positional_pixels(ctx, y, &ctx.scales.y)?;
            let x2s = positional_pixels(ctx, x2, &ctx.scales.x)?;
            let y2s = positional_pixels(ctx, y2, &ctx.scales.y)?;
            let n = xs.len().min(ys.len()).min(x2s.len()).min(y2s.len());
            rule_segments(
                n,
                |i| {
                    Some((
                        xs[i]? + x_offsets[i],
                        ys[i]? + y_offsets[i],
                        x2s[i]? + x_offsets[i],
                        y2s[i]? + y_offsets[i],
                    ))
                },
                style,
            )
        }
        RuleShape::VerticalSegment { x, y, y2 } => {
            let xs = positional_pixels(ctx, x, &ctx.scales.x)?;
            let ys = positional_pixels(ctx, y, &ctx.scales.y)?;
            let y2s = positional_pixels(ctx, y2, &ctx.scales.y)?;
            let n = xs.len().min(ys.len()).min(y2s.len());
            rule_segments(
                n,
                |i| {
                    let px = xs[i]? + x_offsets[i];
                    Some((px, ys[i]? + y_offsets[i], px, y2s[i]? + y_offsets[i]))
                },
                style,
            )
        }
        RuleShape::HorizontalSegment { y, x, x2 } => {
            let ys = positional_pixels(ctx, y, &ctx.scales.y)?;
            let xs = positional_pixels(ctx, x, &ctx.scales.x)?;
            let x2s = positional_pixels(ctx, x2, &ctx.scales.x)?;
            let n = ys.len().min(xs.len()).min(x2s.len());
            rule_segments(
                n,
                |i| {
                    let py = ys[i]? + y_offsets[i];
                    Some((xs[i]? + x_offsets[i], py, x2s[i]? + x_offsets[i], py))
                },
                style,
            )
        }
        // Span shapes anchor their other end to the panel edges, which carry
        // no position offset — only the positional end is offset, matching
        // every previous revision of these two modes byte for byte.
        RuleShape::VerticalSpan { x } => {
            let xs = positional_pixels(ctx, x, &ctx.scales.x)?;
            rule_segments(
                xs.len(),
                |i| {
                    let px = xs[i]? + x_offsets[i];
                    Some((px, panel.y, px, panel.y + panel.h))
                },
                style,
            )
        }
        RuleShape::HorizontalSpan { y } => {
            let ys = positional_pixels(ctx, y, &ctx.scales.y)?;
            rule_segments(
                ys.len(),
                |i| {
                    let py = ys[i]? + y_offsets[i];
                    Some((panel.x, py, panel.x + panel.w, py))
                },
                style,
            )
        }
    };
    Ok(finalize_rule_build(acc, &meta))
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
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn ranged_rule_emits_vertical_segments_for_ordinal_x() {
        // Phase 10c-pre: ordinal x + y + y2 → vertical segment per row (boxplot whisker).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
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
            Field::new("cat", arrow::datatypes::DataType::Utf8, false),
            Field::new("lo",  arrow::datatypes::DataType::Float64, false),
            Field::new("hi",  arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2, "expected 2 ranged rule lines");
    }

    #[test]
    fn ranged_rule_emits_horizontal_segments_for_ordinal_y() {
        // Phase 10d-pre: ordinal y + x + x2 → horizontal segment per row
        // (feature-importance error bars on horizontal-bar charts).
        use arrow::array::StringArray;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                x: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
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
            Field::new("cat", arrow::datatypes::DataType::Utf8, false),
            Field::new("lo",  arrow::datatypes::DataType::Float64, false),
            Field::new("hi",  arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2, "expected 2 horizontal-ranged rule lines");
    }

    #[test]
    fn y_only_rule_emits_horizontal_lines() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: None,
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
 coord: None,
 mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.x = Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0), &crate::layout::ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count(), 2);
    }

    #[test]
    fn ranged_rule_resolves_per_row_color_encoding() {
        // mark_rule(...).encode(color="dir:N") on a ranged vertical rule must
        // map each row's category through the color scale, yielding distinct
        // stroke colors per category (candlestick wicks colored up/down).
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "dir".into(), type_: Some(crate::spec::encoding::DataType::Nominal), ..Default::default() }),
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
            Field::new("cat", DataType::Utf8, false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
            Field::new("dir", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
            Arc::new(StringArray::from(vec!["up", "down"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let strokes: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some((style.color.r, style.color.g, style.color.b)),
            _ => None,
        }).collect();
        assert_eq!(strokes.len(), 2, "expected 2 ranged rule lines");
        assert_ne!(strokes[0], strokes[1], "color encoding must yield distinct per-row stroke colors");
    }

    #[test]
    fn ranged_rule_explicit_stroke_wins_over_color_encoding() {
        // Regression: an explicit constant stroke= in mark_kwargs must NOT be
        // overridden by a per-row color encoding inherited from a parent chart
        // (e.g. boxplot whiskers set stroke="theme:label" but catplot encodes
        // hue via color — the whisker must stay gray, not turn per-row accent).
        use crate::spec::mark_style::MarkKwargsSpec;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                // Color encoding present (simulates chart-level hue=species).
                color: Some(EncodingSpec { field: "dir".into(), type_: Some(crate::spec::encoding::DataType::Nominal), ..Default::default() }),
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
            Field::new("cat", DataType::Utf8, false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
            Field::new("dir", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
            Arc::new(StringArray::from(vec!["up", "down"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        // Explicit constant stroke override — simulates what boxplot whisker layers do.
        let overrides = MarkKwargsSpec { stroke: Some("#6b7280".into()), stroke_dash: Some(vec![]), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule).unwrap();
        assert!(mark_style.paint.stroke_is_user_set, "stroke_is_user_set must be true after explicit override");
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let strokes: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some((style.color.r, style.color.g, style.color.b)),
            _ => None,
        }).collect();
        assert_eq!(strokes.len(), 2, "expected 2 ranged rule lines");
        // Both rows must use the explicit constant stroke, NOT per-row accent colors.
        assert_eq!(strokes[0], strokes[1], "explicit constant stroke must win over per-row color encoding");
        // And the color must match the explicit override (#6b7280).
        assert_eq!(strokes[0], (0x6b, 0x72, 0x80), "stroke must be the explicitly set color, not a per-row accent");
    }

    #[test]
    fn ranged_rule_without_color_uses_constant_stroke() {
        // No color encoding → both rules share the constant mark-style stroke
        // (no regression for the candlestick-without-color case).
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: Some(crate::spec::encoding::DataType::Ordinal), ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
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
            Field::new("cat", DataType::Utf8, false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None, row_strip_title: None, row_facet_key: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default()).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();
        let strokes: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some((style.color.r, style.color.g, style.color.b)),
            _ => None,
        }).collect();
        assert_eq!(strokes.len(), 2);
        assert_eq!(strokes[0], strokes[1], "no color encoding must use a single constant stroke");
    }

    // ── Metadata-alignment regression tests (#6 defect class) ────────────────
    //
    // Rule has four modes. Tests cover the two most common row-skip paths:
    //   - ranged vertical rule (ordinal x + y + y2): null y2 skips the row
    //   - horizontal span (y only): non-finite y skips the row
    // Plus an href channel test for the ranged-horizontal mode.
    //
    // Fail-before: `build_metadata(ctx)` produced full per-row vectors before
    // any loop. When row 1 was skipped, node 1 received row 1's metadata.
    //
    // Pass-after: each mode finalizes its own MarkNodes accumulator and calls
    // `build_metadata_for_indices`, aligning metadata to kept nodes only.

    fn make_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        }
    }

    /// Regression: ranged-vertical rule (ordinal x + y + y2) with a null y2
    /// skips that row. The tooltip on each surviving node must point to its true
    /// source row.
    ///
    /// Batch: 3 rows, y2=[5.0, null, 12.0], tooltip=["tip_a","tip_b","tip_c"].
    /// Row 1 (null y2) is skipped → 2 nodes. Node 1 must have "tip_c", not "tip_b".
    #[test]
    fn rule_ranged_vertical_skipped_null_y2_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal),      ..Default::default() }),
                y:  Some(EncodingSpec { field: "lo".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(),  type_: None,                    ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![Some(5.0_f64), None, Some(12.0)])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 2,
            "expected 2 rule nodes after null-y2 skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a' (row 0); got '{t0}'");

        // Node 1 → row 2 → "tip_c". Old code: "tip_b" (the alignment bug).
        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'. \
             This fails on pre-migration code using build_metadata(ctx).");
    }

    /// Regression: horizontal span rule (y only) with a non-finite y skips that
    /// row. The tooltip on each surviving node must point to its true source row.
    ///
    /// Batch: 3 rows, y=[10.0, NaN, 80.0], tooltip=["tip_a","tip_b","tip_c"].
    /// Row 1 (NaN y) is skipped → 2 nodes. Node 1 must have "tip_c".
    #[test]
    fn rule_horizontal_span_skipped_nan_y_tooltip_aligned() {
        use crate::spec::encoding::DataType as SDT;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y:  Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("y",   DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        // Row 1 has NaN y → skipped by `v.is_finite()` guard in y-only path.
        let mut y_vals = arrow::array::Float64Builder::new();
        y_vals.append_value(10.0);
        y_vals.append_value(f64::NAN);
        y_vals.append_value(80.0);
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(y_vals.finish()),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        // Need an x-scale for resolve_scales; provide a dummy spec with x.
        let mut spec_with_x = spec.clone();
        spec_with_x.encoding.x = Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() });
        let (scales, _) = resolve_scales(&spec_with_x, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 2,
            "expected 2 rule nodes after NaN-y skip; got {}", result.nodes.len());

        let tooltips = result.tooltips.expect("tooltips must be Some when tooltip is encoded");
        assert_eq!(tooltips.len(), 2, "tooltip count must equal node count");

        let t0 = &tooltips[0].fields[0].value;
        assert_eq!(t0, "tip_a", "node 0 tooltip must be 'tip_a'; got '{t0}'");

        let t1 = &tooltips[1].fields[0].value;
        assert_eq!(t1, "tip_c",
            "node 1 tooltip must be 'tip_c' (row 2), not 'tip_b' (row 1); got '{t1}'");
    }

    /// Href-channel alignment on the ranged-horizontal mode (ordinal y + x + x2).
    /// Row 1 has null x2 → skipped. Node 1 href must be "url_c" (row 2), not
    /// "url_b" (row 1, the old bug).
    #[test]
    fn rule_ranged_horizontal_skipped_null_x2_href_aligned() {
        use crate::spec::encoding::DataType as SDT;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y:  Some(EncodingSpec { field: "cat".into(), type_: Some(SDT::Ordinal),      ..Default::default() }),
                x:  Some(EncodingSpec { field: "lo".into(),  type_: Some(SDT::Quantitative), ..Default::default() }),
                x2: Some(EncodingSpec { field: "hi".into(),  type_: None,                    ..Default::default() }),
                href: Some(EncodingSpec { field: "url".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8,    false),
            Field::new("lo",  DataType::Float64, false),
            Field::new("hi",  DataType::Float64, true),   // nullable — row 1 null → skip
            Field::new("url", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![Some(5.0_f64), None, Some(12.0)])),
            Arc::new(StringArray::from(vec!["url_a", "url_b", "url_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 2,
            "expected 2 rule nodes after null-x2 skip; got {}", result.nodes.len());

        let hrefs = result.hrefs.expect("hrefs must be Some when href is encoded");
        assert_eq!(hrefs.len(), 2, "href count must equal node count");
        assert_eq!(hrefs[0].as_deref(), Some("url_a"), "node 0 href must be 'url_a'");
        assert_eq!(hrefs[1].as_deref(), Some("url_c"),
            "node 1 href must be 'url_c' (row 2), not 'url_b' (row 1); \
             old build_metadata would give 'url_b'");
    }

    /// No-skip backward-compat (horizontal span mode): all rows are finite →
    /// all nodes produced, tooltips in original row order.
    #[test]
    fn rule_no_skip_tooltips_unchanged() {
        use crate::spec::encoding::DataType as SDT;

        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("y",   DataType::Float64, false),
            Field::new("tip", DataType::Utf8,    false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0_f64, 50.0, 80.0])),
            Arc::new(StringArray::from(vec!["tip_a", "tip_b", "tip_c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let mut spec_with_x = spec.clone();
        spec_with_x.encoding.x = Some(EncodingSpec { field: "y".into(), type_: Some(SDT::Quantitative), ..Default::default() });
        let (scales, _) = resolve_scales(&spec_with_x, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        assert_eq!(result.nodes.len(), 3, "all 3 rows must produce rule nodes");
        let tooltips = result.tooltips.expect("tooltips must be Some");
        assert_eq!(tooltips.len(), 3, "tooltip count must equal node count");
        let values: Vec<&str> = tooltips.iter().map(|t| t.fields[0].value.as_str()).collect();
        assert_eq!(values, vec!["tip_a", "tip_b", "tip_c"],
            "no-skip: tooltips must be in original row order");
    }

    // ── Batch-A Task 13: four-numeric diagonal segment + stroke_dash ─────────

    /// The fifth rule mode (x + y + x2 + y2, all four quantitative) mirrors
    /// `segment.rs`'s geometry exactly. This is the exact channel shape
    /// `mark_qq(line=True)`'s desugar binds (`heavy_stat.py::desugar_qq`'s
    /// `reference` layer: `x="qq_line_x_start"`, `y="qq_line_y_start"`,
    /// `x2="qq_line_x_end"`, `y2="qq_line_y_end"`) — none of those four
    /// fields is ordinal, so before this mode existed the two ranged-ordinal
    /// branches above both failed to match and the reference diagonal fell
    /// through to `empty()`, silently never rendering (audit F-L06). Field
    /// names here are generic (`x`/`y`/`x2`/`y2`), matching this file's own
    /// convention for its other shape tests (e.g.
    /// `ranged_rule_emits_vertical_segments_for_ordinal_x` uses `cat`/`lo`/
    /// `hi`, not boxplot's real column names) — the channel COMBINATION is
    /// what qq's desugar produces, not the field names, which are
    /// incidental. Endpoints are checked against the same `scales.x`/`scales.y`
    /// the production code calls, not recomputed independently, so this pins
    /// wiring correctness (right column → right axis → right endpoint) rather
    /// than re-deriving the scale math.
    #[test]
    fn four_numeric_diagonal_emits_line_with_correct_endpoints() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: None, ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("y",  DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, -3.0])),
            Arc::new(Float64Array::from(vec![0.0, -3.0])),
            Arc::new(Float64Array::from(vec![10.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 3.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 2, "expected 2 diagonal segments, one per row");

        let expect = |xv: f64, yv: f64| (
            scales.x.to_pixel_f64(xv).expect("x must map to a pixel"),
            scales.y.to_pixel_f64(yv).expect("y must map to a pixel"),
        );
        let (ex0, ey0) = expect(0.0, 0.0);
        let (ex0b, ey0b) = expect(10.0, 10.0);
        assert_eq!(lines[0], (ex0, ey0, ex0b, ey0b), "row 0 endpoints must match the resolved x/y scales");
        let (ex1, ey1) = expect(-3.0, -3.0);
        let (ex1b, ey1b) = expect(3.0, 3.0);
        assert_eq!(lines[1], (ex1, ey1, ex1b, ey1b), "row 1 endpoints must match the resolved x/y scales");
        assert_ne!((lines[0].0, lines[0].1), (lines[0].2, lines[0].3),
            "a real diagonal segment must not collapse to a single point");
    }

    // ── Fix round (spec review, 2026-09-01): dtype-aware ranged shapes ───────
    //
    // The spec reviewer's live repro: `fm.Chart(d3).mark_rule().encode(x="a",
    // y="lo", y2="hi")` with all-NUMERIC columns produced zero mark nodes —
    // `validate_mark_encoding` accepted the shape (x is bound), but rule.rs's
    // `build` only ever tried `col_as_str` on `x` (the ordinal-boxplot-whisker
    // reading), so a numeric `x` fell through every branch to the terminal
    // `empty()`. The horizontal mirror (numeric `y` + `x` + `x2`) was silently
    // WRONG rather than silently empty: it matched the y-only horizontal-span
    // branch and rendered a full-width horizontal line at `y`, ignoring `x`/
    // `x2` entirely. Both fixtures below use plain, unannotated field names
    // bound to `Float64` columns with `type_: None` — exactly what a bare
    // Python `.encode(x="a", y="lo", y2="hi")` call lowers to for numeric
    // dtype columns (type inferred as Quantitative at scale-resolution time,
    // never given an explicit Ordinal annotation), so these are a faithful
    // mirror of the real encode path's output, not a hand-built shape the
    // production path could never emit.

    /// Vertical-segment pin: numeric `x` + `y` + `y2` (no `x2`) now renders a
    /// vertical `Line` per row at the continuous x position, instead of the
    /// pre-fix silent `empty()`. Endpoints are checked against the same
    /// `scales.x`/`scales.y` the production code calls (wiring pin, not a
    /// re-derivation of the scale math).
    #[test]
    fn vertical_segment_numeric_x_emits_line_with_correct_endpoints() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "a".into(),  type_: None, ..Default::default() }),
                y:  Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("a",  DataType::Float64, false),
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
            Arc::new(Float64Array::from(vec![50.0, 80.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 2, "expected 2 vertical segments, one per row — previously 0 (silent empty())");

        let expect_x = |xv: f64| scales.x.to_pixel_f64(xv).expect("x must map to a pixel");
        let expect_y = |yv: f64| scales.y.to_pixel_f64(yv).expect("y must map to a pixel");
        assert_eq!(lines[0], (expect_x(1.0), expect_y(10.0), expect_x(1.0), expect_y(50.0)),
            "row 0: a true vertical segment stays at one x pixel for both endpoints");
        assert_eq!(lines[1], (expect_x(4.0), expect_y(20.0), expect_x(4.0), expect_y(80.0)),
            "row 1 endpoints must match the resolved x/y scales");
        assert_eq!(lines[0].0, lines[0].2, "a vertical segment's x1 must equal its x2");
        assert_ne!(lines[0].1, lines[0].3, "a real segment must not collapse to a single point");
    }

    /// Horizontal-segment pin: numeric `y` + `x` + `x2` (no `y2`) now renders
    /// a horizontal `Line` per row at the continuous y position, instead of
    /// the pre-fix silently WRONG render (a full-width horizontal line at
    /// `y`, ignoring `x`/`x2`).
    #[test]
    fn horizontal_segment_numeric_y_emits_line_with_correct_endpoints() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y:  Some(EncodingSpec { field: "b".into(),  type_: None, ..Default::default() }),
                x:  Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("b",  DataType::Float64, false),
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
            Arc::new(Float64Array::from(vec![50.0, 80.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 2, "expected 2 horizontal segments, one per row");

        let expect_x = |xv: f64| scales.x.to_pixel_f64(xv).expect("x must map to a pixel");
        let expect_y = |yv: f64| scales.y.to_pixel_f64(yv).expect("y must map to a pixel");
        assert_eq!(lines[0], (expect_x(10.0), expect_y(1.0), expect_x(50.0), expect_y(1.0)),
            "row 0: a true horizontal segment stays at one y pixel for both endpoints");
        assert_eq!(lines[1], (expect_x(20.0), expect_y(4.0), expect_x(80.0), expect_y(4.0)),
            "row 1 endpoints must match the resolved x/y scales, not a full-width span at y \
             (the pre-fix bug: x/x2 silently ignored)");
        assert_eq!(lines[0].1, lines[0].3, "a horizontal segment's y1 must equal its y2");
        assert_ne!(lines[0].0, lines[0].2, "a real segment must not collapse to a single point");
    }

    /// Priority regression: when all four channels are bound and numeric
    /// (the diagonal shape), the vertical/horizontal numeric fallbacks above
    /// must NOT shadow the diagonal mode — each fallback is gated on the
    /// other ranged channel (`x2`/`y2`) being absent specifically so this
    /// can't happen. If a future edit dropped that gate, this all-four-bound
    /// fixture would wrongly collapse to a vertical or horizontal segment
    /// instead of a diagonal one.
    #[test]
    fn all_four_numeric_bound_prefers_diagonal_over_ranged_fallbacks() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "x".into(),  type_: None, ..Default::default() }),
                y:  Some(EncodingSpec { field: "y".into(),  type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "x2".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  DataType::Float64, false),
            Field::new("y",  DataType::Float64, false),
            Field::new("x2", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![10.0])),
            Arc::new(Float64Array::from(vec![5.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 1);
        let expect_x = |xv: f64| scales.x.to_pixel_f64(xv).expect("x must map to a pixel");
        let expect_y = |yv: f64| scales.y.to_pixel_f64(yv).expect("y must map to a pixel");
        // Diagonal endpoints: (x, y) -> (x2, y2) = (0,0) -> (10,5). A wrongly
        // preferred vertical fallback would instead produce (x,y)->(x,y2), a
        // fixed-x segment; a wrongly preferred horizontal fallback would
        // produce (x,y)->(x2,y), a fixed-y segment. Neither matches this.
        assert_eq!(lines[0], (expect_x(0.0), expect_y(0.0), expect_x(10.0), expect_y(5.0)),
            "all-four-bound must render the diagonal, not a ranged vertical/horizontal fallback");
    }

    /// T13: a categorical `stroke_dash` field on rule now resolves through
    /// `ctx.scales.stroke_dash` (`StrokeDashScale`) via the shared T12
    /// helpers (`stroke_dash_column_loader`/`resolve_row_stroke_dash`) —
    /// deliberately excluded from T12's own scope (that batch's report names
    /// this rule-specific wiring as left for this task). Before this, rule
    /// read `stroke_dash` only via `col_as_f64`, which silently produces
    /// `None` for a Utf8 categorical column, so a categorical `stroke_dash`
    /// on rule always fell back to the mark-style literal (or solid),
    /// regardless of the row's actual category.
    #[test]
    fn rule_stroke_dash_categorical_encoding_resolves_through_scale() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("y",  DataType::Float64, false),
            Field::new("sd", DataType::Utf8, false),
        ]));
        // First-appearance domain order: solid, dashed, dotted.
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["solid", "dashed", "dotted"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        assert!(scales.stroke_dash.is_some(), "a categorical stroke_dash field must resolve a StrokeDashScale");
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some(style.dash.clone()),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].is_none(), "'solid' (domain index 0) must be the solid slot");
        assert_eq!(lines[1].as_deref(), Some([6.0, 3.0].as_ref()), "'dashed' (domain index 1) must be the long-dash pattern");
        assert_eq!(lines[2].as_deref(), Some([2.0, 3.0].as_ref()), "'dotted' (domain index 2) must be the short-dash pattern");
    }

    /// Byte-identity pin (T13's dash-read wiring change): a numeric
    /// `stroke_dash` field must keep resolving through the `DASH_PALETTE`
    /// index contract exactly as before the `stroke_dash_column_loader`/
    /// `resolve_row_stroke_dash` refactor — no `ctx.scales.stroke_dash` is
    /// resolved for a numeric field (T6's own gate), so `resolve_row_stroke_dash`
    /// takes its numeric branch, byte-identical to the pre-T13 inline
    /// `col_as_f64` + `resolve_stroke_dash` + `.or(base)` read this replaced.
    #[test]
    fn rule_stroke_dash_numeric_encoding_keeps_dash_palette_index_contract() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                stroke_dash: Some(EncodingSpec { field: "sd".into(), type_: Some(crate::spec::encoding::DataType::Quantitative), ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("y",  DataType::Float64, false),
            Field::new("sd", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        assert!(scales.stroke_dash.is_none(), "a numeric stroke_dash field must not resolve a StrokeDashScale");
        // Clear the theme's `reference_line_dash` default (rule picks it up as
        // its literal `stroke_dash` unless overridden — see
        // `resolve_mark_style_empty_stroke_dash_clears_reference_line_dash` in
        // `draw.rs`) so the `base` fallback in `resolve_row_stroke_dash`'s
        // numeric branch is `None`, isolating the index→pattern contract this
        // test pins from that unrelated default.
        use crate::spec::mark_style::MarkKwargsSpec;
        let overrides = MarkKwargsSpec { stroke_dash: Some(vec![]), ..Default::default() };
        let mark_style = resolve_mark_style(Some(&overrides), &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx).unwrap();

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { style, .. } => Some(style.dash.clone()),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].is_none(), "index 0 is the solid sentinel");
        assert_eq!(lines[1].as_deref(), Some([6.0, 3.0].as_ref()), "index 1 is the DASH_PALETTE long-dash pattern");
        assert_eq!(lines[2].as_deref(), Some([2.0, 3.0].as_ref()), "index 2 is the DASH_PALETTE short-dash pattern");
    }

    // ── Fix round (spec review, cycle 2, 2026-09-01): totality invariant ─────
    //
    // The cycle-2 spec review found the SAME silent-empty class relocated to
    // the span modes (the then-terminal fallback, and an early
    // `Err(_) => return empty()` in the x-only span), plus code discarding a
    // typed `UnsupportedDtype` into `empty()` in the ranged modes. All three
    // are fixed by routing every positional read through the single shared
    // [`positional_pixels`] and propagating every read failure via `?`
    // instead of matching it away. Field names below use
    // `type_: None` — exactly what a bare Python `.encode(...)` call lowers
    // to (type inferred from the column's physical Arrow dtype at
    // scale-resolution time, never given an explicit annotation), matching
    // this file's own established convention for the cycle-1 fix-round pins
    // above.

    /// The reviewer's y-only-span live repro: `fm.Chart(df).mark_rule()
    /// .encode(y="cat")` with a Utf8 `y` (`x` and `y2` both absent) landed on
    /// the terminal `empty()` pre-fix — the y-only span mode only ever tried
    /// `col_as_f64`. The shared [`positional_pixels`] read (keyed off the
    /// resolved ordinal scale) renders it instead: one horizontal span per row at that
    /// row's ordinal band center, the same categorical-position semantics
    /// the ranged-horizontal mode already gives an ordinal `y` anchor.
    #[test]
    fn y_only_span_categorical_y_renders_via_ordinal_dispatch() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                y: Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        // resolve_scales needs an x encoding to build a full ResolvedScales;
        // build() itself only ever reads `y` for this shape (x absent),
        // matching `y_only_rule_emits_horizontal_lines`'s established pattern.
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.x = Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx)
            .expect("a Utf8 y-only anchor must render via the ordinal dispatch, not error");

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 2, "expected 2 horizontal spans, one per row — previously 0 (silent empty())");
        let expect_y = |cat: &str| scales.y.to_pixel_str(cat).expect("cat must map to a pixel");
        assert_eq!(lines[0].1, expect_y("a"), "row 0 span must sit at 'a''s ordinal band center");
        assert_eq!(lines[1].1, expect_y("b"), "row 1 span must sit at 'b''s ordinal band center");
        assert_eq!(lines[0].1, lines[0].3, "a horizontal span's y1 must equal its y2");
    }

    /// The reviewer's mirror repro: `fm.Chart(df).mark_rule().encode(x="cat")`
    /// with a Utf8 `x` (`y` absent) hit `Err(_) => return empty()` directly
    /// pre-fix (the old `marks/rule.rs:369`). It now renders one vertical
    /// span per row at that row's ordinal band center, mirroring the
    /// y-only-span fix above.
    #[test]
    fn x_only_span_categorical_x_renders_via_ordinal_dispatch() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        // resolve_scales needs a y encoding to build a full ResolvedScales;
        // build() itself only ever reads `x` for this shape (y absent).
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.y = Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let result = super::build(&ctx)
            .expect("a Utf8 x-only anchor must render via the ordinal dispatch, not error");

        let lines: Vec<_> = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        assert_eq!(lines.len(), 3, "expected 3 vertical spans, one per row — previously 0 (silent empty())");
        let expect_x = |cat: &str| scales.x.to_pixel_str(cat).expect("cat must map to a pixel");
        assert_eq!(lines[0].0, expect_x("a"), "row 0 span must sit at 'a''s ordinal band center");
        assert_eq!(lines[1].0, expect_x("b"), "row 1 span must sit at 'b''s ordinal band center");
        assert_eq!(lines[2].0, expect_x("c"), "row 2 span must sit at 'c''s ordinal band center");
        assert_eq!(lines[0].0, lines[0].2, "a vertical span's x1 must equal its x2");
    }

    /// Issue 2's discarded-error repro: `encode(x="a", y="cat", y2="hi")` —
    /// `x` (the anchor) and `y2` are numeric, but `y` is Utf8. `y`/`y2` are
    /// the ranged VALUE channels for the vertical-segment mode, never
    /// dtype-dispatched (only the anchor `x` is); pre-fix, the failed
    /// `col_as_f64(y)` read was matched away into `empty()` (the old
    /// `marks/rule.rs:253-254`) instead of propagated. It must now surface
    /// as a typed `RenderError::UnsupportedDtype` naming the offending
    /// column, not a blank panel.
    #[test]
    fn ranged_vertical_categorical_y_propagates_typed_error_not_empty() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "a".into(),   type_: None, ..Default::default() }),
                y:  Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(),  type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("a",   DataType::Float64, false),
            Field::new("cat", DataType::Utf8,    false),
            Field::new("hi",  DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["p", "q"])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = make_panel();
        // resolve_scales needs a numeric `y` to build a Linear y-scale; swap
        // in field "a" (numeric) for scale resolution only — build() itself
        // uses the real spec (y="cat"), which is what this test exercises.
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.y = Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        // `MarkBuildResult` (the `Ok` side) has no `Debug` impl, so match
        // explicitly rather than `.expect_err` (which requires `T: Debug`).
        let err = match super::build(&ctx) {
            Err(e) => e,
            Ok(_) => panic!(
                "a Utf8 y (ranged value, not the anchor) must raise a typed error, not render blank"
            ),
        };

        assert!(
            matches!(&err, crate::render::RenderError::UnsupportedDtype { field, .. } if field == "cat"),
            "expected UnsupportedDtype naming the 'cat' column; got {err:?}"
        );
    }

    // ── Fix round (spec review, cycle 3, 2026-09-01): the positional read
    //    keys off the RESOLVED ScaleKind, never the column dtype ────────────
    //
    // The cycle-2 fix hoisted one shared anchor dispatch but kept the wrong
    // discriminant: it asked "is this column Utf8?" instead of "what kind is
    // the resolved scale?". An Int64/Float64 column on an ordinal scale
    // therefore took the numeric path, where `ScaleKind::to_pixel_f64` returns
    // `None` for every row on an ordinal scale — zero marks, no error, while
    // the axis drew the ordinal band ticks. Boolean, a legal ordinal category
    // dtype everywhere else in this crate, hard-raised. `positional_pixels`
    // matches on the scale kind (`point.rs`/`bar.rs`'s canonical discriminant)
    // and reads through `col_as_positional_category_str`, so rule now
    // positions every dtype exactly as its sibling marks do.
    //
    // Fixtures mirror the Python lowering they stand for: `type_:
    // Some(Ordinal)` is what `fm.X('year', type_='ordinal')` lowers to, and
    // `type_: None` is what a bare `.encode(x='flag')` lowers to (the type is
    // inferred from the column's Arrow dtype at scale-resolution time).

    /// Positional channel shorthand: `(field, explicit encoding type)`.
    type Ch<'a> = Option<(&'a str, Option<crate::spec::encoding::DataType>)>;

    /// A single-mark rule spec binding exactly the given positional channels.
    fn rule_spec(x: Ch, y: Ch, x2: Ch, y2: Ch) -> ChartSpec {
        let ch = |c: Ch| {
            c.map(|(field, type_)| EncodingSpec { field: field.into(), type_, ..Default::default() })
        };
        ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding { x: ch(x), y: ch(y), x2: ch(x2), y2: ch(y2), ..Default::default() },
            transforms: Vec::new(), facet: None, layers: None,
            coord: None, mark_style: None, position: None, title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
            chart_description: None, params: Vec::new(),
        }
    }

    /// Resolve scales from `scale_spec` (which must bind both `x` and `y`, as
    /// `resolve_scales` requires for a full `ResolvedScales`), then run
    /// `build` on `spec` and return its line endpoints alongside the resolved
    /// scales — so endpoint assertions can be made against the same scales the
    /// production code called, never a re-derivation of the scale math.
    fn build_rule_lines(
        spec: &ChartSpec,
        scale_spec: &ChartSpec,
        batch: &arrow::record_batch::RecordBatch,
    ) -> (Vec<(f64, f64, f64, f64)>, crate::render::scale_resolve::ResolvedScales) {
        let theme = ThemeInputs::default();
        let panel = make_panel();
        let (scales, _) = resolve_scales(scale_spec, batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Rule).unwrap();
        let ctx = DrawCtx { spec, panel: &panel, theme: &theme, scales: &scales, batch, mark_style: &mark_style };
        let result = super::build(&ctx).expect("build must not refuse a supported shape");
        let lines = result.nodes.iter().filter_map(|n| match n {
            SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
            _ => None,
        }).collect();
        (lines, scales)
    }

    /// Int64 column on an ordinal x-scale (`fm.X('year', type_='ordinal')`):
    /// one vertical span per row at that year's band center. Pre-fix: zero
    /// lines, silently, while the axis drew '2000'/'2001'/'2002' ticks.
    #[test]
    fn x_span_int_ordinal_column_renders_at_band_centers() {
        use crate::spec::encoding::DataType as SDT;
        let spec = rule_spec(Some(("year", Some(SDT::Ordinal))), None, None, None);
        let scale_spec = rule_spec(
            Some(("year", Some(SDT::Ordinal))),
            Some(("year", Some(SDT::Ordinal))),
            None,
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("year", DataType::Int64, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::Int64Array::from(vec![2000_i64, 2001, 2002])),
        ]).unwrap();

        let (lines, scales) = build_rule_lines(&spec, &scale_spec, &batch);
        assert_eq!(lines.len(), 3,
            "an Int64 column on an ordinal scale must render one span per row — previously 0, silently");
        for (line, cat) in lines.iter().zip(["2000", "2001", "2002"]) {
            let expected = scales.x.to_pixel_str(cat).expect("the ordinal domain holds every row's category");
            assert_eq!(line.0, expected, "span must sit at {cat}'s band center");
            assert_eq!(line.0, line.2, "a vertical span keeps one x for both endpoints");
        }
    }

    /// Float64 column on an ordinal y-scale: the mirror of the Int64 case, on
    /// the other axis. `col_as_positional_category_str` formats an
    /// integer-valued float the way the ordinal domain does (`2000.0` →
    /// `"2000"`), so the per-row strings always match the domain entries.
    #[test]
    fn y_span_float_ordinal_column_renders_at_band_centers() {
        use crate::spec::encoding::DataType as SDT;
        let spec = rule_spec(None, Some(("score", Some(SDT::Ordinal))), None, None);
        let scale_spec = rule_spec(
            Some(("score", Some(SDT::Ordinal))),
            Some(("score", Some(SDT::Ordinal))),
            None,
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0_f64, 2.5])),
        ]).unwrap();

        let (lines, scales) = build_rule_lines(&spec, &scale_spec, &batch);
        assert_eq!(lines.len(), 2,
            "a Float64 column on an ordinal scale must render one span per row — previously 0, silently");
        for (line, cat) in lines.iter().zip(["1", "2.5"]) {
            let expected = scales.y.to_pixel_str(cat).expect("the ordinal domain holds every row's category");
            assert_eq!(line.1, expected, "span must sit at {cat}'s band center");
            assert_eq!(line.1, line.3, "a horizontal span keeps one y for both endpoints");
        }
    }

    /// The ranged shape's anchor takes the same scale-keyed read: an Int64
    /// ordinal `x` with numeric `y`/`y2` renders one vertical segment per row
    /// at the band center (`fm.X('year', type_='ordinal')` + `y=`/`y2=`
    /// produced zero lines pre-fix).
    #[test]
    fn ranged_vertical_int_ordinal_anchor_renders_at_band_centers() {
        use crate::spec::encoding::DataType as SDT;
        let spec = rule_spec(
            Some(("year", Some(SDT::Ordinal))),
            Some(("lo", None)),
            None,
            Some(("hi", None)),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("year", DataType::Int64, false),
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::Int64Array::from(vec![2000_i64, 2001])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0])),
        ]).unwrap();

        let (lines, scales) = build_rule_lines(&spec, &spec, &batch);
        assert_eq!(lines.len(), 2, "an Int64 ordinal anchor must render both ranged segments");
        for (line, cat) in lines.iter().zip(["2000", "2001"]) {
            assert_eq!(line.0, scales.x.to_pixel_str(cat).expect("band center"),
                "segment must be anchored at {cat}'s band center");
            assert_eq!(line.0, line.2, "a vertical segment keeps one x for both endpoints");
            assert_ne!(line.1, line.3, "the segment must span lo..hi, not collapse");
        }
    }

    /// Boolean is a legal ordinal category dtype everywhere else in this crate
    /// (`col_as_ordinal_category_str` maps it to `"true"`/`"false"`, and
    /// `mark_point`/`mark_bar` position a Boolean `x` fine). Rule now gives it
    /// band positions too, instead of the `UnsupportedDtype` refusal a
    /// dtype-keyed dispatch raised for a shape the refusal message itself
    /// advertises as supported.
    #[test]
    fn x_span_boolean_column_renders_band_positions_like_point() {
        let spec = rule_spec(Some(("flag", None)), None, None, None);
        let scale_spec = rule_spec(Some(("flag", None)), Some(("flag", None)), None, None);
        let schema = Arc::new(Schema::new(vec![Field::new("flag", DataType::Boolean, false)]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::BooleanArray::from(vec![true, false, true])),
        ]).unwrap();

        let (lines, scales) = build_rule_lines(&spec, &scale_spec, &batch);
        assert_eq!(lines.len(), 3, "a Boolean anchor must render one span per row, not raise");
        for (line, cat) in lines.iter().zip(["true", "false", "true"]) {
            assert_eq!(line.0, scales.x.to_pixel_str(cat).expect("'true'/'false' are the ordinal domain"),
                "span must sit at {cat}'s band center");
        }
    }

    /// A NULL ordinal anchor row bands at the null category, exactly as
    /// `mark_point`/`mark_bar` position it (FA-9) — spec §4.4, "Extended
    /// 2026-09-02, T13 quality review", ruling 2. This is a deliberate
    /// behavior change: rule's dtype-keyed read used `col_as_str`, whose
    /// `None` for a null row silently SKIPPED it, so
    /// `mark_rule().encode(x=fm.X("cat", type_="ordinal"), y="lo", y2="hi")`
    /// over a `cat` column with one null drew 2 segments where the ordinal
    /// axis showed 3 bands (`"a"`, `"b"`, and the `null` band every sibling
    /// mark already positions rows in). It now draws 3.
    ///
    /// RED (verified in place): read the anchor through `col_as_str` instead
    /// of `col_as_positional_category_str` in `positional_pixels` and this
    /// fails with 2 lines — the null row silently absent from a scale that
    /// reserved a band for it.
    #[test]
    fn null_ordinal_anchor_bands_at_the_null_category_like_point() {
        use crate::spec::encoding::DataType as SDT;
        let spec = rule_spec(
            Some(("cat", Some(SDT::Ordinal))),
            Some(("lo", None)),
            None,
            Some(("hi", None)),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![5.0, 8.0, 9.0])),
        ]).unwrap();

        let (lines, scales) = build_rule_lines(&spec, &spec, &batch);
        assert_eq!(lines.len(), 3,
            "the null anchor row must band at the null category, not be skipped (2 lines)");
        for (line, cat) in lines.iter().zip(["a", "b", crate::render::arrow_cast::NULL_CATEGORY]) {
            let expected = scales.x.to_pixel_str(cat)
                .expect("the ordinal domain carries a band for the null category too");
            assert_eq!(line.0, expected, "segment must be anchored at {cat}'s band center");
            assert_eq!(line.0, line.2, "a vertical segment keeps one x for both endpoints");
        }
    }

    /// Timestamp reads numerically through the time scale — nothing about
    /// Timestamp raises (the module doc used to claim it did, while the live
    /// chart rendered).
    #[test]
    fn x_span_timestamp_column_renders_through_the_time_scale() {
        use arrow::datatypes::TimeUnit;
        let spec = rule_spec(Some(("t", None)), None, None, None);
        let scale_spec = rule_spec(Some(("t", None)), Some(("v", None)), None, None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("t", DataType::Timestamp(TimeUnit::Millisecond, None), false),
            Field::new("v", DataType::Float64, false),
        ]));
        let stamps: Vec<i64> = vec![1_600_000_000_000, 1_600_086_400_000];
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::TimestampMillisecondArray::from(stamps.clone())),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ]).unwrap();

        let (lines, scales) = build_rule_lines(&spec, &scale_spec, &batch);
        assert_eq!(lines.len(), 2, "a Timestamp anchor must render one span per row");
        for (line, ts) in lines.iter().zip(stamps) {
            assert_eq!(line.0, scales.x.to_pixel_f64(ts as f64).expect("timestamps map through the time scale"),
                "span must sit at the timestamp's pixel");
            assert_eq!(line.0, line.2, "a vertical span keeps one x for both endpoints");
        }
    }
}
