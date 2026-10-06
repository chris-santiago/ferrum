use ferrum_scene::{
    dash_tapered, dash_tapered_path, Color as FsColor, FillStroke as FsFillStroke, ImageData, PathCmd,
    PathClosure, SceneGraph, SceneNode, StrokeCap, StrokeJoin, StrokeStyle as FsStroke, TaperError,
    TextStyle as FsText,
};

use super::color::from_rgba;
use super::svg::{escape_attr, fmt_f, FillStroke, Stroke, SvgBuffer, TextStyle};
use super::{RenderError, CLIP_ID_PREFIX};
use crate::layout::{Rect, TextAnchor};

pub fn walk_svg(scene: &SceneGraph, embed_fonts: bool) -> Result<String, RenderError> {
    let viewport = Rect {
        x: 0.0,
        y: 0.0,
        w: scene.width,
        h: scene.height,
    };
    let bg = scene.background.map(|c| from_rgba(c.r, c.g, c.b, c.a));
    let mut svg = SvgBuffer::new(viewport, bg, embed_fonts);

    // Chart-level accessibility description — emitted as the first child of
    // <svg> so screen readers encounter it before any visual content.
    if let Some(ref desc) = scene.chart_description {
        svg.desc_elem(desc);
    }

    // Chart title (before panels, matching old render_svg order)
    for node in &scene.title {
        emit_node(&mut svg, node)?;
    }

    for panel in &scene.panels {
        // Ratio-fitted cells (JointChart/ClusterMap marginals) carry a
        // non-identity `layout_scale` mapping this panel's natively-computed
        // geometry into its allocated slot in the composed scene — the same
        // mechanism `scene_load.rs` applies for the WASM renderer (W5 fix).
        // At identity (every flat/faceted panel today) this is a pure no-op:
        // no wrapper is emitted, keeping byte-identical output.
        let has_layout_scale = !panel.layout_scale.is_identity();
        if has_layout_scale {
            let ls = &panel.layout_scale;
            svg.raw(&format!(
                r#"<g transform="translate({},{}) scale({},{})">"#,
                fmt_f(ls.tx),
                fmt_f(ls.ty),
                fmt_f(ls.sx),
                fmt_f(ls.sy),
            ));
        }

        // Gridlines (behind marks)
        for node in &panel.grid {
            emit_gridline_node(&mut svg, node)?;
        }

        // Below-marks content (GH #89B): text annotations with `z ==
        // "below_marks"`. A typed sibling of `grid`, painted immediately
        // after it — the same visual position these nodes held before this
        // slot existed (previously appended onto `grid` itself).
        for node in &panel.below_marks {
            emit_node(&mut svg, node)?;
        }

        // Axes
        for node in &panel.axes {
            emit_node(&mut svg, node)?;
        }

        // Strip title (outside clip, before marks)
        for node in &panel.strip_title {
            emit_node(&mut svg, node)?;
        }

        // Clip open
        let clip_id = format!("{}{}", CLIP_ID_PREFIX, panel.id);
        let clip_rect = Rect {
            x: panel.clip.x,
            y: panel.clip.y,
            w: panel.clip.w,
            h: panel.clip.h,
        };
        svg.clip_open(&clip_id, clip_rect);

        // Marks (non-text/label first, inside clip). A clip-exempt batch
        // (`MarkBatch::clip == false`, batch E Task 9) is skipped here — it
        // does NOT keep its relative position among non-exempt marks; every
        // clip-exempt batch is deferred to the separate pass below, which
        // paints ALL of them above ALL clipped batches (see that pass's
        // comment for the full order and the test pinning it).
        let clipped = panel.marks.iter().filter(|batch| {
            batch.clip
                && !matches!(
                    batch.kind,
                    ferrum_scene::MarkBatchKind::Text | ferrum_scene::MarkBatchKind::Label
                )
        });
        emit_clipped_marks(&mut svg, clipped, &clip_id)?;

        // Clip-exempt marks (batch E Task 9, `mark_function(clip=False)`):
        // rendered OUTSIDE the panel clip region, mirroring the existing
        // Text/Label placement precedent below. Painted immediately after ALL
        // clipped marks close and before `chrome_above`, `annotations`, and
        // Text/Label batches — above every clipped data batch, below
        // annotation/text chrome, a defined order pinned by
        // `clip_exempt_batch_paints_after_clipped_marks_and_before_chrome_annotations_and_text`
        // (svg_walk.rs test module).
        for batch in &panel.marks {
            if matches!(
                batch.kind,
                ferrum_scene::MarkBatchKind::Text | ferrum_scene::MarkBatchKind::Label
            ) || batch.clip
            {
                continue;
            }
            emit_mark_batch(&mut svg, batch, None)?;
        }

        // Above-marks axis/grid chrome (GH #89B): nodes from an axis or its
        // gridlines whose effective `zindex >= 1`. A typed sibling of
        // `annotations`, painted immediately after marks and before
        // `annotations` — the deliberate z-order refinement that makes
        // above-marks user annotations always paint above above-marks axis
        // chrome (previously the two were commingled in one list with chrome
        // prefixed ahead of user content, so chrome painted OVER it).
        for node in &panel.chrome_above {
            emit_node(&mut svg, node)?;
        }

        // Annotations (reference lines, hlines, vlines) — emitted OUTSIDE the
        // clip region so they can span the full panel width/height, and AFTER
        // marks so they render above data (matching WASM z-order in scene_load.rs).
        for node in &panel.annotations {
            emit_node(&mut svg, node)?;
        }

        // Text/Label-kind mark batches (mark_text, mark_label) outside clip so
        // dy/dx offsets near panel edges are not cut off.
        for batch in &panel.marks {
            if matches!(
                batch.kind,
                ferrum_scene::MarkBatchKind::Text | ferrum_scene::MarkBatchKind::Label
            ) {
                emit_mark_batch(&mut svg, batch, None)?;
            }
        }

        if has_layout_scale {
            svg.raw("</g>");
        }
    }

    // Legend (after panels, matching old render_svg order)
    for node in &scene.legend {
        emit_node(&mut svg, node)?;
    }

    // Any remaining decorations
    for node in &scene.decorations {
        emit_node(&mut svg, node)?;
    }

    Ok(svg.finish())
}

/// Emit a panel's clipped mark batches, in order, under the clip path
/// `clip_id`. Unblended batches draw in the panel's clip group. A `clip-path`
/// group isolates its children, so a blended batch draws outside that group,
/// clipped inside its own blend group (see [`emit_mark_batch`]); the panel's
/// clip group closes before it and reopens only for a following unblended
/// batch, so no clip group is left empty. A panel without a blended batch
/// draws one clip group, empty when it has no clipped batch.
fn emit_clipped_marks<'a>(
    svg: &mut SvgBuffer,
    batches: impl Iterator<Item = &'a ferrum_scene::MarkBatch> + Clone,
    clip_id: &str,
) -> Result<(), TaperError> {
    let mut clip_group_open = !batches.clone().any(blends);
    if clip_group_open {
        svg.use_clip_open(clip_id);
    }
    for batch in batches {
        let blended = blends(batch);
        if blended && clip_group_open {
            svg.use_clip_close();
            clip_group_open = false;
        } else if !blended && !clip_group_open {
            svg.use_clip_open(clip_id);
            clip_group_open = true;
        }
        emit_mark_batch(svg, batch, blended.then_some(clip_id))?;
    }
    if clip_group_open {
        svg.use_clip_close();
    }
    Ok(())
}

/// Emit one mark batch: the additive-blend wrapper, the stroke-cap/stroke-join
/// wrapper, and each node's href/tooltip/description wrap. Shared by the
/// clipped-marks pass, the clip-exempt escape pass (`MarkBatch::clip ==
/// false`) and the text/label pass in `walk_svg`, so every batch kind wraps
/// identically; only their placement around the clip group differs. A text or
/// label batch carries no stroke cap or join, so it gets no cap/join wrapper.
///
/// An additive batch draws inside `<g style="mix-blend-mode:screen">`, the
/// nearest compositing mode SVG renderers draw. A `clip-path` group isolates
/// its children, so a blend group inside the panel's clip group would blend
/// with nothing beneath it and draw normal; a blended batch therefore sits
/// outside that group, and `clip` names the clip path it draws under inside
/// its own blend group. `clip` is `None` for every other batch.
fn emit_mark_batch(
    svg: &mut SvgBuffer,
    batch: &ferrum_scene::MarkBatch,
    clip: Option<&str>,
) -> Result<(), TaperError> {
    let blend_wrap = blends(batch);
    if blend_wrap {
        svg.raw("<g style=\"mix-blend-mode:screen\">");
    }
    if let Some(clip_id) = clip {
        svg.use_clip_open(clip_id);
    }
    // Wrap batch in <g stroke-linecap/stroke-linejoin> when present,
    // matching the old draw() pattern.
    let cap_join_wrap = if batch.stroke_cap.is_some() || batch.stroke_join.is_some() {
        let mut g_attrs = String::new();
        if let Some(ref cap) = batch.stroke_cap {
            let s = match cap {
                ferrum_scene::StrokeCap::Butt => "butt",
                ferrum_scene::StrokeCap::Round => "round",
                ferrum_scene::StrokeCap::Square => "square",
            };
            g_attrs.push_str(&format!(" stroke-linecap=\"{}\"", s));
        }
        if let Some(ref join) = batch.stroke_join {
            let s = match join {
                ferrum_scene::StrokeJoin::Miter => "miter",
                ferrum_scene::StrokeJoin::Round => "round",
                ferrum_scene::StrokeJoin::Bevel => "bevel",
            };
            g_attrs.push_str(&format!(" stroke-linejoin=\"{}\"", s));
        }
        Some(g_attrs)
    } else {
        None
    };
    if let Some(ref attrs) = cap_join_wrap {
        svg.raw(&format!("<g{}>", attrs));
    }
    let ends = BatchEnds { cap: batch.stroke_cap, join: batch.stroke_join };
    for (i, node) in batch.nodes.iter().enumerate() {
        let href = batch
            .hrefs
            .as_ref()
            .and_then(|h| h.get(i).and_then(|o| o.as_deref()));
        let tooltip = batch.tooltips.as_ref().and_then(|t| t.get(i));
        let desc = batch
            .descriptions
            .as_ref()
            .and_then(|d| d.get(i).and_then(|o| o.as_deref()));
        let needs_wrap = href.is_some() || tooltip.is_some() || desc.is_some();

        if needs_wrap {
            if let Some(url) = href {
                svg.a_open(url);
            }
            svg.g_open(None);
            if let Some(tt) = tooltip {
                let text: String = tt
                    .fields
                    .iter()
                    .map(|f| format!("{}: {}", f.name, f.value))
                    .collect::<Vec<_>>()
                    .join(", ");
                if !text.is_empty() {
                    svg.title_elem(&text);
                }
            }
            if let Some(d) = desc {
                svg.desc_elem(d);
            }
            emit_node_in(svg, node, ends)?;
            svg.g_close();
            if href.is_some() {
                svg.a_close();
            }
        } else {
            emit_node_in(svg, node, ends)?;
        }
    }
    if cap_join_wrap.is_some() {
        svg.g_close();
    }
    if clip.is_some() {
        svg.use_clip_close();
    }
    if blend_wrap {
        svg.raw("</g>");
    }
    Ok(())
}

/// Whether `batch` composites over what is beneath it rather than drawing
/// normal.
fn blends(batch: &ferrum_scene::MarkBatch) -> bool {
    matches!(batch.blend, ferrum_scene::BlendMode::Additive)
}

/// The stroke cap and join a mark batch's `<g>` wrapper gives the nodes
/// inside it. A stroked element inherits them as attributes; a tapered
/// outline is a fill, so it draws them into its geometry instead.
#[derive(Debug, Clone, Copy, Default)]
struct BatchEnds {
    cap: Option<StrokeCap>,
    join: Option<StrokeJoin>,
}

fn emit_node(svg: &mut SvgBuffer, node: &SceneNode) -> Result<(), TaperError> {
    emit_node_in(svg, node, BatchEnds::default())
}

fn emit_node_in(svg: &mut SvgBuffer, node: &SceneNode, ends: BatchEnds) -> Result<(), TaperError> {
    match node {
        SceneNode::Rect {
            x,
            y,
            w,
            h,
            style,
            corner_radius,
        } => {
            let cr = if *corner_radius > 0.0 {
                Some(*corner_radius)
            } else {
                None
            };
            // Anchor: center of the rect.
            let svg_style = to_svg_fill_stroke_with_anchor(style, x + w / 2.0, y + h / 2.0);
            svg.rect(
                Rect {
                    x: *x,
                    y: *y,
                    w: *w,
                    h: *h,
                },
                &svg_style,
                cr,
            );
        }
        SceneNode::Circle { cx, cy, r, style } => {
            // Anchor: circle center.
            let svg_style = to_svg_fill_stroke_with_anchor(style, *cx, *cy);
            svg.circle(*cx, *cy, *r, &svg_style);
        }
        SceneNode::Line {
            x1,
            y1,
            x2,
            y2,
            style,
        } => {
            svg.line(*x1, *y1, *x2, *y2, &to_svg_stroke(style));
        }
        SceneNode::Path {
            commands,
            style,
            closed,
            widths: Some(widths),
        } => {
            let dash = style.stroke_dash.as_deref().unwrap_or(&[]);
            let pieces = dash_tapered_path(commands, PathClosure::from(*closed), widths, dash)?;
            let (anchor_x, anchor_y) = path_anchor(commands);
            let paint = outline_paint(style.stroke, style.stroke_opacity, style.angle, anchor_x, anchor_y);
            svg.tapered_stroke(&pieces, stroke_ends(None, None, ends), &paint);
        }
        SceneNode::Path {
            commands,
            style,
            closed: _,
            widths: None,
        } => {
            let d = path_cmds_to_d(commands);
            let (anchor_x, anchor_y) = path_anchor(commands);
            let svg_style = to_svg_fill_stroke_with_anchor(style, anchor_x, anchor_y);
            svg.path(&d, &svg_style);
        }
        SceneNode::Text {
            x,
            y,
            content,
            style,
            ..
        } => {
            emit_text(svg, *x, *y, content, style);
        }
        SceneNode::Image { x, y, w, h, data } => match data {
            ImageData::Inline { bytes, mime } => {
                svg.image(*x, *y, *w, *h, bytes, *mime);
            }
            ImageData::Url { url } => {
                svg.image_data_url(*x, *y, *w, *h, url);
            }
        },
        SceneNode::Polyline { points, style, widths: Some(widths) } => {
            let pieces = dash_tapered(points, widths, style.dash.as_deref().unwrap_or(&[]))?;
            let paint = outline_paint(style.color, style.stroke_opacity, 0.0, 0.0, 0.0);
            svg.tapered_stroke(&pieces, stroke_ends(style.stroke_cap, style.stroke_join, ends), &paint);
        }
        SceneNode::Polyline { points, style, widths: None } => {
            svg.polyline(points, &to_svg_stroke(style));
        }
        SceneNode::Polygon { rings, style } => {
            let svg_rings: Vec<Vec<(f64, f64)>> = rings
                .iter()
                .map(|r| r.iter().map(|p| (p[0], p[1])).collect())
                .collect();
            // Anchor: centroid of first ring's points (good enough for rotation anchor).
            let (anchor_x, anchor_y) = if let Some(ring) = rings.first() {
                if ring.is_empty() {
                    (0.0, 0.0)
                } else {
                    let n = ring.len() as f64;
                    let sx: f64 = ring.iter().map(|p| p[0]).sum();
                    let sy: f64 = ring.iter().map(|p| p[1]).sum();
                    (sx / n, sy / n)
                }
            } else {
                (0.0, 0.0)
            };
            svg.polygon(
                &svg_rings,
                &to_svg_fill_stroke_with_anchor(style, anchor_x, anchor_y),
            );
        }
        SceneNode::Group { attrs, children } => {
            if attrs.is_empty() {
                svg.g_open(None);
            } else {
                let attr_str: String = attrs
                    .iter()
                    .map(|(k, v)| format!(" {}=\"{}\"", k, escape_attr(v)))
                    .collect();
                svg.raw(&format!("<g{}>", attr_str));
            }
            for child in children {
                emit_node_in(svg, child, ends)?;
            }
            svg.g_close();
        }
        // `anchor` is intentionally ignored by the static SVG renderer — it is
        // consumed only by the WASM renderer to decide pan/zoom behaviour.
        SceneNode::Raw { svg: raw, .. } => {
            svg.raw(raw);
        }
    }
    Ok(())
}

/// A path's rotation anchor: its first `MoveTo`, else the origin.
fn path_anchor(commands: &[PathCmd]) -> (f64, f64) {
    commands
        .iter()
        .find_map(|c| if let PathCmd::MoveTo { x, y } = c { Some((*x, *y)) } else { None })
        .unwrap_or((0.0, 0.0))
}

/// A tapered stroke's cap and join: the node's own, then the batch's, then
/// SVG's butt cap and the bevel join a variable-width stroke defaults to.
fn stroke_ends(cap: Option<StrokeCap>, join: Option<StrokeJoin>, ends: BatchEnds) -> (StrokeCap, StrokeJoin) {
    (
        cap.or(ends.cap).unwrap_or(StrokeCap::Butt),
        join.or(ends.join).unwrap_or(StrokeJoin::Bevel),
    )
}

/// The fill a tapered outline paints: the stroke's color and, as
/// `fill-opacity`, the opacity the stroke would carry as `stroke-opacity`.
fn outline_paint(color: Option<FsColor>, stroke_opacity: f64, angle: f64, anchor_x: f64, anchor_y: f64) -> FillStroke {
    FillStroke {
        fill: color.map(|c| from_rgba(c.r, c.g, c.b, c.a)),
        fill_opacity: stroke_opacity,
        angle,
        angle_cx: anchor_x,
        angle_cy: anchor_y,
        ..FillStroke::default()
    }
}

fn emit_gridline_node(svg: &mut SvgBuffer, node: &SceneNode) -> Result<(), TaperError> {
    if let SceneNode::Line {
        x1,
        y1,
        x2,
        y2,
        style,
    } = node
    {
        // A cleared paint paints no gridline: `gridline` writes `stroke`
        // unconditionally (its whole purpose is the inline dash/opacity the
        // shared `Stroke` struct does not carry), so the omission happens here
        // rather than inside it.
        let Some(c) = style.color else { return Ok(()) };
        let color = from_rgba(c.r, c.g, c.b, c.a);
        svg.gridline(
            *x1,
            *y1,
            *x2,
            *y2,
            color,
            style.width,
            style.dash.as_deref(),
            style.opacity,
        );
        Ok(())
    } else {
        emit_node(svg, node)
    }
}

fn to_svg_fill_stroke_with_anchor(s: &FsFillStroke, anchor_x: f64, anchor_y: f64) -> FillStroke {
    FillStroke {
        fill: s.fill.map(|c| from_rgba(c.r, c.g, c.b, c.a)),
        stroke: s.stroke.map(|c| from_rgba(c.r, c.g, c.b, c.a)),
        stroke_width: s.stroke_width,
        stroke_opacity: s.stroke_opacity,
        fill_opacity: s.fill_opacity,
        stroke_dash: s.stroke_dash.clone(),
        angle: s.angle,
        angle_cx: anchor_x,
        angle_cy: anchor_y,
    }
}

fn to_svg_stroke(s: &FsStroke) -> Stroke {
    Stroke {
        stroke: s.color.map(|c| from_rgba(c.r, c.g, c.b, c.a)),
        stroke_width: s.width,
        stroke_dash: s.dash.clone(),
        stroke_opacity: s.stroke_opacity,
    }
}

fn emit_text(svg: &mut SvgBuffer, x: f64, y: f64, content: &str, s: &FsText) {
    let anchor = match s.anchor {
        ferrum_scene::TextAnchor::Start => TextAnchor::Start,
        ferrum_scene::TextAnchor::Middle => TextAnchor::Middle,
        ferrum_scene::TextAnchor::End => TextAnchor::End,
    };
    let fw_str: Option<String> = match &s.font_weight {
        ferrum_scene::FontWeight::Bold => Some("bold".to_string()),
        ferrum_scene::FontWeight::Custom(w) => Some(w.clone()),
        ferrum_scene::FontWeight::Normal => None,
    };
    let db_owned: Option<String> = match &s.baseline {
        ferrum_scene::TextBaseline::Top => Some("hanging".to_string()),
        ferrum_scene::TextBaseline::Middle => Some("central".to_string()),
        ferrum_scene::TextBaseline::Bottom => Some("text-after-edge".to_string()),
        ferrum_scene::TextBaseline::Custom(v) => Some(v.clone()),
        ferrum_scene::TextBaseline::Alphabetic => None,
    };
    // Guard NaN opacity: NaN.clamp(0,1) is NaN in Rust, and NaN*255 as u8 = 0,
    // which makes text invisible. Default to 1.0 (fully opaque) when NaN.
    let opacity = if s.opacity.is_nan() { 1.0 } else { s.opacity };
    // A cleared paint (`color: None`) paints no glyphs: `svg.text` writes
    // `fill="none"`, the fill rule, since SVG's default text fill is black.
    let fill = s.color.map(|c| {
        let effective_alpha = ((c.a as f64 / 255.0) * opacity).clamp(0.0, 1.0);
        let alpha_byte = (effective_alpha * 255.0).round() as u8;
        from_rgba(c.r, c.g, c.b, alpha_byte)
    });
    let ts = TextStyle {
        fill,
        font_size: s.font_size,
        anchor,
        angle: s.angle,
        font_family: &s.font_family,
        font_weight: fw_str.as_deref(),
        dominant_baseline: db_owned.as_deref(),
    };
    svg.text(x, y, content, &ts);
}

// Test coverage lives in `tests.rs` (precedent: `projection/tests.rs`): the
// tapered-stroke outline and pipeline-matrix coverage outgrew the walker.
#[cfg(test)]
mod tests;

fn path_cmds_to_d(cmds: &[PathCmd]) -> String {
    let mut d = String::new();
    for cmd in cmds {
        if !d.is_empty() {
            d.push(' ');
        }
        match cmd {
            PathCmd::MoveTo { x, y } => {
                d.push_str(&format!("M{} {}", fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::LineTo { x, y } => {
                d.push_str(&format!("L{} {}", fmt_f(*x), fmt_f(*y)));
            }
            PathCmd::QuadTo { cx, cy, x, y } => {
                d.push_str(&format!(
                    "Q{} {} {} {}",
                    fmt_f(*cx),
                    fmt_f(*cy),
                    fmt_f(*x),
                    fmt_f(*y)
                ));
            }
            PathCmd::CubicTo {
                c1x,
                c1y,
                c2x,
                c2y,
                x,
                y,
            } => {
                d.push_str(&format!(
                    "C{} {} {} {} {} {}",
                    fmt_f(*c1x),
                    fmt_f(*c1y),
                    fmt_f(*c2x),
                    fmt_f(*c2y),
                    fmt_f(*x),
                    fmt_f(*y)
                ));
            }
            PathCmd::HLineTo { x } => {
                d.push_str(&format!("H{}", fmt_f(*x)));
            }
            PathCmd::VLineTo { y } => {
                d.push_str(&format!("V{}", fmt_f(*y)));
            }
            PathCmd::ArcTo {
                rx,
                ry,
                rotation,
                large_arc,
                sweep,
                x,
                y,
            } => {
                d.push_str(&format!(
                    "A{} {} {} {} {} {} {}",
                    fmt_f(*rx),
                    fmt_f(*ry),
                    fmt_f(*rotation),
                    if *large_arc { 1 } else { 0 },
                    if *sweep { 1 } else { 0 },
                    fmt_f(*x),
                    fmt_f(*y)
                ));
            }
            PathCmd::Close => {
                d.push('Z');
            }
        }
    }
    d
}
