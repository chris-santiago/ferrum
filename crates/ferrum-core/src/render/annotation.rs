//! Annotation rendering: converts annotation specs from the Python chart_config
//! into SceneNode elements positioned within a panel's plot area.
//!
//! Annotations are resolved against the panel's data-space scales (for data
//! coordinates) or directly against the pixel/normalized plot-area extent.

use ferrum_scene::{
    Color, FillStroke, FontWeight, RawAnchor, SceneNode, StrokeStyle, TextAnchor, TextBaseline,
    TextStyle,
};
use serde::Deserialize;

use super::break_axis;
use super::color::parse_color as parse_color_str;
use super::draw::{arrowhead_path, arrowhead_triangle, to_scene_color};
use super::scale_resolve::ScaleKind;
use crate::layout::geometry::Rect;

// ── Coordinate types ────────────────────────────────────────────────────────

/// A coordinate value that can be expressed in data-space, pixel offset from
/// plot-area origin, or normalized [0,1] fraction of the plot-area extent.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CoordValue {
    /// `{"px": 100.0}` — pixel offset from the plot-area origin.
    Pixel { px: f64 },
    /// `{"norm": 0.5}` — normalized fraction [0,1] of the plot-area extent.
    Norm { norm: f64 },
    /// Plain number — data-space value mapped through the panel's scale.
    Data(f64),
}

// ── Annotation spec ─────────────────────────────────────────────────────────

/// One annotation spec, deserialized from the chart_config `annotations` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnnotationSpec {
    #[serde(rename = "text")]
    Text {
        x: CoordValue,
        y: CoordValue,
        text: String,
        #[serde(default = "default_font_size")]
        font_size: f64,
        #[serde(default = "default_color_str")]
        color: String,
        #[serde(default = "default_anchor")]
        anchor: String,
        #[serde(default = "default_baseline")]
        baseline: String,
        #[serde(default)]
        angle: f64,
        #[serde(default)]
        dx: f64,
        #[serde(default)]
        dy: f64,
        /// Routes the text annotation into the below-marks (`grid`) or above-marks
        /// (`annotations`) panel bucket. `"below_marks"` emits the node before the
        /// data marks (into the same slot as gridlines); any other value (including
        /// the default `"above_marks"`) emits after the data marks.
        #[serde(default = "default_z")]
        z: String,
    },
    #[serde(rename = "arrow")]
    Arrow {
        x: CoordValue,
        y: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default = "default_color_str")]
        stroke: String,
        #[serde(default = "default_stroke_width")]
        stroke_width: f64,
        #[serde(default = "default_head_size")]
        head_size: f64,
    },
    #[serde(rename = "rect")]
    Rect {
        x1: CoordValue,
        y1: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default = "default_fill_str")]
        fill: String,
        #[serde(default = "default_rect_opacity")]
        opacity: f64,
        #[serde(default)]
        corner_radius: f64,
        #[serde(default)]
        stroke: Option<String>,
    },
    #[serde(rename = "line")]
    Line {
        x1: CoordValue,
        y1: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default = "default_color_str")]
        stroke: String,
        #[serde(default = "default_stroke_width")]
        stroke_width: f64,
        #[serde(default)]
        dash: Option<Vec<f64>>,
    },
    #[serde(rename = "span")]
    Span {
        axis: String,
        start: CoordValue,
        end: CoordValue,
        #[serde(default = "default_fill_str")]
        fill: String,
        #[serde(default = "default_span_opacity")]
        opacity: f64,
        #[serde(default = "default_label_position")]
        label_position: String,
        #[serde(default)]
        label: Option<String>,
    },
    #[serde(rename = "bracket")]
    Bracket {
        x1: CoordValue,
        y1: CoordValue,
        x2: CoordValue,
        y2: CoordValue,
        #[serde(default)]
        label: String,
        #[serde(default = "default_direction")]
        direction: String,
        #[serde(default = "default_color_str")]
        stroke: String,
        #[serde(default = "default_tip_length")]
        tip_length: f64,
    },
    #[serde(rename = "callout")]
    Callout {
        x: CoordValue,
        y: CoordValue,
        text: String,
        #[serde(default = "default_arrow_str")]
        arrow: String,
        #[serde(default = "default_callout_padding")]
        padding: f64,
        #[serde(default = "default_background_str")]
        background: String,
        #[serde(default = "default_border_color_str")]
        border_color: String,
        #[serde(default)]
        border_radius: f64,
        #[serde(default)]
        text_x: Option<CoordValue>,
        #[serde(default)]
        text_y: Option<CoordValue>,
    },
    #[serde(rename = "image")]
    Image {
        x: CoordValue,
        y: CoordValue,
        src: String,
        #[serde(default = "default_image_size")]
        width: f64,
        #[serde(default = "default_image_size")]
        height: f64,
        #[serde(default = "default_anchor")]
        anchor: String,
    },
}

// ── Serde defaults ──────────────────────────────────────────────────────────

fn default_font_size() -> f64 {
    12.0
}
fn default_color_str() -> String {
    "#333333".to_string()
}
fn default_anchor() -> String {
    "middle".to_string()
}
fn default_baseline() -> String {
    "middle".to_string()
}
fn default_z() -> String {
    "above_marks".to_string()
}
fn default_stroke_width() -> f64 {
    1.5
}
fn default_head_size() -> f64 {
    8.0
}
fn default_fill_str() -> String {
    "#cccccc".to_string()
}
fn default_rect_opacity() -> f64 {
    0.3
}
fn default_span_opacity() -> f64 {
    0.2
}
fn default_label_position() -> String {
    "center".to_string()
}
fn default_direction() -> String {
    "up".to_string()
}
fn default_tip_length() -> f64 {
    6.0
}
fn default_arrow_str() -> String {
    "curved".to_string()
}
fn default_callout_padding() -> f64 {
    4.0
}
fn default_background_str() -> String {
    "#ffffff".to_string()
}
fn default_border_color_str() -> String {
    "#333333".to_string()
}
fn default_image_size() -> f64 {
    50.0
}

// ── Scale context ───────────────────────────────────────────────────────────

/// Provides coordinate resolution from data/pixel/norm to absolute pixel position.
pub struct ScaleContext<'a> {
    pub plot_area: Rect,
    pub x_scale: &'a ScaleKind,
    pub y_scale: &'a ScaleKind,
    /// Every `BreakAxis` remap active on this panel, x and y mixed together —
    /// the SAME list `scene_build`'s mark pass folds through (spec §4 H5's
    /// "same mapping as marks" contract), already filtered to axes whose
    /// resolved scale has a continuous `data_domain()` (an ordinal axis has
    /// nothing to reverse-map through, so it contributes no remap there
    /// either — see `remap_mark_batch_through_break`'s caller in
    /// `scene_build.rs`). Empty for every chart without an active break — the
    /// byte-identical default.
    pub breaks: &'a [break_axis::BreakRemap<'a>],
}

impl<'a> ScaleContext<'a> {
    /// Resolve an x-axis coordinate value to absolute pixel position.
    pub fn resolve_x(&self, v: &CoordValue) -> f64 {
        match v {
            CoordValue::Data(d) => {
                let base = self.x_scale.to_pixel_f64(*d).unwrap_or_else(|| {
                    // Fallback: linear interpolation using data domain. Unlike
                    // `resolve_y`'s fallback, this needs no flip: `axis_pixel_
                    // range` only swaps the pixel range for the y channel
                    // (`channel == "y"`), so a continuous x axis's domain[0]
                    // already maps to the LEFT pixel, matching `frac` directly.
                    if let Some((lo, hi)) = self.x_scale.data_domain() {
                        let frac = if (hi - lo).abs() < f64::EPSILON {
                            0.5
                        } else {
                            (d - lo) / (hi - lo)
                        };
                        self.plot_area.x + frac * self.plot_area.w
                    } else {
                        // Ordinal: cannot resolve numeric data to ordinal scale; center.
                        self.plot_area.x + self.plot_area.w * 0.5
                    }
                });
                self.fold_breaks("x", base)
            }
            CoordValue::Pixel { px } => self.plot_area.x + px,
            CoordValue::Norm { norm } => self.plot_area.x + norm * self.plot_area.w,
        }
    }

    /// Resolve a y-axis coordinate value to absolute pixel position.
    ///
    /// The y-scale already maps domain [min, max] to pixel range [top, bottom]
    /// (SVG convention: Y increases downward), so `to_pixel_f64` returns the
    /// correct absolute pixel directly.
    pub fn resolve_y(&self, v: &CoordValue) -> f64 {
        match v {
            CoordValue::Data(d) => {
                let base = self.y_scale.to_pixel_f64(*d).unwrap_or_else(|| {
                    // Fallback: linear interpolation using data domain, for a
                    // continuous y scale's OUT-OF-DOMAIN value (`to_pixel_f64`
                    // returns `None` there by design, not only for ordinal
                    // scales — see its doc comment).
                    //
                    // Every non-ordinal y axis's pixel RANGE is flipped
                    // relative to its domain, unconditionally
                    // (`scale_resolve::positional::axis_pixel_range`:
                    // domain[0] -> the BOTTOM pixel, domain[1] -> the TOP —
                    // `reverse=` only swaps which value sits at domain[0] vs
                    // domain[1], never which pixel each endpoint maps to, so
                    // `data_domain()`'s `(lo, hi)` already encodes `reverse=`
                    // and this flip composes with it for free). `frac` is the
                    // fraction of the domain span past `lo`, so `(1.0 -
                    // frac)` is that SAME flip: frac=0 (at `lo`) lands at the
                    // bottom, frac=1 (at `hi`) at the top — matching every
                    // in-domain annotation and every mark on this axis.
                    //
                    // Before this fix the formula used `frac` directly (no
                    // flip), so an out-of-domain value landed on the WRONG
                    // side once a `BreakAxis` clamped it back onto the
                    // visible canvas (quality-review cycle-3 finding: y=5000
                    // above a domain max of 1000 rendered at the plot BOTTOM
                    // instead of the TOP). Not separately audited under
                    // `CoordFlip` — that swap happens upstream of
                    // `ScaleContext`, which this module does not trace, and
                    // this fallback only fires for the narrow
                    // out-of-domain-annotation edge case.
                    if let Some((lo, hi)) = self.y_scale.data_domain() {
                        let frac = if (hi - lo).abs() < f64::EPSILON {
                            0.5
                        } else {
                            (d - lo) / (hi - lo)
                        };
                        self.plot_area.y + (1.0 - frac) * self.plot_area.h
                    } else {
                        self.plot_area.y + self.plot_area.h * 0.5
                    }
                });
                self.fold_breaks("y", base)
            }
            CoordValue::Pixel { px } => self.plot_area.y + px,
            CoordValue::Norm { norm } => self.plot_area.y + norm * self.plot_area.h,
        }
    }

    /// Fold `base` (a pixel already resolved against the UNBROKEN `x_scale`/
    /// `y_scale`) through every `axis`-matching break, in declaration order.
    ///
    /// This is exactly `remap_mark_batch_through_break`'s sequential
    /// composition (`for remap in &break_remaps { ... remap.coord(px) ... }`)
    /// applied to one coordinate — reused here rather than re-derived, so two
    /// same-axis breaks move a surviving annotation and a mark to the same
    /// pixel instead of the annotation silently picking only the first one.
    /// A step that cannot place `base` (it round-trips to that break's own
    /// gap) hides the coordinate at `BREAK_HIDDEN` — the same fallback a
    /// mark's `Circle`/`Text` node gets from `BreakRemap::remap_node` —
    /// rather than silently reverting to the unbroken position. This also
    /// covers a genuinely out-of-domain data value (not just an in-gap one):
    /// `BreakRemap::coord` clamps its reverse-mapped data value to the
    /// unbroken domain before forward-mapping, so an out-of-domain
    /// coordinate lands at the compressed domain edge rather than at its
    /// (silently wrong) unbroken linear position.
    fn fold_breaks(&self, axis: &str, base: f64) -> f64 {
        self.breaks
            .iter()
            .filter(|remap| remap.axis() == axis)
            .fold(base, |px, remap| {
                remap.coord(px).unwrap_or(break_axis::BREAK_HIDDEN)
            })
    }
}

// ── Break-axis suppression support (spec §4 H5, F-L08-07) ───────────────────

/// The wire `type` tag for `spec` — the annotation "kind" named in
/// `RenderWarning::AnnotationSuppressedByBreak`.
///
/// Mirrors the `#[serde(tag = "type")]` discriminant on `AnnotationSpec`
/// above via an exhaustive match, so a new spec variant is a compile error
/// here rather than silently falling through to an untagged warning.
pub fn kind_name(spec: &AnnotationSpec) -> &'static str {
    match spec {
        AnnotationSpec::Text { .. } => "text",
        AnnotationSpec::Arrow { .. } => "arrow",
        AnnotationSpec::Rect { .. } => "rect",
        AnnotationSpec::Line { .. } => "line",
        AnnotationSpec::Span { .. } => "span",
        AnnotationSpec::Bracket { .. } => "bracket",
        AnnotationSpec::Callout { .. } => "callout",
        AnnotationSpec::Image { .. } => "image",
    }
}

/// The data-space (`CoordValue::Data`) coordinate values `spec` carries along
/// `axis` (`"x"` or `"y"`).
///
/// Pixel/norm coordinates are plot-area-relative, not data positions, so a
/// `BreakAxis` gap never suppresses them — they are excluded here rather than
/// filtered by the caller, so every caller gets that rule for free.
///
/// Exhaustive over `AnnotationSpec`'s 8 shapes, enumerated from this file's
/// own field list (spec §4 H5's coordinate table), so a new spec variant is a
/// compile error here rather than silently skipping break suppression for it.
pub fn data_coords_on_axis(spec: &AnnotationSpec, axis: &str) -> Vec<f64> {
    // Branches on `axis == "y"` with x as the fallback, matching every other
    // break-axis consumer's convention (`break_axis.rs`'s `is_y()`,
    // `scene_build.rs`'s `if axis == "y" { &scales.y } else { &scales.x }`) —
    // not S2 today (`BreakAxis.axis` is validated to `("x", "y")` in Python),
    // but keeping one direction crate-wide means a future third axis value
    // can't be suppressed as one axis and remapped as the other.
    let is_y = axis == "y";
    let fields: Vec<&CoordValue> = match spec {
        AnnotationSpec::Text { x, y, .. } => vec![if is_y { y } else { x }],
        AnnotationSpec::Arrow { x, y, x2, y2, .. } => {
            if is_y { vec![y, y2] } else { vec![x, x2] }
        }
        AnnotationSpec::Rect { x1, y1, x2, y2, .. } => {
            if is_y { vec![y1, y2] } else { vec![x1, x2] }
        }
        AnnotationSpec::Line { x1, y1, x2, y2, .. } => {
            if is_y { vec![y1, y2] } else { vec![x1, x2] }
        }
        // A `Span`'s `start`/`end` are values on its OWN `axis` field, which
        // may differ from the axis under test — an x-span is untouched by a
        // y-break, and vice versa.
        AnnotationSpec::Span { axis: span_axis, start, end, .. } => {
            if span_axis == axis { vec![start, end] } else { Vec::new() }
        }
        AnnotationSpec::Bracket { x1, y1, x2, y2, .. } => {
            if is_y { vec![y1, y2] } else { vec![x1, x2] }
        }
        AnnotationSpec::Callout { x, y, text_x, text_y, .. } => {
            let mut v = vec![if is_y { y } else { x }];
            if is_y {
                if let Some(ty) = text_y {
                    v.push(ty);
                }
            } else if let Some(tx) = text_x {
                v.push(tx);
            }
            v
        }
        AnnotationSpec::Image { x, y, .. } => vec![if is_y { y } else { x }],
    };
    fields
        .into_iter()
        .filter_map(|v| match v {
            CoordValue::Data(d) => Some(*d),
            _ => None,
        })
        .collect()
}

// ── Color helpers ───────────────────────────────────────────────────────────

/// Parse a color string to a scene Color. Falls back to dark gray (#333) on failure.
fn resolve_color(s: &str) -> Color {
    match parse_color_str(s) {
        Ok(c) => to_scene_color(c),
        Err(_) => Color::rgb(51, 51, 51),
    }
}

// ── Anchor / baseline parsing ───────────────────────────────────────────────

fn parse_anchor(s: &str) -> TextAnchor {
    match s {
        "start" | "left" => TextAnchor::Start,
        "end" | "right" => TextAnchor::End,
        _ => TextAnchor::Middle,
    }
}

fn parse_baseline(s: &str) -> TextBaseline {
    // Route through the canonical parser; unrecognized strings default to Middle
    // (the annotation-layer convention for baseline-less text placement).
    super::draw::parse_text_baseline(s).unwrap_or(TextBaseline::Middle)
}

// ── Build annotations ───────────────────────────────────────────────────────

/// Partitioned result from `build_annotations`.
///
/// `below_marks` contains nodes that route into the panel's typed
/// `below_marks` slot (GH #89B; painted immediately after `grid`, before data
/// marks).  `above_marks` contains nodes that route into the post-marks
/// `annotations` slot (painted after marks, and after any `chrome_above`).
pub struct AnnotationNodes {
    pub below_marks: Vec<SceneNode>,
    pub above_marks: Vec<SceneNode>,
}

/// Convert a slice of annotation specs into positioned SceneNodes, partitioned
/// by z-order bucket.
///
/// Only `AnnotationSpec::Text` with `z == "below_marks"` routes to
/// `below_marks`; every other spec (including Text with any other z value)
/// routes to `above_marks`, preserving the historical single-bucket behavior
/// for all existing charts.
pub fn build_annotations(specs: &[AnnotationSpec], ctx: &ScaleContext) -> AnnotationNodes {
    let mut result = AnnotationNodes {
        below_marks: Vec::new(),
        above_marks: Vec::with_capacity(specs.len()),
    };
    for spec in specs {
        let target = match spec {
            AnnotationSpec::Text { z, .. } if z == "below_marks" => &mut result.below_marks,
            _ => &mut result.above_marks,
        };
        build_one(spec, ctx, target);
    }
    result
}

/// Dispatch a single annotation spec to its builder, appending nodes to `out`.
fn build_one(spec: &AnnotationSpec, ctx: &ScaleContext, out: &mut Vec<SceneNode>) {
    match spec {
        AnnotationSpec::Text {
            x,
            y,
            text,
            font_size,
            color,
            anchor,
            baseline,
            angle,
            dx,
            dy,
            ..
        } => {
            let px = ctx.resolve_x(x) + dx;
            let py = ctx.resolve_y(y) + dy;
            out.push(SceneNode::Text {
                x: px,
                y: py,
                content: text.to_string(),
                slot: None,
                style: TextStyle {
                    font_size: *font_size,
                    font_weight: FontWeight::Normal,
                    anchor: parse_anchor(anchor),
                    baseline: parse_baseline(baseline),
                    angle: *angle,
                    color: resolve_color(color),
                    opacity: 1.0,
                    font_family: "Inter, system-ui, sans-serif".to_string(),
                },
            });
        }
        AnnotationSpec::Arrow {
            x,
            y,
            x2,
            y2,
            stroke,
            stroke_width,
            head_size,
            ..
        } => {
            let x1_px = ctx.resolve_x(x);
            let y1_px = ctx.resolve_y(y);
            let x2_px = ctx.resolve_x(x2);
            let y2_px = ctx.resolve_y(y2);
            emit_arrow(
                x1_px,
                y1_px,
                x2_px,
                y2_px,
                stroke,
                *stroke_width,
                *head_size,
                out,
            );
        }
        AnnotationSpec::Rect {
            x1,
            y1,
            x2,
            y2,
            fill,
            opacity,
            corner_radius,
            stroke,
        } => {
            let px1 = ctx.resolve_x(x1);
            let py1 = ctx.resolve_y(y1);
            let px2 = ctx.resolve_x(x2);
            let py2 = ctx.resolve_y(y2);
            let x = px1.min(px2);
            let y = py1.min(py2);
            let w = (px2 - px1).abs();
            let h = (py2 - py1).abs();
            let fill_color = resolve_color(fill);
            let stroke_color = stroke.as_deref().map(resolve_color);
            out.push(SceneNode::Rect {
                x,
                y,
                w,
                h,
                style: FillStroke {
                    fill: Some(fill_color),
                    stroke: stroke_color,
                    stroke_width: if stroke_color.is_some() { 1.0 } else { 0.0 },
                    opacity: 1.0,
                    stroke_dash: None,
                    stroke_opacity: 1.0,
                    // Store the caller-supplied opacity as fill_opacity so the SVG
                    // writer emits fill-opacity="<value>" on the <rect> element.
                    // The FillStroke.opacity field is not forwarded to SVG attributes
                    // by to_svg_fill_stroke_with_anchor; only fill_opacity is.
                    fill_opacity: *opacity,
                    angle: 0.0,
                },
                corner_radius: *corner_radius,
            });
        }
        AnnotationSpec::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
            stroke_width,
            dash,
        } => {
            out.push(SceneNode::Line {
                x1: ctx.resolve_x(x1),
                y1: ctx.resolve_y(y1),
                x2: ctx.resolve_x(x2),
                y2: ctx.resolve_y(y2),
                style: StrokeStyle {
                    color: resolve_color(stroke),
                    width: *stroke_width,
                    opacity: 1.0,
                    dash: dash.as_deref().map(|d| d.to_vec()),
                    stroke_cap: None,
                    stroke_join: None,
                    stroke_opacity: 1.0,
                },
            });
        }
        AnnotationSpec::Span {
            axis,
            start,
            end,
            fill,
            opacity,
            label,
            label_position,
        } => {
            emit_span(
                axis,
                start,
                end,
                fill,
                *opacity,
                label.as_deref(),
                label_position.as_str(),
                ctx,
                out,
            );
        }
        AnnotationSpec::Bracket {
            x1,
            y1,
            x2,
            y2,
            label,
            direction,
            stroke,
            tip_length,
        } => {
            emit_bracket(
                ctx.resolve_x(x1),
                ctx.resolve_y(y1),
                ctx.resolve_x(x2),
                ctx.resolve_y(y2),
                label,
                direction,
                stroke,
                *tip_length,
                out,
            );
        }
        AnnotationSpec::Callout {
            x,
            y,
            text,
            arrow,
            padding,
            background,
            border_color,
            border_radius,
            text_x,
            text_y,
        } => {
            emit_callout(
                ctx,
                x,
                y,
                text,
                arrow,
                *padding,
                background,
                border_color,
                *border_radius,
                text_x.as_ref(),
                text_y.as_ref(),
                out,
            );
        }
        AnnotationSpec::Image {
            x,
            y,
            src,
            width,
            height,
            anchor,
        } => {
            let mut px = ctx.resolve_x(x);
            let mut py = ctx.resolve_y(y);
            match anchor.as_str() {
                "start" | "left" => { /* px is already left edge */ }
                "end" | "right" => {
                    px -= width;
                }
                _ => {
                    px -= width * 0.5;
                    py -= height * 0.5;
                }
            }
            // XML-escape src so that a URL containing '"' cannot break SVG structure.
            let escaped_src = src
                .replace('&', "&amp;")
                .replace('"', "&quot;")
                .replace('<', "&lt;")
                .replace('>', "&gt;");
            // Data: image annotations are positioned via the panel's coordinate
            // system (resolve_x/resolve_y maps data/pixel/norm values through the
            // panel scales). The resulting px/py are in data space relative to the
            // plot area, so this fragment must track the canvas transform on pan/zoom.
            out.push(SceneNode::Raw {
                svg: format!(
                    r#"<image x="{}" y="{}" width="{}" height="{}" href="{}"/>"#,
                    px, py, width, height, escaped_src
                ),
                anchor: RawAnchor::Data,
            });
        }
    }
}

// ── Emitter helpers (extracted to stay under clippy's argument limit) ────────

#[allow(clippy::too_many_arguments)]
fn emit_arrow(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: &str,
    stroke_width: f64,
    head_size: f64,
    out: &mut Vec<SceneNode>,
) {
    let color = resolve_color(stroke);

    // Line shaft.
    out.push(SceneNode::Line {
        x1,
        y1,
        x2,
        y2,
        style: StrokeStyle {
            color,
            width: stroke_width,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        },
    });

    // Arrowhead: a small filled triangle at (x2, y2) pointing in the
    // direction of travel. Geometry lives in `draw::arrowhead_triangle` —
    // shared with `marks::segment::build`'s `arrow=true` path (spec §4 H1) —
    // so this stays byte-identical to the pre-extraction inline math.
    if let Some(points) = arrowhead_triangle(x1, y1, x2, y2, head_size) {
        out.push(arrowhead_path(points, color));
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_span(
    axis: &str,
    start: &CoordValue,
    end: &CoordValue,
    fill: &str,
    opacity: f64,
    label: Option<&str>,
    label_position: &str,
    ctx: &ScaleContext,
    out: &mut Vec<SceneNode>,
) {
    let (x, y, w, h) = if axis == "x" {
        let x_start = ctx.resolve_x(start);
        let x_end = ctx.resolve_x(end);
        // Guard NaN: either coordinate being NaN produces NaN width — skip emission.
        if !x_start.is_finite() || !x_end.is_finite() {
            return;
        }
        let x = x_start.min(x_end);
        let w = (x_end - x_start).abs();
        (x, ctx.plot_area.y, w, ctx.plot_area.h)
    } else {
        let y_start = ctx.resolve_y(start);
        let y_end = ctx.resolve_y(end);
        // Guard NaN: either coordinate being NaN produces NaN height — skip emission.
        if !y_start.is_finite() || !y_end.is_finite() {
            return;
        }
        let y = y_start.min(y_end);
        let h = (y_end - y_start).abs();
        (ctx.plot_area.x, y, ctx.plot_area.w, h)
    };

    out.push(SceneNode::Rect {
        x,
        y,
        w,
        h,
        style: FillStroke {
            fill: Some(resolve_color(fill)),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: opacity,
            angle: 0.0,
        },
        corner_radius: 0.0,
    });

    if let Some(label_text) = label {
        if !label_text.is_empty() {
            // Small inset used for top/bottom placements so the label clears
            // the span edge. Consistent with other annotation label insets.
            const LABEL_INSET: f64 = 6.0;

            // In SVG, y increases downward: lower y = higher on screen.
            // "top" → near the top edge of the span (lower SVG y value),
            // "bottom" → near the bottom edge (higher SVG y value),
            // "middle" / anything else → vertical center.
            let label_y = match label_position {
                "top" => y + LABEL_INSET,
                "bottom" => y + h - LABEL_INSET,
                _ => y + h * 0.5,
            };
            // "top" places the label near the top edge — baseline is Top so
            // text hangs below the anchor point and sits inside the span.
            let baseline = match label_position {
                "top" => TextBaseline::Top,
                "bottom" => TextBaseline::Alphabetic,
                _ => TextBaseline::Middle,
            };

            out.push(SceneNode::Text {
                x: x + w * 0.5,
                y: label_y,
                content: label_text.to_string(),
                slot: None,
                style: TextStyle {
                    font_size: 11.0,
                    font_weight: FontWeight::Normal,
                    anchor: TextAnchor::Middle,
                    baseline,
                    angle: 0.0,
                    color: Color::rgb(51, 51, 51),
                    opacity: 1.0,
                    font_family: "Inter, system-ui, sans-serif".to_string(),
                },
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_bracket(
    px1: f64,
    py1: f64,
    px2: f64,
    py2: f64,
    label: &str,
    direction: &str,
    stroke: &str,
    tip_length: f64,
    out: &mut Vec<SceneNode>,
) {
    let color = resolve_color(stroke);
    let stroke_style = StrokeStyle {
        color,
        width: 1.5,
        opacity: 1.0,
        dash: None,
        stroke_cap: None,
        stroke_join: None,
        stroke_opacity: 1.0,
    };

    // Baseline.
    out.push(SceneNode::Line {
        x1: px1,
        y1: py1,
        x2: px2,
        y2: py2,
        style: stroke_style.clone(),
    });

    let (tip_dx, tip_dy) = match direction {
        "up" => (0.0, -tip_length),
        "down" => (0.0, tip_length),
        "left" => (-tip_length, 0.0),
        "right" => (tip_length, 0.0),
        _ => (0.0, -tip_length),
    };

    // Left tip.
    out.push(SceneNode::Line {
        x1: px1,
        y1: py1,
        x2: px1 + tip_dx,
        y2: py1 + tip_dy,
        style: stroke_style.clone(),
    });

    // Right tip.
    out.push(SceneNode::Line {
        x1: px2,
        y1: py2,
        x2: px2 + tip_dx,
        y2: py2 + tip_dy,
        style: stroke_style,
    });

    // Label centered above/below the bracket.
    if !label.is_empty() {
        let mid_x = (px1 + px2) * 0.5;
        let mid_y = (py1 + py2) * 0.5;
        let label_offset = tip_length + 4.0;
        let (off_x, off_y) = match direction {
            "up" => (0.0, -label_offset),
            "down" => (0.0, label_offset),
            "left" => (-label_offset, 0.0),
            "right" => (label_offset, 0.0),
            _ => (0.0, -label_offset),
        };
        out.push(SceneNode::Text {
            x: mid_x + off_x,
            y: mid_y + off_y,
            content: label.to_string(),
            slot: None,
            style: TextStyle {
                font_size: 11.0,
                font_weight: FontWeight::Normal,
                anchor: TextAnchor::Middle,
                baseline: TextBaseline::Middle,
                angle: 0.0,
                color: Color::rgb(51, 51, 51),
                opacity: 1.0,
                font_family: "Inter, system-ui, sans-serif".to_string(),
            },
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_callout(
    ctx: &ScaleContext,
    x: &CoordValue,
    y: &CoordValue,
    text: &str,
    arrow: &str,
    padding: f64,
    background: &str,
    border_color: &str,
    border_radius: f64,
    text_x: Option<&CoordValue>,
    text_y: Option<&CoordValue>,
    out: &mut Vec<SceneNode>,
) {
    let data_x = ctx.resolve_x(x);
    let data_y = ctx.resolve_y(y);

    let default_offset = 30.0;
    let tx = text_x
        .map(|v| ctx.resolve_x(v))
        .unwrap_or(data_x + default_offset);
    let ty = text_y
        .map(|v| ctx.resolve_y(v))
        .unwrap_or(data_y - default_offset);

    // Approximate text dimensions for the background box.
    // Use chars().count() not len() so multi-byte Unicode characters don't
    // inflate the estimated width (len() returns byte count, not char count).
    let char_width = 7.0;
    let text_w = text.chars().count() as f64 * char_width + padding * 2.0;
    let text_h = 14.0 + padding * 2.0;

    // Background rect.
    out.push(SceneNode::Rect {
        x: tx - text_w * 0.5,
        y: ty - text_h * 0.5,
        w: text_w,
        h: text_h,
        style: FillStroke {
            fill: Some(resolve_color(background)),
            stroke: Some(resolve_color(border_color)),
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        corner_radius: border_radius,
    });

    // Leader line: draw for any style except "none".
    // Documented values: "curved", "straight", "none". Legacy "true"/"yes" also draw.
    if arrow != "none" {
        out.push(SceneNode::Line {
            x1: data_x,
            y1: data_y,
            x2: tx,
            y2: ty,
            style: StrokeStyle {
                color: resolve_color(border_color),
                width: 1.0,
                opacity: 1.0,
                dash: None,
                stroke_cap: None,
                stroke_join: None,
                stroke_opacity: 1.0,
            },
        });
    }

    // Text label.
    out.push(SceneNode::Text {
        x: tx,
        y: ty,
        content: text.to_string(),
        slot: None,
        style: TextStyle {
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Middle,
            angle: 0.0,
            color: Color::rgb(51, 51, 51),
            opacity: 1.0,
            font_family: "Inter, system-ui, sans-serif".to_string(),
        },
    });
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;
    // Batch H Task 1 fix: `PathCmd` is only referenced by test assertions
    // (production code now goes through `draw::arrowhead_path`, which builds
    // its own `PathCmd` variants internally) — scoped here rather than at
    // module scope so a module-wide `use` doesn't go stale/unused again.
    use ferrum_scene::PathCmd;

    /// Create a simple linear scale context for testing.
    /// Domain [0, 100], x pixel range [50, 550], y pixel range [20, 320].
    fn test_ctx() -> (ScaleKind, ScaleKind, Rect) {
        let plot_area = Rect {
            x: 50.0,
            y: 20.0,
            w: 500.0,
            h: 300.0,
        };
        let x_scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![50.0, 550.0],
            false,
            false,
        ));
        let y_scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![20.0, 320.0],
            false,
            false,
        ));
        (x_scale, y_scale, plot_area)
    }

    #[test]
    fn resolve_data_coord_x() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        // Data value 50 should map to the center of the x pixel range.
        let px = ctx.resolve_x(&CoordValue::Data(50.0));
        assert!((px - 300.0).abs() < 1.0, "expected ~300, got {px}");
    }

    #[test]
    fn resolve_pixel_coord() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let px = ctx.resolve_x(&CoordValue::Pixel { px: 100.0 });
        assert!((px - 150.0).abs() < f64::EPSILON, "expected 150, got {px}");
    }

    #[test]
    fn resolve_norm_coord() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let px = ctx.resolve_x(&CoordValue::Norm { norm: 0.5 });
        assert!((px - 300.0).abs() < f64::EPSILON, "expected 300, got {px}");
    }

    #[test]
    fn build_text_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let specs = vec![AnnotationSpec::Text {
            x: CoordValue::Norm { norm: 0.5 },
            y: CoordValue::Norm { norm: 0.5 },
            text: "hello".to_string(),
            font_size: 14.0,
            color: "#ff0000".to_string(),
            anchor: "middle".to_string(),
            baseline: "middle".to_string(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "above_marks".to_string(),
        }];
        let ann = build_annotations(&specs, &ctx);
        // Default z="above_marks" lands in above_marks, not below_marks.
        assert!(ann.below_marks.is_empty());
        assert_eq!(ann.above_marks.len(), 1);
        match &ann.above_marks[0] {
            SceneNode::Text {
                x,
                y,
                content,
                style,
                ..
            } => {
                assert!((x - 300.0).abs() < f64::EPSILON);
                assert!((y - 170.0).abs() < f64::EPSILON);
                assert_eq!(content, "hello");
                assert_eq!(style.font_size, 14.0);
                assert_eq!(style.color, Color::rgb(255, 0, 0));
            }
            _ => panic!("expected Text node"),
        }
    }

    #[test]
    fn build_line_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let specs = vec![AnnotationSpec::Line {
            x1: CoordValue::Norm { norm: 0.0 },
            y1: CoordValue::Norm { norm: 0.5 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 0.5 },
            stroke: "#000000".to_string(),
            stroke_width: 2.0,
            dash: Some(vec![4.0, 2.0]),
        }];
        let ann = build_annotations(&specs, &ctx);
        // Non-Text specs always land in above_marks.
        assert!(ann.below_marks.is_empty());
        assert_eq!(ann.above_marks.len(), 1);
        match &ann.above_marks[0] {
            SceneNode::Line {
                x1,
                y1,
                x2,
                y2,
                style,
            } => {
                assert!((x1 - 50.0).abs() < f64::EPSILON);
                assert!((x2 - 550.0).abs() < f64::EPSILON);
                assert!((y1 - 170.0).abs() < f64::EPSILON);
                assert!((y2 - 170.0).abs() < f64::EPSILON);
                assert_eq!(style.width, 2.0);
                assert_eq!(style.dash.as_deref(), Some(&[4.0, 2.0][..]));
            }
            _ => panic!("expected Line node"),
        }
    }

    #[test]
    fn build_arrow_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        // `curve` is no longer a field on AnnotationSpec::Arrow (dropped as dead code).
        let specs = vec![AnnotationSpec::Arrow {
            x: CoordValue::Norm { norm: 0.0 },
            y: CoordValue::Norm { norm: 0.0 },
            x2: CoordValue::Norm { norm: 1.0 },
            y2: CoordValue::Norm { norm: 1.0 },
            stroke: "#333333".to_string(),
            stroke_width: 1.5,
            head_size: 8.0,
        }];
        let ann = build_annotations(&specs, &ctx);
        // Arrow (non-Text) always lands in above_marks.
        assert!(ann.below_marks.is_empty());
        // Arrow produces a line + arrowhead path.
        assert_eq!(ann.above_marks.len(), 2);
        assert!(matches!(&ann.above_marks[0], SceneNode::Line { .. }));
        assert!(matches!(&ann.above_marks[1], SceneNode::Path { .. }));
    }

    #[test]
    fn build_span_annotation() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let specs = vec![AnnotationSpec::Span {
            axis: "x".to_string(),
            start: CoordValue::Norm { norm: 0.2 },
            end: CoordValue::Norm { norm: 0.8 },
            fill: "#aaaaaa".to_string(),
            opacity: 0.3,
            label_position: "center".to_string(),
            label: Some("span".to_string()),
        }];
        let ann = build_annotations(&specs, &ctx);
        // Non-Text specs always land in above_marks.
        assert!(ann.below_marks.is_empty());
        // Span produces a rect + label text.
        assert_eq!(ann.above_marks.len(), 2);
        assert!(matches!(&ann.above_marks[0], SceneNode::Rect { .. }));
        assert!(matches!(&ann.above_marks[1], SceneNode::Text { .. }));
    }

    #[test]
    fn deser_annotation_spec_text() {
        let json = r#"{"type": "text", "x": 50.0, "y": {"norm": 0.5}, "text": "hi"}"#;
        let spec: AnnotationSpec = serde_json::from_str(json).unwrap();
        assert!(matches!(spec, AnnotationSpec::Text { .. }));
    }

    #[test]
    fn deser_annotation_spec_line() {
        let json = r##"{"type": "line", "x1": {"px": 0}, "y1": {"norm": 0.5}, "x2": {"px": 100}, "y2": {"norm": 0.5}, "stroke": "#000"}"##;
        let spec: AnnotationSpec = serde_json::from_str(json).unwrap();
        assert!(matches!(spec, AnnotationSpec::Line { .. }));
    }

    #[test]
    fn color_fallback_on_invalid() {
        let c = resolve_color("not-a-valid-color-xyz");
        assert_eq!(c, Color::rgb(51, 51, 51));
    }

    #[test]
    fn empty_specs_returns_empty_vecs() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let ann = build_annotations(&[], &ctx);
        assert!(ann.below_marks.is_empty());
        assert!(ann.above_marks.is_empty());
    }

    // ── Regression tests for z-routing (T2.1b / XDEAD-03) ───────────────────
    //
    // These tests must fail on the pre-wiring code where build_annotations
    // returned a flat Vec<SceneNode> with no z-based partition.

    /// A Text spec with z="below_marks" routes to `below_marks`; a Text spec
    /// with z="above_marks" (or default) routes to `above_marks`.
    #[test]
    fn text_z_routing_below_and_above() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };

        let mk_text = |z: &str| AnnotationSpec::Text {
            x: CoordValue::Norm { norm: 0.5 },
            y: CoordValue::Norm { norm: 0.5 },
            text: z.to_string(),
            font_size: 12.0,
            color: "#000000".to_string(),
            anchor: "middle".to_string(),
            baseline: "middle".to_string(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: z.to_string(),
        };

        let specs = vec![mk_text("below_marks"), mk_text("above_marks")];
        let ann = build_annotations(&specs, &ctx);

        // Exactly one node in each bucket.
        assert_eq!(
            ann.below_marks.len(),
            1,
            "below_marks bucket must have the below_marks text"
        );
        assert_eq!(
            ann.above_marks.len(),
            1,
            "above_marks bucket must have the above_marks text"
        );

        // Confirm the content so we know which went where.
        match &ann.below_marks[0] {
            SceneNode::Text { content, .. } => assert_eq!(content, "below_marks"),
            _ => panic!("expected Text in below_marks"),
        }
        match &ann.above_marks[0] {
            SceneNode::Text { content, .. } => assert_eq!(content, "above_marks"),
            _ => panic!("expected Text in above_marks"),
        }
    }

    /// An unknown/unrecognized z value falls through to the above-marks bucket
    /// (not below_marks).  Only the exact string "below_marks" selects the
    /// below bucket; everything else is above (fail-safe default).
    #[test]
    fn text_unknown_z_falls_to_above_marks() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };

        let specs = vec![AnnotationSpec::Text {
            x: CoordValue::Norm { norm: 0.5 },
            y: CoordValue::Norm { norm: 0.5 },
            text: "unknown".to_string(),
            font_size: 12.0,
            color: "#000000".to_string(),
            anchor: "middle".to_string(),
            baseline: "middle".to_string(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "front".to_string(), // old default — must still go above
        }];
        let ann = build_annotations(&specs, &ctx);
        assert!(
            ann.below_marks.is_empty(),
            "unrecognized z must not go to below_marks"
        );
        assert_eq!(ann.above_marks.len(), 1);
    }

    /// Non-Text specs (Arrow, Line, Rect, Span, Bracket, Callout, Image) have
    /// no z field and always land in above_marks.
    #[test]
    fn non_text_specs_always_above_marks() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };

        let specs = vec![
            AnnotationSpec::Line {
                x1: CoordValue::Norm { norm: 0.0 },
                y1: CoordValue::Norm { norm: 0.5 },
                x2: CoordValue::Norm { norm: 1.0 },
                y2: CoordValue::Norm { norm: 0.5 },
                stroke: "#000000".to_string(),
                stroke_width: 1.0,
                dash: None,
            },
            AnnotationSpec::Arrow {
                x: CoordValue::Norm { norm: 0.0 },
                y: CoordValue::Norm { norm: 0.0 },
                x2: CoordValue::Norm { norm: 1.0 },
                y2: CoordValue::Norm { norm: 1.0 },
                stroke: "#000000".to_string(),
                stroke_width: 1.5,
                head_size: 0.0, // no arrowhead → 1 node only
            },
        ];
        let ann = build_annotations(&specs, &ctx);
        assert!(
            ann.below_marks.is_empty(),
            "non-Text annotations must never go to below_marks"
        );
        // Line = 1 node; Arrow with head_size=0 = 1 node (no path triangle).
        assert_eq!(ann.above_marks.len(), 2);
    }

    /// Image annotation Raw nodes must carry `anchor == Data` — they are
    /// positioned via the panel's coordinate system (data/pixel/norm → plot-area
    /// pixels) and must track the canvas transform during pan/zoom.
    #[test]
    fn image_annotation_raw_node_has_data_anchor() {
        use ferrum_scene::RawAnchor;
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let specs = vec![AnnotationSpec::Image {
            x: CoordValue::Norm { norm: 0.5 },
            y: CoordValue::Norm { norm: 0.5 },
            src: "https://example.com/logo.png".to_string(),
            width: 40.0,
            height: 30.0,
            anchor: "middle".to_string(),
        }];
        let ann = build_annotations(&specs, &ctx);
        assert!(
            ann.below_marks.is_empty(),
            "image annotation should not go to below_marks"
        );
        assert_eq!(
            ann.above_marks.len(),
            1,
            "expected exactly one node for image annotation"
        );
        match &ann.above_marks[0] {
            SceneNode::Raw { anchor, .. } => {
                assert_eq!(
                    *anchor,
                    RawAnchor::Data,
                    "image annotation Raw node must have Data anchor"
                );
            }
            _ => panic!("expected SceneNode::Raw for image annotation"),
        }
    }

    /// R1 port (bug_hunt_draw.rs / bug_hunt_render_pipeline.rs): `parse_baseline`
    /// now routes through the single canonical `draw::parse_text_baseline`
    /// vocabulary instead of an independently hand-rolled match, so the
    /// previously-divergent aliases (`hanging`, `text-before-edge`,
    /// `ideographic`) resolve the same way here as they do for mark text.
    #[test]
    fn parse_baseline_delegates_to_canonical_vocabulary() {
        assert_eq!(parse_baseline("hanging"), TextBaseline::Top);
        assert_eq!(parse_baseline("text-before-edge"), TextBaseline::Top);
        assert_eq!(parse_baseline("ideographic"), TextBaseline::Bottom);
        assert_eq!(parse_baseline("unknown-value"), TextBaseline::Middle);
    }

    /// R1 port: `emit_span` skips emission entirely when either resolved
    /// coordinate is non-finite (e.g. an out-of-domain scale mapping), rather
    /// than emitting a rect with a NaN/zero-width dimension.
    #[test]
    fn emit_span_skips_on_non_finite_coordinate() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let mut out = Vec::new();
        emit_span(
            "x",
            &CoordValue::Pixel { px: f64::NAN },
            &CoordValue::Data(50.0),
            "#ff0000",
            0.3,
            None,
            "middle",
            &ctx,
            &mut out,
        );
        assert!(
            out.is_empty(),
            "non-finite span coordinate must skip emission entirely"
        );

        // A fully finite span does emit.
        let mut out2 = Vec::new();
        emit_span(
            "x",
            &CoordValue::Data(10.0),
            &CoordValue::Data(50.0),
            "#ff0000",
            0.3,
            None,
            "middle",
            &ctx,
            &mut out2,
        );
        assert_eq!(out2.len(), 1, "finite span must emit exactly one rect");
    }

    /// R1 port: `emit_arrow` clamps `head_size` to the shaft length so an
    /// epsilon-length arrow does not produce an arrowhead triangle that
    /// vastly overshoots the line's start point.
    #[test]
    fn emit_arrow_clamps_head_size_to_epsilon_shaft() {
        let mut out = Vec::new();
        emit_arrow(100.0, 100.0, 100.001, 100.0, "#000000", 1.0, 8.0, &mut out);
        assert_eq!(out.len(), 2, "expected shaft line + arrowhead path");
        let SceneNode::Path { commands, .. } = &out[1] else {
            panic!("expected arrowhead Path node");
        };
        // The arrowhead's base points must not extend past the shaft's start (x1=100.0).
        for cmd in commands {
            if let ferrum_scene::PathCmd::LineTo { x, .. } = cmd {
                assert!(
                    *x >= 100.0 - 1e-9,
                    "clamped arrowhead must not extend past the shaft start; got x={x}"
                );
            }
        }
    }

    /// Batch H Task 1: `emit_arrow`'s arrowhead Path must carry EXACTLY the
    /// vertices `draw::arrowhead_triangle` computes for the same
    /// endpoints/head_size — proving the annotation path and
    /// `marks::segment::build`'s new `arrow=true` path (segment.rs) share one
    /// geometry function rather than two independently-typed copies that
    /// could silently drift. Fails to compile against the pre-extraction
    /// surface: `draw::arrowhead_triangle` did not exist before this task.
    #[test]
    fn emit_arrow_triangle_matches_shared_arrowhead_triangle_helper() {
        let (x1, y1, x2, y2, head_size) = (10.0, 20.0, 130.0, 65.0, 8.0);
        let mut out = Vec::new();
        emit_arrow(x1, y1, x2, y2, "#123456", 1.0, head_size, &mut out);
        let SceneNode::Path { commands, .. } = &out[1] else {
            panic!("expected arrowhead Path node");
        };
        let emitted: Vec<(f64, f64)> = commands
            .iter()
            .filter_map(|c| match c {
                PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => Some((*x, *y)),
                _ => None,
            })
            .collect();

        let expected = arrowhead_triangle(x1, y1, x2, y2, head_size)
            .expect("non-degenerate segment must yield a triangle");

        assert_eq!(
            emitted,
            vec![expected[0], expected[1], expected[2]],
            "emit_arrow's triangle vertices must equal arrowhead_triangle's output exactly"
        );
    }

    /// R1 port: an unrecognized bracket `direction` defaults to "up" (matching
    /// the fallthrough arm), rather than silently producing a zero-length tip.
    #[test]
    fn emit_bracket_unknown_direction_defaults_to_up() {
        let mut out_up = Vec::new();
        emit_bracket(
            0.0,
            100.0,
            100.0,
            100.0,
            "",
            "up",
            "#000000",
            6.0,
            &mut out_up,
        );
        let mut out_unknown = Vec::new();
        emit_bracket(
            0.0,
            100.0,
            100.0,
            100.0,
            "",
            "diagonal",
            "#000000",
            6.0,
            &mut out_unknown,
        );
        assert_eq!(out_up.len(), out_unknown.len());
        // Both must produce identical tip geometry (same Line endpoints).
        for (a, b) in out_up.iter().zip(out_unknown.iter()) {
            match (a, b) {
                (
                    SceneNode::Line {
                        x1: ax1,
                        y1: ay1,
                        x2: ax2,
                        y2: ay2,
                        ..
                    },
                    SceneNode::Line {
                        x1: bx1,
                        y1: by1,
                        x2: bx2,
                        y2: by2,
                        ..
                    },
                ) => {
                    assert_eq!((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2));
                }
                _ => panic!("expected Line nodes for both directions"),
            }
        }
    }

    /// R1 port: the callout background box width uses Unicode character count
    /// (`chars().count()`), not UTF-8 byte length, so multi-byte labels are not
    /// over-estimated.
    #[test]
    fn emit_callout_width_uses_char_count_not_byte_len() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let ascii = "resume"; // 6 chars, 6 bytes
        let unicode = "r\u{00e9}sum\u{00e9}"; // 6 chars, 8 bytes
        assert_eq!(ascii.chars().count(), unicode.chars().count());

        let build = |text: &str| {
            let mut out = Vec::new();
            emit_callout(
                &ctx,
                &CoordValue::Data(50.0),
                &CoordValue::Data(50.0),
                text,
                "curved",
                4.0,
                "#fff",
                "#333",
                0.0,
                None,
                None,
                &mut out,
            );
            let SceneNode::Rect { w, .. } = out[0] else {
                panic!("expected background Rect")
            };
            w
        };
        assert_eq!(
            build(ascii),
            build(unicode),
            "equal char count must produce equal box width"
        );
    }

    /// R1 port: `arrow == "none"` suppresses the callout's leader line; any
    /// other value (including an empty string) still draws it.
    #[test]
    fn emit_callout_arrow_none_suppresses_leader_line() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let count_lines = |arrow: &str| {
            let mut out = Vec::new();
            emit_callout(
                &ctx,
                &CoordValue::Data(50.0),
                &CoordValue::Data(50.0),
                "hi",
                arrow,
                4.0,
                "#fff",
                "#333",
                0.0,
                None,
                None,
                &mut out,
            );
            out.iter()
                .filter(|n| matches!(n, SceneNode::Line { .. }))
                .count()
        };
        assert_eq!(
            count_lines("none"),
            0,
            "arrow='none' must suppress the leader line"
        );
        assert_eq!(
            count_lines(""),
            1,
            "empty string is not 'none' — leader line still drawn"
        );
        assert_eq!(count_lines("curved"), 1);
    }

    // ── kind_name / data_coords_on_axis (spec §4 H5, F-L08-07) ──────────────

    #[test]
    fn kind_name_covers_every_shape() {
        let cases: Vec<(AnnotationSpec, &str)> = vec![
            (
                AnnotationSpec::Text {
                    x: CoordValue::Data(1.0),
                    y: CoordValue::Data(2.0),
                    text: "t".into(),
                    font_size: 12.0,
                    color: "#000".into(),
                    anchor: "middle".into(),
                    baseline: "middle".into(),
                    angle: 0.0,
                    dx: 0.0,
                    dy: 0.0,
                    z: "above_marks".into(),
                },
                "text",
            ),
            (
                AnnotationSpec::Arrow {
                    x: CoordValue::Data(1.0),
                    y: CoordValue::Data(2.0),
                    x2: CoordValue::Data(3.0),
                    y2: CoordValue::Data(4.0),
                    stroke: "#000".into(),
                    stroke_width: 1.5,
                    head_size: 8.0,
                },
                "arrow",
            ),
            (
                AnnotationSpec::Rect {
                    x1: CoordValue::Data(1.0),
                    y1: CoordValue::Data(2.0),
                    x2: CoordValue::Data(3.0),
                    y2: CoordValue::Data(4.0),
                    fill: "#ccc".into(),
                    opacity: 0.3,
                    corner_radius: 0.0,
                    stroke: None,
                },
                "rect",
            ),
            (
                AnnotationSpec::Line {
                    x1: CoordValue::Data(1.0),
                    y1: CoordValue::Data(2.0),
                    x2: CoordValue::Data(3.0),
                    y2: CoordValue::Data(4.0),
                    stroke: "#000".into(),
                    stroke_width: 1.5,
                    dash: None,
                },
                "line",
            ),
            (
                AnnotationSpec::Span {
                    axis: "x".into(),
                    start: CoordValue::Data(1.0),
                    end: CoordValue::Data(2.0),
                    fill: "#ccc".into(),
                    opacity: 0.2,
                    label_position: "center".into(),
                    label: None,
                },
                "span",
            ),
            (
                AnnotationSpec::Bracket {
                    x1: CoordValue::Data(1.0),
                    y1: CoordValue::Data(2.0),
                    x2: CoordValue::Data(3.0),
                    y2: CoordValue::Data(4.0),
                    label: "".into(),
                    direction: "up".into(),
                    stroke: "#000".into(),
                    tip_length: 6.0,
                },
                "bracket",
            ),
            (
                AnnotationSpec::Callout {
                    x: CoordValue::Data(1.0),
                    y: CoordValue::Data(2.0),
                    text: "c".into(),
                    arrow: "curved".into(),
                    padding: 4.0,
                    background: "#fff".into(),
                    border_color: "#333".into(),
                    border_radius: 0.0,
                    text_x: None,
                    text_y: None,
                },
                "callout",
            ),
            (
                AnnotationSpec::Image {
                    x: CoordValue::Data(1.0),
                    y: CoordValue::Data(2.0),
                    src: "s".into(),
                    width: 10.0,
                    height: 10.0,
                    anchor: "middle".into(),
                },
                "image",
            ),
        ];
        for (spec, expected) in cases {
            assert_eq!(kind_name(&spec), expected);
        }
    }

    #[test]
    fn data_coords_on_axis_text() {
        let spec = AnnotationSpec::Text {
            x: CoordValue::Data(10.0),
            y: CoordValue::Data(20.0),
            text: "t".into(),
            font_size: 12.0,
            color: "#000".into(),
            anchor: "middle".into(),
            baseline: "middle".into(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "above_marks".into(),
        };
        assert_eq!(data_coords_on_axis(&spec, "x"), vec![10.0]);
        assert_eq!(data_coords_on_axis(&spec, "y"), vec![20.0]);
    }

    #[test]
    fn data_coords_on_axis_excludes_px_and_norm() {
        let spec = AnnotationSpec::Text {
            x: CoordValue::Pixel { px: 5.0 },
            y: CoordValue::Norm { norm: 0.5 },
            text: "t".into(),
            font_size: 12.0,
            color: "#000".into(),
            anchor: "middle".into(),
            baseline: "middle".into(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "above_marks".into(),
        };
        assert!(data_coords_on_axis(&spec, "x").is_empty());
        assert!(data_coords_on_axis(&spec, "y").is_empty());
    }

    #[test]
    fn data_coords_on_axis_arrow_both_endpoints() {
        let spec = AnnotationSpec::Arrow {
            x: CoordValue::Data(1.0),
            y: CoordValue::Data(2.0),
            x2: CoordValue::Data(3.0),
            y2: CoordValue::Data(4.0),
            stroke: "#000".into(),
            stroke_width: 1.5,
            head_size: 8.0,
        };
        assert_eq!(data_coords_on_axis(&spec, "x"), vec![1.0, 3.0]);
        assert_eq!(data_coords_on_axis(&spec, "y"), vec![2.0, 4.0]);
    }

    #[test]
    fn data_coords_on_axis_rect_line_bracket_share_shape() {
        let specs = [
            AnnotationSpec::Rect {
                x1: CoordValue::Data(1.0),
                y1: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                fill: "#ccc".into(),
                opacity: 0.3,
                corner_radius: 0.0,
                stroke: None,
            },
            AnnotationSpec::Line {
                x1: CoordValue::Data(1.0),
                y1: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                stroke: "#000".into(),
                stroke_width: 1.5,
                dash: None,
            },
            AnnotationSpec::Bracket {
                x1: CoordValue::Data(1.0),
                y1: CoordValue::Data(2.0),
                x2: CoordValue::Data(3.0),
                y2: CoordValue::Data(4.0),
                label: "".into(),
                direction: "up".into(),
                stroke: "#000".into(),
                tip_length: 6.0,
            },
        ];
        for spec in &specs {
            assert_eq!(data_coords_on_axis(spec, "x"), vec![1.0, 3.0]);
            assert_eq!(data_coords_on_axis(spec, "y"), vec![2.0, 4.0]);
        }
    }

    #[test]
    fn data_coords_on_axis_span_gated_by_its_own_axis_field() {
        let x_span = AnnotationSpec::Span {
            axis: "x".into(),
            start: CoordValue::Data(1.0),
            end: CoordValue::Data(2.0),
            fill: "#ccc".into(),
            opacity: 0.2,
            label_position: "center".into(),
            label: None,
        };
        assert_eq!(data_coords_on_axis(&x_span, "x"), vec![1.0, 2.0]);
        assert!(data_coords_on_axis(&x_span, "y").is_empty());

        let y_span = AnnotationSpec::Span {
            axis: "y".into(),
            start: CoordValue::Data(5.0),
            end: CoordValue::Data(6.0),
            fill: "#ccc".into(),
            opacity: 0.2,
            label_position: "center".into(),
            label: None,
        };
        assert_eq!(data_coords_on_axis(&y_span, "y"), vec![5.0, 6.0]);
        assert!(data_coords_on_axis(&y_span, "x").is_empty());
    }

    #[test]
    fn data_coords_on_axis_callout_includes_optional_text_position() {
        let with_text_pos = AnnotationSpec::Callout {
            x: CoordValue::Data(1.0),
            y: CoordValue::Data(2.0),
            text: "c".into(),
            arrow: "curved".into(),
            padding: 4.0,
            background: "#fff".into(),
            border_color: "#333".into(),
            border_radius: 0.0,
            text_x: Some(CoordValue::Data(11.0)),
            text_y: Some(CoordValue::Data(12.0)),
        };
        assert_eq!(data_coords_on_axis(&with_text_pos, "x"), vec![1.0, 11.0]);
        assert_eq!(data_coords_on_axis(&with_text_pos, "y"), vec![2.0, 12.0]);

        // text_x/text_y omitted: only the anchor coordinate is reported.
        let no_text_pos = AnnotationSpec::Callout {
            x: CoordValue::Data(1.0),
            y: CoordValue::Data(2.0),
            text: "c".into(),
            arrow: "curved".into(),
            padding: 4.0,
            background: "#fff".into(),
            border_color: "#333".into(),
            border_radius: 0.0,
            text_x: None,
            text_y: None,
        };
        assert_eq!(data_coords_on_axis(&no_text_pos, "x"), vec![1.0]);
        assert_eq!(data_coords_on_axis(&no_text_pos, "y"), vec![2.0]);
    }

    #[test]
    fn data_coords_on_axis_image_single_anchor() {
        let spec = AnnotationSpec::Image {
            x: CoordValue::Data(7.0),
            y: CoordValue::Data(8.0),
            src: "s".into(),
            width: 10.0,
            height: 10.0,
            anchor: "middle".into(),
        };
        assert_eq!(data_coords_on_axis(&spec, "x"), vec![7.0]);
        assert_eq!(data_coords_on_axis(&spec, "y"), vec![8.0]);
    }

    // ── break-aware resolve_x/resolve_y (spec §4 H5, F-L08-07) ──────────────

    #[test]
    fn resolve_x_data_coord_routes_through_break_when_present() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let result = crate::render::break_axis::apply_break_to_scale(
            (0.0, 100.0),
            &[[40.0, 60.0]],
            (50.0, 550.0),
            10.0,
        );
        let remap = crate::render::break_axis::BreakRemap::new(
            "x",
            (0.0, 100.0),
            (50.0, 550.0),
            &result,
        );
        let breaks = [remap];
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &breaks,
        };
        let broken_px = ctx.resolve_x(&CoordValue::Data(20.0));
        let expected = crate::render::break_axis::broken_scale_map(20.0, &result).unwrap();
        assert!((broken_px - expected).abs() < f64::EPSILON);

        // Without the break, the same value maps to the plain linear
        // position — proving the break actually moved something.
        let unbroken_ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        let unbroken_px = unbroken_ctx.resolve_x(&CoordValue::Data(20.0));
        assert!(
            (broken_px - unbroken_px).abs() > 1.0,
            "the break must move the compressed position away from the unbroken one"
        );
    }

    /// Two x-axis breaks fold in sequence — the same composition
    /// `remap_mark_batch_through_break`'s `for remap in &break_remaps` loop
    /// applies to marks (spec §4 H5 parity, quality-review cycle-2 finding
    /// #2). A `.find`-first implementation would stop at the first break and
    /// diverge from this.
    #[test]
    fn resolve_x_folds_through_multiple_same_axis_breaks() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let result_a = crate::render::break_axis::apply_break_to_scale(
            (0.0, 100.0),
            &[[20.0, 30.0]],
            (50.0, 550.0),
            10.0,
        );
        let result_b = crate::render::break_axis::apply_break_to_scale(
            (0.0, 100.0),
            &[[70.0, 80.0]],
            (50.0, 550.0),
            10.0,
        );
        let remap_a =
            crate::render::break_axis::BreakRemap::new("x", (0.0, 100.0), (50.0, 550.0), &result_a);
        let remap_b =
            crate::render::break_axis::BreakRemap::new("x", (0.0, 100.0), (50.0, 550.0), &result_b);
        // A second, independent handle on the same first break, so it can be
        // compared against the two-break fold without fighting the borrow
        // checker over `remap_a` (`BreakRemap` is not `Copy`/`Clone`).
        let remap_a_only =
            crate::render::break_axis::BreakRemap::new("x", (0.0, 100.0), (50.0, 550.0), &result_a);

        let folded_ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[remap_a, remap_b],
        };
        let single_ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: std::slice::from_ref(&remap_a_only),
        };
        let folded_px = folded_ctx.resolve_x(&CoordValue::Data(90.0));
        let single_px = single_ctx.resolve_x(&CoordValue::Data(90.0));
        assert!(
            (folded_px - single_px).abs() > 1.0,
            "the second break must further move a coordinate the first break \
             already moved, not be ignored: folded={folded_px} single_only={single_px}"
        );
    }

    #[test]
    fn resolve_y_pixel_and_norm_coords_are_unaffected_by_a_break() {
        let (x_scale, y_scale, plot_area) = test_ctx();
        let result = crate::render::break_axis::apply_break_to_scale(
            (0.0, 100.0),
            &[[40.0, 60.0]],
            (20.0, 320.0),
            10.0,
        );
        let remap = crate::render::break_axis::BreakRemap::new(
            "y",
            (0.0, 100.0),
            (20.0, 320.0),
            &result,
        );
        let broken_ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[remap],
        };
        let plain_ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };
        // Plot-area-relative coordinates are identical with or without an
        // active break on this axis — only `CoordValue::Data` routes through
        // the piecewise mapping.
        assert_eq!(
            broken_ctx.resolve_y(&CoordValue::Pixel { px: 42.0 }),
            plain_ctx.resolve_y(&CoordValue::Pixel { px: 42.0 }),
        );
        assert_eq!(
            broken_ctx.resolve_y(&CoordValue::Norm { norm: 0.25 }),
            plain_ctx.resolve_y(&CoordValue::Norm { norm: 0.25 }),
        );
    }

    /// quality-review cycle-3: `resolve_y`'s out-of-domain fallback used
    /// `frac` directly (no flip), which is backwards relative to how every
    /// non-ordinal y axis actually maps its domain onto pixels
    /// (`scale_resolve::positional::axis_pixel_range` flips domain[0] to the
    /// BOTTOM pixel and domain[1] to the TOP, unconditionally). This test
    /// builds a `y_scale` with that SAME flipped range (`[320.0, 20.0]` —
    /// domain min -> bottom, domain max -> top), the shape a real chart's y
    /// scale actually has, rather than `test_ctx`'s simplified direct range —
    /// so it exercises the fallback against the real invariant it depends on,
    /// not just the formula in isolation.
    #[test]
    fn resolve_y_out_of_domain_value_clamps_to_the_correct_side() {
        use crate::scale::linear::LinearScale;

        let (x_scale, _, plot_area) = test_ctx();
        let y_scale = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 100.0],
            vec![320.0, 20.0],
            false,
            false,
        ));
        let ctx = ScaleContext {
            plot_area,
            x_scale: &x_scale,
            y_scale: &y_scale,
            breaks: &[],
        };

        // Domain is [0, 100]; both values are out of domain, so `to_pixel_f64`
        // returns `None` and the fallback fires (verified separately below).
        let above_max = ctx.resolve_y(&CoordValue::Data(150.0));
        let below_min = ctx.resolve_y(&CoordValue::Data(-50.0));

        assert!(
            y_scale.to_pixel_f64(150.0).is_none() && y_scale.to_pixel_f64(-50.0).is_none(),
            "this test must exercise the out-of-domain fallback, not to_pixel_f64's happy path"
        );

        // A value ABOVE the domain max must land on the TOP side (smaller
        // pixel) of a value BELOW the domain min (BOTTOM side, larger
        // pixel) — matching the real y scale's domain-max-at-top convention.
        // Pre-fix this assertion failed: above_max (470) was LARGER than
        // below_min (-130), the sides swapped.
        assert!(
            above_max < below_min,
            "above_max={above_max} must be nearer the top (smaller) than \
             below_min={below_min}"
        );
        // Exact values, hand-computed from the flipped formula:
        // plot_area.y=20, h=300. above_max: frac=(150-0)/100=1.5 ->
        // 20 + (1-1.5)*300 = -130. below_min: frac=(-50-0)/100=-0.5 ->
        // 20 + (1-(-0.5))*300 = 470.
        assert!((above_max - (-130.0)).abs() < 1e-6, "got {above_max}");
        assert!((below_min - 470.0).abs() < 1e-6, "got {below_min}");
    }
}
