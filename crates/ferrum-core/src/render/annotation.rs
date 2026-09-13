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
use super::scale_resolve::{data_pixel_x, data_pixel_y, ScaleKind};
use super::RenderWarning;
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
///
/// `deny_unknown_fields` closes the #148 array carve-out's defense-in-depth
/// line (the primary refusal is the wire gate, `binding.rs::
/// validate_chart_config_keys`'s array branch, which consults
/// [`accepted_keys_for_annotation_variant`] below and runs BEFORE
/// deserialization — same "manual gate first, deny_unknown_fields second"
/// split `chart_config.rs`'s module doc describes for the dict-shaped
/// sections). Verified empirically that this does not defeat itself the way
/// `#[serde(flatten)]` does for `AxisConfigSpec`: an internally-tagged enum's
/// own `type` tag is excluded from each variant's "expected" field set by
/// serde-derive, so `deny_unknown_fields` here rejects a genuinely unknown
/// field without ever flagging `type` itself as unknown.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
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

// ── Per-variant wire key sets (#92/#148 array-item gate) ────────────────────
//
// One const per `AnnotationSpec` variant, INCLUDING `type` (the item's own
// discriminator key, which is a real key on the wire dict even though it is
// not a named field of the variant struct itself — see the enum's own doc).
// Each is drift-tested against the real variant in this module's `tests`
// (`annotation_*_keys_match_serde`), the same `deny_unknown_fields`-mining
// technique `chart_config.rs`'s `*_keys_match_serde` tests use
// (`accepted_fields_from_deny_unknown_error`, reused from there).

/// `AnnotationSpec::Text`'s wire keys.
pub(crate) const ANNOTATION_TEXT_KEYS: &[&str] = &[
    "type", "x", "y", "text", "font_size", "color", "anchor", "baseline", "angle", "dx", "dy", "z",
];
/// `AnnotationSpec::Arrow`'s wire keys.
pub(crate) const ANNOTATION_ARROW_KEYS: &[&str] =
    &["type", "x", "y", "x2", "y2", "stroke", "stroke_width", "head_size"];
/// `AnnotationSpec::Rect`'s wire keys.
pub(crate) const ANNOTATION_RECT_KEYS: &[&str] =
    &["type", "x1", "y1", "x2", "y2", "fill", "opacity", "corner_radius", "stroke"];
/// `AnnotationSpec::Line`'s wire keys.
pub(crate) const ANNOTATION_LINE_KEYS: &[&str] =
    &["type", "x1", "y1", "x2", "y2", "stroke", "stroke_width", "dash"];
/// `AnnotationSpec::Span`'s wire keys.
pub(crate) const ANNOTATION_SPAN_KEYS: &[&str] =
    &["type", "axis", "start", "end", "fill", "opacity", "label_position", "label"];
/// `AnnotationSpec::Bracket`'s wire keys.
pub(crate) const ANNOTATION_BRACKET_KEYS: &[&str] =
    &["type", "x1", "y1", "x2", "y2", "label", "direction", "stroke", "tip_length"];
/// `AnnotationSpec::Callout`'s wire keys.
pub(crate) const ANNOTATION_CALLOUT_KEYS: &[&str] = &[
    "type",
    "x",
    "y",
    "text",
    "arrow",
    "padding",
    "background",
    "border_color",
    "border_radius",
    "text_x",
    "text_y",
];
/// `AnnotationSpec::Image`'s wire keys.
pub(crate) const ANNOTATION_IMAGE_KEYS: &[&str] =
    &["type", "x", "y", "src", "width", "height", "anchor"];

/// The accepted wire-key set for one `annotations[]` item, keyed by its
/// `type` tag — the per-item counterpart of `chart_config::
/// accepted_keys_for_section`, consulted by the wire gate's array branch
/// (`binding.rs::validate_chart_config_keys`) to close the #148 carve-out.
/// `None` for an unrecognized tag — that case is left to serde's own
/// "unknown variant" error, matching how the gate walk already leaves a
/// non-dict-shaped section value to serde's type-mismatch error.
pub(crate) fn accepted_keys_for_annotation_variant(type_tag: &str) -> Option<&'static [&'static str]> {
    match type_tag {
        "text" => Some(ANNOTATION_TEXT_KEYS),
        "arrow" => Some(ANNOTATION_ARROW_KEYS),
        "rect" => Some(ANNOTATION_RECT_KEYS),
        "line" => Some(ANNOTATION_LINE_KEYS),
        "span" => Some(ANNOTATION_SPAN_KEYS),
        "bracket" => Some(ANNOTATION_BRACKET_KEYS),
        "callout" => Some(ANNOTATION_CALLOUT_KEYS),
        "image" => Some(ANNOTATION_IMAGE_KEYS),
        _ => None,
    }
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
                let base = data_pixel_x(self.x_scale, &self.plot_area, *d);
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
                let base = data_pixel_y(self.y_scale, &self.plot_area, *d);
                self.fold_breaks("y", base)
            }
            CoordValue::Pixel { px } => self.plot_area.y + px,
            CoordValue::Norm { norm } => self.plot_area.y + norm * self.plot_area.h,
        }
    }

    /// Fold `base` (a pixel already resolved against the UNBROKEN `x_scale`/
    /// `y_scale`) through every `axis`-matching break, in declaration order.
    ///
    /// This is exactly `remap_mark_batch_through_break`'s per-coordinate
    /// fold — reused here rather than re-derived, so an annotation and a
    /// mark reach the same pixel through the same arithmetic. In practice
    /// the fold runs 0 or 1 times per axis: `build_structural_nodes`'s
    /// same-axis merge (#230-4) collapses every `BreakAxis` spec sharing an
    /// axis into ONE `BreakResult` before `break_remaps` is built, so this
    /// axis-filtered iterator never holds more than one remap per axis, and
    /// sequential composition of multiple same-axis breaks is not a shape
    /// this method (or its mark-side counterpart) has to get right anymore.
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

// ── Break-axis coordinate classification (spec §4 H5, F-L08-07) ─────────────
//
// What an annotation's coordinates on one axis ARE. What a break then DOES with
// them — clip, split, or drop — is the routing section at the bottom of this
// file (#228).

/// The wire `type` tag for `spec` — the annotation "kind" named in
/// `RenderWarning::AnnotationSuppressedByBreak`.
///
/// Mirrors the `#[serde(tag = "type")]` discriminant on `AnnotationSpec`
/// above via an exhaustive match, so a new spec variant is a compile error
/// here rather than silently falling through to an untagged warning.
fn kind_name(spec: &AnnotationSpec) -> &'static str {
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

/// What an annotation's data-space coordinates on one axis MEAN, which is
/// what decides how a `BreakAxis` gap must test them.
///
/// The distinction is the whole content of spec §4 H5's "any in-gap data
/// coordinate": for a point-shaped annotation the coordinate IS the
/// annotation's position, but for an extent-shaped one the two coordinates
/// bound an interval whose interior is part of the annotation too. Testing
/// only the endpoints made suppression non-monotonic — a rect overlapping a
/// gap vanished while a LARGER rect containing that gap painted straight
/// across the break. It is also what tells the two #228 routings apart: only
/// an interval can be CLIPPED, so only [`AxisCoords::Extent`] can reach the
/// clip at all.
#[derive(Debug, Clone, PartialEq)]
enum AxisCoords {
    /// Point-shaped anchors (`Text`, `Callout`'s anchor and text anchor,
    /// `Image`), each suppressed only when it lands inside a gap. Also the
    /// degenerate form of an extent shape whose endpoints are not both
    /// data-space: with no second data bound there is no interval to test,
    /// only the coordinate(s) that are data.
    Points(Vec<f64>),
    /// Both bounds of an extent shape (`Rect`, `Span`, `Line`, `Arrow`,
    /// `Bracket`) on this axis, in spec order. The break REACHES the interval
    /// when it intersects a gap — an endpoint inside one, or the interval
    /// reaching across one — and what happens then is the kind's own routing
    /// ([`BreakRouting`], #228): clipped per visible segment for the four
    /// splittable kinds, dropped whole for `Bracket` and for an interval with
    /// nothing left.
    Extent(f64, f64),
}

/// The data-space (`CoordValue::Data`) coordinates `spec` carries along
/// `axis` (`"x"` or `"y"`), in the shape the break's suppression rule must
/// test them — see [`AxisCoords`].
///
/// Pixel/norm coordinates are plot-area-relative, not data positions, so a
/// `BreakAxis` gap never suppresses them — they are excluded here rather than
/// filtered by the caller, so every caller gets that rule for free. An extent
/// shape with only one data-space bound therefore degrades to
/// [`AxisCoords::Points`]: half a data interval bounds nothing.
///
/// Exhaustive over `AnnotationSpec`'s 8 shapes, enumerated from this file's
/// own field list (spec §4 H5's coordinate table), so a new spec variant is a
/// compile error here rather than silently skipping break suppression for it.
fn data_coords_on_axis(spec: &AnnotationSpec, axis: &str) -> AxisCoords {
    // Branches on `axis == "y"` with x as the fallback, matching every other
    // break-axis consumer's convention (`break_axis.rs`'s `is_y()`,
    // `scene_build.rs`'s `if axis == "y" { &scales.y } else { &scales.x }`) —
    // not S2 today (`BreakAxis.axis` is validated to `("x", "y")` in Python),
    // but keeping one direction crate-wide means a future third axis value
    // can't be suppressed as one axis and remapped as the other.
    let is_y = axis == "y";
    let (fields, shape): (Vec<&CoordValue>, Shape) = match spec {
        AnnotationSpec::Text { x, y, .. } => (vec![if is_y { y } else { x }], Shape::Point),
        AnnotationSpec::Arrow { x, y, x2, y2, .. } => {
            (if is_y { vec![y, y2] } else { vec![x, x2] }, Shape::Extent)
        }
        AnnotationSpec::Rect { x1, y1, x2, y2, .. } => {
            (if is_y { vec![y1, y2] } else { vec![x1, x2] }, Shape::Extent)
        }
        AnnotationSpec::Line { x1, y1, x2, y2, .. } => {
            (if is_y { vec![y1, y2] } else { vec![x1, x2] }, Shape::Extent)
        }
        // A `Span`'s `start`/`end` are values on its OWN `axis` field, which
        // may differ from the axis under test — an x-span is untouched by a
        // y-break, and vice versa.
        AnnotationSpec::Span { axis: span_axis, start, end, .. } => {
            if span_axis == axis {
                (vec![start, end], Shape::Extent)
            } else {
                (Vec::new(), Shape::Point)
            }
        }
        AnnotationSpec::Bracket { x1, y1, x2, y2, .. } => {
            (if is_y { vec![y1, y2] } else { vec![x1, x2] }, Shape::Extent)
        }
        // A callout's anchor and its text anchor are two independent POINTS,
        // not the ends of one interval — the leader line between them is a
        // pointer, not an extent the annotation claims.
        AnnotationSpec::Callout { x, y, text_x, text_y, .. } => {
            let mut v = vec![if is_y { y } else { x }];
            if is_y {
                if let Some(ty) = text_y {
                    v.push(ty);
                }
            } else if let Some(tx) = text_x {
                v.push(tx);
            }
            (v, Shape::Point)
        }
        AnnotationSpec::Image { x, y, .. } => (vec![if is_y { y } else { x }], Shape::Point),
    };
    let values: Vec<f64> = fields
        .into_iter()
        .filter_map(|v| match v {
            CoordValue::Data(d) => Some(*d),
            _ => None,
        })
        .collect();
    match (shape, values.as_slice()) {
        (Shape::Extent, &[lo, hi]) => AxisCoords::Extent(lo, hi),
        _ => AxisCoords::Points(values),
    }
}

/// Whether a spec's coordinates on one axis bound an interval or stand alone
/// — the per-variant half of [`AxisCoords`], kept next to the field list it
/// is read off.
enum Shape {
    Point,
    Extent,
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
///
/// PRIVATE by design: [`build_annotations_through_breaks`] is this module's only
/// entry from the scene builder. Reaching this directly with a `ctx` that
/// carries breaks would place every extent annotation endpoint-by-endpoint —
/// painting a straddling rect as one solid block across the gap — which is the
/// defect the break routing exists to prevent (#228).
fn build_annotations(specs: &[AnnotationSpec], ctx: &ScaleContext) -> AnnotationNodes {
    let mut result = AnnotationNodes {
        below_marks: Vec::new(),
        above_marks: Vec::with_capacity(specs.len()),
    };
    for spec in specs {
        build_one(spec, ctx, z_bucket(spec, &mut result));
    }
    result
}

/// The z-order bucket `spec`'s nodes belong in — the partition rule
/// [`AnnotationNodes`] documents, read off the spec rather than restated at
/// each of its two call sites (the plain build and the break-routed one).
fn z_bucket<'n>(spec: &AnnotationSpec, nodes: &'n mut AnnotationNodes) -> &'n mut Vec<SceneNode> {
    match spec {
        AnnotationSpec::Text { z, .. } if z == "below_marks" => &mut nodes.below_marks,
        _ => &mut nodes.above_marks,
    }
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

    if let Some(node) = span_label_node(x, y, w, h, label, label_position) {
        out.push(node);
    }
}

/// The label node for a span band occupying `(x, y, w, h)`, or `None` when the
/// span carries no label text.
///
/// Extracted from [`emit_span`] so the break clip can re-anchor the label onto
/// the piece that survived largest ([`clipped_span_label`]): a span straddling a
/// gap has its pre-clip midpoint INSIDE the gap, which is the one place its
/// label must not be drawn (#228).
fn span_label_node(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    label: Option<&str>,
    label_position: &str,
) -> Option<SceneNode> {
    let label_text = label.filter(|t| !t.is_empty())?;

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

    Some(SceneNode::Text {
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
    })
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

// ── Break routing (#228 stratified clip) ────────────────────────────────────

/// Build every annotation on a panel, re-placing each one on any broken axis.
///
/// The single entry the scene builder uses, and the whole of the #228
/// stratified contract. An annotation that a gap actually reaches gets one of
/// three fates, decided by what its geometry MEANS rather than by whether a gap
/// touches it at all ([`BreakRouting`]):
///
/// - it is CUT — clipped or split into one piece per visible segment, exactly as
///   the equivalent mark geometry is, and SILENTLY: a clipped annotation
///   rendered, so there is nothing to warn about;
/// - it is DROPPED whole, when nothing of it survives the gaps or its kind
///   admits no honest split, and rolled into one warning per axis per mechanism;
/// - it is untouched, and built through the ordinary coordinate fold.
///
/// The two rollups (`AnnotationSuppressedByBreak` for a real coordinate inside a
/// gap, `AnnotationSpansBreakGap` for an interval that reaches across one) are
/// unchanged from the pre-clip contract, including their per-axis,
/// first-seen-order bundling (#240-3): they now report a strictly smaller
/// population, because everything that CAN be cut is. Each bundles every dropped
/// instance's kind into one `RenderWarning`, since the per-instance offending
/// value was never independently actionable (one fix regardless of instance:
/// move the gap, or move the annotation).
///
/// `ctx.breaks` is the SAME remap list the mark pass folds through — already
/// filtered to axes whose resolved scale has a continuous `data_domain()`, and
/// (post-#230-4) carrying at most one remap per axis, though it spans BOTH axes,
/// so every entry is still consulted. Empty for every chart without an effective
/// `BreakAxis`, which short-circuits to the plain build: un-broken and
/// ordinal-axis-broken charts stay byte-identical and push no warnings.
pub fn build_annotations_through_breaks(
    specs: &[AnnotationSpec],
    ctx: &ScaleContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) -> AnnotationNodes {
    if ctx.breaks.is_empty() {
        return build_annotations(specs, ctx);
    }
    // Per-axis rollup accumulators (#240-3): small `Vec`s, not `HashMap`s, so
    // the emitted warnings stay in first-seen-axis order — deterministic for a
    // per-render count never large enough for O(n^2) lookup to matter.
    let mut in_gap: Vec<(String, Vec<String>)> = Vec::new();
    let mut spanning: Vec<(String, Vec<String>)> = Vec::new();
    let mut nodes = AnnotationNodes {
        below_marks: Vec::new(),
        above_marks: Vec::with_capacity(specs.len()),
    };
    for spec in specs {
        match break_fate(spec, ctx.breaks) {
            BreakFate::Untouched => build_one(spec, ctx, z_bucket(spec, &mut nodes)),
            BreakFate::Cut { axes, routing } => {
                build_cut(spec, ctx, &axes, routing, z_bucket(spec, &mut nodes))
            }
            BreakFate::Suppressed(BreakConflict::Point { axis, kind }) => {
                accumulate_by_axis(&mut in_gap, axis, kind)
            }
            BreakFate::Suppressed(BreakConflict::Span { axis, kind }) => {
                accumulate_by_axis(&mut spanning, axis, kind)
            }
        }
    }
    for (axis, kinds) in in_gap {
        warnings.push(RenderWarning::AnnotationSuppressedByBreak { axis, kinds });
    }
    for (axis, kinds) in spanning {
        warnings.push(RenderWarning::AnnotationSpansBreakGap { axis, kinds });
    }
    nodes
}

/// Build one CUT annotation: against the unbroken scale, then re-placed piece by
/// piece through each break that cuts it.
///
/// The clip/split machinery performs its own pixel→data→broken-pixel round trip,
/// so the geometry it receives must NOT have been folded already — hence the
/// context rebuilt here, carrying only the breaks this spec is not routed
/// through. Those are the axes on which this annotation is not an extent at all
/// (a rect with one pixel-space bound, say): their coordinates still fold, so a
/// cut on one axis never costs an annotation its placement on the other.
///
/// Multiple cutting axes are applied in the panel's own remap order, which is
/// the order `remap_mark_batch_through_break` applies them to marks — the two
/// compose identically or not at all.
fn build_cut(
    spec: &AnnotationSpec,
    ctx: &ScaleContext<'_>,
    axes: &[usize],
    routing: CutRouting,
    out: &mut Vec<SceneNode>,
) {
    let unrouted: Vec<break_axis::BreakRemap<'_>> = ctx
        .breaks
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, remap)| *remap)
        .collect();
    let unrouted_ctx = ScaleContext {
        plot_area: ctx.plot_area,
        x_scale: ctx.x_scale,
        y_scale: ctx.y_scale,
        breaks: &unrouted,
    };

    let mut nodes: Vec<SceneNode> = Vec::new();
    // A span's label is held back and re-anchored below, on the piece that
    // survives largest; every other kind's nodes all go through the machinery.
    let band_only = span_band_only(spec);
    build_one(band_only.as_ref().unwrap_or(spec), &unrouted_ctx, &mut nodes);
    for &i in axes {
        nodes = routing.place(&ctx.breaks[i], nodes);
    }
    if let Some(label) = clipped_span_label(spec, &nodes) {
        nodes.push(label);
    }
    out.append(&mut nodes);
}

/// How a `BreakAxis` gap re-places one annotation kind's geometry (#228, D1).
///
/// The stratification IS the #228 decision: an annotation whose geometry can be
/// split honestly is split — exactly as the equivalent mark geometry is — and
/// only what cannot be is dropped whole. Which is which follows from what the
/// annotation's coordinates mean, the same question [`BreakPlacement`] asks of a
/// mark batch.
///
/// [`BreakPlacement`]: super::break_axis::BreakPlacement
#[derive(Clone, Copy)]
enum BreakRouting {
    /// The break CUTS this kind: built against the unbroken scale and re-placed
    /// one piece per visible segment, silently.
    Cut(CutRouting),
    /// The break DROPS this kind whole once it meets a gap, rolling it into the
    /// per-axis suppression warning.
    ///
    /// Two populations, one answer. The point-anchored kinds (`text`, `callout`,
    /// `image`) have no extent to cut: their coordinate is either inside a gap
    /// or is not, and a callout's background rect is a TEXT BOX — drawn size
    /// around an anchor, never data — so it is never clipped either.
    ///
    /// `bracket` is the one extent-shaped kind here, and PERMANENTLY so (spec §4
    /// #228, D1 — not a residual awaiting a follow-up). Its two endpoint TIPS
    /// are what say where the bracketed range begins and ends; a cut at a gap
    /// edge would either strand those tips at positions the data never had or
    /// invent new tips at the cut, so a bracket reaching across a gap asserts a
    /// continuity the broken axis cannot carry. Dropping it whole, with the
    /// warning naming the axis, is the honest answer.
    SuppressWhole,
}

/// The three ways a cut annotation's nodes go through the break machinery —
/// [`BreakRouting`] minus the drop case, so the router cannot reach a clip with a
/// kind that has no clip.
#[derive(Clone, Copy)]
enum CutRouting {
    /// `rect`'s rect and `span`'s band: the pixel extent along the broken axis
    /// IS data, so it clips per visible segment exactly as a bar's height does
    /// (annotation rects lower to the same `SceneNode::Rect` shape).
    ClipExtent,
    /// `line`'s shaft: connected stroked geometry whose every vertex is its own
    /// data position, so it is cut at the gap edges it crosses, with the
    /// cross-axis coordinate interpolated at each crossing.
    SplitStroke,
    /// `arrow`: a split shaft plus the head that rides the piece reaching its
    /// tip. [`emit_arrow`] pushes `[shaft Line, head Path]` — the same node pair,
    /// in the same order, `marks::segment::build` emits for `arrow=True` — so
    /// the segment splitter's shaft/head pairing (head rebuilt on the tip piece,
    /// dropped when the tip is in a gap) applies verbatim.
    SplitShaftAndHead,
}

impl CutRouting {
    /// Re-place `nodes` through one break.
    fn place(self, remap: &break_axis::BreakRemap<'_>, nodes: Vec<SceneNode>) -> Vec<SceneNode> {
        match self {
            Self::ClipExtent => remap.clip_nodes(nodes).nodes,
            Self::SplitStroke => remap.split_nodes(nodes).nodes,
            Self::SplitShaftAndHead => remap.split_segment_nodes(nodes).nodes,
        }
    }
}

/// The routing `spec`'s kind takes at a break.
///
/// Exhaustive over `AnnotationSpec`'s 8 shapes, like [`kind_name`] and
/// [`data_coords_on_axis`], so a new kind is a compile error here rather than
/// silently inheriting one of these answers.
fn break_routing(spec: &AnnotationSpec) -> BreakRouting {
    match spec {
        AnnotationSpec::Rect { .. } | AnnotationSpec::Span { .. } => {
            BreakRouting::Cut(CutRouting::ClipExtent)
        }
        AnnotationSpec::Line { .. } => BreakRouting::Cut(CutRouting::SplitStroke),
        AnnotationSpec::Arrow { .. } => BreakRouting::Cut(CutRouting::SplitShaftAndHead),
        AnnotationSpec::Bracket { .. }
        | AnnotationSpec::Text { .. }
        | AnnotationSpec::Callout { .. }
        | AnnotationSpec::Image { .. } => BreakRouting::SuppressWhole,
    }
}

/// One annotation's verdict under all of the panel's breaks.
enum BreakFate {
    /// No gap reaches it — built through the ordinary coordinate fold, which is
    /// what keeps an annotation on a chart whose break misses it byte-identical.
    Untouched,
    /// Cut by the breaks at `axes` (indices into the panel's remap list) and
    /// re-placed by `routing`. Silent: a cut annotation RENDERED.
    Cut { axes: Vec<usize>, routing: CutRouting },
    /// Dropped whole, and rolled into `conflict`'s per-axis warning.
    Suppressed(BreakConflict),
}

/// One break's answer for one spec, before the per-axis answers combine into a
/// [`BreakFate`].
enum AxisFate {
    Untouched,
    Cut,
    Suppressed(BreakConflict),
}

/// Which mechanism dropped one annotation, before it is folded into that
/// mechanism's per-axis rollup — see [`build_annotations_through_breaks`].
enum BreakConflict {
    /// A real data coordinate of the annotation lies inside a gap.
    Point { axis: String, kind: String },
    /// No coordinate does, but the annotation's interval reaches across a gap.
    Span { axis: String, kind: String },
}

/// Combine every break's answer for `spec` into one verdict.
fn break_fate(spec: &AnnotationSpec, breaks: &[break_axis::BreakRemap<'_>]) -> BreakFate {
    let routing = break_routing(spec);
    let mut axes: Vec<usize> = Vec::new();
    for (i, remap) in breaks.iter().enumerate() {
        match axis_break_fate(spec, remap, routing) {
            AxisFate::Untouched => {}
            AxisFate::Cut => axes.push(i),
            // A drop on ANY axis wins over a cut on another: if nothing of the
            // annotation survives one axis's gaps, clipping it on the other
            // would render pieces of something the reader was told is gone.
            AxisFate::Suppressed(conflict) => return BreakFate::Suppressed(conflict),
        }
    }
    match routing {
        BreakRouting::Cut(routing) if !axes.is_empty() => BreakFate::Cut { axes, routing },
        _ => BreakFate::Untouched,
    }
}

/// What one break does to one spec.
///
/// Two questions, one per coordinate shape ([`AxisCoords`]):
///
/// - a POINT coordinate conflicts when it lands inside a gap, and a point has
///   nothing to cut — in-gap is the whole story;
/// - an EXTENT's interval conflicts when it MEETS a gap: an endpoint inside one,
///   or (the containment inversion) the interval reaching across one. A
///   cut-routed kind is then CLIPPED whenever any of its interval survives, and
///   dropped only when none of it does.
///
/// That last rule is what retired the endpoint-first suppression shortcut
/// (#228): an in-gap endpoint used to suppress the whole annotation, so
/// `annotate_rect(y1=100, y2=400)` under a `(150, 850)` gap vanished instead of
/// rendering its visible 100–150 portion. The endpoint test survives for one
/// narrower job — choosing WHICH rollup a dropped annotation lands in, since
/// "a coordinate of yours is in the gap" and "your interval crosses the gap" are
/// different, equally true facts.
fn axis_break_fate(
    spec: &AnnotationSpec,
    remap: &break_axis::BreakRemap<'_>,
    routing: BreakRouting,
) -> AxisFate {
    let axis = remap.axis();
    let result = remap.result();
    let in_gap = |value: &f64| break_axis::gap_containing(*value, result).is_some();
    let dropped = |spanning: bool| {
        let (axis, kind) = (axis.to_string(), kind_name(spec).to_string());
        AxisFate::Suppressed(if spanning {
            BreakConflict::Span { axis, kind }
        } else {
            BreakConflict::Point { axis, kind }
        })
    };
    match data_coords_on_axis(spec, axis) {
        AxisCoords::Points(values) => {
            if values.iter().any(in_gap) {
                dropped(false)
            } else {
                AxisFate::Untouched
            }
        }
        AxisCoords::Extent(lo, hi) => {
            let endpoint_in_gap = [lo, hi].iter().any(in_gap);
            if !endpoint_in_gap && break_axis::gap_spanned_by(lo, hi, result).is_none() {
                return AxisFate::Untouched;
            }
            match routing {
                BreakRouting::Cut(_) if remap.data_extent_visible(lo, hi) => AxisFate::Cut,
                _ => dropped(!endpoint_in_gap),
            }
        }
    }
}

/// Append `kind` to `axis`'s bucket in a per-axis rollup accumulator, creating
/// the bucket on first sight. Shared by both drop mechanisms.
fn accumulate_by_axis(acc: &mut Vec<(String, Vec<String>)>, axis: String, kind: String) {
    match acc.iter_mut().find(|(a, _)| *a == axis) {
        Some((_, kinds)) => kinds.push(kind),
        None => acc.push((axis, vec![kind])),
    }
}

/// `spec` with a `Span`'s label removed, or `None` when there is nothing to
/// remove (any other kind, or a span with no label).
///
/// The band is clipped on its own and the label re-anchored afterwards
/// ([`clipped_span_label`]), so the label never rides the pre-clip midpoint —
/// which, for the straddling span this exists for, is inside the gap.
fn span_band_only(spec: &AnnotationSpec) -> Option<AnnotationSpec> {
    let AnnotationSpec::Span { label: Some(_), .. } = spec else {
        return None;
    };
    let mut band = spec.clone();
    if let AnnotationSpec::Span { label, .. } = &mut band {
        *label = None;
    }
    Some(band)
}

/// A clipped span's label, anchored on the largest surviving piece.
///
/// `pieces` is the band AFTER the clip: one `Rect` per visible segment. The
/// label rides the LARGEST of them — the piece with room for it, and the one a
/// reader takes as the band's body — measured along the span's own axis. Ties
/// keep the first piece, so the choice is deterministic and reads in segment
/// order. `None` when the span carries no label, when `spec` is not a span at
/// all, or when no piece survived (in which case the caller has nothing to draw
/// a label on).
fn clipped_span_label(spec: &AnnotationSpec, pieces: &[SceneNode]) -> Option<SceneNode> {
    let AnnotationSpec::Span { axis, label, label_position, .. } = spec else {
        return None;
    };
    let label = label.as_deref()?;
    let along_is_y = axis == "y";
    let along = |piece: &(f64, f64, f64, f64)| if along_is_y { piece.3 } else { piece.2 };
    let (x, y, w, h) = pieces
        .iter()
        .filter_map(rect_extent)
        .reduce(|a, b| if along(&b) > along(&a) { b } else { a })?;
    span_label_node(x, y, w, h, Some(label), label_position)
}

/// `(x, y, w, h)` of a `Rect` node — the shape a clipped span piece has.
fn rect_extent(node: &SceneNode) -> Option<(f64, f64, f64, f64)> {
    match node {
        SceneNode::Rect { x, y, w, h, .. } => Some((*x, *y, *w, *h)),
        _ => None,
    }
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

    // ── Per-variant wire key drift tests (#92/#148) ──────────────────────────
    //
    // Mines each variant's REAL accepted-field set out of its own
    // `deny_unknown_fields` error (a minimal valid `type` tag plus one bogus
    // key — the tag is required to select the variant, unlike
    // `chart_config.rs`'s plain-struct `mined_fields_of`, which has no tag to
    // supply), and asserts it equals the corresponding const minus `type`
    // (the tag itself is consumed by serde before the variant's own
    // `deny_unknown_fields` check runs, so it never appears in the mined set
    // — see `AnnotationSpec`'s own doc comment).

    fn mined_annotation_variant_fields(type_tag: &str) -> std::collections::BTreeSet<String> {
        let json = format!(r#"{{"type": "{type_tag}", "__definitely_not_a_real_field__": 1}}"#);
        let res: Result<AnnotationSpec, _> = serde_json::from_str(&json);
        let err = res.expect_err("bogus key must be rejected by deny_unknown_fields");
        super::super::chart_config::accepted_fields_from_deny_unknown_error(&err.to_string())
    }

    /// Shared assertion body for every `annotation_*_keys_match_serde` test
    /// below — a plain function rather than a macro (no behavior worth
    /// obscuring, per this crate's macro-avoidance convention).
    fn assert_annotation_variant_keys_match_serde(type_tag: &str, keys_const: &[&str]) {
        let mined = mined_annotation_variant_fields(type_tag);
        let expected = keys_minus_type(keys_const);
        assert_eq!(
            mined, expected,
            "annotation `{type_tag}` variant's key const drifted from AnnotationSpec.\n\
             in serde but not const: {:?}\nin const but not serde: {:?}",
            mined.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&mined).collect::<Vec<_>>(),
        );
    }

    fn keys_minus_type(keys: &[&str]) -> std::collections::BTreeSet<String> {
        keys.iter().filter(|&&k| k != "type").map(|s| s.to_string()).collect()
    }

    #[test]
    fn annotation_text_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("text", ANNOTATION_TEXT_KEYS);
    }
    #[test]
    fn annotation_arrow_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("arrow", ANNOTATION_ARROW_KEYS);
    }
    #[test]
    fn annotation_rect_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("rect", ANNOTATION_RECT_KEYS);
    }
    #[test]
    fn annotation_line_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("line", ANNOTATION_LINE_KEYS);
    }
    #[test]
    fn annotation_span_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("span", ANNOTATION_SPAN_KEYS);
    }
    #[test]
    fn annotation_bracket_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("bracket", ANNOTATION_BRACKET_KEYS);
    }
    #[test]
    fn annotation_callout_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("callout", ANNOTATION_CALLOUT_KEYS);
    }
    #[test]
    fn annotation_image_keys_match_serde() {
        assert_annotation_variant_keys_match_serde("image", ANNOTATION_IMAGE_KEYS);
    }

    /// `accepted_keys_for_annotation_variant` covers every real `AnnotationSpec`
    /// tag and only those — the completeness half of closing #148's array
    /// carve-out (the inverse of the pre-fix state, where every tag returned
    /// no per-item key set at all because the gate never looked).
    #[test]
    fn accepted_keys_for_annotation_variant_covers_every_tag() {
        for tag in ["text", "arrow", "rect", "line", "span", "bracket", "callout", "image"] {
            assert!(
                accepted_keys_for_annotation_variant(tag).is_some(),
                "tag `{tag}` must have an accepted-key set"
            );
        }
        assert!(accepted_keys_for_annotation_variant("not_a_real_type").is_none());
    }

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
        assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Points(vec![10.0]));
        assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Points(vec![20.0]));
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
        assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Points(Vec::new()));
        assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Points(Vec::new()));
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
        assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Extent(1.0, 3.0));
        assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Extent(2.0, 4.0));
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
            assert_eq!(data_coords_on_axis(spec, "x"), AxisCoords::Extent(1.0, 3.0));
            assert_eq!(data_coords_on_axis(spec, "y"), AxisCoords::Extent(2.0, 4.0));
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
        assert_eq!(data_coords_on_axis(&x_span, "x"), AxisCoords::Extent(1.0, 2.0));
        assert_eq!(data_coords_on_axis(&x_span, "y"), AxisCoords::Points(Vec::new()));

        let y_span = AnnotationSpec::Span {
            axis: "y".into(),
            start: CoordValue::Data(5.0),
            end: CoordValue::Data(6.0),
            fill: "#ccc".into(),
            opacity: 0.2,
            label_position: "center".into(),
            label: None,
        };
        assert_eq!(data_coords_on_axis(&y_span, "y"), AxisCoords::Extent(5.0, 6.0));
        assert_eq!(data_coords_on_axis(&y_span, "x"), AxisCoords::Points(Vec::new()));
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
        assert_eq!(
            data_coords_on_axis(&with_text_pos, "x"),
            AxisCoords::Points(vec![1.0, 11.0]),
            "a callout's anchor and text anchor are two points, never one interval"
        );
        assert_eq!(
            data_coords_on_axis(&with_text_pos, "y"),
            AxisCoords::Points(vec![2.0, 12.0])
        );

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
        assert_eq!(data_coords_on_axis(&no_text_pos, "x"), AxisCoords::Points(vec![1.0]));
        assert_eq!(data_coords_on_axis(&no_text_pos, "y"), AxisCoords::Points(vec![2.0]));
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
        assert_eq!(data_coords_on_axis(&spec, "x"), AxisCoords::Points(vec![7.0]));
        assert_eq!(data_coords_on_axis(&spec, "y"), AxisCoords::Points(vec![8.0]));
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

    // `resolve_x_folds_through_multiple_same_axis_breaks` and
    // `resolve_x_folds_same_axis_breaks_in_declaration_order` (quality-review
    // cycle-2 finding #2) used to live here, each hand-building a
    // `ScaleContext` whose `breaks` list held TWO same-axis `BreakRemap`s to
    // pin that `fold_breaks` composes every same-axis break in declaration
    // order rather than taking only the first.
    //
    // #230-4's same-axis merge (`scene_build::build_structural_nodes`) makes
    // that construction unreachable from production: N same-axis `BreakAxis`
    // specs now merge into ONE `BreakResult` — one gap list, run through
    // `normalize_gaps` once — before `break_remaps` is ever built, so
    // `ScaleContext::breaks` carries at most one remap per axis. Composing
    // two independently-built same-axis `BreakRemap`s (each resolved against
    // the ORIGINAL, unbroken domain) was itself the numerically-wrong
    // behavior #230-4 replaces — sequential composition re-interprets the
    // first break's already-broken pixel as an unbroken-domain data value,
    // and (via `BreakRemap::data_at`'s domain clamp) could resurrect a
    // coordinate the first break had hidden. Keeping these two tests green
    // would pin that replaced behavior as if it were still a supported
    // shape, so they are retired rather than kept passing.
    //
    // `fold_breaks` itself is unchanged and stays exercised: with the merge
    // upstream, its loop now always executes 0 or 1 times per axis, and the
    // cross-axis case (independent x and y folds) is what
    // `resolve_y_pixel_and_norm_coords_are_unaffected_by_a_break` below and
    // `test_two_same_axis_breaks_fold_annotations_and_marks_to_the_same_position`
    // (`tests/test_break_axis_geometry.py`) pin end to end. The same-axis
    // merge and the `BREAK_HIDDEN` poison guard are pinned directly in
    // `scene_build.rs` and `break_axis.rs`, closer to where they are now
    // implemented.

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

    // ── Break routing (#228 stratified clip, D1) ─────────────────────────────
    //
    // These pins moved here from `scene_build.rs`, where they tested the
    // `suppress_annotations_for_break` filter this routing replaces. Three of
    // them were REWRITTEN rather than moved, because #228 retires their old
    // answers: an extent that spans a gap is no longer dropped (it clips), N
    // spanning rects no longer roll up as suppressed (nothing is suppressed),
    // and an extent with an in-gap ENDPOINT is no longer dropped whole (it
    // clips to whatever survives). The suppression they pinned is now the
    // narrower fact it always should have been: nothing survives, or the kind
    // admits no honest split.
    //
    // All geometry below is hand-checkable against one fixture: domain
    // [0, 1000] on y pixels 320 (bottom) → 20 (top), 300 pixels total, 12 of
    // them reserved per break indicator.

    /// A spec from its wire JSON — shorter than an 8-field literal, and it
    /// exercises the same `serde` defaults a real chart's annotations arrive
    /// with.
    fn spec(json: &str) -> AnnotationSpec {
        serde_json::from_str(json).expect("test annotation spec must deserialize")
    }

    /// A y-`Rect` spec with the given data bounds; every other coordinate is
    /// norm, so only the y interval is under test.
    fn y_rect(y1: f64, y2: f64) -> AnnotationSpec {
        spec(&format!(
            r#"{{"type":"rect","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2}}}"#
        ))
    }

    /// Like [`y_rect`], with an explicit `corner_radius` — for #230-2
    /// coverage of `annotate_rect(corner_radius=...)` straddling a gap (T7
    /// routed `rect`/`span` annotations through the same clip machinery as
    /// marks, so they inherited the mark-side defect; this is the annotation
    /// half of that fix's regression coverage).
    fn y_rect_with_radius(y1: f64, y2: f64, corner_radius: f64) -> AnnotationSpec {
        spec(&format!(
            r#"{{"type":"rect","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2},"corner_radius":{corner_radius}}}"#
        ))
    }

    /// A y-`Bracket` spec with the given data bounds — the kind that is
    /// PERMANENTLY suppress-whole.
    fn y_bracket(y1: f64, y2: f64) -> AnnotationSpec {
        spec(&format!(
            r#"{{"type":"bracket","x1":{{"norm":0.0}},"y1":{y1},"x2":{{"norm":1.0}},"y2":{y2},"label":"b"}}"#
        ))
    }

    /// A `Text` spec anchored at a y data value; its x is norm, so an x break
    /// never reaches it.
    fn y_text(y: f64, text: &str) -> AnnotationSpec {
        spec(&format!(
            r#"{{"type":"text","x":{{"norm":1.0}},"y":{y},"text":"{text}"}}"#
        ))
    }

    /// The broken panel every routing test shares: domain [0, 1000] on both
    /// axes, y mapped 320 (bottom) → 20 (top) over 300 pixels and x 50 → 550
    /// over 500, each broken by `gaps` with a 12px indicator.
    ///
    /// One `BreakResult` per axis, because a result's segments carry PIXEL
    /// bounds: sharing one across both axes would place x geometry on the y
    /// axis's pixels.
    struct BrokenPanel {
        x_scale: ScaleKind,
        y_scale: ScaleKind,
        plot_area: Rect,
        x_result: break_axis::BreakResult,
        y_result: break_axis::BreakResult,
    }

    const X_PIXELS: (f64, f64) = (50.0, 550.0);
    const Y_PIXELS: (f64, f64) = (320.0, 20.0);

    fn broken_panel(gaps: &[[f64; 2]]) -> BrokenPanel {
        use crate::scale::linear::LinearScale;
        BrokenPanel {
            x_scale: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1000.0],
                vec![X_PIXELS.0, X_PIXELS.1],
                false,
                false,
            )),
            y_scale: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1000.0],
                vec![Y_PIXELS.0, Y_PIXELS.1],
                false,
                false,
            )),
            plot_area: Rect { x: 50.0, y: 20.0, w: 500.0, h: 300.0 },
            x_result: break_axis::apply_break_to_scale((0.0, 1000.0), gaps, X_PIXELS, 12.0),
            y_result: break_axis::apply_break_to_scale((0.0, 1000.0), gaps, Y_PIXELS, 12.0),
        }
    }

    impl BrokenPanel {
        /// The remap for one axis, built exactly as `scene_build` builds it:
        /// the scale's own domain and pixel range.
        fn remap(&self, axis: &'static str) -> break_axis::BreakRemap<'_> {
            let (pixels, result) = if axis == "y" {
                (Y_PIXELS, &self.y_result)
            } else {
                (X_PIXELS, &self.x_result)
            };
            break_axis::BreakRemap::new(axis, (0.0, 1000.0), pixels, result)
        }

        /// Build `specs` through `breaks`, returning the above-marks nodes and
        /// the warnings pushed.
        fn build(
            &self,
            specs: &[AnnotationSpec],
            breaks: &[break_axis::BreakRemap<'_>],
        ) -> (Vec<SceneNode>, Vec<RenderWarning>) {
            let ctx = ScaleContext {
                plot_area: self.plot_area,
                x_scale: &self.x_scale,
                y_scale: &self.y_scale,
                breaks,
            };
            let mut warnings = Vec::new();
            let nodes = build_annotations_through_breaks(specs, &ctx, &mut warnings);
            assert!(
                nodes.below_marks.is_empty(),
                "these fixtures use above-marks specs only"
            );
            (nodes.above_marks, warnings)
        }
    }

    /// `(y, h)` of every `Rect` node, in paint order — how an extent's clipped
    /// pieces are read.
    fn rect_bands(nodes: &[SceneNode]) -> Vec<(f64, f64)> {
        nodes.iter().filter_map(rect_extent).map(|(_, y, _, h)| (y, h)).collect()
    }

    /// The intent-ledger headline: `annotate_rect(y1=0, y2=1000)` on a
    /// `[0, 1000]` axis broken at `(150, 850)` renders TWO pieces, silently.
    ///
    /// Pre-#228 the same rect was dropped whole with an
    /// `AnnotationSpansBreakGap` warning, while identical MARK geometry (a bar
    /// of the same extent) clipped per segment — the same chart answering the
    /// same question two ways.
    #[test]
    fn a_straddling_rect_clips_into_one_piece_per_visible_segment() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_rect(0.0, 1000.0)], &[y]);

        // 288 data pixels split 150:150 → 144 each. Segment 0 (data 0–150)
        // runs 320 → 176; the indicator takes 176 → 164; segment 1 (data
        // 850–1000) runs 164 → 20.
        assert_eq!(
            rect_bands(&nodes),
            vec![(176.0, 144.0), (20.0, 144.0)],
            "one piece per visible segment, cut at the gap edges"
        );
        assert!(
            warnings.is_empty(),
            "a successful clip is SILENT, like a clipped mark; got {warnings:?}"
        );
    }

    /// #230-2, the annotation half: `annotate_rect(corner_radius=8)`
    /// straddling a gap must square the cut edge on each piece, not carry
    /// `corner_radius` onto it verbatim. `annotate_rect` clips through the
    /// exact same [`break_axis::BreakRemap::clip_nodes`] a mark batch does
    /// (see [`a_straddling_rect_clips_into_one_piece_per_visible_segment`]),
    /// so a fix scoped only to the mark-facing call site would leave this
    /// broken — this proves the annotation path gets it too.
    #[test]
    fn a_straddling_rounded_rect_squares_only_its_cut_edge() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_rect_with_radius(0.0, 1000.0, 8.0)], &[y]);

        assert_eq!(nodes.len(), 2, "same two pieces as the unrounded case; got {nodes:?}");
        assert!(warnings.is_empty(), "a successful clip is SILENT; got {warnings:?}");

        // Piece 0 (pixel 176..320, data 0..150): y=320 is the real baseline
        // (data 0) — uncut, rounded. y=176 sits against the gap — cut,
        // squared.
        let lower = path_cmds(&nodes[0]);
        assert_eq!(arc_count(&lower), 2, "only the two uncut (bottom) corners get an arc; got {lower:?}");
        for cmd in &lower {
            if let PathCmd::ArcTo { y, .. } = cmd {
                assert!(*y > 176.0 + 1e-6, "an arc must not sit on the cut (top) edge; got {cmd:?}");
            }
        }

        // Piece 1 (pixel 20..164, data 850..1000): y=20 is the real top
        // (data 1000) — uncut, rounded. y=164 sits against the gap — cut,
        // squared.
        let upper = path_cmds(&nodes[1]);
        assert_eq!(arc_count(&upper), 2, "only the two uncut (top) corners get an arc; got {upper:?}");
        for cmd in &upper {
            if let PathCmd::ArcTo { y, .. } = cmd {
                assert!(*y < 164.0 - 1e-6, "an arc must not sit on the cut (bottom) edge; got {cmd:?}");
            }
        }
    }

    /// The commands of a `Path` node, panicking on anything else — the
    /// squared-corner pieces #230-2 produces for a cut, rounded rect.
    fn path_cmds(node: &SceneNode) -> Vec<PathCmd> {
        let SceneNode::Path { commands, .. } = node else {
            panic!("expected a Path (rounded, cut piece); got {node:?}")
        };
        commands.clone()
    }

    /// The number of `ArcTo` commands in a path — how many of a clipped
    /// rect piece's four corners kept their rounding.
    fn arc_count(cmds: &[PathCmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count()
    }

    /// The other half of that contract: a rect with nothing outside the gap is
    /// still dropped whole, and still rolls up as an in-gap suppression (both
    /// of its endpoints are real coordinates inside the gap).
    #[test]
    fn a_rect_wholly_inside_a_gap_is_suppressed_and_rolled_up() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_rect(300.0, 700.0)], &[y]);

        assert!(nodes.is_empty(), "nothing of it is visible, so nothing renders");
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["rect".to_string()]);
            }
            other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
        }
    }

    /// The retired endpoint-first shortcut (rewrite of
    /// `suppress_annotations_for_break_keeps_the_endpoint_answer_for_extents`):
    /// an extent with an in-gap ENDPOINT whose surviving portion is non-empty
    /// CLIPS to that portion. Pre-#228 the in-gap endpoint suppressed the whole
    /// rect, so `annotate_rect(y1=100, y2=400)` under a `(150, 850)` gap
    /// vanished instead of rendering its visible 100–150 band.
    #[test]
    fn an_extent_with_an_in_gap_endpoint_clips_to_its_surviving_portion() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_rect(100.0, 400.0)], &[y]);

        // Data 100 → 320 − (100/150)·144 = 224; data 150 → 176.
        assert_eq!(
            rect_bands(&nodes),
            vec![(176.0, 48.0)],
            "the visible 100–150 portion renders, cut at the gap edge"
        );
        assert!(warnings.is_empty(), "clipping is silent; got {warnings:?}");
    }

    /// The unchanged half of that same pin: an extent wholly on one side of the
    /// gap is untouched — no clip, no warning, one node.
    #[test]
    fn an_extent_clear_of_every_gap_is_untouched() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_rect(0.0, 140.0)], &[y]);

        // Data 140 → 320 − (140/150)·144 = 185.6; data 0 → 320.
        assert_eq!(rect_bands(&nodes), vec![(185.6, 134.4)]);
        assert!(warnings.is_empty());
    }

    /// The knife edge where a clip-routed extent STILL suppresses as spanning:
    /// a rect whose bounds are exactly the gap's bounds reaches across the gap
    /// (neither endpoint is inside it — gaps are open intervals) yet contributes
    /// no length to any segment, so nothing survives and the drop rolls into
    /// `AnnotationSpansBreakGap`.
    #[test]
    fn a_rect_exactly_covering_the_gap_rolls_up_as_spanning() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_rect(150.0, 850.0)], &[y]);

        assert!(nodes.is_empty());
        match warnings.as_slice() {
            [RenderWarning::AnnotationSpansBreakGap { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["rect".to_string()]);
            }
            other => panic!("expected one AnnotationSpansBreakGap, got {other:?}"),
        }
    }

    /// `bracket` is suppress-whole PERMANENTLY (spec §4 #228, D1 — not a
    /// residual): its two endpoint tips are what say where the bracketed range
    /// begins and ends, so a cut at a gap edge would either strand them at
    /// positions the data never had or invent new tips at the cut. A bracket
    /// straddling a gap is therefore dropped where a rect of the same extent
    /// clips.
    #[test]
    fn a_bracket_straddling_a_gap_is_dropped_whole_not_split() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let (nodes, warnings) = panel.build(&[y_bracket(100.0, 900.0)], &[y]);

        assert!(nodes.is_empty(), "a bracket admits no honest split");
        match warnings.as_slice() {
            [RenderWarning::AnnotationSpansBreakGap { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["bracket".to_string()]);
            }
            other => panic!("expected one AnnotationSpansBreakGap, got {other:?}"),
        }
    }

    /// #240-3's rollup, re-aimed at the population that still suppresses
    /// (rewrite of
    /// `suppress_annotations_for_break_rolls_up_n_spanning_annotations_into_one_warning`,
    /// whose three straddling RECTS now clip instead): N brackets spanning the
    /// same axis's gap roll up into exactly ONE warning carrying every kind.
    #[test]
    fn n_spanning_annotations_roll_up_into_one_warning() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let specs = vec![
            y_bracket(100.0, 900.0),
            y_bracket(120.0, 880.0),
            y_bracket(140.0, 860.0),
        ];
        let (nodes, warnings) = panel.build(&specs, &[y]);

        assert!(nodes.is_empty());
        assert_eq!(warnings.len(), 1, "N drops, one warning; got {warnings:?}");
        match &warnings[0] {
            RenderWarning::AnnotationSpansBreakGap { axis, kinds } => {
                assert_eq!(axis, "y");
                assert_eq!(kinds.len(), 3, "the count of dropped instances survives");
                assert!(kinds.iter().all(|k| k == "bracket"));
            }
            other => panic!("expected AnnotationSpansBreakGap, got {other:?}"),
        }
    }

    /// A span's band clips like a rect, and its label anchors on the midpoint of
    /// the LARGEST surviving piece — not on the pre-clip band's midpoint, which
    /// belongs to no piece at all.
    #[test]
    fn a_span_clips_and_its_label_rides_the_largest_surviving_piece() {
        // Gap (200, 600): 288 data pixels split 200:400 → 96 and 192. Segment 0
        // (data 0–200) runs 320 → 224, the indicator 224 → 212, segment 1 (data
        // 600–1000) runs 212 → 20. The larger piece is segment 1's.
        let panel = broken_panel(&[[200.0, 600.0]]);
        let y = panel.remap("y");
        let span = spec(
            r#"{"type":"span","axis":"y","start":0.0,"end":1000.0,"label":"band",
                "label_position":"middle"}"#,
        );
        let (nodes, warnings) = panel.build(&[span], &[y]);

        assert_eq!(rect_bands(&nodes), vec![(224.0, 96.0), (20.0, 192.0)]);
        assert!(warnings.is_empty(), "clipping is silent; got {warnings:?}");

        let labels: Vec<(f64, f64)> = nodes
            .iter()
            .filter_map(|n| match n {
                SceneNode::Text { x, y, content, .. } if content == "band" => Some((*x, *y)),
                _ => None,
            })
            .collect();
        assert_eq!(
            labels,
            vec![(300.0, 116.0)],
            "one label, on the larger piece's midpoint (20 + 192/2), centered in \
             the plot (50 + 500/2)"
        );
    }

    /// The same rule's discriminating case: with two EQUAL pieces the label
    /// takes the first, and either way it can never land in the gap band — where
    /// the pre-clip midpoint of a full-domain span falls when the gap is centred.
    #[test]
    fn a_clipped_span_label_never_lands_inside_the_gap_band() {
        // Gap (400, 600) splits the 288 data pixels 144:144: segment 0 runs
        // 320 → 176, the indicator 176 → 164, segment 1 runs 164 → 20. The
        // pre-clip band is the whole range 20…320, whose midpoint is 170 —
        // inside the indicator band, which is exactly where the old pre-clip
        // placement would have drawn the label.
        let panel = broken_panel(&[[400.0, 600.0]]);
        let y = panel.remap("y");
        let span = spec(
            r#"{"type":"span","axis":"y","start":0.0,"end":1000.0,"label":"band",
                "label_position":"middle"}"#,
        );
        let (nodes, _) = panel.build(&[span], &[y]);

        let label_y = nodes
            .iter()
            .find_map(|n| match n {
                SceneNode::Text { y, .. } => Some(*y),
                _ => None,
            })
            .expect("the span's label must render");
        assert_eq!(label_y, 248.0, "first of two equal pieces: 176 + 144/2");
        assert!(
            !(164.0..=176.0).contains(&label_y),
            "the label must not land in the invisible gap band; got {label_y}"
        );
    }

    /// An annotation `line` splits through the vertex machinery (#230-1's
    /// splitter, one mechanism for marks and annotations alike): one sub-line
    /// per visible segment, cut at the gap edges, with the CROSS-axis
    /// coordinate interpolated at each crossing rather than dropped back to the
    /// last visible vertex.
    #[test]
    fn an_annotation_line_splits_at_the_gap_with_its_crossing_interpolated() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        // A diagonal from (norm 0, data 0) to (norm 1, data 1000): x runs
        // 50 → 550 as y runs 320 → 20.
        let line = spec(
            r#"{"type":"line","x1":{"norm":0.0},"y1":0.0,"x2":{"norm":1.0},"y2":1000.0}"#,
        );
        let (nodes, warnings) = panel.build(&[line], &[y]);

        let segments: Vec<(f64, f64, f64, f64)> = nodes
            .iter()
            .filter_map(|n| match n {
                SceneNode::Line { x1, y1, x2, y2, .. } => Some((*x1, *y1, *x2, *y2)),
                _ => None,
            })
            .collect();
        // The crossings are at data 150 (15% along the line → x = 125) and data
        // 850 (85% → x = 475), placed at the gap-edge pixels 176 and 164.
        assert_eq!(
            segments,
            vec![(50.0, 320.0, 125.0, 176.0), (475.0, 164.0, 550.0, 20.0)],
            "two pieces, each ending ON the gap edge at the x it actually crosses"
        );
        assert!(warnings.is_empty(), "splitting is silent; got {warnings:?}");
    }

    /// An `arrow`'s shaft splits like a line, and its head rides the piece that
    /// reaches the TIP — rebuilt there, keeping its drawn size and pointing
    /// along the piece a reader sees it attached to.
    #[test]
    fn an_arrow_head_rides_the_tip_piece_with_its_size_kept() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let arrow = spec(
            r#"{"type":"arrow","x":{"norm":0.0},"y":0.0,"x2":{"norm":1.0},"y2":1000.0,
                "head_size":10.0}"#,
        );
        let (nodes, warnings) = panel.build(&[arrow], &[y]);
        assert!(warnings.is_empty(), "splitting is silent; got {warnings:?}");

        let shafts: Vec<(f64, f64)> = nodes
            .iter()
            .filter_map(|n| match n {
                SceneNode::Line { x2, y2, .. } => Some((*x2, *y2)),
                _ => None,
            })
            .collect();
        assert_eq!(
            shafts,
            vec![(125.0, 176.0), (550.0, 20.0)],
            "the shaft splits in two; the second piece reaches the tip"
        );

        let heads: Vec<&SceneNode> = nodes
            .iter()
            .filter(|n| matches!(n, SceneNode::Path { closed: true, .. }))
            .collect();
        assert_eq!(heads.len(), 1, "exactly one head, on the tip piece");
        let (tip, size) = head_tip_and_size(heads[0]);
        assert_eq!(tip, (550.0, 20.0), "the head sits on the placed tip");
        assert!(
            (size - 10.0).abs() < 1e-9,
            "the head keeps its drawn size; got {size}"
        );
    }

    /// `(tip, head_size)` of an arrowhead path — `draw::arrowhead_triangle`
    /// writes `[tip, left, right]`, with the tip `head_size` from the base
    /// midpoint.
    fn head_tip_and_size(node: &SceneNode) -> ((f64, f64), f64) {
        let SceneNode::Path { commands, .. } = node else {
            panic!("expected the head Path, got {node:?}");
        };
        let points: Vec<(f64, f64)> = commands
            .iter()
            .filter_map(|cmd| match cmd {
                PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => Some((*x, *y)),
                _ => None,
            })
            .collect();
        assert_eq!(points.len(), 3, "an arrowhead is three vertices");
        let (tip, left, right) = (points[0], points[1], points[2]);
        let base_mid = ((left.0 + right.0) / 2.0, (left.1 + right.1) / 2.0);
        (tip, (tip.0 - base_mid.0).hypot(tip.1 - base_mid.1))
    }

    /// An arrow with nothing outside the gap is dropped whole and warns — the
    /// suppress path is still reachable for a cut-routed kind.
    #[test]
    fn an_arrow_wholly_inside_a_gap_is_suppressed_and_warns() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let arrow = spec(
            r#"{"type":"arrow","x":{"norm":0.0},"y":300.0,"x2":{"norm":1.0},"y2":700.0}"#,
        );
        let (nodes, warnings) = panel.build(&[arrow], &[y]);

        assert!(nodes.is_empty(), "no piece of the shaft survives, so no head either");
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["arrow".to_string()]);
            }
            other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
        }
    }

    /// An arrow whose TAIL is visible but whose TIP lies inside a gap: the tip
    /// has no piece to terminate the head on, so only the HEAD drops. The
    /// shaft's own visible piece still renders, truncated at the gap edge it
    /// ends on — clip-fate geometry (a faithful rendering of what is visible),
    /// not a silent drop of the annotation, the same fate a
    /// `mark_segment(arrow=True)` shaft gives an in-gap tip (#230-1). This is
    /// the case neither `an_arrow_head_rides_the_tip_piece_with_its_size_kept`
    /// (visible tip) nor `an_arrow_wholly_inside_a_gap_is_suppressed_and_warns`
    /// (nothing survives) covers — the only one where head and shaft disagree.
    #[test]
    fn an_arrow_with_an_in_gap_tip_drops_only_the_head() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let arrow = spec(
            r#"{"type":"arrow","x":{"norm":0.0},"y":0.0,"x2":{"norm":1.0},"y2":500.0,
                "head_size":10.0}"#,
        );
        let (nodes, warnings) = panel.build(&[arrow], &[y]);
        assert!(warnings.is_empty(), "the drop is silent; got {warnings:?}");

        let shafts: Vec<&SceneNode> =
            nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).collect();
        assert_eq!(
            shafts.len(),
            1,
            "the visible tail survives as one shaft piece; got {shafts:?}"
        );

        let heads: Vec<&SceneNode> = nodes
            .iter()
            .filter(|n| matches!(n, SceneNode::Path { closed: true, .. }))
            .collect();
        assert!(heads.is_empty(), "the head drops with its hidden tip; got {heads:?}");
    }

    /// Point-anchored kinds keep the plain membership test: in the gap → gone,
    /// outside it → built, whatever lies between them.
    #[test]
    fn point_kinds_keep_the_membership_test() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let specs = vec![
            y_text(500.0, "inside the break"),
            y_text(100.0, "outside the break"),
            spec(r#"{"type":"callout","x":{"norm":0.5},"y":500.0,"text":"in"}"#),
            spec(r#"{"type":"image","x":{"norm":0.5},"y":500.0,"src":"a.png"}"#),
        ];
        let (nodes, warnings) = panel.build(&specs, &[y]);

        let texts: Vec<&str> = nodes
            .iter()
            .filter_map(|n| match n {
                SceneNode::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["outside the break"]);
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["text", "callout", "image"]);
            }
            other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
        }
    }

    /// A callout's background rect is a TEXT BOX — drawn size around an anchor,
    /// never data — so it is never clipped, even when it straddles the gap band
    /// in pixels. The negative control for the clip's reach.
    #[test]
    fn a_callout_box_is_never_clipped_even_across_the_gap_band() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        // Anchor at data 125 places at pixel 200; the box centres 30px above
        // that (ty = 170) and is 22px tall (14 + 2·4 padding), so it covers
        // 159…181 — straight across the 164…176 indicator band.
        let callout = spec(r#"{"type":"callout","x":{"norm":0.5},"y":125.0,"text":"note"}"#);
        let (nodes, warnings) = panel.build(&[callout], &[y]);

        assert_eq!(
            rect_bands(&nodes),
            vec![(159.0, 22.0)],
            "ONE box, uncut, spanning the gap band"
        );
        assert!(warnings.is_empty());
    }

    /// The desugar fan-out, honestly: one `annotate_hline(label=...)` emits a
    /// `line` plus its own `text` at the same y, so under a y break the pair
    /// shares one fate and rolls into one warning naming both kinds.
    #[test]
    fn an_hline_and_its_label_share_one_fate_under_a_y_break() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let hline = |value: f64| {
            vec![
                spec(&format!(
                    r#"{{"type":"line","x1":{{"norm":0.0}},"y1":{value},"x2":{{"norm":1.0}},
                        "y2":{value}}}"#
                )),
                y_text(value, "label"),
            ]
        };

        let (dropped, warnings) = panel.build(&hline(500.0), &[y]);
        assert!(dropped.is_empty(), "an hline in the gap takes its label with it");
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["line", "text"], "both primitives, one warning");
            }
            other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
        }

        let (kept, warnings) = panel.build(&hline(100.0), &[y]);
        assert_eq!(kept.len(), 2, "a visible hline keeps both primitives");
        assert!(warnings.is_empty());
    }

    /// The mixed-fate half of that fan-out: on an x break, a full-width `line`
    /// CLIPS (its x extent is data and part of it survives) while a `text`
    /// anchored inside the gap point-suppresses. One annotation's pieces render;
    /// the other's warning names only the kind that was actually dropped.
    #[test]
    fn a_clipped_line_and_a_suppressed_label_report_only_the_drop() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let x = panel.remap("x");
        let specs = vec![
            spec(
                r#"{"type":"line","x1":0.0,"y1":{"norm":0.0},"x2":1000.0,"y2":{"norm":0.0}}"#,
            ),
            spec(r#"{"type":"text","x":500.0,"y":{"norm":0.5},"text":"mid"}"#),
        ];
        let (nodes, warnings) = panel.build(&specs, &[x]);

        let xs: Vec<(f64, f64)> = nodes
            .iter()
            .filter_map(|n| match n {
                SceneNode::Line { x1, x2, .. } => Some((*x1, *x2)),
                _ => None,
            })
            .collect();
        // x pixels 50 → 550: 488 data pixels split 150:150 → 244 each, so
        // data 150 lands at 294 and data 850 at 306 across the indicator.
        assert_eq!(
            xs,
            vec![(50.0, 294.0), (306.0, 550.0)],
            "the line clips into two visible pieces"
        );
        assert!(
            !nodes.iter().any(|n| matches!(n, SceneNode::Text { .. })),
            "the in-gap label is gone"
        );
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
                assert_eq!(axis, "x");
                assert_eq!(kinds, &["text".to_string()], "only the DROP is reported");
            }
            other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
        }
    }

    /// A drop on one axis beats a clip on the other: an annotation with nothing
    /// left on y is gone, not rendered in pieces along x.
    #[test]
    fn a_drop_on_one_axis_wins_over_a_cut_on_the_other() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let (x, y) = (panel.remap("x"), panel.remap("y"));
        // x spans the whole domain (clippable); y sits entirely in the gap.
        let rect = spec(
            r#"{"type":"rect","x1":0.0,"y1":300.0,"x2":1000.0,"y2":700.0}"#,
        );
        let (nodes, warnings) = panel.build(&[rect], &[x, y]);

        assert!(nodes.is_empty(), "nothing survives on y, so nothing renders");
        assert_eq!(warnings.len(), 1, "one drop, one warning; got {warnings:?}");
    }

    /// With no breaks the router IS the plain build: same nodes, no warnings.
    /// This is what keeps every un-broken chart byte-identical.
    #[test]
    fn no_breaks_routes_straight_to_the_plain_build() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let specs = vec![y_rect(0.0, 1000.0), y_text(500.0, "t")];
        let ctx = ScaleContext {
            plot_area: panel.plot_area,
            x_scale: &panel.x_scale,
            y_scale: &panel.y_scale,
            breaks: &[],
        };
        let mut warnings = Vec::new();
        let routed = build_annotations_through_breaks(&specs, &ctx, &mut warnings);
        let plain = build_annotations(&specs, &ctx);

        assert_eq!(routed.above_marks, plain.above_marks);
        assert_eq!(routed.below_marks, plain.below_marks);
        assert!(warnings.is_empty());
    }

    /// `z_bucket`'s second call site — the routed build, not just the plain
    /// one — is exercised too: a `z="below_marks"` Text untouched by the break
    /// still lands in `below_marks`, not `above_marks`. Every other fixture in
    /// this module goes through `BrokenPanel::build`, which asserts
    /// `nodes.below_marks.is_empty()` for every case, so without this pin that
    /// assertion is never falsified by a real below_marks case reaching the
    /// routed build.
    #[test]
    fn a_below_marks_text_untouched_by_the_break_still_routes_to_below_marks() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let text = AnnotationSpec::Text {
            x: CoordValue::Norm { norm: 0.5 },
            y: CoordValue::Data(100.0),
            text: "caption".to_string(),
            font_size: 12.0,
            color: "#000000".to_string(),
            anchor: "middle".to_string(),
            baseline: "middle".to_string(),
            angle: 0.0,
            dx: 0.0,
            dy: 0.0,
            z: "below_marks".to_string(),
        };
        let ctx = ScaleContext {
            plot_area: panel.plot_area,
            x_scale: &panel.x_scale,
            y_scale: &panel.y_scale,
            breaks: &[y],
        };
        let mut warnings = Vec::new();
        let nodes = build_annotations_through_breaks(&[text], &ctx, &mut warnings);

        assert!(warnings.is_empty());
        assert!(
            nodes.above_marks.is_empty(),
            "a below_marks Text must not land above marks; got {:?}",
            nodes.above_marks
        );
        assert_eq!(
            nodes.below_marks.len(),
            1,
            "z_bucket must still route it through the router"
        );
        match &nodes.below_marks[0] {
            SceneNode::Text { content, .. } => assert_eq!(content, "caption"),
            other => panic!("expected the below_marks Text, got {other:?}"),
        }
    }

    /// `RenderWarning::AnnotationSuppressedByBreak`'s third named population: a
    /// `Cut`-routed extent kind (`rect`) whose two bounds on the broken axis
    /// are NOT both data — one is pixel-space — degenerates to
    /// `AxisCoords::Points` on that axis. With only one data bound there is no
    /// interval to clip, so an in-gap value there drops the whole rect even
    /// though `rect`'s own routing is `Cut` — enumerated in the warning's own
    /// doc (`render/mod.rs`), not merely asserted here.
    #[test]
    fn a_mixed_coordinate_rect_drops_whole_when_its_data_bound_is_in_the_gap() {
        let panel = broken_panel(&[[150.0, 850.0]]);
        let y = panel.remap("y");
        let rect = spec(
            r#"{"type":"rect","x1":{"norm":0.0},"y1":500.0,"x2":{"norm":1.0},"y2":{"px":50.0}}"#,
        );
        let (nodes, warnings) = panel.build(&[rect], &[y]);

        assert!(nodes.is_empty(), "an unreadable interval cannot be clipped; got {nodes:?}");
        match warnings.as_slice() {
            [RenderWarning::AnnotationSuppressedByBreak { axis, kinds }] => {
                assert_eq!(axis, "y");
                assert_eq!(kinds, &["rect".to_string()]);
            }
            other => panic!("expected one AnnotationSuppressedByBreak, got {other:?}"),
        }
    }
}
