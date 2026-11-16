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
use super::draw::to_scene_color;
use super::marks::arrowhead::{arrowhead_path, arrowhead_triangle, DEFAULT_ARROW_HEAD_SIZE};
use super::scale_resolve::{data_pixel_x, data_pixel_y, ScaleKind};
use super::RenderWarning;
use crate::layout::geometry::Rect;
use crate::layout::ThemeInputs;
use crate::spec::axis::AlignAxis;
use crate::spec::coord::{public_spelling, suppresses_cartesian_axes, CoordKind};

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
    /// `{"category": "b"}`: a category label, placed by the named axis's own
    /// categorical scale (the band centre marks use). A span whose two ends are
    /// both categories covers whole bands instead.
    Category { category: String },
    /// Plain number — data-space value mapped through the panel's scale.
    Data(f64),
}

impl CoordValue {
    fn category(&self) -> Option<&str> {
        match self {
            CoordValue::Category { category } => Some(category),
            _ => None,
        }
    }
}

/// An annotation coordinate the renderer refuses rather than misplaces.
/// `Display` is the message the caller sees; `axis` is always the caller's
/// axis letter.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationRefusal {
    /// A category on an axis whose scale holds no categories.
    CategoryOnContinuousAxis { label: String, axis: AlignAxis },
    /// A category under a projected coord (`CoordPolar`, `CoordGeo`).
    CategoryUnderCoord { label: String, axis: AlignAxis, coord: &'static str },
    /// A span with one category end and one non-category end, on a categorical
    /// axis. `category_end` is the end that holds `label`.
    MixedSpanEnds { axis: AlignAxis, category_end: SpanEnd, label: String },
    /// A colour string the parser rejects and that is not a paint sentinel.
    /// Boxed so this one wide payload does not size every `RenderError`.
    InvalidColor(Box<InvalidColor>),
}

/// The payload of [`AnnotationRefusal::InvalidColor`]: `kind` and `field` name
/// the spec's `type` tag and the field that holds `value`; `reason` is the
/// parser's own message.
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidColor {
    pub kind: &'static str,
    pub field: &'static str,
    pub value: String,
    pub reason: String,
}

/// One end of a span, named as the caller's `start`/`end` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanEnd {
    Start,
    End,
}

impl SpanEnd {
    fn name(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::End => "end",
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Start => Self::End,
            Self::End => Self::Start,
        }
    }
}

impl std::fmt::Display for AnnotationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CategoryOnContinuousAxis { label, axis } => write!(
                f,
                "annotation category '{label}' on the {axis} axis: the {axis} scale is continuous, \
                 so it holds no categories; pass a number, a date, an ISO-8601 string, \
                 fm.NormCoord or fm.PixelCoord"
            ),
            Self::CategoryUnderCoord { label, axis, coord } => write!(
                f,
                "annotation category '{label}' on the {axis} axis: a category cannot be placed \
                 under {coord}; pass fm.NormCoord or fm.PixelCoord"
            ),
            Self::MixedSpanEnds { axis, category_end, label } => write!(
                f,
                "annotation span on the {axis} axis: {} '{label}' is a category and {} is not; \
                 pass categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends",
                category_end.name(),
                category_end.other().name()
            ),
            Self::InvalidColor(c) => write!(
                f,
                "annotation {}: {}='{}' is not a valid color ({})",
                c.kind, c.field, c.value, c.reason
            ),
        }
    }
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
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        color: Option<String>,
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
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        stroke: Option<String>,
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
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        fill: Option<String>,
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
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        stroke: Option<String>,
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
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        fill: Option<String>,
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
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        stroke: Option<String>,
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
        /// `None` draws the panel theme's background colour.
        #[serde(default)]
        background: Option<String>,
        /// `None` draws the panel theme's text colour ([`ink`]).
        #[serde(default)]
        border_color: Option<String>,
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

impl AnnotationSpec {
    /// This spec with every coordinate pair and a span's `axis` exchanged
    /// (`x`/`y`, `x1`/`y1`, `x2`/`y2`, `text_x`/`text_y`): the spec the caller
    /// wrote under a `CoordFlip`, read against the drawn axes. Offsets (`dx`,
    /// `dy`), anchors and label positions are drawn-space and stay where they
    /// are. A bracket's `direction` is geometry along its own baseline, so it
    /// is transposed with the pair: up and above to right, down and below to
    /// left, left to down, right to up (an unrecognized value draws as up, so it
    /// becomes right).
    fn exchanged(&self) -> Self {
        use std::mem::swap;
        let mut spec = self.clone();
        match &mut spec {
            AnnotationSpec::Text { x, y, .. }
            | AnnotationSpec::Callout { x, y, .. }
            | AnnotationSpec::Image { x, y, .. } => swap(x, y),
            AnnotationSpec::Arrow { x, y, x2, y2, .. } => {
                swap(x, y);
                swap(x2, y2);
            }
            AnnotationSpec::Rect { x1, y1, x2, y2, .. }
            | AnnotationSpec::Line { x1, y1, x2, y2, .. } => {
                swap(x1, y1);
                swap(x2, y2);
            }
            AnnotationSpec::Bracket { x1, y1, x2, y2, direction, .. } => {
                swap(x1, y1);
                swap(x2, y2);
                *direction = match direction.as_str() {
                    "down" | "below" => "left",
                    "left" => "down",
                    "right" => "up",
                    _ => "right",
                }
                .to_string();
            }
            AnnotationSpec::Span { axis, .. } => {
                *axis = if axis == "x" { "y" } else { "x" }.to_string();
            }
        }
        if let AnnotationSpec::Callout { text_x, text_y, .. } = &mut spec {
            swap(text_x, text_y);
        }
        spec
    }
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
    DEFAULT_ARROW_HEAD_SIZE
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
    /// The panel's effective theme: an annotation given no colour draws the
    /// theme's text colour ([`ink`]), so it follows `.theme()` and
    /// `set_default_theme` like the text mark does.
    pub theme: &'a ThemeInputs,
}

impl<'a> ScaleContext<'a> {
    /// Resolve an x-axis coordinate value to absolute pixel position.
    pub fn resolve_x(&self, v: &CoordValue) -> f64 {
        match v {
            CoordValue::Data(d) => {
                let base = data_pixel_x(self.x_scale, &self.plot_area, *d);
                self.fold_breaks(AlignAxis::X, base)
            }
            CoordValue::Pixel { px } => self.plot_area.x + px,
            CoordValue::Norm { norm } => self.plot_area.x + norm * self.plot_area.w,
            CoordValue::Category { category } => self
                .category_centre(self.x_scale, category)
                .unwrap_or(self.plot_area.x + self.plot_area.w / 2.0),
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
                self.fold_breaks(AlignAxis::Y, base)
            }
            CoordValue::Pixel { px } => self.plot_area.y + px,
            CoordValue::Norm { norm } => self.plot_area.y + norm * self.plot_area.h,
            CoordValue::Category { category } => self
                .category_centre(self.y_scale, category)
                .unwrap_or(self.plot_area.y + self.plot_area.h / 2.0),
        }
    }

    /// The `(start, end)` pixels of a span along `axis`.
    ///
    /// Two category ends the axis holds cover whole bands: from the outer edge
    /// of the lower-placed band to the outer edge of the higher-placed one,
    /// whichever order they were written in, so `start == end` shades that
    /// band. Every other pair resolves end by end, which places a category
    /// the axis lacks at the plot centre.
    fn resolve_span(&self, axis: AlignAxis, start: &CoordValue, end: &CoordValue) -> (f64, f64) {
        let scale = if axis.is_y() { self.y_scale } else { self.x_scale };
        let whole_bands = start.category().zip(end.category()).and_then(|(a, b)| {
            let (a, b) = (self.category_centre(scale, a)?, self.category_centre(scale, b)?);
            let half = scale.bandwidth()? / 2.0;
            Some((a.min(b) - half, a.max(b) + half))
        });
        whole_bands.unwrap_or_else(|| {
            if axis.is_y() {
                (self.resolve_y(start), self.resolve_y(end))
            } else {
                (self.resolve_x(start), self.resolve_x(end))
            }
        })
    }

    /// Where `scale` places the category `label`, or `None` when the scale is
    /// not categorical or does not hold the label.
    fn category_centre(&self, scale: &ScaleKind, label: &str) -> Option<f64> {
        scale.to_pixel_str(label).filter(|px| px.is_finite())
    }

    /// Fold `base` (a pixel already resolved against the UNBROKEN `x_scale`/
    /// `y_scale`) through every `axis`-matching break, in declaration order.
    ///
    /// This is exactly `remap_mark_batch_through_break`'s per-coordinate
    /// fold — reused here rather than re-derived, so an annotation and a
    /// mark reach the same pixel through the same arithmetic. In practice
    /// the fold runs 0 or 1 times per axis: `merge_break_specs`'s
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
    fn fold_breaks(&self, axis: AlignAxis, base: f64) -> f64 {
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
/// `axis`, in the shape the break's suppression rule must test them — see
/// [`AxisCoords`].
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
fn data_coords_on_axis(spec: &AnnotationSpec, axis: AlignAxis) -> AxisCoords {
    // Reads the axis through `AlignAxis::is_y`, the same single exhaustive
    // match `BreakRemap::is_y` and the indicator glyphs go through (`#229`(2)):
    // one place in the crate decides that the other axis is x, so a future
    // third axis value cannot be suppressed as one axis and remapped as the
    // other.
    let is_y = axis.is_y();
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
        // y-break, and vice versa. That field is a DIFFERENT wire field with
        // its own accepted keys (`#229`(2) leaves it a `String` deliberately),
        // so the two vocabularies meet through [`AlignAxis::wire_name`] rather
        // than by retyping the span.
        AnnotationSpec::Span { axis: span_axis, start, end, .. } => {
            if span_axis == axis.wire_name() {
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

/// Whether `s` is one of the paint words the Python side accepts in place of a
/// colour (trimmed, case-insensitive, as `draw::resolve_paint_color` reads them).
fn is_paint_word(s: &str) -> bool {
    let word = s.trim();
    word.eq_ignore_ascii_case("none") || word.eq_ignore_ascii_case("transparent")
}

/// The scene colour of `value`, the string held by `field` of a `kind`
/// annotation: a parsed colour, or `None` (no paint) for a paint word. A
/// string that is neither is refused.
fn resolve_color(
    kind: &'static str,
    field: &'static str,
    value: &str,
) -> Result<Option<Color>, AnnotationRefusal> {
    if is_paint_word(value) {
        return Ok(None);
    }
    parse_color_str(value).map(|c| Some(to_scene_color(c))).map_err(|e| {
        AnnotationRefusal::InvalidColor(Box::new(InvalidColor {
            kind,
            field,
            value: value.to_string(),
            reason: e.to_string(),
        }))
    })
}

/// Every colour string `spec` carries, as `(field, value)`; the annotation kind
/// is [`kind_name`]`(spec)`.
///
/// Exhaustive over `AnnotationSpec`, so a new variant is a compile error here
/// rather than a colour that skips the check.
fn colour_fields(spec: &AnnotationSpec) -> Vec<(&'static str, &str)> {
    match spec {
        AnnotationSpec::Text { color, .. } => color.iter().map(|c| ("color", c.as_str())).collect(),
        AnnotationSpec::Arrow { stroke, .. }
        | AnnotationSpec::Line { stroke, .. }
        | AnnotationSpec::Bracket { stroke, .. } => stroke.iter().map(|c| ("stroke", c.as_str())).collect(),
        AnnotationSpec::Rect { fill, stroke, .. } => fill
            .iter()
            .map(|c| ("fill", c.as_str()))
            .chain(stroke.iter().map(|c| ("stroke", c.as_str())))
            .collect(),
        AnnotationSpec::Span { fill, .. } => fill.iter().map(|c| ("fill", c.as_str())).collect(),
        AnnotationSpec::Callout { background, border_color, .. } => background
            .iter()
            .map(|c| ("background", c.as_str()))
            .chain(border_color.iter().map(|c| ("border_color", c.as_str())))
            .collect(),
        AnnotationSpec::Image { .. } => Vec::new(),
    }
}

/// Refuse the first colour string of `specs` that is neither a colour nor a
/// paint word, whatever a break later does with its annotation, so no
/// annotation draws a colour the caller did not write.
fn check_colors(specs: &[AnnotationSpec]) -> Result<(), AnnotationRefusal> {
    for spec in specs {
        let kind = kind_name(spec);
        for (field, value) in colour_fields(spec) {
            resolve_color(kind, field, value)?;
        }
    }
    Ok(())
}

/// The colour a `kind` annotation primitive draws for `field`: the caller's
/// when given (`None` for a paint word), else the theme's text colour (the text
/// mark's rule), so unset ink reads on every theme.
fn ink(
    kind: &'static str,
    field: &'static str,
    color: Option<&str>,
    ctx: &ScaleContext,
) -> Result<Option<Color>, AnnotationRefusal> {
    color.map_or_else(|| Ok(Some(theme_ink(ctx))), |c| resolve_color(kind, field, c))
}

/// The theme's text colour, for the labels an annotation draws with no colour
/// parameter of their own.
fn theme_ink(ctx: &ScaleContext) -> Color {
    to_scene_color(ctx.theme.colors.font_color)
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
#[derive(Debug)]
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
fn build_annotations(
    specs: &[AnnotationSpec],
    ctx: &ScaleContext,
) -> Result<AnnotationNodes, AnnotationRefusal> {
    let mut result = AnnotationNodes {
        below_marks: Vec::new(),
        above_marks: Vec::with_capacity(specs.len()),
    };
    for spec in specs {
        build_one(spec, ctx, z_bucket(spec, &mut result))?;
    }
    Ok(result)
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
fn build_one(
    spec: &AnnotationSpec,
    ctx: &ScaleContext,
    out: &mut Vec<SceneNode>,
) -> Result<(), AnnotationRefusal> {
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
                    color: ink("text", "color", color.as_deref(), ctx)?,
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
                ink("arrow", "stroke", stroke.as_deref(), ctx)?,
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
            let fill_color = ink("rect", "fill", fill.as_deref(), ctx)?;
            let stroke_color = stroke
                .as_deref()
                .map(|c| resolve_color("rect", "stroke", c))
                .transpose()?
                .flatten();
            out.push(SceneNode::Rect {
                x,
                y,
                w,
                h,
                style: FillStroke {
                    fill: fill_color,
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
                    color: ink("line", "stroke", stroke.as_deref(), ctx)?,
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
                fill.as_deref(),
                *opacity,
                label.as_deref(),
                label_position.as_str(),
                ctx,
                out,
            )?;
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
                ink("bracket", "stroke", stroke.as_deref(), ctx)?,
                theme_ink(ctx),
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
                background.as_deref(),
                border_color.as_deref(),
                *border_radius,
                text_x.as_ref(),
                text_y.as_ref(),
                out,
            )?;
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
    Ok(())
}

// ── Emitter helpers (extracted to stay under clippy's argument limit) ────────

#[allow(clippy::too_many_arguments)]
fn emit_arrow(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    color: Option<Color>,
    stroke_width: f64,
    head_size: f64,
    out: &mut Vec<SceneNode>,
) {
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
    // direction of travel. Geometry lives in `marks::arrowhead::arrowhead_triangle`
    // — shared with `marks::segment::build`'s `arrow=true` path (spec §4 H1) —
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
    fill: Option<&str>,
    opacity: f64,
    label: Option<&str>,
    label_position: &str,
    ctx: &ScaleContext,
    out: &mut Vec<SceneNode>,
) -> Result<(), AnnotationRefusal> {
    let (x, y, w, h) = if axis == "x" {
        let (x_start, x_end) = ctx.resolve_span(AlignAxis::X, start, end);
        // Guard NaN: either coordinate being NaN produces NaN width — skip emission.
        if !x_start.is_finite() || !x_end.is_finite() {
            return Ok(());
        }
        let x = x_start.min(x_end);
        let w = (x_end - x_start).abs();
        (x, ctx.plot_area.y, w, ctx.plot_area.h)
    } else {
        let (y_start, y_end) = ctx.resolve_span(AlignAxis::Y, start, end);
        // Guard NaN: either coordinate being NaN produces NaN height — skip emission.
        if !y_start.is_finite() || !y_end.is_finite() {
            return Ok(());
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
            fill: ink("span", "fill", fill, ctx)?,
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

    if let Some(node) = span_label_node(x, y, w, h, label, label_position, theme_ink(ctx)) {
        out.push(node);
    }
    Ok(())
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
    color: Color,
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
            color: Some(color),
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
    color: Option<Color>,
    label_color: Color,
    tip_length: f64,
    out: &mut Vec<SceneNode>,
) {
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
        "down" | "below" => (0.0, tip_length),
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
            "down" | "below" => (0.0, label_offset),
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
                color: Some(label_color),
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
    background: Option<&str>,
    border_color: Option<&str>,
    border_radius: f64,
    text_x: Option<&CoordValue>,
    text_y: Option<&CoordValue>,
    out: &mut Vec<SceneNode>,
) -> Result<(), AnnotationRefusal> {
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

    let box_fill = background.map_or_else(
        || Ok(Some(to_scene_color(ctx.theme.colors.background_color))),
        |c| resolve_color("callout", "background", c),
    )?;
    let border = ink("callout", "border_color", border_color, ctx)?;

    // Background rect.
    out.push(SceneNode::Rect {
        x: tx - text_w * 0.5,
        y: ty - text_h * 0.5,
        w: text_w,
        h: text_h,
        style: FillStroke {
            fill: box_fill,
            stroke: border,
            stroke_width: if border.is_some() { 1.0 } else { 0.0 },
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
                color: border,
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
            color: Some(theme_ink(ctx)),
            opacity: 1.0,
            font_family: "Inter, system-ui, sans-serif".to_string(),
        },
    });
    Ok(())
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
///
/// A category coordinate is checked first ([`check_categories`]): it refuses
/// where the panel cannot place it and warns once per label the axis lacks.
/// `coord` is the panel's coord: a projected one (`CoordPolar`/`CoordGeo`, per
/// [`suppresses_cartesian_axes`]) holds no category.
///
/// `names_unflipped` says the caller's own `CoordFlip` put the caller's x on the
/// drawn vertical axis (`prepare::names_unflipped`). The specs then hold the
/// caller's channels, so each coordinate pair is exchanged once here
/// ([`AnnotationSpec::exchanged`]) and every builder reads them as drawn
/// coordinates; refusals and warnings map the drawn axis back to the caller's
/// letter. Offsets, anchors and label positions are drawn-space and
/// are not exchanged.
pub fn build_annotations_through_breaks(
    specs: &[AnnotationSpec],
    ctx: &ScaleContext<'_>,
    coord: Option<&CoordKind>,
    names_unflipped: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<AnnotationNodes, AnnotationRefusal> {
    let exchanged: Vec<AnnotationSpec>;
    let specs = if names_unflipped {
        exchanged = specs.iter().map(AnnotationSpec::exchanged).collect();
        &exchanged[..]
    } else {
        specs
    };
    check_categories(specs, ctx, coord, names_unflipped, warnings)?;
    check_colors(specs)?;
    if ctx.breaks.is_empty() {
        return build_annotations(specs, ctx);
    }
    // Per-axis rollup accumulators (#240-3): small `Vec`s, not `HashMap`s, so
    // the emitted warnings stay in first-seen-axis order — deterministic for a
    // per-render count never large enough for O(n^2) lookup to matter.
    let mut in_gap: Vec<(AlignAxis, Vec<String>)> = Vec::new();
    let mut spanning: Vec<(AlignAxis, Vec<String>)> = Vec::new();
    let mut nodes = AnnotationNodes {
        below_marks: Vec::new(),
        above_marks: Vec::with_capacity(specs.len()),
    };
    for spec in specs {
        match break_fate(spec, ctx.breaks) {
            BreakFate::Untouched => build_one(spec, ctx, z_bucket(spec, &mut nodes))?,
            BreakFate::Cut { axes, routing } => {
                build_cut(spec, ctx, &axes, routing, z_bucket(spec, &mut nodes))?
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
        let axis = callers_axis(names_unflipped, axis);
        warnings.push(RenderWarning::AnnotationSuppressedByBreak { axis, kinds });
    }
    for (axis, kinds) in spanning {
        let axis = callers_axis(names_unflipped, axis);
        warnings.push(RenderWarning::AnnotationSpansBreakGap { axis, kinds });
    }
    Ok(nodes)
}

/// The caller's letter for a drawn `axis`: the other letter when the caller's
/// own `CoordFlip` exchanged the specs' coordinates, else `axis` itself.
fn callers_axis(exchanged: bool, axis: AlignAxis) -> AlignAxis {
    match (exchanged, axis.is_y()) {
        (false, _) => axis,
        (true, true) => AlignAxis::X,
        (true, false) => AlignAxis::Y,
    }
}

/// Every coordinate `spec` carries, with the axis it is measured on.
///
/// Exhaustive over `AnnotationSpec`'s 8 shapes, so a new variant is a compile
/// error here rather than a coordinate that skips the category check.
fn coords_with_axes(spec: &AnnotationSpec) -> Vec<(AlignAxis, &CoordValue)> {
    use AlignAxis::{X, Y};
    match spec {
        AnnotationSpec::Text { x, y, .. } | AnnotationSpec::Image { x, y, .. } => {
            vec![(X, x), (Y, y)]
        }
        AnnotationSpec::Arrow { x, y, x2, y2, .. } => vec![(X, x), (Y, y), (X, x2), (Y, y2)],
        AnnotationSpec::Rect { x1, y1, x2, y2, .. }
        | AnnotationSpec::Line { x1, y1, x2, y2, .. }
        | AnnotationSpec::Bracket { x1, y1, x2, y2, .. } => {
            vec![(X, x1), (Y, y1), (X, x2), (Y, y2)]
        }
        AnnotationSpec::Span { axis, start, end, .. } => {
            let axis = if axis == "x" { X } else { Y };
            vec![(axis, start), (axis, end)]
        }
        AnnotationSpec::Callout { x, y, text_x, text_y, .. } => {
            let mut v = vec![(X, x), (Y, y)];
            v.extend(text_x.as_ref().map(|t| (X, t)));
            v.extend(text_y.as_ref().map(|t| (Y, t)));
            v
        }
    }
}

/// Refuse every category coordinate the panel cannot place, and warn once per
/// label a categorical axis does not hold (it is then drawn at the plot
/// centre). The order is the coord, then the axis kind, then the span's shape,
/// so a span with one category end is refused for its shape only on a
/// categorical axis, where both ends as categories is a remedy that renders.
///
/// `specs` are already in drawn space; `exchanged` says they were exchanged from
/// the caller's, so a message names the other letter than the axis it measured.
fn check_categories(
    specs: &[AnnotationSpec],
    ctx: &ScaleContext<'_>,
    coord: Option<&CoordKind>,
    exchanged: bool,
    warnings: &mut Vec<RenderWarning>,
) -> Result<(), AnnotationRefusal> {
    let callers = |axis: AlignAxis| callers_axis(exchanged, axis);
    let mut unknown: Vec<(AlignAxis, &str)> = Vec::new();
    for spec in specs {
        for (axis, value) in coords_with_axes(spec) {
            let Some(label) = value.category() else { continue };
            if let Some(coord) = coord.filter(|c| suppresses_cartesian_axes(Some(c))) {
                return Err(AnnotationRefusal::CategoryUnderCoord {
                    label: label.to_string(),
                    axis: callers(axis),
                    coord: public_spelling(coord).trim_start_matches("fm."),
                });
            }
            let scale = if axis.is_y() { ctx.y_scale } else { ctx.x_scale };
            if !matches!(scale, ScaleKind::Ordinal(_)) {
                return Err(AnnotationRefusal::CategoryOnContinuousAxis {
                    label: label.to_string(),
                    axis: callers(axis),
                });
            }
            if ctx.category_centre(scale, label).is_none() && !unknown.contains(&(axis, label)) {
                unknown.push((axis, label));
            }
        }
        if let AnnotationSpec::Span { axis, start, end, .. } = spec {
            let (category_end, label) = match (start.category(), end.category()) {
                (Some(label), None) => (SpanEnd::Start, label),
                (None, Some(label)) => (SpanEnd::End, label),
                _ => continue,
            };
            return Err(AnnotationRefusal::MixedSpanEnds {
                axis: callers(if axis == "x" { AlignAxis::X } else { AlignAxis::Y }),
                category_end,
                label: label.to_string(),
            });
        }
    }
    for (axis, label) in unknown {
        let scale = if axis.is_y() { ctx.y_scale } else { ctx.x_scale };
        let mut domain = match scale {
            ScaleKind::Ordinal(ordinal) => ordinal.ticks_internal(),
            _ => Vec::new(),
        };
        domain.sort();
        warnings.push(RenderWarning::AnnotationCategoryNotFound {
            axis: callers(axis),
            label: label.to_string(),
            domain,
        });
    }
    Ok(())
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
) -> Result<(), AnnotationRefusal> {
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
        theme: ctx.theme,
    };

    let mut nodes: Vec<SceneNode> = Vec::new();
    // A span's label is held back and re-anchored below, on the piece that
    // survives largest; every other kind's nodes all go through the machinery.
    let band_only = span_band_only(spec);
    build_one(band_only.as_ref().unwrap_or(spec), &unrouted_ctx, &mut nodes)?;
    for &i in axes {
        nodes = routing.place(&ctx.breaks[i], nodes);
    }
    if let Some(label) = clipped_span_label(spec, &nodes, theme_ink(ctx)) {
        nodes.push(label);
    }
    out.append(&mut nodes);
    Ok(())
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
///
/// Named after [`BreakPlacement`], which asks the same question for mark
/// batches: `ClipExtent`, `MoveVertices`, `MoveVerticesAndGlyph` are the same
/// three answers under the same names. The two glyph placements have
/// annotation counterparts — `Text`/`Callout`/`Image` are point-anchored
/// glyphs too — but those route to [`BreakRouting::SuppressWhole`] instead of
/// moving, for the reason documented on that variant. The two enums are not
/// linked by the type system — keep them in correspondence by hand if either
/// grows a new variant.
///
/// [`BreakPlacement`]: super::break_axis::BreakPlacement
#[derive(Clone, Copy)]
enum CutRouting {
    /// `rect`'s rect and `span`'s band: the pixel extent along the broken axis
    /// IS data, so it clips per visible segment exactly as a bar's height does
    /// (annotation rects lower to the same `SceneNode::Rect` shape).
    ClipExtent,
    /// `line`'s shaft: connected stroked geometry whose every vertex is its own
    /// data position, so it is cut at the gap edges it crosses, with the
    /// cross-axis coordinate interpolated at each crossing. The
    /// [`BreakPlacement::MoveVertices`](super::break_axis::BreakPlacement::MoveVertices)
    /// counterpart.
    MoveVertices,
    /// `arrow`: a split shaft plus the head that rides the piece reaching its
    /// tip. [`emit_arrow`] pushes `[shaft Line, head Path]` — the same node pair,
    /// in the same order, `marks::segment::build` emits for `arrow=True` — so
    /// the segment splitter's shaft/head pairing (head rebuilt on the tip piece,
    /// dropped when the tip is in a gap) applies verbatim. The
    /// [`BreakPlacement::MoveVerticesAndGlyph`](super::break_axis::BreakPlacement::MoveVerticesAndGlyph)
    /// counterpart.
    MoveVerticesAndGlyph,
}

impl CutRouting {
    /// Re-place `nodes` through one break.
    fn place(self, remap: &break_axis::BreakRemap<'_>, nodes: Vec<SceneNode>) -> Vec<SceneNode> {
        match self {
            Self::ClipExtent => remap.clip_nodes(nodes).nodes,
            Self::MoveVertices => remap.split_nodes(nodes).nodes,
            Self::MoveVerticesAndGlyph => remap.split_segment_nodes(nodes).nodes,
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
        AnnotationSpec::Line { .. } => BreakRouting::Cut(CutRouting::MoveVertices),
        AnnotationSpec::Arrow { .. } => BreakRouting::Cut(CutRouting::MoveVerticesAndGlyph),
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
    Point { axis: AlignAxis, kind: String },
    /// No coordinate does, but the annotation's interval reaches across a gap.
    Span { axis: AlignAxis, kind: String },
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
        // `axis` is `Copy` now, so the conflict takes it directly — where the
        // `String` field it replaces had to be cloned per dropped annotation.
        let kind = kind_name(spec).to_string();
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
fn accumulate_by_axis(acc: &mut Vec<(AlignAxis, Vec<String>)>, axis: AlignAxis, kind: String) {
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
fn clipped_span_label(
    spec: &AnnotationSpec,
    pieces: &[SceneNode],
    color: Color,
) -> Option<SceneNode> {
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
    span_label_node(x, y, w, h, Some(label), label_position, color)
}

/// `(x, y, w, h)` of a `Rect` node — the shape a clipped span piece has.
fn rect_extent(node: &SceneNode) -> Option<(f64, f64, f64, f64)> {
    match node {
        SceneNode::Rect { x, y, w, h, .. } => Some((*x, *y, *w, *h)),
        _ => None,
    }
}

// Test coverage lives in `tests.rs` (precedent: `render/scale_resolve/tests.rs`,
// `projection/mod.rs`, `transform/stats/mod.rs`) — the module grew too large for
// an inline `mod tests { ... }` block: at 3554 lines the inline test body
// (`mod tests` starting at line 1534) exceeded the source module above it,
// crossing the promotion threshold in this crate's Rust test-module convention.
#[cfg(test)]
mod tests;
