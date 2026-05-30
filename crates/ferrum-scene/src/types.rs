use serde::{Deserialize, Serialize};

use crate::selection::{InteractionConfig, SelectionSpec};

// ── 3.1 Top-level ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneGraph {
    pub width: f64,
    pub height: f64,
    pub background: Option<Color>,
    pub title: Vec<SceneNode>,
    pub panels: Vec<Panel>,
    pub legend: Vec<SceneNode>,
    pub decorations: Vec<SceneNode>,
    pub selections: Vec<SelectionSpec>,
    pub interaction: InteractionConfig,
    /// Chart-level accessibility description emitted as `<desc>` in the root SVG.
    /// `None` means no `<desc>` element is emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart_description: Option<String>,
}

// ── 3.2 Panel ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Panel {
    pub id: usize,
    pub plot_area: Rect,
    pub clip: Rect,
    pub coord: CoordKind,
    pub grid: Vec<SceneNode>,
    pub marks: Vec<MarkBatch>,
    pub axes: Vec<SceneNode>,
    pub annotations: Vec<SceneNode>,
    pub strip_title: Vec<SceneNode>,
}

// ── 3.3 MarkBatch ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MarkBatch {
    pub kind: MarkBatchKind,
    pub nodes: Vec<SceneNode>,
    pub data_indices: Option<Vec<usize>>,
    pub tooltips: Option<Vec<TooltipContent>>,
    pub hrefs: Option<Vec<Option<String>>>,
    pub descriptions: Option<Vec<Option<String>>>,
    pub keys: Option<Vec<String>>,
    pub blend: BlendMode,
    pub stroke_cap: Option<StrokeCap>,
    pub stroke_join: Option<StrokeJoin>,
    /// Base64-encoded packed instance bytes for high-cardinality batches.
    /// When present, the WASM renderer uses this instead of iterating `nodes`.
    /// Format: raw `CircleInstance` or `RectInstance` bytes (Pod-cast-safe).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_instances: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MarkBatchKind {
    Point,
    Line,
    Bar,
    Area,
    Rule,
    Text,
    /// Label marks — identical rendering to `Text` but serializes as `"label"`
    /// so downstream consumers can distinguish `mark_label` from `mark_text`.
    Label,
    Tick,
    Rect,
    Polygon,
    Image,
    Ribbon,
    Segment,
    Arc,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    Normal,
    Additive,
}

// ── 3.4 SceneNode ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SceneNode {
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        style: FillStroke,
        corner_radius: f64,
    },
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        style: FillStroke,
    },
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        style: StrokeStyle,
    },
    Path {
        commands: Vec<PathCmd>,
        style: FillStroke,
        closed: bool,
    },
    Text {
        x: f64,
        y: f64,
        content: String,
        style: TextStyle,
    },
    Image {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        data: ImageData,
    },
    Polygon {
        /// Rings of the polygon: first ring is the exterior, subsequent rings
        /// are interior holes. Multiple rings are rendered with `fill-rule="evenodd"`
        /// so holes cut through the fill automatically.
        rings: Vec<Vec<[f64; 2]>>,
        style: FillStroke,
    },
    Polyline {
        points: Vec<(f64, f64)>,
        style: StrokeStyle,
    },
    Group {
        attrs: Vec<(String, String)>,
        children: Vec<SceneNode>,
    },
    Raw {
        svg: String,
    },
}

// ── 3.5 Style types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FillStroke {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub stroke_dash: Option<Vec<f64>>,
    /// Per-element stroke opacity in [0, 1]. Default 1.0 (fully opaque).
    #[serde(default = "default_stroke_opacity", skip_serializing_if = "is_one_f64")]
    pub stroke_opacity: f64,
    /// Per-element fill opacity in [0, 1]. Default 1.0 (fully opaque).
    /// Emitted as SVG `fill-opacity` — distinct from `opacity` which bakes
    /// into the fill RGBA alpha. Both can coexist independently.
    #[serde(default = "default_fill_opacity", skip_serializing_if = "is_one_f64")]
    pub fill_opacity: f64,
    /// Per-element rotation in degrees around the element's anchor point. Default 0.0.
    #[serde(default, skip_serializing_if = "is_zero_angle")]
    pub angle: f64,
}

fn is_zero_angle(v: &f64) -> bool { *v == 0.0 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StrokeStyle {
    pub color: Color,
    pub width: f64,
    pub opacity: f64,
    pub dash: Option<Vec<f64>>,
    pub stroke_cap: Option<StrokeCap>,
    pub stroke_join: Option<StrokeJoin>,
    /// Per-element stroke opacity in [0, 1]. Default 1.0.
    #[serde(default = "default_stroke_opacity", skip_serializing_if = "is_one_f64")]
    pub stroke_opacity: f64,
}

fn default_stroke_opacity() -> f64 { 1.0 }
fn default_fill_opacity() -> f64 { 1.0 }
fn is_one_f64(v: &f64) -> bool { (*v - 1.0).abs() < f64::EPSILON }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StrokeCap {
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StrokeJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStyle {
    pub font_size: f64,
    pub font_weight: FontWeight,
    pub anchor: TextAnchor,
    pub baseline: TextBaseline,
    pub angle: f64,
    pub color: Color,
    pub opacity: f64,
    pub font_family: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FontWeight {
    Normal,
    Bold,
    Custom(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TextAnchor {
    Start,
    Middle,
    End,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TextBaseline {
    Top,
    Middle,
    Bottom,
    Alphabetic,
    Custom(String),
}

// ── PathCmd ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PathCmd {
    MoveTo { x: f64, y: f64 },
    LineTo { x: f64, y: f64 },
    QuadTo { cx: f64, cy: f64, x: f64, y: f64 },
    CubicTo { c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64 },
    HLineTo { x: f64 },
    VLineTo { y: f64 },
    ArcTo { rx: f64, ry: f64, rotation: f64, large_arc: bool, sweep: bool, x: f64, y: f64 },
    Close,
}

// ── Image ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageData {
    Inline { bytes: Vec<u8>, mime: ImageMime },
    Url { url: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageMime {
    Png,
    Jpeg,
}

// ── Rect ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

// ── 3.6 Tooltip ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TooltipContent {
    pub fields: Vec<TooltipField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TooltipField {
    pub name: String,
    pub value: String,
}

// ── 3.10 CoordKind ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CoordKind {
    Cartesian {
        x_domain: Option<(f64, f64)>,
        y_domain: Option<(f64, f64)>,
        expand: bool,
        clip: bool,
    },
    Fixed {
        ratio: f64,
        x_domain: Option<(f64, f64)>,
        y_domain: Option<(f64, f64)>,
        expand: bool,
        clip: bool,
    },
    Polar {
        theta: PolarThetaChannel,
        start_angle: f64,
        direction: PolarDirection,
        inner_radius: f64,
        outer_radius: f64,
    },
    Geo {
        projection: GeoProjection,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolarThetaChannel {
    X,
    Y,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolarDirection {
    Clockwise,
    CounterClockwise,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GeoProjection {
    Mercator,
    AlbersUsa,
    EqualEarth,
    NaturalEarth,
    Orthographic,
    Equirectangular,
}
