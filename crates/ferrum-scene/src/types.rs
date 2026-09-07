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
    /// Below-marks text annotations (`AnnotationSpec::Text` with `z ==
    /// "below_marks"`), painted immediately after `grid` and before `marks`.
    ///
    /// This is a typed sibling of `grid`, not a sub-bucket of it (GH #89B):
    /// content (user annotations) and chrome (gridlines) are distinct scene
    /// concerns so a later overlay merge can clear duplicate chrome without
    /// risk of also dropping a user annotation. Empty on every scene that
    /// doesn't use below-marks text annotations — the overwhelming majority —
    /// so `#[serde(skip_serializing_if)]` keeps those scenes byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub below_marks: Vec<SceneNode>,
    pub marks: Vec<MarkBatch>,
    pub axes: Vec<SceneNode>,
    /// Above-marks axis/grid chrome (nodes from an axis or its gridlines
    /// whose effective `zindex >= 1`), painted immediately after `axes` and
    /// before `annotations`.
    ///
    /// This is a typed sibling of `annotations`, not a sub-bucket of it (GH
    /// #89B): chrome (a duplicated axis/grid on a merged overlay) and content
    /// (user annotations, structural annotations) are distinct scene
    /// concerns so a later overlay merge can clear duplicate chrome without
    /// risk of also dropping a user annotation. Painting `chrome_above`
    /// before `annotations` is a deliberate z-order refinement: above-marks
    /// user annotations now always paint above above-marks axis chrome
    /// (previously the two were interleaved within one `annotations` list in
    /// axis-chrome-first order). Empty on every scene without a `zindex >= 1`
    /// axis — the overwhelming majority — so `#[serde(skip_serializing_if)]`
    /// keeps those scenes byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chrome_above: Vec<SceneNode>,
    pub annotations: Vec<SceneNode>,
    pub strip_title: Vec<SceneNode>,
    /// Per-panel layout transform mapping this panel's natively-computed
    /// geometry into its allocated slot in a composed scene (ratio-fitted
    /// cells, e.g. JointChart/ClusterMap marginals).
    ///
    /// Default is identity (`sx = sy = 1`, `tx = ty = 0`), meaning this
    /// panel's geometry is already at its final position — the case for
    /// every flat and faceted chart today. Both `walk_svg` and the WASM
    /// scene loader apply this transform identically so ratio-fitted cells
    /// render proportionally in both renderers (the single mechanism behind
    /// the W5 fix). Absent on deserialize means identity (back-compat with
    /// scenes serialized before this field existed); identity is not
    /// serialized, matching the `#[serde(default)]` precedent used elsewhere
    /// in this schema (see `chart_description` above, `stroke_opacity` /
    /// `fill_opacity` below).
    #[serde(default, skip_serializing_if = "LayoutScale::is_identity")]
    pub layout_scale: LayoutScale,
}

/// A 2D scale + translate transform: `(x, y) -> (sx * x + tx, sy * y + ty)`.
///
/// Deliberately not a full affine (no rotation/shear): neither `walk_svg`
/// nor the WASM renderer needs it, and this shape maps directly onto the
/// WASM `Uniforms`/`Affine2` upload (`sx, sy, tx, ty`) with no truncation.
/// Non-uniform (`sx != sy`) is required — JointChart marginals compute
/// independent x/y scale factors (grid ratio-fit math in
/// `render/composite_render.rs`, absorbed from the deleted `grid_compose.rs`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct LayoutScale {
    pub sx: f64,
    pub sy: f64,
    pub tx: f64,
    pub ty: f64,
}

impl LayoutScale {
    /// The no-op transform: `(x, y) -> (x, y)`.
    pub fn identity() -> Self {
        Self {
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// True when this transform is the identity — the byte-stability anchor
    /// for `skip_serializing_if` and for both walkers' "do nothing extra"
    /// fast path. Uses an epsilon comparison, matching this file's existing
    /// `is_one_f64`/`is_zero_angle` convention.
    pub fn is_identity(&self) -> bool {
        (self.sx - 1.0).abs() < f64::EPSILON
            && (self.sy - 1.0).abs() < f64::EPSILON
            && self.tx.abs() < f64::EPSILON
            && self.ty.abs() < f64::EPSILON
    }

    /// Apply this transform to a point.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.sx * x + self.tx, self.sy * y + self.ty)
    }
}

impl Default for LayoutScale {
    fn default() -> Self {
        Self::identity()
    }
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
    /// Y-scale slot this batch's marks map through (secondary-y-axis, GH #52
    /// Task 8). `0` = the primary/left-axis scale — the byte-stable default,
    /// omitted from JSON so every existing batch is untouched. A layer resolved
    /// against an `independent_y` slot carries that slot's index (`k` = k-th
    /// independent layer, matching [`crate::CoordKind::Cartesian`]'s
    /// `y_domains` list) so the interactive runtime can compose that layer's
    /// own per-slot rescale affine with the panel zoom/pan affine.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub y_slot: usize,
    /// Per-node `(angle_start, angle_end)` in radians for a polar wedge
    /// (`mark_arc` / polar `mark_bar`, `MarkBatchKind::Arc` only) — ground
    /// truth carried straight from the wedge's own construction (batch E
    /// Task 2, cycle 5), aligned by node index like `tooltips`/`hrefs`/
    /// `descriptions`. Exists because a wedge's own emitted path geometry
    /// cannot always be inverted back into its true angular span: a
    /// zero-radial-thickness wedge (`inner_r == outer_r`, e.g. a `y=0`
    /// category under `CoordPolar`) is geometrically indistinguishable from
    /// a genuine full circle by *any* predicate over its `PathCmd`s alone
    /// (both have every point on one ring), and a corner-rounded wedge's
    /// main-arc endpoints are trimmed inward from its true `angle_start`/
    /// `angle_end` by each fillet's own angular width. `None` for every
    /// non-`Arc` batch and for any `Arc` batch built before this field
    /// existed — the byte-stable default; the interactive hit-test reader
    /// treats a missing entry as "no span, skip this node" rather than
    /// guessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polar_spans: Option<Vec<(f64, f64)>>,
    /// Whether this batch is clipped to the panel's plot area (batch E Task
    /// 9, `mark_function(clip=False)`). `true` (the byte-stable default,
    /// skipped from JSON so every existing batch is untouched) means the
    /// batch renders inside the panel clip region, same as before this field
    /// existed. `false` is a per-batch escape: both walkers place the batch
    /// OUTSIDE the panel clip — `walk_svg` mirrors the existing Text/Label
    /// placement precedent (rendered after the clip closes, ahead of
    /// `chrome_above`/`annotations`). The WASM loader (`scene_load.rs`'s
    /// `load_scene_with_packed`) exempts the batch from its scissor via two
    /// different shapes depending on GPU draw path: instanced draw commands
    /// (packed circle/rect batches, and any circle/rect nodes found inside a
    /// mesh batch) get `plot_area: None` — the literal "no scissor"
    /// pass-default `render.rs`'s `ensure_scissor!` reads, with no
    /// float-scale truncation risk; the mesh path's `MarkMeshPanel::plot_area`
    /// has no `Option` to fall back to, so it substitutes the full scene
    /// extent instead, which still scales to the entire render target and is
    /// functionally equivalent to "no scissor". Either way the batch still
    /// tracks pan/zoom but is never cut at the panel boundary. Producer:
    /// `mark_function(clip=False)`'s synthesized line layer (T11, Python);
    /// no Rust producer sets `false` in this batch.
    ///
    /// **Documented static/interactive z-order divergence (quality review
    /// RF3, batch E Task 9).** `walk_svg` HOISTS every `clip == false` batch
    /// to paint after every `clip == true` batch in the same panel (pinned by
    /// `render/svg_walk.rs`'s
    /// `clip_exempt_batch_paints_after_clipped_marks_and_before_chrome_annotations_and_text`).
    /// The WASM loader does **not** perform that hoist — it only swaps the
    /// scissor shape per batch (`None` or the full-scene rect, per the two
    /// paths above), leaving each batch's relative paint position unchanged.
    ///
    /// The divergence model, stated as it actually is: `crates/ferrum-wasm/
    /// src/render.rs` draws every mesh-kind mark batch (lines/areas/paths, via
    /// `mark_mesh_panels`) BEFORE every instanced circle/rect batch (via
    /// `draw_commands`) — its render loop's step 2 versus step 4. That split
    /// is pre-existing and independent of this field, and it means WASM's
    /// paint order is not a function of `panel.marks` index order across the
    /// two categories at all. So a clip-exempt LINE over a scatter of CIRCLES
    /// — the primary intended use case, `mark_function(clip=False)` over a
    /// point layer — paints above the points statically and below them
    /// interactively **whatever layer order the user writes**. Ordering the
    /// layers differently does not change it, and neither does anything this
    /// crate can express in a scene. Within one category (mesh-over-mesh, or
    /// instanced-over-instanced) the two walkers agree except for the SVG
    /// hoist itself, which moves a clip-exempt batch later than its index.
    ///
    /// This supersedes an earlier note here that expected T11 to mask the
    /// divergence by appending `mark_function`'s synthesized clip-exempt line
    /// as its panel's last layer. T11 has landed and the expectation does not
    /// hold: the layer's position is the user's to choose, and the
    /// mesh-before-instanced split makes the position irrelevant anyway.
    ///
    /// Not fixed here because the fix is not local. `panel.marks` index order
    /// is an implicit paint-order invariant relied on independently by
    /// `hit_test.rs` (topmost-first `.rev()` scan), `spatial_index.rs`
    /// (`entries_for_panel`), and `scene_load.rs`'s `batch_instance_spans`
    /// (consumed by `conditional.rs` crossfilter dimming and `transition.rs`
    /// keyed-transition pairing, both of which must stay in lock-step with the
    /// flat circle/rect instance array's layout). Making the WASM loader honor
    /// the SVG hoist means re-deriving all four walks against one explicit
    /// paint-order list, with no existing coverage guarding a silent mismatch
    /// (wrong topmost hit, wrong crossfilter target, wrong transition
    /// pairing).
    ///
    /// That work is tracked as a batch-E close follow-up issue — "unify the
    /// four order-dependent WASM walks against one explicit paint-order list"
    /// — not as an instruction embedded in this doc. This field's own contract
    /// is only the scissor escape described above.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub clip: bool,
}

pub(crate) fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}

/// Shared `serde(default = "...")` value for a bool field whose byte-stable
/// default is `true` (first use: `MarkBatch::clip`, batch E Task 9).
pub(crate) fn default_true() -> bool {
    true
}

/// `skip_serializing_if` companion to [`default_true`]: omit the field from
/// JSON when it holds its default `true` value.
pub(crate) fn is_true(v: &bool) -> bool {
    *v
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
        /// Y-scale slot this tick label belongs to (secondary-y-axis, GH #60/#73).
        /// `None` for every non-axis-tick text node (titles, legend labels, mark
        /// labels, annotations) and for y-axis tick labels on the byte-stable
        /// single-slot default path (`0` = primary/left, `k` = the k-th
        /// independent-y layer, matching [`crate::CoordKind::Cartesian`]'s
        /// `y_domains` list and the `y_slot` attr already carried by the
        /// enclosing axis `Group` on dual-axis panels). Only emitted (`Some`)
        /// on panels with 2+ y-slots, so param-free/single-y scenes are
        /// byte-identical to before this field existed. Absent on legacy wire
        /// bytes deserializes to `None` via `serde(default)`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        slot: Option<usize>,
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
        /// Whether this fragment is fixed chrome (e.g. legend colorbar, inset) or
        /// data-anchored (tracks pan/zoom, e.g. annotation images in data space).
        ///
        /// The static SVG renderer ignores `anchor` and always emits `svg` verbatim.
        /// WASM uses `anchor` to decide whether to apply the canvas transform.
        /// Defaults to `Chrome` so scenes serialized before this field still
        /// deserialize correctly.
        #[serde(default)]
        anchor: RawAnchor,
    },
}

/// Discriminant for `SceneNode::Raw` that controls pan/zoom behavior in the
/// interactive (WASM) renderer. The static SVG renderer ignores it.
///
/// - `Chrome`: the fragment is fixed chrome (legend colorbar gradient, inset
///   overlay) — it stays at its authored position during pan/zoom.
/// - `Data`: the fragment is positioned in panel/data space (e.g. annotation
///   image anchored to a data coordinate) — it rides the canvas transform.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RawAnchor {
    #[default]
    Chrome,
    Data,
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

fn is_zero_angle(v: &f64) -> bool {
    *v == 0.0
}

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

fn default_stroke_opacity() -> f64 {
    1.0
}
fn default_fill_opacity() -> f64 {
    1.0
}
fn is_one_f64(v: &f64) -> bool {
    (*v - 1.0).abs() < f64::EPSILON
}

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
    MoveTo {
        x: f64,
        y: f64,
    },
    LineTo {
        x: f64,
        y: f64,
    },
    QuadTo {
        cx: f64,
        cy: f64,
        x: f64,
        y: f64,
    },
    CubicTo {
        c1x: f64,
        c1y: f64,
        c2x: f64,
        c2y: f64,
        x: f64,
        y: f64,
    },
    HLineTo {
        x: f64,
    },
    VLineTo {
        y: f64,
    },
    ArcTo {
        rx: f64,
        ry: f64,
        rotation: f64,
        large_arc: bool,
        sweep: bool,
        x: f64,
        y: f64,
    },
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
        /// Per-slot y-domain list (secondary-y-axis, GH #52 Task 8), index =
        /// slot (slot 0 mirrors `y_domain` — the primary/left axis). Empty on
        /// every chart with no independent-y layer — the byte-stable default,
        /// omitted from JSON so existing scenes are untouched. Populated only
        /// when the panel resolves one or more `independent_y` layers, so the
        /// interactive runtime can read each slot's own domain for per-slot
        /// rescale, zoom relabeling, and tooltip readout without recomputing
        /// scale resolution client-side (design spec §6 "Scene/WASM contract").
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        y_domains: Vec<Option<(f64, f64)>>,
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
