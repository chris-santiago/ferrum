# Phase 11 — Interactive Renderer (WASM) Design Spec

**Date:** 2026-05-13
**Phase:** 11
**Depends on:** 8a (done), 8b (done), 9 (done), 10 (done)
**Status:** design

---

## 1. Scope

Phase 11 delivers the interactive renderer, all deferred coordinate systems, all
deferred marks, and every remaining `NotImplementedError` / warn-fallback in the
codebase.  After Phase 11 there are zero `NotImplementedError`s, zero
warn-fallback deferrals, and zero features gated behind a future phase.

### 1.1 Deliverables

| Category | Items |
|---|---|
| **Crates** | `ferrum-scene` (shared SceneGraph IR), `ferrum-wasm` (WASM renderer) |
| **Scene graph** | `SceneGraph` IR extracted from `ferrum-core`; SVG/PNG backends refactored to consume it; golden-test-validated byte-identical output |
| **Interactive renderer** | wgpu (WebGPU + WebGL2 fallback), CSS text overlay, instanced draws for simple marks, lyon tessellation for curves/areas |
| **Jupyter integration** | `InteractiveChart` (anywidget subclass), bidirectional Python↔WASM state sync |
| **Selections** | `selection_point`, `selection_interval`, `selection_single`, `selection_multi`, `SelectionMark`, conditional encodings on all appearance channels |
| **Zoom/pan** | GPU matrix transforms, pre-computed multi-level ticks, anywidget recomputation for mark_function / mark_raster on zoom |
| **Tooltips** | DOM hover tooltip from `Tooltip(field)` encoding |
| **Href** | Click-through from `Href(field)` encoding in interactive mode; `<a>` wrapper in SVG mode |
| **Coordinate systems** | `CoordCartesian(xlim, ylim, expand, clip)`, `CoordFixed(ratio)`, `CoordPolar(theta, start, direction)`, `CoordGeo(projection)` |
| **Deferred marks** | `mark_arc`, `mark_label`, `mark_image`, `mark_geoshape` |
| **Output formats** | `.save("chart.html")`, `.save("chart.json")` |
| **Stat/mark gaps** | `mark_density(multiple=...)`, `mark_density(bw_adjust=)` with string rules, `mark_hex` full aggregates, `mark_swarm(dodge=)`, `mark_function` as multi-layer, `blend="additive"` |
| **Encoding gaps** | `condition` kwarg wired on all appearance channels, `legend` kwarg on Size/Shape/Opacity, Key channel for animated transitions |
| **Scale gaps** | TimeScale calendar-aware month/year ticks |
| **Animated transitions** | Key-channel-driven object constancy with GPU-side interpolation |

### 1.2 Sub-phase decomposition

| Sub-phase | Focus | Risk | Ships |
|---|---|---|---|
| **11a** | Scene graph extraction | High (touches stable render code) | `ferrum-scene` crate, refactored DrawCtx, SVG walker, byte-identical goldens |
| **11b** | WASM renderer foundation | Medium (all new code) | `ferrum-wasm` crate, wgpu init, JS glue, CSS text overlay, first chart in browser, `.save("chart.html")`, `.save("chart.json")` |
| **11c** | Selections + zoom/pan + anywidget | Medium | Selection state machine, zoom/pan, conditional encodings, `InteractiveChart`, anywidget Jupyter widget, tooltips, Href |
| **11d** | Coordinate systems + marks | Low–Medium | CoordCartesian, CoordFixed, CoordPolar, CoordGeo, mark_arc, mark_label, mark_image, mark_geoshape |
| **11e** | Stat/mark/encoding gap closure | Low | All remaining NotImplementedError and warn-fallback items, Key channel, animated transitions, TimeScale calendar ticks |

Each sub-phase is independently testable and committable.  11a must pass golden
tests before 11b begins.  11b must render a static chart in a browser before 11c
adds interactivity.  11d and 11e may run in parallel after 11b.

---

## 2. Crate structure

```
Cargo.toml (workspace root)
├── crates/ferrum-scene/     # NEW — shared renderer IR
├── crates/ferrum-core/      # EXISTING — depends on ferrum-scene
└── crates/ferrum-wasm/      # NEW — depends on ferrum-scene
```

### 2.1 ferrum-scene

**Purpose:** Define the SceneGraph IR types consumed by all rendering backends.
Produced by `ferrum-core`'s geometry pass, consumed by the SVG walker (in
`ferrum-core`) and the wgpu renderer (in `ferrum-wasm`).

**Dependencies:** `serde`, `serde_json` only.  No PyO3, no wgpu, no Arrow, no
platform-specific code.  This crate must compile to both native and
`wasm32-unknown-unknown` without conditional compilation.

**Anti-patterns to avoid:**
- No `#[cfg]` gates.  Every type is available on every target.
- No `Box<dyn Trait>` in the IR.  The SceneGraph is a concrete data structure,
  not an extensible visitor pattern.  Extension points (Phase 12) are a
  separate concern.
- No `HashMap` in serialized types where insertion order matters — use `Vec`
  of tuples or a dedicated struct.
- No `String` where an enum is appropriate.  Channel names, blend modes, anchor
  positions, coordinate kinds — all enums.

### 2.2 ferrum-core changes

`ferrum-core` gains a dependency on `ferrum-scene` and the following changes:

- `render/draw.rs`: refactored to emit `Vec<SceneNode>` instead of writing to
  `SvgBuffer` directly.  The geometry calculations are unchanged; the emit
  target changes.
- `render/svg_walk.rs` (new): walks a `SceneGraph` and emits SVG strings via
  the existing `SvgBuffer` API.
- `render/mod.rs`: `render_svg` and `render_png` now call `build_scene()` →
  `walk_svg()`.  New entry point `render_scene_json()` serializes the
  SceneGraph to JSON for the WASM path.
- `render/binding.rs`: new PyO3 function `render_interactive` exposed to
  Python.

**`render_interactive` PyO3 binding signature:**

```rust
#[pyfunction]
fn render_interactive(
    spec: &ChartSpec,
    data: PyRecordBatchReader,      // Arrow CDI stream (same as render_svg)
    theme: Option<HashMap<String, serde_json::Value>>,
    viewport: (f64, f64),
    config: Option<HashMap<String, serde_json::Value>>,
) -> PyResult<String>               // SceneGraph serialized as JSON
```

Same input contract as `render_svg` / `render_png`.  The return type is a
JSON string (the serialized `SceneGraph`), not an SVG string or PNG bytes.
Python deserializes it only for `merge_scene_graphs()` on compound views;
for single charts the JSON string passes through to the WASM renderer
unchanged.

No changes to: `spec/`, `transform/`, `scale/`, `layout/`, `transport/`.
PreparedInputs and LayoutResult are unchanged.

### 2.3 ferrum-wasm

**Purpose:** Receive a serialized SceneGraph, render it via wgpu in the browser,
handle user interactions.

**Dependencies:** `ferrum-scene`, `wgpu` (with `webgl` feature), `wasm-bindgen`,
`web-sys`, `js-sys`, `lyon` (tessellation), `geojson`, `serde`, `serde_json`.

**Target:** `wasm32-unknown-unknown`.  Built via `wasm-pack build --target web`.

**Anti-patterns to avoid:**
- No `unwrap()` on wgpu operations.  GPU initialization and resource creation
  can fail (adapter not found, context lost).  All wgpu fallible operations
  return `Result` with a typed error — see §8 Error handling.
- No `String`-typed messages across the JS boundary.  Use `serde_json::Value`
  or typed structs serialized to JSON.
- No global mutable state.  The renderer is an owned struct; interaction state
  lives inside it.

---

## 3. SceneGraph IR (ferrum-scene)

### 3.1 Top-level structure

```rust
pub struct SceneGraph {
    pub width: f64,
    pub height: f64,
    pub background: Option<Color>,
    pub panels: Vec<Panel>,
    pub decorations: Vec<SceneNode>,  // chart title, legends, colorbar
    pub selections: Vec<SelectionSpec>,
    pub interaction: InteractionConfig,
}
```

`SceneGraph` is the complete, self-contained specification of a rendered chart.
Given a SceneGraph, any backend can produce output without access to the original
data, spec, or transforms.

### 3.2 Panel

```rust
pub struct Panel {
    pub id: usize,
    pub plot_area: Rect,
    pub clip: Rect,
    pub coord: CoordKind,
    pub grid: Vec<SceneNode>,
    pub marks: Vec<MarkBatch>,
    pub axes: Vec<SceneNode>,
    pub annotations: Vec<SceneNode>,
    pub strip_title: Option<SceneNode>,
}
```

`Panel` is the grouping unit for zoom/pan — each panel has its own GPU transform
matrix.  The `coord` field records which coordinate system produced the geometry
so the WASM renderer can apply the correct inverse transform for hit-testing.

### 3.3 MarkBatch

```rust
pub struct MarkBatch {
    pub kind: MarkBatchKind,
    pub nodes: Vec<SceneNode>,
    pub data_indices: Option<Vec<usize>>,
    pub tooltips: Option<Vec<TooltipContent>>,
    pub hrefs: Option<Vec<Option<String>>>,
    pub keys: Option<Vec<String>>,
    pub blend: BlendMode,
}

pub enum MarkBatchKind {
    Point, Bar, Line, Area, Rect, Rule, Tick,
    Segment, Polygon, Ribbon, Text, Image, Arc,
}

pub enum BlendMode {
    Normal,
    Additive,
}
```

`MarkBatch` groups marks of the same type for GPU instanced drawing.  Parallel
vectors (`data_indices`, `tooltips`, `hrefs`, `keys`) align 1:1 with `nodes`.
This is intentional — parallel arrays are the natural GPU layout (struct-of-arrays)
and avoid per-node enum dispatch overhead in the renderer.

**Design note:** `tooltips`, `hrefs`, and `keys` are `Option<Vec<...>>` at the
batch level, not `Option<...>` per node.  When present, the vec length equals
`nodes.len()`.  When absent, the feature is not encoded for that batch.  This
avoids allocating per-node Options when the encoding is unused.

### 3.4 SceneNode

```rust
pub enum SceneNode {
    Rect {
        x: f64, y: f64, w: f64, h: f64,
        style: FillStroke,
        corner_radius: f64,
    },
    Circle {
        cx: f64, cy: f64, r: f64,
        style: FillStroke,
    },
    Line {
        x1: f64, y1: f64, x2: f64, y2: f64,
        style: StrokeStyle,
    },
    Path {
        commands: Vec<PathCmd>,
        style: FillStroke,
        closed: bool,
    },
    Text {
        x: f64, y: f64,
        content: String,
        style: TextStyle,
    },
    Image {
        x: f64, y: f64, w: f64, h: f64,
        data: ImageData,
    },
    Polygon {
        points: Vec<[f64; 2]>,
        style: FillStroke,
    },
}
```

**Design rules for SceneNode:**
- Every variant carries its own style.  No inherited style from parent — the
  scene graph is flat, not a CSS cascade.
- Coordinates are absolute pixels (post-scale, post-coord-transform).  The WASM
  renderer applies only the panel zoom/pan transform on top.
- `Text` nodes in `Panel.axes`, `Panel.strip_title`, and `SceneGraph.decorations`
  are rendered as CSS overlay in the WASM backend.  `Text` nodes in
  `Panel.marks` (from `mark_label`) are also CSS overlay — the WASM renderer
  never rasterizes text.

### 3.5 Style types

```rust
pub struct Color {
    pub r: u8, pub g: u8, pub b: u8, pub a: u8,
}

pub struct FillStroke {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

pub struct StrokeStyle {
    pub color: Color,
    pub width: f64,
    pub opacity: f64,
    pub dash: Option<Vec<f64>>,
}

pub struct TextStyle {
    pub font_size: f64,
    pub font_weight: FontWeight,
    pub anchor: TextAnchor,
    pub baseline: TextBaseline,
    pub angle: f64,
    pub color: Color,
    pub opacity: f64,
}

pub enum FontWeight { Normal, Bold }
pub enum TextAnchor { Start, Middle, End }
pub enum TextBaseline { Top, Middle, Bottom, Alphabetic }

pub enum PathCmd {
    MoveTo(f64, f64),
    LineTo(f64, f64),
    QuadTo { cx: f64, cy: f64, x: f64, y: f64 },
    CubicTo { c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64 },
    ArcTo { rx: f64, ry: f64, rotation: f64, large_arc: bool, sweep: bool, x: f64, y: f64 },
    Close,
}

pub enum ImageData {
    Inline { bytes: Vec<u8>, mime: ImageMime },
    Url(String),
}

pub enum ImageMime { Png, Jpeg }
```

**Design rules for style types:**
- No `String`-typed colors.  `Color` is `(r, g, b, a)`.  Conversion from
  hex/named colors happens in the geometry pass, not in the renderer.
- No `Option<f64>` for opacity.  Default is `1.0`, set explicitly.  The
  geometry pass resolves all theme defaults before emitting SceneNodes.
- `FontWeight` and `TextAnchor` are enums, not strings.  The SVG walker maps
  them to SVG attribute values; the WASM renderer maps them to CSS values.

### 3.6 TooltipContent

```rust
pub struct TooltipContent {
    pub fields: Vec<TooltipField>,
}

pub struct TooltipField {
    pub name: String,
    pub value: String,  // pre-formatted by the geometry pass
}
```

Tooltip values are pre-formatted strings.  The geometry pass applies the
user's `format=` spec (from `Tooltip(field, format=".2f")`) during
SceneGraph construction.  The WASM renderer displays them verbatim.

### 3.7 SelectionSpec

```rust
pub enum SelectionSpec {
    Point {
        name: String,
        fields: Option<Vec<String>>,
        encodings: Option<Vec<ChannelName>>,
        nearest: bool,
        toggle: EventExpr,
        on: EventExpr,
        clear: EventExpr,
        resolve: SelectionResolve,
    },
    Interval {
        name: String,
        fields: Option<Vec<String>>,
        encodings: Option<Vec<ChannelName>>,
        translate: bool,
        zoom: bool,
        mark: Option<SelectionMarkStyle>,
        resolve: SelectionResolve,
    },
}

pub enum SelectionResolve { Global, Union, Intersect }

pub enum ChannelName { X, Y, Color, Size, Shape, Opacity }

/// Brush rectangle styling
pub struct SelectionMarkStyle {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub fill_opacity: f64,
    pub stroke_opacity: f64,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

/// Event expressions — typed, not raw strings.
/// Covers the spec's documented event triggers.
pub enum EventExpr {
    Click,
    Mouseout,
    Mouseover,
    ShiftKey,
    Dblclick,
    Custom(String),  // escape hatch for advanced users
}
```

**Design note on `EventExpr`:** The spec uses string event expressions
(`"event.shiftKey"`, `"click"`, `"mouseout"`).  The typed enum prevents
typos and enables the WASM renderer to dispatch without string parsing.
`Custom(String)` is the escape hatch — it round-trips through JSON
unchanged and the JS glue layer evaluates it.

### 3.8 ConditionalEncoding

```rust
pub struct ConditionalEncoding {
    pub selection_name: String,
    pub channel: ChannelName,
    pub if_selected: EncodingValue,
    pub if_not: EncodingValue,
}

pub enum EncodingValue {
    Color(Color),
    Opacity(f64),
    Size(f64),
    StrokeWidth(f64),
    StrokeDash(Vec<f64>),
    Field(String),  // re-encode from a different field
}
```

`EncodingValue` is a closed enum, not a stringly-typed `serde_json::Value`.
Each variant corresponds to one visual property the WASM renderer can modify
in the GPU instance buffer without rebuilding the SceneGraph.

### 3.9 InteractionConfig

```rust
pub struct InteractionConfig {
    pub zoom_enabled: bool,
    pub pan_enabled: bool,
    pub conditionals: Vec<ConditionalEncoding>,
    pub linked_panels: Vec<Vec<usize>>,
    pub tick_levels: Vec<PanelTickLevels>,
}

/// Pre-computed tick sets at multiple zoom levels per panel.
pub struct PanelTickLevels {
    pub panel_id: usize,
    pub x_levels: Vec<TickLevel>,
    pub y_levels: Vec<TickLevel>,
}

pub struct TickLevel {
    pub min_zoom: f64,     // show this level when zoom >= min_zoom
    pub max_zoom: f64,     // and zoom < max_zoom
    pub ticks: Vec<Tick>,
}

pub struct Tick {
    pub value: f64,        // data-space value
    pub label: String,     // pre-formatted label
    pub pixel: f64,        // pixel position at zoom=1.0
}
```

Pre-computed tick levels eliminate the need for Python round-trips on zoom.
The geometry pass generates 3–4 tick granularities at SceneGraph build time.
The WASM renderer selects the level matching the current zoom factor.

### 3.10 CoordKind

```rust
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

pub enum PolarThetaChannel { X, Y }
pub enum PolarDirection { Clockwise, CounterClockwise }

pub enum GeoProjection {
    Mercator,
    AlbersUsa,
    EqualEarth,
    NaturalEarth,
    Orthographic,
    Equirectangular,
}
```

All coordinate parameters are enums, not strings.  `CoordKind` is stored on
each `Panel` so the WASM renderer knows which inverse transform to apply for
hit-testing (pixel → data-space coordinates for selections).

---

## 4. Scene graph extraction (sub-phase 11a)

### 4.1 Refactor strategy

The geometry calculations in `render/draw.rs` are preserved verbatim.  The
change is mechanical: each mark drawing function emits `Vec<SceneNode>` instead
of calling `SvgBuffer` methods.

**Before** (current):
```rust
fn draw_point(ctx: &DrawCtx, svg: &mut SvgBuffer) {
    let cx = ctx.scales.x.map(value);
    let cy = ctx.scales.y.map(value);
    svg.circle(cx, cy, r, &fill, stroke, opacity);
}
```

**After** (refactored):
```rust
fn build_point(ctx: &DrawCtx) -> Vec<SceneNode> {
    let cx = ctx.scales.x.map(value);
    let cy = ctx.scales.y.map(value);
    vec![SceneNode::Circle { cx, cy, r, style: FillStroke { fill, .. } }]
}
```

### 4.2 Module layout after refactor

```
render/
├── mod.rs              # render_svg, render_png, render_scene_json entry points
├── binding.rs          # PyO3 bindings (adds render_interactive)
├── scene_build.rs      # NEW: build_scene() — orchestrates geometry pass → SceneGraph
├── svg_walk.rs         # NEW: walk_svg(&SceneGraph) → String via SvgBuffer
├── svg.rs              # SvgBuffer (unchanged — still the low-level string builder)
├── config.rs           # RenderConfig (unchanged)
├── prepare.rs          # PreparedInputs (unchanged)
├── scale_resolve.rs    # ResolvedScales (unchanged)
├── position.rs         # Position adjustments (unchanged)
├── draw.rs             # REFACTORED: build_* functions return Vec<SceneNode>
├── marks/              # Per-mark build functions (refactored emit targets)
│   ├── mod.rs
│   ├── point.rs        # build_points() → Vec<SceneNode>
│   ├── line.rs         # build_lines() → Vec<SceneNode>
│   ├── bar.rs          # etc.
│   └── ...
├── color/              # (unchanged)
├── rasterize.rs        # SVG → PNG via resvg (unchanged)
├── png.rs              # (unchanged)
├── compositor.rs       # (unchanged)
└── grid_compose.rs     # (unchanged)
```

**Key design constraint:** `scene_build.rs` is the orchestrator.  `draw.rs` and
`marks/*.rs` are the leaf implementations.  `scene_build.rs` calls
`prepare_render_inputs()` → `compute_layout()` → iterates panels → calls
`build_*` functions → assembles `SceneGraph`.  No mark-building function calls
another mark-building function or accesses the layout directly — it receives
a `DrawCtx` and returns nodes.

`svg_walk.rs` is a pure consumer of `SceneGraph`.  It has no knowledge of
`DrawCtx`, `PreparedInputs`, or `LayoutResult`.  Its only dependency is
`ferrum-scene` types and `SvgBuffer`.

### 4.3 Validation

After the refactor, the full test suite must pass:
- `cargo test` — all existing Rust tests
- `uv run pytest` — all existing Python tests
- SVG goldens must be byte-identical.  Any byte difference is a regression.

The golden test infrastructure (`tests/_snapshots.py`) is the primary safety
net.  If a golden differs, rasterize both (old and new) to PNG and visually
compare before investigating.

---

## 5. WASM renderer (sub-phase 11b)

### 5.1 GPU rendering architecture

Three draw strategies matched to mark type:

| SceneNode variant | GPU strategy | Rationale |
|---|---|---|
| Circle | Instanced quad draw, SDF circle in fragment shader | One draw call for N points. SDF gives antialiased edges at any zoom level. |
| Rect | Instanced quad draw, SDF rounded rect in fragment shader | Same as Circle but rectangular. Corner radius via SDF. |
| Line, Path, Polygon, Arc | CPU tessellation via `lyon` → triangle mesh → single draw call | wgpu has no line-width primitive. Tessellation is pure Rust, compiles to WASM. |
| Text | CSS overlay (`<div>` elements) | Spec §3.17: "real DOM text; accessible, no font bundling required." |
| Image | Textured quad | Upload image data as GPU texture, draw positioned quad. |

### 5.2 Three-layer rendering stack

```
┌──────────────────────────────────────────┐
│  CSS Layer  (z-index: 2)                 │  Text: axis labels, tick labels,
│  HTML <div> elements, position: absolute │  titles, mark_label, tooltips
├──────────────────────────────────────────┤
│  Canvas Layer  (z-index: 1)              │  All geometric primitives
│  <canvas>, wgpu renders here             │  rendered by GPU
├──────────────────────────────────────────┤
│  Background Layer  (z-index: 0)          │  Chart/panel backgrounds,
│  CSS background or wgpu clear color      │  gridlines
└──────────────────────────────────────────┘
```

Text positioning requires coordinate mapping between the GPU canvas and CSS
`position: absolute`.  Zoom/pan must update both: canvas content via GPU matrix
transform, CSS text positions via JS `style.transform` updates.

### 5.3 JS ESM glue module

`src/ferrum/_wasm/ferrum-interactive.js` — the bridge between browser DOM
and WASM renderer.

**Responsibilities:**
1. DOM setup: create `<canvas>`, CSS text overlay container, resize observer
2. WASM initialization: load `.wasm`, initialize wgpu with WebGL2 fallback
3. Text overlay management: position `<div>`s for all `SceneNode::Text` nodes,
   update on zoom/pan
4. Event capture: mousedown/mousemove/mouseup → selection/pan;
   wheel → zoom; click → point selection / href navigation
5. Tooltip rendering: show/hide positioned `<div>` on hover with TooltipContent
6. anywidget bridge: expose `model.get()`/`model.set()` when running in Jupyter

**Two-mode detection:**
```javascript
export function render({ model, el }) {
    // anywidget mode (Jupyter) — model is available
    // standalone mode — called from inline <script> with no model
}
```

The same JS module serves both modes.  `model` presence determines whether
Python-side recomputation is available.

### 5.4 WasmRenderer struct

```rust
pub struct WasmRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    pipelines: RenderPipelines,
    scene: Option<LoadedScene>,
    interaction: InteractionState,
}

struct RenderPipelines {
    instanced_circle: wgpu::RenderPipeline,
    instanced_rect: wgpu::RenderPipeline,
    mesh: wgpu::RenderPipeline,
    textured: wgpu::RenderPipeline,
}

struct LoadedScene {
    graph: SceneGraph,
    gpu_buffers: GpuBuffers,
    text_elements: Vec<TextElement>,  // tracked for CSS positioning
}

struct GpuBuffers {
    circle_instances: wgpu::Buffer,
    rect_instances: wgpu::Buffer,
    mesh_vertices: wgpu::Buffer,
    mesh_indices: wgpu::Buffer,
    textures: Vec<wgpu::Texture>,
}
```

**No global state.**  `WasmRenderer` owns all GPU resources.  Multiple charts
on the same page each have their own renderer instance.

### 5.5 HTML output

**Standalone HTML** (`.save("chart.html")`):

```html
<!DOCTYPE html>
<html>
<head><style>/* layout + text overlay */</style></head>
<body>
  <div id="ferrum-root" style="position:relative">
    <canvas id="ferrum-canvas"></canvas>
    <div id="ferrum-overlay"></div>
  </div>
  <script type="module">
    import init, { WasmRenderer } from './ferrum_wasm.js';
    // WASM binary is base64-inlined for single-file distribution
    await init(/* inline base64 or data URI */);
    const renderer = new WasmRenderer('ferrum-canvas', 'ferrum-overlay');
    renderer.loadScene(SCENE_JSON);
    renderer.setupInteractions(INTERACTION_JSON);
  </script>
</body>
</html>
```

By default the `.wasm` binary is base64-encoded inline for single-file
distribution (adds ~33% size overhead).  `chart.save("chart.html",
embed_wasm=False)` produces a two-file output (`chart.html` + adjacent
`ferrum_wasm_bg.wasm` sidecar) for users saving many charts to the same
directory.

**JSON output** (`.save("chart.json")`): serializes the SceneGraph JSON
directly.  This is the internal IR, not a Vega-Lite spec — document that in
the docstring.

### 5.6 Build toolchain

```
Build sequence (dev):
  1. wasm-pack build crates/ferrum-wasm --target web \
       --out-dir ../../src/ferrum/_wasm/
  2. unset CONDA_PREFIX && uv run --no-sync maturin develop

Build sequence (CI / release):
  1. rustup target add wasm32-unknown-unknown
  2. cargo install wasm-pack (if not cached)
  3. wasm-pack build crates/ferrum-wasm --target web --release \
       --out-dir ../../src/ferrum/_wasm/
  4. maturin build --release
```

WASM artifacts (`ferrum_wasm_bg.wasm`, `ferrum_wasm.js`) are included in the
Python wheel as package data via `[tool.maturin] include`.

**Build commands table update** for CLAUDE.md:

| Action | Command |
|---|---|
| Build WASM module | `wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/` |
| Build WASM + Rust extension (dev) | `wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/ && unset CONDA_PREFIX && uv run --no-sync maturin develop` |

---

## 6. Interaction system (sub-phase 11c)

### 6.1 Selection state machine

```rust
pub struct InteractionState {
    selections: HashMap<String, SelectionState>,
    panel_transforms: Vec<Affine2>,
    hover: Option<HoverState>,
}

pub enum SelectionState {
    Empty,
    Point {
        indices: Vec<usize>,
        field_values: Vec<(String, FieldValue)>,
    },
    Interval {
        x_range: Option<(f64, f64)>,  // data-space, not pixel-space
        y_range: Option<(f64, f64)>,
    },
}

struct HoverState {
    panel_id: usize,
    node_index: usize,
    tooltip: TooltipContent,
}

/// Typed field value for selection state — avoids serde_json::Value.
pub enum FieldValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}
```

**Selection state is data-space, not pixel-space.**  The WASM renderer
converts pixel coordinates to data-space via the inverse of (panel transform ×
scale mapping).  This ensures selections are meaningful when zoom changes.

### 6.2 Hit testing

Point selection requires finding which mark the user clicked.  Strategy:

- For `MarkBatchKind::Point`: spatial hash or brute-force over circle centers
  (circles have radius — check Euclidean distance).  `nearest=true` finds the
  closest even outside the mark.
- For `MarkBatchKind::Rect` / `Bar`: AABB containment test.
- For `Path` / `Polygon`: point-in-polygon (winding number).
- For `Line` / `Segment`: distance-to-segment with stroke-width tolerance.

Hit testing iterates `MarkBatch.data_indices` in z-order (last batch = topmost).
First hit wins (topmost mark).

### 6.3 Conditional encoding resolution

When selection state changes, the WASM renderer updates per-node visual
properties in the GPU instance buffer without rebuilding the SceneGraph:

```rust
fn resolve_conditionals(&mut self) {
    for cond in &self.scene.graph.interaction.conditionals {
        let Some(sel) = self.selections.get(&cond.selection_name) else { continue };
        for panel in &self.scene.graph.panels {
            for batch in &panel.marks {
                if let Some(indices) = &batch.data_indices {
                    for (i, &data_idx) in indices.iter().enumerate() {
                        let selected = sel.contains(data_idx);
                        let value = if selected { &cond.if_selected } else { &cond.if_not };
                        self.update_instance(batch, i, &cond.channel, value);
                    }
                }
            }
        }
    }
    self.render_frame();
}
```

`update_instance` modifies the instance buffer attribute (color, opacity, size)
in-place.  One GPU buffer upload + one draw call — no SceneGraph rebuild.

### 6.4 Zoom and pan

Per-panel `Affine2` transforms stored in `InteractionState.panel_transforms`.
`Affine2` is `glam::Affine2` — glam is a transitive dependency of wgpu
(already in the dependency tree for ferrum-wasm).  For ferrum-scene (which
must not depend on wgpu), panel transforms are serialized as `[f64; 6]`
(the six components of a 2D affine matrix).

```rust
fn on_wheel(&mut self, panel_id: usize, delta: f64, cursor_px: (f64, f64)) {
    let factor = 1.0 + delta * 0.001;
    let t = &mut self.panel_transforms[panel_id];
    // Zoom centered on cursor
    *t = t.pre_translate(-cursor_px.0, -cursor_px.1)
          .pre_scale(factor, factor)
          .pre_translate(cursor_px.0, cursor_px.1);

    // For CoordFixed panels, constrain to uniform scale
    if matches!(self.scene.graph.panels[panel_id].coord, CoordKind::Fixed { .. }) {
        t.constrain_uniform_scale();
    }

    self.select_tick_level(panel_id);
    self.render_frame();
    self.update_text_positions(panel_id);  // JS callback
}
```

**Tick level selection:** The WASM renderer reads
`InteractionConfig.tick_levels` and picks the level whose
`min_zoom..max_zoom` range contains the current zoom factor.  Tick labels
are CSS overlay text, swapped by showing/hiding the appropriate level's
`<div>` elements.

### 6.5 Href click-through

When a mark with an `href` value is clicked:
- In interactive mode: JS calls `window.open(href, '_blank')`
- In SVG mode: the SVG walker wraps the mark element in
  `<a xlink:href="...">` (works in browsers viewing the SVG)

### 6.6 Tooltip rendering

On `mousemove`, the JS glue layer:
1. Calls WASM hit-test → returns `Option<TooltipContent>`
2. If hit: positions a `<div class="ferrum-tooltip">` near the cursor with
   the field/value pairs rendered as a small table
3. If no hit: hides the tooltip

Tooltip styling is controlled by CSS (themeable).  The `<div>` is in the CSS
overlay layer.

### 6.7 Compound views in interactive mode

`HConcatChart`, `VConcatChart`, `FacetChart`, `RepeatChart`, `JointChart`,
and `ClusterMapChart` currently compose by calling `compose_svg_*` on SVG
strings post-render.  For interactive mode, this post-render SVG composition
is bypassed — compound views produce a **single unified SceneGraph** with
multiple panels that a single WASM renderer handles.

The Python-side `.interactive()` on compound views:
1. Each sub-chart calls `_render_to_scene()` (the new Rust binding) instead
   of `_render_to_svg()`.
2. The compound view's `interactive()` method collects the per-sub-chart
   SceneGraphs.
3. A `merge_scene_graphs()` utility (Python-side, in `_interactive.py`)
   combines them: panels are renumbered, decorations merged, viewport
   dimensions computed from the compound layout, `linked_panels` groups set
   based on shared encodings.
4. The merged SceneGraph is passed to a single `InteractiveChart` widget.

This means the WASM renderer sees one SceneGraph with N panels — it does not
know or care that the chart is compound.  Zoom/pan, selections, and linked
views work across panels because the SceneGraph's `InteractionConfig` encodes
the panel relationships.

**Design constraint:** `merge_scene_graphs()` is a pure data transformation
(takes list of SceneGraphs + layout info, returns one SceneGraph).  No side
effects, no renderer awareness.  The compound layout (panel positions,
spacing) is computed by the same layout logic used for SVG composition —
reuse, not rewrite.

### 6.8 anywidget protocol

**Python → JS (state pushed on change):**
- `scene_json`: updated SceneGraph after recomputation
- `interaction_config`: updated interaction config

**JS → Python (state pushed on change):**
- `selection_state`: current selection state (dict of selection name → state)

**Recomputation flow (Jupyter only):**
1. User zooms into mark_function → JS sends `{ type: "recompute",
   panel_id: 0, x_range: [lo, hi] }` via `model.set()`
2. Python `InteractiveChart` observes the change, re-evaluates the function
   callable over the new domain, rebuilds the affected panel's SceneGraph
3. Python sets `model.set('scene_json', updated_json)`
4. JS receives the update, calls `renderer.updateScene(json)` for a partial
   re-render

Only affected panels are re-sent — not the entire SceneGraph.

### 6.8 Animated transitions (Key channel)

When a new SceneGraph arrives (data update via anywidget or programmatic
`.update_data()` call), the renderer diffs old and new `MarkBatch` nodes
using the `keys` parallel array:

```rust
fn transition_scene(&mut self, new_scene: SceneGraph) {
    let old_graph = self.scene.as_ref().map(|s| &s.graph);
    let mut transitions = Vec::new();

    // Zip panels, then zip mark batches within each panel
    if let Some(old) = old_graph {
        for (old_panel, new_panel) in old.panels.iter().zip(&new_scene.panels) {
            for (old_batch, new_batch) in old_panel.marks.iter().zip(&new_panel.marks) {
                let (Some(old_keys), Some(new_keys)) = (&old_batch.keys, &new_batch.keys)
                    else { continue };
                for (new_idx, new_key) in new_keys.iter().enumerate() {
                    if let Some(old_idx) = old_keys.iter().position(|k| k == new_key) {
                        transitions.push(Transition {
                            from: old_batch.nodes[old_idx].clone(),
                            to: new_batch.nodes[new_idx].clone(),
                        });
                    }
                    // Unmatched new → fade in; unmatched old → fade out
                }
            }
        }
    }
    self.animate(transitions, Duration::from_millis(300));
}
```

Animation runs client-side: a `requestAnimationFrame` loop lerps instance
buffer attributes (position, size, color, opacity) between old and new values.
Duration is configurable via theme (`theme.transition_duration`).

---

## 7. Coordinate systems (sub-phase 11d)

### 7.1 CoordCartesian

Explicit axis limits and viewport control.

```python
chart.coord(CoordCartesian(xlim=(0, 100), ylim=(-5, 5), expand=True, clip=True))
```

**Implementation:**
- `xlim`/`ylim` override the auto-computed scale domain in `scale_resolve.rs`.
  The scale's `.map()` function uses the overridden domain for its linear
  mapping.
- `clip=True`: Panel clip rect is set to the plot area bounds (default
  behavior).  `clip=False`: clip rect is expanded to include margins (marks
  can overflow).
- `expand=True`: default padding added beyond the data extent.
  `expand=False`: axis starts/ends exactly at data min/max (or xlim/ylim if
  set).

**Rust-side change:** `CoordKind::Cartesian` fields flow from `ChartSpec` into
the geometry pass.  `scale_resolve.rs` reads `xlim`/`ylim` and overrides the
auto-computed domain.  No new module needed — this is a parameterization of
existing behavior.

**Interactive zoom integration:** When the user zooms, the WASM renderer
computes the visible data-space bounds from the panel transform and sends
them as `xlim`/`ylim` in the recomputation request.  Python rebuilds the
SceneGraph with the new bounds.

### 7.2 CoordFixed

Fixed aspect ratio.  `ratio=1.0` means one data unit on X equals one data
unit on Y in pixels.

```python
chart.coord(CoordFixed(ratio=1.0))
```

**Implementation:**
- Layout constraint: `compute_layout()` in `layout/mod.rs` receives the
  `ratio` and adjusts the panel dimensions to satisfy it.  If the allocated
  width is `w`, then height = `w * (y_range / x_range) * ratio`.
- The panel may be smaller than the allocated space — centered with margins.
- Interactive zoom: the WASM renderer constrains the panel transform to
  uniform scaling (zooming X by `k` also zooms Y by `k`).

### 7.3 CoordPolar

Polar coordinate transform for pie/donut/radial charts.

```python
chart.coord(CoordPolar(theta="x", start=0, direction=1))
```

**Implementation — geometry pass changes:**

A new code path in `scene_build.rs` detects `CoordKind::Polar` and applies
the polar transform to mark coordinates:

```
angle = scale_theta.map(value) * direction + start
radius = scale_r.map(value)
pixel_x = center_x + radius * sin(angle)
pixel_y = center_y - radius * cos(angle)
```

This transform applies after scale mapping but before SceneNode emission.
For `mark_arc`: the geometry pass emits `SceneNode::Path` with arc commands
(wedge shapes).  For `mark_point` in polar: points are positioned at
`(pixel_x, pixel_y)`.  For `mark_line` in polar: lines become spiral curves.

**Axis rendering in polar mode:**
- Angular axis: circle at `outer_radius` with tick marks around the perimeter.
  Tick labels positioned outside the circle, rotated to follow the arc.
- Radial axis: line from center outward with tick marks along it.

**Interactive hit-testing in polar mode:** The WASM renderer's inverse
transform converts pixel coordinates to `(angle, radius)` via
`atan2(px - cx, cy - py)` and `sqrt((px-cx)² + (py-cy)²)`.

### 7.4 CoordGeo

Geographic map projections.

```python
chart.coord(CoordGeo(projection="albers_usa"))
```

**Projections — pure Rust, no external dependency:**

Each projection is a variant of `GeoProjection` with `forward`/`inverse`
as methods on the enum:

```rust
impl GeoProjection {
    pub fn forward(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
        match self {
            Self::Mercator => mercator_forward(lon, lat),
            Self::AlbersUsa => albers_usa_forward(lon, lat),
            // ...
        }
    }
    pub fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        match self {
            Self::Mercator => mercator_inverse(x, y),
            Self::AlbersUsa => albers_usa_inverse(x, y),
            // ...
        }
    }
}
```

`Option` return type handles out-of-domain inputs (e.g., back-hemisphere
points for Orthographic).  `inverse` is needed for interactive hit-testing.

| Projection | Lines of code (approx) | Notes |
|---|---|---|
| Mercator | ~15 | `x = lon`, `y = ln(tan(π/4 + lat/2))` |
| Equirectangular | ~10 | `x = lon`, `y = lat` |
| EqualEarth | ~40 | Newton-Raphson iteration for inverse |
| NaturalEarth | ~40 | Polynomial approximation |
| Orthographic | ~20 | Great-circle clipping for back hemisphere |
| AlbersUsa | ~80 | Albers conic + Alaska/Hawaii insets |

All math uses `f64` standard library functions (`sin`, `cos`, `atan2`, `ln`).
No linear algebra needed — these are scalar trigonometric transforms.

**Implementation note:** `forward`/`inverse` are methods on the
`GeoProjection` enum itself, dispatching to concrete math via `match`.  No
trait, no dynamic dispatch — the enum has six variants with six `forward`
implementations, and the geometry pass calls `projection.forward(lon, lat)`
in a tight loop over vertices.  A `ProjectionFn` trait would be premature
generality (heuristic #9) — there is exactly one dispatch site.  If Phase 12
extension points need user-defined projections, the trait can be introduced
then with the concrete implementations migrated as `impl ProjectionFn for
GeoProjection`.

**mark_geoshape:** Consumes GeoJSON data.  The `geojson` crate (pure Rust,
serde-based) parses `FeatureCollection` → `Feature` → `Geometry`.  Each
polygon's vertices are projected via `forward()`, then emitted as
`SceneNode::Polygon`.  Feature properties are converted to an Arrow
RecordBatch so standard encoding channels (e.g., `color="population"`) work
through the existing pipeline.

**GeoJSON data path:** When the user passes a GeoJSON `FeatureCollection`
(Python dict or string) as `data` to `Chart(data).mark_geoshape()`, the
Python layer (`_coerce.py`) detects the GeoJSON structure (presence of
`"type": "FeatureCollection"` key) and handles it as a special case:

1. Extract feature properties into a polars DataFrame (one row per feature).
   This flows through the standard Arrow CDI transport for encoding resolution.
2. Serialize the GeoJSON geometry array (coordinates only — not properties)
   as a JSON string field in the ChartSpec (`geojson_geometries: Option<String>`).
3. Rust-side: the geometry pass deserializes the geometry JSON via the
   `geojson` crate, projects each vertex via `forward()`, and emits Polygon
   SceneNodes.  The properties RecordBatch provides the encoding values
   (color, opacity, etc.) per feature, matched by row index.

This keeps the Arrow CDI transport for tabular data (properties) and uses a
JSON sidecar for geometry coordinates — a natural split since GeoJSON
coordinates are nested arrays that don't map to columnar format.

---

## 8. Error handling

### 8.1 Rust error types

```rust
// crates/ferrum-scene/src/error.rs — shared
pub enum SceneError {
    Serialization(String),
}

// crates/ferrum-core/src/render/error.rs — existing, extended
pub enum RenderError {
    // existing variants...
    SceneConstruction(String),
    HtmlBundleAssembly(String),
}

// crates/ferrum-wasm/src/error.rs — new
pub enum WasmRenderError {
    GpuInit(String),         // adapter/device not available
    ContextLost,             // GPU context lost mid-render
    SceneDeserialization(String),
    TextureUpload(String),
    ShaderCompilation(String),
}
```

**Rules:**
- No `unwrap()` on fallible GPU operations.  `WasmRenderError` propagates to
  JS via `wasm_bindgen`'s `Result` → `JsValue` conversion.
- No panics in library code.  `ferrum-scene` and `ferrum-wasm` are `#[deny(clippy::unwrap_used)]`.
- `ferrum-core`'s existing `RenderError` is extended, not replaced.

### 8.2 Python error types

```python
class InteractiveRenderError(FerrumError):
    """Raised when the WASM renderer fails to initialize or render."""

class WasmNotAvailableError(FerrumError):
    """Raised when WASM artifacts are missing from the wheel."""
```

`.interactive()` raises `WasmNotAvailableError` if `src/ferrum/_wasm/` is
empty (defensive — should not happen with in-wheel distribution, but covers
development builds where WASM hasn't been compiled yet).

---

## 9. Stat/mark/encoding gap closure (sub-phase 11e)

### 9.1 mark_density(multiple=...)

Currently only `multiple="layer"` works.  Phase 11 adds:

- `"stack"`: Per-group KDE output → Stack position adjustment → stacked area.
  The Rust KDE transform already emits per-group curves.  The gap is applying
  Stack (from Phase 9) to KDE output.  Requires the `y` column of each group's
  density to be cumulatively offset.
- `"fill"`: Same as `"stack"` but normalized so each x-slice sums to 1.0.
  Extends the existing `PositionAdjust` enum with a `NormalizeStack` variant
  that divides each y by the sum at that x.  This is a new variant on the
  Phase 9 enum, not a separate mechanism.
- `"dodge"`: Per-group KDE output → Dodge position adjustment → side-by-side.
  The y-axis extent is subdivided by group count; each group's density is
  scaled to fit its subdivision.

### 9.2 mark_density(bw_adjust=) with string bandwidth

The Rust KDE transform resolves string bandwidth rules (`"scott"`,
`"silverman"`) from data statistics.  Phase 11 changes: after resolving the
rule to a numeric bandwidth, multiply by `bw_adjust`.  One-line change in
`transform/kde.rs` — compute rule result, then `bw *= bw_adjust`.

### 9.3 mark_hex full aggregates

The Rust Hex transform currently supports `count`, `mean`, `sum`.  Phase 11
adds: `min`, `max`, `median`, `std`, `var`.  These reuse the same
per-bin-group iteration; only the reduction function differs.  Follow the
pattern in the existing `Aggregate` transform.

### 9.4 mark_swarm(dodge=...)

Grouped beeswarm: `dodge="category_field"` groups data by category, computes
swarm positions within each group, then applies dodge offset between groups.
Rust-side change in `transform/swarm.rs`: partition points by dodge field →
swarm each partition independently → offset partitions by dodge width.

### 9.5 mark_function as multi-layer

Currently raises `NotImplementedError` when used in a multi-layer chart.
Fix: during `_render_inputs()`, detect `mark_function` layers, evaluate their
Python callables, inject the generated data as named data sources using the
existing `Layer.data_source` / `TransformSpec.name` mechanism from Phase 8a.
The function output becomes an ordinary data column — no special handling in
the renderer.

### 9.6 blend="additive"

**WASM renderer:** GPU-native additive blend state:
```rust
wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent::OVER,
}
```

**SVG backend:** `<feComposite operator="arithmetic" k2="1" k3="1"/>` filter.

**PNG backend (resvg):** resvg supports SVG filters — additive blending
works through the existing SVG → PNG path.

### 9.7 legend kwarg on Size, Shape, Opacity

Follow the pattern already implemented for Color: the `legend` field on
the encoding spec flows through `render/prepare.rs` legend construction.
When `None` or `False`, suppress the legend entry for that channel.

### 9.8 condition kwarg on all appearance channels

The `condition` kwarg on Size, Shape, Opacity, Fill, Stroke, StrokeDash,
StrokeWidth, FillOpacity, StrokeOpacity produces a `ConditionalEncoding`
that gets serialized into the SceneGraph's `InteractionConfig.conditionals`.
The Python-side change: `ChannelBase` validates the condition parameter,
extracts the selection name and if/else values, and passes them through to
the ChartSpec.  The WASM renderer resolves conditions at runtime (§6.3).
In SVG mode, conditions are silently ignored.

### 9.9 TimeScale calendar-aware ticks

Replace the approximate `MONTH = 30 * DAY` in `scale/ticks.rs` with
calendar-aware tick generation.  When the domain spans months or years, snap
ticks to calendar boundaries (Jan 1, Feb 1, etc.) instead of fixed 30-day
intervals.  Use the `time` crate's `Date` type for calendar arithmetic
(already available as a transitive dependency via `chrono` or `arrow`).

### 9.10 Key channel wiring

The `Key(field)` encoding class exists.  Phase 11 wires it through:
1. Python: `Key` encoding values are included in the ChartSpec
2. Rust: geometry pass reads the key field, populates
   `MarkBatch.keys` in the SceneGraph
3. WASM: animated transitions use keys for object constancy (§6.8)

---

## 10. Python API surface

### 10.1 New modules

| Module | Contents |
|---|---|
| `src/ferrum/selection.py` | `selection_point`, `selection_interval`, `selection_single`, `selection_multi`, `SelectionMark`, `Selection` class |
| `src/ferrum/_interactive.py` | `InteractiveChart` (anywidget subclass) |

### 10.2 Selection constructors

```python
def selection_point(
    *,
    fields: list[str] | None = None,
    encodings: list[str] | None = None,
    nearest: bool = False,
    toggle: str = "event.shiftKey",
    on: str = "click",
    clear: str = "mouseout",
    resolve: Literal["global", "union", "intersect"] = "global",
    name: str | None = None,
) -> Selection: ...

def selection_interval(
    *,
    fields: list[str] | None = None,
    encodings: list[str] | None = None,
    translate: bool = True,
    zoom: bool = True,
    mark: SelectionMark | None = None,
    resolve: Literal["global", "union", "intersect"] = "global",
    name: str | None = None,
) -> Selection: ...

selection_single = partial(selection_point, toggle=False)
selection_multi = partial(selection_point, toggle="event.shiftKey")
```

`Selection` is a frozen dataclass with a `.when()` builder for conditional
encodings:

```python
@dataclass(frozen=True)
class Selection:
    name: str
    kind: Literal["point", "interval"]
    params: dict  # all constructor kwargs except name

    def when(self, if_encoding) -> _SelectionCondition:
        """Start a conditional encoding: sel.when(Color("x")).otherwise(value("#ccc"))"""
        return _SelectionCondition(selection=self, if_encoding=if_encoding)

@dataclass(frozen=True)
class _SelectionCondition:
    selection: Selection
    if_encoding: Any

    def otherwise(self, else_encoding) -> ConditionalSpec:
        return ConditionalSpec(
            selection_name=self.selection.name,
            if_selected=self.if_encoding,
            if_not=else_encoding,
        )
```

Usage:

```python
sel = selection_point(name="hover", nearest=True, on="mouseover")
chart.encode(
    color=Color("category", condition=sel.when(Color("category")).otherwise(value("#ccc")))
)
```

### 10.3 InteractiveChart

```python
class InteractiveChart(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "_wasm" / "ferrum-interactive.js"
    _css = pathlib.Path(__file__).parent / "_wasm" / "ferrum-interactive.css"

    scene_json = traitlets.Unicode("").tag(sync=True)
    interaction_config = traitlets.Unicode("").tag(sync=True)
    selection_state = traitlets.Dict({}).tag(sync=True)

    def save(self, path: str | Path, **kwargs) -> None:
        """Save as self-contained HTML file."""

    def on_selection_change(self, callback: Callable) -> None:
        """Register Python callback for selection state changes."""
        self.observe(callback, names=["selection_state"])
```

### 10.4 Updated Chart methods

```python
# Chart.interactive() — returns InteractiveChart (no longer a stub)
def interactive(self) -> InteractiveChart: ...

# Chart.add_selection() — stores selections (no longer ignored)
def add_selection(self, *selections: Selection) -> Chart: ...

# Chart.conditional() — conditional encoding based on selection
def conditional(self, selection: Selection, **encodings) -> Chart: ...
```

### 10.5 Updated display.save_chart

Extends the format dispatch:
- `"html"` → build InteractiveChart, call its `.save()` for standalone bundle
- `"json"` → serialize ChartSpec via `.to_json()`

### 10.6 Coordinate system classes (updated)

```python
# coord.py — all four now functional, no NotImplementedError

@dataclass(frozen=True)
class CoordCartesian:
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    expand: bool = True
    clip: bool = True

@dataclass(frozen=True)
class CoordFixed:
    ratio: float = 1.0
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    expand: bool = True
    clip: bool = True

@dataclass(frozen=True)
class CoordPolar:
    theta: Literal["x", "y"] = "x"
    start: float = 0.0
    direction: Literal[1, -1] = 1

@dataclass(frozen=True)
class CoordGeo:
    projection: Literal[
        "mercator", "albers_usa", "equal_earth",
        "natural_earth", "orthographic", "equirectangular"
    ] = "mercator"
```

All are frozen dataclasses — values, not mutable config objects.  Follows
the existing `CoordFlip` pattern.

### 10.7 Public API exports

Add to `src/ferrum/__init__.py`:

```python
from ferrum.selection import (
    selection_point, selection_interval, selection_single,
    selection_multi, Selection, SelectionMark,
)
from ferrum._interactive import InteractiveChart
from ferrum.coord import CoordCartesian, CoordFixed, CoordPolar, CoordGeo
```

---

## 11. Packaging and distribution

### 11.1 WASM binary in the wheel

The `.wasm` + `.js` artifacts are included in the wheel as package data:

```toml
# pyproject.toml
[tool.maturin]
include = ["src/ferrum/_wasm/*.wasm", "src/ferrum/_wasm/*.js"]
```

### 11.2 New Python dependency

```toml
[project]
dependencies = [
    # existing...
    "anywidget>=0.9",
]
```

`anywidget` is a hard dependency (~50KB, minimal transitive deps).
`.interactive()` always works without optional extras.

### 11.3 New Rust dependencies

**ferrum-scene:**
- `serde` 1.x, `serde_json` 1.x (already workspace deps)

**ferrum-wasm:**
- `wgpu` 24.x with `webgl` feature
- `wasm-bindgen` 0.2.x
- `web-sys` 0.3.x (features: HtmlCanvasElement, etc.)
- `js-sys` 0.3.x
- `lyon` 1.x (path tessellation)
- `geojson` 0.24.x (GeoJSON parsing for mark_geoshape)

**ferrum-core additions:**
- `geojson` 0.24.x (for GeoJSON → Arrow RecordBatch conversion in data
  coercion layer)

---

## 12. Testing strategy

### 12.1 Sub-phase 11a (scene graph extraction)

- **Golden SVG tests:** byte-identical output before and after refactor.  This
  is the primary validation.  Any byte difference is a regression.
- **Round-trip test:** `build_scene()` → `walk_svg()` → compare against
  direct `render_svg()` output (before refactor) for a set of representative
  charts.
- **SceneGraph serialization:** `serde_json::to_string()` →
  `serde_json::from_str()` round-trip for every SceneNode variant.

### 12.2 Sub-phase 11b (WASM renderer)

- **Visual snapshot tests:** render known SceneGraphs via WASM, capture
  screenshots via headless Chromium (playwright or similar), compare against
  reference PNGs.
- **HTML output test:** `.save("chart.html")` produces valid HTML, loads
  without errors in headless browser.
- **JSON output test:** `.save("chart.json")` produces valid JSON,
  deserializes back to `SceneGraph`.

### 12.3 Sub-phase 11c (interaction)

- **Selection state tests:** programmatic event dispatch in headless browser →
  verify selection state matches expected (e.g., click mark → point selection
  contains correct data index).
- **anywidget integration test:** in a Jupyter kernel, create
  `InteractiveChart`, verify widget state sync via `model.get()`/`model.set()`.

### 12.4 Sub-phase 11d (coordinates + marks)

- **New golden SVGs** for each coordinate system and deferred mark.
- **Projection accuracy:** forward/inverse round-trip for each GeoProjection
  variant — `inverse(forward(lon, lat))` should recover `(lon, lat)` within
  `1e-10` tolerance.
- **Polar geometry:** arc marks emit correct SVG path commands (verified by
  golden test).

### 12.5 Sub-phase 11e (stat/mark gaps)

- **mark_density(multiple="stack"):** golden SVG + numeric correctness against
  known density values.
- **mark_hex full aggregates:** `min`, `max`, `median`, `std`, `var` match
  offline-computed reference values.
- **mark_swarm(dodge=...):** golden SVG showing grouped swarm layout.
- **mark_function multi-layer:** chart with function layer + data layer renders
  correctly.
- **TimeScale calendar ticks:** tick labels snap to calendar boundaries
  (Jan, Feb, ...) not 30-day intervals.

---

## 13. Decisions log

| Decision | Choice | Rationale | Date |
|---|---|---|---|
| Scene graph strategy | Extract shared SceneGraph IR consumed by all backends | Prevents rendering drift between static and interactive; enables Phase 12 extension points; matches spec §2 architecture diagram | 2026-05-13 |
| WASM distribution | In the Python wheel as package data | Zero-friction `.interactive()`; wheel grows ~2–4 MB (acceptable given ~10–15 MB native extension) | 2026-05-13 |
| Shared crate name | `ferrum-scene` | Contains only the SceneGraph IR types + serde.  Lighter than `ferrum-shared` (which implies engine code) | 2026-05-13 |
| Text rendering (WASM) | CSS overlay (DOM text) | Spec §3.17 mandates; accessible, no font bundling in browser | 2026-05-13 |
| GPU API | wgpu with `webgl` feature (WebGPU + WebGL2 fallback) | Spec §3.17; no Vello (compute-shader dependency breaks WebGL2 fallback) | 2026-05-13 |
| Path tessellation | `lyon` crate | Pure Rust, compiles to WASM, well-maintained, standard choice for 2D GPU rendering | 2026-05-13 |
| GeoJSON parsing | `geojson` crate | Pure Rust, serde-based, lightweight | 2026-05-13 |
| Projection math | Hand-rolled (~20–80 LOC per projection) | 6 projections at visualization-grade precision; `proj` crate is heavyweight geodesy library | 2026-05-13 |
| Jupyter integration | `anywidget` (hard dependency) | Lightweight (~50KB), handles widget protocol, ESM loading, state sync; avoids heavier ipywidgets | 2026-05-13 |
| Standalone HTML scope | SceneGraph + WASM + client-side selections; no data shipped | Compact files; data-dependent recomputation only in Jupyter | 2026-05-13 |
| Tick updates on zoom | Pre-computed multi-level ticks in SceneGraph | Instant feedback, no Python round-trip for basic zoom | 2026-05-13 |
| CoordGeo scope | Included in Phase 11 (sub-phase 11d) | User decision: full spec delivery, no deferred features after Phase 11 | 2026-05-13 |
| All remaining NotImplementedError | Resolved in Phase 11 | User decision: zero deferred features after Phase 11 | 2026-05-13 |

---

## 14. Spec notes for ferrum-spec.md

After Phase 11 lands, update the following in `ferrum-spec.md`:

1. Remove the Phase 7/8a/8b/9 implementation notes that say "deferred to
   Phase 11" — the features are now implemented.
2. Update §3.17 backend selection table to reflect actual wgpu implementation
   details.
3. Add dated note: "(2026-05-1X) Phase 11: all coordinate systems, deferred
   marks, interactive features, and stat/mark gaps implemented."
4. Update the architecture diagram in §2 to show `ferrum-scene` crate.
