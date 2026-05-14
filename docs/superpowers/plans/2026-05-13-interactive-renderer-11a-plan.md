# Phase 11a — Scene Graph Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a shared `SceneGraph` IR from the ferrum-core render pipeline so all backends (SVG, PNG, WASM) consume the same intermediate representation. SVG/PNG output must be byte-identical after the refactor.

**Architecture:** Create a `ferrum-scene` crate defining the SceneGraph types. Refactor `render/draw.rs` and `render/marks/*.rs` to emit `Vec<SceneNode>` instead of writing to `SvgBuffer`. Add `render/svg_walk.rs` that consumes the SceneGraph and produces identical SVG via the existing `SvgBuffer` API. Wire entry points in `render/mod.rs`.

**Tech Stack:** Rust, serde, serde_json. No new external dependencies in ferrum-core. ferrum-scene depends only on serde + serde_json.

**Spec:** `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §3 (SceneGraph IR), §4 (Scene graph extraction).

---

## File map

### New files

| File | Purpose |
|---|---|
| `crates/ferrum-scene/Cargo.toml` | Crate manifest — serde + serde_json only |
| `crates/ferrum-scene/src/lib.rs` | Re-exports all public types |
| `crates/ferrum-scene/src/types.rs` | SceneGraph, Panel, MarkBatch, SceneNode, style types |
| `crates/ferrum-scene/src/selection.rs` | SelectionSpec, ConditionalEncoding, InteractionConfig (stub — wired in 11c) |
| `crates/ferrum-core/src/render/scene_build.rs` | `build_scene()` orchestrator: PreparedInputs + LayoutResult → SceneGraph |
| `crates/ferrum-core/src/render/svg_walk.rs` | `walk_svg()`: SceneGraph → SVG string via SvgBuffer |

### Modified files

| File | Change |
|---|---|
| `Cargo.toml` (workspace root) | Add `crates/ferrum-scene` to `members` |
| `crates/ferrum-core/Cargo.toml` | Add `ferrum-scene` dependency |
| `crates/ferrum-core/src/render/mod.rs` | Wire `build_scene()` → `walk_svg()` into `render_svg` / `render_png` |
| `crates/ferrum-core/src/render/draw.rs` | Change `dispatch_mark` to return `Vec<SceneNode>` instead of writing to SvgBuffer; `MetadataColumns` emits parallel vecs instead of SVG wrappers |
| `crates/ferrum-core/src/render/marks/point.rs` | `draw(ctx, out)` → `build(ctx) -> MarkBuildResult` |
| `crates/ferrum-core/src/render/marks/line.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/bar.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/area.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/rect.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/rule.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/text.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/tick.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/polygon.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/segment.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/ribbon.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/image.rs` | Same refactor |
| `crates/ferrum-core/src/render/marks/mod.rs` | Update dispatch macro |
| `crates/ferrum-core/src/render/marks/axis.rs` | Emit `Vec<SceneNode>` for axis elements |
| `crates/ferrum-core/src/render/marks/legend.rs` | Emit `Vec<SceneNode>` for legend elements |
| `crates/ferrum-core/src/render/marks/strip_title.rs` | Emit `Vec<SceneNode>` for strip titles |

### Unchanged files

PreparedInputs (`prepare.rs`), LayoutResult (`layout/`), ResolvedScales (`scale_resolve.rs`), position adjustments (`position.rs`), color resolution (`color/`), SvgBuffer (`svg.rs`), rasterize (`rasterize.rs`), PNG encoding (`png.rs`), compositor (`compositor.rs`), grid composition (`grid_compose.rs`), all Python code.

---

## Task 1: Create ferrum-scene crate with SceneGraph types

**Files:**
- Create: `crates/ferrum-scene/Cargo.toml`
- Create: `crates/ferrum-scene/src/lib.rs`
- Create: `crates/ferrum-scene/src/types.rs`
- Create: `crates/ferrum-scene/src/selection.rs`
- Modify: `Cargo.toml` (workspace root)

### Steps

- [ ] **Step 1: Add ferrum-scene to workspace**

In the workspace root `Cargo.toml`, add `"crates/ferrum-scene"` to the `members` array:

```toml
[workspace]
members = ["crates/ferrum-core", "crates/ferrum-scene"]
```

- [ ] **Step 2: Create ferrum-scene Cargo.toml**

```toml
[package]
name = "ferrum-scene"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 3: Create types.rs with all SceneGraph IR types**

Write `crates/ferrum-scene/src/types.rs` with the full type definitions from spec §3. These are the types that both ferrum-core (producer) and ferrum-wasm (consumer) will share. All types derive `Debug, Clone, Serialize, Deserialize`. Every field that could contain a string-typed value uses an enum instead.

The types are: `SceneGraph`, `Panel`, `MarkBatch`, `MarkBatchKind`, `BlendMode`, `SceneNode`, `Color`, `FillStroke`, `StrokeStyle`, `TextStyle`, `FontWeight`, `TextAnchor`, `TextBaseline`, `PathCmd`, `ImageData`, `ImageMime`, `Rect`, `TooltipContent`, `TooltipField`, `CoordKind`, `PolarThetaChannel`, `PolarDirection`, `GeoProjection`.

Refer to spec §3.1–§3.10 for exact field definitions. Key implementation notes:
- `Color` uses `(u8, u8, u8, u8)` — RGBA. Provide a `Color::rgb(r, g, b)` constructor that sets `a = 255`.
- `FillStroke` includes `opacity: f64` and `stroke_dash: Option<Vec<f64>>` — richer than the current SVG buffer's `FillStroke`. The SVG walker will map the extra fields to inline SVG attributes.
- `Rect` is `{ x: f64, y: f64, w: f64, h: f64 }` — same semantics as the layout engine's `Rect`.
- All floats are `f64`. No `f32` — consistency with the existing codebase.

- [ ] **Step 4: Create selection.rs with stub types**

Write `crates/ferrum-scene/src/selection.rs` with the selection and interaction types from spec §3.7–§3.9. These are stubs for 11a — they exist so `SceneGraph` can have the `selections` and `interaction` fields, but they will be empty until 11c wires them.

Types: `SelectionSpec`, `SelectionResolve`, `ChannelName`, `SelectionMarkStyle`, `EventExpr`, `ConditionalEncoding`, `EncodingValue`, `FieldValue`, `InteractionConfig`, `PanelTickLevels`, `TickLevel`, `Tick`.

- [ ] **Step 5: Create lib.rs re-exporting all public types**

```rust
pub mod types;
pub mod selection;

pub use types::*;
pub use selection::*;
```

- [ ] **Step 6: Verify ferrum-scene compiles**

Run: `cargo build -p ferrum-scene`
Expected: compiles with no errors, no warnings.

- [ ] **Step 7: Add serde round-trip test**

Create `crates/ferrum-scene/src/lib.rs` inline test (or `tests/` dir) that constructs a minimal `SceneGraph` with one `Panel`, one `MarkBatch` with a couple of `SceneNode::Circle` and `SceneNode::Rect`, serializes to JSON via `serde_json::to_string`, deserializes back, and asserts equality.

Run: `cargo test -p ferrum-scene`
Expected: PASS.

- [ ] **Step 8: Commit**

```
feat(scene): add ferrum-scene crate with SceneGraph IR types
```

---

## Task 2: Add ferrum-scene dependency to ferrum-core

**Files:**
- Modify: `crates/ferrum-core/Cargo.toml`

### Steps

- [ ] **Step 1: Add dependency**

```toml
[dependencies]
ferrum-scene = { path = "../ferrum-scene" }
```

- [ ] **Step 2: Verify ferrum-core still compiles**

Run: `cargo build -p ferrum-core`
Expected: compiles. No usage of ferrum-scene types yet — just dependency linkage.

- [ ] **Step 3: Verify existing tests still pass**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test`
Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```
chore(core): add ferrum-scene dependency to ferrum-core
```

---

## Task 3: Create MarkBuildResult and refactor dispatch_mark signature

This task changes the mark dispatch to return scene nodes instead of writing to SvgBuffer. It's the bridge between the old and new systems — after this, each mark module can be refactored independently.

**Files:**
- Modify: `crates/ferrum-core/src/render/draw.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs`

### Steps

- [ ] **Step 1: Define MarkBuildResult in draw.rs**

Add to `draw.rs`:

```rust
use ferrum_scene::{SceneNode, MarkBatch, MarkBatchKind, TooltipContent, TooltipField, BlendMode};

pub struct MarkBuildResult {
    pub kind: MarkBatchKind,
    pub nodes: Vec<SceneNode>,
    pub data_indices: Option<Vec<usize>>,
    pub tooltips: Option<Vec<TooltipContent>>,
    pub hrefs: Option<Vec<Option<String>>>,
}
```

This is the return type for all refactored mark builders. It maps directly to `MarkBatch` but lives in ferrum-core (not ferrum-scene) because it's an intermediate — the scene builder assembles `MarkBatch` from it.

- [ ] **Step 2: Add MetadataColumns::build_parallel_vecs method**

Currently `MetadataColumns` wraps SVG elements with `<a>`, `<title>`, `<desc>`. For the scene graph path, it instead produces parallel vectors of tooltip content and hrefs. Add a method:

```rust
impl MetadataColumns {
    pub fn build_parallel(
        &self,
        n: usize,
    ) -> (Option<Vec<TooltipContent>>, Option<Vec<Option<String>>>) {
        let tooltips = self.tooltip.as_ref().map(|col| {
            col.iter()
                .map(|opt| TooltipContent {
                    fields: vec![TooltipField {
                        name: "value".to_string(),
                        value: opt.clone().unwrap_or_default(),
                    }],
                })
                .collect()
        });
        let hrefs = self.href.as_ref().map(|col| col.clone());
        (tooltips, hrefs)
    }
}
```

- [ ] **Step 3: Create dispatch_mark_build function**

Add alongside the existing `dispatch_mark`:

```rust
pub fn dispatch_mark_build(mark: &Mark, ctx: &DrawCtx) -> MarkBuildResult {
    // Uses the same for_each_mark! macro but calls build() instead of draw()
    // Initially, all marks still only have draw() — they'll be refactored one by one
    // This function is a placeholder that calls draw() into a temporary SvgBuffer
    // and returns an empty MarkBuildResult. It will be replaced as marks are refactored.
    todo!("Marks will be refactored incrementally in subsequent tasks")
}
```

This will be filled in as each mark module is refactored. Do NOT wire it into render_svg yet.

- [ ] **Step 4: Verify compilation**

Run: `cargo build -p ferrum-core`
Expected: compiles. The `todo!()` is never called at runtime yet.

- [ ] **Step 5: Commit**

```
feat(render): add MarkBuildResult type and dispatch_mark_build stub
```

---

## Task 4: Refactor mark modules to emit SceneNodes

This is the core refactoring work. Each mark module's `draw(ctx, out)` is split into:
- `build(ctx) -> MarkBuildResult` — pure geometry computation, returns SceneNodes
- The old `draw(ctx, out)` is removed (or kept temporarily as a thin wrapper during the transition)

**The refactoring pattern is identical for every mark:**
1. Replace `out.circle(...)` / `out.rect(...)` / etc. calls with `SceneNode::Circle { ... }` / `SceneNode::Rect { ... }` pushes to a `Vec<SceneNode>`
2. Replace `meta.open(i, out)` / `meta.close(i, out)` with `MetadataColumns::build_parallel_vecs()` called once before the loop
3. Position offsets, color resolution, scale mapping — all stay exactly as-is
4. Return `MarkBuildResult` with the nodes + metadata vecs

**Files:**
- Modify: all files in `crates/ferrum-core/src/render/marks/`

### Steps

- [ ] **Step 1: Refactor point.rs**

Change `pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer)` to `pub fn build(ctx: &DrawCtx) -> MarkBuildResult`.

The geometry math stays identical. Every `out.circle(cx, cy, r, &style)` becomes `nodes.push(SceneNode::Circle { cx, cy, r, style: fill_stroke })`. Every `out.rect(rect, &style, corner_radius)` becomes `nodes.push(SceneNode::Rect { ... })`. The `emit_shape()` helper that dispatches by shape kind (circle/square/cross/diamond/triangle) becomes a match that pushes the appropriate SceneNode variant.

For cross shapes (two perpendicular lines): push two `SceneNode::Line` nodes.
For diamond/triangle shapes: push `SceneNode::Path` with the appropriate path commands.

Metadata: call `MetadataColumns::build_parallel_vecs()` once before the per-row loop. `data_indices`: populate from row indices.

- [ ] **Step 2: Refactor line.rs**

Change `draw` to `build`. The grouping logic (per-color, per-detail) stays identical. Each group's polyline becomes:
- For "linear" interpolation: `SceneNode::Path { commands: [MoveTo, LineTo, LineTo, ...], style, closed: false }`
- For step/step-before/step-after: same but with the step-interpolation point insertion
- For "monotone": same

The `build_line_path()` helper returns `Vec<PathCmd>` instead of a `d` string.

`stroke_cap` / `stroke_join` handling: these become fields on the `StrokeStyle` or `FillStroke` — add `stroke_cap: Option<StrokeCap>` and `stroke_join: Option<StrokeJoin>` enums to `ferrum-scene/src/types.rs` if not already present.

- [ ] **Step 3: Refactor bar.rs**

Change `draw` to `build`. The four dispatch paths (ordinal, ordinal_y, quantitative, quantitative_horizontal) each become a match arm returning `MarkBuildResult`. Each `out.rect(...)` becomes `SceneNode::Rect { ... }`.

- [ ] **Step 4: Refactor area.rs**

Change `draw` to `build`. Each group's filled area path becomes `SceneNode::Path { commands, style, closed: true }`. The `build_area_path` / `build_stacked_area_path` helpers return `Vec<PathCmd>` instead of `d` strings. Border lines (S9, S10) become additional `SceneNode::Path` nodes with stroke-only style.

- [ ] **Step 5: Refactor rect.rs**

Change `draw` to `build`. Three dispatch paths (quantitative_range, ordinal_range, heatmap) each return `MarkBuildResult` with `SceneNode::Rect` nodes.

- [ ] **Step 6: Refactor rule.rs**

Change `draw` to `build`. Four modes (ranged vertical, ranged horizontal, full-width horizontal, full-height vertical) each push `SceneNode::Line` nodes.

- [ ] **Step 7: Refactor text.rs**

Change `draw` to `build`. Each row becomes `SceneNode::Text { x: px + dx, y: py + dy, content: label, style: TextStyle { font_size, font_weight, anchor, baseline, angle, color, opacity } }`.

The text formatting logic (numeric formatting, time formatting, truncation with ellipsis) stays in the builder — the SceneNode receives the final formatted string.

- [ ] **Step 8: Refactor tick.rs, segment.rs, ribbon.rs, polygon.rs, image.rs**

Same pattern for each:
- `tick.rs`: `SceneNode::Line` per tick mark
- `segment.rs`: `SceneNode::Line` per segment (x, y → x2, y2)
- `ribbon.rs`: `SceneNode::Path` per ribbon group (closed polygon from top + bottom edges)
- `polygon.rs`: `SceneNode::Polygon` per group
- `image.rs`: `SceneNode::Image` per row

- [ ] **Step 9: Refactor axis.rs, legend.rs, strip_title.rs**

These emit decoration SceneNodes (not MarkBatch — they go into `Panel.axes`, `Panel.grid`, `SceneGraph.decorations`):
- `axis.rs`: returns `Vec<SceneNode>` for tick marks (Line), tick labels (Text), axis title (Text), gridlines (Line)
- `legend.rs`: returns `Vec<SceneNode>` for legend entries (Rect/Circle + Text)
- `strip_title.rs`: returns `SceneNode::Text`

- [ ] **Step 10: Update marks/mod.rs dispatch**

Update the `for_each_mark!` macro (or the dispatch function) to call `build()` instead of `draw()` and collect `MarkBuildResult`:

```rust
pub fn dispatch_mark_build(mark: &Mark, ctx: &DrawCtx) -> MarkBuildResult {
    match mark {
        Mark::Point => super::marks::point::build(ctx),
        Mark::Line => super::marks::line::build(ctx),
        Mark::Bar => super::marks::bar::build(ctx),
        // ... all marks
    }
}
```

Remove the old `dispatch_mark` function (or mark it `#[deprecated]` during transition).

- [ ] **Step 11: Verify compilation**

Run: `cargo build -p ferrum-core`
Expected: compiles. The old `render_svg` entry point is NOT yet wired to the new build functions — it's broken at this point. The next task wires it.

- [ ] **Step 12: Commit**

```
refactor(render): convert all mark draw() functions to build() returning SceneNodes
```

---

## Task 5: Create scene_build.rs orchestrator

**Files:**
- Create: `crates/ferrum-core/src/render/scene_build.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (add module declaration)

### Steps

- [ ] **Step 1: Write build_scene function**

`scene_build.rs` is the orchestrator that replaces the rendering loop currently in `render_svg`. It takes the same inputs (spec, batch, theme, viewport, config) and produces a `SceneGraph`.

```rust
use ferrum_scene::*;
use super::prepare::PreparedInputs;
use super::draw::{DrawCtx, MarkBuildResult, dispatch_mark_build, resolve_mark_style, MetadataColumns};
// ... other imports from layout, scale_resolve

pub fn build_scene(
    spec: &ChartSpec,
    prepared: &PreparedInputs,
    layout: &LayoutResult,
    theme: &ThemeInputs,
    config: &RenderConfig,
) -> Result<SceneGraph, RenderError> {
    let mut panels = Vec::new();
    let mut decorations = Vec::new();

    // For each panel (same iteration as render_svg currently does)
    for panel_layout in &layout.panels {
        let mut panel_grid = Vec::new();
        let mut panel_marks = Vec::new();
        let mut panel_axes = Vec::new();
        let mut panel_annotations = Vec::new();

        // Resolve scales for this panel
        let scales = scale_resolve::resolve_scales_for_panel(...);

        // Build gridlines
        panel_grid.extend(build_gridlines(&scales, panel_layout, theme));

        // For each layer
        for layer in &prepared.layers {
            let batch = resolve_layer_data(layer, &prepared.transform_outputs);
            let mark_style = resolve_mark_style(layer.mark_style.as_ref(), theme, &layer.mark);
            let ctx = DrawCtx { spec, panel: panel_layout, theme, scales: &scales, batch: &batch, mark_style: &mark_style };

            let result: MarkBuildResult = dispatch_mark_build(&layer.mark, &ctx);
            panel_marks.push(MarkBatch {
                kind: result.kind,
                nodes: result.nodes,
                data_indices: result.data_indices,
                tooltips: result.tooltips,
                hrefs: result.hrefs,
                keys: None,   // 11c
                blend: BlendMode::Normal,
            });
        }

        // Build axes
        panel_axes.extend(build_axes(&scales, panel_layout, theme));

        // Build strip title
        let strip_title = build_strip_title(panel_layout, theme);

        panels.push(Panel {
            id: panels.len(),
            plot_area: Rect { x: panel_layout.plot_area.x, y: panel_layout.plot_area.y,
                              w: panel_layout.plot_area.w, h: panel_layout.plot_area.h },
            clip: Rect { x: panel_layout.plot_area.x, y: panel_layout.plot_area.y,
                         w: panel_layout.plot_area.w, h: panel_layout.plot_area.h },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: panel_grid,
            marks: panel_marks,
            axes: panel_axes,
            annotations: panel_annotations,
            strip_title,
        });
    }

    // Build legend, chart title → decorations
    decorations.extend(build_legend(prepared, layout, theme));
    decorations.extend(build_chart_title(spec, layout, theme));

    Ok(SceneGraph {
        width: layout.viewport.w,
        height: layout.viewport.h,
        background: theme.background.map(|c| Color { r: c.r, g: c.g, b: c.b, a: c.a }),
        panels,
        decorations,
        selections: vec![],
        interaction: InteractionConfig::default(),
    })
}
```

The exact logic mirrors the current `render_svg` function's iteration pattern — same panel loop, same layer loop, same scale resolution. The difference is that instead of writing to `SvgBuffer`, it collects `MarkBuildResult` into `MarkBatch` and assembles the `SceneGraph`.

- [ ] **Step 2: Wire module declaration**

In `render/mod.rs`, add:
```rust
pub mod scene_build;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p ferrum-core`
Expected: compiles. Entry points not yet wired.

- [ ] **Step 4: Commit**

```
feat(render): add scene_build.rs orchestrator producing SceneGraph from PreparedInputs + LayoutResult
```

---

## Task 6: Create svg_walk.rs

**Files:**
- Create: `crates/ferrum-core/src/render/svg_walk.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (add module declaration)

### Steps

- [ ] **Step 1: Write walk_svg function**

`svg_walk.rs` takes a `SceneGraph` reference and produces an SVG string by calling the existing `SvgBuffer` methods. It has no knowledge of `DrawCtx`, `PreparedInputs`, or `LayoutResult`.

```rust
use ferrum_scene::*;
use super::svg::{SvgBuffer, FillStroke as SvgFillStroke, Stroke as SvgStroke, TextStyle as SvgTextStyle, TextAnchor as SvgTextAnchor};

pub fn walk_svg(scene: &SceneGraph) -> String {
    let viewport = Rect { x: 0.0, y: 0.0, w: scene.width, h: scene.height };
    let bg_color = scene.background.as_ref().map(scene_color_to_svg);
    let mut svg = SvgBuffer::new(viewport, bg_color, true /* embed_fonts */);

    for panel in &scene.panels {
        // Open clip group for panel plot area
        let clip_id = format!("panel-{}", panel.id);
        svg.clip_open(&clip_id, rect_to_svg(panel.clip));
        svg.use_clip_open(&clip_id);

        // Gridlines (behind marks)
        for node in &panel.grid {
            emit_node(&mut svg, node);
        }

        // Marks (z-ordered by batch order)
        for batch in &panel.marks {
            for (i, node) in batch.nodes.iter().enumerate() {
                // Href wrapping
                let href = batch.hrefs.as_ref().and_then(|h| h[i].as_deref());
                if let Some(url) = href { svg.a_open(url); }

                // Tooltip wrapping
                if let Some(tooltips) = &batch.tooltips {
                    let tt = &tooltips[i];
                    let text = tt.fields.iter()
                        .map(|f| format!("{}: {}", f.name, f.value))
                        .collect::<Vec<_>>()
                        .join(", ");
                    svg.title_elem(&text);
                }

                emit_node(&mut svg, node);

                if href.is_some() { svg.a_close(); }
            }
        }

        // Axes (on top of marks)
        for node in &panel.axes {
            emit_node(&mut svg, node);
        }

        svg.use_clip_close();

        // Strip title (outside clip)
        if let Some(title) = &panel.strip_title {
            emit_node(&mut svg, title);
        }

        // Annotations (outside clip)
        for node in &panel.annotations {
            emit_node(&mut svg, node);
        }
    }

    // Decorations (legends, chart title)
    for node in &scene.decorations {
        emit_node(&mut svg, node);
    }

    svg.finish()
}

fn emit_node(svg: &mut SvgBuffer, node: &SceneNode) {
    match node {
        SceneNode::Rect { x, y, w, h, style, corner_radius } => {
            let r = Rect { x: *x, y: *y, w: *w, h: *h };
            let svg_style = scene_fill_stroke_to_svg(style);
            let cr = if *corner_radius > 0.0 { Some(*corner_radius) } else { None };
            svg.rect(r, &svg_style, cr);
        }
        SceneNode::Circle { cx, cy, r, style } => {
            let svg_style = scene_fill_stroke_to_svg(style);
            svg.circle(*cx, *cy, *r, &svg_style);
        }
        SceneNode::Line { x1, y1, x2, y2, style } => {
            let svg_stroke = scene_stroke_to_svg(style);
            svg.line(*x1, *y1, *x2, *y2, &svg_stroke);
        }
        SceneNode::Path { commands, style, closed } => {
            let d = path_commands_to_d(commands, *closed);
            let svg_style = scene_fill_stroke_to_svg(style);
            svg.path(&d, &svg_style);
        }
        SceneNode::Text { x, y, content, style } => {
            let svg_style = scene_text_style_to_svg(style);
            svg.text(*x, *y, content, &svg_style);
        }
        SceneNode::Image { x, y, w, h, data } => {
            if let ImageData::Inline { bytes, .. } = data {
                svg.image(*x, *y, *w, *h, bytes);
            }
        }
        SceneNode::Polygon { points, style } => {
            let rings = vec![points.iter().map(|p| (p[0], p[1])).collect()];
            let svg_style = scene_fill_stroke_to_svg(style);
            svg.polygon(&rings, &svg_style);
        }
    }
}

// Conversion helpers: ferrum-scene types → SvgBuffer types
fn scene_color_to_svg(c: &Color) -> svg::Color { /* ... */ }
fn scene_fill_stroke_to_svg(s: &FillStroke) -> SvgFillStroke { /* ... */ }
fn scene_stroke_to_svg(s: &StrokeStyle) -> SvgStroke { /* ... */ }
fn scene_text_style_to_svg(s: &TextStyle) -> SvgTextStyle<'static> { /* ... */ }
fn path_commands_to_d(cmds: &[PathCmd], closed: bool) -> String { /* ... */ }
fn rect_to_svg(r: Rect) -> svg::Rect { /* ... */ }
```

The conversion helpers are thin mappers. The key invariant: `walk_svg(build_scene(inputs))` must produce the exact same SVG string as the old `render_svg(inputs)`.

- [ ] **Step 2: Wire module declaration**

In `render/mod.rs`:
```rust
pub mod svg_walk;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p ferrum-core`
Expected: compiles.

- [ ] **Step 4: Commit**

```
feat(render): add svg_walk.rs — SceneGraph → SVG string via SvgBuffer
```

---

## Task 7: Wire entry points and validate

This is the critical task — switch `render_svg` to go through the scene graph path and prove byte-identical output.

**Files:**
- Modify: `crates/ferrum-core/src/render/mod.rs`

### Steps

- [ ] **Step 1: Update render_svg to use build_scene → walk_svg**

Replace the rendering loop in `render_svg` with:

```rust
pub fn render_svg(...) -> Result<RenderOutput<String>, RenderError> {
    let prepared = prepare_render_inputs(spec, batch)?;
    let layout = compute_layout(spec, &prepared, theme)?;
    let scene = scene_build::build_scene(spec, &prepared, &layout, theme, config)?;
    let svg_string = svg_walk::walk_svg(&scene);
    Ok(RenderOutput { output: svg_string, warnings: prepared.warnings })
}
```

`render_png` still wraps `render_svg` so it gets the scene graph path automatically.

- [ ] **Step 2: Run cargo test**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test`
Expected: all Rust tests pass.

- [ ] **Step 3: Run pytest — golden SVG tests**

Run: `uv run pytest tests/ -v`
Expected: ALL tests pass. Golden SVGs are byte-identical.

If any golden test fails, this is a regression in the scene graph refactor. Do NOT update the golden — fix the scene_build/walk_svg code to match the old output exactly.

- [ ] **Step 4: Specifically test compound views**

The SVG compositor (`compositor.rs`, `grid_compose.rs`) operates on SVG strings AFTER `render_svg` returns. Since `render_svg` now goes through the scene graph, verify that:
- Faceted charts still render correctly
- `HConcatChart` / `VConcatChart` composition still works
- `RepeatChart` grid composition still works
- `JointChart` / `ClusterMapChart` composition still works

Run: `uv run pytest tests/ -k "facet or concat or repeat or joint or cluster" -v`
Expected: ALL pass.

- [ ] **Step 5: Run full snapshot verification**

Regenerate and visually inspect all golden PNGs:

Run: `uv run python scripts/snapshot-goldens.py`

Then read each PNG to confirm charts render correctly — no missing elements, no blank panels, no mis-stacked bars.

- [ ] **Step 6: Commit**

```
refactor(render): wire render_svg through SceneGraph — byte-identical SVG output validated
```

---

## Task 8: Add render_scene_json entry point (for 11b)

**Files:**
- Modify: `crates/ferrum-core/src/render/mod.rs`
- Modify: `crates/ferrum-core/src/render/binding.rs`

### Steps

- [ ] **Step 1: Add render_scene_json function**

In `render/mod.rs`:

```rust
pub fn render_scene_json(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &RenderConfig,
) -> Result<String, RenderError> {
    let prepared = prepare_render_inputs(spec, batch)?;
    let layout = compute_layout(spec, &prepared, theme)?;
    let scene = scene_build::build_scene(spec, &prepared, &layout, theme, config)?;
    serde_json::to_string(&scene).map_err(|e| RenderError::SceneConstruction(e.to_string()))
}
```

- [ ] **Step 2: Add PyO3 binding**

In `binding.rs`, add:

```rust
#[pyfunction]
pub fn render_interactive(
    py: Python<'_>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    // Same input parsing as render_svg
    let theme_inputs = parse_theme(theme)?;
    let config_inputs = parse_config(config)?;
    let batches = collect_batches(data)?;
    let batch = &batches[0];

    super::render_scene_json(spec, batch, &theme_inputs, viewport.into(), &config_inputs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}
```

- [ ] **Step 3: Register in lib.rs**

Add to the `_core` module registration:

```rust
m.add_function(wrap_pyfunction!(render::binding::render_interactive, m)?)?;
```

- [ ] **Step 4: Verify**

Run: `cargo build -p ferrum-core && DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test`
Expected: compiles and all tests pass.

Run: `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import render_interactive; print('OK')"`
Expected: `OK` (function is importable).

- [ ] **Step 5: Commit**

```
feat(render): add render_interactive PyO3 binding returning SceneGraph JSON
```

---

## Task 9: Clean up old draw path

**Files:**
- Modify: `crates/ferrum-core/src/render/draw.rs`
- Modify: `crates/ferrum-core/src/render/marks/*.rs`

### Steps

- [ ] **Step 1: Remove old dispatch_mark function**

Delete the old `dispatch_mark(mark, ctx, out)` that wrote to SvgBuffer. It's no longer called by any entry point.

- [ ] **Step 2: Remove old draw() functions from mark modules**

Each mark module should now only export `build(ctx) -> MarkBuildResult`. Remove any remaining `draw(ctx, out)` functions or transition wrappers.

- [ ] **Step 3: Run full test suite**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test && uv run pytest tests/ -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```
refactor(render): remove old draw-to-SvgBuffer path — all rendering goes through SceneGraph
```

---

## Validation checklist (run before marking 11a done)

- [ ] `cargo test -p ferrum-scene` — SceneGraph serde round-trip
- [ ] `cargo test -p ferrum-core` — all existing Rust tests
- [ ] `uv run pytest tests/ -v` — all Python tests including golden SVG comparisons
- [ ] `uv run pytest tests/ -k "facet or concat or repeat or joint or cluster" -v` — compound view goldens
- [ ] `uv run python scripts/snapshot-goldens.py` — regenerate all golden PNGs, visually inspect
- [ ] `render_interactive` PyO3 binding is importable and returns valid JSON
- [ ] No byte differences in any golden SVG compared to pre-refactor output

---

## 11a Implementation Audit (2026-05-14)

Audit of the completed 11a implementation against the design spec
`docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §3–§4.
Future sessions need this to understand what changed and why.

### Intentional divergences from spec §3 (required for byte-identical golden SVGs)

The spec's type definitions assumed a clean WASM-first design. The actual
implementation needed adjustments so the SVG walker (`svg_walk.rs`) could
reproduce the *exact* byte output of the old `render_svg` path. All changes
are additive — no spec fields were removed or renamed.

| # | Type | Spec says | Implementation has | Reason |
|---|---|---|---|---|
| 1 | `SceneGraph` | `decorations: Vec<SceneNode>` | `title: Vec<SceneNode>`, `legend: Vec<SceneNode>`, `decorations: Vec<SceneNode>` | Old `render_svg` emits title → panels → legend in that order. A single `decorations` vec loses this ordering, producing different SVG. |
| 2 | `Panel.strip_title` | `Option<SceneNode>` | `Vec<SceneNode>` | Strip title is 2 nodes (background rect + text). `Option<SceneNode>` forces a `Group` wrapper → extra `<g>` in SVG not present in old output. |
| 3 | `MarkBatch` | no cap/join fields | `stroke_cap: Option<StrokeCap>`, `stroke_join: Option<StrokeJoin>` | `mark_line` and `mark_area` wrap output in `<g stroke-linecap="..." stroke-linejoin="...">`. This is a batch-level attribute, not per-node. |
| 4 | `SceneNode` | 7 variants (Rect, Circle, Line, Path, Text, Image, Polygon) | +3 variants: `Polyline`, `Group`, `Raw` | `Polyline`: old `mark_line` emits `<polyline>` for linear interpolation, not `<path>`. `Group`: needed for `<g>` attribute wrappers. `Raw`: legend colorbar gradient `<defs>` can't be expressed as typed nodes (`fill="url(#...)"` is not a `Color`). |
| 5 | `FontWeight` | `Normal`, `Bold` | + `Custom(String)` | Themes use numeric CSS weights like `"600"` for axis titles. |
| 6 | `TextBaseline` | `Top`, `Middle`, `Bottom`, `Alphabetic` | + `Custom(String)` | `mark_text(baseline="top")` passes the user-facing string verbatim to SVG `dominant-baseline`; `"top"` ≠ `"hanging"` (the SVG-canonical name). |
| 7 | `PathCmd` | `MoveTo`, `LineTo`, `QuadTo`, `CubicTo`, `ArcTo`, `Close` | + `HLineTo`, `VLineTo` | Step interpolation in `mark_line` emits `H`/`V` SVG path commands. |
| 8 | `PathCmd` field style | positional tuples: `MoveTo(f64, f64)` | named fields: `MoveTo { x: f64, y: f64 }` | serde `#[serde(tag = "op")]` requires struct variants, not tuple variants. |
| 9 | `StrokeStyle` | `color`, `width`, `opacity`, `dash` | + `stroke_cap: Option<StrokeCap>`, `stroke_join: Option<StrokeJoin>` | Needed on `Polyline` nodes so the SVG walker can detect and emit the `<g>` wrapper. (Plan §"Type gaps" identified this pre-implementation.) |
| 10 | `TextStyle` | no `font_family` | + `font_family: String` | Every SVG `<text>` needs a `font-family` attribute. (Plan §"Type gaps" identified this pre-implementation.) |

### Resolved gaps (found during audit, fixed same session)

| # | Gap | Fix |
|---|---|---|
| 1 | `description` encoding channel dropped by `build_tooltips_and_hrefs()` — charts with `Description(field)` lost `<desc>` SVG elements | `build_tooltips_and_hrefs` now returns a third `descriptions` vec; `MarkBuildResult` carries it; `MarkBatch` gains `descriptions: Option<Vec<Option<String>>>`; `svg_walk` emits `<desc>` when present |
| 2 | `FillStroke`-styled Path nodes (area/ribbon) don't carry `stroke_cap`/`stroke_join` — 11b WASM renderer would need to read batch-level field instead of node style | Documented as a known design seam; batch-level `stroke_cap`/`stroke_join` is the canonical source for all backends. Node-level `FillStroke` intentionally does not duplicate this. |
| 3 | Tooltip `name` field hardcoded to `"value"` instead of the encoding field name | `build_tooltips_and_hrefs` now accepts the field name from the encoding and uses it as `TooltipField.name` |
