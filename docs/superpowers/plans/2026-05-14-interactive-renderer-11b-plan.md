# Phase 11b — WASM Renderer Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Deliver the `ferrum-wasm` crate — a WASM-compiled GPU renderer that consumes the SceneGraph IR from 11a and renders to a browser `<canvas>`, plus CSS text overlays, self-contained `.save("chart.html")`, and `.save("chart.json")`. Static rendering only (no selections, no zoom/pan — those are 11c).

## 2. Spec references

- `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §5 — WASM renderer architecture, wgpu + WebGL2 fallback, SDF shaders, lyon tessellation, text as CSS overlays
- Same spec §8 — Error handling (`WasmRenderError` enum, no `unwrap()` in WASM crate)
- Same spec §10.5 — `save_chart` format dispatch (HTML, JSON)
- Same spec §11 — Packaging (wasm-pack build, maturin includes, wheel layout)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `crates/ferrum-wasm/` | New crate: Cargo.toml, lib.rs, error.rs, gpu.rs, pipelines.rs, scene_load.rs, tessellate.rs, render.rs, text.rs |
| Create | `crates/ferrum-wasm/src/shaders/*.wgsl` | SDF circle, SDF rect, mesh, textured quad shaders |
| Create | `src/ferrum/_html.py` | HTML template assembly for `save("chart.html")` |
| Create | `src/ferrum/_wasm/ferrum-interactive.js` | JS ESM glue: DOM bootstrap, canvas sizing, text overlay |
| Create | `src/ferrum/_wasm/ferrum-interactive.css` | Text overlay styling |
| Modify | `Cargo.toml` (workspace root) | Add `ferrum-wasm` to members; add wgpu, wasm-bindgen, web-sys, js-sys, lyon workspace deps |
| Modify | `src/ferrum/display.py` | Wire `"html"` and `"json"` formats in `save_chart` |
| Modify | `pyproject.toml` | Add `[tool.maturin] include` for WASM artifacts |
| Modify | `.gitignore` | Add `src/ferrum/_wasm/*.wasm`, `*.js` build artifacts |

## 4. Constraints

- `ferrum-wasm` targets `wasm32-unknown-unknown` only — no PyO3 dependency
- `crate-type = ["cdylib"]` required for wasm-pack
- `#[deny(clippy::unwrap_used)]` at crate level per spec §8.1
- `ferrum-scene` is the only shared dependency between `ferrum-core` and `ferrum-wasm` — no direct dependency on `ferrum-core`
- Text rendered as CSS overlays, not GPU-rasterized (spec §5.4)
- WebGL2 fallback required via `wgpu` `webgl` feature (spec §5.1)
- Byte-deterministic randomness constraint does not apply to WASM crate (no transforms)
- All existing golden SVGs and Python/Rust tests must continue to pass unchanged
- **One `WasmRenderer` per canvas** — no global singleton; each chart on the page gets its own instance
- **wgpu device config:** `Limits::downlevel_webgl2_defaults()`, `PowerPreference::LowPower`, `CompositeAlphaMode::PreMultiplied` (transparency compositing with CSS overlay)
- **Group flattening:** `Group` nodes recurse children at load time, discard group attrs (read `stroke_cap`/`stroke_join` from `MarkBatch` fields). `Raw` nodes: skip + console.warn (typed gradient representation deferred)
- **Draw order:** mesh (areas/lines/polygons) → rect (bars) → circle (points on top). Painter's algorithm, no depth/stencil buffer. Fully correct interleaved-batch z-order deferred to 11c
- **Pipeline blend/topology:** all four pipelines use alpha blending (`SrcAlpha, OneMinusSrcAlpha`); mesh = `TriangleList` (indexed, from lyon); circle/rect = `TriangleStrip` (instanced quads)
- **`ferrum-interactive.js` is hand-authored and committed** — it imports from wasm-pack-generated glue but is NOT itself generated. Only wasm-pack output (`ferrum_wasm.js`, `ferrum_wasm_bg.wasm`) is gitignored
- **Build order is load-bearing:** `wasm-pack build` must run before `maturin develop`, otherwise WASM files won't be included in the wheel
- **WASM inlining for single-file HTML:** wasm-pack's `init()` accepts a `BufferSource` (Uint8Array), so `init(wasmBytes)` works with base64-decoded bytes

## 5. Tasks

### Task 11b1: Crate scaffold + wgpu initialization
- [ ] Install `wasm-pack`, add `wasm32-unknown-unknown` target
- [ ] Create `ferrum-wasm` crate with Cargo.toml, lib.rs, error.rs, gpu.rs per spec §5.1, §8
- [ ] Add workspace deps (wgpu, wasm-bindgen, web-sys, js-sys, lyon, bytemuck)
- [ ] Implement `WasmRenderer` struct with `async init(canvas_id)` → wgpu device/queue
- [ ] Implement WebGL2 fallback per spec §5.1 (request adapter with `Backends::GL` if WebGPU unavailable)
- [ ] Verify: `wasm-pack build --target web crates/ferrum-wasm` compiles clean

### Task 11b2: Mark rendering pipelines — SDF shaders + lyon tessellation
- [ ] Create four WGSL shader pairs per spec §5.2: SDF circle, SDF rect, mesh (lyon), textured quad
- [ ] Implement `RenderPipelines` struct with `create_pipelines(device)` → all four pipelines
- [ ] Implement `scene_load.rs`: deserialize SceneGraph JSON, flatten Groups, build GPU vertex/index/instance buffers
- [ ] Implement `tessellate.rs`: lyon tessellation for Line, Path, Polygon, Polyline → triangle mesh
- [ ] Implement `render.rs`: per-frame draw calls dispatched by `MarkBatchKind`, clear color, viewport
- [ ] Verify: `WasmRenderer.loadScene(json).render()` draws geometric primitives to canvas

### Task 11b3: JS glue module + text overlay + HTML output
- [ ] Create `ferrum-interactive.js` ESM module: init WASM, create canvas, resize observer, render loop
- [ ] Create `ferrum-interactive.css`: text overlay positioning, font defaults
- [ ] Implement `text.rs`: extract Text/Label SceneNode positions → JS-consumable descriptors
- [ ] JS side: create positioned `<div>` elements for each text descriptor
- [ ] Create `_html.py`: assemble self-contained HTML (inline WASM base64, JS, CSS, SceneGraph JSON)
- [ ] Verify: generated HTML file opens in browser and renders a chart

### Task 11b4: Python save API + packaging
- [ ] Wire `"html"` and `"json"` formats in `display.py` `save_chart()`
- [ ] `"json"` format: write SceneGraph JSON via `render_interactive`
- [ ] `"html"` format: call `_html.assemble_html(scene_json)` → write file
- [ ] Configure `pyproject.toml` maturin include for `_wasm/` artifacts
- [ ] Add `.gitignore` entries for WASM build artifacts
- [ ] Verify: `chart.save("test.html")` and `chart.save("test.json")` produce valid output

## 6. Acceptance checks

- `wasm-pack build --target web crates/ferrum-wasm` — compiles without errors or warnings
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- `DYLD_LIBRARY_PATH=... cargo test` — all existing Rust tests pass
- `uv run pytest tests/ -x --timeout=120` — all existing Python tests pass
- Generated `chart.html` opens in Chrome/Firefox and renders the chart to canvas
- Generated `chart.json` contains valid SceneGraph JSON (round-trips through serde)
- No changes to existing golden SVGs

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
