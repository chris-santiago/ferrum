# R-tree Spatial Indexing & Interactive Toolbar

**Date:** 2026-05-21
**Status:** Draft
**Scope:** Two features for the interactive (WASM) rendering path — R-tree spatial indexing for O(log n) hit-testing, and a Bokeh-style toolbar for visible mode switching.

---

## 1. Problem

Ferrum's interactive renderer has two UX gaps:

1. **Hit-testing is O(n).** `hit_test.rs` linearly scans every mark in every batch on every mousemove. This is imperceptible at 1k marks but degrades at 50k+ — hover tooltips become janky, interval selections lag, and nearest-neighbor queries stall the frame.

2. **Interactions are invisible.** Pan requires Alt/Cmd+drag. Reset requires double-click. There is no on-screen indication of which mode is active or what gestures are available. Users must read source code to discover these interactions.

## 2. Solution Overview

**R-tree:** Add an `rstar`-based R*-tree spatial index to the WASM crate. Built once per scene load, queried on every hover/click/drag. Indexes circle and rect marks only (the high-count types). Lines, paths, polygons, text, and images keep the existing linear scan.

**Toolbar:** A right-side vertical strip of SVG icon buttons rendered in JS. Five tools: Pan, Box Zoom, Box Select (mutually exclusive drag modes), Reset, Save PNG (action buttons). Wheel zoom and hover tooltips are always on. Mode state lives entirely in JS — Rust/WASM doesn't need to know which tool is active.

**Bundled optimizations:** Remove the redundant JS-side `_hitTest()`, coalesce mousemove via RAF, add cursor feedback per mode, add keyboard shortcuts, save PNG at device pixel ratio.

## 3. R-tree Spatial Index

### 3.1 Dependency

`rstar = "0.12"` added to `crates/ferrum-wasm/Cargo.toml`. Compiles to WASM, no-std compatible, ~15KB WASM size increase.

### 3.2 Data Structure

New file: `crates/ferrum-wasm/src/spatial_index.rs`.

```
SpatialIndex {
    trees: Vec<PanelIndex>,       // one per panel
}

PanelIndex {
    tree: RTree<MarkEntry>,       // rstar R*-tree
}

MarkEntry {
    point: [f64; 2],              // center (circle center or rect center)
    batch_idx: usize,
    node_idx: usize,
    data_idx: Option<usize>,
    radius: f64,                  // circle: r, rect: 0.0 (use aabb)
    aabb: AABB<[f64; 2]>,         // bounding box
}
```

`MarkEntry` implements `rstar::RTreeObject` with `envelope()` returning the AABB, and `rstar::PointDistance` for nearest-neighbor queries with accurate distance (Euclidean for circles, AABB-edge for rects).

### 3.3 Build

Built during `WasmRenderer::loadScene()`, after `load_scene_with_packed()` returns. For each panel, iterates mark batches and collects entries for `Circle` and `Rect` nodes (batch kinds: `Point`, `Bar`, `Rect`). Built via `RTree::bulk_load()` — O(n log n), faster than repeated insert.

The index is always built regardless of mark count. Sub-millisecond for small charts, ~5-10ms for 100k marks.

### 3.4 Query Paths

Four query paths accelerated:

**Nearest-neighbor hover** (`hit_test_nearest`): `tree.nearest_neighbor(query_point)` replaces linear scan for circles/rects. Non-indexed mark types (lines, paths, etc.) in the same panel still use linear scan. Returns the closer of the two results.

**Exact hit-test** (`hit_test`): `tree.locate_in_envelope(&tolerance_aabb)` around the cursor (small envelope matching current tolerance), then existing geometry checks (circle radius test, rect bounds test) on candidates only.

**Interval selection** (`handleDrag` / `selectInRect`): `tree.locate_in_envelope(&drag_aabb)` returns all marks within the brush rectangle. Replaces linear containment check over all marks in the panel. O(log n + k) where k = selected marks.

**Snap-distance threshold**: If the nearest mark is >50 scene-space pixels from the cursor (measured in the pre-zoom coordinate system, not visual pixels), return `None`. Prevents showing a tooltip for a mark on the opposite side of the plot. The R-tree nearest-neighbor query returns distance for free — this is a comparison, not an additional query. The 50px value is a constant in `spatial_index.rs`, not configurable from Python.

### 3.5 Zoom/Pan Invariant

The R-tree stores marks in scene-space and is never rebuilt on zoom/pan. The cursor is inverse-transformed from visual-space to scene-space before every query, using the existing `ZoomPanState::inverse_apply()` pattern. This matches the current `hit_test.rs` approach exactly.

## 4. Interactive Toolbar

### 4.1 Tool Set

Three mutually exclusive drag modes:

| Mode | Drag behavior | D3 behavior | Cursor |
|---|---|---|---|
| Pan | Translate the view | d3-zoom | `grab` / `grabbing` |
| Box Zoom | Drag rectangle → zoom to fit | d3-brush → zoomBehavior.transform | `zoom-in` / `crosshair` |
| Box Select | Drag rectangle → select marks | d3-brush → handleDrag/selectInRect | `crosshair` |

Two action buttons:

| Button | Action |
|---|---|
| Reset | `zoomBehavior.transform(zoomIdentity)` — resets view |
| Save PNG | Off-screen canvas at `devicePixelRatio` resolution → `canvas.toBlob()` → download |

Always-on (not modes): wheel zoom, hover tooltips.

### 4.2 Placement

Right-side vertical strip alongside the canvas container. Positioned via CSS flex. The toolbar is a sibling `<div>` of the canvas, not an overlay — it does not obscure the chart.

### 4.3 DOM Structure

```html
<div class="ferrum-container" data-mode="pan" tabindex="0">
  <div style="position:relative">
    <canvas></canvas>
    <svg><!-- text overlay --></svg>
    <div class="ferrum-tooltip">...</div>
  </div>
  <div class="ferrum-toolbar">
    <button class="ferrum-tool active" data-mode="pan" title="Pan (P)">
      <svg><!-- icon --></svg>
    </button>
    <button class="ferrum-tool" data-mode="boxzoom" title="Box Zoom (Z)">
      <svg><!-- icon --></svg>
    </button>
    <button class="ferrum-tool" data-mode="select" title="Box Select (S)">
      <svg><!-- icon --></svg>
    </button>
    <div class="ferrum-tool-separator"></div>
    <button class="ferrum-tool" data-action="reset" title="Reset (R)">
      <svg><!-- icon --></svg>
    </button>
    <button class="ferrum-tool" data-action="save" title="Save PNG">
      <svg><!-- icon --></svg>
    </button>
  </div>
</div>
```

Icons are inline SVG — no external assets.

### 4.4 Mode Switching Mechanism

A `currentMode` variable in the `_render()` closure. The existing d3-zoom and d3-brush `.filter()` functions check this variable:

- **Pan mode:** d3-zoom allows drag (any button), d3-brush filter rejects drag.
- **Select mode:** d3-brush allows drag (left button), d3-zoom filter rejects drag but allows wheel.
- **Box Zoom mode:** d3-brush allows drag (left button), but its `on('end')` handler computes the zoom transform for the selected rectangle (via `zoomBehavior.transform`) instead of calling `handleDrag`.

Button clicks set `currentMode`, toggle `.active` class, and set `container.dataset.mode` for CSS cursor rules.

**Default mode:** `'pan'` when the chart has no selections declared. `'select'` when the chart has interval selections (preserving current implicit behavior).

### 4.5 Keyboard Shortcuts

Keydown listener on the container element (`tabindex="0"`). Active only when container has focus.

| Key | Action |
|---|---|
| `p` | Pan mode |
| `z` | Box Zoom mode |
| `s` | Box Select mode |
| `r` | Reset view |
| `Escape` | Return to default mode |

### 4.6 `toolbar=False` Escape Hatch

`Chart.interactive(toolbar=False)` serializes `"toolbar": false` in `InteractionConfig` JSON. JS checks this flag and skips toolbar DOM creation. All modifier-key interactions (Alt+drag = pan, double-click = reset) continue to work as today.

Default is `toolbar=True`.

## 5. Bundled Optimizations

### 5.1 Remove JS-side `_hitTest()`

Delete the `_hitTest()` function and `marks` array construction from `ferrum-anywidget.js`. All hover and click hit-testing routes through `renderer.hitTestAt()`. Href lookup moves to a new WASM method `getHref(panel, batch, idx)`.

### 5.2 RAF-coalesced Mousemove

Replace direct `mousemove` listener with a `requestAnimationFrame`-gated handler. Caps hit-test queries at the display refresh rate (typically 60/sec).

```
let _rafId = null, _pendingMove = null;
canvas.addEventListener('mousemove', e => {
    _pendingMove = e;
    if (!_rafId) _rafId = requestAnimationFrame(() => {
        _rafId = null;
        if (_pendingMove) handleHover(_pendingMove);
    });
});
```

### 5.3 Cursor Feedback

CSS rules on `.ferrum-container[data-mode="..."]` set the cursor. Toggled by setting `container.dataset.mode` on mode switch.

### 5.4 Save at Device Pixel Ratio

Save PNG creates a temporary off-screen canvas at `w * devicePixelRatio × h * devicePixelRatio`, re-renders via `renderer.resize()` + `renderer.render_frame_js()`, captures the blob, then restores the original canvas size. Download filename: `ferrum-chart.png`.

## 6. WASM API Changes

Two new methods on `WasmRenderer`:

| Method | Signature | Purpose |
|---|---|---|
| `getHref` | `fn get_href(&self, panel_id: u32, batch_idx: u32, node_idx: u32) -> String` | Returns href string for a hit mark, or empty string. Replaces JS-side href lookup. |
| `selectInRect` | `fn select_in_rect(&mut self, panel_id: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> String` | R-tree-accelerated interval selection. Returns selection state JSON. |

Existing methods unchanged.

## 7. Python API Changes

One new keyword argument:

```python
def interactive(self, *, toolbar: bool = True) -> InteractiveChart:
```

`InteractionConfig` in `crates/ferrum-scene/src/lib.rs` gains `toolbar: bool` with `#[serde(default = "default_true")]`.

No other Python-side changes.

## 8. File Inventory

| File | Change |
|---|---|
| `crates/ferrum-wasm/Cargo.toml` | Add `rstar` dependency |
| `crates/ferrum-wasm/src/spatial_index.rs` | **New** — SpatialIndex, PanelIndex, MarkEntry, build + query |
| `crates/ferrum-wasm/src/hit_test.rs` | Refactor to use SpatialIndex for circles/rects |
| `crates/ferrum-wasm/src/lib.rs` | Wire SpatialIndex into WasmRenderer, add `getHref`, `selectInRect` |
| `crates/ferrum-wasm/src/selection_state.rs` | Internal delegation to R-tree for interval containment |
| `crates/ferrum-scene/src/lib.rs` | Add `toolbar: bool` to InteractionConfig |
| `src/ferrum/_wasm/ferrum-anywidget.js` | Remove `_hitTest`, add toolbar DOM, mode switching, keyboard shortcuts, RAF hover, cursor classes |
| `src/ferrum/_wasm/ferrum-interactive.css` | Toolbar styling, cursor rules, active-tool indicator |
| `src/ferrum/_html.py` | Pass toolbar flag through interaction config |
| `src/ferrum/_interactive.py` | `toolbar` kwarg on `interactive()` |

## 9. Testing

**Rust unit tests** (`cargo test`):
- `spatial_index.rs` — bulk_load with 0, 1, 100, 100k marks. Nearest-neighbor correctness. Envelope query correctness. Snap-distance threshold. Verify results match linear scan for identical inputs.
- `hit_test.rs` — existing tests pass unchanged. New tests verify R-tree path produces identical results to old linear path on reference scenes.
- `selection_state.rs` — interval selection with R-tree containment produces same results as linear scan.

**Python integration tests**:
- `interactive(toolbar=False)` produces HTML without `.ferrum-toolbar` div.
- `interactive()` produces HTML with `.ferrum-toolbar` div.
- InteractionConfig JSON round-trip includes `"toolbar": true` by default.

**Manual verification**:
- Build WASM, render interactive scatter with 1k points. Verify all 5 toolbar buttons work. Mode switching changes cursor and drag behavior. Keyboard shortcuts work when container focused.
- Render 50k+ point scatter. Verify hover tooltips are responsive. Interval selection drag is snappy.
- Save PNG at retina resolution. Verify downloaded file is 2x dimensions.
- `toolbar=False` hides toolbar, modifier-key interactions still work.

## 10. Non-Goals

- Lasso (freeform) selection — not in scope, can be added as a future tool.
- Streaming/server-push — different product category (dashboard framework).
- Custom JS callbacks or widget framework — Bokeh/Panel territory.
- R-tree for line/path/polygon marks — elongated AABBs create excessive false positives; these types rarely exceed hundreds of elements per batch.
- Configurable tool set via Python API — fixed toolbar with `toolbar=False` escape hatch is sufficient. Configurability can be added later if needed.
