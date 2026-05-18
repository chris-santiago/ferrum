# D3 Interaction Layer Design Spec

## 1. Scope

Replace the hand-written mouse event handlers in `ferrum-anywidget.js` (~200 lines of mousedown/mousemove/mouseup/wheel/dblclick state machines) with D3-zoom and D3-brush. Replace the CSS-div text overlay (`_placeText`) with SVG `<text>` elements rendered via D3-selection in the same SVG layer as the brush. These modules are the industry standard for canvas interaction and text rendering, and eliminate two entire classes of bugs: interaction state machines (click-vs-drag, zero-size brush, coordinate drift, pan-vs-brush conflicts, zoom state sync) and text positioning (missing rotation, wrong baseline, legend overlap, font-weight dropped).

## 2. Goals

- Zoom, pan, double-click reset, and brush selection use D3's battle-tested implementations instead of hand-written event handlers.
- All text elements (axis labels, tick labels, titles, legends) render as SVG `<text>` with correct rotation, baseline, anchor, font-weight, and font-family — replacing the CSS-div `_placeText` approach.
- The HTML export remains self-contained (no CDN dependency at runtime).
- D3-zoom owns the zoom/pan transform; the WASM renderer receives it as an input.
- D3-brush owns the brush rectangle lifecycle; the WASM renderer receives the final brush extent.
- Tooltip hover remains custom (D3 has no tooltip module).
- All existing regression tests (R1–R6, P1–P13) continue to pass.
- Both Jupyter and standalone HTML paths use the same D3-based interaction code via the existing adapter pattern.

## 3. Non-goals

- Touch/mobile gesture support (D3 supports it natively, but WASM renderer doesn't — defer).
- Replacing the JS hit-test (`_hitTest`) with D3 — the hit-test operates on scene JSON nodes, not DOM.
- Replacing the tooltip hover logic — it's coupled to the WASM `hitTestAt`/`getTooltip` API.
- Changing static SVG/PNG rendering — the `show_svg()`/`show_png()` paths are unaffected. D3 text rendering only applies to the WASM interactive path (Jupyter and HTML export).

## 4. System behavior

### Zoom and pan

D3-zoom attaches to the canvas element. Mousewheel zooms, click-drag pans, double-click resets to identity. The D3 `zoom` event fires with a `transform { k, x, y }` — the WASM renderer receives this transform via a new `setTransform(k, tx, ty)` method that replaces the current incremental `onWheel`/`onPan`/`resetZoom` API. Text labels are repositioned after each transform change.

When an interval selection is active, D3-zoom's pan gesture is filtered out (brush takes priority for drag). Zoom via mousewheel still works. Pan requires Alt/Option key.

### Brush selection

D3-brush attaches to a transparent SVG overlay positioned on top of the canvas (D3-brush requires SVG). Drag creates a visible brush rectangle styled from the `SelectionMark` spec. On brush end, the brush extent `[[x0, y0], [x1, y1]]` is forwarded to the WASM renderer's `handleDrag`. D3-brush natively handles click-vs-drag (single clicks produce no selection), minimum extent, brush move/resize handles, and clearing.

When no interval selection is declared, the SVG brush overlay is not created. D3-zoom handles all drag events (pan).

### Text rendering

All text returned by `loadScene` (axis tick labels, axis titles, chart title, legend text) is rendered as SVG `<text>` elements in the SVG overlay via D3-selection. This replaces the `_placeText` CSS-div approach. SVG `<text>` natively handles:

- **`text-anchor`**: `start` / `middle` / `end` — replaces the CSS `translateX(-50%)` hack.
- **`dominant-baseline`**: `auto` / `central` / `hanging` — replaces the CSS `translateY(-85%)` hack.
- **`transform="rotate(...)"`**: proper rotation around the text origin — replaces the broken CSS `rotate()` that didn't compose correctly with anchor/baseline transforms.
- **`font-weight`**: `normal` / `bold` / `600` etc. — preserves theme font weights that were being dropped.
- **`font-family`**: inherits from the SVG's `@font-face` — Inter renders correctly.

On zoom, `loadScene` is not re-called. Instead, `setTransform` returns updated text element JSON with repositioned tick labels. The SVG text elements are re-rendered via D3's data-join pattern (`selectAll('text').data(texts).join('text')`), which efficiently updates positions without full DOM replacement.

### Tooltip hover

Unchanged. `mousemove` on the canvas fires the existing JS hit-test + WASM `hitTestAt` fallback. Tooltip div is positioned in CSS coordinates. This is the only remaining hand-written mouse handler.

### Click (point selection + href)

Unchanged. `click` on the canvas fires WASM `handleClick` for point selections or opens hrefs. Gated on `_hasPointSelections`.

## 5. Architecture

### Layer stack (top to bottom)

1. **Tooltip div** — `position:absolute`, `pointer-events:none`, `z-index:20`.
2. **SVG overlay** — `position:absolute`, same dimensions as canvas, `pointer-events:none` (except during brush). Hosts: D3-brush group (when `hasInterval`), all text elements (axis labels, tick labels, titles, legends). Always created.
3. **Canvas** — WebGPU rendering surface. D3-zoom attaches here. Click and mousemove listeners also here.

### D3-zoom → WASM bridge

D3-zoom produces a transform `{ k, x, y }` where `k` is scale factor and `(x, y)` is translation. The WASM renderer receives this via a new method `setTransform(k, tx, ty)` which replaces the internal `ZoomPanState` and re-renders. The JS no longer maintains a parallel `_zoom = { sx, sy, tx, ty }` — D3's transform is the single source of truth. The `_invZoom` function reads from `d3.zoomTransform(canvas)` instead of the manual `_zoom` object.

### D3-brush → WASM bridge

D3-brush's `end` event provides `selection: [[x0, y0], [x1, y1]]` or `null` (cleared). When non-null, forward to `renderer.handleDrag(panelId, x0, y0, x1, y1)`. When null (brush cleared by clicking background), clear the interval selection state.

### SVG text rendering

`loadScene` returns a JSON array of text elements with `{ x, y, content, fontSize, fontWeight, fontFamily, color, anchor, baseline, angle }`. A `_placeTextSvg(svgLayer, texts)` function uses D3-selection's data-join to render these as SVG `<text>` elements:

```javascript
d3.select(svgLayer).selectAll('text.ferrum-label')
  .data(texts, (d, i) => i)
  .join('text')
  .attr('class', 'ferrum-label')
  .attr('x', d => d.x)
  .attr('y', d => d.y)
  .attr('text-anchor', d => d.anchor === 'center' ? 'middle' : d.anchor)
  .attr('dominant-baseline', d => mapBaseline(d.baseline))
  .attr('transform', d => d.angle ? `rotate(${d.angle}, ${d.x}, ${d.y})` : null)
  .attr('font-size', d => d.fontSize)
  .attr('font-weight', d => d.fontWeight)
  .attr('font-family', d => d.fontFamily)
  .attr('fill', d => d.color)
  .text(d => d.content);
```

On zoom, `setTransform` returns updated text JSON. The same data-join is called with the new positions — D3 updates only the changed attributes, avoiding full DOM replacement.

### Self-contained bundling

D3 modules (`d3-brush`, `d3-zoom`, `d3-selection`, `d3-dispatch`, `d3-interpolate`, `d3-transition`, `d3-drag`, `d3-timer`, `d3-ease`, `d3-color`) are vendored as a single pre-built ESM bundle in `src/ferrum/_wasm/d3-interactions.js`. This file is created once (via a build script or manual download from esm.sh) and checked into the repo. It is inlined into the HTML export alongside the WASM glue and anywidget JS. For Jupyter, it is prepended to the anywidget ESM string.

## 6. Canonical interfaces / data contracts

### New WASM method: `setTransform`

```rust
#[wasm_bindgen(js_name = "setTransform")]
pub fn set_transform(&mut self, k: f32, tx: f32, ty: f32) -> Result<String, JsValue>
```

Replaces the internal `ZoomPanState` with the given affine transform (uniform scale `k`, translation `(tx, ty)`). Rebuilds GPU uniforms and re-renders. Returns text element JSON (tick labels at new positions). This replaces the incremental `onWheel`/`onPan`/`resetZoom` methods — those remain for backward compatibility but `setTransform` is the canonical entry point.

### D3-zoom configuration

```javascript
d3.zoom()
  .scaleExtent([0.1, 50])
  .filter(event => {
    // Allow wheel always. Allow drag only when no interval selection or Alt held.
    if (event.type === 'wheel') return true;
    if (hasInterval && !event.altKey) return false;
    return !event.button; // left button only
  })
  .on('zoom', event => {
    const { k, x, y } = event.transform;
    renderer.setTransform(k, x, y);
  })
```

### D3-brush configuration

```javascript
d3.brush()
  .extent([[plotArea.x, plotArea.y], [plotArea.x + plotArea.w, plotArea.y + plotArea.h]])
  .on('end', event => {
    if (event.selection) {
      const [[x0, y0], [x1, y1]] = event.selection;
      renderer.handleDrag(panelId, x0, y0, x1, y1);
    }
  })
```

The brush extent is constrained to the panel's `plot_area` rectangle, not the full canvas. This prevents brushing over axis labels and legends.

### Adapter interface (unchanged)

The adapter pattern from the previous spec is unchanged. D3 event handlers call `adapter.onSelectionChange()` and `adapter.onZoomChange()` as before.

## 7. Invariants and constraints

- **D3 modules vendored, not CDN.** The HTML export is self-contained. No runtime network requests.
- **D3-zoom is the single source of zoom state.** No parallel `_zoom` object in JS. The WASM `ZoomPanState` is set (not accumulated) from D3's transform.
- **D3-brush requires SVG.** The brush overlay is an `<svg>` element, not a `<div>`. It must have the same pixel dimensions as the canvas.
- **Tooltip and click handlers remain on canvas.** D3-zoom's `filter` excludes drag events when `hasInterval` is true, so the SVG overlay captures drags (brush) while the canvas captures clicks, wheel, and mousemove.
- **All existing regression tests pass.** R1–R6 (Rust), P1–P13 (Python). No changes to the Python or Rust layers — only JS and the new `setTransform` WASM method.
- **`onWheel`/`onPan`/`resetZoom` remain available.** They are not removed — Jupyter's Python-side zoom rebuild still calls them. But the JS interaction layer uses `setTransform` exclusively.
- **No new Python dependencies.** D3 is JS-only.
- **Inter font embedded.** The `@font-face` block for Inter is declared in the SVG's `<defs><style>` (not CSS — SVG has its own style scope). This ensures SVG `<text>` elements use Inter.
- **`_placeText` and `ferrum-overlay` div are removed.** All text rendering moves to the SVG layer. No CSS-div text elements remain. The `ferrum-text` CSS class is removed.
- **Static rendering unaffected.** `show_svg()` and `show_png()` use the Rust SVG pipeline and are not touched by this change.

## 8. Key decisions and tradeoffs

**D3-zoom `setTransform` vs. incremental `onWheel`/`onPan`.** The current WASM API uses incremental updates: `onWheel(panel, deltaY, cx, cy)` mutates internal state. D3-zoom produces an absolute transform. Rather than decomposing D3's transform into incremental deltas (lossy, error-prone), we add `setTransform(k, tx, ty)` which sets the WASM zoom state absolutely. This eliminates the JS↔WASM zoom state drift that caused several bugs on the abandoned branch.

**Vendored bundle vs. CDN import.** CDN (esm.sh / jsdelivr) is simpler and always up-to-date, but breaks the self-contained HTML guarantee. Vendoring adds ~40KB to the repo but preserves offline operation. Vendoring wins — ferrum already inlines a 3.6MB WASM binary.

**SVG overlay for brush vs. CSS div brush.** D3-brush requires SVG — it creates `<rect>` elements for the selection and handles. The current implementation uses a CSS `<div>` for the brush rectangle. Switching to SVG gives us D3-brush's full feature set (resize handles, move, clear-on-click) at the cost of an additional DOM layer. The SVG overlay has `pointer-events: all` only during brush interactions, otherwise `none`.

**Brush constrained to plot area.** D3-brush's `.extent()` is set to the panel's `plot_area` rectangle, not the full canvas. This prevents users from brushing over axis labels and ensures brush coordinates map directly to data space. The current implementation allows brushing anywhere on the canvas — this is a behavior improvement.

**SVG `<text>` vs. CSS-positioned `<div>` for text.** The current `_placeText` uses absolutely-positioned `<div>` elements with CSS transforms for rotation and baseline. This approach has produced five distinct bugs: missing rotation, wrong baseline, legend overlap, font-weight dropped, and title missing. SVG `<text>` handles `text-anchor`, `dominant-baseline`, `transform="rotate()"`, and `font-weight` natively — these are core SVG features, not CSS workarounds. The SVG overlay already exists for D3-brush, so text adds no new DOM layers. D3-selection's data-join (`selectAll.data.join`) provides efficient incremental updates during zoom without full DOM replacement.

**Unified SVG layer.** Rather than separate layers for brush and text, both share one `<svg>` element. Brush elements go in a `<g class="brush">` group; text elements are siblings. This simplifies z-ordering and ensures consistent coordinate space.

## 9. Acceptance criteria

1. All existing regression tests pass (R1–R6, P1–P13).
2. Zoom via mousewheel works in HTML export and Jupyter.
3. Pan via click-drag works (no interval selection) or Alt+drag (with interval selection).
4. Double-click resets zoom to identity.
5. Brush selection creates a visible rectangle, highlights marks inside via conditional encoding, dims marks outside.
6. Click on canvas background with brush active does NOT grey out marks (D3-brush clears selection cleanly).
7. Point selection via click works (conditional encoding updates).
8. Tooltip hover works on both packed and non-packed mark batches.
9. HTML export is self-contained (no network requests).
10. Composition HTML exports (HConcat, VConcat) render with correct layout.
11. Axis titles ("x", "y") render with correct font-weight and position.
12. Y-axis tick labels are rotated 90° (or per-theme angle).
13. Legend swatches and labels render at correct positions — no overlap with the plot area.
14. Chart title renders above the plot area when `.properties(title="...")` is set.
15. Text uses Inter font-family (loaded via `@font-face` in SVG `<defs>`).

## 10. Validation strategy

### Existing tests (must still pass)

- **Tier 1 (Rust):** `cargo test -p ferrum-wasm` — R1–R6 plus the new `setTransform` tests.
- **Tier 2 (Python):** `pytest tests/test_html_export_regression.py` — P1–P13.
- **Full suite:** `pytest -n auto` — no regressions.

### New tests

- **Rust:** `setTransform` sets zoom state correctly; subsequent `hit_test_at` uses the new transform; text elements are repositioned.
- **Python P14:** Generated HTML contains vendored D3 module (assert `d3.brush` or `d3.zoom` appears in HTML source).
- **Python P15:** Generated HTML does NOT contain hand-written `_panStart`, `_brushOrigin`, or `_isBrushing` variables (assert absence — confirms old code is removed).
- **Python P16:** Generated HTML does NOT contain `ferrum-overlay` div or `_placeText` function (assert absence — confirms CSS-div text rendering is removed).
- **Python P17:** Generated HTML contains `<svg` element and `text.ferrum-label` class (confirms SVG text rendering is present).

### Browser smoke tests (S1–S11 from previous spec, plus text-specific checks)

- **S12.** Y-axis tick labels are rotated (not horizontal).
- **S13.** Legend text and swatches render to the right of the plot area without overlapping marks.
- **S14.** Chart title renders above the plot area (when set).
- **S15.** All text uses Inter font (compare to SVG output of same chart).

Manual verification of all 8 export files before merge.
