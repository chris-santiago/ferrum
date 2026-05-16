# GPU-Native Zoom/Pan Design Spec

*2026-05-14 — closes the Python-round-trip gap in Phase 11*

---

## Problem

The current zoom implementation routes every wheel event through the Python kernel:

```
wheel → model.set('zoom_state') → Jupyter comm → Python kernel
→ render_interactive() (Rust) → new scene_json → Jupyter comm → JS GPU render
```

Round-trip latency is 100–400 ms per scroll event. Bokeh/Plotly/Altair maintain <16 ms zoom by applying a GPU affine transform to existing mark geometry — no re-render, no network round-trip.

The `ZoomPanState`, `PanelTickLevels`, and `Affine2` types are already implemented in `ferrum-wasm` but are not wired to the renderer or exposed to JS. The GPU shader has no transform uniform — mark positions are baked as pixel coords at load time.

---

## Contract

### Zoom behavior visible to users

- Wheel scroll on a Cartesian/Fixed panel: marks and axes transform **instantly** (target ≤1 render frame, ~16 ms).
- Tick labels swap from pre-computed sets at appropriate density for the current zoom level — no Python round-trip needed.
- A Python round-trip fires **only** after zoom/pan settles (400 ms debounce) for marks that require data-space recomputation (`mark_function`, `mark_raster`). For all other mark types (scatter, bar, line, density, hex, etc.) the round-trip is optional and may be omitted entirely.
- `CoordFixed` panels enforce uniform scale (sx = sy) as before.
- Pan (mousedown + drag) applies a translation transform; same GPU path as zoom.
- Reset (double-click) restores identity transform and original tick level.

### Python API surface — unchanged

`Chart.interactive()` returns `InteractiveChart`. No new Python-facing API is added. The `zoom_state` traitlet is retained for backward compat but is no longer the primary zoom path; it fires only on the debounced settle event.

### WASM API surface — new bindings

`WasmRenderer` gains three new JS-callable methods:

```rust
// Apply zoom transform and re-render. Returns updated text-element JSON
// (tick labels for the new zoom level) as a UTF-8 string.
pub fn on_wheel(panel_id: u32, delta_y: f32, cursor_x: f32, cursor_y: f32) -> Result<String, JsValue>;

// Apply pan delta and re-render. Returns updated text-element JSON.
pub fn on_pan(panel_id: u32, dx: f32, dy: f32) -> Result<String, JsValue>;

// Reset panel to identity transform. Returns text-element JSON at zoom=1.
pub fn reset_zoom(panel_id: u32) -> Result<String, JsValue>;
```

All three return the same text-element JSON that `loadScene` already returns, so the JS can call `_placeText(overlay, JSON.parse(result))` to update axis labels without touching the Python model.

### GPU uniform layout — extended

The existing `[f32; 4]` uniform (width, height, 0, 0) expands to `[f32; 16]`:

```
[canvas_w, canvas_h, sx, sy, tx, ty, clip_x, clip_y,
 clip_w, clip_h, 0, 0, 0, 0, 0, 0]
```

Positions 0–1: canvas dimensions (unchanged).
Positions 2–5: per-panel affine transform (sx, sy, tx, ty); identity = (1,1,0,0).
Positions 6–9: clip rectangle in canvas pixels for the current panel.

The vertex shader applies `pos_transformed = (pos * vec2(sx,sy)) + vec2(tx,ty)` before NDC conversion. All instanced pipelines (circle, rect, mesh) share this uniform layout.

Because the transform and clip rect change per panel within one render call, `render_frame` iterates panels, uploads the panel's uniform slice, draws the panel's marks, then advances to the next panel.

### Tick level selection

`PanelTickLevels` (already in `InteractionConfig` inside the scene JSON) carries pre-computed tick label sets at multiple zoom thresholds. `WasmRenderer` stores the deserialized `InteractionConfig` at `load_scene` time. On `on_wheel`, the zoom factor is computed from `ZoomPanState`, the matching `TickLevel` is selected, and its `TextElement` list is returned as JSON. The JS overlay replaces axis tick labels without touching the Python model.

### Debounced Python round-trip (optional)

The JS sets a 400 ms debounce timer on every wheel/pan event. When the timer fires, it reads the current domain from `ZoomPanState` (inverse-transformed from the panel's original domain), sends `zoom_state` to the Python model. Python rebuilds with the new domain and sends back a fresh `scene_json`. The renderer calls `loadScene` on the new scene, resetting the affine transform to identity (since the new scene already encodes the zoomed domain as the base coordinate system).

Charts whose marks do not require data recomputation (everything except `mark_function` and `mark_raster`) may skip the Python round-trip entirely by setting `zoom_recompute: false` in `InteractionConfig`. Default is `false`.

---

## Out of scope

- Multi-panel linked zoom (panels zoom together): deferred; `linked_panels` field exists but this spec does not wire it.
- Touch/pinch gestures: deferred.
- Polar and Geo coord zoom: no affine transform applies; wheel is a no-op on those panels.
- Animation easing on zoom step: no easing; each wheel tick is one discrete transform update.
