# GPU-Native Zoom/Pan Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement task-by-task.

**Goal:** Replace Python-kernel-round-trip zoom with GPU affine transforms so wheel events render in ≤16 ms.

**Architecture:** Extend the WASM uniform buffer to carry a per-panel `(sx, sy, tx, ty)` transform; vertex shaders apply it before NDC conversion. `WasmRenderer` holds a `ZoomPanState`, exposes `on_wheel`/`on_pan`/`reset_zoom` bindings. The JS wheel handler calls these instead of `model.set('zoom_state')`. Axis tick labels swap from pre-computed `PanelTickLevels` stored in the scene JSON.

**Spec:** `docs/superpowers/specs/2026-05-14-gpu-zoom-pan-design.md`

**Tech stack:** Rust/wgpu (WASM), wasm-bindgen, anywidget ESM (JS string in `_interactive.py`), Python

**Build commands:**
- Rebuild WASM: `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`
- Rebuild Python ext: `unset CONDA_PREFIX && uv run --no-sync maturin develop`
- Tests: `uv run pytest tests/ -q --ignore=tests/test_bug_hunt_*.py`

---

### Task 1 — Extend GPU uniform buffer and vertex shaders

**Files:**
- Modify: `crates/ferrum-wasm/src/render.rs` — add transform+clip to uniform struct and upload logic
- Modify: `crates/ferrum-wasm/src/pipelines.rs` — increase uniform buffer size
- Modify: `crates/ferrum-wasm/src/shaders/` (all `.wgsl` files) — apply transform in vertex stage

- [ ] Read `crates/ferrum-wasm/src/render.rs` and find `let viewport: [f32; 4]`. Change to `[f32; 16]` containing `[canvas_w, canvas_h, sx, sy, tx, ty, clip_x, clip_y, clip_w, clip_h, 0,0,0,0,0,0]`; initialize identity: `sx=1.0, sy=1.0, tx=0.0, ty=0.0`, clip = full canvas.

- [ ] In all `.wgsl` vertex shaders, add `struct Uniforms { canvas: vec2<f32>, sx: f32, sy: f32, tx: f32, ty: f32, clip_x: f32, clip_y: f32, clip_w: f32, clip_h: f32, _pad: array<f32, 6> }` and apply transform before NDC: `let tp = vec2(pos.x * uniforms.sx + uniforms.tx, pos.y * uniforms.sy + uniforms.ty);`

- [ ] Run `cargo check -p ferrum-wasm --target wasm32-unknown-unknown` — zero new errors.

- [ ] Run `wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`. Spot-check that `uv run python -c "import ferrum; import polars as pl; fm=ferrum; c=fm.Chart(pl.DataFrame({'x':[1.0,2.0],'y':[3.0,4.0]})).mark_point().encode(x='x:Q',y='y:Q').interactive(); print(type(c))"` produces `<class 'ferrum._interactive.InteractiveChart'>`.

- [ ] Commit: `fix(wasm): extend uniform buffer for per-panel zoom transform`

---

### Task 2 — Wire ZoomPanState into WasmRenderer

**Files:**
- Modify: `crates/ferrum-wasm/src/lib.rs` — add `zoom: ZoomPanState`, `tick_levels: Vec<PanelTickLevels>` fields; expose `on_wheel`, `on_pan`, `reset_zoom` as `#[wasm_bindgen]` methods
- Modify: `crates/ferrum-wasm/src/render.rs` — add `render_with_transform(panel_id, transform)` that uploads uniform and draws only that panel's mark batches

- [ ] In `lib.rs`, add to `WasmRenderer`: `zoom: crate::zoom_pan::ZoomPanState`, `interaction: ferrum_scene::InteractionConfig`. Initialize both in `create()` from an empty/default state.

- [ ] In `load_scene()`, deserialize `scene.interaction` into `self.interaction` and initialize `self.zoom = ZoomPanState::new(scene.panels.len(), &self.interaction)`.

- [ ] Implement `build_text_json_for_level(panel_id, zoom_factor)` helper that selects the matching `TickLevel` from `self.interaction.tick_levels[panel_id]` and returns a `TextElement` list JSON string.

- [ ] Implement `#[wasm_bindgen(js_name = "onWheel")] pub fn on_wheel(&mut self, panel_id: u32, delta_y: f32, cx: f32, cy: f32) -> Result<String, JsValue>`:
  - Call `self.zoom.on_wheel(panel_id as usize, delta_y, ScaleMode::Independent)` (use existing signature)
  - Retrieve `Affine2` transform for the panel
  - Upload transform uniform to GPU (call `render_frame` or a new `render_with_transforms`)
  - Return `build_text_json_for_level(panel_id, self.zoom.transforms[panel_id as usize].zoom_factor())`

- [ ] Implement `on_pan` and `reset_zoom` with the same pattern.

- [ ] `cargo check -p ferrum-wasm --target wasm32-unknown-unknown` — zero errors.

- [ ] Rebuild WASM. Commit: `feat(wasm): wire ZoomPanState into WasmRenderer, expose onWheel/onPan/resetZoom`

---

### Task 3 — Update JS zoom handler to use WASM bindings

**Files:**
- Modify: `src/ferrum/_wasm/ferrum-anywidget.js` — wheel event listener inside `_reload()`
  (JS was extracted from `_interactive.py` to this real source file after the plan was written)

- [ ] In `_render()`, after `renderer = await WasmRenderer.create(canvas)`, add: `let _zoomDebounceId = null;`

- [ ] Replace the `wheel` event listener (currently inside `_reload`, sets `model.set('zoom_state', ...)`) with:
```js
_state.canvas.addEventListener('wheel', e => {
  e.preventDefault();
  if(!_state || !_state.renderer) return;
  const r = _state.canvas.getBoundingClientRect();
  try {
    const textJson = _state.renderer.onWheel(0, e.deltaY, e.clientX - r.left, e.clientY - r.top);
    _placeText(ov, JSON.parse(textJson));
  } catch(err) { /* GPU not ready */ }
  // Debounced Python round-trip for mark_function/mark_raster recompute
  clearTimeout(_zoomDebounceId);
  _zoomDebounceId = setTimeout(() => {
    const sc = _state.scene;
    const p = sc.panels && sc.panels[0];
    if(!p || !p.coord) return;
    const xs = p.coord.x_domain, ys = p.coord.y_domain;
    if(!xs || !ys) return;
    model.set('zoom_state', JSON.stringify({'0': {x_domain: xs, y_domain: ys}}));
    model.save_changes();
  }, 400);
}, {passive: false});
```

Note: the debounce still sends `zoom_state` for charts that opted in to recomputation. For the common case the Python side receives it, rebuilds, sends new `scene_json`, and the renderer calls `loadScene` resetting the transform to identity.

- [ ] Run `uv run python -c "import ferrum._interactive; print('OK')"` — must print OK.

- [ ] Commit: `feat(interactive): use WASM onWheel binding for GPU-native zoom`

---

### Task 4 — Regression tests ✅ ALREADY DONE

**Skipped:** 14 regression tests covering tooltip, zoom domain, selection, arc nodes, and
`merge_scene_graphs` offsets were committed in `tests/test_interactive_regression.py` as
`test(interactive): 14 regression tests for tooltip, zoom domain, selection, arc`
(commit `4411bbc`). No new test file needed.

After completing Task 3, verify the existing suite still passes:

- [ ] Run `uv run pytest tests/test_interactive_regression.py -v` — all 14 pass.

- [ ] No commit needed for this task.

---

### Task 5 — End-to-end notebook check

- [ ] Open `notebooks/interactive_demo.ipynb`, restart kernel, run all cells. Verify:
  - Charts 1–4 (scatter): tooltip appears on hover; scroll zooms without perceptible lag.
  - Chart 5 (bar): tooltip shows category + value on hover.
  - Charts 7–8 (pie/donut): tooltip shows wedge label on hover.
  - Charts 3–4 (selection): click dims non-matching marks.
  - No `[ferrum] GPU init failed` warnings in browser console for the first 8 charts.

- [ ] Commit: `chore: close Phase 11 GPU zoom gap — interactive_demo verified`
