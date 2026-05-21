# R-tree Spatial Indexing & Interactive Toolbar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Add R*-tree spatial indexing for O(log n) hit-testing and a Bokeh-style interactive toolbar to ferrum's WASM renderer, plus five bundled optimizations (JS hit-test removal, RAF hover, cursor feedback, keyboard shortcuts, retina save).

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-21-rtree-toolbar-design.md` — all sections

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-wasm/Cargo.toml` | Add `rstar` dependency |
| Create | `crates/ferrum-wasm/src/spatial_index.rs` | SpatialIndex, PanelIndex, MarkEntry + build/query |
| Modify | `crates/ferrum-wasm/src/hit_test.rs` | Route circle/rect queries through R-tree |
| Modify | `crates/ferrum-wasm/src/lib.rs` | Wire SpatialIndex, add `getHref`, `selectInRect`, declare `mod spatial_index` |
| Modify | `crates/ferrum-wasm/src/selection_state.rs` | Delegate interval containment to R-tree |
| Modify | `crates/ferrum-scene/src/selection.rs` | Add `toolbar: bool` to InteractionConfig |
| Modify | `src/ferrum/chart.py` | `toolbar` kwarg on `interactive()` |
| Modify | `src/ferrum/composition.py` | Forward `toolbar` kwarg in `_ChartLike.interactive()` |
| Modify | `src/ferrum/_interactive.py` | Accept and serialize `toolbar` kwarg |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | Remove `_hitTest`, add toolbar DOM, mode switching, RAF hover, keyboard shortcuts |
| Modify | `src/ferrum/_wasm/ferrum-interactive.css` | Toolbar styling, cursor rules |
| Modify | `src/ferrum/_html.py` | Pass toolbar flag in interaction config |
| Test | `crates/ferrum-wasm/src/spatial_index.rs` (inline) | R-tree build/query correctness |
| Test | `tests/test_interactive_toolbar.py` | Toolbar presence in HTML, toolbar=False |

## 4. Constraints

- `rstar` must compile under `--target wasm32-unknown-unknown` — verify with `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown`
- R-tree is scene-space only, never rebuilt on zoom. Cursor inverse-transformed before query (spec §3.5)
- Only Circle and Rect nodes indexed — all other mark types keep linear scan (spec §3.3)
- Existing `hit_test.rs` unit tests must pass unchanged — R-tree produces identical results to linear scan
- Toolbar state is JS-only — no new Rust state for tool mode (spec §4.4)
- `toolbar` field on InteractionConfig must default to `true` via `#[serde(default)]` with a `fn default_true() -> bool { true }` helper so existing scenes without the field get a toolbar
- Modifier-key interactions (Alt+drag=pan, double-click=reset) must continue working with toolbar present

## 5. Tasks

### Task 1: R-tree spatial index (Rust)
- [ ] Add `rstar = "0.12"` to `crates/ferrum-wasm/Cargo.toml`
- [ ] Create `crates/ferrum-wasm/src/spatial_index.rs` — `SpatialIndex`, `PanelIndex`, `MarkEntry` (spec §3.2). Implement `RTreeObject` and `PointDistance` for `MarkEntry`. Build method takes `&[Panel]` + `&SceneData`, returns `SpatialIndex`. Snap-distance constant = 50.0.
- [ ] Query methods: `nearest(panel_id, x, y) -> Option<(MarkEntry, f64)>`, `in_envelope(panel_id, aabb) -> Vec<&MarkEntry>`, `hit_test(panel_id, x, y, tolerance) -> Option<MarkEntry>`
- [ ] Inline unit tests: empty panels, 1 mark, 100 marks, 100k marks, nearest-neighbor correctness vs linear scan, envelope correctness, snap threshold
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-wasm`

### Task 2: Wire R-tree into hit_test.rs and lib.rs (Rust)
- [ ] Declare `mod spatial_index` in `lib.rs`, store `SpatialIndex` in `WasmRenderer`
- [ ] Build index in `loadScene()` after `load_scene_with_packed()`
- [ ] Refactor `hit_test()` and `hit_test_nearest()` to accept `Option<&SpatialIndex>` — use R-tree for circle/rect batches, linear scan for others, return closer result
- [ ] Add `get_href()` method on `WasmRenderer` — reads `hrefs` from the scene graph panel/batch/node
- [ ] Add `select_in_rect()` method — uses `in_envelope` for R-tree-accelerated interval selection
- [ ] Wire `select_in_rect` through `selection_state.rs` interval handling
- [ ] All existing `hit_test.rs` tests pass unchanged
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-wasm`
- [ ] Verify: `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 3: Toolbar flag plumbing (Rust + Python)
- [ ] Add `toolbar: bool` to `InteractionConfig` in `crates/ferrum-scene/src/selection.rs` with `#[serde(default = "default_true")]`
- [ ] Add `toolbar` kwarg to `Chart.interactive()` in `chart.py`, forward to `InteractiveChart`
- [ ] Forward `toolbar` in `_ChartLike.interactive()` in `composition.py`
- [ ] Accept `toolbar` in `InteractiveChart.__init__()` in `_interactive.py`, serialize into interaction config JSON
- [ ] Pass toolbar flag through in `_html.py` if needed for standalone HTML
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-scene`
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run pytest tests/ -x -q`

### Task 4: Toolbar UI + mode switching (JS/CSS)
- [ ] Remove `_hitTest()` function and `marks` array from `ferrum-anywidget.js` (spec §5.1)
- [ ] Route all hover/click through `renderer.hitTestAt()` and `renderer.getHref()` exclusively
- [ ] Add toolbar DOM creation in `_render()` gated on `cfg.toolbar !== false` (spec §4.3, §4.6)
- [ ] Implement `currentMode` variable, d3-zoom/d3-brush filter rewiring (spec §4.4)
- [ ] Box Zoom mode: brush end handler computes zoom transform for selected rectangle
- [ ] RAF-coalesced mousemove (spec §5.2)
- [ ] Cursor CSS on `.ferrum-container[data-mode=...]` (spec §5.3)
- [ ] Keyboard shortcuts on container with `tabindex="0"` (spec §4.5)
- [ ] Save PNG with `devicePixelRatio` off-screen re-render (spec §5.4)
- [ ] Inline SVG icons for all 5 buttons
- [ ] Add toolbar and cursor styles to `ferrum-interactive.css`
- [ ] Verify: `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`

### Task 5: Integration tests (Python)
- [ ] `tests/test_interactive_toolbar.py` — `Chart(...).interactive()` HTML contains `.ferrum-toolbar`
- [ ] `Chart(...).interactive(toolbar=False)` HTML does not contain `.ferrum-toolbar`
- [ ] InteractionConfig JSON includes `"toolbar": true` by default
- [ ] Existing interactive tests still pass
- [ ] Verify: `uv run pytest tests/test_interactive_toolbar.py tests/test_phase_11_interactive/ -x -v`

### Task 6: Manual verification
- [ ] Build WASM release, render 1k-point interactive scatter, test all 5 tools
- [ ] Render 50k-point scatter, verify hover is smooth
- [ ] Verify keyboard shortcuts, cursor feedback, save PNG at retina
- [ ] Verify `toolbar=False` hides toolbar, modifier keys still work

## 6. Acceptance checks

- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass
- `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- `uv run pytest tests/ -x -q` — all pass
- Interactive scatter with 50k+ points has responsive hover tooltips
- All 5 toolbar buttons functional, mode switching changes cursor and drag behavior
