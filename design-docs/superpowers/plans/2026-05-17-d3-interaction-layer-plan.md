# D3 Interaction Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Replace hand-written interaction handlers and CSS-div text rendering with D3-zoom, D3-brush, and SVG `<text>` elements via D3-selection — eliminating the interaction and text-positioning bug classes documented in the spec.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-17-d3-interaction-layer-design.md`
  - §4 System behavior (zoom, brush, text rendering, tooltip, click)
  - §5 Architecture (layer stack, bridges, SVG text, bundling)
  - §6 Canonical interfaces (`setTransform`, D3-zoom config, D3-brush config, data-join)
  - §7 Invariants (13 constraints)
  - §9 Acceptance criteria (15 items)
  - §10 Validation (R1–R6, P1–P17, S1–S15)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `src/ferrum/_wasm/d3-interactions.js` | Vendored D3 bundle (brush, zoom, selection + deps) |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | Replace interaction handlers + `_placeText` with D3 |
| Modify | `src/ferrum/_wasm/ferrum-interactive.css` | Remove `.ferrum-text`, `.ferrum-overlay`; add SVG styles |
| Modify | `crates/ferrum-wasm/src/lib.rs` | Add `setTransform` WASM method |
| Modify | `crates/ferrum-wasm/src/zoom_pan.rs` | Add `set_absolute(k, tx, ty)` to `ZoomPanState` |
| Modify | `src/ferrum/_html.py` | Inline D3 bundle; update SVG overlay; remove `_placeText`-related stripping |
| Modify | `src/ferrum/_interactive.py` | Prepend D3 bundle to anywidget ESM |
| Test | `tests/test_html_export_regression.py` | Add P14–P17 |
| Test | `crates/ferrum-wasm/src/lib.rs` | Add `setTransform` Rust tests |
| Modify | `scripts/export-interactive-examples.py` | Add titles to all charts for S14 verification |

## 4. Constraints

- **D3 vendored, not CDN** — HTML must be self-contained (spec §7)
- **D3-zoom is single source of zoom state** — delete `_zoom = { sx, sy, tx, ty }` and `_invZoom`; use `d3.zoomTransform(canvas)` instead (spec §7)
- **`onWheel`/`onPan`/`resetZoom` remain** — Jupyter Python-side zoom rebuild uses them (spec §7)
- **`_placeText` and `.ferrum-overlay` div must be fully removed** — no CSS-div text remains (spec §7)
- **Inter `@font-face` in SVG `<defs><style>`** — not in CSS; SVG has its own style scope (spec §7)
- **Brush extent constrained to `plot_area`** — not full canvas (spec §8)
- **All P1–P13 and R1–R6 must still pass** — no regressions (spec §7)
- **Coding dispatch:** Rust → `rust-coder`; Python → `python-coder`; JS is orchestrator-level

## 5. Tasks

### Task 1: Vendor D3 bundle
- [ ] Download D3 ESM bundle containing `d3-brush`, `d3-zoom`, `d3-selection`, and their transitive deps (`d3-dispatch`, `d3-drag`, `d3-interpolate`, `d3-transition`, `d3-timer`, `d3-ease`, `d3-color`) as a single file from `esm.sh` or build via esbuild
- [ ] Save as `src/ferrum/_wasm/d3-interactions.js`
- [ ] Verify exports: `brush`, `zoom`, `select`, `zoomIdentity`, `zoomTransform` are accessible
- [ ] Verify size is <60KB minified

### Task 2: Rust — add `setTransform`
- [ ] Add `set_absolute(k, tx, ty)` to `ZoomPanState` in `zoom_pan.rs` — sets scale and translation directly
- [ ] Add `setTransform(k, tx, ty)` to `WasmRenderer` in `lib.rs` (spec §6): set zoom state, rebuild GPU uniforms, re-render, return text JSON
- [ ] Add Rust tests: `setTransform` identity returns original text positions; non-identity returns shifted positions
- [ ] Verify: `cargo test -p ferrum-wasm` — all pass including new tests
- [ ] Verify: `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 3: WASM rebuild
- [ ] `wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`

### Task 4: Rewrite JS interaction layer
- [ ] Import D3 functions from vendored bundle at top of `ferrum-anywidget.js`
- [ ] **Replace `_placeText`** with `_placeTextSvg(svgEl, texts)` using D3 data-join on SVG `<text>` (spec §5 SVG text rendering). Map `anchor` → `text-anchor`, `baseline` → `dominant-baseline`, `angle` → `rotate(angle, x, y)` transform
- [ ] **Replace zoom/pan handlers** (wheel, mousedown/mousemove/mouseup pan, dblclick) with `d3.zoom()` attached to canvas. Config per spec §6. Zoom event calls `renderer.setTransform(k, x, y)` then `_placeTextSvg(svg, JSON.parse(textJson))`
- [ ] **Replace brush handlers** (mousedown/mousemove/mouseup brush, `_brushDiv`, `_brushOrigin`, `_isBrushing`) with `d3.brush()` attached to SVG overlay `<g>`. Config per spec §6. Brush end calls `renderer.handleDrag`
- [ ] **Delete**: `_zoom`, `_invZoom`, `_panStart`, `_isDragging`, `_isBrushing`, `_brushDiv`, `_brushOrigin`, `_endPan`, `_canvasCoords` (D3 handles coordinate transforms), `.ferrum-overlay` div creation, `.ferrum-brush` div creation
- [ ] **Keep**: `_hitTest`, tooltip mousemove handler (update to use `d3.zoomTransform` for inverse), click handler for point selection + href
- [ ] **DOM change**: replace the `ferrum-overlay` `<div>` with an `<svg>` element (same position/size). Add `<defs><style>@font-face Inter</style></defs>`. Brush `<g>` and text elements are children of this SVG
- [ ] Update `createStandaloneAdapter` and `render()` export — no signature changes, but internal calls use D3
- [ ] Verify: no `_panStart`, `_brushOrigin`, `_isBrushing`, `_placeText`, `ferrum-overlay` remain in file

### Task 5: Update CSS
- [ ] Remove `.ferrum-text`, `.ferrum-overlay` classes from `ferrum-interactive.css`
- [ ] Remove `.ferrum-brush` class
- [ ] Add `.ferrum-label` styles for SVG text (pointer-events: none, user-select: none)
- [ ] Add D3 brush default styles if needed (D3 generates `.overlay`, `.selection`, `.handle` rects)

### Task 6: Update HTML assembly
- [ ] In `_html.py`: inline `d3-interactions.js` content alongside the anywidget JS
- [ ] Update `_strip_anywidget_for_standalone` if D3 import syntax changes
- [ ] Remove Inter `@font-face` from CSS block (it moves into SVG `<defs>`) — or keep in both for tooltip font
- [ ] Verify: generated HTML contains D3 bundle, SVG overlay, no `ferrum-overlay` div

### Task 7: Update Jupyter ESM
- [ ] In `_interactive.py` `_build_anywidget_esm`: prepend D3 bundle content to the ESM string (before the anywidget JS)
- [ ] Verify: Jupyter `.interactive()` still renders

### Task 8: Regression tests P14–P17
- [ ] P14: HTML source contains `d3.brush` or `d3.zoom` (or equivalent export names from bundle)
- [ ] P15: HTML source does NOT contain `_panStart`, `_brushOrigin`, `_isBrushing`
- [ ] P16: HTML source does NOT contain `ferrum-overlay` or `_placeText`
- [ ] P17: HTML source contains `<svg` and `ferrum-label`
- [ ] Verify: `uv run pytest tests/test_html_export_regression.py -v` — all P1–P17 pass

### Task 9: Export script + smoke tests
- [ ] Add `.properties(title="...")` to all charts in `scripts/export-interactive-examples.py`
- [ ] Regenerate all 8 HTML files
- [ ] Walk S1–S15 checklist (spec §10)

### Task 10: Full test suite
- [ ] `uv run pytest -n auto` — all pass
- [ ] `cargo test -p ferrum-wasm` — all pass
- [ ] `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean

## 6. Acceptance checks

- `uv run pytest tests/test_html_export_regression.py -v` — all P1–P17 pass
- `cargo test -p ferrum-wasm` — all pass including `setTransform` tests
- `uv run pytest -n auto` — full suite green
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown` — clean
- Manual: S1–S15 smoke checklist verified in browser
- Manual: Jupyter `.interactive()` unchanged (S11)
- Generated HTML: no `_panStart`, `_brushOrigin`, `_placeText`, `ferrum-overlay`
- Generated HTML: has D3 bundle, SVG overlay, `ferrum-label` text elements
