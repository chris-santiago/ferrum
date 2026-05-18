# Interactive HTML Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Fix broken HTML export, add interval/brush selection support to WASM, extend `.interactive()` to composition types, and unify JS rendering across Jupyter and standalone HTML — with upfront regression tests locking every invariant before implementation begins.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-17-interactive-html-export-design.md` — full spec
  - §5 Architecture (adapter pattern, scene merge, interval conditional resolution)
  - §6 Canonical interfaces (`assemble_html`, `_render_interactive`, `handleDrag`)
  - §7 Invariants (13 constraints — all must hold)
  - §10 Validation strategy (R1–R6 Rust tests, P1–P13 Python tests, S1–S11 smoke tests)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Test | `tests/test_html_export_regression.py` | Tier 2 regression tests (P1–P13) — written first |
| Test | `crates/ferrum-wasm/src/selection_state.rs` | Tier 1 Rust tests (R1–R6) — appended to existing `#[cfg(test)]` |
| Test | `crates/ferrum-wasm/src/conditional.rs` | Tier 1 test R3 (interval conditional) |
| Modify | `crates/ferrum-wasm/src/selection_state.rs` | Add `contains_point(x, y)` for interval spatial containment |
| Modify | `crates/ferrum-wasm/src/conditional.rs` | Pass mark positions to interval containment check |
| Modify | `crates/ferrum-wasm/src/lib.rs` | Expose `handleDrag` via `wasm_bindgen`; add `shift_held` to `handleClick` |
| Modify | `src/ferrum/_html.py` | Accept `packed_data`, embed as base64, bake theme background, inline JS from anywidget |
| Modify | `src/ferrum/display.py` | Thread `packed_data` through `_render_scene_json` → `save_chart` → `assemble_html` |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | Refactor `_render` to accept adapter; add brush overlay + `handleDrag` wiring; CSS-coordinate scaling |
| Modify | `src/ferrum/_interactive.py` | Accept compositions in `_render_scene`; dispatch to `_render_interactive` |
| Modify | `src/ferrum/composition.py` | Add `interactive()`, `_render_interactive()`, `_merge_scenes()` |
| Modify | `scripts/export-interactive-examples.py` | Add composition + brush examples for Tier 3 smoke tests |

## 4. Constraints

- **Tests before implementation.** Tier 1 and Tier 2 tests are written and committed before any implementation code. Tests that assert new behavior (P5, P6) are expected to fail initially — mark with `pytest.mark.xfail(reason="not yet implemented")` and remove the mark when the feature lands.
- **No duplicated JS.** `ferrum-anywidget.js` is the single source. HTML export inlines it. Delete `ferrum-interactive.js` if it still exists from the abandoned branch.
- **WASM method arity is atomic.** Any Rust `wasm_bindgen` signature change must update all JS callers in the same task (spec §7).
- **Composition `show_svg()` must not change.** The SVG path is untouched. Verify with test P7.
- **Background baked into HTML template.** Extract from scene JSON in Python, write into `<body>`/`<div>` style. JS must not override (spec §7, Bug #8).
- **Mouse coords scaled by `canvas.width / rect.width`.** All hit-test coords in JS (spec §7, Bug #13).
- **ResizeObserver not gated behind mode flag** (spec §7, Bug #14).
- **Coding agent dispatch:** Python tasks → `python-coder`; Rust tasks → `rust-coder`. Never use general-purpose agents.

## 5. Tasks

### Task 1: Regression tests — Rust (Tier 1: R1–R6)

- [ ] Append tests R1–R6 to `crates/ferrum-wasm/src/selection_state.rs` `#[cfg(test)]` and `crates/ferrum-wasm/src/conditional.rs` `#[cfg(test)]`
- [ ] R1–R2 test `handle_drag` and `contains_point` — these methods don't exist yet, so stub `contains_point` to return `false` and write tests that currently fail
- [ ] R3 builds a minimal scene with circles + interval selection + conditional; asserts conditional color application — currently fails (interval `contains` returns false)
- [ ] R4–R6 test existing behavior that must not regress (shift-click, empty scene, JSON serialization)
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-wasm` — R4–R6 pass, R1–R3 fail as expected

### Task 2: Regression tests — Python (Tier 2: P1–P13)

- [ ] Create `tests/test_html_export_regression.py` with tests P1–P13 per spec §10
- [ ] P1–P4, P7–P13 test current behavior that must survive the refactor — these should pass now
- [ ] P5–P6 test composition `.interactive()` — mark `xfail(reason="composition interactive not implemented")`
- [ ] Verify: `uv run pytest tests/test_html_export_regression.py -v` — P1–P4 and P7–P13 pass; P5–P6 xfail

### Task 3: Rust — interval selection + `handleDrag`

- [ ] Add `contains_point(x: f64, y: f64) -> bool` to `SelectionState::Interval` in `selection_state.rs` (spec §7)
- [ ] Update `resolve_conditionals` in `conditional.rs` to dispatch spatial containment for interval selections — extract mark position from `SceneNode` circle `cx/cy` or rect `x/y` (spec §5)
- [ ] Expose `handleDrag` via `wasm_bindgen` in `lib.rs` — same pattern as `handleClick`: update state, resolve conditionals, rebuild GPU buffers, re-render, return JSON (spec §6)
- [ ] Add `shift_held: bool` parameter to `handle_click` in `selection_state.rs`; update `wasm_bindgen` `handleClick` in `lib.rs` to accept and forward it (spec §7)
- [ ] Verify: `cargo test -p ferrum-wasm` — all R1–R6 pass
- [ ] Verify: `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 4: WASM rebuild

- [ ] `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`
- [ ] Verify WASM artifacts updated: `ls -la src/ferrum/_wasm/ferrum_wasm_bg.wasm`

### Task 5: JS — adapter pattern + brush + interactions

- [ ] Refactor `_render(container, sceneJson, model)` in `ferrum-anywidget.js` to `_render(container, sceneJson, adapter)` where adapter implements `{ getPackedData, getInteractionConfig, onSelectionChange, onZoomChange }` (spec §5)
- [ ] Update `export async function render({ model, el })` to construct a Jupyter adapter wrapping `model.get/set/save_changes`
- [ ] Add standalone adapter factory: `_standaloneAdapter(packedB64, interactionConfig)` — packed data decoded from base64, selection/zoom callbacks are local-only
- [ ] Add brush overlay logic: `mousedown` creates brush div, `mousemove` resizes, `mouseup` calls `renderer.handleDrag(panel, x0, y0, x1, y1)` and forwards result to `adapter.onSelectionChange` (spec §8)
- [ ] Add CSS-coordinate scaling via `canvas.width / rect.width` to all mouse handlers (spec §7, Bug #13)
- [ ] Update `handleClick` call to pass `e.shiftKey` as 3rd argument (spec §7, Bug #4)
- [ ] Ensure ResizeObserver is outside any mode-gated block (spec §7, Bug #14)
- [ ] Pan vs. brush disambiguation: if scene has interval selection, drag = brush; pan requires Alt key (spec §8)
- [ ] Verify: no `model.get` / `model.set` calls outside the Jupyter adapter factory

### Task 6: Python — fix HTML export pipeline

- [ ] Change `_render_scene_json` in `display.py` to return `(str, bytes)` instead of `str`; update `save_chart` to thread `packed_data` to `assemble_html` (spec §6)
- [ ] Rewrite `assemble_html` in `_html.py`: accept `packed_data: bytes`, embed as base64, inline JS from `ferrum-anywidget.js` (strip ESM export, wrap in standalone `main()` with standalone adapter), call `loadScene(SCENE_JSON, packedArr)` with two args (spec §6, §7)
- [ ] Extract background color from scene JSON in Python, bake into `<body>` and container `<div>` style attributes (spec §7, Bug #8)
- [ ] Delete `ferrum-interactive.js` if still present from abandoned branch — single source is `ferrum-anywidget.js`
- [ ] Verify: `uv run pytest tests/test_html_export_regression.py -v` — P1–P4, P7–P13 pass

### Task 7: Python — composition `.interactive()` + scene merge

- [ ] Add `_render_interactive() -> tuple[str, bytes]` to `_CompositeBase` in `composition.py` (spec §6)
- [ ] `LayerChart._render_interactive`: build merged spec via `ChartSpec.layers`, delegate to `render_interactive` (spec §5)
- [ ] `HConcatChart`/`VConcatChart`/`ConcatChart._render_interactive`: call `render_interactive` per child, merge scene JSONs via `_merge_scenes()` — offset panel positions, re-index panel IDs, concatenate packed data with rewritten header indices (spec §5, §11)
- [ ] `FacetChart`/`RepeatChart._render_interactive`: delegate to `render_interactive` directly (facet is native to ChartSpec)
- [ ] Add `interactive()` method to `_CompositeBase` returning `InteractiveChart(self)`
- [ ] Update `_render_scene` in `_interactive.py` to dispatch: if input has `_render_interactive`, call it; else call `_render_inputs` + `render_interactive` (spec §6)
- [ ] Remove `xfail` marks from P5–P6
- [ ] Verify: `uv run pytest tests/test_html_export_regression.py -v` — all P1–P13 pass

### Task 8: Smoke test script + manual verification

- [ ] Update `scripts/export-interactive-examples.py` to include composition examples (HConcat linked views) and brush selection example
- [ ] Run script, open all HTML files in browser
- [ ] Walk through S1–S11 checklist (spec §10 Tier 3)
- [ ] Verify Jupyter: open a notebook, run `.interactive()` on a chart with selection, confirm tooltip + click + zoom work

### Task 9: Full test suite

- [ ] `uv run pytest -n auto` — all existing tests pass
- [ ] `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass
- [ ] `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean

## 6. Acceptance checks

- `uv run pytest tests/test_html_export_regression.py -v` — all 13 tests pass (no xfail)
- `cargo test -p ferrum-wasm` — all tests pass including R1–R6
- `uv run pytest -n auto` — full suite green
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- Manual: S1–S11 smoke checklist verified in browser
- Manual: Jupyter `.interactive()` unchanged (S11)
