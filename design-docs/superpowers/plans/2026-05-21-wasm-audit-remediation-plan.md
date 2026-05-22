# WASM Audit Remediation — Implementation Plan

> **Status:** All 9 tasks completed and merged to main on 2026-05-22.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.
>
> **Review cycle (mandatory for every task):** coder implements → spec reviewer → quality reviewer → regression tests (chris-code:regression-test) → review-lite commit gate. No task is complete until all five stages pass. Quality and lite reviews are non-negotiable — this plan fixes bugs, and introducing new bugs during bug fixing is unacceptable.

## 1. Objective

Fix 7 bugs, 8 quick-win cleanups, and 9 structural improvements identified by four parallel audits (Rust heavyweight review, Python heavyweight review, scene-pipeline audit, JS-WASM wiring audit) of the `feat/rtree-toolbar` branch. Every bug fix must include regression tests that would catch the exact bug if reintroduced.

## 2. Spec references

- Rust heavyweight review findings D1–D10
- Python heavyweight review findings D1–D10
- Scene pipeline audit: BUG-1/2/3, WARN-1/2/4
- JS-WASM wiring audit: BUG-1/2/3, WARN-4/5/6/8

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/pack_instances.rs` | B1: fill_opacity, B3: stroke opacity baking |
| Modify | `crates/ferrum-wasm/src/tessellate.rs` | B2: stroke_opacity for path/polygon |
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | B3: stroke color baking, M3: accumulator struct |
| Modify | `crates/ferrum-wasm/src/shaders/circle.wgsl` | B3: remove opacity double-apply on stroke |
| Modify | `crates/ferrum-wasm/src/shaders/rect.wgsl` | B3: remove opacity double-apply on stroke |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | B4/B5/B7/R7/R8/M7/M8 |
| Modify | `src/ferrum/_wasm/ferrum-interactive.css` | B5: cursor selectors |
| Modify | `src/ferrum/_html.py` | B6: html.escape title |
| Modify | `crates/ferrum-wasm/src/lib.rs` | B7/R1/R2/R3/M1 |
| Create | `crates/ferrum-wasm/src/text_json.rs` | R1: extracted text serialization |
| Modify | `crates/ferrum-wasm/src/render.rs` | M4: remove redundant uniform clip, M9: annotation z-order |
| Modify | `src/ferrum/display.py` | R4: collapse _render_scene_json |
| Modify | `src/ferrum/selection.py` | R5: hex warning |
| Modify | `src/ferrum/chart.py` | M5: factor _auto_tooltips out of to_spec |
| Modify | `src/ferrum/composition.py` | M6: explicit toolbar param |
| Modify | `src/ferrum/_render.py` | M5: internal _auto_tooltips routing |
| Test | `tests/test_interactive_toolbar.py` | R6 + regression tests for B4/B5/B6 |
| Test | `tests/test_opacity_semantics.py` | B1/B2/B3 regression tests |
| Test | `crates/ferrum-wasm/src/scene_load.rs` (inline) | B3/B7 Rust-side regression tests |
| Test | `tests/test_wasm_audit_regressions.py` | B6/B7/R4/R5/M5/M6 Python regression tests |

## 4. Constraints

- **Do NOT modify golden SVGs.** If goldens fail, the code is wrong.
- **Do NOT use destructive git operations** (`git checkout main --`, `git reset --hard`, etc.).
- Run tests with `uv run pytest tests/ -n auto -q` (xdist).
- `cargo test -p ferrum-wasm` and `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` must pass after each Rust task.
- B3 fix must ensure opacity is applied exactly once to each channel (fill, stroke) across all paths (packed, non-packed, tessellated).
- Existing 3247+ tests must pass after every task.
- **Regression test requirement**: Every bug fix (B1–B7) must include tests that reproduce the exact symptom before the fix. The test must assert specific values, not just "non-empty" or "no error". Use chris-code:regression-test skill guidelines.

## 5. Tasks

### Task 1: Opacity/alpha semantics (B1, B2, B3)
- [x] **B1**: In `pack_instances.rs`, change `push_color(&mut buf, style.fill.as_ref(), style.opacity)` to use `style.fill_opacity` for the fill color alpha at lines 179 and 220.
- [x] **B2**: In `tessellate.rs`, apply `stroke_opacity` to stroke color in `tessellate_path` (line 59) and `tessellate_polygon` (line 131), matching `tessellate_line`/`tessellate_polyline` which already do this correctly.
- [x] **B3**: Stop baking `opacity` into stroke color alpha. In `scene_load.rs` `collect_nodes`, change `opt_color_to_f32(style.stroke.as_ref(), style.opacity)` to `opt_color_to_f32(style.stroke.as_ref(), style.stroke_opacity)` at lines 525, 539. Same fix in `pack_instances.rs` lines 180, 221. The shader's final `color.a *= in.opacity` (circle.wgsl:129, rect.wgsl:111) applies overall opacity once at the end — this is correct and must remain.
- [x] **Regression tests (Rust)**: In `scene_load.rs` inline tests:
  - `test_circle_stroke_color_uses_stroke_opacity_not_opacity` — construct CircleInstance via collect_nodes with opacity=0.5 stroke_opacity=0.8, assert stroke_color alpha ≈ srgb_to_linear(stroke.a * 0.8), NOT * 0.5.
  - `test_rect_stroke_color_uses_stroke_opacity_not_opacity` — same for rects.
  - Verify existing tests `b2_circle_fill_color_uses_fill_opacity_not_opacity` and `b2_rect_fill_color_uses_fill_opacity_not_opacity` still pass (they already test the fill path).
- [x] **Regression tests (Rust, ferrum-core)**: In `pack_instances.rs` tests:
  - `test_packed_circle_fill_uses_fill_opacity` — pack a batch with fill_opacity=0.5 opacity=1.0, assert fill alpha ≈ 0.5.
  - `test_packed_circle_stroke_uses_stroke_opacity` — pack with stroke_opacity=0.75 opacity=1.0, assert stroke alpha ≈ 0.75.
  - `test_packed_rect_fill_uses_fill_opacity` — same for rects.
  - `test_packed_rect_stroke_uses_stroke_opacity` — same for rects.
- [x] **Regression tests (Python)**: Create `tests/test_opacity_semantics.py`:
  - `test_fill_opacity_on_large_batch` — create >1000-point chart with mark_point(fill_opacity=0.3), render to interactive scene, verify packed circle fill alpha reflects 0.3 (not 1.0 or opacity fallback).
  - `test_stroke_opacity_on_area_mark` — create area chart with stroke_opacity=0.5, render SVG, verify stroke-opacity attribute value.
  - `test_opacity_with_stroke_no_double_apply` — create chart with mark_point(opacity=0.5, stroke="black"), render, verify stroke is not doubly-faded relative to fill.
- [x] **Regression tests (Rust, tessellate)**: In `tessellate.rs` or `scene_load.rs` tests:
  - `test_tessellate_path_applies_stroke_opacity` — construct a Path node with stroke_opacity=0.6 opacity=1.0, tessellate it, verify mesh vertex color alpha ≈ srgb_to_linear(stroke.a * 0.6).
  - `test_tessellate_polygon_applies_stroke_opacity` — same for polygon stroke tessellation.
- [x] Verify: `cargo test -p ferrum-wasm && cargo test -p ferrum-core`
- [x] Verify: `uv run pytest tests/ -n auto -q`

### Task 2: Interactive JS/CSS/HTML fixes (B4, B5, B6, B7, R7, R8)
- [x] **B4**: In `ferrum-anywidget.js` `onReset()` (~line 284), capture the return value of `renderer.clearSelections()` and call `adapter.onSelectionChange(JSON.parse(stateJson))`.
- [x] **B5**: In `ferrum-interactive.css`, change cursor selectors to target `svg` not `canvas` for `select` and `boxzoom` modes (SVG overlay sits on top of canvas in those modes).
- [x] **B6**: In `_html.py` line 275, use `html.escape(title)` for the `<title>` tag. Add `import html` at top.
- [x] **B7**: In `lib.rs` `build_zoomed_text_json`, filter out tick labels whose transformed position falls outside the panel's plot area. The function already has `panel_id` — pass `plot_area: Option<(f64,f64,f64,f64)>` as a parameter (from `scene.panels[panel_id].plot_area`). Skip x-tick labels where `new_x < plot_area.x || new_x > plot_area.x + plot_area.w`. Same for y-tick labels with `new_y`.
- [x] **R7**: Remove dead `_downloadBlob` function in `ferrum-anywidget.js` (~lines 156–164).
- [x] **R8**: In `ferrum-anywidget.js` keydown handler (~line 383), match both lowercase and uppercase: `e.key === 'p' || e.key === 'P'`, etc.
- [x] **Regression tests (Python)**: In `tests/test_wasm_audit_regressions.py`:
  - `test_html_title_escapes_special_chars` — save chart with title containing `<script>alert(1)</script>`, load HTML, assert `&lt;script&gt;` appears (escaped), not raw `<script>`.
  - `test_html_title_escapes_ampersand` — title with `A & B`, assert `A &amp; B` in `<title>`.
- [x] **Regression tests (Rust)**: In `lib.rs` or `text_json.rs` tests:
  - `test_zoomed_tick_labels_clipped_to_plot_area` — construct text elements with one x-tick at x=10 (inside plot_area x=50..350) and one at x=200 (inside). After zoom transform that maps x=10 to new_x=20 (still inside? depends on transform), verify only in-bounds labels appear. Better: set up plot_area x=50..350, place tick at scene x=40 that after zoom transform lands at new_x=30 (outside x=50). Assert it is NOT emitted.
  - `test_zoomed_tick_labels_inside_plot_area_kept` — tick at scene position that transforms to inside plot_area. Assert it IS emitted.
- [x] Verify: WASM rebuild (`wasm-pack build`), regenerate test HTMLs, browser check.

### Task 3: Rust lib.rs quick wins (R1, R2, R3)
- [x] **R1**: Create `crates/ferrum-wasm/src/text_json.rs`. Move `build_text_json`, `build_text_json_from`, `build_zoomed_text_json`, `text_element_to_json`, `tick_label_json` from `lib.rs`. Extract a shared `fn text_style_fields(style: &TextStyle) -> (...)` to replace the 3 duplicated FontWeight/TextAnchor/TextBaseline match blocks.
- [x] **R2**: In `lib.rs` `hit_test_at`, remove the ~70-line packed-batch linear fallback loop (lines ~370–442). The spatial index now covers packed instances; the fallback is dead code with inconsistent hit tolerance.
- [x] **R3**: Replace hand-rolled `format!("{{\"panel\":{},...")` JSON in `hit_test_at` and `format_tooltip_content` with `serde_json::json!`. The crate already depends on serde_json.
- [x] **Regression tests (Rust)**: After R2, add a test that verifies hit-testing still works on packed batches (the spatial index path): `test_hit_test_packed_batch_uses_spatial_index` — load a scene with packed circles, call `hit_test_at` at a known circle center, assert it returns the correct panel/batch/idx.
- [x] Verify: `cargo test -p ferrum-wasm`, `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 4: Python small fixes (R4, R5, R6)
- [x] **R4**: In `display.py`, replace `_render_scene_json` body with a delegation to `_interactive._render_scene`. Import and call `_render_scene(chart)` instead of duplicating the logic.
- [x] **R5**: In `selection.py` `_hex_to_color_dict`, add `warnings.warn(f"Unrecognized hex color {hex_str!r}, defaulting to black", stacklevel=2)` before the catch-all `return {"r": 0, ...}`.
- [x] **R6**: Add `test_chart_save_html_toolbar_false` to `tests/test_interactive_toolbar.py` — call `chart.save(path, toolbar=False)` via the `display.save_chart` path and assert `"toolbar": false` in the output HTML.
- [x] **Regression tests (Python)**: In `tests/test_wasm_audit_regressions.py`:
  - `test_hex_to_color_dict_warns_on_malformed` — call `_hex_to_color_dict("#xyz")`, assert `warnings.warn` fires and result is `{"r": 0, "g": 0, "b": 0, "a": 255}`.
  - `test_hex_to_color_dict_3char_expands` — `_hex_to_color_dict("#abc")` returns correct RGB (0xaa, 0xbb, 0xcc).
  - `test_hex_to_color_dict_4char_expands` — `_hex_to_color_dict("#abcd")` returns correct RGBA.
  - `test_render_scene_json_delegates_to_render_scene` — call `_render_scene_json(chart)`, verify it returns the same (scene_json, packed_data) tuple as `_render_scene(chart)`.
  - `test_chart_save_html_toolbar_true_default` — `chart.save(path)` for HTML produces `"toolbar": true`.
- [x] Verify: `uv run pytest tests/ -n auto -q`

### Task 5: Rust structural — accumulator + clip (M3, M4)
- [x] **M3**: Introduce a `SceneCollector` struct in `scene_load.rs` grouping `circles`, `rects`, `mesh`, `static_mesh`, `texts`, `images`, `draw_commands`, `prev_c`, `prev_r`. Replace 8-param `collect_nodes` and `emit_draw_commands` with methods: `collector.collect(nodes, is_mark, plot_area, batch_cap, batch_join)` and `collector.emit(additive, is_mark, plot_area)`.
- [x] **M4**: In shaders (circle.wgsl, rect.wgsl, mesh.wgsl), remove the `u.clip` fragment discard branch — the GPU scissor rect now handles clipping for instanced marks, and the identity uniform's clip was always full-canvas (a no-op). Remove `clip_x/y/w/h` from `Uniforms` struct, update `Uniforms::identity`, and remove clip fields from the WGSL uniform struct. This shrinks the uniform buffer from 48 to 32 bytes.
- [x] **Regression tests**: All existing `scene_load.rs` tests must pass unchanged (they verify SceneData output). Add `test_scene_collector_produces_same_output_as_before` — load the `make_test_scene` fixture through the new SceneCollector path and assert circle/rect/mesh counts match the old path.
- [x] Verify: `cargo test -p ferrum-wasm`, clippy, `uv run pytest tests/ -n auto -q`

### Task 6: Rust lib.rs module split (M1)
- [x] **M1**: After Task 3 extracted text_json.rs, split remaining concerns from `lib.rs`: move tooltip formatting to `text_json.rs` (or a `tooltip.rs`), move `apply_conditionals_and_render` to a `conditional_render.rs` or inline into `conditional.rs`. `lib.rs` should contain only the `WasmRenderer` struct, `#[wasm_bindgen]` methods, and thin delegation to internal modules.
- [x] **Regression tests**: Pure refactor — existing tests cover behavior. Verify no `#[wasm_bindgen]` export signatures changed by running `wasm-pack build` and checking `ferrum_wasm.d.ts` is unchanged.
- [x] Verify: `cargo test -p ferrum-wasm`, clippy

### Task 7: Rust perf + z-order (M2, M9)
- [x] **M2**: Add `GpuBuffers::update_instances(gpu, circles, rects)` that re-uploads only `circle_instance_buffer` and `rect_instance_buffer` without re-creating mesh/static_mesh/uniform/image buffers. Call this from `apply_conditionals_and_render` instead of `GpuBuffers::from_scene`.
- [x] **M9**: Fix annotation z-order: annotation `Line`/`Path` nodes currently go to `static_mesh` (drawn first, behind marks). Either (a) collect annotation mesh into a third `annotation_mesh` buffer drawn after marks, or (b) emit annotation tessellated geometry as post-mark draw commands in a separate pass.
- [x] **Regression tests (Rust)**: `test_annotation_lines_in_post_mark_buffer` — construct scene with panel containing marks + annotation Lines, verify annotation mesh is in the post-mark buffer (not static_mesh). `test_update_instances_preserves_mesh` — call `update_instances` with modified circles, verify mesh_index_count is unchanged.
- [x] Verify: `cargo test -p ferrum-wasm`, clippy, browser check (annotations should appear above marks)

### Task 8: Python API cleanup (M5, M6)
- [x] **M5**: Factor `_auto_tooltips` logic out of `Chart.to_spec()`. Move the tooltip injection block to a private `_inject_auto_tooltips(kw)` method. Have `_render_inputs()` call injection after `to_spec()`. Public `to_spec()` signature drops `_auto_tooltips`.
- [x] **M6**: In `composition.py` `_ChartLike.save()`, replace `kwargs.pop("toolbar", True)` with an explicit `toolbar: bool = True` parameter. Validate it is only meaningful for HTML format.
- [x] **Regression tests (Python)**: In `tests/test_wasm_audit_regressions.py`:
  - `test_to_spec_no_auto_tooltips_param` — verify `Chart.to_spec()` no longer accepts `_auto_tooltips` kwarg (TypeError on unknown kwarg).
  - `test_interactive_render_still_injects_tooltips` — render interactive scene, verify tooltip fields present in scene JSON.
  - `test_svg_render_no_tooltip_injection` — render SVG, verify no tooltip bloat in SVG output.
  - `test_chartlike_save_toolbar_explicit_param` — verify `HConcatChart.save(path, toolbar=False)` works and produces correct HTML.
  - `test_chartlike_save_toolbar_invalid_format_warns` — verify `composition.save("out.svg", toolbar=False)` does not error (toolbar ignored for non-HTML).
- [x] Verify: `uv run pytest tests/ -n auto -q`

### Task 9: JS robustness (M7, M8)
- [x] **M7**: Fix `ResizeObserver` callback to read `entry.contentBoxSize` (or `canvas.clientWidth/Height`) and update `canvas.width/height` before calling `renderer.resize()`.
- [x] **M8**: Fix transition `_step` closure: cancel any in-flight RAF loop before starting a new transition (track the RAF id and call `cancelAnimationFrame` in `_reload`).
- [x] **Regression tests**: These are JS-only changes — no Python/Rust test infrastructure for DOM behavior. Verify manually in browser: (1) resize the browser window, confirm chart redraws at correct size, (2) rapidly change scene data twice within 300ms, confirm no visual glitch.
- [x] Verify: WASM rebuild, browser check

## 6. Acceptance checks

- `cargo test -p ferrum-wasm` — all pass
- `cargo test -p ferrum-core` — all pass
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- `uv run pytest tests/ -n auto -q` — 3280+ pass (3247 existing + ~33 new regression tests), 0 fail
- Golden SVGs unchanged — `git diff -- tests/goldens/` produces empty output
- WASM build + browser verification: grid lines even, marks clipped on zoom, tick labels clipped on zoom, annotations above marks, cursor correct in all modes, Save PNG captures full chart, reset clears selections properly
