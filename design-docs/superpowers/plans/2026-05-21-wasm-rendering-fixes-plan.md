# WASM Rendering Fixes — Implementation Plan

> **Status:** Both tasks completed and merged to main on 2026-05-22.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Fix two remaining visual issues in the interactive WASM renderer: uneven grid lines from sub-pixel rounding, and zoomed marks extending past the plot area into axis margins.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-21-rtree-toolbar-design.md` §4 (interactive rendering)
- Auditor report from this session (zoom scaling, clip region findings)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | Snap grid line positions to pixel centers in WASM path only |
| Modify | `crates/ferrum-wasm/src/render.rs` | Set clip region to panel plot_area for mark draw commands |
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | Store per-panel plot_area in DrawCommand or side structure for clip |

## 4. Constraints

- **Do NOT modify `crates/ferrum-core/`** — these are WASM rendering fixes only. The SVG renderer and scene builder must not be touched. Golden SVG tests must continue to pass unchanged.
- **Do NOT use `git checkout main --`** or any other destructive git operation. All verification must use `uv run pytest` on the current branch without modifying committed files.
- Run tests with `uv run pytest tests/ -n auto -q` (xdist for speed).
- Existing 3247 tests must pass after each task.

## 5. Tasks

### Task 1: Grid pixel-snap in WASM scene loading
- [x] In `scene_load.rs` `load_scene_with_packed`, snap grid `SceneNode::Line` positions to pixel centers (`round() + 0.5`) before passing to `collect_nodes` for `static_mesh`. Only vertical lines (x1≈x2) snap x; only horizontal lines (y1≈y2) snap y.
- [x] This is WASM-only — `ferrum-core/axis.rs` is untouched, so SVG output is unchanged and goldens remain valid.
- [x] Verify: `source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-wasm`
- [x] Verify: `uv run pytest tests/ -n auto -q` — 0 failures

### Task 2: Clip zoomed marks to panel plot_area
- [x] Currently the clip uniform is `(0, 0, canvas_w, canvas_h)` for all draw commands. For `is_mark: true` draw commands, set clip to the panel's `plot_area` so zoomed marks don't leak into axis margins.
- [x] Approach: store `plot_area: Option<[f32; 4]>` on `DrawCommand` (set from `panel.plot_area` during mark batch processing in `scene_load.rs`). In `render.rs`, when uploading uniforms for mark commands, use `cmd.plot_area` as the clip region instead of the full canvas.
- [x] For non-mark commands, clip remains the full canvas (they render in margins intentionally).
- [x] The identity uniform buffer's clip stays at full canvas. The zoom uniform buffer's clip is updated per-command before each mark draw call.
- [x] Verify: `source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-wasm`
- [x] Verify: `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`
- [x] Verify: `uv run pytest tests/ -n auto -q` — 0 failures

## 6. Acceptance checks

- `cargo test -p ferrum-wasm` — all pass
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- `uv run pytest tests/ -n auto -q` — 3247+ pass, 0 fail
- Build WASM, regenerate test HTML, verify in browser: grid lines evenly spaced, zoomed marks clipped to plot area
