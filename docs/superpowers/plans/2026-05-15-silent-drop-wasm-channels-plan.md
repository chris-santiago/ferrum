# Silent-Drop Remediation — WASM Channels & Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Wire `stroke_opacity`, `stroke_width`, `stroke_dash`, and `angle` as per-row field-driven encoding channels in the WASM interactive renderer for Circle and Rect mark kinds; and wire `mark_raster(blend="additive")` GPU additive compositing.

## 2. Spec references

- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §4 System behavior — WASM stroke/angle channels`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §5 Architecture — Interactive WASM`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §7 Invariants`
- `docs/superpowers/specs/2026-05-15-silent-drop-remediation-design.md §11 Open questions — stroke_dash palette`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | add four fields to `CircleInstance` and `RectInstance`; populate from encoded columns |
| Modify | `crates/ferrum-wasm/src/` (shader files) | consume new per-instance attributes in vertex/fragment shaders |
| Modify | `crates/ferrum-core/src/render/marks/point.rs` | emit stroke/angle column values into `MarkBatch` per-row data |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` (and rect.rs) | same for Rect/Bar |
| Modify | `crates/ferrum-core/src/render/scale_resolve.rs` | resolve `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` channels to columns |
| Modify | `crates/ferrum-wasm/src/render_pipeline.rs` (or equivalent) | second pipeline state for additive blend; select per-batch |
| Test | `crates/ferrum-wasm/src/` (inline tests) | assert instance fields populated; assert correct blend state selected |

## 4. Constraints

- Stroke/angle channels are **data-driven constants per row**, not selection-conditional encodings — use the instance-buffer path, not `conditional.rs` (spec §8).
- `stroke_dash` palette is exactly the four entries in spec §6 (solid, dashed 6/3, dotted 2/3, dash-dot 6/3/2/3); integer column values clamp to nearest index. SVG and WASM must use the same palette (spec §7).
- `angle` rotates around the instance anchor in screen-space degrees.
- `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` are removed from `_SILENT_CHANNELS` only after **both** SVG (static plan Task 10) and WASM (this plan Task 5) are wired — coordinate with static SVG plan.
- `blend="additive"` WASM uses a second `wgpu::RenderPipeline` with additive blend state — not a post-process pass and not a fragment shader hack (spec §8).
- SVG blend path (`mix-blend-mode:screen`) already works; do not touch it.
- Before any commit touching `*.rs`: dispatch `rust-review-lite`.

## 5. Tasks

### Task 1: Extend GPU instance structs
- [ ] Add `stroke_opacity: f32`, `stroke_width: f32`, `stroke_dash: u8`, `angle: f32` to `CircleInstance` and `RectInstance` in `scene_load.rs`
- [ ] Define the `stroke_dash` palette (4–8 patterns) as a named constant in `scene_load.rs`; document the mapping from index to pattern
- [ ] Write failing unit tests: construct a `CircleInstance` with non-default stroke fields; assert they round-trip through serialisation
- [ ] Verify: `source ~/.cargo/env && cargo test -p ferrum-wasm --lib`

### Task 2: Resolve stroke/angle channels to batch columns
- [ ] Add `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` to the set of channels `scale_resolve.rs` recognises and maps to output columns
- [ ] Columns should be present in the `RecordBatch` passed to mark renderers when the encoding is set; absent (or filled with defaults) when unset
- [ ] Write failing test: a chart with `encode(stroke_opacity="col")` produces a RecordBatch with a `stroke_opacity` column
- [ ] Verify: `source ~/.cargo/env && DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core --lib`

### Task 3: Populate instance fields from batch columns
- [ ] In `scene_load.rs` batch-building loop for Circle instances, read `stroke_opacity`/`stroke_width`/`stroke_dash`/`angle` columns and set the new instance fields; use spec-defined defaults when column is absent (`stroke_opacity=1.0`, `stroke_width=1.0`, `stroke_dash=0`, `angle=0.0`)
- [ ] Same for Rect instances
- [ ] Write failing unit tests: pass a `MarkBatch` carrying stroke column data; assert the built `CircleInstance`/`RectInstance` fields match the column values row-by-row
- [ ] Verify: `source ~/.cargo/env && cargo test -p ferrum-wasm --lib`

### Task 4: Update WebGPU shaders
- [ ] Add the four new per-instance attributes to the vertex shader input layout
- [ ] Apply `stroke_opacity` to the stroke alpha channel; `stroke_width` to stroke geometry; `stroke_dash` to dash-pattern selection; `angle` to the instance transform matrix
- [ ] Verify: `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 5: Remove channels from _SILENT_CHANNELS
- [ ] Coordinate with static SVG plan Task 10 — remove `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` from `_SILENT_CHANNELS` only after both SVG and WASM paths are wired
- [ ] Write Python smoke test: `Chart(df).mark_point().encode(stroke_opacity="val").show_svg()` does not emit a `UserWarning`
- [ ] Verify: `uv run pytest tests/ -k "stroke" -v`

### Task 6: blend="additive" WASM GPU compositing
- [ ] Create a second `wgpu::RenderPipeline` with additive blend state (`src + dst`) alongside the existing alpha pipeline (spec §4, §5, §8)
- [ ] Per-batch pipeline selection: raster batches with `blend_mode == Additive` use the additive pipeline; all others use alpha
- [ ] Write failing Rust unit test: assert the correct `wgpu::BlendState` is selected for an additive-blend raster batch vs. a default batch
- [ ] Implement; make test pass
- [ ] Verify: `source ~/.cargo/env && cargo test -p ferrum-wasm --lib`

## 6. Acceptance checks

- `source ~/.cargo/env && cargo test -p ferrum-wasm --lib` — all pass
- `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — pre-existing `toggle_point` warning only; no new warnings
- Unit test asserts `circle_instance.stroke_opacity == col_value` for each row
- Unit test asserts additive blend state selected for `blend=additive` raster batch
- Python: `Chart(df).mark_point().encode(stroke_opacity="val").show_svg()` does not emit a `UserWarning` (after Task 5)

## 7. Open questions

- None — `stroke_dash` palette is now specified in spec §6. The four-entry palette may be expanded to 8 without a spec revision (spec §11).
