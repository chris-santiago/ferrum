# Axis Label Layout Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Replace the flat→-45°→elide x-axis collision policy with a graduated cascade (wrap → shrink → rotate → cull → elide), fix y-axis title duplication in facets, and make the bottom margin rotation-aware.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-23-axis-label-layout-overhaul-design.md` — full spec (§4.1–§4.8 behavior, §6 interfaces, §7 invariants)
- `design-docs/superpowers/specs/2026-05-09-layout-engine-design.md §6` — single-pass commitment (bounded two-pass margin estimation is acceptable)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/layout/axis.rs` | Collision cascade, wrap_label(), graduated rotation, tick culling, extended TickLayout |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | Dynamic bottom margin, y-axis title suppression, new constants, cull_threshold in ThemeInputs |
| Modify | `crates/ferrum-core/src/layout/text_metrics.rs` | measure_multiline_width() helper |
| Modify | `crates/ferrum-core/src/render/marks/axis.rs` | Render multi-line labels (split on `\n`), respect per-tick font_size and culled flag |
| Modify | `crates/ferrum-wasm/src/text_json.rs` | Handle `\n` in tick label content for WASM text layer |
| Modify | `crates/ferrum-core/src/render/binding.rs` | Add `cull_threshold` to ThemeOverridesSpec + apply_theme_overrides |
| Modify | `src/ferrum/themes/__init__.py` | Add `cull_threshold` parameter to Theme class |
| Test | `crates/ferrum-core/src/layout/axis.rs` (inline tests) | Per-stage cascade tests with MockMetrics |
| Test | `tests/test_axis_label_overhaul.py` | Python integration: 9-category bar, faceted y-title, no-elision check |

## 4. Constraints

- **Single-pass plot rect.** The collision cascade in step 7 mutates labels/angle/visibility only — never panel geometry. Bottom margin uses worst-case estimation before panels are computed.
- **Backward compatibility.** Charts with no label collision must produce byte-identical TickLayout output (angle=0, no wrapping, no font change, culled=false, label_font_size=None).
- **`label_angle_override` bypasses cascade.** When `Axis(label_angle=...)` is set, skip S1-S4 entirely — apply that angle, then elide if still colliding (current behavior).
- **Wrapping only at 0°.** Multi-line labels are never rotated. If wrapping doesn't resolve collision, rotation uses original unwrapped labels.
- **New TickLayout fields must have serde defaults** so existing serialized LayoutResults deserialize without breaking.

## 5. Tasks

### Task 1: TickLayout + constants (Rust — foundation)
- [ ] Add `culled: bool`, `label_font_size: Option<f64>` to `TickLayout` with `#[serde(default)]` (spec §6.1)
- [ ] Add constants to `mod.rs`: `ANGLE_CASCADE`, `FONT_SHRINK_FACTOR`, `DEFAULT_CULL_THRESHOLD` (spec §6.3)
- [ ] Add `cull_threshold: u32` to `ThemeInputs` with default 8
- [ ] Update all existing `TickLayout` construction sites (axis.rs tests, render/marks/axis.rs tests) to include new fields
- [ ] Verify: `cargo test -p ferrum-core` — no regressions

### Task 2: Multi-line wrapping (Rust — axis.rs)
- [ ] Add `wrap_label(label, slot_w, font_size, metrics) -> Option<String>` — splits at `_` > space > camelCase, returns `\n`-joined string or None if unwrappable (spec §4.2)
- [ ] Add `measure_multiline_width(text, font_size, metrics) -> f64` helper to `text_metrics.rs` — split on `\n`, return max line width
- [ ] Write tests: underscore split, space greedy-fill, camelCase split, no-break-point passthrough, unlimited lines for `very_long_snake_case_name`
- [ ] Verify: `cargo test -p ferrum-core` — new tests pass

### Task 3: Collision cascade (Rust — axis.rs)
- [ ] Implement `cascade_collision_recovery()` with stages S0-S5 per spec §4.1, returning `CascadeResult` (spec §6.2)
- [ ] Replace the current collision logic in `layout_x_axis()` with a call to `cascade_collision_recovery()`
- [ ] Preserve `label_angle_override` bypass: when set, skip cascade, apply override angle + elision only
- [ ] Write tests with MockMetrics for each cascade stage: S0 flat, S1 wrap resolves, S2 font shrink resolves, S3 graduated rotation picks correct angle, S4 culling at stride N, S5 elision last resort
- [ ] Write parameterized sweep test: 4→40 labels in 600px panel, verify cascade degrades gracefully
- [ ] Verify: `cargo test -p ferrum-core` — all pass, no LabelsElided warning for ≤9 snake_case labels in 600px

### Task 4: Y-axis title suppression (Rust — mod.rs)
- [ ] Compute `min_col` from `panel_rects` (spec §4.7)
- [ ] In the per-panel loop, suppress y-axis title when `col > min_col && spec.facet.is_some()` (mirror the existing x-axis `row < max_row` suppression)
- [ ] Write test: 2×2 faceted chart → only 2 y-axis titles emitted (one per row, leftmost column only)
- [ ] Verify: `cargo test -p ferrum-core`

### Task 5: Dynamic bottom margin (Rust — mod.rs)
- [ ] Replace `x_label_band = metrics.line_height(theme.label_font_size)` with a rotation-aware estimate (spec §4.8)
- [ ] Use worst-case label width + estimated slot width to pre-run cascade stages S1-S3 and determine probable angle/wrapping
- [ ] Reserve: wrapping → `max_lines × line_height`; rotation → `max_w × |sin(θ)| + line_h × |cos(θ)|`; flat → `line_height`
- [ ] Write test: rotated labels at -45° → bottom margin > 13px; wrapped 2-line labels → margin ≈ 2×line_height
- [ ] Verify: `cargo test -p ferrum-core`

### Task 6: SVG renderer updates (Rust — render/marks/axis.rs)
- [ ] When `tick.culled == true`, emit tick mark but skip label node
- [ ] When `tick.label` contains `\n`, emit one `SceneNode::Text` per line with stacked y-offsets (each offset by `line_height`)
- [ ] When `tick.label_font_size` is `Some(fs)`, use `fs` instead of `theme.label_font_size` for that tick's text style
- [ ] Update the existing test `axis_builds_line_ticks_and_title` to pass with new TickLayout fields
- [ ] Write test: multi-line label produces multiple text nodes; culled tick produces line but no text
- [ ] Verify: `cargo test -p ferrum-core`

### Task 7: WASM text layer (Rust — ferrum-wasm)
- [ ] In `text_json.rs`, handle `\n` in tick label content — split into multiple `TextElement` entries with stacked y coordinates
- [ ] Respect `label_font_size` override in the text element's font_size field
- [ ] Verify: `cargo test -p ferrum-wasm` (if applicable) or `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 8: Python theme bridge
- [ ] Add `cull_threshold` to `ThemeOverridesSpec` in `render/binding.rs` and wire in `apply_theme_overrides`
- [ ] Add `cull_threshold: int` parameter to `Theme.__init__()` in `src/ferrum/themes/__init__.py`, include in `to_spec_dict()`
- [ ] Verify: `uv run pytest tests/ -k theme` — passes

### Task 9: Integration tests + golden refresh
- [ ] Write `tests/test_axis_label_overhaul.py`: 600×400 bar chart with 9 snake_case categories → no LabelsElided warning, all labels fully readable
- [ ] Write test: faceted chart with 2+ columns → scene graph has exactly 1 y-axis title text node
- [ ] Write test: rotated chart → no label clipping (bottom margin accommodates extent)
- [ ] Rebuild Rust extension: `unset CONDA_PREFIX && uv run --no-sync maturin develop`
- [ ] Run full suite: `uv run pytest -n auto` — all pass
- [ ] Regenerate any affected golden SVGs, rasterize with `snapshot-goldens.py`, visually inspect
- [ ] Verify: existing goldens that don't trigger collision are byte-identical

## 6. Acceptance checks

- `cargo test -p ferrum-core` — all pass (including new cascade, wrapping, y-title suppression, margin tests)
- `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- `unset CONDA_PREFIX && uv run --no-sync maturin develop` — builds
- `uv run pytest -n auto` — all pass
- 9-category snake_case bar chart renders legible labels without elision at 600×400
- Faceted chart shows exactly one y-axis title (leftmost column)
- Rotated labels are not clipped at viewport boundary

## 7. Open questions

None.
