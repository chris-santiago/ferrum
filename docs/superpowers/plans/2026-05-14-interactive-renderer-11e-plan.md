# Phase 11e — Stat/Mark/Encoding Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Close every remaining `NotImplementedError`, warn-fallback, and feature gap in the stat, mark, and encoding layers. Zero deferred stat/mark/encoding features after this phase.

## 2. Spec references

- `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §9.1 — density multiple (stack/fill/dodge)
- Same spec §9.2 — bw_adjust with string bandwidth rules
- Same spec §9.3 — hex full aggregates (min, max, median, std, var)
- Same spec §9.4 — swarm dodge
- Same spec §9.5 — mark_function multi-layer
- Same spec §9.6 — blend="additive" (SVG filter)
- Same spec §9.7 — legend kwarg on Size, Shape, Opacity
- Same spec §9.8 — condition kwarg on all appearance channels
- Same spec §9.9 — TimeScale calendar-aware month/year ticks
- Same spec §9.10 — Key channel wiring
- Same spec §12.5 — Testing requirements

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/transform/kde.rs` | bw_adjust field; shared extent for grouped KDE; normalize_mode="dodge" |
| Modify | `crates/ferrum-core/src/transform/hex.rs` | Extend Aggregator for min/max/median/std/var |
| Modify | `crates/ferrum-core/src/transform/swarm.rs` | Add dodge field + dodged layout logic |
| Modify | `crates/ferrum-core/src/render/svg_walk.rs` | SVG filter for BlendMode::Additive |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | Propagate blend, conditional encodings, Key→MarkBatch.keys |
| Modify | `crates/ferrum-core/src/scale/ticks.rs` | Calendar-aware tick generation (chrono) |
| Modify | `crates/ferrum-core/src/scale/time.rs` | Rewrite time_ticks()/time_nice() to use calendar ticks |
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | Add condition field (opaque JSON), add key field |
| Modify | `crates/ferrum-core/src/render/position.rs` | Continuous-x normalization for density fill (if needed — investigate first) |
| Modify | `crates/ferrum-core/Cargo.toml` | Add chrono direct dependency (already transitive via arrow) |
| Modify | `src/ferrum/marks/statistical.py` | Remove multiple/bw_adjust NotImplementedErrors; wire stack/fill/dodge |
| Modify | `src/ferrum/marks/heavy_stat.py` | Remove blend="additive" warn-fallback |
| Modify | `src/ferrum/chart.py` | Remove mark_function multi-layer NotImplementedError; deferred function eval |
| Modify | `src/ferrum/encoding/appearance.py` | Add "legend" + "condition" to _honored_kwargs on all appearance channels |
| Modify | `src/ferrum/encoding/base.py` | Serialize condition kwarg into encoding spec dict |
| Modify | `crates/ferrum-core/src/render/marks/legend.rs` | Respect legend.disabled for size/shape/opacity (not just color) |
| Create | `tests/test_phase_11e_*.py` | Per-task test files (8 total) |

## 4. Constraints

- **density stack/fill/dodge:** All groups must share the same KDE x-grid for stacking to work. Add a global extent pre-pass in `apply_grouped()`.
- **density dodge uses Rust-side normalize_mode**, not a new PositionAdjust variant (spec §9.1 Approach B).
- **bw_adjust:** Always pass to Rust — remove all Python-side bandwidth multiplication. Rust resolves rule first, then multiplies.
- **hex median/std/var:** Only collect values in `Vec<f64>` when the aggregate actually needs them (memory optimization).
- **blend additive:** Use `feComposite arithmetic k2=1 k3=1`, NOT `mix-blend-mode: screen` — they produce visibly different results. Check resvg compatibility.
- **condition:** SVG renderer silently ignores conditions. Runtime resolution is in ferrum-wasm (11c dependency).
- **calendar ticks:** Changing time_ticks() will break all temporal-axis golden SVGs. Regenerate and re-inspect — the new positions are intentionally better.
- **mark_function multi-layer:** Deferred evaluation (not eager) — store callable, evaluate in `_render_inputs()` when domain info from co-layers is available.
- All existing non-temporal golden SVGs must pass byte-identically.
- **Calendar ticks backward compat:** keep existing `nice_time_interval_ms()` (may be called from other modules); new code path uses `nice_calendar_interval()` + `calendar_ticks()`.
- **Execution order matters:** recommended 11e2→11e3→11e4→11e7→11e1→11e5→11e6→11e8→11e9→11e10. Coordination: 11e1/11e2 share KdeSpec changes, 11e7/11e8 share `_honored_kwargs` in `appearance.py`, 11e8/11e10 both modify `scene_build.rs`.

## 5. Tasks

### Task 11e1: mark_density(multiple="stack"|"fill"|"dodge")
- [ ] **Investigate first:** Read `position.rs` to check if Stack+Normalize handles continuous x. Read `_set_composite_mark()` / `_resolve_pending()` call chain for return tuple shapes.
- [ ] Add global extent pre-pass to KDE `apply_grouped()` so all groups share same x-grid
- [ ] Wire "stack" → Stack(offset="zero"), "fill" → Stack(offset="normalize")
- [ ] Wire "dodge" → KdeSpec normalize_mode="dodge" (Rust-side per-group scaling)
- [ ] Verify: golden tests for stack/fill/dodge + regression test for multiple="layer"

### Task 11e2: mark_density(bw_adjust=) with string bandwidth rules
- [ ] Add `bw_adjust: f64` to KdeSpec (default 1.0); apply `h *= bw_adjust` after rule resolution
- [ ] Remove Python NotImplementedError; always pass bw_adjust through to Rust
- [ ] Verify: tests for bw_adjust with scott, silverman, and numeric bandwidth

### Task 11e3: mark_hex full aggregates
- [ ] Extend Aggregator with min/max/sum_sq/values fields; `push(v, needs_values)` optimization
- [ ] Update validation, accumulation loop, and finalization to support all 8 aggregates
- [ ] Verify: Python tests for each new aggregate + Rust unit tests for median/std

### Task 11e4: mark_swarm(dodge=...)
- [ ] Add `dodge: Option<String>` to SwarmSpec
- [ ] Implement `apply_dodged()`: partition by (category, dodge_field), swarm per sub-group, offset cross-axis
- [ ] Wire dodge kwarg through Python mark_swarm
- [ ] Verify: golden test for grouped swarm + regression for no-dodge

### Task 11e5: mark_function multi-layer
- [ ] Remove NotImplementedError; store callable+params on the layer (deferred eval)
- [ ] Evaluate in `_render_inputs()`: infer domain from co-layers if None, linspace→fn→pyarrow table, inject as named data source
- [ ] Verify: function overlay on scatter, explicit domain, standalone regression

### Task 11e6: blend="additive"
- [ ] Propagate blend from spec/layer to MarkBatch.blend in scene_build.rs
- [ ] Emit `<filter>` with `<feComposite arithmetic>` in svg_walk.rs for Additive
- [ ] Remove Python warn-fallback in heavy_stat.py
- [ ] Verify: SVG contains feComposite; default blend unchanged

### Task 11e7: legend kwarg on Size, Shape, Opacity
- [ ] Add "legend" to _honored_kwargs for Size, Shape, Opacity (+ Fill, Stroke, etc.)
- [ ] Verify Rust legend builder respects legend.disabled for all channels, not just color
- [ ] Verify: tests for legend suppression on size/shape/opacity

### Task 11e8: condition kwarg on all appearance channels
- [ ] Add "condition" to _honored_kwargs for all appearance channels
- [ ] Implement `_serialize_condition()` in base.py — match ConditionalEncoding struct fields exactly (read ferrum-scene first)
- [ ] Add `condition: Option<serde_json::Value>` to EncodingSpec
- [ ] Propagate to SceneGraph InteractionConfig.conditionals in scene_build.rs
- [ ] SVG walker: no-op (add comment)
- [ ] Verify: condition accepted without warn, appears in ChartSpec JSON

### Task 11e9: TimeScale calendar-aware month/year ticks
- [ ] Add chrono direct dependency
- [ ] Implement `CalendarInterval` enum, `nice_calendar_interval()`, `calendar_ticks()` in ticks.rs — snap months to 1st, years to Jan 1
- [ ] Rewrite `time_ticks()` and `time_nice()` to use calendar ticks
- [ ] **Regenerate all temporal-axis goldens** and visually re-inspect
- [ ] Verify: Rust unit tests for month/year boundary snapping; Python temporal tests pass

### Task 11e10: Key channel wiring
- [ ] Add `key: Option<EncodingSpec>` to Encoding struct if not already present
- [ ] Verify Python Key channel flows into ChartSpec JSON
- [ ] Populate `MarkBatch.keys` from key column in scene_build.rs
- [ ] Verify: key in ChartSpec JSON, renders without error, absent by default

## 6. Acceptance checks

- `DYLD_LIBRARY_PATH=... cargo test` — all Rust tests pass
- `uv run pytest tests/ -x --timeout=120` — all Python tests pass
- `grep -rn 'NotImplementedError\|warn_once.*deferred\|warn_once.*Phase 11' src/ferrum/ | grep -v __pycache__ | grep -v deferred.py` — empty (zero remaining gaps)
- New golden SVGs rasterized and visually inspected
- Non-temporal existing goldens byte-identical; temporal goldens regenerated with calendar-snapped ticks

## 7. Open questions

- Does existing Stack+Normalize work on continuous KDE x-data, or does it need a continuous-x code path? (Task 11e1 investigation step — answer determines whether position.rs needs changes.)
- `feComposite` with `in2="BackgroundImage"` may need `enable-background="new"` on the SVG root for resvg. If not supported, fall back to `feBlend mode="screen"` with a comment noting approximation.

### Intentional divergences from spec §3 (required for byte-identical golden SVGs)

The spec's type definitions assumed a clean WASM-first design. The actual
implementation needed adjustments so the SVG walker (`svg_walk.rs`) could
reproduce the *exact* byte output of the old `render_svg` path. All changes
are additive — no spec fields were removed or renamed.

| # | Type | Spec says | Implementation has | Reason |
|---|---|---|---|---|
| 1 | `SceneGraph` | `decorations: Vec<SceneNode>` | `title: Vec<SceneNode>`, `legend: Vec<SceneNode>`, `decorations: Vec<SceneNode>` | Old `render_svg` emits title → panels → legend in that order. A single `decorations` vec loses this ordering, producing different SVG. |
| 2 | `Panel.strip_title` | `Option<SceneNode>` | `Vec<SceneNode>` | Strip title is 2 nodes (background rect + text). `Option<SceneNode>` forces a `Group` wrapper → extra `<g>` in SVG not present in old output. |
| 3 | `MarkBatch` | no cap/join fields | `stroke_cap: Option<StrokeCap>`, `stroke_join: Option<StrokeJoin>` | `mark_line` and `mark_area` wrap output in `<g stroke-linecap="..." stroke-linejoin="...">`. This is a batch-level attribute, not per-node. |
| 4 | `SceneNode` | 7 variants (Rect, Circle, Line, Path, Text, Image, Polygon) | +3 variants: `Polyline`, `Group`, `Raw` | `Polyline`: old `mark_line` emits `<polyline>` for linear interpolation, not `<path>`. `Group`: needed for `<g>` attribute wrappers. `Raw`: legend colorbar gradient `<defs>` can't be expressed as typed nodes (`fill="url(#...)"` is not a `Color`). |
| 5 | `FontWeight` | `Normal`, `Bold` | + `Custom(String)` | Themes use numeric CSS weights like `"600"` for axis titles. |
| 6 | `TextBaseline` | `Top`, `Middle`, `Bottom`, `Alphabetic` | + `Custom(String)` | `mark_text(baseline="top")` passes the user-facing string verbatim to SVG `dominant-baseline`; `"top"` ≠ `"hanging"` (the SVG-canonical name). |
| 7 | `PathCmd` | `MoveTo`, `LineTo`, `QuadTo`, `CubicTo`, `ArcTo`, `Close` | + `HLineTo`, `VLineTo` | Step interpolation in `mark_line` emits `H`/`V` SVG path commands. |
| 8 | `PathCmd` field style | positional tuples: `MoveTo(f64, f64)` | named fields: `MoveTo { x: f64, y: f64 }` | serde `#[serde(tag = "op")]` requires struct variants, not tuple variants. |
| 9 | `StrokeStyle` | `color`, `width`, `opacity`, `dash` | + `stroke_cap: Option<StrokeCap>`, `stroke_join: Option<StrokeJoin>` | Needed on `Polyline` nodes so the SVG walker can detect and emit the `<g>` wrapper. (Plan §"Type gaps" identified this pre-implementation.) |
| 10 | `TextStyle` | no `font_family` | + `font_family: String` | Every SVG `<text>` needs a `font-family` attribute. (Plan §"Type gaps" identified this pre-implementation.) |
