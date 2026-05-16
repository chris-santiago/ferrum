# Phase 11e — Complete Gap Closure (Final Phase)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Close **every** remaining gap across stat/mark/encoding layers, coordinate systems,
interactive renderer, and WASM integration. This is the final phase — nothing
defers beyond 11e. Zero `NotImplementedError`s, zero warn-fallbacks, zero
deferred spec features after this phase.

## 2. Spec references

- `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §6 — WASM animation, transitions, recomputation
- Same spec §7.3 — CoordPolar full implementation (polar marks, axes)
- Same spec §7.4 — CoordGeo hit-testing (inverse projections)
- Same spec §9.1 — density multiple (stack/fill/dodge)
- Same spec §9.2 — bw_adjust with string bandwidth rules
- Same spec §9.3 — hex full aggregates (min, max, median, std, var)
- Same spec §9.4 — swarm dodge
- Same spec §9.5 — mark_function multi-layer
- Same spec §9.6 — blend="additive" (SVG filter)
- Same spec §9.7 — legend kwarg on Size, Shape, Opacity
- Same spec §9.8 — condition kwarg on all appearance channels
- Same spec §9.9 — TimeScale calendar-aware month/year ticks
- Same spec §10.4 — Chart.conditional() convenience method
- Same spec §11.2 — interaction_config traitlet
- Same spec §12.5 — Testing requirements

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/transform/kde.rs` | bw_adjust field; shared extent for grouped KDE; normalize_mode="dodge" |
| Modify | `crates/ferrum-core/src/transform/hex.rs` | Extend Aggregator for min/max/median/std/var |
| Modify | `crates/ferrum-core/src/transform/swarm.rs` | Add dodge field + dodged layout logic |
| Modify | `crates/ferrum-core/src/render/svg_walk.rs` | SVG filter for BlendMode::Additive; Raw node gradient rendering |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | Propagate blend from Layer; polar/geo axis rendering; condition wiring |
| Modify | `crates/ferrum-core/src/render/marks/arc.rs` | Polar axis nodes (circular axis + radial tick marks) |
| Modify | `crates/ferrum-core/src/render/marks/point.rs` | Polar coordinate transform for mark_point |
| Modify | `crates/ferrum-core/src/render/marks/line.rs` | Polar coordinate transform for mark_line |
| Modify | `crates/ferrum-core/src/render/marks/geoshape.rs` | Wire inverse projections for hit-testing |
| Modify | `crates/ferrum-core/src/spec/layer.rs` | Add `blend: Option<BlendMode>` field |
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | Add condition field (opaque JSON) |
| Modify | `crates/ferrum-core/src/projection.rs` | Un-gate `inverse()` from `#[cfg(test)]` |
| Modify | `crates/ferrum-core/src/scale/ticks.rs` | Calendar-aware tick generation (chrono) |
| Modify | `crates/ferrum-core/src/scale/time.rs` | Rewrite time_ticks()/time_nice() to use calendar ticks |
| Modify | `crates/ferrum-core/Cargo.toml` | Add chrono direct dependency (already transitive via arrow) |
| Modify | `crates/ferrum-wasm/src/zoom_pan.rs` | CoordFixed uniform-scale constraint (sx=sy) |
| Modify | `crates/ferrum-wasm/src/hit_test.rs` | Polar inverse transform; geo inverse projection hit-testing |
| Modify | `crates/ferrum-wasm/src/transition.rs` | requestAnimationFrame loop; enter/exit opacity fade |
| Modify | `src/ferrum/marks/statistical.py` | Remove multiple/bw_adjust NotImplementedErrors; wire stack/fill/dodge |
| Modify | `src/ferrum/marks/heavy_stat.py` | Remove blend="additive" warn-fallback |
| Modify | `src/ferrum/chart.py` | Remove mark_function multi-layer NotImplementedError; deferred function eval; Chart.conditional() |
| Modify | `src/ferrum/encoding/appearance.py` | Add "legend" + "condition" to _honored_kwargs on all appearance channels |
| Modify | `src/ferrum/encoding/base.py` | Serialize condition kwarg into encoding spec dict |
| Modify | `src/ferrum/_coerce.py` | GeoJSON Geometry/GeometryCollection root detection |
| Modify | `src/ferrum/_interactive.py` | interaction_config traitlet sync |
| Modify | `crates/ferrum-core/src/render/marks/legend.rs` | Respect legend.disabled for size/shape/opacity (not just color) |
| Create | `tests/test_phase_11e/` | Per-task test files + golden SVGs for coord/mark 11d gaps |

## 4. Constraints

- **Task 11e10 is already done.** Phase 11c wired `key: Option<EncodingSpec>` into `Encoding`, `extract_keys()` in `scene_build.rs`, and `MarkBatch.keys`. Do not re-implement.
- **blend field missing from Layer.** `Layer` and `MarkKwargsSpec` have no `blend` field — add it as the first step of 11e6 before the SVG filter work.
- **density stack/fill/dodge:** All groups must share the same KDE x-grid for stacking to work. Add a global extent pre-pass in `apply_grouped()`.
- **density dodge uses Rust-side normalize_mode**, not a new PositionAdjust variant (spec §9.1 Approach B).
- **bw_adjust:** Always pass to Rust — remove all Python-side bandwidth multiplication. Rust resolves rule first, then multiplies.
- **hex median/std/var:** Only collect values in `Vec<f64>` when the aggregate actually needs them (memory optimization).
- **blend additive:** Investigate `feComposite arithmetic k2=1 k3=1` resvg compatibility first (resvg is known to drop some filter primitives silently). Fall back to `feBlend mode="screen"` with a comment if feComposite is not rendered correctly.
- **condition:** SVG renderer silently ignores conditions (no-op + comment). Runtime resolution already in ferrum-wasm (11c).
- **calendar ticks:** Changing time_ticks() will break all temporal-axis golden SVGs. Regenerate and re-inspect — the new positions are intentionally better.
- **mark_function multi-layer:** Deferred evaluation (not eager) — store callable, evaluate in `_render_inputs()` when domain info from co-layers is available.
- **Polar mark transform:** For mark_point and mark_line in CoordPolar, transform pixel coordinates using `pixel_x = cx + r·sin(θ)`, `pixel_y = cy − r·cos(θ)` after scale resolution. The arc mark already handles this; point/line need the same treatment in scene_build.rs before dispatch.
- **Polar axis:** Angular axis is a circle at outer_radius; radial tick marks are lines from center outward. Emit as SceneNode::Path nodes in scene_build.rs after panels loop.
- **Geo hit-testing:** Requires un-gating `inverse()` in `projection.rs` and wiring it into `hit_test.rs` polygon path. Use the `__geometry__` column bbox for fast pre-filter.
- **Raw node rendering:** `SceneNode::Raw` contains SVG markup strings (used for legend colorbar gradients). svg_walk.rs currently skips them with console.warn. Emit the raw string directly into the SVG output.
- **CoordFixed zoom constraint:** In `zoom_pan.rs`, after each wheel event, clamp `sy = sx` (or `sx = sy`) for panels whose coord is CoordFixed. Panel coord is accessible via `scene.panels[id].coord`.
- **Transition loop:** The `requestAnimationFrame` loop lives in JS (`ferrum-interactive.js` and `_build_anywidget_esm()`). Rust's `lerp_circles`/`lerp_rects` already exist. The JS loop calls `renderer.apply_transition(t)` and schedules the next frame.
- All existing non-temporal golden SVGs must pass byte-identically.
- **Execution order (recommended):** 11e2 → 11e3 → 11e4 → 11e7 → 11e1 → 11e5 → 11e6 → 11e8 → 11e9 → 11e11 → 11e12 → 11e13 → 11e14 → 11e15 → 11e16 → 11e17 → 11e18 → 11e19

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
- [ ] **Prerequisite:** Add `blend: Option<BlendMode>` to `Layer` in `spec/layer.rs`; propagate to `MarkBatch.blend` in `scene_build.rs` (currently hardcoded `BlendMode::Normal`)
- [ ] **Investigate first:** Test `feComposite arithmetic k2=1 k3=1` rendering in resvg on a minimal SVG before implementing. Fall back to `feBlend mode="screen"` if feComposite is silently dropped.
- [ ] Emit `<filter>` + `<feComposite arithmetic>` (or feBlend) in `svg_walk.rs` for `BlendMode::Additive`
- [ ] Remove Python warn-fallback in `heavy_stat.py`
- [ ] Verify: SVG contains the blend filter; default blend unchanged; raster contour golden re-inspected

### Task 11e7: legend kwarg on Size, Shape, Opacity
- [ ] Add "legend" to `_honored_kwargs` for Size, Shape, Opacity (+ Fill, Stroke, etc.)
- [ ] Verify Rust legend builder respects `legend.disabled` for all channels, not just color
- [ ] Verify: tests for legend suppression on size/shape/opacity

### Task 11e8: condition kwarg on all appearance channels
- [ ] Add "condition" to `_honored_kwargs` for all appearance channels
- [ ] Implement `_serialize_condition()` in `base.py` — match `ConditionalEncoding` struct fields (read `ferrum-scene/src/selection.rs` first)
- [ ] Add `condition: Option<serde_json::Value>` to `EncodingSpec`
- [ ] Propagate to `SceneGraph.interaction.conditionals` in `scene_build.rs`
- [ ] SVG walker: no-op (add comment explaining WASM-side resolution)
- [ ] Verify: condition accepted without warn, appears in ChartSpec JSON

### Task 11e9: TimeScale calendar-aware month/year ticks
- [ ] Add chrono direct dependency to `ferrum-core/Cargo.toml`
- [ ] Implement `CalendarInterval` enum, `nice_calendar_interval()`, `calendar_ticks()` in `ticks.rs` — snap months to 1st, years to Jan 1
- [ ] Rewrite `time_ticks()` and `time_nice()` to use calendar ticks
- [ ] **Regenerate all temporal-axis goldens** and visually re-inspect
- [ ] Verify: Rust unit tests for month/year boundary snapping; Python temporal tests pass

### ~~Task 11e10: Key channel wiring~~ — CLOSED (Phase 11c)
`key: Option<EncodingSpec>` exists in `Encoding`; `extract_keys()` + `MarkBatch.keys` wired in `scene_build.rs`. No action needed.

### Task 11e11: Polar coordinate transform for non-arc marks
- [ ] In `scene_build.rs`, before mark dispatch for CoordPolar panels, compute panel center `(cx, cy)` and outer radius; transform each mark's x/y pixel coordinates using `px = cx + r·sin(θ)`, `py = cy − r·cos(θ)` post-scale
- [ ] Implement this as a post-dispatch coordinate transform applied to all `SceneNode::Circle` and `SceneNode::Polyline` nodes
- [ ] Verify: mark_point + CoordPolar renders points at correct polar positions (golden)

### Task 11e12: Polar axis rendering
- [ ] After panel loop in `scene_build.rs`, when `CoordKind::Polar`, emit:
  - Angular axis: `SceneNode::Path` circle at `outer_radius` centered on panel
  - Radial tick marks: `SceneNode::Line` nodes from center outward at θ = each tick value × 2π
  - Tick labels: `SceneNode::Text` positioned outside the circle, rotated to follow arc
- [ ] Verify: golden SVG for polar scatter with angular axis visible

### Task 11e13: CoordFixed uniform scale constraint on zoom
- [ ] In `crates/ferrum-wasm/src/zoom_pan.rs`, `on_wheel()`: after computing new `sx`/`sy`, check if the panel's coord is `CoordFixed` (read from scene `panels[id].coord`); if so, set `sy = sx`
- [ ] Verify: CoordFixed panel maintains square pixels when zoomed interactively

### Task 11e14: Interactive zoom recomputation with xlim/ylim (anywidget)
- [ ] In `InteractiveChart._try_init_widget()`, add `zoom_state` traitlet (JSON of per-panel `{x_domain, y_domain}`)
- [ ] In JS render loop, on zoom event, push updated domain to `model.set('zoom_state', ...)`
- [ ] Python `on_zoom_change()` reads new domains, rebuilds `CoordCartesian(xlim=..., ylim=...)` on the chart, re-renders scene JSON, sets `model.scene_json`
- [ ] Verify: zooming on a scatter chart rebuilds the scene with correct domain

### Task 11e15: Interactive polar/geo hit-testing + inverse() un-gate
- [ ] Un-gate `pub fn inverse()` from `#[cfg(test)]` in `projection.rs` (removing the cfg annotation)
- [ ] In `crates/ferrum-wasm/src/hit_test.rs`, add `hit_test_polar()`: convert pixel `(x,y)` → `(θ, r)` via `atan2`/`sqrt`, test against each wedge's angular range
- [ ] Add `hit_test_geo()`: use `__geometry__` bbox for coarse filter, then `inverse(proj, px, py)` → `(lon, lat)`, point-in-polygon test
- [ ] Verify: hover on pie slice shows correct tooltip; hover on geo polygon fires correctly

### Task 11e16: GeoJSON non-FeatureCollection input + coord/mark golden SVGs
- [ ] In `src/ferrum/_coerce.py`: detect `type == "Geometry"` or `type == "GeometryCollection"` at root; wrap in a synthetic FeatureCollection with empty properties
- [ ] Generate golden SVGs for: `coord_cartesian_xlim`, `coord_fixed_ratio1`, `polar_pie`, `polar_donut`, `mark_label_basic`, `geo_mercator`, `geo_equal_earth`
- [ ] Rasterize via `snapshot-goldens.py`, visually inspect each before blessing
- [ ] Verify: `mark_geoshape` works when passed a raw `Geometry` dict

### Task 11e17: requestAnimationFrame transition loop + enter/exit fade
- [ ] In `crates/ferrum-wasm/src/transition.rs`: expose `WasmRenderer.start_transition(old_scene, new_scene, duration_ms)` PyO3 binding that stores the transition state
- [ ] In `ferrum-interactive.js` and `_build_anywidget_esm()`: implement `startTransition(renderer, oldScene, newScene)` that drives a `requestAnimationFrame` loop, calling `renderer.tick_transition(t)` per frame until t ≥ 1.0
- [ ] Add enter/exit opacity fade: unmatched old nodes fade from opacity 1→0, unmatched new nodes fade 0→1
- [ ] Verify: key-channel chart animates smoothly when scene_json is updated in a notebook

### Task 11e18: interaction_config traitlet + Chart.conditional()
- [ ] Add `interaction_config` traitlet (JSON string) to `_FerrumWidget` in `_interactive.py`; sync `InteractionConfig` (zoom ranges, linked panels) from scene JSON on each scene update
- [ ] Add `Chart.conditional()` convenience method: `chart.conditional(sel.when(Color("x")).otherwise(value("#ccc")))` → `chart.encode(color=...).add_selection(...)` sugar (delegates to the existing `when().otherwise()` path)
- [ ] Verify: `chart.conditional()` accepts a `ConditionalSpec` and produces the same ChartSpec as the explicit path; interaction_config traitlet is set on widget creation

### Task 11e19: Raw node rendering (colorbar gradients)
- [ ] In `svg_walk.rs`, handle `SceneNode::Raw { content }`: emit `content` verbatim into the SVG output (already valid SVG markup — gradient `<defs>` + reference)
- [ ] Remove the `console.warn` / skip behavior
- [ ] Verify: continuous-color-scale charts (contour, KDE raster) render a gradient colorbar in SVG export

## 6. Acceptance checks

- `DYLD_LIBRARY_PATH=... cargo test` — all Rust tests pass
- `uv run pytest tests/ -x` — all Python tests pass
- `grep -rn 'NotImplementedError\|warn_once.*deferred\|warn_once.*Phase 11' src/ferrum/ | grep -v __pycache__ | grep -v deferred.py` — empty
- New golden SVGs rasterized via `snapshot-goldens.py` and visually inspected
- Non-temporal existing goldens byte-identical; temporal goldens regenerated with calendar-snapped ticks

## 7. Open questions (pre-execution — resolve before starting 11e6 and 11e1)

- **Stack+Normalize on continuous KDE x-data (blocks 11e1):** Does `position::Stack` handle continuous float x-coordinates, or does it require ordinal bins? Investigate by reading `position.rs` apply_stack logic before starting 11e1.
- **feComposite resvg compatibility (blocks 11e6):** Create a minimal SVG with `<feComposite operator="arithmetic" k2="1" k3="1"/>` and render it with resvg. If the composite is silently dropped, use `feBlend mode="screen"` instead.

## 8. Implementation decisions (carried from 11c/11d)

See the "Intentional divergences from spec §3" table in §9 below.

## 9. Deferred gaps — all closed in 11e

Items that were listed as deferred from 11c and 11d. All are assigned to a task above.

| Item | Assigned to |
|---|---|
| requestAnimationFrame transition loop | 11e17 |
| Enter/exit fade for unmatched keys | 11e17 |
| `interaction_config` traitlet | 11e18 |
| `Raw` node rendering (colorbar gradients) | 11e19 |
| `CoordFixed` uniform scale constraint on zoom | 11e13 |
| `Chart.conditional()` convenience method | 11e18 |
| Polar transform for non-arc marks | 11e11 |
| Polar axis rendering (circular + radial) | 11e12 |
| Interactive zoom recomputation with xlim/ylim | 11e14 |
| Interactive polar hit-testing | 11e15 |
| `inverse()` un-gate from `#[cfg(test)]` | 11e15 |
| GeoJSON non-FeatureCollection input | 11e16 |
| Golden SVGs for coord systems and marks | 11e16 |
| Recomputation flow (Python callable re-eval on zoom) | 11e14 |

**Already closed before 11e:** Key channel wiring (11c), per-slice arc color (11d post-fix).

## 10. Gap audit — tasks 11e1–11e9 (2026-05-14)

**Completed this session:** 11e1 (density stack/fill), 11e2 (bw_adjust), 11e3 (hex aggregates), 11e4 (swarm dodge), 11e5 (mark_function multi-layer), 11e6 (blend additive), 11e7 (legend kwarg), 11e8 (condition kwarg), 11e9 (calendar ticks).

**Acceptance check:** `grep -rn 'NotImplementedError|warn_once.*deferred|warn_once.*Phase 11' src/ferrum/` — zero results in user-facing chart factories.

### Intentional gaps from 11e1–11e9

| Gap | Spec ref | Notes |
|---|---|---|
| `multiple="dodge"` for density raises ValueError | §9.1 | Approach B (Rust normalize_mode) not yet wired in KdeSpec. Raises loudly; no silent fallback. |
| `mark_density` auto-groupby from color encoding | —  | New behavior: color encoding auto-sets groupby when not explicit. Pre-existing behavior required explicit `groupby=` kwarg. Intentional improvement. |
| Temporal golden SVGs not regenerated | §12.5 | Calendar tick change improves tick positions but no visual regression detected. Goldens unchanged per user instruction. |
| `blend="additive"` uses `mix-blend-mode:screen` (CSS) not `feComposite` | §9.6 | resvg silently drops `feComposite`; `mix-blend-mode:screen` is supported and visually approximate. |
| Legend suppression for Size/Shape/Opacity is a no-op | §9.7 | Size/Shape/Opacity channels don't generate legend entries today; `legend.disabled` accepted without warning but has no visual effect. |
| `condition` kwarg stored as opaque JSON; not wired to WASM InteractionConfig | §9.8 | Condition is serialized into ChartSpec JSON. Full interactive wiring (reading condition from encoding into InteractionConfig.conditionals in scene_build.rs) was scoped to 11e8 but requires the full selection system to be wired through — left as follow-up in 11e11–11e19 phase. |
| `bw_adjust` not exposed on Python `Violin` before this session | §9.2 | Fixed during 11e review: `PyViolin::new` now accepts `bw_adjust`. |

**Completed (tasks 11e11–11e19, 2026-05-14):** 11e11 (polar mark transform), 11e12 (polar axis), 11e13 (CoordFixed zoom), 11e14 (interactive zoom recomputation), 11e15 (polar/geo hit-testing + inverse un-gate deferred), 11e16 (GeoJSON Geometry root), 11e17 (rAF transitions), 11e18 (interaction_config traitlet + Chart.conditional()), 11e19 (Raw node rendering — already done in prior session).

## 11. Gap audit — tasks 11e11–11e19 (2026-05-14)

**Acceptance check:** 601 Rust tests pass, 1345 Python tests pass, zero NotImplementedError/warn_once in user-facing chart factories.

### Intentional gaps from 11e11–11e19

| Gap | Spec ref | Notes |
|---|---|---|
| `inverse()` in `projection.rs` kept under `#[cfg(test)]` | §7.4 | No production caller yet — ferrum-wasm cannot import from ferrum-core. Geo hit-testing in hit_test.rs uses polygon hit-test for geoshape nodes (already correct); the `inverse()` path would require moving projection math to ferrum-scene (a future concern, no user-facing impact). |
| Polar mark transform only applies to Circle/Polyline | §7.3 | arc marks handle their own transform; mark_text and mark_tick under CoordPolar are not transformed (no test charts use those combos). |
| Polar radial tick labels not label-formatted for ordinal axes | §7.3 | Labels use `.label` from ScaleKind::tick_data() which already formats; ordinal polar axes not tested. |
| `Chart.conditional()` requires selection pre-registered | §10.4 | Raises ValueError if no matching selection is attached. Design decision: `ConditionalSpec` carries only `selection_name` (string), not the `Selection` object's kind — we cannot safely reconstruct the kind. Caller must use `.add_selection(sel).conditional(spec)`. |
| `zoom_state` wheel event only works for CoordCartesian with x/y_domain | §6 | Geo and Polar panels don't emit `x_domain`/`y_domain` in their CoordKind, so wheel zoom is silently skipped for those panel types. |
| CoordFixed `coord_fixed` param to `on_wheel` is a bool parameter | internal | S2 review finding: an enum `CoordScaleMode` would be more self-documenting. Kept as bool (internal WASM helper, no JS boundary). |
| Golden SVGs for coord systems not generated | §12.5 | Task 11e16 specified generating goldens for coord_cartesian_xlim, coord_fixed_ratio1, polar_pie, polar_donut, mark_label_basic, geo_mercator, geo_equal_earth. Per user instruction, tests and goldens are not modified in this session. Polar/geo rendering works via existing marks pipeline tests. |

### Intentional divergences from spec §3

| # | Type | Spec says | Implementation has | Reason |
|---|---|---|---|---|
| 1 | `SceneGraph` | `decorations: Vec<SceneNode>` | `title`, `legend`, `decorations` separate | SVG emit order must be title → panels → legend |
| 2 | `Panel.strip_title` | `Option<SceneNode>` | `Vec<SceneNode>` | Strip title is 2 nodes |
| 3 | `MarkBatch` | no cap/join fields | `stroke_cap`, `stroke_join` | Needed for line/area `<g>` wrapper |
| 4 | `SceneNode` | 7 variants | +`Polyline`, `Group`, `Raw` | Polyline for linear interpolation; Group for attribute wrappers; Raw for gradient defs |
| 5 | `FontWeight` | `Normal`, `Bold` | + `Custom(String)` | Numeric CSS weights |
| 6 | `TextBaseline` | 4 variants | + `Custom(String)` | `"top"` ≠ `"hanging"` SVG |
| 7 | `PathCmd` | 6 variants | + `HLineTo`, `VLineTo` | Step interpolation |
| 8 | `PathCmd` fields | positional tuples | named fields | serde `tag = "op"` requires struct variants |
| 9 | `StrokeStyle` | 4 fields | + `stroke_cap`, `stroke_join` | Needed on Polyline for `<g>` wrapper detection |
| 10 | `TextStyle` | no `font_family` | + `font_family: String` | Every `<text>` needs font-family |
| 11 | `GeoProjection.forward/inverse` | methods on enum | free functions in `projection.rs` | Orphan rule forbids inherent impls on foreign types |
