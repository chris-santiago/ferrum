# Phase 11c — Selections + Zoom/Pan + anywidget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Wire the full interaction system onto the SceneGraph IR (11a) and WASM renderer (11b): selection state machines, hit testing, conditional encoding resolution, zoom/pan, tooltips, href click-through, Python selection API, anywidget Jupyter integration, compound view scene graph merging, and animated transitions via Key channel.

## 2. Spec references

- `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §6.1–6.8 — Selection types, hit testing, state machine, conditional resolution, zoom/pan, tooltips, href
- Same spec §10.1–10.4 — Python selection API (`selection_point`, `selection_interval`, `Selection`, `SelectionMark`, conditional builder)
- Same spec §11.2 — anywidget integration, bidirectional state sync
- Same spec §12.3 — Interaction testing strategy

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `src/ferrum/selection.py` | Python selection API: constructors, Selection class, conditional builder |
| Create | `src/ferrum/_interactive.py` | InteractiveChart anywidget subclass, scene graph merging |
| Create | `crates/ferrum-wasm/src/hit_test.rs` | Per-mark-type hit testing (circle, rect, path, polygon, line) |
| Create | `crates/ferrum-wasm/src/selection_state.rs` | InteractionState, SelectionState, event dispatch |
| Create | `crates/ferrum-wasm/src/conditional.rs` | Conditional encoding resolution, GPU buffer updates |
| Create | `crates/ferrum-wasm/src/zoom_pan.rs` | Per-panel Affine2 transforms, tick level selection |
| Create | `crates/ferrum-wasm/src/transition.rs` | Key-channel diffing + lerp for animated transitions |
| Create | `tests/test_selection_api.py` | Python selection API serialization tests |
| Modify | `crates/ferrum-scene/src/selection.rs` | Add `FieldValue` type (spec §6.1, not implemented in 11a) |
| Modify | `crates/ferrum-core/src/spec/chart.rs` | Add `selections`, `conditionals` fields to ChartSpec |
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | Add `key` field to Encoding struct |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | Copy selections/conditionals into SceneGraph; wire Key → MarkBatch.keys |
| Modify | `crates/ferrum-wasm/src/lib.rs` | Register new modules, wire into WasmRenderer event loop |
| Modify | `src/ferrum/_wasm/ferrum-interactive.js` | Event capture, tooltip div, href navigation, anywidget model bridge |
| Modify | `src/ferrum/chart.py` | Wire `add_selection()`, `interactive()`, flow selections through `to_spec()` |
| Modify | `src/ferrum/encoding/base.py` | Wire `condition` kwarg validation and serialization |
| Modify | `src/ferrum/encoding/appearance.py` | Add `"condition"` to `_honored_kwargs` on all channels |
| Modify | `src/ferrum/__init__.py` | Export selection API and InteractiveChart |
| Modify | `pyproject.toml` | Add `anywidget>=0.9` to dependencies |

## 4. Constraints

- `FieldValue` must use a tagged serde enum (`#[serde(tag = "type", rename_all = "snake_case")]`), not `serde_json::Value` — exactly four variants: String, Number(f64), Bool, Null (spec §6.1)
- `selections`/`conditionals` fields default to empty vec with `skip_serializing_if` — existing JSON specs must remain byte-compatible
- **PyO3 constructor:** `selections`/`conditionals` accept `Option<&str>` (JSON strings deserialized via `serde_json::from_str`), not native Python lists/dicts
- Hit testing must handle all mark types in spec §6.2 (circle, rect, path, polygon, line, segment)
- **Hit testing strategy:** reverse z-order iteration (last batch = topmost); skip `Raw`/`Group` as non-data marks; line/polyline minimum 3px hit tolerance (`style.width.max(3.0)`)
- Conditional resolution updates GPU instance buffers in-place — no SceneGraph rebuild (spec §6.4)
- Zoom/pan uses per-panel `Affine2` transforms, not scale domain mutation (spec §6.5)
- **Double-click resets zoom/pan** to identity transform
- **Tick levels:** 4 pre-computed density levels at zoom breakpoints (count_hint 4/8/16/32 at ranges 0–0.5, 0.5–2.0, 2.0–4.0, 4.0–∞); ordinal scales return empty tick levels. Requires `tick_values(count_hint)` method on `ScaleKind`
- **Tooltips/hrefs:** safe DOM nodes only (no `innerHTML`); CSS opacity transition (0.1s ease); href opens with `noopener,noreferrer` security attrs
- **`selection_single`/`selection_multi`** are `functools.partial` wrappers around `selection_point` (not separate classes) — `toggle=False` and `toggle="event.shiftKey"` respectively
- **`SceneNode::Raw` NOT offset** during compound view scene graph merge (known limitation — legend colorbars will be mis-positioned until Raw is replaced with typed gradient nodes)
- **Transitions:** cubic ease-in-out timing, 300ms default duration; finalize via `load_scene()` with new scene on completion (do not leave interpolated state); key matching via `HashMap<&str, usize>` for O(1) lookup
- anywidget bidirectional sync via traitlets — selection state flows Python↔JS (spec §11.2)
- SVG/PNG renderers silently ignore selections and conditionals
- All existing golden SVGs and tests must continue to pass

## 5. Tasks

### Task 11c0: ChartSpec selection/conditional fields + Key encoding (cross-cutting prerequisite)
- [ ] Add `FieldValue` type to `ferrum-scene/src/selection.rs` per spec §6.1
- [ ] Add `selections: Vec<SelectionSpec>` and `conditionals: Vec<ConditionalEncoding>` to ChartSpec
- [ ] Add `key: Option<EncodingSpec>` to Encoding struct
- [ ] Wire selections/conditionals into SceneGraph in `scene_build.rs`; populate `InteractionConfig`
- [ ] Wire Key channel → `MarkBatch.keys` in scene_build.rs
- [ ] Verify: `cargo test`, existing goldens unchanged

### Task 11c1: Selection state machine + hit testing (Rust, ferrum-wasm)
- [ ] Implement `hit_test.rs` per spec §6.2: distance/containment functions per mark type
- [ ] Implement `selection_state.rs` per spec §6.3: `InteractionState`, `SelectionState` enum (Empty/Point/Interval), event dispatch (click → point selection, drag → interval selection)
- [ ] Multi-selection support: `toggle: true` adds/removes from set, `toggle: false` replaces
- [ ] Verify: Rust unit tests for hit-testing math and state transitions

### Task 11c2: Conditional encoding resolution + GPU buffer updates
- [ ] Implement `conditional.rs` per spec §6.4: given selection state + ConditionalEncoding list, compute per-node resolved values
- [ ] Update GPU instance buffers in-place (color, opacity, size) without SceneGraph rebuild
- [ ] Verify: Rust unit tests for conditional resolution logic

### Task 11c3: Zoom/pan + tick level selection
- [ ] Implement `zoom_pan.rs` per spec §6.5: per-panel `Affine2` transform, wheel → zoom, drag → pan
- [ ] Clamp zoom to `InteractionConfig.zoom_range`
- [ ] Tick level selection: swap pre-computed tick labels based on zoom factor per spec §6.5
- [ ] Verify: unit tests for affine transform composition and clamping

### Task 11c4: Tooltips + href click-through (JS)
- [ ] JS side: create tooltip `<div>`, position near cursor, populate from `MarkBatch.tooltips` per spec §6.6
- [ ] JS side: href click-through opens `MarkBatch.hrefs[i]` in new tab on click per spec §6.7
- [ ] Verify: tooltip appears on hover, href navigates on click

### Task 11c5: Python selection API (selection.py)
- [ ] Implement `selection_point()`, `selection_interval()`, `selection_single()`, `selection_multi()` constructors per spec §10.1
- [ ] Implement `Selection` class with `.when().otherwise()` conditional builder per spec §10.2
- [ ] Implement `SelectionMark` per spec §10.3
- [ ] Wire `Chart.add_selection()` and flow `_selections`/`_conditionals` through `to_spec()` per spec §10.4
- [ ] Verify: `uv run pytest tests/test_selection_api.py`

### Task 11c6: InteractiveChart anywidget class + bidirectional state sync
- [ ] Implement `InteractiveChart(anywidget.Widget)` in `_interactive.py` per spec §11.2
- [ ] Traitlets: `scene_json`, `selection_state`, `width`, `height`
- [ ] JS side: bridge anywidget model ↔ WasmRenderer selection state
- [ ] Wire `Chart.interactive()` → returns `InteractiveChart`
- [ ] Verify: widget renders in Jupyter, selection state syncs Python↔JS

### Task 11c7: Compound view scene graph merging
- [ ] Implement `merge_scene_graphs()` in `_interactive.py`: combine multiple Panel SceneGraphs into one unified SceneGraph with correct viewport offsets
- [ ] Handle HConcatChart, VConcatChart, FacetChart, RepeatChart, JointChart, ClusterMapChart
- [ ] Verify: compound view renders as single WASM chart

### Task 11c8: Animated transitions (Key channel)
- [ ] Implement `transition.rs` per spec §6.8: diff old/new MarkBatch via `keys`, lerp position/size/color/opacity
- [ ] Enter/exit transitions for added/removed keys
- [ ] Verify: Rust unit tests for key diffing and lerp

## 6. Acceptance checks

- `cargo test` — all Rust tests pass (ferrum-core, ferrum-scene, ferrum-wasm)
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- `uv run pytest tests/ -x --timeout=120` — all Python tests pass
- `uv run pytest tests/test_selection_api.py -v` — selection API tests pass
- Interactive HTML with selections: click selects points, drag creates interval, conditional encoding resolves visually
- Zoom/pan works in browser, tick labels update on zoom level change
- anywidget renders in Jupyter, selection state round-trips Python↔JS
- No changes to existing golden SVGs

## 7. Open questions

- Does `anywidget>=0.9` introduce conflicts with existing notebook deps? Check before adding to `pyproject.toml`.
- `FieldValue` shape in `ferrum-scene` was specified but not implemented in 11a — verify the existing `SelectionSpec` and `ConditionalEncoding` structs are compatible with the new `FieldValue` type before wiring.

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
