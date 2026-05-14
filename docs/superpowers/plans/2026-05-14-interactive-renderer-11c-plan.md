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

- `FieldValue` must use a tagged serde enum, not `serde_json::Value` (spec §6.1)
- `selections`/`conditionals` fields default to empty vec with `skip_serializing_if` — existing JSON specs must remain byte-compatible
- Hit testing must handle all mark types in spec §6.2 (circle, rect, path, polygon, line, segment)
- Conditional resolution updates GPU instance buffers in-place — no SceneGraph rebuild (spec §6.4)
- Zoom/pan uses per-panel `Affine2` transforms, not scale domain mutation (spec §6.5)
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
