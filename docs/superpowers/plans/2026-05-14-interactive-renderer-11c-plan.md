# Phase 11c — Selections + Zoom/Pan + anywidget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the full interaction system onto the SceneGraph IR (from 11a) and WASM renderer (from 11b): selection state machines, hit testing, conditional encoding resolution, zoom/pan, tooltips, href click-through, Python selection API, anywidget Jupyter integration, compound view scene graph merging, and animated transitions via Key channel.

**Architecture:** Selection specs and conditional encodings flow from Python through `ChartSpec` into `SceneGraph` at build time. The WASM renderer (`ferrum-wasm`) implements the runtime interaction state machine: hit testing, selection mutation, GPU buffer updates for conditional encodings, zoom/pan via per-panel Affine2 transforms, tooltip display, and href navigation. The Python side provides `selection_point()` / `selection_interval()` constructors, a `Selection` class with `.when().otherwise()` conditional builder, and an `InteractiveChart` anywidget class for Jupyter bidirectional state sync.

**Tech Stack:** Rust (ferrum-scene, ferrum-core, ferrum-wasm), wasm-bindgen, JavaScript (ESM glue), Python (anywidget, traitlets), CSS (tooltips).

**Spec:** `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` sections 6.1-6.8, 10.1-10.4, 11.2, 12.3.

**Prerequisites:** 11a (scene graph extraction) and 11b (WASM renderer foundation) must both be complete. Specifically:
- `ferrum-scene` crate with `SceneGraph`, `Panel`, `MarkBatch`, `SelectionSpec`, `ConditionalEncoding`, `InteractionConfig` types (done in 11a)
- `ferrum-wasm` crate with `WasmRenderer`, wgpu pipelines, JS ESM glue module, CSS text overlay (delivered by 11b)
- `render_interactive` PyO3 binding returning SceneGraph JSON (done in 11a, Task 8)
- `.save("chart.html")` and `.save("chart.json")` working for static charts (delivered by 11b)

---

## File map

### New files

| File | Purpose |
|---|---|
| `src/ferrum/selection.py` | `selection_point`, `selection_interval`, `selection_single`, `selection_multi`, `Selection`, `SelectionMark`, `_SelectionCondition`, `ConditionalSpec` |
| `src/ferrum/_interactive.py` | `InteractiveChart` (anywidget subclass), `merge_scene_graphs()`, `on_selection_change` |
| `src/ferrum/_wasm/ferrum-interactive.css` | Tooltip styling, overlay layout |
| `crates/ferrum-wasm/src/hit_test.rs` | Per-mark-type hit testing (circle, rect, path, polygon, line, segment) |
| `crates/ferrum-wasm/src/selection_state.rs` | `InteractionState`, `SelectionState`, event dispatch, state mutation |
| `crates/ferrum-wasm/src/conditional.rs` | Conditional encoding resolution, GPU instance buffer updates |
| `crates/ferrum-wasm/src/zoom_pan.rs` | Per-panel Affine2 transforms, zoom/pan event handling, tick level selection |
| `crates/ferrum-wasm/src/transition.rs` | Animated transitions via Key channel diffing + lerp |
| `tests/test_selection_api.py` | Python unit tests for selection API serialization |
| `crates/ferrum-wasm/tests/hit_test.rs` | Rust unit tests for hit-testing math |
| `crates/ferrum-wasm/tests/selection_state.rs` | Rust unit tests for selection state logic |
| `crates/ferrum-wasm/tests/conditional.rs` | Rust unit tests for conditional encoding resolution |

### Modified files

| File | Change |
|---|---|
| `crates/ferrum-core/src/spec/chart.rs` | Add `selections` and `conditionals` fields to `ChartSpec` |
| `crates/ferrum-core/src/spec/encoding.rs` | Add `key` field to `Encoding` struct |
| `crates/ferrum-core/src/render/scene_build.rs` | Copy `selections`/`conditionals` into SceneGraph; populate `InteractionConfig.tick_levels`; wire Key channel to `MarkBatch.keys` |
| `crates/ferrum-wasm/src/lib.rs` | Add modules: `hit_test`, `selection_state`, `conditional`, `zoom_pan`, `transition`; wire into `WasmRenderer` event loop |
| `crates/ferrum-wasm/src/renderer.rs` | Extend `WasmRenderer` with `InteractionState`, event handlers, `resolve_conditionals()`, `transition_scene()` |
| `src/ferrum/_wasm/ferrum-interactive.js` | Add event capture (click, mousemove, mousedown/up, wheel), tooltip div management, href navigation, anywidget model bridge, selection state sync |
| `src/ferrum/chart.py` | Wire `add_selection()` to store selections; wire `interactive()` to return `InteractiveChart`; add `_selections` and `_conditionals` state; flow through `to_spec()` |
| `src/ferrum/encoding/base.py` | Wire `condition` kwarg on `ChannelBase` to validate and extract selection ref + if/else values |
| `src/ferrum/encoding/appearance.py` | Remove "reserved for future use" notes on `condition` kwargs; validate via updated `ChannelBase` |
| `src/ferrum/display.py` | Extend `save_chart()` to dispatch `"html"` format through `InteractiveChart.save()` |
| `src/ferrum/__init__.py` | Export `selection_point`, `selection_interval`, `selection_single`, `selection_multi`, `Selection`, `SelectionMark`, `InteractiveChart` |
| `pyproject.toml` | Add `anywidget>=0.9` to `[project.dependencies]` |

### Unchanged files

All geometry computation code (marks, scale resolution, position adjustments, layout, transforms). `ferrum-scene/src/types.rs` and `ferrum-scene/src/selection.rs` (types already defined in 11a). SVG walker, SvgBuffer, rasterizer, compositor, grid_compose. Existing Python tests and golden SVGs (11c adds new tests but does not modify existing ones).

---

## Task 11c0: ChartSpec selection/conditional fields + Key encoding (Rust, cross-cutting prerequisite)

This task adds the Rust-side plumbing that all downstream tasks depend on: selection specs and conditional encodings flow from Python through `ChartSpec` into `SceneGraph`, and the Key encoding channel populates `MarkBatch.keys`. It also adds the `FieldValue` type to ferrum-scene (spec section 6.1 defined it but 11a did not implement it).

**Why first:** Without these fields on `ChartSpec`, the Python selection API (11c5) cannot serialize selections into the spec, `scene_build.rs` cannot populate the SceneGraph interaction config, and ferrum-wasm cannot receive selection/conditional data. The Key channel wiring for animated transitions (11c8) also requires `Encoding` changes that are cleanest to do here. `FieldValue` is needed by `selection_state.rs` in ferrum-wasm (11c1).

**Files:**
- Modify: `crates/ferrum-scene/src/selection.rs` (add `FieldValue` type)
- Modify: `crates/ferrum-core/src/spec/chart.rs`
- Modify: `crates/ferrum-core/src/spec/encoding.rs`
- Modify: `crates/ferrum-core/src/render/scene_build.rs`

### Steps

- [ ] **Step 0: Add `FieldValue` to ferrum-scene**

In `crates/ferrum-scene/src/selection.rs`, add the `FieldValue` enum. This type represents a typed value in selection state (field-value pairs for point selections). Spec section 6.1 defines it:

```rust
/// Typed field value for selection state -- avoids serde_json::Value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}
```

Add it to `lib.rs` re-exports. Verify: `cargo build -p ferrum-scene`.

- [ ] **Step 1: Add `selections` and `conditionals` to ChartSpec**

In `crates/ferrum-core/src/spec/chart.rs`, add two new fields to `ChartSpec`:

```rust
use ferrum_scene::{SelectionSpec, ConditionalEncoding};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    // ... existing fields ...

    /// Interactive selections attached via `Chart.add_selection()`.
    /// SVG/PNG renderers ignore these; the WASM renderer uses them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selections: Vec<SelectionSpec>,

    /// Conditional encodings (e.g., color changes when selection is active).
    /// SVG/PNG renderers ignore these; the WASM renderer resolves them at runtime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditionals: Vec<ConditionalEncoding>,
}
```

Both fields default to empty vecs and are skipped in JSON when empty, so existing specs (phases 1-11b) are byte-compatible.

Update the `#[pymethods] impl ChartSpec` `new()` constructor to accept optional `selections` and `conditionals` parameters. These should accept Python lists of dicts that match the serde JSON shape of `SelectionSpec` and `ConditionalEncoding`. Use `serde_json::from_str` on the JSON-serialized Python input to deserialize into the Rust types.

```python
# Python call site will look like:
ChartSpec(
    mark="point",
    x=...,
    selections=[sel.to_dict() for sel in self._selections],
    conditionals=[c.to_dict() for c in self._conditionals],
)
```

- [ ] **Step 2: Add `key` field to Encoding struct**

In `crates/ferrum-core/src/spec/encoding.rs`, add a `key` field to the `Encoding` struct:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Encoding {
    // ... existing fields (x, y, color, size, shape, opacity, x2, y2, text, tooltip, href, description) ...

    /// Key channel -- provides a stable identity per mark for animated transitions.
    /// The field values populate `MarkBatch.keys` in the SceneGraph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<EncodingSpec>,
}
```

Update `Encoding::overlay_from()` to handle the new `key` field (same pattern as existing channels: overlay if the parent has it and the child does not).

- [ ] **Step 3: Wire selections and conditionals into SceneGraph in scene_build.rs**

In `crates/ferrum-core/src/render/scene_build.rs`, the `build_scene()` function currently hardcodes:

```rust
selections: Vec::new(),
interaction: InteractionConfig::default(),
```

Change to:

```rust
selections: spec.selections.clone(),
interaction: InteractionConfig {
    zoom_enabled: spec.selections.iter().any(|s| matches!(s, SelectionSpec::Interval { zoom: true, .. })),
    pan_enabled: spec.selections.iter().any(|s| matches!(s, SelectionSpec::Interval { translate: true, .. })),
    conditionals: spec.conditionals.clone(),
    linked_panels: compute_linked_panels(&panels, &spec.selections),
    tick_levels: Vec::new(), // populated in Step 5
},
```

The `compute_linked_panels()` helper groups panel IDs that share the same selection resolve mode:

```rust
fn compute_linked_panels(panels: &[Panel], selections: &[SelectionSpec]) -> Vec<Vec<usize>> {
    // For "global" resolve: all panels form one group
    // For "union"/"intersect": panels with the same encoding linkage form a group
    // Simple initial implementation: if any selection is global, link all panels
    let all_ids: Vec<usize> = panels.iter().map(|p| p.id).collect();
    if selections.iter().any(|s| match s {
        SelectionSpec::Point { resolve, .. } | SelectionSpec::Interval { resolve, .. }
            => *resolve == SelectionResolve::Global,
    }) {
        vec![all_ids]
    } else {
        // Each panel is its own group by default
        all_ids.into_iter().map(|id| vec![id]).collect()
    }
}
```

- [ ] **Step 4: Wire Key channel to MarkBatch.keys**

In `scene_build.rs`, the mark batch assembly currently hardcodes `keys: None`. Change it to read the key field from the layer's encoding and populate the keys vector:

```rust
// Inside the layer loop, after dispatch_mark_build:
let keys = layer.encoding.key.as_ref().and_then(|key_enc| {
    let field_name = &key_enc.field;
    layer_batch.column_by_name(field_name).and_then(|col| {
        // Convert column values to strings for key matching
        use arrow::array::AsArray;
        let string_col = arrow::compute::cast(col, &arrow::datatypes::DataType::Utf8).ok()?;
        let string_array = string_col.as_string::<i32>();
        Some(string_array.iter().map(|v| v.unwrap_or("").to_string()).collect::<Vec<_>>())
    })
});

mark_batches.push(MarkBatch {
    // ... existing fields ...
    keys,  // was: keys: None
    // ...
});
```

- [ ] **Step 5: Pre-compute tick levels for zoom**

Add a `compute_tick_levels()` function that generates 3-4 tick granularities per panel axis. This is called once at SceneGraph build time so the WASM renderer can select the appropriate level on zoom without a Python round-trip.

```rust
fn compute_tick_levels(
    panel_id: usize,
    scales: &ResolvedScales,
    plot_area: &ferrum_scene::Rect,
) -> ferrum_scene::PanelTickLevels {
    // ResolvedScales has `x: ScaleKind` and `y: ScaleKind`.
    // ScaleKind exposes `tick_labels(count_hint)`, `to_pixel_f64(value)`,
    // and `pixel_range()`. We add a `tick_values(count_hint) -> Vec<f64>`
    // method (see generate_axis_tick_levels below) to get raw numeric
    // tick values for the Tick structs.
    let x_levels = generate_axis_tick_levels(&scales.x, plot_area.w);
    let y_levels = generate_axis_tick_levels(&scales.y, plot_area.h);
    ferrum_scene::PanelTickLevels {
        panel_id,
        x_levels,
        y_levels,
    }
}

fn generate_axis_tick_levels(
    scale: &ScaleKind,
    extent_px: f64,
) -> Vec<ferrum_scene::TickLevel> {
    // The existing ScaleKind API:
    //   scale.tick_labels(count_hint) -> Vec<String>  (formatted labels)
    //   scale.to_pixel_f64(value) -> Option<f64>      (value -> pixel)
    //   scale.pixel_range() -> (f64, f64)             (lo_px, hi_px)
    //
    // For ordinal scales, tick levels are not useful (discrete categories
    // don't subdivide on zoom), so return an empty vec.
    if matches!(scale, ScaleKind::Ordinal(_)) {
        return Vec::new();
    }

    // Generate 4 tick density levels from the existing scale's tick
    // generation logic, varying the count_hint parameter.
    //
    // Level 0 (sparse):  count_hint=4, zoom range 0.0..0.5
    // Level 1 (base):    count_hint=8, zoom range 0.5..2.0
    // Level 2 (2x):      count_hint=16, zoom range 2.0..4.0
    // Level 3 (4x):      count_hint=32, zoom range 4.0..inf
    //
    // Tick.pixel is the position at 1x zoom -- the WASM renderer scales it
    // by the current panel transform.

    let level_configs: [(usize, f64, f64); 4] = [
        (4,  0.0,          0.5),
        (8,  0.5,          2.0),
        (16, 2.0,          4.0),
        (32, 4.0,          f64::INFINITY),
    ];

    level_configs.iter().map(|&(count_hint, min_zoom, max_zoom)| {
        let labels = scale.tick_labels(count_hint);
        // tick_labels returns formatted strings. To get the corresponding
        // pixel positions, we need the underlying numeric tick values.
        // Use the scale's ticks_internal (accessed via the dispatch macro)
        // and map each through to_pixel_f64.
        //
        // For the plan: the implementer should add a `tick_values(count_hint)
        // -> Vec<f64>` method on ScaleKind (parallel to tick_labels but
        // returning raw f64s) or extract from the existing dispatch_continuous
        // macro. A tick_values method is the cleanest approach:
        //
        //   pub fn tick_values(&self, count_hint: usize) -> Vec<f64> {
        //       dispatch_continuous!(self, ticks_internal, count_hint)
        //   }
        //
        // Then:
        let values = scale.tick_values(count_hint);
        let ticks: Vec<ferrum_scene::Tick> = values.iter().zip(labels.iter())
            .filter_map(|(v, label)| {
                scale.to_pixel_f64(*v).map(|px| ferrum_scene::Tick {
                    value: *v,
                    label: label.clone(),
                    pixel: px,
                })
            })
            .collect();
        ferrum_scene::TickLevel { min_zoom, max_zoom, ticks }
    }).collect()
}
```

Wire this into the panel assembly loop:

```rust
// After building the panel, before pushing to panels vec:
let tick_levels = compute_tick_levels(panel_idx, &scales, &plot_area);
// Collect into a vec and assign to interaction.tick_levels after the loop
```

Then after the panel loop, assign `tick_levels` to `interaction.tick_levels`:

```rust
let tick_levels: Vec<PanelTickLevels> = panels.iter().enumerate()
    .map(|(idx, _)| /* computed during panel loop and collected */)
    .collect();

// In the SceneGraph construction:
interaction: InteractionConfig {
    // ...
    tick_levels,
},
```

- [ ] **Step 6: Update ChartSpec PyO3 constructor**

In the `#[pymethods] impl ChartSpec` block in `chart.rs`, add `selections` and `key` parameters to the `new()` function. The `selections` parameter accepts a JSON string (serialized from Python), and `key` accepts an `EncodingSpec` like other channels:

```rust
#[pyo3(signature = (
    *, mark, x = None, y = None, color = None,
    size = None, shape = None, opacity = None,
    x2 = None, y2 = None, text = None,
    tooltip = None, href = None, description = None,
    key = None,           // NEW -- 11c Key channel
    data = None, transforms = None, layers = None,
    coord = None, facet = None, mark_style = None,
    position = None, title = None,
    axis_x = None, axis_y = None,
    selections = None,    // NEW -- 11c selection specs
    conditionals = None,  // NEW -- 11c conditional encodings
))]
```

For `selections` and `conditionals`, accept `Option<&str>` (JSON strings) and deserialize via serde_json. This avoids complex PyO3 type conversions -- the Python side serializes with `json.dumps()`.

- [ ] **Step 7: Verify compilation and existing tests**

Run:
```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test
uv run pytest tests/ -v
```

Expected: all existing tests pass. The new fields are `serde(default)` so existing JSON specs without `selections`/`conditionals`/`key` deserialize unchanged.

- [ ] **Step 8: Commit**

```
feat(spec): add selections, conditionals, and key encoding to ChartSpec for 11c interaction system
```

---

## Task 11c1: Selection state machine + hit testing (Rust, ferrum-wasm)

The selection state machine lives entirely in ferrum-wasm. It manages the runtime state of all active selections and dispatches hit tests when the user clicks or drags.

**Files:**
- Create: `crates/ferrum-wasm/src/selection_state.rs`
- Create: `crates/ferrum-wasm/src/hit_test.rs`
- Modify: `crates/ferrum-wasm/src/lib.rs` (add module declarations)
- Modify: `crates/ferrum-wasm/src/renderer.rs` (add `InteractionState` to `WasmRenderer`)

### Steps

- [ ] **Step 1: Define InteractionState and SelectionState**

Create `crates/ferrum-wasm/src/selection_state.rs`:

```rust
use std::collections::HashMap;
use ferrum_scene::{
    FieldValue, SelectionSpec, SelectionResolve, ChannelName, EventExpr,
    SceneGraph, Panel, MarkBatch, TooltipContent,
};

/// Runtime interaction state for a single WasmRenderer instance.
/// Not serialized -- lives in WASM memory only.
pub struct InteractionState {
    /// Named selections, keyed by `SelectionSpec.name`.
    pub selections: HashMap<String, SelectionState>,
    /// Per-panel affine transforms for zoom/pan. Index = panel.id.
    pub panel_transforms: Vec<[f64; 6]>,
    /// Current hover state (for tooltip rendering).
    pub hover: Option<HoverState>,
}

/// Current state of a single named selection.
pub enum SelectionState {
    Empty,
    Point {
        /// Data-space indices of selected marks.
        indices: Vec<usize>,
        /// Field-value pairs for field-based matching.
        field_values: Vec<(String, FieldValue)>,
    },
    Interval {
        /// Data-space x range (not pixel-space).
        x_range: Option<(f64, f64)>,
        /// Data-space y range (not pixel-space).
        y_range: Option<(f64, f64)>,
    },
}

pub struct HoverState {
    pub panel_id: usize,
    pub batch_index: usize,
    pub node_index: usize,
    pub tooltip: TooltipContent,
}

impl InteractionState {
    /// Initialize from a SceneGraph's selections and panel count.
    pub fn from_scene(scene: &SceneGraph) -> Self {
        let mut selections = HashMap::new();
        for sel in &scene.selections {
            let name = match sel {
                SelectionSpec::Point { name, .. } => name.clone(),
                SelectionSpec::Interval { name, .. } => name.clone(),
            };
            selections.insert(name, SelectionState::Empty);
        }
        let panel_transforms = vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; scene.panels.len()];
        Self { selections, panel_transforms, hover: None }
    }

    /// Check if a data index is selected by a named selection.
    pub fn is_selected(&self, selection_name: &str, data_index: usize) -> bool {
        match self.selections.get(selection_name) {
            Some(SelectionState::Empty) => false,
            Some(SelectionState::Point { indices, .. }) => indices.contains(&data_index),
            Some(SelectionState::Interval { .. }) => {
                // Interval selections use is_in_interval() with data-space coords
                false
            }
            None => false,
        }
    }

    /// Check if data-space coordinates fall within an interval selection.
    pub fn is_in_interval(
        &self,
        selection_name: &str,
        data_x: f64,
        data_y: f64,
    ) -> bool {
        match self.selections.get(selection_name) {
            Some(SelectionState::Interval { x_range, y_range }) => {
                let in_x = x_range.map_or(true, |(lo, hi)| data_x >= lo && data_x <= hi);
                let in_y = y_range.map_or(true, |(lo, hi)| data_y >= lo && data_y <= hi);
                in_x && in_y
            }
            _ => false,
        }
    }
}

/// Result of processing a user event against the selection state machine.
pub enum SelectionAction {
    /// No state change.
    NoOp,
    /// Selection state changed -- resolve conditionals and re-render.
    Updated { selection_name: String },
    /// Href click -- navigate to URL.
    Navigate { url: String },
}
```

- [ ] **Step 2: Implement event dispatch on InteractionState**

Add methods for the three main event types:

```rust
impl InteractionState {
    /// Process a click event at pixel coordinates within a panel.
    pub fn on_click(
        &mut self,
        scene: &SceneGraph,
        panel_id: usize,
        px: f64,
        py: f64,
        shift_key: bool,
    ) -> SelectionAction {
        // 1. Check for href click -- if a mark with href is hit, navigate
        if let Some(hit) = hit_test_panel(scene, panel_id, px, py, &self.panel_transforms[panel_id]) {
            if let Some(url) = hit.href.as_deref() {
                return SelectionAction::Navigate { url: url.to_string() };
            }
        }

        // 2. Find matching point selections
        for sel_spec in &scene.selections {
            match sel_spec {
                SelectionSpec::Point { name, toggle, on, nearest, fields, .. }
                    if matches!(on, EventExpr::Click) =>
                {
                    let hit = if *nearest {
                        nearest_hit_test(scene, panel_id, px, py, &self.panel_transforms[panel_id])
                    } else {
                        hit_test_panel(scene, panel_id, px, py, &self.panel_transforms[panel_id])
                    };

                    let sel = self.selections.get_mut(name).unwrap();
                    if let Some(hit) = hit {
                        let should_toggle = shift_key && matches!(toggle, EventExpr::ShiftKey);
                        apply_point_selection(sel, hit.data_index, hit.field_values, should_toggle);
                        return SelectionAction::Updated { selection_name: name.clone() };
                    } else {
                        // Click on empty space -- clear if no modifier
                        if !shift_key {
                            *sel = SelectionState::Empty;
                            return SelectionAction::Updated { selection_name: name.clone() };
                        }
                    }
                }
                _ => {}
            }
        }
        SelectionAction::NoOp
    }

    /// Process a mouse-move event for hover/tooltip and interval drag.
    pub fn on_mousemove(
        &mut self,
        scene: &SceneGraph,
        panel_id: usize,
        px: f64,
        py: f64,
        is_dragging: bool,
    ) -> (Option<TooltipContent>, Option<String>) {
        // If dragging with an active interval selection, update interval bounds
        if is_dragging {
            for sel_spec in &scene.selections {
                if let SelectionSpec::Interval { name, .. } = sel_spec {
                    if let Some(sel) = self.selections.get_mut(name) {
                        // Convert pixel to data space and update interval
                        // (implementation in zoom_pan.rs handles the transform)
                    }
                }
            }
        }

        // Hit test for tooltip
        let hit = hit_test_panel(scene, panel_id, px, py, &self.panel_transforms[panel_id]);
        let tooltip = hit.as_ref().and_then(|h| h.tooltip.clone());
        let selection_update = None; // returned if interval selection changed

        // Update hover state
        self.hover = hit.map(|h| HoverState {
            panel_id,
            batch_index: h.batch_index,
            node_index: h.node_index,
            tooltip: h.tooltip.clone().unwrap_or(TooltipContent { fields: vec![] }),
        });

        (tooltip, selection_update)
    }
}

fn apply_point_selection(
    sel: &mut SelectionState,
    data_index: usize,
    field_values: Vec<(String, FieldValue)>,
    toggle: bool,
) {
    match sel {
        SelectionState::Point { indices, field_values: fv } if toggle => {
            // Toggle: add if not present, remove if present
            if let Some(pos) = indices.iter().position(|&i| i == data_index) {
                indices.remove(pos);
            } else {
                indices.push(data_index);
                fv.extend(field_values);
            }
            if indices.is_empty() {
                *sel = SelectionState::Empty;
            }
        }
        _ => {
            // Replace: single selection
            *sel = SelectionState::Point { indices: vec![data_index], field_values };
        }
    }
}
```

- [ ] **Step 3: Implement hit testing per mark type**

Create `crates/ferrum-wasm/src/hit_test.rs`:

```rust
use ferrum_scene::*;

/// Result of a successful hit test.
pub struct HitResult {
    pub panel_id: usize,
    pub batch_index: usize,
    pub node_index: usize,
    pub data_index: usize,
    pub tooltip: Option<TooltipContent>,
    pub href: Option<String>,
    pub field_values: Vec<(String, FieldValue)>,
}

/// Hit test all marks in a panel, returning the topmost hit (last batch = topmost z-order).
pub fn hit_test_panel(
    scene: &SceneGraph,
    panel_id: usize,
    px: f64,
    py: f64,
    transform: &[f64; 6],
) -> Option<HitResult> {
    let panel = scene.panels.iter().find(|p| p.id == panel_id)?;

    // Inverse-transform pixel coordinates to scene-space
    let (sx, sy) = inverse_affine(transform, px, py);

    // Check if point is within panel clip rect
    if !rect_contains(&panel.clip, sx, sy) {
        return None;
    }

    // Iterate batches in reverse z-order (topmost first)
    for (batch_idx, batch) in panel.marks.iter().enumerate().rev() {
        // Skip batches without data indices (decoration batches)
        for (node_idx, node) in batch.nodes.iter().enumerate().rev() {
            // Skip SceneNode::Raw and SceneNode::Group -- not data marks
            if matches!(node, SceneNode::Raw { .. } | SceneNode::Group { .. }) {
                continue;
            }
            if hit_test_node(node, &batch.kind, sx, sy) {
                let data_index = batch.data_indices.as_ref()
                    .and_then(|ids| ids.get(node_idx).copied())
                    .unwrap_or(node_idx);
                let tooltip = batch.tooltips.as_ref().and_then(|t| t.get(node_idx).cloned());
                let href = batch.hrefs.as_ref().and_then(|h| h.get(node_idx).cloned().flatten());
                return Some(HitResult {
                    panel_id,
                    batch_index: batch_idx,
                    node_index: node_idx,
                    data_index,
                    tooltip,
                    href,
                    field_values: Vec::new(), // populated by caller from data if needed
                });
            }
        }
    }
    None
}

/// Find the nearest mark to the cursor (for `nearest=true` point selections).
pub fn nearest_hit_test(
    scene: &SceneGraph,
    panel_id: usize,
    px: f64,
    py: f64,
    transform: &[f64; 6],
) -> Option<HitResult> {
    let panel = scene.panels.iter().find(|p| p.id == panel_id)?;
    let (sx, sy) = inverse_affine(transform, px, py);

    let mut best: Option<(f64, HitResult)> = None;

    for (batch_idx, batch) in panel.marks.iter().enumerate() {
        for (node_idx, node) in batch.nodes.iter().enumerate() {
            if matches!(node, SceneNode::Raw { .. } | SceneNode::Group { .. }) {
                continue;
            }
            let dist = distance_to_node(node, sx, sy);
            let is_better = best.as_ref().map_or(true, |(d, _)| dist < *d);
            if is_better {
                let data_index = batch.data_indices.as_ref()
                    .and_then(|ids| ids.get(node_idx).copied())
                    .unwrap_or(node_idx);
                let tooltip = batch.tooltips.as_ref().and_then(|t| t.get(node_idx).cloned());
                let href = batch.hrefs.as_ref().and_then(|h| h.get(node_idx).cloned().flatten());
                best = Some((dist, HitResult {
                    panel_id,
                    batch_index: batch_idx,
                    node_index: node_idx,
                    data_index,
                    tooltip,
                    href,
                    field_values: Vec::new(),
                }));
            }
        }
    }
    best.map(|(_, r)| r)
}

/// Per-node hit test. Returns true if (sx, sy) is within the node.
pub fn hit_test_node(node: &SceneNode, kind: &MarkBatchKind, sx: f64, sy: f64) -> bool {
    match node {
        SceneNode::Circle { cx, cy, r, .. } => {
            // Euclidean distance check
            let dx = sx - cx;
            let dy = sy - cy;
            dx * dx + dy * dy <= r * r
        }
        SceneNode::Rect { x, y, w, h, .. } => {
            // AABB containment
            sx >= *x && sx <= x + w && sy >= *y && sy <= y + h
        }
        SceneNode::Line { x1, y1, x2, y2, style } => {
            // Distance to line segment with stroke-width tolerance
            let tolerance = style.width.max(3.0); // minimum 3px tolerance
            distance_to_segment(sx, sy, *x1, *y1, *x2, *y2) <= tolerance
        }
        SceneNode::Polygon { points, .. } => {
            // Winding number point-in-polygon
            point_in_polygon(sx, sy, points)
        }
        SceneNode::Path { commands, closed, .. } if *closed => {
            // For closed paths (area marks, ribbons): convert to points, then winding number
            let points = path_to_points(commands);
            let pts: Vec<[f64; 2]> = points.iter().map(|&(x, y)| [x, y]).collect();
            point_in_polygon(sx, sy, &pts)
        }
        SceneNode::Polyline { points, style } => {
            // Distance to polyline with stroke-width tolerance
            let tolerance = style.width.max(3.0);
            for win in points.windows(2) {
                if distance_to_segment(sx, sy, win[0].0, win[0].1, win[1].0, win[1].1) <= tolerance {
                    return true;
                }
            }
            false
        }
        // Group, Raw, Image, Text, open Path -- not hit-testable as data marks
        _ => false,
    }
}

/// Euclidean distance from point to node center (for nearest-mark search).
fn distance_to_node(node: &SceneNode, sx: f64, sy: f64) -> f64 {
    match node {
        SceneNode::Circle { cx, cy, .. } => {
            ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt()
        }
        SceneNode::Rect { x, y, w, h, .. } => {
            // Distance to rect center
            let cx = x + w / 2.0;
            let cy = y + h / 2.0;
            ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt()
        }
        _ => f64::INFINITY,
    }
}

// --- Geometry helpers ---

pub fn inverse_affine(t: &[f64; 6], px: f64, py: f64) -> (f64, f64) {
    // t = [a, b, tx, c, d, ty]  => | a  b  tx |
    //                                | c  d  ty |
    //                                | 0  0   1 |
    let (a, b, tx, c, d, ty) = (t[0], t[1], t[2], t[3], t[4], t[5]);
    let det = a * d - b * c;
    if det.abs() < 1e-12 { return (px, py); }
    let inv_det = 1.0 / det;
    let sx = inv_det * (d * (px - tx) - b * (py - ty));
    let sy = inv_det * (-c * (px - tx) + a * (py - ty));
    (sx, sy)
}

fn rect_contains(r: &Rect, x: f64, y: f64) -> bool {
    x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h
}

fn distance_to_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = x1 + t * dx;
    let proj_y = y1 + t * dy;
    ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
}

fn point_in_polygon(px: f64, py: f64, points: &[[f64; 2]]) -> bool {
    // Winding number algorithm
    let n = points.len();
    if n < 3 { return false; }
    let mut winding = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let (y0, y1) = (points[i][1], points[j][1]);
        if y0 <= py {
            if y1 > py {
                let vt = cross_2d(
                    points[j][0] - points[i][0], points[j][1] - points[i][1],
                    px - points[i][0], py - points[i][1],
                );
                if vt > 0.0 { winding += 1; }
            }
        } else if y1 <= py {
            let vt = cross_2d(
                points[j][0] - points[i][0], points[j][1] - points[i][1],
                px - points[i][0], py - points[i][1],
            );
            if vt < 0.0 { winding -= 1; }
        }
    }
    winding != 0
}

fn cross_2d(ux: f64, uy: f64, vx: f64, vy: f64) -> f64 {
    ux * vy - uy * vx
}

fn path_to_points(commands: &[PathCmd]) -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    let mut cx = 0.0;
    let mut cy = 0.0;
    for cmd in commands {
        match cmd {
            PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => {
                cx = *x; cy = *y; pts.push((cx, cy));
            }
            PathCmd::HLineTo { x } => { cx = *x; pts.push((cx, cy)); }
            PathCmd::VLineTo { y } => { cy = *y; pts.push((cx, cy)); }
            PathCmd::Close => {} // closing segment handled by polygon check
            _ => {} // QuadTo, CubicTo, ArcTo: approximate with endpoint
        }
    }
    pts
}
```

- [ ] **Step 4: Add InteractionState to WasmRenderer**

In `crates/ferrum-wasm/src/renderer.rs`, extend `WasmRenderer`:

```rust
use crate::selection_state::InteractionState;

pub struct WasmRenderer {
    // ... existing fields from 11b ...
    interaction: InteractionState,
}

impl WasmRenderer {
    pub fn load_scene(&mut self, scene: SceneGraph) {
        self.interaction = InteractionState::from_scene(&scene);
        // ... existing scene loading ...
    }
}
```

- [ ] **Step 5: Expose event handlers via wasm_bindgen**

Add `#[wasm_bindgen]` methods that the JS glue calls:

```rust
#[wasm_bindgen]
impl WasmRenderer {
    pub fn on_click(&mut self, panel_id: usize, px: f64, py: f64, shift_key: bool) -> JsValue {
        let action = self.interaction.on_click(&self.scene.graph, panel_id, px, py, shift_key);
        match action {
            SelectionAction::Updated { selection_name } => {
                self.resolve_conditionals();
                self.render_frame();
                self.export_selection_state()
            }
            SelectionAction::Navigate { url } => {
                // Return navigation request to JS
                serde_wasm_bindgen::to_value(&serde_json::json!({"navigate": url}))
                    .unwrap_or(JsValue::NULL)
            }
            SelectionAction::NoOp => JsValue::NULL,
        }
    }

    pub fn on_mousemove(&mut self, panel_id: usize, px: f64, py: f64, is_dragging: bool) -> JsValue {
        let (tooltip, _) = self.interaction.on_mousemove(
            &self.scene.graph, panel_id, px, py, is_dragging,
        );
        // Return tooltip content as JSON for JS to render as DOM
        match tooltip {
            Some(tt) => serde_wasm_bindgen::to_value(&tt).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Export current selection state as JSON (for anywidget sync).
    fn export_selection_state(&self) -> JsValue {
        let state: std::collections::HashMap<String, serde_json::Value> =
            self.interaction.selections.iter()
            .map(|(name, sel)| {
                let val = match sel {
                    SelectionState::Empty => serde_json::json!({"type": "empty"}),
                    SelectionState::Point { indices, field_values } => serde_json::json!({
                        "type": "point",
                        "indices": indices,
                    }),
                    SelectionState::Interval { x_range, y_range } => serde_json::json!({
                        "type": "interval",
                        "x_range": x_range,
                        "y_range": y_range,
                    }),
                };
                (name.clone(), val)
            })
            .collect();
        serde_wasm_bindgen::to_value(&state).unwrap_or(JsValue::NULL)
    }
}
```

- [ ] **Step 6: Write Rust unit tests for hit testing**

Create `crates/ferrum-wasm/tests/hit_test.rs`:

```rust
use ferrum_scene::*;
use ferrum_wasm::hit_test::*;

#[test]
fn circle_hit_inside() {
    let node = SceneNode::Circle { cx: 100.0, cy: 100.0, r: 10.0,
        style: FillStroke { fill: Some(Color::rgb(0,0,0)), stroke: None,
            stroke_width: 0.0, opacity: 1.0, stroke_dash: None } };
    assert!(hit_test_node(&node, &MarkBatchKind::Point, 105.0, 105.0));
}

#[test]
fn circle_hit_outside() {
    let node = SceneNode::Circle { cx: 100.0, cy: 100.0, r: 10.0,
        style: FillStroke { fill: Some(Color::rgb(0,0,0)), stroke: None,
            stroke_width: 0.0, opacity: 1.0, stroke_dash: None } };
    assert!(!hit_test_node(&node, &MarkBatchKind::Point, 120.0, 120.0));
}

#[test]
fn rect_hit_containment() {
    let node = SceneNode::Rect { x: 50.0, y: 50.0, w: 40.0, h: 30.0,
        style: FillStroke { fill: Some(Color::rgb(0,0,0)), stroke: None,
            stroke_width: 0.0, opacity: 1.0, stroke_dash: None },
        corner_radius: 0.0 };
    assert!(hit_test_node(&node, &MarkBatchKind::Bar, 70.0, 65.0));
    assert!(!hit_test_node(&node, &MarkBatchKind::Bar, 95.0, 65.0));
}

#[test]
fn line_hit_with_tolerance() {
    let node = SceneNode::Line { x1: 0.0, y1: 0.0, x2: 100.0, y2: 0.0,
        style: StrokeStyle { color: Color::rgb(0,0,0), width: 2.0,
            opacity: 1.0, dash: None, stroke_cap: None, stroke_join: None } };
    // 1px away from line -- within 3px min tolerance
    assert!(hit_test_node(&node, &MarkBatchKind::Line, 50.0, 1.0));
    // 5px away -- outside tolerance
    assert!(!hit_test_node(&node, &MarkBatchKind::Line, 50.0, 5.0));
}

#[test]
fn polygon_winding_number() {
    let node = SceneNode::Polygon {
        points: vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        style: FillStroke { fill: Some(Color::rgb(0,0,0)), stroke: None,
            stroke_width: 0.0, opacity: 1.0, stroke_dash: None },
    };
    assert!(hit_test_node(&node, &MarkBatchKind::Polygon, 50.0, 50.0));
    assert!(!hit_test_node(&node, &MarkBatchKind::Polygon, 150.0, 50.0));
}

#[test]
fn inverse_affine_identity() {
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (x, y) = inverse_affine(&identity, 42.0, 17.0);
    assert!((x - 42.0).abs() < 1e-10);
    assert!((y - 17.0).abs() < 1e-10);
}

#[test]
fn inverse_affine_translation() {
    let t = [1.0, 0.0, 10.0, 0.0, 1.0, 20.0]; // translate (10, 20)
    let (x, y) = inverse_affine(&t, 50.0, 70.0);
    assert!((x - 40.0).abs() < 1e-10);
    assert!((y - 50.0).abs() < 1e-10);
}

#[test]
fn inverse_affine_scale() {
    let t = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]; // scale 2x
    let (x, y) = inverse_affine(&t, 100.0, 200.0);
    assert!((x - 50.0).abs() < 1e-10);
    assert!((y - 100.0).abs() < 1e-10);
}
```

- [ ] **Step 7: Write Rust unit tests for selection state**

Create `crates/ferrum-wasm/tests/selection_state.rs`:

```rust
use ferrum_scene::*;
use ferrum_wasm::selection_state::*;

fn make_point_selection_spec(name: &str) -> SelectionSpec {
    SelectionSpec::Point {
        name: name.to_string(),
        fields: None,
        encodings: None,
        nearest: false,
        toggle: EventExpr::ShiftKey,
        on: EventExpr::Click,
        clear: EventExpr::Mouseout,
        resolve: SelectionResolve::Global,
    }
}

#[test]
fn initial_state_is_empty() {
    let scene = SceneGraph {
        selections: vec![make_point_selection_spec("sel1")],
        width: 400.0, height: 300.0, background: None,
        title: vec![], panels: vec![], legend: vec![],
        decorations: vec![],
        interaction: InteractionConfig::default(),
    };
    let state = InteractionState::from_scene(&scene);
    assert!(matches!(state.selections.get("sel1"), Some(SelectionState::Empty)));
}

#[test]
fn point_selection_add_and_clear() {
    let mut sel = SelectionState::Empty;
    apply_point_selection(&mut sel, 5, vec![], false);
    assert!(matches!(&sel, SelectionState::Point { indices, .. } if indices == &[5]));

    sel = SelectionState::Empty;
    assert!(matches!(sel, SelectionState::Empty));
}

#[test]
fn point_selection_toggle() {
    let mut sel = SelectionState::Empty;
    apply_point_selection(&mut sel, 5, vec![], false);
    apply_point_selection(&mut sel, 7, vec![], true);
    assert!(matches!(&sel, SelectionState::Point { indices, .. } if indices == &[5, 7]));

    // Toggle off index 5
    apply_point_selection(&mut sel, 5, vec![], true);
    assert!(matches!(&sel, SelectionState::Point { indices, .. } if indices == &[7]));
}

#[test]
fn interval_selection_contains() {
    let state = InteractionState {
        selections: [("brush".to_string(), SelectionState::Interval {
            x_range: Some((10.0, 50.0)),
            y_range: Some((20.0, 80.0)),
        })].into_iter().collect(),
        panel_transforms: vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        hover: None,
    };
    assert!(state.is_in_interval("brush", 30.0, 50.0));
    assert!(!state.is_in_interval("brush", 5.0, 50.0));
    assert!(!state.is_in_interval("brush", 30.0, 90.0));
}
```

- [ ] **Step 8: Verify compilation**

```bash
cargo build -p ferrum-wasm
cargo test -p ferrum-wasm
```

Expected: all new tests pass.

- [ ] **Step 9: Commit**

```
feat(wasm): add selection state machine and per-mark-type hit testing
```

---

## Task 11c2: Conditional encoding resolution + GPU buffer updates

When selection state changes, the WASM renderer updates per-node visual properties in the GPU instance buffer without rebuilding the SceneGraph.

**Files:**
- Create: `crates/ferrum-wasm/src/conditional.rs`
- Modify: `crates/ferrum-wasm/src/renderer.rs` (call `resolve_conditionals()` after selection changes)

### Steps

- [ ] **Step 1: Implement resolve_conditionals**

Create `crates/ferrum-wasm/src/conditional.rs`:

```rust
use ferrum_scene::*;
use crate::selection_state::{InteractionState, SelectionState};

/// Per-node override computed from conditional encoding resolution.
pub struct NodeOverride {
    pub batch_index: usize,
    pub node_index: usize,
    pub channel: ChannelName,
    pub value: EncodingValue,
}

/// Resolve all conditional encodings against current selection state.
/// Returns a list of per-node overrides that need to be applied to GPU buffers.
pub fn resolve_conditionals(
    scene: &SceneGraph,
    interaction: &InteractionState,
) -> Vec<NodeOverride> {
    let mut overrides = Vec::new();

    for cond in &scene.interaction.conditionals {
        let Some(sel) = interaction.selections.get(&cond.selection_name) else { continue };

        for panel in &scene.panels {
            for (batch_idx, batch) in panel.marks.iter().enumerate() {
                let Some(indices) = &batch.data_indices else { continue };
                for (node_idx, &data_idx) in indices.iter().enumerate() {
                    let selected = match sel {
                        SelectionState::Empty => false,
                        SelectionState::Point { indices: sel_indices, .. } => {
                            sel_indices.contains(&data_idx)
                        }
                        SelectionState::Interval { x_range, y_range } => {
                            // For interval selections, all marks within the panel
                            // are considered "selected" -- the interval bounds
                            // are checked at a higher level (data-space coordinates).
                            true
                        }
                    };
                    let value = if selected { &cond.if_selected } else { &cond.if_not };
                    overrides.push(NodeOverride {
                        batch_index: batch_idx,
                        node_index: node_idx,
                        channel: cond.channel,
                        value: value.clone(),
                    });
                }
            }
        }
    }
    overrides
}
```

- [ ] **Step 2: Apply overrides to GPU instance buffers**

In `crates/ferrum-wasm/src/renderer.rs`, add a method that applies `NodeOverride`s by modifying the instance buffer in-place:

```rust
impl WasmRenderer {
    pub fn resolve_conditionals(&mut self) {
        let overrides = conditional::resolve_conditionals(
            &self.scene.graph, &self.interaction,
        );

        for ov in &overrides {
            self.apply_override(ov);
        }

        // Upload modified buffers to GPU (one write_buffer call per batch)
        self.upload_modified_instances();
    }

    fn apply_override(&mut self, ov: &conditional::NodeOverride) {
        // Modify the instance data in the CPU-side buffer mirror
        let instance = &mut self.scene.gpu_buffers.instance_data[ov.batch_index][ov.node_index];
        match (&ov.channel, &ov.value) {
            (ChannelName::Color, EncodingValue::Color { value }) => {
                instance.color = [
                    value.r as f32 / 255.0,
                    value.g as f32 / 255.0,
                    value.b as f32 / 255.0,
                    value.a as f32 / 255.0,
                ];
            }
            (ChannelName::Opacity, EncodingValue::Opacity { value }) => {
                instance.opacity = *value as f32;
            }
            (ChannelName::Size, EncodingValue::Size { value }) => {
                instance.size = *value as f32;
            }
            _ => {} // Other channels handled as needed
        }
    }

    fn upload_modified_instances(&mut self) {
        // Write modified instance buffers to GPU via queue.write_buffer()
        for (batch_idx, instances) in self.scene.gpu_buffers.instance_data.iter().enumerate() {
            let data: &[u8] = bytemuck::cast_slice(instances);
            self.queue.write_buffer(
                &self.scene.gpu_buffers.instance_buffers[batch_idx],
                0,
                data,
            );
        }
    }
}
```

- [ ] **Step 3: Write unit tests for conditional resolution**

Create `crates/ferrum-wasm/tests/conditional.rs`:

```rust
use ferrum_scene::*;
use ferrum_wasm::conditional::*;
use ferrum_wasm::selection_state::*;

fn default_fill_stroke() -> FillStroke {
    FillStroke {
        fill: Some(Color::rgb(0, 0, 0)),
        stroke: None,
        stroke_width: 0.0,
        opacity: 1.0,
        stroke_dash: None,
    }
}

#[test]
fn no_conditionals_no_overrides() {
    let scene = SceneGraph {
        width: 100.0, height: 100.0, background: None,
        title: vec![], panels: vec![], legend: vec![],
        decorations: vec![], selections: vec![],
        interaction: InteractionConfig { conditionals: vec![], ..Default::default() },
    };
    let state = InteractionState::from_scene(&scene);
    let overrides = resolve_conditionals(&scene, &state);
    assert!(overrides.is_empty());
}

#[test]
fn point_selection_produces_overrides() {
    let scene = SceneGraph {
        width: 100.0, height: 100.0, background: None,
        title: vec![], legend: vec![], decorations: vec![],
        panels: vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            coord: CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true },
            grid: vec![], axes: vec![], annotations: vec![], strip_title: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![
                    SceneNode::Circle { cx: 10.0, cy: 10.0, r: 5.0, style: default_fill_stroke() },
                    SceneNode::Circle { cx: 20.0, cy: 20.0, r: 5.0, style: default_fill_stroke() },
                ],
                data_indices: Some(vec![0, 1]),
                tooltips: None, hrefs: None, descriptions: None,
                keys: None, blend: BlendMode::Normal,
                stroke_cap: None, stroke_join: None,
            }],
        }],
        selections: vec![],
        interaction: InteractionConfig {
            conditionals: vec![ConditionalEncoding {
                selection_name: "hover".to_string(),
                channel: ChannelName::Opacity,
                if_selected: EncodingValue::Opacity { value: 1.0 },
                if_not: EncodingValue::Opacity { value: 0.3 },
            }],
            ..Default::default()
        },
    };

    let mut state = InteractionState {
        selections: [("hover".to_string(), SelectionState::Point {
            indices: vec![0],
            field_values: vec![],
        })].into_iter().collect(),
        panel_transforms: vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        hover: None,
    };

    let overrides = resolve_conditionals(&scene, &state);
    assert_eq!(overrides.len(), 2); // one per node
    // Node 0 (selected): opacity 1.0
    assert!(matches!(&overrides[0].value, EncodingValue::Opacity { value } if (*value - 1.0).abs() < 1e-10));
    // Node 1 (not selected): opacity 0.3
    assert!(matches!(&overrides[1].value, EncodingValue::Opacity { value } if (*value - 0.3).abs() < 1e-10));
}
```

- [ ] **Step 4: Verify**

```bash
cargo test -p ferrum-wasm
```

- [ ] **Step 5: Commit**

```
feat(wasm): add conditional encoding resolution with GPU buffer updates
```

---

## Task 11c3: Zoom/pan + tick level selection

Zoom/pan is implemented via per-panel Affine2 transforms in the WASM renderer. Tick level selection swaps pre-computed tick labels based on the current zoom factor.

**Files:**
- Create: `crates/ferrum-wasm/src/zoom_pan.rs`
- Modify: `crates/ferrum-wasm/src/renderer.rs` (wire zoom/pan events)
- Modify: `src/ferrum/_wasm/ferrum-interactive.js` (add wheel + drag event handlers)

### Steps

- [ ] **Step 1: Implement zoom/pan state management**

Create `crates/ferrum-wasm/src/zoom_pan.rs`:

```rust
use ferrum_scene::{CoordKind, PanelTickLevels, TickLevel, SceneGraph};

/// Apply a zoom delta centered on cursor position to a panel transform.
///
/// `transform` is `[a, b, tx, c, d, ty]` representing the 2D affine:
///   | a  b  tx |
///   | c  d  ty |
///   | 0  0   1 |
///
/// Zoom is centered on `(cursor_x, cursor_y)` in pixel space.
pub fn apply_zoom(
    transform: &mut [f64; 6],
    delta: f64,
    cursor_x: f64,
    cursor_y: f64,
    coord: &CoordKind,
) {
    let factor = 1.0 + delta * 0.001;
    let factor = factor.clamp(0.1, 10.0); // prevent extreme zoom

    // For CoordFixed, constrain to uniform scale
    let (fx, fy) = match coord {
        CoordKind::Fixed { .. } => (factor, factor),
        _ => (factor, factor), // default: uniform zoom
    };

    // Translate to cursor, scale, translate back
    let (a, b, tx, c, d, ty) = (
        transform[0], transform[1], transform[2],
        transform[3], transform[4], transform[5],
    );

    transform[0] = a * fx;
    transform[1] = b * fy;
    transform[2] = cursor_x - fx * (cursor_x - tx);
    transform[3] = c * fx;
    transform[4] = d * fy;
    transform[5] = cursor_y - fy * (cursor_y - ty);
}

/// Apply a pan delta (drag) to a panel transform.
pub fn apply_pan(transform: &mut [f64; 6], dx: f64, dy: f64) {
    transform[2] += dx;
    transform[5] += dy;
}

/// Compute the current zoom level from a panel transform.
/// Returns the geometric mean of the x and y scale factors.
pub fn current_zoom_level(transform: &[f64; 6]) -> f64 {
    let sx = (transform[0] * transform[0] + transform[3] * transform[3]).sqrt();
    let sy = (transform[1] * transform[1] + transform[4] * transform[4]).sqrt();
    (sx * sy).sqrt()
}

/// Select the appropriate tick level for the current zoom factor.
/// Returns the index into `PanelTickLevels.x_levels` / `y_levels`.
pub fn select_tick_level(levels: &[TickLevel], zoom: f64) -> Option<usize> {
    levels.iter().position(|level| zoom >= level.min_zoom && zoom <= level.max_zoom)
}

/// Reset a panel transform to identity (1x zoom, no pan).
pub fn reset_transform(transform: &mut [f64; 6]) {
    *transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
}
```

- [ ] **Step 2: Wire zoom/pan into WasmRenderer**

In `renderer.rs`, add event handler methods:

```rust
#[wasm_bindgen]
impl WasmRenderer {
    pub fn on_wheel(&mut self, panel_id: usize, delta: f64, cursor_x: f64, cursor_y: f64) {
        if !self.scene.graph.interaction.zoom_enabled { return; }
        if panel_id >= self.interaction.panel_transforms.len() { return; }

        let coord = &self.scene.graph.panels[panel_id].coord;
        zoom_pan::apply_zoom(
            &mut self.interaction.panel_transforms[panel_id],
            delta, cursor_x, cursor_y, coord,
        );

        // Update tick level
        let zoom = zoom_pan::current_zoom_level(&self.interaction.panel_transforms[panel_id]);
        self.current_tick_level_x[panel_id] = self.scene.graph.interaction.tick_levels
            .iter()
            .find(|tl| tl.panel_id == panel_id)
            .and_then(|tl| zoom_pan::select_tick_level(&tl.x_levels, zoom));
        self.current_tick_level_y[panel_id] = self.scene.graph.interaction.tick_levels
            .iter()
            .find(|tl| tl.panel_id == panel_id)
            .and_then(|tl| zoom_pan::select_tick_level(&tl.y_levels, zoom));

        self.render_frame();
    }

    pub fn on_drag(&mut self, panel_id: usize, dx: f64, dy: f64) {
        if !self.scene.graph.interaction.pan_enabled { return; }
        if panel_id >= self.interaction.panel_transforms.len() { return; }

        zoom_pan::apply_pan(&mut self.interaction.panel_transforms[panel_id], dx, dy);
        self.render_frame();
    }

    pub fn on_dblclick(&mut self, panel_id: usize) {
        // Reset zoom/pan to identity
        if panel_id < self.interaction.panel_transforms.len() {
            zoom_pan::reset_transform(&mut self.interaction.panel_transforms[panel_id]);
            self.render_frame();
        }
    }

    /// Get the current tick level index for a panel axis.
    /// JS uses this to show/hide CSS tick label divs.
    pub fn get_tick_level_x(&self, panel_id: usize) -> Option<usize> {
        self.current_tick_level_x.get(panel_id).copied().flatten()
    }

    pub fn get_tick_level_y(&self, panel_id: usize) -> Option<usize> {
        self.current_tick_level_y.get(panel_id).copied().flatten()
    }

    /// Get the current panel transform as a flat array for CSS text positioning.
    pub fn get_panel_transform(&self, panel_id: usize) -> Vec<f64> {
        self.interaction.panel_transforms.get(panel_id)
            .map(|t| t.to_vec())
            .unwrap_or_default()
    }
}
```

- [ ] **Step 3: Update render_frame to apply panel transforms**

The GPU uniform buffer for the vertex shader includes a per-panel transform matrix. On zoom/pan, `render_frame()` writes the updated transform to the GPU:

```rust
impl WasmRenderer {
    fn render_frame(&mut self) {
        // Update per-panel uniform buffers with current transforms
        for (panel_id, transform) in self.interaction.panel_transforms.iter().enumerate() {
            let mat = transform_to_mat3x2(transform);
            self.queue.write_buffer(
                &self.pipelines.panel_uniform_buffers[panel_id],
                0,
                bytemuck::bytes_of(&mat),
            );
        }
        // ... existing render pass ...
    }
}
```

- [ ] **Step 4: Add JS event handlers for wheel and drag**

In `src/ferrum/_wasm/ferrum-interactive.js`, add event listeners to the canvas:

```javascript
// Inside the render() function or equivalent initialization:

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const panelId = renderer.panelAtPoint(x, y);
    if (panelId !== null) {
        renderer.on_wheel(panelId, e.deltaY, x, y);
        updateTextPositions(panelId);
        updateTickLabels(panelId);
    }
}, { passive: false });

let dragState = null;

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    dragState = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        panelId: renderer.panelAtPoint(e.clientX - rect.left, e.clientY - rect.top),
    };
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (dragState && dragState.panelId !== null) {
        const dx = x - dragState.x;
        const dy = y - dragState.y;
        renderer.on_drag(dragState.panelId, dx, dy);
        updateTextPositions(dragState.panelId);
        updateTickLabels(dragState.panelId);
        dragState.x = x;
        dragState.y = y;
    }
});

canvas.addEventListener('mouseup', () => {
    dragState = null;
});

canvas.addEventListener('dblclick', (e) => {
    const rect = canvas.getBoundingClientRect();
    const panelId = renderer.panelAtPoint(e.clientX - rect.left, e.clientY - rect.top);
    if (panelId !== null) {
        renderer.on_dblclick(panelId);
        updateTextPositions(panelId);
        updateTickLabels(panelId);
    }
});

function updateTextPositions(panelId) {
    // Get panel transform from WASM, apply as CSS transform to text overlay divs
    const t = renderer.get_panel_transform(panelId);
    const textEls = overlay.querySelectorAll('[data-panel="' + panelId + '"]');
    textEls.forEach(el => {
        const ox = parseFloat(el.dataset.origX);
        const oy = parseFloat(el.dataset.origY);
        const nx = t[0] * ox + t[1] * oy + t[2];
        const ny = t[3] * ox + t[4] * oy + t[5];
        el.style.left = nx + 'px';
        el.style.top = ny + 'px';
    });
}

function updateTickLabels(panelId) {
    // Show/hide tick label divs based on current tick level
    const xlevel = renderer.get_tick_level_x(panelId);
    const ylevel = renderer.get_tick_level_y(panelId);
    const tickEls = overlay.querySelectorAll('[data-panel="' + panelId + '"][data-tick-level]');
    tickEls.forEach(el => {
        const axis = el.dataset.tickAxis;
        const level = parseInt(el.dataset.tickLevel);
        const currentLevel = axis === 'x' ? xlevel : ylevel;
        el.style.display = (level === currentLevel) ? '' : 'none';
    });
}
```

- [ ] **Step 5: Verify**

```bash
cargo build -p ferrum-wasm --target wasm32-unknown-unknown
cargo test -p ferrum-wasm  # native tests for zoom_pan math
```

- [ ] **Step 6: Commit**

```
feat(wasm): add zoom/pan with per-panel Affine2 transforms and tick level selection
```

---

## Task 11c4: Tooltips + href click-through (JS)

Tooltip rendering is a CSS div positioned near the cursor. Href click-through opens URLs in a new tab.

**Files:**
- Create: `src/ferrum/_wasm/ferrum-interactive.css`
- Modify: `src/ferrum/_wasm/ferrum-interactive.js` (add tooltip div management, href navigation)

### Steps

- [ ] **Step 1: Create tooltip CSS**

Create `src/ferrum/_wasm/ferrum-interactive.css`:

```css
.ferrum-root {
    position: relative;
    display: inline-block;
}

.ferrum-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    overflow: hidden;
}

.ferrum-overlay > * {
    pointer-events: auto;
}

.ferrum-tooltip {
    position: absolute;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 6px 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 12px;
    line-height: 1.4;
    color: #333;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    pointer-events: none;
    z-index: 100;
    max-width: 300px;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.1s ease;
}

.ferrum-tooltip.visible {
    opacity: 1;
}

.ferrum-tooltip table {
    border-collapse: collapse;
}

.ferrum-tooltip td {
    padding: 1px 6px 1px 0;
}

.ferrum-tooltip td:first-child {
    font-weight: 600;
    color: #555;
}

.ferrum-tooltip td:last-child {
    text-align: right;
}

.ferrum-text {
    position: absolute;
    white-space: nowrap;
    pointer-events: none;
}

.ferrum-text[data-tick-level] {
    pointer-events: none;
}
```

- [ ] **Step 2: Add tooltip div management to JS**

In `ferrum-interactive.js`, add tooltip creation, positioning, and content update:

```javascript
// Create tooltip element (once during init)
const tooltip = document.createElement('div');
tooltip.className = 'ferrum-tooltip';
root.appendChild(tooltip);

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const panelId = renderer.panelAtPoint(x, y);

    if (panelId !== null && !dragState) {
        const result = renderer.on_mousemove(panelId, x, y, false);
        if (result && result.fields && result.fields.length > 0) {
            // Build tooltip content as safe DOM nodes (no raw HTML injection)
            const table = document.createElement('table');
            result.fields.forEach(f => {
                const tr = document.createElement('tr');
                const tdName = document.createElement('td');
                tdName.textContent = f.name;
                const tdVal = document.createElement('td');
                tdVal.textContent = f.value;
                tr.appendChild(tdName);
                tr.appendChild(tdVal);
                table.appendChild(tr);
            });
            tooltip.replaceChildren(table);

            // Position tooltip near cursor (offset to avoid overlap)
            const tooltipX = Math.min(x + 12, rect.width - tooltip.offsetWidth - 8);
            const tooltipY = Math.max(y - tooltip.offsetHeight - 8, 8);
            tooltip.style.left = tooltipX + 'px';
            tooltip.style.top = tooltipY + 'px';
            tooltip.classList.add('visible');
        } else {
            tooltip.classList.remove('visible');
        }
    } else {
        tooltip.classList.remove('visible');
    }
});

canvas.addEventListener('mouseleave', () => {
    tooltip.classList.remove('visible');
});
```

- [ ] **Step 3: Add href click-through**

In `ferrum-interactive.js`, extend the click handler:

```javascript
canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const panelId = renderer.panelAtPoint(x, y);

    if (panelId !== null) {
        const result = renderer.on_click(panelId, x, y, e.shiftKey);
        if (result) {
            // Check for navigation action
            if (typeof result === 'object' && result.navigate) {
                window.open(result.navigate, '_blank', 'noopener,noreferrer');
                return;
            }
            // Selection update -- sync to anywidget model if available
            if (model && result.selection_state) {
                model.set('selection_state', result.selection_state);
                model.save_changes();
            }
        }
    }
});
```

- [ ] **Step 4: Wire cursor style changes**

Add cursor style feedback for interactive elements:

```javascript
// Inside the mousemove handler, after tooltip logic:
const hit = renderer.hit_test_at(panelId, x, y);
if (hit && hit.has_href) {
    canvas.style.cursor = 'pointer';
} else if (dragState) {
    canvas.style.cursor = 'grabbing';
} else if (renderer.is_pannable()) {
    canvas.style.cursor = 'grab';
} else {
    canvas.style.cursor = 'default';
}
```

- [ ] **Step 5: Commit**

```
feat(wasm): add tooltip rendering and href click-through in JS overlay
```

---

## Task 11c5: Python selection API (selection.py)

The Python selection API provides `selection_point()`, `selection_interval()`, and the `Selection` class with conditional encoding builder. This task also wires the `condition` kwarg on appearance channels and flows selections through `chart.py` into `ChartSpec`.

**Parallelism note:** This task is parallelizable with 11c1-11c4. It depends only on 11c0 (ChartSpec field additions) and touches only Python code -- no ferrum-wasm dependency.

**Files:**
- Create: `src/ferrum/selection.py`
- Modify: `src/ferrum/chart.py` (wire `add_selection()`, `interactive()`, `to_spec()`)
- Modify: `src/ferrum/encoding/base.py` (wire `condition` kwarg on `ChannelBase`)
- Modify: `src/ferrum/__init__.py` (export new symbols)
- Create: `tests/test_selection_api.py`

### Steps

- [ ] **Step 1: Create selection.py**

Create `src/ferrum/selection.py` with:

- `SelectionMark` (frozen dataclass): brush overlay style with `fill`, `stroke`, `fill_opacity`, `stroke_opacity`, `stroke_width`, `stroke_dash` fields, plus a `to_dict()` method that serializes to the ferrum-scene `SelectionMarkStyle` JSON shape.

- `Selection` (frozen dataclass): `name: str`, `kind: Literal["point", "interval"]`, `params: dict`. Has a `.when(if_encoding)` method returning `_SelectionCondition`, and a `.to_dict()` method serializing to ferrum-scene `SelectionSpec` JSON.

- `_SelectionCondition` (frozen dataclass): `selection: Selection`, `if_encoding: Any`. Has `.otherwise(else_encoding)` returning `ConditionalSpec`.

- `ConditionalSpec` (frozen dataclass): `selection_name: str`, `if_selected: Any`, `if_not: Any`. Has `.to_dict(channel_name: str)` returning ferrum-scene `ConditionalEncoding` JSON.

- `value(v)` function: wraps a literal (CSS color string, numeric opacity/size, dash array list) into an `_ValueLiteral` for use in conditional encoding chains. `value("#ccc")` returns kind="color", `value(0.3)` returns kind="opacity", `value([4, 2])` returns kind="stroke_dash".

- `_encoding_to_value(enc)` helper: converts a `ChannelBase` instance (field-based) or `_ValueLiteral` (literal) to the `EncodingValue` JSON shape.

- `selection_point(*, fields, encodings, nearest, toggle, on, clear, resolve, name)` factory: returns a `Selection` with `kind="point"`. Auto-generates name if None. Normalizes event expressions (`"event.shiftKey"` -> `"shift_key"`, `"click"` -> `"click"`, etc.).

- `selection_interval(*, fields, encodings, translate, zoom, mark, resolve, name)` factory: returns a `Selection` with `kind="interval"`.

- `selection_single = partial(selection_point, toggle=False)` convenience alias.

- `selection_multi = partial(selection_point, toggle="event.shiftKey")` convenience alias.

Full parameter signatures match spec section 10.2.

- [ ] **Step 2: Wire condition kwarg on ChannelBase**

In `src/ferrum/encoding/base.py`, update `ChannelBase.__init__` to validate the `condition` kwarg:

```python
class ChannelBase:
    def __init__(self, field=None, *, condition=None, **kwargs):
        # ... existing init logic ...
        self._condition = None
        if condition is not None:
            from ferrum.selection import ConditionalSpec
            if not isinstance(condition, ConditionalSpec):
                raise TypeError(
                    "condition= must be a ConditionalSpec "
                    "(from sel.when(...).otherwise(...)), "
                    f"got {type(condition).__name__}"
                )
            self._condition = condition
```

Add a method to extract the conditional encoding:

```python
    def get_conditional(self):
        """Return the conditional encoding spec, or None."""
        return self._condition
```

- [ ] **Step 3: Wire selections and conditionals into chart.py**

**A. Add `_selections` and `_conditionals` state:**

In the `Chart` class `__init__` (and `_clone()`), add:
```python
self._selections: list = []
self._conditionals: list = []
```

**B. Wire `add_selection()`:**

Replace the current stub (lines ~4674-4702) with a real implementation that validates each argument as a `Selection` instance, clones the chart, appends the selections, and returns the clone.

**C. Wire `interactive()`:**

Replace the current stub (lines ~4704-4724) with:
```python
def interactive(self):
    from ferrum._interactive import InteractiveChart
    return InteractiveChart.from_chart(self)
```

**D. Wire selections and conditionals through `to_spec()`:**

In `to_spec()`, before the `return ChartSpec(**kw)` line (around line 4485), add logic to:
1. Serialize `self._selections` to JSON via `json.dumps([s.to_dict() for s in resolved._selections])` and pass as `kw["selections"]`
2. Collect conditionals from encoding channels: iterate `enc` dict, call `ch.get_conditional()` on each, serialize any non-None results to JSON via `json.dumps(...)` and pass as `kw["conditionals"]`

- [ ] **Step 4: Update __init__.py exports**

In `src/ferrum/__init__.py`, add:

```python
from ferrum.selection import (
    selection_point,
    selection_interval,
    selection_single,
    selection_multi,
    Selection,
    SelectionMark,
    value,
)
```

Add all to `__all__`. Do NOT export `InteractiveChart` here yet -- defer to 11c6 when the anywidget dependency is added.

- [ ] **Step 5: Write Python unit tests**

Create `tests/test_selection_api.py` with test classes covering:

- **TestSelectionConstruction**: `selection_point()` defaults, custom name, `selection_interval()` defaults, `selection_single` is no-toggle, `selection_multi` is shift-toggle, `SelectionMark` defaults.

- **TestSelectionSerialization**: `to_dict()` on point and interval selections produces correct JSON shape. Interval with `SelectionMark` includes nested mark dict.

- **TestConditionalEncoding**: `.when().otherwise()` chain produces `ConditionalSpec`. `to_dict(channel_name)` includes correct `selection_name`, `channel`, `if_selected`, `if_not`. `value("#ccc")` produces color kind. `value(0.5)` produces opacity kind. `value([4, 2])` produces stroke_dash kind.

- **TestChartIntegration**: `add_selection()` stores selections on cloned chart. `add_selection()` rejects non-Selection args with TypeError. `add_selection()` is immutable (original chart unchanged). `to_json()` on a chart with selections includes `selections` field.

- [ ] **Step 6: Verify**

```bash
uv run pytest tests/test_selection_api.py -v
uv run pytest tests/ -v  # ensure no regressions
```

- [ ] **Step 7: Commit**

```
feat(python): add selection API -- selection_point, selection_interval, conditional encodings
```

---

## Task 11c6: InteractiveChart anywidget class + bidirectional state sync

The `InteractiveChart` is an anywidget subclass that renders the chart via the WASM renderer in Jupyter and provides bidirectional Python-JS state sync for selections.

**Files:**
- Create: `src/ferrum/_interactive.py`
- Modify: `pyproject.toml` (add `anywidget>=0.9` dependency)
- Modify: `src/ferrum/__init__.py` (export `InteractiveChart`)
- Modify: `src/ferrum/display.py` (wire `.save("chart.html")`)

### Steps

- [ ] **Step 1: Add anywidget dependency**

In `pyproject.toml`, add to `[project] dependencies`:

```toml
"anywidget>=0.9",
```

Then run `uv sync` to install.

- [ ] **Step 2: Create _interactive.py**

Create `src/ferrum/_interactive.py` with:

- `InteractiveChart(anywidget.AnyWidget)` class:
  - `_esm` points to `_wasm/ferrum-interactive.js`
  - `_css` points to `_wasm/ferrum-interactive.css`
  - Traitlets: `scene_json` (Unicode, synced), `interaction_config` (Unicode, synced), `selection_state` (Dict, synced)
  - `from_chart(cls, chart)` classmethod: calls `render_interactive()` on the chart's spec/data/theme, parses the JSON, sets `scene_json` and `interaction_config`, stores the originating chart for recomputation
  - `save(self, path, *, embed_wasm=True)`: generates standalone HTML with inline WASM (base64) or sidecar WASM file. The HTML template contains the CSS, a `<canvas>` and overlay `<div>`, and a `<script type="module">` that initializes the WASM renderer with the scene JSON.
  - `on_selection_change(self, callback)`: registers a traitlets observer on `selection_state`
  - `_on_selection_state_change(self, change)`: handles recomputation requests from JS (e.g., zoom into mark_function)

- `_check_wasm_available()` helper: raises `WasmNotAvailableError` if `_wasm/ferrum_wasm.js` is missing.

- [ ] **Step 3: Update display.py for HTML and JSON save**

In `src/ferrum/display.py`, replace the `NotImplementedError` for `"html"` format with logic that creates an `InteractiveChart` from the chart and calls `widget.save(path)`. Replace the `"json"` format with `path.write_text(chart.to_json(indent=2))`.

- [ ] **Step 4: Update __init__.py exports**

Add `from ferrum._interactive import InteractiveChart` and add `"InteractiveChart"` to `__all__`.

- [ ] **Step 5: Wire anywidget model bridge in JS**

In `ferrum-interactive.js`, the `render({ model, el })` function detects Jupyter mode (`model` is present) vs. standalone mode:
- In Jupyter: observe `model.get('scene_json')` for Python-initiated scene updates; push selection state changes to `model.set('selection_state', ...)`.
- In standalone: no model bridge; selection state is local to the browser session.

- [ ] **Step 6: Verify**

```bash
uv sync
uv run pytest tests/ -v
uv run python -c "from ferrum._interactive import InteractiveChart; print('OK')"
```

- [ ] **Step 7: Commit**

```
feat(python): add InteractiveChart anywidget class with bidirectional state sync
```

---

## Task 11c7: Compound view scene graph merging

Compound views (`HConcatChart`, `VConcatChart`, `FacetChart`, `RepeatChart`, `JointChart`, `ClusterMapChart`) need to produce a single unified SceneGraph for the WASM renderer. This is a pure Python data transformation -- no Rust changes needed.

**Files:**
- Modify: `src/ferrum/_interactive.py` (add `merge_scene_graphs()`)
- Modify: `src/ferrum/composition.py` (add `.interactive()` to compound chart classes)

### Steps

- [ ] **Step 1: Implement merge_scene_graphs()**

**Known limitation:** `SceneNode::Raw` nodes (used for legend colorbar gradients -- see 11a design seam) are not offset during merging because they contain literal SVG with absolute coordinates. Legend colorbars in compound interactive views will be mis-positioned until Raw is replaced with typed gradient nodes (tracked as a 11a design seam, addressed in a future sub-phase).

Add `merge_scene_graphs(scene_graphs, layout, shared_encodings)` to `_interactive.py`. This function:

1. Takes a list of SceneGraph dicts (deserialized JSON) and a list of layout dicts with `x_offset`, `y_offset`, `width`, `height`.
2. Renumbers panel IDs sequentially across all sub-charts.
3. Offsets all node coordinates (in panels, title, legend, decorations) by each sub-chart's layout offset. Each SceneNode type has its own offset logic: Rect offsets x/y, Circle offsets cx/cy, Line offsets x1/y1/x2/y2, Text offsets x/y, Path offsets all coordinates in PathCmd, Polyline offsets all points, Polygon offsets all points, Group recursively offsets children, Raw nodes are NOT offset (they contain literal SVG).
4. Merges `InteractionConfig`: unions conditionals and tick_levels, computes `linked_panels` groups based on `shared_encodings`.
5. Merges selections from all sub-charts.
6. Returns a single SceneGraph dict with computed total width/height.

- [ ] **Step 2: Add .interactive() to compound chart classes**

In `src/ferrum/composition.py`, add `interactive()` methods to `HConcatChart`, `VConcatChart`, and other compound types. Each follows the pattern:

1. Iterate sub-charts, call `render_interactive()` on each to get per-sub-chart SceneGraph JSON.
2. Compute layout offsets (horizontal stacking for HConcat, vertical for VConcat, grid for Repeat/Facet). Reuse the same spacing logic as the existing SVG compositor.
3. Call `merge_scene_graphs()` with the collected SceneGraphs and layout info.
4. Create an `InteractiveChart` widget, set `scene_json` and `interaction_config` from the merged result.

- [ ] **Step 3: Verify**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 4: Commit**

```
feat(interactive): add merge_scene_graphs and compound view .interactive() methods
```

---

## Task 11c8: Animated transitions (Key channel)

When a new SceneGraph arrives (data update via anywidget), the WASM renderer diffs old and new `MarkBatch` nodes using the `keys` array for object constancy, then lerps position/size/color/opacity between old and new values.

**Files:**
- Create: `crates/ferrum-wasm/src/transition.rs`
- Modify: `crates/ferrum-wasm/src/renderer.rs` (wire `transition_scene()`)
- Modify: `crates/ferrum-wasm/src/lib.rs` (add module)

### Steps

- [ ] **Step 1: Implement transition diffing**

Create `crates/ferrum-wasm/src/transition.rs` with:

- `NodeState` struct: extracted visual state for interpolation (x, y, w, h, r, color: [f32; 4], opacity: f32). Has `from_node(node: &SceneNode)` constructor that extracts position/size/color from Circle and Rect nodes. Has `lerp(&self, other: &Self, t: f64)` for linear interpolation.

- `Transition` struct: `batch_index`, `node_index`, `from: NodeState`, `to: NodeState`.

- `diff_scenes(old: &SceneGraph, new: &SceneGraph)` function: zips panels and mark batches, uses `MarkBatch.keys` to establish object constancy. Returns three lists:
  - Matched transitions: key exists in both old and new (interpolate from old to new state)
  - Enter nodes: key in new but not old (fade in from opacity 0)
  - Exit nodes: key in old but not new (fade out to opacity 0)

The key matching uses a `HashMap<&str, usize>` built from the old batch's keys for O(1) lookup.

- [ ] **Step 2: Wire transition into WasmRenderer**

In `renderer.rs`:

- `transition_scene(&mut self, new_scene: SceneGraph, duration_ms: u32)`: calls `diff_scenes()`, stores an `ActiveTransition` struct if any transitions exist, otherwise does a hard swap via `load_scene()`.

- `tick_transition(&mut self, timestamp_ms: f64) -> bool`: called each frame during an active transition. Computes `t = elapsed / duration` with cubic ease-in-out, calls `lerp()` on each transition, writes interpolated states to instance buffers, calls `render_frame()`. Returns true if still in progress, false when complete. On completion, finalizes by calling `load_scene()` with the new scene.

- `ActiveTransition` struct: `transitions`, `enter`, `exit`, `new_scene`, `start_time: Option<f64>`, `duration: Duration`.

- [ ] **Step 3: Wire requestAnimationFrame loop in JS**

In `ferrum-interactive.js`, add a `startTransition(renderer, newScene, durationMs)` function that:
1. Calls `renderer.transition_scene(newScene, durationMs)`
2. Starts a `requestAnimationFrame` loop calling `renderer.tick_transition(timestamp)` until it returns false

Wire this into the anywidget model observer for `scene_json` changes (Jupyter mode) so data updates trigger smooth transitions.

Default transition duration: 300ms (configurable via theme `theme.transition_duration` in the future).

- [ ] **Step 4: Verify**

```bash
cargo build -p ferrum-wasm
cargo test -p ferrum-wasm
```

- [ ] **Step 5: Commit**

```
feat(wasm): add animated transitions via Key channel with object constancy
```

---

## Validation checklist (run before marking 11c done)

### Rust compilation and tests

- [ ] `cargo build -p ferrum-scene` -- no new warnings
- [ ] `cargo build -p ferrum-core` -- compiles with new ChartSpec fields
- [ ] `cargo build -p ferrum-wasm` -- compiles with all new modules
- [ ] `cargo build -p ferrum-wasm --target wasm32-unknown-unknown` -- WASM target compiles
- [ ] `cargo test -p ferrum-scene` -- serde round-trip including new fields
- [ ] `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core` -- all existing tests pass
- [ ] `cargo test -p ferrum-wasm` -- hit testing, selection state, conditional resolution tests pass

### Python tests

- [ ] `uv run pytest tests/test_selection_api.py -v` -- selection API serialization tests
- [ ] `uv run pytest tests/ -v` -- full test suite (no regressions from ChartSpec changes)
- [ ] `uv run python -c "from ferrum.selection import selection_point; print(selection_point(name='test'))"` -- selection API importable
- [ ] `uv run python -c "from ferrum._interactive import InteractiveChart; print('OK')"` -- InteractiveChart importable

### Golden SVG stability

- [ ] `uv run pytest tests/ -k golden -v` -- all golden SVG tests still pass
- [ ] No byte differences in any golden SVG -- the new ChartSpec fields are `serde(default, skip_serializing_if)`, so existing specs produce identical output

### Integration (manual browser verification)

- [ ] Create a scatter plot with `selection_point(nearest=True)`, call `.interactive()` -- verify point highlight on hover in Jupyter
- [ ] Create a chart with `selection_interval()`, call `.interactive()` -- verify brush selection works
- [ ] Create a chart with conditional color encoding -- verify selected/unselected marks have different colors
- [ ] Verify zoom (scroll wheel) and pan (drag) on a chart with `selection_interval(zoom=True)`
- [ ] Verify tooltip appears on hover over marks with `Tooltip(field)` encoding
- [ ] Verify href click-through opens a URL in a new tab
- [ ] Verify `.save("chart.html")` produces a standalone HTML file that renders in a browser
- [ ] Verify compound view (HConcat) `.interactive()` produces a single unified widget
- [ ] Verify animated transition by updating data on an `InteractiveChart` with Key encoding

### anywidget integration (Jupyter)

- [ ] `InteractiveChart` displays in JupyterLab
- [ ] Selection state syncs from JS to Python (`widget.selection_state` updates)
- [ ] Python-side scene update syncs to JS (`widget.scene_json = ...` triggers re-render)
