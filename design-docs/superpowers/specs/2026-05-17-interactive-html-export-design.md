# Interactive HTML Export Design Spec

## 1. Scope

Fix the broken HTML export path so standalone `.html` files render charts with full WASM-backed interactivity (tooltips, pan/zoom, click selection, brush selection, conditional encodings). Extend `.interactive()` and `.save("file.html")` to composition types (`HConcatChart`, `VConcatChart`, `LayerChart`, `FacetChart`, `RepeatChart`, `ConcatChart`) so linked selections work across panels in exported HTML.

## 2. Goals

- Any chart that works with `.interactive()` in Jupyter must produce a working standalone HTML file via `.interactive().save("out.html")` or `.save("out.html", format="html")`.
- Composition types gain `.interactive()` and HTML save support.
- Linked selections across composed panels work in both Jupyter and HTML export.
- The Jupyter widget path (`ferrum-anywidget.js`) and the HTML standalone path share one rendering function — no duplicated interaction logic.
- Packed binary data (large mark batches) is correctly threaded through the HTML export path.

## 3. Non-goals

- Server-backed rendering (Streamlit, Dash, Panel embedding).

## 4. System behavior

### Single chart

`chart.interactive().save("out.html")` produces a self-contained HTML file. Opening it in a browser renders the chart on a GPU canvas with:

- **Tooltips** on hover (WASM hit-test → `getTooltip`).
- **Pan** via click-drag (`onPan`).
- **Zoom** via mousewheel (`onWheel`).
- **Double-click** to reset zoom (`resetZoom`).
- **Point selection** via click (`handleClick`) — conditional encodings update visually.
- **Interval/brush selection** via click-drag (`handleDrag`) — dragging creates a brush rectangle; marks inside are selected; conditional encodings update. The brush rectangle is drawn as a CSS overlay.
- **Href** navigation on click (JS hit-test on scene JSON nodes).

### Composition types

`(scatter | bars).interactive().save("out.html")` produces an HTML file where both panels render inside a single WASM canvas. Clicking a mark in one panel updates selection state and applies conditional encodings across all panels in the same scene.

This works because composition types produce a single `SceneGraph` with multiple `Panel` entries via `render_interactive`. The WASM renderer's `handleClick` already iterates all panels and broadcasts selection state. No new Rust/WASM API is needed.

### Jupyter

No behavior change. The Jupyter path continues to use `ferrum-anywidget.js` via the anywidget ESM protocol (`export async function render({ model, el })`). The shared rendering function is called from both the anywidget `render` export and the standalone HTML `main()`.

## 5. Architecture

### Data flow

```
Python                          Rust                        Browser
──────                          ────                        ───────
chart._render_inputs()    →  render_interactive()      →  (scene_json, packed_bytes)
  or                            ↓
composition._render_inputs()    single SceneGraph with
                                multiple panels

                           assemble_html(scene_json,   →  self-contained .html
                                         packed_data)      with inlined WASM +
                                                           scene + packed data
```

### Component responsibilities

**`_render_interactive()` on compositions:** Each composition type implements `_render_interactive() -> (str, bytes)` that returns a merged scene JSON and concatenated packed data. `LayerChart` delegates to `render_interactive` directly (layers are native to `ChartSpec`). `HConcatChart`/`VConcatChart`/`ConcatChart` call `render_interactive` on each child, then merge the resulting scene JSONs in Python — offsetting panel positions, re-indexing panel IDs, and concatenating packed byte streams. `FacetChart`/`RepeatChart` already produce multi-panel scenes natively via the facet spec field. The merge logic lives in a shared `_merge_scenes()` helper.

**`_render_scene()` in `_interactive.py`:** Already works — it calls `chart._render_inputs()` and `render_interactive()`. Once compositions implement `_render_interactive()`, this function handles them without changes.

**Interval conditional resolution in WASM:** The current `resolve_conditionals` calls `SelectionState::contains(data_idx)`, which always returns `false` for interval selections. To support brush selection conditional encodings, `resolve_conditionals` must accept mark positions alongside data indices. For interval selections, containment is: mark position `(x, y)` falls within the brush `x_range × y_range`. Mark positions are available from `SceneNode` coordinates (circle `cx/cy`, rect `x/y`) or from packed `CircleInstance`/`RectInstance` data. The spatial check replaces the index-based check for `SelectionState::Interval` only.

**`assemble_html()` in `_html.py`:** Accepts `scene_json` and `packed_data`. Embeds packed data as a base64 string decoded to `Uint8Array` at runtime. The JS calls `renderer.loadScene(sceneJson, packedArr)` with both arguments.

**`ferrum-anywidget.js`:** The existing `_render(container, sceneJson, model)` function contains all interaction logic (tooltip, pan, zoom, click, selection). This function is refactored to accept a generic state adapter instead of the anywidget `model` object, so both Jupyter and standalone HTML can call it.

### Shared JS rendering — adapter pattern

The core rendering function accepts a state adapter with this interface:

```javascript
// State adapter consumed by the shared _render function
{
  getPackedData()          // → Uint8Array
  getInteractionConfig()   // → string (JSON)
  onSelectionChange(state) // called when selection state changes
  onZoomChange(state)      // called when zoom state changes
}
```

- **Jupyter adapter:** reads from / writes to `model.get()` / `model.set()` / `model.save_changes()`.
- **Standalone adapter:** packed data and interaction config are embedded at build time; selection/zoom callbacks update local state only (no Python round-trip).

This replaces the pattern of checking `model` at runtime and eliminates the need for a `standalone` flag.

## 6. Canonical interfaces / data contracts

### Composition `_render_interactive()`

```python
def _render_interactive(self) -> tuple[str, bytes]:
    """Return (scene_json, packed_data) for the full composition."""
```

Composition types that don't map to a single `ChartSpec` (HConcat, VConcat, Concat) render each child independently and merge scenes in Python. Types that do map natively (Layer via `ChartSpec.layers`, Facet/Repeat via `ChartSpec.facet`) delegate to `render_interactive` directly.

### `assemble_html` signature

```python
def assemble_html(
    scene_json: str,
    *,
    packed_data: bytes = b"",
    title: str = "Ferrum chart",
    embed_wasm: bool = True,
) -> str:
```

### `_render_scene_json` replaced

`display.py`'s `_render_scene_json` must return `(str, bytes)` — both the JSON and the packed data — so `save_chart` can pass both to `assemble_html`.

### WASM `handleDrag` method

```rust
#[wasm_bindgen(js_name = "handleDrag")]
pub fn handle_drag(&mut self, panel_id: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Result<String, JsValue>
```

Updates interval selection state via `InteractionState::handle_drag`, resolves conditional encodings with spatial containment (marks whose positions fall within the brush bounds are selected), rebuilds GPU buffers, re-renders, and returns selection state JSON. Same return contract as `handleClick`.

### `InteractiveChart` accepts compositions

`InteractiveChart.__init__` accepts any object that implements `_render_interactive()`. Charts already have this via `_render_scene()` in `_interactive.py`. Composition types gain it through their new `_render_interactive()` method. `InteractiveChart` calls a unified `_render_scene(chart_or_composition)` helper that dispatches appropriately.

## 7. Invariants and constraints

- **No duplicated JS interaction logic.** One rendering function, two adapters. The anywidget ESM file is the single source; the HTML export inlines it.
- **`loadScene` always receives packed data.** Even when empty (`new Uint8Array(0)`), the second argument is always passed. The current HTML path calling `loadScene(json)` with one argument is the root cause of the `TypeError: Cannot read properties of undefined (reading 'length')` error.
- **Packed data encoding in HTML:** base64 string → `atob` → `Uint8Array`. Decoded synchronously. For charts with <1000 marks per batch, packed data is empty (0 bytes) — this is fine.
- **New WASM method: `handleDrag`.** Exposes the existing `InteractionState::handle_drag` via `wasm_bindgen`. Accepts `(panel_id, x0, y0, x1, y1)`, updates interval selection state, applies conditional encodings, re-renders, and returns selection state JSON. Follows the same pattern as `handleClick`.
- **Interval `contains` must resolve spatially.** Currently `SelectionState::Interval::contains(data_idx)` always returns `false`. For conditional encodings to work with brush selections, the conditional resolution must test whether each mark's position falls within the brush bounds — not just check a data index. This requires passing mark positions (from scene nodes or packed instances) into the containment check alongside the brush `x_range`/`y_range`.
- **Composition SVG path unchanged.** `show_svg()` on compositions continues to work as before (independent child renders, SVG compositing). Only the interactive/HTML path uses the merged scene.
- **WASM method arity is a contract.** Any Rust change that adds/removes a parameter to a `wasm_bindgen` method (e.g., `handleClick` gaining `shift_held`) must update every JS call site in the same commit. The abandoned branch added `shift_held` to the Rust side but forgot to update `ferrum-anywidget.js`, crashing Jupyter.
- **Theme background baked into HTML at template time.** The background color is extracted from the scene JSON in Python and written into the `<body>`/container `<div>` style attributes. The JS renderer must not override it. This prevents the white-flash regression (Bug #8).
- **Text overlay must include rotation and baseline.** Text elements returned by `loadScene` include `angle` and `anchor` fields. The JS `_placeText` function must apply composite CSS transforms: `translateX` (anchor), `rotate` (angle), `translateY` (baseline). Missing any of these breaks axis labels (Bug #10).
- **ResizeObserver runs in both Jupyter and standalone.** It must not be gated behind a mode flag. The abandoned branch lost it inside `wireInteractions()` which only ran in standalone mode (Bug #14).
- **Mouse coordinates must account for CSS scaling.** All hit-test and interaction coordinates must scale by `canvas.width / canvas.getBoundingClientRect().width` to handle CSS-scaled canvases (iframes, responsive layouts). Missing this breaks tooltips, selection, and pan (Bug #13).
- **No matplotlib.** No global mutable state.

## 8. Key decisions and tradeoffs

**Unified spec vs. multi-renderer composition.** The abandoned branch attempted to render each child chart as a separate WASM renderer and synchronize selection state between them via `setSelectionState()`. This is rejected — it requires new WASM API, duplicates state management, and creates synchronization bugs. Instead, compositions produce a single `ChartSpec` rendered by one `WasmRenderer`. The Rust core already handles multi-panel layout and cross-panel selection broadcasting.

**Adapter pattern vs. runtime flag.** The abandoned branch used `options.standalone = true/false` to gate behavior. This created two code paths through one function, with bugs in each. The adapter pattern separates the *what* (rendering) from the *how* (state transport) cleanly.

**Inlining JS vs. separate file.** The HTML export inlines the JS from `ferrum-anywidget.js` rather than maintaining a separate `ferrum-interactive.js`. The abandoned branch had two JS files with diverging interaction logic — the root cause of many silent bugs. One file, inlined where needed.

**JS inlining method.** The anywidget ESM uses `export async function render({ model, el })`. For HTML inlining, the build step strips the export and wraps the standalone entry point. This is a simple string operation (replace the export with a `main()` call) rather than the abandoned branch's line-by-line munging, because the standalone path calls the same `_render()` internal function with a different adapter — the export wrapper is the only thing that changes.

**Brush rectangle rendering.** The brush overlay is drawn as a CSS-positioned `<div>` on the overlay layer, not rendered via the GPU. This matches the pattern used for text elements and tooltips — CSS overlays are simpler and don't require GPU pipeline changes. The brush div is created on `mousedown`, resized on `mousemove`, and finalized on `mouseup` (calling `handleDrag` with the final coordinates). The brush styling (fill, stroke, opacity) comes from the `SelectionMark` in the scene JSON's `SelectionSpec::Interval::mark` field.

**Pan vs. brush disambiguation.** When a chart has both an interval selection and pan enabled, the JS must disambiguate drag intent. The rule: if the chart has an active `selection_interval`, drag creates a brush; pan requires holding a modifier key (e.g., Alt/Option) or is disabled. This mirrors the Vega-Lite convention where selections take priority over pan.

**Composition rendering: multi-scene merge in Python.** `ChartSpec` has no `hconcat`/`vconcat` fields — it only supports `facet` and `layers`. Rather than adding composition operators to the Rust spec (large scope, touches layout engine), each composition type calls `render_interactive` on each child chart independently and merges the resulting `SceneGraph` JSONs in Python. The merge produces one combined `SceneGraph` with offset panels, concatenated packed data, and unified selections/conditionals. This is a Python-side compositor — analogous to how `compose_svg_horizontal` already works for SVG, but operating on scene JSON instead of SVG strings. The WASM renderer receives one merged scene and handles multi-panel selection broadcasting natively.

**LayerChart is the exception.** `LayerChart` already maps to `ChartSpec.layers` — it produces a single-panel scene natively. No merge step needed.

## 9. Acceptance criteria

1. `chart.interactive().save("out.html")` opens in a browser and renders with tooltips, pan, zoom, click selection, and conditional encoding updates — for every example in `docs/site/guide/interactive.md`.
2. Interval/brush selection works in HTML export: dragging creates a visible brush rectangle, marks inside are highlighted via conditional encodings, marks outside are dimmed.
3. `(scatter | bars).interactive().save("out.html")` renders both panels; clicking a point in the scatter updates conditional encodings in both panels.
4. `(scatter & bars).interactive()` works in Jupyter with linked selections.
5. Packed data (>1000 marks) renders correctly in HTML export — no `TypeError`.
6. Existing Jupyter `.interactive()` behavior is unchanged — no regressions in tooltip, pan, zoom, or selection sync.
7. `show_svg()` on all composition types is unchanged.
8. No duplicated interaction logic between Jupyter and HTML paths.

## 10. Validation strategy — regression-first testing

The abandoned branch (`feat/linked-selection-html-export`) introduced 15 distinct regressions across theme handling, Jupyter rendering, packed data, selection logic, and interaction wiring. Every one was a silent failure — no test caught it before manual testing did. This spec mandates **upfront regression tests written before any implementation code**, covering every failure mode observed on that branch plus the new scope.

Tests are grouped into three tiers: Rust unit tests (fast, run in CI), Python integration tests (medium, run via pytest), and manual browser smoke tests (slow, run before merge).

### Tier 1: Rust unit tests (`cargo test`)

Written before any Rust changes. Each test locks an invariant that broke on the abandoned branch.

**R1. `handleDrag` updates interval selection state.** Call `InteractionState::handle_drag` with known coordinates; assert `selections["brush"]` contains the correct `x_range`/`y_range`.

**R2. Interval `contains_point` spatial resolution.** Create an `Interval` selection with known bounds. Assert `contains_point(inside_x, inside_y)` returns true. Assert `contains_point(outside_x, outside_y)` returns false. Test boundary conditions (on edge = inside).

**R3. Interval conditional encoding applies.** Build a scene with circle marks at known positions, an interval selection, and a color conditional. Call `resolve_conditionals`. Assert marks inside the brush have the "if_selected" color; marks outside have the "if_not" color.

**R4. Point selection shift-click semantics.** Call `handle_click` without shift: assert selection is set (replaces previous). Call `handle_click` with shift on a different mark: assert selection is toggled (both marks selected). Call `handle_click` with shift on the same mark again: assert it is deselected.

**R5. `handle_click` with no scene loaded returns empty JSON.** Guard against the null-scene crash.

**R6. `to_json` serialization for interval selections.** Assert the JSON output includes `"type": "interval"`, `"x_range"`, `"y_range"` with correct values.

### Tier 2: Python integration tests (`pytest`)

Written before any Python changes. Each test captures a regression from the abandoned branch or locks a contract that must hold through the refactor.

**P1. HTML export includes packed data.** `chart.interactive().save("out.html")` — read the HTML, assert it contains the base64-encoded packed data string and `loadScene(SCENE_JSON, packedArr)` (two arguments, not one).

**P2. HTML export with empty packed data.** Chart with <1000 marks — assert the HTML still calls `loadScene` with two arguments (second is empty `Uint8Array(0)`).

**P3. Scene JSON round-trip.** `render_interactive` returns `(json_str, packed_bytes)`. Parse `json_str` — assert it has `panels`, `width`, `height`, `selections`, `interaction` keys. Assert `packed_bytes` is `bytes`.

**P4. Theme background preserved in HTML.** Chart with a theme that sets `background` — save to HTML, assert the `<body>` or `<div>` tag contains the background color value from the theme. (Regression: Bug #8 — white flash from missing background.)

**P5. Composition `.interactive()` returns InteractiveChart.** `(chart1 | chart2).interactive()` must not raise `AttributeError`. Assert it returns an `InteractiveChart` instance.

**P6. Composition `.save("out.html")` produces valid HTML.** `(chart1 | chart2).interactive().save("out.html")` — assert file exists, is valid HTML, contains `loadScene`.

**P7. Composition `show_svg()` unchanged.** `(chart1 | chart2).show_svg()` must return the same SVG as before any interactive changes. Compare output before and after the refactor (golden baseline).

**P8. `InteractiveChart` preserves packed data.** `ic = chart.interactive()` — assert `ic._packed_data` is not None and is `bytes`.

**P9. Selection spec serialization round-trip.** Create `selection_point(fields=["group"])` and `selection_interval()`. Add to chart. Call `render_interactive`. Parse scene JSON — assert `selections` array contains both specs with correct types and fields.

**P10. Conditional encoding in scene JSON.** Chart with `.conditional(sel.when(...).otherwise(...))` — assert `interaction.conditionals` in scene JSON contains the conditional with correct `selection_name`, `channel`, `if_selected`, `if_not`.

**P11. `_render_scene` returns tuple.** Assert `_render_scene(chart)` returns `(str, bytes)`, not just `str`. (Regression: Bug #1 — `_render_scene_json` discarded packed data.)

**P12. HTML export JS has no `model.get` / `model.set` calls.** Read the generated HTML — assert it does not contain anywidget model access patterns. (Regression: Bug #3 — Jupyter-only code leaking into standalone.)

**P13. Text elements in scene JSON.** Chart with axis labels — parse scene JSON, call `loadScene`, assert returned text JSON contains elements with `x`, `y`, `fontSize`, `fontWeight`, `fontFamily`, `color`, `content`, `anchor`, and `angle` fields. (Regression: Bug #10 — text rotation missing.)

### Tier 3: Manual browser smoke tests (pre-merge checklist)

A script (`scripts/export-interactive-examples.py`) generates HTML files for every example in `docs/site/guide/interactive.md`. Before merge, open each and verify:

- **S1.** Canvas renders (no "Render error" div).
- **S2.** Tooltip appears on hover with correct field values.
- **S3.** Pan works (click-drag moves the view).
- **S4.** Zoom works (mousewheel zooms in/out, axis ticks update).
- **S5.** Double-click resets zoom.
- **S6.** Point selection: click a mark → conditional encoding updates (color/opacity change).
- **S7.** Brush selection: drag → brush rectangle visible → marks inside highlighted → marks outside dimmed.
- **S8.** Linked views: click in one panel → other panel updates.
- **S9.** Background color matches theme (no white flash on load).
- **S10.** Text labels positioned correctly (y-axis rotated, baselines aligned).
- **S11.** Existing Jupyter notebook `.interactive()` still works (open a notebook, run a cell, verify tooltip + click + zoom).

## 11. Open questions

1. **Scene merge fidelity.** The Python-side scene merge must correctly offset panel `plot_area` and `clip` rects, re-index `panel.id` fields, and merge `packed_data` byte streams (concatenate with updated panel/batch indices in the binary header). The packed data header format (`[panel_idx: u32][batch_idx: u32][kind: u32][count: u32][flags: u32]`) needs panel index rewriting during merge. This is tractable but needs careful implementation.
