# Interactive Wiring Audit — Agent Prompts (2026-05-18)

Four parallel agents were dispatched to drill into every connection point in the
interactive HTML export pipeline. Each agent was scoped to a single integration
seam and instructed to trace actual code paths, not just read signatures.

Found: 5 real bugs (B1-B5), 1 theoretical (B6), 8 high-priority warnings.

---

## Agent 1 — JS ↔ WASM Interaction Wiring

```
You are auditing the wiring between JavaScript and WASM in the ferrum interactive HTML export feature. Your job is to trace every connection point and flag any broken wires, mismatched function signatures, missing arguments, or dead code paths.

## What to check

### 1. ferrum-anywidget.js → WASM API surface
Read `src/ferrum/_wasm/ferrum-anywidget.js` thoroughly. For every WASM method it calls (e.g., `WasmRenderer.create`, `loadScene`, `handleClick`, `handleDrag`, `setTransform`, `onWheel`, `onPan`, `resetZoom`, `getTooltip`), verify the call matches the Rust signature in `crates/ferrum-wasm/src/lib.rs`. Check:
- Argument count and types match
- The `handleClick` signature now takes `shift_held: bool` as a third arg — does JS pass it?
- `handleDrag` takes `(panel_id, x0, y0, x1, y1)` — does JS pass all 5?
- `setTransform` takes `(k, tx, ty)` — does JS call it correctly from D3 zoom?
- Return values: WASM methods return `Result<String, JsValue>` — does JS handle both success and error?

### 2. D3 interactions bundle → anywidget JS
Read `src/ferrum/_wasm/d3-interactions.js`. Check:
- What functions does it export? (brush, zoom, etc.)
- Are those exports consumed by ferrum-anywidget.js?
- Does the D3 zoom callback properly call `setTransform` on the WASM renderer?
- Does the D3 brush callback properly call `handleDrag` on the WASM renderer?
- Are click events properly routed to `handleClick` with the shift key state?

### 3. Standalone adapter (createStandaloneAdapter)
In `ferrum-anywidget.js`, find the `createStandaloneAdapter` function. Check:
- What does it provide vs what the anywidget `render` function provides?
- Does it correctly supply `packed_data` (base64 decoded)?
- Does it correctly supply `interaction_config`?
- Does it implement the model.get/model.set interface that _render expects?
- Is there any model.get/model.set call in the standalone path that would fail?

### 4. _strip_anywidget_for_standalone correctness
Read `src/ferrum/_html.py` function `_strip_anywidget_for_standalone`. Then read the actual `ferrum-anywidget.js` source. Verify:
- The regex patterns in _strip_anywidget_for_standalone actually match what's in the JS source
- After stripping, does `_render` still work? (no missing dependencies on stripped code)
- Is `createStandaloneAdapter` properly un-exported (the `export` keyword removed)?
- Is the anywidget entry point fully removed?
- Are there any other `export` statements that should be stripped but aren't?

### 5. _render function contract
Find `_render` in ferrum-anywidget.js. Trace its full execution:
- What arguments does it take? (container, sceneJson, adapter/model)
- Does it create the canvas, SVG overlay, and tooltip elements?
- Does it call `WasmRenderer.create(canvas)`?
- Does it call `loadScene` or `loadSceneWithPacked` depending on packed data?
- Does it wire up D3 zoom and brush?
- Does it wire up click/mousemove handlers?
- Does it handle the text rendering (SVG labels)?

## Output format

For each connection point, report:
- **GOOD**: wiring is correct, signatures match
- **WARN**: technically works but fragile or has an edge case
- **BUG**: broken wire, mismatched signature, missing argument, dead path

Be specific — cite line numbers and exact function names. If you find a bug, explain exactly what would happen at runtime.
```

---

## Agent 2 — Python → Rust Scene Data Flow

```
You are auditing the data flow from Python through Rust for the ferrum interactive HTML export feature. Trace exactly how scene JSON and packed binary data move from Python API calls to the WASM renderer.

## What to check

### 1. Chart.interactive() → InteractiveChart → _render_scene
Read `src/ferrum/chart.py` (search for the `interactive` method), `src/ferrum/_interactive.py` (InteractiveChart.__init__ and _render_scene).
- When Chart.interactive() is called, what happens step by step?
- _render_scene dispatches to _render_interactive() for compositions and render_interactive for plain Charts — verify both paths produce (str, bytes)
- For plain Charts: verify _render_inputs() → render_interactive() produces correct (scene_json, packed_bytes)
- For empty DataFrames: verify the early return produces valid JSON

### 2. Composition._render_interactive() → _merge_child_scenes
Read each composition type's `_render_interactive` in `src/ferrum/composition.py`:
- HConcatChart, VConcatChart, LayerChart, JointChart, RepeatChart, ClusterMapChart, ConcatChart
- Verify each one calls _merge_child_scenes or _merge_child_scenes_grid correctly
- Check: does LayerChart._render_interactive call _build_merged() then _render_scene(merged)? Is that correct — does the merged chart have all selections/conditionals from both layers?
- Check: does JointChart handle the case where marginals are None?
- Check: does RepeatChart correctly compute n_cols for the grid layout?

### 3. Selection field auto-injection in chart.py
Read the tooltip auto-injection logic in `src/ferrum/chart.py` around the `to_spec` method (search for "sel_fields" or "Auto-inject selection fields").
- Verify: when selection_point(fields=["group"]) is used, "group" gets added to tooltip_fields
- Verify: when tooltip is already set explicitly, the injection merges correctly
- Verify: when selection_interval() is used (no fields), no injection happens
- Edge case: what if the same field appears in both explicit tooltip and selection fields? Is it deduplicated?
- Edge case: what if tooltip is a string shorthand like "x:Q"? Does getattr(kw["tooltip"], "field", "") correctly detect it?

### 4. Scene-merging correctness in composition.py
Read `_merge_child_scenes`, `_merge_one_child`, `_merge_child_scenes_grid`, `_merge_scene_panels`, `_offset_node`.
- Panel ID re-indexing: verify child panels get correctly renumbered (0, 1, 2, ...)
- Coordinate offsets: verify plot_area, clip, marks, axes, grid, annotations all get offset
- Selections: verify they're accumulated from all children
- Conditionals: verify they reference the correct selection names after merging
- tick_levels: verify panel_id in tick_levels gets offset correctly
- _offset_node: verify all node types are handled (circle, rect, line, text, path, group)
- _merge_packed_data: currently always returns b"" — verify this is safe (WASM falls back to JSON)

### 5. InteractiveChart.save() → assemble_html
Read InteractiveChart.save() in _interactive.py and assemble_html in _html.py.
- Verify packed_data flows from InteractiveChart._packed_data to assemble_html
- Verify scene_json flows correctly
- Verify interaction_config is extracted and embedded
- Verify background CSS is extracted and applied
- Verify the standalone adapter receives packed_b64 and interaction_config

## Output format

For each data flow path, report:
- **GOOD**: data flows correctly end-to-end
- **WARN**: works but has an edge case or fragility
- **BUG**: data is lost, malformed, or incorrectly transformed

Be specific — cite file paths, line numbers, and exact variable names.
```

---

## Agent 3 — Rust Selection State Machine

```
You are auditing the Rust selection state machine in ferrum-wasm. Your job is to verify that every state transition is correct, that conditional encoding resolution handles all cases, and that the interaction between click/drag/conditional is coherent.

## What to check

### 1. SelectionState enum and transitions
Read `crates/ferrum-wasm/src/selection_state.rs` thoroughly.
- Verify SelectionState has exactly 3 variants: Empty, Point, Interval
- Verify `contains(data_idx)` semantics for each variant
- Verify `contains_point(x, y)` semantics for each variant, especially boundary conditions (inclusive bounds)
- Check: is Interval with both x_range and y_range as None valid? What does contains_point return?

### 2. handle_click state transitions
Trace handle_click through every code path:
- Hit on a mark with Point selection spec + no shift → should set Point{indices, field_values}
- Hit on a mark with Point selection spec + shift (toggle=ShiftKey) → should call toggle_points
- Hit on a mark with Point selection spec + shift but toggle != ShiftKey → should NOT toggle
- Miss (no hit) → should set Empty for all Point selections
- Interval selection spec → should be a no-op on click
- Verify field_values extraction: when spec has fields=Some(["group"]), does it extract from tooltip?
- Verify collect_matching_indices: does it scan ALL panels (cross-panel linked selection)?

### 3. handle_drag state transitions
Trace handle_drag:
- Verify it only updates Interval selections (ignores Point specs)
- Verify coordinate normalization: x0 > x1 should still produce lo < hi
- Verify panel_id is accepted but currently unused (let _ = panel_id)
- What happens if handle_drag is called with panel_id that doesn't exist?

### 4. toggle_points correctness
Read toggle_points carefully:
- All indices already selected → deselect all, transition to Empty if none remain
- Some indices already selected → add the missing ones
- None selected (Empty state) → create new Point selection
- Verify field_values handling: cleared on deselect, replaced on add
- Edge case: empty indices slice passed to toggle_points — what happens?

### 5. Conditional encoding resolution
Read `crates/ferrum-wasm/src/conditional.rs` thoroughly.
- resolve_conditionals: verify it iterates panels → batches, tracks circle/rect offsets correctly
- Empty selection → conditional is skipped (marks retain original colors) — verify
- Point selection with field_values → uses field_value_matches_tooltip for matching — verify the matching logic
- Point selection without field_values → falls through to data_idx contains check — verify
- Interval selection → uses contains_point on mark center — verify circle center and rect center calculation
- Verify apply_value_to_circle handles Color, Opacity, and Size channels
- Verify apply_value_to_rect handles Color, Opacity, and Size channels
- Check: what happens if a conditional references a selection_name that doesn't exist in the selections map?
- Check: field_value_matches_tooltip — does the epsilon comparison work for all FieldValue variants?

### 6. Interaction state JSON serialization
Read InteractionState::to_json():
- Verify Empty serializes to {"type": "empty"}
- Verify Point serializes with indices AND field_values
- Verify Interval serializes with x_range and y_range (including None case)
- Does the serialization produce valid JSON for all possible states?

## Output format

For each check, report:
- **GOOD**: logic is correct
- **WARN**: edge case that could surprise but doesn't crash
- **BUG**: incorrect state transition, missing case, or logic error

Be specific — cite line numbers and trace the exact code path.
```

---

## Agent 4 — HTML Assembly End-to-End

```
You are auditing the HTML assembly pipeline for ferrum's interactive export feature. Your job is to verify that the generated HTML file is correct, self-contained, and will render properly in a browser.

## What to check

### 1. assemble_html structure
Read `src/ferrum/_html.py` function `assemble_html` completely. Verify:
- The HTML document structure: DOCTYPE, html, head, body
- CSS is inlined from ferrum-interactive.css
- JS glue (ferrum_wasm.js) is inlined
- D3 interactions bundle is inlined (via _convert_d3_exports)
- ferrum-anywidget.js is inlined (via _strip_anywidget_for_standalone)
- SCENE_JSON is embedded as a template literal
- The main() function: WASM init → create container → createStandaloneAdapter → _render
- Error handling: main().catch displays error in the container

### 2. WASM initialization correctness
Check the WASM init block in assemble_html:
- embed_wasm=True path: base64 decode → Uint8Array → __wbg_init({module_or_path: wasmBytes})
- embed_wasm=False path: __wbg_init() with no args (loads from adjacent file)
- Verify the base64 encoding/decoding is correct (atob → charCodeAt loop)
- Check: does `{{ module_or_path: wasmBytes }}` in the f-string correctly produce `{ module_or_path: wasmBytes }` in the output? (Python f-string double-brace escaping)

### 3. Scene JSON escaping
Check the escaping logic for embedding scene_json in a JS template literal:
- `\\` → `\\\\` (backslash escaping)
- `</` → `<\\/` (prevent closing script tag)
- `` ` `` → `` \` `` (template literal delimiter)
- `${` → `\\${` (prevent template literal interpolation)
- Are there any other characters that could break the template literal? (e.g., null bytes, other special sequences)

### 4. Interaction config escaping
Check the interaction_config embedding:
- It's embedded in a JS single-quoted string: `'...'`
- Escaping: `\\` → `\\\\`, `'` → `\\'`
- Are there any characters in the interaction config JSON that could break a single-quoted JS string? (newlines, carriage returns, etc.)
- What if the interaction config contains a field value with a single quote in it?

### 5. Background CSS extraction
Check `_extract_background_css` and `_background_css_from_dict`:
- Verify it produces valid CSS rgba() syntax
- What if background is null/missing in scene JSON? (should default to #ffffff)
- What if background has unexpected keys?
- Is the alpha calculation correct? (bg['a'] / 255.0)

### 6. _strip_anywidget_for_standalone robustness
Read the actual content of `src/ferrum/_wasm/ferrum-anywidget.js` and verify each regex in `_strip_anywidget_for_standalone` matches correctly:
- Regex 1: Remove _B64 bootstrap block — does the pattern match the actual code?
- Regex 2: Remove _ensureWasm — does the pattern match?
- Regex 3: Strip export from createStandaloneAdapter — exact string match?
- Regex 4: Remove re-export line — does the pattern match?
- Regex 5: Remove anywidget entry point — does the comment marker exist?
- What happens if the JS source changes and the regex no longer matches? (silent failure = broken HTML)

### 7. _convert_d3_exports robustness
Read the D3 bundle (`src/ferrum/_wasm/d3-interactions.js`) and verify:
- Does it end with an `export{...}` block?
- Does the regex correctly convert `export{ri as brush, zi as zoom}` to `var brush=ri,zoom=zi;`?
- What if there are named exports without `as` (just plain identifiers)?
- What if the minified bundle has no spaces around `as`?

### 8. Font embedding
Check that the Inter font is properly embedded:
- Is `@font-face` present in `ferrum-interactive.css`?
- Is the font data inlined (base64 data URL) or referenced externally?
- If external, will it work in the self-contained HTML file?

## Output format

For each check, report:
- **GOOD**: correct and robust
- **WARN**: works but fragile (e.g., regex depends on exact formatting)
- **BUG**: will break in a browser, produces invalid HTML/JS, or loses data

Be specific — cite exact strings, line numbers, and what the output looks like.
```
