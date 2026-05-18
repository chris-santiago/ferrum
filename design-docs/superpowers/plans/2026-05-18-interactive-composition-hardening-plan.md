# Interactive Composition Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Fix the 6 bugs and 3 user-facing warnings from the 2026-05-18 wiring audit by hardening the composition interactive merge pipeline as a cohesive unit and resolving two standalone issues (JS transition error, empty-data scene JSON).

## 2. Spec references

- `.claude/output/audit-interactive/2026-05-18-audit.md` — full audit report with file:line citations
- `design-docs/superpowers/audits/2026-05-18-interactive-wiring-audit-prompts.md` — audit methodology
- `CLAUDE.md` §"Known interactive-export limitations" — W2, W4, W5 deferred items

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/composition.py` | B3 B4 B5: corner-mode grid, strip_title offset, packed data rewrite |
| Modify | `src/ferrum/_interactive.py` | B2: complete empty-data scene JSON |
| Modify | `src/ferrum/display.py` | B2: same empty-data fix |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | B1: tickTransition error; W2: per-panel brush |
| Modify | `crates/ferrum-wasm/src/selection_state.rs` | B6: panel_id→array-index lookup |
| Modify | `crates/ferrum-wasm/src/hit_test.rs` | B6: return array position, not panel.id |
| Modify | `src/ferrum/_html.py` | W20: optional CSP nonce support |
| Modify | `CLAUDE.md` | Remove resolved W2/W4/W5 entries |
| Test | `tests/test_html_export_regression.py` | Regression tests for B1-B6, W2, W20 |

## 4. Constraints

- `_merge_packed_data` currently always returns `b""` because packed binary headers contain `(panel_idx, batch_idx)` that must be rewritten during composition merge. The rewrite is straightforward: iterate headers, add `panel_id_offset` to each `panel_idx` u32.
- `hit_test.rs` returns `panel.id` (logical) at line 32 but `tooltip_for_hit` uses it as an array index at line 217. These must use the same semantics — array position is correct since zoom transforms are indexed by position.
- W2 (multi-panel brush) requires per-panel brush overlays in JS and routing the correct `panel_id` to `handleDrag`. The brush extent must use each panel's `plot_area`, not just `panels[0]`.
- B3 (RepeatChart corner) requires sparse grid placement — cells must be positioned at their `(row, col)` coordinates with gaps for the upper triangle, not flat-packed by `_merge_child_scenes_grid`.
- W20 (CSP) is additive — add an optional `csp_nonce` parameter to `assemble_html`. When provided, inline `<script>` and `<style>` tags get `nonce="..."` attributes. Default behavior unchanged.
- All changes must pass: `uv run pytest -x -q`, `cargo test -p ferrum-wasm`, `cargo clippy -p ferrum-wasm -- -D warnings`.

## 5. Tasks

### Task 1: Standalone fixes (B1, B2) — JS + Python
- [ ] B1: In `ferrum-anywidget.js:436`, replace `_state.renderer.tickTransition(t).catch(() => {})` with `try { _state.renderer.tickTransition(t); } catch (_) {}`
- [ ] B2: In `_interactive.py` `_render_scene` and `display.py` `_render_scene_json`, replace the minimal empty-data JSON with a complete SceneGraph: add `background: null`, `title: []`, `legend: []`, `decorations: []`, `selections: []`, `interaction: {zoom_enabled: true, pan_enabled: true, conditionals: [], linked_panels: [], tick_levels: []}`
- [ ] Verify: `uv run pytest tests/test_html_export_regression.py -x -q`

### Task 2: Composition merge completeness (B4, strip_title + merge gaps)
- [ ] B4: In `_merge_scene_panels`, add `strip_title` to the offset loop (alongside `axes`, `grid`, `annotations`)
- [ ] Merge `interaction.linked_panels` from children in `_merge_one_child` (currently dropped)
- [ ] Merge per-child `zoom_enabled`/`pan_enabled` in `_merge_one_child` (use AND — disabled if any child disables)
- [ ] Verify: add test for strip_title offsetting in a faceted composition

### Task 3: Packed data rewriting (B5)
- [ ] In `_merge_packed_data`, implement binary header rewriting: iterate 20-byte headers, add `panel_id_offset` to the `panel_idx` u32 (bytes 0-3, little-endian) for each child's packed data. Concatenate rewritten chunks.
- [ ] Track cumulative `panel_id_offset` per child (same counter used by `_merge_scene_panels`)
- [ ] Pass `panel_id_offsets: list[int]` from the merge loop to `_merge_packed_data`
- [ ] Verify: compose two 1500-point scatter plots with `|`, call `.interactive()`, confirm packed data is non-empty and scene renders both panels

### Task 4: Panel ID consistency (B6)
- [ ] In `hit_test.rs`, change `panel_id: panel.id` to `panel_id: panel_pos` at lines 32 and 70. This makes hit_test return the array index, which is what `tooltip_for_hit`, zoom transforms, and `get_tooltip` all expect.
- [ ] Update any test assertions that check `panel_id` values (e.g., line 732)
- [ ] Verify: `cargo test -p ferrum-wasm`

### Task 5: RepeatChart corner-mode grid (B3)
- [ ] Add `_merge_child_scenes_sparse_grid(charts, spacing, cells)` that accepts `(row, col, chart)` triples and positions each cell at `(col * (cell_w + spacing), row * (cell_h + spacing))`, leaving upper-triangle cells empty
- [ ] In `RepeatChart._render_interactive`, when `corner=True`, pass `expand()` triples to the sparse grid merge instead of `_merge_child_scenes_grid`
- [ ] Verify: `RepeatChart(df, row=["a","b","c"], column=["a","b","c"], corner=True).interactive()` produces the lower-triangle layout matching SVG output

### Task 6: Multi-panel brush (W2)
- [ ] In `ferrum-anywidget.js`, create one D3 brush per panel (iterate `scene.panels`), each with its own extent from `panel.plot_area`
- [ ] Each brush's `end` callback calls `renderer.handleDrag(panel_idx, x0, y0, x1, y1)` with the correct panel index
- [ ] In `selection_state.rs` `handle_drag`, replace `let _ = panel_id;` with actual panel-scoped interval storage (store `panel_id` in a new `Interval` field, or scope the interval name by panel)
- [ ] Verify: compose two charts with `|`, add `selection_interval`, brush on panel 1 → marks in panel 1 highlight
- [ ] Update CLAUDE.md: remove W2 from known limitations

### Task 7: CSP nonce support (W20)
- [ ] Add `csp_nonce: str | None = None` parameter to `assemble_html`
- [ ] When provided, add `nonce="{csp_nonce}"` to the `<style>` and `<script type="module">` tags
- [ ] Thread nonce through `InteractiveChart.save(csp_nonce=...)` and `_ChartLike.save(csp_nonce=...)`
- [ ] Verify: generated HTML with nonce has `nonce="..."` on both tags; without nonce, output is unchanged

### Task 8: Regression tests + CLAUDE.md cleanup
- [ ] Write regression tests for B1-B6 and W2 in `tests/test_html_export_regression.py`
- [ ] Update CLAUDE.md: remove resolved W2/W5 from known limitations, update W4 if still open
- [ ] Run `/audit-interactive` to verify clean audit
- [ ] Verify: `uv run pytest -x -q` and `cargo test -p ferrum-wasm` both green

## 6. Acceptance checks

- `uv run pytest -x -q` — all pass (including new regression tests)
- `cargo test -p ferrum-wasm` — all pass
- `cargo clippy -p ferrum-wasm -- -D warnings` — clean
- `nox -s lint` via `uv run nox -s lint` — clean
- Composed 1500-point scatter plots render in interactive HTML (B5 resolved)
- RepeatChart corner mode interactive matches SVG layout (B3 resolved)
- Multi-panel brush works on non-first panels (W2 resolved)
- Empty-data charts produce valid interactive output (B2 resolved)
- `/audit-interactive` produces 0 BUGs

## 7. Open questions

- B6 alternative: instead of changing hit_test to return array position, we could build a `panel_id → array_index` lookup map in the WASM renderer. Array position is simpler but changes the semantics of `HitResult.panel_id`. Which approach? (Recommendation: array position — it's what every consumer expects.)
- W2 scope: should per-panel brush support `SelectionResolve::Union` / `Intersect` modes, or just `Global` for now? (Recommendation: Global only — Union/Intersect require design work.)
