# Composition Merge-Tail Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development to implement this plan task-by-task.

## 1. Objective

Collapse the duplicated placement→merge tail in `src/ferrum/composition.py` into one `_PlacedChild` record + `_assemble_placed_children` helper (F1), dedup `share_scale` mode validation (F2), and hoist `import struct` to module scope (F3) — all behavior-preserving (byte-identical output).

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-20-composition-merge-tail-refactor-design.md` §6 (interfaces), §7 (invariants), §9 (acceptance)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/composition.py` | F1 record+helper+4 variant rewrites; F2 validator dedup; F3 module-level `import struct` |
| Test | `tests/test_composition_merge_tail.py` | Optional focused unit asserting `_assemble_placed_children` reproduces a hand-built merged scene (only if it clarifies the contract; the byte-diff suite is the binding gate) |

## 4. Constraints

- **Byte-identical output is the only success criterion.** Every existing golden and packed-merge regression test must stay green by byte-diff; any diff is a regression, not an improvement.
- **No public API change, no `ferrum-spec.md` change, no behavior change.** All edited symbols are `_`-private module internals.
- **Do not touch** `_render_single_with_figure_chrome`, `_merge_one_child`, `_merge_scene_panels`, `_offset_node`, `_merge_packed_data`, or `_offset_packed_batch_xy` internals — only their *callers* change.
- **Preserve each variant's empty-children early-return guard verbatim** (`if not …: return '{"panels":[],"width":0,"height":0}', b""`). `_assemble_placed_children` is only called on the non-empty path and need not reproduce that literal.
- **Preserve placement order and `panel_id_offset` semantics:** records appended in current merge order (grid row-major; sparse/nonuniform rendered order); `panel_id_offset` cumulative, advancing by `len(scene["panels"])` per child.
- **Reproduce the assembly order exactly** (spec §6 invariants 1–5): `_empty_scene()` → set width/height → `_merge_one_child` loop → `_inject_figure_chrome` (only if `figure_chrome is not None`, after the loop) → `_merge_packed_data(..., y_offset=header_h)` → `(_json.dumps(merged), merged_packed)`.
- `_PlacedChild` is a `@dataclass` (positional construction in loops); `_FigureChrome` stays a `TypedDict`.
- No matplotlib; no global mutable state.
- Build before testing: `unset CONDA_PREFIX && uv run --no-sync maturin develop` (pure-Python edit needs no rebuild, but the venv must have the extension; it already does).
- Pytest gets **NO** `DYLD_LIBRARY_PATH` prefix (it breaks venv pyarrow). Run `uv run --no-sync pytest … -p no:randomly -q`.

## 5. Tasks

### Task 1: Collapse the merge tail + micro-cleanups (F1 + F2 + F3)
- [ ] Add `from dataclasses import dataclass` and module-level `import struct`; remove the two function-local `import struct` (in `_offset_packed_batch_xy`, `_merge_packed_data`) (F3).
- [ ] Define `_PlacedChild` dataclass and `_assemble_placed_children(placed, width, height, figure_chrome)` per spec §6, honoring invariants 1–5 verbatim.
- [ ] Rewrite `_merge_child_scenes`, `_merge_child_scenes_grid`, `_merge_child_scenes_sparse_grid`, `_merge_child_scenes_nonuniform_grid`: keep each one's placement geometry + empty guard; replace the merge/packed tail with building a `list[_PlacedChild]` and returning `_assemble_placed_children(placed, width, height, figure_chrome)`. In each placement loop, advance `panel_id_offset` by `len(scene.get("panels", []))` (no longer via `_merge_one_child`'s return).
- [ ] Extract a single `share_scale` mode validator (e.g. module-level `_validate_share_modes(channels)` or reuse `_validate_resolve`) and call it from `_ChartLike.share_scale` and `RepeatChart.share_scale`, preserving the exact `ValueError` message `f"share_scale: {ch}={mode!r}; expected 'shared' or 'independent'"` (F2).
- [ ] (Optional) Add `tests/test_composition_merge_tail.py` only if a focused contract unit clarifies `_assemble_placed_children`; skip if the byte-diff suite already covers it.
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest tests/test_bug_hunt_scene_composition.py tests/test_bug_hunt_composition_facet.py tests/test_composite_packed_figure_title.py tests/test_html_export_regression.py -p no:randomly -q`
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest tests/ -k "concat or facet or joint or repeat or cluster" -p no:randomly -q`
- [ ] Verify: `ruff check src/ferrum/composition.py`

## 6. Acceptance checks

- Full suite green (orchestrator runs `uv run --no-sync pytest -p no:randomly -q` — golden byte-diffs included).
- `grep -n "_merge_packed_data\|child_xy\|child_offsets" src/ferrum/composition.py` shows the four variants no longer reference these directly; `_merge_packed_data` is called only from `_assemble_placed_children` (multi-child path) and `_render_single_with_figure_chrome` (single-child, unchanged).
- `grep -n "import struct" src/ferrum/composition.py` shows exactly one module-level occurrence.
- `share_scale` validation logic appears once.
- `ruff check src/ferrum/composition.py` clean.

## 7. Open questions

- None. (The byte-diff suite is the binding oracle; if any golden drifts, the assembly order in §6 invariants was not reproduced exactly — fix the order, do not re-bless the golden.)
