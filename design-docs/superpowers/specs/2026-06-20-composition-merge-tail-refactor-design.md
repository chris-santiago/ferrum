# Composition Merge-Tail Refactor Design Spec

> Source: `/python-review` of the archaeology #6/#7/#8 effort's Python surface (2026-06-20). Findings F1 (S3), F2 (S2), F3 (S1). Behavior-preserving cohesion refactor of `src/ferrum/composition.py`.

## 1. Scope

Collapse the duplicated *placement → merge* tail shared by the four multi-child scene-merge functions in `src/ferrum/composition.py` into one record type plus one helper, so the packed-instance offset can no longer drift from the scene-node offset. Plus two trivial cleanups: dedup the `share_scale` mode validation (F2) and hoist the function-local `import struct` to module scope (F3). No public API change, no behavioral change — output stays byte-identical.

## 2. Goals

- The per-child `(panel_id_offset, dx, dy, scene, packed)` placement is carried by **one record**, not three hand-maintained parallel lists (`child_offsets`, `child_xy`, and the `_merge_one_child` call) duplicated across four functions.
- Exactly one site runs the `_merge_one_child` loop + `_inject_figure_chrome` + `_merge_packed_data` tail.
- Each `_merge_child_scenes*` variant retains only its genuinely-distinct placement geometry.
- `share_scale`'s `"shared"|"independent"` vocabulary is validated in one place.
- `import struct` is module-level.
- Every existing test and golden stays green by **byte-identical** output.

## 3. Non-goals

- No change to `_render_single_with_figure_chrome` (single-child chrome path; it mutates its own scene, has no lockstep lists, and is not part of the drift surface).
- No change to `_merge_one_child`, `_merge_scene_panels`, `_offset_node`, `_merge_packed_data`, or `_offset_packed_batch_xy` internals (the packed byte-format logic is correct and stays as-is).
- No fix for `_offset_node` `raw`-node gap (W4, documented + inert) or the facet wrap/grid `ncols` asymmetry (KG-6, filed as #24).
- No public API, no `ferrum-spec.md` change.

## 4. System behavior

Unchanged and observable-equivalent. For every composite (`HConcat`/`VConcat`/`Concat` wrapping grid, `Repeat` corner/sparse grid, `Joint`/`ClusterMap` nonuniform grid), `.to_svg()`, `.to_html()`, and the interactive packed bytes are byte-for-byte identical to pre-refactor output, titled and untitled. Empty-children and single-child cases are unchanged (their early-return guards are untouched).

## 5. Architecture

Two responsibilities, currently entangled inside each variant, become explicit:

- **Placement** (stays per-variant — this is the part that genuinely differs): render each child, compute its `(dx, dy)` and `panel_id_offset`, accumulate the merged canvas `width`/`height`. `panel_id_offset` advances by `len(scene["panels"])` (the value `_merge_one_child` returns today).
- **Assembly** (shared, new single site): given the ordered placements + final canvas size + optional figure chrome, run the `_merge_one_child` loop into a fresh `_empty_scene()`, inject chrome, merge packed bytes with `y_offset = header_h`, and return `(scene_json, packed)`.

The lockstep that desynced in round-5 P2 (scene offset vs. packed offset) becomes structural: both read `dx`/`dy` from the same record field, so they cannot disagree.

## 6. Canonical interfaces / data contracts

```python
@dataclass
class _PlacedChild:
    scene: dict          # parsed child scene JSON
    packed: bytes        # child packed GPU-instance bytes
    dx: float            # lateral placement offset
    dy: float            # vertical placement offset
    panel_id_offset: int # cumulative panel-id base for this child

def _assemble_placed_children(
    placed: list[_PlacedChild],
    width: float,
    height: float,
    figure_chrome: Optional["_FigureChrome"],
) -> tuple[str, bytes]:
    """Merge placed children into one scene + packed buffer.

    Builds a fresh _empty_scene(), sets width/height, runs _merge_one_child
    for each placement, injects figure chrome (header_h), then merges packed
    bytes with y_offset=header_h. Returns (scene_json, packed).
    """
```

Contract invariants the helper must honor (these reproduce today's exact order):
1. `merged = _empty_scene()`; set `merged["width"] = width`, `merged["height"] = height` **before** the merge loop.
2. Iterate `placed` in order, calling `_merge_one_child(merged, pc.scene, pc.dx, pc.dy, pc.panel_id_offset)`.
3. `header_h = _inject_figure_chrome(merged, **figure_chrome)` only when `figure_chrome is not None`, **after** the loop (chrome shifts panels + grows height exactly as today).
4. `_merge_packed_data([pc.packed ...], [pc.panel_id_offset ...], [(pc.dx, pc.dy) ...], y_offset=header_h)`.
5. Return `(_json.dumps(merged), merged_packed)`.

## 7. Invariants and constraints

- **Byte-identical output.** The only legal evidence of success is that the full golden/byte-diff suite stays green. Any diff is a regression.
- **Empty-children guards untouched.** Each variant keeps its existing `if not …: return '{"panels":[],"width":0,"height":0}', b""` early return; `_assemble_placed_children` is only ever called on the non-empty path (so it need not reproduce that literal).
- **Placement order preserved.** `_PlacedChild` records must be appended in the same order children are merged today (grid: row-major; sparse/nonuniform: rendered order), because `_merge_packed_data` consumes the lists positionally and panel-id assignment is order-dependent.
- **`panel_id_offset` semantics unchanged:** cumulative, advancing by `len(scene["panels"])` per child.
- No matplotlib; no global mutable state.

## 8. Key decisions and tradeoffs

- **Record over parallel lists.** A `dataclass` (not `TypedDict`) because these are constructed positionally in tight loops and benefit from a concrete constructor; `_FigureChrome` stays a `TypedDict` (it models an external kwargs dict, a different role).
- **Placement stays per-variant.** The four variants differ precisely in geometry (linear run vs. wrapping grid vs. sparse vs. per-row/col sizing). Unifying geometry would be speculative; only the *identical* tail is unified.
- **`_render_single_with_figure_chrome` excluded.** It is single-child scene mutation, not multi-child merge; routing it through the helper would distort the helper's contract for no dedup gain.
- **F2/F3 are independent micro-cleanups** bundled because they touch the same module; each is self-contained and byte-neutral.

## 9. Acceptance criteria

- Full pytest suite green, including the golden byte-diffs: `tests/test_bug_hunt_scene_composition.py`, `tests/test_bug_hunt_composition_facet.py`, `tests/test_composite_packed_figure_title.py`, `tests/test_html_export_regression.py`, and all `concat`/`facet`/`joint`/`repeat`/`cluster` golden tests.
- `_merge_packed_data` is called from exactly one site (`_assemble_placed_children`) for the multi-child path; the four variants no longer reference `child_offsets`/`child_xy`/`_merge_packed_data` directly.
- `ruff check` clean; `import struct` at module scope; no function-local `import struct` remains.
- `share_scale` mode validation lives in one helper, called by both `_ChartLike.share_scale` and `RepeatChart.share_scale`.

## 10. Validation strategy

Behavioral equivalence by byte-diff: the existing golden + packed-merge regression suite is the oracle. Edge cases to confirm explicitly: empty child list (early-return guard), single child, titled vs. untitled composite (`header_h` 0 vs. >0), and a child carrying packed data at a nonzero grid offset (the P2 regression case, covered by `test_composite_packed_figure_title.py`). Because the refactor is correctness-neutral, no new behavioral tests are required; a focused unit asserting `_assemble_placed_children` reproduces a hand-built merged scene MAY be added if it clarifies the contract, but the byte-diff suite is the binding gate.
