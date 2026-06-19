# Archaeology #6/#7/#8 Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development. Rust tasks → rust-coder, Python → python-coder. Same three-gate (spec → quality → review-lite) per task as the parent effort.

## 1. Objective

Remediate every heavyweight-pass finding over the #6/#7/#8 fix surface (confirmed label bug, metadata-drops, guard gap, 2-D facet extent, violin dead-field, bin triplication, LayerChart title, cohesion cleanups), spec-first.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-19-archaeology-678-remediation-design.md` — full design
- §5 Architecture, §6 contracts (extended guard, 2-D extent, violin shared_extent, offset key-set), §7 invariants, §9 acceptance, §3 deferred-with-rationale
- Heavyweight findings: `.git/sdd/heavyweight-{rust-6,rust-7,python-8}.md`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/mark_nodes.rs` | guard: add data_indices + keys length checks |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | pass data_indices/keys lengths to the seam guard |
| Modify | `crates/ferrum-core/src/render/marks/label.rs` | migrate to MarkNodes; push_many leader-line; build_metadata_for_indices |
| Modify | `crates/ferrum-core/src/render/marks/geoshape.rs` | wire build_metadata_for_indices (drop hardcoded None) |
| Modify | `crates/ferrum-core/src/render/marks/image.rs` | wire build_metadata_for_indices (drop hardcoded None) |
| Modify | `crates/ferrum-core/src/transform/bin.rs` | extract shared extent/nice helpers; dedup 3 call sites |
| Modify | `crates/ferrum-core/src/transform/kde_2d.rs` | add 2-D global_extent helper |
| Modify | `crates/ferrum-core/src/transform/bin_2d.rs` | add 2-D global_extent helper (niced per axis) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | dispatch facet pin over Kde2D/Bin2D |
| Modify | `crates/ferrum-core/src/transform/violin.rs` | wire shared_extent (mirror kde apply_grouped) |
| Modify | `src/ferrum/` (violin spec emission, if user-facing shared_extent) | parity, if exposed |
| Modify | `src/ferrum/composition.py` | LayerChart HTML title; shared offset key-set constant |
| Modify | `src/ferrum/_overrides.py` | narrow try/except to the rebuild call |
| Test | inline `#[cfg(test)]` + `tests/` | per-finding regression tests (see tasks) |
| Modify | `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` | record remediation outcomes |

## 4. Constraints

- Node-order metadata is the single convention; one `data_indices` entry per emitted node; NO consumer remaps tooltips via `data_indices`.
- The seam guard must assert `nodes.len()` against tooltips, hrefs, descriptions, **data_indices, and keys** (each when present).
- Extent computation lives in the transform layer; `prepare.rs` orchestrates only. Never clobber a user-provided extent (pin only when unset). Bin nices per axis; Kde/Violin/Kde2D raw.
- Bin extent/nice logic must have exactly ONE source after RB1.
- LayerChart fix must NOT reintroduce inner-layer title leakage; LayerChart stays a single-plot overlay (not reparented onto `_CompositeBase`).
- No WASM source change. No packed-format change. No matplotlib. No global mutable state.
- Backward compat: all currently-correct charts (incl. the just-landed #6/#7/#8 fixes) render byte-identically; additions only add missing metadata/extents.
- Deferred (do NOT attempt here; record as limitations): W5 caption-y body layout, Text/Label WASM hit-test, keys WASM consumer.
- Build: `unset CONDA_PREFIX && uv run --no-sync maturin develop`. Rust tests: cargo test; on this machine the test binary links miniforge py3.13 → run with `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib`. Verify by ACTUALLY RUNNING tests (a dyld load-abort masked a real failure last time).

## 5. Tasks

### Task R1: Extend the alignment guard (data_indices + keys)
- [ ] Add `data_indices` and `keys` length params to `debug_assert_nodes_metadata_aligned` (and the checked sibling); assert each present one == nodes_len. Update scene_build.rs seam to pass `result.data_indices`/`keys` lengths.
- [ ] Tests: guard trips on a data_indices-misaligned and a keys-misaligned batch (`#[should_panic]`); passes when aligned.
- [ ] Verify: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core --lib mark_nodes`

### Task R2: Fix label.rs (leader-line multi-node + metadata-drop)
- Consumes: extended guard (R1) — should now catch the pre-fix divergence
- [ ] Migrate `label.rs build` to `MarkNodes`; emit leader-line as `push_many([text, line], row)`; replace hardcoded `None` metadata with `build_metadata_for_indices(&data_indices)`.
- [ ] Tests: leader-line batch `nodes.len() == data_indices.len()` (fail-before: 2N vs N); per-row tooltip/href reaches the text node; conditional/selection on a leader-line label matches correct rows.
- [ ] Verify: `... cargo test -p ferrum-core --lib label`

### Task R3: Wire geoshape.rs + image.rs metadata
- Consumes: guard (R1)
- [ ] Replace hardcoded `tooltips/hrefs/descriptions: None` with `build_metadata_for_indices(&data_indices)` (indices already tracked). geoshape is multi-node-per-row (rings) — confirm data_indices already lockstep; image single-node.
- [ ] Tests: per-row metadata reaches nodes for both marks (SVG `<title>` present; correct value per node).
- [ ] Verify: `... cargo test -p ferrum-core --lib "geoshape|image"`

### Task R4: Dedup bin extent/nice logic
- [ ] Extract `bin_float64_extent(batch, field) -> Option<(f64,f64)>` (cast→clean→fold) and `nice_extent(lo,hi,target) -> (f64,f64)`; refactor `apply_one_group`, `apply_grouped` shared_extent block, and `global_extent` to delegate. One source each.
- [ ] Tests: existing bin tests + `global_extent_nices_for_bin_but_raw_for_kde_and_violin` pass unchanged (behavior-preserving refactor).
- [ ] Verify: `... cargo test -p ferrum-core --lib bin`

### Task R5: 2-D faceted extent pin (Kde2D/Bin2D)
- Consumes: prepare.rs dispatch pattern (parent Task 9); bin nice helper (R4) if reusable for Bin2D
- [ ] Confirm the Kde2D/Bin2D extent field shape (4-tuple vs extent_x/extent_y) — spec §11. Add `global_extent` to each (Bin2D niced per axis, Kde2D raw per axis) over the full pre-facet batch.
- [ ] Extend `fix_transform_extents_for_facet` to dispatch over Kde2D/Bin2D, pinning only when unset.
- [ ] Tests: faceted Kde2D + Bin2D share x AND y extents across panels (disjoint per-panel ranges → fail-if-unpinned); 1-D unchanged.
- [ ] Verify: `... cargo test -p ferrum-core --lib "fix_extents|global_extent|kde_2d|bin_2d"`

### Task R6: Wire ViolinSpec.shared_extent
- [ ] When `shared_extent && groups > 1`, compute cross-group global extent and pin each group's internal KDE to it (mirror `kde::apply_grouped`); when false, today's per-group behavior. Update Python violin spec emission if `shared_extent` should be user-settable (match `mark_density`/`mark_histogram` parity; else rely on serde default).
- [ ] Tests: non-faceted multi-group violin shared (one extent) vs per-group (regression pins both); existing violin tests green.
- [ ] Verify: `... cargo test -p ferrum-core --lib violin` + `uv run pytest -n auto -k violin`

### Task R7: LayerChart HTML title
- [ ] Make `LayerChart.properties(title=)` (and ctor title) resolve the document `<title>` correctly (spec §11 — give LayerChart a figure title accessor that does not require reparenting and does not leak into layers).
- [ ] Tests: `LayerChart(...).properties(title="T")` → HTML `<title>` is `T`; no stray layer title; ctor path still correct.
- [ ] Verify: `uv run pytest -n auto -k "layer and (title or html)"`

### Task R8: Composition cohesion cleanups
- [ ] Consolidate the panel-node offset key-set into one shared definition used by `_inject_figure_chrome`, `_merge_scene_panels`, `_merge_one_child`. Narrow `_overrides._apply_overrides` try/except to wrap only the rebuild call. Make the figure-chrome payload a `TypedDict`.
- [ ] Tests: existing composite/figure-title tests green (no behavior change); a test asserting all three offset paths use the shared key-set.
- [ ] Verify: `uv run pytest -n auto tests/test_composite_figure_title.py tests/test_composite_figure_title_goldens.py`

### Task R9: Close-out
- Consumes: R1–R8
- [ ] Full suite in a consistent env (run + confirm exit codes): `cargo test -p ferrum-core` and `uv run pytest -n auto`.
- [ ] Heavyweight re-review of the remediated surfaces (rust marks+transform, python composition); `/regression-test` per fix.
- [ ] Update archaeology doc with remediation outcomes; record §3 deferred items as named limitations.
- [ ] `chris-code:finishing-a-development-branch`.

## 6. Acceptance checks

- `cargo test -p ferrum-core` exit 0, all binaries; `uv run pytest -n auto` 0 failed — both run directly, exit codes confirmed.
- Guard trips on data_indices/keys misalignment; label leader-line `nodes==data_indices`; geoshape/label/image metadata reaches nodes; faceted Kde2D/Bin2D share x+y; multi-group violin shared_extent works; bin logic single-source; LayerChart HTML title correct.
- Per-fix regression tests fail-before/pass-after; goldens (if any) visually inspected.
- §3 deferred items recorded as limitations, not claimed fixed.

## 7. Open questions

- Kde2D/Bin2D extent field shape (4-tuple vs per-axis) — confirm before R5 (does not change the §6 contract).
- LayerChart title storage mechanism — pick the non-leaking option (R7).
