# Archaeology Bugs #6 / #7 / #8 — Class-Level Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task. Dispatch Rust tasks to `rust-coder`, Python tasks to `python-coder` (CLAUDE.md coding-agent dispatch rule).

## 1. Objective

Fix code-archaeology bugs #6 (metadata/node misalignment, incl. N1 packed face), #7 (faceted shared-extent pin), and #8 (composite figure-title placement) as structural defect-class fixes covering their full enumerated surface.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-19-archaeology-bugs-6-7-8-class-fix-design.md` — full design
- §5 Architecture (accumulator / transform-layer pin / single chrome home)
- §6 Canonical interfaces (alignment contract, extent contract, chrome contract)
- §7 Invariants (node-order convention, assertion, one chrome base)
- §8 Key decisions (N1 folds in; N2 dropped; structural; full interactive parity; multi-group)
- §9 Acceptance criteria, §10 Validation, §11 Open question (interactive title spike)
- CLAUDE.md: build/test commands, golden visual-inspection rule, regression-test rule

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `crates/ferrum-core/src/render/mark_nodes.rs` | `MarkNodes` node+index accumulator + finalize helper + guard |
| Modify | `crates/ferrum-core/src/render/draw.rs` | wire accumulator; `build_metadata_for_indices` as canonical entry |
| Modify | `crates/ferrum-core/src/render/marks/bar.rs` | 5 builders → accumulator |
| Modify | `crates/ferrum-core/src/render/marks/rect.rs` | 3 builders → accumulator |
| Modify | `crates/ferrum-core/src/render/marks/point.rs` | Cross multi-node via `push_many` |
| Modify | `crates/ferrum-core/src/render/marks/{segment,text,tick,rule}.rs` | row-skip builders → accumulator |
| Modify | `crates/ferrum-core/src/render/marks/{area,line,ribbon,polygon}.rs` | group builders → accumulator (representative row) |
| Modify | `crates/ferrum-core/src/render/pack_instances.rs` | packed-path Rust test only (no format change) |
| Modify | `crates/ferrum-core/src/transform/violin.rs` | add `extent`/`shared_extent`; global-extent helper |
| Modify | `crates/ferrum-core/src/transform/{kde,bin}.rs` | expose global-extent helper (reuse existing logic) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | generalize facet extent pin over Kde/Bin/Violin, full-dataset, multi-group |
| Modify | `src/ferrum/` (violin transform emission site) | emit new ViolinSpec fields / rely on serde default parity |
| Modify | `src/ferrum/composition.py` | consolidate chrome into `_CompositeBase`; reparent Joint/Cluster/Repeat; SVG + interactive threading; `to_html` title |
| Modify | `src/ferrum/_chrome.py` | reuse `chrome_kwargs`/`merge_configure_layers` (change only if needed) |
| Test | `crates/ferrum-core/src/render/marks/*` or dedicated `#[cfg(test)]` | per-builder-family alignment tests |
| Test | `crates/ferrum-core/src/render/pack_instances.rs` | packed >1000-node node-order tooltip test |
| Test | `tests/` (new facet-extent test module) | faceted Bin/Violin single+multi-group extent |
| Test | `tests/` (new composite-title test module) | all 7 composites SVG + interactive + HTML title |
| Modify | `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` | update D2/D7/D10 status |

## 4. Constraints

- **Node-order metadata is the single canonical convention.** Every render path indexes metadata by node position; NO path may compensate with a `data_indices` remap. The investigated WASM `data_indices[node_idx]` lookup is forbidden — it would double-map once builders are fixed.
- **`data_indices` has exactly one entry per emitted node** (repeated for multi-node shapes like Cross), so node→source-row mapping stays correct for all consumers.
- A `debug_assert_eq!(nodes.len(), metadata.len())` guards batch construction when metadata is present; it must trip on a deliberately-misaligned builder (proven by test).
- **No WASM source change** for #6/N1. No change to the packed binary format or `data_indices` semantics.
- **#7 extent computation lives in the transform layer**; `prepare.rs` orchestrates only. New `ViolinSpec` fields use serde defaults mirroring `KdeSpec`/`BinSpec` so existing serialized specs still deserialize.
- **#7 pinned extent = niced global range of the value field over the full pre-facet batch**, applied regardless of `groupby` (covers multi-group/hue).
- **#8 figure-chrome lives in exactly one base class.** No per-class copy. Inner panels never receive figure title/subtitle/caption.
- **#8 interactive = full parity**: on-canvas title band (via merged-scene title) AND HTML document `<title>`. Reuse the single-`Chart` scene-title mechanism (Task 11 spike), not a parallel one.
- CLAUDE.md hard constraints: no matplotlib; no global mutable state; `cargo test` green before any task is "done"; goldens rasterized to PNG and visually Read before commit; `/regression-test` per fix.
- Backward compat: charts already correct (1:1 no-skip marks, non-faceted transforms, single `Chart`, `LayerChart`) render byte-identically.
- Build: `unset CONDA_PREFIX && uv run --no-sync maturin develop` after Rust edits. Rust tests: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test`.

## 5. Tasks

### Task 1: MarkNodes accumulator + guard
- [ ] Create `MarkNodes` with `push(node, row)` and `push_many(nodes, row)` per spec §6; finalize yields nodes + indices for `build_metadata_for_indices`.
- [ ] Add `debug_assert_eq!` node/metadata-length guard at batch construction (spec §7).
- [ ] Add a Rust test that a deliberately misaligned construction trips the guard.
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core mark_nodes`

### Task 2: Migrate `bar.rs` (5 builders)
- Consumes: `MarkNodes` from Task 1 → `crates/ferrum-core/src/render/mark_nodes.rs`
- [ ] Convert polar, ordinal, ordinal_y, quantitative, quantitative_horizontal to accumulator; drop full-row `build_metadata`.
- [ ] Extract shared polar theta→radius mapping helper used by `bar.rs`/`arc.rs` (issue #6 fix-direction).
- [ ] Add row-skip alignment test (bar with null/degenerate rows) — prove fail-before/pass-after.
- [ ] Verify: `... cargo test -p ferrum-core` (bar tests)

### Task 3: Migrate `rect.rs` (3 builders)
- Consumes: `MarkNodes` (Task 1)
- [ ] Convert quantitative_range, ordinal_range, heatmap to accumulator.
- [ ] Add row-skip alignment test (heatmap with skipped cells).
- [ ] Verify: `... cargo test -p ferrum-core` (rect tests)

### Task 4: Migrate `point.rs` (Cross multi-node)
- Consumes: `MarkNodes` (Task 1)
- [ ] Convert builder; Cross shape emits via `push_many(.., row)` so 2 nodes map to one row.
- [ ] Add alignment test for Cross with ZERO skipped rows (spec §9) and with skipped rows.
- [ ] Verify: `... cargo test -p ferrum-core` (point tests)

### Task 5: Migrate `segment.rs`, `text.rs`, `tick.rs`, `rule.rs`
- Consumes: `MarkNodes` (Task 1)
- [ ] Convert each row-skip builder to accumulator; drop full-row `build_metadata`.
- [ ] Add one alignment test per mark (skip case).
- [ ] Verify: `... cargo test -p ferrum-core`

### Task 6: Migrate group marks `area.rs`, `line.rs`, `ribbon.rs`, `polygon.rs`
- Consumes: `MarkNodes` (Task 1)
- [ ] Convert each; push one node per group against the group's representative source row; confirm representative-row choice matches existing per-group semantics.
- [ ] Add per-group alignment test (filtered/short groups).
- [ ] Verify: `... cargo test -p ferrum-core`

### Task 7: Packed/interactive (#6/N1) verification — no WASM change
- Consumes: node-order convention from Tasks 2–6
- [ ] Add Rust test in `pack_instances.rs`: pack a >1000-node batch with render-order ≠ data-order (and a Cross batch); decode tooltips by `node_idx`; assert source-row alignment (spec §9).
- [ ] Confirm by inspection that `crates/ferrum-wasm/src/lib.rs::get_tooltip` and `conditional.rs` selection matching are correct under node-order; record that no WASM edit is made.
- [ ] Verify: `... cargo test -p ferrum-core pack`

### Task 8: ViolinSpec fields + transform global-extent helpers
- [ ] Add `extent: Option<(f64,f64)>` + `shared_extent: bool` to `ViolinSpec` with serde defaults mirroring `KdeSpec`/`BinSpec` (spec §6).
- [ ] Expose a global-extent helper in `kde.rs`, `bin.rs`, `violin.rs` (reuse each module's existing fold; remove duplication from `prepare.rs`).
- [ ] Update the Python violin transform-spec emission site to stay consistent (emit fields or confirm serde default suffices).
- [ ] Verify: `... cargo test -p ferrum-core` (transform tests) + `unset CONDA_PREFIX && uv run --no-sync maturin develop`

### Task 9: Generalize facet extent pin (`prepare.rs`)
- Consumes: global-extent helpers (Task 8)
- [ ] Replace KDE-only fold with dispatch over Kde/Bin/Violin; compute extent over full pre-facet batch; pin regardless of `groupby` (remove the `groupby.is_some()` early-return); rename `fix_kde_extents_for_facet` → `fix_transform_extents_for_facet`.
- [ ] Verify: `... cargo test -p ferrum-core`

### Task 10: Facet extent tests + goldens (#7)
- Consumes: Tasks 8–9
- [ ] Python tests: faceted Bin and Violin, single-group AND multi-group/hue, assert shared value extent across panels/groups; KDE unchanged where already correct (spec §9).
- [ ] Regenerate affected goldens; rasterize → Read PNG; confirm panels share axis (CLAUDE.md golden rule; mind resvg-py path-count caveat).
- [ ] Verify: `uv run pytest -n auto tests/<facet-extent-test>.py -v`

### Task 11: Interactive figure-title spike (#8 open question)
- [ ] Determine how a single `Chart` carries its on-canvas title into the WASM scene (scene-graph title representation). Document the mechanism inline in the plan/spec note (spec §11). Decide whether WASM already renders merged-scene title (expected: yes → no WASM change).
- [ ] Verify: documented finding; no code asserted yet.

### Task 12: Consolidate composite chrome into `_CompositeBase`
- Consumes: spike finding (Task 11); chrome contract (spec §6)
- [ ] Move `_figure_title/_subtitle/_caption` storage, `.properties()` interception, and canonical title accessor into `_CompositeBase`; reparent `JointChart`, `ClusterMapChart`, `RepeatChart` onto it; stop forwarding chrome kwargs to inner panels.
- [ ] Verify: `uv run pytest -n auto tests/ -k composition` (no regressions)

### Task 13: SVG chrome threading for Joint/Cluster/Repeat
- Consumes: Task 12
- [ ] Thread `chrome_kwargs(merge_configure_layers(...))` title/subtitle/caption into each composite's `compose_svg_grid` call (also closes issue-#1 x=0 caution for these 3 sites).
- [ ] Verify: `uv run pytest -n auto tests/ -k composition`

### Task 14: Interactive chrome threading + HTML title
- Consumes: Tasks 11–12
- [ ] Add title/subtitle/caption params to scene-merge functions + `_empty_scene`; populate merged-scene title from base chrome so WASM draws the on-canvas band for all composites (full parity, spec §8).
- [ ] Fix `to_html` to read the canonical base title accessor (resolves `_title` vs `_figure_title` for all composites).
- [ ] Verify: `uv run pytest -n auto tests/ -k "interactive or composition"`

### Task 15: Composite title tests + goldens (#8)
- Consumes: Tasks 12–14
- [ ] Tests for all 7 composites (incl. `RepeatChart`): `.properties(title/subtitle/caption=)` → figure-level chrome in SVG; interactive on-canvas band present; HTML `<title>` correct; inner panels carry no stray title (spec §9).
- [ ] Regenerate affected goldens; rasterize → Read PNG; confirm figure-level placement.
- [ ] Verify: `uv run pytest -n auto tests/<composite-title-test>.py -v`

### Task 16: Close-out — reviews, audits, docs
- Consumes: all prior tasks
- [ ] Heavyweight gates (CLAUDE.md): `rust-review` on marks + transform subsystems; `python-review` on the composition family; scene-pipeline + interactive audits.
- [ ] `/regression-test` per fix at the class level.
- [ ] Update `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` D2/D7/D10 status; optionally file N2 `needs-repro` issue.
- [ ] Verify: full `cargo test` + `uv run pytest -n auto` green.

## 6. Acceptance checks

- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass
- `unset CONDA_PREFIX && uv run --no-sync maturin develop` — builds clean
- `uv run pytest -n auto` — all pass
- Per-builder-family alignment tests + packed >1000-node test pass (fail-before/pass-after proven) — spec §9
- Faceted Bin/Violin single+multi-group share pinned extent; goldens visually inspected — spec §9
- All 7 composites: figure-level title in SVG + interactive band + HTML `<title>`; goldens visually inspected — spec §9
- Construction-time alignment assertion present and proven to trip
- No regression in previously-correct charts (byte-stable where applicable)
- Heavyweight reviews + audits clean; archaeology doc updated

## 7. Open questions

- Interactive figure-title scene representation (Task 11 spike) — resolves whether the WASM path needs any change for #8 (expected: no). Does not change any spec contract.
