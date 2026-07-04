# Composite Render Unification (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Move all composition rendering (concat/grid/joint/clustermap/repeat/layer) onto a Rust composite spec tree rendered through the facet scale-sharing machinery, emitting one multi-panel scene for SVG and interactive, then delete the Python merge/injection layers — closing GH #45, W4, and W5.

## 2. Spec references

- `design-docs/superpowers/specs/2026-07-02-composite-render-unification-design.md` (all; §6 contracts, §8 defended decisions, §9 acceptance)
- `.claude/output/intent/2026-07-02-composite-render-unification-intent.md` (frozen outer gate)
- Batch spec Amendment block: `design-docs/superpowers/specs/2026-07-02-remediation-44-45-46-design.md`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `design-docs/superpowers/specs/2026-07-02-composite-render-unification-decisions.md` | three coherent-change decision records (spec §11) |
| Create | `crates/ferrum-core/src/spec/composite.rs` | composite tree types + serde (spec §6) |
| Modify | `crates/ferrum-core/src/spec/mod.rs` | export composite types |
| Modify | `crates/ferrum-scene/src/types.rs` | per-panel layout-scale field (spec §6) |
| Modify | `crates/ferrum-core/src/render/svg_walk.rs` | apply per-panel scale |
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | consume per-panel scale |
| Create | `crates/ferrum-core/src/render/composite.rs` | resolve/layout/scene passes (spec §5) |
| Modify | `crates/ferrum-core/src/render/binding.rs` | `render_composite_svg` / `render_composite_interactive`; later delete `compose_svg_*` |
| Modify | `crates/ferrum-core/src/lib.rs` | register new entries |
| Modify | `src/ferrum/composition.py` | lower forms to tree; delete merge paths |
| Modify | `src/ferrum/_scene.py` | route composites to new entries; delete schema mirror |
| Delete | `src/ferrum/_scene_merge.py` | end-state deletion (spec §5) |
| Delete | `src/ferrum/_scale_share.py` | end-state deletion (injection retired) |
| Delete | `crates/ferrum-core/src/render/compositor.rs`, `render/grid_compose.rs` | absorbed into layout pass |
| Test | `crates/ferrum-core/src/render/composite.rs` (unit tests) + `tests/test_composite_render_*.py` (new) | resolve/congruence/extents behavior |
| Test | `tests/goldens/**` composition goldens (~16) | regenerate per form, PNG-inspected |
| Modify | `CLAUDE.md`, `design-docs/architecture/ARCHITECTURE.md`, `ferrum-spec.md` | W4/W5 removal + dated notes |

## 4. Constraints

- Branch: `feat/composite-render-unification` off `main` **after** Phase A (`fix/issues-44-46-remediation`) merges; Phase A's desugar scale propagation is a prerequisite (spec §7). Never commit to main.
- Dispatch: `rust-coder` for `crates/**`, `python-coder` for `src/ferrum/**` + `tests/**`; clear file boundaries when a task needs both.
- The three spec-§11 sub-decisions are settled in Task 1 via `chris-code:coherent-change` decision-only BEFORE any implementing task consumes them; implementers receive the decision record path, never re-decide.
- Flat (non-composed) and facet output must stay byte-identical throughout; their goldens never regenerate. Composition goldens regenerate once per form-cutover task; the ORCHESTRATOR reads every rasterized PNG (`regen_and_verify` / `scripts/snapshot-goldens.py`) before that task's commit.
- Green at every landed commit: full `uv run pytest -n auto`; `cargo test` via `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test`; `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` when ferrum-wasm changed; WASM rebuild via `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/` when scene_load.rs changed.
- Rust builds: `unset CONDA_PREFIX && uv run --no-sync maturin develop` after any crates/ change before running pytest.
- Resolve semantics per spec §6: composition default `independent`; shared = merged transform-output batches; congruent tree-path pairing only; non-congruent skip; explicit `enc.scale` wins; ordinal unions order-preserving.
- No warning-fallbacks; errors are typed `ValueError` naming the composition node kind.
- During cutover tasks, forms not yet cut over keep the old path working (suite green); the old path is deleted only in Task 10 — but no fallback survives Task 10 (spec §8 D5).
- Commits via `commit-commands:commit`; `python-review-lite` + `rust-review-lite` gates per commit by touched language; no Claude authorship trailers.
- Behavior tests assert rendered axis extents (reuse `tests/test_facet_shared_extent.py` parsing helpers); #45 regression tests must be proven RED against pre-phase code (branch point) before their cutover task lands.

## 5. Tasks

### Task 1: Settle the three §11 sub-decisions (coherent-change decision-only)
- [ ] Run `chris-code:coherent-change` decision-only for: (a) per-panel scale-slot representation in `ferrum-scene`; (b) leaf render seam into `prepare_and_layout`; (c) packed-buffer panel indexing scheme
- [ ] Record the three defended choices in `design-docs/superpowers/specs/2026-07-02-composite-render-unification-decisions.md`; present to user for sign-off
- [ ] Verify: decisions file committed (docs-only commit)

### Task 2: Composite spec types (Rust)
- Consumes: tree contract from spec §6
- [ ] `spec/composite.rs`: node enum (leaf/composite), layout kinds, resolve struct, spacing/ratios/wrap fields, root chrome fields; serde round-trip; PyO3 coercion from the Python-side dict/tree
- [ ] Unit tests: round-trip, validation errors (empty children, ratio arity) as typed errors per spec §4
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core`
- [ ] Commit (rust-review-lite gated)

### Task 3: Scene schema per-panel scale + walkers
- Consumes: decision (a) from Task 1 → decisions file
- [ ] Add per-panel layout-scale field to `ferrum-scene` types (default identity; serde back-compat: absent = identity)
- [ ] `svg_walk.rs` applies it; `scene_load.rs` (WASM) applies it to panel geometry + packed instances
- [ ] Unit tests both sides; rebuild WASM module
- [ ] Verify: `cargo test` + `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` + full `uv run pytest -n auto` (schema is shared contract; flat/facet goldens must pass untouched)
- [ ] Commit (rust-review-lite gated)

### Task 4: Composite resolve pass
- Consumes: Task 2 types; resolve semantics spec §6; decision (b) for the leaf seam
- [ ] `render/composite.rs`: tree walk computing per-channel resolved domains — shared via `ResolveMode`/merged batches (facet mechanism), congruence check + tree-path pairing, non-congruent skip, explicit-scale bypass, ordinal order-preserving union
- [ ] Rust unit tests: flat children shared/independent, box-transform children (stat-extent union), congruent grids pair per position, non-congruent skips, facet-child outer/inner separation
- [ ] Verify: `cargo test -p ferrum-core` (DYLD prefix as in Constraints)
- [ ] Commit (rust-review-lite gated)

> **Task 5 split (2026-07-03, implementer-proposed, orchestrator-accepted):** 5a =
> D4b context threading (scale pipeline, byte-identical when None, lands first);
> 5b = layout + scene renderer core (one SceneGraph, ratio math, renumbering,
> chrome via `figure_chrome::title_nodes` — NOT the stale `build_figure_chrome_nodes`
> name — dead-code allow removals); 5c = PyO3 entries + WASM interaction-geometry
> baking (D4a addendum, double-apply hazard, wasm-clippy gate). Each lands as its
> own gated commit; the checklist items below distribute accordingly.

### Task 5 (split 5a/5b/5c): Composite layout + scene passes + PyO3 entries
- Consumes: Tasks 2–4; decisions (b) leaf seam and (c) packed indexing
- [ ] Resolve the raw-vs-baked panel-geometry split (decisions doc D4a addendum): interaction consumers (hit_test/lib/spatial_index/render upload) must read geometry consistent with baked layout-scale panels
- [ ] Layout pass: hconcat/vconcat/grid/wrap/overlay placement + ratio cells (absorb `grid_compose.rs` math), leaf render via the decided seam with resolved domains; scene pass: one `SceneGraph`, globally unique panel/clip ids, chrome in-scene, `Raw` nodes at final coords
- [ ] `render_composite_svg` / `render_composite_interactive` in `binding.rs` + `lib.rs` registration (signature family per spec §6)
- [ ] Rust unit + snapshot-level tests (scene shape, panel rects, id uniqueness)
- [ ] Verify: `cargo test` + `maturin develop` + full `uv run pytest -n auto` (no Python consumer yet — suite must stay green untouched)
- [ ] Commit (rust-review-lite gated)

> **Task 6 finding (2026-07-03): new Task 5d required.** The 5c PyO3 entry applies
> ONE theme/viewport/chart_config to all leaves (`build_composite_leaves` collapses
> the per-leaf `CompositeLeafInput`). Task 6 therefore GATES the new path to
> homogeneous linear compositions (identical leaf theme+viewport+chart_config, no
> per-node configure layers, x/y-only shared resolve, no nested chrome) with the old
> path retained as fallback. **Task 5d (rust-coder, before Task 7 completes):**
> per-leaf binding inputs on both composite entries (theme/viewport/chart_config per
> leaf; per-node config; composite resolve beyond x/y stays out — color/size sharing
> remains Python-side until re-decided), then Tasks 7-9 widen the gate as forms cut
> over, and Task 10's deletion requires the gate to be fully open.
>
> **Task 7 addition to 5d scope:** per-child panel labels. `_compose_compare` labels
> children via `child.properties(title=name)`; a composite (e.g. VConcat) child's
> label becomes a non-root figure title, which the wire forbids — so residuals
> compare= (the #45 headline) stays gated. 5d must add a per-child label field on
> composite nodes (strip-title-like, rendered as the panel-group header the old
> compositor drew), after which the figure-chrome gate drops and Task 7's strict
> xfail (`test_titled_composite_children_share_axis_position_wise`) flips. Also
> recorded: sparse-grid holes have no wire concept (RepeatChart corner=True) —
> resolve by Task 10 (wire support or documented permanent gate).

### Task 6: Cutover — linear forms (HConcat/VConcat)
- Consumes: Task 5 entries; tree lowering contract spec §5
- [ ] `composition.py`: tree-builder helper; route `_CompositeBase` static + interactive renders for HConcat/VConcat through composite entries
- [ ] Behavior tests (new `tests/test_composite_render_linear.py`): shared/independent rendered extents for flat AND box children (box = #45 slice, RED pre-phase), explicit-scale override, error cases
- [ ] Regenerate linear-form goldens (e.g. `vconcat_full_chrome`); orchestrator PNG-inspects before commit
- [ ] Verify: full `uv run pytest -n auto`
- [ ] Commit (python-review-lite gated)

### Task 7: Cutover — grid/wrap forms (ConcatChart, PairGrid-style, compare=)
- Consumes: Task 6 pattern
- [ ] Route ConcatChart (wrap + sparse grids) static + interactive through composite entries; `_compose_compare` output rides automatically
- [ ] Behavior tests (`tests/test_composite_render_grid.py`): #45 acceptance — `cv_scores_chart` compare= shared y == union (RED pre-phase); `residuals_chart` compare= position-wise pairing; non-congruent skip; pdp compare= independent-x test passes unchanged
- [ ] Regenerate grid/pairplot goldens; orchestrator PNG-inspects
- [ ] Verify: full `uv run pytest -n auto`
- [ ] Commit (python-review-lite gated)

> **Task 8 split (2026-07-03, orchestrator decision after two API-killed attempts):**
> the grid-hole question is decided — option (b), Rust wire hole support: a hole
> child in grid/wrap layouts (kind "hole"), skipped by layout/leaf-counting/packing,
> covering BOTH JointChart's empty 2×2 corner and RepeatChart corner=True — this
> also resolves the recorded Task 10 sparse blocker. **8a (rust-coder):** the hole
> wire + layout skip + validation + PyO3 + tests. **8b (python-coder):** the
> Joint/ClusterMap cutover per the original checklist, emitting holes for empty
> cells.

### Task 8 (split 8a/8b): Cutover — ratio forms (JointChart, ClusterMapChart)
- Consumes: Task 3 per-panel scale; Task 6 pattern
- [ ] Lower 2×2 ratio grids to composite tree with row/col ratios; static + interactive through composite entries (marginal axis hiding preserved)
- [ ] Behavior tests: SVG marginal proportions match ratio; interactive scene carries per-panel scale (W5 slice)
- [ ] Regenerate joint/clustermap goldens; orchestrator PNG-inspects
- [ ] Verify: full `uv run pytest -n auto`
- [ ] Commit (python-review-lite gated)

### Task 9: Cutover — RepeatChart + LayerChart overlay
- Consumes: Task 6 pattern
- [ ] RepeatChart lowers its generated grid; LayerChart lowers to overlay node (children share one panel rect, resolve per spec §6)
- [ ] Behavior tests: repeat shared/independent; overlay shared/independent extents
- [ ] Regenerate repeat/layer goldens; orchestrator PNG-inspects
- [ ] Verify: full `uv run pytest -n auto`
- [ ] Commit (python-review-lite gated)

> **Task 10 pre-flight findings (2026-07-04):** the deletion must reconcile with the
> retained legacy consumers: (a) LayerChart's interactive path stays on the legacy
> merged-Chart route (one-panel interactive contract, Task 9 fix) and conditionally
> calls `_scale_share.compute_union_domain`/`inject_scale` when `resolve=` has a
> shared channel — port those two calls inline (or verify unreachable) before
> deleting `_scale_share.py`; it does NOT touch `_scene_merge.py`. (b) The remaining
> Python-side gates (configure layers — possibly droppable now that 5d's per-leaf
> chart_config exists; non-x/y shared resolve; empty-data leaves) keep legacy
> fallback paths alive — Task 10 must either widen those gates away or surface the
> ledger-statement-7 tension ("_scene_merge/_scale_share/compose_svg_* no longer
> exist") to the user for an explicit scope decision. Do not silently keep remnants.
>
> **USER DECISION (2026-07-04): FULL-LITERAL.** Task 10 splits: **10-pre-b
> (rust-coder):** non-positional (color/size) channel support in the composite
> resolve pass + the D4b seam, so non-x/y shared resolve rides the tree. **10-pre-a
> (python-coder, consumes 10-pre-b):** drop the non-x/y gate; lower empty-data
> leaves as holes (faithful — the legacy placeholder is visually blank); recursive
> lowering for non-leaf grid cells; verify + drop the configure-layers gate (per-leaf
> chart_config); port LayerChart's two inline _scale_share calls (its interactive
> stays legacy-merged by the one-panel contract, but merged-Chart rendering is the
> FLAT path — it needs no composite machinery, just the two-line domain union,
> ported). **10 proper:** the deletion sweep with grep proofs, statement 7 literal.
>
> **10-pre-a FINDINGS (2026-07-04).** (1) A Rust `ctx_opt` gate bug (composite_render.rs
> checked only x/y before threading the resolved context) was found by 10-pre-a and
> fixed inline by the orchestrator via `LeafScaleContext::is_empty()`; color/size-only
> sharing now works end-to-end (regression test
> `test_hconcat_resolve_shared_color_alone_unions_domain`). (2) One narrow gate
> remains after 10-pre-a: composite-level `configure_padding(left=/right=)` /
> `configure_title(anchor=)` (`_composite_chrome_kwargs`) — a genuine composite-render-entry
> gap, since `inject_root_chrome` (composite_render.rs) never reads a per-composite
> chrome override on either path. **Task 10 proper must close it BEFORE the deletion
> sweep** by wiring the composite tree's reserved root-only `config` slot (currently
> dead on the Rust side) into `inject_root_chrome`'s FigureChrome, then dropping the
> `_composite_chrome_kwargs` gate; `tests/test_flexibility_caps/test_d10_figure_title.py`
> (t16/17/18/22/23/24) is the behavioral coverage. Empty-data holes exist only on
> grid/wrap wire layouts; hconcat/vconcat/overlay still decline empty leaves to
> legacy — Task 10 must close this before deletion, faithful to legacy's blank
> rendering (extend hole support to hconcat/vconcat on the wire — `build_placed`'s
> zero-footprint arm already exists — and for overlay skip empty children at
> lowering, since an empty layer draws no marks; an all-empty tree keeps the
> existing decline-to-`None` guard, which after deletion becomes a typed error, not
> a silent fallback). Also: the ported `compute_union_domain`/`inject_scale` call
> sites (`share_scale`, `RepeatChart._apply_resolve`, `LayerChart._build_merged`)
> are the FLAT-path domain union and REMAIN after the sweep — Task 10's grep proofs
> target `_scale_share`/`_scene_merge`/`compose_svg_*` symbols, not these.

### Task 10: Hard deletion + grep proofs
- Consumes: all forms cut over (Tasks 6–9)
- [ ] Delete `_scene_merge.py`, `_scale_share.py` + all imports/call sites (`_apply_resolve` collapses to tree resolve fields), `compose_svg_*` bindings, `compositor.rs`, `grid_compose.rs`, `_scene.py` schema mirror, `figure_title_nodes` binding if unconsumed
- [ ] Consolidate the channel-rebuild idiom (Phase A design-review S2s): with `_scale_share.py::inject_scale` gone, `src/ferrum/_desugar.py:_merge_positional_channel_scale` is the sole clone-channel-with-modified-options site — introduce a `ChannelBase` option-rebuild helper (also usable by `_apply_remap`) and normalize the attached-scale shape (pyclass vs dict) while touching it
- [ ] Grep proofs: no references to deleted symbols anywhere (spec §9)
- [ ] Verify: `cargo test` + `maturin develop` + full `uv run pytest -n auto` + `nox -s lint`
- [ ] Commit (both review-lite gates)

### Task 11: Interactive verification (W4/W5 close)
- Consumes: Tasks 3, 8, 10; headless harness per memory `reference_headless_wasm_capture`
- [ ] Headless-Chrome captures of `.interactive().save()` output: JointChart 2×2 proportions vs SVG (W5), composed export legend/colorbar raw defs placement (W4), tooltip/selection smoke on a concat
- [ ] Orchestrator reads every screenshot; any mismatch loops back to the owning task's coder
- [ ] Verify: screenshots visually confirmed; captures archived under `.claude/output/phase-b-captures/`

### Task 12: Docs + close
- [ ] Remove W4/W5 from CLAUDE.md known-limitations; update `ARCHITECTURE.md` (composition section) and `ferrum-spec.md` dated notes (shared-domain derivation; composed-output regeneration); update code-archaeology tracker (`SceneNode::Raw` WASM item)
- [ ] Verify: `nox -s docs` if docs site references change; `uv run pytest -n auto` final
- [ ] Hand back for verification-before-completion (design reviewers + intent-reviewer against the frozen ledger) + finishing-a-development-branch; close #45 with commit references after user confirmation

## 6. Acceptance checks

- Intent ledger statements 1–7 all observably met (intent-reviewer gate)
- Spec §9 list: per-form shared/independent extents, #45 compare= criteria, override/non-congruent/pdp cases, W4/W5 captures, grep proofs, byte-identical flat/facet goldens
- `uv run pytest -n auto`, `cargo test`, `cargo clippy -p ferrum-wasm`, `nox -s lint` — all green
- Every regenerated golden PNG-inspected by the orchestrator before its commit

## 7. Open questions

- None beyond Task 1's three scheduled decisions (settled before consuming tasks start).
