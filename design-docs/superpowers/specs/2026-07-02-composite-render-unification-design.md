# Composite Render Unification Design Spec (Phase B)

Delivers the #45 north star (user mandate 2026-07-02: full implementation this round,
no deferral). Companion to the batch remediation spec
(`2026-07-02-remediation-44-45-46-design.md`, whose Amendment block points here) and
to the frozen intent ledger
(`.claude/output/intent/2026-07-02-composite-render-unification-intent.md`).

## 1. Scope

Composition rendering (HConcat/VConcat/ConcatChart grids, JointChart, ClusterMapChart,
RepeatChart, LayerChart overlay, PairGrid-style grids) moves from Python-side merging
onto a Rust composite render path: one render call per composition, taking a composite
spec tree, resolving `resolve=` shared scales through the same machinery facets use,
and emitting one multi-panel `SceneGraph` that serves both static SVG and
interactive/WASM output. The Python merge/injection layers are deleted at end state.

## 2. Goals

- `resolve={channel: "shared"}` observably renders shared axes for every composition
  form and every child shape — flat charts, composite-mark (box/strip/layered)
  children, and grid-composite children (position-wise) — closing GH #45.
- One scale-sharing mechanism in the codebase (the Rust `ResolveMode` machinery);
  `_scale_share.py` injection is deleted.
- One scene-schema owner (Rust `ferrum-scene`); both hand-maintained Python mirrors
  are deleted.
- Interactive output for composites is produced the same way as for facets: one scene,
  N panels, packed buffers emitted once at final coordinates — no post-hoc node
  offsetting or packed-byte patching.
- W5 closed: interactive JointChart/ClusterMap render with the same ratio-proportional
  cell geometry as their static SVG (per-panel layout-scale in the scene schema).
- W4 closed: `Raw` scene nodes in composites are placed at final coordinates during
  scene build; no baked stale offsets can exist.
- Public Python composition API unchanged (`|`, `&`, `.concat()`, `resolve=`,
  `JointChart`, `RepeatChart`, `+` overlay, `.interactive()`, `.save()`).

## 3. Non-goals

- No new composition forms or public API surface.
- No change to single-chart (flat or faceted) rendering; the facet machinery is
  reused, not modified in behavior.
- No SHAP `base_value` work, no `_grid_panels` generalization (separate follow-ups).
- No perf targets beyond "no observable regression" (one render call replaces N+merge,
  so composition render time is expected to improve or hold).
- Byte-identity for composed output is explicitly NOT a goal (user decision:
  visual equivalence with full golden regeneration + PNG inspection).

## 4. System behavior

- Rendering any composition produces output visually equivalent to today's for
  unshared cases: same child content, layout geometry, spacing, ratios, figure chrome
  (title/subtitle/caption band), legends, and clip behavior.
- With `resolve={..: "shared"}` at a composition node:
  - Flat/composite-mark children: one domain per shared channel, resolved across the
    children's merged transform-output batches (facet-style). A box compare therefore
    shares the union of stat extents (whiskers), not raw-column min/max — accepted
    behavior delta, usually tighter and more correct axes.
  - Ordinal channels: order-preserving union (semantics locked by #35's fix).
  - Grid-composite children with congruent trees (same node kinds and child counts at
    every level): leaves pair by tree path; leaf *i* shares with leaf *i* across
    children; panels within one grid never share with each other. Non-congruent
    trees: that channel's sharing is skipped for the group (documented in the
    `resolve=` docstring); rendering is otherwise normal.
  - A faceted child keeps its internal facet resolution; the outer node shares only
    the child's outer scales (pdp compare= independent-x behavior preserved).
  - An explicit user scale on a child channel wins over shared resolution (the
    existing `enc.scale` bypass; Phase A's desugar propagation makes this hold for
    composite marks).
- Interactive composites support the same interactions as today (pan/zoom, tooltips,
  selections, params) with panel-id-scoped state, and additionally render ratio-fitted
  cells proportionally (W5).
- Rust-side composite errors (empty children, malformed tree, ratio arity mismatch)
  surface as `ValueError` naming the composition node kind, matching the existing
  render-error idiom. No warning-fallbacks.

## 5. Architecture

- **Python (`composition.py`)**: user-facing classes keep construction, validation,
  and operator semantics; render methods lower the composition to a composite spec
  tree + per-leaf Arrow payloads and call the composite render entry. JointChart,
  ClusterMapChart, RepeatChart, PairGrid lower to the same tree grammar (grid with
  ratios / wrap / overlay). No Python-side scale math, offsetting, or scene surgery.
- **PyO3 boundary**: a composite spec type (tree of nodes; leaves reference a
  `ChartSpec` + data payload index) with new entries `render_composite_svg` /
  `render_composite_interactive`, mirroring the flat pair's signature (viewport,
  theme, config). Data still crosses once per leaf via Arrow CDI.
- **Rust (`ferrum-core`)**: the composite renderer walks the tree in three passes —
  (1) resolve: shared-channel domains via `ResolveMode::Shared`/`Independent`
  (facet mechanism) with tree-path pairing for congruent composite children;
  (2) layout: each leaf runs the existing single-chart prepare/layout internals with
  resolved domains, then node layout kinds place child panels at absolute rects
  (grid/wrap geometry per the facet-grid style; ratio cells per the existing
  grid-compose math); (3) scene: all panels, axes, legends, decorations, and chrome
  emit into one `SceneGraph` with globally unique panel and clip ids. `walk_svg`
  renders it; the same scene JSON + packed buffers feed WASM.
- **Scene schema (`ferrum-scene`)**: gains a per-panel layout-scale/transform field
  (ratio-fitted cells) consumed by `ferrum-wasm`'s loader. `Raw` nodes carry final
  coordinates when emitted.
- **Deleted at end state**: `_scene_merge.py`; `_scale_share.py` injection and its
  call sites; `compose_svg_horizontal/vertical/grid` PyO3 entries and
  `compositor.rs`/`grid_compose.rs` string surgery (geometry math is absorbed into
  the layout pass); `_scene.py`/`_scene_merge.py` schema mirrors.

## 6. Canonical interfaces / data contracts

**Composite spec tree (PyO3).** A node is one of:

```
Leaf      { spec: ChartSpec, data: payload-index }
Composite { kind: hconcat | vconcat | grid | wrap | overlay,
            children: [node, ...],
            resolve: { x: shared|independent, y: shared|independent },   # default independent
            spacing, row_ratios, col_ratios, ncols/nrows (wrap),
            title/subtitle/caption (root only), config }
```

Exact field names/types are plan-level; the contract is: the tree is self-contained
(no Python callback during render), leaves carry no layout, nodes carry no data.

**Render entries.** `render_composite_svg(tree, payloads, *, viewport, theme, config)
-> str` and `render_composite_interactive(...) -> (scene_json, packed_bytes)` —
signature family identical to the existing flat entries. Flat `render_svg`/
`render_interactive` are unchanged.

**Resolve semantics.** Per node, per positional channel. `shared` at a node resolves
one domain across its children's corresponding channels using merged transform-output
batches; recursion into composite children only by congruent tree-path pairing;
non-congruent → skip that channel for the group. Explicit child `enc.scale` domains
bypass sharing (existing rule). Defaults: composition nodes `independent` (today's
behavior); facet internals unchanged (`shared` default within a facet).

**Scene schema additions.** (1) per-panel layout-scale (or transform) field, default
identity, applied by both `walk_svg` and the WASM loader — the single mechanism for
ratio-fitted cells; (2) panel ids unique across the whole scene; clip ids uniquified
at scene build. Scene JSON remains the only contract between core and WASM; no Python
mirror may exist after this phase.

**Congruence.** Two trees are congruent iff same node kind, same child count, and
children pairwise congruent (leaves match leaves). Facet-expanded panels inside a leaf
do not participate in outer pairing.

## 7. Invariants and constraints

- Flat (non-composed) chart output stays byte-identical; facet goldens pass
  un-regenerated.
- Every regenerated or new golden is rasterized and PNG-inspected before commit
  (CLAUDE.md goldens rule); composition goldens (~16) all regenerate in this phase.
- `cargo test` green (hard constraint), `cargo clippy -p ferrum-wasm` clean, full
  pytest green at every landed commit.
- No matplotlib; no global mutable state; data crosses the boundary once per leaf.
- Public composition API signatures unchanged; `ferrum-spec.md` gets dated notes for
  behavior deltas (shared-domain derivation; composed-output regeneration).
- Interactive verification is part of done: headless WASM capture screenshots for the
  composed forms, visually confirmed (W5 proportions, W4 raw-node placement).
- Phase A must land first (desugar scale propagation is prerequisite for the
  explicit-override rule on composite marks).

## 8. Key decisions and tradeoffs (defended choices)

**D1 — Architecture: composite spec tree rendered by the facet machinery.**
*Reframe:* research showed composition today is two unrelated hacks (opaque-SVG string
surgery for static; a 1,049-line Python scene merge with packed-buffer binary patching
for interactive), while facets already do exactly what #45 needs — shared resolution
(`ResolveMode` in `layout/facet.rs`), multi-panel layout (`FacetGrid`), one
`SceneGraph.panels` scene consumed identically by SVG and WASM. `ChartSpec` does not
nest, so *some* new spec surface is unavoidable in every candidate.
*Candidates:* (A) composite spec tree + one Rust render reusing the facet machinery
(precedent: `layout/facet.rs`, `scene_build.rs` panel loop); (B) keep per-child
renders, port `_scene_merge.py` to Rust + a shared-domain pre-pass (precedent:
`compositor.rs` string compositor); (C) Python-side fix only (`_scale_share`
extension — the original Phase A Task 5).
*Chosen:* A. It is the only candidate where sharing flows through the facet mechanism
(the stated intent), where the scene schema gets a single owner, and where W4/W5 are
fixed structurally rather than patched. It reuses the mature per-chart layout
internals per leaf, so the rewrite risk is bounded to tree walking + panel placement.
*Rejected:* B — converts Python debt to Rust debt: keeps two-phase render, keeps
domain-injection semantics bolted beside the facet mechanism (two systems remain, the
incoherence this phase exists to kill), and still needs the per-panel scale slot for
W5 separately; its only advantage (less layout churn) was neutralized by the user's
visual-equivalence decision. C — already superseded by user mandate; kept here as the
record that the proportionate fix was offered and declined in favor of the north star.
*Reach:* every composition form (concat/joint/clustermap/repeat/layer/pairgrid) is
enumerated in §1 and lowers to one tree grammar; forms were inventoried from
`composition.py` (2,174 lines) — anything else that renders via `_scene_merge` or
`compose_svg_*` would be surfaced by deleting those (compile/test failure), which is
the completeness backstop.

**D2 — Congruence: tree-path pairing, non-congruent skip.** Carried from the
coherent-change-defended Phase A decision (user-approved): position-wise pairing is
the only semantics that avoids collapsing heterogeneous grid panels (the #35 pdp
x-collapse failure class); real producers (`compare=`) always emit congruent trees;
there is no semantically right union for mismatched trees, and erroring would break
currently-working hand-built concats. Rust sees the whole tree, so the skip is
implemented (and unit-tested) in one place.

**D3 — Shared domains derive from transform-output batches.** Facet mechanism
property, accepted as a behavior improvement (box compare shares whisker extents).
Alternative (raw-column min/max, today's Python semantics) would require a parallel
domain path in Rust solely to preserve looser axes — rejected as preserving an
artifact of the old implementation, not a contract. `ferrum-spec.md` gets a dated
note.

**D4 — Ratio cells via a per-panel scene field, not nested scenes.** The SVG path's
nested-`<svg preserveAspectRatio>` trick has no scene/WASM analog (that gap IS W5).
One per-panel scale field consumed by both walkers keeps SVG and WASM geometry from
one source. Alternative — teaching the WASM loader to compose nested scenes — creates
a second composition mechanism inside the renderer. The exact representation
(scale factor vs 2×3 transform) is a determined implementation decision: **routed
through coherent-change decision-only during planning** (user process decision
2026-07-02), along with the leaf/`prepare_and_layout` reuse seam and packed-buffer
panel indexing.

**D5 — Incremental cutover, hard deletion at end.** Per-form switch (linear →
grid/wrap → ratio forms → repeat → overlay) with per-form golden regeneration keeps
each landed commit green and inspectable; a big-bang switch would put ~16 golden
inspections and all five forms' risk into one review. No fallback path is kept at end
state (ledger #7; a retained fallback would silently re-bifurcate the mechanism).

## 9. Acceptance criteria

Intent ledger statements 1–7 are the outer gate. Concretely:

- `cv_scores_chart(m, X, y, compare={...})` (and box/strip diagnostics generally):
  rendered per-panel y tick extents equal the union domain (SVG-extent inspection).
- `residuals_chart(..., compare={...})`: leaf-position pairs share; positions within
  one grid do not.
- For each composition form (hconcat, vconcat, concat grid, joint, repeat, layer):
  a behavior test shows `shared` unifying and `independent` isolating rendered axes.
- pdp `compare=` keeps independent per-feature x (existing test passes unchanged).
- Explicit user scale on a composite-mark child overrides sharing.
- Non-congruent grid composition renders without error and without sharing.
- Interactive: headless-capture screenshots confirm (a) JointChart 2×2 with correct
  marginal ratios matching the SVG (W5), (b) legend/colorbar raw defs correctly
  placed in a composed export (W4), (c) tooltips/selections still scoped per panel.
- Flat-chart and facet goldens pass un-regenerated; all composition goldens
  regenerated + PNG-inspected; `grep`-level proof that `_scene_merge.py`,
  `_scale_share` injection, and `compose_svg_*` entries no longer exist.
- Full pytest, `cargo test`, `cargo clippy -p ferrum-wasm` green; `nox -s lint` clean.
- GH #45 closed with commit references; W4/W5 removed from CLAUDE.md's known
  limitations; `ARCHITECTURE.md` + `ferrum-spec.md` updated with dated notes.

## 10. Validation strategy

Four layers: (1) Rust unit tests for tree resolution, congruence pairing, and panel
placement geometry; (2) Python behavior tests asserting rendered axis extents
(reusing the `test_facet_shared_extent` parsing helpers) across all forms and child
shapes — this is where #45's regression coverage lives, RED against the pre-phase
code; (3) golden regeneration with mandatory PNG inspection per form; (4) headless
WASM capture for interactive proof (W4/W5 and interaction smoke). Byte-identity of
flat/facet output validated by the untouched golden suite. The #45 pinned repro flips
green and is kept as a permanent test.

## 11. Open questions

None blocking the spec. Three determined implementation decisions are deliberately
deferred to coherent-change decision-only during planning (per D4): per-panel
scale-slot representation; the leaf render seam into `prepare_and_layout`;
packed-buffer panel indexing scheme.
