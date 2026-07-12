# Post-v0.19 Sweep Remediation (GH #71–#74) Design Spec

Origin: release-scoped bug hunt + design review of v0.19.0..main (issues #69–#76; this
spec covers the four batches routed through coherent-change batch mode). Defended
choices with full candidate rebuttals and research citations live in the session
artifact `DEFENDED_CHOICES_71-74.md`; this spec pins the resulting contracts.

## 1. Scope

Four coherent remediations: (A) #71 — unify independent-y semantics across the two
spellings (`LayerChart(resolve={"y":"independent"})` and `chart + SecondaryY`) and fix
the rename-sentinel and tooltip leaks; (B) #72 — make layer-bound domain params reach
the wire and unify per-slot y-domain resolution so axis ticks, mark placement, scene
`y_domains`, and the WASM initial view provably share one resolution; (C) #73 — make
WASM hit-testing and axis relabeling slot-aware under runtime rescales; (D) #74 —
define and implement nested-composite shared resolve for color/size and make
`configure_legend(orient="none")` suppress the figure legend band.

## 2. Goals

- Every red test in the #71–#74 evidence set (`tests/test_bug_hunt_secondary_y.py`,
  `tests/test_bug_hunt_interactive_slots.py`, `tests/test_bug_hunt_shared_legend.py`,
  the `hit_test.rs`/`text_json.rs` bug-hunt tests, and the tooltip test in
  `tests/test_bug_hunt_figure_legend.py`) flips green.
- Both spellings of dual-axis semantics carry identical error contracts.
- No internal disambiguation sentinel appears in any user-visible output; renders are
  deterministic across runs.
- One resolution of each secondary y-domain feeds all four consumers.
- GH #60 (slot id on axis-tick text nodes) is implemented and closed by (C).
- Nested-resolve composition semantics are defined in the spec/design docs, not left
  emergent.

## 3. Non-goals

- GH #63 (SY-8 1-based vs 0-based slot-list conventions): respected, not unified.
- GH #67 (band-geometry north star): untouched.
- Making `_ident_`/`_auto_` suffixes deterministic (follow-up unless trivially free).
- Any change to x/y positional tree-path pairing semantics for grids.
- Wire back-compat for the removed `secondary_y` structural JSON (intentional, #52).

## 4. System behavior

**A — independent-y semantics (#71).**
- A composition (`HConcat`/`VConcat`/deep nesting) with `resolve={"y": "shared"}` over
  any member carrying independent-y layers raises the #52 §4 conflict error — whether
  the member is a `LayerChart(resolve={"y":"independent"})` or a plain `Chart`
  produced by `chart + SecondaryY(...)`.
- `LayerChart(member, other, resolve={"y": "shared"})` where `member` carries flagged
  layers raises instead of silently forcing shared.
- `resolve={"x": "shared"}` over independent-y members does not raise (unchanged).
- A dual-axis chart over heterogeneous data with colliding column names shows the
  original column name on the right axis (and in tooltips); no `__rhs_` token appears
  in SVG or scene JSON; output is byte-identical across repeated runs.
- Explicit `tooltip=` on the primary layer of a layered chart applies to that layer
  only; other layers still receive auto-injected tooltips (matching the existing
  later-layer behavior).
- `SecondaryY(mark=...)` with a non-primitive mark name raises a `ValueError` naming
  the valid primitive marks and pointing to the `LayerChart` spelling for composite
  overlays.

**B — per-slot y-domain resolution (#72).**
- A `fm.param(...)` used as a scale-domain on a *layer* encoding reaches `spec.params`
  exactly as a chart-level one does.
- For an independent-y layer with a domain param: the static SVG right-axis ticks,
  mark placement, scene-JSON `y_domains[k]`, and the WASM initial view all reflect the
  param's current (initial) value.
- Charts without params, and layouts without secondary layers, render byte-identically
  to today.

**C — WASM slot awareness (#73).**
- After a domainParam/brush rescale on an independent-y layer, pointer hit-testing
  (tooltips, href, selections) finds marks at their *displayed* positions and misses
  their stale pre-rescale positions.
- A right axis relabels correctly under slot rescale regardless of tick count
  (including a single tick from a degenerate domain).
- An out-of-range `y_slot` fails a debug assertion in test builds; release builds keep
  the existing clamp.

**D — nested composite resolve (#74).**
- Semantic rule (new, canonical): for **color/size**, a composite node whose effective
  mode for the channel is `shared` unions the domain across its **entire leaf span**
  (all descendant leaves, through nested composites and spliced overlay subtrees);
  a descendant node's *explicit* resolve for that channel wins over inheritance; an
  unset descendant inherits the outer effective mode. For **x/y**, existing positional
  tree-path pairing across congruent direct children is unchanged.
- The figure-level legend band attaches at the outermost node whose effective legend
  mode for the channel is shared; leaves it covers get per-panel suppression. One
  `grp` legend renders for all three failing nesting shapes.
- `configure_legend(orient="none")` at any level disables legends for the charts it
  covers via the same mechanism as `Color(legend=None)`: per-panel legends and the
  figure band are both suppressed (all-disabled ⇒ no band, per the existing §9.8
  rule). Single-chart `configure_legend(orient="none")` likewise suppresses that
  chart's legends (today it is a silent no-op).
- `.spec` introspection behavior is unchanged; its docstrings state the flat-dict
  raw-view exception instead of claiming equivalence.

## 5. Architecture

- Python owns: independent-y capability detection and conflict raising at lowering
  time; param hoisting onto the wire; rename/display-name hygiene; mapping
  `orient="none"` onto the legend-disabled signal. Rust owns: scale/domain resolution
  (including param substitution), slot planning, band planning, hit-testing.
- Param substitution becomes a `scale_resolve` capability (currently a scene_build
  private pre-pass) so both the prepare stage (axis inputs) and scene_build (marks,
  y_domains) consume the same substituted resolution.
- The layer→slot mapping becomes a single plan computed at prepare time and stored on
  the prepared inputs, consumed by axis-input building, per-panel scale resolution,
  axis routing, and param-binding collection (mirrors the #16 `LegendBandPlan`
  compute-once/consume-later pattern and its index-keying rationale).
- The WASM spatial index carries each mark's `y_slot`; hit-test entry points receive
  the renderer's `slot_rescales` + `panel_slot_counts` and apply the composed
  panel∘slot inverse per candidate using the same `transform_slot_index` mapping the
  render upload uses.
- Axis-tick text nodes carry their slot id in the scene contract; `text_json` selects
  rescale affines by explicit slot (GH #60), retiring the column-frequency heuristic
  for slot identity.
- The two congruence-gated walks (domain-union resolve and legend-band planning) keep
  their bit-for-bit agreement contract; the leaf-span rule for color/size changes in
  both or neither.

## 6. Canonical interfaces / data contracts

- `Chart` gains an internal capability predicate (name at implementer's discretion,
  mirroring `_supports_user_resolve`) that reports whether any of its layers carry
  `independent_y=True`; all composition-level independent-y detection consults it.
- `SecondaryY.mark` contract: must name a primitive mark
  (`point|line|bar|area|rule|text|tick|rect`); violation raises `ValueError` at
  desugar time.
- Scene contract addition (ferrum-scene): axis-tick text nodes gain an optional slot
  field (serde-default `None` for absent/legacy scenes; non-axis text untagged).
  `y_domains`, `y_slot_levels`, `secondary_affines` keep their existing (split)
  indexing conventions.
- WASM hit-test entry points (`hit_test`, `hit_test_nearest`, `*_with_index`, and the
  selection click path) accept slot-rescale state; absence of rescales (identity) must
  reproduce today's behavior exactly.
- `_collect_params` contract: the wire's `params` list is the union of chart-level and
  per-layer `Parameter` references (deduped by name), regardless of which layer
  declares them.
- Nested-resolve wire semantics: resolve remains per-node on the wire; *effective*
  channel mode at a node = its own setting if explicit, else the nearest ancestor's
  effective mode (color/size channels only). This rule is normative for both the
  domain-union walk and the band planner.

## 7. Invariants and constraints

- Existing goldens for param-free charts stay byte-identical:
  `tests/goldens/secondary_y_axis/*.svg`, shared-legend goldens, pre-52 layout
  byte-stability, `.spec` introspection shapes.
- The multi-y-layer member guard, x-shared non-raise, first-member third-axis
  stacking, and grid/hole behaviors are pinned by existing tests and must not change.
- No global mutable state; no new warn-fallbacks or NotImplementedError paths.
- `cargo test` (ferrum-core + ferrum-wasm) and the full pytest suite green at close;
  regenerated goldens (if any) visually inspected per CLAUDE.md before commit.
- Stray-label rejection in zoomed text relabeling must remain: a non-axis text node
  matching a tick string must never be treated as an axis label (structural guarantee
  via slot tagging replaces the `c >= 2` frequency guard).

## 8. Key decisions and tradeoffs

Each decision below was produced by the coherent-change engine (decision-only, batch
mode); full candidate tables and rebuttals in `DEFENDED_CHOICES_71-74.md`.

1. **Capability predicate over isinstance patching or producer normalization** (#71):
   bridges the two independent-y encodings at every consumer; mirrors the #16
   `_supports_user_resolve` idiom. Rejected: making `chart + SecondaryY` return a
   `LayerChart` (public-type change, breaks one-panel pins); point-patching
   `_contains_independent_y_layer` only (leaves the force-shared path silent).
2. **Display-name preservation + deterministic suffix** over Rust-side stripping
   (#71): the rename is a Python data-layer artifact; presentation overrides already
   exist. Determinism via a counter/index replaces one `id()` use; the sibling
   `_ident_`/`_auto_` suffixes are follow-up.
3. **Gate the injector, not the merge** (#71 tooltip): the per-layer Task-10f loop is
   already correct; the chart-level short-circuit must yield to it when layers exist.
   Rejected: stop promoting first-layer encoding in `Chart.__add__` (blast radius).
4. **Python param hoisting is the fix; Rust unification is the hardening** (#72):
   reproduction proved the store is empty (`_collect_params` never walks layers), and
   the prepare-path substitution gap is latent until hoisting lands. Shipping only the
   Rust refactor would not fix the bug. Rejected: hoisting at merge sites (scattered
   producers); Rust-side param derivation (wrong boundary direction).
5. **Relocate param substitution into scale_resolve; plan slots once at prepare**
   (#72): follows the seam doc's one-way dependency and the `LegendBandPlan`
   precedent. Rejected: pub(crate)-exposing the scene_build fn (inverts stage
   dependency).
6. **Slot-aware hit-testing via indexed `y_slot` + per-candidate inverse** (#73):
   mirrors the zoom-inverse exemplar (22333c30) one level deeper. Rejected: per-slot
   query fan-out (N queries per pointer move); re-baking the index per rescale
   (violates static-index design).
7. **Implement GH #60 (slot tags on axis text) instead of relaxing `c >= 2`** (#73):
   the frequency threshold is load-bearing against stray-label false positives;
   relaxation trades one bug for another. The tag is the archaeology-planned proper
   fix and closes #60. Cost: a scene-contract addition (serde-defaulted, additive).
8. **Leaf-span union for color/size; keep positional pairing for x/y** (#74): the
   walk is positional by design for grids; the defect is channel-class-specific.
   Inheritance rule (explicit-wins, unset-inherits) is defined here because the
   design previously left nesting node-local. Rejected: raising on nested composites
   (feature→error, violates no-defer); Python-side resolve propagation (semantics
   duplicated across the boundary).
9. **`orient="none"` joins the legend-disabled path** (#74): one suppression
   mechanism, no new `LegendOrient` variant, single-chart semantics fixed for free.
10. **`.spec` flat-dict pass-through stays** (#74): deliberate identity back-compat;
    docstrings corrected to state the exception.

## 9. Acceptance criteria

1. All #71 red tests pass: HConcat/VConcat/LayerChart/deep-nesting shared-y conflicts
   raise; no `__rhs_` in SVG or scene JSON for heterogeneous dual-axis; primary-layer
   explicit tooltip does not leak; plus a new test: `SecondaryY(mark="boxplot")`
   raises naming primitives.
2. All #71 constraint pins stay green (x-shared no-raise, multi-y guard, first-member
   stacking, default shared-y overlay path untouched).
3. #72 red test passes (secondary-layer domainParam honored in `y_domains[1]`); new
   discriminating tests: static right-axis tick labels equal the substituted domain's
   ticks for a param-bound secondary layer (marks == ticks == y_domains[k]); a
   layer-declared param appears in the wire `params` exactly once when also declared
   chart-level.
4. Param-free goldens byte-identical; pre-52 layout byte-stability test green.
5. #73: displayed-position hit-test passes; stale-position mirror test inverted and
   passes; single-tick relabel passes; two-tick and composition controls stay green;
   axis-text slot tags round-trip through scene JSON; debug assert fires on
   out-of-range slot in test builds.
6. #74: the three nesting shapes render exactly one `grp` legend with a unioned
   domain; explicit-independent child under outer shared stays independent (new
   test); pairplot/compare grids and hole handling byte-stable;
   `orient="none"` suppresses band + panel legends on composites and suppresses
   legends on single charts (new test); `orient="bottom"` golden unchanged.
7. Full `uv run pytest` and `cargo test -p ferrum-core -p ferrum-wasm` green; ruff
   clean; clippy judged by delta against the ~166-warning baseline.
8. Design docs updated: #16 design doc + `ferrum-spec.md` §3.9 dated note (nested
   resolve rule, orient="none" semantics); archaeology doc rows SY-5/#60 closed,
   `__dodge_n_groups__` widened-surface back-link added; #52 §4 note for the
   spelling-independent conflict contract.

## 10. Validation strategy

Behavioral: the evidence tests are the before/after checks (committed RED at
7bd8ff50). New discriminating tests cover every contract decided here but not pinned
by an evidence test (ticks==marks equality, inheritance rule, orient="none" single
chart, SecondaryY mark guard, slot-tag round-trip). Byte-stability is validated by
the existing golden and pre-52 tests, not by re-blessing; any golden that must change
is rasterized and visually inspected per CLAUDE.md. Interactive behavior (hit-test
after rescale) is validated at the Rust unit level against composed affines — the
same layer the render path uses — with the headless-capture harness available as a
fallback if unit-level proof is disputed.

## 11. Open questions

None blocking. (Planner discretion: exact tooltip display-name handling for renamed
columns — title-only vs. tooltip-field label mapping — resolve during implementation
against the red test's assertions.)
