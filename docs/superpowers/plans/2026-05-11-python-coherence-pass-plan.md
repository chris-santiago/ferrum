# Python coherence pass — refactor plan

**Date**: 2026-05-11
**Branch**: stacks on `fix/marks-and-composition-wiring` (same as Rust pass)
**Scope**: `src/ferrum/`. Rust crate untouched.
**Goal**: same as Rust pass — recover architectural cohesion before first release.
"No defer" carries over from Rust: every finding either lands in this pass or
gets explicitly downgraded to "not actually a finding."

This is **refactoring + opportunistic feature completion**: the Phase-10h
"deferred" stubs (multi-panel residuals, multi-class SHAP overlay) and the
F21 grid-axis-sharing follow-up are now in scope. Public API surface stays
stable except where called out below.

---

## Origins

This plan synthesizes four parallel subagent deep-reads dispatched 2026-05-11:

- `chart.py` (4443 LOC, 88 methods) — see report at `/tmp/python-review-chart.md` if archived; consolidated findings below.
- `figures.py` (1792 LOC, 23 functions).
- `composition.py` (944 LOC, 5 classes).
- `_diagnostics/` subtree (~5000 LOC across 14 files).

User direction on 2026-05-11:
1. "Just like with rust we will not defer any remediation" — every finding
   below gets remediated unless explicitly downgraded.
2. Concern about the three-API surface verdict in `_diagnostics/` led to a
   recalibration captured in the **API layering** section below.
3. Verification-before-execution discipline: medium-confidence findings
   get read end-to-end before being acted on.

---

## API layering — intentional, not accidental

The initial subagent report framed the `figures.<x>_chart` / `<X>Visualizer`
/ `_*_chart_from_source` triple as accidental over-abstraction. On
re-examination this is **wrong** — it's intentional dual-API layering with
one broken layer underneath:

- **Internal builders** (`_*_chart_from_source`) are the engine. Single
  source of truth for chart structure. Private, called only from the two
  public surfaces. Correct factoring.
- **Functional API** (`figures.<x>_chart`) matches `ferrum-spec.md §3.14`
  and the matplotlib/seaborn/altair convention. Adds `_resolve_source`
  polymorphism + keyword-only ergonomics. Correct factoring.
- **Class-based API** (`<X>Visualizer`) provides yellowbrick /
  `sklearn.inspection.*Display` parity (`.fit(X, y).show()`). Holds state,
  exposes headline metrics. The **intent is correct**; the **execution**
  is broken — the base contract (`fit → _materialize → _build_chart`) is
  too narrow, 8 of 26 visualizers override `fit` entirely, and
  `ClassBalanceVisualizer` raises `NotImplementedError` from required
  protocol methods (direct CLAUDE.md rule violation).

**Resolution**: keep the dual public surface. Fix the visualizer base
class hierarchy. Document the layering. Drop the dead protocol bits
(`has_score`, `score()`). Drop the `NotImplementedError` stubs.

---

## Findings inventory (with confidence)

50 findings consolidated from the subagents. Confidence reflects whether
a finding is unambiguous drift (HIGH) or could be intentional design
masquerading as drift (MEDIUM/LOW) — medium items get a read-through
before being refactored.

### chart.py (`Chart`, 4443 LOC)

| # | Severity / Conf | Finding |
|---|---|---|
| C1 | S1 / HIGH | 40+ near-identical composite/diagnostic `mark_*` methods. ~1500 LOC of scaffolding. Extract `_set_composite_mark` helper. |
| C2 | S1 / MED | `_resolve_pending` straddles 2-tuple legacy form (density/histogram/smooth) + 3-tuple generic. Could be intentional protocol evolution — verify desugar functions before merging. |
| C3 | S2 / MED | Phase-10 polars data prep leaks into `mark_*` (mark_pca_scree 139 LOC, mark_silhouette, mark_residuals, etc.). Could be intentional standalone-callable design — grep tests for `chart.mark_residuals()` without figure-level wrapper before moving. |
| C4 | S2 / HIGH | 40+ duplicated `validate_position_eligibility` guards. Folded into C1 helper. |
| C5 | S2 / HIGH | Dict-shaped domain values: `_facet`, `_pending_stat_mark`, layer dicts. Layer-dict has documented legacy `mark_kwargs` vs `mark_style` dual-key handling — drift proves the shape isn't pinned. Frozen dataclasses. |
| C6 | S3 / HIGH | `Chart.__add__` (107 LOC) interleaves 4 concerns. Split into helpers. |
| C7 | S3 / LOW | `to_spec` channel allow-list at L4131 drops 10+ registered channels (stroke/fill/tooltip/etc.). Could be intentional renderer-capability gate — investigate before changing. |
| C8 | S3 / MED | Inconsistent NotImplementedError story across `mark_arc`/`mark_function`/`add_selection`/`interactive`. Could be intentional category distinction (deferred mark vs Phase-11 feature vs render-time restriction) — verify before unifying. |
| C9 | S3 / HIGH | `_repr_svg_` / `_repr_html_` swallow all exceptions silently. Log at DEBUG. |
| C10 | S3 / HIGH | `show_svg` / `show_png` duplicate viewport+theme wiring. Extract `_render_inputs`. |
| C11 | S3 / HIGH | `mark_segment` wedged inside "deferred stubs" section, lacks NumPy doc. |
| C12 | S4 / HIGH | `_SpecView` belongs in its own module. |
| C13 | S4 / HIGH | Module-level mutable dict `_CHANNEL_CLASSES_BY_NAME` — `functools.cache` cleaner. |

### figures.py (Phase-10 facade, 1792 LOC)

| # | Severity / Conf | Finding |
|---|---|---|
| F1 | S2 / HIGH | Late imports systematic across all 23 functions. Resolve import cycle once. |
| F2 | S2 / HIGH | `cluster_diagnostics` (134 LOC) violates facade contract — inlines sklearn fit loops + polars assembly. Move body to `_diagnostics/charts.py`. |
| F3 | S2 / MED | `_resolve_source` 4-way dispatch with dict-positional path. Could be intentional ergonomic surface — verify in tests/examples before requiring explicit `compare=`. |
| F4 | S3 / HIGH | `rank_chart` mode-flag creep (1d/2d × ModelSource/raw × algorithm × X). **Visualizer side already splits** (Rank1D / Rank2D) — function side should match. |
| F5 | S3 / HIGH | `shap_chart` hand-rolled `kind` dispatcher. Paired with D9 split. |
| F6 | S3 / MED | `calibration_chart` uses `*model_or_sources` variadic, peers use `compare=`. Could be intentional yellowbrick-parity choice. Verify and pick one idiom. |
| F7 | S3 / HIGH | Validation duplication — `_require(name, value)` extraction. |
| F8 | S3 / MED | `decision_boundary_chart` parameter naming drift (`model:` vs `model_or_source:`). Could be technical reason (grid prediction). Verify. |
| F9 | S4 / HIGH | `intercluster_distance_chart` reaches into `source._model`. Promote to public property. |
| F10 | S4 / HIGH | Phase-10h stubs: "panels='auto' ships single residuals panel", "Multi-panel layout reserved for Phase 10h". `_residuals_panel` infrastructure exists; wire `qq`/`scale_location`/`residuals_vs_leverage` to `panels="auto"`. |

### composition.py (5 wrapper classes, 944 LOC)

| # | Severity / Conf | Finding |
|---|---|---|
| K1 | S4 / LOW | `ClusterMapChart.show_svg` comment says "Compositor ignores row_ratios" — **stale post-F20**. Actual code is correct (pre-resize matches ratios exactly, F20 algorithm produces byte-identical output, intentionally kept to preserve dendrogram tree topology). Update comment only. |
| K2 | S2 / HIGH | `save()` duplicated 5× with error-message drift. |
| K3 | S2 / HIGH | `show` / `_repr_svg_` / `show_png` / `__repr__` boilerplate × 5. No `_repr_mimebundle_`. |
| K4 | S3 / MED | `theme()` ownership inconsistent — 3 of 5 classes. Could be intentional (HConcat/VConcat compose pre-themed charts). Verify. |
| K5 | S3 / HIGH | `_theme` slot is write-only — dead state. |
| K6 | S4 / HIGH | `charts` property drift across 5 classes (list attr, property, absent; ClusterMap order reverses `__init__`). |
| K7 | S3 / MED | `spec` dead-code properties on Joint/Repeat/ClusterMap. Grep external usage before deleting. |
| K8 | S3 / HIGH | `RepeatChart.resolve` / `.columns` / `.layer` dormant kwargs. Per "no defer": implement or remove. |
| K9 | S3 / LOW | `spacing` unit drift (pixels for HConcat/VConcat, fraction for grid). Could be intentional. Verify both paths' downstream consumers. |
| K10 | S3 / LOW | `align` hard-coded in HConcat/VConcat. Could be intentional default. |
| K11 | S4 / MED | `properties()` only on JointChart. Should be on all 5. |
| K12 | S4 / LOW | `__repr__` styles drift. |
| K13 | S4 / HIGH | Dead module-level prose (L7–22). |
| K14 | S4 / LOW | `RepeatChart.expand` silent diagonal demotion (warn → ValueError). |
| K15 | S5 / MED | `ImportError → NotImplementedError` wrapping with misleading message. |
| K16 | NEW / HIGH | F21 follow-up: implement `Chart.share_scale(other, channel)` + `Figure.shared(...)`. Captures `docs/superpowers/followups/2026-05-11-grid-axis-sharing.md`. |

### _diagnostics/ subtree (~5000 LOC)

| # | Severity / Conf | Finding |
|---|---|---|
| D1 | S2 / HIGH | `ModelSource` god class (1508 LOC, 22 methods, 7 domains). Domain headers already mark the seams (`# --- 10a/b/c/d/e/f/g ---`). Split into per-domain mixins. |
| D2 | S2 / HIGH | Validation duplication (3 idioms). Unify to `_require_capability`. |
| D3 | S2 / HIGH | `_cache_key` non-hashable bug surface. Real bug. |
| D4 | RECLASSIFIED | "Three-API surface" — see API layering section above. Keep the dual surface, document it; the real issue is D5/D6 below. |
| D5 | S3 / HIGH | Visualizer-contract violations (8 of 26 override `fit`). Restructure base class. |
| D6 | S3 / HIGH | `ClassBalanceVisualizer` raises `NotImplementedError` from required protocol methods — direct CLAUDE.md rule violation. |
| D7 | S3 / HIGH | Long methods (`discrimination_threshold` 90, `importances` 89, `partial_dependence` 85, `pr_curve` 83, `_decision_boundary_chart_from_source` 143, `_parallel_coords_chart_from_dataframe` 111, `_pdp_chart_from_source` 102). |
| D8 | DOWNGRADED (2026-05-12) | 3 inject-decorator-column idioms in charts.py. **Verified no factorable unification exists.** See P3.7 dispositional note below. |
| D9 | S3 / MED | Mode-flag creep. **Per-case judgment**: SHAP yes (different data per kind, sibling visualizers exist). `cv_scores(kind in {box,bar,strip})` is presentational choice over the same data — legitimate API. Verify each case before splitting. |
| D10 | S3 / MED | Int64-as-Utf8 defensive casts. F16 widened color inference in Rust; may now be obsolete. Test path-by-path before dropping. |
| D11 | S3 / HIGH | `ComparedModelSource._COMPARED_METHODS` manual registration. Drive from D1's domain split. |
| D12 | S4 / HIGH | Builders reach into `source._X` / `_model` / `_y` / `_capabilities` / `_feature_names`. Promote to public properties. |
| D13 | S4 / HIGH | `ManifoldVisualizer._cached_embedding` hidden state. Move to `ModelSource._cache`. |
| D14 | S4 / MED | `schemas.py` (184 LOC) documents but enforces nothing. Verify enforcement path before deleting. |
| D15 | NEW / HIGH | Multi-class SHAP overlay — Phase-10h stub. Per no-defer: implement. |

**Total: 50 findings.** Of these:
- ~35 HIGH confidence → execute directly when scheduled.
- ~12 MEDIUM confidence → read end-to-end before touching.
- ~3 LOW confidence (K1 already known-wrong; K9, K10) → likely no-action.
- 2 NEW features (K16 axis sharing, D15 multi-class SHAP).
- 1 RECLASSIFIED (D4 — keep the layering).

---

## Pre-flight verification — RESULTS (2026-05-11)

Four parallel subagent verifications complete. Six of twelve medium-confidence
items confirmed INTENTIONAL per `ferrum-spec.md`; refactoring them would
force spec changes:

| Item | Verdict | Plan change |
|---|---|---|
| C2 | DRIFT | P1.1 keeps |
| **C3** | **INTENTIONAL** (spec §3.3 L409–438) | **P1.2 dropped** |
| C7 | INTENTIONAL but spec L315 promises a warn-once that isn't emitted | P1.5 reshaped: emit the missing `UserWarning`, keep allow-list |
| C8 | INTENTIONAL three categories; but `add_selection` raise contradicts spec L734 | P1.6 reshaped: loosen `add_selection`/`interactive` to spec-conformant silent-ignore |
| **F3** | **INTENTIONAL** (spec L947–953) | **P4.3 partially dropped** (keep `_resolve_source` 4-way) |
| **F6** | **INTENTIONAL** (spec L1045–1046) | **P4.3 dropped for calibration** |
| **F8** | **INTENTIONAL** (spec L1078–1082, sklearn parallel) | **P4.7 dropped** |
| **K4** | **DRIFT** (spec L810: "All compound views accept `.theme()`") | P2.3 keeps |
| K7 | DRIFT (zero refs) | P2.5 keeps |
| K9 | DRIFT + ACTIVE BUG (spec L806 says fraction, Rust says pixels) | P2.6 reshaped — needs user decision |
| K10 | MIXED (spec doesn't commit) | P2.6 docs-only, surfaces to user |
| D9a | INTENTIONAL (same schema) | P3.6 narrowed |
| D9b | INTENTIONAL | P3.6 narrowed |
| D9c | INTENTIONAL per spec, but upstream `shap` lib splits | P3.6 — user decision |
| D10 | DOWNGRADED (2026-05-12) — pre-flight claim was inverted. See P3.9 note below. | P3.9 closed |
| D14 | DRIFT (zero refs) | needs user decision |

**Net scope change**: ~5 commits dropped, 3 small bug-fix commits added,
1 elevated to active-bug status. Plan now ~28 commits.

## Pre-flight verification pass

Before Tier 0 starts, run a focused read-through on the **12 medium-confidence
items** to confirm drift vs intentional design. Each verification is
~5–10 minutes — cheap, decisive, prevents mid-refactor recalibration like
the API-layering miss.

Items to verify (mark VERIFIED / DROPPED / RECLASSIFIED before scheduling):

- C2: `_resolve_pending` dual protocol — verify desugar function signatures.
- C3: Phase-10 polars data prep — grep tests for standalone `mark_residuals()` use.
- C7: `to_spec` channel allow-list — check renderer-side which channels it actually reads.
- C8: NotImplementedError variations — verify whether the 3 categories are documented.
- F3: `_resolve_source` dict-positional — check tests/examples for the dict form.
- F6: `calibration_chart` variadic — check whether yellowbrick parity is the reason.
- F8: `decision_boundary_chart` naming — read the function body for the technical constraint.
- K4: `theme()` ownership — verify HConcat/VConcat have other reasons for omitting `theme()`.
- K7: `spec` properties — grep tests/docs/examples for `chart.spec` access.
- K9: `spacing` unit drift — read both paths' downstream consumers.
- K10: `align` hard-coded — check whether tests exercise alternative alignments.
- D9: per-flag judgment for `pdp_chart` (kind), `cv_scores` (kind), `SHAPVisualizer` (kind).
- D10: Int64 defensive casts — run a test removing each cast in isolation.
- D14: schemas.py — search for any `pl.Schema` usage that references the constants.

---

## Refactor plan (tiered)

Same shape as the Rust pass. ~30 commits. Goldens byte-identical except
where flagged. Public-API changes are explicit in each tier.

### Tier 0 — Foundation (safe to start before pre-flight verification)

| # | Finding | Commits |
|---|---|---|
| P0.1 | **C1** — Extract `Chart._set_composite_mark(name, desugar_fn, kwargs, *, placeholder, position)`. Migrate 40+ mark methods through it. Folds C4. ~1500 LOC removed. | 1 |
| P0.2 | **K13** — Drop dead module-level prose in composition.py. | 1 |
| P0.3 | **C12** — Move `_SpecView` to `_spec_view.py`. | 1 |
| P0.4 | **D2** — Unify `ModelSource` validation idioms (`_require_capability`). | 1 |

### Tier 1 — chart.py structural cleanup (after pre-flight)

| # | Finding | Commits |
|---|---|---|
| P1.1 | C2 — Collapse `_resolve_pending` protocol shapes (if verified as accidental). | 1 |
| P1.2 | C3 — Move Phase-10 polars data prep to figure-level builders (if standalone-callable design is not intentional). | 2 |
| P1.3 | C5 — Frozen dataclasses for `_facet`, `_pending_stat_mark`, layer shapes. | 2 |
| P1.4 | C6 — Split `Chart.__add__`. | 1 |
| P1.5 | C7 — Investigate `to_spec` channel allow-list. May or may not edit. | 1 |
| P1.6 | C8 — Unify NotImplementedError if categories aren't intentional. | 1 |
| P1.7 | C9 / C10 / C11 / C13 — small cleanups. | 1 |

### Tier 2 — Composition cleanup

| # | Finding | Commits |
|---|---|---|
| P2.1 | K1 — Update stale comment in `ClusterMapChart.show_svg`. | 1 |
| P2.2 | K2 / K3 / K11 / K12 / K15 — `_ChartLike` base providing save/show/_repr_svg_/_repr_mimebundle_/show_png template-method. Promotes `properties()` to all 5. | 1 |
| P2.3 | K4 / K5 — `_GridComposition` base (if K4 is real drift). Read `_theme` consistently. | 1 |
| P2.4 | K6 — Standardize `charts` accessor across all 5 classes. | 1 |
| P2.5 | K7 / K8 — Drop dead `spec` properties (if grep confirms unused). Implement or remove `RepeatChart.resolve` / `.columns` / `.layer`. | 1 |
| P2.6 | K9 / K10 — Normalize spacing units. Expose `align=` (if verification shows demand). | 1 |
| P2.7 | K14 — Diagonal demotion warn → ValueError. | (folded with P2.5) |
| P2.8 | K16 — Implement `Chart.share_scale(other, channel)` + `Figure.shared(...)`. F21 follow-up. **Public API addition.** | 2 |

### Tier 3 — _diagnostics restructure

| # | Finding | Commits |
|---|---|---|
| P3.1 | D1 — Split `ModelSource` along domain headers into 5 mixins on `BaseSource`. | 3 |
| P3.2 | D3 — Fix `_cache_key` non-hashable handling. | 1 |
| P3.3 | D12 — Promote `source._X` / `_model` / `_y` / `_capabilities` / `_feature_names` to public properties. | 1 |
| P3.4 | D11 — Drive `ComparedModelSource` proxying from D1's domain mixins. | 1 |
| P3.5 | D5 / D6 — Restructure `FerrumVisualizer` base into mixins; drop ClassBalance `NotImplementedError`. | 1 |
| P3.6 | D9 (SHAP) — Split into `SHAPBeeswarmVisualizer` / `SHAPBarVisualizer` / `SHAPWaterfallVisualizer`. **Public API addition; old `SHAPVisualizer(kind=...)` retained as shim.** | 1 |
| P3.7 | D8 — Unify 3 inject-decorator-column helpers in charts.py. **DOWNGRADED 2026-05-12: no factorable unification — see note.** | 0 |
| P3.8 | D7 — Split long methods. | 2 |
| P3.9 | D10 — Drop Int64-as-Utf8 casts where F16 obviates them. **DOWNGRADED 2026-05-12: F16 inverted the premise; casts are more necessary post-F16, not less. See note below.** | 0 |
| P3.10 | D13 — Move `ManifoldVisualizer._cached_embedding` to `ModelSource._cache`. | 1 |
| P3.11 | D15 — Multi-class SHAP overlay (Phase-10h stub). | 2 |
| P3.12 | F10 — Multi-panel residuals (Phase-10h stub). | 1 |
| P3.13 | D4 — Document the dual-API layering in module docstrings. | 1 |

### Tier 4 — figures.py polish

| # | Finding | Commits |
|---|---|---|
| P4.1 | F1 — Resolve import cycle; remove 23 late-import sites. | 1 |
| P4.2 | F2 — Move `cluster_diagnostics` body to `_diagnostics/charts.py`. | 1 |
| P4.3 | F3 / F6 — Simplify `_resolve_source`, align `calibration_chart` to `compare=` (if verified as drift). **Possible public-API change.** | 1 |
| P4.4 | F4 — Split `rank_chart` into `rank1d_chart` / `rank2d_chart`. **Public API addition; old `rank_chart` shim retained.** | 1 |
| P4.5 | F5 — Replace `shap_chart` if-elif dispatcher (folded with P3.6). | (folded) |
| P4.6 | F7 — Extract `_require(name, value)`. | 1 |
| P4.7 | F8 — Align `decision_boundary_chart` naming (if verification doesn't reveal a technical reason). | 1 |
| P4.8 | F9 — Promote `intercluster_distance_chart`'s `source._model` access (folded with D12). | (folded) |

---

### P3.9 (D10) — disposition note (2026-05-12)

Pre-flight verification claimed "3 obsolete post-F16, 2 still required."
**Closer reading of primary sources contradicts the obsolescence claim.**

Sources consulted:

- `ferrum-spec.md:420` (F16 audit note): "continuous color is selected
  when ... `type=` is `None` and the column dtype is numeric (any width:
  Float32/64, Int8/16/32/64, UInt8/16/32/64) or temporal".
- `crates/ferrum-core/src/render/scale_resolve.rs:701-721`: Pre-F16 the
  continuous-vs-categorical decision was a narrow dtype check
  (`matches!(dtype, Float64 | UInt64)`); every other numeric dtype
  (including Int64) silently fell into the **categorical** branch. F16
  widened this so **all** numeric dtypes — including Int64 — route to
  **continuous** color when no explicit `type=` is set.
- `src/ferrum/_diagnostics/schemas.py:149-152, 166-170`:
  `SCHEMA_SILHOUETTE.cluster` and `SCHEMA_INTERCLUSTER_DISTANCE.cluster`
  are documented as `pl.Utf8`.
- `tests/diagnostics/test_clustering.py:30` enforces
  `sil["cluster"].dtype == pl.Utf8`.

Empirically: removing the three candidate Utf8 casts
(`_clustering.py:64`, `_clustering.py:206`, `charts.py:1444`)
immediately breaks `tests/diagnostics/test_clustering.py` because the
documented schema is Utf8, and per F16 the underlying Int64 would route
to continuous color (a 1-D viridis gradient over cluster IDs) — the
opposite of what cluster diagnostics need.

The other 2 cast sites in the chart builders
(`charts.py:966`, `:1425`) are sample_id-for-`mark_style.detail`
grouping, not color casts; `charts.py:1438` casts `feature` back to
Utf8 after an `Enum` cast for ordinal x-scale ordering. None of these
are defensive color casts.

**Verdict:** downgrade. The casts must remain. No commit; D10 closed.

---

### P3.7 (D8) — disposition note (2026-05-12)

Closer reading of `src/ferrum/_diagnostics/charts.py` found **no factorable
unification** across the four `_inject_*` helpers:

- `_inject_constant` (L19) — adds 1 column, first-row anchor, `pl.Series` literal.
- `_inject_cook_outliers` (L35) — adds 2 columns, polars `when().then().otherwise()`
  predicate.
- `_inject_curve_annotation` (L301) — adds 3 columns, one-per-group via
  Python lists.
- `_inject_pr_iso_lines` (L413) — appends *rows*, not columns (different
  category entirely).

The first three differ on every dimension that matters: selection mechanism
(first-row / predicate / group-iteration), output shape (1 / 2 / 3 columns),
and construction style. The signatures are already consistent
(`(df, *, kwargs) -> df`). Any unified helper would be a 3-branch dispatcher
that obscures the per-site polars expressions — net negative LOC, net
negative readability.

**Verdict:** downgrade. No commit; D8 closed.

---

## Validation strategy

Standard cadence per commit (same as Rust pass):

1. `unset CONDA_PREFIX && uv run --no-sync maturin develop` (Rust untouched on most commits — fast).
2. `cargo test --lib` (should stay 543/543).
3. `uv run pytest tests/ -x -q --no-header`.
4. **Golden hash sweep** before/after each commit:
   ```bash
   find tests/goldens tests/test_phase_9_e2e/goldens -name '*.svg' \
     | sort | xargs sha256sum > /tmp/goldens-pre.txt
   # … commit ...
   find tests/goldens tests/test_phase_9_e2e/goldens -name '*.svg' \
     | sort | xargs sha256sum > /tmp/goldens-post.txt
   diff /tmp/goldens-pre.txt /tmp/goldens-post.txt
   ```
   Empty diff = pass. Goldens shift only on deliberate behavior commits:
   P1.5, P3.9 (Int64 casts removal), P3.11 (multi-class SHAP),
   P3.12 (multi-panel residuals), K16 (axis sharing — new feature, may
   shift some grid compositions).
5. For each commit that intentionally changes SVG output: regenerate via
   `FERRUM_UPDATE_GOLDENS=1 uv run pytest tests/<affected>` and **visually
   inspect each PNG** via `scripts/snapshot-goldens.py` per CLAUDE.md.

**Pre-flight test inventory** (run once before starting):

- `pytest tests/` confirms 984/984 baseline.
- Grep tests for prose assertions that might break under refactor:
  ```
  rg -n 'mark_kwargs|mark_style|panels="auto"|share_x|share_y|NotImplementedError' tests/
  ```
- Grep tests/examples for `chart.spec` accesses that the K7 deletion might break.
- Grep tests for standalone `chart.mark_<diagnostic>()` calls (without figure-level wrapper) that C3 might break.

**After F2** (P3.1's domain split): `cargo expand` not applicable (Python),
but run a `pyright` / `mypy` pass if available since the type structure shifts.

---

## Public API decisions (require sign-off before execution)

These commits change the public Python API. None break existing user code
when shipped with the shims noted; but they add new names:

1. **P2.8** — `Chart.share_scale(other, channel)` + `Figure.shared(...)` —
   net addition.
2. **P3.6** — `SHAPBeeswarmVisualizer` / `SHAPBarVisualizer` /
   `SHAPWaterfallVisualizer` added; old `SHAPVisualizer(kind=...)` retained
   as deprecation shim.
3. **P4.4** — `rank1d_chart` / `rank2d_chart` added; old `rank_chart(rank=...)`
   retained as deprecation shim.

Implicit behavior changes (no new names, but observable output shifts):

4. **P3.9** — Int64 cluster IDs may now route to continuous color
   (post-F16) instead of being defensively cast to Utf8. Goldens
   re-blessed where affected.
5. **P3.11** — multi-class SHAP overlay (new chart shape).
6. **P3.12** — `panels="auto"` on residuals_chart now ships QQ +
   scale-location + leverage panels instead of single residuals panel.
7. **K16** — `Figure.shared()` may produce different axis ticks/ranges
   on grid-composed charts that use it.

---

## Decisions confirmed by user (2026-05-11)

1. **Branch**: same as Rust pass — stack on `fix/marks-and-composition-wiring`.
2. **No defer**: every finding either lands or gets explicitly downgraded.
   Includes the F21 grid-axis-sharing follow-up and the in-source
   "Phase 10h follow-up" stubs.
3. **Confidence tagging**: medium-confidence items got a pre-flight
   verification read; results integrated above.
4. **API layering**: keep the dual functional + class-based surface.
   The visualizer base class restructure is the real fix.

### Post-verification decisions

5. **K9 spacing units**: update spec to pixels. Bump
   `JointChart` / `RepeatChart` / `ClusterMapChart` `spacing` defaults
   from `0.02` (fractional, never honored) to a visible pixel value
   matching `HConcatChart` / `VConcatChart` (default `10.0`).
   `ferrum-spec.md` L806 gets a dated note. Affected goldens
   re-blessed.
6. **D9c SHAP API**: split. Add `shap_beeswarm_chart` /
   `shap_bar_chart` / `shap_waterfall_chart` + matching `SHAP*Visualizer`
   classes. Old `shap_chart(kind=...)` / `SHAPVisualizer(kind=...)`
   retained as deprecation shims. Spec §3.14 + §3.15 get a dated note.
7. **D14 schemas**: enforce at builder boundaries. Add
   `assert df.schema == SCHEMA_X` (or `df.schema.matches(SCHEMA_X)`)
   at the end of each `ModelSource` builder method whose schema is
   documented.
8. **Bug bundling**: focused commits per surfaced bug. C7-warn,
   C8-conformance, K9-impl each get their own `fix(...)` commit.

---

## Out of scope

- Rust crate (`crates/ferrum-core/`) — already remediated.
- Build / packaging.
- Test infrastructure beyond opportunistic test additions during refactors.
- Wholesale rewrite — refactors are incremental, each with golden-hash
  verification.

---

## Estimated effort

~30 commits across 4 tiers + a pre-flight verification pass:

- **Pre-flight** (12 medium-confidence verifications): 1–2 hours.
- **Tier 0** (4 commits): high-leverage, safe to start before pre-flight.
- **Tier 1** (~9 commits): chart.py structural cleanup.
- **Tier 2** (~8 commits): composition.py cleanup including F21 axis sharing.
- **Tier 3** (~14 commits): `_diagnostics/` restructure including
  Phase-10h feature completion.
- **Tier 4** (~6 commits): figures.py polish.

Execution discipline same as Rust pass: one commit at a time, full
validation between commits, summarize between tiers.

Phase-3 first-patch proposal already drafted in conversation
(P0.1 `Chart._set_composite_mark` helper). Awaiting approval to start.
