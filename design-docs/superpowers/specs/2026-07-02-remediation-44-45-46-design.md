# Remediation of Issues #44 / #45 / #46 — Design Spec

Batch remediation of three pre-existing bugs surfaced during #35 (compare= aggregate
diagnostics). Decisions were settled via coherent-change batch research and approved
2026-07-02. Origin issues: GH #44, #45, #46.

> **Amendment 2026-07-02 (user decision, same day):** #45's composition-layer sharing
> is NOT delivered by the Python `_scale_share` extension described in §4/§6/§8-2,3
> (that part of this spec is superseded and its task was dropped un-built). Instead the
> north-star — unifying concat/grid `resolve=` with the Rust facet-sharing mechanism
> via a Rust-side composite render path — ships this round as its own designed change
> ("Phase B", separate spec + branch, subsumes W4/W5). What remains of #45 in THIS
> spec: the scale-through-desugar propagation rule (§6), which fixes user-set explicit
> scales on composite-mark charts and is prerequisite plumbing for explicit-override
> behavior under Phase B. §3's "no unification" non-goal and the non-congruent-skip
> decision are void; congruence semantics are re-decided in the Phase B spec. #45 stays
> open until Phase B lands.

## 1. Scope

Three independent Python-only fixes: (#44) `cooks_distance_chart` on a non-linear model
raises a raw `IndexError` instead of a clear error; (#45) `resolve={...: "shared"}` in
composition is a silent no-op for children that are not flat Charts with top-level data
(box/composite-mark children and grid-composite children); (#46) `shap_waterfall_chart`
with `per_class=True` on a multiclass model chains all classes through one cumulative
sum (numerically wrong single panel) instead of the per-class faceting its docstring
promises. No Rust changes: #45 rides the existing Rust explicit-domain bypass
(`build_axis_scale` honors `EncodingSpec.scale` before data-driven domain inference).

## 2. Goals

- A no-hat-matrix estimator passed to a leverage-only diagnostic chart produces a
  clear, actionable `ValueError` — on the single-model path and, inherited through the
  shared builder, inside any `compare=` set.
- `_grid_panels` can never IndexError or silently truncate: panel counts outside its
  supported range fail loudly.
- A shared-scale resolve (`resolve=`, and the same helpers used by `compare=`,
  LayerChart, and RepeatChart) visibly takes effect for composite-mark children
  (box/strip/violin/letter-value/cv_scores, …) and for structurally congruent
  grid-composite children.
- A user-set explicit scale domain on a composite-mark chart's positional channel
  (e.g. `Y("score", scale=Scale(domain=...))`) survives desugaring and renders.
- `shap_waterfall_chart(per_class=True)` on a multiclass model renders one waterfall
  panel per class, each with a numerically correct per-class cumulative walk, on a
  shared shap-value x-axis; behavior and docstring agree.
- Every currently-correct rendering named in §7 stays byte-identical.

## 3. Non-goals

- No empty-state/placeholder panel subsystem (no precedent in ferrum; rejected for #44).
- No unification of concat-level `resolve=` with the Rust facet-sharing mechanism
  (`include_final` union path) — logged as a north-star follow-up.
- No `base_value`/`E[f(x)]` column in the SHAP schema; waterfalls remain zero-anchored
  per the existing `SCHEMA_SHAP_VALUES` contract — follow-up if ever wanted.
- No generalization of `_grid_panels` beyond 4 panels (guarded, not implemented).
- No sharing semantics for structurally non-congruent grid children (documented skip).

## 4. System behavior

### #44 — leverage-only charts on no-hat-matrix models

- `cooks_distance_chart(model, X, y)` where the resolved source yields all-NaN leverage
  (non-linear estimator, i.e. no `coef_`; or a precomputed source, which never carries
  leverage) raises `ValueError`. The message names the chart, states that Cook's
  distance / leverage are hat-matrix quantities requiring a linear estimator exposing
  `coef_`, and names the offending estimator type when a model is available.
- The same error surfaces from `residuals_chart(..., panels=["residuals_vs_leverage"])`
  (or any explicit panel list that empties after the leverage drop).
- `compare=` on these charts propagates the same `ValueError` (no per-model catch
  exists in `_compose_compare`; the error must identify which model failed).
- Multi-panel requests where at least one panel survives the leverage drop keep the
  current graceful degradation (e.g. `panels="auto"` on a non-linear model → 3 panels).

### #45 — shared scales for non-flat compose children

- `resolve={"y": "shared"}` (likewise `"x"`) over children that include composite-mark
  charts: the union domain is computed from each child's pre-desugar channel binding
  and data, injected onto the child's chart-level encoding, and now *renders*, because
  chart-level explicit positional scales are propagated through composite-mark
  desugaring onto the derived layer channels (see §6 propagation rule).
- The same applies to a scale the user sets directly on a composite-mark chart's
  positional channel: it takes effect instead of being silently dropped.
- `resolve=` over children that are themselves grid composites (HConcat/VConcat trees,
  e.g. multi-panel `residuals_chart` output composed by `compare=`): when all children
  are structurally congruent (same composite tree shape, leaves pairwise aligned),
  domains are unioned **per leaf position** across children and injected per position.
  Panels at different positions within one grid never share with each other (their
  axes have heterogeneous semantics).
- Children with non-congruent structures (or a mix of flat and composite children where
  no positional pairing exists) are skipped for that channel — same rendering as today,
  and the `resolve=` docstring states this boundary.

### #46 — per-class SHAP waterfall

- `shap_waterfall_chart(..., per_class=True)` on a multiclass model renders a faceted
  chart, one panel per `class_label`, faceting gated by the existing
  `_should_facet_by_class` predicate (per_class requested AND >1 class present).
- Feature selection/ordering stays **global** (unchanged `_shap_order_features`
  aggregation over all rows): every class panel shows the same top-`max_display`
  features in the same order.
- Each panel's cumulative walk (`x0`/`x1`) is computed within its class only, anchored
  at 0 (zero-anchoring is the existing schema contract).
- The x domain is the union of all classes' `x0`/`x1` extents (± existing padding) and
  is shared across panels (facet default resolution).
- `per_class=False`, and `per_class=True` on regression/binary (single class), render
  byte-identically to today.
- The mark-level data transform that re-filters to top features must keep exactly the
  globally-ranked feature set for every class (no class-blind row top-k that would drop
  some classes' rows).

## 5. Architecture

Unchanged. All fixes live in the Python declaration layer: diagnostic chart builders
(`plots/`), the composite-mark desugaring plumbing (`marks/`), and the scale-sharing
helpers (`_scale_share.py`) that all composition call sites funnel through. Rust
consumes the result through the already-existing explicit-domain path
(`EncodingSpec.scale` → `build_from_scale_spec`); the Rust facet-sharing mechanism is a
separate path and is untouched.

## 6. Canonical interfaces / data contracts

**`_grid_panels(charts, theme=None)`** — accepts 1–4 charts. 0 charts or >4 charts
raise `ValueError` naming the received count (internal invariant; public callers must
reject earlier with a domain-specific message).

**Leverage-drop rejection** — the predicate for "leverage unavailable" remains
`df["leverage"].is_nan().all()` (uniform across `ModelSource` and precomputed sources —
the latter has no `.capabilities`). Rejection fires only when the drop empties the
panel list.

**Scale-through-desugar propagation rule.** When a pending composite mark is resolved
into layers: for each positional channel (`x`, `y`) whose chart-level encoding carries
an explicit `scale`, and for each desugared layer channel derived from that positional
axis (including its `x2`/`y2` companion's axis):

- layer channel has **no** scale → attach the chart-level scale;
- layer channel has a scale **without** `domain` (e.g. `{"type": "log"}` in
  validation-curve, size-range scales are non-positional and out of scope) → merge in
  the `domain` only, never overwriting the layer's `type`/`range`/other keys;
- layer channel already has a `domain` → leave untouched (mark-computed domains such as
  shap `x_scale_domain` win).

Sweep basis: 26 registered composite marks (alpha_selection, calibration,
class_prediction_error, confusion, cv_scores, decision_boundary,
discrimination_threshold, gain, importance, intercluster_distance, learning_curve,
lift, parallel_coordinates, pca_scree, pdp, pr, prediction_error, rank1d, rank2d,
residuals, roc, shap_bar, shap_beeswarm, shap_waterfall, silhouette,
validation_curve). Only two desugar-level scale usages exist today: a positional log
`type` (merge case) and a non-positional size `range` (out of scope). No desugar sets a
positional `domain`, so the rule cannot conflict with any existing mark.

**Composite-aware `_scale_share`.** `compute_union_domain(charts, channel)` and
`inject_scale(chart, channel, scale_dict)` gain a composite-child shape:

- *Congruence*: all children being composed are composites of the same tree shape
  (same composite types and child counts at every level, leaves aligned by position).
- Union: for congruent children, `compute_union_domain` is applied recursively per
  leaf position across children; injection rebuilds each composite with per-position
  injected leaves (via the composite's own copy mechanism — composites have no
  `_clone`/`_layers`).
- Non-congruent (or unmatched shape mix): that channel's sharing is skipped for the
  whole group — identical to today's rendering.
- Flat children keep the existing single-domain union/injection semantics unchanged;
  all five existing call sites (ConcatChart, `_CompositeBase`, RepeatChart, LayerChart,
  `_compose_compare`) inherit both extensions solely through these two helpers.

**Per-class waterfall data contract.** `plot_df` handed to `mark_shap_waterfall`
carries per-row `x0`, `x1`, `shap_sign` where, within each `class_label`, rows are in
global feature-rank order and `x0[i] = cumulative sum of that class's shap_value up to
i-1` (0 for the class's first row), `x1[i] = x0[i] + shap_value[i]`. The `class_label`
column is present on every row so `facet(col="class_label")` partitions cleanly.

## 7. Invariants and constraints

- **Byte-identical renders** for: all linear/supported `cooks_distance_chart` uses;
  `residuals_chart` graceful multi-panel degradation; flat-chart `resolve=` sharing;
  ordinal scale sharing (order-preserving, per 0b6bd3f); facet scale resolution
  (pdp per-feature independent x specifically); composite marks rendered without any
  chart-level explicit positional scale; `shap_waterfall_chart` with `per_class=False`
  or single-class data; all existing goldens not deliberately regenerated.
- Errors are raised, never warned: no warn-fallback paths (project constraint).
- No global mutable state; no matplotlib; spec drift in `ferrum-spec.md` (if any
  surface described there is touched) gets a dated note.
- Any new or regenerated golden SVG must be rasterized and visually inspected via
  `regen_and_verify` before commit (CLAUDE.md goldens rule).
- Each fix carries regression coverage proven RED against pre-fix code.

## 8. Key decisions and tradeoffs

1. **#44 rejects with `ValueError` at the drop site**, not a capability pre-check in
   the public function (predicate duplication; precomputed sources lack
   `.capabilities`) and not an empty-state panel (no precedent, disproportionate,
   less actionable). `_grid_panels` guards are defense-in-depth, not the primary UX.
2. **#45 fixes desugar scale-dropping generically** rather than forcing early desugar
   (would reorder the render pipeline for every composed chart and still not fix
   user-set scales) or patching only `_compose_compare` (non-exhaustive, duplicates
   chokepoint logic) or switching injection to Rust coord-level overrides (forks the
   mechanism by chart shape, different padding semantics, no ordinal support). The
   user-visible corollary — explicit scales on composite marks finally working — is
   an intended feature of the fix, not a side effect.
3. **Grid sharing is position-wise, never a flat union** (heterogeneous panel
   semantics; flat union is the same failure class as #35's pdp x-collapse).
   Non-congruent structures: documented skip (user decision 2026-07-02) — real
   producers (`compare=`) always emit congruent children, and there is no semantically
   right union for mismatched trees.
4. **#46 implements the promised faceting** (docstring demotion would leave a
   documented parameter dead on one sibling and misdescribe numerically wrong output).
   Global feature ranking is retained (per-class ranking would break cross-panel
   comparability and perturb the `per_class=False` path). Shared union x across class
   panels (user decision 2026-07-02): x is the shap-value scale, legitimately
   comparable across classes — the beeswarm rationale, and the opposite of the pdp
   case. Zero-anchoring per class is the existing no-base-value schema contract.
5. **Implementation order #44 → #46 → #45** (isolated → family-local → shared
   infrastructure), so the widest-blast-radius change lands against a green suite.

## 9. Acceptance criteria

**#44**
- `cooks_distance_chart(RandomForestRegressor(...), X, y)` raises `ValueError` whose
  message names the hat-matrix/`coef_` requirement and the estimator type; no
  IndexError anywhere on the path.
- Same for `cooks_distance_chart(linear, X, y, compare={"rf": rf})` — the error
  identifies the offending compare member.
- `residuals_chart(rf, X, y, panels="auto")` still renders 3 panels (regression guard).
- `_grid_panels([])` and `_grid_panels([c]*5)` raise clear `ValueError`s.

**#45**
- `cv_scores_chart(m1, X, y, compare={"m2": m2})`: both rendered panels' y-axis tick
  extents equal the union domain (SVG-extent inspection, per the
  `test_facet_shared_extent` pattern) for a case where per-model domains differ.
- A hand-built concat of two box charts with `resolve={"y": "shared"}` shares the
  rendered y-axis; the same concat without `resolve=` renders as today.
- `Chart(df).mark_boxplot(...)` (or cv_scores) with an explicit y `scale` domain
  renders that domain.
- `residuals_chart` `compare=` (congruent grids): each leaf position pair shares its
  axis domain; panels at different positions within one grid do not.
- Non-congruent composition renders identically to today (no error, no sharing).
- Flat-chart sharing, ordinal sharing, and pdp compare= independent-x goldens/tests
  all pass unchanged.

**#46**
- Multiclass `per_class=True` waterfall: one panel per class; per-panel bar count =
  kept features; total bars = features × classes; each class's bars form a correct
  cumulative chain (x0 of first bar = 0; x0[i] = x1[i-1] within the class); panels
  share one x domain.
- `per_class=False` (multiclass) and `per_class=True` (binary/regression) outputs are
  byte-identical to pre-fix.
- Docstring and behavior agree.
- New per-class golden committed only after PNG inspection.
- Hardening: a discriminating test locks shared-x-across-class-facets for
  shap_beeswarm/shap_bar under `per_class=True` + `compare=`.

**Cross-cutting**
- Full `uv run pytest -n auto` green; `nox -s lint` clean.
- Each fix's regression test demonstrated RED on pre-fix code (stash protocol).
- Issues #44/#45/#46 closed with commit references after user confirmation.

## 10. Validation strategy

- Behavior-level pytest coverage per acceptance item above; rendered-axis assertions
  use SVG tick-extent inspection (existing `test_facet_shared_extent` helpers pattern)
  because auto-inferred domains exist only Rust-side.
- Byte-identity claims validated by the existing golden suite plus targeted
  before/after SVG comparison for the §7 invariant list.
- Visual validation: new/regenerated goldens rasterized and inspected
  (`regen_and_verify`).
- The three pinned reproductions from issue triage (scratchpad repro scripts) flip
  from failing/incorrect to passing/correct.

## 11. Open questions

None blocking. (Exact error-message wording and the congruence-check implementation
are plan-level choices bounded by §6.)
