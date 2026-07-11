# Dodge-by-Model `compare=` Layout Design Spec (GH #42)

**Date:** 2026-07-10
**Issue:** #42 — selective dodge-by-model layout for grouped-bar/box `compare=` diagnostics
**Predecessor:** #35 small-multiples compare rendering (`design-docs/superpowers/specs/2026-06-27-compare-aggregate-diagnostics-design.md`)
**Defended choice:** approved via coherent-change, 2026-07-10 (scope includes `shap_bar_chart` by user decision).

## 1. Scope

Three `compare=` diagnostics whose mark is dodge-eligible and whose grouping
structure admits a model dimension switch from small-multiples panels to a
single shared-axis panel with marks dodged by model: `importance_chart`,
`shap_bar_chart` (`per_class=False`), and `cv_scores_chart` (`kind="box"` /
`"strip"`). Supporting work: the `text` mark renderer consumes position
offsets, and the dodge eligibility set admits the affected composite marks.
All other `compare=` diagnostics keep the #35 small-multiples layout.

## 2. Goals

- `importance_chart(..., compare={...})` renders one panel of grouped bars
  dodged by model, with a model legend, in both orientations, with error bars
  and value labels dodging alongside their bars.
- `shap_bar_chart(..., compare={...}, per_class=False)` renders one panel of
  mean-|SHAP| bars dodged by model.
- `cv_scores_chart(..., compare={...}, kind="box"|"strip")` renders one panel
  with `split` on the categorical axis and per-model marks dodged within each
  split band.
- Single-model output (`compare=None` or omitted) stays byte-identical.
- Every other `compare=` diagnostic stays byte-identical.
- The cv_scores layout decision (single-dodge vs facet-by-split) is recorded
  (§8, D3).

## 3. Non-goals

- Dodge layouts for any other compare diagnostic. `pca_scree_chart` carries a
  line layer (not dodge-eligible); `silhouette_chart` has a competing
  per-cluster dimension; the curve/scatter/grid diagnostics have no
  categorical band axis.
- `cv_scores_chart(kind="bar")` dodge (§8, D3 — stays small multiples).
- `shap_bar_chart(per_class=True)` dodge (competing class-facet dimension —
  stays small multiples).
- Rust ordinal-**y** dodge. Horizontal layouts use the established
  normalize-to-x + `CoordFlip` idiom.
- Position-offset consumption in the direct-label subsystem (`label` mark) —
  it has its own placement engine.
- Any change to `_compose_compare` or to the compare gating/validation
  contract in `_resolve_source`.

## 4. System behavior

**importance_chart + compare.** Per-model importances are computed with the
same `method`/`random_state`, concatenated with a `model: Utf8` column
(§6 schema). Features are ranked **once, globally across models** by mean
importance descending; the top-`top_k` features form the shared categorical
axis, in that order, for every model (mirrors shap_bar's cross-class
global-feature-set principle). Each feature band contains one bar per model,
colored by model, in compare registration order (`"base"` first). With
`error_bars=True` the per-model error rules dodge to their bar's sub-band.
With `show_values=True` the value labels dodge to their bar's sub-band.
`orient="horizontal"` (default) renders horizontal grouped bars;
`orient="vertical"` renders vertical grouped bars. Return type is `Chart`
(previously `ConcatChart`).

**shap_bar_chart + compare, per_class=False.** Per-model SHAP aggregation as
today (same `order`/`max_display`/`background`), concatenated with `model`.
Feature ranking follows the existing `_shap_order_features` principle
extended across models: rank once over the pooled per-model SHAP values, keep
one top-`max_display` feature set for all models. One panel, bars dodged by
model, model legend. Return type `Chart`. With `per_class=True`, compare
keeps small multiples (unchanged).

**cv_scores_chart + compare.** Per-model CV scores concatenated with `model`.
`kind="box"` (default): categorical axis stays `split` (`train`/`test`
filtered by `split=` exactly as today); within each split band one box per
model, colored by model. `kind="strip"`: same layout with points; the
single-model jitter is dropped under dodge (position adjustments are not
composable — same rule catplot records). `kind="bar"` keeps the #35 small
multiples (three grouping dimensions: fold × split × model). Return type
`Chart` for box/strip, `ConcatChart` for bar.

**Empty/degenerate compare.** `compare=` normalization (`_resolve_source`,
`ModelSource.compare`) is untouched; whatever it accepts today it accepts
after. A compare that yields one model renders a single-group dodge (a no-op
offset) — one ordinary panel with a one-entry legend.

**Text mark under dodge.** The `text` renderer honors per-row position
offsets the way its sibling renderers do. When no offset columns are present,
text output is unchanged. `mark_text(position=fm.Dodge(...))` becomes legal
at the public API (previously `TypeError`).

## 5. Architecture

Unchanged division of labor: Python builders reshape source data and declare
the chart (combined DataFrame + `color="model"` + `position=Dodge(by="model")`);
Rust `position.rs` remains the single authority that computes dodge offsets;
mark renderers consume them. No transform logic moves to Python. No new Rust
subsystem — the only Rust change is offset consumption in the text renderer.
The dodge path reuses the mechanism shipped for catplot/histogram; the
model-column stamping mirrors the overlay diagnostics (roc/pr) compare
convention.

## 6. Canonical interfaces / data contracts

No public signature changes. The behavior change is the rendered layout and
return type of the three functions under `compare=`.

Combined-DataFrame schemas (the seam between source reshaping and chart
declaration; `model` is the stamped compare-name column, values in
registration order, `"base"` first):

- importance: `feature: Utf8, importance: Float64, std: Float64,
  imp_lower: Float64, imp_upper: Float64, model: Utf8` — rows ordered
  (feature-rank, model-registration); exactly `top_k` distinct features.
- shap_bar: `feature: Utf8, abs_mean_shap: Float64, model: Utf8` — one
  global feature set of ≤ `max_display` features shared by all models.
- cv_scores: `fold: Int, split: Utf8, score: Float64, model: Utf8`.

Dodge eligibility (`_DODGE_ELIGIBLE`) additionally admits: `importance`,
`shap_bar`, `cv_scores` (composite marks that desugar to dodge-consuming
primitive layers, same footing as `histogram`/`density`) and `text`.

Value-axis domain rule under compare: computed over the **combined**
DataFrame with the same formula the single-model builder uses (zero-anchored,
5% headroom past the max upper bound).

## 7. Invariants and constraints

- Single-model output of all three charts is byte-identical to today.
- Every other diagnostic's `compare=` output is byte-identical to today.
- Existing dodged charts (catplot, histogram `multiple="dodge"`, …) are
  byte-identical: text-offset consumption must be zero-effect when offset
  columns are absent.
- Determinism: repeated renders of the same compare call are byte-identical
  (model order = registration order; feature order = global rank; dodge
  sub-band order = row encounter order).
- Renders must be visually verified: dodged bars/boxes actually offset,
  labels sit on their bars, legend present. SVG byte-equality alone is
  insufficient (goldens rule, CLAUDE.md).
- `cargo test` and the full pytest suite pass.
- `ferrum-spec.md` gets a dated note amending the 2026-06-27 small-multiples
  contract for the three affected charts (hard constraint: spec never
  silently drifts).
- No matplotlib; no global mutable state; no warn-fallbacks or
  `NotImplementedError`.

## 8. Key decisions and tradeoffs

**D1 — Combined DataFrame + `color="model"` + `Dodge(by="model")`, auto-upgrade
(no new kwarg).** The canonical seaborn/yellowbrick comparison plot becomes
the default; diagnostics ship opinionated defaults. Reuses the shipped dodge
subsystem, the catplot hue+dodge idiom, and the roc/pr stamped-model-column
convention. *Rejected:* opt-in `layout=` kwarg (new API surface, pushes
judgment onto users, contradicts the issue's acceptance); extending
`_compose_compare` (it composes finished charts — dodge needs the data merged
before construction); per-model `LayerChart` overlay (marks overlap, no
offsetting); Python-side precomputed offsets (position adjustment belongs in
Rust — paradigm violation).

**D2 — Horizontal orientation via vertical form + `CoordFlip`.** Dodge
operates on ordinal-x only; catplot already established normalize-to-x +
`CoordFlip` as the idiom for horizontal dodged layouts. *Rejected:* native
ordinal-y dodge in Rust — new axis generality with no other consumer.

**D3 — cv_scores: single-dodge on `x=split`; `kind="bar"` excluded.** The
recorded decision the issue requires. `split` has ≤ 2 levels and already
reads as x-categories, so `x=split` + dodge-by-model + `color="model"` is the
seaborn `hue` convention exactly; facet-by-split + dodge-by-model would
reintroduce panels to solve a problem two bands don't have. `kind="bar"`
(per-fold bars + per-split mean rules) carries three grouping dimensions —
no coherent single-dodge exists, so it keeps small multiples. `kind="strip"`
drops jitter under dodge (single-position rule, catplot precedent).

**D4 — Global cross-model feature ranking for importance/shap_bar.** One
ranking aggregated across models, one shared top-k feature set — the same
global-feature-set principle shap_bar already applies across classes.
*Rejected:* union of per-model top-k (row count balloons to k×m, breaks the
top-k ≤ k contract); base-model-only ranking (hides features that matter
only to a compared model).

**D5 — Text renderer consumes position offsets; `text` becomes
dodge-eligible.** Required for `show_values` labels to track their bars;
additive and zero-effect when no offsets are present. Eligibility follows
capability (precedent: `text` was added to stack eligibility for annotation
overlays). The `label` mark is deliberately excluded (own placement engine).

**D6 — shap_bar included by scope decision (2026-07-10).** The issue names
only importance/cv_scores, but shap_bar (`per_class=False`) satisfies the
issue's own eligibility rule (bar mark, single grouping dimension) and
already owns the cross-panel feature-ranking logic. `per_class=True` retains
small multiples: class is a competing facet dimension.

## 9. Acceptance criteria

- `importance_chart(m, X, y, compare={"alt": m2})` returns `Chart`; the SVG
  contains one bar per (feature, model) at distinct sub-band positions, a
  model legend, and — with defaults — dodged error rules and dodged value
  labels. Both `orient` values produce the dodged layout in the correct
  visual orientation.
- `shap_bar_chart(m, X, y, compare={...})` returns `Chart` with one bar per
  (feature, model); all models share one feature set of ≤ `max_display`
  features. With `per_class=True` it still returns the small-multiples
  `ConcatChart`.
- `cv_scores_chart(m, X, y, compare={...})` with `kind="box"` and `"strip"`
  returns `Chart` with per-model marks dodged within each split band and a
  model legend; `split="train"`/`"test"` filters bands as today;
  `kind="bar"` still returns the small-multiples `ConcatChart`.
- `compare=None` output byte-identical to the no-compare call for all three
  charts (existing tests stay green unmodified).
- All other compare diagnostics render byte-identical SVGs (existing
  small-multiples and golden tests for those rows stay green unmodified).
- A dodged `text` layer renders its glyphs at the dodged sub-band positions;
  text output without offsets is byte-identical to today.
- Repeated renders byte-identical (determinism tests, mirroring the existing
  importance ordinal-stability test).
- Regenerated/new goldens pass byte-diff AND are rasterized and visually
  inspected before commit.
- `cargo test` green; full `uv run pytest -n auto` green.
- `ferrum-spec.md` carries the dated amendment note.

## 10. Validation strategy

Behavior-level: rewrite the two #35 small-multiples assertions for
importance/cv_scores (and the FLAT-supervised cv_scores golden) to assert the
dodged single-panel contract; add discriminating tests that fail on the
pre-change layout — panel count, per-band mark multiplicity (n_models marks
per band), legend presence, and label/rule offset participation. Rust-side
unit test for text offset consumption. Visual proof via the golden
rasterize-and-inspect protocol (`tests/_snapshots.py::regen_and_verify`).
Byte-identity checks guard every unchanged surface (single-model paths,
other compare diagnostics, offset-free text).

## 11. Open questions

None.
