# Multi-model `compare=` Rendering for Aggregate Diagnostics — Design Spec

> GH issue #35. Output of a `coherent-change` decision pass (decision-only). Defended
> choice settled; this spec is the canonical design artifact for the build.
> Follow-ups already filed: #42 (selective dodge-by-model single-panel layout),
> #43 (algorithm/method-sweep comparison for the sweep-based clustering charts).

## 1. Scope

Make the model-diagnostic chart functions that today raise a documented `ValueError`
on `compare=` instead render the compared models as **small multiples** — one panel
per model, built by the existing single-model builder and composed with a shared
helper. The change is **pure Python**: it wires the already-existing composition layer
(`ConcatChart` + shared/independent scale resolution) to the already-existing
per-model source (`ComparedModelSource`). No Rust transform, mark, or position
subsystem is built — the issue's premise that one is required is incorrect.

## 2. Goals

- Every diagnostic whose output is a function of a per-model `ModelSource` renders
  N model panels when `compare=` is passed, instead of raising.
- The single-model path (no `compare=`) is **behaviorally unchanged** — byte-identical
  output.
- One shared helper expresses the compose-per-model pattern; gate sites call it, they
  do not re-implement it.
- Supervised aggregate panels are directly comparable (shared x/y scales); unsupervised
  panels keep their own coordinate systems (independent scales).
- Diagnostics with no per-model source (`cluster_diagnostics`, `elbow_chart`) carry a
  **refined, accurate** rejection rather than the current overstated "meaningless" one.

## 3. Non-goals

- **Dodge-by-model single-panel layouts** (grouped bars / dodged boxes for
  `importance`, `cv_scores`) — deferred to #42. The uniform small-multiples rule is the
  design here.
- **Method/algorithm-sweep comparison** for the sweep-based clustering charts
  (`cluster_diagnostics`, `elbow_chart`) — comparing `method="kmeans"` vs
  `"hierarchical"` is a distinct, design-open feature (tracked in #43), not part of
  `compare=`.
- Any change to the 6 classification + 2 regression diagnostics that already render
  multi-model by color overlay. They are untouched.
- Any Rust change. `cargo test` must remain green but no `.rs` file is modified.

## 4. System behavior

**Before:** `pdp_chart(model, X, y, compare={"b": m2})` (and the other gated functions)
raises `ValueError`.

**After:** the same call returns a `ConcatChart` of one panel per model. Each panel is
the exact chart the single-model builder produces for that model, labeled with the
model's name. Panels are arranged in a grid (default: a single row, one column per
model). Supervised-aggregate panels share x and y scales so the comparison is on common
axes; unsupervised panels resolve scales independently.

**Charts whose single-model output is itself composite** (the 4-panel `residuals` grid,
per-feature `pdp` facets) nest: the result is a `ConcatChart` whose children are those
composites, one per model.

**Per-model aggregate statistics are correct by construction.** Because each panel is
built from a single-model source, any per-model aggregate (e.g. the residual-quantile
CI / reference band in `prediction_error`) is computed over that model's rows only. This
closes the latent defect whereby a multi-model frame would have produced a single band
over pooled residuals from all models.

**Refined rejection** (sweep-based clustering): `cluster_diagnostics` and `elbow_chart`
still raise on `compare=`, but with a message that names the real reason — the function
sweeps one clusterer class over `k` on a feature matrix and wraps no per-model
`ModelSource`, so there is nothing to compare model-wise; comparing algorithms is a
separate feature.

## 5. Architecture

- **Per-model source** — `ComparedModelSource` (already built) wraps one `ModelSource`
  per model and exposes `.model_names` and the underlying per-model sources. It is the
  iteration surface.
- **Single-model builders** — each diagnostic's existing `_<name>_chart_from_source(
  source, ...)` builder is the unit of per-model work, reused unchanged. They already
  accept one resolved `ModelSource`.
- **Compose helper** — one new helper (in the diagnostics' shared `_helpers` module,
  beside `_resolve_source` / `_color_field_for`) detects a `ComparedModelSource`,
  invokes the builder once per model, labels each child, and composes via `ConcatChart`
  with the caller-supplied scale-resolution policy. It owns the small-multiples pattern;
  nothing else does.
- **Gate sites** — each gated public function replaces its `_reject_compare(...)` /
  conditional `ValueError` with: resolve the source (passing `compare=`); if the result
  is a `ComparedModelSource`, delegate to the compose helper with this chart's builder,
  builder kwargs, and resolve policy; otherwise the existing single-model path runs
  unchanged.

Data flow per compared call: public fn → `_resolve_source(compare=…)` →
`ComparedModelSource` → compose helper → {builder(source_i) → child chart}_i →
`ConcatChart(children, resolve=…)`.

## 6. Canonical interfaces / data contracts

**Compose helper** — the single integration seam. A reviewer verifies a gate site by
reading its one call against this contract.

```python
def _compose_compare(
    source,                 # ComparedModelSource (caller has already confirmed type)
    builder,                # the chart's _<name>_chart_from_source callable
    *,
    builder_kwargs: dict,   # forwarded verbatim to builder for every model
    resolve: dict[str, str],# scale policy, e.g. {"x": "shared", "y": "shared"}
    columns: int | None = None,  # grid columns; default = number of models (one row)
) -> "ConcatChart":
    """Build one panel per model via `builder(model_source, **builder_kwargs)`,
    label each panel with its model name, and compose as small multiples."""
```

- `builder(model_source, **builder_kwargs)` must return the same chart type the
  single-model path returns for that diagnostic (a `Chart` or a composite).
- Each child is labeled with its model name (a chart **title** carrying the name). The
  label must be visible whether the child is a `Chart` or a composite.
- `resolve` is passed through to `ConcatChart`'s shared-scale injection. Keys are
  channel names; values are `"shared"` or `"independent"`.

**Resolve policy by bucket** (semantic rule, not a tuning knob):

| Bucket | `resolve` |
|---|---|
| Supervised per-model aggregates | `{"x": "shared", "y": "shared"}` |
| Unsupervised, source-based | `{"x": "independent", "y": "independent"}` |

**Gate-site contract** (every implemented gate):

```python
source = _resolve_source(model, X, y, ..., compare=compare)
if isinstance(source, ComparedModelSource):
    return _compose_compare(source, _<name>_chart_from_source,
                            builder_kwargs={...}, resolve={...})
# else: existing single-model path, unchanged
```

## 7. Invariants and constraints

- **Single-model output is byte-identical** to today. The helper is only reachable when
  the source is a `ComparedModelSource`; the non-compared path keeps its exact current
  code.
- **No per-model aggregate is computed over pooled rows.** Each per-model panel sees one
  model's data only.
- **No Rust change.** No `.rs` file is modified; `cargo test` stays green.
- **`ferrum-spec.md` is the API contract.** It must record that `compare=` now renders
  small multiples for the affected diagnostics (dated note), and the per-chart
  docstrings that currently document the `ValueError` must be updated to the new
  behavior — or, for the two refined rejections, to the accurate reason.
- **Composite goldens are not blessed until visually inspected** (CLAUDE.md): any new or
  regenerated golden SVG for these multi-panel outputs is rasterized to PNG via
  `scripts/snapshot-goldens.py` and read before commit.
- **Python-only dispatch** to `python-coder` per CLAUDE.md.

## 8. Key decisions and tradeoffs

**D1 — Compose-per-model small multiples (chosen).** Reuses the existing builders and
composition layer; zero Rust; one uniform rule across all implemented gates.
- *Rejected — color overlay (mirror the working curve charts):* these gates exist
  precisely because overlay is illegible for bands, beeswarms, and multi-panel layouts.
- *Rejected — `Chart.facet(col="model")`:* `facet` is defined on `Chart`, not on
  composite charts; these diagnostics are composite. Infeasible.
- *Rejected-now — Rust dodge-by-model on one panel:* the right north-star for a few
  bar/box marks, but a new multi-model mechanism with no precedent in the diagnostics; it
  fractures the uniform rule. Deferred to #42.

**D2 — Uniform layout, no per-chart layout judgment.** Every implemented gate uses the
same helper. The one mental model ("`compare=` → small multiples by model") is the
user-facing contract. Per-mark refinements are opt-in follow-ups (#42).

**D3 — Scale resolution split by bucket.** Shared scales make supervised aggregates
fairly comparable; independent scales are *required* for unsupervised diagnostics
(two embeddings / scree plots live in incomparable coordinate systems — forcing shared
scales would be wrong, not merely ugly).

**D4 — Include `pca_scree`; refine-reject the sweep charts.** The dividing line is
structural: a diagnostic is comparable iff it wraps a per-model `ModelSource` whose
output depends on the model. `pca_scree`, `intercluster_distance`, `silhouette`,
`manifold` resolve such a source (scree depends on the fitted reducer, etc.) → include.
`cluster_diagnostics` and `elbow_chart` take the feature matrix and sweep one clusterer
class over `k` internally — there is no per-model source to iterate, so `compare=` has
no meaning under this design. They keep a rejection, but with an accurate reason.

**D5 — Close the 2 partial conditional gates too.** `residuals(panels≠"single")` and
`prediction_error(ci=/reference_band=)` are the same single-model-aggregate class; the
same compose-per-model treatment closes them, so they are in scope (and doing so fixes
the pooled-residual band defect).

### Scope table — 19 gates

| Bucket | Functions | Treatment | Resolve |
|---|---|---|---|
| 1. Supervised aggregates (11) | `importance`, `shap_beeswarm`, `shap_bar`, `shap_waterfall`, `shap`, `pdp`*, `learning_curve`, `validation_curve`, `cv_scores`, `alpha_selection`, `cooks_distance` | compose-per-model | shared |
| 2. Partial gates (2) | `residuals` (multi-panel)*, `prediction_error` (`ci=`/`reference_band=`) | compose-per-model | shared |
| 3. Unsupervised, source-based (4) | `pca_scree`, `intercluster_distance`, `silhouette`, `manifold` | compose-per-model | independent |
| 4. Sweep-based (2) | `cluster_diagnostics`, `elbow_chart` | refined rejection | — |

\* `pdp` and multi-panel `residuals` produce composite single-model output → nested
compose (a `ConcatChart` of composites). Highest implementation risk; the per-panel
label must attach to the composite child.

## 9. Acceptance criteria

- For each of the 17 implemented gates: a call with `compare={...}` that **raises today**
  returns a `ConcatChart` with one panel per model under the change; each panel matches
  the single-model builder's output for that model and is labeled with the model name.
- For each implemented gate: the no-`compare=` call returns output byte-identical to
  `main`.
- `prediction_error(compare=…, ci=0.9)` renders one panel per model, each band computed
  from that model's residuals only (not pooled).
- `pdp(compare=…)` and `residuals(compare=…, panels="auto")` render nested small
  multiples (per-model × per-feature, per-model × 4-panel) that rasterize to correctly
  populated panels (visually inspected PNG).
- `cluster_diagnostics(compare=…)` and `elbow_chart(compare=…)` raise `ValueError` with a
  message stating the function sweeps a clusterer over `k` with no per-model source
  (no longer the bare "meaningless" wording).
- Unsupervised compared panels (`silhouette`, `manifold`, `pca_scree`,
  `intercluster_distance`) resolve scales independently; supervised compared panels share
  scales — verified by inspecting the rendered axis domains.
- `ferrum-spec.md` records the new behavior with a dated note; affected docstrings no
  longer document a `ValueError` for the implemented gates.
- `cargo test` is green; no `.rs` file changed.

## 10. Validation strategy

- **Gate flip (per implemented gate):** an assertion that the compared call no longer
  raises and yields a composite with `len(models)` panels; paired with an assertion that
  the single-model call is unchanged. The compared assertion must fail on `main` and pass
  under the change.
- **Aggregate correctness:** for `prediction_error` with `ci=`, assert each panel's band
  bounds differ across models given models with differing residual distributions — a
  pooled band would make them identical, so this is the discriminating check.
- **Refined rejections:** assert the new message text for the two sweep charts and that
  they still raise.
- **Visual inspection:** new/regenerated goldens for representative gates from each bucket
  (one nested case — `pdp` or multi-panel `residuals`; one flat supervised — `importance`
  or `cv_scores`; one unsupervised — `silhouette`) are rasterized and read to confirm
  panels are populated and labeled, per the CLAUDE.md golden discipline.
- **Scale policy:** assert shared vs independent domain resolution holds for a supervised
  and an unsupervised compared chart respectively.

## 11. Open questions

- **Per-panel label on a composite child.** `.properties(title=Title(name))` is confirmed
  on `Chart`; the plan must confirm the same labeling works (or find the equivalent) for a
  composite child (nested `pdp` / `residuals`) so every model panel is named. If a
  composite cannot carry a title directly, the helper wraps or titles via the composition
  layer — a mechanism choice for the plan, but the *behavior* (every model panel labeled)
  is fixed here.
