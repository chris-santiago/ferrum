# Multi-model `compare=` Rendering for Aggregate Diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Wire the 17 gated aggregate model-diagnostics to render compared models as small multiples via one shared compose-per-model helper, and refine the 2 sweep-chart rejections.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-27-compare-aggregate-diagnostics-design.md` — full design. Key sections:
  - §6 Canonical interfaces — `_compose_compare` contract, gate-site contract, resolve-policy-by-bucket table
  - §7 Invariants — single-model byte-identical, no Rust, golden discipline
  - §8 Key decisions + scope table (19 gates, 4 buckets)
  - §9 Acceptance criteria, §10 Validation strategy, §11 Open question (composite-child label)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/plots/_helpers.py` | add `_compose_compare` helper |
| Modify | `src/ferrum/plots/explanation.py` | rewire 6 gates + docstrings |
| Modify | `src/ferrum/plots/model_selection.py` | rewire 4 gates + docstrings |
| Modify | `src/ferrum/plots/regression.py` | rewire 3 gates (incl. per-model band) + docstrings |
| Modify | `src/ferrum/plots/clustering.py` | rewire 4 gates + reword 2 rejections + docstrings |
| Modify | `tests/diagnostics/test_compare_exclusions.py` | flip 15 reject-assertions → render; reword 2 sweep-chart rejects |
| Modify | `tests/diagnostics/test_compare.py` | add compose render + discriminating-band tests |
| Modify | `ferrum-spec.md` | dated note: `compare=` now renders small multiples for these diagnostics |
| Test | `tests/diagnostics/test_compare_aggregate_goldens.py` | composite goldens (1 nested, 1 flat supervised, 1 unsupervised) |

## 4. Constraints

- **No Rust change.** No `.rs` file is touched; `cargo test` must stay green.
- **Single-model path byte-identical.** The helper is reached *only* when the resolved source `isinstance` `ComparedModelSource`; the `compare=None` branch keeps its exact current code. Every implemented gate gets a test asserting `compare=None` output equals omitting the kwarg.
- **Resolve policy is semantic, set at the call site:** supervised aggregates (explanation, model_selection, regression) → `resolve={"x":"shared","y":"shared"}`; unsupervised clustering (`pca_scree`, `intercluster_distance`, `silhouette`, `manifold`) → `resolve={"x":"independent","y":"independent"}`.
- **`cluster_diagnostics` and `elbow_chart` stay rejected** (sweep-based, no per-model source). Only reword the `_reject_compare` reason to the accurate structural reason and point at #43. Do NOT implement compose for them.
- **Gate-flip tests must fail on `main`, pass under the change** (RED proof before staging).
- **Composite goldens are not blessed until visually inspected:** rasterize via `python scripts/snapshot-goldens.py <name>` and `Read` the PNG before commit (CLAUDE.md). Sanity-check path counts before declaring a dense render broken.
- **`tests/diagnostics/test_compare_exclusions.py` is edited by multiple tasks** — run tasks sequentially (subagent-driven), not in parallel, to avoid conflicts.
- **Branch:** `feat/compare-aggregate-diagnostics-35` (plain feature branch, ferrum convention — no worktree). Python-only → dispatch coding to `python-coder`.

## 5. Tasks

### Task 1: Compose helper + composite-child labeling
- [ ] Add `_compose_compare(source, builder, *, builder_kwargs, resolve, columns=None)` to `src/ferrum/plots/_helpers.py` per spec §6 (loop `source.model_names` / per-model sources, call `builder(model_source, **builder_kwargs)`, label each child with its model name, compose `ConcatChart(children, columns=, resolve=)`; `columns` defaults to number of models).
- [ ] Resolve spec §11: confirm a model-name **title** attaches to a *composite* child (nested `pdp`/`residuals`); if `.properties(title=Title(...))` is unavailable on composites, implement the equivalent so every model panel is labeled. Behavior is fixed (panel labeled); mechanism is the task's call.
- [ ] Unit test in `tests/diagnostics/test_compare.py`: a fake builder returning a labeled `Chart` and one returning a composite both compose into a `ConcatChart` with N labeled panels; `resolve` is forwarded.
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/test_compare.py`

### Task 2: explanation.py — 6 gates (shared scales)
- Consumes: `_compose_compare` from Task 1 → `src/ferrum/plots/_helpers.py`
- [ ] Rewire `importance_chart`, `shap_beeswarm_chart`, `shap_bar_chart`, `shap_waterfall_chart`, `shap_chart`, `pdp_chart` per gate-site contract (spec §6): resolve source with `compare=`, branch on `ComparedModelSource` → `_compose_compare(... resolve={"x":"shared","y":"shared"})`, else existing path. `pdp_chart` = nested compose (per-feature × per-model).
- [ ] Update each function's docstring: replace the `compare=` `ValueError` note with the small-multiples behavior.
- [ ] In `tests/diagnostics/test_compare_exclusions.py`, flip these 6 from reject-assertions to render-assertions (compared call returns a `ConcatChart` with N panels); add `compare=None` byte-identical assertions.
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/ -k "explanation or compare"`

### Task 3: model_selection.py — 4 gates (shared scales)
- Consumes: `_compose_compare` from Task 1
- [ ] Rewire `learning_curve_chart`, `validation_curve_chart`, `cv_scores_chart`, `alpha_selection_chart` per gate-site contract, `resolve={"x":"shared","y":"shared"}`. Each per-model panel keeps its internal train/test coloring.
- [ ] Update docstrings.
- [ ] Flip these 4 in `test_compare_exclusions.py` to render-assertions + `compare=None` byte-identical.
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/ -k "selection or compare"`

### Task 4: regression.py — 3 gates (shared scales) + latent-bug fix
- Consumes: `_compose_compare` from Task 1
- [ ] Rewire `cooks_distance_chart` (compose-per-model).
- [ ] Rewire `residuals_chart` multi-panel path (currently the `panels not in (None,"single")` `ValueError`): compose the per-model 4-panel grid → nested compose.
- [ ] Rewire `prediction_error_chart` `ci=`/`reference_band=` path (currently the conditional `ValueError`): compose-per-model so each panel's band is computed from that model's residuals only — this closes the latent pooled-residual defect.
- [ ] Update docstrings.
- [ ] Flip these in `test_compare_exclusions.py`; add the discriminating-band test to `test_compare.py`: with two models of differing residual distributions, each panel's band bounds differ (a pooled band would make them identical).
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/ -k "regression or compare"`

### Task 5: clustering.py — 4 implement (independent scales) + 2 reword
- Consumes: `_compose_compare` from Task 1
- [ ] Rewire `pca_scree_chart`, `intercluster_distance_chart`, `silhouette_chart`, `manifold_chart` per gate-site contract with `resolve={"x":"independent","y":"independent"}` (all route through `_resolve_source(..., y=None)`).
- [ ] Reword `_reject_compare` reason for `cluster_diagnostics` and `elbow_chart` to the accurate structural reason (sweeps one clusterer class over `k` on a feature matrix; no per-model `ModelSource` to compare) and point at #43. Keep them raising.
- [ ] Update docstrings for all six (4 new behavior, 2 refined reason).
- [ ] In `test_compare_exclusions.py`: flip the 4 to render-assertions + `compare=None` byte-identical; update the 2 sweep-chart assertions to expect the new message text (still raises).
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/ -k "clustering or compare"`

### Task 6: API contract note + representative goldens
- [ ] `ferrum-spec.md`: add a dated note that `compare=` now renders small multiples for the affected diagnostics (and that the two sweep-based clustering charts remain excluded, ref #43).
- [ ] Add `tests/diagnostics/test_compare_aggregate_goldens.py` with one golden per representative bucket: a nested case (`pdp` or multi-panel `residuals`), a flat supervised case (`importance` or `cv_scores`), an unsupervised case (`silhouette`).
- [ ] Rasterize each new golden via `python scripts/snapshot-goldens.py` and `Read` the PNGs; confirm panels are populated and model-labeled before commit.
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/test_compare_aggregate_goldens.py`

## 6. Acceptance checks

- `unset CONDA_PREFIX && uv run --no-sync pytest -n auto tests/diagnostics/` — all pass
- `unset CONDA_PREFIX && uv run --no-sync pytest -n auto` — full suite green
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — green (no Rust change)
- Each of the 17 implemented gates: compared call returns a `ConcatChart` with one labeled panel per model; `compare=None` output byte-identical to `main`.
- `prediction_error(compare=…, ci=0.9)` panels have per-model bands (discriminating test passes).
- `cluster_diagnostics(compare=…)` / `elbow_chart(compare=…)` raise with the refined message.
- Representative composite goldens rasterized and visually confirmed.

## 7. Open questions

- None blocking. The composite-child label mechanism (spec §11) is resolved inside Task 1 before the nested gates (Tasks 2, 4) consume the helper.
