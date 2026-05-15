# Missing Figure-Level Helpers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Add 7 missing public `*_chart()` figure functions so every `FerrumVisualizer` has a one-liner shortcut in `ferrum.plots.*` and `ferrum.*`.

## 2. Spec references

- `ferrum-spec.md §3 Figure functions` — public API contract for all chart helpers
- `src/ferrum/plots/clustering.py` — internal helpers `_silhouette_chart_from_source`, `_elbow_*` pattern
- `src/ferrum/plots/regression.py` — internal helpers `_prediction_error_chart_from_source`, `_residuals_chart_from_source`
- `src/ferrum/plots/classification.py` — internal helpers `_class_balance_chart_from_dataframe`, `_classification_report_chart`
- `src/ferrum/_diagnostics/visualizers/clustering.py` — `ElbowVisualizer.fit()` k-sweep logic to mirror

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/plots/classification.py` | add `class_balance_chart`, `classification_report_chart` |
| Modify | `src/ferrum/plots/regression.py` | add `prediction_error_chart`, `cooks_distance_chart` |
| Modify | `src/ferrum/plots/clustering.py` | add `silhouette_chart`, `manifold_chart`, `elbow_chart` |
| Modify | `src/ferrum/plots/__init__.py` | export all 7 new functions |
| Modify | `src/ferrum/__init__.py` | re-export all 7 at top level |
| Test | `tests/diagnostics/test_classification.py` | smoke tests for `class_balance_chart`, `classification_report_chart` |
| Test | `tests/diagnostics/test_regression.py` | smoke tests for `prediction_error_chart`, `cooks_distance_chart` |
| Test | `tests/diagnostics/test_clustering.py` | smoke tests for `silhouette_chart`, `manifold_chart`, `elbow_chart` |

## 4. Constraints

- All 7 functions must follow the standard `(model_or_source, X, y, *, mark, encode, properties, layers, theme)` signature — except where the visualizer semantics require a different primary input (see task notes below).
- `elbow_chart` takes `model_class` (uninstantiated class) + `ks` range, not a fitted model — mirror `ElbowVisualizer.__init__` / `.fit()` signature exactly.
- `class_balance_chart` takes `y` only — no model, no X. Signature: `class_balance_chart(y, *, mark, encode, properties, layers, theme)`.
- Each function must delegate to its existing private helper (`_*_chart_from_source` or `_*_chart_from_dataframe`); do not duplicate chart-building logic.
- Use `_resolve_source` (from `ferrum.plots._helpers`) for all model-backed helpers — the same pattern as every other public function in this package.
- `_classification_report_chart` currently calls `source.model.predict(source.X)` directly; the public wrapper just needs to resolve source and delegate.
- Match docstring style of `residuals_chart` / `roc_chart` (Parameters / Returns / Examples sections, NumPy convention).

## 5. Tasks

### Task 1: regression helpers
- [ ] Add `prediction_error_chart(model_or_source, X, y, *, y_true, y_pred, identity_line, ci, reference_band, random_state, mark, encode, properties, layers, theme)` to `regression.py` — resolves source via `_resolve_source`, delegates to `_prediction_error_chart_from_source`
- [ ] Add `cooks_distance_chart(model_or_source, X, y, *, threshold, random_state, mark, encode, properties, layers, theme)` — resolves source, calls `_residuals_chart_from_source(source, kind="studentized", cook_threshold=threshold, panels=["residuals_vs_leverage"], ...)`
- [ ] Verify: `uv run pytest tests/diagnostics/test_regression.py -v`

### Task 2: classification helpers
- [ ] Add `classification_report_chart(model_or_source, X, y, *, random_state, mark, encode, properties, layers, theme)` to `classification.py` — resolves source via `_resolve_source`, delegates to `_classification_report_chart(source, ...)`
- [ ] Add `class_balance_chart(y, *, mark, encode, properties, layers, theme)` to `classification.py` — delegates directly to `_class_balance_chart_from_dataframe(y_series, ...)`
- [ ] Verify: `uv run pytest tests/diagnostics/test_classification.py -v`

### Task 3: clustering helpers
- [ ] Add `silhouette_chart(model_or_source, X, *, random_state, mark, encode, properties, layers, theme)` to `clustering.py` — resolves source, delegates to `_silhouette_chart_from_source(source, ...)`
- [ ] Add `manifold_chart(model_or_source, X, *, method, random_state, mark, encode, properties, layers, theme)` — resolves source, builds chart directly from `source.embeddings(method=method)` (same logic as `ManifoldVisualizer._build_chart`)
- [ ] Add `elbow_chart(model_class, X, *, ks, metric, random_state, mark, encode, properties, layers, theme)` — mirrors `ElbowVisualizer.fit()` k-sweep loop; constructs an `ElbowVisualizer` instance internally and returns `viz.fit(X)._chart` with override/theme applied
- [ ] Verify: `uv run pytest tests/diagnostics/test_clustering.py -v`

### Task 4: exports
- [ ] Add all 7 to `src/ferrum/plots/__init__.py` imports and `__all__`
- [ ] Add all 7 to `src/ferrum/__init__.py` `from ferrum.plots import (...)` block
- [ ] Verify: `uv run python -c "import ferrum as fm; print(fm.silhouette_chart, fm.elbow_chart, fm.manifold_chart, fm.prediction_error_chart, fm.cooks_distance_chart, fm.classification_report_chart, fm.class_balance_chart)"`

## 6. Acceptance checks

- `uv run pytest tests/diagnostics/ -v` — all pass
- `uv run python -c "import ferrum as fm; assert all(hasattr(fm, f) for f in ['silhouette_chart','elbow_chart','manifold_chart','prediction_error_chart','cooks_distance_chart','classification_report_chart','class_balance_chart']); print('OK')"` — prints OK
- Each new function returns a `Chart` instance when called with minimal smoke inputs
