# Remove Unnecessary `.to_numpy()` Conversions

**Date:** 2026-05-13
**Status:** Proposed

## Problem

`src/ferrum/_diagnostics/` contains 69 `.to_numpy()` calls. The diagnostics subsystem wraps scikit-learn estimators and metrics, and every data handoff converts polars Series/DataFrames to numpy arrays before calling sklearn. This was written as if sklearn requires numpy — it does not. Since sklearn 1.2 (2023), estimator methods (`predict`, `predict_proba`, `fit`, `transform`, `score`) and metrics functions (`roc_curve`, `precision_recall_curve`, `silhouette_samples`, `confusion_matrix`, `classification_report`, etc.) accept any array-like input, including polars Series and DataFrames. sklearn calls `check_array` / `validate_data` internally and handles conversion when needed.

The result is 63 unnecessary allocations on every diagnostics call — each one copies the full column or frame into a new numpy array that sklearn would never have asked for.

## Audit summary

| Category | Count | Description |
|---|---|---|
| **UNNECESSARY** | ~18 | Double wrapping: `np.asarray(self._y.to_numpy())` — the inner `.to_numpy()` already returns an ndarray |
| **REPLACEABLE (sklearn)** | ~35 | Premature conversion before sklearn calls that accept array-like inputs natively |
| **REPLACEABLE (polars)** | ~10 | `np.argsort()`, `np.quantile()`, `np.argmax()`, `.mean()` — polars has native equivalents |
| **NECESSARY** | ~6 | CV split integer-array indexing (`X_np[tr]`, `X_np[te]`), SHAP explainer compat, 2D grid ops |

### File-level distribution

| File | Count | Dominant pattern |
|---|---|---|
| `_diagnostics/sources/_classification.py` | 19 | sklearn metrics + double wrapping |
| `_diagnostics/charts.py` | 10 | sklearn calls + polars-replaceable reductions |
| `_diagnostics/sources/_selection.py` | 8 | `learning_curve`, `validation_curve`, `cross_validate` |
| `_diagnostics/visualizers/clustering.py` | 4 | `fit()` calls + polars-native `mean()`/`argmin()` |
| `_diagnostics/sources/_predictions.py` | 4 | `predict()`, `predict_proba()` |
| `_diagnostics/sources/_importance.py` | 4 | `permutation_importance`, SHAP |
| `_diagnostics/visualizers/selection.py` | 3 | `argmax()` on score columns |
| `_diagnostics/visualizers/regression.py` | 3 | polars-replaceable reductions (RMSE, max abs) |
| `_diagnostics/visualizers/classification.py` | 3 | `classification_report`, Series subtraction |
| `_diagnostics/sources/_clustering.py` | 3 | `silhouette_samples`, UMAP/TSNE |
| `_diagnostics/sources/_ranking.py` | 2 | double wrapping |
| `_diagnostics/visualizers/ranking.py` | 2 | sklearn calls |
| `figure/matrix.py` | 1 | `.to_numpy().tolist()` — `.to_list()` exists |
| `chart.py` | 1 | domain inference on Arrow column |
| `_diagnostics/stats.py` | 1 | double wrapping |

## Decision

Remove all UNNECESSARY and REPLACEABLE conversions. Keep the ~6 genuinely necessary ones (with a comment explaining why).

## Changes by tier

### Tier 1 — Remove double wrapping (~18 sites)

Replace `np.asarray(self._y.to_numpy())` and `np.asarray(x.to_numpy(), dtype=np.float64)` with direct Series references where the consumer accepts array-like, or a single `np.asarray(series, dtype=...)` where an explicit dtype cast is needed.

Before:
```python
y_true = np.asarray(self._y.to_numpy())
y_score = proba_df[proba_cols[1]].to_numpy()
fpr, tpr, _ = roc_curve(y_true, y_score)
```

After:
```python
fpr, tpr, _ = roc_curve(self._y, proba_df[proba_cols[1]])
```

### Tier 2 — Remove premature sklearn conversions (~35 sites)

Pass polars DataFrames/Series directly to sklearn. Examples:

| Before | After |
|---|---|
| `X_np = self._X.to_numpy(); model.predict(X_np)` | `model.predict(self._X)` |
| `X_np = self._X.to_numpy(); learning_curve(est, X_np, ...)` | `learning_curve(est, self._X, ...)` |
| `y_pred = model.predict(source.X.to_numpy())` | `y_pred = model.predict(source.X)` |

### Tier 3 — Replace with polars-native operations (~10 sites)

| numpy pattern | polars replacement |
|---|---|
| `np.argsort(-col.to_numpy())` | `col.arg_sort(descending=True)` |
| `np.argmax(scores.to_numpy())` | `scores.arg_max()` |
| `np.quantile(col.to_numpy(), [0.025, 0.975])` | `col.quantile(0.025)`, `col.quantile(0.975)` |
| `col.to_numpy().mean()` | `col.mean()` |
| `np.sqrt((resid.to_numpy()**2).mean())` | `(resid**2).mean()**0.5` or `resid.pow(2).mean().sqrt()` |
| `np.max(np.abs(col.to_numpy()))` | `col.abs().max()` |
| `tbl[c].to_numpy().tolist()` | `tbl[c].to_list()` |
| `df["a"].to_numpy() - df["b"].to_numpy()` | `df["a"] - df["b"]` |

### Tier 4 — Keep with comment (~6 sites)

These genuinely require numpy arrays:

1. **CV split integer-array indexing** (`_classification.py:394`) — `X_np[tr]` where `tr` is an integer index array from `StratifiedKFold.split()`. Polars does not support integer-array row indexing with the same semantics. Keep, add comment.

2. **SHAP explainer compatibility** (`_importance.py:165`) — SHAP's `TreeExplainer` / `KernelExplainer` do not reliably accept polars DataFrames as of shap 0.45. Keep until SHAP adds array-like support, add comment.

3. **2D grid operations** (`charts.py:2263`) — Decision-boundary mesh grid uses `X_np[:, i]` column slicing on a 2D array. Polars DataFrames use named columns, not positional 2D indexing. Keep.

## Verification plan

1. Run the full test suite (`uv run pytest`) after each tier — diagnostics tests exercise all affected code paths.
2. Regenerate any affected golden SVGs and visually inspect PNGs.
3. Spot-check that sklearn is not silently re-converting (profile with a breakpoint in `sklearn.utils.validation.check_array` to confirm polars inputs pass through without a copy where sklearn supports it).

## Risk

**Low.** sklearn's array-like contract is stable and well-tested. The polars-native replacements are trivial arithmetic/aggregation methods. The only risk is an edge case where a specific sklearn function's `check_array` call rejects a polars type — this would surface immediately as a `TypeError` in the test suite, and the fix is to add back `.to_numpy()` at that one site.

## Non-goal

This spec does not propose removing numpy as a dependency. Numpy remains needed for the ~6 genuinely necessary sites and is a transitive dependency of sklearn regardless.
