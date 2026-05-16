# Precomputed Diagnostic Inputs Design Spec

## 1. Scope

Add a precomputed input path to the nine diagnostic figure functions that evaluate model output against ground truth. Users who already have `y_true` and `y_pred` arrays — from their own pipeline, a non-sklearn model, or a cross-validation loop — can call these functions without constructing or passing a fitted estimator. The twelve figure functions that require feature data (`X`) remain model-only.

## 2. Goals

- All nine in-scope figure functions accept `y_true` + `y_pred` as keyword-only args, bypassing `model_or_source` entirely.
- Existing call sites (`roc_chart(model, X, y)`) are unaffected — zero breaking changes.
- `y_pred` is a single array-like parameter covering both soft scores/probabilities and hard labels; each function's semantics determine interpretation.
- Residuals for `residuals_chart` are computed internally as `y_true − y_pred`; callers never compute them manually.
- `ferrum-spec.md` §3.14 reflects the updated signatures.

## 3. Non-goals

- Public `PrecomputedSource` class — unnecessary indirection; arrays pass directly as kwargs.
- Precomputed support for model-selection or feature-space functions (`importance_chart`, `shap_chart`, `learning_curve_chart`, `validation_curve_chart`, `decision_boundary_chart`, `cluster_diagnostics`, `pca_scree_chart`, `intercluster_distance_chart`, `rank_chart`, `parallel_coordinates_chart`, `cv_scores_chart`, `alpha_selection_chart`).
- Shape/dtype pre-validation beyond what `sklearn.metrics.*` raises naturally.
- Precomputed `compare=` multi-model path.

## 4. System behavior

A caller supplies either a model source or precomputed arrays — never both. The figure function produces an identical `Chart` object in both cases; the visual output is indistinguishable.

**Precomputed call examples:**

```python
# Binary classifier — scores path
ferrum.roc_chart(y_true=y_test, y_pred=clf.predict_proba(X_test))

# Multiclass — 2D probability matrix
ferrum.confusion_matrix_chart(y_true=y_test, y_pred=clf.predict(X_test))

# Regression — fitted values; residuals computed inside
ferrum.residuals_chart(y_true=y_test, y_pred=reg.predict(X_test))
```

**Model-backed calls are unchanged:**

```python
ferrum.roc_chart(clf, X_test, y_test)           # existing — still works
ferrum.roc_chart(ModelSource(clf, X_test, y_test))  # existing — still works
```

## 5. Architecture

The precomputed path inserts a lightweight internal adapter, `_PrecomputedSource`, that satisfies the same method protocol `ModelSource` does for the methods each in-scope figure function calls. It routes to the existing `stat_*` transforms (`stat_roc`, `stat_pr`, `stat_confusion`, `stat_calibration`, `stat_lift`) which already accept raw arrays and return polars DataFrames. Chart builders downstream see no change.

`_resolve_source()` (in `ferrum/plots/_helpers.py`) gains a precomputed branch: when `y_true` is not `None`, it returns a `_PrecomputedSource(y_true, y_pred)` rather than a `ModelSource`. All per-function builders consume the resolved source via the existing protocol.

`_PrecomputedSource` is never exported and never appears in `ferrum.__init__` or `ferrum-spec.md`.

## 6. Canonical interfaces

### Updated figure function signature (representative)

```python
def roc_chart(
    model_or_source=None,
    X=None,
    y=None,
    *,
    y_true=None,
    y_pred=None,
    # ... existing kwargs unchanged ...
) -> Chart: ...
```

Same pattern applies to: `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart`, `confusion_matrix_chart`, `class_prediction_error_chart`, `residuals_chart`.

### `y_pred` semantics per function

| Function(s) | `y_pred` interpretation |
|---|---|
| `roc_chart`, `pr_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart` | Soft scores / probabilities. 1D for binary; 2D `(n_samples, n_classes)` for multiclass. |
| `calibration_chart` | Predicted probabilities for the positive class (1D). |
| `confusion_matrix_chart`, `class_prediction_error_chart` | Hard class labels (1D). |
| `residuals_chart` | Fitted values (1D). Residuals = `y_true − y_pred`, computed internally. |

### Internal `_PrecomputedSource` protocol (not public)

Implements only the methods required by in-scope chart builders:

```python
class _PrecomputedSource:
    def __init__(self, y_true: ArrayLike, y_pred: ArrayLike) -> None: ...
    def roc_curve(self, *, average=None, drop_intermediate=True) -> pl.DataFrame: ...
    def pr_curve(self, *, average=None) -> pl.DataFrame: ...
    def calibration_curve(self, *, n_bins=10, strategy="uniform") -> pl.DataFrame: ...
    def gain_curve(self) -> pl.DataFrame: ...
    def lift_curve(self) -> pl.DataFrame: ...
    def confusion_matrix(self, *, normalize=None) -> pl.DataFrame: ...
    def predictions(self) -> pl.DataFrame: ...   # residuals_chart, class_prediction_error
    def discrimination_threshold(self, **kwargs) -> pl.DataFrame: ...
```

Each method delegates to the corresponding `stat_*` transform; no bespoke computation lives in `_PrecomputedSource`.

## 7. Invariants and constraints

- **Zero breaking changes.** Existing positional call `fn(model, X, y)` is valid. `model_or_source` default `None` does not affect callers who pass it positionally.
- **Exactly one input mode.** Supplying both `model_or_source` and `y_true`/`y_pred` raises `ValueError`. Supplying neither raises `ValueError`. Supplying `y_true` without `y_pred` (or vice versa) raises `ValueError`.
- **`compare=` requires model path.** The precomputed path raises `ValueError` if `compare` is also supplied.
- **`y_pred` is the single array parameter** — no separate `y_score`, `y_proba`, or `y_prob` aliases are introduced. Each function's docstring states the expected content.
- **Residuals are never caller-supplied.** `residuals_chart` with precomputed inputs computes `y_true − y_pred` internally; there is no `residuals=` kwarg.

## 8. Key decisions and tradeoffs

**`_PrecomputedSource` is internal, not public.** A public class adds surface area and the symmetry benefit is marginal — reusing precomputed arrays across multiple charts requires only storing them in variables, not wrapping them in an object. Rejected.

**Single `y_pred` parameter, not `y_score` + `y_pred`.** Avoids per-function proliferation. Each function's contract defines what `y_pred` means; sklearn's own `.from_predictions()` uses the same single-parameter approach. Functions whose semantics require soft scores document this; passing hard labels to a curve function fails at `sklearn.metrics.roc_curve` with a clear error.

**Routes through existing `stat_*` transforms.** `stat_roc`, `stat_pr`, `stat_confusion`, `stat_calibration`, `stat_lift` already accept raw arrays and produce the polars DataFrames the chart builders consume. Reusing them keeps a single computation path for each derived quantity. Rejected alternative: inline computation in `_PrecomputedSource` methods — duplicates logic already tested and maintained in stat transforms.

**`compare=` excluded from precomputed path.** Multi-model comparison requires multiple sources; the precomputed path accepts a single `y_true`/`y_pred` pair. A `compare=` dict of precomputed arrays would require a different interface. Deferred; not in scope.

## 9. Acceptance criteria

- `roc_chart(y_true=y, y_pred=scores)` produces a chart visually identical to `roc_chart(model, X_test, y_test)` when `scores = model.predict_proba(X_test)`.
- All nine in-scope functions accept the precomputed path and render without error for binary and multiclass inputs.
- `residuals_chart(y_true=y, y_pred=fitted)` renders all four panels correctly without the caller supplying residuals.
- `ValueError` is raised in all three invalid-input scenarios (neither, both, incomplete pair).
- `roc_chart(model, X, y)` (existing positional call) continues to pass all current tests unmodified.
- No new public symbols appear in `ferrum.__init__` or `ferrum-spec.md §3.1`.

## 10. Validation strategy

- Unit tests for each in-scope function: one test on the model path, one on the precomputed path, assert output `Chart` structure is equivalent.
- Parametrize binary vs. multiclass for curve and matrix functions.
- Explicit tests for each `ValueError` branch (neither / both / incomplete).
- `residuals_chart` precomputed test verifies all four panels render with correct axis labels.
- Existing test suite must pass without modification.
