# Phase 10 — Model Diagnostics Design

**Status:** approved (brainstorming complete 2026-05-10)
**Phase:** 10 — Model Diagnostics
**Predecessor:** Phase 9 (Convenience / Figure-Level API, merged commit `11f956e` on `main`)
**Successors:** Phase 11 (Interactive Renderer), Phase 12 (Extension Points)
**Spec contract:** `ferrum-spec.md` §3.1 (ModelSource), §3.3 (model-diagnostic marks), §3.14 (Group B figure-level functions), §3.15 (sklearn-protocol Visualizers)

---

## 1. Goal, scope, and the philosophical departure

### 1.1 Goal

Ship the full model-diagnostics layer from `ferrum-spec.md`:

- **§3.1 ModelSource** — a duck-typed adapter that wraps any object implementing the sklearn estimator protocol (`predict`, `predict_proba`, `transform`, etc.) and exposes 22 derived-data methods, each returning a `polars.DataFrame` with a documented schema.
- **§3.3 Model-Diagnostic Marks** — 26 new marks: residuals, prediction_error, confusion, roc, pr, calibration, gain, lift, importance, shap_beeswarm, shap_bar, shap_waterfall, pdp, silhouette, learning_curve, validation_curve, decision_boundary, discrimination_threshold, parallel_coordinates, class_prediction_error, pca_scree, rank1d, rank2d, intercluster_distance, cv_scores, alpha_selection.
- **§3.14 Group B figure-level functions** — 21 convenience entry points: `roc_chart`, `pr_chart`, `confusion_matrix_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `residuals_chart`, `importance_chart`, `shap_chart`, `learning_curve_chart`, `validation_curve_chart`, `cluster_diagnostics`, `decision_boundary_chart`, `discrimination_threshold_chart`, `parallel_coordinates_chart`, `class_prediction_error_chart`, `pca_scree_chart`, `rank_chart`, `alpha_selection_chart`, `intercluster_distance_chart`, `cv_scores_chart`.
- **§3.15 sklearn-protocol Visualizers** — 25 classes implementing `fit` / `score` / `show` / `__repr__` over the same chart builders.

### 1.2 Done criteria (from `ferrum-phases.md`)

- [ ] `ModelSource` wraps any object with `predict` / `predict_proba` / `transform`.
- [ ] All model-diagnostic marks from `ferrum-spec.md §3.3` render correctly.
- [ ] sklearn is not imported unless the user's model is from sklearn.

### 1.3 Scope (in)

Everything in §3.1 / §3.3 / §3.14 / §3.15 above. Zero deferred items. Phase 9+ no-defer principle applies — every spec parameter ships with a full implementation, not a warn-fallback.

`ComparedModelSource` (via `ModelSource.compare({...})`) is in scope. The schema commitment for the optional `model` column ships in **10a** so encodings in 10b-10g don't need retrofitting when `compare()` lands in 10h.

### 1.4 Scope (out)

- `mark_arc`, `mark_image`, `mark_geoshape`, `mark_label` stay in `PHASE_9_PLUS_MARKS`. No §3.14 Group B figure function depends on them.
- Interactive selections on diagnostic charts wait for Phase 11.
- Per-layer scale unification follow-ups carried over from Phase 8b stay carried over (separate concern from Phase 10's contract).
- Continuous color **colorbar / legend** (the carry-over from Phase 9) is out of Phase 10 scope but unblocks `mark_confusion`'s color encoding once it lands. Phase 10 emits the scale entries correctly; the legend builder remains a Phase 11+ artifact. This does not gate Phase 10 acceptance because confusion-matrix goldens encode value-as-text, not value-as-color-only.

### 1.5 The philosophical departure (deliberate)

Phases 5-9 placed statistical compute in the **rendering pipeline** as Rust transforms declared in `ChartSpec` (KDE, bootstrap CI, regression, binning, linkage, etc.). Phase 10 deliberately places model-diagnostic compute in the **adapter layer** — Python `ModelSource` methods that delegate to lazy-imported sklearn / shap / umap. This is the only place in Phase 10 that departs from a default reading of `ferrum-spec.md §1`. Rationale:

1. **The user's call site IS the rendering pipeline.** `ferrum-spec.md §1` proscribes statistics-in-userspace; the user writing `ferrum.roc_chart(model, X, y).show()` is not computing ROC in userspace — the figure function is. Whether the internal compute is a Rust transform or a Python call to sklearn is invisible at the call site.
2. **Model-diagnostic compute is entangled with the model protocol.** A generic Rust transform cannot call `model.predict_proba(X)` or read `model.classes_` without reimplementing the sklearn estimator protocol. The adapter must be in Python.
3. **sklearn already implements the long tail of edge cases.** Multiclass ROC averaging (`micro`/`macro`/`weighted`), `drop_intermediate`, label encoding, NaN handling, calibration binning strategies — reimplementing these in Rust regresses correctness without buying any user-visible win.
4. **Compute that doesn't need sklearn already has Rust transforms.** Cumulative gain (sort + cumsum), class prediction error (group_by count), parallel coordinates (Unpivot from Phase 9), beeswarm SHAP (Swarm from Phase 8b) — these compose over existing Phase 5/8b/9 transforms.

A dated drift note in `ferrum-spec.md §1` will record the reasoning.

### 1.6 Sub-batch decomposition

Phase 10 is decomposed into **eight sub-batches**, each a contained vertical slice (ModelSource methods → marks → figure functions → visualizers → goldens) that ships a real user-visible capability. This is build-order decomposition; **no scope is dropped at any sub-batch boundary**.

| Sub-batch | Theme | Ships |
|---|---|---|
| **10a** | Foundation + regression diagnostics | ModelSource class, protocol detection, lazy-import infrastructure, `pyproject.toml` extras, `numeric_precision` field on RenderConfig (Rust), `model`-column schema commitments, `.predictions()`, `.probabilities()`, `mark_residuals`, `mark_prediction_error`, `residuals_chart`, `ResidualsVisualizer`, `PredictionErrorVisualizer`, `CooksDistanceVisualizer` |
| **10b** | Classification curves | `.roc_curve()`, `.pr_curve()`, `.calibration_curve()`, `.cumulative_gain()`, `.lift_curve()`, `.discrimination_threshold()`; `mark_roc`, `mark_pr`, `mark_calibration`, `mark_gain`, `mark_lift`, `mark_discrimination_threshold`; `roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart`; `ROCVisualizer`, `PRVisualizer`, `CalibrationVisualizer`, `DiscriminationThresholdVisualizer` |
| **10c** | Classification matrices | `.confusion_matrix()`; `mark_confusion`, `mark_class_prediction_error`; `confusion_matrix_chart`, `class_prediction_error_chart`; `ConfusionMatrixVisualizer`, `ClassificationReportVisualizer`, `ClassPredictionErrorVisualizer`, `ClassBalanceVisualizer` |
| **10d** | Feature importance + SHAP | `.importances()` (builtin + permutation), `.shap_values()`, `.partial_dependence()`; `mark_importance`, `mark_shap_beeswarm`, `mark_shap_bar`, `mark_shap_waterfall`, `mark_pdp`; `importance_chart`, `shap_chart`; `FeatureImportancesVisualizer`, `SHAPVisualizer`. `shap` optional dep gated by `ferrum[shap]`. |
| **10e** | Model selection / CV | `.learning_curve()`, `.validation_curve()`, `.cv_scores()`, `.alpha_selection()`; `mark_learning_curve`, `mark_validation_curve`, `mark_alpha_selection`, `mark_cv_scores`; `learning_curve_chart`, `validation_curve_chart`, `cv_scores_chart`, `alpha_selection_chart`; `LearningCurveVisualizer`, `ValidationCurveVisualizer`, `CVScoresVisualizer`, `AlphaSelectionVisualizer` |
| **10f** | Clustering / manifold / decision boundary | `.silhouette()`, `.intercluster_distance()`, `.embeddings()` (UMAP optional), `.pca_variance()`; `mark_silhouette`, `mark_intercluster_distance`, `mark_pca_scree`, `mark_decision_boundary`; `cluster_diagnostics`, `intercluster_distance_chart`, `pca_scree_chart`, `decision_boundary_chart`; `SilhouetteVisualizer`, `ElbowVisualizer`, `ManifoldVisualizer`, `InterclusterDistanceVisualizer`, `PCAVarianceVisualizer`. `umap-learn` optional dep gated by `ferrum[umap]`. |
| **10g** | Feature ranking + parallel coordinates | `.rank1d()`, `.rank2d()` (Kendall-tau-b via new `ferrum._core.kendall_tau_b` Rust function); `mark_rank1d`, `mark_rank2d`, `mark_parallel_coordinates`; `rank_chart`, `parallel_coordinates_chart`; `Rank1DVisualizer`, `Rank2DVisualizer`, `ParallelCoordinatesVisualizer` |
| **10h** | Finalize | `ModelSource.compare(...)` and `ComparedModelSource`; SVG goldens for every figure function (tiered byte-identical + quantized); spec drift notes consolidated into `ferrum-spec.md`; `PHASE_9_PLUS_MARKS` audit; Phase 10 marked **done** in `ferrum-phases.md` |

---

## 2. Architecture overview

### 2.1 Three-layer architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: User-facing API surface                                │
│   ferrum.roc_chart(model, X, y)        ferrum.ROCVisualizer     │
│   ferrum.confusion_matrix_chart(...)   ferrum.SHAPVisualizer    │
│   ...21 figure functions...            ...25 visualizers...     │
└─────────────────────────────────────────────────────────────────┘
                              │ both surfaces delegate to
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Private chart-builder layer                            │
│   src/ferrum/_diagnostics/charts.py                              │
│   _roc_chart_from_source(source, **kw) -> Chart                  │
│   _pr_chart_from_source(source, **kw) -> Chart                   │
│   ...one builder per diagnostic family...                        │
│                                                                  │
│   Each: ModelSource method → polars DataFrame                    │
│   → Chart(df).mark_*(...).encode(...) over Phase 8a-9 primitives│
└─────────────────────────────────────────────────────────────────┘
                              │ pulls derived data from
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: ModelSource adapter                                    │
│   src/ferrum/_diagnostics/source.py                              │
│   ModelSource(model, X, y=None, ...)                            │
│     .predictions() .probabilities() .roc_curve() ...22 methods   │
│   Protocol detection: attribute presence (no imports)            │
│   Each method lazy-imports sklearn/shap/umap PER CALL as needed  │
│   ModelSource.compare({...}) → ComparedModelSource               │
└─────────────────────────────────────────────────────────────────┘
                              │ delegates compute (lazy)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Optional third-party libraries (lazy-imported per method)       │
│   sklearn.metrics  sklearn.inspection  sklearn.manifold         │
│   sklearn.calibration  sklearn.model_selection                  │
│   shap  umap-learn                                              │
│                                                                  │
│   (scipy is NOT a runtime dep — used only as a dev/test dep     │
│    for parity validation of in-house Shapiro-Wilk and Kendall.) │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Marks: desugar-only, no new Rust variants

Following the locked decision from Phase 9 (*Composite marks desugar Python-side; no Rust Composite Mark variant*), every Phase 10 mark is **Python-side desugaring** over existing primitives. The full mapping:

| New Phase 10 mark | Desugars to |
|---|---|
| `mark_residuals` | `mark_point` + `mark_rule` (reference) + optional `mark_text` (Cook's labels) |
| `mark_prediction_error` | `mark_point` + `mark_line` (identity) + optional `mark_errorband` (reference band) |
| `mark_confusion` | `mark_rect` (heatmap cells, color encoded by `value`) + `mark_text` (cell values formatted via `text_fmt`) |
| `mark_roc` | `mark_line` per class + `mark_rule` (diagonal reference) + `mark_text` (AUC annotation in legend or chart) |
| `mark_pr` | `mark_line` per class + optional `mark_line` (iso-F1 curves at `iso_lines`) + `mark_text` (AP annotation) |
| `mark_calibration` | `mark_line` (per model when compared) + `mark_rule` (perfect-calibration diagonal) |
| `mark_gain` | `mark_line` (per class) + `mark_rule` (random baseline diagonal) |
| `mark_lift` | `mark_line` + `mark_rule` (lift = 1 baseline) |
| `mark_importance` | `mark_bar` + optional `mark_errorbar` |
| `mark_rank1d` | `mark_bar` (horizontal) sorted by `score` descending, `top_k` clamp |
| `mark_shap_beeswarm` | `mark_swarm` (Phase 8b primitive) keyed by `feature` (y axis), x = `shap_value`, color = `feature_value_normalized` |
| `mark_shap_bar` | `mark_bar` over per-feature mean `\|shap_value\|`, sorted descending |
| `mark_shap_waterfall` | `mark_bar` with pre-computed cumulative `x0`/`x1` columns + `mark_rule` (baseline E[f(X)]) |
| `mark_pdp` | `mark_line` (average partial dependence) + optional `mark_line` per `sample_id` at low alpha (ICE) |
| `mark_silhouette` | `mark_bar` (horizontal) sorted within cluster + `mark_rule` (mean silhouette) |
| `mark_learning_curve` | `mark_line` (train + test split) + `mark_errorband` (CI band, controlled by `ci_style`) |
| `mark_validation_curve` | same as `mark_learning_curve` but over `param_value` |
| `mark_alpha_selection` | `mark_line` + `mark_errorband` + optional `mark_rule` at `highlight_best` α |
| `mark_decision_boundary` | `mark_raster` (when `proba=True`, probability surface) OR `mark_contour` (when `proba=False`, boundary) + optional `mark_point` (scatter overlay) |
| `mark_discrimination_threshold` | `mark_line` per metric (precision/recall/f1/queue_rate) + optional `mark_rule` (estimated optimal threshold) |
| `mark_parallel_coordinates` | `mark_line` over `Unpivot`'d data (Phase 9 transform), grouped by `sample_id`, color by hue |
| `mark_class_prediction_error` | `mark_bar` with `Stack` position (Phase 9), color = actual class |
| `mark_pca_scree` | `mark_bar` per component + `mark_line` cumulative + optional `mark_rule` at `threshold_line` |
| `mark_intercluster_distance` | `mark_point` with `size` encoding (membership count) + `mark_text` (cluster labels) |
| `mark_rank2d` | same desugaring as `mark_confusion`: `mark_rect` + `mark_text` |
| `mark_cv_scores` | `mark_boxplot` (when `kind="box"`) / `mark_bar` (`"bar"`) / `mark_point` (`"strip"`) |

**Result: zero new Rust `Mark` variants, zero new Rust `Transform` variants.**

### 2.3 The one new Rust function: `ferrum._core.kendall_tau_b`

The single Rust addition is `ferrum._core.kendall_tau_b(x: &[f64], y: &[f64]) -> KendallResult` implementing Knight's 1966 O(n log n) algorithm with tie corrections (tau-b variant). Used by `ModelSource.rank2d(algorithm="kendall")`. Rationale: naive O(n²) Kendall in NumPy broadcast either OOMs or blows past 1s at `n_samples > 10k`; Knight's algorithm in pure Python is ~80 LOC of careful merge-sort accounting and still ~10× slower than Rust. For training-set-scale tabular ML data (n=100k–1M samples), only Rust gives acceptable latency.

All other in-house statistics (Pearson r, Spearman ρ, Shapiro-Wilk W, variance/covariance ranking, studentized residuals) are vectorized NumPy in `src/ferrum/_diagnostics/stats.py`. For these, the inner kernel is a BLAS GEMM/GEMV or `np.sort` — Rust ties NumPy because both call essentially the same SIMD-vectorized routines.

### 2.4 File layout

```
src/ferrum/
  __init__.py                  # adds 21 figure functions + ModelSource to public API
  _diagnostics/
    __init__.py                # re-exports ModelSource, ComparedModelSource
    source.py                  # ModelSource, ComparedModelSource, protocol detection
    deps.py                    # lazy-import helpers (require_sklearn, require_shap, require_umap)
    schemas.py                 # polars Schema constants for every derived-data DataFrame
    stats.py                   # pure-NumPy: pearson, spearman, shapiro_w, studentized, kendall (PyO3-backed)
    charts.py                  # private _*_chart_from_source builders (one per diagnostic family)
    visualizers/
      __init__.py              # 25 visualizer classes
      base.py                  # FerrumVisualizer (fit/score/show/__repr__)
      classification.py        # ROC, PR, Confusion, ClassificationReport
      regression.py            # Residuals, PredictionError, CooksDistance
      explanation.py           # FeatureImportances, SHAP, ParallelCoordinates
      selection.py             # LearningCurve, ValidationCurve, CVScores, AlphaSelection
      clustering.py            # Silhouette, Elbow, Manifold, Intercluster, PCAVariance
      ranking.py               # Rank1D, Rank2D
      classification_extra.py  # DiscriminationThreshold, ClassPredictionError, ClassBalance
  marks/
    diagnostic.py              # mark_residuals, mark_roc, ..., mark_shap_*, mark_pdp, ...
  figures.py                   # 22 ferrum.* figure functions

crates/ferrum-core/src/
  diagnostics.rs               # NEW — kendall_tau_b implementation (Knight's algorithm)
  render/svg.rs                # MODIFIED — RenderConfig.numeric_precision support (~10 lines)
  lib.rs                       # MODIFIED — re-export kendall_tau_b PyO3 binding

tests/
  fixtures/
    build.py                   # one-shot script that regenerates all model fixtures via skops
    models/                    # pre-fit sklearn models serialized via skops (.skops files)
    datasets/                  # canned small CSV/parquet datasets
  diagnostics/
    test_source.py             # ModelSource unit tests (per-method)
    test_stats.py              # parity vs scipy for in-house statistics
    test_regression.py         # 10a integration
    test_classification.py     # 10b + 10c integration
    test_explanation.py        # 10d integration
    test_selection.py          # 10e integration
    test_clustering.py         # 10f integration
    test_ranking.py            # 10g integration
    test_compare.py            # 10h ModelSource.compare integration
    test_no_sklearn_at_import.py  # asserts sklearn not loaded after `import ferrum`
  goldens/
    phase_10/
      byte_identical/          # ROC, PR, confusion, calibration, gain, lift, prediction_error,
                               #   residuals, class_prediction_error, rank2d, pca_scree,
                               #   parallel_coords, pdp (deterministic), discrimination_threshold,
                               #   cv_scores (from fixed CV splits), rank1d, importance (builtin)
      quantized_4dp/           # shap_chart, learning_curve, validation_curve, alpha_selection,
                               #   importance (permutation), decision_boundary, intercluster_distance,
                               #   cluster_diagnostics (UMAP/MDS branches)
```

### 2.5 Where Phase 10 touches Rust

Three places, all small:

1. **`crates/ferrum-core/src/diagnostics.rs`** (new) — Knight's O(n log n) Kendall τ-b algorithm. ~100 LOC + ~20 LOC of PyO3 binding.
2. **`crates/ferrum-core/src/render/svg.rs`** (modified) — `numeric_precision: Option<u8>` field on `RenderConfig`. When `Some(p)`, the float formatter rounds to `p` decimal places before emission. When `None`, current behavior. ~10 line diff.
3. **`crates/ferrum-core/src/lib.rs`** (modified) — re-export `kendall_tau_b` and pass `numeric_precision` through to the renderer.

No new `Mark` variants. No new `Transform` variants. No new `Scale` variants. No `ChartSpec` schema changes beyond the `numeric_precision` field on `RenderConfig`.

---

## 3. `ModelSource` adapter

### 3.1 Constructor and protocol detection

```python
class ModelSource:
    def __init__(
        self,
        model,
        X,
        y=None,
        *,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
        random_state: int | None = None,
    ):
        # No sklearn import here. Pure attribute introspection.
        self._model = model
        self._X = _coerce_to_polars(X)
        self._y = _coerce_to_polars(y) if y is not None else None
        self._feature_names = list(feature_names) if feature_names is not None else self._infer_feature_names()
        self._class_names = list(class_names) if class_names is not None else None
        self._sample_weight = sample_weight
        self._random_state = random_state

        self._capabilities = self._detect_capabilities()
        self._cache: dict[tuple, pl.DataFrame] = {}
```

**Capability detection** uses attribute presence only:

```python
_PROTOCOL_ATTRS = (
    "predict", "predict_proba", "decision_function", "transform",
    "fit_transform", "fit_predict", "score",
    "feature_importances_", "coef_", "explained_variance_ratio_",
    "cluster_centers_", "labels_", "classes_",
)

def _detect_capabilities(self) -> frozenset[str]:
    return frozenset(attr for attr in _PROTOCOL_ATTRS if hasattr(self._model, attr))
```

When a method is called that requires a missing capability, `ModelSource` raises a clear `AttributeError` naming the capability and the wrapped class:

```
AttributeError: ModelSource.probabilities() requires the wrapped model to implement
'predict_proba' or 'decision_function'. Got <class 'sklearn.svm._classes.LinearSVC'> which
implements neither. Pass `probability=True` to your estimator's constructor, or use
.predictions() for hard-label outputs.
```

`_coerce_to_polars` accepts numpy arrays, pandas DataFrames, polars DataFrames, pyarrow Tables / RecordBatches, or anything routable via narwhals (Phase 8a's existing coercion shim).

### 3.2 Lazy-import helpers

```python
# src/ferrum/_diagnostics/deps.py
def require_sklearn(method_name: str):
    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires scikit-learn. "
            f"Install it with `pip install ferrum[models]` or `pip install scikit-learn`."
        ) from e
    return sklearn

def require_shap(method_name: str): ...   # → pip install ferrum[shap]
def require_umap(method_name: str): ...   # → pip install ferrum[umap]
```

Every ModelSource method that needs a third-party library calls the corresponding `require_*` as its first line. `import ferrum` and `ModelSource.__init__` never touch sklearn / shap / umap.

A regression test (`tests/diagnostics/test_no_sklearn_at_import.py`) asserts:

```python
def test_import_ferrum_does_not_load_sklearn():
    import sys
    assert "sklearn" not in sys.modules
    import ferrum                                            # noqa: F401
    assert "sklearn" not in sys.modules
    import polars as pl
    source = ferrum.ModelSource(_DuckTypedFakeModel(), pl.DataFrame({"a": [1.0]}))
    assert "sklearn" not in sys.modules
```

### 3.3 Method surface and schema commitments

All 22 derived methods. Every returned `polars.DataFrame` follows a documented schema in `src/ferrum/_diagnostics/schemas.py`. **Schemas are compare-ready from 10a** — every schema documents an *optional* `model: str` column that is present when called through `ComparedModelSource` and absent otherwise. Chart builders use `"model" in df.columns` to decide whether to add a `color="model"` encoding.

| Method | Output schema (columns; `*` = optional `model` column appended for compare) |
|---|---|
| `.predictions()` | `y_true, y_pred, residual, studentized_residual` * |
| `.probabilities()` | `y_true`, one `proba_<class>` column per class * |
| `.roc_curve(*, average=None, drop_intermediate=True)` | `fpr, tpr, threshold, class, auc` * |
| `.pr_curve(*, average=None)` | `precision, recall, threshold, class, ap` * |
| `.confusion_matrix(*, normalize=None)` | `actual, predicted, value, value_fmt` * |
| `.calibration_curve(*, n_bins=10, strategy="uniform")` | `mean_predicted, fraction_positive, count` * |
| `.cumulative_gain()` | `percent_population, gain, class` * (includes baseline diagonal as `class="baseline"`) |
| `.lift_curve()` | `percent_population, lift, class` * (baseline `lift=1.0` as `class="baseline"`) |
| `.importances(*, method="builtin", n_repeats=30, scoring=None, random_state=None)` | `feature, importance, std, rank` * |
| `.shap_values(*, background=None, max_evals=500)` | `sample_id, feature, shap_value, feature_value, feature_value_normalized` * |
| `.partial_dependence(features, *, grid_resolution=100, kind="average")` | `feature, feature_value, pd_value, sample_id` * (`sample_id` null for `kind="average"`) |
| `.silhouette(k)` | `sample_id, cluster, silhouette_value` * |
| `.embeddings(*, method="umap", n_components=2, **method_kwargs)` | `dim_0, dim_1, (dim_2,) label` * |
| `.learning_curve(*, cv=5, scoring=None, train_sizes=None)` | `train_size, split, score, mean_score, std_score, lower, upper` * |
| `.validation_curve(param, values, *, cv=5, scoring=None)` | `param_value, split, score, mean_score, std_score, lower, upper` * |
| `.discrimination_threshold(*, n_thresholds=50, cv=None)` | `threshold, precision, recall, f1, queue_rate` * |
| `.pca_variance(*, n_components=None)` | `component, explained_variance_ratio, cumulative_variance_ratio` * |
| `.rank1d(*, algorithm="shapiro")` | `feature, score, rank` * |
| `.rank2d(*, algorithm="pearson")` | `feature_x, feature_y, correlation` * |
| `.cv_scores(*, cv=5, scoring=None)` | `fold, split, score` * (`split ∈ {"train", "test"}`) |
| `.alpha_selection(alphas, *, cv=5, scoring=None)` | `alpha, fold, score, mean_score, std_score` * |
| `.intercluster_distance(k, *, method="mds")` | `cluster, x, y, size` * |

### 3.4 Pre-computed cumulatives

To avoid introducing a new `Cumsum` Rust transform for `mark_shap_waterfall` and `mark_pca_scree`:

- `.pca_variance()` emits `cumulative_variance_ratio` directly (already in the spec schema).
- `mark_shap_waterfall` works over the long-form `.shap_values()` output for a single `sample_idx`; the waterfall builder computes `x0` and `x1` cumulative columns Python-side inside `_shap_waterfall_chart_from_source` before passing to `mark_bar`. No new transform required.

### 3.5 Method-by-method sklearn / shap / umap delegations

| Method | sklearn entry | scipy entry (dev-only) | shap | umap |
|---|---|---|---|---|
| `predictions` | `model.predict` | — (studentized residual uses in-house NumPy) | — | — |
| `probabilities` | `model.predict_proba` or `decision_function` + softmax | — | — | — |
| `roc_curve` | `sklearn.metrics.roc_curve`, `roc_auc_score` | — | — | — |
| `pr_curve` | `sklearn.metrics.precision_recall_curve`, `average_precision_score` | — | — | — |
| `confusion_matrix` | `sklearn.metrics.confusion_matrix` | — | — | — |
| `calibration_curve` | `sklearn.calibration.calibration_curve` | — | — | — |
| `cumulative_gain`, `lift_curve` | hand-coded (sort by `proba`, cumsum); NumPy only | — | — | — |
| `importances(method="builtin")` | reads `model.feature_importances_` or `\|model.coef_\|` | — | — | — |
| `importances(method="permutation")` | `sklearn.inspection.permutation_importance` | — | — | — |
| `shap_values` | — | — | `shap.Explainer(model).shap_values(X)` (auto-picks linear/tree/kernel) | — |
| `partial_dependence` | `sklearn.inspection.partial_dependence` | — | — | — |
| `silhouette` | `sklearn.metrics.silhouette_samples` + `model.labels_` / `model.predict(X)` | — | — | — |
| `embeddings(method="umap")` | — | — | — | `umap.UMAP(...)` |
| `embeddings(method="tsne")` | `sklearn.manifold.TSNE` | — | — | — |
| `embeddings(method="pca")` | `sklearn.decomposition.PCA` | — | — | — |
| `learning_curve` | `sklearn.model_selection.learning_curve` | — | — | — |
| `validation_curve` | `sklearn.model_selection.validation_curve` | — | — | — |
| `discrimination_threshold` | `sklearn.metrics.precision_recall_fscore_support` swept over `n_thresholds` evenly-spaced thresholds in `[0, 1]`. `queue_rate` is hand-computed at each threshold as `(y_score >= t).mean()` — sklearn does not provide it. When `cv` is set, each fold runs the same sweep on its held-out scores at the **same fixed `n_thresholds` grid**, then per-threshold metrics are averaged across folds. (Alternative — averaging per-fold sklearn outputs at fold-specific thresholds — is rejected because the threshold sets differ per fold.) | — | — | — |
| `pca_variance` | reads `model.explained_variance_ratio_` (no sklearn call needed) | — | — | — |
| `rank1d(algorithm="shapiro")` | — | parity test only | — | — |
| `rank1d(algorithm="variance"/"covariance")` | — (in-house NumPy) | — | — | — |
| `rank2d(algorithm="pearson"/"spearman")` | — (in-house NumPy / polars.corr) | — | — | — |
| `rank2d(algorithm="kendall")` | — (in-house Rust via `ferrum._core.kendall_tau_b`) | parity test only | — | — |
| `rank2d(algorithm="covariance")` | — (in-house NumPy) | — | — | — |
| `cv_scores` | `sklearn.model_selection.cross_validate(return_train_score=True)` | — | — | — |
| `alpha_selection` | `sklearn.model_selection.validation_curve(param_name="alpha", ...)` | — | — | — |
| `intercluster_distance(method="mds")` | `sklearn.manifold.MDS` + `model.cluster_centers_` | — | — | — |
| `intercluster_distance(method="tsne")` | `sklearn.manifold.TSNE` + `model.cluster_centers_` | — | — | — |

### 3.6 `ComparedModelSource`

```python
class ModelSource:
    @classmethod
    def compare(cls, models: dict[str, object], X, y, **kwargs) -> "ComparedModelSource":
        sources = {name: cls(model, X, y, **kwargs) for name, model in models.items()}
        return ComparedModelSource(sources)


class ComparedModelSource:
    """Same method surface as ModelSource; outputs gain a `model` column."""
    def __init__(self, sources: dict[str, ModelSource]):
        self._sources = sources

    def roc_curve(self, **kw) -> pl.DataFrame:
        frames = [
            s.roc_curve(**kw).with_columns(pl.lit(name).alias("model"))
            for name, s in self._sources.items()
        ]
        return pl.concat(frames, how="vertical_relaxed")

    # ...same shape for every other derived method...
```

Because every chart builder already checks `"model" in df.columns` from 10a, no per-builder code changes when `ComparedModelSource` lands in 10h — only the constructor changes.

### 3.7 Computation cache

`ModelSource` caches each derived DataFrame on first computation keyed by `(method_name, frozenset(kwargs.items()))`. `Visualizer.fit()` materializes the cache once; `Visualizer.show()` reads from cache.

The cache makes "build chart exactly once per visualizer" hold without complicating the chart-builder layer. Cache is invalidated by `ModelSource(...)` reconstruction (a new instance per visualizer / figure-function call).

---

## 4. `ferrum._core.kendall_tau_b` — the one new Rust function

### 4.1 Rationale

`ModelSource.rank2d(algorithm="kendall")` requires Kendall τ-b for every pair of features. At feature counts ≤ 100 this is fast; the bottleneck is **sample dimension**. For n_samples ≤ 1000, even O(n²) Python is fine. For n_samples ≥ 10k (typical training-set scale), naive O(n²) blows past 1s per pair. Knight's 1966 algorithm gives O(n log n) and ships τ-b in milliseconds.

NumPy cannot vectorize Knight's algorithm cleanly (it's a merge-sort variant with tie-counting). Pure Python Knight is ~10× slower than Rust Knight at n=100k. For training-set-scale rank2d, only Rust delivers acceptable latency.

### 4.2 Signature

```rust
// crates/ferrum-core/src/diagnostics.rs
pub fn kendall_tau_b(x: &[f64], y: &[f64]) -> KendallResult {
    // Knight's algorithm:
    // 1. Sort by x (stable), counting tied x pairs
    // 2. Count inversions in y via merge sort, accounting for tied y and tied x-and-y pairs
    // 3. Apply tie correction:
    //    tau_b = (n_c - n_d) / sqrt((n0 - n_x_ties) * (n0 - n_y_ties))
    //    where n0 = n*(n-1)/2
}

pub struct KendallResult {
    pub tau: f64,
    pub n_concordant: u64,
    pub n_discordant: u64,
    pub n_tied_x: u64,
    pub n_tied_y: u64,
    pub n_tied_both: u64,
}
```

PyO3 binding exposes it as `ferrum._core.kendall_tau_b(x: NDArray[f64], y: NDArray[f64]) -> dict` returning `{"tau": float, "n_concordant": int, "n_discordant": int, "n_tied_x": int, "n_tied_y": int, "n_tied_both": int}`.

### 4.3 Test plan

`crates/ferrum-core/tests/test_kendall.rs`:
- Synthetic fixtures with hand-computed τ-b: `[1,2,3]` vs `[1,2,3]` → 1.0; `[1,2,3]` vs `[3,2,1]` → -1.0; tied-x and tied-y cases verified against scipy.
- Random fixture cross-checked against scipy in `tests/diagnostics/test_stats.py` (Python-side parity test) with `random_state=0` over 20 random (n, p) pairs covering n ∈ {10, 100, 1000} and tie densities ∈ {0%, 10%, 50%}.
- Tolerance: `abs(rust_tau - scipy_tau) ≤ 1e-12`.

### 4.4 Cross-platform determinism

Knight's algorithm is sort-based with integer tie counts — no floating-point reductions in the core counts. The final `sqrt` and division use IEEE 754 ops that ARE deterministic at the bit level across x86-64 and arm64 for this scale. SVG goldens that include `rank2d(algorithm="kendall")` outputs are byte-identical.

---

## 5. `RenderConfig.numeric_precision` — the Rust-side quantization knob

### 5.1 Motivation

Phase 9 SVG goldens are byte-identical across platforms because Phase 9 RNG is fully Rust-internal (seeded ChaCha8Rng). Phase 10 inherits sklearn / shap / umap numerics. Eigendecomposition sign flips, iterative-solver convergence order, and BLAS reduction order can produce bit-different floats on macOS-arm64 vs Linux-x86 even with identical `random_state`.

The Phase 9 0-xfail / 0-skip discipline requires a determinism strategy. The chosen approach:

- **Pre-fit serialized models** in `tests/fixtures/models/` eliminate training-time platform variance for ~19 figures whose compute is downstream of a fitted model (ROC, PR, confusion, calibration, etc.).
- **Numeric quantization to 4 decimal places** at SVG-emission time absorbs cross-platform residual variance for ~10 figures whose compute uses iterative solvers (SHAP-Kernel, UMAP, t-SNE, MDS, learning_curve, validation_curve, alpha_selection, permutation_importance, decision_boundary on stochastic models, intercluster_distance via MDS/t-SNE).

### 5.2 Implementation

```rust
// crates/ferrum-core/src/spec/render.rs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderConfig {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub format: RenderFormat,
    // ... existing fields ...
    pub numeric_precision: Option<u8>,  // NEW. None = current behavior. Some(p) = round to p decimals.
}
```

In `crates/ferrum-core/src/render/svg.rs`, the float formatter consults `render_config.numeric_precision`:

```rust
fn fmt_float(buf: &mut String, val: f64, precision: Option<u8>) {
    match precision {
        None => write!(buf, "{}", val).unwrap(),
        Some(p) => write!(buf, "{:.*}", p as usize, val).unwrap(),
    }
}
```

All existing float emit sites route through `fmt_float`. Diff is ~10 lines.

### 5.3 Behavioral guarantee

`numeric_precision=None` (default) produces SVG output byte-identical to Phase 9. Existing Phase 9 goldens remain valid without regeneration. `numeric_precision=4` is only set in Phase 10 quantized-golden tests.

### 5.4 Spec drift note

`ferrum-spec.md §3.16` `RenderConfig` documentation gains a row for `numeric_precision`.

---

## 6. Marks (Phase 10 — Python-side desugar)

### 6.1 The desugar pattern

Every Phase 10 mark follows the Phase 9 composite-mark pattern (`mark_boxplot`, `mark_boxen`, etc.):

```python
class mark_roc:
    """User-facing immutable value class. Expanded Python-side at Chart.compile() time."""
    average: AverageMode | None = None
    reference_line: bool = True
    annotate_auc: bool = True

    def _expand(self, chart_ctx: "ChartContext") -> list[LayerSpec]:
        # Resolve color encoding: 'class' is auto-set if the data has a 'class' column.
        # Add 'color="model"' overlay if the data has a 'model' column (ComparedModelSource).
        layers = [
            LayerSpec(
                mark=mark_line(...),
                encoding={"x": "fpr", "y": "tpr", "color": self._color_field(chart_ctx)},
            ),
        ]
        if self.reference_line:
            layers.append(LayerSpec(
                mark=mark_rule(strokeDash=[4, 4]),
                encoding={"x": "fpr", "y": "fpr"},
                data_source="_roc_diagonal",  # synthetic [(0,0), (1,1)] DataFrame
            ))
        if self.annotate_auc:
            layers.append(LayerSpec(
                mark=mark_text(...),
                encoding={"x": 0.95, "y": 0.05 + 0.05*i, "text": f"AUC = {auc:.3f}"},
                data_source="_auc_labels",
            ))
        return layers
```

The expansion happens at `Chart.compile()` (Python-side, before any Rust call), so by the time the `ChartSpec` IR is built and serialized to Rust, the diagnostic mark has been reduced to primitive marks only. Rust sees no Phase 10-specific Mark variant.

### 6.2 Mark drift notes (consolidated in §13)

The following marks need clarifications added to `ferrum-spec.md §3.3`:

| Mark | Drift note |
|---|---|
| `mark_residuals` | `kind="studentized"` is well-defined only for linear estimators (those exposing the residual hat matrix). For non-linear estimators, ferrum falls back to raw residual and logs an INFO message naming the requested-but-unavailable kind. |
| `mark_prediction_error` | Requires `y_pred` and `y_true` continuous; on classifiers, falls back to `predict` outputs (label values) — a less informative chart but valid. |
| `mark_confusion` | Color scale on `value` uses Phase 8b's `ColorScale::Continuous` (`viridis` default). Continuous colorbar legend is a Phase 11+ artifact; cell-text annotation via `value_fmt` conveys magnitude. |
| `mark_roc`, `mark_pr` | `average` accepts `None` (per-class) or `"micro"`/`"macro"`/`"weighted"` (sklearn semantics). When set, the output DataFrame has an extra row per averaged class label. |
| `mark_calibration` | `n_bins` and `strategy` (`"uniform"` / `"quantile"`) are forwarded to `sklearn.calibration.calibration_curve`. |
| `mark_gain`, `mark_lift` | Always include the baseline diagonal as a `class="baseline"` row in the data; mark adds an explicit `mark_rule` only for the perfect-classifier wizard line (gain) or the lift=1.0 line. |
| `mark_importance` | When `method="permutation"` and the model has no `predict` method, raises a clear error before reaching sklearn. `top_k` truncates the bar chart; ranking is on `\|importance\|`. |
| `mark_shap_*` | Require `shap` (gated by `ferrum[shap]`). On `ImportError`, raises with a pointer to `pip install ferrum[shap]`. |
| `mark_shap_beeswarm` | `order` accepts `"abs_mean"` (default) or `"max_abs"`. Color encodes `feature_value_normalized` (z-scored feature value), not raw `feature_value`. |
| `mark_shap_waterfall` | Accepts a required `sample_idx: int` kwarg on the mark itself (e.g. `mark_shap_waterfall(sample_idx=3)`). The mark's `_expand` filters the long-form `.shap_values()` DataFrame to that one sample and computes cumulative `x0`/`x1` columns Python-side. `shap_chart(kind="waterfall", sample_idx=3, ...)` forwards the kwarg. Calling the mark without `sample_idx` raises `TypeError` at expand time. |
| `mark_pdp` | `kind="individual"` and `"both"` produce ICE traces with low alpha. `center=True` subtracts the mean from each trace. |
| `mark_silhouette` | Within each cluster, samples are sorted by descending `silhouette_value` (the canonical Rousseeuw silhouette plot ordering). |
| `mark_learning_curve`, `mark_validation_curve`, `mark_alpha_selection` | `ci_style="band"` uses `mark_errorband`; `ci_style="errorbar"` uses `mark_errorbar`. Both consume pre-computed `lower`/`upper` columns. |
| `mark_decision_boundary` | Requires exactly 2 features. `proba=True` selects `mark_raster` over `predict_proba`; `proba=False` selects `mark_contour` over `decision_function` or hard `predict`. |
| `mark_discrimination_threshold` | `metrics` selects which of `precision`/`recall`/`f1`/`queue_rate` to draw; default all four. `threshold_line=True` adds a `mark_rule` at the F1-maximizing threshold. |
| `mark_parallel_coordinates` | `rescale ∈ {"minmax", "zscore", None}`. Uses Phase 9's `Unpivot` transform. `hue` colors by a single column (typically `y`). |
| `mark_class_prediction_error` | Stacked bar with `Stack` position adjustment from Phase 9. `normalize=True` divides by total per actual-class. |
| `mark_pca_scree` | `cumulative_line=True` overlays a `mark_line` on `cumulative_variance_ratio`. `threshold_line=0.95` draws a `mark_rule` at the 95% cumulative-variance level. |
| `mark_rank1d` | `algorithm ∈ {"shapiro", "variance", "covariance"}`. Default `"shapiro"`. |
| `mark_rank2d` | `algorithm ∈ {"pearson", "spearman", "kendall", "covariance"}`. `kendall` uses `ferrum._core.kendall_tau_b`. |
| `mark_intercluster_distance` | `size` encoding is membership count. `min_size` / `max_size` control the point-size range. `label_clusters=True` annotates each point with cluster index. |
| `mark_cv_scores` | `kind ∈ {"box", "bar", "strip"}` selects the underlying primitive. `split ∈ {"train", "test", "both"}`. |

### 6.3 `PHASE_9_PLUS_MARKS` update

At Phase 10 close-out, every Phase 10 mark must be **removed** from `PHASE_9_PLUS_MARKS`. The deferred list at the end of Phase 10 retains only `arc`, `image`, `geoshape`, `label` (their original Phase 9+ membership, unchanged).

A test (`tests/diagnostics/test_mark_coverage.py`) asserts that every mark named in §6.2 is implemented (not in `PHASE_9_PLUS_MARKS`, callable from `src/ferrum/marks/diagnostic.py`).

---

## 7. Figure-level functions (§3.14 Group B)

### 7.1 Canonical pattern

Every figure function is a thin facade (~10-30 LOC) over the corresponding `_*_chart_from_source` builder:

```python
def roc_chart(
    model_or_source,
    X=None,
    y=None,
    *,
    per_class: bool = True,
    average: AverageMode = "macro",
    annotate_auc: bool = True,
    compare: dict[str, object] | None = None,
    random_state: int | None = None,
    theme: Theme | None = None,
) -> Chart:
    source = _resolve_source(model_or_source, X, y, random_state=random_state, compare=compare)
    return _roc_chart_from_source(
        source,
        per_class=per_class,
        average=average,
        annotate_auc=annotate_auc,
        theme=theme,
    )
```

`_resolve_source` accepts `model | ModelSource | ComparedModelSource | dict[str, model]` and produces the appropriate source. The same helper is used by every figure function — so `compare=` works uniformly across all of them in 10h with one change site, not 22.

### 7.2 `random_state` policy

All 21 figure functions accept `random_state: int | None = None` (per the spec drift note in §13). Functions whose compute is RNG-touching pass it through to the underlying ModelSource method / sklearn / shap / umap. Functions whose compute is deterministic (ROC, PR, confusion, etc.) accept the kwarg silently as a forward-compat no-op. Docs note per function whether `random_state` actually affects output, so users have one uniform API.

### 7.3 The 21 figure functions

Implementation summary (each in `src/ferrum/figures.py`, ~20 LOC apiece). Marks used = the primary mark; supporting layers (reference lines, AUC annotation) are added by the mark's `_expand` method.

| Figure function | Source method | Mark | Notes |
|---|---|---|---|
| `roc_chart(model, X, y, *, per_class, average, annotate_auc, compare, random_state, theme)` | `.roc_curve()` | `mark_roc` | random_state silent unless `compare` triggers stochastic CV |
| `pr_chart(model, X, y, *, per_class, annotate_ap, iso_lines, compare, random_state, theme)` | `.pr_curve()` | `mark_pr` | random_state silent |
| `confusion_matrix_chart(model, X, y, *, normalize, cmap, random_state, theme)` | `.confusion_matrix()` | `mark_confusion` | random_state silent |
| `calibration_chart(*model_or_sources, X, y, *, n_bins, random_state, theme)` | `.calibration_curve()` | `mark_calibration` | variadic models compose into `ComparedModelSource` |
| `gain_chart(model, X, y, *, random_state, theme)` | `.cumulative_gain()` | `mark_gain` | random_state silent |
| `lift_chart(model, X, y, *, random_state, theme)` | `.lift_curve()` | `mark_lift` | random_state silent |
| `residuals_chart(model, X, y, *, kind, panels, random_state, theme)` | `.predictions()` | `mark_residuals` (+ optional `mark_qq`, `mark_residuals` variants per `panels`) | `panels="auto"` selects up to 4 panels: residuals_vs_fitted, qq, scale_location, residuals_vs_leverage. Uses Phase 8a `&` vstack. |
| `importance_chart(model, X, y, *, method, top_k, orient, error_bars, random_state, theme)` | `.importances()` | `mark_importance` | random_state used when `method="permutation"` |
| `shap_chart(model, X, *, kind, max_display, sample_idx, random_state, theme)` | `.shap_values()` | `mark_shap_*` per `kind` | requires `ferrum[shap]`; random_state passes to KernelExplainer |
| `learning_curve_chart(model, X, y, *, cv, scoring, train_sizes, ci_style, n_jobs, random_state, theme)` | `.learning_curve()` | `mark_learning_curve` | random_state passes to CV splitter |
| `validation_curve_chart(model, X, y, param, values, *, cv, scoring, log_scale, ci_style, random_state, theme)` | `.validation_curve()` | `mark_validation_curve` | random_state passes to CV splitter |
| `cluster_diagnostics(X, *, ks, method, scoring, n_init, random_state, theme)` | `.silhouette()` + elbow loop | `mark_silhouette` + `mark_line` (elbow score) | random_state passes to KMeans |
| `decision_boundary_chart(model, X, y, *, features, grid_resolution, proba, scatter, random_state, theme)` | (no source method; grid prediction Python-side) | `mark_decision_boundary` | random_state silent unless model is stochastic |
| `discrimination_threshold_chart(model, X, y, *, n_thresholds, metrics, highlight_best, random_state, theme)` | `.discrimination_threshold()` | `mark_discrimination_threshold` | random_state passes to CV if `cv` provided |
| `parallel_coordinates_chart(data_or_source, X, y, *, features, hue, rescale, alpha, random_state, theme)` | (no source method; Unpivot from raw X) | `mark_parallel_coordinates` | random_state silent; data path bypasses ModelSource when given raw data |
| `class_prediction_error_chart(model, X, y, *, normalize, random_state, theme)` | `.predictions()` + `.confusion_matrix()` | `mark_class_prediction_error` | random_state silent |
| `pca_scree_chart(model, X, *, n_components, cumulative_line, threshold, random_state, theme)` | `.pca_variance()` | `mark_pca_scree` | random_state silent |
| `rank_chart(data_or_source, X, y, *, rank, algorithm, top_k, random_state, theme)` | `.rank1d()` or `.rank2d()` per `rank` | `mark_rank1d` or `mark_rank2d` | random_state silent |
| `alpha_selection_chart(model, X, y, alphas, *, cv, scoring, log_scale, ci_style, random_state, theme)` | `.alpha_selection()` | `mark_alpha_selection` | random_state passes to CV |
| `intercluster_distance_chart(model, X, *, k, method, random_state, theme)` | `.intercluster_distance()` | `mark_intercluster_distance` | random_state passes to MDS/t-SNE |
| `cv_scores_chart(model, X, y, *, cv, scoring, kind, split, random_state, theme)` | `.cv_scores()` | `mark_cv_scores` | random_state passes to CV splitter |

### 7.4 Multi-panel residuals

`residuals_chart(panels="auto")` builds up to four panels using existing Phase 8a/8b/9 vstack and hstack:

```python
panels_to_draw = _resolve_panels(model, panels)  # default ["residuals_vs_fitted", "qq", "scale_location", "residuals_vs_leverage"]
charts = [_residuals_panel(source, panel_name) for panel_name in panels_to_draw]
return reduce(operator.or_, charts[:2]) & reduce(operator.or_, charts[2:])  # 2x2 grid
```

Each panel is a single-layer Chart using existing primitives:

- `residuals_vs_fitted`: `mark_point` + `mark_rule(y=0)`
- `qq`: `mark_qq` (Phase 8b)
- `scale_location`: `mark_point` over `sqrt(|studentized|)` vs `y_pred`
- `residuals_vs_leverage`: `mark_point` colored by Cook's distance (where available) + Cook contour lines

---

## 8. Visualizers (§3.15)

### 8.1 Base class

```python
# src/ferrum/_diagnostics/visualizers/base.py
class FerrumVisualizer:
    def __init__(self, model=None, *, random_state: int | None = None, theme: Theme | None = None, **kwargs):
        self.model = model
        self.random_state = random_state
        self.theme = theme
        self._fitted = False
        self._source: ModelSource | None = None
        self._chart: Chart | None = None
        self._metrics: dict[str, float] = {}

    def fit(self, X, y=None) -> Self:
        self._source = ModelSource(self.model, X, y, random_state=self.random_state)
        self._materialize()                       # subclass-specific: compute derived data + metrics
        self._chart = self._build_chart()         # subclass-specific: call _*_chart_from_source
        self._fitted = True
        return self

    def score(self, X, y) -> float:               # subclass-specific
        raise NotImplementedError

    def show(self) -> Chart:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit before .show(); call .fit(X, y) first.")
        return self._chart

    def __repr__(self) -> str:
        if not self._fitted:
            return f"{type(self).__name__}(unfit)"
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self._metrics.items())
        return f"{type(self).__name__}({metric_str})"
```

### 8.2 The 25 concrete visualizers

Each is ~30-60 LOC. Same delegation pattern as figure functions — the chart builder is shared.

| Visualizer | Wraps | Sub-batch | Notes |
|---|---|---|---|
| `ResidualsVisualizer(model, *, kind, theme)` | `residuals_chart` | 10a | |
| `PredictionErrorVisualizer(model, *, identity_line, theme)` | prediction error builder | 10a | |
| `CooksDistanceVisualizer(model, *, threshold, theme)` | residual + Cook's distance | 10a | |
| `ROCVisualizer(model, *, micro, macro, per_class, theme)` | `roc_chart` | 10b | |
| `PRVisualizer(model, *, theme)` | `pr_chart` | 10b | |
| `CalibrationVisualizer(*models, *, n_bins, theme)` | `calibration_chart` | 10b | variadic; uses compare path |
| `DiscriminationThresholdVisualizer(model, *, n_thresholds, scoring, cv, theme)` | `discrimination_threshold_chart` | 10b | binary classifiers only |
| `ConfusionMatrixVisualizer(model, *, normalize, theme)` | `confusion_matrix_chart` | 10c | |
| `ClassificationReportVisualizer(model, *, theme)` | per-class P/R/F1 heatmap (uses rank2d-style mark_rect + text) | 10c | |
| `ClassPredictionErrorVisualizer(model, *, normalize, theme)` | `class_prediction_error_chart` | 10c | |
| `ClassBalanceVisualizer(*, theme)` | bar of y class counts | 10c | no model required; .fit(X, y) takes raw matrix |
| `FeatureImportancesVisualizer(model, *, method, top_k, theme)` | `importance_chart` | 10d | |
| `SHAPVisualizer(model, *, kind, background, theme)` | `shap_chart` | 10d | requires `ferrum[shap]` |
| `LearningCurveVisualizer(model, *, cv, scoring, train_sizes, theme)` | `learning_curve_chart` | 10e | |
| `ValidationCurveVisualizer(model, param, values, *, cv, scoring, theme)` | `validation_curve_chart` | 10e | |
| `CVScoresVisualizer(model, *, cv, scoring, kind, theme)` | `cv_scores_chart` | 10e | |
| `AlphaSelectionVisualizer(model, alphas, *, cv, scoring, theme)` | `alpha_selection_chart` | 10e | |
| `SilhouetteVisualizer(model, *, theme)` | silhouette chart | 10f | clusterers |
| `ElbowVisualizer(model_class, *, ks, metric, theme)` | elbow curve from `cluster_diagnostics` | 10f | takes a class, not a fitted model; fits one per k |
| `ManifoldVisualizer(model, *, method, theme)` | embedding scatter | 10f | UMAP requires `ferrum[umap]` |
| `InterclusterDistanceVisualizer(model, *, method, theme)` | `intercluster_distance_chart` | 10f | |
| `PCAVarianceVisualizer(model, *, n_components, theme)` | `pca_scree_chart` | 10f | model must expose `explained_variance_ratio_` |
| `Rank1DVisualizer(*, algorithm, top_k, theme)` | `rank_chart(rank="1d")` | 10g | no model required |
| `Rank2DVisualizer(*, algorithm, theme)` | `rank_chart(rank="2d")` | 10g | no model required |
| `ParallelCoordinatesVisualizer(*, features, hue, rescale, theme)` | `parallel_coordinates_chart` | 10g | no model required; .fit(X, y) takes raw matrix |

### 8.3 No-model visualizers and the class-not-instance visualizer

`ClassBalanceVisualizer`, `Rank1DVisualizer`, `Rank2DVisualizer`, `ParallelCoordinatesVisualizer` skip ModelSource entirely — they operate on raw X (and y for some). `ElbowVisualizer` is a related case: it takes a **model class**, not a fitted instance, and fits one model per `k` value inside its own `fit()`. All five subclass `FerrumVisualizer` and override `fit()`. The base-class default would otherwise try to construct `ModelSource(self.model, X, y)`, which fails on a class (no `predict` attribute on the class object) or on `None`. Pattern:

```python
class Rank1DVisualizer(FerrumVisualizer):
    def __init__(self, *, algorithm="shapiro", top_k=None, theme=None):
        super().__init__(model=None, theme=theme)
        self.algorithm = algorithm
        self.top_k = top_k

    def fit(self, X, y=None) -> Self:
        # Skip ModelSource; compute directly via _diagnostics.stats
        from ferrum._diagnostics.stats import rank1d_compute
        df = rank1d_compute(X, algorithm=self.algorithm, top_k=self.top_k)
        self._chart = _rank1d_chart_from_dataframe(df, theme=self.theme)
        self._fitted = True
        return self


class ElbowVisualizer(FerrumVisualizer):
    def __init__(self, model_class, *, ks: Sequence[int], metric: str = "distortion",
                 random_state: int | None = None, theme: Theme | None = None):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.model_class = model_class  # class, not instance
        self.ks = list(ks)
        self.metric = metric

    def fit(self, X, y=None) -> Self:
        # Fit one model per k value; ModelSource is not used (we manage models manually).
        scores = []
        for k in self.ks:
            model = self.model_class(n_clusters=k, random_state=self.random_state).fit(X)
            scores.append({"k": k, "score": _elbow_score(model, X, metric=self.metric)})
        df = pl.DataFrame(scores)
        self._chart = _elbow_chart_from_dataframe(df, theme=self.theme)
        self._fitted = True
        return self
```

---

## 9. Sub-batch contents in detail

### 9.1 Sub-batch 10a — Foundation + regression diagnostics

**Goal:** ship the ModelSource adapter shell, all infrastructure, the schema commitments (compare-ready from day one), the Rust-side `numeric_precision` field, and the first two regression-domain figures end-to-end to validate the pattern.

**New code:**
- `pyproject.toml` adds `[project.optional-dependencies]` `models`, `shap`, `umap`, `ml-all` entries.
- `crates/ferrum-core/src/render/svg.rs` — `numeric_precision` support (~10 lines).
- `crates/ferrum-core/src/spec/render.rs` — `RenderConfig.numeric_precision: Option<u8>` field.
- `src/ferrum/_diagnostics/{__init__,source,deps,schemas,charts,stats}.py` — module skeletons.
- `src/ferrum/_diagnostics/source.py` — `ModelSource` class with constructor, capability detection, cache.
- `src/ferrum/_diagnostics/source.py` — methods: `.predictions()`, `.probabilities()`.
- `src/ferrum/_diagnostics/stats.py` — `studentized_residual(y_true, y_pred, X)` (linear-estimator path).
- `src/ferrum/marks/diagnostic.py` — `mark_residuals`, `mark_prediction_error` (desugars).
- `src/ferrum/_diagnostics/charts.py` — `_residuals_chart_from_source`, `_prediction_error_chart_from_source`.
- `src/ferrum/_diagnostics/visualizers/regression.py` — `ResidualsVisualizer`, `PredictionErrorVisualizer`, `CooksDistanceVisualizer`.
- `src/ferrum/figures.py` — `residuals_chart`.
- `src/ferrum/__init__.py` — re-export ModelSource, residuals_chart, three regression visualizers.

**New tests:**
- `tests/diagnostics/test_no_sklearn_at_import.py` — regression test from §3.2.
- `tests/diagnostics/test_source.py` — ModelSource constructor + capability detection.
- `tests/diagnostics/test_regression.py` — `mark_residuals`, `mark_prediction_error` end-to-end.
- `tests/fixtures/build.py` — first invocation builds `binary_logistic.skops`, `regression_ridge.skops`, `regression_rf.skops`.
- `tests/goldens/phase_10/byte_identical/residuals_chart.svg` — first golden.

**Done when:** `residuals_chart(ridge_model, X, y).show()` renders byte-identically across macOS-arm64 and CI Linux-x86, `import ferrum` doesn't load sklearn, all new tests pass.

### 9.2 Sub-batch 10b — Classification curves

**Goal:** ship six classification curve diagnostics (ROC, PR, calibration, gain, lift, discrimination threshold) including multi-class averaging.

**New code:**
- `ModelSource.roc_curve(*, average, drop_intermediate)`, `.pr_curve(*, average)`, `.calibration_curve(*, n_bins, strategy)`, `.cumulative_gain()`, `.lift_curve()`, `.discrimination_threshold(*, n_thresholds, cv)`.
- Marks: `mark_roc`, `mark_pr`, `mark_calibration`, `mark_gain`, `mark_lift`, `mark_discrimination_threshold`.
- Chart builders: `_roc_chart_from_source`, `_pr_chart_from_source`, `_calibration_chart_from_source`, `_gain_chart_from_source`, `_lift_chart_from_source`, `_discrimination_threshold_chart_from_source`.
- Figures: `roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart`.
- Visualizers: `ROCVisualizer`, `PRVisualizer`, `CalibrationVisualizer`, `DiscriminationThresholdVisualizer`.

**New tests:** ~12 tests across binary + multiclass fixtures; goldens go to `byte_identical/` (all six produce deterministic outputs from pre-fit serialized models).

**Done when:** six figure functions render correctly for the binary classifier fixture and the 3-class multiclass fixture; multi-class averaging modes produce correct extra-row outputs per the schema.

### 9.3 Sub-batch 10c — Classification matrices

**Goal:** ship the two classification-error matrix diagnostics: confusion and class-prediction-error.

**New code:**
- `ModelSource.confusion_matrix(*, normalize)`.
- Marks: `mark_confusion`, `mark_class_prediction_error`.
- Chart builders: `_confusion_chart_from_source`, `_class_prediction_error_chart_from_source`, `_classification_report_chart` (uses mark_rect + text).
- Figures: `confusion_matrix_chart`, `class_prediction_error_chart`.
- Visualizers: `ConfusionMatrixVisualizer`, `ClassificationReportVisualizer`, `ClassPredictionErrorVisualizer`, `ClassBalanceVisualizer`.

**New tests:** ~6 tests; both binary and multiclass goldens.

### 9.4 Sub-batch 10d — Feature importance + SHAP + PDP

**Goal:** ship feature-explanation diagnostics. Includes the first optional-extra (`ferrum[shap]`).

**New code:**
- `ModelSource.importances(*, method, n_repeats, scoring, random_state)`, `.shap_values(*, background, max_evals)`, `.partial_dependence(features, *, grid_resolution, kind)`.
- Marks: `mark_importance`, `mark_shap_beeswarm`, `mark_shap_bar`, `mark_shap_waterfall`, `mark_pdp`.
- Chart builders: `_importance_chart_from_source`, `_shap_beeswarm_chart_from_source`, `_shap_bar_chart_from_source`, `_shap_waterfall_chart_from_source`, `_pdp_chart_from_source`.
- Figures: `importance_chart`, `shap_chart` (dispatches by `kind`).
- Visualizers: `FeatureImportancesVisualizer`, `SHAPVisualizer`.

**New tests:** ~10 tests; SHAP tests use `LinearExplainer` (deterministic, byte-identical). Permutation-importance tests go to `quantized_4dp/`.

### 9.5 Sub-batch 10e — Model selection / CV curves

**Goal:** ship CV-based diagnostics (learning curve, validation curve, CV scores, alpha selection).

**New code:**
- `ModelSource.learning_curve(*, cv, scoring, train_sizes)`, `.validation_curve(param, values, *, cv, scoring)`, `.cv_scores(*, cv, scoring)`, `.alpha_selection(alphas, *, cv, scoring)`.
- Marks: `mark_learning_curve`, `mark_validation_curve`, `mark_alpha_selection`, `mark_cv_scores`.
- Chart builders, figures, visualizers.

**New tests:** ~8 tests; all four go to `quantized_4dp/` because CV splitting introduces solver iteration variance.

### 9.6 Sub-batch 10f — Clustering / manifold / decision boundary

**Goal:** ship clustering and dimensionality-reduction diagnostics. Includes second optional-extra (`ferrum[umap]`).

**New code:**
- `ModelSource.silhouette(k)`, `.intercluster_distance(k, *, method)`, `.embeddings(*, method, n_components, **method_kwargs)`, `.pca_variance(*, n_components)`.
- Marks: `mark_silhouette`, `mark_intercluster_distance`, `mark_pca_scree`, `mark_decision_boundary`.
- Chart builders, figures (`cluster_diagnostics`, `intercluster_distance_chart`, `pca_scree_chart`, `decision_boundary_chart`), visualizers.

**New tests:** ~10 tests. Silhouette + pca_scree go to `byte_identical/`. UMAP / MDS / t-SNE / decision boundary go to `quantized_4dp/`.

### 9.7 Sub-batch 10g — Feature ranking + parallel coordinates

**Goal:** ship feature-analysis diagnostics. Introduces the one new Rust function (`kendall_tau_b`).

**New code:**
- `crates/ferrum-core/src/diagnostics.rs` — Knight's O(n log n) Kendall τ-b (~100 LOC).
- `src/ferrum/_core.pyi` — type stub for `kendall_tau_b`.
- `src/ferrum/_diagnostics/stats.py` — `pearson_r`, `spearman_rho`, `shapiro_w`, `kendall_tau_b` (Python wrapper calling Rust), `variance_rank`, `covariance_rank`.
- `ModelSource.rank1d(*, algorithm)`, `.rank2d(*, algorithm)`.
- Marks: `mark_rank1d`, `mark_rank2d`, `mark_parallel_coordinates`.
- Chart builders: `_rank1d_chart_from_dataframe`, `_rank2d_chart_from_dataframe`, `_parallel_coords_chart_from_dataframe`.
- Figures: `rank_chart`, `parallel_coordinates_chart`.
- Visualizers: `Rank1DVisualizer`, `Rank2DVisualizer`, `ParallelCoordinatesVisualizer`.

**New tests:** ~8 tests including a parity test (`tests/diagnostics/test_stats.py`) that validates in-house `shapiro_w` and `kendall_tau_b` against scipy on a fixture grid with `abs_diff ≤ 1e-10`. All goldens byte-identical (computation is deterministic).

### 9.8 Sub-batch 10h — Finalize

**Goal:** ship `ModelSource.compare`, consolidated spec drift notes, comprehensive golden coverage, Phase 10 marked done.

**New code:**
- `ModelSource.compare({...}) -> ComparedModelSource` class method.
- `ComparedModelSource` class with same method surface as ModelSource.
- `_resolve_source` helper updated to dispatch on `dict[str, model]` argument.
- Every figure function passes `compare=` through transparently (already supported in their kwargs; just enabled).

**Validation:**
- Run full golden suite (~40 SVG goldens across `byte_identical/` and `quantized_4dp/`).
- Apply consolidated spec drift notes from §13 to `ferrum-spec.md`.
- Audit `PHASE_9_PLUS_MARKS` — should contain only `arc`, `image`, `geoshape`, `label`.
- Update `docs/superpowers/ferrum-phases.md` — mark Phase 10 row Status as **done** with the merge commit hash.

**Done when:** all goldens green, all done-criteria checkboxes pass, drift notes committed.

---

## 10. Dependency strategy

### 10.1 `pyproject.toml` extras

```toml
[project.optional-dependencies]
models = ["scikit-learn>=1.3"]
shap = ["scikit-learn>=1.3", "shap>=0.42"]
umap = ["scikit-learn>=1.3", "umap-learn>=0.5"]
ml-all = ["scikit-learn>=1.3", "shap>=0.42", "umap-learn>=0.5"]

# dev-only (NOT exposed to end users — used in CI parity tests + fixture build)
[tool.uv]
dev-dependencies = [
    # ... existing dev deps ...
    "scipy>=1.10",            # parity tests for shapiro_w, kendall_tau_b
    "skops>=0.9",             # secure serialization of fitted sklearn models for fixtures
]
```

**scipy is intentionally NOT in any user-facing extra.** Phase 10 in-house statistics (Pearson, Spearman, Shapiro-Wilk W, studentized residual) are pure NumPy. Kendall τ-b is Rust. scipy stays as a CI dev dep used only for parity tests on `shapiro_w` and `kendall_tau_b`.

`skops` is also dev-only — it is used by `tests/fixtures/build.py` to serialize fitted sklearn models, and by the test harness to load them. End users never need skops.

**sklearn version pin for fixtures.** sklearn can change `predict()` outputs across minor versions even with identical `random_state=0` (default solver swaps, convergence tolerances, default parameter changes). To prevent silent fixture invalidation from a transitive sklearn upgrade, the dev/test environment pins a specific sklearn version. The pinned version lives in `tests/fixtures/SKLEARN_VERSION` (a single-line file, e.g. `1.7.2`). `tests/fixtures/build.py` reads this file and aborts if the installed `sklearn.__version__` doesn't match. The session-level fixture in `tests/conftest.py` performs the same check and aborts the test session if the installed version drifts from the pin. Upgrading sklearn is a deliberate operation: bump the pinned version → re-run `build.py` → regenerate goldens via `pytest --regenerate-goldens`.

### 10.2 Lazy import policy

Three lazy-import helpers in `src/ferrum/_diagnostics/deps.py`:

```python
def require_sklearn(method_name: str) -> "module":
    """Lazy-import sklearn or raise ImportError with `pip install ferrum[models]` hint."""

def require_shap(method_name: str) -> "module":
    """Lazy-import shap or raise ImportError with `pip install ferrum[shap]` hint."""

def require_umap(method_name: str) -> "module":
    """Lazy-import umap or raise ImportError with `pip install ferrum[umap]` hint."""
```

Every `ModelSource` method that needs a third-party lib calls the corresponding helper as its first line. `import ferrum` and `ModelSource.__init__` never trigger any of these.

### 10.3 CI matrix

CI installs `ferrum[ml-all]` + dev deps (which includes scipy and skops) — every Phase 10 test runs unconditionally. `tests/conftest.py` adds a session-level fixture that imports sklearn, shap, umap, scipy, skops and fails the test session loudly if any are missing.

End users on `pip install ferrum` get a working ferrum with zero new deps. End users on `pip install ferrum[models]` get sklearn. End users on `pip install ferrum[ml-all]` get everything Phase 10 supports.

---

## 11. Determinism and golden strategy

### 11.1 Two-tier goldens

**Tier 1 — `tests/goldens/phase_10/byte_identical/`**

SVG byte-identical across macOS-arm64, Linux-x86, and CI. Achieved by:

1. **Pre-fit serialized models** in `tests/fixtures/models/` eliminate platform-variant training. `predict()` and `predict_proba()` are bit-deterministic given fixed model weights.
2. The downstream compute (sort, cumsum, group_by, count) is bit-deterministic in NumPy/polars.
3. SVG renderer is byte-deterministic (Phase 9 invariant) when `numeric_precision=None`.

Figures in this tier: residuals_chart, prediction_error, roc_chart, pr_chart, confusion_matrix_chart, calibration_chart, gain_chart, lift_chart, class_prediction_error_chart, rank2d (all algorithms including Kendall, which is integer-tie-counted), pca_scree_chart (reads pre-computed `explained_variance_ratio_`), parallel_coordinates_chart, pdp_chart (deterministic given fitted model), discrimination_threshold_chart, cv_scores_chart (uses fixed CV split fixtures), rank1d (all algorithms), importance_chart (`method="builtin"`), shap_chart (`kind="bar"` / `kind="beeswarm"` over LinearExplainer — deterministic), silhouette plot.

Count: ~19 figures × 2 fixture flavors (binary/regression + multiclass/clustering as appropriate) ≈ ~25 byte-identical goldens.

**Tier 2 — `tests/goldens/phase_10/quantized_4dp/`**

SVG numeric output rounded to 4 decimal places via `RenderConfig(numeric_precision=4)`. Tolerates solver-iteration and BLAS-reduction-order variance across platforms.

Figures in this tier: shap_chart (`kind="waterfall"` if using KernelExplainer on a non-linear model), learning_curve_chart, validation_curve_chart, alpha_selection_chart, importance_chart (`method="permutation"`), decision_boundary_chart (when model is stochastic), intercluster_distance_chart (MDS/t-SNE), cluster_diagnostics (UMAP/MDS branches), embeddings/ManifoldVisualizer (UMAP).

Count: ~10 figures ≈ ~12 quantized goldens.

### 11.2 Pre-fit fixture script

`tests/fixtures/build.py` is a one-shot script that builds every fitted model and serializes it via `skops.io.dump` to `tests/fixtures/models/<model_name>.skops`. The resulting `.skops` files are committed to the repo.

**Why skops, not pickle:** `skops` was created by sklearn maintainers specifically to provide a safer alternative to pickle for fitted scikit-learn estimators. It refuses to deserialize arbitrary Python types — only sklearn estimators and a documented allow-list of supporting NumPy/SciPy/Pandas types. Pickle's arbitrary-code-execution risk is eliminated. The performance characteristics are equivalent (skops uses Python's pickle module under the hood for the *serialization*, but adds type validation on *deserialization*).

The script is invoked only when:

- A new fixture is added (a new test needs a new model class).
- The pinned sklearn version (in `tests/fixtures/SKLEARN_VERSION`) is bumped — a deliberate operation that requires regenerating both fixtures and downstream goldens.
- A pinned dep upgrade changes the skops on-disk format.

The script seeds every model with `random_state=0`, uses a fixed train/test split, and writes the `.skops` file. The Phase 10 test harness loads them via `skops.io.load(path, trusted=ALLOWED_SK_TYPES)` where `ALLOWED_SK_TYPES` is an explicit allowlist defined in `tests/fixtures/__init__.py`.

Goldens are regenerated by running `pytest --regenerate-goldens` which re-renders charts from the same `.skops` files (no model re-fitting).

### 11.3 0-skip, 0-xfail discipline

Phase 9 finalize closed at 0 new xfails and 0 new skips. Phase 10 matches this. CI installs `ferrum[ml-all]` + scipy + skops via dev deps, so every test runs every time. The session-level fixture in `tests/conftest.py`:

```python
@pytest.fixture(scope="session", autouse=True)
def _require_phase_10_extras():
    """Phase 10 tests assume sklearn/shap/umap/scipy/skops are all available."""
    missing = []
    for mod in ("sklearn", "shap", "umap", "scipy", "skops"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        pytest.exit(
            f"Phase 10 test suite requires: {', '.join(missing)}. "
            f"Install with `uv sync --extra ml-all` plus dev deps "
            f"(`pip install ferrum[ml-all] scipy skops`).",
            returncode=1,
        )
```

End users running their own subset of tests against a partial install hit the same fail-fast message.

---

## 12. Testing strategy

### 12.1 Test inventory by sub-batch

| Sub-batch | Unit tests | Integration tests | Goldens |
|---|---|---|---|
| 10a | test_source.py (5-6), test_no_sklearn_at_import.py (3) | test_regression.py (3) | residuals_chart, prediction_error (byte-id) |
| 10b | — | test_classification.py (12) | roc, pr, calibration, gain, lift, disc_threshold × 2 fixtures (byte-id) |
| 10c | — | test_classification.py (6) | confusion, class_prediction_error × 2, classification_report (byte-id) |
| 10d | — | test_explanation.py (10) | importance(builtin), shap(bar+beeswarm, Linear), pdp (byte-id); importance(permutation), shap(waterfall) (quantized) |
| 10e | — | test_selection.py (8) | learning, validation, cv_scores(fixed), alpha (quantized) |
| 10f | — | test_clustering.py (10) | silhouette, pca_scree (byte-id); UMAP, t-SNE, MDS, intercluster, decision_boundary (quantized) |
| 10g | test_stats.py (8) | test_ranking.py (6) | rank1d (all algos), rank2d (all algos), parallel_coords (byte-id) |
| 10h | — | test_compare.py (8) | every figure with `compare={"a":..., "b":...}` smoke goldens |

Totals: ~30 unit tests + ~60 integration tests + ~40 goldens ≈ ~130 new tests in Phase 10.

### 12.2 In-house statistics parity tests

`tests/diagnostics/test_stats.py` validates the four in-house implementations against scipy:

- `pearson_r` vs `scipy.stats.pearsonr` over 20 random fixtures: tolerance `1e-12`.
- `spearman_rho` vs `scipy.stats.spearmanr` over 20 random fixtures: tolerance `1e-12`.
- `shapiro_w` vs `scipy.stats.shapiro(...).statistic` over 20 fixtures covering `n ∈ {10, 50, 200, 1000}` and four distributions (normal, uniform, exponential, bimodal): tolerance `1e-10`.
- `kendall_tau_b` (Rust) vs `scipy.stats.kendalltau` over 20 fixtures covering `n ∈ {10, 100, 1000}` and tie densities `{0%, 10%, 50%}`: tolerance `1e-12`.

scipy is imported only at test time, never at production runtime.

### 12.3 Mark coverage assertion

`tests/diagnostics/test_mark_coverage.py`:

```python
def test_phase_10_marks_implemented():
    PHASE_10_MARKS = frozenset([
        "residuals", "prediction_error", "confusion", "roc", "pr", "calibration",
        "gain", "lift", "importance", "shap_beeswarm", "shap_bar", "shap_waterfall",
        "pdp", "silhouette", "learning_curve", "validation_curve", "decision_boundary",
        "discrimination_threshold", "parallel_coordinates", "class_prediction_error",
        "pca_scree", "rank1d", "rank2d", "intercluster_distance", "cv_scores",
        "alpha_selection",
    ])
    from ferrum.marks.deferred import PHASE_9_PLUS_MARKS, PHASE_8B_MARKS
    overlap = PHASE_10_MARKS & (PHASE_9_PLUS_MARKS | PHASE_8B_MARKS)
    assert not overlap, f"Phase 10 marks still in deferred list: {overlap}"
    for mark_name in PHASE_10_MARKS:
        mark_cls = getattr(ferrum, f"mark_{mark_name}", None)
        assert mark_cls is not None, f"ferrum.mark_{mark_name} not exported"
```

---

## 13. Spec drift notes (to apply to `ferrum-spec.md` in 10h)

All dated `2026-MM-DD (Phase 10)` consistent with Phase 9's pattern.

### §1 (Philosophy)

Note that Phase 10 places model-diagnostic compute in the `ModelSource` adapter (Python, lazy-imported sklearn delegation) rather than as Rust transforms in the rendering pipeline. Rationale per §1.5 of this design doc. This is a deliberate departure, not silent drift.

### §3.1 ModelSource

Add `random_state: int | None = None` to the constructor signature. Document that `random_state` is propagated to every method that wraps an RNG-using sklearn/shap/umap call.

### §3.3 Marks

Apply the per-mark drift notes from §6.2:
- `mark_residuals`: clarify studentized residual scope (linear estimators).
- `mark_confusion`: note Phase 8b `ColorScale::Continuous` is the color scale; colorbar legend is Phase 11+.
- `mark_decision_boundary`: clarify it requires exactly 2 features.
- `mark_shap_*`: note `ferrum[shap]` requirement.
- `mark_rank2d`: note `kendall` uses Rust Knight algorithm.
- `mark_pca_scree`: clarify that `mark_line` for cumulative is overlay-on-bar.

### §3.14 Group B figure functions

Document that every figure function accepts `random_state: int | None = None`. List per-function whether `random_state` actually affects compute:

| Function | random_state effect |
|---|---|
| roc_chart, pr_chart, confusion_matrix_chart, calibration_chart, gain_chart, lift_chart, residuals_chart, pca_scree_chart, parallel_coordinates_chart, class_prediction_error_chart, decision_boundary_chart, intercluster_distance_chart (MDS only) | accepted as forward-compat; does not affect output |
| importance_chart (`method="permutation"`), shap_chart (KernelExplainer), learning_curve_chart, validation_curve_chart, cv_scores_chart, alpha_selection_chart, cluster_diagnostics, discrimination_threshold_chart (when `cv` provided), intercluster_distance_chart (`method="tsne"`) | propagated to underlying RNG; affects output |

### §3.15 Visualizers

Add `random_state: int | None = None` to every Visualizer constructor for uniform API.

### §3.16 RenderConfig

Add `numeric_precision: int | None = None`. Default `None` keeps current SVG numeric formatting. When set to an integer `p ∈ [1, 12]`, all float coordinates in emitted SVG are rounded to `p` decimal places. Used by Phase 10 quantized goldens to absorb cross-platform solver variance.

---

## 14. Acceptance criteria (Phase 10 done = all of the following)

- [ ] `ModelSource(model, X, y, ...)` wraps any object exposing the sklearn estimator protocol (predict/predict_proba/transform/etc.) via duck-typed attribute presence.
- [ ] `import ferrum` does not load sklearn, shap, or umap-learn.
- [ ] `ModelSource(non_sklearn_model, X, y)` does not load sklearn.
- [ ] All 26 model-diagnostic marks from `ferrum-spec.md §3.3` are implemented in `src/ferrum/marks/diagnostic.py` and exported from `ferrum`.
- [ ] All 21 figure functions from `ferrum-spec.md §3.14` Group B render correctly on canned fixtures and are exported from `ferrum`.
- [ ] All 25 Visualizer classes from `ferrum-spec.md §3.15` implement `fit`/`score`/`show`/`__repr__` and are exported from `ferrum`.
- [ ] `ModelSource.compare({...})` produces a `ComparedModelSource`; every figure function works transparently with both.
- [ ] All 40+ Phase 10 SVG goldens pass (~25 byte-identical, ~12 quantized).
- [ ] `cargo test` passes for `kendall_tau_b` and `numeric_precision`.
- [ ] `pytest` passes with 0 new skips and 0 new xfails on top of Phase 9's totals.
- [ ] `pyproject.toml` has the four new extras (`models`, `shap`, `umap`, `ml-all`); base install picks up zero new deps.
- [ ] All spec drift notes in §13 of this doc are applied to `ferrum-spec.md` with `2026-MM-DD (Phase 10)` date tags.
- [ ] `PHASE_9_PLUS_MARKS` audit passes: only `arc`, `image`, `geoshape`, `label` remain.
- [ ] `ferrum-phases.md` Phase 10 row status updated to **done** with the Phase 10 merge commit hash.

---

## Appendix A — Why scipy is not a runtime dependency

The first-pass design listed scipy as part of `[models]` for three uses: t-distribution CDF (studentized residual p-values), Shapiro-Wilk, and Kendall τ. Re-analysis at brainstorming time:

1. The spec schema for `.predictions()` does not include a p-value column; studentized residual is just the statistic, no t-distribution CDF needed. **Drops scipy use #1.**
2. Shapiro-Wilk W can be implemented in vectorized NumPy in ~120 LOC via Royston's 1992 algorithm. Pearson, Spearman, variance ranking, covariance ranking are trivial vectorized NumPy. **Drops scipy use #2.**
3. Kendall τ-b naive O(n²) in NumPy broadcast OOMs at n > 10k. Knight's O(n log n) in Rust is ~100 LOC and an order of magnitude faster than Python at scale. **Replaces scipy use #3 with a Rust function.**

scipy remains a dev-only dependency for parity testing.

## Appendix B — Why kendall_tau_b is in Rust but shapiro_w is not

Rust beats NumPy when the inner kernel is **scalar-with-branching** or **irregular memory access** (pairwise comparisons, hash lookups, conditional accumulation). Rust ties NumPy when the kernel is already vectorized (BLAS GEMM/GEMV, sort, element-wise ops).

- **Shapiro-Wilk W:** vectorized as `np.sort(X, axis=0)` (one Timsort call) followed by `coeffs @ sorted` (one BLAS GEMV). For n=10k × 100 features: ~5-10ms. Rust does the same operations in essentially the same time (same SIMD-vectorized routines underneath). No win.
- **Kendall τ-b:** the inner kernel is "count concordant/discordant pairs". Naive O(n²) is vectorizable via broadcast but allocates an n×n boolean matrix (800MB at n=10k). Knight's O(n log n) is a merge-sort variant — sequential, sort-based, NOT vectorizable. Pure Python Knight is ~10× slower than Rust Knight. At n=100k, Rust Knight runs in milliseconds; Python Knight in seconds; NumPy broadcast OOMs. **Rust wins decisively.**

The single Rust addition for Phase 10 is targeted at the one place Rust gives a measurable user-visible win. Everything else stays Python because the asymptotic win isn't there.

## Appendix C — Fixture serialization rationale (skops vs pickle)

The fixture-model strategy in §11 requires serializing fitted sklearn estimators to disk so that downstream `predict()` / `predict_proba()` calls produce bit-identical outputs across platforms. Three serialization formats were considered:

1. **Pickle.** sklearn's traditional default. Simple, works for every estimator. Security risk: deserialization executes arbitrary Python code, so loading a malicious `.pkl` is equivalent to running arbitrary code. While Phase 10 fixtures are repo-committed (not user-supplied), the precedent of "load `.pkl` files in CI" is worth avoiding.
2. **skops.** Created by sklearn maintainers to address pickle's security gap. Uses pickle under the hood for serialization but adds type validation on deserialization — only sklearn estimators and a documented allowlist of NumPy/SciPy/Pandas types are accepted. Loading a malicious `.skops` file raises an error rather than executing code. Performance is essentially equivalent to pickle. **Chosen.**
3. **ONNX.** Cross-framework, no security risk. Conversion overhead and dependency footprint are heavier; some sklearn estimators (notably `Pipeline` with custom transformers) require additional converters. Overkill for fixture purposes.

The chosen approach: serialize with `skops.io.dump(model, path)`; load with `skops.io.load(path, trusted=ALLOWED_SK_TYPES)` where the allowlist is an explicit module-level constant in `tests/fixtures/__init__.py`. This eliminates the arbitrary-code-execution risk of pickle while preserving the determinism guarantee. `skops` is a dev/test-only dependency — end users never need to install it.
