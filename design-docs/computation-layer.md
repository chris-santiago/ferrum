# Computation Layer Architecture

**Last updated:** 2026-05-13

## Data flow

```
User DataFrame (polars/pandas/numpy/pyarrow)
    |
    v
narwhals coercion (_coerce.py) --> pyarrow Table
    |
    v
Arrow CDI (zero-copy for polars/pyarrow)
    |
    |---> Rust transforms (smooth, glm, logistic, robust, contour, kde, ...)
    |        uses faer for linalg (Cholesky, corrcoef, hat diagonal)
    |        uses shared linalg.rs (invert_2x2, solve_3x3_spd, xtx_unweighted)
    |        returns Arrow RecordBatch
    |
    |---> Rust stats (stats.rs)
    |        hat_matrix_stats, rank1d, rank2d, shapiro_w, rankdata_average
    |        accepts PyRecordBatch/PyArray, returns PyRecordBatch/PyArray
    |
    |---> Rust diagnostics (diagnostics.rs)
    |        kendall_tau_b (Knight's O(n log n))
    |
    v
polars DataFrame <-- pl.from_arrow(result)
    |
    v
Rust renderer (SVG/PNG via Arrow CDI)
```

## What lives where

### Rust crate (ferrum-core) -- all numerical computation

| Module | Responsibility |
|---|---|
| `transform/linalg.rs` | Shared linear algebra: `invert_2x2`, `scale_2x2`, `xtx_unweighted`, `solve_3x3_spd` (hand-rolled fixed-size), `hat_diagonal`, `corrcoef`, `mat_from_flat` (faer-backed) |
| `transform/stats.rs` | Model diagnostics statistics: hat matrix + studentized residuals + Cook's distance (single pass), Shapiro-Wilk W, rankdata, rank1d/rank2d (pearson, spearman, kendall, covariance, variance) |
| `transform/smooth.rs` | LOESS -- calls `linalg::solve_3x3_spd` for degree-2 |
| `transform/glm.rs` | GLM IRLS -- calls `linalg::invert_2x2` |
| `transform/logistic.rs` | Logistic IRLS -- calls `linalg::invert_2x2` |
| `transform/robust.rs` | Huber M-estimation -- calls `linalg::invert_2x2`, `scale_2x2`, `xtx_unweighted` |
| `transform/linkage.rs` | Hierarchical clustering -- hand-rolled distance metrics (euclidean, cosine, correlation) |
| `diagnostics.rs` | Kendall tau-b (merge-sort inversion count) |

### Python layer -- declaration, sklearn interop, and thin wrappers only

| Module | Responsibility |
|---|---|
| `_diagnostics/sources/_predictions.py` | Calls `_core.hat_matrix_stats` (one Rust call for stud + cooks + leverage) or `_core.studentized_residual_no_x` for non-linear models. Numpy used only for `model.predict()` return. |
| `_diagnostics/sources/_ranking.py` | Calls `_core.py_rank1d` / `py_rank1d_with_y` / `py_rank2d` via Arrow. Zero numpy. |
| `_diagnostics/_rank_helpers.py` | Coerces raw DataFrames to Arrow, calls Rust rank functions. Used by `figures.py` and `visualizers/ranking.py` for the no-ModelSource path. |
| `_diagnostics/charts.py` | Chart construction. Polars for R2/RMSE/MAE metrics, polars for SHAP cumsum. Numpy only for decision-boundary mesh grid and AUC/AP/Brier integration (small curve-resolution arrays). |
| `annotations.py` | Metric labels and outlier detection -- polars for endpoint detection and z-scoring. Numpy only inside AUC/AP/Brier helpers (lazy import). |
| `_direct_label.py` | Endpoint collision staggering -- pure polars. Zero numpy. |

## What numpy still does (8 `.to_numpy()` calls)

All at sklearn boundaries or genuinely 2D operations:

1. `_predictions.py` -- `model.predict(self._X)` returns numpy (sklearn contract)
2. `_predictions.py` -- `model.predict_proba()` returns numpy
3. `_classification.py` (4 sites) -- CV split integer-array indexing, 2D matrix `.ravel()` for multi-class ROC/PR, calibration binning via `np.digitize`
4. `_importance.py` (2 sites) -- SHAP explainer compat, sklearn `partial_dependence` integer column indexing

## The faer foundation

Pure Rust linear algebra, zero external dependencies. Currently used for:

- **Cholesky** (`Llt`) -- hat matrix diagonal via `(X'X)^{-1}`
- **Matrix arithmetic** (`Mat<f64>`) -- `corrcoef` builds the Gram matrix via `Xc' * Xc` and normalizes

Ready for Phase 11+ (PCA via SVD, eigendecomposition for spectral methods) without any build-system changes.
