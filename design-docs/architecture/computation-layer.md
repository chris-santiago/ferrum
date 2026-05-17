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
    |        uses faer for linalg (Cholesky, SVD, eigendecomposition, corrcoef)
    |        uses shared linalg.rs (invert_2x2, solve_3x3_spd, hat_diagonal)
    |        returns Arrow RecordBatch
    |
    |---> Rust stats (stats.rs)
    |        hat_matrix_stats, rank1d, rank2d, shapiro_w, rankdata_average,
    |        pca_scores, pca_variance, mds_classical, silhouette_samples/score,
    |        calinski_harabasz_score, tsne_embedding, umap_embedding
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
| `transform/stats.rs` | Model diagnostics statistics: hat matrix + studentized residuals + Cook's distance (single pass), Shapiro-Wilk W, rankdata, rank1d/rank2d, PCA (thin SVD), classical MDS (eigendecomposition), silhouette samples/score, Calinski-Harabasz, t-SNE and UMAP (via manifolds-rs) |
| `transform/smooth.rs` | LOESS -- calls `linalg::solve_3x3_spd` for degree-2 |
| `transform/glm.rs` | GLM IRLS -- calls `linalg::invert_2x2` |
| `transform/logistic.rs` | Logistic IRLS -- calls `linalg::invert_2x2` |
| `transform/robust.rs` | Huber M-estimation -- calls `linalg::invert_2x2`, `scale_2x2`, `xtx_unweighted` |
| `transform/linkage.rs` | Hierarchical clustering -- pairwise distance metrics (euclidean, cosine, correlation), also used by silhouette and MDS |
| `diagnostics.rs` | Kendall tau-b (merge-sort inversion count) |

### Python layer -- declaration, sklearn interop, and thin wrappers only

| Module | Responsibility |
|---|---|
| `_diagnostics/sources/_predictions.py` | Calls `_core.hat_matrix_stats` (one Rust call for stud + cooks + leverage) or `_core.studentized_residual_no_x` for non-linear models. Numpy used only for `model.predict()` return. |
| `_diagnostics/sources/_ranking.py` | Calls `_core.py_rank1d` / `py_rank1d_with_y` / `py_rank2d` via Arrow. Zero numpy. |
| `_diagnostics/sources/_clustering.py` | Calls `_core.pca_scores`, `_core.pca_variance`, `_core.mds_classical`, `_core.silhouette_samples`, `_core.tsne_embedding`, `_core.umap_embedding` via Arrow. Numpy only for `model.labels_` / `model.cluster_centers_` (sklearn return types). |
| `_diagnostics/_rank_helpers.py` | Coerces raw DataFrames to Arrow, calls Rust rank functions. Used by `figures.py` and `visualizers/ranking.py` for the no-ModelSource path. |
| `_diagnostics/charts.py` | Chart construction. Polars for R2/RMSE/MAE metrics, polars for SHAP cumsum. Numpy for decision-boundary mesh grid, AUC/AP/Brier integration (small curve-resolution arrays), and cluster diagnostics sweep (KMeans.fit loop). |
| `annotations.py` | Metric labels and outlier detection -- polars for endpoint detection and z-scoring. Numpy only inside AUC/AP/Brier helpers (lazy import). |
| `_direct_label.py` | Endpoint collision staggering -- pure polars. Zero numpy. |

## What numpy still does

All at sklearn boundaries or genuinely 2D operations:

1. `_predictions.py` -- `model.predict()` and `model.predict_proba()` return numpy (sklearn contract)
2. `_classification.py` (~49 calls) -- ROC/PR/calibration/gain/lift curve computation, CV split indexing, multi-class averaging. Largest remaining Python-side compute block; spec exists for moving these to Rust TransformSpec variants.
3. `_importance.py` -- SHAP explainer compat, sklearn `partial_dependence` integer column indexing
4. `_clustering.py` -- `model.labels_`, `model.cluster_centers_` (sklearn return types)
5. `charts.py` -- decision-boundary mesh grid (genuinely 2D numpy), cluster diagnostics sweep loop

## Rust dependencies for computation

### faer (0.24)

Pure Rust linear algebra, zero external dependencies. Used for:

- **Cholesky** (`Llt`) -- hat matrix diagonal via `(X'X)^{-1}`
- **Thin SVD** -- PCA scores and explained variance ratios
- **Eigendecomposition** (`SelfAdjointEigen`) -- classical MDS on double-centered Gram matrix
- **Matrix arithmetic** (`Mat<f64>`) -- `corrcoef` builds the Gram matrix via `Xc' * Xc` and normalizes

### manifolds-rs (0.2)

Pure Rust embedding algorithms, built on faer 0.23 (bridged via `faer-compat` renamed dep). Used for:

- **Barnes-Hut t-SNE** -- replaces `sklearn.manifold.TSNE`
- **UMAP** -- replaces `umap-learn` (no longer a runtime dependency)
- Uses `ann-search-rs` for HNSW/NNDescent approximate nearest neighbors
- Parallelized via `rayon`

### kodama (0.3)

Hierarchical clustering (Lance-Williams + nearest-neighbor chain). Used by `linkage.rs`.
