# Native Decomposition and Clustering Statistics

**Date:** 2026-05-13
**Status:** Proposed

## Problem

Ferrum's clustering and dimensionality-reduction diagnostics delegate all numerical work to sklearn, which means:

1. **PCA** (`embeddings(method="pca")` and `pca_variance`) requires a pre-fitted sklearn `PCA` estimator. Users cannot call `pca_scree_chart(X)` on raw data — they must first `PCA(n_components=k).fit(X)` and pass the fitted model. The variance ratios are read from `model.explained_variance_ratio_`, not computed by ferrum.

2. **MDS embedding** (`intercluster_distance(method="mds")`) imports `sklearn.manifold.MDS` for a classical MDS that is mathematically just eigendecomposition of the double-centered distance Gram matrix.

3. **Silhouette** (`silhouette()`, `silhouette_score()`) imports `sklearn.metrics.silhouette_samples` and `silhouette_score` for what is fundamentally a pairwise-distance computation + per-cluster aggregation — infrastructure `linkage.rs` already has.

4. **Calinski-Harabasz** (`ElbowVisualizer` with `metric="calinski_harabasz"`) imports `sklearn.metrics.calinski_harabasz_score` for a between-cluster / within-cluster variance ratio that reduces to a few matrix operations.

All four cross the Python→numpy boundary unnecessarily when the data is already in Arrow on the Rust side. At 500K+ rows, the copy overhead and Python-side computation are the bottleneck.

## Decision

Move PCA, classical MDS, silhouette, and Calinski-Harabasz into Rust using `faer` (already linked) and `linkage.rs`'s pairwise distance infrastructure. t-SNE and UMAP stay in Python — they are iterative gradient-descent algorithms that `faer` cannot accelerate.

## What stays in Python

| Method | Why |
|---|---|
| t-SNE | Iterative Barnes-Hut gradient descent. No decomposition analogue. |
| UMAP | Iterative stochastic gradient optimization on fuzzy simplicial sets. |
| `sklearn.cluster.KMeans.fit()` | Iterative Lloyd's algorithm. ferrum wraps fitted models, doesn't refit them. |
| `sklearn.cluster.AgglomerativeClustering.fit()` | Already using Rust `kodama` via linkage transform for the dendrogram; the sklearn path exists for the `cluster_diagnostics` sweep loop which fits multiple k values. |

## Changes

### 1. Rust `pca` function

New PyO3-exposed function in `transform/stats.rs`:

```rust
#[pyfunction]
pub fn pca(
    _py: Python<'_>,
    x_table: PyRecordBatch,         // n x p feature matrix
    n_components: Option<usize>,    // default: min(n, p)
) -> PyResult<PyRecordBatch>
```

Algorithm:
1. Extract `n x p` matrix from Arrow RecordBatch.
2. Center columns (subtract column means).
3. Compute thin SVD via `faer`: `X_centered = U S V'`.
4. Scores: `U * S` truncated to `k = n_components` columns.
5. Explained variance ratio: `S_i^2 / sum(S^2)` for each component.
6. Cumulative variance ratio: running sum of explained variance ratio.

Returns a RecordBatch with columns:
- `dim_0` … `dim_{k-1}`: Float64 score columns (n rows)
- `explained_variance_ratio`: Float64 (k rows, padded with null for n > k — or returned as a separate batch)

**Design choice:** Return two outputs — a scores RecordBatch (n rows) and a variance RecordBatch (k rows) — since they have different row counts. Expose as two functions:

```rust
#[pyfunction]
pub fn pca_scores(x_table, n_components) -> PyRecordBatch;  // n x k scores

#[pyfunction]
pub fn pca_variance(x_table, n_components) -> PyRecordBatch; // k rows: component, explained_variance_ratio, cumulative_variance_ratio
```

Both call the same internal SVD; `pca_scores` drops the variance info, `pca_variance` drops the scores. The SVD is computed once if both are needed in the same session (callers can cache).

### 2. Rust `mds_classical` function

Classical MDS on a precomputed distance matrix or raw feature matrix:

```rust
#[pyfunction]
pub fn mds_classical(
    _py: Python<'_>,
    x_table: PyRecordBatch,   // n x p feature matrix (or n x n distance matrix)
    n_components: usize,       // output dimensionality (usually 2)
    is_distance: bool,         // if true, x_table is already a distance matrix
    metric: &str,              // "euclidean" (default) — only used when is_distance=false
) -> PyResult<PyRecordBatch>
```

Algorithm:
1. If `!is_distance`: compute condensed distance matrix using `linkage.rs`'s `condensed_distances` (reuse existing infrastructure).
2. Convert condensed to full symmetric n x n distance matrix D.
3. Double-center: `B = -0.5 * J * D^2 * J` where `J = I - (1/n) * 11'`.
4. Eigendecomposition of B via `faer::SelfAdjointEigen`.
5. Take top-`n_components` eigenvectors (largest eigenvalues).
6. Coordinates: `X_embed = V_k * diag(sqrt(lambda_k))`.

Returns a RecordBatch with columns `dim_0` … `dim_{n_components-1}` (n rows).

### 3. Rust `silhouette_samples` function

Per-sample silhouette values, replacing `sklearn.metrics.silhouette_samples`:

```rust
#[pyfunction]
pub fn silhouette_samples(
    _py: Python<'_>,
    x_table: PyRecordBatch,   // n x p feature matrix
    labels: PyArray,           // Int64 or Float64 cluster labels (n elements)
    metric: &str,              // "euclidean" (default)
) -> PyResult<PyArray>         // Float64 silhouette values (n elements)
```

Algorithm:
1. For each sample i with label c_i:
   - a(i) = mean distance to all other samples in cluster c_i
   - b(i) = min over clusters c != c_i of (mean distance to all samples in cluster c)
   - s(i) = (b(i) - a(i)) / max(a(i), b(i))
2. Uses `linkage.rs::DistanceMetric` for pairwise distances (reuse existing code).

Complexity: O(n^2 * feat) for the pairwise distances. Same as sklearn — but avoids the Python→numpy copy and runs on Arrow data already in the Rust process.

### 4. Rust `silhouette_score` function

Mean silhouette score (scalar), replacing `sklearn.metrics.silhouette_score`:

```rust
#[pyfunction]
pub fn silhouette_score(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    labels: PyArray,
    metric: &str,
) -> PyResult<f64>
```

Calls `silhouette_samples` internally, returns the mean. This is what `_cluster_diagnostics_chart` and `ElbowVisualizer(metric="silhouette")` use.

### 5. Rust `calinski_harabasz_score` function

Between-cluster / within-cluster variance ratio:

```rust
#[pyfunction]
pub fn calinski_harabasz_score(
    _py: Python<'_>,
    x_table: PyRecordBatch,
    labels: PyArray,
) -> PyResult<f64>
```

Algorithm:
1. Compute overall centroid `x_bar`.
2. For each cluster c with n_c samples and centroid `mu_c`:
   - Between-cluster: `B += n_c * ||mu_c - x_bar||^2`
   - Within-cluster: `W += sum_{i in c} ||x_i - mu_c||^2`
3. `CH = (B / (k - 1)) / (W / (n - k))` where k = number of clusters.

Pure column arithmetic — no decomposition needed, but benefits from Rust's avoid-the-copy advantage at scale.

### 6. Python caller rewiring

#### `_clustering.py`

| Method | Before | After |
|---|---|---|
| `pca_variance()` | Reads `model.explained_variance_ratio_` from fitted sklearn PCA | Two paths: (a) if model has `explained_variance_ratio_`, read it (backward compat); (b) new overload accepts raw X, calls `_core.pca_variance(x_arrow, n_components)` |
| `embeddings(method="pca")` | `sklearn.decomposition.PCA.fit_transform()` | `_core.pca_scores(x_arrow, n_components)` |
| `embeddings(method="tsne")` | `sklearn.manifold.TSNE.fit_transform()` | **Unchanged** (iterative) |
| `embeddings(method="umap")` | `umap.UMAP.fit_transform()` | **Unchanged** (iterative) |
| `silhouette()` | `sklearn.metrics.silhouette_samples()` | `_core.silhouette_samples(x_arrow, labels_arrow, metric)` |
| `intercluster_distance(method="mds")` | `sklearn.manifold.MDS.fit_transform()` | `_core.mds_classical(centers_arrow, n_components=2)` |
| `intercluster_distance(method="tsne")` | `sklearn.manifold.TSNE.fit_transform()` | **Unchanged** (iterative) |

#### `charts.py` — `_cluster_diagnostics_chart`

| Call site | Before | After |
|---|---|---|
| `silhouette_score(X_np, labels)` (line 2424) | sklearn | `_core.silhouette_score(x_arrow, labels_arrow, "euclidean")` |
| Manual inertia computation (lines 2415–2419) | numpy mask + centroid + squared distance | Keep as-is (runs inside a fit loop over multiple k values; the sklearn `KMeans.fit()` is the bottleneck, not the inertia reduction) |

#### `visualizers/clustering.py` — `ElbowVisualizer`

| Call site | Before | After |
|---|---|---|
| `silhouette_score(X_fit, labels)` (line 174) | sklearn | `_core.silhouette_score(x_arrow, labels_arrow, "euclidean")` |
| `calinski_harabasz_score(X_fit, labels)` (line 181) | sklearn | `_core.calinski_harabasz_score(x_arrow, labels_arrow)` |

#### `figures.py` — `pca_scree_chart`

New code path: when `model_or_source` is a raw DataFrame (not a fitted PCA or ModelSource), call `_core.pca_variance(x_arrow, n_components)` directly. This enables:

```python
fm.pca_scree_chart(X_train, n_components=10)  # no sklearn PCA needed
```

The existing path (`pca_scree_chart(fitted_pca, X, y)`) continues to work via `ModelSource.pca_variance()`.

## New Rust module layout

All new functions go in `transform/stats.rs` (extending the existing module):

```rust
// Existing (Tier 3):
pub fn hat_matrix_stats(...) -> ...;
pub fn studentized_residual_no_x(...) -> ...;
pub fn py_shapiro_w(...) -> ...;
pub fn py_rankdata_average(...) -> ...;
pub fn py_rank1d(...) -> ...;
pub fn py_rank1d_with_y(...) -> ...;
pub fn py_rank2d(...) -> ...;

// New (this spec):
pub fn pca_scores(...) -> ...;
pub fn pca_variance(...) -> ...;
pub fn mds_classical(...) -> ...;
pub fn silhouette_samples(...) -> ...;
pub fn silhouette_score(...) -> ...;
pub fn calinski_harabasz_score(...) -> ...;
```

Distance computation reuses `linkage.rs::condensed_distances` (make it `pub(crate)` if not already).

## Verification plan

### Parity tests

Each Rust implementation must match sklearn within tolerance on a sweep of test cases:

| Function | Reference | Tolerance | Test cases |
|---|---|---|---|
| `pca_scores` | `sklearn.decomposition.PCA.fit_transform()` | 1e-10 (signs may flip per-component) | n in {50, 500, 5000}, p in {5, 20, 50}, k in {2, 5, min(n,p)} |
| `pca_variance` | `sklearn.decomposition.PCA.explained_variance_ratio_` | 1e-12 | same sweep |
| `mds_classical` | `sklearn.manifold.MDS(normalized_stress="auto")` | Procrustes alignment + 1e-6 | n in {10, 50, 200}, p in {3, 10}, metric in {"euclidean"} |
| `silhouette_samples` | `sklearn.metrics.silhouette_samples()` | 1e-12 | n in {50, 200, 1000}, k in {2, 5, 10}, metric in {"euclidean"} |
| `silhouette_score` | `sklearn.metrics.silhouette_score()` | 1e-12 | same |
| `calinski_harabasz_score` | `sklearn.metrics.calinski_harabasz_score()` | 1e-10 | n in {50, 200}, k in {2, 5} |

**PCA sign convention:** SVD eigenvectors are determined up to a sign flip. The parity test must account for this by comparing `abs(ours)` vs `abs(theirs)` per component, or by applying a sign-correction step (flip columns where the max-absolute-value element differs in sign).

**MDS rotation:** Classical MDS solutions are determined up to rotation/reflection. Parity test uses Procrustes alignment before comparing coordinates.

### Integration tests

- `uv run pytest -x -q` — full suite, 0 failures.
- `pca_scree_chart(X_train, n_components=10)` renders without a fitted PCA model.
- `silhouette_chart` produces identical SVGs before and after (byte-compare goldens).
- `intercluster_distance_chart` with `method="mds"` produces visually correct 2D embeddings.

### Rust unit tests

- SVD on identity matrix returns identity eigenvectors.
- PCA on perfectly correlated columns returns one non-zero component.
- Silhouette of perfectly separated clusters returns 1.0 for all samples.
- Calinski-Harabasz of well-separated clusters >> poorly-separated clusters.
- MDS on a triangle (3 points with known distances) recovers the triangle geometry up to rotation.

## Risk

**Low for PCA** — thin SVD via faer is well-tested; the math is textbook.

**Low for silhouette / Calinski-Harabasz** — simple arithmetic over pairwise distances; `linkage.rs` distance code is already battle-tested.

**Medium for classical MDS** — eigendecomposition of the double-centered Gram matrix can produce negative eigenvalues when the distance metric is non-Euclidean. The implementation must handle this gracefully (drop negative eigenvalues, warn if they're large relative to the positive ones). sklearn's MDS uses SMACOF (iterative stress minimization) as default, not classical MDS — so exact parity with sklearn requires Procrustes alignment, and users switching from sklearn MDS may notice slightly different layouts.

## Non-goal

This spec does not implement:
- t-SNE (iterative Barnes-Hut; no decomposition analogue)
- UMAP (iterative stochastic optimization)
- Incremental/streaming PCA (out of scope; batch SVD is sufficient for ferrum's diagnostic use case)
- Sparse SVD / randomized SVD (faer's dense SVD is sufficient for the feature-count ranges in diagnostic charts — typically p < 100)
