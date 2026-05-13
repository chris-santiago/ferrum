"""Parity tests: Rust-native PCA, silhouette, Calinski-Harabasz, MDS vs sklearn."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import polars as pl
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score as sklearn_ch
from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from ferrum import _core


def _make_arrow_batch(X: np.ndarray) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {f"f{j}": X[:, j].tolist() for j in range(X.shape[1])}
    )


def _make_labels_arrow(labels: np.ndarray) -> pa.Array:
    return pa.array(labels.astype(int).tolist(), type=pa.int64())


# ---------------------------------------------------------------------------
# PCA scores
# ---------------------------------------------------------------------------


class TestPCAScores:
    @pytest.mark.parametrize("n,p,k", [(50, 5, 2), (100, 10, 5), (200, 20, 3)])
    def test_parity_with_sklearn(self, n, p, k):
        rng = np.random.RandomState(42)
        X = rng.randn(n, p)
        x_arrow = _make_arrow_batch(X)
        result = _core.pca_scores(x_arrow, k)
        df = pl.from_arrow(result)

        pca = PCA(n_components=k).fit(X)
        sklearn_scores = pca.transform(X)

        for c in range(k):
            ours = np.abs(df[f"dim_{c}"].to_numpy())
            theirs = np.abs(sklearn_scores[:, c])
            np.testing.assert_allclose(ours, theirs, atol=1e-8)


# ---------------------------------------------------------------------------
# PCA variance
# ---------------------------------------------------------------------------


class TestPCAVariance:
    @pytest.mark.parametrize("n,p,k", [(50, 5, 2), (100, 10, 5), (500, 20, 10)])
    def test_parity_with_sklearn(self, n, p, k):
        rng = np.random.RandomState(42)
        X = rng.randn(n, p)
        x_arrow = _make_arrow_batch(X)
        result = _core.pca_variance(x_arrow, k)
        df = pl.from_arrow(result)

        pca = PCA(n_components=k).fit(X)
        np.testing.assert_allclose(
            df["explained_variance_ratio"].to_numpy(),
            pca.explained_variance_ratio_,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            df["cumulative_variance_ratio"].to_numpy(),
            np.cumsum(pca.explained_variance_ratio_),
            atol=1e-12,
        )

    def test_components_column(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 4)
        x_arrow = _make_arrow_batch(X)
        result = _core.pca_variance(x_arrow, 3)
        df = pl.from_arrow(result)
        assert df["component"].to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Silhouette samples
# ---------------------------------------------------------------------------


class TestSilhouetteSamples:
    @pytest.mark.parametrize("n,k", [(50, 2), (100, 3), (200, 5)])
    def test_parity_with_sklearn(self, n, k):
        rng = np.random.RandomState(42)
        X = rng.randn(n, 5)
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        labels = km.labels_

        x_arrow = _make_arrow_batch(X)
        labels_arrow = _make_labels_arrow(labels)
        ours_raw = _core.silhouette_samples(x_arrow, labels_arrow, "euclidean")
        ours = np.array(pa.array(ours_raw), dtype=np.float64)
        theirs = sklearn_silhouette_samples(X, labels, metric="euclidean")
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------


class TestSilhouetteScore:
    @pytest.mark.parametrize("n,k", [(50, 2), (100, 3), (200, 5)])
    def test_parity_with_sklearn(self, n, k):
        rng = np.random.RandomState(42)
        X = rng.randn(n, 5)
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        labels = km.labels_

        x_arrow = _make_arrow_batch(X)
        labels_arrow = _make_labels_arrow(labels)
        ours = _core.silhouette_score(x_arrow, labels_arrow, "euclidean")
        theirs = sklearn_silhouette_score(X, labels, metric="euclidean")
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


# ---------------------------------------------------------------------------
# Calinski-Harabasz score
# ---------------------------------------------------------------------------


class TestCalinskiHarabasz:
    @pytest.mark.parametrize("n,k", [(50, 2), (100, 3), (200, 5)])
    def test_parity_with_sklearn(self, n, k):
        rng = np.random.RandomState(42)
        X = rng.randn(n, 5)
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        labels = km.labels_

        x_arrow = _make_arrow_batch(X)
        labels_arrow = _make_labels_arrow(labels)
        ours = _core.calinski_harabasz_score(x_arrow, labels_arrow)
        theirs = sklearn_ch(X, labels)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)


# ---------------------------------------------------------------------------
# Classical MDS
# ---------------------------------------------------------------------------


class TestMDSClassical:
    def test_triangle_distances_recovered(self):
        """MDS on a right triangle (0,0), (3,0), (0,4) should recover
        pairwise distances 3, 4, 5 up to rotation/reflection."""
        X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        x_arrow = _make_arrow_batch(X)
        result = _core.mds_classical(x_arrow, 2, "features", "euclidean")
        df = pl.from_arrow(result)
        coords = np.column_stack([
            df["dim_0"].to_numpy(),
            df["dim_1"].to_numpy(),
        ])
        from scipy.spatial.distance import pdist

        orig_dists = pdist(X, metric="euclidean")
        embed_dists = pdist(coords, metric="euclidean")
        np.testing.assert_allclose(embed_dists, orig_dists, atol=1e-6)

    def test_square_geometry(self):
        """MDS on a square (4 points) should recover pairwise distances."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        x_arrow = _make_arrow_batch(X)
        result = _core.mds_classical(x_arrow, 2, "features", "euclidean")
        df = pl.from_arrow(result)
        coords = np.column_stack([
            df["dim_0"].to_numpy(),
            df["dim_1"].to_numpy(),
        ])
        from scipy.spatial.distance import pdist

        orig_dists = pdist(X)
        embed_dists = pdist(coords)
        np.testing.assert_allclose(embed_dists, orig_dists, atol=1e-6)

    def test_precomputed_distance_matrix(self):
        """MDS with input_type='distance' on a precomputed distance matrix."""
        D = np.array([
            [0.0, 3.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, 5.0, 0.0],
        ])
        d_arrow = _make_arrow_batch(D)
        result = _core.mds_classical(d_arrow, 2, "distance", "euclidean")
        df = pl.from_arrow(result)
        coords = np.column_stack([
            df["dim_0"].to_numpy(),
            df["dim_1"].to_numpy(),
        ])
        from scipy.spatial.distance import pdist

        embed_dists = pdist(coords)
        np.testing.assert_allclose(embed_dists, [3.0, 4.0, 5.0], atol=1e-6)
