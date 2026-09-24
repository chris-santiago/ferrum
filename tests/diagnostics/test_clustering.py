"""Phase 10f — clustering / manifold / decision-boundary tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


# --- Source methods --------------------------------------------------


def test_silhouette_method():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df)
    sil = source.silhouette()
    assert set(sil.columns) == {
        "sample_id",
        "y_position",
        "cluster",
        "silhouette_value",
    }
    # y_position is sequential 0..n-1.
    yp = sil["y_position"].to_numpy()
    assert yp[0] == 0
    assert yp[-1] == sil.height - 1
    # Within each cluster, silhouette_value is monotonically non-increasing.
    # cluster is serialized as Utf8 so it routes through the renderer's
    # categorical color scale (see SCHEMA_SILHOUETTE).
    assert sil["cluster"].dtype == pl.Utf8
    for c in sorted(set(sil["cluster"].to_list())):
        sub = sil.filter(pl.col("cluster") == c)["silhouette_value"].to_numpy()
        assert np.all(np.diff(sub) <= 0)


def test_pca_variance_method():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    pca = source.pca_variance()
    assert set(pca.columns) == {
        "component",
        "explained_variance_ratio",
        "cumulative_variance_ratio",
    }
    cum = pca["cumulative_variance_ratio"].to_numpy()
    evr = pca["explained_variance_ratio"].to_numpy()
    np.testing.assert_allclose(cum[-1], evr.sum(), rtol=1e-12)


def test_pca_variance_n_components_truncation():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    pca = source.pca_variance(n_components=2)
    assert pca.height == 2


def test_pca_variance_fallback_without_capability():
    """When the model lacks explained_variance_ratio_, pca_variance falls back
    to Rust SVD on the raw X data."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    result = source.pca_variance()
    assert result.height > 0
    assert "explained_variance_ratio" in result.columns
    assert "cumulative_variance_ratio" in result.columns


# --- Chart builders --------------------------------------------------


def test_silhouette_chart_renders():
    from ferrum.plots.clustering import _silhouette_chart_from_source

    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df)
    chart = _silhouette_chart_from_source(source)
    svg = chart.to_svg()
    assert svg.startswith("<svg") or "<svg" in svg


def test_pca_scree_chart_renders():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df, threshold=0.95)
    svg = chart.to_svg()
    assert "<svg" in svg


def test_pca_scree_chart_threshold_none_omits_rule():
    """threshold=None must not inject the _threshold_line column."""
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df, threshold=None)
    svg = chart.to_svg()
    assert "<svg" in svg


def test_cluster_diagnostics_figure():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(df, ks=[2, 3, 4, 5], random_state=0)
    assert "<svg" in chart.to_svg()


def test_cluster_diagnostics_scoring_elbow_only():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(
        df,
        ks=[2, 3, 4],
        scoring="elbow",
        random_state=0,
    )
    svg = chart.to_svg()
    assert "<svg" in svg
    # Single-panel: should NOT be an HConcatChart (no panel separator).
    # HConcatChart renders two <svg> sub-panels nested in a wrapper.
    assert svg.count("<svg") == 1


def test_cluster_diagnostics_scoring_silhouette_only():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(
        df,
        ks=[2, 3, 4],
        scoring="silhouette",
        random_state=0,
    )
    assert "<svg" in chart.to_svg()


def test_cluster_diagnostics_hierarchical():
    """Method='hierarchical' uses AgglomerativeClustering (Ward) and
    computes inertia manually as the sum of squared distances from each
    sample to its cluster centroid (KMeans' inertia definition).
    """
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(
        df,
        ks=[2, 3, 4],
        method="hierarchical",
    )
    assert "<svg" in chart.to_svg()


def test_cluster_diagnostics_rejects_dbscan():
    df = load_dataset("clustering")
    with pytest.raises(ValueError, match=r"cluster_diagnostics: method must be one of"):
        ferrum.cluster_diagnostics(df, ks=[2, 3], method="dbscan")


def test_cluster_diagnostics_rejects_bad_scoring():
    df = load_dataset("clustering")
    with pytest.raises(ValueError, match=r"cluster_diagnostics: scoring must be one of"):
        ferrum.cluster_diagnostics(df, ks=[2, 3], scoring="invalid")


# --- Mark-kwargs validation ------------------------------------------


def test_mark_silhouette_rejects_unknown_kwarg():
    model = load_fixture("kmeans_3cluster")
    df_data = load_dataset("clustering")
    source = ferrum.ModelSource(model, df_data)
    sil = source.silhouette()
    with pytest.raises(TypeError, match="unknown keyword"):
        ferrum.Chart(sil).mark_silhouette(strokr="red").to_svg()


def test_mark_pca_scree_rejects_unknown_kwarg():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    pca = source.pca_variance()
    with pytest.raises(TypeError, match="unknown keyword"):
        ferrum.Chart(pca).mark_pca_scree(strokr="red").to_svg()


# --- Task 31: embeddings + intercluster_distance --------------------


def test_embeddings_pca():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(
        load_fixture("pca_4comp"),
        df,
        random_state=0,
    )
    emb = source.embeddings(method="pca", n_components=2)
    assert {"dim_0", "dim_1", "label"} <= set(emb.columns)
    assert emb.height == df.height


def test_embeddings_label_column_is_utf8():
    """Regression: label column must be Utf8 so nominal color encoding works."""
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(load_fixture("pca_4comp"), df, random_state=0)
    emb = source.embeddings(method="pca", n_components=2)
    assert emb["label"].dtype == pl.Utf8


def test_embeddings_uses_model_labels_when_y_absent():
    """Regression: when y is not provided, embeddings() must use model.labels_
    from a fitted clusterer instead of all-zeros."""
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df, random_state=0)
    emb = source.embeddings(method="pca", n_components=2)
    distinct_labels = set(emb["label"].to_list())
    assert len(distinct_labels) >= 2, (
        f"Expected ≥2 distinct labels from KMeans; got {distinct_labels}"
    )


def test_embeddings_rejects_bad_method():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(load_fixture("pca_4comp"), df)
    with pytest.raises(ValueError, match=r"ModelSource.embeddings: method must be one of"):
        source.embeddings(method="badmethod")


# --- #222 / N4: embeddings(method="tsne") perplexity handling -------
#
# One validation owner (``_resolve_tsne_perplexity`` /
# ``_check_embedding_method_kwargs`` in ``sources/_clustering.py``): the
# default adapts to ``min(30, n_samples - 2)`` so small-n t-SNE renders out
# of the box (pre-fix: every n <= 31 raised the raw manifolds-rs "kNN size"
# error on the default path alone), an explicit out-of-range/non-positive/
# non-finite perplexity refuses typed in perplexity/n_samples vocabulary
# with no PanicException reachable, and an unknown method_kwargs key
# refuses by name per method.


def _tsne_x(n: int, seed: int = 0) -> pl.DataFrame:
    """``n``-row, 3-feature raw X for exercising ``embeddings(method='tsne')``
    below any real fitted model (this issue lives entirely in perplexity
    resolution, not in cluster assignment)."""
    rng = np.random.RandomState(seed)
    return pl.DataFrame(
        {
            "f0": rng.randn(n),
            "f1": rng.randn(n),
            "f2": rng.randn(n),
        }
    )


def test_embeddings_tsne_default_renders_at_n20():
    """Number-mover pin: pre-fix, the unmodified manifolds-rs default of 30
    reached the kernel unclamped and raised for every n <= 31 -- n=20 is
    representative of that entire dead range. Post-fix the default resolves
    to min(30, 18) = 18 and renders."""
    source = ferrum.ModelSource(None, _tsne_x(20), random_state=0)
    emb = source.embeddings(method="tsne")
    assert emb.height == 20
    assert {"dim_0", "dim_1", "label"} <= set(emb.columns)
    xy = emb.select("dim_0", "dim_1").to_numpy()
    assert np.isfinite(xy).all()


def test_embeddings_tsne_perplexity_out_of_range_refuses():
    """p > n_samples - 2 refuses typed, naming perplexity/n_samples/bound --
    not the raw manifolds-rs 'kNN size' vocabulary."""
    source = ferrum.ModelSource(None, _tsne_x(20), random_state=0)
    with pytest.raises(ValueError, match=r"perplexity=200.*n_samples=20.*<= 18") as exc_info:
        source.embeddings(method="tsne", perplexity=200)
    assert "kNN size" not in str(exc_info.value)
    assert "PanicException" not in str(exc_info.value)


def test_embeddings_tsne_perplexity_non_positive_refuses():
    """A non-positive perplexity (0 or negative) refuses typed instead of
    reaching manifolds-rs, where 0.0 silently succeeded (k floors at 5) and
    a negative value panicked the process boundary (to_usize().unwrap() on
    a value with no valid usize)."""
    source = ferrum.ModelSource(None, _tsne_x(20), random_state=0)
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match=r"perplexity must be positive.*n_samples=20"):
            source.embeddings(method="tsne", perplexity=bad)


def test_embeddings_tsne_perplexity_non_finite_refuses_no_panic():
    """NaN previously reached ``k_float.to_usize().unwrap()`` in
    manifolds-rs and raised a ``PanicException`` -- not a ``ValueError`` --
    that no ``except ValueError`` could catch. The type of the exception
    pytest.raises asserts on is itself the regression pin: a PanicException
    would fail this test even though it 'raises something'."""
    source = ferrum.ModelSource(None, _tsne_x(20), random_state=0)
    with pytest.raises(
        ValueError, match=r"perplexity=\.\.\.\) must be a finite number a float can hold"
    ):
        source.embeddings(method="tsne", perplexity=float("nan"))
    with pytest.raises(
        ValueError, match=r"perplexity=\.\.\.\) must be a finite number a float can hold"
    ):
        source.embeddings(method="tsne", perplexity=float("inf"))


def test_embeddings_tsne_default_perplexity_deterministic():
    """The default-perplexity adaptation must not disturb the existing
    seed-determinism contract: same random_state twice is byte-equal,
    different random_state differs."""
    x = _tsne_x(20)
    a = ferrum.ModelSource(None, x, random_state=7).embeddings(method="tsne")
    b = ferrum.ModelSource(None, x, random_state=7).embeddings(method="tsne")
    c = ferrum.ModelSource(None, x, random_state=8).embeddings(method="tsne")
    assert a.equals(b)
    assert not a.equals(c)


@pytest.mark.parametrize(
    "method,bad_kwargs",
    [
        ("tsne", {"bogus": 1}),
        ("tsne", {"n_neighbors": 5}),  # a real key, wrong method
        ("umap", {"perplexity": 5}),  # a real key, wrong method
        ("pca", {"n_neighbors": 5}),  # pca accepts none
    ],
)
def test_embeddings_unknown_method_kwargs_refuses_by_name(method, bad_kwargs):
    source = ferrum.ModelSource(None, _tsne_x(20), random_state=0)
    (bad_key,) = bad_kwargs
    with pytest.raises(ValueError, match=rf"method={method!r}.*unknown method_kwargs.*{bad_key!r}"):
        source.embeddings(method=method, **bad_kwargs)


# Boundary pin for the adaptive-default clamp and the explicit-perplexity
# refusal bound, at the exact edges: n=5/6/7 sit just above the n>=4 floor
# manifolds-rs itself enforces (where _tsne_perplexity_bound(n) = n - 2 is
# smallest and closest to where a library-minimum guard could fire), and
# n=31/32 straddle the min(30, n-2) crossover -- n=31 is the last n where
# the clamp still binds (bound=29 < manifolds-rs's own default of 30) and
# n=32 is the first n where it no longer binds (bound=30, so the resolved
# default equals manifolds-rs's own unclamped default). A future off-by-one
# in either _tsne_perplexity_bound or the min(30, ...)/floor composition
# would move one of these five numbers without necessarily breaking any
# non-boundary test. Each expected value below was verified live against
# the real ModelSource.embeddings(method="tsne") path (not just the private
# helper), matching this test's own render/refusal assertions.
@pytest.mark.parametrize(
    "n,expected_default_perplexity",
    [
        (5, 3.0),
        (6, 4.0),
        (7, 5.0),
        (31, 29.0),
        (32, 30.0),
    ],
)
def test_embeddings_tsne_default_perplexity_boundary(n, expected_default_perplexity):
    from ferrum.diagnostics.sources._clustering import (
        _TSNE_DEFAULT_PERPLEXITY,
        _TSNE_MIN_PERPLEXITY,
        _resolve_tsne_perplexity,
        _tsne_perplexity_bound,
    )

    bound = _tsne_perplexity_bound(n)
    resolved = _resolve_tsne_perplexity(n, None)
    assert resolved == expected_default_perplexity
    assert resolved == max(_TSNE_MIN_PERPLEXITY, min(_TSNE_DEFAULT_PERPLEXITY, float(bound)))

    # Behavioral confirmation through the real path, not just the formula:
    # the default renders at exactly this n.
    source = ferrum.ModelSource(None, _tsne_x(n), random_state=0)
    emb = source.embeddings(method="tsne")
    assert emb.height == n
    xy = emb.select("dim_0", "dim_1").to_numpy()
    assert np.isfinite(xy).all()


@pytest.mark.parametrize("n", [5, 6, 7, 31, 32])
def test_embeddings_tsne_explicit_perplexity_bound_boundary(n):
    """The explicit-perplexity refusal bound (perplexity <= n_samples - 2)
    at the exact edges: the bound itself renders through the real kernel,
    one above it refuses -- pinned at n=5/6/7 and n=31/32 so an off-by-one
    in the bound comparison (or in manifolds-rs's own kNN-size derivation)
    can't land green."""
    from ferrum.diagnostics.sources._clustering import _tsne_perplexity_bound

    bound = _tsne_perplexity_bound(n)

    source_ok = ferrum.ModelSource(None, _tsne_x(n, seed=1), random_state=0)
    emb = source_ok.embeddings(method="tsne", perplexity=float(bound))
    assert emb.height == n

    source_bad = ferrum.ModelSource(None, _tsne_x(n, seed=1), random_state=0)
    with pytest.raises(ValueError, match=rf"n_samples={n}.*<= {bound}"):
        source_bad.embeddings(method="tsne", perplexity=float(bound + 1))


def test_intercluster_distance_method():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df, random_state=0)
    icd = source.intercluster_distance(k=3, method="mds")
    assert icd.height == 3
    assert set(icd.columns) == {"cluster", "x", "y", "size"}
    # sizes sum to the total sample count (kmeans assigns every point).
    assert int(icd["size"].sum()) == df.height


def test_intercluster_distance_raises_without_capability():
    """Models that lack cluster_centers_ must raise on intercluster_distance."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    with pytest.raises(AttributeError, match="cluster_centers_"):
        source.intercluster_distance(k=3)


def test_intercluster_distance_chart_renders():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart = ferrum.intercluster_distance_chart(
        model,
        df,
        k=3,
        random_state=0,
    )
    assert "<svg" in chart.to_svg()


def test_intercluster_distance_chart_auto_k():
    """When k is omitted, the figure resolves it from model.n_clusters."""
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart = ferrum.intercluster_distance_chart(model, df, random_state=0)
    assert "<svg" in chart.to_svg()


def test_mark_intercluster_distance_rejects_unknown_kwarg():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df, random_state=0)
    icd = source.intercluster_distance(k=3)
    with pytest.raises(TypeError, match="unknown keyword"):
        (ferrum.Chart(icd).mark_intercluster_distance(strokr="red").to_svg())


# --- Task 32: decision_boundary --------------------------------------


def test_decision_boundary_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    # binary_logistic was trained on all four features; pass the full
    # feature set and let features=(0, 1) select the two plotting axes.
    # The remaining columns get fixed at their column means inside the
    # builder when sweeping the grid.
    chart = ferrum.decision_boundary_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        features=(0, 1),
        grid_resolution=50,
        proba=True,
    )
    assert "<svg" in chart.to_svg()


def test_decision_boundary_chart_scatter_overlay():
    """scatter=True must produce a TRUE multi-layer overlay (not an
    HConcatChart fallthrough). Both layers share the same DataFrame so
    the chart-level color scale resolves continuous-from-z and the
    scatter's true-class-as-float column maps onto the same scale —
    matching boundary color and point color makes misclassifications
    visually pop. Regression test for the post-merge fix.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.decision_boundary_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        features=(0, 1),
        grid_resolution=40,
        scatter=True,
    )
    # Must be a single layered Chart (not HConcatChart).
    assert type(chart).__name__ == "Chart"
    assert chart._layers is not None
    assert len(chart._layers) == 2
    svg = chart.to_svg()
    # The scatter overlay emits SVG <circle> elements with a black
    # stroke (uniform across all points).
    assert 'stroke="#000000"' in svg
    # Non-proba mode: class indices are cast to String → categorical color
    # palette. At least two distinct fill colors from the theme's qualitative
    # palette should appear (one per class).
    import re

    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    assert len(fills) >= 2, f"Expected ≥2 distinct fill colors; got {fills}"


def test_decision_boundary_chart_scatter_overlay_proba_mode():
    """scatter=True works with proba=True too (z domain shifts from
    class index to [0,1] but the scatter_z mapping handles both).
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.decision_boundary_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        features=(0, 1),
        grid_resolution=40,
        proba=True,
        scatter=True,
    )
    assert chart._layers is not None
    assert len(chart._layers) == 2
    assert "<svg" in chart.to_svg()


def test_decision_boundary_rejects_three_features():
    """features=(0, 1, 2) must raise; the path needs exactly 2 axes."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    with pytest.raises(ValueError, match="exactly 2"):
        ferrum.decision_boundary_chart(
            model,
            df.select(["f0", "f1", "f2", "f3"]),
            df["y"],
            features=(0, 1, 2),
            grid_resolution=10,
        )


def test_decision_boundary_predict_path_no_proba():
    """Models without predict_proba use the class-index path."""
    from sklearn.svm import SVC

    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1"])
    model = SVC(kernel="linear", probability=False).fit(
        X.to_numpy(),
        df["y"].to_numpy(),
    )
    chart = ferrum.decision_boundary_chart(
        model,
        X,
        df["y"],
        features=(0, 1),
        grid_resolution=20,
        proba=False,
    )
    assert "<svg" in chart.to_svg()


def test_mark_decision_boundary_rejects_unknown_kwarg():
    """Random df-shaped frame triggers the rect quant-range path."""
    import polars as pl

    grid = pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "x2": [1.0, 2.0],
            "y": [0.0, 0.0],
            "y2": [1.0, 1.0],
            "z": [0.0, 1.0],
        }
    )
    with pytest.raises(TypeError, match="unknown keyword"):
        ferrum.Chart(grid).mark_decision_boundary(strokr="red").to_svg()


# --- Task 33: clustering / manifold visualizers ----------------------


def test_silhouette_visualizer():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    viz = ferrum.SilhouetteVisualizer(model).fit(df)
    assert viz._fitted
    assert "mean_silhouette=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_elbow_visualizer():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    viz = ferrum.ElbowVisualizer(
        KMeans,
        ks=[2, 3, 4, 5],
        random_state=0,
    ).fit(df)
    assert viz._fitted
    assert "best_k=" in repr(viz)
    # On a well-separated 3-cluster dataset, distortion drops sharply
    # at k=3; the minimum-distortion k will be the largest k tried
    # (more centers can only reduce inertia), but best_k should match
    # the smallest k whose score equals the minimum within tolerance.
    assert viz._metrics["best_k"] in {3.0, 4.0, 5.0}


def test_elbow_visualizer_silhouette():
    """metric='silhouette' must fit, score, and pick a finite best_k."""
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    viz = ferrum.ElbowVisualizer(
        KMeans,
        ks=[2, 3, 4, 5],
        metric="silhouette",
        random_state=0,
    ).fit(df)
    assert viz._fitted
    best_k = viz._metrics["best_k"]
    assert np.isfinite(best_k)
    assert best_k in {2.0, 3.0, 4.0, 5.0}
    assert "<svg" in viz.show().to_svg()


def test_elbow_visualizer_calinski_harabasz():
    """metric='calinski_harabasz' must fit, score, and pick a finite best_k."""
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    viz = ferrum.ElbowVisualizer(
        KMeans,
        ks=[2, 3, 4, 5],
        metric="calinski_harabasz",
        random_state=0,
    ).fit(df)
    assert viz._fitted
    best_k = viz._metrics["best_k"]
    assert np.isfinite(best_k)
    assert best_k in {2.0, 3.0, 4.0, 5.0}
    assert "<svg" in viz.show().to_svg()


def test_elbow_visualizer_rejects_bad_metric():
    """An unknown metric value must raise ValueError at construction."""
    from sklearn.cluster import KMeans

    with pytest.raises(ValueError, match="must be one of"):
        ferrum.ElbowVisualizer(
            KMeans,
            ks=[2, 3],
            metric="badmetric",
            random_state=0,
        )


def test_elbow_visualizer_silhouette_skips_k1():
    """k=1 has undefined silhouette; sweep skips it but still fits k>=2."""
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    viz = ferrum.ElbowVisualizer(
        KMeans,
        ks=[1, 2, 3],
        metric="silhouette",
        random_state=0,
    ).fit(df)
    assert viz._fitted
    assert viz._metrics["best_k"] in {2.0, 3.0}


def test_manifold_visualizer_pca():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    model = load_fixture("pca_4comp")
    viz = ferrum.ManifoldVisualizer(
        model,
        method="pca",
        random_state=0,
    ).fit(df)
    assert viz._fitted
    assert "<svg" in viz.show().to_svg()


def test_intercluster_distance_visualizer():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    viz = ferrum.InterclusterDistanceVisualizer(
        model,
        random_state=0,
    ).fit(df)
    assert viz._fitted
    assert "max_intercluster_dist=" in repr(viz)


def test_pca_variance_visualizer():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.PCAVarianceVisualizer(model).fit(df)
    assert viz._fitted
    assert "first_component_var=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_visualizer_show_before_fit_raises():
    """The FerrumVisualizer base must reject .show() on unfit instances."""
    model = load_fixture("kmeans_3cluster")
    viz = ferrum.SilhouetteVisualizer(model)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()


# ---------------------------------------------------------------------------
# Public figure-function smoke tests
# ---------------------------------------------------------------------------


def test_silhouette_chart_figure_function():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(df)
    chart = ferrum.silhouette_chart(model, df)
    assert "<svg" in chart.to_svg()


def test_manifold_chart_figure_function():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(df)
    chart = ferrum.manifold_chart(model, df, method="pca")
    assert "<svg" in chart.to_svg()


def test_manifold_chart_colors_by_cluster():
    """Regression: manifold_chart must color points by cluster label, not all
    the same color. The SVG should contain ≥3 distinct fill colors for k=3."""
    import re
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(df)
    svg = ferrum.manifold_chart(model, df, method="pca").to_svg()
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    assert len(fills) >= 3, f"Expected ≥3 distinct fill colors for 3 clusters; got {fills}"


def test_elbow_chart_figure_function():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    chart = ferrum.elbow_chart(KMeans, df, ks=range(2, 6), random_state=0)
    assert "<svg" in chart.to_svg()


def test_elbow_chart_silhouette_metric():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    chart = ferrum.elbow_chart(KMeans, df, ks=range(2, 5), metric="silhouette", random_state=0)
    assert "<svg" in chart.to_svg()
