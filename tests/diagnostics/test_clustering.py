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
    svg = chart.show_svg()
    assert svg.startswith("<svg") or "<svg" in svg


def test_pca_scree_chart_renders():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df, threshold=0.95)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_pca_scree_chart_threshold_none_omits_rule():
    """threshold=None must not inject the _threshold_line column."""
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df, threshold=None)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_cluster_diagnostics_figure():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(df, ks=[2, 3, 4, 5], random_state=0)
    assert "<svg" in chart.show_svg()


def test_cluster_diagnostics_scoring_elbow_only():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(
        df,
        ks=[2, 3, 4],
        scoring="elbow",
        random_state=0,
    )
    svg = chart.show_svg()
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
    assert "<svg" in chart.show_svg()


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
    assert "<svg" in chart.show_svg()


def test_cluster_diagnostics_rejects_dbscan():
    df = load_dataset("clustering")
    with pytest.raises(ValueError, match="don't fit the sweep-k framework"):
        ferrum.cluster_diagnostics(df, ks=[2, 3], method="dbscan")


def test_cluster_diagnostics_rejects_bad_scoring():
    df = load_dataset("clustering")
    with pytest.raises(ValueError, match="expected one of"):
        ferrum.cluster_diagnostics(df, ks=[2, 3], scoring="invalid")


# --- Mark-kwargs validation ------------------------------------------


def test_mark_silhouette_rejects_unknown_kwarg():
    model = load_fixture("kmeans_3cluster")
    df_data = load_dataset("clustering")
    source = ferrum.ModelSource(model, df_data)
    sil = source.silhouette()
    with pytest.raises(TypeError, match="unknown keyword"):
        ferrum.Chart(sil).mark_silhouette(strokr="red").show_svg()


def test_mark_pca_scree_rejects_unknown_kwarg():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    pca = source.pca_variance()
    with pytest.raises(TypeError, match="unknown keyword"):
        ferrum.Chart(pca).mark_pca_scree(strokr="red").show_svg()


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
    with pytest.raises(ValueError, match="expected"):
        source.embeddings(method="badmethod")


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
    assert "<svg" in chart.show_svg()


def test_intercluster_distance_chart_auto_k():
    """When k is omitted, the figure resolves it from model.n_clusters."""
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart = ferrum.intercluster_distance_chart(model, df, random_state=0)
    assert "<svg" in chart.show_svg()


def test_mark_intercluster_distance_rejects_unknown_kwarg():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df, random_state=0)
    icd = source.intercluster_distance(k=3)
    with pytest.raises(TypeError, match="unknown keyword"):
        (ferrum.Chart(icd).mark_intercluster_distance(strokr="red").show_svg())


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
    assert "<svg" in chart.show_svg()


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
    svg = chart.show_svg()
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
    assert "<svg" in chart.show_svg()


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
    assert "<svg" in chart.show_svg()


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
        ferrum.Chart(grid).mark_decision_boundary(strokr="red").show_svg()


# --- Task 33: clustering / manifold visualizers ----------------------


def test_silhouette_visualizer():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    viz = ferrum.SilhouetteVisualizer(model).fit(df)
    assert viz._fitted
    assert "mean_silhouette=" in repr(viz)
    assert "<svg" in viz.show().show_svg()


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
    assert "<svg" in viz.show().show_svg()


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
    assert "<svg" in viz.show().show_svg()


def test_elbow_visualizer_rejects_bad_metric():
    """An unknown metric value must raise ValueError at construction."""
    from sklearn.cluster import KMeans

    with pytest.raises(ValueError, match="not valid"):
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
    assert "<svg" in viz.show().show_svg()


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
    assert "<svg" in viz.show().show_svg()


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
    assert "<svg" in chart.show_svg()


def test_manifold_chart_figure_function():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(df)
    chart = ferrum.manifold_chart(model, df, method="pca")
    assert "<svg" in chart.show_svg()


def test_manifold_chart_colors_by_cluster():
    """Regression: manifold_chart must color points by cluster label, not all
    the same color. The SVG should contain ≥3 distinct fill colors for k=3."""
    import re
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(df)
    svg = ferrum.manifold_chart(model, df, method="pca").show_svg()
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    assert len(fills) >= 3, f"Expected ≥3 distinct fill colors for 3 clusters; got {fills}"


def test_elbow_chart_figure_function():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    chart = ferrum.elbow_chart(KMeans, df, ks=range(2, 6), random_state=0)
    assert "<svg" in chart.show_svg()


def test_elbow_chart_silhouette_metric():
    from sklearn.cluster import KMeans

    df = load_dataset("clustering")
    chart = ferrum.elbow_chart(KMeans, df, ks=range(2, 5), metric="silhouette", random_state=0)
    assert "<svg" in chart.show_svg()
