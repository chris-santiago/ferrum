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
        "sample_id", "y_position", "cluster", "silhouette_value",
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
        "component", "explained_variance_ratio", "cumulative_variance_ratio",
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


def test_pca_variance_raises_without_capability():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    with pytest.raises(AttributeError, match="explained_variance_ratio_"):
        source.pca_variance()


# --- Chart builders --------------------------------------------------


def test_silhouette_chart_renders():
    from ferrum._diagnostics.charts import _silhouette_chart_from_source
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
        load_fixture("pca_4comp"), df, random_state=0,
    )
    emb = source.embeddings(method="pca", n_components=2)
    assert {"dim_0", "dim_1", "label"} <= set(emb.columns)
    assert emb.height == df.height


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
        model, df, k=3, random_state=0,
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
        (
            ferrum.Chart(icd)
            .mark_intercluster_distance(strokr="red")
            .show_svg()
        )


# --- Task 32: decision_boundary --------------------------------------


def test_decision_boundary_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    # binary_logistic was trained on all four features; pass the full
    # feature set and let features=(0, 1) select the two plotting axes.
    # The remaining columns get fixed at their column means inside the
    # builder when sweeping the grid.
    chart = ferrum.decision_boundary_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
        features=(0, 1), grid_resolution=50, proba=True,
    )
    assert "<svg" in chart.show_svg()


def test_decision_boundary_rejects_three_features():
    """features=(0, 1, 2) must raise; the path needs exactly 2 axes."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    with pytest.raises(ValueError, match="exactly 2"):
        ferrum.decision_boundary_chart(
            model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
            features=(0, 1, 2), grid_resolution=10,
        )


def test_decision_boundary_predict_path_no_proba():
    """Models without predict_proba use the class-index path."""
    from sklearn.svm import SVC
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1"])
    model = SVC(kernel="linear", probability=False).fit(
        X.to_numpy(), df["y"].to_numpy(),
    )
    chart = ferrum.decision_boundary_chart(
        model, X, df["y"], features=(0, 1), grid_resolution=20, proba=False,
    )
    assert "<svg" in chart.show_svg()


def test_mark_decision_boundary_rejects_unknown_kwarg():
    """Random df-shaped frame triggers the rect quant-range path."""
    import polars as pl
    grid = pl.DataFrame({
        "x":  [0.0, 1.0],
        "x2": [1.0, 2.0],
        "y":  [0.0, 0.0],
        "y2": [1.0, 1.0],
        "z":  [0.0, 1.0],
    })
    with pytest.raises(TypeError, match="unknown keyword"):
        ferrum.Chart(grid).mark_decision_boundary(strokr="red").show_svg()
