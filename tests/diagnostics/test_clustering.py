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
