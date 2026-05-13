"""Phase 10g — feature ranking + parallel coordinates tests."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


_REGRESSION_FEATURES = ["f0", "f1", "f2", "f3", "f4"]
_MULTICLASS_FEATURES = ["f0", "f1", "f2", "f3"]


# --- ModelSource methods ---------------------------------------------


def test_rank1d_shapiro_schema():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(
        load_fixture("regression_ridge"), df,
        load_dataset("regression")["y"],
    )
    out = source.rank1d(algorithm="shapiro")
    assert set(out.columns) == {"feature", "score", "rank"}
    assert out["feature"].dtype == pl.Utf8
    assert out["score"].dtype == pl.Float64
    assert out["rank"].dtype == pl.Int64
    # rank=1 is the top feature; rank values are strictly increasing.
    assert out["rank"].to_list() == sorted(out["rank"].to_list())
    assert out["rank"].min() == 1
    # Scores are sorted descending.
    scores = out["score"].to_numpy()
    assert np.all(np.diff(scores) <= 0)


def test_rank1d_variance():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(
        load_fixture("regression_ridge"), df,
        load_dataset("regression")["y"],
    )
    out = source.rank1d(algorithm="variance")
    assert out.height == len(_REGRESSION_FEATURES)
    assert out["rank"].to_list() == list(range(1, len(_REGRESSION_FEATURES) + 1))


def test_rank1d_covariance_requires_y():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(
        load_fixture("regression_ridge"), df,
    )
    with pytest.raises(ValueError, match="requires y"):
        source.rank1d(algorithm="covariance")


def test_rank1d_invalid_algorithm():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(load_fixture("regression_ridge"), df)
    with pytest.raises(ValueError, match="shapiro"):
        source.rank1d(algorithm="not_a_real_algo")


def test_rank2d_pearson_schema():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(load_fixture("regression_ridge"), df)
    out = source.rank2d(algorithm="pearson")
    assert set(out.columns) == {"feature_x", "feature_y", "correlation"}
    p = len(_REGRESSION_FEATURES)
    assert out.height == p * p
    # Diagonal entries are exactly 1.0.
    diag = out.filter(pl.col("feature_x") == pl.col("feature_y"))
    assert diag.height == p
    assert all(c == 1.0 for c in diag["correlation"].to_list())


def test_rank2d_kendall_parity_with_scipy():
    import scipy.stats as ss

    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(load_fixture("regression_ridge"), df)
    ours = source.rank2d(algorithm="kendall")
    X_np = df.to_numpy()
    for fa in _REGRESSION_FEATURES:
        for fb in _REGRESSION_FEATURES:
            if fa == fb:
                continue
            ferrum_val = (
                ours
                .filter((pl.col("feature_x") == fa) & (pl.col("feature_y") == fb))
                ["correlation"].item()
            )
            scipy_val = float(ss.kendalltau(
                X_np[:, _REGRESSION_FEATURES.index(fa)],
                X_np[:, _REGRESSION_FEATURES.index(fb)],
            ).statistic)
            assert abs(ferrum_val - scipy_val) < 1e-12


def test_rank2d_invalid_algorithm():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(load_fixture("regression_ridge"), df)
    with pytest.raises(ValueError, match="pearson"):
        source.rank2d(algorithm="not_a_real_algo")


# --- Charts via figure functions ------------------------------------


def test_rank_chart_1d_shapiro_renders():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(df, rank="1d", algorithm="shapiro")
    svg = chart.show_svg()
    assert "<svg" in svg and "</svg>" in svg
    # One bar per feature.
    assert svg.count("<rect") >= len(_REGRESSION_FEATURES)


def test_rank_chart_1d_variance_top_k():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(
        df, rank="1d", algorithm="variance", top_k=3,
    )
    svg = chart.show_svg()
    assert "<svg" in svg


def test_rank_chart_2d_pearson_renders():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(df, rank="2d", algorithm="pearson")
    svg = chart.show_svg()
    assert "<svg" in svg
    # 5x5 = 25 cells × {1 rect + 1 text}.
    assert svg.count("<rect") >= 25


def test_rank_chart_2d_kendall_renders():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(df, rank="2d", algorithm="kendall")
    svg = chart.show_svg()
    assert "<svg" in svg


def test_rank_chart_2d_no_annot():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart_annot = ferrum.rank_chart(df, rank="2d", annot=True)
    chart_plain = ferrum.rank_chart(df, rank="2d", annot=False)
    svg_annot = chart_annot.show_svg()
    svg_plain = chart_plain.show_svg()
    # Annotated version has more <text> elements (one per cell beyond axis labels).
    assert svg_annot.count("<text") > svg_plain.count("<text")


def test_rank_chart_invalid_rank():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    with pytest.raises(ValueError, match="'1d' or '2d'"):
        ferrum.rank_chart(df, rank="3d")


def test_parallel_coordinates_with_hue_renders():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=_MULTICLASS_FEATURES, hue="y", rescale="minmax",
    )
    svg = chart.show_svg()
    assert "<svg" in svg
    n_samples = df.height
    assert svg.count("<polyline") == n_samples


def test_parallel_coordinates_no_hue_renders():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=_MULTICLASS_FEATURES, hue=None, alpha=0.3,
    )
    svg = chart.show_svg()
    assert "<svg" in svg
    assert svg.count("<polyline") == df.height


def test_parallel_coordinates_zscore_rescale():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=_MULTICLASS_FEATURES, hue="y", rescale="zscore",
    )
    svg = chart.show_svg()
    assert "<svg" in svg


def test_parallel_coordinates_unknown_feature():
    df = load_dataset("multiclass_classification")
    with pytest.raises(ValueError, match="not in the data"):
        ferrum.parallel_coordinates_chart(
            df, features=["f0", "nope_not_real"], hue="y",
        )


def test_parallel_coordinates_invalid_rescale():
    df = load_dataset("multiclass_classification")
    with pytest.raises(ValueError, match="minmax"):
        ferrum.parallel_coordinates_chart(
            df, features=_MULTICLASS_FEATURES, rescale="wat",
        )


# --- Numpy-array path ------------------------------------------------


def test_rank_chart_1d_from_numpy_array():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    chart = ferrum.rank_chart(X, rank="1d", algorithm="shapiro")
    assert "<svg" in chart.show_svg()


def test_parallel_coordinates_from_numpy_array():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    chart = ferrum.parallel_coordinates_chart(X)
    svg = chart.show_svg()
    assert "<svg" in svg
    assert svg.count("<polyline") == 50


# --- Visualizers ----------------------------------------------------


def test_rank1d_visualizer_shapiro():
    X = load_dataset("regression").select(_REGRESSION_FEATURES)
    viz = ferrum.Rank1DVisualizer(algorithm="shapiro").fit(X)
    assert "top_feature_score=" in repr(viz)
    assert "<svg" in viz.show().show_svg()


def test_rank1d_visualizer_covariance_requires_y():
    X = load_dataset("regression").select(_REGRESSION_FEATURES)
    viz = ferrum.Rank1DVisualizer(algorithm="covariance")
    with pytest.raises(ValueError, match="requires y"):
        viz.fit(X)
    # With y supplied it works.
    y = load_dataset("regression")["y"]
    viz.fit(X, y)
    assert viz._fitted


def test_rank2d_visualizer_pearson():
    X = load_dataset("regression").select(_REGRESSION_FEATURES)
    viz = ferrum.Rank2DVisualizer(algorithm="pearson").fit(X)
    assert "max_abs_corr=" in repr(viz)
    # On the regression fixture the features are independent → small
    # off-diagonal correlations.
    assert viz._metrics["max_abs_corr"] < 0.3


def test_parallel_coordinates_visualizer():
    df = load_dataset("multiclass_classification")
    viz = ferrum.ParallelCoordinatesVisualizer(
        features=_MULTICLASS_FEATURES, hue="y",
    ).fit(df)
    assert viz._fitted
    assert "n_samples=" in repr(viz)
    assert viz._metrics["n_samples"] == df.height
    svg = viz.show().show_svg()
    assert svg.count("<polyline") == df.height


def test_parallel_coordinates_visualizer_y_attached_as_hue():
    """When hue is unset and fit(X, y) is called, y should drive the
    color encoding under '_hue' — sklearn convention that y is the
    supervisory signal.
    """
    df = load_dataset("multiclass_classification")
    X = df.select(_MULTICLASS_FEATURES)
    y = df["y"]
    viz = ferrum.ParallelCoordinatesVisualizer().fit(X, y)
    assert viz._fitted
    svg = viz.show().show_svg()
    # n_classes distinct color groups should mean >=2 stroke colors
    # in the SVG (mark_parallel_coordinates groups polylines by hue).
    assert svg.count("<polyline") == df.height


def test_parallel_coordinates_alpha_reaches_svg():
    """Regression: alpha kwarg must produce opacity in SVG polylines."""
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=_MULTICLASS_FEATURES, hue="y", alpha=0.3,
    )
    svg = chart.show_svg()
    # Opacity is baked into the RGBA stroke color (e.g. rgba(37,99,235,0.302)).
    # Verify at least one polyline carries a stroke with alpha < 1.
    import re
    rgba_alphas = re.findall(r'stroke="rgba\(\d+,\d+,\d+,([\d.]+)\)"', svg)
    assert rgba_alphas, "no rgba stroke colors found in SVG"
    assert any(float(a) < 0.5 for a in rgba_alphas), (
        f"expected alpha ~0.3 in stroke colors but got: {rgba_alphas[:5]}"
    )


def test_visualizer_show_before_fit_raises():
    viz = ferrum.Rank1DVisualizer()
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()
