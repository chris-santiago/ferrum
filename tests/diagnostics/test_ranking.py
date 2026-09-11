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
        load_fixture("regression_ridge"),
        df,
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
        load_fixture("regression_ridge"),
        df,
        load_dataset("regression")["y"],
    )
    out = source.rank1d(algorithm="variance")
    assert out.height == len(_REGRESSION_FEATURES)
    assert out["rank"].to_list() == list(range(1, len(_REGRESSION_FEATURES) + 1))


def test_rank1d_covariance_requires_y():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    source = ferrum.ModelSource(
        load_fixture("regression_ridge"),
        df,
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
            ferrum_val = ours.filter((pl.col("feature_x") == fa) & (pl.col("feature_y") == fb))[
                "correlation"
            ].item()
            scipy_val = float(
                ss.kendalltau(
                    X_np[:, _REGRESSION_FEATURES.index(fa)],
                    X_np[:, _REGRESSION_FEATURES.index(fb)],
                ).statistic
            )
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
    svg = chart.to_svg()
    assert "<svg" in svg and "</svg>" in svg
    # One bar per feature.
    assert svg.count("<rect") >= len(_REGRESSION_FEATURES)


def test_rank_chart_1d_variance_top_k():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(
        df,
        rank="1d",
        algorithm="variance",
        top_k=3,
    )
    svg = chart.to_svg()
    assert "<svg" in svg


def test_rank1d_chart_covariance_raw_dataframe_renders():
    # F-L12-03: rank1d_chart(df, y=..., algorithm="covariance") on a raw
    # DataFrame previously stuffed the frame into _resolve_source's model
    # slot (X=None) and crashed with the leaked Chart(data=None) error.
    # It must route through rank1d_compute_with_y instead -- the same Rust
    # entry the already-working Rank1DVisualizer path uses -- and produce
    # byte-identical output to that path on the same data.
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    y = load_dataset("regression")["y"]

    chart = ferrum.rank1d_chart(df, y=y, algorithm="covariance")
    svg = chart.to_svg()
    assert "<svg" in svg and "</svg>" in svg
    assert svg.count("<rect") >= len(_REGRESSION_FEATURES)

    viz_svg = ferrum.Rank1DVisualizer(algorithm="covariance").fit(df, y).show().to_svg()
    assert svg == viz_svg


def test_rank1d_chart_covariance_raw_dataframe_requires_y():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    with pytest.raises(ValueError, match="requires y"):
        ferrum.rank1d_chart(df, algorithm="covariance")


def test_rank1d_chart_covariance_raw_dataframe_null_y_matches_model_source():
    # Quality-review regression (#1): rank1d_compute_with_y coerced y via
    # np.asarray(y, dtype=np.float64), which casts a polars null to NaN
    # instead of preserving it as an Arrow validity-bitmap null. A single
    # missing target silently zeroed every score, and the chart still
    # rendered -- 0 real bars, literal "NaN" embedded in the SVG, no raise.
    # The raw arm must agree with the model arm's finite scores instead.
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    y_list = load_dataset("regression")["y"].to_list()
    y_list[0] = None
    y_null = pl.Series("y", y_list)
    model = load_fixture("regression_ridge")

    from ferrum.diagnostics._internal._rank_helpers import rank1d_compute_with_y

    raw_df = rank1d_compute_with_y(df, y_null, algorithm="covariance").sort("feature")
    model_df = ferrum.ModelSource(model, df, y_null).rank1d(algorithm="covariance").sort("feature")
    assert not raw_df["score"].is_nan().any()
    assert raw_df["score"].to_list() == pytest.approx(model_df["score"].to_list())

    chart = ferrum.rank1d_chart(df, y=y_null, algorithm="covariance")
    svg = chart.to_svg()
    assert "NaN" not in svg
    assert svg.count("<rect") >= len(_REGRESSION_FEATURES)


def test_rank1d_chart_covariance_string_y_raises_identically_on_both_arms():
    # Quality-review S2 pin: before _y_to_float64_arrow, a numeric-looking
    # string y silently rendered on the raw arm (numpy string-parsed it via
    # np.asarray(y, dtype=np.float64)) while raising ArrowInvalid on the
    # ModelSource arm -- same public function, same algorithm, divergent
    # semantics chosen by the type of the first positional argument.
    # _y_to_float64_arrow closes this by construction: both arms now build
    # the Arrow array the same way (pl.Series(...).to_list() ->
    # pa.array(..., type=pa.float64())), so both must raise the identical
    # exception, not just "some" exception.
    import pyarrow as pa

    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    y_str = pl.Series("y", [str(v) for v in load_dataset("regression")["y"].to_list()])
    model = load_fixture("regression_ridge")

    from ferrum.diagnostics._internal._rank_helpers import rank1d_compute_with_y

    with pytest.raises(pa.lib.ArrowInvalid) as raw_exc:
        rank1d_compute_with_y(df, y_str, algorithm="covariance")
    with pytest.raises(pa.lib.ArrowInvalid) as model_exc:
        ferrum.ModelSource(model, df, y_str).rank1d(algorithm="covariance")

    assert str(raw_exc.value) == str(model_exc.value)


def test_rank1d_chart_covariance_raw_ndarray_renders_and_x_override():
    # Pins the 2-D numpy.ndarray arm of the is_raw_data gate and the
    # raw-arm X-override line (ranking.py: input_data = data_or_source if
    # X is None else X) together: a narrower X restricts which features
    # get ranked even when data_or_source is the full array.
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    y = load_dataset("regression")["y"]
    X_full = df.to_numpy()

    chart_full = ferrum.rank1d_chart(X_full, y=y, algorithm="covariance")
    svg_full = chart_full.to_svg()
    assert "<svg" in svg_full
    assert svg_full.count("<rect") >= len(_REGRESSION_FEATURES)

    X_narrow = df.select(["f0", "f1"]).to_numpy()
    chart_narrow = ferrum.rank1d_chart(X_full, X_narrow, y, algorithm="covariance")
    svg_narrow = chart_narrow.to_svg()
    assert svg_narrow.count("<rect") < svg_full.count("<rect")


def test_rank1d_chart_covariance_raw_pandas_dataframe_matches_polars():
    # Pins the pandas.DataFrame arm of the is_raw_data gate; asserts parity
    # with the polars-frame result on identical data.
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    y = load_dataset("regression")["y"]
    pdf = df.to_pandas()

    chart_pd = ferrum.rank1d_chart(pdf, y=y, algorithm="covariance")
    svg_pd = chart_pd.to_svg()
    assert "<svg" in svg_pd
    assert svg_pd.count("<rect") >= len(_REGRESSION_FEATURES)

    chart_pl = ferrum.rank1d_chart(df, y=y, algorithm="covariance")
    assert svg_pd == chart_pl.to_svg()


def test_rank1d_chart_covariance_fitted_estimator_no_x_typed_refusal():
    # A fitted estimator (not raw data, not a ModelSource) is not caught
    # by an isinstance(ModelSource) check -- it must still route through
    # the model path (_resolve_source), not the raw-data path, so an
    # X-less call surfaces ModelSource's typed refusal instead of a
    # leaked ``TypeError: float() argument must be a string or a real
    # number, not 'Ridge'`` from np.asarray() inside the raw helper.
    y = load_dataset("regression")["y"]
    model = load_fixture("regression_ridge")
    with pytest.raises(ValueError, match="requires the training data"):
        ferrum.rank1d_chart(model, y=y, algorithm="covariance")


def test_rank1d_chart_covariance_fitted_estimator_with_x_matches_model_source_path():
    # Estimator + explicit X + y for algorithm="covariance" must stay on
    # the exact model path a pre-built ModelSource already uses (spec G8:
    # "Model/ModelSource inputs keep the current path unchanged"), not the
    # raw-data path added for bare DataFrame/array input.
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    y = load_dataset("regression")["y"]
    model = load_fixture("regression_ridge")

    chart = ferrum.rank1d_chart(model, df, y, algorithm="covariance")
    via_model_source = ferrum.rank1d_chart(ferrum.ModelSource(model, df, y), algorithm="covariance")
    assert chart.to_svg() == via_model_source.to_svg()


def test_rank_chart_2d_pearson_renders():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(df, rank="2d", algorithm="pearson")
    svg = chart.to_svg()
    assert "<svg" in svg
    # 5x5 = 25 cells × {1 rect + 1 text}.
    assert svg.count("<rect") >= 25


def test_rank_chart_2d_kendall_renders():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart = ferrum.rank_chart(df, rank="2d", algorithm="kendall")
    svg = chart.to_svg()
    assert "<svg" in svg


def test_rank_chart_2d_no_annot():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    chart_annot = ferrum.rank_chart(df, rank="2d", annot=True)
    chart_plain = ferrum.rank_chart(df, rank="2d", annot=False)
    svg_annot = chart_annot.to_svg()
    svg_plain = chart_plain.to_svg()
    # Annotated version has more <text> elements (one per cell beyond axis labels).
    assert svg_annot.count("<text") > svg_plain.count("<text")


def test_rank_chart_invalid_rank():
    df = load_dataset("regression").select(_REGRESSION_FEATURES)
    with pytest.raises(ValueError, match="must be one of"):
        ferrum.rank_chart(df, rank="3d")


def test_parallel_coordinates_with_hue_renders():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df,
        features=_MULTICLASS_FEATURES,
        hue="y",
        rescale="minmax",
    )
    svg = chart.to_svg()
    assert "<svg" in svg
    n_samples = df.height
    assert svg.count("<polyline") == n_samples


def test_parallel_coordinates_no_hue_renders():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df,
        features=_MULTICLASS_FEATURES,
        hue=None,
        alpha=0.3,
    )
    svg = chart.to_svg()
    assert "<svg" in svg
    assert svg.count("<polyline") == df.height


def test_parallel_coordinates_zscore_rescale():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df,
        features=_MULTICLASS_FEATURES,
        hue="y",
        rescale="zscore",
    )
    svg = chart.to_svg()
    assert "<svg" in svg


def test_parallel_coordinates_unknown_feature():
    df = load_dataset("multiclass_classification")
    with pytest.raises(ValueError, match="not in the data"):
        ferrum.parallel_coordinates_chart(
            df,
            features=["f0", "nope_not_real"],
            hue="y",
        )


def test_parallel_coordinates_invalid_rescale():
    df = load_dataset("multiclass_classification")
    with pytest.raises(ValueError, match="minmax"):
        ferrum.parallel_coordinates_chart(
            df,
            features=_MULTICLASS_FEATURES,
            rescale="wat",
        )


# --- Numpy-array path ------------------------------------------------


def test_rank_chart_1d_from_numpy_array():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    chart = ferrum.rank_chart(X, rank="1d", algorithm="shapiro")
    assert "<svg" in chart.to_svg()


def test_parallel_coordinates_from_numpy_array():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    chart = ferrum.parallel_coordinates_chart(X)
    svg = chart.to_svg()
    assert "<svg" in svg
    assert svg.count("<polyline") == 50
    # B3 (cohesion campaign): a bare 2D numpy array (features=None) is auto-named
    # col_0/col_1/... — the single ferrum-wide convention shared with
    # Chart(numpy_array) / ferrum._coerce.to_arrow_table — and those names become
    # the rendered axis labels. This pins the unified naming so it cannot drift
    # back to the retired legacy f{j} convention.
    for j in range(4):
        assert f">col_{j}<" in svg, f"expected axis label col_{j} in parallel-coordinates SVG"
    assert ">f0<" not in svg


# --- Visualizers ----------------------------------------------------


def test_rank1d_visualizer_shapiro():
    X = load_dataset("regression").select(_REGRESSION_FEATURES)
    viz = ferrum.Rank1DVisualizer(algorithm="shapiro").fit(X)
    assert "top_feature_score=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


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
        features=_MULTICLASS_FEATURES,
        hue="y",
    ).fit(df)
    assert viz._fitted
    assert "n_samples=" in repr(viz)
    assert viz._metrics["n_samples"] == df.height
    svg = viz.show().to_svg()
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
    svg = viz.show().to_svg()
    # n_classes distinct color groups should mean >=2 stroke colors
    # in the SVG (mark_parallel_coordinates groups polylines by hue).
    assert svg.count("<polyline") == df.height


def test_parallel_coordinates_alpha_reaches_svg():
    """Regression: alpha kwarg must produce opacity in SVG polylines."""
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df,
        features=_MULTICLASS_FEATURES,
        hue="y",
        alpha=0.3,
    )
    svg = chart.to_svg()
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
