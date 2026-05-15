"""Phase 10a regression-domain mark + builder + visualizer tests."""
from __future__ import annotations

import ferrum
from tests.fixtures import load_dataset, load_fixture


def _ridge_source():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"]), df


def test_chart_mark_residuals_renders():
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_residuals()
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_residuals_raw_kind():
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_residuals(kind="raw")
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_residuals_no_reference_line():
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_residuals(reference_line=False)
    svg = chart.show_svg()
    assert "<svg" in svg
    # No reference line means no _ref_zero column injected
    assert "_ref_zero" not in (chart._data.columns if chart._data is not None else [])


def test_chart_mark_residuals_reference_line_is_single_rule():
    """The injected `_ref_zero` column has one non-null row, so the rule layer
    emits a single horizontal line — not one per data row."""
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_residuals()
    # The desugar runs lazily on render; we can spot-check the injected column.
    # _resolve_pending preserves _data, so the post-resolve chart still has it.
    resolved = chart._resolve_pending()
    assert "_ref_zero" in resolved._data.columns
    series = resolved._data["_ref_zero"]
    non_null = sum(1 for v in series if v is not None)
    assert non_null == 1


def test_chart_mark_prediction_error_renders():
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_prediction_error()
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_prediction_error_no_identity():
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_prediction_error(identity_line=False)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_prediction_error_sorts_by_y_true():
    """identity_line=True sorts the data so the diagonal renders monotonically."""
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_prediction_error()
    ys = chart._data["y_true"].to_list()
    assert ys == sorted(ys)


# --- Task 9: chart builders -------------------------------------------------


def test_residuals_chart_from_source_builder():
    from ferrum.plots.regression import _residuals_chart_from_source
    source, _ = _ridge_source()
    chart = _residuals_chart_from_source(source)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_residuals_chart_from_source_raw_kind():
    from ferrum.plots.regression import _residuals_chart_from_source
    source, _ = _ridge_source()
    chart = _residuals_chart_from_source(source, kind="raw")
    svg = chart.show_svg()
    assert "<svg" in svg


def test_prediction_error_chart_from_source_builder():
    from ferrum.plots.regression import _prediction_error_chart_from_source
    source, _ = _ridge_source()
    chart = _prediction_error_chart_from_source(source)
    svg = chart.show_svg()
    assert "<svg" in svg


# --- Task 10: residuals_chart figure function + visualizers ----------------


def test_residuals_chart_figure_function():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"])
    svg = chart.show_svg()
    assert "<svg" in svg


def test_residuals_chart_accepts_existing_source():
    source, _ = _ridge_source()
    chart = ferrum.residuals_chart(source)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_residuals_visualizer_full_cycle():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.ResidualsVisualizer(model)
    assert "unfit" in repr(viz)

    viz.fit(X, df["y"])
    assert "rmse=" in repr(viz)
    assert "mae=" in repr(viz)
    chart = viz.show()
    assert "<svg" in chart.show_svg()


def test_prediction_error_visualizer():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.PredictionErrorVisualizer(model).fit(X, df["y"])
    chart = viz.show()
    assert "<svg" in chart.show_svg()
    assert "rmse=" in repr(viz)


def test_cooks_distance_visualizer():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.CooksDistanceVisualizer(model).fit(X, df["y"])
    chart = viz.show()
    assert "<svg" in chart.show_svg()
    assert "max_studentized=" in repr(viz)


def test_cooks_distance_visualizer_threshold_overlays_outliers():
    """When threshold='auto' (4/n rule) is set, the residuals-vs-leverage
    panel should render an outlier-highlight overlay layer in tableau
    red — the same color mark_residuals uses for its built-in overlay.
    """
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.CooksDistanceVisualizer(model, threshold="auto").fit(
        X, df["y"],
    )
    svg = viz.show().show_svg()
    assert "<svg" in svg
    # The outlier-overlay layer renders red-filled points; the tableau-red
    # fill color appears in the SVG.
    assert "#e15759" in svg


def test_cooks_distance_visualizer_explicit_threshold():
    """An explicit numeric threshold also wires through (covers the
    non-auto branch of the `4/n` rule)."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.CooksDistanceVisualizer(model, threshold=0.01).fit(
        X, df["y"],
    )
    svg = viz.show().show_svg()
    assert "<svg" in svg


def test_visualizer_show_before_fit_errors():
    import pytest
    viz = ferrum.ResidualsVisualizer(model=None)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()


# --- Phase 9+ no-defer guards: deferred kwargs must raise -------------------


def test_mark_residuals_cook_threshold_highlights_outliers():
    """cook_threshold='auto' (the 4/n rule) overlays a second mark_point
    layer keyed on `_cook_outlier_x/y` — outlier coordinates are
    non-null on rows whose Cook's D exceeds the threshold, null
    elsewhere. mark_point skips null rows so exactly K outliers render.
    """
    source, df_full = _ridge_source()
    chart = ferrum.residuals_chart(
        source, cook_threshold="auto",
    )
    svg = chart.show_svg()
    assert "<svg" in svg
    # The outlier overlay uses the explicit red fill #e15759.
    assert "#e15759" in svg


def test_mark_residuals_cook_threshold_explicit_float():
    """A numeric cook_threshold (not 'auto') routes through the same
    injection helper with the supplied value. We pick a tiny threshold
    so the regression fixture is guaranteed to surface outliers."""
    source, _ = _ridge_source()
    svg = ferrum.residuals_chart(
        source, cook_threshold=0.001,
    ).show_svg()
    assert "#e15759" in svg


def test_mark_residuals_cook_threshold_nonlinear_silent_no_outliers():
    """For non-linear estimators ModelSource.predictions emits NaN
    Cook's D. NaN > threshold is False so no outliers highlight; chart
    still renders cleanly."""
    from sklearn.ensemble import RandomForestRegressor
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
        X.to_numpy(), df["y"].to_numpy(),
    )
    svg = ferrum.residuals_chart(
        model, X, df["y"], cook_threshold="auto",
    ).show_svg()
    assert "<svg" in svg
    # No red outlier overlay since Cook's D is NaN for every row.
    assert "#e15759" not in svg


def test_mark_prediction_error_ci_renders_quantile_ribbon():
    """ci=0.95 emits a ribbon layer spanning the central 95% of
    residuals around the identity line. The chart builder computes
    quantile_low and quantile_high of residual and injects
    _pe_band_lo / _pe_band_hi columns.
    """
    source, _ = _ridge_source()
    viz = ferrum.PredictionErrorVisualizer(source._model, ci=0.95).fit(
        source._X, source._y,
    )
    svg = viz.show().show_svg()
    assert "<svg" in svg
    # mark_ribbon emits a closed <path> element for the filled band.
    assert "<path " in svg


def test_mark_prediction_error_reference_band_renders_rmse_band():
    """reference_band=True (without ci) draws ±1 RMSE around y=x."""
    source, _ = _ridge_source()
    viz = ferrum.PredictionErrorVisualizer(
        source._model, reference_band=True,
    ).fit(source._X, source._y)
    svg = viz.show().show_svg()
    assert "<path " in svg


def test_mark_prediction_error_ci_out_of_range_raises():
    """ci must be in (0, 1); 0.0 or 1.5 raises ValueError, not silent."""
    import pytest
    source, _ = _ridge_source()
    viz = ferrum.PredictionErrorVisualizer(source._model, ci=1.5)
    with pytest.raises(ValueError, match="0, 1"):
        viz.fit(source._X, source._y)


# ---------------------------------------------------------------------------
# Issue-3 regression: has_score sentinel on FerrumVisualizer
# ---------------------------------------------------------------------------


def test_has_score_false_on_base():
    """FerrumVisualizer base class must have has_score = False."""
    from ferrum._diagnostics.visualizers.base import FerrumVisualizer
    assert FerrumVisualizer.has_score is False


def test_has_score_true_on_residuals_visualizer():
    """ResidualsVisualizer.score() is real; sentinel must be True."""
    assert ferrum.ResidualsVisualizer.has_score is True


def test_has_score_true_on_prediction_error_visualizer():
    """PredictionErrorVisualizer.score() is real; sentinel must be True."""
    assert ferrum.PredictionErrorVisualizer.has_score is True


def test_has_score_false_on_unsupervised_visualizer():
    """CooksDistanceVisualizer has no score(); sentinel must be False."""
    assert ferrum.CooksDistanceVisualizer.has_score is False


# ---------------------------------------------------------------------------
# Public figure-function smoke tests
# ---------------------------------------------------------------------------


def test_prediction_error_chart_returns_chart():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.prediction_error_chart(model, X, df["y"])
    assert "<svg" in chart.show_svg()


def test_prediction_error_chart_with_ci():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.prediction_error_chart(model, X, df["y"], ci=0.90)
    assert "<svg" in chart.show_svg()


def test_prediction_error_chart_with_reference_band():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.prediction_error_chart(model, X, df["y"], reference_band=True)
    assert "<svg" in chart.show_svg()


def test_cooks_distance_chart_returns_chart():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.cooks_distance_chart(model, X, df["y"])
    assert "<svg" in chart.show_svg()


def test_cooks_distance_chart_with_auto_threshold():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.cooks_distance_chart(model, X, df["y"], threshold="auto")
    assert "<svg" in chart.show_svg()
