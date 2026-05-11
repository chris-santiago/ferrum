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
    from ferrum._diagnostics.charts import _residuals_chart_from_source
    source, _ = _ridge_source()
    chart = _residuals_chart_from_source(source)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_residuals_chart_from_source_raw_kind():
    from ferrum._diagnostics.charts import _residuals_chart_from_source
    source, _ = _ridge_source()
    chart = _residuals_chart_from_source(source, kind="raw")
    svg = chart.show_svg()
    assert "<svg" in svg


def test_prediction_error_chart_from_source_builder():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
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


def test_visualizer_show_before_fit_errors():
    import pytest
    viz = ferrum.ResidualsVisualizer(model=None)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()


# --- Phase 9+ no-defer guards: deferred kwargs must raise -------------------


def test_mark_residuals_cook_threshold_raises():
    import pytest
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_residuals(cook_threshold=4.0)
    with pytest.raises(NotImplementedError, match="cook_threshold"):
        chart.show_svg()


def test_mark_prediction_error_ci_raises():
    import pytest
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_prediction_error(ci=0.95)
    with pytest.raises(NotImplementedError, match="ci="):
        chart.show_svg()


def test_mark_prediction_error_reference_band_raises():
    import pytest
    source, _ = _ridge_source()
    pred = source.predictions()
    chart = ferrum.Chart(pred).mark_prediction_error(reference_band=True)
    with pytest.raises(NotImplementedError, match="reference_band"):
        chart.show_svg()
