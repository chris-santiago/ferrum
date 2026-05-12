import numpy as np
from sklearn.linear_model import LinearRegression

import ferrum as fm


def test_residuals_default_emits_r2_rmse_mae():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 200)
    model = LinearRegression().fit(X, y)
    chart = fm.residuals_chart(model, X, y)
    svg = chart.show_svg()
    assert "R²" in svg
    assert "RMSE" in svg
    assert "MAE" in svg


def test_residuals_annotate_metrics_false_omits():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 200)
    model = LinearRegression().fit(X, y)
    chart = fm.residuals_chart(model, X, y, annotate_metrics=False)
    svg = chart.show_svg()
    assert "R²" not in svg


def test_residuals_single_panel_also_annotates():
    """The corner annotation rides on both the 4-panel and single-panel
    layouts — see _inject_metrics_corner + _overlay_metrics_corner in
    src/ferrum/_diagnostics/charts.py."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 200)
    model = LinearRegression().fit(X, y)
    chart = fm.residuals_chart(model, X, y, panels="single")
    svg = chart.show_svg()
    assert "R²" in svg
    assert "RMSE" in svg
    assert "MAE" in svg
