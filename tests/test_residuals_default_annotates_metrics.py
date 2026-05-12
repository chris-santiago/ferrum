import numpy as np
from sklearn.linear_model import LinearRegression

import ferrum as fm


def test_residuals_default_emits_r2_rmse_mae():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 200)
    model = LinearRegression().fit(X, y)
    # Schwabish SB3 (2026-05-11): annotate_metrics overlays a top-right
    # R²/RMSE/MAE corner annotation on the single-panel layout only.
    # Main's residuals_chart defaults to panels="auto" (4-panel grid)
    # since 2026-05-11; the corner annotation lives on the single panel.
    chart = fm.residuals_chart(model, X, y, panels="single")
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
