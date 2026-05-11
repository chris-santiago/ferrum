"""Phase 10e — model-selection figure-shape tests.

Figure-shape tests verify that each chart family renders to SVG and
carries the layers expected by the desugar. Quantized SVG byte-equality
goldens live in ``test_goldens_phase_10.py``.
"""
from __future__ import annotations

import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset


def _ridge_xy():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return Ridge(random_state=0), X, df["y"]


# --- learning_curve_chart -------------------------------------------------


def test_learning_curve_chart_renders():
    model, X, y = _ridge_xy()
    chart = ferrum.learning_curve_chart(model, X, y, cv=3, random_state=0)
    svg = chart.show_svg()
    assert svg.startswith("<svg") or svg.lstrip().startswith("<svg")


def test_learning_curve_chart_errorbar_style():
    model, X, y = _ridge_xy()
    chart = ferrum.learning_curve_chart(
        model, X, y, cv=3, ci_style="errorbar", random_state=0,
    )
    assert "<svg" in chart.show_svg()


def test_learning_curve_chart_invalid_ci_style():
    model, X, y = _ridge_xy()
    chart = ferrum.learning_curve_chart(
        model, X, y, cv=3, ci_style="invalid", random_state=0,
    )
    with pytest.raises(ValueError, match="ci_style"):
        chart.show_svg()


# --- validation_curve_chart -----------------------------------------------


def test_validation_curve_chart_linear_x():
    model, X, y = _ridge_xy()
    chart = ferrum.validation_curve_chart(
        model, X, y,
        param="alpha", values=[0.1, 1.0, 10.0],
        cv=3, log_scale=False,
    )
    assert "<svg" in chart.show_svg()


def test_validation_curve_chart_auto_log_scale():
    model, X, y = _ridge_xy()
    # alpha range spans 4 orders of magnitude → auto log.
    chart = ferrum.validation_curve_chart(
        model, X, y,
        param="alpha", values=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=3, log_scale="auto",
    )
    assert "<svg" in chart.show_svg()


def test_validation_curve_chart_requires_values():
    model, X, y = _ridge_xy()
    with pytest.raises(ValueError, match="values"):
        ferrum.validation_curve_chart(model, X, y, param="alpha")


# --- cv_scores_chart ------------------------------------------------------


def test_cv_scores_chart_box():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="box")
    assert "<svg" in chart.show_svg()


def test_cv_scores_chart_bar_aggregates():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="bar")
    assert "<svg" in chart.show_svg()


def test_cv_scores_chart_strip():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="strip")
    assert "<svg" in chart.show_svg()


def test_cv_scores_chart_filter_split():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="box", split="test")
    assert "<svg" in chart.show_svg()


def test_cv_scores_chart_invalid_kind():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="invalid")
    with pytest.raises(ValueError, match="kind"):
        chart.show_svg()


# --- alpha_selection_chart -----------------------------------------------


def test_alpha_selection_chart_default():
    model, X, y = _ridge_xy()
    chart = ferrum.alpha_selection_chart(
        model, X, y, alphas=[0.1, 1.0, 10.0], cv=3,
    )
    assert "<svg" in chart.show_svg()


def test_alpha_selection_chart_highlight_best_injects_column():
    model, X, y = _ridge_xy()
    chart = ferrum.alpha_selection_chart(
        model, X, y, alphas=[0.1, 1.0, 10.0, 100.0], cv=3, highlight_best=True,
    )
    # The Chart's data should now carry a _best_alpha sentinel column.
    df = chart._data  # private; touch only inside diagnostics tests
    assert isinstance(df, pl.DataFrame)
    assert "_best_alpha" in df.columns
    # Exactly one non-null row.
    assert df["_best_alpha"].drop_nulls().len() == 1


def test_alpha_selection_chart_no_highlight():
    model, X, y = _ridge_xy()
    chart = ferrum.alpha_selection_chart(
        model, X, y, alphas=[0.1, 1.0, 10.0], cv=3, highlight_best=False,
    )
    df = chart._data
    assert isinstance(df, pl.DataFrame)
    assert "_best_alpha" not in df.columns


def test_alpha_selection_chart_requires_alphas():
    model, X, y = _ridge_xy()
    with pytest.raises(ValueError, match="alphas"):
        ferrum.alpha_selection_chart(model, X, y)


# --- 10e visualizers ------------------------------------------------------


def test_learning_curve_visualizer_records_final_test_score():
    model, X, y = _ridge_xy()
    viz = ferrum.LearningCurveVisualizer(model, cv=3, random_state=0).fit(X, y)
    r = repr(viz)
    assert "final_test_score=" in r
    assert viz._fitted


def test_validation_curve_visualizer_records_best_param():
    model, X, y = _ridge_xy()
    viz = ferrum.ValidationCurveVisualizer(
        model, "alpha", [0.1, 1.0, 10.0], cv=3,
    ).fit(X, y)
    r = repr(viz)
    assert "best_param=" in r
    assert "best_test_score=" in r


def test_cv_scores_visualizer_records_mean_and_std():
    model, X, y = _ridge_xy()
    viz = ferrum.CVScoresVisualizer(model, cv=3, random_state=0).fit(X, y)
    r = repr(viz)
    assert "test_mean=" in r
    assert "test_std=" in r


def test_alpha_selection_visualizer_records_best_alpha():
    model, X, y = _ridge_xy()
    viz = ferrum.AlphaSelectionVisualizer(
        model, [0.01, 0.1, 1.0, 10.0], cv=3,
    ).fit(X, y)
    r = repr(viz)
    assert "best_alpha=" in r


def test_visualizer_show_returns_chart():
    model, X, y = _ridge_xy()
    viz = ferrum.LearningCurveVisualizer(model, cv=3, random_state=0).fit(X, y)
    chart = viz.show()
    assert "<svg" in chart.show_svg()


def test_visualizer_show_before_fit_raises():
    viz = ferrum.LearningCurveVisualizer(model=None)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()
