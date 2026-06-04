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
    svg = chart.to_svg()
    assert svg.startswith("<svg") or svg.lstrip().startswith("<svg")


def test_learning_curve_chart_errorbar_style():
    model, X, y = _ridge_xy()
    chart = ferrum.learning_curve_chart(
        model,
        X,
        y,
        cv=3,
        ci_style="errorbar",
        random_state=0,
    )
    assert "<svg" in chart.to_svg()


def test_learning_curve_chart_invalid_ci_style():
    # Schwabish SB3 (2026-05-11): the direct-label overlay forces the
    # pending stat-mark to desugar at chart construction time (the ``+``
    # operator resolves it), so invalid ci_style now surfaces at the
    # ``learning_curve_chart`` call rather than later at ``to_svg``.
    model, X, y = _ridge_xy()
    with pytest.raises(ValueError, match="ci_style"):
        ferrum.learning_curve_chart(
            model,
            X,
            y,
            cv=3,
            ci_style="invalid",
            random_state=0,
        )


# --- validation_curve_chart -----------------------------------------------


def test_validation_curve_chart_linear_x():
    model, X, y = _ridge_xy()
    chart = ferrum.validation_curve_chart(
        model,
        X,
        y,
        param="alpha",
        values=[0.1, 1.0, 10.0],
        cv=3,
        log_scale=False,
    )
    assert "<svg" in chart.to_svg()


def test_validation_curve_chart_auto_log_scale():
    model, X, y = _ridge_xy()
    # alpha range spans 4 orders of magnitude → auto log.
    chart = ferrum.validation_curve_chart(
        model,
        X,
        y,
        param="alpha",
        values=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=3,
        log_scale="auto",
    )
    assert "<svg" in chart.to_svg()


def test_validation_curve_chart_requires_values():
    model, X, y = _ridge_xy()
    with pytest.raises(ValueError, match="values"):
        ferrum.validation_curve_chart(model, X, y, param="alpha")


# --- cv_scores_chart ------------------------------------------------------


def test_cv_scores_chart_box():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="box")
    assert "<svg" in chart.to_svg()


def test_cv_scores_chart_bar_aggregates():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="bar")
    assert "<svg" in chart.to_svg()


def test_cv_scores_chart_strip():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="strip")
    assert "<svg" in chart.to_svg()


def test_cv_scores_chart_filter_split():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="box", split="test")
    assert "<svg" in chart.to_svg()


def test_cv_scores_chart_invalid_kind():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="invalid")
    with pytest.raises(ValueError, match="kind"):
        chart.to_svg()


# --- alpha_selection_chart -----------------------------------------------


def test_alpha_selection_chart_default():
    model, X, y = _ridge_xy()
    chart = ferrum.alpha_selection_chart(
        model,
        X,
        y,
        alphas=[0.1, 1.0, 10.0],
        cv=3,
    )
    assert "<svg" in chart.to_svg()


def test_alpha_selection_chart_highlight_best_injects_column():
    model, X, y = _ridge_xy()
    chart = ferrum.alpha_selection_chart(
        model,
        X,
        y,
        alphas=[0.1, 1.0, 10.0, 100.0],
        cv=3,
        highlight_best=True,
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
        model,
        X,
        y,
        alphas=[0.1, 1.0, 10.0],
        cv=3,
        highlight_best=False,
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
        model,
        "alpha",
        [0.1, 1.0, 10.0],
        cv=3,
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
        model,
        [0.01, 0.1, 1.0, 10.0],
        cv=3,
    ).fit(X, y)
    r = repr(viz)
    assert "best_alpha=" in r


def test_validation_curve_visualizer_best_param_matches_max_mean_score():
    """Regression for the dedupe fix: the headline metric must point at
    the row with the maximum mean test score after collapsing the
    per-fold rows down to one row per param_value.
    """
    model, X, y = _ridge_xy()
    values = [0.01, 0.1, 1.0, 10.0, 100.0]
    viz = ferrum.ValidationCurveVisualizer(
        model,
        "alpha",
        values,
        cv=3,
        random_state=0,
    ).fit(X, y)
    df = viz._source.validation_curve(
        "alpha",
        values,
        cv=3,
        scoring=None,
    )
    test_rows = df.filter(pl.col("split") == "test").unique(
        subset=["param_value"], keep="first", maintain_order=True
    )
    expected_best_score = float(test_rows["mean_score"].max())
    expected_best_param = float(
        test_rows.filter(
            pl.col("mean_score") == expected_best_score,
        )["param_value"][0]
    )
    assert viz._metrics["best_test_score"] == pytest.approx(expected_best_score)
    assert viz._metrics["best_param"] == pytest.approx(expected_best_param)


def test_alpha_selection_visualizer_best_alpha_matches_max_mean_score():
    """Regression for the dedupe fix: the headline metric must point at
    the alpha with the maximum mean test score after collapsing per-fold
    rows down to one row per alpha.
    """
    model, X, y = _ridge_xy()
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    viz = ferrum.AlphaSelectionVisualizer(
        model,
        alphas,
        cv=3,
        random_state=0,
    ).fit(X, y)
    df = viz._source.alpha_selection(alphas, cv=3, scoring=None)
    agg = df.unique(subset=["alpha"], keep="first", maintain_order=True)
    expected_best_score = float(agg["mean_score"].max())
    expected_best_alpha = float(agg.filter(pl.col("mean_score") == expected_best_score)["alpha"][0])
    assert viz._metrics["best_score"] == pytest.approx(expected_best_score)
    assert viz._metrics["best_alpha"] == pytest.approx(expected_best_alpha)


def test_visualizer_show_returns_chart():
    model, X, y = _ridge_xy()
    viz = ferrum.LearningCurveVisualizer(model, cv=3, random_state=0).fit(X, y)
    chart = viz.show()
    assert "<svg" in chart.to_svg()


def test_visualizer_show_before_fit_raises():
    viz = ferrum.LearningCurveVisualizer(model=None)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()


# --- mark_kwargs forwarding + validation ---------------------------------


def _line_layer_mark_kwargs(chart, mark_name: str) -> dict:
    """Resolve a pending composite mark and return the line layer's
    ``mark_kwargs`` dict (the conventional 'primary visual' layer for
    multi-layer curve marks).
    """
    resolved = chart._resolve_pending()
    for layer in resolved._layers or []:
        if layer.mark == "line":
            return dict(layer.mark_kwargs or {})
    raise AssertionError(f"no line layer found in resolved {mark_name} chart")


def test_learning_curve_forwards_mark_kwargs_to_line_layer():
    model, X, y = _ridge_xy()
    chart = ferrum.learning_curve_chart(model, X, y, cv=3, random_state=0)
    chart = chart._clone()
    chart = chart.mark_learning_curve(stroke="#ff0000", stroke_width=3)
    kw = _line_layer_mark_kwargs(chart, "learning_curve")
    assert kw["stroke"] == "#ff0000"
    assert kw["stroke_width"] == 3


def test_validation_curve_forwards_mark_kwargs():
    model, X, y = _ridge_xy()
    chart = ferrum.validation_curve_chart(
        model,
        X,
        y,
        param="alpha",
        values=[0.1, 1.0, 10.0],
        cv=3,
    )
    chart = chart._clone().mark_validation_curve(
        log_scale=True,
        stroke_width=2,
    )
    kw = _line_layer_mark_kwargs(chart, "validation_curve")
    assert kw["stroke_width"] == 2


def test_alpha_selection_forwards_mark_kwargs():
    model, X, y = _ridge_xy()
    chart = ferrum.alpha_selection_chart(
        model,
        X,
        y,
        alphas=[0.1, 1.0, 10.0],
        cv=3,
    )
    chart = chart._clone().mark_alpha_selection(stroke="#00aa00")
    kw = _line_layer_mark_kwargs(chart, "alpha_selection")
    assert kw["stroke"] == "#00aa00"


def test_cv_scores_bar_forwards_mark_kwargs():
    model, X, y = _ridge_xy()
    chart = ferrum.cv_scores_chart(model, X, y, cv=3, kind="bar")
    chart = chart._clone().mark_cv_scores(kind="bar", opacity=0.5)
    resolved = chart._resolve_pending()
    bar = next(L for L in (resolved._layers or []) if L.mark == "bar")
    assert (bar.mark_kwargs or {}).get("opacity") == 0.5


def test_mark_kwargs_unknown_kwarg_raises():
    model, X, y = _ridge_xy()
    bad = ferrum.learning_curve_chart(model, X, y, cv=3, random_state=0)
    bad = bad._clone().mark_learning_curve(not_a_real_kwarg=42)
    with pytest.raises(TypeError, match="not_a_real_kwarg"):
        bad.to_svg()


def test_user_mark_kwargs_override_desugar_defaults():
    """User-supplied kwargs should win on conflict with desugar defaults
    (e.g. ribbon's hard-coded opacity=0.3)."""
    model, X, y = _ridge_xy()
    chart = ferrum.learning_curve_chart(model, X, y, cv=3, random_state=0)
    chart = chart._clone().mark_learning_curve(opacity=0.7)
    resolved = chart._resolve_pending()
    ribbon = next(L for L in (resolved._layers or []) if L.mark == "ribbon")
    assert (ribbon.mark_kwargs or {}).get("opacity") == 0.7
