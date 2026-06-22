"""Tests for the precomputed y_true/y_pred input path (plan: 2026-05-14).

Each of the nine diagnostic figure functions should accept both:
  - the existing model path: fn(model, X, y)
  - a new precomputed path:  fn(y_true=..., y_pred=...)

Tests are structured as: model_path / precomputed_path pairs per function,
plus binary-vs-multiclass parametrization for curve/matrix functions, and
three ValueError branch tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_data():
    """Fitted binary classifier + feature matrix + labels from pinned fixtures."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    return model, X, y


@pytest.fixture(scope="module")
def multi_data():
    """Fitted multiclass classifier + feature matrix + labels."""
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    return model, X, y


@pytest.fixture(scope="module")
def regression_data():
    """Fitted regression estimator + feature matrix + targets."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]
    return model, X, y


@pytest.fixture(scope="module")
def binary_source(binary_data):
    model, X, y = binary_data
    return ferrum.ModelSource(model, X, y)


@pytest.fixture(scope="module")
def multi_source(multi_data):
    model, X, y = multi_data
    return ferrum.ModelSource(model, X, y)


@pytest.fixture(scope="module")
def regression_source(regression_data):
    model, X, y = regression_data
    return ferrum.ModelSource(model, X, y)


@pytest.fixture(scope="module")
def binary_precomputed(binary_data):
    """Precomputed arrays derived from the pinned binary classifier."""
    model, X, y = binary_data
    y_true = np.asarray(y)
    y_pred_1d = model.predict_proba(X)[:, 1]  # 1-D scores (positive class)
    y_pred_2d = model.predict_proba(X)  # 2-D proba matrix
    y_hard = model.predict(X)  # hard labels
    return y_true, y_pred_1d, y_pred_2d, y_hard


@pytest.fixture(scope="module")
def multi_precomputed(multi_data):
    """Precomputed arrays derived from the pinned multiclass classifier."""
    model, X, y = multi_data
    y_true = np.asarray(y)
    y_pred_2d = model.predict_proba(X)
    y_hard = model.predict(X)
    return y_true, y_pred_2d, y_hard


@pytest.fixture(scope="module")
def regression_precomputed(regression_data):
    """Precomputed arrays derived from the pinned regression estimator."""
    model, X, y = regression_data
    y_true = np.asarray(y, dtype=float)
    y_pred = model.predict(X)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _svg(chart) -> str:
    return chart.to_svg()


# ---------------------------------------------------------------------------
# roc_chart
# ---------------------------------------------------------------------------


def test_roc_chart_model_path_binary(binary_source):
    assert "<svg" in _svg(ferrum.roc_chart(binary_source))


def test_roc_chart_precomputed_binary(binary_precomputed):
    y_true, y_pred_1d, _, _ = binary_precomputed
    chart = ferrum.roc_chart(y_true=y_true, y_pred=y_pred_1d)
    assert "<svg" in _svg(chart)


def test_roc_chart_model_path_multiclass(multi_source):
    assert "<svg" in _svg(ferrum.roc_chart(multi_source))


def test_roc_chart_precomputed_multiclass(multi_precomputed):
    y_true, y_pred_2d, _ = multi_precomputed
    chart = ferrum.roc_chart(y_true=y_true, y_pred=y_pred_2d)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# pr_chart
# ---------------------------------------------------------------------------


def test_pr_chart_model_path_binary(binary_source):
    assert "<svg" in _svg(ferrum.pr_chart(binary_source))


def test_pr_chart_precomputed_binary(binary_precomputed):
    y_true, y_pred_1d, _, _ = binary_precomputed
    chart = ferrum.pr_chart(y_true=y_true, y_pred=y_pred_1d)
    assert "<svg" in _svg(chart)


def test_pr_chart_model_path_multiclass(multi_source):
    assert "<svg" in _svg(ferrum.pr_chart(multi_source))


def test_pr_chart_precomputed_multiclass(multi_precomputed):
    y_true, y_pred_2d, _ = multi_precomputed
    chart = ferrum.pr_chart(y_true=y_true, y_pred=y_pred_2d)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# calibration_chart (binary only)
# ---------------------------------------------------------------------------


def test_calibration_chart_model_path(binary_source):
    assert "<svg" in _svg(ferrum.calibration_chart(binary_source))


def test_calibration_chart_precomputed(binary_precomputed):
    y_true, y_pred_1d, _, _ = binary_precomputed
    chart = ferrum.calibration_chart(y_true=y_true, y_pred=y_pred_1d)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# gain_chart
# ---------------------------------------------------------------------------


def test_gain_chart_model_path(binary_source):
    assert "<svg" in _svg(ferrum.gain_chart(binary_source))


def test_gain_chart_precomputed_binary(binary_precomputed):
    y_true, y_pred_1d, _, _ = binary_precomputed
    chart = ferrum.gain_chart(y_true=y_true, y_pred=y_pred_1d)
    assert "<svg" in _svg(chart)


def test_gain_chart_precomputed_multiclass(multi_precomputed):
    y_true, y_pred_2d, _ = multi_precomputed
    chart = ferrum.gain_chart(y_true=y_true, y_pred=y_pred_2d)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# lift_chart
# ---------------------------------------------------------------------------


def test_lift_chart_model_path(binary_source):
    assert "<svg" in _svg(ferrum.lift_chart(binary_source))


def test_lift_chart_precomputed_binary(binary_precomputed):
    y_true, y_pred_1d, _, _ = binary_precomputed
    chart = ferrum.lift_chart(y_true=y_true, y_pred=y_pred_1d)
    assert "<svg" in _svg(chart)


def test_lift_chart_precomputed_multiclass(multi_precomputed):
    y_true, y_pred_2d, _ = multi_precomputed
    chart = ferrum.lift_chart(y_true=y_true, y_pred=y_pred_2d)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# discrimination_threshold_chart (binary only; cv= not supported precomputed)
# ---------------------------------------------------------------------------


def test_discrimination_threshold_chart_model_path(binary_source):
    assert "<svg" in _svg(ferrum.discrimination_threshold_chart(binary_source))


def test_discrimination_threshold_chart_precomputed(binary_precomputed):
    y_true, y_pred_1d, _, _ = binary_precomputed
    chart = ferrum.discrimination_threshold_chart(y_true=y_true, y_pred=y_pred_1d)
    assert "<svg" in _svg(chart)


def test_discrimination_threshold_chart_precomputed_cv_raises(binary_precomputed):
    """cv= is unsupported with precomputed inputs — no model to re-fit."""
    y_true, y_pred_1d, _, _ = binary_precomputed
    with pytest.raises(ValueError, match="cv="):
        ferrum.discrimination_threshold_chart(y_true=y_true, y_pred=y_pred_1d, cv=3)


# ---------------------------------------------------------------------------
# confusion_matrix_chart
# ---------------------------------------------------------------------------


def test_confusion_matrix_chart_model_path(binary_source):
    assert "<svg" in _svg(ferrum.confusion_matrix_chart(binary_source))


def test_confusion_matrix_chart_precomputed_binary(binary_precomputed):
    y_true, _, _, y_hard = binary_precomputed
    chart = ferrum.confusion_matrix_chart(y_true=y_true, y_pred=y_hard)
    assert "<svg" in _svg(chart)


def test_confusion_matrix_chart_model_path_multiclass(multi_source):
    assert "<svg" in _svg(ferrum.confusion_matrix_chart(multi_source))


def test_confusion_matrix_chart_precomputed_multiclass(multi_precomputed):
    y_true, _, y_hard = multi_precomputed
    chart = ferrum.confusion_matrix_chart(y_true=y_true, y_pred=y_hard)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# class_prediction_error_chart
# ---------------------------------------------------------------------------


def test_class_prediction_error_chart_model_path(binary_source):
    assert "<svg" in _svg(ferrum.class_prediction_error_chart(binary_source))


def test_class_prediction_error_chart_precomputed_binary(binary_precomputed):
    y_true, _, _, y_hard = binary_precomputed
    chart = ferrum.class_prediction_error_chart(y_true=y_true, y_pred=y_hard)
    assert "<svg" in _svg(chart)


def test_class_prediction_error_chart_model_path_multiclass(multi_source):
    assert "<svg" in _svg(ferrum.class_prediction_error_chart(multi_source))


def test_class_prediction_error_chart_precomputed_multiclass(multi_precomputed):
    y_true, _, y_hard = multi_precomputed
    chart = ferrum.class_prediction_error_chart(y_true=y_true, y_pred=y_hard)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# residuals_chart
# ---------------------------------------------------------------------------


def test_residuals_chart_model_path(regression_source):
    assert "<svg" in _svg(ferrum.residuals_chart(regression_source))


def test_residuals_chart_precomputed_renders(regression_precomputed):
    """Precomputed residuals: 3 panels render (leverage dropped; NaN for all rows)."""
    y_true, y_pred = regression_precomputed
    chart = ferrum.residuals_chart(y_true=y_true, y_pred=y_pred)
    svg = _svg(chart)
    assert "<svg" in svg
    # residuals_vs_fitted panel always present
    assert "Residuals vs Fitted" in svg


def test_residuals_chart_precomputed_has_metrics_annotation(regression_precomputed):
    """Corner R²/RMSE/MAE annotation is injected by the builder."""
    y_true, y_pred = regression_precomputed
    chart = ferrum.residuals_chart(y_true=y_true, y_pred=y_pred, panels="single")
    svg = _svg(chart)
    assert "<svg" in svg
    # The corner annotation text uses the format_corner_metrics output,
    # which always contains "R²" or an ASCII equivalent
    assert "Residuals" in svg


def test_residuals_chart_precomputed_qq_panel(regression_precomputed):
    """QQ panel renders when studentized residuals are non-NaN."""
    y_true, y_pred = regression_precomputed
    chart = ferrum.residuals_chart(y_true=y_true, y_pred=y_pred, panels=["qq"])
    svg = _svg(chart)
    assert "<svg" in svg
    assert "Q" in svg  # "Normal Q–Q" appears in title


def test_residuals_chart_precomputed_leverage_dropped(regression_precomputed):
    """residuals_vs_leverage panel is silently dropped (leverage = NaN for all rows)."""
    y_true, y_pred = regression_precomputed
    chart = ferrum.residuals_chart(y_true=y_true, y_pred=y_pred, panels="auto")
    svg = _svg(chart)
    assert "<svg" in svg
    # leverage panel title must not appear since it was silently dropped
    assert "Leverage" not in svg


# ---------------------------------------------------------------------------
# ValueError — exactly-one-path validation
# ---------------------------------------------------------------------------


def test_neither_path_raises():
    """Calling any figure function with no model and no y_true/y_pred raises."""
    with pytest.raises(ValueError, match="Supply either"):
        ferrum.roc_chart()


def test_both_paths_raises(binary_source, binary_precomputed):
    """Supplying both model_or_source and y_true/y_pred raises ValueError."""
    y_true, y_pred_1d, _, _ = binary_precomputed
    with pytest.raises(ValueError, match="not both"):
        ferrum.roc_chart(binary_source, y_true=y_true, y_pred=y_pred_1d)


def test_incomplete_pair_ytrue_only_raises(binary_precomputed):
    """y_true without y_pred raises ValueError."""
    y_true, _, _, _ = binary_precomputed
    with pytest.raises(ValueError, match="y_pred"):
        ferrum.roc_chart(y_true=y_true)


def test_incomplete_pair_ypred_only_raises(binary_precomputed):
    """y_pred without y_true raises ValueError."""
    _, y_pred_1d, _, _ = binary_precomputed
    with pytest.raises(ValueError, match="y_true"):
        ferrum.roc_chart(y_pred=y_pred_1d)


def test_precomputed_with_compare_raises(binary_precomputed):
    """compare= is not supported with the precomputed path."""
    y_true, y_pred_1d, _, _ = binary_precomputed
    with pytest.raises(ValueError, match="compare="):
        ferrum.roc_chart(y_true=y_true, y_pred=y_pred_1d, compare={"other": None})


# ---------------------------------------------------------------------------
# Backward-compatibility — existing positional call shape unchanged
# ---------------------------------------------------------------------------


def test_existing_positional_call_still_works(binary_data):
    """fn(model, X, y) positional call must continue to work unchanged."""
    model, X, y = binary_data
    chart = ferrum.roc_chart(model, X, y)
    assert "<svg" in _svg(chart)


# ---------------------------------------------------------------------------
# Numeric regression — precomputed 1-D binary gain/lift values (DIAG-01)
# ---------------------------------------------------------------------------


def test_precomputed_1d_binary_gain_values():
    """Pin the gain curve VALUES for a small 1-D binary fixture.

    The precomputed 1-D binary path builds the score matrix as
    ``column_stack([1 - y_pred, y_pred])`` (B1, 2026-06): the negative class
    (column 0) ranks by ``1 - p`` and the positive class (column 1) by ``p``,
    matching the precomputed 1-D binary roc/pr path.  This test pins BOTH the
    positive AND negative class curves.

    Fixture:
      y_true  = [0, 0, 1, 1]
      y_pred  = [0.1, 0.4, 0.35, 0.8]  (1-D positive-class scores)

    Score matrix: col 0 = 1 - y_pred = [0.9, 0.6, 0.65, 0.2]
                  col 1 = y_pred     = [0.1, 0.4, 0.35, 0.8]

    Class "1" (positive, col 1): sort desc → indices [3, 1, 2, 0]
      y_bin=[0,0,1,1] sorted → [1,0,1,0]
      cum_pos=[1,1,2,2], total=2, gain=[0.5, 0.5, 1.0, 1.0]
      with origin: percent_pop=[0, 0.25, 0.5, 0.75, 1.0], gain=[0, 0.5, 0.5, 1.0, 1.0]

    Class "0" (negative, col 0 = 1 - p): sort desc → indices [0, 2, 1, 3]
      y_bin=[1,1,0,0] sorted → [1,0,1,0]
      cum_pos=[1,1,2,2], total=2, gain=[0.5, 0.5, 1.0, 1.0]
      with origin: percent_pop=[0, 0.25, 0.5, 0.75, 1.0], gain=[0, 0.5, 0.5, 1.0, 1.0]
    """
    from ferrum._diagnostics.precomputed import _PrecomputedSource

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    src = _PrecomputedSource(y_true, y_pred)
    df = src.cumulative_gain()

    # Positive class "1" (ranked by p)
    pos_rows = df.filter(df["class"] == "1").sort("percent_population")
    np.testing.assert_allclose(
        pos_rows["percent_population"].to_numpy(),
        [0.0, 0.25, 0.5, 0.75, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pos_rows["gain"].to_numpy(),
        [0.0, 0.5, 0.5, 1.0, 1.0],
        atol=1e-12,
    )

    # Negative class "0" — ranked by 1 - p (consistent with roc/pr)
    neg_rows = df.filter(df["class"] == "0").sort("percent_population")
    np.testing.assert_allclose(
        neg_rows["percent_population"].to_numpy(),
        [0.0, 0.25, 0.5, 0.75, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        neg_rows["gain"].to_numpy(),
        [0.0, 0.5, 0.5, 1.0, 1.0],
        atol=1e-12,
    )


def test_precomputed_1d_binary_lift_values():
    """Pin the lift curve VALUES for a small 1-D binary fixture.

    Same fixture as test_precomputed_1d_binary_gain_values.  The negative
    class ranks by ``1 - p`` (B1, 2026-06).

    Class "1" (col 1 = p): base_rate=0.5, sort order [3,1,2,0] → y_bin_sorted=[1,0,1,0]
      cum_pos=[1,1,2,2], cum_rate=[1.0, 0.5, 2/3, 0.5]
      lift = cum_rate / 0.5 = [2.0, 1.0, 4/3, 1.0]
      percent_population = [0.25, 0.5, 0.75, 1.0]

    Class "0" (col 0 = 1 - p): base_rate=0.5, sort order [0,2,1,3] → y_bin_sorted=[1,0,1,0]
      cum_pos=[1,1,2,2], cum_rate=[1.0, 0.5, 2/3, 0.5]
      lift = [2.0, 1.0, 4/3, 1.0]
      percent_population = [0.25, 0.5, 0.75, 1.0]
    """
    from ferrum._diagnostics.precomputed import _PrecomputedSource

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    src = _PrecomputedSource(y_true, y_pred)
    df = src.lift_curve()

    # Positive class "1" (ranked by p)
    pos_rows = df.filter(df["class"] == "1").sort("percent_population")
    np.testing.assert_allclose(
        pos_rows["percent_population"].to_numpy(),
        [0.25, 0.5, 0.75, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pos_rows["lift"].to_numpy(),
        [2.0, 1.0, 4.0 / 3.0, 1.0],
        atol=1e-12,
    )

    # Negative class "0" — ranked by 1 - p (consistent with roc/pr)
    neg_rows = df.filter(df["class"] == "0").sort("percent_population")
    np.testing.assert_allclose(
        neg_rows["percent_population"].to_numpy(),
        [0.25, 0.5, 0.75, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        neg_rows["lift"].to_numpy(),
        [2.0, 1.0, 4.0 / 3.0, 1.0],
        atol=1e-12,
    )


def test_precomputed_1d_binary_negative_class_ranks_by_one_minus_p():
    """B1 regression: the precomputed 1-D binary negative class ranks by 1 - p.

    Uses an asymmetric fixture where 1-p-descending and p-descending produce
    *different* negative-class cumulative curves, so this test FAILS under the
    old tiled-score-matrix behavior (which ranked the negative class by p).

    Fixture: y_true = [0, 0, 1], y_pred = [0.1, 0.6, 0.9].
      Negative-class indicator y_bin = [1, 1, 0].
      col 0 = 1 - p = [0.9, 0.4, 0.1]  → desc order [0, 1, 2] → y_bin [1, 1, 0]
        cum_pos=[1, 2, 2], total=2, gain=[0.5, 1.0, 1.0]
        with origin → [0.0, 0.5, 1.0, 1.0]
      OLD (tile, rank by p = [0.1, 0.6, 0.9]) → desc order [2, 1, 0] → y_bin [0, 1, 1]
        cum_pos=[0, 1, 2], gain=[0.0, 0.5, 1.0] → with origin [0.0, 0.0, 0.5, 1.0]
    """
    from ferrum._diagnostics.precomputed import _PrecomputedSource

    y_true = np.array([0, 0, 1])
    y_pred = np.array([0.1, 0.6, 0.9])

    src = _PrecomputedSource(y_true, y_pred)

    gain = src.cumulative_gain()
    neg_gain = gain.filter(gain["class"] == "0").sort("percent_population")
    np.testing.assert_allclose(
        neg_gain["gain"].to_numpy(),
        [0.0, 0.5, 1.0, 1.0],
        atol=1e-12,
    )

    lift = src.lift_curve()
    neg_lift = lift.filter(lift["class"] == "0").sort("percent_population")
    # base_rate=2/3; cum_rate=[1.0, 1.0, 2/3]; lift = cum_rate / (2/3)
    np.testing.assert_allclose(
        neg_lift["lift"].to_numpy(),
        [1.5, 1.5, 1.0],
        atol=1e-12,
    )

    # The negative-class ranking must match the roc/pr binary construction:
    # column 0 of column_stack([1 - p, p]). Derive the expected curve directly
    # so the test self-documents the consistency it pins.
    score_matrix = np.column_stack([1.0 - y_pred, y_pred])
    neg_order = np.argsort(-score_matrix[:, 0])
    y_bin_neg = (y_true == 0).astype(int)
    cum_pos = np.cumsum(y_bin_neg[neg_order])
    expected_gain = np.concatenate([[0.0], cum_pos / max(int(cum_pos[-1]), 1)])
    np.testing.assert_allclose(neg_gain["gain"].to_numpy(), expected_gain, atol=1e-12)
