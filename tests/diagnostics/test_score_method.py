"""Tests for score() method across the FerrumVisualizer hierarchy.

Group A: visualizers where score() is semantically meaningful — they wrap
a fitted estimator and should return ``estimator.score(X_test, y_test)``.

Group B: visualizers where score() is a no-op — no test-set metric is
computable. They must NOT raise NotImplementedError; they must return 0.0.
"""
from __future__ import annotations

import numpy as np
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_data():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]
    return model, X, y


@pytest.fixture(scope="module")
def binary_data():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    return model, X, y


@pytest.fixture(scope="module")
def clustering_data():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    return model, df


# ---------------------------------------------------------------------------
# Group A — score() must return a float matching estimator.score(X_test, y_test)
# ---------------------------------------------------------------------------


def test_learning_curve_visualizer_score_returns_float(regression_data):
    model, X, y = regression_data
    viz = ferrum.LearningCurveVisualizer(model, cv=3, random_state=0).fit(X, y)
    result = viz.score(X, y)
    assert isinstance(result, float)


def test_learning_curve_visualizer_score_matches_estimator(regression_data):
    model, X, y = regression_data
    viz = ferrum.LearningCurveVisualizer(model, cv=3, random_state=0).fit(X, y)
    result = viz.score(X, y)
    expected = float(model.score(X, y))
    assert result == pytest.approx(expected)


def test_learning_curve_visualizer_score_does_not_raise(regression_data):
    model, X, y = regression_data
    viz = ferrum.LearningCurveVisualizer(model, cv=3, random_state=0).fit(X, y)
    # Must not raise NotImplementedError
    viz.score(X, y)


def test_validation_curve_visualizer_score_returns_float(regression_data):
    model, X, y = regression_data
    viz = ferrum.ValidationCurveVisualizer(
        model, "alpha", [0.1, 1.0, 10.0], cv=3
    ).fit(X, y)
    result = viz.score(X, y)
    assert isinstance(result, float)


def test_validation_curve_visualizer_score_matches_estimator(regression_data):
    model, X, y = regression_data
    viz = ferrum.ValidationCurveVisualizer(
        model, "alpha", [0.1, 1.0, 10.0], cv=3
    ).fit(X, y)
    result = viz.score(X, y)
    expected = float(model.score(X, y))
    assert result == pytest.approx(expected)


def test_validation_curve_visualizer_score_does_not_raise(regression_data):
    model, X, y = regression_data
    viz = ferrum.ValidationCurveVisualizer(
        model, "alpha", [0.1, 1.0, 10.0], cv=3
    ).fit(X, y)
    viz.score(X, y)


def test_cv_scores_visualizer_score_returns_float(regression_data):
    model, X, y = regression_data
    viz = ferrum.CVScoresVisualizer(model, cv=3, random_state=0).fit(X, y)
    result = viz.score(X, y)
    assert isinstance(result, float)


def test_cv_scores_visualizer_score_matches_estimator(regression_data):
    model, X, y = regression_data
    viz = ferrum.CVScoresVisualizer(model, cv=3, random_state=0).fit(X, y)
    result = viz.score(X, y)
    expected = float(model.score(X, y))
    assert result == pytest.approx(expected)


def test_cv_scores_visualizer_score_does_not_raise(regression_data):
    model, X, y = regression_data
    viz = ferrum.CVScoresVisualizer(model, cv=3, random_state=0).fit(X, y)
    viz.score(X, y)


def test_alpha_selection_visualizer_score_returns_float(regression_data):
    model, X, y = regression_data
    viz = ferrum.AlphaSelectionVisualizer(
        model, [0.01, 0.1, 1.0, 10.0], cv=3
    ).fit(X, y)
    result = viz.score(X, y)
    assert isinstance(result, float)


def test_alpha_selection_visualizer_score_matches_estimator(regression_data):
    model, X, y = regression_data
    viz = ferrum.AlphaSelectionVisualizer(
        model, [0.01, 0.1, 1.0, 10.0], cv=3
    ).fit(X, y)
    result = viz.score(X, y)
    expected = float(model.score(X, y))
    assert result == pytest.approx(expected)


def test_alpha_selection_visualizer_score_does_not_raise(regression_data):
    model, X, y = regression_data
    viz = ferrum.AlphaSelectionVisualizer(
        model, [0.01, 0.1, 1.0, 10.0], cv=3
    ).fit(X, y)
    viz.score(X, y)


# ---------------------------------------------------------------------------
# Group B — score() must NOT raise NotImplementedError; must return 0.0
# ---------------------------------------------------------------------------


def test_class_balance_visualizer_score_does_not_raise(binary_data):
    _, X, y = binary_data
    viz = ferrum.ClassBalanceVisualizer().fit(X, y)
    result = viz.score(X, y)
    assert result == 0.0


def test_rank1d_visualizer_score_does_not_raise(regression_data):
    _, X, y = regression_data
    viz = ferrum.Rank1DVisualizer(algorithm="variance").fit(X, y)
    result = viz.score(X, y)
    assert result == 0.0


def test_rank2d_visualizer_score_does_not_raise(regression_data):
    _, X, _ = regression_data
    viz = ferrum.Rank2DVisualizer(algorithm="pearson").fit(X)
    result = viz.score(X, None)
    assert result == 0.0


def test_manifold_visualizer_score_does_not_raise(clustering_data):
    model, df = clustering_data
    X = df.select([c for c in df.columns if c != "label"])
    viz = ferrum.ManifoldVisualizer(model, method="pca").fit(X)
    result = viz.score(X, None)
    assert result == 0.0


def test_silhouette_visualizer_score_does_not_raise(clustering_data):
    model, df = clustering_data
    viz = ferrum.SilhouetteVisualizer(model).fit(df)
    result = viz.score(df, None)
    assert result == 0.0


def test_elbow_visualizer_score_does_not_raise(clustering_data):
    from sklearn.cluster import KMeans
    _, df = clustering_data
    X = df.select([c for c in df.columns if c != "label"])
    viz = ferrum.ElbowVisualizer(KMeans, ks=range(2, 5)).fit(X)
    result = viz.score(X, None)
    assert result == 0.0


def test_shap_visualizer_score_does_not_raise(binary_data):
    import warnings
    model, X, y = binary_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        viz = ferrum.SHAPVisualizer(model, kind="bar").fit(X, y)
    result = viz.score(X, y)
    assert result == 0.0


def test_parallel_coordinates_visualizer_score_does_not_raise(regression_data):
    _, X, y = regression_data
    viz = ferrum.ParallelCoordinatesVisualizer().fit(X, y)
    result = viz.score(X, y)
    assert result == 0.0
