from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_fixture, load_dataset


class _DuckModel:
    """Minimal duck-typed model: predict only."""
    def predict(self, X):
        return np.zeros(len(X))


def test_constructor_accepts_polars_dataframe():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    source = ferrum.ModelSource(_DuckModel(), df, y=[0, 1])
    assert source.feature_names == ["a", "b"]


def test_constructor_accepts_numpy_array():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    source = ferrum.ModelSource(_DuckModel(), X, y=[0, 1])
    assert source.feature_names == ["f0", "f1"]


def test_capability_detection():
    df = pl.DataFrame({"a": [1.0]})
    source = ferrum.ModelSource(_DuckModel(), df)
    assert "predict" in source.capabilities
    assert "predict_proba" not in source.capabilities


def test_predictions_requires_predict():
    class NoPredict:
        pass
    df = pl.DataFrame({"a": [1.0]})
    source = ferrum.ModelSource(NoPredict(), df, y=[0.0])
    with pytest.raises(AttributeError, match="predict"):
        source.predictions()


def test_predictions_against_ridge_fixture():
    """Studentized residuals computed for linear estimator."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    source = ferrum.ModelSource(model, X, y)
    pred = source.predictions()
    assert pred.columns == ["y_true", "y_pred", "residual", "studentized_residual"]
    assert pred.shape == (df.height, 4)
    np.testing.assert_allclose(
        pred["residual"].to_numpy(),
        pred["y_true"].to_numpy() - pred["y_pred"].to_numpy(),
        rtol=1e-12,
    )


def test_probabilities_against_binary_logistic_fixture():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    source = ferrum.ModelSource(model, X, y)
    proba = source.probabilities()
    proba_cols = [c for c in proba.columns if c.startswith("proba_")]
    assert len(proba_cols) == 2
    sums = proba.select(proba_cols).to_numpy().sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)


def test_probabilities_caching():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    p1 = source.probabilities()
    p2 = source.probabilities()
    assert p1 is p2  # cache returns same object
