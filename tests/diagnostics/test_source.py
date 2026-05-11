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


# --- 10b: classification curves --------------------------------------


def test_roc_curve_binary_schema():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    roc = source.roc_curve()
    assert set(roc.columns) == {"fpr", "tpr", "threshold", "class", "auc"}
    assert roc.height >= 2
    aucs = roc["auc"].unique().to_list()
    assert len(aucs) == 1  # binary → one AUC
    assert 0.0 <= aucs[0] <= 1.0


def test_roc_curve_multiclass_macro_average():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    roc = source.roc_curve(average="macro")
    classes_seen = set(roc["class"].unique().to_list())
    assert "macro" in classes_seen
    assert len(classes_seen - {"macro"}) >= 3


def test_pr_curve_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve()
    assert set(pr.columns) == {"precision", "recall", "threshold", "class", "ap"}
    assert pr.height >= 2
    aps = pr["ap"].unique().to_list()
    assert len(aps) == 1
    assert 0.0 <= aps[0] <= 1.0


def test_pr_curve_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve()
    classes_seen = set(pr["class"].unique().to_list())
    assert len(classes_seen) >= 3


def test_pr_curve_average_raises():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(NotImplementedError):
        source.pr_curve(average="macro")


def test_roc_curve_caching():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    assert source.roc_curve() is source.roc_curve()


def test_calibration_curve_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    cal = source.calibration_curve(n_bins=5)
    assert set(cal.columns) == {"mean_predicted", "fraction_positive", "count"}
    assert cal.height <= 5
    assert cal.height > 0
    assert (cal["mean_predicted"] >= 0).all()
    assert (cal["mean_predicted"] <= 1).all()
    assert (cal["count"] >= 0).all()


def test_calibration_curve_rejects_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(ValueError, match="binary-classifier only"):
        source.calibration_curve()


def test_cumulative_gain_includes_baseline():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    gain = source.cumulative_gain()
    assert set(gain.columns) == {"percent_population", "gain", "class"}
    classes_seen = set(gain["class"].unique().to_list())
    assert "baseline" in classes_seen
    # Per-class curves start at (0, 0) and end at (1, 1).
    assert gain.filter(pl.col("class") != "baseline").height > 0


def test_lift_curve_baseline_at_one():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    lift = source.lift_curve()
    assert set(lift.columns) == {"percent_population", "lift", "class"}
    baseline = lift.filter(pl.col("class") == "baseline")
    assert baseline.height == 2
    assert (baseline["lift"] == 1.0).all()


def test_discrimination_threshold_schema_and_grid():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    dt = source.discrimination_threshold(n_thresholds=20)
    assert set(dt.columns) == {
        "threshold", "precision", "recall", "f1", "queue_rate",
    }
    assert dt.height == 20
    near_zero = dt.filter(pl.col("threshold") < 0.05)["queue_rate"].to_list()
    assert all(qr > 0.95 for qr in near_zero)
    # queue_rate at threshold=1.0 is small.
    near_one = dt.filter(pl.col("threshold") >= 0.99)["queue_rate"].to_list()
    assert all(qr < 0.05 for qr in near_one)


def test_discrimination_threshold_multiclass_rejects():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(ValueError, match="binary-classifier only"):
        source.discrimination_threshold()


def test_discrimination_threshold_cv_averaging():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    dt = source.discrimination_threshold(n_thresholds=10, cv=3)
    assert dt.height == 10
    assert (dt["precision"] >= 0).all() and (dt["precision"] <= 1).all()
    assert (dt["recall"] >= 0).all() and (dt["recall"] <= 1).all()
    # Sort order is ascending by threshold.
    thresholds = dt["threshold"].to_list()
    assert thresholds == sorted(thresholds)


# --- Phase 10e: model-selection methods ---------------------------------


def test_learning_curve_schema_and_splits():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    lc = source.learning_curve(cv=3)
    assert set(lc.columns) == {
        "train_size", "split", "score",
        "mean_score", "std_score", "lower", "upper",
    }
    assert set(lc["split"].unique().to_list()) == {"train", "test"}
    # 5 default train sizes × 2 splits × 3 folds = 30 rows.
    assert lc.height == 30
    # Per-(train_size, split) mean_score is constant across folds.
    agg = lc.group_by(["train_size", "split"]).agg(
        pl.col("mean_score").n_unique().alias("n_unique"),
    )
    assert (agg["n_unique"] == 1).all()


def test_validation_curve_schema_and_values():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    vc = source.validation_curve("alpha", [0.1, 1.0, 10.0], cv=3)
    assert set(vc.columns) == {
        "param_value", "split", "score",
        "mean_score", "std_score", "lower", "upper",
    }
    assert set(vc["param_value"].unique().to_list()) == {0.1, 1.0, 10.0}
    # 3 alphas × 2 splits × 3 folds = 18 rows.
    assert vc.height == 18


def test_cv_scores_schema_and_count():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    cvs = source.cv_scores(cv=3)
    assert set(cvs.columns) == {"fold", "split", "score"}
    assert set(cvs["split"].unique().to_list()) == {"train", "test"}
    # 3 folds × 2 splits = 6 rows.
    assert cvs.height == 6


def test_alpha_selection_schema_and_alphas():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    al = source.alpha_selection([0.1, 1.0, 10.0], cv=3)
    assert set(al.columns) == {
        "alpha", "fold", "score", "mean_score", "std_score",
    }
    assert set(al["alpha"].unique().to_list()) == {0.1, 1.0, 10.0}
    # 3 alphas × 3 folds = 9 rows.
    assert al.height == 9
