"""Phase 10d feature-importance / SHAP / PDP visualizer tests."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


@pytest.fixture(scope="module")
def rf_source():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


@pytest.fixture(scope="module")
def ridge_source():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


# --- .importances() -----------------------------------------------------


def test_importances_builtin_rf(rf_source):
    imp = rf_source.importances(method="builtin")
    assert set(imp.columns) == {"feature", "importance", "std", "rank"}
    assert imp.height == 5
    assert imp["rank"][0] == 1
    # builtin std is zero by construction.
    assert float(imp["std"].max()) == pytest.approx(0.0)
    # Sorted by descending |importance|.
    vals = imp["importance"].to_list()
    assert all(abs(vals[i]) >= abs(vals[i + 1]) for i in range(len(vals) - 1))


def test_importances_builtin_coef_path(ridge_source):
    """Ridge exposes coef_ but not feature_importances_, exercising the
    abs(coef) branch."""
    imp = ridge_source.importances(method="builtin")
    assert imp.height == 5
    assert (imp["importance"] >= 0).all()


def test_importances_permutation_populates_std(rf_source):
    imp = rf_source.importances(
        method="permutation", n_repeats=5, random_state=0,
    )
    assert imp.height == 5
    # At least one feature should have non-zero std after a real permutation run.
    assert float(imp["std"].max()) > 0.0
    assert (imp["std"] >= 0).all()


def test_importances_invalid_method(rf_source):
    with pytest.raises(ValueError, match="importances\\(method="):
        rf_source.importances(method="not_a_method")


def test_importances_no_capability_raises():
    """A model that exposes neither feature_importances_ nor coef_ should
    raise AttributeError on builtin."""

    class _DumbModel:
        def predict(self, X):
            return np.zeros(len(X))

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    src = ferrum.ModelSource(_DumbModel(), X, df["y"])
    with pytest.raises(AttributeError, match="feature_importances_"):
        src.importances(method="builtin")


def test_importances_caches_per_method(rf_source):
    """Builtin and permutation share a method+kwargs cache key; different
    methods must NOT collide."""
    a = rf_source.importances(method="builtin")
    b = rf_source.importances(method="permutation", n_repeats=3, random_state=0)
    # Different std columns prove the two calls produced independent frames.
    assert float(a["std"].sum()) == 0.0
    assert float(b["std"].sum()) > 0.0


# --- importance_chart figure function -----------------------------------


def test_importance_chart_figure_function():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.importance_chart(model, X, df["y"])
    svg = chart.show_svg()
    assert "<svg" in svg
    # 5 features → 5 horizontal bars + chart-border / clip rects.
    assert svg.count("<rect ") >= 5


def test_importance_chart_top_k_truncates():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.importance_chart(model, X, df["y"], top_k=2)
    # Inspect underlying data: only 2 bars worth of rows.
    assert chart._data.height == 2


def test_importance_chart_vertical_orient():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.importance_chart(model, X, df["y"], orient="vertical")
    svg = chart.show_svg()
    assert "<svg" in svg


def test_mark_importance_invalid_orient():
    df = pl.DataFrame({
        "feature": ["a", "b"], "importance": [0.5, 0.3], "std": [0.0, 0.0],
        "imp_lower": [0.5, 0.3], "imp_upper": [0.5, 0.3], "rank": [1, 2],
    })
    with pytest.raises(ValueError, match="orient="):
        ferrum.Chart(df).mark_importance(orient="diagonal").show_svg()


# --- FeatureImportancesVisualizer ---------------------------------------


def test_feature_importances_visualizer_fit_and_repr():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.FeatureImportancesVisualizer(model).fit(X, df["y"])
    assert "top_feature_importance" in repr(viz)
    chart = viz.show()
    assert "<svg" in chart.show_svg()


def test_feature_importances_visualizer_unfit_raises():
    model = load_fixture("regression_rf")
    viz = ferrum.FeatureImportancesVisualizer(model)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()


def test_feature_importances_visualizer_permutation():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.FeatureImportancesVisualizer(
        model, method="permutation", random_state=0,
    ).fit(X, df["y"])
    chart = viz.show()
    assert "<svg" in chart.show_svg()
