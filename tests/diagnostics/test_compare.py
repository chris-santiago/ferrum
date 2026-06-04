"""Phase 10h — ComparedModelSource + compare= figure-function routes."""

from __future__ import annotations

import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


_BINARY_FEATURES = ["f0", "f1", "f2", "f3"]


def _binary_setup():
    df = load_dataset("binary_classification")
    X = df.select(_BINARY_FEATURES)
    m = load_fixture("binary_logistic")
    return X, df["y"], m


# --- ComparedModelSource directly -----------------------------------


def test_compare_returns_compared_source():
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m, "b": m}, X, y)
    assert isinstance(cms, ferrum.ComparedModelSource)
    assert cms.model_names == ["a", "b"]


def test_compare_dispatches_roc_with_model_column():
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"left": m, "right": m}, X, y)
    roc = cms.roc_curve()
    assert "model" in roc.columns
    assert set(roc["model"].unique().to_list()) == {"left", "right"}
    # Each per-model frame is non-empty.
    for name in cms.model_names:
        sub = roc.filter(pl.col("model") == name)
        assert sub.height > 0


def test_compare_dispatches_predictions():
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m, "b": m}, X, y)
    preds = cms.predictions()
    assert "model" in preds.columns
    assert preds.height == 2 * X.height  # one frame per model, concatenated


def test_compare_empty_dict_raises():
    with pytest.raises(ValueError, match="at least one source"):
        ferrum.ComparedModelSource({})


def test_compare_model_attr_raises():
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m}, X, y)
    with pytest.raises(AttributeError, match="no single model"):
        cms._model
    with pytest.raises(AttributeError, match="no single model"):
        cms.model


def test_compare_unknown_attr_raises():
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m}, X, y)
    with pytest.raises(AttributeError, match="Methods routed"):
        cms.nope_not_a_method()


def test_compare_X_y_resolve_from_first_source():
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m, "b": m}, X, y)
    # Internal access — chart builders sometimes need raw X/y.
    assert cms._X.height == X.height
    assert cms._y.len() == X.height


# --- Figure-function dispatch ---------------------------------------


def test_compare_dict_positional_via_roc_chart():
    X, y, m = _binary_setup()
    chart = ferrum.roc_chart({"alpha": m, "beta": m}, X, y)
    svg = chart.to_svg()
    assert "<svg" in svg


def test_compare_kwarg_route_via_roc_chart():
    X, y, m = _binary_setup()
    chart = ferrum.roc_chart(m, X, y, compare={"alt": m})
    svg = chart.to_svg()
    assert "<svg" in svg


def test_compare_kwarg_route_via_pr_chart():
    X, y, m = _binary_setup()
    chart = ferrum.pr_chart(m, X, y, compare={"alt": m})
    svg = chart.to_svg()
    assert "<svg" in svg


def test_compare_invalid_kwarg_raises():
    X, y, m = _binary_setup()
    with pytest.raises(TypeError, match="dict\\[str, model\\]"):
        ferrum.roc_chart(m, X, y, compare=["not", "a", "dict"])  # type: ignore[arg-type]


def test_compare_calibration_compare_kwarg():
    """calibration_chart accepts the canonical compare= multi-model kwarg,
    matching the rest of the figure-function family (roc_chart, pr_chart, ...).
    The base positional model is labelled "base"; compare= keys supply the
    additional model names.
    """
    X, y, m = _binary_setup()
    chart = ferrum.calibration_chart(m, X, y, compare={"alt": m})
    assert "<svg" in chart.to_svg()


def test_compare_calibration_dict_positional():
    X, y, m = _binary_setup()
    chart = ferrum.calibration_chart({"alpha": m, "beta": m}, X=X, y=y)
    assert "<svg" in chart.to_svg()


def test_compared_source_passthrough_via_figure():
    """Figure functions accept an already-built ComparedModelSource."""
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m, "b": m}, X, y)
    chart = ferrum.roc_chart(cms)
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# Issue-2 regression: ComparedModelSource proxies _capabilities
# ---------------------------------------------------------------------------


def test_compare_capabilities_proxied_from_first_source():
    """_capabilities must resolve to the first wrapped source's frozenset."""
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m, "b": m}, X, y)
    caps = cms._capabilities
    assert isinstance(caps, frozenset)
    # The binary_logistic fixture exposes predict_proba.
    assert "predict_proba" in caps


def test_compare_capabilities_not_attribute_error():
    """Accessing _capabilities must not raise AttributeError."""
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m}, X, y)
    # Must not raise
    _ = cms._capabilities
