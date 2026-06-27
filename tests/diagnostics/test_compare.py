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


# ---------------------------------------------------------------------------
# T4.6 part E: the proxied-attribute set is derived from BaseSource's property
# descriptors, not a hand-maintained literal tuple.
# ---------------------------------------------------------------------------


def test_compare_proxied_attrs_derived_from_base_properties():
    """The proxy set equals BaseSource's public properties + private aliases
    (minus ``model``), so a new property proxies automatically."""
    from ferrum.diagnostics.sources._base import BaseSource
    from ferrum.diagnostics.sources._compared import _COMPARED_PROXIED_ATTRS

    public = {
        name
        for name, attr in vars(BaseSource).items()
        if isinstance(attr, property) and name != "model"
    }
    expected = public | {f"_{name}" for name in public}
    assert set(_COMPARED_PROXIED_ATTRS) == expected
    # The current BaseSource surface — X / y / feature_names / capabilities.
    assert "feature_names" in _COMPARED_PROXIED_ATTRS
    assert "_feature_names" in _COMPARED_PROXIED_ATTRS
    assert "model" not in _COMPARED_PROXIED_ATTRS
    assert "_model" not in _COMPARED_PROXIED_ATTRS


def test_compare_proxies_all_public_properties():
    """Every public BaseSource property resolves through the wrapper."""
    X, y, m = _binary_setup()
    cms = ferrum.ModelSource.compare({"a": m, "b": m}, X, y)
    first = next(iter(cms._sources.values()))
    assert list(cms.feature_names) == list(first.feature_names)
    assert cms.capabilities == first.capabilities
    assert cms.X.height == first.X.height
    assert cms.y.len() == first.y.len()


def test_compare_picks_up_new_base_property(monkeypatch):
    """A property added to BaseSource is proxied without editing _compared."""
    from ferrum.diagnostics.sources import _compared
    from ferrum.diagnostics.sources._base import BaseSource

    # Add a temporary property to BaseSource and re-derive the set.
    monkeypatch.setattr(
        BaseSource,
        "n_features",
        property(lambda self: len(self._feature_names)),
        raising=False,
    )
    derived = _compared._collect_compared_proxied_attrs()
    assert "n_features" in derived
    assert "_n_features" in derived


# ---------------------------------------------------------------------------
# Task 1: _compose_compare helper (GH issue #35)
# ---------------------------------------------------------------------------


def _make_fake_chart():
    """Return a minimal renderable Chart for use in fake builders."""
    import polars as pl

    return ferrum.Chart(pl.DataFrame({"x": [1, 2], "y": [3, 4]})).mark_point().encode(x="x", y="y")


def test_compose_compare_returns_concat_chart_with_one_panel_per_model():
    """_compose_compare returns a ConcatChart with exactly one child per model."""
    from ferrum.composition import ConcatChart
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare

    def fake_builder(source, **kwargs):
        return _make_fake_chart()

    cms = ComparedModelSource({"alpha": None, "beta": None})
    result = _compose_compare(
        cms,
        fake_builder,
        builder_kwargs={},
        resolve={"x": "shared", "y": "shared"},
    )

    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2


def test_compose_compare_labels_chart_children_with_model_names():
    """Every Chart child carries its model name as a visible title."""
    from ferrum.composition import ConcatChart
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare
    from ferrum.title import Title

    def fake_builder(source, **kwargs):
        return _make_fake_chart()

    cms = ComparedModelSource({"alpha": None, "beta": None})
    result = _compose_compare(
        cms,
        fake_builder,
        builder_kwargs={},
        resolve={"x": "shared", "y": "shared"},
    )

    for child, name in zip(result.charts, ["alpha", "beta"]):
        # Chart.properties(title=name) stores a Title object on _title.
        assert isinstance(child._title, Title)
        assert child._title.text == name


def test_compose_compare_labels_composite_children_with_model_names():
    """Every composite child carries its model name as a figure-level title."""
    from ferrum.composition import ConcatChart
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare

    def composite_builder(source, **kwargs):
        c1 = _make_fake_chart()
        c2 = _make_fake_chart()
        return ConcatChart(c1, c2)

    cms = ComparedModelSource({"model_a": None, "model_b": None, "model_c": None})
    result = _compose_compare(
        cms,
        composite_builder,
        builder_kwargs={},
        resolve={"x": "independent", "y": "independent"},
        columns=2,
    )

    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 3
    for child, name in zip(result.charts, ["model_a", "model_b", "model_c"]):
        # Composite .properties(title=name) stores in _figure_title.
        assert isinstance(child, ConcatChart)
        assert child._figure_title == name


def test_compose_compare_forwards_resolve_to_concat_chart():
    """The resolve dict is passed through to the outer ConcatChart unchanged."""
    from ferrum.composition import ConcatChart
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare

    def fake_builder(source, **kwargs):
        return _make_fake_chart()

    cms = ComparedModelSource({"m1": None, "m2": None})
    resolve = {"x": "shared", "y": "independent"}
    result = _compose_compare(
        cms,
        fake_builder,
        builder_kwargs={},
        resolve=resolve,
    )

    assert result._resolve == resolve


def test_compose_compare_default_columns_is_number_of_models():
    """When columns is omitted, all panels are in a single row."""
    from ferrum.composition import ConcatChart
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare

    def fake_builder(source, **kwargs):
        return _make_fake_chart()

    cms = ComparedModelSource({"a": None, "b": None, "c": None})
    result = _compose_compare(
        cms,
        fake_builder,
        builder_kwargs={},
        resolve={},
    )

    assert result._columns == 3


def test_compose_compare_explicit_columns_forwarded():
    """Explicit columns= value is passed through to the outer ConcatChart."""
    from ferrum.composition import ConcatChart
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare

    def fake_builder(source, **kwargs):
        return _make_fake_chart()

    cms = ComparedModelSource({"a": None, "b": None, "c": None, "d": None})
    result = _compose_compare(
        cms,
        fake_builder,
        builder_kwargs={},
        resolve={},
        columns=2,
    )

    assert result._columns == 2


def test_compose_compare_builder_kwargs_forwarded():
    """builder_kwargs are forwarded verbatim to the builder for every model."""
    from ferrum.diagnostics.sources._compared import ComparedModelSource
    from ferrum.plots._helpers import _compose_compare

    received_kwargs = []

    def recording_builder(source, **kwargs):
        received_kwargs.append(dict(kwargs))
        return _make_fake_chart()

    cms = ComparedModelSource({"m1": None, "m2": None})
    _compose_compare(
        cms,
        recording_builder,
        builder_kwargs={"feature": "age", "n_jobs": 2},
        resolve={},
    )

    assert len(received_kwargs) == 2
    assert all(kw == {"feature": "age", "n_jobs": 2} for kw in received_kwargs)
