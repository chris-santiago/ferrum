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


def test_compare_items_returns_name_source_pairs_in_construction_order():
    from ferrum.diagnostics.sources._compared import ComparedModelSource

    src_a = object()
    src_b = object()
    src_c = object()
    cms = ComparedModelSource({"a": src_a, "b": src_b, "c": src_c})
    items = cms.items()
    # Public accessor yields (name, source) pairs in registration order.
    assert items == [("a", src_a), ("b", src_b), ("c", src_c)]
    # Names align with the model_names property.
    assert [name for name, _ in items] == cms.model_names


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


def test_compare_calibration_visualizer_render_equals_the_chart():
    """``CalibrationVisualizer(compare=)`` renders what ``calibration_chart
    (compare=)`` renders -- byte for byte (#212, 2026-09-13).

    The visualizer's ``compare=`` is constructor-and-semantics parity, not a
    second render path: both surfaces resolve the same
    ``{"base": ..., **compare}`` source and hand it to the same builder, so
    any divergence here means one of them grew its own resolution.
    """
    from sklearn.linear_model import LogisticRegression

    X, y, base = _binary_setup()
    rival = LogisticRegression(C=0.01, max_iter=1000).fit(X.to_numpy(), y.to_numpy())

    viz = ferrum.CalibrationVisualizer(base, compare={"rival": rival}).fit(X, y)
    assert viz._source.model_names == ["base", "rival"]
    assert (
        viz.show().to_svg()
        == ferrum.calibration_chart(base, X, y, compare={"rival": rival}).to_svg()
    )


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
    first = cms.items()[0][1]
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


# --- prediction_error_chart compare= band is per-model, not pooled ----


_REG_FEATURES = ["f0", "f1", "f2", "f3", "f4"]


def _panel_band_width(panel) -> float:
    """The residual-band width (``q_hi - q_lo``) carried by one compose panel.

    ``_pe_band_lo`` / ``_pe_band_hi`` are ``y_true + q_lo`` / ``y_true + q_hi``,
    so their difference is the constant per-model band width — exactly the
    residual-quantile spread this panel was built from.
    """
    data = panel._data
    assert data is not None
    assert "_pe_band_lo" in data.columns
    assert "_pe_band_hi" in data.columns
    return float((data["_pe_band_hi"] - data["_pe_band_lo"]).max())


def test_prediction_error_chart_compare_band_is_per_model_not_pooled():
    """With ``ci=`` and two models of differing residual distributions, each
    compose panel's band is computed from that model's residuals only.

    A pooled band (the latent defect) would compute one residual-quantile spread
    over the concatenated multi-model frame and apply it to both panels, making
    the band widths identical. Per-model routing makes them differ. The two
    models are chosen so their residual spreads are far apart: a well-fit linear
    model versus a mean-predicting dummy.
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression

    from ferrum.composition import ConcatChart

    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    good = LinearRegression().fit(X.to_numpy(), y.to_numpy())
    poor = DummyRegressor(strategy="mean").fit(X.to_numpy(), y.to_numpy())

    result = ferrum.prediction_error_chart(good, X, y, compare={"poor": poor}, ci=0.9)
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2

    # Panels are ordered base ("base"=good) then compare ("poor").
    good_panel, poor_panel = result.charts
    assert good_panel._title.text == "base"
    assert poor_panel._title.text == "poor"

    good_width = _panel_band_width(good_panel)
    poor_width = _panel_band_width(poor_panel)

    # Per-model bands: the mean-predicting dummy has a far wider residual spread
    # than the fitted linear model. A pooled band would make these equal.
    assert good_width != pytest.approx(poor_width)
    assert poor_width > good_width


# ---------------------------------------------------------------------------
# Determinism regression: shared ordinal scale in importance compare= (#35)
# ---------------------------------------------------------------------------


def test_importance_chart_compare_svg_is_deterministic():
    """importance_chart(model, X, y, compare=...) must produce byte-identical
    SVG across repeated calls in the same process.

    Root cause: the ordinal-domain union helper formerly called col.unique()
    without maintain_order=True.  Polars returns unique values in hash order
    (non-deterministic across calls), making the shared categorical y-axis
    domain non-deterministic and thus the SVG non-deterministic.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    model_a = RandomForestRegressor(n_estimators=10, random_state=0)
    model_b = RandomForestRegressor(n_estimators=10, random_state=1)
    model_a.fit(X.to_numpy(), y.to_numpy())
    model_b.fit(X.to_numpy(), y.to_numpy())

    svgs = {
        ferrum.importance_chart(model_a, X, y, compare={"alt": model_b}).to_svg() for _ in range(3)
    }
    assert len(svgs) == 1, (
        "importance_chart compare= produced non-deterministic SVG across 3 calls; "
        "check the ordinal-domain union helper for maintain_order=True"
    )


def test_pdp_chart_compare_keeps_independent_feature_x():
    """pdp compose must keep each feature's own x range, NOT collapse the two
    features onto one shared x axis.

    pdp facets per feature with intentionally independent x (each feature has
    its own units). The shared-scale union DOES propagate into pdp's faceted
    child, so the compare gate must declare ``resolve={"x":"independent"}`` —
    otherwise the per-feature axes collapse onto one shared range. This test is
    the guard: the two features below have DISJOINT x ranges ([-2,2] and
    [100,110]) with an empty gap between them. With the correct independent x,
    no tick lands in that gap; if the gate ever reverts to a shared x, the
    collapsed [-2,110] axis fills the gap with ticks (~40/60/80) and this
    test fails.
    """
    import re

    import numpy as np
    import polars as pl
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    x_np = np.column_stack(
        [
            rng.uniform(-2, 2, 200),  # f0 in [-2, 2]
            rng.uniform(100, 110, 200),  # f1 in [100, 110]
        ]
    )
    y_np = x_np[:, 0] * 3 - (x_np[:, 1] - 105) * 2 + rng.standard_normal(200) * 0.1
    X = pl.DataFrame({"f0": x_np[:, 0], "f1": x_np[:, 1]})
    y = pl.Series("y", y_np)
    base = RandomForestRegressor(n_estimators=20, random_state=0).fit(x_np, y_np)
    alt = RandomForestRegressor(n_estimators=20, random_state=1).fit(x_np, y_np)

    svg = ferrum.pdp_chart(base, X, y, features=["f0", "f1"], compare={"alt": alt}).to_svg()
    ticks = [float(t) for t in re.findall(r">(-?\d+\.?\d*)</text>", svg)]
    # Each feature's own range is present...
    assert any(-2.5 < t < 2.5 for t in ticks), "f0 x-range [-2,2] missing"
    assert any(95 < t < 115 for t in ticks), "f1 x-range [100,110] missing"
    # ...and the gap between the two disjoint ranges is EMPTY. This is the
    # discriminating check: a collapsed shared x ([-2,110]) places ticks
    # (~40/60/80) in this band; independent per-feature x and the bounded
    # partial-dependence y (|y| < ~16 here) never do.
    gap_ticks = [t for t in ticks if 30 < t < 90]
    assert not gap_ticks, f"x axes collapsed onto a shared range; gap ticks present: {gap_ticks}"


def test_pdp_chart_compare_multiclass_keeps_independent_x_under_feature_times_class():
    """#209 re-verification: with per-class rows the compose gate's
    ``resolve={"x": "independent"}`` still has to hold — the grouping is now
    feature x class, and every model panel carries K curves per feature.
    """
    import re

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 150
    labels = np.array([0, 1, 2] * (n // 3))
    x_np = np.column_stack(
        [
            rng.uniform(-2, 2, n) + labels,  # f0 in ~[-2, 4]
            rng.uniform(100, 110, n) - labels,  # f1 in ~[98, 110]
        ]
    )
    X = pl.DataFrame({"f0": x_np[:, 0], "f1": x_np[:, 1]})
    y = pl.Series("y", labels)
    base = LogisticRegression(max_iter=500).fit(x_np, labels)
    alt = LogisticRegression(max_iter=500, C=0.1).fit(x_np, labels)

    chart = ferrum.pdp_chart(
        base,
        X,
        y,
        features=["f0", "f1"],
        grid_resolution=8,
        compare={"alt": alt},
    )
    svg = chart.to_svg()
    # 2 models x 2 features x 3 classes.
    assert svg.count("<polyline ") == 2 * 2 * 3
    ticks = [float(t) for t in re.findall(r">(-?\d+\.?\d*)</text>", svg)]
    assert any(-2.5 < t < 4.5 for t in ticks), "f0 x-range missing"
    assert any(95 < t < 115 for t in ticks), "f1 x-range missing"
    gap_ticks = [t for t in ticks if 30 < t < 90]
    assert not gap_ticks, f"x axes collapsed onto a shared range; gap ticks present: {gap_ticks}"


# ---------------------------------------------------------------------------
# #213 — discrimination_threshold_chart(compare=) renders small multiples
#
# The classification family otherwise overlays compared models by colour, but
# this chart's colour channel is already spent on the metric legend. Before
# this fix the builder's ``unpivot`` silently dropped the ``model`` column
# (it is in neither ``index=`` nor the resolved metric columns), so both
# models' sweeps stacked into one panel as four interleaved zig-zag polylines
# annotated with a single pooled F1 optimum -- the rival's whenever the rival
# won, contradicting DiscriminationThresholdVisualizer's own repr.
# ---------------------------------------------------------------------------


_N_THRESHOLDS = 20
_N_METRICS = 4  # precision, recall, f1, queue_rate


def _threshold_compare_models():
    """The pinned binary logistic fixture plus a genuinely different rival.

    ``GaussianNB`` is a closed-form MLE fit (no solver, no seed), so it is
    byte-stable within a pinned sklearn version, and its probability
    calibration differs enough from the logistic fixture's that the two
    models' F1 optima land on different thresholds AND different F1 values
    (measured: base 0.918 @ t=0.42, alt 0.886 @ t=0.53 at
    ``n_thresholds=20``). Both halves of the pooled-argmax bug are therefore
    observable: a pooled optimum could only report one of the two pairs.
    """
    from sklearn.naive_bayes import GaussianNB

    X, y, base = _binary_setup()
    alt = GaussianNB().fit(X.to_numpy(), y.to_numpy())
    return X, y, base, alt


def _optimum_texts(chart) -> list[str]:
    """Every ``max F1 = ... @ t=...`` annotation in a chart's rendered SVG."""
    import re

    return re.findall(r"max F1 = [0-9.]+ @ t=[0-9.]+", chart.to_svg())


def _solo_optimum(model, X, y) -> tuple[float, float]:
    """The (threshold, f1) F1-optimum of one model's own solo sweep."""
    sweep = ferrum.ModelSource(model, X, y).discrimination_threshold(n_thresholds=_N_THRESHOLDS)
    idx = sweep["f1"].arg_max()
    return float(sweep["threshold"][idx]), float(sweep["f1"][idx])


def test_compare_discrimination_threshold_renders_small_multiples():
    """One titled panel per model, not one pooled panel."""
    from ferrum.composition import ConcatChart
    from ferrum.title import Title

    X, y, base, alt = _threshold_compare_models()
    result = ferrum.discrimination_threshold_chart(
        base, X, y, n_thresholds=_N_THRESHOLDS, compare={"alt": alt}
    )

    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    for panel, name in zip(result.charts, ["base", "alt"]):
        assert isinstance(panel._title, Title)
        assert panel._title.text == name
    assert "<svg" in result.to_svg()


def test_compare_discrimination_threshold_panel_holds_one_sweep_each():
    """Each panel's frame carries exactly one model's sweep.

    The pooled render's defining symptom was ``2 * n_thresholds`` rows per
    metric in a single frame with no column distinguishing them, which the
    line mark drew as a zig-zag traversing x in [0, 1] twice.
    """
    X, y, base, alt = _threshold_compare_models()
    result = ferrum.discrimination_threshold_chart(
        base, X, y, n_thresholds=_N_THRESHOLDS, compare={"alt": alt}
    )

    for panel in result.charts:
        assert panel._data.height == _N_THRESHOLDS * _N_METRICS
        # One row per (threshold, metric) -- no stacked duplicate thresholds.
        assert panel._data.select(["threshold", "metric"]).n_unique() == panel._data.height


def test_compare_discrimination_threshold_panels_carry_their_own_optimum():
    """Each panel annotates its OWN model's F1 optimum, not a pooled one."""
    X, y, base, alt = _threshold_compare_models()
    result = ferrum.discrimination_threshold_chart(
        base, X, y, n_thresholds=_N_THRESHOLDS, compare={"alt": alt}
    )

    base_t, base_f1 = _solo_optimum(base, X, y)
    alt_t, alt_f1 = _solo_optimum(alt, X, y)
    # Precondition: the two models genuinely disagree, so a pooled optimum
    # could not satisfy both assertions below.
    assert (base_t, base_f1) != (alt_t, alt_f1)

    base_panel, alt_panel = result.charts
    assert _optimum_texts(base_panel) == [f"max F1 = {base_f1:.3f} @ t={base_t:.2f}"]
    assert _optimum_texts(alt_panel) == [f"max F1 = {alt_f1:.3f} @ t={alt_t:.2f}"]


def test_compare_discrimination_threshold_threshold_line_is_per_panel():
    """``threshold_line=True`` draws each panel's own optimum rule.

    ``_threshold_best`` rides the same per-model sweep as ``_optimum_x``, so
    the sentinel siblings cannot disagree about which model a panel describes.
    """
    X, y, base, alt = _threshold_compare_models()
    result = ferrum.discrimination_threshold_chart(
        base, X, y, n_thresholds=_N_THRESHOLDS, threshold_line=True, compare={"alt": alt}
    )

    base_t, _ = _solo_optimum(base, X, y)
    alt_t, _ = _solo_optimum(alt, X, y)
    base_panel, alt_panel = result.charts
    assert base_panel._data["_threshold_best"].drop_nulls().to_list() == [base_t]
    assert alt_panel._data["_threshold_best"].drop_nulls().to_list() == [alt_t]


def test_compare_discrimination_threshold_error_names_the_failing_model():
    """A per-model builder failure re-raises prefixed with that model's name.

    The multiclass fixture is a legal estimator that this binary-only
    diagnostic refuses; the base panel builds fine, so the prefix is the only
    thing identifying which compared model brought the request down.
    """
    X, y, base = _binary_setup()
    multiclass = load_fixture("multiclass_logistic")

    with pytest.raises(ValueError, match=r"compare=\{'alt': \.\.\.\}: .*binary-classifier only"):
        ferrum.discrimination_threshold_chart(
            base, X, y, n_thresholds=_N_THRESHOLDS, compare={"alt": multiclass}
        )


def test_discrimination_threshold_visualizer_compare_render_agrees_with_repr():
    """The visualizer's headline metrics and its rendered chart agree.

    ``_metrics``/``__repr__``/``.score()`` have described the base model since
    batch G T9 (``_base_model_rows``), but ``_build_chart`` forwarded the
    unfiltered compared source to the pooled builder, so the rendered
    annotation could name the rival while the repr named the base -- the
    "byte-identical between chart and visualizer render" half of #213. The
    base panel now annotates exactly the pair the repr prints.
    """
    from ferrum.composition import ConcatChart

    X, y, base, alt = _threshold_compare_models()
    viz = ferrum.DiscriminationThresholdVisualizer(
        base, n_thresholds=_N_THRESHOLDS, compare={"alt": alt}, random_state=0
    ).fit(X, y)

    chart = viz.show()
    assert isinstance(chart, ConcatChart)
    assert [panel._title.text for panel in chart.charts] == ["base", "alt"]

    best_t = viz._metrics["best_threshold"]
    best_f1 = viz._metrics["best_f1"]
    base_panel, alt_panel = chart.charts
    assert _optimum_texts(base_panel) == [f"max F1 = {best_f1:.3f} @ t={best_t:.2f}"]
    # And the rival's panel reports something else entirely -- proof the
    # agreement above is per-panel, not a single pooled string repeated.
    assert _optimum_texts(alt_panel) != _optimum_texts(base_panel)


def test_compare_discrimination_threshold_single_model_path_unchanged():
    """``compare=None`` still returns a plain single-panel Chart."""
    from ferrum.composition import ConcatChart

    X, y, base = _binary_setup()
    chart = ferrum.discrimination_threshold_chart(base, X, y, n_thresholds=_N_THRESHOLDS)

    assert isinstance(chart, ferrum.Chart)
    assert not isinstance(chart, ConcatChart)
    assert chart._data.height == _N_THRESHOLDS * _N_METRICS
