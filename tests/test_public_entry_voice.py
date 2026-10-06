"""One size rule and one entry voice, whichever door the caller came through.

Which marks draw a constant ``size=`` and which values are legal is one Rust
rule (``ferrum._core.check_mark_size``); every door that writes a constant
size into a mark's kwargs runs it at construction, voiced as the public call
the caller made. The doors are ``mark_*`` kwargs, ``fm.Layer(mark_kwargs=)``,
a figure function's ``mark=`` dict, a figure's per-layer kwargs and a
visualizer's ``mark=``; the wire door is pinned on the Rust side.

The voice is one stamp: every public function of a ``ferrum.plots`` module
and every ``FerrumVisualizer`` speaks as itself in warnings and refusals
alike. Both populations are derived from the package, never listed.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import pickle
import pkgutil
import warnings
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.plots
from ferrum._entry import _VOICE_SLOT, EntryRefusal, voiced_as
from ferrum.diagnostics.visualizers._base import FerrumVisualizer
from ferrum.marks.base import honored_kwargs, layer_style_fields
from ferrum.plots._helpers import _figure_entry

_RNG = np.random.default_rng(0)
_DF = pl.DataFrame(
    {
        "g": ["a", "b"] * 20,
        "h": ["x", "x", "y", "y"] * 10,
        "y": _RNG.normal(size=40),
        "x": _RNG.normal(size=40),
    }
)

_STAMP_CODE = _figure_entry(lambda: None).__code__


def _plots_functions() -> list[tuple[str, Callable]]:
    """Every public function defined in a ``ferrum.plots`` submodule."""
    found = []
    for info in pkgutil.iter_modules(ferrum.plots.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"ferrum.plots.{info.name}")
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            # `functools.wraps` keeps the wrapped function's module.
            if obj.__module__ == module.__name__:
                found.append((f"{info.name}.{name}", obj))
    return sorted(found)


def _visualizers() -> list[type]:
    out, todo = [], [FerrumVisualizer]
    while todo:
        cls = todo.pop()
        for sub in cls.__subclasses__():
            out.append(sub)
            todo.append(sub)
    return sorted(set(out), key=lambda c: c.__name__)


# ── The populations ─────────────────────────────────────────────────────────


def test_the_figure_population_covers_every_public_figure_function() -> None:
    names = {fn.__name__ for _, fn in _plots_functions()}
    exported = {
        n
        for n in fm.__all__
        if getattr(getattr(fm, n), "__module__", "").startswith("ferrum.plots.")
    }
    assert exported <= names and len(exported) >= 40
    # The deprecated dispatchers are public too, though out of `__all__`.
    assert {"shap_chart", "rank_chart"} <= names


@pytest.mark.parametrize(
    ("label", "fn"), [pytest.param(label, fn, id=label) for label, fn in _plots_functions()]
)
def test_every_public_figure_function_speaks_as_itself(label: str, fn: Callable) -> None:
    assert fn.__code__ is _STAMP_CODE, f"{label} is not wrapped by _figure_entry"


def test_the_population_check_catches_an_unstamped_function() -> None:
    """Control: a plain function fails the check the cells above pass."""

    def plain_chart() -> Any:
        return None

    assert plain_chart.__code__ is not _STAMP_CODE


def test_no_visualizer_replaces_the_fit_template() -> None:
    """``FerrumVisualizer.fit`` builds every visualizer's chart in the class's
    voice, so no subclass may define its own ``fit`` and skip it."""
    visualizers = _visualizers()
    assert len(visualizers) >= 25
    overriding = [cls.__name__ for cls in visualizers if "fit" in vars(cls)]
    assert overriding == []


def test_the_fit_guard_catches_a_class_that_defines_fit() -> None:
    """Control: the membership test the guard above applies sees a class's
    own ``fit``. A plain class, so the visualizer population is untouched."""
    own_fit = type("OwnFit", (), {"fit": lambda self, X, y=None: self})
    assert "fit" in vars(own_fit)
    assert "fit" not in vars(fm.ROCVisualizer)


# ── The size rule, door by door ─────────────────────────────────────────────

_SIZE_DOORS: dict[str, tuple[Callable[[], Any], str]] = {
    "mark_point(size=0)": (
        lambda: fm.Chart(_DF).mark_point(size=0),
        "mark_point(size=...) must be a positive number; received 0",
    ),
    "Layer(point, size=0)": (
        lambda: fm.Layer(mark="point", mark_kwargs={"size": 0}),
        "Layer(size=...) must be a positive number; received 0",
    ),
    "relplot(scatter, mark=size 0)": (
        lambda: fm.relplot(_DF, x="x", y="y", kind="scatter", mark={"size": 0}),
        "relplot(size=...) must be a positive number; received 0",
    ),
    "relplot(line, mark=size -5)": (
        lambda: fm.relplot(_DF, x="x", y="y", kind="line", mark={"size": -5}),
        "relplot(size=...) must be a positive number; received -5",
    ),
    "lmplot(line_kws=size 0)": (
        lambda: fm.lmplot(_DF, x="x", y="y", line_kws={"size": 0}),
        "lmplot(size=...) must be a positive number; received 0",
    ),
    "mark_area(size=5)": (
        lambda: fm.Chart(_DF).mark_area(size=5),
        "mark_area(size=...) has no effect on mark_area; received 5.",
    ),
    "Layer(area, size=5)": (
        lambda: fm.Layer(mark="area", mark_kwargs={"size": 5}),
        "Layer(size=...) has no effect on mark_area; received 5.",
    ),
    "Layer(text, size=5)": (
        lambda: fm.Layer(mark="text", mark_kwargs={"size": 5}),
        "Layer(size=...) has no effect on mark_text; received 5. Set font_size= instead.",
    ),
    "displot(hist, mark=size 5)": (
        lambda: fm.displot(_DF, x="y", kind="hist", mark={"size": 5}),
        "displot(size=...) has no effect on mark_bar; received 5.",
    ),
    "ClassBalanceVisualizer(mark=size 5)": (
        lambda: fm.ClassBalanceVisualizer(mark={"size": 5}).fit([0, 1, 1, 2, 2, 2]).show(),
        "ClassBalanceVisualizer(size=...) has no effect on mark_bar; received 5.",
    ),
}


@pytest.mark.parametrize("door", sorted(_SIZE_DOORS))
def test_a_constant_size_refuses_at_construction_through_every_door(door: str) -> None:
    build, message = _SIZE_DOORS[door]
    with pytest.raises(ValueError) as exc_info:
        build()
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    "build",
    [
        lambda: fm.Chart(_DF).mark_point(size=30).encode(x="x", y="y"),
        lambda: (
            fm.Chart(_DF)
            .encode(x="x", y="y")
            .layer(fm.Layer(mark="point", mark_kwargs={"size": 30}))
        ),
        lambda: fm.relplot(_DF, x="x", y="y", kind="scatter", mark={"size": 30}),
    ],
    ids=["mark_point", "Layer", "relplot"],
)
def test_a_positive_size_on_a_point_renders_through_every_door(build: Callable) -> None:
    """Control: the same doors accept a size the point draws, and draw it
    (area 30 is a ~3.09 px radius)."""
    svg = build().to_svg()
    assert 'r="3.09' in svg


def test_a_composite_spreads_its_size_only_onto_the_marks_that_draw_it() -> None:
    """``mark_smooth(size=)`` sizes its fit line, not its confidence band."""
    resolved = fm.Chart(_DF).mark_smooth(ci=0.95, size=40).encode(x="x", y="y")._resolve_pending()
    assert resolved._layers is not None
    sizes = {ly.mark: (ly.mark_kwargs or {}).get("size") for ly in resolved._layers}
    assert sizes["line"] == 40
    assert all(size is None for mark, size in sizes.items() if mark != "line")


# ── mark_function's style kwargs ────────────────────────────────────────────


def test_mark_function_style_kwargs_reach_its_line() -> None:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    chart = fm.Chart(df).encode(x="x").mark_function(lambda x: x, stroke_width=3, color="red")
    assert json.loads(chart.to_spec().to_json())["mark_style"] == {
        "stroke": "red",
        "stroke_width": 3.0,
    }
    assert 'stroke="#ff0000" stroke-width="3"' in chart.to_svg()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 0}, "mark_function(size=...) must be a positive number; received 0"),
        (
            {"arrow": True},
            "mark_function: keyword argument(s) ['arrow'] are drawn by no mark this call "
            "draws (['line']), so nothing would take them",
        ),
        (
            {"detail": "x"},
            "mark_function: keyword argument 'detail' is not a style kwarg; set it through "
            "encode(detail=...)",
        ),
    ],
)
def test_mark_function_refuses_a_bad_style_kwarg_in_its_own_voice(
    kwargs: dict, message: str
) -> None:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(df).encode(x="x").mark_function(lambda x: x, **kwargs)
    assert str(exc_info.value) == message


# ── One voice for warnings and refusals ─────────────────────────────────────


def test_a_figure_warns_and_refuses_in_the_same_voice() -> None:
    """catplot's violin warns as ``catplot``, and its layers refuse as
    ``catplot`` too: one stamp carries both."""
    chart = fm.catplot(_DF, x="g", y="y", hue="h", kind="violin")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        spec = json.loads(chart.to_spec().to_json())
    assert [str(w.message) for w in record] == [
        "catplot: inner marks are drawn at each category's center and are not "
        'dodged per hue group; use Chart.mark_violin(inner="none") to drop the inner mark'
    ]
    assert {layer.get("entry") for layer in spec["layers"]} == {"catplot"}


def test_a_flat_chart_entry_does_not_reach_another_operands_layers() -> None:
    """A flat figure chart's entry moves onto its own layer when it is
    layered, so a layer another operand brought falls back to its own mark."""
    figure = fm.relplot(_DF, x="x", y="y", kind="scatter")
    merged = figure + fm.Chart(_DF).mark_line().encode(x="x", y="y")
    spec = json.loads(merged.to_spec().to_json())
    assert "entry" not in spec
    assert [layer.get("entry") for layer in spec["layers"]] == ["relplot", None]
    hosted = figure.layer(fm.Layer(mark="line", encoding={"x": fm.X("x"), "y": fm.Y("y")}))
    spec = json.loads(hosted.to_spec().to_json())
    assert "entry" not in spec
    assert [layer.get("entry") for layer in spec["layers"]] == ["relplot", None]


# ── A visualizer's refusals speak as the class ──────────────────────────────


def _visualizer_cases() -> dict[str, Callable[[Any], Any]]:
    """For each visualizer class name, ``fit(mark)``: build it with *mark* and fit it.

    Fixture recipes only; the population they must cover is derived
    (:func:`_visualizers`) and checked against them below.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(1)
    X = pl.DataFrame({f"f{i}": rng.normal(size=60) for i in range(3)})
    xa = X.to_numpy()
    y_cls = (xa[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    y_reg = xa @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=60)
    clf = LogisticRegression().fit(xa, y_cls)
    reg = LinearRegression().fit(xa, y_reg)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(xa, y_reg)
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=0).fit(xa)
    pca = PCA().fit(xa)

    def classified(cls: type, **kw: Any) -> Callable[[Any], Any]:
        return lambda mark: cls(clf, mark=mark, **kw).fit(xa, y_cls)

    def regressed(cls: type, model: Any = reg, **kw: Any) -> Callable[[Any], Any]:
        return lambda mark: cls(model, mark=mark, **kw).fit(xa, y_reg)

    def clustered(cls: type, model: Any = kmeans, **kw: Any) -> Callable[[Any], Any]:
        return lambda mark: cls(model, mark=mark, **kw).fit(xa)

    return {
        "AlphaSelectionVisualizer": regressed(
            fm.AlphaSelectionVisualizer, Ridge(), alphas=[0.1, 1.0], cv=3
        ),
        "CVScoresVisualizer": regressed(fm.CVScoresVisualizer, cv=3),
        "CalibrationVisualizer": classified(fm.CalibrationVisualizer),
        "ClassBalanceVisualizer": lambda mark: fm.ClassBalanceVisualizer(mark=mark).fit(y_cls),
        "ClassPredictionErrorVisualizer": classified(fm.ClassPredictionErrorVisualizer),
        "ClassificationReportVisualizer": classified(fm.ClassificationReportVisualizer),
        "ConfusionMatrixVisualizer": classified(fm.ConfusionMatrixVisualizer),
        "CooksDistanceVisualizer": regressed(fm.CooksDistanceVisualizer),
        "DiscriminationThresholdVisualizer": classified(fm.DiscriminationThresholdVisualizer),
        "ElbowVisualizer": lambda mark: fm.ElbowVisualizer(KMeans, ks=[2, 3], mark=mark).fit(xa),
        "FeatureImportancesVisualizer": regressed(fm.FeatureImportancesVisualizer, tree),
        "InterclusterDistanceVisualizer": clustered(fm.InterclusterDistanceVisualizer),
        "LearningCurveVisualizer": regressed(fm.LearningCurveVisualizer, cv=3),
        "ManifoldVisualizer": clustered(fm.ManifoldVisualizer, None, method="pca"),
        "PCAVarianceVisualizer": clustered(fm.PCAVarianceVisualizer, pca),
        "PRVisualizer": classified(fm.PRVisualizer),
        "ParallelCoordinatesVisualizer": (
            lambda mark: fm.ParallelCoordinatesVisualizer(mark=mark).fit(X, y_cls)
        ),
        "PredictionErrorVisualizer": regressed(fm.PredictionErrorVisualizer),
        "ROCVisualizer": classified(fm.ROCVisualizer),
        "Rank1DVisualizer": lambda mark: fm.Rank1DVisualizer(mark=mark).fit(X),
        "Rank2DVisualizer": lambda mark: fm.Rank2DVisualizer(mark=mark).fit(X),
        "ResidualsVisualizer": regressed(fm.ResidualsVisualizer),
        "SHAPBarVisualizer": regressed(fm.SHAPBarVisualizer, tree, random_state=0),
        "SHAPBeeswarmVisualizer": regressed(fm.SHAPBeeswarmVisualizer, tree, random_state=0),
        "SHAPVisualizer": regressed(fm.SHAPVisualizer, tree, kind="bar", random_state=0),
        "SHAPWaterfallVisualizer": regressed(
            fm.SHAPWaterfallVisualizer, tree, sample_idx=0, random_state=0
        ),
        "SilhouetteVisualizer": clustered(fm.SilhouetteVisualizer),
        "ValidationCurveVisualizer": regressed(
            fm.ValidationCurveVisualizer, Ridge(), param="alpha", values=[0.1, 1.0], cv=3
        ),
    }


_VISUALIZER_CASES = _visualizer_cases()


def _quietly(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def test_every_visualizer_has_a_voice_case() -> None:
    assert sorted(_VISUALIZER_CASES) == [cls.__name__ for cls in _visualizers()]


@pytest.mark.parametrize("name", sorted(_VISUALIZER_CASES))
def test_a_visualizer_refuses_a_malformed_mark_as_its_class(name: str) -> None:
    fit = _VISUALIZER_CASES[name]
    with pytest.raises(ValueError) as exc_info:
        _quietly(lambda: fit(5))
    assert str(exc_info.value) == (
        f"{name}(mark=...) must be a dict, of mark kwargs or of sub-layer names to "
        "their mark kwargs; got int"
    )


@pytest.mark.parametrize("name", sorted(_VISUALIZER_CASES))
def test_a_visualizer_fits_with_no_mark_override(name: str) -> None:
    """Control: the same recipe fits, so the refusal above is the ``mark=``."""
    viz = _quietly(lambda: _VISUALIZER_CASES[name](None))
    assert viz.show() is not None


# A malformed ``mark=`` is the caller's input, judged before anything is
# computed from the model, so its refusal never depends on what that
# computation would raise first. Each test makes the SHAP step raise a
# sentinel; the control proves the step runs when ``mark=`` is well formed,
# so the mark refusal below wins only because it is judged first.
_SHAP_VISUALIZERS = sorted(n for n in _VISUALIZER_CASES if n.startswith("SHAP"))


class _ShapStepReached(Exception):
    """Raised in place of SHAP values, marking that the SHAP step ran."""


def _make_the_shap_step_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from ferrum.diagnostics.sources._importance import FeatureImportanceMixin

    def raise_instead(self: Any, **kwargs: Any) -> Any:
        raise _ShapStepReached

    monkeypatch.setattr(FeatureImportanceMixin, "shap_values", raise_instead)


@pytest.mark.parametrize("name", _SHAP_VISUALIZERS)
def test_a_shap_visualizer_refuses_a_malformed_mark_before_the_shap_step(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_the_shap_step_raise(monkeypatch)
    with pytest.raises(_ShapStepReached):
        _quietly(lambda: _VISUALIZER_CASES[name](None))
    with pytest.raises(ValueError) as exc_info:
        _quietly(lambda: _VISUALIZER_CASES[name](5))
    assert str(exc_info.value) == (
        f"{name}(mark=...) must be a dict, of mark kwargs or of sub-layer names to "
        "their mark kwargs; got int"
    )


_SHAP_FIGURES: dict[str, dict[str, Any]] = {
    "shap_bar_chart": {},
    "shap_beeswarm_chart": {},
    "shap_waterfall_chart": {"sample_idx": 0},
}


@pytest.mark.parametrize("name", sorted(_SHAP_FIGURES))
def test_a_shap_figure_refuses_a_malformed_mark_before_the_shap_step(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sklearn.tree import DecisionTreeRegressor

    xa = np.random.default_rng(2).normal(size=(40, 3))
    tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(xa, xa[:, 0])
    figure = getattr(fm, name)
    kwargs = {"random_state": 0, **_SHAP_FIGURES[name]}
    _make_the_shap_step_raise(monkeypatch)
    with pytest.raises(_ShapStepReached):
        _quietly(lambda: figure(tree, xa, **kwargs))
    with pytest.raises(ValueError) as exc_info:
        _quietly(lambda: figure(tree, xa, mark=5, **kwargs))
    assert str(exc_info.value) == (
        f"{name}(mark=...) must be a dict, of mark kwargs or of sub-layer names to "
        "their mark kwargs; got int"
    )


def test_the_override_door_called_directly_speaks_as_the_entry_it_is_given() -> None:
    """Outside a figure or visualizer the ``mark=`` door still names a call:
    the one its caller must pass, so the class name above comes from the
    ``fit`` template re-voicing it, not from an empty voice."""
    from ferrum._overrides import _apply_overrides

    with pytest.raises(ValueError) as exc_info:
        _apply_overrides(fm.Chart(_DF).mark_point(), entry="caller", mark=5)  # type: ignore[arg-type]
    assert str(exc_info.value) == (
        "caller(mark=...) must be a dict, of mark kwargs or of sub-layer names to "
        "their mark kwargs; got int"
    )


# ── jointplot's style dicts speak as jointplot, eager and lazy ──────────────


def _jointplot_cells() -> list[Any]:
    from ferrum.plots.matrix import _VALID_CENTER_KINDS, _VALID_MARGINAL_KINDS

    cells = [
        pytest.param({"kind": k, "joint_kws": {"bogus_kw": 1}}, id=f"joint_kws-{k}")
        for k in sorted(_VALID_CENTER_KINDS)
    ]
    cells += [
        pytest.param({"marginal_kind": k, "marginal_kws": {"bogus_kw": 1}}, id=f"marginal_kws-{k}")
        for k in sorted(_VALID_MARGINAL_KINDS)
    ]
    return cells


@pytest.mark.parametrize("kwargs", _jointplot_cells())
def test_a_jointplot_style_dict_refuses_as_jointplot(kwargs: dict) -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.jointplot(_DF, x="x", y="y", **kwargs).to_svg()
    assert str(exc_info.value).startswith("jointplot: unknown keyword argument 'bogus_kw'. ")


def test_a_lazily_resolved_jointplot_cell_refuses_as_jointplot_after_the_build() -> None:
    """The kde center resolves at render, after ``jointplot`` has returned."""
    chart = fm.jointplot(_DF, x="x", y="y", kind="kde", joint_kws={"bogus_kw": 1})
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert str(exc_info.value).startswith("jointplot: unknown keyword argument 'bogus_kw'. ")


@pytest.mark.parametrize(
    ("build", "voice"),
    [
        (lambda: fm.Chart(_DF).mark_point(bogus_kw=1), "mark_point"),
        (
            lambda: fm.Chart(_DF).mark_density(bogus_kw=1).encode(x="x").to_svg(),
            "mark_density",
        ),
    ],
    ids=["eager", "lazy"],
)
def test_a_mark_method_called_directly_speaks_as_itself(build: Callable, voice: str) -> None:
    """Control: with no figure around it, a mark method keeps its own voice."""
    with pytest.raises(ValueError) as exc_info:
        build()
    assert str(exc_info.value).startswith(f"{voice}: unknown keyword argument 'bogus_kw'. ")


# ── one speaker per door, built from a template, never empty ────────────────

_POINT_HONORED = sorted(layer_style_fields("point"))
_FONT_SIZE = "keyword argument 'font_size' is not honored by mark_point. Honored: "

# door -> (build, the exact refusal of a font_size on a point, the error type)
_VOICE_DOORS: dict[str, tuple[Callable[[], Any], str, type[Exception]]] = {
    "mark_point": (
        lambda: fm.Chart(_DF).mark_point(font_size=3),
        f"mark_point: {_FONT_SIZE}{sorted(honored_kwargs('point'))}",
        ValueError,
    ),
    "Layer": (
        lambda: fm.Layer(mark="point", mark_kwargs={"font_size": 3}),
        f"Layer: {_FONT_SIZE}{_POINT_HONORED}",
        ValueError,
    ),
    "Layer markless": (
        lambda: fm.Layer(mark_kwargs={"font_size": 3}),
        f"Layer: {_FONT_SIZE}{_POINT_HONORED}",
        ValueError,
    ),
    "Chart.override": (
        lambda: (
            fm.Chart(_DF).mark_point().encode(x="x", y="y").override(mark_font_size=3).to_spec()
        ),
        f"Chart.override: {_FONT_SIZE}{_POINT_HONORED}",
        fm.FerrumOverrideError,
    ),
    "relplot mark=": (
        lambda: fm.relplot(_DF, x="x", y="y", mark={"font_size": 3}),
        f"relplot: {_FONT_SIZE}{_POINT_HONORED}",
        ValueError,
    ),
    "lmplot scatter_kws": (
        lambda: fm.lmplot(_DF, x="x", y="y", scatter_kws={"font_size": 3}),
        f"lmplot: {_FONT_SIZE}{_POINT_HONORED}",
        ValueError,
    ),
    "regplot scatter_kws": (
        lambda: fm.regplot(_DF, x="x", y="y", scatter_kws={"font_size": 3}),
        f"regplot: {_FONT_SIZE}{_POINT_HONORED}",
        ValueError,
    ),
}


@pytest.mark.parametrize("door", sorted(_VOICE_DOORS))
def test_each_door_refuses_in_the_voice_of_the_call_the_caller_made(door: str) -> None:
    build, message, error = _VOICE_DOORS[door]
    with pytest.raises(error) as exc_info:
        build()
    assert str(exc_info.value) == message
    assert isinstance(exc_info.value, EntryRefusal)


def test_a_door_that_honors_the_field_accepts_it() -> None:
    """Control for the matrix above: the same doors take a field the point draws."""
    assert fm.Layer(mark="point", mark_kwargs={"opacity": 0.5}).mark_kwargs == {"opacity": 0.5}
    spec = fm.Chart(_DF).mark_point().encode(x="x", y="y").override(mark_opacity=0.5).to_spec()
    assert json.loads(spec.to_json())["mark_style"]["opacity"] == 0.5


def test_an_override_refusal_is_re_voiced_by_a_call_that_wrote_it() -> None:
    """``FerrumOverrideError`` is an ``EntryRefusal``: a figure that renders an
    override on the caller's behalf re-voices it, and its type survives."""
    chart = fm.Chart(_DF).mark_point().encode(x="x", y="y").override(mark_font_size=3)
    with pytest.raises(fm.FerrumOverrideError) as exc_info:
        with voiced_as("roc_chart"):
            chart.to_spec()
    assert str(exc_info.value) == f"roc_chart: {_FONT_SIZE}{_POINT_HONORED}"


_POLAR_POINT = fm.Chart(_DF).mark_point().encode(x="x", y="y").coord(fm.CoordPolar(theta="x"))


@pytest.mark.parametrize(
    ("chart", "message"),
    [
        (
            fm.Chart(_DF).mark_point().override(x_axis_lable_angle=-45),
            "Chart.override: Unknown override path 'x_axis_lable_angle'. "
            "Did you mean: 'x_axis_label_angle'?",
        ),
        (
            _POLAR_POINT.override(coord_ratio=1),
            "Chart.override: coord_* override (ratio) is not a field of CoordPolar. Set a "
            "matching coord via .coord(...) first, or override a field that coord exposes.",
        ),
    ],
    ids=["unknown-path", "coord-leaf-of-another-coord"],
)
def test_every_override_refusal_names_chart_override(chart: Any, message: str) -> None:
    with pytest.raises(fm.FerrumOverrideError) as exc_info:
        chart.to_spec()
    assert str(exc_info.value) == message


def test_a_layered_override_refusal_names_chart_override() -> None:
    chart = (fm.Chart(_DF).mark_point() + fm.Chart(_DF).mark_line()).encode(x="x", y="y")
    with pytest.raises(fm.FerrumOverrideError) as exc_info:
        chart.override(mark_opacity=0.5).to_spec()
    assert str(exc_info.value) == (
        "Chart.override: mark_* override (opacity) is ambiguous on a multi-mark chart: a "
        "layered chart has no single primary mark to apply it to. Set the style on the "
        "specific layer's mark_*() call instead."
    )


def test_an_entry_refusal_refuses_an_empty_voice() -> None:
    with pytest.raises(ValueError) as exc_info:
        EntryRefusal(lambda entry: f"{entry}(mark=...) must be a dict", "")
    assert not isinstance(exc_info.value, EntryRefusal)
    assert str(exc_info.value) == (
        "EntryRefusal needs the public call it speaks as; got an empty entry"
    )


@pytest.mark.parametrize(
    "door",
    [
        "ferrum._overrides._apply_overrides",
        "ferrum._overrides._apply_mark_overrides",
        "ferrum._overrides.refuse_malformed_mark",
        "ferrum.plots._helpers._style_kws",
        "ferrum.plots._helpers._finalize_chart",
    ],
)
def test_every_style_door_requires_the_entry_it_speaks_as(door: str) -> None:
    module, name = door.rsplit(".", 1)
    parameter = inspect.signature(getattr(importlib.import_module(module), name)).parameters[
        "entry"
    ]
    assert parameter.default is inspect.Parameter.empty


def test_re_voicing_a_refusal_formats_its_template_without_re_running_the_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal carries its text for any voice; asking for another voice
    re-calls only the one pure validator that refused, never the resolver."""
    import ferrum.marks.base as base

    runs: list[str] = []
    real = base._resolve_style

    def counting(entry: str, *args: Any) -> Any:
        runs.append(entry)
        return real(entry, *args)

    monkeypatch.setattr(base, "_resolve_style", counting)
    for style, voiced in [
        ({"font_size": 3}, f"relplot: {_FONT_SIZE}{_POINT_HONORED}"),
        (
            {"opacity": "x"},
            "relplot(opacity=...) must be a real number (int, float, numpy scalar, or "
            "Decimal), not str; received 'x'",
        ),
    ]:
        with pytest.raises(EntryRefusal) as exc_info:
            base.resolve_layer_style("Layer", "point", style)
        runs.clear()
        assert str(exc_info.value.spoken_as("relplot")) == voiced
        assert runs == []


def test_a_stray_value_error_is_not_relabelled_as_a_style_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ferrum.marks.base as base

    def broken(mark_name: str, key: str) -> str:
        raise ValueError("an internal defect")

    monkeypatch.setattr(base, "_canonical_mark_kwarg", broken)
    with pytest.raises(ValueError) as exc_info:
        base.resolve_layer_style("Layer", "point", {"opacity": 0.5})
    assert not isinstance(exc_info.value, EntryRefusal)
    assert str(exc_info.value) == "an internal defect"


# ── a refusal crosses a process boundary intact ─────────────────────────────


def _refusal_from(build: Callable[[], Any]) -> EntryRefusal:
    with pytest.raises(EntryRefusal) as exc_info:
        build()
    return exc_info.value


def _layer_style_refusal(style: dict[str, Any]) -> EntryRefusal:
    from ferrum.marks.base import resolve_layer_style

    return _refusal_from(lambda: resolve_layer_style("Layer", "point", style))


_REFUSALS: dict[str, Callable[[], EntryRefusal]] = {
    # A template refusal: a style key the mark does not draw.
    "template": lambda: _layer_style_refusal({"font_size": 3}),
    # A validator refusal, whose text for another voice re-calls the validator.
    "validator": lambda: _layer_style_refusal({"opacity": "x"}),
    # A refusal under compare=, its model prefix wrapped around the template.
    "compare-prefixed": lambda: _layer_style_refusal({"font_size": 3}).prefixed(
        "compare={'m2': ...}: "
    ),
    "FerrumOverrideError style": lambda: _refusal_from(
        lambda: fm.Chart(_DF).mark_point().encode(x="x", y="y").override(mark_font_size=3).to_spec()
    ),
    "FerrumOverrideError path": lambda: _refusal_from(
        lambda: fm.Chart(_DF).mark_point().override(x_axis_lable_angle=-45).to_spec()
    ),
    # A color refusal echoes the caller's own string unescaped; when that
    # string is the base voice-slot character, the frozen pickle template
    # must not mistake it for the slot it substitutes on.
    "color echoes the voice slot": lambda: _refusal_from(
        lambda: fm.Chart(_DF).mark_point(color=_VOICE_SLOT)
    ),
}

_COPIES: dict[str, Callable[[EntryRefusal], EntryRefusal]] = {
    "pickle": lambda err: pickle.loads(pickle.dumps(err)),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


@pytest.mark.parametrize("how", sorted(_COPIES))
@pytest.mark.parametrize("refusal", sorted(_REFUSALS))
def test_a_refusal_survives_pickle_and_copy_and_still_re_voices(refusal: str, how: str) -> None:
    """A render in a process pool sends its refusal back pickled: the copy
    keeps its type, text and entry, and re-voices as the original does."""
    original = _REFUSALS[refusal]()
    clone = _COPIES[how](original)
    assert type(clone) is type(original)
    assert str(clone) == str(original)
    assert clone.args == original.args
    assert clone.entry == original.entry
    revoiced = str(original.spoken_as("roc_chart"))
    assert revoiced != str(original)
    assert "roc_chart" in revoiced
    with pytest.raises(type(original)) as exc_info:
        with voiced_as("roc_chart"):
            raise clone
    assert str(exc_info.value) == revoiced


def test_the_public_override_error_pickles_as_itself() -> None:
    """The public type, raised by a real render, round-trips by its own name."""
    err = _REFUSALS["FerrumOverrideError path"]()
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is fm.FerrumOverrideError
    assert str(clone) == (
        "Chart.override: Unknown override path 'x_axis_lable_angle'. "
        "Did you mean: 'x_axis_label_angle'?"
    )
    assert clone.entry == "Chart.override"


# ── compare= names the model on every refusal ───────────────────────────────


def _compare_source(names: list[str]) -> Any:
    class _Source:
        def items(self) -> list[tuple[str, str]]:
            return [(name, name) for name in names]

    return _Source()


def _refuse_for(model: str) -> Callable[..., Any]:
    """A builder that refuses only for *model*: a refusal that depends on which
    model the panel draws (an undrawn sub-layer, a style no layer draws)."""

    def build(source: str, **_: Any) -> Any:
        if source == model:
            from ferrum._overrides import _undrawn_sub_layer_message

            raise EntryRefusal(_undrawn_sub_layer_message("auc_label", {"line"}), "roc_chart")
        return fm.Chart(_DF).mark_point().encode(x="x", y="y")

    return build


def test_a_compare_refusal_names_the_model_whose_panel_refused() -> None:
    from ferrum.plots._helpers import _compose_compare

    with pytest.raises(EntryRefusal) as exc_info:
        with voiced_as("pdp_chart"):
            _compose_compare(
                _compare_source(["base", "m2"]),
                _refuse_for("m2"),
                builder_kwargs={},
                resolve={"x": "shared", "y": "shared"},
            )
    assert str(exc_info.value) == (
        "compare={'m2': ...}: pdp_chart(mark={'auc_label': ...}): this call draws no "
        "'auc_label' sub-layer, so nothing would take the style; it draws ['line']"
    )


def _residual_models() -> tuple[Any, Any, Any, Any]:
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y = X @ np.array([1.0, -2.0]) + rng.normal(size=40)
    linear = LinearRegression().fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    return linear, tree, X, y


_LEVERAGE_PANELS = ["qq", "residuals_vs_leverage"]


def test_a_real_figure_whose_second_model_alone_refuses_names_that_model() -> None:
    """A tree has no leverage, so its residuals grid drops the leverage panel
    and keeps only the QQ panel, which draws no point to take ``shape``. The
    linear model's grid draws one, so only the tree's panel refuses."""
    linear, tree, X, y = _residual_models()
    with pytest.raises(EntryRefusal) as exc_info:
        fm.residuals_chart(
            linear, X, y, compare={"tree": tree}, panels=_LEVERAGE_PANELS, mark={"shape": "square"}
        )
    assert str(exc_info.value) == (
        "compare={'tree': ...}: residuals_chart(mark=...): unknown sub-layer 'shape'; "
        "this chart's sub-layers are ['point', 'reference']"
    )


def test_the_same_style_on_two_models_that_both_draw_it_renders() -> None:
    """Control: the refusal above depends on the model, not on the style."""
    linear, _, X, y = _residual_models()
    chart = fm.residuals_chart(
        linear, X, y, compare={"linear2": linear}, panels=_LEVERAGE_PANELS, mark={"shape": "square"}
    )
    assert chart.to_svg()


def test_a_compare_refusal_that_fires_for_every_model_names_the_first() -> None:
    """Control: a refusal of the caller's input alone fires on the first
    model, and names it too, by the same rule."""
    from sklearn.cluster import KMeans

    X = _RNG.normal(size=(40, 2))
    models = [KMeans(n_clusters=k, n_init=1, random_state=0).fit(X) for k in (2, 3)]
    with pytest.raises(EntryRefusal) as exc_info:
        fm.silhouette_chart(models[0], X, compare={"m2": models[1]}, mark={"bogus_kw": 1})
    assert str(exc_info.value) == (
        "compare={'base': ...}: silhouette_chart(mark=...): unknown sub-layer 'bogus_kw'; "
        "this chart's sub-layers are ['rect', 'reference']"
    )
