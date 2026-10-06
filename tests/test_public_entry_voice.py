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

import importlib
import inspect
import json
import pkgutil
import warnings
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.plots
from ferrum.diagnostics.visualizers._base import FerrumVisualizer
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
        "mark_point(size=...) must be a positive number; received 0",
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
        "mark_area(size=...) has no effect on mark_area; received 5.",
    ),
    "Layer(text, size=5)": (
        lambda: fm.Layer(mark="text", mark_kwargs={"size": 5}),
        "mark_text(size=...) has no effect on mark_text; received 5. Set font_size= instead.",
    ),
    "displot(hist, mark=size 5)": (
        lambda: fm.displot(_DF, x="y", kind="hist", mark={"size": 5}),
        "displot(size=...) has no effect on displot; received 5.",
    ),
    "ClassBalanceVisualizer(mark=size 5)": (
        lambda: fm.ClassBalanceVisualizer(mark={"size": 5}).fit([0, 1, 1, 2, 2, 2]).show(),
        "ClassBalanceVisualizer(size=...) has no effect on ClassBalanceVisualizer; received 5.",
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


def test_the_override_door_alone_has_no_voice() -> None:
    """Control: outside a figure or visualizer the ``mark=`` door names no
    one, so the class name above comes from the ``fit`` template."""
    from ferrum._overrides import _apply_overrides

    with pytest.raises(ValueError) as exc_info:
        _apply_overrides(fm.Chart(_DF).mark_point(), mark=5)  # type: ignore[arg-type]
    assert str(exc_info.value).startswith("(mark=...) must be a dict")


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
