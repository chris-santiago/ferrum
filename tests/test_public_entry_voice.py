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


def test_every_visualizer_speaks_through_the_one_show() -> None:
    """``show()`` is the only public exit of a visualizer's chart and it
    stamps the class name, so no subclass may replace it."""
    visualizers = _visualizers()
    assert len(visualizers) >= 25
    overriding = [cls.__name__ for cls in visualizers if "show" in vars(cls)]
    assert overriding == []


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
            "mark_function: keyword argument 'arrow' is not honored by mark_function.",
        ),
    ],
)
def test_mark_function_refuses_a_bad_style_kwarg_in_its_own_voice(
    kwargs: dict, message: str
) -> None:
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(df).encode(x="x").mark_function(lambda x: x, **kwargs)
    assert str(exc_info.value).startswith(message)


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
