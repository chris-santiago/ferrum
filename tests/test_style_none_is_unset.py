"""A style value of ``None`` is unset at every door that merges style over a style already set.

Each door is fed ``{key: None}`` where ``key`` already holds a value, through
the public call that reaches it, and must leave that value exactly as it was
and render byte-identical to the chart without the ``None``. The vacuity
control on every door is a non-``None`` value, which must replace it. A key
no layer takes is refused whatever its value, so ``None`` does not let a
typo, an undrawn field or an internal field through.
"""

from __future__ import annotations

import json
from typing import Any, Callable, NamedTuple

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._entry import EntryRefusal
from ferrum._overrides import _apply_overrides
from ferrum.exceptions import FerrumOverrideError

_RNG = np.random.default_rng(0)
_DF = pl.DataFrame(
    {
        "x": _RNG.normal(size=30),
        "y": _RNG.normal(size=30),
        "b": [0, 1] * 15,
        "g": ["a", "b", "c"] * 10,
    }
)


def _spec(chart: Any) -> dict:
    return json.loads(chart.to_spec().to_json())


def _flat(chart: Any, key: str) -> Any:
    return (_spec(chart).get("mark_style") or {}).get(key)


def _layer(chart: Any, mark: str, key: str) -> Any:
    (style,) = [ly.get("mark_style") or {} for ly in _spec(chart)["layers"] if ly["mark"] == mark]
    return style.get(key)


def _layer_by_entry(chart: Any, entry: str, key: str) -> Any:
    """Like ``_layer``, but keyed on ``entry`` rather than ``mark``.

    A prior primitive mark kept alongside a fitted-layer mark
    (``mark_line().mark_function(...)``) can share its mark name with the
    fitted layer -- both are ``"line"`` -- so distinguishing them needs the
    entry each layer names: the fitted layer's own call (``"mark_function"``),
    never the prior layer's (``None``, since a primitive mark call sets no
    entry of its own).
    """
    (style,) = [
        ly.get("mark_style") or {} for ly in _spec(chart)["layers"] if ly.get("entry") == entry
    ]
    return style.get(key)


def _base() -> Any:
    return fm.Chart(_DF).encode(x="x", y="y")


class _Door(NamedTuple):
    #: The chart with *value* passed at the door; ``None`` is the case under test.
    build: Callable[[Any], Any]
    #: The same chart with nothing passed at the door.
    unset: Callable[[], Any]
    #: The value the door merges over, read off the built chart's spec.
    read: Callable[[Any], Any]
    #: The value already set before the door merges.
    kept: Any
    #: A non-``None`` value the door must apply (the vacuity control).
    applied: Any


_DOORS: dict[str, _Door] = {
    "Chart.override(mark_<leaf>=)": _Door(
        lambda v: _base().mark_point(opacity=0.3).override(mark_opacity=v),
        lambda: _base().mark_point(opacity=0.3),
        lambda c: _flat(c, "opacity"),
        0.3,
        0.8,
    ),
    "a later Chart.override(mark_<leaf>=)": _Door(
        lambda v: _base().mark_point().override(mark_opacity=0.3).override(mark_opacity=v),
        lambda: _base().mark_point().override(mark_opacity=0.3),
        lambda c: _flat(c, "opacity"),
        0.3,
        0.8,
    ),
    "layer_kwargs= on a layered composite": _Door(
        lambda v: _base().mark_smooth(ci=0.95, layer_kwargs={"ribbon": {"opacity": v}}),
        lambda: _base().mark_smooth(ci=0.95),
        lambda c: _layer(c, "ribbon", "opacity"),
        0.3,
        0.8,
    ),
    "layer_kwargs= beside the composite's catch-all": _Door(
        lambda v: _base().mark_smooth(
            ci=0.95, opacity=0.5, layer_kwargs={"ribbon": {"opacity": v}}
        ),
        lambda: _base().mark_smooth(ci=0.95, opacity=0.5),
        lambda c: _layer(c, "ribbon", "opacity"),
        0.5,
        0.8,
    ),
    "layer_kwargs= on a single-mark composite": _Door(
        lambda v: _base().mark_smooth(stroke_width=3, layer_kwargs={"line": {"stroke_width": v}}),
        lambda: _base().mark_smooth(stroke_width=3),
        lambda c: _flat(c, "stroke_width"),
        3.0,
        6.0,
    ),
    "a figure's mark= on a named sub-layer": _Door(
        lambda v: fm.lmplot(_DF, x="x", y="y", mark={"ribbon": {"opacity": v}}),
        lambda: fm.lmplot(_DF, x="x", y="y"),
        lambda c: _layer(c, "ribbon", "opacity"),
        0.3,
        0.8,
    ),
    # The fit line carries no library-set style, so `None` here must leave
    # the stroke unset, as the unset chart draws it.
    "a figure's line_kws=": _Door(
        lambda v: fm.lmplot(_DF, x="x", y="y", line_kws={"stroke": v}),
        lambda: fm.lmplot(_DF, x="x", y="y"),
        lambda c: _layer(c, "line", "stroke"),
        None,
        "#ff0000",
    ),
    "mark_function(**style) on its own layer": _Door(
        lambda v: _base().mark_line(stroke_width=7).mark_function(np.sin, stroke_width=v),
        lambda: _base().mark_line(stroke_width=7).mark_function(np.sin),
        lambda c: _layer_by_entry(c, "mark_function", "stroke_width"),
        None,
        2.0,
    ),
}


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_none_leaves_the_value_already_set(door: str) -> None:
    d = _DOORS[door]
    chart = d.build(None)
    assert d.read(chart) == d.kept
    assert chart.to_svg() == d.unset().to_svg()


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_value_still_replaces_the_value_already_set(door: str) -> None:
    d = _DOORS[door]
    chart = d.build(d.applied)
    assert d.read(chart) == d.applied
    assert chart.to_svg() != d.unset().to_svg()


def test_an_aimed_none_does_not_record_the_key_as_the_callers() -> None:
    """A `None` aimed at one layer is unset, so it does not claim the key from
    the composite's catch-all: the ribbon takes the catch-all `0.5`, not the
    desugar's `0.3` and not unset."""
    chart = _base().mark_smooth(ci=0.95, opacity=0.5, layer_kwargs={"ribbon": {"opacity": None}})
    assert _layer(chart, "ribbon", "opacity") == 0.5
    assert _layer(chart, "line", "opacity") == 0.5


# A figure's `mark=` dict on a chart with no named sub-layers. No figure draws
# such a chart with a style already set, so the door is driven directly, as
# every figure drives it.


def _spread(chart: Any, style: dict) -> list[dict | None]:
    styled = _apply_overrides(chart, entry="relplot", mark=style)
    if styled._layers:
        return [ly.mark_kwargs for ly in styled._layers]
    return [styled._mark_kwargs]


def _point_and_line() -> Any:
    return (fm.Chart(_DF).mark_point(opacity=0.3) + fm.Chart(_DF).mark_line(opacity=0.6)).encode(
        x="x", y="y"
    )


def test_a_mark_dict_none_leaves_each_layers_own_value() -> None:
    assert _spread(_point_and_line(), {"opacity": None}) == [{"opacity": 0.3}, {"opacity": 0.6}]
    assert _spread(_point_and_line(), {"opacity": 0.8}) == [{"opacity": 0.8}, {"opacity": 0.8}]


def test_a_mark_dict_none_leaves_a_flat_marks_own_value() -> None:
    chart = _base().mark_point(opacity=0.3)
    assert _spread(chart, {"opacity": None}) == [{"opacity": 0.3}]
    assert _spread(chart, {"opacity": 0.8}) == [{"opacity": 0.8}]


def test_a_flat_figure_mark_dict_none_sets_nothing() -> None:
    chart = fm.relplot(_DF, x="x", y="y", mark={"opacity": None})
    assert chart._mark_kwargs == {}
    assert chart.to_svg() == fm.relplot(_DF, x="x", y="y").to_svg()


# ── a single-mark composite desugar merges onto the chart's own style ──────


def test_a_single_mark_composites_own_style_merges_onto_the_charts_own_mark_kwargs() -> None:
    """``_desugar.py``'s single-mark dispatch path (``mark_histogram``'s
    desugar returns one unlayered mark, not ``.layers``) merges the
    composite's own resolved style *onto* the chart's own ``_mark_kwargs``
    via ``merge_style(new._mark_kwargs, merged_mark_kwargs)`` -- not the
    other way around. Calling ``merge_style(merged_mark_kwargs,
    new._mark_kwargs)``, or replacing the chart's kwargs with
    ``merged_mark_kwargs`` outright, would silently drop whichever side
    the chart already carried before the composite call. A primitive
    mark's own style kwarg (``stroke_width=7``, set before the composite
    runs) and the composite's own catch-all (``fill="red"``) must both
    survive."""
    chart = (
        fm.Chart(_DF).mark_bar(stroke_width=7).mark_histogram(bin_count=5, fill="red").encode(x="x")
    )
    assert _flat(chart, "stroke_width") == 7
    assert _flat(chart, "fill") == "red"


# ── a key no layer takes is refused whatever its value ──────────────────────

#: door -> build(key, value). Each is fed a typo, a field its mark never
#: draws, and an internal field; the refusal must not depend on the value.
_REFUSING: dict[str, Callable[[str, Any], Any]] = {
    "Chart.override": lambda k, v: _base().mark_line().override(**{f"mark_{k}": v}).to_spec(),
    "a later Chart.override": lambda k, v: (
        _base().mark_line().override(mark_opacity=0.3).override(**{f"mark_{k}": v}).to_spec()
    ),
    "layer_kwargs=": lambda k, v: (
        _base().mark_smooth(ci=0.95, layer_kwargs={"line": {k: v}}).to_spec()
    ),
    "a figure's mark= sub-layer": lambda k, v: fm.lmplot(_DF, x="x", y="y", mark={"line": {k: v}}),
    "a figure's flat mark=": lambda k, v: fm.relplot(_DF, x="x", y="y", kind="line", mark={k: v}),
    "a figure's line_kws=": lambda k, v: fm.lmplot(_DF, x="x", y="y", line_kws={k: v}),
}

_REFUSED_KEYS = {"typo": "bogus", "undrawn": "shape", "internal": "detail"}


@pytest.mark.parametrize("kind", sorted(_REFUSED_KEYS))
@pytest.mark.parametrize("door", sorted(_REFUSING))
def test_a_key_no_layer_takes_refuses_whatever_its_value(door: str, kind: str) -> None:
    key = _REFUSED_KEYS[kind]
    build = _REFUSING[door]
    with pytest.raises((EntryRefusal, FerrumOverrideError)) as with_value:
        build(key, "x")
    with pytest.raises((EntryRefusal, FerrumOverrideError)) as with_none:
        build(key, None)
    assert str(with_none.value) == str(with_value.value)
    assert key in str(with_none.value)
