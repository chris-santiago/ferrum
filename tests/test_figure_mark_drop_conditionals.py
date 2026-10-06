"""A figure's ``mark={name: False}`` drop takes the dropped layers' own conditionals with it.

A conditional declared on one layer restyles that layer only. When a figure's
``mark=`` drops every layer that owns a conditional, the conditional goes too:
left behind with no owner, it would read as declared on the chart as a whole
and restyle every layer that remains. A conditional some kept layer also owns,
or one declared on the chart as a whole, is untouched by the drop.
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm
from ferrum._overrides import _apply_mark_overrides

_DF = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.0, 1.0]})
_SEL = fm.selection_point(name="s", clear="false")
_XY = {"x": fm.X("x"), "y": fm.Y("y")}


def _cond() -> fm.ConditionalSpec:
    return fm.when(_SEL).then(100).otherwise(20)


def _layer(name: str, mark: str, **encoding: object) -> fm.Layer:
    return fm.Layer(name=name, mark=mark, encoding={**_XY, **encoding})


def _host() -> fm.Chart:
    return fm.Chart(_DF).encode(x="x", y="y")


def _drop(chart: fm.Chart, *names: str) -> fm.Chart:
    """Drop *names* through the figure ``mark=`` door every figure function closes with."""
    return _apply_mark_overrides(chart, {n: False for n in names}, entry="relplot")


def _scopes(chart: fm.Chart) -> dict[str, list[str]]:
    """Each wire conditional's channel mapped to the names of the layers it reaches."""
    wire = json.loads(chart.add_selection(_SEL).to_spec().to_json())
    names = [ly.get("name") for ly in wire["layers"]]
    return {c["channel"]: [names[i] for i in c["layers"]] for c in wire.get("conditionals", [])}


def _line_is_restyled(chart: fm.Chart) -> list[bool]:
    scene = json.loads(chart.add_selection(_SEL).interactive().scene_json)
    return [
        "vertex_rows" in batch
        for panel in scene["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "line"
    ]


def _point_owns_size() -> fm.Chart:
    return _host().layer(_layer("pts", "point", size=_cond()), _layer("trend", "line"))


def test_vacuity_control_the_owners_conditional_reaches_only_its_owner() -> None:
    chart = _point_owns_size()
    assert _scopes(chart) == {"size": ["pts"]}
    assert _scopes(_drop(chart, "trend")) == {"size": ["pts"]}


def test_dropping_a_conditionals_only_owner_drops_the_conditional() -> None:
    dropped = _drop(_point_owns_size(), "pts")
    assert _scopes(dropped) == {}
    assert _line_is_restyled(dropped) == [False]


def test_a_conditional_a_kept_layer_also_owns_stays_on_the_kept_layer() -> None:
    operand = _host().layer(_layer("p1", "point"), _layer("p2", "point")).encode(size=_cond())
    chart = operand + fm.Chart(_DF).mark_line().encode(x="x", y="y")
    assert _scopes(chart) == {"size": [None, "p1", "p2"]}
    assert _scopes(_drop(chart, "p1")) == {"size": [None, "p2"]}


def test_a_chart_wide_conditional_survives_the_drop_and_reaches_every_kept_layer() -> None:
    chart = _host().layer(_layer("pts", "point"), _layer("trend", "line")).encode(size=_cond())
    dropped = _drop(chart, "pts")
    assert _scopes(dropped) == {"size": [None, "trend"]}
    assert _line_is_restyled(dropped) == [True]
