"""`Layer(transforms=)` is refused, in Python and on the wire.

A layer never ran its own ``transforms``: the renderer reads a layer's input
batch from a *named* chart-level transform through ``data_source``, and the
per-layer list was accepted, serialized, deserialized and never read. A
non-empty value is now refused at construction with the text below, and a
hand-built wire spec whose ``layers[i].transforms`` is non-empty is refused at
render with a typed error naming layer ``i``. The route that does work (a
named chart-level transform read through ``data_source``, which ``+`` builds
for a composed chart's own ``.transform(...)``) keeps rendering.
"""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum import ChartSpec
from ferrum._core import render_interactive, render_svg

DF = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
_XY = {"x": fm.X("x"), "y": fm.Y("y")}

_LAYER_TRANSFORMS_REFUSAL = (
    "Layer(transforms=[...]): a layer does not run its own transforms. Give the layer "
    "its own chart with .transform(...) and compose it with +, which routes the "
    "transform to that layer alone."
)


def _filter():
    return fm.transform_filter("datum.x > 2")


# ---------------------------------------------------------------------------
# Python door: a non-empty transforms= refuses at construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transforms",
    [
        pytest.param([_filter()], id="filter-dict"),
        pytest.param([fm.Bin("x")], id="pyo3-bin"),
        pytest.param([_filter(), fm.Bin("x")], id="two"),
        pytest.param((_filter(),), id="tuple"),
        pytest.param(0, id="falsy-int"),
        pytest.param(False, id="falsy-bool"),
        pytest.param("", id="falsy-str"),
        pytest.param({}, id="falsy-dict"),
        pytest.param(pl.Series([1]), id="polars-series"),
        pytest.param(pl.DataFrame({"a": [1]}), id="polars-frame"),
        pytest.param(np.array([1, 2]), id="numpy-array"),
    ],
)
def test_non_empty_layer_transforms_is_refused(transforms):
    with pytest.raises(ValueError) as exc:
        fm.Layer(mark="line", encoding=dict(_XY), transforms=transforms)
    assert str(exc.value) == _LAYER_TRANSFORMS_REFUSAL


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"transforms": None}, id="none"),
        pytest.param({"transforms": []}, id="empty-list"),
        pytest.param({"transforms": ()}, id="empty-tuple"),
    ],
)
def test_empty_or_omitted_layer_transforms_is_accepted_and_wire_identical(kwargs):
    base = fm.Chart(DF).mark_point().encode(x="x", y="y")
    omitted = base.layer(fm.Layer(mark="line", encoding=dict(_XY)))
    chart = base.layer(fm.Layer(mark="line", encoding=dict(_XY), **kwargs))

    assert chart.to_json() == omitted.to_json()
    layers = json.loads(chart.to_json())["layers"]
    assert layers[1] == {"mark": "line", "encoding": {"x": {"field": "x"}, "y": {"field": "y"}}}


# ---------------------------------------------------------------------------
# Wire door: a hand-built spec carrying layers[i].transforms refuses at render.
# ---------------------------------------------------------------------------


def _layered_wire(*, transforms_on: int | None) -> ChartSpec:
    chart = fm.Chart(DF).mark_point().encode(x="x", y="y") + fm.Chart(DF).mark_line().encode(
        x="x", y="y"
    )
    wire = json.loads(chart.to_json())
    if transforms_on is not None:
        wire["layers"][transforms_on]["transforms"] = [
            {"type": "filter", "predicate": "datum.x > 2"}
        ]
    return ChartSpec.from_json(json.dumps(wire))


def _render(spec: ChartSpec, output_kind: str) -> None:
    renderer = render_svg if output_kind == "static" else render_interactive
    renderer(spec, DF.to_arrow(), viewport=(400.0, 300.0))


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
@pytest.mark.parametrize("layer_index", [0, 1])
def test_wire_layer_transforms_is_refused_naming_the_layer(layer_index, output_kind):
    with pytest.raises(ValueError) as exc:
        _render(_layered_wire(transforms_on=layer_index), output_kind)
    assert str(exc.value) == (
        f"layer {layer_index}: layers[{layer_index}].transforms carries 1 transform(s), "
        "but a layer does not run its own transforms. Give the layer its own chart with "
        ".transform(...) and compose it with +, which routes the transform to that layer alone."
    )


@pytest.mark.parametrize("output_kind", ["static", "interactive"])
def test_wire_without_layer_transforms_renders(output_kind):
    """Vacuity control: the same hand-built spec without layer transforms renders."""
    _render(_layered_wire(transforms_on=None), output_kind)


# ---------------------------------------------------------------------------
# The working route: a named chart-level transform read through data_source.
# ---------------------------------------------------------------------------


def test_named_transform_read_through_data_source_still_filters_its_layer():
    points = fm.Chart(DF).mark_point().encode(x="x", y="y")
    filtered_line = fm.Chart(DF).transform(_filter()).mark_line().encode(x="x", y="y")
    chart = points + filtered_line

    wire = json.loads(chart.to_json())
    (named,) = wire["transforms"]
    assert named["type"] == "filter"
    assert "data_source" not in wire["layers"][0]
    assert wire["layers"][1]["data_source"] == named["name"]

    svg = chart.to_svg()
    assert svg.count("<circle") == 4
    (line_points,) = re.findall(r'<polyline points="([^"]*)"', svg)
    assert len(line_points.split()) == 2
