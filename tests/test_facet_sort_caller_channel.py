"""A partial facet sort names the channel the caller wrote.

``facet("c", sort="y")`` reads the caller's ``y`` from each value's rows, and when a value's rows
carry no such channel the warning says "carry no y value" under every coordinate system: a
``CoordFlip`` draws the caller's ``y`` on the horizontal axis, but the sentence keeps the token
the caller typed. The panel order and the sort-object form do not change with the flip.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from ferrum.coord import CoordFlip
from tests._legends import chrome_texts
from tests.test_null_group_warning import _facet_sort_partial_message, _render_capturing

_HOST = pl.DataFrame(
    {"c": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]}
)
#: The rows of value ``s``: both columns are present, but a rule layer draws only one of them.
_S_ROWS = pl.DataFrame({"c": ["s", "s"], "u": [1.0, 2.0], "w": [1.0, 2.0]})


def _partial_sentence(source: str) -> str:
    return _facet_sort_partial_message(["s"], source)


def _chart(s_encoding: dict[str, str], sort: Any, *, flipped: bool) -> fm.Chart:
    """Points of ``a`` and ``b`` at the caller's ``x`` and ``y``, plus a rule layer holding ``s``
    that draws only the channel in ``s_encoding``, faceted by ``c``."""
    host = fm.Chart(_HOST).mark_point().encode(x="x", y="y")
    rule = fm.Chart(_S_ROWS).mark_rule().encode(**s_encoding)
    chart = (host + rule).facet("c", sort=sort)
    return chart.coord(CoordFlip()) if flipped else chart


def _panels(svg: str) -> list[str]:
    texts = ["".join(e.itertext()).strip() for e in chrome_texts(svg)]
    return [text for text in texts if text in {"a", "b", "s"}]


def _render_kind(chart: fm.Chart, kind: str) -> tuple[list[str], list[str]]:
    """The facet panel titles ``chart`` draws in order on ``kind`` (static svg or interactive
    scene) and the warnings raised."""
    if kind == "static":
        svg, shown = _render_capturing(chart.to_svg)
        return _panels(svg), shown
    scene, shown = _render_capturing(lambda: chart.interactive().scene_json)
    return _scene_panels(scene), shown


def _scene_panels(scene_json: str) -> list[str]:
    """The facet panel titles of an interactive scene, in order."""
    scene = json.loads(scene_json)
    return [
        node["content"]
        for panel in scene["panels"]
        for node in panel["strip_title"]
        if node["type"] == "text"
    ]


#: Per shorthand: the rule's encoding that lacks the channel the sort reads, and the panel order
#: the one sorted pair and the value with no rows take (``s`` sorts first ascending, last
#: descending; ``a`` and ``b`` follow the sort).
_SHORTHANDS = {
    "x": ({"y": "w"}, ["s", "a", "b"]),
    "-x": ({"y": "w"}, ["b", "a", "s"]),
    "y": ({"x": "u"}, ["s", "b", "a"]),
    "-y": ({"x": "u"}, ["a", "b", "s"]),
}


_KINDS = ["static", "interactive"]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("flipped", [False, True], ids=["unflipped", "flipped"])
@pytest.mark.parametrize("sort", list(_SHORTHANDS))
def test_a_shorthand_facet_sort_names_the_callers_channel_under_every_flip(sort, flipped, kind):
    """``sort="-y"`` is about the caller's ``y``: the sentence says ``y``, flipped or not."""
    encoding, order = _SHORTHANDS[sort]
    panels, shown = _render_kind(_chart(encoding, sort, flipped=flipped), kind)
    assert shown == [_partial_sentence(f"{sort.lstrip('-')} value")]
    assert panels == order


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("flipped", [False, True], ids=["unflipped", "flipped"])
@pytest.mark.parametrize("sort", list(_SHORTHANDS))
def test_a_shorthand_facet_sort_over_every_value_with_the_channel_warns_nothing(
    sort, flipped, kind
):
    both = {"x": "u", "y": "w"}
    panels, shown = _render_kind(_chart(both, sort, flipped=flipped), kind)
    assert shown == []
    assert sorted(panels) == ["a", "b", "s"]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("flipped", [False, True], ids=["unflipped", "flipped"])
def test_a_sort_object_names_its_field_under_every_flip(flipped, kind):
    """The object form reads a named field, so the sentence names the field, not a channel."""
    sort = {"field": "y", "op": "sum", "order": "descending"}
    panels, shown = _render_kind(_chart({"x": "u"}, sort, flipped=flipped), kind)
    assert shown == [_partial_sentence("'y' value")]
    assert panels == ["a", "b", "s"]


@pytest.mark.parametrize("kind", _KINDS)
def test_the_facet_sort_channel_is_the_callers_on_the_encode_door(kind):
    """``encode(facet=fm.Facet("c", sort="y"))`` is the same declaration as ``.facet``."""
    host = fm.Chart(_HOST).mark_point().encode(x="x", y="y")
    rule = fm.Chart(_S_ROWS).mark_rule().encode(x="u")
    chart = (host + rule).encode(facet=fm.Facet("c", sort="y")).coord(CoordFlip())
    _, shown = _render_kind(chart, kind)
    assert shown == [_partial_sentence("y value")]


@pytest.mark.parametrize("kind", _KINDS)
def test_the_facet_sort_channel_is_the_callers_on_the_chart_layer_door(kind):
    """``Chart.layer`` composes the same rows as ``+``, and the sentence is the same."""
    host = fm.Chart(_HOST).mark_point().encode(x="x", y="y")
    layer = fm.Layer(mark="rule", data=_S_ROWS, encoding={"x": "u"})
    chart = host.layer(layer).facet("c", sort="y").coord(CoordFlip())
    _, shown = _render_kind(chart, kind)
    assert shown == [_partial_sentence("y value")]
