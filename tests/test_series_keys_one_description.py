"""The fields a mark splits into series are read once, by the draw, the null-key drop and the impute.

A colour column splits a series when the colour scale draws it as categories, whatever its dtype: a
text column, and an integer column an area colours (an area that draws colour makes the colour
scale categorical for every layer of the panel). A column a continuous colour scale excludes splits
nothing. The cases below pin four inputs on which the draw, the null-key drop and the impute read
that one description.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any

import polars as pl

import ferrum as fm
from tests._svg_rects import band_paths, mark_layer, polylines


def _capture(render: Callable[[], Any]) -> tuple[Any, list[str]]:
    """What ``render`` returned and every warning it raised, under ``always``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render()
    return out, [str(w.message) for w in caught]


def _band_commands(svg: str) -> list[int]:
    """The path commands of each area band, in document order."""
    return [len(re.findall(r"[MLHVZ]", d)) for _, d in band_paths(mark_layer(svg))]


def _line_vertices(svg: str) -> list[int]:
    return [len(points) for _, points in polylines(mark_layer(svg))]


_IMPUTE = fm.Y("v", impute={"value": 0})

#: Series ``h=2`` lacks ``t=2``; ``h`` is an integer column.
_INT_COLOUR = pl.DataFrame(
    {"t": [1, 2, 3, 1, 3], "h": [1, 1, 1, 2, 2], "v": [1.0, 2.0, 3.0, 4.0, 5.0]}
)
#: The same frame with the row ``(t=2, h=2, v=0)`` written out.
_INT_COLOUR_WRITTEN = pl.DataFrame(
    {"t": [1, 2, 3, 1, 2, 3], "h": [1, 1, 1, 2, 2, 2], "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0]}
)


def _area(frame: pl.DataFrame, y: Any) -> fm.Chart:
    return fm.Chart(frame).mark_area().encode(x="t", y=y, color="h")


def test_an_area_coloured_by_integers_completes_each_coloured_series():
    """The area draws one band per ``h``, so the impute completes the ``h=2`` band: the SVG equals
    the twin with the row written out, differs from the chart without ``impute=`` (whose ``h=2``
    band keeps its two caller rows), and nothing warns."""
    svg, shown = _capture(lambda: _area(_INT_COLOUR, _IMPUTE).to_svg())
    plain, _ = _capture(lambda: _area(_INT_COLOUR, "v").to_svg())
    twin, _ = _capture(lambda: _area(_INT_COLOUR_WRITTEN, "v").to_svg())
    assert shown == []
    assert _band_commands(svg) == [6, 6]
    assert _band_commands(plain) == [6, 5]
    assert svg == twin
    assert svg != plain


#: Integer colour with a null on one row.
_INT_COLOUR_NULL = pl.DataFrame(
    {
        "t": [1, 2, 3, 1, 2, 3, 2],
        "h": [1, 1, 1, 2, 2, 2, None],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)
_NULL_LEVEL_DROPPED = (
    "mark_area: 1 row has a null 'h' and belongs to no group, so it was dropped; "
    "fill or filter the nulls in 'h'"
)


def test_a_null_integer_colour_on_an_area_is_dropped_as_a_null_level():
    """A null in the integer colour belongs to no band, so it is dropped with the null-key
    sentence, as a text colour's null is, and the bands equal the text-coloured chart's. It is not
    left as a one-point series the area refuses to span."""
    svg, shown = _capture(lambda: _area(_INT_COLOUR_NULL, "v").to_svg())
    text, text_shown = _capture(
        lambda: _area(_INT_COLOUR_NULL.with_columns(pl.col("h").cast(pl.Utf8)), "v").to_svg()
    )
    assert shown == [_NULL_LEVEL_DROPPED]
    assert text_shown == [_NULL_LEVEL_DROPPED]
    assert _band_commands(svg) == _band_commands(text) == [6, 6]


def test_a_line_beside_a_colour_grouping_area_completes_the_series_it_draws():
    """The area makes the colour scale categorical, so the line draws one polyline per ``h`` and
    its impute completes the ``h=2`` polyline: the line's polylines equal the same layering over
    ``_INT_COLOUR_WRITTEN`` with no ``impute=`` on the line, and the area's bands are unchanged."""
    area = fm.Chart(_INT_COLOUR).mark_area().encode(x="t", y="v", color="h")
    line = fm.Chart(_INT_COLOUR).mark_line().encode(x="t", y=_IMPUTE, color="h")
    written = fm.Chart(_INT_COLOUR_WRITTEN).mark_line().encode(x="t", y="v", color="h")
    svg, shown = _capture(lambda: (area + line).to_svg())
    twin, twin_shown = _capture(lambda: (area + written).to_svg())
    assert shown == twin_shown == []
    assert _line_vertices(svg) == _line_vertices(twin) == [3, 3]
    assert _band_commands(svg) == _band_commands(twin) == [6, 5]


#: ``q`` is a number and establishes a continuous colour scale on the points; ``s`` is text with
#: one null and cannot join that scale.
_EXCLUDED = pl.DataFrame(
    {
        "t": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "q": [0.1, 0.5, 0.9, 0.2, 0.7, 0.3],
        "s": ["a", "a", "b", None, "b", "a"],
    }
)
_SCALE_EXCLUDED = (
    "layer 1: its color scale differs from the one layer 0 establishes the color scale with, so it "
    "stays out of that scale; its marks draw through that scale and may fall outside its legend"
)


def test_a_colour_excluded_from_the_established_scale_keys_no_series():
    """The line draws one unsplit polyline, so its null ``s`` is kept and drawn: only the scale
    exclusion is reported (no null-key drop of ``s``), and the impute completes the one series the
    line draws, whose polyline holds all six rows."""
    points = fm.Chart(_EXCLUDED).mark_point().encode(x="t", y="v", color="q")
    line = fm.Chart(_EXCLUDED).mark_line().encode(x="t", y=_IMPUTE, color="s")
    svg, shown = _capture(lambda: (points + line).to_svg())
    assert shown == [_SCALE_EXCLUDED]
    assert _line_vertices(svg) == [6]
