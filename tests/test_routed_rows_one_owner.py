"""Every reader of a layer's rows reads the rows that layer draws.

A layer that keys a field some rows hold null drops those rows and reads its own copy of the table,
while a sibling that does not key the field keeps them. Every reader of a layer's rows, the drawing,
the legend, the colour scale and the impute's binned series among them, reads that copy:

* each layer draws what it draws as the only layer, whatever its siblings key;
* a figure legend shared across a composite colours a level that only a moved layer holds as that
  layer's marks colour it;
* the impute of a binned colour on a moved layer keys its series by the bins that layer draws;
* a faceted chart whose secondary y axis is independent keeps each panel's own axis.

Each test builds its charts inside the capture, so a build-time warning is counted too.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import value_axis_ladders
from tests._svg_rects import mark_layer, polylines


def _capture(render: Callable[[], Any]) -> tuple[Any, list[str]]:
    """What ``render`` returned and every warning it raised, under ``always``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render()
    return out, [str(w.message) for w in caught]


def _dropped(entry: str, column: str) -> str:
    """The sentence that names the one row ``entry`` dropped for a null ``column``."""
    return (
        f"{entry}: 1 row has a null '{column}' and belongs to no group, so it was dropped; "
        f"fill or filter the nulls in '{column}'"
    )


def _circles(svg: str) -> int:
    return len(re.findall(r"<circle\b", mark_layer(svg)))


def _line_vertices(svg: str) -> list[int]:
    return sorted(len(points) for _, points in polylines(mark_layer(svg)))


# ---------------------------------------------------------------------------
# Each layer draws the rows it draws alone
# ---------------------------------------------------------------------------

#: Rows that hold a null in a different field each: ``s`` (shape) and ``k`` (colour) lose one row
#: each, ``d`` (detail) one more, and ``c`` (a text x) one more, so the layers keying different
#: fields drop different rows.
_FRAME = pl.DataFrame(
    {
        "t": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "v": [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5, 8.0, 9.5],
        "s": ["p", "p", None, "q", "q", "p", "q", "p", "q", "p", "q", "p"],
        "k": ["u", "u", "w", "w", None, "u", "w", "u", "w", "u", "w", "u"],
        "d": ["m", "n", "m", "n", "m", "n", None, "m", "n", "m", "n", "m"],
        "c": ["a", "b", "a", "b", "a", "b", "a", None, "a", "b", "a", "b"],
        "f": ["F1", "F2"] * 6,
    }
)
_IMPUTED = fm.Y("v", impute={"value": 0})
_POINT_KEYS: dict[str, dict[str, Any]] = {
    "shape": {"x": "t", "y": "v", "shape": "s"},
    "colour": {"x": "t", "y": "v", "color": "k"},
    "text-x": {"x": "c", "y": "v"},
}
_LINE_KEYS: dict[str, dict[str, Any]] = {
    "colour": {"x": "t", "y": "v", "color": "k"},
    "detail": {"x": "t", "y": "v", "detail": "d"},
    "text-x": {"x": "c", "y": "v"},
}
_IMPUTE_ON = {
    "none": (False, False),
    "point": (True, False),
    "line": (False, True),
    "both": (True, True),
}


def _layer_chart(mark: str, keys: dict[str, Any], imputed: bool) -> fm.Chart:
    encoding = {**keys, "y": _IMPUTED} if imputed else keys
    return getattr(fm.Chart(_FRAME), mark)().encode(**encoding)


#: A text x on one layer is a band axis the other cannot share, so it is paired with itself.
_KEY_PAIRS = [
    pytest.param(point, line, id=f"{point}+{line}")
    for point in sorted(_POINT_KEYS)
    for line in sorted(_LINE_KEYS)
    if (point == "text-x") == (line == "text-x")
]


@pytest.mark.parametrize("impute_on", sorted(_IMPUTE_ON))
@pytest.mark.parametrize("point_keys,line_keys", _KEY_PAIRS)
def test_each_layer_draws_the_rows_it_draws_alone_whatever_its_siblings_key(
    point_keys: str, line_keys: str, impute_on: str
):
    point_imputed, line_imputed = _IMPUTE_ON[impute_on]
    point = _layer_chart("mark_point", _POINT_KEYS[point_keys], point_imputed)
    line = _layer_chart("mark_line", _LINE_KEYS[line_keys], line_imputed)
    svg, _ = _capture((point + line).to_svg)
    point_alone, _ = _capture(point.to_svg)
    line_alone, _ = _capture(line.to_svg)
    assert _circles(svg) == _circles(point_alone)
    assert _line_vertices(svg) == _line_vertices(line_alone)


def test_a_layer_with_its_own_data_draws_what_it_draws_alone_beside_a_keyed_sibling():
    other = pl.DataFrame(
        {"t": [1.0, 2.0, 3.0, 4.0], "v": [2.0, 3.0, 1.0, 5.0], "k": ["u", "u", None, "w"]}
    )
    point = _layer_chart("mark_point", _POINT_KEYS["shape"], False)
    own = fm.Chart(other).mark_line().encode(x="t", y="v", color="k")
    layered = point.layer(
        fm.Layer(mark="line", data=other, encoding={"x": "t", "y": "v", "color": "k"})
    )
    svg, _ = _capture(layered.to_svg)
    assert _circles(svg) == _circles(_capture(point.to_svg)[0])
    assert _line_vertices(svg) == _line_vertices(_capture(own.to_svg)[0])


@pytest.mark.parametrize("door", ["plus", "chart-layer"])
def test_each_faceted_layer_draws_the_rows_it_draws_alone(door: str):
    point = _layer_chart("mark_point", _POINT_KEYS["shape"], False)
    line = _layer_chart("mark_line", _LINE_KEYS["colour"], True)
    layered = (
        point + line
        if door == "plus"
        else point.layer(
            fm.Layer(
                mark="line", encoding={"x": "t", "y": fm.Y("v", impute={"value": 0}), "color": "k"}
            )
        )
    )
    svg, _ = _capture(layered.facet(col="f").to_svg)
    point_alone, _ = _capture(point.facet(col="f").to_svg)
    line_alone, _ = _capture(line.facet(col="f").to_svg)
    assert _circles(svg) == _circles(point_alone)
    assert _line_vertices(svg) == _line_vertices(line_alone)


# ---------------------------------------------------------------------------
# The two readers that read another layer's rows
# ---------------------------------------------------------------------------

#: ``x = y = 1..5``, the level ``z`` of ``c`` held by the last row alone, and a null ``g`` on that
#: row: the shape layer (keyed on ``g``) drops it, the colour layer (keyed on ``c``) keeps it, so
#: only the colour layer's own rows hold ``z``.
_LEGEND_ROWS = {
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [1.0, 2.0, 3.0, 4.0, 5.0],
    "c": ["a", "b", "a", "b", "z"],
}
_MOVED_G = ["p", "p", "q", "q", None]
_KEPT_G = ["p", "p", "q", "q", "r"]


def _legend_leaf(g: list[str | None]) -> fm.Chart:
    df = pl.DataFrame({**_LEGEND_ROWS, "g": g})
    shaped = fm.Chart(df).mark_point().encode(x="x", y="y", shape="g")
    coloured = fm.Chart(df).mark_point().encode(x="x", y="y", color="c")
    return shaped + coloured


_LEGEND_OTHER = (
    fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "c": ["a", "b"]}))
    .mark_point()
    .encode(x="x", y="y", color="c")
)


def _swatches(svg: str) -> dict[str, str]:
    """Each figure-legend entry's swatch colour, by its label."""
    found = re.findall(r'fill="(#[0-9a-fA-F]{6})"[^>]*/>\s*<text[^>]*>(a|b|z)</text>', svg)
    return {label: colour for colour, label in found}


def _shared_legend(g: list[str | None]) -> str:
    combo = fm.hconcat(
        _legend_leaf(g), _LEGEND_OTHER, resolve=fm.Resolve(scale={"color": "shared"})
    )
    return combo.to_svg()


def test_a_figure_legend_colours_a_level_only_a_moved_layer_draws_as_its_marks():
    moved, _ = _capture(lambda: _shared_legend(_MOVED_G))
    twin, twin_shown = _capture(lambda: _shared_legend(_KEPT_G))
    alone, _ = _capture(_legend_leaf(_MOVED_G).to_svg)
    assert twin_shown == []
    assert _swatches(twin) == {"a": "#2563eb", "b": "#dc2626", "z": "#d4a017"}
    assert _swatches(alone)["z"] == "#d4a017"
    assert _swatches(moved) == _swatches(twin)


def test_the_drop_of_a_moved_layer_is_named_beside_a_shared_figure_legend():
    _, shown = _capture(lambda: _shared_legend(_MOVED_G))
    assert shown == [_dropped("mark_point", "g")]


#: ``t`` and the bin ``h`` sit on three rows each side of a gap: the line layer keyed on the
#: binned colour reads every row, while the point layer keyed on ``g`` drops the last (``h=30``).
_BINS = {
    "t": [1, 2, 3, 1, 3, 2],
    "h": [1.0, 1.2, 1.1, 8.0, 8.5, 30.0],
    "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}


def _bins_chart(g: list[str | None], *, with_points: bool) -> fm.Chart:
    df = pl.DataFrame({**_BINS, "g": g})
    line = (
        fm.Chart(df)
        .mark_line()
        .encode(x="t", y=fm.Y("v", impute={"value": 0}), color=fm.Color("h", bin=True))
    )
    if not with_points:
        return line
    return fm.Chart(df).mark_point().encode(x="t", y="v", shape="g") + line


def test_an_imputed_binned_colour_on_a_moved_layer_keys_its_series_by_the_bins_it_draws():
    moved, moved_shown = _capture(
        _bins_chart(["p", "p", "p", "p", "p", None], with_points=True).to_svg
    )
    alone, alone_shown = _capture(
        _bins_chart(["p", "p", "p", "p", "p", None], with_points=False).to_svg
    )
    twin, twin_shown = _capture(
        _bins_chart(["p", "p", "p", "p", "p", "q"], with_points=True).to_svg
    )
    assert alone_shown == []
    assert twin_shown == []
    assert len(_line_vertices(alone)) == 3
    assert _line_vertices(twin) == _line_vertices(alone)
    assert _line_vertices(moved) == _line_vertices(alone)
    assert moved_shown == [_dropped("mark_point", "g")]


# ---------------------------------------------------------------------------
# A faceted chart whose secondary y axis is independent
# ---------------------------------------------------------------------------


def test_each_panel_of_a_faceted_independent_secondary_axis_keeps_its_own_ticks():
    frame = pl.DataFrame(
        {
            "f": ["a", "a", "b", "b"],
            "thresh": [1.0, 2.0, 1.5, 2.5],
            "z": [10.0, 20.0, 100.0, 200.0],
        }
    )
    chart = (
        (fm.Chart(frame).mark_rule().encode(x="thresh") + fm.SecondaryY("z"))
        .facet(col="f")
        .share_scale(y="independent")
    )
    svg, shown = _capture(chart.to_svg)
    assert shown == []
    ladders = [list(ladder) for _, ladder in value_axis_ladders(svg, axis="y")]
    assert ladders == [
        [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0],
        [200.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0],
    ]
