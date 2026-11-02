"""A line, area, ribbon or polygon too short to draw in every series names the blank.

A line, area or ribbon joins two or more points per series, and a polygon
three or more vertices per ring, so a series with one point (a ring with
one or two vertices) draws nothing. When that empties the whole layer, the
render warns ``<mark> layer <i> drew nothing: <reason>``, naming how many
series (bands for a ribbon, polygons for a polygon) were too short,
instead of rendering blank in silence. A layer where any series or ring
draws stays silent.
"""

from __future__ import annotations

import re
import warnings
from typing import Callable

import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import step_after_paths

_ONE_ROW = pl.DataFrame({"v": [1.0], "w": [2.0], "lo": [1.0], "hi": [3.0]})

_LINE_ONE = (
    "line layer 0 drew nothing: mark_line() joins two or more points per series, "
    "and its one series has one"
)
_AREA_ONE = (
    "area layer 0 drew nothing: mark_area() spans two or more points per series, "
    "and its one series has one"
)
_RIBBON_ONE = (
    "ribbon layer 0 drew nothing: mark_ribbon() spans two or more points per band, "
    "and its one band has one"
)

#: Rows in five groups of three, for the ECDF's aggregate remedy.
_GROUPED_ROWS = pl.DataFrame(
    {"g": [i // 3 for i in range(15)], "v": [float(i + 1) for i in range(15)]}
)


def _render_capturing(render: Callable[[], str]) -> tuple[str, list[str]]:
    """Render inside a warning capture: the SVG and every warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = render()
    return svg, [str(w.message) for w in caught]


@pytest.mark.parametrize(
    "render,expected",
    [
        (lambda: fm.Chart(_ONE_ROW).mark_line().encode(x="v", y="w").to_svg(), _LINE_ONE),
        (lambda: fm.Chart(_ONE_ROW).mark_area().encode(x="v", y="w").to_svg(), _AREA_ONE),
        (
            lambda: fm.Chart(_ONE_ROW).mark_ribbon().encode(x="v", y="lo", y2="hi").to_svg(),
            _RIBBON_ONE,
        ),
    ],
    ids=["line", "area", "ribbon"],
)
def test_a_series_mark_over_one_row_warns_that_it_drew_nothing(
    render: Callable[[], str], expected: str
):
    _svg, messages = _render_capturing(render)
    assert expected in messages


def test_a_one_row_ecdf_draws_a_riser_and_stays_silent():
    """A plain `mark_line` over one row draws nothing and warns (`_LINE_ONE`
    above); the ECDF's own line declares the zero riser, so its one row
    still draws -- a vertical rise from 0 to 1 -- with no warning."""
    svg, messages = _render_capturing(lambda: fm.displot(_ONE_ROW, x="v", kind="ecdf").to_svg())
    assert messages == [], messages
    (corners,) = [corners for _stroke, corners in step_after_paths(svg)]
    assert len(corners) == 2, corners


def test_the_ecdf_aggregate_remedy_without_a_groupby_draws_like_its_one_row_sibling():
    """An aggregate with no groupby leaves one row, so the ECDF over it has one
    point: the remedy and the same door over the one-row frame polars computes
    both draw the same riser silently, and render the same chart.
    """
    remedied = fm.displot(_GROUPED_ROWS, x="mean_v", kind="ecdf").transform(
        fm.transform_aggregate({"field": "v", "fn": "mean", "as": "mean_v"})
    )
    sibling = fm.displot(_GROUPED_ROWS.select(mean_v=pl.col("v").mean()), x="mean_v", kind="ecdf")
    remedied_svg, remedied_messages = _render_capturing(remedied.to_svg)
    sibling_svg, sibling_messages = _render_capturing(sibling.to_svg)
    assert remedied_messages == [], remedied_messages
    assert sibling_messages == [], sibling_messages
    assert remedied_svg == sibling_svg


def test_an_ecdf_with_one_single_row_hue_level_draws_both_and_stays_silent():
    """The single-row `"a"` level now draws its own riser (not nothing),
    beside the four-row `"b"` level's full step line -- two lines, silent."""
    rows = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0], "h": ["a", "b", "b", "b", "b"]})
    svg, messages = _render_capturing(
        lambda: fm.displot(rows, x="v", kind="ecdf", hue="h").to_svg()
    )
    assert len(step_after_paths(svg)) == 2
    assert not any("drew nothing" in message for message in messages), messages


# ---------------------------------------------------------------------------
# Polygon: a ring needs three or more vertices.
# ---------------------------------------------------------------------------

_POLYGON_ONE = (
    "polygon layer {layer} drew nothing: a polygon joins three or more vertices, "
    "and its one polygon has one or two"
)
_POLYGON_TWO = (
    "polygon layer 0 drew nothing: a polygon joins three or more vertices, "
    "and each of its 2 polygons has one or two"
)


def _polygon_rings(svg: str) -> list[str]:
    """The closed, even-odd-filled ``<path>`` each drawn polygon ring renders as."""
    return re.findall(r'<path d="M[^"]*Z" fill-rule="evenodd"', svg)


def _polygon_layer_over(rows: pl.DataFrame) -> fm.Chart:
    return fm.Chart(rows).encode(x="x", y="y").layer(fm.Layer(mark="polygon"))


def test_a_polygon_layer_over_two_rows_warns_that_it_drew_nothing():
    two = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
    svg, messages = _render_capturing(_polygon_layer_over(two).to_svg)
    assert _polygon_rings(svg) == []
    assert _POLYGON_ONE.format(layer=1) in messages


def test_a_polygon_layer_over_a_triangle_draws_it_and_stays_silent():
    triangle = pl.DataFrame({"x": [0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0]})
    svg, messages = _render_capturing(_polygon_layer_over(triangle).to_svg)
    assert len(_polygon_rings(svg)) == 1
    assert not any("drew nothing" in message for message in messages), messages


@pytest.mark.parametrize(
    "groups,expected",
    [(["a"], _POLYGON_ONE.format(layer=0)), (["a", "b"], _POLYGON_TWO)],
    ids=["one_violin", "two_violins"],
)
def test_a_violin_whose_explicit_domain_leaves_each_outline_short_warns(
    groups: list[str], expected: str
):
    """A y domain thinner than the violin's density grid step keeps at most two
    outline vertices per violin, so no outline closes and the layer is blank.
    """
    rows = pl.DataFrame(
        {
            "g": [group for group in groups for _ in range(30)],
            "v": [float(i % 9) for i in range(30)] * len(groups),
        }
    )
    sliver = fm.Y("v", scale=fm.LinearScale(domain=[0.4, 0.41]))
    svg, messages = _render_capturing(
        lambda: fm.Chart(rows).mark_violin().encode(x="g", y=sliver).to_svg()
    )
    assert _polygon_rings(svg) == []
    assert expected in messages
