"""A line, area, ribbon or polygon too short to draw names the blank it leaves.

A line, area or ribbon joins two or more points per series, and a polygon
three or more vertices per ring, so a series with one point (a ring with
one or two vertices) draws nothing. When that empties the whole layer, the
render warns ``<mark> layer <i> drew nothing: <reason>``, naming how many
series (bands for a ribbon, polygons for a polygon) were too short,
instead of rendering blank in silence. When only some series are too
short, the layer still draws the rest and warns once, naming each dropped
series by the caller's own grouping field (``g=b``) instead of staying
silent about it; a facet chart's drops across every panel still fold into
that one warning. A layer where every series draws stays silent.
"""

from __future__ import annotations

import re
import warnings
from typing import Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import band_paths, polylines, step_after_paths

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
    _, messages = _render_capturing(render)
    assert expected in messages


def test_a_one_row_ecdf_draws_a_riser_and_stays_silent():
    """A plain `mark_line` over one row draws nothing and warns (`_LINE_ONE`
    above); the ECDF's own line declares the zero riser, so its one row
    still draws -- a vertical rise from 0 to 1 -- with no warning."""
    svg, messages = _render_capturing(lambda: fm.displot(_ONE_ROW, x="v", kind="ecdf").to_svg())
    assert messages == [], messages
    (corners,) = [corners for _, corners in step_after_paths(svg)]
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


def _ring_vertex_count(ring: str) -> int:
    """The number of straight-line vertices one ``_polygon_rings`` match draws."""
    d = re.search(r'd="([^"]*)"', ring).group(1)
    return len(re.findall(r"[ML]", d))


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


# ---------------------------------------------------------------------------
# A layer that draws SOME series but drops others too short to draw: warn
# once, naming every dropped series, instead of staying silent.
# ---------------------------------------------------------------------------

#: Group "a" has two points and draws; group "b" has one and is dropped.
_PARTIAL_GROUPS = pl.DataFrame(
    {
        "v": [1.0, 2.0, 1.0],
        "w": [1.0, 2.0, 1.0],
        "lo": [1.0, 2.0, 1.0],
        "hi": [3.0, 4.0, 3.0],
        "g": ["a", "a", "b"],
    }
)

_LINE_PARTIAL = (
    "mark_line: mark_line() joins two or more points per series, "
    "so it did not draw g=b (one point)"
)
_AREA_PARTIAL = (
    "mark_area: mark_area() spans two or more points per series, "
    "so it did not draw g=b (one point)"
)
_RIBBON_PARTIAL = (
    "mark_ribbon: mark_ribbon() spans two or more points per band, "
    "so it did not draw g=b (one point)"
)


@pytest.mark.parametrize(
    "render,expected,drawn_paths",
    [
        (
            lambda: fm.Chart(_PARTIAL_GROUPS).mark_line().encode(x="v", y="w", color="g").to_svg(),
            _LINE_PARTIAL,
            polylines,
        ),
        (
            lambda: fm.Chart(_PARTIAL_GROUPS).mark_area().encode(x="v", y="w", color="g").to_svg(),
            _AREA_PARTIAL,
            band_paths,
        ),
        (
            lambda: fm.Chart(_PARTIAL_GROUPS)
            .mark_ribbon()
            .encode(x="v", y="lo", y2="hi", color="g")
            .to_svg(),
            _RIBBON_PARTIAL,
            band_paths,
        ),
    ],
    ids=["line", "area", "ribbon"],
)
def test_a_partial_short_series_drop_names_the_dropped_series(
    render: Callable[[], str], expected: str, drawn_paths: Callable[[str], list]
):
    svg, messages = _render_capturing(render)
    assert messages == [expected]
    assert len(drawn_paths(svg)) == 1


def test_a_partial_drop_grouped_by_detail_names_the_detail_value():
    """`detail=` splits a line's series the same way `color=` does, with
    no color channel involved at all."""
    rows = pl.DataFrame({"v": [1.0, 2.0, 1.0], "w": [1.0, 2.0, 1.0], "d": ["a", "a", "b"]})
    _, messages = _render_capturing(
        lambda: fm.Chart(rows).mark_line().encode(x="v", y="w", detail="d").to_svg()
    )
    expected = (
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw d=b (one point)"
    )
    assert messages == [expected]


def test_a_null_vertex_in_an_otherwise_two_row_group_still_counts_it_short():
    """Group "b" has two rows but one y is null, so only one vertex is
    placed -- the same "(one point)" outcome as a genuinely one-row group."""
    rows = pl.DataFrame(
        {
            "v": [1.0, 2.0, 1.0, 2.0],
            "w": [1.0, 2.0, 1.0, None],
            "g": ["a", "a", "b", "b"],
        }
    )
    _, messages = _render_capturing(
        lambda: fm.Chart(rows).mark_line().encode(x="v", y="w", color="g").to_svg()
    )
    expected = (
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=b (one point)"
    )
    assert messages == [expected]


_DASH_ROWS = pl.DataFrame(
    {
        "x": [0.0, 1.0, 0.0, 1.0, 0.0],
        "y": [0.0, 1.0, 0.0, 1.0, 0.0],
        "g": ["a", "a", "b", "b", "c"],
        "d": [1.0, 1.0, 2.0, 2.0, 3.0],
    }
)


@pytest.mark.parametrize(
    "render,expected",
    [
        (
            lambda: fm.Chart(_DASH_ROWS)
            .mark_line()
            .encode(x="x", y="y", color="g", stroke_dash="d")
            .to_svg(),
            "mark_line: mark_line() joins two or more points per series, "
            "so it did not draw g=c (one point)",
        ),
        (
            lambda: fm.Chart(_DASH_ROWS)
            .mark_line()
            .encode(x="x", y="y", color="g", stroke_dash="d:N")
            .to_svg(),
            "mark_line: mark_line() joins two or more points per series, "
            "so it did not draw g=c, d=3 (one point)",
        ),
        (
            lambda: fm.Chart(_DASH_ROWS.with_columns((pl.col("y") + 1).alias("hi")))
            .mark_ribbon()
            .encode(x="x", y="y", y2="hi", color="g", stroke_dash="d")
            .to_svg(),
            "mark_ribbon: mark_ribbon() spans two or more points per band, "
            "so it did not draw g=c (one point)",
        ),
        (
            lambda: fm.Chart(_DASH_ROWS.with_columns((pl.col("y") + 1).alias("hi")))
            .mark_ribbon()
            .encode(x="x", y="y", y2="hi", color="g", stroke_dash="d:N")
            .to_svg(),
            "mark_ribbon: mark_ribbon() spans two or more points per band, "
            "so it did not draw g=c, d=3 (one point)",
        ),
    ],
    ids=[
        "line_untyped_numeric_dash",
        "line_typed_nominal_dash",
        "ribbon_untyped_numeric_dash",
        "ribbon_typed_nominal_dash",
    ],
)
def test_a_numeric_stroke_dash_names_a_dropped_series_only_when_it_splits_it(
    render: Callable[[], str], expected: str
):
    """A dash field only widens a dropped series' name when it actually
    splits the series apart. An untyped numeric ``stroke_dash=`` is a dash
    index, not a category -- it draws no dash scale and splits nothing, so
    the series stays named by color alone. Typed nominal (``d:N``) puts the
    field under a dash scale, which reads it as a category and splits the
    series by it, so the same short series is named by both fields."""
    _, messages = _render_capturing(render)
    assert messages == [expected]


_POLYGON_PARTIAL_RINGS = pl.DataFrame(
    {
        "x": [0.0, 1.0, 0.5, 0.0, 1.0],
        "y": [0.0, 0.0, 1.0, 0.0, 0.0],
        "ring": ["p", "p", "p", "q", "q"],
    }
)


def test_a_polygon_layer_with_one_short_ring_names_its_detail_value():
    """Ring "p" closes a triangle and draws; ring "q" has only two vertices
    and is dropped, named by the detail field that distinguishes the rings
    rather than by any internal ring id."""
    svg, messages = _render_capturing(
        lambda: fm.Chart(_POLYGON_PARTIAL_RINGS)
        .encode(x="x", y="y", detail="ring")
        .layer(fm.Layer(mark="polygon"))
        .to_svg()
    )
    rings = _polygon_rings(svg)
    assert len(rings) == 1
    assert _ring_vertex_count(rings[0]) == 3
    expected = (
        "layer 1: mark_polygon: a polygon joins three or more vertices, "
        "so it did not draw ring=q (two vertices)"
    )
    assert messages == [expected]


def test_a_caller_placed_polygon_ring_is_named_by_detail_even_with_color_bound():
    """A caller's own `fm.Layer(mark="polygon")` splits its rings by
    `detail`; a bound `color` does not partition polygon rings at all, so
    the dropped ring is still named by its detail value, not the color
    value every ring shares."""
    rows = pl.DataFrame(
        {
            "x": [0.0, 1.0, 0.5, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 0.0, 0.0],
            "ring": ["p", "p", "p", "q", "q"],
            "kind": ["k1", "k1", "k1", "k1", "k1"],
        }
    )
    _, messages = _render_capturing(
        lambda: fm.Chart(rows)
        .encode(x="x", y="y", detail="ring", color="kind")
        .layer(fm.Layer(mark="polygon"))
        .to_svg()
    )
    expected = (
        "layer 1: mark_polygon: a polygon joins three or more vertices, "
        "so it did not draw ring=q (two vertices)"
    )
    assert messages == [expected]


def test_a_log_scale_short_hex_cell_is_named_by_its_color_field():
    """A hex cell left with only two placed vertices after a LogScale x/y
    drops its off-side rows is named by the color field it draws, never
    by its internal hex id."""
    rng = np.random.default_rng(2)
    rows = pl.DataFrame(
        {"x": list(np.exp(rng.normal(0, 1, 200))), "y": list(np.exp(rng.normal(0, 1, 200)))}
    )
    _, messages = _render_capturing(
        lambda: fm.Chart(rows)
        .mark_hex()
        .encode(x=fm.X("x", scale=fm.LogScale()), y=fm.Y("y", scale=fm.LogScale()))
        .to_svg()
    )
    assert messages == [
        "x scale: 10 row(s) of 'hex_x' are zero or negative, which a LogScale cannot place, "
        "so they are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero",
        "y scale: 24 row(s) of 'hex_y' are zero or negative, which a LogScale cannot place, "
        "so they are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero",
        "mark_hex: a polygon joins three or more vertices, so it did not draw value=7 (two vertices)",
    ]


# ---------------------------------------------------------------------------
# A series an explicit domain already reports as not drawn is not named
# again; a short series beside it still is, and only on a statistic layer,
# whose domain census makes the per-run decision -- never on a primitive
# layer, whose domain census counts rows rather than runs.
# ---------------------------------------------------------------------------

_DOMAIN_CUT_ROWS = pl.DataFrame(
    {"x": [1.0, 2.0, 1.0, 100.0], "y": [1.0, 2.0, 1.0, 2.0], "g": ["a", "a", "c", "c"]}
)
#: Group "b" (one row, genuinely short) inserted before group "c" (whose
#: x=100 row the explicit domain below cuts, leaving it with one point too).
_DOMAIN_CUT_AND_SHORT_ROWS = pl.DataFrame(
    {
        "x": [1.0, 2.0, 1.5, 1.0, 100.0],
        "y": [1.0, 2.0, 1.0, 1.0, 2.0],
        "g": ["a", "a", "b", "c", "c"],
    }
)
_DOMAIN_WARNING_MARK_COUNT = (
    "mark_line: x scale: 1 mark(s) of 'x' fall outside the scale's explicit domain, "
    "so they are not drawn; widen the domain, or pass clamp=True to draw them at the axis edge"
)
_DOMAIN_WARNING_ROW_COUNT = (
    "x scale: 1 row(s) of 'x' fall outside the scale's explicit domain, "
    "so they are not drawn; widen the domain, or pass clamp=True to draw them at the axis edge"
)


def test_a_statistic_run_the_domain_already_reports_is_not_named_again():
    """Group "c"'s x=100 row is cut by the explicit domain, leaving it with
    one aggregated point -- the domain warning already reports it as not
    drawn, so SeriesDropped does not repeat it; no other group is short."""
    chart = fm.Chart(_DOMAIN_CUT_ROWS).mark_line().encode(
        x=fm.X("x", scale=fm.LinearScale(domain=[0.0, 3.0])),
        y=fm.Y("y", aggregate="mean"),
        color="g",
    )
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_DOMAIN_WARNING_MARK_COUNT]


def test_a_short_statistic_run_beside_a_domain_cut_one_is_still_named():
    """Group "b" is short on its own (one row, not touched by the domain);
    group "c" is still exempted. The domain warning and one SeriesDropped
    naming only "b" both fire."""
    chart = fm.Chart(_DOMAIN_CUT_AND_SHORT_ROWS).mark_line().encode(
        x=fm.X("x", scale=fm.LinearScale(domain=[0.0, 3.0])),
        y=fm.Y("y", aggregate="mean"),
        color="g",
    )
    _, messages = _render_capturing(chart.to_svg)
    expected = (
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=b (one point)"
    )
    assert messages == [_DOMAIN_WARNING_MARK_COUNT, expected]


def test_a_primitive_layers_row_census_exempts_no_series():
    """Over the caller's own rows (no aggregate), the domain warning counts
    rows rather than series, so it never exempts a series from being
    named: both "b" and "c" are named beside the row-count domain warning."""
    chart = fm.Chart(_DOMAIN_CUT_AND_SHORT_ROWS).mark_line().encode(
        x=fm.X("x", scale=fm.LinearScale(domain=[0.0, 3.0])), y="y", color="g"
    )
    _, messages = _render_capturing(chart.to_svg)
    expected = (
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=b (one point), g=c (one point)"
    )
    assert messages == [_DOMAIN_WARNING_ROW_COUNT, expected]


#: Two facet panels on "c"; panel "p1" draws group "a" only, panel "p2"
#: draws "a" and drops both "b" and "c".
_FACET_PARTIAL_ROWS = pl.DataFrame(
    {
        "v": [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
        "w": [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
        "g": ["a", "a", "a", "a", "b", "c"],
        "c": ["p1", "p1", "p2", "p2", "p2", "p2"],
    }
)


def test_a_faceted_partial_drop_warns_once_naming_every_short_series():
    """Both drops share panel p2, so both name it -- each dropped series
    carries its own panel clause, even when it repeats the one before it."""
    _, messages = _render_capturing(
        lambda: fm.Chart(_FACET_PARTIAL_ROWS)
        .mark_line()
        .encode(x="v", y="w", color="g", facet="c")
        .to_svg()
    )
    expected = (
        "mark_line: mark_line() joins two or more points per series, so it did not draw "
        "g=b (one point) in facet panel c=p2, g=c (one point) in facet panel c=p2"
    )
    assert messages == [expected]


#: Two facet panels on "c"; panel "p1" draws "a" and drops "b", panel "p2"
#: draws "a" and drops "c" -- each dropped series lives in a different panel.
_FACET_PARTIAL_ACROSS_PANELS_ROWS = pl.DataFrame(
    {
        "v": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "w": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "g": ["a", "a", "b", "a", "a", "c"],
        "c": ["p1", "p1", "p1", "p2", "p2", "p2"],
    }
)


def test_a_faceted_partial_drop_across_two_panels_names_both():
    """Group "b" is dropped in panel p1 and group "c" is dropped in panel
    p2 -- two independent partial drops in two different panels, folded
    into one warning that names each series beside its own panel."""
    _, messages = _render_capturing(
        lambda: fm.Chart(_FACET_PARTIAL_ACROSS_PANELS_ROWS)
        .mark_line()
        .encode(x="v", y="w", color="g", facet="c")
        .to_svg()
    )
    expected = (
        "mark_line: mark_line() joins two or more points per series, so it did not draw "
        "g=b (one point) in facet panel c=p1, g=c (one point) in facet panel c=p2"
    )
    assert messages == [expected]


def test_a_plus_layered_partial_line_names_its_own_layer():
    base = fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})).mark_point().encode(
        x="x", y="y"
    )
    partial_line = fm.Chart(_PARTIAL_GROUPS).mark_line().encode(x="v", y="w", color="g")
    _, messages = _render_capturing((base + partial_line).to_svg)
    expected = (
        "layer 1: mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=b (one point)"
    )
    assert messages == [expected]


def test_an_hconcat_of_two_identical_leaves_warns_once_per_leaf():
    """Each leaf of a composite is its own chart; a drop the two leaves
    happen to report identically is still two facts, one per leaf -- not
    deduplicated into one warning for the pair."""
    d1 = pl.DataFrame({"x": [1.0, 2.0, 1.0], "y": [1.0, 2.0, 1.0], "g": ["a", "a", "b"]})
    c1 = fm.Chart(d1).mark_line().encode(x="x", y="y", color="g")
    _, messages = _render_capturing((c1 | c1).to_svg)
    expected = (
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=b (one point)"
    )
    assert messages == [expected, expected]


def test_an_hconcat_of_two_differing_leaves_names_each_leafs_own_drop():
    d1 = pl.DataFrame({"x": [1.0, 2.0, 1.0], "y": [1.0, 2.0, 1.0], "g": ["a", "a", "b"]})
    d2 = pl.DataFrame({"x": [1.0, 2.0, 1.0], "y": [1.0, 2.0, 1.0], "g": ["a", "a", "c"]})
    c1 = fm.Chart(d1).mark_line().encode(x="x", y="y", color="g")
    c2 = fm.Chart(d2).mark_line().encode(x="x", y="y", color="g")
    _, messages = _render_capturing((c1 | c2).to_svg)
    assert messages == [
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=b (one point)",
        "mark_line: mark_line() joins two or more points per series, "
        "so it did not draw g=c (one point)",
    ]


def test_every_series_drawn_in_a_colored_line_stays_silent():
    all_drawn = pl.DataFrame(
        {
            "v": [1.0, 2.0, 1.0, 2.0],
            "w": [1.0, 2.0, 1.0, 2.0],
            "g": ["a", "a", "b", "b"],
        }
    )
    svg, messages = _render_capturing(
        lambda: fm.Chart(all_drawn).mark_line().encode(x="v", y="w", color="g").to_svg()
    )
    assert len(polylines(svg)) == 2
    assert messages == []
