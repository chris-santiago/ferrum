"""A group, band slot, facet panel or whole layer whose drawn position is null
on every row names itself instead of drawing nothing in silence.

When every row of a legend category, a ``detail``/``shape``/``stroke_dash``
series, a category of a categorical positional axis (its band slot), a facet
panel's share of a layer, or an entire unfaceted layer has a null or
non-finite value on a channel the mark draws its position from, the render
now warns once per layer through ``RenderWarning::GroupsWithNoPosition``,
voiced the way ``RenderWarning::SeriesDropped`` is. The legend keeps the
named group's entry, and the axis keeps the named slot. A row inside a group
or slot that still draws some rows stays the documented, silent per-row null
skip; a group an explicit domain cuts, a primitive mark's or a boxen
category's rows a log scale's off side cuts, a one-point series, and a group a
statistic's own transform warning (``DegenerateFit``, ``IntervalUndefined``)
already names by the same group keep their own, unchanged reports.

A summary statistic keeps the key of a group a null column empties. ``Bin`` (a
histogram) and ``LetterValue`` (a boxen) leave one placeholder row for it, as
``Aggregate`` and ``BoxStats`` do, so a histogram hue group keeps its legend
entry, a facet panel keeps its place on the figure, a boxen category keeps its
band slot, and the census above names each. A kept key keeps its dodge
sub-slot, empty, as every statistic that keeps one leaves it, so the other
groups' marks keep the width and place they have when that group holds a
value. A zero-row frame has no group to keep and still reports that the chart
has no rows to draw. A statistic that bins in a log axis's space sees no value
in a group whose every value is off the axis's side, so a histogram level that
way keeps its legend entry and is named beside the log warning, as the ECDF
kind's is.

A layer is named in the channels its caller wrote. A layer a call built (a
composite mark, a figure function) bound its secondary positional channel
itself, so it names ``x2`` as ``x`` and ``y2`` as ``y``; a layer the caller
placed keeps every channel it wrote. A band slot is named by the column that
holds its categories (the caller's, or a boxen's ``color_field=`` column): the
lowering carries it through the statistic's ``groupby``, except a column named
like one of its outputs, which cannot be carried and reads ``group``.

One figure is the exception. ``catplot`` lowers a horizontal call
(``orient="h"``) to ``x=category, y=value`` plus ``CoordFlip``, so it names the
channels of that lowering: a value the caller bound to ``x`` reads ``y``, as
its explicit-domain and log-scale warnings already read. The strict
expected-failure tests that expect the caller's ``x`` keep that gap visible; a
fix turns them into failures that say to delete their marker.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from typing import Any, Callable, Literal, NamedTuple

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import axis_tick_labels
from tests._svg_rects import data_bar_rects, mark_groups, mark_layer, svg_elements

# The tail every degenerate-fit sentence ends with.
_FIT_DEGRADED = "it degraded to an empty/flat output instead of failing the whole chart"

_TWO_GROUP_ONE_NULL = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, None, None],
        "g": ["a", "a", "c", "c"],
    }
)


def _render_capturing(
    render: Callable[[], str], *, action: Literal["always", "default"] = "always"
) -> tuple[str, list[str]]:
    """Render inside a warning capture: the output and every warning raised.

    ``action`` is the ``warnings`` filter action. ``"always"`` keeps every copy.
    ``"default"`` is Python's own, which shows one text once per location, so
    the sub-layers of one call that raise the same text show a single message:
    what a caller sees without ``-W always``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        out = render()
    return out, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# A group whose drawn position is null on every row is named once, and the
# legend still lists it, on every mark that reads a per-row position.
# ---------------------------------------------------------------------------


def test_a_group_with_no_placed_row_is_named_once():
    chart = fm.Chart(_TWO_GROUP_ONE_NULL).mark_line().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]
    assert ">c<" in svg
    assert ">a<" in svg


@pytest.mark.parametrize("mark_method", ["mark_area", "mark_ribbon"])
def test_mark_area_and_ribbon_name_their_own_null_group(mark_method: str):
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, None, None],
            "g": ["a", "a", "c", "c"],
        }
    )
    chart = getattr(fm.Chart(df), mark_method)().encode(x="x", y="y", color="g")
    if mark_method == "mark_ribbon":
        chart = chart.encode(y2="y")
        expected = (
            "mark_ribbon: every row of g=c has a null or non-finite y and y2, so it draws nothing"
        )
    else:
        expected = "mark_area: every row of g=c has a null or non-finite y, so it draws nothing"
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [expected]
    assert ">c<" in svg


@pytest.mark.parametrize(
    "mark_method,null_channel",
    [
        ("mark_point", "y"),
        ("mark_bar", "y"),
        ("mark_tick", "x"),
        ("mark_text", "y"),
        ("mark_rule", "y"),
    ],
)
def test_every_per_row_mark_names_its_own_null_group(mark_method: str, null_channel: str):
    """Each per-row mark places from its own channel: ``mark_tick`` draws
    from ``x`` alone even with ``y`` also bound, so its null fixture nulls
    ``x``; every other mark here reads ``y``."""
    if null_channel == "x":
        df = pl.DataFrame(
            {"x": [1.0, 2.0, None, None], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "c", "c"]}
        )
    else:
        df = _TWO_GROUP_ONE_NULL
    chart = getattr(fm.Chart(df), mark_method)().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    expected = f"{mark_method}: every row of g=c has a null or non-finite {null_channel}, so it draws nothing"
    assert messages == [expected]
    assert ">c<" in svg


def test_displot_ecdf_hue_names_its_own_null_group():
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, None, None, None],
            "h": ["a", "a", "a", "c", "c", "c"],
        }
    )
    svg, messages = _render_capturing(lambda: fm.displot(df, x="v", kind="ecdf", hue="h").to_svg())
    assert messages == ["displot: every row of h=c has a null or non-finite x, so it draws nothing"]
    assert ">c<" in svg
    assert ">a<" in svg


@pytest.mark.parametrize("group_channel", ["shape", "stroke_dash"])
def test_a_point_mark_names_a_null_group_by_shape_or_stroke_dash(group_channel: str):
    chart = fm.Chart(_TWO_GROUP_ONE_NULL).mark_point().encode(x="x", y="y", **{group_channel: "g"})
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=c has a null or non-finite y, so it draws nothing"
    ]
    assert ">c<" in svg


def test_a_line_mark_names_a_null_group_by_detail():
    chart = fm.Chart(_TWO_GROUP_ONE_NULL).mark_line().encode(x="x", y="y", detail="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_line"])
def test_a_binned_color_bin_with_no_placed_row_is_named(mark_method: str):
    """A binned color legend entry is a group on every mark that draws
    one, read the same way the connected marks already read it, not just
    on ``mark_line``."""
    df = pl.DataFrame({"v": [10.0, 20.0, 90.0, 95.0], "y": [1.0, 2.0, None, None]})
    chart = getattr(fm.Chart(df), mark_method)().encode(x="v", y="y", color=fm.Color("v", bin=True))
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        f"{mark_method}: every row of v=90 has a null or non-finite y, so it draws nothing"
    ]


def test_a_null_group_key_is_named_the_way_a_null_is_rendered():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, None, None], "g": ["a", "a", None, None]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=null has a null or non-finite y, so it draws nothing"
    ]


def test_a_group_null_on_both_axes_names_both_channels():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, None, None], "y": [1.0, 2.0, None, None], "g": ["a", "a", "c", "c"]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=c has a null or non-finite x and y, so it draws nothing"
    ]


def test_a_band_axis_null_drawn_in_the_null_band_stays_silent_same_field():
    """A null category occupies a band exactly when some null-category row
    of the batch carries a present value on the paired channel; the group
    that lands there draws into it and stays silent, never "absent"."""
    df = pl.DataFrame({"cat": ["a", "a", None, None], "val": [1.0, 2.0, 3.0, 4.0]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y="val", color="cat")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == []
    assert ">null<" in svg


@pytest.mark.parametrize("mark_method", ["mark_bar", "mark_point"])
def test_a_band_axis_null_drawn_in_the_null_band_stays_silent_brief_fixture(mark_method: str):
    df = pl.DataFrame(
        {"x": [None, None, "p", "q"], "y": [1.0, 2.0, 3.0, 4.0], "g": ["c", "c", "a", "a"]}
    )
    chart = getattr(fm.Chart(df), mark_method)().encode(x="x:N", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == []
    assert ">null<" in svg


def test_a_single_axis_tick_null_category_draws_in_the_null_band():
    """``mark_tick`` with only ``x`` bound has no paired channel at all, so
    the null band is always occupied -- group ``c``'s null ``x`` draws a
    tick in it rather than being reported absent."""
    df = pl.DataFrame({"x": [1.0, 2.0, None, None], "g": ["a", "a", "c", "c"]})
    chart = fm.Chart(df).mark_tick().encode(x="x:N", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == []
    assert ">null<" in svg


def test_a_band_axis_null_with_no_null_band_is_named_paired_null():
    """Group ``c``'s ``x`` and ``y`` are both null on every row, so no row
    anywhere occupies the null band for it: the band is unoccupied and the
    group is absent, named on both channels."""
    df = pl.DataFrame(
        {"x": ["p", "p", None, None], "y": [1.0, 2.0, None, None], "g": ["a", "a", "c", "c"]}
    )
    chart = fm.Chart(df).mark_bar().encode(x="x:N", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_bar: every row of g=c has a null or non-finite x and y, so it draws nothing"
    ]


def test_a_band_axis_null_with_no_null_band_is_named_declared_domain():
    """An explicit ``BandScale`` domain that omits ``None`` gives the null
    category no band to land in, so group ``c`` is absent even though its
    ``y`` is present."""
    df = pl.DataFrame(
        {"x": ["p", "p", None, None], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "c", "c"]}
    )
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x=fm.X("x", scale=fm.BandScale(domain=["p", "q"])), y="y", color="g")
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_bar: every row of g=c has a null or non-finite x, so it draws nothing"
    ]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
def test_nan_and_inf_in_the_drawn_column_count_as_null(value: float):
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [value, value], "g": ["c", "c"]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


def test_two_field_groups_join_with_semicolons():
    """Each group is named by two fields (``color`` and ``shape``), so its
    own name holds a comma; the group list switches to semicolons."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, None, None, None, None],
            "g": ["a", "a", "c", "c", "d", "d"],
            "k": ["t", "t", "u", "u", "w", "w"],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g", shape="k")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=c, k=u; and of g=d, k=w, has a null or non-finite y, "
        "so they draw nothing"
    ]


def test_a_flipped_chart_names_the_unflipped_channel_the_caller_bound():
    """Under ``CoordFlip``, the caller still wrote ``y=``, so the warning
    names ``y``, not the axis the flip visually draws it on."""
    chart = (
        fm.Chart(_TWO_GROUP_ONE_NULL)
        .mark_point()
        .encode(x="x", y="y", color="g")
        .coord(fm.CoordFlip())
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


# ---------------------------------------------------------------------------
# A whole layer, and a facet panel's share of a layer.
# ---------------------------------------------------------------------------


def test_a_layer_with_no_placed_row_says_so():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [None, None, None]}).cast({"y": pl.Float64})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row has a null or non-finite y, so the layer draws nothing"
    ]


def test_a_one_row_layer_with_a_null_y_and_no_grouping_says_so_too():
    df = pl.DataFrame({"x": [1.0], "y": [None]}).cast({"y": pl.Float64})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row has a null or non-finite y, so the layer draws nothing"
    ]


def test_a_facet_panel_with_no_placed_row_names_the_facet_panel():
    """Panel ``q`` is all-null and has no color grouping of its own; the
    panel phrase moves between "every row" and "has", the one-panel case
    of the several-panel list below, and the sentence turns plural."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, None, None],
            "f": ["p", "p", "q", "q"],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", facet="f")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row in facet panel f=q has a null or non-finite y, so they draw nothing"
    ]


def test_a_no_field_partition_emptied_in_several_panels_joins_them():
    """With no grouping channel at all, two emptied panels join through the
    same list grammar a named group's panel clauses use, rather than
    running the two panel clauses together with no separator."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, None, None, None, None],
            "f": ["p", "p", "q", "q", "r", "r"],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", facet="f")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row in facet panel f=q, and in facet panel f=r, "
        "has a null or non-finite y, so they draw nothing"
    ]


def test_several_emptied_groups_across_facet_panels_join_with_their_panels():
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, None, None, None, None],
            "g": ["a", "a", "c", "c", "d", "d"],
            "f": ["p", "p", "p", "p", "q", "q"],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g", facet="f")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=c in facet panel f=p, and of g=d in facet panel f=q, "
        "has a null or non-finite y, so they draw nothing"
    ]


# ---------------------------------------------------------------------------
# Every composition door warns once per layer or once per leaf.
# ---------------------------------------------------------------------------


def _base_point_chart() -> fm.Chart:
    return (
        fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})).mark_point().encode(x="x", y="y")
    )


def _null_group_line_chart() -> fm.Chart:
    return fm.Chart(_TWO_GROUP_ONE_NULL).mark_line().encode(x="x", y="y", color="g")


def test_a_plus_layer_names_its_own_layer():
    _, messages = _render_capturing((_base_point_chart() + _null_group_line_chart()).to_svg)
    assert messages == [
        "layer 1: mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


def test_an_hconcat_leaf_names_its_own_drop():
    base, null_leaf = _base_point_chart(), _null_group_line_chart()
    _, messages = _render_capturing((base | null_leaf).to_svg)
    assert messages == [
        "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


def test_two_null_leaves_in_one_composite_warn_once_each():
    """Two leaves, each with its own all-null group, warn twice -- each
    leaf reports its own layer, not deduplicated into one warning."""
    _, messages = _render_capturing((_null_group_line_chart() | _null_group_line_chart()).to_svg)
    expected = "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    assert messages == [expected, expected]


def test_a_layer_chart_leaf_names_its_own_drop():
    chart = fm.LayerChart(_base_point_chart(), _null_group_line_chart())
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


def test_a_layer_chart_with_independent_y_names_its_own_layer():
    """With ``resolve={"y": "independent"}``, ``LayerChart`` renders through
    the merged single-panel chart rather than the composite overlay tree
    (its dual-axis route), so the warning carries the same ``layer N:``
    prefix a ``+`` composition does, naming layer 1."""
    chart = fm.LayerChart(
        _base_point_chart(), _null_group_line_chart(), resolve={"y": "independent"}
    )
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "layer 1: mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


def test_interactive_to_html_warns_once_per_render_like_series_dropped():
    """``.interactive().to_html()`` renders both a static fallback and the
    interactive scene, so a render warning reaches the caller once per
    render -- the same count ``SeriesDropped`` gives on this route, not a
    once-per-call count of its own. Fixing the route to render once (a
    structural-wave item) would drop both counts to one together, so the
    comparison still holds then."""
    _, group_messages = _render_capturing(lambda: _null_group_line_chart().interactive().to_html())
    series_dropped_chart = (
        fm.Chart(pl.DataFrame({"x": [1.0, 2.0, 1.0], "y": [1.0, 2.0, 1.0], "g": ["a", "a", "b"]}))
        .mark_line()
        .encode(x="x", y="y", color="g")
    )
    _, series_messages = _render_capturing(lambda: series_dropped_chart.interactive().to_html())
    expected_group = "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    expected_series = "mark_line: mark_line() joins two or more points per series, so it did not draw g=b (one point)"
    assert group_messages == [expected_group] * len(group_messages)
    assert series_messages == [expected_series] * len(series_messages)
    assert len(group_messages) == len(series_messages)
    assert len(group_messages) > 0


def test_chart_to_html_warns_once():
    """``Chart.to_html()`` (no ``.interactive()`` first) renders the scene
    once, so its warning count is one, unlike the double-rendering
    ``.interactive().to_html()`` route above."""
    _, messages = _render_capturing(_null_group_line_chart().to_html)
    assert messages == [
        "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    ]


# ---------------------------------------------------------------------------
# Negative controls: a group that still draws, and every mechanism that
# already owns its own case, keep their own, unchanged reports.
# ---------------------------------------------------------------------------


def test_a_group_that_draws_stays_silent():
    all_drawn = pl.DataFrame(
        {"x": [1.0, 2.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]}
    )
    chart = fm.Chart(all_drawn).mark_line().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == []


def test_a_partial_null_in_a_group_that_still_draws_stays_silent():
    """One null row beside a drawn one in the same group is the documented
    per-row skip, not an all-null group -- it stays silent."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, None, 3.0], "g": ["a", "a", "a"]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == []


@pytest.mark.parametrize(
    "mark_method,encoding",
    [
        ("mark_point", {"x": "x", "y": "y", "color": "g"}),
        ("mark_line", {"x": "x", "y": "y", "color": "g"}),
        ("mark_bar", {"x": "x", "y": "y", "color": "g"}),
        ("mark_area", {"x": "x", "y": "y", "color": "g"}),
        ("mark_tick", {"x": "x", "y": "y", "color": "g"}),
        ("mark_text", {"x": "x", "y": "y", "color": "g"}),
        ("mark_rule", {"x": "x", "y": "y", "color": "g"}),
    ],
)
def test_every_chart_with_no_all_null_group_warns_nothing(mark_method: str, encoding: dict):
    """No mark here has a null row at all, so none of them should start
    warning just because the new feature exists beside them."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]}
    )
    chart = getattr(fm.Chart(df), mark_method)().encode(**encoding)
    _, messages = _render_capturing(chart.to_svg)
    assert messages == []


def test_a_group_cut_by_an_explicit_domain_is_reported_only_by_the_domain_warning():
    """Byte identity for an ordinary chart with no all-null group is the
    golden suite's and the full pytest run's job, not a self-comparison
    inside this file; this test's own job is the domain warning's
    exclusivity, pinned by its full sentence."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 100.0, 100.0], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "c", "c"]}
    )
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale=fm.LinearScale(domain=[0.0, 3.0])), y="y", color="g")
    )
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "x scale: 2 row(s) of 'x' fall outside the scale's explicit domain, so they are not "
        "drawn; widen the domain, or pass clamp=True to draw them at the axis edge"
    ]


def test_a_group_off_side_on_a_log_scale_is_reported_only_by_the_log_warning():
    """On a log y (not a log x), group ``c``'s negative y rows are off-side:
    ``LogScaleRowsLeftOut`` owns the case, pinned by its full sentence."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, -1.0, -2.0], "g": ["a", "a", "c", "c"]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y=fm.Y("y", scale=fm.LogScale()), color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "y scale: 2 row(s) of 'y' are zero or negative, which a LogScale cannot place, so they "
        "are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
    ]


def test_a_one_point_series_is_reported_only_by_series_dropped():
    df = pl.DataFrame({"x": [1.0, 2.0, 1.0], "y": [1.0, 2.0, 1.0], "g": ["a", "a", "b"]})
    chart = fm.Chart(df).mark_line().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_line: mark_line() joins two or more points per series, so it did not draw g=b (one point)"
    ]


def test_a_kde_hue_group_with_no_usable_values_is_reported_by_degenerate_fit_only():
    """``Kde`` already names group ``c`` through ``DegenerateFit`` before
    its NaN-filled grid reaches the draw step, so that existing, more
    specific warning keeps sole ownership of the case: the new census
    defers to it by group name instead of naming the same group again."""
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, None, None, None, None],
            "h": ["a", "a", "a", "a", "c", "c", "c", "c"],
        }
    )
    svg, messages = _render_capturing(lambda: fm.displot(df, x="v", kind="kde", hue="h").to_svg())
    assert messages == [
        "Kde: could not fit group h=c (fewer than 2 usable (non-null/non-NaN) values); "
        f"{_FIT_DEGRADED}"
    ]
    assert ">c<" in svg


def test_arc_is_never_named():
    """``mark_arc`` places nothing by a positional scale, so an all-null
    group under it stays silent and keeps its legend entry."""
    df = pl.DataFrame({"v": [None, None, 1.0, 2.0], "g": ["c", "c", "a", "a"]}).cast(
        {"v": pl.Float64}
    )
    chart = fm.Chart(df).mark_arc().encode(x="v", color="g").coord(fm.CoordPolar(theta="x"))
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == []
    assert ">c<" in svg


def test_a_zero_row_layer_still_warns_no_rows_to_draw_not_a_new_warning():
    df = pl.DataFrame({"x": [1.0], "y": [1.0]}).filter(pl.col("x") > 100)
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "the chart has no rows to draw after its transforms; its panel is drawn empty"
    ]


def test_a_wrong_type_column_refuses_as_it_already_did():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    with pytest.raises(ValueError, match="unsupported dtype"):
        fm.Chart(df).mark_point().encode(x="x", y="y:Q").to_svg()


# ---------------------------------------------------------------------------
# End to end through the public API: log-axis infinities, mixed-channel and
# null-keyed groups, cross-layer ownership, and a fit's row-coverage over
# every layer and name it degrades.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,label",
    [(float("-inf"), "-inf"), (float("inf"), "inf"), (float("nan"), "nan")],
    ids=["-inf", "inf", "nan"],
)
def test_an_infinity_on_a_log_axis_is_left_to_its_own_report(value: float, label: str):
    """On a log y, group ``c``'s own report differs by value: a negative
    value is reported by the log side's own warning, a value outside the
    scale's explicit domain by the domain warning, and only a true NaN (no
    axis report of its own) falls to the census."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, value, value], "g": ["a", "a", "c", "c"]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y=fm.Y("y", scale=fm.LogScale()), color="g")
    _, messages = _render_capturing(chart.to_svg)
    if label == "-inf":
        assert messages == [
            "y scale: 2 row(s) of 'y' are zero or negative, which a LogScale cannot place, so "
            "they are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
        ]
    elif label == "inf":
        assert messages == [
            "y scale: 2 row(s) of 'y' fall outside the scale's explicit domain, so they are not "
            "drawn; widen the domain, or pass clamp=True to draw them at the axis edge"
        ]
    else:
        assert messages == [
            "mark_point: every row of g=c has a null or non-finite y, so it draws nothing"
        ]


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_line"])
def test_a_group_whose_rows_miss_different_channels_is_named(mark_method: str):
    """Group ``a``'s rows each miss a different channel (row 0 misses x,
    row 1 misses y), so no single row of the group has both -- the group is
    absent on both channels, not just the one each individual row lacks."""
    df = pl.DataFrame(
        {"x": [None, 2.0, 3.0, 4.0], "y": [1.0, None, 3.0, 4.0], "g": ["a", "a", "b", "b"]}
    )
    chart = getattr(fm.Chart(df), mark_method)().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        f"{mark_method}: every row of g=a has a null or non-finite x and y, so it draws nothing"
    ]


def test_a_group_null_on_y_everywhere_names_only_y():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [None, None, 3.0, 4.0], "g": ["a", "a", "b", "b"]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=a has a null or non-finite y, so it draws nothing"
    ]


def test_a_null_keyed_group_null_on_every_placement_is_named():
    """Distinct from ``test_a_null_group_key_is_named_the_way_a_null_is_rendered``
    above (which nulls only ``y``): here the null-keyed group's rows are
    null on *every* placement -- both ``x`` and ``y`` -- so the message
    names both channels, not just one."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, None, None], "y": [1.0, 2.0, None, None], "g": ["a", "a", None, None]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=null has a null or non-finite x and y, so it draws nothing"
    ]


def test_an_all_null_layer_bound_to_a_null_color_is_named():
    """``g`` is cast to ``Utf8`` so its all-null column is a real
    categorical (``pl.Null``, which a bare ``[None, ...]`` list types as,
    forms no partition at all and would fall through to the whole-layer
    sentence instead -- not what this test means to pin)."""
    df = pl.DataFrame(
        {
            "x": [None, None, None, None],
            "y": [None, None, None, None],
            "g": [None, None, None, None],
        }
    ).cast({"x": pl.Float64, "y": pl.Float64, "g": pl.Utf8})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=null has a null or non-finite x and y, so it draws nothing"
    ]


def _smooth_and_aggregate_line_charts() -> tuple["fm.Chart", "fm.Chart"]:
    df_a = pl.DataFrame(
        {"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, None, None], "g": ["a", "a", "a", "a"]}
    )
    df_b = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [None, None, None], "g": ["a", "a", "a"]})
    layer_a = (
        fm.Chart(df_a)
        .transform(fm.Smooth(x="x", y="y", groupby=["g"]))
        .mark_line()
        .encode(x="x", y="y", color="g")
    )
    layer_b = fm.Chart(df_b).mark_line().encode(x="x", y=fm.Y("y", aggregate="mean"), color="g")
    return layer_a, layer_b


def _kde_and_ecdf_charts() -> tuple["fm.Chart", "fm.Chart"]:
    df_kde = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, None, None, None, None],
            "h": ["a", "a", "a", "a", "c", "c", "c", "c"],
        }
    )
    df_ecdf = pl.DataFrame(
        {
            "v": [5.0, 6.0, 7.0, 8.0, None, None, None, None],
            "h": ["a", "a", "a", "a", "c", "c", "c", "c"],
        }
    )
    return fm.displot(df_kde, x="v", kind="kde", hue="h"), fm.displot(
        df_ecdf, x="v", kind="ecdf", hue="h"
    )


@pytest.mark.parametrize(
    "build,expected",
    [
        (
            _smooth_and_aggregate_line_charts,
            [
                "Smooth: could not fit group g=a (the loess fit could not be computed for this data); "
                f"{_FIT_DEGRADED}",
                "layer 1: mark_line: every row of g=a has a null or non-finite y, so it draws nothing",
            ],
        ),
        (
            _kde_and_ecdf_charts,
            [
                f"Kde: could not fit group h=c (fewer than 2 usable (non-null/non-NaN) values); {_FIT_DEGRADED}",
                "displot: every row of h=c has a null or non-finite x, so it draws nothing",
            ],
        ),
    ],
    ids=["smooth_and_aggregate_line", "kde_and_ecdf"],
)
def test_another_layers_fit_warning_never_silences_this_layers_group(build, expected):
    """Layer 0's own fit degrades a group; layer 1 is an unrelated chart
    whose own, independently all-null group happens to share the same
    name. The two layers' own rows never mix, so both warn."""
    layer_0, layer_1 = build()
    _, messages = _render_capturing((layer_0 + layer_1).to_svg)
    assert messages == expected


def test_a_user_written_fit_step_owns_its_degenerate_group():
    """``Smooth``'s groupby key ``gi`` is entirely null, so it falls back to
    one ungrouped fit (``group=None``) -- and that one fit cannot be
    computed at all. The ribbon reads the same frame and names its own
    group by the same, all-null ``gi`` column (``gi=null``): a name the
    fit's own ``group=None`` report does not literally match, but whose
    rows the fit's one ungrouped run fully covers, so the ribbon's own
    census report is owned and never fires."""
    df = pl.DataFrame({"x": [1.0], "y": [3.0], "gi": [None]}).cast({"gi": pl.Utf8})
    chart = (
        fm.Chart(df)
        .transform(fm.Smooth(x="x", y="y", method="lm", groupby=["gi"]))
        .mark_ribbon()
        .encode(x="x", y="y", y2="y", color="gi")
    )
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "Smooth: every row has a null in its groupby key (gi), so the data could not be split "
        "into groups; the whole batch was used as one ungrouped computation instead",
        "Smooth: could not fit the data (fewer than 2 usable (non-null/non-NaN) x/y points); "
        f"{_FIT_DEGRADED}",
    ]


def test_an_ungrouped_fallback_fit_owns_its_null_keyed_group():
    """Same null-groupby-key fallback as above, but the line still fits (``n=2``
    leaves no residual degrees of freedom, so only the ``ci`` band is
    undefined): ``IntervalUndefined``'s ``group=None`` still covers the
    ribbon's ``gi=null`` band, and the census never names it again."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 5.0], "gi": [None, None]}).cast({"gi": pl.Utf8})
    chart = (
        fm.Chart(df)
        .transform(fm.Smooth(x="x", y="y", method="lm", ci=0.95, groupby=["gi"]))
        .mark_ribbon()
        .encode(x="x", y="ci_lower", y2="ci_upper", color="gi")
    )
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "Smooth: every row has a null in its groupby key (gi), so the data could not be split "
        "into groups; the whole batch was used as one ungrouped computation instead",
        "Smooth: the ci=0.95 band is undefined (2 points leave no residual degrees of freedom "
        "for a 2-parameter fit; the band needs at least 3); the fit line is drawn without it",
    ]


def test_a_fit_owns_every_layer_drawn_from_its_degraded_groups():
    """``show_metrics=True`` adds a second layer (the metrics text), which
    carries no group field of its own, drawn across a facet panel whose
    *every* fitted group is degenerate. Panel ``sp=p1``'s own ``gi`` is
    entirely null (an ungrouped fallback); panel ``sp=p2``'s two real
    groups each fit a cubic from two points, degenerate in both. With
    every group it draws degenerate, an unpartitioned metrics layer has
    no fitting group left to borrow a report from, so the row-coverage
    fix must own its rows directly -- a weaker, name-matching ownership
    (the previous cycle's) leaves the metrics layer's own census report
    unowned, since it names no group at all."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0],
            "sp": ["p1"] * 4 + ["p2"] * 4,
            "gi": [None, None, None, None, 1, 1, 2, 2],
        }
    )
    chart = (
        fm.Chart(df)
        .encode(x="x", y="y")
        .mark_smooth(method="cubic", show_metrics=True, ci=0.95, groupby="gi")
        .facet(col="sp")
    )
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "Smooth: the ci=0.95 band in facet panel sp=p1 is undefined (4 points leave no "
        "residual degrees of freedom for a 4-parameter fit; the band needs at least 5); "
        "the fit line is drawn without it",
        "Smooth: could not fit group gi=1 in facet panel sp=p2 (the poly(degree=3) fit "
        f"could not be computed for this data); {_FIT_DEGRADED}",
        "Smooth: could not fit group gi=2 in facet panel sp=p2 (the poly(degree=3) fit "
        f"could not be computed for this data); {_FIT_DEGRADED}",
        "Smooth: every row in facet panel sp=p1 has a null in its groupby key (gi), so "
        "that panel's rows could not be split into groups; in that panel, the whole "
        "batch was used as one ungrouped computation instead",
    ]


# ---------------------------------------------------------------------------
# A band slot: a category of a categorical positional axis. When every row of
# one has no drawable position the axis still shows it, empty, so the render
# names it once per layer in the channels the caller wrote. A layer a call
# built names a secondary channel under its primary, so every sub-layer of
# one call raises the same text.
# ---------------------------------------------------------------------------

# Categories ``a`` and ``b`` always hold drawable values; ``z`` holds either.
_VALUES_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
_VALUES_B = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
_DRAWN_Z: list[float | None] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_EMPTY_Z: list[float | None] = [None] * 6

# A value that places nowhere: a null, a NaN, or an infinity on a linear axis.
_UNDRAWABLE = [
    pytest.param(None, id="null"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
]


def _slot_frame(z_values: list[float | None]) -> pl.DataFrame:
    """Categories ``a``, ``b`` and ``z`` of six rows each, valued on ``w``.

    ``g`` alternates two hue levels, so each (category, hue) cell keeps three
    rows and every summary statistic has something to summarise; ``w2`` is
    ``w`` plus one, for marks that span two values.
    """
    values = _VALUES_A + _VALUES_B + z_values
    return pl.DataFrame(
        {
            "c": ["a"] * 6 + ["b"] * 6 + ["z"] * 6,
            "g": ["p", "q"] * 9,
            "w": values,
            "w2": [None if v is None else v + 1.0 for v in values],
        },
        schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64, "w2": pl.Float64},
    )


def _small_slot_frame(
    z_values: list[float | None], z_values2: list[float | None] | None = None
) -> pl.DataFrame:
    """Categories ``a`` and ``b`` of two drawable rows each, then one ``z`` row
    per value of ``z_values``.

    ``w2`` is ``w`` plus one, as in ``_slot_frame``, unless ``z_values2`` sets
    the ``z`` rows' own.
    """
    if z_values2 is None:
        z_values2 = [None if v is None else v + 1.0 for v in z_values]
    return pl.DataFrame(
        {
            "c": ["a", "a", "b", "b"] + ["z"] * len(z_values),
            "w": [1.0, 2.0, 3.0, 4.0] + z_values,
            "w2": [2.0, 3.0, 4.0, 5.0] + z_values2,
        },
        schema={"c": pl.Utf8, "w": pl.Float64, "w2": pl.Float64},
    )


def _slot_message(
    entry: str,
    channels: str = "y",
    category: str = "z",
    column: str = "c",
    *,
    panel: str | None = None,
) -> str:
    """The text that names one unit that draws nothing, ``<column>=<category>`` (a
    band slot or a hue group), on layer ``entry``; in facet panel ``panel`` when given."""
    where = "" if panel is None else f" in facet panel {panel}"
    return (
        f"{entry}: every row of {column}={category}{where} has a null or non-finite {channels}, "
        "so it draws nothing"
    )


def _panel_message(entry: str, channels: str = "y", panel: str = "f=q") -> str:
    """The text that names facet panel ``panel``, none of whose rows of layer ``entry``
    has a position."""
    return (
        f"{entry}: every row in facet panel {panel} has a null or non-finite {channels}, "
        "so they draw nothing"
    )


def _layer_message(entry: str, channels: str = "y") -> str:
    """The text that names layer ``entry``, none of whose rows has a position."""
    return f"{entry}: every row has a null or non-finite {channels}, so the layer draws nothing"


def _census(messages: list[str]) -> list[str]:
    """The null-position messages among ``messages`` (each carries ``null or
    non-finite``), leaving out an unrelated notice a figure raises beside them."""
    return [m for m in messages if "null or non-finite" in m]


# The reasons a density statistic gives for a group it cannot estimate (the sentence's tail,
# ``_FIT_DEGRADED``, is defined at the top of the module).
FIT_FEW = "fewer than 2 usable (non-null/non-NaN) values"
FIT_NONE = "no usable (non-null/non-NaN) values"
FIT_BANDWIDTH = "the bandwidth resolved to a non-positive or non-finite value"


def _fit_message(statistic: str, group: str, reason: str, *, panel: str | None = None) -> str:
    """The ``DegenerateFit`` sentence for one group ``statistic`` could not estimate, in the
    facet panel ``panel`` when given."""
    where = "" if panel is None else f" in facet panel {panel}"
    return f"{statistic}: could not fit group {group}{where} ({reason}); {_FIT_DEGRADED}"


def _panel_fit_message(statistic: str, reason: str, panel: str = "f=q") -> str:
    """The ``DegenerateFit`` sentence for a facet panel ``statistic`` could not estimate."""
    return f"{statistic}: could not fit the data in facet panel {panel} ({reason}); {_FIT_DEGRADED}"


def _whole_fit_message(statistic: str, reason: str) -> str:
    """The ``DegenerateFit`` sentence for a whole call (no group, no facet panel) ``statistic``
    could not estimate."""
    return f"{statistic}: could not fit the data ({reason}); {_FIT_DEGRADED}"


def _fit_warnings(messages: list[str], statistic: str) -> list[str]:
    """The ``DegenerateFit`` messages of ``statistic`` among ``messages``."""
    return [m for m in messages if m.startswith(f"{statistic}: could not fit")]


def _statistic_reports(mark_method: str) -> set[str]:
    """What an emptied category ``z`` adds beside the census report of ``mark_method``: a violin's
    body is a density statistic, which names the category itself; every other mark adds nothing."""
    return {_fit_message("Violin", "c=z", FIT_FEW)} if mark_method == "mark_violin" else set()


def _given(**channels: Any) -> dict[str, Any]:
    """The channels of ``channels`` that are bound, so one left unbound is left out of a call
    instead of being passed as ``None``."""
    return {name: value for name, value in channels.items() if value is not None}


def _slot_chart(df: pl.DataFrame, mark_method: str, encoding: dict) -> fm.Chart:
    return getattr(fm.Chart(df), mark_method)().encode(**encoding)


def _catplot_chart(
    df: pl.DataFrame, kind: str, hue: str | None, *, facet: str | None = None
) -> fm.Chart:
    return fm.catplot(df, x="c", y="w", kind=kind, **_given(hue=hue, col=facet))


# Marks the caller places layer by layer, with the channels each names for an
# empty slot. A mark that binds ``y2`` itself names both: the caller wrote them.
_PLACED_SLOT_MARKS = [
    pytest.param("mark_point", {"x": "c", "y": "w"}, "y", id="point"),
    pytest.param("mark_bar", {"x": "c", "y": "w"}, "y", id="bar"),
    pytest.param("mark_tick", {"x": "c", "y": "w"}, "y", id="tick"),
    pytest.param("mark_line", {"x": "c", "y": "w"}, "y", id="line"),
    pytest.param("mark_area", {"x": "c", "y": "w"}, "y", id="area"),
    pytest.param("mark_text", {"x": "c", "y": "w", "text": "c"}, "y", id="text"),
    pytest.param("mark_label", {"x": "c", "y": "w", "text": "c"}, "y", id="label"),
    pytest.param("mark_rule", {"x": "c", "y": "w", "y2": "w2"}, "y and y2", id="rule"),
    pytest.param("mark_rect", {"x": "c", "y": "w", "y2": "w2"}, "y and y2", id="rect"),
    pytest.param("mark_ribbon", {"x": "c", "y": "w", "y2": "w2"}, "y and y2", id="ribbon"),
]

# Marks a call builds out of several layers of its own (a box and its
# whiskers, a mean bar's summary and interval): the caller bound ``y`` alone.
_CALL_BUILT_SLOT_MARKS = [
    pytest.param("mark_boxplot", {"x": "c", "y": "w"}, id="boxplot"),
    pytest.param("mark_violin", {"x": "c", "y": "w"}, id="violin_inner_box"),
    pytest.param("mark_errorbar", {"x": "c", "y": "w"}, id="errorbar"),
    pytest.param("mark_errorband", {"x": "c", "y": "w"}, id="errorband"),
    pytest.param("mark_bar", {"x": "c", "y": fm.Y("w", aggregate="mean")}, id="mean_bar"),
    pytest.param("mark_swarm", {"x": "c", "y": "w"}, id="swarm"),
]

_CATPLOT_KINDS = ["box", "bar", "strip", "point", "violin", "swarm"]

# The tests it marks expect the caller's channel from a horizontal catplot and
# fail as built: the lowering writes ``x=category, y=value`` plus ``CoordFlip``,
# and the render names the channels it was given. Strict, so when a fix makes
# them pass they fail, and the failure says to delete this marker.
_HORIZONTAL_CATPLOT_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        'catplot lowers orient="h" to x=category, y=value plus CoordFlip, so the render '
        "names the lowering's channels: a value the caller bound to x reads y"
    ),
)


@pytest.mark.parametrize("mark_method,encoding,channels", _PLACED_SLOT_MARKS)
def test_an_ungrouped_band_slot_with_no_placed_row_is_named(
    mark_method: str, encoding: dict, channels: str
):
    """With no grouping channel the layer's one group still draws, so the slot
    ``z`` is the only unit that draws nothing; it is named by its category, in
    its own mark's voice, and the axis still shows it."""
    svg, messages = _render_capturing(
        _slot_chart(_slot_frame(_EMPTY_Z), mark_method, encoding).to_svg
    )
    assert messages == [_slot_message(mark_method, channels)]
    assert ">z<" in svg


def test_a_span_rule_has_no_band_slot_to_name():
    """``mark_rule(x, y)`` with no ``x2`` or ``y2`` is a horizontal line at
    ``y`` across the panel and has no band placement for a slot to key on, so
    the emptied slot is not named although ``z`` is still on the axis. Bound
    to ``y2`` as well it is a vertical segment per row, placed in its category,
    and the same slot is named. This records what is built; the segment form
    is its control."""
    df = _slot_frame(_EMPTY_Z)
    svg, span = _render_capturing(fm.Chart(df).mark_rule().encode(x="c", y="w").to_svg)
    assert span == []
    assert ">z<" in svg
    _, segment = _render_capturing(fm.Chart(df).mark_rule().encode(x="c", y="w", y2="w2").to_svg)
    assert segment == [_slot_message("mark_rule", "y and y2")]


@pytest.mark.parametrize("mark_method,encoding", _CALL_BUILT_SLOT_MARKS)
def test_a_composites_empty_slot_is_named_in_the_callers_channel(mark_method: str, encoding: dict):
    """A composite lays several layers over one slot, some spanning ``y`` to
    ``y2``. The caller bound ``y`` alone, so every layer names ``y`` and the
    default warning filter shows the one text once."""
    chart = _slot_chart(_slot_frame(_EMPTY_Z), mark_method, encoding)
    expected = _slot_message(mark_method)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(shown) == sorted([expected, *_statistic_reports(mark_method)])
    assert ">z<" in svg
    _, every = _render_capturing(chart.to_svg)
    assert set(every) == {expected, *_statistic_reports(mark_method)}


@pytest.mark.parametrize("mark_method", ["mark_boxplot", "mark_errorbar", "mark_violin"])
def test_a_grouped_composites_empty_slot_is_named_in_the_callers_channel(mark_method: str):
    """With a hue splitting the categories every group still draws in ``a`` and
    ``b``, so only the slot ``z`` draws nothing; the composite names it in
    ``y`` as an ungrouped one does, and the same chart with ``z`` drawn is
    silent."""

    def census(z_values: list[float | None]) -> list[str]:
        chart = getattr(fm.Chart(_slot_frame(z_values)), mark_method)().encode(
            x="c", y="w", color="g"
        )
        return _census(_render_capturing(chart.to_svg, action="default")[1])

    assert census(_DRAWN_Z) == []
    assert census(_EMPTY_Z) == [_slot_message(mark_method)]


@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("kind", _CATPLOT_KINDS)
def test_a_catplot_empty_slot_is_named(kind: str, hue: str | None):
    """A figure function names the slot in its own voice and the caller's
    channel, with or without a hue splitting the categories."""
    chart = _catplot_chart(_slot_frame(_EMPTY_Z), kind, hue)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _census(shown) == [_slot_message("catplot")]
    assert ">z<" in svg


@_HORIZONTAL_CATPLOT_XFAIL
@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("kind", _CATPLOT_KINDS)
def test_a_horizontal_catplot_names_the_callers_channel_for_an_empty_slot(
    kind: str, hue: str | None
):
    """The caller bound the values to ``x`` and the categories to ``y``, so the
    empty slot ``z`` should read ``x``, as a horizontal composite mark's does.
    The render names ``y``, the lowering's own channel."""
    chart = fm.catplot(_slot_frame(_EMPTY_Z), y="c", x="w", kind=kind, hue=hue, orient="h")
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _census(shown) == [_slot_message("catplot", "x")]
    assert ">z<" in svg


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_bar"])
def test_a_slot_already_named_by_its_group_is_named_once(mark_method: str):
    """``color`` bound to the category column makes the slot's rows a legend
    group the census already names; the slot adds no second name."""
    chart = getattr(fm.Chart(_slot_frame(_EMPTY_Z)), mark_method)().encode(x="c", y="w", color="c")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message(mark_method)]


def test_a_slot_and_a_group_are_each_named_once():
    """Group ``p`` lies wholly inside slot ``z``, and ``z`` also holds a row of
    group ``q``, which draws elsewhere: each unit that draws nothing is named,
    once, in one sentence."""
    df = pl.DataFrame(
        {
            "c": ["a", "a", "b", "b", "z", "z", "z"],
            "g": ["q", "q", "q", "q", "p", "p", "q"],
            "w": [1.0, 2.0, 3.0, 4.0, None, None, None],
        },
        schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64},
    )
    chart = fm.Chart(df).mark_point().encode(x="c", y="w", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=p, and of c=z, has a null or non-finite y, so they draw nothing"
    ]
    assert ">z<" in svg


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_bar"])
def test_a_slot_empty_for_every_hue_group_is_named_and_no_group_is(mark_method: str):
    """Every hue group draws in categories ``a`` and ``b``, so none is itself
    unplaced; only the slot ``z`` is."""
    chart = getattr(fm.Chart(_slot_frame(_EMPTY_Z)), mark_method)().encode(x="c", y="w", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message(mark_method)]


def test_groups_that_fill_a_slot_between_them_leave_it_unnamed():
    """Groups ``p`` and ``q`` hold every row of slot ``z`` between them and are
    named; the slot is those same rows, so it is not named a second time."""
    df = pl.DataFrame(
        {
            "c": ["a", "a", "b", "b", "z", "z"],
            "g": ["a1", "a1", "b1", "b1", "p", "q"],
            "w": [1.0, 2.0, 3.0, 4.0, None, None],
        },
        schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64},
    )
    chart = fm.Chart(df).mark_point().encode(x="c", y="w", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=p, and of g=q, has a null or non-finite y, so they draw nothing"
    ]


@pytest.mark.parametrize(
    "mark_method,mark_kwargs",
    [
        pytest.param("mark_point", {}, id="point"),
        pytest.param("mark_bar", {}, id="bar"),
        pytest.param("mark_tick", {}, id="tick"),
        pytest.param("mark_line", {}, id="line"),
        pytest.param("mark_area", {}, id="area"),
        pytest.param("mark_boxplot", {"horizontal": True}, id="boxplot"),
        pytest.param("mark_violin", {"horizontal": True}, id="violin"),
        pytest.param("mark_swarm", {"horizontal": True}, id="swarm"),
    ],
)
def test_a_horizontal_band_slot_is_named(mark_method: str, mark_kwargs: dict):
    """The category on ``y`` and the values on ``x``: the caller bound the
    values to ``x``, so that is the channel named, not ``y``."""
    chart = getattr(fm.Chart(_slot_frame(_EMPTY_Z)), mark_method)(**mark_kwargs).encode(
        y="c", x="w"
    )
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(shown) == sorted(
        [_slot_message(mark_method, "x"), *_statistic_reports(mark_method)]
    )
    assert ">z<" in svg


def _faceted_slot_frame(
    empty_in: list[str],
    *,
    blank_in: tuple[str, ...] = (),
    panels: tuple[str, ...] = ("p", "q"),
) -> pl.DataFrame:
    """Facet panels, each holding categories ``a``, ``b`` and ``z`` of two rows.

    ``z`` is null in the panels ``empty_in`` names. Every row is null in the
    panels ``blank_in`` names, which leaves such a panel with no placed row at
    all. ``panels`` names the panels; two by default.
    """
    rows = [
        (
            panel,
            category,
            None if panel in blank_in or (category == "z" and panel in empty_in) else value,
        )
        for panel in panels
        for category, values in (("a", (1.0, 2.0)), ("b", (3.0, 4.0)), ("z", (5.0, 6.0)))
        for value in values
    ]
    return pl.DataFrame(rows, schema={"f": pl.Utf8, "c": pl.Utf8, "w": pl.Float64}, orient="row")


def _with_facet_rows(df: pl.DataFrame) -> pl.DataFrame:
    """``df`` with a facet row field ``r``: ``r1`` for panel ``p``, ``r2`` for the others."""
    return df.with_columns(r=pl.when(pl.col("f") == "p").then(pl.lit("r1")).otherwise(pl.lit("r2")))


def _emptied_group_by_panel_frame() -> pl.DataFrame:
    """Two facet panels of categories ``a`` and ``b``, each split into hue
    groups ``u`` (drawn) and ``v`` (null throughout)."""
    return pl.DataFrame(
        {
            "f": ["p"] * 4 + ["q"] * 4,
            "c": ["a", "a", "b", "b"] * 2,
            "g": ["u", "u", "v", "v"] * 2,
            "w": [1.0, 2.0, None, None, 3.0, 4.0, None, None],
        },
        schema={"f": pl.Utf8, "c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64},
    )


def _faceted_point_chart(df: pl.DataFrame, **facet: str) -> fm.Chart:
    return fm.Chart(df).mark_point().encode(x="c", y="w").facet(**facet)


def _list_message(entry: str, units: str) -> str:
    """The sentence a layer reads when several units draw nothing: ``units`` is
    their list, each in the shape it reads in (a bare panel clause, or ``of`` and
    a name)."""
    return f"{entry}: every row {units}, has a null or non-finite y, so they draw nothing"


@pytest.mark.parametrize(
    "empty_in,expected",
    [
        pytest.param(
            ["q"],
            "mark_point: every row of c=z in facet panel f=q has a null or non-finite y, "
            "so it draws nothing",
            id="one_panel",
        ),
        pytest.param(
            ["p", "q"],
            "mark_point: every row of c=z in facet panel f=p, and of c=z in facet panel f=q, "
            "has a null or non-finite y, so they draw nothing",
            id="both_panels",
        ),
    ],
)
def test_a_band_slot_names_its_facet_panel(empty_in: list[str], expected: str):
    """The slot is named in the panel where it is empty and not in the panel
    where the same category draws; empty in both, it is one warning that names
    each panel."""
    chart = fm.Chart(_faceted_slot_frame(empty_in)).mark_point().encode(x="c", y="w", facet="f")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [expected]
    assert ">z<" in svg


@pytest.mark.parametrize(
    "build,entry,units",
    [
        pytest.param(
            lambda: _faceted_point_chart(_faceted_slot_frame(["q"], blank_in=("p",)), col="f"),
            "mark_point",
            "in facet panel f=p, and of c=z in facet panel f=q",
            id="bare_first",
        ),
        pytest.param(
            lambda: _faceted_point_chart(_faceted_slot_frame(["p"], blank_in=("q",)), col="f"),
            "mark_point",
            "of c=z in facet panel f=p, and in facet panel f=q",
            id="named_first",
        ),
        pytest.param(
            lambda: _faceted_point_chart(
                _faceted_slot_frame(["q"], blank_in=("p", "r"), panels=("p", "q", "r")), col="f"
            ),
            "mark_point",
            "in facet panel f=p, of c=z in facet panel f=q, and in facet panel f=r",
            id="three_panels",
        ),
        pytest.param(
            lambda: _faceted_point_chart(
                _with_facet_rows(_faceted_slot_frame(["q"], blank_in=("p",))), col="f", row="r"
            ),
            "mark_point",
            "in facet panel f=p, r=r1; and of c=z in facet panel f=q, r=r2",
            id="row_and_column",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_faceted_slot_frame(["q"], blank_in=("p",)))
                .mark_boxplot()
                .encode(x="c", y="w")
                .facet(col="f")
            ),
            "mark_boxplot",
            "in facet panel f=p, and of c=z in facet panel f=q",
            id="boxplot",
        ),
        pytest.param(
            lambda: fm.catplot(
                _faceted_slot_frame(["q"], blank_in=("p",)), x="c", y="w", kind="strip", col="f"
            ),
            "catplot",
            "in facet panel f=p, and of c=z in facet panel f=q",
            id="catplot_strip",
        ),
    ],
)
def test_a_wholly_empty_panel_beside_an_empty_slot_reads_as_one_sentence(
    build: Callable[[], fm.Chart], entry: str, units: str
):
    """A faceted layer with no grouping channel reports a panel with no placed
    row at all as a bare panel clause and a slot of another panel by name. Both
    land in one sentence, each in its own shape: the bare clause with no
    ``of`` and no doubled space, the slot as ``of c=z in facet panel ...``,
    in panel order."""
    chart = build()
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _census(shown) == [_list_message(entry, units)]
    assert ">z<" in svg
    _, every = _render_capturing(chart.to_svg)
    assert set(_census(every)) == {_list_message(entry, units)}


@pytest.mark.parametrize(
    "build,units",
    [
        pytest.param(
            lambda: _faceted_point_chart(
                _faceted_slot_frame([], blank_in=("p", "q", "r"), panels=("p", "q", "r")), col="f"
            ),
            "in facet panel f=p, in facet panel f=q, and in facet panel f=r",
            id="three_bare_panels",
        ),
        pytest.param(
            lambda: _faceted_point_chart(
                _with_facet_rows(
                    _faceted_slot_frame([], blank_in=("p", "q"), panels=("p", "q", "s"))
                ),
                col="f",
                row="r",
            ),
            "in facet panel f=p, r=r1; and in facet panel f=q, r=r2",
            id="bare_panels_with_a_row_field",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_with_facet_rows(_emptied_group_by_panel_frame()))
                .mark_point()
                .encode(x="c", y="w", color="g")
                .facet(col="f", row="r")
            ),
            "of g=v in facet panel f=p, r=r1; and of g=v in facet panel f=q, r=r2",
            id="named_groups_with_a_row_field",
        ),
    ],
)
def test_a_list_of_units_of_one_shape_keeps_its_wording(build: Callable[[], fm.Chart], units: str):
    """Entries all of one shape read as they did before a wholly emptied panel
    could sit beside a named slot: bare panel clauses join as ``in facet panel
    ...`` each, named groups as ``of g=... in facet panel ...`` each, and a
    comma inside an entry (a facet row field) switches the list to semicolons.
    The one-panel and two-panel lists are pinned by
    ``test_a_facet_panel_with_no_placed_row_names_the_facet_panel`` and
    ``test_a_no_field_partition_emptied_in_several_panels_joins_them``."""
    _, shown = _render_capturing(build().to_svg, action="default")
    assert _census(shown) == [_list_message("mark_point", units)]


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_bar", "mark_line"])
@pytest.mark.parametrize("value", _UNDRAWABLE)
def test_a_slot_that_draws_stays_silent(mark_method: str, value: float | None):
    """One drawn row keeps its slot drawing; the rest are the documented
    per-row skip. The same chart with that row undrawable too is named, so the
    silence is the slot drawing and not a chart the census cannot see."""
    encoding = {"x": "c", "y": "w"}
    drawing = _slot_frame([value, 5.0, value, value, value, value])
    _, messages = _render_capturing(_slot_chart(drawing, mark_method, encoding).to_svg)
    assert messages == []
    emptied = _slot_frame([value] * 6)
    _, messages = _render_capturing(_slot_chart(emptied, mark_method, encoding).to_svg)
    assert messages == [_slot_message(mark_method)]


@pytest.mark.parametrize("value", _UNDRAWABLE)
def test_a_slot_of_null_or_non_finite_values_is_named(value: float | None):
    """A null, a NaN and either infinity all leave a row unplaced, so a slot
    made of any of them is named; a mix of them is one slot."""
    chart = fm.Chart(_slot_frame([value] * 6)).mark_point().encode(x="c", y="w")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message("mark_point")]


def test_a_slot_of_mixed_null_and_non_finite_values_is_named():
    mixed = [None, float("nan"), float("inf"), float("-inf"), None, float("nan")]
    chart = fm.Chart(_slot_frame(mixed)).mark_point().encode(x="c", y="w")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message("mark_point")]


def test_a_slot_of_one_row_is_named():
    chart = fm.Chart(_small_slot_frame([None])).mark_point().encode(x="c", y="w")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message("mark_point")]
    assert ">z<" in svg


@pytest.mark.parametrize("mark_method", ["mark_bar", "mark_point"])
def test_a_dodged_sub_slot_stays_silent(mark_method: str):
    """Only the cell ``z`` x ``p`` is null; ``z`` still draws group ``q`` and
    ``p`` still draws elsewhere, so no unit is empty. With both ``z`` rows null
    the slot is empty for every group and is named."""
    df = pl.DataFrame(
        {
            "c": ["a", "a", "b", "b", "z", "z"],
            "g": ["p", "q", "p", "q", "p", "q"],
            "w": [1.0, 2.0, 3.0, 4.0, None, 5.0],
        },
        schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64},
    )

    def render(frame: pl.DataFrame) -> list[str]:
        chart = getattr(fm.Chart(frame), mark_method)(position=fm.Dodge()).encode(
            x="c", y="w", color="g"
        )
        return _render_capturing(chart.to_svg)[1]

    assert render(df) == []
    emptied = df.with_columns(
        pl.when(pl.col("c") == "z").then(None).otherwise(pl.col("w")).alias("w")
    )
    assert render(emptied) == [_slot_message(mark_method)]


def test_a_slot_cut_by_a_domain_keeps_its_census_report():
    """Rows an explicit y domain cuts are reported by that domain's own
    warning and nothing else; null rows in the same slot are the slot's."""

    def render(z_values: list[float | None]) -> list[str]:
        y = fm.Y("w", scale=fm.LinearScale(domain=[0.0, 10.0]))
        chart = fm.Chart(_small_slot_frame(z_values)).mark_point().encode(x="c", y=y)
        return _render_capturing(chart.to_svg)[1]

    assert render([90.0, 95.0]) == [
        "y scale: 2 row(s) of 'w' fall outside the scale's explicit domain, so they are not "
        "drawn; widen the domain, or pass clamp=True to draw them at the axis edge"
    ]
    assert render([None, None]) == [_slot_message("mark_point")]


def test_a_slot_off_the_log_side_keeps_its_report():
    """Rows a log scale cannot place are reported by its own warning and
    nothing else; null rows in the same slot are the slot's."""

    def render(z_values: list[float | None]) -> list[str]:
        y = fm.Y("w", scale=fm.LogScale())
        chart = fm.Chart(_small_slot_frame(z_values)).mark_point().encode(x="c", y=y)
        return _render_capturing(chart.to_svg)[1]

    assert render([-1.0, -2.0]) == [
        "y scale: 2 row(s) of 'w' are zero or negative, which a LogScale cannot place, so they "
        "are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
    ]
    assert render([None, None]) == [_slot_message("mark_point")]


def test_a_category_the_explicit_domain_omits_is_reported_by_the_domain_alone():
    """A band scale whose domain leaves ``z`` out has no slot for it: the
    domain warning reports its rows, null or not. Add ``z`` to the domain and
    the slot is on the axis, empty, and is named."""

    def render(domain: list[str]) -> list[str]:
        x = fm.X("c", scale=fm.BandScale(domain=domain))
        chart = fm.Chart(_small_slot_frame([None, None])).mark_point().encode(x=x, y="w")
        return _render_capturing(chart.to_svg)[1]

    assert render(["a", "b"]) == [
        "x scale: 2 row(s) of 'c' name a category the scale's explicit domain leaves out, so "
        "they are not drawn; add it to the domain, or drop domain= to draw every category"
    ]
    assert render(["a", "b", "z"]) == [_slot_message("mark_point")]


def test_a_null_category_that_draws_is_not_named_beside_an_empty_slot():
    """A null category occupies a band when a row of it carries a value; its
    rows draw there and stay silent. A real empty category beside it is still
    named, and only it."""
    only_null = pl.DataFrame(
        {"c": [None, None, "a", "a"], "w": [1.0, 2.0, 3.0, 4.0]},
        schema={"c": pl.Utf8, "w": pl.Float64},
    )
    svg, messages = _render_capturing(
        fm.Chart(only_null).mark_point().encode(x="c:N", y="w").to_svg
    )
    assert messages == []
    assert ">null<" in svg

    beside_empty = pl.DataFrame(
        {"c": [None, None, "a", "a", "z", "z"], "w": [1.0, 2.0, 3.0, 4.0, None, None]},
        schema={"c": pl.Utf8, "w": pl.Float64},
    )
    svg, messages = _render_capturing(
        fm.Chart(beside_empty).mark_point().encode(x="c:N", y="w").to_svg
    )
    assert messages == [_slot_message("mark_point")]
    assert ">null<" in svg


def test_a_null_category_with_no_band_is_not_a_slot():
    """Null-category rows that carry no value either take no band, so the axis
    shows no slot for them and none is named. A real category in the same
    state is a slot and is named."""
    null_category = pl.DataFrame(
        {"c": [None, None, "a", "a"], "w": [None, None, 3.0, 4.0]},
        schema={"c": pl.Utf8, "w": pl.Float64},
    )
    svg, messages = _render_capturing(
        fm.Chart(null_category).mark_point().encode(x="c:N", y="w").to_svg
    )
    assert messages == []
    assert ">null<" not in svg

    real_category = null_category.with_columns(pl.col("c").fill_null("n"))
    svg, messages = _render_capturing(
        fm.Chart(real_category).mark_point().encode(x="c:N", y="w").to_svg
    )
    assert messages == [_slot_message("mark_point", category="n")]
    assert ">n<" in svg


def test_a_null_category_is_a_slot_when_another_layer_opened_the_null_band():
    """The axis places a null band when a row of a null category carries a
    value in any layer that draws it. Layer 0's null-category rows draw, so the
    band exists; layer 1's are all unplaced, so it draws nothing in a band the
    reader sees, and layer 1 names it. With layer 1's null-category rows drawn
    too, nothing is empty."""
    opened = pl.DataFrame(
        {"c": ["a", "a", None, None], "w": [1.0, 2.0, 3.0, 4.0]},
        schema={"c": pl.Utf8, "w": pl.Float64},
    )

    def render(null_category_values: list[float | None]) -> tuple[str, list[str]]:
        """Render layer 0 (``opened``) with a line layer whose null category holds these values."""
        second = pl.DataFrame(
            {"c": ["a", "a", None, None], "w": [1.0, 2.0, *null_category_values]},
            schema={"c": pl.Utf8, "w": pl.Float64},
        )
        points = fm.Chart(opened).mark_point().encode(x="c:N", y="w")
        line = fm.Chart(second).mark_line().encode(x="c:N", y="w")
        return _render_capturing((points + line).to_svg)

    svg, messages = render([None, None])
    assert messages == [
        "layer 1: mark_line: every row of c=null has a null or non-finite y, so it draws nothing"
    ]
    assert ">null<" in svg
    _, control = render([5.0, 6.0])
    assert control == []


def _hue_group_frame() -> pl.DataFrame:
    """Categories ``a``, ``b`` and ``z`` cycle through sixty rows split over
    hue levels ``p``, ``q`` and ``r``; ``r`` is null throughout, so every
    category still draws and only the group ``r`` draws nothing."""
    return pl.DataFrame(
        {
            "c": ["a", "b", "z"] * 20,
            "g": ["p"] * 20 + ["q"] * 20 + ["r"] * 20,
            "w": [float((i * 7) % 11) for i in range(40)] + [None] * 20,
        },
        schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64},
    )


@pytest.mark.parametrize(
    "build,entry",
    [
        pytest.param(
            lambda df: fm.Chart(df).mark_boxplot().encode(x="c", y="w", color="g"),
            "mark_boxplot",
            id="boxplot",
        ),
        pytest.param(
            lambda df: fm.Chart(df).mark_errorbar().encode(x="c", y="w", color="g"),
            "mark_errorbar",
            id="errorbar",
        ),
        pytest.param(
            lambda df: fm.Chart(df).mark_violin().encode(x="c", y="w", color="g"),
            "mark_violin",
            id="violin_inner_box",
        ),
        pytest.param(
            lambda df: fm.catplot(df, x="c", y="w", hue="g", kind="box"),
            "catplot",
            id="catplot_box",
        ),
    ],
)
def test_a_composites_group_is_named_in_the_callers_channel(build: Callable, entry: str):
    """The group a hue level makes, not the slot, is what draws nothing here;
    the composite's layers that span ``y`` to ``y2`` name ``y`` like the rest,
    so the default warning filter shows one text."""
    _, shown = _render_capturing(build(_hue_group_frame()).to_svg, action="default")
    assert _census(shown) == [
        f"{entry}: every row of g=r has a null or non-finite y, so it draws nothing"
    ]


@_HORIZONTAL_CATPLOT_XFAIL
def test_a_horizontal_catplot_names_the_callers_channel_for_an_empty_group():
    """The hue level ``r`` is null throughout; the caller bound the values to
    ``x``, so the group should read ``x``. The render names ``y``, the
    lowering's own channel, on the group report as it does on the slot's."""
    chart = fm.catplot(_hue_group_frame(), y="c", x="w", kind="box", hue="g", orient="h")
    _, shown = _render_capturing(chart.to_svg, action="default")
    assert _census(shown) == [
        "catplot: every row of g=r has a null or non-finite x, so it draws nothing"
    ]


@pytest.mark.parametrize("mark_method", ["mark_ribbon", "mark_rule"])
@pytest.mark.parametrize(
    "z_rows,channels",
    [
        pytest.param([(None, None), (None, None)], "y and y2", id="both_missing"),
        pytest.param([(None, 8.0), (None, 9.0)], "y", id="only_y_missing"),
        pytest.param([(7.0, None), (6.0, None)], "y2", id="only_y2_missing"),
        pytest.param([(None, None), (None, 9.0)], "y", id="y_missing_everywhere_y2_on_one_row"),
        pytest.param([(None, None), (7.0, None)], "y2", id="y2_missing_everywhere_y_on_one_row"),
        pytest.param([(None, 8.0), (7.0, None)], "y and y2", id="rows_miss_different_channels"),
    ],
)
def test_a_caller_placed_layer_names_the_channels_it_wrote(
    mark_method: str, z_rows: list[tuple[float | None, float | None]], channels: str
):
    """The caller bound ``y`` and ``y2`` themselves, so the slot is named by
    the channels every row lacks, the strongest true statement; only when each
    row lacks a different one is each named. A lone ``y2`` stays ``y2``."""
    frame = _small_slot_frame([row[0] for row in z_rows], [row[1] for row in z_rows])
    chart = getattr(fm.Chart(frame), mark_method)().encode(x="c", y="w", y2="w2")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message(mark_method, channels)]


def test_a_call_built_layer_and_a_placed_one_name_the_same_slot_differently():
    """An error band spans ``y`` to ``y2`` and a ribbon the caller wrote spans
    the same two channels, over the same empty slot: the call bound the band's
    ``y2``, so it names ``y``; the caller bound the ribbon's, so it names both."""
    df = _slot_frame(_EMPTY_Z)
    _, band = _render_capturing(fm.Chart(df).mark_errorband().encode(x="c", y="w").to_svg)
    _, ribbon = _render_capturing(fm.Chart(df).mark_ribbon().encode(x="c", y="w", y2="w2").to_svg)
    assert band == [_slot_message("mark_errorband", "y")]
    assert ribbon == [_slot_message("mark_ribbon", "y and y2")]


def test_a_band_slot_reaches_the_interactive_route():
    """``.interactive().to_html()`` renders a static fallback and the
    interactive scene, so the slot is named once per render, the count the
    group census gives on the same route. ``Chart.to_html()`` renders once."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z)).mark_point().encode(x="c", y="w")
    expected = _slot_message("mark_point")
    _, interactive = _render_capturing(lambda: chart.interactive().to_html())
    _, group_interactive = _render_capturing(
        lambda: _null_group_line_chart().interactive().to_html()
    )
    _, plain = _render_capturing(chart.to_html)
    assert plain == [expected]
    assert interactive == [expected] * len(group_interactive)
    assert len(group_interactive) > 0


@pytest.mark.parametrize("mark_method,encoding,channels", _PLACED_SLOT_MARKS)
def test_a_chart_with_no_empty_slot_warns_nothing(mark_method: str, encoding: dict, channels: str):
    """Every slot draws, so there is nothing to name; emptying ``z`` on the same
    chart names it, so the silence is the data and not a chart the census
    cannot see."""
    _, messages = _render_capturing(
        _slot_chart(_slot_frame(_DRAWN_Z), mark_method, encoding).to_svg
    )
    assert messages == []
    _, emptied = _render_capturing(_slot_chart(_slot_frame(_EMPTY_Z), mark_method, encoding).to_svg)
    assert emptied == [_slot_message(mark_method, channels)]


@pytest.mark.parametrize("mark_method,encoding", _CALL_BUILT_SLOT_MARKS)
def test_a_composite_with_no_empty_slot_warns_nothing(mark_method: str, encoding: dict):
    _, messages = _render_capturing(
        _slot_chart(_slot_frame(_DRAWN_Z), mark_method, encoding).to_svg
    )
    assert messages == []
    _, emptied = _render_capturing(_slot_chart(_slot_frame(_EMPTY_Z), mark_method, encoding).to_svg)
    assert set(emptied) == {_slot_message(mark_method)} | _statistic_reports(mark_method)


@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("kind", _CATPLOT_KINDS)
def test_a_catplot_with_no_empty_slot_warns_nothing(kind: str, hue: str | None):
    _, messages = _render_capturing(_catplot_chart(_slot_frame(_DRAWN_Z), kind, hue).to_svg)
    assert _census(messages) == []
    _, emptied = _render_capturing(_catplot_chart(_slot_frame(_EMPTY_Z), kind, hue).to_svg)
    assert set(_census(emptied)) == {_slot_message("catplot")}


def test_a_zero_row_frame_on_a_categorical_axis_still_warns_no_rows_to_draw():
    df = pl.DataFrame({"c": [], "w": []}, schema={"c": pl.Utf8, "w": pl.Float64})
    chart = fm.Chart(df).mark_point().encode(x="c", y="w")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "the chart has no rows to draw after its transforms; its panel is drawn empty"
    ]


@pytest.mark.parametrize(
    "categories",
    [
        pytest.param(["a", "a", "b", "b"], id="two_categories"),
        pytest.param(["z", "z", "z", "z"], id="one_category"),
    ],
)
def test_a_layer_with_no_placed_row_is_named_as_a_layer_not_as_each_slot(categories: list[str]):
    """Every row of every slot is unplaced, so the layer is the unit that
    draws nothing; its slots are the same rows and are not named again."""
    df = pl.DataFrame({"c": categories, "w": [None] * 4}, schema={"c": pl.Utf8, "w": pl.Float64})
    _, messages = _render_capturing(fm.Chart(df).mark_point().encode(x="c", y="w").to_svg)
    assert messages == [
        "mark_point: every row has a null or non-finite y, so the layer draws nothing"
    ]


@pytest.mark.parametrize(
    "first,second,expected",
    [
        pytest.param(
            "z",
            "m",
            "mark_point: every row of c=z, and of c=m, has a null or non-finite y, "
            "so they draw nothing",
            id="z_then_m",
        ),
        pytest.param(
            "m",
            "z",
            "mark_point: every row of c=m, and of c=z, has a null or non-finite y, "
            "so they draw nothing",
            id="m_then_z",
        ),
    ],
)
def test_several_empty_slots_join_in_first_appearance_order(first: str, second: str, expected: str):
    """Slots are listed in the order the data first meets them, not sorted."""
    df = pl.DataFrame(
        {
            "c": ["a", "a", first, first, second, second, "b", "b"],
            "w": [1.0, 2.0, None, None, None, None, 3.0, 4.0],
        },
        schema={"c": pl.Utf8, "w": pl.Float64},
    )
    _, messages = _render_capturing(fm.Chart(df).mark_point().encode(x="c", y="w").to_svg)
    assert messages == [expected]


@pytest.mark.parametrize(
    "dtype,categories,x,label",
    [
        pytest.param(pl.Utf8, ["a", "z"], "c", "z", id="utf8"),
        pytest.param(pl.Categorical, ["a", "z"], "c", "z", id="categorical"),
        pytest.param(pl.Enum(["a", "z"]), ["a", "z"], "c", "z", id="enum"),
        pytest.param(pl.Int64, [1, 2], "c:N", "2", id="int_declared_nominal"),
        pytest.param(pl.Utf8, ["a", "Smith, J"], "c", "Smith, J", id="comma"),
        pytest.param(pl.Utf8, ["a", "café über"], "c", "café über", id="non_ascii"),
        pytest.param(pl.Utf8, ["a", ""], "c", "", id="empty_string"),
    ],
)
def test_a_slot_is_named_by_its_category_whatever_the_column_holds(
    dtype: pl.DataType | type[pl.DataType], categories: list, x: str, label: str
):
    """The slot is spelled with its category as the column holds it, the way a
    legend group bound to the same column is."""
    df = pl.DataFrame(
        {"c": [categories[0]] * 2 + [categories[1]] * 2, "w": [1.0, 2.0, None, None]},
        schema={"c": dtype, "w": pl.Float64},
    )
    _, messages = _render_capturing(fm.Chart(df).mark_point().encode(x=x, y="w").to_svg)
    assert messages == [_slot_message("mark_point", category=label)]


def test_a_slot_of_integer_values_is_named():
    """An integer value column holds nulls but no NaN or infinity; its empty
    slot is named like a float column's."""
    df = pl.DataFrame(
        {"c": ["a", "a", "z", "z"], "w": [1, 2, None, None]}, schema={"c": pl.Utf8, "w": pl.Int64}
    )
    _, messages = _render_capturing(fm.Chart(df).mark_point().encode(x="c", y="w").to_svg)
    assert messages == [_slot_message("mark_point")]


def test_a_numeric_axis_has_no_slot_until_it_is_declared_nominal():
    """An integer column on ``x`` is a continuous axis: a value with no
    drawable ``y`` is the documented per-row skip. Declared nominal, the same
    column is a band axis, and its empty category is a slot."""
    df = pl.DataFrame(
        {"c": [1, 1, 2, 2], "w": [1.0, 2.0, None, None]}, schema={"c": pl.Int64, "w": pl.Float64}
    )
    _, continuous = _render_capturing(fm.Chart(df).mark_point().encode(x="c", y="w").to_svg)
    assert continuous == []
    _, nominal = _render_capturing(fm.Chart(df).mark_point().encode(x="c:N", y="w").to_svg)
    assert nominal == [_slot_message("mark_point", category="2")]


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_line"])
def test_a_temporal_axis_has_no_slot_until_its_values_are_text(mark_method: str):
    """A date on ``x`` is a continuous axis: a date with no drawable ``y`` is
    the documented per-row skip, as in any time series with a gap. The same
    dates as text are categories, and the one with no ``y`` is a slot."""
    dates = pl.DataFrame(
        {"d": [dt.date(2026, 1, day) for day in (1, 2, 3, 4)], "w": [1.0, 2.0, None, 4.0]},
        schema={"d": pl.Date, "w": pl.Float64},
    )
    as_text = dates.with_columns(pl.col("d").cast(pl.Utf8))
    encoding = {"x": "d", "y": "w"}
    _, temporal = _render_capturing(_slot_chart(dates, mark_method, encoding).to_svg)
    assert temporal == []
    _, text = _render_capturing(_slot_chart(as_text, mark_method, encoding).to_svg)
    assert text == [_slot_message(mark_method, category="2026-01-03", column="d")]


def test_a_heatmap_row_with_a_null_color_stays_silent():
    """Both axes are bands, and a null color is not a missing position: the
    cells of row ``r3`` all place and draw, so neither axis has an empty slot."""
    df = pl.DataFrame(
        {
            "r": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "k": ["k1", "k2", "k1", "k2", "k1", "k2"],
            "v": [1.0, 2.0, 3.0, 4.0, None, None],
        },
        schema={"r": pl.Utf8, "k": pl.Utf8, "v": pl.Float64},
    )
    svg, messages = _render_capturing(
        fm.Chart(df).mark_rect().encode(x="k", y="r", color="v").to_svg
    )
    assert messages == []
    assert ">r3<" in svg


def test_a_flipped_chart_names_the_callers_channel_for_a_band_slot():
    """Under ``CoordFlip`` the caller still wrote ``y=``, so that is the
    channel named, as it is for a group."""
    df = _slot_frame(_EMPTY_Z)
    chart = fm.Chart(df).mark_point().encode(x="c", y="w").coord(fm.CoordFlip())
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message("mark_point", "y")]
    boxplot = fm.Chart(df).mark_boxplot().encode(x="c", y="w").coord(fm.CoordFlip())
    _, shown = _render_capturing(boxplot.to_svg, action="default")
    assert shown == [_slot_message("mark_boxplot", "y")]


@pytest.mark.parametrize(
    "compose,expected",
    [
        pytest.param(
            lambda empty, other: empty + other,
            "layer 0: " + _slot_message("mark_point"),
            id="plus_first_layer",
        ),
        pytest.param(
            lambda empty, other: other + empty,
            "layer 1: " + _slot_message("mark_point"),
            id="plus_second_layer",
        ),
        pytest.param(lambda empty, other: other | empty, _slot_message("mark_point"), id="hconcat"),
    ],
)
def test_a_band_slot_in_a_composition_names_its_layer(compose: Callable, expected: str):
    empty = fm.Chart(_slot_frame(_EMPTY_Z)).mark_point().encode(x="c", y="w")
    other = fm.Chart(_slot_frame(_DRAWN_Z)).mark_line().encode(x="c", y="w")
    _, messages = _render_capturing(compose(empty, other).to_svg)
    assert messages == [expected]


def test_a_call_built_layer_in_a_composition_carries_no_layer_index():
    """A layer a call built is located by its call (``mark_errorband``); its
    index counts the call's internal layers, which the caller never wrote."""
    other = fm.Chart(_slot_frame(_DRAWN_Z)).mark_point().encode(x="c", y="w")
    band = fm.Chart(_slot_frame(_EMPTY_Z)).mark_errorband().encode(x="c", y="w")
    _, messages = _render_capturing((other + band).to_svg)
    assert messages == [_slot_message("mark_errorband")]


# ---------------------------------------------------------------------------
# A summary statistic keeps the key of a group a null column empties. ``Bin``
# (a histogram) and ``LetterValue`` (a boxen) leave one placeholder row for such
# a group, as ``Aggregate`` and ``BoxStats`` do, so the legend keeps the hue
# level, the figure keeps the facet panel, the axis keeps the category's band
# slot, and the census names the unit. A zero-row frame has no group to keep
# and still reports that it has no rows to draw.
# ---------------------------------------------------------------------------

_NO_ROWS_TO_DRAW = "the chart has no rows to draw after its transforms; its panel is drawn empty"
_NO_FACET_ROWS_WARNING = (
    "the chart has no rows to draw after its transforms; no facet panel is drawn"
)

# Hue levels ``a`` and ``b`` always hold values to bin; ``c`` holds what a test gives it.
_HIST_A = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
_HIST_B = [2.0, 3.0, 4.0, 4.0, 5.0, 6.0]


def _hist_group_frame(c_values: list[float | None]) -> pl.DataFrame:
    """Hue levels ``a`` and ``b`` of six values each, then level ``c`` holding ``c_values``."""
    return pl.DataFrame(
        {"v": _HIST_A + _HIST_B + c_values, "g": ["a"] * 6 + ["b"] * 6 + ["c"] * len(c_values)},
        schema={"v": pl.Float64, "g": pl.Utf8},
    )


def _hist_facet_frame(q_values: list[float | None]) -> pl.DataFrame:
    """Facet panel ``p`` of six values, then panel ``q`` holding ``q_values``."""
    return pl.DataFrame(
        {"v": _HIST_A + q_values, "f": ["p"] * 6 + ["q"] * len(q_values)},
        schema={"v": pl.Float64, "f": pl.Utf8},
    )


def _hist_panel_level_frame(panels: dict[str, dict[str, list[float | None]]]) -> pl.DataFrame:
    """One frame of ``v`` split by hue level ``g`` and facet panel ``f``: ``panels`` maps each
    panel to the values each of its levels holds."""
    rows = [
        (panel, level, value)
        for panel, levels in panels.items()
        for level, values in levels.items()
        for value in values
    ]
    return pl.DataFrame(rows, schema={"f": pl.Utf8, "g": pl.Utf8, "v": pl.Float64}, orient="row")


def _displot_hist(frame: pl.DataFrame, *, hue=None, facet=None, x="v", **options) -> fm.Chart:
    return fm.displot(frame, x=x, kind="hist", **_given(hue=hue, col=facet), **options)


def _mark_histogram(frame: pl.DataFrame, *, hue=None, facet=None, x="v", **options) -> fm.Chart:
    chart = fm.Chart(frame).mark_histogram(**options).encode(x=x, **_given(color=hue))
    return chart if facet is None else chart.facet(col=facet)


def _binned_bar(frame: pl.DataFrame, *, hue=None, facet=None, **options) -> fm.Chart:
    chart = (
        fm.Chart(frame)
        .mark_bar(**options)
        .encode(x=fm.X("v", bin=True), y="count()", **_given(color=hue))
    )
    return chart if facet is None else chart.facet(col=facet)


class _HistogramDoor(NamedTuple):
    """One way to draw a histogram of ``v``."""

    build: Callable[..., fm.Chart]  # ``build(frame, hue=, facet=, **options)``
    entry: str  # the entry its layer speaks as
    channels: str  # the channels it names for a bar with no position
    variants: dict[str, dict]  # the options that place or scale its bars differently


_HISTOGRAM_DOORS = {
    "displot": _HistogramDoor(
        _displot_hist,
        "displot",
        "x",
        {
            "layer": {},
            "stack": {"multiple": "stack"},
            "dodge": {"multiple": "dodge"},
            "fill": {"multiple": "fill"},
            "density": {"stat": "density"},
            "cumulative": {"cumulative": True},
        },
    ),
    "mark_histogram": _HistogramDoor(
        _mark_histogram,
        "mark_histogram",
        "x",
        {
            "layer": {},
            "stack": {"multiple": "stack"},
            "dodge": {"multiple": "dodge"},
            "fill": {"multiple": "fill"},
            "density": {"density": True},
            "cumulative": {"cumulative": True},
        },
    ),
    # The caller placed this bar layer itself, so it names every channel the binning wrote.
    "binned_bar": _HistogramDoor(_binned_bar, "mark_bar", "x and x2", {"layer": {}}),
}
_HISTOGRAM_DOOR_CASES = [pytest.param(door, id=name) for name, door in _HISTOGRAM_DOORS.items()]
_HISTOGRAM_VARIANTS = [
    (name, door, variant, options)
    for name, door in _HISTOGRAM_DOORS.items()
    for variant, options in door.variants.items()
]
_HISTOGRAM_VARIANT_CASES = [
    pytest.param(door, options, id=f"{name}-{variant}")
    for name, door, variant, options in _HISTOGRAM_VARIANTS
]
# The variants whose bars an emptied level leaves exactly where they are. A dodge is the one
# placement that gives the level a share: it keeps its sub-slot, empty.
_HISTOGRAM_UNMOVED_VARIANT_CASES = [
    pytest.param(door, options, id=f"{name}-{variant}")
    for name, door, variant, options in _HISTOGRAM_VARIANTS
    if variant != "dodge"
]
# The doors that place their bars side by side.
_DODGED_HISTOGRAM_DOORS = [
    pytest.param(door, id=name)
    for name, door in _HISTOGRAM_DOORS.items()
    if "dodge" in door.variants
]


@pytest.mark.parametrize("value", _UNDRAWABLE)
def test_a_histogram_hue_group_a_null_column_empties_is_named(value: float | None):
    """Hue level ``c`` has no value to bin, so it draws no bar; it keeps its legend entry and is
    named, as the same level of an ECDF is. A null, a NaN and either infinity read alike."""
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0] + [value] * 4,
            "h": ["a", "a", "a", "a", "c", "c", "c", "c"],
        },
        schema={"v": pl.Float64, "h": pl.Utf8},
    )
    svg, messages = _render_capturing(lambda: fm.displot(df, x="v", kind="hist", hue="h").to_svg())
    assert messages == [_slot_message("displot", "x", "c", "h")]
    assert ">c<" in svg
    assert ">a<" in svg


def test_a_histogram_hue_group_of_an_integer_column_with_no_value_is_named():
    """An integer column holds nulls but no NaN or infinity; its emptied level is kept and named
    like a float column's."""
    df = pl.DataFrame(
        {"v": [1, 2, 3, 4, None, None, None, None], "h": ["a"] * 4 + ["c"] * 4},
        schema={"v": pl.Int64, "h": pl.Utf8},
    )
    svg, messages = _render_capturing(lambda: fm.displot(df, x="v", kind="hist", hue="h").to_svg())
    assert messages == [_slot_message("displot", "x", "c", "h")]
    assert ">c<" in svg


@pytest.mark.parametrize("door,options", _HISTOGRAM_UNMOVED_VARIANT_CASES)
def test_a_histogram_placeholder_adds_only_the_legend_entry_and_the_warning(
    door: _HistogramDoor, options: dict
):
    """However the bars are overlaid, stacked, normalized, scaled or accumulated, the emptied
    level ``c`` adds its legend entry and one warning and changes no mark: the same chart without
    ``c``'s rows is silent and draws the groups ``a`` and ``b`` byte for byte alike. The sibling
    has no rows for ``c`` and so no legend entry for it; the marks are what is compared. A dodge
    is the one placement the level takes a share of: see
    ``test_a_dodged_histogram_keeps_the_emptied_levels_slot``."""
    frame = _hist_group_frame([None] * 4)
    sibling = frame.filter(pl.col("g") != "c")
    svg, messages = _render_capturing(door.build(frame, hue="g", **options).to_svg)
    sibling_svg, sibling_messages = _render_capturing(
        door.build(sibling, hue="g", **options).to_svg
    )
    assert messages == [_slot_message(door.entry, door.channels, "c", "g")]
    assert sibling_messages == []
    assert ">c<" in svg
    assert data_bar_rects(mark_layer(sibling_svg)), "the sibling drew no bar to compare"
    assert mark_layer(svg) == mark_layer(sibling_svg), (
        "the emptied level changed how the other groups draw"
    )


@pytest.mark.parametrize("door", _HISTOGRAM_DOOR_CASES)
def test_a_histogram_facet_panel_a_null_column_empties_is_named(door: _HistogramDoor):
    """Panel ``q`` has no value to bin and no grouping of its own, so it draws no bar; the panel
    stays on the figure and is named."""
    svg, messages = _render_capturing(door.build(_hist_facet_frame([None] * 4), facet="f").to_svg)
    assert messages == [_panel_message(door.entry, door.channels)]
    assert ">q<" in svg


@pytest.mark.parametrize("door", _HISTOGRAM_DOOR_CASES)
def test_a_histogram_hue_group_a_null_column_empties_in_one_facet_panel_is_named(
    door: _HistogramDoor,
):
    """Level ``a`` has values in panel ``p`` and none in panel ``q``, where level ``b`` still
    draws: the level is named with the panel it is empty in."""
    frame = _hist_panel_level_frame(
        {
            "p": {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 4.0, 5.0]},
            "q": {"a": [None] * 4, "b": [3.0, 4.0, 5.0, 6.0]},
        }
    )
    _, messages = _render_capturing(door.build(frame, hue="g", facet="f").to_svg)
    assert messages == [_slot_message(door.entry, door.channels, "a", "g", panel="f=q")]


@pytest.mark.parametrize("emptied_first", [True, False], ids=["emptied_first", "emptied_second"])
@pytest.mark.parametrize("value", _UNDRAWABLE)
def test_an_all_null_histogram_layer_beside_another_is_named(
    value: float | None, emptied_first: bool
):
    """A histogram layer with no value to bin is named by its call, whichever side of the ``+`` it
    is on and with no layer index, since the caller never wrote the call's own layers; the layer
    beside it still draws all its bars."""
    drawn = _mark_histogram(pl.DataFrame({"v": _HIST_A}))
    emptied = _mark_histogram(pl.DataFrame({"v": [value] * 4}, schema={"v": pl.Float64}))
    pair = (emptied + drawn) if emptied_first else (drawn + emptied)
    svg, messages = _render_capturing(pair.to_svg)
    assert messages == [_layer_message("mark_histogram", "x")]
    alone = len(data_bar_rects(mark_layer(drawn.to_svg())))
    assert alone > 0
    assert len(data_bar_rects(mark_layer(svg))) == alone


@pytest.mark.parametrize("value", _UNDRAWABLE)
@pytest.mark.parametrize("door", _HISTOGRAM_DOOR_CASES)
def test_an_all_null_histogram_names_its_nulls(door: _HistogramDoor, value: float | None):
    """Rows are present and none has a value to bin, so the cause is the values: the layer is
    named, not reported as a chart with no rows to draw."""
    frame = pl.DataFrame({"v": [value] * 4}, schema={"v": pl.Float64})
    _, messages = _render_capturing(door.build(frame).to_svg)
    assert messages == [_layer_message(door.entry, door.channels)]


@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("door", _HISTOGRAM_DOOR_CASES)
def test_a_zero_row_histogram_still_warns_no_rows_to_draw(door: _HistogramDoor, hue: str | None):
    """With no row at all there is no group to keep and no value to blame, so the histogram
    reports a chart with no rows to draw."""
    frame = _hist_group_frame([None]).clear()
    _, messages = _render_capturing(door.build(frame, hue=hue).to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("door,options", _HISTOGRAM_VARIANT_CASES)
def test_a_histogram_group_that_keeps_a_usable_value_warns_nothing(
    door: _HistogramDoor, options: dict
):
    """Level ``c`` keeps one usable value beside its nulls, so it still draws: nothing is named,
    and the nulls change nothing, the chart being byte for byte the one with those rows left
    out."""
    frame = _hist_group_frame([7.0, None, None, None])
    without_nulls = frame.filter(pl.col("v").is_not_null())
    svg, messages = _render_capturing(door.build(frame, hue="g", **options).to_svg)
    expected_svg, expected_messages = _render_capturing(
        door.build(without_nulls, hue="g", **options).to_svg
    )
    assert messages == []
    assert expected_messages == []
    assert ">c<" in svg
    assert svg == expected_svg, "the null rows changed how a group with a usable value draws"


@pytest.mark.parametrize(
    "frame,split",
    [
        pytest.param(_hist_facet_frame([7.0, None, None, None]), {"facet": "f"}, id="panel"),
        pytest.param(
            pl.DataFrame({"v": [1.0, None, 2.0, None, 3.0]}, schema={"v": pl.Float64}),
            {},
            id="layer",
        ),
    ],
)
@pytest.mark.parametrize("door", _HISTOGRAM_DOOR_CASES)
def test_a_histogram_panel_or_layer_that_keeps_a_usable_value_warns_nothing(
    door: _HistogramDoor, frame: pl.DataFrame, split: dict
):
    """A facet panel, and an unfaceted layer, that keeps a usable value beside its nulls still
    draws: nothing is named, and the nulls change nothing, the chart being byte for byte the one
    with those rows left out."""
    svg, messages = _render_capturing(door.build(frame, **split).to_svg)
    expected_svg, expected_messages = _render_capturing(
        door.build(frame.filter(pl.col("v").is_not_null()), **split).to_svg
    )
    assert messages == []
    assert expected_messages == []
    assert svg == expected_svg, "the null rows changed how a panel or layer with a value draws"


def _bars(marks: str) -> list[dict[str, str]]:
    """The bars in ``marks`` (a render's mark layer, or one panel of it), each with every attribute
    it was drawn with, left to right."""
    return sorted(data_bar_rects(marks), key=lambda bar: (float(bar["x"]), bar["fill"]))


def _bar_widths(bars: list[dict[str, str]]) -> set[float]:
    return {float(bar["width"]) for bar in bars}


def _widest_mark(svg: str) -> float:
    """The widest extent across its axis of any mark a render draws: a rect's width or a line's span."""
    marks = mark_layer(svg)
    spans = [float(rect["width"]) for rect in data_bar_rects(marks)]
    lines = svg_elements(marks, "line", lambda attrs: "x1" in attrs and "x2" in attrs)
    spans += [abs(float(line["x2"]) - float(line["x1"])) for line in lines]
    return max(spans)


@pytest.mark.parametrize("door", _DODGED_HISTOGRAM_DOORS)
def test_a_dodged_histogram_keeps_the_emptied_levels_slot(door: _HistogramDoor):
    """A level with no usable value keeps its sub-slot in every dodged bin, empty, as every
    statistic that keeps a group's key does: the bars of ``a`` and ``b`` are rect for rect those
    of the chart whose level ``c`` holds values inside their range, a third narrower than
    without ``c``'s rows, and nothing is drawn in ``c``'s third."""
    options = door.variants["dodge"]
    emptied = _hist_group_frame([None] * 4)
    svg, messages = _render_capturing(door.build(emptied, hue="g", **options).to_svg)
    valued, _ = _render_capturing(
        door.build(_hist_group_frame([2.5, 3.5, 4.5]), hue="g", **options).to_svg
    )
    alone, _ = _render_capturing(
        door.build(emptied.filter(pl.col("g") != "c"), hue="g", **options).to_svg
    )
    drawn, real = _bars(mark_layer(svg)), _bars(mark_layer(valued))
    level_c = {bar["fill"] for bar in real} - {bar["fill"] for bar in drawn}
    assert messages == [_slot_message(door.entry, door.channels, "c", "g")]
    assert ">c<" in svg
    assert len(level_c) == 1, "a bar drew in the emptied level's slot"
    assert drawn == [bar for bar in real if bar["fill"] not in level_c]
    width, alone_width = _bar_widths(drawn), _bar_widths(_bars(mark_layer(alone)))
    assert len(width) == len(alone_width) == 1
    assert width.pop() == pytest.approx(alone_width.pop() * 2 / 3, abs=0.01)


@pytest.mark.parametrize("door", _DODGED_HISTOGRAM_DOORS)
def test_a_dodged_histogram_level_emptied_in_one_panel_keeps_every_panels_bar_width(
    door: _HistogramDoor,
):
    """Level ``c`` holds values in panel ``p`` and only nulls in panel ``q``. Its sub-slot stays in
    ``q``, so the bars of ``a`` and ``b`` have one width in both panels; with ``c``'s rows left
    out of ``q`` they would widen there, one level changing width between the panels of one
    figure."""
    options = door.variants["dodge"]
    levels = {"a": _HIST_A, "b": _HIST_B}
    frame = _hist_panel_level_frame(
        {"p": {**levels, "c": [2.5, 3.5, 4.5]}, "q": {**levels, "c": [None] * 3}}
    )
    svg, messages = _render_capturing(door.build(frame, hue="g", facet="f", **options).to_svg)
    assert messages == [_slot_message(door.entry, door.channels, "c", "g", panel="f=q")]
    assert len({w for panel in mark_groups(svg) for w in _bar_widths(_bars(panel))}) == 1
    without = frame.filter(~((pl.col("g") == "c") & (pl.col("f") == "q")))
    svg_without, _ = _render_capturing(door.build(without, hue="g", facet="f", **options).to_svg)
    widths = [_bar_widths(_bars(panel)) for panel in mark_groups(svg_without)]
    assert len(widths) == 2 and widths[0] != widths[1], "the bars did not widen in panel q"


@pytest.mark.parametrize(
    "mark_method,encoding",
    [
        pytest.param(
            "mark_bar", {"x": "c", "y": fm.Y("w", aggregate="mean"), "color": "g"}, id="mean_bar"
        ),
        pytest.param("mark_boxplot", {"x": "c", "y": "w", "color": "g"}, id="boxplot"),
        pytest.param("mark_errorbar", {"x": "c", "y": "w", "color": "g"}, id="errorbar"),
        pytest.param("mark_boxen", {"x": "c", "y": "w", "color": "g"}, id="boxen"),
    ],
)
def test_an_emptied_hue_level_keeps_its_dodge_slot_on_every_placeholder_statistic(
    mark_method: str, encoding: dict
):
    """A key a statistic keeps keeps its slot. ``Aggregate``, ``BoxStats``, ``ErrorExtent`` and
    ``LetterValue`` each leave a placeholder row for a hue level with no usable value, and a dodge
    counts the groups a layer's rows carry, so the marks of the other two levels are two thirds of
    the width they have in the chart without that level's rows."""
    frame = _hue_group_frame()

    def widest(rows: pl.DataFrame) -> float:
        chart = getattr(fm.Chart(rows), mark_method)(position=fm.Dodge()).encode(**encoding)
        return _widest_mark(_render_capturing(chart.to_svg)[0])

    assert widest(frame) == pytest.approx(
        widest(frame.filter(pl.col("g") != "r")) * 2 / 3, abs=0.01
    )


# A color scale applies an explicit domain together with a color range.
_THREE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]


def test_an_explicit_colour_domain_orders_the_dodge_slots_but_adds_none():
    """A color scale's explicit domain says which level takes which sub-slot of a dodged bin, and
    a level it lists beyond those the rows carry takes none: with ``c`` listed the legend gains
    its entry and the bars of ``a`` and ``b`` keep the width and place they have without it, and
    reversing the domain swaps their places. A domain with no range is not applied, so it
    changes nothing, and cannot reserve a slot either."""
    levels = _hist_group_frame([])

    def dodged(color) -> tuple[str, list[dict[str, str]]]:
        svg, _ = _render_capturing(_displot_hist(levels, hue=color, multiple="dodge").to_svg)
        return svg, _bars(mark_layer(svg))

    def domain(names: list[str], *, colors: list[str] | None = None) -> fm.Color:
        scale = fm.OrdinalScale(domain=names, **({} if colors is None else {"range": colors}))
        return fm.Color("g", scale=scale)

    _, plain = dodged("g")
    _, unapplied = dodged(domain(["a", "b", "c"]))
    svg, listed = dodged(domain(["a", "b", "c"], colors=_THREE_COLORS))
    assert unapplied == plain
    assert ">c<" in svg
    assert [(bar["x"], bar["width"]) for bar in listed] == [
        (bar["x"], bar["width"]) for bar in plain
    ]

    one_each = pl.DataFrame({"c": ["p", "p"], "g": ["a", "b"], "w": [1.0, 2.0]})

    def heights(names: list[str]) -> list[float]:
        color = domain(names, colors=_THREE_COLORS[:2])
        chart = fm.Chart(one_each).mark_bar(position=fm.Dodge()).encode(x="c", y="w", color=color)
        svg, _ = _render_capturing(chart.to_svg)
        return [float(bar["height"]) for bar in _bars(mark_layer(svg))]

    a_then_b = heights(["a", "b"])
    assert a_then_b[0] != a_then_b[1]
    assert heights(["b", "a"]) == a_then_b[::-1]


def _log_message(axis: str, field: str, rows: int) -> str:
    """The text that reports ``rows`` rows of ``field`` a log scale on ``axis`` cannot place."""
    return (
        f"{axis} scale: {rows} row(s) of '{field}' are zero or negative, which a LogScale cannot "
        "place, so they are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
    )


# The options that put each histogram door's value axis on a log scale.
_LOG_AXIS = {
    "displot": {"log_scale": True},
    "mark_histogram": {"x": fm.X("v", scale=fm.LogScale())},
}
_LOG_AXIS_DOORS = list(_LOG_AXIS)


def _log_histogram(door: str, frame: pl.DataFrame, **options) -> tuple[str, list[str]]:
    """``frame`` as ``door``'s histogram on a log value axis, split by ``g``: its svg and warnings."""
    chart = _HISTOGRAM_DOORS[door].build(frame, hue="g", **_LOG_AXIS[door], **options)
    return _render_capturing(chart.to_svg)


@pytest.mark.parametrize("multiple", ["layer", "stack"])
@pytest.mark.parametrize("door", _LOG_AXIS_DOORS)
def test_a_histogram_hue_level_a_log_axis_empties_keeps_its_entry_and_is_named_beside_the_log_warning(
    door: str, multiple: str
):
    """On a log axis the histogram bins in the axis's space, where a level whose every value is
    zero or negative has no value to bin: the level keeps its legend entry as a placeholder and
    the census names it, beside the log warning that counts the rows the axis cannot place. The
    census sentence's cause is the statistic's output row, null or non-finite, not the caller's
    finite values; a statistic-side cause would change this sentence, and this test, with it."""
    svg, messages = _log_histogram(door, _hist_group_frame([-1.0, 0.0, -3.0]), multiple=multiple)
    expected = [
        _log_message("x", "v", 3),
        _slot_message(_HISTOGRAM_DOORS[door].entry, "x", "c", "g"),
    ]
    assert messages == expected
    assert ">c<" in svg


@pytest.mark.parametrize("door", _LOG_AXIS_DOORS)
def test_a_histogram_hue_level_with_an_on_side_value_on_a_log_axis_is_not_named(door: str):
    """Level ``c`` keeps two values on the log axis's side beside the one it cannot place, so it
    draws its bars: the log warning counts the one row and the level is not named."""
    svg, messages = _log_histogram(door, _hist_group_frame([-1.0, 5.0, 7.0]))
    assert messages == [_log_message("x", "v", 1)]
    assert ">c<" in svg


@pytest.mark.parametrize("door", _LOG_AXIS_DOORS)
def test_a_histogram_hue_level_of_nulls_on_a_log_axis_is_named_by_the_census_alone(door: str):
    """No value of a level of nulls is off the axis's side, so the log warning has nothing to count
    and the census names the level, as it does on a linear axis."""
    svg, messages = _log_histogram(door, _hist_group_frame([None] * 4))
    assert messages == [_slot_message(_HISTOGRAM_DOORS[door].entry, "x", "c", "g")]
    assert ">c<" in svg


@pytest.mark.parametrize("door", _LOG_AXIS_DOORS)
def test_a_histogram_hue_level_an_explicit_domain_cuts_is_reported_by_the_domain_warning_alone(
    door: str,
):
    """A level whose every value lies outside the value axis's explicit domain is reported by the
    domain warning, which counts marks for a statistic layer; the census leaves it alone and the
    level keeps its legend entry."""
    x = fm.X("v", scale=fm.LinearScale(domain=[0, 10]))
    chart = _HISTOGRAM_DOORS[door].build(_hist_group_frame([50.0, 55.0, 60.0]), hue="g", x=x)
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        f"{_HISTOGRAM_DOORS[door].entry}: x scale: 3 mark(s) of 'v' fall outside the scale's "
        "explicit domain, so they are not drawn; widen the domain, or pass clamp=True to draw "
        "them at the axis edge"
    ]
    assert ">c<" in svg


@pytest.mark.parametrize("door", _HISTOGRAM_DOOR_CASES)
def test_a_histogram_hue_level_with_one_null_row_keeps_its_entry_and_is_named(door: _HistogramDoor):
    """The smallest emptied level is one row: it keeps its legend entry and is named."""
    svg, messages = _render_capturing(door.build(_hist_group_frame([None]), hue="g").to_svg)
    assert messages == [_slot_message(door.entry, door.channels, "c", "g")]
    assert ">c<" in svg


def test_a_binned_mean_bar_names_the_channels_its_binning_wrote():
    """The caller placed the bar, so the census keeps every channel the binning wrote for it: an
    emptied level has no ``x`` or ``x2``, and the null aggregate reads as a null ``y``."""
    values = _HIST_A + _HIST_B + [None] * 3
    frame = pl.DataFrame(
        {"v": values, "w": [1.0] * 15, "g": ["a"] * 6 + ["b"] * 6 + ["c"] * 3},
        schema={"v": pl.Float64, "w": pl.Float64, "g": pl.Utf8},
    )
    chart = (
        fm.Chart(frame)
        .mark_bar()
        .encode(x=fm.X("v", bin=True), y=fm.Y("w", aggregate="mean"), color="g")
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_slot_message("mark_bar", "x and y and x2", "c", "g")]
    assert ">c<" in svg


def _mark_boxen(frame: pl.DataFrame, hue=None, facet=None) -> fm.Chart:
    chart = _slot_chart(frame, "mark_boxen", {"x": "c", "y": "w", **_given(color=hue)})
    return chart if facet is None else chart.facet(col=facet)


def _catplot_boxen(frame: pl.DataFrame, hue=None, facet=None) -> fm.Chart:
    return _catplot_chart(frame, "boxen", hue, facet=facet)


# The two ways to draw a boxen of ``w`` by category ``c``, keyed by the entry each speaks as.
_BOXEN_DOORS: dict[str, Callable[..., fm.Chart]] = {
    "mark_boxen": _mark_boxen,
    "catplot": _catplot_boxen,
}


@pytest.mark.parametrize(
    "value", [pytest.param(None, id="null"), pytest.param(float("nan"), id="nan")]
)
@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_category_with_no_usable_value_keeps_its_slot(
    door: str, hue: str | None, value: float | None
):
    """The axis still holds category ``z`` when none of its values is usable, whether or not a
    hue splits the boxes."""
    svg, _ = _render_capturing(_BOXEN_DOORS[door](_slot_frame([value] * 6), hue).to_svg)
    assert ">z<" in svg


@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_empty_category_is_named_in_the_callers_terms(door: str, hue: str | None):
    """The caller bound the categories to ``c``, so the empty slot reads ``c=z``, whatever column
    the boxen's own statistic keeps the category in. It is one text, whichever of the boxen's
    layers raises it."""
    chart = _BOXEN_DOORS[door](_slot_frame(_EMPTY_Z), hue)
    _, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [_slot_message(door)]
    _, every = _render_capturing(chart.to_svg)
    assert set(every) == {_slot_message(door)}


@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_hue_group_a_null_column_empties_is_named(door: str):
    """Hue level ``r`` has no usable value in any category while every category still draws
    through the other levels: the level keeps its legend entry and is named."""
    svg, shown = _render_capturing(
        _BOXEN_DOORS[door](_hue_group_frame(), "g").to_svg, action="default"
    )
    assert shown == [_slot_message(door, category="r", column="g")]
    assert ">r<" in svg


@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_facet_panel_a_null_column_empties_is_named(door: str):
    """Panel ``q`` has no usable value in any category, so it draws no box; the panel stays on the
    figure and is named."""
    frame = _faceted_slot_frame([], blank_in=("q",))
    svg, shown = _render_capturing(_BOXEN_DOORS[door](frame, facet="f").to_svg, action="default")
    assert shown == [_panel_message(door)]
    assert ">q<" in svg


@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_empty_category_in_one_facet_panel_is_named_in_the_callers_terms(door: str):
    """Category ``z`` is empty in panel ``q`` alone, where it draws nothing though it draws in
    panel ``p``; it is named with the panel, by the caller's own column."""
    frame = _faceted_slot_frame(["q"])
    _, shown = _render_capturing(_BOXEN_DOORS[door](frame, facet="f").to_svg, action="default")
    assert shown == [_slot_message(door, panel="f=q")]


def test_a_boxen_category_of_an_integer_column_with_no_value_keeps_its_slot():
    """An integer column holds nulls but no NaN or infinity; its emptied category keeps its slot
    like a float column's."""
    df = pl.DataFrame(
        {"c": ["a"] * 6 + ["b"] * 6 + ["z"] * 6, "w": list(range(1, 13)) + [None] * 6},
        schema={"c": pl.Utf8, "w": pl.Int64},
    )
    svg, _ = _render_capturing(_mark_boxen(df).to_svg)
    assert ">z<" in svg


@pytest.mark.parametrize(
    "order,expected",
    [
        pytest.param("ascending", ["a", "d", "b", "z", "y"], id="ascending"),
        pytest.param("descending", ["b", "d", "a", "z", "y"], id="descending"),
    ],
)
def test_a_boxen_sorted_by_an_aggregate_keeps_its_order_with_the_emptied_categories_last(
    order: str, expected: list[str]
):
    """The categories with values are ordered by their mean as they would be without the empty
    ones; ``z`` and ``y`` have no value, and follow them in either direction, in the order the
    data meets them."""
    df = pl.DataFrame(
        {
            "c": ["z"] * 6 + ["a"] * 6 + ["b"] * 6 + ["d"] * 6 + ["y"] * 6,
            "w": [None] * 6
            + [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]
            + [7.0, 8.0, 9.0, 8.0, 7.0, 9.0]
            + [4.0, 5.0, 6.0, 5.0, 4.0, 6.0]
            + [None] * 6,
        },
        schema={"c": pl.Utf8, "w": pl.Float64},
    )
    sort = {"field": "w", "op": "mean", "order": order}
    chart = fm.Chart(df).mark_boxen().encode(x=fm.X("c", sort=sort), y="w")
    svg, _ = _render_capturing(chart.to_svg)
    assert axis_tick_labels(svg, axis="x") == expected


@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_an_all_null_boxen_names_its_nulls(door: str):
    """Rows are present and none has a value, so the cause is the values: the layer is named,
    not reported as a chart with no rows to draw, and every category keeps its slot."""
    df = pl.DataFrame(
        {"c": ["a"] * 3 + ["b"] * 3, "w": [None] * 6}, schema={"c": pl.Utf8, "w": pl.Float64}
    )
    svg, shown = _render_capturing(_BOXEN_DOORS[door](df).to_svg, action="default")
    assert shown == [_layer_message(door)]
    assert ">a<" in svg
    assert ">b<" in svg


@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_zero_row_boxen_still_warns_no_rows_to_draw(door: str, hue: str | None):
    """With no row at all there is no category to keep and no value to blame, so the boxen
    reports a chart with no rows to draw."""
    frame = _slot_frame(_DRAWN_Z).clear()
    _, messages = _render_capturing(_BOXEN_DOORS[door](frame, hue).to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]


_SOME_NULL_Z = [3.0, None, 5.0, 6.0, None, 8.0]
# The cases where category ``z`` keeps a usable value; the last three also hold a null row.
_BOXEN_USABLE_Z_CASES = [
    pytest.param(_DRAWN_Z, None, id="all_usable"),
    pytest.param(_DRAWN_Z, "g", id="all_usable_hue"),
    pytest.param(_SOME_NULL_Z, None, id="some_null"),
    pytest.param(_SOME_NULL_Z, "g", id="some_null_hue"),
    pytest.param([None, 5.0, None, None, None, None], None, id="one_usable"),
]


@pytest.mark.parametrize("z_values,hue", _BOXEN_USABLE_Z_CASES)
@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_category_that_keeps_a_usable_value_warns_nothing(
    door: str, z_values: list[float | None], hue: str | None
):
    """Category ``z`` keeps a usable value, or all of them, so it still draws: nothing is named
    and its slot stays on the axis."""
    svg, shown = _render_capturing(
        _BOXEN_DOORS[door](_slot_frame(z_values), hue).to_svg, action="default"
    )
    assert shown == []
    assert ">z<" in svg


@pytest.mark.parametrize("z_values,hue", _BOXEN_USABLE_Z_CASES[2:])
@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_category_with_null_rows_beside_a_usable_value_draws_as_without_them(
    door: str, z_values: list[float | None], hue: str | None
):
    """The null rows of a category that keeps a usable value change nothing: the chart is byte for
    byte the one with those rows left out."""
    frame = _slot_frame(z_values)
    build = _BOXEN_DOORS[door]
    svg, _ = _render_capturing(build(frame, hue).to_svg)
    expected_svg, _ = _render_capturing(build(frame.filter(pl.col("w").is_not_null()), hue).to_svg)
    assert svg == expected_svg, "the null rows changed how a category with a usable value draws"


@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_category_with_one_null_row_keeps_its_slot_and_is_named(door: str):
    """The smallest emptied category is one row: it keeps its slot and is named."""
    chart = _BOXEN_DOORS[door](_slot_frame(_EMPTY_Z).head(13))
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [_slot_message(door)]
    assert ">z<" in svg


def test_a_horizontal_mark_boxen_empty_category_is_named_in_the_callers_terms():
    """The horizontal form puts the categories on ``y`` and the values on ``x``, so the empty slot
    reads ``c=z`` and the channel the caller wrote, ``x``: the call writes no flip of its own."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z)).mark_boxen(horizontal=True).encode(x="w", y="c")
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [_slot_message("mark_boxen", "x")]
    assert ">z<" in svg


def test_a_boxen_category_bound_to_color_is_named_once():
    """With the color on the category column too, the empty slot is the color's own group: the one
    unit is named once, by the caller's column, whichever layer of the boxen raises it."""
    chart = _slot_chart(_slot_frame(_EMPTY_Z), "mark_boxen", {"x": "c", "y": "w", "color": "c"})
    _, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [_slot_message("mark_boxen")]
    _, every = _render_capturing(chart.to_svg)
    assert set(every) == {_slot_message("mark_boxen")}


@pytest.mark.parametrize("color", [None, "g"], ids=["color_unbound", "color_on_the_level"])
@pytest.mark.parametrize("horizontal", [False, True], ids=["vertical", "horizontal"])
def test_a_boxen_color_field_level_a_null_column_empties_is_named_by_the_color_column(
    horizontal: bool, color: str | None
):
    """``color_field="g"`` makes the levels of ``g`` the boxen's categories, so the lowering carries
    ``g``, not the ``c`` the call binds to the axis, and the axis lists the levels of ``g``. The
    level ``r`` no usable value reaches keeps its slot and reads ``g=r`` in the channel the caller
    wrote for the values, named once whether or not the color is bound to ``g`` as well."""
    value, category = ("x", "y") if horizontal else ("y", "x")
    chart = (
        fm.Chart(_hue_group_frame())
        .mark_boxen(color_field="g", horizontal=horizontal)
        .encode(**{value: "w", category: "c"}, **_given(color=color))
    )
    expected = _slot_message("mark_boxen", value, "r", "g")
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [expected]
    assert axis_tick_labels(svg, axis=category) == ["p", "q", "r"]
    _, every = _render_capturing(chart.to_svg)
    assert set(every) == {expected}


# Marks that band a category on an axis and carry its column to the layers they draw. A channel
# other than ``x`` and ``y`` bound to that column is read from what each carries: a tooltip takes
# effect, and an opacity bound to a text column is refused.
_CATEGORY_BANDED_MARKS = [
    "mark_boxen",
    "mark_boxplot",
    "mark_violin",
    "mark_errorbar",
    "mark_point",
]


@pytest.mark.parametrize("mark_method", _CATEGORY_BANDED_MARKS)
def test_a_tooltip_bound_to_the_category_column_reaches_what_a_category_mark_draws(
    mark_method: str,
):
    """The boxen's statistic carries the category column, so a tooltip bound to it is read there as
    it is on the other marks that band a category: nothing is dropped, and each category's marks
    carry its text."""
    chart = _slot_chart(_slot_frame(_DRAWN_Z), mark_method, {"x": "c", "y": "w", "tooltip": "c"})
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    for category in ("a", "b", "z"):
        assert f"<title>c: {category}</title>" in svg


@pytest.mark.parametrize("mark_method", _CATEGORY_BANDED_MARKS)
def test_an_opacity_bound_to_a_text_category_column_is_refused_by_every_category_mark(
    mark_method: str,
):
    """A text column has no opacity scale, and the boxen reaches the category column as the other
    marks that band a category do, so the same refusal meets an opacity bound to it, naming the
    channel and the column."""
    chart = _slot_chart(_slot_frame(_DRAWN_Z), mark_method, {"x": "c", "y": "w", "opacity": "c"})
    with pytest.raises(ValueError, match="opacity: column 'c' has unsupported dtype: Utf8"):
        chart.to_svg()


# The category column of ``_slot_frame`` as the text a, b, z and as the integers 1, 2, 4.
_CATEGORY_FRAMES = {
    "text": _slot_frame(_DRAWN_Z),
    "integer": _slot_frame(_DRAWN_Z).with_columns(
        pl.col("c").replace_strict({"a": 1, "b": 2, "z": 4}, return_dtype=pl.Int64)
    ),
}

# What a channel bound to the category column does to a boxen of a text and of an integer
# category. ``title``, ``desc`` and ``anchor`` add that element to the unbound chart's SVG,
# ``unbound`` draws its bytes, ``scaled`` changes them through the channel's scale, ``rotated``
# adds rotations to the unbound chart's SVG (the outlier points turn), and ``refused`` raises the
# unsupported-dtype error. A binding the boxen accepts warns nothing.
_CARRIED_CHANNEL_CASES = [
    pytest.param(category, channel, effect, id=f"{category}-{channel}")
    for channel, text_effect, integer_effect in [
        ("tooltip", "title", "title"),
        ("description", "desc", "desc"),
        ("href", "anchor", "unbound"),
        ("key", "unbound", "unbound"),
        ("text", "unbound", "unbound"),
        ("opacity", "refused", "scaled"),
        ("fill_opacity", "refused", "scaled"),
        ("stroke_opacity", "refused", "scaled"),
        ("size", "refused", "scaled"),
        ("stroke_width", "refused", "scaled"),
        ("angle", "unbound", "rotated"),
    ]
    for category, effect in (("text", text_effect), ("integer", integer_effect))
]
_EFFECT_ELEMENTS = {"title": "<title>", "desc": "<desc>", "anchor": "<a "}


@pytest.mark.parametrize("category,channel,effect", _CARRIED_CHANNEL_CASES)
def test_a_channel_bound_to_a_boxens_category_column_reads_the_carried_column(
    category: str, channel: str, effect: str
):
    """The boxen's statistic carries the category column, so a channel bound to it is judged by what
    the column holds, as on any layer that draws it. A text category feeds a tooltip, a description
    and a link, leaves a key, a text label and an angle drawing as the unbound chart does, and is
    refused by the channels that read numbers (the opacity family, ``size`` and ``stroke_width``).
    An integer category feeds a tooltip and a description, leaves a link, a key and a text label
    drawing as the unbound chart does, feeds the channels that read numbers through their scales
    and turns the outlier points by an angle. Nothing the boxen accepts warns."""
    frame = _CATEGORY_FRAMES[category]
    chart = _slot_chart(frame, "mark_boxen", {"x": "c", "y": "w", channel: "c"})
    if effect == "refused":
        with pytest.raises(ValueError, match=f"{channel}: column 'c' has unsupported dtype: Utf8"):
            chart.to_svg()
        return
    unbound, _ = _render_capturing(_mark_boxen(frame).to_svg)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    if effect == "unbound":
        assert svg == unbound
    elif effect == "scaled":
        assert svg != unbound
    elif effect == "rotated":
        assert svg.count("rotate(") > unbound.count("rotate(")
    else:
        assert _EFFECT_ELEMENTS[effect] in svg
        assert _EFFECT_ELEMENTS[effect] not in unbound


def test_a_channel_bound_to_a_column_the_boxens_statistic_does_not_carry_is_still_dropped_with_a_warning():
    """The statistic carries the category column and no other column of the source. An opacity bound
    to the value column is dropped as on every statistic's layer: the chart is the unbound chart,
    and each warning says the binding was dropped. A lowering that carried every column would pass
    the table above and fail here."""
    frame = _CATEGORY_FRAMES["text"]
    unbound, _ = _render_capturing(_mark_boxen(frame).to_svg)
    chart = _slot_chart(frame, "mark_boxen", {"x": "c", "y": "w", "opacity": "w"})
    svg, shown = _render_capturing(chart.to_svg)
    assert svg == unbound
    assert shown, "the dropped binding was not reported"
    assert all("opacity 'w' was dropped" in message for message in shown)


def _scene_keys(chart: fm.Chart) -> list[list[str] | None]:
    """The ``keys`` of every mark of every panel of ``chart``'s interactive scene."""
    scene, _ = _render_capturing(lambda: chart.interactive().scene_json)
    return [mark["keys"] for panel in json.loads(scene)["panels"] for mark in panel["marks"]]


def test_a_key_bound_to_a_boxens_category_column_reaches_the_interactive_scene():
    """A key draws nothing in a static render, but the interactive scene pairs marks by it: with
    ``key="c"`` the boxen's marks carry the category of each mark (or row) and the unbound chart's
    carry none."""
    frame = _CATEGORY_FRAMES["text"]
    unbound = _scene_keys(_mark_boxen(frame))
    keyed = _scene_keys(_slot_chart(frame, "mark_boxen", {"x": "c", "y": "w", "key": "c"}))
    assert unbound and all(keys is None for keys in unbound)
    carried = [keys for keys in keyed if keys is not None]
    assert carried, "no mark of the interactive scene carries the key"
    assert {key for keys in carried for key in keys} == {"a", "b", "z"}


@pytest.mark.parametrize(
    "channel,column",
    [
        pytest.param("tooltip", "g", id="tooltip-on-the-level"),
        pytest.param("opacity", "g", id="opacity-on-the-level"),
        pytest.param("tooltip", "c", id="tooltip-on-the-x-column"),
        pytest.param("opacity", "c", id="opacity-on-the-x-column"),
    ],
)
def test_a_boxen_color_field_carries_its_own_column_to_the_channels_bound_to_it(
    channel: str, column: str
):
    """Under ``color_field="g"`` the carried column is ``g``: a tooltip bound to it draws titles
    and an opacity bound to it is refused as a text column, while the same channels bound to the
    ``c`` the call puts on ``x``, which the statistic does not carry, draw the unbound chart and
    warn that the binding was dropped."""
    frame = _CATEGORY_FRAMES["text"]
    plain = fm.Chart(frame).mark_boxen(color_field="g")
    chart = plain.encode(x="c", y="w", **{channel: column})
    if column == "g" and channel == "opacity":
        with pytest.raises(ValueError, match="opacity: column 'g' has unsupported dtype: Utf8"):
            chart.to_svg()
        return
    unbound, _ = _render_capturing(plain.encode(x="c", y="w").to_svg)
    svg, shown = _render_capturing(chart.to_svg)
    if column == "g":
        assert shown == []
        assert "<title>g: p</title>" in svg and "<title>g: q</title>" in svg
    else:
        assert svg == unbound
        assert shown, "the dropped binding was not reported"
        assert all(f"{channel} 'c' was dropped" in message for message in shown)


def test_a_channel_bound_to_the_carried_column_of_an_emptied_category_leaves_it_named_once():
    """The placeholder row of an emptied category carries its key like any row: the census names
    ``c=z`` once beside a tooltip bound to the category, and the other categories' marks carry the
    tooltip while ``z`` draws no title; an opacity bound to the text category refuses."""
    frame = _slot_frame(_EMPTY_Z)
    chart = _slot_chart(frame, "mark_boxen", {"x": "c", "y": "w", "tooltip": "c"})
    svg, shown = _render_capturing(chart.to_svg)
    assert set(shown) == {_slot_message("mark_boxen")}
    assert "<title>c: a</title>" in svg and "<title>c: b</title>" in svg
    assert "<title>c: z</title>" not in svg
    refused = _slot_chart(frame, "mark_boxen", {"x": "c", "y": "w", "opacity": "c"})
    with pytest.raises(ValueError, match="opacity: column 'c' has unsupported dtype: Utf8"):
        refused.to_svg()


# The shapes of a boxen the two lowerings below must draw alike.
_BOXEN_SHAPES: dict[str, dict[str, Any]] = {
    "plain": {},
    "hue": {"hue": "g"},
    "facet": {"facet": "g"},
    "hue_and_facet": {"hue": "g", "facet": "g"},
    "horizontal": {"horizontal": True},
    "sorted_by_name": {"sort": "descending"},
    "sorted_by_list": {"sort": ["z", "a", "b"]},
    "sorted_by_aggregate": {"sort": {"field": "w", "op": "mean", "order": "descending"}},
}
_Z_BY_STATE = {"drawn": _DRAWN_Z, "emptied": _EMPTY_Z}


def _titled_boxen(
    frame: pl.DataFrame,
    column: str,
    *,
    horizontal: bool = False,
    sort: Any = None,
    hue: str | None = None,
    facet: str | None = None,
) -> fm.Chart:
    """A boxen of ``w`` by the category column ``column`` under the axis title ``Category``, so two
    charts over differently named columns show the same text."""
    channel = fm.Y if horizontal else fm.X
    category = channel(column, title="Category", **_given(sort=sort))
    encoding = {"x": "w", "y": category} if horizontal else {"x": category, "y": "w"}
    chart = (
        fm.Chart(frame).mark_boxen(horizontal=horizontal).encode(**encoding, **_given(color=hue))
    )
    return chart if facet is None else chart.facet(col=facet)


@pytest.mark.parametrize("state", _Z_BY_STATE)
@pytest.mark.parametrize("shape", _BOXEN_SHAPES)
def test_a_boxen_draws_and_warns_the_same_whether_or_not_its_category_is_carried(
    shape: str, state: str
):
    """A category column named like one of the statistic's outputs, ``level`` here, cannot be
    carried through its ``groupby``, so the layers read the statistic's own ``group`` column
    instead. The two lowerings differ in the column the layers read and in nothing else: the same
    data under the two names draws the same bytes in every shape of boxen, with a category emptied
    or not, and warns the same text but for the column the slot is named by, ``c`` against
    ``group``."""
    frame = _slot_frame(_Z_BY_STATE[state])
    shape_kwargs = _BOXEN_SHAPES[shape]
    carried_svg, carried_shown = _render_capturing(_titled_boxen(frame, "c", **shape_kwargs).to_svg)
    uncarried = _titled_boxen(frame.rename({"c": "level"}), "level", **shape_kwargs)
    uncarried_svg, uncarried_shown = _render_capturing(uncarried.to_svg)
    assert bool(carried_shown) == (state == "emptied"), (
        "the census names the emptied category alone"
    )
    assert carried_svg == uncarried_svg
    assert all(" c=" in m for m in carried_shown), (
        "the carried slot is named by the caller's column"
    )
    assert uncarried_shown == [m.replace(" c=", " group=") for m in carried_shown]


@pytest.mark.parametrize("value", _UNDRAWABLE[2:])
@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_category_of_only_infinities_is_named_in_the_callers_terms(door: str, value: float):
    """An infinity is a value the statistic keeps, so the category has rows and nothing to place:
    it is named, in the caller's column, as the same category of nulls is."""
    svg, shown = _render_capturing(
        _BOXEN_DOORS[door](_slot_frame([value] * 6)).to_svg, action="default"
    )
    assert shown == [_slot_message(door)]
    assert ">z<" in svg


@pytest.mark.parametrize("mark_method", ["mark_boxen", "mark_boxplot", "mark_point"])
def test_a_boxen_category_under_a_custom_title_is_named_by_its_column(mark_method: str):
    """The census names columns and the axis shows the title: under ``title="Category"`` the empty
    slot still reads ``c=z``, as it does for ``mark_boxplot`` and ``mark_point``, and the axis
    reads ``Category``."""
    x = fm.X("c", title="Category")
    chart = getattr(fm.Chart(_slot_frame(_EMPTY_Z)), mark_method)().encode(x=x, y="w")
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [_slot_message(mark_method)]
    assert ">Category<" in svg


def test_a_boxen_category_named_group_renders_and_is_named_group():
    """A category column called ``group`` is named like the statistic's own output column, so it
    cannot be carried through the statistic's ``groupby``; the layers read the statistic's column,
    which holds the caller's values under the caller's own name. The chart renders and the slot
    reads ``group=z``."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z).rename({"c": "group"})).mark_boxen()
    svg, shown = _render_capturing(chart.encode(x="group", y="w").to_svg, action="default")
    assert shown == [_slot_message("mark_boxen", column="group")]
    assert ">z<" in svg


# The columns the boxen's statistic writes besides ``group``, which a category column cannot be
# named like and still be carried through its ``groupby``.
_TRANSFORM_OUTPUT_NAMES = ["level", "value", "depth", "lower", "upper", "is_outlier"]


@pytest.mark.parametrize("name", _TRANSFORM_OUTPUT_NAMES)
def test_a_boxen_category_named_like_a_transform_output_still_renders(name: str):
    """The statistic refuses a ``groupby`` field named like one of its own output columns, so such
    a category cannot be carried: the layers keep reading its ``group`` column, the chart renders
    with the caller's own name on its axis, the slot is kept, and it reads ``group=z``, a
    documented limit."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z).rename({"c": name})).mark_boxen()
    svg, shown = _render_capturing(chart.encode(x=name, y="w").to_svg, action="default")
    assert shown == [_slot_message("mark_boxen", column="group")]
    assert ">z<" in svg
    assert f">{name}<" in svg, "the axis lost the caller's name for the category"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "a category column named like one of LetterValue's output columns cannot be carried "
        "through its groupby, so its slot reads group=z; when a fix makes this pass, delete this "
        "marker and the spec's sentence that states the limit"
    ),
)
def test_a_boxen_category_named_level_is_named_in_the_callers_terms():
    """The caller's column is ``level``, so the empty slot should read ``level=z``; it reads
    ``group=z``, the statistic's own column, while the category cannot be carried."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z).rename({"c": "level"})).mark_boxen()
    _, shown = _render_capturing(chart.encode(x="level", y="w").to_svg, action="default")
    assert shown == [_slot_message("mark_boxen", column="level")]


_EMPTY_CELL_Z = [3.0, None, 5.0, None, 7.0, None]


@pytest.mark.parametrize("door", _BOXEN_DOORS)
def test_a_boxen_hue_cell_a_null_column_empties_inside_a_drawn_category_is_silent(door: str):
    """Hue level ``q`` has no usable value in category ``z`` while ``p`` still draws there: the
    category draws, so nothing is named, and the chart is byte for byte the one with the cell's
    rows left out."""
    frame = _slot_frame(_EMPTY_CELL_Z)
    build = _BOXEN_DOORS[door]
    svg, shown = _render_capturing(build(frame, "g").to_svg, action="default")
    expected_svg, _ = _render_capturing(build(frame.filter(pl.col("w").is_not_null()), "g").to_svg)
    assert shown == []
    assert ">z<" in svg
    assert svg == expected_svg


@pytest.mark.parametrize(
    "mark_method,sort,expected",
    [
        pytest.param("mark_boxen", None, ["a", "b", "d", "z"], id="boxen-unsorted"),
        pytest.param("mark_boxen", "ascending", ["a", "b", "d", "z"], id="boxen-ascending"),
        pytest.param("mark_boxen", "descending", ["z", "d", "b", "a"], id="boxen-descending"),
        pytest.param("mark_boxplot", None, ["a", "b", "z", "d"], id="boxplot-unsorted"),
    ],
)
def test_an_emptied_boxen_category_follows_the_categories_with_values_unless_sorted_by_name(
    mark_method: str, sort: str | None, expected: list[str]
):
    """The boxen's statistic keeps its placeholders after its groups with values, so ``z``, met
    third in the data, comes after ``d`` unsorted and under an ascending name sort, and a
    descending name sort puts it first, at its name's place. ``mark_boxplot``, whose placeholders
    keep their place in the data, reads ``a, b, z, d``."""
    frame = pl.DataFrame(
        {
            "c": ["a"] * 6 + ["b"] * 6 + ["z"] * 6 + ["d"] * 6,
            "w": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]
            + [7.0, 8.0, 9.0, 8.0, 7.0, 9.0]
            + [None] * 6
            + [4.0, 5.0, 6.0, 5.0, 4.0, 6.0],
        },
        schema={"c": pl.Utf8, "w": pl.Float64},
    )
    x = "c" if sort is None else fm.X("c", sort=sort)
    svg, _ = _render_capturing(getattr(fm.Chart(frame), mark_method)().encode(x=x, y="w").to_svg)
    assert axis_tick_labels(svg, axis="x") == expected


def test_a_boxen_category_a_log_axis_cuts_is_reported_by_the_log_warning_alone():
    """The boxen's statistic does not work in the axis's space, so a category whose values are all
    off a log axis's side has letter values the axis cannot place: the log warning reports them,
    the census has nothing to add, and the category keeps its slot."""
    chart = fm.Chart(_slot_frame([-1.0] * 6)).mark_boxen()
    svg, messages = _render_capturing(
        chart.encode(x="c", y=fm.Y("w", scale=fm.LogScale())).to_svg, action="default"
    )
    assert messages == [_log_message("y", "lower", 1)]
    assert ">z<" in svg


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _displot_hist(_hist_group_frame([None] * 3), hue="g"), id="histogram"),
        pytest.param(lambda: _mark_boxen(_slot_frame(_EMPTY_Z)), id="boxen"),
    ],
)
def test_an_emptied_histogram_level_and_boxen_category_reach_the_interactive_route(
    build: Callable[[], fm.Chart],
):
    """The interactive export renders a static fallback and the scene, so an emptied unit is named
    once per layer per render, twice what a static render names; the scene it ships holds no NaN
    or infinity, since the placeholder row draws nothing."""
    chart = build()
    _, static = _render_capturing(chart.to_svg)
    _, interactive = _render_capturing(lambda: chart.interactive().to_html())
    scene, _ = _render_capturing(lambda: chart.interactive().scene_json)
    assert static
    assert interactive == static * 2
    assert "NaN" not in scene and "Infinity" not in scene


# ---------------------------------------------------------------------------
# A violin's all-null category or panel: the body is the statistic's to name, the
# inner marks over it are the census's. Each text appears once per layer that owns it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inner", ["none", "box", "quartile", "point"])
def test_an_all_null_violin_category_is_named_by_its_body_and_its_inner_marks_each_once(
    inner: str,
):
    """The body's report is the statistic's, so the census names the category only for the inner
    marks laid over it: one report per inner layer (the chart's layers beside the body), never
    one more for the body, and none at all for ``inner='none'``, whose body is the sole layer."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z)).mark_violin(inner=inner).encode(x="c", y="w")
    inner_layers = len(chart.to_dict()["layers"]) - 1
    svg, shown = _render_capturing(chart.to_svg)
    assert _fit_warnings(shown, "Violin") == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert _census(shown) == [_slot_message("mark_violin")] * inner_layers
    assert (inner_layers == 0) == (inner == "none")
    assert ">z<" in svg


def test_an_all_null_violin_category_reads_once_per_text_by_default_with_its_box():
    """What a caller sees under Python's own filter: the statistic's sentence and the box's."""
    chart = fm.Chart(_slot_frame(_EMPTY_Z)).mark_violin().encode(x="c", y="w")
    _, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(shown) == sorted(
        [_fit_message("Violin", "c=z", FIT_FEW), _slot_message("mark_violin")]
    )


def test_a_catplot_violin_names_an_all_null_category_by_its_body_and_its_box():
    chart = _catplot_chart(_slot_frame(_EMPTY_Z), "violin", None)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(shown) == sorted(
        [_fit_message("Violin", "c=z", FIT_FEW), _slot_message("catplot")]
    )
    assert ">z<" in svg


def test_an_emptied_violin_panel_is_named_by_its_categories_and_by_the_boxs_panel_report():
    """Panel ``q`` holds only nulls: the statistic names each of its categories with the panel
    clause, and the box names the panel once; the body adds no panel report of its own."""
    df = _faceted_slot_frame(["q"], blank_in=("q",))
    chart = fm.Chart(df).mark_violin().encode(x="c", y="w").facet(col="f")
    _, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(_fit_warnings(shown, "Violin")) == [
        _fit_message("Violin", f"c={category}", FIT_FEW, panel="f=q") for category in "abz"
    ]
    assert _census(shown) == [_panel_message("mark_violin")]
