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
skip; a group an explicit domain or a log scale's off side already reports, a
one-point series, a group a statistic's own transform warning
(``DegenerateFit``, ``IntervalUndefined``) already names by the same group,
and a histogram hue group a null column emptied also keep their own,
unchanged reports.

A layer is named in the channels its caller wrote. A layer a call built (a
composite mark, a figure function) bound its secondary positional channel
itself, so it names ``x2`` as ``x`` and ``y2`` as ``y``; a layer the caller
placed keeps every channel it wrote.

One figure is the exception. ``catplot`` lowers a horizontal call
(``orient="h"``) to ``x=category, y=value`` plus ``CoordFlip``, so it names the
channels of that lowering: a value the caller bound to ``x`` reads ``y``, as
its explicit-domain and log-scale warnings already read. The strict
expected-failure tests that expect the caller's ``x`` keep that gap visible; a
fix turns them into failures that say to delete their marker.
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Callable, Literal

import polars as pl
import pytest

import ferrum as fm

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


def test_a_histogram_hue_group_a_null_column_empties_stays_silent():
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, None, None, None, None],
            "h": ["a", "a", "a", "a", "c", "c", "c", "c"],
        }
    )
    svg, messages = _render_capturing(lambda: fm.displot(df, x="v", kind="hist", hue="h").to_svg())
    assert messages == []
    assert ">c<" not in svg


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
        "it degraded to an empty/flat output instead of failing the whole chart"
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
                "it degraded to an empty/flat output instead of failing the whole chart",
                "layer 1: mark_line: every row of g=a has a null or non-finite y, so it draws nothing",
            ],
        ),
        (
            _kde_and_ecdf_charts,
            [
                "Kde: could not fit group h=c (fewer than 2 usable (non-null/non-NaN) values); it "
                "degraded to an empty/flat output instead of failing the whole chart",
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
        "Smooth: could not fit the data (fewer than 2 usable (non-null/non-NaN) x/y points); it "
        "degraded to an empty/flat output instead of failing the whole chart",
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
        "could not be computed for this data); it degraded to an empty/flat output "
        "instead of failing the whole chart",
        "Smooth: could not fit group gi=2 in facet panel sp=p2 (the poly(degree=3) fit "
        "could not be computed for this data); it degraded to an empty/flat output "
        "instead of failing the whole chart",
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


def _slot_message(entry: str, channels: str = "y", category: str = "z", column: str = "c") -> str:
    """The text that names the empty slot ``<column>=<category>`` on layer ``entry``."""
    return (
        f"{entry}: every row of {column}={category} has a null or non-finite {channels}, "
        "so it draws nothing"
    )


def _census(messages: list[str]) -> list[str]:
    """The null-position messages among ``messages`` (each carries ``null or
    non-finite``), leaving out an unrelated notice a figure raises beside them."""
    return [m for m in messages if "null or non-finite" in m]


def _slot_chart(df: pl.DataFrame, mark_method: str, encoding: dict) -> fm.Chart:
    return getattr(fm.Chart(df), mark_method)().encode(**encoding)


def _catplot_chart(df: pl.DataFrame, kind: str, hue: str | None) -> fm.Chart:
    return fm.catplot(df, x="c", y="w", kind=kind, hue=hue)


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
    assert shown == [expected]
    assert ">z<" in svg
    _, every = _render_capturing(chart.to_svg)
    assert set(every) == {expected}


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
    assert shown == [_slot_message(mark_method, "x")]
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
    assert set(emptied) == {_slot_message(mark_method)}


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
