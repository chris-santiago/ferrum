"""A group, facet panel or whole layer whose drawn position is null on
every row names itself instead of drawing nothing in silence.

When every row of a legend category, a ``detail``/``shape``/``stroke_dash``
series, a facet panel's share of a layer, or an entire unfaceted layer has a
null or non-finite value on a channel the mark draws its position from, the
render now warns once per layer through ``RenderWarning::GroupsWithNoPosition``,
voiced the way ``RenderWarning::SeriesDropped`` is. The legend keeps the
named group's entry. A row inside a group that still draws some rows stays
the documented, silent per-row null skip; a group an explicit domain or a
log scale's off side already reports, a one-point series, a group a
statistic's own transform warning (``DegenerateFit``, ``IntervalUndefined``)
already names by the same group, and a histogram hue group a null column
emptied also keep their own, unchanged reports.
"""

from __future__ import annotations

import warnings
from typing import Callable

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


def _render_capturing(render: Callable[[], str]) -> tuple[str, list[str]]:
    """Render inside a warning capture: the output and every warning raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render()
    return out, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# A group whose drawn position is null on every row is named once, and the
# legend still lists it, on every mark that reads a per-row position.
# ---------------------------------------------------------------------------


def test_a_group_with_no_placed_row_is_named_once():
    chart = fm.Chart(_TWO_GROUP_ONE_NULL).mark_line().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_line: every row of g=c has a null or non-finite y, so it draws nothing"]
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
        expected = "mark_ribbon: every row of g=c has a null or non-finite y and y2, so it draws nothing"
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
    svg, messages = _render_capturing(
        lambda: fm.displot(df, x="v", kind="ecdf", hue="h").to_svg()
    )
    assert messages == ["displot: every row of h=c has a null or non-finite x, so it draws nothing"]
    assert ">c<" in svg
    assert ">a<" in svg


@pytest.mark.parametrize("group_channel", ["shape", "stroke_dash"])
def test_a_point_mark_names_a_null_group_by_shape_or_stroke_dash(group_channel: str):
    chart = fm.Chart(_TWO_GROUP_ONE_NULL).mark_point().encode(x="x", y="y", **{group_channel: "g"})
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_point: every row of g=c has a null or non-finite y, so it draws nothing"]
    assert ">c<" in svg


def test_a_line_mark_names_a_null_group_by_detail():
    chart = fm.Chart(_TWO_GROUP_ONE_NULL).mark_line().encode(x="x", y="y", detail="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_line: every row of g=c has a null or non-finite y, so it draws nothing"]


@pytest.mark.parametrize("mark_method", ["mark_point", "mark_line"])
def test_a_binned_color_bin_with_no_placed_row_is_named(mark_method: str):
    """A binned color legend entry is a group on every mark that draws
    one, read the same way the connected marks already read it, not just
    on ``mark_line``."""
    df = pl.DataFrame({"v": [10.0, 20.0, 90.0, 95.0], "y": [1.0, 2.0, None, None]})
    chart = getattr(fm.Chart(df), mark_method)().encode(
        x="v", y="y", color=fm.Color("v", bin=True)
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [f"{mark_method}: every row of v=90 has a null or non-finite y, so it draws nothing"]


def test_a_null_group_key_is_named_the_way_a_null_is_rendered():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, None, None], "g": ["a", "a", None, None]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_point: every row of g=null has a null or non-finite y, so it draws nothing"]


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
    chart = fm.Chart(df).mark_bar().encode(
        x=fm.X("x", scale=fm.BandScale(domain=["p", "q"])), y="y", color="g"
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_bar: every row of g=c has a null or non-finite x, so it draws nothing"]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
def test_nan_and_inf_in_the_drawn_column_count_as_null(value: float):
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [value, value], "g": ["c", "c"]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_point: every row of g=c has a null or non-finite y, so it draws nothing"]


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
    assert messages == ["mark_point: every row of g=c has a null or non-finite y, so it draws nothing"]


# ---------------------------------------------------------------------------
# A whole layer, and a facet panel's share of a layer.
# ---------------------------------------------------------------------------


def test_a_layer_with_no_placed_row_says_so():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [None, None, None]}).cast({"y": pl.Float64})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_point: every row has a null or non-finite y, so the layer draws nothing"]


def test_a_one_row_layer_with_a_null_y_and_no_grouping_says_so_too():
    df = pl.DataFrame({"x": [1.0], "y": [None]}).cast({"y": pl.Float64})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_point: every row has a null or non-finite y, so the layer draws nothing"]


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
    return fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})).mark_point().encode(
        x="x", y="y"
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
    assert messages == ["mark_line: every row of g=c has a null or non-finite y, so it draws nothing"]


def test_two_null_leaves_in_one_composite_warn_once_each():
    """Two leaves, each with its own all-null group, warn twice -- each
    leaf reports its own layer, not deduplicated into one warning."""
    _, messages = _render_capturing((_null_group_line_chart() | _null_group_line_chart()).to_svg)
    expected = "mark_line: every row of g=c has a null or non-finite y, so it draws nothing"
    assert messages == [expected, expected]


def test_a_layer_chart_leaf_names_its_own_drop():
    chart = fm.LayerChart(_base_point_chart(), _null_group_line_chart())
    _, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_line: every row of g=c has a null or non-finite y, so it draws nothing"]


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
    expected_series = (
        "mark_line: mark_line() joins two or more points per series, so it did not draw g=b (one point)"
    )
    assert group_messages == [expected_group] * len(group_messages)
    assert series_messages == [expected_series] * len(series_messages)
    assert len(group_messages) == len(series_messages)
    assert len(group_messages) > 0


def test_chart_to_html_warns_once():
    """``Chart.to_html()`` (no ``.interactive()`` first) renders the scene
    once, so its warning count is one, unlike the double-rendering
    ``.interactive().to_html()`` route above."""
    _, messages = _render_capturing(_null_group_line_chart().to_html)
    assert messages == ["mark_line: every row of g=c has a null or non-finite y, so it draws nothing"]


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
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1.0, None, 3.0], "g": ["a", "a", "a"]}
    )
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
    chart = fm.Chart(df).mark_point().encode(
        x=fm.X("x", scale=fm.LinearScale(domain=[0.0, 3.0])), y="y", color="g"
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
    svg, messages = _render_capturing(
        lambda: fm.displot(df, x="v", kind="hist", hue="h").to_svg()
    )
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
    svg, messages = _render_capturing(
        lambda: fm.displot(df, x="v", kind="kde", hue="h").to_svg()
    )
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
    assert messages == ["the chart has no rows to draw after its transforms; its panel is drawn empty"]


def test_a_wrong_type_column_refuses_as_it_already_did():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    with pytest.raises(ValueError, match="unsupported dtype"):
        fm.Chart(df).mark_point().encode(x="x", y="y:Q").to_svg()


# ---------------------------------------------------------------------------
# End to end through the public API: log-axis infinities, mixed-channel and
# null-keyed groups, cross-layer ownership, and a fit's row-coverage over
# every layer and name it degrades.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,label", [(float("-inf"), "-inf"), (float("inf"), "inf"), (float("nan"), "nan")], ids=["-inf", "inf", "nan"])
def test_an_infinity_on_a_log_axis_is_left_to_its_own_report(value: float, label: str):
    """On a log y, group ``c``'s own report differs by value: a negative
    value is reported by the log side's own warning, a value outside the
    scale's explicit domain by the domain warning, and only a true NaN (no
    axis report of its own) falls to the census."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, value, value], "g": ["a", "a", "c", "c"]})
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
        assert messages == ["mark_point: every row of g=c has a null or non-finite y, so it draws nothing"]


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
    assert messages == [f"{mark_method}: every row of g=a has a null or non-finite x and y, so it draws nothing"]


def test_a_group_null_on_y_everywhere_names_only_y():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [None, None, 3.0, 4.0], "g": ["a", "a", "b", "b"]}
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == ["mark_point: every row of g=a has a null or non-finite y, so it draws nothing"]


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
        {"x": [None, None, None, None], "y": [None, None, None, None], "g": [None, None, None, None]}
    ).cast({"x": pl.Float64, "y": pl.Float64, "g": pl.Utf8})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "mark_point: every row of g=null has a null or non-finite x and y, so it draws nothing"
    ]


def _smooth_and_aggregate_line_charts() -> tuple["fm.Chart", "fm.Chart"]:
    df_a = pl.DataFrame({"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, None, None], "g": ["a", "a", "a", "a"]})
    df_b = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [None, None, None], "g": ["a", "a", "a"]})
    layer_a = fm.Chart(df_a).transform(fm.Smooth(x="x", y="y", groupby=["g"])).mark_line().encode(
        x="x", y="y", color="g"
    )
    layer_b = fm.Chart(df_b).mark_line().encode(x="x", y=fm.Y("y", aggregate="mean"), color="g")
    return layer_a, layer_b


def _kde_and_ecdf_charts() -> tuple["fm.Chart", "fm.Chart"]:
    df_kde = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, None, None, None], "h": ["a", "a", "a", "a", "c", "c", "c", "c"]}
    )
    df_ecdf = pl.DataFrame(
        {"v": [5.0, 6.0, 7.0, 8.0, None, None, None, None], "h": ["a", "a", "a", "a", "c", "c", "c", "c"]}
    )
    return fm.displot(df_kde, x="v", kind="kde", hue="h"), fm.displot(df_ecdf, x="v", kind="ecdf", hue="h")


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
