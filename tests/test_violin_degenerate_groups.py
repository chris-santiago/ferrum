"""A density statistic names a group it cannot estimate: ``Kde``, ``Violin`` and ``QQ``.

A density or violin group, hue level or facet panel whose values cannot be estimated (no usable
value, one value or an infinite one, tied values, a zero density) has no body to draw. The
statistic keeps its slot, legend entry or panel and names it once through ``DegenerateFit``.
``Kde`` and ``Violin`` give one reason for one cause, in one order: fewer than two usable values,
an infinity, tied values. A ``Kde`` group with no usable value takes no position, so it never
widens the axis. ``QQ`` does the same for an all-null group or facet panel, and for a call with
no usable value at all: it warns and draws its empty frame on every door. A chart that estimates
every group is byte-identical and warns nothing. Rows for a report shared with the render's census
(the violin's inner box layers) are in ``test_null_group_warning.py``.
"""

from __future__ import annotations

import re
from typing import Callable

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import AxisExtent, axis_tick_labels, x_axis_extents, y_axis_extents
from tests._svg_rects import mark_layer
from tests.test_null_group_warning import (
    FIT_BANDWIDTH,
    FIT_FEW,
    FIT_NONE,
    FIT_TIED,
    _NO_FACET_ROWS_WARNING,
    _NO_ROWS_TO_DRAW,
    _all_keys_null_message,
    _census,
    _fit_message,
    _facet_key_dropped_message,
    _fit_warnings,
    _null_key_dropped_message,
    _panel_fit_message,
    _render_capturing,
    _small_slot_frame,
    _whole_fit_message,
)

_ZERO_DENSITY = "its density is zero across the evaluation range"

# Category ``z`` is the one the statistic cannot estimate; ``a`` and ``b`` hold two rows each.
_Z_CASES = [
    pytest.param([3.0], FIT_FEW, id="one"),
    pytest.param([3.0, 3.0, 3.0, 3.0], FIT_TIED, id="tied"),
    pytest.param([None, None, None], FIT_FEW, id="null"),
]
_INNERS = ["none", "box", "quartile", "point"]
# (door, inner): the figure draws the default inner box and has no knob for it.
_DOORS = [pytest.param("mark", inner, id=f"mark-{inner}") for inner in _INNERS] + [
    pytest.param("catplot", "box", id="catplot")
]


def _violin_door(door: str, df: pl.DataFrame, inner: str, hue: str | None = None) -> fm.Chart:
    """The violin of ``w`` by category ``c`` through ``door``: the mark method or the figure.

    The figure draws the default inner box and has no ``inner`` knob, so ``_DOORS`` pairs it
    with ``"box"`` alone. A horizontal catplot is not built here: it still names the
    lowering's channel."""
    if door == "mark":
        chart = fm.Chart(df).mark_violin(inner=inner)
        return chart.encode(x="c", y="w", **({} if hue is None else {"color": hue}))
    return fm.catplot(df, x="c", y="w", kind="violin", **({} if hue is None else {"hue": hue}))


def _faceted_violin(door: str, inner: str, df: pl.DataFrame) -> fm.Chart:
    if door == "mark":
        return fm.Chart(df).mark_violin(inner=inner).encode(x="c", y="w").facet(col="f")
    return fm.catplot(df, x="c", y="w", kind="violin", col="f")


def _hue_frame() -> pl.DataFrame:
    """Categories ``a`` and ``b`` of hue levels ``p`` (four usable values) and ``q`` (all null)."""
    rows = [
        (category, level, None if level == "q" else value)
        for category in ("a", "b")
        for level, values in (("p", (1.0, 2.0, 3.0, 4.0)), ("q", (5.0, 6.0, 7.0, 8.0)))
        for value in values
    ]
    return pl.DataFrame(rows, schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64}, orient="row")


def _panel_frame() -> pl.DataFrame:
    """Facet panels ``p`` (categories ``a`` and ``b`` of four values) and ``q`` (all null)."""
    rows = [
        (panel, category, None if panel == "q" else value)
        for panel in ("p", "q")
        for category in ("a", "b")
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    return pl.DataFrame(rows, schema={"f": pl.Utf8, "c": pl.Utf8, "w": pl.Float64}, orient="row")


# ---------------------------------------------------------------------------
# Violin: a category it cannot estimate is named by its own statistic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("door,inner", _DOORS)
@pytest.mark.parametrize("z_values,reason", [c for c in _Z_CASES if c.id != "null"])
def test_a_violin_group_it_cannot_estimate_is_named(
    z_values: list[float | None], reason: str, inner: str, door: str
):
    """One value, or values all equal, give a category no density. It keeps its slot, the
    statistic names it once whatever the inner mark, and the inner marks, which have a value
    to draw, add no census report."""
    chart = _violin_door(door, _small_slot_frame(z_values), inner)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _fit_warnings(shown, "Violin") == [_fit_message("Violin", "c=z", reason)]
    assert _census(shown) == []
    assert ">z<" in svg


def test_an_all_null_violin_category_keeps_its_slot_and_is_named_once_by_violin():
    """With ``inner='none'`` the body is the only layer over the category, so the statistic names
    it and the census, which defers to it by lineage, adds nothing on the body. (The figure
    always draws an inner box, so its census rows are in ``test_null_group_warning.py``.)"""
    chart = _violin_door("mark", _small_slot_frame([None, None, None]), "none")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg


@pytest.mark.parametrize("inner", _INNERS[1:])
def test_an_all_null_violin_category_is_named_by_violin_whatever_the_inner_mark(inner: str):
    """The statistic's warning does not depend on which inner marks sit over the category."""
    chart = _violin_door("mark", _small_slot_frame([None, None, None]), inner)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _fit_warnings(shown, "Violin") == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg


@pytest.mark.parametrize("door,inner", [_DOORS[0], _DOORS[1], _DOORS[-1]])
def test_a_violin_hue_group_a_null_column_empties_is_named(door: str, inner: str):
    """Hue level ``q`` is null in every category: each category's ``q`` group is named once,
    and the legend keeps the level."""
    chart = _violin_door(door, _hue_frame(), inner, hue="g")
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(_fit_warnings(shown, "Violin")) == [
        _fit_message("Violin", f"c={category}, g=q", FIT_FEW) for category in "ab"
    ]
    assert ">q<" in svg
    if inner == "none":
        assert _census(shown) == []


@pytest.mark.parametrize("door,inner", [_DOORS[0], _DOORS[-1]])
def test_a_violin_facet_panel_a_null_column_empties_is_named(door: str, inner: str):
    """Panel ``q`` holds only nulls: each of its categories is named once with the panel clause,
    and the panel is still drawn (its strip title stays)."""
    svg, shown = _render_capturing(
        _faceted_violin(door, inner, _panel_frame()).to_svg, action="default"
    )
    assert sorted(_fit_warnings(shown, "Violin")) == [
        _fit_message("Violin", f"c={category}", FIT_FEW, panel="f=q") for category in "ab"
    ]
    assert ">q<" in svg
    if inner == "none":
        assert _census(shown) == []


# ---------------------------------------------------------------------------
# What does not change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z_values,reason", _Z_CASES)
def test_a_density_mark_names_the_same_cause_as_the_violin(
    z_values: list[float | None], reason: str
):
    """``mark_density`` reads the same estimate through ``Kde``, which gives the reason the
    violin gives for the same data: tied values read ``its usable values are all equal`` on both."""
    chart = fm.Chart(_small_slot_frame(z_values)).mark_density().encode(x="w", color="c")
    _, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Kde", "c=z", reason)]


@pytest.mark.parametrize("door,inner", _DOORS)
def test_a_well_formed_violin_warns_nothing(door: str, inner: str):
    df = pl.DataFrame(
        {
            "c": ["a"] * 4 + ["b"] * 4 + ["z"] * 4,
            "w": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 6.0, 3.0, 4.0, 5.0, 8.0],
        }
    )
    svg, shown = _render_capturing(_violin_door(door, df, inner).to_svg, action="always")
    assert shown == []
    assert ">z<" in svg


# ---------------------------------------------------------------------------
# QQ: an all-null hue group or facet panel.
# ---------------------------------------------------------------------------


def _qq_hue_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, None, None, None], "g": ["a"] * 4 + ["c"] * 4},
        schema={"v": pl.Float64, "g": pl.Utf8},
    )


def _qq_panel_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, None, None, None], "f": ["p"] * 4 + ["q"] * 4},
        schema={"v": pl.Float64, "f": pl.Utf8},
    )


def test_a_qq_group_a_null_column_empties_is_named():
    """Hue group ``c`` has no usable value: the statistic names it once and the legend keeps it."""
    chart = fm.Chart(_qq_hue_frame()).mark_qq().encode(x="v", color="g")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("QQ", "g=c", FIT_NONE)]
    assert ">c<" in svg


def test_a_qq_facet_panel_a_null_column_empties_is_named():
    """An all-null panel draws empty and is named, instead of failing the whole faceted chart."""
    chart = fm.Chart(_qq_panel_frame()).mark_qq().encode(x="v").facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_panel_fit_message("QQ", FIT_NONE)]
    assert ">p<" in svg and ">q<" in svg


def test_a_well_formed_qq_warns_nothing():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 5.0, 8.0], "g": ["a"] * 4 + ["c"] * 4})
    chart: Callable[[], str] = fm.Chart(df).mark_qq().encode(x="v", color="g").to_svg
    assert _render_capturing(chart, action="always")[1] == []


def test_a_violin_group_holding_an_infinity_is_named_in_kdes_non_finite_wording():
    """An infinite value leaves a group with no finite extent to estimate over. The category keeps
    its slot and the statistic names it, with the cause ``Kde`` gives the same data."""
    df = pl.DataFrame(
        {"c": ["a"] * 4 + ["z"] * 4, "w": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, float("inf"), 3.0]}
    )
    reason = FIT_BANDWIDTH
    svg, shown = _render_capturing(_violin_door("mark", df, "none").to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", reason)]
    assert ">z<" in svg
    _, kde = _render_capturing(
        fm.Chart(df).mark_density().encode(x="w", color="c").to_svg, action="always"
    )
    assert kde == [_fit_message("Kde", "c=z", reason)]


def test_a_qq_facet_panel_that_is_emptied_in_every_group_names_each_group():
    """With a hue the panel holds one group per level, so the statistic names each emptied group
    (with the panel clause) instead of the panel; the panel and its legend entries stay."""
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, None, None, None, None] * 2,
            "f": (["p"] * 4 + ["q"] * 4) * 2,
            "g": ["a"] * 8 + ["b"] * 8,
        },
        schema={"v": pl.Float64, "f": pl.Utf8, "g": pl.Utf8},
    )
    chart = fm.Chart(df).mark_qq().encode(x="v", color="g").facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert sorted(shown) == [
        _fit_message("QQ", f"g={level}, f=q", FIT_NONE, panel="f=q") for level in ("a", "b")
    ]
    assert ">q<" in svg


def test_a_violin_category_of_nan_values_is_named_like_a_null_one():
    """A NaN is no usable value, as a null is: the category keeps its slot and is named."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["z"] * 3, "w": [1.0, 2.0, 3.0, 4.0] + [float("nan")] * 3})
    svg, shown = _render_capturing(_violin_door("mark", df, "none").to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg


def test_a_violin_whose_density_is_zero_everywhere_names_each_group():
    """A bandwidth so small every density underflows to zero leaves no body to draw; each category
    is named with the cause, and keeps its slot."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["b"] * 4, "w": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]})
    chart = fm.Chart(df).mark_violin(inner="none", bandwidth=1e-320).encode(x="c", y="w")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert sorted(shown) == [_fit_message("Violin", f"c={c}", _ZERO_DENSITY) for c in "ab"]
    assert ">a<" in svg and ">b<" in svg


def test_a_violin_group_of_one_infinite_value_is_named_with_the_reason_kde_gives():
    """One infinite value is one value, not a degenerate bandwidth: the count is the cause,
    as ``Kde`` reads the same data."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["z"], "w": [1.0, 2.0, 3.0, 4.0, float("inf")]})
    svg, shown = _render_capturing(_violin_door("mark", df, "none").to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg
    _, kde = _render_capturing(
        fm.Chart(df).mark_density().encode(x="w", color="c").to_svg, action="always"
    )
    assert kde == [_fit_message("Kde", "c=z", FIT_FEW)]


def test_a_qq_hue_group_of_nan_values_is_named_like_a_null_one():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0] + [float("nan")] * 4, "g": ["a"] * 4 + ["c"] * 4})
    svg, shown = _render_capturing(
        fm.Chart(df).mark_qq().encode(x="v", color="g").to_svg, action="always"
    )
    assert shown == [_fit_message("QQ", "g=c", FIT_NONE)]
    assert ">c<" in svg


# ---------------------------------------------------------------------------
# Kde and Violin: one reason for one cause, whatever the door or the bandwidth.
# ---------------------------------------------------------------------------

_LEVEL_BASE = [100.0, 101.0, 103.0, 104.0, 106.0]
_TIED_GROUPS = pl.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "v": [1.0, 2.0, 3.0, 4.0] + [3.0] * 4})


def _level_frame(z_values: list[float | None]) -> pl.DataFrame:
    """Level ``a`` over 100 to 106, then level ``z`` holding ``z_values`` (none for no level)."""
    return pl.DataFrame(
        {"g": ["a"] * len(_LEVEL_BASE) + ["z"] * len(z_values), "v": _LEVEL_BASE + z_values},
        schema={"g": pl.Utf8, "v": pl.Float64},
    )


def _x_ticks(svg: str) -> list[str]:
    return axis_tick_labels(svg, axis="x")


def _body_widths(svg: str) -> list[float]:
    """The horizontal extent in pixels of every path the marks draw."""
    widths = []
    for d in re.findall(r'<path[^>]*\bd="([^"]+)"', mark_layer(svg)):
        xs = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)[0::2]]
        widths.append(max(xs) - min(xs))
    return widths


def _tied_door(door: str, df: pl.DataFrame) -> tuple[fm.Chart, str]:
    """The density of ``w`` by category ``c`` through ``door``, and the statistic it runs."""
    if door == "mark_density":
        return fm.Chart(df).mark_density().encode(x="w", color="c"), "Kde"
    if door == "displot":
        return fm.displot(df, x="w", hue="c", kind="kde"), "Kde"
    return _violin_door("mark", df, "none"), "Violin"


@pytest.mark.parametrize("door", ["mark_density", "displot", "mark_violin"])
def test_a_tied_group_reads_one_reason_on_every_density_door(door: str):
    """Values all equal are the cause whichever statistic meets them, so ``mark_density``,
    ``displot(kind='kde')`` and ``mark_violin`` word it alike, and keep the group's slot."""
    chart, statistic = _tied_door(door, _small_slot_frame([3.0, 3.0, 3.0, 3.0]))
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message(statistic, "c=z", FIT_TIED)]
    assert ">z<" in svg


def test_an_ungrouped_density_of_tied_values_is_named_with_the_tied_reason():
    chart = fm.Chart(pl.DataFrame({"v": [3.0] * 4})).mark_density().encode(x="v")
    _, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_whole_fit_message("Kde", FIT_TIED)]


def test_a_tied_group_under_a_fixed_bandwidth_with_no_wider_range_is_named():
    """A caller's bandwidth is always usable; with no range wider than the tie nothing can be
    drawn, so the group is named, keeps its legend entry and leaves no zero-width sliver."""
    chart = fm.Chart(_TIED_GROUPS).mark_density(bandwidth=0.5).encode(x="v", color="g")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Kde", "g=b", FIT_TIED)]
    assert ">b<" in svg
    assert _body_widths(svg) and min(_body_widths(svg)) > 1.0


def test_an_ungrouped_tied_density_under_a_fixed_bandwidth_is_named():
    chart = fm.Chart(pl.DataFrame({"v": [3.0] * 4})).mark_density(bandwidth=0.5).encode(x="v")
    _, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_whole_fit_message("Kde", FIT_TIED)]


def test_a_tied_group_under_a_fixed_bandwidth_and_a_wide_extent_still_draws_its_bump():
    """The control for the rows above: a range with width gives the same bandwidth a body to
    draw, so the group is not degraded and nothing is warned."""
    chart = (
        fm.Chart(_TIED_GROUPS)
        .mark_density(bandwidth=0.5, extent=(0.0, 6.0))
        .encode(x="v", color="g")
    )
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == []
    assert len(_body_widths(svg)) == 2 and min(_body_widths(svg)) > 100.0


@pytest.mark.parametrize("multiple", ["stack", "fill"])
def test_a_tied_group_under_a_shared_extent_and_a_fixed_bandwidth_still_draws_its_bump(
    multiple: str,
):
    """Stacked or filled groups share one range across the chart, wider than the tie, so the tied
    group draws its bump (two bodies) and nothing is warned."""
    chart = (
        fm.Chart(_TIED_GROUPS)
        .mark_density(bandwidth=0.5, multiple=multiple)
        .encode(x="v", color="g")
    )
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == []
    assert len(_body_widths(svg)) == 2 and min(_body_widths(svg)) > 100.0


def test_a_tied_facet_panel_under_a_fixed_bandwidth_draws_on_the_frames_extent():
    """A facet pins every panel's range to the frame's, which is wider than the tie, so a tied
    panel still draws its bump and warns nothing."""
    df = pl.DataFrame({"f": ["p"] * 4 + ["q"] * 4, "v": [1.0, 2.0, 3.0, 4.0] + [3.0] * 4})
    chart = fm.Chart(df).mark_density(bandwidth=0.5).encode(x="v").facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == []
    assert len(_body_widths(svg)) == 2 and min(_body_widths(svg)) > 50.0


def test_a_displot_group_holding_an_infinity_is_named_with_the_non_finite_reason():
    """``displot(kind='kde')`` reads the same statistic: an infinite value in a group names the
    bandwidth reason through the figure too."""
    df = pl.DataFrame(
        {"c": ["a"] * 4 + ["z"] * 4, "w": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, float("inf"), 3.0]}
    )
    svg, shown = _render_capturing(
        fm.displot(df, x="w", hue="c", kind="kde").to_svg, action="always"
    )
    assert shown == [_fit_message("Kde", "c=z", FIT_BANDWIDTH)]
    assert ">z<" in svg


def test_a_bandwidth_that_overflows_to_infinity_is_named_with_the_bandwidth_reason():
    """A usable data spread scaled by an enormous ``bw_adjust`` resolves to an infinite bandwidth,
    which the statistic names, where a modest spread under the same factor still draws."""
    wide = pl.DataFrame({"v": [float(i) for i in range(1, 101)]})
    _, shown = _render_capturing(
        fm.Chart(wide).mark_density(bw_adjust=1e308).encode(x="v").to_svg, action="always"
    )
    assert shown == [_whole_fit_message("Kde", FIT_BANDWIDTH)]


def test_a_group_holding_an_infinity_under_a_fixed_bandwidth_is_named_not_left_to_the_census():
    """The bandwidth resolved fine and the range did not: the statistic names the group (with the
    wording an infinite value has under the default bandwidth) instead of the census blaming null
    or non-finite x."""
    chart = (
        fm.Chart(_level_frame([102.0, float("inf"), 103.0, 104.0]))
        .mark_density(bandwidth=0.5)
        .encode(x="v", color="g")
    )
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Kde", "g=z", FIT_BANDWIDTH)]
    assert ">z<" in svg


@pytest.mark.parametrize("bandwidth", [0, -1.0, float("nan"), float("inf")])
def test_a_density_refuses_an_unusable_fixed_bandwidth_before_any_group_is_fitted(bandwidth: float):
    """A zero, negative or non-finite fixed bandwidth never reaches a group: the knob refuses."""
    with pytest.raises(ValueError, match="numeric bandwidth must be a positive finite number"):
        fm.Chart(_TIED_GROUPS).mark_density(bandwidth=bandwidth).encode(x="v").to_svg()


def test_a_violin_tied_under_a_fixed_bandwidth_still_degrades():
    """A violin needs a range to scale its ring, so a tie degrades it under any bandwidth."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["b"] * 4, "w": [1.0, 2.0, 3.0, 4.0] + [3.0] * 4})
    chart = fm.Chart(df).mark_violin(inner="none", bandwidth=0.5).encode(x="c", y="w")
    _, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=b", FIT_TIED)]


# ---------------------------------------------------------------------------
# A Kde group with no usable value adds no position to the axis.
# ---------------------------------------------------------------------------

_ALL_NULL_LEVEL_DOORS = {
    "mark_density": lambda df: fm.Chart(df).mark_density().encode(x="v", color="g"),
    "mark_density_log": lambda df: (
        fm.Chart(df).mark_density().encode(x=fm.X("v", scale=fm.LogScale()), color="g")
    ),
    "displot": lambda df: fm.displot(df, x="v", hue="g", kind="kde"),
    "displot_log": lambda df: fm.displot(df, x="v", hue="g", kind="kde", log_scale=True),
}


@pytest.mark.parametrize("blank", [None, float("nan")], ids=["null", "nan"])
@pytest.mark.parametrize("door", sorted(_ALL_NULL_LEVEL_DOORS))
def test_an_all_null_density_level_adds_no_position_to_the_axis(door: str, blank: float | None):
    """The level is named once and keeps its legend entry, and the x ticks are those of the
    data alone (starting at 100), not the 0 (or the 1 of a log axis) its empty grid used to land at."""
    build = _ALL_NULL_LEVEL_DOORS[door]
    svg, shown = _render_capturing(build(_level_frame([blank, blank])).to_svg, action="always")
    alone = build(_level_frame([])).to_svg()
    assert _x_ticks(alone)[0] == "100"
    assert _x_ticks(svg) == _x_ticks(alone)
    assert shown == [_fit_message("Kde", "g=z", FIT_FEW)]
    assert ">z<" in svg


@pytest.mark.parametrize("position", ["density_first", "point_first"])
def test_an_all_null_density_layered_with_a_point_leaves_the_axis_to_the_points(position: str):
    """An ungrouped all-null density has no position to contribute, so the shared x axis is the
    point layer's alone, whichever operand comes first."""
    points = fm.Chart(_level_frame([])).mark_point().encode(x="v", y="v")
    density = fm.Chart(pl.DataFrame({"v": [None, None]}, schema={"v": pl.Float64}))
    density = density.mark_density().encode(x="v")
    chart = density + points if position == "density_first" else points + density
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert _x_ticks(svg) == _x_ticks(points.to_svg())
    assert _x_ticks(svg)[0] == "100"
    assert shown == [_whole_fit_message("Kde", FIT_FEW)]


@pytest.mark.parametrize(
    "z_values,reason",
    [
        pytest.param([102.0] * 4, FIT_TIED, id="tied"),
        pytest.param([102.0, float("inf"), 103.0], FIT_BANDWIDTH, id="infinity"),
        pytest.param([102.0], FIT_FEW, id="one-value"),
    ],
)
def test_a_degraded_density_group_with_a_value_stays_inside_the_data(
    z_values: list[float | None], reason: str
):
    """Only a group with no usable value invents no position; a tied group, one with an infinity
    and a one-value group keep the data's own ticks."""
    chart = fm.Chart(_level_frame(z_values)).mark_density().encode(x="v", color="g")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Kde", "g=z", reason)]
    assert _x_ticks(svg) == _x_ticks(
        fm.Chart(_level_frame([])).mark_density().encode(x="v", color="g").to_svg()
    )


def test_a_one_value_density_group_far_from_the_data_still_widens_the_axis():
    """A lone value is a position the caller wrote, so its range stays at that value."""
    chart = fm.Chart(_level_frame([500.0])).mark_density().encode(x="v", color="g")
    svg, _ = _render_capturing(chart.to_svg, action="always")
    assert _x_ticks(svg)[-1] == "500"


def test_an_all_null_density_facet_panel_keeps_the_frames_extent():
    """A facet pins each panel's range to the frame's, so an all-null panel never reaches 0."""
    values = [float(v) for v in _LEVEL_BASE]
    df = pl.DataFrame(
        {"f": ["p"] * 5 + ["q"] * 5, "v": values + [None] * 5},
        schema={"f": pl.Utf8, "v": pl.Float64},
    )
    full = df.with_columns(pl.col("v").fill_null(pl.Series(values + values)))
    null_panel, shown = _render_capturing(
        fm.Chart(df).mark_density().encode(x="v").facet(col="f").to_svg, action="always"
    )
    assert shown == [_panel_fit_message("Kde", FIT_FEW, "f=q")]
    assert x_axis_extents(null_panel) == x_axis_extents(
        fm.Chart(full).mark_density().encode(x="v").facet(col="f").to_svg()
    )


def test_an_all_null_violin_category_adds_no_position_to_the_value_axis():
    """The violin degrades before it estimates, so its value axis is the data's alone."""
    svg, _ = _render_capturing(
        fm.Chart(_level_frame([None, None])).mark_violin(inner="none").encode(x="g", y="v").to_svg,
        action="always",
    )
    alone = fm.Chart(_level_frame([])).mark_violin(inner="none").encode(x="g", y="v").to_svg()
    assert axis_tick_labels(svg, axis="y") == axis_tick_labels(alone, axis="y")


# ---------------------------------------------------------------------------
# QQ: nothing to plot warns and draws the empty frame, on every door.
# ---------------------------------------------------------------------------


def _qq_null_frame() -> pl.DataFrame:
    """Four rows with no usable ``v``, in groups ``a``, ``b`` and panels ``p``, ``q``."""
    return pl.DataFrame(
        {"v": [None] * 4, "g": ["a", "b", "a", "b"], "f": ["p", "p", "q", "q"]},
        schema={"v": pl.Float64, "g": pl.Utf8, "f": pl.Utf8},
    )


def _qq_chart(df: pl.DataFrame, *, hue: bool = False, **mark_kwargs) -> fm.Chart:
    encoding = {"x": "v", **({"color": "g"} if hue else {})}
    return fm.Chart(df).mark_qq(**mark_kwargs).encode(**encoding)


def _assert_empty_qq_frame(svg: str, panels: int = 1) -> None:
    """The plot area is there (every panel's axes read 0 to 1) and holds no point and no line."""
    unit = [AxisExtent(0.0, 1.0)] * panels
    assert x_axis_extents(svg) == unit
    assert y_axis_extents(svg) == unit
    assert mark_layer(svg) == ""


@pytest.mark.parametrize("line", [True, False], ids=["line", "no-line"])
def test_an_unfaceted_ungrouped_qq_with_no_usable_value_draws_its_empty_frame(line: bool):
    """Every value null: the frame and axes draw with no point and no reference line, and one
    warning names the cause, with no refusal."""
    svg, shown = _render_capturing(_qq_chart(_qq_null_frame(), line=line).to_svg, action="always")
    assert shown == [_whole_fit_message("QQ", FIT_NONE)]
    _assert_empty_qq_frame(svg)


def test_an_unfaceted_ungrouped_qq_of_nan_values_is_named_like_a_null_one():
    df = pl.DataFrame({"v": [float("nan")] * 4})
    svg, shown = _render_capturing(fm.Chart(df).mark_qq().encode(x="v").to_svg, action="always")
    assert shown == [_whole_fit_message("QQ", FIT_NONE)]
    _assert_empty_qq_frame(svg)


def test_an_unfaceted_grouped_qq_with_no_usable_value_names_each_group_and_keeps_its_legend():
    svg, shown = _render_capturing(_qq_chart(_qq_null_frame(), hue=True).to_svg, action="always")
    assert shown == [_fit_message("QQ", "g=a", FIT_NONE), _fit_message("QQ", "g=b", FIT_NONE)]
    assert ">a<" in svg and ">b<" in svg
    _assert_empty_qq_frame(svg)


def test_a_faceted_ungrouped_qq_with_no_usable_value_in_any_panel_draws_every_panel_empty():
    """Every panel is named once with its panel clause and drawn empty: no refusal and no
    ``data_source 'qq_main_qq_line' not found`` error."""
    chart = _qq_chart(_qq_null_frame()).facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [
        _panel_fit_message("QQ", FIT_NONE, "f=p"),
        _panel_fit_message("QQ", FIT_NONE, "f=q"),
    ]
    assert ">p<" in svg and ">q<" in svg
    _assert_empty_qq_frame(svg, 2)


def test_a_faceted_grouped_qq_with_no_usable_value_in_any_panel_names_every_group_of_every_panel():
    chart = _qq_chart(_qq_null_frame(), hue=True).facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [
        _fit_message("QQ", f"g={g}, f={f}", FIT_NONE, panel=f"f={f}") for f in "pq" for g in "ab"
    ]
    assert ">p<" in svg and ">q<" in svg
    assert ">a<" in svg and ">b<" in svg
    _assert_empty_qq_frame(svg, 2)


def test_a_faceted_qq_with_one_panel_of_values_names_only_the_emptied_panel():
    df = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, None, None, None], "f": ["p"] * 4 + ["q"] * 4},
        schema={"v": pl.Float64, "f": pl.Utf8},
    )
    svg, shown = _render_capturing(
        fm.Chart(df).mark_qq().encode(x="v").facet(col="f").to_svg, action="always"
    )
    assert shown == [_panel_fit_message("QQ", FIT_NONE, "f=q")]
    assert mark_layer(svg).count("<circle") == 4


@pytest.mark.parametrize("hue", [False, True], ids=["ungrouped", "grouped"])
def test_a_qq_on_a_zero_row_frame_reports_that_the_chart_has_no_rows(hue: bool):
    df = pl.DataFrame(
        {"v": [], "g": [], "f": []}, schema={"v": pl.Float64, "g": pl.Utf8, "f": pl.Utf8}
    )
    _, shown = _render_capturing(_qq_chart(df, hue=hue).to_svg, action="always")
    assert shown == [_NO_ROWS_TO_DRAW]
    _, faceted = _render_capturing(_qq_chart(df, hue=hue).facet(row="f").to_svg, action="always")
    assert faceted == [_NO_FACET_ROWS_WARNING]


def test_a_qq_whose_values_sit_only_on_rows_with_a_null_group_names_the_drop_and_each_group():
    """The drop is counted, and each group left with no value is named, as a density names it."""
    df = pl.DataFrame(
        {"v": [None, None, 1.0, 2.0, 3.0], "g": ["a", "b", None, None, None]},
        schema={"v": pl.Float64, "g": pl.Utf8},
    )
    svg, shown = _render_capturing(_qq_chart(df, hue=True).to_svg, action="always")
    assert shown == [
        _null_key_dropped_message("QQ", 3, "g"),
        _fit_message("QQ", "g=a", FIT_NONE),
        _fit_message("QQ", "g=b", FIT_NONE),
    ]
    assert ">a<" in svg and ">b<" in svg


def test_a_qq_with_every_group_key_null_warns_the_empty_split_and_the_empty_chart():
    df = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0], "g": [None] * 4}, schema={"v": pl.Float64, "g": pl.Utf8}
    )
    _, shown = _render_capturing(_qq_chart(df, hue=True).to_svg, action="always")
    assert shown == [_all_keys_null_message("QQ", "g"), _NO_ROWS_TO_DRAW]


def test_a_faceted_qq_with_every_group_key_null_warns_once_and_draws_every_panel_empty():
    df = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0], "g": [None] * 4, "f": ["p", "p", "q", "q"]},
        schema={"v": pl.Float64, "g": pl.Utf8, "f": pl.Utf8},
    )
    svg, shown = _render_capturing(_qq_chart(df, hue=True).facet(col="f").to_svg, action="always")
    assert shown == [_all_keys_null_message("QQ", "g")]
    assert ">p<" in svg and ">q<" in svg
    _assert_empty_qq_frame(svg, 2)


def test_a_qq_facet_key_that_is_null_on_every_row_drops_them_and_draws_no_panel():
    """The facet drops rows with a null key before the statistic runs, so no row belongs to a
    panel: the chart reports the empty frame."""
    df = pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0], "f": [None] * 4}, schema={"v": pl.Float64, "f": pl.Utf8}
    )
    _, shown = _render_capturing(_qq_chart(df).facet(col="f").to_svg, action="always")
    assert shown == [_facet_key_dropped_message(4, "f"), _NO_FACET_ROWS_WARNING]


_QQ_POINTS = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})


def _points_chart() -> fm.Chart:
    return fm.Chart(_QQ_POINTS).mark_point().encode(x="v", y="v")


_QQ_COMPOSED_DOORS = {
    "plus_qq_first": lambda qq, pt: qq + pt,
    "plus_qq_second": lambda qq, pt: pt + qq,
    "layer_chart": lambda qq, pt: fm.LayerChart(qq, pt),
    "layer_chart_reversed": lambda qq, pt: fm.LayerChart(pt, qq),
    "hconcat_leaf": lambda qq, pt: fm.hconcat(qq, pt),
}


@pytest.mark.parametrize("door", sorted(_QQ_COMPOSED_DOORS))
def test_an_all_null_qq_beside_another_chart_renders_with_its_warning_and_the_other_drawn(
    door: str,
):
    """No door refuses: the QQ names itself once and the other operand's points are drawn."""
    qq = _qq_chart(_qq_null_frame())
    svg, shown = _render_capturing(_QQ_COMPOSED_DOORS[door](qq, _points_chart()).to_svg)
    assert shown == [_whole_fit_message("QQ", FIT_NONE)]
    assert mark_layer(svg).count("<circle") == len(_QQ_POINTS)


def test_an_all_null_qq_gives_the_interactive_route_what_its_sibling_statistic_gives():
    """The whole interactive route (``.interactive()`` and ``to_html()``) warns for an all-null
    QQ exactly as it does for an all-null density over the same frame, with the statistic's name
    and reason swapped, and raises nothing else."""

    def route(chart: fm.Chart) -> list[str]:
        return _render_capturing(lambda: chart.interactive().to_html())[1]

    df = _qq_null_frame()
    density = route(fm.Chart(df).mark_density().encode(x="v"))
    assert density
    kde_message = _whole_fit_message("Kde", FIT_FEW)
    assert all(m == kde_message for m in density)
    expected = [_whole_fit_message("QQ", FIT_NONE) if m == kde_message else m for m in density]
    assert route(_qq_chart(df)) == expected


def _filtered_panel_frame() -> pl.DataFrame:
    """Panels ``p`` and ``q`` of six rows each, of which the filter ``k > 0`` keeps panel ``p``'s."""
    return pl.DataFrame(
        {
            "v": [float(i) for i in range(12)],
            "f": ["p", "q"] * 6,
            "k": [1, 0] * 6,
            "g": ["a", "a", "b", "b"] * 3,
        }
    )


def _filtered_faceted(mark_method: str, **encoding: str) -> fm.Chart:
    filtered = fm.Chart(_filtered_panel_frame()).transform(fm.transform_filter("datum.k > 0"))
    return getattr(filtered, mark_method)().encode(x="v", **encoding).facet(col="f")


@pytest.mark.parametrize("hue", [{}, {"color": "g"}], ids=["ungrouped", "grouped"])
def test_a_qq_facet_panel_a_filter_empties_is_left_out_as_the_plain_marks_leave_it(hue):
    """The filter leaves panel ``q`` no row, so it is left out with no message, as ``mark_point``
    leaves it (a grouped ``mark_density`` does too; an ungrouped one alone names it)."""
    point_svg, point = _render_capturing(_filtered_faceted("mark_point", y="v", **hue).to_svg)
    assert point == [] and ">q<" not in point_svg
    svg, shown = _render_capturing(_filtered_faceted("mark_qq", **hue).to_svg)
    assert shown == []
    assert ">p<" in svg and ">q<" not in svg


def test_a_qq_with_an_infinite_value_draws_the_finite_points_and_warns_nothing():
    """As built: an infinite value is no usable QQ value to place, so four of the five values are
    drawn and nothing is warned."""
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, float("inf")]})
    svg, shown = _render_capturing(fm.Chart(df).mark_qq().encode(x="v").to_svg)
    assert shown == []
    assert mark_layer(svg).count("<circle") == 4
