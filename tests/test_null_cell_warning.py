"""A hue x category cell whose rows all have a null or non-finite value is named, once.

When a hue level draws in other categories and a category draws under other hue levels, but
every row of one (category, hue) pair places nowhere, neither the hue group nor the band slot
draws nothing, so the pair was the one unit that drew nothing without a name. The null census
names it by both keys in the order the layer reads them (the category column, then the hue
column), the way it names a group of two fields (``g=c, k=u``), in the channels its caller
wrote: ``every row of c=a, g=q has a null or non-finite y, so it draws nothing``.

A whole empty hue group stays named as the group and a whole empty category as the slot, never
again per cell; a cell with a usable row, a cell with no rows at all, and a cell an explicit
domain or a log scale cuts keep the reports they had. A violin is named by its own fit, which
the census defers to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests.test_null_group_warning import (
    _DRAWN_Z,
    _EMPTY_Z,
    FIT_FEW,
    _fit_message,
    _hue_group_frame,
    _log_message,
    _render_capturing,
    _slot_frame,
    _slot_message,
)

#: The cell every case empties: category ``a`` under hue ``q``. Hue ``q`` draws in ``b`` and ``z``,
#: and category ``a`` draws under hue ``p``, so only the pair itself draws nothing.
_CELL = ("a", "q")
_CELL_NAME = "a, g=q"


def _cell_message(entry: str, channels: str = "y") -> str:
    """The text that names the empty cell ``c=a, g=q`` of layer ``entry``."""
    return _slot_message(entry, channels, _CELL_NAME, "c")


def _cell_frame(value: float | None, *, only_rows: int | None = None) -> pl.DataFrame:
    """``_slot_frame`` with ``w`` set to ``value`` on the cell's rows (all three, or the first
    ``only_rows`` of them), so every other cell keeps its usable values."""
    frame = _slot_frame(_DRAWN_Z).with_row_index("i")
    in_cell = (pl.col("c") == _CELL[0]) & (pl.col("g") == _CELL[1])
    if only_rows is not None:
        in_cell = in_cell & (
            pl.col("i").is_in(frame.filter(in_cell)["i"].head(only_rows).to_list())
        )
    return frame.with_columns(w=pl.when(in_cell).then(value).otherwise(pl.col("w"))).drop("i")


def _encoding(horizontal: bool, value: Any = "w") -> dict[str, Any]:
    return (
        {"y": "c", "x": value, "color": "g"} if horizontal else {"x": "c", "y": value, "color": "g"}
    )


def _primitive(mark: str) -> Callable[[pl.DataFrame, bool], fm.Chart]:
    return lambda df, horizontal: getattr(fm.Chart(df), mark)().encode(**_encoding(horizontal))


def _composite(mark: str) -> Callable[[pl.DataFrame, bool], fm.Chart]:
    return lambda df, horizontal: getattr(fm.Chart(df), mark)(horizontal=horizontal).encode(
        **_encoding(horizontal)
    )


def _mean_bar(df: pl.DataFrame, horizontal: bool) -> fm.Chart:
    value = fm.X("w", aggregate="mean") if horizontal else fm.Y("w", aggregate="mean")
    return fm.Chart(df).mark_bar().encode(**_encoding(horizontal, value))


def _catplot(kind: str) -> Callable[[pl.DataFrame, bool], fm.Chart]:
    def build(df: pl.DataFrame, horizontal: bool) -> fm.Chart:
        if horizontal:
            return fm.catplot(df, y="c", x="w", kind=kind, hue="g", orient="h")
        return fm.catplot(df, x="c", y="w", kind=kind, hue="g")

    return build


#: ``(builder(frame, horizontal), the entry the census names)``.
_DOORS = [
    pytest.param(_primitive("mark_point"), "mark_point", id="point"),
    pytest.param(_primitive("mark_tick"), "mark_tick", id="tick"),
    pytest.param(_primitive("mark_line"), "mark_line", id="line"),
    pytest.param(_composite("mark_boxplot"), "mark_boxplot", id="boxplot"),
    pytest.param(_composite("mark_swarm"), "mark_swarm", id="swarm"),
    pytest.param(_primitive("mark_errorbar"), "mark_errorbar", id="errorbar"),
    pytest.param(_composite("mark_boxen"), "mark_boxen", id="boxen"),
    pytest.param(_mean_bar, "mark_bar", id="mean_bar"),
    pytest.param(_catplot("box"), "catplot", id="catplot_box"),
    pytest.param(_catplot("strip"), "catplot", id="catplot_strip"),
    pytest.param(_catplot("bar"), "catplot", id="catplot_bar"),
]

_UNDRAWABLE_VALUES = [
    pytest.param(None, id="null"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
]


def _shown(build: Callable[[pl.DataFrame, bool], fm.Chart], df: pl.DataFrame, horizontal: bool):
    """Every warning the chart raises under Python's own filter, which shows the one text a
    composite's several layers raise once."""
    return _render_capturing(build(df, horizontal).to_svg, action="default")[1]


@pytest.mark.parametrize("build,entry", _DOORS)
def test_an_empty_hue_by_category_cell_is_named(
    build: Callable[[pl.DataFrame, bool], fm.Chart], entry: str
):
    assert _shown(build, _cell_frame(None), False) == [_cell_message(entry)]


#: Every door but the error bar, which is drawn vertically only.
_HORIZONTAL_DOORS = [door for door in _DOORS if door.id != "errorbar"]


@pytest.mark.parametrize("build,entry", _HORIZONTAL_DOORS)
def test_a_horizontal_cell_is_named_in_the_channel_the_caller_wrote(
    build: Callable[[pl.DataFrame, bool], fm.Chart], entry: str
):
    """The caller bound the values to ``x`` and the categories to ``y``, so the cell reads ``x``."""
    assert _shown(build, _cell_frame(None), True) == [_cell_message(entry, "x")]


@pytest.mark.parametrize("value", _UNDRAWABLE_VALUES)
@pytest.mark.parametrize("mark", ["mark_point", "mark_tick", "mark_line"])
def test_a_cell_of_null_nan_or_infinite_values_is_named_under_every_warning_copy(
    mark: str, value: float | None
):
    """A placed mark raises the census once, whatever the filter: ``always`` keeps every copy."""
    chart = _primitive(mark)(_cell_frame(value), False)
    assert _render_capturing(chart.to_svg)[1] == [_cell_message(mark)]


@pytest.mark.parametrize("build,entry", _DOORS)
def test_a_cell_with_a_usable_row_is_not_named(
    build: Callable[[pl.DataFrame, bool], fm.Chart], entry: str
):
    assert _shown(build, _cell_frame(None, only_rows=1), False) == []


@pytest.mark.parametrize("build,entry", _DOORS)
def test_a_cell_with_no_rows_is_not_named(
    build: Callable[[pl.DataFrame, bool], fm.Chart], entry: str
):
    """The (``a``, ``q``) combination is absent from the data: nothing was asked to draw."""
    frame = _slot_frame(_DRAWN_Z).filter(~((pl.col("c") == "a") & (pl.col("g") == "q")))
    assert _shown(build, frame, False) == []


def test_a_violin_is_named_by_its_own_fit_and_the_census_adds_nothing():
    frame = _cell_frame(None)
    chart = fm.Chart(frame).mark_violin(inner="none").encode(x="c", y="w", color="g")
    shown = _render_capturing(chart.to_svg, action="default")[1]
    assert shown == [_fit_message("Violin", "c=a, g=q", FIT_FEW)]


@pytest.mark.parametrize("build,entry", _DOORS)
def test_a_whole_empty_hue_group_is_named_as_the_group_and_not_again_per_cell(
    build: Callable[[pl.DataFrame, bool], fm.Chart], entry: str
):
    """Hue ``r`` is null in every category: one group name, and no cell of it."""
    shown = _shown(build, _hue_group_frame(), False)
    assert shown == [_slot_message(entry, "y", "r", "g")]


@pytest.mark.parametrize("mark", ["mark_point", "mark_tick", "mark_line"])
def test_a_whole_empty_category_is_named_as_the_slot_and_not_again_per_cell(mark: str):
    chart = _primitive(mark)(_slot_frame(_EMPTY_Z), False)
    assert _render_capturing(chart.to_svg)[1] == [_slot_message(mark)]


def test_two_empty_cells_are_named_together_in_first_appearance_order():
    """Cells ``(a, q)`` and ``(b, p)`` draw nothing; each is a two-field name, so the list joins
    with semicolons, as two-field groups do."""
    frame = _slot_frame(_DRAWN_Z).with_columns(
        w=pl.when(
            ((pl.col("c") == "a") & (pl.col("g") == "q"))
            | ((pl.col("c") == "b") & (pl.col("g") == "p"))
        )
        .then(None)
        .otherwise(pl.col("w"))
    )
    chart = fm.Chart(frame).mark_point().encode(x="c", y="w", color="g")
    shown = _render_capturing(chart.to_svg)[1]
    assert shown == [
        "mark_point: every row of c=a, g=q; and of c=b, g=p, has a null or non-finite y, "
        "so they draw nothing"
    ]


def test_a_cell_cut_by_an_explicit_domain_is_reported_by_the_domain_alone():
    frame = _cell_frame(90.0)
    y = fm.Y("w", scale=fm.LinearScale(domain=[0.0, 10.0]))
    chart = fm.Chart(frame).mark_point().encode(x="c", y=y, color="g")
    shown = _render_capturing(chart.to_svg)[1]
    assert shown == [
        "y scale: 3 row(s) of 'w' fall outside the scale's explicit domain, so they are not "
        "drawn; widen the domain, or pass clamp=True to draw them at the axis edge"
    ]


def test_a_cell_off_the_log_side_is_reported_by_the_log_scale_alone():
    chart = (
        fm.Chart(_cell_frame(-1.0))
        .mark_point()
        .encode(x="c", y=fm.Y("w", scale=fm.LogScale()), color="g")
    )
    assert _render_capturing(chart.to_svg)[1] == [_log_message("y", "w", 3)]
