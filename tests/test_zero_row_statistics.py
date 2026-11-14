"""A frame with no rows through a box, summary or error-extent statistic draws its frame and warns.

``BoxStats``, ``Summary`` and ``ErrorExtent`` (and every mark and figure that lowers to them:
``mark_boxplot``, ``mark_violin``, ``mark_errorbar``, ``mark_errorband``, ``catplot`` box, violin,
bar and point) used to refuse a zero-row batch with ``stat_*: empty input batch``. No rows is no
group, so each now returns zero rows of its declared schema and the chart reports that it has no
rows to draw once, as every other statistic and mark does. A malformed ``groupby`` is still
refused, a key that is null on every row of a non-empty frame still has its own warning, a fit
statistic still degrades with its fit warning, and a grid whose every child is empty still raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import mark_groups_by_frame
from tests.test_group_by_date import (
    _DOORS,
    _KIND_SHAPES,
    _kind_frame,
    _kind_twin,
)
from tests.test_null_group_warning import (
    _NO_FACET_ROWS_WARNING,
    _NO_ROWS_TO_DRAW,
    _all_keys_null_message,
    _render_capturing,
    _whole_fit_message,
    FIT_FEW,
    FIT_FEW_POINTS,
)

_SCHEMA = {"k": pl.Utf8, "v": pl.Float64, "w": pl.Float64, "f": pl.Utf8}
_STATISTICS = ["BoxStats", "Summary", "ErrorExtent"]
_STATISTIC_COLUMNS = {
    "BoxStats": ("median", "q3"),
    "Summary": ("mean", "upper"),
    "ErrorExtent": ("mean", "lower"),
}


def _empty(value: pl.DataType = pl.Float64) -> pl.DataFrame:
    schema = {**_SCHEMA, "v": value}
    return pl.DataFrame({name: [] for name in schema}, schema=schema)


def _populated() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "k": ["a", "a", "b", "b"],
            "v": [1.0, 2.0, 3.0, 5.0],
            "w": [1.0, 2.0, 3.0, 4.0],
            "f": ["p", "q", "p", "q"],
        }
    )


def _statistic(name: str, *, groupby: list[str] | None) -> Any:
    step = {"BoxStats": fm.BoxStats, "Summary": fm.Summary, "ErrorExtent": fm.ErrorExtent}[name]
    return step("v", groupby=groupby) if groupby is not None else step("v")


def _written(name: str, df: pl.DataFrame, *, groupby: list[str] | None) -> fm.Chart:
    x, y = _STATISTIC_COLUMNS[name]
    chart = fm.Chart(df).transform(_statistic(name, groupby=groupby))
    return chart.mark_point().encode(x=x, y=y)


def _marks() -> dict[str, Callable[[pl.DataFrame], fm.Chart]]:
    return {
        "mark_boxplot": lambda df: fm.Chart(df).mark_boxplot().encode(x="k", y="v"),
        "mark_violin": lambda df: fm.Chart(df).mark_violin().encode(x="k", y="v"),
        "mark_errorbar": lambda df: fm.Chart(df).mark_errorbar().encode(x="k", y="v"),
        "mark_errorband": lambda df: fm.Chart(df).mark_errorband().encode(x="k", y="v"),
        "catplot_box": lambda df: fm.catplot(df, x="k", y="v", kind="box"),
        "catplot_violin": lambda df: fm.catplot(df, x="k", y="v", kind="violin"),
        "catplot_bar": lambda df: fm.catplot(df, x="k", y="v", kind="bar"),
        "catplot_point": lambda df: fm.catplot(df, x="k", y="v", kind="point"),
    }


_MARKS = _marks()
_FACETED_MARKS = {
    "mark_boxplot": lambda df: _MARKS["mark_boxplot"](df).facet("f"),
    "mark_errorbar": lambda df: _MARKS["mark_errorbar"](df).facet("f"),
    "catplot_box": lambda df: fm.catplot(df, x="k", y="v", kind="box", col="f"),
    "catplot_bar": lambda df: fm.catplot(df, x="k", y="v", kind="bar", col="f"),
    "catplot_point": lambda df: fm.catplot(df, x="k", y="v", kind="point", col="f"),
    "catplot_violin": lambda df: fm.catplot(df, x="k", y="v", kind="violin", col="f"),
    "BoxStats": lambda df: _written("BoxStats", df, groupby=["k"]).facet("f"),
    "Summary": lambda df: _written("Summary", df, groupby=["k"]).facet("f"),
    "ErrorExtent": lambda df: _written("ErrorExtent", df, groupby=["k"]).facet("f"),
}


def _interactive(chart: Any) -> Callable[[], Any]:
    return lambda: chart.interactive().scene_json


# ---------------------------------------------------------------------------
# Marks and figures that lower to the statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark", sorted(_MARKS))
def test_a_zero_row_mark_draws_its_frame_and_axes_and_warns_once(mark: str):
    svg, messages = _render_capturing(_MARKS[mark](_empty()).to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    assert ">k<" in svg and ">v<" in svg


@pytest.mark.parametrize("mark", sorted(_MARKS))
def test_a_zero_row_mark_warns_once_on_the_interactive_route(mark: str):
    _, messages = _render_capturing(_interactive(_MARKS[mark](_empty())))
    assert messages == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("mark", sorted(_MARKS))
def test_the_same_mark_over_rows_warns_nothing(mark: str):
    _, messages = _render_capturing(_MARKS[mark](_populated()).to_svg)
    assert messages == []


_VALUE_DTYPES = [pl.Int64, pl.Int32, pl.Float32, pl.Null]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("mark", ["mark_boxplot", "mark_errorbar", "catplot_bar", "catplot_box"])
@pytest.mark.parametrize("value", _VALUE_DTYPES, ids=str)
def test_a_zero_row_value_column_of_any_dtype_warns_once(
    value: pl.DataType, mark: str, output: str
):
    chart = _MARKS[mark](_empty(value))
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("mark", ["mark_boxplot", "mark_errorbar", "catplot_bar"])
def test_a_schema_less_empty_frame_warns_once(mark: str):
    """``pl.DataFrame({"k": [], "v": []})`` types every column Null."""
    frame = pl.DataFrame({"k": [], "v": [], "w": [], "f": []})
    assert _render_capturing(_MARKS[mark](frame).to_svg)[1] == [_NO_ROWS_TO_DRAW]


# ---------------------------------------------------------------------------
# Written statistic steps, grouped and ungrouped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groupby", [None, ["k"], ["k", "f"]], ids=["ungrouped", "k", "k_f"])
@pytest.mark.parametrize("name", _STATISTICS)
def test_a_written_statistic_over_no_rows_draws_and_warns_once(name: str, groupby):
    chart = _written(name, _empty(), groupby=groupby)
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    assert "<svg" in svg
    _, interactive = _render_capturing(_interactive(chart))
    assert interactive == [_NO_ROWS_TO_DRAW]


_KEY_DTYPES = [pl.Utf8, pl.Float64, pl.Date]


def _key_axis_page(frame: pl.DataFrame, y: str) -> tuple[str, list[str]]:
    return _render_capturing(fm.Chart(frame).mark_point().encode(x="k", y=y).to_svg)


@pytest.mark.parametrize("key", _KEY_DTYPES, ids=str)
@pytest.mark.parametrize("name", _STATISTICS)
def test_a_written_statistic_keeps_the_key_column_at_its_input_dtype(name: str, key: pl.DataType):
    """The page drawn from the statistic's zero-row output, with the key on the x axis, equals the
    page drawn from a hand-built zero-row frame of the declared columns whose key has the input
    dtype, and differs from the page of the same frame with the key under another dtype."""
    _, y = _STATISTIC_COLUMNS[name]
    schema = {"k": key, "v": pl.Float64}
    empty = pl.DataFrame({column: [] for column in schema}, schema=schema)
    chart = fm.Chart(empty).transform(_statistic(name, groupby=["k"]))
    page, messages = _render_capturing(chart.mark_point().encode(x="k", y=y).to_svg)
    declared = pl.DataFrame({"k": [], y: []}, schema={"k": key, y: pl.Float64})
    reference, reference_messages = _key_axis_page(declared, y)
    assert messages == reference_messages == [_NO_ROWS_TO_DRAW]
    assert page == reference
    for other in _KEY_DTYPES:
        if other != key:
            other_page, _ = _key_axis_page(
                pl.DataFrame({"k": [], y: []}, schema={"k": other, y: pl.Float64}), y
            )
            assert page != other_page, f"{key} and {other} draw the same page"


@pytest.mark.parametrize("name", _STATISTICS)
def test_a_populated_frame_still_draws_its_groups_without_a_warning(name: str):
    _, messages = _render_capturing(_written(name, _populated(), groupby=["k"]).to_svg)
    assert messages == []


# ---------------------------------------------------------------------------
# Every key dtype equals its epoch twin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("name", _STATISTICS)
@pytest.mark.parametrize(("kind", "shape"), _KIND_SHAPES)
def test_a_zero_row_frame_of_every_key_dtype_equals_its_epoch_twin(
    kind: str, shape: str, name: str, output: str
):
    def render(frame: pl.DataFrame):
        chart = _DOORS[name](frame.head(0))
        if output == "static":
            return _render_capturing(chart.to_svg)
        return _render_capturing(_interactive(chart))

    page, messages = render(_kind_frame(kind, shape))
    twin_page, twin_messages = render(_kind_twin(kind, shape))
    assert messages == twin_messages == [_NO_ROWS_TO_DRAW]
    assert page == twin_page


# ---------------------------------------------------------------------------
# Faceted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("door", sorted(_FACETED_MARKS))
def test_a_zero_row_faceted_frame_draws_no_panel_and_says_so(door: str):
    svg, messages = _render_capturing(_FACETED_MARKS[door](_empty()).to_svg)
    assert messages == [_NO_FACET_ROWS_WARNING]
    assert mark_groups_by_frame(svg) == []
    _, interactive = _render_capturing(_interactive(_FACETED_MARKS[door](_empty())))
    assert interactive == [_NO_FACET_ROWS_WARNING]


# ---------------------------------------------------------------------------
# Under composition doors
# ---------------------------------------------------------------------------


def _beside(df: pl.DataFrame) -> fm.Chart:
    return fm.Chart(df).mark_point().encode(x="k", y="v")


@pytest.mark.parametrize("mark", ["mark_boxplot", "mark_errorbar", "catplot_bar"])
def test_a_zero_row_statistic_layer_under_plus_warns_once(mark: str):
    chart = _MARKS[mark](_empty()) + _beside(_empty())
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    _, interactive = _render_capturing(_interactive(chart))
    assert interactive == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("mark", ["mark_boxplot", "mark_errorbar", "catplot_bar"])
def test_a_zero_row_statistic_layer_in_a_layer_chart_warns_once(mark: str):
    chart = fm.LayerChart(_MARKS[mark](_empty()), _beside(_empty()))
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    _, interactive = _render_capturing(_interactive(chart))
    assert interactive == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("mark", ["mark_boxplot", "mark_errorbar", "catplot_bar"])
def test_a_zero_row_statistic_chart_with_a_layer_warns_once(mark: str):
    chart = _MARKS[mark](_empty()).layer(fm.Layer(mark="point"))
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_NO_ROWS_TO_DRAW]
    _, interactive = _render_capturing(_interactive(chart))
    assert interactive == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("mark", sorted(_MARKS))
def test_a_zero_row_statistic_leaf_beside_a_populated_leaf_renders_both_routes(mark: str):
    """A grid with one populated child renders (the all-empty grid raises, below): the zero-row
    child's statistic no longer fails the whole composite, and the populated child still draws."""
    grid = fm.hconcat(_MARKS[mark](_empty()), _MARKS[mark](_populated()))
    svg, messages = _render_capturing(grid.to_svg)
    assert messages == []
    assert ">a<" in svg and ">b<" in svg
    _, interactive = _render_capturing(_interactive(grid))
    assert interactive == []


def test_a_zero_row_statistic_layer_beside_a_populated_layer_still_draws_the_populated_one():
    chart = _MARKS["mark_boxplot"](_empty()) + _beside(_populated())
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == []
    assert svg.count("<circle") == len(_populated())


# ---------------------------------------------------------------------------
# Unchanged cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _STATISTICS)
def test_a_groupby_column_the_frame_lacks_is_still_refused(name: str):
    stat = {
        "BoxStats": "stat_box_stats",
        "Summary": "stat_summary",
        "ErrorExtent": "stat_error_extent",
    }[name]
    chart = _written(name, _empty(), groupby=["zz"])
    with pytest.raises(
        ValueError,
        match=rf"^transform failed: ValueError: {stat}: groupby column 'zz' not found$",
    ):
        chart.to_svg()


@pytest.mark.parametrize("name", _STATISTICS)
def test_a_repeated_groupby_name_is_still_refused(name: str):
    with pytest.raises(ValueError, match=r"groupby column 'k' is repeated in `groupby`"):
        _written(name, _empty(), groupby=["k", "k"]).to_svg()


@pytest.mark.parametrize("name", _STATISTICS)
def test_a_groupby_column_named_for_an_output_is_still_refused(name: str):
    output = _STATISTIC_COLUMNS[name][0]
    chart = _written(name, _empty().rename({"k": output}), groupby=[output])
    with pytest.raises(
        ValueError,
        match=rf"groupby column '{output}' collides with this transform's own output column",
    ):
        chart.to_svg()


@pytest.mark.parametrize("name", _STATISTICS)
def test_a_key_null_on_every_row_of_a_populated_frame_is_unchanged(name: str):
    frame = _populated().with_columns(k=pl.lit(None, dtype=pl.Utf8))
    _, messages = _render_capturing(_written(name, frame, groupby=["k"]).to_svg)
    assert messages == [_all_keys_null_message(name, "k"), _NO_ROWS_TO_DRAW]


def test_a_zero_row_density_still_degrades_with_its_fit_warning():
    chart = fm.Chart(_empty()).mark_density().encode(x="v")
    _, messages = _render_capturing(chart.to_svg)
    assert messages == [_whole_fit_message("Kde", FIT_FEW)]


def test_a_zero_row_smooth_and_lmplot_still_degrade_with_the_fit_warning():
    smooth = fm.Chart(_empty()).mark_smooth().encode(x="w", y="v")
    _, messages = _render_capturing(smooth.to_svg)
    assert messages == [_whole_fit_message("Smooth", FIT_FEW_POINTS)]
    _, lm = _render_capturing(fm.lmplot(_empty(), x="w", y="v").to_svg)
    assert lm == messages
    written = fm.Chart(_empty()).transform(fm.Smooth("w", "v", method="lm", groupby=["k"]))
    _, whole = _render_capturing(written.mark_point().encode(x="x", y="y").to_svg)
    assert whole == [_whole_fit_message("Smooth", FIT_FEW_POINTS)]


def test_a_grid_whose_every_child_is_zero_row_still_raises():
    grid = fm.hconcat(_MARKS["mark_boxplot"](_empty()), _MARKS["mark_errorbar"](_empty()))
    with pytest.raises(ValueError, match="every child's data is empty; nothing to render"):
        grid.to_svg()
