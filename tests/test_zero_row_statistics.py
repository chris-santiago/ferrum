"""A frame with no rows through a box, summary or error-extent statistic draws its frame and warns.

``BoxStats``, ``Summary`` and ``ErrorExtent`` (and every mark and figure that lowers to them:
``mark_boxplot``, ``mark_violin``, ``mark_errorbar``, ``mark_errorband``, ``catplot`` box, violin,
bar and point) used to refuse a zero-row batch with ``stat_*: empty input batch``. No rows is no
group, so each now returns zero rows of its declared schema and the chart reports that it has no
rows to draw once, as every other statistic and mark does. A malformed ``groupby`` is still
refused, a key that is null on every row of a non-empty frame still has its own warning, a fit
statistic still degrades with its fit warning, and a grid whose every child is empty still raises.

A zero-row ``transform_pivot`` publishes the columns its readers ask for as empty ``Float64``
columns: the encodings, the steps after it (read before they are computed, including a step that
rewrites a pivoted name in place), composition doors and facet panels. A pivot over rows is
byte-identical to what it drew before, and a name its values lack is still refused.
"""

from __future__ import annotations

import hashlib
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
    FIT_FEW_LOESS_POINTS,
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


# ---------------------------------------------------------------------------
# A zero-row pivot publishes the columns its readers ask for
# ---------------------------------------------------------------------------

_PIVOT = fm.transform_pivot("h", "v", groupby=["k"])


def _pivot_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {"k": ["p", "p", "q", "q"], "h": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]}
    )


def _pivot_empty() -> pl.DataFrame:
    return _pivot_rows().head(0)


def _pivot_chart(df: pl.DataFrame, **encoding: str) -> fm.Chart:
    return fm.Chart(df).transform(_PIVOT).mark_point().encode(**encoding)


_PIVOT_DOORS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "x_y": lambda df: _pivot_chart(df, x="a", y="b"),
    "size": lambda df: _pivot_chart(df, x="k", y="k", size="a"),
    "color": lambda df: _pivot_chart(df, x="k", y="k", color="b"),
    "calculate": lambda df: (
        fm.Chart(df)
        .transform(_PIVOT, fm.transform_calculate("z", "datum.a + 1"))
        .mark_point()
        .encode(x="z", y="k")
    ),
    # Bin per_row appends bin_start, and refuses an input column of that name: the pivot must
    # not publish the name the step computes, or the step refuses its own output.
    "computed_name": lambda df: (
        fm.Chart(df)
        .transform(_PIVOT, fm.Bin("a", per_row=True))
        .mark_point()
        .encode(x="bin_start", y="k")
    ),
    "groupby_only": lambda df: _pivot_chart(df, x="k", y="k"),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_PIVOT_DOORS))
def test_a_zero_row_pivot_draws_its_frame_and_warns_once(door: str, output: str):
    chart = _PIVOT_DOORS[door](_pivot_empty())
    render = chart.to_svg if output == "static" else _interactive(chart)
    _, messages = _render_capturing(render)
    assert messages == [_NO_ROWS_TO_DRAW]


def test_a_zero_row_pivot_draws_the_pivoted_names_on_its_axes():
    svg, _ = _render_capturing(_PIVOT_DOORS["x_y"](_pivot_empty()).to_svg)
    assert ">a<" in svg and ">b<" in svg


def test_a_zero_row_pivot_beside_a_populated_pivot_draws_the_populated_one():
    chart = _PIVOT_DOORS["x_y"](_pivot_empty()) + _PIVOT_DOORS["x_y"](_pivot_rows())
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == []
    assert svg.count("<circle") == 2


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_zero_row_pivot_under_plus_warns_once(output: str):
    chart = _PIVOT_DOORS["x_y"](_pivot_empty()) + _PIVOT_DOORS["x_y"](_pivot_empty())
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_zero_row_pivot_in_a_layer_chart_warns_once(output: str):
    chart = fm.LayerChart(_PIVOT_DOORS["x_y"](_pivot_empty()), _PIVOT_DOORS["x_y"](_pivot_empty()))
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_zero_row_pivot_with_a_layer_warns_once(output: str):
    chart = _PIVOT_DOORS["x_y"](_pivot_empty()).layer(fm.Layer(mark="point"))
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_zero_row_faceted_pivot_draws_no_panel_and_says_so(output: str):
    chart = _PIVOT_DOORS["x_y"](_pivot_empty()).facet("k")
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_FACET_ROWS_WARNING]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize(("kind", "shape"), _KIND_SHAPES)
def test_a_zero_row_pivot_keyed_by_every_dtype_equals_its_epoch_twin(
    kind: str, shape: str, output: str
):
    def render(frame: pl.DataFrame):
        chart = _DOORS["pivot"](frame.head(0))
        return _render_capturing(chart.to_svg if output == "static" else _interactive(chart))

    page, messages = render(_kind_frame(kind, shape))
    twin_page, twin_messages = render(_kind_twin(kind, shape))
    assert messages == twin_messages == [_NO_ROWS_TO_DRAW]
    assert page == twin_page


def test_a_pivot_over_rows_is_the_chart_over_its_pivoted_frame():
    wide = pl.DataFrame({"k": ["p", "q"], "a": [1.0, 3.0], "b": [2.0, 4.0]})
    page, messages = _render_capturing(_PIVOT_DOORS["x_y"](_pivot_rows()).to_svg)
    reference, _ = _render_capturing(fm.Chart(wide).mark_point().encode(x="a", y="b").to_svg)
    assert messages == []
    assert page == reference


_UNKNOWN_ZZ = "unknown column 'zz' referenced by an encoding"


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_pivot_over_rows_still_refuses_a_name_its_values_lack(output: str):
    chart = _pivot_chart(_pivot_rows(), x="zz", y="b")
    render = chart.to_svg if output == "static" else _interactive(chart)
    with pytest.raises(ValueError) as raised:
        render()
    assert str(raised.value) == _UNKNOWN_ZZ


def test_a_zero_row_frame_without_a_pivot_still_refuses_a_missing_column():
    chart = fm.Chart(_pivot_empty()).mark_point().encode(x="zz", y="k")
    with pytest.raises(ValueError) as raised:
        chart.to_svg()
    assert str(raised.value) == _UNKNOWN_ZZ


def test_a_malformed_pivot_is_still_refused_on_a_zero_row_frame():
    pivot = fm.transform_pivot("h", "v", groupby=["zz"])
    chart = fm.Chart(_pivot_empty()).transform(pivot).mark_point().encode(x="a", y="b")
    with pytest.raises(ValueError) as raised:
        chart.to_svg()
    assert (
        str(raised.value)
        == "transform failed: ValueError: data_pivot: groupby column 'zz' not found"
    )


def _populated_pivots() -> dict[str, fm.Chart]:
    return {
        "x_y": _pivot_chart(_pivot_rows(), x="a", y="b"),
        "color": _pivot_chart(_pivot_rows(), x="a", y="b", color="k"),
        "calculate": (
            fm.Chart(_pivot_rows())
            .transform(_PIVOT, fm.transform_calculate("z", "datum.a + datum.b"))
            .mark_point()
            .encode(x="z", y="k")
        ),
    }


# SHA-256 of the static SVG and of the interactive scene JSON, measured before the zero-row pivot
# change (the scene JSON carries no per-run id, so it is compared as is).
_POPULATED_PIVOT_DIGESTS = {
    "x_y": (
        "17598a74093638d6487d2e8b466619f927cae45e41293d62eb535f86e0164f7b",
        "816a114c63029a9d24ef7e855e573a55a5743d4ba37b9090c0c768425d012b0c",
    ),
    "color": (
        "310c379ef93e961efccf25234c8020f9c56244dc8cb415447c7e231d622258b4",
        "032e78b51d5409b915e0fe19cfd3a5c7df0325e9d05a645a199a818e1ced363a",
    ),
    "calculate": (
        "c8fbb75f14388a0c85a940e9f3ef09fc2e75cf4c24663106e51288e3fdd7be11",
        "9670b68f1eacbaff4f51ec2b4d9ff99aa7d0eeeed9a584348105de8de7ab23ac",
    ),
}


@pytest.mark.parametrize("door", sorted(_POPULATED_PIVOT_DIGESTS))
def test_a_populated_pivot_is_byte_identical_to_before(door: str):
    chart = _populated_pivots()[door]
    svg = hashlib.sha256(chart.to_svg().encode()).hexdigest()
    scene = hashlib.sha256(str(chart.interactive().scene_json).encode()).hexdigest()
    assert (svg, scene) == _POPULATED_PIVOT_DIGESTS[door]


# ---------------------------------------------------------------------------
# Steps that follow a zero-row pivot read its published columns
# ---------------------------------------------------------------------------


def _following_frame() -> pl.DataFrame:
    schema = {"g": pl.Utf8, "h": pl.Utf8, "v": pl.Float64}
    return pl.DataFrame({name: [] for name in schema}, schema=schema)


_FOLLOWING_PIVOT = fm.transform_pivot("h", "v", groupby=["g"])

# (steps after the pivot, x, y); the chart draws its mark_point over the last step's output.
_FOLLOWING_STEPS: dict[str, tuple[list[dict], str, str | None]] = {
    "calculate": ([fm.transform_calculate("c", "datum.a + 1")], "a", "c"),
    "fold": ([fm.transform_fold(["a", "b"], as_=("key", "value"))], "key", "value"),
    "window_rank": (
        [fm.transform_window({"op": "rank", "as": "r"}, sort=["a"])],
        "a",
        "r",
    ),
    "stack": ([fm.transform_stack("a", groupby=["g"], as_=("y0", "y1"))], "g", "y1"),
    "density": ([fm.transform_density("a", as_=("value", "density"))], "value", "density"),
    "loess": (
        [fm.transform_loess("a", "b", bandwidth=0.5, as_=("lx", "ly"))],
        "lx",
        "ly",
    ),
    "regression": (
        [fm.transform_regression("a", "b", as_=("rx", "ry"))],
        "rx",
        "ry",
    ),
    "timeunit_default": ([fm.transform_timeunit("a", "year")], "year_a", None),
    "timeunit_named": ([fm.transform_timeunit("a", "year", as_="yr")], "yr", None),
    "bin_default": ([fm.transform_bin("a")], "a_bin", None),
    "bin_named": ([fm.transform_bin("a", as_="bb")], "bb", None),
    "flatten": ([fm.transform_flatten(["a"], as_=["fa"])], "fa", None),
}


def _following_chart(steps: list[dict], x: str, y: str | None) -> fm.Chart:
    chart = fm.Chart(_following_frame()).transform(_FOLLOWING_PIVOT, *steps)
    return chart.mark_point().encode(x=x, y=y if y is not None else "g")


_FOLLOWING_FIT_WARNINGS = {
    "loess": _whole_fit_message("LoessData", FIT_FEW_LOESS_POINTS),
    "regression": _whole_fit_message("RegressionData", FIT_FEW_POINTS),
}
_FOLLOWING_REFUSED = ["flatten"]
_FOLLOWING_DRAWN = [
    step
    for step in _FOLLOWING_STEPS
    if step not in _FOLLOWING_FIT_WARNINGS and step not in _FOLLOWING_REFUSED
]


def _render_following(chart: fm.Chart, output: str) -> list[str]:
    render = chart.to_svg if output == "static" else _interactive(chart)
    return _render_capturing(render)[1]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("step", _FOLLOWING_DRAWN)
def test_a_step_after_a_zero_row_pivot_draws_its_frame_and_warns_once(step: str, output: str):
    chart = _following_chart(*_FOLLOWING_STEPS[step])
    assert _render_following(chart, output) == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("step", sorted(_FOLLOWING_FIT_WARNINGS))
def test_a_fit_step_after_a_zero_row_pivot_warns_as_it_does_over_the_published_columns(
    step: str, output: str
):
    """The pivot route gives the warning list of the same step over a zero-row frame that already
    holds the columns, and that list is the step's own degenerate-fit warning."""
    steps, x, y = _FOLLOWING_STEPS[step]
    published = pl.DataFrame(
        {"g": [], "a": [], "b": []}, schema={"g": pl.Utf8, "a": pl.Float64, "b": pl.Float64}
    )
    twin = fm.Chart(published).transform(*steps).mark_point().encode(x=x, y=y)
    expected = [_FOLLOWING_FIT_WARNINGS[step]]
    assert _render_following(twin, output) == expected
    assert _render_following(_following_chart(steps, x, y), output) == expected


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_flatten_after_a_zero_row_pivot_is_refused_as_after_a_populated_pivot(output: str):
    steps, x, y = _FOLLOWING_STEPS["flatten"]
    populated = (
        fm.Chart(_pivot_rows().rename({"k": "g"}))
        .transform(_FOLLOWING_PIVOT, *steps)
        .mark_point()
        .encode(x=x, y="g")
    )

    def refusal(chart: fm.Chart) -> tuple[type, str]:
        render = chart.to_svg if output == "static" else _interactive(chart)
        with pytest.raises(ValueError) as raised:
            render()
        return type(raised.value), str(raised.value)

    zero_row = refusal(_following_chart(steps, x, y))
    assert zero_row == refusal(populated)
    assert zero_row[1] == (
        "transform failed: ValueError: data_flatten: column 'a' has dtype Float64, which cannot "
        "be flattened; expected a List column or a Utf8 column of ';'-delimited values"
    )


def _composed(steps: list[dict], x: str, y: str | None) -> fm.Chart:
    layer = lambda: fm.Chart(_following_frame()).mark_point().encode(x=x, y=y or "g")  # noqa: E731
    return (layer() + layer()).transform(_FOLLOWING_PIVOT, *steps)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("step", ["calculate", "fold"])
def test_a_step_after_a_zero_row_pivot_on_a_composed_chart_warns_once(step: str, output: str):
    chart = _composed(*_FOLLOWING_STEPS[step])
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_bare_zero_row_pivot_on_a_composed_chart_warns_once(output: str):
    chart = _composed([], "a", "b")
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_zero_row_pivot_read_only_by_a_detail_binding_warns_once(output: str):
    chart = (
        fm.Chart(_following_frame())
        .transform(_FOLLOWING_PIVOT)
        .mark_line()
        .encode(x="g", y="a", detail="b")
    )
    render = chart.to_svg if output == "static" else _interactive(chart)
    assert _render_capturing(render)[1] == [_NO_ROWS_TO_DRAW]


# ---------------------------------------------------------------------------
# A step that reads a pivoted name and writes the same name
# ---------------------------------------------------------------------------

_IN_PLACE_STEPS: dict[str, list[dict]] = {
    "calculate": [fm.transform_calculate("a", "datum.a * 2")],
    "join_aggregate": [fm.transform_join_aggregate({"fn": "mean", "field": "a", "as": "a"})],
    "bin": [fm.transform_bin("a", as_="a")],
    "window": [fm.transform_window({"op": "row_number", "as": "a"}, sort=["a"])],
    "calculate_chain": [
        fm.transform_calculate("c", "datum.a + 1"),
        fm.transform_calculate("a", "datum.c"),
    ],
    "read_after_compute": [
        fm.transform_calculate("c", "datum.a + 1"),
        fm.transform_calculate("d", "datum.c * 2"),
    ],
}


def _in_place_chart(frame: pl.DataFrame, step: str, *, composed: bool) -> fm.Chart:
    x = "d" if step == "read_after_compute" else "a"
    if composed:
        layer = lambda: fm.Chart(frame).mark_point().encode(x=x, y="g")  # noqa: E731
        return (layer() + layer()).transform(_FOLLOWING_PIVOT, *_IN_PLACE_STEPS[step])
    chart = fm.Chart(frame).transform(_FOLLOWING_PIVOT, *_IN_PLACE_STEPS[step])
    return chart.mark_point().encode(x=x, y="g")


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("composed", [False, True], ids=["plain", "composed"])
@pytest.mark.parametrize("step", list(_IN_PLACE_STEPS))
def test_a_step_that_rewrites_a_pivoted_name_draws_and_warns_once_over_no_rows(
    step: str, composed: bool, output: str
):
    populated = _in_place_chart(_pivot_rows().rename({"k": "g"}), step, composed=composed)
    assert _render_following(populated, output) == []
    chart = _in_place_chart(_following_frame(), step, composed=composed)
    assert _render_following(chart, output) == [_NO_ROWS_TO_DRAW]


# ---------------------------------------------------------------------------
# Composite leaves and facet panels
# ---------------------------------------------------------------------------


def _published_twin() -> pl.DataFrame:
    return pl.DataFrame(
        {"g": [], "a": [], "b": []}, schema={"g": pl.Utf8, "a": pl.Float64, "b": pl.Float64}
    )


def _wide_leaf(frame: pl.DataFrame) -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x="a", y="b")


def _zero_row_pivot_leaf() -> fm.Chart:
    return _following_chart([], "a", "b")


def _populated_leaf() -> fm.Chart:
    wide = pl.DataFrame({"g": ["p", "q"], "a": [1.0, 3.0], "b": [2.0, 4.0]})
    return _wide_leaf(wide)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("concat", [fm.hconcat, fm.vconcat], ids=["hconcat", "vconcat"])
def test_a_zero_row_pivot_leaf_beside_a_populated_leaf_equals_the_plain_zero_row_leaf(
    concat, output: str
):
    def render(leaf: fm.Chart):
        grid = concat(leaf, _populated_leaf())
        return _render_capturing(grid.to_svg if output == "static" else _interactive(grid))

    page, messages = render(_zero_row_pivot_leaf())
    twin_page, twin_messages = render(_wide_leaf(_published_twin()))
    assert messages == twin_messages == []
    assert page == twin_page


@pytest.mark.parametrize("concat", [fm.hconcat, fm.vconcat], ids=["hconcat", "vconcat"])
def test_a_grid_of_only_zero_row_pivots_still_raises(concat):
    grid = concat(_zero_row_pivot_leaf(), _zero_row_pivot_leaf())
    with pytest.raises(ValueError, match="every child's data is empty; nothing to render"):
        grid.to_svg()


def _faceted_pivot_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "k": ["m", "m", "m", "m"],
            "f": ["p", "p", "q", "q"],
            "h": ["a", "b", "a", "b"],
            "v": [1.0, 2.0, -3.0, -4.0],
        }
    )


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_filter_before_a_pivot_that_empties_one_facet_panel_draws_the_other(output: str):
    """The filter leaves panel ``q`` without rows, so the pivot runs once per panel and ``q``'s
    run has no rows; the chart equals the one over the hand-filtered, hand-pivoted frame."""
    pivot = fm.transform_pivot("h", "v", groupby=["k", "f"])
    chart = (
        fm.Chart(_faceted_pivot_rows())
        .transform(fm.transform_filter("datum.v > 0"), pivot)
        .mark_point()
        .encode(x="a", y="b")
        .facet("f")
    )
    wide = pl.DataFrame({"k": ["m"], "f": ["p"], "a": [1.0], "b": [2.0]})
    twin = fm.Chart(wide).mark_point().encode(x="a", y="b").facet("f")
    page, messages = _render_capturing(chart.to_svg if output == "static" else _interactive(chart))
    twin_page, twin_messages = _render_capturing(
        twin.to_svg if output == "static" else _interactive(twin)
    )
    assert messages == twin_messages == []
    assert page == twin_page
    if output == "static":
        assert page.count("<circle") == 1


def test_a_zero_row_pivot_draws_the_page_of_a_zero_row_float64_frame():
    """The published columns are ``Float64``: the page equals the zero-row ``Float64`` frame's and
    differs from a text-typed frame's."""
    page, messages = _render_capturing(_zero_row_pivot_leaf().to_svg)
    twin, twin_messages = _render_capturing(_wide_leaf(_published_twin()).to_svg)
    text = _published_twin().cast({"a": pl.Utf8, "b": pl.Utf8})
    text_page, _ = _render_capturing(_wide_leaf(text).to_svg)
    assert messages == twin_messages == [_NO_ROWS_TO_DRAW]
    assert page == twin
    assert page != text_page


# ---------------------------------------------------------------------------
# Steps that read what the pivot publishes without naming it
# ---------------------------------------------------------------------------


def _melted_chart(frame: pl.DataFrame, unpivot: Any, *, composed: bool) -> fm.Chart:
    if composed:
        layer = lambda: fm.Chart(frame).mark_point().encode(x="variable", y="value")  # noqa: E731
        return (layer() + layer()).transform(_FOLLOWING_PIVOT, unpivot)
    chart = fm.Chart(frame).transform(_FOLLOWING_PIVOT, unpivot)
    return chart.mark_point().encode(x="variable", y="value")


_UNPIVOTS = {
    "no_value_vars": lambda: fm.Unpivot(id_vars=["g"]),
    "value_vars": lambda: fm.Unpivot(id_vars=["g"], value_vars=["a", "b"]),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("composed", [False, True], ids=["plain", "composed"])
@pytest.mark.parametrize("unpivot", list(_UNPIVOTS))
def test_an_unpivot_after_a_zero_row_pivot_draws_and_warns_once(
    unpivot: str, composed: bool, output: str
):
    populated = _melted_chart(
        _pivot_rows().rename({"k": "g"}), _UNPIVOTS[unpivot](), composed=composed
    )
    assert _render_following(populated, output) == []
    zero_row = _melted_chart(_following_frame(), _UNPIVOTS[unpivot](), composed=composed)
    assert _render_following(zero_row, output) == [_NO_ROWS_TO_DRAW]


def _named(step: dict, name: str) -> dict:
    return {**step, "name": name}


_NAMED_STEP_CHARTS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    # a named step publishes beside the chain: the steps after it still read the pivot's `a`
    "read_after_named": lambda df: (
        fm.Chart(df)
        .transform(
            _FOLLOWING_PIVOT,
            _named(fm.transform_calculate("a", "1"), "side"),
            fm.transform_calculate("z", "datum.a + 1"),
        )
        .mark_point()
        .encode(x="z", y="g")
    ),
    "draw_after_named": lambda df: (
        fm.Chart(df)
        .transform(_FOLLOWING_PIVOT, _named(fm.transform_calculate("a", "1"), "side"))
        .mark_point()
        .encode(x="a", y="g")
    ),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("door", list(_NAMED_STEP_CHARTS))
def test_a_named_step_after_a_zero_row_pivot_does_not_hide_the_name_it_computes(
    door: str, output: str
):
    populated = _NAMED_STEP_CHARTS[door](_pivot_rows().rename({"k": "g"}))
    assert _render_following(populated, output) == []
    assert _render_following(_NAMED_STEP_CHARTS[door](_following_frame()), output) == [
        _NO_ROWS_TO_DRAW
    ]
