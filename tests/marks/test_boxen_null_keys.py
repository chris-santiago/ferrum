"""A boxen whose group keys are null warns and draws what is left, on every door.

``LetterValue`` splits a boxen by its category and by every other grouping channel (``color``,
``detail``, a caller's ``groupby``). A row with a null on any of them belongs to no group and is
dropped, as every grouped statistic drops it, the category included. When that drop removes every
row the boxen draws its empty frame and says so with the shared sentence ``mark_density`` gives
(``every row has a null in its groupby key (h), ...; the grouped output is empty``), instead of
failing the whole chart. Faceted, the sentence is given once when every panel is emptied, and a
panel alone emptied keeps its place beside the drop count. A drop that leaves groups keeps its
count and draws them. A boxen with usable keys is unchanged.

Net-new coverage for a mark, so it lives in its own module under the repo's test-file
convention (see ``test_boxen_palette.py``). The empty frame an all-null boxen draws is compared
with the one ``mark_boxplot`` (``catplot(kind="box")``) draws on the same frame, which has no
group to place either: the same plot area, the same axes, no mark and no legend.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import axis_tick_labels, text_runs
from tests._svg_rects import mark_groups_by_frame, mark_layer
from tests.test_null_group_warning import (
    _NO_FACET_ROWS_WARNING,
    _NO_ROWS_TO_DRAW,
    _all_keys_null_message,
    _layer_message,
    _null_key_dropped_message,
    _panel_message,
    _render_capturing,
    _slot_message,
)

_STATISTIC = "LetterValue"
_NAN = float("nan")
_INF = float("inf")


# ---------------------------------------------------------------------------
# Frames and doors
# ---------------------------------------------------------------------------


def _frame(**columns: list[Any] | None) -> pl.DataFrame:
    """Eight rows: categories ``c`` (``a``, ``b``), hues ``h`` (``x``, ``y``), details ``d`` (``p``,
    ``q``), facet panels ``f`` (``F1``, ``F2``) and values ``v``. A column given as ``None`` is
    null on every row; a list replaces the column."""
    base: dict[str, list[Any]] = {
        "c": ["a"] * 4 + ["b"] * 4,
        "h": ["x", "y"] * 4,
        "d": ["p", "q"] * 4,
        "f": ["F1"] * 4 + ["F2"] * 4,
        "v": [float(i) for i in range(1, 9)],
    }
    for name, values in columns.items():
        base[name] = [None] * 8 if values is None else values
    schema = {"c": pl.Utf8, "h": pl.Utf8, "d": pl.Utf8, "f": pl.Utf8, "v": pl.Float64}
    return pl.DataFrame(base, schema=schema)


_MARK_OF_KIND = {"boxen": "mark_boxen", "box": "mark_boxplot"}


def _mark_door(
    df: pl.DataFrame, kind: str, *, horizontal: bool = False, grouping: str = "color"
) -> fm.Chart:
    """The mark method's boxen (or its box sibling) of ``v`` by ``c`` split by ``h``."""
    options = {"horizontal": True} if horizontal else {}
    positions = {"x": "v", "y": "c"} if horizontal else {"x": "c", "y": "v"}
    chart = getattr(fm.Chart(df), _MARK_OF_KIND[kind])(**options)
    return chart.encode(**positions, **{grouping: "h"})


def _catplot_door(df: pl.DataFrame, kind: str, *, horizontal: bool = False, **facet: str):
    positions = {"x": "v", "y": "c", "orient": "h"} if horizontal else {"x": "c", "y": "v"}
    return fm.catplot(df, hue="h", kind=kind, **positions, **facet)


#: Every way to spell an unfaceted boxen split by ``h``: ``(kind, df) -> chart``. ``kind`` is
#: ``"boxen"`` or its sibling ``"box"``.
_UNFACETED_DOORS: dict[str, Callable[[pl.DataFrame, str], Any]] = {
    "color": lambda df, kind: _mark_door(df, kind),
    "detail": lambda df, kind: _mark_door(df, kind, grouping="detail"),
    "horizontal": lambda df, kind: _mark_door(df, kind, horizontal=True),
    "catplot": lambda df, kind: _catplot_door(df, kind),
    "catplot-orient-h": lambda df, kind: _catplot_door(df, kind, horizontal=True),
}

#: Every way to spell a boxen split by ``h`` and faceted by ``f``.
_FACETED_DOORS: dict[str, Callable[[pl.DataFrame, str], Any]] = {
    "facet": lambda df, kind: _mark_door(df, kind).facet("f"),
    "catplot-col": lambda df, kind: _catplot_door(df, kind, col="f"),
}

_NULL_HUE_FRAMES = {
    "h-null": lambda: _frame(h=None),
    "h-and-c-null": lambda: _frame(h=None, c=None),
}


def _band_rects(svg: str) -> int:
    """The boxen band rects a render's marks hold (the plot area's own rect is not in them)."""
    return mark_layer(svg).count("<rect")


def _assert_empty_frame(svg: str, sibling_svg: str, titles: tuple[str, ...] = ("c", "v")) -> None:
    """``svg`` draws the plot area, axes and titles ``sibling_svg`` draws, with no mark, no
    legend entry and no hue title. ``titles`` are the axis titles both carry."""
    frames = mark_groups_by_frame(svg)
    assert frames, "no plot area was drawn"
    assert frames == mark_groups_by_frame(sibling_svg)
    assert all(content == "" for _, content in frames)
    assert text_runs(svg) == text_runs(sibling_svg)
    assert set(titles) <= set(text_runs(svg))
    assert ">h<" not in svg


def _assert_plot_area(svg: str, panels: int = 1, titles: tuple[str, ...] = ("c", "v")) -> None:
    """``svg`` draws ``panels`` plot areas, each with no mark, and the axis ``titles``."""
    frames = mark_groups_by_frame(svg)
    assert len(frames) == panels
    assert all(content == "" for _, content in frames)
    assert set(titles) <= set(text_runs(svg))


def _chart_and_sibling(door: Callable[[pl.DataFrame, str], Any], df: pl.DataFrame):
    return door(df, "boxen"), door(df, "box")


# ---------------------------------------------------------------------------
# Every key null, unfaceted: the frame draws and the chart says why it is empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame", sorted(_NULL_HUE_FRAMES))
@pytest.mark.parametrize("door", sorted(_UNFACETED_DOORS))
def test_a_boxen_whose_every_hue_key_is_null_draws_its_empty_frame_and_names_the_split(
    door: str, frame: str
):
    """Every ``h`` null, with the category null too or not, on every spelling of the boxen: the
    empty split and the empty chart are each named once, the frame has its axes and nothing is
    drawn."""
    df = _NULL_HUE_FRAMES[frame]()
    boxen, box = _chart_and_sibling(_UNFACETED_DOORS[door], df)
    svg, shown = _render_capturing(boxen.to_svg)
    assert shown == [_all_keys_null_message(_STATISTIC, "c, h"), _NO_ROWS_TO_DRAW]
    _assert_empty_frame(svg, _render_capturing(box.to_svg)[0])


_NULL_HUE_DTYPES = {
    "null-dtype": pl.lit(None),
    "float64": pl.lit(None, dtype=pl.Float64),
    "int64": pl.lit(None, dtype=pl.Int64),
}


@pytest.mark.parametrize("faceted", [False, True], ids=["unfaceted", "faceted"])
@pytest.mark.parametrize("dtype", sorted(_NULL_HUE_DTYPES))
def test_a_boxen_whose_hue_is_null_on_every_row_of_any_dtype_names_the_split_and_draws_no_band(
    dtype: str, faceted: bool
):
    """A ``pl.Null`` hue (what an all-``None`` column is), and a float or integer hue null on
    every row, are as unusable as a text one: the same sentences as a text hue, the plot area and
    axis titles drawn with no band. Faceted, the one sentence stands alone and both panels are
    drawn empty with their labels. (The colour legend such a numeric column leaves is not
    asserted.)"""
    df = _frame().with_columns(_NULL_HUE_DTYPES[dtype].alias("h"))
    chart = fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h")
    svg, shown = _render_capturing((chart.facet("f") if faceted else chart).to_svg)
    expected = [_all_keys_null_message(_STATISTIC, "c, h")]
    assert shown == (expected if faceted else expected + [_NO_ROWS_TO_DRAW])
    assert _band_rects(svg) == 0
    _assert_plot_area(svg, 2 if faceted else 1)
    if faceted:
        assert {"F1", "F2"} <= set(text_runs(svg))


def test_a_boxen_whose_float_hue_mixes_null_and_nan_drops_the_nulls_and_keeps_the_nan_group():
    """A null hue is dropped and counted; a NaN hue is a key, so its group is drawn."""
    df = _frame().with_columns(pl.Series("h", [None, _NAN] * 4, dtype=pl.Float64))
    svg, shown = _render_capturing(fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h").to_svg)
    assert shown == [_null_key_dropped_message(_STATISTIC, 4, "h")]
    assert _band_rects(svg) > 0


def test_a_boxen_whose_hue_and_detail_nulls_together_cover_every_row_names_both_columns():
    """Neither column is null on every row, but no row has both: the boxen has no group, and the
    sentence names the two columns it could not split by."""
    df = _frame(h=["x", "x", None, None] * 2, d=[None, None, "p", "q"] * 2)
    chart = fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h", detail="d")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message(_STATISTIC, "c, h, d"), _NO_ROWS_TO_DRAW]
    sibling = fm.Chart(df).mark_boxplot().encode(x="c", y="v", color="h", detail="d")
    _assert_empty_frame(svg, _render_capturing(sibling.to_svg)[0])


# ---------------------------------------------------------------------------
# Every key null, faceted: one sentence, every panel drawn empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame", sorted(_NULL_HUE_FRAMES))
@pytest.mark.parametrize("door", sorted(_FACETED_DOORS))
def test_a_faceted_boxen_whose_every_hue_key_is_null_names_the_split_once_and_draws_every_panel(
    door: str, frame: str
):
    """Every panel is emptied, so the one empty-split sentence stands in for the per-panel counts.
    No ``NoRowsToDraw`` (the panels exist), and the two panels' frames are drawn with their labels."""
    df = _NULL_HUE_FRAMES[frame]()
    boxen, box = _chart_and_sibling(_FACETED_DOORS[door], df)
    svg, shown = _render_capturing(boxen.to_svg)
    assert shown == [_all_keys_null_message(_STATISTIC, "c, h")]
    assert len(mark_groups_by_frame(svg)) == 2
    assert {"F1", "F2"} <= set(text_runs(svg))
    _assert_empty_frame(svg, _render_capturing(box.to_svg)[0])


@pytest.mark.parametrize("door", sorted(_FACETED_DOORS))
def test_a_faceted_boxen_with_one_panel_of_null_hues_draws_that_panel_empty_beside_the_count(
    door: str,
):
    """Panel ``F2``'s hue is null on every row: the count is over the whole frame, ``F2`` is drawn
    empty and ``F1`` keeps its bands."""
    df = _frame(h=["x", "y"] * 2 + [None] * 4)
    svg, shown = _render_capturing(_FACETED_DOORS[door](df, "boxen").to_svg)
    assert shown == [_null_key_dropped_message(_STATISTIC, 4, "h")]
    assert [content != "" for _, content in mark_groups_by_frame(svg)] == [True, False]
    assert {"F1", "F2"} <= set(text_runs(svg))


# ---------------------------------------------------------------------------
# A drop that leaves groups, and a drop that leaves only emptied groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("door", sorted(_UNFACETED_DOORS))
def test_a_boxen_with_some_null_hues_counts_the_drop_and_draws_the_usable_groups(door: str):
    """The null rows are dropped and counted, and the chart is the one drawn from the other rows
    alone."""
    df = _frame(h=["x", "y", "x", "y", "x", "y", None, None])
    svg, shown = _render_capturing(_UNFACETED_DOORS[door](df, "boxen").to_svg)
    assert shown == [_null_key_dropped_message(_STATISTIC, 2, "h")]
    assert mark_layer(svg) != ""
    kept = _UNFACETED_DOORS[door](df.head(6), "boxen")
    assert svg == _render_capturing(kept.to_svg)[0]


_ON_NULL_KEYS = [
    pytest.param(None, id="null"),
    pytest.param(_NAN, id="nan"),
    pytest.param(_INF, id="infinite"),
]


@pytest.mark.parametrize("unusable", _ON_NULL_KEYS)
def test_a_boxen_whose_valued_rows_all_have_a_null_hue_keeps_the_empty_group_and_names_it(
    unusable: float | None,
):
    """Hue ``x`` has no usable value and the only valued rows have a null hue: the drop is counted
    and group ``x`` keeps its place, named once as a group that draws nothing, as it is when its
    values are infinite."""
    df = pl.DataFrame(
        {"c": ["a"] * 4, "h": ["x", "x", None, None], "v": [unusable, unusable, 1.0, 2.0]},
        schema={"c": pl.Utf8, "h": pl.Utf8, "v": pl.Float64},
    )
    chart = fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [
        _null_key_dropped_message(_STATISTIC, 2, "h"),
        _slot_message("mark_boxen", "y", "x", "h"),
        _layer_message("mark_boxen"),
    ]
    assert ">x<" in svg and ">a<" in svg


def test_a_faceted_boxen_panel_whose_valued_rows_all_have_a_null_hue_keeps_its_place():
    """Panel ``F2``'s usable hue has no value and its valued rows have a null hue: the drop is
    counted over the frame, the panel keeps its frame and its group (named, drawing nothing),
    while ``F1`` draws."""
    df = pl.DataFrame(
        {
            "c": ["a"] * 8,
            "h": ["x", "y", "x", "y", "x", "x", None, None],
            "f": ["F1"] * 4 + ["F2"] * 4,
            "v": [1.0, 2.0, 3.0, 4.0, None, None, 5.0, 6.0],
        },
        schema={"c": pl.Utf8, "h": pl.Utf8, "f": pl.Utf8, "v": pl.Float64},
    )
    chart = fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h").facet("f")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [
        _null_key_dropped_message(_STATISTIC, 2, "h"),
        _slot_message("mark_boxen", "y", "x", "h", panel="f=F2"),
        _panel_message("mark_boxen", "y", "f=F2"),
    ]
    assert [content != "" for _, content in mark_groups_by_frame(svg)] == [True, False]
    assert {"F1", "F2"} <= set(text_runs(svg))


# ---------------------------------------------------------------------------
# The null category is a group key: its rows are dropped, as every other key's are
# ---------------------------------------------------------------------------

_ALL_NULL_C = _all_keys_null_message(_STATISTIC, "c")

_CATEGORY_DOORS = {
    "mark": lambda df: fm.Chart(df).mark_boxen().encode(x="c", y="v"),
    "catplot": lambda df: fm.catplot(df, x="c", y="v", kind="boxen"),
}

_FACETED_CATEGORY_DOORS = {
    "facet": lambda df: fm.Chart(df).mark_boxen().encode(x="c", y="v").facet("f"),
    "catplot-col": lambda df: fm.catplot(df, x="c", y="v", col="f", kind="boxen"),
}


@pytest.mark.parametrize("door", sorted(_CATEGORY_DOORS))
def test_a_boxen_whose_every_category_is_null_warns_and_draws_its_empty_frame(door: str):
    """No category is usable, so the boxen has no group: the empty-split and empty-chart sentences
    and the frame its box sibling draws, with no ``null`` band."""
    df = _frame(c=None)
    svg, shown = _render_capturing(_CATEGORY_DOORS[door](df).to_svg)
    assert shown == [_ALL_NULL_C, _NO_ROWS_TO_DRAW]
    assert ">null<" not in svg
    sibling = fm.Chart(df).mark_boxplot().encode(x="c", y="v")
    _assert_empty_frame(svg, _render_capturing(sibling.to_svg)[0])


@pytest.mark.parametrize("door", sorted(_FACETED_CATEGORY_DOORS))
def test_a_faceted_boxen_whose_every_category_is_null_names_it_once_and_draws_every_panel(
    door: str,
):
    df = _frame(c=None)
    svg, shown = _render_capturing(_FACETED_CATEGORY_DOORS[door](df).to_svg)
    assert shown == [_ALL_NULL_C]
    assert ">null<" not in svg
    assert {"F1", "F2"} <= set(text_runs(svg))
    _assert_plot_area(svg, 2)
    assert _band_rects(svg) == 0


def test_a_boxen_whose_every_category_is_null_gives_the_interactive_route_the_static_warnings():
    """The interactive route renders once, at ``.interactive()``, and ``to_html()`` reuses that
    scene, so its warnings are the static route's once, and the scene holds no mark and no
    ``null`` label."""
    chart = _CATEGORY_DOORS["mark"](_frame(c=None))
    static = _render_capturing(chart.to_svg)[1]
    assert static == [_ALL_NULL_C, _NO_ROWS_TO_DRAW]
    scene: dict[str, Any] = {}

    def build() -> str:
        interactive = chart.interactive()
        scene["json"] = interactive.scene_json
        return interactive.to_html()

    assert _render_capturing(build)[1] == static
    panels = json.loads(scene["json"])["panels"]
    assert sum(len(b["nodes"]) for p in panels for b in p["marks"]) == 0
    assert '"null"' not in scene["json"]


def test_a_boxen_whose_color_field_column_is_null_on_every_row_warns_and_draws_its_empty_frame():
    """``color_field=`` moves the category to ``h``, so its nulls are dropped as the category's
    are."""
    chart = fm.Chart(_frame(h=None)).mark_boxen(color_field="h").encode(x="c", y="v")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message(_STATISTIC, "h"), _NO_ROWS_TO_DRAW]
    assert ">null<" not in svg
    sibling_svg = _render_capturing(
        fm.Chart(_frame(h=None)).mark_boxplot().encode(x="h", y="v").to_svg
    )[0]
    frames = mark_groups_by_frame(svg)
    assert frames and all(content == "" for _, content in frames)
    assert frames == mark_groups_by_frame(sibling_svg)
    assert text_runs(svg) == text_runs(sibling_svg)
    assert {"h", "v"} <= set(text_runs(svg)), (
        "the category the colour field names is the axis title"
    )


def test_a_boxen_with_some_null_categories_drops_and_counts_them_and_draws_the_rest():
    df = _frame(c=["a", "a", "b", "b", "a", "a", None, None])
    svg, shown = _render_capturing(fm.Chart(df).mark_boxen().encode(x="c", y="v").to_svg)
    assert shown == [_null_key_dropped_message(_STATISTIC, 2, "c")]
    assert ">null<" not in svg
    kept = _render_capturing(fm.Chart(df.head(6)).mark_boxen().encode(x="c", y="v").to_svg)
    assert kept[1] == []
    assert svg == kept[0]
    assert _band_rects(svg) > 0


def test_a_nan_hue_is_a_key_of_its_own_and_is_not_dropped():
    """A float hue of NaN is not a null key: no row is dropped and the boxen draws."""
    df = _frame().with_columns(pl.lit(_NAN).alias("h"))
    svg, shown = _render_capturing(fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h").to_svg)
    assert shown == []
    assert mark_layer(svg) != ""


# ---------------------------------------------------------------------------
# A zero-row frame has no key to drop: it still reports that there is nothing to draw
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hue", [{}, {"color": "h"}], ids=["no-hue", "hue"])
def test_a_boxen_on_a_zero_row_frame_reports_that_the_chart_has_no_rows(hue: dict[str, str]):
    df = _frame().head(0)
    chart = fm.Chart(df).mark_boxen().encode(x="c", y="v", **hue)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_NO_ROWS_TO_DRAW]
    assert axis_tick_labels(svg, axis="y")
    assert mark_layer(svg) == ""
    _, faceted = _render_capturing(chart.facet("f").to_svg)
    assert faceted == [_NO_FACET_ROWS_WARNING]


# ---------------------------------------------------------------------------
# A caller-written step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame", sorted(_NULL_HUE_FRAMES))
def test_a_caller_written_letter_value_step_whose_every_group_key_is_null_returns_no_rows(
    frame: str,
):
    """The same empty split, from the step itself: no refusal, and the chart reading its output
    draws the frame a zero-row frame draws."""

    def chart(df: pl.DataFrame) -> fm.Chart:
        step = fm.LetterValue("v", group="c", groupby=["h"])
        return fm.Chart(df).transform(step).mark_point().encode(x="group", y="lower")

    df = _NULL_HUE_FRAMES[frame]()
    svg, shown = _render_capturing(chart(df).to_svg)
    assert shown == [_all_keys_null_message(_STATISTIC, "c, h"), _NO_ROWS_TO_DRAW]
    zero_rows = _render_capturing(chart(df.head(0)).to_svg)[0]
    assert mark_groups_by_frame(svg) == mark_groups_by_frame(zero_rows)
    assert text_runs(svg) == text_runs(zero_rows)
    assert mark_layer(svg) == ""


# ---------------------------------------------------------------------------
# Composition: the other operand still draws
# ---------------------------------------------------------------------------

_OTHER_ROWS = pl.DataFrame({"c": ["a", "b"], "v": [1.0, 2.0]})


def _all_null_boxen() -> fm.Chart:
    return fm.Chart(_frame(h=None)).mark_boxen().encode(x="c", y="v", color="h")


def _points() -> fm.Chart:
    return fm.Chart(_OTHER_ROWS).mark_point().encode(x="c", y="v")


_COMPOSED_DOORS: dict[str, tuple[Callable[[fm.Chart, fm.Chart], Any], list[str]]] = {
    "plus_boxen_first": (lambda b, p: b + p, []),
    "plus_boxen_second": (lambda b, p: p + b, []),
    "layer_chart": (lambda b, p: fm.LayerChart(b, p), []),
    "layer_chart_reversed": (lambda b, p: fm.LayerChart(p, b), []),
    # A concat leaf is its own panel, so the boxen's draws nothing and says so.
    "hconcat_leaf": (lambda b, p: fm.hconcat(b, p), [_NO_ROWS_TO_DRAW]),
}


@pytest.mark.parametrize("door", sorted(_COMPOSED_DOORS))
def test_an_all_null_boxen_beside_another_chart_renders_with_its_warning_and_the_other_drawn(
    door: str,
):
    build, leaf_messages = _COMPOSED_DOORS[door]
    svg, shown = _render_capturing(build(_all_null_boxen(), _points()).to_svg)
    assert shown == [_all_keys_null_message(_STATISTIC, "c, h"), *leaf_messages]
    assert mark_layer(svg).count("<circle") == len(_OTHER_ROWS)
    assert _band_rects(svg) == 0
    assert ">h<" not in svg and ">null<" not in svg
    if door == "hconcat_leaf":
        # The boxen's own panel is there, with its axes and no band, beside the points' panel.
        contents = [content for _, content in mark_groups_by_frame(svg)]
        assert len(contents) == 2
        assert contents[0] == ""
        assert contents[1].count("<circle") == len(_OTHER_ROWS)
        assert "c" in text_runs(svg) and "v" in text_runs(svg)


# ---------------------------------------------------------------------------
# The interactive route gives the sibling statistic's warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("faceted", [False, True], ids=["unfaceted", "faceted"])
def test_an_all_null_boxen_gives_the_interactive_route_what_its_sibling_statistic_gives(
    faceted: bool,
):
    """The whole route (``.interactive()`` and ``to_html()``) over the same frame warns for an
    all-null boxen exactly as it does for an all-null density, with the statistic's name swapped,
    and raises nothing else."""

    def route(chart: fm.Chart) -> list[str]:
        chart = chart.facet("f") if faceted else chart
        return _render_capturing(lambda: chart.interactive().to_html())[1]

    df = _frame(h=None)
    density = route(fm.Chart(df).mark_density().encode(x="v", color="h"))
    kde = _all_keys_null_message("Kde", "h")
    assert kde in density
    expected = [_all_keys_null_message(_STATISTIC, "c, h") if m == kde else m for m in density]
    assert route(fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h")) == expected


# ---------------------------------------------------------------------------
# Unchanged: a boxen with usable keys, and the scaled-channel refusal
# ---------------------------------------------------------------------------

_WELL_FORMED = {
    "hue": lambda: _mark_door(_frame(), "boxen"),
    "faceted-hue": lambda: _mark_door(_frame(), "boxen").facet("f"),
    "catplot-hue": lambda: _catplot_door(_frame(), "boxen"),
}

#: SHA-256 of each well-formed boxen's SVG, so a boxen with usable keys stays byte for byte as it
#: was drawn.
_WELL_FORMED_DIGESTS = {
    "hue": "c1bfdf8ab209ca736f7256b955b0de40cf3fd98458a9616a59eeeccf3fea34c3",
    "faceted-hue": "c6cb500737e3b977c3f7fa9ae431297d12b480d043a09a896c5ad5fb4fcdab4c",
    "catplot-hue": "c1bfdf8ab209ca736f7256b955b0de40cf3fd98458a9616a59eeeccf3fea34c3",
}


@pytest.mark.parametrize("case", sorted(_WELL_FORMED))
def test_a_boxen_with_usable_keys_warns_nothing_and_draws_as_it_did(case: str):
    svg, shown = _render_capturing(_WELL_FORMED[case]().to_svg)
    assert shown == []
    assert hashlib.sha256(svg.encode()).hexdigest() == _WELL_FORMED_DIGESTS[case]


@pytest.mark.parametrize(
    "channel", ["opacity", "fill_opacity", "stroke_opacity", "size", "stroke_width"]
)
def test_a_boxen_channel_bound_to_a_text_category_is_still_refused(channel: str):
    chart = fm.Chart(_frame()).mark_boxen().encode(x="c", y="v", **{channel: "c"})
    with pytest.raises(ValueError, match=f"{channel}: column 'c' has unsupported dtype: Utf8"):
        chart.to_svg()
