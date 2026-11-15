"""A row whose category, hue or grouping value is null belongs to no group, on every mark.

One rule across every category mark, statistic and plain mark: a row with a null categorical key
on a channel the mark draws is dropped and counted once per field in the shared sentence
(``<entry>: N rows have a null 'c' and belong to no group, so they were dropped; fill or filter
the nulls in 'c'``). When every row is null on a key the shared every-key-null sentence names the
column, the frame draws with its axes and no mark, and no ``null`` band, legend entry or
misattributed mark is ever drawn. A statistic keeps speaking for its own groupby keys. A null on
a channel the mark does not draw, on a quantitative colour or on a value channel (size, opacity,
width, angle) is not a key and is left as it was, and a chart with no null key draws as it did.

Each drop case compares the chart with the same chart built from the rows the drop leaves: a
band, a legend entry, a value-axis extent or a mark that was kept or went missing makes the two
differ. The null rows carry a value far outside the kept rows' range (100 and up), so a value
axis that still spans a dropped row differs too.

Net-new coverage across the mark families, so it has its own module under the repo's test-file
convention (see ``test_boxen_palette.py``).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import warnings
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import text_runs, value_axis_ladders
from tests._svg_rects import mark_groups, mark_groups_by_frame, mark_layer
from tests.marks.test_boxen_null_keys import _assert_empty_frame, _assert_plot_area, _band_rects
from tests.test_null_group_warning import (
    _NO_FACET_ROWS_WARNING,
    FIT_FEW_POINTS,
    _NO_ROWS_TO_DRAW,
    _all_keys_null_fallback_message,
    _all_keys_null_message,
    _detail_ignored,
    _facet_key_dropped_message,
    _layer_message,
    _list_message,
    _not_drawn,
    _slot_message,
    _null_key_dropped_message,
    _panel_fit_message,
    _render_capturing,
)
from ferrum._warn import reset_warnings
from ferrum.annotation.coords import OrdinalCategoryCoord
from ferrum.marks.composite import _boxen_materialized_depth

_NAN = float("nan")
_SOME = "some"
_ALL = "all"

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _frame(*, c: Any = "none", h: Any = "none", f: Any = "none") -> pl.DataFrame:
    """Twelve rows: category ``c`` (``a``, ``b``), hue ``h`` (``x``, ``y``), panel ``f``
    (``F1``, ``F2``), value ``v`` (1 to 8, then 100 to 103) and a quantitative ``w`` (0 to 11).

    A column given as ``"some"`` is null on rows 9 to 12 (the four rows of large ``v``), as
    ``"all"`` is null on every row, and a list replaces the column."""
    bases: dict[str, list[Any]] = {
        "c": ["a", "b"] * 6,
        "h": ["x", "y"] * 6,
        "f": ["F1", "F2"] * 6,
    }
    columns: dict[str, list[Any]] = {}
    for name, mode in {"c": c, "h": h, "f": f}.items():
        if isinstance(mode, list):
            columns[name] = mode
        elif mode == _ALL:
            columns[name] = [None] * 12
        elif mode == _SOME:
            columns[name] = bases[name][:8] + [None] * 4
        else:
            columns[name] = bases[name]
    return pl.DataFrame(
        {
            **columns,
            "v": [float(i) for i in range(1, 9)] + [100.0, 101.0, 102.0, 103.0],
            "w": [float(i) for i in range(12)],
        },
        schema={"c": pl.Utf8, "h": pl.Utf8, "f": pl.Utf8, "v": pl.Float64, "w": pl.Float64},
    )


def _bands(svg: str) -> set[str]:
    """The category labels among the rendered text (a literal ``"null"`` category reads ``null``)."""
    return {run for run in text_runs(svg) if run in ("a", "b", "null")}


def _drawn(svg: str, tag: str) -> int:
    """How many ``<tag>`` elements the plot area holds."""
    return len(re.findall(rf"<{tag}\b", mark_layer(svg)))


def _panel_marks(svg: str) -> list[bool]:
    """Whether each panel's plot area holds a mark, in document order."""
    return [content != "" for _, content in mark_groups_by_frame(svg)]


def _scene_marks(scene_json: str) -> int:
    panels = json.loads(scene_json)["panels"]
    return sum(len(b["nodes"]) for p in panels for b in p["marks"])


def _route(chart: Any) -> tuple[list[str], str]:
    """The whole interactive route (``.interactive()`` then ``to_html()``) in one capture: every
    warning it raised and the scene it built."""
    scene: dict[str, str] = {}

    def build() -> str:
        interactive = chart.interactive()
        scene["json"] = interactive.scene_json
        return interactive.to_html()

    shown = _render_capturing(build)[1]
    return shown, scene["json"]


def _assert_drops(
    build: Callable[[pl.DataFrame], Any],
    df: pl.DataFrame,
    kept: pl.DataFrame,
    expected: list[str],
    *,
    kept_expected: list[str] | None = None,
) -> str:
    """``build(df)`` raises ``expected`` and draws exactly what ``build(kept)`` draws."""
    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == expected
    kept_svg, kept_shown = _render_capturing(build(kept).to_svg)
    assert kept_shown == (kept_expected or [])
    assert svg == kept_svg
    assert "null" not in text_runs(svg)
    return svg


def _assert_route_matches_static(
    build: Callable[[pl.DataFrame], Any],
    df: pl.DataFrame,
    kept: pl.DataFrame,
    *,
    draws: bool = True,
) -> None:
    """The interactive route warns the static route's list once and builds the scene the
    kept rows build, with no ``null`` label."""
    static = _render_capturing(build(df).to_svg)[1]
    shown, scene = _route(build(df))
    assert shown == static
    kept_shown, kept_scene = _route(build(kept))
    assert kept_shown == []
    assert scene == kept_scene
    if draws:
        assert _scene_marks(scene) > 0
    assert '"null"' not in scene


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------

#: Who speaks for a null category on each ``catplot`` kind: the figure for the kinds that draw a
#: mark per row, the statistic for the others.
_SPEAKER = {
    "strip": "catplot",
    "swarm": "catplot",
    "boxen": "LetterValue",
    "box": "BoxStats",
    "violin": "Violin",
    "point": "Summary",
    "bar": "Summary",
    "count": "Aggregate",
}
#: The kind whose empty frame each kind's is compared with (an empty frame is one frame).
_SIBLING = {
    "strip": "box",
    "swarm": "box",
    "boxen": "box",
    "box": "violin",
    "violin": "box",
    "point": "bar",
    "bar": "point",
}


def _catplot(
    kind: str, *, horizontal: bool = False, **options: Any
) -> Callable[[pl.DataFrame], Any]:
    def build(df: pl.DataFrame) -> Any:
        if kind == "count":
            positions = {"y": "c", "orient": "h"} if horizontal else {"x": "c"}
        elif horizontal:
            positions = {"x": "v", "y": "c", "orient": "h"}
        else:
            positions = {"x": "c", "y": "v"}
        return fm.catplot(df, kind=kind, **positions, **options)

    return build


def _titles(kind: str) -> tuple[str, ...]:
    return ("c",) if kind == "count" else ("c", "v")


class _Plain:
    """A plain mark of ``c`` by ``v``: how to build it, who speaks for it, and the element it
    draws once per row (``None`` when no element of the plot area can be counted)."""

    def __init__(self, method: str, encoding: dict[str, str], element: str | None):
        self.method, self.encoding, self.element = method, encoding, element

    def __call__(self, df: pl.DataFrame, **extra: Any) -> Any:
        return getattr(fm.Chart(df), self.method)().encode(**self.encoding, **extra)


_PLAIN = {
    "point": _Plain("mark_point", {"x": "c", "y": "v"}, "circle"),
    "tick": _Plain("mark_tick", {"x": "c", "y": "v"}, "line"),
    "bar": _Plain("mark_bar", {"x": "c", "y": "v"}, "rect"),
    "line": _Plain("mark_line", {"x": "c", "y": "v"}, None),
    "text": _Plain("mark_text", {"x": "c", "y": "v", "text": "v"}, None),
    "rule": _Plain("mark_rule", {"x": "c", "y": "v"}, "line"),
    "rect": _Plain("mark_rect", {"x": "c", "y": "h", "color": "v"}, "rect"),
    "area": _Plain("mark_area", {"x": "c", "y": "v"}, None),
    "swarm": _Plain("mark_swarm", {"x": "c", "y": "v"}, "circle"),
    "tick-one-axis": _Plain("mark_tick", {"x": "c"}, "line"),
}


def _speaker(name: str) -> str:
    return _PLAIN[name].method


# ---------------------------------------------------------------------------
# A null category: dropped and counted on every mark, no ``null`` band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizontal", [False, True], ids=["vertical", "horizontal"])
@pytest.mark.parametrize("kind", sorted(_SPEAKER))
def test_a_catplot_with_some_null_categories_draws_the_rest_and_counts_the_drop(
    kind: str, horizontal: bool
):
    """Four of twelve rows have a null category: every kind draws bands ``a`` and ``b`` and counts
    the four once, the figure speaking for ``strip`` and ``swarm`` and the statistic for the rest."""
    df = _frame(c=_SOME)
    build = _catplot(kind, horizontal=horizontal)
    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message(_SPEAKER[kind], 4, "c")])
    assert _bands(svg) == {"a", "b"}
    if kind in ("strip", "swarm"):
        assert _drawn(svg, "circle") == 8


@pytest.mark.parametrize("horizontal", [False, True], ids=["vertical", "horizontal"])
@pytest.mark.parametrize("kind", sorted(_SPEAKER))
def test_a_catplot_with_some_null_categories_gives_the_interactive_route_the_static_list(
    kind: str, horizontal: bool
):
    df = _frame(c=_SOME)
    _assert_route_matches_static(_catplot(kind, horizontal=horizontal), df, df.head(8))


def test_a_boxen_mark_with_some_null_categories_counts_the_drop_under_letter_value():
    df = _frame(c=_SOME)

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_boxen().encode(x="c", y="v")

    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message("LetterValue", 4, "c")])
    assert _bands(svg) == {"a", "b"}
    assert _band_rects(svg) > 0


_STATISTIC_MARKS: dict[str, tuple[Callable[[pl.DataFrame], Any], str]] = {
    "boxplot": (lambda df: fm.Chart(df).mark_boxplot().encode(x="c", y="v"), "BoxStats"),
    "violin": (lambda df: fm.Chart(df).mark_violin().encode(x="c", y="v"), "Violin"),
    "errorbar": (lambda df: fm.Chart(df).mark_errorbar().encode(x="c", y="v"), "ErrorExtent"),
    "bar-mean": (
        lambda df: fm.Chart(df).mark_bar().encode(x="c", y=fm.Y("v", aggregate="mean")),
        "Aggregate",
    ),
    "bar-count": (lambda df: fm.Chart(df).mark_bar().encode(x="c", y="count()"), "Aggregate"),
}


@pytest.mark.parametrize("name", sorted(_STATISTIC_MARKS))
def test_a_statistic_mark_with_some_null_categories_keeps_its_own_count(name: str):
    """The statistic marks speak for their own category drop, once, and the mark adds nothing."""
    build, speaker = _STATISTIC_MARKS[name]
    df = _frame(c=_SOME)
    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message(speaker, 4, "c")])
    assert _bands(svg) == {"a", "b"}


@pytest.mark.parametrize("name", sorted(_PLAIN))
def test_a_plain_mark_with_some_null_categories_draws_the_rest_and_counts_the_drop(name: str):
    """No ``null`` band: the four rows are dropped, counted under the mark's entry, and the value
    axis spans only the rows drawn."""
    mark = _PLAIN[name]
    df = _frame(c=_SOME)
    svg = _assert_drops(mark, df, df.head(8), [_null_key_dropped_message(_speaker(name), 4, "c")])
    assert _bands(svg) == {"a", "b"}
    if mark.element is not None:
        assert _drawn(svg, mark.element) == 8


@pytest.mark.parametrize("name", ["point", "tick", "bar", "line", "swarm", "tick-one-axis"])
def test_a_plain_mark_with_some_null_categories_gives_the_interactive_route_the_static_list(
    name: str,
):
    df = _frame(c=_SOME)
    _assert_route_matches_static(_PLAIN[name], df, df.head(8))


# ---------------------------------------------------------------------------
# Every category null: the frame draws with no band and no mark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizontal", [False, True], ids=["vertical", "horizontal"])
@pytest.mark.parametrize("kind", sorted(_SPEAKER))
def test_a_catplot_whose_every_category_is_null_draws_its_empty_frame(kind: str, horizontal: bool):
    """The shared every-key-null sentence and the no-rows sentence, an empty frame, no band: the
    frame its sibling kind draws (``count`` has no sibling with its axis title)."""
    df = _frame(c=_ALL)
    svg, shown = _render_capturing(_catplot(kind, horizontal=horizontal)(df).to_svg)
    assert shown == [_all_keys_null_message(_SPEAKER[kind], "c"), _NO_ROWS_TO_DRAW]
    assert _bands(svg) == set()
    if kind in _SIBLING:
        sibling = _catplot(_SIBLING[kind], horizontal=horizontal)(df)
        _assert_empty_frame(svg, _render_capturing(sibling.to_svg)[0], _titles(kind))
    else:
        _assert_plot_area(svg, titles=_titles(kind))


@pytest.mark.parametrize("kind", ["strip", "swarm", "boxen", "box", "violin", "point", "bar"])
def test_a_catplot_whose_every_category_is_null_gives_the_interactive_route_the_static_list(
    kind: str,
):
    chart = _catplot(kind)(_frame(c=_ALL))
    static = _render_capturing(chart.to_svg)[1]
    shown, scene = _route(chart)
    assert shown == static
    assert _scene_marks(scene) == 0
    assert '"null"' not in scene


_FACETED_KINDS = ["strip", "swarm", "boxen", "box", "violin", "point"]


@pytest.mark.parametrize("door", ["col", "facet"])
@pytest.mark.parametrize("kind", _FACETED_KINDS)
def test_a_faceted_catplot_whose_every_category_is_null_names_it_once_and_draws_every_panel(
    kind: str, door: str
):
    """Once, with no no-rows sentence (the panels exist), and both panels drawn empty with their
    labels."""
    df = _frame(c=_ALL)

    def build(frame: pl.DataFrame, kind: str = kind) -> Any:
        if door == "col":
            return fm.catplot(frame, x="c", y="v", kind=kind, col="f")
        return _catplot(kind)(frame).facet("f")

    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == [_all_keys_null_message(_SPEAKER[kind], "c")]
    assert {"F1", "F2"} <= set(text_runs(svg))
    assert _bands(svg) == set()
    _assert_plot_area(svg, 2)
    sibling = build(df, _SIBLING[kind])
    assert mark_groups_by_frame(svg) == mark_groups_by_frame(_render_capturing(sibling.to_svg)[0])


@pytest.mark.parametrize("name", sorted(_PLAIN))
def test_a_plain_mark_whose_every_category_is_null_draws_its_empty_frame(name: str):
    df = _frame(c=_ALL)
    svg, shown = _render_capturing(_PLAIN[name](df).to_svg)
    assert shown == [_all_keys_null_message(_speaker(name), "c"), _NO_ROWS_TO_DRAW]
    assert _bands(svg) == set()
    _assert_plot_area(svg, titles=("c",))
    if _PLAIN[name].element is not None:
        assert _drawn(svg, _PLAIN[name].element or "") == 0


@pytest.mark.parametrize("name", ["point", "bar", "tick", "line", "swarm"])
def test_a_faceted_plain_mark_whose_every_category_is_null_names_it_once(name: str):
    df = _frame(c=_ALL)
    svg, shown = _render_capturing(_PLAIN[name](df).facet("f").to_svg)
    assert shown == [_all_keys_null_message(_speaker(name), "c")]
    assert {"F1", "F2"} <= set(text_runs(svg))
    _assert_plot_area(svg, 2, titles=("c",))


@pytest.mark.parametrize(
    "dtype", [pl.Null, pl.Int64, pl.Float64, pl.Utf8], ids=["null", "int64", "float64", "utf8"]
)
def test_a_category_typed_nominal_that_is_null_on_every_row_is_a_key_of_any_dtype(dtype: Any):
    """A column the caller types nominal is a category whatever its dtype, so its all-null rows
    give the every-key-null sentence, not a null-position census sentence."""
    df = _frame().with_columns(pl.lit(None, dtype=dtype).alias("c"))
    chart = fm.Chart(df).mark_point().encode(x=fm.X("c", type_="N"), y="v")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message("mark_point", "c"), _NO_ROWS_TO_DRAW]
    _assert_plot_area(svg, titles=("c",))
    assert _drawn(svg, "circle") == 0


@pytest.mark.parametrize("dtype", [pl.Null, pl.Int64, pl.Float64], ids=["null", "int64", "float64"])
def test_an_untyped_numeric_position_that_is_null_on_every_row_stays_a_null_position(dtype: Any):
    """Control: an untyped numeric column is a quantitative position, not a category, so the
    census names the layer as it did and the mark counts nothing."""
    df = _frame().with_columns(pl.lit(None, dtype=dtype).alias("c"))
    svg, shown = _render_capturing(fm.Chart(df).mark_point().encode(x="c", y="v").to_svg)
    assert shown == [_layer_message("mark_point", "x")]
    assert _drawn(svg, "circle") == 0


@pytest.mark.parametrize("column", ["c", "h"])
def test_a_chart_whose_every_row_is_null_on_one_key_column_names_that_column(column: str):
    frame = _frame(**{column: _ALL})
    chart = fm.Chart(frame).mark_point().encode(x="c", y="v", color="h")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message("mark_point", column), _NO_ROWS_TO_DRAW]
    _assert_plot_area(svg, titles=("c",))


# ---------------------------------------------------------------------------
# NaN is not null, and a nominal float column holding both
# ---------------------------------------------------------------------------


def test_a_nominal_category_drops_its_nulls_and_keeps_its_nan_band():
    """A NaN category is a category of its own (not a null), so it keeps its band and its marks."""
    df = pl.DataFrame(
        {"c": [1.0, 2.0, _NAN, None] * 3, "v": [float(i) for i in range(12)]},
    )
    chart = fm.Chart(df).mark_point().encode(x=fm.X("c", type_="N"), y="v")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 3, "c")]
    assert {"1", "2", "NaN"} <= set(text_runs(svg))
    assert "null" not in text_runs(svg)
    assert _drawn(svg, "circle") == 9


def test_a_nominal_hue_drops_its_nulls_and_keeps_its_nan_level():
    df = pl.DataFrame(
        {
            "c": ["a", "b"] * 6,
            "h": [1.0, 2.0, _NAN, None] * 3,
            "v": [float(i) for i in range(12)],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="c", y="v", color=fm.Color("h", type_="N"))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 3, "h")]
    assert {"1", "2", "NaN"} <= set(text_runs(svg))
    assert _drawn(svg, "circle") == 9


# ---------------------------------------------------------------------------
# A null hue or grouping value
# ---------------------------------------------------------------------------

#: The doors whose grouping channel draws no legend entry.
_NO_LEGEND = {"line-detail"}

#: Every plain door that groups rows: ``(build, speaker, element drawn once per row)``.
_HUE_DOORS: dict[str, tuple[Callable[[pl.DataFrame], Any], str, str | None]] = {
    "point-color": (
        lambda df: fm.Chart(df).mark_point().encode(x="c", y="v", color="h"),
        "mark_point",
        "circle",
    ),
    "point-fill": (
        lambda df: fm.Chart(df).mark_point().encode(x="c", y="v", fill="h"),
        "mark_point",
        "circle",
    ),
    "point-stroke": (
        lambda df: fm.Chart(df).mark_point().encode(x="c", y="v", stroke="h"),
        "mark_point",
        "circle",
    ),
    # A shape binding draws the first level as a circle and the second as another symbol, so no
    # one element count holds; the equality with the kept rows' chart pins every mark.
    "point-shape": (
        lambda df: fm.Chart(df).mark_point().encode(x="c", y="v", shape="h"),
        "mark_point",
        None,
    ),
    "point-stroke_dash": (
        lambda df: fm.Chart(df).mark_point().encode(x="c", y="v", stroke_dash="h"),
        "mark_point",
        "circle",
    ),
    "line-color": (
        lambda df: fm.Chart(df).mark_line().encode(x="c", y="v", color="h"),
        "mark_line",
        None,
    ),
    "line-detail": (
        lambda df: fm.Chart(df).mark_line().encode(x="c", y="v", detail="h"),
        "mark_line",
        None,
    ),
    "line-stroke_dash": (
        lambda df: fm.Chart(df).mark_line().encode(x="c", y="v", stroke_dash="h"),
        "mark_line",
        None,
    ),
    "bar-color": (
        lambda df: fm.Chart(df).mark_bar().encode(x="c", y="v", color="h"),
        "mark_bar",
        "rect",
    ),
    "tick-color": (
        lambda df: fm.Chart(df).mark_tick().encode(x="c", y="v", color="h"),
        "mark_tick",
        "line",
    ),
    "swarm-color": (
        lambda df: fm.Chart(df).mark_swarm().encode(x="c", y="v", color="h"),
        "mark_swarm",
        "circle",
    ),
    "strip": (lambda df: fm.catplot(df, x="c", y="v", hue="h", kind="strip"), "catplot", "circle"),
    "strip-dodge": (
        lambda df: fm.catplot(df, x="c", y="v", hue="h", kind="strip", dodge=True),
        "catplot",
        "circle",
    ),
    "swarm": (lambda df: fm.catplot(df, x="c", y="v", hue="h", kind="swarm"), "catplot", "circle"),
    "swarm-dodge": (
        lambda df: fm.catplot(df, x="c", y="v", hue="h", kind="swarm", dodge=True),
        "catplot",
        "circle",
    ),
}


@pytest.mark.parametrize("door", sorted(_HUE_DOORS))
def test_a_null_hue_is_dropped_and_counted_and_no_row_wears_another_levels_colour(door: str):
    """Four of twelve rows have a null hue: they are dropped and counted under the mark's entry,
    the legend keeps ``x`` and ``y`` and the chart is the one drawn from the other rows (no
    dodge sub-slot for the null group, no row in the first level's colour)."""
    build, speaker, element = _HUE_DOORS[door]
    df = _frame(h=_SOME)
    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message(speaker, 4, "h")])
    assert door in _NO_LEGEND or {"x", "y"} <= set(text_runs(svg))
    assert _bands(svg) == {"a", "b"}
    if element is not None:
        assert _drawn(svg, element) == 8


@pytest.mark.parametrize("door", ["point-color", "line-color", "strip", "swarm-dodge"])
def test_a_null_hue_gives_the_interactive_route_the_static_list(door: str):
    df = _frame(h=_SOME)
    _assert_route_matches_static(_HUE_DOORS[door][0], df, df.head(8))


@pytest.mark.parametrize(
    "door", ["point-color", "line-color", "bar-color", "tick-color", "strip", "swarm"]
)
def test_a_plain_mark_whose_every_hue_is_null_draws_its_empty_frame(door: str):
    build, speaker, _ = _HUE_DOORS[door]
    df = _frame(h=_ALL)
    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == [_all_keys_null_message(speaker, "h"), _NO_ROWS_TO_DRAW]
    sibling = fm.Chart(df).mark_boxplot().encode(x="c", y="v", color="h")
    _assert_empty_frame(svg, _render_capturing(sibling.to_svg)[0])


@pytest.mark.parametrize("door", ["point-color", "strip"])
def test_a_faceted_mark_whose_every_hue_is_null_names_it_once(door: str):
    build, speaker, _ = _HUE_DOORS[door]
    svg, shown = _render_capturing(build(_frame(h=_ALL)).facet("f").to_svg)
    assert shown == [_all_keys_null_message(speaker, "h")]
    assert {"F1", "F2"} <= set(text_runs(svg))
    _assert_plot_area(svg, 2)


@pytest.mark.parametrize("dtype", [pl.Null, pl.Int64, pl.Float64], ids=["null", "int64", "float64"])
def test_a_nominal_hue_null_on_every_row_is_a_key_of_any_dtype(dtype: Any):
    df = _frame().with_columns(pl.lit(None, dtype=dtype).alias("h"))
    chart = fm.Chart(df).mark_point().encode(x="c", y="v", color=fm.Color("h", type_="N"))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message("mark_point", "h"), _NO_ROWS_TO_DRAW]
    _assert_plot_area(svg)


def test_a_binned_colour_drops_the_rows_whose_value_is_null():
    """A ``bin`` makes the field a discretizing scale, so a null value has no bin."""
    df = _frame().with_columns(
        pl.when(pl.int_range(12) >= 8).then(None).otherwise(pl.col("w")).alias("w")
    )

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_point().encode(x="c", y="v", color=fm.Color("w", bin=True))

    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message("mark_point", 4, "w")])
    assert _drawn(svg, "circle") == 8


def test_a_row_null_on_the_category_and_on_the_hue_counts_under_each_field():
    """Rows 9 and 10 are null on both: counted under ``c`` and under ``h``, and dropped once."""
    df = _frame(
        c=["a", "b"] * 4 + [None] * 4,
        h=["x", "y"] * 3 + [None] * 4 + ["x", "y"],
    )

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_point().encode(x="c", y="v", color="h")

    svg = _assert_drops(
        build,
        df,
        df.head(6),
        [
            _null_key_dropped_message("mark_point", 4, "c"),
            _null_key_dropped_message("mark_point", 4, "h"),
        ],
    )
    assert _drawn(svg, "circle") == 6


def test_every_row_null_on_the_category_or_the_hue_names_both_columns():
    df = _frame(c=[None] * 6 + ["a", "b"] * 3, h=["x", "y"] * 3 + [None] * 6)
    chart = fm.Chart(df).mark_point().encode(x="c", y="v", color="h")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message("mark_point", "c, h"), _NO_ROWS_TO_DRAW]
    _assert_plot_area(svg, titles=("c",))


# ---------------------------------------------------------------------------
# Pies and maps
# ---------------------------------------------------------------------------


def _pie(levels: list[str | None]) -> Any:
    df = pl.DataFrame(
        {"n": [3.0, 2.0, 1.0][: len(levels)], "h": levels}, schema={"n": pl.Float64, "h": pl.Utf8}
    )
    return fm.Chart(df).mark_arc().encode(theta="n", color="h").coord(fm.CoordPolar())


def test_a_pie_drops_its_null_slice_and_the_others_fill_the_circle():
    svg, shown = _render_capturing(_pie(["x", "y", None]).to_svg)
    assert shown == [_null_key_dropped_message("mark_arc", 1, "h")]
    kept_svg, kept_shown = _render_capturing(
        fm.Chart(pl.DataFrame({"n": [3.0, 2.0], "h": ["x", "y"]}))
        .mark_arc()
        .encode(theta="n", color="h")
        .coord(fm.CoordPolar())
        .to_svg
    )
    assert kept_shown == []
    assert svg == kept_svg
    assert {"x", "y"} <= set(text_runs(svg))


def test_a_pie_drops_its_null_slice_on_the_interactive_route_too():
    shown, scene = _route(_pie(["x", "y", None]))
    kept = fm.Chart(pl.DataFrame({"n": [3.0, 2.0], "h": ["x", "y"]}))
    kept_chart = kept.mark_arc().encode(theta="n", color="h").coord(fm.CoordPolar())
    kept_shown, kept_scene = _route(kept_chart)
    assert shown == [_null_key_dropped_message("mark_arc", 1, "h")]
    assert kept_shown == []
    assert _scene_marks(scene) > 0
    assert scene == kept_scene


def test_an_all_null_pie_gives_the_interactive_route_the_static_list():
    chart = _pie([None, None, None])
    static = _render_capturing(chart.to_svg)[1]
    shown, scene = _route(chart)
    assert static == [_all_keys_null_message("mark_arc", "h"), _NO_ROWS_TO_DRAW]
    assert shown == static
    assert _scene_marks(scene) == 0


def test_a_pie_whose_every_colour_is_null_draws_no_slice():
    svg, shown = _render_capturing(_pie([None, None, None]).to_svg)
    assert shown == [_all_keys_null_message("mark_arc", "h"), _NO_ROWS_TO_DRAW]
    zero_rows = fm.Chart(pl.DataFrame({"n": [], "h": []}, schema={"n": pl.Float64, "h": pl.Utf8}))
    empty = zero_rows.mark_arc().encode(theta="n", color="h").coord(fm.CoordPolar())
    assert svg == _render_capturing(empty.to_svg)[0]


def _geo_chart(**encoding: str) -> Any:
    def square(x0: int) -> dict[str, Any]:
        corners = [[x0, 0], [x0 + 1, 0], [x0 + 1, 1], [x0, 1], [x0, 0]]
        return {"type": "Polygon", "coordinates": [corners]}

    features = [
        {"type": "Feature", "geometry": square(i), "properties": {"h": h}}
        for i, h in enumerate(["x", "y", None, "x"])
    ]
    collection = {"type": "FeatureCollection", "features": features}
    return fm.Chart(collection).mark_geoshape().encode(**encoding).coord(fm.CoordGeo())


def test_a_geoshape_draws_no_colour_so_a_null_colour_value_is_not_a_key():
    """Control: every region is drawn in the constant mark fill, the existing ignored-binding
    notice is the only warning, and no null count appears."""
    svg, shown = _render_capturing(_geo_chart(color="h").to_svg)
    assert shown == [_not_drawn("mark_geoshape", "color", "h", "geoshape")]
    assert re.findall(r'<path[^>]*fill="([^"]*)"', mark_layer(svg)) == ["#2563eb"] * 4
    assert svg == _render_capturing(_geo_chart().to_svg)[0]


# ---------------------------------------------------------------------------
# A binding that is not a key stays as it was
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mark,channel,notice",
    [
        ("mark_tick", "stroke_dash", _not_drawn("mark_tick", "stroke_dash", "h", "tick")),
        ("mark_line", "shape", _not_drawn("mark_line", "shape", "h", "line")),
        ("mark_bar", "shape", _not_drawn("mark_bar", "shape", "h", "bar")),
        ("mark_point", "detail", _detail_ignored("point")),
    ],
)
def test_a_binding_on_a_channel_the_mark_does_not_draw_is_never_a_null_key(
    mark: str, channel: str, notice: str
):
    """The channel is ignored, so a null in it drops and counts nothing: the chart is the one
    drawn with the nulls filled."""
    df = _frame(h=_SOME)

    def build(frame: pl.DataFrame) -> Any:
        return getattr(fm.Chart(frame), mark)().encode(x="c", y="v", **{channel: "h"})

    # The detail notice is a warn-once of the Python layer: clear its registry so the capture
    # sees it whichever test ran first, and build inside the capture in case it is raised there.
    reset_warnings()
    svg, shown = _render_capturing(lambda: build(df).to_svg())
    assert shown == [notice]
    filled = df.with_columns(pl.col("h").fill_null("x"))
    assert svg == _render_capturing(build(filled).to_svg)[0]


#: A null on one of these is not a key: ``(mark, channel, encoding of the other channels)``.
_VALUE_CHANNEL_DOORS = [
    *(
        ("mark_point", channel, {"x": "c", "y": "v"})
        for channel in ("color", "fill", "stroke", "size", "opacity", "fill_opacity")
    ),
    ("mark_point", "stroke_opacity", {"x": "c", "y": "v"}),
    ("mark_point", "angle", {"x": "c", "y": "v"}),
    ("mark_text", "angle", {"x": "c", "y": "v", "text": "v"}),
    ("mark_rule", "stroke_width", {"x": "c", "y": "v"}),
    ("mark_line", "stroke_width", {"x": "c", "y": "v"}),
]


@pytest.mark.parametrize(
    "method,channel,positions",
    _VALUE_CHANNEL_DOORS,
    ids=[f"{method}-{channel}" for method, channel, _ in _VALUE_CHANNEL_DOORS],
)
def test_a_null_on_a_quantitative_or_value_channel_is_not_a_key(
    method: str, channel: str, positions: dict[str, str]
):
    """Control: no row is dropped or counted, nothing is said, and the mark draws every row it
    draws when the channel holds a value on each."""
    df = _frame().with_columns(
        pl.when(pl.int_range(12) >= 8).then(None).otherwise(pl.col("w")).alias("w")
    )
    filled = df.with_columns(pl.col("w").fill_null(0.0))

    def build(frame: pl.DataFrame) -> Any:
        return getattr(fm.Chart(frame), method)().encode(**positions, **{channel: "w"})

    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == []
    filled_svg = _render_capturing(build(filled).to_svg)[0]
    for tag in ("circle", "line", "polyline"):
        assert _drawn(svg, tag) == _drawn(filled_svg, tag)
    assert len(text_runs(svg)) == len(text_runs(filled_svg))
    if method == "mark_point":
        assert _drawn(svg, "circle") == 12


# ---------------------------------------------------------------------------
# Swarm's own dodge
# ---------------------------------------------------------------------------


def test_a_swarm_dodged_by_a_hue_with_nulls_counts_the_drop_under_swarm():
    """``Swarm`` already dropped these rows without a word: it now says so, and the mark adds no
    second count for the same field."""
    df = _frame(h=_SOME)

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_swarm(dodge="h").encode(x="c", y="v", color="h")

    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message("Swarm", 4, "h")])
    assert _drawn(svg, "circle") == 8


# ---------------------------------------------------------------------------
# Statistics keep speaking for their own keys
# ---------------------------------------------------------------------------

_VIOLIN_ADVISORY = (
    "mark_violin: inner marks are drawn at each category's center and are not dodged per hue "
    'group; use Chart.mark_violin(inner="none") to drop the inner mark'
)

_STATISTIC_HUE_MARKS: dict[str, tuple[Callable[[pl.DataFrame], Any], list[str]]] = {
    "boxplot": (
        lambda df: fm.Chart(df).mark_boxplot().encode(x="c", y="v", color="h"),
        [_null_key_dropped_message("BoxStats", 4, "h")],
    ),
    "violin": (
        lambda df: fm.Chart(df).mark_violin().encode(x="c", y="v", color="h"),
        [_VIOLIN_ADVISORY, _null_key_dropped_message("Violin", 4, "h")],
    ),
    "boxen": (
        lambda df: fm.Chart(df).mark_boxen().encode(x="c", y="v", color="h"),
        [_null_key_dropped_message("LetterValue", 4, "h")],
    ),
    "errorbar": (
        lambda df: fm.Chart(df).mark_errorbar().encode(x="c", y="v", color="h"),
        [_null_key_dropped_message("ErrorExtent", 4, "h")],
    ),
    "bar-mean": (
        lambda df: fm.Chart(df).mark_bar().encode(x="c", y=fm.Y("v", aggregate="mean"), color="h"),
        [_null_key_dropped_message("Aggregate", 4, "h")],
    ),
    "bar-count": (
        lambda df: fm.Chart(df).mark_bar().encode(x="c", y="count()", color="h"),
        [_null_key_dropped_message("Aggregate", 4, "h")],
    ),
    "histogram": (
        lambda df: fm.Chart(df).mark_histogram().encode(x="v", color="h"),
        [_null_key_dropped_message("Bin", 4, "h")],
    ),
    "density": (
        lambda df: fm.Chart(df).mark_density().encode(x="v", color="h"),
        [_null_key_dropped_message("Kde", 4, "h")],
    ),
    "displot": (
        lambda df: fm.displot(df, x="v", hue="h"),
        [_null_key_dropped_message("Bin", 4, "h")],
    ),
}


@pytest.mark.parametrize("name", sorted(_STATISTIC_HUE_MARKS))
def test_a_statistic_mark_with_some_null_hues_says_only_what_its_statistic_says(name: str):
    build, expected = _STATISTIC_HUE_MARKS[name]
    svg, shown = _render_capturing(build(_frame(h=_SOME)).to_svg)
    assert shown == expected
    assert "null" not in text_runs(svg)


def test_a_smooth_whose_every_hue_is_null_keeps_its_one_pooled_fit():
    """Control: a fit statistic that pools its batch keeps the null group on purpose, so the
    mark drops nothing and the fit line is drawn."""
    chart = fm.Chart(_frame(h=_ALL)).mark_smooth().encode(x="w", y="v", color="h")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_fallback_message("Smooth", "h")]
    assert _drawn(svg, "path") > 0 or "<polyline" in mark_layer(svg)


def test_a_window_step_grouped_by_a_hue_with_nulls_leaves_the_mark_to_drop_them():
    """A step that pools a null key keeps the row, so the line mark that reads ``h`` as its
    colour drops and counts the rows."""
    window = fm.transform_window({"op": "row_number", "as": "rk"}, groupby=["h"])

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).transform(window).mark_line().encode(x="c", y="v", color="h")

    df = _frame(h=_SOME)
    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == [_null_key_dropped_message("mark_line", 4, "h")]
    assert {"x", "y"} <= set(text_runs(svg))


# ---------------------------------------------------------------------------
# Facets
# ---------------------------------------------------------------------------


def test_a_null_facet_value_is_dropped_by_the_facet_and_no_later_drop_counts_those_rows():
    """The same four rows are null on the facet and on the category: the facet drops them first,
    so the mark counts nothing more."""
    df = _frame(f=_SOME, c=_SOME)

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_point().encode(x="c", y="v").facet("f")

    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == [_facet_key_dropped_message(4, "f")]
    kept = build(df.head(8))
    assert svg == _render_capturing(kept.to_svg)[0]


def test_a_faceted_panel_whose_every_category_is_null_is_counted_over_the_frame_and_drawn_empty():
    df = _frame(f=["F1"] * 6 + ["F2"] * 6, c=["a", "b"] * 3 + [None] * 6)
    chart = fm.Chart(df).mark_point().encode(x="c", y="v").facet("f")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 6, "c")]
    assert _panel_marks(svg) == [True, False]
    assert {"F1", "F2"} <= set(text_runs(svg))
    assert mark_groups(svg)[0].count("<circle") == 6
    assert "null" not in text_runs(svg)


@pytest.mark.parametrize("door", ["col", "facet"])
def test_a_faceted_catplot_with_some_null_categories_counts_the_frame_once(door: str):
    df = _frame(c=_SOME)

    def build(frame: pl.DataFrame) -> Any:
        if door == "col":
            return fm.catplot(frame, x="c", y="v", kind="strip", col="f")
        return _catplot("strip")(frame).facet("f")

    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message("catplot", 4, "c")])
    assert {"F1", "F2"} <= set(text_runs(svg))


# ---------------------------------------------------------------------------
# A null that is not a key of a placed row, and the merge's padding
# ---------------------------------------------------------------------------


def test_a_row_null_on_the_category_and_on_the_value_is_not_dropped_by_the_mark():
    """Control: a row with no value places nothing whatever its category, so the mark does not
    count it and the census, silent inside a layer that draws, does not either."""
    df = _frame(c=_SOME).with_columns(
        pl.when(pl.int_range(12) >= 8).then(None).otherwise(pl.col("v")).alias("v")
    )
    svg, shown = _render_capturing(fm.Chart(df).mark_point().encode(x="c", y="v").to_svg)
    assert shown == []
    assert _bands(svg) == {"a", "b"}
    assert _drawn(svg, "circle") == 8


def test_a_merge_pads_a_shared_category_without_opening_a_band_or_a_warning():
    short = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    tall = pl.DataFrame({"cat": ["a", "b", "c", "d", "e"], "val": [4.0, 5.0, 6.0, 7.0, 8.0]})
    chart = fm.Chart(short).mark_point().encode(x=fm.X("cat", type_="N"), y="val") + fm.Chart(
        tall
    ).mark_point().encode(x=fm.X("cat", type_="N"), y="val")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert "null" not in text_runs(svg)
    assert _drawn(svg, "circle") == 8


def test_an_injected_reference_column_under_an_inherited_coordinate_opens_no_band():
    frame = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    chart = fm.Chart(frame).mark_bar().encode(x=fm.X("cat", type_="N"), y="val") + fm.Chart(
        pl.DataFrame({"ref": ["b", None, None]})
    ).mark_rule().encode(x=fm.X("ref", type_="N"))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert "null" not in text_runs(svg)
    assert _drawn(svg, "rect") == 3


def test_a_literal_null_string_is_its_own_band_and_a_true_null_never_joins_it():
    c = ["a", "b"] * 2 + ["null"] * 4 + [None] * 4
    df = _frame(c=c)

    def build(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_point().encode(x="c", y="v")

    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 4, "c")]
    assert text_runs(svg).count("null") == 1
    assert {"a", "b"} <= set(text_runs(svg))
    assert _drawn(svg, "circle") == 8
    assert svg == _render_capturing(build(df.head(8)).to_svg)[0]


def test_a_category_the_explicit_domain_omits_is_still_named_beside_the_null_drop():
    df = pl.DataFrame(
        {"x": ["p", "q", None, None], "y": [1.0, 2.0, 3.0, 4.0]},
    )
    chart = fm.Chart(df).mark_bar().encode(x=fm.X("x", scale=fm.BandScale(domain=["p"])), y="y")
    _, shown = _render_capturing(chart.to_svg)
    assert shown == [
        _null_key_dropped_message("mark_bar", 2, "x"),
        "x scale: 1 row(s) of 'x' name a category the scale's explicit domain leaves out, so "
        "they are not drawn; add it to the domain, or drop domain= to draw every category",
    ]


# ---------------------------------------------------------------------------
# Compositions: each layer or leaf judged on its own rows
# ---------------------------------------------------------------------------


def _points(df: pl.DataFrame) -> fm.Chart:
    return fm.Chart(df).mark_point().encode(x="c", y="v")


def _leaf_marks(svg: str) -> list[int]:
    return [group.count("<circle") for group in mark_groups(svg)]


_NULL_C = _null_key_dropped_message("mark_point", 4, "c")


def test_a_plus_layer_counts_its_own_rows_only():
    chart = _points(_frame(c=_SOME)) + _points(_frame())
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_NULL_C]
    assert _bands(svg) == {"a", "b"}
    assert _drawn(svg, "circle") == 20


def test_chart_layer_counts_each_layer_under_its_own_entry():
    chart = _points(_frame(c=_SOME)).layer(fm.Layer(mark="line", encoding={"x": "c", "y": "v"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_NULL_C, _null_key_dropped_message("mark_line", 4, "c")]
    assert _bands(svg) == {"a", "b"}
    assert _drawn(svg, "circle") == 8


@pytest.mark.parametrize("door", ["layer", "layer_chart", "layer_chart_independent_y"])
def test_an_overlay_counts_the_leaf_whose_rows_hold_a_null_category(door: str):
    dirty, clean = _points(_frame(c=_SOME)), _points(_frame())
    chart = {
        "layer": lambda: fm.layer(dirty, clean),
        "layer_chart": lambda: fm.LayerChart(dirty, clean),
        "layer_chart_independent_y": lambda: fm.LayerChart(
            dirty, clean, resolve={"y": "independent"}
        ),
    }[door]()
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_NULL_C]
    assert _bands(svg) == {"a", "b"}
    if door == "layer_chart_independent_y":
        assert _drawn(svg, "circle") == 20
    else:
        assert _leaf_marks(svg) == [8, 12]


@pytest.mark.parametrize("door", ["hconcat", "vconcat", "concat"])
def test_a_concat_leaf_counts_its_own_rows_and_a_repeat_shows_once(door: str):
    build = {
        "hconcat": fm.hconcat,
        "vconcat": fm.vconcat,
        "concat": lambda *charts: fm.concat(*charts, columns=2),
    }[door]
    dirty, clean = _points(_frame(c=_SOME)), _points(_frame())
    svg, shown = _render_capturing(build(dirty, clean).to_svg)
    assert shown == [_NULL_C]
    assert _leaf_marks(svg) == [8, 12]
    both, shown_both = _render_capturing(build(dirty, dirty).to_svg)
    assert shown_both == [_NULL_C, _NULL_C]
    assert _leaf_marks(both) == [8, 8]
    _, once = _render_capturing(build(dirty, dirty).to_svg, action="default")
    assert once == [_NULL_C]


@pytest.mark.parametrize("door", ["plus", "hconcat"])
def test_a_composition_gives_the_interactive_route_the_static_list(door: str):
    def build(df: pl.DataFrame) -> Any:
        return (
            _points(df) + _points(_frame())
            if door == "plus"
            else fm.hconcat(_points(df), _points(_frame()))
        )

    df = _frame(c=_SOME)
    shown, scene = _route(build(df))
    static = _render_capturing(build(df).to_svg)[1]
    assert shown == static == [_NULL_C]
    assert '"null"' not in scene


def test_a_pairplot_with_a_hue_counts_each_scatter_leaf_beside_the_diagonal_statistics():
    df = _frame(h=_SOME).select(["v", "w", "h"])
    svg, shown = _render_capturing(lambda: fm.pairplot(df, hue="h").to_svg())
    kde = _null_key_dropped_message("Kde", 4, "h")
    point = _null_key_dropped_message("pairplot", 4, "h")
    assert shown == [kde, point, point, kde]
    assert _leaf_marks(svg) == [0, 8, 8, 0]
    _, once = _render_capturing(lambda: fm.pairplot(df, hue="h").to_svg(), action="default")
    assert once == [kde, point]


def test_a_jointplot_with_a_hue_counts_its_scatter_beside_the_marginal_statistics():
    df = _frame(h=_SOME)
    svg, shown = _render_capturing(lambda: fm.jointplot(df, x="v", y="w", hue="h").to_svg())
    histogram = _null_key_dropped_message("Bin", 4, "h")
    point = _null_key_dropped_message("jointplot", 4, "h")
    assert shown == [histogram, point, histogram]
    assert _leaf_marks(svg) == [0, 8, 0]


# ---------------------------------------------------------------------------
# The boxen depth estimate counts the groups the statistic draws
# ---------------------------------------------------------------------------


def _depth(df: pl.DataFrame, **options: Any) -> int | None:
    return _boxen_materialized_depth(
        df, value_field="v", group_field="c", k_depth="tukey", k_proportion=0.007, **options
    )


def test_the_boxen_depth_estimate_ignores_a_group_of_null_categories():
    """A null category is not a group, so a large run of null-category rows does not deepen the
    ramp: the estimate is the one over the rows the statistic draws."""
    values = [float(i) for i in range(40)]
    df = pl.DataFrame({"c": ["a"] * 40 + [None] * 4000, "v": values + [1.0] * 4000})
    assert _depth(df) == _depth(df.head(40)) == 2


def test_the_boxen_depth_estimate_has_no_depth_when_every_category_is_null():
    df = pl.DataFrame({"c": [None] * 60, "v": [float(i) for i in range(60)]}).cast({"c": pl.Utf8})
    assert _depth(df) is None


def test_the_boxen_depth_estimate_ignores_rows_with_a_null_facet_value():
    df = pl.DataFrame(
        {
            "c": ["a"] * 4040,
            "f": ["F1"] * 40 + [None] * 4000,
            "v": [float(i) for i in range(4040)],
        }
    )
    assert _depth(df, extra_groupby_fields=("f",)) == _depth(
        df.head(40), extra_groupby_fields=("f",)
    )


def test_the_boxen_depth_estimate_counts_a_nan_category_as_a_group():
    """A NaN key is not null: it is a group of its own, sized as any other."""
    df = pl.DataFrame(
        {"c": [1.0] * 40 + [_NAN] * 4000, "v": [float(i) for i in range(4040)]},
    )
    with_nan, without_nan = _depth(df), _depth(df.head(40))
    assert with_nan is not None and without_nan is not None
    assert with_nan > without_nan


# ---------------------------------------------------------------------------
# A chart with no null key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_SPEAKER))
def test_a_catplot_with_no_null_key_warns_nothing(kind: str):
    svg, shown = _render_capturing(_catplot(kind)(_frame()).to_svg)
    assert shown == []
    assert _bands(svg) == {"a", "b"}


@pytest.mark.parametrize("name", sorted(_PLAIN))
def test_a_plain_mark_with_no_null_key_warns_nothing(name: str):
    svg, shown = _render_capturing(_PLAIN[name](_frame()).to_svg)
    assert shown == []
    assert _bands(svg) == {"a", "b"}


@pytest.mark.parametrize("door", sorted(_HUE_DOORS))
def test_a_grouped_mark_with_no_null_key_warns_nothing(door: str):
    svg, shown = _render_capturing(_HUE_DOORS[door][0](_frame()).to_svg)
    assert shown == []
    assert door in _NO_LEGEND or {"x", "y"} <= set(text_runs(svg))


# ---------------------------------------------------------------------------
# Two layers on one row set that key different fields
# ---------------------------------------------------------------------------


def _layered_frame(*, null_h: bool = True) -> pl.DataFrame:
    """Twelve rows in two panels (``f``): the point layer keys ``c`` and the line layer keys ``c``
    and ``h``. With ``null_h`` the last four rows have a null ``h`` and a ``v2`` of 1000 or more
    (two in each panel), so a y axis that still spans them reaches 1000. The kept ``v2`` of panel
    ``F1`` runs near 100 and of ``F2`` near 500, so the panels' own ranges differ."""
    v2 = [100.0, 101.0, 500.0, 501.0, 104.0, 105.0, 504.0, 505.0, 1000.0, 1001.0, 1002.0, 1003.0]
    h = ["x", "y"] * 4 + ([None] * 4 if null_h else ["x", "y"] * 2)
    return pl.DataFrame(
        {
            "c": ["a", "b"] * 6,
            "h": h,
            "f": ["F1", "F1", "F2", "F2"] * 3,
            "v": [float(i) for i in range(1, 13)],
            "v2": v2,
        },
        schema={"c": pl.Utf8, "h": pl.Utf8, "f": pl.Utf8, "v": pl.Float64, "v2": pl.Float64},
    )


def _neutral(df: pl.DataFrame) -> pl.DataFrame:
    """The frame whose line draws what the drop leaves: the null-``h`` rows keep a level and lose
    their ``v2``, so they place no line vertex and widen no axis."""
    null_h = pl.col("h").is_null()
    return df.with_columns(
        pl.when(null_h).then(None).otherwise(pl.col("v2")).alias("v2"),
        pl.col("h").fill_null("x"),
    )


def _layered_point(df: pl.DataFrame) -> fm.Chart:
    return fm.Chart(df).mark_point().encode(x="c", y="v")


def _layered_line(df: pl.DataFrame) -> fm.Chart:
    return fm.Chart(df).mark_line().encode(x="c", y="v2", color="h")


_LAYERED_DOORS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "chart-layer": lambda df: _layered_point(df).layer(
        fm.Layer(mark="line", encoding={"x": "c", "y": "v2", "color": fm.Color("h")})
    ),
    "plus": lambda df: _layered_point(df) + _layered_line(df),
    "layer-chart-independent-y": lambda df: fm.LayerChart(
        _layered_point(df), _layered_line(df), resolve={"y": "independent"}
    ),
    "hconcat-leaf": lambda df: fm.hconcat(
        _layered_point(df).layer(
            fm.Layer(mark="line", encoding={"x": "c", "y": "v2", "color": fm.Color("h")})
        ),
        _layered_point(_layered_frame(null_h=False)),
    ),
}

_LINE_DROP = _null_key_dropped_message("mark_line", 4, "h")
#: The neutral frame gives the null-``h`` rows level ``x`` and no ``v2``. Level ``x`` draws under
#: category ``a`` and category ``b`` under level ``y``, so the two rows of ``(b, x)`` are a cell
#: that draws nothing. Giving them a ``v2`` would add line vertices and widen the axis, which the
#: comparison with the drop exists to rule out, so the twin names the cell instead.
_NEUTRAL_CELL = _slot_message("layer 1: mark_line", "y", "b, h=x", "c")
_NEUTRAL_CELL_PANELS = _list_message(
    "layer 1: mark_line", "of c=b, h=x in facet panel f=F1; and of c=b, h=x in facet panel f=F2"
)


def _y_ranges(svg: str) -> list[tuple[float, float]]:
    """The lowest and highest y tick of each panel's axis, left to right."""
    return [(min(ladder), max(ladder)) for _, ladder in value_axis_ladders(svg, axis="y")]


def _numbers(svg: str) -> list[float]:
    """Every numeric text run (the value-axis ticks of a chart whose category axis is text)."""
    numbers = []
    for run in text_runs(svg):
        try:
            numbers.append(float(run))
        except ValueError:
            continue
    return numbers


@pytest.mark.parametrize("door", sorted(_LAYERED_DOORS))
def test_a_layer_keyed_on_another_field_drops_only_its_own_null_rows(door: str):
    """The point layer keys ``c`` alone and keeps all twelve rows without a word. The line layer
    keys ``h`` too: it drops its four null-``h`` rows in its own voice, and no scale spans them."""
    df = _layered_frame()
    svg, shown = _render_capturing(_LAYERED_DOORS[door](df).to_svg)
    assert shown == [_LINE_DROP]
    assert _drawn(svg, "circle") >= 12
    assert max(_numbers(svg)) < 1000
    neutral_svg, neutral_shown = _render_capturing(_LAYERED_DOORS[door](_neutral(df)).to_svg)
    assert neutral_shown == [_NEUTRAL_CELL]
    assert _numbers(svg) == _numbers(neutral_svg)
    assert _drawn(svg, "circle") == _drawn(neutral_svg, "circle")


@pytest.mark.parametrize("door", ["chart-layer", "plus", "layer-chart-independent-y"])
def test_a_layer_keyed_on_another_field_gives_the_interactive_route_the_static_list(door: str):
    df = _layered_frame()
    chart = _LAYERED_DOORS[door](df)
    static = _render_capturing(chart.to_svg)[1]
    shown, scene = _route(chart)
    assert static == [_LINE_DROP]
    assert shown == static
    neutral_shown, neutral_scene = _route(_LAYERED_DOORS[door](_neutral(df)))
    assert neutral_shown == [_NEUTRAL_CELL]
    assert _scene_marks(scene) > 0
    assert scene == neutral_scene
    assert '"null"' not in scene


_FACETED_LAYERED_DOORS = {
    "chart-layer": _LAYERED_DOORS["chart-layer"],
    "plus": _LAYERED_DOORS["plus"],
}


@pytest.mark.parametrize("share", ["shared", "independent"])
@pytest.mark.parametrize("door", sorted(_FACETED_LAYERED_DOORS))
def test_a_faceted_layer_keyed_on_another_field_reads_each_panels_own_kept_rows(
    door: str, share: str
):
    """Each panel's y range is the one its own kept rows give: the chart built so that the line
    holds only the kept rows draws the same ranges, and none reaches a dropped ``v2``. A line
    read across every panel's rows, or the dropped rows still read, fails it."""
    df = _layered_frame()

    def build(frame: pl.DataFrame) -> Any:
        chart = _FACETED_LAYERED_DOORS[door](frame).facet(col="f")
        return chart.share_scale(y="independent") if share == "independent" else chart

    svg, shown = _render_capturing(build(df).to_svg)
    assert shown == [_LINE_DROP]
    neutral_svg, neutral_shown = _render_capturing(build(_neutral(df)).to_svg)
    assert neutral_shown == [_NEUTRAL_CELL_PANELS]
    ranges = _y_ranges(svg)
    assert ranges == _y_ranges(neutral_svg)
    assert len(ranges) == 2
    assert all(high < 1000 for _, high in ranges)
    assert {"F1", "F2"} <= set(text_runs(svg))
    if share == "independent" and door == "chart-layer":
        (_, low_high), (_, high_high) = ranges
        assert low_high < 200 <= 500 <= high_high
    else:
        assert max(high for _, high in ranges) >= 500


def test_a_layer_keyed_on_the_same_fields_counts_in_place_under_its_own_entry():
    """Control: two layers that count the same rows each report their own count, and the rows leave
    every layer's reading."""
    df = _frame(c=_SOME)
    chart = _points(df).layer(fm.Layer(mark="line", encoding={"x": "c", "y": "v"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_NULL_C, _null_key_dropped_message("mark_line", 4, "c")]
    assert _bands(svg) == {"a", "b"}
    assert max(_numbers(svg)) < 100


#: SHA-256 of each layered chart with no null key, so a layered chart that routes nothing draws
#: byte for byte as it was.
_LAYERED_DIGESTS = {
    "chart-layer": "391c60a1449e6e835d2a184c3b145649701866a33166dcc623505bc8dd521c51",
    "plus": "391c60a1449e6e835d2a184c3b145649701866a33166dcc623505bc8dd521c51",
    "layer-chart-independent-y": "c0eba9403962ae76b31b1657eb3889ba11878ee014a3f4613a2883abd063e591",
    "hconcat-leaf": "bdcbcdf3014d0c8d82ad11232945a2b5ad14aa7863a31f541cc70959730ee21b",
}


@pytest.mark.parametrize("door", sorted(_LAYERED_DOORS))
def test_a_layered_chart_with_no_null_key_warns_nothing_and_draws_as_it_did(door: str):
    import hashlib

    svg, shown = _render_capturing(_LAYERED_DOORS[door](_layered_frame(null_h=False)).to_svg)
    assert shown == []
    assert hashlib.sha256(svg.encode()).hexdigest() == _LAYERED_DIGESTS[door]


# ---------------------------------------------------------------------------
# The encoding impute completes each series with every key it draws
# ---------------------------------------------------------------------------

#: Two colours by two details over three instants; series ``(y, p)`` and ``(y, q)`` miss ``t=2``.
_IMPUTE_FRAME = pl.DataFrame(
    {
        "t": [1.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 1.0, 3.0],
        "h": ["x", "x", "x", "y", "y", "x", "x", "x", "y", "y"],
        "d": ["p", "p", "p", "p", "p", "q", "q", "q", "q", "q"],
        "f": ["F1"] * 5 + ["F2"] * 5,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }
)
#: The rows the impute adds, written out at their key's place in their series: panel ``F1``'s
#: ``(2, y)`` follows its ``(1, y)`` and panel ``F2``'s follows its own. The frame with them is the
#: chart the impute must draw.
_IMPUTE_COMPLETED = pl.concat(
    [
        _IMPUTE_FRAME[:4],
        pl.DataFrame({"t": [2.0], "h": ["y"], "d": ["p"], "f": ["F1"], "v": [0.0]}),
        _IMPUTE_FRAME[4:9],
        pl.DataFrame({"t": [2.0], "h": ["y"], "d": ["q"], "f": ["F2"], "v": [0.0]}),
        _IMPUTE_FRAME[9:],
    ]
)
#: What a colour-alone impute draws over the whole frame: series ``y`` is one series across both
#: ``d`` levels and misses ``t=2`` once, so one row is written, after the last row with an earlier ``t``.
_IMPUTE_COLOUR_COMPLETED = pl.concat(
    [
        _IMPUTE_FRAME[:9],
        pl.DataFrame({"t": [2.0], "h": ["y"], "d": ["q"], "f": ["F2"], "v": [0.0]}),
        _IMPUTE_FRAME[9:],
    ]
)
_IMPUTED_Y = fm.Y("v", impute={"value": 0})


def _impute_chart(df: pl.DataFrame, method: str, imputed: bool, **channels: str) -> fm.Chart:
    y = _IMPUTED_Y if imputed else "v"
    return getattr(fm.Chart(df), method)().encode(x="t", y=y, color="h", **channels)


@pytest.mark.parametrize(
    "method,channels",
    [
        ("mark_line", {"detail": "d"}),
        ("mark_line", {"stroke_dash": "d"}),
        ("mark_area", {"detail": "d"}),
    ],
    ids=["line-detail", "line-stroke_dash", "area-detail"],
)
def test_an_imputed_series_keyed_by_a_second_channel_is_complete_and_warns_nothing(
    method: str, channels: dict[str, str]
):
    """The impute gives each series its missing point with every key the series draws, so no
    series is left with one point and no row the caller did not write is counted: the chart is the
    one drawn from the frame with the imputed rows written out."""
    svg, shown = _render_capturing(_impute_chart(_IMPUTE_FRAME, method, True, **channels).to_svg)
    assert shown == []
    completed, completed_shown = _render_capturing(
        _impute_chart(_IMPUTE_COMPLETED, method, False, **channels).to_svg
    )
    assert completed_shown == []
    assert svg == completed


def test_an_imputed_point_keyed_by_a_shape_draws_every_imputed_point_with_its_shape():
    """Twelve points for the ten rows and the two imputed ones, each in the shape of its own
    ``d`` (the legend lists ``p`` and ``q``), and no warning."""
    chart = _impute_chart(_IMPUTE_FRAME, "mark_point", True, shape="d")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert {"x", "y", "p", "q"} <= set(text_runs(svg))
    assert _scene_marks(chart.interactive().scene_json) == 12


@pytest.mark.parametrize("method", ["mark_line", "mark_area", "mark_point"])
def test_an_impute_with_a_colour_alone_draws_the_frame_with_its_row_written_out(method: str):
    """The one imputed row sits after its series' last earlier row, so the line draws ``1, 2, 3`` and
    the chart is the one the frame with that row written down draws, byte for byte."""
    svg, shown = _render_capturing(_impute_chart(_IMPUTE_FRAME, method, True).to_svg)
    assert shown == []
    written, written_shown = _render_capturing(
        _impute_chart(_IMPUTE_COLOUR_COMPLETED, method, False).to_svg
    )
    assert written_shown == []
    assert svg == written
    import hashlib

    assert hashlib.sha256(svg.encode()).hexdigest() == _IMPUTE_COLOUR_DIGESTS[method]


#: SHA-256 of the written-out chart (``_IMPUTE_COLOUR_COMPLETED``, no ``impute=``), measured on the
#: build before the encoding door inserted imputed rows, so it pins the bytes the impute must draw.
_IMPUTE_COLOUR_DIGESTS = {
    "mark_line": "b718672469c1b957b6f15baf7b3f5635d757ef336b03dd8ad7feda509dc3cc3c",
    "mark_area": "fc29a2a6bd9674d8a8fd70c7ceb4a654d953a9a685e933bcbaa80fe32c6c4dbd",
    "mark_point": "1f98ecf0c19ba63b30b7398a465f2798f2adea9bdbbf6c7102e1693e2f433ecf",
}

_FACET_DOORS: dict[str, Callable[[fm.Chart], fm.Chart]] = {
    "facet": lambda chart: chart.facet("f"),
    "facet-col": lambda chart: chart.facet(col="f"),
}


def _assert_faceted_impute_draws_its_points(
    imputed: Callable[[], fm.Chart],
    written_out: Callable[[], fm.Chart],
    *,
    interactive: bool = True,
) -> None:
    """``imputed()`` draws the chart ``written_out()`` draws (the frame with the imputed rows
    written down, no ``impute=``): each imputed row takes its panel, so it is drawn there. Neither
    warns, and the interactive route (when asked) builds the same scene, value for value."""
    svg, shown = _render_capturing(imputed().to_svg)
    completed, completed_shown = _render_capturing(written_out().to_svg)
    assert shown == [] and completed_shown == []
    assert svg == completed
    if interactive:
        route_shown, scene = _route(imputed())
        completed_route_shown, completed_scene = _route(written_out())
        assert route_shown == completed_route_shown == []
        assert _scene_marks(scene) > 0
        assert scene == completed_scene


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
@pytest.mark.parametrize("method", ["mark_line", "mark_area", "mark_point"])
def test_a_faceted_impute_with_a_colour_alone_draws_each_imputed_point_in_its_panel(
    method: str, door: str
):
    facet = _FACET_DOORS[door]
    _assert_faceted_impute_draws_its_points(
        lambda: facet(_impute_chart(_IMPUTE_FRAME, method, True)),
        lambda: facet(_impute_chart(_IMPUTE_COMPLETED, method, False)),
    )


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
@pytest.mark.parametrize(
    "method,channels",
    [
        ("mark_line", {"detail": "d"}),
        ("mark_line", {"stroke_dash": "d"}),
        ("mark_area", {"detail": "d"}),
        ("mark_point", {"shape": "d"}),
    ],
    ids=["line-detail", "line-stroke_dash", "area-detail", "point-shape"],
)
def test_a_faceted_impute_keyed_by_a_second_channel_draws_each_imputed_point_in_its_panel(
    method: str, channels: dict[str, str], door: str
):
    facet = _FACET_DOORS[door]
    _assert_faceted_impute_draws_its_points(
        lambda: facet(_impute_chart(_IMPUTE_FRAME, method, True, **channels)),
        lambda: facet(_impute_chart(_IMPUTE_COMPLETED, method, False, **channels)),
    )


#: Panel ``F1`` misses ``(t=2, y)``. Panel ``F2`` holds series ``x`` over ``t`` of 1 and 3 and
#: series ``y`` at 1 alone, so it misses ``(t=3, y)`` over its own ``t`` values; ``F3`` is
#: complete and holds no ``y`` level at all, so it gains no series for it.
_PANEL_IMPUTE_FRAME = pl.DataFrame(
    {
        "t": [1.0, 2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 2.0],
        "h": ["x", "x", "x", "y", "y", "x", "x", "y", "x", "x"],
        "f": ["F1"] * 5 + ["F2"] * 3 + ["F3"] * 2,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }
)
_PANEL_IMPUTE_COMPLETED = pl.concat(
    [
        _PANEL_IMPUTE_FRAME[:4],
        pl.DataFrame({"t": [2.0], "h": ["y"], "f": ["F1"], "v": [0.0]}),
        _PANEL_IMPUTE_FRAME[4:8],
        pl.DataFrame({"t": [3.0], "h": ["y"], "f": ["F2"], "v": [0.0]}),
        _PANEL_IMPUTE_FRAME[8:],
    ]
)


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
@pytest.mark.parametrize("method", ["mark_line", "mark_point"])
def test_each_panel_completes_the_series_it_holds_over_its_own_x_values(method: str, door: str):
    """``F1`` gains ``(2, y)``, ``F2`` gains ``(3, y)`` and not ``(2, y)`` (``2`` is not among its
    ``t``), and ``F3``, which holds no ``y`` level, gains no ``y`` series: the chart is the one the
    frame with exactly those two rows written out draws."""
    facet = _FACET_DOORS[door]
    _assert_faceted_impute_draws_its_points(
        lambda: facet(_impute_chart(_PANEL_IMPUTE_FRAME, method, True)),
        lambda: facet(_impute_chart(_PANEL_IMPUTE_COMPLETED, method, False)),
    )


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
@pytest.mark.parametrize("method", ["mark_line", "mark_point"])
def test_a_faceted_impute_over_a_callers_null_hue_counts_only_the_callers_rows(
    method: str, door: str
):
    """The caller's null-``h`` rows (one in each panel) are counted over the frame, once; the
    imputed rows, which the caller did not write, are never counted and are drawn in their panels."""
    facet = _FACET_DOORS[door]
    caller_nulls = pl.DataFrame(
        {"t": [4.0, 5.0], "h": [None, None], "f": ["F1", "F2"], "v": [20.0, 21.0]},
        schema={"t": pl.Float64, "h": pl.Utf8, "f": pl.Utf8, "v": pl.Float64},
    )
    colour_only = _IMPUTE_FRAME.drop("d")
    df = pl.concat([colour_only, caller_nulls])
    svg, shown = _render_capturing(facet(_impute_chart(df, method, True)).to_svg)
    assert shown == [_null_key_dropped_message(method, 2, "h")]
    completed = facet(_impute_chart(_IMPUTE_COMPLETED.drop("d"), method, False))
    completed_svg, completed_shown = _render_capturing(completed.to_svg)
    assert completed_shown == []
    assert svg == completed_svg


@pytest.mark.parametrize("method", ["mark_line", "mark_point"])
def test_an_impute_over_a_callers_null_hue_drops_and_counts_only_the_callers_rows(method: str):
    """The caller's two null-``h`` rows are counted; the ``t`` values they carry (4 and 5) complete
    no series, so the chart is the one the rows without them draw."""
    colour_only = _IMPUTE_FRAME.filter(pl.col("d") == "p").drop(["d", "f"])
    caller_nulls = pl.DataFrame(
        {"t": [4.0, 5.0], "h": [None, None], "v": [20.0, 21.0]},
        schema={"t": pl.Float64, "h": pl.Utf8, "v": pl.Float64},
    )
    df = pl.concat([colour_only, caller_nulls])
    svg, shown = _render_capturing(_impute_chart(df, method, True).to_svg)
    assert shown == [_null_key_dropped_message(method, 2, "h")]
    kept_svg, kept_shown = _render_capturing(_impute_chart(colour_only, method, True).to_svg)
    assert kept_shown == []
    assert svg == kept_svg


_TRANSFORM_IMPUTE_FRAME = pl.DataFrame(
    {
        "t": [1.0, 2.0, 3.0, 1.0, 3.0],
        "g": ["A", "A", "A", "B", "B"],
        "h": ["x", "x", "x", "y", "y"],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)


def _transform_impute_chart(groupby: list[str]) -> fm.Chart:
    step = fm.transform_impute("v", method="value", value=0.0, key="t", groupby=groupby)
    return (
        fm.Chart(_TRANSFORM_IMPUTE_FRAME)
        .transform(step)
        .mark_point()
        .encode(x="t", y="v", color="h")
    )


def test_a_callers_keyed_impute_that_leaves_the_drawn_field_out_has_its_row_counted():
    """The appended ``(t=2, g=B)`` row is null on ``h``, which the mark draws and the impute did
    not group by: it is the caller's transform's row, counted and dropped under the usual sentence."""
    svg, shown = _render_capturing(_transform_impute_chart(["g"]).to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 1, "h")]
    assert _drawn(svg, "circle") == 5
    assert {"x", "y"} <= set(text_runs(svg))


def test_a_callers_keyed_impute_grouped_by_the_drawn_field_draws_it_with_its_key():
    """Control: the appended row carries its ``h``, so it is drawn with its key and nothing is
    counted."""
    svg, shown = _render_capturing(_transform_impute_chart(["g", "h"]).to_svg)
    assert shown == []
    assert _drawn(svg, "circle") == 6


# ---------------------------------------------------------------------------
# A numeric field a mark draws discrete levels from is a key
# ---------------------------------------------------------------------------


def _numeric_key_frame() -> pl.DataFrame:
    """Twelve placed rows: an integer ``k`` of 0 and 1 on the first eight and null on the last four."""
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(12)],
            "c": ["a", "b"] * 6,
            "h": ["x", "x", "y", "y"] * 3,
            "v": [float(i) for i in range(1, 13)],
            "k": pl.Series([0, 1] * 4 + [None] * 4, dtype=pl.Int64),
        }
    )


def _numeric_shape(method: str) -> Callable[[pl.DataFrame], Any]:
    return lambda df: getattr(fm.Chart(df), method)().encode(x="t", y="v", shape="k")


@pytest.mark.parametrize("method", ["mark_point", "mark_swarm"])
def test_an_integer_shape_field_draws_levels_so_its_null_rows_are_dropped_and_counted(method: str):
    """``shape='k'`` on an integer column draws a legend of 0 and 1, so a null ``k`` belongs to no
    level: the four rows are dropped and counted, the legend is unchanged and the chart is the one
    the other eight rows draw."""
    build = _numeric_shape(method)
    df = _numeric_key_frame()
    svg = _assert_drops(build, df, df.head(8), [_null_key_dropped_message(method, 4, "k")])
    assert {"k", "0", "1"} <= set(text_runs(svg))
    _assert_route_matches_static(build, df, df.head(8))


@pytest.mark.parametrize("method", ["mark_point", "mark_swarm"])
def test_an_integer_shape_field_that_is_null_on_every_row_draws_its_empty_frame(method: str):
    df = _numeric_key_frame().with_columns(pl.lit(None, dtype=pl.Int64).alias("k"))
    svg, shown = _render_capturing(_numeric_shape(method)(df).to_svg)
    assert shown == [_all_keys_null_message(method, "k"), _NO_ROWS_TO_DRAW]
    assert _drawn(svg, "circle") == 0 and _drawn(svg, "rect") == 0


#: Each mark's encoding of the integer ``k`` as ``stroke_dash``, whether the mark dashes a row by
#: its index (``None``: read it off the render), and the element it draws once per row.
_DASH_DOORS: dict[str, tuple[dict[str, str], bool | None, str]] = {
    "mark_point": ({"x": "t", "y": "v"}, True, "circle"),
    "mark_bar": ({"x": "t", "y": "v"}, True, "rect"),
    "mark_rect": ({"x": "c", "y": "h"}, True, "rect"),
    "mark_rule": ({"x": "t", "y": "v"}, True, "line"),
    "mark_tick": ({"x": "t", "y": "v"}, False, "line"),
    "mark_line": ({"x": "t", "y": "v"}, False, "polyline"),
    "mark_swarm": ({"x": "t", "y": "v"}, None, "circle"),
}


@pytest.mark.parametrize("method", sorted(_DASH_DOORS))
def test_an_integer_stroke_dash_field_is_a_key_exactly_where_the_mark_dashes_by_it(method: str):
    """A mark that dashes each row by the integer index draws a level per value, so a null ``k``
    row belongs to none: dropped, counted, and the chart is the one the other eight rows draw (a
    dash drawn with no legend entry, as before). A mark that draws nothing from the field has no
    level to leave a row out of: nothing is counted and every row is drawn. Whether the mark
    dashes is read off the render of the eight rows that hold a value."""
    encoding, expected, element = _DASH_DOORS[method]

    def build(df: pl.DataFrame) -> Any:
        return getattr(fm.Chart(df), method)().encode(**encoding, stroke_dash="k")

    df = _numeric_key_frame()
    kept_svg = _render_capturing(build(df.head(8)).to_svg)[0]
    dashed = "stroke-dasharray" in mark_layer(kept_svg)
    if expected is not None:
        assert dashed is expected
    if dashed:
        _assert_drops(build, df, df.head(8), [_null_key_dropped_message(method, 4, "k")])
        _assert_route_matches_static(build, df, df.head(8))
    else:
        svg, shown = _render_capturing(build(df).to_svg)
        notice = (
            [_not_drawn("mark_tick", "stroke_dash", "k", "tick")] if method == "mark_tick" else []
        )
        assert shown == notice
        assert "k" not in text_runs(svg)
        rows = 1 if element == "polyline" else 12
        assert _drawn(svg, element) == rows


# ---------------------------------------------------------------------------
# What sizes or places a figure reads the levels the axis and the facet draw
# ---------------------------------------------------------------------------

_A, _B = OrdinalCategoryCoord("a"), OrdinalCategoryCoord("b")

_BAND_ANNOTATIONS: dict[str, Callable[[], Any]] = {
    "text": lambda: fm.annotation.text(_B, 2.0, "label"),
    "arrow": lambda: fm.annotation.arrow(_A, 1.0, _B, 3.0),
    "rect": lambda: fm.annotation.rect(_A, 1.0, _B, 3.0, fill="#dc2626"),
    "line": lambda: fm.annotation.line(_A, 1.0, _B, 3.0),
    "bracket": lambda: fm.annotation.bracket(_A, 3.0, _B, 3.0, label="range"),
    "callout": lambda: fm.annotation.callout(_B, 3.0, "note"),
}


def _annotated(kind: str, df: pl.DataFrame) -> fm.Chart:
    chart = fm.Chart(df).mark_point().encode(x="c", y="v")
    return chart + _BAND_ANNOTATIONS[kind]()


@pytest.mark.parametrize("kind", sorted(_BAND_ANNOTATIONS))
def test_an_annotation_on_a_band_lands_where_it_does_without_the_null_category(kind: str):
    """A null category has no band, so the bands are ``a`` and ``b`` and an annotation anchored on
    ``b`` sits at the centre of the second of two: the chart, annotation included, is the one the
    frame without the null row draws."""
    dirty = pl.DataFrame({"c": ["a", None, "b"], "v": [1.0, 2.0, 3.0]})
    clean = dirty.filter(pl.col("c").is_not_null())
    svg, shown = _render_capturing(_annotated(kind, dirty).to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 1, "c")]
    clean_svg, clean_shown = _render_capturing(_annotated(kind, clean).to_svg)
    assert clean_shown == []
    assert svg == clean_svg


@pytest.mark.parametrize("kind", ["text", "rect", "callout"])
def test_an_annotation_on_a_band_gives_the_interactive_route_the_static_scene(kind: str):
    dirty = pl.DataFrame({"c": ["a", None, "b"], "v": [1.0, 2.0, 3.0]})
    clean = dirty.filter(pl.col("c").is_not_null())
    static = _render_capturing(_annotated(kind, dirty).to_svg)[1]
    shown, scene = _route(_annotated(kind, dirty))
    clean_shown, clean_scene = _route(_annotated(kind, clean))
    assert shown == static == [_null_key_dropped_message("mark_point", 1, "c")]
    assert clean_shown == []
    assert scene == clean_scene


def _faceted_frame() -> pl.DataFrame:
    """Two panels of three rows and two rows whose facet value is null, their values inside the
    panels' range so that only the facet's sizing, not a shared extent, tells the two frames apart."""
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 5.0],
            "f": ["F1", "F1", "F1", "F2", "F2", "F2", None, None],
        },
        schema={"x": pl.Float64, "v": pl.Float64, "f": pl.Utf8},
    )


_FACET_SIZED_DOORS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "col-facet": lambda df: fm.Chart(df).mark_point().encode(x="x", y="v").facet(col="f"),
    "row-facet": lambda df: fm.Chart(df).mark_point().encode(x="x", y="v").facet(row="f"),
    "displot-col": lambda df: fm.displot(df, x="v", col="f"),
    "displot-row": lambda df: fm.displot(df, x="v", row="f"),
    "catplot-col": lambda df: fm.catplot(df, x="f", y="v", col="f", kind="strip"),
}


@pytest.mark.parametrize("door", sorted(_FACET_SIZED_DOORS))
def test_a_facet_with_a_null_value_is_sized_for_the_panels_it_draws(door: str):
    """A null facet value has no panel: the grid and the canvas are the ones the frame without
    those rows draws, beside the facet's own count of the dropped rows."""
    dirty = _faceted_frame()
    clean = dirty.filter(pl.col("f").is_not_null())
    svg, shown = _render_capturing(_FACET_SIZED_DOORS[door](dirty).to_svg)
    assert shown == [_facet_key_dropped_message(2, "f")]
    clean_svg, clean_shown = _render_capturing(_FACET_SIZED_DOORS[door](clean).to_svg)
    assert clean_shown == []
    assert svg == clean_svg


# ---------------------------------------------------------------------------
# A frame with no row, and a facet key that is null on every row
# ---------------------------------------------------------------------------

_ZERO_ROW_UNFACETED_DOORS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "mark_point": lambda df: fm.Chart(df).mark_point().encode(x="c", y="v"),
    "mark_point-color": lambda df: fm.Chart(df).mark_point().encode(x="c", y="v", color="h"),
    "catplot-strip": lambda df: fm.catplot(df, x="c", y="v", kind="strip"),
}


@pytest.mark.parametrize("door", sorted(_ZERO_ROW_UNFACETED_DOORS))
def test_a_zero_row_frame_on_a_category_mark_says_only_that_it_has_no_rows(door: str):
    """No key holds a null, so no key sentence is said: the frame, its axes and the one
    no-rows sentence."""
    svg, shown = _render_capturing(_ZERO_ROW_UNFACETED_DOORS[door](_frame().head(0)).to_svg)
    assert shown == [_NO_ROWS_TO_DRAW]
    _assert_plot_area(svg)


_ZERO_ROW_FACET_DOORS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "mark_point col": lambda df: fm.Chart(df).mark_point().encode(x="c", y="v").facet(col="f"),
    "mark_point row": lambda df: fm.Chart(df).mark_point().encode(x="c", y="v").facet(row="f"),
    "mark_point wrap": lambda df: fm.Chart(df).mark_point().encode(x="c", y="v").facet("f"),
    "mark_point grid": lambda df: (
        fm.Chart(df).mark_point().encode(x="c", y="v").facet(row="h", col="f")
    ),
    "mark_point-color col": lambda df: (
        fm.Chart(df).mark_point().encode(x="c", y="v", color="h").facet(col="f")
    ),
    "catplot col": lambda df: fm.catplot(df, x="c", y="v", col="f"),
    "catplot row": lambda df: fm.catplot(df, x="c", y="v", row="f"),
    "displot col": lambda df: fm.displot(df, x="v", col="f"),
    "displot row": lambda df: fm.displot(df, x="v", row="f"),
    "relplot col": lambda df: fm.relplot(df, x="v", y="w", col="f"),
}


@pytest.mark.parametrize("door", sorted(_ZERO_ROW_FACET_DOORS))
def test_a_zero_row_frame_faceted_on_any_door_draws_no_panel_and_says_so(door: str):
    """Every facet door renders the frame with no panel and the one sentence that says so, never a
    refusal of the grid: the panel count is the number of levels the column holds (none)."""
    svg, shown = _render_capturing(_ZERO_ROW_FACET_DOORS[door](_frame().head(0)).to_svg)
    assert shown == [_NO_FACET_ROWS_WARNING]
    assert mark_groups_by_frame(svg) == []


#: The fit figure's text for a zero-row frame: it names an empty panel value (``f=``) that no drawn
#: panel has.
_ZERO_ROW_LMPLOT_FIT = _panel_fit_message("Smooth", FIT_FEW_POINTS, panel="f=")


@pytest.mark.parametrize("door", ["col", "row"])
def test_a_zero_row_lmplot_facet_draws_what_the_other_facet_direction_draws(door: str):
    """A zero-row frame faceted by column or by row draws the same svg and gives the fit's degrade
    text, which names an empty panel value (``f=``) that no drawn panel has; the column and the row
    facet agree on both."""
    frame = _frame().head(0)
    other = "row" if door == "col" else "col"
    a_svg, a_shown = _render_capturing(
        lambda: fm.lmplot(frame, x="v", y="w", ci=None, **{door: "f"}).to_svg()
    )
    b_svg, b_shown = _render_capturing(
        lambda: fm.lmplot(frame, x="v", y="w", ci=None, **{other: "f"}).to_svg()
    )
    assert a_shown == b_shown == [_ZERO_ROW_LMPLOT_FIT]
    assert a_svg == b_svg


@pytest.mark.parametrize("c", ["none", _SOME, _ALL])
@pytest.mark.parametrize("door", ["facet", "catplot-col", "col-facet", "grid"])
def test_a_facet_key_null_on_every_row_speaks_alone_and_the_mark_counts_nothing(door: str, c: str):
    """The facet drops every row first, so the mark sees none: the facet's count and the no-panel
    sentence, never a null-category count or an every-key-null sentence for the mark."""
    df = _frame(f=_ALL, c=c)
    builders: dict[str, Callable[[], Any]] = {
        "facet": lambda: fm.Chart(df).mark_point().encode(x="c", y="v").facet("f"),
        "catplot-col": lambda: fm.catplot(df, x="c", y="v", col="f"),
        "col-facet": lambda: fm.Chart(df).mark_point().encode(x="c", y="v").facet(col="f"),
        "grid": lambda: fm.Chart(df).mark_point().encode(x="c", y="v").facet(row="h", col="f"),
    }
    svg, shown = _render_capturing(builders[door]().to_svg)
    assert shown == [_facet_key_dropped_message(12, "f"), _NO_FACET_ROWS_WARNING]
    assert mark_groups_by_frame(svg) == []


# ---------------------------------------------------------------------------
# Each layer reports its own drop
# ---------------------------------------------------------------------------


def _two_key_frame() -> pl.DataFrame:
    """``c`` is null on rows 9 to 12 and ``h`` on rows 5 to 8, so a layer keyed on both counts
    four rows under each field."""
    return _frame(c=["a", "b"] * 4 + [None] * 4, h=["x", "y"] * 2 + [None] * 4 + ["x", "y"] * 2)


_TICK_ENCODING = {"x": "c", "y": "v"}


def test_each_layer_reports_its_own_drop_even_when_two_lines_are_identical():
    """Two layers of one mark over one frame each drop the same rows and each says so, in layer
    order: nothing is folded across layers, as a statistic's layers and a composite's leaves are
    not."""
    df = _frame(c=_SOME)
    layered = _points(df).layer(fm.Layer(mark="point", encoding={"x": "c", "y": "v"}))
    assert _render_capturing(layered.to_svg)[1] == [_NULL_C, _NULL_C]
    plus = _render_capturing((_points(df) + _points(df)).to_svg)[1]
    assert plus == [_NULL_C, _NULL_C]
    layer_chart = _render_capturing(fm.LayerChart(_points(df), _points(df)).to_svg)[1]
    assert layer_chart == [_NULL_C, _NULL_C]


def test_two_layers_whose_lines_differ_by_mark_say_both_in_layer_order():
    df = _frame(c=_SOME)
    layered = _points(df).layer(fm.Layer(mark="tick", encoding=_TICK_ENCODING))
    assert _render_capturing(layered.to_svg)[1] == [
        _NULL_C,
        _null_key_dropped_message("mark_tick", 4, "c"),
    ]


def test_two_layers_whose_lines_differ_by_field_say_both_in_layer_order():
    """Layer 0 keys ``c``; layer 1 keys ``c`` and ``h``. The layers' lists differ, so neither is
    folded into the other: layer 0's ``c`` sentence, then layer 1's own ``c`` and ``h`` sentences,
    each counting the rows that layer drops."""
    df = _two_key_frame()
    layered = _points(df).layer(
        fm.Layer(mark="point", encoding={"x": "c", "y": "v", "color": fm.Color("h")})
    )
    assert _render_capturing(layered.to_svg)[1] == [
        _NULL_C,
        _NULL_C,
        _null_key_dropped_message("mark_point", 4, "h"),
    ]


# ---------------------------------------------------------------------------
# A sibling layer keyed by another channel beside an imputing layer
# ---------------------------------------------------------------------------


def test_a_layer_keyed_by_a_channel_the_imputed_series_lack_completes_its_own_series():
    """The host completes ``(2, y)`` for its colour series (eleven points). The shape layer inherits
    the impute and also keys ``d``, so it completes its own ``(h, d)`` series, ``(2, y, p)`` and
    ``(2, y, q)``, and draws twelve: no warning, twenty-three marks."""
    host = fm.Chart(_IMPUTE_FRAME.drop("f")).mark_point().encode(x="t", y=_IMPUTED_Y, color="h")
    chart = host.layer(fm.Layer(mark="point", encoding={"shape": "d"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert {"x", "y", "p", "q"} <= set(text_runs(svg))
    route_shown, scene = _route(chart)
    assert route_shown == []
    # Eleven host points (ten rows and one imputed) and twelve shape points (ten rows and two).
    assert _scene_marks(scene) == 23


# ---------------------------------------------------------------------------
# A matrix reads the rows it draws
# ---------------------------------------------------------------------------

_MATRIX_ROWS = pl.DataFrame(
    {
        "row": ["r1", "r2", "r3", None],
        "a": [1.0, 2.0, 3.0, 1000.0],
        "b": [4.0, 5.0, 6.0, 900.0],
    },
    schema={"row": pl.Utf8, "a": pl.Float64, "b": pl.Float64},
)
#: What the dropped row's two cells are counted as by the heatmap: one cell per value column.
_HEATMAP_DROP = _null_key_dropped_message("heatmap", 2, "row")


@pytest.mark.parametrize("annot", [False, True], ids=["plain", "annotated"])
@pytest.mark.parametrize(
    "options",
    [
        {"robust": True},
        {"vmin": 0.0},
        {"vmax": 10.0},
        {"center": 20.0},
    ],
    ids=["robust", "vmin", "vmax", "center"],
)
def test_a_heatmaps_colour_domain_is_read_from_the_rows_it_draws(
    options: dict[str, Any], annot: bool
):
    """The null-labelled row holds the extreme values (1000 and 900) but belongs to no row: the
    heatmap is the one the frame without it draws, with the same domain warnings, and says only
    that it dropped the row's two cells. An annotated heatmap says it twice: the call builds two
    layers (the cells and their labels) and each reports its own drop once."""
    clean_rows = _MATRIX_ROWS.filter(pl.col("row").is_not_null())
    svg, shown = _render_capturing(
        lambda: fm.heatmap(_MATRIX_ROWS, annot=annot, **options).to_svg()
    )
    clean_svg, clean_shown = _render_capturing(
        lambda: fm.heatmap(clean_rows, annot=annot, **options).to_svg()
    )
    drops = [_HEATMAP_DROP] * (2 if annot else 1)
    assert shown == [*clean_shown, *drops]
    assert svg == clean_svg


_CLUSTER_ROWS = pl.DataFrame(
    {
        "row": ["r1", None, "r3", "r4"],
        "a": [1.0, 2.0, 9.0, 3.0],
        "b": [4.0, 5.0, 1.0, 6.0],
        "c": [7.0, 8.0, 2.0, 9.0],
    },
    schema={"row": pl.Utf8, "a": pl.Float64, "b": pl.Float64, "c": pl.Float64},
)
#: What each of the clustermap's three clustering leaves says at render about the one null-labelled
#: row: the center heatmap's two linkages share one lowering (so one line), and the row and column
#: dendrograms each run their own.
_CLUSTER_DROP = _null_key_dropped_message("Linkage", 1, "row")


def _build_capturing(build: Callable[[], Any]) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
    return chart, [str(w.message) for w in caught]


def test_a_clustermap_clusters_and_sizes_over_the_rows_it_draws():
    """A null row label belongs to no row: the linkages skip it, so the dendrogram has a leaf for
    each drawn row in the order the frame without it gives, and the engine reports the skipped
    row at render, nothing at construction."""
    clean_rows = _CLUSTER_ROWS.filter(pl.col("row").is_not_null())
    chart, built = _build_capturing(lambda: fm.clustermap(_CLUSTER_ROWS))
    assert built == []
    svg, shown = _render_capturing(chart.to_svg)
    clean_svg, clean_shown = _render_capturing(fm.clustermap(clean_rows).to_svg)
    assert clean_shown == []
    assert shown == [*clean_shown, *[_CLUSTER_DROP] * 3]
    assert svg == clean_svg
    assert [run for run in text_runs(svg) if run.startswith("r")] == [
        run for run in text_runs(clean_svg) if run.startswith("r")
    ]


def test_a_clustermap_gives_the_interactive_route_the_drawn_rows():
    clean_rows = _CLUSTER_ROWS.filter(pl.col("row").is_not_null())
    chart, built = _build_capturing(lambda: fm.clustermap(_CLUSTER_ROWS))
    assert built == []
    static = _render_capturing(chart.to_svg)[1]
    shown, scene = _route(chart)
    clean_shown, clean_scene = _route(fm.clustermap(clean_rows))
    assert clean_shown == []
    assert shown == static == [_CLUSTER_DROP] * 3
    assert scene == clean_scene


def test_a_clustermap_with_no_row_refuses_as_every_empty_composition_does():
    with pytest.raises(ValueError, match="every panel's data is empty"):
        fm.clustermap(_CLUSTER_ROWS.head(0)).to_svg()


def test_a_swarm_dodged_by_a_hue_that_is_null_on_every_row_gives_the_every_row_sentence():
    """The shared every-row sentence names the swarm's dodge field, then the chart says it has no
    rows: an empty frame, as every other statistic gives."""
    chart = fm.Chart(_frame(h=_ALL)).mark_swarm(dodge="h").encode(x="c", y="v", color="h")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_all_keys_null_message("Swarm", "h"), _NO_ROWS_TO_DRAW]
    _assert_plot_area(svg, titles=("c", "v"))


def test_a_clustermap_whose_every_row_label_is_null_says_so_per_step_and_draws_its_empty_frame():
    """Every label is null, so no row is clustered or drawn: each of the three leaves (center,
    row dendrogram, column dendrogram) says the every-row sentence (the center's two steps share
    one, as with a single null label), then says it has no rows. The frame is drawn, with no cell
    and no dendrogram segment, and the column ``Reorder`` over its empty order raises nothing."""
    rows = _CLUSTER_ROWS.with_columns(pl.lit(None, dtype=pl.Utf8).alias("row"))
    chart, built = _build_capturing(lambda: fm.clustermap(rows))
    assert built == []
    svg, shown = _render_capturing(chart.to_svg)
    expected = [_all_keys_null_message("Linkage", "row"), _NO_ROWS_TO_DRAW] * 3
    assert shown == expected
    assert mark_layer(svg) == ""
    route_shown, scene = _route(chart)
    assert route_shown == expected
    assert _scene_marks(scene) == 0


# ---------------------------------------------------------------------------
# A heatmap whose row labels are all null, or whose finite values sit on null-labelled rows
# ---------------------------------------------------------------------------


def _fill_failure(given: str, value: float, reason: str) -> str:
    """The warning for a one-sided colour bound whose other end cannot be filled."""
    missing = "vmax" if given == "vmin" else "vmin"
    return (
        f"heatmap: {given}={value} was given but {missing} could not be filled from the data "
        f"extent ({reason}); the color-domain override is dropped."
    )


_NO_FINITE_VALUES = "the heatmap's numeric columns have no finite values"
_ON_NULL_LABELS = "every finite value sits on a row whose 'id' label is null"
_ID_ROWS = pl.DataFrame(
    {"id": pl.Series([None] * 3, dtype=pl.Utf8), "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
)
_ID_ZERO_ROWS = _ID_ROWS.head(0)


@pytest.mark.parametrize("given,value", [("vmin", 0.0), ("vmax", 10.0)])
@pytest.mark.parametrize("case", ["every-label-null", "mixed", "zero-row"])
def test_a_one_sided_domain_whose_every_finite_value_is_on_a_null_labelled_row_names_the_null_labels(
    case: str, given: str, value: float
):
    """The drawn rows give the fill nothing, and the warning says why: a null-labelled row holds
    every finite value (the rows that are drawn have none), or the table has no finite value at
    all, as for a zero-row frame."""
    kwargs = {given: value}
    zero_svg = _render_capturing(lambda: fm.heatmap(_ID_ZERO_ROWS, annot=False, **kwargs).to_svg())[
        0
    ]
    if case == "every-label-null":
        frame = _ID_ROWS
        expected = [
            _fill_failure(given, value, _ON_NULL_LABELS),
            _all_keys_null_message("heatmap", "id"),
            _NO_ROWS_TO_DRAW,
        ]
    elif case == "mixed":
        frame = pl.DataFrame(
            {"id": ["r1", "r2", None], "a": [_NAN, _NAN, 5.0]},
            schema={"id": pl.Utf8, "a": pl.Float64},
        )
        expected = [
            _fill_failure(given, value, _ON_NULL_LABELS),
            _null_key_dropped_message("heatmap", 1, "id"),
        ]
    else:
        frame = _ID_ZERO_ROWS
        expected = [_fill_failure(given, value, _NO_FINITE_VALUES), _NO_ROWS_TO_DRAW]
    svg, shown = _render_capturing(lambda: fm.heatmap(frame, annot=False, **kwargs).to_svg())
    assert shown == expected
    if case != "mixed":
        assert svg == zero_svg


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"square": True},
        {"robust": True},
        {"center": 100.0},
        {"vmin": 0.0, "vmax": 10.0},
        {"annot": True},
    ],
    ids=["plain", "square", "robust", "center", "two-sided", "annot"],
)
def test_a_heatmap_whose_every_row_label_is_null_draws_what_a_zero_row_heatmap_draws(
    kwargs: dict[str, Any],
):
    """Every label is null, so no row is drawn: the frame the zero-row heatmap draws with the same
    options (a square grid keeps the square ratio of an empty grid, not of the three rows the table
    holds), the every-row sentence once per layer the call builds, then the no-rows sentence."""
    options = {"annot": False, **kwargs}
    svg, shown = _render_capturing(lambda: fm.heatmap(_ID_ROWS, **options).to_svg())
    zero_svg, zero_shown = _render_capturing(lambda: fm.heatmap(_ID_ZERO_ROWS, **options).to_svg())
    assert zero_shown == [_NO_ROWS_TO_DRAW]
    layers = 2 if options["annot"] else 1
    assert shown == [*[_all_keys_null_message("heatmap", "id")] * layers, _NO_ROWS_TO_DRAW]
    assert svg == zero_svg
    if not kwargs.get("annot") and not set(kwargs) - {"square"}:
        scene = _route(fm.heatmap(_ID_ROWS, **options))[1]
        zero_scene = _route(fm.heatmap(_ID_ZERO_ROWS, **options))[1]
        assert scene == zero_scene


# ---------------------------------------------------------------------------
# An order read by Reorder, and the label a Linkage reads
# ---------------------------------------------------------------------------

_ORDER_FRAME = pl.DataFrame(
    {
        "id": ["r1", "r2", "r3", "r4"],
        "a": [1.0, 2.0, 3.0, 9.0],
        "b": [4.0, 1.0, 0.5, 2.0],
        "c": [3.0, 1.0, 2.0, 1.0],
    }
)


@pytest.mark.parametrize("labelled", [False, True], ids=["unlabelled", "labelled"])
def test_a_reorder_refuses_an_order_read_over_rows_another_step_resized(labelled: bool):
    """The linkage read four rows and the unpivot made twelve: a reorder over them cannot place the
    order, whatever the linkage skipped, and is refused rather than drawing a subset."""
    frame = (
        _ORDER_FRAME.with_columns(pl.Series("id", ["r1", None, "r3", "r4"]))
        if labelled
        else _ORDER_FRAME
    )
    linkage = fm.Linkage(name="o", labels="id") if labelled else fm.Linkage(name="o")
    chart = (
        fm.Chart(frame)
        .transform(
            linkage,
            fm.Unpivot(id_vars=["id"]),
            fm.Reorder("original_idx", from_="o_order", drop_index=False),
        )
        .mark_point()
        .encode(x="variable", y="value")
    )
    refusal = (
        "stat_reorder: from='o_order' orders the 4 rows its linkage read but batch has 12 rows; "
        "place the Reorder before any step that adds or removes rows"
        if labelled
        else "stat_reorder: from='o_order' produced 4 indices but batch has 12 rows"
    )
    with pytest.raises(ValueError, match=re.escape(refusal)):
        _render_capturing(chart.to_svg)


def _clustermap_chain(df: pl.DataFrame) -> fm.Chart:
    """The clustermap's own steps over ``df``, written out: both linkages with ``labels='id'``, both
    reorders, then the unpivot the cells read."""
    steps = [
        fm.Linkage(axis="rows", name="rl", labels="id"),
        fm.Linkage(axis="columns", name="cl", labels="id"),
        fm.Reorder("original_idx", from_="rl_order", drop_index=False),
        fm.Reorder("original_idx", from_="cl_order", axis="columns", drop_index=False),
        fm.Unpivot(id_vars=["id"], value_vars=None, var_name="variable", value_name="value"),
    ]
    return (
        fm.Chart(df)
        .transform(*steps)
        .mark_rect()
        .encode(x="variable", y=fm.Y("id", type_="N"), color="value")
    )


_FEATURES = {
    "A": [1.0, 2.0, 3.0, 4.5],
    "B": [10.0, 0.0, 9.0, 1.0],
    "C": [1.1, 2.2, 2.9, 4.4],
    "D": [9.0, 1.0, 10.0, 0.5],
}


def test_a_linkage_label_column_is_never_a_feature_and_its_reorder_agrees():
    """A numeric label column is neither clustered nor counted among the columns the column
    reorder permutes: the chain renders and orders the features as it does with a text label."""
    numeric = pl.DataFrame({"id": pl.Series([1, 2, 3, 4], dtype=pl.Int64), **_FEATURES})
    text = pl.DataFrame({"id": ["r1", "r2", "r3", "r4"], **_FEATURES})
    svg, shown = _render_capturing(_clustermap_chain(numeric).to_svg)
    text_svg, text_shown = _render_capturing(_clustermap_chain(text).to_svg)
    assert shown == text_shown == []
    order = [run for run in text_runs(svg) if run in "ABCD"]
    assert order == [run for run in text_runs(text_svg) if run in "ABCD"]
    assert order != ["A", "B", "C", "D"]


def test_a_linkage_labels_column_the_frame_lacks_is_refused_naming_labels():
    chart = (
        fm.Chart(_ORDER_FRAME)
        .transform(
            fm.Linkage(name="o", labels="nope"),
            fm.Reorder("original_idx", from_="o_order", drop_index=False),
        )
        .mark_point()
        .encode(x="a", y="b")
    )
    with pytest.raises(ValueError, match=re.escape("linkage: labels column 'nope' not found")):
        _render_capturing(chart.to_svg)


def test_a_linkage_label_column_of_a_date_dtype_is_read_for_its_nulls():
    """Only the label's nulls are read, so a date or timestamp label works as a text one does: the
    null row is skipped and counted, and the chart is the one the text label draws."""
    stamps = [dt.datetime(2020, 1, 1), None, dt.datetime(2020, 1, 3), dt.datetime(2020, 1, 4)]

    def chart(ids: pl.Series) -> fm.Chart:
        frame = _ORDER_FRAME.with_columns(ids.alias("id"))
        return (
            fm.Chart(frame)
            .transform(
                fm.Linkage(name="o", labels="id"),
                fm.Reorder("original_idx", from_="o_order", drop_index=False),
            )
            .mark_point()
            .encode(x="a", y="b")
        )

    ids = pl.Series("id", stamps, dtype=pl.Datetime("ms"))
    svg, shown = _render_capturing(chart(ids).to_svg)
    text_svg, text_shown = _render_capturing(chart(ids.cast(pl.Utf8)).to_svg)
    assert shown == text_shown == [_null_key_dropped_message("Linkage", 1, "id")]
    assert svg == text_svg
    dates = _ORDER_FRAME.with_columns(
        pl.Series("id", [dt.date(2020, 1, day) for day in (1, 2, 3, 4)])
    )
    _, clustered = _render_capturing(lambda: fm.clustermap(dates).to_svg())
    assert not [message for message in clustered if message.startswith("Linkage")]


def test_a_faceted_linkage_with_labels_reorders_each_panel_over_its_own_rows():
    """Each panel's linkage reads its own rows and skips the one null id of ``F1``: seven points
    and the one count line."""
    frame = pl.DataFrame(
        {
            "id": ["p1", None, "p3", "p4", "q1", "q2", "q3", "q4"],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
            "f": ["F1"] * 4 + ["F2"] * 4,
        }
    )
    chart = (
        fm.Chart(frame)
        .transform(
            fm.Linkage(name="o", labels="id"),
            fm.Reorder("original_idx", from_="o_order", drop_index=False),
        )
        .mark_point()
        .encode(x="a", y="b")
        .facet(col="f")
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_null_key_dropped_message("Linkage", 1, "id")]
    assert _drawn(svg, "circle") == 7
