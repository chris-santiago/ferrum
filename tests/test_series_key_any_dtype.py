"""A column of any dtype splits a mark into series.

A line, an area, a ribbon or a bare polygon layer partitions its rows by the column its ``detail``
binds (and a statistic's grouping series splits its line the same way). Every dtype the group-key
reader takes is a key: text, integers, unsigned integers, floats, ``Boolean``, and now a ``Date``, a
``Datetime`` of any unit and zone and a ``Duration`` too, where those used to merge every row into
one series with no warning. A temporal key names an instant, so two instants of one day are two
series, and two instants that read alike on a local wall clock (the hour the clocks go back) are
two series as well.

Each expected chart is stated against twins that need no temporal partition: the same frame with
the key written as an ``Int64`` of epoch milliseconds, and as text. A twin is compared in bytes and
in the ordered full list of warnings (under ``action="always"``), and every comparison also pins the
number of series drawn, so two sides that break together (one series on both) cannot pass.

What stays as it was: a partition key of another dtype draws the same bytes and raises the same
warnings (digests below), a temporal ``color``, ``shape`` or ``stroke_dash`` is refused with the text
it always had, and a temporal column on a positional band axis is read as a band, not as a
partition.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

import polars as pl
import pyarrow as pa
import pytest

import ferrum as fm
from ferrum._coerce import normalize_for_rust, to_arrow_table
from ferrum._warn import reset_warnings
from tests._svg_extents import text_runs
from tests._svg_rects import band_paths, mark_layer, polylines
from tests.test_null_group_warning import (
    _NO_FACET_ROWS_WARNING,
    _NO_ROWS_TO_DRAW,
    _all_keys_null_message,
    _detail_ignored,
    _not_drawn,
    _scale_excluded,
    _null_key_dropped_message,
    _render_capturing,
    _short_unit_message,
    _unenumerable_refusal,
    _unreadable_key_refusal,
)

# ---------------------------------------------------------------------------
# Keys and frames
# ---------------------------------------------------------------------------

_DAY = 86_400_000
_HALF_DAY = 43_200_000
_MIDNIGHTS = [1_704_067_200_000, 1_704_067_200_000 + _DAY]  # 2024-01-01 and 2024-01-02, UTC
# Two instants in one day.
_SAME_DAY = [_MIDNIGHTS[0], _MIDNIGHTS[0] + _HALF_DAY]
# 2024-10-27T00:30Z and 01:30Z: the clocks in Europe/Paris go back at 01:00Z, so both read 02:30.
_FALL_BACK = [1_729_989_000_000, 1_729_992_600_000]

# The instants of a kind that cannot tell two keys apart if it reads them by their calendar day or
# by their local wall clock: a date has only days, a Paris datetime gets the clock-change hour (two
# instants that read alike locally), every other datetime two instants of one day.
_THREE_FALL_BACK = [1_729_985_400_000, *_FALL_BACK]


def _two_keys(kind: str) -> list[int]:
    if kind == "Date":
        return _MIDNIGHTS
    return _FALL_BACK if kind == "Datetime-Paris" else _SAME_DAY


def _three_keys(kind: str) -> list[int]:
    """Three instants, listed out of time order."""
    if kind == "Date":
        return [_MIDNIGHTS[1], _MIDNIGHTS[0], _MIDNIGHTS[1] + _DAY]
    base = (
        _THREE_FALL_BACK
        if kind == "Datetime-Paris"
        else [*_SAME_DAY, _SAME_DAY[1] + _HALF_DAY // 2]
    )
    return [base[2], base[0], base[1]]


_Output = Literal["static", "interactive"]
_OUTPUTS: list[_Output] = ["static", "interactive"]


def _temporal(kind: str, millis: Sequence[int | None]) -> pl.Series:
    """The key column ``d`` of ``kind`` holding the instants ``millis`` (UTC epoch milliseconds)."""
    base = pl.Series("d", list(millis), dtype=pl.Int64)
    if kind.startswith("Duration"):
        return base.cast(pl.Duration("ms" if kind.endswith("ms") else "us"))
    instant = base.cast(pl.Datetime("ms"))
    match kind:
        case "Date":
            return instant.cast(pl.Date)
        case "Datetime-ms":
            return instant
        case "Datetime-us":
            return instant.cast(pl.Datetime("us"))
        case "Datetime-ns":
            return instant.cast(pl.Datetime("ns"))
        case "Datetime-UTC":
            return instant.dt.replace_time_zone("UTC")
        case "Datetime-Paris":
            return instant.dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/Paris")
    raise AssertionError(kind)


_KINDS = [
    "Date",
    "Datetime-ms",
    "Datetime-us",
    "Datetime-ns",
    "Datetime-UTC",
    "Datetime-Paris",
    "Duration-ms",
    "Duration-us",
]
_INSTANT_KINDS = [kind for kind in _KINDS if kind != "Date"]

# The type each kind has when it reaches the engine: milliseconds, in the column's own zone.
_REACHES_ENGINE: dict[str, pa.DataType] = {
    "Date": pa.timestamp("ms"),
    "Datetime-ms": pa.timestamp("ms"),
    "Datetime-us": pa.timestamp("ms"),
    "Datetime-ns": pa.timestamp("ms"),
    "Datetime-UTC": pa.timestamp("ms", tz="UTC"),
    "Datetime-Paris": pa.timestamp("ms", tz="Europe/Paris"),
    "Duration-ms": pa.timestamp("ms"),
    "Duration-us": pa.timestamp("ms"),
}


def _epoch(millis: Sequence[int | None]) -> pl.Series:
    """The numeric twin's key: the same instants as ``Int64`` epoch milliseconds."""
    return pl.Series("d", list(millis), dtype=pl.Int64)


def _text(millis: Sequence[int | None]) -> pl.Series:
    """The text twin's key: the same instants as text."""
    return pl.Series("d", [None if m is None else f"i{m}" for m in millis], dtype=pl.Utf8)


def _text_of_the_keys(millis: Sequence[int]) -> pl.Series:
    """Text that spells each instant as the engine keys it (epoch milliseconds): a categorical
    scale built over it lists exactly the keys of the temporal column drawn beside it."""
    return pl.Series("d", [str(m) for m in millis], dtype=pl.Utf8)


def _frame(key: pl.Series, *, panels: Sequence[str] = ()) -> pl.DataFrame:
    """Three rows per key (per facet panel when ``panels`` is given): the x ``t`` (and ``t2``, half
    a step on), two values ``v`` and ``v2`` (``v2`` above ``v``), the key ``d`` and its ordinal
    ``r`` as a ring id, and ``rx``, ``ry`` as the two ordinal positions of a rect.
    The values differ by key, so every key's series is a distinct, non-degenerate triangle."""
    t: list[float] = []
    v: list[float] = []
    f: list[str] = []
    ry: list[str] = []
    index: list[int] = []
    for panel in panels or [""]:
        for i in range(len(key)):
            for j, height in enumerate((1.0, 3.0, 2.0)):
                t.append(float(j + 1))
                v.append(height + 0.5 * i)
                f.append(panel)
                ry.append("uvw"[j])
                index.append(i)
    columns: dict[str, Any] = {
        "t": t,
        "t2": [x + 0.5 for x in t],
        "v": v,
        "v2": [x + 1.0 for x in v],
        "d": key.gather(index),
        "r": index,
        # two ordinal positions for a rect: the key's ordinal and the row's place in its series
        "rx": [str(i) for i in index],
        "ry": ry,
    }
    if panels:
        columns["f"] = f
    return pl.DataFrame(columns)


# ---------------------------------------------------------------------------
# Charts, and what each draws
# ---------------------------------------------------------------------------

_SERIES_MARKS = ["line", "area", "ribbon", "polygon"]
# The scene node a mark's series are drawn as.
_NODE_TYPE = {"line": "polyline", "area": "path", "ribbon": "path", "polygon": "polygon"}


def _chart(mark: str, frame: pl.DataFrame, **channels: Any) -> fm.Chart:
    """``mark`` over x ``t`` and y ``v`` with the channels given; a polygon is a bare polygon
    layer over a point chart, as a caller writes it."""
    encoding: dict[str, Any] = {"x": "t", "y": "v", **channels}
    if mark == "ribbon":
        encoding["y2"] = "v2"
    if mark == "text":
        encoding.setdefault("text", "d")
    if mark == "rect":
        encoding.update(x="rx", y="ry")
    if mark == "polygon":
        return fm.Chart(frame).encode(**encoding).layer(fm.Layer(mark="polygon"))
    return getattr(fm.Chart(frame), f"mark_{mark}")().encode(**encoding)


def _by_detail(mark: str, frame: pl.DataFrame) -> fm.Chart:
    return _chart(mark, frame, detail="d")


def _series_drawn(svg: str, mark: str) -> int:
    """The series (polylines, closed bands, polygon rings) a static render holds."""
    if mark == "line":
        return len(polylines(svg))
    if mark in ("area", "ribbon"):
        return len(band_paths(svg))
    return len(re.findall(r'<path d="M[^"]*Z" fill-rule="evenodd"', svg))


def _scene_nodes(chart: Any) -> list[dict[str, Any]]:
    panels = json.loads(chart.interactive().scene_json)["panels"]
    return [node for panel in panels for batch in panel["marks"] for node in batch["nodes"]]


def _draw(chart: Any, mark: str, output: _Output) -> tuple[Any, list[str]]:
    """What a chart draws for its series mark, and every warning the render raised: the svg's
    series count with the svg, or the scene nodes of the mark's kind."""
    if output == "static":
        svg, messages = _render_capturing(chart.to_svg)
        return (_series_drawn(svg, mark), svg), messages
    nodes, messages = _render_capturing(lambda: _scene_nodes(chart))
    return [n for n in nodes if n["type"] == _NODE_TYPE[mark]], messages


def _count(drawn: Any, output: _Output) -> int:
    return drawn[0] if output == "static" else len(drawn)


def _assert_draws_like_its_twins(
    build: Callable[[pl.DataFrame], Any],
    mark: str,
    output: _Output,
    millis: Sequence[int | None],
    key: pl.Series,
    *,
    series: int,
    panels: Sequence[str] = (),
) -> tuple[Any, list[str]]:
    """The chart over the temporal ``key`` draws ``series`` series, and draws byte for byte what
    the epoch and text twins draw, raising the same ordered warnings. Returns what it drew."""
    drawn, messages = _draw(build(_frame(key, panels=panels)), mark, output)
    assert _count(drawn, output) == series
    for twin in (_epoch(millis), _text(millis)):
        twin_drawn, twin_messages = _draw(build(_frame(twin, panels=panels)), mark, output)
        assert _count(twin_drawn, output) == series, "the twin must draw the series too"
        assert drawn == twin_drawn
        assert messages == twin_messages
    return drawn, messages


# ---------------------------------------------------------------------------
# One series per value, on every series mark and both outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_temporal_detail_draws_one_series_per_value(mark: str, kind: str, output: _Output):
    """Two days (or two instants) in ``detail`` are two series, as the same values as text or as
    epoch milliseconds are, where every row used to draw as one series."""
    _, messages = _assert_draws_like_its_twins(
        lambda frame: _by_detail(mark, frame),
        mark,
        output,
        _MIDNIGHTS,
        _temporal(kind, _MIDNIGHTS),
        series=2,
    )
    assert messages == []


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _INSTANT_KINDS)
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_two_instants_in_one_day_are_two_series(mark: str, kind: str, output: _Output):
    """The key is an instant, not the day it falls on."""
    _assert_draws_like_its_twins(
        lambda frame: _by_detail(mark, frame),
        mark,
        output,
        _SAME_DAY,
        _temporal(kind, _SAME_DAY),
        series=2,
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Datetime-UTC", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_two_instants_that_read_alike_on_a_local_clock_are_two_series(
    mark: str, kind: str, output: _Output
):
    """In Europe/Paris both instants read 02:30 on the hour the clocks go back; they are two
    series there and in UTC, as the epoch twin draws them."""
    _assert_draws_like_its_twins(
        lambda frame: _by_detail(mark, frame),
        mark,
        output,
        _FALL_BACK,
        _temporal(kind, _FALL_BACK),
        series=2,
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_a_zoned_or_unzoned_key_reaches_the_engine_in_milliseconds_in_its_own_zone(kind: str):
    """A zoned key must reach the engine zoned, or the matrix above runs a naive column."""
    frame = _frame(_temporal(kind, _MIDNIGHTS))
    reached = normalize_for_rust(to_arrow_table(frame)).schema.field("d").type
    assert reached == _REACHES_ENGINE[kind]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Date", "Datetime-UTC", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_key_of_three_instants_out_of_time_order_draws_three_series(
    mark: str, kind: str, output: _Output
):
    millis = _three_keys(kind)
    _assert_draws_like_its_twins(
        lambda frame: _by_detail(mark, frame),
        mark,
        output,
        millis,
        _temporal(kind, millis),
        series=3,
    )


# ---------------------------------------------------------------------------
# Every door
# ---------------------------------------------------------------------------


def _points(frame: pl.DataFrame) -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x="t", y="v")


def _layer_over_points(mark: str, frame: pl.DataFrame) -> fm.Chart:
    encoding: dict[str, Any] = {"detail": "d"}
    if mark == "ribbon":
        encoding["y2"] = "v2"
    return _points(frame).layer(fm.Layer(mark=mark, encoding=encoding))


# door -> (builder, series drawn, panels the frame carries)
_DOORS: dict[str, tuple[Callable[[str, pl.DataFrame], Any], int, tuple[str, ...]]] = {
    "single": (_by_detail, 2, ()),
    "plus": (lambda m, f: _by_detail(m, f) + _points(f), 2, ()),
    "chart-layer": (_layer_over_points, 2, ()),
    "layer-chart": (lambda m, f: fm.LayerChart(_by_detail(m, f), _points(f)), 2, ()),
    "hconcat": (lambda m, f: fm.hconcat(_by_detail(m, f), _by_detail(m, f)), 4, ()),
    "facet": (lambda m, f: _by_detail(m, f).facet("f"), 4, ("p", "q")),
    "facet-col": (lambda m, f: _by_detail(m, f).facet(col="f"), 4, ("p", "q")),
    "facet-row": (lambda m, f: _by_detail(m, f).facet(row="f"), 4, ("p", "q")),
}
# A polygon is itself a layer over a point chart, so the doors that layer a chart on it do not apply.
_POLYGON_DOORS = ["single", "hconcat", "facet", "facet-col", "facet-row"]
_DOOR_KINDS = ["Date", "Datetime-us", "Datetime-Paris", "Duration-ms"]


_DOOR_CASES = [
    (door, mark)
    for door in _DOORS
    for mark in _SERIES_MARKS
    if mark != "polygon" or door in _POLYGON_DOORS
]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _DOOR_KINDS)
@pytest.mark.parametrize("door,mark", _DOOR_CASES, ids=[f"{d}-{m}" for d, m in _DOOR_CASES])
def test_every_door_draws_one_series_per_temporal_value(
    door: str, mark: str, kind: str, output: _Output
):
    """The same series per panel and per member, through ``+``, ``Chart.layer``, ``LayerChart``,
    ``hconcat``, facet (as ``f``, ``col`` and ``row``) and the interactive route."""
    build, series, panels = _DOORS[door]
    millis = _two_keys(kind)
    _assert_draws_like_its_twins(
        lambda frame: build(mark, frame),
        mark,
        output,
        millis,
        _temporal(kind, millis),
        series=series,
        panels=panels,
    )


@pytest.mark.parametrize("kind", ["Date", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_faceted_series_splits_in_the_panel_that_holds_each_key(mark: str, kind: str):
    """Panel ``p`` holds both keys and panel ``q`` only the first: two series, then one."""
    millis = _two_keys(kind)

    def uneven(key: pl.Series) -> pl.DataFrame:
        frame = _frame(key, panels=("p", "q"))
        return frame.filter(~((pl.col("f") == "q") & (pl.col("r") == 1)))

    drawn = [
        sum(
            1
            for n in _scene_nodes(_by_detail(mark, uneven(key)).facet("f"))
            if n["type"] == _NODE_TYPE[mark]
        )
        for key in (_temporal(kind, millis), _epoch(millis))
    ]
    assert drawn == [3, 3]


# ---------------------------------------------------------------------------
# Null and all-null keys, a zero-row frame
# ---------------------------------------------------------------------------

_ENTRY = {
    "line": "mark_line",
    "area": "mark_area",
    "ribbon": "mark_ribbon",
    "polygon": "mark_polygon",
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Date", "Datetime-UTC", "Datetime-Paris", "Duration-ms"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_null_temporal_key_is_dropped_and_counted_as_any_null_key_is(
    mark: str, kind: str, output: _Output
):
    """Three rows hold a null key beside two keys: two series, the null rows counted once."""
    millis = [*_two_keys(kind), None]
    _, messages = _assert_draws_like_its_twins(
        lambda frame: _by_detail(mark, frame),
        mark,
        output,
        millis,
        _temporal(kind, millis),
        series=2,
    )
    assert messages == [_null_key_dropped_message(_ENTRY[mark], 3, "d")]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Date", "Datetime-UTC", "Duration-ms"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_temporal_key_that_is_null_on_every_row_names_it(mark: str, kind: str, output: _Output):
    millis: list[int | None] = [None, None]
    drawn, messages = _draw(_by_detail(mark, _frame(_temporal(kind, millis))), mark, output)
    twin_drawn, twin_messages = _draw(_by_detail(mark, _frame(_epoch(millis))), mark, output)
    assert _count(drawn, output) == 0
    assert drawn == twin_drawn
    assert messages == twin_messages
    expected = [_all_keys_null_message(_ENTRY[mark], "d")]
    if mark != "polygon":
        expected.append(_NO_ROWS_TO_DRAW)
    assert messages == expected


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Date", "Datetime-Paris", "Duration-us"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_zero_row_frame_with_a_temporal_key_keeps_its_one_notice(
    mark: str, kind: str, output: _Output
):
    """A frame with no rows draws no series and says so once, as the twin does."""
    frame = _frame(_temporal(kind, _MIDNIGHTS)).head(0)
    drawn, messages = _draw(_by_detail(mark, frame), mark, output)
    twin_drawn, twin_messages = _draw(
        _by_detail(mark, _frame(_epoch(_MIDNIGHTS)).head(0)), mark, output
    )
    assert _count(drawn, output) == 0
    assert drawn == twin_drawn
    assert messages == twin_messages == [_NO_ROWS_TO_DRAW]


_FACETS: dict[str, Callable[[fm.Chart], fm.Chart]] = {
    "facet": lambda chart: chart.facet("f"),
    "facet-col": lambda chart: chart.facet(col="f"),
    "facet-row": lambda chart: chart.facet(row="f"),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("facet", list(_FACETS))
@pytest.mark.parametrize("kind", ["Date", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_faceted_null_temporal_key_is_dropped_and_counted_once(
    mark: str, kind: str, facet: str, output: _Output
):
    """Each of two panels holds three null-key rows beside two keys: four series, six rows
    counted in one sentence."""
    millis = [*_two_keys(kind), None]
    panels = ("p", "q")
    drawn, messages = _draw(
        _FACETS[facet](_by_detail(mark, _frame(_temporal(kind, millis), panels=panels))),
        mark,
        output,
    )
    twin_drawn, twin_messages = _draw(
        _FACETS[facet](_by_detail(mark, _frame(_epoch(millis), panels=panels))), mark, output
    )
    assert _count(drawn, output) == 4
    assert drawn == twin_drawn
    assert messages == twin_messages == [_null_key_dropped_message(_ENTRY[mark], 6, "d")]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("facet", list(_FACETS))
@pytest.mark.parametrize("kind", ["Date", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_faceted_zero_row_frame_with_a_temporal_key_draws_no_panel(
    mark: str, kind: str, facet: str, output: _Output
):
    millis = _two_keys(kind)
    frame = _frame(_temporal(kind, millis), panels=("p", "q")).head(0)
    twin = _frame(_epoch(millis), panels=("p", "q")).head(0)
    drawn, messages = _draw(_FACETS[facet](_by_detail(mark, frame)), mark, output)
    twin_drawn, twin_messages = _draw(_FACETS[facet](_by_detail(mark, twin)), mark, output)
    assert _count(drawn, output) == 0
    assert drawn == twin_drawn
    assert messages == twin_messages == [_NO_FACET_ROWS_WARNING]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("facet", list(_FACETS))
@pytest.mark.parametrize("kind", ["Date", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_facet_panel_whose_temporal_key_is_all_null_draws_only_the_other_panel(
    mark: str, kind: str, facet: str, output: _Output
):
    """Panel ``p`` has a null key on every row, panel ``q`` two keys: two series, and the three
    rows of each key of ``p`` named in one sentence, as the epoch twin names them."""
    millis = _two_keys(kind)

    def frame(key: pl.Series) -> pl.DataFrame:
        base = _frame(key, panels=("p", "q"))
        return base.with_columns(
            pl.when(pl.col("f") == "p").then(None).otherwise(pl.col("d")).alias("d")
        )

    drawn, messages = _draw(
        _FACETS[facet](_by_detail(mark, frame(_temporal(kind, millis)))), mark, output
    )
    twin_drawn, twin_messages = _draw(
        _FACETS[facet](_by_detail(mark, frame(_epoch(millis)))), mark, output
    )
    assert _count(drawn, output) == 2
    assert drawn == twin_drawn
    assert messages == twin_messages
    assert messages[0] == _null_key_dropped_message(_ENTRY[mark], 6, "d")


# ---------------------------------------------------------------------------
# A series or ring too short to draw is named by the instant that keys it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Datetime-ms", "Datetime-UTC", "Datetime-Paris"])
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_series_or_ring_too_short_to_draw_is_named_by_its_instant(
    mark: str, kind: str, output: _Output
):
    """The second key keeps one row (two on a polygon, which needs three vertices): the other
    series draws and the short one is named once. The name is the instant read as a UTC wall clock,
    whatever the column's zone (the Paris key is 01:00 locally and still reads 00:00)."""
    frame = _frame(_temporal(kind, _MIDNIGHTS))
    keep = 3 if mark == "polygon" else 2
    short = frame.filter(~((pl.col("r") == 1) & (pl.col("t") > keep - 1)))
    twin = _frame(_epoch(_MIDNIGHTS)).filter(~((pl.col("r") == 1) & (pl.col("t") > keep - 1)))
    _, messages = _draw(_by_detail(mark, short), mark, output)
    _, twin_messages = _draw(_by_detail(mark, twin), mark, output)
    expected = _short_unit_message(mark, "d=2024-01-02T00:00:00", layer=1)
    assert messages == [expected]
    assert twin_messages == [_short_unit_message(mark, f"d={_MIDNIGHTS[1]}", layer=1)]


# ---------------------------------------------------------------------------
# A statistic's grouping series splits its line
# ---------------------------------------------------------------------------


def _grouped_frame(key: pl.Series) -> pl.DataFrame:
    """Two levels ``h`` of two keys each, three x positions per series."""
    base = _frame(key)
    return pl.concat([base.with_columns(pl.lit(level).alias("h")) for level in ("a", "b")])


_STATISTIC_DOORS: dict[str, tuple[Callable[[pl.DataFrame], fm.Chart], str, int]] = {
    "encoding aggregate": (
        lambda f: fm.Chart(f).mark_line().encode(x="t", y="mean(v)", detail="d"),
        "line",
        2,
    ),
    "encoding aggregate area": (
        lambda f: fm.Chart(f).mark_area().encode(x="t", y="mean(v)", detail="d"),
        "area",
        2,
    ),
    "encoding aggregate ribbon": (
        lambda f: fm.Chart(f).mark_ribbon().encode(x="t", y="mean(v)", y2="mean(v2)", detail="d"),
        "ribbon",
        2,
    ),
    "encoding aggregate by colour and key": (
        lambda f: fm.Chart(f).mark_line().encode(x="t", y="mean(v)", color="h", detail="d"),
        "line",
        4,
    ),
    "transform_aggregate": (
        lambda f: (
            fm.Chart(f)
            .transform(
                fm.transform_aggregate({"field": "v", "fn": "mean", "as": "m"}, groupby=["t", "d"])
            )
            .mark_line()
            .encode(x="t", y="m", detail="d")
        ),
        "line",
        2,
    ),
    "Smooth": (
        lambda f: (
            fm.Chart(f)
            .transform(fm.Smooth("t", "v", method="lm", groupby=["d"]))
            .mark_line()
            .encode(x="x", y="y", detail="d")
        ),
        "line",
        2,
    ),
    # No ``detail``: the line is split only by the field its statistic grouped by, so these cells
    # fail when the statistic's grouping series stops reading a temporal key.
    "transform_aggregate, no detail": (
        lambda f: (
            fm.Chart(f)
            .transform(
                fm.transform_aggregate({"field": "v", "fn": "mean", "as": "m"}, groupby=["t", "d"])
            )
            .mark_line()
            .encode(x="t", y="m")
        ),
        "line",
        2,
    ),
    "Smooth, no detail": (
        lambda f: (
            fm.Chart(f)
            .transform(fm.Smooth("t", "v", method="lm", groupby=["d"]))
            .mark_line()
            .encode(x="x", y="y")
        ),
        "line",
        2,
    ),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", ["Date", "Datetime-UTC", "Datetime-Paris"])
@pytest.mark.parametrize("door", list(_STATISTIC_DOORS))
def test_a_statistics_temporal_grouping_series_splits_its_line(
    door: str, kind: str, output: _Output
):
    build, mark, series = _STATISTIC_DOORS[door]
    millis = _two_keys(kind)
    drawn, messages = _draw(build(_grouped_frame(_temporal(kind, millis))), mark, output)
    for twin in (_epoch(millis), _text(millis)):
        twin_drawn, twin_messages = _draw(build(_grouped_frame(twin)), mark, output)
        assert _count(twin_drawn, output) == series
        assert drawn == twin_drawn
        assert messages == twin_messages
    assert _count(drawn, output) == series
    assert messages == []


# ---------------------------------------------------------------------------
# Unchanged: a temporal column that is not a partition key is what it was
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark", ["line", "ribbon"])
def test_a_temporal_colour_on_a_series_mark_is_what_a_numeric_colour_is(mark: str):
    """An untyped date colour is a continuous scale, which a connected mark does not split by:
    one series, the numeric twin's bytes, the numeric twin's warnings."""
    millis = _MIDNIGHTS
    outcome = []
    for key in (_temporal("Date", millis), _epoch(millis)):
        svg, messages = _render_capturing(_chart(mark, _frame(key), color="d").to_svg)
        outcome.append((svg, messages))
    assert outcome[0] == outcome[1]
    assert _series_drawn(outcome[0][0], mark) == 1
    assert outcome[0][1] == [
        f"continuous color scale is not supported on {mark}; the channel has no per-mark "
        "effect and its legend was suppressed"
    ]


_TEMPORAL_REFUSAL = _unenumerable_refusal()


def _refused(chart: Any, output: _Output) -> str:
    with pytest.raises(ValueError) as refusal:
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    return str(refusal.value)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize(
    "channel",
    [
        pytest.param({"color": fm.Color("d", type_="N")}, id="colour-nominal"),
        pytest.param({"color": fm.Color("d", type_="O")}, id="colour-ordinal"),
        pytest.param({"shape": fm.Shape("d", type_="N")}, id="shape-nominal"),
        pytest.param({"shape": fm.Shape("d", type_="O")}, id="shape-ordinal"),
        pytest.param({"shape": "d"}, id="shape-untyped"),
        pytest.param({"stroke_dash": fm.StrokeDash("d", type_="N")}, id="dash-nominal"),
        pytest.param({"stroke_dash": "d"}, id="dash-untyped"),
    ],
)
@pytest.mark.parametrize("mark", ["point", "line", "area", "ribbon", "bar", "tick", "rule", "text"])
def test_a_temporal_colour_shape_or_dash_that_names_categories_is_still_refused(
    mark: str, channel: dict[str, Any], output: _Output
):
    chart = _chart(mark, _frame(_temporal("Date", _MIDNIGHTS)), **channel)
    assert _refused(chart, output) == _TEMPORAL_REFUSAL


@pytest.mark.parametrize("output", _OUTPUTS)
def test_an_area_coloured_by_an_untyped_date_is_still_refused(output: _Output):
    chart = _chart("area", _frame(_temporal("Date", _MIDNIGHTS)), color="d")
    assert _refused(chart, output) == _TEMPORAL_REFUSAL


@pytest.mark.parametrize("mark", ["bar", "point"])
def test_a_temporal_ordinal_position_is_still_a_band_refusal_not_a_partition(mark: str):
    """Band placement reads a category column by its own rule: a date on a band axis is refused
    with the text it always had."""
    chart = _chart(mark, _frame(_temporal("Date", _MIDNIGHTS)), x=fm.X("d", type_="O"))
    assert _refused(chart, "static") == _TEMPORAL_REFUSAL


_PER_ROW_DETAIL_MARKS = ["point", "bar", "tick", "rule", "text"]


def _built_and_drawn(
    mark: str, key: pl.Series, output: _Output, *, detail: bool
) -> tuple[Any, list[str]]:
    """What a per-row mark draws (the svg, or the scene's nodes) and every warning raised while
    the chart is built and rendered. The accepted-but-ignored notice is shown once per context, so
    each call starts from a clean record of what was shown."""
    reset_warnings()
    channels = {"detail": "d"} if detail else {}
    if output == "static":
        return _render_capturing(lambda: _chart(mark, _frame(key), **channels).to_svg())
    return _render_capturing(lambda: _scene_nodes(_chart(mark, _frame(key), **channels)))


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("key", ["Date", "epoch", "text"])
@pytest.mark.parametrize("mark", _PER_ROW_DETAIL_MARKS)
def test_a_detail_on_a_per_row_mark_is_ignored_whatever_its_dtype(
    mark: str, key: str, output: _Output
):
    """A per-row mark never reads ``detail``: a date one is accepted with the notice the epoch and
    text twins get, and draws the chart that has no ``detail`` at all, byte for byte."""
    column = {
        "Date": _temporal("Date", _MIDNIGHTS),
        "epoch": _epoch(_MIDNIGHTS),
        "text": _text(_MIDNIGHTS),
    }[key]
    drawn, messages = _built_and_drawn(mark, column, output, detail=True)
    plain, plain_messages = _built_and_drawn(mark, column, output, detail=False)
    assert messages == [_detail_ignored(mark)]
    assert plain_messages == []
    assert drawn == plain


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("mark", _PER_ROW_DETAIL_MARKS)
def test_a_date_detail_on_a_per_row_mark_draws_what_its_epoch_twin_draws(
    mark: str, output: _Output
):
    date, _ = _built_and_drawn(mark, _temporal("Date", _MIDNIGHTS), output, detail=True)
    epoch, _ = _built_and_drawn(mark, _epoch(_MIDNIGHTS), output, detail=True)
    assert date == epoch


# ---------------------------------------------------------------------------
# A detail column no key reader takes is refused, not merged into one series
# ---------------------------------------------------------------------------

# Each dtype with the Arrow type it reaches the engine as, spelled as the engine spells it. A
# nominal colour on the same column is refused too (its legend cannot list the values), so a
# ``detail`` that reads what a categorical colour reads refuses what it refuses.
_UNREADABLE_KEYS: dict[str, tuple[pl.Series, str]] = {
    "Decimal": (
        pl.Series("d", [decimal.Decimal("1.5"), decimal.Decimal("2.5")], dtype=pl.Decimal(10, 2)),
        "Decimal128(10, 2)",
    ),
    "Time": (pl.Series("d", [dt.time(1), dt.time(2)]), "Time64(Nanosecond)"),
    "Binary": (pl.Series("d", [b"a", b"b"]), "LargeBinary"),
    "List": (
        pl.Series("d", [[1], [2]]),
        "LargeList(Field { data_type: Int64, nullable: true })",
    ),
    "Array": (
        pl.Series("d", [[1, 2], [2, 3]], dtype=pl.Array(pl.Int64, 2)),
        "FixedSizeList(Field { data_type: Int64, nullable: true }, 2)",
    ),
    "Struct": (
        pl.Series("d", [{"a": 1}, {"a": 2}]),
        'Struct([Field { name: "a", data_type: Int64, nullable: true }])',
    ),
}


def _unreadable_prefix(door: str, mark: str, output: _Output) -> tuple[str, int | None]:
    """How the engine names where the refusal arose: the composite leaf a static composite
    renders (a ``LayerChart`` interactive route renders one flat chart), then the layer on a
    layered chart. A polygon is always layer 1, over the point layer it is placed on."""
    leaf = "failed to render composite leaf leaf #0: "
    if door == "single":
        return "", None
    if door == "plus":
        return "", 0
    if door == "chart-layer":
        return "", 1
    if door == "layer-chart":
        return (leaf if output == "static" else ""), 0
    if door == "hconcat":
        return leaf, None
    return "", None  # a facet


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door,mark", _DOOR_CASES, ids=[f"{d}-{m}" for d, m in _DOOR_CASES])
@pytest.mark.parametrize("dtype", list(_UNREADABLE_KEYS))
def test_a_detail_column_no_key_reader_takes_is_refused_not_merged(
    dtype: str, door: str, mark: str, output: _Output
):
    """A ``detail`` of a dtype nothing can key is refused with the key reader's own text, on every
    door and both outputs, where every row used to draw as one series and no warning said so."""
    key, arrow_type = _UNREADABLE_KEYS[dtype]
    build, _, panels = _DOORS[door]
    prefix, layer = _unreadable_prefix(door, mark, output)
    if mark == "polygon":
        layer = 1
    chart = build(mark, _frame(key, panels=panels))
    assert _refused(chart, output) == _unreadable_key_refusal(
        _ENTRY[mark], "d", arrow_type, prefix=prefix, layer=layer
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("mark", _SERIES_MARKS)
def test_a_float16_detail_column_is_refused_too(mark: str, output: _Output):
    """A dtype a pyarrow table can carry that polars has no spelling of."""
    import numpy as np

    table = _frame(_epoch(_MIDNIGHTS)).drop("d").to_arrow()
    table = table.append_column("d", pa.array(np.array([1.5] * 3 + [2.5] * 3, dtype=np.float16)))
    chart = _by_detail(mark, table)
    assert _refused(chart, output) == _unreadable_key_refusal(
        _ENTRY[mark], "d", "Float16", layer=1 if mark == "polygon" else None
    )


def _as_strings(key: pl.Series) -> pl.Series:
    """The column converted to strings, as the refusal's remedy says: a list or array by joining
    its items, a struct as its JSON text, binary as its hex text."""
    if isinstance(key.dtype, pl.Array):
        key = key.cast(pl.List(key.dtype.inner))
    if isinstance(key.dtype, pl.List):
        return key.list.eval(pl.element().cast(pl.Utf8)).list.join(",")
    if isinstance(key.dtype, pl.Struct):
        return key.struct.json_encode()
    if key.dtype == pl.Binary:
        return key.bin.encode("hex")
    return key.cast(pl.Utf8)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("mark", ["line", "area", "ribbon", "polygon"])
@pytest.mark.parametrize("dtype", [*_UNREADABLE_KEYS, "Float16"])
def test_the_unreadable_detail_refusals_remedy_splits_the_mark(
    dtype: str, mark: str, output: _Output
):
    """Following the remedy as written (the column converted to strings) draws two series and
    raises nothing."""
    if dtype == "Float16":
        import numpy as np

        table = _frame(_epoch(_MIDNIGHTS)).drop("d").to_arrow()
        floats = pa.array(np.array([1.5] * 3 + [2.5] * 3, dtype=np.float16))
        data: Any = table.append_column("d", floats.cast(pa.string()))
    else:
        data = _frame(_as_strings(_UNREADABLE_KEYS[dtype][0]))
    drawn, messages = _draw(_by_detail(mark, data), mark, output)
    assert _count(drawn, output) == 2
    assert messages == []


@pytest.mark.parametrize("dtype", list(_UNREADABLE_KEYS))
def test_a_nominal_colour_on_a_column_no_key_reader_takes_is_refused_as_well(dtype: str):
    """The behavior the detail refusal follows: a categorical colour of the same column is refused
    where its legend lists the values."""
    key, arrow_type = _UNREADABLE_KEYS[dtype]
    chart = _chart("line", _frame(key), color=fm.Color("d", type_="N"))
    assert _refused(chart, "static") == _unenumerable_refusal("d", arrow_type)


# ---------------------------------------------------------------------------
# A date under a categorical scale another layer establishes is read as its twin is
# ---------------------------------------------------------------------------
#
# A layer whose own frame holds a date, drawn beside a layer whose text column has already
# established a categorical colour (or stroke_dash) scale, reads that date as a category: the
# date's Int64 epoch twin draws the same marks, with the same notice that the layer's scale
# differs from the established one. The first layer's text spells the keys as the engine keys a
# date (epoch milliseconds), so the date layer's marks map through distinct entries of that scale.
# The text twin has no such notice (its scale agrees), so it is compared on the marks it draws,
# not on its bytes.

_ROUTE_DOORS = ["plus", "chart-layer", "layer-chart"]
# The type a kind reaches the engine as, as a refusal spells it.
_ENGINE_TYPE_TEXT = {
    "Date": "Timestamp(Millisecond, None)",
    "Datetime-Paris": 'Timestamp(Millisecond, Some("Europe/Paris"))',
}
_ROUTE_KINDS = ["Date", "Datetime-Paris"]
# mark -> (the mark method, its encoding beside the channel under test)
_ROUTE_MARKS: dict[str, tuple[str, dict[str, Any]]] = {
    "point": ("point", {"x": "t", "y": "v"}),
    "line": ("line", {"x": "t", "y": "v"}),
    "ribbon": ("ribbon", {"x": "t", "y": "v", "y2": "v2"}),
    "bar": ("bar", {"x": "t", "y": "v"}),
    "text": ("text", {"x": "t", "y": "v", "text": "d"}),
    "rect": ("rect", {"x": "rx", "y": "ry"}),
    "arc": ("arc", {"x": "v"}),
    "rule": ("rule", {"x": "t", "y": "v"}),
    "tick": ("tick", {"x": "t", "y": "v"}),
    "segment-rule": ("rule", {"x": "t", "y": "v", "x2": "t2", "y2": "v2"}),
    "segment": ("segment", {"x": "t", "y": "v", "x2": "t2", "y2": "v2"}),
}
_SERIES_ROUTE_MARKS = {"line", "ribbon"}


def _route_chart(
    door: str, mark: str, channel: dict[str, Any], text: pl.DataFrame, date: pl.DataFrame
) -> Any:
    """The text layer then the layer under test, drawn through ``door``."""
    method, positions = _ROUTE_MARKS[mark]
    encoding = {**positions, **channel}

    def layer(frame: pl.DataFrame) -> fm.Chart:
        chart = getattr(fm.Chart(frame), f"mark_{method}")().encode(**encoding)
        return chart.coord(fm.CoordPolar(theta="x")) if mark == "arc" else chart

    first = layer(text)
    if door == "plus":
        return first + layer(date)
    if door == "chart-layer":
        return first.layer(fm.Layer(mark=method, data=date, encoding=encoding))
    return fm.LayerChart(first, layer(date))


def _route_draw(chart: Any, output: _Output) -> tuple[Any, list[str]]:
    """The svg, or every scene node, and every warning raised."""
    if output == "static":
        return _render_capturing(chart.to_svg)
    return _render_capturing(lambda: _scene_nodes(chart))


def _scene_node_count(chart: Any) -> int:
    """The marks the interactive scene holds (any notice the build raises is not this count's)."""
    return _render_capturing(lambda: len(_scene_nodes(chart)))[0]


def _assert_route_draws_like_its_twins(
    build: Callable[[pl.DataFrame, pl.DataFrame], Any],
    millis: Sequence[int],
    kind: str,
    output: _Output,
) -> None:
    text = _frame(pl.Series("d", ["a", "b"]))
    chart = build(text, _frame(_temporal(kind, millis)))
    epoch = build(text, _frame(_epoch(millis)))
    twin_text = build(text, _frame(_text(millis)))
    drawn, messages = _route_draw(chart, output)
    epoch_drawn, epoch_messages = _route_draw(epoch, output)
    assert drawn == epoch_drawn
    assert messages == epoch_messages
    assert _scene_node_count(chart) == _scene_node_count(epoch) == _scene_node_count(twin_text)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize("mark", ["point", "line", "ribbon", "bar", "text", "rect", "arc"])
def test_a_date_colour_under_a_sibling_categorical_colour_draws_as_its_epoch_twin_does(
    mark: str, door: str, kind: str, output: _Output
):
    _assert_route_draws_like_its_twins(
        lambda text, date: _route_chart(door, mark, {"color": "d"}, text, date),
        _two_keys(kind),
        kind,
        output,
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize("mark", sorted(_SERIES_ROUTE_MARKS))
def test_a_date_colour_beside_a_text_colour_splits_each_layer_into_its_series(
    mark: str, door: str, kind: str, output: _Output
):
    """Two series in each layer: the date layer splits as its epoch twin does, where the date
    colour used to leave that layer one series."""
    chart = _route_chart(
        door,
        mark,
        {"color": "d"},
        _frame(_text_of_the_keys(_two_keys(kind))),
        _frame(_temporal(kind, _two_keys(kind))),
    )
    drawn, _ = _route_draw(chart, output)
    if output == "static":
        assert _series_drawn(drawn, mark) == 4
    else:
        assert len([n for n in drawn if n["type"] == _NODE_TYPE[mark]]) == 4


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize("mark", ["rule", "tick", "segment-rule", "segment"])
def test_a_date_colour_on_a_rule_tick_or_segment_draws_as_its_epoch_twin_does(
    mark: str, door: str, kind: str, output: _Output
):
    """The per-row colour read agrees with the partition read: the marks that read it for each row
    draw a date as every other mark draws it."""
    _assert_route_draws_like_its_twins(
        lambda text, date: _route_chart(door, mark, {"color": "d"}, text, date),
        _two_keys(kind),
        kind,
        output,
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
def test_a_date_colour_on_an_area_under_a_sibling_colour_is_still_refused(
    door: str, kind: str, output: _Output
):
    text = _frame(_text_of_the_keys(_two_keys(kind)))
    date = _frame(_temporal(kind, _two_keys(kind)))
    encoding = {"x": "t", "y": "v", "color": "d"}
    first = fm.Chart(text).mark_area().encode(**encoding)
    if door == "plus":
        chart = first + fm.Chart(date).mark_area().encode(**encoding)
    elif door == "chart-layer":
        chart = first.layer(fm.Layer(mark="area", data=date, encoding=encoding))
    else:
        chart = fm.LayerChart(first, fm.Chart(date).mark_area().encode(**encoding))
    assert _refused(chart, output) == _sibling_refusal(door, _ENGINE_TYPE_TEXT[kind], output)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize("mark", ["point", "line", "ribbon", "bar", "rect", "rule", "segment-rule"])
def test_a_date_stroke_dash_under_a_sibling_dash_scale_draws_as_its_epoch_twin_does(
    mark: str, door: str, kind: str, output: _Output
):
    _assert_route_draws_like_its_twins(
        lambda text, date: _route_chart(door, mark, {"stroke_dash": "d"}, text, date),
        _two_keys(kind),
        kind,
        output,
    )


def _sibling_refusal(door: str, arrow_type: str, output: _Output) -> str:
    """What a later layer's column with no categories is refused with when another layer
    establishes the scale: the text a legend gives, with the leaf a static ``LayerChart`` renders."""
    leaf = (
        "failed to render composite leaf leaf #0: "
        if door == "layer-chart" and output == "static"
        else ""
    )
    return leaf + _unenumerable_refusal("d", arrow_type)


def _later_point_layer(
    door: str, lead: fm.Chart, frame: pl.DataFrame, encoding: dict[str, Any], mark: str = "point"
) -> Any:
    """``lead`` with a second layer of ``mark`` over ``frame``, through ``door``."""
    later = getattr(fm.Chart(frame), f"mark_{mark}")().encode(**encoding)
    if door == "plus":
        return lead + later
    if door == "chart-layer":
        return lead.layer(fm.Layer(mark=mark, data=frame, encoding=encoding))
    return fm.LayerChart(lead, later)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize("lead_domain", [False, True], ids=["plain", "explicit-domain"])
def test_a_later_layers_date_shape_under_a_sibling_shape_scale_is_refused_as_when_it_establishes_it(
    lead_domain: bool, door: str, kind: str, output: _Output
):
    """A shape scale reads every drawing layer's column as categories, so a later layer's date is
    refused with the text it gets as the first layer; its Int64 twin joins the scale and draws."""
    millis = _two_keys(kind)
    text = _frame(_text_of_the_keys(millis))
    shape = _explicit_scale("shape", [str(ms) for ms in millis]) if lead_domain else "d"
    lead = fm.Chart(text).mark_point().encode(x="t", y="v", shape=shape)
    encoding = {"x": "t", "y": "v", "shape": "d"}
    date = _later_point_layer(door, lead, _frame(_temporal(kind, millis)), encoding)
    assert _refused(date, output) == _sibling_refusal(door, _ENGINE_TYPE_TEXT[kind], output)
    twin = _later_point_layer(door, lead, _frame(_epoch(millis)), encoding)
    drawn, messages = _route_draw(twin, output)
    assert messages == []
    if output == "static":
        # twelve marks (six circles and six squares): the legend lists both keys
        assert len(re.findall(r"<circle", mark_layer(drawn))) == 6
        assert {str(ms) for ms in millis} <= set(text_runs(drawn))
    else:
        assert Counter(node["type"] for node in drawn) == {"circle": 6, "rect": 6}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize(
    "channel,mark", [("stroke_dash", "line"), ("stroke_dash", "point"), ("shape", "point")]
)
@pytest.mark.parametrize("dtype", ["Decimal", "Time"])
def test_a_later_layers_unreadable_dash_or_shape_is_refused_by_its_scale(
    dtype: str, channel: str, mark: str, door: str, output: _Output
):
    key, arrow_type = _UNREADABLE_KEYS[dtype]
    text = _frame(_text_of_the_keys(_MIDNIGHTS))
    encoding = {"x": "t", "y": "v", channel: "d"}
    lead = getattr(fm.Chart(text), f"mark_{mark}")().encode(**encoding)
    chart = _later_point_layer(door, lead, _frame(key), encoding, mark)
    assert _refused(chart, output) == _sibling_refusal(door, arrow_type, output)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
def test_a_later_layers_date_dash_beside_an_explicit_domain_stays_out_and_splits(
    door: str, kind: str, output: _Output
):
    """The lead's text dash carries an explicit domain; the later layer's date dash reads as
    numbers, stays out of that scale with the notice, and splits its line by instant as the Int64
    twin's does: four polylines in all."""
    millis = _two_keys(kind)
    text = _frame(_text_of_the_keys(millis))
    lead_dash = fm.StrokeDash(
        "d", type_="N", scale=fm.OrdinalScale(domain=[str(ms) for ms in millis])
    )
    lead = fm.Chart(text).mark_line().encode(x="t", y="v", stroke_dash=lead_dash)
    encoding = {"x": "t", "y": "v", "stroke_dash": "d"}

    def draw(key: pl.Series) -> Any:
        return _route_draw(_later_point_layer(door, lead, _frame(key), encoding, "line"), output)

    drawn, messages = draw(_temporal(kind, millis))
    twin, twin_messages = draw(_epoch(millis))
    assert drawn == twin
    assert messages == twin_messages
    assert messages == [_scale_excluded("stroke_dash")]
    if output == "static":
        assert _series_drawn(drawn, "line") == 4
    else:
        assert len([n for n in drawn if n["type"] == "polyline"]) == 4


def _paints(svg: str) -> list[str]:
    """Every fill, stroke and dash the marks of a static render carry, in drawing order."""
    return re.findall(r'(?:fill|stroke|stroke-dasharray)="([^"]*)"', mark_layer(svg))


@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("door", _ROUTE_DOORS)
@pytest.mark.parametrize(
    "channel,mark",
    [
        *[
            ("color", m)
            for m in ("point", "line", "ribbon", "bar", "rect", "arc", "rule", "segment-rule")
        ],
        *[
            ("stroke_dash", m)
            for m in ("point", "line", "ribbon", "bar", "rect", "rule", "segment-rule")
        ],
    ],
)
def test_a_date_layer_maps_each_key_through_the_scale_a_sibling_lists_it_in(
    mark: str, channel: str, door: str, kind: str
):
    """The first layer's text spells the two keys, so the scale lists them and the date layer's
    marks take two distinct paints from it, in the order the epoch twin's marks take them. (The
    legend then spells the dates differently, so the bytes are not compared, only each mark's
    paint.)"""
    millis = _two_keys(kind)
    text = _frame(_text_of_the_keys(millis))

    def paints(key: pl.Series) -> list[str]:
        chart = _route_chart(door, mark, {channel: "d"}, text, _frame(key))
        return _paints(_render_capturing(chart.to_svg)[0])

    drawn = paints(_temporal(kind, millis))
    assert drawn == paints(_epoch(millis))
    # not vacuous: one key for both rows takes other paints than two keys do
    assert drawn != paints(_temporal(kind, [millis[0], millis[0]]))


def _polygon_paints(svg: str) -> list[str]:
    """The fill of each polygon ring, in drawing order."""
    return re.findall(r'<path d="M[^"]*Z" fill-rule="evenodd" fill="([^"]*)"', svg)


def _ring_count(drawn: Any, output: _Output) -> int:
    """The rings a static render (an svg) or an interactive one (its scene nodes) holds."""
    if output == "static":
        return len(_polygon_paints(drawn))
    return len([n for n in drawn if n["type"] == "polygon"])


def _bare_polygon_route(text: pl.DataFrame, date: pl.DataFrame, *, detail: bool) -> fm.Chart:
    """A text colour on a point layer, then a polygon layer over ``date`` coloured by ``d``,
    with or without a ``detail`` that names its rings."""
    encoding = {"x": "t", "y": "v", "color": "d", **({"detail": "r"} if detail else {})}
    return (
        fm.Chart(text)
        .mark_point()
        .encode(x="t", y="v", color="d")
        .layer(fm.Layer(mark="polygon", data=date, encoding=encoding))
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
def test_a_date_colour_paints_a_polygon_layers_ring_and_does_not_split_it(
    kind: str, output: _Output
):
    """Rings come from ``detail`` alone: with none, the bare polygon layer is one ring whatever
    its colour holds, as its Int64 twin and a text colour draw it, with the notice that the
    layer's colour scale stays out of the first layer's."""
    millis = _two_keys(kind)
    text = _frame(_text_of_the_keys(millis))

    def draw(key: pl.Series) -> Any:
        return _route_draw(_bare_polygon_route(text, _frame(key), detail=False), output)

    drawn, messages = draw(_temporal(kind, millis))
    twin, twin_messages = draw(_epoch(millis))
    assert drawn == twin
    assert messages == twin_messages == [_scale_excluded("color")]
    assert _ring_count(drawn, output) == 1
    # a text colour is one ring too (its scale agrees, so it raises no notice)
    text_drawn, text_messages = draw(_text_of_the_keys(millis))
    assert text_messages == []
    assert _ring_count(text_drawn, output) == 1


@pytest.mark.parametrize("kind", _ROUTE_KINDS)
def test_a_polygon_layers_rings_take_the_paints_of_the_instants_their_colour_holds(kind: str):
    """With ``detail`` naming the two rings, each takes the paint its instant's key maps to in the
    scale the first layer lists (the spellings of the keys), where a merged key paints both the
    first one's."""
    millis = _two_keys(kind)
    text = _frame(_text_of_the_keys(millis))

    def paints(key: pl.Series) -> list[str]:
        chart = _bare_polygon_route(text, _frame(key), detail=True)
        return _polygon_paints(_render_capturing(chart.to_svg)[0])

    drawn = paints(_temporal(kind, millis))
    assert len(drawn) == 2
    assert drawn[0] != drawn[1]
    assert drawn == paints(_epoch(millis))
    assert drawn == paints(_text_of_the_keys(millis))


def _polygon_route(text: pl.DataFrame, date: pl.DataFrame) -> fm.Chart:
    encoding = {"x": "t", "y": "v", "detail": "r", "color": "d"}
    return (
        fm.Chart(text)
        .mark_point()
        .encode(x="t", y="v", color="d")
        .layer(fm.Layer(mark="polygon", data=date, encoding=encoding))
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
def test_a_date_colour_on_a_polygon_layer_under_a_sibling_colour_draws_as_its_epoch_twin_does(
    kind: str, output: _Output
):
    _assert_route_draws_like_its_twins(_polygon_route, _two_keys(kind), kind, output)


# ---------------------------------------------------------------------------
# An explicit domain does not make a column with no categories a category
# ---------------------------------------------------------------------------
#
# A categorical stroke_dash, shape or colour scale reads its column as categories. A column nothing
# can key as one (a date, a datetime, a duration, a decimal, a time, binary or nested values) is
# refused when the scale is resolved, with or without an explicit ``domain=``, on every door: the
# declared domain does not stand in for the column's values. The Int64 twin of the same column
# draws, a line splitting by it.


def _explicit_scale(channel: str, domain: list[str], type_: str | None = "N") -> Any:
    kinds = {"stroke_dash": fm.StrokeDash, "shape": fm.Shape, "color": fm.Color}
    typed = {} if type_ is None else {"type_": type_}
    return kinds[channel]("d", scale=fm.OrdinalScale(domain=domain), **typed)


_EXPLICIT_REFUSED = [
    *[
        ("stroke_dash", m)
        for m in ("point", "line", "ribbon", "bar", "rect", "rule", "segment-rule")
    ],
    ("shape", "point"),
]
_EXPLICIT_DOORS = [
    "single",
    "plus",
    "chart-layer",
    "layer-chart",
    "hconcat",
    "facet",
    "facet-col",
    "facet-row",
]


def _explicit_marks(
    mark: str, channel: str, frame: pl.DataFrame, domain: list[str], type_: str | None
) -> tuple[fm.Chart, str, dict[str, Any]]:
    """The mark with no partition, and the method and encoding of the same mark partitioned by
    ``d`` through an explicit ordinal scale over ``domain``."""
    method, positions = _ROUTE_MARKS[mark]
    plain = getattr(fm.Chart(frame), f"mark_{method}")().encode(**positions)
    return plain, method, {**positions, channel: _explicit_scale(channel, domain, type_)}


def _explicit_door(
    door: str,
    mark: str,
    channel: str,
    frame: pl.DataFrame,
    *,
    domain: list[str] | None = None,
    type_: str | None = "N",
) -> Any:
    """The partitioned mark through ``door``; where the door layers, it is the second layer."""
    plain, method, encoding = _explicit_marks(mark, channel, frame, domain or ["p", "q"], type_)
    partitioned = getattr(fm.Chart(frame), f"mark_{method}")().encode(**encoding)
    if door in ("single", "facet", "facet-col", "facet-row"):
        return {
            "single": lambda c: c,
            "facet": lambda c: c.facet("f"),
            "facet-col": lambda c: c.facet(col="f"),
            "facet-row": lambda c: c.facet(row="f"),
        }[door](partitioned)
    if door == "plus":
        return plain + partitioned
    if door == "chart-layer":
        return plain.layer(fm.Layer(mark=method, encoding=encoding))
    if door == "layer-chart":
        return fm.LayerChart(plain, partitioned)
    return fm.hconcat(plain, partitioned)


def _explicit_refusal(door: str, arrow_type: str, output: _Output) -> str:
    """The refusal every door raises: the scale cannot list the column's values. A composite adds
    the leaf it renders, an ``hconcat`` leaf and a static ``LayerChart`` member."""
    composite = door == "hconcat" or (door == "layer-chart" and output == "static")
    leaf = "failed to render composite leaf leaf #1: " if composite else ""
    return leaf + _unenumerable_refusal("d", arrow_type)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _EXPLICIT_DOORS)
@pytest.mark.parametrize(
    "channel,mark", _EXPLICIT_REFUSED, ids=[f"{c}-{m}" for c, m in _EXPLICIT_REFUSED]
)
@pytest.mark.parametrize("dtype", list(_UNREADABLE_KEYS))
def test_an_explicit_scale_over_a_column_no_key_reader_takes_is_refused_not_ignored(
    dtype: str, channel: str, mark: str, door: str, output: _Output
):
    key, arrow_type = _UNREADABLE_KEYS[dtype]
    panels = ("p", "q") if door.startswith("facet") else ()
    chart = _explicit_door(door, mark, channel, _frame(key, panels=panels))
    assert _refused(chart, output) == _explicit_refusal(door, arrow_type, output)


def _iso(ms: int) -> str:
    return (dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=ms)).isoformat()


# (channel, mark, type): every drawn channel and mark typed nominal, and the other typings where a
# column with no category could be read either way
_DATE_DOMAIN_CASES = [
    *[(c, m, "N") for c, m in _EXPLICIT_REFUSED],
    ("color", "line", "N"),
    ("color", "point", "N"),
    *[(c, m, t) for c, m in (("stroke_dash", "line"), ("shape", "point")) for t in ("O", None)],
]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _EXPLICIT_DOORS)
@pytest.mark.parametrize("spelling", ["iso", "epoch"])
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize(
    "channel,mark,type_", _DATE_DOMAIN_CASES, ids=[f"{c}-{m}-{t}" for c, m, t in _DATE_DOMAIN_CASES]
)
def test_an_explicit_domain_does_not_make_a_temporal_colour_dash_or_shape_a_category(
    channel: str, mark: str, type_: str | None, kind: str, spelling: str, door: str, output: _Output
):
    """A date, datetime or Paris datetime under an explicit domain (spelled as dates or as the
    epoch milliseconds the engine keys them by) is refused as it is without one: where the chart
    used to draw every mark in the default dash or glyph under a legend of the domain's."""
    millis = _two_keys(kind)
    domain = [_iso(ms) if spelling == "iso" else str(ms) for ms in millis]
    panels = ("p", "q") if door.startswith("facet") else ()
    chart = _explicit_door(
        door,
        mark,
        channel,
        _frame(_temporal(kind, millis), panels=panels),
        domain=domain,
        type_=type_,
    )
    assert _refused(chart, output) == _explicit_refusal(door, _ENGINE_TYPE_TEXT[kind], output)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _EXPLICIT_DOORS)
@pytest.mark.parametrize(
    "channel,mark", _EXPLICIT_REFUSED, ids=[f"{c}-{m}" for c, m in _EXPLICIT_REFUSED]
)
def test_the_same_explicit_scale_over_the_int64_twin_draws(
    channel: str, mark: str, door: str, output: _Output
):
    """The control: the twin is not refused, and a line splits into its two series by it."""
    panels = ("p", "q") if door.startswith("facet") else ()
    chart = _explicit_door(door, mark, channel, _frame(_epoch(_MIDNIGHTS), panels=panels))
    drawn, messages = _route_draw(chart, output)
    assert messages == []
    if mark == "line" and channel == "stroke_dash" and output == "static":
        # the partitioned line draws two series; the plain one beside it draws one
        lines = {"single": 2, "plus": 3, "chart-layer": 3, "layer-chart": 3, "hconcat": 3}
        assert _series_drawn(drawn, "line") == lines.get(door, 4)
    assert drawn


# A mark that does not draw the channel never reads the column: the explicit scale on a column of
# any dtype is ignored with the notice that names it, exactly as the Int64 twin's is.

_UNDRAWN_DASH_MARKS = ["area", "polygon", "arc", "tick", "segment"]
_UNDRAWN_SHAPE_MARKS = [
    "line",
    "ribbon",
    "bar",
    "rect",
    "rule",
    "segment-rule",
    "tick",
    "segment",
    "area",
    "polygon",
    "arc",
]
_UNDRAWN_MARK_NAMES = {"segment-rule": "rule"}


def _undrawn_chart(mark: str, channel: str, frame: pl.DataFrame) -> fm.Chart:
    scale = _explicit_scale(channel, ["p", "q"])
    if mark == "polygon":
        return (
            fm.Chart(frame)
            .mark_point()
            .encode(x="t", y="v")
            .layer(
                fm.Layer(
                    mark="polygon", encoding={"x": "t", "y": "v", "detail": "r", channel: scale}
                )
            )
        )
    if mark == "arc":
        chart = fm.Chart(frame).mark_arc().encode(x="v", **{channel: scale})
        return chart.coord(fm.CoordPolar(theta="x"))
    if mark == "segment":
        positions = {"x": "t", "y": "v", "x2": "t2", "y2": "v2"}
        return fm.Chart(frame).mark_segment().encode(**positions, **{channel: scale})
    if mark == "tick":
        return fm.Chart(frame).mark_tick().encode(x="t", y="v", **{channel: scale})
    method, positions = _ROUTE_MARKS.get(mark, (mark, {"x": "t", "y": "v"}))
    return getattr(fm.Chart(frame), f"mark_{method}")().encode(**positions, **{channel: scale})


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize(
    "channel,mark",
    [
        *[("stroke_dash", m) for m in _UNDRAWN_DASH_MARKS],
        *[("shape", m) for m in _UNDRAWN_SHAPE_MARKS],
    ],
)
@pytest.mark.parametrize("dtype", list(_UNREADABLE_KEYS))
def test_a_mark_that_does_not_draw_the_channel_ignores_a_column_no_key_reader_takes(
    dtype: str, channel: str, mark: str, output: _Output
):
    """Not refused and not merged: the notice that the channel has no effect, named once, and the
    chart the Int64 twin draws, byte for byte."""
    key, _ = _UNREADABLE_KEYS[dtype]
    drawn, messages = _route_draw(_undrawn_chart(mark, channel, _frame(key)), output)
    twin, twin_messages = _route_draw(
        _undrawn_chart(mark, channel, _frame(_epoch(_MIDNIGHTS))), output
    )
    name = _UNDRAWN_MARK_NAMES.get(mark, mark)
    entry = f"layer 1: mark_{name}" if mark == "polygon" else f"mark_{name}"
    assert messages == [_not_drawn(entry, channel, "d", name)]
    assert drawn == twin
    assert messages == twin_messages


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", ["single", "plus-right", "plus-left"])
@pytest.mark.parametrize("kind", ["Date", "Datetime-Paris"])
@pytest.mark.parametrize(
    "channel,mark",
    [
        *[("stroke_dash", m) for m in _UNDRAWN_DASH_MARKS],
        *[("shape", m) for m in _UNDRAWN_SHAPE_MARKS],
    ],
)
def test_a_mark_that_does_not_draw_the_channel_ignores_an_explicit_domain_over_a_date(
    channel: str, mark: str, kind: str, door: str, output: _Output
):
    """An explicit domain over a date on a mark that ignores the channel is not refused: the one
    notice that the channel has no effect (naming its layer beside another), and the chart the
    Int64 twin draws, byte for byte. Through ``+`` the undrawn chart is the right or the left
    operand."""

    def build(key: pl.Series) -> Any:
        frame = _frame(key)
        chart = _undrawn_chart(mark, channel, frame)
        # a rect sits on two ordinal axes, which a point on continuous ones cannot share
        other = (
            fm.Chart(frame).mark_rect().encode(x="rx", y="ry")
            if mark == "rect"
            else fm.Chart(frame).mark_point().encode(x="t", y="v")
        )
        return {"single": chart, "plus-right": other + chart, "plus-left": chart + other}[door]

    millis = _two_keys(kind)
    drawn, messages = _route_draw(build(_temporal(kind, millis)), output)
    twin, twin_messages = _route_draw(build(_epoch(millis)), output)
    name = _UNDRAWN_MARK_NAMES.get(mark, mark)
    if mark == "polygon":
        entry = f"layer {2 if door == 'plus-right' else 1}: mark_{name}"
    else:
        entry = {"single": "", "plus-right": "layer 1: ", "plus-left": "layer 0: "}[
            door
        ] + f"mark_{name}"
    assert messages == [_not_drawn(entry, channel, "d", name)]
    assert messages == twin_messages
    assert drawn == twin


# ---------------------------------------------------------------------------
# A position adjustment by a temporal column groups as its twins do
# ---------------------------------------------------------------------------

# mark -> its encoding, for the marks that take each adjustment
_STACKED: dict[str, dict[str, str]] = {
    "area": {"x": "t", "y": "v"},
    "bar": {"x": "cat", "y": "v"},
    "ribbon": {"x": "t", "y": "v", "y2": "v2"},
    "point": {"x": "cat", "y": "v"},
    "text": {"x": "cat", "y": "v", "text": "v"},
    "rule": {"x": "cat", "y": "v"},
    "tick": {"x": "cat", "y": "v"},
}
_DODGED: dict[str, dict[str, str]] = {
    "bar": {"x": "cat", "y": "v"},
    "point": {"x": "cat", "y": "v"},
    "ribbon": {"x": "cat", "y": "v", "y2": "v2"},
    "text": {"x": "cat", "y": "v", "text": "v"},
}


def _adjusted_chart(mark: str, adjustment: str, key: pl.Series, *, by: bool = True) -> fm.Chart:
    """``mark`` over three positions ``cat`` with the key ``d`` stacked (or dodged) by; with
    ``by=False`` the same chart with no adjustment at all."""
    encoding = (_STACKED if adjustment == "stack" else _DODGED)[mark]
    frame = _frame(key).with_columns(cat=pl.Series(["u", "v", "w"] * len(key)))
    # The first key's last row listed last, so at that position the rows' order is the reverse of
    # the order the groups first appear in: a stack by the key and a stack of one group differ.
    frame = frame[[0, 1, 3, 4, 5, 2]]
    kwargs: dict[str, Any] = {}
    if by:
        kwargs["position"] = fm.Stack(by="d") if adjustment == "stack" else fm.Dodge(by="d")
    return getattr(fm.Chart(frame), f"mark_{mark}")(**kwargs).encode(**encoding)


_ADJUSTMENT_CASES = [("stack", m) for m in _STACKED] + [("dodge", m) for m in _DODGED]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "adjustment,mark", _ADJUSTMENT_CASES, ids=[f"{a}-{m}" for a, m in _ADJUSTMENT_CASES]
)
def test_a_position_adjustment_by_a_temporal_column_groups_as_its_twins_do(
    adjustment: str, mark: str, kind: str, output: _Output
):
    """Stacking or dodging by a date, a datetime or a duration offsets the marks as the same
    column written as epoch milliseconds or as text does, and is not the chart with no
    adjustment, where the marks used to stay in place with a warning that nothing was offset."""
    # listed latest first: the groups stack in time order, not in the order the rows list them
    millis = list(reversed(_two_keys(kind)))
    date_out, date_messages = _route_draw(
        _adjusted_chart(mark, adjustment, _temporal(kind, millis)), output
    )
    for twin in (_epoch(millis), _text(millis)):
        twin_out, twin_messages = _route_draw(_adjusted_chart(mark, adjustment, twin), output)
        assert date_out == twin_out
        assert date_messages == twin_messages
    plain, _ = _route_draw(_adjusted_chart(mark, adjustment, _epoch(millis), by=False), output)
    assert date_out != plain
    assert date_messages == []


_UNREADABLE_BY = ["Decimal", "Time"]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("dtype", _UNREADABLE_BY)
@pytest.mark.parametrize(
    "adjustment,mark", _ADJUSTMENT_CASES, ids=[f"{a}-{m}" for a, m in _ADJUSTMENT_CASES]
)
def test_a_position_adjustment_by_a_column_no_reader_takes_still_says_nothing_was_offset(
    adjustment: str, mark: str, dtype: str, output: _Output
):
    key, arrow_type = _UNREADABLE_KEYS[dtype]
    _, messages = _route_draw(_adjusted_chart(mark, adjustment, key), output)
    assert messages == [
        f"{adjustment} could not be applied (by-column 'd' has dtype {arrow_type}, which "
        "cannot group categories); marks were not offset"
    ]


# ---------------------------------------------------------------------------
# Unchanged: every other partition key draws what it drew. Each digest is the SHA-256 of the
# static render (or of the interactive scene's nodes) with the ordered warnings it raised; a case
# that raises records the refusal. The static digests and the scene digests of the ``detail``
# cases were measured before the partition reads changed; the others (scene digests of colour,
# shape and stroke_dash, and the rect dash cases) were measured on the changed engine, where
# every dtype but a temporal one keys through the same reader as before, which the earlier
# digests of the same charts, unchanged, show.
# ---------------------------------------------------------------------------

_PLAIN_KEYS: dict[str, pl.Series] = {
    "Utf8": pl.Series("d", ["a", "b"], dtype=pl.Utf8),
    "Categorical": pl.Series("d", ["a", "b"], dtype=pl.Categorical),
    "Enum": pl.Series("d", ["a", "b"], dtype=pl.Enum(["a", "b"])),
    "Int64": pl.Series("d", [1, 2], dtype=pl.Int64),
    "Int8": pl.Series("d", [1, 2], dtype=pl.Int8),
    "UInt8": pl.Series("d", [1, 2], dtype=pl.UInt8),
    "UInt32": pl.Series("d", [1, 2], dtype=pl.UInt32),
    "Float64": pl.Series("d", [1.5, 2.5], dtype=pl.Float64),
    "Float32": pl.Series("d", [1.5, 2.5], dtype=pl.Float32),
    "Boolean": pl.Series("d", [False, True], dtype=pl.Boolean),
}
# The dtypes a categorical colour, shape or stroke_dash is read from, one per family.
_CHANNEL_KEYS = ["Utf8", "Int64", "UInt8", "Float64", "Boolean"]
_PER_ROW_MARKS = ["point", "bar", "tick", "rule", "text"]


def _unchanged_chart(case: str) -> fm.Chart:
    """The chart a case id names: ``channel:mark:dtype``."""
    channel, mark, dtype = case.split(":")
    frame = _frame(_PLAIN_KEYS[dtype])
    if channel == "detail":
        return _by_detail(mark, frame)
    nominal = {"color": fm.Color, "shape": fm.Shape, "stroke_dash": fm.StrokeDash}[channel]
    extra = {"detail": "r"} if mark == "polygon" else {}
    return _chart(mark, frame, **{channel: nominal("d", type_="N")}, **extra)


def _unchanged_cases() -> list[str]:
    cases = [f"detail:{m}:{k}" for m in _SERIES_MARKS for k in _PLAIN_KEYS]
    marks = [*_SERIES_MARKS, *_PER_ROW_MARKS]
    cases += [f"color:{m}:{k}" for m in marks for k in _CHANNEL_KEYS]
    cases += [f"shape:point:{k}" for k in _CHANNEL_KEYS]
    cases += [
        f"stroke_dash:{m}:{k}"
        for m in ("line", "ribbon", "point", "bar", "rule", "rect")
        for k in _CHANNEL_KEYS
    ]
    return cases


def _outcome(case: str, output: _Output) -> tuple[str, tuple[str, ...]]:
    """The digest a case draws and its warnings, or the refusal it raises."""
    try:
        chart = _unchanged_chart(case)
        if output == "static":
            text, messages = _render_capturing(chart.to_svg)
        else:
            nodes, messages = _render_capturing(lambda: _scene_nodes(chart))
            text = json.dumps(nodes, sort_keys=True)
    except ValueError as refusal:
        return f"refused: {refusal}", ()
    return hashlib.sha256(text.encode()).hexdigest(), tuple(messages)


_STATIC_DIGESTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "detail:line:Utf8": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:Categorical": (
        "542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e",
        (),
    ),
    "detail:line:Enum": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:Int64": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:Int8": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:UInt8": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:UInt32": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:Float64": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:Float32": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:line:Boolean": ("542721ab8da6120009e2c9899c362a92e186ce8480b210c20ffc4a40368d4b8e", ()),
    "detail:area:Utf8": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:Categorical": (
        "19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9",
        (),
    ),
    "detail:area:Enum": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:Int64": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:Int8": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:UInt8": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:UInt32": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:Float64": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:Float32": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:area:Boolean": ("19ba35f4efb58b0669eb501aea1f007b48e4fe0c407a3dbd3df116599ef857a9", ()),
    "detail:ribbon:Utf8": ("7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265", ()),
    "detail:ribbon:Categorical": (
        "7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265",
        (),
    ),
    "detail:ribbon:Enum": ("7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265", ()),
    "detail:ribbon:Int64": ("7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265", ()),
    "detail:ribbon:Int8": ("7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265", ()),
    "detail:ribbon:UInt8": ("7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265", ()),
    "detail:ribbon:UInt32": (
        "7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265",
        (),
    ),
    "detail:ribbon:Float64": (
        "7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265",
        (),
    ),
    "detail:ribbon:Float32": (
        "7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265",
        (),
    ),
    "detail:ribbon:Boolean": (
        "7396619d22eb8d2da5a97dd481e8865b202b59c65828016e8d4ddf020657e265",
        (),
    ),
    "detail:polygon:Utf8": ("1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2", ()),
    "detail:polygon:Categorical": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "detail:polygon:Enum": ("1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2", ()),
    "detail:polygon:Int64": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "detail:polygon:Int8": ("1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2", ()),
    "detail:polygon:UInt8": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "detail:polygon:UInt32": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "detail:polygon:Float64": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "detail:polygon:Float32": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "detail:polygon:Boolean": (
        "1761797cd1ba91fea0c9deab0b99f116fe7a2bb0965d150581fc6aa4c4ebdec2",
        (),
    ),
    "color:line:Utf8": ("31c6fc29f854f2047d4d54b8b2c72156f33e0d18cb64ef0f23e380b530e70d74", ()),
    "color:line:Int64": ("da7a1ca730be70218b5bd7412c1a56012d2d90b0565259771fb2b472ed177bc1", ()),
    "color:line:UInt8": ("da7a1ca730be70218b5bd7412c1a56012d2d90b0565259771fb2b472ed177bc1", ()),
    "color:line:Float64": ("a99882e93271521e4ae6dcba49a32b380db13b7080abbd789ecc641712dc9aab", ()),
    "color:line:Boolean": ("c2f7413f8d7e068ee3060c04e663f02ae37ecb2c6c5c548ff0b3cc9801bedead", ()),
    "color:area:Utf8": ("f8cc868654074439230729b4891652542d3089f617a2d677345e643f1ac486aa", ()),
    "color:area:Int64": ("7cd1e971d276e53e93efa82bc95cfce8acf9a4dd01ab7bae486be3d4ab914d8d", ()),
    "color:area:UInt8": ("7cd1e971d276e53e93efa82bc95cfce8acf9a4dd01ab7bae486be3d4ab914d8d", ()),
    "color:area:Float64": ("5b0f3ea32649de85cf5daf0e4e1eca6ef930c7ad6f336db9142e462b1adc440d", ()),
    "color:area:Boolean": ("6aa6e84a19ba5ec2cf5276d17d40b8a40122173a44e5cca1dbab7cfe389207b2", ()),
    "color:ribbon:Utf8": ("918a5cee6e6adc863f588cb7a7690db2574d4e36068a8a8a08f78cb68c305817", ()),
    "color:ribbon:Int64": ("9df678cc3cbdb7814e21fa3ca92b1c8d3c3431d46de9c23020500522a5296cc6", ()),
    "color:ribbon:UInt8": ("9df678cc3cbdb7814e21fa3ca92b1c8d3c3431d46de9c23020500522a5296cc6", ()),
    "color:ribbon:Float64": (
        "6f5a7e9721048fc82acb20950a966bb446f8a905cd223a82c181b9e9e7c462ff",
        (),
    ),
    "color:ribbon:Boolean": (
        "5a29def587c4b0aaa43e252debc727328dac82cdc015a375256e896adbc0d082",
        (),
    ),
    "color:polygon:Utf8": ("f52314a46f7a5f6c26d10bdc8cbd161107ea99c977ec4dcecf8dee7d7f018327", ()),
    "color:polygon:Int64": ("264dfa8f984db70dc5d1a3b949b57f410440362d9112cb1597be0f261f5a3d54", ()),
    "color:polygon:UInt8": ("264dfa8f984db70dc5d1a3b949b57f410440362d9112cb1597be0f261f5a3d54", ()),
    "color:polygon:Float64": (
        "1bd19db1b908824389b1c1b299c7bc9f0a4631f24206ac5e67e1e9e0ce47e378",
        (),
    ),
    "color:polygon:Boolean": (
        "8d6c9a0c8019f001734e3db8c677aa253b791c8fe6ec2c4956ad50a6268cfd2d",
        (),
    ),
    "color:point:Utf8": ("a6fa30ec6b6ffe2ddf345a1972afebc382ad00fd66b758b206572dc05e9a95af", ()),
    "color:point:Int64": ("b156382c217ead011ee2dbd4cc7993834cb06d3c892fb118a983e18e250eb31b", ()),
    "color:point:UInt8": ("b156382c217ead011ee2dbd4cc7993834cb06d3c892fb118a983e18e250eb31b", ()),
    "color:point:Float64": ("bca4afa4e8f0ba0c4e71bf5311952cd7e921cc7b3863b4389f391f0f265e640d", ()),
    "color:point:Boolean": ("e318b47a10507f62a6d9ab48c5249c93d2874763dde966be584170b261047830", ()),
    "color:bar:Utf8": ("2bcce9bf946846343f2b9143ee204b84f504564ce5fecd046dfa9418cb112293", ()),
    "color:bar:Int64": ("5d22a7a72cf153d3c7a8285216cae35a370f9ab2a4c9e3c66ee72ff63f237eaa", ()),
    "color:bar:UInt8": ("5d22a7a72cf153d3c7a8285216cae35a370f9ab2a4c9e3c66ee72ff63f237eaa", ()),
    "color:bar:Float64": ("ecd9e2e1c2bcc74f517875c75a6fe1401b96796db4e499ae39499330b07af001", ()),
    "color:bar:Boolean": ("baa9226c3a45bcc0614c5abb4406718b690dbe9cf6beebc0b5bb9e08fcc4dfb4", ()),
    "color:tick:Utf8": ("e5c1260e6d5d2a20c8dafe47b5d8b3068a09a240eb504ecea504ec1fc361fb0d", ()),
    "color:tick:Int64": ("d43e6e6d927c0d2d009e473048fb970f888d25f9ddd0257331c0286104d450d8", ()),
    "color:tick:UInt8": ("d43e6e6d927c0d2d009e473048fb970f888d25f9ddd0257331c0286104d450d8", ()),
    "color:tick:Float64": ("6e90c8d60f2a0b8dc28671cd8058e99bdd47b903732d50dac8c8d296601f2c84", ()),
    "color:tick:Boolean": ("926e6222ed651189394e4bdb125628fa53edb7e1dd26af6ed8d39d97e4125e69", ()),
    "color:rule:Utf8": ("4ed53a8c5c5ebf77ab5a97725a37f45ee31ba87f1348d8d95062bb18292a0f5f", ()),
    "color:rule:Int64": ("fa6c42218e9efda37ec798e630cfc4a0f91a9c165eee50094ad80318474a7d42", ()),
    "color:rule:UInt8": ("fa6c42218e9efda37ec798e630cfc4a0f91a9c165eee50094ad80318474a7d42", ()),
    "color:rule:Float64": ("458395258d316dae088b02456eaf51e1f99e3655a9142a1097667e8de57219b1", ()),
    "color:rule:Boolean": ("b6be2fd33af34deeafef90b201b5444442c82a092fdc56ce6b17fe568c674b77", ()),
    "color:text:Utf8": ("7a1888f0467b3824fde8f90cd4f9c758e9903034d91cc7a35e491af4d7ca8c4d", ()),
    "color:text:Int64": ("03bf243b6dcbf516ece5faf79b9f53d31476ceb664392f36779dfc1ee96aaa9f", ()),
    "color:text:UInt8": ("03bf243b6dcbf516ece5faf79b9f53d31476ceb664392f36779dfc1ee96aaa9f", ()),
    "color:text:Float64": ("ed108f1d16490ef1bf30357ad1ac2d1730d215704770fbdd9b7d972de02378d9", ()),
    "color:text:Boolean": ("refused: mark_text: column 'd' has unsupported dtype: Boolean", ()),
    "shape:point:Utf8": ("178058310b6a004cc4cc3123378e2e36f9cb8f624fc6c430167ab2be5959b137", ()),
    "shape:point:Int64": ("0f97e731f8f66d7700d5ed5b200a2d1dc42fa6d2b076f1237240ea8f09bc346d", ()),
    "shape:point:UInt8": ("0f97e731f8f66d7700d5ed5b200a2d1dc42fa6d2b076f1237240ea8f09bc346d", ()),
    "shape:point:Float64": ("f9989f2319b9d2809c4de70882eb3ffce648e53feeda3347797ae9190760116b", ()),
    "shape:point:Boolean": ("c425c1d8d8a29faa255faebbe49ca4c9674919ae7d28679e89699ae222b061f9", ()),
    "stroke_dash:line:Utf8": (
        "bf1ff84d4f19c70436aaac10d4f1388809c21d967bbd78201be37801cd39c63f",
        (),
    ),
    "stroke_dash:line:Int64": (
        "23c60e65f41570c13e3d9195af02af9491f608315aa151278e3edd8c162b0295",
        (),
    ),
    "stroke_dash:line:UInt8": (
        "23c60e65f41570c13e3d9195af02af9491f608315aa151278e3edd8c162b0295",
        (),
    ),
    "stroke_dash:line:Float64": (
        "1f97a3486fd4848d806a331b2aa03a547d7453e9ed196f16fe58b0f4616f241c",
        (),
    ),
    "stroke_dash:line:Boolean": (
        "22de93d6448d0fedfb7b16b1073e12b3cc0ceda9417e08e113860cb99ff904f2",
        (),
    ),
    "stroke_dash:ribbon:Utf8": (
        "59a76593f2203b37f418782364a89528d6994419da3c03ac96dfd45bd090a2a2",
        (),
    ),
    "stroke_dash:ribbon:Int64": (
        "4b5e8c1bd9e0f50c9ffd9f5439d53093fdc12d2caeff7eafb6bace22067b2fc6",
        (),
    ),
    "stroke_dash:ribbon:UInt8": (
        "4b5e8c1bd9e0f50c9ffd9f5439d53093fdc12d2caeff7eafb6bace22067b2fc6",
        (),
    ),
    "stroke_dash:ribbon:Float64": (
        "3e1b281caa351fb53ff40ef20af4344a1b547b94ba25011584323da289f45815",
        (),
    ),
    "stroke_dash:ribbon:Boolean": (
        "b9154801f405947026e5f18fa299e86a7ab81d7ebc86bca5dad37aadbb578851",
        (),
    ),
    "stroke_dash:point:Utf8": (
        "4b043e2a7f4966b9eba7be1338ca7db68d379dad44a1191a621fd6503f8a1e12",
        (),
    ),
    "stroke_dash:point:Int64": (
        "40506fa1e53485d98c3459a0033e450a7015cd9a1eff72a6d1753714841a81b6",
        (),
    ),
    "stroke_dash:point:UInt8": (
        "40506fa1e53485d98c3459a0033e450a7015cd9a1eff72a6d1753714841a81b6",
        (),
    ),
    "stroke_dash:point:Float64": (
        "31eb2c6feb4cd7a10c3fd5a6fd27de89a61cd868183458c0e78eeb0e97fd7f30",
        (),
    ),
    "stroke_dash:point:Boolean": (
        "2d04bff753ef49065faaebcfe3f98b0a5193169c9d8748f97ff757048680f390",
        (),
    ),
    "stroke_dash:bar:Utf8": (
        "8c9f7208871828ada9f4aed70b1beb14c3c33e2b6f3b07a8ff746186e905758a",
        (),
    ),
    "stroke_dash:bar:Int64": (
        "3ade726b85645061bbdd537d6f8a93037aba71c398a0ed49f1d95b9b65e96ebd",
        (),
    ),
    "stroke_dash:bar:UInt8": (
        "3ade726b85645061bbdd537d6f8a93037aba71c398a0ed49f1d95b9b65e96ebd",
        (),
    ),
    "stroke_dash:bar:Float64": (
        "08f16e8a9b11b86d5549c8f330d8fb1acaeff41bb0c394466e07e020c57ba5bf",
        (),
    ),
    "stroke_dash:bar:Boolean": (
        "945a148b0210dbcc1e874ba0dc0fa1bbba55c87183abea9d27c998a68c32bb76",
        (),
    ),
    "stroke_dash:rule:Utf8": (
        "16c3a2ed6e2321bab541ef7277ff465cb7d1eed43516ebc00448817ebfd66c54",
        (),
    ),
    "stroke_dash:rule:Int64": (
        "a4251130a17a28a270c03ffe8c4939c54dd7c6d13dc1c18f89cbbd4389c0d348",
        (),
    ),
    "stroke_dash:rule:UInt8": (
        "a4251130a17a28a270c03ffe8c4939c54dd7c6d13dc1c18f89cbbd4389c0d348",
        (),
    ),
    "stroke_dash:rule:Float64": (
        "9c6bea3c5da8fa2eac35e5453ccafd7fab91251a1dcce634fada625026d6aec0",
        (),
    ),
    "stroke_dash:rule:Boolean": (
        "ba5dbe774d03b42f7aac7f247c9ee34bde437f37967d4d90426d1432af3f466a",
        (),
    ),
    "stroke_dash:rect:Utf8": (
        "aef92e0c17d2965baebd13d331249e3ff592028fc05ab0b47efcad1f5fa16aeb",
        (),
    ),
    "stroke_dash:rect:Int64": (
        "5fdcdd2e993b49d22747a9806b293b244080ed1946f5364c5c8684e745b0f53a",
        (),
    ),
    "stroke_dash:rect:UInt8": (
        "5fdcdd2e993b49d22747a9806b293b244080ed1946f5364c5c8684e745b0f53a",
        (),
    ),
    "stroke_dash:rect:Float64": (
        "ba5effde3ced4006e665094b972e40edb9077a28380b5546a25ca8c2eb696788",
        (),
    ),
    "stroke_dash:rect:Boolean": (
        "cc681c8a207796355ba2a7483b9c6a2205b2426a30bf6323d265096690f599a8",
        (),
    ),
}
_SCENE_DIGESTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "detail:line:Utf8": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:Categorical": (
        "9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99",
        (),
    ),
    "detail:line:Enum": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:Int64": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:Int8": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:UInt8": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:UInt32": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:Float64": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:Float32": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:line:Boolean": ("9ec7b8ba3a47e1a228a5f58a69549a46adfd03850a91b67401b727b969219a99", ()),
    "detail:area:Utf8": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:Categorical": (
        "751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8",
        (),
    ),
    "detail:area:Enum": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:Int64": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:Int8": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:UInt8": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:UInt32": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:Float64": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:Float32": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:area:Boolean": ("751b9539cfd2c43da1c2da773de1bede821c1b9fb0c02c2b425779a75efb01c8", ()),
    "detail:ribbon:Utf8": ("9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f", ()),
    "detail:ribbon:Categorical": (
        "9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f",
        (),
    ),
    "detail:ribbon:Enum": ("9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f", ()),
    "detail:ribbon:Int64": ("9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f", ()),
    "detail:ribbon:Int8": ("9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f", ()),
    "detail:ribbon:UInt8": ("9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f", ()),
    "detail:ribbon:UInt32": (
        "9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f",
        (),
    ),
    "detail:ribbon:Float64": (
        "9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f",
        (),
    ),
    "detail:ribbon:Float32": (
        "9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f",
        (),
    ),
    "detail:ribbon:Boolean": (
        "9379a57450a0061bf24a876234bcf343745d736484d53378ae2ec4c6e5e55e4f",
        (),
    ),
    "detail:polygon:Utf8": ("635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76", ()),
    "detail:polygon:Categorical": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "detail:polygon:Enum": ("635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76", ()),
    "detail:polygon:Int64": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "detail:polygon:Int8": ("635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76", ()),
    "detail:polygon:UInt8": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "detail:polygon:UInt32": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "detail:polygon:Float64": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "detail:polygon:Float32": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "detail:polygon:Boolean": (
        "635d620e1d7fda987b8d2af57136507247056cff5580a196b33264ef54958b76",
        (),
    ),
    "color:line:Utf8": ("e05abbe94557e64740ffd6df8331c9059c8b24e773e9774806f1ff46e7e6ecd4", ()),
    "color:line:Int64": ("098259d059aef7a60b2076f062ba923506ee0b754f05725561ab2f0cab6caaeb", ()),
    "color:line:UInt8": ("098259d059aef7a60b2076f062ba923506ee0b754f05725561ab2f0cab6caaeb", ()),
    "color:line:Float64": ("132d693409d5e593194b12e6c54085294a8cf8a3b2cdc46c2a5291bd2b40049a", ()),
    "color:line:Boolean": ("232ac15d21b4d78c236731eab7b3acf12fde6b4729bdafa2058d0df4df91cef7", ()),
    "color:area:Utf8": ("3c2b94309e89f3b881c1cd4795a23790e19eb3893f9d58ad9e40b23e57dc329e", ()),
    "color:area:Int64": ("7909b3f863e986540fcaba0bb2180a5c80b15977e213ded7112cae817c3e17b0", ()),
    "color:area:UInt8": ("7909b3f863e986540fcaba0bb2180a5c80b15977e213ded7112cae817c3e17b0", ()),
    "color:area:Float64": ("d3c6374d99265519f834e70b2b406fcf897a275f864a85f258e349ed625a7f52", ()),
    "color:area:Boolean": ("cd202949f50b7528cd2beeefe55f936dda8f61047bc827f22497b307c12c546a", ()),
    "color:ribbon:Utf8": ("1cdfef9e10d09e80857a7f924c1a22f281a2d810f90f5d6ec19493ae1e1139af", ()),
    "color:ribbon:Int64": ("088c6ebd814a0c9479190a497b09543c62029dad37fd6a4fa25791cf2aa77893", ()),
    "color:ribbon:UInt8": ("088c6ebd814a0c9479190a497b09543c62029dad37fd6a4fa25791cf2aa77893", ()),
    "color:ribbon:Float64": (
        "5f0ce6710f5517294837fde1d61a2eac2c66cf11b350e448b24b2c925f91bdd7",
        (),
    ),
    "color:ribbon:Boolean": (
        "067b9f06464228209c88ab44da8c052c6dc27fd8f00a3f090b8281fb7e5c6099",
        (),
    ),
    "color:polygon:Utf8": ("7c35c4bef6d65d7fa198e0059c686fdb7e56c364013b58f43bcc812f596d84a5", ()),
    "color:polygon:Int64": ("8081aaad2f445761696db0cdc0736d3ca61921e135f79a417f9b01f1600cf61e", ()),
    "color:polygon:UInt8": ("8081aaad2f445761696db0cdc0736d3ca61921e135f79a417f9b01f1600cf61e", ()),
    "color:polygon:Float64": (
        "87b7cceb7e08144586530a8d1642d9c9f8a1c0b3f81d611e5631985a5eab1e20",
        (),
    ),
    "color:polygon:Boolean": (
        "19e6618eb425bf89c19c0cd59b6edfdbd03a78d4f89f21c1080233ddb65d3352",
        (),
    ),
    "color:point:Utf8": ("1ed66421226f943b1dd6055d75b1d0ab5b841eb510eebb5a7971ac6023ac9113", ()),
    "color:point:Int64": ("4aa7c2b6e32cb41623835a44e2e2dcac55a1491d8e496cd8de6e9f0c20f7dda3", ()),
    "color:point:UInt8": ("4aa7c2b6e32cb41623835a44e2e2dcac55a1491d8e496cd8de6e9f0c20f7dda3", ()),
    "color:point:Float64": ("59b72ec6fed83d899bb9c115fe2433b427f924f86a386597229f517af3806234", ()),
    "color:point:Boolean": ("6d5f82ac436b4073528ea3a4a9cffd0c8c0204845252382fc7661111f25670a9", ()),
    "color:bar:Utf8": ("70c4082e7cebb3f78d50376e1c4776c0dc24a216e9e822b56289e1b5c947b57d", ()),
    "color:bar:Int64": ("c230f4b652b3f0d38d3317089ca3fadeabd35dc991453cda81ecd2a8125abb23", ()),
    "color:bar:UInt8": ("c230f4b652b3f0d38d3317089ca3fadeabd35dc991453cda81ecd2a8125abb23", ()),
    "color:bar:Float64": ("44fa8176411c02649580aa74ae0bb71ee7b2a8e7de677bad1a9e5a0f9f54e0d3", ()),
    "color:bar:Boolean": ("ec3f76fa078bb15c3b95ca0475044fcf6ad9f2cf059a22fc479722fc08e3a55c", ()),
    "color:tick:Utf8": ("2922f7025f72314192b6e360f007aec09eb8b4ae59d9b094707a5488888cde2a", ()),
    "color:tick:Int64": ("240aa6437f9eb06cdb06c1c5789a2e8d377746e635c9e12de3fa42d6deb87354", ()),
    "color:tick:UInt8": ("240aa6437f9eb06cdb06c1c5789a2e8d377746e635c9e12de3fa42d6deb87354", ()),
    "color:tick:Float64": ("905123ccf8753c4211bb129573a23cc880289a0173acd2025f0c1ab03dcf895a", ()),
    "color:tick:Boolean": ("4f2105c7e5579227c25dd3436e1b7c7d0ea42471c5af2cbf98e8b3a80227c7c4", ()),
    "color:rule:Utf8": ("35a574eef10f2ddb033221df733f98aad49cce14f19fb24de372f5bb2b6cf610", ()),
    "color:rule:Int64": ("42a772272214c4049dfa14f62c63209d8b5ccb2d4741950557f728d0a9b652d7", ()),
    "color:rule:UInt8": ("42a772272214c4049dfa14f62c63209d8b5ccb2d4741950557f728d0a9b652d7", ()),
    "color:rule:Float64": ("3b41a679042c29216e4ec9973f80a9e0f1f95af83df9b26bb5b7aae8c465d18f", ()),
    "color:rule:Boolean": ("81db2f527a2410ec40a82f7c5f498314ffa44a356ddec06932fee7af380a45d3", ()),
    "color:text:Utf8": ("3c5c7c26a3872715f33cdff2b420bd4db67123030bd888bdcbb85a30f1f40ebd", ()),
    "color:text:Int64": ("6edc9f3fae1c302ae8c57947bb287a5df4180fd44a263c258bd891db76fdecf6", ()),
    "color:text:UInt8": ("6edc9f3fae1c302ae8c57947bb287a5df4180fd44a263c258bd891db76fdecf6", ()),
    "color:text:Float64": ("3d687e311f5319c3241cd3b69c004aae3b887df979d6b6f1f540725ca5ae575f", ()),
    "color:text:Boolean": ("refused: mark_text: column 'd' has unsupported dtype: Boolean", ()),
    "shape:point:Utf8": ("6fdffdeba7ee009803d5b6294db506630925302ad9ab99dd4d9abc1d34f37ce7", ()),
    "shape:point:Int64": ("14851dab34ac7007431e2841e15a9cb1d5bdfb97eabf394d4984b05e433d75b5", ()),
    "shape:point:UInt8": ("14851dab34ac7007431e2841e15a9cb1d5bdfb97eabf394d4984b05e433d75b5", ()),
    "shape:point:Float64": ("22ec0819c753f781a3c629438892950138b62d0ddb4169a13f945c44d6d09f5a", ()),
    "shape:point:Boolean": ("06f7dc985eb2a0c004b0ed3f35138725261b31d2522fcd5f0ebda2291f29df94", ()),
    "stroke_dash:line:Utf8": (
        "12c870a0a77967e837432cc3ed489fca7c715e55ceac3e030e1950fe48970ead",
        (),
    ),
    "stroke_dash:line:Int64": (
        "938aaa7138634538edf1ece33c8e4cb7faa260d8d3af4ddd6abf52f7ff661bef",
        (),
    ),
    "stroke_dash:line:UInt8": (
        "938aaa7138634538edf1ece33c8e4cb7faa260d8d3af4ddd6abf52f7ff661bef",
        (),
    ),
    "stroke_dash:line:Float64": (
        "b1917a4e2656b06928b65979674f11cc2a6accea5fe2c5ac9399bd5336059f52",
        (),
    ),
    "stroke_dash:line:Boolean": (
        "335752e3446d35e03392df0b172c8b40e50792e06b4c03a8a15bbb178d2ebdfc",
        (),
    ),
    "stroke_dash:ribbon:Utf8": (
        "363e97f9a385e6b86ff464305fe393c3f67f508ace5c004f57ecb5b935e53c29",
        (),
    ),
    "stroke_dash:ribbon:Int64": (
        "d3d4bec392101a5e119f0c97373adad2cf071bb232cd157fe841b8f16bfc6007",
        (),
    ),
    "stroke_dash:ribbon:UInt8": (
        "d3d4bec392101a5e119f0c97373adad2cf071bb232cd157fe841b8f16bfc6007",
        (),
    ),
    "stroke_dash:ribbon:Float64": (
        "d6aac6e3b8621e85f25a031feec744e0312b6b98b5c5c1f14fda82286458a139",
        (),
    ),
    "stroke_dash:ribbon:Boolean": (
        "b41ecd6756c66a1f6bca16f59a819e8b8ca149ab217c6e2b3ea7a5b4115e66ab",
        (),
    ),
    "stroke_dash:point:Utf8": (
        "851cf1b6f7780f453230a6128aa5d1cea148a91c61932c1a127a22b68279275c",
        (),
    ),
    "stroke_dash:point:Int64": (
        "1a4052fee1221d7c785187528b183ba4174628d62fe2b76736c6e6229a35d81e",
        (),
    ),
    "stroke_dash:point:UInt8": (
        "1a4052fee1221d7c785187528b183ba4174628d62fe2b76736c6e6229a35d81e",
        (),
    ),
    "stroke_dash:point:Float64": (
        "071defe39036844427c88a2d2b79f47357de67c88fcb8649d265180828e05292",
        (),
    ),
    "stroke_dash:point:Boolean": (
        "faee50e3aff05d1ee91ddcca6ad6769b86f2690c2ca2242a8f95e51e9a6ead56",
        (),
    ),
    "stroke_dash:bar:Utf8": (
        "ba38a80fa52d067897c044632339e6c2d040b054225dbb6c1d2e4cee0550b8a3",
        (),
    ),
    "stroke_dash:bar:Int64": (
        "e69fc1d9b99d5e039180e00763126c42d1a32e2064e1a8af74d7f5f799679066",
        (),
    ),
    "stroke_dash:bar:UInt8": (
        "e69fc1d9b99d5e039180e00763126c42d1a32e2064e1a8af74d7f5f799679066",
        (),
    ),
    "stroke_dash:bar:Float64": (
        "25d7beceb51879ad2c5c24e5a4cde14fcabbe65cd28717195bd71fd6eddd294d",
        (),
    ),
    "stroke_dash:bar:Boolean": (
        "bb312334cd1a2a27f2e4ec3f50091839c03acffe918a63caa76fec50b2ae6f15",
        (),
    ),
    "stroke_dash:rule:Utf8": (
        "cef4c91272c991c7c9c7a061de8e05a4678d2ef0865a284140f852793b3e337b",
        (),
    ),
    "stroke_dash:rule:Int64": (
        "57e38da1fe94ae7f30e951b1e25f3c3bb62a133e8c9c2ebaf17fe51be2ec72e6",
        (),
    ),
    "stroke_dash:rule:UInt8": (
        "57e38da1fe94ae7f30e951b1e25f3c3bb62a133e8c9c2ebaf17fe51be2ec72e6",
        (),
    ),
    "stroke_dash:rule:Float64": (
        "907af0e523220e75cdce0bc62d96af29f611d34d35f07952d86966ac5090ea24",
        (),
    ),
    "stroke_dash:rule:Boolean": (
        "31331518ba5c84bc6356df1c7a2269212aed457284c4405337dc5ed778f97372",
        (),
    ),
    "stroke_dash:rect:Utf8": (
        "61073625b1dbb333f57584d1bb564a63a2d3878697088ef07263b105d2137708",
        (),
    ),
    "stroke_dash:rect:Int64": (
        "ffeafe2cd98803ba70479c8fede7772cdf57503f1d35205625606777dc36dcd8",
        (),
    ),
    "stroke_dash:rect:UInt8": (
        "ffeafe2cd98803ba70479c8fede7772cdf57503f1d35205625606777dc36dcd8",
        (),
    ),
    "stroke_dash:rect:Float64": (
        "0e2528feadf73e004957fa20a7a6773c3bf7991c9045d1baac41c7de8c15e4a8",
        (),
    ),
    "stroke_dash:rect:Boolean": (
        "2a2065e00462cdd98668d330cd9e44a7650895d92313a6e285343eb8fcdd2b5f",
        (),
    ),
}


@pytest.mark.parametrize("case", _unchanged_cases())
def test_a_partition_key_of_another_dtype_draws_as_it_did(case: str):
    assert _outcome(case, "static") == _STATIC_DIGESTS[case]


@pytest.mark.parametrize("case", _unchanged_cases())
def test_a_partition_key_of_another_dtype_builds_the_scene_it_did(case: str):
    assert _outcome(case, "interactive") == _SCENE_DIGESTS[case]


_DATE_AXIS_DIGESTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "line": ("1fbf92c2cd42e56c5d1790e4ea81f2b079fdf7c44ec98050e20beead313aa5d2", ()),
    "area": ("254f8c13748c03cd74ae5c8bca78f6e299df449ad46e6daf1f9f78b3f75c6c35", ()),
    "ribbon": ("be4aebdee1a2ef270ca98559f7fbb664395b6bd660612e0a3a86957aede3628e", ()),
}


@pytest.mark.parametrize("mark", ["line", "area", "ribbon"])
def test_a_chart_with_a_date_axis_and_no_temporal_key_draws_as_it_did(mark: str):
    """The date is the x position, drawn on the axis: the partition is text."""
    frame = _frame(_PLAIN_KEYS["Utf8"]).with_columns(
        pl.col("t")
        .cast(pl.Int64)
        .mul(_DAY)
        .add(_MIDNIGHTS[0])
        .cast(pl.Datetime("ms"))
        .cast(pl.Date)
    )
    svg, messages = _render_capturing(_by_detail(mark, frame).to_svg)
    assert (hashlib.sha256(svg.encode()).hexdigest(), tuple(messages)) == _DATE_AXIS_DIGESTS[mark]
