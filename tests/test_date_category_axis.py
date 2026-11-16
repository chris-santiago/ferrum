"""A Date or Datetime column read as categories draws one category per distinct value.

A ``Date`` or ``Datetime`` column drawn where a category axis is wanted (a box, errorbar, violin,
boxen or catplot category, ``x='k:O'``/``x='k:N'``, a ``:N`` colour, a shape, a stroke dash, a
facet) draws one category per distinct instant, in time order, labelled per value: a value at
midnight UTC reads its ISO date (``2020-01-01``), any other its full ISO date-time
(``2020-01-01T12:00:00``, with the fraction when it has one).

Each expected output is stated against a twin that needs no date reading: the same chart over the
labels as text, its rows in time order. The twin's band order, tick text and mark positions are
the ones a date chart must draw. Positions are also read against the axis's own ticks: a mark's
pixel is compared with the pixel of the tick that names it, and its value is read back through the
value axis's own ladder, so a label that sits on the wrong band fails.

A column that never reads as categories keeps its meaning: a temporal ``x``, a continuous colour
and a statistic's own key draw as they always did (pinned by digests measured on the v0.25.0
build). Layers over one date category axis share it, in time order, whatever frame they come
from. A Duration (which crosses as a timestamp) is refused on every binding that reads
categories, naming the caller's own column; distinct date-times within one millisecond (they reach
the renderer at millisecond precision) draw as one category and warn once, naming the counts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import ordered_text_entries, value_axis_ladder, value_for_pixel
from tests._svg_rects import data_bar_rects, mark_layer, svg_elements
from tests.test_null_group_warning import _render_capturing

# ---------------------------------------------------------------------------
# Frames and the label rule
# ---------------------------------------------------------------------------

_UTC = dt.timezone.utc
_MIDNIGHTS = [dt.datetime(2020, 1, d) for d in (1, 2, 3)]
# The rows list the dates out of time order, so first-appearance order and time order differ.
_UNSORTED = [_MIDNIGHTS[2], _MIDNIGHTS[0], _MIDNIGHTS[1]]
_MIXED = [
    dt.datetime(2020, 1, 1),
    dt.datetime(2020, 1, 1, 12),
    dt.datetime(2020, 1, 2),
]
_FRACTION = [dt.datetime(2020, 1, 1, 12, 0, 0, 250_000), dt.datetime(2020, 1, 2)]
_REPEATS = 5


def _label(value: dt.datetime) -> str:
    """The category label of an instant: its ISO date at midnight, else its full ISO date-time."""
    if value.time() == dt.time(0):
        return value.date().isoformat()
    if value.microsecond == 0:
        return value.isoformat(timespec="seconds")
    return value.isoformat(timespec="milliseconds")


def _labels(instants: list[dt.datetime]) -> list[str]:
    """The labels of the distinct instants, in time order."""
    return [_label(v) for v in sorted(set(instants))]


def _rows(instants: list[dt.datetime]) -> pl.DataFrame:
    """Each instant repeated, a value per row that differs by instant and by repeat: the values
    of one instant lie in a band of their own (instant 0 in 1..5, instant 1 in 11..15, ...), so
    a mark's height says which instant's rows it was drawn from."""
    rows = [
        {"k": instant, "v": 10.0 * rank + r + 1.0, "w": float(r)}
        for rank, instant in enumerate(instants)
        for r in range(_REPEATS)
    ]
    return pl.DataFrame(rows)


def _by_time(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort("k", maintain_order=True)


def _as_date(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("k").cast(pl.Date))


def _as_text(frame: pl.DataFrame) -> pl.DataFrame:
    """The twin: the column's labels as text, the rows as they are."""
    labels = [_label(v) for v in frame["k"].to_list()]
    return frame.with_columns(k=pl.Series(labels, dtype=pl.Utf8))


def _as_unit(unit: str, zone: str | None = None) -> Callable[[pl.DataFrame], pl.DataFrame]:
    def build(frame: pl.DataFrame) -> pl.DataFrame:
        column = pl.col("k").cast(pl.Datetime(unit))
        if zone is not None:
            column = column.dt.replace_time_zone("UTC").dt.convert_time_zone(zone)
        return frame.with_columns(column)

    return build


# Every dtype a midnight date can arrive as: the same instants, so the same three categories.
_KINDS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "Date": _as_date,
    "Datetime-ms": _as_unit("ms"),
    "Datetime-us": _as_unit("us"),
    "Datetime-ns": _as_unit("ns"),
    "Datetime-UTC": _as_unit("ms", "UTC"),
    "Datetime-Paris": _as_unit("us", "Europe/Paris"),
    "Datetime-Tokyo-ns": _as_unit("ns", "Asia/Tokyo"),
}

# ---------------------------------------------------------------------------
# Readers of a rendered chart
# ---------------------------------------------------------------------------


def _x_ticks(svg: str, expected: list[str]) -> list[tuple[float, str]]:
    """The x-axis tick row as ``(x, text)``, left to right: the entries of the bottom-most row
    that holds the expected labels."""
    entries = [(x, y, t) for x, y, t in ordered_text_entries(svg) if t in set(expected)]
    assert entries, f"no tick of {expected} was drawn"
    row = max(y for _, y, _ in entries)
    return sorted((x, t) for x, y, t in entries if abs(y - row) < 1)


def _tick_labels(svg: str, expected: list[str]) -> list[str]:
    return [t for _, t in _x_ticks(svg, expected)]


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _element_xs(svg: str) -> list[tuple[float, float]]:
    """The x extent ``(lo, hi)`` of every mark element in the plot area: a rect, a line, a circle,
    a path."""
    layer = mark_layer(svg)
    extents: list[tuple[float, float]] = []
    for a in svg_elements(layer, "rect", lambda a: True):
        extents.append((float(a["x"]), float(a["x"]) + float(a["width"])))
    for a in svg_elements(layer, "line", lambda a: True):
        extents.append((min(float(a["x1"]), float(a["x2"])), max(float(a["x1"]), float(a["x2"]))))
    for a in svg_elements(layer, "circle", lambda a: True):
        extents.append((float(a["cx"]), float(a["cx"])))
    for a in svg_elements(layer, "path", lambda a: "d" in a):
        xs = [float(n) for n in _NUMBER.findall(a["d"])[0::2]]
        extents.append((min(xs), max(xs)))
    return extents


def _centres(svg: str) -> list[float]:
    """The distinct band centres the marks sit on, left to right."""
    return sorted({round((lo + hi) / 2, 1) for lo, hi in _element_xs(svg)})


def _marks_sit_on_the_ticks(svg: str, labels: list[str]) -> None:
    ticks = _x_ticks(svg, labels)
    assert [t for _, t in ticks] == labels
    centres = _centres(svg)
    assert len(centres) == len(labels), centres
    for (x, text), centre in zip(ticks, centres, strict=True):
        assert abs(x - centre) < 0.6, f"the mark for {text} sits at {centre}, its tick at {x}"


def _whiskers(svg: str) -> list[tuple[float, float, float]]:
    """The vertical whisker lines of a box layer as ``(x, low value, high value)``, the values
    read back through the chart's own value axis."""
    ladder = value_axis_ladder(svg, axis="y")
    lines = svg_elements(
        mark_layer(svg),
        "line",
        lambda a: a["x1"] == a["x2"] and a.get("stroke-width") == "1.5",
    )
    out = []
    for a in lines:
        ys = sorted((float(a["y1"]), float(a["y2"])))
        hi, lo = value_for_pixel(ladder, ys[0]), value_for_pixel(ladder, ys[1])
        out.append((float(a["x1"]), lo, hi))
    return sorted(out)


def _scene(chart: Any) -> dict:
    return json.loads(chart.interactive().scene_json)


def _scene_texts(node: Any) -> list[str]:
    """Every text content in a scene, in document order."""
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "text" and isinstance(node.get("content"), str):
            found.append(node["content"])
        for child in node.values():
            found.extend(_scene_texts(child))
    elif isinstance(node, list):
        for child in node:
            found.extend(_scene_texts(child))
    return found


def _interactive_ticks(chart: Any, labels: list[str]) -> list[str]:
    """The labels' occurrences in the interactive scene's text, in order."""
    texts = _scene_texts(_scene(chart))
    return [t for t in texts if t in set(labels)]


_OUTPUTS = ["static", "interactive"]


def _scene_without_tooltips(chart: Any) -> dict:
    """The interactive scene of *chart* without its tooltips: a tooltip reads the date's epoch
    text (a timestamp's tooltip value is the one reading the label rule does not reach)."""
    scene = _scene(chart)
    for panel in scene.get("panels", []):
        for batch in panel.get("marks", []):
            batch.pop("tooltips", None)
    return scene


def _drawn(chart: Any, output: str = "static") -> tuple[Any, list[str]]:
    """What *chart* draws (the svg, or its scene without tooltips) and every warning it raised,
    in order."""
    if output == "static":
        return _render_capturing(chart.to_svg, action="always")
    return _render_capturing(lambda: _scene_without_tooltips(chart), action="always")


def _assert_twin(chart: Any, twin: Any, output: str = "static") -> tuple[Any, list[str]]:
    """*chart* draws what *twin* draws and raises the same ordered warnings."""
    drawn, messages = _drawn(chart, output)
    twin_drawn, twin_messages = _drawn(twin, output)
    assert drawn == twin_drawn
    assert messages == twin_messages
    return drawn, messages


# ---------------------------------------------------------------------------
# One band per date, in time order, on the four band marks
# ---------------------------------------------------------------------------

_BAND_MARKS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "boxplot": lambda df: fm.Chart(df).mark_boxplot().encode(x="k", y="v"),
    "errorbar": lambda df: fm.Chart(df).mark_errorbar().encode(x="k", y="v"),
    "violin": lambda df: fm.Chart(df).mark_violin().encode(x="k", y="v"),
    "boxen": lambda df: fm.Chart(df).mark_boxen().encode(x="k", y="v"),
}


@pytest.mark.parametrize("mark", list(_BAND_MARKS))
def test_a_band_mark_over_unsorted_dates_draws_a_band_per_date_in_time_order(mark: str):
    svg = _BAND_MARKS[mark](_as_date(_rows(_UNSORTED))).to_svg()
    _marks_sit_on_the_ticks(svg, _labels(_UNSORTED))


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("mark", list(_BAND_MARKS))
def test_a_date_category_mark_draws_as_its_label_text_twin(mark: str, output: str):
    rows = _by_time(_rows(_UNSORTED))
    _, messages = _assert_twin(
        _BAND_MARKS[mark](_as_date(rows)), _BAND_MARKS[mark](_as_text(rows)), output
    )
    assert messages == []


def test_the_box_over_each_date_spans_that_dates_own_rows():
    """The date at the third band holds the third rank's values (21..25), read back through the
    value axis: a label placed on the wrong band, or bands in encounter order, moves a box."""
    svg = fm.Chart(_as_date(_rows(_UNSORTED))).mark_boxplot().encode(x="k", y="v").to_svg()
    boxes = _whiskers(svg)
    assert len(boxes) == len(_UNSORTED)
    # rank is the row order of _UNSORTED: 2020-01-03 holds 1..5, 2020-01-01 holds 11..15,
    # 2020-01-02 holds 21..25; the bands read left to right by date.
    by_date = {_label(v): rank for rank, v in enumerate(_UNSORTED)}
    for (x, low, high), text in zip(boxes, _labels(_UNSORTED), strict=True):
        rank = by_date[text]
        assert low == pytest.approx(10.0 * rank + 1.0, abs=0.3), (text, x)
        assert high == pytest.approx(10.0 * rank + _REPEATS, abs=0.3), (text, x)
    ticks = _x_ticks(svg, _labels(_UNSORTED))
    for (tick, _), (x, _, _) in zip(ticks, boxes, strict=True):
        assert abs(tick - x) < 0.6


@pytest.mark.parametrize("mark", list(_BAND_MARKS))
@pytest.mark.parametrize("kind", list(_KINDS))
def test_every_dtype_of_a_midnight_reads_as_the_date_does(mark: str, kind: str):
    rows = _by_time(_rows(_UNSORTED))
    svg, _ = _assert_twin(_BAND_MARKS[mark](_KINDS[kind](rows)), _BAND_MARKS[mark](_as_date(rows)))
    assert _tick_labels(svg, _labels(_UNSORTED)) == _labels(_UNSORTED)


@pytest.mark.parametrize("kind", ["Datetime-UTC", "Datetime-Paris", "Datetime-Tokyo-ns"])
def test_a_zoned_midnight_is_labelled_in_utc(kind: str):
    """The same instants in another zone are the same categories, named by their UTC reading."""
    svg = _BAND_MARKS["boxplot"](_KINDS[kind](_by_time(_rows(_UNSORTED)))).to_svg()
    assert _tick_labels(svg, _labels(_UNSORTED)) == ["2020-01-01", "2020-01-02", "2020-01-03"]


# ---------------------------------------------------------------------------
# The label rule, per value
# ---------------------------------------------------------------------------


def test_a_datetime_with_times_of_day_is_labelled_per_value():
    labels = _labels(_MIXED)
    assert labels == ["2020-01-01", "2020-01-01T12:00:00", "2020-01-02"]
    svg = _BAND_MARKS["boxplot"](_rows(_MIXED)).to_svg()
    _marks_sit_on_the_ticks(svg, labels)


def test_a_datetime_column_of_times_of_day_draws_what_its_text_twin_draws():
    rows = _by_time(_rows(_MIXED))
    _, messages = _assert_twin(_BAND_MARKS["boxplot"](rows), _BAND_MARKS["boxplot"](_as_text(rows)))
    assert messages == []


def test_a_value_with_a_fraction_is_labelled_with_it():
    labels = _labels(_FRACTION)
    assert labels == ["2020-01-01T12:00:00.250", "2020-01-02"]
    svg = _BAND_MARKS["boxplot"](_rows(_FRACTION)).to_svg()
    _marks_sit_on_the_ticks(svg, labels)


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_a_fraction_reads_the_same_at_every_unit(unit: str):
    svg = _BAND_MARKS["boxplot"](_as_unit(unit)(_rows(_FRACTION))).to_svg()
    assert _tick_labels(svg, _labels(_FRACTION)) == _labels(_FRACTION)


def test_an_outlier_layer_of_midnights_lands_on_its_own_dates_band():
    """A box layer over every row of a column mixing midnights and other times, and a point layer
    whose rows are all midnight: one instant has one label, so each point meets its band."""
    boxes = _rows(_MIXED)
    outliers = _as_unit("ms")(pl.DataFrame({"k": [_MIXED[0], _MIXED[2]], "v": [99.0, 98.0]}))
    chart = fm.Chart(boxes).mark_boxplot().encode(x="k", y="v") + fm.Chart(
        outliers
    ).mark_point().encode(x=fm.X("k", type_="N"), y="v")
    svg = chart.to_svg()
    labels = _labels(_MIXED)
    ticks = dict((t, x) for x, t in _x_ticks(svg, labels))
    assert list(ticks) == labels or sorted(ticks, key=ticks.get) == labels
    circles = svg_elements(mark_layer(svg), "circle", lambda a: True)
    xs = sorted(float(c["cx"]) for c in circles)
    assert len(xs) == 2
    assert xs[0] == pytest.approx(ticks["2020-01-01"], abs=0.6)
    assert xs[1] == pytest.approx(ticks["2020-01-02"], abs=0.6)


# ---------------------------------------------------------------------------
# Every figure and channel that reads a category
# ---------------------------------------------------------------------------

_CATPLOT_KINDS = ["box", "bar", "point", "violin", "strip", "swarm", "boxen", "count"]


def _catplot(kind: str, df: pl.DataFrame) -> fm.Chart:
    if kind == "count":
        return fm.catplot(df, x="k", kind=kind)
    return fm.catplot(df, x="k", y="v", kind=kind, seed=1)


@pytest.mark.parametrize("kind", _CATPLOT_KINDS)
def test_a_catplot_over_dates_draws_a_category_axis_of_dates(kind: str):
    rows = _by_time(_rows(_UNSORTED))
    svg, _ = _assert_twin(_catplot(kind, _as_date(rows)), _catplot(kind, _as_text(rows)))
    labels = _labels(_UNSORTED)
    assert _tick_labels(svg, labels) == labels


@pytest.mark.parametrize("kind", _CATPLOT_KINDS)
def test_a_catplot_over_unsorted_date_rows_draws_the_dates_in_time_order(kind: str):
    svg = _catplot(kind, _as_date(_rows(_UNSORTED))).to_svg()
    labels = _labels(_UNSORTED)
    assert _tick_labels(svg, labels) == labels


@pytest.mark.parametrize("channel", ["k:O", "k:N"])
def test_a_bar_declared_ordinal_or_nominal_draws_a_bar_per_date(channel: str):
    """A bar is as tall as its date's own mean, read back through the value axis."""
    frame = _as_date(_rows(_UNSORTED))
    svg = fm.Chart(frame).mark_bar().encode(x=channel, y="mean(v)").to_svg()
    labels = _labels(_UNSORTED)
    ticks = _x_ticks(svg, labels)
    assert [t for _, t in ticks] == labels
    ladder = value_axis_ladder(svg, axis="y")
    bars = sorted(
        (float(r["x"]) + float(r["width"]) / 2, value_for_pixel(ladder, float(r["y"])))
        for r in data_bar_rects(mark_layer(svg))
    )
    by_date = {_label(v): rank for rank, v in enumerate(_UNSORTED)}
    assert len(bars) == len(labels)
    for (tick, text), (centre, top) in zip(ticks, bars, strict=True):
        assert abs(tick - centre) < 0.6
        assert top == pytest.approx(10.0 * by_date[text] + 3.0, abs=0.3), text


@pytest.mark.parametrize("channel", ["k:O", "k:N"])
def test_a_bar_declared_categorical_draws_what_its_text_twin_draws(channel: str):
    rows = _by_time(_rows(_UNSORTED))
    _, messages = _assert_twin(
        fm.Chart(_as_date(rows)).mark_bar().encode(x=channel, y="v"),
        fm.Chart(_as_text(rows)).mark_bar().encode(x="k", y="v"),
    )
    assert messages == []


_LEGEND_CHARTS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "color": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", color="k:N"),
    "shape": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", shape="k"),
    "stroke_dash": lambda df: fm.Chart(df).mark_line().encode(x="w", y="v", stroke_dash="k"),
}


@pytest.mark.parametrize("channel", list(_LEGEND_CHARTS))
def test_a_categorical_legend_over_dates_lists_the_dates_in_time_order(channel: str):
    rows = _by_time(_rows(_UNSORTED))
    svg, _ = _assert_twin(
        _LEGEND_CHARTS[channel](_as_date(rows)), _LEGEND_CHARTS[channel](_as_text(rows))
    )
    labels = _labels(_UNSORTED)
    texts = [t for _, _, t in ordered_text_entries(svg) if t in set(labels)]
    assert texts == labels


@pytest.mark.parametrize("channel", list(_LEGEND_CHARTS))
def test_a_legend_over_unsorted_date_rows_lists_the_dates_in_time_order(channel: str):
    svg = _LEGEND_CHARTS[channel](_as_date(_rows(_UNSORTED))).to_svg()
    labels = _labels(_UNSORTED)
    texts = [t for _, _, t in ordered_text_entries(svg) if t in set(labels)]
    assert texts == labels


_FACETS: dict[str, Callable[[fm.Chart], fm.Chart]] = {
    "facet": lambda c: c.facet("k"),
    "row": lambda c: c.facet(row="k"),
    "col": lambda c: c.facet(col="k"),
}


@pytest.mark.parametrize("door", list(_FACETS))
def test_a_facet_over_dates_draws_a_panel_per_date_headed_by_its_text(door: str):
    rows = _by_time(_rows(_UNSORTED))
    chart = lambda df: _FACETS[door](fm.Chart(df).mark_point().encode(x="w", y="v"))  # noqa: E731
    svg, _ = _assert_twin(chart(_as_date(rows)), chart(_as_text(rows)))
    labels = _labels(_UNSORTED)
    headers = [t for _, _, t in ordered_text_entries(svg) if t in set(labels)]
    assert headers == labels


@pytest.mark.parametrize("door", list(_FACETS))
def test_a_facet_over_unsorted_date_rows_orders_its_panels_by_time(door: str):
    chart = _FACETS[door](fm.Chart(_as_date(_rows(_UNSORTED))).mark_point().encode(x="w", y="v"))
    labels = _labels(_UNSORTED)
    headers = [t for _, _, t in ordered_text_entries(chart.to_svg()) if t in set(labels)]
    assert headers == labels


# ---------------------------------------------------------------------------
# A null date, an explicit order
# ---------------------------------------------------------------------------


def _with_a_null_date(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("k")).alias("k")
    )


def test_a_null_date_is_the_null_category_the_text_twin_has():
    """A null date is dropped from the categories and named once, in the words a null text
    category is: the date chart's warnings and drawing are the twin's."""
    rows = _with_a_null_date(_by_time(_rows(_UNSORTED)))
    twin = rows.with_columns(
        k=pl.Series([None if v is None else _label(v) for v in rows["k"].to_list()])
    )
    twin_out, twin_messages = _render_capturing(
        lambda: _BAND_MARKS["boxplot"](twin).to_svg(), action="always"
    )
    out, messages = _render_capturing(
        lambda: _BAND_MARKS["boxplot"](_as_date(rows)).to_svg(), action="always"
    )
    assert twin_messages, "the twin names the null category, so the comparison proves something"
    assert out == twin_out
    assert messages == twin_messages
    assert _tick_labels(out, _labels(_UNSORTED)) == _labels(_UNSORTED)


_DOMAIN_SPELLINGS: dict[str, tuple[Callable[[list[str]], Any], list[str]]] = {
    "sort": (lambda order: fm.X("k", sort=order), ["2020-01-03", "2020-01-01", "2020-01-02"]),
    "band_domain": (
        lambda order: fm.X("k", scale=fm.BandScale(domain=order)),
        ["2020-01-02", "2020-01-03", "2020-01-01"],
    ),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("spelling", list(_DOMAIN_SPELLINGS))
def test_a_sort_or_band_domain_of_date_labels_is_honored(spelling: str, output: str):
    """The labels a caller writes name the dates (a midnight reads ``YYYY-MM-DD``): the bands
    follow that order, as the label-text column's bands do."""
    channel, order = _DOMAIN_SPELLINGS[spelling]
    rows = _by_time(_rows(_UNSORTED))

    def chart(frame: pl.DataFrame) -> fm.Chart:
        return fm.Chart(frame).mark_boxplot().encode(x=channel(order), y="v")

    drawn, messages = _assert_twin(chart(_as_date(rows)), chart(_as_text(rows)), output)
    assert messages == []
    if output == "static":
        assert _tick_labels(drawn, order) == order
    else:
        assert _interactive_ticks(chart(_as_date(rows)), order) == order


# ---------------------------------------------------------------------------
# Static, interactive and every composition door
# ---------------------------------------------------------------------------


def _bar(df: pl.DataFrame) -> fm.Chart:
    return fm.Chart(df).mark_bar().encode(x="k:O", y="mean(v)")


_DOORS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "static": _bar,
    "layer": lambda df: _bar(df) + _bar(df),
    "Chart.layer": lambda df: _bar(df).layer(
        fm.Layer(mark="bar", encoding={"x": "k:O", "y": "mean(v)"}),
    ),
    "LayerChart": lambda df: fm.LayerChart(_bar(df), _bar(df)),
    "hconcat": lambda df: _bar(df) | _bar(df),
    "boxplot_layer": lambda df: _BAND_MARKS["boxplot"](df) + _BAND_MARKS["boxplot"](df),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", list(_DOORS))
def test_every_door_draws_the_dates_as_its_text_twin_does(door: str, output: str):
    rows = _by_time(_rows(_UNSORTED))
    _, messages = _assert_twin(_DOORS[door](_as_date(rows)), _DOORS[door](_as_text(rows)), output)
    assert messages == []


@pytest.mark.parametrize("door", list(_DOORS))
def test_every_door_draws_the_dates_in_time_order(door: str):
    svg = _DOORS[door](_as_date(_rows(_UNSORTED))).to_svg()
    labels = _labels(_UNSORTED)
    drawn = [t for _, _, t in ordered_text_entries(svg) if t in set(labels)]
    assert drawn[: len(labels)] == labels


@pytest.mark.parametrize("mark", list(_BAND_MARKS))
def test_the_interactive_scene_draws_the_ticks_the_static_chart_does(mark: str):
    chart = _BAND_MARKS[mark](_as_date(_rows(_UNSORTED)))
    labels = _labels(_UNSORTED)
    assert _interactive_ticks(chart, labels) == labels
    assert _tick_labels(chart.to_svg(), labels) == labels


def test_the_interactive_scene_of_a_date_facet_names_its_panels_by_date():
    chart = fm.Chart(_as_date(_rows(_UNSORTED))).mark_point().encode(x="w", y="v").facet("k")
    labels = _labels(_UNSORTED)
    assert _interactive_ticks(chart, labels) == labels


def test_the_interactive_legend_of_a_date_lists_the_dates_in_time_order():
    chart = _LEGEND_CHARTS["color"](_as_date(_rows(_UNSORTED)))
    labels = _labels(_UNSORTED)
    assert _interactive_ticks(chart, labels) == labels


# ---------------------------------------------------------------------------
# Layers over one date category axis
# ---------------------------------------------------------------------------
#
# The first layer that draws a channel establishes its scale, and every layer on that panel draws
# through it. A date category axis (a box over dates) holds every later layer's dates in time
# order, whichever frame they come from and whichever door layers them.

_LAYER_DOORS = ["+", "Chart.layer", "fm.layer", "LayerChart"]


def _compose(door: str, lead: fm.Chart, mark: str, encoding: dict[str, Any], frame: pl.DataFrame):
    """*lead* with a second layer of *mark* over *frame*, through *door*."""
    later = getattr(fm.Chart(frame), f"mark_{mark}")().encode(**encoding)
    if door == "+":
        return lead + later
    if door == "Chart.layer":
        return lead.layer(fm.Layer(mark=mark, data=frame, encoding=encoding))
    if door == "fm.layer":
        return fm.layer(lead, later)
    return fm.LayerChart(lead, later)


def _circle_xs(drawn: Any, output: str) -> list[float]:
    """The sorted x of every circle a static render draws (or an interactive scene holds)."""
    if output == "static":
        return sorted(
            float(c["cx"]) for c in svg_elements(mark_layer(drawn), "circle", lambda a: 1)
        )
    return sorted(
        float(node["cx"])
        for panel in drawn["panels"]
        for batch in panel["marks"]
        for node in batch["nodes"]
        if node["type"] == "circle"
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS)
@pytest.mark.parametrize("point_x", ["k", "k:T"], ids=["undeclared", "declared_T"])
def test_a_point_layer_over_the_box_axis_dates_lands_on_its_band(
    point_x: str, door: str, output: str
):
    """A box over dates establishes a date category axis; a later point layer over the same
    column, undeclared or declared temporal, draws on that axis: every row's circle sits on its
    date's band, the chart is the label-text twin's, and nothing warns."""
    rows = _by_time(_rows(_UNSORTED))

    def chart(frame: pl.DataFrame, x: str) -> Any:
        return _compose(door, _BAND_MARKS["boxplot"](frame), "point", {"x": x, "y": "v"}, frame)

    drawn, messages = _assert_twin(
        chart(_as_date(rows), point_x), chart(_as_text(rows), "k"), output
    )
    assert messages == []
    xs = _circle_xs(drawn, output)
    assert len(xs) == len(rows)
    if output == "static":
        ticks = {text: x for x, text in _x_ticks(drawn, _labels(_UNSORTED))}
        expected = sorted(ticks[_label(v)] for v in rows["k"].to_list())
        assert xs == pytest.approx(expected, abs=0.6)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS)
def test_a_point_layer_over_another_frames_dates_widens_the_box_axis_in_time_order(
    door: str, output: str
):
    """The point layer's frame holds a date the box layer lacks (2020-01-04): the axis holds
    every date in time order and each point lands on its own band."""
    boxes = _by_time(_rows(_UNSORTED))
    points = pl.DataFrame(
        {"k": [dt.datetime(2020, 1, 4), dt.datetime(2020, 1, 2)], "v": [30.0, 12.0]}
    )
    labels = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]

    def chart(box_frame: pl.DataFrame, point_frame: pl.DataFrame) -> Any:
        return _compose(
            door,
            _BAND_MARKS["boxplot"](box_frame),
            "point",
            {"x": "k", "y": "v"},
            point_frame,
        )

    drawn, messages = _assert_twin(
        chart(_as_date(boxes), _as_date(points)), chart(_as_text(boxes), _as_text(points)), output
    )
    assert messages == []
    xs = _circle_xs(drawn, output)
    assert len(xs) == 2
    if output == "static":
        ticks = _x_ticks(drawn, labels)
        assert [text for _, text in ticks] == labels
        assert xs == pytest.approx([ticks[1][0], ticks[3][0]], abs=0.6)
    else:
        assert _interactive_ticks(chart(_as_date(boxes), _as_date(points)), labels) == labels


def test_a_point_layer_first_then_a_box_over_dates_matches_its_integer_twin():
    """With the point layer first its undeclared date establishes the axis (layer 0 owns it), as an
    integer column does: the same circles and rects, the same ordered warnings."""
    rows = _by_time(_rows(_UNSORTED))
    twin = _plain_kinds(rows)["Int64"]

    def chart(frame: pl.DataFrame) -> Any:
        return fm.Chart(frame).mark_point().encode(x="k", y="v") + _BAND_MARKS["boxplot"](frame)

    drawn, messages = _drawn(chart(_as_date(rows)))
    twin_drawn, twin_messages = _drawn(chart(twin))
    count = lambda svg, tag: len(svg_elements(mark_layer(svg), tag, lambda a: 1))  # noqa: E731
    assert (count(drawn, "circle"), count(drawn, "rect")) == (
        count(twin_drawn, "circle"),
        count(twin_drawn, "rect"),
    )
    assert count(twin_drawn, "circle") > 0
    assert messages == twin_messages


_FRAME_DATES = {
    "x": ([3, 1], [2, 1], [1, 2, 3]),
    "color": ([3, 2], [1, 4], [1, 2, 3, 4]),
    "shape": ([3, 2], [1, 4], [1, 2, 3, 4]),
    "stroke_dash": ([3, 2], [1, 4], [1, 2, 3, 4]),
}
_LAYER_CHANNELS: dict[str, tuple[str, dict[str, Any]]] = {
    "x": ("bar", {"x": fm.X("k", type_="N"), "y": "v"}),
    "color": ("point", {"x": "w", "y": "v", "color": fm.Color("k", type_="N")}),
    "shape": ("point", {"x": "w", "y": "v", "shape": "k"}),
    "stroke_dash": ("line", {"x": "w", "y": "v", "stroke_dash": "k"}),
}


def _dated(days: list[int], *, as_text: bool = False) -> pl.DataFrame:
    """Two rows per day (2020-01-<day>), the dates as Datetime or as their label text."""
    frame = pl.DataFrame(
        {
            "k": [dt.datetime(2020, 1, d) for d in days for _ in range(2)],
            "v": [float(d + r) for d in days for r in range(2)],
            "w": [float(r) for _ in days for r in range(2)],
        }
    )
    return _as_text(frame) if as_text else frame


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS)
@pytest.mark.parametrize("channel", list(_LAYER_CHANNELS))
def test_layers_over_different_frames_draw_their_dates_in_time_order(
    channel: str, door: str, output: str
):
    """Each layer lists its own dates (the first layer's later date first); the shared scale lists
    them all in time order, the order of the single chart over the concatenated rows, on every
    door and both outputs."""
    first, second, expected = _FRAME_DATES[channel]
    mark, encoding = _LAYER_CHANNELS[channel]
    labels = [f"2020-01-0{d}" for d in expected]

    def layered(*, as_text: bool) -> Any:
        lead = getattr(fm.Chart(_dated(first, as_text=as_text)), f"mark_{mark}")().encode(
            **encoding
        )
        return _compose(door, lead, mark, encoding, _dated(second, as_text=as_text))

    single = getattr(fm.Chart(_dated(sorted(set(first + second)))), f"mark_{mark}")().encode(
        **encoding
    )
    chart = layered(as_text=False)
    if output == "static":
        order = lambda c: [t for _, _, t in ordered_text_entries(c.to_svg()) if t in set(labels)]  # noqa: E731
    else:
        order = lambda c: _interactive_ticks(c, labels)  # noqa: E731
    assert order(chart) == labels
    assert order(single) == labels
    _, messages = _drawn(chart, output)
    assert messages == []


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS)
def test_a_text_axis_a_declared_date_layer_joins_keeps_its_text_first(door: str, output: str):
    """A text category axis keeps its own categories first; a later layer's dates declared nominal
    join after them, in time order."""
    text = pl.DataFrame({"k": ["c", "a"], "v": [1.0, 2.0]})
    lead = fm.Chart(text).mark_bar().encode(x=fm.X("k", type_="N"), y="v")
    chart = _compose(door, lead, "bar", {"x": fm.X("k", type_="N"), "y": "v"}, _dated([3, 1]))
    expected = ["c", "a", "2020-01-01", "2020-01-03"]
    if output == "static":
        assert _tick_labels(chart.to_svg(), expected) == expected
    else:
        assert _interactive_ticks(chart, expected) == expected


@pytest.mark.parametrize("door", ["+", "Chart.layer", "LayerChart"])
def test_a_layered_date_category_chart_draws_its_text_twins_scene_interactively(door: str):
    """The interactive overlay of a box and a point layer over dates is the label-text twin's
    scene (tooltips apart): the same 15 circles at the same positions, the same warnings."""
    rows = _by_time(_rows(_UNSORTED))

    def chart(frame: pl.DataFrame) -> Any:
        return _compose(door, _BAND_MARKS["boxplot"](frame), "point", {"x": "k", "y": "v"}, frame)

    drawn, messages = _assert_twin(chart(_as_date(rows)), chart(_as_text(rows)), "interactive")
    assert messages == []
    assert len(_circle_xs(drawn, "interactive")) == len(rows)


# ---------------------------------------------------------------------------
# Unchanged: a date that is not read as categories, and every other category dtype
# ---------------------------------------------------------------------------


def _digest(charts: list[Callable[[], Any]]) -> str:
    """The MD5 of the SVG each chart draws (or of the refusal it raises) with the warnings it
    raised, in order."""
    md5 = hashlib.md5()
    for build in charts:
        try:
            out, messages = _render_capturing(lambda b=build: b().to_svg(), action="always")
        except ValueError as refusal:
            out, messages = f"refused: {refusal}", []
        md5.update(out.encode())
        md5.update("\n".join(messages).encode())
    return md5.hexdigest()


_TEMPORAL_CONTROLS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "bar": lambda df: fm.Chart(df).mark_bar().encode(x="k", y="v"),
    "line": lambda df: fm.Chart(df).mark_line().encode(x="k", y="v"),
    "tick": lambda df: fm.Chart(df).mark_tick().encode(x="k"),
    "point_colour": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", color="k"),
    "box_stats": lambda df: (
        fm.Chart(df)
        .transform(fm.BoxStats("v", groupby=["k"]))
        .mark_point()
        .encode(x="k", y="median")
    ),
}

# Measured on the v0.25.0 release build (the build before dates drew as categories).
_TEMPORAL_DIGESTS = {
    "bar": "08605b38eec7379cc4424ea93b42ebc3",
    "line": "3d1d4ab6569cdf21a1bb9a26a95b437d",
    "tick": "986c1ae952d9dbe2d2439c06cde66e3b",
    "point_colour": "370fd7e63de0c6d2ec853a2334130f1e",
    "box_stats": "f2986d7bbc934ae08e3a6147333ad979",
}


@pytest.mark.parametrize("control", list(_TEMPORAL_CONTROLS))
def test_a_date_read_as_time_draws_as_it_did(control: str):
    frame = _rows(_UNSORTED)
    digest = _digest(
        [
            lambda f=build: _TEMPORAL_CONTROLS[control](f(frame))
            for build in (_as_date, _as_unit("ms"), _as_unit("ns", "Europe/Paris"))
        ]
    )
    assert digest == _TEMPORAL_DIGESTS[control]


def _plain_kinds(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    ranked = frame.with_columns(pl.col("k").rank("dense").cast(pl.Int64).alias("rank"))
    return {
        "Utf8": ranked.with_columns(k=pl.col("rank").cast(pl.Utf8)).drop("rank"),
        "Int64": ranked.with_columns(k=pl.col("rank")).drop("rank"),
        "Float64": ranked.with_columns(k=pl.col("rank").cast(pl.Float64)).drop("rank"),
        "Boolean": ranked.with_columns(k=pl.col("rank") > 1).drop("rank"),
    }


_PLAIN_CHARTS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    **_BAND_MARKS,
    "bar_N": lambda df: fm.Chart(df).mark_bar().encode(x="k:N", y="v"),
    "colour": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", color="k:N"),
    "shape": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", shape="k"),
    "facet": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v").facet("k"),
    "catplot": lambda df: fm.catplot(df, x="k", y="v", kind="box"),
}

# Measured on the v0.25.0 release build (the build before dates drew as categories).
_PLAIN_DIGESTS = {
    "boxplot": "bfa9c13c86bedf9bdffba55b2b18e4bd",
    "errorbar": "b13b5d84db91f1808191281d2e60c8f1",
    "violin": "b1af1ce362caa9ff4d8c7b5613936ca8",
    "boxen": "f8bd9cbed43b0684bd657de886c169b6",
    "bar_N": "91efad8944ad48afb82063da061b6e01",
    "colour": "7d8190ad64d396766a44bec73817fd4b",
    "shape": "99e8aeb9ce2d33257db5442ba5ec3b92",
    "facet": "1e2037bb1d1bef384b64d109d6208a58",
    "catplot": "bfa9c13c86bedf9bdffba55b2b18e4bd",
}


@pytest.mark.parametrize("chart", list(_PLAIN_CHARTS))
def test_an_integer_text_boolean_or_float_category_draws_as_it_did(chart: str):
    frames = _plain_kinds(_rows(_MIDNIGHTS))
    digest = _digest([lambda f=f: _PLAIN_CHARTS[chart](f) for f in frames.values()])
    assert digest == _PLAIN_DIGESTS[chart]


@pytest.mark.parametrize(
    ("channel", "kind"),
    [
        (fm.Color("k", type_="N"), "categorical"),
        (fm.Color("k", type_="O"), "categorical"),
        (fm.Color("k"), "temporal"),
    ],
)
def test_the_raster_colour_gate_reads_a_date_by_how_the_channel_declares_it(
    channel: Any, kind: str
):
    """A date declared nominal or ordinal is categorical to the auto-raster gate, as it is to the
    engine; undeclared it stays temporal."""
    from ferrum._render import _color_type_kind

    assert _color_type_kind(channel, _as_date(_rows(_UNSORTED))) == kind


# ---------------------------------------------------------------------------
# A Duration is not a date: a category binding refuses it before it crosses
# ---------------------------------------------------------------------------
#
# A Duration crosses to the engine as a timestamp, so reading it as categories would label them
# with 1970 epoch dates. Every binding the engine reads as categories refuses it in Python,
# naming the column the caller wrote, with a remedy that draws.


def _durations() -> pl.DataFrame:
    """Three durations (out of order, five rows each) beside two values, a day per duration and
    a second text column."""
    seconds = [10, 5, 15]
    return pl.DataFrame(
        {
            "k": [s * 1000 for s in seconds for _ in range(_REPEATS)],
            "v": [float(i * 10 + r + 1) for i in range(3) for r in range(_REPEATS)],
            "w": [float(r) for _ in range(3) for r in range(_REPEATS)],
            "d": [dt.date(2020, 1, 1 + i) for i in range(3) for _ in range(_REPEATS)],
            "o": ["a" if r % 2 else "b" for _ in range(3) for r in range(_REPEATS)],
        }
    ).with_columns(pl.col("k").cast(pl.Duration("ms")))


def _as_text_durations(frame: pl.DataFrame) -> pl.DataFrame:
    """The remedy the refusal names, applied: the column as its Polars duration text."""
    return frame.with_columns(pl.col("k").dt.to_string("polars"))


def _duration_refusal(channel: str, column: str = "k") -> str:
    where = (
        "it is a facet field" if channel == "facet" else f"the {channel} channel reads categories"
    )
    return (
        f"column '{column}' is a Duration, which cannot be read as categories ({where}); convert "
        f"it to text with pl.col('{column}').dt.to_string('polars') for one category per value"
    )


def _area(**channel: Any) -> Callable[[pl.DataFrame], fm.Chart]:
    return lambda df: fm.Chart(df).mark_area().encode(x="w", y="v", **channel)


def _grid(**facet: str) -> Callable[[pl.DataFrame], fm.Chart]:
    return lambda df: fm.Chart(df).mark_point().encode(x="w", y="v").facet(**facet)


# binding -> (the chart over a frame with the Duration column k, the channel the refusal names;
# an area's fill and stroke are spelled as its colour channel)
_DURATION_BINDINGS: dict[str, tuple[Callable[[pl.DataFrame], Any], str]] = {
    **{f"mark_{name}": (build, "x") for name, build in _BAND_MARKS.items()},
    **{
        f"catplot_{kind}": (lambda df, kind=kind: _catplot(kind, df), "x")
        for kind in _CATPLOT_KINDS
    },
    "bar_ordinal": (lambda df: fm.Chart(df).mark_bar().encode(x="k:O", y="mean(v)"), "x"),
    "bar_nominal": (lambda df: fm.Chart(df).mark_bar().encode(x="k:N", y="mean(v)"), "x"),
    "bar_band_scale": (
        lambda df: fm.Chart(df).mark_bar().encode(x=fm.X("k", scale=fm.BandScale()), y="mean(v)"),
        "x",
    ),
    "colour_nominal": (_LEGEND_CHARTS["color"], "color"),
    "area_colour": (_area(color="k"), "color"),
    "area_fill": (_area(fill="k"), "color"),
    "area_stroke": (_area(stroke="k"), "color"),
    "shape": (_LEGEND_CHARTS["shape"], "shape"),
    "stroke_dash": (_LEGEND_CHARTS["stroke_dash"], "stroke_dash"),
    "facet": (lambda df: _facetted("facet", df), "facet"),
    "facet_row": (lambda df: _facetted("row", df), "facet"),
    "facet_col": (lambda df: _facetted("col", df), "facet"),
    "grid_duration_row": (_grid(row="k", col="o"), "facet"),
    "grid_duration_col": (_grid(row="o", col="k"), "facet"),
    "point_over_a_box_axis_of_dates": (
        lambda df: (
            fm.Chart(df).mark_boxplot().encode(x="d", y="v")
            + fm.Chart(df).mark_point().encode(x="k", y="v")
        ),
        "x",
    ),
}


def _facetted(door: str, df: pl.DataFrame) -> fm.Chart:
    return _FACETS[door](fm.Chart(df).mark_point().encode(x="w", y="v"))


def _refusal_of(chart: Any, output: str) -> str:
    with pytest.raises(ValueError) as raised:
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    return str(raised.value)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("binding", list(_DURATION_BINDINGS))
def test_a_duration_on_a_category_binding_is_refused_naming_the_callers_column(
    binding: str, output: str
):
    build, channel = _DURATION_BINDINGS[binding]
    assert _refusal_of(build(_durations()), output) == _duration_refusal(channel)


def _colour_chart(frame: pl.DataFrame) -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x="w", y="v", color=fm.Color("k", type_="N"))


_COLOUR = {"x": "w", "y": "v", "color": fm.Color("k", type_="N")}


def _door_over(door: str, frame: pl.DataFrame, lead: pl.DataFrame) -> Any:
    """The colour binding over *frame* through *door*; where the door layers, *lead* is the
    first layer's frame (a text column of the same name)."""
    if door == "single":
        return _colour_chart(frame)
    if door == "plus_same_frame":
        return fm.Chart(frame).mark_point().encode(x="w", y="v") + _colour_chart(frame)
    if door == "hconcat":
        return fm.Chart(frame).mark_point().encode(x="w", y="v") | _colour_chart(frame)
    return _compose(door, _colour_chart(lead), "point", _COLOUR, frame)


_DURATION_DOORS = [
    "single",
    "plus_same_frame",
    "+",
    "Chart.layer",
    "fm.layer",
    "LayerChart",
    "hconcat",
]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _DURATION_DOORS)
def test_a_duration_on_a_category_binding_is_refused_naming_the_callers_column_on_every_door(
    door: str, output: str
):
    """Every door, a layer over a distinct frame included (where ``+`` and ``Chart.layer`` rename
    the operand's columns), refuses with the one text naming ``k``, the caller's own column."""
    durations = _durations()
    chart = _door_over(door, durations, _as_text_durations(durations))
    assert _refusal_of(chart, output) == _duration_refusal("color")


@pytest.mark.parametrize("binding", list(_DURATION_BINDINGS))
def test_the_duration_refusals_remedy_draws_on_every_binding(binding: str):
    """Following the remedy as written draws the chart, one category per distinct value."""
    build, _ = _DURATION_BINDINGS[binding]
    svg, messages = _render_capturing(
        build(_as_text_durations(_durations())).to_svg, action="always"
    )
    drawn = {t for _, _, t in ordered_text_entries(svg)}
    assert {"5s", "10s", "15s"} <= drawn, (binding, sorted(drawn))
    assert messages == []


def test_the_duration_remedy_draws_one_category_per_value_including_sub_second():
    """Sub-second durations stay apart under the remedy: four values, four legend entries in
    first-appearance order, no warning."""
    millis = [250, 500, 1500, -250]
    frame = pl.DataFrame(
        {
            "k": [m for m in millis for _ in range(2)],
            "v": [float(i + r) for i in range(4) for r in range(2)],
            "w": [float(r) for _ in range(4) for r in range(2)],
        }
    ).with_columns(pl.col("k").cast(pl.Duration("ms")))
    svg, messages = _render_capturing(
        _colour_chart(_as_text_durations(frame)).to_svg, action="always"
    )
    labels = ["250ms", "500ms", "1s 500ms", "-250ms"]
    assert [t for _, _, t in ordered_text_entries(svg) if t in set(labels)] == labels
    assert messages == []


@pytest.mark.parametrize(
    "binding",
    ["point_x", "bar_x", "untyped_colour", "detail", "line_x"],
)
def test_a_duration_no_category_binding_reads_draws_as_its_epoch_datetime_twin(binding: str):
    """A Duration on a plain position, an untyped colour or a detail is not a category: it draws,
    and draws what the same instants as a date-time column draw."""
    builds = {
        "point_x": lambda df: fm.Chart(df).mark_point().encode(x="k", y="v"),
        "bar_x": lambda df: fm.Chart(df).mark_bar().encode(x="k", y="v"),
        "line_x": lambda df: fm.Chart(df).mark_line().encode(x="k", y="v"),
        "untyped_colour": lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", color="k"),
        "detail": lambda df: fm.Chart(df).mark_line().encode(x="w", y="v", detail="k"),
    }
    frame = _durations()
    twin = frame.with_columns(
        (pl.lit(dt.datetime(1970, 1, 1)) + pl.col("k")).cast(pl.Datetime("ms")).alias("k")
    )
    for output in _OUTPUTS:
        drawn, _ = _assert_twin(builds[binding](frame), builds[binding](twin), output)
        assert drawn


def test_a_duration_in_a_layer_the_facet_and_the_marks_never_read_is_untouched():
    """The refusal reads the columns a category binding names, not every Duration in the frame."""
    frame = _durations().with_columns(pl.col("k").alias("unused"))
    plain = _durations()
    chart = lambda df: fm.Chart(df).mark_point().encode(x="w", y="v")  # noqa: E731
    _assert_twin(chart(frame), chart(plain))


# ---------------------------------------------------------------------------
# Date-times within one millisecond share a category, and say so once
# ---------------------------------------------------------------------------
#
# Date-times reach the renderer at millisecond precision, so distinct microsecond or nanosecond
# values within one millisecond draw as one category. A binding that reads categories warns once
# per column, naming the counts.

_COLLISION_WARNING = (
    "column 'k' holds 3 distinct date-times, which draw as 2 categories: date-times reach the "
    "renderer at millisecond precision, so values within one millisecond share a category"
)


def _colliding(unit: str = "us", *, distinct_ms: bool = False) -> pl.DataFrame:
    """Three date-times (two within one millisecond unless ``distinct_ms``), five rows each."""
    base = dt.datetime(2020, 1, 1, 12)
    gap = 1000 if distinct_ms else 100
    values = [base, base + dt.timedelta(microseconds=gap), dt.datetime(2020, 1, 2)]
    frame = pl.DataFrame(
        {
            "k": [v for v in values for _ in range(_REPEATS)],
            "v": [float(i * 10 + r + 1) for i in range(3) for r in range(_REPEATS)],
            "w": [float(r) for _ in range(3) for r in range(_REPEATS)],
        }
    )
    return frame.with_columns(pl.col("k").cast(pl.Datetime(unit)))


def _at_ms(frame: pl.DataFrame) -> pl.DataFrame:
    """The same column at millisecond precision, as it reaches the engine."""
    return frame.with_columns(pl.col("k").cast(pl.Datetime("ms")))


_COLLISION_BINDINGS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "box_x": _BAND_MARKS["boxplot"],
    "colour_nominal": _LEGEND_CHARTS["color"],
    "shape": _LEGEND_CHARTS["shape"],
    "facet": lambda df: _facetted("facet", df),
}


def _captured(chart: Any, output: str) -> tuple[Any, list[str]]:
    return _render_capturing(
        chart.to_svg if output == "static" else (lambda: chart.interactive().scene_json),
        action="always",
    )


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("binding", list(_COLLISION_BINDINGS))
@pytest.mark.parametrize("unit", ["us", "ns"])
def test_datetimes_sharing_a_millisecond_on_a_category_binding_warn_once_naming_the_count(
    unit: str, binding: str, output: str
):
    frame = _colliding(unit)
    drawn, messages = _captured(_COLLISION_BINDINGS[binding](frame), output)
    twin, twin_messages = _captured(_COLLISION_BINDINGS[binding](_at_ms(frame)), output)
    assert messages == [_COLLISION_WARNING]
    assert twin_messages == []
    assert drawn == twin


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize(
    "door", ["single", "plus_same_frame", "+", "Chart.layer", "fm.layer", "LayerChart"]
)
def test_datetimes_sharing_a_millisecond_warn_once_on_every_door(door: str, output: str):
    """One chart is one check: a layered chart warns once, a layer over another frame included."""
    frame = _colliding()
    plain = pl.DataFrame({"k": ["x"], "v": [1.0], "w": [0.0]})
    chart = _door_over(door, frame, plain)
    _, messages = _captured(chart, output)
    assert messages == [_COLLISION_WARNING]


@pytest.mark.parametrize("output", _OUTPUTS)
def test_a_colliding_column_bound_twice_warns_once(output: str):
    frame = _colliding()
    chart = (
        fm.Chart(frame).mark_point().encode(x="w", y="v", color=fm.Color("k", type_="N"), shape="k")
    )
    assert _captured(chart, output)[1] == [_COLLISION_WARNING]
    layered = _colour_chart(frame) + fm.Chart(frame).mark_point().encode(x="w", y="v", shape="k")
    assert _captured(layered, output)[1] == [_COLLISION_WARNING]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(_colliding(distinct_ms=True), id="us_on_distinct_milliseconds"),
        pytest.param(_at_ms(_colliding()), id="ms_column"),
        pytest.param(_colliding().with_columns(pl.col("k").cast(pl.Date)), id="date_column"),
    ],
)
def test_datetimes_that_do_not_share_a_millisecond_draw_without_a_warning(
    frame: pl.DataFrame, output: str
):
    for binding in _COLLISION_BINDINGS.values():
        assert _captured(binding(frame), output)[1] == []


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("unit", ["us", "ns"])
def test_a_colliding_column_read_on_a_time_scale_does_not_warn(unit: str, output: str):
    """A time axis, an untyped colour and a temporal-declared position read the instants, not
    categories: the chart over the colliding column is the chart over its ms column."""
    builds = [
        lambda df: fm.Chart(df).mark_line().encode(x="k", y="v"),
        lambda df: fm.Chart(df).mark_point().encode(x="w", y="v", color="k"),
        lambda df: fm.Chart(df).mark_point().encode(x=fm.X("k", type_="T"), y="v"),
    ]
    frame = _colliding(unit)
    for build in builds:
        drawn, messages = _assert_twin(build(frame), build(_at_ms(frame)), output)
        assert messages == []


@pytest.mark.parametrize("output", _OUTPUTS)
def test_each_concat_leaf_warns_for_its_own_colliding_column(output: str):
    frame = _colliding()
    chart = _colour_chart(frame) | _colour_chart(frame)
    assert _captured(chart, output)[1] == [_COLLISION_WARNING, _COLLISION_WARNING]


# ---------------------------------------------------------------------------
# Every mark reads a date category as its label, never as an epoch number
# ---------------------------------------------------------------------------
#
# A date declared nominal, or one joining a date category axis, places and counts as its label on
# every mark that draws on a category axis: the chart is the label-text twin's, with the same
# ordered warnings (a census that read the date as an epoch number on an ordinal scale would name
# rows "outside the explicit domain" that the text twin does not).

_SWEEP_MARKS: dict[str, dict[str, Any]] = {
    "point": {"y": "v"},
    "line": {"y": "v"},
    "bar": {"y": "v"},
    "area": {"y": "v"},
    "ribbon": {"y": "v", "y2": "v2"},
    "tick": {},
    "rule": {},
    "text": {"y": "v", "text": "lab"},
    "rect": {"y": "v", "y2": "v2"},
    "segment": {"y": "v", "x2": "k2", "y2": "v2"},
}


def _sweep_frame(as_text: bool) -> pl.DataFrame:
    """Two rows per date; ``k2`` repeats ``k`` (a segment's far end), ``lab`` is a text column."""
    days = [dt.datetime(2020, 1, d) for d in (3, 1, 2)]
    frame = pl.DataFrame(
        {
            "k": [d for d in days for _ in range(2)],
            "k2": [d for d in days for _ in range(2)],
            "v": [float(i * 10 + r + 1) for i in range(3) for r in range(2)],
            "v2": [float(i * 10 + r + 4) for i in range(3) for r in range(2)],
            "lab": ["p", "q"] * 3,
        }
    ).sort("k", maintain_order=True)
    if not as_text:
        return frame
    return frame.with_columns(
        pl.col("k").dt.strftime("%Y-%m-%d"), pl.col("k2").dt.strftime("%Y-%m-%d")
    )


_SWEEP_X = {
    "declared": lambda: fm.X("k", type_="N"),
    "undeclared": lambda: "k",
    "declared_temporal": lambda: fm.X("k", type_="T"),
}


def _sweep_mark(mark: str, x: Any, frame: pl.DataFrame) -> fm.Chart:
    return getattr(fm.Chart(frame), f"mark_{mark}")().encode(x=x, **_SWEEP_MARKS[mark])


def _sweep_lead(frame: pl.DataFrame) -> fm.Chart:
    return fm.Chart(frame).mark_bar().encode(x=fm.X("k", type_="N"), y="v")


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("mark", list(_SWEEP_MARKS))
def test_a_date_declared_nominal_draws_and_warns_as_its_label_text_twin_on_every_mark(
    mark: str, output: str
):
    def chart(as_text: bool) -> fm.Chart:
        return _sweep_mark(mark, fm.X("k", type_="N"), _sweep_frame(as_text))

    _assert_twin(chart(False), chart(True), output)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", ["+", "Chart.layer", "LayerChart"])
@pytest.mark.parametrize("declared", list(_SWEEP_X))
@pytest.mark.parametrize("mark", list(_SWEEP_MARKS))
def test_a_date_joining_a_date_category_axis_draws_and_warns_as_its_label_text_twin_on_every_mark(
    mark: str, declared: str, door: str, output: str
):
    """A later layer's date, declared nominal, temporal or not at all, joins the date category axis a
    bar over the same dates establishes, whatever mark it draws."""

    def chart(as_text: bool) -> Any:
        frame = _sweep_frame(as_text)
        # the text twin has no type to declare: a column of text is a category already
        x = _SWEEP_X["declared"]() if declared == "declared" else "k"
        if not as_text:
            x = _SWEEP_X[declared]()
        encoding = {"x": x, **_SWEEP_MARKS[mark]}
        return _compose(door, _sweep_lead(frame), mark, encoding, frame)

    _assert_twin(chart(False), chart(True), output)


# ---------------------------------------------------------------------------
# A channel's date fact is the establishing read's, not any binding's
# ---------------------------------------------------------------------------
#
# A text category axis stays a text axis when a later layer declares a date nominal: a third layer
# whose undeclared column is a Duration or a sub-millisecond date-time still reads as time there
# (as it does beside the text axis alone), so it is neither refused nor warned. Where a date layer
# establishes the axis the same third layer reads as categories, and the refusal and the warning
# fire.

_THIRD_FRAMES: dict[str, pl.DataFrame] = {
    "duration": pl.DataFrame({"u": [1000, 2000, 3000], "v": [1.0, 2.0, 3.0]}).with_columns(
        pl.col("u").cast(pl.Duration("ms"))
    ),
    "collision": pl.DataFrame(
        {
            "u": [
                dt.datetime(2020, 1, 1, 12, 0, 0, 100),
                dt.datetime(2020, 1, 1, 12, 0, 0, 200),
                dt.datetime(2020, 1, 2),
            ],
            "v": [1.0, 2.0, 3.0],
        }
    ),
}


def _third_as_time(frame: pl.DataFrame) -> pl.DataFrame:
    """The third frame's column as the date-time it is read as on a time scale (a Duration as the
    instants it crosses as, a microsecond date-time at millisecond precision)."""
    column = pl.col("u")
    if isinstance(frame["u"].dtype, pl.Duration):
        column = pl.lit(dt.datetime(1970, 1, 1)) + column
    return frame.with_columns(column.cast(pl.Datetime("ms")))


def _layered(door: str, lead: fm.Chart, layers: list[tuple[str, dict[str, Any], pl.DataFrame]]):
    """*lead* with every ``(mark, encoding, frame)`` layer over it, through *door*."""
    charts = [getattr(fm.Chart(f), f"mark_{m}")().encode(**e) for m, e, f in layers]
    if door == "+":
        out = lead
        for chart in charts:
            out = out + chart
        return out
    if door == "Chart.layer":
        return lead.layer(*[fm.Layer(mark=m, data=f, encoding=e) for m, e, f in layers])
    if door == "fm.layer":
        return fm.layer(lead, *charts)
    return fm.LayerChart(lead, *charts)


_TEXT_LEAD = pl.DataFrame({"cat": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
_DATE_LEAD = pl.DataFrame({"k": [dt.datetime(2020, 1, d) for d in (1, 2, 3)], "v": [1.0, 2.0, 3.0]})
_THIRD = ("point", {"x": "u", "y": "v"})
_LAYER_DOORS_ALL = ["+", "Chart.layer", "fm.layer", "LayerChart"]


def _text_axis_chart(door: str, *, third: pl.DataFrame, with_date_layer: bool):
    lead = fm.Chart(_TEXT_LEAD).mark_bar().encode(x=fm.X("cat", type_="N"), y="v")
    layers = []
    if with_date_layer:
        layers.append(("bar", {"x": fm.X("k", type_="N"), "y": "v"}, _DATE_LEAD))
    layers.append((*_THIRD, third))
    return _layered(door, lead, layers)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS_ALL)
@pytest.mark.parametrize("kind", list(_THIRD_FRAMES))
def test_a_third_layer_beside_a_text_axis_reads_as_time_whatever_a_date_layer_declares(
    kind: str, door: str, output: str
):
    """Text axis, a declared-date layer, then an undeclared Duration (or colliding date-time) layer:
    the third layer is time, as the same column read as a millisecond date-time is, so the chart
    is that twin's, with no refusal and no warning."""
    frame = _THIRD_FRAMES[kind]
    drawn, messages = _assert_twin(
        _text_axis_chart(door, third=frame, with_date_layer=True),
        _text_axis_chart(door, third=_third_as_time(frame), with_date_layer=True),
        output,
    )
    assert messages == []


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS_ALL)
@pytest.mark.parametrize("kind", list(_THIRD_FRAMES))
def test_a_third_layer_beside_a_text_axis_alone_draws_as_it_did_at_v0_25_0(
    kind: str, door: str, output: str
):
    """Without the date layer the chart is the v0.25.0 chart: the recorded digest and warnings."""
    drawn, messages = _drawn(
        _text_axis_chart(door, third=_THIRD_FRAMES[kind], with_date_layer=False), output
    )
    expected = _THIRD_AT_V025[(kind, door, output)]
    assert (_digest16(drawn), tuple(messages)) == expected


def _digest16(drawn: Any) -> str:
    text = drawn if isinstance(drawn, str) else json.dumps(drawn, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Measured on the v0.25.0 release build (its python and extension, PYTHONPATH on a v0.25.0 copy).
_THIRD_AT_V025: dict[tuple[str, str, str], tuple] = {
    ("duration", "+", "static"): ("230f8a2223a2e55e", ()),
    ("duration", "+", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("duration", "Chart.layer", "static"): ("230f8a2223a2e55e", ()),
    ("duration", "Chart.layer", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("duration", "fm.layer", "static"): ("c802dcfc4754ef24", ()),
    ("duration", "fm.layer", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("duration", "LayerChart", "static"): ("c802dcfc4754ef24", ()),
    ("duration", "LayerChart", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("collision", "+", "static"): ("230f8a2223a2e55e", ()),
    ("collision", "+", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("collision", "Chart.layer", "static"): ("230f8a2223a2e55e", ()),
    ("collision", "Chart.layer", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("collision", "fm.layer", "static"): ("95da813b9a805a6f", ()),
    ("collision", "fm.layer", "interactive"): ("470bc5bdb1ce39f2", ()),
    ("collision", "LayerChart", "static"): ("95da813b9a805a6f", ()),
    ("collision", "LayerChart", "interactive"): ("470bc5bdb1ce39f2", ()),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS_ALL)
def test_a_duration_third_layer_over_a_date_axis_is_refused(door: str, output: str):
    """A date layer establishes the axis, so a later undeclared column joins it as categories: a
    Duration is refused, naming the column and the x channel."""
    lead = fm.Chart(_DATE_LEAD).mark_bar().encode(x=fm.X("k", type_="N"), y="v")
    chart = _layered(door, lead, [(*_THIRD, _THIRD_FRAMES["duration"])])
    assert _refusal_of(chart, output) == _duration_refusal("x", column="u")


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", _LAYER_DOORS_ALL)
def test_a_colliding_third_layer_over_a_date_axis_warns_once(door: str, output: str):
    lead = fm.Chart(_DATE_LEAD).mark_bar().encode(x=fm.X("k", type_="N"), y="v")
    chart = _layered(door, lead, [(*_THIRD, _THIRD_FRAMES["collision"])])
    assert _captured(chart, output)[1] == [_COLLISION_WARNING.replace("'k'", "'u'")]


# ---------------------------------------------------------------------------
# An explicit band domain that leaves a date out
# ---------------------------------------------------------------------------
#
# A date the caller's band domain omits is left out as its label text is: every mark warns the
# twin's ordered list (naming no remedy the twin's lacks), and draws the twin's marks.

_OMITTING_DOMAIN = ["2020-01-01", "2020-01-02"]


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("mark", list(_SWEEP_MARKS))
def test_a_declared_date_a_band_domain_omits_warns_as_its_label_text_twin_on_every_mark(
    mark: str, output: str
):
    def chart(as_text: bool) -> fm.Chart:
        x = fm.X("k", type_="N", scale=fm.BandScale(domain=_OMITTING_DOMAIN))
        return _sweep_mark(mark, x, _sweep_frame(as_text))

    _assert_twin(chart(False), chart(True), output)


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", ["+", "Chart.layer", "LayerChart"])
@pytest.mark.parametrize("declared", list(_SWEEP_X))
@pytest.mark.parametrize("mark", list(_SWEEP_MARKS))
def test_a_date_joining_a_band_domain_that_omits_it_warns_as_its_label_text_twin_on_every_mark(
    mark: str, declared: str, door: str, output: str
):
    """The lead bar's band domain omits 2020-01-03; a later layer over the same dates joins that
    axis and leaves the date out as the twin does."""

    def chart(as_text: bool) -> Any:
        frame = _sweep_frame(as_text)
        lead = (
            fm.Chart(frame)
            .mark_bar()
            .encode(x=fm.X("k", type_="N", scale=fm.BandScale(domain=_OMITTING_DOMAIN)), y="v")
        )
        x = _SWEEP_X["declared"]() if declared == "declared" else "k"
        if not as_text:
            x = _SWEEP_X[declared]()
        encoding = {"x": x, **_SWEEP_MARKS[mark]}
        return _compose(door, lead, mark, encoding, frame)

    _assert_twin(chart(False), chart(True), output)


# ---------------------------------------------------------------------------
# A static LayerChart whose merge refuses is checked as the one merged chart
# ---------------------------------------------------------------------------
#
# Two members carrying conditionals on one wire channel are drawn by the static overlay and
# refused by the merge the interactive route builds. The category check reads bindings and
# facets, never a conditional, so it reads that one merged reading all the same.


def _conditional(name: str) -> Any:
    return (
        fm.selection_point(fields=["c"], name=name)
        .when(fm.Color("c"))
        .otherwise(fm.value("#cccccc"))
    )


def _merge_refusing_pair(later_frame: pl.DataFrame) -> fm.LayerChart:
    """A date-axis bar lead and a later point layer over ``later_frame``'s ``u`` column, each
    carrying a conditional on colour."""
    days = [dt.date(2020, 1, d) for d in (1, 2, 3)]
    lead_frame = pl.DataFrame({"k": days, "v": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    lead = (
        fm.Chart(lead_frame)
        .mark_bar()
        .encode(x=fm.X("k", type_="N"), y="v", color=_conditional("p1"))
    )
    later = fm.Chart(later_frame).mark_point().encode(x="u", y="v", color=_conditional("p2"))
    return fm.LayerChart(lead, later)


def _later_frame(*, colliding: bool) -> pl.DataFrame:
    if colliding:
        base = dt.datetime(2020, 1, 1, 12)
        values = [base, base + dt.timedelta(microseconds=100), dt.datetime(2020, 1, 2)]
        column = pl.Series("u", values, dtype=pl.Datetime("us"))
    else:
        column = pl.Series("u", [1000, 2000, 3000]).cast(pl.Duration("ms"))
    return pl.DataFrame({"v": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]}).with_columns(column)


def test_a_layerchart_whose_merge_refuses_refuses_a_duration_joining_another_members_date_axis():
    chart = _merge_refusing_pair(_later_frame(colliding=False))
    assert _refusal_of(chart, "static") == _duration_refusal("x", column="u")


def test_a_layerchart_whose_merge_refuses_warns_for_a_colliding_member_on_another_members_date_axis():
    chart = _merge_refusing_pair(_later_frame(colliding=True))
    _, messages = _captured(chart, "static")
    assert messages == [_COLLISION_WARNING.replace("'k'", "'u'")]


def test_a_layerchart_whose_merge_refuses_warns_once_for_a_column_both_members_read():
    frame = (
        _colliding()
        .with_columns(pl.col("v").cast(pl.Float64))
        .with_columns(pl.Series("c", ["a", "b", "c", "a", "b"] * 3))
    )

    def member(name: str) -> fm.Chart:
        return (
            fm.Chart(frame).mark_point().encode(x="w", y="v", color=_conditional(name), shape="k")
        )

    _, messages = _captured(fm.LayerChart(member("s1"), member("s2")), "static")
    assert messages == [_COLLISION_WARNING]


def test_a_layerchart_whose_merge_refuses_keeps_its_interactive_conditional_refusal():
    chart = _merge_refusing_pair(_later_frame(colliding=False))
    with pytest.raises(ValueError, match="conditional"):
        chart.interactive()


# ---------------------------------------------------------------------------
# A layer the engine reads as categories is checked as the engine reads it
# ---------------------------------------------------------------------------
#
# The engine decides per scale whether a layer's colour reads categories (an area that draws
# colour groups by category, a box or violin layer declares its colour nominal), so the check
# reads the same layers the engine does: a Duration is refused and a same-millisecond
# collision warned wherever a layer reads categories, not only on the chart-level binding.


def _layered_colour_frame() -> pl.DataFrame:
    """Text ``s``, a value ``b``, a date-time ``d``, a Duration ``dur`` and a column ``us`` whose
    first two values share a millisecond."""
    base = dt.datetime(2020, 1, 1, 12)
    return pl.DataFrame(
        {
            "s": ["a", "b", "c", "a", "b", "c"],
            "b": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "d": [dt.datetime(2020, 1, 1 + i % 3) for i in range(6)],
            "dur": [1000, 2000, 3000, 1000, 2000, 3000],
            "us": [
                base,
                base + dt.timedelta(microseconds=100),
                dt.datetime(2020, 1, 2),
                base,
                base + dt.timedelta(microseconds=100),
                dt.datetime(2020, 1, 2),
            ],
        }
    ).with_columns(pl.col("dur").cast(pl.Duration("ms")), pl.col("us").cast(pl.Datetime("us")))


def _point(**channel: Any) -> fm.Chart:
    return fm.Chart(_layered_colour_frame()).mark_point().encode(x="s", y="b", **channel)


def _area_chart(**channel: Any) -> fm.Chart:
    return fm.Chart(_layered_colour_frame()).mark_area().encode(x="s", y="b", **channel)


def _figure_layer(mark: str, column: str) -> fm.Chart:
    box = getattr(fm.Chart(_layered_colour_frame()), f"mark_{mark}")()
    return box.encode(x="s", y="b", color="d").layer(
        fm.Layer(mark="point", encoding={"x": "s", "y": "b", "color": column})
    )


_LAYERED_DURATION_DOORS: dict[str, Callable[[], Any]] = {
    "boxplot_layer": lambda: _figure_layer("boxplot", "dur"),
    "violin_layer": lambda: _figure_layer("violin", "dur"),
    "point_lead_area_plus": lambda: _point(color="dur") + _area_chart(color="s"),
    "point_lead_area_layerchart": lambda: fm.LayerChart(
        _point(color="dur"), _area_chart(color="s")
    ),
    "inherited_area": lambda: _point(color="dur").layer(fm.Layer(mark="area")),
    "area_lead_point": lambda: _area_chart(color="s") + _point(color="dur"),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", list(_LAYERED_DURATION_DOORS))
def test_a_duration_the_engine_reads_as_categories_is_refused_on_every_layered_door(
    door: str, output: str
):
    chart = _LAYERED_DURATION_DOORS[door]()
    assert _refusal_of(chart, output) == _duration_refusal("color", column="dur")


_LAYERED_COLLISION_DOORS: dict[str, Callable[[], Any]] = {
    "boxplot_layer": lambda: _figure_layer("boxplot", "us"),
    "area_sibling": lambda: _point(color="us") + _area_chart(color="s"),
}


@pytest.mark.parametrize("output", _OUTPUTS)
@pytest.mark.parametrize("door", list(_LAYERED_COLLISION_DOORS))
def test_datetimes_sharing_a_millisecond_on_a_layer_the_engine_reads_as_categories_warn_once(
    door: str, output: str
):
    _, messages = _captured(_LAYERED_COLLISION_DOORS[door](), output)
    # The column holds three distinct microsecond values, two of them within one millisecond.
    assert messages == [_COLLISION_WARNING.replace("'k'", "'us'")]
