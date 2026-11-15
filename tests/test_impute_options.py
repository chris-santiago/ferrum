"""``transform_impute`` and the impute option dict, with Vega-Lite's options.

One impute rule serves the transform and the encoding dict (``Y(impute=...)``, ``X(impute=...)``):
``method`` (default ``'value'``), ``value`` (default 0, of the column's own kind), ``keyvals`` (added to
the keys the frame holds, as a list, as dates, or as a ``{'start', 'stop', 'step'}`` sequence) and
``frame`` (the windowed ``mean``, ``median``, ``min`` or ``max``). An imputed row is inserted at its
key's place in its series, so a line draws ``1, 2, 3`` and not ``1, 3, 2``, and a caller's row never
moves.

How each case is read:

* A row order is pinned through a text chart: ``x`` is the value column and ``y`` the group, so every
  row is one ``<text>`` mark in the order the rows hold, and an imputed row (value ``-1``) shows where
  it was placed. A line's vertex order is pinned through its drawn polyline.
* A written-out twin is the same chart over the frame with the imputed rows and the filled cells
  written in, by hand, at their place. It draws byte for byte what the imputing chart draws.
* An unchanged behavior is a digest measured on the build before this rule existed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import random
import re
import warnings
from collections.abc import Callable, Sequence, Set
from collections import OrderedDict
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pytest

import ferrum as fm
from ferrum._core import transform_registry
from ferrum._warn import reset_warnings, warn_once
from ferrum.annotation.coords import _NOT_A_TIME
from tests._svg_rects import polylines
from tests.test_transform_bare_string_sequence_params import (
    _bytes_text,
    _mapping_text,
    _set_text,
    _SetSubclass,
)
from tests.test_null_group_warning import (
    IMPUTE_KIND_CASES,
    _NO_ROWS_TO_DRAW,
    _group_dtype_refusal,
    _impute_kind_refusal,
    _render_capturing,
)

# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

_W, _H = 300, 200
_METHODS = "value|mean|median|min|max"
_OPTIONS = "method|value|keyvals|frame"
_OPTIONS_LIST = "method, value, keyvals, frame"


def _engine(text: str) -> str:
    """An engine transform refusal as Python surfaces it."""
    return f"transform failed: ValueError: {text}"


def _frame_fallback(group: str, rows: int, method: str = "mean") -> str:
    """The notice for rows whose frame holds no usable value and take the group's whole statistic."""
    return (
        f"Impute: group {group} has {rows} row(s) with no usable (non-null, finite) value within "
        f"the frame; the group's whole {method} was used instead"
    )


def _instant(millis: int) -> dict[str, int]:
    """A date or time as it crosses the wire: tagged, so a column that is not temporal refuses it."""
    return {"epoch_ms": millis}


def _marks(svg: str) -> list[tuple[float, str]]:
    """The text marks of a text chart, in drawing order, as ``(y pixel, label)``. Marks are drawn
    after the plot area's clip group, which holds nothing here."""
    return [
        (float(y), text)
        for y, text in re.findall(
            r'<text x="[^"]+" y="([^"]+)"[^>]*>(.*?)</text>', svg.rsplit("</g>", 1)[-1]
        )
    ]


def _text_chart(frame: pl.DataFrame, *steps: dict, text: str = "v", x: str = "v") -> fm.Chart:
    chart = fm.Chart(frame, width=_W, height=_H)
    for step in steps:
        chart = chart.transform(step)
    return chart.mark_text().encode(x=x, y="g:N", text=text)


def _rows(
    frame: pl.DataFrame, *steps: dict, groups: tuple[str, ...] = ("x", "y"), text: str = "v"
) -> list[tuple[str, str]]:
    """What the steps leave of ``frame``, row by row, as ``(group, label)``: the group is read off the
    mark's band (the first group listed draws highest), the label is the ``text`` column."""
    marks = _marks(_text_chart(frame, *steps, text=text).to_svg())
    bands = sorted({y for y, _ in marks})
    assert len(bands) <= len(groups), (bands, groups)
    rows = [(groups[bands.index(y)], label) for y, label in marks]
    fills_minus_one = (
        len(steps) == 1 and steps[0].get("method") == "value" and steps[0].get("value") == -1
    )
    values = frame["v"].to_list() if "v" in frame.columns else [None]
    if text == "v" and fills_minus_one and all(v is not None and v >= 0 for v in values):
        # an imputed row reads -1: every other row is a caller's, and none has moved
        assert [label for _, label in rows if label != "-1"] == [f"{v:g}" for v in values]
    return rows


def _impute(**kwargs: Any) -> dict:
    return fm.transform_impute("v", key="t", groupby=["g"], **kwargs)


def _series_x(svg: str) -> list[list[float]]:
    """The x of each drawn vertex of each polyline, in vertex order."""
    return [[x for x, _ in points] for _, points in polylines(svg)]


def _in_key_order(svg: str, *, series: int, vertices: int) -> None:
    drawn = _series_x(svg)
    assert len(drawn) == series, drawn
    for xs in drawn:
        assert len(xs) == vertices, xs
        assert xs == sorted(xs), f"a series draws its vertices out of key order: {xs}"
        assert len(set(xs)) == len(xs), xs


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

_KEYS: dict[str, tuple[Any, list[Any]]] = {
    "Int64": (pl.Int64, [1, 2, 3]),
    "UInt8": (pl.UInt8, [1, 2, 3]),
    "Int8": (pl.Int8, [1, 2, 3]),
    "Float64": (pl.Float64, [1.0, 2.0, 3.0]),
    "Float32": (pl.Float32, [1.0, 2.0, 3.0]),
    "Date": (pl.Date, [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 5)]),
    "Datetime[ms]": (
        pl.Datetime("ms"),
        [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 5)],
    ),
    "Datetime[us]": (
        pl.Datetime("us"),
        [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 5)],
    ),
    "Datetime[ns]": (
        pl.Datetime("ns"),
        [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 5)],
    ),
    "Datetime[us, UTC]": (
        pl.Datetime("us", "UTC"),
        [dt.datetime(2024, 1, d, tzinfo=dt.timezone.utc) for d in (1, 2, 5)],
    ),
    "Datetime[us, Europe/Paris]": (
        pl.Datetime("us", "Europe/Paris"),
        [dt.datetime(2024, 1, d, 12) for d in (1, 2, 5)],
    ),
    "Duration": (pl.Duration("ms"), [dt.timedelta(days=d) for d in (1, 2, 5)]),
}


def _keyed(kind: str, rows: Sequence[tuple[str, int, float | None]]) -> pl.DataFrame:
    """Rows of ``(group, key index, value)``, the key written as the kind's own dtype."""
    dtype, keys = _KEYS[kind]
    return pl.DataFrame(
        {
            "g": [g for g, _, _ in rows],
            "t": pl.Series("t", [keys[i] for _, i, _ in rows], dtype=dtype),
            "v": [v for _, _, v in rows],
        }
    )


# Group x holds keys 0 and 2, group y holds all three: x lacks the middle key.
_INTERIOR = [("x", 0, 10.0), ("x", 2, 30.0), ("y", 0, 1.0), ("y", 1, 2.0), ("y", 2, 3.0)]
_INTERIOR_ROWS = [("x", "10"), ("x", "-1"), ("x", "30"), ("y", "1"), ("y", "2"), ("y", "3")]


# ---------------------------------------------------------------------------
# Order: an imputed row is inserted at its key's place
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(_KEYS))
def test_an_interior_imputed_row_is_inserted_at_its_keys_place(kind: str):
    assert _rows(_keyed(kind, _INTERIOR), _impute(value=-1)) == _INTERIOR_ROWS


def test_an_integer_key_draws_its_line_in_key_order_as_the_written_out_twin_does():
    frame = pl.DataFrame(
        {
            "t": [1, 3, 1, 2, 3],
            "h": ["x", "x", "y", "y", "y"],
            "v": [1.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    twin = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3],
            "h": ["x", "x", "x", "y", "y", "y"],
            "v": [1.0, 0.0, 3.0, 1.0, 2.0, 3.0],
        }
    )

    def line(f: pl.DataFrame, *steps: dict) -> str:
        chart = fm.Chart(f, width=_W, height=_H)
        for step in steps:
            chart = chart.transform(step)
        return chart.mark_line().encode(x="t", y="v", color="h").to_svg()

    svg = line(frame, fm.transform_impute("v", value=0, key="t", groupby=["h"]))
    _in_key_order(svg, series=2, vertices=3)
    # the imputed vertex (v=0) is the middle one of series x, and the lowest on the page
    (_, first), (_, second) = polylines(svg)
    assert first[1][1] > max(first[0][1], first[2][1])
    assert [y for _, y in second] == sorted((y for _, y in second), reverse=True)
    assert svg == line(twin)


def test_a_date_key_draws_its_line_in_time_order_as_the_written_out_twin_does():
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 5)]
    frame = pl.DataFrame(
        {
            "t": [days[0], days[2], days[0], days[1], days[2]],
            "h": ["x", "x", "y", "y", "y"],
            "v": [1.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    twin = pl.DataFrame(
        {
            "t": [days[0], days[1], days[2], days[0], days[1], days[2]],
            "h": ["x", "x", "x", "y", "y", "y"],
            "v": [1.0, 0.0, 3.0, 1.0, 2.0, 3.0],
        }
    )

    def line(f: pl.DataFrame, *steps: dict) -> str:
        chart = fm.Chart(f, width=_W, height=_H)
        for step in steps:
            chart = chart.transform(step)
        return chart.mark_line().encode(x="t", y="v", color="h").to_svg()

    svg = line(frame, fm.transform_impute("v", value=0, key="t", groupby=["h"]))
    _in_key_order(svg, series=2, vertices=3)
    assert svg == line(twin)


def test_a_boolean_key_places_false_before_true():
    frame = pl.DataFrame({"g": ["x", "y", "y"], "t": [True, False, True], "v": [10.0, 1.0, 2.0]})
    assert _rows(frame, _impute(value=-1)) == [("x", "-1"), ("x", "10"), ("y", "1"), ("y", "2")]


def test_a_text_key_is_placed_in_the_keys_first_appearance_order():
    # first appearance across the frame is a, b, c: group y lacks the middle one
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y"],
            "t": ["a", "b", "c", "a", "c"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "1"),
        ("x", "2"),
        ("x", "3"),
        ("y", "4"),
        ("y", "-1"),
        ("y", "5"),
    ]


def test_a_text_keys_band_order_is_the_order_the_frame_gave_it():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y"],
            "t": ["c", "a", "b", "a", "b"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    def bands(f: pl.DataFrame, *steps: dict) -> list[str]:
        chart = fm.Chart(f, width=_W, height=_H)
        for step in steps:
            chart = chart.transform(step)
        svg = chart.mark_point().encode(x="t:N", y="v").to_svg()
        return re.findall(r'text-anchor="middle">([a-c])</text>', svg)

    assert bands(frame, _impute(value=-1)) == bands(frame) == ["c", "a", "b"]


def test_a_caller_row_never_moves_and_the_imputed_row_follows_the_groups_last_row():
    # group x lists keys 2, 0 on purpose: it lacks key 1, which no later row of x follows
    frame = _keyed(
        "Int64", [("x", 2, 30.0), ("x", 0, 10.0), ("y", 0, 1.0), ("y", 1, 2.0), ("y", 2, 3.0)]
    )
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "30"),
        ("x", "10"),
        ("x", "-1"),
        ("y", "1"),
        ("y", "2"),
        ("y", "3"),
    ]


def test_every_key_a_group_lacks_is_inserted_at_its_own_place_in_one_group():
    frame = _keyed("Int64", [("x", 1, 20.0)])
    # the one row holds key 2: keys 0 and 1 land before it, key 3 after it
    assert _rows(frame, _impute(value=-1, keyvals=[0, 1, 3]), groups=("x",)) == [
        ("x", "-1"),
        ("x", "-1"),
        ("x", "20"),
        ("x", "-1"),
    ]


def test_with_no_groupby_the_whole_frame_is_one_group_in_key_order():
    frame = pl.DataFrame({"t": [1, 3, 4], "v": [10.0, 30.0, 40.0]})
    step = fm.transform_impute("v", value=-1, key="t", keyvals=[2])
    chart = (
        fm.Chart(frame, width=_W, height=_H)
        .transform(step)
        .mark_text()
        .encode(x="v", y="t", text="v")
    )
    # key 2 lands between keys 1 and 3 of the one group
    assert [label for _, label in _marks(chart.to_svg())] == ["10", "-1", "30", "40"]


def test_a_group_holding_one_row_is_completed_around_it():
    frame = pl.DataFrame({"g": ["x", "y", "y", "y"], "t": [2, 1, 2, 3], "v": [20.0, 1.0, 2.0, 3.0]})
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "-1"),
        ("x", "20"),
        ("x", "-1"),
        ("y", "1"),
        ("y", "2"),
        ("y", "3"),
    ]


def test_a_frame_with_no_row_is_left_empty():
    frame = pl.DataFrame(
        {"g": [], "t": [], "v": []}, schema={"g": pl.Utf8, "t": pl.Int64, "v": pl.Float64}
    )
    svg, messages = _render_capturing(_text_chart(frame, _impute(value=-1)).to_svg)
    assert _marks(svg) == []
    assert messages == [_NO_ROWS_TO_DRAW]


# ---------------------------------------------------------------------------
# Order on every door
# ---------------------------------------------------------------------------

_DOOR_FRAME = pl.DataFrame(
    {
        "t": [1, 3, 1, 2, 3, 1, 3, 1, 2, 3],
        "h": ["x", "x", "y", "y", "y"] * 2,
        "f": ["a"] * 5 + ["b"] * 5,
        "v": [1.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 4.0, 5.0, 6.0],
    }
)
_DOOR_IMPUTE = fm.transform_impute("v", value=0, key="t", groupby=["f", "h"])


def _door_line(frame: pl.DataFrame = _DOOR_FRAME) -> fm.Chart:
    return (
        fm.Chart(frame, width=_W, height=_H).mark_line().encode(x="t", y="v", color="h", detail="f")
    )


def _door_point() -> fm.Chart:
    return fm.Chart(_DOOR_FRAME).mark_point().encode(x="t", y="v", color="h")


def _scene_vertices(chart: Any) -> list[list[float]]:
    panels = json.loads(chart.interactive().scene_json)["panels"]
    return [
        [x for x, _ in node["points"]]
        for panel in panels
        for batch in panel["marks"]
        for node in batch["nodes"]
        if node["type"] == "polyline"
    ]


# Each door's chart, the lines it draws, and how many of them hold the three vertices a complete
# series has: series x of both panels lack key 2 without the impute (2 vertices, so 2 of 4 hold three).
_DOORS: dict[str, tuple[Callable[[], Any], int, int]] = {
    "plain": (lambda: _door_line().transform(_DOOR_IMPUTE), 4, 4),
    "facet": (lambda: _door_line().transform(_DOOR_IMPUTE).facet("f"), 4, 4),
    "plus": (lambda: _door_line().transform(_DOOR_IMPUTE) + _door_point(), 4, 4),
    "Chart.layer": (
        lambda: (
            _door_line()
            .transform(_DOOR_IMPUTE)
            .layer(fm.Layer(mark="point", encoding={"x": "t", "y": "v", "color": "h"}))
        ),
        4,
        4,
    ),
    "LayerChart": (lambda: fm.layer(_door_line().transform(_DOOR_IMPUTE), _door_point()), 4, 4),
    # the second chart has no impute: its series x draw two vertices, its series y three
    "hconcat": (lambda: fm.hconcat(_door_line().transform(_DOOR_IMPUTE), _door_line()), 8, 6),
    "chained": (
        lambda: (
            _door_line()
            .transform(fm.transform_filter("datum.v >= 0"))
            .transform(_DOOR_IMPUTE)
            .transform(fm.transform_filter("datum.t < 99"))
        ),
        4,
        4,
    ),
}


@pytest.mark.parametrize("door", list(_DOORS))
def test_a_line_draws_its_imputed_vertex_in_key_order_on_every_door(door: str):
    build, series, complete = _DOORS[door]
    drawn = _series_x(build().to_svg())
    assert len(drawn) == series, drawn
    assert len([xs for xs in drawn if len(xs) == 3]) == complete, drawn
    assert all(xs == sorted(xs) for xs in drawn), drawn


@pytest.mark.parametrize("door", ["plain", "facet", "chained"])
def test_an_interactive_line_draws_its_imputed_vertex_in_key_order(door: str):
    build, series, complete = _DOORS[door]
    drawn = _scene_vertices(build())
    assert len(drawn) == series, drawn
    assert len([xs for xs in drawn if len(xs) == 3]) == complete, drawn
    assert all(xs == sorted(xs) for xs in drawn), drawn


# ---------------------------------------------------------------------------
# keyvals: added to the observed keys
# ---------------------------------------------------------------------------


def test_keyvals_add_to_the_observed_keys_and_never_replace_them():
    # key 3 is observed (in group x) though keyvals names only 1 and 2: group y still gains it
    frame = _keyed(
        "Int64", [("x", 0, 10.0), ("x", 1, 20.0), ("x", 2, 30.0), ("y", 0, 1.0), ("y", 1, 2.0)]
    )
    assert _rows(frame, _impute(value=-1, keyvals=[1, 2])) == [
        ("x", "10"),
        ("x", "20"),
        ("x", "30"),
        ("y", "1"),
        ("y", "2"),
        ("y", "-1"),
    ]


def test_keyvals_naming_keys_nobody_holds_are_added_to_every_group_in_key_order():
    frame = pl.DataFrame(
        {"g": ["x", "x", "y", "y"], "t": [1, 2, 1, 2], "v": [10.0, 20.0, 1.0, 2.0]}
    )
    expected = [
        ("x", "-1"),
        ("x", "10"),
        ("x", "20"),
        ("x", "-1"),
        ("y", "-1"),
        ("y", "1"),
        ("y", "2"),
        ("y", "-1"),
    ]
    assert _rows(frame, _impute(value=-1, keyvals=[0, 3])) == expected
    # the order keyvals is written in does not matter to a numeric key
    assert _rows(frame, _impute(value=-1, keyvals=[3, 0])) == expected


def test_a_text_keys_keyvals_come_first_in_its_key_order():
    frame = pl.DataFrame({"g": ["x", "x"], "t": ["a", "b"], "v": [1.0, 2.0]})
    # the order is z, a, b: z lands before a, the first row of x whose key is later
    step = _impute(value=-1, keyvals=["z", "a"])
    assert _rows(frame, step, groups=("x",)) == [("x", "-1"), ("x", "1"), ("x", "2")]
    # keyvals leave the first-appearance order of the keys it does not name where they were
    step = _impute(value=-1, keyvals=["b", "z"])
    frame = pl.DataFrame({"g": ["x", "x"], "t": ["a", "c"], "v": [1.0, 2.0]})
    # the order is b, z, a, c: b and z are before every row of x, so both land at the start
    assert _rows(frame, step, groups=("x",)) == [("x", "-1"), ("x", "-1"), ("x", "1"), ("x", "2")]


_KEYVALS_DATES = [dt.date(2024, 1, 3), dt.date(2024, 1, 7)]


def test_keyvals_of_dates_are_added_to_a_date_key_in_time_order():
    frame = pl.DataFrame(
        {
            "g": ["x", "x"],
            "t": [dt.date(2024, 1, 1), dt.date(2024, 1, 5)],
            "v": [10.0, 50.0],
        }
    )
    step = _impute(value=-1, keyvals=_KEYVALS_DATES)
    # 1 Jan, 3 Jan, 5 Jan, 7 Jan: the two imputed rows each land at their place
    assert _rows(frame, step, groups=("x",)) == [("x", "10"), ("x", "-1"), ("x", "50"), ("x", "-1")]


@pytest.mark.parametrize(
    "spelling",
    [
        lambda d: d,
        lambda d: dt.datetime(d.year, d.month, d.day),
        lambda d: dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc),
        lambda d: np.datetime64(d.isoformat()),
    ],
    ids=["date", "datetime", "aware datetime", "numpy datetime64"],
)
def test_every_spelling_of_a_date_keyval_rides_the_wire_as_the_same_tagged_instant(spelling):
    wire = fm.transform_impute("v", key="t", keyvals=[spelling(d) for d in _KEYVALS_DATES])[
        "keyvals"
    ]
    assert wire == [_instant(1704240000000), _instant(1704585600000)]
    assert all(type(instant["epoch_ms"]) is int for instant in wire)


def test_a_keyval_of_no_accepted_kind_is_refused_in_the_keyvals_voice():
    no_kind = object()
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", key="t", keyvals=[no_kind])
    assert str(excinfo.value) == (
        f"transform_impute: object {no_kind!r} not allowed in keyvals for key 't'; "
        "accepted: string, number, bool, date or datetime"
    )


_NAT = "np.datetime64('NaT') is NaT (not-a-time), which is not a valid temporal value"


@pytest.mark.parametrize(
    ("build", "refusal"),
    [
        (
            lambda: fm.transform_impute("v", key="t", keyvals=[np.datetime64("NaT")]),
            f"transform_impute: keyvals value for key 't': {_NAT}",
        ),
        (
            lambda: fm.transform_impute("v", value=np.datetime64("NaT")),
            f"transform_impute(value=...): {_NAT}",
        ),
        (
            lambda: fm.Y("v", impute={"value": np.datetime64("NaT")}),
            f"Y(impute=...) value: {_NAT}",
        ),
        (
            lambda: fm.Y("v", impute={"keyvals": [np.datetime64("NaT")]}),
            f"Y(impute=...): keyvals value: {_NAT}",
        ),
    ],
    ids=["keyval", "value", "encoding value", "encoding keyval"],
)
def test_a_not_a_time_is_refused_in_the_voice_of_the_door_that_got_it(
    build: Callable, refusal: str
):
    with pytest.raises(ValueError) as excinfo:
        build()
    assert str(excinfo.value) == refusal


# the sequence form: {'start', 'stop', 'step'}, stop exclusive, start 0 and step 1 (or -1 when
# stop < start) by default; the keys it makes are added to the observed ones
_SEQUENCES: list[tuple[dict, list[str]]] = [
    ({"start": 0, "stop": 5}, ["-1", "10", "-1", "30", "-1"]),
    ({"start": 5, "stop": 0}, ["10", "-1", "30", "-1", "-1"]),
    ({"stop": 3}, ["-1", "10", "-1", "30"]),
    ({"start": 0, "stop": 6, "step": 4}, ["-1", "10", "30", "-1"]),
    ({"start": 5, "stop": 0, "step": -2}, ["10", "30", "-1"]),
    ({"start": 0, "stop": 2, "step": 5}, ["-1", "10", "30"]),
]


@pytest.mark.parametrize(("sequence", "labels"), _SEQUENCES, ids=[str(s) for s, _ in _SEQUENCES])
def test_a_keyvals_sequence_adds_its_keys_to_the_observed_ones(sequence: dict, labels: list[str]):
    # observed keys 1 and 3
    frame = pl.DataFrame({"g": ["x", "x"], "t": [1, 3], "v": [10.0, 30.0]})
    rows = _rows(frame, _impute(value=-1, keyvals=sequence), groups=("x",))
    assert rows == [("x", label) for label in labels]


def test_a_keyvals_sequence_rides_the_wire_as_numbers():
    wire = fm.transform_impute(
        "v", key="t", keyvals={"start": np.int64(0), "stop": np.float64(5.5), "step": 1}
    )["keyvals"]
    assert wire == {"start": 0, "stop": 5.5, "step": 1}
    assert [type(n) for n in wire.values()] == [int, float, int]
    assert fm.transform_impute("v", key="t", keyvals={"stop": 3})["keyvals"] == {"stop": 3}


def test_a_keyvals_sequence_beside_a_group_with_every_key_changes_nothing():
    frame = pl.DataFrame({"g": ["x"] * 3, "t": [0, 1, 2], "v": [1.0, 2.0, 3.0]})
    step = _impute(value=-1, keyvals={"start": 0, "stop": 3})
    assert _rows(frame, step, groups=("x",)) == [("x", "1"), ("x", "2"), ("x", "3")]


@pytest.mark.parametrize(
    ("keyvals", "refusal"),
    [
        ({"start": 0}, "transform_impute: keyvals={...} needs 'stop'; got {'start': 0}"),
        (
            {"stop": 3, "by": 2},
            "transform_impute: keyvals={...} has unknown key 'by'; expected start|stop|step",
        ),
    ],
    ids=["no stop", "unknown key"],
)
def test_a_keyvals_sequence_that_is_not_one_is_refused_at_construction(keyvals: dict, refusal: str):
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", key="t", keyvals=keyvals)
    assert str(excinfo.value) == refusal


_NOT_REAL = "must be a real number (int, float, numpy scalar, or Decimal)"
_NOT_FINITE = "must be a finite number a float can hold"


@pytest.mark.parametrize(
    ("bad", "refusal"),
    [
        (float("nan"), f"{_NOT_FINITE}; received nan"),
        (float("inf"), f"{_NOT_FINITE}; received inf"),
        ("5", f"{_NOT_REAL}, not str; received '5'"),
        (True, f"{_NOT_REAL}, not bool; received True"),
        (None, f"{_NOT_REAL}, not NoneType; received None"),
        ([1], f"{_NOT_REAL}, not list; received [1]"),
    ],
    ids=["nan", "inf", "str", "bool", "None", "list"],
)
def test_a_keyvals_sequence_number_that_is_not_finite_is_refused_at_construction(
    bad: Any, refusal: str
):
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", key="t", keyvals={"start": 0, "stop": bad})
    assert str(excinfo.value) == f"transform_impute: keyvals stop {refusal}"


def test_keyvals_that_are_not_a_sequence_are_refused_at_construction():
    bad: Any = 5
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", key="t", keyvals=bad)
    assert str(excinfo.value) == (
        "transform_impute: keyvals must be a sequence of key values or a "
        "{'start', 'stop', 'step'} dict, got int 5"
    )


def test_a_zero_step_is_refused():
    frame = pl.DataFrame({"g": ["x"], "t": [1], "v": [10.0]})
    step = _impute(value=-1, keyvals={"start": 0, "stop": 5, "step": 0})
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, step).to_svg()
    assert str(excinfo.value) == _engine(
        "data_impute: keyvals={'start': ..., 'step': 0} never reaches stop; use a non-zero step"
    )


@pytest.mark.parametrize(
    ("key", "holds"),
    [(pl.Series("k", ["a", "b"]), "text"), (pl.Series("k", [True, False]), "booleans")],
    ids=["text", "boolean"],
)
def test_a_keyvals_sequence_on_a_key_that_is_not_a_number_is_refused(key: pl.Series, holds: str):
    frame = pl.DataFrame({"g": ["x", "x"], "v": [10.0, 20.0]}).with_columns(key)
    step = fm.transform_impute("v", key="k", groupby=["g"], keyvals={"start": 0, "stop": 3})
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, step).to_svg()
    assert str(excinfo.value) == _engine(
        f"data_impute: keyvals={{'start': ...}} makes numbers, but the key 'k' holds {holds}; "
        "list the keys instead"
    )


# ---------------------------------------------------------------------------
# value: the column's own kind, default 0
# ---------------------------------------------------------------------------


def test_no_value_fills_with_zero():
    frame = pl.DataFrame({"g": ["x", "x", "x"], "t": [1, 2, 3], "v": [1.0, None, 3.0]})
    assert _rows(frame, fm.transform_impute("v"), groups=("x",)) == [
        ("x", "1"),
        ("x", "0"),
        ("x", "3"),
    ]
    assert fm.transform_impute("v") == fm.transform_impute("v", value=None)
    assert "value" not in fm.transform_impute("v")


def test_a_missing_row_and_a_null_cell_take_the_same_default_fill():
    frame = pl.DataFrame({"g": ["x", "x"], "t": [1, 3], "v": [None, 3.0]})
    step = fm.transform_impute("v", key="t", groupby=["g"], keyvals=[2])
    assert _rows(frame, step, groups=("x",)) == [("x", "0"), ("x", "0"), ("x", "3")]


@pytest.mark.parametrize("method", ["mean", "median", "min", "max"])
def test_a_value_beside_a_statistic_is_refused(method: str):
    frame = pl.DataFrame({"g": ["x"], "t": [1], "v": [1.0]})
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, fm.transform_impute("v", method=method, value=0)).to_svg()
    assert str(excinfo.value) == _engine(
        f"data_impute: value= applies to method='value', not '{method}'; "
        "drop value= or set method='value'"
    )


def test_a_value_beside_the_default_method_is_filled_not_refused():
    frame = pl.DataFrame({"g": ["x", "x"], "t": [1, 2], "v": [1.0, None]})
    step = fm.transform_impute("v", method="value", value=7)
    assert _rows(frame, step, groups=("x",)) == [("x", "1"), ("x", "7")]


_WHEN = [dt.date(2024, 1, 1), None, dt.date(2024, 1, 3)]
_DAY2 = dt.date(2024, 1, 2)
_TWIN_WHEN = [dt.date(2024, 1, 1), _DAY2, dt.date(2024, 1, 3)]


def _kind_chart(frame: pl.DataFrame, *steps: dict, column: str) -> str:
    chart = fm.Chart(frame, width=_W, height=_H)
    for step in steps:
        chart = chart.transform(step)
    return chart.mark_point().encode(x="t", y=column).to_svg()


@pytest.mark.parametrize(
    "value",
    [
        _DAY2,
        dt.datetime(2024, 1, 2),
        dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc),
        np.datetime64("2024-01-02"),
        1704153600000,
        1704153600000.0,
    ],
    ids=[
        "date",
        "datetime",
        "aware datetime",
        "numpy datetime64",
        "epoch ms int",
        "epoch ms float",
    ],
)
def test_a_date_value_fills_a_date_column_at_its_own_dtype(value: Any):
    frame = pl.DataFrame({"t": [1, 2, 3], "when": _WHEN})
    twin = pl.DataFrame({"t": [1, 2, 3], "when": _TWIN_WHEN})
    step = fm.transform_impute("when", value=value)
    svg = _kind_chart(frame, step, column="when")
    assert svg == _kind_chart(twin, column="when")
    assert svg != _kind_chart(frame, column="when")


def test_a_date_value_fills_a_missing_key_row_of_a_date_column():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y"],
            "t": [1, 3, 2],
            "when": [dt.date(2024, 1, 1), dt.date(2024, 1, 3), None],
        }
    )
    # x gains key 2, y gains keys 1 and 3 and fills its null: every filled cell is 2 Jan
    twin = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y", "y"],
            "t": [1, 2, 3, 1, 2, 3],
            "when": [dt.date(2024, 1, 1), _DAY2, dt.date(2024, 1, 3), _DAY2, _DAY2, _DAY2],
        }
    )
    step = fm.transform_impute("when", value=_DAY2, key="t", groupby=["g"])
    svg = _kind_chart(frame, step, column="when")
    assert svg == _kind_chart(twin, column="when")
    assert svg != _kind_chart(frame, column="when")


def test_a_text_value_fills_a_text_column():
    frame = pl.DataFrame({"g": ["x"] * 3, "t": [1, 2, 3], "grade": ["a", None, "c"]})
    twin = frame.with_columns(grade=pl.Series(["a", "none", "c"]))
    step = fm.transform_impute("grade", value="none")
    assert _kind_text(frame, step, column="grade") == _kind_text(twin, column="grade")
    assert _kind_text(frame, step, column="grade") != _kind_text(frame, column="grade")


def _kind_text(frame: pl.DataFrame, *steps: dict, column: str) -> str:
    return _text_chart(frame, *steps, text=column, x="t").to_svg()


def test_a_text_value_fills_a_missing_key_row_of_a_text_column():
    frame = pl.DataFrame(
        {"g": ["x", "x", "y", "y", "y"], "t": [1, 3, 1, 2, 3], "grade": ["a", "c", "d", "e", "f"]}
    )
    twin = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y", "y"],
            "t": [1, 2, 3, 1, 2, 3],
            "grade": ["a", "none", "c", "d", "e", "f"],
        }
    )
    step = fm.transform_impute("grade", value="none", key="t", groupby=["g"])
    assert _kind_text(frame, step, column="grade") == _kind_text(twin, column="grade")


def _boolean_points(frame: pl.DataFrame, *steps: dict) -> str:
    """A chart that draws a Boolean column: ``mark_text`` refuses the dtype, a colour reads it.
    Every point the frame holds is drawn, so a twin that cannot render fails here, loudly."""
    chart = fm.Chart(frame, width=_W, height=_H)
    for step in steps:
        chart = chart.transform(step)
    svg = chart.mark_point().encode(x="t", y="t", color="ok").to_svg()
    assert svg.count("<circle") >= 1, "the chart drew no point"
    return svg


def test_a_boolean_value_fills_a_boolean_column():
    frame = pl.DataFrame({"g": ["x"] * 3, "t": [1, 2, 3], "ok": [True, None, False]})
    twin = frame.with_columns(ok=pl.Series([True, False, False]))
    step = fm.transform_impute("ok", value=False)
    assert _boolean_points(frame, step) == _boolean_points(twin)
    assert _boolean_points(frame, step) != _boolean_points(frame)
    other = frame.with_columns(ok=pl.Series([True, True, False]))
    assert _boolean_points(frame, fm.transform_impute("ok", value=True)) == _boolean_points(other)
    assert _boolean_points(frame, step) != _boolean_points(other)


def test_a_boolean_value_fills_a_missing_key_row_of_a_boolean_column():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y", "y", "y"],
            "t": [1, 3, 1, 2, 3],
            "ok": [True, True, False, True, False],
        }
    )
    twin = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y", "y"],
            "t": [1, 2, 3, 1, 2, 3],
            "ok": [True, False, True, False, True, False],
        }
    )
    step = fm.transform_impute("ok", value=False, key="t", groupby=["g"])
    svg = _boolean_points(frame, step)
    assert svg == _boolean_points(twin)
    assert svg != _boolean_points(
        twin.with_columns(ok=pl.Series([True, True, True, False, True, False]))
    )


def test_a_number_value_fills_an_integer_column_at_the_value_it_was_given():
    # an integer column is read and written as a float: 2.5 is drawn at 2.5
    frame = pl.DataFrame({"g": ["x", "x"], "t": [1, 2], "v": pl.Series([4, None], dtype=pl.Int64)})
    assert _rows(frame, fm.transform_impute("v", value=2.5), groups=("x",)) == [
        ("x", "4"),
        ("x", "2.5"),
    ]


_KIND_COLUMNS = {
    "number": ("v", [1.0, None]),
    "date": ("when", [dt.date(2024, 1, 1), None]),
    "text": ("grade", ["a", None]),
    "boolean": ("ok", [True, None]),
}


@pytest.mark.parametrize("case", list(IMPUTE_KIND_CASES))
def test_a_value_of_another_kind_than_its_column_is_refused_naming_both(case: str):
    column, values = _KIND_COLUMNS[case.split("-")[0]]
    value = IMPUTE_KIND_CASES[case][0]
    frame = pl.DataFrame({"t": [1, 2], column: values})
    chart = fm.Chart(frame).transform(fm.transform_impute(column, value=value))
    with pytest.raises(ValueError) as excinfo:
        chart.mark_point().encode(x="t", y="t").to_svg()
    assert str(excinfo.value) == _engine(_impute_kind_refusal("data_impute", column, case))


@pytest.mark.parametrize("method", ["mean", "median", "min", "max"])
def test_a_statistic_on_a_text_column_is_refused_as_it_was(method: str):
    frame = pl.DataFrame({"t": [1, 2], "grade": ["a", None]})
    chart = fm.Chart(frame).transform(fm.transform_impute("grade", method=method))
    with pytest.raises(ValueError) as excinfo:
        chart.mark_point().encode(x="t", y="t").to_svg()
    assert str(excinfo.value) == _engine("data_impute: column 'grade' must be numeric, got Utf8")


def test_an_unknown_method_is_refused_as_it_was():
    frame = pl.DataFrame({"t": [1], "v": [1.0]})
    chart = fm.Chart(frame).transform(fm.transform_impute("v", method="bogus"))
    with pytest.raises(ValueError) as excinfo:
        chart.mark_point().encode(x="t", y="v").to_svg()
    assert str(excinfo.value) == _engine(
        f"data_impute: unknown method 'bogus'; expected {_METHODS}"
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_value_that_is_not_finite_is_refused_as_it_was(bad: float):
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", value=bad)
    assert str(excinfo.value) == (
        f"transform_impute(value=...) must be a finite number a float can hold; received {bad!r}"
    )


@pytest.mark.parametrize(
    "bad", [[1], {"a": 1}, (1,), object()], ids=["list", "dict", "tuple", "object"]
)
def test_a_value_of_no_kind_a_column_holds_is_refused_naming_the_kinds_it_takes(bad: Any):
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", value=bad)
    assert str(excinfo.value) == (
        "transform_impute(value=...) must be a number, a string, True or False, a date or a "
        f"datetime, not {type(bad).__name__}; received {bad!r}"
    )


def test_a_key_the_engine_cannot_read_is_refused_as_it_was():
    frame = pl.DataFrame({"g": ["x"], "v": [1.0], "k": [dt.time(1)]})
    step = fm.transform_impute("v", key="k", groupby=["g"])
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, step).to_svg()
    assert str(excinfo.value) == _group_dtype_refusal(
        "data_impute", "key", "k", "Time64(Nanosecond)"
    )


# ---------------------------------------------------------------------------
# What a cell is: null and NaN are filled, an infinity is a value
# ---------------------------------------------------------------------------


def test_a_null_and_a_nan_cell_are_filled_and_an_infinite_cell_is_kept():
    frame = pl.DataFrame(
        {
            "g": ["x"] * 4,
            "t": [1, 2, 3, 4],
            "v": [1.0, None, float("nan"), float("inf")],
        }
    )
    # the infinite cell stays an infinity, which draws no text
    assert _rows(frame, fm.transform_impute("v", value=-1), groups=("x",)) == [
        ("x", "1"),
        ("x", "-1"),
        ("x", "-1"),
    ]


def test_a_statistic_fills_from_the_finite_values_and_leaves_an_infinite_cell():
    frame = pl.DataFrame({"g": ["x"] * 4, "t": [1, 2, 3, 4], "v": [1.0, float("inf"), None, 3.0]})
    assert _rows(frame, fm.transform_impute("v", method="mean"), groups=("x",)) == [
        ("x", "1"),
        ("x", "2"),
        ("x", "3"),
    ]


def test_a_nan_key_is_a_key_of_its_own_completed_into_every_group():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y"],
            "t": [1.0, 2.0, float("nan"), 1.0, 2.0],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    # x's own NaN-key row and the one y gains are both drawn at their value (x is the value here)
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "1"),
        ("x", "2"),
        ("x", "3"),
        ("y", "4"),
        ("y", "5"),
        ("y", "-1"),
    ]


@pytest.mark.parametrize("infinity", [float("inf"), float("-inf")], ids=["+inf", "-inf"])
def test_an_infinite_key_is_a_key_of_its_own_completed_into_every_group(infinity: float):
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y", "y", "y"],
            "t": [1.0, 2.0, 1.0, 2.0, infinity],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    # x gains the infinite key: after its last row for +inf, before its first for -inf
    placed = (
        [("x", "1"), ("x", "2"), ("x", "-1")]
        if infinity > 0
        else [("x", "-1"), ("x", "1"), ("x", "2")]
    )
    assert _rows(frame, _impute(value=-1), text="v") == placed + [
        ("y", "3"),
        ("y", "4"),
        ("y", "5"),
    ]


def test_a_null_key_is_no_key_and_is_completed_into_no_group():
    frame = pl.DataFrame(
        {"g": ["y", "y", "x", "x"], "t": [1.0, 2.0, None, 1.0], "v": [3.0, 4.0, 2.0, 1.0]}
    )
    # x keeps its null-key row and gains key 2; the null key is not added to y
    assert _rows(frame, _impute(value=-1), groups=("y", "x")) == [
        ("y", "3"),
        ("y", "4"),
        ("x", "2"),
        ("x", "1"),
        ("x", "-1"),
    ]


# ---------------------------------------------------------------------------
# frame: the windowed statistic
# ---------------------------------------------------------------------------

_FRAME_FRAME = pl.DataFrame(
    {"g": ["x"] * 5, "t": [1, 2, 3, 4, 5], "v": [10.0, None, 40.0, 100.0, None]}
)


def _window(method: str, frame: tuple[int | None, int | None] | None, **kwargs: Any) -> dict:
    return fm.transform_impute("v", method=method, key="t", groupby=["g"], frame=frame, **kwargs)


@pytest.mark.parametrize(
    ("method", "labels"),
    [
        ("mean", ["10", "50", "40", "100", "70"]),
        ("median", ["10", "40", "40", "100", "70"]),
        ("min", ["10", "10", "40", "100", "40"]),
        ("max", ["10", "100", "40", "100", "100"]),
    ],
)
def test_a_frame_gives_each_fill_the_statistic_of_the_usable_values_in_its_window(
    method: str, labels: list[str]
):
    # rows two either side: key 2 sees keys 1..4 (10, 40, 100), key 5 sees keys 3..5 (40, 100)
    rows = _rows(_FRAME_FRAME, _window(method, (-2, 2)), groups=("x",))
    assert rows == [("x", label) for label in labels]


def test_a_frame_with_no_bound_is_the_whole_groups_statistic():
    whole = _rows(_FRAME_FRAME, _window("mean", None), groups=("x",))
    assert whole == _rows(_FRAME_FRAME, _window("mean", (None, None)), groups=("x",))
    assert whole == _rows(
        _FRAME_FRAME, fm.transform_impute("v", method="mean", key="t", groupby=["g"]), groups=("x",)
    )
    assert [label for _, label in whole] == ["10", "50", "40", "100", "50"]


@pytest.mark.parametrize(
    ("frame", "labels"),
    [
        ((None, 0), ["10", "10", "40", "100", "50"]),
        ((-1, 1), ["10", "25", "40", "100", "100"]),
        ((0, None), ["10", "70", "40", "100", "50"]),
    ],
    ids=["cumulative", "one either side", "from here on"],
)
def test_a_frame_with_an_open_end_reads_to_the_groups_edge(
    frame: tuple[int | None, int | None], labels: list[str]
):
    # (0, None) at key 5 sees only key 5, which holds no usable value: the group's mean, 50
    rows = _rows(_FRAME_FRAME, _window("mean", frame), groups=("x",))
    assert rows == [("x", label) for label in labels]


def test_a_frame_counts_in_key_order_not_row_order():
    shuffled = _FRAME_FRAME[[3, 0, 4, 2, 1]]
    expected = {"1": "10", "2": "50", "3": "40", "4": "100", "5": "70"}
    rows = _rows(shuffled, _window("mean", (-2, 2)), groups=("x",), text="t")
    # the rows keep the order they came in, and each is the filled value of its own key
    assert [label for _, label in rows] == ["4", "1", "5", "3", "2"]
    values = _rows(shuffled, _window("mean", (-2, 2)), groups=("x",))
    assert [label for _, label in values] == [expected[k] for k in ("4", "1", "5", "3", "2")]


def test_an_imputed_row_counts_in_a_frame_as_a_row_holding_no_value():
    # key 3 is missing (keyvals) and key 4 is null: key 4 sees keys 3..5, where 3 holds nothing
    frame = pl.DataFrame({"g": ["x"] * 4, "t": [1, 2, 4, 5], "v": [10.0, 20.0, None, 50.0]})
    step = _window("mean", (-1, 1), keyvals=[3])
    assert _rows(frame, step, groups=("x",)) == [
        ("x", "10"),
        ("x", "20"),
        ("x", "20"),
        ("x", "50"),
        ("x", "50"),
    ]


def test_a_frame_counts_each_group_apart():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "x", "y", "y", "y"],
            "t": [1, 2, 3, 1, 2, 3],
            "v": [10.0, None, 30.0, 100.0, None, 300.0],
        }
    )
    assert _rows(frame, _window("mean", (-1, 1))) == [
        ("x", "10"),
        ("x", "20"),
        ("x", "30"),
        ("y", "100"),
        ("y", "200"),
        ("y", "300"),
    ]


def test_a_frame_with_no_key_counts_in_the_groups_row_order():
    frame = pl.DataFrame({"g": ["x"] * 4, "v": [10.0, None, 30.0, None]})
    step = fm.transform_impute("v", method="mean", groupby=["g"], frame=(-1, 0))
    chart = (
        fm.Chart(frame, width=_W, height=_H)
        .transform(step)
        .mark_text()
        .encode(x="v", y="g:N", text="v")
    )
    assert [label for _, label in _marks(chart.to_svg())] == ["10", "10", "30", "30"]


def test_a_frame_whose_window_holds_no_usable_value_takes_the_groups_statistic_with_a_notice():
    frame = pl.DataFrame({"g": ["x"] * 4, "t": [1, 2, 3, 4], "v": [10.0, None, None, 40.0]})
    step = _window("mean", (-1, 0))
    chart = _text_chart(frame, step)
    svg, messages = _render_capturing(chart.to_svg)
    # key 2 sees keys 1, 2: 10. Key 3 sees keys 2, 3: nothing, so the group's mean, 25
    assert [label for _, label in _marks(svg)] == ["10", "10", "25", "40"]
    assert messages == [_frame_fallback("g=x", 1)]


def test_a_frame_in_a_group_with_no_usable_value_takes_the_whole_columns_statistic_with_a_notice():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y", "y"],
            "t": [1, 2, 1, 2],
            "v": [None, None, 10.0, 30.0],
        }
    )
    svg, messages = _render_capturing(_text_chart(frame, _window("mean", (-1, 1))).to_svg)
    assert [label for _, label in _marks(svg)] == ["20", "20", "10", "30"]
    assert messages == [
        "Impute: group g=x has no usable (non-null, finite) values; the whole column's statistic "
        "was used instead"
    ]


def test_a_frame_over_a_column_with_no_usable_value_takes_the_degenerate_default_with_a_notice():
    frame = pl.DataFrame(
        {"g": ["x", "x"], "t": [1, 2], "v": pl.Series([None, None], dtype=pl.Float64)}
    )
    svg, messages = _render_capturing(_text_chart(frame, _window("mean", (-1, 1))).to_svg)
    assert [label for _, label in _marks(svg)] == ["0", "0"]
    assert messages == [
        "Impute: the column has no usable (non-null, finite) values at all; used the mean "
        "method's degenerate default instead",
        "Impute: group g=x has no usable (non-null, finite) values; the whole column's statistic "
        "was used instead",
    ]


@pytest.mark.parametrize(
    "build",
    [
        lambda: fm.transform_impute("v", value=0, frame=(-1, 1)),
        lambda: fm.transform_impute("v", method="value", frame=(-1, 1)),
        lambda: fm.transform_impute("v", frame=(None, None)),
    ],
    ids=["with a value", "with the method named", "with the default method"],
)
def test_a_frame_beside_a_constant_fill_is_refused(build: Callable[[], dict]):
    frame = pl.DataFrame({"g": ["x"], "t": [1], "v": [1.0]})
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, build()).to_svg()
    assert str(excinfo.value) == _engine(
        "data_impute: frame= applies to mean, median, min and max; method 'value' fills a constant"
    )


def test_a_frame_rides_the_wire_as_a_pair_with_none_for_an_open_end():
    assert fm.transform_impute("v", method="mean", frame=(-1, 1))["frame"] == [-1, 1]
    assert fm.transform_impute("v", method="mean", frame=(None, 0))["frame"] == [None, 0]
    numpy_frame: Any = [np.int64(-2), None]
    assert fm.transform_impute("v", method="mean", frame=numpy_frame)["frame"] == [-2, None]
    assert "frame" not in fm.transform_impute("v", method="mean")
    assert "frame" not in fm.transform_impute("v", method="mean", frame=None)


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3), "ab", (1.5, 2), (True, 1), 3, (-1, "x")])
def test_a_frame_that_is_not_a_pair_of_offsets_is_refused_at_construction(bad: Any):
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", method="mean", frame=bad)
    assert str(excinfo.value) == (
        "transform_impute: frame must be a (preceding, following) pair of row offsets, "
        f"each an integer or None; got {bad!r}"
    )


# ---------------------------------------------------------------------------
# Unchanged: a call whose imputed rows are already where they belong, and a chart with no impute
# ---------------------------------------------------------------------------

_NAN, _INF = float("nan"), float("inf")


def _line_over(frame: pl.DataFrame, *steps: dict, color: str | None = "h") -> fm.Chart:
    chart = fm.Chart(frame, width=_W, height=_H)
    for step in steps:
        chart = chart.transform(step)
    return chart.mark_line().encode(x="t", y="v", color=color)


def _point_over(frame: pl.DataFrame, *steps: dict) -> fm.Chart:
    chart = fm.Chart(frame, width=_W, height=_H)
    for step in steps:
        chart = chart.transform(step)
    return chart.mark_point().encode(x="t", y="v", color="h")


# The last group lacks the last key: the row an imputing call appends is where it is inserted.
_LAST = pl.DataFrame(
    {"t": [1, 2, 3, 1, 2], "h": ["x", "x", "x", "y", "y"], "v": [1.0, 2.0, 3.0, 4.0, 5.0]}
)
# Group x is last and lacks the NaN key group y holds.
_NAN_LAST = pl.DataFrame(
    {
        "t": [1.0, 2.0, _NAN, 1.0, 2.0],
        "h": ["y", "y", "y", "x", "x"],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)
_INF_LAST = _NAN_LAST.with_columns(t=pl.Series([1.0, 2.0, _INF, 1.0, 2.0]))
# Group y, the last, lacks key 4 (which the keyvals add): its row is appended and inserted alike.
_SUPERSET_LAST = pl.DataFrame(
    {
        "t": [1, 2, 3, 4, 1, 2, 3],
        "h": ["x"] * 4 + ["y"] * 3,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)
# One group lacking its last key.
_ONE_GROUP = pl.DataFrame({"t": [1, 2], "h": ["x", "x"], "v": [1.0, 2.0]})
# Nulls and NaNs in place, no key: nothing is added, so nothing can move.
_GAPPY = pl.DataFrame(
    {
        "t": [1, 2, 3, 4, 1, 2, 3, 4],
        "h": ["x"] * 4 + ["y"] * 4,
        "v": [1.0, None, 3.0, _NAN, None, None, 7.0, 8.0],
    }
)
_ALL_NULL_GROUP = _GAPPY.with_columns(
    v=pl.Series([1.0, None, 3.0, 4.0, None, None, None, None], dtype=pl.Float64)
)
_FACETED = pl.concat(
    [_LAST.with_columns(f=pl.lit("a")), _LAST.with_columns(f=pl.lit("b"), v=pl.col("v") + 10)]
)
_PLAIN = pl.DataFrame(
    {"t": [1, 2, 3, 1, 2, 3], "h": ["x"] * 3 + ["y"] * 3, "v": [1.0, 3.0, 2.0, 2.0, 1.0, 4.0]}
)

_UNCHANGED: dict[str, Callable[[], fm.Chart]] = {
    "line, last key missing": lambda: _line_over(
        _LAST, fm.transform_impute("v", value=0, key="t", groupby=["h"])
    ),
    "point, last key missing in the last group": lambda: _point_over(
        _LAST, fm.transform_impute("v", value=0, key="t", groupby=["h"])
    ),
    "point, mean fill, last key missing": lambda: _point_over(
        _LAST, fm.transform_impute("v", method="mean", key="t", groupby=["h"])
    ),
    "point, keyvals a superset, the imputed group ends the frame": lambda: _point_over(
        _SUPERSET_LAST,
        fm.transform_impute("v", value=0, key="t", groupby=["h"], keyvals=[1, 2, 3, 4]),
    ),
    "point, one group lacking its last key": lambda: _point_over(
        _ONE_GROUP, fm.transform_impute("v", value=0, key="t", groupby=["h"], keyvals=[1, 2, 3])
    ),
    "point, one group with no groupby lacking its last key": lambda: _point_over(
        _ONE_GROUP, fm.transform_impute("v", value=0, key="t", keyvals=[1, 2, 3])
    ),
    "point, no key missing": lambda: _point_over(
        _PLAIN, fm.transform_impute("v", value=0, key="t", groupby=["h"])
    ),
    "point, a NaN key held by the first group": lambda: _point_over(
        _NAN_LAST, fm.transform_impute("v", value=0, key="t", groupby=["h"])
    ),
    "point, an infinite key held by the first group": lambda: _point_over(
        _INF_LAST, fm.transform_impute("v", value=0, key="t", groupby=["h"])
    ),
    "point, median fill of nulls and NaNs in place": lambda: _point_over(
        _GAPPY, fm.transform_impute("v", method="median", groupby=["h"])
    ),
    "point, min fill, a group with no usable value": lambda: _point_over(
        _ALL_NULL_GROUP, fm.transform_impute("v", method="min", groupby=["h"])
    ),
    "point, max fill ungrouped": lambda: _point_over(
        _GAPPY, fm.transform_impute("v", method="max")
    ),
    "point, default fill of nulls in place": lambda: _point_over(_GAPPY, fm.transform_impute("v")),
    "line, faceted, last key missing": lambda: _line_over(
        _FACETED, fm.transform_impute("v", value=0, key="t", groupby=["f", "h"])
    ).facet("f"),
    "no impute, line": lambda: _line_over(_PLAIN),
    "no impute, point": lambda: _point_over(_PLAIN),
    "no impute, faceted line": lambda: _line_over(_FACETED).facet("f"),
    "no impute, an unset impute on y": lambda: (
        fm.Chart(_PLAIN, width=_W, height=_H)
        .mark_line()
        .encode(x="t", y=fm.Y("v", impute=None), color="h")
    ),
    "no impute, area": lambda: (
        fm.Chart(_PLAIN, width=_W, height=_H).mark_area().encode(x="t", y="v", color="h")
    ),
}

# Measured on the build before this rule existed (the engine as released in v0.24.0 plus the date
# group keys): each digest is the svg's, and the interactive scene's, with the ordered warnings.
_UNCHANGED_DIGESTS: dict[str, str] = {
    "line, last key missing": "064693e376a3cc6c19a063e373bef2ea530542c49ce3719e7c1623074a3eb98b",
    "point, last key missing in the last group": "03ee704106c02113116a87ebdd7f723da1a5364154fbcbe7382f5093165354ca",
    "point, mean fill, last key missing": "184cdca14b7a311d6cec564ab3113566ce3b686d00810d71f60ebe5216bea06b",
    "point, keyvals a superset, the imputed group ends the frame": "7ec46a77d4c4a40bef402da5100a16393818b57c926e0f6c70c3deaa6b284afd",
    "point, one group lacking its last key": "2f878dd801086dda2b667ddef5208d8196547a9b82f997f799434968e00105f2",
    "point, one group with no groupby lacking its last key": "e68c047307aed1b513538468d19c84262a18fcdd7e957b879359679786a873f9",
    "point, no key missing": "6a4a3b604b41469f8e67653d45b6c80e7652c368c398cc8ae52f13d54103ad79",
    "point, a NaN key held by the first group": "a8443be15576487e77eaab7bbcd27c2f0b7e648f7019ee82b0da1f6846a7ef30",
    "point, an infinite key held by the first group": "a8443be15576487e77eaab7bbcd27c2f0b7e648f7019ee82b0da1f6846a7ef30",
    "point, median fill of nulls and NaNs in place": "a6a48e170c484a978f9f9600a6e26d9f139d014cc92f4b48cdd0d79c51776556",
    "point, min fill, a group with no usable value": "6525131e7904f7ede5978b4092c42e4f90659e07bb647d7dec6d7eb475374c43",
    "point, max fill ungrouped": "96a9ce2eb8819befa462545188e9ac89770de58cda2b2ef49c9e8abe486b81d4",
    "point, default fill of nulls in place": "8715f655412baa88d5915f66b8c83da0f3d203a6ccda3df2797b095607bd94ac",
    "line, faceted, last key missing": "ab138e29b156b5007e84a170b8445cb4b6ce3f4e6fe83bf0d141d8c82f1a00ad",
    "no impute, line": "bc15de21a705cc3920bdb5b1d4d5cc66c60071bb44613ee049c37c6b9ed9d5f2",
    "no impute, point": "6a4a3b604b41469f8e67653d45b6c80e7652c368c398cc8ae52f13d54103ad79",
    "no impute, faceted line": "2a0c2baf0a3caf29cab9701a9856c0b88acec593426e9b06f2ab3a6892c6b6a9",
    "no impute, an unset impute on y": "bc15de21a705cc3920bdb5b1d4d5cc66c60071bb44613ee049c37c6b9ed9d5f2",
    "no impute, area": "6eb9cd9544270f8691a8a407afe900edaf700436ba0d92a81d380ffba41d6b53",
}


def _render_digest(chart: fm.Chart) -> str:
    svg, svg_messages = _render_capturing(chart.to_svg)
    scene, scene_messages = _render_capturing(lambda: chart.interactive().scene_json)
    return _digest(json.dumps([svg, svg_messages, scene, scene_messages]))


@pytest.mark.parametrize("case", list(_UNCHANGED))
def test_a_call_whose_imputed_rows_are_already_in_place_and_a_chart_with_no_impute_draw_as_before(
    case: str,
):
    assert _render_digest(_UNCHANGED[case]()) == _UNCHANGED_DIGESTS[case]


def test_a_row_imputed_in_a_group_that_does_not_end_the_frame_is_inserted_in_its_own_group():
    # x lacks key 4 (added by keyvals) and y lacks keys 3 and 4. Before rows were inserted, every
    # imputed row came after every caller row; now x's follows x's last row.
    step = fm.transform_impute("v", value=0, key="t", groupby=["h"], keyvals=[1, 2, 3, 4])
    assert [(g, v) for g, v in _output_rows(_LAST, step)] == [
        ("x", "1"),
        ("x", "2"),
        ("x", "3"),
        ("x", "0"),
        ("y", "4"),
        ("y", "5"),
        ("y", "0"),
        ("y", "0"),
    ]
    # the drawn element order is the written-out frame's, and not the appended order's
    inserted = pl.DataFrame(
        {
            "t": [1, 2, 3, 4, 1, 2, 3, 4],
            "h": ["x"] * 4 + ["y"] * 4,
            "v": [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0],
        }
    )
    appended = pl.concat(
        [_LAST, pl.DataFrame({"t": [4, 3, 4], "h": ["x", "y", "y"], "v": [0.0] * 3})]
    )
    svg = _point_over(_LAST, step).to_svg()
    assert svg == _point_over(inserted).to_svg()
    assert svg != _point_over(appended).to_svg()


def _output_rows(frame: pl.DataFrame, step: dict) -> list[tuple[str, str]]:
    """The rows ``step`` leaves of ``frame`` as ``(group h, value)``, in order, read off a text
    chart that puts the group on y and the value on x."""
    chart = (
        fm.Chart(frame, width=_W, height=_H)
        .transform(step)
        .mark_text()
        .encode(x="v", y="h:N", text="v")
    )
    marks = _marks(chart.to_svg())
    bands = sorted({y for y, _ in marks})
    return [(("x", "y")[bands.index(y)], label) for y, label in marks]


# ---------------------------------------------------------------------------
# The wire, and the construction of the encoding dict
# ---------------------------------------------------------------------------


def test_the_transform_spec_keeps_its_keys_in_their_order():
    spec = fm.transform_impute(
        "v", method="mean", value=None, groupby=["g"], key="t", keyvals=[1, 2], frame=(-1, 1)
    )
    assert list(spec) == ["type", "field", "method", "groupby", "key", "keyvals", "frame"]
    assert list(fm.transform_impute("v", value=3)) == ["type", "field", "method", "value"]


def test_an_unset_method_is_the_default_method():
    assert fm.transform_impute("v", method=None)["method"] == "value"  # type: ignore[arg-type]
    assert fm.Y("v", impute={"method": None, "value": 1}).to_encoding_spec_dict()["impute"] == {
        "value": 1
    }


def test_the_registry_publishes_the_four_impute_options_in_their_order():
    params = transform_registry("impute")["params"]
    options = [p for p in params if p not in {"field", "groupby", "key", "name"}]
    assert options == ["method", "value", "keyvals", "frame"]


@pytest.mark.parametrize("channel", [fm.X, fm.Y])
@pytest.mark.parametrize(
    ("given", "got"),
    [
        ("zero", "str 'zero'"),
        (0, "int 0"),
        (1.5, "float 1.5"),
        (True, "bool True"),
        ([("value", 0)], "list [('value', 0)]"),
    ],
    ids=["str", "int", "float", "bool", "list"],
)
def test_an_impute_that_is_not_a_dict_is_refused_at_construction(
    channel: Any, given: Any, got: str
):
    with pytest.raises(ValueError) as excinfo:
        channel("v", impute=given)
    assert str(excinfo.value) == (
        f"{channel.__name__}(impute=...) must be a dict of impute options ({_OPTIONS_LIST}), got {got}"
    )


@pytest.mark.parametrize("channel", [fm.X, fm.Y])
def test_an_impute_option_nobody_reads_is_refused_at_construction(channel: Any):
    with pytest.raises(ValueError) as excinfo:
        channel("v", impute={"value": 0, "bogus_key": 1})
    assert str(excinfo.value) == (
        f"{channel.__name__}(impute=...): unknown option 'bogus_key'; expected {_OPTIONS}"
    )


@pytest.mark.parametrize("option", ["field", "groupby", "key", "name"])
def test_an_impute_option_that_belongs_to_the_transform_alone_is_refused_at_construction(
    option: str,
):
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={option: "t"})
    assert str(excinfo.value) == f"Y(impute=...): unknown option '{option}'; expected {_OPTIONS}"


@pytest.mark.parametrize("channel", [fm.X, fm.Y])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_an_impute_value_that_is_not_finite_is_refused_at_construction(channel: Any, bad: float):
    with pytest.raises(ValueError) as excinfo:
        channel("v", impute={"value": bad})
    assert str(excinfo.value) == (
        f"{channel.__name__}(impute=...) value must be a finite number a float can hold; "
        f"received {bad!r}"
    )


def test_an_impute_value_of_no_kind_a_column_holds_is_refused_at_construction():
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"value": [1]})
    assert str(excinfo.value) == (
        "Y(impute=...) value must be a number, a string, True or False, a date or a datetime, "
        "not list; received [1]"
    )


def test_an_impute_method_that_is_not_text_is_refused_at_construction():
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"method": 3})
    assert str(excinfo.value) == "Y(impute=...): method must be a string; got 3"


def test_an_impute_keyvals_that_is_a_bare_string_is_refused_at_construction():
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"keyvals": "Mar"})
    assert str(excinfo.value) == (
        "Y(impute=...): keyvals must be a sequence of key values (list or tuple), "
        "got a bare string 'Mar'; iterating a string yields individual characters"
    )


class _ReadRecordingSet(Set):
    """A ``Set`` that records every read, to show a refused keyvals is never iterated."""

    def __init__(self) -> None:
        self.reads = 0

    def __contains__(self, item: object) -> bool:
        return False

    def __iter__(self):
        self.reads += 1
        return iter(("a", "b"))

    def __len__(self) -> int:
        return 2


_UNORDERED_DOORS: dict[str, Callable[[Any], Any]] = {
    "transform_impute": lambda keyvals: fm.transform_impute("v", key="t", keyvals=keyvals),
    "Y(impute=...)": lambda keyvals: fm.Y("v", impute={"keyvals": keyvals}),
    "X(impute=...)": lambda keyvals: fm.X("v", impute={"keyvals": keyvals}),
}


@pytest.mark.parametrize("door", list(_UNORDERED_DOORS))
@pytest.mark.parametrize(
    ("make", "builder"),
    [
        (lambda: {"a", "b", "c"}, _set_text),
        (lambda: frozenset({"a", "b", "c"}), _set_text),
        (lambda: _SetSubclass("a", "b"), _set_text),
        (lambda: MappingProxyType({"start": 0, "stop": 3}), _mapping_text),
        (lambda: b"abc", _bytes_text),
        (lambda: bytearray(b"abc"), _bytes_text),
        (lambda: memoryview(b"abc"), _bytes_text),
    ],
    ids=[
        "set",
        "frozenset",
        "abc Set subclass",
        "mappingproxy",
        "bytes",
        "bytearray",
        "memoryview",
    ],
)
def test_a_keyvals_with_no_defined_order_or_made_of_bytes_is_refused_at_construction(
    door: str, make: Callable[[], Any], builder: Callable[..., str]
):
    keyvals = make()
    with pytest.raises(ValueError) as excinfo:
        _UNORDERED_DOORS[door](keyvals)
    assert str(excinfo.value) == builder(door, "keyvals", "key values", keyvals)


@pytest.mark.parametrize("door", list(_UNORDERED_DOORS))
@pytest.mark.parametrize(
    "form", [{"start": 0, "stop": 3}, OrderedDict(start=0, stop=3)], ids=["dict", "OrderedDict"]
)
def test_a_dict_keyvals_is_the_sequence_form_on_every_door(door: str, form: dict):
    result = _UNORDERED_DOORS[door](form)
    wire = result["keyvals"] if door == "transform_impute" else result._kwargs["impute"]["keyvals"]
    assert wire == {"start": 0, "stop": 3}


@pytest.mark.parametrize("door", list(_UNORDERED_DOORS))
@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (lambda d: d.keys(), ["z", "a", "m"]),
        (lambda d: d.values(), [2, 0, 1]),
    ],
    ids=["keys", "values"],
)
def test_a_dict_view_is_read_in_insertion_order_like_a_list(
    door: str, view: Callable[[dict], Any], expected: list
):
    keyvals = view({"z": 2, "a": 0, "m": 1})
    listed = _UNORDERED_DOORS[door](expected)
    result = _UNORDERED_DOORS[door](keyvals)
    read = (
        (lambda r: r["keyvals"])
        if door == "transform_impute"
        else (lambda r: r._kwargs["impute"]["keyvals"])
    )
    assert read(result) == read(listed)


@pytest.mark.parametrize("door", list(_UNORDERED_DOORS))
def test_a_dict_items_view_is_read_like_the_list_of_pairs(door: str):
    """``items()`` is ordered, so it is read as ``list(items)`` is: its pairs are not key
    values, and both spellings are refused for that reason, not for having no order."""
    source = {"z": 2, "a": 0}
    with pytest.raises(ValueError) as from_list:
        _UNORDERED_DOORS[door](list(source.items()))
    with pytest.raises(ValueError) as from_view:
        _UNORDERED_DOORS[door](source.items())
    assert str(from_view.value) == str(from_list.value)
    assert "has no order" not in str(from_view.value)


@pytest.mark.parametrize("door", list(_UNORDERED_DOORS))
def test_a_refused_keyvals_set_is_never_iterated(door: str):
    keyvals = _ReadRecordingSet()
    with pytest.raises(ValueError, match="a set has no order"):
        _UNORDERED_DOORS[door](keyvals)
    assert keyvals.reads == 0


@pytest.mark.parametrize("door", list(_UNORDERED_DOORS))
@pytest.mark.parametrize(
    "make",
    [
        lambda: [0, 1, 2],
        lambda: (0, 1, 2),
        lambda: range(3),
        lambda: np.array([0, 1, 2]),
        lambda: pl.Series([0, 1, 2]),
        lambda: pd.Series([0, 1, 2]),
        lambda: (n for n in range(3)),
        lambda: map(int, ["0", "1", "2"]),
    ],
    ids=["list", "tuple", "range", "numpy", "polars", "pandas", "generator", "map"],
)
def test_an_ordered_keyvals_is_still_read_once_into_its_list(door: str, make: Callable[[], Any]):
    result = _UNORDERED_DOORS[door](make())
    wire = result["keyvals"] if door == "transform_impute" else result._kwargs["impute"]["keyvals"]
    assert wire == [0, 1, 2]


def test_an_impute_keyval_that_is_nothing_a_key_holds_is_refused_in_the_encoding_voice():
    no_kind = object()
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"keyvals": [no_kind]})
    assert str(excinfo.value) == (
        f"Y(impute=...): object {no_kind!r} not allowed in keyvals; "
        "accepted: string, number, bool, date or datetime"
    )


def test_an_impute_frame_that_is_not_a_pair_is_refused_at_construction():
    with pytest.raises(ValueError) as excinfo:
        fm.Y("v", impute={"method": "mean", "frame": (1,)})
    assert str(excinfo.value) == (
        "Y(impute=...): frame must be a (preceding, following) pair of row offsets, "
        "each an integer or None; got (1,)"
    )


_PLUS_ONE = dt.timezone(dt.timedelta(hours=1))
_WIRE: list[tuple[dict, dict]] = [
    ({}, {}),
    ({"value": 0}, {"value": 0}),
    ({"method": "value", "value": 0}, {"method": "value", "value": 0}),
    ({"value": np.int64(3)}, {"value": 3}),
    ({"value": np.float32(0.5)}, {"value": 0.5}),
    ({"value": "none"}, {"value": "none"}),
    ({"value": True}, {"value": True}),
    ({"value": np.bool_(False)}, {"value": False}),
    ({"value": dt.date(2024, 1, 2)}, {"value": _instant(1704153600000)}),
    ({"value": 1704153600000}, {"value": 1704153600000}),
    ({"value": None, "method": "mean"}, {"method": "mean"}),
    (
        {"keyvals": [1, "a", True, dt.date(2024, 1, 2)]},
        {"keyvals": [1, "a", True, _instant(1704153600000)]},
    ),
    ({"value": dt.datetime(2024, 1, 2, 1, tzinfo=_PLUS_ONE)}, {"value": _instant(1704153600000)}),
    ({"value": dt.datetime(2024, 1, 2)}, {"value": _instant(1704153600000)}),
    ({"value": np.datetime64("2024-01-02T00:00:00.000")}, {"value": _instant(1704153600000)}),
    (
        {
            "keyvals": [
                np.int64(1),
                np.float64(2.5),
                np.bool_(True),
                dt.datetime(2024, 1, 2, 1, tzinfo=_PLUS_ONE),
            ]
        },
        {"keyvals": [1, 2.5, True, _instant(1704153600000)]},
    ),
    ({"keyvals": {"start": 0, "stop": 5}}, {"keyvals": {"start": 0, "stop": 5}}),
    ({"method": "mean", "frame": (-1, 1)}, {"method": "mean", "frame": [-1, 1]}),
    # the wire order is the options' own order, whatever order they were written in
    (
        {"frame": (None, 0), "keyvals": [1], "value": 2, "method": "mean"},
        {"method": "mean", "value": 2, "keyvals": [1], "frame": [None, 0]},
    ),
]


@pytest.mark.parametrize("channel", [fm.X, fm.Y])
@pytest.mark.parametrize(("given", "wire"), _WIRE, ids=[str(g) for g, _ in _WIRE])
def test_an_impute_dict_rides_the_wire_normalized(channel: Any, given: dict, wire: dict):
    emitted = channel("v", impute=given).to_encoding_spec_dict()["impute"]
    assert emitted == wire
    assert list(emitted) == list(wire)
    assert json.dumps(emitted)  # JSON-serializable: no numpy scalar, no date object


def test_an_impute_dict_is_normalized_once_and_the_callers_dict_is_left_alone():
    given = {"value": np.int64(3), "keyvals": [np.int64(1)]}
    channel = fm.Y("v", impute=given)
    assert type(given["value"]) is np.int64
    assert channel.to_encoding_spec_dict()["impute"] == {"value": 3, "keyvals": [1]}
    assert fm.Y("v", impute=channel.to_encoding_spec_dict()["impute"]).to_encoding_spec_dict() == (
        channel.to_encoding_spec_dict()
    )


def test_an_impute_of_none_is_no_impute():
    assert "impute" not in fm.Y("v", impute=None).to_encoding_spec_dict()


@pytest.mark.parametrize(
    ("bad", "refusal"),
    [
        (
            "zero",
            "Y(impute=...) must be a dict of impute options (method, value, keyvals, frame), "
            "got str 'zero'",
        ),
        (
            {"bogus": 1},
            "Y(impute=...): unknown option 'bogus'; expected method|value|keyvals|frame",
        ),
        (
            {"value": float("nan")},
            "Y(impute=...) value must be a finite number a float can hold; received nan",
        ),
    ],
    ids=["str", "unknown option", "nan"],
)
def test_a_layers_own_y_refuses_a_malformed_impute_where_the_channel_is_written(
    bad: Any, refusal: str
):
    with pytest.raises(ValueError) as excinfo:
        fm.Layer(mark="point", encoding={"y": fm.Y("v", impute=bad)})
    assert str(excinfo.value) == refusal


@pytest.mark.parametrize("channel_name", ["color", "size"])
def test_a_channel_that_has_no_impute_still_warns_and_drops_it_instead_of_refusing_it(
    channel_name: str,
):
    channel = getattr(fm, channel_name.capitalize())
    reset_warnings()
    notice = _warn_once_notice(channel_name)
    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        channel("v", impute="zero")
    assert [str(w.message) for w in caught] == [notice]


def _warn_once_notice(channel: str) -> str:
    """The notice ``warn_once`` gives a kwarg a channel does not honor, read from its own spelling."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_once(channel, "impute")
    return str(caught[0].message)


# ---------------------------------------------------------------------------
# A statistic grouped by a date draws one series per date
# ---------------------------------------------------------------------------


def test_a_statistic_grouped_by_a_date_and_a_level_splits_one_line_per_date():
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
    frame = pl.DataFrame(
        {
            "day": [d for d in days for _ in range(6)],
            "h": ["a", "a", "a", "b", "b", "b"] * 2,
            "t": [1, 2, 3] * 4,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    chart = (
        fm.Chart(frame, width=_W, height=_H)
        .transform(
            fm.transform_aggregate(
                {"field": "v", "fn": "sum", "as": "total"}, groupby=["day", "h", "t"]
            )
        )
        .mark_line()
        .encode(x="t", y="total", detail="day", color="h")
    )
    svg = chart.to_svg()
    drawn = polylines(svg)
    assert len(drawn) == 4  # one per (day, level)
    assert all(len(points) == 3 for _, points in drawn)
    # a line per day: the two days' sums differ, so the two series of one level differ
    by_level: dict[str | None, list[Any]] = {}
    for stroke, points in drawn:
        by_level.setdefault(stroke, []).append(points)
    assert sorted(len(series) for series in by_level.values()) == [2, 2]
    assert all(first != second for first, second in by_level.values())


# ---------------------------------------------------------------------------
# A time crosses the wire tagged: temporal columns and keys read it, no other kind holds it
# ---------------------------------------------------------------------------

_JAN_1 = dt.date(2024, 1, 1)
_JAN_1_ISO = "2024-01-01T00:00:00"


@pytest.mark.parametrize(
    "value",
    [
        dt.datetime(2024, 1, 1, 0, 0, 0, 500),
        dt.datetime(2024, 1, 1, 0, 0, 0, 1500, tzinfo=dt.timezone.utc),
        np.datetime64("2024-01-01T00:00:00.000500"),
        np.datetime64("2024-01-01T00:00:00.000000001"),
    ],
    ids=["datetime 0.5 ms", "aware datetime 1.5 ms", "datetime64[us]", "datetime64[ns]"],
)
def test_an_instant_finer_than_a_millisecond_is_refused_not_rounded(value: Any):
    refusal = (
        f"{value!r} is finer than a millisecond, which an impute value or key cannot carry; "
        "round it to a whole millisecond"
    )
    with pytest.raises(ValueError) as as_value:
        fm.transform_impute("v", value=value)
    assert str(as_value.value) == f"transform_impute(value=...): {refusal}"
    with pytest.raises(ValueError) as as_keyval:
        fm.transform_impute("v", key="t", keyvals=[value])
    assert str(as_keyval.value) == f"transform_impute: keyvals value for key 't': {refusal}"
    with pytest.raises(ValueError) as on_channel:
        fm.Y("v", impute={"value": value})
    assert str(on_channel.value) == f"Y(impute=...) value: {refusal}"


@pytest.mark.parametrize(
    "value", [dt.datetime(2024, 1, 1, 0, 0, 0, 2000), np.datetime64("2024-01-01T00:00:00.002")]
)
def test_an_instant_of_a_whole_millisecond_is_carried_exactly(value: Any):
    assert fm.transform_impute("v", value=value)["value"] == _instant(1704067200002)


@pytest.mark.parametrize(
    ("column", "frame", "holds", "must"),
    [
        ("v", pl.DataFrame({"t": [1, 2], "v": [1.0, None]}), "numbers", "must be a number"),
        ("grade", pl.DataFrame({"t": [1, 2], "grade": ["a", None]}), "text", "must be a string"),
        (
            "ok",
            pl.DataFrame({"t": [1, 2], "ok": [True, None]}),
            "booleans",
            "must be True or False",
        ),
    ],
    ids=["numbers", "text", "booleans"],
)
@pytest.mark.parametrize(
    "value",
    [_JAN_1, dt.datetime(2024, 1, 1), np.datetime64("2024-01-01")],
    ids=["date", "datetime", "datetime64"],
)
def test_a_date_value_on_a_column_that_is_not_temporal_is_refused_naming_the_instant(
    column: str, frame: pl.DataFrame, holds: str, must: str, value: Any
):
    chart = fm.Chart(frame).transform(fm.transform_impute(column, value=value))
    with pytest.raises(ValueError) as excinfo:
        chart.mark_point().encode(x="t", y="t").to_svg()
    assert str(excinfo.value) == _engine(
        f"data_impute: column '{column}' holds {holds}, so value= {must}, but {_JAN_1_ISO} is a "
        "date or datetime"
    )


_TEMPORAL_COLUMNS: dict[str, pl.Series] = {
    name: pl.Series("when", [dt.datetime(2024, 1, 1), None, dt.datetime(2024, 1, 3)], dtype=dtype)
    for name, dtype in {
        "Date": pl.Date,
        "Datetime[ms]": pl.Datetime("ms"),
        "Datetime[us]": pl.Datetime("us"),
        "Datetime[ns]": pl.Datetime("ns"),
        "Datetime[us, UTC]": pl.Datetime("us", "UTC"),
        "Datetime[us, Europe/Paris]": pl.Datetime("us", "Europe/Paris"),
    }.items()
}


@pytest.mark.parametrize("kind", list(_TEMPORAL_COLUMNS))
@pytest.mark.parametrize(
    "value", [dt.date(2024, 1, 2), dt.datetime(2024, 1, 2), np.datetime64("2024-01-02")]
)
def test_a_date_value_fills_a_temporal_column_of_every_unit_and_zone_like_its_epoch_number(
    kind: str, value: Any
):
    frame = pl.DataFrame({"t": [1, 2, 3]}).with_columns(_TEMPORAL_COLUMNS[kind])
    # a bare number on a temporal column is epoch milliseconds, the instant's own spelling
    by_number = fm.transform_impute("when", value=1704153600000)
    svg = _kind_chart(frame, fm.transform_impute("when", value=value), column="when")
    assert svg == _kind_chart(frame, by_number, column="when")
    assert svg != _kind_chart(frame, column="when")


@pytest.mark.parametrize(
    ("key", "dtype"),
    [
        (pl.Series("k", [1, 2]), "Int64"),
        (pl.Series("k", [1.0, 2.0]), "Float64"),
        (pl.Series("k", ["a", "b"]), "Utf8"),
        (pl.Series("k", [True, False]), "Boolean"),
    ],
    ids=["integer", "float", "text", "boolean"],
)
def test_a_date_keyval_on_a_key_that_is_not_temporal_is_refused(key: pl.Series, dtype: str):
    frame = pl.DataFrame({"g": ["x", "x"], "v": [10.0, 20.0]}).with_columns(key)
    step = fm.transform_impute("v", key="k", groupby=["g"], keyvals=[_JAN_1])
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, step).to_svg()
    assert str(excinfo.value) == _engine(
        "data_impute: keyvals value 2024-01-01T00:00:00 (a date or datetime) is not compatible "
        f"with key column 'k' dtype {dtype}"
    )


_TEMPORAL_KEY_KINDS = [k for k in _KEYS if k.startswith(("Date", "Datetime"))]


@pytest.mark.parametrize("kind", _TEMPORAL_KEY_KINDS)
@pytest.mark.parametrize("spelling", ["date", "datetime", "datetime64"])
def test_a_date_keyval_is_added_to_a_temporal_key_of_every_unit_and_zone(kind: str, spelling: str):
    dtype, keys = _KEYS[kind]
    # the group holds 1 and 5 Jan; 3 Jan lies between them in every unit and zone
    frame = pl.DataFrame(
        {"g": ["x", "x"], "t": pl.Series("t", [keys[0], keys[2]], dtype=dtype), "v": [10.0, 50.0]}
    )
    day = {
        "date": dt.date(2024, 1, 3),
        "datetime": dt.datetime(2024, 1, 3),
        "datetime64": np.datetime64("2024-01-03"),
    }[spelling]
    rows = _rows(frame, _impute(value=-1, keyvals=[day]), groups=("x",))
    assert rows == [("x", "10"), ("x", "-1"), ("x", "50")]


# ---------------------------------------------------------------------------
# Order beyond the one case: an imputed row follows the last row of its series with an earlier key
# ---------------------------------------------------------------------------


def test_an_interleaved_series_gets_its_imputed_row_right_after_its_last_earlier_row():
    # series x lacks key 2; its rows alternate with series y's, and its last earlier row is x1
    frame = pl.DataFrame(
        {
            "g": ["x", "y", "x", "y", "y"],
            "t": [1, 1, 3, 2, 3],
            "v": [10.0, 1.0, 30.0, 2.0, 3.0],
        }
    )
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "10"),
        ("x", "-1"),
        ("y", "1"),
        ("x", "30"),
        ("y", "2"),
        ("y", "3"),
    ]


def test_a_trajectory_that_revisits_a_key_gets_the_row_after_its_last_earlier_row():
    frame = pl.DataFrame({"g": ["x"] * 4, "t": [1, 3, 1, 3], "v": [10.0, 30.0, 11.0, 31.0]})
    assert _rows(frame, _impute(value=-1, keyvals=[2]), groups=("x",)) == [
        ("x", "10"),
        ("x", "30"),
        ("x", "11"),
        ("x", "-1"),
        ("x", "31"),
    ]


def test_a_series_lacking_its_smallest_key_gets_the_row_before_its_first_row():
    frame = pl.DataFrame(
        {"g": ["x", "x", "y", "y", "y"], "t": [2, 3, 1, 2, 3], "v": [20.0, 30.0, 1.0, 2.0, 3.0]}
    )
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "-1"),
        ("x", "20"),
        ("x", "30"),
        ("y", "1"),
        ("y", "2"),
        ("y", "3"),
    ]


def test_a_series_lacking_its_largest_key_gets_the_row_after_its_last_row():
    frame = pl.DataFrame(
        {"g": ["x", "x", "y", "y", "y"], "t": [1, 2, 1, 2, 3], "v": [10.0, 20.0, 1.0, 2.0, 3.0]}
    )
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "10"),
        ("x", "20"),
        ("x", "-1"),
        ("y", "1"),
        ("y", "2"),
        ("y", "3"),
    ]


@pytest.mark.parametrize("keyvals", [[1, 2, 4, 6, 8], [8, 6, 4, 2, 1]])
def test_several_missing_keys_land_in_key_order_among_themselves_and_around_the_callers_rows(
    keyvals: list[int],
):
    frame = pl.DataFrame({"g": ["x"] * 3, "t": [3, 5, 7], "v": [30.0, 50.0, 70.0]})
    # by key: 1 and 2 before the first row, 4 and 6 in the gaps, 8 after the last
    assert [
        label
        for _, label in _rows(frame, _impute(value=-1, keyvals=keyvals), groups=("x",), text="t")
    ] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
    _rows(frame, _impute(value=-1, keyvals=keyvals), groups=("x",))  # no caller row moved


def test_several_missing_keys_of_an_unsorted_series_follow_the_last_earlier_row_in_key_order():
    frame = pl.DataFrame({"g": ["x"] * 3, "t": [5, 1, 3], "v": [50.0, 10.0, 30.0]})
    # 2 follows the 1 (idx 1); 4 follows the 3 (the last row, whose key is earlier); 0 precedes all
    rows = _rows(frame, _impute(value=-1, keyvals=[0, 2, 4]), groups=("x",), text="t")
    assert [label for _, label in rows] == ["0", "5", "1", "2", "3", "4"]


def test_a_text_key_without_keyvals_is_placed_by_its_first_appearance_in_the_frame():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y", "y", "y", "y"],
            "t": ["b", "d", "a", "b", "c", "d"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # the order is b, d, a, c: x lacks a and c, which follow its last earlier row, d, in that order
    rows = _rows(frame, _impute(value=-1), text="t")
    assert [label for g, label in rows if g == "x"] == ["b", "d", "a", "c"]
    assert [label for g, label in rows if g == "y"] == ["a", "b", "c", "d"]


def test_a_text_key_with_keyvals_puts_them_first_in_its_order():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y", "y", "y", "y"],
            "t": ["b", "d", "a", "b", "c", "d"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # keyvals d, c lead: the order is d, c, b, a, so x's c follows its d and its a follows its c
    rows = _rows(frame, _impute(value=-1, keyvals=["d", "c"]), text="t")
    assert [label for g, label in rows if g == "x"] == ["b", "d", "c", "a"]
    assert [label for g, label in rows if g == "y"] == ["a", "b", "c", "d"]


def test_a_boolean_series_lacking_true_gets_the_row_after_its_false_row():
    frame = pl.DataFrame({"g": ["x", "y", "y"], "t": [False, False, True], "v": [10.0, 1.0, 2.0]})
    assert _rows(frame, _impute(value=-1)) == [("x", "10"), ("x", "-1"), ("y", "1"), ("y", "2")]


# ---------------------------------------------------------------------------
# What the engine does that the options leave to it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "values", "must"),
    [
        ("grade", ["a", None], "holds text, so value= must be a string"),
        (
            "when",
            [dt.date(2024, 1, 1), None],
            "holds dates and times, so value= must be a datetime.date, a datetime.datetime or an "
            "epoch-millisecond number",
        ),
        ("ok", [True, None], "holds booleans, so value= must be True or False"),
    ],
    ids=["text", "date", "boolean"],
)
def test_a_column_that_is_not_numeric_has_no_default_fill_and_asks_for_value(
    column: str, values: list, must: str
):
    frame = pl.DataFrame({"t": [1, 2], column: values})
    chart = fm.Chart(frame).transform(fm.transform_impute(column))
    with pytest.raises(ValueError) as excinfo:
        chart.mark_point().encode(x="t", y="t").to_svg()
    assert str(excinfo.value) == _engine(
        f"data_impute: column '{column}' {must}, and none was given; pass value="
    )


def test_a_keyvals_sequence_on_a_date_key_reads_its_numbers_as_epoch_milliseconds():
    frame = pl.DataFrame(
        {"g": ["x", "x"], "t": [dt.date(2024, 1, 1), dt.date(2024, 1, 5)], "v": [10.0, 50.0]}
    )
    # 2 Jan up to, not including, 5 Jan, a day apart: 2, 3 and 4 Jan between the two rows
    sequence = {"start": 1704153600000, "stop": 1704412800000, "step": 86400000}
    rows = _rows(frame, _impute(value=-1, keyvals=sequence), groups=("x",))
    assert rows == [("x", "10"), ("x", "-1"), ("x", "-1"), ("x", "-1"), ("x", "50")]


def test_a_keyvals_sequence_of_more_than_a_million_keys_is_refused():
    frame = pl.DataFrame({"g": ["x"], "t": [1], "v": [1.0]})
    step = _impute(value=-1, keyvals={"start": 0, "stop": 1000002})
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, step).to_svg()
    assert str(excinfo.value) == _engine(
        "data_impute: keyvals={'start': ...} would make 1000002 keys, more than the 1000000 a "
        "series can be completed over; list the keys, or use a larger step"
    )


@pytest.mark.parametrize("method", ["mean", "median", "min", "max"])
def test_a_frame_open_on_both_sides_draws_what_no_frame_draws_byte_for_byte(method: str):
    frame = pl.DataFrame(
        {
            "g": ["x"] * 5,
            "t": [5, 1, 4, 2, 3],
            "v": [0.1, None, 0.2, 1e16, None],
        }
    )

    def drawn(**kwargs: Any) -> tuple[str, list[str]]:
        step = fm.transform_impute("v", method=method, key="t", groupby=["g"], **kwargs)
        return _render_capturing(_text_chart(frame, step).to_svg)

    assert drawn(frame=(None, None)) == drawn()
    assert drawn(frame=(None, None)) != drawn(frame=(0, 1))


def test_a_nan_and_infinite_keys_sort_after_the_finite_ones_and_before_it_for_minus_infinity():
    inf, nan = float("inf"), float("nan")
    frame = pl.DataFrame(
        {
            "g": ["x"] + ["y"] * 6,
            "t": [0.0, 2.0, nan, -inf, inf, -5.0, 1.0],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    chart = (
        fm.Chart(frame, width=_W, height=_H)
        .transform(_impute(value=-1))
        .mark_point()
        .encode(x="v", y="g:N", tooltip=fm.Tooltip("t"))
    )
    panels = json.loads(chart.interactive().scene_json)["panels"]
    keys = [
        tip["fields"][0]["value"]
        for panel in panels
        for batch in panel["marks"]
        for tip in batch["tooltips"]
    ]
    # group x: -inf (blank) first, the finite keys ascending, then +inf and NaN (blank, blank)
    assert keys[:7] == ["", "-5", "0", "1", "2", "", ""]
    # group y keeps its own rows where they were and gains 0 after its last earlier row, -5
    assert keys[7:] == ["2", "", "", "", "-5", "0", "1"]


def test_a_row_with_a_null_key_is_ignored_when_placing_and_stays_where_it_was():
    frame = pl.DataFrame(
        {
            "g": ["x", "x", "y", "y"],
            "t": [1.0, None, 1.0, 2.0],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # x gains 0 before its 1 and 2 after it, which is before its null-key row
    rows = _rows(frame, _impute(value=-1, keyvals=[0.0, 2.0]), text="v")
    assert rows == [
        ("x", "-1"),
        ("x", "1"),
        ("x", "-1"),
        ("x", "2"),
        ("y", "-1"),
        ("y", "3"),
        ("y", "4"),
    ]


# ---------------------------------------------------------------------------
# One case of each family through every door, static and interactive, against its written-out twin
# ---------------------------------------------------------------------------

_J = [dt.date(2024, 1, d) for d in (1, 2, 3)]


def _family_date(*, imputed: bool) -> tuple[pl.DataFrame, dict | None, str]:
    frame = pl.DataFrame(
        {"t": [1, 2, 3], "f": ["a"] * 3, "when": [_J[0], _J[1] if not imputed else None, _J[2]]}
    )
    return frame, (fm.transform_impute("when", value=_J[1]) if imputed else None), "when"


def _family_keyvals(*, imputed: bool) -> tuple[pl.DataFrame, dict | None, str]:
    if not imputed:
        return pl.DataFrame({"t": [1, 2, 3], "f": ["a"] * 3, "v": [10.0, 0.0, 30.0]}), None, "v"
    frame = pl.DataFrame({"t": [1, 3], "f": ["a"] * 2, "v": [10.0, 30.0]})
    return frame, fm.transform_impute("v", key="t", groupby=["f"], keyvals=[2]), "v"


def _family_frame(*, imputed: bool) -> tuple[pl.DataFrame, dict | None, str]:
    values = [10.0, None if imputed else 25.0, 40.0]
    frame = pl.DataFrame({"t": [1, 2, 3], "f": ["a"] * 3, "v": values})
    step = fm.transform_impute("v", method="mean", key="t", groupby=["f"], frame=(-1, 1))
    return frame, (step if imputed else None), "v"


_FAMILIES = {"date value": _family_date, "keyvals": _family_keyvals, "frame": _family_frame}


def _door_chart(door: str, family: Callable[..., Any], *, imputed: bool) -> Any:
    frame, step, column = family(imputed=imputed)
    base = fm.Chart(frame, width=_W, height=_H)
    if step is not None:
        base = base.transform(step)
    base = base.mark_point().encode(x="t", y=column)
    # the sibling layer or panel is the same chart on both sides, drawn from the complete frame
    sibling = fm.Chart(family(imputed=False)[0]).mark_point().encode(x="t", y="t")
    return {
        "plain": lambda: base,
        "facet": lambda: base.facet("f"),
        "plus": lambda: base + sibling,
        "Chart.layer": lambda: base.layer(fm.Layer(mark="point", encoding={"x": "t", "y": "t"})),
        "LayerChart": lambda: fm.layer(base, sibling),
        "hconcat": lambda: fm.hconcat(base, sibling),
    }[door]()


_DOOR_NAMES = ["plain", "facet", "plus", "Chart.layer", "LayerChart", "hconcat"]


@pytest.mark.parametrize("family", list(_FAMILIES))
@pytest.mark.parametrize("door", _DOOR_NAMES)
def test_every_door_draws_an_imputing_chart_as_its_written_out_twin_in_both_output_kinds(
    door: str, family: str
):
    build = _FAMILIES[family]
    imputing = _door_chart(door, build, imputed=True)
    twin = _door_chart(door, build, imputed=False)
    # the twin is rendered through the same door, and a twin that cannot render fails here
    svg, messages = _render_capturing(imputing.to_svg)
    twin_svg, twin_messages = _render_capturing(twin.to_svg)
    assert (svg, messages) == (twin_svg, twin_messages)
    assert messages == []
    # the control: the frame without its step draws something else
    frame, _, column = build(imputed=True)
    plain = fm.Chart(frame, width=_W, height=_H).mark_point().encode(x="t", y=column)
    if door == "plain":
        assert svg != plain.to_svg()
    scene, scene_messages = _render_capturing(lambda: imputing.interactive().scene_json)
    twin_scene, twin_scene_messages = _render_capturing(lambda: twin.interactive().scene_json)
    assert (scene, scene_messages) == (twin_scene, twin_scene_messages)


# ---------------------------------------------------------------------------
# A time crosses the wire as its exact whole millisecond
# ---------------------------------------------------------------------------

_EPOCH = dt.datetime(1970, 1, 1)
_UTC = dt.timezone.utc


def _wire_instants(value: Any) -> tuple[Any, Any]:
    """The tagged instant ``value=`` and a ``keyvals`` entry carry for ``value``."""
    as_value = fm.transform_impute("v", value=value)["value"]
    as_keyval = fm.transform_impute("v", key="t", keyvals=[value])["keyvals"][0]
    return as_value, as_keyval


def test_a_whole_millisecond_datetime_crosses_the_wire_as_its_exact_millisecond():
    # a float conversion truncated this one to ...944
    awkward = dt.datetime(2004, 3, 4, 5, 56, 53, 945000)
    assert _wire_instants(awkward) == (_instant(1078379813945), _instant(1078379813945))
    pre_epoch = dt.datetime(1960, 1, 1, 0, 0, 0, 945000)
    assert _wire_instants(pre_epoch) == (_instant(-315619199055), _instant(-315619199055))
    rng = random.Random(16)
    span = int((dt.datetime(2191, 1, 1) - dt.datetime(1811, 1, 1)) / dt.timedelta(milliseconds=1))
    start = int((dt.datetime(1811, 1, 1) - _EPOCH) // dt.timedelta(milliseconds=1))
    for _ in range(2000):
        millis = start + rng.randrange(span)
        moment = _EPOCH + dt.timedelta(milliseconds=millis)
        assert _wire_instants(moment) == (_instant(millis), _instant(millis)), moment


@pytest.mark.parametrize(
    "moment",
    [
        dt.datetime(2024, 3, 31, 1, 59, 59, 945000, tzinfo=ZoneInfo("Europe/Paris")),
        dt.datetime(2024, 3, 31, 3, 0, 0, 1000, tzinfo=ZoneInfo("Europe/Paris")),
        dt.datetime(2024, 10, 27, 2, 30, 0, 945000, tzinfo=ZoneInfo("Europe/Paris"), fold=0),
        dt.datetime(2024, 10, 27, 2, 30, 0, 945000, tzinfo=ZoneInfo("Europe/Paris"), fold=1),
        dt.datetime(1960, 7, 1, 12, 0, 0, 945000, tzinfo=ZoneInfo("America/New_York")),
    ],
    ids=["before spring", "after spring", "autumn fold 0", "autumn fold 1", "pre-epoch New York"],
)
def test_an_aware_datetime_across_a_dst_change_crosses_as_its_exact_utc_millisecond(moment: Any):
    utc = moment.astimezone(_UTC).replace(tzinfo=None)
    millis = (utc - _EPOCH) // dt.timedelta(milliseconds=1)
    assert _wire_instants(moment) == (_instant(millis), _instant(millis))


def test_keyvals_built_from_a_millisecond_key_column_add_no_row():
    keys = [dt.datetime(2024, 1, 1, 0, 0, 0, 945000), dt.datetime(2024, 1, 1, 0, 0, 0, 946000)]
    frame = pl.DataFrame(
        {"g": ["x", "x"], "t": pl.Series("t", keys, dtype=pl.Datetime("ms")), "v": [2.0, 3.0]}
    )
    step = _impute(value=-1, keyvals=frame["t"].to_list())
    rows, messages = _render_capturing(lambda: _rows(frame, step, groups=("x",)))
    assert rows == [("x", "2"), ("x", "3")]
    assert messages == []


@pytest.mark.parametrize(
    "unit_value",
    [
        np.datetime64("2024", "Y"),
        np.datetime64("2024-03", "M"),
        np.datetime64("2024-03-04", "W"),
        np.datetime64("2024-03-04", "D"),
        np.datetime64("2024-03-04T05", "h"),
        np.datetime64("2024-03-04T05:56", "m"),
        np.datetime64("2024-03-04T05:56:53", "s"),
        np.datetime64("2024-03-04T05:56:53.945", "ms"),
        np.datetime64("2024-03-04T05:56:53.945000", "us"),
        np.datetime64("2024-03-04T05:56:53.945000000", "ns"),
        np.datetime64("1970-01-02T05:56:53.945000000000", "ps"),
        np.datetime64("1960-01-01T00:00:00.945", "ms"),
        np.datetime64("1960-01-01T00:00:00.945000", "us"),
        np.datetime64("1960-01-01T00:00:00.945000000", "ns"),
    ],
)
def test_a_datetime64_of_every_unit_crosses_as_its_exact_millisecond(unit_value: Any):
    millis = int(unit_value.astype("datetime64[ms]").astype("int64"))
    assert _wire_instants(unit_value) == (_instant(millis), _instant(millis))
    assert _wire_instants(np.datetime64("1960-01-01T00:00:00.945", "ms"))[0] == _instant(
        -315619199055
    )


def test_a_pandas_timestamp_with_a_nanosecond_part_is_refused_as_finer_than_a_millisecond():
    stamp: Any = pd.Timestamp("2024-01-01 00:00:00.000000500")
    refusal = (
        f"{stamp!r} is finer than a millisecond, which an impute value or key cannot carry; "
        "round it to a whole millisecond"
    )
    with pytest.raises(ValueError) as as_value:
        fm.transform_impute("v", value=stamp)
    assert str(as_value.value) == f"transform_impute(value=...): {refusal}"
    with pytest.raises(ValueError) as as_keyval:
        fm.transform_impute("v", key="t", keyvals=[stamp])
    assert str(as_keyval.value) == f"transform_impute: keyvals value for key 't': {refusal}"


def test_a_datetime64_beyond_an_epoch_offset_is_refused_as_out_of_range():
    far = np.datetime64(10**13, "D")
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", value=far)
    assert str(excinfo.value) == (
        f"transform_impute(value=...): {far!r} is out of range to represent as a "
        "microsecond-precision epoch offset"
    )


def test_an_aware_pandas_timestamp_crosses_as_its_exact_millisecond():
    stamp = pd.Timestamp("2024-01-01 00:00:00.945", tz="Europe/Paris")
    assert _wire_instants(stamp) == (_instant(1704063600945), _instant(1704063600945))
    assert _wire_instants(pd.Timestamp("2024-01-01 00:00:00.945"))[0] == _instant(1704067200945)


def test_a_pandas_not_a_time_is_refused_as_numpy_is():
    not_a_time: Any = pd.NaT
    with pytest.raises(ValueError) as excinfo:
        fm.transform_impute("v", value=not_a_time)
    assert str(excinfo.value) == f"transform_impute(value=...): NaT {_NOT_A_TIME}"


# ---------------------------------------------------------------------------
# A channel keeps the impute dict its caller wrote and emits the wire form
# ---------------------------------------------------------------------------

_DAY = dt.date(2024, 1, 5)
_DAY_MS = 1704412800000
_DAY_WIRE = _instant(_DAY_MS)


def _impute_values(node: Any) -> list[Any]:
    """Every ``impute`` value anywhere in a chart's JSON, in document order."""
    found: list[Any] = []
    if isinstance(node, dict):
        for key, child in node.items():
            if key == "impute":
                found.append(child)
            else:
                found.extend(_impute_values(child))
    elif isinstance(node, list):
        for child in node:
            found.extend(_impute_values(child))
    return found


_AGG_FRAME = pl.DataFrame({"t": [1, 1, 2, 2, 3], "h": ["x"] * 5, "v": [1.0, 2.0, 3.0, 4.0, 5.0]})


def _aggregated_y() -> Any:
    return fm.Y("v", aggregate="mean", impute={"keyvals": [_DAY], "value": _DAY})


#: A date key and a date value: the frame a date ``keyvals`` and a date fill fit. Each series gains
#: ``_DAY`` with ``_DAY`` as its value; ``_DATED_TWIN`` writes those rows down.
_DAYS = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
_DATED = pl.DataFrame({"t": _DAYS * 2, "h": ["x", "x", "y", "y"], "when": [*_DAYS, *_DAYS]})
_DATED_TWIN = pl.DataFrame(
    {
        "t": [*_DAYS, _DAY, *_DAYS, _DAY],
        "h": ["x"] * 3 + ["y"] * 3,
        "when": [*_DAYS, _DAY, *_DAYS, _DAY],
    }
)


def _dated_y(*, impute: dict[str, Any] | None) -> Any:
    return fm.Y("when", impute=impute) if impute is not None else "when"


def _dated_chart(frame: pl.DataFrame, y: Any) -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x="t", y=y, color="h")


def _assert_tagged_fill_is_honoured(
    imputed: Callable[[], Any], written_out: Callable[[], Any], caller_rows: Callable[[], Any]
) -> None:
    """The tagged date literal reaches the engine and is honoured: the chart with ``impute=`` draws
    the one over the frame with the rows written down, which is not the chart over the caller's rows."""
    svg, shown = _render_capturing(lambda: imputed().to_svg())
    twin, twin_shown = _render_capturing(lambda: written_out().to_svg())
    plain, _ = _render_capturing(lambda: caller_rows().to_svg())
    assert shown == twin_shown == []
    assert svg == twin
    assert svg != plain


def test_a_date_impute_lowers_to_its_wire_form_and_renders_only_on_a_date_key():
    """An aggregated channel carries the date ``keyvals`` and fill in their tagged wire form, but
    its integer key refuses a date keyval at render; the same literals on a date key and a temporal
    ``y`` draw the frame with the rows written down."""
    chart = fm.Chart(_AGG_FRAME).mark_line().encode(x="t", y=_aggregated_y(), color="h")
    assert _impute_values(json.loads(chart.to_json())) == [
        {"value": _DAY_WIRE, "keyvals": [_DAY_WIRE]}
    ]
    # a date keyval on an integer key is refused at render, in the engine's words
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    assert str(excinfo.value) == (
        "impute: keyvals value 2024-01-05T00:00:00 (a date or datetime) is not compatible with "
        "key column 't' dtype Int64"
    )
    # the same literals on a date key and a date value draw as the frame with the rows written down
    dated = {"keyvals": [_DAY], "value": _DAY}
    _assert_tagged_fill_is_honoured(
        lambda: _dated_chart(_DATED, _dated_y(impute=dated)),
        lambda: _dated_chart(_DATED_TWIN, _dated_y(impute=None)),
        lambda: _dated_chart(_DATED, _dated_y(impute=None)),
    )
    assert _dated_chart(_DATED, _dated_y(impute=dated)).interactive().scene_json


def test_an_aggregated_channel_renders_a_tagged_date_keyval_on_a_date_key():
    """``mean(v)`` by day and series, with a date ``keyvals``: each series gains the day, filled with
    0, and the chart is the one over the frame with those rows written down."""
    days = [_DAYS[0], _DAYS[0], _DAYS[1], _DAYS[1]]
    frame = pl.DataFrame(
        {"t": days * 2, "h": ["x"] * 4 + ["y"] * 4, "v": [1.0, 3.0, 4.0, 6.0, 2.0, 4.0, 5.0, 7.0]}
    )
    twin = pl.DataFrame(
        {
            "t": [*days, _DAY, *days, _DAY],
            "h": ["x"] * 5 + ["y"] * 5,
            "v": [1.0, 3.0, 4.0, 6.0, 0.0, 2.0, 4.0, 5.0, 7.0, 0.0],
        }
    )

    def build(data: pl.DataFrame, impute: dict[str, Any] | None) -> fm.Chart:
        y = fm.Y("v", aggregate="mean", impute=impute) if impute else fm.Y("v", aggregate="mean")
        return fm.Chart(data).mark_line().encode(x="t", y=y, color="h")

    _assert_tagged_fill_is_honoured(
        lambda: build(frame, {"keyvals": [_DAY], "value": 0}),
        lambda: build(twin, None),
        lambda: build(frame, None),
    )


def test_a_channel_with_a_date_impute_rebuilds_through_with_options():
    channel = fm.Y("v", impute={"keyvals": [_DAY], "value": _DAY, "frame": (-1, 1)})
    rebuilt = channel._with_options(title="T")
    assert rebuilt.to_encoding_spec_dict()["impute"] == channel.to_encoding_spec_dict()["impute"]
    assert rebuilt.to_encoding_spec_dict()["impute"] == {
        "value": _DAY_WIRE,
        "keyvals": [_DAY_WIRE],
        "frame": [-1, 1],
    }


@pytest.mark.parametrize("door", ["plus operand", "Layer", "plain plus", "Chart.layer"])
def test_a_date_impute_on_a_plus_operand_and_a_layer_draws_the_written_out_rows(door: str):
    """An aggregated channel on this door lowers to the tagged wire form (the first assertion); on a
    date key with a temporal ``y`` the operand or layer that writes the impute alone draws the rows
    written down. ``plain plus`` and ``Chart.layer`` spell the fill as the epoch-millisecond number
    of the same day."""
    options = {
        "keyvals": [_DAY],
        "value": _DAY_MS if "plain" in door or door == "Chart.layer" else _DAY,
    }
    aggregated_base = fm.Chart(_AGG_FRAME).mark_line().encode(x="t", y="v", color="h")
    if door in ("plus operand", "plain plus"):
        aggregated = aggregated_base + fm.Chart(_AGG_FRAME).mark_point().encode(
            x="t", y=_aggregated_y()
        )
    else:
        aggregated = aggregated_base.layer(
            fm.Layer(mark="point", encoding={"x": "t", "y": _aggregated_y()})
        )
    assert _impute_values(json.loads(aggregated.to_json())) == [
        {"value": _DAY_WIRE, "keyvals": [_DAY_WIRE]}
    ]

    def build(frame: pl.DataFrame, imputed: bool) -> Any:
        base = _dated_chart(_DATED, "when").mark_line()
        y = _dated_y(impute=options if imputed else None)
        if door in ("plus operand", "plain plus"):
            return base + _dated_chart(frame, y)
        return base.layer(
            fm.Layer(mark="point", data=None if imputed else frame, encoding={"x": "t", "y": y})
        )

    _assert_tagged_fill_is_honoured(
        lambda: build(_DATED, True), lambda: build(_DATED_TWIN, False), lambda: build(_DATED, False)
    )
    assert build(_DATED, True).interactive().scene_json


def test_a_channel_keeps_the_impute_its_caller_wrote_and_emits_the_wire_form():
    given = {"frame": (-1, 1), "keyvals": [_DAY], "value": _DAY, "method": "mean"}
    channel = fm.Y("v", impute=given)
    assert channel.option("impute") == given
    assert repr(channel) == f"Y('v', impute={given!r})"
    assert channel == fm.Y("v", impute=dict(given))
    assert channel != fm.Y("v", impute={**given, "frame": [-1, 1]})
    assert channel.to_encoding_spec_dict()["impute"] == {
        "method": "mean",
        "value": _DAY_WIRE,
        "keyvals": [_DAY_WIRE],
        "frame": [-1, 1],
    }


# ---------------------------------------------------------------------------
# One literal dispatch serves value= and keyvals
# ---------------------------------------------------------------------------


def test_a_literal_of_no_accepted_kind_is_refused_with_one_error_on_both_doors():
    no_kind: Any = object()
    kinds = "string, number, bool, date or datetime"
    value_refusal = (
        "{label} must be a number, a string, True or False, a date or a datetime, "
        f"not object; received {no_kind!r}"
    )
    routes: list[tuple[Callable[[], Any], str]] = [
        (
            lambda: fm.transform_impute("v", key="t", keyvals=[no_kind]),
            f"transform_impute: object {no_kind!r} not allowed in keyvals for key 't'; accepted: {kinds}",
        ),
        (
            lambda: fm.Y("v", impute={"keyvals": [no_kind]}),
            f"Y(impute=...): object {no_kind!r} not allowed in keyvals; accepted: {kinds}",
        ),
        (
            lambda: fm.transform_impute("v", value=no_kind),
            value_refusal.format(label="transform_impute(value=...)"),
        ),
        (
            lambda: fm.Y("v", impute={"value": no_kind}),
            value_refusal.format(label="Y(impute=...) value"),
        ),
    ]
    for build, refusal in routes:
        with pytest.raises(ValueError) as excinfo:
            build()
        assert type(excinfo.value) is ValueError
        assert str(excinfo.value) == refusal


@pytest.mark.parametrize(
    "literal",
    [True, np.bool_(False), "a", np.int64(3), 2.5, dt.date(2024, 1, 1)],
    ids=["True", "np.bool_", "str", "np.int64", "float", "date"],
)
def test_value_and_keyvals_read_each_literal_kind_alike(literal: Any):
    as_value, as_keyval = (
        fm.transform_impute("v", value=literal)["value"],
        fm.transform_impute("v", key="t", keyvals=[literal])["keyvals"][0],
    )
    assert as_value == as_keyval
    assert type(as_value) is type(as_keyval)
    on_channel = fm.Y("v", impute={"value": literal, "keyvals": [literal]}).to_encoding_spec_dict()
    assert on_channel["impute"] == {"value": as_value, "keyvals": [as_keyval]}


# ---------------------------------------------------------------------------
# One frame check serves transform_window and transform_impute
# ---------------------------------------------------------------------------

_FRAME_SHAPE = (
    "{door}: frame must be a (preceding, following) pair of row offsets, each an integer or None; "
    "got {frame!r}"
)
_FRAME_BEYOND = (
    "{door}: frame offset {end} is beyond a 64-bit row offset; use None for an unbounded end"
)
_SUM = {"op": "sum", "field": "v", "as": "s"}
_FRAME_DOORS: dict[str, Callable[[Any], dict]] = {
    "transform_impute": lambda frame: fm.transform_impute("v", method="mean", frame=frame),
    "transform_window": lambda frame: fm.transform_window(_SUM, frame=frame),
    "Y(impute=...)": lambda frame: fm.Y(
        "v", impute={"method": "mean", "frame": frame}
    ).to_encoding_spec_dict()["impute"],
}


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
@pytest.mark.parametrize(
    "frame",
    [("a", 1), (1.0, 2), (1,), (True, 1), "ab", 3],
    ids=["text", "float", "one end", "bool", "string", "number"],
)
def test_a_frame_that_is_not_a_pair_of_integers_is_refused_on_both_doors(door: str, frame: Any):
    with pytest.raises(ValueError) as excinfo:
        _FRAME_DOORS[door](frame)
    assert str(excinfo.value) == _FRAME_SHAPE.format(door=door, frame=frame)


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
@pytest.mark.parametrize("frame", [(0, 2**63), (-(2**63) - 1, 0), (2**64, None)])
def test_a_frame_end_beyond_64_bits_is_refused_on_both_doors(door: str, frame: tuple):
    end = next(e for e in frame if e is not None and not -(2**63) <= e < 2**63)
    with pytest.raises(ValueError) as excinfo:
        _FRAME_DOORS[door](frame)
    assert str(excinfo.value) == _FRAME_BEYOND.format(door=door, end=end)


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
def test_a_frame_of_numpy_and_extreme_integers_rides_as_python_integers(door: str):
    numpy_frame: Any = (np.int64(-1), np.int32(1))
    wire = _FRAME_DOORS[door](numpy_frame)["frame"]
    assert wire == [-1, 1]
    assert all(type(end) is int for end in wire)
    assert _FRAME_DOORS[door]((-(2**63), 2**63 - 1))["frame"] == [-(2**63), 2**63 - 1]
    assert _FRAME_DOORS[door]((None, None))["frame"] == [None, None]


def test_a_frame_offset_past_the_group_reads_to_its_edge():
    frame = pl.DataFrame({"g": ["x"] * 4, "t": [1, 2, 3, 4], "v": [1.0, None, 3.0, 4.0]})

    def filled(bounds: tuple[int | None, int | None]) -> tuple[list[str], list[str]]:
        step = fm.transform_impute("v", method="mean", key="t", groupby=["g"], frame=bounds)
        rows, messages = _render_capturing(lambda: _rows(frame, step, groups=("x",)))
        return [label for _, label in rows], messages

    big = 2**63 - 1
    assert filled((0, big)) == (["1", "3.5", "3", "4"], [])
    # an open preceding end reaches the group's first row: the whole group, as no frame reads it
    assert filled((None, big)) == filled((None, None)) == (["1", "2.6667", "3", "4"], [])
    step = fm.transform_impute("v", method="mean", key="t", groupby=["g"])
    assert filled((None, big))[0] == [label for _, label in _rows(frame, step, groups=("x",))]
    assert filled((-(2**63), 0)) == (["1", "1", "3", "4"], [])
    # sums over a window that reaches the batch's edge on either side
    window = pl.DataFrame({"g": ["x"] * 4, "t": [1, 2, 3, 4], "v": [1.0, 2.0, 3.0, 4.0]})

    def sums(bounds: tuple[int, int]) -> list[str]:
        chart = (
            fm.Chart(window, width=_W, height=_H)
            .transform(fm.transform_window(_SUM, sort=["t"], frame=bounds))
            .mark_text()
            .encode(x="t", y="g:N", text="s")
        )
        return [label for _, label in _marks(chart.to_svg())]

    assert sums((0, big)) == ["10", "9", "7", "4"]
    assert sums((-(2**63), 0)) == ["1", "3", "6", "10"]


# ---------------------------------------------------------------------------
# Rows of two groups at one place, and a null group key under a constant fill
# ---------------------------------------------------------------------------


def test_imputed_rows_of_two_groups_at_one_place_close_the_earlier_group_first():
    frame = pl.DataFrame({"g": ["x", "y", "y"], "t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    # x gains keys 2 and 3, y gains key 1: x's two rows follow x's row, y's row precedes y's
    assert _rows(frame, _impute(value=-1)) == [
        ("x", "1"),
        ("x", "-1"),
        ("x", "-1"),
        ("y", "-1"),
        ("y", "2"),
        ("y", "3"),
    ]


_NULL_GROUP = pl.DataFrame({"g": ["x", "x", None], "t": [1, 2, 1], "v": [1.0, None, 3.0]})


def test_a_null_groupby_row_under_a_constant_fill_is_named_only_when_a_key_completes_the_groups():
    def notices(step: dict) -> list[str]:
        chart = fm.Chart(_NULL_GROUP, width=_W, height=_H).transform(step).mark_point()
        return _render_capturing(chart.encode(x="t", y="v").to_svg)[1]

    keyed = fm.transform_impute("v", value=-1, key="t", groupby=["g"])
    unkeyed = fm.transform_impute("v", value=-1, groupby=["g"])
    statistic = fm.transform_impute("v", method="mean", key="t", groupby=["g"])
    assert notices(keyed) == [
        "Impute: 1 row(s) have a null groupby key, so they belong to no group and no key is "
        "completed for them"
    ]
    assert notices(unkeyed) == []
    assert notices(statistic) == [
        "Impute: 1 row(s) have a null groupby key; the whole column's statistic was used instead"
    ]


# ---------------------------------------------------------------------------
# A channel snapshots the caller's impute dict; an iterable keyvals or frame is read once
# ---------------------------------------------------------------------------


def _iterables() -> dict[str, Callable[[], Any]]:
    return {
        "generator": lambda: (k for k in [1, 2, 3]),
        "map": lambda: map(int, "123"),
        "iterator": lambda: iter([1, 2, 3]),
        "numpy array": lambda: np.array([1, 2, 3]),
        "polars Series": lambda: pl.Series([1, 2, 3]),
        "pandas Series": lambda: pd.Series([1, 2, 3]),
    }


_ITERABLE_NAMES = list(_iterables())


@pytest.mark.parametrize("name", _ITERABLE_NAMES)
@pytest.mark.parametrize("channel", [fm.X, fm.Y])
def test_an_iterable_keyvals_on_a_channel_reaches_the_wire_in_order(channel: Any, name: str):
    keys = _iterables()[name]()
    built = channel("v", impute={"keyvals": keys})
    assert built.to_encoding_spec_dict()["impute"] == {"keyvals": [1, 2, 3]}
    # emitting again, and rebuilding, read the same snapshot (a one-shot iterator is not read twice)
    assert built.to_encoding_spec_dict()["impute"] == {"keyvals": [1, 2, 3]}
    rebuilt = built._with_options(title="T")
    assert rebuilt.to_encoding_spec_dict()["impute"] == {"keyvals": [1, 2, 3]}


@pytest.mark.parametrize("name", _ITERABLE_NAMES)
def test_an_iterable_keyvals_on_transform_impute_reaches_the_wire_in_order(name: str):
    spec = fm.transform_impute("v", key="t", keyvals=_iterables()[name]())
    assert spec["keyvals"] == [1, 2, 3]


def test_the_callers_impute_dict_is_left_as_written():
    keys = (k for k in [1, 2, 3])
    given: dict[str, Any] = {"keyvals": keys, "value": 0, "frame": (-1, 1), "method": "mean"}
    fm.Y("v", impute=given)
    assert list(given) == ["keyvals", "value", "frame", "method"]
    assert given["keyvals"] is keys
    assert given["frame"] == (-1, 1)


def test_a_channel_holds_a_snapshot_the_caller_cannot_change_afterwards():
    keyvals = [1, 2]
    sequence = {"start": 0, "stop": 3}
    frame = [-1, 1]
    given: dict[str, Any] = {"keyvals": keyvals, "method": "mean", "frame": frame}
    by_sequence: dict[str, Any] = {"keyvals": sequence}
    channel = fm.Y("v", impute=given)
    sequenced = fm.Y("v", impute=by_sequence)
    before = (channel.to_encoding_spec_dict(), sequenced.to_encoding_spec_dict())
    given["bogus"] = 1
    given["value"] = "late"
    keyvals.append(99)
    frame[0] = -5
    sequence["stop"] = 99
    sequence["bogus"] = 1
    assert (channel.to_encoding_spec_dict(), sequenced.to_encoding_spec_dict()) == before
    assert channel.to_encoding_spec_dict()["impute"] == {
        "method": "mean",
        "keyvals": [1, 2],
        "frame": [-1, 1],
    }
    assert channel.option("impute") == {"keyvals": [1, 2], "method": "mean", "frame": [-1, 1]}
    assert channel.option("impute") is not given
    assert sequenced.to_encoding_spec_dict()["impute"] == {"keyvals": {"start": 0, "stop": 3}}


def test_transform_impute_holds_what_it_was_given_when_the_caller_edits_it_afterwards():
    keyvals = [1, 2]
    sequence = {"start": 0, "stop": 3}
    frame = [-1, 1]
    listed = fm.transform_impute("v", method="mean", key="t", keyvals=keyvals, frame=frame)
    sequenced = fm.transform_impute("v", key="t", keyvals=sequence)
    keyvals.append(99)
    frame[0] = -5
    sequence["stop"] = 99
    assert listed["keyvals"] == [1, 2]
    assert listed["frame"] == [-1, 1]
    assert sequenced["keyvals"] == {"start": 0, "stop": 3}


def test_a_stored_snapshot_keeps_the_callers_spelling_in_option_repr_and_equality():
    given = {"frame": (-1, 1), "keyvals": [_DAY], "value": _DAY, "method": "mean"}
    channel = fm.Y("v", impute=given)
    assert channel.option("impute") == given
    assert channel.option("impute") is not given
    assert repr(channel) == f"Y('v', impute={given!r})"
    assert channel == fm.Y("v", impute=dict(given))
    generated = fm.Y("v", impute={"keyvals": (k for k in [1, 2])})
    assert generated.option("impute") == {"keyvals": [1, 2]}
    assert generated == fm.Y("v", impute={"keyvals": [1, 2]})


def test_every_rebuild_route_works_with_a_channel_built_from_a_generator():
    def aggregated() -> Any:
        return fm.Y("v", aggregate="mean", impute={"keyvals": (d for d in [_DAY]), "value": _DAY})

    def dated() -> Any:
        return fm.Y("when", impute={"keyvals": (d for d in [_DAY]), "value": _DAY})

    chart = fm.Chart(_AGG_FRAME).mark_line().encode(x="t", y=aggregated(), color="h")
    assert _impute_values(json.loads(chart.to_json())) == [
        {"value": _DAY_WIRE, "keyvals": [_DAY_WIRE]}
    ]
    base = _dated_chart(_DATED, "when").mark_line()
    routes = {
        "plus": lambda frame, y: base + _dated_chart(frame, y),
        "Layer": lambda frame, y: base.layer(
            fm.Layer(mark="point", data=frame if y == "when" else None, encoding={"x": "t", "y": y})
        ),
    }
    for build in routes.values():
        _assert_tagged_fill_is_honoured(
            lambda: build(_DATED, dated()),
            lambda: build(_DATED_TWIN, "when"),
            lambda: build(_DATED, "when"),
        )
    rebuilt = aggregated()._with_options(title="T")
    assert rebuilt.to_encoding_spec_dict()["impute"] == {
        "value": _DAY_WIRE,
        "keyvals": [_DAY_WIRE],
    }


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
@pytest.mark.parametrize(
    ("make", "wire"),
    [
        (lambda: range(-1, 2, 2), [-1, 1]),
        (lambda: np.array([-1, 1]), [-1, 1]),
        (lambda: (np.int64(-1), None), [-1, None]),
        (lambda: (np.int64(-1), np.int32(1)), [-1, 1]),
        (lambda: [-1, 1], [-1, 1]),
        (lambda: (-1, 1), [-1, 1]),
    ],
    ids=["range", "numpy array", "numpy scalar and None", "numpy scalars", "list", "tuple"],
)
def test_a_frame_may_be_an_ordered_bounded_pair_of_integers(
    door: str, make: Callable[[], Any], wire: list
):
    got = _FRAME_DOORS[door](make())["frame"]
    assert got == wire
    assert all(end is None or type(end) is int for end in got)


class _NeverRead:
    """A frame that fails the test if anything iterates, indexes or measures it."""

    def __iter__(self) -> Any:
        raise AssertionError("the frame was read")

    def __len__(self) -> int:
        raise AssertionError("the frame was read")

    def __getitem__(self, index: Any) -> Any:
        raise AssertionError("the frame was read")


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
@pytest.mark.parametrize(
    "make",
    [
        lambda: {1, -1},
        lambda: frozenset({1, -1}),
        lambda: (end for end in (-1, 1)),
        lambda: iter([-1, 1]),
        lambda: itertools.count(),
        lambda: {"a": 1, "b": 2},
        lambda: "ab",
        lambda: b"ab",
        lambda: np.array([[-1, 1]]),
        lambda: _NeverRead(),
    ],
    ids=[
        "set",
        "frozenset",
        "generator",
        "iterator",
        "unbounded iterator",
        "dict",
        "str",
        "bytes",
        "two-dimensional array",
        "never read",
    ],
)
def test_a_frame_that_is_not_an_ordered_bounded_pair_is_refused_without_being_read(
    door: str, make: Callable[[], Any]
):
    frame = make()
    with pytest.raises(ValueError) as excinfo:
        _FRAME_DOORS[door](frame)
    assert str(excinfo.value) == _FRAME_SHAPE.format(door=door, frame=frame)
    if isinstance(frame, (itertools.count, type(end for end in ()))):
        assert next(frame) is not None  # still unread: it yields its first item


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
@pytest.mark.parametrize(
    "frame",
    [np.array([-1.0, 1.0]), np.array([True, False]), range(0, 3), [1], (1, 2, 3)],
    ids=["float array", "bool array", "range of three", "list of one", "tuple of three"],
)
def test_a_pair_that_is_not_two_integers_is_refused_whatever_it_is_held_in(door: str, frame: Any):
    with pytest.raises(ValueError) as excinfo:
        _FRAME_DOORS[door](frame)
    assert str(excinfo.value) == _FRAME_SHAPE.format(door=door, frame=frame)


@pytest.mark.parametrize("door", list(_FRAME_DOORS))
def test_a_frame_end_beyond_64_bits_is_refused_in_any_pair(door: str):
    with pytest.raises(ValueError) as excinfo:
        _FRAME_DOORS[door](np.array([0, 2**63], dtype=object))
    assert str(excinfo.value) == _FRAME_BEYOND.format(door=door, end=2**63)


# ---------------------------------------------------------------------------
# A Float32 keyval the column cannot hold is refused, not narrowed to an infinity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keyval", "shown"),
    [(1e39, "1e+39"), (-1e39, "-1e+39"), (1e-50, "9.999999999999999e-51")],
)
def test_a_float32_keyval_beyond_the_columns_range_is_refused_not_narrowed(
    keyval: float, shown: str
):
    frame = pl.DataFrame(
        {"g": ["x", "x"], "t": pl.Series([1.0, 2.0], dtype=pl.Float32), "v": [1.0, 2.0]}
    )
    step = _impute(value=-1, keyvals=[keyval])
    with pytest.raises(ValueError) as excinfo:
        _text_chart(frame, step).to_svg()
    assert str(excinfo.value) == _engine(
        f"data_impute: keyvals value {shown} is not compatible with key column 't' dtype Float32"
    )
