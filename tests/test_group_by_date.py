"""A date or datetime column is a group key on every grouped transform and statistic.

The engine's one shared group-key reader used to refuse a temporal column, so a mean per day
(``mark_line(x='day', y='mean(v)')``), a sum per day by colour, ``transform_aggregate(groupby=['day'])``,
``transform_window(groupby=['day'])`` and ``transform_impute(key='day')`` all raised
``groupby column 'day' has unsupported dtype Timestamp(...)``. Now a ``Date``, a ``Datetime`` of any
unit and zone and a ``Duration`` group like any other key: the key keeps its own dtype in the
output, groups sort in time order, a warning names a group as an ISO-8601 date-time, a null key is
dropped and counted as any null key is, and a chart with no temporal key is unchanged.

Each expected output is stated against a twin that needs no date grouping:

* the same frame with the key written as its epoch-millisecond ``Int64`` (a numeric key groups,
  sorts and draws the same marks), or as its ISO-8601 text, where the key is not drawn;
* a written-out frame (the aggregate, the filled rows) plotted directly, where the key is drawn, so
  the date axis, its tick labels and the mark order are the ones a date column always had.

A group key a twin cannot show (time order, dtype) is pinned with explicit values.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import warnings
from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import pyarrow as pa
import pytest

import ferrum as fm
from ferrum._coerce import normalize_for_rust, to_arrow_table
from tests._svg_extents import axis_tick_labels
from tests._svg_rects import mark_layer, polylines, svg_elements
from tests.test_null_group_warning import (
    FIT_FEW_POINTS,
    FIT_SEPARATED,
    FIT_SINGULAR,
    _NO_ROWS_TO_DRAW,
    _group_dtype_refusal,
    _whole_fit_message,
    _fit_message,
    _null_key_dropped_message,
    _render_capturing,
)

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

_DAYS = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 5)]
_EPOCH_MS = [
    int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    for d in _DAYS
]
_ISO = [f"{d.isoformat()}T00:00:00" for d in _DAYS]
_HALF_DAY_MS = 43_200_000


def _base() -> pl.DataFrame:
    """Twenty-four rows: three days by two levels of ``h`` by four repeats, the days listed out of
    time order (day 2, day 0, day 1) so that first-appearance order and time order differ."""
    rows = []
    n = 0
    for di in (2, 0, 1):
        for h in ("b", "a"):
            for r in range(4):
                n += 1
                rows.append(
                    {
                        "di": di,
                        "r": r,
                        "h": h,
                        "v": float((n * 7) % 11) + 0.5 * r + di,
                        "w": float((n * 5) % 13) + r,
                        "b": (n + r) % 2,
                        "b2": int((n * 7) % 5 < 2),
                        # a value missing once per (day, level) whose group mean differs by day
                        # and level, so a grouped mean impute fills it differently per group
                        "vn": None
                        if r == 0
                        else float((n * 7) % 11) + 10.0 * di + (5.0 if h == "a" else 0.0),
                        # one far value per day among otherwise close ones, the days far apart:
                        # it is an outlier only when the days are grouped apart
                        "vo": 100.0 * di + 0.1 * r + (40.0 if (r == 3 and h == "a") else 0.0),
                    }
                )
    return pl.DataFrame(rows)


def _with_key(column: pl.Expr) -> pl.DataFrame:
    return _base().with_columns(k=column).drop("di")


def _epoch_frame() -> pl.DataFrame:
    """The key as epoch milliseconds: the numeric twin of every temporal key below."""
    return _with_key(pl.col("di").replace_strict(dict(enumerate(_EPOCH_MS)), return_dtype=pl.Int64))


def _iso_frame() -> pl.DataFrame:
    """The key as ISO-8601 text: the text twin, which names a group as the engine now names a date."""
    return _with_key(pl.col("di").replace_strict(dict(enumerate(_ISO)), return_dtype=pl.Utf8))


def _date_frame() -> pl.DataFrame:
    return _with_key(pl.col("di").replace_strict(dict(enumerate(_DAYS)), return_dtype=pl.Date))


def _datetime(
    unit: Literal["ms", "us", "ns"], zone: str | None = None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    def build(df: pl.DataFrame) -> pl.DataFrame:
        column = pl.col("k").cast(pl.Datetime("ms")).cast(pl.Datetime(unit))
        if zone is not None:
            column = column.dt.replace_time_zone(zone)
        return df.with_columns(column)

    return build


def _duration(unit: Literal["ms", "us", "ns"]) -> Callable[[pl.DataFrame], pl.DataFrame]:
    def build(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("k")
            .cast(pl.Datetime("ms"))
            .cast(pl.Int64)
            .cast(pl.Duration("ms"))
            .cast(pl.Duration(unit))
        )

    return build


# Each kind: how its key column is built from the Date frame, and how the same instant reads as
# epoch milliseconds (the numeric twin's key).
_KINDS: dict[str, tuple[Callable[[pl.DataFrame], pl.DataFrame], pl.Expr]] = {
    "Date": (lambda df: df, pl.col("k").cast(pl.Datetime("ms")).dt.epoch("ms")),
    "Datetime-ms": (_datetime("ms"), pl.col("k").dt.epoch("ms")),
    "Datetime-us": (_datetime("us"), pl.col("k").dt.epoch("ms")),
    "Datetime-ns": (_datetime("ns"), pl.col("k").dt.epoch("ms")),
    "Datetime-UTC": (_datetime("ms", "UTC"), pl.col("k").dt.epoch("ms")),
    "Datetime-Paris": (_datetime("us", "Europe/Paris"), pl.col("k").dt.epoch("ms")),
    "Duration-ms": (_duration("ms"), pl.col("k").dt.total_milliseconds()),
    "Duration-us": (_duration("us"), pl.col("k").dt.total_milliseconds()),
}


def _kind_frame(kind: str, shape: str = "days") -> pl.DataFrame:
    """The frame with its key as ``kind``: ``days`` (three midnights), ``twelve-hours`` (two
    instants per day) or ``fall-back`` (three instants, two reading alike in Europe/Paris)."""
    if shape == "days":
        return _KINDS[kind][0](_date_frame())
    if shape == "twelve-hours":
        return _KINDS[kind][0](_instants_frame("twelve-hours")[0])
    return _fall_back_frame({"Datetime-UTC": "UTC", "Datetime-Paris": "Europe/Paris"}[kind])


def _kind_twin(kind: str, shape: str = "days") -> pl.DataFrame:
    """The ``Int64`` epoch-millisecond twin of the ``kind`` frame: same rows, same instants."""
    _, epoch = _KINDS[kind]
    return _kind_frame(kind, shape).with_columns(epoch.alias("k"))


# Every kind over midnights; every kind that can hold a time of day over two instants a day; the
# zoned kinds over the hour the clocks go back.
_KIND_SHAPES = [
    *((kind, "days") for kind in _KINDS),
    *((kind, "twelve-hours") for kind in _KINDS if kind != "Date"),
    ("Datetime-UTC", "fall-back"),
    ("Datetime-Paris", "fall-back"),
]


# ---------------------------------------------------------------------------
# The grouped doors: every transform and statistic that takes a ``groupby``, each grouped by the
# column ``k`` and drawn from its outputs, so the key itself is not on an axis. A door's chart over
# a temporal key must draw what the same chart draws over the key's epoch twin.
# ---------------------------------------------------------------------------

_G = ["k"]


def _through(transform: Any, x: str, y: str) -> Callable[[pl.DataFrame], fm.Chart]:
    return lambda df: fm.Chart(df).transform(transform).mark_point().encode(x=x, y=y)


_DOORS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "Kde": _through(fm.Kde("v", groupby=_G), "value", "density"),
    "Violin": _through(fm.Violin("v", groupby=_G), "violin_x", "violin_y"),
    "BoxStats": _through(fm.BoxStats("v", groupby=_G), "median", "q3"),
    "Smooth": _through(fm.Smooth("w", "v", method="lm", groupby=_G), "x", "y"),
    "Summary": _through(fm.Summary("v", groupby=_G), "mean", "upper"),
    "ErrorExtent": _through(fm.ErrorExtent("v", groupby=_G), "mean", "lower"),
    "Bin": _through(fm.Bin("v", groupby=_G), "bin_start", "count"),
    "QQ": _through(fm.QQ("v", groupby=_G), "theoretical", "sample"),
    "LetterValue": _through(fm.LetterValue("v", groupby=_G), "lower", "upper"),
    "Robust": _through(fm.Robust("w", "v", groupby=_G), "x", "y"),
    "Glm": _through(fm.Glm("w", "v", groupby=_G), "x", "y"),
    "Logistic": _through(fm.Logistic("w", "b2", groupby=_G), "x", "y"),
    # every response of ``b`` is 1 within each day, so no day can be fitted
    "Logistic_unfittable": _through(fm.Logistic("w", "b", groupby=_G), "x", "y"),
    "Kde2D": _through(fm.Kde2D("w", "v", groupby=_G), "nx", "ny"),
    "aggregate": _through(
        fm.transform_aggregate(
            {"field": "v", "fn": "sum", "as": "total"},
            {"field": "v", "fn": "count", "as": "n"},
            groupby=["k", "h"],
        ),
        "total",
        "n",
    ),
    "join_aggregate": _through(
        fm.transform_join_aggregate({"field": "v", "fn": "mean", "as": "m"}, groupby=_G), "w", "m"
    ),
    "window": _through(
        fm.transform_window({"op": "row_number", "as": "c"}, sort=["v"], groupby=_G), "w", "c"
    ),
    "stack": _through(fm.transform_stack("v", groupby=_G, sort=["w"]), "y0", "y1"),
    "pivot": _through(fm.transform_pivot("h", "v", groupby=_G), "a", "b"),
    "top_k": _through(fm.transform_top_k(2, field="v", groupby=_G), "w", "v"),
    "density": _through(fm.transform_density("v", groupby=_G), "value", "density"),
    "regression": _through(fm.transform_regression("w", "v", groupby=_G), "x", "y"),
    "loess": _through(fm.transform_loess("w", "v", groupby=_G), "x", "y"),
    "aggregate_key": _through(
        fm.transform_aggregate(
            {"field": "v", "fn": "sum", "as": "total"},
            {"field": "v", "fn": "count", "as": "n"},
            groupby=["k"],
        ),
        "total",
        "n",
    ),
    "Outliers": _through(fm.Outliers("vo", groupby=_G), "w", "vo"),
    "impute_groupby": _through(fm.transform_impute("vn", method="mean", groupby=_G), "w", "vn"),
    # encoding aggregates: the date is the grouping detail of a mean per level
    "encoding_aggregate_detail": lambda df: (
        fm.Chart(df).mark_point().encode(x="h", y="mean(v)", detail="k")
    ),
}

# The grouped impute the plain-key digests were measured with: its column holds no null, so it is
# independent of its grouping and serves only the unchanged-chart control.
_PLAIN_DOORS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    **_DOORS,
    "impute_groupby_complete": _through(
        fm.transform_impute("v", method="mean", groupby=_G), "w", "v"
    ),
}

# The doors the kind and composition matrices run, one per family of reader.
_REPRESENTATIVE = ["aggregate", "BoxStats", "Kde", "window"]


def _circles(svg: str) -> list[dict[str, str]]:
    return svg_elements(svg, "circle", lambda a: True)


def _render(chart: fm.Chart) -> tuple[str, list[str]]:
    return _render_capturing(chart.to_svg)


def _iso_of_epoch_ms(millis: int) -> str:
    return (dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=millis)).isoformat(
        timespec="seconds"
    )


def _as_instants(messages: list[str]) -> list[str]:
    """A numeric twin's messages with each epoch-millisecond group name spelled as the date-time
    the date key's group is named by."""
    return [
        re.sub(r"(?<=k=)(-?\d+)(?= \()", lambda found: _iso_of_epoch_ms(int(found[1])), message)
        for message in messages
    ]


def _scene(chart: fm.Chart) -> tuple[str, list[str]]:
    """The interactive page (scene and packed data) and the warnings raised building it."""
    return _render_capturing(lambda: chart.interactive().to_html())


def _normalized_type(frame: pl.DataFrame, column: str = "k") -> pa.DataType:
    """The dtype ``column`` has when it reaches the engine."""
    return normalize_for_rust(to_arrow_table(frame)).schema.field(column).type


@pytest.mark.parametrize("door", list(_DOORS))
def test_a_date_key_draws_what_its_epoch_twin_draws(door: str):
    """Every grouped door over a ``Date`` key draws byte-for-byte the chart of the same key as an
    ``Int64`` of epoch milliseconds, and warns the same sentences with each group named as the
    date-time rather than the number."""
    twin_svg, twin_messages = _render(_DOORS[door](_epoch_frame()))
    if door != "Logistic_unfittable":  # that door draws nothing: its days cannot be fitted
        assert _circles(twin_svg), (
            f"{door}: the twin draws no marks, so the comparison proves nothing"
        )
    svg, messages = _render(_DOORS[door](_date_frame()))
    assert svg == twin_svg
    assert messages == _as_instants(twin_messages)
    if door == "Logistic_unfittable":
        assert messages
        assert all(f"k={_ISO[i]} (" in m for i, m in zip((2, 0, 1), messages))


@pytest.mark.parametrize("kind,shape", _KIND_SHAPES)
@pytest.mark.parametrize("door", list(_DOORS))
@pytest.mark.parametrize("output", ["static", "interactive"])
def test_every_unit_and_zone_groups_like_its_epoch_twin(
    kind: str, shape: str, door: str, output: str
):
    """A ``Datetime`` in any unit, with or without a time zone, and a ``Duration`` in any unit
    group as the same instants do written as epoch milliseconds, on every grouped door, over
    midnights, over two instants a day and over the hour the clocks go back."""
    render = _render if output == "static" else _scene
    twin_out, twin_messages = render(_DOORS[door](_kind_twin(kind, shape)))
    out, messages = render(_DOORS[door](_kind_frame(kind, shape)))
    assert out == twin_out
    assert messages == _as_instants(twin_messages)


_KIND_TYPES: dict[str, pa.DataType] = {
    "Date": pa.timestamp("ms"),
    "Datetime-ms": pa.timestamp("ms"),
    "Datetime-us": pa.timestamp("ms"),
    "Datetime-ns": pa.timestamp("ms"),
    "Datetime-UTC": pa.timestamp("ms", tz="UTC"),
    "Datetime-Paris": pa.timestamp("ms", tz="Europe/Paris"),
    "Duration-ms": pa.timestamp("ms"),
    "Duration-us": pa.timestamp("ms"),
}


@pytest.mark.parametrize("kind", list(_KINDS))
def test_every_kind_reaches_the_engine_as_milliseconds_in_its_own_zone(kind: str):
    """A zoned key must reach the engine zoned, or the matrix above runs a naive column."""
    assert _normalized_type(_kind_frame(kind)) == _KIND_TYPES[kind]


@pytest.mark.parametrize("zone", [None, "UTC", "Europe/Paris"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_a_timestamp_of_any_unit_reaches_the_engine_in_milliseconds_in_its_zone(
    unit: Literal["s", "ms", "us", "ns"], zone: str | None
):
    table = pa.table(
        {"k": pa.array([0, 86_400, 172_800], pa.timestamp("s")).cast(pa.timestamp(unit, tz=zone))}
    )
    assert normalize_for_rust(to_arrow_table(table)).schema.field("k").type == pa.timestamp(
        "ms", tz=zone
    )
    assert normalize_for_rust(table).schema.field("k").type == pa.timestamp("ms", tz=zone)
    if unit != "s":
        frame = pl.DataFrame({"k": pl.Series([0, 86_400_000, 172_800_000]).cast(pl.Datetime("ms"))})
        frame = frame.with_columns(pl.col("k").cast(pl.Datetime(unit)))
        if zone is not None:
            frame = frame.with_columns(pl.col("k").dt.replace_time_zone(zone))
        assert _normalized_type(frame) == pa.timestamp("ms", tz=zone)


# ---------------------------------------------------------------------------
# The key keeps its own dtype and sorts in time order: a written-out aggregate drawn directly.
# ---------------------------------------------------------------------------


def _written_out_sum(frame: pl.DataFrame) -> pl.DataFrame:
    """The per-key sum ``transform_aggregate`` makes, written out in polars at the key's dtype, the
    groups listed in the order their keys first appear (as the engine lists every key dtype)."""
    return frame.group_by("k", maintain_order=True).agg(total=pl.col("v").sum())


@pytest.mark.parametrize("kind", list(_KINDS))
def test_an_aggregate_by_a_date_returns_the_key_at_its_own_dtype_in_first_appearance_order(
    kind: str,
):
    """The key column comes back as the column it was (its axis is a date axis, not epoch numbers)
    and the groups come back in the order their keys first appear, as an integer key's do."""
    frame = _kind_frame(kind)
    chart = (
        fm.Chart(frame)
        .transform(
            fm.transform_aggregate({"field": "v", "fn": "sum", "as": "total"}, groupby=["k"])
        )
        .mark_line()
        .encode(x="k", y="total")
    )
    written_out = fm.Chart(_written_out_sum(frame)).mark_line().encode(x="k", y="total")
    svg, messages = _render(chart)
    twin_svg, _ = _render(written_out)
    assert svg == twin_svg
    assert messages == []
    labels = axis_tick_labels(svg, axis="x")
    assert labels and all(":" in label for label in labels), (
        f"{kind}: the x axis reads as numbers, not as dates: {labels}"
    )
    # three groups, one vertex per day, joined in row order: the days are listed out of time
    # order, so the line doubles back on screen
    [(_, points)] = [p for p in polylines(svg) if len(p[1]) > 1]
    assert len(points) == 3
    assert [x for x, _ in points] != sorted(x for x, _ in points)
    twin = fm.Chart(_kind_twin(kind)).transform(
        fm.transform_aggregate({"field": "v", "fn": "sum", "as": "total"}, groupby=["k"])
    )
    [(_, twin_points)] = [
        p for p in polylines(_render(twin.mark_line().encode(x="k", y="total"))[0]) if len(p[1]) > 1
    ]
    assert [y for _, y in points] == [y for _, y in twin_points]


def test_the_groups_of_an_unsorted_date_column_come_back_in_first_appearance_order():
    """The rows list the days 2024-01-05, 2024-01-01, 2024-01-02; the groups are drawn in that
    order, the order an integer-key twin draws them and the order of the totals polars computes
    with ``maintain_order``."""
    frame = _date_frame()
    chart = (
        fm.Chart(frame)
        .transform(
            fm.transform_aggregate({"field": "v", "fn": "sum", "as": "total"}, groupby=["k"])
        )
        .mark_line()
        .encode(x="k", y="total")
    )
    expected = _written_out_sum(frame)
    assert expected["k"].to_list() == [_DAYS[2], _DAYS[0], _DAYS[1]]
    assert frame["k"].unique(maintain_order=True).to_list() == [_DAYS[2], _DAYS[0], _DAYS[1]]
    svg, _ = _render(chart)
    [(_, points)] = [p for p in polylines(svg) if len(p[1]) > 1]
    [(_, written_points)] = [
        p
        for p in polylines(_render(fm.Chart(expected).mark_line().encode(x="k", y="total"))[0])
        if len(p[1]) > 1
    ]
    assert points == written_points
    twin = fm.Chart(_epoch_frame()).transform(
        fm.transform_aggregate({"field": "v", "fn": "sum", "as": "total"}, groupby=["k"])
    )
    [(_, twin_points)] = [
        p for p in polylines(_render(twin.mark_line().encode(x="k", y="total"))[0]) if len(p[1]) > 1
    ]
    assert [y for _, y in points] == [y for _, y in twin_points]


@pytest.mark.parametrize("kind", list(_KINDS))
def test_a_mean_per_day_line_draws_over_every_kind_as_its_written_out_twin(kind: str):
    """``mark_line(x=key, y='mean(v)')`` over a Datetime in any unit and zone and a Duration."""
    frame = _kind_frame(kind)
    chart = fm.Chart(frame).mark_line().encode(x="k", y="mean(v)")
    means = frame.group_by("k", maintain_order=True).agg(m=pl.col("v").mean())
    twin = fm.Chart(means).mark_line().encode(x="k", y="m")
    svg, messages = _render(chart)
    twin_svg, _ = _render(twin)
    assert messages == []
    assert mark_layer(svg) == mark_layer(twin_svg)
    assert axis_tick_labels(svg, axis="x") == axis_tick_labels(twin_svg, axis="x")
    [(_, points)] = [p for p in polylines(svg) if len(p[1]) > 1]
    assert len(points) == 3


@pytest.mark.parametrize("kind", list(_KINDS))
def test_a_sum_per_day_stacked_bar_draws_over_every_kind_as_its_written_out_twin(kind: str):
    frame = _kind_frame(kind)
    chart = fm.Chart(frame).mark_bar().encode(x="k", y="sum(v)", color="h")
    sums = frame.group_by("k", "h", maintain_order=True).agg(s=pl.col("v").sum())
    twin = fm.Chart(sums).mark_bar().encode(x="k", y="s", color="h")
    svg, messages = _render(chart)
    twin_svg, _ = _render(twin)
    assert messages == []
    assert mark_layer(svg) == mark_layer(twin_svg)
    assert len(svg_elements(mark_layer(svg), "rect", lambda a: True)) == 6


# ---------------------------------------------------------------------------
# Charts that group by a date
# ---------------------------------------------------------------------------


def _daily() -> pl.DataFrame:
    """Three days by two levels, two readings each, the days out of time order."""
    return pl.DataFrame(
        {
            "day": [_DAYS[2], _DAYS[2], _DAYS[0], _DAYS[0], _DAYS[1], _DAYS[1]] * 2,
            "h": ["a", "b"] * 6,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 2.5, 8.5, 6.5, 1.5, 9.5, 3.5],
        }
    )


def test_a_line_of_the_mean_per_day_draws_one_vertex_per_day_in_row_order():
    frame = _daily()
    chart = fm.Chart(frame).mark_line().encode(x="day", y="mean(v)")
    means = frame.group_by("day", maintain_order=True).agg(m=pl.col("v").mean())
    written = fm.Chart(means).mark_line().encode(x="day", y="m")
    svg, messages = _render(chart)
    written_svg, _ = _render(written)
    assert messages == []
    assert mark_layer(svg) == mark_layer(written_svg)
    assert axis_tick_labels(svg, axis="x") == axis_tick_labels(written_svg, axis="x")
    [(_, points)] = [p for p in polylines(svg) if len(p[1]) > 1]
    assert len(points) == 3


def test_a_bar_of_the_sum_per_day_by_colour_stacks_a_bar_per_day_and_level_in_row_order():
    frame = _daily()
    chart = fm.Chart(frame).mark_bar().encode(x="day", y="sum(v)", color="h")
    sums = frame.group_by("day", "h", maintain_order=True).agg(s=pl.col("v").sum())
    written = fm.Chart(sums).mark_bar().encode(x="day", y="s", color="h")
    svg, messages = _render(chart)
    written_svg, _ = _render(written)
    assert messages == []
    assert mark_layer(svg) == mark_layer(written_svg)
    assert len(svg_elements(mark_layer(svg), "rect", lambda a: True)) == 6


def test_an_aggregate_by_a_date_colour_draws_the_circles_of_its_epoch_twin():
    """A date column bound to ``color`` groups the mean per level; its continuous colour scale
    paints the same marks the epoch number paints."""
    frame = _date_frame()
    twin = _epoch_frame()
    chart = lambda f: fm.Chart(f).mark_point().encode(x="h", y="mean(v)", color="k")  # noqa: E731
    svg, _ = _render(chart(frame))
    twin_svg, _ = _render(chart(twin))
    fills = lambda s: [c["cx"] + c["cy"] + c.get("fill", "") for c in _circles(s)]  # noqa: E731
    assert len(fills(svg)) == 6
    assert fills(svg) == fills(twin_svg)


def test_transform_aggregate_by_a_date_leaves_one_point_per_day_at_its_own_dtype():
    frame = _daily()
    total = fm.transform_aggregate({"field": "v", "fn": "sum", "as": "total"}, groupby=["day"])
    chart = fm.Chart(frame).transform(total).mark_point().encode(x="day", y="total")
    sums = frame.group_by("day", maintain_order=True).agg(total=pl.col("v").sum())
    svg, messages = _render(chart)
    written_svg, _ = _render(fm.Chart(sums).mark_point().encode(x="day", y="total"))
    assert messages == []
    assert svg == written_svg
    assert len(_circles(svg)) == 3


def test_transform_window_by_a_date_numbers_the_rows_as_it_does_by_the_days_epoch():
    """``row_number`` within each day numbers the rows as it does within each epoch-millisecond
    key (the numbering is the engine's own and a date is one more key)."""
    window = fm.transform_window({"op": "row_number", "as": "c"}, sort=["v"], groupby=["day"])
    twin = _daily().with_columns(day=pl.col("day").cast(pl.Datetime("ms")).dt.epoch("ms"))
    chart = lambda f: fm.Chart(f).transform(window).mark_point().encode(x="v", y="c")  # noqa: E731
    twin_svg, _ = _render(chart(twin))
    svg, messages = _render(chart(_daily()))
    assert len(_circles(twin_svg)) == 12
    assert messages == []
    assert svg == twin_svg


def test_transform_join_aggregate_by_a_date_adds_each_days_mean_to_its_rows():
    frame = _daily()
    join = fm.transform_join_aggregate({"field": "v", "fn": "mean", "as": "m"}, groupby=["day"])
    chart = fm.Chart(frame).transform(join).mark_point().encode(x="v", y="m")
    joined = frame.with_columns(m=pl.col("v").mean().over("day"))
    svg, _ = _render(chart)
    twin_svg, _ = _render(fm.Chart(joined).mark_point().encode(x="v", y="m"))
    assert svg == twin_svg


def test_transform_stack_by_a_date_stacks_within_each_day():
    frame = _daily()
    stack = fm.transform_stack("v", groupby=["day"], sort=["h"])
    chart = fm.Chart(frame).transform(stack).mark_point().encode(x="y0", y="y1")
    twin = frame.with_columns(day=pl.col("day").cast(pl.Datetime("ms")).dt.epoch("ms"))
    twin_svg, _ = _render(fm.Chart(twin).transform(stack).mark_point().encode(x="y0", y="y1"))
    svg, _ = _render(chart)
    assert svg == twin_svg


def test_transform_pivot_by_a_date_makes_a_row_per_day():
    frame = _daily()
    chart = (
        fm.Chart(frame)
        .transform(fm.transform_pivot("h", "v", groupby=["day"]))
        .mark_point()
        .encode(x="a", y="b")
    )
    svg, _ = _render(chart)
    assert len(_circles(svg)) == 3


def test_transform_top_k_by_a_date_keeps_the_two_best_days():
    frame = _daily()
    chart = (
        fm.Chart(frame)
        .transform(fm.transform_top_k(2, field="v", groupby=["day"]))
        .mark_point()
        .encode(x="v", y="v")
    )
    twin = frame.with_columns(day=pl.col("day").cast(pl.Datetime("ms")).dt.epoch("ms"))
    twin_chart = (
        fm.Chart(twin)
        .transform(fm.transform_top_k(2, field="v", groupby=["day"]))
        .mark_point()
        .encode(x="v", y="v")
    )
    assert _render(chart)[0] == _render(twin_chart)[0]


def _category_clusters(svg: str) -> list[tuple[str, ...]]:
    """The points of each swarm category as a tuple of their ``cy`` values: points are grouped by
    ``cx`` (each category sits well apart from the next), categories left to right."""
    points = sorted((float(c["cx"]), c["cy"]) for c in _circles(svg))
    clusters: list[list[tuple[float, str]]] = []
    for point in points:
        if clusters and point[0] - clusters[-1][-1][0] < 30:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return [tuple(sorted(cy for _, cy in cluster)) for cluster in clusters]


def test_a_swarm_categorised_by_a_date_places_each_point_in_its_days_category():
    """The category is the x axis (a date axis against a number axis), so each point's position
    is compared with its category: three categories of eight points whose values are those of the
    same-day rows, as the epoch twin places them."""
    swarm = lambda f: fm.Chart(f).transform(fm.Swarm("k", "v")).mark_point().encode(x="k", y="v")  # noqa: E731
    svg, _ = _render(swarm(_date_frame()))
    twin_svg, _ = _render(swarm(_epoch_frame()))
    clusters = _category_clusters(svg)
    assert [len(c) for c in clusters] == [8, 8, 8]
    assert clusters == _category_clusters(twin_svg)


def test_a_boxplot_by_a_date_ordinal_draws_a_box_per_day_in_time_order():
    """The box statistic groups by day and the band axis reads the dates as categories: one box
    per day, labelled by its ISO date in time order, on a point chart's ordinal axis too."""
    labels = [d.isoformat() for d in sorted(_DAYS)]
    for mark in ("mark_point", "mark_boxplot"):
        chart = getattr(fm.Chart(_date_frame()), mark)().encode(x=fm.X("k", type_="O"), y="v")
        assert axis_tick_labels(chart.to_svg(), axis="x") == labels


# ---------------------------------------------------------------------------
# transform_impute over a date (each imputed row inserted at its key's place, as for every other key dtype)
# ---------------------------------------------------------------------------


def _series_missing_its_last_day() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day": [_DAYS[0], _DAYS[1], _DAYS[2], _DAYS[0], _DAYS[1]],
            "h": ["x", "x", "x", "y", "y"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_impute_keyed_by_a_date_adds_the_series_missing_row():
    frame = _series_missing_its_last_day()
    impute = fm.transform_impute("v", value=0, key="day", groupby=["h"])
    chart = fm.Chart(frame).transform(impute).mark_point().encode(x="day", y="v", color="h")
    # series y ends the frame, so its missing last day is inserted at the end of the frame
    written_out = pl.concat(
        [frame, pl.DataFrame({"day": [_DAYS[2]], "h": ["y"], "v": [0.0]})], how="vertical"
    )
    twin = fm.Chart(written_out).mark_point().encode(x="day", y="v", color="h")
    svg, messages = _render(chart)
    twin_svg, _ = _render(twin)
    assert messages == []
    assert svg == twin_svg
    assert len(svg_elements(mark_layer(svg), "circle", lambda a: True)) == 6


def test_impute_by_a_date_group_fills_each_days_nulls_with_that_days_mean():
    frame = pl.DataFrame(
        {
            "day": [_DAYS[0], _DAYS[0], _DAYS[0], _DAYS[1], _DAYS[1], _DAYS[1]],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "v": [10.0, None, 30.0, 4.0, 8.0, None],
        }
    )
    chart = (
        fm.Chart(frame)
        .transform(fm.transform_impute("v", method="mean", groupby=["day"]))
        .mark_point()
        .encode(x="w", y="v")
    )
    filled = frame.with_columns(pl.col("v").fill_null(pl.col("v").mean().over("day")))
    assert filled["v"].to_list() == [10.0, 20.0, 30.0, 4.0, 8.0, 6.0]
    svg, messages = _render(chart)
    twin_svg, _ = _render(fm.Chart(filled).mark_point().encode(x="w", y="v"))
    assert messages == []
    assert svg == twin_svg


# ---------------------------------------------------------------------------
# Messages name a date group as an ISO-8601 date-time
# ---------------------------------------------------------------------------


def test_a_group_with_no_usable_value_is_named_as_an_iso_date_time():
    frame = pl.DataFrame(
        {
            "day": [_DAYS[0], _DAYS[0], _DAYS[1], _DAYS[1], _DAYS[2], _DAYS[2]],
            "h": ["a", "b"] * 3,
            "v": [1.0, 2.0, None, None, 5.0, 6.0],
        }
    )
    chart = (
        fm.Chart(frame)
        .transform(fm.transform_impute("v", method="mean", groupby=["day"]))
        .mark_point()
        .encode(x="day", y="v")
    )
    _, messages = _render(chart)
    assert messages == [
        "Impute: group day=2024-01-02T00:00:00 has no usable (non-null, finite) values; "
        "the whole column's statistic was used instead"
    ]


def test_a_fit_warning_names_each_date_group_as_the_text_twin_does():
    """``Logistic`` cannot fit any day of the frame (its responses are all one value); its
    warnings, in group order, name each day as ISO-8601 text, the same as a text key does."""
    _, messages = _render(_DOORS["Logistic_unfittable"](_date_frame()))
    _, text_messages = _render(_DOORS["Logistic_unfittable"](_iso_frame()))
    assert messages == text_messages
    assert messages == [_fit_message("Logistic", f"k={_ISO[i]}", FIT_SEPARATED) for i in (2, 0, 1)]


@pytest.mark.parametrize(
    "kind", ["Date", "Datetime-ms", "Datetime-us", "Datetime-ns", "Duration-ms"]
)
def test_a_naive_instant_is_named_the_same_whatever_its_unit(kind: str):
    _, messages = _render(_DOORS["Logistic_unfittable"](_kind_frame(kind)))
    assert messages == [_fit_message("Logistic", f"k={_ISO[i]}", FIT_SEPARATED) for i in (2, 0, 1)]


# ---------------------------------------------------------------------------
# A null date key is skipped and counted as any null key is
# ---------------------------------------------------------------------------


def _null_key_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """The date frame and its epoch twin with the key of the first two rows null."""
    date = _date_frame().with_columns(
        k=pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col("k"))
    )
    twin = _epoch_frame().with_columns(
        k=pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col("k"))
    )
    return date, twin


_SKIPPING_DOORS = ["BoxStats", "Kde", "aggregate", "Summary", "Bin", "QQ"]
# The doors that pool a null key into a group of its own, or fit the whole batch beside it.
_POOLING_DOORS = ["window", "stack", "join_aggregate", "Smooth", "Robust", "impute_groupby"]


@pytest.mark.parametrize("door", _SKIPPING_DOORS + _POOLING_DOORS)
def test_a_null_date_key_is_dropped_and_counted_as_a_null_numeric_key_is(door: str):
    date, twin = _null_key_frames()
    twin_svg, twin_messages = _render(_DOORS[door](twin))
    svg, messages = _render(_DOORS[door](date))
    if door in _SKIPPING_DOORS:
        assert twin_messages, (
            f"{door}: the twin names no dropped null key, so the pin proves nothing"
        )
    assert messages == twin_messages
    assert svg == twin_svg


def test_a_null_date_key_names_the_statistic_the_row_count_and_the_column():
    date, twin = _null_key_frames()
    svg, messages = _render(_DOORS["BoxStats"](date))
    twin_svg, _ = _render(_DOORS["BoxStats"](twin))
    assert messages == [_null_key_dropped_message("BoxStats", 2, "k")]
    assert svg == twin_svg


def test_a_date_key_null_on_every_row_matches_the_numeric_twin():
    date = _date_frame().with_columns(k=pl.lit(None, dtype=pl.Date))
    twin = _epoch_frame().with_columns(k=pl.lit(None, dtype=pl.Int64))
    for door in ("BoxStats", "aggregate", "Kde"):
        svg, messages = _render(_DOORS[door](date))
        twin_svg, twin_messages = _render(_DOORS[door](twin))
        assert messages == twin_messages, door
        assert svg == twin_svg, door


# ---------------------------------------------------------------------------
# A zero-row frame with a date group column
# ---------------------------------------------------------------------------


# What each door does over a frame with no rows, whatever the key's dtype: the chart notes that it
# has no rows to draw, or the statistic degrades once for the whole frame, or the call raises.
_ZERO_ROW_OUTCOME: dict[str, tuple[str, list[str] | str]] = {
    "Kde": ("warns", [_NO_ROWS_TO_DRAW]),
    "Violin": ("warns", [_NO_ROWS_TO_DRAW]),
    "BoxStats": ("warns", [_NO_ROWS_TO_DRAW]),
    "Smooth": ("warns", [_whole_fit_message("Smooth", FIT_FEW_POINTS)]),
    "Summary": ("warns", [_NO_ROWS_TO_DRAW]),
    "ErrorExtent": ("warns", [_NO_ROWS_TO_DRAW]),
    "Bin": ("warns", [_NO_ROWS_TO_DRAW]),
    "QQ": ("warns", [_NO_ROWS_TO_DRAW]),
    "LetterValue": ("warns", [_NO_ROWS_TO_DRAW]),
    "Robust": ("warns", [_whole_fit_message("Robust", FIT_FEW_POINTS)]),
    "Glm": ("warns", [_whole_fit_message("Glm", FIT_FEW_POINTS)]),
    "Logistic": ("warns", [_whole_fit_message("Logistic", FIT_FEW_POINTS)]),
    "Logistic_unfittable": ("warns", [_whole_fit_message("Logistic", FIT_FEW_POINTS)]),
    "Kde2D": ("warns", [_NO_ROWS_TO_DRAW]),
    "aggregate": ("warns", [_NO_ROWS_TO_DRAW]),
    "aggregate_key": ("warns", [_NO_ROWS_TO_DRAW]),
    "Outliers": ("warns", [_NO_ROWS_TO_DRAW]),
    "join_aggregate": ("warns", [_NO_ROWS_TO_DRAW]),
    "window": ("warns", [_NO_ROWS_TO_DRAW]),
    "stack": ("warns", [_NO_ROWS_TO_DRAW]),
    "pivot": ("warns", [_NO_ROWS_TO_DRAW]),
    "top_k": ("warns", [_NO_ROWS_TO_DRAW]),
    "density": ("warns", [_NO_ROWS_TO_DRAW]),
    "regression": ("warns", [_NO_ROWS_TO_DRAW]),
    "loess": ("warns", [_NO_ROWS_TO_DRAW]),
    "impute_groupby": (
        "warns",
        [
            "Impute: the column has no usable (non-null, finite) values at all; used the mean "
            "method's degenerate default instead",
            _NO_ROWS_TO_DRAW,
        ],
    ),
    "encoding_aggregate_detail": ("warns", [_NO_ROWS_TO_DRAW]),
}


def _outcome(chart: fm.Chart, output: str) -> tuple[str, str | list[str], str]:
    """What a render of ``chart`` does in ``output``: the warnings it gives (with the page it
    drew) or the error it raises."""
    try:
        if output == "static":
            page, messages = _render(chart)
        else:
            page, messages = _render_capturing(lambda: chart.interactive().scene_json)
    except ValueError as raised:
        return "raises", str(raised), ""
    return "warns", messages, page


def test_every_door_has_a_zero_row_outcome():
    assert set(_ZERO_ROW_OUTCOME) == set(_DOORS)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("door", list(_DOORS))
def test_a_zero_row_frame_with_a_date_key_does_what_its_epoch_twin_does(door: str, output: str):
    """No rows is no group, whatever the key's dtype: the date frame gives the twin's page and
    full message list (or its refusal), and that outcome is the door's own, static and
    interactive."""
    date = _outcome(_DOORS[door](_date_frame().head(0)), output)
    twin = _outcome(_DOORS[door](_epoch_frame().head(0)), output)
    assert date == twin
    assert date[:2] == _ZERO_ROW_OUTCOME[door]


# ---------------------------------------------------------------------------
# The ends of the instant range group, and sort, like any other instant
# ---------------------------------------------------------------------------


def _far_instants_frame() -> pl.DataFrame:
    """Keys a hundred million years either side of 1970 (no calendar date holds them) and one
    ordinary day holding two instants (00:00 and 12:00), the rows listing the ordinary day first."""
    far = {0: -(2**62), 1: 2**62, 2: _EPOCH_MS[0]}
    millis = pl.col("di").replace_strict(far, return_dtype=pl.Int64)
    millis = millis + pl.when((pl.col("di") == 2) & (pl.col("r") % 2 == 1)).then(
        _HALF_DAY_MS
    ).otherwise(0)
    return _with_key(millis).with_columns(pl.col("k").cast(pl.Datetime("ms")))


@pytest.mark.parametrize("door", ["aggregate_key", "BoxStats", "Kde", "Summary"])
def test_an_instant_far_outside_the_calendar_groups_beside_ordinary_ones(door: str):
    """A key a hundred million years either side of 1970 needs no calendar date to be a group,
    and the two instants of the ordinary day beside it stay two groups."""
    date = _far_instants_frame()
    twin = date.with_columns(pl.col("k").dt.epoch("ms"))
    svg, messages = _render(_DOORS[door](date))
    twin_svg, twin_messages = _render(_DOORS[door](twin))
    assert _circles(twin_svg)
    assert svg == twin_svg
    assert messages == twin_messages


def test_a_fit_warning_names_an_instant_far_outside_the_calendar_by_its_epoch_offset():
    """An instant no calendar date holds is named by its raw value and unit; the ordinary day
    beside it is named as ISO-8601 text. Groups are listed in first-appearance order."""
    _, messages = _render(_DOORS["Logistic_unfittable"](_far_instants_frame()))
    groups = [
        (_ISO[0], FIT_SINGULAR),
        ("2024-01-01T12:00:00", FIT_SINGULAR),
        (f"{-(2**62)} ms since the Unix epoch", FIT_SEPARATED),
        (f"{2**62} ms since the Unix epoch", FIT_SEPARATED),
    ]
    assert messages == [_fit_message("Logistic", f"k={n}", reason) for n, reason in groups]


# Where a door takes an order from the key itself, instants order by time. The far instants sort
# by time as -2**62 < 2024 < 2**62, which their names (text) do not.
_KEY_ORDERED_DOORS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "window_sorted_by_key": _through(
        fm.transform_window({"op": "row_number", "as": "c"}, sort=["k"], groupby=["h"]), "w", "c"
    ),
    "stack_ordered_by_key": _through(
        fm.transform_stack("v", groupby=["h"], sort=["k"]), "y0", "y1"
    ),
}


@pytest.mark.parametrize("door", list(_KEY_ORDERED_DOORS))
def test_a_door_that_orders_by_a_date_key_orders_instants_by_time(door: str):
    frame = _far_instants_frame()
    twin = frame.with_columns(pl.col("k").dt.epoch("ms"))
    twin_svg, _ = _render(_KEY_ORDERED_DOORS[door](twin))
    assert _circles(twin_svg)
    svg, _ = _render(_KEY_ORDERED_DOORS[door](frame))
    assert svg == twin_svg


# ---------------------------------------------------------------------------
# Groups are instants: not days, not texts
# ---------------------------------------------------------------------------

# 2024-10-26T23:30Z, 2024-10-27T00:30Z and 2024-10-27T01:30Z: in Europe/Paris the last two both
# read 02:30 on the wall clock (the clocks go back at 03:00 local), so a key grouped by its local
# text would merge two instants.
_FALL_BACK_MS = [1729985400000, 1729989000000, 1729992600000]


def _instants_frame(kind: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A frame whose keys are instants and the same frame with the keys as epoch milliseconds.

    ``twelve-hours``: each day holds two instants (00:00 and 12:00, by the row's repeat), so a
    grouping by calendar day would make three groups where there are six; the far value of ``vo``
    is an outlier inside its own half-day but not among the whole day's two clusters.
    ``fall-back``: three instants, two of which share a local wall-clock text in Europe/Paris."""
    base = _base()
    if kind == "twelve-hours":
        millis = pl.col("di").replace_strict(dict(enumerate(_EPOCH_MS)), return_dtype=pl.Int64)
        millis = millis + pl.when(pl.col("r") % 2 == 1).then(_HALF_DAY_MS).otherwise(0)
        base = base.with_columns(
            vo=100.0 * pl.col("di")
            + 50.0 * (pl.col("r") % 2)
            + 0.1 * pl.col("r")
            + pl.when((pl.col("r") == 3) & (pl.col("h") == "a")).then(40.0).otherwise(0.0)
        )
    else:
        millis = pl.col("di").replace_strict(dict(enumerate(_FALL_BACK_MS)), return_dtype=pl.Int64)
    twin = base.with_columns(k=millis).drop("di")
    instant = pl.col("k").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    if kind == "twelve-hours":
        return twin.with_columns(instant.dt.replace_time_zone(None)), twin
    return twin.with_columns(instant.dt.convert_time_zone("Europe/Paris")), twin


@pytest.mark.parametrize("kind", ["twelve-hours", "fall-back"])
@pytest.mark.parametrize(
    "door",
    [
        "aggregate_key",
        "BoxStats",
        "Kde",
        "window",
        "Summary",
        "impute_groupby",
        "top_k",
        "Outliers",
    ],
)
def test_a_group_is_an_instant_not_a_calendar_day_nor_a_wall_clock_text(kind: str, door: str):
    frame, twin = _instants_frame(kind)
    twin_svg, _ = _render(_DOORS[door](twin))
    assert _circles(twin_svg)
    svg, _ = _render(_DOORS[door](frame))
    assert svg == twin_svg


@pytest.mark.parametrize(
    "kind,groups", [("twelve-hours", 6), ("fall-back", 3)], ids=["twelve-hours", "fall-back"]
)
def test_an_aggregate_over_instants_makes_one_group_per_instant(kind: str, groups: int):
    """The count of groups is explicit: one per distinct instant."""
    frame, _ = _instants_frame(kind)
    chart = (
        fm.Chart(frame)
        .transform(fm.transform_aggregate({"field": "v", "fn": "count", "as": "n"}, groupby=["k"]))
        .mark_point()
        .encode(x="k", y="n")
    )
    svg, _ = _render(chart)
    assert len(_circles(svg)) == groups


def test_transform_pivot_by_a_date_matches_its_epoch_twin():
    frame = _daily()
    twin = frame.with_columns(day=pl.col("day").cast(pl.Datetime("ms")).dt.epoch("ms"))
    pivot = fm.transform_pivot("h", "v", groupby=["day"])
    chart = lambda f: fm.Chart(f).transform(pivot).mark_point().encode(x="a", y="b")  # noqa: E731
    twin_svg, _ = _render(chart(twin))
    assert len(_circles(twin_svg)) == 3
    assert _render(chart(frame))[0] == twin_svg


def test_impute_keyed_by_a_date_matches_its_epoch_twin_and_keeps_the_day_axis():
    frame = _series_missing_its_last_day()
    twin = frame.with_columns(day=pl.col("day").cast(pl.Datetime("ms")).dt.epoch("ms"))
    impute = fm.transform_impute("v", value=0, key="day", groupby=["h"])
    chart = lambda f: fm.Chart(f).transform(impute).mark_point().encode(x="day", y="v", color="h")  # noqa: E731
    svg, _ = _render(chart(frame))
    twin_svg, _ = _render(chart(twin))
    assert [(c["cx"], c["cy"], c["fill"]) for c in _circles(svg)] == [
        (c["cx"], c["cy"], c["fill"]) for c in _circles(twin_svg)
    ]
    assert all(":" in label for label in axis_tick_labels(svg, axis="x"))


def test_impute_by_a_date_group_matches_its_epoch_twin():
    frame = pl.DataFrame(
        {
            "day": [_DAYS[0], _DAYS[0], _DAYS[0], _DAYS[1], _DAYS[1], _DAYS[1]],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "v": [10.0, None, 30.0, 4.0, 8.0, None],
        }
    )
    twin = frame.with_columns(day=pl.col("day").cast(pl.Datetime("ms")).dt.epoch("ms"))
    impute = fm.transform_impute("v", method="mean", groupby=["day"])
    chart = lambda f: fm.Chart(f).transform(impute).mark_point().encode(x="w", y="v")  # noqa: E731
    assert _render(chart(frame)) == _render(chart(twin))


# ---------------------------------------------------------------------------
# A zoned group is named in UTC wall-clock time with no zone marker, as the axis shows it
# ---------------------------------------------------------------------------


def _fall_back_frame(zone: str) -> pl.DataFrame:
    """The three fall-back instants as a column of ``zone`` (in Europe/Paris two read 02:30)."""
    _, twin = _instants_frame("fall-back")
    instant = pl.col("k").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    return twin.with_columns(instant.dt.convert_time_zone(zone))


# Group names in first-appearance order: the frame lists its instants out of time order.
_ZONED_NAMES = {
    "UTC": ["2024-10-27T01:30:00", "2024-10-26T23:30:00", "2024-10-27T00:30:00"],
    "Europe/Paris": ["2024-10-27T01:30:00", "2024-10-26T23:30:00", "2024-10-27T00:30:00"],
}


@pytest.mark.parametrize("zone", ["UTC", "Europe/Paris"])
@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_zoned_group_is_named_in_utc_and_agrees_with_the_axis(zone: str, output: str):
    frame = _fall_back_frame(zone)
    chart = _DOORS["Logistic_unfittable"](frame)
    _, messages = _render_capturing(
        chart.to_svg if output == "static" else lambda: chart.interactive().scene_json
    )
    names = _ZONED_NAMES[zone]
    assert messages == [_fit_message("Logistic", f"k={n}", FIT_SEPARATED) for n in names]
    # the name is the instant the same chart's axis labels, so the two cannot drift apart
    ticks = axis_tick_labels(
        fm.Chart(frame.unique("k")).mark_point().encode(x="k", y="v").to_svg(), axis="x"
    )
    for name in names:
        assert name.removeprefix(name[:11]) in ticks, (name, ticks)


def test_the_two_paris_instants_that_read_alike_locally_get_two_utc_names():
    frame = _fall_back_frame("Europe/Paris")
    assert _normalized_type(frame) == pa.timestamp("ms", tz="Europe/Paris")
    local = frame["k"].dt.strftime("%H:%M").unique().to_list()
    assert len(local) == 2, "the frame must hold two instants with one wall-clock text"
    _, messages = _render(_DOORS["Logistic_unfittable"](frame))
    assert messages == [
        _fit_message("Logistic", f"k={n}", FIT_SEPARATED) for n in _ZONED_NAMES["Europe/Paris"]
    ]


@pytest.mark.parametrize(
    "kind,names",
    [
        ("Datetime-UTC", [_ISO[2], _ISO[0], _ISO[1]]),
        # Paris-local midnights on 5 Jan, 1 Jan and 2 Jan 2024, one hour ahead of UTC
        ("Datetime-Paris", ["2024-01-04T23:00:00", "2023-12-31T23:00:00", "2024-01-01T23:00:00"]),
    ],
)
def test_a_zoned_midnight_is_named_by_its_utc_instant(kind: str, names: list[str]):
    """Paris-local midnight on 2 Jan 2024 is 23:00 UTC on 1 Jan; a UTC column's midnight is its own."""
    frame = _kind_frame(kind)
    assert _normalized_type(frame) == _KIND_TYPES[kind]
    _, messages = _render(_DOORS["Logistic_unfittable"](frame))
    assert messages == [_fit_message("Logistic", f"k={n}", FIT_SEPARATED) for n in names]


# ---------------------------------------------------------------------------
# Every door the grouped chart can be built through
# ---------------------------------------------------------------------------


def _summary_chart(frame: pl.DataFrame, y: str = "upper") -> fm.Chart:
    return (
        fm.Chart(frame).transform(fm.Summary("v", groupby=["k"])).mark_point().encode(x="mean", y=y)
    )


_COMPOSITIONS: dict[str, Callable[[pl.DataFrame], Any]] = {
    "plus": lambda df: _summary_chart(df) + _summary_chart(df, "lower"),
    "layer": lambda df: _summary_chart(df).layer(
        fm.Layer(mark="point", encoding={"x": "mean", "y": "lower"})
    ),
    "LayerChart": lambda df: fm.LayerChart(_summary_chart(df), _summary_chart(df, "lower")),
    "facet_col": lambda df: _summary_chart(df.with_columns(f=pl.col("h"))).facet(col="f"),
    "facet_row": lambda df: _summary_chart(df.with_columns(f=pl.col("h"))).facet(row="f"),
    "hconcat": lambda df: fm.hconcat(_summary_chart(df), _summary_chart(df, "lower")),
}


@pytest.mark.parametrize("composition", list(_COMPOSITIONS))
def test_a_grouped_statistic_over_a_date_draws_on_every_composition_door(composition: str):
    build = _COMPOSITIONS[composition]
    twin_svg, twin_messages = _render_capturing(build(_epoch_frame()).to_svg)
    svg, messages = _render_capturing(build(_date_frame()).to_svg)
    assert _circles(twin_svg) or "<circle" in twin_svg
    assert svg == twin_svg
    assert messages == twin_messages


@pytest.mark.parametrize("composition", list(_COMPOSITIONS))
def test_a_grouped_statistic_over_a_date_draws_the_same_interactive_scene_on_every_door(
    composition: str,
):
    build = _COMPOSITIONS[composition]
    scene = lambda frame: _render_capturing(build(frame).interactive().to_html)[0]  # noqa: E731
    assert scene(_date_frame()) == scene(_epoch_frame())


# ---------------------------------------------------------------------------
# Dtypes that stay unsupported keep one refusal text, shared by every grouped door
# ---------------------------------------------------------------------------

_STILL_REFUSED: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "Decimal": lambda df: df.with_columns(k=pl.col("w").cast(pl.Decimal(10, 2))),
    "Time": lambda df: df.with_columns(
        k=(pl.col("w").cast(pl.Int64) * 1_000_000_000).cast(pl.Time)
    ),
    "List": lambda df: df.with_columns(k=pl.concat_list("v", "w")),
}

# The dtype as the engine names it when the column reaches it.
_STILL_REFUSED_DTYPE = {
    "Decimal": "Decimal128(10, 2)",
    "Time": "Time64(Nanosecond)",
    "List": "LargeList(Field { data_type: Float64, nullable: true })",
}

# Each door that reads a group key through the shared reader, with the entry that speaks.
_REFUSING_DOORS = {
    "aggregate": "data_aggregate",
    "BoxStats": "stat_box_stats",
    "Kde": "stat_kde",
    "window": "data_window",
    "join_aggregate": "data_join_aggregate",
    "Summary": "stat_summary",
    "stack": "data_stack",
}


@pytest.mark.parametrize("kind", list(_STILL_REFUSED))
@pytest.mark.parametrize("door", list(_REFUSING_DOORS))
def test_a_dtype_still_unsupported_is_refused_with_the_one_text_by_every_door(kind: str, door: str):
    frame = _STILL_REFUSED[kind](_date_frame())
    with pytest.raises(ValueError) as raised:
        _DOORS[door](frame).to_svg()
    assert str(raised.value) == _group_dtype_refusal(
        _REFUSING_DOORS[door], "groupby", "k", _STILL_REFUSED_DTYPE[kind]
    )


@pytest.mark.parametrize("kind", list(_STILL_REFUSED))
def test_transform_impute_refuses_an_unsupported_key_in_the_same_text(kind: str):
    frame = _STILL_REFUSED[kind](_series_missing_its_last_day().with_columns(w=pl.col("v")))
    impute = fm.transform_impute("v", value=0, key="k", groupby=["h"])
    with pytest.raises(ValueError) as raised:
        fm.Chart(frame).transform(impute).mark_point().encode(x="w", y="v").to_svg()
    assert str(raised.value) == _group_dtype_refusal(
        "data_impute", "key", "k", _STILL_REFUSED_DTYPE[kind]
    )


@pytest.mark.parametrize("kind", list(_STILL_REFUSED))
def test_a_swarm_category_refuses_an_unsupported_dtype_in_the_same_text(kind: str):
    frame = _STILL_REFUSED[kind](_date_frame())
    with pytest.raises(ValueError) as raised:
        fm.Chart(frame).transform(fm.Swarm("k", "v")).mark_point().encode(x="k", y="v").to_svg()
    assert str(raised.value) == _group_dtype_refusal(
        "stat_swarm", "category", "k", _STILL_REFUSED_DTYPE[kind]
    )


# ---------------------------------------------------------------------------
# A chart with no temporal group key is byte-identical to before
# ---------------------------------------------------------------------------

_PLAIN_KINDS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "Utf8": lambda df: df.with_columns(k=pl.col("di").cast(pl.Utf8)).drop("di"),
    "Int64": lambda df: df.with_columns(k=pl.col("di").cast(pl.Int64)).drop("di"),
    "UInt8": lambda df: df.with_columns(k=pl.col("di").cast(pl.UInt8)).drop("di"),
    "Float64": lambda df: df.with_columns(k=pl.col("di").cast(pl.Float64)).drop("di"),
    "Boolean": lambda df: df.with_columns(k=pl.col("di") > 0).drop("di"),
}

# Each digest is the MD5 of the SVGs and warnings the door gave over the five plain key dtypes
# above, measured before dates grouped.
_PLAIN_KEY_DIGESTS = {
    "aggregate": "8cf7e322be90a183cee29ab950a249c7",
    "BoxStats": "21413847b62b339fee92fee827a07320",
    "Kde": "4758b63d7880167d8e2c55fba3ee81bf",
    "window": "daffb66e96a4ce6f3eb2577f0b2de9f5",
    "Summary": "afe9373861275eae39788e31e33eabb3",
    "impute_groupby_complete": "1e5c7f39c9b906091b5c43c066bf0cc0",
    "Logistic": "3982c6899ad9b39de9dc079b147e15c6",
}


@pytest.mark.parametrize("door", list(_PLAIN_KEY_DIGESTS))
def test_a_door_over_a_non_temporal_key_is_byte_identical_to_before(door: str):
    digest = hashlib.md5()
    for build in _PLAIN_KINDS.values():
        svg, messages = _render(_PLAIN_DOORS[door](build(_base())))
        digest.update(svg.encode())
        digest.update("\n".join(messages).encode())
    assert digest.hexdigest() == _PLAIN_KEY_DIGESTS[door]


def test_a_chart_with_no_group_key_is_unchanged_by_the_presence_of_a_date_column():
    """A date column the chart never groups by (here on an axis) draws as it always did."""
    frame = _daily()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        svg = fm.Chart(frame).mark_point().encode(x="day", y="v").to_svg()
    assert len(_circles(svg)) == 12
