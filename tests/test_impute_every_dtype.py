"""``impute=`` on a value channel completes every series the layer draws.

Writing ``impute=`` on ``y`` (or on ``x``, with ``y`` as the key) completes each series a layer draws
over the keys the layer's rows hold: a series that lacks a key gains one row there, inserted at the
key's place in the series, and a null or ``NaN`` value in a series is filled the same way. It holds on
every key dtype, every option of the dict and every layering door, and each layer completes its own
series, so an ``impute=`` written on one layer is never read by a sibling.

How each case is read:

* A *written-out twin* is the same chart without ``impute=`` over the frame with the missing rows
  written down at their key's place (and the filled cells written in). The chart with ``impute=``
  draws it byte for byte, warnings included, and the interactive route builds the same scene
  value for value.
* A twin is only evidence when the chart without the missing rows draws something else, so each
  comparison also asserts that the chart over the frame as the caller wrote it differs.
* A layered door whose twin cannot be the same structure (a sibling must keep drawing the caller's
  rows) is read through the marks each layer drew: the circles and the polyline vertices.
* An unchanged behavior is a chart compared with the one drawn without ``impute=``.
"""

from __future__ import annotations

import datetime as dt
import decimal
import re
from collections.abc import Callable, Sequence
from typing import Any

import polars as pl
import pyarrow as pa
import pytest

import ferrum as fm
from tests._svg_extents import text_runs
from tests._svg_rects import mark_layer, polylines, svg_elements
from tests.test_null_category_one_rule import (
    _FACET_DOORS,
    _IMPUTE_COMPLETED,
    _IMPUTE_FRAME,
    _route,
    _scene_marks,
)
from tests.test_null_group_warning import (
    IMPUTE_KIND_CASES,
    _facet_key_dropped_message,
    _impute_kind_refusal,
    _null_key_dropped_message,
    _census,
    _render_capturing,
    _slot_message,
    _unreadable_key_refusal,
)

_NAN = float("nan")
_INF = float("inf")

# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _y(imputed: bool, field: str = "v", **options: Any) -> Any:
    """The value channel: ``field`` with an impute dict (``value`` 0 unless options are given), or bare."""
    return fm.Y(field, impute=options or {"value": 0}) if imputed else field


def _chart(
    frame: Any,
    mark: str = "mark_point",
    *,
    y: Any = "v",
    x: Any = "t",
    color: Any = "h",
    **channels: Any,
) -> fm.Chart:
    encoding: dict[str, Any] = {"x": x, "y": y, **channels}
    if color is not None:
        encoding["color"] = color
    return getattr(fm.Chart(frame), mark)().encode(**encoding)


def _assert_twin(
    imputed: Callable[[], Any],
    written: Callable[[], Any],
    *,
    plain: Callable[[], Any] | None = None,
    shown: list[str] | None = None,
    twin_shown: list[str] | None = None,
    interactive: bool = False,
) -> str:
    """``imputed()`` draws what ``written()`` draws (the frame with the missing rows written out, no
    ``impute=``), and ``plain()`` (the caller's rows) draws something else.

    ``shown`` is the ordered full warning list ``imputed()`` raises and ``twin_shown`` the one the
    twin raises (the twin's, when left out). With neither given the two must raise the same list.
    Each chart is built inside the capture, so a build-time warning is counted too."""
    svg, messages = _render_capturing(lambda: imputed().to_svg())
    twin_svg, twin_messages = _render_capturing(lambda: written().to_svg())
    if shown is not None:
        assert messages == shown
    if twin_shown is None and shown is not None:
        twin_shown = shown
    if twin_shown is not None:
        assert twin_messages == twin_shown
    if shown is None and twin_shown is None:
        assert messages == twin_messages
    assert svg == twin_svg
    if plain is not None:
        plain_svg, _ = _render_capturing(lambda: plain().to_svg())
        assert svg != plain_svg
    if interactive:
        route_shown, scene = _route(imputed())
        twin_route_shown, twin_scene = _route(written())
        assert route_shown == twin_route_shown
        assert _scene_marks(scene) > 0
        # the whole scene, so an imputed row drawn at another value or place fails it
        assert scene == twin_scene
    return svg


def _circles(svg: str) -> list[tuple[float, float]]:
    """The centre of each circle the marks drew, in drawing order."""
    return [
        (float(a["cx"]), float(a["cy"]))
        for a in svg_elements(mark_layer(svg), "circle", lambda a: "cx" in a)
    ]


def _glyphs(svg: str) -> int:
    """How many point glyphs (a circle, or the square or path another shape draws) the marks drew."""
    return len(re.findall(r"<(?:circle|rect|path|polygon)\b", mark_layer(svg)))


def _vertices(svg: str) -> list[int]:
    """The vertex count of each drawn polyline, in drawing order."""
    return [len(points) for _, points in polylines(mark_layer(svg))]


# ---------------------------------------------------------------------------
# Frames: series ``x`` holds every key, series ``y`` lacks the middle one
# ---------------------------------------------------------------------------

#: A key column of the first ``n`` keys' dtype, from each key's ordinal: ``(maker, key count)``.
_KeyMaker = Callable[[Sequence[int]], Any]


def _ints(dtype: Any) -> _KeyMaker:
    return lambda ks: pl.Series("t", [k + 1 for k in ks], dtype=dtype)


def _datetimes(unit: str, zone: str | None = None) -> _KeyMaker:
    def make(ks: Sequence[int]) -> pl.Series:
        values = pl.Series(
            "t", [dt.datetime(2024, 1, 1 + k, 6) for k in ks], dtype=pl.Datetime(unit)
        )
        return values.dt.replace_time_zone(zone) if zone else values

    return make


def _durations(unit: str) -> _KeyMaker:
    return lambda ks: pl.Series(
        "t", [dt.timedelta(days=k + 1) for k in ks], dtype=pl.Duration(unit)
    )


def _words(dtype: Any) -> _KeyMaker:
    return lambda ks: pl.Series("t", ["abc"[k] for k in ks], dtype=dtype)


def _arrow_words(wrap: Callable[[pa.Array], pa.Array]) -> _KeyMaker:
    return lambda ks: wrap(pa.array(["abc"[k] for k in ks]))


def _arrow_string_view(ks: Sequence[int]) -> pa.Array:
    """A ``string_view`` array built with ``type=``: a cast one segfaults ``pl.from_arrow``."""
    return pa.array(["abc"[k] for k in ks], type=pa.string_view())


_KEYS: dict[str, tuple[_KeyMaker, int]] = {
    "Int8": (_ints(pl.Int8), 3),
    "Int16": (_ints(pl.Int16), 3),
    "Int32": (_ints(pl.Int32), 3),
    "Int64": (_ints(pl.Int64), 3),
    "UInt8": (_ints(pl.UInt8), 3),
    "UInt16": (_ints(pl.UInt16), 3),
    "UInt32": (_ints(pl.UInt32), 3),
    "UInt64": (_ints(pl.UInt64), 3),
    "Float32": (_ints(pl.Float32), 3),
    "Float64": (_ints(pl.Float64), 3),
    "Date": (lambda ks: pl.Series("t", [dt.date(2024, 1, 1 + k) for k in ks]), 3),
    "Datetime-us": (_datetimes("us"), 3),
    "Datetime-ms": (_datetimes("ms"), 3),
    "Datetime-ns": (_datetimes("ns"), 3),
    "Datetime-UTC": (_datetimes("us", "UTC"), 3),
    "Datetime-Paris": (_datetimes("us", "Europe/Paris"), 3),
    "Duration-us": (_durations("us"), 3),
    "Duration-ms": (_durations("ms"), 3),
    "Utf8": (_words(pl.Utf8), 3),
    "Categorical": (_words(pl.Categorical), 3),
    "Enum": (_words(pl.Enum(["a", "b", "c"])), 3),
    "LargeUtf8": (_arrow_words(lambda a: a.cast(pa.large_string())), 3),
    "Dictionary": (_arrow_words(lambda a: a.dictionary_encode()), 3),
    "Utf8View": (_arrow_string_view, 3),
    "Boolean": (lambda ks: pl.Series("t", [bool(k) for k in ks]), 2),
}

#: ``(key ordinal, series, value)``; the value is the series' own, 0 for the row the chart imputes.
_Row = tuple[int, str, float]


def _frame_of(key: Any, rows: Sequence[_Row], **columns: Sequence[Any]) -> Any:
    """The frame of ``rows`` keyed by ``key`` (a polars Series or a pyarrow array), as a polars
    frame, or a pyarrow table when the key is a pyarrow array."""
    data = {"t": key, "h": [r[1] for r in rows], "v": [r[2] for r in rows], **columns}
    return pa.table(data) if isinstance(key, pa.Array) else pl.DataFrame(data)


def _grid(make: _KeyMaker, count: int, *, panels: Sequence[str] = ()) -> tuple[Any, Any]:
    """The frame as the caller wrote it and its written-out twin.

    Series ``x`` holds every key; series ``y`` lacks key 1 (the last key when there are two), so the
    twin writes ``(key 1, y)`` with value 0 at its place in ``y``. With ``panels`` the rows repeat in
    each facet panel, a column ``f`` naming it."""
    kept = [k for k in range(count) if k != 1]
    x_rows: list[_Row] = [(k, "x", float(k + 1)) for k in range(count)]
    wrote: list[_Row] = x_rows + [(k, "y", 10.0 + k) for k in kept]
    twin: list[_Row] = x_rows + [(k, "y", 0.0 if k == 1 else 10.0 + k) for k in range(count)]

    def build(rows: Sequence[_Row]) -> Any:
        copies = list(panels) or [None]
        repeated = [row for _ in copies for row in rows]
        key = make([r[0] for r in repeated])
        extra = {"f": [name for name in copies for _ in rows]} if panels else {}
        return _frame_of(key, repeated, **extra)

    return build(wrote), build(twin)


# ---------------------------------------------------------------------------
# An integer x: the first case
# ---------------------------------------------------------------------------


def test_an_integer_x_completes_the_series_that_lacks_a_key():
    """``t = [1, 2, 3, 1, 3]`` as integers: series ``y`` gains ``(2, y)``, drawn where the frame with
    that row written out draws it."""
    frame = pl.DataFrame(
        {"t": [1, 2, 3, 1, 3], "h": ["x", "x", "x", "y", "y"], "v": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    twin = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3],
            "h": ["x", "x", "x", "y", "y", "y"],
            "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
        }
    )
    svg = _assert_twin(
        lambda: _chart(frame, y=_y(True)),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )
    assert len(_circles(svg)) == 6


# ---------------------------------------------------------------------------
# Every key dtype, on every mark that connects or places a key
# ---------------------------------------------------------------------------

_KEYED_MARKS = ["mark_point", "mark_line", "mark_area", "mark_bar"]


@pytest.mark.parametrize("mark", _KEYED_MARKS)
@pytest.mark.parametrize("kind", sorted(_KEYS))
def test_a_series_is_completed_on_every_key_dtype_and_draws_as_its_written_out_twin(
    kind: str, mark: str
):
    """The imputed key rides at the column's own dtype (an axis of dates or text would not match the
    twin's otherwise), and a ``Boolean`` key draws the line its two points make with no warning."""
    make, count = _KEYS[kind]
    frame, twin = _grid(make, count)
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)),
        lambda: _chart(twin, mark),
        plain=lambda: _chart(frame, mark),
        shown=[],
    )


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize("kind", ["Int64", "Date", "Utf8", "Boolean"])
def test_an_interactive_route_completes_each_dtype_family_as_the_twin_does(kind: str, mark: str):
    make, count = _KEYS[kind]
    frame, twin = _grid(make, count)
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)),
        lambda: _chart(twin, mark),
        plain=lambda: _chart(frame, mark),
        shown=[],
        interactive=True,
    )


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize("kind", ["Int64", "Date", "Utf8", "Boolean"])
def test_each_facet_panel_completes_its_own_series_for_every_dtype_family(
    kind: str, mark: str, door: str
):
    """Each panel holds both series; each gains its own missing row, in its own panel."""
    make, count = _KEYS[kind]
    frame, twin = _grid(make, count, panels=("F1", "F2"))
    facet = _FACET_DOORS[door]
    _assert_twin(
        lambda: facet(_chart(frame, mark, y=_y(True))),
        lambda: facet(_chart(twin, mark)),
        plain=lambda: facet(_chart(frame, mark)),
        shown=[],
        interactive=True,
    )


# ---------------------------------------------------------------------------
# Where an imputed row goes
# ---------------------------------------------------------------------------

_SIMPLE_KEYS = [1, 2, 3, 1, 3]
_SIMPLE_SERIES = ["x", "x", "x", "y", "y"]
_SIMPLE_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]


def _simple(keys: Sequence[Any], series: Sequence[str], values: Sequence[float]) -> pl.DataFrame:
    return pl.DataFrame({"t": list(keys), "h": list(series), "v": list(values)})


_SIMPLE = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, _SIMPLE_VALUES)
#: The frame with ``(2, y)`` written at its place in series ``y``.
_SIMPLE_TWIN = _simple(
    [1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [1.0, 2.0, 3.0, 4.0, 0.0, 5.0]
)
#: The same rows with the imputed one appended after every caller row, where it was drawn last.
_SIMPLE_APPENDED = _simple(
    [1, 2, 3, 1, 3, 2], ["x", "x", "x", "y", "y", "y"], [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
)


@pytest.mark.parametrize("mark", ["mark_line", "mark_area"])
def test_an_interior_missing_key_is_drawn_in_key_order_on_a_connected_mark(mark: str):
    """The line joins ``1 -> 2 -> 3``, not ``1 -> 3 -> 2``, and the area does not fold back: both
    draw the in-place twin, and neither draws what the appended row drew."""
    svg = _assert_twin(
        lambda: _chart(_SIMPLE, mark, y=_y(True)),
        lambda: _chart(_SIMPLE_TWIN, mark),
        plain=lambda: _chart(_SIMPLE, mark),
        shown=[],
    )
    appended, _ = _render_capturing(lambda: _chart(_SIMPLE_APPENDED, mark).to_svg())
    assert svg != appended
    if mark == "mark_line":
        for _, points in polylines(svg):
            xs = [x for x, _ in points]
            assert len(xs) == 3 and xs == sorted(xs)


def test_an_imputed_point_is_drawn_in_its_rows_place():
    svg = _assert_twin(
        lambda: _chart(_SIMPLE, y=_y(True)),
        lambda: _chart(_SIMPLE_TWIN),
        plain=lambda: _chart(_SIMPLE),
        shown=[],
    )
    appended, _ = _render_capturing(lambda: _chart(_SIMPLE_APPENDED).to_svg())
    assert svg != appended
    assert _circles(svg) == _circles(_render_capturing(lambda: _chart(_SIMPLE_TWIN).to_svg())[0])


#: ``(series y's keys, where the imputed row lands)``: every caller row stays where it was, and the
#: imputed row follows the series' last row whose key comes earlier, or leads when none does.
_UNSORTED = [
    pytest.param([3, 1], [3, 1, 2], id="follows-the-last-earlier-row"),
    pytest.param([2, 1], [2, 1, 3], id="follows-the-last-row"),
    pytest.param([3, 2], [1, 3, 2], id="leads-the-series"),
    pytest.param([1, 3, 1], [1, 3, 1, 2], id="a-trajectory-that-revisits-a-key"),
]


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize(("held", "placed"), _UNSORTED)
def test_a_callers_unsorted_series_keeps_every_row_and_takes_each_imputed_row_at_its_place(
    held: list[int], placed: list[int], mark: str
):
    """Series ``x`` is written ``3, 1, 2`` and never moves. Each key series ``y`` lacks is inserted
    by the rule, and the chart is the twin with the rows written down in that order."""
    x_keys = [3, 1, 2]
    missing = sorted(set(x_keys) - set(held))
    held_rows = _simple(
        x_keys + held,
        ["x"] * 3 + ["y"] * len(held),
        [1.0, 2.0, 3.0] + [10.0 + k for k in held],
    )
    written = _simple(
        x_keys + placed,
        ["x"] * 3 + ["y"] * len(placed),
        [1.0, 2.0, 3.0] + [0.0 if k in missing else 10.0 + k for k in placed],
    )
    assert set(placed) == set(held) | set(missing)
    _assert_twin(
        lambda: _chart(held_rows, mark, y=_y(True)),
        lambda: _chart(written, mark),
        plain=lambda: _chart(held_rows, mark),
        shown=[],
    )


def test_several_keys_a_series_lacks_each_land_at_their_own_place():
    """Series ``y`` lacks 2 and 4 of ``1..4``: ``1, 2, 3, 4`` with both inserted, in key order."""
    frame = _simple([1, 2, 3, 4, 1, 3], ["x"] * 4 + ["y"] * 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    twin = _simple(
        [1, 2, 3, 4, 1, 2, 3, 4],
        ["x"] * 4 + ["y"] * 4,
        [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 6.0, 0.0],
    )
    _assert_twin(
        lambda: _chart(frame, "mark_line", y=_y(True)),
        lambda: _chart(twin, "mark_line"),
        plain=lambda: _chart(frame, "mark_line"),
        shown=[],
    )


def _band_order(svg: str) -> list[str]:
    return [run for run in text_runs(svg) if run in ("a", "b", "c")]


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
def test_a_text_key_places_imputed_rows_in_first_appearance_order_and_keeps_the_band_order(
    mark: str,
):
    """Keys appear ``b, a, c``: series ``y`` lacks ``a`` and gets it between its ``b`` and ``c``. The
    band axis lists ``b, a, c``, as it does without the impute."""
    frame = _simple(["b", "a", "c", "b", "c"], _SIMPLE_SERIES, _SIMPLE_VALUES)
    twin = _simple(
        ["b", "a", "c", "b", "a", "c"],
        ["x", "x", "x", "y", "y", "y"],
        [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
    )
    svg = _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)),
        lambda: _chart(twin, mark),
        plain=lambda: _chart(frame, mark),
        shown=[],
    )
    plain_svg, _ = _render_capturing(lambda: _chart(frame, mark).to_svg())
    assert _band_order(svg) == _band_order(plain_svg) == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# A key that no axis places completes no series
# ---------------------------------------------------------------------------

#: ``(maker of the good keys, the bad key, its dtype)``.
_UNPLACED: dict[str, tuple[str, Any, Any]] = {
    "Float64-null": ("Float64", None, pl.Float64),
    "Float64-nan": ("Float64", _NAN, pl.Float64),
    "Float64-inf": ("Float64", _INF, pl.Float64),
    "Float64--inf": ("Float64", -_INF, pl.Float64),
    "Float32-nan": ("Float32", _NAN, pl.Float32),
    "Float32-inf": ("Float32", _INF, pl.Float32),
    "Int64-null": ("Int64", None, pl.Int64),
    "Date-null": ("Date", None, pl.Date),
    "Datetime-null": ("Datetime-us", None, pl.Datetime("us")),
    "Utf8-null": ("Utf8", None, pl.Utf8),
}


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize("name", sorted(_UNPLACED))
def test_a_null_nan_or_infinite_key_completes_no_series(name: str, mark: str):
    """Series ``y`` holds a row whose key no axis places. It adds no key, so no series gains a row
    for it and the one genuine gap (``(2, y)``) is the only row imputed."""
    kind, bad, dtype = _UNPLACED[name]
    make, _ = _KEYS[kind]

    def build(y_keys: Sequence[int], values: Sequence[float]) -> pl.DataFrame:
        good = make([0, 1, 2] + list(y_keys))
        key = pl.concat([good, pl.Series("t", [bad], dtype=dtype)])
        series = ["x"] * 3 + ["y"] * len(y_keys) + ["y"]
        return pl.DataFrame({"t": key, "h": series, "v": [1.0, 2.0, 3.0, *values, 7.0]})

    frame = build([0, 2], [10.0, 12.0])
    twin = build([0, 1, 2], [10.0, 0.0, 12.0])
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)),
        lambda: _chart(twin, mark),
        plain=lambda: _chart(frame, mark),
    )


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize("name", sorted(_UNPLACED))
def test_a_frame_holding_only_unplaced_keys_beside_complete_series_draws_as_it_did(
    name: str, mark: str
):
    """No series lacks a placed key, so the impute changes nothing: the chart is the one drawn
    without it."""
    kind, bad, dtype = _UNPLACED[name]
    make, _ = _KEYS[kind]
    key = pl.concat([make([0, 1, 2, 0, 1, 2]), pl.Series("t", [bad], dtype=dtype)])
    frame = pl.DataFrame(
        {"t": key, "h": ["x"] * 3 + ["y"] * 3 + ["y"], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    )
    imputed, imputed_shown = _render_capturing(lambda: _chart(frame, mark, y=_y(True)).to_svg())
    plain, plain_shown = _render_capturing(lambda: _chart(frame, mark).to_svg())
    assert imputed_shown == plain_shown
    assert imputed == plain


def test_a_key_bound_to_the_colour_field_is_not_a_series_key():
    """``x`` and ``colour`` are one field: one series holds every x, so nothing is imputed."""
    frame = pl.DataFrame({"h": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    imputed, shown = _render_capturing(
        lambda: fm.Chart(frame).mark_bar().encode(x="h", y=_y(True), color="h").to_svg()
    )
    plain, plain_shown = _render_capturing(
        lambda: fm.Chart(frame).mark_bar().encode(x="h", y="v", color="h").to_svg()
    )
    assert shown == plain_shown == []
    assert imputed == plain


# ---------------------------------------------------------------------------
# A key beside the position: the key's own range, a bin
# ---------------------------------------------------------------------------

_BIN = {"bin_width": 1.0, "extent": (0.0, 3.0), "nice": False}


def test_a_binned_x_copies_bin_end_with_its_key_so_every_bar_is_drawn():
    """Series ``y`` lacks the ``[1, 2)`` bin. The imputed row carries that bin's end beside its
    start, and the bar draws as the one the row written into the bin draws."""
    frame = _simple([0.5, 1.5, 2.5, 0.6, 2.6], _SIMPLE_SERIES, _SIMPLE_VALUES)
    twin = _simple(
        [0.5, 1.5, 2.5, 0.6, 1.6, 2.6],
        ["x", "x", "x", "y", "y", "y"],
        [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
    )

    def build(data: pl.DataFrame, imputed: bool) -> fm.Chart:
        y = (
            fm.Y("v", aggregate="sum", impute={"value": 0})
            if imputed
            else fm.Y("v", aggregate="sum")
        )
        return fm.Chart(data).mark_bar().encode(x=fm.X("t", bin=_BIN), y=y, color="h")

    _assert_twin(
        lambda: build(frame, True),
        lambda: build(twin, False),
        plain=lambda: build(frame, False),
        shown=[],
    )


def _ranged(
    keys: Sequence[float], series: Sequence[str], ends: Sequence[float], values: Sequence[float]
) -> pl.DataFrame:
    return pl.DataFrame({"t": keys, "t2": ends, "h": series, "v": values})


def _ranged_bar(frame: pl.DataFrame, *, imputed: bool) -> fm.Chart:
    return fm.Chart(frame).mark_bar().encode(x="t", x2="t2", y=_y(imputed), color="h")


def test_a_ranged_x_whose_end_follows_from_it_copies_the_end_with_the_key():
    frame = _ranged(
        [1.0, 2.0, 3.0, 1.0, 3.0], _SIMPLE_SERIES, [1.5, 2.5, 3.5, 1.5, 3.5], _SIMPLE_VALUES
    )
    twin = _ranged(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        ["x", "x", "x", "y", "y", "y"],
        [1.5, 2.5, 3.5, 1.5, 2.5, 3.5],
        [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
    )
    _assert_twin(
        lambda: _ranged_bar(frame, imputed=True),
        lambda: _ranged_bar(twin, imputed=False),
        plain=lambda: _ranged_bar(frame, imputed=False),
        shown=[],
    )


_X2_DIFFERS = (
    "impute: rows with x 2 hold different x2 values, so an imputed row has no x2; impute an x whose "
    "x2 follows from it"
)


def test_a_ranged_x_whose_end_differs_among_rows_sharing_an_imputed_key_is_refused():
    """Series ``x`` holds ``x = 2`` twice with another ``x2`` each, and series ``y`` lacks ``x = 2``:
    an imputed row would have no one end to copy."""
    frame = _ranged(
        [1.0, 2.0, 2.0, 3.0, 1.0, 3.0],
        ["x", "x", "x", "x", "y", "y"],
        [1.5, 2.5, 2.7, 3.5, 1.5, 3.5],
        [1.0, 2.0, 2.5, 3.0, 4.0, 5.0],
    )
    with pytest.raises(ValueError) as excinfo:
        _ranged_bar(frame, imputed=True).to_svg()
    assert str(excinfo.value) == _X2_DIFFERS


def test_a_ranged_x_whose_end_differs_on_a_key_no_series_lacks_draws_the_twin():
    """Both series hold ``x = 1`` with another ``x2``; only ``x = 2`` is imputed and it has one end,
    so the disagreement is nothing an imputed row copies."""
    frame = _ranged(
        [1.0, 2.0, 3.0, 1.0, 3.0], _SIMPLE_SERIES, [1.5, 2.5, 3.5, 1.7, 3.5], _SIMPLE_VALUES
    )
    twin = _ranged(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        ["x", "x", "x", "y", "y", "y"],
        [1.5, 2.5, 3.5, 1.7, 2.5, 3.5],
        [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
    )
    _assert_twin(
        lambda: _ranged_bar(frame, imputed=True),
        lambda: _ranged_bar(twin, imputed=False),
        plain=lambda: _ranged_bar(frame, imputed=False),
        shown=[],
    )


def test_a_complete_ranged_chart_whose_ends_differ_is_not_refused_for_an_impute_it_never_needed():
    """No series lacks a key, so no row is imputed and no end needs copying: the chart that drew
    before the impute was written draws still."""
    frame = _ranged(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        ["x", "x", "x", "y", "y", "y"],
        [1.5, 2.5, 3.5, 1.7, 2.5, 3.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    imputed, shown = _render_capturing(lambda: _ranged_bar(frame, imputed=True).to_svg())
    plain, plain_shown = _render_capturing(lambda: _ranged_bar(frame, imputed=False).to_svg())
    assert shown == plain_shown
    assert imputed == plain


# ---------------------------------------------------------------------------
# What keys a series: the colour the mark splits by, and the channels beside it
# ---------------------------------------------------------------------------


def _coloured(first: Any, second: Any, dtype: Any = None) -> pl.DataFrame:
    """Series ``first`` over keys 1 to 3, series ``second`` over keys 1 and 3."""
    keys = [first] * 3 + [second] * 2
    colours = pl.Series("h", keys, dtype=dtype) if dtype is not None else pl.Series("h", keys)
    return pl.DataFrame({"t": [1, 2, 3, 1, 3], "h": colours, "v": _SIMPLE_VALUES})


def _coloured_twin(first: Any, second: Any, dtype: Any = None) -> pl.DataFrame:
    keys = [first] * 3 + [second] * 3
    colours = pl.Series("h", keys, dtype=dtype) if dtype is not None else pl.Series("h", keys)
    return pl.DataFrame(
        {"t": [1, 2, 3, 1, 2, 3], "h": colours, "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0]}
    )


#: ``(first, second, dtype, colour channel)``: a colour the mark reads as categories.
_CATEGORICAL_COLOURS = {
    "Boolean": (True, False, pl.Boolean, "h"),
    "Int64-nominal": (1, 2, pl.Int64, fm.Color("h", type_="N")),
    "Int64-ordinal": (1, 2, pl.Int64, fm.Color("h", type_="O")),
    "UInt8-nominal": (1, 2, pl.UInt8, fm.Color("h", type_="N")),
    "Float64-nominal": (1.5, 2.5, pl.Float64, fm.Color("h", type_="N")),
    "Float64-ordinal": (1.5, 2.5, pl.Float64, fm.Color("h", type_="O")),
}


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize("name", sorted(_CATEGORICAL_COLOURS))
def test_a_colour_the_mark_reads_as_categories_keys_its_series(name: str, mark: str):
    first, second, dtype, channel = _CATEGORICAL_COLOURS[name]
    frame, twin = _coloured(first, second, dtype), _coloured_twin(first, second, dtype)
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True), color=channel),
        lambda: _chart(twin, mark, color=channel),
        plain=lambda: _chart(frame, mark, color=channel),
        shown=[],
    )


def test_an_untyped_numeric_colour_under_an_ordinal_scale_is_drawn_continuous_and_keys_no_series():
    """The colour is drawn as one continuous colour whatever the ordinal scale beside it says, so it
    splits nothing: the imputed chart is the chart without ``impute=`` (no row gains a mark), and
    differs from the twin that writes the per-``h`` row out. ``type_="N"`` on the same frame is the
    categorical contrast (``_CATEGORICAL_COLOURS``): there each series is completed."""
    frame, twin = _coloured(1, 2, pl.Int64), _coloured_twin(1, 2, pl.Int64)
    channel = fm.Color("h", scale=fm.OrdinalScale(domain=["1", "2"]))
    svg, shown = _render_capturing(lambda: _chart(frame, y=_y(True), color=channel).to_svg())
    plain, _ = _render_capturing(lambda: _chart(frame, color=channel).to_svg())
    written, _ = _render_capturing(lambda: _chart(twin, color=channel).to_svg())
    assert shown == []
    assert svg == plain
    assert svg != written


_CONTINUOUS_COLOURS = {
    "Int64": (1, 2, pl.Int64),
    "UInt8": (1, 2, pl.UInt8),
    "Float64": (1.5, 2.5, pl.Float64),
    "Date": (dt.date(2024, 1, 1), dt.date(2024, 1, 2), pl.Date),
}
_NOT_PER_MARK = (
    "continuous color scale is not supported on line; the channel has no per-mark effect and its "
    "legend was suppressed"
)


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize("name", sorted(_CONTINUOUS_COLOURS))
def test_an_untyped_numeric_or_date_colour_keys_nothing_and_the_chart_draws_as_it_did(
    name: str, mark: str
):
    """A continuous colour is not a partition (a line draws one series): the series that holds every
    key holds all of them, so nothing is imputed, and the line keeps its notice."""
    first, second, dtype = _CONTINUOUS_COLOURS[name]
    frame = _coloured(first, second, dtype)
    imputed, shown = _render_capturing(lambda: _chart(frame, mark, y=_y(True)).to_svg())
    plain, plain_shown = _render_capturing(lambda: _chart(frame, mark).to_svg())
    assert shown == plain_shown == ([_NOT_PER_MARK] if mark == "mark_line" else [])
    assert imputed == plain


_BIN_BY_FIVE = {"bin_width": 5.0, "extent": (0.0, 10.0), "nice": False}


@pytest.mark.parametrize("mark", ["mark_line", "mark_point"])
def test_a_binned_colour_keys_a_series_by_bin_and_an_imputed_row_copies_a_value_from_its_bin(
    mark: str,
):
    """Series by bin: raw values ``1.0, 1.2, 1.1`` share one, ``8.0, 8.5`` the other. The second
    gains its ``(2, .)`` row copying ``8.0`` from its own bin; a line draws ``[3, 3]`` vertices where
    it drew ``[3, 2]``."""
    frame = pl.DataFrame(
        {"t": [1, 2, 3, 1, 3], "h": [1.0, 1.2, 1.1, 8.0, 8.5], "v": _SIMPLE_VALUES}
    )
    twin = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3],
            "h": [1.0, 1.2, 1.1, 8.0, 8.0, 8.5],
            "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
        }
    )
    channel = fm.Color("h", bin=_BIN_BY_FIVE)
    svg = _assert_twin(
        lambda: _chart(frame, mark, y=_y(True), color=channel),
        lambda: _chart(twin, mark, color=channel),
        plain=lambda: _chart(frame, mark, color=channel),
        shown=[],
    )
    if mark == "mark_line":
        assert _vertices(svg) == [3, 3]


def _by_second_key(rows: Sequence[tuple[int, str]], values: Sequence[float]) -> pl.DataFrame:
    return pl.DataFrame({"t": [r[0] for r in rows], "h": [r[1] for r in rows], "v": list(values)})


_P_ROWS = [(1, "p"), (2, "p"), (3, "p"), (1, "q"), (3, "q")]
_P_TWIN_ROWS = [(1, "p"), (2, "p"), (3, "p"), (1, "q"), (2, "q"), (3, "q")]

#: ``(mark, the channel that splits it, the series column)``: no colour is bound, so the series come
#: from this channel alone.
_SPLIT_WITHOUT_COLOUR = [
    pytest.param("mark_line", "detail", id="line-detail"),
    pytest.param("mark_line", "stroke_dash", id="line-stroke_dash"),
    pytest.param("mark_area", "detail", id="area-detail"),
    pytest.param("mark_point", "shape", id="point-shape"),
]


@pytest.mark.parametrize(("mark", "channel"), _SPLIT_WITHOUT_COLOUR)
def test_with_no_colour_a_detail_shape_or_dash_keys_the_series(mark: str, channel: str):
    frame = _by_second_key(_P_ROWS, _SIMPLE_VALUES)
    twin = _by_second_key(_P_TWIN_ROWS, [1.0, 2.0, 3.0, 4.0, 0.0, 5.0])
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True), color=None, **{channel: "h"}),
        lambda: _chart(twin, mark, color=None, **{channel: "h"}),
        plain=lambda: _chart(frame, mark, color=None, **{channel: "h"}),
        shown=[],
        interactive=True,
    )


@pytest.mark.parametrize("mark", ["mark_line", "mark_area"])
def test_a_date_detail_beside_the_impute_keys_the_series_by_date(mark: str):
    first, second = dt.date(2024, 5, 1), dt.date(2024, 5, 2)
    frame = pl.DataFrame(
        {"t": [1, 2, 3, 1, 3], "h": [first] * 3 + [second] * 2, "v": _SIMPLE_VALUES}
    )
    twin = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3],
            "h": [first] * 3 + [second] * 3,
            "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
        }
    )
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True), color=None, detail="h"),
        lambda: _chart(twin, mark, color=None, detail="h"),
        plain=lambda: _chart(frame, mark, color=None, detail="h"),
        shown=[],
    )


@pytest.mark.parametrize("mark", ["mark_line", "mark_point"])
def test_a_second_key_beside_a_boolean_colour_keys_the_series_by_both(mark: str):
    """Series ``(h, d)``: colour ``h`` is a ``Boolean``, the second key a shape or a detail."""
    channel = {"mark_line": "detail", "mark_point": "shape"}[mark]
    rows = [(1, "p"), (2, "p"), (3, "p"), (1, "q"), (3, "q")] * 2
    flags = [True] * 5 + [False] * 5
    frame = pl.DataFrame(
        {
            "t": [r[0] for r in rows],
            "h": flags,
            "d": [r[1] for r in rows],
            "v": [float(i) for i in range(1, 11)],
        }
    )
    full = [(1, "p"), (2, "p"), (3, "p"), (1, "q"), (2, "q"), (3, "q")] * 2
    twin = pl.DataFrame(
        {
            "t": [r[0] for r in full],
            "h": [True] * 6 + [False] * 6,
            "d": [r[1] for r in full],
            "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 10.0],
        }
    )
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True), **{channel: "d"}),
        lambda: _chart(twin, mark, **{channel: "d"}),
        plain=lambda: _chart(frame, mark, **{channel: "d"}),
        shown=[],
    )


def test_a_key_that_is_the_value_field_is_not_a_series_key():
    """Colour ``v`` is the value: it cannot name a series, so one series holds every key."""
    imputed, shown = _render_capturing(
        lambda: _chart(_SIMPLE, y=_y(True), color=fm.Color("v", type_="N")).to_svg()
    )
    plain, plain_shown = _render_capturing(
        lambda: _chart(_SIMPLE, color=fm.Color("v", type_="N")).to_svg()
    )
    assert shown == plain_shown
    assert imputed == plain


# ---------------------------------------------------------------------------
# The value: its dtype, and a fill of its kind
# ---------------------------------------------------------------------------


def _value_frames(dtype: Any, filled: Any = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The caller's frame and its twin with ``filled`` written at ``(2, y)``, both over ``dtype``."""
    frame = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [1, 2, 3, 4, 5]).with_columns(
        pl.col("v").cast(dtype)
    )
    twin = _simple(
        [1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [1, 2, 3, 4, filled, 5]
    ).with_columns(pl.col("v").cast(dtype))
    return frame, twin


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
@pytest.mark.parametrize(
    "dtype",
    [pl.Float32, pl.Int8, pl.Int32, pl.Int64, pl.UInt8, pl.UInt32, pl.UInt64],
    ids=lambda d: str(d),
)
def test_a_value_of_any_numeric_dtype_is_completed_as_its_written_out_twin(dtype: Any, mark: str):
    frame, twin = _value_frames(dtype)
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)),
        lambda: _chart(twin, mark),
        plain=lambda: _chart(frame, mark),
        shown=[],
    )


@pytest.mark.parametrize(
    ("dtype", "fill"), [(pl.Int64, 2.5), (pl.UInt8, -1)], ids=["Int64-2.5", "UInt8--1"]
)
def test_a_fill_an_integer_column_cannot_hold_is_drawn_at_the_value_it_was_given(
    dtype: Any, fill: float
):
    """The value column is read as ``Float64``, so ``2.5`` and ``-1`` are drawn where they say."""
    frame, _ = _value_frames(dtype)
    twin = _simple(
        [1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [1.0, 2.0, 3.0, 4.0, float(fill), 5.0]
    )
    _assert_twin(
        lambda: _chart(frame, y=_y(True, value=fill)),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )


_DAY = dt.date(2024, 1, 10)
_DAY_MS = int(dt.datetime(2024, 1, 10, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _when(maker: Callable[[int], Any], day: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    first = [maker(i) for i in range(1, 6)]
    frame = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [0.0] * 5).with_columns(v=pl.Series("v", first))
    twin = _simple([1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [0.0] * 6).with_columns(
        v=pl.Series("v", [*first[:4], day, first[4]])
    )
    return frame, twin


#: ``(a day of the column's dtype from its index, the written fill, the day it fills)``.
_TEMPORAL_FILLS = {
    "Date-number": (lambda i: dt.date(2024, 1, i), _DAY_MS, _DAY),
    "Date-date": (lambda i: dt.date(2024, 1, i), _DAY, _DAY),
    "Date-datetime": (lambda i: dt.date(2024, 1, i), dt.datetime(2024, 1, 10), _DAY),
    "Datetime-number": (lambda i: dt.datetime(2024, 1, i), _DAY_MS, dt.datetime(2024, 1, 10)),
    "Datetime-date": (lambda i: dt.datetime(2024, 1, i), _DAY, dt.datetime(2024, 1, 10)),
    "Datetime-datetime": (
        lambda i: dt.datetime(2024, 1, i),
        dt.datetime(2024, 1, 10),
        dt.datetime(2024, 1, 10),
    ),
}


@pytest.mark.parametrize("name", sorted(_TEMPORAL_FILLS))
def test_a_temporal_value_is_filled_with_a_number_a_date_or_a_datetime_at_its_own_dtype(name: str):
    maker, fill, day = _TEMPORAL_FILLS[name]
    frame, twin = _when(maker, day)
    _assert_twin(
        lambda: _chart(frame, y=_y(True, value=fill)),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )


def test_a_text_value_is_filled_with_a_string():
    words = ["lo", "mid", "hi", "lo", "hi"]
    frame = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [0.0] * 5).with_columns(v=pl.Series("v", words))
    twin = _simple([1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [0.0] * 6).with_columns(
        v=pl.Series("v", ["lo", "mid", "hi", "lo", "none", "hi"])
    )
    _assert_twin(
        lambda: _chart(frame, y=_y(True, value="none")),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )


@pytest.mark.parametrize("fill", [True, False])
def test_a_boolean_value_is_filled_with_a_bool(fill: bool):
    flags = [True, False, True, False, True]
    frame = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [0.0] * 5).with_columns(v=pl.Series("v", flags))
    twin = _simple([1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [0.0] * 6).with_columns(
        v=pl.Series("v", [True, False, True, False, fill, True])
    )
    _assert_twin(
        lambda: _chart(frame, y=_y(True, value=fill)),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )


_WRONG_KIND_COLUMNS = {
    "number": [1.0, 2.0, 3.0, 4.0, 5.0],
    "date": [dt.date(2024, 1, i) for i in range(1, 6)],
    "text": ["a", "b", "c", "d", "e"],
    "boolean": [True, False, True, False, True],
}


@pytest.mark.parametrize("case", list(IMPUTE_KIND_CASES))
def test_a_fill_of_a_kind_the_column_cannot_hold_is_refused_naming_both(case: str):
    column = _WRONG_KIND_COLUMNS[case.split("-")[0]]
    fill = IMPUTE_KIND_CASES[case][0]
    frame = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [0.0] * 5).with_columns(v=pl.Series("v", column))
    with pytest.raises(ValueError) as excinfo:
        _chart(frame, y=_y(True, value=fill)).to_svg()
    assert str(excinfo.value) == _impute_kind_refusal("impute", "v", case)


# ---------------------------------------------------------------------------
# The options of the dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{}, {"method": "value"}, {"method": "value", "value": 0}],
    ids=["empty", "method-value", "method-value-and-value"],
)
def test_an_impute_with_no_value_fills_zero(options: dict[str, Any]):
    _assert_twin(
        lambda: _chart(_SIMPLE, y=fm.Y("v", impute=options)),
        lambda: _chart(_SIMPLE_TWIN),
        plain=lambda: _chart(_SIMPLE),
        shown=[],
    )


#: Series ``y`` lacks key 2 and holds ``4, 6, 11`` at keys 1, 3, 4: mean 7, median 6, min 4, max 11.
_STAT_FRAME = _simple(
    [1, 2, 3, 4, 1, 3, 4], ["x"] * 4 + ["y"] * 3, [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 11.0]
)


def _stat_twin(filled: float) -> pl.DataFrame:
    return _simple(
        [1, 2, 3, 4, 1, 2, 3, 4],
        ["x"] * 4 + ["y"] * 4,
        [1.0, 2.0, 3.0, 5.0, 4.0, filled, 6.0, 11.0],
    )


@pytest.mark.parametrize(
    ("method", "filled"),
    [("mean", 7.0), ("median", 6.0), ("min", 4.0), ("max", 11.0)],
)
@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
def test_a_statistic_method_fills_each_imputed_row_with_its_series_statistic(
    method: str, filled: float, mark: str
):
    _assert_twin(
        lambda: _chart(_STAT_FRAME, mark, y=fm.Y("v", impute={"method": method})),
        lambda: _chart(_stat_twin(filled), mark),
        plain=lambda: _chart(_STAT_FRAME, mark),
        shown=[],
    )


def test_a_series_with_no_usable_value_takes_the_whole_columns_statistic_with_a_notice():
    """Series ``y`` holds two null cells and lacks key 2; the whole column's usable values are series
    ``x``'s ``1, 2, 3``, whose mean is 2. The notice names the series by its key."""
    frame = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [1.0, 2.0, 3.0, None, None]).with_columns(
        pl.col("v").cast(pl.Float64)
    )
    twin = _simple(
        [1, 2, 3, 1, 2, 3], ["x", "x", "x", "y", "y", "y"], [1.0, 2.0, 3.0, 2.0, 2.0, 2.0]
    )
    _assert_twin(
        lambda: _chart(frame, y=fm.Y("v", impute={"method": "mean"})),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[
            "Impute: group h=y has no usable (non-null, finite) values; the whole column's "
            "statistic was used instead"
        ],
        twin_shown=[],
    )


@pytest.mark.parametrize("fill", [None, _NAN], ids=["null", "nan"])
@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
def test_a_callers_own_null_or_nan_value_is_filled_like_a_missing_row(fill: Any, mark: str):
    """Series ``y`` holds ``4, <bad>, 6`` and lacks key 4 of ``1..4``: both the cell and the row are
    filled with 0, and nothing is counted as a gap."""
    frame = _simple(
        [1, 2, 3, 4, 1, 2, 3], ["x"] * 4 + ["y"] * 3, [1.0, 2.0, 3.0, 5.0, 4.0, fill, 6.0]
    ).with_columns(pl.col("v").cast(pl.Float64))
    twin = _simple(
        [1, 2, 3, 4, 1, 2, 3, 4],
        ["x"] * 4 + ["y"] * 4,
        [1.0, 2.0, 3.0, 5.0, 4.0, 0.0, 6.0, 0.0],
    )
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)),
        lambda: _chart(twin, mark),
        plain=lambda: _chart(frame, mark),
        shown=[],
    )


_STATISTICS = [("mean", 5.0), ("median", 5.0), ("min", 4.0), ("max", 6.0)]


@pytest.mark.parametrize("fill", [None, _NAN], ids=["null", "nan"])
@pytest.mark.parametrize(("method", "filled"), _STATISTICS)
def test_a_callers_own_null_or_nan_value_takes_its_series_statistic_under_a_statistic_method(
    fill: Any, method: str, filled: float
):
    """Series ``y`` holds ``4, <bad>, 6``: the cell takes the statistic of the usable ``4`` and ``6``."""
    frame = _simple(
        [1, 2, 3, 1, 2, 3], ["x"] * 3 + ["y"] * 3, [1.0, 2.0, 3.0, 4.0, fill, 6.0]
    ).with_columns(pl.col("v").cast(pl.Float64))
    twin = _simple([1, 2, 3, 1, 2, 3], ["x"] * 3 + ["y"] * 3, [1.0, 2.0, 3.0, 4.0, filled, 6.0])
    _assert_twin(
        lambda: _chart(frame, y=fm.Y("v", impute={"method": method})),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )


@pytest.mark.parametrize(("method", "filled"), _STATISTICS)
def test_an_infinite_value_is_kept_and_left_out_of_a_statistic(method: str, filled: float):
    """Series ``y`` holds ``4, inf, 6`` and lacks key 2: the imputed row takes the statistic of the
    finite ``4`` and ``6``, and the infinite cell stays where it is."""
    frame = _simple(
        [1, 2, 3, 4, 1, 3, 4], ["x"] * 4 + ["y"] * 3, [1.0, 2.0, 3.0, 5.0, 4.0, _INF, 6.0]
    )
    twin = _simple(
        [1, 2, 3, 4, 1, 2, 3, 4],
        ["x"] * 4 + ["y"] * 4,
        [1.0, 2.0, 3.0, 5.0, 4.0, filled, _INF, 6.0],
    )
    _assert_twin(
        lambda: _chart(frame, y=fm.Y("v", impute={"method": method})),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
    )


@pytest.mark.parametrize("mark", ["mark_point", "mark_line"])
def test_an_infinite_value_is_kept_and_not_filled(mark: str):
    """An infinite value is a value, not a gap: the chart is the one drawn without the impute."""
    frame = _simple([1, 2, 3, 1, 2, 3], ["x"] * 3 + ["y"] * 3, [1.0, 2.0, 3.0, 4.0, _INF, 6.0])
    imputed, shown = _render_capturing(lambda: _chart(frame, mark, y=_y(True)).to_svg())
    plain, plain_shown = _render_capturing(lambda: _chart(frame, mark).to_svg())
    assert shown == plain_shown
    assert imputed == plain


def test_keyvals_are_added_to_the_keys_every_series_holds():
    """``keyvals`` 1 to 4 add key 4 to both series beside the keys the frame holds, and series ``y``
    still gains the key it lacks."""
    twin = _simple(
        [1, 2, 3, 4, 1, 2, 3, 4], ["x"] * 4 + ["y"] * 4, [1.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]
    )
    _assert_twin(
        lambda: _chart(_SIMPLE, y=_y(True, value=0, keyvals=[1, 2, 3, 4])),
        lambda: _chart(twin),
        plain=lambda: _chart(_SIMPLE),
        shown=[],
    )


def test_a_keyvals_sequence_adds_its_keys_to_every_series():
    twin = _simple(
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        ["x"] * 5 + ["y"] * 5,
        [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0],
    )
    _assert_twin(
        lambda: _chart(_SIMPLE, y=_y(True, value=0, keyvals={"start": 0, "stop": 5})),
        lambda: _chart(twin),
        plain=lambda: _chart(_SIMPLE),
        shown=[],
    )


def test_a_frame_gives_each_fill_the_mean_of_the_usable_values_one_row_either_side():
    """Series ``y`` holds ``4, 10, 20`` at keys 1, 3, 4: key 2 reads keys 1 and 3, so ``(4 + 10) / 2``."""
    frame = _simple(
        [1, 2, 3, 4, 1, 3, 4], ["x"] * 4 + ["y"] * 3, [1.0, 2.0, 3.0, 5.0, 4.0, 10.0, 20.0]
    )
    twin = _simple(
        [1, 2, 3, 4, 1, 2, 3, 4],
        ["x"] * 4 + ["y"] * 4,
        [1.0, 2.0, 3.0, 5.0, 4.0, 7.0, 10.0, 20.0],
    )
    _assert_twin(
        lambda: _chart(frame, y=fm.Y("v", impute={"method": "mean", "frame": [-1, 1]})),
        lambda: _chart(twin),
        plain=lambda: _chart(frame),
        shown=[],
    )


def _refusal(chart: fm.Chart) -> str:
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    return str(excinfo.value)


def test_an_unknown_method_is_refused():
    assert _refusal(_chart(_SIMPLE, y=fm.Y("v", impute={"method": "bogus"}))) == (
        "impute: unknown method 'bogus'; expected value|mean|median|min|max"
    )


def test_a_value_beside_a_statistic_method_is_refused():
    chart = _chart(_SIMPLE, y=fm.Y("v", impute={"method": "mean", "value": 0}))
    assert _refusal(chart) == (
        "impute: value= applies to method='value', not 'mean'; drop value= or set method='value'"
    )


@pytest.mark.parametrize(
    "options",
    [{"value": 0, "frame": [-1, 1]}, {"method": "value", "frame": [-1, 1]}, {"frame": [None, 1]}],
    ids=["value", "method-value", "default-method"],
)
def test_a_frame_beside_a_constant_fill_is_refused(options: dict[str, Any]):
    assert _refusal(_chart(_SIMPLE, y=fm.Y("v", impute=options))) == (
        "impute: frame= applies to mean, median, min and max; method 'value' fills a constant"
    )


@pytest.mark.parametrize(
    ("key", "holds"),
    [(["a", "b", "c", "a", "c"], "text"), ([True, False, True, True, False], "booleans")],
    ids=["text", "boolean"],
)
def test_a_keyvals_sequence_on_a_key_that_is_not_a_number_is_refused(key: list[Any], holds: str):
    frame = _simple(key, _SIMPLE_SERIES, _SIMPLE_VALUES)
    chart = _chart(frame, y=_y(True, value=0, keyvals={"start": 0, "stop": 5}))
    assert _refusal(chart) == (
        f"impute: keyvals={{'start': ...}} makes numbers, but the key 't' holds {holds}; list the "
        "keys instead"
    )


def test_a_keyvals_sequence_with_a_zero_step_is_refused():
    chart = _chart(_SIMPLE, y=_y(True, value=0, keyvals={"start": 0, "stop": 5, "step": 0}))
    assert _refusal(chart) == (
        "impute: keyvals={'start': ..., 'step': 0} never reaches stop; use a non-zero step"
    )


# ---------------------------------------------------------------------------
# Horizontal doors: the authored value channel along the authored key
# ---------------------------------------------------------------------------


def _horizontal(frame: pl.DataFrame, mark: str, *, imputed: bool) -> fm.Chart:
    """``y`` is the key and ``x`` the value, which carries the impute."""
    x = fm.X("v", impute={"value": 0}) if imputed else "v"
    return getattr(fm.Chart(frame), mark)().encode(y="t", x=x, color="h")


@pytest.mark.parametrize("mark", ["mark_point", "mark_line", "mark_bar"])
def test_an_impute_on_x_completes_each_series_along_y(mark: str):
    _assert_twin(
        lambda: _horizontal(_SIMPLE, mark, imputed=True),
        lambda: _horizontal(_SIMPLE_TWIN, mark, imputed=False),
        plain=lambda: _horizontal(_SIMPLE, mark, imputed=False),
        shown=[],
        interactive=True,
    )


@pytest.mark.parametrize("kind", ["Date", "Utf8", "Boolean"])
def test_an_impute_on_x_completes_each_series_along_a_y_of_any_dtype(kind: str):
    make, count = _KEYS[kind]
    frame, twin = _grid(make, count)

    _assert_twin(
        lambda: _horizontal(frame, "mark_point", imputed=True),
        lambda: _horizontal(twin, "mark_point", imputed=False),
        plain=lambda: _horizontal(frame, "mark_point", imputed=False),
        shown=[],
    )


@pytest.mark.parametrize("mark", ["mark_point", "mark_line", "mark_bar"])
@pytest.mark.parametrize("kind", ["Int64", "Date", "Utf8"])
def test_a_flipped_coord_completes_the_authored_value_along_the_authored_key(kind: str, mark: str):
    """Under ``CoordFlip`` the impute on the written ``y`` is still keyed by the written ``x``."""
    make, count = _KEYS[kind]
    frame, twin = _grid(make, count)
    flip = fm.CoordFlip()
    _assert_twin(
        lambda: _chart(frame, mark, y=_y(True)).coord(flip),
        lambda: _chart(twin, mark).coord(flip),
        plain=lambda: _chart(frame, mark).coord(flip),
        shown=[],
        interactive=True,
    )


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
def test_a_flipped_coord_completes_each_facet_panel(door: str):
    facet = _FACET_DOORS[door]
    flip = fm.CoordFlip()
    _assert_twin(
        lambda: facet(_chart(_IMPUTE_FRAME, y=_y(True)).coord(flip)),
        lambda: facet(_chart(_IMPUTE_COMPLETED).coord(flip)),
        plain=lambda: facet(_chart(_IMPUTE_FRAME).coord(flip)),
        shown=[],
        interactive=True,
    )


def test_an_impute_on_x_under_a_flipped_coord_completes_along_y_as_written():
    flip = fm.CoordFlip()
    _assert_twin(
        lambda: _horizontal(_SIMPLE, "mark_point", imputed=True).coord(flip),
        lambda: _horizontal(_SIMPLE_TWIN, "mark_point", imputed=False).coord(flip),
        plain=lambda: _horizontal(_SIMPLE, "mark_point", imputed=False).coord(flip),
        shown=[],
    )


# ---------------------------------------------------------------------------
# An impute that cannot be honoured is refused with the remedy named
# ---------------------------------------------------------------------------


def test_an_impute_written_on_both_x_and_y_is_refused():
    chart = (
        fm.Chart(_SIMPLE)
        .mark_point()
        .encode(x=fm.X("t", impute={"value": 0}), y=_y(True), color="h")
    )
    assert _refusal(chart) == (
        "impute: impute= is written on both x and y; a series is completed along one axis, so "
        "write it on the value channel only"
    )


def test_a_y_impute_with_no_x_is_refused():
    chart = fm.Chart(_SIMPLE).mark_tick().encode(y=_y(True))
    assert _refusal(chart) == (
        "impute: impute= on y completes each series along x, but the layer draws no x; bind x or "
        "remove impute="
    )


def test_an_x_impute_with_no_y_is_refused():
    chart = fm.Chart(_SIMPLE).mark_tick().encode(x=fm.X("v", impute={"value": 0}))
    assert _refusal(chart) == (
        "impute: impute= on x completes each series along y, but the layer draws no y; bind y or "
        "remove impute="
    )


_WITH_END = _SIMPLE.with_columns(v2=pl.col("v") + 1.0)


def test_a_y_impute_on_a_layer_that_draws_y2_is_refused():
    chart = fm.Chart(_WITH_END).mark_bar().encode(x="t", y=_y(True), y2="v2", color="h")
    assert _refusal(chart) == (
        "impute: the layer also draws y2 'v2', which an imputed row would leave empty; remove y2 or "
        "write the missing rows"
    )


def test_an_x_impute_on_a_layer_that_draws_x2_is_refused():
    chart = (
        fm.Chart(_WITH_END)
        .mark_bar()
        .encode(y="t", x=fm.X("v", impute={"value": 0}), x2="v2", color="h")
    )
    assert _refusal(chart) == (
        "impute: the layer also draws x2 'v2', which an imputed row would leave empty; remove x2 or "
        "write the missing rows"
    )


_UNPLACEABLE = {
    "Decimal": [decimal.Decimal(k) for k in (1, 2, 3, 1, 3)],
    "Time64": [dt.time(k) for k in (1, 2, 3, 1, 3)],
    "list": [[k] for k in (1, 2, 3, 1, 3)],
}


@pytest.mark.parametrize("kind", sorted(_UNPLACEABLE))
def test_an_x_no_axis_can_place_is_refused_as_it_was_without_the_impute(kind: str):
    """The render's own dtype refusal speaks, in the words it uses with no impute written."""
    frame = _simple(_UNPLACEABLE[kind], _SIMPLE_SERIES, _SIMPLE_VALUES)
    imputed = _refusal(_chart(frame, y=_y(True)))
    assert imputed == _refusal(_chart(frame))
    assert "unsupported dtype" in imputed


# ---------------------------------------------------------------------------
# Each layer completes its own series
# ---------------------------------------------------------------------------

#: Two series over three keys, series ``y`` missing key 2, with a column for a secondary axis.
_LAYERED = _SIMPLE.with_columns(w=pl.col("v") * 2.0)
_LAYERED_TWIN = _SIMPLE_TWIN.with_columns(w=pl.col("v") * 2.0)


def _point_layer(frame: Any, *, imputed: bool) -> fm.Chart:
    return _chart(frame, "mark_point", y=_y(imputed))


def _line_layer(frame: Any) -> fm.Chart:
    return _chart(frame, "mark_line")


def _inherited(frame: Any, *, imputed: bool) -> fm.Chart:
    return _point_layer(frame, imputed=imputed).layer(fm.Layer(mark="line"))


def _own_y_layer(frame: Any, *, imputed: bool, twin: Any = None) -> fm.Chart:
    """A line chart whose point layer writes its own ``y``, with the impute on it alone. The twin
    draws the written-out rows through a layer that carries them as its own data."""
    base = _chart(frame, "mark_line", y="v")
    if imputed:
        layer = fm.Layer(mark="point", encoding={"y": fm.Y("v", impute={"value": 0})})
        return base.layer(layer)
    return base.layer(
        fm.Layer(mark="point", data=twin, encoding={"x": "t", "y": "v", "color": "h"})
    )


def _host_with_plain_layer(frame: Any, *, imputed: bool, twin: Any = None) -> fm.Chart:
    """A host that writes the impute beside a line layer that writes its own ``y`` without one."""
    host = _chart(
        twin if (twin is not None and not imputed) else frame, "mark_point", y=_y(imputed)
    )
    layer = fm.Layer(mark="line", encoding={"y": "v"})
    if imputed:
        return host.layer(layer)
    return host.layer(
        fm.Layer(mark="line", data=frame, encoding={"x": "t", "y": "v", "color": "h"})
    )


#: ``(builder(imputing rows, sibling rows, imputed), the sibling reads the imputed rows, byte twin)``.
#: A door whose twin cannot be the same structure is read through the marks each layer drew.
_LAYER_DOORS: dict[str, tuple[Callable[[Any, Any, bool], Any], bool, bool]] = {
    "plus-point-first": (
        lambda a, b, imputed: _point_layer(a, imputed=imputed) + _line_layer(b),
        False,
        True,
    ),
    "plus-line-first": (
        lambda a, b, imputed: _line_layer(b) + _point_layer(a, imputed=imputed),
        False,
        True,
    ),
    "layer-chart-shared": (
        lambda a, b, imputed: fm.LayerChart(_point_layer(a, imputed=imputed), _line_layer(b)),
        False,
        True,
    ),
    "layer-chart-independent-y": (
        lambda a, b, imputed: fm.LayerChart(
            _point_layer(a, imputed=imputed), _line_layer(b), resolve={"y": "independent"}
        ),
        False,
        True,
    ),
    "chart-layer-inherited": (
        lambda a, b, imputed: _inherited(a, imputed=imputed),
        True,
        True,
    ),
    "layer-own-y": (
        lambda a, b, imputed: _own_y_layer(b, imputed=imputed, twin=a),
        False,
        False,
    ),
    "host-and-a-layer-with-no-impute": (
        lambda a, b, imputed: _host_with_plain_layer(b, imputed=imputed, twin=a),
        False,
        False,
    ),
    "secondary-y": (
        lambda a, b, imputed: (
            _point_layer(a, imputed=imputed) + fm.SecondaryY("w", mark="line", color="#d62728")
        ),
        False,
        False,
    ),
}


def _render_door(door: str, *, imputed: bool, written: bool = False) -> tuple[str, list[str]]:
    """The door's chart with the impute, or over the frames written out (``written``), or plain."""
    build, sibling_reads, _ = _LAYER_DOORS[door]
    imputing = _LAYERED_TWIN if written else _LAYERED
    sibling = _LAYERED_TWIN if (written and sibling_reads) else _LAYERED
    return _render_capturing(lambda: build(imputing, sibling, imputed).to_svg())


@pytest.mark.parametrize("door", sorted(_LAYER_DOORS))
def test_an_impute_on_one_layer_is_read_by_that_layer_alone(door: str):
    """The layer that writes (or, for ``chart-layer-inherited``, shares) the impute draws the
    imputed point and nothing else gains a mark: a sibling draws the caller's rows, here the line's
    ``[3, 2]`` vertices."""
    _, sibling_reads, _ = _LAYER_DOORS[door]
    svg, shown = _render_door(door, imputed=True)
    plain, plain_shown = _render_door(door, imputed=False)
    assert shown == plain_shown == []
    assert len(_circles(svg)) == len(_circles(plain)) + 1
    plain_vertices = _vertices(plain)
    assert sum(_vertices(svg)) == sum(plain_vertices) + (1 if sibling_reads else 0)
    if not sibling_reads:
        assert _vertices(svg) == plain_vertices


@pytest.mark.parametrize("door", [d for d in sorted(_LAYER_DOORS) if _LAYER_DOORS[d][2]])
def test_a_layer_door_draws_the_chart_with_the_missing_rows_written_out(door: str):
    svg, shown = _render_door(door, imputed=True)
    twin, twin_shown = _render_door(door, imputed=False, written=True)
    assert shown == twin_shown == []
    assert svg == twin
    build, sibling_reads, _ = _LAYER_DOORS[door]
    sibling = _LAYERED_TWIN if sibling_reads else _LAYERED
    route_shown, scene = _route(build(_LAYERED, _LAYERED, True))
    twin_route_shown, twin_scene = _route(build(_LAYERED_TWIN, sibling, False))
    assert route_shown == twin_route_shown == []
    assert scene == twin_scene


@pytest.mark.parametrize("door", [d for d in sorted(_LAYER_DOORS) if not _LAYER_DOORS[d][2]])
def test_a_layer_door_whose_twin_is_not_the_same_structure_draws_the_twins_marks(door: str):
    """The imputing layer draws the circles the written-out rows draw. A ``SecondaryY`` line reads
    the base table, so its twin cannot hold the written-out row without drawing it too; it keeps
    the caller's rows, as it does with no impute written."""
    svg, _ = _render_door(door, imputed=True)
    twin, _ = _render_door(door, imputed=False, written=True)
    plain, _ = _render_door(door, imputed=False)
    assert sorted(_circles(svg)) == sorted(_circles(twin))
    assert _vertices(svg) == (_vertices(plain) if door == "secondary-y" else _vertices(twin))


@pytest.mark.parametrize("door", sorted(_LAYER_DOORS))
def test_an_interactive_layer_door_builds_one_imputed_mark_more_than_the_caller_wrote(door: str):
    build, _, _ = _LAYER_DOORS[door]
    shown, scene = _route(build(_LAYERED, _LAYERED, True))
    plain_shown, plain_scene = _route(build(_LAYERED, _LAYERED, False))
    assert shown == plain_shown == []
    assert _scene_marks(scene) == _scene_marks(plain_scene) + 1


_FACETED_DOORS = ["plus-point-first", "plus-line-first", "chart-layer-inherited"]


@pytest.mark.parametrize("facet", sorted(_FACET_DOORS))
@pytest.mark.parametrize("door", _FACETED_DOORS)
def test_each_facet_panel_holds_its_own_imputed_rows_for_the_layer_that_wrote_them(
    door: str, facet: str
):
    """``_IMPUTE_FRAME`` misses ``(2, y)`` in each of its two panels: the imputing layer draws one
    imputed point in each, the sibling's rows are the caller's."""
    build, sibling_reads, _ = _LAYER_DOORS[door]
    arrange = _FACET_DOORS[facet]
    frame = _IMPUTE_FRAME
    svg, shown = _render_capturing(lambda: arrange(build(frame, frame, True)).to_svg())
    plain, plain_shown = _render_capturing(lambda: arrange(build(frame, frame, False)).to_svg())
    twin_sibling = _IMPUTE_COMPLETED if sibling_reads else frame
    twin, twin_shown = _render_capturing(
        lambda: arrange(build(_IMPUTE_COMPLETED, twin_sibling, False)).to_svg()
    )
    assert shown == plain_shown == twin_shown == []
    assert len(_circles(svg)) == len(_circles(plain)) + 2
    assert svg == twin
    route_shown, scene = _route(arrange(build(frame, frame, True)))
    plain_route_shown, plain_scene = _route(arrange(build(frame, frame, False)))
    assert route_shown == plain_route_shown == []
    twin_route_shown, twin_scene = _route(arrange(build(_IMPUTE_COMPLETED, twin_sibling, False)))
    assert twin_route_shown == []
    assert scene == twin_scene
    assert _scene_marks(scene) == _scene_marks(plain_scene) + 2


_COUNTED_DOORS = [d for d in sorted(_LAYER_DOORS) if not _LAYER_DOORS[d][2]]


@pytest.mark.parametrize("facet", sorted(_FACET_DOORS))
@pytest.mark.parametrize("door", _COUNTED_DOORS)
def test_each_facet_panel_holds_the_imputed_row_of_a_layer_that_has_no_same_structure_twin(
    door: str, facet: str
):
    """A layer that writes its own ``y``, a host beside a layer with no impute and a ``SecondaryY``
    base layer impute in each of the two panels, and the sibling keeps the caller's rows."""
    build, _, _ = _LAYER_DOORS[door]
    arrange = _FACET_DOORS[facet]
    frame = _IMPUTE_FRAME.with_columns(w=pl.col("v") * 2.0)
    svg, shown = _render_capturing(lambda: arrange(build(frame, frame, True)).to_svg())
    plain, plain_shown = _render_capturing(lambda: arrange(build(frame, frame, False)).to_svg())
    assert shown == plain_shown == []
    assert len(_circles(svg)) == len(_circles(plain)) + 2
    assert _vertices(svg) == _vertices(plain)


def test_a_chart_level_impute_inherited_by_a_layer_keyed_beyond_its_series_completes_that_layers_own():
    """The host keys ``h`` (one missing ``(2, y)``: 11 points); the shape layer inherits the impute
    and also keys ``d``, so it completes its ``(h, d)`` series (12 points): 23 marks."""
    frame = _IMPUTE_FRAME.drop("f")
    chart = _chart(frame, y=_y(True)).layer(fm.Layer(mark="point", encoding={"shape": "d"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _glyphs(svg) == 23
    route_shown, scene = _route(chart)
    assert route_shown == []
    assert _scene_marks(scene) == 23


@pytest.mark.parametrize("facet", sorted(_FACET_DOORS))
def test_a_faceted_chart_level_impute_inherited_by_a_layer_keyed_beyond_its_series(facet: str):
    """Each panel holds one ``(h, d)`` pair of series: the host completes ``h`` (one point per
    panel), the shape layer its ``(h, d)`` series (one per panel)."""
    arrange = _FACET_DOORS[facet]
    shaped = fm.Layer(mark="point", encoding={"shape": "d"})
    svg, shown = _render_capturing(
        lambda: arrange(_chart(_IMPUTE_FRAME, y=_y(True)).layer(shaped)).to_svg()
    )
    plain, plain_shown = _render_capturing(
        lambda: arrange(_chart(_IMPUTE_FRAME).layer(shaped)).to_svg()
    )
    assert shown == plain_shown == []
    assert _glyphs(svg) == _glyphs(plain) + 4


_DROP_FRAME = pl.concat(
    [
        _IMPUTE_FRAME.drop("f"),
        pl.DataFrame(
            {"t": [2.0], "h": ["y"], "d": [None], "v": [9.0]},
            schema={"t": pl.Float64, "h": pl.Utf8, "d": pl.Utf8, "v": pl.Float64},
        ),
    ]
)


def test_a_layer_the_null_key_drop_moved_is_imputed_in_place_under_its_own_rows():
    """A null ``d`` is dropped from the shape layer alone. The host holds every key of every series,
    so it imputes nothing and draws its 11 rows; the shape layer, on its 10 kept rows, completes
    ``(y, p)`` and ``(y, q)``: 12 points."""
    chart = _chart(_DROP_FRAME, y=_y(True)).layer(fm.Layer(mark="point", encoding={"shape": "d"}))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_null_key_dropped_message("mark_point", 1, "d")]
    assert _glyphs(svg) == 23
    route_shown, scene = _route(chart)
    assert route_shown == shown
    assert _scene_marks(scene) == 23


# ---------------------------------------------------------------------------
# A figure leaf and a concatenation hold the same rule on every leaf
# ---------------------------------------------------------------------------


def _relplot(frame: Any, *, imputed: bool) -> fm.Chart:
    return fm.relplot(frame, x="t", y=_y(imputed), hue="h", kind="line")


def test_a_figure_leaf_completes_its_series_as_the_twin_does():
    _assert_twin(
        lambda: _relplot(_SIMPLE, imputed=True),
        lambda: _relplot(_SIMPLE_TWIN, imputed=False),
        plain=lambda: _relplot(_SIMPLE, imputed=False),
        shown=[],
        interactive=True,
    )


def test_an_hconcat_completes_each_leaf_from_its_own_rows():
    other = pl.DataFrame({"t": [1, 2, 3], "h": ["x", "x", "x"], "v": [7.0, 8.0, 9.0]})
    _assert_twin(
        lambda: fm.hconcat(_point_layer(_SIMPLE, imputed=True), _relplot(other, imputed=True)),
        lambda: fm.hconcat(
            _point_layer(_SIMPLE_TWIN, imputed=False), _relplot(other, imputed=False)
        ),
        plain=lambda: fm.hconcat(
            _point_layer(_SIMPLE, imputed=False), _relplot(other, imputed=False)
        ),
        shown=[],
        interactive=True,
    )


# ---------------------------------------------------------------------------
# Inputs the step must leave alone
# ---------------------------------------------------------------------------


def test_a_chart_with_no_series_key_is_one_series_and_nothing_is_imputed():
    """No colour, detail, shape or dash: one series holds every key."""
    imputed, shown = _render_capturing(lambda: _chart(_SIMPLE, y=_y(True), color=None).to_svg())
    plain, plain_shown = _render_capturing(lambda: _chart(_SIMPLE, color=None).to_svg())
    assert shown == plain_shown == []
    assert imputed == plain


def test_a_temporal_ordinal_colour_is_imputed_per_date_series_as_its_label_text_is():
    """A date declared ordinal is a category: each date is a series of its own, labelled by its ISO
    text, so the impute completes the missing key of the second date exactly as it does for the
    same series spelled as text, and draws the point the unimputed chart lacks."""
    first, second = dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    frame = pl.DataFrame(
        {"t": [1, 2, 3, 1, 3], "h": [first] * 3 + [second] * 2, "v": _SIMPLE_VALUES}
    )
    twin = frame.with_columns(h=pl.col("h").cast(pl.Utf8))
    channel = fm.Color("h", type_="O")
    svg, shown = _render_capturing(_chart(frame, y=_y(True), color=channel).to_svg)
    twin_svg, twin_shown = _render_capturing(_chart(twin, y=_y(True), color=channel).to_svg)
    plain, _ = _render_capturing(_chart(frame, color=channel).to_svg)
    assert svg == twin_svg
    assert shown == twin_shown == []
    assert _glyphs(svg) == _glyphs(plain) + 1
    assert {"2024-01-01", "2024-01-02"} <= set(text_runs(svg))


def test_a_detail_no_key_reader_takes_is_refused_as_it_was_without_the_impute():
    """The series key cannot be read, so the render's own refusal speaks and nothing is imputed."""
    frame = pl.DataFrame(
        {"t": [1, 2, 3, 1, 3], "h": [dt.time(i) for i in (1, 1, 1, 2, 2)], "v": _SIMPLE_VALUES}
    )
    refusal = _refusal(_chart(frame, "mark_line", y=_y(True), color=None, detail="h"))
    assert refusal == _refusal(_chart(frame, "mark_line", color=None, detail="h"))
    assert refusal == _unreadable_key_refusal("mark_line", "h", "Time64(Nanosecond)")


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
def test_a_row_with_no_panel_completes_no_series(door: str):
    """Two rows with a null facet value are dropped from every panel; the keys they carry (7 and 8)
    complete no series, and the panels are the ones the frame without them imputes."""
    nulls = pl.DataFrame(
        {"t": [7.0, 8.0], "h": ["x", "x"], "d": ["p", "p"], "f": [None, None], "v": [50.0, 60.0]},
        schema={"t": pl.Float64, "h": pl.Utf8, "d": pl.Utf8, "f": pl.Utf8, "v": pl.Float64},
    )
    facet = _FACET_DOORS[door]
    _assert_twin(
        lambda: facet(_chart(pl.concat([_IMPUTE_FRAME, nulls]), y=_y(True))),
        lambda: facet(_chart(_IMPUTE_COMPLETED)),
        shown=[_facet_key_dropped_message(2, "f")],
        twin_shown=[],
    )


@pytest.mark.parametrize("door", ["unfaceted", *sorted(_FACET_DOORS)])
def test_an_impute_over_a_frame_with_no_rows_draws_as_the_chart_without_it_does(door: str):
    empty = _SIMPLE.clear().with_columns(f=pl.Series("f", [], dtype=pl.Utf8))

    def build(imputed: bool) -> fm.Chart:
        chart = _chart(empty, y=_y(imputed))
        return chart if door == "unfaceted" else _FACET_DOORS[door](chart)

    imputed, shown = _render_capturing(lambda: build(True).to_svg())
    plain, plain_shown = _render_capturing(lambda: build(False).to_svg())
    assert shown == plain_shown
    assert imputed == plain


@pytest.mark.parametrize("door", ["unfaceted", *sorted(_FACET_DOORS)])
def test_an_impute_over_rows_a_filter_emptied_draws_as_the_chart_without_it_does(door: str):
    def build(imputed: bool) -> fm.Chart:
        chart = _chart(_IMPUTE_FRAME, y=_y(imputed)).transform(fm.transform_filter("datum.t > 100"))
        return chart if door == "unfaceted" else _FACET_DOORS[door](chart)

    imputed, shown = _render_capturing(lambda: build(True).to_svg())
    plain, plain_shown = _render_capturing(lambda: build(False).to_svg())
    assert shown == plain_shown
    assert imputed == plain


# ---------------------------------------------------------------------------
# A row in no series is neither filled nor drawn
# ---------------------------------------------------------------------------


def _no_series_notice(rows: int, column: str, cause: str = "a null") -> str:
    """The notice for rows null on a series key (``cause`` names it: a binned colour's non-finite
    value is un-series too): they belong to no series, so they are left as written (never filled,
    and never drawn under another series' colour) and complete no key."""
    return (
        f"Impute: {rows} row(s) have {cause} in '{column}', so they belong to no series; they were "
        "left as written and no key is completed for them"
    )


def _with_null_series_row(
    rows: Sequence[_Row], null_row: tuple[int, float | None], *, panels: Sequence[str] = ()
) -> pl.DataFrame:
    """``rows`` (series ``x`` and ``y``) then one row with a null ``h`` and ``null_row`` as its key and
    value, once per panel when ``panels`` names them."""
    copies = list(panels) or [None]
    frame = {
        "t": [r[0] for _ in copies for r in rows] + [null_row[0] for _ in copies],
        "h": [r[1] for _ in copies for r in rows] + [None for _ in copies],
        "v": [r[2] for _ in copies for r in rows] + [null_row[1] for _ in copies],
    }
    if panels:
        frame["f"] = [name for name in copies for _ in rows] + list(copies)
    return pl.DataFrame(frame, schema_overrides={"h": pl.Utf8, "v": pl.Float64})


_NO_SERIES_WROTE: list[_Row] = [
    (1, "x", 1.0),
    (2, "x", 2.0),
    (3, "x", 3.0),
    (1, "y", 4.0),
    (3, "y", 5.0),
]
_NO_SERIES_NULL = (2, None)


def _no_series_twin(filled: float) -> list[_Row]:
    return [
        (1, "x", 1.0),
        (2, "x", 2.0),
        (3, "x", 3.0),
        (1, "y", 4.0),
        (2, "y", filled),
        (3, "y", 5.0),
    ]


#: ``(method options, what series y's imputed row is filled with)``: the usable values are 4 and 5.
_NO_SERIES_FILLS = [
    pytest.param({"value": 0}, 0.0, id="value"),
    pytest.param({"method": "mean"}, 4.5, id="mean"),
    pytest.param({"method": "median"}, 4.5, id="median"),
    pytest.param({"method": "min"}, 4.0, id="min"),
    pytest.param({"method": "max"}, 5.0, id="max"),
]


@pytest.mark.parametrize(("options", "filled"), _NO_SERIES_FILLS)
def test_a_row_null_on_the_colour_is_not_filled_and_not_drawn_and_the_census_names_it_once(
    options: dict[str, Any], filled: float
):
    """The null-``h`` row keeps its own null value: it is not filled (not with ``value``, not with a
    statistic) and so draws nothing, where it had drawn under series ``x``'s colour. The chart is the
    one drawn with only the genuine imputed row written down, and the census names the null row once."""
    frame = _with_null_series_row(_NO_SERIES_WROTE, _NO_SERIES_NULL)
    twin = _with_null_series_row(_no_series_twin(filled), _NO_SERIES_NULL)
    census = _slot_message("mark_point", "y", "null", "h")
    svg, shown = _render_capturing(lambda: _chart(frame, y=fm.Y("v", impute=options)).to_svg())
    twin_svg, twin_shown = _render_capturing(lambda: _chart(twin).to_svg())
    assert twin_shown == [census]
    assert shown == [_no_series_notice(1, "h"), census]
    assert svg == twin_svg
    assert len(_circles(svg)) == 6


@pytest.mark.parametrize("mark", ["mark_line", "mark_area"])
def test_a_row_null_on_the_colour_draws_as_it_does_without_the_impute_on_a_connected_mark(
    mark: str,
):
    frame = _with_null_series_row(_NO_SERIES_WROTE, _NO_SERIES_NULL)
    twin = _with_null_series_row(_no_series_twin(0.0), _NO_SERIES_NULL)
    svg, shown = _render_capturing(lambda: _chart(frame, mark, y=_y(True)).to_svg())
    twin_svg, twin_shown = _render_capturing(lambda: _chart(twin, mark).to_svg())
    assert shown == [_no_series_notice(1, "h"), *twin_shown]
    assert svg == twin_svg


@pytest.mark.parametrize("channel", ["detail", "stroke_dash"])
def test_a_row_null_on_a_second_series_key_is_not_filled_either(channel: str):
    """The null is on the key that makes the series (``detail`` or ``stroke_dash``, no colour)."""
    wrote = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 3, 2],
            "h": ["x", "x", "x", "y", "y", None],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, None],
        }
    )
    twin = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3, 2],
            "h": ["x", "x", "x", "y", "y", "y", None],
            "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, None],
        }
    )
    kwargs = {"color": None, channel: "h"}
    svg, shown = _render_capturing(
        lambda: _chart(wrote, "mark_line", y=_y(True), **kwargs).to_svg()
    )
    twin_svg, twin_shown = _render_capturing(lambda: _chart(twin, "mark_line", **kwargs).to_svg())
    assert shown == [_no_series_notice(1, "h"), *twin_shown]
    assert svg == twin_svg


_NULL_PANELS = ("F1", "F2")

_NO_SERIES_DOORS: dict[str, Callable[[Any, Any, bool], Any]] = {
    "facet": lambda a, b, imputed: _FACET_DOORS["facet"](_point_layer(a, imputed=imputed)),
    "facet-col": lambda a, b, imputed: _FACET_DOORS["facet-col"](_point_layer(a, imputed=imputed)),
    "plus": lambda a, b, imputed: _point_layer(a, imputed=imputed) + _line_layer(b),
    "chart-layer-inherited": lambda a, b, imputed: _inherited(a, imputed=imputed),
}


@pytest.mark.parametrize("door", sorted(_NO_SERIES_DOORS))
def test_a_row_null_on_the_colour_is_left_as_written_on_every_layer_door_and_panel(door: str):
    """Each panel holds the null row once. The imputing layer draws the genuine imputed row and not
    the null one, and the ordered full warning list is the twin's with the notice in its place
    (once per panel-aggregated count); the sibling of a ``+`` keeps the caller's rows."""
    panels = _NULL_PANELS if door.startswith("facet") else ()
    frame = _with_null_series_row(_NO_SERIES_WROTE, _NO_SERIES_NULL, panels=panels)
    twin = _with_null_series_row(_no_series_twin(0.0), _NO_SERIES_NULL, panels=panels)
    sibling_reads = door == "chart-layer-inherited"
    build = _NO_SERIES_DOORS[door]
    svg, shown = _render_capturing(lambda: build(frame, frame, True).to_svg())
    twin_svg, twin_shown = _render_capturing(
        lambda: build(twin, twin if sibling_reads else frame, False).to_svg()
    )
    assert svg == twin_svg
    notice = _no_series_notice(len(panels) or 1, "h")
    # the notice precedes the census; on an inherited impute the line layer's census comes first
    expected = (
        [twin_shown[0], notice, twin_shown[1]]
        if door == "chart-layer-inherited"
        else [notice, *twin_shown]
    )
    assert shown == expected
    assert len(_census(shown)) == len(_census(twin_shown)) > 0


# ---------------------------------------------------------------------------
# A refusal names its layer the same way on every door and both output kinds
# ---------------------------------------------------------------------------

_LEAF = "failed to render composite leaf leaf #1: "
_BAD_FILL = {"value": "none"}
_BAD_FILL_REFUSAL = _impute_kind_refusal("impute", "v", "number-column-text")


def _bad_point(frame: Any) -> fm.Chart:
    return _chart(frame, y=fm.Y("v", impute=_BAD_FILL))


#: ``(builder, the text prefix on a static render, on an interactive one)``: the point member is
#: layer 1 of every layered door, member #1 of a concatenation, and the only unit of a flat chart.
_REFUSAL_DOORS: dict[str, tuple[Callable[[Any], Any], str, str]] = {
    "flat": (_bad_point, "", ""),
    "facet": (lambda f: _FACET_DOORS["facet"](_bad_point(f)), "", ""),
    "facet-col": (lambda f: _FACET_DOORS["facet-col"](_bad_point(f)), "", ""),
    "plus": (lambda f: _line_layer(f) + _bad_point(f), "layer 1: ", "layer 1: "),
    "chart-layer": (
        lambda f: _line_layer(f).layer(
            fm.Layer(mark="point", encoding={"y": fm.Y("v", impute=_BAD_FILL)})
        ),
        "layer 1: ",
        "layer 1: ",
    ),
    "layer-chart": (
        lambda f: fm.LayerChart(_line_layer(f), _bad_point(f)),
        _LEAF + "layer 1: ",
        "layer 1: ",
    ),
    "layer-chart-independent-y": (
        lambda f: fm.LayerChart(_line_layer(f), _bad_point(f), resolve={"y": "independent"}),
        "layer 1: ",
        "layer 1: ",
    ),
    "hconcat": (lambda f: fm.hconcat(_line_layer(f), _bad_point(f)), _LEAF, _LEAF),
}


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("door", sorted(_REFUSAL_DOORS))
def test_a_refusal_names_its_layer_the_same_way_on_every_door_and_output(door: str, output: str):
    """The same refusal text (``_BAD_FILL_REFUSAL``) arrives with its layer named once, through one
    mechanism: the bare text on a flat or faceted chart, ``layer 1: `` on every layered door, inside
    the composite-leaf envelope a static ``LayerChart`` or a concatenation render through."""
    build, static_prefix, interactive_prefix = _REFUSAL_DOORS[door]
    chart = build(_IMPUTE_FRAME)
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg() if output == "static" else chart.interactive().scene_json
    prefix = static_prefix if output == "static" else interactive_prefix
    assert str(excinfo.value) == prefix + _BAD_FILL_REFUSAL


# ---------------------------------------------------------------------------
# A notice names a series in the caller's terms
# ---------------------------------------------------------------------------


def _no_usable_notice(series: str) -> str:
    return (
        f"Impute: group {series} has no usable (non-null, finite) values; the whole column's "
        "statistic was used instead"
    )


_DETAIL_DAYS = {
    "Date": (dt.date(2024, 1, 1), dt.date(2024, 1, 2)),
    "Datetime": (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)),
}


@pytest.mark.parametrize("kind", sorted(_DETAIL_DAYS))
def test_a_date_or_datetime_series_is_named_as_a_date_the_way_the_transform_door_names_it(
    kind: str,
):
    """Series ``d`` of the second day holds no usable value: the notice names it
    ``d=2024-01-02T00:00:00``, as ``transform_impute`` does, and never as an epoch number."""
    first, second = _DETAIL_DAYS[kind]
    frame = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2],
            "d": [first] * 3 + [second] * 2,
            "v": pl.Series("v", [1.0, 2.0, 3.0, None, None], dtype=pl.Float64),
        }
    )
    _, shown = _render_capturing(
        lambda: (
            fm.Chart(frame)
            .mark_line()
            .encode(x="t", y=fm.Y("v", impute={"method": "mean"}), detail="d")
            .to_svg()
        )
    )
    step = fm.transform_impute("v", method="mean", key="t", groupby=["d"])
    _, transform_shown = _render_capturing(
        lambda: (
            fm.Chart(frame).transform(step).mark_line().encode(x="t", y="v", detail="d").to_svg()
        )
    )
    assert shown == [_no_usable_notice("d=2024-01-02T00:00:00")]
    assert shown == transform_shown


def test_a_binned_colours_series_is_named_by_its_range_not_its_bucket_index():
    """The first bin ``[0, 5)`` holds no usable value: the notice names it by the range the legend
    shows for it."""
    frame = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 3],
            "h": [1.0, 1.2, 1.1, 8.0, 8.5],
            "v": pl.Series("v", [None, None, None, 4.0, 5.0], dtype=pl.Float64),
        }
    )
    _, shown = _render_capturing(
        lambda: _chart(
            frame,
            "mark_line",
            y=fm.Y("v", impute={"method": "mean"}),
            color=fm.Color("h", bin=_BIN_BY_FIVE),
        ).to_svg()
    )
    assert shown == [_no_usable_notice("h=[0, 5)")]


# ---------------------------------------------------------------------------
# A fallback is reported once per computation, not once per layer that reads it
# ---------------------------------------------------------------------------

#: Series ``y`` holds no usable value: a ``mean`` takes the whole column's, with a notice.
_FALLBACK_FRAME = _simple(_SIMPLE_KEYS, _SIMPLE_SERIES, [1.0, 2.0, 3.0, None, None]).with_columns(
    pl.col("v").cast(pl.Float64)
)
_FALLBACK = _no_usable_notice("h=y")


def test_layers_that_share_one_impute_report_its_fallback_once():
    """``Chart.layer(point, line)``: both layers read the chart's one completed rows, computed once."""
    chart = _chart(_FALLBACK_FRAME, y=fm.Y("v", impute={"method": "mean"})).layer(
        fm.Layer(mark="line")
    )
    assert _render_capturing(chart.to_svg)[1] == [_FALLBACK]
    route_shown, _ = _route(chart)
    assert route_shown == [_FALLBACK]


def test_layers_that_impute_separately_each_report_their_own_fallback_in_layer_order():
    """Two ``+`` operands write different options, so each completes its own rows and each reports."""
    point = _chart(_FALLBACK_FRAME, y=fm.Y("v", impute={"method": "mean"}))
    line = _chart(_FALLBACK_FRAME, "mark_line", y=fm.Y("v", impute={"method": "median"}))
    assert _render_capturing((point + line).to_svg)[1] == [_FALLBACK, _FALLBACK]


@pytest.mark.parametrize("bad", [_NAN, _INF, -_INF], ids=["nan", "inf", "-inf"])
def test_a_non_finite_binned_colour_row_is_left_as_written_and_the_notice_names_its_cause(
    bad: float,
):
    """A binned colour value no bin takes (NaN or infinite) puts its row in no series. The row is
    left as written, the genuine missing row is imputed from its bin, and the notice says the key
    is null or non-finite, not only null."""
    frame = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 3, 2],
            "h": [1.0, 1.2, 1.1, 8.0, 8.5, bad],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    twin = pl.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3, 2],
            "h": [1.0, 1.2, 1.1, 8.0, 8.0, 8.5, bad],
            "v": [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0],
        }
    )
    channel = fm.Color("h", bin=_BIN_BY_FIVE)
    svg, shown = _render_capturing(lambda: _chart(frame, y=_y(True), color=channel).to_svg())
    twin_svg, twin_shown = _render_capturing(lambda: _chart(twin, color=channel).to_svg())
    assert shown == [_no_series_notice(1, "h", "a null or non-finite value"), *twin_shown]
    assert svg == twin_svg
