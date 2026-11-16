"""Under a caller's ``CoordFlip`` every annotation names the caller's channels.

A caller who writes ``encode(x="a", y="b").coord(fm.CoordFlip())`` still calls ``a`` the x channel
and ``b`` the y channel; the flip only moves them on the page. An annotation follows the same
words: its ``x`` is the caller's x and lands where the caller's x data lands after the flip, its
``y`` the caller's y, and a span's ``axis`` names the caller's axis. Pixel offsets (``dx``,
``dy``), text anchors and label positions stay in drawn space. A bracket's ``direction`` exchanges
with its coordinate pair, so its tips and label still point away from its baseline. A
``NormCoord`` or ``PixelCoord`` measures along the drawn axis its field now names, from the left
or the top.

The oracle for every kind is the same chart drawn without the flip, with its channels exchanged
(``encode(x="b", y="a")``) and the annotation's pairs (and a bracket's direction) exchanged: the
two renders are byte-equal.
The unflipped twin is built from the unflipped path alone, so the comparison cannot break on both
sides together. Key rows also pin an independent position oracle (the mark drawn for the data
the annotation names, or an axis tick), so a pair of renders that both lost the annotation cannot
pass.

A flip a figure applied itself (``catplot(orient="h")``) swaps nothing, and a chart with no flip
draws the bytes it drew before; both are pinned by digest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Callable, NamedTuple

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from tests.test_annotation_categories import (
    _digest,
    _COLOUR,
    _DF,
    _FACETED,
    _IMAGE_SRC,
    _OTHER,
    _TEMPORAL,
    _all,
    _assert_refused_on_both_routes,
    _band_edges,
    _continuous_message,
    _elements,
    _lo_hi,
    _mixed_span_message,
    _near,
    _one,
    _scene_annotations,
    _span_extent,
    _ticks,
    _unknown_message,
    _value_message,
)
from tests.test_null_group_warning import _NO_ROWS_TO_DRAW, _render_capturing

_FLIP = fm.CoordFlip()
_NUM = pl.DataFrame({"a": [1.0, 2.0, 3.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0]})
_FRAME_FILL = "#ff00f0"


# ---------------------------------------------------------------------------
# A chart drawn with the flip, and its unflipped twin with the channels exchanged
# ---------------------------------------------------------------------------


class _Twin(NamedTuple):
    """The caller's flipped chart and the unflipped chart that draws the same picture."""

    flipped: Callable[[], fm.Chart]
    native: Callable[[], fm.Chart]
    points: "_Points"


class _Points(NamedTuple):
    """Two caller-space positions: ``(x1, y1)`` and ``(x2, y2)``."""

    x1: Any
    y1: Any
    x2: Any
    y2: Any


def _numeric_twin() -> _Twin:
    return _Twin(
        lambda: fm.Chart(_NUM).mark_point().encode(x="a", y="b").coord(_FLIP),
        lambda: fm.Chart(_NUM).mark_point().encode(x="b", y="a"),
        _Points(2.0, 3.0, 4.0, 7.0),
    )


def _category_twin() -> _Twin:
    return _Twin(
        lambda: fm.Chart(_DF).mark_bar().encode(x="c", y="v").coord(_FLIP),
        lambda: fm.Chart(_DF).mark_bar().encode(x="v", y="c"),
        _Points("b", 1.0, "c", 3.0),
    )


def _sorted_category_twin() -> _Twin:
    return _Twin(
        lambda: fm.Chart(_DF).mark_bar().encode(x=fm.X("c", sort="-y"), y="v").coord(_FLIP),
        lambda: fm.Chart(_DF).mark_bar().encode(x="v", y=fm.Y("c", sort="-x")),
        _Points("b", 1.0, "c", 3.0),
    )


def _temporal_twin(x1: Any, x2: Any) -> _Twin:
    return _Twin(
        lambda: fm.Chart(_TEMPORAL).mark_line().encode(x="date", y="v").coord(_FLIP),
        lambda: fm.Chart(_TEMPORAL).mark_line().encode(x="v", y="date"),
        _Points(x1, 1.5, x2, 2.5),
    )


_TWINS = {
    "numeric": _numeric_twin,
    "category": _category_twin,
    "sorted_category": _sorted_category_twin,
    "temporal_date": lambda: _temporal_twin(dt.date(2020, 1, 2), dt.date(2020, 1, 3)),
    "temporal_iso": lambda: _temporal_twin("2020-01-02", "2020-01-03"),
}

# A bracket's direction exchanges with its pair: the caller's "up" (larger y) runs right on the
# page, "right" (larger x) runs up, "down" runs left and "left" runs down. "above" and "below"
# are the synonyms of "up" and "down".
_EXCHANGED_DIRECTION = {
    "above": "right",
    "up": "right",
    "below": "left",
    "down": "left",
    "left": "down",
    "right": "up",
}


def _pair(swap: bool, x: Any, y: Any) -> tuple[Any, Any]:
    """The coordinate pair as written on the flipped chart, or exchanged for its twin."""
    return (y, x) if swap else (x, y)


def _letter(swap: bool, axis: str) -> str:
    return {"x": "y", "y": "x"}[axis] if swap else axis


# Each builder writes one annotation in the caller's channels (``swap=False``, for the flipped
# chart) or with every pair, and a bracket's direction, exchanged (``swap=True``, for the
# unflipped twin). The pixel offsets, anchors and label positions are the same on both sides: they
# stay in drawn space. ``top-right`` and ``bottom`` are not their own transposes, so an exchange of
# either would change the bytes.
def _bracket(
    swap: bool, p: "_Points", direction: str | None = None, *, stroke: str = _COLOUR["bracket"]
) -> Any:
    """A bracket with ``direction`` (the library default when ``None``); the twin takes the
    exchanged direction."""
    kwargs: dict[str, Any] = {}
    if swap:
        kwargs["direction"] = _EXCHANGED_DIRECTION[direction or "above"]
    elif direction is not None:
        kwargs["direction"] = direction
    return ann.bracket(
        *_pair(swap, p.x1, p.y1), *_pair(swap, p.x2, p.y2), label="range", stroke=stroke, **kwargs
    )


_BUILDERS: dict[str, Callable[[bool, _Points], Any]] = {
    "text": lambda s, p: ann.text(
        *_pair(s, p.x1, p.y1), "HELLO", color=_COLOUR["text"], anchor="end", dx=7, dy=-5
    ),
    "arrow": lambda s, p: ann.arrow(
        *_pair(s, p.x1, p.y1), *_pair(s, p.x2, p.y2), stroke=_COLOUR["arrow"]
    ),
    "rect": lambda s, p: ann.rect(
        *_pair(s, p.x1, p.y1), *_pair(s, p.x2, p.y2), fill=_COLOUR["rect"]
    ),
    "line": lambda s, p: ann.line(
        *_pair(s, p.x1, p.y1), *_pair(s, p.x2, p.y2), stroke=_COLOUR["line"]
    ),
    "bracket": lambda s, p: _bracket(s, p, "left"),
    "callout": lambda s, p: ann.callout(
        *_pair(s, p.x1, p.y1),
        "note",
        text_x=_pair(s, p.x2, p.y2)[0],
        text_y=_pair(s, p.x2, p.y2)[1],
        border_color=_COLOUR["callout"],
    ),
    "image": lambda s, p: ann.image(
        *_pair(s, p.x1, p.y1), _IMAGE_SRC, width=10, height=10, anchor="top-right"
    ),
    "span_x": lambda s, p: ann.span(
        _letter(s, "x"), p.x1, p.x2, fill=_COLOUR["span"], label="band", label_position="bottom"
    ),
    "span_y": lambda s, p: ann.span(_letter(s, "y"), p.y1, p.y2, fill=_COLOUR["span"]),
}
_KINDS = sorted(_BUILDERS)


def _drawn(twin: _Twin, kind: str) -> tuple[str, str]:
    """The flipped chart's SVG and its twin's, each with the annotation ``kind``."""
    return (
        (twin.flipped() + _BUILDERS[kind](False, twin.points)).to_svg(),
        (twin.native() + _BUILDERS[kind](True, twin.points)).to_svg(),
    )


def _scenes(twin: _Twin, kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        _scene_annotations(twin.flipped() + _BUILDERS[kind](False, twin.points)),
        _scene_annotations(twin.native() + _BUILDERS[kind](True, twin.points)),
    )


def test_the_unflipped_twin_draws_the_flipped_chart_with_no_annotation():
    """Without an annotation the two charts are byte-equal, so any difference below is the
    annotation's."""
    for name, make in _TWINS.items():
        twin = make()
        assert twin.flipped().to_svg() == twin.native().to_svg(), name


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_an_annotation_under_a_caller_flip_is_the_twin_drawn_with_axes_exchanged(name, kind):
    """The flipped chart is the unflipped chart with its axes exchanged: every kind, on a numeric,
    category, sorted-category, date and ISO-string axis."""
    flipped, native = _drawn(_TWINS[name](), kind)
    assert flipped == native
    # The annotation is drawn: its colour or its image is in the output.
    colour = _COLOUR.get(kind.split("_")[0], "")
    assert (_IMAGE_SRC if kind == "image" else colour) in flipped


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_the_interactive_route_draws_the_same_nodes_as_the_twin(name, kind):
    """The interactive scene of the flipped chart carries the nodes of the twin's scene."""
    flipped, native = _scenes(_TWINS[name](), kind)
    assert flipped
    assert flipped == native


# ---------------------------------------------------------------------------
# Position oracles that do not use the twin
# ---------------------------------------------------------------------------


def _circles(svg: str) -> list[tuple[float, float]]:
    return [
        (float(x), float(y)) for x, y in re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"', svg)
    ]


def _tick_at(ticks: dict[str, float], value: float) -> float:
    (position,) = [pos for label, pos in ticks.items() if float(label) == value]
    return position


def _text_at(svg: str, content: str = "HELLO") -> tuple[float, float]:
    hits = [e for e in _elements(svg) if e.tag == "text" and e.text == content]
    assert len(hits) == 1, f"{content!r} is drawn {len(hits)} times"
    return float(hits[0].attrs["x"]), float(hits[0].attrs["y"])


def _frame(chart: fm.Chart) -> tuple[float, float, float, float]:
    """The plot rectangle as ``(left, top, width, height)``: a rect from the plot's top-left to
    its bottom-right corner, in normalised coordinates."""
    cover = ann.rect(
        fm.NormCoord(0), fm.NormCoord(0), fm.NormCoord(1), fm.NormCoord(1), fill=_FRAME_FILL
    )
    rect = _one((chart + cover).to_svg(), "rect", fill=_FRAME_FILL)
    return tuple(  # type: ignore[return-value]
        float(rect.attrs[k]) for k in ("x", "y", "width", "height")
    )


def test_a_text_at_a_caller_data_point_lands_on_the_mark_drawn_for_that_row():
    """``text(a, b)`` at a row's own ``(a, b)`` sits on that row's point after the flip."""
    svg = (_numeric_twin().flipped() + ann.text(3.0, 6.0, "HELLO", anchor="middle")).to_svg()
    point = [c for c in _circles(svg) if c == pytest.approx(_text_at(svg), abs=0.01)]
    assert len(point) == 1


def test_a_text_at_a_caller_category_lands_on_that_categorys_band_and_the_callers_value():
    """``text("b", 2.0)``: band ``b``'s centre on the vertical axis, caller-y 2.0 on the horizontal
    one."""
    svg = (_category_twin().flipped() + ann.text("b", 2.0, "HELLO", anchor="middle")).to_svg()
    x, y = _text_at(svg)
    _near(y, _ticks(svg, "y")["b"])
    _near(x, _tick_at(_ticks(svg, "x"), 2.0))


def test_a_text_at_a_caller_category_lands_on_its_band_under_a_sort():
    svg = (
        _sorted_category_twin().flipped() + ann.text("b", 2.0, "HELLO", anchor="middle")
    ).to_svg()
    ticks = _ticks(svg, "y")
    assert [c for c, _ in sorted(ticks.items(), key=lambda kv: kv[1])] == ["c", "d", "a", "b"]
    _near(_text_at(svg)[1], ticks["b"])


def test_a_category_span_shades_whole_bands_on_the_vertical_axis_across_the_plot_width():
    """``span("x", "b", "c")``: whole bands ``b`` and ``c`` on the vertical axis, the full plot
    width."""
    chart = _sorted_category_twin().flipped()
    svg = (chart + ann.span("x", "b", "c", fill=_COLOUR["span"])).to_svg()
    ticks = _ticks(svg, "y")
    lo_b, hi_b = _band_edges(ticks, "b", 0.0)
    lo_c, hi_c = _band_edges(ticks, "c", 0.0)
    _near(list(_span_extent(svg, "y")), [min(lo_b, lo_c), max(hi_b, hi_c)])
    left, _, width, _ = _frame(chart)
    (rect,) = _all(svg, "rect", fill=_COLOUR["span"])
    _near(list(_lo_hi(rect, "x")), [left, left + width])


def test_a_numeric_span_is_a_horizontal_band_at_the_callers_x_inside_the_plot():
    """``span("x", 2, 4)``: caller-x 2..4 sits on the vertical axis, so the band is horizontal."""
    chart = _numeric_twin().flipped()
    svg = (chart + ann.span("x", 2.0, 4.0, fill=_COLOUR["span"])).to_svg()
    (rect,) = _all(svg, "rect", fill=_COLOUR["span"])
    ticks = _ticks(svg, "y")
    _near(sorted(_lo_hi(rect, "y")), sorted([_tick_at(ticks, 2.0), _tick_at(ticks, 4.0)]))
    left, _, width, _ = _frame(chart)
    _near(list(_lo_hi(rect, "x")), [left, left + width])


def test_a_numeric_span_on_the_callers_y_is_a_vertical_band():
    chart = _numeric_twin().flipped()
    svg = (chart + ann.span("y", 3.0, 6.0, fill=_COLOUR["span"])).to_svg()
    (rect,) = _all(svg, "rect", fill=_COLOUR["span"])
    ticks = _ticks(svg, "x")
    _near(list(_lo_hi(rect, "x")), [_tick_at(ticks, 3.0), _tick_at(ticks, 6.0)])
    _, top, _, height = _frame(chart)
    _near(list(_lo_hi(rect, "y")), [top, top + height])


def test_annotate_hline_is_a_vertical_line_at_the_callers_y():
    """The caller's y runs along the horizontal axis, so the line is vertical."""
    svg = (_numeric_twin().flipped() + fm.annotate_hline(5.0, stroke=_COLOUR["line"])).to_svg()
    lines = _all(svg, "line", stroke=_COLOUR["line"])
    assert lines
    x_at_five = _tick_at(_ticks(svg, "x"), 5.0)
    for line in lines:
        _near([float(line.attrs["x1"]), float(line.attrs["x2"])], [x_at_five, x_at_five])
        assert float(line.attrs["y1"]) != float(line.attrs["y2"])


def test_annotate_vline_is_a_horizontal_line_at_the_callers_x():
    """The caller's x runs along the vertical axis, so the line is horizontal."""
    svg = (_numeric_twin().flipped() + fm.annotate_vline(3.0, stroke=_COLOUR["line"])).to_svg()
    lines = _all(svg, "line", stroke=_COLOUR["line"])
    assert lines
    y_at_three = _tick_at(_ticks(svg, "y"), 3.0)
    for line in lines:
        _near([float(line.attrs["y1"]), float(line.attrs["y2"])], [y_at_three, y_at_three])
        assert float(line.attrs["x1"]) != float(line.attrs["x2"])


def test_annotate_vline_at_a_category_is_a_horizontal_line_on_that_band():
    """The category is the caller's x, so its band is on the vertical axis."""
    svg = (_sorted_category_twin().flipped() + fm.annotate_vline("b", stroke="#ff0010")).to_svg()
    lines = _all(svg, "line", stroke="#ff0010")
    assert lines
    for line in lines:
        centre = _ticks(svg, "y")["b"]
        _near([float(line.attrs["y1"]), float(line.attrs["y2"])], [centre, centre])


def test_annotate_hline_on_the_callers_value_beside_a_category_is_a_vertical_line():
    svg = (_category_twin().flipped() + fm.annotate_hline(2.0, stroke="#ff0011")).to_svg()
    lines = _all(svg, "line", stroke="#ff0011")
    assert lines
    for line in lines:
        at_two = _tick_at(_ticks(svg, "x"), 2.0)
        _near([float(line.attrs["x1"]), float(line.attrs["x2"])], [at_two, at_two])


def _line_in_callers_coordinates(svg: str, stroke: str) -> list[tuple[float, float]]:
    """Two points of the drawn line, inside the plot, as ``(caller x, caller y)``: the line is
    sampled where the horizontal axis reads 3 and 7 and each sample is read back through the
    ticks (the vertical axis carries the caller's x, the horizontal axis the caller's y). The
    line itself runs far past the plot and is trimmed by the clip path, so its own ends say
    nothing."""
    x_ticks, y_ticks = _ticks(svg, "x"), _ticks(svg, "y")
    (line,) = _all(svg, "line", stroke=stroke)
    x1, y1, x2, y2 = (float(line.attrs[k]) for k in ("x1", "y1", "x2", "y2"))
    (p0, v0), (p1, v1) = [(pos, float(label)) for label, pos in list(y_ticks.items())[:2]]
    samples = []
    for caller_y in (3.0, 7.0):
        px = _tick_at(x_ticks, caller_y)
        py = y1 + (px - x1) * (y2 - y1) / (x2 - x1)
        samples.append((v0 + (py - p0) * (v1 - v0) / (p1 - p0), caller_y))
    return samples


@pytest.mark.parametrize(
    "slope,intercept",
    [(2.0, 1.0), (0.5, 2.0), (1.0, 0.0)],
    ids=["2x+1", "x/2+2", "identity"],
)
def test_annotate_abline_is_the_line_y_equals_slope_x_plus_intercept_in_the_callers_coordinates(
    slope, intercept
):
    """Every drawn point ``(x, y)`` of the line, read back as the caller's data, satisfies
    ``y = slope * x + intercept``. ``(1, 0)`` is its own mirror image, so it also holds with the
    axes unexchanged; the other two do not."""
    chart = _numeric_twin().flipped() + fm.annotate_abline(slope, intercept, stroke="#ff0012")
    ends = _line_in_callers_coordinates(chart.to_svg(), "#ff0012")
    for x, y in ends:
        # The line's ends are millions of pixels out, where single precision leaves a few pixels
        # of slack, so the read-back tolerance is 0.15 data units (about 15 pixels).
        assert y == pytest.approx(slope * x + intercept, abs=0.15)


# ---------------------------------------------------------------------------
# NormCoord and PixelCoord measure along the drawn axis the field now names
# ---------------------------------------------------------------------------


def test_a_norm_x_is_a_fraction_down_from_the_top_at_the_callers_y():
    """``text(NormCoord(0.1), 3.0)``: 10% down from the top, at caller-y 3.0 on the horizontal
    axis."""
    twin = _numeric_twin()
    svg = (twin.flipped() + ann.text(fm.NormCoord(0.1), 6.0, "HELLO", anchor="middle")).to_svg()
    left, top, width, height = _frame(twin.flipped())
    x, y = _text_at(svg)
    _near(y, top + 0.1 * height)
    _near(x, _tick_at(_ticks(svg, "x"), 6.0))


def test_a_norm_y_is_a_fraction_from_the_left_at_the_callers_x():
    twin = _numeric_twin()
    svg = (twin.flipped() + ann.text(3.0, fm.NormCoord(0.25), "HELLO", anchor="middle")).to_svg()
    left, top, width, height = _frame(twin.flipped())
    x, y = _text_at(svg)
    _near(x, left + 0.25 * width)
    _near(y, _tick_at(_ticks(svg, "y"), 3.0))


def test_a_pixel_offset_in_a_swapped_field_measures_along_the_drawn_axis_it_names():
    """``text(PixelCoord(40), PixelCoord(60))``: x names the vertical axis (40 from the top),
    y the horizontal axis (60 from the left)."""
    twin = _numeric_twin()
    svg = (
        twin.flipped()
        + ann.text(fm.PixelCoord(40.0), fm.PixelCoord(60.0), "HELLO", anchor="middle")
    ).to_svg()
    left, top, _, _ = _frame(twin.flipped())
    x, y = _text_at(svg)
    _near([x, y], [left + 60.0, top + 40.0])


def test_a_pixel_x_beside_a_data_y_sits_below_the_top_at_the_callers_y():
    twin = _numeric_twin()
    svg = (twin.flipped() + ann.text(fm.PixelCoord(40.0), 6.0, "HELLO", anchor="middle")).to_svg()
    _, top, _, _ = _frame(twin.flipped())
    x, y = _text_at(svg)
    _near(y, top + 40.0)
    _near(x, _tick_at(_ticks(svg, "x"), 6.0))


@pytest.mark.parametrize(
    "build",
    [
        lambda s: ann.text(*_pair(s, fm.NormCoord(0.1), 6.0), "HELLO", color=_COLOUR["text"]),
        lambda s: ann.text(*_pair(s, 3.0, fm.NormCoord(0.25)), "HELLO", color=_COLOUR["text"]),
        lambda s: ann.text(
            *_pair(s, fm.PixelCoord(40.0), fm.PixelCoord(60.0)), "HELLO", color=_COLOUR["text"]
        ),
        lambda s: ann.line(
            *_pair(s, fm.NormCoord(0.2), 3.0),
            *_pair(s, fm.NormCoord(0.8), 7.0),
            stroke=_COLOUR["line"],
        ),
        lambda s: ann.rect(
            *_pair(s, fm.PixelCoord(30.0), fm.NormCoord(0.1)),
            *_pair(s, fm.PixelCoord(90.0), fm.NormCoord(0.6)),
            fill=_COLOUR["rect"],
        ),
        lambda s: ann.span(_letter(s, "x"), fm.NormCoord(0.2), fm.NormCoord(0.6), fill="#ff0005"),
        lambda s: ann.span(
            _letter(s, "y"), fm.PixelCoord(50.0), fm.PixelCoord(150.0), fill="#ff0005"
        ),
    ],
    ids=["norm_x", "norm_y", "pixel_pair", "norm_line", "mixed_rect", "norm_span", "pixel_span"],
)
def test_a_norm_or_pixel_annotation_is_the_twin_drawn_with_axes_exchanged(build):
    twin = _numeric_twin()
    assert (twin.flipped() + build(False)).to_svg() == (twin.native() + build(True)).to_svg()
    assert _scene_annotations(twin.flipped() + build(False)) == _scene_annotations(
        twin.native() + build(True)
    )


# ---------------------------------------------------------------------------
# Drawn-space offsets, anchors, directions and label positions do not exchange
# ---------------------------------------------------------------------------


def test_a_pixel_offset_and_an_anchor_stay_in_drawn_space():
    """The twin takes the same ``dx``, ``dy`` and ``anchor``. A text with ``dx=7`` sits 7 pixels
    right of its data position on the page and ``anchor="end"`` ends there, whichever way the
    axes run."""
    twin = _numeric_twin()
    plain = (twin.flipped() + ann.text(3.0, 6.0, "HELLO", anchor="middle")).to_svg()
    shifted = (twin.flipped() + ann.text(3.0, 6.0, "HELLO", anchor="middle", dx=7, dy=-5)).to_svg()
    px, py = _text_at(plain)
    sx, sy = _text_at(shifted)
    _near([sx - px, sy - py], [7.0, -5.0])
    ended = (twin.flipped() + ann.text(3.0, 6.0, "HELLO", anchor="end")).to_svg()
    el = [e for e in _elements(ended) if e.text == "HELLO"][0]
    assert el.attrs["text-anchor"] == "end"
    _near(float(el.attrs["x"]), px)


# ---------------------------------------------------------------------------
# Refusals and warnings name the caller's axis letter
# ---------------------------------------------------------------------------


def test_an_unknown_category_on_the_callers_x_warns_naming_x_and_is_not_drawn():
    """The caller's x is the category axis (drawn vertically), on both output kinds."""
    plain = _category_twin().flipped()
    chart = plain + ann.text("zz", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_unknown_message("x")]
    assert svg == plain.to_svg()
    assert _render_capturing(chart.interactive)[1] == [_unknown_message("x")]
    assert [n for n in _scene_annotations(chart) if n["type"] == "text"] == []


def test_an_unknown_span_end_names_the_callers_axis_letter():
    span = ann.span("x", "b", "zz", fill=_COLOUR["span"])
    chart = _category_twin().flipped() + span
    assert _render_capturing(chart.to_svg)[1] == [_unknown_message("x")]
    assert _render_capturing(chart.interactive)[1] == [_unknown_message("x")]


def test_an_unknown_callout_text_position_names_the_callers_axis_letter():
    callout = ann.callout(
        "b", 3.0, "note", text_x="zz", text_y=2.0, border_color=_COLOUR["callout"]
    )
    chart = _category_twin().flipped() + callout
    assert _render_capturing(chart.to_svg)[1] == [_unknown_message("x")]
    assert _render_capturing(chart.interactive)[1] == [_unknown_message("x")]


def test_a_category_on_the_callers_continuous_y_is_refused_naming_y_on_both_routes():
    """``"q"`` is the caller's y, drawn on the continuous horizontal axis, and ``"b"`` (the
    caller's x) is a valid category."""
    chart = _category_twin().flipped() + ann.text("b", "q", "t")
    _assert_refused_on_both_routes(chart, _continuous_message("q", "y"))


def test_a_span_over_categories_on_the_callers_y_is_refused_naming_y():
    """The caller's y is continuous, so a category span on it is refused."""
    chart = _category_twin().flipped() + ann.span("y", "b", "c", fill="#eee")
    _assert_refused_on_both_routes(chart, _continuous_message("b", "y"))


@pytest.mark.parametrize(
    "start,end,category_end",
    [("b", fm.NormCoord(0.5), "start"), (fm.PixelCoord(40.0), "b", "end")],
    ids=["category-then-position", "position-then-category"],
)
def test_a_mixed_span_on_the_callers_category_axis_is_refused_naming_x(start, end, category_end):
    """The caller's x is the category axis, so the refusal is the mixed-span text naming x."""
    chart = _category_twin().flipped() + ann.span("x", start, end, fill="#eee")
    _assert_refused_on_both_routes(chart, _mixed_span_message("x", category_end, "b"))


@pytest.mark.parametrize(
    "start,end",
    [("b", 2.5), (2.5, "b")],
    ids=["category-then-number", "number-then-category"],
)
def test_a_span_number_end_on_the_callers_category_axis_is_refused_as_a_value_naming_x(start, end):
    chart = _category_twin().flipped() + ann.span("x", start, end, fill="#eee")
    _assert_refused_on_both_routes(chart, _value_message("2.5", "x"))


def test_a_category_on_the_callers_numeric_axes_is_refused_naming_the_callers_letter():
    twin = _numeric_twin()
    _assert_refused_on_both_routes(
        twin.flipped() + ann.text("b", 3.0, "t"), _continuous_message("b", "x")
    )
    _assert_refused_on_both_routes(
        twin.flipped() + ann.text(3.0, "b", "t"), _continuous_message("b", "y")
    )
    _assert_refused_on_both_routes(
        twin.flipped() + ann.span("y", "b", "c", fill="#eee"), _continuous_message("b", "y")
    )


@pytest.mark.parametrize(
    "remedy,offset",
    [(fm.NormCoord(0.5), lambda width: 0.5 * width), (fm.PixelCoord(40.0), lambda width: 40.0)],
    ids=["norm", "pixel"],
)
def test_the_remedy_the_refusal_names_renders_under_the_flip(remedy, offset):
    """The refusal tells the caller to pass a number, ``NormCoord`` or ``PixelCoord`` for the
    refused channel; each lands on the same chart under the flip: band ``b`` on the vertical
    axis, the fraction or pixel offset from the left on the horizontal one."""
    chart = _category_twin().flipped() + ann.text("b", remedy, "HELLO", anchor="middle")
    svg = chart.to_svg()
    left, _, width, _ = _frame(_category_twin().flipped())
    x, y = _text_at(svg)
    _near(y, _ticks(svg, "y")["b"])
    _near(x, left + offset(width))
    (node,) = [n for n in _scene_annotations(chart) if n["type"] == "text"]
    _near([node["x"], node["y"]], [x, y])


# ---------------------------------------------------------------------------
# The same placement through every door, static and interactive
# ---------------------------------------------------------------------------


def _bars(flip: bool) -> fm.Chart:
    return _category_twin().flipped() if flip else _category_twin().native()


def _point_layer(flip: bool) -> fm.Chart:
    point = fm.Chart(_DF).mark_point()
    return point.encode(x="c", y="v").coord(_FLIP) if flip else point.encode(x="v", y="c")


def _other_frame(flip: bool) -> fm.Chart:
    point = fm.Chart(_OTHER).mark_point()
    return point.encode(x="c", y="v").coord(_FLIP) if flip else point.encode(x="v", y="c")


def _host(flip: bool) -> fm.Chart:
    host = fm.Chart(_DF)
    return host.encode(x="c", y="v").coord(_FLIP) if flip else host.encode(x="v", y="c")


def _faceted(flip: bool) -> fm.Chart:
    bars = fm.Chart(_FACETED).mark_bar()
    chart = bars.encode(x="c", y="v").coord(_FLIP) if flip else bars.encode(x="v", y="c")
    return chart.facet("f")


_DOORS: dict[str, Callable[[bool, Any], Any]] = {
    "plus": lambda flip, a: _bars(flip) + a,
    "chart_layer": lambda flip, a: _host(flip).layer(fm.Layer(mark="bar")) + a,
    "layer_chart": lambda flip, a: fm.LayerChart(_bars(flip), _point_layer(flip) + a),
    "operand_with_its_own_frame": lambda flip, a: _other_frame(flip) + _bars(flip) + a,
    "hconcat_leaf": lambda flip, a: (_bars(flip) + a) | _bars(flip),
    "facet_panel_zero": lambda flip, a: _faceted(flip) + a,
}


def _door_pair(door: str, kind: str) -> tuple[Any, Any]:
    points = _category_twin().points
    return (
        _DOORS[door](True, _BUILDERS[kind](False, points)),
        _DOORS[door](False, _BUILDERS[kind](True, points)),
    )


@pytest.mark.parametrize("kind", ["text", "span_x", "rect", "callout"])
@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_flipped_annotation_draws_the_twins_picture_through_every_door(door, kind):
    """The same annotation through each door draws the twin's picture."""
    flipped, native = _door_pair(door, kind)
    assert flipped.to_svg() == native.to_svg()


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_flipped_text_lands_on_its_band_through_every_door(door):
    chart = _DOORS[door](True, ann.text("b", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"))
    svg = chart.to_svg()
    _near(_text_at(svg)[1], _ticks(svg, "y")["b"])


@pytest.mark.parametrize("kind", ["text", "span_x"])
@pytest.mark.parametrize("door", sorted(_DOORS))
def test_the_interactive_route_agrees_with_the_static_twin_through_every_door(door, kind):
    flipped, native = _door_pair(door, kind)
    interactive = flipped.interactive()
    assert interactive.to_html()
    scene = json.loads(interactive.scene_json)["panels"][0]["annotations"]
    twin_scene = json.loads(native.interactive().scene_json)["panels"][0]["annotations"]
    assert scene
    assert scene == twin_scene


# ---------------------------------------------------------------------------
# A break axis clips the annotation with the marks
# ---------------------------------------------------------------------------

# The break on the drawn horizontal axis: the caller's y under a flip, the x of the unflipped twin.
_BREAK_FLIPPED = fm.BreakAxis(axis="y", gap=(4.5, 5.5))
_BREAK_NATIVE = fm.BreakAxis(axis="x", gap=(4.5, 5.5))


def test_a_text_past_a_break_lands_on_the_mark_drawn_for_that_row():
    """The break names the caller's y, drawn horizontally. A text at the
    caller's ``(5.0, 8.0)`` sits on the point drawn for that row, past the gap."""
    chart = _numeric_twin().flipped() + _BREAK_FLIPPED
    svg = (chart + ann.text(5.0, 8.0, "HELLO", anchor="middle")).to_svg()
    x, y = _text_at(svg)
    assert any(abs(x - cx) < 0.01 and abs(y - cy) < 0.01 for cx, cy in _circles(svg))
    assert x > 500.0


@pytest.mark.parametrize("kind", ["text", "arrow", "line", "rect", "span_y", "bracket"])
def test_an_annotation_over_a_break_is_the_twins_picture(kind):
    """The unflipped twin carries the break on the same drawn axis (its x); the annotation clips
    with the marks there. The warnings (none, or the suppressed bracket) are the twin's, in
    full, on both routes."""
    points = _Points(2.0, 3.0, 4.0, 7.0)
    twin = _numeric_twin()
    flipped = twin.flipped() + _BREAK_FLIPPED + _BUILDERS[kind](False, points)
    native = twin.native() + _BREAK_NATIVE + _BUILDERS[kind](True, points)
    flipped_svg, flipped_shown = _render_capturing(flipped.to_svg)
    native_svg, native_shown = _render_capturing(native.to_svg)
    assert flipped_svg == native_svg
    expected = (
        [_break_message("bracket", "spans a BreakAxis gap", "y")] if kind == "bracket" else []
    )
    twin_expected = (
        [_break_message("bracket", "spans a BreakAxis gap", "x")] if kind == "bracket" else []
    )
    assert flipped_shown == expected
    assert native_shown == twin_expected
    _, flipped_shown_interactive = _render_capturing(flipped.interactive)
    assert flipped_shown_interactive == expected
    assert _scene_annotations(flipped) == _scene_annotations(native)


# ---------------------------------------------------------------------------
# Unchanged: a figure-applied flip and no flip draw the bytes they drew before
#
# Each digest is the output of a figure-applied flip, or of no flip: neither moves.
# ---------------------------------------------------------------------------


def _catplot_h() -> fm.Chart:
    return fm.catplot(_DF, x="v", y="c", kind="bar", orient="h")


_FIGURE_FLIP = {
    "text_category_on_y": lambda: _catplot_h() + ann.text(2.0, "b", "HELLO", anchor="middle"),
    "text_second_category": lambda: _catplot_h() + ann.text(2.0, "c", "HELLO", anchor="middle"),
    "span_y_categories": lambda: _catplot_h() + ann.span("y", "b", "c", fill=_COLOUR["span"]),
    "span_x_numbers": lambda: _catplot_h() + ann.span("x", 1.0, 2.0, fill=_COLOUR["span"]),
    "rect": lambda: _catplot_h() + ann.rect(1.0, "a", 3.0, "b", fill=_COLOUR["rect"]),
    "arrow": lambda: _catplot_h() + ann.arrow(1.0, "a", 3.0, "c", stroke=_COLOUR["arrow"]),
    "norm_and_pixel": lambda: (
        _catplot_h() + ann.text(fm.NormCoord(0.1), fm.PixelCoord(40.0), "HELLO")
    ),
    "hline_category": lambda: _catplot_h() + fm.annotate_hline("b", stroke="#ff0010"),
    "vline_number": lambda: _catplot_h() + fm.annotate_vline(2.0, stroke="#ff0010"),
}

_NO_FLIP = {
    "text_category": lambda: _bars(False) + ann.text(2.0, "b", "HELLO", anchor="middle"),
    "text_number": lambda: _numeric_twin().native() + ann.text(2.0, 3.0, "HELLO"),
    "span_y_categories": lambda: _bars(False) + ann.span("y", "b", "c", fill=_COLOUR["span"]),
    "span_x_numbers": lambda: (
        _numeric_twin().native() + ann.span("x", 2.0, 4.0, fill=_COLOUR["span"])
    ),
    "rect": lambda: _numeric_twin().native() + ann.rect(2.0, 3.0, 4.0, 7.0, fill=_COLOUR["rect"]),
    "callout": lambda: (
        _numeric_twin().native()
        + ann.callout(2.0, 3.0, "note", text_x=4.0, text_y=7.0, border_color=_COLOUR["callout"])
    ),
    "bracket": lambda: (
        _numeric_twin().native()
        + ann.bracket(2.0, 3.0, 4.0, 7.0, label="range", stroke=_COLOUR["bracket"])
    ),
    "norm_and_pixel": lambda: (
        _numeric_twin().native() + ann.text(fm.NormCoord(0.1), fm.PixelCoord(40.0), "HELLO")
    ),
    "hline": lambda: _numeric_twin().native() + fm.annotate_hline(5.0, stroke="#ff0010"),
    "abline": lambda: _numeric_twin().native() + fm.annotate_abline(2.0, 1.0, stroke="#ff0012"),
}

_FIGURE_FLIP_DIGESTS: dict[str, str] = {
    "arrow": "0c998162fe1d83b4f7035b4d4c52bfddbf7f41d7ea52f5f72ffb96a52e9068bd",
    "hline_category": "198fdaa45b2f8a6429be194c81b558f3591898903692c663fdb8fbc7bdaeb223",
    "norm_and_pixel": "20fbd902d4c7c07013fb0a9327a11e1ab6c0bbef4491222d11c2db67c07e0d12",
    "rect": "d7c28e2fafd9ff27f4e9f09077a129173b1e6b18dbb30ff0321d0502dcd6ac0c",
    "span_x_numbers": "4283f6b201c2538d961c7b26f1fe405b19d329c08ae41a2fa6d4230581601d9b",
    "span_y_categories": "ab20bf8567da928552fab4ab3f558d63598bdd894387649ebda37c6439b75900",
    "text_category_on_y": "94807e77cb1c2e0678af5367cbcc8c5c4f83ac3d94e8972d64ba0a9d1dd0946a",
    "text_second_category": "e4cef91342ccfec92dddb59d5ed45c5ea202b30f2ba9ff4830fe319408a6fbc2",
    "vline_number": "dc169ea201550db0b8a0733171d0913754abef7581590f972db18beba3d5bf15",
}
_NO_FLIP_DIGESTS: dict[str, str] = {
    "abline": "0286023814c11f69c650289b57b0e0dd03466be3ab85be086d017fbf91a676c1",
    "bracket": "5bf0e565a6344966e8626b37bd04aaea5021b47c1d718a8f0f11a9e1a5a88aca",
    "callout": "3da31f607e69540b344adb32876e91573c4d26d8737144867d3239174dc222d3",
    "hline": "77c822a873c7174355083e8a1bb3a4e1479ddc58445b03ee5817af4650077b94",
    "norm_and_pixel": "2493602171140a14b6b4fa168c43f0839e4053fc2afc1ff6cd83353315e76bb2",
    "rect": "d2740fc78bb402b15483480745c9fc4fb2b8279c863d1a258d83b3d263779547",
    "span_x_numbers": "3e45faac8b9de0910b620c47729e6f8f10ff51166c331c143b6e8382bfcff6e1",
    "span_y_categories": "ab20bf8567da928552fab4ab3f558d63598bdd894387649ebda37c6439b75900",
    "text_category": "94807e77cb1c2e0678af5367cbcc8c5c4f83ac3d94e8972d64ba0a9d1dd0946a",
    "text_number": "73cd36c6f471887c2027d613f9f54cc1cf6832affa522733dd3cfbe5e516868a",
}


@pytest.mark.parametrize("name", sorted(_FIGURE_FLIP))
def test_a_figure_applied_flip_swaps_nothing(name):
    assert _digest(_FIGURE_FLIP[name]().to_svg()) == _FIGURE_FLIP_DIGESTS[name]


@pytest.mark.parametrize("name", sorted(set(_NO_FLIP) - {"callout"}))
def test_output_with_no_flip_draws_the_bytes_it_drew_before(name):
    assert _digest(_NO_FLIP[name]().to_svg()) == _NO_FLIP_DIGESTS[name]


def test_a_callout_with_no_flip_draws_its_box_and_text_in_the_theme_colours():
    """Only the box fill (theme background) and the text fill (theme text colour) differ from
    the bytes the callout drew before it followed the theme; the border colour is the caller's."""
    assert _digest(_NO_FLIP["callout"]().to_svg()) == _NO_FLIP_DIGESTS["callout"]


def test_the_figure_applied_flip_is_not_a_caller_flip():
    """On a figure-applied flip the annotation's y is still the drawn y: a category written as
    the y coordinate lands on its band on the vertical axis, with no exchange."""
    svg = (_catplot_h() + ann.text(2.0, "b", "HELLO", anchor="middle")).to_svg()
    _near(_text_at(svg)[1], _ticks(svg, "y")["b"])
    _near(_text_at(svg)[0], _tick_at(_ticks(svg, "x"), 2.0))


# ---------------------------------------------------------------------------
# A bracket's direction exchanges with its pair
# ---------------------------------------------------------------------------

_BRACKET_DIRECTIONS = [None, "above", "below", "up", "down", "left", "right"]
# The tip vector a drawn direction gives (tip length 6) and the label offset (tip length + 4).
_DRAWN_TIP = {"up": (0.0, -6.0), "down": (0.0, 6.0), "left": (-6.0, 0.0), "right": (6.0, 0.0)}
# A bracket whose baseline runs along the caller's x, so along the vertical axis on the page.
_BRACKET_POINTS = _Points(2.0, 6.0, 4.0, 6.0)


def _plain_bracket(direction: str) -> fm.Chart:
    return _numeric_twin().native() + ann.bracket(
        2.0, 3.0, 4.0, 3.0, label="range", direction=direction, stroke=_COLOUR["bracket"]
    )


def test_an_unflipped_below_bracket_draws_like_down_and_above_like_up():
    """ "below" is the synonym of "down" and "above" of "up", on both output kinds; "below" is not
    the default's picture."""
    for synonym, direction in (("below", "down"), ("above", "up")):
        assert _plain_bracket(synonym).to_svg() == _plain_bracket(direction).to_svg()
        assert _scene_annotations(_plain_bracket(synonym)) == _scene_annotations(
            _plain_bracket(direction)
        )
    assert _plain_bracket("below").to_svg() != _plain_bracket("above").to_svg()
    assert _plain_bracket("below").to_svg() != _plain_bracket("up").to_svg()
    assert _scene_annotations(_plain_bracket("below")) != _scene_annotations(
        _plain_bracket("above")
    )


@pytest.mark.parametrize("direction", _BRACKET_DIRECTIONS, ids=lambda d: d or "default")
@pytest.mark.parametrize("name", ["numeric", "category"])
def test_a_bracket_under_a_flip_is_the_twin_with_its_direction_exchanged(name, direction):
    twin = _TWINS[name]()
    points = _BRACKET_POINTS if name == "numeric" else _Points("b", 1.0, "c", 1.0)
    flipped = twin.flipped() + _bracket(False, points, direction)
    native = twin.native() + _bracket(True, points, direction)
    assert flipped.to_svg() == native.to_svg()
    assert _scene_annotations(flipped) == _scene_annotations(native)


@pytest.mark.parametrize("direction", _BRACKET_DIRECTIONS, ids=lambda d: d or "default")
def test_a_bracket_under_a_flip_draws_its_tips_and_label_where_the_exchanged_direction_says(
    direction,
):
    """Read off the page: the baseline runs along the vertical axis, each tip leaves its baseline
    end by the exchanged direction's vector and the label sits one tip length plus 4 beyond the
    baseline's middle. The default ("above") and every up/down value give tips off the baseline,
    on the page's horizontal; left and right give tips along it."""
    chart = _numeric_twin().flipped() + _bracket(False, _BRACKET_POINTS, direction)
    svg = chart.to_svg()
    base, tip_a, tip_b = _all(svg, "line", stroke=_COLOUR["bracket"])
    (bx1, by1, bx2, by2) = (float(base.attrs[k]) for k in ("x1", "y1", "x2", "y2"))
    assert bx1 == pytest.approx(bx2, abs=0.01)
    assert by1 != pytest.approx(by2, abs=1.0)
    dx, dy = _DRAWN_TIP[_EXCHANGED_DIRECTION[direction or "above"]]
    for tip, (ex, ey) in ((tip_a, (bx1, by1)), (tip_b, (bx2, by2))):
        _near(
            [float(tip.attrs[k]) for k in ("x1", "y1", "x2", "y2")],
            [ex, ey, ex + dx, ey + dy],
        )
    label_x, label_y = _text_at(svg, "range")
    _near(
        [label_x, label_y],
        [(bx1 + bx2) / 2 + dx * 10.0 / 6.0, (by1 + by2) / 2 + dy * 10.0 / 6.0],
    )
    if direction in (None, "above", "below", "up", "down"):
        assert dy == 0.0 and dx != 0.0


# ---------------------------------------------------------------------------
# Dates, ISO strings, temporal axes and zero-row frames under a flip
# ---------------------------------------------------------------------------


def test_an_iso_string_coordinate_draws_what_the_date_draws_under_a_flip():
    date = _temporal_twin(dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    iso = _temporal_twin("2020-01-02", "2020-01-03")
    for kind in _KINDS:
        by_date = (date.flipped() + _BUILDERS[kind](False, date.points)).to_svg()
        by_iso = (iso.flipped() + _BUILDERS[kind](False, iso.points)).to_svg()
        assert by_date == by_iso, kind
        assert _COLOUR.get(kind.split("_")[0], _IMAGE_SRC) in by_date


def test_a_date_on_the_callers_x_lands_on_the_temporal_axis_the_flip_moved_to_the_vertical():
    """The caller's x is the date axis, drawn vertically: Jan 2 sits half way between Jan 1 and
    Jan 3, and where the unflipped twin puts it."""
    twin = _temporal_twin(dt.date(2020, 1, 2), dt.date(2020, 1, 3))

    def drawn(chart: fm.Chart, day: int, *, swap: bool) -> tuple[float, float]:
        date = dt.date(2020, 1, day)
        text = ann.text(*_pair(swap, date, 2.0), "HELLO", anchor="middle")
        return _text_at((chart + text).to_svg())

    first, second, third = (drawn(twin.flipped(), day, swap=False) for day in (1, 2, 3))
    _near(second[1], (first[1] + third[1]) / 2)
    assert first[1] != pytest.approx(third[1], abs=1.0)
    _near(first[0], second[0])
    _near(list(drawn(twin.native(), 2, swap=True)), list(second))


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda: _temporal_twin(0, 0).flipped() + ann.text("b", 1.5, "t"), ("b", "x")),
        (
            lambda: _temporal_twin(0, 0).flipped() + ann.text(dt.date(2020, 1, 2), "b", "t"),
            ("b", "y"),
        ),
        (
            lambda: _temporal_twin(0, 0).flipped() + ann.span("x", "b", "c", fill="#eee"),
            ("b", "x"),
        ),
    ],
    ids=["category-on-callers-date-x", "category-on-callers-numeric-y", "category-span-on-date-x"],
)
def test_a_category_on_a_flipped_temporal_chart_is_refused_naming_the_callers_letter(
    build, expected
):
    _assert_refused_on_both_routes(build(), _continuous_message(*expected))


def test_a_category_on_the_callers_temporal_y_is_refused_naming_y():
    chart = fm.Chart(_TEMPORAL).mark_line().encode(x="v", y="date").coord(_FLIP) + ann.text(
        1.5, "b", "t"
    )
    _assert_refused_on_both_routes(chart, _continuous_message("b", "y"))


_EMPTY = pl.DataFrame({"c": [], "v": []}, schema={"c": pl.Utf8, "v": pl.Float64})


def test_a_zero_row_frame_under_a_flip_warns_once_per_cause_in_the_callers_letter():
    """The empty frame draws its frame and warns it has no rows; the category the empty axis
    cannot place is named as the caller's x, with its empty domain."""
    flipped = fm.Chart(_EMPTY).mark_bar().encode(x="c", y="v").coord(_FLIP)
    native = fm.Chart(_EMPTY).mark_bar().encode(x="v", y="c")
    chart = flipped + ann.text("b", 2.0, "HELLO")
    twin = native + ann.text(2.0, "b", "HELLO")
    unknown = _unknown_message("x", "b").replace(str(sorted("abcd")), "[]")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_NO_ROWS_TO_DRAW, unknown]
    assert _render_capturing(chart.interactive)[1] == [_NO_ROWS_TO_DRAW, unknown]
    twin_svg, twin_shown = _render_capturing(twin.to_svg)
    assert twin_shown == [_NO_ROWS_TO_DRAW, unknown.replace("x-axis", "y-axis")]
    assert svg == twin_svg


# ---------------------------------------------------------------------------
# Every door, both output kinds: warnings, refusals and placements
# ---------------------------------------------------------------------------

# A refusal inside a composite leaf is prefixed by the leaf that failed; the prefix differs by
# route for a LayerChart (the static route renders its overlay leaves, the interactive route the
# merged panel).
_STATIC_PREFIX = {
    "layer_chart": "failed to render composite leaf leaf #1: ",
    "hconcat_leaf": "failed to render composite leaf leaf #0: ",
}
_INTERACTIVE_PREFIX = {"hconcat_leaf": "failed to render composite leaf leaf #0: "}


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_an_unknown_category_warns_naming_the_callers_x_through_every_door_on_both_routes(door):
    chart = _DOORS[door](True, ann.text("zz", 2.0, "t"))
    assert _render_capturing(chart.to_svg)[1] == [_unknown_message("x")]
    assert _render_capturing(chart.interactive)[1] == [_unknown_message("x")]


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_category_on_the_callers_y_is_refused_through_every_door_on_both_routes(door):
    chart = _DOORS[door](True, ann.text("b", "q", "t"))
    message = _continuous_message("q", "y")
    with pytest.raises(ValueError) as static:
        chart.to_svg()
    assert str(static.value) == _STATIC_PREFIX.get(door, "") + message
    with pytest.raises(ValueError) as interactive:
        chart.interactive().to_html()
    assert str(interactive.value) == _INTERACTIVE_PREFIX.get(door, "") + message


@pytest.mark.parametrize(
    "door", ["plus", "chart_layer", "layer_chart", "operand_with_its_own_frame"]
)
def test_the_interactive_text_node_sits_where_the_static_text_does_through_every_door(door):
    chart = _DOORS[door](True, ann.text("b", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"))
    x, y = _text_at(chart.to_svg())
    (node,) = [n for n in _scene_annotations(chart) if n["type"] == "text"]
    _near([node["x"], node["y"]], [x, y])


# ---------------------------------------------------------------------------
# Break-axis warnings name the caller's channel
# ---------------------------------------------------------------------------


def _break_message(kind: str, what: str, axis: str) -> str:
    """The warning for an annotation a break removes. It names the caller's channel the
    ``BreakAxis(axis=...)`` names: ``y`` under a flip (drawn horizontally), ``x`` on the twin."""
    return f"1 annotation element ({kind}) on the {axis} axis {what} and was suppressed"


def test_a_text_inside_a_break_gap_warns_with_the_callers_channel_letter_on_both_routes():
    """The caller's y 5.0 lies in the gap (4.5, 5.5) of ``BreakAxis(axis="y")``, the caller's
    channel drawn horizontally. The warning says "y"; the unflipped twin, whose break names its
    x, says "x"."""
    expected = [_break_message("text", "fell inside a BreakAxis gap", "y")]
    twin_expected = [_break_message("text", "fell inside a BreakAxis gap", "x")]
    flipped = _numeric_twin().flipped() + _BREAK_FLIPPED + ann.text(2.0, 5.0, "HELLO")
    twin = _numeric_twin().native() + _BREAK_NATIVE + ann.text(5.0, 2.0, "HELLO")
    assert _render_capturing(flipped.to_svg)[1] == expected
    assert _render_capturing(flipped.interactive)[1] == expected
    assert _render_capturing(twin.to_svg)[1] == twin_expected


def test_a_number_on_a_figure_applied_flips_category_axis_is_refused_naming_y():
    _assert_refused_on_both_routes(_catplot_h() + ann.text(2.0, 3.0, "t"), _value_message("3", "y"))
