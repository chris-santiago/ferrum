"""A categorical annotation coordinate lands where the axis places its category.

Every annotation kind resolves a category through the axis's own scale, so it lands on the band
the axis draws for it whatever the sort, domain, order or padding, never by data order. A span
over categories shades whole bands (``span("x", "b", "b")`` shades the whole ``b`` slot, from the
band's outer edge to its outer edge). A category on a continuous axis, or under ``CoordPolar`` or
``CoordGeo``, is refused with a remedy, and a span with one category end and one value end is
refused.

Positions are read off the rendered SVG and compared with the axis's own tick positions (a
category tick is its band's centre) and with the band edges those ticks and the band padding
imply. The fixtures use a value column whose ``sort="-y"`` order differs from the data order for
both ``b`` and ``c``, so a placement by data order cannot pass.

The one rule has no per-door code: each door in :data:`_DOORS` runs the same placement check.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Callable, NamedTuple

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from ferrum.annotation.coords import OrdinalCategoryCoord
from tests.test_null_group_warning import (
    _NO_ROWS_TO_DRAW,
    _all_keys_null_message,
    _render_capturing,
)

_CATS = ["a", "b", "c", "d"]
# ``sort="-y"`` orders the bands c, d, a, b: ``b`` and ``c`` each move off their data-order band.
_DF = pl.DataFrame({"c": _CATS, "v": [2.0, 1.0, 4.0, 3.0]})
_OTHER = pl.DataFrame({"c": ["a"], "v": [0.5]})
_FACETED = pl.concat(
    [_DF.with_columns(pl.lit("F1").alias("f")), _DF.with_columns(pl.lit("F2").alias("f"))]
)
_NUMERIC = pl.DataFrame({"n": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
_TEMPORAL = pl.DataFrame(
    {
        "date": pl.date_range(dt.date(2020, 1, 1), dt.date(2020, 1, 3), "1d", eager=True),
        "v": [1.0, 2.0, 3.0],
    }
)

_AXIS_GREY = "#6b7280"
_BAR_FILL = "#2563eb"
# One stroke or fill per annotation kind, so each is picked out of the SVG by its own colour.
_COLOUR = {
    "text": "#ff0001",
    "arrow": "#ff0002",
    "rect": "#ff0003",
    "line": "#ff0004",
    "span": "#ff0005",
    "bracket": "#ff0006",
    "callout": "#ff0007",
}
_IMAGE_SRC = "data:image/png;base64,iVBORw0KGgo="


# ---------------------------------------------------------------------------
# SVG readers
# ---------------------------------------------------------------------------


class _El(NamedTuple):
    tag: str
    attrs: dict[str, str]
    text: str


def _elements(svg: str) -> list[_El]:
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.S)
    found = []
    for m in re.finditer(r"<(rect|line|image|text)\b([^>]*?)/?>(?:([^<]*)</text>)?", body):
        found.append(
            _El(m.group(1), dict(re.findall(r'([\w:-]+)="([^"]*)"', m.group(2))), m.group(3) or "")
        )
    return found


def _ticks(svg: str, axis: str) -> dict[str, float]:
    """Label to tick position on ``axis``, first panel only: a tick is the short axis-grey line
    that precedes its label."""
    out: dict[str, float] = {}
    els = _elements(svg)
    for prev, cur in zip(els, els[1:]):
        if prev.tag != "line" or cur.tag != "text" or prev.attrs.get("stroke") != _AXIS_GREY:
            continue
        x1, y1, x2, y2 = (float(prev.attrs[k]) for k in ("x1", "y1", "x2", "y2"))
        anchor = cur.attrs.get("text-anchor")
        if axis == "x" and x1 == x2 and anchor == "middle":
            out.setdefault(cur.text, x1)
        if axis == "y" and y1 == y2 and anchor == "end":
            out.setdefault(cur.text, y1)
    return out


def _band_edges(ticks: dict[str, float], label: str, inner: float) -> tuple[float, float]:
    """The outer edges of ``label``'s band: its tick is the band's centre and the band is
    ``step * (1 - inner)`` wide, ``step`` being the tick spacing."""
    centres = [ticks[c] for c in _CATS]
    step = (max(centres) - min(centres)) / (len(centres) - 1)
    half = step * (1.0 - inner) / 2.0
    return ticks[label] - half, ticks[label] + half


def _near(actual: list[float] | float, expected: list[float] | float) -> None:
    assert actual == pytest.approx(expected, abs=0.01)


def _one(svg: str, tag: str, **attrs: str) -> _El:
    hits = [e for e in _elements(svg) if e.tag == tag and attrs.items() <= e.attrs.items()]
    assert len(hits) == 1, (tag, attrs, hits)
    return hits[0]


def _all(svg: str, tag: str, **attrs: str) -> list[_El]:
    return [e for e in _elements(svg) if e.tag == tag and attrs.items() <= e.attrs.items()]


def _lo_hi(el: _El, axis: str) -> tuple[float, float]:
    lo = float(el.attrs[axis])
    return lo, lo + float(el.attrs["width" if axis == "x" else "height"])


# ---------------------------------------------------------------------------
# Fixtures: a category axis on x or on y, under each axis setting
# ---------------------------------------------------------------------------

# variant -> the inner padding the variant sets on its band scale
_VARIANTS = {
    "default": 0.0,
    "sort": 0.0,
    "domain": 0.0,
    "padding": 0.3,
    "range_inverted": 0.0,
    "reverse": 0.0,
}
# A point scale's slot is its step, so a span over its categories covers centre +/- step / 2
# (inner padding 0), the same oracle a band scale with no padding uses.
_BAND_VARIANTS = ["default", "sort", "domain", "padding", "range_inverted"]
_ALL_VARIANTS = list(_VARIANTS)
_SPAN_VARIANTS = _ALL_VARIANTS


def _channel(axis: str, variant: str) -> Any:
    cls = fm.X if axis == "x" else fm.Y
    return {
        "default": cls("c"),
        "sort": cls("c", sort="-y" if axis == "x" else "-x"),
        "domain": cls("c", scale=fm.BandScale(domain=["d", "c", "b", "a"])),
        "padding": cls("c", scale=fm.BandScale(padding=0.3)),
        "range_inverted": cls("c", scale=fm.BandScale(range=[560.0, 60.0])),
        "reverse": cls("c", scale=fm.PointScale(reverse=True)),
    }[variant]


def _bars(axis: str, variant: str) -> fm.Chart:
    channel = _channel(axis, variant)
    if axis == "x":
        return fm.Chart(_DF).mark_bar().encode(x=channel, y="v")
    return fm.Chart(_DF).mark_bar().encode(x="v", y=channel)


def _at(axis: str, cat: Any, val: Any) -> tuple[Any, Any]:
    """The ``(x, y)`` pair with ``cat`` on the category axis."""
    return (cat, val) if axis == "x" else (val, cat)


def _primitive(kind: str, axis: str) -> Any:
    colour = _COLOUR.get(kind, "")
    if kind == "text":
        return ann.text(*_at(axis, "b", 2.0), "HELLO", color=colour, anchor="middle")
    if kind == "arrow":
        return ann.arrow(*_at(axis, "b", 1.0), *_at(axis, "c", 3.0), stroke=colour)
    if kind == "rect":
        # ``a`` and ``b`` sit at the two ends of a reversed or re-ordered axis's lower half, so
        # no axis setting hands this pair back to a placement by data order.
        return ann.rect(*_at(axis, "a", 1.0), *_at(axis, "b", 3.0), fill=colour)
    if kind == "line":
        return ann.line(*_at(axis, "b", 1.0), *_at(axis, "c", 3.0), stroke=colour)
    if kind == "bracket":
        return ann.bracket(*_at(axis, "b", 3.0), *_at(axis, "c", 3.0), label="range", stroke=colour)
    if kind == "callout":
        x, y = _at(axis, "b", 3.0)
        text_x, text_y = _at(axis, "c", 2.0)
        return ann.callout(x, y, "note", text_x=text_x, text_y=text_y, border_color=colour)
    if kind == "image":
        return ann.image(*_at(axis, "b", 3.0), _IMAGE_SRC, width=10, height=10)
    raise AssertionError(kind)


def _measured(kind: str, svg: str, axis: str) -> list[float]:
    """Where the annotation sits along the category axis."""
    lo, hi = ("x1", "x2") if axis == "x" else ("y1", "y2")
    colour = _COLOUR.get(kind, "")
    if kind == "text":
        return [float(_one(svg, "text", fill=colour).attrs[axis])]
    if kind in ("arrow", "line"):
        line = _one(svg, "line", stroke=colour)
        return [float(line.attrs[lo]), float(line.attrs[hi])]
    if kind == "bracket":
        line = _all(svg, "line", stroke=colour)[0]
        return [float(line.attrs[lo]), float(line.attrs[hi])]
    if kind == "rect":
        return sorted(_lo_hi(_one(svg, "rect", fill=colour), axis))
    if kind == "callout":
        line = _one(svg, "line", stroke=colour)
        note = [e for e in _elements(svg) if e.tag == "text" and e.text == "note"][0]
        return [float(line.attrs[lo]), float(line.attrs[hi]), float(note.attrs[axis])]
    if kind == "image":
        image = _one(svg, "image", href=_IMAGE_SRC)
        start, end = _lo_hi(image, axis)
        return [(start + end) / 2.0]
    raise AssertionError(kind)


def _expected(kind: str, ticks: dict[str, float]) -> list[float]:
    b, c = ticks["b"], ticks["c"]
    if kind in ("text", "image"):
        return [b]
    if kind == "rect":
        return sorted([ticks["a"], b])
    if kind == "callout":
        return [b, c, c]
    return [b, c]


_POINT_KINDS = ["text", "arrow", "rect", "line", "bracket", "callout", "image"]


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("variant", _ALL_VARIANTS)
@pytest.mark.parametrize("kind", _POINT_KINDS)
def test_a_point_annotation_lands_on_the_band_centre_the_axis_draws(kind, variant, axis):
    svg = (_bars(axis, variant) + _primitive(kind, axis)).to_svg()
    _near(_measured(kind, svg, axis), _expected(kind, _ticks(svg, axis)))


def test_the_placement_check_reads_the_axis_not_data_order():
    """The fixture's sorted axis moves ``b`` and ``c`` off their data-order bands, so a placement
    by data order (``(i + 0.5) / n`` over first appearance) cannot satisfy the check above."""
    svg = _bars("x", "sort").to_svg()
    ticks = _ticks(svg, "x")
    assert ticks["b"] > ticks["a"]
    assert [label for label, _ in sorted(ticks.items(), key=lambda kv: kv[1])] == [
        "c",
        "d",
        "a",
        "b",
    ]


# ---------------------------------------------------------------------------
# A span covers whole bands
# ---------------------------------------------------------------------------

_SPAN_ENDS = [("b", "c"), ("c", "b"), ("b", "b"), ("a", "d")]


def _span_extent(svg: str, axis: str) -> tuple[float, float]:
    pieces = [_lo_hi(e, axis) for e in _all(svg, "rect", fill=_COLOUR["span"])]
    assert pieces
    return min(lo for lo, _ in pieces), max(hi for _, hi in pieces)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("variant", _SPAN_VARIANTS)
@pytest.mark.parametrize("start,end", _SPAN_ENDS)
def test_a_span_over_categories_shades_whole_bands(start, end, variant, axis):
    span = ann.span(axis, start, end, fill=_COLOUR["span"])
    svg = (_bars(axis, variant) + span).to_svg()
    ticks = _ticks(svg, axis)
    inner = _VARIANTS[variant]
    lo_start, hi_start = _band_edges(ticks, start, inner)
    lo_end, hi_end = _band_edges(ticks, end, inner)
    _near(_span_extent(svg, axis), (min(lo_start, lo_end), max(hi_start, hi_end)))


def test_the_band_oracle_matches_the_drawn_bars():
    """A bar fills 0.8 of its band, so the band the span tests compare against is the one the
    marks use: ``bar width / 0.8``, centred on the tick."""
    for variant in _BAND_VARIANTS:
        svg = _bars("x", variant).to_svg()
        ticks = _ticks(svg, "x")
        for bar in _all(svg, "rect", fill=_BAR_FILL):
            lo, hi = _lo_hi(bar, "x")
            centre = (lo + hi) / 2.0
            label = min(ticks, key=lambda t: abs(ticks[t] - centre))
            band_lo, band_hi = _band_edges(ticks, label, _VARIANTS[variant])
            _near(
                [lo, hi], [centre - 0.4 * (band_hi - band_lo), centre + 0.4 * (band_hi - band_lo)]
            )


def test_a_span_over_one_category_is_the_whole_band_not_a_zero_width_rule():
    svg = (_bars("x", "sort") + ann.span("x", "b", "b", fill=_COLOUR["span"])).to_svg()
    lo, hi = _span_extent(svg, "x")
    ticks = _ticks(svg, "x")
    step = abs(ticks["a"] - ticks["b"])
    _near([lo, hi], [ticks["b"] - step / 2, ticks["b"] + step / 2])
    assert hi - lo > 100


# ---------------------------------------------------------------------------
# The helpers and the primitives take the same coordinates
# ---------------------------------------------------------------------------


def test_the_annotate_helpers_land_on_band_centres_under_a_sort():
    chart = _bars("x", "sort")
    ticks = _ticks(chart.to_svg(), "x")
    vline = (chart + fm.annotate_vline("b", stroke="#ff0010")).to_svg()
    line = _one(vline, "line", stroke="#ff0010")
    _near([float(line.attrs["x1"]), float(line.attrs["x2"])], [ticks["b"], ticks["b"]])
    text = (chart + fm.annotate_text("b", 2.0, "HELLO", anchor="middle")).to_svg()
    hello = [e for e in _all(text, "text") if e.text == "HELLO"]
    assert len(hello) == 1
    _near(float(hello[0].attrs["x"]), ticks["b"])
    rect = (chart + fm.annotate_rect("b", "c", 1.0, 3.0, fill="#ff0011")).to_svg()
    _near(sorted(_lo_hi(_one(rect, "rect", fill="#ff0011"), "x")), sorted([ticks["b"], ticks["c"]]))


def test_the_annotate_helpers_land_on_a_vertical_category_axis_too():
    chart = _bars("y", "sort")
    ticks = _ticks(chart.to_svg(), "y")
    hline = (chart + fm.annotate_hline("b", stroke="#ff0012")).to_svg()
    line = _one(hline, "line", stroke="#ff0012")
    _near([float(line.attrs["y1"]), float(line.attrs["y2"])], [ticks["b"], ticks["b"]])
    arrow = (chart + fm.annotate_arrow(1.0, "b", 3.0, "c", stroke="#ff0013")).to_svg()
    shaft = _one(arrow, "line", stroke="#ff0013")
    _near([float(shaft.attrs["y1"]), float(shaft.attrs["y2"])], [ticks["b"], ticks["c"]])


def test_the_primitives_accept_plain_strings_as_categories():
    span = ann.span("x", "b", "c", fill="#eee")
    assert (span.to_dict()["start"], span.to_dict()["end"]) == (
        {"category": "b"},
        {"category": "c"},
    )
    text = ann.text("b", 2.0, "t")
    assert text.to_dict()["x"] == {"category": "b"}
    callout = ann.callout("b", 1.0, "t", text_x="c", text_y="d").to_dict()
    assert (callout["x"], callout["text_x"], callout["text_y"]) == (
        {"category": "b"},
        {"category": "c"},
        {"category": "d"},
    )


def test_an_iso_string_stays_a_temporal_value():
    assert ann.text("2020-01-02", 2.0, "t").to_dict()["x"] == {
        "instant": 1577923200000.0,
        "text": "2020-01-02",
    }
    assert ann.span("x", "2020-01-01", "2020-01-02", fill="#eee").to_dict()["start"] == {
        "instant": 1577836800000.0,
        "text": "2020-01-01",
    }
    chart = fm.Chart(_TEMPORAL).mark_line().encode(x="date", y="v")
    as_string = (chart + ann.text("2020-01-02", 2.0, "t")).to_svg()
    as_date = (chart + ann.text(dt.date(2020, 1, 2), 2.0, "t")).to_svg()
    assert as_string == as_date
    vline = (chart + fm.annotate_vline("2020-01-02")).to_svg()
    assert vline == (chart + fm.annotate_vline(dt.date(2020, 1, 2))).to_svg()


@pytest.mark.parametrize(
    "build",
    [
        lambda: _bars("x", "default") + fm.annotate_text("b", 2.0, "t"),
        lambda: _bars("x", "default") + ann.text("b", 2.0, "t"),
    ],
)
def test_a_category_crosses_to_the_renderer_unresolved(build):
    sent = build()._resolve_chart_config()["annotations"]
    assert [a["x"] for a in sent] == [{"category": "b"}]


def test_a_span_and_a_callout_cross_unresolved():
    chart = _bars("x", "default")
    span = (chart + ann.span("x", "b", "c", fill="#eee"))._resolve_chart_config()["annotations"][0]
    assert (span["start"], span["end"]) == ({"category": "b"}, {"category": "c"})
    callout = chart + ann.callout("b", 1.0, "t", text_x="c", text_y=2.0)
    sent = callout._resolve_chart_config()["annotations"][0]
    assert (sent["x"], sent["text_x"], sent["text_y"]) == (
        {"category": "b"},
        {"category": "c"},
        2.0,
    )


def test_python_keeps_no_category_resolver():
    import ferrum._render_prepare as prepare

    for name in (
        "_resolve_category_coords_in_annotations",
        "_extract_ordinal_domain",
        "_build_ordinal_norm_map",
    ):
        assert not hasattr(prepare, name), name


# ---------------------------------------------------------------------------
# Every door places a category where the axis does
# ---------------------------------------------------------------------------


def _doors_bars(channel: Any) -> fm.Chart:
    return fm.Chart(_DF).mark_bar().encode(x=channel, y="v")


_DOORS: dict[str, Callable[[Any, Any], Any]] = {
    "plus": lambda ch, a: _doors_bars(ch) + a,
    "chart_layer": lambda ch, a: fm.Chart(_DF).encode(x=ch, y="v").layer(fm.Layer(mark="bar")) + a,
    "layer_chart": lambda ch, a: fm.LayerChart(
        _doors_bars(ch), fm.Chart(_DF).mark_point().encode(x=ch, y="v") + a
    ),
    "operand_with_its_own_frame": lambda ch, a: (
        fm.Chart(_OTHER).mark_point().encode(x="c", y="v") + _doors_bars(ch) + a
    ),
    "hconcat_leaf": lambda ch, a: (_doors_bars(ch) + a) | _doors_bars(ch),
    "facet_panel_zero": lambda ch, a: (
        fm.Chart(_FACETED).mark_bar().encode(x=ch, y="v").facet("f") + a
    ),
    "break_axis": lambda ch, a: _doors_bars(ch) + fm.BreakAxis(axis="y", gap=(2.2, 2.8)) + a,
}


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_point_annotation_lands_on_its_band_through_every_door(door):
    svg = _DOORS[door](fm.X("c", sort="-y"), _primitive("text", "x")).to_svg()
    _near(_measured("text", svg, "x"), _expected("text", _ticks(svg, "x")))


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_span_shades_whole_bands_through_every_door(door):
    span = ann.span("x", "b", "c", fill=_COLOUR["span"])
    svg = _DOORS[door](fm.X("c", sort="-y"), span).to_svg()
    ticks = _ticks(svg, "x")
    # The layered bars draw on the merged scale, which pads the plot differently from the
    # single-chart one, so the band is read from this render's own ticks.
    lo_b, hi_b = _band_edges(ticks, "b", 0.0)
    lo_c, hi_c = _band_edges(ticks, "c", 0.0)
    _near(_span_extent(svg, "x"), (min(lo_b, lo_c), max(hi_b, hi_c)))


def _scene_annotations(chart: Any) -> list[dict[str, Any]]:
    return json.loads(chart.interactive().scene_json)["panels"][0]["annotations"]


@pytest.mark.parametrize("variant", ["default", "sort", "domain", "padding"])
def test_the_interactive_route_places_a_category_as_the_static_route_does(variant):
    chart = _bars("x", variant)
    span = ann.span("x", "b", "c", fill=_COLOUR["span"])
    text = _primitive("text", "x")
    static = (chart + span + text).to_svg()
    nodes = _scene_annotations(chart + span + text)
    scene_text = next(n for n in nodes if n["type"] == "text")
    scene_span = next(n for n in nodes if n["type"] == "rect")
    ticks = _ticks(static, "x")
    lo_b, hi_b = _band_edges(ticks, "b", _VARIANTS[variant])
    lo_c, hi_c = _band_edges(ticks, "c", _VARIANTS[variant])
    _near(_measured("text", static, "x"), [ticks["b"]])
    _near(scene_text["x"], ticks["b"])
    _near(_span_extent(static, "x"), (min(lo_b, lo_c), max(hi_b, hi_c)))
    _near([scene_span["x"], scene_span["x"] + scene_span["w"]], _span_extent(static, "x"))


# ---------------------------------------------------------------------------
# Refusals: each message in full, each remedy executed, both output kinds
# ---------------------------------------------------------------------------


def _continuous_message(label: str, axis: str) -> str:
    return (
        f"annotation category '{label}' on the {axis} axis: the {axis} scale is continuous, so it "
        "holds no categories; pass a number, a date, an ISO-8601 string, fm.NormCoord or "
        "fm.PixelCoord"
    )


def _coord_message(label: str, axis: str, coord: str) -> str:
    return (
        f"annotation category '{label}' on the {axis} axis: a category cannot be placed under "
        f"{coord}; pass fm.NormCoord or fm.PixelCoord"
    )


def _mixed_span_message(axis: str, end: str, label: str) -> str:
    """The refusal of a span with one category end and one end that is not: ``end`` is the
    category end the caller wrote (``start`` or ``end``)."""
    other = "end" if end == "start" else "start"
    return (
        f"annotation span on the {axis} axis: {end} '{label}' is a category and {other} is not; "
        "pass categories at both ends, or fm.NormCoord or fm.PixelCoord at both ends"
    )


def _unknown_message(axis: str, label: str = "zz", domain: list[str] | None = None) -> str:
    """The warning for a category the axis does not hold: the annotation is not drawn."""
    listed = sorted(_CATS) if domain is None else domain
    return (
        f"annotation category '{label}' not found in {axis}-axis domain "
        f"{listed}; the annotation is not drawn"
    )


def _value_message(value: str, axis: str, label: str | None = None) -> str:
    """The refusal of a number or date on a categorical axis. ``label`` is the category the value
    spells, when the axis holds one; without it the remedy names a category label generically."""
    clause = f"the category label '{label}'" if label is not None else "a category label"
    return (
        f"annotation value {value} on the {axis} axis: the {axis} scale is categorical, so it "
        f"places categories, not numbers or dates; pass {clause}, fm.NormCoord or fm.PixelCoord"
    )


def _value_under_coord_message(value: str, axis: str, coord: str) -> str:
    """The refusal of a number or date on a categorical axis under a projected coord: the remedy
    names positions only, because a category cannot be placed there either."""
    return (
        f"annotation value {value} on the {axis} axis: the {axis} scale is categorical, so it "
        f"places categories, not numbers or dates, and a category cannot be placed under {coord}; "
        "pass fm.NormCoord or fm.PixelCoord"
    )


def _helper_message(entry: str, axis: str) -> str:
    """The refusal of a helper whose own coordinates meet a categorical axis."""
    return (
        f"{entry}: the {axis} scale is categorical, so it places categories, not numbers or "
        f"dates, and {entry} takes no category; draw it on a chart whose x and y scales are "
        "continuous"
    )


def _assert_refused_on_both_routes(chart: Any, message: str) -> None:
    with pytest.raises(ValueError) as static:
        chart.to_svg()
    assert str(static.value) == message
    with pytest.raises(ValueError) as interactive:
        chart.interactive().to_html()
    assert str(interactive.value) == message


def _numeric_chart() -> fm.Chart:
    return fm.Chart(_NUMERIC).mark_point().encode(x="n", y="v")


def _temporal_chart() -> fm.Chart:
    return fm.Chart(_TEMPORAL).mark_line().encode(x="date", y="v")


def test_a_category_on_a_numeric_x_axis_is_refused():
    _assert_refused_on_both_routes(
        _numeric_chart() + ann.text("b", 1.0, "t"), _continuous_message("b", "x")
    )


def test_a_category_on_a_numeric_y_axis_is_refused_naming_y():
    _assert_refused_on_both_routes(
        _bars("x", "default") + ann.text("b", "b", "t"), _continuous_message("b", "y")
    )


def test_a_category_on_a_log_axis_is_refused():
    chart = fm.Chart(_NUMERIC).mark_point().encode(x=fm.X("n", scale=fm.LogScale()), y="v")
    _assert_refused_on_both_routes(chart + ann.text("b", 1.0, "t"), _continuous_message("b", "x"))


def test_a_category_on_a_temporal_axis_is_refused():
    _assert_refused_on_both_routes(
        _temporal_chart() + fm.annotate_vline("peak_date"), _continuous_message("peak_date", "x")
    )


def test_a_span_category_end_on_a_numeric_axis_is_refused():
    chart = _numeric_chart() + ann.span("x", "b", "c", fill="#eee")
    _assert_refused_on_both_routes(chart, _continuous_message("b", "x"))


@pytest.mark.parametrize(
    "coord",
    [2.0, fm.NormCoord(0.5), fm.PixelCoord(40.0)],
    ids=["number", "norm", "pixel"],
)
def test_the_continuous_axis_remedy_renders_on_a_numeric_axis(coord):
    svg = (_numeric_chart() + ann.text(coord, 1.0, "HELLO")).to_svg()
    assert any(e.tag == "text" and e.text == "HELLO" for e in _elements(svg))
    interactive = _scene_annotations(_numeric_chart() + ann.text(coord, 1.0, "HELLO"))
    assert [n["content"] for n in interactive if n["type"] == "text"] == ["HELLO"]


@pytest.mark.parametrize(
    "coord",
    [dt.date(2020, 1, 2), "2020-01-02", fm.NormCoord(0.5), fm.PixelCoord(40.0)],
    ids=["date", "iso", "norm", "pixel"],
)
def test_the_continuous_axis_remedy_renders_on_a_temporal_axis(coord):
    svg = (_temporal_chart() + ann.text(coord, 2.0, "HELLO")).to_svg()
    assert any(e.tag == "text" and e.text == "HELLO" for e in _elements(svg))
    interactive = _scene_annotations(_temporal_chart() + ann.text(coord, 2.0, "HELLO"))
    assert [n["content"] for n in interactive if n["type"] == "text"] == ["HELLO"]


_POLAR_GEO = {
    "CoordPolar": lambda: fm.Chart(_DF).mark_bar().encode(x="c", y="v").coord(fm.CoordPolar()),
    "CoordGeo": lambda: (
        fm.Chart(pl.DataFrame({"lon": [-100.0, -90.0, -80.0], "lat": [30.0, 35.0, 40.0]}))
        .mark_point()
        .encode(x="lon", y="lat")
        .coord(fm.CoordGeo())
    ),
}


@pytest.mark.parametrize("coord", sorted(_POLAR_GEO))
def test_a_category_under_polar_or_geo_is_refused(coord):
    _assert_refused_on_both_routes(
        _POLAR_GEO[coord]() + ann.text("b", 2.0, "t"), _coord_message("b", "x", coord)
    )


@pytest.mark.parametrize("coord", sorted(_POLAR_GEO))
@pytest.mark.parametrize("remedy", [fm.NormCoord(0.5), fm.PixelCoord(200.0)], ids=["norm", "pixel"])
def test_the_polar_and_geo_remedy_renders(coord, remedy):
    chart = _POLAR_GEO[coord]() + ann.text(remedy, remedy, "HELLO", anchor="middle")
    assert any(e.tag == "text" and e.text == "HELLO" for e in _elements(chart.to_svg()))
    assert [n["content"] for n in _scene_annotations(chart) if n["type"] == "text"] == ["HELLO"]


@pytest.mark.parametrize(
    "axis,start,end,category_end",
    [
        ("x", "b", fm.NormCoord(0.5), "start"),
        ("x", fm.PixelCoord(40.0), "b", "end"),
        ("y", "b", fm.NormCoord(0.5), "start"),
    ],
    ids=["category-then-norm", "pixel-then-category", "y-axis"],
)
def test_a_span_with_one_category_end_and_one_position_end_is_refused(
    axis, start, end, category_end
):
    span = ann.span(axis, start, end, fill="#eee")
    _assert_refused_on_both_routes(
        _bars(axis, "default") + span, _mixed_span_message(axis, category_end, "b")
    )


@pytest.mark.parametrize(
    "axis,start,end,value",
    [
        ("x", "b", 2.5, "2.5"),
        ("x", 2.5, "b", "2.5"),
        ("x", "b", dt.date(2020, 1, 2), "2020-01-02"),
        ("y", "b", 2.5, "2.5"),
    ],
    ids=["category-then-number", "number-then-category", "date-end", "y-axis"],
)
def test_a_span_with_one_category_end_and_one_number_end_is_refused_as_a_value(
    axis, start, end, value
):
    """The value on the categorical axis is refused before the mixed-span check runs."""
    span = ann.span(axis, start, end, fill="#eee")
    _assert_refused_on_both_routes(_bars(axis, "default") + span, _value_message(value, axis))


@pytest.mark.parametrize(
    "chart,start,end,label",
    [
        (_temporal_chart, "Jan 2024", "2024-03-01", "Jan 2024"),
        (_numeric_chart, "b", 2.5, "b"),
    ],
    ids=["temporal", "numeric"],
)
def test_a_mixed_span_on_a_continuous_axis_is_refused_as_a_category_on_that_axis(
    chart, start, end, label
):
    _assert_refused_on_both_routes(
        chart() + ann.span("x", start, end, fill="#eee"), _continuous_message(label, "x")
    )


@pytest.mark.parametrize("coord", sorted(_POLAR_GEO))
def test_a_mixed_span_under_polar_or_geo_is_refused_by_the_coord(coord):
    _assert_refused_on_both_routes(
        _POLAR_GEO[coord]() + ann.span("x", "b", 2.5, fill="#eee"), _coord_message("b", "x", coord)
    )
    remedy = _POLAR_GEO[coord]() + ann.span(
        "x", fm.NormCoord(0.25), fm.NormCoord(0.75), fill=_COLOUR["span"]
    )
    assert _all(remedy.to_svg(), "rect", fill=_COLOUR["span"])


def test_a_number_on_the_polar_band_axis_is_refused_as_a_value():
    """The value check reads the scale, not the coord: the polar chart's x is still categorical,
    and the remedy names positions only because a category cannot be placed there either."""
    chart = _POLAR_GEO["CoordPolar"]()
    message = _value_under_coord_message
    _assert_refused_on_both_routes(
        chart + ann.text(2.5, 2.0, "t"), message("2.5", "x", "CoordPolar")
    )
    _assert_refused_on_both_routes(
        chart + ann.span("x", 1.0, 2.0, fill="#eee"), message("1", "x", "CoordPolar")
    )


def test_a_number_spelling_a_category_under_polar_names_only_the_position_remedies():
    df = pl.DataFrame({"yr": [2019, 2020, 2021, 2022], "v": [2.0, 1.0, 4.0, 3.0]})
    polar = fm.Chart(df).mark_bar().encode(x="yr:O", y="v").coord(fm.CoordPolar())
    _assert_refused_on_both_routes(
        polar + ann.text(2021, 2.0, "t"), _value_under_coord_message("2021", "x", "CoordPolar")
    )
    _assert_refused_on_both_routes(
        _POLAR_GEO["CoordPolar"]() + ann.text(dt.date(2020, 1, 2), 2.0, "t"),
        _value_under_coord_message("2020-01-02", "x", "CoordPolar"),
    )


@pytest.mark.parametrize("remedy", [fm.NormCoord(0.5), fm.PixelCoord(200.0)], ids=["norm", "pixel"])
def test_the_position_remedy_of_a_number_under_polar_renders(remedy):
    chart = _POLAR_GEO["CoordPolar"]() + ann.text(remedy, remedy, "HELLO", anchor="middle")
    assert any(e.tag == "text" and e.text == "HELLO" for e in _elements(chart.to_svg()))


def test_the_mixed_span_remedies_render_on_the_category_axis():
    """The refusal names categories at both ends, or ``fm.NormCoord`` or ``fm.PixelCoord`` at
    both ends; each renders on the band chart where the refusal fired."""
    colour = _COLOUR["span"]
    remedies = {
        "categories": ann.span("x", "b", "c", fill=colour),
        "norm": ann.span("x", fm.NormCoord(0.25), fm.NormCoord(0.75), fill=colour),
        "pixel": ann.span("x", fm.PixelCoord(100.0), fm.PixelCoord(300.0), fill=colour),
    }
    plain = _bars("x", "default").to_svg()
    ticks = _ticks(plain, "x")
    step = abs(ticks["b"] - ticks["a"])
    left, width = ticks["a"] - step / 2, step * len(_CATS)
    expected = {
        "categories": (ticks["b"] - step / 2, ticks["c"] + step / 2),
        "norm": (left + 0.25 * width, left + 0.75 * width),
    }
    for name, remedy in remedies.items():
        chart = _bars("x", "default") + remedy
        svg, shown = _render_capturing(chart.to_svg)
        assert shown == []
        extent = _span_extent(svg, "x")
        assert extent[1] - extent[0] > 100
        if name in expected:
            _near(extent, expected[name])
        nodes = _scene_annotations(chart)
        assert [n["type"] for n in nodes] == ["rect"]
        _near([nodes[0]["x"], nodes[0]["x"] + nodes[0]["w"]], extent)


@pytest.mark.parametrize(
    "chart,start,end",
    [
        (_numeric_chart, 1.5, 2.5),
        (_temporal_chart, dt.date(2020, 1, 1), dt.date(2020, 1, 2)),
    ],
    ids=["number", "date"],
)
def test_the_continuous_axis_remedy_renders_a_span(chart, start, end):
    built = chart() + ann.span("x", start, end, fill=_COLOUR["span"])
    assert _all(built.to_svg(), "rect", fill=_COLOUR["span"])
    assert [n["type"] for n in _scene_annotations(built)] == ["rect"]


# ---------------------------------------------------------------------------
# A category the band axis lacks: warned once in the caller's axis letter, not drawn
# ---------------------------------------------------------------------------


def _scene_annotation_count(chart: Any) -> int:
    """Annotation nodes over every panel of the chart's interactive scene."""
    panels = json.loads(chart.interactive().scene_json)["panels"]
    return sum(len(p["annotations"]) for p in panels)


def _assert_not_drawn(annotated: Any, plain: Any, *warnings: str) -> None:
    """``annotated`` draws the bytes ``plain`` draws, on the static and the interactive route, and
    warns exactly ``warnings`` on each: the unknown category left no annotation node behind."""
    svg, shown = _render_capturing(annotated.to_svg)
    assert shown == list(warnings)
    assert svg == plain.to_svg()
    count, shown = _render_capturing(lambda: _scene_annotation_count(annotated))
    assert shown == list(warnings)
    assert count == _scene_annotation_count(plain) == 0


def test_an_unknown_category_warns_with_the_axis_letter_and_is_not_drawn():
    plain = _bars("x", "sort")
    text = ann.text("zz", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    _assert_not_drawn(plain + text, plain, _unknown_message("x"))


def test_an_unknown_category_on_the_y_axis_names_y():
    plain = _bars("y", "sort")
    _assert_not_drawn(
        plain + ann.text(2.0, "zz", "HELLO", color=_COLOUR["text"]), plain, _unknown_message("y")
    )


def test_an_unknown_callout_text_position_names_the_axis_not_the_field():
    plain = _bars("x", "sort")
    callout = ann.callout(
        "b", 3.0, "note", text_x="zz", text_y=2.0, border_color=_COLOUR["callout"]
    )
    _assert_not_drawn(plain + callout, plain, _unknown_message("x"))


def test_an_unknown_arrow_end_names_the_axis_and_draws_nothing_of_the_arrow():
    plain = _bars("x", "sort")
    arrow = ann.arrow("b", 1.0, "zz", 3.0, stroke=_COLOUR["arrow"])
    _assert_not_drawn(plain + arrow, plain, _unknown_message("x"))


def test_an_unknown_span_end_names_the_axis_not_start_or_end():
    plain = _bars("x", "sort")
    span = ann.span("x", "b", "zz", fill=_COLOUR["span"])
    _assert_not_drawn(plain + span, plain, _unknown_message("x"))


def test_a_span_with_two_unknown_ends_warns_once_per_label():
    plain = _bars("x", "sort")
    span = ann.span("x", "yy", "zz", fill=_COLOUR["span"])
    _assert_not_drawn(plain + span, plain, _unknown_message("x", "yy"), _unknown_message("x", "zz"))


def test_a_span_naming_one_unknown_label_twice_warns_once():
    plain = _bars("x", "sort")
    span = ann.span("x", "zz", "zz", fill=_COLOUR["span"])
    _assert_not_drawn(plain + span, plain, _unknown_message("x"))


# Each kind with one unknown end (or anchor, or text position), the rest of its coordinates known.
_UNKNOWN_PRIMITIVES = {
    "text": lambda: ann.text("zz", 2.0, "HELLO", color=_COLOUR["text"]),
    "arrow": lambda: ann.arrow("b", 1.0, "zz", 3.0, stroke=_COLOUR["arrow"]),
    "arrow_start": lambda: ann.arrow("zz", 1.0, "c", 3.0, stroke=_COLOUR["arrow"]),
    "rect": lambda: ann.rect("b", 1.0, "zz", 3.0, fill=_COLOUR["rect"]),
    "line": lambda: ann.line("b", 1.0, "zz", 3.0, stroke=_COLOUR["line"]),
    "span": lambda: ann.span("x", "b", "zz", fill=_COLOUR["span"]),
    "span_start": lambda: ann.span("x", "zz", "c", fill=_COLOUR["span"]),
    "bracket": lambda: ann.bracket("b", 3.0, "zz", 3.0, label="r", stroke=_COLOUR["bracket"]),
    "callout_anchor": lambda: ann.callout(
        "zz", 3.0, "note", text_x="c", text_y=2.0, border_color=_COLOUR["callout"]
    ),
    "callout_text_x": lambda: ann.callout(
        "b", 3.0, "note", text_x="zz", text_y=2.0, border_color=_COLOUR["callout"]
    ),
    "image": lambda: ann.image("zz", 3.0, _IMAGE_SRC, width=10, height=10),
    "annotate_vline": lambda: fm.annotate_vline("zz", stroke=_COLOUR["line"]),
    "annotate_vline_labelled": lambda: fm.annotate_vline("zz", label="LBL", stroke="#ff0020"),
    "annotate_text": lambda: fm.annotate_text("zz", 2.0, "HELLO", color=_COLOUR["text"]),
    "annotate_rect": lambda: fm.annotate_rect("b", "zz", 1.0, 3.0, fill=_COLOUR["rect"]),
    "annotate_arrow": lambda: fm.annotate_arrow("b", 1.0, "zz", 3.0, stroke=_COLOUR["arrow"]),
}


@pytest.mark.parametrize("kind", sorted(_UNKNOWN_PRIMITIVES))
def test_every_annotation_kind_with_an_unknown_category_is_not_drawn(kind):
    plain = _bars("x", "sort")
    _assert_not_drawn(plain + _UNKNOWN_PRIMITIVES[kind](), plain, _unknown_message("x"))


@pytest.mark.parametrize(
    "build",
    [
        lambda: fm.annotate_hline("zz", stroke=_COLOUR["line"]),
        lambda: ann.rect(1.0, "b", 3.0, "zz", fill=_COLOUR["rect"]),
        lambda: ann.span("y", "b", "zz", fill=_COLOUR["span"]),
        lambda: fm.annotate_arrow(1.0, "b", 3.0, "zz", stroke=_COLOUR["arrow"]),
    ],
    ids=["annotate_hline", "rect", "span", "annotate_arrow"],
)
def test_an_unknown_category_on_the_y_axis_is_not_drawn_for_every_shape(build):
    plain = _bars("y", "sort")
    _assert_not_drawn(plain + build(), plain, _unknown_message("y"))


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_an_unknown_category_is_not_drawn_through_every_door(door):
    chart = _DOORS[door](fm.X("c", sort="-y"), ann.text("zz", 2.0, "HELLO", color=_COLOUR["text"]))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_unknown_message("x")]
    assert [e for e in _elements(svg) if e.tag == "text" and e.text == "HELLO"] == []
    count, shown = _render_capturing(lambda: _scene_annotation_count(chart))
    assert shown == [_unknown_message("x")]
    # A door may draw nodes of its own (a break's indicators); a known text adds exactly one.
    known = _DOORS[door](fm.X("c", sort="-y"), _primitive("text", "x"))
    assert count == _scene_annotation_count(known) - 1


def test_an_unknown_category_leaves_the_known_ones_drawn():
    """A skipped annotation takes only itself: a known one beside it still lands on its band."""
    plain = _bars("x", "sort")
    chart = plain + ann.text("zz", 2.0, "SKIP") + _primitive("text", "x")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_unknown_message("x")]
    assert [e for e in _elements(svg) if e.text == "SKIP"] == []
    _near(_measured("text", svg, "x"), [_ticks(svg, "x")["b"]])
    assert _scene_annotation_count(chart) == 1


def test_a_known_category_raises_no_warning():
    _, shown = _render_capturing((_bars("x", "sort") + _primitive("text", "x")).to_svg)
    assert shown == []


# ---------------------------------------------------------------------------
# A number or a date on a categorical axis is refused, in the caller's axis letter
# ---------------------------------------------------------------------------

_NUMBER = 7.5


def _number_on_category(kind: str) -> Any:
    """``kind`` with one number on the category (x) axis and every other coordinate a known
    category or a value on the continuous axis."""
    n = _NUMBER
    return {
        "text": lambda: ann.text(n, 2.0, "t"),
        "arrow": lambda: ann.arrow(n, 1.0, "c", 3.0),
        "arrow_end": lambda: ann.arrow("b", 1.0, n, 3.0),
        "rect": lambda: ann.rect("b", 1.0, n, 3.0, fill="#eee"),
        "line": lambda: ann.line("b", 1.0, n, 3.0),
        "span": lambda: ann.span("x", n, "c", fill="#eee"),
        "bracket": lambda: ann.bracket(n, 3.0, "c", 3.0, label="r"),
        "callout_anchor": lambda: ann.callout(n, 3.0, "note", text_x="c", text_y=2.0),
        "callout_text_x": lambda: ann.callout("b", 3.0, "note", text_x=n, text_y=2.0),
        "image": lambda: ann.image(n, 3.0, _IMAGE_SRC, width=10, height=10),
        "annotate_vline": lambda: fm.annotate_vline(n),
        "annotate_text": lambda: fm.annotate_text(n, 2.0, "t"),
        "annotate_rect": lambda: fm.annotate_rect("b", n, 1.0, 3.0),
        "annotate_arrow": lambda: fm.annotate_arrow(n, 1.0, "c", 3.0),
    }[kind]()


_NUMBER_KINDS = [
    "text",
    "arrow",
    "arrow_end",
    "rect",
    "line",
    "span",
    "bracket",
    "callout_anchor",
    "callout_text_x",
    "image",
    "annotate_vline",
    "annotate_text",
    "annotate_rect",
    "annotate_arrow",
]


@pytest.mark.parametrize("kind", _NUMBER_KINDS)
def test_a_number_on_a_band_axis_is_refused_for_every_annotation_kind(kind):
    _assert_refused_on_both_routes(
        _bars("x", "sort") + _number_on_category(kind), _value_message("7.5", "x")
    )


@pytest.mark.parametrize(
    "build,value",
    [
        (lambda: ann.text(2.0, 2.0, "t"), "2"),
        (lambda: fm.annotate_vline(1), "1"),
        (lambda: ann.span("x", 1.0, 2.0, fill="#eee"), "1"),
    ],
    ids=["text", "annotate_vline", "span"],
)
def test_a_number_on_a_band_axis_is_refused_with_the_generic_remedy(build, value):
    _assert_refused_on_both_routes(_bars("x", "default") + build(), _value_message(value, "x"))


@pytest.mark.parametrize(
    "build",
    [
        lambda: ann.text(2.0, 7.5, "t"),
        lambda: fm.annotate_hline(7.5),
        lambda: ann.span("y", 7.5, "c", fill="#eee"),
        lambda: ann.rect(1.0, 7.5, 3.0, "b", fill="#eee"),
    ],
    ids=["text", "annotate_hline", "span", "rect"],
)
def test_a_number_on_a_vertical_category_axis_is_refused_naming_y(build):
    _assert_refused_on_both_routes(_bars("y", "sort") + build(), _value_message("7.5", "y"))


def test_a_number_on_a_category_axis_under_a_callers_flip_names_the_callers_letter():
    """The caller's x is the category axis, drawn vertically; the refusal still says ``x``."""
    flipped = fm.Chart(_DF).mark_bar().encode(x="c", y="v").coord(fm.CoordFlip())
    _assert_refused_on_both_routes(
        flipped + ann.text(_NUMBER, 2.0, "t"), _value_message("7.5", "x")
    )
    on_y = fm.Chart(_DF).mark_bar().encode(x="v", y="c").coord(fm.CoordFlip())
    _assert_refused_on_both_routes(on_y + ann.text(2.0, _NUMBER, "t"), _value_message("7.5", "y"))


def test_a_number_on_the_continuous_axis_beside_a_category_still_draws():
    """Only the categorical axis refuses a number: ``y`` here is the value axis."""
    svg, shown = _render_capturing(
        (_bars("x", "sort") + ann.text("b", _NUMBER, "HELLO", color=_COLOUR["text"])).to_svg
    )
    assert shown == []
    assert [e.text for e in _elements(svg) if e.attrs.get("fill") == _COLOUR["text"]] == ["HELLO"]


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_number_on_a_band_axis_is_refused_through_every_door_on_both_routes(door):
    chart = _DOORS[door](fm.X("c", sort="-y"), ann.text(_NUMBER, 2.0, "t"))
    message = _value_message("7.5", "x")
    with pytest.raises(ValueError) as static:
        chart.to_svg()
    assert str(static.value) == _STATIC_PREFIX.get(door, "") + message
    with pytest.raises(ValueError) as interactive:
        chart.interactive().to_html()
    assert str(interactive.value) == _INTERACTIVE_PREFIX.get(door, "") + message


# A refusal inside a composite leaf is prefixed by the leaf that failed; the prefix differs by
# route for a LayerChart (the static route renders its overlay leaves, the interactive route the
# merged panel).
_STATIC_PREFIX = {
    "layer_chart": "failed to render composite leaf leaf #1: ",
    "hconcat_leaf": "failed to render composite leaf leaf #0: ",
}
_INTERACTIVE_PREFIX = {"hconcat_leaf": "failed to render composite leaf leaf #0: "}


def test_a_number_beside_a_category_on_the_continuous_axis_refuses_the_category_first():
    """Refusal order: the category on the continuous axis speaks before the number on the
    categorical one."""
    chart = _bars("x", "default") + ann.text(_NUMBER, "q", "t")
    _assert_refused_on_both_routes(chart, _continuous_message("q", "y"))


@pytest.mark.parametrize("coord", sorted(_POLAR_GEO))
def test_a_number_before_a_category_under_polar_or_geo_keeps_the_coord_refusal_for_the_category(
    coord,
):
    """Under geo the 2.5 lies on the continuous lon axis; under polar it is the number first in
    the spec. Either way the category is read first and named."""
    _assert_refused_on_both_routes(
        _POLAR_GEO[coord]() + ann.span("x", 2.5, "b", fill="#eee"), _coord_message("b", "x", coord)
    )


def test_a_number_on_an_integer_ordinal_axis_names_the_label_that_spells_it():
    df = pl.DataFrame({"yr": [2019, 2020, 2021, 2022], "v": [2.0, 1.0, 4.0, 3.0]})
    plain = fm.Chart(df).mark_bar().encode(x="yr:O", y="v")
    _assert_refused_on_both_routes(
        plain + ann.text(2021, 2.0, "t"), _value_message("2021", "x", "2021")
    )
    # A number that spells no category gets the generic remedy.
    _assert_refused_on_both_routes(plain + ann.text(2030, 2.0, "t"), _value_message("2030", "x"))


def test_the_label_remedy_of_an_integer_ordinal_axis_lands_on_the_tick():
    df = pl.DataFrame({"yr": [2019, 2020, 2021, 2022], "v": [2.0, 1.0, 4.0, 3.0]})
    chart = fm.Chart(df).mark_bar().encode(x="yr:O", y="v") + ann.text(
        "2021", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    tick = _ticks(svg, "x")["2021"]
    _near(_measured("text", svg, "x"), [tick])
    (node,) = [n for n in _scene_annotations(chart) if n["type"] == "text"]
    _near(node["x"], tick)


def test_a_number_on_a_float_ordinal_axis_names_the_label_that_spells_it():
    df = pl.DataFrame({"c": [1.5, 2.5, 3.5, 4.5], "v": [2.0, 1.0, 4.0, 3.0]})
    plain = fm.Chart(df).mark_bar().encode(x="c:O", y="v")
    _assert_refused_on_both_routes(
        plain + ann.text(2.5, 2.0, "t"), _value_message("2.5", "x", "2.5")
    )


@pytest.mark.parametrize(
    "coord",
    [fm.NormCoord(0.5), fm.PixelCoord(200.0)],
    ids=["norm", "pixel"],
)
def test_the_position_remedy_renders_in_place_of_the_refused_number(coord):
    chart = _bars("x", "sort") + ann.text(coord, 2.0, "HELLO", color=_COLOUR["text"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert [e.text for e in _elements(svg) if e.attrs.get("fill") == _COLOUR["text"]] == ["HELLO"]
    assert _scene_annotation_count(chart) == 1


@pytest.mark.parametrize("coord", [fm.NormCoord(0.25), fm.PixelCoord(120.0)], ids=["norm", "pixel"])
def test_the_position_remedy_renders_in_place_of_a_refused_helper_number(coord):
    chart = _bars("x", "sort") + fm.annotate_vline(coord, stroke=_COLOUR["line"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert _all(svg, "line", stroke=_COLOUR["line"])
    assert _scene_annotation_count(chart) == 1


# Each spelling of 2020-01-02 is a date: refused on a string band axis, with no label clause
# because no category of that axis is a date.
_DATE_SPELLINGS = {
    "date": dt.date(2020, 1, 2),
    "datetime": dt.datetime(2020, 1, 2),
    "iso": "2020-01-02",
    "datetime64": np.datetime64("2020-01-02"),
}


@pytest.mark.parametrize("spelling", sorted(_DATE_SPELLINGS))
def test_a_date_on_a_string_band_axis_is_refused_without_a_label_clause(spelling):
    chart = _bars("x", "sort") + ann.text(_DATE_SPELLINGS[spelling], 2.0, "t")
    _assert_refused_on_both_routes(chart, _value_message("2020-01-02", "x"))


def test_a_datetime_on_a_string_band_axis_is_refused_naming_its_full_text():
    chart = _bars("x", "sort") + ann.text(dt.datetime(2020, 1, 2, 12), 2.0, "t")
    _assert_refused_on_both_routes(chart, _value_message("2020-01-02T12:00:00", "x"))


def test_a_date_on_a_string_band_axis_whose_category_spells_it_is_still_refused():
    """The axis holds the string ``'2020-01-02'``, which is not a date category: the date is
    refused without a label clause, and the explicit category spelling lands on the band."""
    df = pl.DataFrame({"c": ["2020-01-01", "2020-01-02", "2020-01-03"], "v": [1.0, 2.0, 3.0]})
    plain = fm.Chart(df).mark_bar().encode(x="c", y="v")
    _assert_refused_on_both_routes(
        plain + ann.text(dt.date(2020, 1, 2), 2.0, "t"), _value_message("2020-01-02", "x")
    )
    chart = plain + ann.text("2020-01-02", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    _near(_measured("text", svg, "x"), [_ticks(svg, "x")["2020-01-02"]])


@pytest.mark.parametrize("coord", [fm.NormCoord(0.5), fm.PixelCoord(200.0)], ids=["norm", "pixel"])
def test_the_position_remedy_renders_in_place_of_a_refused_date(coord):
    refused = _bars("x", "sort") + ann.text(dt.date(2020, 1, 2), 2.0, "HELLO")
    _assert_refused_on_both_routes(refused, _value_message("2020-01-02", "x"))
    chart = _bars("x", "sort") + ann.text(coord, 2.0, "HELLO", color=_COLOUR["text"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    assert [e.text for e in _elements(svg) if e.attrs.get("fill") == _COLOUR["text"]] == ["HELLO"]


# ---------------------------------------------------------------------------
# A date on an axis whose categories are dates lands on its band
# ---------------------------------------------------------------------------

_DAY_ONE, _DAY_TWO, _DAY_THREE = dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)
_DAYS = pl.DataFrame(
    {
        "day": [_DAY_ONE] * 3 + [_DAY_TWO] * 3 + [_DAY_THREE] * 3,
        "v": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
    }
)
# The same frame with each day spelled as its category label: the string band axis the date axis
# must draw identically.
_DAY_LABELS = _DAYS.with_columns(pl.col("day").dt.to_string("%Y-%m-%d"))
_DAY_TEXT = ("2020-01-01", "2020-01-02", "2020-01-03")
_NOON = pl.DataFrame(
    {
        "day": [dt.datetime(2020, 1, 1, 12)] * 3 + [dt.datetime(2020, 1, 2)] * 3,
        "v": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
    }
)


def _day_boxes(frame: pl.DataFrame = _DAYS) -> fm.Chart:
    return fm.catplot(frame, x="day", y="v", kind="box")


def _hello(svg: str) -> float:
    (el,) = [e for e in _elements(svg) if e.tag == "text" and e.text == "HELLO"]
    return float(el.attrs["x"])


def _hello_text(x: Any) -> Any:
    return ann.text(x, 2.0, "HELLO", anchor="middle")


@pytest.mark.parametrize("spelling", sorted(_DATE_SPELLINGS))
def test_a_date_lands_on_the_band_centre_of_its_date_category(spelling):
    chart = _day_boxes() + _hello_text(_DATE_SPELLINGS[spelling])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    ticks = _ticks(svg, "x")
    assert list(ticks) == list(_DAY_TEXT)
    _near(_hello(svg), ticks["2020-01-02"])
    (node,) = [n for n in _scene_annotations(chart) if n["type"] == "text"]
    _near(node["x"], ticks["2020-01-02"])


def test_a_date_vline_lands_on_the_band_centre_of_the_date_axis():
    chart = _day_boxes() + fm.annotate_vline(_DAY_TWO, stroke=_COLOUR["line"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    centre = _ticks(svg, "x")["2020-01-02"]
    line = _one(svg, "line", stroke=_COLOUR["line"])
    _near([float(line.attrs["x1"]), float(line.attrs["x2"])], [centre, centre])
    (node,) = [n for n in _scene_annotations(chart) if n["type"] == "line"]
    _near([node["x1"], node["x2"]], [centre, centre])


def test_each_date_of_the_axis_lands_on_its_own_band():
    """The three days land on three different ticks: a placement that ignored the date and
    used one band for every date would fail."""
    ticks = _ticks(_day_boxes().to_svg(), "x")
    for day, label in (
        (_DAY_ONE, "2020-01-01"),
        (_DAY_TWO, "2020-01-02"),
        (_DAY_THREE, "2020-01-03"),
    ):
        svg = (_day_boxes() + _hello_text(day)).to_svg()
        _near(_hello(svg), ticks[label])
    assert len(set(ticks.values())) == 3


def test_a_date_span_shades_whole_bands_of_both_dates():
    span = ann.span("x", _DAY_ONE, _DAY_TWO, fill=_COLOUR["span"])
    svg, shown = _render_capturing((_day_boxes() + span).to_svg)
    assert shown == []
    ticks = _ticks(svg, "x")
    lo, hi = _span_extent(svg, "x")
    step = ticks["2020-01-02"] - ticks["2020-01-01"]
    assert lo < ticks["2020-01-01"] - step / 4
    assert hi > ticks["2020-01-02"] + step / 4
    assert hi < ticks["2020-01-03"]
    twin = _day_boxes(_DAY_LABELS) + ann.span(
        "x",
        OrdinalCategoryCoord("2020-01-01"),
        OrdinalCategoryCoord("2020-01-02"),
        fill=_COLOUR["span"],
    )
    _near(_span_extent(twin.to_svg(), "x"), (lo, hi))
    nodes = [n for n in _scene_annotations(_day_boxes() + span) if n["type"] == "rect"]
    _near((min(n["x"] for n in nodes), max(n["x"] + n["w"] for n in nodes)), (lo, hi))


def test_a_date_annotation_places_as_the_equal_category_text_does():
    """The date chart against its text-label twin: the same ticks, and a date lands where the
    explicit category text does."""
    date_svg = (_day_boxes() + _hello_text(_DAY_TWO)).to_svg()
    twin_svg = (_day_boxes(_DAY_LABELS) + _hello_text(OrdinalCategoryCoord("2020-01-02"))).to_svg()
    assert _ticks(date_svg, "x") == _ticks(twin_svg, "x")
    _near(_hello(date_svg), _hello(twin_svg))


def test_a_non_midnight_date_category_axis_holds_the_full_text_label():
    chart = _day_boxes(_NOON) + _hello_text(dt.datetime(2020, 1, 1, 12))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    ticks = _ticks(svg, "x")
    assert list(ticks) == ["2020-01-01T12:00:00", "2020-01-02"]
    _near(_hello(svg), ticks["2020-01-01T12:00:00"])
    iso = (_day_boxes(_NOON) + _hello_text("2020-01-01T12:00:00")).to_svg()
    _near(_hello(iso), ticks["2020-01-01T12:00:00"])


def test_a_midnight_date_on_a_non_midnight_date_axis_lands_on_its_own_band():
    svg = (_day_boxes(_NOON) + _hello_text(_DAY_TWO)).to_svg()
    _near(_hello(svg), _ticks(svg, "x")["2020-01-02"])


def test_the_date_whose_instant_is_not_a_category_is_not_drawn():
    """``2020-01-01`` at midnight is not the ``2020-01-01T12:00:00`` category."""
    plain = _day_boxes(_NOON)
    absent = ["2020-01-01T12:00:00", "2020-01-02"]
    _assert_not_drawn(
        plain + _hello_text(_DAY_ONE), plain, _unknown_message("x", "2020-01-01", absent)
    )


@pytest.mark.parametrize(
    "build,label",
    [
        (lambda: _hello_text(dt.date(2020, 1, 5)), "2020-01-05"),
        (lambda: fm.annotate_vline(dt.date(2020, 1, 5)), "2020-01-05"),
        (lambda: fm.annotate_text("2020-01-05", 2.0, "HELLO"), "2020-01-05"),
        (lambda: _hello_text(np.datetime64("2020-01-05")), "2020-01-05"),
        (lambda: ann.span("x", _DAY_ONE, dt.date(2020, 1, 5), fill=_COLOUR["span"]), "2020-01-05"),
    ],
    ids=["text", "annotate_vline", "annotate_text", "datetime64", "span-end"],
)
def test_a_date_absent_from_the_date_axis_is_not_drawn_and_warns_with_the_date_labels(build, label):
    plain = _day_boxes()
    _assert_not_drawn(plain + build(), plain, _unknown_message("x", label, list(_DAY_TEXT)))


def test_a_span_of_two_absent_dates_warns_once_per_label():
    plain = _day_boxes()
    span = ann.span("x", dt.date(2020, 1, 5), dt.date(2020, 1, 6), fill=_COLOUR["span"])
    _assert_not_drawn(
        plain + span,
        plain,
        _unknown_message("x", "2020-01-05", list(_DAY_TEXT)),
        _unknown_message("x", "2020-01-06", list(_DAY_TEXT)),
    )


def test_a_number_on_a_date_axis_is_refused_without_a_label_clause():
    _assert_refused_on_both_routes(_day_boxes() + ann.text(3.0, 2.0, "t"), _value_message("3", "x"))
    _assert_refused_on_both_routes(
        _day_boxes() + fm.annotate_vline(1577923200000.0), _value_message("1577923200000", "x")
    )


def test_a_category_label_on_a_date_axis_lands_on_its_band():
    """The text label of a date category is a category: it lands on the band."""
    chart = _day_boxes() + _hello_text(OrdinalCategoryCoord("2020-01-02"))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    _near(_hello(svg), _ticks(svg, "x")["2020-01-02"])


def test_a_date_on_a_date_axis_under_a_callers_flip_lands_on_the_callers_axis():
    """The caller's x holds the dates; drawn vertically, the band sits on the page's y."""
    flipped = _day_boxes().coord(fm.CoordFlip())
    chart = flipped + ann.text(_DAY_TWO, 2.0, "HELLO", anchor="middle")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    (el,) = [e for e in _elements(svg) if e.tag == "text" and e.text == "HELLO"]
    _near(float(el.attrs["y"]), _ticks(svg, "y")["2020-01-02"])
    _assert_refused_on_both_routes(flipped + ann.text(3.0, 2.0, "t"), _value_message("3", "x"))
    absent = flipped + ann.text(dt.date(2020, 1, 5), 2.0, "t")
    _assert_not_drawn(absent, flipped, _unknown_message("x", "2020-01-05", list(_DAY_TEXT)))


def test_a_date_lands_on_the_date_axis_through_the_layered_doors():
    plus = _day_boxes() + _hello_text(_DAY_TWO)
    via_layer = fm.Chart(_DAYS).encode(x="day:O", y="v").layer(fm.Layer(mark="bar")) + _hello_text(
        _DAY_TWO
    )
    for chart in (plus, via_layer):
        svg, shown = _render_capturing(chart.to_svg)
        assert shown == []
        _near(_hello(svg), _ticks(svg, "x")["2020-01-02"])


# ---------------------------------------------------------------------------
# A date, a datetime, a datetime64 and an ISO string cross to the renderer as an instant
# ---------------------------------------------------------------------------

_JAN_TWO_MS = 1577923200000.0


@pytest.mark.parametrize("spelling", sorted(_DATE_SPELLINGS))
def test_every_temporal_spelling_crosses_as_an_instant(spelling):
    """A date, datetime and datetime64 cross as the bare instant; an ISO string also carries the
    text as written."""
    value = _DATE_SPELLINGS[spelling]
    wire = {"instant": _JAN_TWO_MS} | ({"text": value} if spelling == "iso" else {})
    assert ann.text(value, 2.0, "t").to_dict()["x"] == wire
    span = ann.span("x", value, value, fill="#eee").to_dict()
    assert (span["start"], span["end"]) == (wire,) * 2
    sent = (_temporal_chart() + fm.annotate_vline(value))._resolve_chart_config()["annotations"]
    assert [a["x1"] for a in sent] == [wire]
    sent = (_temporal_chart() + ann.text(value, 2.0, "t"))._resolve_chart_config()["annotations"]
    assert [a["x"] for a in sent] == [wire]


def test_a_number_a_category_and_a_position_keep_their_own_wire_forms():
    """An epoch-millisecond number is not an instant: only a date spelling crosses as one."""
    assert ann.text(_JAN_TWO_MS, 2.0, "t").to_dict()["x"] == _JAN_TWO_MS
    assert ann.text("b", 2.0, "t").to_dict()["x"] == {"category": "b"}
    assert ann.text(fm.NormCoord(0.5), 2.0, "t").to_dict()["x"] == {"norm": 0.5}
    assert ann.text(fm.PixelCoord(40.0), 2.0, "t").to_dict()["x"] == {"px": 40.0}


def test_a_numpy_datetime64_crosses_as_the_instant_of_the_equal_datetime():
    for unit in ("D", "s", "ms", "us"):
        value = np.datetime64("2020-01-02T00:00:00", unit)
        assert ann.text(value, 2.0, "t").to_dict()["x"] == {"instant": _JAN_TWO_MS}
    noon = np.datetime64("2020-01-02T12:00:00.250")
    twin = dt.datetime(2020, 1, 2, 12, 0, 0, 250000)
    assert ann.text(noon, 2.0, "t").to_dict()["x"] == ann.text(twin, 2.0, "t").to_dict()["x"]


def test_a_numpy_datetime64_is_placed_as_the_equal_datetime_on_a_temporal_axis():
    chart = _temporal_chart()
    for build in (
        lambda v: ann.text(v, 2.0, "t"),
        lambda v: fm.annotate_vline(v),
        lambda v: ann.span("x", v, v, fill=_COLOUR["span"]),
    ):
        as_datetime = (chart + build(dt.datetime(2020, 1, 2))).to_svg()
        as_numpy = (chart + build(np.datetime64("2020-01-02"))).to_svg()
        assert as_numpy == as_datetime


def test_every_temporal_spelling_is_placed_alike_on_a_temporal_axis():
    chart = _temporal_chart()
    placed = {
        name: (chart + ann.text(value, 2.0, "HELLO", color=_COLOUR["text"])).to_svg()
        for name, value in _DATE_SPELLINGS.items()
    }
    assert len(set(placed.values())) == 1
    millis = (chart + ann.text(_JAN_TWO_MS, 2.0, "HELLO", color=_COLOUR["text"])).to_svg()
    assert millis == placed["date"]


# ---------------------------------------------------------------------------
# Numeric, date, norm and pixel coordinates (no flip) draw the same bytes, bar a callout's theme colours
#
# Each digest was measured on the build before the category change reached the renderer, except
# the numeric, date, norm and pixel cells: those four were measured again after the callout began
# following the theme, and differ from the earlier build only in the callout's box and text fills.
# ---------------------------------------------------------------------------


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


def _every_kind(x: Any, y: Any, x2: Any, y2: Any) -> list[Any]:
    return [
        ann.text(x, y, "t", color=_COLOUR["text"]),
        ann.arrow(x, y, x2, y2, stroke=_COLOUR["arrow"]),
        ann.rect(x, y, x2, y2, fill=_COLOUR["rect"]),
        ann.line(x, y, x2, y2, stroke=_COLOUR["line"]),
        ann.span("x", x, x2, fill=_COLOUR["span"]),
        ann.span("y", y, y2, fill="#ff0008"),
        ann.bracket(x, y, x2, y2, label="r", stroke=_COLOUR["bracket"]),
        ann.callout(x, y, "n", text_x=x2, text_y=y2, border_color=_COLOUR["callout"]),
        ann.image(x, y, _IMAGE_SRC, width=10, height=10),
    ]


def _layered(chart: fm.Chart, primitives: list[Any]) -> str:
    for primitive in primitives:
        chart = chart + primitive
    return chart.to_svg()


_COORDINATE_CHARTS = {
    "numeric": (
        lambda: _layered(
            _numeric_chart(),
            _every_kind(1.2, 1.4, 2.6, 2.7)
            + [fm.annotate_hline(2.0), fm.annotate_vline(2.0), fm.annotate_abline(1.0, 0.0)],
        ),
        "4d872254ef351e8f79d0bd419b3e17fb0fb04e986562e676906a1126a94e2d19",
    ),
    "date": (
        lambda: _layered(
            _temporal_chart(),
            _every_kind(dt.date(2020, 1, 1), 1.4, dt.datetime(2020, 1, 2, 12), 2.7)
            + [fm.annotate_vline("2020-01-02"), fm.annotate_text("2020-01-02", 2.0, "iso")],
        ),
        "1155b0e83a1514dad691ddebde2a40ce36437edd8ff7c78943a96a5e494b342e",
    ),
    "norm": (
        lambda: _layered(
            _numeric_chart(),
            _every_kind(fm.NormCoord(0.2), fm.NormCoord(0.3), fm.NormCoord(0.6), fm.NormCoord(0.7)),
        ),
        "0d599a333c822bd885201a04629881131066216644df831bdfa9d7619b4e63bb",
    ),
    "pixel": (
        lambda: _layered(
            _numeric_chart(),
            _every_kind(
                fm.PixelCoord(90.0), fm.PixelCoord(70.0), fm.PixelCoord(300.0), fm.PixelCoord(200.0)
            ),
        ),
        "7e63cfe3009dcb9f321acdfa10b9716753546bb0bc7b1d29be8c465381a3ab8c",
    ),
    "numeric_on_a_band_axis": (
        lambda: _layered(
            _bars("x", "sort"),
            [ann.text(fm.NormCoord(0.4), 2.0, "t"), ann.span("y", 1.5, 2.5, fill=_COLOUR["span"])],
        ),
        "d0cbb86bcc059214783bc4071ae793590dae306d16631689ea37a8e640bb10ef",
    ),
}


# The charts of these coordinate kinds carry a callout with no background, so its box takes the
# theme's background colour and its text the theme's text colour; every other element draws the
# bytes it drew before the callout followed the theme.
_CALLOUT_FOLLOWS_THEME = ("numeric", "date", "norm", "pixel")


@pytest.mark.parametrize("name", sorted(set(_COORDINATE_CHARTS) - set(_CALLOUT_FOLLOWS_THEME)))
def test_a_non_category_coordinate_draws_the_bytes_it_drew_before(name):
    build, digest = _COORDINATE_CHARTS[name]
    assert _digest(build()) == digest


@pytest.mark.parametrize("name", sorted(_CALLOUT_FOLLOWS_THEME))
def test_a_non_category_coordinate_draws_a_callout_in_the_theme_colours(name):
    build, digest = _COORDINATE_CHARTS[name]
    assert _digest(build()) == digest


def test_a_null_category_leaves_the_band_layout_to_the_frame_without_it():
    """The twin of the band annotation tests in ``test_null_category_one_rule.py``: the null row
    has no band, so a category annotation sits where the frame without the row draws it."""
    dirty = pl.DataFrame({"c": ["a", None, "b"], "v": [1.0, 2.0, 3.0]})
    clean = dirty.filter(pl.col("c").is_not_null())

    def build(df: pl.DataFrame) -> fm.Chart:
        return fm.Chart(df).mark_point().encode(x="c", y="v") + ann.text("b", 2.0, "t")

    assert _render_capturing(build(dirty).to_svg)[0] == _render_capturing(build(clean).to_svg)[0]


# ---------------------------------------------------------------------------
# Every ``annotate_*`` helper, labelled and not, on a band axis on x and on y
#
# An unlabelled helper once drew its mark layer from zero placeholders for a category,
# Pixel or Norm end; each must land on its band (or its pixel position) with no warning.
# ---------------------------------------------------------------------------

_HELPER_COLOUR = "#ff0020"
_LABEL = "LBL"


def _helper(name: str, axis: str, label: str | None) -> Any:
    """The helper's annotation with its category ends on ``axis``."""
    kw: dict[str, Any] = {} if label is None else {"label": label}
    if name == "line":
        if axis == "x":
            return fm.annotate_vline("b", stroke=_HELPER_COLOUR, **kw)
        return fm.annotate_hline("b", stroke=_HELPER_COLOUR, **kw)
    if name == "arrow":
        ends = _at(axis, "b", 1.0) + _at(axis, "c", 3.0)
        return fm.annotate_arrow(*ends, stroke=_HELPER_COLOUR, **kw)
    if name == "rect":
        ends = ("b", "c", 1.0, 3.0) if axis == "x" else (1.0, 3.0, "b", "c")
        return fm.annotate_rect(*ends, fill=_HELPER_COLOUR, **kw)
    if name == "text":
        return fm.annotate_text(
            *_at(axis, "b", 2.0), "HELLO", color=_HELPER_COLOUR, anchor="middle"
        )
    raise AssertionError(name)


def _helper_expected(name: str, ticks: dict[str, float]) -> list[float]:
    if name == "line":
        return [ticks["b"], ticks["b"]]
    if name == "rect":
        return sorted([ticks["b"], ticks["c"]])
    if name == "text":
        return [ticks["b"]]
    return [ticks["b"], ticks["c"]]


def _helper_static(name: str, svg: str, axis: str) -> list[float]:
    lo, hi = ("x1", "x2") if axis == "x" else ("y1", "y2")
    if name in ("line", "arrow"):
        line = _one(svg, "line", stroke=_HELPER_COLOUR)
        return [float(line.attrs[lo]), float(line.attrs[hi])]
    if name == "rect":
        return sorted(_lo_hi(_one(svg, "rect", fill=_HELPER_COLOUR), axis))
    return [float(_one(svg, "text", fill=_HELPER_COLOUR).attrs[axis])]


def _helper_scene(name: str, nodes: list[dict[str, Any]], axis: str) -> list[float]:
    lo, hi = ("x1", "x2") if axis == "x" else ("y1", "y2")
    if name in ("line", "arrow"):
        node = next(n for n in nodes if n["type"] == "line")
        return [node[lo], node[hi]]
    if name == "rect":
        node = next(n for n in nodes if n["type"] == "rect")
        size = node["w" if axis == "x" else "h"]
        return sorted([node[axis], node[axis] + size])
    return [next(n for n in nodes if n["type"] == "text" and n["content"] == "HELLO")[axis]]


_HELPER_CASES = [
    (name, label)
    for name in ("line", "arrow", "rect", "text")
    for label in ([None] if name == "text" else [None, _LABEL])
]


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("name,label", _HELPER_CASES)
def test_an_annotate_helper_lands_on_its_band_without_a_warning(name, label, axis):
    chart = _bars(axis, "sort") + _helper(name, axis, label)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    expected = _helper_expected(name, _ticks(svg, axis))
    _near(_helper_static(name, svg, axis), expected)
    nodes, shown = _render_capturing(lambda: _scene_annotations(chart))
    assert shown == []
    _near(_helper_scene(name, nodes, axis), expected)


@pytest.mark.parametrize("label", [None, _LABEL])
@pytest.mark.parametrize(
    "coords",
    [
        (fm.NormCoord(0.2), fm.NormCoord(0.3), fm.NormCoord(0.6), fm.NormCoord(0.7)),
        (fm.PixelCoord(90.0), fm.PixelCoord(70.0), fm.PixelCoord(300.0), fm.PixelCoord(200.0)),
    ],
    ids=["norm", "pixel"],
)
def test_an_arrow_helper_with_norm_or_pixel_ends_takes_the_same_route_labelled_or_not(
    coords, label
):
    kw: dict[str, Any] = {} if label is None else {"label": label}
    chart = _numeric_chart() + fm.annotate_arrow(*coords, stroke=_HELPER_COLOUR, **kw)
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    twin = _numeric_chart() + ann.arrow(*coords, stroke=_HELPER_COLOUR)
    line = _one(svg, "line", stroke=_HELPER_COLOUR)
    twin_line = _one(twin.to_svg(), "line", stroke=_HELPER_COLOUR)
    assert line.attrs == twin_line.attrs


def test_an_all_numeric_unlabelled_arrow_keeps_the_mark_merge_route():
    """Numeric ends stay on the ``mark_segment`` layer, which takes the themed stroke."""
    arrow = fm.annotate_arrow(1.0, 1.0, 2.0, 2.0)
    assert arrow._annotation_primitive is None


# ---------------------------------------------------------------------------
# A category on an ``Axis(label_map=...)`` axis: the data value lands on its band
#
# ``label_map`` relabels tick text only, so an annotation category names the data value, as
# ``sort=`` and ``domain=`` do. The display label is not a category: it is an unknown one.
# ---------------------------------------------------------------------------

_BEE = {"b": "Bee"}


def _mapped(label_map: dict[str, str], *, sort: str | None = None) -> Any:
    return fm.X("c", axis=fm.Axis(label_map=label_map), sort=sort)


def _line_x(svg: str, stroke: str) -> float:
    return float(_one(svg, "line", stroke=stroke).attrs["x1"])


def _scene_line_x(chart: Any, stroke_nodes: int = 1) -> float:
    nodes = [n for n in _scene_annotations(chart) if n["type"] == "line"]
    assert len(nodes) == stroke_nodes
    return nodes[0]["x1"]


def test_a_category_on_a_label_mapped_axis_lands_on_its_band():
    chart = _doors_bars(_mapped(_BEE, sort="-y")) + fm.annotate_vline("b", stroke="#ff0030")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    tick = _ticks(svg, "x")["Bee"]
    _near(_line_x(svg, "#ff0030"), tick)
    scene, shown = _render_capturing(lambda: _scene_line_x(chart))
    assert shown == []
    _near(scene, tick)


def test_the_display_label_is_an_unknown_category():
    """``'Bee'`` is not one of the axis's categories (``a`` to ``d``): the unknown-category
    warning names it on the static and the interactive routes, and nothing is drawn."""
    plain = _doors_bars(_mapped(_BEE, sort="-y"))
    chart = plain + ann.text("Bee", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    _assert_not_drawn(chart, plain, _unknown_message("x", "Bee"))


def test_a_mixed_span_on_a_label_mapped_axis_echoes_the_data_value():
    """The tick reads ``Bee``; the refusal names the category the caller wrote, ``b``."""
    chart = _doors_bars(_mapped(_BEE)) + ann.span("x", "b", fm.NormCoord(0.9), fill="#eee")
    _assert_refused_on_both_routes(chart, _mixed_span_message("x", "start", "b"))


def test_a_swapping_label_map_places_the_data_value_not_the_display_text():
    """``{'a': 'b', 'b': 'a'}``: the raw ``b`` rows are drawn under the tick labelled ``a``,
    and an annotation on the data value ``b`` lands there, as a mark for those rows does."""
    chart = _doors_bars(_mapped({"a": "b", "b": "a"})) + ann.text(
        "b", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    ticks = _ticks(svg, "x")
    _near(_measured("text", svg, "x"), [ticks["a"]])
    assert ticks["a"] != ticks["b"]


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_label_mapped_category_lands_through_every_door(door):
    chart = _DOORS[door](_mapped(_BEE, sort="-y"), _primitive("text", "x"))
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    # The first layer that draws the axis decides its tick text: the own-frame door starts with a
    # layer that has no map, so its axis reads ``b``.
    label = "b" if door == "operand_with_its_own_frame" else "Bee"
    tick = _ticks(svg, "x")[label]
    _near(_measured("text", svg, "x"), [tick])
    nodes = _first_annotated_panel(chart)
    _near([n["x"] for n in nodes if n["type"] == "text"], [tick])


def _mapped_y(label_map: dict[str, str]) -> fm.Chart:
    return (
        fm.Chart(_DF)
        .mark_bar()
        .encode(x="v", y=fm.Y("c", axis=fm.Axis(label_map=label_map), sort="-x"))
    )


def _text_y(label: str) -> Any:
    return ann.text(2.0, label, "HELLO", color=_COLOUR["text"])


def _scene_rects(chart: Any) -> list[dict[str, Any]]:
    return [n for n in _scene_annotations(chart) if n["type"] == "rect"]


def test_a_category_on_a_label_mapped_y_axis_lands_on_its_band():
    point = _mapped_y(_BEE) + _text_y("b")
    svg, shown = _render_capturing(point.to_svg)
    assert shown == []
    ticks = _ticks(svg, "y")
    _near(_measured("text", svg, "y"), [ticks["Bee"]])
    nodes = [n for n in _scene_annotations(point) if n["type"] == "text"]
    _near([n["y"] for n in nodes], [ticks["Bee"]])

    span = _mapped_y(_BEE) + ann.span("y", "b", "c", fill=_COLOUR["span"])
    span_svg, shown = _render_capturing(span.to_svg)
    assert shown == []
    ticks = _ticks(span_svg, "y")
    lo_b, hi_b = _band_edges(ticks | {"b": ticks["Bee"]}, "b", 0.0)
    lo_c, hi_c = _band_edges(ticks | {"b": ticks["Bee"]}, "c", 0.0)
    expected = (min(lo_b, lo_c), max(hi_b, hi_c))
    _near(_span_extent(span_svg, "y"), expected)
    (rect,) = _scene_rects(span)
    _near((rect["y"], rect["y"] + rect["h"]), expected)


def test_the_display_label_is_an_unknown_category_on_the_y_axis():
    _assert_not_drawn(
        _mapped_y(_BEE) + _text_y("Bee"), _mapped_y(_BEE), _unknown_message("y", "Bee")
    )


def test_a_swapping_label_map_on_the_y_axis_places_the_data_value():
    swap = {"a": "b", "b": "a"}
    chart = _mapped_y(swap) + _text_y("b")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    ticks = _ticks(svg, "y")
    assert ticks["a"] != ticks["b"]
    _near(_measured("text", svg, "y"), [ticks["a"]])
    span = _mapped_y(swap) + ann.span("y", "b", "b", fill=_COLOUR["span"])
    span_svg = span.to_svg()
    lo, hi = _band_edges(_ticks(span_svg, "y"), "a", 0.0)
    _near(_span_extent(span_svg, "y"), (lo, hi))
    (rect,) = _scene_rects(span)
    _near((rect["y"], rect["y"] + rect["h"]), (lo, hi))


# Different maps on the two axes, so a map routed to the wrong axis (or dropped) moves a category.
_XY = pl.DataFrame({"c": _CATS, "k": ["d", "c", "b", "a"], "v": [2.0, 1.0, 4.0, 3.0]})


def _both_axes_mapped() -> fm.Chart:
    return (
        fm.Chart(_XY)
        .mark_point()
        .encode(
            x=fm.X("c", axis=fm.Axis(label_map={"b": "Bx"})),
            y=fm.Y("k", axis=fm.Axis(label_map={"b": "By"})),
        )
    )


def _check_both_axes_mapped() -> None:
    colour = _COLOUR["span"]
    chart = _both_axes_mapped() + ann.text("b", "b", "HELLO", color=_COLOUR["text"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    x_ticks, y_ticks = _ticks(svg, "x"), _ticks(svg, "y")
    _near(_measured("text", svg, "x"), [x_ticks["Bx"]])
    _near(_measured("text", svg, "y"), [y_ticks["By"]])
    scene = next(n for n in _scene_annotations(chart) if n["type"] == "text")
    _near((scene["x"], scene["y"]), (x_ticks["Bx"], y_ticks["By"]))
    for axis, ticks, renamed in (("x", x_ticks, "Bx"), ("y", y_ticks, "By")):
        span = _both_axes_mapped() + ann.span(axis, "b", "b", fill=colour)
        span_svg, shown = _render_capturing(span.to_svg)
        assert shown == []
        span_ticks = _ticks(span_svg, axis)
        positions = list(span_ticks.values())
        step = (max(positions) - min(positions)) / (len(positions) - 1)
        centre = span_ticks[renamed]
        _near(_span_extent(span_svg, axis), (centre - step / 2, centre + step / 2))


def test_different_label_maps_on_x_and_y_each_rename_their_own_axis():
    _check_both_axes_mapped()


def test_a_label_map_on_an_enum_or_ordinal_integer_axis_relabels_and_the_data_value_lands():
    """The map reads tick text, so it relabels an Enum column and an integer column under
    ``:O`` too; the annotation names the data value and lands on that relabelled tick."""
    enum_df = pl.DataFrame({"c": pl.Series(_CATS).cast(pl.Enum(_CATS)), "v": [2.0, 1.0, 4.0, 3.0]})
    int_df = pl.DataFrame({"c": [2019, 2020, 2021, 2022], "v": [2.0, 1.0, 4.0, 3.0]})
    cases = [
        (enum_df, "c", {"b": "Bee"}, "b", "Bee"),
        (int_df, "c:O", {"2021": "Late"}, "2021", "Late"),
    ]
    for df, field, label_map, label, shown_as in cases:
        chart = fm.Chart(df).mark_bar().encode(
            x=fm.X(field, axis=fm.Axis(label_map=label_map)), y="v"
        ) + ann.text(label, 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
        svg, shown = _render_capturing(chart.to_svg)
        assert shown == []
        _near(_measured("text", svg, "x"), [_ticks(svg, "x")[shown_as]])


# ---------------------------------------------------------------------------
# Pins: every door on the interactive route, every category dtype, empty category axes
# ---------------------------------------------------------------------------


def _first_annotated_panel(chart: Any) -> list[dict[str, Any]]:
    panels = json.loads(chart.interactive().scene_json)["panels"]
    annotated = [p["annotations"] for p in panels if p["annotations"]]
    assert annotated
    return annotated[0]


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_the_interactive_route_places_a_category_through_every_door(door):
    channel = fm.X("c", sort="-y")
    colour = _COLOUR["span"]
    text_chart = _DOORS[door](channel, _primitive("text", "x"))
    span_chart = _DOORS[door](channel, ann.span("x", "b", "c", fill=colour))
    static_text, static_span = text_chart.to_svg(), span_chart.to_svg()
    ticks = _ticks(static_text, "x")
    scene_text = [n for n in _first_annotated_panel(text_chart) if n["type"] == "text"]
    _near([n["x"] for n in scene_text], [ticks["b"]])
    pieces = [n for n in _first_annotated_panel(span_chart) if n["type"] == "rect"]
    assert pieces
    lo = min(n["x"] for n in pieces)
    hi = max(n["x"] + n["w"] for n in pieces)
    _near((lo, hi), _span_extent(static_span, "x"))
    lo_b, hi_b = _band_edges(ticks, "b", 0.0)
    lo_c, hi_c = _band_edges(ticks, "c", 0.0)
    _near((lo, hi), (min(lo_b, lo_c), max(hi_b, hi_c)))


def _dtype_frames() -> dict[str, tuple[pl.DataFrame, str, str]]:
    values = [2.0, 1.0, 4.0, 3.0]
    return {
        "categorical": (
            pl.DataFrame({"c": pl.Series(_CATS).cast(pl.Categorical), "v": values}),
            "c",
            "b",
        ),
        # The enum's order differs from the data order, so the tick is not a data-order guess.
        "enum": (
            pl.DataFrame({"c": pl.Series(_CATS).cast(pl.Enum(["d", "c", "b", "a"])), "v": values}),
            "c",
            "b",
        ),
        "int_ordinal": (pl.DataFrame({"c": [2019, 2020, 2021, 2022], "v": values}), "c:O", "2021"),
        "float_ordinal": (
            pl.DataFrame({"c": [1.5, 2.5, 3.5, 4.5], "v": values}),
            "c:O",
            "2.5",
        ),
        "boolean": (pl.DataFrame({"c": [True, False, True, False], "v": values}), "c:O", "true"),
    }


@pytest.mark.parametrize("dtype", sorted(_dtype_frames()))
def test_a_category_lands_on_its_tick_for_every_category_dtype(dtype):
    df, field, label = _dtype_frames()[dtype]
    chart = fm.Chart(df).mark_bar().encode(x=field, y="v") + ann.text(
        label, 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    _near(_measured("text", svg, "x"), [_ticks(svg, "x")[label]])
    nodes = [n for n in _scene_annotations(chart) if n["type"] == "text"]
    _near([n["x"] for n in nodes], [_ticks(svg, "x")[label]])


def test_a_boolean_category_is_matched_by_its_axis_label_only():
    """The axis labels a boolean ``true``/``false``; ``True`` is not one of them."""
    df, field, _ = _dtype_frames()["boolean"]
    plain = fm.Chart(df).mark_bar().encode(x=field, y="v")
    chart = plain + ann.text("True", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    _assert_not_drawn(chart, plain, _unknown_message("x", "True", ["false", "true"]))


_NO_ROWS = _NO_ROWS_TO_DRAW
_ALL_NULL = _all_keys_null_message("mark_bar", "c")


@pytest.mark.parametrize(
    "frame,leading",
    [
        (
            pl.DataFrame({"c": pl.Series([], dtype=pl.Utf8), "v": pl.Series([], dtype=pl.Float64)}),
            [_NO_ROWS],
        ),
        (
            pl.DataFrame({"c": pl.Series([None, None], dtype=pl.Utf8), "v": [1.0, 2.0]}),
            [_ALL_NULL, _NO_ROWS],
        ),
    ],
    ids=["zero-row", "all-null"],
)
def test_a_category_on_an_empty_category_axis_warns_and_is_not_drawn(frame, leading):
    plain = fm.Chart(frame).mark_bar().encode(x="c", y="v")
    chart = plain + ann.text("b", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [*leading, _unknown_message("x", "b", [])]
    assert svg == _render_capturing(plain.to_svg)[0]
    count, shown = _render_capturing(lambda: _scene_annotation_count(chart))
    assert shown == [*leading, _unknown_message("x", "b", [])]
    assert count == 0


# ---------------------------------------------------------------------------
# A date on a date-category axis is its category for every check, not only for placement
# ---------------------------------------------------------------------------


def _polar_days() -> fm.Chart:
    return fm.Chart(_DAYS).mark_bar().encode(x="day:O", y="v").coord(fm.CoordPolar())


@pytest.mark.parametrize(
    "day,label",
    [(dt.date(2020, 1, 2), "2020-01-02"), (dt.date(2020, 1, 9), "2020-01-09")],
    ids=["present", "absent"],
)
def test_a_date_on_a_date_axis_under_polar_is_refused_as_its_category(day, label):
    _assert_refused_on_both_routes(
        _polar_days() + _hello_text(day), _coord_message(label, "x", "CoordPolar")
    )


def test_a_date_on_a_date_axis_under_polar_has_a_position_remedy_that_renders():
    chart = _polar_days() + ann.text(fm.NormCoord(0.5), fm.NormCoord(0.5), "HELLO")
    assert any(e.tag == "text" and e.text == "HELLO" for e in _elements(chart.to_svg()))


@pytest.mark.parametrize(
    "start,end,category_end",
    [(_DAY_TWO, fm.NormCoord(1.0), "start"), (fm.PixelCoord(5.0), _DAY_TWO, "end")],
    ids=["date-first", "pixel-first"],
)
def test_a_date_span_with_one_position_end_is_refused_as_a_mixed_span(start, end, category_end):
    chart = _day_boxes() + ann.span("x", start, end, fill=_COLOUR["span"])
    _assert_refused_on_both_routes(chart, _mixed_span_message("x", category_end, "2020-01-02"))


def test_a_date_span_with_an_unknown_category_end_is_not_drawn_and_warns():
    plain = _day_boxes()
    span = ann.span("x", _DAY_TWO, "zz", fill=_COLOUR["span"])
    _assert_not_drawn(plain + span, plain, _unknown_message("x", "zz", list(_DAY_TEXT)))


def test_a_category_and_a_date_span_shades_the_bands_two_dates_shade():
    dates = ann.span("x", _DAY_ONE, _DAY_TWO, fill=_COLOUR["span"])
    mixed = ann.span("x", OrdinalCategoryCoord("2020-01-01"), _DAY_TWO, fill=_COLOUR["span"])
    svg, shown = _render_capturing((_day_boxes() + mixed).to_svg)
    assert shown == []
    _near(_span_extent(svg, "x"), _span_extent((_day_boxes() + dates).to_svg(), "x"))
    scene = [n for n in _scene_annotations(_day_boxes() + mixed) if n["type"] == "rect"]
    twin = [n for n in _scene_annotations(_day_boxes() + dates) if n["type"] == "rect"]
    _near(
        (min(n["x"] for n in scene), max(n["x"] + n["w"] for n in scene)),
        (min(n["x"] for n in twin), max(n["x"] + n["w"] for n in twin)),
    )


# ---------------------------------------------------------------------------
# A helper's own coordinates on a categorical axis are refused in the helper's voice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chart,axis",
    [
        (lambda: _bars("x", "default"), "x"),
        (lambda: _bars("y", "default"), "y"),
        (lambda: fm.Chart(_DF).mark_bar().encode(x="c", y="v").coord(fm.CoordFlip()), "x"),
        (lambda: _POLAR_GEO["CoordPolar"](), "x"),
    ],
    ids=["x", "y", "flip", "polar"],
)
def test_annotate_abline_on_a_categorical_axis_is_refused_in_its_own_voice(chart, axis):
    _assert_refused_on_both_routes(
        chart() + fm.annotate_abline(1.0, 0.0), _helper_message("annotate_abline", axis)
    )


def test_only_annotate_abline_stamps_its_line_with_its_entry():
    assert (
        fm.annotate_abline(1.0, 0.0)._annotation_primitive.to_dict()["entry"] == "annotate_abline"
    )
    assert "entry" not in ann.line(0.0, 0.0, 1.0, 1.0).to_dict()
    assert "entry" not in fm.annotate_vline(1.0)._annotation_primitive.to_dict()
    assert "entry" not in fm.annotate_hline(1.0)._annotation_primitive.to_dict()


def test_a_line_the_caller_wrote_keeps_the_value_voice_on_a_band_axis():
    _assert_refused_on_both_routes(
        _bars("x", "default") + ann.line("b", 1.0, _NUMBER, 3.0), _value_message("7.5", "x")
    )


# ---------------------------------------------------------------------------
# Boundaries of the value refusal and the skip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"c": pl.Series([], dtype=pl.Utf8), "v": pl.Series([], dtype=pl.Float64)}),
        pl.DataFrame({"c": pl.Series([None, None], dtype=pl.Utf8), "v": [1.0, 2.0]}),
    ],
    ids=["zero-row", "all-null"],
)
def test_a_number_on_an_empty_category_axis_is_refused(frame):
    chart = fm.Chart(frame).mark_bar().encode(x="c", y="v") + ann.text(2.0, 2.0, "t")
    _assert_refused_on_both_routes(chart, _value_message("2", "x"))


@pytest.mark.parametrize(
    "frame,number,label",
    [
        (
            pl.DataFrame({"c": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32), "v": [1.0, 2.0, 3.0]}),
            2.5,
            "2.5",
        ),
        (pl.DataFrame({"c": pl.Series([1, 2, 3], dtype=pl.Int8), "v": [1.0, 2.0, 3.0]}), 2, "2"),
    ],
    ids=["float32", "int8"],
)
def test_a_number_on_a_narrow_ordinal_axis_names_the_label_that_spells_it(frame, number, label):
    chart = fm.Chart(frame).mark_bar().encode(x="c:O", y="v") + ann.text(number, 2.0, "t")
    _assert_refused_on_both_routes(chart, _value_message(label, "x", label))


# ---------------------------------------------------------------------------
# An ISO string equal to a label of a string axis names that category
# ---------------------------------------------------------------------------


def _iso_text_days() -> fm.Chart:
    return _day_boxes(_DAY_LABELS)


def test_an_iso_string_that_is_a_label_of_a_string_axis_lands_on_its_band():
    df = pl.DataFrame({"d": list(_DAY_TEXT), "v": [1.0, 2.0, 3.0]})
    plain = fm.Chart(df).mark_bar().encode(x="d", y="v")
    chart = plain + ann.text("2020-01-02", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    tick = _ticks(svg, "x")["2020-01-02"]
    _near(_measured("text", svg, "x"), [tick])
    (node,) = [n for n in _scene_annotations(chart) if n["type"] == "text"]
    _near(node["x"], tick)
    twin = plain + ann.text(
        OrdinalCategoryCoord("2020-01-02"), 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    assert svg == twin.to_svg()
    # The same chart with the label shifted off the tick: the text moved with the label.
    other = plain + ann.text("2020-01-03", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    assert other.to_svg() != svg
    # The vline and the span on the ISO-text twin of the date axis.
    base = _iso_text_days()
    ticks = _ticks(base.to_svg(), "x")
    vline = _render_capturing((base + fm.annotate_vline("2020-01-02", stroke="#ff0010")).to_svg)
    assert vline[1] == []
    line = _one(vline[0], "line", stroke="#ff0010")
    _near(float(line.attrs["x1"]), ticks["2020-01-02"])
    span = ann.span("x", "2020-01-01", "2020-01-02", fill=_COLOUR["span"])
    twin_span = ann.span(
        "x",
        OrdinalCategoryCoord("2020-01-01"),
        OrdinalCategoryCoord("2020-01-02"),
        fill=_COLOUR["span"],
    )
    svg, shown = _render_capturing((base + span).to_svg)
    assert shown == []
    _near(_span_extent(svg, "x"), _span_extent((base + twin_span).to_svg(), "x"))
    lo, hi = _span_extent(svg, "x")
    assert lo < ticks["2020-01-01"] < ticks["2020-01-02"] < hi < ticks["2020-01-03"]


def test_an_iso_string_that_is_not_a_label_of_a_string_axis_is_refused_and_its_remedy_lands():
    base = _iso_text_days()
    _assert_refused_on_both_routes(
        base + ann.text("2020-01-09", 2.0, "t"), _value_message("2020-01-09", "x")
    )
    remedy = base + ann.text("2020-01-02", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
    svg, shown = _render_capturing(remedy.to_svg)
    assert shown == []
    _near(_measured("text", svg, "x"), [_ticks(svg, "x")["2020-01-02"]])


def test_an_iso_string_on_a_string_band_axis_is_refused_quoting_its_text():
    chart = _bars("x", "default") + ann.text("2020-01-02T00:00:00", 2.0, "t")
    _assert_refused_on_both_routes(chart, _value_message("2020-01-02T00:00:00", "x"))


def test_an_iso_string_that_is_a_label_is_refused_under_polar_as_its_category():
    polar = fm.Chart(_DAY_LABELS).mark_bar().encode(x="day", y="v").coord(fm.CoordPolar())
    _assert_refused_on_both_routes(
        polar + ann.text("2020-01-02", 2.0, "t"), _coord_message("2020-01-02", "x", "CoordPolar")
    )
    span = ann.span("x", "2020-01-02", fm.NormCoord(0.9), fill="#eee")
    _assert_refused_on_both_routes(
        _iso_text_days() + span, _mixed_span_message("x", "start", "2020-01-02")
    )
