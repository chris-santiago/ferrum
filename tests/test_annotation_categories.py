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

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
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
    _near(float(_one(text, "text", fill="#333333").attrs["x"]), ticks["b"])
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
    assert ann.text("2020-01-02", 2.0, "t").to_dict()["x"] == 1577923200000.0
    assert (
        ann.span("x", "2020-01-01", "2020-01-02", fill="#eee").to_dict()["start"] == 1577836800000.0
    )
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


def _unknown_message(axis: str, label: str = "zz") -> str:
    return (
        f"annotation category '{label}' not found in {axis}-axis domain "
        f"{sorted(_CATS)}; placing at plot center"
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
        ("x", "b", 2.5, "start"),
        ("x", 2.5, "b", "end"),
        ("x", "b", dt.date(2020, 1, 2), "start"),
        ("x", "b", fm.NormCoord(0.5), "start"),
        ("y", "b", 2.5, "start"),
    ],
    ids=["category-then-value", "value-then-category", "date-end", "norm-end", "y-axis"],
)
def test_a_span_with_one_category_end_and_one_value_end_is_refused(axis, start, end, category_end):
    span = ann.span(axis, start, end, fill="#eee")
    _assert_refused_on_both_routes(
        _bars(axis, "default") + span, _mixed_span_message(axis, category_end, "b")
    )


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
    remedy = _POLAR_GEO[coord]() + ann.span("x", fm.NormCoord(0.5), 2.5, fill=_COLOUR["span"])
    assert _all(remedy.to_svg(), "rect", fill=_COLOUR["span"])


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
# A category the band axis lacks: warned once in the caller's axis letter, drawn at the centre
# ---------------------------------------------------------------------------


def _plot_centre(svg: str, axis: str) -> float:
    ticks = _ticks(svg, axis)
    values = [ticks[c] for c in _CATS]
    return (min(values) + max(values)) / 2.0


def test_an_unknown_category_warns_with_the_axis_letter_and_sits_at_the_centre():
    chart = _bars("x", "sort") + ann.text(
        "zz", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_unknown_message("x")]
    _near(_measured("text", svg, "x"), [_plot_centre(svg, "x")])


def test_an_unknown_category_on_the_y_axis_names_y():
    chart = _bars("y", "sort") + ann.text(2.0, "zz", "HELLO", color=_COLOUR["text"])
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [_unknown_message("y")]
    _near(float(_one(svg, "text", fill=_COLOUR["text"]).attrs["y"]), _plot_centre(svg, "y"))


def test_an_unknown_callout_text_position_names_the_axis_not_the_field():
    callout = ann.callout(
        "b", 3.0, "note", text_x="zz", text_y=2.0, border_color=_COLOUR["callout"]
    )
    svg, shown = _render_capturing((_bars("x", "sort") + callout).to_svg)
    assert shown == [_unknown_message("x")]
    note = [e for e in _elements(svg) if e.tag == "text" and e.text == "note"][0]
    _near(float(note.attrs["x"]), _plot_centre(svg, "x"))


def test_an_unknown_arrow_end_names_the_axis_and_sits_at_the_centre():
    arrow = ann.arrow("b", 1.0, "zz", 3.0, stroke=_COLOUR["arrow"])
    svg, shown = _render_capturing((_bars("x", "sort") + arrow).to_svg)
    assert shown == [_unknown_message("x")]
    line = _one(svg, "line", stroke=_COLOUR["arrow"])
    _near(float(line.attrs["x1"]), _ticks(svg, "x")["b"])
    _near(float(line.attrs["x2"]), _plot_centre(svg, "x"))


def test_an_unknown_span_end_names_the_axis_not_start_or_end():
    span = ann.span("x", "b", "zz", fill=_COLOUR["span"])
    svg, shown = _render_capturing((_bars("x", "sort") + span).to_svg)
    assert shown == [_unknown_message("x")]
    # The known end is band b's centre and the unknown end the plot centre.
    extent = _span_extent(svg, "x")
    _near(sorted(extent), sorted([_ticks(svg, "x")["b"], _plot_centre(svg, "x")]))


def test_a_span_with_two_unknown_ends_warns_once_per_label():
    span = ann.span("x", "yy", "zz", fill=_COLOUR["span"])
    svg, shown = _render_capturing((_bars("x", "sort") + span).to_svg)
    assert shown == [_unknown_message("x", "yy"), _unknown_message("x", "zz")]
    (rect,) = _all(svg, "rect", fill=_COLOUR["span"])
    _near(float(rect.attrs["width"]), 0.0)
    _near(float(rect.attrs["x"]), _plot_centre(svg, "x"))


def test_a_known_category_raises_no_warning():
    _, shown = _render_capturing((_bars("x", "sort") + _primitive("text", "x")).to_svg)
    assert shown == []


# ---------------------------------------------------------------------------
# Unchanged: numeric, date, norm and pixel coordinates (no flip) draw the same bytes
#
# Each digest was measured on the build before the category change reached the renderer.
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


_UNCHANGED = {
    "numeric": (
        lambda: _layered(
            _numeric_chart(),
            _every_kind(1.2, 1.4, 2.6, 2.7)
            + [fm.annotate_hline(2.0), fm.annotate_vline(2.0), fm.annotate_abline(1.0, 0.0)],
        ),
        "70875b22bef951dbb5c78e2cd67be29ffc0581403119ff94b5e7f7d4ed66694c",
    ),
    "date": (
        lambda: _layered(
            _temporal_chart(),
            _every_kind(dt.date(2020, 1, 1), 1.4, dt.datetime(2020, 1, 2, 12), 2.7)
            + [fm.annotate_vline("2020-01-02"), fm.annotate_text("2020-01-02", 2.0, "iso")],
        ),
        "0f912ea1d046fedf38b3b146d02f509a3b888cd2835fd13b23247aadca6b7af2",
    ),
    "norm": (
        lambda: _layered(
            _numeric_chart(),
            _every_kind(fm.NormCoord(0.2), fm.NormCoord(0.3), fm.NormCoord(0.6), fm.NormCoord(0.7)),
        ),
        "fa299e565c9373ba82d60c6a91f17236e74fdc9b7e40e5beff118c0e5169daca",
    ),
    "pixel": (
        lambda: _layered(
            _numeric_chart(),
            _every_kind(
                fm.PixelCoord(90.0), fm.PixelCoord(70.0), fm.PixelCoord(300.0), fm.PixelCoord(200.0)
            ),
        ),
        "1a07a89b90966626c3ca90f11c61256b1397d9c621df1368ffd2e17b39923780",
    ),
    "numeric_on_a_band_axis": (
        lambda: _layered(
            _bars("x", "sort"),
            [ann.text(fm.NormCoord(0.4), 2.0, "t"), ann.span("y", 1.5, 2.5, fill=_COLOUR["span"])],
        ),
        "ed24db0bc00a46208985db7a22f8980b10699969bc868c885804f435654685d0",
    ),
}


@pytest.mark.parametrize("name", sorted(_UNCHANGED))
def test_a_non_category_coordinate_draws_the_bytes_it_drew_before(name):
    build, digest = _UNCHANGED[name]
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
# A category on an ``Axis(label_map=...)`` axis: the data value lands, so does the display label
#
# ``label_map`` renames the column's values before they reach the renderer, so the renderer's
# categories are the display labels; an annotation category follows the same map.
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


@pytest.mark.parametrize("label", ["b", "Bee"], ids=["data-value", "display-label"])
def test_a_category_on_a_label_mapped_axis_lands_on_its_band(label):
    chart = _doors_bars(_mapped(_BEE, sort="-y")) + fm.annotate_vline(label, stroke="#ff0030")
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == []
    tick = _ticks(svg, "x")["Bee"]
    _near(_line_x(svg, "#ff0030"), tick)
    scene, shown = _render_capturing(lambda: _scene_line_x(chart))
    assert shown == []
    _near(scene, tick)


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
    # A ``+`` operand with its own frame keeps its raw rows and ticks (``label_map`` renames the
    # chart's own data only), so the axis there displays ``b``; the category follows the rename
    # the data took and crosses as ``b``.
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


@pytest.mark.parametrize("label", ["b", "Bee"], ids=["data-value", "display-label"])
def test_a_category_on_a_label_mapped_y_axis_lands_on_its_band(label):
    point = _mapped_y(_BEE) + _text_y(label)
    svg, shown = _render_capturing(point.to_svg)
    assert shown == []
    ticks = _ticks(svg, "y")
    _near(_measured("text", svg, "y"), [ticks["Bee"]])
    nodes = [n for n in _scene_annotations(point) if n["type"] == "text"]
    _near([n["y"] for n in nodes], [ticks["Bee"]])

    span = _mapped_y(_BEE) + ann.span("y", label, "c", fill=_COLOUR["span"])
    span_svg, shown = _render_capturing(span.to_svg)
    assert shown == []
    ticks = _ticks(span_svg, "y")
    lo_b, hi_b = _band_edges(ticks | {"b": ticks["Bee"]}, "b", 0.0)
    lo_c, hi_c = _band_edges(ticks | {"b": ticks["Bee"]}, "c", 0.0)
    expected = (min(lo_b, lo_c), max(hi_b, hi_c))
    _near(_span_extent(span_svg, "y"), expected)
    (rect,) = _scene_rects(span)
    _near((rect["y"], rect["y"] + rect["h"]), expected)


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


@pytest.mark.parametrize("break_kind", ["y_map_routed_to_x", "y_map_dropped"])
def test_the_two_axis_label_map_test_fails_under_a_misrouted_map(monkeypatch, break_kind):
    """The reviewer's breaks, applied in memory: a y map routed into x, and a y map dropped."""
    import ferrum._render as render

    real = render._axis_label_maps

    def broken(chart: Any, applied: Any) -> dict[str, dict[str, str]]:
        axes = real(chart, applied)
        if break_kind == "y_map_routed_to_x":
            return {"x": {**axes["x"], **axes["y"]}, "y": {}}
        return {"x": axes["x"], "y": {}}

    monkeypatch.setattr(render, "_axis_label_maps", broken)
    with pytest.raises(AssertionError):
        _check_both_axes_mapped()


def _rename_every_field(primitives: list[Any]) -> list[dict[str, Any]]:
    from ferrum._render_prepare import _rename_annotation_categories

    dicts = [dict(p.to_dict()) for p in primitives]
    maps = {"x": {"b": "ViaX"}, "y": {"b": "ViaY"}}
    return _rename_annotation_categories(dicts, maps)


def test_every_annotation_coordinate_field_is_renamed_through_its_axis_label_map():
    """Each primitive carries a category on every coordinate field it has; a field the axis
    table misses stays ``b`` and fails here."""
    c = "b"
    primitives = [
        ann.text(c, c, "t"),
        ann.arrow(c, c, c, c),
        ann.rect(c, c, c, c, fill="#eee"),
        ann.line(c, c, c, c),
        ann.span("x", c, c, fill="#eee"),
        ann.span("y", c, c, fill="#eee"),
        ann.bracket(c, c, c, c, label="r"),
        ann.callout(c, c, "t", text_x=c, text_y=c),
        ann.image(c, c, _IMAGE_SRC),
    ]
    renamed = _rename_every_field(primitives)
    x_fields = {"x", "x1", "x2", "text_x"}
    y_fields = {"y", "y1", "y2", "text_y"}
    seen = 0
    for original, out in zip(primitives, renamed):
        for key, value in out.items():
            if not (isinstance(value, dict) and "category" in value):
                continue
            seen += 1
            if key in x_fields:
                expected = "ViaX"
            elif key in y_fields:
                expected = "ViaY"
            else:
                assert key in ("start", "end"), key
                expected = "ViaX" if out["axis"] == "x" else "ViaY"
            assert value == {"category": expected}, (out["type"], key)
    assert seen == 2 + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 2


def test_a_label_map_the_data_does_not_take_leaves_the_category_alone():
    """An Enum column and an integer column under ``:O`` are not renamed by ``label_map``, so
    the annotation is not renamed either: it lands on the raw tick, with no warning."""
    enum_df = pl.DataFrame({"c": pl.Series(_CATS).cast(pl.Enum(_CATS)), "v": [2.0, 1.0, 4.0, 3.0]})
    int_df = pl.DataFrame({"c": [2019, 2020, 2021, 2022], "v": [2.0, 1.0, 4.0, 3.0]})
    cases = [
        (enum_df, "c", {"b": "Bee"}, "b"),
        (int_df, "c:O", {"2021": "Late"}, "2021"),
    ]
    for df, field, label_map, label in cases:
        chart = fm.Chart(df).mark_bar().encode(
            x=fm.X(field, axis=fm.Axis(label_map=label_map)), y="v"
        ) + ann.text(label, 2.0, "HELLO", color=_COLOUR["text"], anchor="middle")
        svg, shown = _render_capturing(chart.to_svg)
        assert shown == []
        _near(_measured("text", svg, "x"), [_ticks(svg, "x")[label]])


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
    chart = fm.Chart(df).mark_bar().encode(x=field, y="v") + ann.text(
        "True", 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    svg, shown = _render_capturing(chart.to_svg)
    assert shown == [
        "annotation category 'True' not found in x-axis domain ['false', 'true']; "
        "placing at plot center"
    ]
    twin = fm.Chart(df).mark_bar().encode(x=field, y="v") + ann.text(
        fm.NormCoord(0.5), 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
    )
    _near(_measured("text", svg, "x"), _measured("text", twin.to_svg(), "x"))


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
def test_a_category_on_an_empty_category_axis_warns_and_sits_at_the_centre(frame, leading):
    def annotated(coord: Any) -> fm.Chart:
        return fm.Chart(frame).mark_bar().encode(x="c", y="v") + ann.text(
            coord, 2.0, "HELLO", color=_COLOUR["text"], anchor="middle"
        )

    svg, shown = _render_capturing(annotated("b").to_svg)
    assert shown == [
        *leading,
        "annotation category 'b' not found in x-axis domain []; placing at plot center",
    ]
    centre_svg = _render_capturing(annotated(fm.NormCoord(0.5)).to_svg)[0]
    _near(_measured("text", svg, "x"), _measured("text", centre_svg, "x"))
