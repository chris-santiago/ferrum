"""An annotation with no colour given draws the theme's text colour on every route.

Every annotation primitive (hline, vline, abline, arrow, bracket, text, a span's label) used to
pick its unset colour in Python before the theme was known (`#333`, `black`) or from a mark's
own theme token, so one call drew a different colour depending on route, label and coordinate
type. The unset colour is now ``None`` on the wire and the renderer resolves it from the
effective theme's ``font_color``; an explicit colour always wins.

Each case is built twice on the same chart, once unset and once with an explicit colour that no
theme uses. The two renders lay out identically, so the only difference is the annotation's ink:
the theme's text colour must appear more often in the unset render than in the explicit one.
Text is also read directly, by the fill of the ``<text>`` that carries its string.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from tests.test_log_remedies_render import _interactive

_EXPLICIT = "#ff0010"
_CUSTOM_FONT = "#123456"
_THEMES = {
    "custom": (fm.Theme(font_color=_CUSTOM_FONT), _CUSTOM_FONT),
    "dark": (fm.themes.dark, "#e6e6e6"),
}
_DF = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})


def _base() -> fm.Chart:
    return fm.Chart(_DF).mark_point(color="#abcdef").encode(x="x", y="y")


def _count(svg: str, colour: str) -> int:
    return svg.lower().count(colour.lower())


def _text_fill(svg: str, string: str) -> str:
    """The fill of the single ``<text>`` element that carries *string*."""
    found = re.findall(rf'<text[^>]*fill="(#[0-9a-fA-F]{{6}})"[^>]*>{re.escape(string)}<', svg)
    assert len(found) == 1, f"expected one <text> reading {string!r}, got {found}"
    return found[0]


# id -> (build(colour) -> annotation chart or primitive, route, label string or None)
# `colour` is None for the unset call and `_EXPLICIT` for the explicit one.
_CASES: dict[str, tuple[Callable[[str | None], object], str, str | None]] = {
    "hline-plus": (lambda c: fm.annotate_hline(1.0, stroke=c), "plus", None),
    "hline-standalone": (lambda c: fm.annotate_hline(1.0, stroke=c), "standalone", None),
    "vline-plus": (lambda c: fm.annotate_vline(1.0, stroke=c), "plus", None),
    "vline-standalone": (lambda c: fm.annotate_vline(1.0, stroke=c), "standalone", None),
    "hline_label-plus": (lambda c: fm.annotate_hline(1.0, label="HLBL", stroke=c), "plus", "HLBL"),
    "hline_label-standalone": (
        lambda c: fm.annotate_hline(1.0, label="HLBL", stroke=c),
        "standalone",
        "HLBL",
    ),
    "arrow_numeric-plus": (lambda c: fm.annotate_arrow(0.2, 0.2, 1.5, 1.5, stroke=c), "plus", None),
    "arrow_numeric-standalone": (
        lambda c: fm.annotate_arrow(0.2, 0.2, 1.5, 1.5, stroke=c),
        "standalone",
        None,
    ),
    "arrow_norm-plus": (
        lambda c: fm.annotate_arrow(
            fm.norm(0.2), fm.norm(0.2), fm.norm(0.8), fm.norm(0.8), stroke=c
        ),
        "plus",
        None,
    ),
    "arrow_pixel-plus": (
        lambda c: fm.annotate_arrow(fm.px(60), fm.px(60), fm.px(200), fm.px(120), stroke=c),
        "plus",
        None,
    ),
    "arrow_labelled-plus": (
        lambda c: fm.annotate_arrow(0.2, 0.2, 1.5, 1.5, label="ALBL", stroke=c),
        "plus",
        "ALBL",
    ),
    "arrow_labelled-standalone": (
        lambda c: fm.annotate_arrow(0.2, 0.2, 1.5, 1.5, label="ALBL", stroke=c),
        "standalone",
        "ALBL",
    ),
    "abline-plus": (lambda c: fm.annotate_abline(1.0, 0.0, stroke=c), "plus", None),
    "abline-standalone": (lambda c: fm.annotate_abline(1.0, 0.0, stroke=c), "standalone", None),
    "text-plus": (lambda c: fm.annotate_text(1.0, 1.0, "TXT", color=c), "plus", "TXT"),
    "text-standalone": (lambda c: fm.annotate_text(1.0, 1.0, "TXT", color=c), "standalone", "TXT"),
    "prim_text-plus": (lambda c: ann.text(1.0, 1.0, "PTXT", color=c), "plus", "PTXT"),
    "prim_line-plus": (lambda c: ann.line(0.2, 0.2, 1.5, 1.5, stroke=c), "plus", None),
    "prim_arrow-plus": (lambda c: ann.arrow(0.2, 0.2, 1.5, 1.5, stroke=c), "plus", None),
    "bracket-plus": (
        lambda c: ann.bracket(0.2, 0.5, 1.5, 0.5, label="BRK", stroke=c),
        "plus",
        "BRK",
    ),
}


def _render(build: Callable[[str | None], object], route: str, colour: str | None, theme) -> str:
    annotation = build(colour)
    chart = _base() + annotation if route == "plus" else annotation
    return chart.theme(theme).to_svg()


@pytest.mark.parametrize("theme_id", sorted(_THEMES))
@pytest.mark.parametrize("case", sorted(_CASES))
def test_unset_colour_is_theme_font_color(case, theme_id):
    theme, font = _THEMES[theme_id]
    build, route, label = _CASES[case]
    unset = _render(build, route, None, theme)
    explicit = _render(build, route, _EXPLICIT, theme)
    if label is not None:
        assert _text_fill(unset, label).lower() == font
    # A primitive-only helper has no ink beyond its label to compare when the label is the
    # whole annotation, so the count comparison applies where a stroke is drawn.
    if not (label is not None and case.startswith(("text-", "prim_text"))):
        assert _count(unset, font) > _count(explicit, font), (
            f"{case}: the unset annotation does not draw the theme text colour {font}"
        )
    assert _count(explicit, _EXPLICIT) > 0


@pytest.mark.parametrize("case", sorted(c for c in _CASES if _CASES[c][1] == "plus"))
def test_explicit_colour_wins_on_every_route(case):
    build, route, label = _CASES[case]
    theme, font = _THEMES["custom"]
    explicit = _render(build, route, _EXPLICIT, theme)
    unannotated = _base().theme(theme).to_svg()
    assert _count(explicit, _EXPLICIT) > 0
    if label is None:
        # No label draws, so nothing in the annotation takes the theme colour.
        assert _count(explicit, font) == _count(unannotated, font)
    else:
        # The label of a stroke-only helper stays theme coloured; the text primitives take
        # the explicit colour for the string itself.
        expected = _EXPLICIT if case.startswith(("text-", "prim_text")) else font
        assert _text_fill(explicit, label).lower() == expected


def test_unset_colour_is_omitted_from_the_wire():
    unset = {
        "text": ann.text(1.0, 2.0, "t"),
        "arrow": ann.arrow(0, 0, 1, 1),
        "line": ann.line(0, 0, 1, 1),
        "bracket": ann.bracket(0, 0, 1, 0, label="b"),
    }
    key = {"text": "color", "arrow": "stroke", "line": "stroke", "bracket": "stroke"}
    for kind, primitive in unset.items():
        assert key[kind] not in primitive.to_dict(), kind
    assert ann.text(1.0, 2.0, "t", color="#111111").to_dict()["color"] == "#111111"
    assert ann.arrow(0, 0, 1, 1, stroke="#111111").to_dict()["stroke"] == "#111111"
    assert ann.line(0, 0, 1, 1, stroke="#111111").to_dict()["stroke"] == "#111111"
    assert ann.bracket(0, 0, 1, 0, label="b", stroke="#111111").to_dict()["stroke"] == "#111111"


def _tick_numbers(svg: str) -> list[float]:
    return [float(v) for v in re.findall(r">(-?\d+(?:\.\d+)?)</text>", svg)]


def test_unlabelled_numeric_arrow_keeps_widening_the_domain():
    """Colour moved off the mark route; the route's domain participation (spec 3.11) did not."""
    widened = (_base() + fm.annotate_arrow(0.1, 0.1, 9.0, 9.0)).to_svg()
    labelled = (_base() + fm.annotate_arrow(0.1, 0.1, 9.0, 9.0, label="L")).to_svg()
    assert max(_tick_numbers(widened)) >= 8.0
    assert max(_tick_numbers(labelled)) <= 3.0


def test_below_marks_text_follows_the_theme():
    theme, font = _THEMES["custom"]
    svg = (_base() + ann.text(1.0, 1.0, "BELOW", z="below_marks")).theme(theme).to_svg()
    assert _text_fill(svg, "BELOW").lower() == font


def test_span_label_follows_the_theme():
    theme, font = _THEMES["custom"]
    span = ann.span("x", 0.5, 1.5, fill="#ccddee", label="BAND")
    svg = (_base() + span).theme(theme).to_svg()
    assert _text_fill(svg, "BAND").lower() == font


def _new_colours(chart_with: fm.Chart, chart_without: fm.Chart) -> set[str]:
    token = re.compile(r"#[0-9a-fA-F]{6}")
    return set(token.findall(chart_with.to_svg().lower())) - set(
        token.findall(chart_without.to_svg().lower())
    )


_ELEMENT = re.compile(r"<(?:rect|line|path|text)[^>]*>")
_PAINT = re.compile(r'(fill|stroke)="(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\))"')


def _elements(svg: str) -> list[str]:
    return _ELEMENT.findall(re.sub(r"<style>.*?</style>", "", svg, flags=re.S))


def _new_elements(svg: str, plain: str) -> list[str]:
    drawn = set(_elements(plain))
    return [e for e in _elements(svg) if e not in drawn]


def _rgb(paint: str) -> tuple[int, int, int]:
    """The (r, g, b) of a hex or ``rgba(...)`` paint attribute value."""
    if paint.startswith("rgb"):
        r, g, b = (int(float(v)) for v in re.findall(r"[\d.]+", paint)[:3])
        return r, g, b
    hexed = paint.lstrip("#")
    return int(hexed[0:2], 16), int(hexed[2:4], 16), int(hexed[4:6], 16)


def _alpha(element: str, paint: str) -> float:
    if paint.startswith("rgba"):
        return float(re.findall(r"[\d.]+", paint)[3])
    found = re.search(r'fill-opacity="([\d.]+)"', element)
    return float(found.group(1)) if found else 1.0


def _theme_colours(theme: fm.Theme) -> tuple[str, str]:
    """The theme's text and background colours, read off a plain render."""
    text = (_base() + fm.annotate_text(1.0, 1.0, "T")).theme(theme).to_svg()
    font = _text_fill(text, "T").lower()
    background = re.search(r'<rect[^>]*fill="(#[0-9a-fA-F]{6})"', text)
    assert background is not None
    return font, background.group(1).lower()


_BUILTIN_THEMES = {
    name: getattr(fm.themes, name)
    for name in (
        "default",
        "arctic_signal",
        "dark",
        "economist",
        "fivethirtyeight",
        "minimal",
        "observable",
        "paper_ink",
        "publication",
        "slate_citrus",
        "solarized_dark",
        "solarized_light",
    )
}
_FILL_THEMES = {**_BUILTIN_THEMES, "custom": _THEMES["custom"][0]}


def _scene_nodes(chart: fm.Chart) -> list[dict]:
    """The annotation nodes of the interactive scene, and every node of a rect mark batch."""
    scene = json.loads(_interactive(chart))
    nodes: list[dict] = []
    for panel in scene["panels"]:
        nodes += panel.get("annotations") or []
        nodes += panel.get("below_marks") or []
        for batch in panel.get("marks") or []:
            if batch["kind"] == "rect":
                nodes += batch["nodes"]
    return nodes


def _hex(colour: dict | None) -> str | None:
    """The ``#rrggbb`` of a scene colour, or ``None`` for a colour the scene leaves unset."""
    if colour is None:
        return None
    return f"#{colour['r']:02x}{colour['g']:02x}{colour['b']:02x}"


def _filled_rects(
    chart: fm.Chart, theme: fm.Theme, output: str
) -> list[tuple[tuple[int, int, int], float]]:
    """The (rgb, opacity) of every annotation fill the chart adds to its plain twin."""
    if output == "interactive":
        out = []
        for node in _scene_nodes(chart.theme(theme)):
            fill = node["type"] == "rect" and node["style"]["fill"]
            if fill:
                alpha = fill["a"] / 255 * node["style"].get("fill_opacity", 1.0)
                out.append(((fill["r"], fill["g"], fill["b"]), alpha))
        return out
    svg = chart.theme(theme).to_svg()
    plain = _base().theme(theme).to_svg()
    out = []
    for element in _new_elements(svg, plain):
        if not element.startswith("<rect"):
            continue
        for kind, paint in _PAINT.findall(element):
            # A translucent fill is the annotation's; an opaque one is a panel background.
            if kind == "fill" and _alpha(element, paint) < 1:
                out.append((_rgb(paint), _alpha(element, paint)))
    return out


def _rect_routes() -> dict[str, Callable[[], fm.Chart]]:
    rect = lambda: fm.annotate_rect(0.5, 1.5, 1, 3)  # noqa: E731
    return {
        "plus": lambda: _base() + rect(),
        "standalone": rect,
        "hconcat": lambda: _base() | rect(),
        "vconcat": lambda: _base() & rect(),
        "layerchart": lambda: fm.LayerChart(_base(), rect()),
        "primitive": lambda: _base() + ann.rect(0.5, 1.5, 1, 3),
    }


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("route", sorted(_rect_routes()))
@pytest.mark.parametrize("theme_id", sorted(_FILL_THEMES))
def test_an_unfilled_rect_draws_the_theme_text_colour_at_its_opacity(route, theme_id, output):
    theme = _FILL_THEMES[theme_id]
    font, _bg = _theme_colours(theme)
    fills = _filled_rects(_rect_routes()[route](), theme, output)
    assert fills, f"{route}: no annotation fill was drawn"
    for rgb, alpha in fills:
        assert rgb == _rgb(font)
        assert alpha == pytest.approx(0.1, abs=0.005)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("theme_id", sorted(_FILL_THEMES))
def test_an_unfilled_span_draws_the_theme_text_colour_at_its_opacity(theme_id, output):
    theme = _FILL_THEMES[theme_id]
    font, _bg = _theme_colours(theme)
    fills = _filled_rects(_base() + ann.span("x", 0.5, 1.5), theme, output)
    assert fills
    for rgb, alpha in fills:
        assert rgb == _rgb(font)
        assert alpha == pytest.approx(0.3, abs=0.005)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("route", sorted(_rect_routes()))
def test_an_explicit_fill_wins_on_every_route(route, output):
    chart = {
        "plus": lambda: _base() + fm.annotate_rect(0.5, 1.5, 1, 3, fill=_EXPLICIT),
        "standalone": lambda: fm.annotate_rect(0.5, 1.5, 1, 3, fill=_EXPLICIT),
        "hconcat": lambda: _base() | fm.annotate_rect(0.5, 1.5, 1, 3, fill=_EXPLICIT),
        "vconcat": lambda: _base() & fm.annotate_rect(0.5, 1.5, 1, 3, fill=_EXPLICIT),
        "layerchart": lambda: fm.LayerChart(
            _base(), fm.annotate_rect(0.5, 1.5, 1, 3, fill=_EXPLICIT)
        ),
        "primitive": lambda: _base() + ann.rect(0.5, 1.5, 1, 3, fill=_EXPLICIT),
    }[route]()
    theme = _THEMES["custom"][0]
    fills = _filled_rects(chart, theme, output)
    assert fills
    assert {rgb for rgb, _ in fills} == {(255, 0, 16)}


def test_an_unset_fill_is_omitted_from_the_wire_and_none_is_the_same_as_omitted():
    assert "fill" not in ann.rect(0, 0, 1, 1).to_dict()
    assert "fill" not in ann.span("x", 0, 1).to_dict()
    assert ann.rect(0, 0, 1, 1, fill=None) == ann.rect(0, 0, 1, 1)
    assert ann.span("x", 0, 1, fill=None) == ann.span("x", 0, 1)
    assert ann.rect(0, 0, 1, 1, fill="#111111").to_dict()["fill"] == "#111111"
    assert ann.span("x", 0, 1, fill="#111111").to_dict()["fill"] == "#111111"


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_an_unfilled_primitive_rect_matches_the_helper_on_the_same_theme(output):
    theme = _THEMES["custom"][0]
    helper = _filled_rects(_base() + fm.annotate_rect(0.5, 1.5, 1, 3), theme, output)
    primitive = _filled_rects(_base() + ann.rect(0.5, 1.5, 1, 3), theme, output)
    assert {rgb for rgb, _ in helper} == {rgb for rgb, _ in primitive} == {_rgb(_CUSTOM_FONT)}


# Explicit fills draw the bytes they always drew: digests of ``_base() + <annotation>`` on the
# default theme.
_EXPLICIT_FILL_DIGESTS = {
    "rect": ("72884210a8a3", lambda: _base() + ann.rect(0.5, 1.5, 1, 3, fill="#123456")),
    "span": ("021ba557060a", lambda: _base() + ann.span("x", 0.5, 1.5, fill="#123456")),
    "helper": ("af66694de4f7", lambda: _base() + fm.annotate_rect(0.5, 1.5, 1, 3, fill="#123456")),
}


@pytest.mark.parametrize("carrier", sorted(_EXPLICIT_FILL_DIGESTS))
def test_an_explicit_fill_draws_the_bytes_it_always_drew(carrier):
    digest, build = _EXPLICIT_FILL_DIGESTS[carrier]
    assert hashlib.sha256(build().to_svg().encode()).hexdigest()[:12] == digest


_CALLOUT_BACKGROUND = "#ddeeff"
_CALLOUT_BORDER = "#1a56db"


def _callout_parts(chart: fm.Chart, theme: fm.Theme, output: str) -> dict[str, str | None]:
    """The paints of the callout the chart adds: box fill and stroke, leader stroke, text fill."""
    parts: dict[str, str | None] = {}
    if output == "interactive":
        for node in _scene_nodes(chart.theme(theme)):
            style = node["style"]
            if node["type"] == "rect" and node.get("corner_radius", 0) > 0:
                parts["box_fill"], parts["box_stroke"] = _hex(style["fill"]), _hex(style["stroke"])
            elif node["type"] == "line":
                parts["leader"] = _hex(style.get("color"))
            elif node["type"] == "text":
                parts["text"] = _hex(style.get("color"))
        return parts
    svg = chart.theme(theme).to_svg()
    plain = _base().theme(theme).to_svg()
    for element in _new_elements(svg, plain):
        paints = {kind: value.lower() for kind, value in _PAINT.findall(element)}
        if element.startswith("<rect") and "rx=" in element:
            parts["box_fill"], parts["box_stroke"] = paints.get("fill"), paints.get("stroke")
        elif element.startswith("<line"):
            parts["leader"] = paints.get("stroke")
        elif element.startswith("<text"):
            parts["text"] = paints.get("fill")
    return parts


def _callout(**colours: str | None) -> fm.Chart:
    return _base() + ann.callout(1.0, 1.0, "note", text_x=1.5, text_y=1.8, **colours)


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("theme_id", sorted(_FILL_THEMES))
def test_an_unstyled_callout_follows_the_theme(theme_id, output):
    theme = _FILL_THEMES[theme_id]
    font, background = _theme_colours(theme)
    assert _callout_parts(_callout(), theme, output) == {
        "box_fill": background,
        "box_stroke": font,
        "leader": font,
        "text": font,
    }


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_an_unstyled_callout_on_a_dark_theme_is_legible(output):
    font, background = _theme_colours(fm.themes.dark)
    parts = _callout_parts(_callout(), fm.themes.dark, output)
    assert (parts["box_fill"], parts["text"]) == (background, font)
    assert background != font


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_callout_with_only_a_border_colour_keeps_the_theme_box_and_text(output):
    theme = _THEMES["custom"][0]
    font, background = _theme_colours(theme)
    parts = _callout_parts(_callout(border_color=_CALLOUT_BORDER), theme, output)
    assert parts == {
        "box_fill": background,
        "box_stroke": _CALLOUT_BORDER,
        "leader": _CALLOUT_BORDER,
        "text": font,
    }


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_callout_with_only_a_background_keeps_the_theme_border_leader_and_text(output):
    theme = _THEMES["custom"][0]
    font, _background = _theme_colours(theme)
    parts = _callout_parts(_callout(background=_CALLOUT_BACKGROUND), theme, output)
    assert parts == {
        "box_fill": _CALLOUT_BACKGROUND,
        "box_stroke": font,
        "leader": font,
        "text": font,
    }


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_callout_with_both_colours_draws_its_text_in_the_theme_text_colour(output):
    theme = _THEMES["custom"][0]
    font, _background = _theme_colours(theme)
    parts = _callout_parts(
        _callout(background=_CALLOUT_BACKGROUND, border_color=_CALLOUT_BORDER), theme, output
    )
    assert parts == {
        "box_fill": _CALLOUT_BACKGROUND,
        "box_stroke": _CALLOUT_BORDER,
        "leader": _CALLOUT_BORDER,
        "text": font,
    }


@pytest.mark.parametrize("output", ["static", "interactive"])
def test_a_callout_colour_given_as_none_is_the_same_as_omitted(output):
    theme = _THEMES["custom"][0]
    assert ann.callout(1.0, 1.0, "n", background=None, border_color=None) == ann.callout(
        1.0, 1.0, "n"
    )
    draw = {
        "static": lambda chart: chart.theme(theme).to_svg(),
        "interactive": lambda chart: _interactive(chart.theme(theme)),
    }[output]
    assert draw(_callout(background=None, border_color=None)) == draw(_callout())


def test_an_unset_callout_colour_is_omitted_from_the_wire():
    unset = ann.callout(0, 0, "n").to_dict()
    assert "background" not in unset
    assert "border_color" not in unset
    both = ann.callout(0, 0, "n", background="#111111", border_color="#222222").to_dict()
    assert (both["background"], both["border_color"]) == ("#111111", "#222222")


def test_interactive_scene_annotation_colour_follows_the_theme():
    theme, _font = _THEMES["custom"]
    for annotation in (fm.annotate_hline(1.0), fm.annotate_text(1.0, 1.0, "hi")):
        nodes = _scene_nodes((_base() + annotation).theme(theme))
        assert [_hex(n["style"]["color"]) for n in nodes] == ["#123456"]
    explicit = _scene_nodes(_base() + fm.annotate_hline(1.0, stroke=_EXPLICIT))
    assert [_hex(n["style"]["color"]) for n in explicit] == [_EXPLICIT]


def test_composite_leaf_annotation_follows_its_own_theme():
    own = fm.Theme(font_color="#654321")
    annotated = _base().theme(own) + fm.annotate_hline(1.0)
    plain = _base()
    explicit = _base().theme(own) + fm.annotate_hline(1.0, stroke=_EXPLICIT)
    with_unset = (annotated | plain).to_svg()
    with_explicit = (explicit | plain).to_svg()
    assert _count(with_unset, "#654321") > _count(with_explicit, "#654321")


def test_theme_set_after_the_annotation_wins():
    chart = _base() + fm.annotate_hline(1.0)
    explicit = _base() + fm.annotate_hline(1.0, stroke=_EXPLICIT)
    theme, font = _THEMES["custom"]
    assert _count(chart.theme(theme).to_svg(), font) > _count(explicit.theme(theme).to_svg(), font)


def test_default_theme_context_reaches_annotations():
    chart = _base() + fm.annotate_hline(1.0)
    explicit = _base() + fm.annotate_hline(1.0, stroke=_EXPLICIT)
    theme, font = _THEMES["custom"]
    with fm.set_default_theme(theme):
        unset_svg, explicit_svg = chart.to_svg(), explicit.to_svg()
    assert _count(unset_svg, font) > _count(explicit_svg, font)
    assert _count(chart.to_svg(), font) == _count(explicit.to_svg(), font) == 0
