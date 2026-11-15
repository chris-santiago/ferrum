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


def test_callout_keeps_its_bubble_ink():
    """A callout's text sits on its own bubble, so its colours do not follow the theme."""
    callout = ann.callout(1.0, 1.0, "note", text_x=1.5, text_y=1.8)
    custom = _new_colours(
        (_base() + callout).theme(_THEMES["custom"][0]), _base().theme(_THEMES["custom"][0])
    )
    default = _new_colours((_base() + callout), _base())
    assert custom == default
    assert _CUSTOM_FONT not in custom


def _scene_annotation_colours(chart: fm.Chart) -> list[tuple[int, int, int]]:
    scene = json.loads(_interactive(chart))
    out = []
    for panel in scene["panels"]:
        for node in panel.get("annotations") or []:
            c = node["style"]["color"]
            out.append((c["r"], c["g"], c["b"]))
    return out


def test_interactive_scene_annotation_colour_follows_the_theme():
    theme, _font = _THEMES["custom"]
    rgb = (0x12, 0x34, 0x56)
    for annotation in (fm.annotate_hline(1.0), fm.annotate_text(1.0, 1.0, "hi")):
        colours = _scene_annotation_colours((_base() + annotation).theme(theme))
        assert colours == [rgb]
    explicit = _scene_annotation_colours((_base() + fm.annotate_hline(1.0, stroke=_EXPLICIT)))
    assert explicit == [(255, 0, 16)]


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
