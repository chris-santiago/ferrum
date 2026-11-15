"""An unparseable annotation colour is refused at construction, in the call the caller wrote.

Every colour field of every annotation primitive passes the gate the marks' literal fill and
stroke pass: ``None`` and the sentinels ``"none"``/``"transparent"`` are accepted, a non-string
is refused as the marks refuse it, and any other string must parse as a colour. An
``annotate_*`` helper re-voices every refusal it raises as itself. A colour that reaches the
renderer by the wire (a primitive mutated after construction) is refused there; nothing paints
a fallback colour.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from ferrum._entry import EntryRefusal
from ferrum.color import to_hex

_BAD = "notacolor"
_DF = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})


def _parser_message(value: str) -> str:
    try:
        to_hex(value)
    except ValueError as err:
        return str(err)
    raise AssertionError(f"{value!r} parsed as a colour")


def _not_valid(call: str, param: str, value: str = _BAD) -> str:
    return f"{call}: {param}={value!r} is not a valid color ({_parser_message(value)})"


def _base() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y")


# kind -> (factory taking one colour keyword value, its colour parameter names)
_PRIMITIVES: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {
    "text": (lambda **kw: ann.text(1.0, 1.0, "t", **kw), ("color",)),
    "arrow": (lambda **kw: ann.arrow(0.0, 0.0, 1.0, 1.0, **kw), ("stroke",)),
    "rect": (
        lambda **kw: ann.rect(0.0, 0.0, 1.0, 1.0, **{"fill": "red", **kw}),
        ("fill", "stroke"),
    ),
    "line": (lambda **kw: ann.line(0.0, 0.0, 1.0, 1.0, **kw), ("stroke",)),
    "span": (lambda **kw: ann.span("x", 0.0, 1.0, **{"fill": "red", **kw}), ("fill",)),
    "bracket": (lambda **kw: ann.bracket(0.0, 0.0, 1.0, 1.0, label="b", **kw), ("stroke",)),
    "callout": (
        lambda **kw: ann.callout(1.0, 1.0, "t", **kw),
        ("background", "border_color"),
    ),
}
_FIELDS = [(kind, param) for kind, (_, params) in _PRIMITIVES.items() for param in params]


@pytest.mark.parametrize(("kind", "param"), _FIELDS)
def test_a_primitive_refuses_an_unparseable_colour_naming_its_call_and_parameter(kind, param):
    build = _PRIMITIVES[kind][0]
    with pytest.raises(EntryRefusal) as err:
        build(**{param: _BAD})
    assert str(err.value) == _not_valid(f"annotation.{kind}", param)


@pytest.mark.parametrize(("kind", "param"), _FIELDS)
@pytest.mark.parametrize("value", [3, (1, 0, 0)])
def test_a_primitive_refuses_a_non_string_colour_as_the_marks_do(kind, param, value):
    build = _PRIMITIVES[kind][0]
    with pytest.raises(EntryRefusal) as err:
        build(**{param: value})
    assert str(err.value) == f"annotation.{kind}: {param} must be a string; got {value!r}"


def test_the_primitive_refusal_is_the_one_the_mark_door_raises_for_the_same_colour():
    with pytest.raises(EntryRefusal) as mark:
        fm.Chart(_DF).mark_rule(stroke=_BAD)
    with pytest.raises(EntryRefusal) as primitive:
        ann.line(0.0, 0.0, 1.0, 1.0, stroke=_BAD)
    assert str(mark.value) == _not_valid("mark_rule", "stroke")
    assert str(primitive.value) == _not_valid("annotation.line", "stroke")


def test_the_refusal_survives_pickling_and_can_be_revoiced():
    with pytest.raises(EntryRefusal) as err:
        ann.rect(0.0, 0.0, 1.0, 1.0, fill=_BAD)
    copy = pickle.loads(pickle.dumps(err.value))
    assert str(copy) == str(err.value)
    assert str(copy.spoken_as("annotate_rect")) == _not_valid("annotate_rect", "fill")


# Accepted colour -> digest of the SVG of ``_base() + <primitive carrying it>``, measured on the
# build before the renderer's colour change, so the accepted values draw as they always did.
_ACCEPTED: dict[str, dict[str, str]] = {
    "none": {"rect": "4b1055a9529e", "text": "ee68431c4553", "callout": "1711c99db57d"},
    "transparent": {"rect": "4b1055a9529e", "text": "ee68431c4553", "callout": "1711c99db57d"},
    "#123": {"rect": "4938fdba6bd8", "text": "a6f6c5436b5f", "callout": "3d0d697243bd"},
    "#11223344": {"rect": "18f7706cab4d", "text": "d6bf1f2bba4e", "callout": "01499f33898d"},
    "rgb(1,2,3)": {"rect": "880486e99256", "text": "003165175c47", "callout": "2981b6dff3bd"},
    "red": {"rect": "194db27b6039", "text": "4384e87cceb4", "callout": "d377e513b754"},
}
_CARRIERS: dict[str, Callable[[str], Any]] = {
    "rect": lambda v: ann.rect(0.0, 0.0, 1.0, 1.0, fill=v),
    "text": lambda v: ann.text(1.0, 1.0, "t", color=v),
    "callout": lambda v: ann.callout(1.0, 1.0, "t", background=v, border_color=v),
}


@pytest.mark.parametrize("value", sorted(_ACCEPTED))
@pytest.mark.parametrize("carrier", sorted(_CARRIERS))
def test_an_accepted_colour_draws_the_bytes_it_always_drew(value, carrier):
    svg = (_base() + _CARRIERS[carrier](value)).to_svg()
    assert hashlib.sha256(svg.encode()).hexdigest()[:12] == _ACCEPTED[value][carrier]


@pytest.mark.parametrize(("kind", "param"), _FIELDS)
@pytest.mark.parametrize("value", ["none", "transparent", "#123", "#11223344", "rgb(1,2,3)", "red"])
def test_every_colour_field_accepts_the_sentinels_and_every_colour_form(kind, param, value):
    _PRIMITIVES[kind][0](**{param: value})


# The colour fields whose value may be left unset.
_OPTIONAL = [("text", "color"), ("arrow", "stroke"), ("rect", "stroke"), ("line", "stroke")]


@pytest.mark.parametrize(("kind", "param"), _OPTIONAL)
def test_an_unset_colour_is_accepted_where_the_field_is_optional(kind, param):
    primitive = _PRIMITIVES[kind][0](**{param: None})
    assert getattr(primitive, param) is None


# helper -> (call that passes one colour keyword, the keyword the caller wrote)
_HELPERS: dict[str, tuple[Callable[..., Any], str]] = {
    "annotate_hline": (lambda **kw: fm.annotate_hline(1.5, **kw), "stroke"),
    "annotate_vline": (lambda **kw: fm.annotate_vline(1.5, **kw), "stroke"),
    "annotate_rect": (lambda **kw: fm.annotate_rect(0.0, 1.0, 0.0, 1.0, **kw), "fill"),
    "annotate_text": (lambda **kw: fm.annotate_text(1.0, 1.0, "t", **kw), "color"),
    "annotate_abline": (lambda **kw: fm.annotate_abline(1.0, 0.0, **kw), "stroke"),
    "annotate_arrow": (lambda **kw: fm.annotate_arrow(0.0, 0.0, 1.0, 1.0, **kw), "stroke"),
}


@pytest.mark.parametrize("helper", sorted(_HELPERS))
def test_a_helper_refuses_an_unparseable_colour_naming_the_helper(helper):
    call, param = _HELPERS[helper]
    with pytest.raises(EntryRefusal) as err:
        call(**{param: _BAD})
    assert str(err.value) == _not_valid(helper, param)


@pytest.mark.parametrize("helper", sorted(_HELPERS))
def test_a_helper_refuses_a_non_string_colour_naming_the_helper(helper):
    call, param = _HELPERS[helper]
    with pytest.raises(EntryRefusal) as err:
        call(**{param: 3})
    assert str(err.value) == f"{helper}: {param} must be a string; got 3"


@pytest.mark.parametrize("helper", sorted(_HELPERS))
@pytest.mark.parametrize("value", ["none", "transparent", "red", "#123", "rgb(1,2,3)"])
def test_a_helper_accepts_the_sentinels_and_every_colour_form(helper, value):
    call, param = _HELPERS[helper]
    call(**{param: value})


_DASH_TEXT = (
    "{helper}: stroke_dash='zz' is not a valid stroke_dash value. Expected a numeric list "
    '(e.g. [4.0, 2.0]), a comma-separated numeric string (e.g. "4,2"), or a named linetype '
    "(one of ['dashdot', 'dashed', 'dotted', 'longdash', 'solid'])."
)


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        ("annotate_hline", lambda: fm.annotate_hline(1.5, stroke_dash="zz")),
        ("annotate_vline", lambda: fm.annotate_vline(1.5, stroke_dash="zz")),
        ("annotate_abline", lambda: fm.annotate_abline(1.0, 0.0, stroke_dash="zz")),
    ],
)
def test_a_helper_stroke_dash_refusal_keeps_its_text(helper, call):
    with pytest.raises(ValueError) as err:
        call()
    assert str(err.value) == _DASH_TEXT.format(helper=helper)


_NAT = "np.datetime64('NaT') is NaT (not-a-time), which is not a valid temporal value"
_FLOAT = "float() argument must be a string or a real number, not 'object'"


@pytest.mark.parametrize(
    ("call", "error", "text"),
    [
        (lambda: fm.annotate_hline(np.datetime64("NaT")), ValueError, _NAT),
        (lambda: fm.annotate_vline(np.datetime64("NaT")), ValueError, _NAT),
        (lambda: fm.annotate_rect(object(), 1, 0, 1), TypeError, _FLOAT),
        (lambda: fm.annotate_text(object(), 1, "t"), TypeError, _FLOAT),
        (lambda: fm.annotate_arrow(object(), 0, 1, 1), TypeError, _FLOAT),
        (
            lambda: fm.annotate_abline("a", 0),
            TypeError,
            "can't multiply sequence by non-int of type 'float'",
        ),
    ],
    ids=["hline", "vline", "rect", "text", "arrow", "abline"],
)
def test_a_helper_coordinate_refusal_keeps_its_text(call, error, text):
    with pytest.raises(error) as err:
        call()
    assert str(err.value) == text


@pytest.mark.parametrize("token", ["theme:font", "theme:label"])
@pytest.mark.parametrize(("kind", "param"), _FIELDS)
def test_a_primitive_refuses_an_internal_theme_token_the_renderer_does_not_draw(kind, param, token):
    with pytest.raises(EntryRefusal) as err:
        _PRIMITIVES[kind][0](**{param: token})
    assert str(err.value) == _not_valid(f"annotation.{kind}", param, token)


@pytest.mark.parametrize("token", ["theme:font", "theme:label"])
@pytest.mark.parametrize("helper", sorted(_HELPERS))
def test_a_helper_refuses_an_internal_theme_token_on_every_route(helper, token):
    call, param = _HELPERS[helper]
    with pytest.raises(EntryRefusal) as err:
        call(**{param: token})
    assert str(err.value) == _not_valid(helper, param, token)


_THEME_INK = "#123456"
_EXPLICIT_INK = "#ff0010"


def _ink_count(svg: str, colour: str) -> int:
    return svg.lower().count(colour.lower())


# annotate_rect is absent: its unset fill is the rect mark's own default colour, not the text ink.
_INKED_HELPERS = sorted(set(_HELPERS) - {"annotate_rect"})


@pytest.mark.parametrize("route", ["standalone", "overlaid"])
@pytest.mark.parametrize("helper", _INKED_HELPERS)
def test_a_helper_with_no_colour_draws_the_theme_ink(helper, route):
    call, param = _HELPERS[helper]
    # The explicit colour differs from the theme ink, so each of the two colours found in
    # the output is attributable to one source.
    theme = fm.Theme(font_color=_THEME_INK)

    def draw(**colour: str) -> str:
        chart = call(**colour)
        return ((_base() + chart) if route == "overlaid" else chart).theme(theme).to_svg()

    unset, explicit = draw(), draw(**{param: _EXPLICIT_INK})
    assert _ink_count(unset, _THEME_INK) > _ink_count(explicit, _THEME_INK)
    assert _ink_count(explicit, _EXPLICIT_INK) > _ink_count(unset, _EXPLICIT_INK) == 0


@pytest.mark.parametrize(("kind", "param"), _FIELDS)
@pytest.mark.parametrize("output", ["static", "interactive"])
def test_an_accepted_colour_draws_the_same_bytes_in_every_spelling(kind, param, output):
    def draw(value: str) -> str:
        chart = _base() + _PRIMITIVES[kind][0](**{param: value})
        return chart.to_svg() if output == "static" else chart.interactive().scene_json

    assert draw(" NONE ") == draw("none") == draw("Transparent") == draw("transparent")
    assert draw("none") != draw("red")
    assert draw("red") == draw("#ff0000") == draw("rgb(255,0,0)")


def test_a_helper_refusal_that_is_not_a_colour_is_voiced_as_the_helper():
    with pytest.raises(EntryRefusal) as err:
        fm.annotate_hline(1.5, stroke=3)
    assert str(err.value) == "annotate_hline: stroke must be a string; got 3"
    with pytest.raises(ValueError) as unmoved:
        fm.annotate_text(1.0, 1.0, "t", align="zz")
    assert (
        str(unmoved.value)
        == "annotate_text: align must be one of ['center', 'left', 'right']; got 'zz'"
    )


def _corrupted(kind: str, param: str) -> Any:
    """A primitive built with a valid colour whose *param* is then set to the bad one.

    Construction refuses the bad colour, so this is the only way to put one on the wire.
    """
    primitive = _PRIMITIVES[kind][0](**{param: "red"})
    object.__setattr__(primitive, param, _BAD)
    return primitive


@pytest.mark.parametrize(("kind", "param"), _FIELDS)
def test_a_colour_that_reaches_the_renderer_unparsed_is_refused_not_painted(kind, param):
    chart = _base() + _corrupted(kind, param)
    message = f"annotation {kind}: {param}={_BAD!r} is not a valid color ({_parser_message(_BAD)})"
    with pytest.raises(ValueError) as static:
        chart.to_svg()
    assert str(static.value) == message
    with pytest.raises(ValueError) as interactive:
        chart.interactive().to_html()
    assert str(interactive.value) == message
