"""Selection value routing through the one Rust color parser (Batch A, task 10).

``fm.value(...)`` and ``SelectionMark(fill=..., stroke=...)`` used to parse
colors through two hand-rolled hex-only parsers in ``ferrum/selection.py``:
``_hex_to_color_dict`` warned and silently fell back to black on anything
that wasn't a 6/8-digit hex string, and ``_resolve_encoding_value`` silently
treated any non-``#``-prefixed string as opacity 1.0. Both parsers now route
through ``ferrum.color.to_hex`` (the single Rust color parser): a parseable
string (CSS name, hex, ``rgb()``/``rgba()``) resolves to a color dict, a
number still resolves to the appropriate numeric wire kind (opacity by
default), and an unparseable string raises ``ValueError`` naming the
accepted forms — no warn-and-fallback anywhere in this path.

See ``.claude/output/specs/2026-08-28-batch-a-appearance-resolution-design.md``
§4.1 and NF-A4 in the frozen findings ledger.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import render_interactive
from ferrum.selection import (
    SelectionMark,
    _hex_to_color_dict,
    _resolve_encoding_value,
    selection_point,
    value,
)


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 1.0],
            "group": ["a", "b", "a"],
        }
    )


# ── _hex_to_color_dict: named colors resolve, hex stays byte-identical ────────


def test_hex_to_color_dict_named_css_color_resolves():
    """A CSS name (previously silently warned-and-black'd) now resolves for real."""
    assert _hex_to_color_dict("lightgray", context="test") == {
        "r": 211,
        "g": 211,
        "b": 211,
        "a": 255,
    }


def test_hex_to_color_dict_hex_byte_identical():
    """Hex input still resolves to the exact rgb the old hand-rolled parser gave."""
    assert _hex_to_color_dict("#d3d3d3", context="test") == {
        "r": 211,
        "g": 211,
        "b": 211,
        "a": 255,
    }
    assert _hex_to_color_dict("#fff", context="test") == {
        "r": 255,
        "g": 255,
        "b": 255,
        "a": 255,
    }
    assert _hex_to_color_dict("#d3d3d3ff", context="test") == {
        "r": 211,
        "g": 211,
        "b": 211,
        "a": 255,
    }


def test_hex_to_color_dict_rgba_alpha_preserved():
    assert _hex_to_color_dict("rgba(211, 211, 211, 0.5)", context="test") == {
        "r": 211,
        "g": 211,
        "b": 211,
        "a": 128,
    }


def test_hex_to_color_dict_unparseable_raises_naming_accepted_forms():
    with pytest.raises(ValueError, match="expected a CSS color name"):
        _hex_to_color_dict("nonsense", context="test")


def test_hex_to_color_dict_unparseable_message_carries_caller_context():
    """The generic accepted-forms message is prefixed with the caller's
    *context* (mirrors marks/base.py's _validate_literal_color prefix shape),
    so a bad literal several frames inside .to_svg() is traceable back to the
    call that produced it — not just an anonymous 'invalid color' message."""
    with pytest.raises(
        ValueError,
        match=r"SelectionMark: fill='nonsense' is not a valid color \(invalid color 'nonsense'",
    ):
        _hex_to_color_dict("nonsense", context="SelectionMark: fill='nonsense'")


@pytest.mark.parametrize(
    "spelling",
    [
        "none",
        "None",
        "NONE",
        " none ",
        "transparent",
        "Transparent",
        "TRANSPARENT",
        " transparent ",
    ],
)
def test_hex_to_color_dict_none_gets_dedicated_refusal_message(spelling):
    """Spec §4.1 amendment (2026-09-01): 'none' stays a refusal on the
    selection surface (the wire {r,g,b,a} dict has no cleared-paint
    representation), but it gets its own message rather than the generic
    accepted-forms text — it's a real color-clearing request this surface
    can't fulfil, not an unrecognized vocabulary item. The match is trimmed
    and case-insensitive (the canonical sentinel spelling, matching Rust's
    draw.rs and marks/base.py's gate) — "none", "None", "NONE", and
    " none " all hit the dedicated message. The dedicated message keeps the
    caller's context prefix too.

    Spec §4.1 (superseded 2026-09-01, T8 quality review): 'transparent'
    joins 'none' as a clearing spelling via the shared
    ``is_none_color_sentinel`` predicate, so it hits the same dedicated
    refusal on this surface (the selection wire still can't express a
    cleared paint), trimmed and case-insensitive identically to 'none'."""
    with pytest.raises(ValueError, match="test: selection styling cannot express a cleared paint"):
        _hex_to_color_dict(spelling, context="test")


def test_hex_to_color_dict_other_unparseable_strings_keep_generic_message():
    """Only a string matching the paint-clear sentinel predicate
    (``ferrum._validate.is_none_color_sentinel`` — currently 'none'/
    'transparent', trimmed and case-folded) gets the dedicated message;
    every other unparseable string, including any spelling the predicate
    doesn't match, still gets the generic accepted-forms text."""
    with pytest.raises(ValueError, match="expected a CSS color name") as excinfo:
        _hex_to_color_dict("nonsense", context="test")
    assert "cannot express" not in str(excinfo.value)


# ── value(): string -> color, number -> opacity (unchanged), else raises ──────


def test_value_lightgray_is_color():
    """fm.value('lightgray') is a color, not a silent opacity-1.0 fallback."""
    resolved = _resolve_encoding_value(value("lightgray"))
    assert resolved == {"kind": "color", "value": {"r": 211, "g": 211, "b": 211, "a": 255}}


def test_value_hex_byte_identical():
    resolved = _resolve_encoding_value(value("#d3d3d3"))
    assert resolved == {"kind": "color", "value": {"r": 211, "g": 211, "b": 211, "a": 255}}


def test_value_number_opacity_byte_identical():
    resolved = _resolve_encoding_value(value(0.5))
    assert resolved == {"kind": "opacity", "value": 0.5}


def test_value_number_respects_explicit_channel_unchanged():
    resolved = _resolve_encoding_value(value(2.0), channel="stroke_width")
    assert resolved == {"kind": "stroke_width", "value": 2.0}


def test_value_nonsense_string_raises_valueerror_naming_accepted_forms():
    with pytest.raises(ValueError, match="expected a CSS color name"):
        _resolve_encoding_value(value("nonsense"))


def test_value_nonsense_string_message_carries_value_and_channel_context():
    """_resolve_encoding_value threads the fm.value(...) literal and its
    resolved channel into the message, so a bad literal raised many frames
    inside .to_svg() is traceable back to which fm.value(...) call and
    which conditional channel produced it."""
    with pytest.raises(
        ValueError, match=r"fm\.value\('nonsense'\) for channel='opacity' is not a valid color"
    ):
        _resolve_encoding_value(value("nonsense"), channel="opacity")


def test_value_rgb_functional_form_is_color():
    resolved = _resolve_encoding_value(value("rgb(70, 130, 180)"))
    assert resolved["kind"] == "color"
    assert resolved["value"] == {"r": 70, "g": 130, "b": 180, "a": 255}


def test_value_non_str_non_number_raises_typeerror():
    """Neither a color string nor a number: no silent opacity-1.0 fallback."""
    with pytest.raises(TypeError, match="color string or a number"):
        _resolve_encoding_value(value([1, 2, 3]))


# ── SelectionMark: fill/stroke route through the same parser ──────────────────


def test_selection_mark_named_color_fill():
    mark = SelectionMark(fill="lightgray")
    d = mark.to_spec_dict()
    assert d["fill"] == {"r": 211, "g": 211, "b": 211, "a": 255}


def test_selection_mark_hex_fill_byte_identical():
    mark = SelectionMark(fill="#4287f5")
    d = mark.to_spec_dict()
    assert d["fill"] == {"r": 0x42, "g": 0x87, "b": 0xF5, "a": 255}


def test_selection_mark_unparseable_stroke_raises():
    mark = SelectionMark(stroke="nonsense")
    with pytest.raises(ValueError, match="expected a CSS color name"):
        mark.to_spec_dict()


def test_selection_mark_unparseable_stroke_message_carries_context():
    """SelectionMark.to_spec_dict threads which key/value failed into the
    message (mirrors marks/base.py's construction-time color error shape)."""
    mark = SelectionMark(stroke="nonsense")
    with pytest.raises(ValueError, match=r"SelectionMark: stroke='nonsense' is not a valid color"):
        mark.to_spec_dict()


def test_selection_mark_bare_hex_no_longer_resolves():
    """Behavior change (disclosed in the task report's byte-identity
    section): the old hand-rolled parser did ``hex_str.lstrip('#')``, so a
    bare (no leading '#') hex string like 'ffffff' resolved anyway. Routing
    through ``ferrum.color.to_hex`` — the one Rust parser every other color
    boundary uses — no longer accepts bare hex; it must be '#'-prefixed like
    everywhere else. All in-repo/in-docs SelectionMark call sites already use
    '#'-prefixed hex, so this does not break any known caller."""
    mark = SelectionMark(fill="ffffff")
    with pytest.raises(ValueError, match="expected a CSS color name"):
        mark.to_spec_dict()


# ── End-to-end: conditional encoding through the full interactive render ──────


def _render(chart: fm.Chart) -> tuple[str, bytes]:
    spec, data, viewport, theme, _ = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme)


def test_conditional_named_color_reaches_scene(df):
    sel = selection_point(fields=["group"], name="named_color_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("lightgray"))
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").add_selection(sel).conditional(cond)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 1
    assert conditionals[0]["if_not"] == {
        "kind": "color",
        "value": {"r": 211, "g": 211, "b": 211, "a": 255},
    }


def test_conditional_hex_color_scene_byte_identical(df):
    """Hex-string conditional still serializes to the pre-existing shape."""
    sel = selection_point(fields=["group"], name="hex_color_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").add_selection(sel).conditional(cond)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert conditionals[0]["if_not"] == {
        "kind": "color",
        "value": {"r": 0xCC, "g": 0xCC, "b": 0xCC, "a": 255},
    }


def test_conditional_opacity_number_scene_byte_identical(df):
    sel = selection_point(fields=["group"], name="opacity_sel")
    cond = sel.when(fm.Opacity("group")).otherwise(value(0.2))
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").add_selection(sel).conditional(cond)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert conditionals[0]["if_not"] == {"kind": "opacity", "value": 0.2}


def test_conditional_unparseable_value_raises_at_spec_build_time(df):
    """The failure surfaces in Python at spec-construction time, not deferred
    to a Rust boundary error or silently swallowed into a render."""
    sel = selection_point(fields=["group"], name="bad_color_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("not-a-real-color"))
    with pytest.raises(ValueError, match="expected a CSS color name"):
        cond.to_spec_dict()
