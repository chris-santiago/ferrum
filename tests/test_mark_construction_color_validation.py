"""Construction-time color validation + stroke_dash comma-split (Batch A, T9).

``MarkBase.__init__`` validates literal ``fill=``/``stroke=``/``color=``
string kwargs through ``ferrum.color.to_hex`` — ferrum's single Rust color
parser — so a bad color string raises ``ValueError`` immediately at
construction instead of silently reaching the renderer. The non-color
sentinels ``"none"``/``"transparent"`` (explicit paint-clear) and
``"theme:label"`` (an internal theme-lookup token, see
``marks/composite.py``) are checked BEFORE ``to_hex`` and never treated as
invalid colors — ``"theme:label"`` raises inside ``to_hex`` by design (see
``tests/test_color_vocabulary.py::TestSentinelsAreNotColors``), so the
short-circuit in ``MarkBase.__init__`` must run first. ``to_hex`` itself
keeps raising on a bare ``"transparent"`` (it is a conversion utility, and
the clearing sentinels stay out of the parser vocabulary by design — the
refusal is sentinel-aware, not a claim that ``transparent`` lacks a hex
form) — this short-circuit
is what keeps ``fill=`` construction from ever reaching that raise.

This module also pins the ``stroke_dash="4,2"`` comma-split: the same
comma-split ``linetype=``/``line_type=`` already used for named dash forms
now applies directly to the canonical ``stroke_dash=`` kwarg's string form,
while list/tuple and named-linetype forms stay unchanged.

Sentinel matching mirrors ``resolve_paint_color`` in
``crates/ferrum-core/src/render/draw.rs`` exactly (spec-reviewer finding,
2026-09-01): ``"none"``/``"transparent"`` are trimmed and case-insensitive
(``trimmed.eq_ignore_ascii_case("none") || trimmed.eq_ignore_ascii_case("transparent")``
in Rust), so ``"None"``, ``"NONE"``, ``" none "``, ``"Transparent"``,
``"TRANSPARENT"``, and ``" transparent "`` must all construct without
raising; but ``"theme:label"`` is an exact-string match in Rust, so any
other casing or whitespace variant is NOT a recognized sentinel and falls
through to ``to_hex``, which rejects it as an invalid color.

``"transparent"`` joining ``"none"`` as a clearing spelling is a documented
contract change (spec §4.1, superseded 2026-09-01 T8 quality review):
before this change, ``fill="transparent"`` hard-failed construction with a
self-contradicting message (``transparent`` IS a CSS Color 4 keyword); now
it clears paint identically to ``"none"``, in both languages, via the
shared ``ferrum._validate.is_none_color_sentinel`` predicate.

See ``.claude/output/specs/2026-08-28-batch-a-appearance-resolution-design.md``
§4.1 (color parsing) and §4.4 (the ``"a,b"`` comma-split bullet).
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.marks.base import MarkBase


# ---------------------------------------------------------------------------
# Construction-time color validation
# ---------------------------------------------------------------------------


class TestConstructionColorValidation:
    def test_invalid_fill_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="CSS color name"):
            MarkBase("point", fill="not-a-color")

    def test_invalid_stroke_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="CSS color name"):
            MarkBase("line", stroke="not-a-color")

    def test_invalid_color_alias_resolves_to_fill_for_fill_primary_mark(self) -> None:
        # ``point`` is fill-primary: color= -> fill=, validated as fill.
        with pytest.raises(ValueError, match="CSS color name"):
            MarkBase("point", color="not-a-color")

    def test_invalid_color_alias_resolves_to_stroke_for_stroke_primary_mark(self) -> None:
        # ``line`` is stroke-primary: color= -> stroke=, validated as stroke.
        with pytest.raises(ValueError, match="CSS color name"):
            MarkBase("line", color="not-a-color")

    def test_error_message_names_accepted_forms(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"expected a CSS color name, #rrggbb\[aa\]/#rgb\[a\] hex, or rgb\(\)/rgba\(\)",
        ):
            MarkBase("bar", fill="bogus")

    def test_error_message_names_offending_mark_and_key(self) -> None:
        with pytest.raises(ValueError, match=r"mark_bar: fill='bogus'"):
            MarkBase("bar", fill="bogus")

    @pytest.mark.parametrize(
        "value",
        ["steelblue", "#4682b4", "#4682b4aa", "rgb(70, 130, 180)", "rgba(70, 130, 180, 0.5)"],
    )
    def test_valid_color_forms_pass_construction(self, value: str) -> None:
        mb = MarkBase("point", fill=value)
        assert mb.kwargs["fill"] == value

    def test_original_string_preserved_not_normalized_to_hex(self) -> None:
        # Storage keeps the user's literal spelling; normalization to hex
        # happens in Rust, not here (spec §4.1: "the stored value remains
        # the user's original string").
        mb = MarkBase("point", fill="STEELBLUE")
        assert mb.kwargs["fill"] == "STEELBLUE"

    def test_non_string_fill_is_not_validated(self) -> None:
        # Only literal string fill/stroke values are validated; a non-string
        # value must construct and round-trip unchanged rather than being
        # routed into the color parser. This pins the
        # `isinstance(paint_val, str)` guard in MarkBase.__init__: without
        # it, `_is_paint_sentinel(0)` (or `to_hex(0)`, once the sentinel
        # check is bypassed) raises an unwrapped low-level exception
        # instead of constructing cleanly (quality-reviewer finding).
        mb = MarkBase("point", fill=0)
        assert mb.kwargs["fill"] == 0


# ---------------------------------------------------------------------------
# Sentinel short-circuit ("none", "theme:label")
# ---------------------------------------------------------------------------


class TestSentinelShortCircuit:
    def test_fill_none_does_not_raise(self) -> None:
        mb = MarkBase("point", fill="none")
        assert mb.kwargs["fill"] == "none"

    def test_stroke_none_does_not_raise(self) -> None:
        mb = MarkBase("line", stroke="none")
        assert mb.kwargs["stroke"] == "none"

    @pytest.mark.parametrize("value", ["None", "NONE", " none ", "  NoNe  ", "none "])
    def test_none_is_trimmed_and_case_insensitive_like_rust(self, value: str) -> None:
        # draw.rs's resolve_paint_color: value.trim().eq_ignore_ascii_case("none").
        # The Python construction-time gate must accept the identical set of
        # spellings, or a value the renderer would happily clear paint for
        # raises ValueError at construction instead (spec-reviewer finding).
        mb = MarkBase("point", fill=value)
        assert mb.kwargs["fill"] == value  # original spelling preserved, not normalized

    def test_fill_transparent_does_not_raise(self) -> None:
        # "transparent" joins "none" as a clearing spelling (spec §4.1,
        # superseded 2026-09-01 T8 quality review): before this change this
        # hard-failed construction with a self-contradicting message, since
        # "transparent" IS a real CSS Color 4 keyword.
        mb = MarkBase("point", fill="transparent")
        assert mb.kwargs["fill"] == "transparent"

    def test_stroke_transparent_does_not_raise(self) -> None:
        mb = MarkBase("line", stroke="transparent")
        assert mb.kwargs["stroke"] == "transparent"

    @pytest.mark.parametrize(
        "value", ["Transparent", "TRANSPARENT", " transparent ", "  TranspareNT  ", "transparent "]
    )
    def test_transparent_is_trimmed_and_case_insensitive_like_rust(self, value: str) -> None:
        # draw.rs's resolve_paint_color also matches
        # trimmed.eq_ignore_ascii_case("transparent"); the Python
        # construction-time gate must accept the identical set of spellings.
        mb = MarkBase("point", fill=value)
        assert mb.kwargs["fill"] == value  # original spelling preserved, not normalized

    def test_color_alias_transparent_does_not_raise(self) -> None:
        mb = MarkBase("point", color="transparent")
        assert mb.kwargs["fill"] == "transparent"

    def test_stroke_theme_label_sentinel_does_not_raise(self) -> None:
        # Mirrors marks/composite.py's internal boxplot whisker/cap styling.
        mb = MarkBase("rule", stroke="theme:label")
        assert mb.kwargs["stroke"] == "theme:label"

    def test_fill_theme_label_sentinel_does_not_raise(self) -> None:
        mb = MarkBase("point", fill="theme:label")
        assert mb.kwargs["fill"] == "theme:label"

    def test_color_alias_none_does_not_raise(self) -> None:
        mb = MarkBase("point", color="none")
        assert mb.kwargs["fill"] == "none"

    @pytest.mark.parametrize(
        "value", ["Theme:Label", "THEME:LABEL", " theme:label", "theme:label "]
    )
    def test_theme_label_sentinel_is_exact_match_only(self, value: str) -> None:
        # draw.rs compares "theme:label" exactly (no trim, no case-fold),
        # unlike "none". Any other spelling is NOT the sentinel and must
        # fall through to to_hex, which rejects it as an invalid color.
        with pytest.raises(ValueError, match="CSS color name"):
            MarkBase("rule", stroke=value)


# ---------------------------------------------------------------------------
# stroke_dash="a,b" comma-split (canonical key, not just linetype= alias)
# ---------------------------------------------------------------------------


class TestStrokeDashCommaSplit:
    def test_stroke_dash_two_value_comma_string_parses(self) -> None:
        mb = MarkBase("rule", stroke_dash="4,2")
        assert mb.kwargs["stroke_dash"] == [4.0, 2.0]

    def test_stroke_dash_single_value_comma_string_parses(self) -> None:
        mb = MarkBase("rule", stroke_dash="6")
        assert mb.kwargs["stroke_dash"] == [6.0]

    def test_stroke_dash_four_value_comma_string_parses(self) -> None:
        mb = MarkBase("rule", stroke_dash="4,2,1,2")
        assert mb.kwargs["stroke_dash"] == [4.0, 2.0, 1.0, 2.0]

    def test_stroke_dash_list_passthrough_unchanged(self) -> None:
        mb = MarkBase("line", stroke_dash=[4.0, 2.0])
        assert mb.kwargs["stroke_dash"] == [4.0, 2.0]

    def test_stroke_dash_tuple_passthrough_unchanged(self) -> None:
        mb = MarkBase("line", stroke_dash=(4.0, 2.0))
        assert mb.kwargs["stroke_dash"] == (4.0, 2.0)

    def test_linetype_named_form_still_resolves_via_alias(self) -> None:
        # Named forms stay attached to linetype=/line_type= only.
        mb = MarkBase("line", linetype="dashed")
        assert mb.kwargs["stroke_dash"] == [4.0, 2.0]

    def test_linetype_raw_comma_string_still_parses(self) -> None:
        mb = MarkBase("line", linetype="6,3")
        assert mb.kwargs["stroke_dash"] == [6.0, 3.0]

    def test_stroke_dash_canonical_key_does_not_resolve_named_forms(self) -> None:
        # "dashed" is a linetype= name, not a comma-separated dash array;
        # the canonical stroke_dash= kwarg only understands the comma-split
        # form, so a named string is refused. The refusal must name the
        # mark, the kwarg, the offending value, and the accepted forms
        # (spec-reviewer finding: a bare "could not convert string to
        # float" ValueError names none of those and falls below the
        # batch's "fail loudly" bar).
        with pytest.raises(ValueError) as exc_info:
            MarkBase("line", stroke_dash="dashed")
        message = str(exc_info.value)
        assert "mark_line" in message
        assert "stroke_dash" in message
        assert "'dashed'" in message
        assert "[4.0, 2.0]" in message  # numeric-list example
        assert '"4,2"' in message  # comma-separated-string example
        assert "linetype" in message  # named-linetype escape hatch

    def test_linetype_alias_bad_value_message_names_accepted_forms(self) -> None:
        # Same refusal, reached via the linetype= alias with a value that is
        # neither a recognized name nor a valid comma-split numeric string.
        with pytest.raises(ValueError) as exc_info:
            MarkBase("line", linetype="squiggly")
        message = str(exc_info.value)
        assert "mark_line" in message
        assert "linetype" in message
        assert "'squiggly'" in message


# ---------------------------------------------------------------------------
# Discrimination: bad vs. good values actually differ
# ---------------------------------------------------------------------------


def test_valid_fill_does_not_raise_but_invalid_fill_does() -> None:
    MarkBase("point", fill="tomato")  # does not raise
    with pytest.raises(ValueError, match="CSS color name"):
        MarkBase("point", fill="not-a-real-color-name")


# ---------------------------------------------------------------------------
# Public-path coverage + spec §7 byte-identity pin (quality-reviewer finding)
#
# The tests above all poke the private MarkBase class directly. These two
# exercise the user-facing entry point (fm.Chart().mark_*()) so the
# construction-time guarantee and its byte-neutrality are pinned as
# committed regression coverage, not just ephemeral reviewer probes.
# ---------------------------------------------------------------------------


def test_public_mark_point_raises_on_bad_fill() -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(ValueError, match="CSS color name"):
        fm.Chart(df).mark_point(fill="bogus")


def test_none_spelling_variants_render_byte_identical_svg() -> None:
    # Spec §7: byte-identity for the absent/unaffected case. A "none"
    # spelling ferrum's construction-time gate merely tolerates (rather
    # than normalizes) must render identically to the canonical lowercase
    # spelling — the renderer, not Python, does the case-folding.
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

    def render(stroke: str) -> str:
        return (
            fm.Chart(df).mark_point(fill="steelblue", stroke=stroke).encode(x="x", y="y").to_svg()
        )

    baseline = render("none")
    for variant in ["None", "NONE", " none "]:
        assert render(variant) == baseline


def test_transparent_renders_identically_to_none() -> None:
    # Spec §4.1 (superseded 2026-09-01, T8 quality review): "transparent"
    # joins "none" as a clearing spelling at the mark/selection boundaries,
    # in both languages — a cleared paint emits NO attribute either way, so
    # fill="transparent" must render byte-identical SVG to fill="none".
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

    def render(stroke: str) -> str:
        return (
            fm.Chart(df).mark_point(fill="steelblue", stroke=stroke).encode(x="x", y="y").to_svg()
        )

    assert render("transparent") == render("none")
