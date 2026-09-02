"""Full-CSS color vocabulary coverage for ``ferrum.color.to_hex`` (Batch A Task 1).

``to_hex``'s string path is a thin wrapper over the one Rust color parser
(``parse_color`` in ``crates/ferrum-core/src/render/color/primitive.rs``), so
this module pins the same full-vocabulary contract from the Python side:
a representative sample of the 148 CSS Color 4 named colors (including the
corrected ``mediumpurple``), hex forms (``#rgb``/``#rgba``/``#rrggbb``/
``#rrggbbaa``), the ``rgb()``/``rgba()`` functional forms, case/whitespace
tolerance, the sentinel literals that must never reach this parser
unparsed, and the canonical accepted-forms error message on invalid input.

The full 148-name table (with CSS Color 4 reference values) lives once, in
Rust, as ``CSS_COLOR_4_REFERENCE`` in
``crates/ferrum-core/src/render/color/primitive.rs``'s test module — that is
the source of truth for parser correctness. Duplicating all 148 entries here
would only re-assert the Rust table through the Python binding (the parsing
logic is 100% Rust-side) while adding a second place to keep in sync; the
sample below exists to pin the Python-boundary wiring (`to_hex` calls the
Rust parser and normalizes correctly), not to re-verify the color table.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm

# A representative sample of CSS_COLOR_4_REFERENCE (see module docstring):
# primaries, a multi-word name, the gray/grey British-spelling alias, the
# newest CSS Color 4 addition, and mediumpurple (the regression this task
# fixes) — enough to prove `to_hex` routes strings through the real parser
# without re-duplicating all 148 entries.
CSS_COLOR_4_SAMPLE: list[tuple[str, str]] = [
    ("black", "#000000"),
    ("white", "#ffffff"),
    ("red", "#ff0000"),
    ("green", "#008000"),
    ("blue", "#0000ff"),
    ("gray", "#808080"),
    ("grey", "#808080"),
    ("darkslategrey", "#2f4f4f"),
    ("steelblue", "#4682b4"),
    ("rebeccapurple", "#663399"),
    ("mediumpurple", "#9370db"),
]


@pytest.mark.parametrize(
    "name,expected_hex", CSS_COLOR_4_SAMPLE, ids=[n for n, _ in CSS_COLOR_4_SAMPLE]
)
def test_to_hex_named_color_matches_css_color_4_reference(name: str, expected_hex: str) -> None:
    assert fm.color.to_hex(name) == expected_hex


def test_to_hex_mediumpurple_is_corrected_value() -> None:
    # #97-shaped regression: mediumpurple was transcribed as (147, 111, 219);
    # CSS Color 4's actual value is (147, 112, 219) = #9370db.
    assert fm.color.to_hex("mediumpurple") == "#9370db"


def test_to_hex_named_color_case_insensitive() -> None:
    assert fm.color.to_hex("SteelBlue") == fm.color.to_hex("steelblue")
    assert fm.color.to_hex("STEELBLUE") == fm.color.to_hex("steelblue")


def test_to_hex_named_color_whitespace_tolerant() -> None:
    assert fm.color.to_hex("  steelblue  ") == fm.color.to_hex("steelblue")


class TestHexForms:
    def test_six_digit_hex_passthrough(self) -> None:
        assert fm.color.to_hex("#4682b4") == "#4682b4"

    def test_eight_digit_hex_preserves_alpha(self) -> None:
        assert fm.color.to_hex("#4682b4cc") == "#4682b4cc"

    def test_three_digit_shorthand_expands(self) -> None:
        assert fm.color.to_hex("#abc") == "#aabbcc"

    def test_four_digit_shorthand_expands_with_alpha(self) -> None:
        assert fm.color.to_hex("#abcd") == "#aabbccdd"

    def test_hex_is_case_insensitive(self) -> None:
        assert fm.color.to_hex("#ABCDEF") == "#abcdef"


class TestRgbFunctionalForms:
    def test_rgb_matches_named_equivalent(self) -> None:
        assert fm.color.to_hex("rgb(70, 130, 180)") == fm.color.to_hex("steelblue")

    def test_rgb_no_spaces(self) -> None:
        assert fm.color.to_hex("rgb(70,130,180)") == "#4682b4"

    def test_rgb_case_insensitive(self) -> None:
        assert fm.color.to_hex("RGB(70, 130, 180)") == "#4682b4"

    def test_rgba_float_alpha_encodes_alpha_byte(self) -> None:
        assert fm.color.to_hex("rgba(70, 130, 180, 0.5)") == "#4682b480"

    def test_rgba_alpha_endpoints(self) -> None:
        assert fm.color.to_hex("rgba(10, 20, 30, 0)") == "#0a141e00"
        assert fm.color.to_hex("rgba(10, 20, 30, 1)") == "#0a141e"

    def test_rgba_percentage_free_integer_alpha_rejected(self) -> None:
        # Spec: alpha must be a float in 0..=1; the CSS percentage-free
        # 0-255 integer alpha form is NOT accepted.
        with pytest.raises(ValueError, match="CSS color name"):
            fm.color.to_hex("rgba(10, 20, 30, 255)")

    def test_rgb_channel_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="CSS color name"):
            fm.color.to_hex("rgb(256, 0, 0)")

    def test_rgb_wrong_arity_rejected(self) -> None:
        with pytest.raises(ValueError, match="CSS color name"):
            fm.color.to_hex("rgb(1, 2)")


class TestInvalidInput:
    def test_unrecognized_name_names_accepted_forms(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"expected a CSS color name, #rrggbb\[aa\]/#rgb\[a\] hex, or rgb\(\)/rgba\(\)",
        ):
            fm.color.to_hex("not-a-color")

    def test_malformed_hex_rejected(self) -> None:
        with pytest.raises(ValueError):
            fm.color.to_hex("#zzzzzz")

    def test_wrong_length_hex_rejected(self) -> None:
        # 5 hex digits matches none of the accepted lengths (3, 4, 6, 8).
        with pytest.raises(ValueError):
            fm.color.to_hex("#12345")

    @pytest.mark.parametrize(
        "bad_hex", ["#a€", "#d50中", "#中"], ids=["ascii-plus-euro", "ascii-plus-cjk", "cjk-only"]
    )
    def test_non_ascii_hex_shaped_string_raises_value_error_not_panic(self, bad_hex: str) -> None:
        # S4 regression (rust-quality-reviewer, Task 1): the Rust hex parser
        # sliced by byte offset, so a hex-shaped string with a multi-byte
        # UTF-8 character (e.g. a euro sign or a CJK character right after
        # "#") panicked with pyo3_runtime.PanicException instead of raising
        # ValueError. Pinning ValueError here at the Python boundary is what
        # would have caught this before it reached a fuzzer.
        with pytest.raises(ValueError, match="CSS color name"):
            fm.color.to_hex(bad_hex)


class TestSentinelsAreNotColors:
    """``to_hex`` correctly refuses ferrum's non-color sentinel literals.

    ``"none"``/``"transparent"`` (clear a fill/stroke paint at the mark/
    selection boundaries, spec §4.1 — ``"transparent"`` joined ``"none"`` as
    a clearing spelling in the 2026-09-01 T8 quality-review supersession)
    and ``"theme:label"`` (a theme lookup token) are NOT colors — they are
    sentinels consumed elsewhere in the pipeline. This module pins that
    `parse_color`/`to_hex` never lets any of the three resolve as a color.

    The two clearing spellings raise a DIFFERENT message than
    ``"theme:label"`` does (spec §4.1, extended 2026-09-01 T9 re-confirm):
    ``to_hex`` short-circuits ``"none"``/``"transparent"`` before the parser
    with a dedicated, sentinel-aware message naming the spelling and
    stating it clears paint at mark/selection boundaries and has no hex
    form *here* — never the generic accepted-forms text, which would be
    self-contradicting for ``"transparent"`` (a real CSS Color 4 keyword).
    The reason it has no hex form here is the sentinel/vocabulary
    separation this boundary enforces, not an absence of a hex
    representation — CSS Color 4 defines ``transparent`` as fully
    transparent black, ``#00000000``. ``"theme:label"`` has no dedicated
    handling in ``to_hex`` and still falls through to the parser, raising
    the ordinary unrecognized-string message.

    This is exactly *why* Task 8 (`resolve_mark_style`) and Task 9
    (`MarkBase.__init__` construction-time validation) must check for
    ``"none"``/``"transparent"``/``"theme:label"`` and short-circuit
    *before* calling into `ferrum.color.to_hex`/`parse_color` — if any
    task's dispatch order is wrong, `fill="none"`/`fill="transparent"` or
    `color="theme:label"` will start raising at the call site that no
    longer short-circuits.
    """

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
    def test_clearing_sentinel_gets_dedicated_message(self, spelling: str) -> None:
        with pytest.raises(ValueError, match="clears paint at mark/selection") as excinfo:
            fm.color.to_hex(spelling)
        assert "CSS color name" not in str(excinfo.value)

    def test_theme_label_sentinel_is_not_a_color(self) -> None:
        with pytest.raises(ValueError, match="CSS color name"):
            fm.color.to_hex("theme:label")

    # -----------------------------------------------------------------
    # Cross-language end-to-end pin (Batch A Task 14, Lane A). Every test
    # above this point exercises exactly one language: `to_hex` proves the
    # Python-only half of the "theme:label" contract (it raises, matching
    # `_is_paint_sentinel`'s docstring claim that Rust's `draw.rs` treats it
    # identically) but never proves the Rust half actually agrees --  a
    # mirror asserted, not exercised (see this class's docstring, "Task 8
    # ... Task 9" paragraph, and `ferrum/marks/base.py:61`). A regression in
    # either language alone passed every test above unnoticed:
    #   * Python regression: `_is_paint_sentinel` (base.py:67) stops
    #     matching `"theme:label"` -> `MarkBase.__init__` calls
    #     `_validate_literal_color`, which raises `ValueError` at
    #     construction instead of letting the sentinel through.
    #   * Rust regression: `resolve_paint_color` (draw.rs) drops its
    #     `value == "theme:label"` arm -> the sentinel falls through to
    #     `parse_color`, which raises `RenderError::InvalidColor` at
    #     `.to_svg()` instead of resolving `theme.colors.label_color`.
    # One assertion below exercises both arms in sequence: construction
    # must succeed (Python's exact-match arm) AND the render must reflect
    # the theme's `label_color` (Rust's exact-match arm) -- not a
    # hardcoded default, so the assertion cannot pass by accident.
    # -----------------------------------------------------------------

    def test_theme_label_sentinel_renders_theme_label_color_end_to_end_via_color_kwarg(
        self,
    ) -> None:
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        chart = fm.Chart(df).mark_point(color="theme:label").encode(x="x", y="y")

        svg_a = chart.theme(fm.Theme(label_color="#123456")).to_svg()
        assert 'fill="#123456"' in svg_a

        # A second, distinct theme value proves the render is driven by the
        # theme (Rust's sentinel arm actually consulting `theme.colors.
        # label_color`), not a coincidental match against one hardcoded hex.
        svg_b = chart.theme(fm.Theme(label_color="#abcdef")).to_svg()
        assert 'fill="#abcdef"' in svg_b
        assert svg_a != svg_b

    def test_theme_label_sentinel_renders_theme_label_color_end_to_end_via_fill_kwarg(
        self,
    ) -> None:
        df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        chart = (
            fm.Chart(df)
            .mark_point(fill="theme:label")
            .encode(x="x", y="y")
            .theme(fm.Theme(label_color="#654321"))
        )
        svg = chart.to_svg()
        assert 'fill="#654321"' in svg

    def test_theme_label_sentinel_construction_does_not_raise(self) -> None:
        """Isolates the Python-side arm of the two tests above: passing
        ``"theme:label"`` to ``mark_point`` must not raise at construction
        time (the exact-match short-circuit in ``_is_paint_sentinel`` must
        fire before ``_validate_literal_color``/``to_hex`` ever sees it)."""
        df = pl.DataFrame({"x": [0.0], "y": [0.0]})
        fm.Chart(df).mark_point(color="theme:label").encode(x="x", y="y")
        fm.Chart(df).mark_point(fill="theme:label").encode(x="x", y="y")

    def test_theme_label_sentinel_is_not_a_literal_hex_passthrough(self) -> None:
        """Discriminating guard: the rendered fill must equal the THEME's
        resolved label color, not merely "some hex string" -- rules out a
        vacuous regex match against an unrelated fill in the SVG."""
        df = pl.DataFrame({"x": [0.0], "y": [0.0]})
        svg = (
            fm.Chart(df)
            .mark_point(color="theme:label")
            .encode(x="x", y="y")
            .theme(fm.Theme(label_color="#0f0f0f"))
            .to_svg()
        )
        circle_fills = re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]{6})"', svg)
        assert circle_fills == ["#0f0f0f"]
