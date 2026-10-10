"""Tests for padding configuration validation (NF-B5/B6/B7) and padding.auto
(F-L07-08, D10, #135).

Validates that padding values are rejected at the Python boundary when:
- Negative: typed ValueError at construction
- Non-numeric (e.g., NormCoord): typed ValueError naming the pixel contract
- Valid pixel values: accepted without change (byte-identity verified)

Also validates the Rust-side consumer added in Task 9 (D10, spec §4.7):
- `padding.auto` expands an unset side's margin to keep a continuous axis's
  edge-tick-label overhang from clipping past the viewport edge, via both
  wire spellings (`configure_padding(auto=...)` and
  `.override(padding_auto=...)`).
- `PaddingExceedsViewport` reports the CALLER's actual value and the
  specific side, not the theme default.

`padding.auto`'s built-in default is `True`, owned solely by Rust
(`ThemePadding::default()`). Python's `PaddingConfig.auto` /
`configure_padding(auto=...)` are `bool | None = None`: a chart that never
mentions `auto` keeps that built-in default. Every test below that compares
a bare call (no `configure_padding`, or a `configure_padding(...)` call that
never names `auto`) expects it to match an explicit `auto=True`.
"""

from __future__ import annotations

import re

import pytest
import polars as pl

import ferrum as fr
from ferrum.configure import PaddingConfig
from ferrum.annotation.coords import NormCoord


class TestPaddingConfigConstruction:
    """Tests for PaddingConfig validation at construction time."""

    def test_negative_padding_raises_valueerror(self):
        """Negative padding values are rejected with ValueError at construction."""
        with pytest.raises(ValueError, match="top.*non-negative"):
            PaddingConfig(top=-10)

    def test_normcoord_padding_raises_valueerror(self):
        """NormCoord padding is rejected with ValueError (spec §4.7)."""
        with pytest.raises(ValueError, match="numeric pixel value"):
            PaddingConfig(top=NormCoord(0.5))  # type: ignore[arg-type]

    def test_string_padding_raises_valueerror(self):
        """String padding is rejected with ValueError."""
        with pytest.raises(ValueError, match="numeric pixel value.*str"):
            PaddingConfig(right="10px")  # type: ignore[arg-type]

    def test_bool_padding_raises_valueerror(self):
        """Boolean padding is rejected with ValueError (bool is not a pixel value)."""
        with pytest.raises(ValueError, match="numeric pixel value.*bool"):
            PaddingConfig(top=True)  # type: ignore[arg-type]

    def test_nan_padding_raises_valueerror(self):
        """NaN padding is rejected with a typed ValueError naming the pixel contract.

        Regression: NaN is an ``isinstance(value, float)`` numeric and
        ``nan < 0`` is False, so it silently passed both the old type check
        and the old negative check, then died downstream as an opaque
        serializer error (``chart_config: expected value at line 1 column
        21``) instead of a typed refusal at the Python boundary.
        """
        with pytest.raises(ValueError, match="finite numeric pixel value"):
            PaddingConfig(top=float("nan"))

    def test_positive_infinity_padding_raises_valueerror(self):
        """+inf padding is rejected with a typed ValueError naming the pixel contract."""
        with pytest.raises(ValueError, match="finite numeric pixel value"):
            PaddingConfig(top=float("inf"))

    def test_negative_infinity_padding_raises_valueerror(self):
        """-inf padding is rejected (previously caught only incidentally by ``< 0``)."""
        with pytest.raises(ValueError, match="finite numeric pixel value"):
            PaddingConfig(top=float("-inf"))

    def test_valid_int_padding_succeeds(self):
        """Integer padding values are valid."""
        cfg = PaddingConfig(top=20, right=15, bottom=10, left=5)
        d = cfg.to_dict()
        assert d["top"] == 20
        assert d["right"] == 15
        assert d["bottom"] == 10
        assert d["left"] == 5

    def test_valid_float_padding_succeeds(self):
        """Float padding values are valid."""
        cfg = PaddingConfig(top=20.5, right=15.3, bottom=10.1, left=5.2)
        d = cfg.to_dict()
        assert d["top"] == 20.5
        assert d["right"] == 15.3
        assert d["bottom"] == 10.1
        assert d["left"] == 5.2

    def test_zero_padding_is_valid(self):
        """Zero padding is valid (not negative)."""
        cfg = PaddingConfig(top=0, right=0, bottom=0, left=0)
        d = cfg.to_dict()
        assert d["top"] == 0
        assert d["right"] == 0
        assert d["bottom"] == 0
        assert d["left"] == 0


class TestChartLevelPaddingValidation:
    """Tests for padding validation through Chart.configure_padding."""

    def test_chart_with_valid_padding_renders(self):
        """A chart with valid padding values renders without error."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg = (
            fr.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_padding(top=20, right=15, bottom=10, left=5)
            .to_svg()
        )
        assert "<svg" in svg
        assert "NaN" not in svg

    def test_chart_with_negative_padding_raises(self):
        """A chart with negative padding raises ValueError at construction."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match="non-negative"):
            fr.Chart(df).mark_point().encode(x="x", y="y").configure_padding(top=-10)

    def test_chart_with_override_negative_padding_raises(self):
        """A chart with override negative padding raises FerrumOverrideError
        in the Chart.override voice (override path validation)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(
            fr.FerrumOverrideError,
            match=r"^Chart\.override\(padding_top=\.\.\.\) must be non-negative; received -10$",
        ):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(padding_top=-10).to_svg()

    def test_chart_with_override_non_numeric_padding_raises(self):
        """A chart with override non-numeric padding raises FerrumOverrideError."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(
            fr.FerrumOverrideError,
            match=(
                r"^Chart\.override\(padding_right=\.\.\.\) must be a numeric pixel value "
                r"\(int or float\), not str; received '10px'$"
            ),
        ):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(padding_right="10px").to_svg()

    @pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
    @pytest.mark.parametrize(
        ("bad", "clause"),
        [
            (-1, "must be non-negative; received -1"),
            ("a", "must be a numeric pixel value (int or float), not str; received 'a'"),
            (float("nan"), "must be a finite numeric pixel value; received nan"),
            (float("inf"), "must be a finite numeric pixel value; received inf"),
            (True, "must be a numeric pixel value (int or float), not bool; received True"),
        ],
        ids=["negative", "non-numeric-str", "nan", "inf", "bool"],
    )
    def test_chart_with_override_bad_padding_value_raises_on_every_side(self, side, bad, clause):
        """``.override(padding_<side>=...)`` raises ``FerrumOverrideError`` in
        the ``Chart.override`` voice for every value the pixel contract (spec
        §4.7) refuses, on every side -- not only ``top``."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        chart = fr.Chart(df).mark_point().encode(x="x", y="y").override(**{f"padding_{side}": bad})
        with pytest.raises(
            fr.FerrumOverrideError,
            match=rf"^Chart\.override\(padding_{side}=\.\.\.\) {re.escape(clause)}$",
        ):
            chart.to_svg()

    def test_chart_with_configure_padding_nan_raises_typed_valueerror(self):
        """configure_padding(top=nan) raises a typed pixel-contract ValueError,
        not an opaque serializer error, at construction time (before render)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match="finite numeric pixel value"):
            fr.Chart(df).mark_point().encode(x="x", y="y").configure_padding(top=float("nan"))

    def test_chart_with_configure_padding_inf_raises_typed_valueerror(self):
        """configure_padding(top=inf) raises a typed pixel-contract ValueError."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match="finite numeric pixel value"):
            fr.Chart(df).mark_point().encode(x="x", y="y").configure_padding(top=float("inf"))

    def test_chart_with_override_nan_padding_raises_typed_valueerror(self):
        """.override(padding_top=nan).to_svg() raises a typed FerrumOverrideError
        naming Chart.override, not the opaque wire-serializer error
        (``chart_config: expected value at line 1 column 21``) that NaN/inf
        produced before this predicate covered finiteness."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(
            fr.FerrumOverrideError,
            match=r"^Chart\.override\(padding_top=\.\.\.\) must be a finite numeric pixel value; received nan$",
        ):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(
                padding_top=float("nan")
            ).to_svg()

    def test_chart_with_override_inf_padding_raises_typed_valueerror(self):
        """.override(padding_top=inf).to_svg() raises a typed FerrumOverrideError."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(
            fr.FerrumOverrideError,
            match=r"^Chart\.override\(padding_top=\.\.\.\) must be a finite numeric pixel value; received inf$",
        ):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(
                padding_top=float("inf")
            ).to_svg()

    def test_chart_with_zero_padding_renders(self):
        """A chart with zero padding (not negative) renders."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg = (
            fr.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_padding(top=0, right=0, bottom=0, left=0)
            .to_svg()
        )
        assert "<svg" in svg

    def test_chart_with_float_padding_renders(self):
        """A chart with float padding values renders."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg = (
            fr.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_padding(top=20.5, right=15.3, bottom=10.1, left=5.2)
            .to_svg()
        )
        assert "<svg" in svg


class TestPaddingByteIdentity:
    """Byte-identity + discriminating pins for padding's rendered geometry.

    ``test_padding_top_moves_plot_rect_by_exact_pixel_value`` is the load-bearing
    instrument here: it is pinned against a specific, extracted coordinate in real
    SVG output (the plot rect's top-y, read off the first grid line), not a
    same-process self-comparison. A same-process comparison (render X twice, assert
    equal) cannot fail if this task's validation change broke padding handling
    itself — it only proves the code is deterministic against itself, not that it
    does the right thing. This test fails if padding stops being applied, is
    clamped, silently rounded, or ignored (a blank or padding-ignoring render would
    not produce ``y1 == "37"``).
    """

    def test_padding_top_moves_plot_rect_by_exact_pixel_value(self):
        """PaddingConfig(top=37) sets the plot rect's top-y coordinate to 37."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg = fr.Chart(df).mark_point().encode(x="x", y="y").configure_padding(top=37).to_svg()
        # The first <line> in the plot body is a grid line; its y1 is the plot
        # rect's top edge, set directly to the requested top-padding pixel value.
        match = re.search(r'<line x1="[^"]+" y1="([^"]+)"', svg)
        assert match is not None, "expected at least one grid line in the SVG"
        assert match.group(1) == "37"

    def test_unpadded_chart_bytes_match(self):
        """An unpadded chart renders consistently (sanity check)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg1 = fr.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        svg2 = fr.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        # Two renders of the same spec should produce byte-identical output
        assert svg1 == svg2
        assert "NaN" not in svg1

    def test_padding_config_differs_from_unpadded(self):
        """A padded chart produces different output than an unpadded chart (discriminating test).

        This validates that valid padding *does* affect the output, i.e., the
        padding configuration is actually being honored and not silently dropped.
        """
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})

        svg_unpadded = fr.Chart(df).mark_point().encode(x="x", y="y").to_svg()

        svg_padded = (
            fr.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_padding(top=30, right=30, bottom=30, left=30)
            .to_svg()
        )

        # Padded and unpadded should NOT be identical (padding changes layout)
        assert svg_unpadded != svg_padded
        # Both should be valid SVGs
        assert "<svg" in svg_unpadded
        assert "<svg" in svg_padded
        assert "NaN" not in svg_unpadded
        assert "NaN" not in svg_padded

    def test_identical_valid_padding_produces_identical_bytes(self):
        """Two charts with identical valid padding produce byte-identical output."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})

        svg1 = (
            fr.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_padding(top=20, right=15, bottom=10, left=5)
            .to_svg()
        )

        svg2 = (
            fr.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_padding(top=20, right=15, bottom=10, left=5)
            .to_svg()
        )

        # Byte-identity for identical configurations
        assert svg1 == svg2

    def test_all_four_explicit_sides_are_byte_identical_regardless_of_auto(self):
        """`auto` never touches a side the caller set explicitly (spec §4.7:
        "explicit side values still win over auto on their side").

        With all four sides given, there is no unset side for auto to
        expand, so `auto=True` and `auto=False` must render
        byte-identically — the untouched-field byte-identity guarantee
        (claim 3), specialized to padding.auto.

        Uses ``TestPaddingAutoEdgeTickOverhang._clipping_repro_chart()`` (a
        long-label chart whose ``auto_padding_for_edge_ticks`` contribution
        is nonzero on the right side), not a plain small-integer chart: on
        the latter the auto contribution computes to 0.0 on every side, so
        `auto=True`/`auto=False` are byte-identical no matter whether the
        explicit-side-wins rule is honored or broken — a vacuous pin. This
        fixture discriminates: mutating the explicit-side-wins rule in
        `compute_layout` (letting `auto`'s contribution leak onto an
        explicitly-set side) makes this fail.
        """
        base = TestPaddingAutoEdgeTickOverhang._clipping_repro_chart().properties(
            width=400, height=200
        )
        svg_auto_true = base.configure_padding(
            top=20, right=15, bottom=10, left=5, auto=True
        ).to_svg()
        svg_auto_false = base.configure_padding(
            top=20, right=15, bottom=10, left=5, auto=False
        ).to_svg()
        assert svg_auto_true == svg_auto_false

        # Anti-vacuity control: leaving the right side unset on this SAME
        # fixture lets `auto` actually reserve its edge-tick-overhang
        # cushion there, proving the equality above reflects the
        # explicit-side-wins rule and not `auto` being a no-op on this
        # fixture/viewport. Pinned on the plot's right gridline (a
        # horizontal grid line's `x2`, its right endpoint), not a bare
        # whole-SVG diff, so the control proves the RIGHT side specifically
        # moved rather than merely that something, somewhere, changed.
        svg_partial_auto_true = base.configure_padding(
            top=20, bottom=10, left=5, auto=True
        ).to_svg()
        svg_partial_auto_false = base.configure_padding(
            top=20, bottom=10, left=5, auto=False
        ).to_svg()
        assert svg_partial_auto_true != svg_partial_auto_false

        right_edge_auto_true = self._right_gridline_x2(svg_partial_auto_true)
        right_edge_auto_false = self._right_gridline_x2(svg_partial_auto_false)
        assert right_edge_auto_false == pytest.approx(384.0, abs=1e-6)
        # The rescue reserves the drawn label's reach past the plot edge,
        # less the 8 px the scale's padding insets the edge tick by.
        assert right_edge_auto_true == pytest.approx(350.502, abs=1e-3), (
            "leaving the right side unset should let `auto` narrow the "
            f"plot's right edge to reserve its tick-overhang cushion "
            f"({right_edge_auto_true} vs {right_edge_auto_false})"
        )

    @staticmethod
    def _right_gridline_x2(svg: str) -> float:
        """The plot's right edge: a horizontal grid line's `x2` (its
        right endpoint, `y1 == y2`). The first horizontal grid line in
        document order shares the same right edge as every other."""
        m = re.search(r'<line x1="[^"]+" y1="([^"]+)" x2="([^"]+)" y2="\1"', svg)
        assert m is not None, f"expected at least one horizontal grid line in the SVG: {svg}"
        return float(m.group(2))


class TestPaddingAutoEdgeTickOverhang:
    """F-L07-08, D10, spec §4.7: `padding.auto` actually does something now.

    The repro: a continuous x-axis whose last tick label is a very long
    number (``999,999,999,999,999`` via explicit ``tick_values=`` so there
    are only two ticks — no adjacent-tick collision to force rotation, so
    the label stays flat/middle-anchored and its overhang lands squarely on
    the untouched 16px default right padding). At `width=400` this
    demonstrably clips off the right edge of the canvas at the pre-fix
    default (`auto` was parsed but never consumed — F-L07-08); visually
    confirmed via ``resvg_py`` rasterization during this task's
    implementation (not re-asserted as a pixel test here — this project's
    own convention, per ``tests/_snapshots.py``, is that SVG structure is
    the automated regression instrument and PNG rasterization is for
    one-time visual confirmation, not a repeated pixel-diff assertion).

    The regression pin below is geometry-based (mirroring
    ``test_padding_top_moves_plot_rect_by_exact_pixel_value`` above): the
    last x tick's rendered position must move further left (the plot must
    narrow) under `auto=True` than under `auto=False`, proving the fix
    actually engages on a REAL clipping scenario found through real font
    metrics — not a synthetic shape.
    """

    @staticmethod
    def _last_x_tick_position(chart) -> float:
        svg = chart.properties(width=400, height=200).to_svg()
        texts = re.findall(
            r'<text[^>]*x="(-?[\d.]+)"[^>]*text-anchor="(middle|end|start)"[^>]*>([^<]*)</text>',
            svg,
        )
        x_ticks = [t for t in texts if t[1] == "middle" and t[2] not in ("x", "y")]
        assert x_ticks, f"expected at least one flat (middle-anchored) x tick label: {svg}"
        return float(x_ticks[-1][0])

    @staticmethod
    def _clipping_repro_chart():
        df = pl.DataFrame({"x": [0.0, 999999999999999.0], "y": [0.0, 1.0]})
        return (
            fr.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .configure_axis(label_format="integer", tick_values=[0, 999999999999999])
        )

    def test_configure_padding_auto_true_narrows_plot_past_auto_false(self):
        """`configure_padding(auto=True)` reserves real extra right-side
        margin for this repro; an explicit `auto=False` does not (#135:
        `auto` unset is the built-in-default-inheriting default; both
        explicit values still mean what they say)."""
        base = self._clipping_repro_chart()
        x_auto_false = self._last_x_tick_position(base.configure_padding(auto=False))
        x_auto_true = self._last_x_tick_position(base.configure_padding(auto=True))
        assert x_auto_true < x_auto_false - 5.0, (
            f"padding.auto=True should move the last x tick well left of its "
            f"auto=False position ({x_auto_true} vs {x_auto_false})"
        )

    def test_configure_padding_bare_default_inherits_the_built_in_auto_default(self):
        """`configure_padding()`'s own bare call never mentions `auto`, so it
        sends nothing for `auto` on the wire and keeps ferrum's built-in
        default (#135: on) — it renders like an explicit `auto=True` call,
        not `auto=False`."""
        base = self._clipping_repro_chart()
        svg_bare = base.configure_padding().properties(width=400, height=200).to_svg()
        svg_explicit_true = (
            base.configure_padding(auto=True).properties(width=400, height=200).to_svg()
        )
        svg_explicit_false = (
            base.configure_padding(auto=False).properties(width=400, height=200).to_svg()
        )
        assert svg_bare == svg_explicit_true
        assert svg_bare != svg_explicit_false

    def test_configure_padding_with_only_one_side_never_silently_turns_auto_off(self):
        """A caller who sets one side (the common `configure_padding(top=...)`
        shape) and never mentions `auto` must keep the built-in auto default
        ON for the other three sides (#135's no-reset guarantee) — it must
        NOT render like an explicit `auto=False` call."""
        base = self._clipping_repro_chart()
        svg_one_side = base.configure_padding(top=5).properties(width=400, height=200).to_svg()
        svg_one_side_explicit_auto_true = (
            base.configure_padding(top=5, auto=True).properties(width=400, height=200).to_svg()
        )
        svg_one_side_explicit_no_auto = (
            base.configure_padding(top=5, auto=False).properties(width=400, height=200).to_svg()
        )
        assert svg_one_side == svg_one_side_explicit_auto_true
        assert svg_one_side != svg_one_side_explicit_no_auto

    def test_override_padding_auto_true_matches_configure_padding_auto_true(self):
        """Both wire spellings — `configure_padding(auto=True)` and the
        deprecated `.override(padding_auto=True)` — reach the SAME Rust
        consumer (`ChartConfig.padding.auto` -> `theme.padding.padding_auto`
        -> `auto_padding_for_edge_ticks`), so they must render byte-identically."""
        base = self._clipping_repro_chart()
        svg_configure = base.configure_padding(auto=True).properties(width=400, height=200).to_svg()
        with pytest.warns(DeprecationWarning, match="padding_auto"):
            svg_override = (
                base.override(padding_auto=True).properties(width=400, height=200).to_svg()
            )
        assert svg_configure == svg_override

    def test_override_padding_auto_false_matches_configure_padding_auto_false(self):
        """The `False` spelling of both paths also converges (T8's
        `_chart_config_wire_fragment` pinned `.override(padding_auto=False)`
        reaching the wire; this pins it reaching the SAME rendered geometry as
        `configure_padding(auto=False)`, not just "doesn't crash")."""
        base = self._clipping_repro_chart()
        svg_configure = (
            base.configure_padding(auto=False).properties(width=400, height=200).to_svg()
        )
        with pytest.warns(DeprecationWarning, match="padding_auto"):
            svg_override = (
                base.override(padding_auto=False).properties(width=400, height=200).to_svg()
            )
        assert svg_configure == svg_override

    def test_a_chart_never_calling_configure_padding_inherits_the_built_in_default(self):
        """A chart that never touches ``padding`` at all sends NO chart-config
        padding section, so it renders identically to an explicit
        ``configure_padding(auto=True)`` call and differently from
        ``configure_padding(auto=False)`` (#135: the Rust built-in default,
        `ThemePadding::default().padding_auto`, is now `True`, and it is the
        only thing that applies when no `Configure.padding` is present).

        Pinned on the real clipping-repro fixture (spec-review cycle 6,
        S2 — the original version of this test built the SAME
        short-tick-label chart twice and asserted the two renders were
        equal, which pins render determinism only: it would pass unchanged
        if `apply_chart_config` wrote `padding_auto` unconditionally, or if
        the whole auto mechanism were reverted). This fixture's `auto=True`
        vs `auto=False` geometry difference is already pinned above, so a
        regression on either of those fronts shows up here as a byte
        mismatch instead of silently passing.
        """
        base = self._clipping_repro_chart()
        svg_no_config = base.properties(width=400, height=200).to_svg()
        svg_auto_true = base.configure_padding(auto=True).properties(width=400, height=200).to_svg()
        svg_auto_false = (
            base.configure_padding(auto=False).properties(width=400, height=200).to_svg()
        )
        assert svg_no_config == svg_auto_true
        assert svg_no_config != svg_auto_false


class TestPaddingAutoAxisTitleRecentering:
    """Spec-review cycle 2, finding 1: a naive symmetric-only reading of
    "padding.auto expands margins to fit measured labels/titles" wrongly
    concluded axis TITLES could never be rescued by padding at all. An
    ASYMMETRIC expansion (recentering the plot, not just shrinking it) DOES
    fix a real title clip when the title is no wider than the viewport
    itself — only a title wider than the viewport is a genuine residual.

    The repro (`width=270`, a moderately long x-axis title) visibly clips
    ("...title her") at `auto=False` and renders fully intact ("...title
    here") at `auto=True`, confirmed via ``resvg_py`` PNG rasterization
    during this fix's implementation (see the module docstring on why this
    is not re-asserted as a pixel test here). The pytest pin below is
    geometry-based: the title's rendered anchor position must move under
    `auto=True`, proving the asymmetric recentering mechanism engages on a
    real title-clipping scenario found through real font metrics.
    """

    TITLE = "A moderately long horizontal axis title here"

    @classmethod
    def _title_x(cls, chart) -> float:
        svg = chart.properties(width=270, height=200).to_svg()
        m = re.search(
            r'<text[^>]*x="(-?[\d.]+)"[^>]*text-anchor="middle"[^>]*>'
            + re.escape(cls.TITLE)
            + r"</text>",
            svg,
        )
        assert m is not None, f"expected the x-axis title text in the SVG: {svg}"
        return float(m.group(1))

    @classmethod
    def _repro_chart(cls):
        # x spans 0..10, so the x tick labels stay flat and their count is
        # the same at auto=True and auto=False -- the title recenter below
        # must not also change which or how many x labels draw.
        df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 1.0]})
        return fr.Chart(df).mark_point().encode(x=fr.X("x", title=cls.TITLE), y="y")

    def test_auto_true_recenters_the_title_away_from_its_auto_false_position(self):
        base = self._repro_chart()
        x_auto_false = self._title_x(base.configure_padding(auto=False))
        x_auto_true = self._title_x(base.configure_padding(auto=True))
        assert x_auto_true < x_auto_false - 10.0, (
            f"padding.auto=True should recenter (shift left) an x title that "
            f"overflows the right edge at auto=False ({x_auto_true} vs {x_auto_false})"
        )

    def test_bare_chart_inherits_the_built_in_default_and_recenters_the_title(self):
        """A chart that never calls `configure_padding` keeps ferrum's
        built-in `auto` default (#135: on), so it renders like an explicit
        `auto=True` call — recentering the title — and NOT like `auto=False`."""
        base = self._repro_chart()
        svg_no_call = base.properties(width=270, height=200).to_svg()
        svg_auto_true = base.configure_padding(auto=True).properties(width=270, height=200).to_svg()
        svg_auto_false = (
            base.configure_padding(auto=False).properties(width=270, height=200).to_svg()
        )
        assert svg_no_call == svg_auto_true
        assert svg_no_call != svg_auto_false


class TestPaddingAutoHonorsMaxBand:
    """Spec-review cycle 2, finding 2: the left/right tick-overhang cushion
    previously used the y-axis's UNCLAMPED label-band width, so
    ``fm.Axis(max_band=...)`` capping the REAL reserved gutter smaller made
    the cushion look bigger than it really is — `auto` silently under-
    reserved and a long edge tick label clipped identically under
    `auto=True` and `auto=False`. Live repro (reviewer-banked): a very long
    negative first x tick label paired with a tightly `max_band`-capped
    y-axis, visually confirmed clipped at `auto=False` and fully visible at
    `auto=True` via ``resvg_py`` PNG rasterization during this fix's
    implementation.
    """

    @staticmethod
    def _repro_chart():
        df = pl.DataFrame({"x": [-999999999999999.0, 0.0], "y": [0.0, 1.0]})
        return (
            fr.Chart(df)
            .mark_point()
            .encode(x="x:Q", y=fr.Y("y", axis=fr.Axis(max_band=10)))
            .configure_axis(label_format="integer", tick_values=[-999999999999999, 0])
        )

    @staticmethod
    def _first_x_tick_x(chart) -> float:
        svg = chart.properties(width=400, height=200).to_svg()
        texts = re.findall(
            r'<text[^>]*x="(-?[\d.]+)"[^>]*text-anchor="(middle|end|start)"[^>]*>([^<]*)</text>',
            svg,
        )
        x_ticks = [t for t in texts if t[1] == "middle" and t[2] not in ("x", "y")]
        assert x_ticks, f"expected at least one flat x tick label: {svg}"
        return float(x_ticks[0][0])

    def test_auto_true_moves_the_first_tick_right_past_its_max_band_capped_cushion(self):
        base = self._repro_chart()
        x_auto_false = self._first_x_tick_x(base.configure_padding(auto=False))
        x_auto_true = self._first_x_tick_x(base.configure_padding(auto=True))
        assert x_auto_true > x_auto_false + 10.0, (
            f"padding.auto=True should move the first x tick well right of its "
            f"auto=False position once max_band's real (capped) cushion is honored "
            f"({x_auto_true} vs {x_auto_false})"
        )


class TestPaddingAutoTitleAndTickComposition:
    """Spec-review cycle 6, finding 2 / cycle 7, finding 2: the cycle-1 S3
    that started this whole composition-correctness thread was found ONLY
    in this real-font path, not Rust's ``MockMetrics`` unit tests — this
    file is where this batch keeps its real-metrics geometry pins, and
    `TestPaddingAutoHonorsMaxBand` already builds the exact tick-overhang
    fixture (``-999999999999999.0`` first x value, ``fm.Axis(max_band=10)``,
    integer tick labels) the composed regression needs. This class adds an
    x-axis title to that same fixture and pins both composed outcomes at
    the real-metrics level — mirroring, but not duplicating, Rust's
    `compute_layout` pins of the same shape
    (``padding_auto_keeps_title_on_canvas_when_a_tick_correction_also_fires``
    and
    ``padding_auto_never_worsens_an_infeasible_title_when_a_tick_correction_also_fires``
    in ``crates/ferrum-core/src/layout/mod.rs``), so a regression that only
    shows up under real font metrics (as the original S3 did) is caught
    here even if the Rust-only ``MockMetrics`` pins keep passing.
    """

    @staticmethod
    def _repro_chart(title: str):
        df = pl.DataFrame({"x": [-999999999999999.0, 0.0], "y": [0.0, 1.0]})
        return (
            fr.Chart(df)
            .mark_point()
            .encode(x=fr.X("x", title=title), y=fr.Y("y", axis=fr.Axis(max_band=10)))
            .configure_axis(label_format="integer", tick_values=[-999999999999999, 0])
        )

    @staticmethod
    def _title_x(chart, width: int, height: int, title: str) -> float:
        svg = chart.properties(width=width, height=height).to_svg()
        m = re.search(
            r'<text[^>]*x="(-?[\d.]+)"[^>]*text-anchor="middle"[^>]*>'
            + re.escape(title)
            + r"</text>",
            svg,
        )
        assert m is not None, f"expected the x-axis title text in the SVG: {svg}"
        return float(m.group(1))

    @staticmethod
    def _first_x_tick_x(chart, width: int, height: int) -> float:
        svg = chart.properties(width=width, height=height).to_svg()
        texts = re.findall(
            r'<text[^>]*x="(-?[\d.]+)"[^>]*text-anchor="(middle|end|start)"[^>]*>([^<]*)</text>',
            svg,
        )
        # Numeric (comma-grouped) tick labels only — excludes the "y" axis
        # title and the custom x-axis title this fixture adds.
        x_ticks = [t for t in texts if t[1] == "middle" and re.fullmatch(r"-?[\d,]+", t[2])]
        assert x_ticks, f"expected at least one flat, numeric x tick label: {svg}"
        return float(x_ticks[0][0])

    FEASIBLE_TITLE = "Total revenue per fiscal quarter in millions USD"

    def test_title_renders_whole_when_a_tick_correction_also_fires(self):
        """Finding 1's own live repro (banked in the T9 quality reviewer's
        cycle-6 record): at ``width=300``, the tick pass alone wants real
        left padding for the huge first tick label (same fixture as
        `TestPaddingAutoHonorsMaxBand`); the title pass alone, evaluated
        against the UNTOUCHED base, would report "fits, no correction" —
        the exact shape that let an elementwise-max composition ship a
        title-clipping bug for five cycles. Both corrections must still be
        live and composed correctly: the tick pass keeps moving the first
        tick label well clear of the canvas edge, and the title pass keeps
        recentering by the small amount this specific real-font geometry
        needs. Exact pixel values pinned from the verified-correct
        rebuilt-extension run (banked in the quality reviewer's cycle-6
        record and reconfirmed here); visually confirmed via ``resvg_py``
        PNG rasterization during this fix's implementation that the title
        renders in full (not re-asserted as a pixel test, per this file's
        established convention).
        """
        chart = self._repro_chart(self.FEASIBLE_TITLE)
        auto_false = chart.configure_padding(auto=False)
        auto_true = chart.configure_padding(auto=True)

        tick_auto_false = self._first_x_tick_x(auto_false, 300, 200)
        tick_auto_true = self._first_x_tick_x(auto_true, 300, 200)
        assert tick_auto_true > tick_auto_false + 20.0, (
            "the tick pass must still be firing in this composed fixture — "
            f"this test would be vacuous otherwise ({tick_auto_true} vs {tick_auto_false})"
        )

        title_auto_false = self._title_x(auto_false, 300, 200, self.FEASIBLE_TITLE)
        title_auto_true = self._title_x(auto_true, 300, 200, self.FEASIBLE_TITLE)
        assert title_auto_false == pytest.approx(155.0, abs=1e-6)
        assert title_auto_true == pytest.approx(155.486, abs=1e-2), (
            "the title pass, solved against the tick-adjusted base rather "
            f"than the untouched one, must still recenter it ({title_auto_true})"
        )

    INFEASIBLE_TITLE = "X" * 120
    SHORT_TITLE = "Revenue"

    def test_never_worse_than_auto_false_when_the_title_is_infeasible(self):
        """The composed sibling of `TestPaddingAutoAxisTitleRecentering`'s
        genuine-residual case (cycle 2, finding 1's narrowed claim) and the
        real-metrics pin of Rust's
        ``padding_auto_never_worsens_an_infeasible_title_when_a_tick_correction_also_fires``
        (cycle 7, finding 1): the same tick-overhang fixture as the test
        above, but with a title (120 characters) far too wide for the
        viewport to ever fully contain, at the quality reviewer's own
        ``W=280`` fixture width. Before the cycle-7 fix, the tick pass's
        own real, asymmetric left correction still applied even though the
        title's own correction was infeasible (declined), shifting the
        plot's (and title's) center and making an already-clipped title
        clip WORSE than `auto=False` — live-measured by the reviewer as
        9.51px -> 26.53px at this exact width. The fixed mechanism
        suppresses BOTH corrections on this axis once the title cannot be
        helped, so `auto=True` must render BYTE-IDENTICAL to `auto=False`
        here — the true guarantee (never worse), not the vacuous
        `result.is_ok()` this regime's test previously asserted. Visually
        confirmed via ``resvg_py`` PNG rasterization during this fix's
        implementation that both renders clip the title identically (not
        re-asserted as a pixel test, per this file's established
        convention).
        """
        chart = self._repro_chart(self.INFEASIBLE_TITLE)
        svg_auto_false = (
            chart.configure_padding(auto=False).properties(width=280, height=200).to_svg()
        )
        svg_auto_true = (
            chart.configure_padding(auto=True).properties(width=280, height=200).to_svg()
        )
        assert svg_auto_true == svg_auto_false, (
            "an infeasible title's auto=True render must be byte-identical to "
            "auto=False — the tick pass's own correction must be suppressed "
            "too, not just the title's, once the title cannot be recentered "
            "into the viewport at all"
        )

        # Anti-vacuity control: the SAME fixture with a short (feasible)
        # title at the same W=280 is NOT byte-identical between
        # `auto=True` and `auto=False` — the tick pass's own left
        # correction still fires here (the title is short enough that
        # nothing suppresses it), proving the byte-identity above reflects
        # genuine suppression of an infeasible title's own correction, not
        # the tick/title machinery being globally inert on this
        # fixture/viewport. Pinned on the class's own `_first_x_tick_x`
        # (and the title anchor `_title_x`), not a bare whole-SVG diff, so
        # a regression that made the tick pass inert while the title still
        # recentered could not leave this control green.
        short_chart = self._repro_chart(self.SHORT_TITLE)
        short_auto_false = short_chart.configure_padding(auto=False)
        short_auto_true = short_chart.configure_padding(auto=True)

        short_tick_auto_false = self._first_x_tick_x(short_auto_false, 280, 200)
        short_tick_auto_true = self._first_x_tick_x(short_auto_true, 280, 200)
        assert short_tick_auto_false == pytest.approx(34.0, abs=1e-6)
        # Less the 8 px padding inset of the edge tick (the title center
        # below moves half of that).
        assert short_tick_auto_true == pytest.approx(60.027, abs=1e-3), (
            "the tick pass must still fire on this short-title fixture "
            f"({short_tick_auto_true} vs {short_tick_auto_false})"
        )

        short_title_auto_false = self._title_x(short_auto_false, 280, 200, self.SHORT_TITLE)
        short_title_auto_true = self._title_x(short_auto_true, 280, 200, self.SHORT_TITLE)
        assert short_title_auto_false == pytest.approx(145.0, abs=1e-6)
        assert short_title_auto_true == pytest.approx(158.014, abs=1e-3), (
            "the title pass must still recenter this short (feasible) "
            f"title too ({short_title_auto_true} vs {short_title_auto_false})"
        )


class TestPaddingAutoCrossAxisTitleInfeasibility:
    """A composed case with two titles, on different axes, in different
    feasibility regimes: this class's fixture puts an infeasible
    (too-wide) title on the x-axis and an independently genuine (feasible,
    needs a real recenter) title on the y-axis in the SAME chart, and pins
    both fates at once — which side's correction is suppressed (x, whose
    own title cannot fit the viewport no matter what) and which title's
    correction survives (y, whose own recenter is unrelated to x's
    infeasibility and must still fire). `x_infeasible`/`y_infeasible` are
    decided from title measurement and viewport alone
    (`recentering_is_feasible`), independently per axis, so a regression
    that let one axis's suppression leak into the other axis's solve would
    show up here as either the x title moving (wrongly un-suppressed) or
    the y title staying put (wrongly suppressed too).
    """

    INFEASIBLE_X_TITLE = "X" * 120
    FEASIBLE_X_TITLE = "A moderately long horizontal axis title here"
    NEEDS_CORRECTION_Y_TITLE = "A moderately long vertical axis title here"

    @classmethod
    def _repro_chart(cls, x_title: str | None = None, x_values=None, y_values=None):
        if x_title is None:
            x_title = cls.INFEASIBLE_X_TITLE
        if x_values is None:
            x_values = [0.0, 1.0, 2.0, 3.0]
        if y_values is None:
            y_values = [0.0, 1.0, 2.0, 3.0]
        df = pl.DataFrame({"x": x_values, "y": y_values})
        return (
            fr.Chart(df)
            .mark_point()
            .encode(
                x=fr.X("x", title=x_title),
                y=fr.Y("y", title=cls.NEEDS_CORRECTION_Y_TITLE),
            )
            .properties(width=280, height=250)
        )

    @staticmethod
    def _x_title_x(chart, title: str) -> float:
        svg = chart.to_svg()
        m = re.search(
            r'<text[^>]*x="(-?[\d.]+)"[^>]*text-anchor="middle"[^>]*>'
            + re.escape(title)
            + r"</text>",
            svg,
        )
        assert m is not None, f"expected the x-axis title text in the SVG: {svg}"
        return float(m.group(1))

    @classmethod
    def _y_title_y(cls, chart) -> float:
        svg = chart.to_svg()
        m = re.search(
            r'<text[^>]*y="(-?[\d.]+)"[^>]*transform="rotate\(-90[^"]*"[^>]*>'
            + re.escape(cls.NEEDS_CORRECTION_Y_TITLE)
            + r"</text>",
            svg,
        )
        if m is None:
            m = re.search(
                r'<text[^>]*transform="rotate\(-90[^"]*"[^>]*y="(-?[\d.]+)"[^>]*>'
                + re.escape(cls.NEEDS_CORRECTION_Y_TITLE)
                + r"</text>",
                svg,
            )
        assert m is not None, f"expected the y-axis title text in the SVG: {svg}"
        return float(m.group(1))

    def test_x_title_suppressed_while_y_title_still_recenters(self):
        base = self._repro_chart()
        auto_false = base.configure_padding(auto=False)
        auto_true = base.configure_padding(auto=True)

        x_title_auto_false = self._x_title_x(auto_false, self.INFEASIBLE_X_TITLE)
        x_title_auto_true = self._x_title_x(auto_true, self.INFEASIBLE_X_TITLE)
        assert x_title_auto_false == pytest.approx(163.472, abs=1e-3)
        assert x_title_auto_true == pytest.approx(x_title_auto_false, abs=1e-9), (
            "the x title is wider than the viewport itself (the genuine "
            "residual) — its own correction must stay suppressed regardless "
            f"of the unrelated y title's fate ({x_title_auto_true} vs "
            f"{x_title_auto_false})"
        )

        y_title_auto_false = self._y_title_y(auto_false)
        y_title_auto_true = self._y_title_y(auto_true)
        assert y_title_auto_false == pytest.approx(103.48, abs=1e-3)
        assert y_title_auto_true == pytest.approx(123.145, abs=1e-3), (
            "the y title is no wider than the viewport (a genuine, feasible "
            "recenter) and must still move under auto=True even though the "
            f"x axis's own title is infeasible ({y_title_auto_true} vs "
            f"{y_title_auto_false})"
        )

        # Anti-vacuity control: the SAME y title, width and height, but with
        # a FEASIBLE x title on a sparser x domain (0..10, not 0..3) so the
        # x tick labels stay flat and equal in count at auto=True and
        # auto=False -- its own anchor DOES move under `auto=True`, proving
        # the equality pin above is x's genuine infeasibility residual and
        # not `auto` being inert on x for this fixture/viewport.
        feasible_base = self._repro_chart(
            self.FEASIBLE_X_TITLE, x_values=[0.0, 10.0], y_values=[0.0, 1.0]
        )
        feasible_auto_false = feasible_base.configure_padding(auto=False)
        feasible_auto_true = feasible_base.configure_padding(auto=True)

        feasible_x_title_auto_false = self._x_title_x(feasible_auto_false, self.FEASIBLE_X_TITLE)
        feasible_x_title_auto_true = self._x_title_x(feasible_auto_true, self.FEASIBLE_X_TITLE)
        assert feasible_x_title_auto_false == pytest.approx(163.472, abs=1e-3)
        assert feasible_x_title_auto_true == pytest.approx(148.943, abs=1e-3), (
            "a feasible x title on this same fixture must still recenter "
            f"under auto=True ({feasible_x_title_auto_true} vs "
            f"{feasible_x_title_auto_false})"
        )


class TestPaddingExceedsViewportMessage:
    """D10, spec §4.7 — the T6 quality-review banked repro:
    ``configure_padding(top=1e9)`` previously raised ``"layout failed:
    padding 16 exceeds viewport dimension 480"`` — the theme DEFAULT (16),
    not the caller's 1e9, and it never named which side. The fixed message
    must name the caller's own value and the side.
    """

    def test_huge_top_padding_names_caller_value_and_side(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match=r"padding top=1000000000 exceeds"):
            (fr.Chart(df).mark_point().encode(x="x", y="y").configure_padding(top=1e9).to_svg())

    def test_huge_right_padding_names_right_not_top(self):
        """A different side set huge must name THAT side, not `"top"`
        (proving the fix picks the actual offending side, not a hardcoded one)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match=r"padding right=1000000000 exceeds"):
            (fr.Chart(df).mark_point().encode(x="x", y="y").configure_padding(right=1e9).to_svg())
