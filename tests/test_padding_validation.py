"""Tests for padding configuration validation (NF-B5/B6/B7).

Validates that padding values are rejected at the Python boundary when:
- Negative: typed ValueError at construction
- Non-numeric (e.g., NormCoord): typed ValueError naming the pixel contract
- Valid pixel values: accepted without change (byte-identity verified)
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
        """A chart with override negative padding raises ValueError (override path validation)."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        # .override(padding_top=-10) should raise ValueError during validation
        with pytest.raises(ValueError, match="padding_top.*non-negative"):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(padding_top=-10).to_svg()

    def test_chart_with_override_non_numeric_padding_raises(self):
        """A chart with override non-numeric padding raises ValueError."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match="padding_right.*numeric pixel value"):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(padding_right="10px").to_svg()

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
        """.override(padding_top=nan).to_svg() raises a typed pixel-contract
        ValueError naming the override path, not the opaque wire-serializer
        error (``chart_config: expected value at line 1 column 21``) that
        NaN/inf produced before this predicate covered finiteness."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match="padding_top.*finite numeric pixel value"):
            fr.Chart(df).mark_point().encode(x="x", y="y").override(
                padding_top=float("nan")
            ).to_svg()

    def test_chart_with_override_inf_padding_raises_typed_valueerror(self):
        """.override(padding_top=inf).to_svg() raises a typed pixel-contract ValueError."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        with pytest.raises(ValueError, match="padding_top.*finite numeric pixel value"):
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
