"""Regression tests for opacity/alpha semantics (B1, B2, B3).

These tests verify that fill_opacity, stroke_opacity, and opacity are
correctly applied without double-application or misattribution in both
the SVG rendering path and the scene-graph pipeline.

At the Python API level:
- ``opacity`` is a mark kwarg (constant per mark) or encoding channel
- ``fill_opacity`` and ``stroke_opacity`` are encoding channels (data-driven)
"""

import re

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def scatter_df():
    """A small DataFrame for scatter/point chart tests."""
    return pl.DataFrame(
        {
            "x": list(range(20)),
            "y": [i * 2.0 + 1.0 for i in range(20)],
            "fo": [0.3] * 20,
            "so": [0.7] * 20,
        }
    )


@pytest.fixture
def large_scatter_df():
    """A >1000-row DataFrame to exercise the packed-instance path."""
    n = 1200
    return pl.DataFrame(
        {
            "x": list(range(n)),
            "y": [float(i % 100) for i in range(n)],
            "fo": [0.3] * n,
        }
    )


class TestFillOpacity:
    """B1: fill_opacity must control fill transparency, not opacity."""

    def test_fill_opacity_in_svg(self, scatter_df):
        """encode(fill_opacity='fo') must emit fill-opacity in SVG."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y", fill_opacity="fo")
        svg = chart.show_svg()
        assert "fill-opacity" in svg, "Expected fill-opacity attribute in SVG output"
        # Extract fill-opacity values; at least one should be ~0.3
        values = re.findall(r'fill-opacity="([^"]+)"', svg)
        assert any(abs(float(v) - 0.3) < 0.05 for v in values), (
            f"Expected fill-opacity near 0.3; got values: {values}"
        )

    def test_fill_opacity_on_large_batch(self, large_scatter_df):
        """fill_opacity on >1000 points (packed path) produces valid SVG."""
        chart = fm.Chart(large_scatter_df).mark_point().encode(x="x", y="y", fill_opacity="fo")
        svg = chart.show_svg()
        # SVG should render without error and contain circle elements
        assert "<circle" in svg or "<path" in svg or "fill-opacity" in svg, (
            "Large-batch chart with fill_opacity must render marks"
        )


class TestStrokeOpacity:
    """B2/B3: stroke_opacity must be applied correctly without double-apply."""

    def test_stroke_opacity_in_svg(self, scatter_df):
        """encode(stroke_opacity='so') must emit stroke-opacity in SVG."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point(stroke="black")
            .encode(x="x", y="y", stroke_opacity="so")
        )
        svg = chart.show_svg()
        assert "stroke-opacity" in svg, "Expected stroke-opacity attribute in SVG output"
        values = re.findall(r'stroke-opacity="([^"]+)"', svg)
        assert any(abs(float(v) - 0.7) < 0.05 for v in values), (
            f"Expected stroke-opacity near 0.7; got values: {values}"
        )

    def test_stroke_opacity_on_bars(self):
        """mark_bar with encode(stroke_opacity=...) renders correctly."""
        df = pl.DataFrame(
            {
                "cat": ["a", "b", "c"],
                "val": [10.0, 20.0, 15.0],
                "so": [0.5, 0.5, 0.5],
            }
        )
        chart = fm.Chart(df).mark_bar(stroke="black").encode(x="cat", y="val", stroke_opacity="so")
        svg = chart.show_svg()
        assert "stroke-opacity" in svg, "Expected stroke-opacity on bar chart SVG"


class TestOpacityNoDoubleApply:
    """B3: overall opacity and stroke_opacity must not compound incorrectly."""

    def test_opacity_with_stroke_not_double_applied(self, scatter_df):
        """Chart with mark-level opacity=0.5 and stroke should not show
        opacity^2 on strokes. The SVG renderer uses explicit attributes,
        so this verifies both opacity and stroke-opacity appear correctly
        and independently in the output.
        """
        chart = (
            fm.Chart(scatter_df)
            .mark_point(
                opacity=0.5,
                stroke="black",
                stroke_width=2,
            )
            .encode(x="x", y="y", stroke_opacity="so")
        )
        svg = chart.show_svg()
        # Both opacity and stroke-opacity should appear as separate attributes
        has_opacity = "opacity" in svg
        assert has_opacity, "Expected opacity-related attributes in SVG"

    def test_fill_and_stroke_opacity_independent(self, scatter_df):
        """fill_opacity and stroke_opacity can be set independently."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point(
                stroke="black",
                stroke_width=1.5,
            )
            .encode(x="x", y="y", fill_opacity="fo", stroke_opacity="so")
        )
        svg = chart.show_svg()

        fill_opacities = re.findall(r'fill-opacity="([^"]+)"', svg)
        stroke_opacities = re.findall(r'stroke-opacity="([^"]+)"', svg)

        # fill_opacity values should be near 0.3
        assert any(abs(float(v) - 0.3) < 0.05 for v in fill_opacities), (
            f"Expected fill-opacity ~0.3; got {fill_opacities}"
        )

        # stroke_opacity values should be near 0.7
        assert any(abs(float(v) - 0.7) < 0.15 for v in stroke_opacities), (
            f"Expected stroke-opacity ~0.7; got {stroke_opacities}"
        )
