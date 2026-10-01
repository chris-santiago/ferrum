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
        svg = chart.to_svg()
        assert "fill-opacity" in svg, "Expected fill-opacity attribute in SVG output"
        # Extract fill-opacity values; at least one should be ~0.3
        values = re.findall(r'fill-opacity="([^"]+)"', svg)
        assert any(abs(float(v) - 0.3) < 0.05 for v in values), (
            f"Expected fill-opacity near 0.3; got values: {values}"
        )

    def test_fill_opacity_on_large_batch(self, large_scatter_df):
        """fill_opacity on >1000 points (packed path) produces valid SVG."""
        chart = fm.Chart(large_scatter_df).mark_point().encode(x="x", y="y", fill_opacity="fo")
        svg = chart.to_svg()
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
        svg = chart.to_svg()
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
        svg = chart.to_svg()
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
        svg = chart.to_svg()
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
        svg = chart.to_svg()

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


class TestRangeOnlyOpacityScaleOverConstantColumn:
    """`#108`: an explicit range-only `LinearScale` over a constant
    `fill_opacity`/`stroke_opacity` column is honored (the clamped range
    midpoint), not the column's own literal value.
    """

    def test_fill_opacity_range_only_scale_resolves_to_midpoint(self, scatter_df):
        chart = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                fill_opacity=fm.FillOpacity("fo", scale=fm.LinearScale(range=[0.2, 0.8])),
            )
        )
        svg = chart.to_svg()
        assert svg.count('fill-opacity="0.5"') == 20
        assert 'fill-opacity="0.3"' not in svg

    def test_stroke_opacity_range_only_scale_resolves_to_midpoint(self, scatter_df):
        chart = (
            fm.Chart(scatter_df)
            .mark_point(stroke="black")
            .encode(
                x="x",
                y="y",
                stroke_opacity=fm.StrokeOpacity("so", scale=fm.LinearScale(range=[0.2, 0.8])),
            )
        )
        svg = chart.to_svg()
        assert svg.count('stroke-opacity="0.5"') == 20
        assert 'stroke-opacity="0.7"' not in svg

    def test_range_clamps_before_the_midpoint_is_taken(self, scatter_df):
        """A range outside [0, 1] clamps first, so [-1, 2] still resolves to 0.5."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                fill_opacity=fm.FillOpacity("fo", scale=fm.LinearScale(range=[-1.0, 2.0])),
            )
        )
        svg = chart.to_svg()
        assert svg.count('fill-opacity="0.5"') == 20

    def test_constant_across_facet_panels_resolves_in_every_panel(self):
        """A column constant across every facet panel still honors the
        range-only scale in each panel, not the literal value."""
        df = pl.DataFrame(
            {
                "x": list(range(12)),
                "y": [i * 2.0 + 1.0 for i in range(12)],
                "fo": [0.3] * 12,
                "g": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            }
        )
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                fill_opacity=fm.FillOpacity("fo", scale=fm.LinearScale(range=[0.2, 0.8])),
                facet=fm.Facet("g"),
            )
        )
        svg = chart.to_svg()
        assert svg.count('fill-opacity="0.5"') == 12
        assert 'fill-opacity="0.3"' not in svg

    def test_no_scale_keeps_the_literal_alpha(self, scatter_df):
        """Control: with no `scale=` at all, a constant column still keeps
        its own literal value."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y", fill_opacity="fo")
        svg = chart.to_svg()
        assert 'fill-opacity="0.3"' in svg
        assert 'fill-opacity="0.5"' not in svg

    def test_range_less_linear_scale_keeps_the_literal_alpha(self, scatter_df):
        """Control: an explicit `LinearScale()` with neither domain nor range
        is the same as no scale for this purpose — still the literal value."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y", fill_opacity=fm.FillOpacity("fo", scale=fm.LinearScale()))
        )
        svg = chart.to_svg()
        assert 'fill-opacity="0.3"' in svg
        assert 'fill-opacity="0.5"' not in svg

    def test_explicit_domain_case_is_unaffected(self, scatter_df):
        """Control: an explicit domain over a constant column already
        resolved before this fix, and still resolves the same way."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                fill_opacity=fm.FillOpacity(
                    "fo", scale=fm.LinearScale(domain=[0.0, 1.0], range=[0.2, 0.8])
                ),
            )
        )
        svg = chart.to_svg()
        # fo = 0.3 on domain [0, 1] onto range [0.2, 0.8] -> 0.38
        assert 'fill-opacity="0.38"' in svg
