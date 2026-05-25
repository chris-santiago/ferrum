"""Integration tests for every configure_*() field reaching the render pipeline.

Each test verifies that a specific configure field actually changes the rendered SVG.
Written TDD-style: these should fail until the Rust pipeline consumes the field.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.configure import AxisConfig


@pytest.fixture()
def scatter_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})


@pytest.fixture()
def color_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "g": ["a", "b", "c", "d"],
        }
    )


# ---------------------------------------------------------------------------
# Axis config fields
# ---------------------------------------------------------------------------


class TestAxisConfigIntegration:
    def test_label_format_currency(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(label_format='currency') should produce $ in tick labels."""
        chart = (
            fm.Chart(scatter_df)
            .mark_bar()
            .encode(x="x:N", y="y:Q")
            .configure_axis(y=True, x=False, label_format="currency")
        )
        svg = chart.show_svg()
        assert "$" in svg, "Currency format should produce $ in tick labels"

    def test_label_format_raw(self) -> None:
        """configure_axis(label_format_raw='.0%') should produce % in tick labels."""
        df = pl.DataFrame({"x": ["a", "b", "c"], "y": [0.1, 0.5, 0.9]})
        chart = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="x:N", y="y:Q")
            .configure_axis(y=True, x=False, label_format_raw=".0%")
        )
        svg = chart.show_svg()
        assert "%" in svg, "Raw format .0% should produce % in tick labels"

    def test_tick_values(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(tick_values=[10, 30, 50]) should place ticks at exactly those values."""
        chart_default = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").show_svg()
        chart_custom = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure(axis_y=AxisConfig(tick_values=[10, 30, 50]))
            .show_svg()
        )
        assert chart_default != chart_custom, "Custom tick_values should change SVG"
        # The specific values should appear as tick labels
        assert ">10<" in chart_custom or ">10</text>" in chart_custom
        assert ">30<" in chart_custom or ">30</text>" in chart_custom
        assert ">50<" in chart_custom or ">50</text>" in chart_custom

    def test_title_font_size(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(title_font_size=20) should change axis title rendering."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").labs(y="Values")
        svg_default = base.show_svg()
        svg_large = base.configure_axis(title_font_size=20).show_svg()
        assert svg_default != svg_large, "title_font_size should change SVG"

    def test_title_color(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(title_color='#ff0000') should color axis titles red."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .labs(y="Values")
            .configure_axis(title_color="#ff0000")
        )
        svg = chart.show_svg()
        assert "ff0000" in svg.lower(), "Title color #ff0000 should appear in SVG"

    def test_title_padding(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(title_padding=30) should change layout."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").labs(y="Values")
        svg_default = base.show_svg()
        svg_padded = base.configure_axis(title_padding=30).show_svg()
        assert svg_default != svg_padded, "title_padding should change SVG layout"


# ---------------------------------------------------------------------------
# Color config fields
# ---------------------------------------------------------------------------


class TestColorConfigIntegration:
    def test_color_range(self, color_df: pl.DataFrame) -> None:
        """configure_color(range=[...]) should use those exact colors."""
        custom_colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        chart = (
            fm.Chart(color_df)
            .mark_point()
            .encode(x="x", y="y", color="g:N")
            .configure_color(range=custom_colors)
        )
        svg = chart.show_svg()
        # At least one of the custom colors should appear in the SVG
        found = any(c[1:].lower() in svg.lower() for c in custom_colors)
        assert found, f"At least one of {custom_colors} should appear in SVG"

    def test_color_domain(self) -> None:
        """configure_color(domain=[0, 100]) should change color scale mapping."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "v": [10.0, 50.0, 90.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="v:Q")
        svg_default = base.show_svg()
        svg_custom = base.configure_color(domain=[0, 100]).show_svg()
        assert svg_default != svg_custom, "Custom color domain should change SVG"


# ---------------------------------------------------------------------------
# Legend config fields
# ---------------------------------------------------------------------------


class TestLegendConfigIntegration:
    def test_gradient_length(self) -> None:
        """configure_legend(gradient_length=200) should change legend size."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "v": [10.0, 50.0, 90.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="v:Q")
        svg_default = base.show_svg()
        svg_long = base.configure_legend(gradient_length=200).show_svg()
        assert svg_default != svg_long, "gradient_length should change SVG"


# ---------------------------------------------------------------------------
# Grid config fields
# ---------------------------------------------------------------------------


class TestGridConfigIntegration:
    def test_band_colors(self, scatter_df: pl.DataFrame) -> None:
        """configure_grid(band_colors=['#f0f0f0', '#ffffff']) should add alternating bands."""
        chart = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_grid(y=True, band_colors=["#f0f0f0", "#ffffff"])
        )
        svg = chart.show_svg()
        assert "f0f0f0" in svg.lower(), "Band color should appear in SVG"


# ---------------------------------------------------------------------------
# Rendering correctness
# ---------------------------------------------------------------------------


class TestRenderingCorrectness:
    def test_rect_annotation_opacity(self, scatter_df: pl.DataFrame) -> None:
        """Rect annotation with opacity=0.2 should have fill-opacity < 1."""
        import ferrum.annotation as ann

        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y") + ann.rect(
            1, 10, 5, 50, fill="#ff0000", opacity=0.2
        )
        svg = chart.show_svg()
        assert "fill-opacity" in svg, "Rect annotation should have fill-opacity attribute in SVG"

    def test_break_axis_bars_visible(self) -> None:
        """Break axis should show bars in retained segments, not hide them."""
        df = pl.DataFrame(
            {
                "server": ["web-01", "web-02", "web-03", "web-04", "db-01"],
                "response_ms": [42.0, 38.0, 45.0, 1240.0, 51.0],
            }
        )
        chart = fm.Chart(df).mark_bar().encode(x="server:N", y="response_ms:Q") + fm.BreakAxis(
            axis="y", gap=(80, 1180), break_style="zigzag"
        )
        svg = chart.show_svg()
        # Should have rect elements for bars (not all hidden at -99999)
        rects = re.findall(r"<rect [^>]*>", svg)
        visible_rects = [r for r in rects if "-99999" not in r]
        assert len(visible_rects) >= 5, f"Expected ≥5 visible rects, got {len(visible_rects)}"
