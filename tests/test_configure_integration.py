"""Integration tests for every configure_*() field reaching the render pipeline.

Each test verifies that a specific configure field actually changes the rendered SVG.
Written TDD-style: these should fail until the Rust pipeline consumes the field.
"""

from __future__ import annotations

import re
import warnings

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
            .configure_axis(label_format="currency")
        )
        svg = chart.to_svg()
        assert "$" in svg, "Currency format should produce $ in tick labels"

    def test_label_format_raw(self) -> None:
        """configure_axis(label_format_raw='.0%') should produce % in tick labels."""
        df = pl.DataFrame({"x": ["a", "b", "c"], "y": [0.1, 0.5, 0.9]})
        chart = (
            fm.Chart(df).mark_bar().encode(x="x:N", y="y:Q").configure_axis(label_format_raw=".0%")
        )
        svg = chart.to_svg()
        assert "%" in svg, "Raw format .0% should produce % in tick labels"

    def test_tick_values(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(tick_values=[10, 30, 50]) should place ticks at exactly those values."""
        chart_default = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        chart_custom = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure(axis_y=AxisConfig(tick_values=[10, 30, 50]))
            .to_svg()
        )
        assert chart_default != chart_custom, "Custom tick_values should change SVG"
        # The specific values should appear as tick labels
        assert ">10<" in chart_custom or ">10</text>" in chart_custom
        assert ">30<" in chart_custom or ">30</text>" in chart_custom
        assert ">50<" in chart_custom or ">50</text>" in chart_custom

    def test_title_font_size(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(title_font_size=20) should change axis title rendering."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").labs(y="Values")
        svg_default = base.to_svg()
        svg_large = base.configure_axis(title_font_size=20).to_svg()
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
        svg = chart.to_svg()
        assert "ff0000" in svg.lower(), "Title color #ff0000 should appear in SVG"

    def test_title_padding(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(title_padding=30) should change layout."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").labs(y="Values")
        svg_default = base.to_svg()
        svg_padded = base.configure_axis(title_padding=30).to_svg()
        assert svg_default != svg_padded, "title_padding should change SVG layout"

    def test_grid_opacity(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(grid_opacity=0.3) should set grid-line opacity in the SVG."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        svg_default = base.to_svg()
        svg_faint = base.configure_axis(grid_opacity=0.3).to_svg()
        assert svg_default != svg_faint, "grid_opacity should change SVG"
        assert 'opacity="0.3"' in svg_faint, "grid_opacity=0.3 should appear in SVG"

    def test_min_extent(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(min_extent=120) should change axis-margin layout."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        svg_default = base.to_svg()
        svg_extent = base.configure_axis(min_extent=120).to_svg()
        assert svg_default != svg_extent, "min_extent should change SVG layout"

    def test_title_orient(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(title_orient='top') should reposition axis titles."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").labs(x="X", y="Y")
        svg_default = base.to_svg()
        svg_oriented = base.configure_axis(title_orient="top").to_svg()
        assert svg_default != svg_oriented, "title_orient should change SVG"

    def test_orient_not_a_general_axis_kwarg(self, scatter_df: pl.DataFrame) -> None:
        """`orient` is rejected on the general configure_axis (ambiguous across x/y).

        A single side value cannot be valid for both axes, so `orient` is not
        exposed as a configure_axis keyword; it is set per-axis instead.
        """
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.raises(TypeError):
            chart.configure_axis(orient="top")

    def test_orient_via_axis_x_renders(self, scatter_df: pl.DataFrame) -> None:
        """`configure(axis_x=AxisConfig(orient='top'))` moves the x axis to the top."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        svg_default = base.to_svg()
        svg_top = base.configure(axis_x=AxisConfig(orient="top")).to_svg()
        assert svg_default != svg_top, "axis_x orient='top' should change SVG"

    def test_orient_via_axis_y_renders(self, scatter_df: pl.DataFrame) -> None:
        """`configure(axis_y=AxisConfig(orient='right'))` moves the y axis to the right."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        svg_default = base.to_svg()
        svg_right = base.configure(axis_y=AxisConfig(orient="right")).to_svg()
        assert svg_default != svg_right, "axis_y orient='right' should change SVG"

    def test_general_axis_orient_cross_dimension_raises(self, scatter_df: pl.DataFrame) -> None:
        """A general-axis `orient` (via AxisConfig) is always cross-dimension for one axis.

        ``configure(axis=AxisConfig(orient=...))`` applies the same side to both x
        and y, so any value fails on whichever axis it does not belong to. This is
        the Rust-side fail-loud contract; we assert it surfaces as ``ValueError``.
        """
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match="invalid for the"):
            chart.configure(axis=AxisConfig(orient="top")).to_svg()

    def test_per_channel_grid_opacity_beats_chart_level(self, scatter_df: pl.DataFrame) -> None:
        """Per-channel fm.Axis(grid_opacity) wins over configure_axis(grid_opacity).

        With per-channel grid_opacity set on BOTH axes, the chart-level value is
        fully shadowed, so the output is byte-identical to the per-channel-only
        render.
        """
        per_channel = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(
                x=fm.X("x", axis=fm.Axis(grid_opacity=0.2)),
                y=fm.Y("y", axis=fm.Axis(grid_opacity=0.2)),
            )
        )
        svg_per_channel = per_channel.to_svg()
        svg_with_chart_level = per_channel.configure_axis(grid_opacity=0.8).to_svg()
        assert svg_with_chart_level == svg_per_channel, (
            "per-channel grid_opacity must fully shadow chart-level configure_axis"
        )
        assert 'opacity="0.2"' in svg_per_channel
        assert 'opacity="0.8"' not in svg_with_chart_level


class TestConfigureAxisXYDeprecation:
    """configure_axis(x=False)/(y=False) are vestigial no-ops (BUG 3).

    They must warn and steer the user to the working ``Chart.axis`` API, while
    still returning a renderable chart (non-breaking).
    """

    def test_configure_axis_y_false_warns(self, scatter_df: pl.DataFrame) -> None:
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.warns(DeprecationWarning, match="Chart.axis"):
            chart.configure_axis(y=False)

    def test_configure_axis_x_false_warns(self, scatter_df: pl.DataFrame) -> None:
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.warns(DeprecationWarning, match="Chart.axis"):
            chart.configure_axis(x=False)

    def test_configure_axis_x_false_still_renders(self, scatter_df: pl.DataFrame) -> None:
        """Non-breaking: deprecated flag is accepted and the chart still renders."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.warns(DeprecationWarning):
            configured = chart.configure_axis(x=False, label_angle=-45)
        svg = configured.to_svg()
        assert "<svg" in svg

    def test_configure_axis_grid_does_not_warn(self, scatter_df: pl.DataFrame) -> None:
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            chart.configure_axis(grid=True)
            chart.configure_axis()
            chart.configure_axis(label_angle=-45)


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
        svg = chart.to_svg()
        # At least one of the custom colors should appear in the SVG
        found = any(c[1:].lower() in svg.lower() for c in custom_colors)
        assert found, f"At least one of {custom_colors} should appear in SVG"

    def test_color_domain(self) -> None:
        """configure_color(domain=[0, 100]) should change color scale mapping."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "v": [10.0, 50.0, 90.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="v:Q")
        svg_default = base.to_svg()
        svg_custom = base.configure_color(domain=[0, 100]).to_svg()
        assert svg_default != svg_custom, "Custom color domain should change SVG"


# ---------------------------------------------------------------------------
# Legend config fields
# ---------------------------------------------------------------------------


class TestLegendConfigIntegration:
    def test_gradient_length(self) -> None:
        """configure_legend(gradient_length=200) should change legend size."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "v": [10.0, 50.0, 90.0]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="v:Q")
        svg_default = base.to_svg()
        svg_long = base.configure_legend(gradient_length=200).to_svg()
        assert svg_default != svg_long, "gradient_length should change SVG"

    def test_symbol_stroke_width(self, color_df: pl.DataFrame) -> None:
        """configure_legend(symbol_stroke_width=3) should set swatch stroke width to 3."""
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N")
        svg_default = base.to_svg()
        svg_thick = base.configure_legend(symbol_stroke_width=3).to_svg()
        assert svg_default != svg_thick, "symbol_stroke_width should change SVG"
        assert 'stroke-width="3"' in svg_thick, "symbol_stroke_width=3 should appear on swatches"

    def test_label_limit_truncates_with_ellipsis(self) -> None:
        """configure_legend(label_limit=40) should truncate a long label with an ellipsis."""
        long_label = "an extremely long category label that overflows"
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [10, 20, 30, 40],
                "g": [long_label, "short", long_label, "short"],
            }
        )
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="g:N")
        svg_default = base.to_svg()
        svg_limited = base.configure_legend(label_limit=40).to_svg()
        assert svg_default != svg_limited, "label_limit should change SVG"
        assert "…" in svg_limited, "label_limit should truncate with an ellipsis (…)"

    def test_row_padding(self, color_df: pl.DataFrame) -> None:
        """configure_legend(row_padding=20) should change vertical entry spacing."""
        base = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N")
        svg_default = base.to_svg()
        svg_spaced = base.configure_legend(row_padding=20).to_svg()
        assert svg_default != svg_spaced, "row_padding should change SVG layout"


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
        svg = chart.to_svg()
        assert "f0f0f0" in svg.lower(), "Band color should appear in SVG"


# ---------------------------------------------------------------------------
# Rendering correctness
# ---------------------------------------------------------------------------


class TestGoldenStability:
    """The new config fields must not perturb the default (no-config) render."""

    def test_no_config_render_byte_identical(self, color_df: pl.DataFrame) -> None:
        """Constructing a chart twice with no configure_* call yields identical SVG."""
        chart = fm.Chart(color_df).mark_point().encode(x="x", y="y", color="g:N")
        assert chart.to_svg() == chart.to_svg()

    def test_default_construction_unset_orphans_no_change(self, scatter_df: pl.DataFrame) -> None:
        """An AxisConfig/LegendConfig with only None orphans must not change output."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 50, 90], "g": ["a", "b", "c"]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y", color="g:N")
        svg_base = base.to_svg()
        # configure_* with every new param left at its default None must be a no-op.
        svg_axis = base.configure_axis().to_svg()
        svg_legend = base.configure_legend().to_svg()
        assert svg_axis == svg_base, "configure_axis() with no args must not change SVG"
        assert svg_legend == svg_base, "configure_legend() with no args must not change SVG"


class TestRenderingCorrectness:
    def test_rect_annotation_opacity(self, scatter_df: pl.DataFrame) -> None:
        """Rect annotation with opacity=0.2 should have fill-opacity < 1."""
        import ferrum.annotation as ann

        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y") + ann.rect(
            1, 10, 5, 50, fill="#ff0000", opacity=0.2
        )
        svg = chart.to_svg()
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
        svg = chart.to_svg()
        # Should have rect elements for bars (not all hidden at -99999)
        rects = re.findall(r"<rect [^>]*>", svg)
        visible_rects = [r for r in rects if "-99999" not in r]
        assert len(visible_rects) >= 5, f"Expected ≥5 visible rects, got {len(visible_rects)}"
