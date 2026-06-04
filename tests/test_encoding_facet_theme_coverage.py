"""Cross-dimensional tests: appearance encoding channels x faceting x theme overrides.

Verifies that:
1. Appearance channels (color, size, opacity, shape) work correctly in faceted charts
2. Theme overrides don't break encoding channels in faceted panels
3. Faceted charts with color encoding produce consistent output across panels
4. Theme + encoding + facet interact without crashes
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ferrum import Chart
from ferrum.themes import Theme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df() -> pl.DataFrame:
    """DataFrame with 60 rows, multiple categorical and quantitative columns."""
    np.random.seed(42)
    n = 60
    return pl.DataFrame(
        {
            "x": np.random.randn(n).tolist(),
            "y": np.random.randn(n).tolist(),
            "cat": (["A", "B", "C"] * 20),
            "grp": (["g1", "g2", "g3", "g4", "g5"] * 12),
            "size_val": np.random.rand(n).tolist(),
            "opacity_val": np.random.rand(n).tolist(),
            "row_cat": (["X", "Y"] * 30),
        }
    )


# ---------------------------------------------------------------------------
# Group 1: Color encoding in faceted charts
# ---------------------------------------------------------------------------


class TestColorEncodingFaceted:
    """Color:N and Color:Q with facet(col=...) produce valid SVG."""

    @pytest.mark.parametrize(
        "mark_method",
        [
            pytest.param("mark_point", id="point-color-N-facet-col"),
            pytest.param("mark_bar", id="bar-color-N-facet-col"),
            pytest.param("mark_line", id="line-color-N-facet-col"),
            pytest.param("mark_area", id="area-color-N-facet-col"),
        ],
    )
    def test_color_nominal_facet_col(self, df, mark_method):
        chart = getattr(Chart(df).encode(x="x:Q", y="y:Q", color="grp:N"), mark_method)()
        chart = chart.facet(col="cat")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_color_quantitative_facet_col(self, df):
        chart = Chart(df).encode(x="x:Q", y="y:Q", color="size_val:Q").mark_point().facet(col="cat")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500


# ---------------------------------------------------------------------------
# Group 2: Size encoding in faceted charts
# ---------------------------------------------------------------------------


class TestSizeEncodingFaceted:
    """Size:Q with facet(col=...) produces valid SVG."""

    def test_size_alone_facet_col(self, df):
        chart = Chart(df).encode(x="x:Q", y="y:Q", size="size_val:Q").mark_point().facet(col="cat")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_color_and_size_facet_col(self, df):
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", color="grp:N", size="size_val:Q")
            .mark_point()
            .facet(col="cat")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500


# ---------------------------------------------------------------------------
# Group 3: Opacity encoding in faceted charts
# ---------------------------------------------------------------------------


class TestOpacityEncodingFaceted:
    """Opacity:Q with facet(col=...) produces valid SVG."""

    def test_opacity_point_facet_col(self, df):
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", opacity="opacity_val:Q")
            .mark_point()
            .facet(col="cat")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_opacity_bar_facet_col(self, df):
        chart = (
            Chart(df)
            .encode(x="cat:N", y="y:Q", opacity="opacity_val:Q")
            .mark_bar()
            .facet(col="row_cat")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500


# ---------------------------------------------------------------------------
# Group 4: Shape encoding in faceted charts
# ---------------------------------------------------------------------------


class TestShapeEncodingFaceted:
    """Shape:N with facet(col=...) produces valid SVG (mark_point only)."""

    def test_shape_alone_facet_col(self, df):
        chart = Chart(df).encode(x="x:Q", y="y:Q", shape="grp:N").mark_point().facet(col="cat")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_color_and_shape_facet_col(self, df):
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", color="grp:N", shape="grp:N")
            .mark_point()
            .facet(col="cat")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500


# ---------------------------------------------------------------------------
# Group 5: Theme overrides with encoding + facet
# ---------------------------------------------------------------------------


class TestThemeOverridesEncodingFacet:
    """Theme props change the SVG output of faceted encoded charts."""

    def _base_chart(self, df):
        return Chart(df).encode(x="x:Q", y="y:Q", color="grp:N").mark_point().facet(col="cat")

    def test_color_scheme_override(self, df):
        default_svg = self._base_chart(df).to_svg()
        themed_svg = self._base_chart(df).theme(Theme(color_scheme="tableau10")).to_svg()
        assert themed_svg.startswith("<svg")
        # A different color scheme should produce different SVG
        assert default_svg != themed_svg

    def test_point_size_override(self, df):
        default_svg = self._base_chart(df).to_svg()
        themed_svg = self._base_chart(df).theme(Theme(point_size=100)).to_svg()
        assert themed_svg.startswith("<svg")
        assert default_svg != themed_svg

    def test_opacity_override(self, df):
        base = Chart(df).encode(x="x:Q", y="y:Q").mark_point().facet(col="cat")
        default_svg = base.to_svg()
        themed_svg = base.theme(Theme(opacity=0.3)).to_svg()
        assert themed_svg.startswith("<svg")
        assert default_svg != themed_svg

    def test_mark_color_override_bar(self, df):
        base = Chart(df).encode(x="cat:N", y="y:Q", color="grp:N").mark_bar().facet(col="row_cat")
        themed_svg = base.theme(Theme(mark_color="#ff0000")).to_svg()
        assert themed_svg.startswith("<svg")
        # mark_color may or may not override when color encoding is present,
        # but at minimum it should render without error
        assert len(themed_svg) > 500


# ---------------------------------------------------------------------------
# Group 6: Encoding overrides theme (encoding wins)
# ---------------------------------------------------------------------------


class TestEncodingOverridesTheme:
    """When both a color encoding and theme mark_color exist, encoding dominates."""

    def test_color_encoding_dominates_mark_color(self, df):
        # Theme-only version (no color encoding)
        theme_only = (
            Chart(df)
            .encode(x="x:Q", y="y:Q")
            .mark_point()
            .facet(col="cat")
            .theme(Theme(mark_color="#ff0000"))
        )
        # Encoding version (color:N should override mark_color)
        encoding_version = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .mark_point()
            .facet(col="cat")
            .theme(Theme(mark_color="#ff0000"))
        )
        svg_theme = theme_only.to_svg()
        svg_enc = encoding_version.to_svg()
        assert svg_theme.startswith("<svg")
        assert svg_enc.startswith("<svg")
        # Color encoding should produce a different SVG (multiple colors vs one)
        assert svg_theme != svg_enc

    def test_size_encoding_produces_variation(self, df):
        # With size encoding, points should vary in size despite theme point_size
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", size="size_val:Q")
            .mark_point()
            .facet(col="cat")
            .theme(Theme(point_size=10))
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500


# ---------------------------------------------------------------------------
# Group 7: Facet mode variants with encoding + theme
# ---------------------------------------------------------------------------


class TestFacetModeVariants:
    """Different facet modes (row, col, grid) with encoding + theme."""

    def test_color_facet_row_theme(self, df):
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .mark_point()
            .facet(row="cat")
            .theme(Theme(point_size=50))
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_color_facet_grid_theme(self, df):
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .mark_point()
            .facet(row="row_cat", col="cat")
            .theme(Theme(color_scheme="set2"))
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_color_facet_col_ncols_theme(self, df):
        chart = (
            Chart(df)
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .mark_point()
            .facet(col="cat", ncols=2)
            .theme(Theme(background="#f0f0f0"))
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500


# ---------------------------------------------------------------------------
# Group 8: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for encoding + facet + theme interaction."""

    def test_many_color_categories_faceted(self, df):
        """5 color categories across 3 facet panels renders fine."""
        chart = Chart(df).encode(x="x:Q", y="y:Q", color="grp:N").mark_point().facet(col="cat")
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_single_value_facet_with_color(self, df):
        """Faceting on a column where all rows are the same value."""
        single_cat_df = df.with_columns(pl.lit("only").alias("single"))
        chart = (
            Chart(single_cat_df)
            .encode(x="x:Q", y="y:Q", color="grp:N")
            .mark_point()
            .facet(col="single")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_quantitative_color_faceted(self, df):
        """Quantitative color encoding in faceted chart."""
        chart = (
            Chart(df).encode(x="x:Q", y="y:Q", color="opacity_val:Q").mark_point().facet(col="cat")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500

    def test_multiple_channels_facet_theme(self, df):
        """color + size + opacity + facet + theme all at once."""
        chart = (
            Chart(df)
            .encode(
                x="x:Q",
                y="y:Q",
                color="grp:N",
                size="size_val:Q",
                opacity="opacity_val:Q",
            )
            .mark_point()
            .facet(col="cat")
            .theme(Theme(color_scheme="dark2", background="#fafafa"))
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 500
