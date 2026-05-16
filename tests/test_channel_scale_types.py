"""Verify encoding channels work correctly with different scale types.

Catches cases where a channel silently falls back to defaults when
an unexpected scale type is provided (e.g. ordinal size, ordinal opacity).
"""

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 10.0],
            "cat": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "sz": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "op": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _render(chart) -> str:
    """Render a chart to SVG and return the string."""
    return chart.show_svg()


def _is_valid_svg(svg: str) -> bool:
    """Check that the output looks like a valid SVG."""
    return "<svg" in svg and "</svg>" in svg


# ---------------------------------------------------------------------------
# Color channel + quantitative (continuous colormap)
# ---------------------------------------------------------------------------

class TestColorQuantitative:
    def test_color_quantitative_renders(self, df):
        """Color channel with quantitative field produces valid SVG."""
        svg = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="sz:Q")
        )
        assert _is_valid_svg(svg)

    def test_color_quantitative_affects_output(self, df):
        """Color encoding with quantitative field changes SVG vs no color."""
        svg_with_color = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="sz:Q")
        )
        svg_no_color = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        )
        assert svg_with_color != svg_no_color, (
            "Color quantitative encoding did not change SVG output"
        )

    def test_color_quantitative_explicit_channel(self, df):
        """Color channel via explicit fm.Color object with quantitative type."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color=fm.Color("sz", type_="Q"))
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# Color channel + ordinal/nominal (categorical palette)
# ---------------------------------------------------------------------------

class TestColorOrdinal:
    def test_color_nominal_renders(self, df):
        """Color channel with nominal field produces valid SVG."""
        svg = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
        )
        assert _is_valid_svg(svg)

    def test_color_nominal_affects_output(self, df):
        """Color encoding with nominal field changes SVG vs no color."""
        svg_with_color = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
        )
        svg_no_color = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        )
        assert svg_with_color != svg_no_color, (
            "Color nominal encoding did not change SVG output"
        )

    def test_color_nominal_explicit_channel(self, df):
        """Color channel via explicit fm.Color object with nominal type."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color=fm.Color("cat", type_="N"))
        )
        assert _is_valid_svg(svg)

    def test_color_quantitative_vs_nominal_differ(self, df):
        """Quantitative and nominal color produce different SVG (different scales)."""
        svg_quant = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="sz:Q")
        )
        svg_nominal = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
        )
        assert svg_quant != svg_nominal, (
            "Quantitative and nominal color encodings produced identical SVG"
        )


# ---------------------------------------------------------------------------
# Size channel + quantitative (continuous size scale)
# ---------------------------------------------------------------------------

class TestSizeQuantitative:
    def test_size_quantitative_renders(self, df):
        """Size channel with quantitative field produces valid SVG."""
        svg = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", size="sz:Q")
        )
        assert _is_valid_svg(svg)

    def test_size_quantitative_affects_output(self, df):
        """Size encoding with quantitative field changes SVG vs no size."""
        svg_with_size = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", size="sz:Q")
        )
        svg_no_size = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        )
        assert svg_with_size != svg_no_size, (
            "Size quantitative encoding did not change SVG output"
        )

    def test_size_quantitative_explicit_channel(self, df):
        """Size channel via explicit fm.Size object with quantitative type."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", size=fm.Size("sz", type_="Q"))
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# Size channel + ordinal (discrete sizes — potentially unsupported)
# ---------------------------------------------------------------------------

class TestSizeOrdinal:
    def test_size_ordinal_no_crash(self, df):
        """Size channel with ordinal field does not crash."""
        svg = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", size="cat:N")
        )
        assert _is_valid_svg(svg)

    def test_size_ordinal_explicit_channel_no_crash(self, df):
        """Size channel via explicit fm.Size with nominal type does not crash."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", size=fm.Size("cat", type_="N"))
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# Opacity channel + quantitative (continuous opacity)
# ---------------------------------------------------------------------------

class TestOpacityQuantitative:
    def test_opacity_quantitative_renders(self, df):
        """Opacity channel with quantitative field produces valid SVG."""
        svg = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", opacity="op:Q")
        )
        assert _is_valid_svg(svg)

    def test_opacity_quantitative_affects_output(self, df):
        """Opacity encoding with quantitative field changes SVG vs no opacity."""
        svg_with_opacity = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", opacity="op:Q")
        )
        svg_no_opacity = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        )
        assert svg_with_opacity != svg_no_opacity, (
            "Opacity quantitative encoding did not change SVG output"
        )

    def test_opacity_quantitative_explicit_channel(self, df):
        """Opacity channel via explicit fm.Opacity object with quantitative type."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", opacity=fm.Opacity("op", type_="Q"))
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# Opacity channel + ordinal (discrete opacity — potentially unsupported)
# ---------------------------------------------------------------------------

class TestOpacityOrdinal:
    def test_opacity_ordinal_no_crash(self, df):
        """Opacity channel with nominal field does not crash."""
        svg = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", opacity="cat:N")
        )
        assert _is_valid_svg(svg)

    def test_opacity_ordinal_explicit_channel_no_crash(self, df):
        """Opacity channel via explicit fm.Opacity with nominal type does not crash."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", opacity=fm.Opacity("cat", type_="N"))
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# StrokeWidth channel + quantitative (continuous width)
# ---------------------------------------------------------------------------

class TestStrokeWidthQuantitative:
    def test_stroke_width_quantitative_renders(self, df):
        """StrokeWidth channel with quantitative field produces valid SVG."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", stroke_width="sz:Q")
        )
        assert _is_valid_svg(svg)

    def test_stroke_width_quantitative_affects_output(self, df):
        """StrokeWidth encoding with quantitative field changes SVG vs none."""
        svg_with_sw = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", stroke_width="sz:Q")
        )
        svg_no_sw = _render(
            fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        )
        assert svg_with_sw != svg_no_sw, (
            "StrokeWidth quantitative encoding did not change SVG output"
        )

    def test_stroke_width_quantitative_explicit_channel(self, df):
        """StrokeWidth via explicit fm.StrokeWidth object with quantitative type."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q", y="y:Q", stroke_width=fm.StrokeWidth("sz", type_="Q")
            )
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# Log scale tests
# ---------------------------------------------------------------------------

class TestLogScale:
    def test_log_scale_x_axis_renders(self, df):
        """Log scale on x-axis with positive data produces valid SVG."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x:Q", scale={"type": "log"}), y="y:Q")
        )
        assert _is_valid_svg(svg)

    def test_log_scale_y_axis_renders(self, df):
        """Log scale on y-axis with positive data produces valid SVG."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y=fm.Y("y:Q", scale={"type": "log"}))
        )
        assert _is_valid_svg(svg)

    def test_log_scale_with_log_scale_object(self, df):
        """LogScale object on x-axis with positive data produces valid SVG."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x:Q", scale=fm.LogScale(base=10.0)), y="y:Q")
        )
        assert _is_valid_svg(svg)

    def test_log_scale_differs_from_linear(self, df):
        """Log scale produces different output than default linear scale."""
        svg_log = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x:Q", scale={"type": "log"}), y="y:Q")
        )
        svg_linear = _render(
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
        )
        assert svg_log != svg_linear, (
            "Log scale produced identical SVG to linear scale"
        )

    def test_log_scale_with_explicit_domain(self, df):
        """LogScale with explicit domain renders without crash."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(
                x=fm.X("x:Q", scale=fm.LogScale(domain=[1.0, 200.0], base=10.0)),
                y="y:Q",
            )
        )
        assert _is_valid_svg(svg)

    def test_log_scale_on_color_channel(self, df):
        """Log scale on color channel does not crash."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                color=fm.Color("sz", type_="Q", scale={"type": "log"}),
            )
        )
        assert _is_valid_svg(svg)


# ---------------------------------------------------------------------------
# Combined channel tests
# ---------------------------------------------------------------------------

class TestCombinedChannels:
    def test_multiple_quantitative_channels(self, df):
        """Multiple quantitative channels simultaneously produce valid SVG."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                color="sz:Q",
                size="sz:Q",
                opacity="op:Q",
            )
        )
        assert _is_valid_svg(svg)

    def test_mixed_quantitative_and_nominal(self, df):
        """Mix of quantitative and nominal channels produces valid SVG."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                color="cat:N",
                size="sz:Q",
                opacity="op:Q",
            )
        )
        assert _is_valid_svg(svg)

    def test_all_channels_nominal(self, df):
        """All appearance channels with nominal field do not crash."""
        svg = _render(
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                color="cat:N",
                size="cat:N",
                opacity="cat:N",
            )
        )
        assert _is_valid_svg(svg)
