"""Regression: infinity in data columns must not poison scale domain computation."""

import polars as pl

import ferrum as fm


def _count_circles(svg: str) -> int:
    return svg.count("<circle")


def _count_visible_rects(svg: str) -> int:
    import re
    rects = re.findall(r'<rect[^>]*width="([^"]+)"', svg)
    return sum(1 for w in rects if float(w) > 1.0)


class TestInfinityDomainRegression:
    """Regression: inf/NaN in data must be skipped during domain computation,
    not included — including them produces [min, inf] domains that make
    all finite-valued marks invisible."""

    def test_positive_inf_in_y_skips_row(self):
        """Regression: +inf y value skipped, finite points still render."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, float("inf"), 30.0]})
        svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        assert _count_circles(svg) == 2

    def test_negative_inf_in_y_skips_row(self):
        """Regression: -inf y value skipped, finite points still render."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, float("-inf"), 30.0]})
        svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        assert _count_circles(svg) == 2

    def test_positive_inf_in_x_skips_row(self):
        """Regression: +inf x value skipped, finite points still render."""
        df = pl.DataFrame({"x": [1.0, float("inf"), 3.0], "y": [10.0, 20.0, 30.0]})
        svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        assert _count_circles(svg) == 2

    def test_negative_inf_in_x_skips_row(self):
        """Regression: -inf x value skipped, finite points still render."""
        df = pl.DataFrame({"x": [float("-inf"), 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        assert _count_circles(svg) == 2

    def test_all_inf_produces_valid_svg(self):
        """Regression: all-infinity data must not crash, should produce empty chart."""
        df = pl.DataFrame({"x": [float("inf"), float("-inf")], "y": [float("inf"), float("-inf")]})
        svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        assert svg.startswith("<svg")

    def test_inf_in_bar_chart(self):
        """Regression: bar mark with inf in y still renders finite bars."""
        df = pl.DataFrame({"x": ["A", "B", "C"], "y": [10.0, float("inf"), 30.0]})
        svg = fm.Chart(df).mark_bar().encode(x="x:N", y="y").show_svg()
        assert _count_visible_rects(svg) >= 2

    def test_inf_in_line_chart(self):
        """Regression: line mark with inf in y still renders line segments."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, float("inf"), 30.0, 40.0]})
        svg = fm.Chart(df).mark_line().encode(x="x", y="y").show_svg()
        assert "<path" in svg or "<line" in svg

    def test_inf_in_area_chart(self):
        """Regression: area mark with inf in y still renders."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, float("inf"), 30.0, 40.0]})
        svg = fm.Chart(df).mark_area().encode(x="x", y="y").show_svg()
        assert svg.startswith("<svg")

    def test_inf_in_size_encoding(self):
        """Regression: inf in size column doesn't poison size scale."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "s": [5.0, float("inf"), 15.0],
        })
        svg = fm.Chart(df).mark_point().encode(x="x", y="y", size="s").show_svg()
        assert _count_circles(svg) >= 2
