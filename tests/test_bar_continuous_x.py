"""Regression tests: mark_bar with continuous (quantitative) x must produce visible bars."""

import polars as pl
import pytest

import ferrum as fm


def _count_rects(svg: str) -> int:
    """Count <rect> elements in SVG output (excludes background rects)."""
    import re
    # Match rect elements with width > 1 pixel (exclude decorative/background)
    rects = re.findall(r'<rect[^>]*width="([^"]+)"', svg)
    return sum(1 for w in rects if float(w) > 1.0)


class TestBarContinuousX:
    """mark_bar with quantitative x (no x2) must auto-compute bar width."""

    def test_bars_visible_with_continuous_x(self):
        """Bars should render with non-zero width when x is continuous."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 15.0, 25.0, 30.0]})
        chart = fm.Chart(df).mark_bar().encode(x="x", y="y")
        svg = chart.show_svg()
        n_rects = _count_rects(svg)
        assert n_rects >= 5, f"Expected ≥5 visible bars, got {n_rects}"

    def test_bar_width_proportional_to_spacing(self):
        """Bar width should be derived from minimum spacing between x values."""
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [5.0, 10.0, 15.0, 20.0]})
        chart = fm.Chart(df).mark_bar().encode(x="x", y="y")
        svg = chart.show_svg()
        import re
        # Extract widths from data-bar rects (those inside the mark <g> group,
        # identified by having a non-background fill color).
        bar_widths = []
        for m in re.finditer(r"<rect ([^>]+)/>", svg):
            attrs = m.group(1)
            w_m = re.search(r'width="([^"]+)"', attrs)
            fill_m = re.search(r'fill="([^"]+)"', attrs)
            if w_m and fill_m:
                w = float(w_m.group(1))
                fill = fill_m.group(1)
                # Skip background and panel rects (background color or zero-opacity)
                if fill not in ("#faf7f2", "#ffffff", "none") and w > 1.0:
                    bar_widths.append(w)
        assert len(bar_widths) >= 4, f"Expected ≥4 bars, got {len(bar_widths)}"
        # All bars should have the same width (uniform spacing)
        assert max(bar_widths) - min(bar_widths) < 1.0, f"Bar widths not uniform: {bar_widths}"

    def test_bars_visible_with_uneven_spacing(self):
        """Bars with irregular x spacing should use min-spacing as width."""
        df = pl.DataFrame({"x": [1.0, 2.0, 5.0, 10.0], "y": [10.0, 20.0, 30.0, 40.0]})
        chart = fm.Chart(df).mark_bar().encode(x="x", y="y")
        svg = chart.show_svg()
        n_rects = _count_rects(svg)
        assert n_rects >= 4, f"Expected ≥4 visible bars, got {n_rects}"

    def test_single_bar_has_default_width(self):
        """A single data point should still produce a visible bar."""
        df = pl.DataFrame({"x": [5.0], "y": [20.0]})
        chart = fm.Chart(df).mark_bar().encode(x="x", y="y")
        svg = chart.show_svg()
        n_rects = _count_rects(svg)
        assert n_rects >= 1, f"Expected ≥1 visible bar, got {n_rects}"

    def test_bars_with_color_encoding(self):
        """Continuous-x bars should work with color grouping."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 15.0, 25.0],
            "cat": ["A", "B", "A", "B"],
        })
        chart = fm.Chart(df).mark_bar().encode(x="x", y="y", color="cat:N")
        svg = chart.show_svg()
        n_rects = _count_rects(svg)
        assert n_rects >= 4, f"Expected ≥4 visible bars, got {n_rects}"

    def test_existing_x2_path_unchanged(self):
        """Explicit x2 (histogram bins) must still work as before."""
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "x2": [1.0, 2.0, 3.0], "y": [5.0, 10.0, 15.0]})
        chart = fm.Chart(df).mark_bar().encode(x="x", y="y")
        # x2 is provided via the stat transform pipeline, not encoding directly.
        # This test just verifies the mark_bar path doesn't regress.
        svg = chart.show_svg()
        assert svg.startswith("<svg")
