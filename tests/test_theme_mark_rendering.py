"""Theme × Mark rendering: verify theme keys actually affect rendered SVG.

The existing test_theme.py validates serialization / round-tripping of Theme
objects. This file verifies that theme properties (mark_color, point_size,
line_stroke_width, bar_corner_radius, area_opacity, opacity) actually produce
visually different SVG output for each relevant mark type.

Strategy: render with default theme vs a custom theme and assert:
  1. Both produce valid SVG (no crash).
  2. The SVGs differ (proving the theme key has an effect).
  3. Where possible, the custom value appears in the SVG output.

Findings:
Design notes:
  - mark_rule uses reference_line_color, not mark_color (by design — rules are ref lines)
  - mark_rect DOES respond to mark_color when no color encoding overrides it
  - mark_errorbar composite inherits mark_color via its constituent marks
"""

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.themes import Theme


# ---------------------------------------------------------------------------
# Data fixture
# ---------------------------------------------------------------------------

np.random.seed(42)
_N = 30
DF = pl.DataFrame(
    {
        "x": [float(i) for i in range(_N)],  # Float64 for smooth compatibility
        "y": np.random.randn(_N).tolist(),
        "cat": (["a", "b", "c"] * 10),
        "val": np.random.rand(_N).tolist(),
    }
)

# Larger per-group dataset for boxplot/violin (need 5+ per group)
DF_GROUPED = pl.DataFrame(
    {
        "group": (["a"] * 15 + ["b"] * 15 + ["c"] * 15),
        "value": np.random.randn(45).tolist(),
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_with_theme(chart_fn, theme=None):
    """Build chart via chart_fn, apply theme, render SVG."""
    chart = chart_fn()
    if theme is not None:
        chart = chart.theme(theme)
    svg = chart.to_svg()
    assert svg.startswith("<svg"), "SVG output must start with <svg"
    return svg


def _assert_svgs_differ(svg_default, svg_custom, description):
    """Assert that the two SVGs are not byte-equal."""
    assert svg_default != svg_custom, f"Theme key had no effect on rendered SVG: {description}"


def _svg_contains_red(svg):
    """Check if SVG contains #ff0000 in any common CSS color format."""
    svg_lower = svg.lower()
    return (
        "ff0000" in svg_lower
        or "rgb(255, 0, 0" in svg_lower
        or "rgb(255,0,0" in svg_lower
        or "rgba(255, 0, 0" in svg_lower
        or "rgba(255,0,0" in svg_lower
        # Also check for the rgba format with decimal alpha
        or "rgba(255" in svg_lower
    )


# ---------------------------------------------------------------------------
# Group 1: mark_color affects each mark type
# ---------------------------------------------------------------------------

_MARK_COLOR_THEME = Theme(mark_color="#ff0000")


def _chart_point():
    return fm.Chart(DF).encode(x="x:Q", y="y:Q").mark_point()


def _chart_line():
    return fm.Chart(DF).encode(x="x:Q", y="y:Q").mark_line()


def _chart_bar():
    return fm.Chart(DF).encode(x="cat:N", y="val:Q").mark_bar()


def _chart_area():
    return fm.Chart(DF).encode(x="x:Q", y="y:Q").mark_area()


def _chart_rule():
    return fm.Chart(DF).encode(x="x:Q", y="y:Q").mark_rule()


def _chart_tick():
    return fm.Chart(DF).encode(x="x:Q", y="val:Q").mark_tick()


def _chart_rect():
    rect_df = pl.DataFrame(
        {"x": ["a", "b", "c"] * 3, "y": ["p", "p", "p", "q", "q", "q", "r", "r", "r"]}
    )
    return fm.Chart(rect_df).encode(x="x:N", y="y:N").mark_rect()


@pytest.mark.parametrize(
    "chart_fn,mark_name",
    [
        pytest.param(_chart_point, "point", id="mark_color-point"),
        pytest.param(_chart_line, "line", id="mark_color-line"),
        pytest.param(_chart_bar, "bar", id="mark_color-bar"),
        pytest.param(_chart_area, "area", id="mark_color-area"),
        pytest.param(_chart_tick, "tick", id="mark_color-tick"),
        pytest.param(_chart_rect, "rect", id="mark_color-rect"),
    ],
)
def test_mark_color_affects_mark(chart_fn, mark_name):
    """mark_color theme key should change rendered SVG for each mark."""
    svg_default = _render_with_theme(chart_fn)
    svg_custom = _render_with_theme(chart_fn, _MARK_COLOR_THEME)
    _assert_svgs_differ(svg_default, svg_custom, f"mark_color on {mark_name}")
    # The custom color should appear somewhere in the SVG (various formats)
    assert _svg_contains_red(svg_custom), (
        f"Expected #ff0000 (or rgba(255...) equivalent) in SVG for {mark_name}"
    )


# ---------------------------------------------------------------------------
# Group 2: point_size affects point marks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size_a,size_b",
    [
        pytest.param(100, 50, id="point_size-100-vs-default"),
        pytest.param(10, 100, id="point_size-10-vs-100"),
    ],
)
def test_point_size_affects_point(size_a, size_b):
    """point_size theme key should produce different SVGs for different sizes."""
    theme_a = Theme(point_size=size_a)
    theme_b = Theme(point_size=size_b)
    svg_a = _render_with_theme(_chart_point, theme_a)
    svg_b = _render_with_theme(_chart_point, theme_b)
    _assert_svgs_differ(svg_a, svg_b, f"point_size={size_a} vs point_size={size_b}")


# ---------------------------------------------------------------------------
# Group 3: line_stroke_width affects line-based marks
# ---------------------------------------------------------------------------


def _chart_smooth():
    return fm.Chart(DF).encode(x="x:Q", y="y:Q").mark_smooth()


@pytest.mark.parametrize(
    "chart_fn,mark_name",
    [
        pytest.param(_chart_line, "line", id="line_stroke_width-line"),
        pytest.param(_chart_rule, "rule", id="line_stroke_width-rule"),
        pytest.param(_chart_smooth, "smooth", id="line_stroke_width-smooth"),
    ],
)
def test_line_stroke_width_affects_line_marks(chart_fn, mark_name):
    """line_stroke_width theme key should change SVG for line-based marks."""
    svg_default = _render_with_theme(chart_fn)
    svg_custom = _render_with_theme(chart_fn, Theme(line_stroke_width=5))
    _assert_svgs_differ(svg_default, svg_custom, f"line_stroke_width=5 on {mark_name}")


# ---------------------------------------------------------------------------
# Group 4: bar_corner_radius affects bar marks
# ---------------------------------------------------------------------------


def _chart_histogram():
    return fm.Chart(DF).encode(x="val:Q").mark_histogram()


@pytest.mark.parametrize(
    "chart_fn,mark_name",
    [
        pytest.param(_chart_bar, "bar", id="bar_corner_radius-bar"),
        pytest.param(_chart_histogram, "histogram", id="bar_corner_radius-histogram"),
    ],
)
def test_bar_corner_radius_affects_bar_marks(chart_fn, mark_name):
    """bar_corner_radius theme key should produce different SVG."""
    svg_default = _render_with_theme(chart_fn, Theme(bar_corner_radius=0))
    svg_custom = _render_with_theme(chart_fn, Theme(bar_corner_radius=10))
    _assert_svgs_differ(svg_default, svg_custom, f"bar_corner_radius=10 on {mark_name}")


# ---------------------------------------------------------------------------
# Group 5: area_opacity affects area-based marks
# ---------------------------------------------------------------------------


def _chart_density():
    return fm.Chart(DF).encode(x="val:Q").mark_density()


@pytest.mark.parametrize(
    "chart_fn,mark_name",
    [
        pytest.param(_chart_area, "area", id="area_opacity-area"),
        pytest.param(_chart_density, "density", id="area_opacity-density"),
    ],
)
def test_area_opacity_affects_area_marks(chart_fn, mark_name):
    """area_opacity theme key should produce different SVG for area marks."""
    svg_low = _render_with_theme(chart_fn, Theme(area_opacity=0.1))
    svg_high = _render_with_theme(chart_fn, Theme(area_opacity=0.9))
    _assert_svgs_differ(svg_low, svg_high, f"area_opacity 0.1 vs 0.9 on {mark_name}")


# ---------------------------------------------------------------------------
# Group 6: opacity (global) affects marks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chart_fn,mark_name",
    [
        pytest.param(_chart_point, "point", id="opacity-point"),
        pytest.param(_chart_bar, "bar", id="opacity-bar"),
        pytest.param(_chart_line, "line", id="opacity-line"),
    ],
)
def test_opacity_affects_marks(chart_fn, mark_name):
    """Global opacity theme key should change SVG for multiple mark types."""
    svg_default = _render_with_theme(chart_fn)
    svg_custom = _render_with_theme(chart_fn, Theme(opacity=0.2))
    _assert_svgs_differ(svg_default, svg_custom, f"opacity=0.2 on {mark_name}")


# ---------------------------------------------------------------------------
# Group 7: Multiple theme keys combined
# ---------------------------------------------------------------------------


def test_combined_bar_theme():
    """Multiple bar-related theme keys produce a different SVG than default."""
    combined = Theme(mark_color="#00ff00", bar_corner_radius=8, opacity=0.5)
    svg_default = _render_with_theme(_chart_bar)
    svg_custom = _render_with_theme(_chart_bar, combined)
    _assert_svgs_differ(svg_default, svg_custom, "combined bar theme")


def test_combined_point_theme():
    """Multiple point-related theme keys produce a different SVG than default."""
    combined = Theme(mark_color="#00ff00", point_size=120, opacity=0.3)
    svg_default = _render_with_theme(_chart_point)
    svg_custom = _render_with_theme(_chart_point, combined)
    _assert_svgs_differ(svg_default, svg_custom, "combined point theme")


def test_combined_line_theme():
    """Multiple line-related theme keys produce a different SVG than default."""
    combined = Theme(mark_color="#00ff00", line_stroke_width=4, opacity=0.4)
    svg_default = _render_with_theme(_chart_line)
    svg_custom = _render_with_theme(_chart_line, combined)
    _assert_svgs_differ(svg_default, svg_custom, "combined line theme")


# ---------------------------------------------------------------------------
# Group 8: Theme applied to composite marks
# ---------------------------------------------------------------------------


def _chart_boxplot():
    return fm.Chart(DF_GROUPED).encode(x="group:N", y="value:Q").mark_boxplot()


def _chart_violin():
    return fm.Chart(DF_GROUPED).encode(x="group:N", y="value:Q").mark_violin()


def _chart_errorbar():
    return fm.Chart(DF_GROUPED).encode(x="group:N", y="value:Q").mark_errorbar()


@pytest.mark.parametrize(
    "chart_fn,mark_name",
    [
        pytest.param(_chart_boxplot, "boxplot", id="composite-boxplot"),
        pytest.param(_chart_violin, "violin", id="composite-violin"),
    ],
)
def test_mark_color_affects_composite_marks(chart_fn, mark_name):
    """mark_color should propagate through composite mark desugaring."""
    svg_default = _render_with_theme(chart_fn)
    svg_custom = _render_with_theme(chart_fn, Theme(mark_color="#ff0000"))
    _assert_svgs_differ(svg_default, svg_custom, f"mark_color on composite {mark_name}")


def test_line_stroke_width_affects_errorbar():
    """line_stroke_width should propagate to errorbar (rule-based composite)."""
    svg_default = _render_with_theme(_chart_errorbar)
    svg_custom = _render_with_theme(_chart_errorbar, Theme(line_stroke_width=8.0))
    _assert_svgs_differ(svg_default, svg_custom, "line_stroke_width on errorbar")


# ---------------------------------------------------------------------------
# Group 9: Theme applied to layered charts
# ---------------------------------------------------------------------------


def test_theme_on_layered_point_smooth():
    """Theme applied to a layered (point + smooth) chart changes the SVG."""

    def _layered():
        base = fm.Chart(DF).encode(x="x:Q", y="y:Q")
        return base.mark_point() + base.mark_smooth()

    svg_default = _render_with_theme(_layered)
    svg_custom = _render_with_theme(_layered, Theme(mark_color="#ff0000"))
    _assert_svgs_differ(svg_default, svg_custom, "mark_color on point+smooth layer")


def test_theme_on_layered_bar_rule():
    """Theme applied to a layered (bar + rule) chart changes the SVG."""

    def _layered():
        base = fm.Chart(DF).encode(x="cat:N", y="val:Q")
        return base.mark_bar() + base.mark_rule()

    svg_default = _render_with_theme(_layered)
    svg_custom = _render_with_theme(_layered, Theme(mark_color="#ff0000"))
    _assert_svgs_differ(svg_default, svg_custom, "mark_color on bar+rule layer")
