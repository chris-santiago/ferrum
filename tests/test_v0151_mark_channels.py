"""v0.15.1 audit: channel-loading drift regression tests.

Three build paths silently dropped encoding channels their Cartesian/heatmap
siblings honor.  Each test below targets exactly one fix:

  1. bar.rs build_polar  — now loads StrokeChannels + MetadataColumns
  2. rect.rs build_quantitative_range + build_ordinal_range — now load
     fill_opacity + per-row stroke_width
  3. arc.rs build_nominal_theta + build_annular — now route href/description
     through meta.build_metadata() instead of hardcoding None
"""

from __future__ import annotations
import re

import polars as pl
import pytest
import ferrum as fr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attr(svg: str, element: str, attr: str) -> list[str]:
    """Return every value of `attr` found on elements matching `element` tag."""
    pattern = rf"<{element}[^>]*\s{re.escape(attr)}=\"([^\"]+)\""
    return re.findall(pattern, svg)


def _path_stroke_widths(svg: str) -> list[str]:
    """Return stroke-width attribute values from <path> elements."""
    return _attr(svg, "path", "stroke-width")


def _path_fill_opacities(svg: str) -> list[str]:
    """Return fill-opacity attribute values from <path> elements."""
    return _attr(svg, "path", "fill-opacity")


def _rect_stroke_widths(svg: str) -> list[str]:
    """Return stroke-width attribute values from <rect> elements."""
    return _attr(svg, "rect", "stroke-width")


def _rect_fill_opacities(svg: str) -> list[str]:
    """Return fill-opacity attribute values from <rect> elements."""
    return _attr(svg, "rect", "fill-opacity")


def _link_hrefs(svg: str) -> list[str]:
    """Return href / xlink:href attribute values from <a> elements."""
    hrefs = re.findall(r'<a[^>]*\shref="([^"]+)"', svg)
    hrefs += re.findall(r'<a[^>]*\sxlink:href="([^"]+)"', svg)
    return hrefs


def _data_tooltip_attrs(svg: str) -> list[str]:
    """Return data-tooltip or title attribute values in the SVG."""
    return re.findall(r'data-tooltip="([^"]+)"', svg) + re.findall(r"<title>([^<]+)</title>", svg)


# ---------------------------------------------------------------------------
# 1. Polar bar (build_polar) must honor stroke_width + fill_opacity per row
# ---------------------------------------------------------------------------


def test_polar_bar_honors_stroke_width_encoding():
    """build_polar must apply per-row stroke_width; fix verifies attribute appears."""
    df = pl.DataFrame(
        {
            "dir": ["N", "E", "S", "W"],
            "val": [10.0, 20.0, 15.0, 25.0],
            "sw": [1.0, 3.0, 1.0, 3.0],  # alternating narrow / wide
        }
    )
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("dir", type="ordinal"),
            y=fr.Y("val", type="quantitative"),
            stroke_width="sw",
        )
        .coord(fr.CoordPolar(theta="x"))
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg"), "render must produce valid SVG"

    # At least one path element must carry a stroke-width attribute.
    sw_values = _path_stroke_widths(svg)
    assert sw_values, (
        "build_polar dropped stroke_width encoding: no stroke-width found on path elements"
    )

    # With alternating 1.0 / 3.0 values both widths should appear.
    numeric_widths = {round(float(v), 6) for v in sw_values}
    assert len(numeric_widths) >= 2, (
        f"expected both stroke widths (1.0 and 3.0) to appear, got {numeric_widths}"
    )


def test_polar_bar_honors_fill_opacity_encoding():
    """build_polar must apply per-row fill_opacity; SVG fill-opacity must vary."""
    df = pl.DataFrame(
        {
            "dir": ["N", "E", "S", "W"],
            "val": [10.0, 20.0, 15.0, 25.0],
            "fo": [0.2, 0.9, 0.2, 0.9],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("dir", type="ordinal"),
            y=fr.Y("val", type="quantitative"),
            fill_opacity="fo",
        )
        .coord(fr.CoordPolar(theta="x"))
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    fo_values = _path_fill_opacities(svg)
    assert fo_values, (
        "build_polar dropped fill_opacity encoding: no fill-opacity found on path elements"
    )
    numeric_fos = {round(float(v), 6) for v in fo_values}
    assert len(numeric_fos) >= 2, (
        f"expected at least two distinct fill-opacity values (0.2 and 0.9), got {numeric_fos}"
    )


def test_polar_bar_emits_tooltip_data():
    """build_polar with tooltip encoding must emit tooltip data in the SVG."""
    df = pl.DataFrame(
        {
            "dir": ["N", "E", "S"],
            "val": [10.0, 20.0, 15.0],
            "label": ["North", "East", "South"],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("dir", type="ordinal"),
            y=fr.Y("val", type="quantitative"),
            tooltip="label",
        )
        .coord(fr.CoordPolar(theta="x"))
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")
    # Tooltip presence: the SVG layer or interactive layer must carry the label
    # content in a data-tooltip attribute or <title> element.
    tooltip_data = _data_tooltip_attrs(svg)
    assert any("North" in t or "East" in t or "South" in t for t in tooltip_data), (
        f"build_polar dropped tooltip encoding; tooltip content not found in SVG. "
        f"Found tooltip attrs: {tooltip_data}"
    )


# ---------------------------------------------------------------------------
# 2. Range rect paths must honor fill_opacity + per-row stroke_width
# ---------------------------------------------------------------------------


def test_rect_quantitative_range_honors_stroke_width_encoding():
    """build_quantitative_range must apply per-row stroke_width."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "x2": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0],
            "y2": [1.0, 1.0, 1.0],
            "sw": [1.0, 4.0, 1.0],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_rect()
        .encode(
            x=fr.X("x", type="quantitative"),
            x2=fr.X2("x2"),
            y=fr.Y("y", type="quantitative"),
            y2=fr.Y2("y2"),
            stroke_width="sw",
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    sw_values = _rect_stroke_widths(svg)
    # Filter out axis / background rects (width=0 or very large).
    assert sw_values, (
        "build_quantitative_range dropped stroke_width encoding: no stroke-width on rect elements"
    )
    numeric_widths = {round(float(v), 6) for v in sw_values}
    assert len(numeric_widths) >= 2, (
        f"expected both stroke widths (1.0 and 4.0) to appear, got {numeric_widths}"
    )


def test_rect_quantitative_range_honors_fill_opacity_encoding():
    """build_quantitative_range must apply per-row fill_opacity."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "x2": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0],
            "y2": [1.0, 1.0, 1.0],
            "fo": [0.2, 0.9, 0.5],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_rect()
        .encode(
            x=fr.X("x", type="quantitative"),
            x2=fr.X2("x2"),
            y=fr.Y("y", type="quantitative"),
            y2=fr.Y2("y2"),
            fill_opacity="fo",
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    fo_values = _rect_fill_opacities(svg)
    assert fo_values, (
        "build_quantitative_range dropped fill_opacity encoding: no fill-opacity on rect elements"
    )
    numeric_fos = {round(float(v), 6) for v in fo_values}
    assert len(numeric_fos) >= 2, (
        f"expected at least two distinct fill-opacity values, got {numeric_fos}"
    )


def test_rect_ordinal_range_honors_stroke_width_encoding():
    """build_ordinal_range must apply per-row stroke_width (boxplot box body path)."""
    df = pl.DataFrame(
        {
            "cat": ["A", "B", "C"],
            "q1": [1.0, 2.0, 3.0],
            "q3": [4.0, 5.0, 6.0],
            "sw": [1.0, 4.0, 1.0],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_rect()
        .encode(
            x=fr.X("cat", type="ordinal"),
            y=fr.Y("q1", type="quantitative"),
            y2=fr.Y2("q3"),
            stroke_width="sw",
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    sw_values = _rect_stroke_widths(svg)
    assert sw_values, (
        "build_ordinal_range dropped stroke_width encoding: no stroke-width on rect elements"
    )
    numeric_widths = {round(float(v), 6) for v in sw_values}
    assert len(numeric_widths) >= 2, (
        f"expected both stroke widths (1.0 and 4.0) to appear, got {numeric_widths}"
    )


def test_rect_ordinal_range_honors_fill_opacity_encoding():
    """build_ordinal_range must apply per-row fill_opacity."""
    df = pl.DataFrame(
        {
            "cat": ["A", "B"],
            "q1": [1.0, 2.0],
            "q3": [4.0, 5.0],
            "fo": [0.2, 0.9],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_rect()
        .encode(
            x=fr.X("cat", type="ordinal"),
            y=fr.Y("q1", type="quantitative"),
            y2=fr.Y2("q3"),
            fill_opacity="fo",
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    fo_values = _rect_fill_opacities(svg)
    assert fo_values, (
        "build_ordinal_range dropped fill_opacity encoding: no fill-opacity on rect elements"
    )
    numeric_fos = {round(float(v), 6) for v in fo_values}
    assert len(numeric_fos) >= 2, (
        f"expected at least two distinct fill-opacity values (0.2 and 0.9), got {numeric_fos}"
    )


# ---------------------------------------------------------------------------
# 3. Arc nominal_theta + annular must pass href/description through
# ---------------------------------------------------------------------------


def test_arc_nominal_theta_href_passes_through():
    """build_nominal_theta must route href through meta.build_metadata; links appear."""
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [30.0, 50.0, 20.0],
            "link": [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_arc()
        .encode(
            x=fr.X("category", type="nominal"),
            y=fr.Y("value", type="quantitative"),
            href="link",
        )
        .coord(fr.CoordPolar(theta="x"))
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    hrefs = _link_hrefs(svg)
    assert hrefs, "build_nominal_theta dropped href encoding: no <a href> links found in SVG"
    assert any("example.com" in h for h in hrefs), f"expected href URLs in SVG, got: {hrefs}"


def test_arc_nominal_theta_tooltip_still_works():
    """build_nominal_theta tooltip must still work after refactor to build_metadata."""
    df = pl.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [30.0, 50.0, 20.0],
            "label": ["Alpha", "Beta", "Gamma"],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_arc()
        .encode(
            x=fr.X("category", type="nominal"),
            y=fr.Y("value", type="quantitative"),
            tooltip="label",
        )
        .coord(fr.CoordPolar(theta="x"))
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg")

    tooltip_data = _data_tooltip_attrs(svg)
    assert any("Alpha" in t or "Beta" in t or "Gamma" in t for t in tooltip_data), (
        f"build_nominal_theta tooltip broken after refactor; tooltip_data: {tooltip_data}"
    )
