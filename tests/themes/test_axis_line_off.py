"""axis_line=False suppresses axis stroke; tick_width / label_color flow to SVG."""

import polars as pl

import ferrum as fm


def test_axis_line_false_suppresses_axis_stroke() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg_on = chart.theme(fm.Theme(axis_line=True)).show_svg()
    svg_off = chart.theme(fm.Theme(axis_line=False)).show_svg()
    # SVG with axis_line off has fewer <line> elements (axis strokes removed).
    assert svg_off.count("<line") < svg_on.count("<line")


def test_label_color_overrides_tick_label_fill() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg = chart.theme(fm.Theme(font_color="#000000", label_color="#888888")).show_svg()
    # Tick labels use label_color (#888888).
    assert "#888888" in svg.lower()
    # Axis title fill comes from title_color which falls back to font_color (#000000).
    assert "#000000" in svg.lower()


def test_tick_width_distinct_from_axis_line_width() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    # Set tick_width different from axis_line_width and verify both appear.
    svg = chart.theme(fm.Theme(axis_line_width=3.0, tick_width=1.0)).show_svg()
    # Both stroke widths should appear somewhere in the SVG.
    assert 'stroke-width="3"' in svg or 'stroke-width="3.0"' in svg
    assert 'stroke-width="1"' in svg or 'stroke-width="1.0"' in svg
