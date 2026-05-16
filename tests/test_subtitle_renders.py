"""Schwabish SB1 — subtitle renders as a second text line in the title band."""

from __future__ import annotations

import polars as pl

from ferrum import Chart, Title


def test_subtitle_renders_as_second_text_element():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    chart = Chart(df, title=Title("Main", subtitle="Sub")).encode(x="x", y="y").mark_point()
    svg = chart.show_svg()
    assert ">Main<" in svg
    assert ">Sub<" in svg
    # Subtitle text node appears after main title in document order.
    assert svg.index(">Main<") < svg.index(">Sub<")


def test_no_subtitle_byte_identical_to_string_title():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    a = Chart(df, title="Main").encode(x="x", y="y").mark_point().show_svg()
    b = Chart(df, title=Title("Main")).encode(x="x", y="y").mark_point().show_svg()
    assert a == b


def test_no_title_does_not_emit_subtitle_band():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    svg = Chart(df).encode(x="x", y="y").mark_point().show_svg()
    # If a subtitle node were emitted with no title, the test would fail.
    assert ">Sub<" not in svg
