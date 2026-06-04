"""Theme key roundtrip — every spec §3.13 key flows Python → Rust without crash.

Tight assertions for keys whose render consumers are already wired in T1
(mark_color, background, padding, point_size). Loose ``startswith('<svg')``
assertions for keys whose consumers land in T2/T3 — those tests get
tightened in later sub-phases.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture(scope="module")
def base_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


def _render(chart: fm.Chart, **theme_kwargs: object) -> str:
    return chart.theme(fm.Theme(**theme_kwargs)).to_svg()


# Tight assertions — consumers already wired pre-T1.


def test_mark_color_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, mark_color="#ff0000")
    s = svg.lower()
    assert "#ff0000" in s or "rgb(255,0,0)" in s or "rgb(255, 0, 0)" in s


def test_background_alias_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, background="#abcdef")
    assert "#abcdef" in svg.lower()


def test_background_color_canonical_reaches_svg(base_chart: fm.Chart) -> None:
    svg = _render(base_chart, background_color="#abcdef")
    assert "#abcdef" in svg.lower()


def test_invalid_hex_raises(base_chart: fm.Chart) -> None:
    chart = base_chart.theme(fm.Theme(mark_color="not-a-hex"))
    with pytest.raises(ValueError):
        chart.to_svg()


# Loose assertions — consumers land in T2/T3; for T1 we just verify the key
# flows through the binding without crashing.


@pytest.mark.parametrize(
    "key,value",
    [
        ("font_family", "Helvetica"),
        ("font_weight", "bold"),
        ("font_color", "#444444"),
        ("font_size", 14.0),
        ("title_font_family", "Inter"),
        ("title_font_size", 16.0),
        ("title_font_weight", "600"),
        ("title_color", "#000000"),
        ("title_anchor", "start"),
        ("title_anchor", "middle"),
        ("title_anchor", "end"),
        ("title_offset", 8.0),
        ("label_font_family", "Helvetica"),
        ("label_color", "#888888"),
        ("axis_line", False),
        ("axis_line", True),
        ("axis_line_color", "#000000"),
        ("axis_line_width", 2.0),
        ("tick_color", "#444444"),
        ("tick_size", 6.0),
        ("tick_width", 1.5),
        ("grid", True),
        ("grid", False),
        ("grid_color", "#dddddd"),
        ("grid_width", 0.5),
        ("grid_dash", [3, 3]),
        ("grid_opacity", 0.5),
        ("point_opacity", 0.7),
        ("opacity", 0.8),
        ("color_scheme", "tableau10"),
        ("color_scheme", "set1"),
        ("color_scheme", "viridis"),
        ("strip_background_color", "#f0f0f0"),
        ("legend_orient", "right"),
        ("legend_orient", "left"),
        ("legend_orient", "top"),
        ("legend_orient", "bottom"),
        ("legend_direction", "horizontal"),
        ("legend_direction", "vertical"),
        ("legend_title_font_size", 12.0),
        ("axis_title_padding", 6.0),
        ("column_padding", 10.0),
        ("row_padding", 10.0),
    ],
)
def test_key_roundtrips_without_crash(base_chart: fm.Chart, key: str, value: object) -> None:
    svg = _render(base_chart, **{key: value})
    assert svg.startswith("<svg")


def test_invalid_title_anchor_raises(base_chart: fm.Chart) -> None:
    chart = base_chart.theme(fm.Theme(title_anchor="bogus"))
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    assert "title_anchor must be one of" in str(excinfo.value)


def test_invalid_legend_orient_raises(base_chart: fm.Chart) -> None:
    chart = base_chart.theme(fm.Theme(legend_orient="diagonal"))
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    assert "legend_orient must be one of" in str(excinfo.value)


def test_invalid_legend_direction_raises(base_chart: fm.Chart) -> None:
    chart = base_chart.theme(fm.Theme(legend_direction="vertical-ish"))
    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()
    assert "legend_direction must be one of" in str(excinfo.value)
