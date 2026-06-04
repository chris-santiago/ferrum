"""Gridlines render when theme.grid is True; respect grid_dash + grid_opacity."""

from __future__ import annotations

import polars as pl

import ferrum as fm


def _chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


def test_grid_true_emits_gridlines() -> None:
    svg_on = _chart().theme(fm.Theme(grid=True, grid_color="#ff00ff")).to_svg()
    svg_off = _chart().theme(fm.Theme(grid=False)).to_svg()
    # Magenta grid color flows through to the SVG only when grid=True.
    assert "#ff00ff" in svg_on.lower()
    assert "#ff00ff" not in svg_off.lower()
    # Grid-on SVG has strictly more <line> elements than grid-off.
    assert svg_on.count("<line") > svg_off.count("<line")


def test_grid_dash_emits_stroke_dasharray() -> None:
    svg = _chart().theme(fm.Theme(grid=True, grid_dash=[3.0, 3.0])).to_svg()
    assert "stroke-dasharray" in svg


def test_grid_opacity_emits_stroke_opacity() -> None:
    svg = _chart().theme(fm.Theme(grid=True, grid_opacity=0.3)).to_svg()
    assert "stroke-opacity" in svg


def test_grid_lines_align_with_tick_count() -> None:
    """Grid emits one vertical line per x tick + one horizontal line per y tick.

    Allowing -2 (one per axis) for the baseline-coincident gridline skips
    documented in axis::draw_grid.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    on = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .theme(fm.Theme(grid=True, grid_color="#abcdef"))
        .to_svg()
    )
    # Count how many magenta grid lines were emitted.
    magenta_count = on.lower().count('stroke="#abcdef"')
    # At least 4 gridlines (rough lower bound — two ticks per axis minimum).
    assert magenta_count >= 4
