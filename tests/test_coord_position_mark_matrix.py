"""Matrix test: coordinate system x position adjustment x mark type.

Fills the gap between test_coord_flip_matrix.py (marks without position
adjustments) and test_position_matrix.py (position adjustments under default
coords). This module exercises every meaningful combination of position
adjustment (Stack, Dodge, Jitter) paired with non-default coordinate systems
(CoordFlip, CoordFixed, CoordPolar).
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.position import Stack, Dodge, Jitter
from ferrum.coord import CoordFlip, CoordFixed, CoordPolar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grouped_df() -> pl.DataFrame:
    """DataFrame with categorical x, quantitative y, and a grouping column."""
    return pl.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [3.0, 5.0, 2.0, 4.0, 6.0, 1.0],
            "grp": ["x", "y", "x", "y", "x", "y"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _render(df, mark_method, encoding, position=None, coord=None) -> str:
    """Build a chart with the given mark/encoding/position/coord and return SVG."""
    chart = fm.Chart(df)
    mark_fn = getattr(chart, mark_method)
    if position is not None:
        chart = mark_fn(position=position)
    else:
        chart = mark_fn()
    chart = chart.encode(**encoding)
    if coord is not None:
        chart = chart.coord(coord)
    return chart.show_svg()


# ---------------------------------------------------------------------------
# Group 1: Stack + CoordFlip
# ---------------------------------------------------------------------------

_STACK_FLIP_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="stack-bar-flip",
    ),
    pytest.param(
        "mark_area",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="stack-area-flip",
    ),
    pytest.param(
        "mark_histogram",
        {"x": "num:Q", "color": "grp:N"},
        id="stack-histogram-flip",
    ),
]


@pytest.mark.parametrize("mark_method,encoding", _STACK_FLIP_CASES)
def test_stack_coord_flip_renders(grouped_df, mark_method, encoding):
    """Stack + CoordFlip renders valid SVG with mark elements."""
    svg = _render(grouped_df, mark_method, encoding, position=Stack(), coord=CoordFlip())
    assert "<svg" in svg, f"{mark_method} + Stack + CoordFlip produced invalid SVG"
    # At least one graphical element should be present
    assert any(tag in svg for tag in ["<rect", "<path", "<circle", "<line", "<polyline"]), (
        f"{mark_method} + Stack + CoordFlip: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 2: Dodge + CoordFlip
# ---------------------------------------------------------------------------

_DODGE_FLIP_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="dodge-bar-flip",
    ),
    pytest.param(
        "mark_point",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="dodge-point-flip",
    ),
]


@pytest.mark.parametrize("mark_method,encoding", _DODGE_FLIP_CASES)
def test_dodge_coord_flip_renders(grouped_df, mark_method, encoding):
    """Dodge + CoordFlip renders valid SVG with mark elements."""
    svg = _render(grouped_df, mark_method, encoding, position=Dodge(), coord=CoordFlip())
    assert "<svg" in svg, f"{mark_method} + Dodge + CoordFlip produced invalid SVG"
    assert any(tag in svg for tag in ["<rect", "<path", "<circle", "<line"]), (
        f"{mark_method} + Dodge + CoordFlip: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 3: Jitter + CoordFlip
# ---------------------------------------------------------------------------

_JITTER_FLIP_CASES = [
    pytest.param(
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        Jitter(seed=42),
        id="jitter-point-flip",
    ),
    pytest.param(
        "mark_tick",
        {"x": "num:Q", "y": "val:Q"},
        Jitter(axis="x", seed=42),
        id="jitter-tick-flip",
    ),
]


@pytest.mark.parametrize("mark_method,encoding,jitter", _JITTER_FLIP_CASES)
def test_jitter_coord_flip_renders(grouped_df, mark_method, encoding, jitter):
    """Jitter + CoordFlip renders valid SVG with mark elements."""
    svg = _render(grouped_df, mark_method, encoding, position=jitter, coord=CoordFlip())
    assert "<svg" in svg, f"{mark_method} + Jitter + CoordFlip produced invalid SVG"
    assert any(tag in svg for tag in ["<circle", "<line"]), (
        f"{mark_method} + Jitter + CoordFlip: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 4: Stack + CoordFixed
# ---------------------------------------------------------------------------

_STACK_FIXED_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="stack-bar-fixed",
    ),
    pytest.param(
        "mark_area",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="stack-area-fixed",
    ),
]


@pytest.mark.parametrize("mark_method,encoding", _STACK_FIXED_CASES)
def test_stack_coord_fixed_renders(grouped_df, mark_method, encoding):
    """Stack + CoordFixed renders valid SVG with mark elements."""
    svg = _render(grouped_df, mark_method, encoding, position=Stack(), coord=CoordFixed(ratio=1.0))
    assert "<svg" in svg, f"{mark_method} + Stack + CoordFixed produced invalid SVG"
    assert any(tag in svg for tag in ["<rect", "<path", "<circle", "<line", "<polyline"]), (
        f"{mark_method} + Stack + CoordFixed: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 5: Dodge + CoordFixed
# ---------------------------------------------------------------------------

_DODGE_FIXED_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="dodge-bar-fixed",
    ),
    pytest.param(
        "mark_point",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="dodge-point-fixed",
    ),
]


@pytest.mark.parametrize("mark_method,encoding", _DODGE_FIXED_CASES)
def test_dodge_coord_fixed_renders(grouped_df, mark_method, encoding):
    """Dodge + CoordFixed renders valid SVG with mark elements."""
    svg = _render(grouped_df, mark_method, encoding, position=Dodge(), coord=CoordFixed(ratio=1.0))
    assert "<svg" in svg, f"{mark_method} + Dodge + CoordFixed produced invalid SVG"
    assert any(tag in svg for tag in ["<rect", "<path", "<circle", "<line"]), (
        f"{mark_method} + Dodge + CoordFixed: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 6: Jitter + CoordFixed
# ---------------------------------------------------------------------------


def test_jitter_coord_fixed_renders(grouped_df):
    """Jitter + CoordFixed renders valid SVG with mark elements."""
    svg = _render(
        grouped_df,
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        position=Jitter(seed=42),
        coord=CoordFixed(ratio=1.0),
    )
    assert "<svg" in svg, "mark_point + Jitter + CoordFixed produced invalid SVG"
    assert "<circle" in svg, "mark_point + Jitter + CoordFixed: no circles in SVG"


# ---------------------------------------------------------------------------
# Group 7: Stack + CoordPolar
# ---------------------------------------------------------------------------

_STACK_POLAR_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="stack-bar-polar",
    ),
    pytest.param(
        "mark_area",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="stack-area-polar",
    ),
]


@pytest.mark.parametrize("mark_method,encoding", _STACK_POLAR_CASES)
def test_stack_coord_polar_renders(grouped_df, mark_method, encoding):
    """Stack + CoordPolar renders valid SVG (polar stacked bars/areas)."""
    svg = _render(grouped_df, mark_method, encoding, position=Stack(), coord=CoordPolar(theta="x"))
    assert "<svg" in svg, f"{mark_method} + Stack + CoordPolar produced invalid SVG"
    # Polar coords may use <path> arcs rather than <rect>
    assert any(tag in svg for tag in ["<rect", "<path", "<circle", "<line", "<polyline"]), (
        f"{mark_method} + Stack + CoordPolar: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 8: Dodge + CoordPolar
# ---------------------------------------------------------------------------

_DODGE_POLAR_CASES = [
    pytest.param(
        "mark_bar",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="dodge-bar-polar",
    ),
    pytest.param(
        "mark_point",
        {"x": "cat:N", "y": "val:Q", "color": "grp:N"},
        id="dodge-point-polar",
    ),
]


@pytest.mark.parametrize("mark_method,encoding", _DODGE_POLAR_CASES)
def test_dodge_coord_polar_renders(grouped_df, mark_method, encoding):
    """Dodge + CoordPolar renders valid SVG."""
    svg = _render(grouped_df, mark_method, encoding, position=Dodge(), coord=CoordPolar(theta="x"))
    assert "<svg" in svg, f"{mark_method} + Dodge + CoordPolar produced invalid SVG"
    assert any(tag in svg for tag in ["<rect", "<path", "<circle", "<line", "<polyline"]), (
        f"{mark_method} + Dodge + CoordPolar: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 9: Jitter + CoordPolar
# ---------------------------------------------------------------------------


def test_jitter_coord_polar_renders(grouped_df):
    """Jitter + CoordPolar renders valid SVG."""
    svg = _render(
        grouped_df,
        "mark_point",
        {"x": "num:Q", "y": "val:Q"},
        position=Jitter(seed=42),
        coord=CoordPolar(theta="x"),
    )
    assert "<svg" in svg, "mark_point + Jitter + CoordPolar produced invalid SVG"
    assert any(tag in svg for tag in ["<circle", "<path"]), (
        "mark_point + Jitter + CoordPolar: no mark elements in SVG"
    )


# ---------------------------------------------------------------------------
# Group 10: Coord changes SVG even with position adjustment
# ---------------------------------------------------------------------------


def test_stack_bar_coord_flip_changes_svg(grouped_df):
    """mark_bar + Stack: default coord vs CoordFlip produces different SVG."""
    enc = {"x": "cat:N", "y": "val:Q", "color": "grp:N"}
    svg_default = _render(grouped_df, "mark_bar", enc, position=Stack())
    svg_flipped = _render(grouped_df, "mark_bar", enc, position=Stack(), coord=CoordFlip())
    assert svg_default != svg_flipped, (
        "mark_bar + Stack: CoordFlip did not change SVG — coord may be ignored"
    )


def test_dodge_bar_coord_flip_changes_svg(grouped_df):
    """mark_bar + Dodge: default coord vs CoordFlip produces different SVG."""
    enc = {"x": "cat:N", "y": "val:Q", "color": "grp:N"}
    svg_default = _render(grouped_df, "mark_bar", enc, position=Dodge())
    svg_flipped = _render(grouped_df, "mark_bar", enc, position=Dodge(), coord=CoordFlip())
    assert svg_default != svg_flipped, (
        "mark_bar + Dodge: CoordFlip did not change SVG — coord may be ignored"
    )


def test_jitter_point_coord_flip_changes_svg(grouped_df):
    """mark_point + Jitter: default coord vs CoordFlip produces different SVG."""
    enc = {"x": "num:Q", "y": "val:Q"}
    svg_default = _render(grouped_df, "mark_point", enc, position=Jitter(seed=42))
    svg_flipped = _render(
        grouped_df, "mark_point", enc, position=Jitter(seed=42), coord=CoordFlip()
    )
    assert svg_default != svg_flipped, (
        "mark_point + Jitter: CoordFlip did not change SVG — coord may be ignored"
    )


# ---------------------------------------------------------------------------
# Group 11: Position changes SVG even under non-default coord
# ---------------------------------------------------------------------------


def test_stack_changes_svg_under_coord_flip(grouped_df):
    """mark_bar under CoordFlip: Stack vs no-position produces different SVG."""
    enc = {"x": "cat:N", "y": "val:Q", "color": "grp:N"}
    svg_no_pos = _render(grouped_df, "mark_bar", enc, coord=CoordFlip())
    svg_stacked = _render(grouped_df, "mark_bar", enc, position=Stack(), coord=CoordFlip())
    assert svg_no_pos != svg_stacked, (
        "mark_bar under CoordFlip: Stack() did not change SVG — position may be ignored"
    )


def test_dodge_changes_svg_under_coord_flip(grouped_df):
    """mark_bar under CoordFlip: Dodge vs no-position produces different SVG."""
    enc = {"x": "cat:N", "y": "val:Q", "color": "grp:N"}
    svg_no_pos = _render(grouped_df, "mark_bar", enc, coord=CoordFlip())
    svg_dodged = _render(grouped_df, "mark_bar", enc, position=Dodge(), coord=CoordFlip())
    assert svg_no_pos != svg_dodged, (
        "mark_bar under CoordFlip: Dodge() did not change SVG — position may be ignored"
    )


def test_jitter_changes_svg_under_coord_fixed(grouped_df):
    """mark_point under CoordFixed: Jitter vs no-position produces different SVG."""
    enc = {"x": "num:Q", "y": "val:Q"}
    svg_no_pos = _render(grouped_df, "mark_point", enc, coord=CoordFixed(ratio=1.0))
    svg_jittered = _render(
        grouped_df, "mark_point", enc, position=Jitter(seed=42), coord=CoordFixed(ratio=1.0)
    )
    assert svg_no_pos != svg_jittered, (
        "mark_point under CoordFixed: Jitter() did not change SVG — position may be ignored"
    )
