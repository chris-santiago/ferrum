"""Tests for mark kwarg aliases (K1, K2, F3) and convenience features (K3, F4).

K1  — color="red"   → fill="red"
K2  — alpha=0.5     → opacity=0.5
F3  — linetype="dashed" → stroke_dash="4,2"
K3  — mark_line(point=True) overlays points on the line
F4  — mark_circle() / mark_square() wrappers
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.marks.base import MarkBase


# ---------------------------------------------------------------------------
# K1 — color alias
# ---------------------------------------------------------------------------


def test_color_alias():
    mb = MarkBase("point", color="red")
    assert mb.kwargs == {"fill": "red"}


def test_color_alias_canonical_still_works():
    mb = MarkBase("point", fill="blue")
    assert mb.kwargs == {"fill": "blue"}


def test_color_and_fill_together():
    # If both provided, fill wins (last write in resolution order), or color
    # maps to fill — whichever comes first the canonical key is used.
    # The implementation resolves aliases in iteration order; Python dicts
    # preserve insertion order, so the last canonical assignment wins.
    # Providing both is unusual; just check it doesn't raise.
    mb = MarkBase("point", fill="blue", color="red")
    assert "fill" in mb.kwargs  # both collapse to same key


# ---------------------------------------------------------------------------
# K2 — alpha alias
# ---------------------------------------------------------------------------


def test_alpha_alias():
    mb = MarkBase("point", alpha=0.5)
    assert mb.kwargs == {"opacity": 0.5}


def test_alpha_alias_canonical_still_works():
    mb = MarkBase("point", opacity=0.8)
    assert mb.kwargs == {"opacity": 0.8}


# ---------------------------------------------------------------------------
# F3 — linetype aliases
# ---------------------------------------------------------------------------


def test_linetype_dashed():
    mb = MarkBase("line", linetype="dashed")
    assert mb.kwargs == {"stroke_dash": [4.0, 2.0]}


def test_linetype_dotted():
    mb = MarkBase("line", linetype="dotted")
    assert mb.kwargs == {"stroke_dash": [1.0, 3.0]}


def test_linetype_solid():
    mb = MarkBase("line", linetype="solid")
    assert mb.kwargs == {"stroke_dash": []}


def test_linetype_dashdot():
    mb = MarkBase("line", linetype="dashdot")
    assert mb.kwargs == {"stroke_dash": [4.0, 2.0, 1.0, 2.0]}


def test_linetype_longdash():
    mb = MarkBase("line", linetype="longdash")
    assert mb.kwargs == {"stroke_dash": [8.0, 4.0]}


def test_line_type_underscore_alias():
    """line_type (underscore variant) should also work."""
    mb = MarkBase("line", line_type="dashed")
    assert mb.kwargs == {"stroke_dash": [4.0, 2.0]}


def test_linetype_raw_string_passthrough():
    """A raw dash-array string is parsed to a float list."""
    mb = MarkBase("line", linetype="6,3")
    assert mb.kwargs == {"stroke_dash": [6.0, 3.0]}


def test_stroke_dash_canonical_still_works():
    mb = MarkBase("line", stroke_dash=[4.0, 2.0])
    assert mb.kwargs == {"stroke_dash": [4.0, 2.0]}


# ---------------------------------------------------------------------------
# Alias + canonical combination
# ---------------------------------------------------------------------------


def test_canonical_still_works_combined():
    mb = MarkBase("point", fill="blue", opacity=0.8)
    assert mb.kwargs == {"fill": "blue", "opacity": 0.8}


# ---------------------------------------------------------------------------
# Unknown kwarg still raises
# ---------------------------------------------------------------------------


def test_unknown_kwarg_still_errors():
    with pytest.raises(TypeError, match="banana"):
        MarkBase("point", banana=True)


def test_unknown_kwarg_after_alias_resolution():
    """Aliases that don't resolve to a valid key should raise, not silently pass."""
    with pytest.raises(TypeError):
        MarkBase("point", colouur="red")  # typo, not in alias map


# ---------------------------------------------------------------------------
# K3 — mark_line(point=True) overlays points
# ---------------------------------------------------------------------------


def test_mark_line_point_true():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]})
    chart = fm.Chart(df).mark_line(point=True).encode(x="x", y="y")
    svg = chart.to_svg()
    # Points rendered (circles in SVG)
    assert "<circle" in svg, "Expected <circle elements for overlaid points"
    # Line rendered (path or polyline)
    assert "<path" in svg or "<polyline" in svg or 'd="' in svg, (
        "Expected path/polyline for the line layer"
    )


def test_mark_line_point_false_no_circles():
    """point=False (or absent) must not overlay points."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]})
    svg = fm.Chart(df).mark_line().encode(x="x", y="y").to_svg()
    assert "<circle" not in svg


def test_mark_line_kwargs_still_forwarded_with_point():
    """Other kwargs like stroke_width are forwarded when point=True."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    # Should not raise and should render
    svg = fm.Chart(df).mark_line(point=True, stroke_width=3).encode(x="x", y="y").to_svg()
    assert "<circle" in svg


# ---------------------------------------------------------------------------
# F4 — mark_circle() / mark_square()
# ---------------------------------------------------------------------------


def test_mark_circle():
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    svg = fm.Chart(df).mark_circle().encode(x="x", y="y").to_svg()
    assert "<circle" in svg


def test_mark_square():
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    svg = fm.Chart(df).mark_square().encode(x="x", y="y").to_svg()
    assert "<rect" in svg


def test_mark_circle_passes_kwargs():
    """mark_circle should forward kwargs (e.g. size, opacity) to mark_point."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    # Should not raise
    svg = fm.Chart(df).mark_circle(size=100, opacity=0.5).encode(x="x", y="y").to_svg()
    assert "<circle" in svg


def test_mark_square_passes_kwargs():
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    svg = fm.Chart(df).mark_square(size=100).encode(x="x", y="y").to_svg()
    assert "<rect" in svg


def test_mark_circle_alias_kwargs():
    """mark_circle should also accept alias kwargs like color."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    svg = fm.Chart(df).mark_circle(color="red").encode(x="x", y="y").to_svg()
    assert "<circle" in svg


def test_mark_circle_unknown_kwarg_errors():
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(TypeError, match="banana"):
        fm.Chart(df).mark_circle(banana=True).encode(x="x", y="y").to_svg()
