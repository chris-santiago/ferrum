"""v0.15.1 audit: x2/y2 channel-combination guard tests.

Two mark geometry edge cases previously silently dropped or mis-rendered a
second positional extent.  Both are now rejected with a clear ValueError
before any geometry is built.

Guard 1 — mark_area with x2:
    x2 is not a documented area channel.  Horizontal-band areas are mark_rect's
    domain; vertical-band areas use y2.  Passing x2 to mark_area now raises
    ValueError naming "x2" and pointing at the alternatives.

Guard 2 — mark_bar with both x2 AND y2:
    A bar with both a horizontal and a vertical extent is definitionally a 2-D
    rectangle; that is mark_rect's role.  Passing both x2 and y2 to mark_bar
    now raises ValueError naming "x2 and y2" and pointing at mark_rect.

Regression paths (must not raise):
    - mark_bar with only y2  (floating / diverging bars, D3)
    - mark_bar with only x2  (horizontal bin bars, histograms)
    - mark_area with only y2  (band area, D3)
"""

from __future__ import annotations

import pytest
import polars as pl
import ferrum as fr


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

_DF_AREA = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "lo": [2.0, 3.0, 4.0, 3.0],
        "hi": [5.0, 6.0, 7.0, 6.0],
    }
)

_DF_BAR = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0],
        "x2": [1.0, 2.0, 3.0],
        "y": [5.0, 10.0, 3.0],
        "y2": [0.0, 2.0, 1.0],
        "cat": ["a", "b", "c"],
    }
)


# ---------------------------------------------------------------------------
# Guard 1: mark_area + x2 must raise
# ---------------------------------------------------------------------------


def test_mark_area_with_x2_raises():
    """mark_area with x2 bound must raise ValueError before rendering."""
    chart = (
        fr.Chart(_DF_AREA)
        .mark_area()
        .encode(
            x=fr.X("x", type="quantitative"),
            y=fr.Y("lo", type="quantitative"),
            x2="hi",
        )
    )
    with pytest.raises(ValueError, match="mark_area") as exc_info:
        chart.show_svg()
    msg = str(exc_info.value)
    assert "x2" in msg, f"error message should name the channel 'x2'; got: {msg!r}"


def test_mark_area_with_x2_error_names_alternative():
    """The mark_area+x2 error message must mention an actionable alternative."""
    chart = (
        fr.Chart(_DF_AREA)
        .mark_area()
        .encode(
            x=fr.X("x", type="quantitative"),
            y=fr.Y("lo", type="quantitative"),
            x2="hi",
        )
    )
    with pytest.raises(ValueError) as exc_info:
        chart.show_svg()
    msg = str(exc_info.value)
    # The hint should guide users toward y2 or mark_rect.
    has_alternative = "y2" in msg or "mark_rect" in msg or "rect" in msg.lower()
    assert has_alternative, f"error should name an alternative (y2= or mark_rect); got: {msg!r}"


# ---------------------------------------------------------------------------
# Guard 2: mark_bar + x2 + y2 must raise
# ---------------------------------------------------------------------------


def test_mark_bar_with_x2_and_y2_raises():
    """mark_bar with both x2 and y2 bound must raise ValueError before rendering."""
    chart = (
        fr.Chart(_DF_BAR)
        .mark_bar()
        .encode(
            x=fr.X("x", type="quantitative"),
            x2="x2",
            y=fr.Y("y", type="quantitative"),
            y2="y2",
        )
    )
    with pytest.raises(ValueError, match="mark_bar") as exc_info:
        chart.show_svg()
    msg = str(exc_info.value)
    assert "x2" in msg, f"error message should name 'x2'; got: {msg!r}"
    assert "y2" in msg, f"error message should name 'y2'; got: {msg!r}"


def test_mark_bar_with_x2_and_y2_error_names_mark_rect():
    """The mark_bar+x2+y2 error message must point at mark_rect."""
    chart = (
        fr.Chart(_DF_BAR)
        .mark_bar()
        .encode(
            x=fr.X("x", type="quantitative"),
            x2="x2",
            y=fr.Y("y", type="quantitative"),
            y2="y2",
        )
    )
    with pytest.raises(ValueError) as exc_info:
        chart.show_svg()
    msg = str(exc_info.value)
    assert "mark_rect" in msg or "rect" in msg.lower(), (
        f"error should name 'mark_rect' as the alternative; got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Regression: supported single-extent paths must NOT raise
# ---------------------------------------------------------------------------


def test_mark_bar_with_only_y2_renders(tmp_path):
    """mark_bar with y2 only (floating/diverging bar) must render without error."""
    df = pl.DataFrame(
        {
            "cat": ["a", "b", "c"],
            "y_lo": [1.0, 2.0, -1.0],
            "y_hi": [4.0, 5.0, 2.0],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", type="ordinal"),
            y=fr.Y("y_lo", type="quantitative"),
            y2="y_hi",
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg"), "mark_bar with y2 only must render valid SVG"
    assert "<rect" in svg, "mark_bar with y2 only must emit rect elements"


def test_mark_bar_with_only_x2_renders():
    """mark_bar with x2 only (histogram / bin bar) must render without error."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "x2": [1.0, 2.0, 3.0],
            "count": [5.0, 10.0, 3.0],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("x", type="quantitative"),
            x2="x2",
            y=fr.Y("count", type="quantitative"),
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg"), "mark_bar with x2 only must render valid SVG"
    assert "<rect" in svg, "mark_bar with x2 only must emit rect elements"


def test_mark_area_with_y2_renders():
    """mark_area with y2 (band area, D3) must render without error."""
    chart = (
        fr.Chart(_DF_AREA)
        .mark_area()
        .encode(
            x=fr.X("x", type="quantitative"),
            y=fr.Y("lo", type="quantitative"),
            y2="hi",
        )
    )
    svg = chart.show_svg()
    assert svg.startswith("<svg"), "mark_area with y2 must render valid SVG"
    assert "<path" in svg, "mark_area with y2 must emit path elements"
