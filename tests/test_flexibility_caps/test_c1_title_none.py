"""Tests for C1 -- ``title=None`` on a positional channel suppresses the axis title.

Desired behavior:
  - ``X("col", title=None)`` / ``Y("col", title=None)`` -> no axis title rendered.
    The field name must NOT appear as the axis title text.
  - ``X("col")`` (title absent) -> field name IS the axis title (default behavior,
    no regression).
  - ``X("col", title="Custom")`` -> "Custom" IS the axis title.

The fix lives in ``crates/ferrum-core/src/render/prepare.rs``: the axis-title
resolution now treats an explicit empty string as a suppression sentinel (None),
so the rest of the pipeline never reserves a title margin and never emits a
``<text>`` node for a suppressed title.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _axis_title_texts(svg: str) -> list[str]:
    """Return all text strings that appear as axis titles in the SVG.

    Extracts every ``<text>`` element's content and filters to non-numeric,
    non-empty strings (axis/legend titles rather than tick labels).
    """
    raw = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
    titles = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        try:
            float(t)
            continue
        except ValueError:
            pass
        titles.append(t)
    return titles


def _panel_rect(svg: str) -> tuple[float, float, float, float] | None:
    """Return (x, y, width, height) of the second <rect> (the panel/plot area rect).

    The first rect is always the chart background.  The second is the panel.
    Returns None if fewer than two rects are present.
    """
    rects = re.findall(
        r'<rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"',
        svg,
    )
    if len(rects) < 2:
        return None
    x, y, w, h = rects[1]
    return float(x), float(y), float(w), float(h)


def _has_empty_text_node(svg: str) -> bool:
    """Return True if the SVG contains any empty <text ...></text> node."""
    return bool(re.search(r"<text[^>]*>\s*</text>", svg))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "horsepower": [130.0, 165.0, 150.0, 150.0, 140.0],
            "miles_per_gallon": [18.0, 15.0, 18.0, 16.0, 17.0],
        }
    )


# ---------------------------------------------------------------------------
# C1-A: title=None suppresses axis title on X channel
# ---------------------------------------------------------------------------


def test_x_title_none_suppresses_axis_title(simple_df: pl.DataFrame) -> None:
    """X('horsepower', title=None) must suppress the x-axis title.

    The fix forwards an empty-string sentinel from Python to Rust; Rust
    resolves that as None so neither a margin nor a <text> node is emitted.
    """
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title=None), y="miles_per_gallon")
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "horsepower" not in titles, (
        "X('horsepower', title=None) must suppress the axis title; "
        f"found 'horsepower' in axis title texts: {titles}"
    )


def test_x_title_none_emits_no_empty_text_node(simple_df: pl.DataFrame) -> None:
    """X with title=None must not emit a phantom empty <text></text> node."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title=None), y="miles_per_gallon")
        .show_svg()
    )
    assert not _has_empty_text_node(svg), (
        "X('horsepower', title=None) must not emit an empty <text> node in the SVG"
    )


def test_x_title_none_reclaims_bottom_margin(simple_df: pl.DataFrame) -> None:
    """Suppressing the x-axis title must reclaim bottom margin.

    The panel height when title=None must exceed the height when a title is
    present, because the title row is no longer reserved.
    """
    svg_with = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title="HP"), y="miles_per_gallon")
        .show_svg()
    )
    svg_none = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title=None), y="miles_per_gallon")
        .show_svg()
    )
    rect_with = _panel_rect(svg_with)
    rect_none = _panel_rect(svg_none)
    assert rect_with is not None and rect_none is not None, "Could not parse panel rect"
    _, _, _, h_with = rect_with
    _, _, _, h_none = rect_none
    assert h_none > h_with, (
        f"Panel height with title=None ({h_none}) must exceed height with a title ({h_with}); "
        "bottom margin was not reclaimed"
    )


# ---------------------------------------------------------------------------
# C1-B: title=None suppresses axis title on Y channel
# ---------------------------------------------------------------------------


def test_y_title_none_suppresses_axis_title(simple_df: pl.DataFrame) -> None:
    """Y('miles_per_gallon', title=None) must suppress the y-axis title."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", title=None))
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "miles_per_gallon" not in titles, (
        "Y('miles_per_gallon', title=None) must suppress the axis title; "
        f"found 'miles_per_gallon' in axis title texts: {titles}"
    )


def test_y_title_none_emits_no_empty_text_node(simple_df: pl.DataFrame) -> None:
    """Y with title=None must not emit a phantom empty <text></text> node."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", title=None))
        .show_svg()
    )
    assert not _has_empty_text_node(svg), (
        "Y('miles_per_gallon', title=None) must not emit an empty <text> node in the SVG"
    )


def test_y_title_none_reclaims_left_margin(simple_df: pl.DataFrame) -> None:
    """Suppressing the y-axis title must reclaim left margin.

    The panel x-coordinate when title=None must be smaller than when a title
    is present, because the title column is no longer reserved.
    """
    svg_with = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", title="MPG"))
        .show_svg()
    )
    svg_none = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", title=None))
        .show_svg()
    )
    rect_with = _panel_rect(svg_with)
    rect_none = _panel_rect(svg_none)
    assert rect_with is not None and rect_none is not None, "Could not parse panel rect"
    x_with, _, _, _ = rect_with
    x_none, _, _, _ = rect_none
    assert x_none < x_with, (
        f"Panel x with title=None ({x_none}) must be smaller than with a title ({x_with}); "
        "left margin was not reclaimed"
    )


# ---------------------------------------------------------------------------
# C1-C: title=None on both X and Y channels
# ---------------------------------------------------------------------------


def test_both_title_none_suppresses_both_axis_titles(simple_df: pl.DataFrame) -> None:
    """title=None on both X and Y must suppress both axis titles."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title=None), y=Y("miles_per_gallon", title=None))
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "horsepower" not in titles, (
        f"X title=None must suppress x-axis title; axis title texts: {titles}"
    )
    assert "miles_per_gallon" not in titles, (
        f"Y title=None must suppress y-axis title; axis title texts: {titles}"
    )


def test_both_title_none_emits_no_empty_text_nodes(simple_df: pl.DataFrame) -> None:
    """title=None on both channels must not emit any empty <text> nodes."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title=None), y=Y("miles_per_gallon", title=None))
        .show_svg()
    )
    assert not _has_empty_text_node(svg), (
        "Both title=None must not emit any empty <text> nodes in the SVG"
    )


def test_both_title_none_reclaims_both_margins(simple_df: pl.DataFrame) -> None:
    """Suppressing both axis titles must reclaim left and bottom margins simultaneously.

    Panel with both=None must extend further left (smaller x) and taller than
    a chart with both titles present.
    """
    svg_both = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(
            x=X("horsepower", title="HP"),
            y=Y("miles_per_gallon", title="MPG"),
        )
        .show_svg()
    )
    svg_none = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(
            x=X("horsepower", title=None),
            y=Y("miles_per_gallon", title=None),
        )
        .show_svg()
    )
    rect_both = _panel_rect(svg_both)
    rect_none = _panel_rect(svg_none)
    assert rect_both is not None and rect_none is not None, "Could not parse panel rect"
    x_both, _, _, h_both = rect_both
    x_none, _, _, h_none = rect_none
    assert x_none < x_both, (
        f"Panel x with both=None ({x_none}) must be smaller than with titles ({x_both})"
    )
    assert h_none > h_both, (
        f"Panel height with both=None ({h_none}) must exceed height with titles ({h_both})"
    )


# ---------------------------------------------------------------------------
# Guard: title omitted -> field name IS the axis title (no regression)
# ---------------------------------------------------------------------------


def test_x_title_omitted_shows_field_name(simple_df: pl.DataFrame) -> None:
    """X('horsepower') with no title kwarg must render field name as axis title.

    This is the existing default behavior and must remain GREEN both before
    and after the C1 fix.
    """
    svg = (
        fm.Chart(simple_df).mark_point().encode(x=X("horsepower"), y="miles_per_gallon").show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "horsepower" in titles, (
        "X('horsepower') with no title kwarg must show 'horsepower' as axis title; "
        f"axis title texts: {titles}"
    )


def test_y_title_omitted_shows_field_name(simple_df: pl.DataFrame) -> None:
    """Y('miles_per_gallon') with no title kwarg must render field name as axis title."""
    svg = (
        fm.Chart(simple_df).mark_point().encode(x="horsepower", y=Y("miles_per_gallon")).show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "miles_per_gallon" in titles, (
        "Y('miles_per_gallon') with no title kwarg must show 'miles_per_gallon' as axis title; "
        f"axis title texts: {titles}"
    )


# ---------------------------------------------------------------------------
# Guard: title="Custom" -> custom text IS the axis title
# ---------------------------------------------------------------------------


def test_x_custom_title_appears(simple_df: pl.DataFrame) -> None:
    """X('horsepower', title='Engine Power') must render 'Engine Power' as x-axis title."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title="Engine Power"), y="miles_per_gallon")
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "Engine Power" in titles, (
        "X('horsepower', title='Engine Power') must show 'Engine Power' as axis title; "
        f"axis title texts: {titles}"
    )
    assert "horsepower" not in titles, (
        "Field name 'horsepower' must not appear when a custom title is set; "
        f"axis title texts: {titles}"
    )


def test_y_custom_title_appears(simple_df: pl.DataFrame) -> None:
    """Y('miles_per_gallon', title='Fuel Economy') must render 'Fuel Economy' as y-axis title."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", title="Fuel Economy"))
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "Fuel Economy" in titles, (
        f"Y title='Fuel Economy' must appear as axis title; axis title texts: {titles}"
    )
    assert "miles_per_gallon" not in titles, (
        f"Field name must not appear when a custom title is set; axis title texts: {titles}"
    )


# ---------------------------------------------------------------------------
# C1-D: Axis(title="") suppresses axis title (parallel to title=None contract)
#
# An explicit empty Axis title is a suppression signal: it must NOT fall back
# to the field name.  This locks the regression introduced when the Axis(title=)
# path was added — the `.filter(|s| !s.trim().is_empty())` collapsed
# Some("") and None into the same None, letting .or(field) re-introduce "hp".
# ---------------------------------------------------------------------------


def test_x_axis_title_empty_suppresses_axis_title(simple_df: pl.DataFrame) -> None:
    """X('horsepower', axis=fm.Axis(title='')) must suppress the x-axis title.

    An explicit empty Axis title is a suppression signal, same as title=None.
    The field name must NOT appear as the axis title.
    """
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", axis=fm.Axis(title="")), y="miles_per_gallon")
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "horsepower" not in titles, (
        "X('horsepower', axis=fm.Axis(title='')) must suppress the axis title; "
        f"found 'horsepower' in axis title texts: {titles}"
    )


def test_x_axis_title_empty_emits_no_empty_text_node(simple_df: pl.DataFrame) -> None:
    """Axis(title='') must not emit a phantom empty <text></text> node."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", axis=fm.Axis(title="")), y="miles_per_gallon")
        .show_svg()
    )
    assert not _has_empty_text_node(svg), (
        "X('horsepower', axis=fm.Axis(title='')) must not emit an empty <text> node"
    )


def test_x_axis_title_empty_reclaims_bottom_margin(simple_df: pl.DataFrame) -> None:
    """Axis(title='') must reclaim bottom margin (same as title=None)."""
    svg_with = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", title="HP"), y="miles_per_gallon")
        .show_svg()
    )
    svg_empty = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", axis=fm.Axis(title="")), y="miles_per_gallon")
        .show_svg()
    )
    rect_with = _panel_rect(svg_with)
    rect_empty = _panel_rect(svg_empty)
    assert rect_with is not None and rect_empty is not None, "Could not parse panel rect"
    _, _, _, h_with = rect_with
    _, _, _, h_empty = rect_empty
    assert h_empty > h_with, (
        f"Panel height with Axis(title='') ({h_empty}) must exceed height with a title ({h_with}); "
        "bottom margin was not reclaimed"
    )


def test_y_axis_title_empty_suppresses_axis_title(simple_df: pl.DataFrame) -> None:
    """Y('miles_per_gallon', axis=fm.Axis(title='')) must suppress the y-axis title."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", axis=fm.Axis(title="")))
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "miles_per_gallon" not in titles, (
        "Y('miles_per_gallon', axis=fm.Axis(title='')) must suppress the axis title; "
        f"found 'miles_per_gallon' in axis title texts: {titles}"
    )


def test_y_axis_title_empty_reclaims_left_margin(simple_df: pl.DataFrame) -> None:
    """Axis(title='') on Y must reclaim left margin."""
    svg_with = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", title="MPG"))
        .show_svg()
    )
    svg_empty = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", axis=fm.Axis(title="")))
        .show_svg()
    )
    rect_with = _panel_rect(svg_with)
    rect_empty = _panel_rect(svg_empty)
    assert rect_with is not None and rect_empty is not None, "Could not parse panel rect"
    x_with, _, _, _ = rect_with
    x_empty, _, _, _ = rect_empty
    assert x_empty < x_with, (
        f"Panel x with Axis(title='') ({x_empty}) must be smaller than with a title ({x_with}); "
        "left margin was not reclaimed"
    )


# ---------------------------------------------------------------------------
# C1-E: Axis(title="Custom") shows the custom title (not the field name)
# ---------------------------------------------------------------------------


def test_x_axis_title_custom_appears(simple_df: pl.DataFrame) -> None:
    """X('horsepower', axis=fm.Axis(title='Engine Power')) must show 'Engine Power'."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x=X("horsepower", axis=fm.Axis(title="Engine Power")), y="miles_per_gallon")
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "Engine Power" in titles, (
        "X with Axis(title='Engine Power') must show 'Engine Power' as axis title; "
        f"axis title texts: {titles}"
    )
    assert "horsepower" not in titles, (
        "Field name 'horsepower' must not appear when Axis(title=) is set; "
        f"axis title texts: {titles}"
    )


def test_y_axis_title_custom_appears(simple_df: pl.DataFrame) -> None:
    """Y('miles_per_gallon', axis=fm.Axis(title='Fuel Economy')) must show 'Fuel Economy'."""
    svg = (
        fm.Chart(simple_df)
        .mark_point()
        .encode(x="horsepower", y=Y("miles_per_gallon", axis=fm.Axis(title="Fuel Economy")))
        .show_svg()
    )
    titles = _axis_title_texts(svg)

    assert "Fuel Economy" in titles, (
        "Y with Axis(title='Fuel Economy') must show 'Fuel Economy' as axis title; "
        f"axis title texts: {titles}"
    )
    assert "miles_per_gallon" not in titles, (
        f"Field name must not appear when Axis(title=) is set; axis title texts: {titles}"
    )
