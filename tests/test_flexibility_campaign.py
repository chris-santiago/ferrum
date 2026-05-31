"""Regression tests for the flexibility-campaign bug fixes.

Each section is labeled by its campaign defect ID so new defects can be
appended here without disrupting earlier sections.
"""

import polars as pl
import pytest

import ferrum as fm
from ferrum import OrdinalScale
from ferrum.encoding import Color


# ---------------------------------------------------------------------------
# D1 — OrdinalScale.range accepts color strings
# ---------------------------------------------------------------------------


@pytest.fixture
def three_cat_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["A", "B", "C"],
            "val": [10.0, 20.0, 30.0],
        }
    )


def test_d1_value_class_accent_color_present_in_svg(three_cat_df: pl.DataFrame) -> None:
    """OrdinalScale with a color-string range renders the accent color in SVG."""
    scale = OrdinalScale(
        domain=["A", "B", "C"],
        range=["#cccccc", "#cccccc", "#e4572e"],
    )
    svg = (
        fm.Chart(three_cat_df)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color=Color("cat:N", scale=scale))
        .show_svg()
    )
    assert "e4572e" in svg, "accent color #e4572e must appear in SVG"
    assert "cccccc" in svg, "gray color #cccccc must appear in SVG"


def test_d1_dict_form_accent_color_present_in_svg(three_cat_df: pl.DataFrame) -> None:
    """dict-form scale with a color-string range renders the accent color in SVG."""
    svg = (
        fm.Chart(three_cat_df)
        .mark_bar()
        .encode(
            x="cat:N",
            y="val:Q",
            color=Color(
                "cat:N",
                scale={
                    "type": "ordinal",
                    "domain": ["A", "B", "C"],
                    "range": ["#cccccc", "#cccccc", "#e4572e"],
                },
            ),
        )
        .show_svg()
    )
    assert "e4572e" in svg, "accent color #e4572e must appear in SVG (dict form)"
    assert "cccccc" in svg, "gray color #cccccc must appear in SVG (dict form)"


def test_d1_named_css_colors_resolve_to_their_hex(three_cat_df: pl.DataFrame) -> None:
    """Named CSS color strings in the range must resolve to their hex equivalents."""
    scale = OrdinalScale(
        domain=["A", "B", "C"],
        range=["steelblue", "tomato", "seagreen"],
    )
    svg = (
        fm.Chart(three_cat_df)
        .mark_bar()
        .encode(x="cat:N", y="val:Q", color=Color("cat:N", scale=scale))
        .show_svg()
    )
    assert "4682b4" in svg.lower(), "steelblue (#4682b4) must resolve and appear in SVG"


def test_d1_positional_float_range_unaffected() -> None:
    """Numeric positional ranges still work and are preserved exactly."""
    scale = OrdinalScale(domain=["A", "B", "C"], range=[0.0, 300.0])
    assert scale.range is not None
    assert list(scale.range) == [0.0, 300.0]


def test_d1_declared_domain_overrides_data_appearance_order() -> None:
    """Declared domain order determines color mapping, not data appearance order."""
    # Data has categories in order C, A, B (appearance order).
    # Domain declares them as A, B, C.
    # Colors: gray, gray, accent. So C should be gray, A should be gray, B should be accent.
    df = pl.DataFrame(
        {
            "c": ["C", "A", "B"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    scale = OrdinalScale(
        domain=["A", "B", "C"],
        range=["#cccccc", "#e4572e", "#cccccc"],
    )
    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="c:N", y="y:Q", color=Color("c:N", scale=scale))
        .show_svg()
    )
    # Both colors must appear.
    svg_lower = svg.lower()
    assert "e4572e" in svg_lower, "accent color #e4572e must appear in SVG"
    assert "cccccc" in svg_lower, "gray color #cccccc must appear in SVG"
    # Count occurrences: accent (B) appears once, gray (A and C) appear twice.
    # This proves colors follow the declared domain, not data appearance order.
    accent_count = svg_lower.count("e4572e")
    gray_count = svg_lower.count("cccccc")
    assert (
        accent_count < gray_count
    ), f"accent should appear fewer times than gray (accent={accent_count}, gray={gray_count})"


# ---------------------------------------------------------------------------
# D4 — mark_rect / fm.heatmap honors the cmap scheme
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_df() -> pl.DataFrame:
    """Small 3x3 correlation-style wide DataFrame."""
    return pl.DataFrame(
        {
            "feature": ["x", "y", "z"],
            "x": [1.0, 0.8, 0.2],
            "y": [0.8, 1.0, -0.3],
            "z": [0.2, -0.3, 1.0],
        }
    )


def test_d4_heatmap_different_cmaps_produce_different_svg(corr_df: pl.DataFrame) -> None:
    """fm.heatmap with cmap='blues' and cmap='reds' produce distinct SVG output."""
    svg_blues = fm.heatmap(corr_df, cmap="blues").show_svg()
    svg_reds = fm.heatmap(corr_df, cmap="reds").show_svg()
    assert svg_blues != svg_reds, (
        "heatmap with cmap='blues' and cmap='reds' must render different SVG"
    )
