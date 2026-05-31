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
        fm.Chart(df).mark_bar().encode(x="c:N", y="y:Q", color=Color("c:N", scale=scale)).show_svg()
    )
    # Both colors must appear.
    svg_lower = svg.lower()
    assert "e4572e" in svg_lower, "accent color #e4572e must appear in SVG"
    assert "cccccc" in svg_lower, "gray color #cccccc must appear in SVG"
    # Count occurrences: accent (B) appears once, gray (A and C) appear twice.
    # This proves colors follow the declared domain, not data appearance order.
    accent_count = svg_lower.count("e4572e")
    gray_count = svg_lower.count("cccccc")
    assert accent_count < gray_count, (
        f"accent should appear fewer times than gray (accent={accent_count}, gray={gray_count})"
    )


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


# ---------------------------------------------------------------------------
# D3 — per-channel Axis(label_format=...) and tick_count now work
# ---------------------------------------------------------------------------

import re
from datetime import date


@pytest.fixture
def two_cat_numeric_df() -> pl.DataFrame:
    return pl.DataFrame({"cat": ["A", "B"], "val": [10_000.0, 20_000.0]})


@pytest.fixture
def two_cat_large_df() -> pl.DataFrame:
    return pl.DataFrame({"cat": ["A", "B"], "val": [1_000_000.0, 3_000_000.0]})


@pytest.fixture
def two_cat_fraction_df() -> pl.DataFrame:
    return pl.DataFrame({"cat": ["A", "B"], "val": [0.25, 0.75]})


@pytest.fixture
def monthly_date_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2021, 6, 1), "1mo", eager=True),
            "val": list(range(18)),
        }
    )


@pytest.fixture
def long_monthly_date_df() -> pl.DataFrame:
    """30-month frame for tick_count assertions."""
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2022, 6, 1), "1mo", eager=True),
            "val": list(range(30)),
        }
    )


def _tick_texts(svg: str) -> list[str]:
    """Extract inner text from all SVG <text> elements."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def test_d3_numeric_grouping_format(two_cat_numeric_df: pl.DataFrame) -> None:
    """Axis(label_format=',.0f') produces comma-grouped labels like '10,000'."""
    svg = (
        fm.Chart(two_cat_numeric_df)
        .mark_bar()
        .encode(
            x="cat:N",
            y=fm.Y("val:Q", axis=fm.Axis(label_format=",.0f")),
        )
        .show_svg()
    )
    tick_labels = _tick_texts(svg)
    assert "10,000" in tick_labels, f"expected '10,000' in tick labels; got {tick_labels}"


def test_d3_si_prefix_format(two_cat_large_df: pl.DataFrame) -> None:
    """Axis(label_format='~s') trims trailing zeros and applies SI suffixes (k, M)."""
    svg = (
        fm.Chart(two_cat_large_df)
        .mark_bar()
        .encode(
            x="cat:N",
            y=fm.Y("val:Q", axis=fm.Axis(label_format="~s")),
        )
        .show_svg()
    )
    tick_labels = _tick_texts(svg)
    # At least one label must carry the 'M' (mega) SI suffix.
    assert any("M" in label for label in tick_labels), (
        f"expected at least one 'M' SI-suffix label; got {tick_labels}"
    )


def test_d3_percent_format(two_cat_fraction_df: pl.DataFrame) -> None:
    """Axis(label_format='.0%') renders percent labels like '50%' on a 0-1 axis."""
    svg = (
        fm.Chart(two_cat_fraction_df)
        .mark_bar()
        .encode(
            x="cat:N",
            y=fm.Y("val:Q", axis=fm.Axis(label_format=".0%")),
        )
        .show_svg()
    )
    tick_labels = _tick_texts(svg)
    assert "50%" in tick_labels, f"expected '50%' in tick labels; got {tick_labels}"


def test_d3_temporal_format_month_year(monthly_date_df: pl.DataFrame) -> None:
    """Axis(label_format='%b %Y') on a :T x-axis produces 'Jan 2020'-style labels."""
    svg = (
        fm.Chart(monthly_date_df)
        .mark_line()
        .encode(
            x=fm.X("date:T", axis=fm.Axis(label_format="%b %Y")),
            y="val:Q",
        )
        .show_svg()
    )
    tick_labels = _tick_texts(svg)
    # Must have at least one label matching the '<MonthAbbrev> <Year>' pattern.
    month_year_labels = [t for t in tick_labels if re.match(r"[A-Z][a-z]{2} 20\d{2}$", t)]
    assert month_year_labels, f"expected at least one 'MMM YYYY' label; got {tick_labels}"
    # Specifically confirm 'Jan 2020' (the first tick in the domain) is present.
    assert "Jan 2020" in month_year_labels, (
        f"expected 'Jan 2020' among month-year labels; got {month_year_labels}"
    )


def test_d3_tick_count_limits_temporal_ticks(long_monthly_date_df: pl.DataFrame) -> None:
    """Axis(tick_count=4) on a 30-month :T axis produces far fewer labels than default."""
    svg_default = (
        fm.Chart(long_monthly_date_df).mark_line().encode(x="date:T", y="val:Q").show_svg()
    )
    svg_limited = (
        fm.Chart(long_monthly_date_df)
        .mark_line()
        .encode(
            x=fm.X("date:T", axis=fm.Axis(tick_count=4)),
            y="val:Q",
        )
        .show_svg()
    )

    # Count date-like tick labels: text elements that contain a 4-digit year.
    def _date_tick_count(svg: str) -> int:
        return sum(1 for t in _tick_texts(svg) if re.search(r"20\d{2}", t))

    default_count = _date_tick_count(svg_default)
    limited_count = _date_tick_count(svg_limited)
    assert limited_count < default_count, (
        f"tick_count=4 should produce fewer date labels than default; "
        f"limited={limited_count}, default={default_count}"
    )
    # Sanity: a coarse label (year only) should appear in the limited axis.
    assert re.search(r"20\d{2}", svg_limited), (
        "limited axis must still render at least one year-level tick label"
    )


def test_d3_default_quantitative_axis_still_renders(
    two_cat_numeric_df: pl.DataFrame,
) -> None:
    """A quantitative axis with no label_format renders plain numeric labels (default path)."""
    svg = fm.Chart(two_cat_numeric_df).mark_bar().encode(x="cat:N", y="val:Q").show_svg()
    tick_labels = _tick_texts(svg)
    # Expect plain integer-style labels (no commas, no percent, no SI suffix).
    numeric_labels = [t for t in tick_labels if re.match(r"^\d+$", t)]
    assert numeric_labels, (
        f"default quantitative axis should produce plain numeric labels; got {tick_labels}"
    )


# ---------------------------------------------------------------------------
# D2 — order-independent layer merge
# ---------------------------------------------------------------------------


def _polyline_strokes(svg: str) -> set[str]:
    """Extract distinct hex stroke colors from <polyline> elements."""
    return set(re.findall(r'<polyline[^>]*stroke="(#[0-9a-fA-F]{6})"', svg))


@pytest.fixture
def layer_order_dfs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two disjoint DataFrames: background (gray group) and highlight (colored group)."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for country in ["China", "Nigeria"]:
        for year in range(2000, 2010):
            rows.append({"year": year, "country": country, "value": float(rng.uniform(10, 50))})
    hl_df = pl.DataFrame(rows)

    rows2 = []
    for country in ["USA", "Japan"]:
        for year in range(2000, 2010):
            rows2.append({"year": year, "country": country, "value": float(rng.uniform(10, 50))})
    bg_df = pl.DataFrame(rows2)

    return bg_df, hl_df


def test_d2_color_scale_order_independent(
    layer_order_dfs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """(base + highlight) and (highlight + base) must resolve the same color set.

    The highlight layer uses scheme='set1' with 2 categories (China, Nigeria).
    Without the fix, (base + highlight) collapses both categories to a single
    theme color because the chart-level color encoding is None (inherited from
    base, which has no color encoding).

    After the fix both orderings must contain all accent colors from the
    highlight layer.
    """
    bg_df, hl_df = layer_order_dfs

    base = (
        fm.Chart(bg_df)
        .mark_line(stroke="#cccccc", stroke_width=1)
        .encode(x="year:Q", y="value:Q", detail="country:N")
    )
    highlight = (
        fm.Chart(hl_df)
        .mark_line(stroke_width=2.5)
        .encode(
            x="year:Q",
            y="value:Q",
            color=fm.Color("country:N", scheme="set1"),
        )
    )

    # Establish the ground-truth color set from highlight rendered alone.
    standalone_colors = _polyline_strokes(highlight.show_svg())
    assert len(standalone_colors) == 2, (
        f"highlight standalone should produce exactly 2 distinct colors; got {standalone_colors}"
    )

    svg_bh = (base + highlight).show_svg()
    svg_hb = (highlight + base).show_svg()

    colors_bh = _polyline_strokes(svg_bh)
    colors_hb = _polyline_strokes(svg_hb)

    # Both orderings must contain every accent color from the highlight layer.
    missing_in_bh = standalone_colors - colors_bh
    assert not missing_in_bh, (
        f"base + highlight is missing highlight colors {missing_in_bh}; "
        f"got {colors_bh} (standalone: {standalone_colors})"
    )
    missing_in_hb = standalone_colors - colors_hb
    assert not missing_in_hb, (
        f"highlight + base is missing highlight colors {missing_in_hb}; "
        f"got {colors_hb} (standalone: {standalone_colors})"
    )


def test_d2_color_scale_not_collapsed_to_single_theme_color(
    layer_order_dfs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """When base has no color encoding, (base + highlight) must not collapse
    2 highlight categories to a single theme color.

    Regression guard: before the fix, both highlight countries rendered
    identically with the theme default blue (#2563eb) instead of two distinct
    set1 colors.
    """
    bg_df, hl_df = layer_order_dfs

    base = (
        fm.Chart(bg_df)
        .mark_line(stroke="#cccccc", stroke_width=1)
        .encode(x="year:Q", y="value:Q", detail="country:N")
    )
    highlight = (
        fm.Chart(hl_df)
        .mark_line(stroke_width=2.5)
        .encode(
            x="year:Q",
            y="value:Q",
            color=fm.Color("country:N", scheme="set1"),
        )
    )

    svg = (base + highlight).show_svg()
    colors = _polyline_strokes(svg)

    # There must be at least 2 non-gray colors — one per highlight category.
    non_gray = {c for c in colors if c.lower() != "#cccccc"}
    assert len(non_gray) >= 2, (
        f"base + highlight must have at least 2 non-gray colors (one per highlight category); "
        f"got colors={colors!r}"
    )


def test_d2_annotation_does_not_supply_axis_titles() -> None:
    """Annotation layers used as LHS must not rename axes to internal field names.

    annotate_rect encodes x='_x1', y='_y1' etc. internally. When used as the
    base (LHS) of a layer, the data layer's axis titles (x, y) must survive —
    not be replaced by _x1/_y1 from the annotation layer.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    data = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    rect = fm.annotate_rect(x1=1.5, x2=2.5, y1=3.5, y2=6.5)

    for label, chart in [
        ("data + rect", data + rect),
        ("rect + data", rect + data),
    ]:
        svg = chart.show_svg()
        assert "_x1" not in svg, f"{label}: internal annotation field '_x1' must not appear in SVG"
        assert "_y1" not in svg, f"{label}: internal annotation field '_y1' must not appear in SVG"

    tick_labels_dr = _tick_texts((data + rect).show_svg())
    tick_labels_rd = _tick_texts((rect + data).show_svg())
    # Both orderings must produce the same axis-label set.
    assert set(tick_labels_dr) == set(tick_labels_rd), (
        f"axis labels must be order-independent: "
        f"data+rect={tick_labels_dr!r} vs rect+data={tick_labels_rd!r}"
    )


def test_d2_annotation_hline_does_not_pollute_axes() -> None:
    """annotate_hline as base (LHS) must not supply axis titles from its internal _y field."""
    df = pl.DataFrame({"t": [1.0, 2.0, 3.0], "r": [0.5, 1.5, 2.5]})
    data = fm.Chart(df).mark_line().encode(x="t:Q", y="r:Q")
    hline = fm.annotate_hline(y=1.0, stroke="red")

    for label, chart in [
        ("data + hline", data + hline),
        ("hline + data", hline + data),
    ]:
        svg = chart.show_svg()
        assert "_y" not in svg, f"{label}: internal annotation field '_y' must not appear in SVG"
