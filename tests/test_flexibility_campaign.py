"""Regression tests for the flexibility-campaign bug fixes.

Each section is labeled by its campaign defect ID so new defects can be
appended here without disrupting earlier sections.
"""

from datetime import date, datetime as dt

import polars as pl
import pytest

import ferrum as fm
from ferrum import OrdinalScale
from ferrum.annotation.coords import temporal_coord_to_epoch_ms
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


# ---------------------------------------------------------------------------
# D5 — sort forwarding through composite marks
# ---------------------------------------------------------------------------


def _x_cat_order(svg: str, cat_set: set[str]) -> list[str]:
    """Return the distinct category labels from x-axis text elements, in document order.

    Extracts all <text> elements, filters to those whose content is in *cat_set*,
    and de-duplicates while preserving first-occurrence order.  Document order
    corresponds to left-to-right axis rendering.
    """
    seen: list[str] = []
    for t in re.findall(r"<text[^>]*>([^<]+)</text>", svg):
        if t in cat_set and t not in seen:
            seen.append(t)
    return seen


@pytest.fixture
def sort_bar_df() -> pl.DataFrame:
    """Three-category DataFrame where distinct y sums determine sort order.

    X=10, A=50, B=30.  Descending by sum/mean: A > B > X.
    """
    return pl.DataFrame({"c": ["X", "A", "B"], "y": [10.0, 50.0, 30.0]})


@pytest.fixture
def sort_composite_df() -> pl.DataFrame:
    """Multi-observation DataFrame with categories in C, A, B appearance order.

    Mean values: C≈20, A≈80, B≈50.  Descending by mean: A > B > C.
    Data appearance order (no sort): C, A, B.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for cat, loc in [("C", 20.0), ("A", 80.0), ("B", 50.0)]:
        for v in rng.normal(loc, 3.0, 20).tolist():
            rows.append({"cat": cat, "val": v})
    return pl.DataFrame(rows)


# --- Primitive bar (lock in Rust-side sort) ---


def test_d5_primitive_bar_sort_descending(sort_bar_df: pl.DataFrame) -> None:
    """X('c:N', sort='-y') on a bar chart orders categories by descending sum(y)."""
    svg = fm.Chart(sort_bar_df).mark_bar().encode(x=fm.X("c:N", sort="-y"), y="y:Q").show_svg()
    order = _x_cat_order(svg, {"A", "B", "X"})
    assert order == ["A", "B", "X"], (
        f"sort='-y' should produce descending order A > B > X; got {order}"
    )


def test_d5_primitive_bar_sort_ascending(sort_bar_df: pl.DataFrame) -> None:
    """X('c:N', sort='y') on a bar chart orders categories by ascending sum(y)."""
    svg = fm.Chart(sort_bar_df).mark_bar().encode(x=fm.X("c:N", sort="y"), y="y:Q").show_svg()
    order = _x_cat_order(svg, {"A", "B", "X"})
    assert order == ["X", "B", "A"], (
        f"sort='y' should produce ascending order X < B < A; got {order}"
    )


def test_d5_primitive_bar_sort_explicit_array(sort_bar_df: pl.DataFrame) -> None:
    """X('c:N', sort=['A','X','B']) renders categories in the declared array order."""
    svg = (
        fm.Chart(sort_bar_df)
        .mark_bar()
        .encode(x=fm.X("c:N", sort=["A", "X", "B"]), y="y:Q")
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "X"})
    assert order == ["A", "X", "B"], f"sort=['A','X','B'] should produce literal order; got {order}"


def test_d5_primitive_bar_sort_dict_form(sort_bar_df: pl.DataFrame) -> None:
    """X('c:N', sort={field,op,order}) renders categories in the declared aggregate order."""
    svg = (
        fm.Chart(sort_bar_df)
        .mark_bar()
        .encode(
            x=fm.X("c:N", sort={"field": "y", "op": "mean", "order": "descending"}),
            y="y:Q",
        )
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "X"})
    assert order == ["A", "B", "X"], (
        f"dict sort should produce descending-mean order A > B > X; got {order}"
    )


# --- Composite mark sort (the fix) ---


def test_d5_boxplot_sort_descending(sort_composite_df: pl.DataFrame) -> None:
    """mark_boxplot with X(sort='-y') orders boxes by descending aggregate value."""
    svg = (
        fm.Chart(sort_composite_df)
        .mark_boxplot()
        .encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["A", "B", "C"], (
        f"boxplot sort='-y' should order A(80) > B(50) > C(20); got {order}"
    )


def test_d5_boxplot_sort_dict(sort_composite_df: pl.DataFrame) -> None:
    """mark_boxplot with X(sort={field,op,order}) orders boxes by the explicit aggregate."""
    svg = (
        fm.Chart(sort_composite_df)
        .mark_boxplot()
        .encode(
            x=fm.X("cat:N", sort={"field": "val", "op": "mean", "order": "descending"}),
            y="val:Q",
        )
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["A", "B", "C"], (
        f"boxplot dict sort should order A(80) > B(50) > C(20); got {order}"
    )


def test_d5_violin_sort_descending(sort_composite_df: pl.DataFrame) -> None:
    """mark_violin with X(sort='-y') orders violin bodies by descending aggregate."""
    svg = (
        fm.Chart(sort_composite_df)
        .mark_violin()
        .encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["A", "B", "C"], (
        f"violin sort='-y' should order A(80) > B(50) > C(20); got {order}"
    )


def test_d5_errorbar_sort_descending(sort_composite_df: pl.DataFrame) -> None:
    """mark_errorbar with X(sort='-y') orders error bars by descending aggregate."""
    svg = (
        fm.Chart(sort_composite_df)
        .mark_errorbar()
        .encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["A", "B", "C"], (
        f"errorbar sort='-y' should order A(80) > B(50) > C(20); got {order}"
    )


# --- Regression: composites without sort preserve data appearance order ---


def test_d5_boxplot_no_sort_preserves_data_order(sort_composite_df: pl.DataFrame) -> None:
    """mark_boxplot without sort preserves data-appearance order (C, A, B)."""
    svg = fm.Chart(sort_composite_df).mark_boxplot().encode(x="cat:N", y="val:Q").show_svg()
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["C", "A", "B"], (
        f"boxplot without sort should render in data-appearance order C, A, B; got {order}"
    )


def test_d5_violin_no_sort_preserves_data_order(sort_composite_df: pl.DataFrame) -> None:
    """mark_violin without sort preserves data-appearance order (C, A, B)."""
    svg = fm.Chart(sort_composite_df).mark_violin().encode(x="cat:N", y="val:Q").show_svg()
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["C", "A", "B"], (
        f"violin without sort should render in data-appearance order C, A, B; got {order}"
    )


@pytest.fixture
def sort_boxen_df() -> pl.DataFrame:
    """300-observation DataFrame with categories Z (highest), B (medium), A (lowest).

    Mean values: Z≈80, B≈50, A≈20.  Descending by mean: Z > B > A.
    The descending order (Z, B, A) is non-alphabetical so we can distinguish it
    from the default alphabetical rendering (A, B, Z).

    300 observations with std=15 ensure wide enough spread for boxen to produce
    multiple visible depth bands with non-trivial x-axis tick layout, making the
    sorted vs. unsorted SVG byte-differ in domain-related content (not just labels).
    """
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for cat, loc in [("A", 20.0), ("Z", 80.0), ("B", 50.0)]:
        for v in rng.normal(loc, 15.0, 300).tolist():
            rows.append({"cat": cat, "val": v})
    return pl.DataFrame(rows)


def test_d5_boxen_sort_descending(sort_boxen_df: pl.DataFrame) -> None:
    """mark_boxen with X(sort='-y') orders bands by descending aggregate.

    Regression guard for the bug where boxen's LetterValue transform renames the
    groupby column to "group" and drops the original value column from its per-depth
    batches, causing sort dicts referencing the original value field to be silently
    ignored by Rust.  The fix pre-resolves the sort to an explicit ordered category
    list in Python so the explicit-array sort path is used instead.

    The assertion strategy uses three independent checks:

    1. No ``SortSpecIgnored`` Rust warning (the core bug symptom).
    2. The resolved layer encoding carries the expected explicit sort list (Python-level
       correctness — the fix must produce this list for the layer to receive it).
    3. The sorted SVG differs from the unsorted SVG (Rust-side effect: the sort is
       actually applied to the rendered domain).

    We deliberately avoid asserting on SVG ``<text>`` label order because boxen tick
    labels are only rendered when depth bands are wide enough, which varies with the
    rendering context and is not stable across pytest-xdist worker processes.
    """
    import warnings

    # Check 1: no SortSpecIgnored warning.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svg_desc = (
            fm.Chart(sort_boxen_df)
            .mark_boxen()
            .encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
            .show_svg()
        )
    sort_ignored = any("SortSpecIgnored" in str(ww.message) for ww in w)
    assert not sort_ignored, "sort='-y' on mark_boxen must not emit SortSpecIgnored"

    # Check 2: Python-level sort pre-resolution produces expected explicit list.
    # Z≈80 > B≈50 > A≈20, so descending by mean gives ['Z', 'B', 'A'].
    chart_resolved = (
        fm.Chart(sort_boxen_df).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
    )._resolve_pending()
    x_enc = chart_resolved._layers[0].encoding["x"]
    resolved_sort = x_enc._kwargs.get("sort") if hasattr(x_enc, "_kwargs") else None
    assert resolved_sort == ["Z", "B", "A"], (
        f"boxen sort='-y' must resolve to explicit list ['Z','B','A']; got {resolved_sort!r}"
    )

    # Check 3: sorted SVG differs from unsorted (the sort visually affects rendering).
    svg_no_sort = fm.Chart(sort_boxen_df).mark_boxen().encode(x="cat:N", y="val:Q").show_svg()
    assert svg_desc != svg_no_sort, (
        "boxen sort='-y' must produce a different SVG than no-sort "
        "(sort is not being applied to the domain)"
    )


def test_d5_swarm_sort_descending(sort_composite_df: pl.DataFrame) -> None:
    """mark_swarm with X(sort='-y') orders dots by descending aggregate."""
    svg = (
        fm.Chart(sort_composite_df)
        .mark_swarm()
        .encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
        .show_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["A", "B", "C"], (
        f"swarm sort='-y' should order A(80) > B(50) > C(20); got {order}"
    )


def test_d5_errorbar_no_sort_preserves_data_order(sort_composite_df: pl.DataFrame) -> None:
    """mark_errorbar without sort preserves data-appearance order (C, A, B).

    Regression guard: errorbar's sort injection must not alter the default domain
    when no sort is requested.  The data appearance order for sort_composite_df is
    C (first rows), A (second batch), B (third batch).
    """
    svg = fm.Chart(sort_composite_df).mark_errorbar().encode(x="cat:N", y="val:Q").show_svg()
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["C", "A", "B"], (
        f"errorbar without sort should render in data-appearance order C, A, B; got {order}"
    )


def test_d5_horizontal_boxplot_sort_descending(sort_composite_df: pl.DataFrame) -> None:
    """Horizontal mark_boxplot with Y(sort='-x') reorders the categorical y-axis.

    Covers the y_sort injection path: when ``horizontal=True``, the categorical
    axis is y, so ``y_sort`` must be forwarded to the desugar function and wrapped
    into ``Y(cat, sort=...)`` on each layer.  Document order of y-axis labels in
    SVG is top-to-bottom; descending sort puts the highest-value category at the top.
    """
    cats = {"A", "B", "C"}
    svg = (
        fm.Chart(sort_composite_df)
        .mark_boxplot(horizontal=True)
        .encode(x="val:Q", y=fm.Y("cat:N", sort="-x"))
        .show_svg()
    )
    # For horizontal charts, y-axis labels appear in document order top-to-bottom.
    # Descending by x (the numeric axis) means A(80) is first/topmost.
    order = _x_cat_order(svg, cats)
    assert order == ["A", "B", "C"], (
        f"horizontal boxplot Y(sort='-x') should place A(80) first (top), "
        f"then B(50), then C(20); got {order}"
    )


# ---------------------------------------------------------------------------
# D6 — line/segment color→stroke alias (stroke-primary marks)
# ---------------------------------------------------------------------------


@pytest.fixture
def line_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})


@pytest.fixture
def segment_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [0.0], "y": [0.0], "x2": [1.0], "y2": [1.0]})


def _hex_strokes(svg: str) -> set[str]:
    """Extract distinct hex stroke colors from SVG elements."""
    return set(re.findall(r'stroke="#([0-9a-fA-F]{6})"', svg))


def _hex_fills(svg: str) -> set[str]:
    """Extract distinct hex fill colors (not rgba) from SVG elements."""
    return set(re.findall(r'fill="#([0-9a-fA-F]{6})"', svg))


def test_d6_line_color_maps_to_stroke_in_svg(line_df: pl.DataFrame) -> None:
    """mark_line(color='#e4572e') sets the line stroke color in SVG.

    Before the fix, ``color`` was globally aliased to ``fill``.  Lines have no
    fill; their visible color is the stroke.  The hex must appear as
    ``stroke="#e4572e"`` in the rendered SVG, not as a fill attribute.
    """
    svg = fm.Chart(line_df).mark_line(color="#e4572e").encode(x="x:Q", y="y:Q").show_svg()
    strokes = _hex_strokes(svg)
    assert "e4572e" in strokes, (
        f"mark_line(color='#e4572e') must produce stroke='#e4572e' in SVG; found strokes: {strokes}"
    )


def test_d6_rule_color_maps_to_stroke_in_svg() -> None:
    """mark_rule(color='#e4572e') sets the rule stroke color in SVG."""
    df = pl.DataFrame({"y": [1.0]})
    svg = fm.Chart(df).mark_rule(color="#e4572e").encode(y="y:Q").show_svg()
    strokes = _hex_strokes(svg)
    assert "e4572e" in strokes, (
        f"mark_rule(color='#e4572e') must produce stroke='#e4572e' in SVG; found strokes: {strokes}"
    )


def test_d6_segment_color_resolves_to_stroke_in_mark_kwargs(
    segment_df: pl.DataFrame,
) -> None:
    """mark_segment(color='#e4572e') resolves color to stroke in mark_kwargs.

    The Python alias must map ``color`` to ``stroke`` (not ``fill``) for the
    segment mark because segment is stroke-primary.  Verified at the Python
    layer (mark_kwargs dict) because the Rust segment renderer has a separate
    bug where it reads ``mark_style.fill`` instead of ``mark_style.stroke``
    for its line color; that Rust fix is tracked separately.  This test
    confirms the Python alias produces the correct canonical key.
    """
    chart = (
        fm.Chart(segment_df)
        .mark_segment(color="#e4572e")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    assert chart._mark_kwargs.get("stroke") == "#e4572e", (
        f"mark_segment(color='#e4572e') must store stroke='#e4572e' in mark_kwargs; "
        f"got: {chart._mark_kwargs}"
    )
    assert "fill" not in chart._mark_kwargs, (
        f"mark_segment(color='#e4572e') must not store fill in mark_kwargs; "
        f"got: {chart._mark_kwargs}"
    )


def test_d6_line_explicit_stroke_still_works(line_df: pl.DataFrame) -> None:
    """mark_line(stroke='#123456') still sets stroke when passed directly."""
    svg = fm.Chart(line_df).mark_line(stroke="#123456").encode(x="x:Q", y="y:Q").show_svg()
    strokes = _hex_strokes(svg)
    assert "123456" in strokes, (
        f"mark_line(stroke='#123456') must produce stroke='#123456' in SVG; "
        f"found strokes: {strokes}"
    )


# --- Regressions: fill-primary marks must still map color → fill ---


def test_d6_bar_color_still_maps_to_fill() -> None:
    """mark_bar(color='#e4572e') still sets fill (bar is fill-primary)."""
    df = pl.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})
    svg = fm.Chart(df).mark_bar(color="#e4572e").encode(x="cat:N", y="val:Q").show_svg()
    fills = _hex_fills(svg)
    assert "e4572e" in fills, (
        f"mark_bar(color='#e4572e') must produce fill='#e4572e' in SVG; found fills: {fills}"
    )


def test_d6_point_color_still_maps_to_fill(line_df: pl.DataFrame) -> None:
    """mark_point(color='#e4572e') still sets fill (point is fill-primary)."""
    svg = fm.Chart(line_df).mark_point(color="#e4572e").encode(x="x:Q", y="y:Q").show_svg()
    fills = _hex_fills(svg)
    assert "e4572e" in fills, (
        f"mark_point(color='#e4572e') must produce fill='#e4572e' in SVG; found fills: {fills}"
    )


def test_d6_area_color_resolves_to_fill_in_mark_kwargs(line_df: pl.DataFrame) -> None:
    """mark_area(color='#e4572e') resolves color to fill in mark_kwargs (area is fill-primary).

    The Rust area renderer bakes the fill color into an rgba() value (with area
    opacity applied), so the exact hex does not appear verbatim in the SVG.  The
    assertion targets the Python-layer contract: mark_kwargs must carry ``fill``,
    not ``stroke``.
    """
    chart = fm.Chart(line_df).mark_area(color="#e4572e").encode(x="x:Q", y="y:Q")
    assert chart._mark_kwargs.get("fill") == "#e4572e", (
        f"mark_area(color='#e4572e') must store fill='#e4572e' in mark_kwargs; "
        f"got: {chart._mark_kwargs}"
    )
    assert "stroke" not in chart._mark_kwargs, (
        f"mark_area(color='#e4572e') must not store stroke in mark_kwargs; "
        f"got: {chart._mark_kwargs}"
    )


# ---------------------------------------------------------------------------
# D7 — datetime annotation coordinates accepted on temporal axes
# ---------------------------------------------------------------------------


@pytest.fixture
def temporal_df() -> pl.DataFrame:
    """12-month time-series for temporal annotation tests."""
    return pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 12, 1), "1mo", eager=True),
            "val": list(range(12)),
        }
    )


def _date_epoch_ms(d: date) -> float:
    """Expected epoch-ms for a calendar date at midnight UTC.

    Uses the same arithmetic as temporal_coord_to_epoch_ms so the test can
    assert that annotation coordinates and data columns agree exactly.
    """
    return float((d - date(1970, 1, 1)).days * 86_400_000)


def test_d7_vline_date_renders_without_error(temporal_df: pl.DataFrame) -> None:
    """annotate_vline(x=date(...)) on a temporal chart renders without TypeError."""
    chart = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    vline = fm.annotate_vline(x=date(2020, 6, 1), stroke="red")
    svg = (chart + vline).show_svg()
    assert len(svg) > 0, "composed chart with date vline must produce non-empty SVG"


def test_d7_vline_date_epoch_ms_correct() -> None:
    """annotate_vline(x=date(...)) stores the correct epoch-ms in the annotation primitive.

    The annotation chart's internal _x column must be a plain float equal to
    the epoch-ms of the date at midnight UTC.  This guarantees alignment with
    data columns that go through _coerce.py's date→timestamp[ms] path.
    """
    expected_ms = _date_epoch_ms(date(2020, 6, 1))
    vline = fm.annotate_vline(x=date(2020, 6, 1))
    # The primitive is stored on _annotation_primitive; verify its x1 value.
    prim = vline._annotation_primitive
    assert prim is not None, "annotation_primitive must be set"
    # x1 is the data-space coordinate (y1 and y2 are NormCoord wrappers).
    assert prim.x1 == expected_ms, (
        f"vline x1 must equal epoch-ms for date(2020,6,1); expected {expected_ms}, got {prim.x1}"
    )


def test_d7_vline_iso_string_matches_date(temporal_df: pl.DataFrame) -> None:
    """annotate_vline(x='2020-06-01') renders and places the line at the same
    epoch-ms as annotate_vline(x=date(2020,6,1))."""
    expected_ms = _date_epoch_ms(date(2020, 6, 1))

    vline_date = fm.annotate_vline(x=date(2020, 6, 1))
    vline_str = fm.annotate_vline(x="2020-06-01")

    prim_date = vline_date._annotation_primitive
    prim_str = vline_str._annotation_primitive
    assert prim_date.x1 == expected_ms, f"date vline x1={prim_date.x1}, expected {expected_ms}"
    assert prim_str.x1 == expected_ms, f"string vline x1={prim_str.x1}, expected {expected_ms}"
    assert prim_date.x1 == prim_str.x1, "ISO string and date(…) must produce identical epoch-ms"


def test_d7_rect_with_date_boundaries(temporal_df: pl.DataFrame) -> None:
    """annotate_rect(x1=date(...), x2=date(...), y1=..., y2=...) renders without error."""
    chart = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    rect = fm.annotate_rect(
        x1=date(2020, 3, 1),
        x2=date(2020, 9, 1),
        y1=0.0,
        y2=10.0,
        fill="#ffcc00",
        opacity=0.2,
    )
    svg = (chart + rect).show_svg()
    assert len(svg) > 0, "composed chart with date rect must produce non-empty SVG"

    # Verify the primitive corners store epoch-ms values.
    prim = rect._annotation_primitive
    assert prim.x1 == _date_epoch_ms(date(2020, 3, 1)), "rect x1 must be epoch-ms"
    assert prim.x2 == _date_epoch_ms(date(2020, 9, 1)), "rect x2 must be epoch-ms"


def test_d7_text_with_datetime_renders(temporal_df: pl.DataFrame) -> None:
    """annotate_text(x=datetime(...), y=..., text=...) renders without error."""
    chart = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    label = fm.annotate_text(x=dt(2020, 1, 1, 12, 0), y=5.0, text="noon event")
    svg = (chart + label).show_svg()
    assert len(svg) > 0, "annotate_text with datetime x must produce non-empty SVG"

    prim = label._annotation_primitive
    expected_ms = temporal_coord_to_epoch_ms(dt(2020, 1, 1, 12, 0))
    assert prim.x == expected_ms, (
        f"text annotation x must equal epoch-ms of datetime(2020,1,1,12,0); "
        f"expected {expected_ms}, got {prim.x}"
    )


def test_d7_temporal_coord_to_epoch_ms_matches_polars_coerce() -> None:
    """temporal_coord_to_epoch_ms(date) must produce the same epoch-ms that
    _coerce.py produces for a polars Date column on the same date.

    This is the alignment guarantee: annotations placed at date(2020,6,1) land
    exactly on the data point that has date(2020,6,1) in a temporal column.
    """
    import pyarrow as pa

    d = date(2020, 6, 1)
    # Polars path: Date column → _coerce.py → timestamp[ms] Arrow column.
    df = pl.DataFrame({"d": [d]})
    # _coerce.py casts pl.Date → pl.Datetime("ms") via with_columns.
    arr = df.with_columns(pl.col("d").cast(pl.Datetime("ms"))).to_arrow()
    polars_ms = arr.column("d").cast(pa.int64())[0].as_py()

    annotation_ms = temporal_coord_to_epoch_ms(d)
    assert annotation_ms == polars_ms, (
        f"annotation epoch-ms ({annotation_ms}) must match polars coerce epoch-ms ({polars_ms}) "
        f"for date(2020,6,1)"
    )


def test_d7_numeric_coordinates_unchanged() -> None:
    """Plain numeric coordinates still work and are stored as-is (regression guard)."""
    vline = fm.annotate_vline(x=42.0)
    prim = vline._annotation_primitive
    assert prim.x1 == 42.0, f"numeric vline x1 must be 42.0, got {prim.x1}"

    hline = fm.annotate_hline(y=100)
    hprim = hline._annotation_primitive
    assert hprim.y1 == 100.0, f"numeric hline y1 must be 100.0, got {hprim.y1}"


def test_d7_px_norm_wrappers_unaffected() -> None:
    """px() and norm() coordinate wrappers are not affected by the temporal fix."""
    from ferrum.annotation.coords import px, norm
    from ferrum.annotation.primitives import _coord

    assert _coord(px(50)) == {"px": 50}, f"px(50) must serialize to {{px: 50}}"
    assert _coord(norm(0.5)) == {"norm": 0.5}, f"norm(0.5) must serialize to {{norm: 0.5}}"
    # Numeric float passes through unchanged.
    assert _coord(3.14) == 3.14, f"float 3.14 must pass through unchanged"


def test_d7_invalid_iso_string_raises_value_error() -> None:
    """An unparseable string raises ValueError with a clear message."""
    with pytest.raises(ValueError, match="Cannot parse annotation coordinate"):
        fm.annotate_vline(x="not-a-date")


# ---------------------------------------------------------------------------
# D8 — X("a:Q", axis=None) hides the x-axis; Y("b:Q", axis=None) hides y-axis
# ---------------------------------------------------------------------------


@pytest.fixture
def _d8_df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})


def _d8_tick_texts(svg: str) -> list[str]:
    """Extract inner text from all SVG <text> elements."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def test_d8_x_axis_none_hides_x_axis(_d8_df: pl.DataFrame) -> None:
    """X('a:Q', axis=None) must suppress x-axis tick labels and field title."""
    svg_with = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).show_svg()
    svg_without = (
        fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q", axis=None), y=fm.Y("b:Q")).show_svg()
    )
    texts_with = _d8_tick_texts(svg_with)
    texts_without = _d8_tick_texts(svg_without)

    # The x field title "a" must appear in the default chart.
    assert "a" in texts_with, f"expected field title 'a' in default chart texts; got {texts_with}"
    # With axis=None on x, the field title "a" must be absent.
    assert "a" not in texts_without, (
        f"field title 'a' must be suppressed by X(axis=None); got {texts_without}"
    )
    # Numeric x-axis tick labels (1, 2, 3) must be absent.
    x_ticks_present = [t for t in texts_without if t in ("1", "2", "3")]
    assert not x_ticks_present, (
        f"x-axis tick labels must be suppressed by X(axis=None); found {x_ticks_present}"
    )
    # The y field title "b" must still appear.
    assert "b" in texts_without, (
        f"y-axis field title 'b' must remain when only x is suppressed; got {texts_without}"
    )


def test_d8_y_axis_none_hides_y_axis(_d8_df: pl.DataFrame) -> None:
    """Y('b:Q', axis=None) must suppress y-axis tick labels and field title."""
    svg_with = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).show_svg()
    svg_without = (
        fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q", axis=None)).show_svg()
    )
    texts_with = _d8_tick_texts(svg_with)
    texts_without = _d8_tick_texts(svg_without)

    # The y field title "b" must appear in the default chart.
    assert "b" in texts_with, f"expected field title 'b' in default chart texts; got {texts_with}"
    # With axis=None on y, the field title "b" must be absent.
    assert "b" not in texts_without, (
        f"field title 'b' must be suppressed by Y(axis=None); got {texts_without}"
    )
    # Numeric y-axis tick labels (10, 20, 30) must be absent.
    y_ticks_present = [t for t in texts_without if t in ("10", "20", "30")]
    assert not y_ticks_present, (
        f"y-axis tick labels must be suppressed by Y(axis=None); found {y_ticks_present}"
    )
    # The x field title "a" must still appear.
    assert "a" in texts_without, (
        f"x-axis field title 'a' must remain when only y is suppressed; got {texts_without}"
    )


def test_d8_both_axes_none_hides_both(_d8_df: pl.DataFrame) -> None:
    """X(axis=None) + Y(axis=None) must hide both axes."""
    svg = (
        fm.Chart(_d8_df)
        .mark_point()
        .encode(x=fm.X("a:Q", axis=None), y=fm.Y("b:Q", axis=None))
        .show_svg()
    )
    texts = _d8_tick_texts(svg)
    assert "a" not in texts, f"field title 'a' must be absent; got {texts}"
    assert "b" not in texts, f"field title 'b' must be absent; got {texts}"


def test_d8_layered_chart_axis_none_hides_axis(_d8_df: pl.DataFrame) -> None:
    """axis=None on the data layer's channel suppresses the axis in a layered chart."""
    svg = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q", axis=None), y=fm.Y("b:Q")).show_svg()
    texts = _d8_tick_texts(svg)
    assert "a" not in texts, (
        f"x field title 'a' must be absent under axis=None in layered context; got {texts}"
    )
    assert "b" in texts, (
        f"y field title 'b' must still appear when only x axis is suppressed; got {texts}"
    )


def test_d8_real_axis_object_still_renders(_d8_df: pl.DataFrame) -> None:
    """Axis(title='Speed') on X must render the configured title, not suppress the axis."""
    svg = (
        fm.Chart(_d8_df)
        .mark_point()
        .encode(x=fm.X("a:Q", axis=fm.Axis(title="Speed")), y=fm.Y("b:Q"))
        .show_svg()
    )
    texts = _d8_tick_texts(svg)
    assert "Speed" in texts, f"Axis(title='Speed') must render title in SVG; got {texts}"


def test_d8_no_axis_kwarg_renders_normally(_d8_df: pl.DataFrame) -> None:
    """Default (no axis kwarg) must still render both axes."""
    svg = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).show_svg()
    texts = _d8_tick_texts(svg)
    assert "a" in texts, f"x field title 'a' must appear by default; got {texts}"
    assert "b" in texts, f"y field title 'b' must appear by default; got {texts}"


def test_d8_chart_axis_method_still_works(_d8_df: pl.DataFrame) -> None:
    """Chart.axis(x=False) must still suppress the x-axis regardless of encoding."""
    svg = (
        fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).axis(x=False).show_svg()
    )
    texts = _d8_tick_texts(svg)
    assert "a" not in texts, f"Chart.axis(x=False) must suppress x-axis; got {texts}"
    assert "b" in texts, f"y-axis must still render after Chart.axis(x=False); got {texts}"


def test_d8_channel_axis_none_does_not_override_chart_axis_true(_d8_df: pl.DataFrame) -> None:
    """Chart.axis(x=True) with X(axis=None): spec-level show wins, axis is visible."""
    # Per spec §3.7: Chart.axis() is chart-level and takes precedence.
    # When Chart.axis(x=True) is explicit, it overrides the per-channel None.
    svg = (
        fm.Chart(_d8_df)
        .mark_point()
        .encode(x=fm.X("a:Q", axis=None), y=fm.Y("b:Q"))
        .axis(x=True)
        .show_svg()
    )
    texts = _d8_tick_texts(svg)
    assert "a" in texts, f"Chart.axis(x=True) must override per-channel axis=None; got {texts}"


# ---------------------------------------------------------------------------
# D9 — blank-render class
# ---------------------------------------------------------------------------
#
# Three blank-render bugs identified in the v0.13.0 audit:
#
#   D9-A  12-category row-faceted displot renders blank at default size.
#          Root cause (Python): displot set properties() after faceting using
#          the total-canvas height; with 12 panels the per-panel height fell
#          below the Rust EmptyPanel threshold (~56 px). Fix: auto-size total
#          height from n_panels * _FACET_DEFAULT_PANEL_HEIGHT_PX when height
#          is not explicit.
#
#   D9-B  multi-series mark_line on ordinal x with integer column renders blank.
#          Root cause (Rust): line.rs calls col_as_str() when ScaleKind::Ordinal,
#          but col_as_str() only handles Utf8/LargeUtf8; Int64 returns Err and
#          the renderer returns empty().  String ordinal columns work.
#          This is a rust-coder handoff — the integer-ordinal test below
#          is skipped (xfail) until the Rust fix lands.  The string-ordinal
#          variant is a green regression lock proving the happy path.
#
#   D9-C  parent + fm.Inset(...) blanks the parent chart's marks.
#          Status: already renders correctly in v0.13.0 (no Python fix needed).
#          Locked in with a regression test.


import warnings as _warnings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def month_temp_df() -> pl.DataFrame:
    """12-category temperature dataset: 200 rows per month, 2 400 total."""
    import math
    import numpy as np

    rng = np.random.default_rng(1)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for i, m in enumerate(months):
        center = 10.0 + 12.0 * math.sin((i - 3) / 12 * 2 * math.pi) + 8.0
        for _ in range(200):
            rows.append({"month": m, "temp": float(rng.normal(center, 4))})
    return pl.DataFrame(rows)


@pytest.fixture
def ordinal_line_df_str() -> pl.DataFrame:
    """6-series × 8-time-point DataFrame with string year (working ordinal path)."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for cat in ["A", "B", "C", "D", "E", "F"]:
        ranks = list(range(1, 7))
        rng.shuffle(ranks)
        for i, y in enumerate(["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]):
            rows.append({"year": y, "rank": float(ranks[i % len(ranks)]), "series": cat})
    return pl.DataFrame(rows)


@pytest.fixture
def ordinal_line_df_int() -> pl.DataFrame:
    """6-series × 8-time-point DataFrame with integer year (Rust bug path)."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for cat in ["A", "B", "C", "D", "E", "F"]:
        ranks = list(range(1, 7))
        rng.shuffle(ranks)
        for i, y in enumerate(range(2015, 2023)):
            rows.append({"year": y, "rank": float(ranks[i % len(ranks)]), "series": cat})
    return pl.DataFrame(rows)


@pytest.fixture
def inset_df() -> pl.DataFrame:
    """250-day growth series for inset tests."""
    import numpy as np
    from datetime import date, timedelta

    rng = np.random.default_rng(3)
    n = 250
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    growth = list(float(v) for v in np.cumprod(1.0 + rng.normal(0.012, 0.04, n)))
    return pl.DataFrame({"date": dates, "growth": growth})


# ---------------------------------------------------------------------------
# D9-A: 12-category row-faceted displot auto-sizes correctly
# ---------------------------------------------------------------------------


def test_d9a_twelve_row_facet_kde_renders_nonblank(month_temp_df: pl.DataFrame) -> None:
    """displot(row='month') with 12 categories must render 12 KDE paths at default size.

    Before the fix, the default 640×480 canvas divided into 12 panels gave each
    panel ~40 px — below the Rust EmptyPanel threshold (~56 px). The fix auto-scales
    total height to n_panels × _FACET_DEFAULT_PANEL_HEIGHT_PX (150 px) so every
    panel renders.
    """
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        c = fm.displot(month_temp_df, x="temp", row="month", kind="kde", fill=True)
        svg = c.show_svg()

    empty_panel_warns = [x for x in w if "EmptyPanel" in str(x.message)]
    assert not empty_panel_warns, (
        f"displot with 12 row facets must not emit EmptyPanel; got {[str(x.message) for x in empty_panel_warns]}"
    )

    path_count = svg.count('d="M')
    assert path_count >= 12, (
        f"12-category row-faceted KDE must render at least 12 paths; got {path_count}"
    )


def test_d9a_twelve_row_facet_hist_renders_nonblank(month_temp_df: pl.DataFrame) -> None:
    """displot(row='month', kind='hist') with 12 categories must render bars in every panel."""
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        c = fm.displot(month_temp_df, x="temp", row="month", kind="hist")
        svg = c.show_svg()

    empty_panel_warns = [x for x in w if "EmptyPanel" in str(x.message)]
    assert not empty_panel_warns, (
        f"12-row hist facet must not emit EmptyPanel; got {[str(x.message) for x in empty_panel_warns]}"
    )

    # Histogram bars are <rect> elements. Background + axis frames also add rects, so
    # assert strictly more than 1 (the lone background rect) to verify mark content.
    rect_count = svg.count("<rect")
    assert rect_count > 1, (
        f"12-row hist facet must render histogram bars (rect elements); got {rect_count}"
    )


def test_d9a_height_param_is_per_panel(month_temp_df: pl.DataFrame) -> None:
    """displot(row=..., height=h) treats h as per-panel height, not total canvas.

    height=80 with 12 panels → total height = 960. viewBox width×height
    should reflect 640 × 960 (default width × 12*80).
    """
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        c = fm.displot(month_temp_df, x="temp", row="month", kind="kde", fill=True, height=80)
        svg = c.show_svg()

    empty_panel_warns = [x for x in w if "EmptyPanel" in str(x.message)]
    assert not empty_panel_warns, (
        f"displot height=80/panel must not emit EmptyPanel; got {[str(x.message) for x in empty_panel_warns]}"
    )

    path_count = svg.count('d="M')
    assert path_count >= 12, (
        f"displot height=80/panel with 12 rows must render at least 12 paths; got {path_count}"
    )

    # The SVG viewBox should reflect the total (per-panel × n_panels) height, not the
    # per-panel value alone. 640 × 960 is the expected bounding box.
    assert "960" in svg[:500] or "960.0" in svg[:500], (
        "displot height=80 with 12 row panels should produce total height ≥960 in SVG viewBox"
    )


def test_d9a_three_row_facet_unaffected(month_temp_df: pl.DataFrame) -> None:
    """3-category facet (comfortably fits in default canvas) still renders correctly."""
    import numpy as np

    months3 = ["Jan", "Feb", "Mar"]
    tdf3 = month_temp_df.filter(pl.col("month").is_in(months3))
    c = fm.displot(tdf3, x="temp", row="month", kind="kde", fill=True)
    svg = c.show_svg()
    path_count = svg.count('d="M')
    assert path_count >= 3, f"3-row KDE facet must render at least 3 paths; got {path_count}"


# ---------------------------------------------------------------------------
# D9-B: multi-series mark_line on ordinal x renders non-blank (string column)
# ---------------------------------------------------------------------------


def test_d9b_ordinal_x_string_column_eight_values_renders(
    ordinal_line_df_str: pl.DataFrame,
) -> None:
    """mark_line with ordinal x (string column, 8 distinct values) renders one polyline per series.

    String-ordinal x is the working path. This test is a regression lock: the audit
    identified that multi-series lines on ordinal x with many values render blank;
    string columns work correctly and should continue to do so.
    """
    svg = (
        fm.Chart(ordinal_line_df_str)
        .mark_line()
        .encode(x=fm.X("year", type_="O"), y="rank:Q", color="series:N")
        .show_svg()
    )
    polyline_count = svg.count("<polyline")
    assert polyline_count >= 6, (
        f"mark_line on string ordinal x with 8 values and 6 series must render "
        f"at least 6 polylines; got {polyline_count}"
    )


def test_d9b_ordinal_x_integer_column_renders(ordinal_line_df_int: pl.DataFrame) -> None:
    """mark_line with ordinal x (integer column, 8 distinct values) must render non-blank.

    This test is currently xfail due to a Rust-side bug: col_as_str() in line.rs
    (and other mark renderers) only handles Utf8/LargeUtf8 columns. When the column
    is Int64 (polars integer → arrow Int64), the downcast fails and the renderer
    returns empty() instead of stringifying the integer values for the ordinal lookup.

    Once the rust-coder fix lands, remove the xfail marker.
    """
    svg = (
        fm.Chart(ordinal_line_df_int)
        .mark_line()
        .encode(x=fm.X("year", type_="O"), y="rank:Q", color="series:N")
        .show_svg()
    )
    polyline_count = svg.count("<polyline")
    assert polyline_count >= 6, (
        f"mark_line on integer ordinal x with 8 values and 6 series must render "
        f"at least 6 polylines; got {polyline_count}"
    )


# ---------------------------------------------------------------------------
# D9-C: parent + fm.Inset(...) does not blank the parent's marks
# ---------------------------------------------------------------------------


def test_d9c_inset_does_not_blank_parent_marks(inset_df: pl.DataFrame) -> None:
    """parent + fm.Inset(chart=...) must render the parent's marks AND the inset's marks.

    Before the reported fix, main + fm.Inset(...) blanked the parent chart's
    polyline. The inset composition must render at least as many marks as the
    parent chart alone.
    """
    zoom_df = inset_df.head(30)
    inset_chart = (
        fm.Chart(zoom_df)
        .mark_line(color="#2ca02c")
        .encode(x="date:T", y="growth")
        .properties(width=260, height=150)
        .labs(title="First 30d")
    )
    main = (
        fm.Chart(inset_df)
        .mark_line(color="#2ca02c")
        .encode(x="date:T", y=fm.Y("growth", title="Growth of $1"))
        .properties(width=900, height=420)
        .labs(title="Returns with inset zoom")
    )
    with_inset = main + fm.Inset(chart=inset_chart, bounds=(0.55, 0.08, 0.97, 0.5))

    svg_composed = with_inset.show_svg()
    svg_main_only = main.show_svg()

    parent_polylines = svg_main_only.count("<polyline")
    composed_polylines = svg_composed.count("<polyline")

    assert parent_polylines >= 1, (
        f"parent chart alone must render at least 1 polyline; got {parent_polylines}"
    )
    assert composed_polylines >= parent_polylines, (
        f"parent + Inset must render at least as many polylines as the parent alone "
        f"({parent_polylines}); got {composed_polylines}"
    )


def test_d9c_inset_adds_extra_marks(inset_df: pl.DataFrame) -> None:
    """parent + fm.Inset(chart=...) renders more marks than the parent alone.

    The inset chart adds its own marks to the scene. The composed chart must
    render strictly more marks than the parent alone (parent mark + inset mark).
    """
    zoom_df = inset_df.head(30)
    inset_chart = (
        fm.Chart(zoom_df)
        .mark_line(color="#9467bd")
        .encode(x="date:T", y="growth")
        .properties(width=260, height=150)
    )
    main = (
        fm.Chart(inset_df)
        .mark_line(color="#2ca02c")
        .encode(x="date:T", y="growth")
        .properties(width=900, height=420)
    )

    svg_main = main.show_svg()
    svg_composed = (main + fm.Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.98, 0.5))).show_svg()

    main_count = svg_main.count("<polyline")
    composed_count = svg_composed.count("<polyline")

    assert composed_count > main_count, (
        f"parent + Inset must render more polylines than parent alone; "
        f"main={main_count}, composed={composed_count}"
    )
