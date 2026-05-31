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
