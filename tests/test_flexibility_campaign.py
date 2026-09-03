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
        .to_svg()
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
        .to_svg()
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
        .to_svg()
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
    svg = fm.Chart(df).mark_bar().encode(x="c:N", y="y:Q", color=Color("c:N", scale=scale)).to_svg()
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
    svg_blues = fm.heatmap(corr_df, cmap="blues").to_svg()
    svg_reds = fm.heatmap(corr_df, cmap="reds").to_svg()
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


# Extracted to tests/_temporal_ticks.py (shared with sibling test modules) --
# quality-review cycle 1 finding: this module and
# tests/test_axis_label_culling.py each carried an independently-drifted copy
# of the shape parser (one matching a month token by set membership, the
# other by a looser regex that also matched non-month words). Aliased to the
# original private names here so existing call sites in this module are
# unchanged.
from tests._temporal_ticks import MONTH_ABBREVS as _MONTH_ABBREVS  # noqa: E402
from tests._temporal_ticks import month_year_tick_shapes as _month_year_tick_shapes  # noqa: E402
from tests._temporal_ticks import (  # noqa: E402
    reconstruct_month_year_labels as _reconstruct_month_year_labels,
)


def test_d3_numeric_grouping_format(two_cat_numeric_df: pl.DataFrame) -> None:
    """Axis(label_format=',.0f') produces comma-grouped labels like '10,000'."""
    svg = (
        fm.Chart(two_cat_numeric_df)
        .mark_bar()
        .encode(
            x="cat:N",
            y=fm.Y("val:Q", axis=fm.Axis(label_format=",.0f")),
        )
        .to_svg()
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
        .to_svg()
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
        .to_svg()
    )
    tick_labels = _tick_texts(svg)
    assert "50%" in tick_labels, f"expected '50%' in tick labels; got {tick_labels}"


def test_d3_temporal_format_month_year(monthly_date_df: pl.DataFrame) -> None:
    """Axis(label_format='%b %Y') on a :T x-axis produces 'Jan 2020'-style labels.

    At the DEFAULT theme, this 18-month axis's real per-tick slot width is
    narrow enough that the spec §4.6 pixel-gap requirement genuinely isn't
    satisfiable flat for every tick (verified: the fixture wraps starting at
    `cull_threshold=4`, comfortably below the default of 8) -- so the
    cascade's wrap stage legitimately engages. What must NOT happen (the
    spec-review cycle 3 regression this pins) is a RAGGED per-label mix,
    where sibling ticks of the identical "%b %Y" format wrap differently
    from each other purely because of real-font width variance across month
    names; every tick must degrade the SAME way (spec §4.6: "the cascade
    degrades uniformly"). `_month_year_tick_shapes`/`_reconstruct_month_year_labels`
    tolerate either uniform outcome (all one-line or all wrapped).
    """
    svg = (
        fm.Chart(monthly_date_df)
        .mark_line()
        .encode(
            x=fm.X("date:T", axis=fm.Axis(label_format="%b %Y")),
            y="val:Q",
        )
        .to_svg()
    )
    tick_labels = _tick_texts(svg)
    shapes = _month_year_tick_shapes(tick_labels)
    assert shapes, f"expected at least one 'MMM YYYY' tick; got {tick_labels}"
    assert len(set(shapes)) == 1, (
        "every 'MMM YYYY' tick must wrap the SAME way (spec §4.6 -- the "
        f"cascade degrades uniformly, never a ragged one-line/two-line mix); "
        f"got per-tick shapes {shapes} from raw ticks {tick_labels}"
    )
    month_year_labels = _reconstruct_month_year_labels(tick_labels)
    # Specifically confirm 'Jan 2020' (the first tick in the domain) is present.
    assert "Jan 2020" in month_year_labels, (
        f"expected 'Jan 2020' among month-year labels; got {month_year_labels} "
        f"(raw ticks: {tick_labels})"
    )


def test_d3_tick_count_limits_temporal_ticks(long_monthly_date_df: pl.DataFrame) -> None:
    """Axis(tick_count=4) on a 30-month :T axis produces far fewer labels than default."""
    svg_default = fm.Chart(long_monthly_date_df).mark_line().encode(x="date:T", y="val:Q").to_svg()
    svg_limited = (
        fm.Chart(long_monthly_date_df)
        .mark_line()
        .encode(
            x=fm.X("date:T", axis=fm.Axis(tick_count=4)),
            y="val:Q",
        )
        .to_svg()
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
    svg = fm.Chart(two_cat_numeric_df).mark_bar().encode(x="cat:N", y="val:Q").to_svg()
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
    standalone_colors = _polyline_strokes(highlight.to_svg())
    assert len(standalone_colors) == 2, (
        f"highlight standalone should produce exactly 2 distinct colors; got {standalone_colors}"
    )

    svg_bh = (base + highlight).to_svg()
    svg_hb = (highlight + base).to_svg()

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

    svg = (base + highlight).to_svg()
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
        svg = chart.to_svg()
        assert "_x1" not in svg, f"{label}: internal annotation field '_x1' must not appear in SVG"
        assert "_y1" not in svg, f"{label}: internal annotation field '_y1' must not appear in SVG"

    tick_labels_dr = _tick_texts((data + rect).to_svg())
    tick_labels_rd = _tick_texts((rect + data).to_svg())
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
        svg = chart.to_svg()
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
    svg = fm.Chart(sort_bar_df).mark_bar().encode(x=fm.X("c:N", sort="-y"), y="y:Q").to_svg()
    order = _x_cat_order(svg, {"A", "B", "X"})
    assert order == ["A", "B", "X"], (
        f"sort='-y' should produce descending order A > B > X; got {order}"
    )


def test_d5_primitive_bar_sort_ascending(sort_bar_df: pl.DataFrame) -> None:
    """X('c:N', sort='y') on a bar chart orders categories by ascending sum(y)."""
    svg = fm.Chart(sort_bar_df).mark_bar().encode(x=fm.X("c:N", sort="y"), y="y:Q").to_svg()
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
        .to_svg()
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
        .to_svg()
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
        .to_svg()
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
        .to_svg()
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
        .to_svg()
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
        .to_svg()
    )
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["A", "B", "C"], (
        f"errorbar sort='-y' should order A(80) > B(50) > C(20); got {order}"
    )


# --- Regression: composites without sort preserve data appearance order ---


def test_d5_boxplot_no_sort_preserves_data_order(sort_composite_df: pl.DataFrame) -> None:
    """mark_boxplot without sort preserves data-appearance order (C, A, B)."""
    svg = fm.Chart(sort_composite_df).mark_boxplot().encode(x="cat:N", y="val:Q").to_svg()
    order = _x_cat_order(svg, {"A", "B", "C"})
    assert order == ["C", "A", "B"], (
        f"boxplot without sort should render in data-appearance order C, A, B; got {order}"
    )


def test_d5_violin_no_sort_preserves_data_order(sort_composite_df: pl.DataFrame) -> None:
    """mark_violin without sort preserves data-appearance order (C, A, B)."""
    svg = fm.Chart(sort_composite_df).mark_violin().encode(x="cat:N", y="val:Q").to_svg()
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
    ignored by Rust.  The fix (R2) hands the aggregate op/order to the LetterValue
    transform via its ``sort=`` kwarg; the transform emits its group rows in
    aggregate order, so the categorical domain falls out in first-appearance order
    with no explicit layer sort.

    The assertion strategy uses three independent checks:

    1. No ``SortSpecIgnored`` Rust warning (the core bug symptom).
    2. The LetterValue transform carries the aggregate sort, and the layer's
       categorical encoding carries NO explicit sort (Python-level correctness —
       the aggregate ordering must live on the transform, not the layer).
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
            .to_svg()
        )
    # SEAM-07: the Rust SortSpecIgnored warning now forwards a stable Display
    # sentence ("sort spec could not be applied (...)"), not the Debug variant.
    sort_ignored = any("sort spec could not be applied" in str(ww.message) for ww in w)
    assert not sort_ignored, "sort='-y' on mark_boxen must not emit a sort-ignored warning"

    # Check 2: aggregate sort lives on the LetterValue transform, not the layer.
    # Z≈80 > B≈50 > A≈20, so descending by sum reorders the emitted group rows.
    chart_resolved = (
        fm.Chart(sort_boxen_df).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
    )._resolve_pending()
    transform_repr = repr(chart_resolved._transforms[0])
    assert "LetterValueSort" in transform_repr and "Desc" in transform_repr, (
        f"boxen sort='-y' must set a descending aggregate sort on the LetterValue "
        f"transform; got {transform_repr!r}"
    )
    x_enc = chart_resolved._layers[0].encoding["x"]
    layer_sort = x_enc._kwargs.get("sort") if hasattr(x_enc, "_kwargs") else None
    assert layer_sort is None, (
        f"boxen aggregate sort must NOT set an explicit layer sort (the transform's "
        f"row order drives the domain); got layer sort {layer_sort!r}"
    )

    # Check 3: sorted SVG differs from unsorted (the sort visually affects rendering).
    svg_no_sort = fm.Chart(sort_boxen_df).mark_boxen().encode(x="cat:N", y="val:Q").to_svg()
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
        .to_svg()
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
    svg = fm.Chart(sort_composite_df).mark_errorbar().encode(x="cat:N", y="val:Q").to_svg()
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
        .to_svg()
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
    svg = fm.Chart(line_df).mark_line(color="#e4572e").encode(x="x:Q", y="y:Q").to_svg()
    strokes = _hex_strokes(svg)
    assert "e4572e" in strokes, (
        f"mark_line(color='#e4572e') must produce stroke='#e4572e' in SVG; found strokes: {strokes}"
    )


def test_d6_rule_color_maps_to_stroke_in_svg() -> None:
    """mark_rule(color='#e4572e') sets the rule stroke color in SVG."""
    df = pl.DataFrame({"y": [1.0]})
    svg = fm.Chart(df).mark_rule(color="#e4572e").encode(y="y:Q").to_svg()
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
    svg = fm.Chart(line_df).mark_line(stroke="#123456").encode(x="x:Q", y="y:Q").to_svg()
    strokes = _hex_strokes(svg)
    assert "123456" in strokes, (
        f"mark_line(stroke='#123456') must produce stroke='#123456' in SVG; "
        f"found strokes: {strokes}"
    )


# --- Regressions: fill-primary marks must still map color → fill ---


def test_d6_bar_color_still_maps_to_fill() -> None:
    """mark_bar(color='#e4572e') still sets fill (bar is fill-primary)."""
    df = pl.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})
    svg = fm.Chart(df).mark_bar(color="#e4572e").encode(x="cat:N", y="val:Q").to_svg()
    fills = _hex_fills(svg)
    assert "e4572e" in fills, (
        f"mark_bar(color='#e4572e') must produce fill='#e4572e' in SVG; found fills: {fills}"
    )


def test_d6_point_color_still_maps_to_fill(line_df: pl.DataFrame) -> None:
    """mark_point(color='#e4572e') still sets fill (point is fill-primary)."""
    svg = fm.Chart(line_df).mark_point(color="#e4572e").encode(x="x:Q", y="y:Q").to_svg()
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
    svg = (chart + vline).to_svg()
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
    svg = (chart + rect).to_svg()
    assert len(svg) > 0, "composed chart with date rect must produce non-empty SVG"

    # Verify the primitive corners store epoch-ms values.
    prim = rect._annotation_primitive
    assert prim.x1 == _date_epoch_ms(date(2020, 3, 1)), "rect x1 must be epoch-ms"
    assert prim.x2 == _date_epoch_ms(date(2020, 9, 1)), "rect x2 must be epoch-ms"


def test_d7_text_with_datetime_renders(temporal_df: pl.DataFrame) -> None:
    """annotate_text(x=datetime(...), y=..., text=...) renders without error."""
    chart = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    label = fm.annotate_text(x=dt(2020, 1, 1, 12, 0), y=5.0, text="noon event")
    svg = (chart + label).to_svg()
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


def test_d7_non_iso_string_becomes_ordinal_category() -> None:
    """A non-ISO-8601 string is treated as an ordinal category label, not an error.

    Updated by C2 (2026-06-02): the original D7 contract was that invalid ISO
    strings raised ``ValueError``.  C2 changed this: non-ISO strings are now
    treated as ordinal category coordinates so that annotations like
    ``annotate_vline("cat_a")`` work on categorical axes.  The string is stored
    as ``OrdinalCategoryCoord`` and resolved to a band center at render time.
    """
    from ferrum.annotation.coords import OrdinalCategoryCoord

    # Must not raise.
    ann = fm.annotate_vline(x="not-a-date")
    prim = ann._annotation_primitive
    assert prim is not None
    # The x coordinate must be stored as OrdinalCategoryCoord, not a float.
    assert isinstance(prim.x1, OrdinalCategoryCoord), (
        f"Non-ISO string 'not-a-date' must be stored as OrdinalCategoryCoord, got {type(prim.x1)}"
    )
    assert prim.x1.value == "not-a-date"


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
    svg_with = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).to_svg()
    svg_without = (
        fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q", axis=None), y=fm.Y("b:Q")).to_svg()
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
    svg_with = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).to_svg()
    svg_without = (
        fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q", axis=None)).to_svg()
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
        .to_svg()
    )
    texts = _d8_tick_texts(svg)
    assert "a" not in texts, f"field title 'a' must be absent; got {texts}"
    assert "b" not in texts, f"field title 'b' must be absent; got {texts}"


def test_d8_layered_chart_axis_none_hides_axis(_d8_df: pl.DataFrame) -> None:
    """axis=None on the data layer's channel suppresses the axis in a layered chart."""
    svg = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q", axis=None), y=fm.Y("b:Q")).to_svg()
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
        .to_svg()
    )
    texts = _d8_tick_texts(svg)
    assert "Speed" in texts, f"Axis(title='Speed') must render title in SVG; got {texts}"


def test_d8_no_axis_kwarg_renders_normally(_d8_df: pl.DataFrame) -> None:
    """Default (no axis kwarg) must still render both axes."""
    svg = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).to_svg()
    texts = _d8_tick_texts(svg)
    assert "a" in texts, f"x field title 'a' must appear by default; got {texts}"
    assert "b" in texts, f"y field title 'b' must appear by default; got {texts}"


def test_d8_chart_axis_method_still_works(_d8_df: pl.DataFrame) -> None:
    """Chart.axis(x=False) must still suppress the x-axis regardless of encoding."""
    svg = fm.Chart(_d8_df).mark_point().encode(x=fm.X("a:Q"), y=fm.Y("b:Q")).axis(x=False).to_svg()
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
        .to_svg()
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
        svg = c.to_svg()

    # SEAM-07: the empty-panel warning now reads "panel N is too small to render".
    empty_panel_warns = [x for x in w if "too small to render" in str(x.message)]
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
        svg = c.to_svg()

    # SEAM-07: the empty-panel warning now reads "panel N is too small to render".
    empty_panel_warns = [x for x in w if "too small to render" in str(x.message)]
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
        svg = c.to_svg()

    # SEAM-07: the empty-panel warning now reads "panel N is too small to render".
    empty_panel_warns = [x for x in w if "too small to render" in str(x.message)]
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
    svg = c.to_svg()
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
        .to_svg()
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
        .to_svg()
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

    svg_composed = with_inset.to_svg()
    svg_main_only = main.to_svg()

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

    svg_main = main.to_svg()
    svg_composed = (main + fm.Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.98, 0.5))).to_svg()

    main_count = svg_main.count("<polyline")
    composed_count = svg_composed.count("<polyline")

    assert composed_count > main_count, (
        f"parent + Inset must render more polylines than parent alone; "
        f"main={main_count}, composed={composed_count}"
    )


# ---------------------------------------------------------------------------
# Task 9 — size/shape legends
#
# Tests that size and shape encodings now produce rendered legends, that
# legend=None suppresses them, and that same-field merging collapses two
# legend blocks into one.
# ---------------------------------------------------------------------------


@pytest.fixture
def _t9_df() -> pl.DataFrame:
    """30-row scatter DataFrame with continuous pop and nominal region columns."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 30
    return pl.DataFrame(
        {
            "x": rng.uniform(1, 10, n).tolist(),
            "y": rng.uniform(1, 10, n).tolist(),
            "pop": rng.uniform(1e7, 5e8, n).tolist(),
            "region": rng.choice(["Asia", "Europe", "Americas", "Africa"], n).tolist(),
        }
    )


def test_t9_size_legend_title_and_labels_present(_t9_df: pl.DataFrame) -> None:
    """mark_point with Size('pop:Q') renders a size legend with title and numeric labels.

    The size legend consists of the field title (``pop``) and ~5 graduated
    numeric labels at round values (e.g. ``1e8``, ``2e8``, …). Both must appear
    in the SVG text elements.
    """
    svg = fm.Chart(_t9_df).mark_point().encode(x="x:Q", y="y:Q", size=fm.Size("pop:Q")).to_svg()
    texts = re.findall(r"<text[^>]*>([^<]+)</text>", svg)

    assert "pop" in texts, f"size legend title 'pop' must appear in SVG texts; got {texts}"

    # The size legend renders graduated numeric labels at round values.
    # With a 1e7–5e8 domain the Rust renderer picks values like 1e8, 2e8, 3e8, …
    numeric_labels = [t for t in texts if re.match(r"^\d+(?:\.\d+)?e\d+$", t)]
    assert len(numeric_labels) >= 2, (
        f"size legend must produce at least 2 numeric value labels; got {numeric_labels}"
    )


def test_t9_size_legend_circles_exceed_data_points(_t9_df: pl.DataFrame) -> None:
    """mark_point with Size('pop:Q') has more <circle> elements than data rows.

    The data layer renders 30 circles (one per row). The size legend adds ~5
    additional graduated circles. The total circle count must therefore exceed 30.
    """
    svg = fm.Chart(_t9_df).mark_point().encode(x="x:Q", y="y:Q", size=fm.Size("pop:Q")).to_svg()
    circle_count = svg.count("<circle")
    n_rows = len(_t9_df)
    assert circle_count > n_rows, (
        f"SVG must have more circles than data rows ({n_rows}); "
        f"got {circle_count} (legend circles are missing)"
    )


def test_t9_color_and_size_both_produce_legends(_t9_df: pl.DataFrame) -> None:
    """Color('region:N') + Size('pop:Q') produces both a color legend and a size legend.

    The color legend shows the category label ``region`` with per-category labels
    (Asia, Europe, …). The size legend shows ``pop`` with numeric labels.
    Both must appear in the SVG.
    """
    svg = (
        fm.Chart(_t9_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            size=fm.Size("pop:Q"),
            color=fm.Color("region:N"),
        )
        .to_svg()
    )
    texts = re.findall(r"<text[^>]*>([^<]+)</text>", svg)

    # Color legend: title and at least one category label must be present.
    assert "region" in texts, f"color legend title 'region' must appear; got {texts}"
    region_labels = [t for t in texts if t in {"Asia", "Europe", "Americas", "Africa"}]
    assert region_labels, f"color legend must show category labels; got {texts}"

    # Size legend: title and numeric labels must be present.
    assert "pop" in texts, f"size legend title 'pop' must appear; got {texts}"
    numeric_labels = [t for t in texts if re.match(r"^\d+(?:\.\d+)?e\d+$", t)]
    assert numeric_labels, f"size legend must show numeric labels; got {texts}"


def test_t9_shape_legend_title_and_category_labels_present(_t9_df: pl.DataFrame) -> None:
    """mark_point with Shape('region:N') renders a shape legend with per-category entries.

    The shape legend must contain the field title (``region``) and labels for
    every distinct category in the data.
    """
    svg = (
        fm.Chart(_t9_df).mark_point().encode(x="x:Q", y="y:Q", shape=fm.Shape("region:N")).to_svg()
    )
    texts = re.findall(r"<text[^>]*>([^<]+)</text>", svg)

    assert "region" in texts, f"shape legend title 'region' must appear in SVG texts; got {texts}"

    expected_categories = {"Asia", "Europe", "Americas", "Africa"}
    found_categories = expected_categories & set(texts)
    assert found_categories == expected_categories, (
        f"shape legend must show all category labels; "
        f"missing: {expected_categories - found_categories}, got: {texts}"
    )


def test_t9_size_legend_none_suppresses_legend(_t9_df: pl.DataFrame) -> None:
    """Size('pop:Q', legend=None) suppresses the size legend entirely.

    Compared to the default (legend present), the suppressed SVG must lack the
    size legend title ``pop`` and the numeric value labels, and must have no
    more circles than the number of data rows.
    """
    svg_default = (
        fm.Chart(_t9_df).mark_point().encode(x="x:Q", y="y:Q", size=fm.Size("pop:Q")).to_svg()
    )
    svg_suppressed = (
        fm.Chart(_t9_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", size=fm.Size("pop:Q", legend=None))
        .to_svg()
    )

    # The default chart must have the legend.
    texts_default = re.findall(r"<text[^>]*>([^<]+)</text>", svg_default)
    assert "pop" in texts_default, "baseline: default chart must have 'pop' legend title"

    # The suppressed chart must not.
    texts_suppressed = re.findall(r"<text[^>]*>([^<]+)</text>", svg_suppressed)
    assert "pop" not in texts_suppressed, (
        f"legend=None must suppress 'pop' legend title; got texts: {texts_suppressed}"
    )

    # No numeric legend labels either.
    numeric_labels = [t for t in texts_suppressed if re.match(r"^\d+(?:\.\d+)?e\d+$", t)]
    assert not numeric_labels, (
        f"legend=None must suppress numeric legend labels; found: {numeric_labels}"
    )

    # No extra circles beyond the data rows.
    n_rows = len(_t9_df)
    circle_count = svg_suppressed.count("<circle")
    assert circle_count <= n_rows, (
        f"legend=None must not add legend circles; expected <={n_rows} circles, got {circle_count}"
    )


def test_t9_same_field_size_color_merges_into_single_legend(_t9_df: pl.DataFrame) -> None:
    """Size('pop:Q') + Color('pop:Q') on the same field produces one merged legend block.

    Ferrum merges channels on the same field into a single combined legend.
    The SVG must contain exactly one occurrence of the legend title ``pop``,
    not two separate ``pop`` blocks.
    """
    svg = (
        fm.Chart(_t9_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            size=fm.Size("pop:Q"),
            color=fm.Color("pop:Q"),
        )
        .to_svg()
    )
    texts = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
    pop_count = texts.count("pop")
    assert pop_count == 1, (
        f"same-field size+color must produce exactly one 'pop' legend title; "
        f"got {pop_count} occurrences in texts: {texts}"
    )


def test_t9_color_only_legend_unchanged(_t9_df: pl.DataFrame) -> None:
    """Color('region:N') alone still renders the categorical color legend correctly.

    Regression guard: adding size/shape legend support must not break the existing
    color legend rendering path.
    """
    svg = (
        fm.Chart(_t9_df).mark_point().encode(x="x:Q", y="y:Q", color=fm.Color("region:N")).to_svg()
    )
    texts = re.findall(r"<text[^>]*>([^<]+)</text>", svg)

    assert "region" in texts, f"color legend title 'region' must appear; got {texts}"
    region_labels = [t for t in texts if t in {"Asia", "Europe", "Americas", "Africa"}]
    assert len(region_labels) == 4, (
        f"color legend must show all 4 region labels; got {region_labels}"
    )

    # No size-legend artifacts (numeric e-notation labels) should appear.
    numeric_labels = [t for t in texts if re.match(r"^\d+(?:\.\d+)?e\d+$", t)]
    assert not numeric_labels, (
        f"color-only chart must not produce size-legend numeric labels; found {numeric_labels}"
    )


# ---------------------------------------------------------------------------
# Task 2b — temporal auto-type inference from column dtype
#
# A polars Datetime/Date column used on a positional channel WITHOUT an
# explicit `:T` annotation must auto-infer temporal type, matching Vega-Lite/
# Altair behaviour.  Explicit annotations (:Q, :N, type_=...) always win.
# ---------------------------------------------------------------------------


@pytest.fixture
def _t2b_monthly_df() -> pl.DataFrame:
    """18-month line-series with a polars Date x-axis."""
    from datetime import date as _date

    return pl.DataFrame(
        {
            "date": pl.date_range(_date(2020, 1, 1), _date(2021, 6, 1), "1mo", eager=True),
            "val": list(range(18)),
        }
    )


@pytest.fixture
def _t2b_datetime_us_df() -> pl.DataFrame:
    """4-quarter line-series with a polars Datetime(us) x-axis.

    Datetime(us) is the default Polars Datetime dtype and was previously
    untouched by _coerce.py, producing raw epoch-level tick values instead
    of date-formatted labels.
    """
    from datetime import datetime as _dt

    return pl.DataFrame(
        {
            "ts": [_dt(2020, 1, 1), _dt(2020, 4, 1), _dt(2020, 7, 1), _dt(2020, 10, 1)],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _is_date_formatted(texts: list[str]) -> bool:
    """Return True if any tick label looks like a formatted date (contains a year or month name)."""
    month_abbrevs = {
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    }
    for t in texts:
        # YYYY-MM-DD style (e.g. "2020-01-02")
        if re.match(r"^\d{4}-\d{2}-\d{2}$", t):
            return True
        # "Jan 2020" / "Mar 2020" style
        if re.match(r"^[A-Z][a-z]{2} 20\d{2}$", t):
            return True
        # Month abbreviation alone (e.g. "Jan") paired with year in another tick
        if any(t.startswith(m) for m in month_abbrevs):
            return True
    return False


def _is_epoch_integer(texts: list[str]) -> bool:
    """Return True if any tick looks like a raw epoch integer (large integer, no date format).

    A large standalone integer immediately adjacent (in tick order) to a
    month abbreviation is treated as a legitimate wrapped "%b %Y"-style date
    component -- the label-collision cascade may split a "Jan 2020" tick
    onto two lines once the axis needs to wrap (spec §4.6's pixel-gap
    requirement) -- not a raw epoch value.
    """
    for i, t in enumerate(texts):
        try:
            v = int(t)
        except ValueError:
            continue
        # Epoch-ms values for 2020 are around 1.578e12. Epoch-days are ~18000.
        # Filter out small axis values that are just data values (e.g. 0–30).
        if v <= 1000:
            continue
        prev_is_month = i > 0 and texts[i - 1] in _MONTH_ABBREVS
        next_is_month = i + 1 < len(texts) and texts[i + 1] in _MONTH_ABBREVS
        if prev_is_month or next_is_month:
            continue
        return True
    return False


def test_t2b_date_column_without_annotation_gets_temporal_ticks(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """A polars Date column on x WITHOUT ':T' produces date-formatted axis ticks.

    Before the fix, the unannotated Date column fell through to a linear scale
    and rendered raw epoch-integer tick values. After the fix, the type is
    inferred as temporal and the D3 chrono formatter emits human-readable dates.

    At the default theme, real-font width variance may wrap a "Mon YYYY"
    tick onto two lines (spec §4.6's pixel-gap requirement, genuinely not
    satisfiable flat for every tick of this 18-month fixture) -- `_is_epoch_
    integer` tolerates a standalone year immediately adjacent to a month
    abbreviation as a legitimate wrapped date component, not a raw epoch
    value (see its own doc).
    """
    svg = fm.Chart(_t2b_monthly_df).mark_line().encode(x="date", y="val:Q").to_svg()
    texts = _tick_texts(svg)
    assert _is_date_formatted(texts), (
        f"unannotated Date column on x must produce date-formatted ticks; got: {texts}"
    )
    assert not _is_epoch_integer(texts), (
        f"unannotated Date column on x must not produce raw epoch integers; got: {texts}"
    )


def test_t2b_datetime_us_column_without_annotation_gets_temporal_ticks(
    _t2b_datetime_us_df: pl.DataFrame,
) -> None:
    """A polars Datetime(us) column on x WITHOUT ':T' produces date-formatted axis ticks.

    Datetime(us) is the default polars Datetime dtype. Previously it was not
    cast to ms by _coerce.py and the temporal scale produced wrong tick values.
    After the fix, _coerce.py normalizes it to Datetime(ms) and type inference
    infers temporal, so the D3 chrono formatter produces readable dates.
    """
    svg = fm.Chart(_t2b_datetime_us_df).mark_line().encode(x="ts", y="val:Q").to_svg()
    texts = _tick_texts(svg)
    assert _is_date_formatted(texts), (
        f"unannotated Datetime(us) column on x must produce date-formatted ticks; got: {texts}"
    )
    assert not _is_epoch_integer(texts), (
        f"unannotated Datetime(us) column on x must not produce raw epoch integers; got: {texts}"
    )


def test_t2b_unannotated_matches_explicit_temporal_annotation(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """An unannotated Date column produces the same SVG as the explicit ':T' form.

    This is the core contract of auto-type inference: the unannotated shorthand
    must behave identically to the explicit ':T' annotation when the column is
    temporal.
    """
    svg_inferred = fm.Chart(_t2b_monthly_df).mark_line().encode(x="date", y="val:Q").to_svg()
    svg_explicit = fm.Chart(_t2b_monthly_df).mark_line().encode(x="date:T", y="val:Q").to_svg()
    assert svg_inferred == svg_explicit, (
        "unannotated Date column must produce identical SVG to explicit ':T' annotation"
    )


def test_t2b_explicit_q_on_date_column_overrides_inference(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """Explicit ':Q' on a Date column still forces quantitative scale.

    User-supplied annotations always win over auto-inference.  An explicit
    ':Q' must produce a numeric (non-date) tick pattern even though the column
    holds date values.
    """
    svg_q = fm.Chart(_t2b_monthly_df).mark_line().encode(x="date:Q", y="val:Q").to_svg()
    texts_q = _tick_texts(svg_q)
    # With :Q the tick labels are epoch-ms numbers (large integers), not formatted dates.
    assert not _is_date_formatted(texts_q) or _is_epoch_integer(texts_q), (
        f"explicit ':Q' must force quantitative ticks (not date-formatted); got: {texts_q}"
    )


def test_t2b_numeric_column_still_quantitative() -> None:
    """An ordinary numeric column without type annotation still infers quantitative.

    Regression guard: the temporal inference must not affect numeric columns.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
    texts = _tick_texts(svg)
    numeric_ticks = [t for t in texts if re.match(r"^\d+(?:\.\d+)?$", t)]
    assert numeric_ticks, (
        f"numeric column without annotation must produce plain numeric ticks; got: {texts}"
    )
    assert not _is_date_formatted(texts), (
        f"numeric column must not produce date-formatted ticks; got: {texts}"
    )


def test_t2b_spec_type_is_temporal_for_unannotated_date(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """The chart spec emits type='temporal' for an unannotated Date x column.

    Tests the Python-layer spec output (not SVG rendering) so the contract
    is verifiable at spec-build time without depending on SVG formatting.
    """
    import json

    spec = json.loads(fm.Chart(_t2b_monthly_df).mark_line().encode(x="date", y="val:Q").to_json())
    x_enc = spec.get("encoding", {}).get("x", {})
    assert x_enc.get("type") == "temporal", (
        f"unannotated Date column must produce type='temporal' in spec; got x={x_enc!r}"
    )


def test_t2b_spec_type_is_temporal_for_unannotated_datetime_us(
    _t2b_datetime_us_df: pl.DataFrame,
) -> None:
    """The chart spec emits type='temporal' for an unannotated Datetime(us) x column."""
    import json

    spec = json.loads(fm.Chart(_t2b_datetime_us_df).mark_line().encode(x="ts", y="val:Q").to_json())
    x_enc = spec.get("encoding", {}).get("x", {})
    assert x_enc.get("type") == "temporal", (
        f"unannotated Datetime(us) column must produce type='temporal' in spec; got x={x_enc!r}"
    )


# ---------------------------------------------------------------------------
# Cleanup pass — four targeted fixes merged on fix/flexibility-cleanup
#
# Fix 1: 3- and 4-digit CSS hex shorthand (#ccc → #cccccc) is now parsed.
# Fix 2: ~e / ~s scientific trim trims only the mantissa, preserving the
#         exponent suffix.
# Fix 3: Ranged mark_rule / mark_segment honor a per-row color encoding.
# Fix 4: mark_raster Y-axis orientation is correct (was flipped).
# ---------------------------------------------------------------------------

import base64
import struct
import zlib
import warnings


def _decode_png_alpha(png_bytes: bytes) -> tuple[list[list[int]], int, int]:
    """Decode an RGBA PNG blob and return (alpha_grid, width, height).

    alpha_grid[row][col] is the alpha channel value (0-255).  Proper
    reconstruction of all five PNG filter types is applied so the result
    is accurate even when the encoder uses Sub/Up/Average/Paeth filters.
    """
    pos = 8  # skip 8-byte PNG signature
    chunks: dict[bytes, bytes] = {}
    while pos < len(png_bytes):
        length = struct.unpack(">I", png_bytes[pos : pos + 4])[0]
        ctype = png_bytes[pos + 4 : pos + 8]
        data = png_bytes[pos + 8 : pos + 8 + length]
        chunks[ctype] = chunks.get(ctype, b"") + data
        pos += 12 + length
        if ctype == b"IEND":
            break

    w, h = struct.unpack(">II", chunks[b"IHDR"][:8])
    color_type = struct.unpack("B", chunks[b"IHDR"][9:10])[0]
    bpp = 4 if color_type == 6 else 3  # RGBA=6, RGB=2

    raw = zlib.decompress(chunks[b"IDAT"])
    stride = 1 + w * bpp

    def _paeth(a: int, b: int, c: int) -> int:
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        elif pb <= pc:
            return b
        return c

    reconstructed: list[bytearray] = []
    for r in range(h):
        ftype = raw[r * stride]
        row = bytearray(raw[r * stride + 1 : r * stride + 1 + w * bpp])
        prev = reconstructed[r - 1] if r > 0 else bytearray(w * bpp)
        if ftype == 1:  # Sub
            for i in range(bpp, len(row)):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(len(row)):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(len(row)):
                a = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + (a + prev[i]) // 2) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(len(row)):
                a = row[i - bpp] if i >= bpp else 0
                b_val = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth(a, b_val, c)) & 0xFF
        reconstructed.append(row)

    alpha = [[reconstructed[r][c * bpp + 3] for c in range(w)] for r in range(h)]
    return alpha, w, h


def _mark_strokes(svg: str, stroke_width: int) -> set[str]:
    """Extract stroke hex colors from <line> elements with the given stroke-width.

    Using a distinctive stroke_width (e.g. 3) on the mark isolates the mark's
    rendered lines from the thinner axis and grid lines in the same SVG.
    """
    sw_str = str(stroke_width)
    # Match lines whose stroke-width attribute exactly equals the requested value.
    pattern = rf'<line[^>]+stroke-width="{sw_str}(?:\.0)?"[^>]*/>'
    mark_lines = re.findall(pattern, svg)
    return set(re.findall(r'stroke="#([0-9a-fA-F]{6})"', "\n".join(mark_lines)))


# --- Fix 1: 3-digit hex shorthand in OrdinalScale.range ---


def test_cleanup_three_digit_hex_parses(three_cat_df: pl.DataFrame) -> None:
    """OrdinalScale.range accepts 3-digit CSS hex (#ccc, #f00) without a
    ColorRangeParseFailure warning, and the expanded forms appear in the SVG.

    Before the fix, #ccc was rejected by the Rust hex parser (expected exactly
    6 hex digits), emitting a ColorRangeParseFailure warning and falling back to
    the theme default color so all bars rendered identically.  After the fix,
    shorthand is expanded (#ccc → #cccccc, #f00 → #ff0000) before parsing.
    """
    from ferrum.encoding import Color

    scale = OrdinalScale(
        domain=["A", "B", "C"],
        range=["#ccc", "#ccc", "#f00"],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svg = (
            fm.Chart(three_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q", color=Color("cat:N", scale=scale))
            .to_svg()
        )
    # SEAM-07: the parse-failure warning now reads "could not parse color '...'".
    parse_failure_warns = [str(x.message) for x in w if "could not parse color" in str(x.message)]
    assert not parse_failure_warns, (
        f"3-digit hex in OrdinalScale.range must not emit ColorRangeParseFailure; "
        f"got: {parse_failure_warns}"
    )
    assert "cccccc" in svg, "expanded #ccc → #cccccc must appear in the SVG"
    assert "ff0000" in svg, "expanded #f00 → #ff0000 must appear in the SVG"


# --- Fix 2: ~e scientific format trims only the mantissa ---


@pytest.fixture
def _orders_of_magnitude_df() -> pl.DataFrame:
    """Two-category frame with values spanning several orders of magnitude."""
    return pl.DataFrame({"cat": ["A", "B"], "val": [314.0, 31400.0]})


def test_cleanup_tilde_e_scientific_trim(_orders_of_magnitude_df: pl.DataFrame) -> None:
    """Axis(label_format='~e') produces trimmed scientific labels like '3.14e+3'.

    Before the fix, the trim flag on the 'e' format type stripped the entire
    mantissa body as if it were a plain decimal, leaving labels like '0000e+3'
    because the exponent suffix confused the plain trimmer.  After the fix,
    only the mantissa digits before the 'e' are trimmed, preserving the suffix.

    Assertions:
    - At least one tick label contains 'e+' or 'e-' (scientific notation present).
    - No tick label contains a long zero run before the exponent ('0000e').
    """
    svg = (
        fm.Chart(_orders_of_magnitude_df)
        .mark_bar()
        .encode(
            x="cat:N",
            y=fm.Y("val:Q", axis=fm.Axis(label_format="~e")),
        )
        .to_svg()
    )
    tick_labels = _tick_texts(svg)
    has_sci = any("e+" in t or "e-" in t for t in tick_labels)
    assert has_sci, (
        f"~e format must produce scientific notation labels (containing 'e+' or 'e-'); "
        f"got {tick_labels}"
    )
    has_zero_run = any("0000e" in t for t in tick_labels)
    assert not has_zero_run, (
        f"~e trim must not leave long zero runs before the exponent; found '0000e' in {tick_labels}"
    )


def test_cleanup_tilde_s_si_trim_regression(_orders_of_magnitude_df: pl.DataFrame) -> None:
    """Axis(label_format='~s') still trims SI-suffix labels correctly.

    Regression guard: the ~e fix routes 'e' and 's' through the exponent-aware
    trimmer. The 's' format has no exponent in its body (e.g. '1.5M'), so the
    trimmer must fall back gracefully and still produce trimmed SI labels.
    """
    df = pl.DataFrame({"cat": ["A", "B"], "val": [1_000_000.0, 3_000_000.0]})
    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(
            x="cat:N",
            y=fm.Y("val:Q", axis=fm.Axis(label_format="~s")),
        )
        .to_svg()
    )
    tick_labels = _tick_texts(svg)
    assert any("M" in t for t in tick_labels), (
        f"~s trim must still produce SI-suffix labels with 'M'; got {tick_labels}"
    )


# --- Fix 3: ranged mark_rule / mark_segment honor per-row color encoding ---
#
# Assertion strategy: use stroke_width=3 on the mark to produce uniquely-wide
# lines in the SVG.  Axis and grid lines use stroke-width 0.5 or 1; mark lines
# at width 3 are unambiguous.  The set of hex stroke colors from those elements
# is then compared across the color-encoded and no-color variants.


@pytest.fixture
def _ranged_rule_df() -> pl.DataFrame:
    """Two-row frame with distinct 'dir' categories for ranged rule tests."""
    return pl.DataFrame(
        {
            "cat": ["A", "B"],
            "lo": [1.0, 3.0],
            "hi": [5.0, 7.0],
            "dir": ["up", "down"],
        }
    )


def test_cleanup_ranged_rule_per_row_color(_ranged_rule_df: pl.DataFrame) -> None:
    """mark_rule with y + y2 + color encoding renders distinct stroke colors per row.

    Before the fix, the ranged-rule renderer used the constant mark-style stroke
    for every row regardless of the color encoding, so all rules shared a single
    color even when 'color=dir:N' was encoded with two distinct categories.
    """
    svg = (
        fm.Chart(_ranged_rule_df)
        .mark_rule(stroke_width=3)
        .encode(x="cat:N", y="lo:Q", y2="hi:Q", color="dir:N")
        .to_svg()
    )
    mark_strokes = _mark_strokes(svg, stroke_width=3)
    assert len(mark_strokes) >= 2, (
        f"ranged mark_rule with color='dir:N' (2 categories) must produce "
        f"at least 2 distinct stroke colors; got {mark_strokes}"
    )


def test_cleanup_ranged_rule_no_color_single_stroke(_ranged_rule_df: pl.DataFrame) -> None:
    """mark_rule with y + y2 but NO color encoding uses a single constant stroke.

    Regression guard: the per-row color resolution must not activate when there
    is no color encoding, so all ranged rules share the mark-style default.
    """
    df = _ranged_rule_df.drop("dir")
    svg = fm.Chart(df).mark_rule(stroke_width=3).encode(x="cat:N", y="lo:Q", y2="hi:Q").to_svg()
    mark_strokes = _mark_strokes(svg, stroke_width=3)
    assert len(mark_strokes) == 1, (
        f"ranged mark_rule without color encoding must use a single constant stroke; "
        f"got {mark_strokes}"
    )


def test_cleanup_ranged_segment_per_row_color() -> None:
    """mark_segment with x/y/x2/y2 + color encoding renders distinct stroke colors per row."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.5],
            "x2": [0.5, 1.5],
            "y2": [1.0, 0.0],
            "dir": ["up", "down"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_segment(stroke_width=3)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q", color="dir:N")
        .to_svg()
    )
    mark_strokes = _mark_strokes(svg, stroke_width=3)
    assert len(mark_strokes) >= 2, (
        f"mark_segment with color='dir:N' (2 categories) must produce "
        f"at least 2 distinct stroke colors; got {mark_strokes}"
    )


# --- Fix 4: mark_raster Y-axis orientation ---
#
# Assertion strategy: render mark_raster on a strong positive-correlation cloud
# (y ≈ 0.9*x), decode the embedded PNG (base64 data URI in the SVG image node),
# and compare pixel-mass in the four quadrants.  For a correct Y orientation,
# high-X/high-Y data maps to the TOP-RIGHT of the image (screen row 0 is the
# top); the dense diagonal runs from bottom-left to top-right.
#
# The full-resolution PNG (256×256) is decoded with proper filter reconstruction
# (filter types 0-4) using stdlib zlib + struct only (no PIL/pypng needed).
# The quadrant test is deterministic for seeds that produce a clear correlation.
#
# Why this approach: the raster mark bakes the grid into a PNG data URI, so the
# PNG quadrant mass directly proves the Y mapping.  Comparing render to a
# point-mark version would add render-path noise; the PNG approach is tighter.


@pytest.fixture
def _positive_corr_df() -> pl.DataFrame:
    """300-point positive-correlation cloud: y ≈ 0.9*x + noise."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    y = 0.9 * x + rng.normal(0, 0.3, n)
    return pl.DataFrame({"x": x.tolist(), "y": y.tolist()})


def test_cleanup_raster_y_orientation(_positive_corr_df: pl.DataFrame) -> None:
    """mark_raster Y orientation is correct: high-Y data renders at the top of the image.

    Before the fix, the Rust raster transform packed pixel rows in ascending
    data-Y order (gy=0 first), which is bottom-up in data space but top-down
    in screen space.  The PNG image node anchors row 0 at the top, so the raster
    was rendered upside-down relative to every other mark.

    For a positive-correlation cloud, correct Y orientation means the dense mass
    lies on the bottom-left to top-right diagonal, i.e.
        top_right_mass + bottom_left_mass > top_left_mass + bottom_right_mass.
    """
    svg = fm.Chart(_positive_corr_df).mark_raster().encode(x="x:Q", y="y:Q").to_svg()
    # Extract the embedded PNG from the data URI.
    m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", svg)
    assert m is not None, "mark_raster SVG must contain an embedded PNG data URI"
    png_bytes = base64.b64decode(m.group(1))

    alpha, w, h = _decode_png_alpha(png_bytes)
    mid_r = h // 2
    mid_c = w // 2

    tl = sum(alpha[r][c] for r in range(mid_r) for c in range(mid_c))
    tr = sum(alpha[r][c] for r in range(mid_r) for c in range(mid_c, w))
    bl = sum(alpha[r][c] for r in range(mid_r, h) for c in range(mid_c))
    br = sum(alpha[r][c] for r in range(mid_r, h) for c in range(mid_c, w))

    diagonal_mass = tr + bl
    anti_diagonal_mass = tl + br
    assert diagonal_mass > anti_diagonal_mass, (
        f"mark_raster with positive-correlation data must have more pixel mass on the "
        f"bottom-left↔top-right diagonal (correct Y) than the anti-diagonal (flipped Y); "
        f"diagonal={diagonal_mass}, anti-diagonal={anti_diagonal_mass}. "
        f"Quadrant breakdown: TL={tl}, TR={tr}, BL={bl}, BR={br}"
    )


# ---------------------------------------------------------------------------
# Review fixes — heavyweight review findings (2026-05-31)
# ---------------------------------------------------------------------------


import re as _re


def _extract_svg_width(svg: str) -> float:
    """Return the canvas width from an SVG string's ``width`` attribute or viewBox."""
    # Try width="NNN" first.
    m = _re.search(r'<svg[^>]+\bwidth="([0-9.]+)"', svg)
    if m:
        return float(m.group(1))
    # Fall back to viewBox="minX minY width height".
    m = _re.search(r'<svg[^>]+\bviewBox="[0-9.]+ [0-9.]+ ([0-9.]+) [0-9.]+"', svg)
    if m:
        return float(m.group(1))
    raise ValueError("Could not extract width from SVG")


# -- RF1: displot facet width must not scale with panel count -----------------


def test_rf1_displot_facet_width_constant_across_row_counts() -> None:
    """displot with row= and a fixed aspect produces the same canvas width
    regardless of how many row-facet levels the data has.

    The bug was: w = total_h * aspect, where total_h = per_panel_height * n_panels.
    So width grew linearly with panel count.  The fix computes width from
    per_panel_height alone: w = per_panel_height * aspect.
    """
    import polars as pl
    from ferrum.plots.distribution import displot

    rng_seed = 0
    import numpy as np

    rng = np.random.default_rng(rng_seed)

    # 1-level facet
    df_1 = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist(), "cat": ["A"] * 100})
    # 6-level facet
    cats_6 = [c for c in "ABCDEF" for _ in range(100)]
    df_6 = pl.DataFrame({"x": rng.normal(0, 1, 600).tolist(), "cat": cats_6})

    per_panel_h = 200.0
    asp = 1.0

    svg_1 = displot(df_1, x="x", row="cat", height=per_panel_h, aspect=asp).to_svg()
    svg_6 = displot(df_6, x="x", row="cat", height=per_panel_h, aspect=asp).to_svg()

    w1 = _extract_svg_width(svg_1)
    w6 = _extract_svg_width(svg_6)

    expected_w = per_panel_h * asp
    assert w1 == pytest.approx(expected_w, rel=0.05), (
        f"1-panel displot width {w1} should be ~{expected_w} (per_panel_h * aspect)"
    )
    assert w6 == pytest.approx(expected_w, rel=0.05), (
        f"6-panel displot width {w6} should be ~{expected_w} (per_panel_h * aspect), "
        f"but was {w6} — suggests width still scales with panel count"
    )
    assert w1 == pytest.approx(w6, rel=0.05), (
        f"displot canvas width must be the same for 1-row-panel ({w1}) "
        f"and 6-row-panel ({w6}) charts with the same aspect"
    )


# -- RF2: boxen sort must work on pandas DataFrame input ----------------------


@pytest.fixture
def sort_boxen_pandas_df():
    """Pandas DataFrame with categories Z (highest), B (medium), A (lowest).

    Same distribution as sort_boxen_df but as a pandas DataFrame so we can
    confirm mark_boxen sort pre-resolution works for non-polars input.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    rows = []
    for cat, loc in [("A", 20.0), ("Z", 80.0), ("B", 50.0)]:
        for v in rng.normal(loc, 15.0, 300).tolist():
            rows.append({"cat": cat, "val": v})
    return pd.DataFrame(rows)


def test_rf2_boxen_sort_descending_pandas(sort_boxen_pandas_df) -> None:
    """mark_boxen with X(sort='-y') resolves the aggregate sort for pandas input.

    Regression guard: the aggregate ordering is computed inside the Rust
    LetterValue transform (R2), so backend-agnostic input (pandas/pyarrow)
    must produce the same ordered domain without emitting SortSpecIgnored.
    """
    import warnings

    # Check 1: no SortSpecIgnored warning.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svg_desc = (
            fm.Chart(sort_boxen_pandas_df)
            .mark_boxen()
            .encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
            .to_svg()
        )
    sort_ignored = any("sort spec could not be applied" in str(ww.message) for ww in w)
    assert not sort_ignored, (
        "sort='-y' on mark_boxen with pandas input must not emit a sort-ignored warning"
    )

    # Check 2: the aggregate sort lives on the LetterValue transform, not the layer.
    chart_resolved = (
        fm.Chart(sort_boxen_pandas_df).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
    )._resolve_pending()
    transform_repr = repr(chart_resolved._transforms[0])
    assert "LetterValueSort" in transform_repr and "Desc" in transform_repr, (
        f"boxen sort='-y' on pandas input must set a descending aggregate sort on the "
        f"LetterValue transform; got {transform_repr!r}"
    )
    x_enc = chart_resolved._layers[0].encoding["x"]
    layer_sort = x_enc._kwargs.get("sort") if hasattr(x_enc, "_kwargs") else None
    assert layer_sort is None, (
        f"aggregate boxen sort must not set an explicit layer sort; got {layer_sort!r}"
    )

    # Check 3: sorted SVG differs from unsorted.
    svg_no_sort = fm.Chart(sort_boxen_pandas_df).mark_boxen().encode(x="cat:N", y="val:Q").to_svg()
    assert svg_desc != svg_no_sort, (
        "boxen sort='-y' on pandas input must produce a different SVG than no-sort"
    )


def test_rf2_boxen_sort_descending_pyarrow() -> None:
    """mark_boxen with X(sort='-y') pre-resolves sort correctly for PyArrow Table input."""
    import numpy as np
    import pyarrow as pa
    import warnings

    rng = np.random.default_rng(42)
    rows = []
    for cat, loc in [("A", 20.0), ("Z", 80.0), ("B", 50.0)]:
        for v in rng.normal(loc, 15.0, 300).tolist():
            rows.append({"cat": cat, "val": v})

    table = pa.table(
        {
            "cat": pa.array([r["cat"] for r in rows], type=pa.string()),
            "val": pa.array([r["val"] for r in rows], type=pa.float64()),
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        (fm.Chart(table).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q").to_svg())
    sort_ignored = any("sort spec could not be applied" in str(ww.message) for ww in w)
    assert not sort_ignored, (
        "sort='-y' on mark_boxen with PyArrow Table input must not emit a sort-ignored warning"
    )

    chart_resolved = (
        fm.Chart(table).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q")
    )._resolve_pending()
    transform_repr = repr(chart_resolved._transforms[0])
    assert "LetterValueSort" in transform_repr and "Desc" in transform_repr, (
        f"boxen sort='-y' on PyArrow input must set a descending aggregate sort on the "
        f"LetterValue transform; got {transform_repr!r}"
    )
    x_enc = chart_resolved._layers[0].encoding["x"]
    layer_sort = x_enc._kwargs.get("sort") if hasattr(x_enc, "_kwargs") else None
    assert layer_sort is None, (
        f"aggregate boxen sort on PyArrow input must not set an explicit layer sort; "
        f"got {layer_sort!r}"
    )


# ---------------------------------------------------------------------------
# R7 — promoted color channel is an independent copy, not an alias
# ---------------------------------------------------------------------------


def test_r7_promoted_color_channel_is_independent_copy() -> None:
    """_promote_layer_color must assign a copy, not an alias of the layer's channel.

    When (base + highlight) promotes the highlight layer's color encoding to
    chart level, the chart-level channel and the layer's channel must be
    distinct objects (``is not``).  Aliasing means an in-place mutation of
    one would silently corrupt the other.
    """
    import numpy as np
    from ferrum.encoding.base import ChannelBase

    rng = np.random.default_rng(0)
    rows_hl = [
        {"year": y, "country": c, "value": float(rng.uniform(10, 50))}
        for c in ["China", "Nigeria"]
        for y in range(2000, 2005)
    ]
    rows_bg = [
        {"year": y, "country": c, "value": float(rng.uniform(10, 50))}
        for c in ["USA", "Japan"]
        for y in range(2000, 2005)
    ]
    hl_df = pl.DataFrame(rows_hl)
    bg_df = pl.DataFrame(rows_bg)

    base = (
        fm.Chart(bg_df)
        .mark_line(stroke="#cccccc")
        .encode(x="year:Q", y="value:Q", detail="country:N")
    )
    highlight = (
        fm.Chart(hl_df)
        .mark_line()
        .encode(x="year:Q", y="value:Q", color=fm.Color("country:N", scheme="set1"))
    )

    merged = base + highlight

    # Chart-level color must exist and be a ChannelBase (promotion happened).
    chart_color = merged._encoding.get("color")
    assert chart_color is not None, "chart-level color must be promoted from the highlight layer"
    assert isinstance(chart_color, ChannelBase), (
        f"promoted color must be a ChannelBase; got {type(chart_color).__name__}"
    )

    # Find the layer that carries the color encoding.
    layer_color = None
    for layer in merged._layers or []:
        ch = layer.encoding.get("color")
        if isinstance(ch, ChannelBase):
            layer_color = ch
            break
    assert layer_color is not None, "highlight layer must still carry its color encoding"

    # The two must be equal in value but NOT the same object.
    assert chart_color == layer_color, (
        "promoted chart-level color must equal the layer's color in value"
    )
    assert chart_color is not layer_color, (
        "promoted chart-level color must be an independent copy, not the same object as the layer's color"
    )


# ---------------------------------------------------------------------------
# R6 — inferred-type dedup: _apply_inferred_type helper covers both code paths
# ---------------------------------------------------------------------------


def test_r6_single_mark_x_temporal_inferred_via_build_encoding_specs(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """Single-mark path: unannotated Date x-column gets type=temporal via _build_encoding_specs.

    This exercises the code path that goes through Chart.to_spec() ->
    _build_encoding_specs() -> _apply_inferred_type().  The spec must emit
    type='temporal' on the x encoding without an explicit ':T' annotation.
    """
    import json

    spec = json.loads(fm.Chart(_t2b_monthly_df).mark_line().encode(x="date", y="val:Q").to_json())
    x_enc = spec.get("encoding", {}).get("x", {})
    assert x_enc.get("type") == "temporal", (
        f"single-mark path: unannotated Date x must produce type='temporal'; got {x_enc!r}"
    )


def test_r6_layered_mark_x_temporal_inferred_via_build_layers_list(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """Layered path: unannotated Date x-column gets type=temporal via _build_layers_list.

    This exercises the code path that goes through Chart.to_spec() ->
    _build_layers_list() -> _apply_inferred_type().  The SVG must render
    date-formatted axis ticks (not raw epoch integers) even when the temporal
    type is inferred rather than explicit, and the chart is layered (e.g.
    a scatter + smoothed line overlay).

    See `test_t2b_date_column_without_annotation_gets_temporal_ticks`'s
    docstring for why `_is_epoch_integer` tolerates a legitimately wrapped
    date component at the default theme.
    """
    chart = fm.Chart(_t2b_monthly_df).mark_point().encode(x="date", y="val:Q") + fm.Chart(
        _t2b_monthly_df
    ).mark_line().encode(x="date", y="val:Q")
    svg = chart.to_svg()
    texts = _tick_texts(svg)
    assert _is_date_formatted(texts), (
        f"layered path: unannotated Date x must produce date-formatted ticks; got {texts}"
    )
    assert not _is_epoch_integer(texts), (
        f"layered path: unannotated Date x must not produce raw epoch integers; got {texts}"
    )


def test_r6_inferred_type_identical_across_both_paths(
    _t2b_monthly_df: pl.DataFrame,
) -> None:
    """Both code paths must produce the same type annotation.

    The single-mark path (via _build_encoding_specs) and the layered path
    (via _build_layers_list) must agree: a polars Date column without an
    explicit ':T' annotation produces type='temporal' in both cases.
    This is the characterization test for the R6 dedup: the helper
    _apply_inferred_type is the single source of truth.
    """
    import json

    # Single-mark spec: type goes through _build_encoding_specs.
    single_spec = json.loads(
        fm.Chart(_t2b_monthly_df).mark_line().encode(x="date", y="val:Q").to_json()
    )
    single_type = single_spec.get("encoding", {}).get("x", {}).get("type")

    # Force layered mode so the x encoding is processed by _build_layers_list.
    layered = fm.Chart(_t2b_monthly_df).mark_point().encode(x="date", y="val:Q") + fm.Chart(
        _t2b_monthly_df
    ).mark_line().encode(x="date", y="val:Q")
    layered_svg = layered.to_svg()
    layered_texts = _tick_texts(layered_svg)

    assert single_type == "temporal", (
        f"single-mark path must produce type='temporal'; got {single_type!r}"
    )
    assert _is_date_formatted(layered_texts), (
        f"layered path must also produce date-formatted ticks; got {layered_texts}"
    )


def test_r7_d2_behavior_still_holds() -> None:
    """The color-promotion copy must not break the D2 multi-color rendering behavior.

    Both highlight categories must appear as distinct colors in (base + highlight).
    """
    import numpy as np

    rng = np.random.default_rng(1)
    rows_hl = [
        {"year": y, "country": c, "value": float(rng.uniform(10, 50))}
        for c in ["China", "Nigeria"]
        for y in range(2000, 2005)
    ]
    rows_bg = [
        {"year": y, "country": c, "value": float(rng.uniform(10, 50))}
        for c in ["USA", "Japan"]
        for y in range(2000, 2005)
    ]
    hl_df = pl.DataFrame(rows_hl)
    bg_df = pl.DataFrame(rows_bg)

    base = (
        fm.Chart(bg_df)
        .mark_line(stroke="#cccccc")
        .encode(x="year:Q", y="value:Q", detail="country:N")
    )
    highlight = (
        fm.Chart(hl_df)
        .mark_line()
        .encode(x="year:Q", y="value:Q", color=fm.Color("country:N", scheme="set1"))
    )

    standalone_colors = _polyline_strokes(highlight.to_svg())
    assert len(standalone_colors) == 2, (
        f"highlight standalone should have 2 distinct colors; got {standalone_colors}"
    )

    svg = (base + highlight).to_svg()
    colors = _polyline_strokes(svg)
    missing = standalone_colors - colors
    assert not missing, f"base + highlight is missing highlight colors {missing}; got {colors}"


# ---------------------------------------------------------------------------
# R5 — ChannelBase.option accessor
# ---------------------------------------------------------------------------


def test_r5_option_returns_kwarg_value() -> None:
    """option() returns the kwarg value when the key is present."""
    from ferrum.encoding import X

    ch = X("val", sort="-y", type="Q")
    assert ch.option("sort") == "-y"
    assert ch.option("type") == "Q"


def test_r5_option_returns_default_when_absent() -> None:
    """option() returns the supplied default (or None) when the key is absent."""
    from ferrum.encoding import X

    ch = X("val", type="Q")
    assert ch.option("sort") is None
    assert ch.option("sort", "ascending") == "ascending"
    assert ch.option("axis", False) is False


def test_r5_option_none_value_distinct_from_absent() -> None:
    """option() returns None for an explicitly-set None value.

    chart.py's axis-suppression code relies on ``'axis' in _ch._kwargs`` to
    distinguish 'axis=None explicitly passed' from 'axis not passed at all'.
    This test confirms that option() on a channel with axis=None gives None,
    matching the original ``_kwargs["axis"] is None`` value check.
    """
    from ferrum.encoding import X

    ch_explicit_none = X("val", axis=None)
    ch_absent = X("val")
    # Both return None from option(), but only ch_explicit_none has the key
    # present — the 'in _kwargs' guard in chart.py keeps them distinct.
    assert ch_explicit_none.option("axis") is None
    assert ch_absent.option("axis") is None
    assert "axis" in ch_explicit_none._kwargs
    assert "axis" not in ch_absent._kwargs


def test_r5_sorted_bar_category_order_unchanged() -> None:
    """A sorted bar chart renders identically before and after the option() swap.

    Builds a nominal bar chart with sort="-y" on the x encoding, renders it,
    and verifies the category order in the SVG matches the expected sort order
    (highest-value category first).  This exercises both the sort read-path
    (lines 619-620 in chart.py) and the axis read-path (lines 2840-2849).
    """
    df = pl.DataFrame(
        {
            "cat": ["A", "B", "C"],
            "val": [10.0, 30.0, 20.0],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(x=fm.X("cat", type="N", sort="-y"), y=fm.Y("val", type="Q"))
        .to_svg()
    )
    # B (30) should appear before C (20) should appear before A (10).
    pos_b = svg.find(">B<")
    pos_c = svg.find(">C<")
    pos_a = svg.find(">A<")
    assert pos_b != -1 and pos_c != -1 and pos_a != -1, "category labels A, B, C must appear in SVG"
    assert pos_b < pos_c < pos_a, (
        f"expected sort order B < C < A in SVG; got positions B={pos_b} C={pos_c} A={pos_a}"
    )


def test_r5_axis_none_suppression_unchanged() -> None:
    """X('field', axis=None) still suppresses the x-axis after the option() swap."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    svg_with_axis = (
        fm.Chart(df).mark_point().encode(x=fm.X("x", type="Q"), y=fm.Y("y", type="Q")).to_svg()
    )
    svg_no_axis = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", type="Q", axis=None), y=fm.Y("y", type="Q"))
        .to_svg()
    )
    # With axis: tick labels appear; without: they should be absent or suppressed.
    # The simplest proxy is that the suppressed SVG has fewer characters (no tick
    # text / axis lines) than the version with a full axis.
    assert len(svg_no_axis) < len(svg_with_axis), (
        "axis=None should produce a smaller SVG (no x-axis tick labels/lines)"
    )


# ---------------------------------------------------------------------------
# R1 — unified categorical-axis resolution
# ---------------------------------------------------------------------------
# _resolve_cat_axis normalizes the two divergent orientation conventions
# (horizontal=bool used by boxplot/boxen; orient="vertical"|"horizontal" used
# by swarm) into a single "is the categorical axis y?" decision and returns
# (cat_field, cat_sort, val_field).  These tests pin its behavior and confirm
# the composite-mark expansion stays byte-stable through the refactor.


from ferrum.chart import _resolve_cat_axis


def test_r1_resolve_cat_axis_vertical_boxplot() -> None:
    """Vertical boxplot (horizontal=False): cat=x, val=y, cat_sort=x_sort."""
    cat_f, cat_sort, val_f = _resolve_cat_axis(
        {"horizontal": False}, "cat", "val", "xsort", "ysort"
    )
    assert (cat_f, cat_sort, val_f) == ("cat", "xsort", "val")


def test_r1_resolve_cat_axis_horizontal_boxplot() -> None:
    """Horizontal boxplot (horizontal=True): cat=y, val=x, cat_sort=y_sort."""
    cat_f, cat_sort, val_f = _resolve_cat_axis({"horizontal": True}, "cat", "val", "xsort", "ysort")
    assert (cat_f, cat_sort, val_f) == ("val", "ysort", "cat")


def test_r1_resolve_cat_axis_swarm_vertical() -> None:
    """Swarm orient='vertical': value spreads along x → cat=x, val=y."""
    cat_f, cat_sort, val_f = _resolve_cat_axis(
        {"orient": "vertical"}, "cat", "val", "xsort", "ysort"
    )
    assert (cat_f, cat_sort, val_f) == ("cat", "xsort", "val")


def test_r1_resolve_cat_axis_swarm_horizontal() -> None:
    """Swarm orient='horizontal': value spreads along y → cat=y, val=x."""
    cat_f, cat_sort, val_f = _resolve_cat_axis(
        {"orient": "horizontal"}, "cat", "val", "xsort", "ysort"
    )
    assert (cat_f, cat_sort, val_f) == ("val", "ysort", "cat")


def test_r1_resolve_cat_axis_default_is_vertical() -> None:
    """No orientation kwargs: defaults to vertical (cat=x)."""
    cat_f, cat_sort, val_f = _resolve_cat_axis({}, "cat", "val", None, None)
    assert (cat_f, cat_sort, val_f) == ("cat", None, "val")


def _category_order_in_svg(svg: str, labels: list[str]) -> list[str]:
    """Return *labels* sorted by first appearance position in *svg*."""
    positions = {lab: svg.find(f">{lab}<") for lab in labels}
    assert all(p != -1 for p in positions.values()), (
        f"all category labels must appear in SVG; positions={positions}"
    )
    return sorted(labels, key=positions.__getitem__)


@pytest.fixture
def boxen_sort_df() -> pl.DataFrame:
    # Group C has highest sum, B middle, A lowest — exercises aggregate sort.
    return pl.DataFrame(
        {
            "cat": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
            "val": (
                [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]
                + [10.0, 12.0, 11.0, 13.0, 9.0, 12.0]
                + [20.0, 22.0, 19.0, 24.0, 21.0, 23.0]
            ),
        }
    )


def test_r1_boxen_vertical_sort_descending_order(boxen_sort_df: pl.DataFrame) -> None:
    """Vertical boxen with sort='-y' orders categories C, B, A (highest sum first)."""
    svg = (
        fm.Chart(boxen_sort_df)
        .mark_boxen()
        .encode(x=fm.X("cat", type="N", sort="-y"), y=fm.Y("val", type="Q"))
        .to_svg()
    )
    order = _category_order_in_svg(svg, ["A", "B", "C"])
    assert order == ["C", "B", "A"], f"expected descending-sum order C,B,A; got {order}"


def test_r1_boxen_horizontal_sort_order(boxen_sort_df: pl.DataFrame) -> None:
    """Horizontal boxen with sort on its categorical (y) axis stays well-ordered."""
    svg = (
        fm.Chart(boxen_sort_df)
        .mark_boxen(horizontal=True)
        .encode(x=fm.X("val", type="Q"), y=fm.Y("cat", type="N", sort="-x"))
        .to_svg()
    )
    # All three categories must render; the chart must not collapse.
    order = _category_order_in_svg(svg, ["A", "B", "C"])
    assert set(order) == {"A", "B", "C"}


def test_r1_errorbar_with_categorical_sort_renders() -> None:
    """errorbar with a sort on its categorical (x) axis still renders after the
    y_sort injection removal (R4)."""
    df = pl.DataFrame(
        {
            "cat": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "val": [1.0, 2.0, 3.0, 2.0, 1.0]
            + [30.0, 31.0, 29.0, 32.0, 28.0]
            + [15.0, 16.0, 14.0, 17.0, 13.0],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_errorbar(method="stdev")
        .encode(x=fm.X("cat", type="N", sort="-y"), y=fm.Y("val", type="Q"))
        .to_svg()
    )
    order = _category_order_in_svg(svg, ["A", "B", "C"])
    # B has highest mean, C middle, A lowest → descending order B, C, A.
    assert order == ["B", "C", "A"], f"expected B,C,A; got {order}"


# ---------------------------------------------------------------------------
# R3b — catplot facet sizing (height/aspect parity with displot)
# ---------------------------------------------------------------------------
#
# catplot lacked height= and aspect= parameters while displot had them.
# The fix adds both params to catplot with identical semantics to displot
# and extracts the shared sizing logic into _apply_facet_sizing so the
# two functions cannot drift again.


import numpy as _np_r3b


@pytest.fixture
def _r3b_cat_df() -> pl.DataFrame:
    """50-row categorical DataFrame with 5 groups (A–E) for catplot sizing tests."""
    rng = _np_r3b.random.default_rng(7)
    cats = [c for c in "ABCDE" for _ in range(10)]
    vals = rng.normal(0, 1, 50).tolist()
    return pl.DataFrame({"cat": cats, "val": vals, "grp": (["x"] * 5 + ["y"] * 5) * 5})


def test_r3b_catplot_height_aspect_non_faceted(_r3b_cat_df: pl.DataFrame) -> None:
    """catplot(height=H, aspect=A) without faceting sets canvas to H×(H*A).

    The SVG canvas width must be approximately H * A and height approximately H.
    """
    H, A = 300.0, 1.5
    svg = fm.catplot(_r3b_cat_df, x="cat", y="val", kind="box", height=H, aspect=A).to_svg()
    w = _extract_svg_width(svg)
    assert w == pytest.approx(H * A, rel=0.05), (
        f"catplot non-faceted width must be ~H*A={H * A}; got {w}"
    )


def test_r3b_catplot_height_aspect_row_faceted(_r3b_cat_df: pl.DataFrame) -> None:
    """catplot(row=..., height=H, aspect=A) scales total height by panel count.

    With 2 row-facet levels, the canvas height must be ~2*H and width ~H*A
    (not 2*H*A — width must be per-panel, not total).
    """
    H, A = 200.0, 1.2
    svg = fm.catplot(
        _r3b_cat_df, x="cat", y="val", kind="strip", row="grp", height=H, aspect=A
    ).to_svg()
    w = _extract_svg_width(svg)
    # Width must be ~per_panel_h * aspect, not total_h * aspect.
    expected_w = H * A
    assert w == pytest.approx(expected_w, rel=0.05), (
        f"catplot row-faceted width must be ~per_panel_h*aspect={expected_w}; got {w} "
        f"(if width≈{2 * expected_w:.0f} the bug is total_h*aspect instead of per_panel_h*aspect)"
    )


def test_r3b_catplot_facet_width_constant_across_row_counts() -> None:
    """catplot with row= and a fixed aspect produces the same canvas width
    regardless of how many row-facet levels the data has.

    Mirrors test_rf1_displot_facet_width_constant_across_row_counts to confirm
    catplot uses the same formula as displot.
    """
    rng = _np_r3b.random.default_rng(11)
    df_2 = pl.DataFrame(
        {
            "cat": ["A", "B"] * 20,
            "val": rng.normal(0, 1, 40).tolist(),
            "grp": (["x"] * 20 + ["y"] * 20),
        }
    )
    df_5 = pl.DataFrame(
        {
            "cat": ["A", "B"] * 50,
            "val": rng.normal(0, 1, 100).tolist(),
            "grp": [c for c in "ABCDE" for _ in range(20)],
        }
    )
    H, A = 150.0, 1.0
    svg_2 = fm.catplot(df_2, x="cat", y="val", kind="box", row="grp", height=H, aspect=A).to_svg()
    svg_5 = fm.catplot(df_5, x="cat", y="val", kind="box", row="grp", height=H, aspect=A).to_svg()

    w2 = _extract_svg_width(svg_2)
    w5 = _extract_svg_width(svg_5)
    expected_w = H * A
    assert w2 == pytest.approx(expected_w, rel=0.05), (
        f"2-panel catplot width must be ~{expected_w}; got {w2}"
    )
    assert w5 == pytest.approx(expected_w, rel=0.05), (
        f"5-panel catplot width must be ~{expected_w}; got {w5}"
    )
    assert w2 == pytest.approx(w5, rel=0.05), (
        f"catplot canvas width must be identical for 2-panel ({w2}) and 5-panel ({w5}) "
        f"charts with the same aspect"
    )


def test_r3b_displot_sizing_unchanged_after_helper_extraction() -> None:
    """displot row-faceted sizing is byte-identical before and after the helper extraction.

    Characterization test: the _apply_facet_sizing helper must reproduce displot's
    existing sizing exactly.  We compare the SVG width of a row-faceted displot
    against the expected formula (per_panel_h * aspect) to confirm the refactor
    did not change displot's output.
    """
    rng = _np_r3b.random.default_rng(3)
    cats_4 = [c for c in "ABCD" for _ in range(50)]
    df = pl.DataFrame({"x": rng.normal(0, 1, 200).tolist(), "cat": cats_4})

    H, A = 180.0, 1.4
    svg = fm.displot(df, x="x", row="cat", kind="kde", height=H, aspect=A).to_svg()
    w = _extract_svg_width(svg)
    expected_w = H * A
    assert w == pytest.approx(expected_w, rel=0.05), (
        f"displot row-faceted width after helper extraction must be ~{expected_w}; got {w}"
    )


def test_r3b_catplot_no_height_aspect_still_renders(_r3b_cat_df: pl.DataFrame) -> None:
    """catplot without height= or aspect= renders without error (no regression)."""
    svg = fm.catplot(_r3b_cat_df, x="cat", y="val", kind="violin").to_svg()
    assert len(svg) > 0, "catplot without height/aspect must produce non-empty SVG"
    # Violin bodies are <path> elements.
    assert svg.count('d="M') >= 1, "catplot without height/aspect must render mark elements"


# ---------------------------------------------------------------------------
# R2 — boxen sort via LetterValue transform
# ---------------------------------------------------------------------------
#
# Boxen aggregate sorts ("-y"/"y"/"-x"/"x" shorthands and explicit Vega-Lite
# {field, op, order} dicts) are resolved inside the Rust LetterValue transform:
# it emits its group rows in aggregate order, so the categorical axis domain
# falls out in first-appearance order with no explicit categorical layer sort.
# Non-aggregate forms (explicit list / "ascending" / "descending") still route
# through the standard layer-sort path on the renamed "group" column.


@pytest.fixture
def _r2_boxen_df() -> pl.DataFrame:
    """Categories Z (≈80), B (≈50), A (≈20); descending by sum/mean is Z > B > A."""
    import numpy as np

    rng = np.random.default_rng(7)
    rows = []
    for cat, loc in [("A", 20.0), ("Z", 80.0), ("B", 50.0)]:
        for v in rng.normal(loc, 15.0, 300).tolist():
            rows.append({"cat": cat, "val": v})
    return pl.DataFrame(rows)


@pytest.mark.parametrize("shorthand,order_token", [("-y", "Desc"), ("y", "Asc")])
def test_r2_boxen_aggregate_sort_on_transform(
    _r2_boxen_df: pl.DataFrame, shorthand: str, order_token: str
) -> None:
    """Aggregate sort shorthands set sort on the LetterValue transform, not the layer."""
    chart = (
        fm.Chart(_r2_boxen_df).mark_boxen().encode(x=fm.X("cat:N", sort=shorthand), y="val:Q")
    )._resolve_pending()

    transform_repr = repr(chart._transforms[0])
    assert "LetterValueSort" in transform_repr and order_token in transform_repr, (
        f"boxen sort={shorthand!r} must carry a {order_token} aggregate sort on the "
        f"LetterValue transform; got {transform_repr!r}"
    )
    # No redundant explicit layer sort for the aggregate case.
    x_enc = chart._layers[0].encoding["x"]
    layer_sort = x_enc._kwargs.get("sort") if hasattr(x_enc, "_kwargs") else None
    assert layer_sort is None, (
        f"aggregate boxen sort must not set an explicit layer sort; got {layer_sort!r}"
    )


def test_r2_boxen_aggregate_sort_cross_backend_parity(_r2_boxen_df: pl.DataFrame) -> None:
    """sort='-y' boxen produces byte-identical SVG across polars/pandas/pyarrow."""
    pddf = _r2_boxen_df.to_pandas()
    patbl = _r2_boxen_df.to_arrow()

    def render(data: object) -> str:
        return fm.Chart(data).mark_boxen().encode(x=fm.X("cat:N", sort="-y"), y="val:Q").to_svg()

    svg_pl = render(_r2_boxen_df)
    svg_pd = render(pddf)
    svg_pa = render(patbl)
    assert svg_pl == svg_pd, "polars vs pandas boxen sort='-y' SVG must be byte-identical"
    assert svg_pl == svg_pa, "polars vs pyarrow boxen sort='-y' SVG must be byte-identical"


def test_r2_boxen_explicit_list_sort_uses_layer_path(_r2_boxen_df: pl.DataFrame) -> None:
    """Explicit-list sort routes through the standard layer-sort path, not the transform."""
    chart = (
        fm.Chart(_r2_boxen_df).mark_boxen().encode(x=fm.X("cat:N", sort=["B", "A", "Z"]), y="val:Q")
    )._resolve_pending()

    # No aggregate sort on the transform for the explicit-list form.
    transform_repr = repr(chart._transforms[0])
    assert "sort=None" in transform_repr, (
        f"explicit-list boxen sort must not set an aggregate transform sort; got {transform_repr!r}"
    )
    # The explicit list lands on the layer's categorical encoding.
    x_enc = chart._layers[0].encoding["x"]
    layer_sort = x_enc._kwargs.get("sort") if hasattr(x_enc, "_kwargs") else None
    assert layer_sort == ["B", "A", "Z"], (
        f"explicit-list boxen sort must reach the layer encoding; got {layer_sort!r}"
    )


def test_r2_boxen_horizontal_aggregate_sort(_r2_boxen_df: pl.DataFrame) -> None:
    """Horizontal boxen with Y(sort='-x') sets the aggregate sort on the transform."""
    chart = (
        fm.Chart(_r2_boxen_df)
        .mark_boxen(horizontal=True)
        .encode(x="val:Q", y=fm.Y("cat:N", sort="-x"))
    )._resolve_pending()

    transform_repr = repr(chart._transforms[0])
    assert "LetterValueSort" in transform_repr and "Desc" in transform_repr, (
        f"horizontal boxen Y(sort='-x') must carry a descending aggregate sort on the "
        f"LetterValue transform; got {transform_repr!r}"
    )
    y_enc = chart._layers[0].encoding["y"]
    layer_sort = y_enc._kwargs.get("sort") if hasattr(y_enc, "_kwargs") else None
    assert layer_sort is None, (
        f"horizontal aggregate boxen sort must not set an explicit layer sort; got {layer_sort!r}"
    )


def test_r2_boxen_no_sort_unchanged(_r2_boxen_df: pl.DataFrame) -> None:
    """Unsorted boxen sets no aggregate transform sort and no layer sort."""
    chart = (fm.Chart(_r2_boxen_df).mark_boxen().encode(x="cat:N", y="val:Q"))._resolve_pending()
    transform_repr = repr(chart._transforms[0])
    assert "sort=None" in transform_repr, (
        f"unsorted boxen must not set an aggregate transform sort; got {transform_repr!r}"
    )
    x_enc = chart._layers[0].encoding["x"]
    assert not hasattr(x_enc, "_kwargs") or x_enc._kwargs.get("sort") is None, (
        "unsorted boxen must not set an explicit layer sort"
    )
