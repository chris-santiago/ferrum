"""Regression tests for defect D4 — integer and nominal columns silently
blanked or hard-rejected on channels where float/ordinal columns work.

Five observable failure modes under the bug:

1. Integer-keyed categorical heatmap (``mark_rect`` with integer columns
   declared ``:O`` or ``:N``) renders zero cells instead of rows×cols.
2. String-keyed heatmap with ``:N`` — already works for ``:O``; both
   must produce all cells after the fix.
3. ``mark_bar`` with a nominal string ``y`` raises
   ``ValueError: scale: column '...' has unsupported dtype: Utf8``
   instead of rendering one bar per category.
4. Integer column with no type suffix is treated as quantitative
   (matching Altair behavior) — already works; test asserts it
   explicitly so a regression is caught.
5. ``Stack`` raises ``ValueError: Stack: y must be Float64 or UInt64; got
   Int64`` when the measure column has dtype Int64.

Root cause (Python/Rust boundary): the dtype coercion layer in
``src/ferrum/_coerce.py`` does not promote integer storage to Float64
before handing it to the Rust stack transform, and the categorical scale
resolver rejects Utf8/Int64 on positional channels other than X.

Tests assert OBSERVABLE rendered output (SVG element counts, render
succeeds without error).  They will FAIL until the fix lands.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _colored_rects(svg: str) -> list[str]:
    """Return all self-closing ``<rect>`` elements that carry a non-structural
    fill color.

    Excludes the chart background (#faf7f2), the plot-area frame (no fill),
    and the color-bar gradient rect (fill="url(...)").

    Returns
    -------
    list[str]
        The raw ``<rect … />`` strings for colored mark cells.
    """
    structural_fills = {"faf7f2"}
    rects = re.findall(r"<rect[^>]*/?>", svg)
    result = []
    for r in rects:
        # skip gradient fills (colorbar)
        if "url(" in r:
            continue
        fill_match = re.search(r'fill="#([0-9a-fA-F]{6})"', r)
        if fill_match and fill_match.group(1).lower() not in structural_fills:
            result.append(r)
    return result


def _colored_bar_rects(svg: str) -> list[str]:
    """Return all positioned ``<rect>`` elements that represent bars.

    Bar rects carry x="..." and width="..." attributes and a non-structural
    fill.  Returns a list of raw element strings.
    """
    structural_fills = {"faf7f2"}
    rects = re.findall(r"<rect[^>]*/?>", svg)
    result = []
    for r in rects:
        if 'x="' not in r and 'y="' not in r:
            continue
        # skip gradient fills (colorbar)
        if "url(" in r:
            continue
        # require a non-structural fill hex color — rects without any fill
        # attribute are plot-area frames, not bar marks
        fill_match = re.search(r'fill="#([0-9a-fA-F]{6})"', r)
        if not fill_match or fill_match.group(1).lower() in structural_fills:
            continue
        # require width/height to exist (real bars have both)
        if 'width="' in r and 'height="' in r:
            result.append(r)
    return result


# ---------------------------------------------------------------------------
# D4-A — Integer-keyed categorical heatmap: must render all cells
# ---------------------------------------------------------------------------


@pytest.fixture
def int_heatmap_3x3() -> pl.DataFrame:
    """3×3 heatmap with integer row/col keys and float values."""
    return pl.DataFrame(
        {
            "row_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "col_id": [10, 20, 30, 10, 20, 30, 10, 20, 30],
            "val": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )


def test_integer_keyed_heatmap_ordinal_renders_all_cells(
    int_heatmap_3x3: pl.DataFrame,
) -> None:
    """``mark_rect`` with integer columns declared ``:O`` must render all 9
    cells of a 3×3 grid.

    Under the bug, the integer columns are not recognized as an ordinal/
    categorical domain and the cell rects are silently omitted — the SVG
    contains a color-bar and axes but zero colored ``<rect>`` elements.

    The assertion checks that the count of colored cell rects equals
    rows × cols = 9.

    Bug indicator: 0 colored rects instead of 9.
    """
    svg = (
        fm.Chart(int_heatmap_3x3)
        .mark_rect()
        .encode(x="col_id:O", y="row_id:O", color="val:Q")
        .show_svg()
    )
    cells = _colored_rects(svg)
    assert len(cells) == 9, (
        f"Expected 9 colored cell rects for a 3×3 integer-keyed heatmap "
        f"(col_id:O / row_id:O) but got {len(cells)}.  The integer columns "
        f"are not being recognized as ordinal categories — cells are silently "
        f"dropped."
    )


def test_integer_keyed_heatmap_nominal_renders_all_cells(
    int_heatmap_3x3: pl.DataFrame,
) -> None:
    """``mark_rect`` with integer columns declared ``:N`` must render all 9
    cells of a 3×3 grid.

    Same as the ``:O`` case but with the nominal type suffix.  Both ``:O``
    and ``:N`` must activate categorical scale resolution for integer columns.

    Bug indicator: 0 colored rects instead of 9.
    """
    svg = (
        fm.Chart(int_heatmap_3x3)
        .mark_rect()
        .encode(x="col_id:N", y="row_id:N", color="val:Q")
        .show_svg()
    )
    cells = _colored_rects(svg)
    assert len(cells) == 9, (
        f"Expected 9 colored cell rects for a 3×3 integer-keyed heatmap "
        f"(col_id:N / row_id:N) but got {len(cells)}.  The integer columns "
        f"are not being recognized as nominal categories — cells are silently "
        f"dropped."
    )


# ---------------------------------------------------------------------------
# D4-B — String-keyed heatmap: :O and :N both render all cells
# ---------------------------------------------------------------------------


@pytest.fixture
def str_heatmap_3x3() -> pl.DataFrame:
    """3×3 heatmap with string row/col keys and float values."""
    return pl.DataFrame(
        {
            "row": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "col": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
            "val": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )


def test_string_keyed_heatmap_ordinal_renders_all_cells(
    str_heatmap_3x3: pl.DataFrame,
) -> None:
    """``mark_rect`` with string columns declared ``:O`` must render all 9
    cells.

    This case already works (control).  The test is present so any regression
    is caught alongside the integer-column fix, and to ensure the fix does not
    break the working string path.

    Expected: 9 colored cell rects.
    """
    svg = (
        fm.Chart(str_heatmap_3x3).mark_rect().encode(x="col:O", y="row:O", color="val:Q").show_svg()
    )
    cells = _colored_rects(svg)
    assert len(cells) == 9, (
        f"Expected 9 colored cell rects for a 3×3 string-keyed heatmap "
        f"(col:O / row:O) but got {len(cells)}."
    )


def test_string_keyed_heatmap_nominal_renders_all_cells(
    str_heatmap_3x3: pl.DataFrame,
) -> None:
    """``mark_rect`` with string columns declared ``:N`` must render all 9
    cells.

    Under the bug, the Rust scale resolver rejects ``Utf8`` columns on the
    Y positional channel with:
    ``ValueError: scale: column 'row' has unsupported dtype: Utf8``

    This test asserts successful rendering without error and that all 9
    cells appear in the SVG.

    Bug indicator: ``ValueError`` raised, or 0 cells if it silently blanks.
    """
    svg = (
        fm.Chart(str_heatmap_3x3).mark_rect().encode(x="col:N", y="row:N", color="val:Q").show_svg()
    )
    cells = _colored_rects(svg)
    assert len(cells) == 9, (
        f"Expected 9 colored cell rects for a 3×3 string-keyed heatmap "
        f"(col:N / row:N) but got {len(cells)}.  String columns declared "
        f"as nominal (``:N``) are being rejected or silently dropped on the "
        f"y channel."
    )


# ---------------------------------------------------------------------------
# D4-C — mark_bar with nominal y: one bar per category
# ---------------------------------------------------------------------------


@pytest.fixture
def categorical_bar_df() -> pl.DataFrame:
    """Four-row dataset for a horizontal categorical bar chart."""
    return pl.DataFrame(
        {
            "category": ["alpha", "beta", "gamma", "delta"],
            "value": [10.0, 25.0, 15.0, 30.0],
        }
    )


def test_mark_bar_nominal_y_renders_one_bar_per_category(
    categorical_bar_df: pl.DataFrame,
) -> None:
    """``mark_bar`` with a string ``y:N`` and numeric ``x`` (horizontal bars)
    must render one bar per category without raising.

    Under the bug, using a string column on the y-axis raises:
    ``ValueError: scale: column 'category' has unsupported dtype: Utf8``

    This test asserts:
    1. The render completes without an exception.
    2. Exactly 4 bars appear in the SVG (one per category).

    Bug indicator: ``ValueError`` raised.
    """
    svg = fm.Chart(categorical_bar_df).mark_bar().encode(y="category:N", x="value:Q").show_svg()
    bars = _colored_bar_rects(svg)
    assert len(bars) == 4, (
        f"Expected 4 bars (one per category) for a horizontal bar chart with "
        f"string y:N but got {len(bars)}.  Either the render raised or bars "
        f"were silently dropped."
    )


def test_mark_bar_ordinal_y_renders_one_bar_per_category(
    categorical_bar_df: pl.DataFrame,
) -> None:
    """``mark_bar`` with a string ``y:O`` and numeric ``x`` must render one
    bar per category.

    Same as the ``:N`` case; both ordinal and nominal suffixes must be accepted
    on a string y-axis.

    Bug indicator: ``ValueError`` raised.
    """
    svg = fm.Chart(categorical_bar_df).mark_bar().encode(y="category:O", x="value:Q").show_svg()
    bars = _colored_bar_rects(svg)
    assert len(bars) == 4, (
        f"Expected 4 bars (one per category) for a horizontal bar chart with "
        f"string y:O but got {len(bars)}."
    )


# ---------------------------------------------------------------------------
# D4-D — Integer column as quantitative: default and explicit :Q
# ---------------------------------------------------------------------------


@pytest.fixture
def int_scatter_df() -> pl.DataFrame:
    """Five-row scatter dataset with integer x and y columns."""
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 15, 25, 5],
        }
    )


def test_integer_column_no_suffix_treated_as_quantitative(
    int_scatter_df: pl.DataFrame,
) -> None:
    """An integer column with no type suffix must be treated as quantitative
    (continuous) — matching Altair behavior.

    All 5 rows must render as point marks on a continuous numeric scale.
    The SVG must contain exactly 5 circle elements.

    This assertion exists to lock in the default inference behavior so that
    it cannot be silently changed by the D4 dtype-normalization fix.
    """
    svg = fm.Chart(int_scatter_df).mark_point().encode(x="x", y="y").show_svg()
    circles = re.findall(r"<circle[^>]*/?>", svg)
    assert len(circles) == 5, (
        f"Expected 5 circle marks for a 5-row scatter with integer columns "
        f"and no type suffix, but got {len(circles)}.  Integer columns with "
        f"no suffix must default to quantitative (continuous) treatment."
    )


def test_integer_column_explicit_Q_treated_as_quantitative(
    int_scatter_df: pl.DataFrame,
) -> None:
    """An integer column declared ``:Q`` must render as continuous numeric —
    same result as a float column with ``:Q``.

    All 5 rows must appear as circle marks.
    """
    svg = fm.Chart(int_scatter_df).mark_point().encode(x="x:Q", y="y:Q").show_svg()
    circles = re.findall(r"<circle[^>]*/?>", svg)
    assert len(circles) == 5, (
        f"Expected 5 circle marks for a 5-row scatter with integer columns "
        f"declared as :Q but got {len(circles)}."
    )


def test_integer_no_suffix_equals_explicit_Q(
    int_scatter_df: pl.DataFrame,
) -> None:
    """An integer column with no type suffix must produce identical SVG to the
    same column declared explicitly as ``:Q``.

    This confirms that the default type inference for integer storage is
    quantitative (continuous), matching Altair.
    """
    svg_no_suffix = fm.Chart(int_scatter_df).mark_point().encode(x="x", y="y").show_svg()
    svg_explicit_q = fm.Chart(int_scatter_df).mark_point().encode(x="x:Q", y="y:Q").show_svg()
    assert svg_no_suffix == svg_explicit_q, (
        "Integer columns with no type suffix produced different SVG from the "
        "same columns declared as :Q.  The default type inference for integers "
        "must be quantitative (continuous)."
    )


# ---------------------------------------------------------------------------
# D4-E — Stack accepts integer measure columns
# ---------------------------------------------------------------------------


@pytest.fixture
def int_stacked_bar_df() -> pl.DataFrame:
    """Stacked bar dataset with an integer measure column."""
    return pl.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "segment": ["s1", "s2", "s1", "s2"],
            "count": [10, 20, 15, 25],  # Int64
        }
    )


def test_stack_accepts_integer_measure_without_error(
    int_stacked_bar_df: pl.DataFrame,
) -> None:
    """``Stack`` must accept an Int64 measure column without raising.

    Under the bug, stacking an integer column raises:
    ``ValueError: Stack: y must be Float64 or UInt64; got Int64``

    This test asserts the render completes and that stacked bar segments
    appear in the SVG.  There are 2 categories × 2 segments = 4 stacked
    segment rects expected.

    Bug indicator: ``ValueError`` raised.
    """
    svg = (
        fm.Chart(int_stacked_bar_df)
        .mark_bar(position=fm.Stack())
        .encode(x="category:N", y="count:Q", color="segment:N")
        .show_svg()
    )
    bars = _colored_bar_rects(svg)
    assert len(bars) == 4, (
        f"Expected 4 stacked bar segments (2 categories × 2 segments) for a "
        f"stacked bar chart with an Int64 measure column but got {len(bars)}.  "
        f"Stack must accept integer measure columns without requiring an "
        f"explicit cast to Float64."
    )


def test_stack_integer_measure_matches_float_measure(
    int_stacked_bar_df: pl.DataFrame,
) -> None:
    """Stack on an Int64 measure must produce the same number of bar segments
    as an identical chart with a Float64 measure.

    This confirms that the fix promotes integers to Float64 transparently
    rather than changing the visual output.
    """
    df_float = int_stacked_bar_df.with_columns(pl.col("count").cast(pl.Float64))

    svg_int = (
        fm.Chart(int_stacked_bar_df)
        .mark_bar(position=fm.Stack())
        .encode(x="category:N", y="count:Q", color="segment:N")
        .show_svg()
    )
    svg_float = (
        fm.Chart(df_float)
        .mark_bar(position=fm.Stack())
        .encode(x="category:N", y="count:Q", color="segment:N")
        .show_svg()
    )

    bars_int = _colored_bar_rects(svg_int)
    bars_float = _colored_bar_rects(svg_float)

    assert len(bars_int) == len(bars_float), (
        f"Stacked bar with Int64 measure produced {len(bars_int)} segments "
        f"but the same chart with Float64 produced {len(bars_float)}.  "
        f"Integer measure columns must stack identically to float columns."
    )


# ---------------------------------------------------------------------------
# D4-F — Dual-use column regression: same integer column on :O AND :Q
# ---------------------------------------------------------------------------


@pytest.fixture
def dual_use_df() -> pl.DataFrame:
    """Three-row dataset where 'yr' (integer) is used as both ordinal and
    quantitative in the same chart spec.

    This exercises the regression introduced by the reverted Python-side cast:
    a column-global Utf8 cast would convert 'yr' to string for the entire Arrow
    table, causing the :Q (quantitative) channel to fail with
    ``unsupported dtype: Utf8`` or ``expected numeric column ... got Utf8``.
    """
    return pl.DataFrame(
        {
            "yr": [2000, 2001, 2002],
            "amt": [100.0, 150.0, 130.0],
        }
    )


def test_dual_use_integer_column_ordinal_and_quantitative_point(
    dual_use_df: pl.DataFrame,
) -> None:
    """An integer column used as both ``:O`` (ordinal, categorical axis) and
    ``:Q`` (quantitative, size encoding) in the same chart must render without
    error and produce at least one mark per data row.

    The SVG includes 3 data circles plus size-legend circles, so the total
    count is >= 3.  The primary assertion is that the render does not raise —
    a column-global Utf8 cast would cause the ``:Q`` size channel to receive a
    string column and raise ``expected numeric column ... got Utf8``.  This test
    will FAIL under that regression.
    """
    svg = fm.Chart(dual_use_df).mark_point().encode(x="yr:O", y="amt:Q", size="yr:Q").show_svg()
    circles = re.findall(r"<circle[^>]*/?>", svg)
    assert len(circles) >= 3, (
        f"Expected at least 3 circle marks (one per data row) for a chart using "
        f"'yr' as both :O (x) and :Q (size) but got {len(circles)}.  A column-global "
        f"Utf8 cast would fail the :Q channel — this test detects that regression."
    )


def test_dual_use_integer_column_ordinal_and_quantitative_bar(
    dual_use_df: pl.DataFrame,
) -> None:
    """An integer column used as both ``:O`` (ordinal x-axis) and ``:Q``
    (quantitative color) in a bar chart must render without error and produce
    one bar per row.

    ``mark_bar`` with an ordinal x and a quantitative color channel exercises
    a different rendering path than ``mark_point``.  A column-global Utf8 cast
    would cause the color channel to receive a string column and raise
    ``expected numeric column ... got Utf8``.

    This test will FAIL under a reintroduced column-global cast.
    """
    svg = fm.Chart(dual_use_df).mark_bar().encode(x="yr:O", y="amt:Q", color="yr:Q").show_svg()
    bars = _colored_bar_rects(svg)
    assert len(bars) == 3, (
        f"Expected 3 bars (one per row) for a chart using 'yr' as both :O (x) "
        f"and :Q (color) but got {len(bars)}.  A column-global Utf8 cast would "
        f"fail the :Q color channel — this test detects that regression."
    )
