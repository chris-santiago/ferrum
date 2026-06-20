"""Regression tests for defect D2 — faceting grammar.

Four sub-defects, each with a distinct observable failure mode:

  2a — Two-way faceting drops the row dimension.
       `chart.facet(row="a", col="b")` on a 3×3 categorical cross renders only
       3 column-headers and 3 panels (one per col-value) instead of 9 panels
       arranged in a true 3×3 grid with both row and column headers.
       Root cause: `prepare.rs:partition_batch_by_field` partitions on
       `fspec.field` only (the column field), ignoring `fspec.row`.

  2b — Faceted statistical/composite marks render blank.
       `Chart(df).mark_density(...).encode(x=...).facet(col="group")` with 2
       groups having very different value ranges should render one KDE per panel
       whose peak x-position reflects that panel's data.  Instead, both panels
       show the same KDE curve (the transform ran on the merged dataset before
       partitioning).
       Root cause: `prepare.rs` applies the facet-before-transform path using
       a single partition field (`fspec.field`); the KDE peak pixel position is
       identical across panels rather than tracking each partition's data
       distribution.

  2c — Faceted layered chart with distinct DataFrames loses layers.
       When two `Chart` objects backed by different DataFrames are combined with
       `+` and then faceted, the second layer's rows have `null` in the facet
       column (they come from a DataFrame that has no facet column or whose
       column was renamed during the `+` merge).  Partitioning by the facet
       column drops those null-group rows, so the second layer's marks
       disappear entirely in the faceted output.
       Root cause: the `+` merge produces a wide DataFrame that renames the
       second layer's facet-field column (e.g. `group__rhs_...`).  Rows from
       the second layer therefore have `group = null` and are excluded from
       every facet partition.

  2d — Facet scale resolution (shared vs. independent).
       A faceted `Chart` should support `chart.facet(...).share_scale(y=...)`
       to switch per-channel scale resolution.  With `"independent"`, each
       panel uses its own y-domain (panels with different y-ranges show
       different tick labels); with `"shared"`, all panels lock to the same
       domain.  `Chart` has no `share_scale` method today — the attribute is
       only on `_ChartLike` (the composition base class), not on `Chart`.

All tests assert on OBSERVABLE rendered SVG geometry: mark element counts,
strip-title text elements, axis tick-label values, and peak-density x-pixel
positions.  Tests fail for the RIGHT reason today (wrong panel count, shared
KDE, dropped layer marks, missing API) and must pass after the fix lands.

Root-cause file references:
  Python (2a/2b/2c): src/ferrum/chart.py — `_build_facet_dict()` (grid mode
    serialisation) and `__add__` / layer-merge logic.
  Python (2d): src/ferrum/chart.py — Chart class missing `share_scale`.
  Rust (2a/2b): crates/ferrum-core/src/render/prepare.rs —
    `partition_batch_by_field` (line ~1340) uses only `fspec.field` for the
    partition key; `fspec.row` is stored on `FacetSpec` (layout/facet.rs) but
    never used during partitioning or facet_group counting.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG extraction helpers
# ---------------------------------------------------------------------------


def _panel_strip_labels(svg: str) -> list[str]:
    """Return the text content of all strip/header labels in the SVG.

    Facet strip titles appear as ``<text ...>value</text>`` elements.  We
    collect everything that is a short string (≤20 chars) and appears more
    than once OR matches one of the test category names.  The caller filters
    to a specific set of expected category values.
    """
    return [t.strip() for t in re.findall(r">([^<]{1,20})</text>", svg) if t.strip()]


def _circles(svg: str) -> int:
    """Count ``<circle`` elements in the SVG (mark_point marks)."""
    return len(re.findall(r"<circle", svg))


def _data_circle_xy(svg: str) -> list[tuple[float, float]]:
    """Return (cx, cy) for every data-mark ``<circle>`` (theme radius ≈ 3.x).

    Legend swatches use r="4"; data marks use the theme point radius (~3.385),
    so the radius discriminates marks from legend swatches.
    """
    out: list[tuple[float, float]] = []
    for cx, cy, r in re.findall(r'<circle cx="([0-9.]+)" cy="([0-9.]+)" r="([0-9.]+)"', svg):
        if abs(float(r) - 4.0) > 1e-6:
            out.append((float(cx), float(cy)))
    return out


def _paths(svg: str) -> list[str]:
    """Return all ``<path d="...">`` d-string values from the SVG."""
    return re.findall(r'<path d="([^"]+)"', svg)


def _y_tick_labels(svg: str) -> list[str]:
    """Return numeric y-axis tick-label strings from the SVG.

    Y-axis labels use ``text-anchor="end"`` (right-aligned to the left of the
    axis line).  Returns only values that parse as floats.
    """
    labels = []
    for attrs, content in re.findall(r"<text\s([^>]+)>([^<]+)</text>", svg):
        if 'text-anchor="end"' in attrs:
            try:
                float(content.strip())
                labels.append(content.strip())
            except ValueError:
                pass
    return labels


def _path_peak_x(path_d: str) -> float | None:
    """Return the x-coordinate of the y-minimum point in the path.

    In an SVG KDE area, the peak density corresponds to the point with the
    smallest y-pixel value (SVG y increases downward).  Returns ``None`` when
    the path has fewer than 2 coordinate pairs.
    """
    coords = re.findall(r"[ML]\s*([\d.]+)\s+([\d.]+)", path_d)
    if len(coords) < 2:
        return None
    pairs = [(float(x), float(y)) for x, y in coords]
    peak = min(pairs, key=lambda p: p[1])
    return peak[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_3x3_df() -> pl.DataFrame:
    """3×3 categorical cross-product dataset for two-way faceting.

    Each (row_cat, col_cat) cell contains 3 data points.  A correctly rendered
    two-way facet should produce 9 panels (3 row-categories × 3 col-categories),
    each with 3 circles.
    """
    rows = []
    for row_val in ["r1", "r2", "r3"]:
        for col_val in ["c1", "c2", "c3"]:
            for i in range(3):
                rows.append({"row_cat": row_val, "col_cat": col_val, "x": float(i), "y": float(i)})
    return pl.DataFrame(rows)


@pytest.fixture
def bimodal_facet_df() -> pl.DataFrame:
    """Dataset for faceted density: two groups with non-overlapping value ranges.

    Group "low" has values 0–4; group "high" has values 10–14.  After a
    correct facet-before-transform, each panel's KDE peak x-pixel should track
    its group's center (≈2 for "low", ≈12 for "high").  Both values map to
    very different pixel columns on a shared x-axis.
    """
    rows: list[dict] = []
    for i in range(30):
        rows.append({"group": "low", "val": float(i % 5)})  # range 0–4
    for i in range(30):
        rows.append({"group": "high", "val": float(10 + i % 5)})  # range 10–14
    return pl.DataFrame(rows)


@pytest.fixture
def layered_facet_df() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two DataFrames for a layered+faceted chart test.

    ``df_a``: four rows with a 'group' column (the facet column).  Two rows per
    group.  Layer 1 renders these as points (x, y).

    ``df_b``: two rows, one per group, with different 'highlight_y' values.
    Layer 2 renders these as points at distinct y values per group.  After the
    ``+`` merge, df_b rows have ``group = null`` in the merged DataFrame, so
    they are dropped from every facet partition.
    """
    df_a = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, 4.0, 5.0],
        }
    )
    df_b = pl.DataFrame(
        {
            "group": ["A", "B"],
            "highlight_y": [1.5, 4.5],
            "highlight_x": [1.5, 1.5],
        }
    )
    return df_a, df_b


@pytest.fixture
def scale_resolution_df() -> pl.DataFrame:
    """Dataset for scale-resolution tests: two groups with very different y ranges.

    Group "low" has y ≈ 0–2; group "high" has y ≈ 100–102.  With shared y,
    both panels show ticks spanning 0–102.  With independent y, panel "low"
    shows ticks near 0–2 and panel "high" shows ticks near 100–102.
    """
    rows: list[dict] = []
    for i in range(3):
        rows.append({"group": "low", "x": float(i), "y": float(i)})
    for i in range(3):
        rows.append({"group": "high", "x": float(i), "y": 100.0 + float(i)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2a — Two-way faceting: row dimension must not be dropped
# ---------------------------------------------------------------------------


def test_two_way_facet_produces_nine_panels(grid_3x3_df: pl.DataFrame) -> None:
    """facet(row=, col=) on a 3×3 categorical cross must render 9 populated panels.

    nrows and ncols are intentionally NOT passed so this test exercises
    cardinality inference from the data.  The renderer must infer nrows=3 and
    ncols=3 from the distinct values in row_cat and col_cat respectively.

    Each (row_cat, col_cat) cell contains 3 data points → 27 circles total.
    Under the bug, nrows/ncols defaulted to 1×1, dropping all but one panel.

    Observable check:
      - Circle count must be 27 (all data present in all panels).
      - All three row labels (r1, r2, r3) appear as strip titles.
      - All three col labels (c1, c2, c3) appear as strip titles.
    """
    svg = (
        fm.Chart(grid_3x3_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .facet(row="row_cat", col="col_cat")
        .to_svg()
    )

    circle_count = _circles(svg)
    assert circle_count == 27, (
        f"Expected 27 circles (9 panels × 3 points each) but got {circle_count}.  "
        "Grid cardinality may not have been inferred correctly from the data — "
        "check that _build_facet_dict infers nrows/ncols via n_unique() when not supplied."
    )

    labels = _panel_strip_labels(svg)

    for col_label in ("c1", "c2", "c3"):
        assert col_label in labels, (
            f"Column strip label {col_label!r} not found in SVG text elements.  "
            f"Labels found: {[l for l in labels if l in ('r1', 'r2', 'r3', 'c1', 'c2', 'c3')]}"
        )

    for row_label in ("r1", "r2", "r3"):
        assert row_label in labels, (
            f"Row strip label {row_label!r} not found in SVG text elements — the row "
            "dimension is being dropped and only column headers are rendered.  "
            f"Labels found: {[l for l in labels if l in ('r1', 'r2', 'r3', 'c1', 'c2', 'c3')]}.  "
            "Root cause: prepare.rs partition_batch_by_field uses only fspec.field "
            "(column), ignoring fspec.row."
        )


def test_two_way_facet_explicit_nrows_ncols(grid_3x3_df: pl.DataFrame) -> None:
    """facet(row=, col=, nrows=3, ncols=3) with explicit geometry must also produce 9 panels.

    This covers the path where the user supplies explicit nrows/ncols.  The
    explicit values must take precedence over inference and produce the same
    27-circle, 9-panel result.
    """
    svg = (
        fm.Chart(grid_3x3_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .facet(row="row_cat", col="col_cat", nrows=3, ncols=3)
        .to_svg()
    )

    circle_count = _circles(svg)
    assert circle_count == 27, (
        f"Expected 27 circles with explicit nrows=3, ncols=3 but got {circle_count}."
    )

    labels = _panel_strip_labels(svg)
    for label in ("r1", "r2", "r3", "c1", "c2", "c3"):
        assert label in labels, (
            f"Strip label {label!r} missing from SVG with explicit nrows/ncols.  "
            f"Labels found: {[l for l in labels if l in ('r1', 'r2', 'r3', 'c1', 'c2', 'c3')]}"
        )


def test_two_way_facet_row_labels_absent_from_col_only_facet(
    grid_3x3_df: pl.DataFrame,
) -> None:
    """Row strip labels (r1, r2, r3) must appear in two-way facet but NOT in col-only.

    When the row dimension is correctly rendered, the SVG contains strip-title
    text elements for each row-category value ('r1', 'r2', 'r3').  The col-only
    facet renders only column headers; row values are absent from its text
    elements.

    Under the current bug, both the two-way facet and the col-only facet produce
    the same result: only column strip titles appear, because the row dimension
    is silently dropped.  Both SVGs lack row labels ('r1', 'r2', 'r3').

    This test asserts: row labels ARE present in the two-way SVG.  It fails
    today because row labels are absent from the two-way output.
    """
    svg_grid = (
        fm.Chart(grid_3x3_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .facet(row="row_cat", col="col_cat")
        .to_svg()
    )
    svg_col_only = (
        fm.Chart(grid_3x3_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .facet(col="col_cat", ncols=3)
        .to_svg()
    )

    labels_grid = _panel_strip_labels(svg_grid)
    labels_col = _panel_strip_labels(svg_col_only)

    # Row labels must appear in the two-way facet SVG.
    for row_label in ("r1", "r2", "r3"):
        assert row_label in labels_grid, (
            f"Row label {row_label!r} is absent from the two-way facet SVG — the row "
            "dimension is being dropped.  Only column headers are rendered.  "
            f"Labels in two-way SVG: "
            f"{[l for l in labels_grid if l in ('r1', 'r2', 'r3', 'c1', 'c2', 'c3')]}"
        )

    # Row labels must NOT appear in the col-only facet SVG (guard against
    # accidentally including row_cat data values in axis labels).
    for row_label in ("r1", "r2", "r3"):
        assert row_label not in labels_col, (
            f"Row label {row_label!r} unexpectedly appears in the col-only facet SVG.  "
            "The fixture data should not cause row-category values to appear as "
            f"axis tick labels.  Labels: {labels_col}"
        )


# ---------------------------------------------------------------------------
# 2b — Faceted statistical marks: per-panel transform execution
# ---------------------------------------------------------------------------


def test_faceted_density_peak_differs_between_panels(
    bimodal_facet_df: pl.DataFrame,
) -> None:
    """Faceted mark_density must compute the KDE within each panel's partition.

    The bimodal_facet_df fixture has:
      - group "low":  values 0–4   (center ≈ 2)
      - group "high": values 10–14 (center ≈ 12)

    A correctly partitioned KDE places the peak density at the center of each
    group's value range.  On a shared x-axis spanning 0–14, group "low" peaks
    near the left side of the plot and group "high" peaks near the right side —
    the peak x-pixels must differ.

    Under the bug, the KDE runs on the merged dataset (0–14) before
    partitioning.  Both panels show the same curve covering 0–14, with identical
    peak x-pixel positions.

    Observable check: the KDE path peak x-position must DIFFER between the two
    panels (by at least 100 pixels on a 640-pixel canvas).
    """
    svg = fm.Chart(bimodal_facet_df).mark_density().encode(x="val:Q").facet(col="group").to_svg()

    panel_paths = _paths(svg)
    assert len(panel_paths) >= 2, (
        f"Expected at least 2 KDE paths (one per panel) but found {len(panel_paths)}.  "
        "Panels may be blank (transforms not run per-partition) or all paths are absent."
    )

    # Each KDE area has many path points; take the first two distinct-looking paths
    # (the largest ones, likely the filled areas rather than clip-rect paths).
    long_paths = [p for p in panel_paths if len(p) > 100]
    assert len(long_paths) >= 2, (
        f"Expected at least 2 non-trivial KDE area paths but found {len(long_paths)}.  "
        f"Total paths: {len(panel_paths)}.  Panels may be rendering blank."
    )

    peak_x_0 = _path_peak_x(long_paths[0])
    peak_x_1 = _path_peak_x(long_paths[1])

    assert peak_x_0 is not None and peak_x_1 is not None, (
        "Could not extract peak x-coordinates from the KDE paths.  "
        f"Path 0 length: {len(long_paths[0])}, path 1 length: {len(long_paths[1])}"
    )

    diff = abs(peak_x_0 - peak_x_1)
    assert diff > 100, (
        f"KDE peak x-positions differ by only {diff:.1f} pixels "
        f"(path 0 peak: {peak_x_0:.1f}, path 1 peak: {peak_x_1:.1f}).  "
        "Both panels are showing the same curve — the KDE transform ran on the "
        "merged dataset before facet partitioning, not within each panel's data.  "
        "Group 'low' (vals 0–4) should peak far LEFT; group 'high' (vals 10–14) "
        "should peak far RIGHT.  On a 640px canvas the difference should be "
        ">100px.  Root cause: prepare.rs runs transforms before partition."
    )


def test_faceted_density_both_panels_have_marks(bimodal_facet_df: pl.DataFrame) -> None:
    """Both panels of a faceted density chart must be non-blank.

    The minimum requirement for a correctly faceted statistical mark: every
    panel has at least one non-trivial path element (the KDE area fill).  A
    blank panel (path with no coordinates or only a closure 'Z') indicates the
    transform produced zero output for that partition.

    This is a looser guard than test_faceted_density_peak_differs_between_panels
    and will catch the case where panels are entirely empty.
    """
    svg = fm.Chart(bimodal_facet_df).mark_density().encode(x="val:Q").facet(col="group").to_svg()

    panel_labels = [l for l in _panel_strip_labels(svg) if l in ("low", "high")]
    assert "low" in panel_labels and "high" in panel_labels, (
        f"Expected both 'low' and 'high' panel strip titles but found: {panel_labels}"
    )

    # At least 2 paths with real content
    non_trivial = [p for p in _paths(svg) if len(p) > 50]
    assert len(non_trivial) >= 2, (
        f"Expected ≥2 non-trivial KDE paths (one per panel) but found "
        f"{len(non_trivial)} of {len(_paths(svg))} total paths.  "
        "One or both panels are rendering blank — the KDE transform may not "
        "be running on the partitioned data."
    )


# ---------------------------------------------------------------------------
# 2c — Layered chart with distinct DataFrames: second layer must survive faceting
# ---------------------------------------------------------------------------


def test_faceted_layered_chart_second_layer_visible(
    layered_facet_df: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """A layered chart faceted by group must show marks from BOTH layers in each panel.

    Setup:
      - Layer 1 (df_a): 4 rows, 'group' column, x/y data → mark_point.
        2 rows per group (A, B), so 2 circles per panel = 4 circles total.
      - Layer 2 (df_b): 2 rows, 'group' column, highlight_x/highlight_y →
        mark_point.  1 row per group, so 1 circle per panel = 2 circles total.
      - Expected total circles: 6 (4 from layer 1 + 2 from layer 2).

    Under the bug, the ``+`` merge produces a wide DataFrame where layer 2's
    rows have ``group = null`` (the column was renamed during the outer join).
    The facet partition by 'group' drops all null-group rows, so layer 2
    contributes 0 marks in the faceted output.  Only 4 circles are rendered.

    Observable check: circle count must be 6, not 4.

    Current behavior: 4 circles (layer 2 is dropped).
    """
    df_a, df_b = layered_facet_df

    layer1 = fm.Chart(df_a).mark_point().encode(x="x:Q", y="y:Q")
    layer2 = fm.Chart(df_b).mark_point().encode(x="highlight_x:Q", y="highlight_y:Q")

    svg = (layer1 + layer2).facet(col="group").to_svg()

    circle_count = _circles(svg)
    assert circle_count == 6, (
        f"Expected 6 circles (4 from layer 1 + 2 from layer 2) but got {circle_count}.  "
        "The second layer is being dropped during facet partitioning because the "
        "'group' column in the merged DataFrame has null values for layer-2 rows "
        "(it was renamed to 'group__rhs_...' by the + merge).  "
        "Those null-group rows are excluded from every facet partition.  "
        "Root cause: Chart.__add__ renames the second DataFrame's facet column "
        "during the outer join; prepare.rs partition_batch_by_field then discards "
        "all rows with null in the partition field."
    )


def test_faceted_layered_chart_second_layer_present_per_panel(
    layered_facet_df: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """Each facet panel must contain marks from both layers.

    This is a stronger constraint than total circle count: the second layer
    must contribute at least one mark per panel, not just be present globally.

    Panel A should have 2 circles from layer 1 (x=1,y=1 and x=2,y=2) plus 1
    from layer 2 (highlight at x=1.5, y=1.5) = 3 circles in the vicinity of
    y∈[1,2].  Panel B should have 2 from layer 1 (y=4, y=5) plus 1 from layer
    2 (highlight y=4.5) = 3 circles near y∈[4,5].

    Both panels appear as strip-titled sections in the SVG.  We confirm that
    both panels exist and that the total circle count is consistent with all
    layers being present across both.

    Current behavior: total is 4 (layer 2 absent), not 6.
    """
    df_a, df_b = layered_facet_df

    layer1 = fm.Chart(df_a).mark_point().encode(x="x:Q", y="y:Q")
    layer2 = fm.Chart(df_b).mark_point().encode(x="highlight_x:Q", y="highlight_y:Q")

    svg = (layer1 + layer2).facet(col="group").to_svg()

    labels = _panel_strip_labels(svg)
    assert "A" in labels and "B" in labels, (
        f"Expected panel labels 'A' and 'B' in SVG; found: {labels}"
    )

    circle_count = _circles(svg)
    # With 2 panels × 3 circles each (2 from layer1 + 1 from layer2), expect 6.
    assert circle_count >= 6, (
        f"Expected ≥6 circles across 2 panels (both layers contributing) "
        f"but got {circle_count}.  "
        "One layer is not contributing marks to the faceted panels."
    )


# ---------------------------------------------------------------------------
# 2d — Facet scale resolution: share_scale on a faceted Chart
# ---------------------------------------------------------------------------


def test_faceted_chart_has_share_scale_method(scale_resolution_df: pl.DataFrame) -> None:
    """A faceted Chart must expose a share_scale method.

    The spec API is:
        chart.facet(...).share_scale(x="shared" | "independent",
                                     y="shared" | "independent")

    Today Chart has no share_scale attribute — it is only defined on _ChartLike
    (the composition base class used by ConcatChart, JointChart, RepeatChart).

    This test fails with AttributeError because the method does not exist.
    It will pass once share_scale is added to Chart (or to the facet-result
    object returned by facet()).
    """
    faceted = fm.Chart(scale_resolution_df).mark_point().encode(x="x:Q", y="y:Q").facet(col="group")

    assert hasattr(faceted, "share_scale"), (
        "Chart.facet(...) result has no 'share_scale' method.  "
        "The spec requires chart.facet(...).share_scale(y='independent') to "
        "produce per-panel independent y-axis domains.  "
        "Chart inherits from _RenderMixin but NOT from _ChartLike (the class "
        "that implements share_scale).  Add share_scale to Chart or return a "
        "FacetedChart wrapper from Chart.facet() that exposes this method."
    )

    # Also confirm it is callable with the expected kwargs.
    try:
        result = faceted.share_scale(y="independent")
    except Exception as exc:
        pytest.fail(
            f"faceted.share_scale(y='independent') raised {type(exc).__name__}: {exc}.  "
            "The method must accept keyword arguments matching channel names."
        )

    assert result is not None, "share_scale() must return a chart object, not None."


def test_faceted_independent_y_produces_distinct_tick_domains(
    scale_resolution_df: pl.DataFrame,
) -> None:
    """share_scale(y='independent') on a faceted Chart must yield different y-domains.

    The scale_resolution_df fixture has:
      - group 'low':  y ∈ {0, 1, 2}
      - group 'high': y ∈ {100, 101, 102}

    With independent y-scales each panel's y-axis shows ticks that track its
    own data range.  Panel 'low' shows ticks near 0–2 (e.g. '0', '0.5', '1',
    '2'); panel 'high' shows ticks near 100–102 (e.g. '100', '101', '102').
    The tick-label sets must differ.

    With shared y (the default), both panels show the same wide domain spanning
    0–102.  Both panels have identical tick-label sets.

    This test requires share_scale to exist (tested by
    test_faceted_chart_has_share_scale_method) and exercises the observable
    rendering difference.
    """
    faceted = fm.Chart(scale_resolution_df).mark_point().encode(x="x:Q", y="y:Q").facet(col="group")

    if not hasattr(faceted, "share_scale"):
        pytest.fail(
            "Cannot test scale-resolution rendering: Chart.facet() result has "
            "no share_scale method.  Fix the missing-API issue (tested by "
            "test_faceted_chart_has_share_scale_method) first."
        )

    svg_independent = faceted.share_scale(y="independent").to_svg()
    svg_shared = faceted.to_svg()  # default is shared

    ticks_independent = _y_tick_labels(svg_independent)
    ticks_shared = _y_tick_labels(svg_shared)

    assert ticks_independent, "No numeric y-axis tick labels found in independent-y SVG."
    assert ticks_shared, "No numeric y-axis tick labels found in shared-y SVG."

    # Shared: both panels see the wide 0–102 domain.  The tick set covers the
    # full range, so both '0' (or a tick near 0) AND '100' (or a tick near 100)
    # must appear in the shared SVG.
    float_ticks_shared = [float(t) for t in ticks_shared]
    assert min(float_ticks_shared) <= 5.0, (
        f"Shared y-scale: expected a tick near 0 (group 'low' data) but "
        f"minimum tick is {min(float_ticks_shared)}.  "
        f"Tick labels: {ticks_shared}"
    )
    assert max(float_ticks_shared) >= 95.0, (
        f"Shared y-scale: expected a tick near 100 (group 'high' data) but "
        f"maximum tick is {max(float_ticks_shared)}.  "
        f"Tick labels: {ticks_shared}"
    )

    # Independent: the two panels should have different tick ranges.
    # The simplest observable signal: with independent y, the full tick set
    # should NOT span 0 to 100+ in a way that makes both extremes equally dense.
    # A more robust check: the SVG must differ between shared and independent.
    assert svg_independent != svg_shared, (
        "share_scale(y='independent') produced the same SVG as the default "
        "shared-y faceted chart.  Independent y-scales must produce visibly "
        "different tick labels per panel — group 'low' (y≈0–2) should show "
        "only low-range ticks while group 'high' (y≈100–102) shows high-range "
        "ticks.  With shared y, both panels show the same 0–102 axis."
    )

    # With independent y, the tick set should include ticks in the low range
    # (near 0–2) AND separately in the high range (near 100–102).  The key
    # observable: max and min of the float tick values must span both regions,
    # but intermediate values (say, 50) should be absent — the two domains
    # are non-overlapping and no panel bridges them.
    float_ticks_ind = [float(t) for t in ticks_independent]
    mid_ticks = [t for t in float_ticks_ind if 10.0 < t < 90.0]
    assert len(mid_ticks) == 0, (
        f"share_scale(y='independent') SVG has tick labels in the intermediate "
        f"range 10–90: {mid_ticks}.  With independent y-scales, panel 'low' "
        f"should show ticks only near 0–2 and panel 'high' only near 100–102.  "
        f"A shared scale spanning 0–102 would produce ticks throughout the range.  "
        f"All independent ticks: {ticks_independent}"
    )

    # T4 escape-hatch mark-position guard: under independent y, EACH panel's marks
    # fill its OWN panel (the 'low' panel's y∈[0,2] spreads across the full panel
    # height, unlike the bunched shared case). This proves the per-panel escape
    # hatch is preserved — the complement of the shared-mode mark assertion.
    circles_ind = _data_circle_xy(svg_independent)
    assert len(circles_ind) == 6, (
        f"expected 6 data marks in independent-y SVG, got {len(circles_ind)}: {circles_ind}"
    )
    cys = sorted(c[1] for c in circles_ind)
    mid_cy = (cys[0] + cys[-1]) / 2
    top = sorted(c[1] for c in circles_ind if c[1] < mid_cy)
    bottom = sorted(c[1] for c in circles_ind if c[1] >= mid_cy)
    top_span = (top[-1] - top[0]) if len(top) >= 2 else 0.0
    bottom_span = (bottom[-1] - bottom[0]) if len(bottom) >= 2 else 0.0
    # Both panels' marks fill their own height under independent scaling.
    assert top_span > 50.0 and bottom_span > 50.0, (
        f"share_scale(y='independent'): each panel's y-marks must fill its own "
        f"height (top span={top_span:.1f}px, bottom span={bottom_span:.1f}px). "
        "A small span would mean the marks are still on a shared domain — the "
        "independent escape hatch must scale each panel to its own y-range."
    )


def test_faceted_independent_y_respects_format_override(
    scale_resolution_df: pl.DataFrame,
) -> None:
    """share_scale(y='independent') with a y format override must format per-panel tick labels.

    The scale_resolution_df fixture has:
      - group 'low':  y in {0, 1, 2}
      - group 'high': y in {100, 101, 102}

    With independent y-scales and format=".0%", every per-panel y-axis tick label
    must contain a percent sign (e.g. "0%" for 0, "100%" for 1).  The default
    formatter never emits "%" so any label with "%" proves the format was applied.

    Under the bug (D2 S3), the independent-axis rebuild in scene_build.rs clones
    prep.axes.y (which carries label_format_override=".0%") but then OVERWRITES
    tick_labels with scales.y.tick_labels(count) — raw, unformatted strings.
    The format override is silently ignored and the tick labels render as plain
    numbers ("0", "0.2", "1", "100", etc.) with no "%" suffix.

    This test must FAIL before the fix (no "%" in any tick label) and PASS after
    (at least one tick label contains "%").
    """
    if not hasattr(
        fm.Chart(scale_resolution_df).mark_point().encode(x="x:Q", y="y:Q").facet(col="group"),
        "share_scale",
    ):
        pytest.skip("share_scale not available — skipping format-override test")

    svg = (
        fm.Chart(scale_resolution_df)
        .mark_point()
        .encode(x="x:Q", y=fm.Y("y", format=".0%"))
        .facet(col="group")
        .share_scale(y="independent")
        .to_svg()
    )

    # Collect all text elements with text-anchor="end" (y-axis labels).
    y_labels_raw: list[str] = []
    for attrs, content in re.findall(r"<text\s([^>]+)>([^<]+)</text>", svg):
        if 'text-anchor="end"' in attrs and content.strip():
            y_labels_raw.append(content.strip())

    assert y_labels_raw, (
        "No y-axis tick labels found in the independent-y SVG.  Cannot verify format override."
    )

    # With format=".0%", at least one tick label must contain a percent sign.
    labels_with_percent = [l for l in y_labels_raw if "%" in l]
    assert len(labels_with_percent) > 0, (
        f"format='.0%' on an independent-y facet produced NO tick labels with "
        f"a percent sign.  The format override was not applied to the per-panel "
        f"axis rebuild.  All y-axis labels found: {y_labels_raw}.  "
        "Root cause: scene_build.rs independent_y_layout overwrites tick_labels "
        "with scales.y.tick_labels(count) (raw unformatted strings) without "
        "re-applying the label_format_override carried in prep.axes.y."
    )


def test_faceted_shared_y_produces_identical_per_panel_domains(
    scale_resolution_df: pl.DataFrame,
) -> None:
    """The default shared-y facet must produce the same y-domain in all panels.

    When scale resolution is shared (the default behavior), every panel locks
    to the union domain.  With 'low' y∈[0,2] and 'high' y∈[100,102], the union
    is [0, 102].  Both panels should have identical y-axis tick labels.

    Observable check: the y tick labels form a set that spans 0 to ≥100, and
    the SVG for each panel contains the same tick values (we check this by
    confirming both extremes appear together, i.e. '0'-ish AND '100'-ish ticks
    are present in the same SVG output).

    This test should PASS today (shared is the current default) and acts as a
    regression guard — we must not break the default shared behavior when adding
    the independent-y escape.
    """
    svg = (
        fm.Chart(scale_resolution_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .facet(col="group")
        .to_svg()
    )

    float_ticks = [float(t) for t in _y_tick_labels(svg)]
    assert float_ticks, "No numeric y-axis tick labels found in shared-y SVG."

    assert min(float_ticks) <= 5.0, (
        f"Shared y-scale: expected a tick near 0 (group 'low' range) but "
        f"minimum tick is {min(float_ticks)}.  Ticks: {_y_tick_labels(svg)}"
    )
    assert max(float_ticks) >= 95.0, (
        f"Shared y-scale: expected a tick near 100 (group 'high' range) but "
        f"maximum tick is {max(float_ticks)}.  Ticks: {_y_tick_labels(svg)}"
    )

    # T4 mark-position guard: axis ticks already showed the global domain even
    # when the bug was live (the bug was per-panel MARK scaling, not the axis).
    # Assert the MARKS map through the shared global y-domain [0, 102]:
    #   - The 'low' panel's y∈{0,1,2} marks are bunched together at the very
    #     bottom of the [0,102] range (their cy values span only a few pixels).
    #   - Under the per-panel bug they would fill the whole panel (cy span ≈ the
    #     panel height), since 0/1/2 would scale to the panel's local [0,2] domain.
    circles = _data_circle_xy(svg)
    assert len(circles) == 6, f"expected 6 data marks, got {len(circles)}: {circles}"
    cys = sorted(c[1] for c in circles)
    mid_cy = (cys[0] + cys[-1]) / 2
    # The 'low' panel is the one whose three marks have the SMALLER cy-spread
    # (bunched on the global domain). Identify by splitting into two cy clusters.
    top = sorted(c[1] for c in circles if c[1] < mid_cy)
    bottom = sorted(c[1] for c in circles if c[1] >= mid_cy)
    top_span = (top[-1] - top[0]) if len(top) >= 2 else 0.0
    bottom_span = (bottom[-1] - bottom[0]) if len(bottom) >= 2 else 0.0
    low_panel_span = min(top_span, bottom_span)
    # Full vertical mark extent across both panels (a proxy for panel height).
    full_span = cys[-1] - cys[0]
    assert low_panel_span < 0.15 * full_span, (
        f"Shared y: the 'low' panel's y∈[0,2] marks span {low_panel_span:.1f}px, "
        f"{100 * low_panel_span / full_span:.0f}% of the {full_span:.1f}px extent. "
        "On the shared global [0,102] domain they must be bunched (<15%); a wide "
        "spread means per-panel MARK scaling — the marks would disagree with the "
        "shared axis (the T4 bug)."
    )
