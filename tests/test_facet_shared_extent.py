"""Python end-to-end verification of archaeology bug #7: faceted shared-extent pin.

Spec §4/#7: A faceted chart whose panels use Bin, Kde, or Violin with auto extent
renders every panel (and every hue group within a panel) on the same value-axis
range.

Discriminating strategy
-----------------------
Each test builds data where per-panel ranges are completely NON-OVERLAPPING (e.g.
Panel A has values 1–5, Panel B has values 12–20) so that WITHOUT the pin the two
panels would show detectably different axis extents.  WITH the pin both panels must
show tick labels spanning the FULL global range.

The assertions fail if and only if the pin is removed from the Rust
``fix_transform_extents_for_facet`` path.

Parsing approach
----------------
Axis tick label values are extracted from rendered SVG ``<text>`` elements.
For column-faceted charts:
  - Histogram / KDE: value axis is x.  Each panel's x-axis tick row has a
    distinct y-position in the SVG.  We group numeric ``<text>`` elements by
    their rounded y-coordinate; rows with ≥ 3 entries are x-axis tick rows (one
    per panel).
  - Violin: value axis is y.  All panels share the same x-coordinate for their
    y-axis ticks.  We split entries by y-position gaps (a gap > 3× the average
    tick spacing indicates a panel boundary) to recover per-panel tick groups.

NOTE on bin edges vs. extent:
  Sturges auto-binning may produce different interior edge counts per panel, but
  the shared-extent guarantee is about the AXIS RANGE (min/max of the tick
  labels), not identical interior edges.  The assertions check shared (min, max),
  not edge-by-edge equality.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

GOLDENS_DIR = Path(__file__).parent / "goldens" / "facet_shared_extent"
UPDATE = os.environ.get("FERRUM_UPDATE_GOLDENS") == "1"


@pytest.fixture
def df_nonoverlapping() -> pl.DataFrame:
    """Two panels with completely non-overlapping value ranges.

    Panel 'A': values 1–5 (30 rows).
    Panel 'B': values 12–20 (30 rows).

    Without the shared-extent pin a histogram/KDE/violin for panel A would show
    x-range ~[0, 5] and panel B ~[10, 20]; with the pin both panels share the
    full ~[0, 20] range.
    """
    return pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0, 5.0] * 6 + [12.0, 14.0, 16.0, 18.0, 20.0] * 6,
            "cat": ["A"] * 30 + ["B"] * 30,
        }
    )


@pytest.fixture
def df_hue_nonoverlapping() -> pl.DataFrame:
    """Two panels, two hue groups, each panel/group combination non-overlapping.

    Panel 'A': val 1–5.  Panel 'B': val 12–20.
    Hue group 'x' and 'y' both present in each panel (interleaved rows).
    This is the multi-group (hue) case that Tasks 8-9 newly enabled.
    Without the fix the groupby early-return in the Rust pin would skip pinning
    for KDE and leave Bin/Violin un-pinned in the multi-group path.
    """
    return pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0, 5.0] * 6 + [12.0, 14.0, 16.0, 18.0, 20.0] * 6,
            "grp": ["x", "y"] * 30,
            "cat": ["A"] * 30 + ["B"] * 30,
        }
    )


# ---------------------------------------------------------------------------
# SVG parsing helpers
# ---------------------------------------------------------------------------


class AxisExtent(NamedTuple):
    lo: float
    hi: float


def _numeric_text_entries(svg: str) -> list[tuple[float, float, float]]:
    """Return (x, y, value) for every ``<text>`` element whose content parses as float."""
    entries = []
    for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
        try:
            val = float(text.strip())
        except ValueError:
            continue
        x_m = re.search(r'x="([^"]+)"', attrs)
        y_m = re.search(r'y="([^"]+)"', attrs)
        if x_m and y_m:
            entries.append((float(x_m.group(1)), float(y_m.group(1)), val))
    return entries


def x_axis_extents(svg: str) -> list[AxisExtent]:
    """Per-panel x-axis tick extents for a column-faceted chart.

    Groups numeric ``<text>`` elements by their rounded y-coordinate.  Each
    group with at least 3 entries corresponds to one panel's x-axis tick row.
    Returns a list of (lo, hi) sorted by y-position (top panel first).
    """
    entries = _numeric_text_entries(svg)
    y_groups: dict[int, list[float]] = defaultdict(list)
    for _, y, val in entries:
        y_groups[round(y)].append(val)
    rows = [(y, vals) for y, vals in y_groups.items() if len(vals) >= 3]
    rows.sort()
    return [AxisExtent(min(vals), max(vals)) for _, vals in rows]


def y_axis_extents(svg: str) -> list[AxisExtent]:
    """Per-panel y-axis tick extents for a column-faceted chart.

    All panels' y-axis ticks share the same x-coordinate (leftmost y-axis).
    Entries are split into panel groups by detecting y-position gaps larger
    than 3× the average tick spacing.

    Returns a list of (lo, hi) sorted by panel order (top panel first).
    """
    entries = _numeric_text_entries(svg)
    # Collect entries grouped by x-coordinate (same x = same y-axis column).
    x_groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y, val in entries:
        x_groups[round(x)].append((y, val))

    # The y-axis column is the leftmost group with at least 3 entries.
    axis_cols = {x: ents for x, ents in x_groups.items() if len(ents) >= 3}
    if not axis_cols:
        return []

    axis_x = min(axis_cols)
    ents = sorted(axis_cols[axis_x])  # sorted by y-position ascending

    if len(ents) < 2:
        return [AxisExtent(ents[0][1], ents[0][1])]

    # Split by gaps to identify panel boundaries.
    avg_spacing = (ents[-1][0] - ents[0][0]) / (len(ents) - 1)
    panel_groups: list[list[float]] = []
    current: list[float] = [ents[0][1]]
    for i in range(1, len(ents)):
        if ents[i][0] - ents[i - 1][0] > avg_spacing * 3:
            panel_groups.append(current)
            current = [ents[i][1]]
        else:
            current.append(ents[i][1])
    panel_groups.append(current)

    return [AxisExtent(min(g), max(g)) for g in panel_groups]


def _extents_all_equal(extents: list[AxisExtent]) -> bool:
    """True iff every panel's (lo, hi) is the same."""
    if len(extents) < 2:
        return True
    first = extents[0]
    return all(e == first for e in extents[1:])


# ---------------------------------------------------------------------------
# 1. Faceted histogram — single group
# ---------------------------------------------------------------------------


class TestHistogramFacetSingleGroup:
    """Faceted mark_histogram with no hue grouping.

    Without the pin: each panel uses its own local bin range.
    With the pin: all panels share the niced global range.
    """

    def test_two_panels_share_x_extent(self, df_nonoverlapping: pl.DataFrame) -> None:
        """Both panels show the same x-axis tick range covering all data."""
        svg = (
            fm.Chart(df_nonoverlapping)
            .mark_histogram()
            .encode(x="val:Q", y="count")
            .facet(col="cat")
            .to_svg()
        )
        extents = x_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Panel x-extents differ: {extents}. "
            "Without the shared-extent pin each panel would show its local range "
            "(~[0,5] vs ~[10,20]); with the pin both panels share the niced global range."
        )
        # Sanity: the shared range must cover BOTH panels' data.
        # For histogram, bin edges start at a niced value <= data min, so lo <= data min.
        lo, hi = extents[0]
        assert lo <= 1.0, f"Shared lo={lo} does not cover Panel A min=1.0"
        assert hi >= 20.0, f"Shared hi={hi} does not cover Panel B max=20.0"

    def test_three_panels_share_x_extent(self) -> None:
        """Three panels with disjoint ranges all share one global extent."""
        df = pl.DataFrame(
            {
                "val": ([1.0, 2.0, 3.0] * 5 + [10.0, 11.0, 12.0] * 5 + [20.0, 21.0, 22.0] * 5),
                "cat": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
            }
        )
        svg = fm.Chart(df).mark_histogram().encode(x="val:Q", y="count").facet(col="cat").to_svg()
        extents = x_axis_extents(svg)
        assert len(extents) == 3, f"Expected 3 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Panel x-extents differ: {extents}. Expected shared global extent."
        )
        lo, hi = extents[0]
        assert lo <= 1.0 and hi >= 22.0, (
            f"Shared extent [{lo}, {hi}] does not cover global range [1, 22]"
        )


# ---------------------------------------------------------------------------
# 2. Faceted histogram — multi-group (hue)
# ---------------------------------------------------------------------------


class TestHistogramFacetMultiGroup:
    """Faceted mark_histogram with a hue (groupby) split.

    This is the case that Tasks 8-9 newly enabled at the Rust level.  Before
    the fix, Bin with a groupby might not receive the pinned global extent,
    causing each panel's histogram to use its local data range.
    """

    def test_hue_split_two_panels_share_x_extent(self, df_hue_nonoverlapping: pl.DataFrame) -> None:
        """Both panels share the x-axis range even with a per-hue histogram split."""
        svg = (
            fm.Chart(df_hue_nonoverlapping)
            .mark_histogram(groupby="grp")
            .encode(x="val:Q", y="count", color="grp:N")
            .facet(col="cat")
            .to_svg()
        )
        extents = x_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Multi-group panel x-extents differ: {extents}. "
            "The hue split must not prevent global-extent pinning."
        )
        lo, hi = extents[0]
        assert lo <= 1.0, f"Shared lo={lo} does not cover Panel A min=1.0"
        assert hi >= 20.0, f"Shared hi={hi} does not cover Panel B max=20.0"


# ---------------------------------------------------------------------------
# 3. Faceted violin — single group
# ---------------------------------------------------------------------------


class TestViolinFacetSingleGroup:
    """Faceted mark_violin with no per-group split.

    Without the pin each panel's violin KDE would fit to its local y data range.
    With the pin all panels share the global y-axis extent.
    """

    def test_two_panels_share_y_extent(self, df_nonoverlapping: pl.DataFrame) -> None:
        """Both violin panels show the same y-axis tick range."""
        # For violin: x=cat (within-panel category), y=val (value axis).
        # We need a within-panel categorical x.  Use a constant 'grp' column.
        df = df_nonoverlapping.with_columns(pl.lit("all").alias("grp"))
        svg = fm.Chart(df).mark_violin().encode(x="grp:N", y="val:Q").facet(col="cat").to_svg()
        extents = y_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Panel y-extents differ: {extents}. "
            "Without the shared-extent pin panel A would show y up to ~5 and "
            "panel B from ~12 to ~20; with the pin both span the global range."
        )
        # Sanity: the shared range must encompass both panels' data.
        # For violin/KDE, ticks are at even intervals; the axis extent covers the
        # full global range but the lowest tick may sit at 2 even when data min=1.
        # The critical check is that Panel B's max (20) is covered AND that
        # Panel A's data range (1–5) is within the tick range (lo <= 5).
        lo, hi = extents[0]
        assert lo <= 5.0, (
            f"Shared lo={lo} does not cover Panel A data range [1, 5]. "
            "Without the pin, Panel A's axis would top out at ~5 while Panel B "
            "starts at ~12 — they would have non-overlapping extents."
        )
        assert hi >= 20.0, f"Shared hi={hi} does not cover Panel B max=20.0"


# ---------------------------------------------------------------------------
# 4. Faceted violin — multi-group (hue via x-axis categories)
# ---------------------------------------------------------------------------


class TestViolinFacetMultiGroup:
    """Faceted mark_violin with multiple within-panel categories.

    The two groups within each panel have values from different ranges, so
    without the shared extent the per-group KDEs would be evaluated over
    different y windows.  With the pin all groups AND panels share one y-axis.
    """

    def test_two_groups_two_panels_share_y_extent(
        self, df_hue_nonoverlapping: pl.DataFrame
    ) -> None:
        """Two panels × two within-panel groups all share the same y-axis extent."""
        svg = (
            fm.Chart(df_hue_nonoverlapping)
            .mark_violin()
            .encode(x="grp:N", y="val:Q")
            .facet(col="cat")
            .to_svg()
        )
        extents = y_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Multi-group violin panel y-extents differ: {extents}. "
            "All panels and within-panel groups must share the global y-axis extent."
        )
        lo, hi = extents[0]
        assert lo <= 5.0, (
            f"Shared lo={lo} does not cover Panel A data range [1, 5]. "
            "Without the pin the y-axis would not span both panels' data."
        )
        assert hi >= 20.0, f"Shared hi={hi} does not cover Panel B max=20.0"


# ---------------------------------------------------------------------------
# 5. Faceted KDE — single group (regression: was correct before, must stay)
# ---------------------------------------------------------------------------


class TestKDEFacetSingleGroup:
    """Faceted mark_density single-group — behavior unchanged by tasks 8-9.

    Single-group KDE was already pinned before this fix.  This test guards
    against regression.
    """

    def test_two_panels_share_x_extent(self, df_nonoverlapping: pl.DataFrame) -> None:
        """Both KDE panels show the same x-axis tick range."""
        svg = fm.Chart(df_nonoverlapping).mark_density().encode(x="val:Q").facet(col="cat").to_svg()
        extents = x_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Single-group KDE panel x-extents differ: {extents}. "
            "This is a regression: single-group KDE was already pinned before task 8-9."
        )
        # KDE x-axis ticks are at nice intervals within the KDE evaluation range.
        # The lowest tick may be above the raw data min (1.0) since the KDE range
        # extends to cover the density tails; the tick below the data min may not
        # be shown.  The critical check: both panels must cover Panel A (lo <= 5)
        # and Panel B (hi >= 20).
        lo, hi = extents[0]
        assert lo <= 5.0, f"Shared lo={lo} does not cover Panel A data range [1, 5]."
        assert hi >= 20.0, f"Shared hi={hi} does not cover Panel B max=20.0"


# ---------------------------------------------------------------------------
# 6. Faceted KDE — multi-group (hue): NEW behavior from tasks 8-9
# ---------------------------------------------------------------------------


class TestKDEFacetMultiGroup:
    """Faceted mark_density with groupby — the multi-group fix from task 9.

    Before task 9, the Rust code had an early-return when ``groupby.is_some()``
    that skipped extent pinning for grouped KDEs.  This test verifies the fix.
    """

    def test_hue_split_two_panels_share_x_extent(self, df_hue_nonoverlapping: pl.DataFrame) -> None:
        """Both panels share the x-axis range even with per-hue KDE curves."""
        svg = (
            fm.Chart(df_hue_nonoverlapping)
            .mark_density(groupby="grp")
            .encode(x="val:Q", color="grp:N")
            .facet(col="cat")
            .to_svg()
        )
        extents = x_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        assert _extents_all_equal(extents), (
            f"Multi-group KDE panel x-extents differ: {extents}. "
            "The groupby early-return bug (pre-task-9) would leave each panel on its "
            "local extent instead of the global extent."
        )
        lo, hi = extents[0]
        assert lo <= 5.0, (
            f"Shared lo={lo} does not cover Panel A data range [1, 5]. "
            "The groupby early-return bug (pre-task-9) would leave each panel on "
            "its local extent."
        )
        assert hi >= 20.0, f"Shared hi={hi} does not cover Panel B max=20.0"


# ---------------------------------------------------------------------------
# 7. User-specified extent is preserved and not overridden by the pin
# ---------------------------------------------------------------------------


class TestUserExtentPreserved:
    """When the user specifies an explicit extent, the pin must NOT override it.

    Spec: ``fix_transform_extents_for_facet`` skips transforms that already have
    an explicit ``extent`` set.  This test verifies that user intent is respected
    even in a faceted context.
    """

    def test_user_extent_density(self, df_nonoverlapping: pl.DataFrame) -> None:
        """User extent (-5, 30) is used instead of the auto-pinned global extent."""
        user_lo, user_hi = -5.0, 30.0
        svg = (
            fm.Chart(df_nonoverlapping)
            .mark_density(extent=(user_lo, user_hi))
            .encode(x="val:Q")
            .facet(col="cat")
            .to_svg()
        )
        extents = x_axis_extents(svg)
        assert len(extents) == 2, f"Expected 2 panels, found {len(extents)}"
        # Both panels should use the user-specified range.
        for ext in extents:
            assert ext.lo <= user_lo + 1, f"Panel lo={ext.lo} should be near user_lo={user_lo}"
            assert ext.hi >= user_hi - 1, f"Panel hi={ext.hi} should be near user_hi={user_hi}"
        # Verify the extents are equal (pinned to the same user extent).
        assert _extents_all_equal(extents), (
            f"User-extent panels differ: {extents} — even user extents should be shared."
        )

    def test_user_extent_overrides_data_range(self, df_nonoverlapping: pl.DataFrame) -> None:
        """User extent pins to (-5, 30), NOT to the auto-computed data range."""
        user_lo, user_hi = -5.0, 30.0
        svg = (
            fm.Chart(df_nonoverlapping)
            .mark_density(extent=(user_lo, user_hi))
            .encode(x="val:Q")
            .facet(col="cat")
            .to_svg()
        )
        extents = x_axis_extents(svg)
        assert len(extents) == 2
        lo, hi = extents[0]
        # If the pin had overridden the user extent with the auto extent, lo would
        # be ~0 and hi would be ~20 (global data range).  The user extent makes
        # the x-axis go to -5 and 30 instead.
        assert lo < 0.0, (
            f"Expected lo < 0 (from user extent -5), but got lo={lo}. "
            "Auto-pin may have overridden user extent."
        )
        assert hi > 25.0, (
            f"Expected hi > 25 (from user extent 30), but got hi={hi}. "
            "Auto-pin may have overridden user extent."
        )


# ---------------------------------------------------------------------------
# 8. Golden tests for the new multi-group faceted cases
# ---------------------------------------------------------------------------


class TestFacetSharedExtentGoldens:
    """Golden SVG tests for the faceted shared-extent cases.

    Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate on-disk goldens.
    A missing golden is an explicit test failure unless the update flag is set.
    """

    def _check_or_update(self, name: str, svg: str) -> None:
        GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
        golden = GOLDENS_DIR / name
        if UPDATE:
            golden.write_text(svg)
            return
        if not golden.exists():
            pytest.fail(
                f"golden {name!r} does not exist; rerun with FERRUM_UPDATE_GOLDENS=1 to regenerate"
            )
        from tests._snapshots import assert_svg_eq

        assert_svg_eq(
            svg,
            golden.read_text(),
            name=name,
            regen_hint="FERRUM_UPDATE_GOLDENS=1 uv run pytest tests/test_facet_shared_extent.py",
        )

    def test_golden_histogram_facet_single_group(self, df_nonoverlapping: pl.DataFrame) -> None:
        svg = (
            fm.Chart(df_nonoverlapping)
            .mark_histogram()
            .encode(x="val:Q", y="count")
            .facet(col="cat")
            .to_svg()
        )
        self._check_or_update("histogram_facet_single_group.svg", svg)

    def test_golden_histogram_facet_multi_group(self, df_hue_nonoverlapping: pl.DataFrame) -> None:
        svg = (
            fm.Chart(df_hue_nonoverlapping)
            .mark_histogram(groupby="grp")
            .encode(x="val:Q", y="count", color="grp:N")
            .facet(col="cat")
            .to_svg()
        )
        self._check_or_update("histogram_facet_multi_group.svg", svg)

    def test_golden_violin_facet_single_group(self, df_nonoverlapping: pl.DataFrame) -> None:
        df = df_nonoverlapping.with_columns(pl.lit("all").alias("grp"))
        svg = fm.Chart(df).mark_violin().encode(x="grp:N", y="val:Q").facet(col="cat").to_svg()
        self._check_or_update("violin_facet_single_group.svg", svg)

    def test_golden_violin_facet_multi_group(self, df_hue_nonoverlapping: pl.DataFrame) -> None:
        svg = (
            fm.Chart(df_hue_nonoverlapping)
            .mark_violin()
            .encode(x="grp:N", y="val:Q")
            .facet(col="cat")
            .to_svg()
        )
        self._check_or_update("violin_facet_multi_group.svg", svg)

    def test_golden_kde_facet_multi_group(self, df_hue_nonoverlapping: pl.DataFrame) -> None:
        svg = (
            fm.Chart(df_hue_nonoverlapping)
            .mark_density(groupby="grp")
            .encode(x="val:Q", color="grp:N")
            .facet(col="cat")
            .to_svg()
        )
        self._check_or_update("kde_facet_multi_group.svg", svg)


# ---------------------------------------------------------------------------
# 9. T4 — shared faceted positional scale for RAW fields (no transform)
# ---------------------------------------------------------------------------
#
# Archaeology round-7 T4. A faceted RAW (untransformed) continuous/categorical
# field with the default ResolveMode::Shared must position marks through the
# GLOBAL all-panels domain, not its own per-panel partition. Before T4 a faceted
# scatter scaled each panel's marks to that panel's local range while the shared
# axis showed the global domain — marks and axis labels DISAGREED. These tests
# assert on mark PIXEL positions (the existing facet tests only check axis ticks,
# which already showed the global domain even when the bug was live).

RAW_GOLDENS_DIR = Path(__file__).parent / "goldens" / "facet_shared_raw"


def _data_circles(svg: str) -> list[tuple[float, float]]:
    """Return (cx, cy) for every data-mark ``<circle>`` (theme radius ≈ 3.x).

    Legend swatches use r="4"; data points use the theme point radius (~3.385),
    so the radius discriminates marks from swatches.
    """
    out: list[tuple[float, float]] = []
    for cx, cy, r in re.findall(
        r'<circle cx="([0-9.]+)" cy="([0-9.]+)" r="([0-9.]+)"', svg
    ):
        if abs(float(r) - 4.0) > 1e-6:
            out.append((float(cx), float(cy)))
    return out


def _split_panels_by_cy(circles: list[tuple[float, float]]) -> tuple[list, list]:
    """Split data circles into the top and bottom panels by cy (row-stacked facet).

    The fixture stacks two panels vertically, so the top panel's marks have
    smaller cy than the bottom panel's. Returns (top_panel, bottom_panel), each
    a list of (cx, cy) sorted by cx.
    """
    if not circles:
        return [], []
    cys = sorted(c[1] for c in circles)
    mid = (cys[0] + cys[-1]) / 2
    top = sorted((c for c in circles if c[1] < mid), key=lambda c: c[0])
    bottom = sorted((c for c in circles if c[1] >= mid), key=lambda c: c[0])
    return top, bottom


@pytest.fixture
def raw_disjoint_df() -> pl.DataFrame:
    """Two panels with disjoint RAW x ranges (no transform).

    Panel 'A': x ∈ {0, 5, 10}.   Panel 'B': x ∈ {0, 50, 100}.
    Global x ∈ [0, 100].

    Under Shared (default), panel A's x=10 mark must land near the LEFT of its
    panel (10% of the global [0,100] domain), not at the right edge.
    """
    return pl.DataFrame(
        {
            "x": [0.0, 5.0, 10.0, 0.0, 50.0, 100.0],
            "y": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "cat": ["A", "A", "A", "B", "B", "B"],
        }
    )


class TestRawFieldSharedFacet:
    """T4 mark-position discrimination for faceted RAW fields.

    The fixture stacks two panels vertically (the panels share the full canvas
    width, so a SHARED x-axis spans 0–100 in both). Panel A holds x∈{0,5,10};
    panel B holds x∈{0,50,100}. Under the shared global [0,100] domain, panel A's
    x=10 mark must land at the SAME cx as panel B's x=10 position (the '10' tick),
    far LEFT of where x=100 lands. Under the per-panel bug, panel A's x=10 would
    fill its own panel and land near the right edge instead.
    """

    def test_shared_x_marks_use_global_domain(self, raw_disjoint_df: pl.DataFrame) -> None:
        """Panel A's x=10 mark lands at the global '10' position, not the right edge."""
        svg = (
            fm.Chart(raw_disjoint_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .facet(col="cat")
            .to_svg()
        )
        circles = _data_circles(svg)
        assert len(circles) == 6, f"expected 6 data marks, got {len(circles)}: {circles}"
        top, bottom = _split_panels_by_cy(circles)
        assert len(top) == 3 and len(bottom) == 3, (
            f"each panel must have 3 marks: top={top}, bottom={bottom}"
        )
        # The full x-mark extent across BOTH panels: x=0 (min) → x=100 (max).
        all_cx = sorted(c[0] for c in circles)
        x0_px, x100_px = all_cx[0], all_cx[-1]
        full_span = x100_px - x0_px

        # Panel A is the one whose marks are x∈{0,5,10} — its 3 cx values are
        # bunched in the left ~10% of the global span. Identify it as the panel
        # with the SMALLER cx-spread.
        top_span = top[-1][0] - top[0][0]
        bottom_span = bottom[-1][0] - bottom[0][0]
        panel_a = top if top_span < bottom_span else bottom
        a_span = min(top_span, bottom_span)

        # Under the SHARED global [0,100] domain, panel A's x∈[0,10] occupies only
        # ~10% of the full span. Under the per-panel bug it would occupy the FULL
        # panel width (≈ full_span). Require panel A's span be < 25% of the full
        # span — impossible under per-panel scaling, easy under shared.
        assert a_span < 0.25 * full_span, (
            f"Panel A x∈[0,10] spans {a_span:.1f}px of the {full_span:.1f}px global "
            f"extent ({100 * a_span / full_span:.0f}%). Under the shared [0,100] "
            "domain it must be <25%; a larger fraction means per-panel scaling (the "
            "T4 bug placed panel A's x=10 at the right edge while the axis showed 100)."
        )
        # Panel A's x=10 (its max) must sit near the global '10' position — close to
        # x=0, far from x=100.
        a_max_cx = panel_a[-1][0]
        assert a_max_cx - x0_px < 0.20 * full_span, (
            f"Panel A's x=10 mark is at cx={a_max_cx:.1f}, {a_max_cx - x0_px:.1f}px "
            f"from x=0 — should be within 20% of the {full_span:.1f}px global span "
            "(the '10' tick), not near the right edge."
        )

    def test_independent_x_marks_use_per_panel_domain(
        self, raw_disjoint_df: pl.DataFrame
    ) -> None:
        """share_scale(x='independent') keeps per-panel scaling (escape hatch)."""
        svg = (
            fm.Chart(raw_disjoint_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .facet(col="cat")
            .share_scale(x="independent")
            .to_svg()
        )
        circles = _data_circles(svg)
        assert len(circles) == 6, f"expected 6 data marks, got {len(circles)}: {circles}"
        top, bottom = _split_panels_by_cy(circles)
        assert len(top) == 3 and len(bottom) == 3, (
            f"each panel must have 3 marks: top={top}, bottom={bottom}"
        )
        top_span = top[-1][0] - top[0][0]
        bottom_span = bottom[-1][0] - bottom[0][0]
        # Independent: EACH panel fills its own width, so BOTH panels' x-mark spans
        # are large and roughly equal (x=0..10 and x=0..100 both fill the panel).
        assert top_span > 300.0 and bottom_span > 300.0, (
            f"Independent mode: each panel's x-marks must fill its own width "
            f"(top span={top_span:.1f}px, bottom span={bottom_span:.1f}px). "
            "Both should be wide; the per-panel escape hatch must be preserved."
        )

    def test_shared_categorical_panel_keeps_all_bands(self) -> None:
        """A faceted categorical x where a panel is missing a category present in
        another panel must, under Shared, still lay out all categories' bands.

        Fixture:
          Panel A has categories {a, c}; Panel B has categories {b, c}.
          Global union is {a, b, c}.

        Under SHARED: every panel renders all three bands, so 'a', 'b', and 'c'
        each appear as axis tick labels exactly twice in the SVG (once per panel).

        Under INDEPENDENT: panel A renders only {a, c} and panel B only {b, c},
        so 'a' appears once, 'b' appears once, and 'c' appears twice.

        This discrimination fails if the shared-categorical bug is re-introduced:
        a regression would revert shared mode to per-panel domains, causing 'a'
        and 'b' to appear only once each even in the shared SVG.
        """
        df = pl.DataFrame(
            {
                "cat": ["a", "c", "b", "c"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "panel": ["A", "A", "B", "B"],
            }
        )
        svg_shared = (
            fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q").facet(col="panel").to_svg()
        )
        svg_indep = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="cat:N", y="y:Q")
            .facet(col="panel")
            .share_scale(x="independent")
            .to_svg()
        )

        def _category_label_counts(svg: str) -> dict[str, int]:
            """Count occurrences of each single-letter category label in axis tick <text> elements."""
            counts: dict[str, int] = {}
            for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
                label = text.strip()
                if label in ("a", "b", "c"):
                    counts[label] = counts.get(label, 0) + 1
            return counts

        shared_counts = _category_label_counts(svg_shared)
        indep_counts = _category_label_counts(svg_indep)

        # Under SHARED: all three categories appear in every panel (2 panels),
        # so each label must appear exactly twice.
        assert shared_counts.get("a") == 2, (
            f"Shared mode: expected 'a' to appear twice (once per panel, global union), "
            f"got {shared_counts.get('a', 0)}. "
            "If 'a' appears only once, panel B is not showing the full band union."
        )
        assert shared_counts.get("b") == 2, (
            f"Shared mode: expected 'b' to appear twice (once per panel, global union), "
            f"got {shared_counts.get('b', 0)}. "
            "If 'b' appears only once, panel A is not showing the full band union."
        )
        assert shared_counts.get("c") == 2, (
            f"Shared mode: expected 'c' to appear twice (once per panel), "
            f"got {shared_counts.get('c', 0)}."
        )

        # Under INDEPENDENT: each panel shows only its own categories,
        # so 'a' and 'b' each appear only once (in their own panel).
        assert indep_counts.get("a", 0) == 1, (
            f"Independent mode: expected 'a' once (panel A only), "
            f"got {indep_counts.get('a', 0)}."
        )
        assert indep_counts.get("b", 0) == 1, (
            f"Independent mode: expected 'b' once (panel B only), "
            f"got {indep_counts.get('b', 0)}."
        )
        assert indep_counts.get("c", 0) == 2, (
            f"Independent mode: expected 'c' twice (present in both panels), "
            f"got {indep_counts.get('c', 0)}."
        )


class TestRawFieldSharedFacetGolden:
    """Golden for a faceted RAW scatter on the shared domain (orchestrator-blessed)."""

    def _check_or_update(self, name: str, svg: str) -> None:
        RAW_GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
        golden = RAW_GOLDENS_DIR / name
        if UPDATE:
            golden.write_text(svg)
            return
        if not golden.exists():
            pytest.fail(
                f"golden {name!r} does not exist; rerun with FERRUM_UPDATE_GOLDENS=1 to regenerate"
            )
        from tests._snapshots import assert_svg_eq

        assert_svg_eq(
            svg,
            golden.read_text(),
            name=name,
            regen_hint="FERRUM_UPDATE_GOLDENS=1 uv run pytest tests/test_facet_shared_extent.py",
        )

    def test_golden_raw_scatter_shared_facet(self, raw_disjoint_df: pl.DataFrame) -> None:
        svg = (
            fm.Chart(raw_disjoint_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .facet(col="cat")
            .to_svg()
        )
        self._check_or_update("raw_scatter_shared_facet.svg", svg)
