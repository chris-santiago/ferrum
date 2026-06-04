"""Failing regression tests for defect D7 — polar coordinate gaps.

Three gaps are covered:

1. **Stacked radial bars overlap at r=0** — when a stacked bar chart uses
   CoordPolar, all segments share the same inner radius (r=0) instead of
   accumulating outward.  The correct behaviour is that each stacked segment's
   inner radius equals the previous segment's outer radius (wind-rose /
   coxcomb accumulation).

2. **Theta2/Radius2 channels are absent** — the spec requires ``Theta2(field)``
   and ``Radius2(field)`` channels for annular wedge marks (inner/outer angular
   and radial extents).  Neither channel class exists in ``ferrum.encoding``
   nor in ``ferrum.__init__``.

3. **Sunburst via pre-laid-out hierarchy rows** — encoding theta + theta2 +
   radius + radius2 on a dataset with pre-computed ring/slice boundaries should
   render nested wedge rings.  This requires all four polar extent channels.

Root causes (identified, not yet fixed):

  Python — ``ferrum/encoding/positional.py``:
    ``Theta2`` and ``Radius2`` channel classes are entirely absent.
    ``ferrum/encoding/__init__.py`` does not import or export them.
    ``ferrum/__init__.py`` does not export them.
    The channel-class map in ``ferrum/encoding/__init__.py``
    (``_channel_class_map``) has no entries for ``"theta2"`` or ``"radius2"``.

  Python — ``ferrum/coord.py``:
    ``_resolve_polar_remapping()`` in ``chart.py`` maps ``theta`` → x/y and
    ``radius`` → y/x but has no code path for ``theta2`` → x2/y2 or
    ``radius2`` → y2/x2.

  Rust — ``crates/ferrum-core/src/render/marks/arc.rs`` ``build()``:
    Arc renderer reads a single theta field and sweeps from a fixed inner
    radius to a fixed outer radius, both taken from ``CoordPolar`` coord
    params.  It has no support for per-row ``theta2`` (angular end) or
    per-row ``radius`` / ``radius2`` (inner/outer radii from data columns).

  Rust — ``crates/ferrum-core/src/render/position.rs`` ``apply_stack()``:
    ``apply_stack`` accumulates a Cartesian y column
    (``__stack_y_base__`` → new y).  When the chart has CoordPolar, the
    accumulated *y* values map through the radius scale, not through an
    outward-accumulating radius scale.  There is no polar-stack path that
    treats the stacked extent as radial accumulation: each segment's
    rendered ``inner_r`` is always taken from ``CoordPolar.inner_radius``
    (a fixed coord-level constant), so all segments start at r=0.

All tests below target the intended spec API as documented in ferrum-spec.md
§4 (channels) and §9 (CoordPolar / wind-rose / sunburst).  They will be RED
until the Python channel additions and Rust geometry changes land.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG geometry helpers
# ---------------------------------------------------------------------------


def _arc_paths(svg: str) -> list[str]:
    """Return data-mark arc paths from the SVG (wedge/arc geometry only).

    Filters to paths that:
      - contain at least one Arc command (``A``)
      - have a ``fill`` attribute that is not ``none`` (excludes axis gridlines
        and clip paths, which use ``fill="none"``)

    This avoids false positives from circular gridlines (``fill="none"``
    stroke paths) that appear in polar charts.
    """
    # Match <path ... d="..." ... fill="<color>" ...> or <path fill="<color>" ... d="...">
    # The key is: fill must not be "none" (gridlines/clip paths have fill="none").
    all_tags = re.findall(r"<path([^>]+)>", svg)
    result = []
    for attrs in all_tags:
        # Extract d= value
        d_match = re.search(r'd="([^"]+)"', attrs)
        if d_match is None:
            continue
        d = d_match.group(1)
        if "A" not in d:
            continue
        # Exclude axis/grid paths: they have fill="none"
        fill_match = re.search(r'fill="([^"]+)"', attrs)
        if fill_match is None:
            continue
        if fill_match.group(1) == "none":
            continue
        result.append(d)
    return result


def _path_first_radius(path_d: str) -> float:
    """Extract the rx (first radius) value from the first Arc command in a path.

    SVG arc syntax: ``A rx ry x-rot large-arc sweep x y``
    Returns the first ``rx`` value found in the path string.
    """
    m = re.search(r"A\s*([\d.]+)", path_d)
    if m is None:
        raise ValueError(f"no Arc command found in path: {path_d[:80]!r}")
    return float(m.group(1))


def _path_inner_outer_radii(path_d: str) -> tuple[float, float]:
    """Return (inner_radius, outer_radius) from a wedge path.

    A ferrum wedge path has the structure:
      M outer_start_x outer_start_y
      A outer_r outer_r 0 large_arc 1 outer_end_x outer_end_y
      L inner_end_x inner_end_y              <- or L cx cy for solid wedge
      A inner_r inner_r 0 large_arc 0 inner_start_x inner_start_y  (annular only)
      Z

    For a solid wedge (inner_r == 0) the path ends with ``L cx cy Z``; the
    only Arc has the outer radius.  For an annular wedge both arcs are present.

    Returns (inner_r, outer_r).  For a solid wedge, inner_r = 0.0.
    """
    arc_radii = [float(m.group(1)) for m in re.finditer(r"A\s*([\d.]+)", path_d)]
    if len(arc_radii) == 0:
        raise ValueError(f"no Arc commands in path: {path_d[:80]!r}")
    if len(arc_radii) == 1:
        # Solid wedge: only outer arc present, inner radius is 0.
        return 0.0, arc_radii[0]
    # Annular wedge: first arc = outer, second arc = inner.
    return arc_radii[1], arc_radii[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stacked_polar_df() -> pl.DataFrame:
    """Wind-rose dataset: 4 directions * 3 categories per direction.

    Each direction (N/E/S/W) has three stacked segments (A/B/C) with
    values 10/20/30.  In correct polar-stacking, the segment radii should
    accumulate: A is at r=0..10, B at r=10..30, C at r=30..60 (data units
    mapped through the radius scale).
    """
    directions = ["N", "E", "S", "W"] * 3
    categories = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    values = [10.0] * 4 + [20.0] * 4 + [30.0] * 4
    return pl.DataFrame({"direction": directions, "category": categories, "value": values})


@pytest.fixture
def annular_wedge_df() -> pl.DataFrame:
    """Dataset with explicit angular start/end and radial inner/outer extents.

    Represents two annular wedge segments; used to test theta2/radius2 channels.
    """
    return pl.DataFrame(
        {
            "theta_start": [0.0, 1.0],
            "theta_end": [1.0, 2.5],
            "r_inner": [20.0, 20.0],
            "r_outer": [80.0, 80.0],
            "category": ["A", "B"],
        }
    )


@pytest.fixture
def sunburst_df() -> pl.DataFrame:
    """Pre-laid-out hierarchy rows with theta/theta2/radius/radius2 columns.

    Simulates a two-ring sunburst:
      - Outer ring (radius 40..80): four equal slices spanning 0..tau
      - Inner ring (radius 0..40): two equal slices spanning 0..tau
    """
    import math

    tau = 2 * math.pi
    return pl.DataFrame(
        {
            "t0": [0.0, tau / 4, tau / 2, 3 * tau / 4, 0.0, tau / 2],
            "t1": [tau / 4, tau / 2, 3 * tau / 4, tau, tau / 2, tau],
            "r0": [40.0, 40.0, 40.0, 40.0, 0.0, 0.0],
            "r1": [80.0, 80.0, 80.0, 80.0, 40.0, 40.0],
            "label": ["NE", "SE", "SW", "NW", "Top", "Bottom"],
        }
    )


# ---------------------------------------------------------------------------
# D7-A: Stacked radial bars must accumulate outward (wind rose)
# ---------------------------------------------------------------------------


class TestD7StackedRadialBars:
    """Stacked polar bars should accumulate radii outward, not overlap at r=0.

    Current behaviour (BUG): all segments share inner_r = CoordPolar.inner_radius
    (a fixed coord-level constant, default 0.0).  Each segment's path has the
    same inner radius, so they all overlap at the origin instead of stacking.

    Expected behaviour: segment B's inner_r should equal segment A's outer_r;
    segment C's inner_r should equal segment A's outer_r + segment B's outer_r.
    """

    def test_theta2_channel_exists_on_ferrum_module(self) -> None:
        """fm.Theta2 must be importable — it does not exist yet.

        RED: AttributeError because Theta2 is not exported from ferrum.
        """
        assert hasattr(fm, "Theta2"), (
            "fm.Theta2 does not exist; the Theta2 channel class is missing from "
            "ferrum.encoding.positional and ferrum.__init__"
        )

    def test_radius2_channel_exists_on_ferrum_module(self) -> None:
        """fm.Radius2 must be importable — it does not exist yet.

        RED: AttributeError because Radius2 is not exported from ferrum.
        """
        assert hasattr(fm, "Radius2"), (
            "fm.Radius2 does not exist; the Radius2 channel class is missing from "
            "ferrum.encoding.positional and ferrum.__init__"
        )

    def test_stacked_polar_bar_renders_without_error(self, stacked_polar_df: pl.DataFrame) -> None:
        """A stacked polar bar chart must render to SVG without raising.

        This is a prerequisite for the geometry checks below.  If this fails
        with an error before we can inspect geometry, the geometry tests are
        also blocked.
        """
        from ferrum.position import Stack

        svg = (
            fm.Chart(stacked_polar_df)
            .mark_bar(position=Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        assert "<svg" in svg, "stacked polar bar did not produce valid SVG"

    def test_stacked_radial_bars_have_distinct_inner_radii(
        self, stacked_polar_df: pl.DataFrame
    ) -> None:
        """Stacked radial segments must render as arc wedges with distinct inner radii.

        Current (buggy) behaviour: mark_bar under CoordPolar produces Rect nodes,
        not Arc/wedge paths.  There are no arc paths with non-none fill in the SVG
        at all — the stacking produces Cartesian rect geometry that happens to be
        drawn in polar space without wedge conversion.

        Expected behaviour: each stacked segment for each angular position renders
        as an arc wedge path.  The dataset has 4 directions * 3 categories = 12
        rows; 12 wedge paths must appear, and the segments for each direction must
        have non-overlapping radial extents (inner_r > 0 for the B and C tiers).

        RED reason: ``_arc_paths()`` returns empty because the current bar renderer
        emits ``<rect>`` elements, not ``<path fill=... A...>`` wedge paths.
        Root cause: arc.rs ``build()`` reads a constant inner_radius from CoordPolar;
        bar.rs ``build()`` has no polar-specific path (it emits Rect nodes regardless
        of the coord system).
        """
        from ferrum.position import Stack

        svg = (
            fm.Chart(stacked_polar_df)
            .mark_bar(position=Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _arc_paths(svg)
        assert paths, (
            "No filled arc wedge paths in stacked polar bar SVG. "
            "mark_bar under CoordPolar must render as Arc/wedge paths, not Rect nodes. "
            "Root cause: bar.rs build() has no polar code path (always emits Rect); "
            "arc.rs build() uses CoordPolar.inner_radius as a fixed constant."
        )

        inner_radii = []
        for d in paths:
            try:
                inner_r, _ = _path_inner_outer_radii(d)
                inner_radii.append(inner_r)
            except ValueError:
                continue

        # Under the bug all inner_r values are 0.0 (or CoordPolar.inner_radius).
        # After the fix at least one segment must have inner_r > 1 px.
        non_zero_inner = [r for r in inner_radii if r > 1.0]
        assert non_zero_inner, (
            f"All stacked polar segments have inner_r <= 1.0: {inner_radii}. "
            "Stacked segments must accumulate outward (wind-rose / coxcomb). "
            "Root cause: apply_stack() Cartesian accumulation is used unchanged "
            "under CoordPolar; the arc builder always uses CoordPolar.inner_radius "
            "as the constant inner extent for every segment (arc.rs build())."
        )

    def test_stacked_radial_bars_segments_are_strictly_ordered(
        self, stacked_polar_df: pl.DataFrame
    ) -> None:
        """For a single angular position, stacked segments must not overlap.

        For each angular bin (direction), the wedge segments for categories A, B,
        C should have non-overlapping radial extents: outer_r(A) == inner_r(B)
        and outer_r(B) == inner_r(C) within a small tolerance.

        This test uses only a single direction to isolate one angular column.
        Under the bug, all three segments would have inner_r == 0, outer_r == 10,
        outer_r == 30, outer_r == 60 respectively (all starting from 0).
        """
        from ferrum.position import Stack

        # Single direction to isolate one angular bin.
        single_dir = stacked_polar_df.filter(pl.col("direction") == "N")
        svg = (
            fm.Chart(single_dir)
            .mark_bar(position=Stack())
            .encode(x="direction:N", y="value:Q", color="category:N")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _arc_paths(svg)
        assert paths, "no arc paths in single-direction stacked polar SVG"

        # Collect (inner_r, outer_r) for the three segments sorted by inner_r.
        segments: list[tuple[float, float]] = []
        for d in paths:
            try:
                inner_r, outer_r = _path_inner_outer_radii(d)
                segments.append((inner_r, outer_r))
            except ValueError:
                continue

        assert len(segments) >= 2, (
            f"Expected at least 2 arc segments for stacked bars, got {len(segments)}"
        )
        segments.sort()

        # Adjacent segments must share a boundary: outer_r[i] == inner_r[i+1].
        for i in range(len(segments) - 1):
            _, outer = segments[i]
            inner_next, _ = segments[i + 1]
            assert abs(outer - inner_next) < 2.0, (
                f"Segment {i} outer_r={outer:.2f} != segment {i + 1} inner_r={inner_next:.2f}: "
                "segments are not contiguous — they all start at r=0 (overlap bug). "
                "The stack position pass must set inner_r = previous segment's outer_r."
            )


# ---------------------------------------------------------------------------
# D7-B: Theta2 / Radius2 channels for annular wedge rendering
# ---------------------------------------------------------------------------


class TestD7AnnularWedge:
    """Theta2 and Radius2 channels must exist and drive wedge geometry.

    Current behaviour (BUG): Neither channel class exists.  Attempting
    ``encode(theta2="field")`` on a chart either raises AttributeError
    (when using fm.Theta2) or silently drops the channel (when passed as a
    string shorthand with no class backing it).  The rendered arc ignores
    the data-driven angular end and always sweeps to the computed total,
    and uses CoordPolar's fixed inner_radius rather than a data column.

    Expected behaviour after fix:
      - ``fm.Theta2("theta_end")`` encodes the angular end position from data.
      - ``fm.Radius("r_inner")`` + ``fm.Radius2("r_outer")`` encode per-row
        inner and outer radii from data columns.
      - Each rendered path describes an annular wedge spanning
        [theta, theta2] angularly and [radius, radius2] radially.
    """

    def test_theta2_class_importable_from_encoding_module(self) -> None:
        """ferrum.encoding.Theta2 must be importable.

        RED: ImportError / AttributeError because Theta2 is absent from
        ferrum/encoding/positional.py and ferrum/encoding/__init__.py.
        """
        from ferrum import encoding as enc

        assert hasattr(enc, "Theta2"), (
            "ferrum.encoding.Theta2 does not exist. "
            "Add Theta2 channel class to ferrum/encoding/positional.py and "
            "export it from ferrum/encoding/__init__.py."
        )

    def test_radius2_class_importable_from_encoding_module(self) -> None:
        """ferrum.encoding.Radius2 must be importable.

        RED: ImportError / AttributeError because Radius2 is absent from
        ferrum/encoding/positional.py and ferrum/encoding/__init__.py.
        """
        from ferrum import encoding as enc

        assert hasattr(enc, "Radius2"), (
            "ferrum.encoding.Radius2 does not exist. "
            "Add Radius2 channel class to ferrum/encoding/positional.py and "
            "export it from ferrum/encoding/__init__.py."
        )

    def test_theta2_has_correct_channel_name(self) -> None:
        """Theta2._channel_name must be 'theta2' to round-trip through to_spec().

        RED: AttributeError because Theta2 does not exist.
        """
        Theta2 = getattr(fm, "Theta2", None)
        assert Theta2 is not None, "fm.Theta2 does not exist"
        ch = Theta2("some_field")
        assert ch._channel_name == "theta2", (
            f"Theta2._channel_name == {ch._channel_name!r}, expected 'theta2'"
        )

    def test_radius2_has_correct_channel_name(self) -> None:
        """Radius2._channel_name must be 'radius2' to round-trip through to_spec().

        RED: AttributeError because Radius2 does not exist.
        """
        Radius2 = getattr(fm, "Radius2", None)
        assert Radius2 is not None, "fm.Radius2 does not exist"
        ch = Radius2("some_field")
        assert ch._channel_name == "radius2", (
            f"Radius2._channel_name == {ch._channel_name!r}, expected 'radius2'"
        )

    def test_annular_wedge_renders_without_error(self, annular_wedge_df: pl.DataFrame) -> None:
        """An annular wedge chart using Theta2/Radius/Radius2 must render to SVG.

        RED: AttributeError on fm.Theta2 / fm.Radius2 before any rendering occurs.
        """
        Theta2 = getattr(fm, "Theta2", None)
        Radius2 = getattr(fm, "Radius2", None)
        assert Theta2 is not None, "fm.Theta2 does not exist — cannot run this test"
        assert Radius2 is not None, "fm.Radius2 does not exist — cannot run this test"

        svg = (
            fm.Chart(annular_wedge_df)
            .mark_arc()
            .encode(
                theta=fm.Theta("theta_start"),
                theta2=Theta2("theta_end"),
                radius=fm.Radius("r_inner"),
                radius2=Radius2("r_outer"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        assert "<svg" in svg, "annular wedge chart did not produce valid SVG"

    def test_annular_wedge_paths_have_nonzero_inner_radius(
        self, annular_wedge_df: pl.DataFrame
    ) -> None:
        """Annular wedges bound via Radius/Radius2 must have inner_r > 0.

        With r_inner=20 in the data, each rendered wedge path must show an
        inner arc at ~20 mapped pixels, not at the CoordPolar.inner_radius
        default of 0.

        RED: AttributeError on fm.Theta2 / fm.Radius2 before any rendering.
        After channels land but before Rust geometry fix: inner_r == 0.
        """
        Theta2 = getattr(fm, "Theta2", None)
        Radius2 = getattr(fm, "Radius2", None)
        assert Theta2 is not None, "fm.Theta2 does not exist"
        assert Radius2 is not None, "fm.Radius2 does not exist"

        svg = (
            fm.Chart(annular_wedge_df)
            .mark_arc()
            .encode(
                theta=fm.Theta("theta_start"),
                theta2=Theta2("theta_end"),
                radius=fm.Radius("r_inner"),
                radius2=Radius2("r_outer"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _arc_paths(svg)
        assert paths, "no arc paths in annular wedge SVG"

        for d in paths:
            try:
                inner_r, outer_r = _path_inner_outer_radii(d)
            except ValueError:
                continue
            assert inner_r > 1.0, (
                f"Annular wedge path has inner_r={inner_r:.2f} <= 1.0. "
                "The radius channel (r_inner=20) must set a non-zero inner radius "
                "on each arc path. Root cause: arc.rs uses CoordPolar.inner_radius "
                "(a coord-level constant) instead of the per-row radius column."
            )

    def test_annular_wedge_does_not_span_full_circle(self, annular_wedge_df: pl.DataFrame) -> None:
        """Each wedge path should span only a partial arc (theta to theta2).

        With theta_start=[0, 1] and theta_end=[1, 2.5], no single wedge
        spans the full circle.  Under the current arc.rs implementation that
        computes sweep from the total sum of the theta column, the sweep
        ratios would be wrong (the arc sweeps the proportional fraction of
        the total sum, ignoring the explicit theta2 data column entirely).

        After the fix, each path should span [theta_start, theta_end] radians
        directly, not a fraction of tau computed from column sums.

        RED: AttributeError on fm.Theta2 before rendering.
        After channels land but Rust geometry is unchanged: paths span the
        wrong angular range (sum-proportional sweep, not data-column span).
        """
        Theta2 = getattr(fm, "Theta2", None)
        Radius2 = getattr(fm, "Radius2", None)
        assert Theta2 is not None, "fm.Theta2 does not exist"
        assert Radius2 is not None, "fm.Radius2 does not exist"

        svg = (
            fm.Chart(annular_wedge_df)
            .mark_arc()
            .encode(
                theta=fm.Theta("theta_start"),
                theta2=Theta2("theta_end"),
                radius=fm.Radius("r_inner"),
                radius2=Radius2("r_outer"),
                color="category:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _arc_paths(svg)
        assert paths, "no arc paths in SVG — cannot check angular span"

        # Each path should not be a full circle (large_arc=1 for both arcs
        # and the start/end points coinciding would indicate full-circle rendering).
        # We check by counting paths: there should be at least 2 distinct wedges
        # rather than one dominant full-circle arc.
        assert len(paths) >= 2, (
            f"Expected at least 2 arc paths for 2 data rows, got {len(paths)}. "
            "If Theta2 is silently dropped, the arc builder falls back to a "
            "single proportional sweep computed from the theta column sum."
        )


# ---------------------------------------------------------------------------
# D7-C: Hand-built sunburst from pre-laid-out hierarchy rows
# ---------------------------------------------------------------------------


class TestD7Sunburst:
    """A pre-laid-out sunburst dataset with theta/theta2/radius/radius2 columns
    must render nested wedge rings, not a single flat pie layer.

    Current behaviour (BUG): Theta2 and Radius2 are absent, so the chart
    either raises or silently uses only the theta column and CoordPolar's
    fixed inner_radius.  The two rings cannot be distinguished in the output.

    Expected behaviour: six arc paths are rendered, four in the outer ring
    (r0=40..r1=80) and two in the inner ring (r0=0..r1=40).  The outer ring
    paths must have inner_r > 1 px and the inner ring paths must have
    inner_r near 0.
    """

    def test_sunburst_renders_correct_number_of_wedges(self, sunburst_df: pl.DataFrame) -> None:
        """Six pre-laid-out rows must produce six distinct arc wedge paths.

        RED: AttributeError on fm.Theta2 / fm.Radius2 before rendering.
        After channels land but before Rust geometry: may produce only 1–2
        arcs because the proportional sweep collapses rows.
        """
        Theta2 = getattr(fm, "Theta2", None)
        Radius2 = getattr(fm, "Radius2", None)
        assert Theta2 is not None, "fm.Theta2 does not exist"
        assert Radius2 is not None, "fm.Radius2 does not exist"

        svg = (
            fm.Chart(sunburst_df)
            .mark_arc()
            .encode(
                theta=fm.Theta("t0"),
                theta2=Theta2("t1"),
                radius=fm.Radius("r0"),
                radius2=Radius2("r1"),
                color="label:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _arc_paths(svg)
        assert len(paths) == 6, (
            f"Expected 6 arc paths for 6 sunburst rows, got {len(paths)}. "
            "Each pre-laid-out hierarchy row must produce its own wedge path "
            "spanning [t0, t1] angularly and [r0, r1] radially."
        )

    def test_sunburst_outer_ring_has_larger_inner_radius_than_inner_ring(
        self, sunburst_df: pl.DataFrame
    ) -> None:
        """Outer ring wedges (r0=40) must have inner_r > inner ring (r0=0).

        Radial distinction between the two rings is the core property of a
        sunburst.  Under the bug, all six paths have inner_r == 0.

        RED: AttributeError on fm.Theta2 / fm.Radius2 before rendering.
        """
        Theta2 = getattr(fm, "Theta2", None)
        Radius2 = getattr(fm, "Radius2", None)
        assert Theta2 is not None, "fm.Theta2 does not exist"
        assert Radius2 is not None, "fm.Radius2 does not exist"

        svg = (
            fm.Chart(sunburst_df)
            .mark_arc()
            .encode(
                theta=fm.Theta("t0"),
                theta2=Theta2("t1"),
                radius=fm.Radius("r0"),
                radius2=Radius2("r1"),
                color="label:N",
            )
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _arc_paths(svg)
        assert paths, "no arc paths in sunburst SVG"

        inner_radii = []
        for d in paths:
            try:
                inner_r, _ = _path_inner_outer_radii(d)
                inner_radii.append(inner_r)
            except ValueError:
                continue

        # Outer ring rows have r0=40 → inner_r should be > 1 px.
        # Inner ring rows have r0=0 → inner_r should be <= 1 px.
        # At minimum: the set of inner_r values must not be all the same.
        unique_inner = set(round(r, 1) for r in inner_radii)
        assert len(unique_inner) > 1, (
            f"All arc paths have the same inner_r (unique values: {unique_inner}). "
            "The outer ring (r0=40) and inner ring (r0=0) must produce paths with "
            "different inner radii. Root cause: arc.rs reads CoordPolar.inner_radius "
            "as a fixed constant instead of the per-row radius (r0) column."
        )
