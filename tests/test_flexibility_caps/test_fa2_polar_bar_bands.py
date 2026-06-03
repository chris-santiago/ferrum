"""FA-2: Polar bar angular layout must produce equal full-circle bands.

Root cause (identified 2026-06-02):
  ``build_polar`` in ``bar.rs`` generates correctly positioned arc-wedge
  ``SceneNode::Path`` nodes, but ``scene_build.rs`` then applies
  ``apply_polar_node_transform`` to them because the exclusion gate checks
  ``layer.mark == Mark::Arc``, not ``result.kind == MarkBatchKind::Arc``.
  The double-transform distorts the arc paths.

Fix: extend the exclusion condition in ``scene_build.rs`` to also skip the
  polar-node transform when ``result.kind == MarkBatchKind::Arc`` (i.e. when
  ``build_polar`` returned arc geometry).

These tests target the intended geometry: N categories must produce N equal
angular wedges spanning the full 2pi, each occupying tau/N radians.
"""

from __future__ import annotations

import math
import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG geometry helpers
# ---------------------------------------------------------------------------


def _filled_arc_paths(svg: str) -> list[str]:
    """Return ``d=`` values for <path> elements with arc commands and non-none fill."""
    result = []
    for attrs in re.findall(r"<path([^>]+)>", svg):
        d_m = re.search(r'd="([^"]+)"', attrs)
        if d_m is None:
            continue
        d = d_m.group(1)
        if "A" not in d:
            continue
        fill_m = re.search(r'fill="([^"]+)"', attrs)
        if fill_m is None or fill_m.group(1) == "none":
            continue
        result.append(d)
    return result


def _arc_sweep_radians(path_d: str) -> float:
    """Compute the angular sweep of the first arc in a wedge path.

    A wedge path structure:
      M ox0 oy0  (outer start)
      A outer_r ... ox1 oy1  (outer arc)
      L ix cy  (inner point / center)
      [inner arc or Z]

    We recover the sweep by computing the angle subtended by the chord
    from ox0 to ox1 on the circle of the given radius.  For two points on
    a circle of radius R separated by chord length c, the central angle is
    2 * arcsin(c / (2R)).

    Because ``build_polar`` paths are SVG-valid the arc radius is always
    consistent with the endpoint distance (the start/end ARE on the circle),
    except under the double-transform bug where they are not.  If the
    chord significantly exceeds 2R (bugged path) we return NaN.

    Note: for the half-circle case the chord equals the diameter (2R exactly),
    so floating-point gives ratio ~= 1.0 ± epsilon; we clamp to [0, 1] before
    asin to handle this robustly.  We only return NaN when the chord clearly
    exceeds the diameter by more than 1% (indicating genuinely corrupt geometry).
    """
    # Extract: outer_r from first A command, (ox0,oy0) from M, (ox1,oy1) from A endpoint.
    m_match = re.match(r"M\s*([\d.]+)\s+([\d.]+)", path_d.strip())
    a_match = re.search(
        r"A\s*([\d.]+)\s+[\d.]+\s+[\d.]+\s+[01]\s+[01]\s+([\d.]+)\s+([\d.]+)", path_d
    )
    if m_match is None or a_match is None:
        return math.nan
    ox0 = float(m_match.group(1))
    oy0 = float(m_match.group(2))
    r = float(a_match.group(1))
    ox1 = float(a_match.group(2))
    oy1 = float(a_match.group(3))
    chord = math.sqrt((ox1 - ox0) ** 2 + (oy1 - oy0) ** 2)
    ratio = chord / (2.0 * r)
    # More than 1% over the diameter means genuinely corrupt geometry (wrong transform).
    if ratio > 1.01:
        return math.nan
    ratio = min(ratio, 1.0)
    return 2.0 * math.asin(ratio)


def _arc_center(path_d: str) -> tuple[float, float]:
    """Return the (cx, cy) center from the L-to-center command in a solid wedge.

    A solid wedge ends with: L cx cy Z (line to center, then close).
    """
    l_match = re.search(r"L\s*([\d.]+)\s+([\d.]+)", path_d)
    if l_match is None:
        return math.nan, math.nan
    return float(l_match.group(1)), float(l_match.group(2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_cat_df() -> pl.DataFrame:
    """2-category coxcomb: each cat should occupy exactly 180° (pi radians)."""
    return pl.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})


@pytest.fixture
def four_cat_df() -> pl.DataFrame:
    """4-category coxcomb: each cat should occupy exactly 90° (pi/2 radians)."""
    return pl.DataFrame({"cat": ["N", "E", "S", "W"], "val": [10.0, 15.0, 20.0, 8.0]})


# ---------------------------------------------------------------------------
# FA-2 Tests
# ---------------------------------------------------------------------------


class TestFA2PolarBarBands:
    """mark_bar + CoordPolar must produce N equal angular wedges spanning 2pi."""

    # --- 2-category ---

    def test_two_cat_renders_two_filled_arc_paths(self, two_cat_df: pl.DataFrame) -> None:
        """2 categories must produce exactly 2 filled arc wedge paths."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2, (
            f"Expected 2 filled arc paths for 2-category coxcomb, got {len(paths)}.\n"
            "Bug: build_polar output goes through apply_polar_node_transform (double-transform)."
        )

    def test_two_cat_sweeps_are_equal(self, two_cat_df: pl.DataFrame) -> None:
        """Both wedges in a 2-category coxcomb must have equal angular sweeps."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"

        sweeps = [_arc_sweep_radians(d) for d in paths]
        assert all(math.isfinite(s) for s in sweeps), (
            f"Arc sweep computation returned NaN for paths: {paths}\n"
            "This usually means the start/end points are not on the arc circle "
            "(double-transform bug: the wedge path coordinates are distorted)."
        )
        assert abs(sweeps[0] - sweeps[1]) < 0.05, (
            f"Wedge sweeps differ: {[math.degrees(s) for s in sweeps]}°. Both should be 180°."
        )

    def test_two_cat_each_sweep_is_pi(self, two_cat_df: pl.DataFrame) -> None:
        """Each of 2 wedges should span pi radians (180°)."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2, f"Expected 2 paths"

        pi = math.pi
        for d in paths:
            sweep = _arc_sweep_radians(d)
            assert math.isfinite(sweep), (
                f"NaN sweep for path {d[:80]!r}: points not on the arc circle "
                "(double-transform corrupts wedge geometry)."
            )
            assert abs(sweep - pi) < 0.1, (
                f"Wedge sweep = {math.degrees(sweep):.1f}°, expected 180°. "
                "2 categories must each get half the circle."
            )

    def test_two_cat_total_sweep_near_tau(self, two_cat_df: pl.DataFrame) -> None:
        """Sum of all wedge sweeps must be approximately 2pi (full circle)."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        sweeps = [_arc_sweep_radians(d) for d in paths]
        valid = [s for s in sweeps if math.isfinite(s)]
        total = sum(valid)
        assert abs(total - math.tau) < 0.2, (
            f"Total wedge sweep = {math.degrees(total):.1f}°, expected 360°. "
            f"Individual sweeps: {[math.degrees(s) for s in sweeps]}°."
        )

    # --- 4-category ---

    def test_four_cat_renders_four_filled_arc_paths(self, four_cat_df: pl.DataFrame) -> None:
        """4 categories must produce exactly 4 filled arc wedge paths."""
        svg = (
            fm.Chart(four_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 4, (
            f"Expected 4 filled arc paths for 4-category coxcomb, got {len(paths)}."
        )

    def test_four_cat_sweeps_are_equal(self, four_cat_df: pl.DataFrame) -> None:
        """All 4 wedges must have equal angular sweeps (pi/2 each)."""
        svg = (
            fm.Chart(four_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 4, f"Expected 4 paths"

        sweeps = [_arc_sweep_radians(d) for d in paths]
        assert all(math.isfinite(s) for s in sweeps), (
            f"NaN sweep detected in 4-cat coxcomb: sweeps={sweeps}.\n"
            "Double-transform bug distorts arc geometry."
        )

        half_pi = math.pi / 2.0
        for i, sweep in enumerate(sweeps):
            assert abs(sweep - half_pi) < 0.1, (
                f"Wedge {i} sweep = {math.degrees(sweep):.1f}°, expected 90°. "
                "4 categories must each get a quarter circle."
            )

    def test_four_cat_total_sweep_near_tau(self, four_cat_df: pl.DataFrame) -> None:
        """Sum of all 4 wedge sweeps must be approximately 2pi."""
        svg = (
            fm.Chart(four_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        sweeps = [_arc_sweep_radians(d) for d in paths]
        valid = [s for s in sweeps if math.isfinite(s)]
        total = sum(valid)
        assert abs(total - math.tau) < 0.2, (
            f"Total sweep = {math.degrees(total):.1f}°, expected 360°."
        )

    def test_two_cat_all_centers_coincide(self, two_cat_df: pl.DataFrame) -> None:
        """All wedges must share the same center (cx, cy)."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .show_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2
        centers = [_arc_center(d) for d in paths]
        cx0, cy0 = centers[0]
        for cx, cy in centers[1:]:
            assert abs(cx - cx0) < 1.0, f"Wedge centers have different cx: {centers}"
            assert abs(cy - cy0) < 1.0, f"Wedge centers have different cy: {centers}"
