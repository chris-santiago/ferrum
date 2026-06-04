"""FA-1: mark_arc(theta:N, radius:Q) must render non-empty arc wedges.

Root cause (identified 2026-06-02):
  ``arc.rs build()`` calls ``col_as_f64(batch, theta_field)`` on the theta
  channel. When theta is a nominal (string) column, ``col_as_f64`` fails
  with a cast error and the renderer returns an empty result.

Fix chosen: option (a) — detect nominal/ordinal theta in ``build()`` and
  route to equal-band angular layout (same machinery as ``build_polar`` in
  ``bar.rs``). Each category gets an equal angular slice; the radius channel
  determines the outer radius for that slice.  This enables::

      fm.Chart(df).mark_arc().encode(theta="cat:N", radius="val:Q").coord(...)

  and produces the same visual result as the mark_bar + CoordPolar coxcomb
  path that FA-2 fixes.

Test strategy: assert N arc paths with correct radii are produced; a blank
  render is NOT acceptable.
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


def _outer_radius_from_path(path_d: str) -> float:
    """Return the outer arc radius from the first A command in a path."""
    m = re.search(r"A\s*([\d.]+)", path_d)
    return float(m.group(1)) if m else math.nan


def _arc_sweep_radians(path_d: str) -> float:
    """Compute the sweep angle of the first arc in a wedge path (same as FA-2 helper).

    Clamps ratio to [0, 1] before asin; returns NaN only when chord clearly
    exceeds 2r by > 1% (indicating corrupt geometry from a bad transform).
    """
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
    if ratio > 1.01:
        return math.nan
    return 2.0 * math.asin(min(ratio, 1.0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_cat_df() -> pl.DataFrame:
    """2-category dataset for nominal-theta arc coxcomb."""
    return pl.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})


@pytest.fixture
def four_cat_df() -> pl.DataFrame:
    """4-category dataset for nominal-theta arc coxcomb."""
    return pl.DataFrame({"cat": ["N", "E", "S", "W"], "val": [10.0, 15.0, 20.0, 8.0]})


# ---------------------------------------------------------------------------
# FA-1 Tests (option a: nominal theta → equal-band arc wedges)
# ---------------------------------------------------------------------------


class TestFA1ArcNominalTheta:
    """mark_arc with a nominal theta channel must render non-empty arc wedges.

    Before the fix: ``col_as_f64`` fails on a string column → empty render.
    After the fix: equal-band layout gives each category an equal angular slice
    and the radius channel determines the outer radius.
    """

    def test_two_cat_arc_nominal_theta_renders_nonempty(self, two_cat_df: pl.DataFrame) -> None:
        """mark_arc(theta:N, radius:Q) must produce a non-empty SVG.

        RED before fix: col_as_f64 fails on string column → empty SceneGraph
        (no arc paths in the SVG).
        """
        svg = (
            fm.Chart(two_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert paths, (
            "mark_arc with nominal theta rendered blank (zero filled arc paths). "
            "Root cause: arc.rs build() calls col_as_f64 on string theta column → "
            "cast error → empty result. Fix: detect nominal/ordinal theta and route "
            "to equal-band angular layout."
        )

    def test_two_cat_arc_produces_two_paths(self, two_cat_df: pl.DataFrame) -> None:
        """2 categories must produce exactly 2 arc wedge paths."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2, (
            f"Expected 2 arc paths for 2-category nominal-theta arc, got {len(paths)}."
        )

    def test_four_cat_arc_produces_four_paths(self, four_cat_df: pl.DataFrame) -> None:
        """4 categories must produce exactly 4 arc wedge paths."""
        svg = (
            fm.Chart(four_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 4, (
            f"Expected 4 arc paths for 4-category nominal-theta arc, got {len(paths)}."
        )

    def test_arc_radii_reflect_data_values(self, two_cat_df: pl.DataFrame) -> None:
        """The outer radius of each wedge must reflect the radius channel value.

        val=[10, 20]: the wedge for val=20 must have outer_r > wedge for val=10.
        """
        svg = (
            fm.Chart(two_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"

        radii = [_outer_radius_from_path(d) for d in paths]
        assert all(math.isfinite(r) for r in radii), f"Could not extract outer radii: {radii}"

        # The two radii must be distinct (val=10 vs val=20 → different outer r).
        assert radii[0] != radii[1], (
            f"Both arcs have the same outer radius {radii[0]:.2f}; "
            "the radius channel (val:Q) must produce distinct outer radii."
        )
        # Their ratio should be approximately 1:2 (10:20).
        r_min, r_max = sorted(radii)
        ratio = r_max / r_min
        assert abs(ratio - 2.0) < 0.3, (
            f"Radius ratio = {ratio:.2f}, expected ~2.0 (val=10 vs val=20). Radii: {radii}"
        )

    def test_arc_sweeps_are_equal(self, two_cat_df: pl.DataFrame) -> None:
        """Both wedges must have equal angular sweeps (equal-band layout)."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 2

        sweeps = [_arc_sweep_radians(d) for d in paths]
        assert all(math.isfinite(s) for s in sweeps), (
            f"NaN sweep detected: {sweeps}. Points not on arc circle."
        )
        assert abs(sweeps[0] - sweeps[1]) < 0.1, (
            f"Nominal-theta arc wedges have unequal sweeps: "
            f"{[math.degrees(s) for s in sweeps]}°. "
            "Both categories should get equal angular bands."
        )

    def test_four_cat_arc_sweeps_are_equal(self, four_cat_df: pl.DataFrame) -> None:
        """All 4 wedges must have equal angular sweeps (pi/2 each)."""
        svg = (
            fm.Chart(four_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        assert len(paths) == 4

        sweeps = [_arc_sweep_radians(d) for d in paths]
        assert all(math.isfinite(s) for s in sweeps), f"NaN sweeps in 4-cat arc: {sweeps}"

        half_pi = math.pi / 2.0
        for i, sweep in enumerate(sweeps):
            assert abs(sweep - half_pi) < 0.1, (
                f"Wedge {i} sweep = {math.degrees(sweep):.1f}°, expected 90°."
            )

    def test_arc_nominal_theta_total_sweep_near_tau(self, two_cat_df: pl.DataFrame) -> None:
        """Sum of wedge sweeps must be approximately 2pi (full circle coverage)."""
        svg = (
            fm.Chart(two_cat_df)
            .mark_arc()
            .encode(theta="cat:N", radius="val:Q")
            .coord(fm.CoordPolar(theta="x"))
            .to_svg()
        )
        paths = _filled_arc_paths(svg)
        sweeps = [_arc_sweep_radians(d) for d in paths]
        valid = [s for s in sweeps if math.isfinite(s)]
        total = sum(valid)
        assert abs(total - math.tau) < 0.2, (
            f"Total sweep = {math.degrees(total):.1f}°, expected 360°."
        )
