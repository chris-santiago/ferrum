"""Regression + behavioral tests for the continuous-axis scale-projection fix.

Before this fix (feat/render-gaps-17-19-21), continuous-axis major ticks and
gridlines were placed using uniform slot centers derived from the tick-label
count (``panel.w / n`` per slot, ignoring the padded inset).  Data marks are
placed by ``scale.to_pixel(value)``, which does account for the padding inset.
The two coordinate systems diverged: a gridline at value v did NOT land on the
same pixel as a data mark at value v.

After the fix, continuous-axis ticks derive their pixel positions from the same
scale projection used by data marks (via ``tick_data()`` in the Rust layer), so
a gridline at value v and a data mark at value v share a pixel coordinate.
Categorical (ordinal/band) axes keep their correct uniform-slot placement.

Reference: design-docs/superpowers/specs/2026-05-30-continuous-axis-scale-projection-design.md
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum import X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vertical_gridline_xs(svg: str) -> list[float]:
    """Return sorted x-coordinates of all vertical gridlines in the SVG.

    Vertical gridlines are ``<line>`` elements where ``x1 == x2`` (within
    floating-point tolerance) and the vertical span is large (>50px), which
    rules out the short axis tick-marks themselves.
    """
    xs: list[float] = []
    for m in re.finditer(r"<line([^/]+)/>", svg):
        attrs = dict(re.findall(r"([\w-]+)=\"([^\"]+)\"", m.group(1)))
        try:
            x1 = float(attrs["x1"])
            x2 = float(attrs["x2"])
            y1 = float(attrs["y1"])
            y2 = float(attrs["y2"])
        except (KeyError, ValueError):
            continue
        if abs(x1 - x2) < 0.5 and abs(y2 - y1) > 50:
            xs.append(x1)
    return sorted(xs)


def _circle_cxs(svg: str) -> list[float]:
    """Return sorted cx values from all ``<circle>`` elements in the SVG."""
    return sorted(float(v) for v in re.findall(r'<circle[^>]*cx="([^"]+)"', svg))


# ---------------------------------------------------------------------------
# Test 1: Tick ↔ mark coincidence on a linear axis (the core regression guard)
# ---------------------------------------------------------------------------


class TestLinearAxisTickMarkCoincidence:
    """Verify that on a linear continuous axis the first and last gridlines land
    at the same pixel as the data marks at the domain extremes.

    This is the canonical regression test for the continuous-axis scale-
    projection fix.  It **must fail** against old uniform-slot placement, where
    the first gridline was at ``panel_x + slot_w / 2`` (ignoring the inset) and
    the last was at ``panel_x + panel_w - slot_w / 2``.  For a 600px-wide chart
    that difference is ~34px at the minimum, which is detectable at ±2px.

    Design reference: §9 acceptance criteria — 'On a linear axis, major gridline
    pixels equal the scale-projected pixels of the tick values and coincide with
    data marks of those values.'
    """

    _WIDTH = 600
    _HEIGHT = 400
    _TOLERANCE = 2.0  # pixels; old misalignment was ~34px so this is tight enough

    def _build_svg(self) -> str:
        """Scatter chart with data at x-domain extremes (x=0 and x=100)."""
        df = pl.DataFrame(
            {
                "x": [0.0, 50.0, 100.0],
                "y": [10.0, 20.0, 30.0],
            }
        )
        return (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(width=self._WIDTH, height=self._HEIGHT)
            .show_svg()
        )

    def test_leftmost_gridline_coincides_with_domain_min_mark(self) -> None:
        """Vertical gridline at the left edge equals cx of the x=0 data mark.

        Before the fix: the first gridline was at slot_center ≈ panel_x + slot_w/2
        (no inset awareness), while the data mark at x=0 was at the padded domain
        extent (~61.5px in).  These two diverged by ~34px.

        After the fix: both the gridline and the data mark at x=0 are produced by
        the same scale projection, so they share a pixel within floating-point
        precision.
        """
        svg = self._build_svg()
        gridlines = _vertical_gridline_xs(svg)
        circles = _circle_cxs(svg)

        assert gridlines, "No vertical gridlines found in SVG"
        assert circles, "No circles found in SVG"

        leftmost_gridline = gridlines[0]
        leftmost_circle = circles[0]  # sorted; x=0 maps to smallest cx

        assert abs(leftmost_gridline - leftmost_circle) <= self._TOLERANCE, (
            f"Left gridline ({leftmost_gridline:.3f}) does not coincide with the "
            f"data mark at x=0 (cx={leftmost_circle:.3f}).  "
            f"Difference: {abs(leftmost_gridline - leftmost_circle):.3f}px.  "
            f"This suggests continuous gridlines are still using uniform-slot "
            f"placement rather than scale projection."
        )

    def test_rightmost_gridline_coincides_with_domain_max_mark(self) -> None:
        """Vertical gridline at the right edge equals cx of the x=100 data mark."""
        svg = self._build_svg()
        gridlines = _vertical_gridline_xs(svg)
        circles = _circle_cxs(svg)

        assert gridlines, "No vertical gridlines found in SVG"
        assert circles, "No circles found in SVG"

        rightmost_gridline = gridlines[-1]
        rightmost_circle = circles[-1]  # x=100 maps to largest cx

        assert abs(rightmost_gridline - rightmost_circle) <= self._TOLERANCE, (
            f"Right gridline ({rightmost_gridline:.3f}) does not coincide with "
            f"the data mark at x=100 (cx={rightmost_circle:.3f}).  "
            f"Difference: {abs(rightmost_gridline - rightmost_circle):.3f}px."
        )

    def test_middle_gridline_coincides_with_midpoint_mark(self) -> None:
        """Gridlines at the mid-value (x=50) coincide with the mark at x=50."""
        svg = self._build_svg()
        gridlines = _vertical_gridline_xs(svg)
        circles = _circle_cxs(svg)

        assert len(circles) >= 3, f"Expected 3 circles; got {len(circles)}"

        mid_circle = circles[1]  # sorted; x=50 is the middle value

        # The mid-point gridline should exist within tolerance of the mid circle.
        closest_gridline = min(gridlines, key=lambda g: abs(g - mid_circle))
        assert abs(closest_gridline - mid_circle) <= self._TOLERANCE, (
            f"No gridline near the x=50 mark (cx={mid_circle:.3f}).  "
            f"Closest gridline: {closest_gridline:.3f}px away.  "
            f"Difference: {abs(closest_gridline - mid_circle):.3f}px."
        )

    def test_old_uniform_slot_placement_would_fail_this_test(self) -> None:
        """Self-documenting: compute what old uniform-slot positions would have been.

        For a 600px-wide chart with 11 gridlines and no inset awareness:
          slot_w = 600 / 11 ≈ 54.5px
          first slot center ≈ 27.3px
        The data mark at x=0 is at ~61.5px (inset-padded).
        The difference is ~34px, which is far outside the 2px tolerance above.
        This test asserts that the difference so it can never be silently
        compressed to zero without someone noticing.
        """
        svg = self._build_svg()
        circles = _circle_cxs(svg)
        gridlines = _vertical_gridline_xs(svg)
        assert circles and gridlines

        leftmost_circle = circles[0]

        # Reconstruct what old code would have produced:
        # total panel pixel span (without inset) = WIDTH
        # n_gridlines from the actual render (we just want the count)
        n = len(gridlines)
        old_slot_w = self._WIDTH / n
        old_first_gridline = old_slot_w / 2

        # The old first gridline must differ significantly from the data mark.
        old_error = abs(old_first_gridline - leftmost_circle)
        assert old_error > 20, (
            f"Expected old uniform-slot placement to differ from the mark by >20px; "
            f"got {old_error:.1f}px.  If this fails, the test domain or width may "
            f"need adjusting so the inset-mismatch is visible."
        )


# ---------------------------------------------------------------------------
# Test 2: Log scale — gridlines coincide with scale-projected mark positions
# ---------------------------------------------------------------------------


class TestLogAxisTickMarkCoincidence:
    """For a log-scale x axis, gridlines must coincide with scale-projected mark positions.

    On a log scale spanning decades (e.g. 1–100), the tick values at 1 and 100
    project to the padded domain extremes.  Old uniform-slot code (n slots,
    panel_w / n per slot) would have placed the first gridline at
    ``panel_x + slot_w/2`` — far from the actual data mark at x=1 which is at
    the padded left edge.

    After the fix, every tick pixel comes from ``scale.to_pixel(value)``, so:
    - the gridline at value 1 sits at the same pixel as a data mark at x=1
    - the gridline at value 100 sits at the same pixel as a data mark at x=100

    Tick labels on a log scale are geometrically spaced (1, 1.58, 2.51, …, 100)
    rather than linearly spaced (33, 67, 100), which is verified separately.
    """

    _WIDTH = 600
    _HEIGHT = 400
    _TOLERANCE = 2.0

    def _build_svg(self) -> str:
        df = pl.DataFrame(
            {
                "x": [1.0, 10.0, 100.0],
                "y": [1.0, 2.0, 3.0],
            }
        )
        return (
            fm.Chart(df)
            .mark_point()
            .encode(x=X("x:Q", scale={"type": "log"}), y="y:Q")
            .properties(width=self._WIDTH, height=self._HEIGHT)
            .show_svg()
        )

    def test_log_leftmost_gridline_coincides_with_x_min_mark(self) -> None:
        """Gridline at the log-scale left edge coincides with the mark at x=1."""
        svg = self._build_svg()
        gridlines = _vertical_gridline_xs(svg)
        circles = _circle_cxs(svg)

        assert gridlines, "No vertical gridlines in log-scale SVG"
        assert circles, "No circles in log-scale SVG"

        leftmost_gridline = gridlines[0]
        leftmost_circle = circles[0]  # x=1 is the minimum value

        assert abs(leftmost_gridline - leftmost_circle) <= self._TOLERANCE, (
            f"Log-scale: left gridline ({leftmost_gridline:.3f}) does not coincide "
            f"with data mark at x=1 (cx={leftmost_circle:.3f}).  "
            f"Difference: {abs(leftmost_gridline - leftmost_circle):.3f}px.  "
            f"Scale projection fix may not apply to log axes."
        )

    def test_log_rightmost_gridline_coincides_with_x_max_mark(self) -> None:
        """Gridline at the log-scale right edge coincides with the mark at x=100."""
        svg = self._build_svg()
        gridlines = _vertical_gridline_xs(svg)
        circles = _circle_cxs(svg)

        assert gridlines, "No vertical gridlines in log-scale SVG"
        assert circles, "No circles in log-scale SVG"

        rightmost_gridline = gridlines[-1]
        rightmost_circle = circles[-1]  # x=100 is the maximum value

        assert abs(rightmost_gridline - rightmost_circle) <= self._TOLERANCE, (
            f"Log-scale: right gridline ({rightmost_gridline:.3f}) does not coincide "
            f"with data mark at x=100 (cx={rightmost_circle:.3f}).  "
            f"Difference: {abs(rightmost_gridline - rightmost_circle):.3f}px."
        )

    def test_log_tick_labels_are_geometrically_spaced(self) -> None:
        """X-axis tick labels on a log scale are geometrically spaced values.

        A linear scale over [1, 100] would produce uniformly spaced tick labels
        like 20, 40, 60, 80, 100.  A log scale produces geometrically spaced
        labels like 1, 1.58, 2.51, 3.98, 6.31, 10, …, 100.  The presence of
        '10' and absence of '50' (or '60') in the tick labels confirms the log
        scale is active and its ticks are scale-projected, not uniform-slot.
        """
        svg = self._build_svg()
        text_labels = re.findall(r"<text[^>]*>([^<]+)</text>", svg)

        # '10' must appear as a tick label — it is a log-decade value.
        assert "10" in text_labels, (
            f"Log-scale x axis missing '10' tick label.  "
            f"Got labels: {text_labels}.  "
            f"This suggests the axis reverted to linear-uniform ticking."
        )

        # Linear-scale ticks over [1, 100] would include '50' or '60'; log scale
        # has geometrically spaced labels and '50'/'60' do not appear.
        linear_spill = {"50", "60"} & set(text_labels)
        assert not linear_spill, (
            f"Found linear-scale tick labels {linear_spill} on a log-scale axis.  "
            f"All text labels: {text_labels}"
        )

    def test_log_gridlines_at_decade_values_match_mark_positions(self) -> None:
        """Each data mark (at x=1, 10, 100) has a coincident gridline.

        This is the core tick↔mark coincidence assertion for log axes.  All three
        circle cx values must be within tolerance of some gridline.
        """
        svg = self._build_svg()
        gridlines = _vertical_gridline_xs(svg)
        circles = _circle_cxs(svg)

        assert len(circles) == 3, f"Expected 3 circles (x=1,10,100); got {len(circles)}"

        for cx in circles:
            closest = min(gridlines, key=lambda g: abs(g - cx))
            assert abs(closest - cx) <= self._TOLERANCE, (
                f"Log-scale circle at cx={cx:.3f} has no coincident gridline.  "
                f"Closest gridline: {closest:.3f} (gap={abs(closest - cx):.3f}px)."
            )


# ---------------------------------------------------------------------------
# Test 3: Categorical axis — uniform slot placement preserved
# ---------------------------------------------------------------------------


class TestCategoricalAxisUniformSlots:
    """Verify that categorical (ordinal/nominal) x axes keep uniform slot placement.

    The fix must not change the categorical code path.  For a chart with n
    distinct categories, each bar is centered in a slot of width ``panel_w / n``,
    and the corresponding gridlines (if visible) are at those same slot centers.
    The slot centers are uniformly spaced: the gap between consecutive centers
    equals ``panel_w / n``.

    This test is not a byte-equality golden (which the Rust tests cover) — it
    asserts the geometric invariant: equal inter-bar spacing, equal intra-bar
    offset from left and right edges.
    """

    _WIDTH = 600
    _HEIGHT = 400
    _TOLERANCE = 2.0  # pixels

    def _build_svg(self) -> str:
        df = pl.DataFrame(
            {
                "cat": ["a", "b", "c", "d"],
                "val": [5.0, 10.0, 15.0, 20.0],
            }
        )
        return (
            fm.Chart(df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .properties(width=self._WIDTH, height=self._HEIGHT)
            .show_svg()
        )

    def test_categorical_bar_centers_are_uniformly_spaced(self) -> None:
        """Bar centers (x + width/2) are uniformly spaced across the 4 categories."""
        svg = self._build_svg()

        # Extract (x, width) from non-background bar rects.
        bar_segments: list[tuple[float, float]] = []
        for m in re.finditer(r"<rect([^/]+)/>", svg):
            attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
            try:
                x = float(attrs["x"])
                w = float(attrs["width"])
                h = float(attrs["height"])
            except (KeyError, ValueError):
                continue
            # Skip background panel rects and the chart outline
            if x > 10 and 10 < w < 500 and 10 < h < 350:
                bar_segments.append((x, w))

        bar_segments.sort()
        assert len(bar_segments) == 4, (
            f"Expected 4 bar rects for 4 categories; got {len(bar_segments)}.  "
            f"Check the rect-filtering heuristic."
        )

        centers = [x + w / 2 for x, w in bar_segments]
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

        assert len(gaps) == 3, f"Expected 3 inter-center gaps for 4 bars; got {len(gaps)}"

        max_gap, min_gap = max(gaps), min(gaps)
        assert max_gap - min_gap <= self._TOLERANCE, (
            f"Categorical bar centers are not uniformly spaced.  "
            f"Gaps: {[round(g, 3) for g in gaps]}.  "
            f"Max-min spread: {max_gap - min_gap:.3f}px (tolerance {self._TOLERANCE}px).  "
            f"The categorical slot path may have been accidentally modified."
        )

    def test_categorical_bar_widths_are_equal(self) -> None:
        """All bars in a uniform categorical chart have equal width."""
        svg = self._build_svg()

        widths: list[float] = []
        for m in re.finditer(r"<rect([^/]+)/>", svg):
            attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
            try:
                x = float(attrs["x"])
                w = float(attrs["width"])
                h = float(attrs["height"])
            except (KeyError, ValueError):
                continue
            if x > 10 and 10 < w < 500 and 10 < h < 350:
                widths.append(w)

        widths.sort()
        assert len(widths) == 4, f"Expected 4 bar widths; got {len(widths)}"

        max_w, min_w = max(widths), min(widths)
        assert max_w - min_w <= self._TOLERANCE, (
            f"Categorical bar widths are not equal.  "
            f"Widths: {[round(w, 3) for w in widths]}.  "
            f"Spread: {max_w - min_w:.3f}px."
        )

    def test_categorical_gridlines_are_uniformly_spaced(self) -> None:
        """Vertical gridlines under a categorical axis are uniformly spaced.

        For n=4 categories the inter-gridline gap should equal panel_w / 4 within
        tolerance.  This confirms that the categorical code path was not touched
        by the continuous-axis projection fix.
        """
        # Categorical axis may emit no gridlines by default (theme-dependent).
        # Build with an explicit grid theme to guarantee gridlines.
        df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [5.0, 10.0, 15.0, 20.0]})
        svg_grid = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .theme(fm.Theme(grid=True, grid_color="#abcdef"))
            .properties(width=self._WIDTH, height=self._HEIGHT)
            .show_svg()
        )
        gridlines_grid = _vertical_gridline_xs(svg_grid)

        if len(gridlines_grid) < 2:
            pytest.skip("Theme did not emit vertical gridlines for categorical axis")

        gaps = [gridlines_grid[i + 1] - gridlines_grid[i] for i in range(len(gridlines_grid) - 1)]
        max_gap, min_gap = max(gaps), min(gaps)
        assert max_gap - min_gap <= self._TOLERANCE, (
            f"Categorical vertical gridlines are not uniformly spaced.  "
            f"Gridlines: {[round(g, 3) for g in gridlines_grid]}.  "
            f"Gaps: {[round(g, 3) for g in gaps]}.  "
            f"Spread: {max_gap - min_gap:.3f}px."
        )

    def test_categorical_vs_continuous_first_gridline_differs(self) -> None:
        """Categorical and continuous axes place their first gridlines differently.

        On a categorical 4-slot chart the first gridline is at the center of the
        first slot (``panel_x + slot_w / 2``).  On a continuous chart with data
        at the domain extremes the first gridline is at the padded domain extent
        (farther left, at the mark position).  This contrast confirms the two
        code paths are independent.
        """
        # Categorical chart
        df_cat = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [5.0, 10.0, 15.0, 20.0]})
        svg_cat = (
            fm.Chart(df_cat)
            .mark_bar()
            .encode(x="cat:N", y="val:Q")
            .theme(fm.Theme(grid=True, grid_color="#abcdef"))
            .properties(width=self._WIDTH, height=self._HEIGHT)
            .show_svg()
        )
        gridlines_cat = _vertical_gridline_xs(svg_cat)

        # Continuous chart with data at domain extremes
        df_cont = pl.DataFrame({"x": [0.0, 33.0, 67.0, 100.0], "y": [1.0, 2.0, 3.0, 4.0]})
        svg_cont = (
            fm.Chart(df_cont)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .theme(fm.Theme(grid=True, grid_color="#abcdef"))
            .properties(width=self._WIDTH, height=self._HEIGHT)
            .show_svg()
        )
        gridlines_cont = _vertical_gridline_xs(svg_cont)

        if not gridlines_cat or not gridlines_cont:
            pytest.skip("Could not extract gridlines from one or both chart types")

        # For a 4-category bar chart the first gridline is the first slot center —
        # it must be further right than the padded domain-min of a continuous chart.
        # (Categorical: panel_x + slot_w/2; continuous: padded_min = padded domain extent)
        # These should be meaningfully different.
        cat_first = gridlines_cat[0]
        cont_first = gridlines_cont[0]

        # They should not be within 2px of each other — different placement schemes.
        assert abs(cat_first - cont_first) > 5, (
            f"Categorical first gridline ({cat_first:.3f}) is suspiciously close to "
            f"continuous first gridline ({cont_first:.3f}).  "
            f"The two placement schemes should produce distinct positions."
        )
