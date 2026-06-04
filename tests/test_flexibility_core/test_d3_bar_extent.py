"""Regression tests for defect D3 — mark_bar and mark_area ignore y2/x2 extent.

The bug: binding y2 on mark_bar (ordinal-x path) has no effect. The bar
always anchors at the zero/axis baseline and extends to y, regardless of
whether y2 is bound. mark_area also ignores y2 — the area always fills
from the y top edge down to the baseline rather than forming a band between
y and y2.

Root causes (Rust):
  bar.rs   — build_ordinal(): reads only y, never reads y2 column.
  area.rs  — build(): never reads spec.encoding.y2.

Python gap:
  mark_bar has no `zero=` parameter; it cannot suppress the zero anchor
  without providing a full custom scale with domain.

These tests assert on OBSERVABLE rendered SVG geometry (rect y/height for
bars, path d= coordinates for area). They will FAIL until the Rust fix lands
and the Python zero= escape is added.

All five test scenarios are distinct and isolate a different facet of the bug:

  1. Candlestick body (ordinal x + y + y2, both positive, no zero in range)
  2. Floating/gap bar (y=lo > 0, y2=hi > 0, both off the baseline)
  3. Diverging/mixed-sign bar (y < 0, y2 > 0, spans the zero line)
  4. mark_bar(zero=False) with only y bound (zero anchor suppressed)
  5. mark_bar() default guard: zero=True with only y bound still anchors (passes today)
  6. mark_area with y2 (band fill rather than baseline fill)
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG extraction helpers
# ---------------------------------------------------------------------------


def _rects(svg: str) -> list[tuple[float, float, float, float]]:
    """Extract all <rect> elements from SVG as (x, y, width, height) float tuples.

    Skips the two background/axis chrome rects that appear at the start of
    every ferrum SVG: the full-viewport background at (0,0) and the plot-area
    clip rect.  Returns only the data-mark rects.
    """
    raw = re.findall(
        r'<rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"[^/]*/>',
        svg,
    )
    tuples = [(float(x), float(y), float(w), float(h)) for x, y, w, h in raw]
    # Drop the two chrome rects: background at (0, 0) and the axis-area rect.
    # Data rects always come after the first two.  Use a simple heuristic:
    # skip rects whose y=0 (background) or whose w > 500 and h > 300 (plot chrome).
    data_rects = []
    for r in tuples:
        x, y, w, h = r
        if y == 0 and x == 0:
            continue  # background rect
        if w > 500 and h > 300:
            continue  # axis chrome / plot area rect
        data_rects.append(r)
    return data_rects


def _baseline_y(svg: str) -> float:
    """Return the y-coordinate of the zero-baseline from the SVG plot-area rect.

    The plot-area rect (second rect in the SVG, the one with w > 500 and h > 300)
    has y=plot_y and h=plot_h; the baseline is at y + h (bottom of the plot
    area, where y=0 on the data scale maps in SVG screen coordinates).
    """
    raw = re.findall(
        r'<rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"[^/]*/>',
        svg,
    )
    for x_s, y_s, w_s, h_s in raw:
        x, y, w, h = float(x_s), float(y_s), float(w_s), float(h_s)
        if w > 500 and h > 300 and x > 0:
            return y + h
    # Fallback: return the maximum y+h across all rects.
    return max(float(y_s) + float(h_s) for _, y_s, _, h_s in raw)


def _y_axis_tick_labels(svg: str) -> list[str]:
    """Return all numeric y-axis tick label strings from the SVG.

    Y-axis labels use text-anchor="end" (they are right-aligned, to the left of
    the axis line).  Filter to elements that parse as floats to exclude axis
    title, legend, and x-axis category labels.
    """
    elements = re.findall(r"<text\s([^>]+)>([^<]+)</text>", svg)
    labels = []
    for attrs, content in elements:
        if 'text-anchor="end"' in attrs:
            try:
                float(content.strip())
                labels.append(content.strip())
            except ValueError:
                pass
    return labels


def _zero_pixel(svg: str) -> float | None:
    """Return the SVG y-coordinate of the data-value-0 row from the y-axis ticks.

    Finds the two y-axis tick labels that bracket value 0 (or the exact "0"
    label if present) and interpolates to find the pixel row where data=0.
    Returns None if the axis does not span 0 at all.

    This is more reliable than using the plot-area bottom as a zero proxy:
    the plot bottom corresponds to the data minimum, not necessarily 0.
    """
    elements = re.findall(r"<text\s([^>]+)>([^<]+)</text>", svg)
    # Collect (pixel_y, data_value) for numeric y-axis labels only.
    tick_points: list[tuple[float, float]] = []
    for attrs, content in elements:
        if 'text-anchor="end"' in attrs:
            try:
                val = float(content.strip())
                # Extract the y= attribute from the text element's attrs.
                y_match = re.search(r'\by="([^"]+)"', attrs)
                if y_match:
                    tick_points.append((float(y_match.group(1)), val))
            except ValueError:
                pass

    if not tick_points:
        return None

    # Sort by data value ascending.
    tick_points.sort(key=lambda p: p[1])
    data_values = [p[1] for p in tick_points]
    pixel_ys = [p[0] for p in tick_points]

    # Check if 0 is within the span of the axis values.
    if data_values[0] > 0 or data_values[-1] < 0:
        return None  # axis does not include 0

    # Find exact match first.
    for py, dv in tick_points:
        if dv == 0.0:
            return py

    # Interpolate between the two ticks bracketing 0.
    for i in range(len(data_values) - 1):
        lo_val, hi_val = data_values[i], data_values[i + 1]
        if lo_val <= 0 <= hi_val:
            lo_py, hi_py = pixel_ys[i], pixel_ys[i + 1]
            # SVG y increases downward; higher data value = smaller pixel y.
            frac = (0 - lo_val) / (hi_val - lo_val)
            return lo_py + frac * (hi_py - lo_py)

    return None


def _area_paths(svg: str) -> list[str]:
    """Extract all <path d="..."> d-strings from the SVG."""
    return re.findall(r'<path d="([^"]+)"', svg)


def _path_y_coords(path_d: str) -> list[float]:
    """Return all y-coordinate values that appear in an SVG path d-string.

    Handles M/L/Z commands.  Returns all numeric y values found (every second
    number in M/L coordinate pairs, plus the implicit closure).
    """
    # Extract all coordinate pairs from M and L commands.
    coords = re.findall(r"[ML]\s*([\d.]+)\s+([\d.]+)", path_d)
    return [float(y) for _, y in coords]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def candlestick_df() -> pl.DataFrame:
    """Three-row candlestick dataset with open != close and neither equals zero.

    open=3, close=8  (close > open, bar should span [3,8], not [0,3])
    open=7, close=2  (close < open, inverted candle, spans [2,7])
    open=5, close=10 (straightforward rising candle)

    All values are positive and well away from zero.  A zero-anchored bar
    would produce a visually different rect (bottom at axis baseline rather
    than at the open-value pixel).
    """
    return pl.DataFrame(
        {
            "cat": ["a", "b", "c"],
            "open": [3.0, 7.0, 5.0],
            "close": [8.0, 2.0, 10.0],
        }
    )


@pytest.fixture
def float_bar_df() -> pl.DataFrame:
    """Two-row dataset where lo and hi are both strictly positive.

    lo=5, hi=10 — a bar that floats off the zero baseline.
    Under the bug, the bar anchors at the baseline rather than at lo.
    """
    return pl.DataFrame(
        {
            "cat": ["a", "b"],
            "lo": [5.0, 7.0],
            "hi": [10.0, 12.0],
        }
    )


@pytest.fixture
def diverging_df() -> pl.DataFrame:
    """Single bar spanning a negative-to-positive range.

    y=-3, y2=2: the bar should cross the zero line and appear both above and
    below the axis baseline.  Under the bug the bar just goes from baseline
    to y=-3 (negative direction only).
    """
    return pl.DataFrame(
        {
            "cat": ["a"],
            "lo": [-3.0],
            "hi": [2.0],
        }
    )


@pytest.fixture
def area_band_df() -> pl.DataFrame:
    """Four-point dataset for an interval/band area chart.

    lo = [2, 3, 4, 3], hi = [5, 6, 7, 6]: the area should fill the band
    between lo and hi.  Under the bug the area anchors at the baseline
    regardless of y2.
    """
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "lo": [2.0, 3.0, 4.0, 3.0],
            "hi": [5.0, 6.0, 7.0, 6.0],
        }
    )


@pytest.fixture
def zero_anchor_df() -> pl.DataFrame:
    """Standard bar-chart dataset used for the zero-anchor guard.

    val = [5, 8]: under default zero=True, bars anchor at the axis baseline.
    """
    return pl.DataFrame(
        {
            "cat": ["a", "b"],
            "val": [5.0, 8.0],
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — Candlestick body: ordinal x + y=open + y2=close
# ---------------------------------------------------------------------------


def test_bar_candlestick_body_spans_open_to_close(
    candlestick_df: pl.DataFrame,
) -> None:
    """mark_bar with ordinal x, y=open, y2=close must draw each bar spanning
    from the open value to the close value — NOT from the zero baseline to open.

    Observable check: the bottom of at least one data rect (y + height) does NOT
    coincide with the axis baseline.  Under the bug every rect's y + height equals
    the baseline_y because build_ordinal() ignores y2 and always anchors there.

    Additionally, binding y2 must change the rendered rects relative to not
    binding y2 — the two SVGs must differ.
    """
    svg_with_y2 = (
        fm.Chart(candlestick_df).mark_bar().encode(x="cat:N", y="open:Q", y2="close:Q").to_svg()
    )
    svg_no_y2 = fm.Chart(candlestick_df).mark_bar().encode(x="cat:N", y="open:Q").to_svg()

    baseline = _baseline_y(svg_with_y2)
    data_rects = _rects(svg_with_y2)

    # Primary assertion: at least one rect must NOT bottom out at the baseline.
    # Under the bug, all rects touch the baseline (y2 is silently dropped).
    tol = 2.0  # pixel tolerance for floating-point comparison
    any_off_baseline = any(abs((r[1] + r[3]) - baseline) > tol for r in data_rects)
    assert any_off_baseline, (
        "All data bars still bottom out at the axis baseline even though y2=close "
        "is bound.  y2 is being silently ignored — the ordinal build_ordinal() "
        "path never reads the y2 column.  Expected at least one bar to have its "
        "bottom at the open-value pixel row, not at the baseline.  "
        f"baseline_y={baseline:.3f}, rects (x,y,w,h)={data_rects}"
    )

    # Secondary assertion: binding y2 must change the rendered output.
    assert svg_with_y2 != svg_no_y2, (
        "mark_bar() with y2=close produces the same SVG as without y2.  "
        "y2 must visibly change the bar geometry, not be silently dropped."
    )


# ---------------------------------------------------------------------------
# Test 2 — Floating/gap bar: both y and y2 are strictly above the baseline
# ---------------------------------------------------------------------------


def test_bar_floating_gap_off_baseline(float_bar_df: pl.DataFrame) -> None:
    """mark_bar with y=lo > 0 and y2=hi > 0 renders bars that do NOT touch
    the zero baseline.

    Observable check: no data rect should have its bottom (y + height) within
    a few pixels of the axis baseline.  Under the bug, build_ordinal() always
    sets the bottom at baseline_y regardless of y2.

    This is the "floating/gap bar" case: both extents are positive, so a
    correctly rendered bar is entirely lifted off the axis.
    """
    svg = fm.Chart(float_bar_df).mark_bar().encode(x="cat:N", y="lo:Q", y2="hi:Q").to_svg()

    baseline = _baseline_y(svg)
    data_rects = _rects(svg)

    assert data_rects, (
        "No data bars were emitted for the floating-bar chart (lo=5,hi=10).  "
        "Expected 2 rects (one per row)."
    )

    tol = 2.0
    any_touches_baseline = any(abs((r[1] + r[3]) - baseline) <= tol for r in data_rects)
    assert not any_touches_baseline, (
        "At least one floating bar (lo=5, hi=10) still bottoms out at the axis "
        "baseline — y2 is being ignored.  With both lo > 0 and hi > 0, a correctly "
        "rendered bar should be entirely above the baseline.  "
        f"baseline_y={baseline:.3f}, data rects (x,y,w,h)={data_rects}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Diverging/mixed-sign: bar spans y < 0 to y2 > 0
# ---------------------------------------------------------------------------


def test_bar_diverging_mixed_sign_crosses_zero(diverging_df: pl.DataFrame) -> None:
    """mark_bar with y=-3 and y2=2 must render a rect that straddles the zero line.

    The data domain is [-3, 2].  The plot-area BOTTOM corresponds to data=-3,
    not data=0.  The zero line sits in the INTERIOR of the plot.  Comparing the
    rect bottom against _baseline_y (the plot bottom) is therefore wrong for
    mixed-sign data — the correct check is that the rect spans both sides of
    the actual y=0 data pixel.

    Observable check:
      - Extract the zero-data pixel row from the y-axis tick labels via
        _zero_pixel().  For domain [-3, 2] this is well inside the plot area.
      - Assert rect_top < zero_pixel < rect_bottom (the rect covers both the
        positive side and the negative side of zero).

    Bug-sensitivity: under the pre-D3 bug, y2 is ignored and the bar anchors
    at 0 extending down to -3.  The rect top would sit AT the zero pixel, so
    rect_top < zero_pixel would fail.

    Also: binding y2=2 must change the SVG relative to not binding y2.
    """
    svg_with_y2 = fm.Chart(diverging_df).mark_bar().encode(x="cat:N", y="lo:Q", y2="hi:Q").to_svg()
    svg_no_y2 = fm.Chart(diverging_df).mark_bar().encode(x="cat:N", y="lo:Q").to_svg()

    data_rects = _rects(svg_with_y2)

    assert data_rects, (
        "No data bars were emitted for the diverging bar chart (lo=-3, hi=2).  Expected 1 rect."
    )

    # Locate the zero-data pixel from the y-axis tick labels.  For domain [-3, 2]
    # this is interior to the plot (not at the plot-area edge).
    zero_px = _zero_pixel(svg_with_y2)
    assert zero_px is not None, (
        "Could not locate the y=0 data pixel from the y-axis tick labels.  "
        "The axis for domain [-3, 2] must include a 0 tick or ticks that bracket 0."
    )

    rect = data_rects[0]
    rect_top = rect[1]  # SVG y of the rect's top edge (hi=2 side)
    rect_bottom = rect[1] + rect[3]  # SVG y of the rect's bottom edge (lo=-3 side)

    tol = 2.0

    # The rect must extend above the zero line (hi=2 is positive).
    assert rect_top < zero_px - tol, (
        "Diverging bar (lo=-3, hi=2): rect top should be ABOVE the zero data "
        f"pixel (rect_top={rect_top:.3f} should be < zero_px={zero_px:.3f} - tol).  "
        "The positive extent (hi=2) is not being rendered above zero — y2 may be "
        "silently ignored.  "
        f"rect (x,y,w,h)={rect}"
    )

    # The rect must also extend below the zero line (lo=-3 is negative).
    assert rect_bottom > zero_px + tol, (
        "Diverging bar (lo=-3, hi=2): rect bottom should be BELOW the zero data "
        f"pixel (rect_bottom={rect_bottom:.3f} should be > zero_px={zero_px:.3f} + tol).  "
        "The negative extent (lo=-3) is not being rendered below zero.  "
        f"rect (x,y,w,h)={rect}"
    )

    # Binding y2 must visibly change the output.
    assert svg_with_y2 != svg_no_y2, (
        "mark_bar() with y2=hi=2 produces the same SVG as without y2.  "
        "The diverging extent must change the rendered geometry."
    )


# ---------------------------------------------------------------------------
# Test 4 — zero= escape: mark_bar(zero=False) with only y bound
# ---------------------------------------------------------------------------


def test_bar_zero_false_suppresses_baseline_anchor(zero_anchor_df: pl.DataFrame) -> None:
    """mark_bar(zero=False) changes the y-axis domain so it excludes 0.

    The spec-defined effect of zero=False (spec §8 "zero-anchor becomes opt-out")
    is on the y-axis DOMAIN, not on whether bars "float."  With all-positive data
    (val=[5,8]) and zero=True (default), the domain is anchored at 0 so the axis
    includes a "0" tick.  With zero=False, the domain follows the data range [5,8]
    and the "0" tick must be absent.

    Bars in both cases correctly anchor to the domain floor (the bottom of the plot
    area), which is 0 for zero=True and ≈5 for zero=False.  This is correct bar
    semantics, not "floating."

    Observable check (domain-level, not geometry-level):
      - zero=True:  "0" appears in the y-axis tick labels.
      - zero=False: "0" does NOT appear; minimum tick value is ≥ 5.0 (the data min).

    Bug-sensitivity: before D3, mark_bar(zero=False) raises TypeError because
    'zero' is not an accepted kwarg.  After the fix, it must render and produce
    an axis domain that excludes 0.
    """
    svg_true = fm.Chart(zero_anchor_df).mark_bar().encode(x="cat:N", y="val:Q").to_svg()
    svg_false = fm.Chart(zero_anchor_df).mark_bar(zero=False).encode(x="cat:N", y="val:Q").to_svg()

    ticks_true = _y_axis_tick_labels(svg_true)
    ticks_false = _y_axis_tick_labels(svg_false)

    assert ticks_true, "No numeric y-axis ticks found in zero=True chart."
    assert ticks_false, "No numeric y-axis ticks found in zero=False chart."

    # zero=True: the y-axis must include a "0" tick (domain anchored to zero).
    assert "0" in ticks_true, (
        f"mark_bar() default (zero=True) — expected '0' in y-axis ticks but got: {ticks_true}. "
        "The zero-anchor default must produce a domain that starts at 0."
    )

    # zero=False: the y-axis must NOT include "0"; domain tracks the data range.
    assert "0" not in ticks_false, (
        f"mark_bar(zero=False) — '0' still appears in y-axis ticks: {ticks_false}.  "
        "zero=False must suppress the zero-anchor so the domain starts near the "
        "data minimum (~5), not at 0."
    )

    epsilon = 0.5
    min_tick = min(float(t) for t in ticks_false)
    assert min_tick >= 5.0 - epsilon, (
        f"mark_bar(zero=False) — y-axis minimum tick is {min_tick:.2f}, expected ≥ 5.0 "
        f"(the data minimum).  Ticks: {ticks_false}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Default guard: mark_bar() zero=True still anchors at baseline
# ---------------------------------------------------------------------------


def test_bar_default_zero_true_still_anchors_at_baseline(
    zero_anchor_df: pl.DataFrame,
) -> None:
    """mark_bar() default (zero=True) with only y bound must still anchor bars
    at the zero baseline.  This test passes today and guards against regressing
    the zero-anchor default when the zero= escape is added.

    Observable check: all data bars' bottoms coincide with the axis baseline.
    """
    svg = fm.Chart(zero_anchor_df).mark_bar().encode(x="cat:N", y="val:Q").to_svg()

    baseline = _baseline_y(svg)
    data_rects = _rects(svg)

    assert data_rects, (
        "No data bars emitted for default-zero-anchor chart (val=[5,8]).  Expected 2 rects."
    )

    tol = 2.0
    all_at_baseline = all(abs((r[1] + r[3]) - baseline) <= tol for r in data_rects)
    assert all_at_baseline, (
        "mark_bar() default (zero=True) — at least one bar does NOT bottom out "
        "at the axis baseline.  The default zero-anchor must be preserved.  "
        f"baseline_y={baseline:.3f}, data rects (x,y,w,h)={data_rects}"
    )


# ---------------------------------------------------------------------------
# Test 6 — mark_area with y2: band fill between y and y2
# ---------------------------------------------------------------------------


def test_area_y2_forms_band_not_baseline_fill(area_band_df: pl.DataFrame) -> None:
    """mark_area with y=lo and y2=hi must fill the band between lo and hi, NOT
    from lo down to the baseline.

    Observable check: when y2 is bound, the rendered path must include
    coordinates at BOTH the lo and hi pixel rows (a band spanning between them).
    Without y2, the path always includes the baseline y-coordinate at the end
    of the polygon closure.

    We assert:
    1. The path with y2 differs from the path without y2 (y2 changes geometry).
    2. With y2, the path does NOT close at the axis baseline — the path
       should not include the baseline_y coordinate as a y value.

    NOTE: If assertion 2 is too fragile (floating-point variation), the test
    falls back to assertion 1 alone as the minimum observable change.
    """
    svg_no_y2 = fm.Chart(area_band_df).mark_area().encode(x="x:Q", y="lo:Q").to_svg()
    svg_with_y2 = fm.Chart(area_band_df).mark_area().encode(x="x:Q", y="lo:Q", y2="hi:Q").to_svg()

    paths_no_y2 = _area_paths(svg_no_y2)
    paths_with_y2 = _area_paths(svg_with_y2)

    assert paths_no_y2, "Expected at least one area path without y2."
    assert paths_with_y2, "Expected at least one area path with y2."

    # Primary: the main (first closed) area path must change when y2 is bound.
    # Under the bug, the paths differ only because the y-axis domain expands
    # (y2 values enter the scale domain), but the shape still anchors at the
    # baseline. The fix makes y2 a structural part of the closed polygon.
    main_path_no_y2 = paths_no_y2[0]
    main_path_with_y2 = paths_with_y2[0]

    assert main_path_no_y2 != main_path_with_y2, (
        "mark_area() with y2=hi produces the same path d= as without y2.  "
        "Binding y2 must change the area shape, not just the axis scale domain.  "
        f"path without y2: {main_path_no_y2!r}  path with y2: {main_path_with_y2!r}"
    )

    # Secondary: the path with y2 should NOT include the axis baseline as a
    # y-coordinate in the polygon closure.  With a band area, the polygon
    # closes along the hi-edge (upper boundary) rather than at the baseline.
    #
    # This checks that y2 changes the shape structurally — the baseline
    # closure (L ... baseline_y L x0 baseline_y Z) should be absent.
    baseline = _baseline_y(svg_with_y2)
    ys_with_y2 = _path_y_coords(main_path_with_y2)
    tol = 2.0
    baseline_in_path = any(abs(y - baseline) <= tol for y in ys_with_y2)
    assert not baseline_in_path, (
        "mark_area() with y2=hi — the rendered path still includes the axis "
        "baseline coordinate as a closure point.  With y2 bound, the area "
        "should fill between y and y2 (a band); the baseline should not appear "
        "in the path.  Under the bug, the area is drawn from y down to the "
        "baseline and y2 is ignored — only the axis scale domain changes.  "
        f"baseline_y={baseline:.3f}, path y-coords={ys_with_y2}, "
        f"path={main_path_with_y2!r}"
    )
