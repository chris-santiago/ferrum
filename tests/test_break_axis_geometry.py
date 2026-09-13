"""Geometry contracts for ``BreakAxis`` (spec §4 H4, F-L08-06).

The guide's clipping model (``docs/site/guide/concepts/break-axes.md``) says a
mark whose extent spans a break renders as one clipped sub-mark per visible
segment: a bar from 0 to 900 under a ``(150, 850)`` gap is two bars, 0–150 and
850–900. These tests pin that model end to end through the real render path,
plus the riders it travels with: a value sitting *inside* a gap keeps its
visible portion instead of collapsing to ``height=0``, overlapping and nested
gaps normalize to one removed range, a ``SecondaryY`` layer is exempt from the
primary axis's break, and the axis title (which no data value positions) is not
dragged into the remap.

Pixel expectations are derived from the segment allocation by hand — retained
span fractions of ``plot_height - n_breaks * break_size`` — rather than read
back out of the implementation.
"""

from __future__ import annotations

import collections
import json
import math
import re
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from _svg_extents import axis_tick_dash_label_pairs, label_along, tick_along


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_MARK_FILL = "#2563eb"


def _rects(svg: str, fill: str = _MARK_FILL) -> list[tuple[float, float, float, float]]:
    """``(x, y, w, h)`` for every mark rect painted in ``fill``."""
    pattern = (
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)" '
        rf'fill="{re.escape(fill)}"'
    )
    return [tuple(float(g) for g in m.groups()) for m in re.finditer(pattern, svg)]


def _paths(svg: str) -> list[str]:
    return re.findall(r'<path d="([^"]*)"', svg)


def _path_coords(d: str) -> list[tuple[float, float]]:
    return [
        (float(m.group(1)), float(m.group(2))) for m in re.finditer(r"[ML]([-\d.]+) ([-\d.]+)", d)
    ]


def _bar_svg(values: list[float], gap, categories: list[str] | None = None) -> str:
    cats = categories or [chr(ord("A") + i) for i in range(len(values))]
    df = pl.DataFrame({"c": cats, "v": values})
    chart = fm.Chart(df).mark_bar().encode(x="c:N", y="v:Q") + fm.BreakAxis(axis="y", gap=gap)
    return chart.to_svg()


# --------------------------------------------------------------------------
# extent clipping
# --------------------------------------------------------------------------


def test_bar_crossing_a_gap_renders_two_clipped_segments():
    """The guide's worked example: 0→900 under (150, 850) is two bars."""
    svg = _bar_svg([72.0, 68.0, 75.0, 900.0, 81.0], (150, 850))
    rects = _rects(svg)
    assert len(rects) == 6, f"four short bars + the crossing bar's two pieces; got {rects}"

    by_x: dict[float, list[tuple[float, float, float, float]]] = {}
    for r in rects:
        by_x.setdefault(round(r[0], 3), []).append(r)
    split = [group for group in by_x.values() if len(group) == 2]
    assert len(split) == 1, f"exactly one bar splits; got {by_x}"
    lower, upper = sorted(split[0], key=lambda r: -r[1])
    assert lower[2] == upper[2], "both pieces keep the bar's width"
    assert lower[3] > 0.5 and upper[3] > 0.5, "neither piece is a zero-height artifact"
    assert upper[1] + upper[3] < lower[1], "the pieces are separated by the break"

    # No bar spans the whole plot: that was the pre-fix symptom.
    plot_top = min(r[1] for r in rects)
    plot_bottom = max(r[1] + r[3] for r in rects)
    plot_h = plot_bottom - plot_top
    assert not any(r[1] <= plot_top + 1.0 and r[3] >= 0.95 * plot_h for r in rects), (
        f"no unclipped full-height bar may survive a break; got {rects}"
    )


def test_value_inside_a_gap_keeps_its_visible_portion():
    """A bar whose top lands inside the gap draws its below-gap portion."""
    svg = _bar_svg([400.0, 900.0], (150, 850))
    rects = sorted(_rects(svg), key=lambda r: (r[0], r[1]))
    in_gap = [r for r in rects if abs(r[0] - rects[0][0]) < 1e-9]
    assert len(in_gap) == 1, f"one visible piece for the in-gap value; got {rects}"
    assert in_gap[0][3] > 0.5, "the visible portion has real height, not height=0"

    crossing_lower = [r for r in rects if r not in in_gap and r[1] > 100.0]
    assert crossing_lower, f"expected the crossing bar's lower piece; got {rects}"
    assert abs(crossing_lower[0][1] - in_gap[0][1]) < 1e-6, (
        "both bars are cut at the same gap edge — the clip is a property of the "
        "scale, not of the bar"
    )


def test_no_zero_height_bar_rects_under_multiple_gaps():
    """Multi-gap repro from the finding: no bar may vanish into height=0."""
    svg = _bar_svg([30.0, 120.0, 250.0, 550.0, 700.0, 950.0], [(50, 200), (500, 900)])
    rects = _rects(svg)
    assert rects, "bars must render at all"
    assert all(r[3] > 0.5 for r in rects), f"zero-height bar rects remain: {rects}"
    # Three segments: a bar above the top gap yields three pieces.
    tallest_column = max(_rects(svg), key=lambda r: r[0])[0]
    pieces = [r for r in rects if abs(r[0] - tallest_column) < 1e-9]
    assert len(pieces) == 3, f"the 950 bar crosses two gaps -> three pieces; got {pieces}"


def test_extent_entirely_inside_a_gap_emits_no_rect():
    """A bar with no visible segment leaves nothing behind, not a flat rect."""
    # Gap (0, 500) removes the bottom of the domain, so bar A (300) is wholly
    # hidden while bar B (800) keeps its 500–800 portion.
    svg = _bar_svg([300.0, 800.0], (0, 500))
    rects = _rects(svg)
    assert len(rects) == 1, f"only the partly-visible bar renders; got {rects}"
    assert rects[0][3] > 0.5


def test_nested_gaps_do_not_re_admit_the_enclosing_range():
    """``[(100, 900), (200, 300)]`` removes one range, not two disjoint ones.

    Before normalization the segment walk took the *last* gap's upper bound as
    its cursor, which re-admitted data 300–900 that the enclosing gap had
    removed — a bar of 500 then drew a second piece inside the break.
    """
    svg = _bar_svg([500.0, 950.0], [(100, 900), (200, 300)])
    rects = sorted(_rects(svg), key=lambda r: (r[0], r[1]))
    left = [r for r in rects if abs(r[0] - rects[0][0]) < 1e-9]
    assert len(left) == 1, (
        "the 500 bar is visible only below the merged gap (its 300–500 range is "
        f"inside it); got {rects}"
    )
    right = [r for r in rects if r not in left]
    assert len(right) == 2, f"the 950 bar crosses the merged gap; got {rects}"


def test_area_crossing_a_gap_clips_into_two_paths_without_offscreen_spikes():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [50.0, 900.0, 950.0, 60.0]})
    svg = (
        fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q") + fm.BreakAxis(axis="y", gap=(150, 850))
    ).to_svg()
    area_paths = [d for d in _paths(svg) if d.endswith("Z")]
    assert len(area_paths) == 2, f"one closed path per visible segment; got {area_paths}"
    for d in area_paths:
        ys = [y for _, y in _path_coords(d)]
        assert ys, f"path has no vertices: {d}"
        assert all(-1.0 <= y <= 500.0 for y in ys), (
            f"a clipped area must not carry BREAK_HIDDEN spikes; got {d}"
        )
    # The two bands are separated: the upper band ends above the lower starts.
    lower, upper = sorted(area_paths, key=lambda d: -max(y for _, y in _path_coords(d)))
    assert max(y for _, y in _path_coords(upper)) < min(y for _, y in _path_coords(lower))


def test_point_marks_keep_hide_and_compress_semantics():
    """Single-value geometry is hidden off-screen, never dropped (spec §4 H4)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 500.0, 900.0]})
    svg = (
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q") + fm.BreakAxis(axis="y", gap=(150, 850))
    ).to_svg()
    circles = re.findall(r'<circle[^>]*cy="([-\d.]+)"[^>]*>', svg)
    assert len(circles) == 3, "point marks stay 1:1 with their rows"
    assert sum(1 for cy in circles if float(cy) < -1000) == 1, (
        f"the in-gap point is hidden off-screen; got {circles}"
    )


def _glyph_nodes(svg: str, shape: str) -> list[str]:
    """The mark-coloured nodes a point glyph of ``shape`` emits."""
    tag = {"circle": "circle", "square": "rect", "diamond": "path", "triangle-up": "path"}[shape]
    return [m for m in re.findall(rf"<{tag}[^>]*>", svg) if _MARK_FILL in m]


@pytest.mark.parametrize("shape", ["circle", "square", "diamond", "triangle-up"])
def test_every_point_shape_keeps_one_node_per_row_under_a_break(shape):
    """A glyph moves or hides whole — it is never clipped, resized or dropped.

    Point glyphs are fixed-size symbols anchored at one data position, so the
    extent clip does not apply to them: it used to delete the in-gap diamond
    and square outright (taking their ``data_indices``/tooltip rows with them)
    while the in-gap circle was merely hidden, and cut the survivors flat at
    the gap edge. Every shape now behaves like the default circle.
    """
    # Values are kept clear of the gap edges: a mark sitting exactly ON an edge
    # is a knife-edge for the pixel->data round trip (and, for an asymmetric
    # glyph, for the bounding-box anchor), which is a different question from
    # the 1:1 contract under test here.
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 500.0, 100.0, 900.0]})
    chart = fm.Chart(df).mark_point(shape=shape).encode(x="x:Q", y="y:Q")
    plain = _glyph_nodes(chart.to_svg(), shape)
    broken_svg = (chart + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg()
    broken = _glyph_nodes(broken_svg, shape)
    assert len(broken) == len(plain) == 4, (
        f"{shape}: one node per row before and after the break; got {broken}"
    )

    def y_coords(node: str) -> list[float]:
        if node.startswith("<circle"):
            return [float(re.search(r'cy="([-\d.]+)"', node).group(1))]
        if node.startswith("<rect"):
            y = float(re.search(r'y="([-\d.]+)"', node).group(1))
            return [y, y + float(re.search(r'height="([\d.]+)"', node).group(1))]
        return [float(y) for _, y in _path_coords(re.search(r'd="([^"]*)"', node).group(1))]

    hidden = [n for n in broken if all(y < -1000 for y in y_coords(n))]
    assert len(hidden) == 1, f"{shape}: exactly the in-gap glyph hides; got {broken}"
    for node in broken:
        ys = y_coords(node)
        assert not (min(ys) < -1000 < max(ys)), (
            f"{shape}: a glyph must not straddle the viewport edge — it moves or hides "
            f"whole, never partly; got {node}"
        )
    # Size is preserved: a glyph is not resized by the scale compression.
    # Tolerance is the SVG serializer's 3-decimal rounding, applied per
    # coordinate, so a span can differ by up to 0.001px without any geometry
    # having changed.
    spans = sorted(max(y_coords(n)) - min(y_coords(n)) for n in broken)
    plain_spans = sorted(max(y_coords(n)) - min(y_coords(n)) for n in plain)
    assert spans == pytest.approx(plain_spans, abs=0.002), (
        f"{shape}: the break moves glyphs, it does not resize them; {spans} vs {plain_spans}"
    )


def test_line_marks_follow_the_broken_scale():
    """``mark_line``'s Polyline is placed on the broken scale, not ignored.

    Before the Polyline arm existed, a line chart under a break rendered
    byte-identically to the unbroken chart — i.e. against ticks it no longer
    agreed with. The placement model it got then (per-vertex, in-gap vertices
    parked at the sentinel) is now the split — see the vertex-splitting section
    below for the geometry.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 500.0, 900.0]})
    chart = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    plain = re.findall(r'<polyline points="([^"]*)"', chart.to_svg())
    broken = re.findall(
        r'<polyline points="([^"]*)"',
        (chart + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg(),
    )
    assert plain and broken and broken != plain, (
        f"the line must move onto the broken scale; plain={plain} broken={broken}"
    )


def test_violin_is_clipped_per_segment_rather_than_left_on_the_unbroken_scale():
    """A polygon mark follows the break, and its filled rings are CLIPPED.

    ``mark_violin`` lowers to the polygon mark, whose ``SceneNode::Polygon``
    used to fall through the remap's catch-all: the violin rendered
    byte-identically to the unbroken chart while its own ticks compressed.
    Placing a filled ring vertex-by-vertex is not the fix — an in-gap vertex
    parked off-canvas would paint the fill as a column across the break — so
    the rings clip, exactly as the area fill's do.
    """
    rng = np.random.default_rng(1)
    values = list(rng.normal(100, 25, 40)) + list(rng.normal(800, 25, 40))
    df = pl.DataFrame({"g": ["a"] * 80, "v": values})
    chart = fm.Chart(df).mark_violin().encode(x="g:N", y="v:Q")
    plain = _paths(chart.to_svg())
    broken = _paths((chart + fm.BreakAxis(axis="y", gap=(250, 650))).to_svg())
    assert plain and broken, "expected a violin body in both renders"
    assert broken != plain, "the violin must be placed on the broken scale, not ignored"
    assert len(broken) == len(plain) == 1, (
        f"the violin stays ONE node (one row); got {len(broken)} vs {len(plain)}"
    )
    # One ring per visible segment, and no ring escapes the canvas.
    assert broken[0].count("M") == 2, (
        f"the spanning violin is cut into one lobe per segment; got {broken[0][:200]}"
    )
    ys = [y for _, y in _path_coords(broken[0])]
    assert all(y > -1000 for y in ys), (
        f"a clipped ring must carry no off-canvas sentinel vertex; got min={min(ys)}"
    )


# --------------------------------------------------------------------------
# vertex splitting (#230-1)
#
# Vertex-placed stroked geometry (line, rule, segment shaft) crossing a gap
# emits ONE SUB-NODE PER VISIBLE SEGMENT: the stroke is cut at the gap edges,
# each piece keeps the vertices lying in one segment plus the interpolated
# crossing points where it enters and leaves, and nothing is drawn across the
# gap. Measured pre-fix symptom: ``<polyline points="74.714,420.961
# 345.357,-99999 616,16">`` — one stroke detouring off-canvas and back.
# --------------------------------------------------------------------------


def _polylines(svg: str) -> list[list[tuple[float, float]]]:
    """Every ``<polyline>``'s vertices, in document order."""
    return [
        [tuple(float(c) for c in pair.split(",")) for pair in points.split()]
        for points in re.findall(r'<polyline points="([^"]*)"', svg)
    ]


def _lines(svg: str, stroke: str = _MARK_FILL) -> list[tuple[float, float, float, float]]:
    """``(x1, y1, x2, y2)`` for every mark ``<line>`` painted in ``stroke``.

    The colour filter is load-bearing: gridlines and tick dashes are ``<line>``
    too, and a hidden GRIDLINE legitimately carries ``BREAK_HIDDEN`` (chrome
    hides whole — spec §4 #230-9), so an unfiltered sentinel check over the
    whole SVG proves nothing about mark geometry.
    """
    pattern = (
        r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" '
        rf'stroke="{re.escape(stroke)}"'
    )
    return [tuple(float(g) for g in m.groups()) for m in re.finditer(pattern, svg)]


def _vertical_strokes(svg: str) -> list[tuple[float, float, float]]:
    """``(x, y1, y2)`` for every tall vertical ``<line>`` thicker than a gridline.

    Finds stroked data geometry — a rule, a box-plot whisker — without coupling
    to a theme colour (a rule's default stroke and the axis chrome's are not
    reliably distinct). Gridlines are 0.5px wide, tick dashes are 4px long, and
    the axis domain line is horizontal, so all three fall outside the filter.
    """
    out = []
    for tag in re.findall(r"<line[^>]*>", svg):
        x1, y1, x2, y2 = (float(v) for v in re.findall(r'"([-\d.]+)"', tag)[:4])
        width = re.search(r'stroke-width="([\d.]+)"', tag)
        if abs(x1 - x2) < 0.01 and abs(y1 - y2) > 20.0 and float(width.group(1)) > 1.0:
            out.append((x1, y1, y2))
    return out


def _mark_paths(svg: str) -> list[str]:
    """The ``d`` of every ``<path>`` painted in the mark colour."""
    return re.findall(rf'<path d="([^"]*)"[^>]*{re.escape(_MARK_FILL)}', svg)


def _line_chart(gap=None, **encode):
    """A four-point line whose third vertex sits inside ``(150, 850)``.

    Vertex 2 (data 100) is interior to the lower segment, so its placed pixel
    proves the piece landed on the COMPRESSED scale rather than being left
    where the unbroken scale put it — the two domain endpoints (10 and 900) are
    fixed points of any break on this axis and discriminate nothing.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 100.0, 500.0, 900.0]})
    chart = fm.Chart(df).mark_line(**encode).encode(x="x:Q", y="y:Q")
    return chart if gap is None else chart + fm.BreakAxis(axis="y", gap=gap)


def test_line_crossing_a_gap_has_no_connecting_segment_and_no_sentinel():
    """The measured #230-1 repro: one stroke per visible piece, no spike."""
    svg = _line_chart(gap=(150, 850)).to_svg()
    pieces = _polylines(svg)
    assert len(pieces) == 2, f"one sub-polyline per visible segment; got {pieces}"
    for piece in pieces:
        for x, y in piece:
            assert y > -1000, f"no sentinel coordinate may survive the split; got ({x}, {y})"
            assert 0.0 <= y <= 500.0, f"every vertex is a real pixel; got ({x}, {y})"
    # The pieces are separated by the break band, so no geometry crosses it.
    lower, upper = sorted(pieces, key=lambda p: -max(y for _, y in p))
    assert max(y for _, y in upper) < min(y for _, y in lower), (
        f"the pieces must not meet across the gap; got {pieces}"
    )


def test_split_line_pieces_land_on_the_compressed_scale():
    """The visible sub-paths sit where the BROKEN scale puts their data.

    Discriminating against "the line was left on the unbroken scale": vertex 2
    (data 100) is interior to the lower segment, so the compression moves it.
    """
    plain = _polylines(_line_chart().to_svg())[0]
    pieces = _polylines(_line_chart(gap=(150, 850)).to_svg())
    lower, upper = sorted(pieces, key=lambda p: -max(y for _, y in p))
    assert lower[1][1] != pytest.approx(plain[1][1], abs=1.0), (
        f"data 100 must move onto segment 0's compressed position; "
        f"plain={plain[1]} broken={lower[1]}"
    )
    assert lower[1][1] < plain[1][1], (
        "and it moves UP: segment 0 stretches data 10..150 over half the plot"
    )
    # Order is preserved: a break does not re-order a series' vertices.
    assert [x for x, _ in lower] == sorted(x for x, _ in lower)
    assert [x for x, _ in upper] == sorted(x for x, _ in upper)


def test_split_line_interpolates_its_crossing_points_at_the_gap_edges():
    """The geometry contract for a vertex INSIDE a gap.

    A y-axis break leaves x untouched, so the crossing x is predictable from the
    unbroken render: the piece must end where the line actually meets the gap
    edge (data y=150, ``(150-100)/(500-100)`` of the way from vertex 2 to vertex
    3), not at the last visible vertex — dropping back to that vertex would bend
    the line and invent a slope the data never had.
    """
    (v1, v2, v3, v4) = _polylines(_line_chart().to_svg())[0]
    pieces = _polylines(_line_chart(gap=(150, 850)).to_svg())
    lower, upper = sorted(pieces, key=lambda p: -max(y for _, y in p))
    # The vertex inside the gap contributes no point of its own: the lower piece
    # is [v1, v2, exit] and the upper [entry, v4].
    assert len(lower) == 3, f"two visible vertices plus the exit crossing; got {lower}"
    assert len(upper) == 2, f"the re-entry crossing plus the last vertex; got {upper}"

    exit_x = v2[0] + (150 - 100) / (500 - 100) * (v3[0] - v2[0])
    entry_x = v3[0] + (850 - 500) / (900 - 500) * (v4[0] - v3[0])
    assert lower[-1][0] == pytest.approx(exit_x, abs=0.01), (
        f"the exit crossing is interpolated along the edge; got {lower[-1]}"
    )
    assert upper[0][0] == pytest.approx(entry_x, abs=0.01), (
        f"the re-entry crossing is interpolated along the edge; got {upper[0]}"
    )
    # The two surviving vertices keep their own x exactly.
    assert (lower[0][0], lower[1][0]) == pytest.approx((v1[0], v2[0]), abs=1e-9)
    assert upper[1][0] == pytest.approx(v4[0], abs=1e-9)
    # And the crossings are not the neighbouring vertices' x.
    assert lower[-1][0] != pytest.approx(v2[0], abs=0.01)
    assert upper[0][0] != pytest.approx(v4[0], abs=0.01)


def test_line_wholly_inside_a_gap_draws_nothing():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [400.0, 600.0]})
    chart = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    svg = (chart + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg()
    assert _polylines(svg) == [], "a stroke with no visible piece is dropped, not hidden"


def test_ranged_rule_spanning_a_gap_splits_into_one_line_per_segment():
    """The line-extent (rule) case: ``y``/``y2`` is a stroked data extent."""
    df = pl.DataFrame({"c": ["A"], "lo": [20.0], "hi": [900.0]})
    chart = fm.Chart(df).mark_rule().encode(x="c:N", y="lo:Q", y2="hi:Q")
    plain = _vertical_strokes(chart.to_svg())
    broken = _vertical_strokes((chart + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg())
    assert len(plain) == 1, f"one rule before the break; got {plain}"
    assert len(broken) == 2, f"one piece per visible segment; got {broken}"
    assert {round(x, 3) for x, _, _ in broken} == {round(plain[0][0], 3)}, (
        "both pieces stay on the rule's own x"
    )
    for _, y1, y2 in broken:
        assert min(y1, y2) > -1000, "no sentinel end"
    lower, upper = sorted(broken, key=lambda line: -max(line[1], line[2]))
    assert max(upper[1], upper[2]) < min(lower[1], lower[2]), (
        f"the pieces are separated by the break; got {broken}"
    )


def test_boxplot_whiskers_crossing_a_gap_split_rather_than_spike():
    """``mark_boxplot`` draws its whiskers as a ranged ``rule`` layer, so the
    split is what keeps a box-plot honest under a break: the whisker stops at
    the gap edge instead of running off-canvas."""
    rng = np.random.default_rng(7)
    values = list(rng.normal(100, 10, 30)) + list(rng.normal(900, 10, 30))
    df = pl.DataFrame({"g": ["a"] * 60, "v": values})
    chart = fm.Chart(df).mark_boxplot(outliers=False).encode(x="g:N", y="v:Q")
    plain = _vertical_strokes(chart.to_svg())
    broken = _vertical_strokes((chart + fm.BreakAxis(axis="y", gap=(300, 700))).to_svg())
    assert len(plain) == 1, f"one whisker rule before the break; got {plain}"
    assert len(broken) == 2, f"the whisker is cut into two pieces; got {broken}"
    for _, y1, y2 in broken:
        assert min(y1, y2) > -1000, f"no whisker may reach the sentinel; got {broken}"
    lower, upper = sorted(broken, key=lambda line: -max(line[1], line[2]))
    assert max(upper[1], upper[2]) < min(lower[1], lower[2]), (
        f"the whisker stops at the gap edge on both sides; got {broken}"
    )


def test_segment_with_an_arrow_keeps_its_head_on_the_tip_piece():
    """A split shaft's arrowhead rides the piece that REACHES the tip, at its
    own drawn size and pointing along that piece."""
    df = pl.DataFrame({"x1": [1.0], "y1": [20.0], "x2": [3.0], "y2": [900.0]})
    chart = (
        fm.Chart(df)
        .mark_segment(arrow=True, arrow_size=10)
        .encode(x="x1:Q", y="y1:Q", x2="x2:Q", y2="y2:Q")
    )
    plain_svg = chart.to_svg()
    svg = (chart + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg()

    shafts = _lines(svg)
    assert len(shafts) == 2, f"the shaft is cut into two pieces; got {shafts}"
    for x1, y1, x2, y2 in shafts:
        assert min(y1, y2) > -1000, f"no shaft piece may reach the sentinel; got {shafts}"
    heads = [_path_coords(d) for d in _mark_paths(svg) if len(_path_coords(d)) == 3]
    assert len(heads) == 1, f"exactly one arrowhead survives; got {heads}"
    assert all(y > -1000 for _, y in heads[0]), "nor may the head"

    # The head's tip is the placed end of the piece that reaches it.
    tip = heads[0][0]
    tip_piece = max(shafts, key=lambda line: max(line[0], line[2]))
    assert (tip[0], tip[1]) == pytest.approx((tip_piece[2], tip_piece[3]), abs=0.01), (
        f"the head sits on the tip piece's end; got tip={tip} piece={tip_piece}"
    )

    def head_size(vertices):
        (tx, ty), (lx, ly), (rx, ry) = vertices
        return ((tx - (lx + rx) / 2) ** 2 + (ty - (ly + ry) / 2) ** 2) ** 0.5

    plain_head = [_path_coords(d) for d in _mark_paths(plain_svg) if len(_path_coords(d)) == 3]
    assert head_size(heads[0]) == pytest.approx(head_size(plain_head[0]), abs=0.01), (
        "the break moves the head, it does not resize it"
    )
    # And it points along the piece it terminates, whose slope the break changed.
    head_angle = math.atan2(
        tip[1] - (heads[0][1][1] + heads[0][2][1]) / 2,
        tip[0] - (heads[0][1][0] + heads[0][2][0]) / 2,
    )
    piece_angle = math.atan2(tip_piece[3] - tip_piece[1], tip_piece[2] - tip_piece[0])
    assert head_angle == pytest.approx(piece_angle, abs=0.02), (
        f"head={head_angle} vs tip piece={piece_angle}"
    )


def test_arrowhead_whose_tip_is_in_a_gap_is_dropped():
    """No piece reaches an in-gap tip, so there is nothing for a head to
    terminate: it is dropped rather than parked off-canvas."""
    df = pl.DataFrame({"x1": [1.0], "y1": [20.0], "x2": [3.0], "y2": [500.0]})
    chart = (
        fm.Chart(df)
        .mark_segment(arrow=True, arrow_size=10)
        .encode(x="x1:Q", y="y1:Q", x2="x2:Q", y2="y2:Q")
    )
    svg = (chart + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg()
    assert [d for d in _mark_paths(svg) if len(_path_coords(d)) == 3] == [], (
        "the head goes with the tip it has no shaft for"
    )
    shafts = _lines(svg)
    assert len(shafts) == 1, f"the shaft keeps its one visible piece; got {shafts}"
    assert min(shafts[0][1], shafts[0][3]) > -1000, "and it carries no sentinel end"


def test_step_interpolated_line_splits_under_a_break():
    """A ``step`` line is straight-line geometry spelled as a path, so it
    splits: the boundary is the GEOMETRY, not the node kind."""
    svg = _line_chart(gap=(150, 850), interpolate="step").to_svg()
    open_paths = [d for d in _paths(svg) if not d.endswith("Z")]
    step_paths = [d for d in open_paths if "L" in d or "M" in d]
    assert len(step_paths) == 2, f"one sub-path per visible segment; got {step_paths}"
    for d in step_paths:
        for _, y in _path_coords(d):
            assert y > -1000, f"a split step line carries no sentinel; got {d}"


def test_step_line_entirely_on_one_side_of_the_gap_keeps_its_hv_shorthand():
    """Spec-review cycle-2 finding 1: the split's re-spelling (``H``/``V``
    shorthand rewritten as explicit ``L`` vertices) must be confined to a
    piece a gap actually CUTS, not fired by every step path under an
    effective break. A step line whose every vertex sits on one side of the
    gap comes back from the split as ONE piece spanning the whole source, so
    it keeps its unbroken command spelling — contrasted directly against the
    same shape under a gap wide enough to reach it, which genuinely
    re-spells. This is the reviewer's measured repro (an uncut step line's
    bytes changing SPELLING, not just position), flipped.

    The y scale's domain is pinned to ``[0, 1000]`` (well past the line's own
    10..100 data range) so a gap can sit inside the DOMAIN — and so be
    effective — while remaining entirely outside the line's own vertices and
    edges, which a domain that tracked the data exactly would make
    impossible for a monotonic series.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 30.0, 60.0, 100.0]})
    wide_y = fm.Y("y:Q", scale=fm.LinearScale(domain=[0.0, 1000.0]))
    chart = fm.Chart(df).mark_line(interpolate="step").encode(x="x:Q", y=wide_y)
    plain_cmds = re.findall(r"[MLHV]", _paths(chart.to_svg())[0])

    # Gap (500, 900) is inside the pinned domain but nowhere near this line's
    # own y range (10..100): an effective break the path never crosses.
    broken_svg = (chart + fm.BreakAxis(axis="y", gap=(500, 900))).to_svg()
    broken_paths = [d for d in _paths(broken_svg) if not d.endswith("Z")]
    assert len(broken_paths) == 1, f"never reaches the gap, so it is not split; got {broken_paths}"
    broken_cmds = re.findall(r"[MLHV]", broken_paths[0])
    assert broken_cmds == plain_cmds, (
        f"an effective break this line never crosses must keep the original "
        f"command spelling; plain={plain_cmds} broken={broken_cmds}"
    )
    assert "L" not in broken_cmds, f"H/V shorthand must survive uncut; got {broken_cmds}"

    # The same shape, with a gap narrow enough to sit BETWEEN two of its
    # vertices (data 30 and 100 both survive, one on each side; data 60 is
    # inside the gap): this one IS cut, and genuinely re-spells.
    crossing_svg = (chart + fm.BreakAxis(axis="y", gap=(45, 65))).to_svg()
    crossing_paths = [d for d in _paths(crossing_svg) if not d.endswith("Z")]
    assert len(crossing_paths) == 2, f"this one is cut at the gap; got {crossing_paths}"
    for d in crossing_paths:
        assert "L" in re.findall(r"[MLHV]", d), f"a genuinely cut piece still re-spells; got {d}"


def _path_numbers(d: str) -> list[float]:
    """Every number a path's ``d`` carries, whatever command spells it.

    ``_path_coords`` reads ``M``/``L`` pairs only, so it cannot see the
    coordinate an ``H``/``V`` shorthand carries — and a sentinel emitted by the
    per-vertex fallback on a step path lands in exactly one of those.
    """
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]


def _boundary_in_gap_chart(axis: str, interpolate: str, position: str):
    """A four-vertex step line whose FIRST or LAST vertex lies inside the gap.

    The break axis' domain is pinned to ``[0, 1000]`` so the gap ``(150, 850)``
    is effective while the line's other three vertices stay on one side of it:
    the shape in which a cut DROPS one boundary vertex and INSERTS one crossing,
    leaving the piece count and the vertex count unchanged.
    """
    values = [10.0, 30.0, 60.0, 500.0] if position == "last" else [500.0, 900.0, 920.0, 950.0]
    other = [1.0, 2.0, 3.0, 4.0]
    pinned = fm.LinearScale(domain=[0.0, 1000.0])
    if axis == "y":
        df = pl.DataFrame({"x": other, "y": values})
        encode = {"x": "x:Q", "y": fm.Y("y:Q", scale=pinned)}
    else:
        df = pl.DataFrame({"x": values, "y": other})
        encode = {"x": fm.X("x:Q", scale=pinned), "y": "y:Q"}
    chart = fm.Chart(df).mark_line(interpolate=interpolate).encode(**encode)
    return chart + fm.BreakAxis(axis=axis, gap=(150, 850))


@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize("interpolate", ["step", "step-before", "step-after"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_step_line_with_a_boundary_vertex_in_a_gap_is_cut_not_spiked(axis, interpolate, position):
    """Spec-review cycle-3 finding: a cut that drops a BOUNDARY vertex and
    inserts a crossing in the same edge nets to zero point-count change, so the
    split returns one piece carrying the source's own vertex count — a shape an
    uncut walk also produces. Cycle 2 inferred "uncut" from those counts and
    handed the path to the per-vertex fallback, which parks the in-gap vertex at
    ``BREAK_HIDDEN``: the #230-1 spike, back in a straight step line.
    ``split_polyline`` now REPORTS cut-ness, so the fallback is only ever
    reached by a caller that chose it.

    Every axis × step spelling × boundary end, because which end nets to zero
    depends on where the riser sits in the spelling — the reviewer reproduced it
    on five separate shapes.
    """
    svg = _boundary_in_gap_chart(axis, interpolate, position).to_svg()
    paths = _mark_paths(svg)
    assert paths, "the visible part of the line must still draw"
    for d in paths:
        for n in _path_numbers(d):
            assert n > -1000.0, f"no sentinel coordinate may survive the split; got {d}"
            # Not merely "not the sentinel": a real pixel, inside the canvas.
            assert -50.0 < n < 1200.0, f"every coordinate is a real pixel; got {d}"


def test_two_series_sharing_a_domain_with_one_ending_in_a_gap_has_no_sentinel():
    """The same defect from ORDINARY user code — no pinned domain, no exotic
    API, just two series sharing one y domain where the shorter one's last
    vertex falls inside the gap (the reviewer's measured cycle-2 repro, whose
    series "a" rendered ``M74.714 412.961 H321.99 V373.281 H569.265 V-99999``).
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [10.0, 30.0, 500.0, 10.0, 400.0, 900.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )
    chart = fm.Chart(df).mark_line(interpolate="step-after").encode(
        x="x:Q", y="y:Q", color="g:N"
    ) + fm.BreakAxis(axis="y", gap=(150, 850))
    svg = chart.to_svg()
    paths = [d for d in _paths(svg) if not d.endswith("Z") and _path_numbers(d)]
    assert len(paths) >= 2, f"both series must still draw; got {paths}"
    for d in paths:
        for n in _path_numbers(d):
            assert n > -1000.0, f"series ending in the gap must be CUT, not spiked; got {d}"


def test_curve_interpolated_line_keeps_the_per_vertex_fallback():
    """The other side of that boundary: a CURVED path is not split.

    Cutting a cubic honestly needs subdivision of the placed curve; guessing a
    straight cut would silently redraw it. The per-vertex model's sentinel is
    the documented degradation, and this test is what makes the boundary
    explicit rather than incidental.
    """
    svg = _line_chart(gap=(150, 850), interpolate="basis").to_svg()
    curved = [d for d in _paths(svg) if "C" in d]
    assert len(curved) == 1, f"a curve stays one node; got {curved}"
    assert "-99999" in curved[0], (
        "the curve keeps the per-vertex fallback, sentinel included — if this "
        f"changes, the split boundary moved and the docs must move with it; got {curved[0]}"
    )


def test_split_line_pieces_realign_their_batch_sidecars():
    """1→N splitting goes through the clip machinery's sidecar realignment, so
    each piece inherits its source row's ``data_indices``/``keys``.

    Keys DUPLICATE across pieces for now — suffixing is #230-5's own work
    (spec §4.5); what this pins is that the realignment happens at all, i.e.
    that a split batch's sidecars stay one-entry-per-node.
    """
    # Local, like `test_finding_p1.py`'s uses of it: `_render_scene` is the
    # private scene-JSON door, and the import documents that at the use site.
    from ferrum._scene import _render_scene

    df = pl.DataFrame(
        {
            # Series "a" crosses the gap (two pieces); "b" stays below it (one).
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [10.0, 500.0, 900.0, 20.0, 60.0, 100.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )
    chart = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q", color="g:N", key="g") + fm.BreakAxis(
        axis="y", gap=(150, 850)
    )
    scene = json.loads(_render_scene(chart)[0])
    batches = [b for b in scene["panels"][0]["marks"] if b.get("kind") == "line"]
    assert batches, "expected a line batch"
    nodes, indices, keys = 0, [], []
    for batch in batches:
        nodes += len(batch["nodes"])
        indices += batch.get("data_indices") or []
        keys += batch.get("keys") or []
    assert nodes == 3, f"two pieces for series a, one for series b; got {nodes}"
    assert len(indices) == nodes, f"one data_index per node; got {indices}"
    assert len(keys) == nodes, f"one key per node; got {keys}"
    assert keys == ["a", "a", "b"], f"each piece inherits its source row's key; got {keys}"
    assert indices[0] == indices[1] != indices[2], (
        f"the two pieces of series a share its row; got {indices}"
    )


# --------------------------------------------------------------------------
# area borders (#230-1, quality-review cycle-3 finding)
#
# An ``Area`` batch carries BOTH halves of one mark: the closed fill, which the
# extent clip cuts per segment, and ``mark_area(line=True)``/``borders=True``'s
# OPEN border stroke. The border used to take the per-vertex fallback, so a
# broken chart rendered a correctly clipped fill underneath an unclipped
# outline. Measured pre-fix symptoms, both from ordinary user code: an in-gap
# border vertex parked at the sentinel (``M79.022 333.137 L327.511 -99999
# L576 94.24``) and, with no in-gap vertex at all, one connected stroke running
# straight through the break band over a fill split in two.
# --------------------------------------------------------------------------


def _area_border_paths(svg: str) -> list[str]:
    """The ``d`` of every area BORDER stroke — an open, unfilled mark path."""
    return re.findall(rf'<path d="([^"]*)" fill="none" stroke="{re.escape(_MARK_FILL)}"', svg)


def _area_fill_bands(svg: str) -> list[str]:
    """The ``d`` of every area FILL band — a closed, translucently filled path."""
    return [d for d in re.findall(r'<path d="([^"]*)" fill="rgba\(', svg) if d.endswith("Z")]


def _area_border_chart(values: list[float], gap, **style):
    """An area whose top edge crosses ``gap``, with the y domain pinned.

    The domain is pinned to ``[0, 1000]`` so the gap is effective independently
    of the values chosen, exactly as ``_boundary_in_gap_chart`` does.
    """
    df = pl.DataFrame({"x": [float(i + 1) for i in range(len(values))], "y": values})
    chart = (
        fm.Chart(df)
        .mark_area(**style)
        .encode(x="x:Q", y=fm.Y("y:Q", scale=fm.LinearScale(domain=[0.0, 1000.0])))
    )
    return chart + fm.BreakAxis(axis="y", gap=gap)


def test_area_border_with_a_vertex_in_a_gap_is_cut_not_spiked():
    """The measured repro: the middle vertex (data 500) is inside the gap, so
    the border must be CUT on either side of it rather than parked at
    ``BREAK_HIDDEN`` and drawn as a spike off-canvas and back."""
    svg = _area_border_chart([10.0, 500.0, 900.0], (200, 800), line=True).to_svg()
    borders = _area_border_paths(svg)
    assert len(borders) == 2, f"one border piece each side of the in-gap vertex; got {borders}"
    for d in borders:
        for n in _path_numbers(d):
            assert n > -1000.0, f"no sentinel may survive in a border stroke; got {d}"
            assert -50.0 < n < 1200.0, f"every border coordinate is a real pixel; got {d}"


@pytest.mark.parametrize("style", ["line", "borders"])
def test_area_border_crossing_a_gap_is_cut_like_the_fill_beneath_it(style):
    """The subtler half: with NO vertex inside the gap the old fallback emitted
    no sentinel at all, it just drew one connected stroke straight across the
    break band while the fill under it was clipped into two — the two halves of
    one mark disagreeing.

    Pinned two ways: the piece count (the top edge crosses the gap twice, so it
    is cut into three), and the invariant behind it — no border EDGE may span
    the empty band the fill's own clip leaves behind. The second check is what
    catches a straight-across stroke, whose vertices all sit outside the band
    even though the line between them crosses it.
    """
    svg = _area_border_chart([50.0, 900.0, 950.0, 60.0], (150, 850), **{style: True}).to_svg()
    bands = _area_fill_bands(svg)
    assert len(bands) == 2, f"the fill itself clips into two bands; got {bands}"
    # The empty pixel strip between the two clipped fill bands: the upper band
    # (higher data, smaller y) ends at its own max y, the lower one starts at
    # its min y.
    upper, lower = sorted(bands, key=lambda d: max(y for _, y in _path_coords(d)))
    band_top = max(y for _, y in _path_coords(upper))
    band_bottom = min(y for _, y in _path_coords(lower))
    assert band_top < band_bottom, f"the two fill bands are separated; got {bands}"

    borders = _area_border_paths(svg)
    # The top edge crosses the gap twice: below, above, below again. `borders`
    # adds the flat baseline edge, which lies wholly in the lower segment.
    expected = {"line": 3, "borders": 4}[style]
    assert len(borders) == expected, f"one piece per visible run of the border; got {borders}"
    for d in borders:
        coords = _path_coords(d)
        for (_, y0), (_, y1) in zip(coords, coords[1:]):
            assert not (min(y0, y1) < band_top and max(y0, y1) > band_bottom), (
                f"no border edge may span the break band ({band_top}, {band_bottom}); got {d}"
            )


def test_area_without_borders_is_unaffected_by_the_border_split():
    """A plain ``mark_area`` emits no border node at all, so routing the open
    path through the split cannot move the five break goldens (all of which
    carry point, bar and plain-area geometry)."""
    svg = _area_border_chart([50.0, 900.0, 950.0, 60.0], (150, 850)).to_svg()
    assert _area_border_paths(svg) == [], "a plain area has no border stroke"
    assert len(_area_fill_bands(svg)) == 2, "and its fill still clips into two bands"


def test_line_chart_with_an_ineffective_gap_is_identical_to_unbroken():
    """The split must not perturb geometry a break does not touch: a gap
    outside the domain leaves a line chart byte-identical to the unbroken one
    (the C1 pin's line-mark counterpart)."""
    chart = _line_chart()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = _line_chart(gap=(5000, 6000)).to_svg()
    assert _polylines(broken) == _polylines(chart.to_svg()), (
        "an ineffective gap must leave the stroke exactly where it was"
    )


# --------------------------------------------------------------------------
# riders
# --------------------------------------------------------------------------


def test_secondary_y_layer_is_exempt_from_the_primary_axis_break():
    """A SecondaryY layer maps through its own scale, so the break is not its."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [10.0, 25.0, 15.0], "revenue": [100.0, 200.0, 150.0]}
    )
    base = fm.Chart(df).mark_point().encode(x="x", y="y") + fm.SecondaryY(
        "revenue", mark="point", color="#e45756"
    )
    secondary = lambda svg: re.findall(  # noqa: E731 - local extractor
        r'<circle cx="([-\d.]+)" cy="([-\d.]+)"[^>]*fill="#e45756"', svg
    )
    plain = secondary(base.to_svg())
    broken_svg = (base + fm.BreakAxis(axis="y", gap=(14, 22))).to_svg()
    assert secondary(broken_svg) == plain, (
        "secondary-axis marks must be untouched by a primary-axis break"
    )
    # ... while the primary layer *is* remapped, so the test discriminates.
    primary = re.findall(r'<circle cx="[-\d.]+" cy="([-\d.]+)"[^>]*fill="#2563eb"', broken_svg)
    assert any(float(cy) < -1000 for cy in primary), (
        f"the primary layer's in-gap mark should be hidden; got {primary}"
    )


def test_label_leader_line_keeps_its_data_end_on_the_data_point():
    """A leader line's two ends are two positions, so each is placed on its own.

    ``mark_label(leader_line=True)`` draws a line FROM the data point TO the
    label. Moving it as one rigid glyph (by its midpoint's displacement) pulled
    the data end ~25px off the point it exists to touch. The data end must land
    exactly where the same row's point mark lands.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 500.0, 900.0], "t": ["a", "b", "c"]})
    brk = fm.BreakAxis(axis="y", gap=(150, 850))
    labels = fm.Chart(df).mark_label(leader_line=True).encode(x="x:Q", y="y:Q", text="t:N")
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")

    leader_ends = [
        float(m.group(1))
        for m in re.finditer(
            r'<line x1="[-\d.]+" y1="([-\d.]+)"[^>]*stroke-width="0.75"', (labels + brk).to_svg()
        )
    ]
    point_ys = [
        float(m) for m in re.findall(r'<circle[^>]*cy="([-\d.]+)"', (points + brk).to_svg())
    ]
    assert leader_ends and len(leader_ends) == len(point_ys), (
        f"expected one leader per row; got {leader_ends} vs {point_ys}"
    )
    assert leader_ends == point_ys, (
        f"each leader's data end must sit exactly where its data point sits; "
        f"leaders={leader_ends} points={point_ys}"
    )


def test_rug_dashes_keep_their_drawn_length_under_a_break():
    """A tick dash is drawn size, not data — the break moves it, intact.

    ``SceneNode::Line`` has two producers with opposite meanings: a label's
    leader (a connector between two data positions) and a tick's dash (a fixed
    8px mark). Scoping the connector rule by NODE kind collapsed every rug dash
    lying along the broken axis to zero length; the rule is scoped by BATCH
    kind instead, so a rug survives.
    """
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"v": rng.normal(100, 30, 200)})
    chart = fm.displot(df, x="v", rug=True)

    def dash_lengths(svg: str) -> collections.Counter:
        lengths: collections.Counter = collections.Counter()
        for m in re.finditer(
            r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"', svg
        ):
            x1, y1, x2, y2 = (float(g) for g in m.groups())
            if abs(x2 - x1) < 0.01:
                lengths[round(abs(y2 - y1), 3)] += 1
        return lengths

    plain = dash_lengths(chart.to_svg())
    broken = dash_lengths((chart + fm.BreakAxis(axis="y", gap=(0.004, 0.008))).to_svg())
    assert plain[8.0] == 200, f"expected 200 8px rug dashes in the plain render; got {plain}"
    assert broken[8.0] == 200, (
        f"every rug dash must keep its drawn length through the break; got {broken}"
    )
    assert broken[0.0] == 0, f"no rug dash may collapse to zero length; got {broken}"


def test_axis_title_survives_a_break_across_its_midpoint():
    """The axis title is not positioned by a data value, so it is not remapped."""
    svg = _bar_svg([72.0, 68.0, 75.0, 900.0, 81.0], (150, 850))
    titles = re.findall(
        r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*font-weight="600"[^>]*>([^<]*)</text>', svg
    )
    by_content = {content: (float(x), float(y)) for x, y, content in titles}
    assert "v" in by_content, f"y-axis title missing entirely; got {titles}"
    assert by_content["v"][1] > 0, (
        f"the y-axis title was teleported off-screen by the break: {by_content['v']}"
    )
    assert "c" in by_content and by_content["c"][0] > 0


def test_in_gap_ticks_and_gridlines_are_hidden_and_the_rest_compress():
    svg = _bar_svg([72.0, 68.0, 75.0, 900.0, 81.0], (150, 850))
    labels = {
        m.group(2): float(m.group(1))
        for m in re.finditer(
            r'<text x="[-\d.]+" y="([-\d.]+)"[^>]*text-anchor="end">(\d+)</text>', svg
        )
    }
    assert labels, "expected numeric y tick labels"
    for value, y in labels.items():
        if 150 < float(value) < 850:
            assert y < -1000, f"tick {value} is inside the gap and must be hidden; got {y}"
        else:
            assert y > 0, f"tick {value} is outside the gap and must stay visible; got {y}"
    # Ticks retain their order across the break.
    assert labels["900"] < labels["100"] < labels["0"]

    # Horizontal gridlines (y1 == y2) follow their ticks: one hidden per hidden
    # tick, the rest at compressed positions inside the plot.
    grid_ys = [
        float(m.group(1))
        for m in re.finditer(
            r'<line x1="[-\d.]+" y1="([-\d.]+)" x2="[-\d.]+" y2="\1" stroke="#d6d3d1"', svg
        )
    ]
    assert grid_ys, "expected horizontal gridlines"
    hidden_ticks = sum(1 for v in labels if 150 < float(v) < 850)
    assert sum(1 for y in grid_ys if y < -1000) == hidden_ticks, (
        f"one hidden gridline per in-gap tick; ticks={labels} grid={grid_ys}"
    )
    assert all(y > 0 for y in grid_ys if y > -1000), (
        f"surviving gridlines stay on canvas: {grid_ys}"
    )


def test_a_gap_touching_the_domain_edge_reserves_no_indicator_space():
    """No indicator is drawn beside a domain edge, so no pixels are held for one.

    Reservation counts indicators emitted (one between each pair of adjacent
    segments), not removed ranges: an edge gap leaves a single segment and no
    indicator, and used to withhold ``break_size`` pixels for it — a 12px dead
    band that pushed the tallest bar down to y=28 where a mid-domain gap put
    it at y=16.
    """
    mid = _rects(_bar_svg([300.0, 1000.0], (400, 500)))
    edge = _rects(_bar_svg([300.0, 1000.0], (0, 500)))
    top_of = lambda rects: min(r[1] for r in rects)  # noqa: E731 - local extractor
    assert top_of(mid) == pytest.approx(top_of(edge), abs=1e-6), (
        f"an edge gap must reach the same plot top as a mid-domain one; "
        f"mid={top_of(mid)} edge={top_of(edge)}"
    )


@pytest.mark.parametrize("style", ["slash", "zigzag", "wave", "gap"])
def test_break_indicator_styles_still_render_with_clipped_bars(style):
    df = pl.DataFrame({"c": ["A", "B"], "v": [72.0, 900.0]})
    svg = (
        fm.Chart(df).mark_bar().encode(x="c:N", y="v:Q")
        + fm.BreakAxis(axis="y", gap=(150, 850), break_style=style)
    ).to_svg()
    assert len(_rects(svg)) == 3, "one short bar + the crossing bar's two pieces"


# --------------------------------------------------------------------------
# annotation suppression + remap (spec §4 H5, F-L08-07)
# --------------------------------------------------------------------------
#
# The guide (docs/site/guide/concepts/break-axes.md) promises: "Annotations
# whose positions fall within a break region are suppressed with a warning.
# The warning names the annotation and its position." Pre-fix, the annotation
# rendered anyway with no warning, and a SURVIVING annotation was not even
# position-remapped through the break — it pointed at the wrong data.


def test_annotation_inside_a_gap_is_suppressed_with_a_named_warning():
    """The finding's exact repro: a labelled hline inside the gap is dropped.

    Pre-fix: "inside the break" rendered in the SVG and no warning fired.
    Rolled up per #240-3: the warning names the axis and how many instances
    were suppressed rather than each instance's own value/gap (an
    `annotate_hline` desugars to a `line` primitive plus its own label
    `text` primitive, so one user-facing annotation suppresses as 2).
    """
    df = pl.DataFrame(
        {"category": ["A", "B", "C", "D", "E"], "value": [72.0, 68.0, 75.0, 900.0, 81.0]}
    )
    chart = (
        fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
        + fm.BreakAxis(axis="y", gap=(150, 850))
        + fm.annotate_hline(y=500.0, label="inside the break")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "inside the break" not in svg, "the suppressed label must not render"
    messages = [str(w.message) for w in caught]
    matches = [m for m in messages if "break" in m.lower() and "y axis" in m and "suppressed" in m]
    assert matches, f"expected a warning naming the annotation's axis; got {messages}"
    assert len(matches) == 1, (
        f"N suppressed instances must roll up into ONE warning; got {messages}"
    )


def test_boundary_value_annotation_is_not_suppressed():
    """Gaps are open intervals: a value exactly on the gap edge survives."""
    df = pl.DataFrame({"category": ["A", "B"], "value": [72.0, 900.0]})
    chart = (
        fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
        + fm.BreakAxis(axis="y", gap=(150, 850))
        + fm.annotate_hline(y=150.0, label="on the edge")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "on the edge" in svg, "a boundary value must not be suppressed"
    assert not any("break" in str(w.message).lower() for w in caught), (
        f"no suppression warning expected for a boundary value; got "
        f"{[str(w.message) for w in caught]}"
    )


def test_span_inside_a_gap_is_suppressed():
    """A `Span` with nothing outside the gap suppresses (#228's drop case).

    Rewritten for #228's stratified clip: a span is suppressed because NOTHING
    OF IT SURVIVES the gaps, not because a gap touches it — a span that
    straddles a gap now clips into one piece per visible segment
    (`tests/test_break_axis_annotation_clip.py`). This one's range (300, 700)
    lies wholly inside the gap, so there is nothing to clip.

    Both of its endpoints are real coordinates inside the gap, which is what
    picks the rollup bucket: `AnnotationSuppressedByBreak`, not
    `AnnotationSpansBreakGap`. The message names the axis and count (#240-3),
    not the specific kind.
    """
    df = pl.DataFrame({"category": ["A", "B"], "value": [72.0, 900.0]})
    chart = (
        fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q")
        + fm.BreakAxis(axis="y", gap=(150, 850))
        + fm.Annotate(ann.span("y", 300.0, 700.0, fill="#eeeeee"))
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert 'fill="#eeeeee"' not in svg, "the suppressed span must not render its rect"
    messages = [str(w.message) for w in caught]
    assert any("y axis" in m and "break" in m.lower() and "suppressed" in m for m in messages), (
        f"expected a suppression warning naming the y axis; got {messages}"
    )


def _horiz_line_ys(svg: str) -> collections.Counter:
    """Multiset of y-values shared by a horizontal `<line>`'s two endpoints.

    A `Counter`, not a `set`: the annotation's compressed position can — and,
    in the pass cases here, DOES — land exactly on an existing gridline's y,
    so a set-difference against the no-annotation render is always empty
    even when the annotation is genuinely present and correctly placed.
    Only a per-position COUNT can tell "one more line at this y" apart from
    "no new line at all."
    """
    return collections.Counter(
        float(m.group(1))
        for m in re.finditer(r'<line x1="-?[\d.]+" y1="(-?[\d.]+)" x2="-?[\d.]+" y2="\1"', svg)
    )


def _count_at(counts: collections.Counter, target: float) -> int:
    """The `Counter` entry whose key is within float noise of `target`."""
    for key, n in counts.items():
        if abs(key - target) < 1e-6:
            return n
    return 0


def test_surviving_annotation_lands_at_the_marks_compressed_position():
    """A surviving data-space annotation moves exactly like a mark does.

    Pre-fix, a surviving annotation was never remapped at all and stayed at
    the unbroken linear pixel. Discriminates via a per-position COUNT delta
    against the same broken chart rendered WITHOUT the annotation
    (quality-review cycle-3): a bare `any(mark_cy in <all horizontal
    lines>)` membership check is vacuous here, and even a SET-difference
    delta is — the annotation's correct, compressed position lands EXACTLY
    on an existing gridline's y, so both shapes pass with the annotation
    deleted entirely (the set of distinct y-values doesn't change; only the
    COUNT at that one y-value does).

    The reference value is deliberately NOT a domain endpoint (#226): the
    segments tile the scale's own padded range, so the extreme values land on
    exactly the pixels the unbroken scale gives them and the closing
    "the break moved it" discrimination below would be vacuous. y=100 sits
    inside the (0, 900) domain, so the break genuinely compresses it.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 100.0, 900.0]})
    brk = fm.BreakAxis(axis="y", gap=(150, 850))
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    ref = fm.annotate_hline(y=100.0)  # unlabelled: one AnnotationLine, "line" kind

    broken_with = (points + ref + brk).to_svg()
    broken_without = (points + brk).to_svg()
    counts_with = _horiz_line_ys(broken_with)
    counts_without = _horiz_line_ys(broken_without)

    pairs = [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'<circle cx="(-?[\d.]+)" cy="(-?[\d.]+)"', broken_with)
    ]
    assert len(pairs) == 3, f"expected all three point marks; got {pairs}"
    # x=2.0 carries y=100, and is the middle cx on this quantitative x scale.
    y100_cy = sorted(pairs)[1][1]

    assert _count_at(counts_with, y100_cy) == _count_at(counts_without, y100_cy) + 1, (
        f"adding the annotation must add exactly one horizontal line at the "
        f"mark's compressed position {y100_cy}; with={dict(counts_with)} "
        f"without={dict(counts_without)}"
    )

    # Discriminate against a second pre-fix bug: without the break, the
    # mark's own position is different (the unbroken linear position) —
    # proving the break actually moved both the mark and the annotation.
    plain_svg = (points + ref).to_svg()
    plain_pairs = [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'<circle cx="(-?[\d.]+)" cy="(-?[\d.]+)"', plain_svg)
    ]
    plain_y100_cy = sorted(plain_pairs)[1][1]
    assert abs(y100_cy - plain_y100_cy) > 1.0, (
        "the break must move the mark (and the annotation with it); "
        f"broken={y100_cy} unbroken={plain_y100_cy}"
    )


def test_norm_coordinate_annotation_is_untouched_by_a_break():
    """Norm/px annotation coordinates are plot-area-relative, never remapped."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [100.0, 900.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    label = fm.annotate_text(fm.norm(0.5), fm.norm(0.5), "center")

    def text_pos(svg: str) -> tuple[float, float]:
        m = re.search(r'<text x="(-?[\d.]+)" y="(-?[\d.]+)"[^>]*>center</text>', svg)
        assert m, f"expected the annotation text in the SVG; got: {svg[:500]}"
        return (float(m.group(1)), float(m.group(2)))

    plain_pos = text_pos((chart + label).to_svg())
    broken_pos = text_pos((chart + label + fm.BreakAxis(axis="y", gap=(150, 850))).to_svg())
    assert plain_pos == broken_pos, (
        f"a norm-coordinate annotation must not move under a break; "
        f"plain={plain_pos} broken={broken_pos}"
    )


# --------------------------------------------------------------------------
# cycle-2 fixes: ordinal-axis filtering, multi-break fold, x-axis parity
# (quality-review cycle-2, .sdd/task-4-rust-quality-reviewer.md)
# --------------------------------------------------------------------------


def test_ordinal_axis_break_leaves_annotations_and_bars_unmoved():
    """An ordinal axis has no continuous domain, so its `BreakAxis` is a
    documented no-op for marks; annotations must agree (finding #1).

    Pre-fix (measured by the reviewer): a categorical-x bar chart's
    `annotate_vline` moved 123px under `BreakAxis(axis="x")` while the bars
    stayed put, and an in-"gap" vline was silently suppressed against a
    fabricated `(0.0, 1.0)` domain the ordinal scale never had.
    """
    df = pl.DataFrame({"c": ["A", "B", "C", "D", "E"], "v": [72.0, 68.0, 75.0, 90.0, 81.0]})
    base = fm.Chart(df).mark_bar().encode(x="c:N", y="v:Q")
    labelled = base + fm.annotate_vline(x=0.8, label="L") + fm.annotate_vline(x=0.45, label="M")

    plain_svg = labelled.to_svg()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        broken_svg = (labelled + fm.BreakAxis(axis="x", gap=(0.3, 0.6))).to_svg()

    messages = [str(w.message) for w in caught]
    assert not any("suppressed" in m for m in messages), (
        f"an ordinal-axis BreakAxis must not suppress annotations; got {messages}"
    )
    # It DOES say it did nothing (close-review cycle-2, N1): an ordinal axis has
    # no numeric domain for a gap to clip against, so the break is a no-op — and
    # a structural feature that silently does nothing is indistinguishable from
    # one the renderer failed to honor. Before that fix the check ran against a
    # fabricated `(0.0, 1.0)` domain, so this very gap "survived" it: no warning,
    # and a slash indicator painted over a scale nothing had moved on.
    assert any("had no effect" in m and "ordinal" in m for m in messages), (
        f"an ineffective ordinal break must say so; got {messages}"
    )
    for label in ("L", "M"):
        assert re.search(rf'<text x="-?[\d.]+" y="-?[\d.]+"[^>]*>{label}</text>', broken_svg), (
            f"label {label!r} must render (not suppressed) on an ordinal-axis break"
        )

    plain_bar_x = sorted(r[0] for r in _rects(plain_svg))
    broken_bar_x = sorted(r[0] for r in _rects(broken_svg))
    assert plain_bar_x == broken_bar_x, "bars must not move on an ordinal-axis break"

    def label_x(svg: str, label: str) -> float:
        m = re.search(rf'<text x="(-?[\d.]+)"[^>]*>{label}</text>', svg)
        assert m, f"expected the {label!r} label; got: {svg[:500]}"
        return float(m.group(1))

    for label in ("L", "M"):
        assert label_x(plain_svg, label) == label_x(broken_svg, label), (
            f"the {label!r} annotation must not move on an ordinal-axis break either"
        )

    # And it draws no indicator: a break that moves nothing must not paint a
    # marker claiming it did (close-review cycle-2, N1 — measured as two extra
    # `<line>` elements over a byte-identical chart).
    assert broken_svg.count("<line") == plain_svg.count("<line"), (
        "an ordinal-axis break must not add a break indicator"
    )


def test_two_same_axis_breaks_fold_annotations_and_marks_to_the_same_position():
    """Two same-axis breaks compose identically for marks and annotations
    (finding #2).

    Pre-fix (measured by the reviewer): the annotation remap took only the
    FIRST same-axis break (`.find`) while marks folded through every one, a
    52px divergence with no warning for `annotate_hline(y=450)` under two
    `BreakAxis(axis="y")` gaps.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [50.0, 450.0, 900.0]})
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    ref = fm.annotate_hline(y=450.0)  # unlabelled: one AnnotationLine, "line" kind
    chart = (
        points
        + ref
        + fm.BreakAxis(axis="y", gap=(150, 400))
        + fm.BreakAxis(axis="y", gap=(500, 850))
    )
    svg = chart.to_svg()

    pairs = [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'<circle cx="(-?[\d.]+)" cy="(-?[\d.]+)"', svg)
    ]
    assert len(pairs) == 3, f"expected all three point marks; got {pairs}"
    pairs.sort(key=lambda p: p[0])
    y450_cy = pairs[1][1]  # x=2.0 is the middle cx on this quantitative x scale

    horiz_line_ys = {
        float(m.group(1))
        for m in re.finditer(r'<line x1="-?[\d.]+" y1="(-?[\d.]+)" x2="-?[\d.]+" y2="\1"', svg)
    }
    assert any(abs(y450_cy - ly) < 1e-6 for ly in horiz_line_ys), (
        f"the annotation must land at the mark's FULLY-FOLDED position {y450_cy} "
        f"(both breaks applied, not just the first); got line ys {horiz_line_ys}"
    )


def test_two_same_axis_breaks_render_identically_to_one_merged_spec_and_hide_the_in_gap_mark():
    """The resurrection repro (#230-4, #230-9): two same-axis `BreakAxis`
    specs must merge into one break result and render byte-identically to a
    single spec declaring the union of gaps — and a mark inside the FIRST
    spec's gap must stay hidden, not resurface at the plot's top edge.

    Pre-fix (measured): each `BreakAxis(axis="y", ...)` built its own
    `BreakResult` against the ORIGINAL domain, and the second remap's
    `data_at` clamped the first remap's `BREAK_HIDDEN` sentinel back into the
    domain — landing the y=200 mark (inside the first gap, (150, 400)) at the
    plot's top pixel, overplotting the y=1000 mark, instead of hiding it.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [200.0, 450.0, 1000.0]})
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")

    two_specs = (
        points + fm.BreakAxis(axis="y", gap=(150, 400)) + fm.BreakAxis(axis="y", gap=(500, 850))
    )
    one_spec = points + fm.BreakAxis(axis="y", gap=[(150, 400), (500, 850)])

    two_svg = two_specs.to_svg()
    one_svg = one_spec.to_svg()
    assert two_svg == one_svg, (
        "two same-axis BreakAxis specs must render byte-identically to one "
        "spec declaring the union of their gaps"
    )

    cys = sorted(
        float(m.group(1)) for m in re.finditer(r'<circle cx="-?[\d.]+" cy="(-?[\d.]+)"', two_svg)
    )
    # The y=200 mark (inside the first gap) must be hidden off-canvas, not
    # resurrected near the plot's top pixel (which the y=1000 mark occupies).
    assert cys[0] < -1000, f"the in-gap mark must be hidden off-canvas; got cys={cys}"
    assert not any(abs(cys[0] - cy) < 1e-6 for cy in cys[1:]), (
        f"the hidden mark must not overplot a visible mark's position: {cys}"
    )


def test_two_same_axis_breaks_with_grid_bands_render_identically_to_one_merged_spec():
    """T1 reviewer's condition (spec §4 #230-4): with two same-axis specs,
    each clipped grid piece could claim structural rect ends on a second
    pass. The same-axis merge makes that second pass unreachable — this pins
    it end to end with a banded grid, whose outermost stripes carry the
    structural (plot-rect) ends a per-spec double-pass could corrupt.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [50.0, 450.0, 950.0]})
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    banded = lambda chart: chart.configure_grid(band_colors=["#f0f0f0", "#e8e8e8"])  # noqa: E731

    two_specs = banded(
        points + fm.BreakAxis(axis="y", gap=(150, 400)) + fm.BreakAxis(axis="y", gap=(500, 850))
    )
    one_spec = banded(points + fm.BreakAxis(axis="y", gap=[(150, 400), (500, 850)]))

    two_svg = two_specs.to_svg()
    one_svg = one_spec.to_svg()
    assert two_svg == one_svg, (
        "two same-axis BreakAxis specs with a banded grid must render "
        "byte-identically to one spec declaring the union of their gaps"
    )
    band_rects = re.findall(
        r'<rect x="[-\d.]+" y="(-?[\d.]+)" width="[\d.]+" height="([\d.]+)"', two_svg
    )
    assert len(band_rects) >= 3, (
        f"test sanity: the banded grid must draw stripes; got {two_svg[:800]}"
    )


def test_break_axis_merge_conflicting_style_warns_once_first_wins():
    """A conflicting `break_style` between merged same-axis specs keeps the
    FIRST declaration's style and warns exactly once naming the axis and
    field (#230-4, spec §6).
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [50.0, 900.0]})
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    two_specs = (
        points
        + fm.BreakAxis(axis="y", gap=(150, 400), break_style="gap")
        + fm.BreakAxis(axis="y", gap=(500, 850), break_style="slash")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two_svg = two_specs.to_svg()
    messages = [str(w.message) for w in caught]
    matches = [m for m in messages if "disagree" in m and "break_style" in m]
    assert len(matches) == 1, f"expected exactly one break_style conflict warning; got {messages}"

    # First-wins: break_style="gap" draws no indicator at all, so the merged
    # render must match a single spec declared with that style.
    one_spec_gap = points + fm.BreakAxis(axis="y", gap=[(150, 400), (500, 850)], break_style="gap")
    assert two_svg == one_spec_gap.to_svg(), "the FIRST spec's break_style must win"


def test_break_axis_merge_conflicting_size_warns_once_first_wins():
    """The `break_size` mirror of the style-conflict pin: first spec wins,
    one warning.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [50.0, 900.0]})
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    two_specs = (
        points
        + fm.BreakAxis(axis="y", gap=(150, 400), break_size=12)
        + fm.BreakAxis(axis="y", gap=(500, 850), break_size=40)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two_svg = two_specs.to_svg()
    messages = [str(w.message) for w in caught]
    matches = [m for m in messages if "disagree" in m and "break_size" in m]
    assert len(matches) == 1, f"expected exactly one break_size conflict warning; got {messages}"

    one_spec = points + fm.BreakAxis(axis="y", gap=[(150, 400), (500, 850)], break_size=12)
    assert two_svg == one_spec.to_svg(), "the FIRST spec's break_size must win"


def test_break_axis_merge_without_conflicting_style_or_size_is_silent():
    """Merging same-axis gaps is the documented semantics of a same-axis
    break and never warns on its own — only a conflicting `break_style`/
    `break_size` does (spec §6).
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [50.0, 900.0]})
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    two_specs = (
        points + fm.BreakAxis(axis="y", gap=(150, 400)) + fm.BreakAxis(axis="y", gap=(500, 850))
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two_specs.to_svg()
    messages = [str(w.message) for w in caught]
    assert not any("disagree" in m for m in messages), (
        f"a non-conflicting same-axis merge must not warn; got {messages}"
    )


def test_x_axis_annotation_inside_a_gap_is_suppressed_with_a_named_warning():
    """The x-axis mirror of the y-axis suppression pin (finding #3): every
    pin so far was y-only, leaving the `find(axis == "x")` wiring unpinned.
    """
    df = pl.DataFrame({"x": [10.0, 20.0, 30.0, 900.0, 50.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    chart = (
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        + fm.BreakAxis(axis="x", gap=(150, 850))
        + fm.annotate_vline(x=500.0, label="inside the break")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()

    assert "inside the break" not in svg, "the suppressed label must not render"
    messages = [str(w.message) for w in caught]
    matches = [m for m in messages if "break" in m.lower() and "x axis" in m and "suppressed" in m]
    assert matches, f"expected a warning naming the annotation's axis; got {messages}"
    assert len(matches) == 1, (
        f"N suppressed instances must roll up into ONE warning; got {messages}"
    )


def _vert_line_xs(svg: str) -> collections.Counter:
    """Multiset of x-values shared by a vertical `<line>`'s two endpoints.

    A `Counter`, not a `set` — see `_horiz_line_ys`'s docstring for why: the
    annotation's correct position can coincide exactly with an existing
    gridline's x, which a set-difference can't distinguish from "no line
    added at all."
    """
    return collections.Counter(
        float(m.group(1)) for m in re.finditer(r'<line x1="(-?[\d.]+)" y1="-?[\d.]+" x2="\1"', svg)
    )


def test_x_axis_surviving_annotation_lands_at_the_marks_compressed_position():
    """The x-axis mirror of the y-axis remap pin (finding #3).

    Same per-position COUNT-delta discrimination as the y-axis pin above
    (quality-review cycle-3): a vertical gridline already sits at the mark's
    compressed x even with the annotation removed, so neither a bare
    membership check nor a set-difference catches a missing annotation here.

    As in the y-axis pin, the reference value sits INSIDE the domain rather
    than on its edge (#226): a domain endpoint lands on the same pixel broken
    or not, which would leave the closing discrimination vacuous.
    """
    df = pl.DataFrame({"x": [0.0, 100.0, 900.0], "y": [1.0, 2.0, 3.0]})
    brk = fm.BreakAxis(axis="x", gap=(150, 850))
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    ref = fm.annotate_vline(x=100.0)  # unlabelled: one AnnotationLine, "line" kind

    broken_with = (points + ref + brk).to_svg()
    broken_without = (points + brk).to_svg()
    counts_with = _vert_line_xs(broken_with)
    counts_without = _vert_line_xs(broken_without)

    cxs = sorted(float(m) for m in re.findall(r'<circle cx="(-?[\d.]+)"', broken_with))
    assert len(cxs) == 3, f"expected all three point marks; got {cxs}"
    x100_cx = cxs[1]  # x=100 is the middle data value on this quantitative x scale

    assert _count_at(counts_with, x100_cx) == _count_at(counts_without, x100_cx) + 1, (
        f"adding the annotation must add exactly one vertical line at the "
        f"mark's compressed position {x100_cx}; with={dict(counts_with)} "
        f"without={dict(counts_without)}"
    )

    # Discriminate against a second pre-fix bug: without the break, the
    # mark's own position is different (the unbroken linear position).
    plain_svg = (points + ref).to_svg()
    plain_cxs = sorted(float(m) for m in re.findall(r'<circle cx="(-?[\d.]+)"', plain_svg))
    plain_x100_cx = plain_cxs[1]
    assert abs(x100_cx - plain_x100_cx) > 1.0, (
        "the break must move the mark (and the annotation with it); "
        f"broken={x100_cx} unbroken={plain_x100_cx}"
    )


# --------------------------------------------------------------------------
# cycle-3 fix: an ineffective break on a continuous axis renders as if
# undeclared (design-review cycle 3, C1)
# --------------------------------------------------------------------------


def test_break_whose_gaps_miss_the_padded_domain_renders_identically_to_unbroken():
    """A `BreakAxis` whose gaps miss the (padded) domain must be a true no-op.

    The cycle-2 N1 fix made ``BreakAxisIneffective`` fire for this case, but
    kept pushing a single-segment "identity" result so the render stayed
    byte-identical to the *pre-batch broken* chart — not to the *unbroken*
    chart the warning's own wording promises ("had no effect"). That
    single-segment map is laid over the plot rect while the real remap
    reverse-maps from the scale's own PADDED pixel range, so it silently
    collapsed the scale's padding: every gridline, tick, tick label, and
    mark shifted by up to the padding width (measured by the reviewer as a
    35-element, up-to-8px diff). This pins the adjudicated fix: drop the
    `break_results` entry when `effective_gaps == 0`, so the continuous
    branch matches the ordinal branch exactly and the SVG is byte-identical
    to the unbroken chart.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    unbroken_svg = chart.to_svg()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        broken_svg = (chart + fm.BreakAxis(axis="y", gap=(500, 900))).to_svg()

    messages = [str(w.message) for w in caught]
    assert any("had no effect" in m and "on the y axis" in m for m in messages), (
        f"expected the ineffective-break warning naming the y axis; got {messages}"
    )
    assert broken_svg == unbroken_svg, (
        "a break whose gaps miss the domain must render exactly the unbroken "
        "chart — the warning says it 'had no effect', so the render must "
        "agree, not merely stay stable relative to the old (also-wrong) "
        "broken behavior"
    )


# --------------------------------------------------------------------------
# GH #226: the segments tile the SCALE's own (padded) pixel range
# --------------------------------------------------------------------------
#
# `apply_break_to_scale` used to receive the raw plot rect while `BreakRemap`
# reverse-maps from `scale.pixel_range()` — the same rect inset by the
# positional-scale padding band (5% of the span, capped at 8px per end). The
# round trip was therefore lossy in exactly that band: every EFFECTIVE break
# silently deleted the axis padding, and since the C1 fix above made the
# INEFFECTIVE case exact, the two sides of the domain boundary disagreed by
# the full 8px.

_PADDED_REF_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 25.0, 15.0, 30.0, 20.0]})
"""The reference frame the #226 measurements were taken on: a (10, 30) y
domain on the default 640x480 canvas, whose plot rect runs y 16.0…420.961
and whose padded scale range runs 24.0…412.961."""


def _grid_ys(svg: str) -> list[float]:
    """Every horizontal gridline's y, in emission (tick) order."""
    return [
        float(m.group(1))
        for m in re.finditer(
            r'<line x1="[-\d.]+" y1="([-\d.]+)" x2="[-\d.]+" y2="\1" stroke="#d6d3d1"', svg
        )
    ]


def _y_axis_domain_line_span(svg: str) -> tuple[float, float]:
    """``(top, bottom)`` of the y-axis domain line — the plot rect's full height.

    The domain line is the axis's longest near-vertical stroke: tick dashes on
    a Left axis are horizontal, and the x-axis's own tick dashes are only
    ``tick_size`` tall. It is structural chrome, so a break must leave it
    spanning the whole rect (#226) — which also makes it the landmark the
    padded span is measured against.
    """
    spans = [
        (min(float(y1), float(y2)), max(float(y1), float(y2)))
        for x1, y1, x2, y2 in re.findall(
            r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"'
            r'[^>]*stroke="#6b7280"',
            svg,
        )
        if abs(float(x2) - float(x1)) <= 1.0
    ]
    assert spans, f"expected a y-axis domain line; got: {svg[:500]}"
    return max(spans, key=lambda s: s[1] - s[0])


def _cys(svg: str) -> list[float]:
    return [float(v) for v in re.findall(r'<circle cx="[-\d.]+" cy="([-\d.]+)"', svg)]


def test_broken_axis_ticks_tile_the_scales_padded_pixel_range():
    """A broken axis's ticks span the padded range, exactly as unbroken.

    Measured pre-fix on this chart: unbroken y ticks ran 24.0…412.961 (the
    scale's padded pixel range, 8px inside the rect at each end) while the
    same chart under ANY effective break ran 16.0…420.961 (the raw plot
    rect). Every tick, gridline and mark on a broken axis was displaced by up
    to the padding width, and the break's own indicator with them.
    """
    chart = fm.Chart(_PADDED_REF_DF).mark_point().encode(x="x:Q", y="y:Q")
    unbroken_svg = chart.to_svg()
    broken_svg = (chart + fm.BreakAxis(axis="y", gap=(18, 22))).to_svg()

    unbroken = _grid_ys(unbroken_svg)
    visible = [y for y in _grid_ys(broken_svg) if y > -1000]
    assert unbroken and visible, "expected horizontal gridlines in both renders"

    # The reference numbers from the finding, so a regression that moves the
    # padded range itself is caught here rather than silently agreeing.
    assert min(unbroken) == pytest.approx(24.0, abs=1e-3)
    assert max(unbroken) == pytest.approx(412.961, abs=1e-3)
    assert (min(visible), max(visible)) == pytest.approx(
        (min(unbroken), max(unbroken)), abs=1e-3
    ), (
        f"a break must tile the same padded span the unbroken scale uses; "
        f"broken={min(visible)}…{max(visible)} unbroken={min(unbroken)}…{max(unbroken)}"
    )

    # ...and that span is strictly INSIDE the plot rect, which is what makes
    # the assertion above discriminating: the pre-fix segments tiled the rect,
    # so they reached the domain line's own ends.
    top, bottom = _y_axis_domain_line_span(broken_svg)
    assert top < min(visible) and max(visible) < bottom, (
        f"the padded span must sit inside the plot rect ({top}…{bottom}); "
        f"got {min(visible)}…{max(visible)}"
    )
    assert (top, bottom) == pytest.approx(_y_axis_domain_line_span(unbroken_svg), abs=1e-3), (
        "the axis domain line is structural chrome: a break must not shrink it onto the padded span"
    )


def _grid_lines(svg: str) -> list[tuple[float, float, float, float]]:
    """Every gridline as ``(x1, y1, x2, y2)``, in emission order."""
    return [
        (float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in re.findall(
            r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" '
            r'stroke="#d6d3d1"',
            svg,
        )
    ]


def _vertical_gridlines(svg: str) -> list[tuple[float, float, float, float]]:
    """The gridlines placed at x tick values — vertical, spanning the rect's height."""
    return [line for line in _grid_lines(svg) if line[0] == line[2]]


def _horizontal_gridlines(svg: str) -> list[tuple[float, float, float, float]]:
    """The gridlines placed at y tick values — horizontal, spanning the rect's width."""
    return [line for line in _grid_lines(svg) if line[1] == line[3]]


def test_a_break_leaves_the_cross_axis_gridlines_spanning_the_whole_plot_rect():
    """A break must not touch the OTHER axis's gridlines (#226, cycle 2).

    A vertical gridline is placed at an x tick value, but its ``y1``/``y2``
    are the plot rect's corners — the same structural provenance that exempts
    the axis domain line. Threading the scale's padded range into the break
    without that exemption pulled every vertical gridline's ends onto the
    padded edges, so on a y-broken chart they stopped 8px short of the x-axis
    line they used to meet, while the axis line itself and an
    ``annotate_vline`` beside them still spanned the rect. Pinned in both
    directions, each with the sanity half (the gridlines the break DOES own
    move), so neither can pass on a chart the break never reached.
    """
    chart = fm.Chart(_PADDED_REF_DF).mark_point().encode(x="x:Q", y="y:Q")
    unbroken = chart.to_svg()
    y_broken = (chart + fm.BreakAxis(axis="y", gap=(18, 22))).to_svg()
    x_broken = (chart + fm.BreakAxis(axis="x", gap=(2.4, 3.6))).to_svg()

    # The reference frame's rect, from the structural chrome that already
    # spans it: 16.0…420.961, 8px outside the padded span the ticks tile.
    rect_top, rect_bottom = _y_axis_domain_line_span(unbroken)
    assert (rect_top, rect_bottom) == pytest.approx((16.0, 420.961), abs=1e-3)

    assert _vertical_gridlines(unbroken), "test sanity: the x ticks draw gridlines"
    for x1, y1, _x2, y2 in _vertical_gridlines(unbroken):
        assert (y1, y2) == pytest.approx((rect_top, rect_bottom), abs=1e-3), (
            f"test sanity: an unbroken vertical gridline at x={x1} spans the rect"
        )

    assert _vertical_gridlines(y_broken) == _vertical_gridlines(unbroken), (
        "a y break must leave the vertical (x-tick) gridlines exactly where the "
        "unbroken chart drew them — their ends are the plot rect's corners, not "
        "tick values"
    )
    assert _horizontal_gridlines(y_broken) != _horizontal_gridlines(unbroken), (
        "test sanity: a y break DOES move the gridlines it owns"
    )

    assert _horizontal_gridlines(x_broken) == _horizontal_gridlines(unbroken), (
        "an x break must leave the horizontal (y-tick) gridlines spanning the whole plot width"
    )
    assert _vertical_gridlines(x_broken) != _vertical_gridlines(unbroken), (
        "test sanity: an x break DOES move the gridlines it owns"
    )


_BAND_FILL = "#f0f0f0"


def _band_spans(svg: str) -> list[tuple[float, float]]:
    """``(top, bottom)`` for every banded row fill, in paint order."""
    return [(y, y + h) for _x, y, _w, h in _rects(svg, _BAND_FILL)]


def _banded_chart():
    """The reference frame striped by a SINGLE band fill.

    ``fm.Axis(values=[])`` leaves the y axis with no boundary strictly inside
    the plot rect, so the banded grid emits one stripe spanning it — the shape
    where one node carries BOTH of the band family's structural ends.
    """
    return (
        fm.Chart(_PADDED_REF_DF)
        .mark_point()
        .encode(x="x:Q", y=fm.Y("y:Q", axis=fm.Axis(values=[])))
        .configure(grid=fm.GridConfig(band_colors=[_BAND_FILL]))
    )


def test_a_break_keeps_a_single_bands_edges_on_the_plot_rect():
    """A lone band still stripes the whole rect under a break (#226, cycle 3).

    A band fill's outermost boundary is the plot rect's own edge, structural
    for the same reason the axis domain line's ends are — so a break puts it
    back on the rect after clipping the tick-valued boundary beside it. When
    one band carries BOTH ends, restoring them from a copy of the piece
    geometry taken before either write made the second write hand back the
    padded edge the first had just replaced: this chart rendered
    ``24.0…420.961`` and the plot's top 8px padding strip lost its stripe.
    Both orientations, since a y break owns a band's extent and an x break
    owns none of it.
    """
    chart = _banded_chart()
    unbroken_svg = chart.to_svg()
    rect_top, rect_bottom = _y_axis_domain_line_span(unbroken_svg)
    assert (rect_top, rect_bottom) == pytest.approx((16.0, 420.961), abs=1e-3)
    assert _band_spans(unbroken_svg) == [pytest.approx((rect_top, rect_bottom), abs=1e-2)], (
        "test sanity: an empty `values=` stripes the rect with exactly one band"
    )

    # A gap strictly inside the domain cuts the band in two: each structural
    # end lands on its own piece, and both still reach the rect.
    two = _band_spans((chart + fm.BreakAxis(axis="y", gap=(18, 22))).to_svg())
    assert len(two) == 2, f"test sanity: a mid-domain gap cuts the band; got {two}"
    assert (min(t for t, _ in two), max(b for _, b in two)) == pytest.approx(
        (rect_top, rect_bottom), abs=1e-2
    )

    # ...and a gap reaching the domain's own low end leaves ONE piece carrying
    # both ends, which is the case the pre-fix restore could not serve.
    one = _band_spans((chart + fm.BreakAxis(axis="y", gap=(10, 22))).to_svg())
    assert len(one) == 1, f"test sanity: this gap must leave a single piece; got {one}"
    assert one[0] == pytest.approx((rect_top, rect_bottom), abs=1e-2), (
        "a single band's two boundaries are BOTH the plot rect's edges, so a "
        "break leaves it spanning the rect"
    )

    # The other orientation: an x break owns no part of a band.
    x_broken_svg = (chart + fm.BreakAxis(axis="x", gap=(2.4, 3.6))).to_svg()
    assert _rects(x_broken_svg, _BAND_FILL) == _rects(unbroken_svg, _BAND_FILL), (
        "an x break must leave the band exactly where the unbroken chart drew it"
    )


def test_an_epsilon_gap_does_not_step_the_chart_across_the_domain_boundary():
    """A gap that barely reaches into the domain renders ~the unbroken chart.

    The two sides of the domain boundary must agree. A gap that MISSES the
    (10, 30) domain is dropped outright (C1) and renders the unbroken chart
    byte for byte; a gap of 0.002 data units INSIDE it, reserving no
    indicator space (``break_size=0``), removes almost nothing and so must
    render within float noise of the same chart. Pre-fix the effective side
    tiled the raw rect instead: the marks stepped 412.961/24.0 →
    420.961/16.0, a hard 8px jump the moment the gap crossed the boundary.
    """
    chart = fm.Chart(_PADDED_REF_DF).mark_point().encode(x="x:Q", y="y:Q")
    unbroken = _cys(chart.to_svg())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outside_svg = (chart + fm.BreakAxis(axis="y", gap=(500, 900), break_size=0)).to_svg()
    assert any("had no effect" in str(w.message) for w in caught), (
        "test sanity: the outside-the-domain gap is the ineffective (C1) side"
    )
    assert _cys(outside_svg) == unbroken, "test sanity: the ineffective side is unbroken"

    inside = _cys((chart + fm.BreakAxis(axis="y", gap=(19.999, 20.001), break_size=0)).to_svg())
    assert len(inside) == len(unbroken)
    # y=20.0 is genuinely inside the declared gap, so that mark hides — the
    # one honest difference between the two sides.
    hidden = [i for i, cy in enumerate(inside) if cy < -1000]
    assert hidden == [4], f"only the in-gap y=20 mark may hide; got {inside}"
    for i, (got, want) in enumerate(zip(inside, unbroken)):
        if i in hidden:
            continue
        assert got == pytest.approx(want, abs=0.05), (
            f"mark {i} stepped {want} -> {got}; an epsilon gap must not move the "
            f"chart off the unbroken positions (broken={inside} unbroken={unbroken})"
        )


def test_out_of_domain_annotation_clamps_to_the_correct_edge_under_a_break():
    """quality-review cycle-3 (S1 follow-up to F-L08-07's `None`-arm comment
    fix): a continuous y axis's out-of-domain fallback formula was un-flipped
    relative to the real (SVG-down, domain-max-at-top) scale. Invisible
    pre-fix (an out-of-domain value's naive linear position was always far
    off-canvas either way), but a `BreakAxis`'s clamp-to-domain-edge makes it
    VISIBLE — and on the wrong side: a value above the domain max used to
    clamp to the plot BOTTOM instead of the TOP. Fixed in `resolve_y`'s
    fallback; pinned here on both sides via the same delta-vs-no-annotation
    shape as the position pins above.

    The edges the clamp lands on are the PADDED ones (#226): the break's
    segments tile the scale's own pixel range, so a value past the domain
    clamps to the domain's end, which the unbroken scale puts 8px inside the
    plot rect. Pre-#226 the segments tiled the raw rect and both clamps
    landed on the rect's own edges instead — i.e. 8px outside where the
    scale places the extreme data value.
    """
    df = pl.DataFrame(
        {"category": ["A", "B", "C", "D", "E"], "value": [72.0, 68.0, 75.0, 900.0, 81.0]}
    )
    chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q") + fm.BreakAxis(
        axis="y", gap=(150, 850)
    )
    without = chart.to_svg()
    counts_without = _horiz_line_ys(without)

    def hline_y(value: float) -> float:
        with_ref = (chart + fm.annotate_hline(y=value)).to_svg()
        counts_with = _horiz_line_ys(with_ref)
        # The one y whose count grew by exactly one is the annotation's own
        # line — a set-difference would miss it if it coincides with an
        # existing gridline's y (see `_horiz_line_ys`'s docstring).
        added = [
            y for y in counts_with if _count_at(counts_with, y) == _count_at(counts_without, y) + 1
        ]
        assert len(added) == 1, (
            f"expected exactly one horizontal line whose count grew by one "
            f"for y={value}; with={dict(counts_with)} without={dict(counts_without)}"
        )
        return added[0]

    above_max = hline_y(5000.0)  # far above the domain max (~900-ish, padded)
    below_min = hline_y(-500.0)  # far below the domain min (0)

    assert above_max < below_min, (
        "a value ABOVE the domain max must clamp to a SMALLER pixel (nearer "
        f"the plot top) than a value BELOW the domain min; got "
        f"above_max={above_max} below_min={below_min}"
    )

    # The clamp lands where the scale itself places the extreme data value:
    # the tallest bar's top is the domain max (900) and every bar's base is
    # the domain min (0), both on the PADDED range.
    rects = _rects(without)
    domain_max_px = min(r[1] for r in rects)
    domain_min_px = max(r[1] + r[3] for r in rects)
    assert above_max == pytest.approx(domain_max_px, abs=1e-3), (
        f"above_max={above_max} must clamp to the domain max's own pixel "
        f"({domain_max_px}), not past it"
    )
    assert below_min == pytest.approx(domain_min_px, abs=1e-3), (
        f"below_min={below_min} must clamp to the domain min's own pixel "
        f"({domain_min_px}), not past it"
    )

    # Those are the PADDED edges, strictly inside the plot rect — the
    # discriminating half, since the pre-#226 segments tiled the rect and
    # clamped onto its own edges.
    rect_top, rect_bottom = _y_axis_domain_line_span(without)
    assert rect_top < above_max and below_min < rect_bottom, (
        f"the clamp edges ({above_max}, {below_min}) must sit inside the plot "
        f"rect ({rect_top}, {rect_bottom}) by the scale's padding"
    )


# --------------------------------------------------------------------------
# GH #231 / D2: rigid tick-label offsets under a break
# --------------------------------------------------------------------------
#
# `remap_node`'s `Text` arm used to reverse-map a label's own drawn baseline
# (`tick.position + font/3` on a vertical axis, or a `label_flush` edge's
# `tick.position` exactly) as if THAT were the tick's data position, so each
# label's offset from its own dash compressed by whatever ratio that
# DIFFERENT pixel implied — a different amount per tick, a flush edge label
# reading an offset of exactly zero as if it had none, and an offset that
# could push a label's DRAWN position into a gap its own tick never crossed.
# `marks::axis::AxisTickNode` now carries each node's own tick anchor to the
# remap, and `BreakRemap::remap_axis_tick_node` translates rigidly on it
# (spec §4 "Tick labels (D2)"), so a label rides its own dash exactly instead
# of being re-derived from wherever it happens to be drawn.


def _numeric_labels(labels):
    """Filter a tick group's labels to the numeric ones.

    `axis_tick_dash_label_pairs` groups by adjacency: nothing separates the
    LAST tick's label from a following axis title `<text>` (no `<line>`
    intervenes), so the title can land in that group too. Filtering to
    float-parseable label text drops it without needing the chart to
    suppress its title.
    """
    numeric = []
    for label in labels:
        try:
            float(label[2])
        except ValueError:
            continue
        numeric.append(label)
    return numeric


@pytest.mark.parametrize("axis, gap", [("y", (18, 22)), ("x", (2.4, 3.6))])
def test_tick_label_offset_from_its_own_dash_survives_a_break(axis, gap):
    """A label's drawn offset from its own tick's dash is preserved verbatim
    under a break, for every visible tick — never compressed by the segment
    ratio, and a label never hides while its own dash stays visible.

    This measures the offset from each tick's PAIRED dash (read straight off
    the rendered SVG, never a hardcoded formula), so it holds regardless of
    which axis carries the break — a vertical axis's default label offset
    (`font_size / 3`) is where the old bug was visible; a horizontal axis's
    default offset happens to be zero, which this still confirms survives a
    break intact rather than assuming it trivially does.
    """
    chart = fm.Chart(_PADDED_REF_DF).mark_point().encode(x="x:Q", y="y:Q")
    unbroken_svg = chart.to_svg()
    broken_svg = (chart + fm.BreakAxis(axis=axis, gap=gap)).to_svg()

    unbroken_ticks = axis_tick_dash_label_pairs(unbroken_svg, axis=axis)
    broken_ticks = axis_tick_dash_label_pairs(broken_svg, axis=axis)
    assert unbroken_ticks, f"test sanity: expected {axis}-axis tick dashes"
    assert len(broken_ticks) == len(unbroken_ticks), (
        "a break must not change which ticks are drawn, only where they land"
    )

    saw_a_hidden_tick = False
    for unbroken_tick, broken_tick in zip(unbroken_ticks, broken_ticks):
        unbroken_numeric = _numeric_labels(unbroken_tick.labels)
        broken_numeric = _numeric_labels(broken_tick.labels)
        assert len(broken_numeric) == len(unbroken_numeric), (
            f"a break must not cull or add label lines for a tick; "
            f"unbroken={unbroken_tick.labels} broken={broken_tick.labels}"
        )
        for unbroken_label, broken_label in zip(unbroken_numeric, broken_numeric):
            unbroken_offset = label_along(unbroken_label, axis=axis) - tick_along(
                unbroken_tick.dash, axis=axis
            )
            broken_offset = label_along(broken_label, axis=axis) - tick_along(
                broken_tick.dash, axis=axis
            )
            assert broken_offset == pytest.approx(unbroken_offset, abs=1e-2), (
                f"tick {unbroken_label[2]!r}: drawn offset from its own dash must "
                f"survive a break verbatim; unbroken={unbroken_offset} "
                f"broken={broken_offset}"
            )

        # D2's hide rule: a label hides iff its OWN tick's dash hides — never
        # independently, and never at a different threshold than the dash.
        dash_hidden = tick_along(broken_tick.dash, axis=axis) < -1000
        for broken_label in broken_numeric:
            label_hidden = label_along(broken_label, axis=axis) < -1000
            assert label_hidden == dash_hidden, (
                f"tick {broken_label[2]!r}: a label must hide exactly when its "
                f"own tick's dash does, never independently (dash_hidden="
                f"{dash_hidden}, label_hidden={label_hidden})"
            )
        saw_a_hidden_tick = saw_a_hidden_tick or dash_hidden

    assert saw_a_hidden_tick, f"test sanity: gap={gap} must hide at least one {axis} tick"


def test_flush_edge_tick_label_offset_survives_a_break():
    """`label_flush`'s edge ticks carry a DIFFERENT drawn offset than every
    interior tick (0px at one edge, a full `font_size` at the other, per
    `build_axis`'s flush formula) — and that offset survives a break exactly
    like every interior tick's, closing the case the finding named
    specifically: a flush edge label used to read its (non-representative)
    zero offset as "no compression happened", while every interior label's
    offset silently shrank around it.
    """
    chart = (
        fm.Chart(_PADDED_REF_DF)
        .mark_point()
        .encode(x="x:Q", y=fm.Y("y:Q", axis=fm.Axis(label_flush=True)))
    )
    unbroken_svg = chart.to_svg()
    broken_svg = (chart + fm.BreakAxis(axis="y", gap=(18, 22))).to_svg()

    unbroken_ticks = axis_tick_dash_label_pairs(unbroken_svg, axis="y")
    broken_ticks = axis_tick_dash_label_pairs(broken_svg, axis="y")
    assert len(unbroken_ticks) == len(broken_ticks) and unbroken_ticks

    first_offset = label_along(_numeric_labels(unbroken_ticks[0].labels)[0], axis="y") - tick_along(
        unbroken_ticks[0].dash, axis="y"
    )
    last_offset = label_along(_numeric_labels(unbroken_ticks[-1].labels)[0], axis="y") - tick_along(
        unbroken_ticks[-1].dash, axis="y"
    )
    assert first_offset != pytest.approx(last_offset, abs=1e-2), (
        "test sanity: label_flush must give the two edge ticks DIFFERENT "
        f"offsets (0 vs font_size); got first={first_offset} last={last_offset}"
    )

    for unbroken_tick, broken_tick in zip(
        (unbroken_ticks[0], unbroken_ticks[-1]), (broken_ticks[0], broken_ticks[-1])
    ):
        # The offset assertion below would pass vacuously if this gap hid the
        # edge tick (rigid translation preserves an offset at BREAK_HIDDEN
        # too), so confirm the tick this test is actually about is visible.
        dash_hidden = tick_along(broken_tick.dash, axis="y") < -1000
        assert not dash_hidden, (
            f"test sanity: gap=(18, 22) must not hide flush edge tick "
            f"{_numeric_labels(broken_tick.labels)[0][2]!r}"
        )
        unbroken_label = _numeric_labels(unbroken_tick.labels)[0]
        broken_label = _numeric_labels(broken_tick.labels)[0]
        unbroken_offset = label_along(unbroken_label, axis="y") - tick_along(
            unbroken_tick.dash, axis="y"
        )
        broken_offset = label_along(broken_label, axis="y") - tick_along(broken_tick.dash, axis="y")
        assert broken_offset == pytest.approx(unbroken_offset, abs=1e-2), (
            f"flush edge tick {unbroken_label[2]!r}: its distinctive offset "
            f"must survive a break just like an interior tick's; "
            f"unbroken={unbroken_offset} broken={broken_offset}"
        )


# --------------------------------------------------------------------------
# #230-8: polar + BreakAxis refuses (D3: REFUSE MALFORMED)
# --------------------------------------------------------------------------


def _polar_pie_chart(df: pl.DataFrame) -> fm.Chart:
    return fm.Chart(df).mark_arc().encode(x="val:Q", color="cat:N").coord(fm.CoordPolar(theta="x"))


def _polar_pie_df() -> pl.DataFrame:
    return pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})


def test_polar_coord_with_break_axis_refuses_with_a_named_message():
    """#230-8: under `CoordPolar`, arc marks build their own polar geometry
    (`marks/arc.rs::build` — wedge centers and radii, not positions on the
    Cartesian scale), so the break fold's reverse map through that scale has
    no meaning for them; the pre-fix pipeline applied it anyway rather than
    suppressing it as the (since-removed) comment near
    `BreakPlacement::for_kind`'s `Arc` arm claimed. (This fixture's own gap
    misses the y domain — the chart has no y encoding — so pre-fix it took
    the `BreakAxisIneffective` branch; a domain-overlapping gap was the
    corrupting shape.) The combination now refuses with a typed
    `ValueError` (Rust `RenderError::BreakAxisIncompatibleCoord`), before
    any geometry is built."""
    chart = _polar_pie_chart(_polar_pie_df()) + fm.BreakAxis(axis="y", gap=(150, 850))
    with pytest.raises(ValueError, match=r"BreakAxis"):
        chart.to_svg()


def test_polar_coord_with_break_axis_message_names_the_combination():
    chart = _polar_pie_chart(_polar_pie_df()) + fm.BreakAxis(axis="y", gap=(150, 850))
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    message = str(exc_info.value)
    assert "BreakAxis" in message, message
    assert "y" in message, message
    assert "Polar" in message, message


def test_polar_coord_with_break_axis_on_x_names_that_axis():
    chart = _polar_pie_chart(_polar_pie_df()) + fm.BreakAxis(axis="x", gap=(5, 15))
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    # The literal `axis=` in every message contains an "x", so a bare
    # `"x" in message` check passes even when the gate reported `axis="y"`.
    # Pin the full token so this is the one Python assertion that actually
    # discriminates the x-axis case.
    assert 'axis="x"' in str(exc_info.value)


def test_polar_coord_without_break_axis_still_renders():
    """The refusal is scoped to the combination, not to polar coords alone —
    a polar chart with no `BreakAxis` is untouched."""
    svg = _polar_pie_chart(_polar_pie_df()).to_svg()
    assert "<svg" in svg


def test_cartesian_break_axis_is_unaffected_by_the_polar_refusal():
    plain_chart = (
        fm.Chart(pl.DataFrame({"c": ["A", "B"], "v": [1.0, 900.0]}))
        .mark_bar()
        .encode(x="c:N", y="v:Q")
    )
    chart = plain_chart + fm.BreakAxis(axis="y", gap=(150, 850))
    plain_svg = plain_chart.to_svg()
    svg = chart.to_svg()
    assert "<svg" in svg
    # A silently dropped break would render byte-identically to the
    # unbroken chart; the break indicator's `<line>` elements are the
    # discriminating signal that the break actually happened.
    assert svg.count("<line") > plain_svg.count("<line"), (
        "the Cartesian break must still draw its indicator, not be silently dropped"
    )


def test_polar_break_axis_leaf_inside_a_composite_also_refuses():
    """A composite leaf combining polar coords with a `BreakAxis` refuses
    per-leaf, exactly like every other per-leaf render error: composite
    rendering funnels every leaf through the same `build_scene` call the
    standalone path uses, so the refusal is not something a composite can
    route around."""
    ok_chart = (
        fm.Chart(pl.DataFrame({"c": ["A", "B"], "v": [1.0, 2.0]}))
        .mark_bar()
        .encode(x="c:N", y="v:Q")
    )
    bad_leaf = _polar_pie_chart(_polar_pie_df()) + fm.BreakAxis(axis="y", gap=(150, 850))
    with pytest.raises(ValueError, match=r"BreakAxis"):
        fm.hconcat(ok_chart, bad_leaf).to_svg()


# --------------------------------------------------------------------------
# #230-8 extended (adjudicated 2026-09-12): Geo + BreakAxis refuses too, on
# the same root cause as Polar — a coordinate system whose projection runs
# before the break fold refuses rather than silently corrupting.
# --------------------------------------------------------------------------

# A unit-square GeoJSON polygon (mirrors tests/test_coord_guards.py's
# `_BARE_POLYGON`) — enough for `mark_geoshape` to project real geometry.
_GEO_UNIT_SQUARE = {
    "type": "Polygon",
    "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
}


def _geoshape_chart() -> fm.Chart:
    return (
        fm.Chart(_GEO_UNIT_SQUARE).mark_geoshape().coord(fm.CoordGeo(projection="equirectangular"))
    )


def test_geo_coord_with_break_axis_refuses_with_a_named_message():
    """Under `CoordGeo`, `mark_geoshape` places every vertex by
    bbox-fitting its projected lon/lat into the plot rect
    (`crates/ferrum-core/src/render/marks/geoshape.rs`'s scaling pass),
    independent of the resolved x/y scales. `BreakAxis`'s reverse-map runs
    through the Cartesian scale it replaces, so it has no scale-relative
    position left to correct for geo geometry — the break contract cannot
    be honored for `Geo`, hence this combination refuses with a typed
    `ValueError` (Rust `RenderError::BreakAxisIncompatibleCoord`), before
    any geometry is built — extending the #230-8 gate from `Polar`-only to
    also cover `Geo`.

    (Note: a plain-render `<path>` count dropping to zero under an
    overlapping `BreakAxis` is not itself Geo-specific evidence of
    corruption — it is `BreakPlacement::ClipExtent`'s documented,
    coord-agnostic behavior for any polygon/area batch wholly inside the
    gap, pinned coord-agnostically by
    `break_axis.rs::polygon_wholly_inside_a_gap_keeps_its_node_and_loses_its_rings`
    and reproducible with a plain Cartesian `mark_area` under the same gap
    shape. The assertion below is a sanity check that the plain geo chart
    draws real geometry before checking that the combined chart refuses
    instead of silently rendering it away.)"""
    plain_svg = _geoshape_chart().to_svg()
    assert plain_svg.count("<path") == 1, "sanity: the plain geo chart draws one path"

    chart = _geoshape_chart() + fm.BreakAxis(axis="y", gap=(-2.0, 2.0))
    with pytest.raises(ValueError, match=r"BreakAxis"):
        chart.to_svg()


def test_geo_coord_with_break_axis_message_names_the_combination():
    chart = _geoshape_chart() + fm.BreakAxis(axis="y", gap=(-2.0, 2.0))
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    message = str(exc_info.value)
    assert "BreakAxis" in message, message
    assert "Geo" in message, message


def test_geo_coord_without_break_axis_still_renders():
    """The refusal is scoped to the combination, not to Geo coords alone —
    a Geo chart with no `BreakAxis` is untouched."""
    svg = _geoshape_chart().to_svg()
    assert svg.count("<path") == 1
