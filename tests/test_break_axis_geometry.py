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
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm


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

    Spec §4 H4 gives line vertices the hide/compress rule. Before the Polyline
    arm existed, a line chart under a break rendered byte-identically to the
    unbroken chart — i.e. against ticks it no longer agreed with.
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
    ys = [float(p.split(",")[1]) for p in broken[0].split()]
    assert sum(1 for y in ys if y < -1000) == 1, (
        f"the in-gap vertex is hidden, the others placed; got {ys}"
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
