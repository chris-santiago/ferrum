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
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann


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
    matches = [
        m for m in messages if "break" in m.lower() and "y=500" in m and "(150, 850)" in m
    ]
    assert matches, (
        f"expected a warning naming the annotation's axis/value/gap; got {messages}"
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
    """A `Span` whose range lies inside a gap suppresses (spec §4 H5's rider)."""
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
    assert any("'span'" in m and "break" in m.lower() for m in messages), (
        f"expected a span-kind suppression warning; got {messages}"
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
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [100.0, 900.0]})
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
    assert len(pairs) == 2, f"expected both point marks; got {pairs}"
    # x=1.0 has the smaller cx on this quantitative x scale.
    y100_cy = min(pairs, key=lambda p: p[0])[1]

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
    plain_y100_cy = min(plain_pairs, key=lambda p: p[0])[1]
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

    assert not any("break" in str(w.message).lower() for w in caught), (
        f"an ordinal-axis BreakAxis must not suppress annotations; got "
        f"{[str(w.message) for w in caught]}"
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
        for m in re.finditer(
            r'<line x1="-?[\d.]+" y1="(-?[\d.]+)" x2="-?[\d.]+" y2="\1"', svg
        )
    }
    assert any(abs(y450_cy - ly) < 1e-6 for ly in horiz_line_ys), (
        f"the annotation must land at the mark's FULLY-FOLDED position {y450_cy} "
        f"(both breaks applied, not just the first); got line ys {horiz_line_ys}"
    )


def test_x_axis_annotation_inside_a_gap_is_suppressed_with_a_named_warning():
    """The x-axis mirror of the y-axis suppression pin (finding #3): every
    pin so far was y-only, leaving the `find(axis == "x")` wiring unpinned.
    """
    df = pl.DataFrame(
        {"x": [10.0, 20.0, 30.0, 900.0, 50.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
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
    matches = [
        m for m in messages if "break" in m.lower() and "x=500" in m and "(150, 850)" in m
    ]
    assert matches, (
        f"expected a warning naming the annotation's axis/value/gap; got {messages}"
    )


def _vert_line_xs(svg: str) -> collections.Counter:
    """Multiset of x-values shared by a vertical `<line>`'s two endpoints.

    A `Counter`, not a `set` — see `_horiz_line_ys`'s docstring for why: the
    annotation's correct position can coincide exactly with an existing
    gridline's x, which a set-difference can't distinguish from "no line
    added at all."
    """
    return collections.Counter(
        float(m.group(1))
        for m in re.finditer(r'<line x1="(-?[\d.]+)" y1="-?[\d.]+" x2="\1"', svg)
    )


def test_x_axis_surviving_annotation_lands_at_the_marks_compressed_position():
    """The x-axis mirror of the y-axis remap pin (finding #3).

    Same per-position COUNT-delta discrimination as the y-axis pin above
    (quality-review cycle-3): a vertical gridline already sits at the mark's
    compressed x even with the annotation removed, so neither a bare
    membership check nor a set-difference catches a missing annotation here.
    """
    df = pl.DataFrame({"x": [100.0, 900.0], "y": [1.0, 2.0]})
    brk = fm.BreakAxis(axis="x", gap=(150, 850))
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    ref = fm.annotate_vline(x=100.0)  # unlabelled: one AnnotationLine, "line" kind

    broken_with = (points + ref + brk).to_svg()
    broken_without = (points + brk).to_svg()
    counts_with = _vert_line_xs(broken_with)
    counts_without = _vert_line_xs(broken_without)

    cxs = [float(m) for m in re.findall(r'<circle cx="(-?[\d.]+)"', broken_with)]
    assert len(cxs) == 2, f"expected both point marks; got {cxs}"
    x100_cx = min(cxs)  # x=100 is the smaller data value on this quantitative x scale

    assert _count_at(counts_with, x100_cx) == _count_at(counts_without, x100_cx) + 1, (
        f"adding the annotation must add exactly one vertical line at the "
        f"mark's compressed position {x100_cx}; with={dict(counts_with)} "
        f"without={dict(counts_without)}"
    )

    # Discriminate against a second pre-fix bug: without the break, the
    # mark's own position is different (the unbroken linear position).
    plain_svg = (points + ref).to_svg()
    plain_cxs = [float(m) for m in re.findall(r'<circle cx="(-?[\d.]+)"', plain_svg)]
    plain_x100_cx = min(plain_cxs)
    assert abs(x100_cx - plain_x100_cx) > 1.0, (
        "the break must move the mark (and the annotation with it); "
        f"broken={x100_cx} unbroken={plain_x100_cx}"
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
            y
            for y in counts_with
            if _count_at(counts_with, y) == _count_at(counts_without, y) + 1
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

    rects = _rects(without)
    plot_top = min(r[1] for r in rects)
    plot_bottom = max(r[1] + r[3] for r in rects)
    assert plot_top - 5.0 <= above_max <= plot_top + 5.0, (
        f"above_max={above_max} should clamp to the plot's top edge "
        f"(plot_top={plot_top}, plot_bottom={plot_bottom})"
    )
    assert plot_bottom - 5.0 <= below_min <= plot_bottom + 5.0, (
        f"below_min={below_min} should clamp to the plot's bottom edge "
        f"(plot_top={plot_top}, plot_bottom={plot_bottom})"
    )
