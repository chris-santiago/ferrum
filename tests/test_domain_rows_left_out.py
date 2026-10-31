"""Rows an explicit positional scale domain leaves out do not draw, and warn
once per channel and field, naming the count and a remedy, the same way on
every door: a plain scale ``domain=``, ``xlim``/``ylim``, a bound category
domain, a data-space ``Jitter``, a facet and every composition route (``+``,
``Chart.layer``, ``fm.LayerChart`` static and ``.interactive()``,
``fm.hconcat``). ``clamp=True``, no explicit domain, and a stacked bar whose
implicit base sits outside the domain all draw every row with no warning.
A ``LogScale``'s own off-side warning and this one never double-count the
same row, and a null/NaN/inf positional value is not counted by either.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum import Chart
from ferrum.coord import CoordCartesian, CoordFlip


def _warnings_of(render: Callable[[], Any]) -> tuple[list[str], Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render()
    return [str(w.message) for w in caught], result


def _scene(chart: Any) -> dict:
    return json.loads(chart.interactive().scene_json)


def _drawn_node_count(scene: dict) -> int:
    return sum(len(batch["nodes"]) for panel in scene["panels"] for batch in panel["marks"])


def _line_vertex_count(scene: dict) -> int:
    return sum(
        len(node["points"])
        for panel in scene["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "line"
        for node in batch["nodes"]
    )


def _area_vertex_count(scene: dict) -> int:
    """Each ``area`` node's command list is the top edge's vertices plus a
    fixed three-command baseline close (two corner commands and a close); the
    row count is the command count less that fixed overhead."""
    return sum(
        len(node["commands"]) - 3
        for panel in scene["panels"]
        for batch in panel["marks"]
        if batch["kind"] == "area"
        for node in batch["nodes"]
    )


def _domain_warning(channel: str, field: str, rows: int, *, categorical: bool = False) -> str:
    if categorical:
        return (
            f"{channel} scale: {rows} row(s) of '{field}' name a category the scale's "
            "explicit domain leaves out, so they are not drawn; add it to the domain, or "
            "drop domain= to draw every category"
        )
    return (
        f"{channel} scale: {rows} row(s) of '{field}' fall outside the scale's explicit "
        "domain, so they are not drawn; widen the domain, or pass clamp=True to draw them "
        "at the axis edge"
    )


def _statistic_domain_warning(entry: str, channel: str, field: str, marks: int) -> str:
    return (
        f"{entry}: {channel} scale: {marks} mark(s) of '{field}' fall outside the scale's "
        "explicit domain, so they are not drawn; widen the domain, or pass clamp=True to draw "
        "them at the axis edge"
    )


def _statistic_cut_warning(entry: str, channel: str, field: str, marks: int) -> str:
    return (
        f"{entry}: {channel} scale: {marks} mark(s) of '{field}' cross the scale's explicit "
        "domain, so only their part inside it is drawn; widen the domain, or pass clamp=True "
        "to draw them at the axis edge"
    )


def _statistic_category_cut_warning(entry: str, channel: str, field: str, marks: int) -> str:
    return (
        f"{entry}: {channel} scale: {marks} mark(s) of '{field}' reach a category the scale's "
        "explicit domain leaves out, so only their part inside it is drawn; add it to the "
        "domain, or drop domain= to draw every category"
    )


def _domain_messages(msgs: list[str]) -> list[str]:
    """The explicit-domain left-out messages among *msgs*."""
    return [m for m in msgs if "explicit domain" in m]


_DF10 = pl.DataFrame({"x": list(range(10)), "y": [1.0] * 10})

#: A seeded 300-row standard-normal pair, for the statistic marks whose own
#: fit (a density curve, a hex-binned heatmap) needs a real distribution to
#: produce a curve or cells that cross a narrow explicit domain, rather
#: than dropping or keeping every unit uniformly.
_normal_rng = np.random.default_rng(0)
_NORMAL_300 = _normal_rng.normal(size=300).tolist()
_NORMAL_300_W = _normal_rng.normal(size=300).tolist()


# ---------------------------------------------------------------------------
# Point under an explicit continuous domain, by scale= and by xlim()
# ---------------------------------------------------------------------------


def test_point_under_an_explicit_linear_domain_warns_and_draws_the_same_marks_as_a_filtered_render():
    """Rejects a wrong fix that silently drops the rows (no warning) and one
    that draws or clamps them (which would also change the rendered bytes):
    the warned render's SVG is byte-identical to building the chart directly
    from the rows the domain admits, with the same domain stated explicitly
    so neither chart's axis infers anything."""
    scale = fm.LinearScale(domain=[2.0, 8.0])
    full = Chart(_DF10).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    admitted = _DF10.filter((pl.col("x") >= 2) & (pl.col("x") <= 8))
    filtered = (
        Chart(admitted)
        .mark_point()
        .encode(x=fm.X("x", scale=fm.LinearScale(domain=[2.0, 8.0])), y="y")
    )

    full_warnings, full_svg = _warnings_of(full.to_svg)
    filtered_warnings, filtered_svg = _warnings_of(filtered.to_svg)

    assert full_warnings == [_domain_warning("x", "x", 3)]
    assert filtered_warnings == []
    assert full_svg == filtered_svg


def test_xlim_warns_the_same_as_an_equivalent_explicit_scale_domain():
    """``xlim()`` is sugar for the same explicit-domain exclusion: it must
    warn with the identical text, not a silent drop or a differently worded
    message."""
    via_xlim = Chart(_DF10).mark_point().encode(x="x", y="y").xlim(2, 8)
    via_scale = (
        Chart(_DF10)
        .mark_point()
        .encode(x=fm.X("x", scale=fm.LinearScale(domain=[2.0, 8.0])), y="y")
    )
    xlim_warnings, _ = _warnings_of(via_xlim.to_svg)
    scale_warnings, _ = _warnings_of(via_scale.to_svg)
    assert xlim_warnings == scale_warnings == [_domain_warning("x", "x", 3)]


# ---------------------------------------------------------------------------
# bar / rule / tick / text / segment: same marks drawn, warned per channel
# ---------------------------------------------------------------------------


def test_bar_value_outside_a_y_domain_warns_and_still_draws_the_admitted_bars():
    """Rejects a fix that drops the bars silently or that draws every bar
    (ignoring the domain): the scene's drawn node count must equal exactly
    the rows the domain admits."""
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(_DF10).mark_bar().encode(x="x", y=fm.Y("x", scale=scale))
    msgs, _ = _warnings_of(chart.to_svg)
    assert msgs == [_domain_warning("y", "x", 3)]
    scene_msgs, scene = _warnings_of(lambda: _scene(chart))
    assert scene_msgs == msgs
    assert _drawn_node_count(scene) == 7


def test_rule_outside_an_x_domain_warns_and_still_draws_the_admitted_rules():
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(_DF10).mark_rule().encode(x=fm.X("x", scale=scale))
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "x", 3)]
    assert _drawn_node_count(scene) == 7


def test_tick_outside_an_x_domain_warns_and_still_draws_the_admitted_ticks():
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(_DF10).mark_tick().encode(x=fm.X("x", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "x", 3)]
    assert _drawn_node_count(scene) == 7


def test_text_outside_an_x_domain_warns_and_still_draws_the_admitted_labels():
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(_DF10).mark_text().encode(x=fm.X("x", scale=scale), y="y", text="x")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "x", 3)]
    assert _drawn_node_count(scene) == 7


def test_segment_with_either_end_outside_its_domain_warns_each_end_separately():
    """``x2`` shares ``x``'s axis scale (the one explicit domain), so a
    segment with either end outside it warns once per END, naming its own
    field and its own count: rejects a fix that merges the two ends into one
    count, or only checks one end."""
    df = pl.DataFrame(
        {
            "x": list(range(10)),
            "x2": [v + 0.3 for v in range(10)],
            "y": [1.0] * 10,
            "y2": [2.0] * 10,
        }
    )
    chart = (
        Chart(df)
        .mark_segment()
        .encode(
            x=fm.X("x", scale=fm.LinearScale(domain=[2.0, 8.0])),
            y="y",
            x2="x2",
            y2="y2",
        )
    )
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert sorted(msgs) == sorted([_domain_warning("x", "x", 3), _domain_warning("x2", "x2", 4)])
    assert _drawn_node_count(scene) == 6


# ---------------------------------------------------------------------------
# line / area: dropped and warned per vertex
# ---------------------------------------------------------------------------


def test_line_drops_and_warns_the_vertices_an_explicit_domain_leaves_out():
    """Rejects a fix that counts at scale-resolution time (pre-position):
    the line still draws a continuous polyline over exactly the admitted
    vertices, and the warned row count matches what the polyline actually
    lost."""
    df = pl.DataFrame({"x": list(range(10)), "y": [float(v) for v in range(10)]})
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(df).mark_line().encode(x=fm.X("x", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "x", 3)]
    assert _line_vertex_count(scene) == 7


def test_area_drops_and_warns_the_vertices_an_explicit_domain_leaves_out():
    df = pl.DataFrame({"x": list(range(10)), "y": [float(v) for v in range(10)]})
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(df).mark_area().encode(x=fm.X("x", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "x", 3)]
    assert _area_vertex_count(scene) == 7


def test_mark_function_clip_false_draws_past_the_domain_where_clip_true_warns():
    """The same function line, differing only in ``clip=``: ``clip=True``
    (the mark's own default) drops and warns the vertices past ``xlim``,
    and ``clip=False`` draws every one of them with no warning. Pairing
    both on one chart rejects a fix that never counts a ``mark_function``
    layer at all, which would also pass a ``clip=False``-only assertion."""

    def chart_with(clip):
        return (
            Chart(None)
            .mark_function(lambda x: x, domain=[0.0, 20.0], clip=clip)
            .coord(CoordCartesian(xlim=(2, 8)))
        )

    clipped_msgs, clipped_scene = _warnings_of(lambda: _scene(chart_with(True)))
    assert clipped_msgs == [_domain_warning("x", "x", 140)]
    assert _line_vertex_count(clipped_scene) == 60

    unclipped_msgs, unclipped_scene = _warnings_of(lambda: _scene(chart_with(False)))
    assert unclipped_msgs == []
    assert _line_vertex_count(unclipped_scene) == 200


def test_clip_false_is_scoped_to_the_line_family_a_clipped_bar_still_warns():
    """Confirms ``clip=False``'s no-warning exemption is read from the
    projection's own extrapolation (which only the line/area/ribbon family
    uses), not treated as a chart-wide or coord-wide silence: a bar beside a
    ``clip=False`` function line still warns only for itself, and a bar
    under ``CoordCartesian(clip=False)`` still warns too."""
    scale = fm.LinearScale(domain=[2.0, 8.0])
    bar = Chart(_DF10).mark_bar().encode(x=fm.X("x", scale=scale), y="y")

    beside_clip_false_line = bar + Chart(None).mark_function(
        lambda x: x, domain=[0.0, 20.0], clip=False
    )
    msgs, _ = _warnings_of(beside_clip_false_line.to_svg)
    assert msgs == [_domain_warning("x", "x", 3)]

    coord_clip_false = (
        Chart(_DF10).mark_bar().encode(x="x", y="y").coord(CoordCartesian(xlim=(2, 8), clip=False))
    )
    msgs, _ = _warnings_of(coord_clip_false.to_svg)
    assert msgs == [_domain_warning("x", "x", 3)]


# ---------------------------------------------------------------------------
# A stacked bar's implicit base below the domain draws fully, no warning
# ---------------------------------------------------------------------------


def test_a_stacked_bars_base_below_the_domain_draws_every_bar_with_no_warning():
    """Rejects a fix that counts a stacked bar's implicit base (0) as a
    left-out row merely because the domain excludes 0: every segment's top
    stays inside [5, 100], so nothing is actually left out."""
    df = pl.DataFrame(
        {"cat": ["a", "a", "b", "b"], "g": ["x", "y", "x", "y"], "v": [10.0, 20.0, 15.0, 25.0]}
    )
    scale = fm.LinearScale(domain=[5.0, 100.0])
    chart = Chart(df).mark_bar().encode(x="cat", y=fm.Y("v", scale=scale), color="g")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == []
    assert _drawn_node_count(scene) == 4


# ---------------------------------------------------------------------------
# BandScale / PointScale / OrdinalScale: an explicit domain missing categories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_cls", [fm.BandScale, fm.PointScale, fm.OrdinalScale])
def test_a_categorical_explicit_domain_missing_categories_warns_with_the_category_count(
    scale_cls,
):
    """Rejects a fix that only handles the continuous (numeric) case: the
    categorical message names rows and a category, not a numeric range."""
    df = pl.DataFrame({"c": list("abcdefghij"), "y": [float(v) for v in range(10)]})
    mark = "bar" if scale_cls is fm.BandScale else "point"
    scale = scale_cls(domain=list("cde"))
    chart = getattr(Chart(df), f"mark_{mark}")().encode(x=fm.X("c", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "c", 7, categorical=True)]
    assert _drawn_node_count(scene) == 3


# ---------------------------------------------------------------------------
# Seeded data-space Jitter under an explicit domain, x and y
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y"])
def test_seeded_jitter_under_an_explicit_domain_warns_the_points_it_moved_out(axis):
    """This seed moves at least one boundary row across the domain edge on
    both axes, so the reported count differs from the jitter-free sibling's
    (9, the rows the unjittered values alone put outside [5, 15]): rejects a
    fix that counts at scale-resolution time, before the jitter offset has
    moved the rows, which would report the unjittered count unchanged."""
    df = pl.DataFrame({"g": ["a"] * 20, "v": [float(i) for i in range(20)]})
    scale = fm.LinearScale(domain=[5.0, 15.0])
    position = fm.Jitter(width=0.5, axis=axis, seed=1)
    if axis == "x":
        chart = Chart(df).mark_point(position=position).encode(x=fm.X("v", scale=scale), y="g")
        unjittered = Chart(df).mark_point().encode(x=fm.X("v", scale=scale), y="g")
        reported, drawn = 10, 10
    else:
        chart = Chart(df).mark_point(position=position).encode(x="g", y=fm.Y("v", scale=scale))
        unjittered = Chart(df).mark_point().encode(x="g", y=fm.Y("v", scale=scale))
        reported, drawn = 11, 9

    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning(axis, "v", reported)]
    assert _drawn_node_count(scene) == drawn

    unjittered_msgs, _ = _warnings_of(unjittered.to_svg)
    assert unjittered_msgs == [_domain_warning(axis, "v", 9)]
    assert reported != 9


# ---------------------------------------------------------------------------
# clamp=True and no explicit domain: all drawn, no warning
# ---------------------------------------------------------------------------


def test_clamp_true_draws_every_row_at_the_axis_edge_with_no_warning():
    scale = fm.LinearScale(domain=[2.0, 8.0], clamp=True)
    chart = Chart(_DF10).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == []
    assert _drawn_node_count(scene) == 10


def test_no_explicit_domain_draws_every_row_with_no_warning():
    """A data-derived domain (or a jitter preview widened to fit every
    moved point) never exercises this warning: rejects a fix that fires
    whenever any value sits beyond today's resolved domain rather than only
    beyond an *explicit* one."""
    chart = Chart(_DF10).mark_point().encode(x="x", y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == []
    assert _drawn_node_count(scene) == 10


# ---------------------------------------------------------------------------
# Faceted: shared and independent scale resolution both sum over panels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolve", ["shared", "independent"])
def test_a_facet_sums_domain_left_out_rows_across_panels(resolve):
    """Rejects a fix that warns once per panel instead of summing into one
    chart-wide fact, and one that reads the facet's scale-resolution mode
    (shared vs. independent) as changing an *explicit* domain's bounds,
    which it must not: both modes drop (and report) the same total."""
    df = pl.DataFrame(
        {
            "x": list(range(10)) * 2,
            "y": [1.0] * 20,
            "panel": ["p1"] * 10 + ["p2"] * 10,
        }
    )
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y", facet="panel")
    chart = chart.share_scale(x=resolve)
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_domain_warning("x", "x", 6)]
    assert _drawn_node_count(scene) == 14


def test_a_layer_repeated_in_every_panel_counts_once_per_panel_it_repeats_in():
    """A layer whose frame carries no facet field repeats in every panel.
    Its own 3 excluded rows count once per panel it repeats in (2 panels,
    6), on top of the faceted layer's own 6 (3 per panel), for 12 total:
    rejects a fix that counts the repeated layer once for the whole chart
    instead of once per panel."""
    df = pl.DataFrame(
        {
            "x": list(range(10)) * 2,
            "y": [1.0] * 20,
            "panel": ["p1"] * 10 + ["p2"] * 10,
        }
    )
    scale = fm.LinearScale(domain=[2.0, 8.0])
    faceted = Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y", facet="panel")
    repeated = Chart(_DF10).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    chart = faceted + repeated
    msgs, _ = _warnings_of(chart.to_svg)
    assert msgs == [_domain_warning("x", "x", 12)]


# ---------------------------------------------------------------------------
# Every composition door reports the same channel, field and count
# ---------------------------------------------------------------------------


def test_every_composition_door_warns_the_same_channel_field_and_count():
    """Rejects a fix wired into only one door (the flat chart): every door
    below reports the identical single warning the flat chart does, with
    one contributor on each route. The sibling test below pairs two
    contributors per route to pin that their counts sum into one warning."""
    scale = fm.LinearScale(domain=[2.0, 8.0])
    flat = Chart(_DF10).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    rule = Chart(_DF10).mark_rule().encode(y="y")
    expected = [_domain_warning("x", "x", 3)]

    plus = flat + rule
    layered = flat.layer(fm.Layer(mark="rule", encoding={"y": "y"}))
    lc = fm.LayerChart(flat, rule)
    hc = fm.hconcat(flat, rule)

    for route, render in [
        ("flat", flat.to_svg),
        ("flat_interactive", lambda: _scene(flat)),
        ("plus", plus.to_svg),
        ("plus_interactive", lambda: _scene(plus)),
        ("chart_layer", layered.to_svg),
        ("chart_layer_interactive", lambda: _scene(layered)),
        ("layerchart_static", lc.to_svg),
        ("layerchart_interactive", lambda: _scene(lc)),
        ("hconcat", hc.to_svg),
        ("hconcat_interactive", lambda: _scene(hc)),
    ]:
        msgs, _ = _warnings_of(render)
        assert msgs == expected, f"{route} warned {msgs!r}, expected {expected!r}"


def test_two_contributors_on_every_composite_door_sum_into_one_warning():
    """Two members of a composite route, each leaving 3 rows out on the
    same channel and field, warn 6, not two separate warnings of 3: rejects
    a fix that counts per leaf or per member instead of summing into one
    chart-wide fact."""
    scale = fm.LinearScale(domain=[2.0, 8.0])
    flat = Chart(_DF10).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    expected = [_domain_warning("x", "x", 6)]

    plus = flat + flat
    lc = fm.LayerChart(flat, flat)
    hc = fm.hconcat(flat, flat)

    for route, render in [
        ("plus", plus.to_svg),
        ("layerchart_static", lc.to_svg),
        ("layerchart_interactive", lambda: _scene(lc)),
        ("hconcat", hc.to_svg),
    ]:
        msgs, _ = _warnings_of(render)
        assert msgs == expected, f"{route} warned {msgs!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# CoordFlip: the warning names the channel the caller wrote
# ---------------------------------------------------------------------------


def test_coord_flip_names_the_channel_the_caller_wrote():
    """The caller bound the domain to ``y``; under ``CoordFlip`` that draws
    on the horizontal axis, but the warning must still say ``y scale:``, not
    the post-flip internal slot: rejects a fix that un-flips at the wrong
    point and names the drawn axis instead of the written channel."""
    scale = fm.LinearScale(domain=[2.0, 8.0])
    chart = Chart(_DF10).mark_point().encode(x="y", y=fm.Y("x", scale=scale)).coord(CoordFlip())
    msgs, _ = _warnings_of(chart.to_svg)
    assert msgs == [_domain_warning("y", "x", 3)]


def test_a_plus_member_with_its_own_coord_flip_names_its_own_channel_and_field():
    """The member's own ``CoordFlip`` draws its ``x='hi'`` on the vertical
    axis the host's ``ylim`` bounds: the warning names the channel and field
    the member wrote (``x``, ``'hi'``), not the host's ``y`` or its
    ``'lo'``. Rejects a fix that un-flips with the composition's coord
    instead of the member's own, and one that names the drawn axis."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2, 3, 4, 5, 6],
            "lo": [1.0, 2, 3, 4, 5, 6],
            "hi": [2.0, 3, 14, 15, 16, 30],
        }
    )
    host = Chart(df).mark_point().encode(x="a", y="lo")
    member = Chart(df).mark_point().encode(x="hi", y="a").coord(CoordFlip())
    msgs, _ = _warnings_of((host + member).ylim(0, 10).to_svg)
    assert _domain_messages(msgs) == [_domain_warning("x", "hi", 4)]


# ---------------------------------------------------------------------------
# A statistic layer speaks in its entry's voice and counts marks, not rows
# ---------------------------------------------------------------------------


def test_a_histogram_bin_cut_on_one_end_counts_one_mark_in_the_entry_voice():
    """``mark_histogram``'s bar layer draws a statistic's output (its own
    ``Bin`` transform): the warning speaks as ``mark_histogram``, counts the
    one undrawn bar as ``1 mark(s)`` of the user's field ``'v'``, not
    ``1 row(s) of 'bin_end'``. Rejects a fix that keeps the primitive
    per-row voice for a layer whose rows a statistic produced."""
    df = pl.DataFrame({"v": [float(i) for i in range(20)]})
    chart = Chart(df).mark_histogram().encode(x="v").xlim(0, 17)
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_statistic_domain_warning("mark_histogram", "x", "v", 1)]
    assert _drawn_node_count(scene) == 5


def test_a_histogram_bin_cut_on_both_ends_counts_once_not_twice():
    """A bin whose ``bin_start`` and ``bin_end`` both fall outside the
    domain is one undrawn bar, not two: rejects a fix that reports
    ``bin_start``'s and ``bin_end``'s excluded counts as two separate
    warnings, or sums them and double-counts the bins cut on both ends."""
    df = pl.DataFrame({"v": [float(i) for i in range(20)]})
    chart = Chart(df).mark_histogram().encode(x="v").xlim(2, 8)
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_statistic_domain_warning("mark_histogram", "x", "v", 5)]
    assert _drawn_node_count(scene) == 1


def test_a_density_curve_cut_at_one_end_warns_the_cut_sentence_with_one_mark():
    """A domain narrower than the curve's own range still draws one path,
    trimmed at the edge: the census reports one mark crossing the domain,
    under the cut sentence, not the dropped one. Rejects a fix that still
    reports the vertex count (hundreds) or the dropped-whole sentence for a
    mark that in fact still draws."""
    df = pl.DataFrame({"v": _NORMAL_300})
    chart = Chart(df).mark_density().encode(x="v").coord(CoordCartesian(xlim=(-1, 1)))
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_statistic_cut_warning("mark_density", "x", "v", 1)]
    assert sum(len(b["nodes"]) for p in scene["panels"] for b in p["marks"]) == 1


def test_a_density_curve_entirely_outside_the_domain_warns_the_dropped_sentence():
    """A domain disjoint from the curve's own range leaves nothing drawn:
    the whole curve is one undrawn mark, under the dropped sentence, not
    the cut one. Rejects a fix that reports the cut sentence for a mark
    that in fact draws nothing."""
    df = pl.DataFrame({"v": _NORMAL_300})
    chart = Chart(df).mark_density().encode(x="v").coord(CoordCartesian(xlim=(20, 30)))
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == [_statistic_domain_warning("mark_density", "x", "v", 1)]
    assert sum(len(b["nodes"]) for p in scene["panels"] for b in p["marks"]) == 0


def test_a_statistic_line_left_out_on_both_axes_is_not_drawn_on_either():
    """``xlim`` and ``ylim`` each remove part of one fitted line, and
    together they leave too few vertices for a path: no path draws, and the
    one outcome reaches both axes under the not-drawn sentence. Rejects a
    fix that decides per axis, which reports the line cut on ``x`` (that
    axis alone leaves vertices) beside not drawn on ``y``."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(5, 2, 300), "w": rng.normal(0, 1, 300)})
    chart = Chart(df).mark_smooth().encode(x="v", y="w").xlim(4, 6).ylim(0.5, 1)
    msgs, svg = _warnings_of(chart.to_svg)
    assert sorted(_domain_messages(msgs)) == sorted(
        [
            _statistic_domain_warning("mark_smooth", "x", "v", 1),
            _statistic_domain_warning("mark_smooth", "y", "w", 1),
        ]
    )
    assert "<polyline" not in svg and "<path" not in svg


def test_a_statistic_line_reaching_a_left_out_category_warns_the_category_cut_sentence():
    """A mean line over four categories under a band domain of three still
    draws, without the fourth: the warning names the categorical cause and
    remedy with the drawn-cut consequence. Rejects the continuous cut
    sentence (its ``clamp=True`` does nothing on a categorical scale) and
    the categorical not-drawn sentence (the line is drawn)."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(5, 2, 300), "g": rng.choice(["a", "b", "c", "d"], 300)})
    for scale in (fm.BandScale(domain=["a", "b", "c"]), fm.PointScale(domain=["a", "b", "c"])):
        chart = (
            Chart(df).mark_line().encode(x=fm.X("g", scale=scale), y=fm.Y("v", aggregate="mean"))
        )
        msgs, svg = _warnings_of(chart.to_svg)
        assert _domain_messages(msgs) == [
            _statistic_category_cut_warning("mark_line", "x", "g", 1)
        ], type(scale).__name__
        assert svg.count("<polyline") == 1, type(scale).__name__


def _statistic_category_dropped_warning(entry: str, channel: str, field: str, marks: int) -> str:
    return (
        f"{entry}: {channel} scale: {marks} mark(s) of '{field}' name a category the scale's "
        "explicit domain leaves out, so they are not drawn; add it to the domain, or drop "
        "domain= to draw every category"
    )


def _group_channel(cls: Any) -> Any:
    """The caller's titled categorical channel, under a band domain that
    leaves out category ``'d'``."""
    return cls("g", scale=fm.BandScale(domain=["a", "b", "c"]), title="Group")


#: Each composite with a categorical axis, in every orientation it draws:
#: the build, and the one warning it gives for the marks of category ``'d'``.
_CATEGORICAL_COMPOSITES: dict[str, tuple[Callable[[pl.DataFrame], Any], str]] = {
    "mark_errorband": (
        lambda df: Chart(df).mark_errorband().encode(x=_group_channel(fm.X), y="v"),
        _statistic_category_cut_warning("mark_errorband", "x", "Group", 1),
    ),
    "mark_errorbar": (
        lambda df: Chart(df).mark_errorbar().encode(x=_group_channel(fm.X), y="v"),
        _statistic_category_dropped_warning("mark_errorbar", "x", "Group", 3),
    ),
    "mark_boxplot": (
        lambda df: Chart(df).mark_boxplot().encode(x=_group_channel(fm.X), y="v"),
        _statistic_category_dropped_warning("mark_boxplot", "x", "Group", 10),
    ),
    "mark_boxplot[horizontal]": (
        lambda df: Chart(df).mark_boxplot(horizontal=True).encode(y=_group_channel(fm.Y), x="v"),
        _statistic_category_dropped_warning("mark_boxplot", "y", "Group", 10),
    ),
    "mark_violin": (
        lambda df: Chart(df).mark_violin().encode(x=_group_channel(fm.X), y="v"),
        _statistic_category_dropped_warning("mark_violin", "x", "Group", 6),
    ),
    "mark_violin[horizontal]": (
        lambda df: Chart(df).mark_violin(horizontal=True).encode(y=_group_channel(fm.Y), x="v"),
        _statistic_category_dropped_warning("mark_violin", "y", "Group", 6),
    ),
    "mark_swarm": (
        lambda df: Chart(df).mark_swarm().encode(x=_group_channel(fm.X), y="v"),
        _statistic_category_dropped_warning("mark_swarm", "x", "Group", 76),
    ),
    "mark_swarm[horizontal]": (
        lambda df: Chart(df).mark_swarm(orient="horizontal").encode(y=_group_channel(fm.Y), x="v"),
        _statistic_category_dropped_warning("mark_swarm", "y", "Group", 76),
    ),
    "mark_boxen": (
        lambda df: Chart(df).mark_boxen().encode(x=_group_channel(fm.X), y="v"),
        _statistic_category_dropped_warning("mark_boxen", "x", "Group", 24),
    ),
    "mark_boxen[horizontal]": (
        lambda df: Chart(df).mark_boxen(horizontal=True).encode(y=_group_channel(fm.Y), x="v"),
        _statistic_category_dropped_warning("mark_boxen", "y", "Group", 24),
    ),
}


@pytest.mark.parametrize("case", sorted(_CATEGORICAL_COMPOSITES))
def test_a_composites_categorical_axis_names_the_callers_title(case):
    """A ``title=`` on a composite's categorical channel reaches its
    statistic layers' categorical binding too, in either orientation, so a
    category the domain leaves out is warned under the name the axis shows.
    Rejects a fix that stamps the caller's title on the value axis alone, or
    reads the categorical title for the vertical orientation only."""
    build, expected = _CATEGORICAL_COMPOSITES[case]
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.normal(5, 2, 300), "g": rng.choice(["a", "b", "c", "d"], 300)})
    msgs, _ = _warnings_of(build(df).to_svg)
    assert _domain_messages(msgs) == [expected], case


def _unnamed_statistic_warning(entry: str, channel: str, marks: int, *, cut: bool = False) -> str:
    """A statistic sentence whose layer shows no name: the ``of '<field>'``
    clause is dropped and the rest is byte-identical."""
    if cut:
        return (
            f"{entry}: {channel} scale: {marks} mark(s) cross the scale's explicit domain, so "
            "only their part inside it is drawn; widen the domain, or pass clamp=True to draw "
            "them at the axis edge"
        )
    return (
        f"{entry}: {channel} scale: {marks} mark(s) fall outside the scale's explicit domain, "
        "so they are not drawn; widen the domain, or pass clamp=True to draw them at the axis "
        "edge"
    )


_SUPPRESSED_TITLE_DF = pl.DataFrame(
    {"v": [float(i) for i in range(20)], "w": [float(i) * 2 for i in range(20)]}
)

#: Statistic layers on four doors, each under an x title the case sets, with
#: its x limit, its mark count and whether the sentence is the cut one.
_SUPPRESSED_TITLE_DOORS: dict[str, tuple[Callable[[Any], Any], str, int, bool]] = {
    "mark_histogram": (
        lambda x: Chart(_SUPPRESSED_TITLE_DF).mark_histogram().encode(x=x).xlim(0, 17),
        "mark_histogram",
        1,
        False,
    ),
    "mark_density": (
        lambda x: Chart(pl.DataFrame({"v": _NORMAL_300})).mark_density().encode(x=x).xlim(-1, 1),
        "mark_density",
        1,
        True,
    ),
    "residplot": (
        lambda x: fm.residplot(_SUPPRESSED_TITLE_DF, x=x, y="w").xlim(5, 6),
        "residplot",
        36,
        False,
    ),
    # `cut=True`: the ecdf steps at the caller's own observations
    # (`v = 0..19`), so `v=5` and `v=6` are drawn vertices inside
    # `xlim(5, 6)` while the rest of the line's vertices sit outside it --
    # one mark, drawn cut.
    "displot ecdf": (
        lambda x: fm.displot(_SUPPRESSED_TITLE_DF, x=x, kind="ecdf").xlim(5, 6),
        "displot",
        1,
        True,
    ),
}


#: Every spelling that suppresses an axis title: ``None``, empty and blank.
_SUPPRESSED_TITLES = pytest.mark.parametrize(
    "title", [None, "", "  "], ids=["None", "empty", "blank"]
)


@_SUPPRESSED_TITLES
@pytest.mark.parametrize("door", sorted(_SUPPRESSED_TITLE_DOORS))
def test_a_statistic_layer_with_a_suppressed_title_drops_the_field_clause(door, title):
    """A statistic layer whose axis shows no name warns without the
    ``of '<field>'`` clause, the sentence otherwise byte-identical, on the
    not-drawn and the cut sentence alike. Rejects naming an empty field
    (``of ''``) and falling back to the internal column the layer reads."""
    build, entry, marks, cut = _SUPPRESSED_TITLE_DOORS[door]
    msgs, _ = _warnings_of(build(fm.X("v", title=title)).to_svg)
    assert _domain_messages(msgs) == [_unnamed_statistic_warning(entry, "x", marks, cut=cut)]


_GROUPED_DF = _SUPPRESSED_TITLE_DF.with_columns(g=pl.Series(["a", "b"] * 10))

#: Doors whose lowering binds the caller's channel at the chart level and
#: rebinds its field on the statistic layers: the build over a channel
#: object, the axis it limits, and the one statistic warning it gives.
_FIELD_KEEPING_DOORS: dict[str, tuple[Callable[[Any], Any], str]] = {
    "mark_boxplot": (
        lambda y: Chart(_GROUPED_DF).mark_boxplot().encode(x="g", y=y).ylim(100, 200),
        _statistic_domain_warning("mark_boxplot", "y", "v", 10),
    ),
    "catplot bar": (
        lambda y: fm.catplot(_GROUPED_DF, x="g", y=y, kind="bar").ylim(100, 200),
        _statistic_domain_warning("catplot", "y", "v", 4),
    ),
    "lmplot": (
        lambda x: fm.lmplot(_SUPPRESSED_TITLE_DF, x=x, y="w").xlim(100, 200),
        _statistic_domain_warning("lmplot", "x", "v", 202),
    ),
}


@_SUPPRESSED_TITLES
@pytest.mark.parametrize("door", sorted(_FIELD_KEEPING_DOORS))
def test_a_suppressed_title_keeps_the_field_where_the_lowering_knows_it(door, title):
    """A composite, catplot or lmplot under a suppressed title (None, empty or
    blank) still names the caller's field: its statistic layers rebind the
    column the caller wrote. Rejects passing a blank title through as drawn,
    which drops the clause on these doors too."""
    build, expected = _FIELD_KEEPING_DOORS[door]
    channel = (fm.X if door == "lmplot" else fm.Y)("v", title=title)
    msgs, _ = _warnings_of(build(channel).to_svg)
    statistic = [m for m in _domain_messages(msgs) if "mark(s)" in m]
    assert statistic == [expected]


@pytest.mark.parametrize("door", sorted(_SUPPRESSED_TITLE_DOORS))
def test_a_statistic_layer_with_a_title_names_it(door):
    """Control: the same doors under ``title='Value'`` keep the clause and
    name the title."""
    build, entry, marks, cut = _SUPPRESSED_TITLE_DOORS[door]
    msgs, _ = _warnings_of(build(fm.X("v", title="Value")).to_svg)
    sentence = _statistic_cut_warning if cut else _statistic_domain_warning
    assert _domain_messages(msgs) == [sentence(entry, "x", "Value", marks)]


def test_a_primitive_layer_with_a_suppressed_title_keeps_naming_its_field():
    """The row voice is unchanged: a primitive layer names the field it
    reads whatever its title."""
    chart = Chart(_DF10).mark_point().encode(x=fm.X("x", title=None), y="y").xlim(2, 8)
    msgs, _ = _warnings_of(chart.to_svg)
    assert msgs == [_domain_warning("x", "x", 3)]


def test_a_raster_outside_the_domain_warns_one_mark_and_draws_no_image():
    """The raster is one image mark: a domain disjoint from the grid on
    either axis drops it whole and warns once, naming the caller's field;
    a domain covering the grid draws the image and warns nothing. Rejects
    a census that counts the raster's cells or its source rows, and one
    that names the extent columns the image is placed by."""
    grid = pl.DataFrame(
        {
            "x": [float(i) for i in range(10) for _ in range(10)],
            "y": [float(j) for _ in range(10) for j in range(10)],
        }
    )
    raster = Chart(grid).mark_raster().encode(x="x", y="y")
    for axis, limited in (("x", raster.xlim(20, 80)), ("y", raster.ylim(20, 80))):
        msgs, svg = _warnings_of(limited.to_svg)
        assert _domain_messages(msgs) == [
            _statistic_domain_warning("mark_raster", axis, axis, 1)
        ], axis
        assert "<image" not in svg, axis
    msgs, svg = _warnings_of(raster.xlim(-1, 11).to_svg)
    assert _domain_messages(msgs) == []
    assert svg.count("<image") == 1


def test_mark_hex_reports_dropped_and_cut_hexes_separately():
    """Some hexes sit entirely outside the domain (dropped whole) and
    others straddle its edge (drawn, cut): rejects a fix that reports only
    one of the two sentences, or that sums the two counts into one."""
    df = pl.DataFrame({"v": _NORMAL_300, "w": _NORMAL_300_W})
    chart = Chart(df).mark_hex().encode(x="v", y="w").coord(CoordCartesian(xlim=(-1, 1)))
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert sorted(msgs) == sorted(
        [
            _statistic_domain_warning("mark_hex", "x", "v", 51),
            _statistic_cut_warning("mark_hex", "x", "v", 10),
        ]
    )
    assert sum(len(b["nodes"]) for p in scene["panels"] for b in p["marks"]) == 69


def test_a_second_statistic_layer_also_speaks_in_its_entry_voice():
    """A different statistic layer (a ``displot`` KDE overlay, reached
    through a named ``data_source`` rather than a composite mark's own
    primary layer) confirms the entry-voice rule is not special-cased to
    one mark or one wiring shape, and names the user's field ``'v'``, not
    the overlay's own internal column ``'value'``. The KDE curve is one
    mark whose domain-crossing end is trimmed, not dropped whole, so it
    warns under the cut sentence, separately from the histogram bars'
    dropped-bar warning on the same chart."""
    df = pl.DataFrame({"v": [float(i) for i in range(1, 21)]})
    chart = fm.displot(df, x="v", kind="hist", kde=True).coord(CoordCartesian(xlim=(5, 15)))
    msgs, _ = _warnings_of(chart.to_svg)
    assert sorted(msgs) == sorted(
        [
            _statistic_domain_warning("displot", "x", "v", 4),
            _statistic_cut_warning("displot", "x", "v", 1),
        ]
    )


# ---------------------------------------------------------------------------
# A LogScale's off-side warning and this one never double-count a row
# ---------------------------------------------------------------------------


def test_log_off_side_rows_are_not_double_counted_with_the_explicit_domain_warning():
    """One row (``-5.0``) is off-side for the ``LogScale`` (reported by
    ``LogScaleRowsLeftOut``) and a different row (``50.0``) is on-side but
    outside the explicit domain (reported by this warning): rejects a fix
    that either merges the two into one count or reports the off-side row
    under both mechanisms."""
    df = pl.DataFrame({"x": [-5.0, 1.0, 2.0, 50.0], "y": [1.0, 2.0, 3.0, 4.0]})
    scale = fm.LogScale(domain=[1.0, 10.0])
    chart = Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    off_side = (
        "x scale: 1 row(s) of 'x' are zero or negative, which a LogScale cannot place, "
        "so they are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
    )
    assert sorted(msgs) == sorted([off_side, _domain_warning("x", "x", 1)])
    # Two rows are left out in total (one off-side, one outside the domain);
    # the other two (1.0 and 2.0) draw.
    assert _drawn_node_count(scene) == 2


# ---------------------------------------------------------------------------
# A placement column naming a column the batch lacks refuses, not drops
# ---------------------------------------------------------------------------


def test_an_unlayered_bars_x2_naming_an_unknown_column_refuses_in_the_marks_voice():
    """An unlayered ``mark_bar``'s ``x2`` naming a column its rows lack
    refuses in the mark's own voice, naming the column: rejects a fix that
    swallows the column read and renders the bar with ``x2`` unbound, and
    one that drops the entry prefix every other voiced producer on this
    door keeps (``mark_rule``'s ``x2``, ``mark_text``'s ``text``, below)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError) as exc_info:
        Chart(df).mark_bar().encode(x="x", x2="zzz", y="y").to_svg()
    assert str(exc_info.value) == "mark_bar: unknown column 'zzz' referenced by an encoding"


def test_mark_rule_x2_and_mark_text_text_naming_an_unknown_column_keep_the_entry_voice():
    """``mark_rule``'s ``x2`` and ``mark_text``'s ``text`` are voiced
    producers on the unlayered door independently of the domain census:
    rejects a fix that makes any in-layer ``UnknownColumn`` bare, which
    would also strip these two refusals' entry prefix."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError) as exc_info:
        Chart(df).mark_rule().encode(x="x", x2="zzz", y="y").to_svg()
    assert str(exc_info.value) == "mark_rule: unknown column 'zzz' referenced by an encoding"

    with pytest.raises(ValueError) as exc_info:
        Chart(df).mark_text().encode(x="x", y="y", text="zzz").to_svg()
    assert str(exc_info.value) == "mark_text: unknown column 'zzz' referenced by an encoding"


def test_the_same_unknown_x2_column_on_a_layered_door_names_the_layer():
    """The layered door's existing voice is unchanged: it still names the
    layer and wraps the column gate's own text, distinct from the
    unlayered door's bare refusal above."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    layered = Chart(df).mark_bar().encode(x="x", x2="zzz", y="y") + Chart(df).mark_point().encode(
        x="x", y="y"
    )
    with pytest.raises(ValueError) as exc_info:
        layered.to_svg()
    assert str(exc_info.value) == (
        "layer 0: encoding 'x2' references unknown column 'zzz' (not present in that layer's data)"
    )


# ---------------------------------------------------------------------------
# A non-positional binding naming a column a statistic layer's own output
# lacks drops and warns, the same way on .layer(), a figure's own door and +
# ---------------------------------------------------------------------------


def test_a_statistic_layers_inherited_binding_drops_and_warns_like_the_plus_door():
    """``residplot``'s own door and ``mark_histogram().layer()`` both
    inherit a non-positional binding (``opacity``) naming a source column
    their own statistic's output does not carry: both drop the binding,
    warn naming it, and still render, exactly as ``+`` already does for
    the same mistake. Rejects a fix that refuses instead of warning on
    either the own-figure or the ``.layer()`` door."""
    df = pl.DataFrame({"v": [float(i) for i in range(20)], "w": [float(i) * 2 for i in range(20)]})

    plus_msgs, plus_svg = _warnings_of(
        (Chart(df).mark_histogram().encode(x="v") + Chart(df).mark_point().encode(x="v", y="w"))
        .encode(opacity="v")
        .to_svg
    )
    expected = "mark_histogram: opacity 'v' was dropped: the statistic this bar mark draws does not carry 'v', so it draws without opacity"
    assert plus_msgs == [expected]
    assert "<svg" in plus_svg

    layer_msgs, layer_svg = _warnings_of(
        Chart(df).mark_histogram().encode(x="v").layer().encode(opacity="v").to_svg
    )
    assert layer_msgs == [expected]
    assert "<svg" in layer_svg

    residplot_msgs, residplot_svg = _warnings_of(
        fm.residplot(df, x="v", y="w").encode(opacity="v").to_svg
    )
    assert residplot_msgs == [
        "residplot: opacity 'v' was dropped: the statistic this point mark draws does not "
        "carry 'v', so it draws without opacity",
        "residplot: opacity 'v' was dropped: the statistic this rule mark draws does not "
        "carry 'v', so it draws without opacity",
        "residplot: opacity 'v' was dropped: the statistic this text mark draws does not "
        "carry 'v', so it draws without opacity",
    ]
    assert "<svg" in residplot_svg


# ---------------------------------------------------------------------------
# null, NaN and inf positional values are not counted
# ---------------------------------------------------------------------------


def test_null_nan_and_inf_positional_values_are_not_counted():
    """A non-finite or null positional value is dropped with no warning from
    either mechanism; this warning does not count them either. Rejects a fix
    that treats a non-finite value as "outside the domain" and folds it into
    this warning's count."""
    df = pl.DataFrame(
        {
            "x": [2.0, None, float("nan"), float("inf"), 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    scale = fm.LinearScale(domain=[0.0, 10.0])
    chart = Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    msgs, scene = _warnings_of(lambda: _scene(chart))
    assert msgs == []
    assert _drawn_node_count(scene) == 2
