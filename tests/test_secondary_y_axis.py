"""Structural SVG tests for ``LayerChart(resolve={"y": "independent"})`` --
the secondary-y-axis / dual-axis feature (GH #52, Task 4).

These are behavioral, not golden-byte, assertions: they parse the rendered
SVG's text/mark elements to prove the dual-axis geometry actually exists
(two distinct y-axis tick columns with per-layer titles, each layer's marks
spanning the full plot height off its OWN scale rather than being squashed
by a shared union domain, N stacked right axes with no overlap, per-axis
temporal/numeric tick formatting) rather than pinning exact pixel
coordinates that would break on any layout tweak.

RED discipline: on ``main`` (pre-#52 Task 4), every ``resolve={"y":
"independent"}`` construction here raises ``ValueError`` (the overlay
contract), so every test in this module fails at the ``LayerChart(...)``
call itself until the Python-side routing lands.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import HConcatChart, LayerChart, RepeatChart, _lower_composite
from tests._svg_extents import y_axis_extents


def _svg_body(svg: str) -> str:
    """Strip the embedded base64 font block so regexes run fast."""
    idx = svg.find("</defs>")
    return svg[idx:] if idx != -1 else svg


def _rotated_text(svg: str) -> list[tuple[float, str]]:
    """Return (x, text) for every ``<text>`` with a ``rotate(...)`` transform.

    Axis titles are the only rotated text nodes ferrum's SVG renderer emits
    for these single-mark-per-layer charts; tick labels are unrotated. This
    recovers each y-axis's title (and its x position, to order left-to-right)
    regardless of which side it renders on.
    """
    body = _svg_body(svg)
    out = []
    for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", body):
        if "rotate(" not in attrs:
            continue
        x_m = re.search(r'x="([^"]+)"', attrs)
        out.append((float(x_m.group(1)), text.strip()))
    return out


def _bar_rect_heights(svg: str) -> list[float]:
    """Return the ``height`` of every mark ``<rect>`` (excludes the background/plot-area rects)."""
    body = _svg_body(svg)
    heights = []
    for attrs in re.findall(r"<rect ([^/]*)/>", body):
        if 'fill="#faf7f2"' in attrs:  # figure background
            continue
        w_m = re.search(r'width="([^"]+)"', attrs)
        h_m = re.search(r'height="([^"]+)"', attrs)
        if w_m and h_m and float(w_m.group(1)) < 400:  # exclude the plot-area rect
            heights.append(float(h_m.group(1)))
    return heights


def _polyline_y_values(svg: str) -> list[float]:
    body = _svg_body(svg)
    ys = []
    for points in re.findall(r'<polyline points="([^"]+)"', body):
        for pt in points.split():
            _, y = pt.split(",")
            ys.append(float(y))
    return ys


def _plot_height(svg: str) -> float:
    """The plot-area rect's ``height``.

    ferrum's SVG output emits the figure background rect first (``x="0"
    y="0"``, spans the full viewBox), then the plot-area rect (offset by
    the left/top margins). Skip the background by requiring ``x > 0``.
    """
    body = _svg_body(svg)
    for attrs in re.findall(r"<rect ([^/]*)/>", body):
        x_m = re.search(r'x="([^"]+)"', attrs)
        h_m = re.search(r'height="([^"]+)"', attrs)
        if x_m and h_m and float(x_m.group(1)) > 0:
            return float(h_m.group(1))
    raise AssertionError("plot-area rect not found")


# ---------------------------------------------------------------------------
# Criterion 1 / 5: two-layer independent y renders two y-axes.
# ---------------------------------------------------------------------------


def test_two_layer_independent_y_renders_two_axes():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "y2": [100.0, 200.0, 150.0, 300.0],
        }
    )
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    line = fm.Chart(df).mark_line().encode(x="x", y="y2")
    svg = LayerChart(bars, line, resolve={"y": "independent"}).to_svg()

    titles = [t for _, t in _rotated_text(svg)]
    assert "y" in titles
    assert "y2" in titles

    extents = y_axis_extents(svg)
    assert len(extents) == 2, f"expected 2 distinct y-axis tick columns, got {extents}"


def test_independent_y_interactive_renders_without_raising():
    """Same chart's .interactive() scene render (one-panel merged path)."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "y2": [100.0, 200.0, 150.0, 300.0],
        }
    )
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    line = fm.Chart(df).mark_line().encode(x="x", y="y2")
    layered = LayerChart(bars, line, resolve={"y": "independent"})
    scene_json, packed_data = layered._render_interactive()
    assert scene_json
    assert isinstance(packed_data, (bytes, bytearray))
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 1, "LayerChart interactive is always one scene panel"


# ---------------------------------------------------------------------------
# Per-layer positioning: disjoint y domains must NOT collapse one layer to a
# sliver -- each layer resolves its own scale (spec §4 "Scales").
# ---------------------------------------------------------------------------


def test_disjoint_domain_layers_each_use_the_full_plot_height():
    """Bar layer (domain ~[0, 4]) and line layer (domain ~[100, 300]) both
    span close to the full plot height off their OWN scales.

    If the layers wrongly shared one union domain (~[0, 300]), the bar
    layer's marks would collapse to a sliver a few percent of the plot
    height tall instead of spanning nearly all of it -- that's the failure
    this test discriminates against.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "y2": [100.0, 200.0, 150.0, 300.0],
        }
    )
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    line = fm.Chart(df).mark_line().encode(x="x", y="y2")
    svg = LayerChart(bars, line, resolve={"y": "independent"}).to_svg()

    plot_h = _plot_height(svg)
    tallest_bar = max(_bar_rect_heights(svg))
    assert tallest_bar > 0.9 * plot_h, (
        f"tallest bar ({tallest_bar}) should span most of the plot height "
        f"({plot_h}) under its own [0, 4] scale, not a shared [0, 300] union"
    )

    line_ys = _polyline_y_values(svg)
    line_span = max(line_ys) - min(line_ys)
    assert line_span > 0.9 * plot_h, (
        f"line pixel span ({line_span}) should cover most of the plot height "
        f"({plot_h}) under its own [100, 300] scale"
    )


# ---------------------------------------------------------------------------
# Criterion 3: three-layer independent y stacks two right axes outward, no
# overlap.
# ---------------------------------------------------------------------------


def test_three_layer_independent_y_stacks_right_axes_outward():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "y2": [100.0, 200.0, 150.0, 300.0],
            "y3": [-5.0, -2.0, -8.0, -1.0],
        }
    )
    a = fm.Chart(df).mark_bar().encode(x="x", y="y")
    b = fm.Chart(df).mark_line().encode(x="x", y="y2")
    c = fm.Chart(df).mark_point().encode(x="x", y="y3")
    svg = LayerChart(a, b, c, resolve={"y": "independent"}).to_svg()

    titled = sorted(_rotated_text(svg))
    labels = [t for _, t in titled]
    assert labels == ["y", "y2", "y3"], f"expected left-to-right y, y2, y3, got {titled}"

    xs = [x for x, _ in titled]
    # Left axis (primary), then two right axes stacked outward: each
    # subsequent axis's title sits strictly further right than the last,
    # with a real gap (no overlapping label bands).
    assert xs[0] < xs[1] < xs[2]
    assert xs[2] - xs[1] > 10.0, "third axis must reserve its own band beyond the second"

    extents = y_axis_extents(svg)
    assert len(extents) == 3


# ---------------------------------------------------------------------------
# Criterion 10: temporal y in one layer + numeric y in the other -- per-axis
# tick formatting.
# ---------------------------------------------------------------------------


def test_temporal_and_numeric_independent_y_axes_format_independently():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "t": [
                dt.date(2024, 1, 1),
                dt.date(2024, 2, 1),
                dt.date(2024, 3, 1),
                dt.date(2024, 4, 1),
            ],
        }
    )
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    line = fm.Chart(df).mark_line().encode(x="x", y="t")
    svg = LayerChart(bars, line, resolve={"y": "independent"}).to_svg()

    body = _svg_body(svg)
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    all_texts = [text.strip() for _, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", body)]
    date_ticks = [t for t in all_texts if date_pattern.match(t)]
    assert date_ticks, "expected at least one ISO-date-formatted tick on the temporal y-axis"

    numeric_ticks = [t for t in all_texts if re.match(r"^-?\d+(\.\d+)?$", t)]
    assert any(t in numeric_ticks for t in ("0", "1", "2", "3", "4")), (
        "expected plain numeric ticks on the numeric y-axis"
    )

    titles = [t for _, t in _rotated_text(svg)]
    assert "y" in titles
    assert "t" in titles


# ---------------------------------------------------------------------------
# Criterion 11: a layer with no y encoding joins the primary scale -- no
# crash, no phantom axis.
# ---------------------------------------------------------------------------


def test_layer_with_no_y_encoding_joins_primary_no_phantom_axis():
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    rule_df = pl.DataFrame({"thresh": [2.5]})
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    rule = fm.Chart(rule_df).mark_rule().encode(x="thresh")

    layered = LayerChart(bars, rule, resolve={"y": "independent"})
    svg = layered.to_svg()  # must not raise/error

    extents = y_axis_extents(svg)
    assert len(extents) == 1, f"expected exactly one y-axis (no phantom axis), got {extents}"

    titles = [t for _, t in _rotated_text(svg)]
    assert titles == ["y"], f"expected only the primary layer's title, got {titles}"

    # The interactive path must also survive the no-y layer without raising.
    layered._render_interactive()


# ---------------------------------------------------------------------------
# Criterion 11 (single-layer degenerate): independent y with only one chart
# renders normally -- left axis only, identical to that chart's own render.
# ---------------------------------------------------------------------------


def test_single_layer_independent_y_renders_left_axis_only():
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")

    single = LayerChart(chart, resolve={"y": "independent"})
    svg = single.to_svg()

    extents = y_axis_extents(svg)
    assert len(extents) == 1
    titles = [t for _, t in _rotated_text(svg)]
    assert titles == ["y"]

    # A single-layer LayerChart has no "non-primary" layer to make
    # independent, so it degenerates to exactly the wrapped chart's own render.
    assert svg == chart.to_svg()


# ---------------------------------------------------------------------------
# Criterion 4: default / y:"shared" LayerChart output is unaffected.
# ---------------------------------------------------------------------------


def test_default_shared_y_layer_chart_unaffected():
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y")
    b = fm.Chart(df).mark_line().encode(x="x", y="y")

    default_svg = LayerChart(a, b).to_svg()
    explicit_shared_svg = LayerChart(a, b, resolve={"y": "shared"}).to_svg()
    assert default_svg == explicit_shared_svg

    extents = y_axis_extents(default_svg)
    assert len(extents) == 1, "shared-y LayerChart renders exactly one y-axis"

    # The default/shared render still goes through the composite overlay
    # tree, not the merged flat path -- confirm no independent_y flag rides
    # the wire spec for either layer.
    lowered = LayerChart(a, b)._composite_tree(auto_tooltips=False)
    assert lowered.tree["resolve"]["y"] == "shared"


# ---------------------------------------------------------------------------
# Task 5: nesting -- independent-y LayerChart inside a composite lowers as
# ONE leaf carrying its per-layer slots (spec §4 "Nesting").
# ---------------------------------------------------------------------------


def _dual_axis_layer_chart(df):
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    line = fm.Chart(df).mark_line().encode(x="x", y="y2")
    return LayerChart(bars, line, resolve={"y": "independent"})


def test_independent_y_layer_chart_nested_in_hconcat_lowers_to_one_leaf():
    """A dual-axis LayerChart nested inside HConcat lowers as ONE leaf, not a
    nested overlay composite tree -- it carries its per-layer slots via the
    merged flat spec (GH #52 spec §4 "Nesting")."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    dual = _dual_axis_layer_chart(df)

    composite = HConcatChart([other, dual])
    lowered = _lower_composite(composite, auto_tooltips=False)

    kinds = [c["kind"] for c in lowered.tree["children"]]
    assert kinds == ["leaf", "leaf"], (
        f"dual-axis LayerChart must nest as ONE leaf, not a nested composite; got {kinds}"
    )


def test_independent_y_layer_chart_nested_in_hconcat_renders():
    """The composite renders without raising; the plain sibling panel's one
    y-axis and the dual-axis panel's two y-axes are both present."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    dual = _dual_axis_layer_chart(df)

    svg = (other | dual).to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 3, (
        f"expected 1 (plain panel) + 2 (dual-axis panel) y-axis tick columns, got {extents}"
    )


def test_independent_y_layer_chart_nested_in_hconcat_interactive_renders():
    """The interactive scene render survives nesting an independent-y
    LayerChart in an HConcat: two scene panels, no raise."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    dual = _dual_axis_layer_chart(df)

    scene_json, packed_data = (other | dual)._render_interactive()
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 2
    assert isinstance(packed_data, (bytes, bytearray))


def test_parent_explicit_shared_y_over_independent_y_layer_raises():
    """An explicit parent resolve={"y": "shared"} over a subtree containing
    a dual-axis LayerChart is contradictory: the leaf's per-layer y slots
    don't participate in cross-panel sharing (spec §6 errors)."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    dual = _dual_axis_layer_chart(df)

    composite = HConcatChart([other, dual], resolve={"y": "shared"})
    with pytest.raises(ValueError, match=r"HConcatChart:.*'y'.*'shared'.*independent-y"):
        composite.to_svg()


def test_parent_shared_x_over_independent_y_layer_does_not_raise():
    """x sharing is unaffected by a nested independent-y LayerChart -- only
    an explicit y:"shared" conflicts with it."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    other = fm.Chart(df).mark_point().encode(x="x", y="y")
    dual = _dual_axis_layer_chart(df)

    composite = HConcatChart([other, dual], resolve={"x": "shared"})
    svg = composite.to_svg()  # must not raise
    assert svg


# ---------------------------------------------------------------------------
# Task 5: multi-y-layer non-first member under independent y raises (Task
# 4 quality-review follow-up) -- the per-layer boolean wire cannot group a
# member's internal layers into one right-axis slot.
# ---------------------------------------------------------------------------


def test_multi_y_layer_non_first_member_raises():
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    primary = fm.Chart(df).mark_bar().encode(x="x", y="y")
    a = fm.Chart(df).mark_line().encode(x="x", y="y2")
    b = fm.Chart(df).mark_point().encode(x="x", y="y2")
    multi_layer_member = a + b  # merges into one Chart with two y-bearing layers

    layered = LayerChart(primary, multi_layer_member, resolve={"y": "independent"})
    with pytest.raises(ValueError, match=r"LayerChart:.*position 1.*2 y-bearing layers"):
        layered.to_svg()


def test_multi_y_layer_non_first_member_raises_on_interactive_entry_too():
    """The multi-y-bearing-member guard lives in ``_build_merged``, which is
    shared by both render entries — prove it fires through the interactive
    path as well, not just ``to_svg``."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    primary = fm.Chart(df).mark_bar().encode(x="x", y="y")
    a = fm.Chart(df).mark_line().encode(x="x", y="y2")
    b = fm.Chart(df).mark_point().encode(x="x", y="y2")
    multi_layer_member = a + b

    layered = LayerChart(primary, multi_layer_member, resolve={"y": "independent"})
    with pytest.raises(ValueError, match=r"LayerChart:.*position 1.*2 y-bearing layers"):
        layered._render_interactive()


def test_multi_layer_primary_member_is_fine():
    """The primary (first) member chart may be multi-layer -- only non-first
    members are restricted to a single y-bearing layer under independent y."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    a = fm.Chart(df).mark_bar().encode(x="x", y="y")
    b = fm.Chart(df).mark_point().encode(x="x", y="y")
    multi_layer_primary = a + b  # two layers, both y-bearing, both primary (shared)
    secondary = fm.Chart(df).mark_line().encode(x="x", y="y2")

    layered = LayerChart(multi_layer_primary, secondary, resolve={"y": "independent"})
    svg = layered.to_svg()  # must not raise

    extents = y_axis_extents(svg)
    assert len(extents) == 2, f"expected 1 (shared primary) + 1 (secondary) axis, got {extents}"


# ---------------------------------------------------------------------------
# Task 10f bug #1: mark_line(point=True) merges its line+point layers via
# ``+`` BEFORE ``.encode()`` runs, so each layer's OWN encoding snapshot is
# empty and the chart-level y only lives on the composite chart's
# ``_encoding``. A non-first member built this way must hit the same
# multi-y-layer guard as the pre-merge ``a + b`` idiom (test above), not
# silently join the primary scale.
# ---------------------------------------------------------------------------


def test_point_line_composite_mark_non_first_member_raises():
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    primary = fm.Chart(df).mark_bar().encode(x="x", y="y")
    secondary = fm.Chart(df).mark_line(point=True).encode(x="x", y="y2")

    layered = LayerChart(primary, secondary, resolve={"y": "independent"})
    with pytest.raises(ValueError, match=r"LayerChart:.*position 1.*2 y-bearing layers"):
        layered.to_svg()

    # The interactive path shares the same _build_merged -- must raise too.
    with pytest.raises(ValueError, match=r"LayerChart:.*position 1.*2 y-bearing layers"):
        layered._render_interactive()


def test_point_line_composite_mark_single_layer_member_unaffected():
    """A single-layer member (no ``point=True``) with chart-level y keeps
    working -- ``_expand_layers`` already backfills the chart-level encoding
    onto the layer's own snapshot before the y-bearing count runs."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    line = fm.Chart(df).mark_line().encode(x="x", y="y2")

    layered = LayerChart(bars, line, resolve={"y": "independent"})
    svg = layered.to_svg()  # must not raise

    extents = y_axis_extents(svg)
    assert len(extents) == 2, f"expected two y-axes (primary + secondary), got {extents}"


def test_point_line_composite_mark_default_shared_y_unaffected():
    """A default/shared LayerChart whose member uses mark_line(point=True)
    is unaffected by the y-bearing count -- that guard only fires under
    resolve={"y": "independent"}. Default and explicit resolve={"y":
    "shared"} must render byte-identically (mirrors
    ``test_default_shared_y_layer_chart_unaffected`` above), and the merged
    flat path (``resolve={"y": "independent"}``'s guard) must never be
    consulted for this shared-y render."""
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    bars = fm.Chart(df).mark_bar().encode(x="x", y="y")
    point_line = fm.Chart(df).mark_line(point=True).encode(x="x", y="y")

    default_svg = LayerChart(bars, point_line).to_svg()  # must not raise
    explicit_shared_svg = LayerChart(bars, point_line, resolve={"y": "shared"}).to_svg()
    assert default_svg == explicit_shared_svg

    # The shared/default composite overlay tree renders one axis title per
    # TOP-LEVEL member chart (each member is its own overlay leaf, regardless
    # of how many internal layers a composite-mark member like
    # mark_line(point=True) carries -- confirmed above that point_line alone
    # renders exactly one "y" title). Two members (bars, point_line) -> two
    # "y" titles; no duplication introduced by the composite-mark shape.
    titles = [t for _, t in _rotated_text(default_svg)]
    assert titles == ["y", "y"], f"expected one title per member chart, got {titles}"


# ---------------------------------------------------------------------------
# GH #57: grid composites (Joint/ClusterMap/Repeat) must run the SAME
# resolve={"y": "shared"} vs. independent-y conflict guard as the generic
# HConcat/VConcat/Concat branch above. They route through
# _build_grid_tree instead of _lower_any's node.charts walk (JointChart,
# ClusterMapChart, and RepeatChart all match _lower_any's
# ``isinstance(node, (JointChart, ClusterMapChart, RepeatChart,
# LayerChart))`` branch first and delegate to their own
# ``_composite_tree``), which previously never consulted
# _contains_independent_y_layer at all -- an explicit parent
# resolve={"y": "shared"} silently rendered the dual-axis panel anyway.
# ---------------------------------------------------------------------------


def test_repeat_chart_explicit_shared_y_over_independent_y_template_raises():
    """A RepeatChart whose template is a dual-axis chart (``chart +
    SecondaryY(...)``) under an explicit ``resolve={"y": "shared"}`` hits
    the same typed conflict as the HConcat/VConcat/Concat forms above."""
    df = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0], "y2": [100.0, 200.0, 150.0, 300.0]}
    )
    template = fm.Chart(df).mark_bar().encode(x="x", y="y") + fm.SecondaryY("y2")
    repeat = RepeatChart(template, column=["p1", "p2"], resolve={"y": "shared"})
    with pytest.raises(ValueError, match=r"RepeatChart:.*'y': 'shared'.*independent-y"):
        repeat.to_svg()


def test_repeat_chart_shared_y_over_normal_template_still_lowers():
    """Negative case: resolve={"y": "shared"} over a template with no
    independent-y layer is unaffected by the GH #57 guard."""
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    template = fm.Chart(df).mark_bar().encode(x="x", y="y")
    repeat = RepeatChart(template, column=["p1", "p2"], resolve={"y": "shared"})
    svg = repeat.to_svg()  # must not raise
    assert svg


# JointChart and ClusterMapChart also route through _build_grid_tree, and
# so are covered by the same guard call inside it, but neither's public
# constructor exposes a resolve= parameter capable of requesting
# "y": "shared": JointChart's private ``_resolve`` slot only ever carries
# color/size (jointplot(hue=...)'s figure-legend wiring, never "y"), and
# ClusterMapChart has no resolve concept at all -- its panels are fixed
# heatmap/dendrogram geometry, not scale-shareable charts. Neither can
# express the request this guard checks for, so there is no reachable
# positive test for them here.
