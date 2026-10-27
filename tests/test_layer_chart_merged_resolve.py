"""LayerChart's merged (interactive) path shares scales the way ``+`` does.

``LayerChart._build_merged`` reduces its members with ``+``; that reduction
resolves one scale per channel per panel over every layer's drawn batch, the
same rule the static overlay tree applies in ``_composite_tree``. A
``resolve={"<channel>": "shared"}`` member draws with that same union: no
channel-specific handling runs on the merged path beyond the ``+`` reduction
itself.

Every case compares ``LayerChart(...)._build_merged()`` (the same chart
``.interactive()`` renders) against the plain ``chart_a + chart_b`` reduction
of the identical members, and/or the static route, since ``_build_merged``
with no title/configure is defined to BE that reduction.
"""

from __future__ import annotations

import re

import polars as pl

import ferrum as fm
from ferrum._scene import _render_scene
from tests._hue_probe import legend_labels
from tests._svg_extents import AxisExtent, y_axis_extents

_CIRCLE_RADIUS_RE = re.compile(r'<circle[^>]*\br="([\d.]+)"')
_LEGEND_TITLE_RE = re.compile(r'font-weight="600" text-anchor="start">([^<]+)</text>')


def _interactive_matches(chart_a, chart_b, layer) -> bool:
    """True when *layer*'s interactive render is byte-identical to the
    plain ``chart_a + chart_b`` chart's own interactive render."""
    scene_json, packed = layer._render_interactive()
    plus_scene_json, plus_packed = _render_scene(chart_a + chart_b)
    return scene_json == plus_scene_json and bytes(packed) == bytes(plus_packed)


def _circle_radii(svg: str) -> set[str]:
    """Every distinct circle radius drawn in *svg* (point marks and size-legend swatches)."""
    return set(_CIRCLE_RADIUS_RE.findall(svg))


def test_shared_y_over_a_summed_bar_matches_the_plus_chart_and_static_axis():
    """A bar summing several rows per category draws at its true summed
    height on the merged path, matching the static overlay tree and the
    plain ``+`` chart -- not the raw (unaggregated) per-row column max.
    """
    df = pl.DataFrame({"cat": ["a"] * 5, "v": [10.0, 20.0, 30.0, 40.0, 50.0]})
    sum_bar = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("v", aggregate="sum"))
    point = fm.Chart(df).mark_point().encode(x="cat", y="v")

    layer = fm.LayerChart(sum_bar, point, resolve={"y": "shared"})
    merged_svg = layer._build_merged().to_svg()
    plus_svg = (sum_bar + point).to_svg()
    static_svg = layer.to_svg()  # not independent-y, not faceted -> overlay tree

    assert merged_svg == plus_svg
    assert _interactive_matches(sum_bar, point, layer)
    # All three routes agree on the rendered y extent (the true summed height,
    # not the raw per-row max of 50).
    assert y_axis_extents(merged_svg) == y_axis_extents(plus_svg)
    assert y_axis_extents(merged_svg) == y_axis_extents(static_svg)
    assert y_axis_extents(merged_svg)[0].hi > 50.0


def test_shared_size_over_two_frames_matches_the_plus_chart():
    """A size channel driven by an aggregate in one member, raw in the other,
    across two different frames, draws the same point and legend radii on
    the merged path, the static overlay tree and the plain ``+`` chart.
    """
    df1 = pl.DataFrame({"g": ["a", "a", "b", "b"], "s": [1.0, 2.0, 3.0, 4.0]})
    df2 = pl.DataFrame({"g": ["a", "b"], "s": [50.0, 60.0]})
    c1 = fm.Chart(df1).mark_point().encode(x="g", y="g", size=fm.Size("s", aggregate="sum"))
    c2 = fm.Chart(df2).mark_point().encode(x="g", y="g", size="s")

    layer = fm.LayerChart(c1, c2, resolve={"size": "shared"})
    merged_svg = layer._build_merged().to_svg()
    plus_svg = (c1 + c2).to_svg()
    static_svg = layer.to_svg()  # not independent-y, not faceted -> overlay tree

    assert merged_svg == plus_svg
    assert _interactive_matches(c1, c2, layer)
    assert _circle_radii(merged_svg) == _circle_radii(plus_svg)
    assert _circle_radii(merged_svg) == _circle_radii(static_svg)


def test_shared_x_over_a_binned_histogram_and_mean_rule_matches_the_plus_chart():
    """A binned histogram beside a mean reference rule shares x the same way
    the ``+`` chart does: the merged x domain reads the histogram's bin
    edges and the rule's mean, not each layer's raw ``v`` column.
    """
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    hist = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y("v", aggregate="count"))
    mean_rule = fm.Chart(df).mark_rule().encode(x=fm.X("v", aggregate="mean"))

    layer = fm.LayerChart(hist, mean_rule, resolve={"x": "shared"})
    merged_svg = layer._build_merged().to_svg()
    plus_svg = (hist + mean_rule).to_svg()

    assert merged_svg == plus_svg
    assert _interactive_matches(hist, mean_rule, layer)


def test_shared_categorical_color_over_two_frames_matches_resolve_none():
    """Requesting resolve={"color": "shared"} explicitly over two frames
    changes nothing: the merged chart is byte-identical to the same two
    members with no ``resolve=`` at all, and the legend lists the union
    of both frames' categories."""
    df_a = pl.DataFrame({"g": ["a", "b"], "x": [1.0, 2.0], "y": [1.0, 2.0]})
    df_b = pl.DataFrame({"g": ["c", "d"], "x": [3.0, 4.0], "y": [3.0, 4.0]})
    c_a = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    c_b = fm.Chart(df_b).mark_point().encode(x="x", y="y", color="g")

    shared = fm.LayerChart(c_a, c_b, resolve={"color": "shared"})
    unset = fm.LayerChart(c_a, c_b)

    shared_svg = shared._build_merged().to_svg()
    unset_svg = unset._build_merged().to_svg()
    assert shared_svg == unset_svg
    assert legend_labels(shared_svg) == ["a", "b", "c", "d"]


def test_resolve_none_merged_matches_the_plus_chart():
    """No resolve= request at all -- the merged chart is simply ``a + b``."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "v": [50.0, 60.0, 70.0], "w": [5.0, 6.0, 7.0]})
    c1 = fm.Chart(df).mark_bar().encode(x="cat", y="v")
    c2 = fm.Chart(df).mark_line().encode(x="cat", y="w")

    layer = fm.LayerChart(c1, c2)
    assert layer._build_merged().to_svg() == (c1 + c2).to_svg()
    assert _interactive_matches(c1, c2, layer)


def test_independent_y_draws_two_own_axes_and_matches_secondary_y_sugar():
    """resolve={"y": "independent"} draws two y axes, each spanning only
    its own member's values (not the union of both): the primary bar's
    axis to the bar's own max, the secondary line's axis to the line's own
    max. The ``SecondaryY`` sugar renders byte-identical to this explicit
    ``LayerChart`` form, in both static and interactive output."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "v": [50.0, 60.0, 70.0], "w": [5.0, 6.0, 7.0]})
    base = fm.Chart(df).mark_bar().encode(x="cat", y="v")
    secondary = fm.Chart(df).mark_line().encode(x="cat", y="w")

    layer = fm.LayerChart(base, secondary, resolve={"y": "independent"})
    static_svg = layer.to_svg()
    assert y_axis_extents(static_svg) == [AxisExtent(0.0, 70.0), AxisExtent(5.0, 7.0)]

    sugared = base + fm.SecondaryY("w")
    assert static_svg == sugared.to_svg()

    packed = layer._render_interactive()[1]
    sugared_packed = _render_scene(sugared)[1]
    assert bytes(packed) == bytes(sugared_packed)


def test_static_overlay_tree_shared_y_extent_matches_resolve_none_and_the_plus_chart():
    """A non-faceted, non-independent-y LayerChart's static to_svg() renders
    through the Rust overlay tree (_composite_tree): its shared y extent
    spans the union of every member's values, equal to the ``+`` chart's
    own axis, and requesting resolve={"y": "shared"} explicitly renders the
    same bytes as leaving resolve unset."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "v": [50.0, 60.0, 70.0], "w": [5.0, 6.0, 7.0]})
    c1 = fm.Chart(df).mark_bar().encode(x="cat", y="v")
    c2 = fm.Chart(df).mark_line().encode(x="cat", y="w")

    layer = fm.LayerChart(c1, c2, resolve={"y": "shared"})
    assert not layer._renders_merged()
    static_svg = layer.to_svg()
    plus_svg = (c1 + c2).to_svg()
    assert y_axis_extents(static_svg) == y_axis_extents(plus_svg) == [AxisExtent(0.0, 70.0)]
    assert static_svg == fm.LayerChart(c1, c2).to_svg()


def test_faceted_member_shares_x_the_way_the_plus_chart_does():
    """A faceted member forces the static route through the merged chart
    too (an overlay panel cannot carry a facet grid); its shared x must be
    the ``+`` chart's facet-shared union on both static and interactive.
    """
    df_faceted = pl.DataFrame(
        {"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}
    )
    df_other = pl.DataFrame({"g": ["a", "b"], "x": [100.0, 5.0], "y": [1.0, 2.0]})
    faceted = fm.Chart(df_faceted).mark_point().encode(x="x", y="y").facet(col="g")
    other = fm.Chart(df_other).mark_point().encode(x="x", y="y")

    layer = fm.LayerChart(faceted, other, resolve={"x": "shared"})
    assert layer._renders_merged()
    assert layer.to_svg() == (faceted + other).to_svg()
    assert _interactive_matches(faceted, other, layer)


def test_independent_color_merged_draws_one_legend_static_draws_two():
    """resolve={"color": "independent"} is the one mode where the merged
    path and the static overlay tree draw different output: static draws
    two legends, titled "g" and "h", one per member; merged draws a single
    legend titled "g" that lists the union of both members' categories."""
    df_a = pl.DataFrame({"g": ["a", "b"], "x": [1.0, 2.0], "y": [1.0, 2.0]})
    df_b = pl.DataFrame({"h": ["c", "d"], "x": [3.0, 4.0], "y": [3.0, 4.0]})
    c_a = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    c_b = fm.Chart(df_b).mark_point().encode(x="x", y="y", color="h")

    layer = fm.LayerChart(c_a, c_b, resolve={"color": "independent"})
    static_svg = layer.to_svg()
    merged_svg = layer._build_merged().to_svg()
    unset_svg = fm.LayerChart(c_a, c_b)._build_merged().to_svg()

    assert _LEGEND_TITLE_RE.findall(static_svg) == ["g", "h"]
    assert _LEGEND_TITLE_RE.findall(merged_svg) == ["g"]
    assert legend_labels(merged_svg) == ["a", "b", "c", "d"]
    assert merged_svg == unset_svg
