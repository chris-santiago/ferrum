"""A data-space ``Jitter`` or continuous ``Dodge`` draws every point: the
axis spans the positions the adjustment draws, as ``Stack``'s does. An
explicit caller domain still clips, and the ordinal (pixel-offset)
branches keep their digests.
"""

from __future__ import annotations

import hashlib
import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum import Chart
from ferrum._layer import _Layer
from tests._svg_extents import panel_x_ladder, value_axis_ladder, value_for_pixel
from tests._svg_rects import mark_groups, mark_groups_by_frame, svg_elements


def _points(svg: str) -> list[dict[str, str]]:
    """Every rendered data-point circle, excluding legend swatches (fixed ``r="4"``)."""
    return svg_elements(svg, "circle", lambda a: a.get("r") != "4")


def _warnings_of(render) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render()
        return [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# Data-space Jitter on x: every seed, seed=None, axis=y/both, Int64.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_jitter_x_all_points_draw_every_seed(seed):
    """Every seed must draw both points: a jittered x landing outside the
    unadjusted domain must not be silently skipped."""
    df = pl.DataFrame({"g": ["a", "a"], "v": [40.0, 60.0]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=seed)).encode(x="v", y="g")
    assert len(_points(chart.to_svg())) == 2


def test_jitter_x_seed_none_all_points_draw():
    """``seed=None`` derives a per-row seed from the data (still
    deterministic) -- it must widen the domain the same as an explicit seed."""
    df = pl.DataFrame({"g": ["a", "a"], "v": [40.0, 60.0]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=None)).encode(x="v", y="g")
    assert len(_points(chart.to_svg())) == 2


@pytest.mark.parametrize("axis", ["y", "both"])
def test_jitter_continuous_axes_all_points_draw(axis):
    """``axis="y"``/``"both"`` widen the y axis (and x too for ``"both"``) the
    same way ``axis="x"`` widens x."""
    df = pl.DataFrame({"v": [40.0, 60.0], "w": [1.0, 2.0]})
    chart = (
        Chart(df).mark_point(position=fm.Jitter(width=0.5, axis=axis, seed=1)).encode(x="w", y="v")
    )
    assert len(_points(chart.to_svg())) == 2


def test_jitter_int64_value_column_all_points_draw():
    df = pl.DataFrame({"g": ["a", "a"], "v": [40, 60]})  # Int64, read through col_as_f64
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="g")
    assert len(_points(chart.to_svg())) == 2


# ---------------------------------------------------------------------------
# x="v", y="v" with x-only jitter: both axes cover the drawn positions
# (whether jittering x also moves y is a separate, open issue and is not
# asserted here).
# ---------------------------------------------------------------------------


def test_jitter_shared_field_both_axes_draw():
    df = pl.DataFrame({"v": [40.0, 60.0]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="v")
    assert len(_points(chart.to_svg())) == 2


# ---------------------------------------------------------------------------
# Continuous Dodge(by="c"), with and without color="c".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_color", [True, False])
def test_dodge_continuous_all_points_draw(with_color):
    df = pl.DataFrame(
        {"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
    )
    encode = {"x": "x", "y": "y"}
    if with_color:
        encode["color"] = "c"
    chart = Chart(df).mark_point(position=fm.Dodge(by="c")).encode(**encode)
    assert len(_points(chart.to_svg())) == 4


def test_dodge_continuous_positions_unaffected_under_an_explicit_domain():
    """Only the AUTO domain widens for a continuous `Dodge`: under an
    explicit (already-wide) domain the pixel mapping is unaffected, so
    every dodged point's data-space position matches `apply_dodge`'s own
    geometry, reused here rather than reimplemented."""
    df = pl.DataFrame(
        {"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
    )
    scale = fm.LinearScale(domain=(0.0, 3.0))
    chart = (
        Chart(df)
        .mark_point(position=fm.Dodge(by="c"))
        .encode(x=fm.X("x", scale=scale), y="y", color="c")
    )
    svg = chart.to_svg()
    frame, content = mark_groups_by_frame(svg)[0]
    x0, _, x1, _ = frame
    values = sorted(
        0.0 + (float(p["cx"]) - x0) / (x1 - x0) * 3.0
        for p in svg_elements(content, "circle", lambda a: a.get("r") != "4")
    )
    assert len(values) == 4
    for value, expected in zip(values, [0.775, 1.225, 1.775, 2.225]):
        assert value == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# A Dodge group absent at the axis's extreme x: the resolved domain's low
# and high equal the leftmost and rightmost drawn circle exactly, not the
# unadjusted row that never draws its own mark there. `p` absent at the min
# x and alone at the max x (`q` absent there) is the one fixture that
# discriminates "domain spans the drawn rows" from "domain spans every row,
# drawn or not": the former gives the drawn extremes below, the latter
# would give the unadjusted row positions (1.0 and 3.0) instead.
# ---------------------------------------------------------------------------


def test_dodge_group_absent_at_extreme_x_matches_the_drawn_extent():
    """The domain spans exactly the positions the dodge actually draws
    (1.225 to 2.775 here, by `apply_dodge`'s own geometry), not the absent
    group's own unadjusted x (1.0 and 3.0, where nothing draws): a chart
    drawing those exact dodged x positions directly, with no adjustment
    and no absent group to inflate the domain, renders the identical
    frame and the identical circle positions."""
    df = pl.DataFrame(
        {"x": [2.0, 1.0, 3.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
    )
    chart = Chart(df).mark_point(position=fm.Dodge(by="c")).encode(x="x", y="y", color="c")
    svg = chart.to_svg()
    assert len(_points(svg)) == 4

    df_direct = pl.DataFrame(
        {"x": [1.775, 1.225, 2.775, 2.225], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
    )
    chart_direct = Chart(df_direct).mark_point().encode(x="x", y="y", color="c")
    svg_direct = chart_direct.to_svg()

    assert mark_groups_by_frame(svg)[0][0] == mark_groups_by_frame(svg_direct)[0][0]
    assert sorted(p["cx"] for p in _points(svg)) == sorted(p["cx"] for p in _points(svg_direct))


def test_dodge_group_absent_at_extreme_x_follows_a_reordered_color_domain():
    """The same fixture under a reversed color domain swaps which group's
    offset sits at each x (0.775 to 3.225) -- the preview must read the
    color scale the draw actually uses, not a first-appearance order of
    its own."""
    df = pl.DataFrame(
        {"x": [2.0, 1.0, 3.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
    )
    chart = (
        Chart(df)
        .mark_point(position=fm.Dodge(by="c"))
        .encode(x="x", y="y", color=fm.Color("c", sort="descending"))
    )
    svg = chart.to_svg()
    assert len(_points(svg)) == 4

    df_direct = pl.DataFrame(
        {"x": [2.225, 0.775, 3.225, 1.775], "y": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "p", "q"]}
    )
    chart_direct = (
        Chart(df_direct).mark_point().encode(x="x", y="y", color=fm.Color("c", sort="descending"))
    )
    svg_direct = chart_direct.to_svg()

    assert mark_groups_by_frame(svg)[0][0] == mark_groups_by_frame(svg_direct)[0][0]
    assert sorted(p["cx"] for p in _points(svg)) == sorted(p["cx"] for p in _points(svg_direct))


# ---------------------------------------------------------------------------
# A faceted Dodge whose first panel has only one `by=` group (a no-op: its
# x column keeps its input dtype) beside a panel that actually dodges (its
# x column retypes to Float64): the no-op panel's dtype must never
# truncate the dodging panel's previewed positions, on every numeric input
# dtype and both panel orders.
# ---------------------------------------------------------------------------


def _dodge_schema_fixture(order: str, dtype) -> pl.DataFrame:
    """Two facet panels sharing one `by=c` column: one panel has a single
    group (`p` alone, a Dodge no-op), the other has two (`p`/`q`, an
    actual dodge). `order="AB"` puts the no-op panel first; `order="BA"`
    puts the dodging panel first -- the same two row sets, reordered."""
    if order == "AB":
        rows = {
            "x": [1, 2, 1, 2, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": ["p", "p", "p", "q", "p"],
            "f": ["A", "A", "B", "B", "B"],
        }
    else:
        rows = {
            "x": [1, 2, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": ["p", "q", "p", "p", "p"],
            "f": ["A", "A", "A", "B", "B"],
        }
    return pl.DataFrame(rows).with_columns(pl.col("x").cast(dtype))


def _dodge_flat_route(df: pl.DataFrame) -> "Chart":
    return (
        Chart(df)
        .mark_point(position=fm.Dodge(by="c"))
        .encode(x="x", y="y", color="c")
        .facet(col="f")
    )


def _dodge_then_rule_route(df: pl.DataFrame) -> "Chart":
    dodge = Chart(df).mark_point(position=fm.Dodge(by="c")).encode(x="x", y="y", color="c")
    rule = Chart(df).mark_rule().encode(x="x")
    return (dodge + rule).facet(col="f")


def _rule_then_dodge_route(df: pl.DataFrame) -> "Chart":
    dodge = Chart(df).mark_point(position=fm.Dodge(by="c")).encode(x="x", y="y", color="c")
    rule = Chart(df).mark_rule().encode(x="x")
    return (rule + dodge).facet(col="f")


def _dodge_independent_x_route(df: pl.DataFrame) -> "Chart":
    return (
        Chart(df)
        .mark_point(position=fm.Dodge(by="c"))
        .encode(x="x", y="y", color="c")
        .facet(col="f")
        .share_scale(x="independent")
    )


_DODGE_SCHEMA_ROUTES = {
    "flat": _dodge_flat_route,
    "dodge_then_rule": _dodge_then_rule_route,
    "rule_then_dodge": _rule_then_dodge_route,
    "independent_x": _dodge_independent_x_route,
}


@pytest.mark.parametrize("route_name", sorted(_DODGE_SCHEMA_ROUTES))
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize(
    "dtype",
    [pl.Int64, pl.Int32, pl.UInt8, pl.Float32, pl.Float64],
    ids=["int64", "int32", "uint8", "float32", "float64"],
)
def test_faceted_dodge_no_op_panel_schema_does_not_truncate_the_other_panel(
    dtype, order, route_name
):
    build = _DODGE_SCHEMA_ROUTES[route_name]
    df = _dodge_schema_fixture(order, dtype)
    df_f64 = _dodge_schema_fixture(order, pl.Float64)
    chart = build(df)
    chart_f64 = build(df_f64)

    result: dict[str, str] = {}

    def render():
        result["svg"] = chart.to_svg()

    msgs = _warnings_of(render)
    assert not msgs, msgs
    svg = result["svg"]
    svg_f64 = chart_f64.to_svg()

    assert len(_points(svg)) == 5

    panels = mark_groups_by_frame(svg)
    panels_f64 = mark_groups_by_frame(svg_f64)
    assert len(panels) == len(panels_f64) == 2
    for (frame, content), (frame_f64, content_f64) in zip(panels, panels_f64):
        assert frame == frame_f64
        cx = sorted(p["cx"] for p in svg_elements(content, "circle", lambda a: a.get("r") != "4"))
        cx_f64 = sorted(
            p["cx"] for p in svg_elements(content_f64, "circle", lambda a: a.get("r") != "4")
        )
        assert cx == cx_f64


# ---------------------------------------------------------------------------
# Faceted seeded jitter, shared and independent x: every point draws in
# every panel, with the same per-row jittered value as an unfaceted render
# of that panel's own rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("share_mode", ["shared", "independent"])
def test_faceted_jitter_every_panel_draws_all_points(share_mode):
    df = pl.DataFrame({"v": [40.0, 60.0, 140.0, 160.0], "panel": ["p1", "p1", "p2", "p2"]})
    chart = (
        Chart(df)
        .mark_point(position=fm.Jitter(width=0.5, seed=1))
        .encode(x="v", y="panel")
        .facet(col="panel")
        .share_scale(x=share_mode)
    )
    svg = chart.to_svg()
    groups = mark_groups(svg)
    assert len(groups) == 2
    for group in groups:
        assert len(svg_elements(group, "circle", lambda a: a.get("r") != "4")) == 2


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("share_mode", ["shared", "independent"])
def test_faceted_jitter_positions_match_an_unfaceted_render_of_the_same_rows(seed, share_mode):
    """Every panel's own jittered positions, not just the first panel's,
    match an unfaceted render of that panel's own rows -- the draw and the
    preview must agree on BOTH panels' per-row seed, not only one."""
    df = pl.DataFrame({"v": [40.0, 60.0, 140.0, 160.0], "panel": ["p1", "p1", "p2", "p2"]})
    chart = (
        Chart(df)
        .mark_point(position=fm.Jitter(width=0.5, seed=seed))
        .encode(x="v", y="panel")
        .facet(col="panel")
        .share_scale(x=share_mode)
    )
    svg = chart.to_svg()
    frames = mark_groups_by_frame(svg)
    assert len(frames) == 2

    panel_rows = {"p1": [40.0, 60.0], "p2": [140.0, 160.0]}
    for (frame, content), (panel, rows) in zip(frames, panel_rows.items()):
        ladder = panel_x_ladder(svg, frame)
        faceted_values = sorted(
            value_for_pixel(ladder, float(p["cx"]))
            for p in svg_elements(content, "circle", lambda a: a.get("r") != "4")
        )

        df_panel = pl.DataFrame({"v": rows, "panel": [panel] * len(rows)})
        chart_panel = (
            Chart(df_panel)
            .mark_point(position=fm.Jitter(width=0.5, seed=seed))
            .encode(x="v", y="panel")
        )
        svg_panel = chart_panel.to_svg()
        unfaceted_ladder = value_axis_ladder(svg_panel, axis="x")
        _, content_panel = mark_groups_by_frame(svg_panel)[0]
        unfaceted_values = sorted(
            value_for_pixel(unfaceted_ladder, float(p["cx"]))
            for p in svg_elements(content_panel, "circle", lambda a: a.get("r") != "4")
        )

        assert len(faceted_values) == len(unfaceted_values) == 2
        span = max(unfaceted_ladder) - min(unfaceted_ladder)
        tol = max(span * 1e-4, 1e-6)
        for faceted, unfaceted in zip(faceted_values, unfaceted_values):
            assert faceted == pytest.approx(unfaceted, abs=tol)


# ---------------------------------------------------------------------------
# A jittered layer as layer 1 of `+`, `Chart.layer`, `LayerChart`; a
# composite hconcat with a shared x.
# ---------------------------------------------------------------------------


def test_jitter_as_second_layer_of_plus_all_points_draw():
    df = pl.DataFrame({"v": [40.0, 60.0]})
    base = Chart(df).mark_line().encode(x="v", y="v")
    jittered = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="v")
    assert len(_points((base + jittered).to_svg())) == 2


def test_jitter_as_second_layer_of_chart_layer_all_points_draw():
    df = pl.DataFrame({"v": [40.0, 60.0]})
    base = Chart(df).mark_line().encode(x="v", y="v")
    # `fm.Layer`'s public `mark_kwargs` door refuses `position` (it belongs
    # to a mark method, not a layer's own style) -- the internal `_Layer`
    # it and every other `Chart.layer()` branch ultimately builds carries a
    # dedicated `position` field, which is what this exercises.
    jittered_layer = _Layer(
        mark="point",
        encoding={"x": "v", "y": "v"},
        position=fm.Jitter(width=0.5, seed=1),
        chart_data=True,
    )
    assert len(_points(base.layer(jittered_layer).to_svg())) == 2


def test_jitter_as_second_layer_of_layerchart_all_points_draw():
    df = pl.DataFrame({"v": [40.0, 60.0]})
    base = Chart(df).mark_line().encode(x="v", y="v")
    jittered = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="v")
    assert len(_points(fm.LayerChart(base, jittered).to_svg())) == 2


def test_jitter_in_hconcat_with_shared_x_all_points_draw():
    df = pl.DataFrame({"v": [40.0, 60.0]})
    left = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="v")
    right = Chart(df).mark_line().encode(x="v", y="v")
    hc = fm.hconcat(left, right, resolve={"x": "shared"})
    assert len(_points(mark_groups(hc.to_svg())[0])) == 2


# ---------------------------------------------------------------------------
# CoordFlip with jitter on the value axis.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_coord_flip_jitter_all_points_draw(seed):
    df = pl.DataFrame({"v": [40.0, 60.0], "w": [1.0, 2.0]})
    chart = (
        Chart(df)
        .mark_point(position=fm.Jitter(width=0.5, axis="x", seed=seed))
        .encode(x="v", y="w")
        .coord(fm.CoordFlip())
    )
    assert len(_points(chart.to_svg())) == 2


# ---------------------------------------------------------------------------
# Stack (flat, layered, composite, faceted), span-branch dodge (displot
# multiple="dodge"), ordinal jitter/dodge (catplot strip): all unaffected,
# pinned byte-identical.
# ---------------------------------------------------------------------------


def _digest(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


def test_flat_stack_svg_digest_is_pinned():
    df = pl.DataFrame(
        {"x": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0], "g": ["p", "q", "p", "q"]}
    )
    chart = Chart(df).mark_bar(position=fm.Stack()).encode(x="x", y="y", color="g")
    assert (
        _digest(chart.to_svg())
        == "61362a0fc608d20bf28a53657d92157bc48e4d04c300f34da2ab1b03a0328c15"
    )


def test_layered_stack_svg_digest_is_pinned():
    df = pl.DataFrame(
        {"x": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0], "g": ["p", "q", "p", "q"]}
    )
    chart = Chart(df).mark_bar(position=fm.Stack()).encode(x="x", y="y", color="g") + Chart(
        df
    ).mark_line().encode(x="x", y="y")
    assert (
        _digest(chart.to_svg())
        == "1aa7f3e551c46222dde453ef18e32df788492309306b596469deca3c87d34b32"
    )


def test_composite_stack_svg_digest_is_pinned():
    df = pl.DataFrame(
        {"x": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0], "g": ["p", "q", "p", "q"]}
    )
    chart = Chart(df).mark_bar(position=fm.Stack()).encode(x="x", y="y", color="g")
    hc = fm.hconcat(chart, chart)
    assert (
        _digest(hc.to_svg()) == "b39b8ae2eeb1f55da00272b7e49ce9126bbe2a1ca32a7c7f4a942ee4490a5fa5"
    )


def test_faceted_stack_svg_digest_is_pinned():
    df = pl.DataFrame(
        {"x": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0], "g": ["p", "q", "p", "q"]}
    )
    chart = Chart(df).mark_bar(position=fm.Stack()).encode(x="x", y="y", color="g").facet(col="g")
    assert (
        _digest(chart.to_svg())
        == "7ed39dd30997acfebfa3e5b3a4111963a403643875e4bb2961871c2f08a2a57c"
    )


def test_displot_span_dodge_svg_digest_is_pinned():
    df = pl.DataFrame({"x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "g": ["p", "q", "p", "q", "p", "q"]})
    chart = fm.displot(df, x="x", hue="g", multiple="dodge", kind="hist")
    assert (
        _digest(chart.to_svg())
        == "6b3c91a95dc190b4f1c88b1500d92d2b4db9acd15ef4915864dd359098362546"
    )


def test_catplot_ordinal_strip_jitter_svg_digest_is_pinned():
    df = pl.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 2.0, 3.0, 4.0]})
    chart = fm.catplot(df, x="cat", y="val", kind="strip", jitter=True, seed=0)
    assert (
        _digest(chart.to_svg())
        == "217451f82623d32e0dabae59396b099869bbb992822be6a94030f09fb43263a2"
    )


# ---------------------------------------------------------------------------
# An explicit domain excluding a jittered point still clips it: the auto
# domain widens to the drawn positions, but a caller-declared domain is
# never second-guessed.
# ---------------------------------------------------------------------------


def test_explicit_domain_still_excludes_a_jittered_point():
    df = pl.DataFrame({"v": [40.0, 60.0]})
    scale = fm.LinearScale(domain=(39.5, 40.5))
    chart = (
        Chart(df)
        .mark_point(position=fm.Jitter(width=0.5, seed=0))
        .encode(x=fm.X("v", scale=scale), y="v")
    )
    assert len(_points(chart.to_svg())) == 1


# ---------------------------------------------------------------------------
# 0 rows draws the frame and warns; 1 row draws; a null/NaN value row is
# unchanged.
# ---------------------------------------------------------------------------


def test_jitter_zero_rows_draws_empty_frame_and_warns():
    df = pl.DataFrame({"v": pl.Series([], dtype=pl.Float64)})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=0)).encode(x="v", y="v")
    svg = None

    def render():
        nonlocal svg
        svg = chart.to_svg()

    msgs = _warnings_of(render)
    assert svg is not None, "render() must set svg"
    assert len(_points(svg)) == 0
    assert any("drawn empty" in m for m in msgs)


def test_jitter_one_row_draws():
    df = pl.DataFrame({"v": [40.0]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=0)).encode(x="v", y="v")
    assert len(_points(chart.to_svg())) == 1


def test_jitter_null_value_row_unchanged():
    df = pl.DataFrame({"v": [40.0, None]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=0)).encode(x="v", y="v")
    assert len(_points(chart.to_svg())) == 1


# ---------------------------------------------------------------------------
# .interactive() carries the same (widened) domain as static, and places
# every mark node inside the panel it belongs to.
# ---------------------------------------------------------------------------


def test_interactive_scene_draws_the_same_points_as_static():
    import json

    df = pl.DataFrame({"v": [40.0, 60.0]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="v")
    static_count = len(_points(chart.to_svg()))

    scene = json.loads(chart.interactive().scene_json)
    interactive_count = len(scene["panels"][0]["marks"][0]["nodes"])

    assert static_count == 2
    assert interactive_count == static_count


def test_interactive_scene_domain_matches_static_and_marks_lie_in_the_plot_area():
    """The interactive scene's own tick ladder (at the zoom level containing
    1.0, its un-zoomed "home" view) names the same axis domain the static
    SVG's ticks do, and every drawn mark node's pixel sits inside the
    panel's own plot area -- a widened-but-wrong domain (e.g. one computed
    in the wrong space) would place a node outside it even while both tick
    ladders happened to agree."""
    import json

    df = pl.DataFrame({"v": [40.0, 60.0]})
    chart = Chart(df).mark_point(position=fm.Jitter(width=0.5, seed=1)).encode(x="v", y="v")
    svg = chart.to_svg()
    static_x = value_axis_ladder(svg, axis="x")
    static_y = value_axis_ladder(svg, axis="y")

    scene = json.loads(chart.interactive().scene_json)
    panel = scene["panels"][0]
    panel_tick_levels = next(
        tl for tl in scene["interaction"]["tick_levels"] if tl["panel_id"] == panel["id"]
    )

    def home_ticks(levels):
        home = next(lv for lv in levels if lv["min_zoom"] <= 1.0 <= lv["max_zoom"])
        return {t["value"]: t["pixel"] for t in home["ticks"]}

    assert home_ticks(panel_tick_levels["x_levels"]) == pytest.approx(static_x, abs=1e-2)
    assert home_ticks(panel_tick_levels["y_levels"]) == pytest.approx(static_y, abs=1e-2)

    plot_area = panel["plot_area"]
    x0, x1 = plot_area["x"], plot_area["x"] + plot_area["w"]
    y0, y1 = plot_area["y"], plot_area["y"] + plot_area["h"]
    nodes = panel["marks"][0]["nodes"]
    assert len(nodes) == 2
    for node in nodes:
        assert x0 <= node["cx"] <= x1
        assert y0 <= node["cy"] <= y1
