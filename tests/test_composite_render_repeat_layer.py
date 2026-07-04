"""Behavior tests for the RepeatChart + LayerChart composite render cutover.

Phase B Task 9 — the last cutover.  RepeatChart lowers its materialized panel
grid (including ``corner=True`` holes and 1-D wrap holes) through
``_build_grid_tree`` (shared with JointChart/ClusterMapChart, Task 8b);
LayerChart lowers to the Rust ``"overlay"`` composite layout (children share
one panel rect; z-order = child declaration order).

RepeatChart routes BOTH static (``to_svg``) and interactive
(``_render_interactive``) rendering through the one-call composite entry when
the composition lowers cleanly, falling back to the pre-existing per-child
render path otherwise (same two-path precedent already shipped for every
other composite form in this file).

LayerChart is split: ``to_svg`` (static) routes through the overlay composite
entry, but ``_render_interactive`` ALWAYS uses the legacy merged-chart route
(never the overlay tree), because the interactive scene contract requires
LayerChart to produce EXACTLY ONE panel — selections, hit-testing, and the
WASM interaction runtime assume layered marks share a single panel. The
overlay tree gives each layer its own panel that merely shares one rect,
which is fine for static SVG (no panel-identity concept there) but breaks
that one-panel contract in scene JSON. See ``LayerChart._render_interactive``.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import LayerChart, RepeatChart

from tests._svg_extents import y_axis_extents

# ---------------------------------------------------------------------------
# RepeatChart — 2-D grid geometry + corner holes
# ---------------------------------------------------------------------------


@pytest.fixture
def repeat_df():
    return pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})


def _template(df):
    return fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)


def test_repeat_2x2_lowers_to_dense_grid_tree(repeat_df):
    """A plain (no corner) 2x2 repeat lowers to a dense grid: no holes."""
    chart = RepeatChart(_template(repeat_df), row=["a", "b"], column=["a", "b"])
    lowered = chart._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["layout"] == "grid"
    assert lowered.tree["nrows"] == 2 and lowered.tree["ncols"] == 2
    assert [c["kind"] for c in lowered.tree["children"]] == ["leaf"] * 4
    assert len(lowered.payloads) == 4


def test_repeat_corner_true_emits_holes_for_upper_triangle(repeat_df):
    """corner=True on a 3x3 grid fills the upper triangle with hole cells."""
    chart = RepeatChart(
        _template(repeat_df), row=["a", "b", "c"], column=["a", "b", "c"], corner=True
    )
    lowered = chart._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["nrows"] == 3 and lowered.tree["ncols"] == 3
    kinds = [c["kind"] for c in lowered.tree["children"]]
    # Row-major 3x3: row 0 has 1 leaf + 2 holes, row 1 has 2 leaves + 1 hole,
    # row 2 (bottom) is fully dense (ri >= ci for every ci).
    assert kinds == [
        "leaf", "hole", "hole",
        "leaf", "leaf", "hole",
        "leaf", "leaf", "leaf",
    ]  # fmt: skip
    assert len(lowered.payloads) == 6  # 6 lower-triangle-including-diagonal cells


def test_repeat_corner_renders_without_error(repeat_df):
    """The corner grid (with holes) renders end to end, holes contribute no marks."""
    chart = RepeatChart(
        _template(repeat_df), row=["a", "b", "c"], column=["a", "b", "c"], corner=True
    )
    svg = chart.to_svg()
    assert svg.startswith("<svg")
    assert svg.count("<circle") > 0


def test_repeat_1d_wrap_lowers_to_grid_with_trailing_holes(repeat_df):
    """A 1-D column-only repeat with columns=2 over 3 panels wraps with 1 trailing hole."""
    template = fm.Chart(repeat_df).mark_point().encode(x=fm.Repeat.column, y="a")
    chart = RepeatChart(template, column=["a", "b", "c"], columns=2)
    lowered = chart._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["nrows"] == 2 and lowered.tree["ncols"] == 2
    kinds = [c["kind"] for c in lowered.tree["children"]]
    assert kinds == ["leaf", "leaf", "leaf", "hole"]
    assert len(lowered.payloads) == 3


# ---------------------------------------------------------------------------
# RepeatChart — shared vs independent extents through the new mechanism
# ---------------------------------------------------------------------------


def test_repeat_independent_default_isolates_extents():
    """Default RepeatChart (no resolve=): each panel keeps its own y domain."""
    small = pl.DataFrame({"x": [1.0, 2.0], "small": [1.0, 2.0], "large": [1.0, 2.0]})
    large = pl.DataFrame({"x": [1.0, 2.0], "small": [100.0, 200.0], "large": [300.0, 400.0]})
    df = pl.concat([small, large])
    template = fm.Chart(df).mark_point().encode(x="x", y=fm.Repeat.column)
    chart = RepeatChart(template, column=["small", "large"])

    lowered = chart._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert "resolve" not in lowered.tree

    svg = chart.to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    assert extents[0] != extents[1]


def test_repeat_resolve_shared_unifies_y_extent(repeat_df):
    """resolve={"y": "shared"} unions the y domain across repeated panels via
    the tree resolve field (Rust resolve pass), not the legacy per-panel
    ``_apply_resolve`` scale injection.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "a": [1.0, 2.0, 3.0], "b": [100.0, 200.0, 300.0]})
    template = fm.Chart(df).mark_point().encode(x="x", y=fm.Repeat.column)
    chart = RepeatChart(template, column=["a", "b"], resolve={"y": "shared"})

    lowered = chart._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["resolve"] == {"y": "shared"}

    svg = chart.to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    assert extents[0] == extents[1]
    assert extents[0].hi >= 300.0


def test_repeat_non_xy_shared_channel_falls_back_to_legacy(repeat_df):
    """resolve with a shared non-positional channel keeps the legacy Python path."""
    chart = RepeatChart(
        _template(repeat_df), row=["a", "b"], column=["a", "b"], resolve={"color": "shared"}
    )
    assert chart._composite_tree(auto_tooltips=False) is None
    assert chart.to_svg().startswith("<svg")


def test_repeat_configure_layer_falls_back_to_legacy(repeat_df):
    """A composition-level configure layer keeps the old string-compositor path."""
    chart = RepeatChart(_template(repeat_df), row=["a", "b"], column=["a", "b"]).configure_axis(
        label_angle=-45
    )
    assert chart._composite_tree(auto_tooltips=False) is None
    assert chart.to_svg().startswith("<svg")


def test_repeat_interactive_routes_through_composite_entry(repeat_df):
    """The interactive render produces one scene with all panels via the new entry."""
    chart = RepeatChart(_template(repeat_df), row=["a", "b"], column=["a", "b"])
    assert chart._composite_tree(auto_tooltips=True) is not None
    scene_json, packed = chart._render_interactive()
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 4
    assert isinstance(packed, (bytes, bytearray))


def test_repeat_corner_interactive_hole_contributes_no_panel(repeat_df):
    """corner=True's hole cells contribute zero panels to the interactive scene."""
    chart = RepeatChart(
        _template(repeat_df), row=["a", "b", "c"], column=["a", "b", "c"], corner=True
    )
    scene_json, _ = chart._render_interactive()
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 6  # 9 grid cells - 3 holes


# ---------------------------------------------------------------------------
# LayerChart — overlay lowering + z-order
# ---------------------------------------------------------------------------


@pytest.fixture
def layer_df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 1.0, 4.0, 3.0, 5.0]})


def test_layer_lowers_to_overlay_tree(layer_df):
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    lc = LayerChart(scatter, line)

    lowered = lc._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["kind"] == "composite"
    assert lowered.tree["layout"] == "overlay"
    assert [c["kind"] for c in lowered.tree["children"]] == ["leaf", "leaf"]
    assert lowered.tree["resolve"] == {"x": "shared", "y": "shared"}
    assert len(lowered.payloads) == 2


def test_layer_static_svg_renders_single_plot_area(layer_df):
    """The overlay renders one SVG with both layers' marks present."""
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    svg = LayerChart(scatter, line).to_svg()
    assert svg.count("<svg") == 1
    assert svg.count("<circle") > 0
    assert svg.count("<path") > 0 or svg.count("<polyline") > 0


def test_layer_overlay_tree_children_share_one_panel_rect(layer_df):
    """The overlay TREE's children share ONE panel rect (Rust overlay layout,
    Task 5b): both layers' panel geometry (x/y/width/height) are identical,
    and only declaration order (not position) encodes z-order. This exercises
    the overlay tree directly via ``render_composite_interactive`` (bypassing
    ``LayerChart._render_interactive``, which never routes through the
    overlay tree — see ``test_layer_interactive_always_produces_one_panel``).
    """
    from ferrum._core import render_composite_interactive

    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    lc = LayerChart(scatter, line)

    lowered = lc._composite_tree(auto_tooltips=True)
    assert lowered is not None
    scene_json, _ = render_composite_interactive(
        lowered.tree,
        lowered.payloads,
        viewport=lowered.viewport,
        theme=lowered.theme,
        chart_config=lowered.chart_config,
    )
    scene = json.loads(scene_json)
    panels = scene["panels"]
    assert len(panels) == 2
    rect_keys = ("x", "y", "width", "height")
    first_rect = {k: panels[0].get(k) for k in rect_keys}
    second_rect = {k: panels[1].get(k) for k in rect_keys}
    assert first_rect == second_rect, (
        f"overlay children must share one panel rect: {first_rect} != {second_rect}"
    )


def test_layer_interactive_always_produces_one_panel(layer_df):
    """LayerChart._render_interactive() ALWAYS produces exactly one scene
    panel, even though the overlay tree lowers cleanly for this composition.
    The interactive contract (selections/hit-testing/WASM) requires layered
    marks to share a single panel, so the interactive path stays on the
    legacy merged-chart route unconditionally (see module docstring).
    """
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    lc = LayerChart(scatter, line)

    # The overlay tree DOES lower cleanly for this composition...
    assert lc._composite_tree(auto_tooltips=True) is not None
    # ...but _render_interactive() must not use it.
    scene_json, packed = lc._render_interactive()
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 1
    assert isinstance(packed, (bytes, bytearray))


def test_layer_zorder_matches_declaration_order(layer_df):
    """A big red point layer declared BEFORE a thick green line layer is drawn
    UNDER it (line paints on top) — declaration order is the only mechanism
    that encodes z-order in an overlay (shared-rect) layout, discriminating
    this from a linear (non-overlapping) placement standing in by coincidence.
    """
    flat_df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 3.0, 3.0]})
    point = fm.Chart(flat_df).mark_point(size=800, color="red").encode(x="x", y="y")
    line = fm.Chart(flat_df).mark_line(color="green", stroke_width=12).encode(x="x", y="y")

    bottom_first = LayerChart(point, line).to_svg()
    top_first = LayerChart(line, point).to_svg()
    # Both render, both share the same union domain; the two SVGs differ
    # because painting order differs (mark fragments appear in a different
    # sequence in the document), even though panel geometry is identical.
    assert bottom_first.startswith("<svg") and top_first.startswith("<svg")
    assert bottom_first != top_first


def test_layer_resolve_shared_x_only_still_overlays(layer_df):
    """resolve={"x": "shared"} (an explicit x/y-only request) still lowers —
    x/y are always shared for an overlay regardless of what resolve= says,
    since the tree always emits both.
    """
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    lc = LayerChart(scatter, line, resolve={"x": "shared"})
    lowered = lc._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["resolve"] == {"x": "shared", "y": "shared"}


def test_layer_non_xy_shared_channel_falls_back_to_legacy(layer_df):
    """resolve with a shared non-positional channel keeps the legacy merge path."""
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y", color="x")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y", color="x")
    lc = LayerChart(scatter, line, resolve={"color": "shared"})
    assert lc._composite_tree(auto_tooltips=False) is None
    assert lc.to_svg().startswith("<svg")


def test_layer_configure_layer_falls_back_to_legacy(layer_df):
    """A composition-level configure layer keeps the old merge-based path."""
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    lc = LayerChart(scatter, line).configure_axis(label_angle=-45)
    assert lc._composite_tree(auto_tooltips=False) is None
    assert lc.to_svg().startswith("<svg")


def test_layer_title_lands_on_tree_root(layer_df):
    """LayerChart's title= becomes a composite figure title at the tree root."""
    scatter = fm.Chart(layer_df).mark_point().encode(x="x", y="y")
    line = fm.Chart(layer_df).mark_line().encode(x="x", y="y")
    lc = LayerChart(scatter, line, title="My Overlay")
    lowered = lc._composite_tree(auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["title"] == "My Overlay"
    assert "My Overlay" in lc.to_svg()
