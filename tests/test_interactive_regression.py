"""Regression tests for interactive renderer fixes.

Covers:
- Multi-field tooltip serialization → scene JSON (tooltip_fields)
- Computed axis domains injected into scene coord for JS zoom
- merge_scene_graphs _offset_nodes using correct "type" tag
- Single-field Tooltip backward compat
- interaction_config includes selections array for JS click handler
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import render_interactive
from ferrum.selection import selection_point, value


# ── helpers ──────────────────────────────────────────────────────────────────

def _render(chart: fm.Chart) -> dict:
    spec, data, viewport, theme = chart._render_inputs()
    return json.loads(render_interactive(spec, data, viewport=viewport, theme=theme))


def _simple_scatter(tooltip=None, **kwargs) -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "label": ["a", "b", "c"]})
    enc = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    if tooltip is not None:
        enc = enc.encode(tooltip=tooltip)
    return enc.properties(width=300, height=200, **kwargs)


# ── multi-field tooltip ───────────────────────────────────────────────────────

def test_multi_field_tooltip_three_fields():
    scene = _render(_simple_scatter(tooltip=fm.Tooltip("x", "y", "label")))
    batch = scene["panels"][0]["marks"][0]
    assert batch["tooltips"] is not None, "tooltips missing from batch"
    first = batch["tooltips"][0]
    assert len(first["fields"]) == 3
    field_names = {f["name"] for f in first["fields"]}
    assert field_names == {"x", "y", "label"}


def test_multi_field_tooltip_with_tooltip_field_format():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (fm.Chart(df).mark_point()
             .encode(x="x:Q", y="y:Q",
                     tooltip=fm.Tooltip(fm.TooltipField("x", format=".2f"), "y"))
             .properties(width=300, height=200))
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert batch["tooltips"] is not None
    field_names = {f["name"] for f in batch["tooltips"][0]["fields"]}
    assert field_names == {"x", "y"}


def test_single_field_tooltip_backward_compat():
    scene = _render(_simple_scatter(tooltip=fm.Tooltip("x")))
    batch = scene["panels"][0]["marks"][0]
    assert batch["tooltips"] is not None
    assert len(batch["tooltips"][0]["fields"]) == 1
    assert batch["tooltips"][0]["fields"][0]["name"] == "x"


def test_no_tooltip_produces_null_tooltips():
    scene = _render(_simple_scatter())
    batch = scene["panels"][0]["marks"][0]
    # Without tooltip encoding, tooltips should be absent (null/None)
    assert not batch.get("tooltips"), "expected no tooltips when not encoded"


def test_tooltip_values_are_formatted_correctly():
    df = pl.DataFrame({"x": [1.5], "y": [2.7], "label": ["hello"]})
    chart = (fm.Chart(df).mark_point()
             .encode(x="x:Q", y="y:Q", tooltip=fm.Tooltip("x", "label"))
             .properties(width=300, height=200))
    scene = _render(chart)
    fields = scene["panels"][0]["marks"][0]["tooltips"][0]["fields"]
    field_map = {f["name"]: f["value"] for f in fields}
    assert "x" in field_map
    assert "label" in field_map
    assert field_map["label"] == "hello"


# ── computed axis domains for zoom ────────────────────────────────────────────

def test_auto_scaled_chart_coord_has_x_domain():
    scene = _render(_simple_scatter())
    coord = scene["panels"][0]["coord"]
    assert coord.get("x_domain") is not None, "x_domain missing from auto-scaled chart"
    xlo, xhi = coord["x_domain"]
    # Domain must cover the full data range [1, 3] (may be exactly that after nicing)
    assert xlo <= 1.0
    assert xhi >= 3.0


def test_auto_scaled_chart_coord_has_y_domain():
    scene = _render(_simple_scatter())
    coord = scene["panels"][0]["coord"]
    assert coord.get("y_domain") is not None, "y_domain missing from auto-scaled chart"
    ylo, yhi = coord["y_domain"]
    assert ylo <= 4.0
    assert yhi >= 6.0


def test_explicit_domain_not_overwritten():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = (fm.Chart(df).mark_point()
             .encode(x="x:Q", y="y:Q")
             .coord(fm.CoordCartesian(xlim=(0.0, 10.0)))
             .properties(width=300, height=200))
    scene = _render(chart)
    coord = scene["panels"][0]["coord"]
    xlo, xhi = coord["x_domain"]
    # User-specified domain must be preserved exactly
    assert abs(xlo - 0.0) < 1e-6
    assert abs(xhi - 10.0) < 1e-6


# ── merge_scene_graphs offset_nodes (type tag fix) ────────────────────────────

def test_merge_scene_graphs_offsets_circle_nodes():
    from ferrum._interactive import merge_scene_graphs

    # Use 3 distinct points so scale is non-degenerate and nodes are rendered.
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    spec, data, viewport, theme = chart._render_inputs()
    scene_json = render_interactive(spec, data, viewport=viewport, theme=theme)

    merged = json.loads(merge_scene_graphs(
        [scene_json, scene_json],
        [{"x_offset": 0, "y_offset": 0}, {"x_offset": 300, "y_offset": 0}],
    ))

    p0_nodes = merged["panels"][0]["marks"][0]["nodes"]
    p1_nodes = merged["panels"][1]["marks"][0]["nodes"]
    assert len(p0_nodes) > 0, "panel 0 has no circle nodes"
    assert len(p1_nodes) > 0, "panel 1 has no circle nodes"
    # Second panel's circles must be shifted by 300 px in x
    p0_cx = p0_nodes[0]["cx"]
    p1_cx = p1_nodes[0]["cx"]
    assert abs(p1_cx - p0_cx - 300) < 1.0, f"expected x offset 300, got {p1_cx - p0_cx:.2f}"


def test_merge_scene_graphs_offsets_y():
    from ferrum._interactive import merge_scene_graphs

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    spec, data, viewport, theme = chart._render_inputs()
    scene_json = render_interactive(spec, data, viewport=viewport, theme=theme)

    merged = json.loads(merge_scene_graphs(
        [scene_json, scene_json],
        [{"x_offset": 0, "y_offset": 0}, {"x_offset": 0, "y_offset": 250}],
    ))

    p0_cy = merged["panels"][0]["marks"][0]["nodes"][0]["cy"]
    p1_cy = merged["panels"][1]["marks"][0]["nodes"][0]["cy"]
    assert abs(p1_cy - p0_cy - 250) < 1.0, f"expected y offset 250, got {p1_cy - p0_cy:.2f}"


# ── interaction_config includes selections for JS click handler ───────────────

def test_interaction_config_includes_selections():
    from ferrum._interactive import InteractiveChart

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="grp_sel")
    chart = (fm.Chart(df).mark_point()
             .encode(x="x:Q", y="y:Q", color="group:N", tooltip=fm.Tooltip("group"))
             .add_selection(sel)
             .properties(width=300, height=200))

    spec, data, viewport, theme = chart._render_inputs()
    scene_json = render_interactive(spec, data, viewport=viewport, theme=theme)

    cfg = json.loads(InteractiveChart._extract_interaction_config(scene_json))
    assert "selections" in cfg, "selections missing from interaction_config"
    assert len(cfg["selections"]) == 1
    sel_spec = cfg["selections"][0]
    assert sel_spec["name"] == "grp_sel"
    assert "group" in sel_spec["fields"]


def test_interaction_config_empty_when_no_selections():
    from ferrum._interactive import InteractiveChart

    chart = _simple_scatter()
    spec, data, viewport, theme = chart._render_inputs()
    scene_json = render_interactive(spec, data, viewport=viewport, theme=theme)
    cfg = json.loads(InteractiveChart._extract_interaction_config(scene_json))
    assert cfg.get("selections", []) == []


# ── on_selection_change output routing ───────────────────────────────────────

def test_on_selection_change_callback_fires_on_trait_update():
    from ferrum._interactive import InteractiveChart

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="cb_sel")
    chart = (fm.Chart(df).mark_point()
             .encode(x="x:Q", y="y:Q", color="group:N", tooltip=fm.Tooltip("group"))
             .add_selection(sel)
             .properties(width=300, height=200))
    ic = InteractiveChart(chart)

    received = []
    ic.on_selection_change(lambda state: received.append(state))

    # Simulate the comm message that the JS click handler sends.
    new_state = {"cb_sel": {"type": "point", "indices": [0], "field_values": []}}
    ic._widget.selection_state = new_state

    assert len(received) == 1, "callback must fire exactly once"
    assert received[0] == new_state


def test_on_selection_change_creates_output_widget():
    import ipywidgets
    from ferrum._interactive import InteractiveChart

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    ic = InteractiveChart(
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    )
    assert ic._output_widget is None, "_output_widget must be None before any callback"

    ic.on_selection_change(lambda _: None)
    assert isinstance(ic._output_widget, ipywidgets.Output), \
        "on_selection_change must create an ipywidgets.Output widget"


def test_repr_mimebundle_includes_output_widget_when_callback_registered():
    import ipywidgets
    from ferrum._interactive import InteractiveChart

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    ic = InteractiveChart(
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    )

    # Without a callback, repr returns the bare widget bundle.
    mb_bare = ic._repr_mimebundle_()
    assert mb_bare is not None

    ic.on_selection_change(lambda _: None)

    # After registering a callback, repr must return a VBox bundle so the
    # output area is displayed in the same cell as the chart.
    mb_with_out = ic._repr_mimebundle_()
    assert mb_with_out is not None
    # VBox produces a widget-view mimetype entry
    assert "application/vnd.jupyter.widget-view+json" in mb_with_out


# ── arc hit-test (pie tooltip) ────────────────────────────────────────────────

def test_pie_chart_has_arc_nodes_with_path_type():
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    chart = (fm.Chart(df).mark_arc()
             .encode(x="val:Q", color="cat:N", tooltip=fm.Tooltip("cat", "val"))
             .coord(fm.CoordPolar(theta="x"))
             .properties(width=300, height=300))
    scene = _render(chart)
    assert scene["panels"], "no panels in pie chart scene"
    batch = scene["panels"][0]["marks"][0]
    assert batch["kind"] == "arc"
    assert len(batch["nodes"]) == 3, "expected 3 arc nodes"
    assert all(n["type"] == "path" for n in batch["nodes"]), "arc nodes must be path type"
    assert batch["tooltips"] is not None, "arc tooltips must be populated"
    field_names = {f["name"] for f in batch["tooltips"][0]["fields"]}
    assert field_names == {"cat", "val"}


def test_pie_chart_arc_nodes_have_path_commands():
    df = pl.DataFrame({"cat": ["A", "B"], "val": [40.0, 60.0]})
    chart = (fm.Chart(df).mark_arc()
             .encode(x="val:Q", color="cat:N")
             .coord(fm.CoordPolar(theta="x"))
             .properties(width=300, height=300))
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    for node in batch["nodes"]:
        cmds = node["commands"]
        assert any(c["op"] == "move_to" for c in cmds), "arc wedge missing move_to"
        assert any(c["op"] == "arc_to" for c in cmds), "arc wedge missing arc_to"
        arc_cmd = next(c for c in cmds if c["op"] == "arc_to")
        assert arc_cmd["rx"] > 0, "arc_to must have positive rx (outer radius)"


def test_arc_tooltip_count_equals_node_count():
    # Tooltip array must be 1-to-1 with rendered nodes (not 1-to-1 with raw rows).
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    chart = (fm.Chart(df).mark_arc()
             .encode(x="val:Q", color="cat:N", tooltip=fm.Tooltip("cat", "val"))
             .coord(fm.CoordPolar(theta="x"))
             .properties(width=300, height=300))
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert len(batch["tooltips"]) == len(batch["nodes"]), "one tooltip entry per rendered node"


def test_arc_tooltip_excludes_zero_value_rows():
    # Rows with val=0 are skipped by the arc renderer; their tooltips must also be absent.
    df = pl.DataFrame({"cat": ["A", "B", "C", "D"], "val": [10.0, 0.0, 20.0, 30.0]})
    chart = (fm.Chart(df).mark_arc()
             .encode(x="val:Q", color="cat:N", tooltip=fm.Tooltip("cat", "val"))
             .coord(fm.CoordPolar(theta="x"))
             .properties(width=300, height=300))
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert len(batch["nodes"]) == 3, "zero-value row must be excluded"
    assert len(batch["tooltips"]) == 3, "tooltip count must match node count, not row count"
    labels = {t["fields"][0]["value"] for t in batch["tooltips"]}
    assert "B" not in labels, "category B (val=0) must be absent from tooltips"


def test_arc_no_tooltip_encoding_produces_null_tooltips():
    # Without tooltip encoding, arc marks must have no tooltip data.
    df = pl.DataFrame({"cat": ["A", "B"], "val": [40.0, 60.0]})
    chart = (fm.Chart(df).mark_arc()
             .encode(x="val:Q", color="cat:N")
             .coord(fm.CoordPolar(theta="x"))
             .properties(width=300, height=300))
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert not batch.get("tooltips"), "no tooltips when encoding absent"


def test_polar_chart_coord_kind_is_polar():
    # Polar coord kind must be present in scene so the GPU zoom guard can inspect it.
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    chart = (fm.Chart(df).mark_arc()
             .encode(x="val:Q", color="cat:N")
             .coord(fm.CoordPolar(theta="x"))
             .properties(width=300, height=300))
    scene = _render(chart)
    coord = scene["panels"][0]["coord"]
    assert coord.get("kind") == "polar", "polar chart must have coord.kind == 'polar'"


# ── tick_levels structure (zoom scale label fix) ──────────────────────────────
#
# Regression for: build_zoomed_text_json matched tick labels by scale-function
# pixel position, which diverges from the axis layout's uniform-band positions.
# Fixed by clustering axis text by shared coordinate instead.
# These tests prove the preconditions that the clustering approach relies on.

def _scatter_scene(width: int = 400, height: int = 300) -> dict:
    """
    A quantitative scatter with x in [100, 500] and y in [0.1, 0.5] so tick
    labels on the two axes are guaranteed non-overlapping (no shared strings).
    """
    df = pl.DataFrame({
        "x": [100.0, 200.0, 300.0, 400.0, 500.0],
        "y": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    return _render(
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=width, height=height)
    )


def test_tick_levels_present_for_quantitative_axes():
    scene = _scatter_scene()
    tl = scene["interaction"]["tick_levels"]
    assert len(tl) == 1, "one PanelTickLevels entry per panel"
    assert tl[0]["panel_id"] == 0
    assert len(tl[0]["x_levels"]) > 0, "x_levels must be non-empty for Q x-axis"
    assert len(tl[0]["y_levels"]) > 0, "y_levels must be non-empty for Q y-axis"


def test_tick_levels_have_nonempty_ticks():
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]
    # Every zoom level should have at least some ticks.
    for lvl in ptl["x_levels"]:
        assert len(lvl["ticks"]) > 0, f"x_level {lvl} has no ticks"
    for lvl in ptl["y_levels"]:
        assert len(lvl["ticks"]) > 0, f"y_level {lvl} has no ticks"


def test_tick_level_labels_appear_in_axis_text_nodes():
    """
    Tick label strings from the *initial* zoom level (level[1], count=8) must
    appear as text-node content in panel.axes.  The clustering algorithm
    identifies tick labels by content match, so this is the core precondition.
    Only level[1] is checked because finer-grained levels (16, 32 ticks) may
    not all be present in the initial scene render.
    """
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]

    axis_texts: set[str] = set()
    for node in scene["panels"][0]["axes"]:
        if node.get("type") == "text":
            axis_texts.add(node["content"])

    # level[1] is the default 8-tick zoom level (zoom factor 0.5–2.0).
    x_labels = {t["label"] for t in ptl["x_levels"][1]["ticks"]}
    y_labels = {t["label"] for t in ptl["y_levels"][1]["ticks"]}

    assert x_labels & axis_texts == x_labels, (
        f"x tick labels (level 1) not found in axis text: {x_labels - axis_texts}"
    )
    assert y_labels & axis_texts == y_labels, (
        f"y tick labels (level 1) not found in axis text: {y_labels - axis_texts}"
    )


def test_x_axis_tick_labels_share_y_coordinate():
    """
    All x-axis tick labels (level[1]) must have the same canvas y coordinate.
    This is the invariant the clustering approach exploits: group by the most
    common y → that cluster is the x-axis row.
    """
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]

    x_labels = {t["label"] for t in ptl["x_levels"][1]["ticks"]}
    x_tick_nodes = [
        n for n in scene["panels"][0]["axes"]
        if n.get("type") == "text" and n["content"] in x_labels
    ]
    assert len(x_tick_nodes) >= 2, "need at least 2 x-axis tick labels to test clustering"

    ys = [n["y"] for n in x_tick_nodes]
    assert max(ys) - min(ys) < 2.0, (
        f"x-axis tick labels must share a y-coordinate; spread={max(ys)-min(ys):.2f}"
    )


def test_y_axis_tick_labels_share_x_coordinate():
    """
    All y-axis tick labels (level[1]) must have the same canvas x coordinate.
    """
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]

    y_labels = {t["label"] for t in ptl["y_levels"][1]["ticks"]}
    y_tick_nodes = [
        n for n in scene["panels"][0]["axes"]
        if n.get("type") == "text" and n["content"] in y_labels
    ]
    assert len(y_tick_nodes) >= 2, "need at least 2 y-axis tick labels to test clustering"

    xs = [n["x"] for n in y_tick_nodes]
    assert max(xs) - min(xs) < 2.0, (
        f"y-axis tick labels must share an x-coordinate; spread={max(xs)-min(xs):.2f}"
    )


def test_tick_data_pixels_differ_from_axis_text_positions():
    """
    Documents the root cause of the old pixel-match failure: tick_data
    scale-function outputs do NOT match axis text positions.  The axis
    layout uses uniform band centering while tick_data uses the actual
    scale function — systematically different mappings.
    """
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]

    td_px = {t["label"]: t["pixel"] for t in ptl["x_levels"][1]["ticks"]}
    x_labels = set(td_px)

    x_tick_nodes = [
        n for n in scene["panels"][0]["axes"]
        if n.get("type") == "text" and n["content"] in x_labels
    ]

    # At least some labels must have mismatched positions.  A perfect match
    # would mean the old approach worked, which we know it didn't.
    mismatches = [
        n for n in x_tick_nodes
        if abs(n["x"] - td_px[n["content"]]) > 0.5
    ]
    assert len(mismatches) > 0, (
        "Expected at least one label whose axis text x differs from tick_data pixel; "
        "if all match, the old pixel-match approach would have worked fine."
    )


# ── zoom transform math (tooltip inverse-transform fix) ───────────────────────
#
# Regression for: JS tooltip hit-test used original mark positions after zoom.
# Fixed by tracking a JS _zoom object and inverse-transforming the mouse before
# _hitTest.  The math mirrors Rust ZoomPanState.  These tests prove correctness.

def _zoom_identity():
    return {"sx": 1.0, "sy": 1.0, "tx": 0.0, "ty": 0.0}


def _apply_wheel(t, delta_y, cx, cy, zmin=0.1, zmax=50.0):
    """Mirror of JS wheel handler / Rust ZoomPanState::on_wheel."""
    f = 1.0 + delta_y * 0.001
    sx = min(zmax, max(zmin, t["sx"] * f))
    sy = min(zmax, max(zmin, t["sy"] * f))
    return {
        "sx": sx,
        "sy": sy,
        "tx": cx - sx * ((cx - t["tx"]) / t["sx"]),
        "ty": cy - sy * ((cy - t["ty"]) / t["sy"]),
    }


def _apply_pan(t, dx, dy):
    return {"sx": t["sx"], "sy": t["sy"], "tx": t["tx"] + dx, "ty": t["ty"] + dy}


def _inv_zoom(t, x, y):
    ox = (x - t["tx"]) / t["sx"] if abs(t["sx"]) > 1e-10 else x
    oy = (y - t["ty"]) / t["sy"] if abs(t["sy"]) > 1e-10 else y
    return ox, oy


def _fwd_zoom(t, x, y):
    return x * t["sx"] + t["tx"], y * t["sy"] + t["ty"]


def test_zoom_identity_is_noop():
    t = _zoom_identity()
    ox, oy = _inv_zoom(t, 150.0, 200.0)
    assert abs(ox - 150.0) < 1e-10 and abs(oy - 200.0) < 1e-10


def test_zoom_wheel_forward_inverse_roundtrip():
    """Applying wheel zoom then inverse-transforming must recover the original position."""
    t = _zoom_identity()
    t = _apply_wheel(t, delta_y=300.0, cx=200.0, cy=150.0)  # zoom in ~1.3×
    # A mark at (200, 150) — the cursor — stays at (200, 150) after zoom.
    fx, fy = _fwd_zoom(t, 200.0, 150.0)
    assert abs(fx - 200.0) < 1e-6 and abs(fy - 150.0) < 1e-6
    # Any arbitrary point round-trips.
    orig = (80.0, 60.0)
    rx, ry = _fwd_zoom(t, *orig)
    bx, by = _inv_zoom(t, rx, ry)
    assert abs(bx - orig[0]) < 1e-8 and abs(by - orig[1]) < 1e-8


def test_zoom_pan_forward_inverse_roundtrip():
    t = _zoom_identity()
    t = _apply_pan(t, dx=40.0, dy=-20.0)
    orig = (100.0, 200.0)
    rx, ry = _fwd_zoom(t, *orig)
    bx, by = _inv_zoom(t, rx, ry)
    assert abs(bx - orig[0]) < 1e-10 and abs(by - orig[1]) < 1e-10


def test_zoom_wheel_then_pan_roundtrip():
    t = _zoom_identity()
    t = _apply_wheel(t, 500.0, 200.0, 150.0)
    t = _apply_pan(t, 30.0, -15.0)
    orig = (55.0, 120.0)
    rx, ry = _fwd_zoom(t, *orig)
    bx, by = _inv_zoom(t, rx, ry)
    assert abs(bx - orig[0]) < 1e-8 and abs(by - orig[1]) < 1e-8


def test_zoom_reset_restores_identity():
    t = _zoom_identity()
    t = _apply_wheel(t, 800.0, 200.0, 150.0)
    t = _apply_pan(t, 50.0, 50.0)
    t = _zoom_identity()  # reset
    assert t["sx"] == 1.0 and t["sy"] == 1.0
    assert t["tx"] == 0.0 and t["ty"] == 0.0


def test_zoom_wheel_clamps_to_min():
    t = _zoom_identity()
    for _ in range(300):
        t = _apply_wheel(t, -5000.0, 100.0, 100.0)
    assert t["sx"] >= 0.1 and t["sy"] >= 0.1


def test_zoom_wheel_clamps_to_max():
    t = _zoom_identity()
    for _ in range(300):
        t = _apply_wheel(t, 5000.0, 100.0, 100.0)
    assert t["sx"] <= 50.0 and t["sy"] <= 50.0


def test_zoom_cursor_stays_fixed_under_wheel():
    """The point under the cursor must not move visually when zooming."""
    t = _zoom_identity()
    cursor = (180.0, 130.0)
    # Place a mark at the cursor position, zoom around it.
    for delta in [200.0, 500.0, -150.0]:
        t = _apply_wheel(t, delta, *cursor)
        rx, ry = _fwd_zoom(t, *cursor)
        assert abs(rx - cursor[0]) < 1e-6, f"cursor x moved after delta={delta}"
        assert abs(ry - cursor[1]) < 1e-6, f"cursor y moved after delta={delta}"
