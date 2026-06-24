"""Regression tests for interactive renderer fixes.

Covers:
- Multi-field tooltip serialization → scene JSON (tooltip_fields)
- Computed axis domains injected into scene coord for JS zoom
- Single-field Tooltip backward compat
- interaction_config includes selections array for JS click handler
- Polar/Geo coord kind guards for zoom/pan (labels must not vanish)
- Hex and geoshape scene JSON well-formedness
- Binary packed instances produce GPU data for high-cardinality batches
- Packed tooltips/data_indices travel as binary sidecar, not JSON
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann
from ferrum._core import render_interactive
from ferrum.selection import selection_point, value


# ── helpers ──────────────────────────────────────────────────────────────────


def _render(chart: fm.Chart) -> dict:
    spec, data, viewport, theme, _ = chart._render_inputs()
    result = render_interactive(spec, data, viewport=viewport, theme=theme)
    json_str = result[0] if isinstance(result, tuple) else result
    return json.loads(json_str)


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
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", tooltip=fm.Tooltip(fm.TooltipField("x", format=".2f"), "y"))
        .properties(width=300, height=200)
    )
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
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", tooltip=fm.Tooltip("x", "label"))
        .properties(width=300, height=200)
    )
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
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .coord(fm.CoordCartesian(xlim=(0.0, 10.0)))
        .properties(width=300, height=200)
    )
    scene = _render(chart)
    coord = scene["panels"][0]["coord"]
    xlo, xhi = coord["x_domain"]
    # User-specified domain must be preserved exactly
    assert abs(xlo - 0.0) < 1e-6
    assert abs(xhi - 10.0) < 1e-6


# ── interaction_config includes selections for JS click handler ───────────────


def test_interaction_config_includes_selections():
    from ferrum._interactive import InteractiveChart

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="grp_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="group:N", tooltip=fm.Tooltip("group"))
        .add_selection(sel)
        .properties(width=300, height=200)
    )

    spec, data, viewport, theme, _ = chart._render_inputs()
    scene_json, _ = render_interactive(spec, data, viewport=viewport, theme=theme)

    cfg = json.loads(InteractiveChart._extract_interaction_config(scene_json))
    assert "selections" in cfg, "selections missing from interaction_config"
    assert len(cfg["selections"]) == 1
    sel_spec = cfg["selections"][0]
    assert sel_spec["name"] == "grp_sel"
    assert "group" in sel_spec["fields"]


def test_interaction_config_empty_when_no_selections():
    from ferrum._interactive import InteractiveChart

    chart = _simple_scatter()
    spec, data, viewport, theme, _ = chart._render_inputs()
    scene_json, _ = render_interactive(spec, data, viewport=viewport, theme=theme)
    cfg = json.loads(InteractiveChart._extract_interaction_config(scene_json))
    assert cfg.get("selections", []) == []


# ── on_selection_change output routing ───────────────────────────────────────


def test_on_selection_change_callback_fires_on_trait_update():
    from ferrum._interactive import InteractiveChart

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="cb_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="group:N", tooltip=fm.Tooltip("group"))
        .add_selection(sel)
        .properties(width=300, height=200)
    )
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
    assert isinstance(ic._output_widget, ipywidgets.Output), (
        "on_selection_change must create an ipywidgets.Output widget"
    )


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
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N", tooltip=fm.Tooltip("cat", "val"))
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
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
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
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
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N", tooltip=fm.Tooltip("cat", "val"))
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert len(batch["tooltips"]) == len(batch["nodes"]), "one tooltip entry per rendered node"


def test_arc_tooltip_excludes_zero_value_rows():
    # Rows with val=0 are skipped by the arc renderer; their tooltips must also be absent.
    df = pl.DataFrame({"cat": ["A", "B", "C", "D"], "val": [10.0, 0.0, 20.0, 30.0]})
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N", tooltip=fm.Tooltip("cat", "val"))
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert len(batch["nodes"]) == 3, "zero-value row must be excluded"
    assert len(batch["tooltips"]) == 3, "tooltip count must match node count, not row count"
    labels = {t["fields"][0]["value"] for t in batch["tooltips"]}
    assert "B" not in labels, "category B (val=0) must be absent from tooltips"


def test_arc_no_tooltip_encoding_produces_null_tooltips():
    # Without tooltip encoding, arc marks must have no tooltip data.
    df = pl.DataFrame({"cat": ["A", "B"], "val": [40.0, 60.0]})
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert not batch.get("tooltips"), "no tooltips when encoding absent"


def test_polar_chart_coord_kind_is_polar():
    # Polar coord kind must be present in scene so the GPU zoom guard can inspect it.
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
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
    df = pl.DataFrame(
        {
            "x": [100.0, 200.0, 300.0, 400.0, 500.0],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
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
        n
        for n in scene["panels"][0]["axes"]
        if n.get("type") == "text" and n["content"] in x_labels
    ]
    assert len(x_tick_nodes) >= 2, "need at least 2 x-axis tick labels to test clustering"

    ys = [n["y"] for n in x_tick_nodes]
    assert max(ys) - min(ys) < 2.0, (
        f"x-axis tick labels must share a y-coordinate; spread={max(ys) - min(ys):.2f}"
    )


def test_y_axis_tick_labels_share_x_coordinate():
    """
    All y-axis tick labels (level[1]) must have the same canvas x coordinate.
    """
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]

    y_labels = {t["label"] for t in ptl["y_levels"][1]["ticks"]}
    y_tick_nodes = [
        n
        for n in scene["panels"][0]["axes"]
        if n.get("type") == "text" and n["content"] in y_labels
    ]
    assert len(y_tick_nodes) >= 2, "need at least 2 y-axis tick labels to test clustering"

    xs = [n["x"] for n in y_tick_nodes]
    assert max(xs) - min(xs) < 2.0, (
        f"y-axis tick labels must share an x-coordinate; spread={max(xs) - min(xs):.2f}"
    )


def test_tick_data_pixels_coincide_with_axis_text_positions():
    """
    Pins the continuous-axis scale-projection fix (2026-05-30).

    Previously the x-axis layout placed major ticks at uniform band centers
    while ``tick_data`` reported the scale function's actual pixel — two
    systematically different mappings, so axis text and tick_data pixels did
    NOT line up (this test used to assert they *differ*, documenting that old
    bug). The fix projects each continuous tick through the same scale (with
    the same padding inset that places data marks), so a tick at value ``v``
    and the data mark at value ``v`` now share a pixel. Axis text x and
    tick_data pixel must therefore coincide.
    """
    scene = _scatter_scene()
    ptl = scene["interaction"]["tick_levels"][0]

    td_px = {t["label"]: t["pixel"] for t in ptl["x_levels"][1]["ticks"]}
    x_labels = set(td_px)

    x_tick_nodes = [
        n
        for n in scene["panels"][0]["axes"]
        if n.get("type") == "text" and n["content"] in x_labels
    ]
    assert x_tick_nodes, "expected at least one matching x-axis tick label"

    # Every axis text label's x must now coincide with its tick_data pixel
    # (within sub-pixel tolerance), since both are the same scale projection.
    mismatches = [n for n in x_tick_nodes if abs(n["x"] - td_px[n["content"]]) > 0.5]
    assert not mismatches, (
        "continuous-axis scale-projection fix should make axis text x and "
        f"tick_data pixel coincide; mismatched labels: "
        f"{[(n['content'], n['x'], td_px[n['content']]) for n in mismatches]}"
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


# ── Polar/Geo coord kind guards (zoom/pan must not clear labels) ──────────


def test_polar_coord_kind_blocks_zoom_domain_extraction():
    """Polar scene coord must NOT have x_domain/y_domain — the JS debounced
    handler checks `if (!xs || !ys) return` to skip the Python round-trip.
    If a polar chart accidentally exposes domains, the zoom handler will
    inject a CoordCartesian and destroy the polar layout."""
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    coord = scene["panels"][0]["coord"]
    assert coord["kind"] == "polar"
    assert coord.get("x_domain") is None, "polar coord must not expose x_domain"
    assert coord.get("y_domain") is None, "polar coord must not expose y_domain"


def test_geo_coord_kind_present_in_scene():
    """Geo scene coord must be kind='geo' so the WASM on_wheel/on_pan
    guards can skip affine transforms for geographic projections."""
    bare_polygon = {
        "type": "Polygon",
        "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
    }
    chart = (
        fm.Chart(bare_polygon)
        .mark_geoshape()
        .encode(color="__geometry__:N")
        .coord(fm.CoordGeo(projection="equirectangular"))
        .properties(width=400, height=300)
    )
    scene = _render(chart)
    coord = scene["panels"][0]["coord"]
    assert coord["kind"] == "geo", "geo chart must have coord.kind == 'geo'"
    assert coord.get("x_domain") is None, "geo coord must not expose x_domain"
    assert coord.get("y_domain") is None, "geo coord must not expose y_domain"


def test_polar_scene_has_text_nodes_for_labels():
    """Regression: polar pie chart must have text nodes (wedge labels or legend)
    that WASM must preserve on wheel/pan events instead of clearing."""
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    chart = (
        fm.Chart(df)
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordPolar(theta="x"))
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    all_text = []
    for node in scene.get("legend", []):
        if node.get("type") == "text":
            all_text.append(node["content"])
    for node in scene.get("title", []):
        if node.get("type") == "text":
            all_text.append(node["content"])
    assert len(all_text) > 0, (
        "polar chart must have text nodes (legend/title) that zoom guard preserves"
    )


# ── Hexbin and geoshape scene JSON well-formedness ────────────────────────


def test_hexbin_scene_json_has_polygon_nodes():
    """Regression: hexbin interactive scene must contain polygon nodes."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.normal(0, 2, 500).tolist(), "y": rng.normal(0, 2, 500).tolist()})
    chart = (
        fm.Chart(df)
        .mark_hex(aggregate="count")
        .encode(x="x:Q", y="y:Q")
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert batch["kind"] == "polygon", "hex mark must produce polygon batch"
    assert len(batch["nodes"]) > 0, "hex must have polygon nodes"
    node = batch["nodes"][0]
    assert node["type"] == "polygon"
    assert len(node["rings"]) >= 1, "polygon must have at least one ring"
    assert len(node["rings"][0]) >= 3, "polygon ring must have ≥3 points"


def test_hexbin_polygons_inside_plot_area():
    """Regression: every hex polygon centroid must fall within the plot_area clip
    rect. If hex coordinates are computed in data space but not transformed to
    pixel space, they'll be near-zero and outside the plot area — which the WASM
    tessellator will render as invisible (clipped or off-canvas)."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.normal(0, 2, 2000).tolist(), "y": rng.normal(0, 2, 2000).tolist()})
    chart = (
        fm.Chart(df)
        .mark_hex(aggregate="count")
        .encode(x="x:Q", y="y:Q")
        .properties(width=450, height=400)
    )
    scene = _render(chart)
    clip = scene["panels"][0]["clip"]
    cx, cy, cw, ch = clip["x"], clip["y"], clip["w"], clip["h"]
    nodes = scene["panels"][0]["marks"][0]["nodes"]
    assert len(nodes) >= 10, f"expected ≥10 hex bins; got {len(nodes)}"
    outside = 0
    for node in nodes:
        ring = node["rings"][0]
        centroid_x = sum(p[0] for p in ring) / len(ring)
        centroid_y = sum(p[1] for p in ring) / len(ring)
        if not (cx - 20 <= centroid_x <= cx + cw + 20 and cy - 20 <= centroid_y <= cy + ch + 20):
            outside += 1
    assert outside == 0, (
        f"{outside}/{len(nodes)} hex polygon centroids outside plot area "
        f"[{cx:.0f},{cy:.0f},{cx + cw:.0f},{cy + ch:.0f}]"
    )


def test_hexbin_polygons_have_nonzero_area():
    """Regression: hex polygons must have non-degenerate area. A polygon whose
    vertices all collapse to a single point (area ≈ 0) will tessellate to zero
    triangles in the WASM renderer, producing invisible geometry."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.normal(0, 2, 500).tolist(), "y": rng.normal(0, 2, 500).tolist()})
    chart = (
        fm.Chart(df)
        .mark_hex(aggregate="count")
        .encode(x="x:Q", y="y:Q")
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    nodes = scene["panels"][0]["marks"][0]["nodes"]
    for i, node in enumerate(nodes):
        ring = node["rings"][0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        area = (
            abs(
                sum(
                    xs[j] * ys[(j + 1) % len(xs)] - xs[(j + 1) % len(xs)] * ys[j]
                    for j in range(len(xs))
                )
            )
            / 2.0
        )
        assert area > 1.0, f"hex polygon {i} has degenerate area={area:.4f}"


def test_hexbin_polygons_have_fill_color():
    """Regression: hex polygons must have a non-null fill so the WASM tessellator
    actually emits triangles. A polygon with fill=null renders as invisible."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.normal(0, 2, 500).tolist(), "y": rng.normal(0, 2, 500).tolist()})
    chart = (
        fm.Chart(df)
        .mark_hex(aggregate="count")
        .encode(x="x:Q", y="y:Q")
        .properties(width=300, height=300)
    )
    scene = _render(chart)
    nodes = scene["panels"][0]["marks"][0]["nodes"]
    for i, node in enumerate(nodes):
        fill = node["style"].get("fill")
        assert fill is not None, f"hex polygon {i} has null fill — will be invisible in WASM"


def test_geoshape_scene_json_has_polygon_nodes():
    """Regression: geoshape interactive scene must produce polygon nodes."""
    bare_polygon = {
        "type": "Polygon",
        "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
    }
    chart = (
        fm.Chart(bare_polygon)
        .mark_geoshape()
        .encode(color="__geometry__:N")
        .coord(fm.CoordGeo(projection="equirectangular"))
        .properties(width=400, height=300)
    )
    scene = _render(chart)
    batch = scene["panels"][0]["marks"][0]
    assert batch["kind"] == "polygon", "geoshape must produce polygon batch"
    assert len(batch["nodes"]) >= 1
    node = batch["nodes"][0]
    assert node["type"] == "polygon"
    ring = node["rings"][0]
    assert len(ring) >= 4, "projected polygon ring must have ≥4 points"


def test_geoshape_polygon_inside_plot_area():
    """Regression: geoshape projected polygon must be within the plot area."""
    bare_polygon = {
        "type": "Polygon",
        "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
    }
    chart = (
        fm.Chart(bare_polygon)
        .mark_geoshape()
        .encode(color="__geometry__:N")
        .coord(fm.CoordGeo(projection="equirectangular"))
        .properties(width=400, height=300)
    )
    scene = _render(chart)
    clip = scene["panels"][0]["clip"]
    cx, cy, cw, ch = clip["x"], clip["y"], clip["w"], clip["h"]
    ring = scene["panels"][0]["marks"][0]["nodes"][0]["rings"][0]
    for pt in ring:
        assert cx - 5 <= pt[0] <= cx + cw + 5 and cy - 5 <= pt[1] <= cy + ch + 5, (
            f"geoshape point {pt} outside plot area [{cx:.0f},{cy:.0f},{cx + cw:.0f},{cy + ch:.0f}]"
        )
    for pt in ring:
        assert 0 <= pt[0] <= 500 and 0 <= pt[1] <= 400, (
            f"projected polygon point {pt} must be in pixel space"
        )


# ── Binary packed instance bridge regression tests ────────────────────


def _render_with_packed(chart: fm.Chart) -> tuple[dict, bytes]:
    """Return (scene_dict, packed_bytes) for a chart."""
    spec, data, viewport, theme, _ = chart._render_inputs()
    result = render_interactive(spec, data, viewport=viewport, theme=theme)
    json_str, packed = result
    return json.loads(json_str), packed


def _packed_scatter(n: int = 2000, tooltips: bool = False) -> fm.Chart:
    """Create a scatter above the packing threshold (1000)."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    if tooltips:
        chart = chart.encode(tooltip=fm.Tooltip("x", "y"))
    return chart.properties(
        width=300, height=200, render_config=fm.RenderConfig(raster_threshold=None)
    )


class TestPackedInstanceBridge:
    """Regression: packed batches (>1000 marks) must produce GPU instance data
    via the binary sidecar, not per-node JSON."""

    def test_packed_batch_has_empty_nodes_in_json(self):
        """Regression: nodes are cleared from JSON for packed batches so the
        JSON stays small. If nodes are accidentally left in, JSON bloats."""
        scene, packed = _render_with_packed(_packed_scatter(2000))
        batch = scene["panels"][0]["marks"][0]
        assert len(batch.get("nodes", [])) == 0, "packed batch should have empty nodes in JSON"

    def test_packed_bytes_are_nonempty(self):
        """Regression: the packed sidecar must contain instance data. If empty,
        the WASM renderer has no circles to draw (blank chart)."""
        _, packed = _render_with_packed(_packed_scatter(2000))
        assert len(packed) > 0, "packed bytes must be non-empty for >1000 marks"

    def test_packed_header_has_20_bytes(self):
        """Regression: header must be 20 bytes (5 × u32) with flags field.
        The old 16-byte header caused the WASM unpacker to misalign."""
        import struct

        _, packed = _render_with_packed(_packed_scatter(2000))
        assert len(packed) >= 20, "packed data must have at least a 20-byte header"
        pi, bi, kind, count, flags = struct.unpack_from("<5I", packed)
        assert pi == 0, "panel_idx should be 0"
        assert bi == 0, "batch_idx should be 0"
        assert kind == 0, "kind should be 0 (circle)"
        assert count == 2000, f"count should be 2000; got {count}"

    def test_packed_instance_bytes_correct_size(self):
        """Regression: instance data must be count × 64 bytes (CircleInstance).
        Wrong size means the WASM bytemuck cast fails silently."""
        import struct

        _, packed = _render_with_packed(_packed_scatter(2000))
        _, _, kind, count, flags = struct.unpack_from("<5I", packed)
        instance_size = 64  # CircleInstance: 16 × f32
        expected_bytes = 20 + count * instance_size  # header + instances
        assert len(packed) >= expected_bytes, (
            f"packed data too small: {len(packed)} < {expected_bytes}"
        )

    def test_small_chart_not_packed(self):
        """Regression: charts with <1000 marks must NOT be packed. They keep
        per-node JSON for JS hit-testing and tooltip lookup."""
        scene, packed = _render_with_packed(
            fm.Chart(pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}))
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .properties(width=200, height=200, render_config=fm.RenderConfig(raster_threshold=None))
        )
        batch = scene["panels"][0]["marks"][0]
        assert len(batch.get("nodes", [])) == 3, "small chart should keep nodes in JSON"
        assert len(packed) == 0, "small chart should have no packed bytes"

    def test_packed_json_size_under_1mb(self):
        """Regression: the whole point of packing is compact JSON. A 2000-point
        scatter's JSON should be tiny (axes + legend only), not 200KB+."""
        scene_str = json.dumps(_render_with_packed(_packed_scatter(2000))[0])
        size_kb = len(scene_str) / 1024
        assert size_kb < 100, f"packed scene JSON should be <100KB; got {size_kb:.0f}KB"


class TestPackedTooltips:
    """Regression: tooltip data for packed batches must travel as binary in the
    sidecar and be cleared from JSON."""

    def test_tooltips_cleared_from_json_for_packed_batch(self):
        """Regression: if tooltips stay in JSON for a packed batch, JSON bloats
        back to megabytes (the original bottleneck)."""
        scene, _ = _render_with_packed(_packed_scatter(2000, tooltips=True))
        batch = scene["panels"][0]["marks"][0]
        tooltips = batch.get("tooltips")
        assert tooltips is None or len(tooltips) == 0, (
            "packed batch should have tooltips cleared from JSON"
        )

    def test_packed_bytes_contain_tooltip_data(self):
        """Regression: tooltip string table must be in the sidecar. If missing,
        hover shows nothing on packed batches."""
        import struct

        _, packed = _render_with_packed(_packed_scatter(2000, tooltips=True))
        _, _, _, count, flags = struct.unpack_from("<5I", packed)
        assert flags & 0x1 != 0, f"HAS_TOOLTIPS flag (0x1) must be set; got flags=0x{flags:x}"

    def test_data_indices_in_packed_bytes(self):
        """Regression: data_indices must be packed for selection state to work
        on packed batches."""
        import struct

        _, packed = _render_with_packed(_packed_scatter(2000))
        _, _, _, count, flags = struct.unpack_from("<5I", packed)
        assert flags & 0x2 != 0, f"HAS_DATA_INDICES flag (0x2) must be set; got flags=0x{flags:x}"

    def test_small_chart_tooltips_in_json(self):
        """Regression: small charts (<1000 marks) must keep tooltips in JSON
        for the JS tooltip display path."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", tooltip=fm.Tooltip("x", "y"))
            .properties(width=200, height=200)
        )
        scene = _render(chart)
        batch = scene["panels"][0]["marks"][0]
        assert batch.get("tooltips") is not None, "small chart must keep tooltips in JSON"
        assert len(batch["tooltips"]) == 3
        assert len(batch["tooltips"][0]["fields"]) == 2


# ── Item 19: SceneNode::Raw not silently dropped in WASM scene graph ───────────
#
# Regression against the previous `console.warn` silent drop of Raw nodes.
# The scene graph is the shared representation consumed by both the static SVG
# renderer and the WASM renderer. These tests assert that Raw nodes appear in
# the scene JSON so they can reach the WASM export path.


def _continuous_color_chart() -> fm.Chart:
    """A continuous-color scatter that produces a colorbar gradient Raw node."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.0, 3.0, 5.0],
            "z": [0.1, 0.4, 0.2, 0.9, 0.6],
        }
    )
    return (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="z:Q")
        .properties(width=300, height=200)
    )


def _find_raw_nodes(scene: dict) -> list[dict]:
    """Collect all SceneNode::Raw nodes from a scene graph dict."""
    raw_nodes: list[dict] = []

    def _walk(nodes: list[dict]) -> None:
        for node in nodes:
            if node.get("type") == "raw":
                raw_nodes.append(node)
            elif node.get("type") == "group":
                _walk(node.get("children", []))

    for region in ("title", "legend", "decorations"):
        _walk(scene.get(region, []))
    for panel in scene.get("panels", []):
        for region in ("grid", "axes", "annotations", "strip_title"):
            _walk(panel.get(region, []))
        for batch in panel.get("marks", []):
            _walk(batch.get("nodes", []))

    return raw_nodes


class TestRawNodeNotSilentlyDropped:
    """Item 19 regression: SceneNode::Raw must appear in scene JSON."""

    def test_continuous_color_chart_has_raw_nodes(self):
        """A continuous-color chart must produce Raw nodes (colorbar gradient).

        This is the core regression test: previously Raw nodes were dropped
        with a console.warn in the WASM renderer. The scene graph must contain
        them so the JS injection path can receive them.
        """
        scene = _render(_continuous_color_chart())
        raw_nodes = _find_raw_nodes(scene)
        assert len(raw_nodes) > 0, (
            "continuous-color chart must produce Raw scene nodes (colorbar gradient); "
            "found none. This means the gradient is silently dropped."
        )

    def test_continuous_color_colorbar_contains_linearGradient(self):
        """The Raw nodes from a continuous-color chart must contain a linearGradient.

        Regression: the colorbar SVG was previously silently dropped in the WASM
        renderer. The `svg` field of the Raw node must contain the gradient markup
        that the JS overlay will inject into the text-overlay <svg>.
        """
        scene = _render(_continuous_color_chart())
        raw_nodes = _find_raw_nodes(scene)
        assert any("linearGradient" in node.get("svg", "") for node in raw_nodes), (
            "at least one Raw node must contain 'linearGradient'; "
            f"found raw svgs: {[n.get('svg', '')[:80] for n in raw_nodes]}"
        )

    def test_continuous_color_raw_nodes_have_chrome_anchor(self):
        """Colorbar Raw nodes must have anchor='chrome' (fixed during pan/zoom).

        Chrome anchor means the colorbar stays fixed in the overlay during
        pan/zoom, consistent with the legend being non-data-space UI chrome.
        """
        scene = _render(_continuous_color_chart())
        raw_nodes = _find_raw_nodes(scene)
        grad_nodes = [n for n in raw_nodes if "linearGradient" in n.get("svg", "")]
        assert len(grad_nodes) > 0, "must have at least one linearGradient Raw node"
        for node in grad_nodes:
            assert node.get("anchor") in ("chrome", None), (
                f"colorbar gradient Raw node must have anchor='chrome' or default; "
                f"got anchor={node.get('anchor')!r}"
            )

    def test_colorbar_emits_single_self_contained_raw_fragment(self):
        """The continuous-color colorbar must emit ONE Raw fragment that carries
        both its gradient ``<defs>`` and the consuming ``<rect fill="url(#…)">``.

        Part 1 of the R5 fix merged the previously-separate defs and rect Raw
        nodes into a single self-contained fragment.  Co-locating the id
        definition with its only consumer is the precondition that lets the JS
        overlay namespace each fragment independently (Part 2) without dangling
        the ``url(#…)`` reference.
        """
        import re

        scene = _render(_continuous_color_chart())
        raw_nodes = _find_raw_nodes(scene)
        grad_frags = [n.get("svg", "") for n in raw_nodes if "linearGradient" in n.get("svg", "")]
        assert len(grad_frags) == 1, (
            "colorbar must emit exactly one gradient Raw fragment (defs + rect "
            f"merged); found {len(grad_frags)}"
        )
        frag = grad_frags[0]
        ids = re.findall(r'\bid="([^"]+)"', frag)
        urls = re.findall(r"url\(#([^)]+)\)", frag)
        assert ids, "fragment must define the gradient id"
        # Every url(#X) consumer must reference an id DEFINED in the same fragment.
        assert set(urls) <= set(ids), (
            f"colorbar fragment has a url(#…) with no co-located id def: defs={ids!r} refs={urls!r}"
        )

    def test_per_fragment_namespacing_distinguishes_colliding_colorbar_ids(self):
        """Two self-contained Raw fragments that each author ``ferrum-colorbar-0``
        must receive DISTINCT ids after per-fragment namespacing — the R5 bug.

        Trigger: a continuous-color chart (outer colorbar) that also contains an
        ``.inset()`` of another continuous-color chart.  The inset is rendered in
        a separate pass and embedded verbatim as one nested-``<svg>`` Raw
        fragment, so it carries its OWN ``ferrum-colorbar-0``.  Both fragments are
        self-contained (defs + consumer co-located).

        This test replicates both the OLD shared-map logic and the NEW
        per-fragment logic from ``ferrum-anywidget.js``:

        - OLD shared map: one id→namespaced map across ALL fragments collapses
          both ``ferrum-colorbar-0`` defs to the SAME namespaced id → one
          ``<linearGradient>`` wins, the other colorbar renders with the wrong
          gradient.
        - NEW per-fragment: each fragment gets prefix ``ferrum-raw-{load}-{frag}-``
          so the two colorbars get distinct ids AND no reference dangles.
        """
        import re

        from ferrum.structural import Inset

        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [2.0, 4.0, 1.0, 3.0, 5.0],
                "z": [0.1, 0.4, 0.2, 0.9, 0.6],
            }
        )
        inner = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="z:Q")
            .properties(width=300, height=200)
        )
        outer = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color="z:Q")
            .properties(width=700, height=500)
        )
        outer = outer + Inset(chart=inner, bounds=(0.55, 0.1, 0.95, 0.55))

        spec, data, viewport, theme, chart_config = outer._render_inputs()
        result = render_interactive(
            spec,
            data,
            viewport=viewport,
            theme=theme,
            chart_config=chart_config or None,
        )
        scene = json.loads(result[0] if isinstance(result, tuple) else result)
        raw_nodes = _find_raw_nodes(scene)
        svgs = [n.get("svg", "") for n in raw_nodes]

        # Precondition: two distinct fragments must each author ferrum-colorbar-0.
        colliding = [s for s in svgs if 'id="ferrum-colorbar-0"' in s]
        assert len(colliding) >= 2, (
            "fixture must produce two Raw fragments that each author "
            f"id='ferrum-colorbar-0' (outer + inset colorbars); found {len(colliding)}"
        )

        load_idx = 0

        def _build_id_map(fragment: str, prefix: str) -> dict[str, str]:
            """Mirror of JS _buildIdMap (per fragment)."""
            id_map: dict[str, str] = {}
            for id_val in re.findall(r'\bid="([^"]+)"', fragment):
                id_map.setdefault(id_val, f"{prefix}{id_val}")
            return id_map

        def _apply_id_map(svg: str, id_map: dict[str, str]) -> str:
            """Mirror of JS _applyIdMap."""
            result = svg
            for id_val, namespaced in id_map.items():
                esc = re.escape(id_val)
                result = re.sub(rf'\bid="{esc}"', f'id="{namespaced}"', result)
                result = re.sub(rf"url\(#{esc}\)", f"url(#{namespaced})", result)
                result = re.sub(rf'\bhref="#{esc}"', f'href="#{namespaced}"', result)
                result = re.sub(rf'\bxlink:href="#{esc}"', f'xlink:href="#{namespaced}"', result)
            return result

        # ── NEW per-fragment namespacing ─────────────────────────────────────
        new_rewritten = [
            _apply_id_map(s, _build_id_map(s, f"ferrum-raw-{load_idx}-{i}-"))
            for i, s in enumerate(svgs)
        ]
        new_defined: set[str] = set()
        for frag in new_rewritten:
            new_defined.update(re.findall(r'\bid="([^"]+)"', frag))

        # The two colliding colorbars must now carry DISTINCT namespaced ids.
        new_colorbar_ids = sorted(i for i in new_defined if i.endswith("-ferrum-colorbar-0"))
        assert len(new_colorbar_ids) >= 2, (
            "per-fragment namespacing must give each colorbar a distinct id; "
            f"got {new_colorbar_ids!r}"
        )
        assert len(new_colorbar_ids) == len(set(new_colorbar_ids))

        # No url(#X)/href references in any fragment may dangle after namespacing.
        for frag in new_rewritten:
            refs = set(re.findall(r"url\(#([^)]+)\)", frag))
            refs |= set(re.findall(r'\bhref="#([^"]+)"', frag))
            refs |= set(re.findall(r'\bxlink:href="#([^"]+)"', frag))
            dangling = {r for r in refs if "ferrum-raw-" in r} - new_defined
            assert dangling == set(), (
                f"per-fragment namespacing left dangling references: {dangling!r}"
            )

        # ── OLD shared-map namespacing collapses the two colorbars (R5 bug) ───
        shared_map: dict[str, str] = {}
        for frag in svgs:
            for id_val in re.findall(r'\bid="([^"]+)"', frag):
                shared_map.setdefault(id_val, f"ferrum-raw-{load_idx}-{id_val}")
        old_rewritten = [_apply_id_map(s, shared_map) for s in svgs]
        old_defined: set[str] = set()
        for frag in old_rewritten:
            old_defined.update(re.findall(r'\bid="([^"]+)"', frag))
        old_colorbar_ids = sorted(i for i in old_defined if i.endswith("ferrum-colorbar-0"))
        assert old_colorbar_ids == ["ferrum-raw-0-ferrum-colorbar-0"], (
            "OLD shared-map logic must collapse both ferrum-colorbar-0 fragments "
            f"to ONE namespaced id (the R5 bug); got {old_colorbar_ids!r}"
        )


# ── Issue #34: composed interactive offsets raw nodes in right-child ──────────
#
# Regression for: data-anchored raw nodes (e.g. <image> annotations) on the
# right-hand child of a horizontal concat were left at child-local coords.
# Fixed in _scene_merge._offset_node: the raw handler wraps each fragment in
# `<g transform="translate(dx,dy)">` so the node rides along with its panel.


def test_composed_interactive_offsets_data_anchored_image_raw_node():
    """Regression (issue #34): in a composed interactive render, a data-anchored
    <image> annotation on the RIGHT concat child must be offset into that child's
    placement via a translate-group wrapper, not left at child-local coords.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 3.0]})
    png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4"
        "2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q") + ann.image(
        2.0, 3.0, png, width=24, height=24, anchor="center"
    )
    scene = json.loads((left | right).interactive().scene_json)

    # Collect every raw node in the merged scene.
    raws = []

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "raw":
                raws.append(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(scene)

    image_raws = [r for r in raws if "<image" in r.get("svg", "")]
    assert image_raws, "expected a data-anchored <image> raw node in the merged scene"
    svg = image_raws[0]["svg"]
    # The right child is horizontally offset, so the fragment must be wrapped in a
    # non-zero translate group, with the inner <image> preserved.
    assert svg.startswith('<g transform="translate('), (
        f"image raw node was not offset-wrapped: {svg[:80]!r}"
    )
    inner_dx = float(svg[len('<g transform="translate(') :].split(",")[0])
    assert inner_dx > 0.0, f"expected positive dx for the right child, got {inner_dx}"
    assert "<image" in svg.split(">", 1)[1], (
        "inner <image> element must be preserved inside the wrapper"
    )
