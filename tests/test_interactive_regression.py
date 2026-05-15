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
