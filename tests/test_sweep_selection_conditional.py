"""Combinatorial sweep: selection x conditional x composition.

Covers the cross-cutting matrix of selection types, conditional encodings,
and composition primitives to ensure no crashes and correct scene JSON
structure across all combinations.

Test groups:
  A  — Scene JSON validity (no-crash + structure)
  B  — Selection serialization
  C  — Conditional encoding in composed scenes
  D  — Cross-panel field-value matching
  E  — Edge cases
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import render_interactive
from ferrum._interactive import InteractiveChart, _render_scene
from ferrum.selection import ConditionalSpec, selection_interval, selection_point, value


# ── helpers ──────────────────────────────────────────────────────────────────


def _render(chart: fm.Chart) -> tuple[str, bytes]:
    """Render a chart to (scene_json_str, packed_bytes)."""
    spec, data, viewport, theme, _ = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme)


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0],
            "group": ["a", "b", "a", "b", "a"],
        }
    )


def _make_chart(
    df: pl.DataFrame,
    *,
    sel=None,
    cond=None,
    width: int = 200,
    height: int = 200,
) -> fm.Chart:
    """Build a point chart, optionally adding selection and conditional."""
    chart = (
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=width, height=height)
    )
    if sel is not None:
        chart = chart.add_selection(sel)
    if cond is not None:
        chart = chart.conditional(cond)
    return chart


# ── Group A: Scene JSON validity ─────────────────────────────────────────────

# Axis 1: selection types (including None)
_SEL_TYPES = ["point_no_fields", "point_with_fields", "interval", "none"]

# Axis 3: composition types
_COMP_TYPES = ["single", "hconcat", "vconcat", "layer"]


def _make_selection(sel_type: str, name: str = "sweep_sel"):
    """Create a selection from a type key, or None."""
    if sel_type == "point_no_fields":
        return selection_point(name=name)
    if sel_type == "point_with_fields":
        return selection_point(fields=["group"], name=name)
    if sel_type == "interval":
        return selection_interval(name=name)
    return None


def _make_composed(df, sel, comp_type: str):
    """Create a composed chart (or single chart) with optional selection."""
    c1 = _make_chart(df, sel=sel)
    c2 = _make_chart(df, sel=sel)
    if comp_type == "single":
        return c1
    if comp_type == "hconcat":
        return c1 | c2
    if comp_type == "vconcat":
        return c1 & c2
    if comp_type == "layer":
        return fm.LayerChart(c1, c2)
    raise ValueError(f"unknown comp_type: {comp_type}")


_A_COMBOS = [(sel_type, comp_type) for sel_type in _SEL_TYPES for comp_type in _COMP_TYPES]


@pytest.mark.parametrize("sel_type,comp_type", _A_COMBOS, ids=[f"{s}-{c}" for s, c in _A_COMBOS])
def test_a_scene_validity(df, sel_type, comp_type):
    """_render_scene succeeds and produces valid scene JSON structure."""
    sel = _make_selection(sel_type)
    composed = _make_composed(df, sel, comp_type)
    scene_json, packed = _render_scene(composed)

    assert isinstance(scene_json, str)
    assert isinstance(packed, bytes)

    scene = json.loads(scene_json)
    for key in ("panels", "width", "height"):
        assert key in scene, f"top-level key {key!r} missing from scene JSON"

    # interaction key may be absent for plain Chart scenes (comes from Rust),
    # but must be present for compositions.
    if comp_type != "single":
        assert "interaction" in scene
        interaction = scene["interaction"]
        for ikey in ("zoom_enabled", "pan_enabled", "conditionals", "tick_levels"):
            assert ikey in interaction, f"interaction key {ikey!r} missing"

    # Panel count check.
    panels = scene["panels"]
    if comp_type in ("single", "layer"):
        assert len(panels) == 1, f"expected 1 panel for {comp_type}; got {len(panels)}"
    else:
        assert len(panels) == 2, f"expected 2 panels for {comp_type}; got {len(panels)}"


@pytest.mark.parametrize("comp_type", ["hconcat", "vconcat"])
def test_a_panel_id_reindexing(df, comp_type):
    """Composed scenes must have correctly re-indexed panel IDs (0, 1)."""
    sel = selection_point(name="id_test")
    composed = _make_composed(df, sel, comp_type)
    scene_json, _ = _render_scene(composed)
    scene = json.loads(scene_json)
    panel_ids = [p["id"] for p in scene["panels"]]
    assert panel_ids == [0, 1], f"panel IDs must be [0, 1]; got {panel_ids}"


# ── Group B: Selection serialization ─────────────────────────────────────────


def test_b_point_selection_type_in_scene(df):
    """Point selection serializes with type='point' in scene JSON."""
    sel = selection_point(name="b_pt")
    chart = _make_chart(df, sel=sel)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    selections = scene.get("selections", [])
    assert any(s["type"] == "point" for s in selections), "must have point selection"
    assert any(s["name"] == "b_pt" for s in selections), "must have selection named b_pt"


def test_b_interval_selection_type_in_scene(df):
    """Interval selection serializes with type='interval' in scene JSON."""
    sel = selection_interval(name="b_iv")
    chart = _make_chart(df, sel=sel)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    selections = scene.get("selections", [])
    assert any(s["type"] == "interval" for s in selections), "must have interval selection"
    assert any(s["name"] == "b_iv" for s in selections), "must have selection named b_iv"


def test_b_field_based_selection_tooltip_injection(df):
    """Field-based selection auto-injects the field into mark tooltips."""
    sel = selection_point(fields=["group"], name="b_field")
    chart = _make_chart(df, sel=sel)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    panel = scene["panels"][0]
    found_group = False
    for batch in panel["marks"]:
        tooltips = batch.get("tooltips")
        if tooltips is not None:
            for tip in tooltips:
                field_names = [f["name"] for f in tip.get("fields", [])]
                if "group" in field_names:
                    found_group = True
                    break
    assert found_group, "selection field 'group' must appear in mark tooltips"


def test_b_both_point_and_interval_present(df):
    """Charts with both point and interval selections serialize both."""
    sel_pt = selection_point(name="b_both_pt")
    sel_iv = selection_interval(name="b_both_iv")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel_pt, sel_iv)
        .properties(width=200, height=200)
    )
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)
    selections = scene.get("selections", [])
    assert len(selections) == 2, f"expected 2 selections; got {len(selections)}"
    types = {s["type"] for s in selections}
    assert types == {"point", "interval"}, f"expected {{point, interval}}; got {types}"


# ── Group C: Conditional encoding in composed scenes ─────────────────────────


def test_c_hconcat_point_color_conditional_both_children(df):
    """HConcat with point selection + color conditional on both children
    produces merged scene with conditionals from both."""
    sel = selection_point(fields=["group"], name="c_hc_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    left = _make_chart(df, sel=sel, cond=cond)
    right = _make_chart(df, sel=sel, cond=cond)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 2, (
        f"expected 2 conditionals (one per child); got {len(conditionals)}"
    )
    for c in conditionals:
        assert c["selection_name"] == "c_hc_sel"
        assert c["channel"] == "color"


def test_c_vconcat_interval_color_conditional(df):
    """VConcat with interval selection + color conditional produces
    conditionals with correct selection_name."""
    sel = selection_interval(name="c_vc_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    top = _make_chart(df, sel=sel, cond=cond)
    bot = _make_chart(df, sel=sel, cond=cond)
    composed = top & bot
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) >= 1, "must have at least 1 conditional"
    for c in conditionals:
        assert c["selection_name"] == "c_vc_sel"


def test_c_layer_with_selection(df):
    """LayerChart with selection produces single panel with selection present."""
    sel = selection_point(fields=["group"], name="c_layer_sel")
    c1 = _make_chart(df, sel=sel)
    c2 = _make_chart(df, sel=sel)
    layer = fm.LayerChart(c1, c2)
    scene_json, _ = _render_scene(layer)
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 1, "LayerChart must produce exactly 1 panel"
    selections = scene.get("selections", [])
    assert any(s["name"] == "c_layer_sel" for s in selections), (
        "selection must be present in layered scene"
    )


def test_c_hconcat_opacity_conditional(df):
    """HConcat with opacity conditional produces conditionals with channel='opacity'."""
    sel = selection_point(fields=["group"], name="c_op_sel")
    cond = sel.when(fm.Opacity("group")).otherwise(value(0.2))
    left = _make_chart(df, sel=sel, cond=cond)
    right = _make_chart(df, sel=sel, cond=cond)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 2
    for c in conditionals:
        assert c["channel"] == "opacity", f"expected opacity channel; got {c['channel']}"


# ── Group D: Cross-panel field-value matching ────────────────────────────────


def test_d_cross_panel_shared_selection(df):
    """Two charts composed with | sharing selection_point(fields=['group'])
    produce correct merged scene structure."""
    sel = selection_point(fields=["group"], name="d_shared")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    left = _make_chart(df, sel=sel, cond=cond)
    right = _make_chart(df, sel=sel, cond=cond)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)

    # 2 panels
    assert len(scene["panels"]) == 2

    # Tooltips on both panels' marks contain the "group" field
    for panel_idx, panel in enumerate(scene["panels"]):
        found_group = False
        for batch in panel["marks"]:
            tooltips = batch.get("tooltips")
            if tooltips is not None:
                for tip in tooltips:
                    field_names = [f["name"] for f in tip.get("fields", [])]
                    if "group" in field_names:
                        found_group = True
                        break
        assert found_group, (
            f"panel {panel_idx}: tooltip must contain 'group' field for cross-panel matching"
        )

    # Selection in the selections array
    selections = scene.get("selections", [])
    assert any(s["name"] == "d_shared" for s in selections)

    # Conditional in the conditionals array
    conditionals = scene["interaction"]["conditionals"]
    assert any(c["selection_name"] == "d_shared" for c in conditionals)


def test_d_cross_panel_vconcat_shared_selection(df):
    """Two charts composed with & sharing the same field-based selection."""
    sel = selection_point(fields=["group"], name="d_v_shared")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    top = _make_chart(df, sel=sel, cond=cond)
    bot = _make_chart(df, sel=sel, cond=cond)
    composed = top & bot
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)

    assert len(scene["panels"]) == 2
    selections = scene.get("selections", [])
    assert len(selections) == 2, f"expected 2 selections; got {len(selections)}"
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 2, f"expected 2 conditionals; got {len(conditionals)}"


# ── Group E: Edge cases ─────────────────────────────────────────────────────


def test_e_selection_without_conditional(df):
    """Chart with selection but no conditional: scene valid, no conditionals."""
    sel = selection_point(fields=["group"], name="e_no_cond")
    chart = _make_chart(df, sel=sel)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)

    # Scene is valid
    assert "panels" in scene
    assert len(scene["panels"]) == 1

    # Selections present
    selections = scene.get("selections", [])
    assert any(s["name"] == "e_no_cond" for s in selections)

    # No conditionals
    interaction = scene.get("interaction", {})
    conditionals = interaction.get("conditionals", [])
    assert len(conditionals) == 0, f"expected no conditionals; got {len(conditionals)}"


def test_e_interval_empty_fields_index_fallback(df):
    """Interval selection with no fields falls back to index-based matching
    (no tooltip auto-injection)."""
    sel = selection_interval(name="e_iv_no_fields")
    chart = _make_chart(df, sel=sel)
    scene_json, _ = _render(chart)
    scene = json.loads(scene_json)

    # No tooltip auto-injection for interval without fields
    panel = scene["panels"][0]
    for batch in panel["marks"]:
        tooltips = batch.get("tooltips")
        assert tooltips is None, (
            f"interval selection without fields must not produce tooltips; got {tooltips!r}"
        )


def test_e_one_child_has_selection(df):
    """Composition where only one child has selection: selection appears in merged scene."""
    sel = selection_point(name="e_one_child")
    left = _make_chart(df, sel=sel)
    right = _make_chart(df)  # No selection
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)

    selections = scene.get("selections", [])
    assert any(s["name"] == "e_one_child" for s in selections), (
        "selection from one child must appear in merged scene"
    )


def test_e_empty_dataframe_with_selection():
    """Empty DataFrame with selection does not crash."""
    empty_df = pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
            "group": pl.Series([], dtype=pl.Utf8),
        }
    )
    sel = selection_point(fields=["group"], name="e_empty")
    chart = (
        fm.Chart(empty_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    scene_json, packed = _render_scene(chart)
    scene = json.loads(scene_json)
    assert "panels" in scene
    assert isinstance(scene["panels"], list)


def test_e_composition_interaction_keys_present(df):
    """Merged composition has zoom_enabled, pan_enabled, conditionals, tick_levels."""
    left = _make_chart(df)
    right = _make_chart(df)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    interaction = scene["interaction"]
    assert "zoom_enabled" in interaction
    assert "pan_enabled" in interaction
    assert "conditionals" in interaction
    assert "tick_levels" in interaction
    assert isinstance(interaction["tick_levels"], list)


def test_e_conditional_without_matching_selection_raises(df):
    """Chart.conditional() raises ValueError when the selection truly is not
    available: the spec carries no source Selection and none is registered.

    A spec built via ``sel.when(...).otherwise(...)`` carries its source
    Selection, so conditional() auto-registers it; this test exercises the
    genuinely-unavailable path with a bare ConditionalSpec.
    """
    cond = ConditionalSpec(
        selection_name="e_unreg",
        if_selected=fm.Color("group"),
        if_not=value("#cccccc"),
    )
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    with pytest.raises(ValueError, match="no selection named"):
        chart.conditional(cond)


def test_e_composed_empty_and_nonempty(df):
    """Composing an empty-data chart with a normal chart does not crash."""
    empty_df = pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    left = (
        fm.Chart(empty_df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    )
    right = _make_chart(df)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    assert "panels" in scene
    # At least one panel from the non-empty chart
    assert len(scene["panels"]) >= 1


def test_e_hconcat_no_selection_no_conditional(df):
    """HConcat with no selection or conditional produces valid merged scene
    with empty selections and conditionals."""
    left = _make_chart(df)
    right = _make_chart(df)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 2
    assert scene.get("selections", []) == []
    conditionals = scene["interaction"]["conditionals"]
    assert conditionals == []


def test_e_interactive_chart_from_composition_with_selection(df):
    """InteractiveChart from a composition with selection produces valid scene_json."""
    sel = selection_point(fields=["group"], name="e_ic_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    left = _make_chart(df, sel=sel, cond=cond)
    right = _make_chart(df, sel=sel, cond=cond)
    composed = left | right
    ic = composed.interactive()
    assert isinstance(ic, InteractiveChart)
    scene = json.loads(ic.scene_json)
    assert len(scene["panels"]) == 2
    assert len(scene.get("selections", [])) >= 1
    assert len(scene["interaction"]["conditionals"]) >= 1
