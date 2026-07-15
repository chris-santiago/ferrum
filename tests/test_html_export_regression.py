"""Regression tests for the HTML export pipeline.

Covers:
- Packed data contract: large charts produce non-empty packed bytes (P1)
- Small charts produce empty packed data (P2)
- Scene JSON round-trip structure (P3)
- Theme background preserved in scene JSON (P4)
- Composition `.interactive()` returns InteractiveChart (P5)
- Composition `.save("out.html")` produces valid HTML (P6)
- Composition `to_svg()` unchanged (P7)
- InteractiveChart preserves packed data (P8)
- Selection spec serialization round-trip (P9)
- Conditional encoding in scene JSON (P10)
- `_render_scene` returns tuple (P11)
- HTML export JS has no `model.get` / `model.set` calls (P12)
- Text elements in scene JSON (P13)
- Generated HTML contains D3 bundle (P14)
- No hand-written interaction state in HTML (P15)
- No CSS-div text overlay in HTML (P16)
- SVG text rendering present in HTML (P17)
- Auto-inject selection fields into tooltips (P18)
- Auto-inject merges with explicit tooltip (P19)
- Interval selection no fields → no tooltip injection (P20)
- Multi-field selection auto-injection (P21)
- LayerChart .interactive() returns InteractiveChart (P22)
- HConcatChart .interactive() correct x-offsets (P23)
- VConcatChart .interactive() correct y-offsets (P24)
- Composition to_svg() identical before/after .interactive() (P25)
- Composition .save(format="html") via _ChartLike.save() (P26)
- Both point and interval selections in scene JSON (P27)
- Conditional encoding structure in scene JSON (P28)
- Merged composition interaction keys (P29)
- HTML with selection_point contains handleClick (P30)
- HTML with selection_interval contains brush (P31)
- HTML contains setTransform (P32)
- HTML contains @font-face with Inter (P33)
- HTML background matches scene JSON background (P34)
- HTML does NOT contain _ensureWasm (P35)
- 999 marks → empty packed data (P36)
- 1001 marks → non-empty packed data (P37)
- Cross-panel composed selections and conditionals (P38)
- Merged composition panel ID re-indexing (P39)
- Empty DataFrame .interactive() (P40)
- No encoding .interactive() raises cleanly (P41)
- InteractiveChart.save() with .html extension (P42)
- Chart.__add__ preserves RHS selections and conditionals (B3)
- LayerChart._render_interactive preserves both layers' selections (B3)
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import render_interactive
from ferrum._interactive import InteractiveChart, _render_scene
from ferrum.selection import selection_interval, selection_point, value


# ── helpers ──────────────────────────────────────────────────────────────────


def _render(chart: fm.Chart) -> tuple[str, bytes]:
    """Render a chart to (scene_json_str, packed_bytes)."""
    spec, data, viewport, theme, _ = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme)


# ── P1. HTML export includes packed data for large charts ────────────────────


def test_p1_large_chart_produces_nonempty_packed_data():
    """Large charts (>1000 marks) must produce non-empty packed bytes from
    render_interactive.  This locks the contract that packed data exists
    for the HTML export pipeline to consume."""
    df = pl.DataFrame({"x": list(range(1500)), "y": list(range(1500))})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    json_str, packed = _render(chart)
    assert isinstance(packed, bytes)
    assert len(packed) > 0, "packed data must be non-empty for >1000 marks"
    # Scene JSON must still be valid
    scene = json.loads(json_str)
    assert "panels" in scene


# ── P2. HTML export with small chart has empty packed data ───────────────────


def test_p2_small_chart_produces_empty_packed_data():
    """Small charts (<1000 marks) must produce empty packed bytes.  Per-node
    JSON is retained for JS hit-testing and tooltip lookup."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    _, packed = _render(chart)
    assert isinstance(packed, bytes)
    assert len(packed) == 0, "packed data must be empty for small charts"


# ── P3. Scene JSON round-trip structure ──────────────────────────────────────


def test_p3_scene_json_has_required_top_level_keys():
    """Scene JSON from render_interactive must have panels, width, height,
    and interaction at the top level.  Each panel must have marks and
    plot_area."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    for key in ("panels", "width", "height", "interaction"):
        assert key in scene, f"top-level key {key!r} missing from scene JSON"

    panels = scene["panels"]
    assert isinstance(panels, list)
    assert len(panels) >= 1, "scene must have at least 1 panel"

    for panel in panels:
        assert "marks" in panel, "panel missing 'marks' key"
        assert "plot_area" in panel, "panel missing 'plot_area' key"


# ── P4. Theme background preserved in scene JSON ────────────────────────────


def test_p4_theme_background_preserved_in_scene_json():
    """A custom theme background must appear as RGBA in scene['background'].
    Steelblue is #4682B4 -> (r=70, g=130, b=180, a=255)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .theme(fm.Theme(background="#4682b4"))
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    bg = scene.get("background")
    assert bg is not None, "background must be present in scene JSON when theme sets it"
    assert bg["r"] == 70, f"steelblue red should be 70; got {bg['r']}"
    assert bg["g"] == 130, f"steelblue green should be 130; got {bg['g']}"
    assert bg["b"] == 180, f"steelblue blue should be 180; got {bg['b']}"
    assert bg["a"] == 255, f"alpha should be 255; got {bg['a']}"


# ── P5. Composition .interactive() returns InteractiveChart ──────────────────


def test_p5_composition_interactive_returns_interactive_chart():
    """Composing two charts with | and calling .interactive() should return
    an InteractiveChart instance."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    ic = composed.interactive()
    assert isinstance(ic, InteractiveChart)


# ── P6. Composition .save("out.html") produces valid HTML ────────────────────


def test_p6_composition_save_html_produces_valid_file(tmp_path):
    """Composing two charts with | and saving as HTML via .interactive().save()
    should produce a file containing 'loadScene'."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    out = tmp_path / "comp.html"
    composed.interactive().save(str(out))
    assert out.exists(), "HTML file must be written"
    content = out.read_text()
    assert "loadScene" in content, "HTML must contain loadScene call"


# ── P7. Composition to_svg() unchanged ───────────────────────────────────────


def test_p7_composition_to_svg_unchanged():
    """Composing two charts with | and calling to_svg() must return valid
    SVG markup.  This locks that the SVG path is not broken by interactive
    changes."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    svg = composed.to_svg()
    assert isinstance(svg, str)
    assert "<svg" in svg, "to_svg() must return SVG markup"


# ── P8. InteractiveChart preserves packed data ───────────────────────────────


def test_p8_interactive_chart_preserves_packed_data():
    """InteractiveChart._packed_data must be bytes with non-zero length for
    charts above the packing threshold (>1000 marks)."""
    df = pl.DataFrame({"x": list(range(1500)), "y": list(range(1500))})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    ic = InteractiveChart(chart)
    assert isinstance(ic._packed_data, bytes)
    assert len(ic._packed_data) > 0, "InteractiveChart must preserve non-empty packed data"


# ── P9. Selection spec serialization round-trip ──────────────────────────────


def test_p9_selection_spec_serialization_round_trip():
    """A chart with both a point and interval selection must serialize both
    into scene['selections'] with correct types."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel_pt = selection_point(fields=["group"], name="pt_sel")
    sel_iv = selection_interval(name="iv_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="group:N")
        .add_selection(sel_pt, sel_iv)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    selections = scene.get("selections", [])
    assert isinstance(selections, list)
    assert len(selections) == 2, f"expected 2 selections; got {len(selections)}"

    types = {s["type"] for s in selections}
    assert "point" in types, "must have a point selection"
    assert "interval" in types, "must have an interval selection"

    names = {s["name"] for s in selections}
    assert "pt_sel" in names
    assert "iv_sel" in names


# ── P10. Conditional encoding in scene JSON ──────────────────────────────────


def test_p10_conditional_encoding_in_scene_json():
    """A chart with a conditional encoding must serialize it into
    scene['interaction']['conditionals'] with the expected keys."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="cond_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .conditional(cond)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    conditionals = scene["interaction"]["conditionals"]
    assert isinstance(conditionals, list)
    assert len(conditionals) >= 1, "conditionals must be non-empty"

    first = conditionals[0]
    assert first["selection_name"] == "cond_sel"
    for key in ("channel", "if_selected", "if_not"):
        assert key in first, f"conditional missing key {key!r}"


# ── P11. _render_scene returns tuple ─────────────────────────────────────────


def test_p11_render_scene_returns_tuple():
    """_render_scene must return a tuple of (str, bytes)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    result = _render_scene(chart)
    assert isinstance(result, tuple), f"expected tuple; got {type(result).__name__}"
    assert len(result) == 2, f"expected 2-tuple; got {len(result)}-tuple"
    scene_json, packed = result
    assert isinstance(scene_json, str), "first element must be str (scene JSON)"
    assert isinstance(packed, bytes), "second element must be bytes (packed data)"


# ── P12. HTML export JS has no model.get / model.set calls ───────────────────


def test_p12_html_template_has_no_model_get_set():
    """The self-contained HTML template in _html.py must NOT contain
    model.get or model.set -- those are anywidget patterns that don't
    apply to standalone HTML exports."""
    import inspect

    from ferrum._html import assemble_html

    source = inspect.getsource(assemble_html)
    assert "model.get" not in source, "HTML template must not contain model.get"
    assert "model.set" not in source, "HTML template must not contain model.set"


# ── P13. Text elements in scene JSON ─────────────────────────────────────────


def test_p13_text_elements_in_scene_json():
    """A chart with x/y encodings must produce text nodes in the axes array
    (tick labels, axis titles) with positional and styling information."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    axes = scene["panels"][0].get("axes", [])
    text_nodes = [n for n in axes if n.get("type") == "text"]
    assert len(text_nodes) > 0, "axes must contain text nodes (tick labels)"

    # Each text node must have position (x, y), content, and style
    for node in text_nodes:
        assert "x" in node, "text node must have x position"
        assert "y" in node, "text node must have y position"
        assert "content" in node, "text node must have content"
        assert "style" in node, "text node must have style"
        style = node["style"]
        assert "font_size" in style, "text node style must have font_size"
        assert "color" in style, "text node style must have color"


# ── P14. Generated HTML contains D3 bundle ──────────────────────────────


def test_p14_html_contains_d3_bundle(tmp_path):
    """The generated HTML must contain the vendored D3 bundle with brush
    and zoom functionality."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "d3_test.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "zoom" in content, "HTML must contain D3 zoom code"
    assert "brush" in content, "HTML must contain D3 brush code"


# ── P15. No hand-written interaction state in HTML ──────────────────────


def test_p15_no_handwritten_interaction_state(tmp_path):
    """The generated HTML must NOT contain hand-written interaction state
    variables that were replaced by D3."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "d3_test.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "_panStart" not in content, "HTML must not contain _panStart"
    assert "_brushOrigin" not in content, "HTML must not contain _brushOrigin"
    assert "_isBrushing" not in content, "HTML must not contain _isBrushing"


# ── P16. No CSS-div text overlay in HTML ────────────────────────────────


def test_p16_no_css_div_text_overlay(tmp_path):
    """The generated HTML must NOT contain the old ferrum-overlay div or
    _placeText function — replaced by SVG text rendering."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "d3_test.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "ferrum-overlay" not in content, "HTML must not contain ferrum-overlay"


# ── P17. SVG text rendering present in HTML ─────────────────────────────


def test_p17_svg_text_rendering_present(tmp_path):
    """The generated HTML must contain SVG text rendering via the
    ferrum-label class."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "d3_test.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "ferrum-label" in content, "HTML must contain ferrum-label class for SVG text"
    assert "createElementNS" in content or "_placeTextSvg" in content, (
        "HTML must contain SVG element creation"
    )


# ── P18. Auto-inject selection fields into tooltips (no explicit tooltip) ──


def test_p18_selection_fields_auto_injected_into_tooltips():
    """A chart with selection_point(fields=["group"]) and NO explicit tooltip
    encoding must auto-inject the "group" field into the scene JSON marks'
    tooltips so cross-panel linked selection can match by field values."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="p18_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)
    panel = scene["panels"][0]
    # Find tooltips in mark batches
    found_group = False
    for batch in panel["marks"]:
        tooltips = batch.get("tooltips")
        if tooltips is not None:
            for tip in tooltips:
                field_names = [f["name"] for f in tip.get("fields", [])]
                if "group" in field_names:
                    found_group = True
                    break
    assert found_group, "auto-injected selection field 'group' must appear in mark tooltips"


# ── P19. Auto-inject merges with explicit tooltip ──────────────────────────


def test_p19_selection_fields_merge_with_explicit_tooltip():
    """A chart with selection_point(fields=["group"]) AND explicit tooltip="x"
    must produce tooltips containing BOTH "x" and "group" fields."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="p19_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", tooltip="x:Q")
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)
    panel = scene["panels"][0]
    for batch in panel["marks"]:
        tooltips = batch.get("tooltips")
        if tooltips is not None and len(tooltips) > 0:
            first_fields = {f["name"] for f in tooltips[0].get("fields", [])}
            assert "x" in first_fields, "explicit tooltip field 'x' must be present"
            assert "group" in first_fields, "auto-injected selection field 'group' must be present"
            return
    pytest.fail("no tooltips found in mark batches")


# ── P20. Interval selection with no fields — no auto-injection ─────────────


def test_p20_interval_selection_no_fields_no_tooltip_injection():
    """A chart with selection_interval() (no fields) must not auto-inject
    tooltip fields.  Tooltips should be absent unless explicitly set."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_interval(name="p20_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)
    panel = scene["panels"][0]
    for batch in panel["marks"]:
        tooltips = batch.get("tooltips")
        assert tooltips is None, (
            f"interval selection without fields must not produce tooltips; got {tooltips!r}"
        )


# ── P21. Multi-field selection auto-injection ──────────────────────────────


def test_p21_multi_field_selection_auto_injection():
    """A chart with selection_point(fields=["a", "b"]) must auto-inject both
    "a" and "b" into the mark tooltips."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "a": ["A", "B"],
            "b": ["C", "D"],
        }
    )
    sel = selection_point(fields=["a", "b"], name="p21_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)
    panel = scene["panels"][0]
    for batch in panel["marks"]:
        tooltips = batch.get("tooltips")
        if tooltips is not None and len(tooltips) > 0:
            first_fields = {f["name"] for f in tooltips[0].get("fields", [])}
            assert "a" in first_fields, "selection field 'a' must be in tooltips"
            assert "b" in first_fields, "selection field 'b' must be in tooltips"
            return
    pytest.fail("no tooltips found in mark batches")


# ── P22. LayerChart .interactive() returns InteractiveChart ────────────────


def test_p22_layer_chart_interactive_returns_interactive_chart():
    """LayerChart .interactive() must return an InteractiveChart and produce
    valid scene JSON with 1 panel (layers merge into a single plot area)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    c1 = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    c2 = fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    layer = fm.LayerChart(c1, c2)
    ic = layer.interactive()
    assert isinstance(ic, InteractiveChart), "LayerChart.interactive() must return InteractiveChart"
    scene = json.loads(ic.scene_json)
    assert "panels" in scene
    assert len(scene["panels"]) == 1, "LayerChart must produce exactly 1 panel"


# ── P23. HConcatChart .interactive() produces correct x-offsets ────────────


def test_p23_hconcat_interactive_correct_x_offsets():
    """HConcatChart .interactive() must produce scene JSON with 2 panels
    where panel 1's plot_area.x > panel 0's (plot_area.x + plot_area.w)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    ic = composed.interactive()
    scene = json.loads(ic.scene_json)
    panels = scene["panels"]
    assert len(panels) == 2, f"expected 2 panels; got {len(panels)}"

    pa0 = panels[0]["plot_area"]
    pa1 = panels[1]["plot_area"]
    # Panel 1 must start to the right of panel 0's extent
    assert pa1["x"] > pa0["x"] + pa0["w"], (
        f"panel 1 x-offset ({pa1['x']}) must be > panel 0 right edge ({pa0['x'] + pa0['w']})"
    )


# ── P24. VConcatChart .interactive() produces correct y-offsets ────────────


def test_p24_vconcat_interactive_correct_y_offsets():
    """VConcatChart .interactive() must produce scene JSON with 2 panels
    where panel 1's plot_area.y > panel 0's (plot_area.y + plot_area.h)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    top = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    bot = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = top & bot
    ic = composed.interactive()
    scene = json.loads(ic.scene_json)
    panels = scene["panels"]
    assert len(panels) == 2, f"expected 2 panels; got {len(panels)}"

    pa0 = panels[0]["plot_area"]
    pa1 = panels[1]["plot_area"]
    # Panel 1 must start below panel 0's extent
    assert pa1["y"] > pa0["y"] + pa0["h"], (
        f"panel 1 y-offset ({pa1['y']}) must be > panel 0 bottom edge ({pa0['y'] + pa0['h']})"
    )


# ── P25. Composition to_svg() identical before/after .interactive() ────────


def test_p25_composition_svg_unaffected_by_interactive():
    """Calling .interactive() on a composition must not mutate the composition
    or change the SVG output from to_svg()."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    svg_before = composed.to_svg()
    _ = composed.interactive()
    svg_after = composed.to_svg()
    assert svg_before == svg_after, (
        "to_svg() must produce identical output before and after .interactive()"
    )


# ── P26. Composition .save(format="html") produces valid HTML ──────────────


def test_p26_composition_save_html_via_chartlike(tmp_path):
    """Composition .save("out.html", format="html") via the _ChartLike.save()
    path must produce valid HTML containing loadScene or _render."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    out = tmp_path / "comp_save.html"
    composed.save(str(out), format="html")
    assert out.exists(), "HTML file must be written"
    content = out.read_text()
    assert "<!DOCTYPE html>" in content, "must be a valid HTML document"
    assert "_render" in content or "loadScene" in content, (
        "HTML must contain _render or loadScene call"
    )


# ── P27. Both point AND interval selections in scene JSON ──────────────────


def test_p27_both_point_and_interval_selections():
    """A chart with both point and interval selections must serialize both
    into scene['selections'] with correct types."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel_pt = selection_point(fields=["group"], name="p27_pt")
    sel_iv = selection_interval(name="p27_iv")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="group:N")
        .add_selection(sel_pt, sel_iv)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    selections = scene.get("selections", [])
    assert len(selections) == 2, f"expected 2 selections; got {len(selections)}"
    types = {s["type"] for s in selections}
    assert "point" in types, "must have a point selection"
    assert "interval" in types, "must have an interval selection"


# ── P28. Conditional encoding structure in scene JSON ──────────────────────


def test_p28_conditional_encoding_structure():
    """A chart with a conditional encoding must produce an interaction.conditionals
    entry with selection_name, channel, if_selected, and if_not keys."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="p28_sel")
    cond = sel.when(fm.Color("group")).otherwise(value("#cccccc"))
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .conditional(cond)
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)

    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) >= 1, "must have at least 1 conditional"
    first = conditionals[0]
    assert first["selection_name"] == "p28_sel"
    assert "channel" in first
    assert "if_selected" in first
    assert "if_not" in first


# ── P29. Merged composition scene has required interaction keys ────────────


def test_p29_merged_composition_interaction_keys():
    """A merged composition scene must have zoom_enabled, pan_enabled, and
    tick_levels keys in the interaction dict — these are required by the
    WASM deserializer."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)

    interaction = scene["interaction"]
    assert "zoom_enabled" in interaction, "interaction must have zoom_enabled"
    assert "pan_enabled" in interaction, "interaction must have pan_enabled"
    assert "tick_levels" in interaction, "interaction must have tick_levels"
    assert isinstance(interaction["tick_levels"], list), "tick_levels must be a list"


# ── P30. HTML export with selection_point contains handleClick ─────────────


def test_p30_html_with_point_selection_contains_handleclick(tmp_path):
    """HTML export of a chart with selection_point must contain handleClick
    in the JS (the WASM click handler is wired)."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "group": ["A", "B"]})
    sel = selection_point(fields=["group"], name="p30_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    out = tmp_path / "p30.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "handleClick" in content, "HTML must contain handleClick for point selection"


# ── P31. HTML export with selection_interval contains brush ────────────────


def test_p31_html_with_interval_selection_contains_brush(tmp_path):
    """HTML export of a chart with selection_interval must contain 'brush'
    (D3-brush is wired)."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    sel = selection_interval(name="p31_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=300, height=200)
    )
    out = tmp_path / "p31.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "brush" in content, "HTML must contain 'brush' for interval selection"


# ── P32. HTML export contains setTransform ─────────────────────────────────


def test_p32_html_contains_set_transform(tmp_path):
    """HTML export must contain setTransform (D3-zoom to WASM bridge)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "p32.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "setTransform" in content, "HTML must contain setTransform for D3-zoom bridge"


# ── P33. HTML export contains @font-face with Inter ───────────────────────


def test_p33_html_contains_inter_font(tmp_path):
    """HTML export must contain @font-face with Inter font embedding."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "p33.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "@font-face" in content, "HTML must contain @font-face"
    assert "Inter" in content, "HTML must embed Inter font"


# ── P33b. Interactive @font-face covers bold weight (GH #80) ───────────────


def test_p33b_interactive_font_face_covers_bold_weight(tmp_path):
    """GH #80 regression: the interactive @font-face must cover bold (600)
    weights via the Inter *variable* font (``font-weight:100 900``).

    A single-weight (Regular/400) face forces the browser to synthesize
    faux-bold for the theme's 600-weight text (figure title, axis titles,
    legend title), which renders blurry — while normal-weight ticks/legend
    entries stay crisp. Guards against regressing to a single-weight face.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(title="Title")
    out = tmp_path / "gh80.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "font-weight:100 900" in content, (
        "interactive @font-face must span the variable weight range (covers 600), "
        "not a single 400 face — else bold text faux-bolds and renders blurry"
    )
    assert "data:font/woff2" in content, "variable Inter font must be embedded as woff2"


# ── P34. HTML export background matches scene JSON background ──────────────


def test_p34_html_background_matches_scene_json():
    """HTML export background style must match the scene JSON background
    color when a custom theme background is set."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .theme(fm.Theme(background="#ff0000"))
        .properties(width=300, height=200)
    )
    json_str, _ = _render(chart)
    scene = json.loads(json_str)
    bg = scene["background"]

    from ferrum._html import assemble_html

    html = assemble_html(json_str, packed_data=b"")
    expected_rgba = f"rgba({bg['r']},{bg['g']},{bg['b']},{bg['a'] / 255.0})"
    assert expected_rgba in html, (
        f"HTML background must contain {expected_rgba}; not found in output"
    )


# ── P35. HTML export does NOT contain _ensureWasm ──────────────────────────


def test_p35_html_no_ensure_wasm(tmp_path):
    """HTML export must NOT contain _ensureWasm — it is stripped for
    standalone use (WASM is initialized directly in main())."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "p35.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "_ensureWasm" not in content, "HTML must not contain _ensureWasm"


# ── P36. Chart with 999 marks produces empty packed data ──────────────────


def test_p36_999_marks_empty_packed_data():
    """A chart with exactly 999 marks must produce empty packed data
    (below the 1000-mark packing threshold)."""
    df = pl.DataFrame({"x": list(range(999)), "y": list(range(999))})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    _, packed = _render(chart)
    assert len(packed) == 0, f"999 marks must produce empty packed data; got {len(packed)} bytes"


# ── P37. Chart with 1001 marks produces non-empty packed data ─────────────


def test_p37_1001_marks_nonempty_packed_data():
    """A chart with exactly 1001 marks must produce non-empty packed data
    (above the 1000-mark packing threshold)."""
    df = pl.DataFrame({"x": list(range(1001)), "y": list(range(1001))})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    _, packed = _render(chart)
    assert len(packed) > 0, "1001 marks must produce non-empty packed data"


# ── P38. Cross-panel composed selections and conditionals ──────────────────


def test_p38_cross_panel_selections_and_conditionals():
    """Two charts composed with |, both with the same selection_point(fields=["cat"])
    and conditional, must produce a merged scene with selections and conditionals
    from both children."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "cat": ["A", "B", "C"]})
    sel = selection_point(fields=["cat"], name="shared_sel")
    cond = sel.when(fm.Color("cat")).otherwise(value("#cccccc"))
    left = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .conditional(cond)
        .properties(width=200, height=200)
    )
    right = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .conditional(cond)
        .properties(width=200, height=200)
    )
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)

    # Both children contribute selections
    selections = scene.get("selections", [])
    assert len(selections) == 2, f"expected 2 selections (one per child); got {len(selections)}"

    # Both children contribute conditionals
    conditionals = scene["interaction"]["conditionals"]
    assert len(conditionals) == 2, (
        f"expected 2 conditionals (one per child); got {len(conditionals)}"
    )


# ── P39. Merged composition panel ID re-indexing ──────────────────────────


def test_p39_merged_panel_id_reindexing():
    """Merged composition scene must have correct panel.id re-indexing:
    panel 0 → 0, panel 1 → 1 (not both 0)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)

    panel_ids = [p["id"] for p in scene["panels"]]
    assert panel_ids == [0, 1], f"panel IDs must be [0, 1]; got {panel_ids}"


# ── P40. Empty DataFrame chart .interactive() ──────────────────────────────


def test_p40_empty_dataframe_interactive():
    """An empty-DataFrame chart calling .interactive() must not crash and
    must produce valid scene JSON with no panels."""
    df = pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    ic = chart.interactive()
    assert isinstance(ic, InteractiveChart)
    scene = json.loads(ic.scene_json)
    assert "panels" in scene
    assert isinstance(scene["panels"], list)
    assert scene["width"] == 300
    assert scene["height"] == 200


# ── P41. Chart with no encoding raises cleanly ────────────────────────────


def test_p41_no_encoding_interactive_raises():
    """A chart with no encoding (no x, y, etc.) must raise a clear error
    when .interactive() is called, not an opaque crash."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().properties(width=300, height=200)
    with pytest.raises((ValueError, TypeError)):
        chart.interactive()


# ── P42. InteractiveChart.save() with .html extension ──────────────────────


def test_p42_interactive_chart_save_html(tmp_path):
    """InteractiveChart.save() to a path with .html extension must produce
    a valid HTML file."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    ic = InteractiveChart(chart)
    out = tmp_path / "chart_out.html"
    ic.save(str(out))
    assert out.exists(), "HTML file must be written"
    content = out.read_text()
    assert "<!DOCTYPE html>" in content, "must be a valid HTML document"
    assert "_render" in content, "HTML must contain _render call"


# ── B3. LayerChart.__add__ drops RHS selections and conditionals ──────────


def test_layer_add_preserves_rhs_selections_and_conditionals():
    """Chart.__add__ (used by LayerChart._build_merged) must preserve
    selections and conditionals from BOTH the LHS and RHS charts."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "g": ["a", "b", "c"]})

    sel_a = fm.selection_point(fields=["g"], name="sel_a")
    cond_a = sel_a.when(fm.Color("g")).otherwise(fm.value("#cccccc"))
    chart_a = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel_a)
        .conditional(cond_a)
        .properties(width=200, height=200)
    )

    sel_b = fm.selection_point(fields=["g"], name="sel_b")
    cond_b = sel_b.when(fm.Opacity("g")).otherwise(fm.value(0.2))
    chart_b = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel_b)
        .conditional(cond_b)
        .properties(width=200, height=200)
    )

    merged = chart_a + chart_b

    # Both selections must be present
    sel_names = {s.name for s in (merged._selections or [])}
    assert "sel_a" in sel_names, f"LHS selection 'sel_a' missing from merged; got {sel_names}"
    assert "sel_b" in sel_names, f"RHS selection 'sel_b' missing from merged; got {sel_names}"

    # Both conditionals must be present
    assert len(merged._conditionals or []) >= 2, (
        f"expected >= 2 conditionals (one from each layer); got {len(merged._conditionals or [])}"
    )


def test_layer_chart_interactive_preserves_both_selections():
    """LayerChart._render_interactive must produce scene JSON containing
    selections from all layers, not just the first."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "g": ["a", "b", "c"]})

    sel_a = fm.selection_point(fields=["g"], name="layer_sel_a")
    chart_a = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel_a)
        .properties(width=200, height=200)
    )

    sel_b = fm.selection_point(fields=["g"], name="layer_sel_b")
    chart_b = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel_b)
        .properties(width=200, height=200)
    )

    layer = fm.LayerChart(chart_a, chart_b)
    scene_json, _ = layer._render_interactive()
    scene = json.loads(scene_json)

    sel_names = {s["name"] for s in scene.get("selections", [])}
    assert "layer_sel_a" in sel_names, f"LHS selection missing from layered scene; got {sel_names}"
    assert "layer_sel_b" in sel_names, f"RHS selection missing from layered scene; got {sel_names}"


# ── W6. HTML <title> uses text, not Title dataclass repr ─────────────────


def test_html_title_uses_text_not_repr(tmp_path):
    """HTML <title> must contain the title text, not the Title dataclass repr."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .properties(title="My Custom Title", width=300, height=200)
    )
    out = tmp_path / "title_test.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "<title>My Custom Title</title>" in content, (
        "HTML title must be 'My Custom Title', not Title dataclass repr"
    )
    assert "Title(" not in content, "HTML must not contain Title dataclass repr"


def test_html_title_via_save_chart_uses_text(tmp_path):
    """display.save_chart HTML path must also extract Title.text, not repr."""
    from ferrum.display import save_chart

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .properties(title="Save Chart Title", width=300, height=200)
    )
    out = tmp_path / "save_chart_title.html"
    save_chart(chart, str(out))
    content = out.read_text()
    assert "<title>Save Chart Title</title>" in content, (
        "save_chart HTML title must be 'Save Chart Title'"
    )
    assert "Title(" not in content, "save_chart HTML must not contain Title dataclass repr"


def test_html_title_without_title_defaults_to_ferrum(tmp_path):
    """A chart with no title must use 'Ferrum chart' as the HTML title."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "no_title.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "<title>Ferrum chart</title>" in content, (
        "Chart with no title must use 'Ferrum chart' as HTML title"
    )


# ── W8. _strip_anywidget_for_standalone post-transform assertions ────────


def test_strip_anywidget_raises_on_residual_export():
    """_strip_anywidget_for_standalone must raise RuntimeError if
    a residual 'export' keyword remains after stripping."""
    from ferrum._html import _strip_anywidget_for_standalone

    # Feed it JS that has an export the regexes won't match
    bad_js = "export function foo() {}\nexport function createStandaloneAdapter() {}"
    with pytest.raises(RuntimeError, match="residual export"):
        _strip_anywidget_for_standalone(bad_js)


def test_strip_anywidget_real_source_strips_cleanly():
    """The real ferrum-anywidget.js must strip without residual exports."""
    from ferrum._html import _strip_anywidget_for_standalone, _WASM_DIR

    source = (_WASM_DIR / "ferrum-anywidget.js").read_text()
    result = _strip_anywidget_for_standalone(source)
    # If we get here without RuntimeError, the strip was clean.
    # Also verify no export keywords remain on non-comment lines.
    for line in result.splitlines():
        stripped = line.lstrip()
        assert not stripped.startswith("export ") and not stripped.startswith("export{"), (
            f"residual export found in stripped JS: {stripped[:80]!r}"
        )


# ── B2 (empty-data SceneGraph). Empty-data scene JSON missing required fields ──


_REQUIRED_SCENE_KEYS = {
    "panels",
    "width",
    "height",
    "background",
    "title",
    "legend",
    "decorations",
    "selections",
    "interaction",
}

_REQUIRED_INTERACTION_KEYS = {
    "zoom_enabled",
    "pan_enabled",
    "conditionals",
    "linked_panels",
    "tick_levels",
}


def test_b2_empty_data_scene_json_has_all_required_keys():
    """Empty-DataFrame charts must produce scene JSON with ALL required
    SceneGraph fields, not just panels/width/height.  The Rust SceneGraph
    struct has no #[serde(default)] — omitting background, title, legend,
    decorations, selections, or interaction causes WASM loadScene to fail."""
    df = pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    scene_json, packed = _render_scene(chart)
    scene = json.loads(scene_json)

    # All required top-level keys must be present
    missing = _REQUIRED_SCENE_KEYS - scene.keys()
    assert not missing, f"empty-data scene JSON missing required keys: {missing}"

    # panels must be empty list
    assert scene["panels"] == []
    assert scene["width"] == 300
    assert scene["height"] == 200

    # interaction sub-object must have all required keys
    interaction = scene["interaction"]
    missing_int = _REQUIRED_INTERACTION_KEYS - interaction.keys()
    assert not missing_int, f"interaction missing required keys: {missing_int}"

    # packed data must be empty bytes
    assert packed == b""


def test_b2_empty_data_interactive_chart_does_not_crash():
    """Creating an InteractiveChart from an empty-DataFrame chart must not
    crash and must produce valid scene JSON with all required SceneGraph
    fields."""
    df = pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    ic = chart.interactive()
    assert isinstance(ic, InteractiveChart)

    scene = json.loads(ic.scene_json)
    missing = _REQUIRED_SCENE_KEYS - scene.keys()
    assert not missing, f"InteractiveChart scene JSON missing required keys: {missing}"

    interaction = scene["interaction"]
    missing_int = _REQUIRED_INTERACTION_KEYS - interaction.keys()
    assert not missing_int, f"interaction missing required keys: {missing_int}"


# ── W20: CSP nonce support ────────────────────────────────────────────────────


def test_w20_csp_nonce_on_style_and_script():
    """assemble_html with csp_nonce="abc123" must add nonce="abc123" to
    both the <style> and <script type="module"> tags."""
    from ferrum._html import assemble_html

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    scene_json, packed = _render(chart)

    html = assemble_html(scene_json, packed_data=packed, csp_nonce="abc123")

    assert '<style nonce="abc123">' in html, (
        'HTML must contain <style nonce="abc123"> when csp_nonce is provided'
    )
    assert '<script type="module" nonce="abc123">' in html, (
        'HTML must contain <script type="module" nonce="abc123"> when csp_nonce is provided'
    )


def test_w20_no_nonce_by_default():
    """assemble_html with no csp_nonce must produce <style> and
    <script type="module"> without a nonce attribute."""
    from ferrum._html import assemble_html

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    scene_json, packed = _render(chart)

    html = assemble_html(scene_json, packed_data=packed)

    assert "<style>" in html, "HTML must contain plain <style> when no csp_nonce"
    assert "nonce=" not in html, "HTML must not contain nonce= attribute when csp_nonce is not set"


def test_w20_csp_nonce_via_interactive_save(tmp_path):
    """InteractiveChart.save() with csp_nonce="test123" kwarg must produce
    HTML containing nonce="test123" on style and script tags."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)
    out = tmp_path / "nonce_test.html"
    chart.interactive().save(str(out), csp_nonce="test123")
    content = out.read_text()
    assert 'nonce="test123"' in content, (
        'HTML saved via InteractiveChart.save() must contain nonce="test123"'
    )


# ── B5. Composed packed data must preserve marks and rewrite panel_idx ───


def test_b5_packed_data_preserved_in_composition():
    """Composed charts with >1000 marks must produce non-empty packed data
    so large scatter plots are visible in interactive HTML exports."""
    df = pl.DataFrame({"x": list(range(1500)), "y": list(range(1500))})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    scene_json, packed = composed._render_interactive()
    assert len(packed) > 0, "composed 1500-point charts must have non-empty packed data"
    # Verify scene has 2 panels
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 2


def test_b5_packed_data_panel_idx_rewritten():
    """Packed data headers must have panel_idx rewritten with composition offset.

    A left | right composite of two 1500-point charts must produce packed data
    whose first batch has panel_idx == 0 and second batch has panel_idx == 1.
    """
    df = pl.DataFrame({"x": list(range(1500)), "y": list(range(1500))})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    _, packed = composed._render_interactive()

    batches, _ = _parse_real_packed_batches(packed)
    assert len(batches) >= 2, (
        f"expected at least 2 parsed batches, got {len(batches)} "
        f"(packed length={len(packed)} bytes)"
    )
    assert batches[0]["panel_idx"] == 0, (
        f"first batch panel_idx should be 0, got {batches[0]['panel_idx']}"
    )
    assert batches[1]["panel_idx"] == 1, (
        f"second batch panel_idx should be 1 (rewritten with composition offset), "
        f"got {batches[1]['panel_idx']}"
    )


# ---------------------------------------------------------------------------
# B5-discriminating: multi-batch packed composite with tooltips preserves
# ALL batches (finding A regression test).
# ---------------------------------------------------------------------------


def _parse_real_packed_batches(packed: bytes) -> tuple[list[dict], int]:
    """Parse all batches from packed binary data using the REAL format.

    Returns (batches, final_pos) where batches is a list of dicts with keys:
    panel_idx, batch_idx, kind, count; and final_pos is the byte offset after
    the last successfully parsed batch.
    Mirrors scene_load.rs::unpack_binary_instances exactly.
    """
    import struct

    INSTANCE_SIZES = {0: 64, 1: 72}
    batches = []
    pos = 0
    while pos + 20 <= len(packed):
        panel_idx, batch_idx, kind, count, flags = struct.unpack_from("<5I", packed, pos)
        if kind not in INSTANCE_SIZES:
            break
        batch_end = pos + 20 + count * INSTANCE_SIZES[kind]
        if flags & 0x2:
            batch_end += count * 4
        if flags & 0x1:
            if batch_end + 4 > len(packed):
                break
            num_fields = struct.unpack_from("<I", packed, batch_end)[0]
            batch_end += 4
            for _ in range(num_fields):
                if batch_end + 4 > len(packed):
                    break
                slen = struct.unpack_from("<I", packed, batch_end)[0]
                batch_end += 4 + slen
                if batch_end > len(packed):
                    break
            for _ in range(count * num_fields):
                if batch_end + 4 > len(packed):
                    break
                slen = struct.unpack_from("<I", packed, batch_end)[0]
                batch_end += 4 + slen
                if batch_end > len(packed):
                    break
        if batch_end > len(packed):
            break
        batches.append(
            {"panel_idx": panel_idx, "batch_idx": batch_idx, "kind": kind, "count": count}
        )
        pos = batch_end
    return batches, pos


def test_b5_multi_batch_tooltips_both_batches_survive_untitled():
    """Two >1000-mark scatters composed with `|` must both appear in packed data.

    Before the finding-A fix: the old scan reads num_fields (e.g. 2) as the
    tooltip byte-length, advances ~6 bytes, desyncs, and the 2nd batch is
    either dropped or the parse aborts early — this assertion would fail.

    After the fix: both batches are present, panel_idx 0 and 1, count > 1000.
    """
    n = 1500
    df = pl.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "y": [float(i % 50) for i in range(n)],
        }
    )
    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    composed = c | c

    _scene_json, packed = composed._render_interactive()

    assert packed, "a >1000-mark point composite must produce non-empty packed data"

    batches, final_pos = _parse_real_packed_batches(packed)

    assert len(batches) == 2, (
        f"Expected 2 packed batches (one per panel); got {len(batches)}. "
        "If 0 or 1, the old tooltip-scan misparse is still active."
    )
    panel_ids = {b["panel_idx"] for b in batches}
    assert panel_ids == {0, 1}, f"Expected panel_idx {{0, 1}}; got {panel_ids}"
    for b in batches:
        assert b["count"] > 1000, (
            f"batch for panel {b['panel_idx']} has count={b['count']}, expected >1000"
        )

    # No trailing garbage: the parser consumed the entire buffer.
    assert final_pos == len(packed), (
        f"Parser stopped at byte {final_pos} but packed has {len(packed)} bytes — "
        "trailing garbage or desync detected"
    )


def test_b5_multi_batch_tooltips_both_batches_survive_titled():
    """Same as untitled test but with a figure title (exercises tooltip-scan + T3 Y-offset).

    The title injects a header band; the composite render entry must both survive the
    tooltip scan AND apply the Y-offset correctly.  Both batches must be present in the
    output.
    """
    n = 1500
    df = pl.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "y": [float(i % 50) for i in range(n)],
        }
    )
    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    composed = (c | c).properties(title="MultiPanelTitle")

    _scene_json, packed = composed._render_interactive()

    assert packed, "a titled >1000-mark point composite must produce non-empty packed data"

    batches, final_pos = _parse_real_packed_batches(packed)

    assert len(batches) == 2, (
        f"Expected 2 packed batches in titled composite; got {len(batches)}. "
        "Tooltip scan or Y-offset is corrupting the second batch."
    )
    panel_ids = {b["panel_idx"] for b in batches}
    assert panel_ids == {0, 1}, f"Expected panel_idx {{0, 1}}; got {panel_ids}"
    for b in batches:
        assert b["count"] > 1000, (
            f"batch for panel {b['panel_idx']} has count={b['count']}, expected >1000"
        )

    assert final_pos == len(packed), (
        f"Parser stopped at byte {final_pos} but packed has {len(packed)} bytes"
    )


def test_b3_repeat_corner_interactive_sparse_layout():
    """RepeatChart corner=True interactive layout must place cells at correct
    (row, col) grid positions with gaps for the upper triangle, matching the
    SVG layout.

    Bug (historical, pre-Task-9): the legacy interactive scene-merge fed
    corner cells as a flat list into its grid merge, wrapping linearly
    (row 0 = cells 0-2, row 1 = cells 3-5) while the SVG path placed them at
    their true grid coordinates: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2).
    Both paths now share the one Rust composite entry, which this test pins.
    """
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    template = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.Repeat.column, y=fm.Repeat.row)
        .properties(width=100, height=100)
    )
    chart = fm.RepeatChart(template, row=["a", "b", "c"], column=["a", "b", "c"], corner=True)
    scene_json, _ = chart._render_interactive()
    scene = json.loads(scene_json)

    # Corner mode for 3 vars produces 6 cells:
    # (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
    panels = scene["panels"]
    assert len(panels) == 6, f"expected 6 panels for 3x3 corner; got {len(panels)}"

    pa = [p["plot_area"] for p in panels]

    # Row 0: 1 cell at col 0  -> panel 0
    # Row 1: 2 cells at col 0, col 1 -> panels 1, 2
    # Row 2: 3 cells at col 0, col 1, col 2 -> panels 3, 4, 5

    # Cells in the same row must share the same y offset.
    assert abs(pa[1]["y"] - pa[2]["y"]) < 1.0, "row 1 cells must have same y offset"
    assert abs(pa[3]["y"] - pa[4]["y"]) < 1.0, "row 2 cells must have same y offset"
    assert abs(pa[4]["y"] - pa[5]["y"]) < 1.0, "row 2 cells must have same y offset"

    # Rows must be stacked vertically in order.
    assert pa[1]["y"] > pa[0]["y"], "row 1 must be below row 0"
    assert pa[3]["y"] > pa[1]["y"], "row 2 must be below row 1"

    # Within row 1, col 1 must be right of col 0.
    assert pa[2]["x"] > pa[1]["x"], "col 1 must be right of col 0 in row 1"

    # Column alignment across rows: cells in the same column should share
    # approximate x.  Tolerance is wider than row-y checks because cells
    # render independently and axis-label widths differ per field.
    # col 0: panels 0, 1, 3
    assert abs(pa[0]["x"] - pa[1]["x"]) < 5.0, "col 0 panels must share approximate x"
    assert abs(pa[1]["x"] - pa[3]["x"]) < 5.0, "col 0 panels must share approximate x"
    # col 1: panels 2, 4
    assert abs(pa[2]["x"] - pa[4]["x"]) < 5.0, "col 1 panels must share approximate x"


def test_b3_repeat_corner_2var_interactive_layout():
    """Corner repeat with 2 vars produces 3 cells in a lower triangle."""
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    template = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.Repeat.column, y=fm.Repeat.row)
        .properties(width=100, height=100)
    )
    chart = fm.RepeatChart(template, row=["a", "b"], column=["a", "b"], corner=True)
    scene_json, _ = chart._render_interactive()
    scene = json.loads(scene_json)

    panels = scene["panels"]
    assert len(panels) == 3, f"expected 3 panels for 2x2 corner; got {len(panels)}"

    pa = [p["plot_area"] for p in panels]

    # (0,0), (1,0), (1,1)
    # Row 1 cells must share y.
    assert abs(pa[1]["y"] - pa[2]["y"]) < 1.0, "row 1 cells must share y"
    # Row 1 below row 0.
    assert pa[1]["y"] > pa[0]["y"], "row 1 below row 0"
    # Within row 1, col 1 right of col 0.
    assert pa[2]["x"] > pa[1]["x"], "col 1 right of col 0"
    # Col 0 alignment: panel 0 and panel 1 should be at similar x.
    # Tolerance is wider than row-y checks because cells render independently
    # and axis-label widths differ by field, shifting plot_area.x slightly.
    assert abs(pa[0]["x"] - pa[1]["x"]) < 5.0, "col 0 panels share approximate x"


def test_b5_vconcat_packed_data_preserved():
    """VConcatChart with >1000 marks must also produce non-empty packed data."""
    df = pl.DataFrame({"x": list(range(1500)), "y": list(range(1500))})
    top = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    bot = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = top & bot
    scene_json, packed = composed._render_interactive()
    assert len(packed) > 0, "VConcat 1500-point charts must have non-empty packed data"
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 2


# ── W2. Multi-panel brush overlays for interval selection ──────────────────


def test_w2_multi_panel_brush_js_iterates_panels(tmp_path):
    """Composed charts with interval selection must produce HTML whose JS
    iterates over all panels to create per-panel brush overlays, not a
    single brush hardcoded to panel 0.

    Since the HTML is static (D3 creates <g> elements at runtime), we verify
    the JS source structure: the per-panel loop, the ferrum-brush class name,
    the panelIdx variable, and that the scene JSON has multiple panels."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    sel = selection_interval(name="brush_sel")
    left = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    right = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    composed = left | right
    out = tmp_path / "multi_brush.html"
    composed.interactive().save(str(out))
    content = out.read_text()

    # JS must contain the per-panel brush loop, not a single hardcoded brush
    assert "ferrum-brush" in content, (
        "JS must use 'ferrum-brush' class for per-panel brush overlays"
    )
    assert "handleDrag(panelIdx," in content, (
        "brush handler must pass panelIdx to handleDrag, not hardcoded 0"
    )
    assert "data-panel" in content, (
        "brush groups must have data-panel attribute for panel identification"
    )
    # The old hardcoded single-panel pattern must be gone
    assert "handleDrag(0," not in content, (
        "must not have hardcoded handleDrag(0, ...) -- use panelIdx instead"
    )


def test_w2_single_panel_brush_in_html(tmp_path):
    """A single chart with interval selection must produce HTML containing
    the per-panel brush loop (which will create exactly one brush at runtime)."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    sel = selection_interval(name="brush_sel")
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    out = tmp_path / "single_brush.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    # Must have the per-panel brush loop in JS
    assert "ferrum-brush" in content, (
        "single chart with interval selection must have ferrum-brush in JS"
    )
    assert "handleDrag(panelIdx," in content, "brush handler must use panelIdx variable"


def test_w2_vconcat_multi_panel_brush(tmp_path):
    """VConcat composed charts with interval selection must produce HTML
    with per-panel brush loop and 2 panels in scene JSON."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    sel = selection_interval(name="brush_sel")
    top = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    bot = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    composed = top & bot
    out = tmp_path / "vconcat_brush.html"
    composed.interactive().save(str(out))
    content = out.read_text()
    assert "ferrum-brush" in content, "VConcat JS must use ferrum-brush class"
    assert "handleDrag(panelIdx," in content, "VConcat brush handler must use panelIdx"


def test_w2_scene_json_has_multiple_panels_for_composition():
    """Composed charts with interval selection produce scene JSON with
    multiple panels, each having its own plot_area for brush extent."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    sel = selection_interval(name="brush_sel")
    left = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    right = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .add_selection(sel)
        .properties(width=200, height=200)
    )
    composed = left | right
    scene_json, _ = composed._render_interactive()
    scene = json.loads(scene_json)
    panels = scene["panels"]
    assert len(panels) == 2, f"expected 2 panels; got {len(panels)}"
    # Each panel must have a plot_area (used as brush extent)
    for i, panel in enumerate(panels):
        pa = panel.get("plot_area")
        assert pa is not None, f"panel {i} must have plot_area"
        assert pa["w"] > 0, f"panel {i} plot_area width must be > 0"
        assert pa["h"] > 0, f"panel {i} plot_area height must be > 0"
    # Panels must have distinct plot_area positions (not overlapping)
    assert panels[1]["plot_area"]["x"] > panels[0]["plot_area"]["x"], (
        "panel 1 must be offset to the right of panel 0"
    )
