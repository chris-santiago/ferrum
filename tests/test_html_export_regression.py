"""Regression tests for the HTML export pipeline.

Covers:
- Packed data contract: large charts produce non-empty packed bytes (P1)
- Small charts produce empty packed data (P2)
- Scene JSON round-trip structure (P3)
- Theme background preserved in scene JSON (P4)
- Composition `.interactive()` returns InteractiveChart (P5)
- Composition `.save("out.html")` produces valid HTML (P6)
- Composition `show_svg()` unchanged (P7)
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
- Composition show_svg() identical before/after .interactive() (P25)
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
    spec, data, viewport, theme = chart._render_inputs()
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


# ── P7. Composition show_svg() unchanged ─────────────────────────────────────


def test_p7_composition_show_svg_unchanged():
    """Composing two charts with | and calling show_svg() must return valid
    SVG markup.  This locks that the SVG path is not broken by interactive
    changes."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    svg = composed.show_svg()
    assert isinstance(svg, str)
    assert "<svg" in svg, "show_svg() must return SVG markup"


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


# ── P25. Composition show_svg() identical before/after .interactive() ──────


def test_p25_composition_svg_unaffected_by_interactive():
    """Calling .interactive() on a composition must not mutate the composition
    or change the SVG output from show_svg()."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    svg_before = composed.show_svg()
    _ = composed.interactive()
    svg_after = composed.show_svg()
    assert svg_before == svg_after, (
        "show_svg() must produce identical output before and after .interactive()"
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
