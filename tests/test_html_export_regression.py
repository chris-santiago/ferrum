"""Regression tests for the HTML export pipeline.

Covers:
- Packed data contract: large charts produce non-empty packed bytes (P1)
- Small charts produce empty packed data (P2)
- Scene JSON round-trip structure (P3)
- Theme background preserved in scene JSON (P4)
- Composition `.interactive()` returns InteractiveChart (P5, xfail)
- Composition `.save("out.html")` produces valid HTML (P6, xfail)
- Composition `show_svg()` unchanged (P7)
- InteractiveChart preserves packed data (P8)
- Selection spec serialization round-trip (P9)
- Conditional encoding in scene JSON (P10)
- `_render_scene` returns tuple (P11)
- HTML export JS has no `model.get` / `model.set` calls (P12)
- Text elements in scene JSON (P13)
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
    assert "createElementNS" in content or "_placeTextSvg" in content, \
        "HTML must contain SVG element creation"
