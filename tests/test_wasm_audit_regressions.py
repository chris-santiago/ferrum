"""Regression tests for WASM audit remediation."""

import warnings


def test_html_title_escapes_special_chars(tmp_path):
    """Regression: B6 — title with HTML special chars must be escaped."""
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(title='<script>alert("xss")</script>')
    )
    out = tmp_path / "xss_test.html"
    chart.save(str(out))
    content = out.read_text()
    title_content = content.split("<title>")[1].split("</title>")[0]
    assert "<script>" not in title_content
    assert "&lt;script&gt;" in title_content


def test_html_title_escapes_ampersand(tmp_path):
    """Regression: B6 — ampersand in title must be escaped."""
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(title="A & B")
    out = tmp_path / "amp_test.html"
    chart.save(str(out))
    content = out.read_text()
    title_section = content.split("<title>")[1].split("</title>")[0]
    assert "&amp;" in title_section


# ── R5: malformed hex warns ────────────────────────────────────────────────────


def test_hex_to_color_dict_warns_on_malformed():
    """Regression: R5 — malformed hex should warn, not silently return black."""
    from ferrum.selection import _hex_to_color_dict

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _hex_to_color_dict("#xyz")
        assert len(w) == 1
        assert "Unrecognized hex" in str(w[0].message)
    assert result == {"r": 0, "g": 0, "b": 0, "a": 255}


def test_hex_to_color_dict_3char_expands():
    """Regression: 3-char hex shorthand correctly expands."""
    from ferrum.selection import _hex_to_color_dict

    result = _hex_to_color_dict("#abc")
    assert result == {"r": 0xAA, "g": 0xBB, "b": 0xCC, "a": 255}


def test_hex_to_color_dict_4char_expands():
    """Regression: 4-char hex shorthand correctly expands."""
    from ferrum.selection import _hex_to_color_dict

    result = _hex_to_color_dict("#abcd")
    assert result == {"r": 0xAA, "g": 0xBB, "b": 0xCC, "a": 0xDD}


# ── R4: _render_scene_json delegates to _render_scene ─────────────────────────


def test_render_scene_json_delegates():
    """Regression: R4 — _render_scene_json delegates to _render_scene."""
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")

    from ferrum._interactive import _render_scene
    from ferrum.display import _render_scene_json

    json1, packed1 = _render_scene_json(chart)
    json2, packed2 = _render_scene(chart)
    assert json1 == json2
    assert packed1 == packed2


# ── ResizeObserver vs Save PNG race condition ────────────────────────────────


def test_wide_hconcat_saves_html_without_error(tmp_path):
    """Regression: wide HConcat charts (>viewport) must produce valid HTML.

    The ResizeObserver previously interfered with the Save PNG DPR resize
    flow on charts wider than the viewport. The observer saw the
    CSS-constrained contentRect (< canvas.width) and reset canvas.width,
    corrupting the capture and leaving the chart empty afterward.
    """
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": list(range(8)), "y": [2, 4, 1, 5, 3, 6, 2, 4], "g": ["a", "b"] * 4})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="g:N")
    right = fm.Chart(df).mark_bar().encode(x="g:N", y="y:Q", color="g:N")
    comp = left | right
    out = tmp_path / "wide_hconcat.html"
    comp.interactive().save(str(out))
    content = out.read_text()
    # The JS must contain the ResizeObserver disconnect guard in onSave
    assert "_ro" in content, "ResizeObserver variable should exist"
    assert "disconnect" in content, "onSave should disconnect ResizeObserver"


def test_save_png_injects_phys_dpi_metadata(tmp_path):
    """Regression: Save PNG must inject a pHYs chunk for DPI metadata instead
    of upscaling the canvas by devicePixelRatio (which caused gridline
    artifacts in composed charts due to stale pixel-snap positions).
    """
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": list(range(8)), "y": [2, 4, 1, 5, 3, 6, 2, 4], "g": ["a", "b"] * 4})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="g:N")
    right = fm.Chart(df).mark_bar().encode(x="g:N", y="y:Q", color="g:N")
    comp = left | right
    out = tmp_path / "phys_dpi.html"
    comp.interactive().save(str(out))
    content = out.read_text()
    assert "injectPHYs" in content, "onSave must inject pHYs DPI metadata"
    assert "pHYs" in content, "pHYs chunk builder must be present"
    assert "captureScale" not in content, "DPR upscale logic must be removed"


def test_save_png_no_canvas_resize(tmp_path):
    """Regression: Save PNG must not resize the canvas — 1:1 capture avoids
    gridline snap artifacts from stale tessellation.  The viewBox override
    and origW/origH tracking are no longer needed.
    """
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    out = tmp_path / "no_resize.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    # Extract onSave body — it sits between "async function onSave" and the
    # next top-level "// ──" section divider.
    start = content.find("async function onSave")
    end = content.find("// ──", start + 1) if start != -1 else -1
    on_save_body = content[start:end] if start != -1 and end != -1 else ""
    assert on_save_body, "onSave function must exist in the HTML"
    assert "renderer.resize" not in on_save_body, "onSave must not call renderer.resize"
    assert "off.width = exportW" in on_save_body, "offscreen canvas should use full-DPR export size"


def test_canvas_dpr_clamped_to_max_texture(tmp_path):
    """Regression: Wide HConcat at DPR=2 must not exceed GPU max texture size.

    Canvas init creates at CSS size, then resizes to clamped DPR after
    querying maxTextureSize.  The JS must contain the clamping logic.
    """
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": list(range(8)), "y": [2, 4, 1, 5, 3, 6, 2, 4], "g": ["a", "b"] * 4})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="g:N")
    right = fm.Chart(df).mark_bar().encode(x="g:N", y="y:Q", color="g:N")
    comp = left | right
    out = tmp_path / "dpr_clamp.html"
    comp.interactive().save(str(out))
    content = out.read_text()
    assert "maxTextureSize" in content, "must query GPU max texture size"
    assert "effectiveDpr" in content, "must compute clamped effective DPR"
    assert "canvas.width = w;" in content, "canvas must start at CSS size"


# ── M5: _auto_tooltips removed from public to_spec() ─────────────────────────


def test_to_spec_no_auto_tooltips_param():
    """Regression: M5 — to_spec() no longer accepts _auto_tooltips kwarg."""
    import pytest
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    # Public to_spec should NOT accept _auto_tooltips
    with pytest.raises(TypeError):
        chart.to_spec(_auto_tooltips=True)


def test_interactive_render_still_injects_tooltips():
    """Regression: M5 — interactive rendering still gets tooltip injection."""
    import json
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    ic = chart.interactive()
    scene_json = ic._scene_json
    scene = json.loads(scene_json)
    # Check that tooltip data exists in the scene
    has_tooltips = any(
        batch.get("tooltips") is not None
        for panel in scene.get("panels", [])
        for batch in panel.get("marks", [])
    )
    assert has_tooltips, "Interactive render should inject auto-tooltips"


def test_svg_render_no_tooltip_injection():
    """Regression: M5 — SVG render should NOT have tooltip bloat."""
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    svg = chart.show_svg()
    # SVG should not contain tooltip-related data attributes
    assert "data-tooltip" not in svg


# ── M6: toolbar as explicit param on _ChartLike.save() ───────────────────────


def test_chartlike_save_toolbar_explicit_param(tmp_path):
    """Regression: M6 — compositions accept explicit toolbar param."""
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "g": ["a", "b", "a"]})
    top = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    bottom = fm.Chart(df).mark_bar().encode(x="g:N", y="y:Q")
    comp = top & bottom
    out = tmp_path / "comp_toolbar.html"
    comp.save(str(out), toolbar=False)
    content = out.read_text()
    assert '"toolbar": false' in content or '"toolbar":false' in content
