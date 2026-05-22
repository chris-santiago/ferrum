"""Regression tests for WASM audit remediation."""
import warnings


def test_html_title_escapes_special_chars(tmp_path):
    """Regression: B6 — title with HTML special chars must be escaped."""
    import ferrum as fm
    import polars as pl

    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(
        title='<script>alert("xss")</script>'
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
