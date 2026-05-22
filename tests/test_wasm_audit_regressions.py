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
    assert "<script>" not in content.split("</head>")[0]  # not in <head>
    assert "&lt;script&gt;" in content or "alert" not in content.split("<title>")[1].split("</title>")[0]


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
