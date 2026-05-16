import polars as pl
import pytest

from ferrum import Chart


@pytest.fixture
def chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return Chart(df).mark_point().encode(x="a", y="b")


def test_save_svg(chart, tmp_path):
    out = tmp_path / "out.svg"
    chart.save(out)
    text = out.read_text()
    assert text.startswith("<svg") or text.startswith("<?xml")


def test_save_png(chart, tmp_path):
    out = tmp_path / "out.png"
    chart.save(out)
    bytes_ = out.read_bytes()
    assert bytes_.startswith(b"\x89PNG\r\n\x1a\n")


def test_save_html(chart, tmp_path):
    out = tmp_path / "out.html"
    chart.save(out)
    text = out.read_text()
    assert "<!DOCTYPE html>" in text
    assert "ferrum" in text.lower()


def test_save_json(chart, tmp_path):
    import json

    out = tmp_path / "out.json"
    chart.save(out)
    scene = json.loads(out.read_text())
    assert "panels" in scene
    assert "width" in scene


def test_save_unknown_extension_raises(chart, tmp_path):
    with pytest.raises(ValueError, match="extension"):
        chart.save(tmp_path / "out.weird")


def test_save_explicit_format_overrides_extension(chart, tmp_path):
    out = tmp_path / "out.txt"
    chart.save(out, format="svg")
    text = out.read_text()
    assert "<svg" in text or "<?xml" in text


# Task 33: show browser fallback + Jupyter rich-display paths


def test_show_in_non_jupyter_opens_browser(chart, monkeypatch):
    """When not in Jupyter, .show() writes a temp HTML and calls webbrowser.open."""
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("ferrum.display._is_jupyter", lambda: False)
    chart.show()
    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith(".html")


def test_repr_svg_returns_string_for_jupyter(chart):
    s = chart._repr_svg_()
    assert s is not None
    assert "<svg" in s or "<?xml" in s


def test_repr_html_returns_div_wrapped_svg(chart):
    s = chart._repr_html_()
    assert s is not None
    assert s.startswith("<div>")


# ── Phase 11c: WASM renderer regression tests ──────────────────────


def test_save_html_contains_wasm_init_and_canvas(chart, tmp_path):
    """HTML output must contain WASM init, canvas creation, and tooltip wiring."""
    out = tmp_path / "out.html"
    chart.save(out)
    text = out.read_text()
    assert "__wbg_init" in text, "missing WASM init call"
    assert "createElement('canvas')" in text, "missing canvas creation"
    assert "ferrum-tooltip" in text, "missing tooltip CSS class"
    assert "hitTest" in text, "missing JS hit-test function"


def test_scene_json_has_no_null_floats(chart, tmp_path):
    """SceneGraph JSON must not contain null where f64 is expected (e.g. max_zoom)."""
    import json

    out = tmp_path / "scene.json"
    chart.save(out, format="json")
    text = out.read_text()
    scene = json.loads(text)
    for ptl in scene.get("interaction", {}).get("tick_levels", []):
        for level in ptl.get("x_levels", []) + ptl.get("y_levels", []):
            assert level["min_zoom"] is not None, "min_zoom is null"
            assert level["max_zoom"] is not None, "max_zoom is null"
            assert isinstance(level["max_zoom"], (int, float)), (
                f"max_zoom is {type(level['max_zoom'])}, expected number"
            )


def test_scene_json_round_trips_through_serde(chart, tmp_path):
    """SceneGraph JSON must deserialize without errors (catches alignment/type mismatches)."""
    import json

    out = tmp_path / "scene.json"
    chart.save(out, format="json")
    scene = json.loads(out.read_text())
    assert isinstance(scene["panels"], list)
    assert len(scene["panels"]) > 0
    assert "marks" in scene["panels"][0]
    for batch in scene["panels"][0]["marks"]:
        for node in batch["nodes"]:
            assert "type" in node, "SceneNode missing 'type' discriminator"


def test_save_html_base64_wasm_decodable(chart, tmp_path):
    """The base64-inlined WASM must be decodable (catches encoding corruption)."""
    import base64
    import re

    out = tmp_path / "out.html"
    chart.save(out)
    text = out.read_text()
    m = re.search(r"const wasmB64 = '([A-Za-z0-9+/=]+)'", text)
    assert m, "base64 WASM blob not found in HTML"
    decoded = base64.b64decode(m.group(1))
    assert decoded[:4] == b"\x00asm", "decoded bytes are not a valid WASM module"
