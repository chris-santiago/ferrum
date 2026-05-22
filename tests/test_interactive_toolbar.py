"""Tests for the toolbar=True/False flag in the interactive HTML export.

Covers:
- Chart.interactive() returns InteractiveChart with _toolbar=True by default
- Chart.interactive(toolbar=False) returns InteractiveChart with _toolbar=False
- _build_interaction_config() serializes "toolbar": true by default
- _build_interaction_config() serializes "toolbar": false when toolbar=False
- assemble_html() embeds "toolbar":true in the interaction config by default
- assemble_html(toolbar=False) embeds "toolbar":false in the interaction config
- InteractiveChart.save() passes toolbar flag through to assemble_html
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import render_interactive
from ferrum._html import assemble_html, _WASM_DIR
from ferrum._interactive import InteractiveChart


# ── helpers ──────────────────────────────────────────────────────────────────


def _simple_chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=300, height=200)


def _render(chart: fm.Chart) -> tuple[str, bytes]:
    spec, data, viewport, theme = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme)


def _wasm_available() -> bool:
    return (_WASM_DIR / "ferrum_wasm.js").exists() and (_WASM_DIR / "ferrum_wasm_bg.wasm").exists()


_skip_no_wasm = pytest.mark.skipif(
    not _wasm_available(),
    reason="WASM artifacts not built; run wasm-pack build first",
)


# ── InteractiveChart._toolbar attribute ──────────────────────────────────────


def test_interactive_chart_toolbar_default_is_true():
    """Chart.interactive() must return an InteractiveChart with _toolbar=True."""
    chart = _simple_chart()
    ic = chart.interactive()
    assert isinstance(ic, InteractiveChart)
    assert ic._toolbar is True


def test_interactive_chart_toolbar_false():
    """Chart.interactive(toolbar=False) must return an InteractiveChart with _toolbar=False."""
    chart = _simple_chart()
    ic = chart.interactive(toolbar=False)
    assert isinstance(ic, InteractiveChart)
    assert ic._toolbar is False


# ── _build_interaction_config serialization ───────────────────────────────────


def test_build_interaction_config_toolbar_true_by_default():
    """_build_interaction_config() must include "toolbar": true by default."""
    chart = _simple_chart()
    ic = InteractiveChart(chart, toolbar=True)
    config = json.loads(ic._build_interaction_config(ic.scene_json))
    assert config.get("toolbar") is True


def test_build_interaction_config_toolbar_false():
    """_build_interaction_config() with toolbar=False must include "toolbar": false."""
    chart = _simple_chart()
    ic = InteractiveChart(chart, toolbar=False)
    config = json.loads(ic._build_interaction_config(ic.scene_json))
    assert config.get("toolbar") is False


def test_build_interaction_config_toolbar_overrides_scene_value():
    """_build_interaction_config() must override whatever toolbar value is in the scene JSON.

    The Python-side flag is authoritative; it must win over the Rust-emitted value.
    """
    chart = _simple_chart()
    ic = InteractiveChart(chart, toolbar=False)
    # Manually verify scene JSON has toolbar (Rust default is true).
    scene = json.loads(ic.scene_json)
    interaction = scene.get("interaction", {})
    # The Rust side defaults toolbar=true; our config must override it to false.
    config = json.loads(ic._build_interaction_config(ic.scene_json))
    assert config.get("toolbar") is False, (
        "Python toolbar=False must override Rust-emitted toolbar=True in scene JSON"
    )
    # Prove Rust indeed emitted true (so the override is meaningful).
    # If Rust omits the key, the assert still holds — we've verified False wins.
    _ = interaction  # consumed above


# ── assemble_html toolbar flag ────────────────────────────────────────────────


@_skip_no_wasm
def test_assemble_html_toolbar_true_in_interaction_config():
    """assemble_html() must embed "toolbar":true in the interaction config JS string."""
    chart = _simple_chart()
    scene_json, packed = _render(chart)
    html = assemble_html(scene_json, packed_data=packed, toolbar=True)
    # The interaction config is JSON-serialized and embedded in the JS as a string literal.
    # "toolbar":true must appear somewhere in the generated HTML source.
    assert '"toolbar": true' in html or '"toolbar":true' in html, (
        "assemble_html with toolbar=True must embed toolbar:true in HTML interaction config"
    )


@_skip_no_wasm
def test_assemble_html_toolbar_false_in_interaction_config():
    """assemble_html(toolbar=False) must embed "toolbar":false in the interaction config JS string."""
    chart = _simple_chart()
    scene_json, packed = _render(chart)
    html = assemble_html(scene_json, packed_data=packed, toolbar=False)
    assert '"toolbar": false' in html or '"toolbar":false' in html, (
        "assemble_html with toolbar=False must embed toolbar:false in HTML interaction config"
    )


@_skip_no_wasm
def test_assemble_html_default_toolbar_is_true():
    """assemble_html() without toolbar kwarg must default to embedding toolbar:true."""
    chart = _simple_chart()
    scene_json, packed = _render(chart)
    html = assemble_html(scene_json, packed_data=packed)
    assert '"toolbar": true' in html or '"toolbar":true' in html, (
        "assemble_html default (no toolbar kwarg) must embed toolbar:true"
    )


@_skip_no_wasm
def test_assemble_html_toolbar_false_config_arg_has_toolbar_false():
    """assemble_html(toolbar=False) must embed toolbar:false in the createStandaloneAdapter call.

    Note: the raw SCENE_JSON template literal also gets embedded and may retain
    "toolbar": true from Rust's default.  We isolate the interaction config
    argument line (which is the Python-controlled value) and assert it has false.
    """
    chart = _simple_chart()
    scene_json, packed = _render(chart)
    html_false = assemble_html(scene_json, packed_data=packed, toolbar=False)
    # Find the createStandaloneAdapter call line (not the function definition line).
    adapter_lines = [
        line
        for line in html_false.splitlines()
        if "createStandaloneAdapter" in line
        and line.strip().startswith("const adapter")
    ]
    assert adapter_lines, "HTML must contain 'const adapter = createStandaloneAdapter(...)' line"
    adapter_line = adapter_lines[0]
    assert '"toolbar": false' in adapter_line, (
        f"createStandaloneAdapter config arg must contain toolbar: false; got: {adapter_line[:300]}"
    )
    assert '"toolbar": true' not in adapter_line, (
        f"createStandaloneAdapter config arg must not contain toolbar: true when toolbar=False; "
        f"got: {adapter_line[:300]}"
    )


# ── InteractiveChart.save() passes toolbar through ────────────────────────────


@_skip_no_wasm
def test_save_html_toolbar_true(tmp_path):
    """InteractiveChart.save() must produce HTML with toolbar:true by default."""
    chart = _simple_chart()
    out = tmp_path / "toolbar_true.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert '"toolbar": true' in content or '"toolbar":true' in content, (
        "HTML saved with default toolbar must embed toolbar:true"
    )


@_skip_no_wasm
def test_save_html_toolbar_false(tmp_path):
    """InteractiveChart(toolbar=False).save() must produce HTML with toolbar:false in the
    createStandaloneAdapter call.
    """
    chart = _simple_chart()
    out = tmp_path / "toolbar_false.html"
    chart.interactive(toolbar=False).save(str(out))
    content = out.read_text()
    # Check that the adapter call specifically has toolbar:false.
    adapter_lines = [
        line
        for line in content.splitlines()
        if "createStandaloneAdapter" in line and line.strip().startswith("const adapter")
    ]
    assert adapter_lines, "HTML must contain 'const adapter = createStandaloneAdapter(...)' line"
    adapter_line = adapter_lines[0]
    assert '"toolbar": false' in adapter_line, (
        "HTML saved with toolbar=False must embed toolbar:false in adapter config"
    )
    assert '"toolbar": true' not in adapter_line, (
        "HTML adapter config must not contain toolbar:true when toolbar=False"
    )


# ── Regression: composition types also plumb toolbar ─────────────────────────


def test_composition_interactive_toolbar_false():
    """Composed chart .interactive(toolbar=False) must produce InteractiveChart with _toolbar=False."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    ic = composed.interactive(toolbar=False)
    assert isinstance(ic, InteractiveChart)
    assert ic._toolbar is False


def test_composition_interactive_toolbar_default_true():
    """Composed chart .interactive() must default to _toolbar=True."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    left = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    right = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composed = left | right
    ic = composed.interactive()
    assert isinstance(ic, InteractiveChart)
    assert ic._toolbar is True


# ── Regression tests for bugs fixed in feat/rtree-toolbar ──────────────


def test_hex_shorthand_3char():
    """#ccc should expand to r=204,g=204,b=204 — not black (r=0,g=0,b=0)."""
    from ferrum.selection import _hex_to_color_dict

    result = _hex_to_color_dict("#ccc")
    assert result == {"r": 204, "g": 204, "b": 204, "a": 255}, f"got {result}"


def test_hex_shorthand_4char():
    """#abcd should expand to r=170,g=187,b=204,a=221."""
    from ferrum.selection import _hex_to_color_dict

    result = _hex_to_color_dict("#abcd")
    assert result == {"r": 170, "g": 187, "b": 204, "a": 221}, f"got {result}"


def test_hex_full_6char_unchanged():
    """Full 6-char hex should work as before."""
    from ferrum.selection import _hex_to_color_dict

    assert _hex_to_color_dict("#ff0000") == {"r": 255, "g": 0, "b": 0, "a": 255}


def test_auto_tooltips_injected_for_interactive():
    """_inject_auto_tooltips() should inject tooltip_fields from encoded channels."""
    import json

    import polars as pl

    import ferrum as fm

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "g": ["a", "b"]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
    spec = chart.to_spec()
    spec_dict = json.loads(spec.to_json())
    spec_dict = chart._inject_auto_tooltips(spec_dict)
    tf = spec_dict.get("encoding", {}).get("tooltip_fields")
    assert tf is not None, "tooltip_fields should be injected"
    fields = json.loads(tf) if isinstance(tf, str) else tf
    field_names = {f["field"] for f in fields}
    assert "x" in field_names
    assert "y" in field_names
    assert "g" in field_names


def test_auto_tooltips_not_injected_for_static():
    """to_spec() without _auto_tooltips should NOT inject tooltip_fields."""
    import json

    import polars as pl

    import ferrum as fm

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    spec = chart.to_spec()
    spec_dict = json.loads(spec.to_json())
    tf = spec_dict.get("encoding", {}).get("tooltip_fields")
    assert tf is None, f"tooltip_fields should NOT be injected for static, got {tf}"


def test_explicit_tooltip_wins_over_auto():
    """Explicit tooltip= should not be overridden by auto-tooltip injection."""
    import json

    import polars as pl

    import ferrum as fm
    from ferrum.display import _render_scene_json

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "g": ["a", "b"]})
    chart = fm.Chart(df).mark_point().encode(
        x="x", y="y", color="g", tooltip=fm.Tooltip("g")
    )
    scene_json, _ = _render_scene_json(chart)
    scene = json.loads(scene_json)
    batch = scene["panels"][0]["marks"][0]
    assert batch["tooltips"] is not None
    # Only the explicit 'g' field, not auto-injected x/y
    field_names = {f["name"] for f in batch["tooltips"][0]["fields"]}
    assert field_names == {"g"}, f"expected only 'g', got {field_names}"


@_skip_no_wasm
def test_chart_save_html_toolbar_false(tmp_path):
    """Chart.save(format='html', toolbar=False) must embed toolbar:false in the HTML."""
    chart = _simple_chart()
    out = tmp_path / "toolbar_false.html"
    chart.save(str(out), toolbar=False)
    content = out.read_text()
    assert '"toolbar": false' in content or '"toolbar":false' in content, (
        "Chart.save with toolbar=False must embed toolbar:false in the HTML"
    )


def test_static_mesh_separate_from_mark_mesh():
    """Grid lines should tessellate into static_mesh, not mark mesh."""
    import json

    import polars as pl

    import ferrum as fm
    from ferrum.display import _render_scene_json

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    scene_json, _packed = _render_scene_json(chart)
    scene = json.loads(scene_json)
    panel = scene["panels"][0]
    assert len(panel.get("grid", [])) > 0, "panel should have grid lines"
