"""Assemble self-contained HTML files for the WASM renderer."""

from __future__ import annotations

import base64
import json as _json
import re
from pathlib import Path

_WASM_DIR = Path(__file__).parent / "_wasm"


def _read_wasm_artifact(name: str) -> bytes:
    artifact = _WASM_DIR / name
    if not artifact.exists():
        raise FileNotFoundError(
            f"WASM artifact {name!r} not found at {artifact}. "
            "Run: wasm-pack build crates/ferrum-wasm --target web "
            "--out-dir ../../src/ferrum/_wasm/"
        )
    return artifact.read_bytes()


def _strip_anywidget_for_standalone(source: str) -> str:
    """Strip ESM exports and anywidget-only code from ferrum-anywidget.js.

    The returned JS is suitable for inlining in a ``<script type="module">``
    block.  Specifically:

    1. Remove the top-level ``const _B64 = ...`` WASM bootstrap block
       (lines 11-14 in the source) — the HTML template has its own init.
    2. Remove ``let _ready`` / ``_initP`` and the ``_ensureWasm`` function —
       standalone HTML calls ``__wbg_init`` directly before ``_render``.
    3. Strip the ``export`` keyword from ``export function createStandaloneAdapter``.
    4. Remove the ``export { _render as renderChart };`` re-export line.
    5. Remove the entire ``export async function render({ model, el }) { ... }``
       anywidget entry point (everything from that line to end of file).
    """
    # 1. Remove _B64 bootstrap block (4 lines: const _B64 through the for loop)
    source = re.sub(
        r"^const _B64 = .*\n"
        r"const _raw = .*\n"
        r"const _bytes = .*\n"
        r"for \(let i = 0; i < _raw\.length.*\n",
        "",
        source,
        flags=re.MULTILINE,
    )

    # 2. Remove _ensureWasm and its state variables, and replace call sites
    source = re.sub(r"^let _ready = false.*\n", "", source, flags=re.MULTILINE)
    source = re.sub(
        r"^async function _ensureWasm\(\) \{.*?\n\}\n",
        "",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    # Replace calls to _ensureWasm() — WASM is already initialized in main()
    source = source.replace("await _ensureWasm();", "// WASM already initialized")

    # 3. Strip `export` from `export function createStandaloneAdapter`
    source = source.replace(
        "export function createStandaloneAdapter",
        "function createStandaloneAdapter",
    )

    # 4. Remove the re-export line
    source = re.sub(
        r"^export \{ _render as renderChart \};\n?",
        "",
        source,
        flags=re.MULTILINE,
    )

    # 5. Remove the entire anywidget `render` export (last function in file)
    source = re.sub(
        r"// ── anywidget entry point ──.*",
        "",
        source,
        flags=re.DOTALL,
    )

    return source.strip()


def _extract_background_css(scene_json: str) -> str:
    """Extract a CSS background color from the scene JSON, defaulting to white."""
    try:
        scene = _json.loads(scene_json)
        bg = scene.get("background")
        if bg:
            return f"rgba({bg['r']},{bg['g']},{bg['b']},{bg['a'] / 255.0})"
    except Exception:
        pass
    return "#ffffff"


def _extract_interaction_config(scene_json: str) -> str:
    """Extract selections + conditionals from scene JSON for the standalone adapter."""
    try:
        scene = _json.loads(scene_json)
        config: dict = dict(scene.get("interaction", {}))
        config["selections"] = scene.get("selections", [])
        return _json.dumps(config)
    except Exception:
        return "{}"


def assemble_html(
    scene_json: str,
    *,
    packed_data: bytes = b"",
    title: str = "Ferrum chart",
    embed_wasm: bool = True,
) -> str:
    """Build a self-contained HTML string that renders a chart via WASM.

    Parameters
    ----------
    scene_json
        Serialized SceneGraph JSON from ``render_interactive``.
    packed_data
        Binary packed mark data from ``render_interactive``.  Embedded as
        a base64 string and passed to the WASM renderer via the standalone
        adapter.
    title
        HTML ``<title>`` content.
    embed_wasm
        When True (default), base64-encode the ``.wasm`` binary inline for
        single-file distribution.  When False, the HTML references an
        adjacent ``ferrum_wasm_bg.wasm`` sidecar file.
    """
    js_glue = _read_wasm_artifact("ferrum_wasm.js").decode("utf-8")
    css = (_WASM_DIR / "ferrum-interactive.css").read_text()

    # Inter @font-face is embedded in ferrum-interactive.css (shared by Jupyter and HTML).

    # Inline the D3 interactions bundle with exports converted to module-scoped vars.
    d3_source = (_WASM_DIR / "d3-interactions.js").read_text()
    d3_js = re.sub(
        r"export\{([^}]+)\}",
        lambda m: "var " + ",".join(
            f"{parts[-1].strip()}={parts[0].strip()}" if len(parts := p.split(" as ")) > 1
            else p.strip()
            for p in m.group(1).split(",")
        ) + ";",
        d3_source,
    )

    # Inline the anywidget JS with ESM exports stripped for standalone use.
    anywidget_source = (_WASM_DIR / "ferrum-anywidget.js").read_text()
    anywidget_js = _strip_anywidget_for_standalone(anywidget_source)

    # WASM initialization block.
    if embed_wasm:
        wasm_bytes = _read_wasm_artifact("ferrum_wasm_bg.wasm")
        wasm_b64 = base64.b64encode(wasm_bytes).decode("ascii")
        wasm_init_block = (
            "const wasmB64 = '{b64}';\n"
            "  const raw = atob(wasmB64);\n"
            "  const wasmBytes = new Uint8Array(raw.length);\n"
            "  for (let i = 0; i < raw.length; i++) wasmBytes[i] = raw.charCodeAt(i);\n"
            "  await __wbg_init({{ module_or_path: wasmBytes }});"
        ).format(b64=wasm_b64)
    else:
        wasm_init_block = "await __wbg_init();"

    # Escape scene JSON for embedding in a JS template literal.
    escaped_json = (
        scene_json.replace("\\", "\\\\")
        .replace("</", "<\\/")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    # Packed data as base64 for the standalone adapter.
    packed_b64 = base64.b64encode(packed_data).decode("ascii") if packed_data else ""

    # Interaction config for the standalone adapter.
    interaction_config = _extract_interaction_config(scene_json)
    # Escape for embedding in a JS single-quoted string.
    interaction_config_escaped = interaction_config.replace("\\", "\\\\").replace("'", "\\'")

    # Background color for the HTML body.
    bg_css = _extract_background_css(scene_json)

    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n"
        f'<body style="background:{bg_css};margin:0;display:flex;'
        'justify-content:center;align-items:center;min-height:100vh">\n'
        f'<div id="ferrum-root" style="background:{bg_css}"></div>\n'
        '<script type="module">\n'
        f"{js_glue}\n"
        "\n"
        f"{d3_js}\n"
        "\n"
        f"{anywidget_js}\n"
        "\n"
        f"const SCENE_JSON = `{escaped_json}`;\n"
        "\n"
        "async function main() {\n"
        f"  {wasm_init_block}\n"
        "\n"
        "  const container = document.getElementById('ferrum-root');\n"
        f"  const adapter = createStandaloneAdapter('{packed_b64}', "
        f"'{interaction_config_escaped}');\n"
        "  await _render(container, SCENE_JSON, adapter);\n"
        "}\n"
        "\n"
        "main().catch(e => {\n"
        "  console.error('ferrum render error:', e);\n"
        "  document.getElementById('ferrum-root').textContent = 'Render error: ' + e;\n"
        "});\n"
        "</script>\n"
        "</body>\n"
        "</html>"
    )
