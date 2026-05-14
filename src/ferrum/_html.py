"""Assemble self-contained HTML files for the WASM renderer."""

from __future__ import annotations

import base64
from pathlib import Path


def _read_wasm_artifact(name: str) -> bytes:
    wasm_dir = Path(__file__).parent / "_wasm"
    artifact = wasm_dir / name
    if not artifact.exists():
        raise FileNotFoundError(
            f"WASM artifact {name!r} not found at {artifact}. "
            "Run: wasm-pack build crates/ferrum-wasm --target web "
            "--out-dir ../../src/ferrum/_wasm/"
        )
    return artifact.read_bytes()


def assemble_html(
    scene_json: str,
    *,
    title: str = "Ferrum chart",
    embed_wasm: bool = True,
) -> str:
    """Build a self-contained HTML string that renders a chart via WASM.

    Parameters
    ----------
    scene_json
        Serialized SceneGraph JSON from ``render_interactive``.
    title
        HTML ``<title>`` content.
    embed_wasm
        When True (default), base64-encode the ``.wasm`` binary inline for
        single-file distribution.  When False, the HTML references an
        adjacent ``ferrum_wasm_bg.wasm`` sidecar file.
    """
    js_glue = _read_wasm_artifact("ferrum_wasm.js").decode("utf-8")
    css = (Path(__file__).parent / "_wasm" / "ferrum-interactive.css").read_text()

    if embed_wasm:
        wasm_bytes = _read_wasm_artifact("ferrum_wasm_bg.wasm")
        wasm_b64 = base64.b64encode(wasm_bytes).decode("ascii")
        wasm_init_block = (
            "const wasmB64 = '{}';\\n"
            "  const wasmBytes = Uint8Array.from(atob(wasmB64), c => c.charCodeAt(0));\\n"
            "  await __wbg_init(wasmBytes);"
        ).format(wasm_b64)
    else:
        wasm_init_block = "await __wbg_init();"

    escaped_json = (
        scene_json
        .replace("\\", "\\\\")
        .replace("</", "<\\/")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n"
        "<body>\n"
        '<div id="ferrum-root"></div>\n'
        '<script type="module">\n'
        f"{js_glue}\n"
        "\n"
        f"const SCENE_JSON = `{escaped_json}`;\n"
        "\n"
        "async function main() {\n"
        f"  {wasm_init_block}\n"
        "\n"
        "  const container = document.getElementById('ferrum-root');\n"
        "  container.style.position = 'relative';\n"
        "  const scene = JSON.parse(SCENE_JSON);\n"
        "  const width = scene.width || 640;\n"
        "  const height = scene.height || 480;\n"
        "\n"
        "  const canvas = document.createElement('canvas');\n"
        "  canvas.width = width;\n"
        "  canvas.height = height;\n"
        "  canvas.style.display = 'block';\n"
        "  container.appendChild(canvas);\n"
        "\n"
        "  const overlay = document.createElement('div');\n"
        "  overlay.className = 'ferrum-overlay';\n"
        "  overlay.style.position = 'absolute';\n"
        "  overlay.style.top = '0';\n"
        "  overlay.style.left = '0';\n"
        "  overlay.style.width = width + 'px';\n"
        "  overlay.style.height = height + 'px';\n"
        "  overlay.style.pointerEvents = 'none';\n"
        "  container.appendChild(overlay);\n"
        "\n"
        "  const renderer = await WasmRenderer.create(canvas);\n"
        "  const textJson = renderer.loadScene(SCENE_JSON);\n"
        "  const textElements = JSON.parse(textJson);\n"
        "\n"
        "  for (const t of textElements) {\n"
        "    const div = document.createElement('div');\n"
        "    div.className = 'ferrum-text';\n"
        "    div.style.position = 'absolute';\n"
        "    div.style.left = t.x + 'px';\n"
        "    div.style.top = t.y + 'px';\n"
        "    div.style.fontSize = t.fontSize + 'px';\n"
        "    div.style.fontWeight = t.fontWeight;\n"
        "    div.style.fontFamily = t.fontFamily;\n"
        "    div.style.color = t.color;\n"
        "    div.style.whiteSpace = 'nowrap';\n"
        "    div.style.pointerEvents = 'none';\n"
        "    div.style.lineHeight = '1';\n"
        "    div.textContent = t.content;\n"
        "    overlay.appendChild(div);\n"
        "  }\n"
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
