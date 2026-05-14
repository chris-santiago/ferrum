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
            "const wasmB64 = '{b64}';\n"
            "  const raw = atob(wasmB64);\n"
            "  const wasmBytes = new Uint8Array(raw.length);\n"
            "  for (let i = 0; i < raw.length; i++) wasmBytes[i] = raw.charCodeAt(i);\n"
            "  await __wbg_init(wasmBytes);"
        ).format(b64=wasm_b64)
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
        "\n"
        "  // Tooltip + href\n"
        "  const tip = document.createElement('div');\n"
        "  tip.className = 'ferrum-tooltip';\n"
        "  tip.style.position = 'absolute';\n"
        "  tip.style.pointerEvents = 'none';\n"
        "  tip.style.opacity = '0';\n"
        "  tip.style.transition = 'opacity 0.1s ease';\n"
        "  container.appendChild(tip);\n"
        "  const marks = scene.panels ? scene.panels.flatMap(p => p.marks || []) : [];\n"
        "  function hitTest(x, y) {\n"
        "    for (let bi = marks.length - 1; bi >= 0; bi--) {\n"
        "      const b = marks[bi];\n"
        "      if (!b.nodes) continue;\n"
        "      for (let ni = b.nodes.length - 1; ni >= 0; ni--) {\n"
        "        const n = b.nodes[ni];\n"
        "        let hit = false;\n"
        "        if (n.type === 'circle') {\n"
        "          const dx = x - n.cx, dy = y - n.cy;\n"
        "          hit = dx*dx + dy*dy <= n.r*n.r;\n"
        "        } else if (n.type === 'rect') {\n"
        "          hit = x >= n.x && x <= n.x+n.w && y >= n.y && y <= n.y+n.h;\n"
        "        }\n"
        "        if (hit) return { batch: b, idx: ni };\n"
        "      }\n"
        "    }\n"
        "    return null;\n"
        "  }\n"
        "  canvas.style.pointerEvents = 'auto';\n"
        "  canvas.addEventListener('mousemove', e => {\n"
        "    const r = canvas.getBoundingClientRect();\n"
        "    const h = hitTest(e.clientX - r.left, e.clientY - r.top);\n"
        "    if (h && h.batch.tooltips && h.batch.tooltips[h.idx]) {\n"
        "      const t = h.batch.tooltips[h.idx];\n"
        "      tip.replaceChildren();\n"
        "      const tbl = document.createElement('table');\n"
        "      for (const f of t.fields) {\n"
        "        const tr = document.createElement('tr');\n"
        "        const k = document.createElement('td');\n"
        "        k.textContent = f.name; k.style.fontWeight = 'bold'; k.style.paddingRight = '6px';\n"
        "        const v = document.createElement('td'); v.textContent = f.value;\n"
        "        tr.appendChild(k); tr.appendChild(v); tbl.appendChild(tr);\n"
        "      }\n"
        "      tip.appendChild(tbl);\n"
        "      tip.style.left = (e.clientX - r.left + 12) + 'px';\n"
        "      tip.style.top = (e.clientY - r.top - 12) + 'px';\n"
        "      tip.style.opacity = '1';\n"
        "    } else { tip.style.opacity = '0'; }\n"
        "  });\n"
        "  canvas.addEventListener('mouseleave', () => { tip.style.opacity = '0'; });\n"
        "  canvas.addEventListener('click', e => {\n"
        "    const r = canvas.getBoundingClientRect();\n"
        "    const h = hitTest(e.clientX - r.left, e.clientY - r.top);\n"
        "    if (h && h.batch.hrefs && h.batch.hrefs[h.idx]) {\n"
        "      window.open(h.batch.hrefs[h.idx], '_blank', 'noopener,noreferrer');\n"
        "    }\n"
        "  });\n"
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
