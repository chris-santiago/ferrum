/**
 * ferrum-interactive.js — ESM glue module for ferrum's WASM renderer.
 *
 * Two modes:
 *   1. Standalone HTML — called from inline <script> with sceneJson + container element
 *   2. anywidget (Jupyter) — called via render({ model, el }) with model state sync
 */

let wasmInit = null;
let WasmRenderer = null;

async function ensureWasm(wasmModule) {
  if (WasmRenderer) return;
  const mod = wasmModule || await import("./ferrum_wasm.js");
  await mod.default();
  WasmRenderer = mod.WasmRenderer;
  wasmInit = mod;
}

function placeTextOverlay(overlay, textElements, offsetX = 0, offsetY = 0) {
  overlay.replaceChildren();
  for (const t of textElements) {
    const div = document.createElement("div");
    div.className = "ferrum-text";
    div.style.position = "absolute";
    div.style.left = (t.x + offsetX) + "px";
    div.style.top = (t.y + offsetY) + "px";
    div.style.fontSize = t.fontSize + "px";
    div.style.fontWeight = t.fontWeight;
    div.style.fontFamily = t.fontFamily;
    div.style.color = t.color;
    div.style.whiteSpace = "nowrap";
    div.style.pointerEvents = "none";
    div.style.lineHeight = "1";

    if (t.anchor === "center") {
      div.style.transform = `translateX(-50%) rotate(${t.angle}deg)`;
    } else if (t.anchor === "end") {
      div.style.transform = `translateX(-100%) rotate(${t.angle}deg)`;
    } else if (t.angle !== 0) {
      div.style.transform = `rotate(${t.angle}deg)`;
    }

    if (t.baseline === "middle") {
      div.style.transform = (div.style.transform || "") + " translateY(-50%)";
    } else if (t.baseline === "bottom") {
      div.style.transform = (div.style.transform || "") + " translateY(-100%)";
    } else if (t.baseline === "alphabetic") {
      div.style.transform = (div.style.transform || "") + " translateY(-85%)";
    }

    div.textContent = t.content;
    overlay.appendChild(div);
  }
}

/**
 * Render a ferrum chart into a container element.
 *
 * @param {HTMLElement} container - The DOM element to render into.
 * @param {string} sceneJson - Serialized SceneGraph JSON.
 * @param {object} [options] - Optional settings.
 * @param {object} [options.wasmModule] - Pre-loaded WASM module (for inline HTML).
 */
export async function renderChart(container, sceneJson, options = {}) {
  await ensureWasm(options.wasmModule);

  container.style.position = "relative";
  container.replaceChildren();

  const scene = JSON.parse(sceneJson);
  const width = scene.width || 640;
  const height = scene.height || 480;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.style.display = "block";
  container.appendChild(canvas);

  const overlay = document.createElement("div");
  overlay.className = "ferrum-overlay";
  overlay.style.position = "absolute";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.width = width + "px";
  overlay.style.height = height + "px";
  overlay.style.pointerEvents = "none";
  container.appendChild(overlay);

  const renderer = await WasmRenderer.create(canvas);
  const textJson = renderer.loadScene(sceneJson);
  const textElements = JSON.parse(textJson);
  placeTextOverlay(overlay, textElements);

  const observer = new ResizeObserver(() => {
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    canvas.width = w;
    canvas.height = h;
    renderer.resize(w, h);
  });
  observer.observe(container);

  return { renderer, canvas, overlay };
}

/**
 * anywidget render function.
 * Called by anywidget when running in Jupyter.
 */
export function render({ model, el }) {
  const container = document.createElement("div");
  el.appendChild(container);

  const sceneJson = model.get("scene_json");
  if (sceneJson) {
    renderChart(container, sceneJson);
  }

  model.on("change:scene_json", () => {
    const updated = model.get("scene_json");
    if (updated) {
      renderChart(container, updated);
    }
  });
}

export default { renderChart, render };
