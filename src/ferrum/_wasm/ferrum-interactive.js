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

  // Tooltip element
  const tooltip = document.createElement("div");
  tooltip.className = "ferrum-tooltip";
  tooltip.style.position = "absolute";
  tooltip.style.pointerEvents = "none";
  tooltip.style.opacity = "0";
  tooltip.style.transition = "opacity 0.1s ease";
  container.appendChild(tooltip);

  // Tooltip + href event handling
  canvas.style.pointerEvents = "auto";
  const marks = scene.panels ? scene.panels.flatMap(p => p.marks || []) : [];

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hit = findTooltipHit(marks, x, y);
    if (hit) {
      tooltip.replaceChildren();
      const table = document.createElement("table");
      for (const field of hit.fields) {
        const row = document.createElement("tr");
        const name = document.createElement("td");
        name.textContent = field.name;
        name.style.fontWeight = "bold";
        name.style.paddingRight = "6px";
        const val = document.createElement("td");
        val.textContent = field.value;
        row.appendChild(name);
        row.appendChild(val);
        table.appendChild(row);
      }
      tooltip.appendChild(table);
      tooltip.style.left = (x + 12) + "px";
      tooltip.style.top = (y - 12) + "px";
      tooltip.style.opacity = "1";
    } else {
      tooltip.style.opacity = "0";
    }
  });

  canvas.addEventListener("mouseleave", () => {
    tooltip.style.opacity = "0";
  });

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const href = findHrefHit(marks, x, y);
    if (href) {
      window.open(href, "_blank", "noopener,noreferrer");
    }
  });

  const observer = new ResizeObserver(() => {
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    canvas.width = w;
    canvas.height = h;
    renderer.resize(w, h);
  });
  observer.observe(container);

  return { renderer, canvas, overlay, tooltip };
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

function findTooltipHit(marks, x, y) {
  for (let bi = marks.length - 1; bi >= 0; bi--) {
    const batch = marks[bi];
    if (!batch.tooltips) continue;
    const nodes = batch.nodes || [];
    for (let ni = nodes.length - 1; ni >= 0; ni--) {
      if (nodeContains(nodes[ni], x, y)) {
        const tip = batch.tooltips[ni];
        if (tip) return tip;
      }
    }
  }
  return null;
}

function findHrefHit(marks, x, y) {
  for (let bi = marks.length - 1; bi >= 0; bi--) {
    const batch = marks[bi];
    if (!batch.hrefs) continue;
    const nodes = batch.nodes || [];
    for (let ni = nodes.length - 1; ni >= 0; ni--) {
      if (nodeContains(nodes[ni], x, y)) {
        const href = batch.hrefs[ni];
        if (href) return href;
      }
    }
  }
  return null;
}

function nodeContains(node, x, y) {
  if (!node || !node.type) return false;
  switch (node.type) {
    case "circle": {
      const dx = x - node.cx;
      const dy = y - node.cy;
      return dx * dx + dy * dy <= node.r * node.r;
    }
    case "rect":
      return x >= node.x && x <= node.x + node.w &&
             y >= node.y && y <= node.y + node.h;
    default:
      return false;
  }
}

export default { renderChart, render };
