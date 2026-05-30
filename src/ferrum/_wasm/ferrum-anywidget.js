// ── anywidget render entry point ──────────────────────────────────────────
// This file is read by _interactive.py which replaces __B64__ with the
// base64-encoded ferrum_wasm_bg.wasm blob before sending to the browser.
// Keep all JS here — never embed JS strings in Python.
//
// Adapter pattern: _render() accepts an adapter object (not a raw model).
// Two adapters exist:
//   1. Jupyter adapter — constructed in render() for anywidget
//   2. Standalone adapter — exported as createStandaloneAdapter() for HTML exports

const _B64 = '__B64__';
const _raw = atob(_B64);
const _bytes = new Uint8Array(_raw.length);
for (let i = 0; i < _raw.length; i++) _bytes[i] = _raw.charCodeAt(i);

let _ready = false, _initP = null;
async function _ensureWasm() {
  if (_ready) return;
  if (!_initP) _initP = __wbg_init({ module_or_path: _bytes }).then(() => { _ready = true; });
  await _initP;
}

// D3 interactions (brush, zoom, select, zoomTransform, zoomIdentity, pointer) are provided
// by d3-interactions.js which is inlined before this file in both standalone
// HTML and Jupyter ESM builds.  The D3 bundle's `export { ... }` is stripped
// by the assembler, leaving the symbols in module scope.

// ── PNG pHYs DPI injection ────────────────────────────────────────────────
// Splices a pHYs chunk into a PNG data URL so the exported file carries the
// screen DPI as metadata.  The chunk is inserted at byte offset 33 (immediately
// after the 8-byte PNG signature and the 25-byte IHDR chunk), which is the
// required position before any IDAT chunks.
//
// pHYs chunk layout (21 bytes total):
//   4 bytes — chunk data length = 9 (big-endian uint32)
//   4 bytes — chunk type "pHYs" (0x70 0x48 0x59 0x73)
//   4 bytes — pixels-per-unit X (big-endian uint32)
//   4 bytes — pixels-per-unit Y (big-endian uint32)
//   1 byte  — unit specifier = 1 (metres)
//   4 bytes — CRC32 of type + data (13 bytes)
function injectPHYs(dataUrl, dprOverride) {
  // Decode base64 payload.
  const b64 = dataUrl.slice('data:image/png;base64,'.length);
  const bin = atob(b64);
  const src = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) src[i] = bin.charCodeAt(i);

  // Compute pixels-per-metre from the effective DPR (clamped to GPU limits).
  const dpi = (dprOverride || window.devicePixelRatio || 1) * 72;
  const ppm = Math.round(dpi * 39.3701);

  // Build the 9-byte chunk data: X ppm (4), Y ppm (4), unit=1 (1).
  const data = new Uint8Array(9);
  const dv = new DataView(data.buffer);
  dv.setUint32(0, ppm, false);
  dv.setUint32(4, ppm, false);
  data[8] = 1;

  // Build chunk type bytes.
  const type = new Uint8Array([0x70, 0x48, 0x59, 0x73]); // "pHYs"

  // CRC32 over type (4 bytes) + data (9 bytes) = 13 bytes.
  const crcBuf = new Uint8Array(13);
  crcBuf.set(type, 0);
  crcBuf.set(data, 4);
  const crc = _crc32(crcBuf);

  // Assemble the 21-byte pHYs chunk.
  const chunk = new Uint8Array(21);
  const cv = new DataView(chunk.buffer);
  cv.setUint32(0, 9, false);          // data length
  chunk.set(type, 4);                 // chunk type
  chunk.set(data, 8);                 // chunk data
  cv.setUint32(17, crc, false);       // CRC

  // Splice: everything before offset 33 | pHYs chunk | rest.
  const out = new Uint8Array(src.length + 21);
  out.set(src.subarray(0, 33), 0);
  out.set(chunk, 33);
  out.set(src.subarray(33), 54);

  // Re-encode to base64 data URL.
  let outBin = '';
  for (let i = 0; i < out.length; i++) outBin += String.fromCharCode(out[i]);
  return 'data:image/png;base64,' + btoa(outBin);
}

// Standard reflected CRC32 (PNG polynomial 0xEDB88320).
function _crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc & 1) ? (crc >>> 1) ^ 0xEDB88320 : crc >>> 1;
    }
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

// ── SVG text placement ───────────────────────────────────────────────────
function _placeTextSvg(svgEl, texts) {
  const svg = select(svgEl);
  svg.selectAll('text.ferrum-label').remove();
  for (const t of texts) {
    const anchor = t.anchor === 'center' ? 'middle' : t.anchor;
    let baseline;
    switch (t.baseline) {
      case 'top': baseline = 'hanging'; break;
      case 'middle': baseline = 'central'; break;
      case 'bottom': baseline = 'text-after-edge'; break;
      case 'alphabetic': default: baseline = 'auto'; break;
    }
    const el = svg.append('text')
      .attr('class', 'ferrum-label')
      .attr('x', t.x)
      .attr('y', t.y)
      .attr('text-anchor', anchor)
      .attr('dominant-baseline', baseline)
      .attr('font-size', t.fontSize + 'px')
      .attr('font-weight', t.fontWeight)
      .attr('font-family', t.fontFamily)
      .attr('fill', t.color)
      .attr('pointer-events', 'none')
      .text(t.content);
    if (t.angle) {
      el.attr('transform', `rotate(${t.angle}, ${t.x}, ${t.y})`);
    }
  }
}

// ── Raw SVG fragment injection ────────────────────────────────────────────
// Injects verbatim SVG fragments from SceneNode::Raw into the text-overlay
// <svg>.  Two groups are maintained:
//   .ferrum-raw-chrome — fixed during pan/zoom (legend colorbar, inset)
//   .ferrum-raw-data   — receives the same canvas transform (data-anchored)
//
// ID namespacing: ALL fragments are scanned in a SINGLE pass to build one
// shared id map, so a <defs> definition in fragment A and its url(#...) or
// href="#..." consumer in fragment B both resolve to the SAME namespaced id.
// The legend colorbar emits the gradient definition and its consuming rect as
// two separate Raw nodes; per-fragment namespacing would leave the consumer's
// url(#...) reference pointing at a non-existent id.
//
// Namespacing contract:
//   - Every id="X" defined across all fragments is mapped to
//     id="ferrum-raw-{loadIdx}-X" where {loadIdx} is a per-loadScene counter.
//   - In every fragment: id="X" → id="ferrum-raw-{loadIdx}-X"
//   - In every fragment: url(#X) → url(#ferrum-raw-{loadIdx}-X)
//   - In every fragment: href="#X" → href="#ferrum-raw-{loadIdx}-X"
//   - In every fragment: xlink:href="#X" → xlink:href="#ferrum-raw-{loadIdx}-X"
//   - Only ids that are actually defined somewhere in the fragment set are
//     rewritten.  External URLs, data URIs, and unrelated # anchors are left
//     untouched.
//
// Injection is done ONCE per loadScene call.  On pan/zoom ticks, only the
// data group's transform attribute is updated — no DOM re-parse, no re-inject.
//
// Transform model: the data Raw group carries the same CSS-pixel affine as
// the GPU canvas.  D3-zoom (k, tx, ty) defines x' = x*k + tx, y' = y*k + ty
// in CSS pixels.  The GPU renderer applies the same affine (via setTransform).
// Both operate in CSS pixel space; DPR scaling is only in the canvas backing
// store and does not affect the SVG coordinate system.  Text labels are
// re-generated with new pixel positions on each tick (Rust setTransform), so
// they do not need a group transform — their approach is incompatible with Raw
// (opaque SVG), hence the separate transform-on-group approach here.
//
// The overlay remains pointer-events:none at all times.

// Build the global id→namespaced-id map from all fragments.
// loadIdx is incremented by _buildRawOverlay on each loadScene call to ensure
// ids from stale scenes can't collide with the new scene.
function _buildIdMap(rawFragments, loadIdx) {
  const idMap = new Map(); // id → namespaced id
  const idDefRe = /\bid="([^"]+)"/g;
  for (const frag of rawFragments) {
    let m;
    idDefRe.lastIndex = 0;
    while ((m = idDefRe.exec(frag.svg)) !== null) {
      const id = m[1];
      if (!idMap.has(id)) {
        idMap.set(id, `ferrum-raw-${loadIdx}-${id}`);
      }
    }
  }
  return idMap;
}

// Rewrite a single SVG string using the shared id map.
function _applyIdMap(svgStr, idMap) {
  if (idMap.size === 0) return svgStr;
  let result = svgStr;
  for (const [id, namespaced] of idMap) {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // id="X" → id="namespaced"
    result = result.replace(
      new RegExp(`\\bid="${escaped}"`, 'g'),
      `id="${namespaced}"`
    );
    // url(#X) → url(#namespaced)
    result = result.replace(
      new RegExp(`url\\(#${escaped}\\)`, 'g'),
      `url(#${namespaced})`
    );
    // href="#X" → href="#namespaced"  (only exact #X, not data URIs or external)
    result = result.replace(
      new RegExp(`\\bhref="#${escaped}"`, 'g'),
      `href="#${namespaced}"`
    );
    // xlink:href="#X" → xlink:href="#namespaced"
    result = result.replace(
      new RegExp(`\\bxlink:href="#${escaped}"`, 'g'),
      `xlink:href="#${namespaced}"`
    );
  }
  return result;
}

// Build and inject the chrome/data Raw groups once per loadScene.
// Returns { chromeG, dataG } for later transform-only updates, or null if
// rawFragments is empty.
let _rawLoadCounter = 0;
function _buildRawOverlay(svgEl, rawFragments) {
  // Remove any previously injected raw groups from a prior loadScene.
  svgEl.querySelectorAll('g.ferrum-raw-chrome, g.ferrum-raw-data').forEach(g => g.remove());

  if (!rawFragments || rawFragments.length === 0) return null;

  const loadIdx = _rawLoadCounter++;
  const ns = 'http://www.w3.org/2000/svg';

  // Single-pass: build the global id map from all fragments.
  const idMap = _buildIdMap(rawFragments, loadIdx);

  // Chrome group: fixed, no pan/zoom transform.
  const chromeG = document.createElementNS(ns, 'g');
  chromeG.setAttribute('class', 'ferrum-raw-chrome');
  chromeG.style.pointerEvents = 'none';

  // Data group: receives the canvas pan/zoom transform.
  const dataG = document.createElementNS(ns, 'g');
  dataG.setAttribute('class', 'ferrum-raw-data');
  dataG.style.pointerEvents = 'none';

  for (const frag of rawFragments) {
    // Apply the shared id map to this fragment.
    const namespacedSvg = _applyIdMap(frag.svg, idMap);
    // Wrap in a <g> and insert via innerHTML — only the fragment content,
    // not a full SVG document.
    const wrapper = document.createElementNS(ns, 'g');
    // Use innerHTML on a detached SVG element to parse fragment markup.
    // This is intentional: the fragment is trusted ferrum-generated SVG,
    // not user-controlled content, and verbatim injection is the spec'd approach.
    const parseEl = document.createElementNS(ns, 'svg');
    parseEl.innerHTML = namespacedSvg;
    while (parseEl.firstChild) {
      wrapper.appendChild(parseEl.firstChild);
    }
    if (frag.anchor === 'data') {
      dataG.appendChild(wrapper);
    } else {
      // 'chrome' (default) — fixed overlay.
      chromeG.appendChild(wrapper);
    }
  }

  // Append both groups. Chrome painted first, data on top.
  svgEl.appendChild(chromeG);
  svgEl.appendChild(dataG);

  return { chromeG, dataG };
}

// Build a CSS matrix transform string from D3-zoom k/tx/ty values.
// Matches the GPU affine: x' = x*k + tx, y' = y*k + ty (CSS pixel space).
function _zoomMatrix(k, tx, ty) {
  return `matrix(${k},0,0,${k},${tx},${ty})`;
}

// ── SVG icon builder (safe DOM construction, no innerHTML) ───────────────
// All icons are 16x16, stroke-based, currentColor. Built via DOM API to
// avoid innerHTML and potential XSS vectors.
function _svgIcon(children) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '16');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.5');
  for (const child of children) {
    const el = document.createElementNS(ns, child.tag);
    for (const [k, v] of Object.entries(child.attrs || {})) {
      el.setAttribute(k, v);
    }
    svg.appendChild(el);
  }
  return svg;
}

function _iconPan() {
  return _svgIcon([
    { tag: 'path', attrs: { d: 'M8 2v12M2 8h12M8 2l-2 2M8 2l2 2M8 14l-2-2M8 14l2-2M2 8l2-2M2 8l2 2M14 8l-2-2M14 8l-2 2' } },
  ]);
}

function _iconBoxZoom() {
  return _svgIcon([
    { tag: 'circle', attrs: { cx: '7', cy: '7', r: '4' } },
    { tag: 'path', attrs: { d: 'M10 10l4 4' } },
    { tag: 'path', attrs: { d: 'M5 7h4M7 5v4' } },
  ]);
}

function _iconSelect() {
  return _svgIcon([
    { tag: 'rect', attrs: { x: '2', y: '2', width: '12', height: '12', rx: '1', 'stroke-dasharray': '2 2' } },
    { tag: 'path', attrs: { d: 'M6 8h4M8 6v4', 'stroke-dasharray': 'none' } },
  ]);
}

function _iconReset() {
  return _svgIcon([
    { tag: 'path', attrs: { d: 'M3 8a5 5 0 1 1 1 3' } },
    { tag: 'path', attrs: { d: 'M3 11V8h3' } },
  ]);
}

function _iconSave() {
  return _svgIcon([
    { tag: 'path', attrs: { d: 'M8 2v8M8 10l-3-3M8 10l3-3M3 13h10' } },
  ]);
}

// ── Toolbar creation ─────────────────────────────────────────────────────
function _createToolbar(setMode, onReset, onSave, defaultMode) {
  const toolbar = document.createElement('div');
  toolbar.className = 'ferrum-toolbar';

  const tools = [
    { mode: 'pan', title: 'Pan (P)', iconFn: _iconPan },
    { mode: 'boxzoom', title: 'Box Zoom (Z)', iconFn: _iconBoxZoom },
    { mode: 'select', title: 'Box Select (S)', iconFn: _iconSelect },
    null, // separator
    { action: 'reset', title: 'Reset (R)', iconFn: _iconReset },
    { action: 'save', title: 'Save PNG', iconFn: _iconSave },
  ];

  for (const t of tools) {
    if (t === null) {
      const sep = document.createElement('div');
      sep.className = 'ferrum-tool-separator';
      toolbar.appendChild(sep);
      continue;
    }
    const btn = document.createElement('button');
    btn.className = 'ferrum-tool';
    btn.title = t.title;
    btn.appendChild(t.iconFn());
    if (t.mode) {
      btn.dataset.mode = t.mode;
      if (t.mode === defaultMode) btn.classList.add('active');
      btn.addEventListener('click', () => setMode(t.mode));
    } else if (t.action === 'reset') {
      btn.addEventListener('click', onReset);
    } else if (t.action === 'save') {
      btn.addEventListener('click', onSave);
    }
    toolbar.appendChild(btn);
  }

  return toolbar;
}

// ── Adapter interface (duck-typed) ───────────────────────────────────────
// {
//   getPackedData()           → Uint8Array
//   getInteractionConfig()    → string (JSON)
//   onSelectionChange(state)  → void (called when selection changes)
//   onZoomChange(state)       → void (called when zoom changes)
// }

async function _render(container, sceneJson, adapter) {
  // Clean up resources from a previous _render() call on the same container.
  if (container._ferrumCleanup) {
    container._ferrumCleanup();
    container._ferrumCleanup = null;
  }
  container.replaceChildren();

  const scene = JSON.parse(sceneJson);
  const w = scene.width || 640, h = scene.height || 480;

  // ── Outer flex container ──────────────────────────────────────────
  container.className = 'ferrum-container';
  container.setAttribute('tabindex', '0');
  container.style.display = 'flex';
  container.style.outline = 'none';

  // ── Inner chart wrapper (position:relative for canvas + SVG + tooltip) ─
  const chartWrapper = document.createElement('div');
  chartWrapper.style.position = 'relative';
  container.appendChild(chartWrapper);

  // ── Canvas ───────────────────────────────────────────────────────
  // Start at CSS-pixel size (safe for any chart width). After the GPU
  // renderer is created we query maxTextureSize and resize to the highest
  // DPR the hardware supports.
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  canvas.style.display = 'block';
  chartWrapper.appendChild(canvas);

  // ── SVG overlay for text labels ──────────────────────────────────
  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgEl.setAttribute('width', w);
  svgEl.setAttribute('height', h);
  // SVG inherits CSS @font-face from the parent HTML document (Inter).
  svgEl.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
  chartWrapper.appendChild(svgEl);

  // ── Tooltip ──────────────────────────────────────────────────────
  const tip = document.createElement('div');
  tip.className = 'ferrum-tooltip';
  Object.assign(tip.style, { position: 'absolute', pointerEvents: 'none',
    opacity: '0', transition: 'opacity 0.1s ease' });
  chartWrapper.appendChild(tip);

  // ── Interaction config ────────────────────────────────────────────
  const cfg = JSON.parse(adapter.getInteractionConfig());
  const _hasPointSelections = (cfg.selections || []).some(s => s.type === 'point');
  const hasInterval = (cfg.selections || []).some(s => s.type === 'interval');

  // ── GPU init (may fail when WebGPU/WebGL context limit exceeded) ──
  // Effective DPR after clamping to GPU max texture size. Stored in
  // closure scope so ResizeObserver and pHYs injection can reuse it.
  let effectiveDpr = 1;
  let maxTex = 2048;
  let renderer = null;
  // Raw overlay groups built once per loadScene.  Only dataG.transform is
  // mutated on zoom/pan ticks — no re-parsing or re-injection.
  let _rawOverlay = null; // { chromeG, dataG } or null
  try {
    await _ensureWasm();
    renderer = await WasmRenderer.create(canvas);
    // Query actual GPU limit and compute the highest safe DPR.
    maxTex = renderer.maxTextureSize ? renderer.maxTextureSize() : 2048;
    const rawDpr = window.devicePixelRatio || 1;
    effectiveDpr = Math.min(rawDpr, maxTex / Math.max(w, h));
    if (effectiveDpr < 1) effectiveDpr = 1;
    // Resize canvas to DPR-clamped physical pixels.
    const physW = Math.round(w * effectiveDpr);
    const physH = Math.round(h * effectiveDpr);
    if (physW !== canvas.width || physH !== canvas.height) {
      canvas.width = physW;
      canvas.height = physH;
      renderer.resize(physW, physH);
    }
    const packedArr = adapter.getPackedData();
    const overlayJson = renderer.loadScene(sceneJson, packedArr);
    // loadScene returns {"text": [...], "raw": [...]} since Task 6 (item 19).
    const overlay = JSON.parse(overlayJson);
    const textArr = Array.isArray(overlay) ? overlay : (overlay.text || []);
    const rawFragments = Array.isArray(overlay) ? [] : (overlay.raw || []);
    _placeTextSvg(svgEl, textArr);
    // Inject raw fragments once; subsequent zoom/pan ticks only update transform.
    _rawOverlay = _buildRawOverlay(svgEl, rawFragments);
  } catch (e) {
    console.warn('[ferrum] GPU init failed — rendering disabled, tooltips still active.', e);
  }

  // ── Mode switching ────────────────────────────────────────────────
  const defaultMode = hasInterval ? 'select' : 'pan';
  let currentMode = defaultMode;
  container.dataset.mode = currentMode;

  function setMode(mode) {
    currentMode = mode;
    container.dataset.mode = mode;
    container.querySelectorAll('.ferrum-tool[data-mode]').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    // In pan mode, disable pointer-events on SVG + brush overlays so
    // hover/click/double-click events reach chartWrapper unblocked.
    // In select/boxzoom, enable so D3 brush can capture drag gestures.
    svgEl.style.pointerEvents = (mode === 'pan') ? 'none' : 'all';
  }

  // ── D3-zoom on chart wrapper ──────────────────────────────────────
  let _zoomDebounceId = null;
  const zoomBehavior = zoom()
    .scaleExtent([0.1, 50])
    .filter(event => {
      // Always allow wheel-zoom.
      if (event.type === 'wheel') return true;
      // Only pan mode allows drag-zoom.
      if (currentMode !== 'pan') return false;
      // Only left-button drags.
      return !event.button;
    })
    .on('zoom', event => {
      if (!renderer) return;
      const { k, x, y } = event.transform;
      try {
        const textJson = renderer.setTransform(k, x, y);
        _placeTextSvg(svgEl, JSON.parse(textJson));
        // Update only the data group's transform — no re-injection.
        if (_rawOverlay) {
          _rawOverlay.dataG.setAttribute('transform', _zoomMatrix(k, x, y));
        }
      } catch (err) { /* GPU not ready */ }
      // Debounced adapter callback for Jupyter zoom rebuild.
      clearTimeout(_zoomDebounceId);
      _zoomDebounceId = setTimeout(() => {
        adapter.onZoomChange({ '0': { k, x, y } });
      }, 400);
    });

  // Attach zoom to the inner chart wrapper (not the outer flex container)
  // so toolbar button clicks don't trigger zoom events.
  select(chartWrapper).call(zoomBehavior);

  // ── Reset handler ─────────────────────────────────────────────────
  function onReset() {
    if (!renderer) return;
    select(chartWrapper).call(zoomBehavior.transform, zoomIdentity);
    // Reset data Raw group to identity — remove the transform attribute.
    if (_rawOverlay) {
      _rawOverlay.dataG.removeAttribute('transform');
    }
    try { const stateJson = renderer.clearSelections(); adapter.onSelectionChange(JSON.parse(stateJson)); } catch (_) {}
  }

  // Double-click: reset zoom to identity.
  select(chartWrapper).on('dblclick.zoom', onReset);

  // ── Save PNG handler ──────────────────────────────────────────────
  // Composites the GPU canvas (marks, gridlines) with the SVG text overlay
  // (axis labels, title, legend text) into a single PNG.  WebGPU canvases
  // clear after present(), so we render → wait one RAF → blit to 2D canvas.
  // The SVG is serialized with an inlined @font-face so text renders even
  // without the parent document's CSS.
  async function onSave() {
    if (!renderer) return;
    // Disconnect ResizeObserver during save to prevent layout-triggered resize
    // events from mutating canvas dimensions during the async capture sequence.
    if (_ro) _ro.disconnect();
    try {
      renderer.renderFrame();
      await new Promise(r => requestAnimationFrame(r));

      // Export at full Retina resolution (w × rawDPR).  The GPU canvas may be
      // clamped below rawDPR by the texture limit; drawImage upscales in that
      // case.  The 2D offscreen canvas has no GPU texture limit (16k+).
      const rawDpr = window.devicePixelRatio || 1;
      const exportW = Math.round(w * rawDpr);
      const exportH = Math.round(h * rawDpr);
      const off = document.createElement('canvas');
      off.width = exportW; off.height = exportH;
      const ctx = off.getContext('2d');
      ctx.drawImage(canvas, 0, 0, exportW, exportH);

      // Composite SVG text overlay (axis labels, title, legend text).
      // The SVG is authored in CSS pixels; viewBox scales to export size.
      try {
        const svgClone = svgEl.cloneNode(true);
        svgClone.setAttribute('width', String(exportW));
        svgClone.setAttribute('height', String(exportH));
        svgClone.setAttribute('viewBox', `0 0 ${w} ${h}`);
        // Inline @font-face from the document's stylesheets so the SVG
        // renders text correctly when rasterized via Image.
        const fontRules = [];
        try {
          for (const sheet of document.styleSheets) {
            for (const rule of sheet.cssRules || []) {
              if (rule.cssText && rule.cssText.startsWith('@font-face')) {
                fontRules.push(rule.cssText);
              }
            }
          }
        } catch (_) { /* cross-origin stylesheet, skip */ }
        if (fontRules.length > 0) {
          const styleEl = document.createElementNS('http://www.w3.org/2000/svg', 'style');
          styleEl.textContent = fontRules.join('\n');
          svgClone.insertBefore(styleEl, svgClone.firstChild);
        }
        const svgXml = new XMLSerializer().serializeToString(svgClone);
        const svgBlob = new Blob([svgXml], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);
        const img = await new Promise((resolve, reject) => {
          const i = new Image();
          i.onload = () => resolve(i);
          i.onerror = reject;
          i.src = svgUrl;
        });
        ctx.drawImage(img, 0, 0, off.width, off.height);
        URL.revokeObjectURL(svgUrl);
      } catch (svgErr) {
        console.warn('[ferrum] SVG text composite failed, exporting without text:', svgErr);
      }

      const a = document.createElement('a');
      a.href = injectPHYs(off.toDataURL('image/png'), rawDpr);
      a.download = 'ferrum-chart.png';
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (err) {
      console.warn('[ferrum] save PNG error:', err);
    }
    if (_ro) _ro.observe(canvas);
  }

  // ── Toolbar (gated on cfg.toolbar !== false) ──────────────────────
  if (cfg.toolbar !== false) {
    const toolbar = _createToolbar(setMode, onReset, onSave, defaultMode);
    container.appendChild(toolbar);
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────
  function _onKeydown(e) {
    const k = e.key.toLowerCase();
    if (k === 'p') setMode('pan');
    else if (k === 'z') setMode('boxzoom');
    else if (k === 's') setMode('select');
    else if (k === 'r') onReset();
    else if (e.key === 'Escape') setMode(defaultMode);
    else return;
    e.preventDefault();
  }
  container.addEventListener('keydown', _onKeydown);

  // ── D3-brush on SVG (per-panel overlays for interval/boxzoom) ─────
  if (scene.panels) {
    // Extract brush styling from the interval selection's SelectionMark.
    let brushFill = 'rgba(51, 136, 204, 0.2)';
    let brushStroke = 'rgba(51, 136, 204, 0.6)';
    const intervalSel = (cfg.selections || []).find(s => s.type === 'interval');
    if (intervalSel && intervalSel.mark) {
      if (intervalSel.mark.fill) brushFill = intervalSel.mark.fill;
      if (intervalSel.mark.stroke) brushStroke = intervalSel.mark.stroke;
    }

    for (let pi = 0; pi < scene.panels.length; pi++) {
      const pa = scene.panels[pi].plot_area;
      if (!pa) continue;

      const panelIdx = pi;
      const brushBehavior = brush()
        .extent([[pa.x, pa.y], [pa.x + pa.w, pa.y + pa.h]])
        .filter(event => {
          // Pan mode blocks brush entirely.
          if (currentMode === 'pan') return false;
          return event.button === 0;
        });

      brushBehavior.on('end', function(event) {
        if (!renderer) return;
        if (!event.selection) return;
        const [[x0, y0], [x1, y1]] = event.selection;

        if (currentMode === 'boxzoom') {
          // Compute zoom transform to fit the selected rectangle.
          const plotW = w, plotH = h;
          const selW = x1 - x0, selH = y1 - y0;
          const k = Math.min(plotW / selW, plotH / selH);
          const tx = -x0 * k, ty = -y0 * k;
          select(chartWrapper).call(zoomBehavior.transform,
            zoomIdentity.translate(tx, ty).scale(k));
          // Clear the brush.
          select(this).call(brushBehavior.move, null);
        } else if (currentMode === 'select') {
          // Interval selection via WASM.
          try {
            const resultJson = renderer.handleDrag(panelIdx, x0, y0, x1, y1);
            adapter.onSelectionChange(JSON.parse(resultJson));
            // Re-render text with current zoom preserved.
            const t = zoomTransform(chartWrapper);
            const textJson = renderer.setTransform(t.k, t.x, t.y);
            _placeTextSvg(svgEl, JSON.parse(textJson));
            // Raw overlay already injected; only update data group transform.
            if (_rawOverlay) {
              _rawOverlay.dataG.setAttribute('transform', _zoomMatrix(t.k, t.x, t.y));
            }
          } catch (err) {
            console.warn('[ferrum] handleDrag error:', err);
          }
          // Clear the brush rectangle after selection is applied.
          select(this).call(brushBehavior.move, null);
        }
      });

      const brushG = select(svgEl).append('g')
        .attr('class', 'ferrum-brush')
        .attr('data-panel', panelIdx)
        .call(brushBehavior);

      // Style the brush rectangle.
      brushG.selectAll('.selection')
        .style('fill', brushFill)
        .style('stroke', brushStroke);
    }
  }

  // Apply initial mode — sets container.dataset.mode and SVG pointer-events.
  setMode(defaultMode);

  // ── Tooltip hover handler ─────────────────────────────────────────
  function handleHover(e) {
    const r = canvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (w / r.width);
    const my = (e.clientY - r.top) * (h / r.height);

    let tooltipData = null;
    // WASM-only hit-test via renderer.hitTestAt + getTooltip.
    if (renderer) {
      try {
        const hitJson = renderer.hitTestAt(mx, my);
        const hit = JSON.parse(hitJson);
        if (hit.panel != null && hit.batch != null && hit.idx != null) {
          const tJson = renderer.getTooltip(hit.panel, hit.batch, hit.idx);
          const parsed = JSON.parse(tJson);
          if (parsed.fields && parsed.fields.length > 0) tooltipData = parsed;
        }
      } catch (err) { /* WASM not ready or no tooltip data */ }
    }
    if (tooltipData) {
      tip.replaceChildren();
      const tbl = document.createElement('table');
      for (const f of tooltipData.fields) {
        const tr = document.createElement('tr');
        const k = document.createElement('td');
        k.textContent = f.name; k.style.fontWeight = 'bold'; k.style.paddingRight = '6px';
        const v = document.createElement('td'); v.textContent = f.value;
        tr.appendChild(k); tr.appendChild(v); tbl.appendChild(tr);
      }
      tip.appendChild(tbl);
      // Position tooltip in CSS coords (scene → element CSS).
      const cssMx = mx * (r.width / w);
      const csMy = my * (r.height / h);
      tip.style.left = (cssMx + 12) + 'px';
      tip.style.top = (csMy - 12) + 'px';
      tip.style.opacity = '1';
    } else {
      tip.style.opacity = '0';
    }
  }

  // ── RAF-coalesced mousemove ───────────────────────────────────────
  let _rafId = null, _pendingMove = null;
  // Listeners on chartWrapper (not canvas) because the SVG overlay with
  // pointer-events:all for D3 brush sits on top of the canvas and
  // intercepts events.  chartWrapper receives bubbled events from both.
  chartWrapper.addEventListener('mousemove', e => {
    _pendingMove = e;
    if (!_rafId) _rafId = requestAnimationFrame(() => {
      _rafId = null;
      if (_pendingMove) handleHover(_pendingMove);
    });
  });

  chartWrapper.addEventListener('mouseleave', () => {
    tip.style.opacity = '0';
  });

  // ── Click: href navigation + point selection (WASM-only) ──────────
  chartWrapper.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (w / r.width);
    const cy = (e.clientY - r.top) * (h / r.height);

    // Href navigation via WASM hit-test.
    if (renderer) {
      try {
        const hitJson = renderer.hitTestAt(cx, cy);
        const hit = JSON.parse(hitJson);
        if (hit.panel != null && hit.batch != null && hit.idx != null) {
          const href = renderer.getHref(hit.panel, hit.batch, hit.idx);
          if (href) {
            window.open(href, '_blank', 'noopener,noreferrer');
            return;
          }
        }
      } catch (err) { /* WASM not ready */ }
    }

    // Delegate clicks to WASM handleClick only when point selections exist.
    // Interval selections only respond to drags (handleDrag), not clicks.
    if (renderer && _hasPointSelections) {
      try {
        const stateJson = renderer.handleClick(cx, cy, e.shiftKey);
        const state = JSON.parse(stateJson);
        adapter.onSelectionChange(state);
      } catch (err) {
        console.warn('[ferrum] handleClick error:', err);
      }
    }
  });

  // ── ResizeObserver ────────────────────────────────────────────────
  let _ro = null;
  if (renderer) {
    _ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        const curDpr = window.devicePixelRatio || 1;
        const safeDpr = Math.min(curDpr, maxTex / Math.max(cr.width, cr.height, 1));
        const newW = Math.round(cr.width * Math.max(1, safeDpr));
        const newH = Math.round(cr.height * Math.max(1, safeDpr));
        if (newW > 0 && newH > 0 && (newW !== canvas.width || newH !== canvas.height)) {
          canvas.width = newW;
          canvas.height = newH;
          try { renderer.resize(newW, newH); renderer.renderFrame(); } catch (err) { /* ignore */ }
        }
      }
    });
    _ro.observe(canvas);
  }

  // ── Cleanup registration ─────────────────────────────────────────
  container._ferrumCleanup = () => {
    container.removeEventListener('keydown', _onKeydown);
    if (_ro) _ro.disconnect();
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    clearTimeout(_zoomDebounceId);
  };

  return { canvas, renderer, scene, svgEl };
}

// ── Standalone adapter factory (for HTML exports) ────────────────────────
export function createStandaloneAdapter(packedB64, interactionConfig) {
  let packedArr;
  if (packedB64) {
    const raw = atob(packedB64);
    packedArr = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) packedArr[i] = raw.charCodeAt(i);
  } else {
    packedArr = new Uint8Array(0);
  }
  return {
    getPackedData() { return packedArr; },
    getInteractionConfig() { return interactionConfig || '{}'; },
    onSelectionChange(_state) { /* local-only, no Python round-trip */ },
    onZoomChange(_state) { /* local-only, no Python round-trip */ },
  };
}

// ── Export _render as renderChart for standalone HTML templates ───────────
export { _render as renderChart };

// ── anywidget entry point ────────────────────────────────────────────────
export async function render({ model, el }) {
  // Jupyter adapter: bridges the anywidget model to the adapter interface.
  const adapter = {
    getPackedData() {
      const pd = model.get('packed_data');
      if (pd instanceof Uint8Array) return pd;
      if (pd && pd.buffer instanceof ArrayBuffer) return new Uint8Array(pd.buffer, pd.byteOffset, pd.byteLength);
      if (pd instanceof ArrayBuffer) return new Uint8Array(pd);
      return new Uint8Array(pd || []);
    },
    getInteractionConfig() { return model.get('interaction_config') || '{}'; },
    onSelectionChange(state) { model.set('selection_state', state); model.save_changes(); },
    onZoomChange(state) { model.set('zoom_state', JSON.stringify(state)); model.save_changes(); },
  };

  const container = document.createElement('div');
  el.appendChild(container);
  let _state = null;
  let _prevJson = null;
  let _transitionRafId = null;

  async function _reload(s) {
    try {
      const prev = _prevJson;
      _prevJson = s;

      // Cancel any in-flight transition animation before overwriting _state.
      if (_transitionRafId !== null) {
        cancelAnimationFrame(_transitionRafId);
        _transitionRafId = null;
      }

      _state = await _render(container, s, adapter);

      if (_state && prev && _state.renderer) {
        // Animate transition from previous scene to the current one.
        // B4 fix: pass `prev` (the OLD scene JSON), not `s` (the new scene).
        // loadScene(s) already loaded the new scene into the renderer, so
        // startTransition needs the old scene to interpolate FROM.
        try {
          // Capture renderer locally so the closure does not read stale _state
          // if _reload fires again while this loop is still running.
          const _renderer = _state.renderer;
          _state.renderer.startTransition(prev);
          const dur = 300;
          const t0 = performance.now();
          function _step() {
            const t = Math.min((performance.now() - t0) / dur, 1.0);
            try { _renderer.tickTransition(t); } catch (_) {}
            if (t < 1.0) {
              _transitionRafId = requestAnimationFrame(_step);
            } else {
              _transitionRafId = null;
            }
          }
          _transitionRafId = requestAnimationFrame(_step);
        } catch (e) { /* transition not supported — fall back to static render */ }
      }

      // Zoom, dblclick-reset, brush, and ResizeObserver are now wired
      // inside _render() so they work in both Jupyter and standalone
      // HTML modes.
    } catch (e) {
      console.error('[ferrum] widget reload failed:', e);
    }
  }

  const s = model.get('scene_json');
  if (s) await _reload(s);
  model.on('change:scene_json', async () => {
    const u = model.get('scene_json');
    if (u) await _reload(u);
  });
}
