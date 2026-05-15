// ── anywidget render entry point ──────────────────────────────────────────
// This file is read by _interactive.py which replaces __B64__ with the
// base64-encoded ferrum_wasm_bg.wasm blob before sending to the browser.
// Keep all JS here — never embed JS strings in Python.

const _B64 = '__B64__';
const _raw = atob(_B64);
const _bytes = new Uint8Array(_raw.length);
for (let i = 0; i < _raw.length; i++) _bytes[i] = _raw.charCodeAt(i);

let _ready = false, _initP = null;
async function _ensureWasm() {
  if (_ready) return;
  if (!_initP) _initP = __wbg_init(_bytes).then(() => { _ready = true; });
  await _initP;
}

function _placeText(overlay, texts) {
  overlay.replaceChildren();
  for (const t of texts) {
    const d = document.createElement('div');
    d.className = 'ferrum-text';
    d.style.cssText = `position:absolute;left:${t.x}px;top:${t.y}px;` +
      `font-size:${t.fontSize}px;font-weight:${t.fontWeight};` +
      `font-family:${t.fontFamily};color:${t.color};` +
      `white-space:nowrap;pointer-events:none;line-height:1`;
    if (t.anchor === 'center') d.style.transform = 'translateX(-50%)';
    else if (t.anchor === 'end') d.style.transform = 'translateX(-100%)';
    d.textContent = t.content;
    overlay.appendChild(d);
  }
}

// Hit-test pixel (x, y) against the mark batches.
// marks is an array of {batch, panel} pairs so arc paths can use panel.plot_area.
function _hitTest(marks, x, y) {
  for (let bi = marks.length - 1; bi >= 0; bi--) {
    const { batch: b, panel } = marks[bi];
    if (!b.nodes) continue;
    for (let ni = b.nodes.length - 1; ni >= 0; ni--) {
      const n = b.nodes[ni];
      let hit = false;
      if (n.type === 'circle') {
        const dx = x - n.cx, dy = y - n.cy;
        hit = dx * dx + dy * dy <= n.r * n.r;
      } else if (n.type === 'rect') {
        hit = x >= n.x && x <= n.x + n.w && y >= n.y && y <= n.y + n.h;
      } else if (n.type === 'path' && b.kind === 'arc') {
        // Pie / donut wedge hit test from plot_area center + path commands.
        const pa = panel.plot_area;
        const cx = pa.x + pa.w / 2, cy = pa.y + pa.h / 2;
        const dx = x - cx, dy = y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const arcCmd = n.commands && n.commands.find(c => c.op === 'arc_to');
        const outerR = arcCmd ? arcCmd.rx : 0;
        if (dist <= outerR) {
          const lineTo = n.commands && n.commands.find(c => c.op === 'line_to');
          const innerR = lineTo
            ? Math.sqrt((lineTo.x - cx) ** 2 + (lineTo.y - cy) ** 2)
            : 0;
          if (dist >= innerR) {
            const moveTo = n.commands && n.commands.find(c => c.op === 'move_to');
            if (moveTo) {
              const norm = a => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
              const pointAngle = Math.atan2(dx, -dy);
              const startAngle = Math.atan2(moveTo.x - cx, -(moveTo.y - cy));
              const endAngle = arcCmd
                ? Math.atan2(arcCmd.x - cx, -(arcCmd.y - cy))
                : startAngle;
              const sa = norm(startAngle);
              let ea = norm(endAngle);
              if (ea <= sa) ea += 2 * Math.PI;
              const pa2 = norm(pointAngle);
              const pa3 = pa2 < sa ? pa2 + 2 * Math.PI : pa2;
              hit = pa3 >= sa && pa3 <= ea;
            } else {
              hit = true; // no move_to — treat as full circle
            }
          }
        }
      }
      if (hit) return { batch: b, idx: ni };
    }
  }
  return null;
}

async function _render(container, sceneJson, model) {
  container.replaceChildren();
  container.style.position = 'relative';

  const scene = JSON.parse(sceneJson);
  const w = scene.width || 640, h = scene.height || 480;

  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h; canvas.style.display = 'block';
  container.appendChild(canvas);

  const ov = document.createElement('div');
  ov.className = 'ferrum-overlay';
  Object.assign(ov.style, { position: 'absolute', top: '0', left: '0',
    width: w + 'px', height: h + 'px', pointerEvents: 'none' });
  container.appendChild(ov);

  const tip = document.createElement('div');
  tip.className = 'ferrum-tooltip';
  Object.assign(tip.style, { position: 'absolute', pointerEvents: 'none',
    opacity: '0', transition: 'opacity 0.1s ease' });
  container.appendChild(tip);

  // marks carries {batch, panel} pairs so hit-testers have panel context.
  const marks = scene.panels
    ? scene.panels.flatMap(p => (p.marks || []).map(b => ({ batch: b, panel: p })))
    : [];

  // ── Hover tooltip ────────────────────────────────────────────────────
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    const h = _hitTest(marks, e.clientX - r.left, e.clientY - r.top);
    if (h && h.batch.tooltips && h.batch.tooltips[h.idx]) {
      const t = h.batch.tooltips[h.idx];
      tip.replaceChildren();
      const tbl = document.createElement('table');
      for (const f of t.fields) {
        const tr = document.createElement('tr');
        const k = document.createElement('td');
        k.textContent = f.name; k.style.fontWeight = 'bold'; k.style.paddingRight = '6px';
        const v = document.createElement('td'); v.textContent = f.value;
        tr.appendChild(k); tr.appendChild(v); tbl.appendChild(tr);
      }
      tip.appendChild(tbl);
      tip.style.left = (e.clientX - r.left + 12) + 'px';
      tip.style.top = (e.clientY - r.top - 12) + 'px';
      tip.style.opacity = '1';
    } else {
      tip.style.opacity = '0';
    }
  });
  canvas.addEventListener('mouseleave', () => { tip.style.opacity = '0'; });

  // ── Click: href navigation + selection ──────────────────────────────
  // renderer is declared below (let), captured by closure — sees the updated
  // value after GPU init completes (JS let closures capture the binding).
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    const h = _hitTest(marks, cx, cy);
    if (!h) return;

    if (h.batch.hrefs && h.batch.hrefs[h.idx]) {
      window.open(h.batch.hrefs[h.idx], '_blank', 'noopener,noreferrer');
      return;
    }

    // Delegate to WASM handleClick when GPU renderer is available.
    // handleClick() does authoritative hit-test, updates selection state,
    // dims non-selected marks via conditional encodings, re-renders frame,
    // and returns the new selection JSON for Python sync.
    if (renderer) {
      try {
        const stateJson = renderer.handleClick(cx, cy);
        const state = JSON.parse(stateJson);
        model.set('selection_state', state);
        model.save_changes();
      } catch (err) {
        console.warn('[ferrum] handleClick error:', err);
      }
      return;
    }

    // Fallback (no GPU): extract field values from tooltip, build selection
    // state from registered selection specs in interaction_config.
    const cfg = model.get('interaction_config');
    let selConfig = {};
    try { selConfig = JSON.parse(cfg || '{}'); } catch (e) { /* ignore */ }
    const selections = selConfig.selections || [];
    const tooltip = h.batch.tooltips && h.batch.tooltips[h.idx];
    const fieldMap = {};
    if (tooltip) { for (const f of tooltip.fields) fieldMap[f.name] = f.value; }
    const selState = {};
    for (const sel of selections) {
      if (!sel.fields) continue;
      const vals = {};
      for (const field of sel.fields) {
        if (fieldMap[field] !== undefined) vals[field] = fieldMap[field];
      }
      if (Object.keys(vals).length > 0) selState[sel.name] = vals;
    }
    if (Object.keys(selState).length > 0) {
      model.set('selection_state', selState);
      model.save_changes();
    }
  });

  // ── GPU init (may fail when WebGPU/WebGL context limit exceeded) ────
  // Event listeners above still work without GPU — tooltips + click state.
  let renderer = null;
  try {
    await _ensureWasm();
    renderer = await WasmRenderer.create(canvas);
    const textJson = renderer.loadScene(sceneJson);
    _placeText(ov, JSON.parse(textJson));
  } catch (e) {
    console.warn('[ferrum] GPU init failed — rendering disabled, tooltips still active.', e);
  }

  return { canvas, renderer, scene };
}

export async function render({ model, el }) {
  const container = document.createElement('div');
  el.appendChild(container);
  let _state = null;
  let _prevJson = null;

  async function _reload(s) {
    try {
      const prev = _prevJson;
      _prevJson = s;
      _state = await _render(container, s, model);

      if (_state && prev && _state.renderer) {
        // Animate transition from previous scene.
        try {
          _state.renderer.startTransition(s);
          const dur = 300;
          const t0 = performance.now();
          function _step() {
            const t = Math.min((performance.now() - t0) / dur, 1.0);
            _state.renderer.tickTransition(t).catch(() => {});
            if (t < 1.0) requestAnimationFrame(_step);
          }
          requestAnimationFrame(_step);
        } catch (e) { /* transition not supported — fall back to static render */ }
      }

      if (_state) {
        // ── Scroll zoom (Python round-trip) ───────────────────────────
        // TODO Phase 12: replace with WASM-side GPU affine transform
        //   via renderer.onWheel() — eliminates the ~200ms round-trip.
        _state.canvas.addEventListener('wheel', e => {
          e.preventDefault();
          if (!_state) return;
          const sc = _state.scene;
          const p = sc.panels && sc.panels[0];
          if (!p) return;
          const factor = 1 - e.deltaY * 0.001;
          const xs = p.coord && p.coord.x_domain;
          const ys = p.coord && p.coord.y_domain;
          if (!xs || !ys) return;
          const xSpan = xs[1] - xs[0], ySpan = ys[1] - ys[0];
          const xc = xs[0] + xSpan / 2, yc = ys[0] + ySpan / 2;
          const nxSpan = xSpan / factor, nySpan = ySpan / factor;
          const zs = JSON.stringify({ '0': {
            x_domain: [xc - nxSpan / 2, xc + nxSpan / 2],
            y_domain: [yc - nySpan / 2, yc + nySpan / 2],
          }});
          model.set('zoom_state', zs);
          model.save_changes();
        }, { passive: false });
      }
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
