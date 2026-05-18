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
    // Build composite transform: anchor (translateX) + rotation + baseline (translateY)
    const parts = [];
    if (t.anchor === 'center') parts.push('translateX(-50%)');
    else if (t.anchor === 'end') parts.push('translateX(-100%)');
    if (t.angle) parts.push(`rotate(${t.angle}deg)`);
    if (t.baseline === 'middle') parts.push('translateY(-50%)');
    else if (t.baseline === 'bottom') parts.push('translateY(-100%)');
    else if (t.baseline === 'alphabetic') parts.push('translateY(-85%)');
    if (parts.length > 0) d.style.transform = parts.join(' ');
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

// ── CSS-coordinate scaling helper ────────────────────────────────────────
// Converts a mouse event to canvas-space coordinates, accounting for any
// CSS scaling between the canvas element's intrinsic size and its layout size.
function _canvasCoords(canvas, e) {
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width;
  const sy = canvas.height / r.height;
  return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
}

// ── Adapter interface (duck-typed) ───────────────────────────────────────
// {
//   getPackedData()           → Uint8Array
//   getInteractionConfig()    → string (JSON)
//   onSelectionChange(state)  → void (called when selection changes)
//   onZoomChange(state)       → void (called when zoom changes)
// }

async function _render(container, sceneJson, adapter) {
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

  // ── Brush / interval selection detection ──────────────────────────
  const cfg = JSON.parse(adapter.getInteractionConfig());
  const _hasPointSelections = (cfg.selections || []).some(s => s.type === 'point');
  const hasInterval = (cfg.selections || []).some(s => s.type === 'interval');

  // Extract brush styling from the interval selection's SelectionMark, if present.
  let _brushBg = 'rgba(51, 136, 204, 0.2)';
  let _brushBorder = '1px solid rgba(51, 136, 204, 0.6)';
  if (hasInterval) {
    const intervalSel = (cfg.selections || []).find(s => s.type === 'interval');
    if (intervalSel && intervalSel.mark) {
      const m = intervalSel.mark;
      if (m.fill) _brushBg = m.fill;
      if (m.stroke) _brushBorder = `1px solid ${m.stroke}`;
    }
  }

  // ── Pan + hover tooltip (combined mousemove) ─────────────────────
  // renderer is declared below (let), captured by closure.
  //
  // _zoom mirrors the Rust ZoomPanState affine transform so the JS-side
  // hit-test can inverse-transform mouse positions to original mark space.
  // Updated in-place by pan events here, and by wheel/dblclick handlers
  // in the outer render() scope via the returned _state.zoom reference.
  const _zoom = { sx: 1, sy: 1, tx: 0, ty: 0 };

  // Inverse-transform a canvas (post-zoom) point to original mark space.
  function _invZoom(x, y) {
    return [
      Math.abs(_zoom.sx) > 1e-10 ? (x - _zoom.tx) / _zoom.sx : x,
      Math.abs(_zoom.sy) > 1e-10 ? (y - _zoom.ty) / _zoom.sy : y,
    ];
  }

  let _isDragging = false;
  let _panStart = null;
  let _isBrushing = false;
  let _brushDiv = null;
  let _brushOrigin = null; // {x, y} in canvas coords at mousedown

  canvas.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    const [mx, my] = _canvasCoords(canvas, e);

    if (hasInterval && !e.altKey) {
      // Start brush mode.
      _isBrushing = true;
      _brushOrigin = { x: mx, y: my };
      _brushDiv = document.createElement('div');
      _brushDiv.className = 'ferrum-brush';
      // Position the brush in CSS coords (relative to container).
      const r = canvas.getBoundingClientRect();
      const cssX = mx / (canvas.width / r.width);
      const cssY = my / (canvas.height / r.height);
      Object.assign(_brushDiv.style, {
        position: 'absolute',
        left: cssX + 'px',
        top: cssY + 'px',
        width: '0px',
        height: '0px',
        background: _brushBg,
        border: _brushBorder,
        pointerEvents: 'none',
        zIndex: '10',
      });
      container.appendChild(_brushDiv);
    } else {
      // Pan mode.
      _panStart = { x: mx, y: my };
      _isDragging = false;
    }
  });

  canvas.addEventListener('mousemove', e => {
    const [mx, my] = _canvasCoords(canvas, e);

    // ── Brush resize ──────────────────────────────────────────────
    if (_isBrushing && _brushDiv && _brushOrigin) {
      const r = canvas.getBoundingClientRect();
      const scaleX = canvas.width / r.width;
      const scaleY = canvas.height / r.height;
      const ox = _brushOrigin.x / scaleX;
      const oy = _brushOrigin.y / scaleY;
      const cx = mx / scaleX;
      const cy = my / scaleY;
      _brushDiv.style.left = Math.min(ox, cx) + 'px';
      _brushDiv.style.top = Math.min(oy, cy) + 'px';
      _brushDiv.style.width = Math.abs(cx - ox) + 'px';
      _brushDiv.style.height = Math.abs(cy - oy) + 'px';
      return; // don't process tooltip or pan while brushing
    }

    // Tooltip hover: inverse-transform mouse into original mark space before hit-testing.
    const [hx, hy] = _invZoom(mx, my);
    let tooltipData = null;
    // Try JS hit-test first (non-packed batches with nodes).
    const hh = _hitTest(marks, hx, hy);
    if (hh && hh.batch.tooltips && hh.batch.tooltips[hh.idx]) {
      tooltipData = hh.batch.tooltips[hh.idx];
    }
    // Fallback: WASM hit-test + getTooltip for packed batches (empty nodes).
    if (!tooltipData && renderer) {
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
      // Position tooltip in CSS coords.
      const r = canvas.getBoundingClientRect();
      const cssMx = mx / (canvas.width / r.width);
      const csMy = my / (canvas.height / r.height);
      tip.style.left = (cssMx + 12) + 'px';
      tip.style.top = (csMy - 12) + 'px';
      tip.style.opacity = '1';
    } else {
      tip.style.opacity = '0';
    }
    // Pan drag.
    if (!_panStart) return;
    const dx = mx - _panStart.x, dy = my - _panStart.y;
    if (!_isDragging && (Math.abs(dx) > 3 || Math.abs(dy) > 3)) _isDragging = true;
    if (!_isDragging || !renderer) return;
    try {
      const textJson = renderer.onPan(0, dx, dy);
      _placeText(ov, JSON.parse(textJson));
      _zoom.tx += dx;
      _zoom.ty += dy;
    } catch (err) { /* GPU not ready */ }
    _panStart = { x: mx, y: my };
  });

  const _endPan = () => { _panStart = null; setTimeout(() => { _isDragging = false; }, 0); };

  canvas.addEventListener('mouseup', e => {
    // ── End brush ─────────────────────────────────────────────────
    if (_isBrushing && _brushOrigin) {
      const [mx, my] = _canvasCoords(canvas, e);
      const dx = Math.abs(mx - _brushOrigin.x);
      const dy = Math.abs(my - _brushOrigin.y);
      // Only fire handleDrag for real brush gestures (>4px), not clicks.
      if (renderer && dx > 4 && dy > 4) {
        try {
          const x0 = Math.min(_brushOrigin.x, mx);
          const y0 = Math.min(_brushOrigin.y, my);
          const x1 = Math.max(_brushOrigin.x, mx);
          const y1 = Math.max(_brushOrigin.y, my);
          const resultJson = renderer.handleDrag(0, x0, y0, x1, y1);
          adapter.onSelectionChange(JSON.parse(resultJson));
        } catch (err) {
          console.warn('[ferrum] handleDrag error:', err);
        }
      }
      // Remove brush div.
      if (_brushDiv && _brushDiv.parentNode) {
        _brushDiv.parentNode.removeChild(_brushDiv);
      }
      _isBrushing = false;
      _brushDiv = null;
      _brushOrigin = null;
      return;
    }
    _endPan();
  });

  canvas.addEventListener('mouseleave', () => {
    tip.style.opacity = '0';
    // Clean up brush if mouse leaves canvas.
    if (_isBrushing) {
      if (_brushDiv && _brushDiv.parentNode) {
        _brushDiv.parentNode.removeChild(_brushDiv);
      }
      _isBrushing = false;
      _brushDiv = null;
      _brushOrigin = null;
    }
    _endPan();
  });

  // ── Click: href navigation + selection ──────────────────────────────
  canvas.addEventListener('click', e => {
    if (_isDragging) return; // suppress click fired immediately after a pan drag
    const [cx, cy] = _canvasCoords(canvas, e);

    // Href navigation: inverse-transform then JS hit test.
    const [hx, hy] = _invZoom(cx, cy);
    const h = _hitTest(marks, hx, hy);
    if (h && h.batch.hrefs && h.batch.hrefs[h.idx]) {
      window.open(h.batch.hrefs[h.idx], '_blank', 'noopener,noreferrer');
      return;
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
      return;
    }

    // Fallback (no GPU): use JS hit test + tooltip field extraction.
    if (!h) return;
    const icfg = adapter.getInteractionConfig();
    let selConfig = {};
    try { selConfig = JSON.parse(icfg || '{}'); } catch (_e) { /* ignore */ }
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
      adapter.onSelectionChange(selState);
    }
  });

  // ── GPU init (may fail when WebGPU/WebGL context limit exceeded) ────
  // Event listeners above still work without GPU — tooltips + click state.
  let renderer = null;
  try {
    await _ensureWasm();
    renderer = await WasmRenderer.create(canvas);
    const packedArr = adapter.getPackedData();
    const textJson = renderer.loadScene(sceneJson, packedArr);
    _placeText(ov, JSON.parse(textJson));
  } catch (e) {
    console.warn('[ferrum] GPU init failed — rendering disabled, tooltips still active.', e);
  }

  // ── Scroll zoom — GPU affine transform via WASM onWheel ──────────
  let _zoomDebounceId = null;
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (!renderer) return;
    const [cx, cy] = _canvasCoords(canvas, e);
    try {
      const textJson = renderer.onWheel(0, e.deltaY, cx, cy);
      _placeText(ov, JSON.parse(textJson));
      const t = _zoom;
      const f = 1.0 + e.deltaY * 0.001;
      const ZMIN = 0.1, ZMAX = 50.0;
      const sx = Math.min(ZMAX, Math.max(ZMIN, t.sx * f));
      const sy = Math.min(ZMAX, Math.max(ZMIN, t.sy * f));
      t.tx = cx - sx * ((cx - t.tx) / t.sx);
      t.ty = cy - sy * ((cy - t.ty) / t.sy);
      t.sx = sx;
      t.sy = sy;
    } catch (err) { /* GPU not ready */ }
    clearTimeout(_zoomDebounceId);
    _zoomDebounceId = setTimeout(() => {
      const p = scene.panels && scene.panels[0];
      if (!p || !p.coord) return;
      const xs = p.coord.x_domain, ys = p.coord.y_domain;
      if (!xs || !ys) return;
      adapter.onZoomChange({ '0': { x_domain: xs, y_domain: ys }});
    }, 400);
  }, { passive: false });

  // ── Double-click: reset zoom to identity ──────────────────────────
  canvas.addEventListener('dblclick', () => {
    if (!renderer) return;
    try {
      const textJson = renderer.resetZoom(0);
      _placeText(ov, JSON.parse(textJson));
      Object.assign(_zoom, { sx: 1, sy: 1, tx: 0, ty: 0 });
    } catch (err) { /* ignore */ }
  });

  // ── ResizeObserver ────────────────────────────────────────────────
  if (renderer) {
    const ro = new ResizeObserver(() => {
      try { renderer.resize(canvas.width, canvas.height); } catch (err) { /* ignore */ }
    });
    ro.observe(canvas);
  }

  return { canvas, renderer, scene, ov, zoom: _zoom };
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

  async function _reload(s) {
    try {
      const prev = _prevJson;
      _prevJson = s;
      _state = await _render(container, s, adapter);

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

      // Zoom, dblclick-reset, and ResizeObserver are now wired inside _render()
      // so they work in both Jupyter and standalone HTML modes.
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
