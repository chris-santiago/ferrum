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

// D3 interactions (brush, zoom, select, zoomTransform, pointer) are provided
// by d3-interactions.js which is inlined before this file in both standalone
// HTML and Jupyter ESM builds.  The D3 bundle's `export { ... }` is stripped
// by the assembler, leaving the symbols in module scope.

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

  // ── Canvas ───────────────────────────────────────────────────────
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h; canvas.style.display = 'block';
  container.appendChild(canvas);

  // ── SVG overlay for text labels ──────────────────────────────────
  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgEl.setAttribute('width', w);
  svgEl.setAttribute('height', h);
  // SVG inherits CSS @font-face from the parent HTML document (Inter).
  svgEl.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
  container.appendChild(svgEl);

  // ── Tooltip ──────────────────────────────────────────────────────
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

  // ── GPU init (may fail when WebGPU/WebGL context limit exceeded) ──
  // Event listeners below still work without GPU — tooltips + click state.
  let renderer = null;
  try {
    await _ensureWasm();
    renderer = await WasmRenderer.create(canvas);
    const packedArr = adapter.getPackedData();
    const textJson = renderer.loadScene(sceneJson, packedArr);
    _placeTextSvg(svgEl, JSON.parse(textJson));
  } catch (e) {
    console.warn('[ferrum] GPU init failed — rendering disabled, tooltips still active.', e);
  }

  // ── D3-zoom on canvas ─────────────────────────────────────────────
  let _zoomDebounceId = null;
  const zoomBehavior = zoom()
    .scaleExtent([0.1, 50])
    .filter(event => {
      // Always allow wheel-zoom.
      if (event.type === 'wheel') return true;
      // When interval selections are active, require Alt/Option or Cmd/Meta
      // for pan (drag without modifier belongs to the brush).
      if (hasInterval && !event.altKey && !event.metaKey) return false;
      // Only left-button drags.
      return !event.button;
    })
    .on('zoom', event => {
      if (!renderer) return;
      const { k, x, y } = event.transform;
      try {
        const textJson = renderer.setTransform(k, x, y);
        _placeTextSvg(svgEl, JSON.parse(textJson));
      } catch (err) { /* GPU not ready */ }
      // Debounced adapter callback for Jupyter zoom rebuild.
      clearTimeout(_zoomDebounceId);
      _zoomDebounceId = setTimeout(() => {
        adapter.onZoomChange({ '0': { k, x, y } });
      }, 400);
    });

  // Attach zoom to the container (wraps both canvas and SVG) so wheel/pan
  // events work regardless of which layer captures them.
  select(container).call(zoomBehavior);

  // Double-click: reset zoom to identity.
  select(container).on('dblclick.zoom', () => {
    if (!renderer) return;
    select(container).call(zoomBehavior.transform, zoomIdentity);
  });

  // ── D3-brush on SVG (per-panel overlays for interval selections) ────
  if (hasInterval && scene.panels) {
    // Extract brush styling from the interval selection's SelectionMark.
    let brushFill = 'rgba(51, 136, 204, 0.2)';
    let brushStroke = 'rgba(51, 136, 204, 0.6)';
    const intervalSel = (cfg.selections || []).find(s => s.type === 'interval');
    if (intervalSel && intervalSel.mark) {
      if (intervalSel.mark.fill) brushFill = intervalSel.mark.fill;
      if (intervalSel.mark.stroke) brushStroke = intervalSel.mark.stroke;
    }

    // Enable pointer events on the SVG so brushes can capture gestures.
    svgEl.style.pointerEvents = 'all';

    for (let pi = 0; pi < scene.panels.length; pi++) {
      const pa = scene.panels[pi].plot_area;
      if (!pa) continue;

      const brushBehavior = brush()
        .extent([[pa.x, pa.y], [pa.x + pa.w, pa.y + pa.h]])
        .filter(event => !event.altKey && !event.metaKey && event.button === 0);

      // Capture panel index for the closure.
      const panelIdx = pi;
      brushBehavior.on('end', function(event) {
        if (!renderer) return;
        if (!event.selection) return;
        const [[x0, y0], [x1, y1]] = event.selection;
        try {
          const resultJson = renderer.handleDrag(panelIdx, x0, y0, x1, y1);
          adapter.onSelectionChange(JSON.parse(resultJson));
          // Re-render text with current zoom preserved.
          const t = zoomTransform(container);
          const textJson = renderer.setTransform(t.k, t.x, t.y);
          _placeTextSvg(svgEl, JSON.parse(textJson));
        } catch (err) {
          console.warn('[ferrum] handleDrag error:', err);
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

  // ── Tooltip mousemove ─────────────────────────────────────────────
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (canvas.width / r.width);
    const my = (e.clientY - r.top) * (canvas.height / r.height);

    // Inverse-zoom for hit-test in original mark space.
    const t = zoomTransform(container);
    const hx = t.k !== 0 ? (mx - t.x) / t.k : mx;
    const hy = t.k !== 0 ? (my - t.y) / t.k : my;

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
      const cssMx = mx / (canvas.width / r.width);
      const csMy = my / (canvas.height / r.height);
      tip.style.left = (cssMx + 12) + 'px';
      tip.style.top = (csMy - 12) + 'px';
      tip.style.opacity = '1';
    } else {
      tip.style.opacity = '0';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    tip.style.opacity = '0';
  });

  // ── Click: href navigation + point selection ──────────────────────
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (canvas.width / r.width);
    const cy = (e.clientY - r.top) * (canvas.height / r.height);

    // Inverse-zoom for JS hit-test.
    const t = zoomTransform(container);
    const hx = t.k !== 0 ? (cx - t.x) / t.k : cx;
    const hy = t.k !== 0 ? (cy - t.y) / t.k : cy;

    // Href navigation.
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

  // ── ResizeObserver ────────────────────────────────────────────────
  if (renderer) {
    const ro = new ResizeObserver(() => {
      try { renderer.resize(canvas.width, canvas.height); } catch (err) { /* ignore */ }
    });
    ro.observe(canvas);
  }

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

  async function _reload(s) {
    try {
      const prev = _prevJson;
      _prevJson = s;
      _state = await _render(container, s, adapter);

      if (_state && prev && _state.renderer) {
        // Animate transition from previous scene to the current one.
        // B4 fix: pass `prev` (the OLD scene JSON), not `s` (the new scene).
        // loadScene(s) already loaded the new scene into the renderer, so
        // startTransition needs the old scene to interpolate FROM.
        try {
          _state.renderer.startTransition(prev);
          const dur = 300;
          const t0 = performance.now();
          function _step() {
            const t = Math.min((performance.now() - t0) / dur, 1.0);
            try { _state.renderer.tickTransition(t); } catch (_) {}
            if (t < 1.0) requestAnimationFrame(_step);
          }
          requestAnimationFrame(_step);
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
