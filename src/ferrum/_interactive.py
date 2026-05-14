"""InteractiveChart — anywidget-based Jupyter integration (Phase 11c).

Provides ``InteractiveChart``, an ``anywidget.AnyWidget`` subclass that
renders a ferrum chart using the WASM GPU renderer with bidirectional
selection state sync.

``Chart.interactive()`` returns an ``InteractiveChart`` when anywidget
is installed; otherwise it falls back to returning a clone (SVG path).
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ferrum.chart import Chart

_log = logging.getLogger(__name__)

_WASM_DIR = pathlib.Path(__file__).parent / "_wasm"


def _build_anywidget_esm() -> str:
    """Build a self-contained anywidget ESM with inlined WASM.

    ``ferrum-interactive.js`` relies on ``import('./ferrum_wasm.js')`` which
    fails in the anywidget/Jupyter context (no sibling-file serving).  This
    function generates a standalone module that initialises the WASM from an
    inline base64 blob — same strategy used by the standalone HTML export.
    """
    import base64

    from ferrum._html import _read_wasm_artifact

    js_glue = _read_wasm_artifact("ferrum_wasm.js").decode("utf-8")
    wasm_b64 = base64.b64encode(_read_wasm_artifact("ferrum_wasm_bg.wasm")).decode("ascii")

    return js_glue + "\n\n" + (
        "// ── anywidget render entry point ──────────────────────────────────\n"
        "const _B64='__B64__';\n"
        "const _raw=atob(_B64);\n"
        "const _bytes=new Uint8Array(_raw.length);\n"
        "for(let i=0;i<_raw.length;i++) _bytes[i]=_raw.charCodeAt(i);\n"
        "\n"
        "let _ready=false, _initP=null;\n"
        "async function _ensureWasm(){\n"
        "  if(_ready)return;\n"
        "  if(!_initP)_initP=__wbg_init(_bytes).then(()=>{_ready=true;});\n"
        "  await _initP;\n"
        "}\n"
        "\n"
        "function _placeText(overlay,texts){\n"
        "  overlay.replaceChildren();\n"
        "  for(const t of texts){\n"
        "    const d=document.createElement('div');\n"
        "    d.className='ferrum-text';\n"
        "    d.style.cssText=`position:absolute;left:${t.x}px;top:${t.y}px;"
        "font-size:${t.fontSize}px;font-weight:${t.fontWeight};"
        "font-family:${t.fontFamily};color:${t.color};"
        "white-space:nowrap;pointer-events:none;line-height:1`;\n"
        "    if(t.anchor==='center')d.style.transform='translateX(-50%)';\n"
        "    else if(t.anchor==='end')d.style.transform='translateX(-100%)';\n"
        "    d.textContent=t.content;\n"
        "    overlay.appendChild(d);\n"
        "  }\n"
        "}\n"
        "\n"
        "function _hitTest(marks,x,y){\n"
        "  for(let bi=marks.length-1;bi>=0;bi--){\n"
        "    const b=marks[bi]; if(!b.nodes)continue;\n"
        "    for(let ni=b.nodes.length-1;ni>=0;ni--){\n"
        "      const n=b.nodes[ni]; let hit=false;\n"
        "      if(n.type==='circle'){const dx=x-n.cx,dy=y-n.cy;hit=dx*dx+dy*dy<=n.r*n.r;}\n"
        "      else if(n.type==='rect'){hit=x>=n.x&&x<=n.x+n.w&&y>=n.y&&y<=n.y+n.h;}\n"
        "      if(hit)return{batch:b,idx:ni};\n"
        "    }\n"
        "  }\n"
        "  return null;\n"
        "}\n"
        "\n"
        "async function _render(container,sceneJson){\n"
        "  await _ensureWasm();\n"
        "  container.replaceChildren();\n"
        "  container.style.position='relative';\n"
        "  const scene=JSON.parse(sceneJson);\n"
        "  const w=scene.width||640, h=scene.height||480;\n"
        "  const canvas=document.createElement('canvas');\n"
        "  canvas.width=w; canvas.height=h; canvas.style.display='block';\n"
        "  container.appendChild(canvas);\n"
        "  const ov=document.createElement('div');\n"
        "  ov.className='ferrum-overlay';\n"
        "  Object.assign(ov.style,{position:'absolute',top:'0',left:'0',\n"
        "    width:w+'px',height:h+'px',pointerEvents:'none'});\n"
        "  container.appendChild(ov);\n"
        "  const tip=document.createElement('div');\n"
        "  tip.className='ferrum-tooltip';\n"
        "  Object.assign(tip.style,{position:'absolute',pointerEvents:'none',\n"
        "    opacity:'0',transition:'opacity 0.1s ease'});\n"
        "  container.appendChild(tip);\n"
        "  const renderer=await WasmRenderer.create(canvas);\n"
        "  const textJson=renderer.loadScene(sceneJson);\n"
        "  _placeText(ov,JSON.parse(textJson));\n"
        "  const marks=scene.panels?scene.panels.flatMap(p=>p.marks||[]):[];\n"
        "  canvas.style.pointerEvents='auto';\n"
        "  canvas.addEventListener('mousemove',e=>{\n"
        "    const r=canvas.getBoundingClientRect();\n"
        "    const h=_hitTest(marks,e.clientX-r.left,e.clientY-r.top);\n"
        "    if(h&&h.batch.tooltips&&h.batch.tooltips[h.idx]){\n"
        "      const t=h.batch.tooltips[h.idx];\n"
        "      tip.replaceChildren();\n"
        "      const tbl=document.createElement('table');\n"
        "      for(const f of t.fields){\n"
        "        const tr=document.createElement('tr');\n"
        "        const k=document.createElement('td');\n"
        "        k.textContent=f.name;k.style.fontWeight='bold';k.style.paddingRight='6px';\n"
        "        const v=document.createElement('td');v.textContent=f.value;\n"
        "        tr.appendChild(k);tr.appendChild(v);tbl.appendChild(tr);\n"
        "      }\n"
        "      tip.appendChild(tbl);\n"
        "      tip.style.left=(e.clientX-r.left+12)+'px';\n"
        "      tip.style.top=(e.clientY-r.top-12)+'px';\n"
        "      tip.style.opacity='1';\n"
        "    } else { tip.style.opacity='0'; }\n"
        "  });\n"
        "  canvas.addEventListener('mouseleave',()=>{tip.style.opacity='0';});\n"
        "  canvas.addEventListener('click',e=>{\n"
        "    const r=canvas.getBoundingClientRect();\n"
        "    const h=_hitTest(marks,e.clientX-r.left,e.clientY-r.top);\n"
        "    if(h&&h.batch.hrefs&&h.batch.hrefs[h.idx])\n"
        "      window.open(h.batch.hrefs[h.idx],'_blank','noopener,noreferrer');\n"
        "  });\n"
        "  return {canvas,renderer,scene};\n"
        "}\n"
        "\n"
        "export async function render({model,el}){\n"
        "  const container=document.createElement('div');\n"
        "  el.appendChild(container);\n"
        "  let _state=null;\n"
        "  let _prevJson=null;\n"
        "  async function _reload(s){\n"
        "    const prev=_prevJson;\n"
        "    _prevJson=s;\n"
        "    _state=await _render(container,s);\n"
        "    if(_state&&prev){\n"
        "      // Animate transition from previous scene\n"
        "      try{\n"
        "        _state.renderer.startTransition(prev,s);\n"
        "        const dur=300;\n"
        "        const t0=performance.now();\n"
        "        function _step(){\n"
        "          const t=Math.min((performance.now()-t0)/dur,1.0);\n"
        "          _state.renderer.tickTransition(t).catch(()=>{});\n"
        "          if(t<1.0)requestAnimationFrame(_step);\n"
        "        }\n"
        "        requestAnimationFrame(_step);\n"
        "      }catch(e){/* transition not supported — fall back to static render */}\n"
        "    }\n"
        "    if(_state){\n"
        "      _state.canvas.addEventListener('wheel',e=>{\n"
        "        e.preventDefault();\n"
        "        if(!_state)return;\n"
        "        const sc=_state.scene;\n"
        "        const p=sc.panels&&sc.panels[0];\n"
        "        if(!p)return;\n"
        "        const factor=1-e.deltaY*0.001;\n"
        "        const xs=p.coord&&p.coord.x_domain;\n"
        "        const ys=p.coord&&p.coord.y_domain;\n"
        "        if(!xs||!ys)return;\n"
        "        const xSpan=(xs[1]-xs[0]);\n"
        "        const ySpan=(ys[1]-ys[0]);\n"
        "        const xc=xs[0]+xSpan/2, yc=ys[0]+ySpan/2;\n"
        "        const nxSpan=xSpan/factor, nySpan=ySpan/factor;\n"
        "        const zs=JSON.stringify({'0':{\n"
        "          x_domain:[xc-nxSpan/2,xc+nxSpan/2],\n"
        "          y_domain:[yc-nySpan/2,yc+nySpan/2]\n"
        "        }});\n"
        "        model.set('zoom_state',zs);\n"
        "        model.save_changes();\n"
        "      },{passive:false});\n"
        "    }\n"
        "  }\n"
        "  const s=model.get('scene_json');\n"
        "  if(s) await _reload(s);\n"
        "  model.on('change:scene_json',async()=>{\n"
        "    const u=model.get('scene_json');\n"
        "    if(u) await _reload(u);\n"
        "  });\n"
        "}\n"
    ).replace("__B64__", wasm_b64)


class InteractiveRenderError(RuntimeError):
    """Raised when the WASM interactive renderer fails."""


class WasmNotAvailableError(RuntimeError):
    """Raised when WASM artifacts are not found in the package."""


class InteractiveChart:
    """Interactive chart widget backed by the ferrum WASM renderer.

    Uses ``anywidget`` for Jupyter integration when available. Falls back
    to a non-interactive container when ``anywidget`` is not installed.

    Parameters
    ----------
    chart : Chart
        The chart to render interactively.
    """

    def __init__(self, chart: "Chart") -> None:
        self._chart = chart
        self._scene_json = _render_scene_json(chart)
        self._selection_callbacks: list[Callable] = []
        self._widget: Any = None
        self._try_init_widget()

    def _try_init_widget(self) -> None:
        try:
            import anywidget
            import traitlets

            esm = _build_anywidget_esm()

            class _FerrumWidget(anywidget.AnyWidget):
                _esm = esm
                _css = (_WASM_DIR / "ferrum-interactive.css").read_text()
                scene_json = traitlets.Unicode("").tag(sync=True)
                selection_state = traitlets.Dict({}).tag(sync=True)
                interaction_config = traitlets.Unicode("{}").tag(sync=True)
                zoom_state = traitlets.Unicode("{}").tag(sync=True)

            w = _FerrumWidget()
            w.scene_json = self._scene_json
            w.interaction_config = self._extract_interaction_config(self._scene_json)
            w.observe(self._on_zoom_change, names=["zoom_state"])
            self._widget = w
        except ImportError:
            pass

    def _on_zoom_change(self, change: Any) -> None:
        """Rebuild the scene with updated domain when the JS zoom state changes."""
        import json as _json
        zoom = _json.loads(change.get("new", "{}"))
        if not zoom:
            return
        try:
            # Apply per-panel xlim/ylim overrides from JS zoom state.
            new_chart = self._apply_zoom_domains(zoom)
            new_scene = _render_scene_json(new_chart)
            if self._widget is not None:
                self._widget.scene_json = new_scene
                self._widget.interaction_config = self._extract_interaction_config(new_scene)
        except Exception as exc:
            _log.warning("zoom rebuild failed: %s", exc, exc_info=True)

    def _apply_zoom_domains(self, zoom: dict) -> "Chart":
        """Apply per-panel domain overrides from a zoom_state dict to a cloned chart."""
        from ferrum.coord import CoordCartesian
        new_chart = self._chart._clone()
        # zoom_state structure: {"0": {"x_domain": [lo, hi], "y_domain": [lo, hi]}, ...}
        panel_zoom = zoom.get("0", zoom)  # single-panel shorthand
        if not isinstance(panel_zoom, dict):
            return new_chart
        x_dom = panel_zoom.get("x_domain")
        y_dom = panel_zoom.get("y_domain")
        if x_dom or y_dom:
            xlim = tuple(x_dom) if x_dom else None
            ylim = tuple(y_dom) if y_dom else None
            new_chart = new_chart.coord(CoordCartesian(xlim=xlim, ylim=ylim))
        return new_chart

    @staticmethod
    def _extract_interaction_config(scene_json: str) -> str:
        """Extract the interaction sub-object from a scene JSON string."""
        import json as _json
        try:
            scene = _json.loads(scene_json)
            interaction = scene.get("interaction", {})
            return _json.dumps(interaction)
        except Exception as exc:
            _log.debug("could not extract interaction config: %s", exc)
            return "{}"

    @property
    def scene_json(self) -> str:
        return self._scene_json

    @property
    def selection_state(self) -> dict:
        if self._widget is not None:
            return self._widget.selection_state
        return {}

    def on_selection_change(self, callback: Callable) -> None:
        """Register a Python callback for selection state changes."""
        self._selection_callbacks.append(callback)
        if self._widget is not None:
            self._widget.observe(
                lambda change: callback(change["new"]),
                names=["selection_state"],
            )

    def save(self, path: str, **kwargs: Any) -> None:
        """Save as self-contained HTML file."""
        from ferrum.display import save_chart

        save_chart(self._chart, path, format="html", **kwargs)

    def _repr_mimebundle_(self, **kwargs: Any) -> dict | None:
        if self._widget is not None:
            return self._widget._repr_mimebundle_(**kwargs)
        return None

    def __repr__(self) -> str:
        return f"InteractiveChart(selections={len(self._chart._selections)})"


def _render_scene_json(chart: "Chart") -> str:
    from ferrum._core import render_interactive

    spec, data, viewport, theme_dict = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme_dict)


def merge_scene_graphs(
    scene_jsons: list[str],
    layout: list[dict],
) -> str:
    """Merge multiple SceneGraph JSONs into a single unified SceneGraph.

    Parameters
    ----------
    scene_jsons : list[str]
        Per-sub-chart SceneGraph JSON strings.
    layout : list[dict]
        Per-sub-chart layout info: ``{"x_offset": float, "y_offset": float}``.

    Returns
    -------
    str
        Merged SceneGraph JSON.
    """
    import json

    if not scene_jsons:
        return "{}"

    merged = json.loads(scene_jsons[0])
    all_panels = list(merged.get("panels", []))
    all_title = list(merged.get("title", []))
    all_legend = list(merged.get("legend", []))
    all_decorations = list(merged.get("decorations", []))
    max_w = merged.get("width", 0)
    max_h = merged.get("height", 0)

    for i, sj in enumerate(scene_jsons[1:], start=1):
        scene = json.loads(sj)
        offset = layout[i] if i < len(layout) else {"x_offset": 0, "y_offset": 0}
        dx = offset.get("x_offset", 0)
        dy = offset.get("y_offset", 0)

        for panel in scene.get("panels", []):
            panel["id"] = len(all_panels)
            pa = panel.get("plot_area", {})
            pa["x"] = pa.get("x", 0) + dx
            pa["y"] = pa.get("y", 0) + dy
            clip = panel.get("clip", {})
            clip["x"] = clip.get("x", 0) + dx
            clip["y"] = clip.get("y", 0) + dy
            _offset_nodes(panel.get("grid", []), dx, dy)
            _offset_nodes(panel.get("axes", []), dx, dy)
            _offset_nodes(panel.get("annotations", []), dx, dy)
            _offset_nodes(panel.get("strip_title", []), dx, dy)
            for batch in panel.get("marks", []):
                _offset_nodes(batch.get("nodes", []), dx, dy)
            all_panels.append(panel)

        _offset_nodes(scene.get("title", []), dx, dy)
        all_title.extend(scene.get("title", []))
        _offset_nodes(scene.get("legend", []), dx, dy)
        all_legend.extend(scene.get("legend", []))
        all_decorations.extend(scene.get("decorations", []))

        sw = scene.get("width", 0) + dx
        sh = scene.get("height", 0) + dy
        max_w = max(max_w, sw)
        max_h = max(max_h, sh)

    merged["panels"] = all_panels
    merged["title"] = all_title
    merged["legend"] = all_legend
    merged["decorations"] = all_decorations
    merged["width"] = max_w
    merged["height"] = max_h
    return json.dumps(merged)


def _offset_nodes(nodes: list, dx: float, dy: float) -> None:
    """Shift SceneNode positions by (dx, dy) in-place."""
    for node in nodes:
        op = node.get("op")
        if op == "circle":
            node["cx"] = node.get("cx", 0) + dx
            node["cy"] = node.get("cy", 0) + dy
        elif op == "rect":
            node["x"] = node.get("x", 0) + dx
            node["y"] = node.get("y", 0) + dy
        elif op == "line":
            node["x1"] = node.get("x1", 0) + dx
            node["y1"] = node.get("y1", 0) + dy
            node["x2"] = node.get("x2", 0) + dx
            node["y2"] = node.get("y2", 0) + dy
        elif op == "text":
            node["x"] = node.get("x", 0) + dx
            node["y"] = node.get("y", 0) + dy
        elif op == "image":
            node["x"] = node.get("x", 0) + dx
            node["y"] = node.get("y", 0) + dy
        elif op == "polyline":
            pts = node.get("points", [])
            node["points"] = [[p[0] + dx, p[1] + dy] for p in pts]
        elif op == "polygon":
            pts = node.get("points", [])
            node["points"] = [[p[0] + dx, p[1] + dy] for p in pts]
        elif op == "path":
            _offset_path_cmds(node.get("commands", []), dx, dy)
        elif op == "group":
            _offset_nodes(node.get("children", []), dx, dy)


def _offset_path_cmds(cmds: list, dx: float, dy: float) -> None:
    for cmd in cmds:
        op = cmd.get("op")
        if op in ("move_to", "line_to"):
            cmd["x"] = cmd.get("x", 0) + dx
            cmd["y"] = cmd.get("y", 0) + dy
        elif op == "h_line_to":
            cmd["x"] = cmd.get("x", 0) + dx
        elif op == "v_line_to":
            cmd["y"] = cmd.get("y", 0) + dy
        elif op == "quad_to":
            cmd["cx"] = cmd.get("cx", 0) + dx
            cmd["cy"] = cmd.get("cy", 0) + dy
            cmd["x"] = cmd.get("x", 0) + dx
            cmd["y"] = cmd.get("y", 0) + dy
        elif op == "cubic_to":
            cmd["c1x"] = cmd.get("c1x", 0) + dx
            cmd["c1y"] = cmd.get("c1y", 0) + dy
            cmd["c2x"] = cmd.get("c2x", 0) + dx
            cmd["c2y"] = cmd.get("c2y", 0) + dy
            cmd["x"] = cmd.get("x", 0) + dx
            cmd["y"] = cmd.get("y", 0) + dy
        elif op == "arc_to":
            cmd["x"] = cmd.get("x", 0) + dx
            cmd["y"] = cmd.get("y", 0) + dy
