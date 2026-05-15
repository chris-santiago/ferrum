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

# Module-level singleton: built once, reused across all InteractiveChart instances.
_WIDGET_CLASS: Any = None
_WIDGET_CLASS_UNAVAILABLE: bool = False


def _build_anywidget_esm() -> str:
    """Build a self-contained anywidget ESM with inlined WASM.

    Reads ``ferrum-anywidget.js`` (the real JS source file in ``_wasm/``),
    prepends the wasm-bindgen glue, and substitutes ``__B64__`` with the
    base64-encoded WASM blob.  All JS lives in ``ferrum-anywidget.js`` —
    never as embedded strings in Python.
    """
    import base64

    from ferrum._html import _read_wasm_artifact

    js_glue = _read_wasm_artifact("ferrum_wasm.js").decode("utf-8")
    wasm_b64 = base64.b64encode(_read_wasm_artifact("ferrum_wasm_bg.wasm")).decode("ascii")
    anywidget_js = (_WASM_DIR / "ferrum-anywidget.js").read_text()

    return js_glue + "\n\n" + anywidget_js.replace("'__B64__'", f"'{wasm_b64}'")


def _get_widget_class() -> Any:
    """Return the singleton _FerrumWidget anywidget class, building it on first call.

    The class (and its multi-MB inlined WASM ESM string) is created once per
    Python process.  All InteractiveChart instances share the same class so
    anywidget loads the JS module exactly once, preventing repeated WASM
    base64 decoding and GPU context pressure from redundant module reloads.
    """
    global _WIDGET_CLASS, _WIDGET_CLASS_UNAVAILABLE
    if _WIDGET_CLASS_UNAVAILABLE:
        return None
    if _WIDGET_CLASS is not None:
        return _WIDGET_CLASS
    try:
        import anywidget
        import traitlets

        esm = _build_anywidget_esm()
        css = (_WASM_DIR / "ferrum-interactive.css").read_text()

        class _FerrumWidget(anywidget.AnyWidget):
            _esm = esm
            _css = css
            scene_json = traitlets.Unicode("").tag(sync=True)
            selection_state = traitlets.Dict({}).tag(sync=True)
            interaction_config = traitlets.Unicode("{}").tag(sync=True)
            zoom_state = traitlets.Unicode("{}").tag(sync=True)

        _WIDGET_CLASS = _FerrumWidget
        return _WIDGET_CLASS
    except ImportError:
        _WIDGET_CLASS_UNAVAILABLE = True
        return None


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
        self._output_widget: Any = None  # ipywidgets.Output, created lazily by on_selection_change
        self._try_init_widget()

    def _try_init_widget(self) -> None:
        cls = _get_widget_class()
        if cls is None:
            return
        w = cls()
        w.scene_json = self._scene_json
        w.interaction_config = self._extract_interaction_config(self._scene_json)
        w.observe(self._on_zoom_change, names=["zoom_state"])
        self._widget = w

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
        """Extract interaction config + top-level selections from a scene JSON string.

        The JS click handler reads `config.selections` (array of SelectionSpec
        objects with `name` and `fields`) to know which field values to capture
        when a mark is clicked.
        """
        import json as _json
        try:
            scene = _json.loads(scene_json)
            config = dict(scene.get("interaction", {}))
            config["selections"] = scene.get("selections", [])
            return _json.dumps(config)
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
        """Register a Python callback for selection state changes.

        Output from the callback is routed to an ``ipywidgets.Output`` widget
        displayed below the chart (when ipywidgets is available), ensuring
        that ``print()`` calls inside the callback appear in the notebook cell
        rather than the kernel log.  The output area clears on each new
        selection so it always shows the latest state.
        """
        self._selection_callbacks.append(callback)
        if self._widget is None:
            return

        try:
            import ipywidgets as _ipy

            if self._output_widget is None:
                self._output_widget = _ipy.Output()
            out = self._output_widget

            def _wrapped(change: Any) -> None:
                with out:
                    out.clear_output(wait=True)
                    callback(change["new"])

            self._widget.observe(_wrapped, names=["selection_state"])
        except ImportError:
            # ipywidgets absent — observe fires but print() goes to kernel log
            self._widget.observe(
                lambda change: callback(change["new"]),
                names=["selection_state"],
            )

    def save(self, path: str, **kwargs: Any) -> None:
        """Save as self-contained HTML file."""
        from ferrum.display import save_chart

        save_chart(self._chart, path, format="html", **kwargs)

    def _repr_mimebundle_(self, **kwargs: Any) -> dict | None:
        if self._widget is None:
            return None
        if self._output_widget is not None:
            try:
                import ipywidgets as _ipy
                box = _ipy.VBox([self._widget, self._output_widget])
                return box._repr_mimebundle_(**kwargs)
            except ImportError:
                pass
        return self._widget._repr_mimebundle_(**kwargs)

    def __repr__(self) -> str:
        return f"InteractiveChart(selections={len(self._chart._selections)})"


def _render_scene_json(chart: "Chart") -> str:
    import json as _json

    from ferrum._core import render_interactive

    spec, data, viewport, theme_dict = chart._render_inputs()
    if data.num_rows == 0:
        w, h = viewport
        return _json.dumps({"panels": [], "width": w, "height": h})
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
        op = node.get("type")  # SceneNode uses "type" tag; PathCmd uses "op"
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
