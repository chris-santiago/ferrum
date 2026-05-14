"""InteractiveChart — anywidget-based Jupyter integration (Phase 11c).

Provides ``InteractiveChart``, an ``anywidget.AnyWidget`` subclass that
renders a ferrum chart using the WASM GPU renderer with bidirectional
selection state sync.

``Chart.interactive()`` returns an ``InteractiveChart`` when anywidget
is installed; otherwise it falls back to returning a clone (SVG path).
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ferrum.chart import Chart

_WASM_DIR = pathlib.Path(__file__).parent / "_wasm"


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

            class _FerrumWidget(anywidget.AnyWidget):
                _esm = _WASM_DIR / "ferrum-interactive.js"
                _css = _WASM_DIR / "ferrum-interactive.css"
                scene_json = traitlets.Unicode("").tag(sync=True)
                selection_state = traitlets.Dict({}).tag(sync=True)

            w = _FerrumWidget()
            w.scene_json = self._scene_json
            self._widget = w
        except ImportError:
            pass

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
