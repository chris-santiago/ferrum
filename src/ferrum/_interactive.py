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

from ferrum._scene import ZoomRebuildable, _render_scene

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
    prepends the wasm-bindgen glue and the D3 interactions bundle, and
    substitutes ``__B64__`` with the base64-encoded WASM blob.  All JS
    lives in source files in ``_wasm/`` — never as embedded strings in Python.
    """
    import base64

    from ferrum._html import _convert_d3_exports, _read_wasm_artifact

    js_glue = _read_wasm_artifact("ferrum_wasm.js").decode("utf-8")
    wasm_b64 = base64.b64encode(_read_wasm_artifact("ferrum_wasm_bg.wasm")).decode("ascii")

    # D3 bundle: convert `export { ri as brush, ... }` to `var brush=ri, ...;`
    # so D3 functions are module-scoped (accessible to anywidget JS below).
    d3_source = (_WASM_DIR / "d3-interactions.js").read_text()
    d3_js = _convert_d3_exports(d3_source)

    anywidget_js = (_WASM_DIR / "ferrum-anywidget.js").read_text()

    return js_glue + "\n\n" + d3_js + "\n\n" + anywidget_js.replace("'__B64__'", f"'{wasm_b64}'")


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
            packed_data = traitlets.Bytes(b"").tag(sync=True)
            selection_state = traitlets.Dict({}).tag(sync=True)
            interaction_config = traitlets.Unicode("{}").tag(sync=True)
            zoom_state = traitlets.Unicode("{}").tag(sync=True)

        _WIDGET_CLASS = _FerrumWidget
        return _WIDGET_CLASS
    except ImportError:
        _WIDGET_CLASS_UNAVAILABLE = True
        return None


class InteractiveChart:
    """Interactive chart widget backed by the ferrum WASM renderer.

    Uses ``anywidget`` for Jupyter integration when available. Falls back
    to a non-interactive container when ``anywidget`` is not installed.

    Parameters
    ----------
    chart : Chart or _ChartLike
        A chart or composition to render interactively.
    toolbar : bool, default True
        Whether to show the interactive toolbar (zoom/pan controls, export
        button). Set to ``False`` to render without the toolbar.
    """

    def __init__(self, chart: "Chart", *, toolbar: bool = True) -> None:
        self._chart = chart
        self._toolbar = toolbar
        self._scene_json, self._packed_data = _render_scene(chart)
        self._selection_callbacks: list[Callable] = []
        self._widget: Any = None
        self._output_widget: Any = None  # ipywidgets.Output, created lazily by on_selection_change
        self._try_init_widget()

    def _build_interaction_config(self, scene_json: str) -> str:
        """Extract interaction config from scene JSON, overriding toolbar with the stored flag."""
        import json as _json

        try:
            cfg = _json.loads(self._extract_interaction_config(scene_json))
        except (ValueError, TypeError, KeyError):
            cfg = {}
        cfg["toolbar"] = self._toolbar
        return _json.dumps(cfg)

    def _try_init_widget(self) -> None:
        cls = _get_widget_class()
        if cls is None:
            return
        w = cls()
        with w.hold_sync():
            w.scene_json = self._scene_json
            w.packed_data = self._packed_data
            w.interaction_config = self._build_interaction_config(self._scene_json)
        w.observe(self._on_zoom_change, names=["zoom_state"])
        self._widget = w

    def _on_zoom_change(self, change: Any) -> None:
        """Rebuild the scene with updated domain when the JS zoom state changes."""
        import json as _json

        # Compositions don't support zoom rebuild (no _clone / coord override).
        if not isinstance(self._chart, ZoomRebuildable):
            return

        zoom = _json.loads(change.get("new", "{}"))
        if not zoom:
            return
        try:
            # Apply per-panel xlim/ylim overrides from JS zoom state.
            new_chart = self._apply_zoom_domains(zoom)
            new_json, new_packed = _render_scene(new_chart)
            if self._widget is not None:
                with self._widget.hold_sync():
                    self._widget.scene_json = new_json
                    self._widget.packed_data = new_packed
                    self._widget.interaction_config = self._build_interaction_config(new_json)
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

        Delegates to :func:`ferrum._html._extract_interaction_config`.
        """
        from ferrum._html import _extract_interaction_config

        return _extract_interaction_config(scene_json)

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

    def save(
        self,
        path: str,
        *,
        format: str | None = None,
        embed_wasm: bool = True,
        csp_nonce: str | None = None,
        scale: float = 2.0,
    ) -> None:
        """Save the underlying chart to a file, dispatching on extension/format.

        Routes through :func:`ferrum.display.save_chart` — the single
        save-format router shared with ``Chart.save`` and the composition
        ``save`` methods — so the path extension is honored (``.html`` /
        ``.svg`` / ``.png`` / ``.json`` / ``.pdf``) rather than always writing
        HTML.  The interactive ``toolbar`` flag captured at construction time
        is carried into HTML output.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : {"svg", "png", "html", "json", "pdf"}, optional
            Explicit format override.  When omitted the extension of ``path``
            is used.
        embed_wasm : bool, default True
            For ``"html"`` format only.  When True, the WASM binary is
            base64-inlined for single-file distribution.  When False, an
            adjacent ``ferrum_wasm_bg.wasm`` sidecar is written alongside.
        csp_nonce : str, optional
            For ``"html"`` format only.  When provided, both the ``<style>``
            and ``<script type="module">`` tags receive a ``nonce="..."``
            attribute so they pass strict Content-Security-Policy headers.
        scale : float, default 2.0
            Pixel-density multiplier for PNG and PDF output.  Has no effect on
            SVG, HTML, or JSON exports.

        Raises
        ------
        ValueError
            If the path extension (or *format*) is not a recognised export
            format.
        """
        from ferrum.display import save_chart

        save_chart(
            self._chart,
            path,
            format=format,
            embed_wasm=embed_wasm,
            csp_nonce=csp_nonce,
            scale=scale,
            toolbar=self._toolbar,
        )

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
        selections = getattr(self._chart, "_selections", [])
        return f"InteractiveChart(selections={len(selections)})"
