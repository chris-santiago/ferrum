"""Output orchestration: save, show, _repr_*_."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ferrum.chart import Chart


def save_chart(
    chart: "Chart",
    path: Union[str, Path],
    *,
    format: str | None = None,
    embed_wasm: bool = True,
) -> None:
    """Save a chart to disk.

    Parameters
    ----------
    chart : Chart
        The chart to save.  Callers typically pass a chart with render
        overrides already applied (e.g. via ``Chart.save(raster=False)``).
    path : str or Path
        Destination file path.  The extension determines the format unless
        ``format`` is given explicitly.
    format : {"svg", "png", "html", "json"}, optional
        Explicit format override.  When omitted the extension of ``path``
        is used.
    embed_wasm : bool
        For ``"html"`` format only.  When True (default), the WASM binary is
        base64-inlined for single-file distribution.  When False, an adjacent
        ``ferrum_wasm_bg.wasm`` sidecar file is required.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    >>> fm.save_chart(chart, "scatter.svg")
    >>> fm.save_chart(chart, "scatter.png")
    >>> fm.save_chart(chart, "scatter.html")
    >>> fm.save_chart(chart, "scatter.json")
    """
    path = Path(path)
    fmt = format or path.suffix.lstrip(".").lower()
    if fmt == "svg":
        path.write_text(chart.show_svg())
    elif fmt == "png":
        path.write_bytes(chart.show_png())
    elif fmt == "html":
        scene_json, packed_data = _render_scene_json(chart)
        from ferrum._html import assemble_html, _copy_wasm_sidecar

        html = assemble_html(
            scene_json,
            packed_data=packed_data,
            title=chart._title or "Ferrum chart",
            embed_wasm=embed_wasm,
        )
        path.write_text(html)
        if not embed_wasm:
            _copy_wasm_sidecar(path)
    elif fmt == "json":
        scene_json, _ = _render_scene_json(chart)
        path.write_text(scene_json)
    elif fmt == "":
        raise ValueError(f"save({str(path)!r}) requires a format= or a path with extension.")
    else:
        raise ValueError(f"unknown extension {fmt!r}; supported: svg, png, html, json.")


def show_chart(chart: "Chart") -> None:
    """Display a chart inline in Jupyter or open it in a browser.

    Attempts Jupyter inline display first (``IPython.display.SVG``); falls
    back to writing a temporary HTML file and opening it with
    ``webbrowser.open`` when not running inside a kernel.

    Parameters
    ----------
    chart : Chart
        The chart to display.

    Returns
    -------
    None

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    >>> fm.show_chart(chart)
    """
    if _is_jupyter():
        try:
            from IPython.display import display, SVG

            display(SVG(chart.show_svg()))
            return
        except Exception:
            pass
    # Browser fallback: write temp HTML, open in browser
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(_wrap_svg_in_html(chart.show_svg(), title=chart._title or "Ferrum chart"))
        url = f"file://{f.name}"
    webbrowser.open(url)


def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel (ZMQ or terminal shell)."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ in (
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        )
    except ImportError:
        return False


def _wrap_svg_in_html(svg: str, *, title: str = "Ferrum chart") -> str:
    """Wrap an SVG string in a minimal HTML document."""
    return (
        f"<!doctype html><html><head><title>{title}</title></head>"
        f"<body style='margin:0;padding:20px;font-family:sans-serif'>"
        f"<h2>{title}</h2>{svg}</body></html>"
    )


def _render_scene_json(chart: "Chart") -> tuple[str, bytes]:
    """Render a chart to SceneGraph JSON + packed binary data for the WASM renderer.

    Returns
    -------
    tuple[str, bytes]
        (scene_json, packed_data) from ``render_interactive``.
    """
    from ferrum._core import render_interactive

    spec, data, viewport, theme_dict = chart._render_inputs()
    return render_interactive(spec, data, viewport=viewport, theme=theme_dict)
