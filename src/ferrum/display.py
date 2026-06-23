"""Output orchestration: save, show, _repr_*_."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from ferrum.chart import Chart


def save_chart(
    chart: object,
    path: Union[str, Path],
    *,
    format: str | None = None,
    embed_wasm: bool = True,
    csp_nonce: str | None = None,
    scale: float = 2.0,
    toolbar: bool = True,
) -> None:
    """Save a chart-like object to disk, dispatching on extension or ``format``.

    This is the single save-format router for the whole library: ``Chart``,
    every composition wrapper (``HConcatChart`` / ``VConcatChart`` /
    ``JointChart`` / ``LayerChart`` / ...), and ``InteractiveChart`` all route
    their ``.save()`` through here so there is one format table and one HTML /
    title resolution path.

    Parameters
    ----------
    chart : chart-like
        Any object exposing the chart-like rendering surface — ``to_svg()``,
        ``to_png(scale=...)`` and the ``(scene_json, packed_data)`` producer
        used for HTML / JSON.  Callers typically pass a chart with render
        overrides already applied (e.g. via ``Chart.save(raster=False)``).
    path : str or Path
        Destination file path.  The extension determines the format unless
        ``format`` is given explicitly.
    format : {"svg", "png", "html", "json", "pdf"}, optional
        Explicit format override.  When omitted the extension of ``path``
        is used.
    embed_wasm : bool
        For ``"html"`` format only.  When True (default), the WASM binary is
        base64-inlined for single-file distribution.  When False, an adjacent
        ``ferrum_wasm_bg.wasm`` sidecar file is required.
    csp_nonce : str, optional
        For ``"html"`` format only.  When provided, both the ``<style>`` and
        ``<script type="module">`` tags receive a ``nonce="..."`` attribute
        so they pass strict Content-Security-Policy headers.
    scale : float, default 2.0
        Pixel-density multiplier for PNG and PDF output.  Has no effect on
        SVG, HTML, or JSON exports.
    toolbar : bool, default True
        For ``"html"`` format only.  When False, the interactive toolbar is
        hidden in the rendered HTML.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    >>> fm.save_chart(chart, "scatter.svg")
    >>> fm.save_chart(chart, "scatter.png")
    >>> fm.save_chart(chart, "scatter.html")
    >>> fm.save_chart(chart, "scatter.json")
    >>> fm.save_chart(chart, "scatter.pdf")
    """
    path = Path(path)
    fmt = format or path.suffix.lstrip(".").lower()
    if fmt == "svg":
        path.write_text(chart.to_svg())
    elif fmt == "png":
        path.write_bytes(chart.to_png(scale=scale))
    elif fmt == "html":
        from ferrum._html import _copy_wasm_sidecar

        html = html_string(
            chart,
            embed_wasm=embed_wasm,
            toolbar=toolbar,
            csp_nonce=csp_nonce,
        )
        path.write_text(html)
        if not embed_wasm:
            _copy_wasm_sidecar(path)
    elif fmt == "json":
        scene_json, _ = _render_scene_json(chart)
        path.write_text(scene_json)
    elif fmt == "pdf":
        save_chart_svg(chart.to_svg(), str(path), scale=scale)
    elif fmt == "":
        raise ValueError(f"save({str(path)!r}) requires a format= or a path with extension.")
    else:
        raise ValueError(f"unknown extension {fmt!r}; supported: svg, png, html, json, pdf.")


def html_string(
    chart_like: object,
    *,
    embed_wasm: bool = True,
    toolbar: bool = True,
    csp_nonce: str | None = None,
) -> str:
    """Assemble a self-contained interactive HTML document for any chart-like.

    This is the single HTML-assembly path for the whole library.  It produces
    ``(scene_json, packed_data)`` through the uniform scene producer
    (:func:`_render_scene_json`, which dispatches to a plain ``Chart``'s Rust
    renderer or a composition's ``_render_interactive``), resolves the
    browser-tab ``<title>`` via :func:`figure_title_text`, and calls
    :func:`ferrum._html.assemble_html`.

    ``Chart.to_html``, ``_CompositeBase.to_html``, :func:`save_chart`'s HTML
    branch, and ``InteractiveChart.save`` all route through here, so the HTML
    output and tab title are identical regardless of which entry point a caller
    reaches.  It never constructs a live widget, so it works in environments
    without ``anywidget`` installed.

    Parameters
    ----------
    chart_like : chart-like
        Any ``Chart`` or composition wrapper.
    embed_wasm : bool, default True
        When True, base64-inline the WASM binary for single-file distribution.
        When False, the document references an adjacent ``ferrum_wasm_bg.wasm``
        sidecar that the caller must place beside it.
    toolbar : bool, default True
        When False, the interactive toolbar is hidden in the rendered HTML.
    csp_nonce : str, optional
        When provided, both the ``<style>`` and ``<script type="module">`` tags
        receive a ``nonce="..."`` attribute for strict Content-Security-Policy.
    """
    from ferrum._html import assemble_html

    scene_json, packed_data = _render_scene_json(chart_like)
    return assemble_html(
        scene_json,
        packed_data=packed_data,
        title=figure_title_text(chart_like),
        embed_wasm=embed_wasm,
        csp_nonce=csp_nonce,
        toolbar=toolbar,
    )


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

            display(SVG(chart.to_svg()))
            return
        except Exception:
            pass
    # Browser fallback: write temp HTML, open in browser
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(_wrap_svg_in_html(chart.to_svg(), title=figure_title_text(chart)))
        url = f"file://{f.name}"
    webbrowser.open(url)


def _extract_title_text(raw_title: object) -> str:
    """Extract a plain text string from a Title dataclass or fallback.

    ``Chart._title`` is a ``Title`` dataclass (with a ``.text`` attribute)
    or ``None``.  This helper avoids embedding ``Title(text='...', ...)``
    repr into HTML ``<title>`` tags and headings.
    """
    if raw_title is not None and hasattr(raw_title, "text"):
        return raw_title.text or "Ferrum chart"
    return str(raw_title) if raw_title else "Ferrum chart"


def figure_title_text(chart_like: object) -> str:
    """Resolve the document ``<title>`` text for any chart-like object.

    This is the **sole** title-resolution entry point for every HTML / browser
    export path.  Title resolution is three-tiered:

    - :func:`figure_title_text` (this function) — the dispatcher every caller
      uses.  It prefers a chart-like's per-object ``_figure_title_text()``
      accessor when present, and otherwise reads ``_title`` directly.
    - ``_figure_title_text()`` — the per-object hook.  ``_ChartLike`` resolves
      a single chart's ``_title``; ``_CompositeBase`` overrides it to resolve
      the composite's ``_figure_title``.
    - :func:`_extract_title_text` — the leaf that turns a ``Title`` dataclass
      (or plain string / ``None``) into the displayed text.

    Routing every caller through this dispatcher (rather than the leaf) is what
    keeps the browser-tab title identical across ``Chart.to_html``,
    ``_CompositeBase.to_html``, :func:`save_chart`, and ``InteractiveChart.save``.
    """
    accessor = getattr(chart_like, "_figure_title_text", None)
    if callable(accessor):
        return cast(str, accessor())
    return _extract_title_text(getattr(chart_like, "_title", None))


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


def save_chart_svg(svg: str, path: str, *, scale: float = 2.0) -> None:
    """Save a chart to PDF from its SVG string.

    This entry point is used by composition types (``_ChartLike.save()``) that
    produce SVG but do not have a single ``ChartSpec`` + data to pass through
    ``save_chart``.  It always writes a PDF file — format selection is the
    caller's responsibility.

    Parameters
    ----------
    svg : str
        Complete SVG document string (as returned by ``to_svg()``).
    path : str
        Destination file path for the PDF output.
    scale : float, default 2.0
        Pixel-density multiplier for rasterisation.
    """
    from ferrum._core import rasterize_svg
    from ferrum._pdf import _png_to_minimal_pdf

    png_bytes = bytes(rasterize_svg(svg, scale=scale))
    Path(path).write_bytes(_png_to_minimal_pdf(png_bytes))


def _render_scene_json(chart: object) -> tuple[str, bytes]:
    """Render any chart-like to SceneGraph JSON + packed binary data.

    This is the uniform ``(scene_json, packed_data)`` producer used by every
    HTML / JSON export path.  It delegates to :func:`ferrum._interactive._render_scene`,
    which dispatches to a composition's ``_render_interactive()`` when present
    and otherwise drives the Rust ``render_interactive`` for a plain ``Chart``.

    Returns
    -------
    tuple[str, bytes]
        ``(scene_json, packed_data)``.
    """
    from ferrum._interactive import _render_scene

    return _render_scene(chart)
