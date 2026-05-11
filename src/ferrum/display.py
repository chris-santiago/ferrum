"""Output orchestration: save, show, _repr_*_."""
from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ferrum.chart import Chart


def save_chart(chart: "Chart", path: Union[str, Path], *,
               format: str | None = None, **render_kwargs) -> None:
    """Save a chart to disk as SVG or PNG.

    The output format is derived from ``path``'s file extension when
    ``format`` is not supplied.  HTML and JSON output raise
    ``NotImplementedError`` (planned for Phase 11+).

    Parameters
    ----------
    chart : Chart
        The chart to save.
    path : str or Path
        Destination file path.  The extension determines the format unless
        ``format`` is given explicitly.
    format : {"svg", "png"}, optional
        Explicit format override.  When omitted the extension of ``path``
        is used.  Raises ``ValueError`` if the path has no extension and
        ``format`` is also omitted.
    **render_kwargs
        Additional keyword arguments forwarded to ``chart.show_svg()`` or
        ``chart.show_png()``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the format cannot be determined or is not ``"svg"`` / ``"png"``.
    NotImplementedError
        If ``format`` is ``"html"`` or ``"json"`` (planned for Phase 11+).

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).mark_point().encode(x="hp", y="mpg")
    >>> fm.save_chart(chart, "scatter.svg")
    >>> fm.save_chart(chart, "scatter.png")
    >>> fm.save_chart(chart, "output", format="svg")
    """
    path = Path(path)
    fmt = format or path.suffix.lstrip(".").lower()
    if fmt == "svg":
        path.write_text(chart.show_svg(**render_kwargs))
    elif fmt == "png":
        path.write_bytes(chart.show_png(**render_kwargs))
    elif fmt in ("html", "json"):
        raise NotImplementedError(
            f"save({fmt!r}) is planned for Phase 11+. "
            f"Use 'svg' or 'png' today."
        )
    elif fmt == "":
        raise ValueError(
            f"save({str(path)!r}) requires a format= or a path with extension."
        )
    else:
        raise ValueError(
            f"unknown extension {fmt!r}; supported: svg, png. "
            f"(html, json planned for Phase 11+.)"
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
            "ZMQInteractiveShell", "TerminalInteractiveShell"
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
