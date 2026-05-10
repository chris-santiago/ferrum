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
    """Save chart to disk. Format inferred from extension when format=None."""
    path = Path(path)
    fmt = format or path.suffix.lstrip(".").lower()
    if fmt == "svg":
        path.write_text(chart.show_svg(**render_kwargs))
    elif fmt == "png":
        path.write_bytes(chart.show_png(**render_kwargs))
    elif fmt in ("html", "json"):
        raise NotImplementedError(
            f"save({fmt!r}) is planned for Phase 9. "
            f"Use 'svg' or 'png' in Phase 8a."
        )
    elif fmt == "":
        raise ValueError(
            f"save({str(path)!r}) requires a format= or a path with extension."
        )
    else:
        raise ValueError(
            f"unknown extension {fmt!r}; supported: svg, png. "
            f"(html, json planned for Phase 9.)"
        )


def show_chart(chart: "Chart") -> None:
    """Display chart. Order: Jupyter inline → browser fallback."""
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
