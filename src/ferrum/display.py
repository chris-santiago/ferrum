"""Output orchestration: save, show, _repr_*_."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from ferrum.chart import Chart


def save_chart(
    chart: "Chart",
    path: Union[str, Path],
    *,
    format: str | None = None,
    embed_wasm: bool = True,
    csp_nonce: str | None = None,
    scale: float = 2.0,
    toolbar: bool = True,
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
        scene_json, packed_data = _render_scene_json(chart)
        from ferrum._html import assemble_html, _copy_wasm_sidecar

        title = _extract_title_text(chart._title)
        html = assemble_html(
            scene_json,
            packed_data=packed_data,
            title=title,
            embed_wasm=embed_wasm,
            csp_nonce=csp_nonce,
            toolbar=toolbar,
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
        f.write(_wrap_svg_in_html(chart.to_svg(), title=_extract_title_text(chart._title)))
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

    Composites expose a canonical ``_figure_title_text()`` accessor that
    resolves their figure-level title; a plain ``Chart`` carries ``_title``
    (a ``Title`` dataclass).  This helper dispatches to the accessor when it
    exists and otherwise reads ``_title``, so every HTML export path sets the
    browser-tab title consistently.
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

    png_bytes = bytes(rasterize_svg(svg, scale=scale))
    Path(path).write_bytes(_png_to_minimal_pdf(png_bytes))


def _png_to_minimal_pdf(png_bytes: bytes) -> bytes:
    """Wrap PNG image bytes in a minimal single-page PDF (zero external dependencies).

    The PDF embeds the full PNG compressed stream (``/FlateDecode``) as an
    ``XObject``.  The pixel dimensions are read from the PNG IHDR chunk so
    the page size matches the image exactly.

    Parameters
    ----------
    png_bytes : bytes
        Raw PNG data (must start with the PNG magic bytes).

    Returns
    -------
    bytes
        Valid PDF/1.4 file content.
    """
    import struct
    import zlib

    # Read image dimensions from PNG IHDR (bytes 16-23).
    img_w = struct.unpack(">I", png_bytes[16:20])[0]
    img_h = struct.unpack(">I", png_bytes[20:24])[0]

    # Decompress the PNG IDAT stream to get raw pixel data for the PDF image.
    # Walk chunks: signature (8 bytes) + chunk sequence.
    idat_blocks: list[bytes] = []
    pos = 8
    while pos < len(png_bytes):
        if pos + 8 > len(png_bytes):
            break
        chunk_len = struct.unpack(">I", png_bytes[pos : pos + 4])[0]
        chunk_type = png_bytes[pos + 4 : pos + 8]
        chunk_data = png_bytes[pos + 8 : pos + 8 + chunk_len]
        pos += 12 + chunk_len
        if chunk_type == b"IDAT":
            idat_blocks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    # Keep the raw IDAT stream (already deflated with PNG prediction filters).
    # PDF's /FlateDecode with /Predictor 15 understands PNG prediction natively,
    # so we pass the compressed IDAT bytes through unchanged.
    compressed = b"".join(idat_blocks)

    channels = _png_channels(png_bytes)
    if channels == 4:
        # RGBA: strip alpha channel by decompressing, removing alpha bytes, recompressing.
        raw = zlib.decompress(compressed)
        row_stride_in = 1 + img_w * 4  # filter byte + RGBA
        rgb_rows: list[bytes] = []
        for row_idx in range(img_h):
            row_start = row_idx * row_stride_in
            row_data = raw[row_start + 1 : row_start + row_stride_in]
            # Strip every 4th byte (alpha) — only correct for filter type 0 (None).
            # For other filter types, PNG prediction interleaves channels, so
            # stripping alpha here is lossy. Use filter 0 by re-encoding:
            rgb_pixels = bytearray()
            for px in range(img_w):
                rgb_pixels.extend(row_data[px * 4 : px * 4 + 3])
            rgb_rows.append(b"\x00" + bytes(rgb_pixels))  # filter byte 0 = None
        compressed = zlib.compress(b"".join(rgb_rows), level=6)
        channels = 3

    color_space = "/DeviceRGB"
    bits_per_component = 8

    # Build the PDF object tree.
    objects: list[bytes] = []

    def _add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)  # 1-based object number

    # Object 1: Catalog
    _add(b"<< /Type /Catalog /Pages 2 0 R >>")
    # Object 2: Pages (forward ref to page 3)
    _add(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # Object 3: Page
    page_obj = (
        f"<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {img_w} {img_h}] "
        f"/Resources << /XObject << /Im0 5 0 R >> >> "
        f"/Contents 4 0 R >>"
    ).encode()
    _add(page_obj)
    # Object 4: Content stream (draws the image filling the page)
    content_str = f"q {img_w} 0 0 {img_h} 0 0 cm /Im0 Do Q"
    content_bytes = content_str.encode()
    content_obj = (
        f"<< /Length {len(content_bytes)} >>\nstream\n".encode() + content_bytes + b"\nendstream"
    )
    _add(content_obj)
    # Object 5: Image XObject
    # Use /Predictor 15 (PNG optimal) so the PDF decoder reverses PNG row filters.
    # For RGBA→RGB stripped data, we wrote filter byte 0 (None) so /Predictor 1 works.
    decode_parms = (
        f"/DecodeParms << /Predictor 15 /Colors {channels} "
        f"/BitsPerComponent {bits_per_component} /Columns {img_w} >>"
    )
    image_obj = (
        (
            f"<< /Type /XObject /Subtype /Image "
            f"/Width {img_w} /Height {img_h} "
            f"/ColorSpace {color_space} "
            f"/BitsPerComponent {bits_per_component} "
            f"/Filter /FlateDecode {decode_parms} "
            f"/Length {len(compressed)} >>\nstream\n"
        ).encode()
        + compressed
        + b"\nendstream"
    )
    _add(image_obj)

    # Serialise the PDF body.
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = bytearray()
    offsets: list[int] = []

    for i, obj_content in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += f"{i} 0 obj\n".encode() + obj_content + b"\nendobj\n"

    # Cross-reference table.
    xref_offset = len(header) + len(body)
    n_objs = len(objects)
    xref = f"xref\n0 {n_objs + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {n_objs + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + bytes(body) + xref + trailer


def _png_channels(png_bytes: bytes) -> int:
    """Return the number of colour channels from a PNG IHDR chunk.

    Reads the colour-type byte (offset 25) and maps it to channel count.
    Falls back to 3 (RGB) for unrecognised types.
    """
    colour_type = png_bytes[25]
    # PNG colour types: 0=grey, 2=RGB, 3=indexed, 4=grey+alpha, 6=RGBA
    return {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}.get(colour_type, 3)


def _render_scene_json(chart: "Chart") -> tuple[str, bytes]:
    """Render a chart to SceneGraph JSON + packed binary data for the WASM renderer.

    Returns
    -------
    tuple[str, bytes]
        (scene_json, packed_data) from ``render_interactive``.
    """
    from ferrum._interactive import _render_scene

    return _render_scene(chart)
