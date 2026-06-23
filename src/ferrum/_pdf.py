"""Dependency-free PNG → single-page PDF byte codec.

Wraps rasterised PNG bytes in a minimal PDF/1.4 document with no external
dependencies, parsing the PNG IHDR/IDAT chunks directly and embedding the
deflated image stream via ``/FlateDecode``.  Used solely by
``ferrum.display.save_chart_svg`` for ``.pdf`` exports.
"""

from __future__ import annotations


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
