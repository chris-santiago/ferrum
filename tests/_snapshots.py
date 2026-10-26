"""SVG-to-PNG rasterization helpers for visually inspecting ferrum goldens.

SVG byte-equality remains the regression check in the test suite; the PNGs
produced here are a side-channel for humans and multimodal agents to look at
what a golden actually renders as.
"""

from __future__ import annotations

import os
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np


def legend_texts(svg: str) -> list[str]:
    """Extract all SVG ``<text>`` node contents, in document order.

    Shared probe for legend-dedup assertions: counting occurrences of a
    field's title/category-label text is the discriminating signal between
    a single figure-level legend and one legend rendered per participating
    panel.
    """
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def control_chars(text: str) -> set[str]:
    """Return the set of C0 control characters (excluding tab/newline/CR) in *text*.

    Shared probe for the "an unresolved format preset reached the SVG as a
    literal control byte" regression class (NF-B1): a d3-format parser that
    misreads a preset name as a type character can emit ``\\x18``/``\\x1a``/
    ``\\x1c`` straight into rendered text, which this catches without
    depending on any particular preset's output shape.
    """
    return {c for c in text if ord(c) < 0x20 and c not in "\t\n\r"}


def assert_svg_eq(actual: str, expected: str, *, name: str, regen_hint: str) -> None:
    """Assert two SVG strings are byte-equal with a fast, compact failure message.

    Equivalent in semantics to ``assert actual == expected``, but raises
    ``AssertionError`` directly rather than going through pytest's
    assertion rewriter — which would otherwise spend 60-120 seconds
    building a colorised unified diff between two ~500 KB SVG strings on a
    mismatch. This helper instead reports byte counts, the first byte at
    which the strings diverge, and a ~80-char context window on each side.

    Parameters
    ----------
    actual, expected : str
        The two SVG strings to compare.
    name : str
        Test/golden identifier surfaced in the failure message.
    regen_hint : str
        One-line instruction for refreshing the on-disk golden (e.g.
        ``"FERRUM_UPDATE_GOLDENS=1 to refresh"``); appended to the
        failure message so the operator knows what to run.

    Raises
    ------
    AssertionError
        If ``actual != expected``. The message stays small regardless of
        the SVG size so pytest's failure-path latency does not scale with
        golden size.
    """
    if actual == expected:
        return
    n = min(len(actual), len(expected))
    divergence = next((i for i in range(n) if actual[i] != expected[i]), n)
    ctx = 80
    a_ctx = actual[max(0, divergence - ctx) : divergence + ctx]
    b_ctx = expected[max(0, divergence - ctx) : divergence + ctx]
    raise AssertionError(
        f"golden mismatch for {name!r}: "
        f"got {len(actual)} bytes, expected {len(expected)} bytes; "
        f"first divergence at offset {divergence}.\n"
        f"  actual   ...{a_ctx!r}...\n"
        f"  expected ...{b_ctx!r}...\n"
        f"  hint: {regen_hint}"
    )


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = _REPO_ROOT / "tests" / "snapshots"


def _load_resvg():
    try:
        import resvg_py
    except ImportError as e:
        raise ImportError(
            "resvg-py is required for SVG snapshot rasterization. "
            "It ships in the dev dependency group: `uv sync` (or "
            "`pip install resvg-py>=0.3`)."
        ) from e
    return resvg_py


def rasterize_svg(svg: str | bytes, *, dpi: int = 96) -> bytes:
    """Convert an SVG string (or bytes) to PNG bytes."""
    resvg_py = _load_resvg()
    if isinstance(svg, bytes):
        svg = svg.decode("utf-8")
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, dpi=dpi))


def decode_png_rgba(png: bytes) -> tuple[list[list[tuple[int, int, int, int]]], int, int]:
    """Decode a PNG into ``(rows, width, height)``, ``rows[y][x]`` an RGBA tuple.

    Applies the PNG filter-byte reconstruction (None/Sub/Up/Average/Paeth)
    per scanline, without a PIL dependency. An RGB (no alpha channel) source
    reports alpha 255 for every pixel. Shared by every test that needs to
    sample a rendered pixel's color or alpha channel, so a stdlib PNG
    decoder exists exactly once in the test suite.
    """
    pos, chunks = 8, {}
    while pos < len(png):
        n = struct.unpack(">I", png[pos : pos + 4])[0]
        t = png[pos + 4 : pos + 8]
        chunks[t] = chunks.get(t, b"") + png[pos + 8 : pos + 8 + n]
        pos += 12 + n
    w, h = struct.unpack(">II", chunks[b"IHDR"][:8])
    bpp = 4 if chunks[b"IHDR"][9] == 6 else 3
    raw, stride = zlib.decompress(chunks[b"IDAT"]), 1 + w * bpp
    rows: list[bytearray] = []
    for r in range(h):
        ftype, row = raw[r * stride], bytearray(raw[r * stride + 1 : (r + 1) * stride])
        prev = rows[-1] if rows else bytearray(w * bpp)
        for i in range(len(row)):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                row[i] = (row[i] + a) & 255
            elif ftype == 2:
                row[i] = (row[i] + b) & 255
            elif ftype == 3:
                row[i] = (row[i] + (a + b) // 2) & 255
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(row)
    pixels = [
        [tuple(row[x * bpp : x * bpp + bpp]) + ((255,) if bpp == 3 else ()) for x in range(w)]
        for row in rows
    ]
    return pixels, w, h


def decode_png_ink_mask(png: bytes, threshold: int = 382) -> tuple[int, int, int, int] | None:
    """The pixel bounding box of every "ink" pixel in ``png``, or ``None``.

    A faster sibling to :func:`decode_png_rgba` for a caller that only
    needs a dark-pixel bounding box, not every pixel's full RGBA:
    ``decode_png_rgba``'s generic nested-tuple build takes seconds per
    image, too slow for a caller rasterizing many small images (e.g. one
    per label). Filter types None/Sub/Up are vectorized across the whole
    row with numpy; Average/Paeth are walked one pixel column at a time,
    following their own same-row dependency. A pixel counts as ink at
    ``threshold`` coverage or more (``r + g + b <= threshold``). Only
    handles 8-bit, non-interlaced RGB/RGBA; a PNG in any other format
    raises rather than being silently misread as if it were one of those
    two.
    """
    pos, chunks = 8, {}
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        chunk_type = png[pos + 4 : pos + 8]
        chunks[chunk_type] = chunks.get(chunk_type, b"") + png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    width, height = struct.unpack(">II", chunks[b"IHDR"][:8])
    bit_depth, color_type, _compression_method, _filter_method, interlace_method = chunks[b"IHDR"][
        8:13
    ]
    if bit_depth != 8 or color_type not in (2, 6) or interlace_method != 0:
        raise ValueError(
            f"unsupported PNG format: bit_depth={bit_depth}, color_type={color_type}, "
            f"interlace={interlace_method} (only 8-bit, non-interlaced RGB [2] or RGBA [6] "
            "is decoded)"
        )
    bpp = 4 if color_type == 6 else 3
    raw = zlib.decompress(chunks[b"IDAT"])
    stride = 1 + width * bpp
    prev = np.zeros((width, bpp), dtype=np.int16)
    xs: list[int] = []
    ys: list[int] = []
    for row_idx in range(height):
        offset = row_idx * stride
        filter_type = raw[offset]
        row = (
            np.frombuffer(raw, dtype=np.uint8, count=width * bpp, offset=offset + 1)
            .astype(np.int16)
            .reshape(width, bpp)
        )
        if filter_type == 0:
            cur = row
        elif filter_type == 1:
            cur = np.cumsum(row, axis=0) % 256
        elif filter_type == 2:
            cur = (row + prev) % 256
        elif filter_type == 3:
            cur = row.copy()
            for x in range(width):
                left = cur[x - 1] if x > 0 else 0
                cur[x] = (cur[x] + (left + prev[x]) // 2) % 256
        elif filter_type == 4:
            cur = row.copy()
            for x in range(width):
                left = cur[x - 1] if x > 0 else np.zeros(bpp, dtype=np.int16)
                up = prev[x]
                up_left = prev[x - 1] if x > 0 else np.zeros(bpp, dtype=np.int16)
                predictor = left + up - up_left
                pa = np.abs(predictor - left)
                pb = np.abs(predictor - up)
                pc = np.abs(predictor - up_left)
                nearest = np.where(
                    pa <= pb, np.where(pa <= pc, left, up_left), np.where(pb <= pc, up, up_left)
                )
                cur[x] = (cur[x] + nearest) % 256
        else:
            raise ValueError(f"unsupported PNG filter type {filter_type}")
        if bpp == 4:
            ink = (cur[:, 3] > 50) & (cur[:, 0].astype(int) + cur[:, 1] + cur[:, 2] <= threshold)
        else:
            ink = cur[:, 0].astype(int) + cur[:, 1] + cur[:, 2] <= threshold
        if ink.any():
            cols = np.nonzero(ink)[0]
            xs.extend(cols.tolist())
            ys.extend([row_idx] * len(cols))
        prev = cur
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def snapshot_golden(
    svg_path: str | Path,
    *,
    out_dir: Path | None = None,
    dpi: int = 96,
) -> Path:
    """Read an SVG golden file, rasterize it, write a PNG, and return the PNG path.

    The output path mirrors the golden's relative location under the repo:
    ``tests/goldens/phase_10/foo.svg`` -> ``<out_dir>/phase_10/foo.png``.
    """
    svg_path = Path(svg_path).resolve()
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    out_dir = Path(out_dir)

    rel = _mirror_relpath(svg_path)
    out_path = out_dir / rel.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    svg_bytes = svg_path.read_bytes()
    png_bytes = rasterize_svg(svg_bytes, dpi=dpi)
    out_path.write_bytes(png_bytes)
    return out_path


def regen_and_verify(
    golden_path: str | Path,
    svg: str | bytes,
    *,
    out_dir: Path | None = None,
    dpi: int = 96,
    emit: bool = True,
) -> Path:
    """Write ``svg`` to ``golden_path``, rasterize a PNG, and return the PNG path.

    The intended entry point for any script or fixture that regenerates a
    golden SVG. Writing the SVG and producing the inspection PNG happen in
    one call so a regen flow cannot silently skip the visual check — when
    ``emit=True`` (default), a line of the form::

        regenerated <golden>
        inspect    <png>

    is printed to stdout so the calling agent can immediately ``Read`` the
    PNG and confirm the new golden renders correctly before committing.

    See the "Goldens are not blessed until visually inspected" rule in
    CLAUDE.md for the obligation this helper exists to support.
    """
    golden_path = Path(golden_path)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(svg, bytes):
        golden_path.write_bytes(svg)
    else:
        golden_path.write_text(svg)

    png_path = snapshot_golden(golden_path, out_dir=out_dir, dpi=dpi)

    if emit:
        try:
            golden_rel = golden_path.resolve().relative_to(_REPO_ROOT)
            png_rel = png_path.resolve().relative_to(_REPO_ROOT)
        except ValueError:
            golden_rel, png_rel = golden_path, png_path
        print(f"regenerated {golden_rel}", file=sys.stdout)
        print(f"inspect    {png_rel}", file=sys.stdout)

    return png_path


def check_or_update_svg_golden(
    golden_dir: Path,
    name: str,
    svg: str,
    *,
    regen_hint: str = "rerun with FERRUM_UPDATE_GOLDENS=1 to refresh",
) -> None:
    """Compare ``svg`` against the on-disk golden ``golden_dir / name``.

    The shared compare-or-regenerate step the golden modules call (grep for
    callers; not every golden module routes through it): with
    ``FERRUM_UPDATE_GOLDENS=1`` set,
    the golden is (re)written through :func:`regen_and_verify`, which also
    rasterizes an inspection PNG and prints both paths, so a regen can never
    silently skip the visual check CLAUDE.md's golden-blessing rule
    requires. Otherwise asserts byte-for-byte equality via
    :func:`assert_svg_eq`; a missing golden is an explicit test failure, not
    a silent skip.

    Imports ``pytest`` lazily so this module (and ``scripts/snapshot-goldens.py``,
    which imports it) stays usable without pytest installed.
    """
    import pytest

    path = golden_dir / name
    if os.environ.get("FERRUM_UPDATE_GOLDENS") == "1":
        regen_and_verify(path, svg)
        return
    if not path.exists():
        pytest.fail(f"golden {name!r} does not exist; {regen_hint}")
    assert_svg_eq(svg, path.read_text(), name=name, regen_hint=regen_hint)


def find_goldens(*roots: str | Path) -> list[Path]:
    """Return all ``*.svg`` files under the given root directories."""
    out: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        out.extend(sorted(root_path.rglob("*.svg")))
    return out


def _mirror_relpath(svg_path: Path) -> Path:
    """Strip the ``tests/goldens/`` or ``tests/<dir>/goldens/`` prefix.

    Produces a relative path suitable for placement under tests/snapshots/.
    Falls back to the file's basename if no recognized prefix is found.
    """
    parts = svg_path.parts
    if "goldens" in parts:
        idx = parts.index("goldens")
        # Discriminate by what sits above "goldens":
        #   tests/goldens/phase_10/foo.svg -> phase_10/foo.svg
        #   tests/test_phase_9_e2e/goldens/foo.svg -> test_phase_9_e2e/foo.svg
        if idx >= 2 and parts[idx - 1] != "tests":
            parent_dir = parts[idx - 1]
            tail = parts[idx + 1 :]
            return Path(parent_dir, *tail)
        return Path(*parts[idx + 1 :])
    return Path(svg_path.name)


def iter_default_roots() -> Iterable[Path]:
    """The two known ferrum golden roots."""
    yield _REPO_ROOT / "tests" / "goldens"
    yield _REPO_ROOT / "tests" / "test_phase_9_e2e" / "goldens"
