"""SVG-to-PNG rasterization helpers for visually inspecting ferrum goldens.

SVG byte-equality remains the regression check in the test suite; the PNGs
produced here are a side-channel for humans and multimodal agents to look at
what a golden actually renders as.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


def legend_texts(svg: str) -> list[str]:
    """Extract all SVG ``<text>`` node contents, in document order.

    Shared probe for legend-dedup assertions: counting occurrences of a
    field's title/category-label text is the discriminating signal between
    a single figure-level legend and one legend rendered per participating
    panel.
    """
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


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
