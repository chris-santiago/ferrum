"""Generate the inline PNGs of the guide and getting-started pages from their own code blocks.

Every Python code block on a page is executed in document order in one shared
namespace, so a continuation block can use names an earlier block defined,
exactly as a reader running the page top to bottom would.

A block renders an image when the page names one for it: the first image
reference into the page's ``img/`` directory that follows the block, before
the next Python block, is that block's image. The image file name is the
block's stable name, so editing, adding or removing another block never
renames or shifts it. A block with no such reference renders nothing.

A block that names an image must end its display with a bare identifier line
(the notebook display idiom, e.g. just ``chart`` on its own line); the last
such line names the variable rendered with ``to_png()``. A named block that
raises, has no display line, or displays something without ``to_png`` is a
failure, and the script exits non-zero after rendering everything else.

Usage:
    unset CONDA_PREFIX && uv run --no-sync python scripts/generate-guide-pngs.py
"""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"

#: Pages whose blocks render their inline images, each into its own ``img/``.
PAGES = [
    SITE / "getting-started" / "first-plot.md",
    SITE / "guide" / "marks-encodings.md",
    SITE / "guide" / "composition.md",
    SITE / "guide" / "themes.md",
    SITE / "guide" / "figure-helpers.md",
    SITE / "guide" / "model-diagnostics.md",
    SITE / "guide" / "recipes.md",
]

#: Directory, relative to the page, that holds the page's block images.
IMG_DIR = "img"

_FENCE_OPEN_RE = re.compile(r"^```python[ \t]*$")
_FENCE_CLOSE_RE = re.compile(r"^```[ \t]*$")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(" + IMG_DIR + r"/([^)\s]+\.png)\)")
# A line that is exactly a bare Python identifier (optionally indented).
_BARE_IDENT_RE = re.compile(r"^[ \t]*([A-Za-z_]\w*)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class Block:
    """One Python code block of a page.

    ``image`` is the file name (inside the page's ``img/``) the page names for
    this block, or ``None`` when the block renders nothing. ``display`` is the
    last bare-identifier line of the block, or ``None`` when it has none.
    """

    code: str
    line: int
    image: str | None
    display: str | None


def extract_blocks(md_text: str) -> list[Block]:
    """Split a page into its Python blocks, each bound to the image the page names for it."""
    lines = md_text.splitlines()
    blocks: list[Block] = []
    pending: tuple[str, int] | None = None  # (code, 1-based fence line) awaiting its image
    i = 0
    while i < len(lines):
        if _FENCE_OPEN_RE.match(lines[i]):
            fence_line = i + 1
            i += 1
            body: list[str] = []
            while i < len(lines) and not _FENCE_CLOSE_RE.match(lines[i]):
                body.append(lines[i])
                i += 1
            # A new block ends the previous block's search for an image.
            if pending is not None:
                blocks.append(_bind(*pending, image=None))
            pending = ("\n".join(body) + "\n", fence_line)
        elif pending is not None and (match := _IMAGE_REF_RE.search(lines[i])):
            blocks.append(_bind(*pending, image=match.group(1)))
            pending = None
        i += 1
    if pending is not None:
        blocks.append(_bind(*pending, image=None))
    return blocks


def _bind(code: str, line: int, *, image: str | None) -> Block:
    displays = _BARE_IDENT_RE.findall(code)
    return Block(code=code, line=line, image=image, display=displays[-1] if displays else None)


def render_page(md_path: Path) -> tuple[list[Path], list[str]]:
    """Execute every block of one page and write the images its blocks name.

    Returns the written image paths and one message per failed named block.
    """
    img_dir = md_path.parent / IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    page = md_path.relative_to(SITE)
    namespace: dict = {}
    written: list[Path] = []
    failures: list[str] = []
    for block in extract_blocks(md_path.read_text()):
        where = f"{page}:{block.line}"
        try:
            exec(compile(block.code, f"<{where}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            if block.image is not None:
                failures.append(f"{where} -> {block.image}: {exc!r}")
                traceback.print_exc()
            else:
                print(f"  [WARN] {where} (renders no image) raised {exc!r}")
            continue
        if block.image is None:
            continue
        chart = namespace.get(block.display) if block.display else None
        if not hasattr(chart, "to_png"):
            failures.append(
                f"{where} -> {block.image}: the block must end with a bare identifier"
                f" naming a chart (found {block.display!r})"
            )
            continue
        out_path = img_dir / block.image
        if out_path in written:
            failures.append(f"{where} -> {block.image}: another block already names this image")
            continue
        try:
            png = chart.to_png()
        except Exception as exc:
            failures.append(f"{where} -> {block.image}: {exc!r}")
            traceback.print_exc()
            continue
        out_path.write_bytes(png)
        written.append(out_path)
        print(f"  [OK]   {out_path.relative_to(SITE)} ({len(png)} bytes)")
    return written, failures


def main() -> int:
    """Render the named blocks of every page; return the process exit status."""
    failures: list[str] = []
    total = 0
    for md_path in PAGES:
        print(f"\n{md_path.relative_to(SITE)}")
        written, page_failures = render_page(md_path)
        total += len(written)
        failures.extend(page_failures)
    print(f"\n{total} PNGs written.")
    for failure in failures:
        print(f"  [FAIL] {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
