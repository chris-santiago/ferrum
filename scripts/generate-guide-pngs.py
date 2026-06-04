"""Generate PNGs from guide-page code blocks for inline docs images.

Reads each markdown file, extracts python code blocks that produce charts,
runs them, and saves PNGs.

A code block is treated as a chart block when it ends with a bare identifier
line (the notebook display idiom, e.g. just ``chart`` on its own line). The
last such bare identifier in the block names the variable that gets rendered.
The variable must resolve to an object with a ``to_png`` method; blocks where
it does not (or where no bare identifier is found) are skipped silently.

Usage:
    uv run scripts/generate-guide-pngs.py
"""

from __future__ import annotations

import re
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "docs" / "site" / "guide"
GETTING_STARTED_DIR = ROOT / "docs" / "site" / "getting-started"

PAGES = [
    (GETTING_STARTED_DIR / "first-plot.md", GETTING_STARTED_DIR / "img"),
    (GUIDE_DIR / "marks-encodings.md", GUIDE_DIR / "img"),
    (GUIDE_DIR / "composition.md", GUIDE_DIR / "img"),
    (GUIDE_DIR / "themes.md", GUIDE_DIR / "img"),
    (GUIDE_DIR / "figure-helpers.md", GUIDE_DIR / "img"),
    (GUIDE_DIR / "model-diagnostics.md", GUIDE_DIR / "img"),
    (GUIDE_DIR / "recipes.md", GUIDE_DIR / "img"),
]

# Matches a line that is exactly a bare Python identifier (optionally indented).
_BARE_IDENT_RE = re.compile(r"^[ \t]*([A-Za-z_]\w*)[ \t]*$", re.MULTILINE)


def extract_blocks(md_text: str) -> list[tuple[str, str | None]]:
    """Extract python code blocks. Returns list of (code, chart_var_name | None).

    The chart variable is the last bare-identifier line in the block. If no
    such line exists, the second element is None and the block will be skipped.
    """
    blocks = []
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(md_text):
        code = match.group(1)
        matches = _BARE_IDENT_RE.findall(code)
        var_name = matches[-1] if matches else None
        blocks.append((code, var_name))
    return blocks


def run_block(code: str, var_name: str) -> bytes | None:
    """Run a code block and return PNG bytes, or None if var is not a chart.

    Returns None when the named variable does not have a ``to_png`` method,
    which causes the caller to skip the block silently.
    """
    ns: dict = {}
    compiled = compile(code, "<guide-block>", "exec")
    # Using types.FunctionType to create an isolated execution context
    fn = types.FunctionType(compiled, ns)
    fn()
    chart_obj = ns[var_name]
    if not hasattr(chart_obj, "to_png"):
        return None
    return chart_obj.to_png()


def main() -> None:
    """Run the PNG generator for all guide pages."""
    total = 0
    ok = 0

    for md_path, img_dir in PAGES:
        img_dir.mkdir(parents=True, exist_ok=True)
        text = md_path.read_text()
        blocks = extract_blocks(text)
        page_name = md_path.stem

        print(f"\n{page_name} ({len(blocks)} blocks)")
        chart_idx = 0

        for code, var_name in blocks:
            if var_name is None:
                continue

            chart_idx += 1
            total += 1
            out_name = f"{page_name}_{chart_idx:02d}.png"
            out_path = img_dir / out_name

            try:
                png = run_block(code, var_name)
                if png is None:
                    # Bare identifier resolved to a non-chart object; skip silently.
                    chart_idx -= 1
                    total -= 1
                    continue
                out_path.write_bytes(png)
                print(f"  [OK]   {out_name} ({len(png)} bytes)")
                ok += 1
            except Exception as e:
                print(f"  [FAIL] {out_name}: {e}")
                traceback.print_exc()

    print(f"\n{ok}/{total} PNGs generated successfully.")


if __name__ == "__main__":
    main()
