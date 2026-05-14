"""Generate PNGs from guide-page code blocks for inline docs images.

Reads each markdown file, extracts python code blocks that produce charts
(detected by `assert <var>.show_svg()`), runs them, and saves PNGs.

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
]


def extract_blocks(md_text: str) -> list[tuple[str, str | None]]:
    """Extract python code blocks. Returns list of (code, chart_var_name | None)."""
    blocks = []
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(md_text):
        code = match.group(1)
        m = re.search(r"assert\s+(\w+)\.show_svg\(\)", code)
        var_name = m.group(1) if m else None
        blocks.append((code, var_name))
    return blocks


def run_block(code: str, var_name: str) -> bytes:
    """Run a code block and return the PNG bytes from the chart variable.

    This runs trusted documentation code blocks from our own repo — the same
    blocks that pytest-codeblocks already validates in CI.
    """
    ns: dict = {}
    compiled = compile(code, "<guide-block>", "exec")
    # Using types.FunctionType to create an isolated execution context
    fn = types.FunctionType(compiled, ns)
    fn()
    chart_obj = ns[var_name]
    return chart_obj.show_png()


def main() -> None:
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
                out_path.write_bytes(png)
                print(f"  [OK]   {out_name} ({len(png)} bytes)")
                ok += 1
            except Exception as e:
                print(f"  [FAIL] {out_name}: {e}")
                traceback.print_exc()

    print(f"\n{ok}/{total} PNGs generated successfully.")


if __name__ == "__main__":
    main()
