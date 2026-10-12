"""Generate a 4x3 theme grid PNG showing all 12 built-in themes.

Each theme renders the same scatter-with-smooth chart as its own PNG, and the
panels are tiled by placing them as embedded images in one SVG rasterized with
``resvg-py`` (a dev dependency), so no image library is needed.

Usage:
    unset CONDA_PREFIX && uv run --no-sync python scripts/generate-theme-grid.py
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import numpy as np
import polars as pl
import resvg_py

import ferrum as fm
import ferrum.themes.builtins as builtins

OUT = Path(__file__).resolve().parent.parent / "docs" / "site" / "guide" / "img" / "theme-grid.png"

THEMES = [
    ("default", "Default"),
    ("paper_ink", "Paper Ink"),
    ("slate_citrus", "Slate Citrus"),
    ("arctic_signal", "Arctic Signal"),
    ("observable", "Observable"),
    ("minimal", "Minimal"),
    ("dark", "Dark"),
    ("publication", "Publication"),
    ("economist", "Economist"),
    ("fivethirtyeight", "FiveThirtyEight"),
    ("solarized_light", "Solarized Light"),
    ("solarized_dark", "Solarized Dark"),
]

COLS = 4
ROWS = 3
#: Transparent gap between panels, in pixels.
GAP = 4


def _png_size(png: bytes) -> tuple[int, int]:
    """Return a PNG's (width, height) from its IHDR chunk."""
    return struct.unpack(">II", png[16:24])


def _panel(df: pl.DataFrame, attr_name: str, display_name: str) -> bytes:
    points = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    trend = fm.Chart(df).mark_smooth(method="lm").encode(x="x:Q", y="y:Q")
    chart = (points + trend).properties(title=display_name, width=280, height=220)
    return chart.theme(getattr(builtins, attr_name)).to_png()


def _tile(panels: list[bytes]) -> bytes:
    """Tile equally sized PNG panels row-major into one PNG."""
    sizes = {_png_size(png) for png in panels}
    if len(sizes) != 1:
        raise ValueError(f"theme grid panels differ in size: {sorted(sizes)}")
    pw, ph = sizes.pop()
    grid_w = COLS * pw + (COLS - 1) * GAP
    grid_h = ROWS * ph + (ROWS - 1) * GAP
    images = []
    for i, png in enumerate(panels):
        x, y = (i % COLS) * (pw + GAP), (i // COLS) * (ph + GAP)
        href = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        images.append(f'<image x="{x}" y="{y}" width="{pw}" height="{ph}" href="{href}"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{grid_w}" height="{grid_h}"'
        f' viewBox="0 0 {grid_w} {grid_h}">{"".join(images)}</svg>'
    )
    return bytes(resvg_py.svg_to_bytes(svg_string=svg))


def main() -> None:
    """Render every theme's panel and write the tiled grid."""
    rng = np.random.default_rng(42)
    n = 80
    x = rng.uniform(1, 9, n)
    y = 2 * x + rng.normal(0, 2, n)
    df = pl.DataFrame({"x": x, "y": y})

    panels = []
    for attr_name, display_name in THEMES:
        panels.append(_panel(df, attr_name, display_name))
        print(f"  {display_name}")

    print(f"\nTiling {ROWS}x{COLS} grid from {len(panels)} panels...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(_tile(panels))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
