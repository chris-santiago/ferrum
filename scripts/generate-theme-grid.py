"""Generate a 4x3 theme grid PNG showing all 12 built-in themes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ferrum as fm
import ferrum.themes.builtins as builtins
import numpy as np
import polars as pl

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

rng = np.random.default_rng(42)
n = 80
x = rng.uniform(1, 9, n)
y = 2 * x + rng.normal(0, 2, n)
df = pl.DataFrame({"x": x, "y": y})

tmp = Path(tempfile.mkdtemp(prefix="theme_grid_"))

panels: list[Path] = []
for attr_name, display_name in THEMES:
    theme = getattr(builtins, attr_name)
    chart = (
        fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
        + fm.Chart(df).mark_smooth(method="lm").encode(x="x:Q", y="y:Q")
    )
    chart = chart.properties(title=display_name, width=280, height=220).theme(theme)
    png_path = tmp / f"{attr_name}.png"
    chart.save(str(png_path), format="png")
    panels.append(png_path)
    print(f"  {display_name}")

print(f"\nCompositing {ROWS}x{COLS} grid from {len(panels)} panels...")

from PIL import Image

imgs = [Image.open(p).convert("RGBA") for p in panels]
pw, ph = imgs[0].size
gap = 4

grid_w = COLS * pw + (COLS - 1) * gap
grid_h = ROWS * ph + (ROWS - 1) * gap
grid = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 0))

for i, im in enumerate(imgs):
    col = i % COLS
    row = i // COLS
    x_pos = col * (pw + gap)
    y_pos = row * (ph + gap)
    grid.paste(im, (x_pos, y_pos), im)

out = Path("docs/site/guide/img/theme-grid.png")
grid.save(str(out), optimize=True)
print(f"Saved: {out} ({grid_w}x{grid_h})")
