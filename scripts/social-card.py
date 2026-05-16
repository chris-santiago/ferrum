"""Generate the 1200x630 social card PNG for ferrumviz.com.

Renders an iris lmplot (scatter + loess per species with CI bands)
using the paper_ink theme. Output goes to docs/site/assets/images/social-card.png.

Usage:
    uv run scripts/social-card.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import ferrum as fm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "site" / "assets" / "images" / "social-card.png"

df = pl.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

chart = fm.lmplot(
    df,
    x="sepal_length",
    y="petal_length",
    hue="species",
    method="loess",
    properties={"width": 1200, "height": 630},
    theme=fm.themes.paper_ink,
)

chart = chart.properties(
    title=fm.Title(
        "Ferrum",
        subtitle="Grammar-of-graphics visualization for Python, powered by Rust",
        anchor="start",
        font_size=28,
        font_weight="bold",
        subtitle_font_size=16,
    )
)

OUT.parent.mkdir(parents=True, exist_ok=True)
chart.save(str(OUT))
print(f"Saved social card to {OUT}  ({OUT.stat().st_size:,} bytes)")
