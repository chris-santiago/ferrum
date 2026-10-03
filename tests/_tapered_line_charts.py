"""The tapered-line charts behind ``tests/goldens/tapered_line/``.

Shared by the golden SVG tests (``tests/test_golden_tapered_line.py``) and the
WASM capture tests (``tests/test_tapered_line_pixels.py``), so the capture
measures exactly the chart each golden pins. No assertions live here.

Every chart's size column rises left to right, so a correct render thickens
toward the right and its legend swatches thicken down the legend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl

import ferrum as fm

GOLDENS_DIR = Path(__file__).parent / "goldens" / "tapered_line"

# A visible taper: point areas 20..400 px^2 map to widths ~5.0..22.6 px.
WIDE = fm.LinearScale(range=[20.0, 400.0])


def zigzag(n: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "y": [float(i % 2) * 2.0 + i * 0.3 for i in range(n)],
            "v": [float(i + 1) for i in range(n)],
        }
    )


def sized(df: pl.DataFrame, size: object, title: str, **mark_kwargs: object) -> fm.Chart:
    return (
        fm.Chart(df)
        .mark_line(**mark_kwargs)
        .encode(x="x", y="y", size=size)
        .properties(width=420, height=260, title=title)
    )


def linear_legend() -> fm.Chart:
    """Default size band (widths ~2.3..6.8 px), line-swatch legend."""
    return sized(zigzag(), "v", "Linear taper")


def curved() -> fm.Chart:
    return sized(zigzag(), fm.Size("v", scale=WIDE), "Monotone", interpolate="monotone")


def step() -> fm.Chart:
    return sized(zigzag(), fm.Size("v", scale=WIDE), "Step", interpolate="step")


def _dashed_chart(**mark_kwargs: object) -> fm.Chart:
    return sized(zigzag(), fm.Size("v", scale=WIDE), "Dashed", **mark_kwargs)


def dashed() -> fm.Chart:
    return _dashed_chart(stroke_dash=[14.0, 6.0])


def dashed_solid() -> fm.Chart:
    """The solid twin of ``dashed()``: same data, size and title, minus the dash."""
    return _dashed_chart()


def hairpin_opacity() -> fm.Chart:
    """A path that doubles back over itself paints its overlap once."""
    df = pl.DataFrame(
        {"x": [0.0, 4.0, 0.3, 3.0], "y": [0.0, 1.0, 1.3, 0.2], "v": [1.0, 2.0, 3.0, 4.0]}
    )
    return sized(df, fm.Size("v", scale=WIDE), "Hairpin at opacity 0.5", opacity=0.5)


def color_groups() -> fm.Chart:
    """Each color group tapers along its own vertices on the shared scale."""
    base = zigzag()
    df = pl.concat(
        [
            base.with_columns(g=pl.lit("a")),
            base.with_columns(y=pl.col("y") + 3.0, v=pl.col("v") * 0.5, g=pl.lit("b")),
        ]
    )
    return (
        fm.Chart(df)
        .mark_line()
        .encode(x="x", y="y", color="g", size=fm.Size("v", scale=WIDE))
        .properties(width=420, height=260, title="Color groups")
    )


def break_axis() -> fm.Chart:
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 100.0, 500.0, 900.0, 950.0],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return sized(df, fm.Size("v", scale=WIDE), "Break axis") + fm.BreakAxis(
        axis="y", gap=(150.0, 850.0)
    )


def facet() -> fm.Chart:
    base = zigzag()
    df = pl.concat([base.with_columns(f=pl.lit("left")), base.with_columns(f=pl.lit("right"))])
    return (
        fm.Chart(df)
        .mark_line()
        .encode(x="x", y="y", size=fm.Size("v", scale=WIDE))
        .properties(width=560, height=300, title="Facets")
        .facet("f")
    )


# Golden file name -> the chart it pins.
GOLDEN_CHARTS: dict[str, Callable[[], fm.Chart]] = {
    "linear_legend.svg": linear_legend,
    "curved.svg": curved,
    "step.svg": step,
    "dashed.svg": dashed,
    "hairpin_opacity.svg": hairpin_opacity,
    "color_groups.svg": color_groups,
    "break_axis.svg": break_axis,
    "facet.svg": facet,
}
