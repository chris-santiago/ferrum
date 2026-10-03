"""Golden SVG tests for tapered lines (`mark_line` with an encoded `size`).

A sized line draws one filled outline per series whose thickness at each
vertex is the dot diameter `2*sqrt(size_scale(v)/pi)` of a point drawn at the
same value, and its size legend draws line swatches at those widths. The
cases cover the straight, curved and step interpolations, a dashed taper, a
half-opacity hairpin whose overlap must paint once, color groups each
tapering on their own, a primary-y break cutting a taper, and facets. Every
chart's size column rises left to right, so a correct render thickens toward
the right and its legend swatches thicken down the legend.

These are byte-comparison goldens (mirroring
``tests/test_golden_secondary_y_axis.py``), not the behavioral assertions in
``tests/test_tapered_line.py``.

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens through
``tests._snapshots.regen_and_verify``, which also rasterizes each PNG for
inspection; the default behavior is byte-identical comparison.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import ferrum as fm
from tests._snapshots import check_or_update_svg_golden

GOLDENS_DIR = Path(__file__).parent / "goldens" / "tapered_line"

# A visible taper: point areas 20..400 px^2 map to widths ~5.0..22.6 px.
_WIDE = fm.LinearScale(range=[20.0, 400.0])


def _check_or_update(name: str, svg: str) -> None:
    check_or_update_svg_golden(GOLDENS_DIR, name, svg)


def _zigzag(n: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "y": [float(i % 2) * 2.0 + i * 0.3 for i in range(n)],
            "v": [float(i + 1) for i in range(n)],
        }
    )


def _sized(df: pl.DataFrame, size: object, title: str, **mark_kwargs: object) -> fm.Chart:
    return (
        fm.Chart(df)
        .mark_line(**mark_kwargs)
        .encode(x="x", y="y", size=size)
        .properties(width=420, height=260, title=title)
    )


class TestTaperedLineGoldens:
    def test_linear_with_legend(self):
        """Default size band (widths ~2.3..6.8 px), line-swatch legend."""
        _check_or_update("linear_legend.svg", _sized(_zigzag(), "v", "Linear taper").to_svg())

    def test_curved(self):
        _check_or_update(
            "curved.svg",
            _sized(
                _zigzag(), fm.Size("v", scale=_WIDE), "Monotone", interpolate="monotone"
            ).to_svg(),
        )

    def test_step(self):
        _check_or_update(
            "step.svg",
            _sized(_zigzag(), fm.Size("v", scale=_WIDE), "Step", interpolate="step").to_svg(),
        )

    def test_dashed(self):
        _check_or_update(
            "dashed.svg",
            _sized(
                _zigzag(), fm.Size("v", scale=_WIDE), "Dashed", stroke_dash=[14.0, 6.0]
            ).to_svg(),
        )

    def test_half_opacity_hairpin(self):
        """A path that doubles back over itself paints its overlap once."""
        df = pl.DataFrame(
            {"x": [0.0, 4.0, 0.3, 3.0], "y": [0.0, 1.0, 1.3, 0.2], "v": [1.0, 2.0, 3.0, 4.0]}
        )
        chart = _sized(df, fm.Size("v", scale=_WIDE), "Hairpin at opacity 0.5", opacity=0.5)
        _check_or_update("hairpin_opacity.svg", chart.to_svg())

    def test_color_groups(self):
        """Each color group tapers along its own vertices on the shared scale."""
        base = _zigzag()
        df = pl.concat(
            [
                base.with_columns(g=pl.lit("a")),
                base.with_columns(y=pl.col("y") + 3.0, v=pl.col("v") * 0.5, g=pl.lit("b")),
            ]
        )
        chart = (
            fm.Chart(df)
            .mark_line()
            .encode(x="x", y="y", color="g", size=fm.Size("v", scale=_WIDE))
            .properties(width=420, height=260, title="Color groups")
        )
        _check_or_update("color_groups.svg", chart.to_svg())

    def test_break_axis(self):
        df = pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "y": [0.0, 100.0, 500.0, 900.0, 950.0],
                "v": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        chart = _sized(df, fm.Size("v", scale=_WIDE), "Break axis") + fm.BreakAxis(
            axis="y", gap=(150.0, 850.0)
        )
        _check_or_update("break_axis.svg", chart.to_svg())

    def test_facet(self):
        base = _zigzag()
        df = pl.concat([base.with_columns(f=pl.lit("left")), base.with_columns(f=pl.lit("right"))])
        chart = (
            fm.Chart(df)
            .mark_line()
            .encode(x="x", y="y", size=fm.Size("v", scale=_WIDE))
            .properties(width=560, height=300, title="Facets")
            .facet("f")
        )
        _check_or_update("facet.svg", chart.to_svg())
