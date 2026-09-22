"""`mark_area(fill="none", borders=True)` with no color channel: the fill
clear must reach the borders too.

`area.rs`'s border-paint resolver (~line 409) falls back to
``ctx.mark_style.paint.fill_paint()`` when there is no color-channel scale
hit, so a cleared fill (``fill="none"``) is carried through to the border
paint as a clear as well -- the in-code contract there reads "a cleared fill
leaves the borders cleared too". The existing Rust pins for this resolver
(``render_svg_area_fill_clear_beats_a_color_channel`` and
``render_svg_area_fill_clear_keeps_palette_borders_under_a_color_channel``,
both in `crates/ferrum-core/src/render/mod.rs`) only exercise the
color-channel arm, where a scale hit always outranks the clear regardless of
what the fallback does -- so neither can see a broken fallback. This is the
no-color-channel arm's pin.
"""

from __future__ import annotations

import polars as pl

import ferrum as fm

from _svg_rects import mark_layer, svg_elements


def _border_stroke_colors(svg: str) -> set[str]:
    """Every ``stroke`` color painted by a mark ``<path>`` (area body or border)."""
    return {a["stroke"] for a in svg_elements(mark_layer(svg), "path", lambda a: "stroke" in a)}


def test_area_cleared_fill_clears_its_borders_with_no_color_channel():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 1.0, 5.0, 3.0]})
    chart = fm.Chart(df).mark_area(fill="none", borders=True).encode(x="x:Q", y="y:Q")

    cleared_svg = chart.to_svg()
    strokes = _border_stroke_colors(cleared_svg)
    assert strokes == set(), (
        "a cleared fill with no color channel must clear the borders too -- no area "
        f"path should carry a stroke attribute; got strokes={strokes}"
    )

    # Control: the identical shape under a color channel keeps its palette
    # border (the scale hit outranks the fill clear -- the arm the existing
    # Rust pins already cover). Kept here so the no-channel arm above is
    # contrasted against a render that DOES paint a border stroke, proving
    # the helper can see one when the resolver actually paints it.
    df_grouped = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            "y": [2.0, 4.0, 1.0, 5.0, 3.0] * 2,
            "g": ["a"] * 5 + ["b"] * 5,
        }
    )
    colored_svg = (
        fm.Chart(df_grouped)
        .mark_area(fill="none", borders=True)
        .encode(x="x:Q", y="y:Q", color="g:N")
        .to_svg()
    )
    colored_strokes = _border_stroke_colors(colored_svg)
    assert colored_strokes, (
        "a color channel must still paint its palette border over the fill clear"
    )
