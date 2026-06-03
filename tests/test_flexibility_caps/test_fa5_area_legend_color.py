"""FA-5: mark_area ordinal/quantitative color — legend swatches must match fill colors.

The T11 fix (area.rs) switched color grouping to ``col_as_ordinal_category_str``,
enabling Int*, Float*, and Bool color columns to split into separate area paths.
However the color scale was still resolved as *Continuous* for numeric dtypes
(Int64/Float64 without an explicit Ordinal type annotation), producing a gradient
colorbar legend while the fills used discrete per-group colors from the same
continuous ramp.  Legend ≠ fill: the colorbar showed a continuous gradient whereas
each area's fill color was a sampled point on that ramp.

Root cause: ``build_color_scale`` in ``scale_resolve/color.rs`` routes
``Quantitative | Temporal`` dtypes to ``ColorScale::Continuous``, which yields a
colorbar (gradient) legend.  But ``area.rs`` always groups by distinct color values
(via ``col_as_ordinal_category_str``) regardless of the numeric dtype, so the
effective color encoding is always discrete.  The legend must reflect that.

Fix: ``build_color_scale`` now forces ``Categorical`` resolution for ``mark_area``
so both fill colors and legend swatches share the same categorical palette lookup.

Test strategy:
- Render SVG for ordinal-color area and quantitative-color area.
- Extract the fill color of each closed path (area polygon) and the fill of each
  legend circle swatch.  Compare the RGB components: every group's path fill RGB
  must equal the swatch RGB (opacity/alpha differences are ignored because area
  fills carry an alpha component but legend swatches are fully opaque).
- Nominal-color area regression guard: swatch=fill still holds (pre-existing behavior).
- Visual inspection PNG saved to ``/tmp/fa5-inspect/area_ordinal_color.png``.

Color format notes:
  Ferrum area path fills use ``rgba(r,g,b,a)`` CSS syntax (with alpha for opacity).
  Legend circle swatches use ``#rrggbb`` hex.  Comparison is done on the RGB
  components only (ignoring alpha) so ``rgba(37,99,235,0.35)`` == ``#2563eb``.
"""

from __future__ import annotations

import os
import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _parse_fill_rgb(fill_str: str) -> tuple[int, int, int] | None:
    """Parse an SVG fill attribute value to (r, g, b) integers.

    Handles both ``#rrggbb`` hex and ``rgba(r, g, b, a)`` / ``rgb(r, g, b)``
    CSS syntax.  Returns None for unrecognised formats or transparent/none.
    """
    s = fill_str.strip().lower()
    if s in ("none", "transparent", "white", "#ffffff", "#faf7f2"):
        return None
    # rgba(r, g, b, a) or rgb(r, g, b)
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # #rrggbb or #rrggbbaa
    m = re.match(r"#([0-9a-f]{6})", s)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return None


def _closed_path_fill_rgbs(svg: str) -> list[tuple[int, int, int]]:
    """Return (r,g,b) tuples of area path fills (paths with ``d="M`` — data paths)."""
    paths = re.findall(r"<path\b[^>]*>", svg)
    fills = []
    for p in paths:
        if 'd="M' not in p:
            continue
        m = re.search(r'fill="([^"]+)"', p)
        if not m:
            continue
        rgb = _parse_fill_rgb(m.group(1))
        if rgb is not None:
            fills.append(rgb)
    return fills


def _legend_swatch_fill_rgbs(svg: str) -> list[tuple[int, int, int]]:
    """Return (r,g,b) tuples of legend swatch fills (``<circle>`` elements)."""
    circles = re.findall(r"<circle\b[^>]*>", svg)
    fills = []
    for c in circles:
        m = re.search(r'fill="([^"]+)"', c)
        if not m:
            continue
        rgb = _parse_fill_rgb(m.group(1))
        if rgb is not None:
            fills.append(rgb)
    return fills


def _all_text_nodes(svg: str) -> list[str]:
    """Return text content of every ``<text>`` element in the SVG."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def _has_legend_label(svg: str, label: str) -> bool:
    return label in _all_text_nodes(svg)


def _has_colorbar(svg: str) -> bool:
    """Return True if SVG contains a gradient colorbar (continuous legend)."""
    return "linearGradient" in svg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ordinal_df() -> pl.DataFrame:
    """Three groups [0, 1, 2] stored as Int64 with Ordinal annotation.

    x=0,1,2,3 repeated for each group.  y values are separated by group so
    each area is visually distinct.
    """
    xs = [0.0, 1.0, 2.0, 3.0] * 3
    ys = [
        1.0,
        2.0,
        3.0,
        2.0,  # group 0
        5.0,
        6.0,
        7.0,
        6.0,  # group 1
        9.0,
        10.0,
        11.0,
        10.0,
    ]  # group 2
    gs = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    return pl.DataFrame({"x": xs, "y": ys, "g": gs})


def _make_quantitative_df() -> pl.DataFrame:
    """Three groups [0.0, 1.0, 2.0] as Float64 with Quantitative (inferred) annotation.

    Same shape as ordinal but the color field is Float64 to exercise the Q path.
    """
    xs = [0.0, 1.0, 2.0, 3.0] * 3
    ys = [1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 6.0, 9.0, 10.0, 11.0, 10.0]
    gs = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    return pl.DataFrame({"x": xs, "y": ys, "g": gs})


def _make_nominal_df() -> pl.DataFrame:
    """Three groups ['a', 'b', 'c'] as Utf8 — the pre-T11 nominal case."""
    xs = [0.0, 1.0, 2.0, 3.0] * 3
    ys = [1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 6.0, 9.0, 10.0, 11.0, 10.0]
    gs = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    return pl.DataFrame({"x": xs, "y": ys, "g": gs})


# ---------------------------------------------------------------------------
# FA-5 ordinal-color area: swatch == fill
# ---------------------------------------------------------------------------


def test_ordinal_area_no_colorbar():
    """An ordinal-color area must NOT produce a gradient colorbar legend.

    The fix forces categorical scale resolution for mark_area, so the legend
    must show discrete swatches, not a gradient.
    """
    df = _make_ordinal_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:O").show_svg()
    assert not _has_colorbar(svg), (
        "ordinal-color mark_area must NOT produce a gradient colorbar; "
        "expected discrete legend swatches.  Got linearGradient in SVG."
    )


def test_ordinal_area_has_discrete_legend_swatches():
    """An ordinal-color area must have legend circle swatches, one per group."""
    df = _make_ordinal_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:O").show_svg()
    swatches = _legend_swatch_fill_rgbs(svg)
    assert len(swatches) >= 3, (
        f"expected >= 3 legend swatches for 3 groups; got {len(swatches)}: {swatches}"
    )


def test_ordinal_area_fill_equals_swatch():
    """Each area path fill must equal the legend swatch color for that group.

    Strategy: render a 3-group ordinal-color area, extract the 3 distinct
    closed-path fill RGB values and the 3 swatch RGB values, and assert the
    same palette is used for both.  Order may differ but the SET must be equal.

    Area path fills use rgba(r,g,b,a) CSS format (alpha for opacity).
    Legend swatches use #rrggbb hex.  Comparison ignores alpha.
    """
    df = _make_ordinal_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:O").show_svg()
    path_rgb = set(_closed_path_fill_rgbs(svg))
    swatch_rgb = set(_legend_swatch_fill_rgbs(svg))

    assert path_rgb, "expected at least one area path fill color"
    assert swatch_rgb, "expected at least one legend swatch color"

    # Every swatch RGB must appear among the path fill RGBs.
    missing_in_paths = swatch_rgb - path_rgb
    assert not missing_in_paths, (
        f"Legend swatch RGBs {missing_in_paths} not found among area path fills {path_rgb}.\n"
        "This means legend and fill use different color sources."
    )

    # Every path fill RGB must appear among the swatch RGBs.
    missing_in_swatches = path_rgb - swatch_rgb
    assert not missing_in_swatches, (
        f"Area path fill RGBs {missing_in_swatches} not found among legend swatches {swatch_rgb}.\n"
        "This means a fill color has no matching legend entry."
    )


# ---------------------------------------------------------------------------
# FA-5 quantitative-color area: swatch == fill
# ---------------------------------------------------------------------------


def test_quantitative_area_no_colorbar():
    """A quantitative-color area must NOT produce a gradient colorbar legend.

    With the fix, even an untyped or Q-typed numeric color field on mark_area
    must resolve to a categorical scale (since area always groups discretely).
    """
    df = _make_quantitative_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:Q").show_svg()
    assert not _has_colorbar(svg), (
        "quantitative-color mark_area must NOT produce a gradient colorbar; "
        "expected discrete legend swatches.  Got linearGradient in SVG."
    )


def test_quantitative_area_fill_equals_swatch():
    """Quantitative-color area: path fill RGBs == legend swatch RGBs (set equality)."""
    df = _make_quantitative_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:Q").show_svg()
    path_rgb = set(_closed_path_fill_rgbs(svg))
    swatch_rgb = set(_legend_swatch_fill_rgbs(svg))

    assert path_rgb, "expected area path fill colors"
    assert swatch_rgb, "expected legend swatch colors"

    missing_in_paths = swatch_rgb - path_rgb
    assert not missing_in_paths, (
        f"Quantitative-color area: swatch RGBs {missing_in_paths} missing from path fills {path_rgb}"
    )

    missing_in_swatches = path_rgb - swatch_rgb
    assert not missing_in_swatches, (
        f"Quantitative-color area: path fill RGBs {missing_in_swatches} missing from swatches {swatch_rgb}"
    )


# ---------------------------------------------------------------------------
# FA-5 nominal-color area: regression guard — swatch == fill still holds
# ---------------------------------------------------------------------------


def test_nominal_area_fill_equals_swatch():
    """Nominal-color area (pre-T11 working case): path fills == swatch colors.

    This test guards the pre-existing nominal path against regression.
    """
    df = _make_nominal_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:N").show_svg()
    path_rgb = set(_closed_path_fill_rgbs(svg))
    swatch_rgb = set(_legend_swatch_fill_rgbs(svg))

    assert path_rgb, "expected area path fill colors"
    assert swatch_rgb, "expected legend swatch colors"

    missing_in_paths = swatch_rgb - path_rgb
    assert not missing_in_paths, (
        f"Nominal-color area regression: swatch RGBs {missing_in_paths} not in path fills {path_rgb}"
    )

    missing_in_swatches = path_rgb - swatch_rgb
    assert not missing_in_swatches, (
        f"Nominal-color area regression: path fill RGBs {missing_in_swatches} not in swatches {swatch_rgb}"
    )


def test_nominal_area_has_no_colorbar():
    """Nominal-color area must never show a gradient colorbar."""
    df = _make_nominal_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:N").show_svg()
    assert not _has_colorbar(svg), "nominal-color mark_area must not produce a gradient colorbar"


# ---------------------------------------------------------------------------
# Visual inspection PNG
# ---------------------------------------------------------------------------


def test_save_inspection_png():
    """Render ordinal-color area to PNG and save to /tmp/fa5-inspect/ for inspection.

    The test always passes once the file is saved.  The orchestrator inspects
    the PNG to confirm swatches visually match area fills.
    """
    df = _make_ordinal_df()
    svg = fm.Chart(df).mark_area().encode(x="x:Q", y="y:Q", color="g:O").show_svg()

    out_dir = "/tmp/fa5-inspect"
    os.makedirs(out_dir, exist_ok=True)
    svg_path = os.path.join(out_dir, "area_ordinal_color.svg")
    png_path = os.path.join(out_dir, "area_ordinal_color.png")

    with open(svg_path, "w") as f:
        f.write(svg)

    # Rasterize via resvg-py (ferrum's canonical rasterizer, same as snapshot-goldens).
    try:
        import resvg_py  # type: ignore[import]

        png_bytes = resvg_py.svg_to_bytes(svg_string=svg)
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"\nInspection PNG saved: {png_path}")
    except ImportError:
        # Fall back gracefully — still save SVG.
        print(f"\nresvg_py not available; SVG saved at: {svg_path}")
        pytest.skip("resvg_py not installed — SVG saved but no PNG")

    assert os.path.exists(png_path), f"PNG not created at {png_path}"
    assert os.path.getsize(png_path) > 0, "PNG is empty"
