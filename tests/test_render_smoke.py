"""Smoke test suite: render a broad matrix of (mark × encoding × data) combinations.

Two properties checked per case:
  1. No exception raised during Chart construction or SVG render.
  2. The returned string is a valid, non-empty SVG document.

These tests catch crashes and spec-construction failures across the full
mark × encoding space.  They do NOT check visual correctness — use
/gallery-audit for that.

To add a case: append a (_case_id, factory_fn) tuple to SMOKE_CASES via
_case().  factory_fn takes no arguments and returns a Chart-like object.

Deferred marks (arc, label, geoshape, image) are included as xfail to
document the expected failure and alert when they become implemented.
"""
from __future__ import annotations

import pytest
import polars as pl
import ferrum as fm
from ferrum.encoding import X, Y, Y2, X2, Color, Size, Shape, Text

# ---------------------------------------------------------------------------
# Shared synthetic datasets — deterministic, no numpy/sklearn required
# ---------------------------------------------------------------------------

# 10-row bivariate quantitative
_Q2 = pl.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "y": [2.1, 3.9, 1.2, 5.0, 3.3, 6.8, 2.4, 5.7, 4.1, 7.9],
})

# 10-row trivariate quantitative
_Q3 = _Q2.with_columns(pl.Series("z", [3.0, 1.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 2.0]))

# 12-row nominal + quantitative (3 categories × 4 observations each)
_NQ = pl.DataFrame({
    "cat": ["a", "b", "c"] * 4,
    "val": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 2.2, 3.1, 1.8, 2.8, 3.8],
})

# 10-row bivariate + nominal group
_Q2N = _Q2.with_columns(pl.Series("grp", ["X", "Y"] * 5))

# 5-row with text labels
_TEXT_DF = pl.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [2.0, 4.0, 1.0, 5.0, 3.0],
    "label": ["alpha", "beta", "gamma", "delta", "epsilon"],
})

# 5-row ribbon band (lower / upper)
_BAND = pl.DataFrame({
    "x":  [1.0, 2.0, 3.0, 4.0, 5.0],
    "lo": [0.5, 1.5, 0.5, 4.0, 2.5],
    "hi": [1.5, 2.5, 1.5, 6.0, 3.5],
})

# 3-row segment (x, y) → (x2, y2)
_SEG = pl.DataFrame({
    "x":  [1.0, 2.0, 3.0],
    "y":  [1.0, 2.0, 3.0],
    "x2": [1.5, 2.5, 3.5],
    "y2": [2.0, 3.0, 4.0],
})

# 6-row grid for rect heatmap
_GRID = pl.DataFrame({
    "xcat": ["A", "A", "B", "B", "C", "C"],
    "ycat": ["X", "Y", "X", "Y", "X", "Y"],
    "val":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
})

# 1-row edge case
_SINGLE = _Q2[:1]

# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

SMOKE_CASES: list[tuple[str, object]] = []
_XFAIL_IDS: set[str] = set()


def _case(case_id: str, factory, *, xfail: bool = False, reason: str = "") -> None:
    SMOKE_CASES.append((case_id, factory))
    if xfail:
        _XFAIL_IDS.add(case_id)


# ---------------------------------------------------------------------------
# mark_point
# ---------------------------------------------------------------------------
_case("point/xy_q",          lambda: fm.Chart(_Q2).mark_point().encode(x="x:Q", y="y:Q"))
_case("point/xy_color_n",    lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", color="grp:N"))
_case("point/xy_size_q",     lambda: fm.Chart(_Q3).mark_point().encode(x="x:Q", y="y:Q", size="z:Q"))
_case("point/xy_shape_n",    lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", shape="grp:N"))
_case("point/xy_opacity_q",  lambda: fm.Chart(_Q2).mark_point().encode(x="x:Q", y="y:Q", opacity="x:Q"))
_case("point/xy_color_q",    lambda: fm.Chart(_Q3).mark_point().encode(x="x:Q", y="y:Q", color="z:Q"))
_case("point/single_row",    lambda: fm.Chart(_SINGLE).mark_point().encode(x="x:Q", y="y:Q"))

# ---------------------------------------------------------------------------
# mark_line
# ---------------------------------------------------------------------------
_case("line/xy_q",           lambda: fm.Chart(_Q2).mark_line().encode(x="x:Q", y="y:Q"))
_case("line/xy_color_n",     lambda: fm.Chart(_Q2N).mark_line().encode(x="x:Q", y="y:Q", color="grp:N"))

# ---------------------------------------------------------------------------
# mark_bar
# ---------------------------------------------------------------------------
_case("bar/cat_x",           lambda: fm.Chart(_NQ).mark_bar().encode(x="cat:N", y="val:Q"))
_case("bar/horiz_coord_flip", lambda: fm.Chart(_NQ).mark_bar().encode(x="cat:N", y="val:Q").coord(fm.CoordFlip()))
_case("bar/cat_x_color_n",   lambda: fm.Chart(_NQ).mark_bar().encode(x="cat:N", y="val:Q", color="cat:N"))

# ---------------------------------------------------------------------------
# mark_area
# ---------------------------------------------------------------------------
_case("area/xy",             lambda: fm.Chart(_Q2).mark_area().encode(x="x:Q", y="y:Q"))
_case("area/xy_color_n",     lambda: fm.Chart(_Q2N).mark_area().encode(x="x:Q", y="y:Q", color="grp:N"))

# ---------------------------------------------------------------------------
# mark_tick — all four modes; x-rug + y-rug are regression guards
# ---------------------------------------------------------------------------
_case("tick/x_rug",              lambda: fm.Chart(_Q2).mark_tick().encode(x="x:Q"))
_case("tick/y_rug",              lambda: fm.Chart(_Q2).mark_tick().encode(y="y:Q"))
_case("tick/strip_ordinal_y",    lambda: fm.Chart(_NQ).mark_tick().encode(x="val:Q", y="cat:N"))
_case("tick/strip_ordinal_x",    lambda: fm.Chart(_NQ).mark_tick().encode(x="cat:N", y="val:Q"))

# ---------------------------------------------------------------------------
# mark_rule
# ---------------------------------------------------------------------------
_case("rule/y_only",         lambda: fm.Chart(_Q2).mark_rule().encode(y="y:Q"))
_case("rule/x_only",         lambda: fm.Chart(_Q2).mark_rule().encode(x="x:Q"))

# ---------------------------------------------------------------------------
# mark_text
# ---------------------------------------------------------------------------
_case("text/xy_text",        lambda: fm.Chart(_TEXT_DF).mark_text().encode(x="x:Q", y="y:Q", text="label"))

# ---------------------------------------------------------------------------
# mark_rect
# ---------------------------------------------------------------------------
_case("rect/heatmap",        lambda: fm.Chart(_GRID).mark_rect().encode(x="xcat:N", y="ycat:N", color="val:Q"))

# ---------------------------------------------------------------------------
# mark_boxplot
# ---------------------------------------------------------------------------
_case("boxplot/cat_x",       lambda: fm.Chart(_NQ).mark_boxplot().encode(x="cat:N", y="val:Q"))
_case("boxplot/horiz",       lambda: fm.Chart(_NQ).mark_boxplot().encode(x="cat:N", y="val:Q").coord(fm.CoordFlip()))

# ---------------------------------------------------------------------------
# mark_violin
# ---------------------------------------------------------------------------
_case("violin/cat_x",        lambda: fm.Chart(_NQ).mark_violin().encode(x="cat:N", y="val:Q"))
_case("violin/horiz",        lambda: fm.Chart(_NQ).mark_violin().encode(x="cat:N", y="val:Q").coord(fm.CoordFlip()))

# ---------------------------------------------------------------------------
# mark_histogram
# ---------------------------------------------------------------------------
_case("histogram/basic",     lambda: fm.Chart(_Q2).mark_histogram().encode(x="x", y="count"))
_case("histogram/grouped",   lambda: fm.Chart(_Q2N).mark_histogram(groupby="grp").encode(x="x", y="count", color="grp:N"))

# ---------------------------------------------------------------------------
# mark_density
# ---------------------------------------------------------------------------
_case("density/basic",       lambda: fm.Chart(_Q2).mark_density().encode(x="x"))
_case("density/grouped",     lambda: fm.Chart(_Q2N).mark_density(groupby="grp").encode(x="x", color="grp:N"))

# ---------------------------------------------------------------------------
# mark_hex
# ---------------------------------------------------------------------------
_case("hex/xy",              lambda: fm.Chart(_Q2).mark_hex().encode(x="x:Q", y="y:Q"))

# ---------------------------------------------------------------------------
# mark_smooth
# ---------------------------------------------------------------------------
_case("smooth/loess",        lambda: fm.Chart(_Q2).mark_smooth().encode(x="x:Q", y="y:Q"))
_case("smooth/lm",           lambda: fm.Chart(_Q2).mark_smooth(method="lm").encode(x="x:Q", y="y:Q"))
_case("smooth/ci_0.95",      lambda: fm.Chart(_Q2).mark_smooth(ci=0.95).encode(x="x:Q", y="y:Q"))
_case("smooth/grouped",      lambda: fm.Chart(_Q2N).mark_smooth(method="lm", groupby="grp").encode(x="x:Q", y="y:Q", color="grp:N"))

# ---------------------------------------------------------------------------
# mark_contour
# ---------------------------------------------------------------------------
_case("contour/xy",          lambda: fm.Chart(_Q2).mark_contour().encode(x="x:Q", y="y:Q"))

# ---------------------------------------------------------------------------
# mark_raster
# ---------------------------------------------------------------------------
_case("raster/xy",           lambda: fm.Chart(_Q2).mark_raster().encode(x="x:Q", y="y:Q"))

# ---------------------------------------------------------------------------
# mark_swarm
# ---------------------------------------------------------------------------
_case("swarm/cat_x",         lambda: fm.Chart(_NQ).mark_swarm().encode(x="cat:N", y="val:Q"))

# ---------------------------------------------------------------------------
# mark_boxen
# ---------------------------------------------------------------------------
_case("boxen/cat_x",         lambda: fm.Chart(_NQ).mark_boxen().encode(x="cat:N", y="val:Q"))

# ---------------------------------------------------------------------------
# mark_qq
# ---------------------------------------------------------------------------
_case("qq/x_q",              lambda: fm.Chart(_Q2).mark_qq().encode(x="x:Q"))

# ---------------------------------------------------------------------------
# mark_ribbon
# ---------------------------------------------------------------------------
_case("ribbon/lo_hi",        lambda: fm.Chart(_BAND).mark_ribbon().encode(x="x:Q", y="lo:Q", y2="hi:Q"))

# ---------------------------------------------------------------------------
# mark_segment
# ---------------------------------------------------------------------------
_case("segment/xy_x2y2",     lambda: fm.Chart(_SEG).mark_segment().encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q"))

# ---------------------------------------------------------------------------
# mark_errorbar
# ---------------------------------------------------------------------------
_case("errorbar/stdev",      lambda: fm.Chart(_NQ).mark_errorbar(extent="stdev").encode(x="cat:N", y="val:Q"))
_case("errorbar/ci",         lambda: fm.Chart(_NQ).mark_errorbar(extent="ci").encode(x="cat:N", y="val:Q"))

# ---------------------------------------------------------------------------
# mark_errorband
# ---------------------------------------------------------------------------
_case("errorband/ci",        lambda: fm.Chart(_NQ).mark_errorband(extent="ci").encode(x="cat:N", y="val:Q"))

# ---------------------------------------------------------------------------
# mark_function  (no data or encodings required)
# ---------------------------------------------------------------------------
_case("function/parabola",   lambda: fm.Chart(None).mark_function(lambda x: x ** 2, domain=[0, 5], n=50))

# ---------------------------------------------------------------------------
# Deferred marks — expected failures; become passing when implemented
# ---------------------------------------------------------------------------
_case("arc/theta",     lambda: fm.Chart(_NQ).mark_arc().encode(x="cat:N"),
      xfail=True, reason="mark_arc not yet implemented (deferred)")
_case("label/xy",      lambda: fm.Chart(_TEXT_DF).mark_label().encode(x="x:Q", y="y:Q", text="label"),
      xfail=True, reason="mark_label not yet implemented (deferred)")
_case("geoshape/basic", lambda: fm.Chart({}).mark_geoshape(),
      xfail=True, reason="mark_geoshape not yet implemented (deferred)")
_case("image/basic",   lambda: fm.Chart(_Q2).mark_image().encode(x="x:Q", y="y:Q"),
      xfail=True, reason="mark_image not yet implemented (deferred)")

# ---------------------------------------------------------------------------
# Facet — regression for encode(facet_col/row=...) being silently dropped
# ---------------------------------------------------------------------------
_case("facet/encode_facet_col",  lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", facet_col="grp:N"))
_case("facet/encode_facet_row",  lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", facet_row="grp:N"))
_case("facet/encode_facet_bare", lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", facet="grp:N"))
_case("facet/method_col",        lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q").facet(col="grp"))
_case("facet/method_field",      lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q").facet(field="grp", ncols=2))

# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------
_case("compose/point_plus_smooth",     lambda: (
    fm.Chart(_Q2).mark_point(opacity=0.3).mark_smooth().encode(x="x:Q", y="y:Q")
))
_case("compose/scatter_smooth_grouped", lambda: (
    fm.Chart(_Q2N).mark_point(opacity=0.3).mark_smooth(method="lm", groupby="grp")
    .encode(x="x:Q", y="y:Q", color="grp:N")
))
_case("compose/hconcat",               lambda: (
    fm.hconcat(
        fm.Chart(_Q2).mark_point().encode(x="x:Q", y="y:Q").properties(width=250, height=250),
        fm.Chart(_NQ).mark_bar().encode(x="cat:N", y="val:Q").properties(width=250, height=250),
    )
))
_case("compose/bar_plus_errorbar",     lambda: (
    fm.Chart(_NQ).mark_bar().encode(x="cat:N", y="val:Q")
    + fm.Chart(_NQ).mark_errorbar(extent="stdev").encode(x="cat:N", y="val:Q")
))
_case("compose/ribbon_plus_line",      lambda: (
    fm.Chart(_BAND).mark_ribbon().encode(x="x:Q", y="lo:Q", y2="hi:Q")
    + fm.Chart(_BAND).mark_line().encode(x="x:Q", y="hi:Q")
))

# ---------------------------------------------------------------------------
# Encoding type variants
# ---------------------------------------------------------------------------
_case("encode/nominal_color",    lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", color="grp:N"))
_case("encode/ordinal_color",    lambda: fm.Chart(_Q2N).mark_point().encode(x="x:Q", y="y:Q", color="grp:O"))
_case("encode/quant_color",      lambda: fm.Chart(_Q3).mark_point().encode(x="x:Q", y="y:Q", color="z:Q"))
_case("encode/channel_objects",  lambda: fm.Chart(_Q3).mark_point().encode(
    x=X("x", type="Q"), y=Y("y", type="Q"), color=Color("z", type="Q"), size=Size("z", type="Q")
))

# ---------------------------------------------------------------------------
# Scale overrides
# ---------------------------------------------------------------------------
_case("scale/log_x",           lambda: fm.Chart(_Q2).mark_point().encode(
    x=X("x:Q", scale={"type": "log"}), y="y:Q"
))
_case("scale/zero_false",      lambda: fm.Chart(_Q2).mark_point().encode(
    x=X("x:Q", scale={"zero": False}), y="y:Q"
))

# ---------------------------------------------------------------------------
# Properties and themes
# ---------------------------------------------------------------------------
_case("props/width_height_title", lambda: (
    fm.Chart(_Q2).mark_point().encode(x="x:Q", y="y:Q")
    .properties(width=300, height=200, title="Smoke test")
))
_case("theme/paper_ink",          lambda: (
    fm.Chart(_Q2).mark_point().encode(x="x:Q", y="y:Q").theme(fm.themes.paper_ink)
))

# ---------------------------------------------------------------------------
# Coord
# ---------------------------------------------------------------------------
_case("coord/flip_bar",     lambda: fm.Chart(_NQ).mark_bar().encode(x="cat:N", y="val:Q").coord(fm.CoordFlip()))
_case("coord/flip_boxplot", lambda: fm.Chart(_NQ).mark_boxplot().encode(x="cat:N", y="val:Q").coord(fm.CoordFlip()))

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

_CASE_IDS = [c[0] for c in SMOKE_CASES]


def _pytest_marks_for(case_id: str):
    if case_id in _XFAIL_IDS:
        return [pytest.mark.xfail(strict=False, reason="deferred mark (not yet implemented)")]
    return []


@pytest.mark.parametrize(
    "case_id,factory",
    [
        pytest.param(case_id, factory, marks=_pytest_marks_for(case_id), id=case_id)
        for case_id, factory in SMOKE_CASES
    ],
)
def test_smoke_renders(case_id: str, factory) -> None:
    """Every case must render to a valid SVG without raising any exception.

    Catches: ValueError, TypeError, RenderError, spec-construction failures,
    and any other exception that surfaces during chart build or SVG render.
    """
    svg = factory().show_svg()
    assert "<svg" in svg, f"[{case_id}] expected a valid SVG document; got {svg[:120]!r}"
    # A non-empty chart always exceeds 500 characters (axes + at least one mark element).
    assert len(svg) > 500, f"[{case_id}] SVG suspiciously small ({len(svg)} chars) — data marks may not be rendering"
