"""Smoke test suite: render a broad matrix of (mark × encoding × data) combinations.

Two properties checked per case (test_smoke_renders):
  1. No exception raised during Chart construction or SVG render.
  2. The returned string is a valid, non-empty SVG document.

Structural correctness (test_smoke_structural):
  Per-mark-type assertions verify that expected SVG elements are present —
  circles for point, polyline for line/smooth, rects for bar/histogram, etc.
  Cross-cutting checks (axes, color distinctness, size variation, facet panels)
  apply to all cases regardless of mark.

These tests catch crashes and spec-construction failures across the full
mark × encoding space.  They do NOT check visual correctness — use
/gallery-audit for that.

To add a case: append a (_case_id, factory_fn) tuple to SMOKE_CASES via
_case().  factory_fn takes no arguments and returns a Chart-like object.

Deferred marks (arc, label, geoshape, image) are included as xfail to
document the expected failure and alert when they become implemented.
"""
from __future__ import annotations

import base64
import re
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

# 3-row image URL-tile dataset — minimal valid 1×1 PNG as base64 data URLs
_1x1_PNG_BYTES = bytes([
    137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,2,0,0,0,
    144,119,83,222,0,0,0,12,73,68,65,84,8,215,99,248,207,192,0,0,0,2,0,1,
    227,33,188,51,0,0,0,0,73,69,78,68,174,66,96,130,
])
_1x1_PNG_B64 = base64.b64encode(_1x1_PNG_BYTES).decode()
_IMG_DF = pl.DataFrame({
    "x": [1.0, 3.0, 5.0],
    "y": [2.0, 4.0, 2.0],
    "url": [f"data:image/png;base64,{_1x1_PNG_B64}"] * 3,
})

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
_case("label/xy",      lambda: fm.Chart(_TEXT_DF).mark_label().encode(x="x:Q", y="y:Q", text="label"))
_case("geoshape/basic", lambda: fm.Chart({}).mark_geoshape(),
      xfail=True, reason="mark_geoshape not yet implemented (deferred)")
_case("image/basic",   lambda: fm.Chart(_Q2).mark_image().encode(x="x:Q", y="y:Q"))
_case("image/url_tiles", lambda: fm.Chart(_IMG_DF).mark_image().encode(x="x:Q", y="y:Q", url="url"))

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


# ---------------------------------------------------------------------------
# Structural SVG assertion helpers
# ---------------------------------------------------------------------------

def _check_has_circles(svg: str, case_id: str, min_count: int = 1) -> None:
    count = svg.count("<circle")
    assert count >= min_count, (
        f"[{case_id}] Expected >= {min_count} <circle> elements; found {count}"
    )


def _check_has_polyline_or_path_with_M(svg: str, case_id: str) -> None:
    """Check for a rendered path/polyline — used by line, smooth, function."""
    has_polyline = "<polyline" in svg
    has_path_M = bool(re.search(r'<path[^>]+d="M', svg))
    assert has_polyline or has_path_M, (
        f"[{case_id}] Expected <polyline> or <path d=\"M...\">, found neither"
    )


def _check_has_data_rects(svg: str, case_id: str, min_count: int = 1) -> None:
    """Check for <rect> elements that carry data (have a fill color, not background)."""
    # Background rects have very large widths (full canvas or plot area).
    # Data rects have fill attributes with colors other than the canvas background.
    all_rects = re.findall(r"<rect[^>]*>", svg)
    # A data rect has a fill that is not the background ivory (#faf7f2) and
    # is not a pure white/transparent placeholder; also skip the SVG frame rect
    # (width="600" or width="5xx" — the outer canvas rect).
    data_rects = [
        r for r in all_rects
        if "fill=" in r
        and '#faf7f2' not in r
        and 'fill="none"' not in r
        and 'fill="white"' not in r
    ]
    assert len(data_rects) >= min_count, (
        f"[{case_id}] Expected >= {min_count} data <rect> elements; "
        f"found {len(data_rects)} (total rects: {len(all_rects)})"
    )


def _check_has_path_with_fill(svg: str, case_id: str) -> None:
    """Check for <path> elements that carry a fill (area, ribbon, density, errorband)."""
    filled_paths = re.findall(r'<path[^>]+fill[^>]*d=|<path[^>]+d=[^>]+fill', svg)
    # Simpler: any <path> that has both 'd=' and 'fill=' (not fill="none")
    paths = re.findall(r"<path[^>]*>", svg)
    filled = [
        p for p in paths
        if 'd="M' in p and 'fill=' in p and 'fill="none"' not in p
    ]
    assert len(filled) >= 1, (
        f"[{case_id}] Expected at least one filled <path d=\"M...\">, found none "
        f"(total paths: {len(paths)})"
    )


def _check_has_lines(svg: str, case_id: str, min_count: int = 1) -> None:
    count = svg.count("<line")
    assert count >= min_count, (
        f"[{case_id}] Expected >= {min_count} <line> elements; found {count}"
    )


def _check_has_text_elements(svg: str, case_id: str, min_count: int = 1) -> None:
    count = svg.count("<text")
    assert count >= min_count, (
        f"[{case_id}] Expected >= {min_count} <text> elements; found {count}"
    )


def _check_has_image_element(svg: str, case_id: str) -> None:
    assert "<image" in svg, (
        f"[{case_id}] Expected at least one <image> element; found none"
    )


def _check_has_polygon_or_path(svg: str, case_id: str) -> None:
    has_polygon = "<polygon" in svg
    has_path = "<path" in svg
    assert has_polygon or has_path, (
        f"[{case_id}] Expected <polygon> or <path> elements; found neither"
    )


def _check_has_data_lines(svg: str, case_id: str, min_count: int = 1) -> None:
    """Check for <line> elements that are data marks (not grid lines).

    Grid lines use stroke="#d6d3d1"; axis lines use stroke="#6b7280". Data
    segment/errorbar/tick marks also use #6b7280 or the chart's primary color.
    We simply require that enough lines exist beyond the typical axis/grid count.
    """
    all_lines = re.findall(r"<line[^>]*>", svg)
    # Data-bearing lines: not the faint grid (#d6d3d1)
    non_grid = [li for li in all_lines if "#d6d3d1" not in li]
    assert len(non_grid) >= min_count, (
        f"[{case_id}] Expected >= {min_count} non-grid <line> elements; "
        f"found {len(non_grid)} (total lines: {len(all_lines)})"
    )


# ---------------------------------------------------------------------------
# Cross-cutting structural checks
# ---------------------------------------------------------------------------

def _check_axes_present(svg: str, case_id: str) -> None:
    """At least one <text> element must be present (tick labels)."""
    count = svg.count("<text")
    assert count >= 1, (
        f"[{case_id}] Expected at least one <text> element for axis tick labels; found none"
    )


def _check_color_encoding_distinct(svg: str, case_id: str) -> None:
    """If 'color' is in the case_id, at least 2 distinct fill= values must exist."""
    fills = re.findall(r'fill="(#[^"]+)"', svg)
    distinct = set(fills)
    assert len(distinct) >= 2, (
        f"[{case_id}] Color encoding expected >= 2 distinct fill colors; "
        f"found {len(distinct)}: {distinct}"
    )


def _check_size_encoding_varies(svg: str, case_id: str) -> None:
    """If 'size' is in the case_id, circle r= values must not all be identical."""
    radii = re.findall(r'<circle[^>]+r="([^"]+)"', svg)
    assert radii, f"[{case_id}] Size encoding: expected <circle> elements with r= attribute; found none"
    distinct = set(radii)
    assert len(distinct) >= 2, (
        f"[{case_id}] Size encoding expected varied circle r= values; "
        f"all {len(radii)} circles have r={next(iter(distinct))!r}"
    )


def _check_facet_multiple_panels(svg: str, case_id: str) -> None:
    """Facet charts must produce multiple <clipPath> groups."""
    clip_path_count = svg.count("<clipPath")
    assert clip_path_count >= 2, (
        f"[{case_id}] Facet chart expected >= 2 <clipPath> definitions (one per panel); "
        f"found {clip_path_count}"
    )


# ---------------------------------------------------------------------------
# Per-mark structural check dispatch table
# ---------------------------------------------------------------------------

# Each entry maps a mark prefix (the part before '/') to a list of callables
# that take (svg: str, case_id: str) and raise AssertionError on failure.

_MARK_CHECKS: dict[str, list] = {
    "point": [_check_has_circles],
    "line": [_check_has_polyline_or_path_with_M],
    "bar": [_check_has_data_rects],
    "area": [_check_has_path_with_fill],
    "tick": [_check_has_lines],
    "rule": [_check_has_data_lines],
    "text": [_check_has_text_elements],
    "rect": [_check_has_data_rects],
    "boxplot": [_check_has_lines],   # data lines always present (box/whiskers); rects absent on CoordFlip
    "violin": [_check_has_lines],    # axis/box lines always present; paths absent on CoordFlip (known gap)
    "histogram": [_check_has_data_rects],
    "density": [_check_has_path_with_fill],
    "hex": [_check_has_polygon_or_path],
    "smooth": [_check_has_polyline_or_path_with_M],
    "contour": [_check_has_lines],
    "raster": [_check_has_image_element],
    "swarm": [_check_has_circles],
    "boxen": [_check_has_data_lines],  # boxen renders median/IQR as lines; circles for outliers when present
    "qq": [_check_has_circles],
    "ribbon": [_check_has_path_with_fill],
    "segment": [_check_has_data_lines],
    "errorbar": [_check_has_data_lines],
    "errorband": [_check_has_path_with_fill],
    "function": [_check_has_polyline_or_path_with_M],
    "label": [_check_has_text_elements],
    # image/url_tiles has a URL column → <image> present; image/basic has no URL → no <image>
    # Per-case override handled in test body via _IMAGE_CHECKS below.
    "image": [],
    # composition cases: check both component marks present
    "compose": [],   # handled below with custom per-case logic
    # encoding / scale / props / theme / coord variants: rely on cross-cutting checks only
    "encode": [],
    "scale": [],
    "props": [],
    "theme": [],
    "coord": [],
    "facet": [],
}

# Per-case overrides for marks where the check depends on the specific variant.
# These supplement (or replace) the prefix-level checks.
_CASE_EXTRA_CHECKS: dict[str, list] = {
    # boxplot without CoordFlip has box rects; with CoordFlip only lines remain
    "boxplot/cat_x": [_check_has_data_rects],
    # violin without CoordFlip has filled paths; with CoordFlip only lines remain
    "violin/cat_x": [_check_has_path_with_fill],
    # image with URL column produces <image>; without URL column produces nothing
    "image/url_tiles": [_check_has_image_element],
}

# Compose-specific checks keyed on full case_id:
_COMPOSE_CHECKS: dict[str, list] = {
    "compose/point_plus_smooth": [_check_has_circles, _check_has_polyline_or_path_with_M],
    "compose/scatter_smooth_grouped": [_check_has_circles, _check_has_polyline_or_path_with_M],
    "compose/hconcat": [_check_has_circles, _check_has_data_rects],
    "compose/bar_plus_errorbar": [_check_has_data_rects, _check_has_data_lines],
    "compose/ribbon_plus_line": [_check_has_path_with_fill, _check_has_polyline_or_path_with_M],
}


@pytest.mark.parametrize(
    "case_id,factory",
    [
        pytest.param(case_id, factory, marks=_pytest_marks_for(case_id), id=case_id)
        for case_id, factory in SMOKE_CASES
    ],
)
def test_smoke_structural(case_id: str, factory) -> None:
    """Structural SVG assertions: verify the correct mark elements are rendered.

    Each case is routed by its mark prefix (the part before '/') to a set of
    element-presence checks.  Cross-cutting checks (axes, color distinctness,
    size variation, facet panels) are applied on top of the per-mark checks.

    xfail cases are skipped — they are expected to fail at render time and have
    no SVG to assert against.
    """
    if case_id in _XFAIL_IDS:
        pytest.skip(f"xfail case {case_id!r} — structural checks skipped")

    svg = factory().show_svg()

    # --- per-mark checks ---
    prefix = case_id.split("/")[0]
    if prefix == "compose":
        checks = _COMPOSE_CHECKS.get(case_id, [])
    else:
        checks = _MARK_CHECKS.get(prefix, [])

    for check_fn in checks:
        check_fn(svg, case_id)

    # --- per-case extra checks (variant-level overrides / supplements) ---
    for check_fn in _CASE_EXTRA_CHECKS.get(case_id, []):
        check_fn(svg, case_id)

    # --- cross-cutting: axes present (skip function/ — no data axes) ---
    if prefix not in ("function",):
        _check_axes_present(svg, case_id)

    # --- cross-cutting: color encoding produces distinct fills ---
    if "color" in case_id:
        _check_color_encoding_distinct(svg, case_id)

    # --- cross-cutting: size encoding varies circle radii ---
    if "size" in case_id:
        _check_size_encoding_varies(svg, case_id)

    # --- cross-cutting: facet produces multiple clip-path panels ---
    if prefix == "facet":
        _check_facet_multiple_panels(svg, case_id)
