"""Shared SVG element parsing helpers for rendered-chart assertions.

The same regex-based element parser had been copied into
`test_bug_hunt_band_point_range.py`, `test_regression_band_point_range.py`, and
`test_bug_hunt_stack_value_axis.py`. This module is where new sibling sites bind
instead of adding another copy -- it stopped the fourth
(`test_displot_kde_overlay.py`), not the first three: **those three still carry
their own copies, and migrating them is a filed follow-up, not something this
module did.** Read this as the new home, not as a completed consolidation.
Mirrors the `tests/_svg_extents.py` precedent for cross-cutting test helpers:
one module owns the parsing regex, sibling test modules import it.

`svg_elements` is the base parser: it owns the `<tag ...attrs.../>` regex for
every module that imports it. `data_bar_rects` filters that down to the
theme's data-fill rects (the common case each caller of `svg_elements(svg,
"rect", ...)` would otherwise spell itself, with the same "not the panel
background, not a gradient/pattern url()" predicate). `polylines` and
`rug_tick_x1_by_stroke` do the same for the two non-rect shapes the displot
family asserts on; they were the duplication `test_displot_ecdf_hue.py`
recorded as pending when it was written, and the copy it carried read
attributes positionally (`<line x1="..." ... stroke="...">`), so it would
have started failing on any attribute-order change the renderer made. Both
now go through `svg_elements`, which reads attributes by name.

`plot_rect` is the fifth: `test_bug_hunt_stack_value_axis.py` reaches the plot
area as `_svg_rects(svg)[1]["x"]` -- by POSITION in the document, which is
silently wrong the moment a chart emits a rect before the clip rect -- and
`test_area_ribbon_baseline_axis.py` needed the same rectangle to assert which
panel edge an area fill closes on. It lives here rather than in either caller
because "the plot area is the one unfilled rect" is a fact about ferrum's SVG
output, not about any one test's subject.

`mark_layer` is the sixth: `tests/_category_dtype_matrix.py` and
`tests/test_single_axis_eligibility.py` each carried an identical private
copy of the plot-area clip-group regex to scope a query to drawn marks and
exclude chrome. It lives here for the same reason `svg_elements` does --
scoping to "the marks, and nothing else" is a fact about ferrum's SVG
structure, not about either module's subject -- and both import it rather
than keep their own copy.
"""

from __future__ import annotations

import re

THEME_BACKGROUND = "#faf7f2"

#: Theme chrome stroke colors (axis line, gridline). Excluded when a test is
#: after the marks a chart drew rather than the frame it drew them in.
THEME_CHROME_STROKES = frozenset({"#6b7280", "#d6d3d1"})


def svg_elements(svg: str, tag: str, keep) -> list[dict[str, str]]:
    """Return the attribute dict of every self-closed `<tag .../>` for which `keep(attrs)` is true."""
    out = []
    for m in re.finditer(rf"<{tag}([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        if keep(attrs):
            out.append(attrs)
    return out


def data_bar_rects(svg: str) -> list[dict[str, str]]:
    """Return every rendered data `<rect>` (bars, cells, ...), excluding the panel background fill and gradient/pattern fills."""
    return svg_elements(
        svg,
        "rect",
        lambda a: (
            bool(a.get("fill"))
            and a["fill"] != THEME_BACKGROUND
            and not a["fill"].startswith("url(")
        ),
    )


#: The plot-area clip group: every mark a chart draws is emitted inside one;
#: no chrome (axes, gridlines, legend swatches) is. Non-greedy, and every
#: caller below uses it only on single-panel charts, where the group does
#: not nest.
_MARK_GROUP_RE = re.compile(r'<g clip-path="url\(#ferrum-clip-\d+\)">(.*?)</g>', re.DOTALL)


def mark_layer(svg: str) -> str:
    """Return the concatenation of every plot-area clip group's contents.

    The marks, and nothing else. This regex was carried as an identical
    private copy in both `tests/_category_dtype_matrix.py` and
    `tests/test_single_axis_eligibility.py`; this is the one copy both
    import. Returns `""` for a chart that drew no marks at all, which is
    itself a meaningful (and asserted-on) outcome for some callers.
    """
    return "".join(_MARK_GROUP_RE.findall(svg))


def polylines(svg: str) -> list[tuple[str | None, list[tuple[float, float]]]]:
    """Return `(stroke, points)` for every `<polyline>` in *svg*.

    `stroke` is `None` for a polyline that carries no stroke attribute, so a
    caller filtering for colored curves states that filter itself rather than
    relying on the parser to have dropped them.
    """
    out: list[tuple[str | None, list[tuple[float, float]]]] = []
    for attrs in svg_elements(svg, "polyline", lambda a: "points" in a):
        points = [(float(p.split(",")[0]), float(p.split(",")[1])) for p in attrs["points"].split()]
        out.append((attrs.get("stroke"), points))
    return out


def rug_tick_x1_by_stroke(svg: str) -> dict[str, set[float]]:
    """Map each rug tick `<line>`'s `stroke` color to its set of `x1` positions.

    Excludes `THEME_CHROME_STROKES` so only the per-group tick lines remain --
    this lets a test compare "this group's rug ticks" between two renders
    without depending on emission order or on the theme's own frame lines. A
    `<line>` with no `stroke` attribute at all is also excluded (mirroring
    `polylines`' explicit stance on the same question): the return type is
    keyed by stroke color, so a strokeless line has no key to live under, and
    a `None` key would collapse every unstyled line into one bucket instead.
    """
    out: dict[str, set[float]] = {}
    for attrs in svg_elements(
        svg,
        "line",
        lambda a: a.get("stroke") is not None and a["stroke"] not in THEME_CHROME_STROKES,
    ):
        out.setdefault(attrs["stroke"], set()).add(round(float(attrs["x1"]), 3))
    return out


def fill_rects(svg: str, fill: str = "#2563eb") -> list[tuple[float, float, float, float]]:
    """Return `(x, y, w, h)` for every `<rect>` painted in `fill`.

    This was triplicated (identically) across `test_break_axis_geometry.py`,
    `test_break_axis_facet.py`, and `test_break_axis_coordflip.py` as `_rects`
    / `_bars`, each wrapping `svg_elements` with the same fill-equality
    predicate. Unlike `data_bar_rects`, which merges every non-background data
    fill into one list, this keeps rects of a single, caller-chosen fill
    separable from siblings painted a different color (e.g. mark rects vs.
    grid-band rects in the same chart).
    """
    return [
        (float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
        for a in svg_elements(svg, "rect", lambda a: a.get("fill") == fill)
    ]


def plot_rect(svg: str) -> tuple[float, float, float, float]:
    """Return `(x, y, w, h)` of a single-panel chart's plot area.

    Identified by the one structural property that holds whatever the chart
    draws: the plot area is the only `<rect>` a chart emits with NO `fill`
    attribute (it exists to define the clip path; the page background carries
    the theme fill, and every data rect carries a mark fill). Callers asserting
    "this mark sits on the panel's left/bottom edge" need this rectangle, and
    reaching it by document position -- as `test_bug_hunt_stack_value_axis.py`
    still does -- breaks silently the moment a chart emits an unfilled rect
    earlier.

    Raises `AssertionError` for a chart with zero or several panels: the return
    type is one rectangle, so a faceted or composed SVG has no single answer
    and must not be handed the first one found.
    """
    rects = svg_elements(svg, "rect", lambda a: "fill" not in a)
    assert len(rects) == 1, (
        f"expected exactly one unfilled plot rect (a single-panel chart); got {len(rects)}"
    )
    a = rects[0]
    return (float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
