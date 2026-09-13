"""Shared SVG `<rect>`/element parsing helpers for rendered-chart assertions.

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
background, not a gradient/pattern url()" predicate).
"""

from __future__ import annotations

import re

THEME_BACKGROUND = "#faf7f2"


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
