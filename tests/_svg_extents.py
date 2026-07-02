"""Shared SVG axis-extent parsing helpers for rendered-chart assertions.

Extracted from ``test_facet_shared_extent.py`` so sibling test modules can
assert per-panel axis domains without importing another test module's private
names (the ``tests/_snapshots.py`` precedent for cross-cutting test helpers).

The parsers recover per-panel tick extents from a rendered SVG: numeric
``<text>`` tick labels are grouped by row/column position and split into
panels, giving each panel's ``(lo, hi)`` value range. Auto-inferred scale
domains exist only Rust-side, so rendered tick extents are the observable
proxy for "these panels share (or don't share) an axis domain".
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import NamedTuple


class AxisExtent(NamedTuple):
    lo: float
    hi: float


def _numeric_text_entries(svg: str) -> list[tuple[float, float, float]]:
    """Return (x, y, value) for every ``<text>`` element whose content parses as float."""
    entries = []
    for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
        try:
            val = float(text.strip())
        except ValueError:
            continue
        x_m = re.search(r'x="([^"]+)"', attrs)
        y_m = re.search(r'y="([^"]+)"', attrs)
        if x_m and y_m:
            entries.append((float(x_m.group(1)), float(y_m.group(1)), val))
    return entries


def x_axis_extents(svg: str) -> list[AxisExtent]:
    """Per-panel x-axis tick extents for a column-faceted chart.

    Column facets are laid out side-by-side (issue #24), so every panel's
    x-axis tick row shares the same bottom y-coordinate while spanning a
    distinct x-band.  We take the bottom-most tick row (the x-axis row with the
    largest y among rows of at least 3 numeric entries) and split it into panels
    by value-reset: tick values ascend within a panel and drop at the boundary
    where the next panel's x-axis restarts from its own minimum value.

    Returns a list of (lo, hi) sorted by panel order (left panel first).
    """
    entries = _numeric_text_entries(svg)
    y_groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y, val in entries:
        y_groups[round(y)].append((x, val))
    rows = [(y, ents) for y, ents in y_groups.items() if len(ents) >= 3]
    if not rows:
        return []
    # x-axis tick rows sit at the bottom of each panel; take the lowest row.
    _, axis_row = max(rows, key=lambda r: r[0])
    return _split_axis_groups_by_position(axis_row)


def y_axis_extents(svg: str) -> list[AxisExtent]:
    """Per-panel y-axis tick extents for a column-faceted chart.

    Column facets are laid out side-by-side (issue #24), so each panel carries
    its own y-axis at a distinct x-coordinate.  We group numeric ``<text>``
    elements by their rounded x-coordinate; every column with at least 3 entries
    is one panel's y-axis.

    Returns a list of (lo, hi) sorted by panel order (left panel first).
    """
    entries = _numeric_text_entries(svg)
    x_groups: dict[int, list[float]] = defaultdict(list)
    for x, _, val in entries:
        x_groups[round(x)].append(val)
    cols = [(x, vals) for x, vals in x_groups.items() if len(vals) >= 3]
    cols.sort()
    return [AxisExtent(min(vals), max(vals)) for _, vals in cols]


def _split_axis_groups_by_position(
    positioned_vals: list[tuple[float, float]],
) -> list[AxisExtent]:
    """Split (position, value) tick entries into per-panel extents.

    Side-by-side panels (issue #24) each render the SAME shared value range, so
    the tick values ascend within a panel and then RESET (drop) at the next
    panel's leftmost tick.  We sort by *position* and break into a new group
    whenever the value decreases — the unambiguous panel boundary even when
    panels are tightly tick-packed and the x-gap between them is small.
    Returns each group's (min, max) value, left panel first.
    """
    ents = sorted(positioned_vals)
    if not ents:
        return []
    groups: list[list[float]] = [[ents[0][1]]]
    for i in range(1, len(ents)):
        if ents[i][1] < ents[i - 1][1]:  # value reset → next panel
            groups.append([ents[i][1]])
        else:
            groups[-1].append(ents[i][1])
    return [AxisExtent(min(g), max(g)) for g in groups]


def extents_all_equal(extents: list[AxisExtent]) -> bool:
    """True iff every panel's (lo, hi) is the same."""
    if len(extents) < 2:
        return True
    first = extents[0]
    return all(e == first for e in extents[1:])
