"""Shared SVG axis-extent parsing helpers for rendered-chart assertions.

Extracted from ``test_facet_shared_extent.py`` so sibling test modules can
assert per-panel axis domains without importing another test module's private
names (the ``tests/_snapshots.py`` precedent for cross-cutting test helpers).

The parsers recover per-panel tick extents from a rendered SVG: numeric
``<text>`` tick labels are grouped by row/column position and split into
panels, giving each panel's ``(lo, hi)`` value range. Auto-inferred scale
domains exist only Rust-side, so rendered tick extents are the observable
proxy for "these panels share (or don't share) an axis domain".

``ordered_text_entries`` is the base parser: it owns the
``<text x y>value</text>`` regex once so no sibling module re-copies it.
``numeric_text_entries`` filters that down to float-parseable ticks (the
multi-panel extent helpers below only ever compare numeric domains).
``axis_tick_labels`` is the single-panel, raw-string counterpart — it keeps
non-numeric tick text (e.g. ``TimeScale``'s ``"HH:MM:SS"`` formatting) and
preserves screen-order, which the extent helpers below deliberately discard
in favor of per-panel ``(min, max)``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import NamedTuple


class AxisExtent(NamedTuple):
    lo: float
    hi: float


def ordered_text_entries(svg: str) -> list[tuple[float, float, str]]:
    """Return ``(x, y, text)`` for every non-empty ``<text>`` element, in document order.

    This is the single text-node parser shared by every helper in this
    module and by sibling render tests (e.g. ``test_composite_render_grid.py``
    via ``numeric_text_entries``, ``test_scale_reverse.py`` via
    ``axis_tick_labels``): the ``<text x y>value</text>`` regex lives here
    once so new tests never re-copy it. Unlike ``numeric_text_entries``, this
    keeps the raw label string — callers that need float values (and don't
    care about non-numeric labels) should use ``numeric_text_entries``.
    """
    entries: list[tuple[float, float, str]] = []
    for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
        text = text.strip()
        if not text:
            continue
        x_m = re.search(r'x="([^"]+)"', attrs)
        y_m = re.search(r'y="([^"]+)"', attrs)
        if x_m and y_m:
            entries.append((float(x_m.group(1)), float(y_m.group(1)), text))
    return entries


def text_runs(svg: str) -> list[str]:
    """Return every rendered ``<text>`` run, in document order, positions dropped.

    The position-free sibling of ``ordered_text_entries``: it keeps runs that
    carry no ``x``/``y`` attributes (a ``<text>`` positioned by its parent
    group, e.g. a chart title), which is why an assertion about *what a chart
    says* uses this and an assertion about *where it says it* uses the entry
    parsers above. Shared by the metric-display tests (#219) so "one metric
    string per curve" is counted the same way everywhere.
    """
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


def numeric_text_entries(svg: str) -> list[tuple[float, float, float]]:
    """Return (x, y, value) for every ``<text>`` element whose content parses as float."""
    entries = []
    for x, y, text in ordered_text_entries(svg):
        try:
            val = float(text)
        except ValueError:
            continue
        entries.append((x, y, val))
    return entries


def axis_tick_labels(svg: str, *, axis: str) -> list[str]:
    """Return raw tick-label text for one axis of a single-panel SVG, in screen order.

    x-axis ticks share the largest rounded ``y`` (the bottom row); returned
    left-to-right (ascending screen ``x``). y-axis ticks share the smallest
    rounded ``x`` (the left column); returned top-to-bottom (ascending
    screen ``y``). Groups smaller than 3 entries are axis/legend titles, not
    the tick row/column, and are excluded.

    This is the single-panel, raw-string sibling of ``x_axis_extents``/
    ``y_axis_extents`` below: those assume multiple side-by-side panels and
    split a tick row/column by value-reset into per-panel ``(min, max)``,
    discarding both non-numeric text and screen order. Use this instead when
    the assertion cares about the exact rendered order (e.g. proving a
    ``reverse=True`` scale's ticks descend) or when labels aren't
    float-parseable (e.g. ``TimeScale``'s ``"HH:MM:SS"`` formatting).
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    entries = ordered_text_entries(svg)

    if axis == "x":
        rows: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for x, y, text in entries:
            rows[round(y)].append((x, text))
        candidates = [(y, ents) for y, ents in rows.items() if len(ents) >= 3]
        if not candidates:
            return []
        _, ents = max(candidates, key=lambda item: item[0])
        return [text for _, text in sorted(ents)]

    cols: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for x, y, text in entries:
        cols[round(x)].append((y, text))
    candidates = [(x, ents) for x, ents in cols.items() if len(ents) >= 3]
    if not candidates:
        return []
    _, ents = min(candidates, key=lambda item: item[0])
    return [text for _, text in sorted(ents)]


def y_axis_units_per_pixel(svg: str) -> float:
    """Data units per vertical pixel, read off a single-panel chart's y ticks.

    Rendered geometry is in pixels, so no ABSOLUTE claim about a distance in
    it ("this interval is 0.28 data units wide", "these points are spread like
    the data") can be made until the axis says what a pixel is worth. The
    scale comes from the axis's own rendered tick ladder, so it is read off
    the same artifact the assertion is about.

    A RATIO of pixel distances, not a position: tick labels are placed by
    their text baseline, a few pixels off the tick they name, so the ladder
    fixes the axis's scale exactly but its offset only approximately. Callers
    convert LENGTHS (a whisker's height, a point's distance from another
    point), never a lone coordinate.

    The ladder is the leftmost column of at least three numeric tick labels --
    the same rule :func:`axis_tick_labels` uses for ``axis="y"``. Raises when
    there is none: a chart with no numeric y ticks cannot answer the question.
    """
    columns: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y, value in numeric_text_entries(svg):
        columns[round(x)].append((y, value))
    for x_key in sorted(columns):
        ladder = sorted(columns[x_key])
        if len(ladder) >= 3:
            break
    else:
        raise AssertionError("no y-axis tick ladder in the rendered SVG")
    (pixel_top, value_top), (pixel_bottom, value_bottom) = ladder[0], ladder[-1]
    return abs((value_bottom - value_top) / (pixel_bottom - pixel_top))


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
    entries = numeric_text_entries(svg)
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
    entries = numeric_text_entries(svg)
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


# --------------------------------------------------------------------------
# Tick dash <-> label pairing (wave-2 break-axis D2, #231)
# --------------------------------------------------------------------------


class AxisTickDashLabels(NamedTuple):
    """One axis tick's dash `<line>`, paired with the label `<text>` node(s)
    (`(x, y, text)`) drawn for that same tick, in document order."""

    dash: tuple[float, float, float, float]
    labels: list[tuple[float, float, str]]


def _ordered_lines_and_labels(svg: str) -> list[tuple[str, tuple]]:
    """``(kind, coords)`` for every ``<line>`` or non-empty ``<text>`` element,
    in document order — ``kind`` is ``"line"`` (``coords`` = ``(x1, y1, x2,
    y2)``) or ``"text"`` (``coords`` = ``(x, y, text)``).

    A single combined scan, unlike :func:`ordered_text_entries` (which only
    sees ``<text>``): pairing a tick's dash with its label needs their
    RELATIVE document order, and ``marks/axis.rs``'s ``build_axis`` emits a
    tick's dash immediately before that tick's label(s) — see
    :func:`axis_tick_dash_label_pairs`, the one caller that needs this.
    """
    entries: list[tuple[str, tuple]] = []
    for m in re.finditer(r"<line\s+([^>]*)/?>|<text\s+([^>]*)>([^<]*)</text>", svg):
        if m.group(1) is not None:
            attrs = m.group(1)
            x1 = float(re.search(r'x1="([^"]+)"', attrs).group(1))
            y1 = float(re.search(r'y1="([^"]+)"', attrs).group(1))
            x2 = float(re.search(r'x2="([^"]+)"', attrs).group(1))
            y2 = float(re.search(r'y2="([^"]+)"', attrs).group(1))
            entries.append(("line", (x1, y1, x2, y2)))
        else:
            text = m.group(3).strip()
            if not text:
                continue
            attrs = m.group(2)
            x = float(re.search(r'x="([^"]+)"', attrs).group(1))
            y = float(re.search(r'y="([^"]+)"', attrs).group(1))
            entries.append(("text", (x, y, text)))
    return entries


def axis_tick_dash_label_pairs(
    svg: str, *, axis: str, dash_max_length: float = 20.0
) -> list[AxisTickDashLabels]:
    """Pair each ``axis``'s tick dash with the label(s) drawn for that tick.

    ``build_axis`` (``crates/ferrum-core/src/render/marks/axis.rs``) emits
    one tick's dash ``<line>`` immediately before that tick's label
    ``<text>`` node(s) (one, or several for a multi-line label), and nothing
    else interleaves between one tick's own chrome and the next's. This
    walks the SVG once, in document order, and groups each qualifying dash
    with the label(s) that immediately follow it — up to the next ``<line>``
    element, which always ends a group. That means the axis's own domain
    line, its gridlines, and the OTHER axis's chrome (a foreign dash, or that
    axis's own domain line) can never leak into a group: each is itself a
    ``<line>`` and so closes whatever group came before it, contributing no
    labels of its own.

    A dash "qualifies" for ``axis`` when it is aligned to that axis's tick
    orientation (an x-tick dash is vertical, ``x1 == x2``; a y-tick dash is
    horizontal, ``y1 == y2``) and short (``<= dash_max_length``, default well
    above the theme's default 4px tick length and well below a gridline's or
    the domain line's span) — the same length-based landmark
    ``test_break_axis_geometry.py``'s ``_y_axis_domain_line_span`` uses to
    tell the domain line from a tick dash, applied the other way round.

    A HIDDEN dash (its tick's anchor fell inside a break's gap) still
    qualifies and is returned here at its actual (off-canvas, sentinel)
    position, never omitted: both its along-axis coordinates collapse to the
    same pixel under D2's rigid-translation model (spec §4 "Tick labels"),
    so ``x1 == x2``/``y1 == y2`` still holds, and its across-axis length is
    unchanged.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    is_x = axis == "x"
    groups: list[AxisTickDashLabels] = []
    current: AxisTickDashLabels | None = None
    for kind, coords in _ordered_lines_and_labels(svg):
        if kind == "line":
            x1, y1, x2, y2 = coords
            aligned = (x1 == x2) if is_x else (y1 == y2)
            length = abs(y2 - y1) if is_x else abs(x2 - x1)
            if aligned and length <= dash_max_length:
                current = AxisTickDashLabels(dash=coords, labels=[])
                groups.append(current)
            else:
                current = None
        elif current is not None:
            current.labels.append(coords)
    return groups


def tick_along(dash: tuple[float, float, float, float], *, axis: str) -> float:
    """A tick dash's own along-axis pixel — both ends of a qualifying dash
    (see :func:`axis_tick_dash_label_pairs`) agree on it by construction."""
    x1, y1, _x2, _y2 = dash
    return x1 if axis == "x" else y1


def label_along(label: tuple[float, float, str], *, axis: str) -> float:
    """A label's along-axis drawn coordinate (``x`` for an x-tick, ``y`` for a y-tick)."""
    x, y, _text = label
    return x if axis == "x" else y
