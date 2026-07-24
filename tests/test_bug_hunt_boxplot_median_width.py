"""Regression tests: the boxplot median tick must render at the box's full width.

The ``tick`` mark's ``band_size`` and the ``rect`` mark's ``band_size`` both
use the same FULL-width convention (rendered length = ``band_size * slot``).
Historically ``tick``'s ``band_size`` was a HALF-width factor (full rendered
length = ``2 * band_size * slot``), which made the boxplot median tick render
at twice the box's width and, under dodge, bleed into the neighbouring hue
sub-band. ``desugar_boxplot`` originally worked around that mismatch by
halving ``band`` for the median tick's ``band_size``; once ``tick`` adopted
the same full-width convention as ``rect`` (T9, GH #85), the median layer
passes ``band`` straight through and literally spans the box's width.

Parse conventions mirror ``tests/test_bug_hunt_band_point_range.py``.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm

_THEME_BACKGROUND = "#faf7f2"
_MEDIAN_STROKE_WIDTH = "2"  # desugar_boxplot's median layer sets stroke_width=2


def _svg_elements(svg: str, tag: str, keep) -> list[dict[str, str]]:
    out = []
    for m in re.finditer(rf"<{tag}([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        if keep(attrs):
            out.append(attrs)
    return out


def _box_rects(svg: str) -> list[dict[str, str]]:
    """The IQR box ``rect`` layer: filled, non-background."""
    return _svg_elements(
        svg,
        "rect",
        lambda a: (
            bool(a.get("fill"))
            and a["fill"] != _THEME_BACKGROUND
            and not a["fill"].startswith("url(")
        ),
    )


def _median_lines(svg: str) -> list[dict[str, str]]:
    """The median ``tick`` layer: the only line with ``stroke-width="2"``."""
    return _svg_elements(
        svg,
        "line",
        lambda a: a.get("stroke-width") == _MEDIAN_STROKE_WIDTH,
    )


def test_median_tick_matches_box_width_undodged():
    """Undodged 2-category boxplot: median tick length == box width.

    Regression test for the band_size half-width/full-width mismatch that
    predated the tick/rect ``band_size`` unification: before the original fix,
    the median tick's full rendered length was ``2 * box_width``.
    """
    df = pl.DataFrame(
        {
            "cat": ["a"] * 5 + ["b"] * 5,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_boxplot(outliers=False)
        .encode(x="cat", y="val")
        .properties(width=600, height=400)
        .to_svg()
    )
    boxes = _box_rects(svg)
    medians = _median_lines(svg)
    assert len(boxes) == 2, f"expected 2 boxes, got {len(boxes)}"
    assert len(medians) == 2, f"expected 2 median ticks, got {len(medians)}"
    for box in boxes:
        box_width = float(box["width"])
        box_x0 = float(box["x"])
        box_x1 = box_x0 + box_width
        # Match the median tick centered over this box's x-span.
        median = next(
            m
            for m in medians
            if box_x0 - 1.0 <= (float(m["x1"]) + float(m["x2"])) / 2.0 <= box_x1 + 1.0
        )
        median_length = abs(float(median["x2"]) - float(median["x1"]))
        assert median_length == pytest.approx(box_width, abs=0.01), (
            f"median tick length {median_length} != box width {box_width}"
        )


def test_median_tick_matches_box_width_and_stays_within_box_dodged():
    """Dodged (hue-split) boxplot: each median tick matches its own box width
    and does not bleed into a neighbouring sub-band.

    Regression test: pre-fix, the oversized median tick (2x box width) spilled
    past its box's x-extent into the adjacent hue sub-band under dodge.
    """
    rows = []
    values_by_group = {
        ("x", "g1"): [1.0, 2.0, 3.0, 4.0, 5.0],
        ("x", "g2"): [6.0, 7.0, 8.0, 9.0, 10.0],
        ("y", "g1"): [11.0, 12.0, 13.0, 14.0, 15.0],
        ("y", "g2"): [16.0, 17.0, 18.0, 19.0, 20.0],
    }
    for (cat, grp), values in values_by_group.items():
        for v in values:
            rows.append({"cat": cat, "grp": grp, "val": v})
    df = pl.DataFrame(rows)

    chart = fm.catplot(df, x="cat", y="val", hue="grp", kind="box", dodge=True).properties(
        width=600, height=400
    )
    svg = chart.to_svg()

    boxes = _box_rects(svg)
    medians = _median_lines(svg)
    assert len(boxes) == 4, f"expected 4 dodged boxes (2 cat x 2 hue), got {len(boxes)}"
    assert len(medians) == 4, f"expected 4 median ticks, got {len(medians)}"

    for box in boxes:
        box_width = float(box["width"])
        box_x0 = float(box["x"])
        box_x1 = box_x0 + box_width
        median = next(
            m
            for m in medians
            if box_x0 - 1.0 <= (float(m["x1"]) + float(m["x2"])) / 2.0 <= box_x1 + 1.0
        )
        median_x0 = min(float(median["x1"]), float(median["x2"]))
        median_x1 = max(float(median["x1"]), float(median["x2"]))
        median_length = median_x1 - median_x0
        assert median_length == pytest.approx(box_width, abs=0.01), (
            f"median tick length {median_length} != box width {box_width}"
        )
        assert median_x0 >= box_x0 - 0.01 and median_x1 <= box_x1 + 0.01, (
            f"median tick [{median_x0}, {median_x1}] bleeds outside its box "
            f"[{box_x0}, {box_x1}] into a neighbouring sub-band"
        )
