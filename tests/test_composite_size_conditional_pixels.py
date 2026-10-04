"""Pixel-level proof that a composite child's ``size`` conditional resizes only
its own child's marks in the interactive view.

``pt | ln`` places a conditional-free point chart beside a line whose width is
a ``size`` conditional. After a click selects a point, each point's on-screen
radius must stay as it was at rest, whether the line's conditional reads the
same selection or another one. The control gives the point chart its own
``size`` conditional and the line none, which must shrink the unselected points,
so a click that never registered cannot pass the other cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from tests import _pixel_capture as pc

pytestmark = pytest.mark.wasm_pixels

_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 2.0, 4.0]})
_RED = (255, 0, 0)


def _chart(line_selection: str | None, point_sized: bool) -> Any:
    sel = fm.selection_point(name="s", clear="false")
    point = fm.Chart(_DF).mark_point(size=400, fill="#ff0000", stroke="#ff0000")
    enc: dict[str, Any] = {"x": "x", "y": "y"}
    if point_sized:
        enc["size"] = fm.when(sel).then(400).otherwise(20)
    pt = point.encode(**enc).add_params(sel)
    ln = fm.Chart(_DF).mark_line(stroke_width=2).encode(x="x", y="y")
    if line_selection is not None:
        line_sel = (
            sel if line_selection == "s" else fm.selection_point(name=line_selection, clear="false")
        )
        ln = (
            fm.Chart(_DF)
            .mark_line()
            .encode(x="x", y="y", size=fm.when(line_sel).then(8).otherwise(2))
            .add_params(line_sel)
        )
    return pt | ln


def _radii(browser: Any, chart: Any, html: Path) -> tuple[list[float], list[float]]:
    svg = chart.to_svg()
    w, h = pc.parse_svg_size(svg)
    circles = [c for c in pc.parse_circles(svg) if c[2] > 5]
    assert len(circles) == 4, circles
    chart.interactive().save(str(html))
    page = browser.new_page(device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, html)
        rect = pc.canvas_rect(page)
        centers = [(cx, cy) for cx, cy, _ in circles]
        before = pc.screenshot(page)
        rest = [pc.scan_radius(page, before, rect, w, h, c, _RED) for c in centers]
        pc.click_at(page, pc.svg_to_client(rect, w, h, centers[0]))
        after = pc.screenshot(page)
        clicked = [pc.scan_radius(page, after, rect, w, h, c, _RED) for c in centers]
    finally:
        page.close()
    return rest, clicked


@pytest.mark.parametrize("line_selection", ["s", "other"])
def test_a_sibling_childs_line_size_conditional_leaves_the_points_unchanged(
    line_selection: str, browser: Any, tmp_path: Path
) -> None:
    rest, clicked = _radii(browser, _chart(line_selection, point_sized=False), tmp_path / "c.html")
    assert min(rest) >= 5, rest
    assert all(abs(a - b) <= 1 for a, b in zip(rest, clicked)), (rest, clicked)


def test_control_the_points_own_size_conditional_shrinks_the_unselected(
    browser: Any, tmp_path: Path
) -> None:
    rest, clicked = _radii(browser, _chart(None, point_sized=True), tmp_path / "c.html")
    assert min(rest) >= 5, rest
    assert abs(clicked[0] - rest[0]) <= 1, (rest, clicked)
    assert all(b <= a - 4 for a, b in zip(rest[1:], clicked[1:])), (rest, clicked)
