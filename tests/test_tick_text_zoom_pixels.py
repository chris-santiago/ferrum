"""A wheel zoom moves every tick label with its axis, whatever the tick's text is.

The interactive runtime repositions the axis text of the panel the wheel zooms and leaves all
other text where it was drawn. Which text is axis text is decided by the scene, so a tick moves
with its axis whether its text is the data's own (``2``), a relabelled one (``Axis(label_map=)``),
a formatted one (``Axis(label_format=)``) or a band's label, and text that is not a tick of the
zoomed panel (a mark's own text label, another panel's ticks) stays where it was drawn.

Each row loads the exported ``.interactive()`` page in headless Chrome, reads the text the page
draws (the ``text.ferrum-label`` elements of its overlay) before and after one wheel zoom, and
compares each tick with where the zoom's own transform (``k``, ``x``, ``y``, read from the page)
puts it: ``p * k + t`` on the axis' own coordinate, unchanged on the other.

Every capture tests the committed WASM bundle at ``src/ferrum/_wasm/ferrum_wasm_bg.wasm``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

import polars as pl
import pytest

import ferrum as fm
from tests import _pixel_capture as pc

pytestmark = pytest.mark.wasm_pixels

# Where a tick is allowed to differ from its predicted position, in CSS pixels.
_TOL = 2.0

_LABELS_JS = """() => [...document.querySelectorAll('svg text.ferrum-label')].map(t => {
  const r = t.getBoundingClientRect();
  return [t.textContent, r.x + r.width / 2, r.y + r.height / 2];
})"""

_XY = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "t": ["P", "Q", "L", "M"],
    }
)
_WIDE = pl.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 5.0, 10.0]})
_BARS = pl.DataFrame({"c": ["a", "b", "c", "d"], "v": [1.0, 3.0, 2.0, 4.0]})


class Label(NamedTuple):
    text: str
    x: float
    y: float


class Row(NamedTuple):
    """A chart, where the wheel lands on it, and what must move or stay.

    ``moves`` picks the labels (by their position before the zoom) that are ticks of the zoomed
    panel and so follow its transform on ``axis``; ``stays`` picks the labels that must be where
    they were drawn.
    """

    build: Callable[[], Any]
    wheel_at: tuple[float, float]
    axis: str
    moves: Callable[[Label, pc.CanvasRect], bool] | None
    stays: Callable[[Label, pc.CanvasRect], bool] | None


def _axis_text(*texts: str) -> Callable[[Label, pc.CanvasRect], bool]:
    return lambda label, rect: label.text in texts


def _numeric_ticks() -> Any:
    """x ticks relabelled by a map and y ticks formatted to one decimal, with the data's own
    point labels layered over the same panel."""
    points = fm.Chart(_XY).mark_point()
    ticks = points.encode(
        x=fm.X("x", axis=fm.Axis(label_map={"2": "two"})),
        y=fm.Y("y", axis=fm.Axis(label_format=".1f")),
    )
    labels = fm.Chart(_XY).mark_text().encode(x="x", y="y", text="t")
    return ticks + labels


def _band_ticks() -> Any:
    axis = fm.Axis(label_map={"b": "Bee"})
    return fm.Chart(_BARS).mark_bar().encode(x=fm.X("c", axis=axis), y="v")


def _two_panels() -> Any:
    """Two side-by-side panels with the same tick text; the wheel zooms the left one."""
    left = (
        fm.Chart(_WIDE)
        .mark_point()
        .encode(x=fm.X("x", axis=fm.Axis(label_map={"4": "four"})), y="y")
        .properties(width=300, height=300)
    )
    right = fm.Chart(_WIDE).mark_point().encode(x="x", y="y").properties(width=300, height=300)
    return fm.hconcat(left, right)


def _right_half(label: Label, rect: pc.CanvasRect) -> bool:
    return label.x > rect.x + rect.w * 0.5


def _left_half(label: Label, rect: pc.CanvasRect) -> bool:
    return label.x <= rect.x + rect.w * 0.5


_ROWS: dict[str, Row] = {
    "label_map tick": Row(_numeric_ticks, (0.5, 0.4), "x", _axis_text("two"), _axis_text("L", "M")),
    "label_format tick": Row(
        _numeric_ticks, (0.5, 0.4), "y", _axis_text("2.0"), _axis_text("L", "M")
    ),
    "band label": Row(_band_ticks, (0.5, 0.4), "x", _axis_text("Bee"), None),
    "mark text label": Row(_numeric_ticks, (0.5, 0.4), "x", None, _axis_text("P", "Q", "L", "M")),
    "another panel's ticks": Row(
        _two_panels,
        (0.12, 0.4),
        "x",
        lambda label, rect: _left_half(label, rect) and label.text == "four",
        _right_half,
    ),
}


def _labels(page: Any) -> list[Label]:
    return [Label(*item) for item in page.evaluate(_LABELS_JS)]


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= _TOL


def _drawn_at(labels: list[Label], text: str, x: float, y: float) -> bool:
    return any(label.text == text and _near(label.x, x) and _near(label.y, y) for label in labels)


def _run(row: Row, browser: Any, tmp_path: Path) -> list[str]:
    """What went wrong in one row: ticks that did not move, labels that did, and a row that
    picked nothing on a side it names."""
    html = tmp_path / "chart.html"
    row.build().interactive().save(str(html))
    page = browser.new_page(viewport=pc.VIEWPORT, device_scale_factor=pc.DEVICE_SCALE_FACTOR)
    try:
        pc.load_interactive(page, html)
        rect = pc.canvas_rect(page)
        before = _labels(page)
        pc.wheel_zoom(
            page, (rect.x + rect.w * row.wheel_at[0], rect.y + rect.h * row.wheel_at[1]), -300.0
        )
        k, tx, ty = pc.zoom_transform(page)
        assert k > 1.2, f"the wheel did not zoom the panel (k={k})"
        after = _labels(page)
    finally:
        page.close()
    problems: list[str] = []
    moving = [label for label in before if row.moves and row.moves(label, rect)]
    staying = [label for label in before if row.stays and row.stays(label, rect)]
    for named, picked in ((row.moves, moving), (row.stays, staying)):
        if named is not None and not picked:
            problems.append("the row selects no label on a side it names")
    for label in moving:
        if row.axis == "x":
            want = (rect.x + (label.x - rect.x) * k + tx, label.y)
        else:
            want = (label.x, rect.y + (label.y - rect.y) * k + ty)
        if not _drawn_at(after, label.text, *want):
            problems.append(f"{label.text!r} drawn at {(label.x, label.y)} is not at {want}")
    for label in staying:
        if not _drawn_at(after, label.text, label.x, label.y):
            problems.append(f"{label.text!r} left {(label.x, label.y)}")
    return problems


@pytest.mark.parametrize("name", sorted(_ROWS))
def test_a_zoom_moves_each_tick_with_its_axis_and_leaves_other_text(
    name: str, browser: Any, tmp_path: Path
) -> None:
    assert _run(_ROWS[name], browser, tmp_path) == []
