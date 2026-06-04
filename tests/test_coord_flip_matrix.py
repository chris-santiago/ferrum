"""Matrix test: verify every primitive mark renders correctly under CoordFlip.

For each mark, render twice — once without coord_flip and once with — and assert:
1. The SVGs differ (CoordFlip should swap axes).
2. The flipped SVG still contains mark elements (no silently dropped marks).
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm


@pytest.fixture()
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0],
            "x2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "y2": [3.0, 5.0, 2.0, 6.0, 4.0],
            "cat": ["a", "b", "c", "d", "e"],
            "cat2": ["p", "q", "r", "s", "t"],
            "label": ["one", "two", "three", "four", "five"],
        }
    )


MARK_CONFIGS = {
    "mark_point": {
        "encoding": {"x": "x:Q", "y": "y:Q"},
        "expected_elements": ["<circle"],
    },
    "mark_line": {
        "encoding": {"x": "x:Q", "y": "y:Q"},
        "expected_elements": ["<polyline"],
    },
    "mark_bar": {
        "encoding": {"x": "cat:N", "y": "y:Q"},
        "expected_elements": ["<rect"],
    },
    "mark_area": {
        "encoding": {"x": "x:Q", "y": "y:Q"},
        "expected_elements": ["<path"],
    },
    "mark_rule": {
        "encoding": {"x": "x:Q", "y": "y:Q"},
        "expected_elements": ["<line"],
    },
    "mark_text": {
        "encoding": {"x": "x:Q", "y": "y:Q", "text": "label:N"},
        "expected_elements": ["<text"],
    },
    "mark_tick": {
        "encoding": {"x": "x:Q", "y": "y:Q"},
        "expected_elements": ["<line"],
    },
    "mark_rect": {
        "encoding": {"x": "cat:N", "y": "cat2:N"},
        "expected_elements": ["<rect"],
    },
    "mark_segment": {
        "encoding": {"x": "x:Q", "y": "y:Q", "x2": "x2:Q", "y2": "y2:Q"},
        "expected_elements": ["<line"],
    },
}


@pytest.mark.parametrize("mark_method", list(MARK_CONFIGS.keys()))
def test_coord_flip_changes_output(df, mark_method):
    """CoordFlip produces a different SVG than the non-flipped baseline."""
    cfg = MARK_CONFIGS[mark_method]
    enc = cfg["encoding"]

    base_chart = getattr(fm.Chart(df), mark_method)().encode(**enc)
    base_svg = base_chart.to_svg()

    flipped_chart = getattr(fm.Chart(df), mark_method)().encode(**enc).coord(fm.CoordFlip())
    flipped_svg = flipped_chart.to_svg()

    assert base_svg != flipped_svg, (
        f"{mark_method}: SVG unchanged after coord_flip — flip may be silently ignored"
    )


@pytest.mark.parametrize("mark_method", list(MARK_CONFIGS.keys()))
def test_coord_flip_preserves_marks(df, mark_method):
    """CoordFlip does not silently drop mark elements from the SVG."""
    cfg = MARK_CONFIGS[mark_method]
    enc = cfg["encoding"]

    flipped_chart = getattr(fm.Chart(df), mark_method)().encode(**enc).coord(fm.CoordFlip())
    flipped_svg = flipped_chart.to_svg()

    assert "<svg" in flipped_svg, f"{mark_method}: coord_flip produced invalid SVG (no <svg root)"

    for element in cfg["expected_elements"]:
        assert element in flipped_svg, (
            f"{mark_method}: coord_flip SVG missing expected element '{element}' "
            f"— mark may be silently dropped under CoordFlip"
        )
