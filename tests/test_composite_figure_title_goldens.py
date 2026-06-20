"""SVG golden tests for composite figure-level chrome (#8).

These goldens prove that `.properties(title=…, subtitle=…, caption=…)` on
composite charts places chrome at the *figure* level in the rendered SVG, not
inside an inner panel.  Four representative cases are covered:

- ``vconcat_full_chrome``: VConcat with title + subtitle + caption (exercises
  the full header/footer chrome band).
- ``joint_figure_title``: JointChart with a figure title and marginal panels.
- ``clustermap_figure_title``: ClusterMapChart with a figure title.
- ``repeat_figure_title``: RepeatChart (2×2 grid) with a figure title.

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens; the
default behavior is byte-identical comparison.  A missing golden is an explicit
failure, not a silent skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from tests._snapshots import assert_svg_eq


GOLDENS_DIR = Path(__file__).parent / "goldens" / "composite_figure_title"
UPDATE = os.environ.get("FERRUM_UPDATE_GOLDENS") == "1"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _check_or_update(name: str, svg: str) -> None:
    """Compare ``svg`` to the on-disk golden ``name``.

    With ``FERRUM_UPDATE_GOLDENS=1`` the golden is (re)written.  Otherwise
    asserts byte-for-byte equality.  A missing golden is an explicit test
    failure rather than a silent skip.
    """
    path = GOLDENS_DIR / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        return
    if not path.exists():
        pytest.fail(
            f"golden {name!r} does not exist; rerun with FERRUM_UPDATE_GOLDENS=1 to generate it"
        )
    expected = path.read_text()
    assert_svg_eq(
        svg,
        expected,
        name=name,
        regen_hint="rerun with FERRUM_UPDATE_GOLDENS=1 to refresh",
    )


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    """Small, fully deterministic 4-row dataset for stable goldens."""
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})


@pytest.fixture
def base_chart(df):
    return fm.Chart(df).mark_point().encode(x="x", y="y")


@pytest.fixture
def bar_chart(df):
    return fm.Chart(df).mark_bar().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Golden 1: VConcat with full chrome (title + subtitle + caption)
# ---------------------------------------------------------------------------


def test_vconcat_full_chrome_golden(base_chart, bar_chart):
    """VConcat with title, subtitle, and caption exercises the complete header+footer band.

    Visual check: title and subtitle appear above both panels; caption appears
    below the lower panel.  Neither inner panel carries the chrome text.
    """
    chart = (base_chart & bar_chart).properties(
        title="VConcat Figure Title",
        subtitle="A subtitle",
        caption="A caption",
    )
    svg = chart.to_svg()

    # Sanity assertions before golden comparison: chrome text is in the composed SVG.
    assert "VConcat Figure Title" in svg
    assert "A subtitle" in svg
    assert "A caption" in svg
    # Inner panels carry no stray chrome.
    for child in chart.charts:
        child_svg = child.to_svg()
        assert "VConcat Figure Title" not in child_svg
        assert "A caption" not in child_svg

    _check_or_update("vconcat_full_chrome.svg", svg)


# ---------------------------------------------------------------------------
# Golden 2: JointChart with figure title and marginal panels
# ---------------------------------------------------------------------------


def test_joint_figure_title_golden(base_chart, bar_chart):
    """JointChart (center + top + right marginals) with a figure-level title.

    Visual check: title appears above the whole joint layout (above the top
    marginal), not inside the center scatter panel.
    """
    chart = fm.JointChart(base_chart, top=bar_chart, right=bar_chart).properties(
        title="Joint Figure Title"
    )
    svg = chart.to_svg()

    assert "Joint Figure Title" in svg
    # Center and marginal panels must not carry the title independently.
    assert "Joint Figure Title" not in chart.center.to_svg()
    assert "Joint Figure Title" not in chart.top.to_svg()
    assert "Joint Figure Title" not in chart.right.to_svg()

    _check_or_update("joint_figure_title.svg", svg)


# ---------------------------------------------------------------------------
# Golden 3: ClusterMapChart with figure title
# ---------------------------------------------------------------------------


def test_clustermap_figure_title_golden():
    """ClusterMapChart with a figure-level title.

    Visual check: title appears above the heatmap grid, not inside it.
    """
    wide = pl.DataFrame(
        {
            "row": ["a", "b", "c", "d"],
            "x": [1.0, 0.5, 0.3, 0.9],
            "y": [0.5, 1.0, 0.7, 0.2],
            "z": [0.3, 0.7, 1.0, 0.6],
        }
    )
    heatmap = fm.Chart(wide).mark_rect().encode(x="row", y="row", color="x")
    chart = fm.ClusterMapChart(heatmap).properties(title="Cluster Figure Title")
    svg = chart.to_svg()

    assert "Cluster Figure Title" in svg
    assert "Cluster Figure Title" not in chart.heatmap.to_svg()

    _check_or_update("clustermap_figure_title.svg", svg)


# ---------------------------------------------------------------------------
# Golden 4: RepeatChart (2×2 grid) with figure title
# ---------------------------------------------------------------------------


def test_repeat_figure_title_golden(df):
    """RepeatChart (2×2 row×column grid) with a figure-level title.

    Visual check: title appears above the entire 2×2 grid, not inside any
    individual cell panel.
    """
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    chart = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"]).properties(
        title="Repeat Figure Title"
    )
    svg = chart.to_svg()

    assert "Repeat Figure Title" in svg
    # No materialized cell carries the figure title independently.
    assert chart.template._title is None
    for _r, _c, cell in chart.expand():
        assert "Repeat Figure Title" not in cell.to_svg()

    _check_or_update("repeat_figure_title.svg", svg)
