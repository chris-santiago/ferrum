"""Regression tests for F-L09-08: clustermap dendrogram phantom-gutter collapse.

``fm.clustermap(dendrogram_ratio=0.1)`` — an in-range, documented value — and
the default ratio (0.2) on a short heatmap (``figsize=(480, 200)``) used to
silently blank one or both dendrogram panels with a "collapsed to zero size"
warning, even though the requested ratio left plenty of room for the tree
branches.

Root cause: the column/row dendrogram leaves are built with ``.axis(show=False)``
(clustermap's dendrograms have no meaningful axis — only the tree structure
matters), but the layout code reserved a fixed axis-title gutter regardless of
``show``. Blanking the dendrogram layers' ``x``/``y`` titles (mirroring the
center heatmap's own ``title=""`` at ``matrix.py``'s encode call) collapses
that gutter to zero — *provided* the per-layer spec builder actually forwards
an empty-string title onto the wire. It did not:
``SpecBuildMixin._build_layers_list`` filtered optional encoding keys by
truthiness (``if d.get(opt_key):``), which silently dropped ``title=""``
(a real, falsy value meaning "no title") exactly like an absent key. The fix
touches both sites: ``matrix.py`` blanks the dendrogram titles, and
``_spec_build.py`` forwards them by presence (``if opt_key in d:``) instead of
truthiness.

Sub-floor ratios (too little room for anything, dendrogram or not) still raise
the pre-existing hard ``PaddingExceedsViewport`` error — that refusal is
unrelated to this fix and must keep firing.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def df_small():
    """5 id rows x 4 numeric columns -> row dendrogram has 4 merges (12 lines),
    col dendrogram has 3 merges (9 lines), each merge drawing a 3-segment
    bracket (mark_segment emits one SVG ``<line>`` per row of the linkage
    transform's ``segments`` output).
    """
    rng = np.random.default_rng(2)
    return pl.DataFrame(
        {
            "row": [f"r{i}" for i in range(5)],
            "A": rng.normal(0, 1, 5).tolist(),
            "B": rng.normal(0, 1, 5).tolist(),
            "C": rng.normal(0, 1, 5).tolist(),
            "D": rng.normal(0, 1, 5).tolist(),
        }
    )


def _panel_window(svg: str, clip_index: int) -> str:
    """Return one composite leaf's own marks group, e.g. ``ferrum-clip-1``.

    ``ClusterMapChart._composite_tree`` lays cells out in a fixed,
    documented order -- ``[HOLE, col_dendrogram, row_dendrogram, heatmap]``
    -- so the leaves render in that order and claim
    ``ferrum-clip-0``/``ferrum-clip-1``/``ferrum-clip-2`` respectively. Each
    leaf's marks render as a flat (non-nested) ``<g clip-path=...>...</g>``
    group, so the slice up to the very next ``</g>`` is exactly that leaf's
    marks -- not the grid/axis chrome the SVG walker draws around it, which
    lives in separate sibling ``<g>`` groups outside this clip wrapper.
    """
    marker = f'<g clip-path="url(#ferrum-clip-{clip_index})"'
    start = svg.index(marker)
    end = svg.index("</g>", start) + len("</g>")
    return svg[start:end]


def _collapse_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if "collapsed" in str(w.message) or "too small" in str(w.message)
    ]


# A dendrogram panel that rendered as a near-invisible sliver (a future
# regression that reintroduces some OTHER phantom reservation, not the one
# this fix closes) would still draw the expected `<line>` count -- the
# count alone doesn't pin that the branches actually got usable space.
# This floor is well below every value this fix produces today (~11-16px,
# measured on both scenarios below) and well above a 1px collapse.
_MIN_EXTENT_PX = 5.0


def _line_spans(panel: str) -> tuple[float, float]:
    """Return (x-span, y-span) covering every ``<line>`` endpoint in *panel*.

    The discriminating dimension differs by dendrogram: the column
    dendrogram's tree depth reads along y (its leaves fan out along x, the
    shared heatmap width), while the row dendrogram is ``CoordFlip``-ped so
    its tree depth reads along x. Callers check whichever span the phantom
    gutter used to eat into.
    """
    xs: list[float] = []
    ys: list[float] = []
    for m in re.finditer(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"', panel):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        xs += [x1, x2]
        ys += [y1, y2]
    return max(xs) - min(xs), max(ys) - min(ys)


def test_ratio_0_1_default_size_renders_both_dendrograms_warning_free(df_small):
    """dendrogram_ratio=0.1 is in the documented (0, 1) range and must render
    both dendrograms with tree content, not blank them (F-L09-08).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm = fm.clustermap(df_small, dendrogram_ratio=0.1)
        svg = cm.to_svg()

    assert _collapse_warnings(caught) == []

    col_panel = _panel_window(svg, 0)
    row_panel = _panel_window(svg, 1)
    assert len(re.findall(r"<line\b", col_panel)) == 9, (
        "column dendrogram (3 merges) must draw its branches"
    )
    assert len(re.findall(r"<line\b", row_panel)) == 12, (
        "row dendrogram (4 merges) must draw its branches"
    )

    # Extent, not just count: a squashed 1px-tall/wide dendrogram would still
    # draw the right number of lines and pass the assertions above.
    _, col_y_span = _line_spans(col_panel)
    row_x_span, _ = _line_spans(row_panel)
    assert col_y_span > _MIN_EXTENT_PX, "column dendrogram must get usable tree-depth height"
    assert row_x_span > _MIN_EXTENT_PX, "row dendrogram must get usable tree-depth width"

    # The heatmap itself must still render intact alongside the dendrograms.
    heatmap_panel = _panel_window(svg, 2)
    assert "<rect" in heatmap_panel


def test_short_heatmap_default_ratio_renders_column_dendrogram(df_small):
    """A short heatmap (figsize=(480, 200)) at the DEFAULT ratio (0.2) used
    to blank the column dendrogram (its allotted height shrinks with the
    heatmap's own height) even though 0.2 is unremarkable. Must render clean.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm = fm.clustermap(df_small, figsize=(480, 200))
        svg = cm.to_svg()

    assert _collapse_warnings(caught) == []

    col_panel = _panel_window(svg, 0)
    assert len(re.findall(r"<line\b", col_panel)) == 9, (
        "column dendrogram (3 merges) must draw its branches"
    )
    _, col_y_span = _line_spans(col_panel)
    assert col_y_span > _MIN_EXTENT_PX, "column dendrogram must get usable tree-depth height"


@pytest.mark.parametrize("ratio", [0.074, 0.05])
def test_sub_floor_ratio_still_raises_padding_exceeds_viewport(df_small, ratio):
    """A ratio too small for ANYTHING to render (not just the dendrogram) is
    a genuine hard failure and must keep raising -- this fix only removes a
    phantom reservation, it does not lower the real floor.
    """
    with pytest.raises(ValueError, match=r"padding top=16 exceeds viewport dimension"):
        fm.clustermap(df_small, dendrogram_ratio=ratio).to_svg()
