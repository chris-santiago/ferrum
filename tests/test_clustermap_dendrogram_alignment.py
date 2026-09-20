"""A clustermap's dendrograms span exactly the heatmap they describe.

A dendrogram is a claim about the heatmap beside it: the leaf above column B
has to sit above *column B*, and the leaf beside row r3 beside *row r3*, or the
figure is drawing a clustering it did not compute. Until now that held only by
coincidence, and then stopped holding at all.

Each cell of a `ClusterMapChart` grid lays out independently -- a dendrogram is
its own leaf, rendered at its own viewport and fitted into its slot -- so the
heatmap's plot rect (inset by its axis bands and colorbar) and a dendrogram's
had no reason to agree, and did not. With `#182`'s phantom gutter removed, the
column dendrogram in the `clustermap_basic` golden spanned figure x
`176.0..744.0` against the heatmap's `213.4..674.8`: 37px past it on the left
and 69px on the right, its four leaves landing between the columns rather than
on them. The row dendrogram was 2.7px/13.6px out on y.

The fix is the same one a jointplot's marginals use, and deliberately not a
second mechanism: the grid node CARRIES the relationship on the wire
(`extent_alignments`, emitted by `ClusterMapChart._composite_tree` -- the column
dendrogram follows the heatmap on x, the row dendrogram on y), and
`render::composite_render`'s `impose_extent_alignments` gives the follower the
leader's extent on the declared axis. An earlier cut inferred the relationship
from the grid's SHAPE, which could not serve this figure at all: a clustermap
is the same 2x2-with-a-hole as a jointplot and only the hole's corner told them
apart.

Every assertion here is a measured, FIGURE-SPACE extent (`panel_extents`,
`tests/_svg_rects.py`): a dendrogram cell is rendered at its own viewport and
then scaled into its slot, so a leaf-local comparison would pass while the
rendered figure stayed misaligned.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import panel_extents

#: Pixel tolerance, and why it is not `1e-3` like the jointplot module's.
#: A clustermap's cells do NOT share a native viewport (the dendrograms are
#: pre-resized), so the compositor fits each at its own scale and the SVG emits
#: that scale ROUNDED TO THREE DECIMALS. Composing it back, as any reader of
#: the rendered file must, costs up to `5e-4 * coordinate` -- ~0.35px out at
#: x=700. The renderer's own arithmetic is exact; this slack is the file
#: format's. It still discriminates by two orders of magnitude: the
#: misalignment this pins was 37px and 69px.
EPS = 0.5


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
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


def clustermap_panels(
    extents: dict[str, tuple[float, float, float, float]],
) -> tuple[str, str, str]:
    """Identify `(heatmap, column_dendrogram, row_dendrogram)` by position.

    The column dendrogram is the panel lying entirely ABOVE another; the row
    dendrogram the one entirely to the LEFT of another. Positional rather than
    by size, because after the fix each dendrogram TIES with the heatmap on its
    shared axis, so a "smallest extent" rule would be picking panels for a
    reason the fix does not guarantee.
    """
    assert len(extents) == 3, f"a two-dendrogram clustermap draws three panels; got {len(extents)}"
    above = [c for c, e in extents.items() if any(e[3] <= o[2] for o in extents.values())]
    left = [c for c, e in extents.items() if any(e[1] <= o[0] for o in extents.values())]
    assert len(above) == 1 and len(left) == 1 and above[0] != left[0], (
        f"expected one above-panel and one left-panel; got above={above} left={left}"
    )
    rest = [c for c in extents if c not in (above[0], left[0])]
    assert len(rest) == 1
    return rest[0], above[0], left[0]


def test_column_dendrogram_spans_exactly_the_heatmaps_x(frame):
    """The column dendrogram's x-extent equals the heatmap's.

    Equality, not "overlaps": the dendrogram's leaves are drawn on the same
    categorical x positions the heatmap's columns are, so a leaf and its column
    must be the same pixels. Pre-fix this figure's dendrogram was 37px wide on
    the left of the heatmap and 69px on the right.
    """
    extents = panel_extents(fm.clustermap(frame).to_svg())
    heatmap, col_dendro, _row = clustermap_panels(extents)
    assert extents[col_dendro][:2] == pytest.approx(extents[heatmap][:2], abs=EPS), (
        f"column dendrogram x {extents[col_dendro][:2]} must equal "
        f"the heatmap's {extents[heatmap][:2]}"
    )


def test_row_dendrogram_spans_exactly_the_heatmaps_y(frame):
    """The transpose: the row dendrogram's y-extent equals the heatmap's.

    This is the arm the containment guard used to refuse: the heatmap's 16px
    top padding scales to 13.33px in the dendrogram's frame, so the imposed
    extent starts inside the dendrogram's own padding. Padding is blank by
    construction, and refusing there left the two panels visibly unaligned to
    protect empty space (see `extent_fits_within`).
    """
    extents = panel_extents(fm.clustermap(frame).to_svg())
    heatmap, _col, row_dendro = clustermap_panels(extents)
    assert extents[row_dendro][2:] == pytest.approx(extents[heatmap][2:], abs=EPS), (
        f"row dendrogram y {extents[row_dendro][2:]} must equal "
        f"the heatmap's {extents[heatmap][2:]}"
    )


def test_each_dendrogram_keeps_its_own_extent_on_the_other_axis(frame):
    """Alignment is per-axis, not a shared rect.

    A column dendrogram that adopted the heatmap's y-extent too would be drawn
    on top of it. Each keeps its own band on the axis it does NOT share -- also
    what distinguishes this from the Overlay pre-pass, where members share the
    whole rect.
    """
    extents = panel_extents(fm.clustermap(frame).to_svg())
    heatmap, col_dendro, row_dendro = clustermap_panels(extents)
    assert extents[col_dendro][3] <= extents[heatmap][2] + EPS, (
        "the column dendrogram must sit entirely above the heatmap"
    )
    assert extents[row_dendro][1] <= extents[heatmap][0] + EPS, (
        "the row dendrogram must sit entirely to the left of the heatmap"
    )


def test_the_declaration_is_what_moves_the_dendrograms(frame):
    """The discriminating control: strip `extent_alignments` from the lowered
    wire tree, render the SAME tree, and the dendrograms go back to hanging
    past the heatmap.

    Without this, every assertion above is equally satisfied by a renderer that
    aligned these panels for some unrelated reason (a shape predicate, say, or
    a coincidence of gutters). It also pins that the leader is untouched: the
    heatmap's own extent is IDENTICAL in both renders, so imposition moves
    followers only.
    """
    lowered = fm.clustermap(frame)._composite_tree(auto_tooltips=False)
    declared = panel_extents(lowered.render_svg())

    assert lowered.tree["extent_alignments"] == [
        {"follower": 1, "leader": 3, "axis": "x"},
        {"follower": 2, "leader": 3, "axis": "y"},
    ], "the lowering must declare both dendrograms against the heatmap CELL (3)"
    lowered.tree.pop("extent_alignments")
    undeclared = panel_extents(lowered.render_svg())

    heatmap, col_dendro, row_dendro = clustermap_panels(declared)
    assert undeclared[heatmap] == pytest.approx(declared[heatmap], abs=EPS), (
        "the heatmap is the leader and must not move either way"
    )
    assert undeclared[col_dendro][:2] != pytest.approx(declared[col_dendro][:2], abs=EPS), (
        "without the declaration the column dendrogram must NOT span the heatmap's x"
    )
    assert undeclared[row_dendro][2:] != pytest.approx(declared[row_dendro][2:], abs=EPS), (
        "without the declaration the row dendrogram must NOT span the heatmap's y"
    )


def test_a_single_dendrogram_clustermap_aligns_too(frame):
    """`ClusterMapChart` lowers a one-dendrogram figure to a 2x1 / 1x2 grid
    with no hole -- a different shape, a different declaration, the same
    contract. Built directly because `clustermap()` always makes both."""
    from ferrum.composition import ClusterMapChart

    full = fm.clustermap(frame)
    col_only = ClusterMapChart(
        full.heatmap, col_dendrogram=full.col_dendrogram, dendrogram_ratio=full.dendrogram_ratio
    )
    extents = panel_extents(col_only.to_svg())
    assert len(extents) == 2
    (_, top), (_, bottom) = sorted(extents.items(), key=lambda kv: kv[1][2])
    assert top[:2] == pytest.approx(bottom[:2], abs=EPS), (
        f"the column dendrogram's x-extent {top[:2]} must equal the heatmap's {bottom[:2]}"
    )

    row_only = ClusterMapChart(
        full.heatmap, row_dendrogram=full.row_dendrogram, dendrogram_ratio=full.dendrogram_ratio
    )
    extents = panel_extents(row_only.to_svg())
    assert len(extents) == 2
    (_, left), (_, right) = sorted(extents.items(), key=lambda kv: kv[1][0])
    assert left[2:] == pytest.approx(right[2:], abs=EPS), (
        f"the row dendrogram's y-extent {left[2:]} must equal the heatmap's {right[2:]}"
    )
