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

Equal panel extents are necessary, not sufficient: they say nothing about where
a leaf lands inside its panel. The leaf-to-cell tests at the bottom measure the
leaves themselves. Every leaf sits on the centre of a heatmap cell, the leaf at
that cell is the row or column the tree names there (read from the linkage's
own `order` output, drawn through the dendrogram's own scale), and each tree's
leaves face the heatmap.

Every assertion here is a measured, FIGURE-SPACE coordinate (`panel_extents`,
`figure_elements`, `tests/_svg_rects.py`): a dendrogram cell is rendered at its
own viewport and then scaled into its slot, so a leaf-local comparison would
pass while the rendered figure stayed misaligned.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._core import secondary_key
from ferrum._layer import _Layer
from ferrum.composition import ClusterMapChart
from tests._svg_rects import THEME_BACKGROUND, figure_elements, panel_extents

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


# ---------------------------------------------------------------------------
# Leaf to cell
# ---------------------------------------------------------------------------

#: Fill of the probe text layer that names each leaf; no chart chrome uses it.
LEAF_ID_FILL = "#010203"

VALUE_COLS = ["A", "B", "C", "D"]


def _with_leaf_ids(dendrogram, link_name: str):
    """A copy of `dendrogram` with a text layer naming every leaf.

    The layer reads the linkage's `order` output and writes each leaf's
    `original_idx` at its `new_idx` on the leaf axis: the same coordinate the
    linkage gives that leaf's segments, mapped through the same chart scale.
    So the text sits wherever the leaf is drawn, and says which row or column
    the tree put there.
    """
    assert dendrogram._layers[0].data_source == secondary_key(link_name, "segments")
    probe = _Layer(
        name="leaf_ids",
        mark="text",
        encoding={"x": "new_idx", "y": "new_idx", "text": "original_idx"},
        mark_kwargs={"fill": LEAF_ID_FILL},
        data_source=secondary_key(link_name, "order"),
    )
    out = dendrogram.properties()
    out._layers = [*out._layers, probe]
    return out


def _heatmap(elements):
    """`(x0, x1, y0, y1)` of the heatmap's cells, plus its column and row cell centres."""
    cells = [
        e
        for e in elements
        if e["tag"] == "rect" and e["panel"] and e["fill"] not in (None, THEME_BACKGROUND)
    ]
    assert len({e["panel"] for e in cells}) == 1, "exactly one panel draws filled cells"
    xs = sorted({round(e["x"] + e["w"] / 2, 3) for e in cells})
    ys = sorted({round(e["y"] + e["h"] / 2, 3) for e in cells})
    box = (
        min(e["x"] for e in cells),
        max(e["x"] + e["w"] for e in cells),
        min(e["y"] for e in cells),
        max(e["y"] + e["h"] for e in cells),
    )
    return box, xs, ys


def _tick_labels(elements, box, *, columns: list[str], rows: list[str]):
    """Map each heatmap column centre x and row centre y to its tick label."""
    x0, x1, y0, y1 = box
    col_ticks = {
        e["text"]: e["x"]
        for e in elements
        if e["tag"] == "text" and e["panel"] is None and e["text"] in columns and e["y"] > y1
    }
    row_ticks = {
        e["text"]: e["y"]
        for e in elements
        if e["tag"] == "text" and e["panel"] is None and e["text"] in rows and e["x"] < x0
    }
    return col_ticks, row_ticks


def _tree_leaves(elements, box, side: str) -> list[float]:
    """Leaf positions of the tree drawn on `side` ("top" or "left") of the heatmap.

    A leaf is a riser (a segment along the distance axis) ending on the tree's
    heatmap-facing edge: the bottom of the top tree, the right of the left
    tree. A tree drawn with its leaves facing away yields only the root's two
    arms there, so the count also pins orientation.
    """
    x0, _x1, y0, _y1 = box
    if side == "top":
        lines = [
            e
            for e in elements
            if e["tag"] == "line" and e["panel"] and e["y2"] < y0 and e["y1"] < y0
        ]
        risers = [e for e in lines if abs(e["x1"] - e["x2"]) < 1e-6]
        edge = max(max(e["y1"], e["y2"]) for e in lines)
        return sorted(e["x1"] for e in risers if abs(max(e["y1"], e["y2"]) - edge) < 1e-6)
    lines = [
        e for e in elements if e["tag"] == "line" and e["panel"] and e["x2"] < x0 and e["x1"] < x0
    ]
    risers = [e for e in lines if abs(e["y1"] - e["y2"]) < 1e-6]
    edge = max(max(e["x1"], e["x2"]) for e in lines)
    return sorted(e["y1"] for e in risers if abs(max(e["x1"], e["x2"]) - edge) < 1e-6)


def _leaf_ids(elements, box, side: str) -> dict[float, int]:
    """Map each probe text's leaf-axis position to the `original_idx` it names."""
    x0, _x1, y0, _y1 = box
    texts = [e for e in elements if e["tag"] == "text" and e["fill"] == LEAF_ID_FILL]
    if side == "top":
        return {e["x"]: int(float(e["text"])) for e in texts if e["y"] < y0}
    return {e["y"]: int(float(e["text"])) for e in texts if e["x"] < x0}


def _clustermap_variants(frame: pl.DataFrame):
    full = fm.clustermap(frame)
    sized = fm.clustermap(frame, figsize=(520, 380), dendrogram_ratio=0.3)
    return {
        "both": (full, ("top", "left")),
        "figsize_ratio": (sized, ("top", "left")),
        "columns_only": (
            ClusterMapChart(
                full.heatmap,
                col_dendrogram=full.col_dendrogram,
                dendrogram_ratio=full.dendrogram_ratio,
            ),
            ("top",),
        ),
        "rows_only": (
            ClusterMapChart(
                full.heatmap,
                row_dendrogram=full.row_dendrogram,
                dendrogram_ratio=full.dendrogram_ratio,
            ),
            ("left",),
        ),
    }


VARIANTS = ["both", "figsize_ratio", "columns_only", "rows_only"]


def _with_probes(cm: ClusterMapChart) -> ClusterMapChart:
    return ClusterMapChart(
        cm.heatmap,
        row_dendrogram=None
        if cm.row_dendrogram is None
        else _with_leaf_ids(cm.row_dendrogram, "row_link"),
        col_dendrogram=None
        if cm.col_dendrogram is None
        else _with_leaf_ids(cm.col_dendrogram, "col_link"),
        dendrogram_ratio=cm.dendrogram_ratio,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_leaf_sits_on_its_own_cell_centre(frame, variant):
    """Per leaf: its pixel is a heatmap cell centre on that axis, one leaf per cell.

    Linkage puts leaf k at data coordinate k. The heatmap's cell k is centred at
    `lo + (k + 0.5) * span / n`, so the dendrogram's leaf axis must map
    `[-0.5, n - 0.5]` onto the heatmap's span with no padding. An auto-domain
    scale maps `[0, n - 1]` with a padding inset instead, and only the middle
    leaf of an odd count coincides.
    """
    cm, sides = _clustermap_variants(frame)[variant]
    elements = figure_elements(cm.to_svg())
    box, xs, ys = _heatmap(elements)
    for side in sides:
        centres = xs if side == "top" else ys
        leaves = _tree_leaves(elements, box, side)
        assert len(leaves) == len(centres), (
            f"{side} tree: {len(leaves)} leaves end on its heatmap-facing edge, "
            f"expected one per cell ({len(centres)}): the leaves must face the heatmap"
        )
        for k, (leaf, centre) in enumerate(zip(leaves, centres)):
            assert leaf == pytest.approx(centre, abs=EPS), (
                f"{side} tree leaf {k} is drawn at {leaf:.3f}px, "
                f"but its cell is centred at {centre:.3f}px"
            )


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_leaf_names_the_cell_it_sits_on(frame, variant):
    """Per leaf: the row or column the tree names there is the cell's tick label.

    Pixel alignment alone passes a tree drawn mirror-image to the heatmap, or a
    heatmap whose columns are not in the tree's leaf order. The leaf's identity
    comes from the linkage's `order` output drawn through the dendrogram's own
    scale; the cell's comes from the heatmap's tick label at that centre.
    """
    cm, sides = _clustermap_variants(frame)[variant]
    plain = figure_elements(cm.to_svg())
    elements = figure_elements(_with_probes(cm).to_svg())
    box, xs, ys = _heatmap(elements)
    plain_box = _heatmap(plain)[0]
    rows = frame["row"].to_list()
    col_ticks, row_ticks = _tick_labels(elements, box, columns=VALUE_COLS, rows=rows)
    assert sorted(col_ticks) == VALUE_COLS and sorted(row_ticks) == sorted(rows)
    for side in sides:
        names = VALUE_COLS if side == "top" else rows
        ticks = col_ticks if side == "top" else row_ticks
        centres = xs if side == "top" else ys
        ids = _leaf_ids(elements, box, side)
        assert len(ids) == len(centres), f"{side} tree: one leaf id per cell"
        # Relative to the heatmap's own corner: the probed figure is rebuilt
        # without the figure title, which shifts every panel alike.
        lo, plain_lo = (box[0], plain_box[0]) if side == "top" else (box[2], plain_box[2])
        assert [p - lo for p in sorted(ids)] == pytest.approx(
            [p - plain_lo for p in _tree_leaves(plain, plain_box, side)], abs=EPS
        ), f"{side} tree: the probe layer must not move the leaves"
        for pos, idx in sorted(ids.items()):
            # Nearest tick: a row label's baseline sits a few px below its cell centre.
            label = min(ticks, key=lambda t: abs(ticks[t] - pos))
            centre = min(centres, key=lambda c: abs(c - pos))
            assert pos == pytest.approx(centre, abs=EPS), (
                f"{side} tree leaf {names[idx]!r} is drawn at {pos:.3f}px, off every cell centre"
            )
            assert label == names[idx], (
                f"{side} tree puts leaf {names[idx]!r} at {pos:.3f}px, "
                f"but the heatmap cell there is {label!r}"
            )


def test_columns_follow_the_column_clustering_with_extra_non_numeric_columns(frame):
    """A non-numeric column that is not the id is left out of the heatmap.

    The column reorder melts every non-id column of the reordered batch, so the
    heatmap must see only the id and the numeric value columns; the extra
    column neither breaks the render nor changes a leaf.
    """
    extra = frame.with_columns(pl.lit("g").alias("group"))
    elements = figure_elements(_with_probes(fm.clustermap(extra)).to_svg())
    box, xs, _ys = _heatmap(elements)
    col_ticks, _ = _tick_labels(elements, box, columns=VALUE_COLS, rows=[])
    assert len(xs) == len(VALUE_COLS)
    ids = _leaf_ids(elements, box, "top")
    assert [VALUE_COLS[i] for _pos, i in sorted(ids.items())] == sorted(
        col_ticks, key=col_ticks.get
    )
