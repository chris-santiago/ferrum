"""A jointplot's marginals span exactly the joint panel's axis.

A marginal distribution is only meaningful *on the joint panel's axis*: the
bar above x=0.5 has to sit above the joint panel's x=0.5, or the figure is
lying about which part of the distribution it is summarizing. Until now that
held only by coincidence.

Each cell of a `JointChart` grid lays out independently — the marginal is its
own leaf, rendered at its own viewport and fitted into its slot — so the joint
panel's plot rect (inset by its axis bands and any colorbar) and the marginal's
(inset by whatever the marginal reserved) had no reason to agree, and did not:
in the pre-`#182` `jointplot_kde_hist` golden the top marginal spanned x
`53.377..624.0` while the joint panel spanned `67.423..547.025` — 14px out on
the left and 77px out on the right. It *looked* about right because the
marginal's own phantom axis gutter (`#94`/`#182`) happened to inset it by
roughly the joint panel's axis band. Removing the phantom removed the
coincidence and the misalignment became obvious, which is how it was found.

The fix makes the relationship explicit rather than restoring the coincidence:
the grid node CARRIES the relationship on the wire (`extent_alignments`, one
`{follower, leader, axis}` entry per marginal, emitted by
`JointChart._composite_tree`), and `render::composite_render`'s
`impose_extent_alignments` gives each marginal the joint panel's extent on the
declared axis through the same `imposed_plot_region` seam the Overlay
chrome-dedup pre-pass (`#89A`) uses. An earlier cut inferred the relationship
from the grid's shape instead; that predicate was deleted, because a clustermap
is the same 2x2-with-a-hole and only the hole's corner told them apart.

Every assertion here is a measured, FIGURE-SPACE extent: a marginal cell is
rendered at its leaf's own viewport and then scaled into its slot by
`plan_grid`, so a leaf-local comparison would pass while the rendered figure
stayed misaligned. `panel_extents` (`tests/_svg_rects.py`) applies each panel's
group transform before comparing, so what is asserted is what a reader sees.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import panel_extents

#: Pixel tolerance. A jointplot's cells share one native viewport, so the
#: compositor places them at identical (here, identity) scales and the composed
#: figure coordinates carry no scale-rounding error -- only the 3-decimal
#: rounding of the coordinates themselves. Anything above ~1e-3 would be a real
#: disagreement rather than formatting. (`ClusterMapChart`'s cells do NOT share
#: a viewport, which is why `test_clustermap_dendrogram_alignment.py` has to
#: allow for the rounded `scale(...)` attribute; see `panel_extents`.)
EPS = 1e-3


def joint_top_right(
    extents: dict[str, tuple[float, float, float, float]],
) -> tuple[str, str, str]:
    """Identify the three panels of a two-marginal jointplot, by position.

    The top marginal is the panel that lies entirely ABOVE another panel; the
    right marginal is the one entirely to the RIGHT of another. Positional
    rather than by size, because after the fix each marginal TIES with the
    joint panel on its shared axis -- a "smallest extent" rule would then be
    ambiguous, and one that happened to pick the right panel today would be
    picking it for a reason the fix does not guarantee.
    """
    assert len(extents) == 3, f"a two-marginal jointplot draws three panels; got {len(extents)}"
    top = [c for c, e in extents.items() if any(e[3] <= o[2] for o in extents.values())]
    right = [c for c, e in extents.items() if any(e[0] >= o[1] for o in extents.values())]
    assert len(top) == 1 and len(right) == 1 and top[0] != right[0], (
        f"expected one above-panel and one right-panel; got top={top} right={right}"
    )
    rest = [c for c in extents if c not in (top[0], right[0])]
    assert len(rest) == 1
    return rest[0], top[0], right[0]


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300) * 2 + 1})


@pytest.mark.parametrize("kind", ["scatter", "hist", "kde"])
def test_top_marginal_spans_exactly_the_joint_panels_x(frame, kind):
    """The top marginal's x-extent equals the joint panel's, to the pixel.

    Equality, not "close": the two panels describe the same x scale, so a bar
    at x=0.5 and the joint panel's x=0.5 must be the same pixel. Pre-fix this
    fixture's marginal was 14px wide on the left and 77px on the right of the
    joint panel's span.
    """
    svg = fm.jointplot(frame, x="x", y="y", kind=kind).to_svg()
    extents = panel_extents(svg)
    joint, top, _right = joint_top_right(extents)
    assert extents[top][:2] == pytest.approx(extents[joint][:2], abs=EPS), (
        f"top marginal x {extents[top][:2]} must equal the joint panel's {extents[joint][:2]}"
    )


@pytest.mark.parametrize("kind", ["scatter", "hist", "kde"])
def test_right_marginal_spans_exactly_the_joint_panels_y(frame, kind):
    """The transpose: the right marginal's y-extent equals the joint panel's."""
    svg = fm.jointplot(frame, x="x", y="y", kind=kind).to_svg()
    extents = panel_extents(svg)
    joint, _top, right = joint_top_right(extents)
    assert extents[right][2:] == pytest.approx(extents[joint][2:], abs=EPS), (
        f"right marginal y {extents[right][2:]} must equal the joint panel's {extents[joint][2:]}"
    )


def test_each_marginal_keeps_its_own_extent_on_the_other_axis(frame):
    """Alignment is per-axis, not a shared rect.

    A top marginal that adopted the joint panel's y-extent too would be drawn
    on top of it. The marginal must keep its own band on the axis it does NOT
    share — which is also what distinguishes this from the Overlay pre-pass,
    where members share the whole rect.
    """
    extents = panel_extents(fm.jointplot(frame, x="x", y="y", kind="hist").to_svg())
    joint, top, right = joint_top_right(extents)
    assert extents[top][3] <= extents[joint][2] + EPS, (
        "the top marginal must sit entirely above the joint panel"
    )
    assert extents[right][0] >= extents[joint][1] - EPS, (
        "the right marginal must sit entirely to the right of the joint panel"
    )


def test_marginal_alignment_survives_a_colorbar_on_the_joint_panel(frame):
    """The discriminating case, and the one the golden caught.

    `kind="kde"` puts a colorbar in the joint panel's right gutter, so the
    joint panel's plot rect is ~77px narrower than the cell while the
    marginal's (which has no colorbar) is not. Any implementation that aligns
    by cell rather than by measured plot region passes the plain cases above
    and fails this one.
    """
    svg = fm.jointplot(frame, x="x", y="y", kind="kde").to_svg()
    extents = panel_extents(svg)
    joint, top, _right = joint_top_right(extents)
    joint_w = extents[joint][1] - extents[joint][0]
    top_w = extents[top][1] - extents[top][0]
    assert top_w == pytest.approx(joint_w, abs=EPS)
    # And the colorbar really is there, taking real space -- otherwise this
    # asserts the same thing as the plain case.
    assert 'id="ferrum-grad' in svg or "linearGradient" in svg, (
        "kind='kde' must draw a colorbar for this test to discriminate"
    )


def test_a_single_marginal_jointplot_aligns_too(frame):
    """`JointChart` lowers a one-marginal figure to a 2x1 / 1x2 grid with no
    hole, a different shape from the 2x2 both-marginals case. Both shapes are
    recognized, so the alignment cannot be a property of the 2x2 form alone."""
    from ferrum.composition import JointChart

    base = fm.Chart(frame).mark_point().encode(x="x", y="y")
    marginal = fm.Chart(frame).mark_bar().encode(x="x", y="y")
    top_only = JointChart(base, top=marginal)
    extents = panel_extents(top_only.to_svg())
    assert len(extents) == 2
    (_, a), (_, b) = sorted(extents.items(), key=lambda kv: kv[1][2])
    assert a[:2] == pytest.approx(b[:2], abs=EPS), (
        f"the top marginal's x-extent {a[:2]} must equal the joint panel's {b[:2]}"
    )


def test_an_empty_marginal_drops_its_declaration_rather_than_refusing(frame):
    """An empty-data marginal lowers to a HOLE, and a hole has no plot extent
    to adopt — so the declaration naming it is dropped at lowering.

    The two halves of that sentence must not be confused. The wire contract
    stays strict: a declaration naming a hole is malformed and Rust refuses it
    naming the entry, because no lowering may emit one. What happens here is
    the FIGURE changing shape under the lowering's feet (`_lower_leaf` turns a
    zero-row cell into a hole), which is the lowering's own business to
    reconcile. Before it did, this figure raised
    `ValueError: grid: extent_alignments[0] names a 'hole' cell` instead of
    rendering — caught by `test_composite_render_ratio.py`'s empty-marginal
    cases, and pinned here beside the contract it qualifies.
    """
    from ferrum.composition import JointChart

    center = fm.Chart(frame).mark_point().encode(x="x:Q", y="y:Q")
    empty_top = fm.Chart(frame.head(0)).mark_histogram().encode(x="x:Q")
    lowered = JointChart(center, top=empty_top)._composite_tree(auto_tooltips=False)

    assert [c["kind"] for c in lowered.tree["children"]] == ["hole", "leaf"]
    assert "extent_alignments" not in lowered.tree, (
        "the only declared follower became a hole, so the key is omitted entirely"
    )
    assert lowered.render_svg().startswith("<svg")

    # A live marginal beside an empty one keeps ITS declaration: the drop is
    # per-entry, not all-or-nothing.
    empty_right = fm.Chart(frame.head(0)).mark_histogram().encode(x="x:Q")
    right = fm.Chart(frame).mark_point().encode(x="x:Q", y="y:Q")
    both = JointChart(center, top=empty_right, right=right)._composite_tree(auto_tooltips=False)
    assert both.tree["extent_alignments"] == [{"follower": 3, "leader": 2, "axis": "y"}]


def test_a_layered_centre_warns_that_the_marginal_was_not_aligned(frame):
    """A `LayerChart` centre is a legitimate figure the renderer cannot align.

    Its cell spans a subtree of leaves laid out in its own frame, so there is
    no single plot rect for a marginal to adopt. The render proceeds and the
    marginal lays out on its own terms — degraded, not malformed — so under the
    campaign's REFUSE MALFORMED / WARN DEGRADED policy it must SAY so, naming
    the declaration it dropped (which follower, which leader, which axis). A
    figure with several declarations is otherwise told only that something was
    ignored.

    Asserted on the message's content through Python's own `warnings` channel,
    because that is the surface a caller reads: a `RenderWarning` variant that
    is defined and unit-tested but never reaches here would change nothing
    observable.
    """
    import warnings as _warnings

    from ferrum.composition import JointChart

    centre = fm.layer(
        fm.Chart(frame).mark_point().encode(x="x:Q", y="y:Q"),
        fm.Chart(frame).mark_line().encode(x="x:Q", y="y:Q"),
    )
    marginal = fm.Chart(frame).mark_histogram().encode(x="x:Q")

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        svg = JointChart(centre, top=marginal).to_svg()

    assert svg.startswith("<svg"), "the figure still renders in full"
    messages = [str(w.message) for w in caught]
    named = [m for m in messages if "cannot span" in m]
    assert len(named) == 1, f"exactly one dropped declaration, announced once: {messages}"
    assert "cell 0" in named[0] and "cell 1" in named[0], (
        f"the warning must name the follower and leader CELLS: {named[0]}"
    )
    assert "x extent" in named[0] and "nested composite" in named[0], (
        f"the warning must name the axis and why it could not be honored: {named[0]}"
    )
