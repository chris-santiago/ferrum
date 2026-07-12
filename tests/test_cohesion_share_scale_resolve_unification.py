"""Regression tests — share_scale() unified onto the resolve= mechanism (#45 close).

Phase B's verification-before-completion gate found ``share_scale()``/``expand()``
still computed a shared scale by walking each member chart's RAW data column
(``compute_union_domain`` + ``inject_scale``, injecting an explicit ``scale=``
dict into each chart's encoding), while ``resolve=`` passed at construction time
already routed through the Rust composite resolve pass -- a transform-aware
union computed *after* each panel's stat transform runs. Two public routes to
"shared scales" therefore could render DIFFERENT axes for the same intent, the
exact drift class #45 fixed elsewhere in this phase.

Fix: ``share_scale()`` is now pure sugar for ``resolve=`` -- it merges the
requested channels into the composition's ``_resolve`` dict and reconstructs
via the same constructor ``resolve=`` already uses, so both entry points are
byte-identical at render time. No ``scale=`` dict is computed or injected by
``share_scale()``/``expand()`` anymore.

This file's first test is the RED-proof: box-mark panels with disjoint value
ranges, where the OLD raw-column injection and the Rust resolve pass computed
*different* axis geometry (same labeled tick VALUES, but different pixel
gridline positions -- confirmed empirically via ``git stash`` against HEAD
before this fix landed). Proven RED against HEAD by stashing only
``src/ferrum/composition.py`` (flags before ``--``, immediately reverted after
recording the failure) and rerunning this test; it failed there, passes here.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import (
    ClusterMapChart,
    HConcatChart,
    JointChart,
    LayerChart,
    RepeatChart,
    Resolve,
)
from tests._svg_extents import y_axis_extents


# ---------------------------------------------------------------------------
# THE regression test: share_scale(y="shared") must render byte-identical to
# resolve={"y": "shared"} on box-mark (stat-transform) panels.
# ---------------------------------------------------------------------------


@pytest.fixture
def box_small():
    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    return fm.Chart(df).mark_boxplot().encode(x="g", y="v")


@pytest.fixture
def box_large():
    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "v": [200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0],
        }
    )
    return fm.Chart(df).mark_boxplot().encode(x="g", y="v")


def test_share_scale_renders_identically_to_resolve_construction(box_small, box_large):
    """share_scale(y='shared') and resolve={'y': 'shared'} must render the same SVG.

    Both must union the box mark's whisker/outlier reach, not stop at raw
    column min/max read in Python -- there is no separate injection path left
    for ``share_scale`` to diverge through.
    """
    via_share_scale = (box_small | box_large).share_scale(y="shared")
    via_resolve = HConcatChart([box_small, box_large], resolve={"y": "shared"})

    assert via_share_scale.to_svg() == via_resolve.to_svg()

    extents = y_axis_extents(via_share_scale.to_svg())
    assert len(extents) == 2
    assert extents[0] == extents[1]
    # The shared domain must cover the large panel's whisker reach, not just
    # collapse to whatever the small panel alone would show.
    assert extents[0].hi >= 260.0


def test_share_scale_children_carry_no_injected_scale(box_small, box_large):
    """share_scale() must not write a scale= dict onto any member chart's encoding."""
    result = (box_small | box_large).share_scale(y="shared")
    for child in result.charts:
        assert child._encoding["y"]._kwargs.get("scale") is None
    assert result._resolve == Resolve(scale={"y": "shared"})


# ---------------------------------------------------------------------------
# Unsupported forms: JointChart / ClusterMapChart carry no resolve= field at
# all, so share_scale must raise a typed error rather than silently
# performing the old (now-removed) raw-column injection.
# ---------------------------------------------------------------------------


def test_joint_chart_share_scale_raises_unsupported():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 50.0, 90.0], "ym": [5.0, 25.0, 75.0]})
    center = fm.Chart(df).mark_point().encode(x="x", y="y")
    right = fm.Chart(df).mark_point().encode(x="x", y="ym")
    joint = JointChart(center, right=right)
    with pytest.raises(ValueError, match="JointChart.*resolve="):
        joint.share_scale(y="shared")


def test_clustermap_share_scale_raises_unsupported():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "r": [0, 1, 2]})
    heat = fm.Chart(df).mark_rect().encode(x="a", y="r", fill="b")
    col_d = fm.Chart(df).mark_rule().encode(x="b", y="r")
    cm = ClusterMapChart(heat, col_dendrogram=col_d)
    with pytest.raises(ValueError, match="ClusterMapChart.*resolve="):
        cm.share_scale(x="shared")


# ---------------------------------------------------------------------------
# LayerChart: explicit x "independent" is a typed error (overlay contract,
# dual-x-axis is GH #55), via both the constructor and the share_scale
# sugar; y "independent" is a supported secondary-axis feature (GH #52,
# see tests/test_secondary_y_axis.py for its structural coverage) and
# color/size independent stays legal.
# ---------------------------------------------------------------------------


def test_layer_chart_constructor_rejects_independent_x():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y")
    b = fm.Chart(df).mark_line().encode(x="x", y="y")
    with pytest.raises(ValueError, match="LayerChart.*overlay contract.*#55"):
        LayerChart(a, b, resolve={"x": "independent"})


def test_layer_chart_constructor_accepts_independent_y():
    """resolve={'y': 'independent'} no longer raises -- it renders a dual-axis chart (GH #52)."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y")
    b = fm.Chart(df).mark_line().encode(x="x", y="y")
    layered = LayerChart(a, b, resolve={"y": "independent"})
    assert layered._resolve == {"y": "independent"}
    layered.to_svg()  # must render without raising
    layered._render_interactive()  # must render without raising


def test_layer_chart_share_scale_rejects_independent_x():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y")
    b = fm.Chart(df).mark_line().encode(x="x", y="y")
    layered = LayerChart(a, b)
    with pytest.raises(ValueError, match="LayerChart.*overlay contract.*#55"):
        layered.share_scale(x="independent")


def test_layer_chart_share_scale_accepts_independent_y():
    """share_scale(y='independent') is resolve= sugar -- same rule as the constructor."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y")
    b = fm.Chart(df).mark_line().encode(x="x", y="y")
    layered = LayerChart(a, b).share_scale(y="independent")
    assert layered._resolve == Resolve(scale={"y": "independent"})
    layered.to_svg()


def test_layer_chart_color_independent_stays_legal():
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "c": ["p", "q"]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y", color="c")
    b = fm.Chart(df).mark_line().encode(x="x", y="y")
    # Constructor route.
    layered = LayerChart(a, b, resolve={"color": "independent"})
    assert layered._resolve == {"color": "independent"}
    layered.to_svg()  # must render without raising

    # share_scale sugar route.
    shared = LayerChart(a, b).share_scale(color="independent")
    assert shared._resolve == Resolve(scale={"color": "independent"})
    shared.to_svg()


# ---------------------------------------------------------------------------
# RepeatChart.expand() no longer injects a resolve scale.
# ---------------------------------------------------------------------------


def test_repeat_expand_never_injects_scale():
    df = pl.DataFrame({"small": [0.0, 1.0, 2.0], "big": [10.0, 20.0, 30.0], "y": [1, 2, 3]})
    tpl = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y="y")
    rc = RepeatChart(tpl, column=["small", "big"], resolve={"x": "shared"})
    for _, _, chart in rc.expand():
        assert chart._encoding["x"]._kwargs.get("scale") is None
    # Rendering still shares the axis -- the resolve field rides the
    # composite tree, not expand()'s materialized panels.
    lowered = rc._composite_tree(auto_tooltips=False)
    assert lowered.tree["resolve"] == {"x": "shared"}
