"""Behavior tests for the linear-form (HConcat/VConcat) composite render cutover.

Phase B Task 6 routes HConcat/VConcat static (``to_svg``) and interactive
(``_render_interactive``) rendering through the one-call Rust composite entries
(``render_composite_svg`` / ``render_composite_interactive``) instead of the
per-child string-compositor / scene-merge path.

These tests lock **parity through the new mechanism**.  Linear-form scale
sharing already worked via Phase A's legacy scale-share injection, so shared /
independent extents are not a RED→GREEN flip here — the point is that the same
observable behavior now flows through the composite tree + Rust resolve pass.
The RED-provable #45 remainder (grid-composite children) is Task 7's.

Rendered per-panel tick extents are the observable proxy for "these panels
share (or don't share) an axis domain" (the ``tests/_svg_extents.py`` helpers,
the same parsing the facet-shared-extent tests use).
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.annotations import annotate_hline
from ferrum.composition import (
    HConcatChart,
    VConcatChart,
    _composite_resolve_field,
    _lower_composite,
)

from tests._svg_extents import y_axis_extents


# ---------------------------------------------------------------------------
# Fixtures: two flat charts with disjoint y-ranges (1..4 vs 100..400).
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def large_df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [100.0, 200.0, 300.0, 400.0]})


@pytest.fixture
def small_chart(small_df):
    return fm.Chart(small_df).mark_point().encode(x="x", y="y")


@pytest.fixture
def large_chart(large_df):
    return fm.Chart(large_df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Flat children — shared vs independent rendered extents (new mechanism).
# ---------------------------------------------------------------------------


def test_hconcat_independent_y_isolates_extents(small_chart, large_chart):
    """Default (independent) HConcat: each panel keeps its own y domain."""
    svg = (small_chart | large_chart).to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    left, right = extents
    # Panels are isolated: the small panel tops out far below the large panel.
    assert left.hi <= 10.0
    assert right.hi >= 300.0
    assert left != right


def test_hconcat_shared_y_unifies_extents(small_chart, large_chart):
    """resolve={"y": "shared"} makes both HConcat panels render one y domain."""
    svg = HConcatChart([small_chart, large_chart], resolve={"y": "shared"}).to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    left, right = extents
    # Both panels now share one domain (identical rendered tick extents)...
    assert left == right
    # ...and that domain spans the union of the two source ranges.
    assert left.hi >= 400.0
    assert left.lo <= 1.0 or left.lo <= 50.0


def _stacked_x_row_extents(svg: str) -> list[tuple[float, float]]:
    """Per-panel x-axis (lo, hi) for a *vertically stacked* layout.

    The shared ``x_axis_extents`` helper assumes side-by-side column facets and
    keeps only the bottom-most tick row.  A VConcat stacks panels, so each
    panel's x-axis is its own tick row at a distinct y.  We group numeric
    ``<text>`` by rounded y, keep rows spanning ≥3 distinct x positions (an
    x-axis tick row), and return each row's value range top-to-bottom.
    """
    import re
    from collections import defaultdict

    rows: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
        try:
            val = float(text.strip())
        except ValueError:
            continue
        xm = re.search(r'x="([^"]+)"', attrs)
        ym = re.search(r'y="([^"]+)"', attrs)
        if xm and ym:
            rows[round(float(ym.group(1)))].append((float(xm.group(1)), val))
    axis_rows = [(y, ents) for y, ents in rows.items() if len({round(x) for x, _ in ents}) >= 3]
    axis_rows.sort()
    return [(min(v for _, v in ents), max(v for _, v in ents)) for _y, ents in axis_rows]


def test_vconcat_shared_x_unifies_extents():
    """resolve={"x": "shared"} unifies the x domain across stacked VConcat panels."""
    top_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    bottom_df = pl.DataFrame({"x": [100.0, 200.0, 300.0, 400.0], "y": [1.0, 2.0, 3.0, 4.0]})
    top = fm.Chart(top_df).mark_point().encode(x="x", y="y")
    bottom = fm.Chart(bottom_df).mark_point().encode(x="x", y="y")

    indep_x = _stacked_x_row_extents((top & bottom).to_svg())
    shared_x = _stacked_x_row_extents(VConcatChart([top, bottom], resolve={"x": "shared"}).to_svg())
    assert len(indep_x) == 2 and len(shared_x) == 2
    # Independent x differs between the stacked panels; shared x is identical.
    assert indep_x[0] != indep_x[1]
    assert shared_x[0] == shared_x[1]


# ---------------------------------------------------------------------------
# Composite-mark (box) children — the #45 slice, through the new mechanism.
# ---------------------------------------------------------------------------


@pytest.fixture
def box_small():
    df = pl.DataFrame({"g": ["a", "a", "a", "b", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_boxplot().encode(x="g", y="v")


@pytest.fixture
def box_large():
    df = pl.DataFrame(
        {"g": ["a", "a", "a", "b", "b", "b"], "v": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]}
    )
    return fm.Chart(df).mark_boxplot().encode(x="g", y="v")


def test_box_children_independent_isolates_extents(box_small, box_large):
    """Independent box-child concat: each box panel keeps its own y extent."""
    svg = (box_small | box_large).to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    assert extents[0] != extents[1]
    assert extents[0].hi <= 20.0
    assert extents[1].hi >= 300.0


def test_box_children_shared_y_unifies_extents(box_small, box_large):
    """Shared box-child concat: both box panels resolve one y domain (whisker union).

    This is the composite-mark half of #45: the shared domain derives from the
    box transform-output batches (whisker extents), not raw-column min/max, and
    is resolved by the Rust composite pass rather than Python injection.
    """
    svg = HConcatChart([box_small, box_large], resolve={"y": "shared"}).to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    assert extents[0] == extents[1]
    assert extents[0].hi >= 600.0


# ---------------------------------------------------------------------------
# Explicit user scale on a child overrides shared resolution.
# ---------------------------------------------------------------------------


def test_explicit_child_scale_overrides_shared(small_df, large_df):
    """A child's explicit ``scale=`` domain wins over resolve={"y":"shared"}."""
    pinned = (
        fm.Chart(small_df).mark_point().encode(x="x", y=fm.Y("y", scale={"domain": [0.0, 50.0]}))
    )
    other = fm.Chart(large_df).mark_point().encode(x="x", y="y")

    svg = HConcatChart([pinned, other], resolve={"y": "shared"}).to_svg()
    extents = y_axis_extents(svg)
    assert len(extents) == 2
    pinned_extent, other_extent = extents
    # The pinned child keeps its explicit [0, 50] domain — sharing did not
    # widen it to include the other panel's 100..400 range.
    assert pinned_extent.lo == 0.0
    assert pinned_extent.hi == 50.0
    assert pinned_extent != other_extent


# ---------------------------------------------------------------------------
# A no-resolve concat renders one <svg> with both panels present.
# ---------------------------------------------------------------------------


def test_no_resolve_concat_renders_single_svg_with_panels(small_df):
    """A plain HConcat renders a single well-formed SVG containing both panels."""
    a = fm.Chart(small_df).mark_point().encode(x="x", y="y")
    b = fm.Chart(small_df).mark_bar().encode(x="x", y="y")
    svg = (a | b).to_svg()
    assert svg.count("<svg") == 1
    assert svg.startswith("<svg")
    # Both marks are present: circles from the point panel, rects from the bar panel.
    assert "<circle" in svg
    assert "<rect" in svg
    assert "NaN" not in svg


# ---------------------------------------------------------------------------
# Routing: linear forms take the new composite path; other shapes fall back.
# ---------------------------------------------------------------------------


def test_hconcat_lowers_to_composite_tree(small_chart, large_chart):
    """An HConcat of flat charts lowers to a two-leaf hconcat tree."""
    lowered = _lower_composite(small_chart | large_chart, auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["kind"] == "composite"
    assert lowered.tree["layout"] == "hconcat"
    assert [c["kind"] for c in lowered.tree["children"]] == ["leaf", "leaf"]
    assert len(lowered.payloads) == 2


def test_vconcat_lowers_with_vconcat_layout(small_chart, large_chart):
    lowered = _lower_composite(small_chart & large_chart, auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["layout"] == "vconcat"


def test_resolve_maps_onto_tree_resolve_field(small_chart, large_chart):
    lowered = _lower_composite(
        HConcatChart([small_chart, large_chart], resolve={"y": "shared"}),
        auto_tooltips=False,
    )
    assert lowered is not None
    assert lowered.tree["resolve"] == {"y": "shared"}


def test_root_chrome_lands_on_tree_root(small_chart, large_chart):
    lowered = _lower_composite(
        (small_chart & large_chart).properties(title="T", subtitle="S", caption="C"),
        auto_tooltips=False,
    )
    assert lowered is not None
    assert lowered.tree["title"] == "T"
    assert lowered.tree["subtitle"] == "S"
    assert lowered.tree["caption"] == "C"


def test_nested_linear_composite_lowers_recursively(small_chart, large_chart):
    """(a | b) & c lowers to a nested composite tree (vconcat of hconcat + leaf)."""
    nested = (small_chart | large_chart) & small_chart
    lowered = _lower_composite(nested, auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["layout"] == "vconcat"
    kinds = [c["kind"] for c in lowered.tree["children"]]
    assert kinds == ["composite", "leaf"]
    assert lowered.tree["children"][0]["layout"] == "hconcat"
    assert len(lowered.payloads) == 3


def _vertical_gridline_x_positions(svg: str) -> list[float]:
    """X positions of every vertical panel gridline (``x1 == x2``, gridline stroke).

    Each panel draws its own column of vertical gridlines spanning the full
    plot height, clustered around that panel's own x-range with a wide gap to
    the next panel's cluster. Used by ``_panel_boundary_x`` to locate the
    HConcat panel boundary without hardcoding pixel geometry.
    """
    xs = []
    for m in re.finditer(r'<line\s+([^>]*)stroke="#d6d3d1"[^>]*/>', svg):
        attrs = m.group(1)
        x1 = float(re.search(r'x1="([^"]+)"', attrs).group(1))
        x2 = float(re.search(r'x2="([^"]+)"', attrs).group(1))
        if x1 == x2:
            xs.append(x1)
    return xs


def _panel_boundary_x(svg: str) -> float:
    """Midpoint of the widest gap between clustered vertical-gridline x positions.

    For a two-panel side-by-side HConcat, this is the x boundary between the
    left and right panel's plot areas: everything left of it belongs to the
    left child, everything at or right of it belongs to the right child.
    """
    xs = sorted(set(_vertical_gridline_x_positions(svg)))
    assert len(xs) >= 2, "expected vertical gridlines from at least two panels"
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gap, i = max(gaps)
    return (xs[i] + xs[i + 1]) / 2


def test_heterogeneous_child_config_lowers_with_per_leaf_binding(small_chart, large_chart):
    """A child with its own annotation now lowers via per-leaf binding (Task 5d).

    The composite entry no longer requires a single ``chart_config`` for every
    leaf: a leaf whose ``chart_config`` differs carries its own override on the
    tree node, and a leaf with none carries an explicit empty override rather
    than an absent key -- an absent key would mean "inherit the call-level
    default" (leaf 0's config), silently bleeding leaf 0's annotation onto
    every unconfigured sibling.
    """
    annotated = small_chart + annotate_hline(3.0)
    composite = annotated | large_chart

    lowered = _lower_composite(composite, auto_tooltips=False)
    assert lowered is not None
    # The annotated leaf carries a per-leaf chart_config with the annotation;
    # the plain leaf carries an explicit empty override, not an absent key.
    annotated_leaf, plain_leaf = lowered.tree["children"]
    assert annotated_leaf["chart_config"]["annotations"], "annotation dropped from leaf binding"
    assert plain_leaf["chart_config"] == {}, "plain leaf must carry an explicit empty override"

    # The annotation renders overall: the composite gains the hline vs. an
    # un-annotated pair.
    with_anno = composite.to_svg()
    without_anno = (small_chart | large_chart).to_svg()
    assert with_anno.startswith("<svg")
    assert with_anno.count("<line") > without_anno.count("<line")

    # Discriminating check: the annotation line renders only in the annotated
    # (left) leaf's panel region, never in the plain (right) sibling's panel --
    # this is what would silently break if the plain leaf inherited leaf 0's
    # chart_config instead of its own empty override.
    boundary = _panel_boundary_x(with_anno)
    annotation_lines = re.findall(r'<line\s+([^>]*)stroke="#333333"[^>]*/>', with_anno)
    assert annotation_lines, "expected the hline annotation to render at least one <line>"
    for attrs in annotation_lines:
        x1 = float(re.search(r'x1="([^"]+)"', attrs).group(1))
        x2 = float(re.search(r'x2="([^"]+)"', attrs).group(1))
        assert x1 < boundary and x2 <= boundary, (
            f"annotation line ({x1}, {x2}) bled into the plain sibling's panel "
            f"(boundary={boundary})"
        )


def test_configure_layer_composite_lowers_via_composite_path(small_chart, large_chart):
    """A composition-level configure layer now lowers via the composite path.

    Task 10-pre-a sub-task 4: ``_lower_any`` pushes the composite's
    ``_configure_layers`` onto every child via ``_inject_parent_config`` before
    lowering (the mechanism JointChart/ClusterMapChart already relied on), so
    each leaf's own ``chart_config`` carries the composite-level configure --
    no separate gate/fallback is needed. RED before this task (the gate
    unconditionally returned ``None``); GREEN after.
    """
    composite = (small_chart | large_chart).configure_axis(label_angle=-45)
    lowered = _lower_composite(composite, auto_tooltips=False)
    assert lowered is not None
    # Both children get the SAME composite-level configure injected, so their
    # chart_configs are identical and the uniform-tree optimization keeps the
    # override on the call-level default rather than duplicating it per leaf
    # (see _apply_leaf_binding_overrides).
    assert lowered.chart_config["axis"]["label_angle"] == -45

    svg = composite.to_svg()
    assert svg.startswith("<svg")


def _point_fill_colors(svg: str) -> list[str]:
    """Return the ``fill`` of every DATA-point ``<circle>``, excluding legend swatches.

    Legend category swatches also render as ``<circle>`` elements but at the
    legend's fixed marker radius (``r="4"``), distinct from a mark_point data
    circle's data-driven radius -- excluding them keeps this a pure per-point
    color check.
    """
    return [
        fill
        for attrs, fill in re.findall(r'<circle\s+([^>]*)fill="([^"]+)"', svg)
        if 'r="4"' not in attrs
    ]


def test_hconcat_resolve_shared_color_unions_categorical_domain(small_df, large_df):
    """resolve={"color": "shared", ...} unions the categorical color domain.

    Task 10-pre-a sub-task 1: ``_composite_resolve_field`` now passes ``color``/
    ``size`` through to the Rust composite resolve pass (10-pre-b) instead of
    forcing the whole tree to the legacy scale-share injection path. RED
    before this task (any non-x/y shared channel returned ``None``, forcing
    ``_lower_composite`` to decline); GREEN after.

    Two children whose ``color``-encoded column has disjoint category sets:
    with color shared, the SAME category string gets the SAME fill color in
    both panels' rendered points (a discriminating check -- independent
    per-panel categorical domains would coincidentally reuse the same palette
    index-0/1 colors for each panel's own 2 categories, so plain color-set
    equality would not distinguish shared from independent; requiring the
    first point of each panel to match, and the second of each panel to
    match, pins the actual union mapping).
    """
    left_df = small_df.with_columns(pl.Series("cat", ["a", "b", "a", "b"]))
    right_df = large_df.with_columns(pl.Series("cat", ["c", "d", "c", "d"]))
    left = fm.Chart(left_df).mark_point().encode(x="x", y="y", color="cat")
    right = fm.Chart(right_df).mark_point().encode(x="x", y="y", color="cat")
    # x is ALSO shared here; the standalone color-only case is covered by
    # test_hconcat_resolve_shared_color_alone_unions_domain below.
    composite = HConcatChart([left, right], resolve={"color": "shared", "x": "shared"})

    lowered = _lower_composite(composite, auto_tooltips=False)
    assert lowered is not None, "color-shared resolve must lower via the composite path"
    assert lowered.tree["resolve"] == {"color": "shared", "x": "shared"}

    svg = composite.to_svg()
    fills = _point_fill_colors(svg)
    assert len(fills) == 8, f"expected 4 points per panel, got {fills}"
    left_fills, right_fills = fills[:4], fills[4:]
    # "a"/"c" are both the first category encountered in their own panel;
    # under a shared (union-ordered) domain they land on DIFFERENT palette
    # entries (a=index0, c=index2), unlike an independent per-panel domain
    # where both would collapse onto local index0.
    assert left_fills[0] != right_fills[0], (
        f"'a' and 'c' must get different colors under a shared union domain: {fills}"
    )
    assert left_fills[1] != right_fills[1], (
        f"'b' and 'd' must get different colors under a shared union domain: {fills}"
    )


def test_hconcat_resolve_shared_color_alone_unions_domain(small_df, large_df):
    """Standalone (no x/y sharing) color-shared resolve unions the domain.

    Regression: composite_render.rs's per-leaf ctx gate originally checked
    only x/y presence before threading the resolved LeafScaleContext into a
    leaf's render, so a color/size-ONLY shared resolve was silently discarded
    (discovered while widening the Python gate in Task 10-pre-a; fixed via
    LeafScaleContext::is_empty). The sibling test above covers the combined
    positional+color case.
    """
    left_df = small_df.with_columns(pl.Series("cat", ["a", "b", "a", "b"]))
    right_df = large_df.with_columns(pl.Series("cat", ["c", "d", "c", "d"]))
    left = fm.Chart(left_df).mark_point().encode(x="x", y="y", color="cat")
    right = fm.Chart(right_df).mark_point().encode(x="x", y="y", color="cat")
    composite = HConcatChart([left, right], resolve={"color": "shared"})

    svg = composite.to_svg()
    fills = _point_fill_colors(svg)
    left_fills, right_fills = fills[:4], fills[4:]
    assert left_fills[0] != right_fills[0]


# ---------------------------------------------------------------------------
# Interactive path routes through the composite entry.
# ---------------------------------------------------------------------------


def test_hconcat_interactive_routes_through_composite_entry(small_chart, large_chart):
    """The interactive render produces one scene with both panels via the new entry."""
    import json

    composite = small_chart | large_chart
    assert _lower_composite(composite, auto_tooltips=True) is not None
    scene_json, packed = composite._render_interactive()
    scene = json.loads(scene_json)
    assert len(scene["panels"]) == 2
    assert isinstance(packed, (bytes, bytearray))


# ---------------------------------------------------------------------------
# _composite_resolve_field unit coverage.
# ---------------------------------------------------------------------------


def test_composite_resolve_field_positional_and_color_size():
    kind = {"kind": "hconcat"}
    assert _composite_resolve_field(None, **kind) == {}
    assert _composite_resolve_field({}, **kind) == {}
    assert _composite_resolve_field({"x": "shared"}, **kind) == {"x": "shared"}
    assert _composite_resolve_field({"x": "shared", "y": "independent"}, **kind) == {
        "x": "shared",
        "y": "independent",
    }
    # color/size are supported (10-pre-b Rust support + Task 10-pre-a gate drop).
    assert _composite_resolve_field({"color": "shared"}, **kind) == {"color": "shared"}
    assert _composite_resolve_field({"size": "shared"}, **kind) == {"size": "shared"}
    assert _composite_resolve_field({"color": "independent"}, **kind) == {"color": "independent"}
    # A shared channel outside {x, y, color, size} is not representable: with
    # no legacy path left to defer to (Task 10), it raises a typed error
    # naming the composition node kind and the offending channel.
    with pytest.raises(ValueError, match=r"hconcat: resolve= marks 'shape' 'shared'"):
        _composite_resolve_field({"shape": "shared"}, **kind)
    # An independent unsupported channel is the default -> nothing to share.
    assert _composite_resolve_field({"shape": "independent"}, **kind) == {}


# ---------------------------------------------------------------------------
# Error propagation: the composite entry surfaces malformed trees as ValueError
# naming the offending node kind (matches the render-error idiom).
# ---------------------------------------------------------------------------


def _one_leaf_inputs(chart):
    """Return (spec, payload) for a single flat chart via its render inputs."""
    spec, data, _viewport, _theme, _cc = chart._render_inputs()
    return spec, data


def test_empty_children_composite_raises_value_error(small_chart):
    from ferrum._core import render_composite_svg

    spec, data = _one_leaf_inputs(small_chart)
    tree = {"kind": "composite", "layout": "hconcat", "children": []}
    with pytest.raises(ValueError, match="hconcat"):
        render_composite_svg(tree, [data], viewport=(640.0, 480.0))


def test_leaf_data_index_out_of_bounds_raises(small_chart):
    from ferrum._core import render_composite_svg

    spec, data = _one_leaf_inputs(small_chart)
    tree = {
        "kind": "composite",
        "layout": "hconcat",
        "children": [{"kind": "leaf", "spec": spec, "data": 7}],
    }
    with pytest.raises(ValueError):
        render_composite_svg(tree, [data], viewport=(640.0, 480.0))


def test_unknown_node_kind_raises(small_chart):
    from ferrum._core import render_composite_svg

    spec, data = _one_leaf_inputs(small_chart)
    tree = {"kind": "spaghetti", "spec": spec, "data": 0}
    with pytest.raises(ValueError, match="spaghetti"):
        render_composite_svg(tree, [data], viewport=(640.0, 480.0))
