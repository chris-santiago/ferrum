"""Behavior tests for the linear-form (HConcat/VConcat) composite render cutover.

Phase B Task 6 routes HConcat/VConcat static (``to_svg``) and interactive
(``_render_interactive``) rendering through the one-call Rust composite entries
(``render_composite_svg`` / ``render_composite_interactive``) instead of the
per-child string-compositor / scene-merge path.

These tests lock **parity through the new mechanism**.  Linear-form scale
sharing already worked via Phase A's ``_scale_share`` injection, so shared /
independent extents are not a RED→GREEN flip here — the point is that the same
observable behavior now flows through the composite tree + Rust resolve pass.
The RED-provable #45 remainder (grid-composite children) is Task 7's.

Rendered per-panel tick extents are the observable proxy for "these panels
share (or don't share) an axis domain" (the ``tests/_svg_extents.py`` helpers,
the same parsing the facet-shared-extent tests use).
"""

from __future__ import annotations

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


def test_heterogeneous_child_config_falls_back(small_chart, large_chart):
    """A child with its own annotation cannot cross the uniform entry -> old path.

    The uniform composite entry applies one ``chart_config`` to every leaf, so
    a per-child annotation would be lost; the composition keeps the legacy
    string-compositor path (``_lower_composite`` returns ``None``), and
    still renders valid SVG.
    """
    annotated = small_chart + annotate_hline(3.0)
    composite = annotated | large_chart
    assert _lower_composite(composite, auto_tooltips=False) is None
    svg = composite.to_svg()
    assert svg.startswith("<svg")


def test_configure_layer_composite_falls_back(small_chart, large_chart):
    """A composition-level configure layer keeps the old path (default chrome)."""
    composite = (small_chart | large_chart).configure_axis(label_angle=-45)
    assert _lower_composite(composite, auto_tooltips=False) is None
    assert composite.to_svg().startswith("<svg")


def test_non_xy_shared_channel_falls_back(small_df):
    """resolve with a shared non-positional channel stays on the injection path."""
    a = fm.Chart(small_df).mark_point().encode(x="x", y="y", color="x")
    b = fm.Chart(small_df).mark_point().encode(x="x", y="y", color="x")
    composite = HConcatChart([a, b], resolve={"color": "shared"})
    assert _lower_composite(composite, auto_tooltips=False) is None
    assert composite.to_svg().startswith("<svg")


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


def test_composite_resolve_field_positional_only():
    assert _composite_resolve_field(None) == {}
    assert _composite_resolve_field({}) == {}
    assert _composite_resolve_field({"x": "shared"}) == {"x": "shared"}
    assert _composite_resolve_field({"x": "shared", "y": "independent"}) == {
        "x": "shared",
        "y": "independent",
    }
    # A shared non-positional channel is not representable on the composite path.
    assert _composite_resolve_field({"color": "shared"}) is None
    # An independent non-positional channel is the default -> nothing to share.
    assert _composite_resolve_field({"color": "independent"}) == {}


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
