"""Regression tests for finding P2 (design review, 2026-08-27) and GH #89A.

Static ``LayerChart`` under the default/shared-``y`` resolve mode used to
render every axis line, tick label, grid line, and chart title once PER
LAYER instead of once for the whole overlay: a 2-layer ``LayerChart`` emitted
92 ``<line>`` elements (46 distinct coordinate tuples, every one drawn
twice) and 48 ``<text>`` elements (24 distinct), and two layers binding
different ``y`` fields overprinted both axis titles at the same origin.

Root cause and fix: each overlay child used to render full standalone chrome
over its own independently computed panel rect. Now
(``crates/ferrum-core/src/render/composite_render.rs``) a pre-pass computes
ONE shared plot rect per overlay group -- the intersection of the layers'
natural plot regions, i.e. per side the largest gutter any layer reserves --
and every layer lays out against it, so all layout products (panel rects,
tick pixel positions, axis titles, legend placement) describe that one rect.
``merge_children`` then drops every non-primary layer's duplicate chrome
(per-panel ``grid``/``axes``/``chrome_above`` plus the scene-level ``title``);
content (``marks``, ``below_marks``/``annotations``, ``strip_title``,
``legend``) is never touched. The pre-pass is *suppression-aware* and
*coupled* to that drop: a non-primary layer's title band is excluded from the
shared rect (it is never drawn), and chrome is dropped only for layers that
actually laid out against the shared rect.

GH #89A retired the three refusal doors the first fix shipped with (a
per-leaf legend, a ``zindex >= 1`` axis, or a below-marks annotation on a
non-primary layer each used to disable dedup, leaving duplicate chrome). The
only shape still exempt is an overlay with a nested *composite* child, which
is unreachable from Python: ``LayerChart`` rejects a non-leaf layer.

Acceptance covered here:
    1. ``LayerChart(point, line).to_svg()`` element counts equal the flat
       ``(point + line).to_svg()`` path, and no ``<line>`` coordinate tuple
       repeats (the 92/46 -> 46/46 case).
    2. Two layers binding different ``y`` fields: exactly one y-axis title
       (layer 0's field), the second layer's field name never appears as an
       axis-title text node, and its marks still render.
    3. A ``LayerChart`` nested inside a grid composition shows no duplicated
       axis chrome within the layer panel.
    4. ``resolve={"y": "independent"}`` is unaffected (it already took the
       flat merged path, not the overlay composite tree the fix touches).
    5. Totality (#89A spec §9.2): a color-encoded non-primary layer dedups
       its chrome AND renders its legend.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import ferrum as fm

_LINE_RE = re.compile(r"<line\b[^>]*/?>")
_LINE_COORDS_RE = re.compile(r'x1="([^"]*)" y1="([^"]*)" x2="([^"]*)" y2="([^"]*)"')
_TEXT_RE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.DOTALL)


def _line_elements(svg: str) -> list[str]:
    return _LINE_RE.findall(svg)


def _line_coord_tuples(svg: str) -> list[tuple[str, str, str, str]]:
    """Extract the (x1, y1, x2, y2) tuple for every ``<line>`` element."""
    coords = []
    for element in _line_elements(svg):
        match = _LINE_COORDS_RE.search(element)
        if match is not None:
            coords.append(match.groups())
    return coords


def _text_contents(svg: str) -> list[str]:
    return _TEXT_RE.findall(svg)


def _df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0],
        }
    )


def _grouped_df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.0, 5.0, 3.0],
            "g": ["a", "b", "a", "b", "a"],
        }
    )


def _dual_y_df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_alpha": [2.0, 4.0, 1.0, 5.0, 3.0],
            "y_beta": [10.0, 8.0, 12.0, 7.0, 9.0],
        }
    )


def test_layer_chart_matches_flat_merge_element_counts():
    """LayerChart(point, line) chrome must match the flat (point + line) path.

    Before the fix: 92 <line>/46 distinct, 48 <text>/24 distinct (everything
    drawn twice). After the fix: both element counts equal the flat merge,
    and every <line> in the overlay output is unique.
    """
    df = _df()
    point = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_point()
    line = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_line()

    layered_svg = fm.LayerChart(point, line).to_svg()
    flat_svg = (point + line).to_svg()

    layered_lines = _line_elements(layered_svg)
    flat_lines = _line_elements(flat_svg)
    assert len(layered_lines) == len(flat_lines), (
        f"LayerChart drew {len(layered_lines)} <line> elements; flat merge drew {len(flat_lines)}"
    )

    layered_texts = _text_contents(layered_svg)
    flat_texts = _text_contents(flat_svg)
    assert len(layered_texts) == len(flat_texts), (
        f"LayerChart drew {len(layered_texts)} <text> elements; flat merge drew {len(flat_texts)}"
    )

    coord_tuples = _line_coord_tuples(layered_svg)
    assert len(coord_tuples) == len(set(coord_tuples)), (
        "LayerChart <line> coordinates must be unique; found duplicates: "
        f"{[c for c in coord_tuples if coord_tuples.count(c) > 1]}"
    )


def test_layer_chart_two_y_fields_emits_only_layer_zero_title():
    """Layers binding different y fields must not overprint both axis titles.

    Layer 0 encodes ``y_alpha``, layer 1 encodes ``y_beta`` on a line mark.
    Only "y_alpha" (layer 0's field) may appear as axis-title text; "y_beta"
    must never appear as a text node, even though layer 1's marks still
    render (its own y-scale/domain still participates in the shared union).
    """
    df = _dual_y_df()
    layer0 = fm.Chart(df).encode(x="x:Q", y="y_alpha:Q").mark_point()
    layer1 = fm.Chart(df).encode(x="x:Q", y="y_beta:Q").mark_line()

    svg = fm.LayerChart(layer0, layer1).to_svg()
    texts = _text_contents(svg)

    assert texts.count("y_alpha") == 1, f"expected exactly one 'y_alpha' title, got {texts}"
    assert "y_beta" not in texts, f"'y_beta' must not appear as an axis-title text node: {texts}"

    # Layer 1's marks (the line) must still render despite its chrome being
    # suppressed -- chrome suppression must not drop layer content.
    assert "<polyline" in svg, "layer 1's line mark must still render"
    assert svg.count("<circle") == len(df), "layer 0's point mark must still render"


def test_layer_chart_nested_in_grid_has_no_duplicated_chrome():
    """A LayerChart nested in a grid composition must not duplicate its axes.

    ``(LayerChart(a, b) | c).to_svg()`` places two panels side by side: the
    overlay panel (a, b sharing one rect) and c's own panel. Panel offsets
    differ, so genuinely distinct panels never collide on <line> coordinates
    -- any duplicate coordinate tuple in the composed output can only come
    from the overlay panel re-drawing its own chrome, which is exactly what
    finding P2 reported.
    """
    df = _df()
    a = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_point()
    b = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_line()
    c = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_bar()

    composed = fm.LayerChart(a, b) | c
    svg = composed.to_svg()

    coord_tuples = _line_coord_tuples(svg)
    assert len(coord_tuples) == len(set(coord_tuples)), (
        "grid-nested LayerChart panel must not duplicate axis/grid lines; "
        f"duplicates: {[c for c in coord_tuples if coord_tuples.count(c) > 1]}"
    )


def test_layer_chart_independent_y_unaffected():
    """resolve={"y": "independent"} already used the flat merged path.

    This is not the overlay composite tree the P2 fix touches, so its output
    is unaffected by the fix. We assert internal consistency (no duplicate
    <line> coordinates, both y-axis titles present by design since dual-axis
    is intentional here) rather than a byte comparison against a stale
    snapshot, per the plan's global-constraints note.
    """
    df = _dual_y_df()
    layer0 = fm.Chart(df).encode(x="x:Q", y="y_alpha:Q").mark_point()
    layer1 = fm.Chart(df).encode(x="x:Q", y="y_beta:Q").mark_line()

    svg = fm.LayerChart(layer0, layer1, resolve={"y": "independent"}).to_svg()
    texts = _text_contents(svg)

    # Dual-axis is intentional here: both fields legitimately get their own
    # axis title, unlike the shared-y overlay case above.
    assert "y_alpha" in texts
    assert "y_beta" in texts

    coord_tuples = _line_coord_tuples(svg)
    assert len(coord_tuples) == len(set(coord_tuples)), (
        "independent-y LayerChart must not duplicate <line> coordinates either"
    )

    # The interactive path renders through the same merged flat chart
    # regardless of resolve, and must keep working after the P2 fix (which
    # only touches the overlay composite-tree merge, not this path).
    interactive = fm.LayerChart(layer0, layer1, resolve={"y": "independent"}).interactive()
    assert interactive._scene_json
    assert isinstance(interactive._packed_data, bytes)


def test_layer_chart_per_leaf_legend_dedups_chrome_and_keeps_the_legend():
    """A layer carrying its own legend dedups its chrome AND renders it.

    This is GH #89A acceptance §9.2, and the shape that used to be refusal
    door 1: the color layer's legend gutter made its natural rect differ from
    the line layer's, so the old fix refused to equalize them and therefore
    refused to dedup, leaving BOTH layers' axes and grids in the output. With
    one shared rect (the intersection -- so the legend gutter is reserved for
    the whole group) the refusal is unnecessary and gone.

    Discriminators, in increasing strength:
      * no ``<line>`` coordinate tuple repeats -- the chrome is drawn once;
      * the legend still renders -- dedup dropped chrome, not content;
      * the chrome geometry is byte-identical to the flat ``line + points``
        merge, i.e. the overlay reserves the legend gutter exactly once and
        places every axis/grid line where a single chart would. A dedup that
        forgot to share the rect would drop a layer's chrome while leaving
        its marks at their own (wider, gutter-free) geometry, and these
        coordinates would not match;
      * swapping the layer order leaves the chrome geometry unchanged --
        the shared rect is the layers' INTERSECTION, not "whatever layer 0
        happened to compute".
    """
    df = _grouped_df()
    line = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_line()
    points = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_point().encode(color="g:N")

    svg = fm.LayerChart(line, points).to_svg()
    coord_tuples = _line_coord_tuples(svg)
    texts = _text_contents(svg)

    assert len(coord_tuples) == len(set(coord_tuples)), (
        "the legend-bearing layer's chrome must be deduplicated; duplicates: "
        f"{[c for c in coord_tuples if coord_tuples.count(c) > 1]}"
    )
    assert "g" in texts, "the color legend's title ('g') must render"
    assert "a" in texts and "b" in texts, "the color legend's category swatches must render"

    flat_svg = (line + points).to_svg()
    assert set(coord_tuples) == set(_line_coord_tuples(flat_svg)), (
        "the overlay's chrome must land exactly where the flat merge's does -- "
        "otherwise the shared rect did not reserve the legend gutter for the group"
    )
    assert sorted(texts) == sorted(_text_contents(flat_svg))

    swapped = fm.LayerChart(points, line).to_svg()
    assert set(coord_tuples) == set(_line_coord_tuples(swapped)), (
        "the shared rect is the intersection of the layers' natural rects, so "
        "layer order must not move any chrome"
    )


def test_layer_chart_degrades_to_per_layer_chrome_when_the_shared_rect_collapses():
    """The sanctioned degradation arm of GH #89A spec §4.2's coupling bullet.

    Chrome is deduplicated only for layers that actually laid out against the
    shared rect. When the layers' natural plot regions are *disjoint* -- here
    each layer reserves a y-axis band wider than half the canvas, on opposite
    sides -- their per-side-max intersection degenerates, the pre-pass imposes
    nothing, and the merge seam must therefore suppress nothing. Every layer
    keeps its own geometry AND its own chrome (including its own title): dual
    chrome is the honest outcome, and is strictly better than chrome describing
    a rect the marks never used.

    The degradation is *announced*: a ``RenderWarning::OverlayGuttersDiverged``
    reaches Python through the normal ``warnings.warn`` channel, naming the
    cause ("overlay gutters diverged; N layers render with independent
    chrome") so a doubled-chrome chart is diagnosable rather than merely
    visible. The dedup control must stay silent.

    Note the geometric discriminator is the element *count*, not duplicate
    coordinates: in this arm the two chromes sit at genuinely different rects,
    so their ``<line>`` tuples do not repeat -- that is exactly what "no shared
    rect" means. The ``band=40`` control renders the same two layers with
    overlapping reservations and shows the normal deduplicated result, so this
    test cannot pass by accident on a build that never dedups at all.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 1.0]})

    def _layers(band: float):
        left = (
            fm.Chart(df)
            .encode(x="x:Q", y=fm.Y("y:Q", axis=fm.Axis(min_band=band)))
            .mark_line()
            .properties(width=400, height=240)
        )
        right = (
            fm.Chart(df)
            .encode(x="x:Q", y=fm.Y("y:Q", axis=fm.Axis(orient="right", min_band=band)))
            .mark_point()
            .properties(width=400, height=240, title="LAYER2")
        )
        return left, right

    # Disjoint: each layer reserves 220px of a 400px canvas, on opposite sides.
    collapsed_left, collapsed_right = _layers(220.0)
    with pytest.warns(UserWarning, match="overlay gutters diverged") as caught:
        collapsed = fm.LayerChart(collapsed_left, collapsed_right).to_svg()
    assert "2 layers render with independent chrome" in str(caught[0].message), (
        "the warning must name the cause and the group size, not just that something "
        f"happened: {caught[0].message}"
    )
    one_layer_chrome = len(_line_coord_tuples(collapsed_left.to_svg()))

    assert len(_line_coord_tuples(collapsed)) == 2 * one_layer_chrome, (
        "both layers must keep their own chrome when no shared rect was imposed"
    )
    assert "LAYER2" in collapsed, (
        "the non-primary layer's title must render too -- nothing about this layer "
        "was suppressed, so its title band is real"
    )
    assert collapsed.count("<circle") == len(df), "both layers' marks still render"

    # Control: the same two layers with overlapping reservations equalize and
    # dedup normally -- one chrome, non-primary title dropped, and NO warning
    # (the warning must mark the degradation, not every overlay).
    overlap_left, overlap_right = _layers(40.0)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        overlapping = fm.LayerChart(overlap_left, overlap_right).to_svg()
    assert not [w for w in recorded if "overlay gutters diverged" in str(w.message)], (
        f"the normal dedup path must not warn: {[str(w.message) for w in recorded]}"
    )
    assert len(_line_coord_tuples(overlapping)) == len(_line_coord_tuples(overlap_left.to_svg())), (
        "overlapping reservations must still produce exactly one chrome"
    )
    assert "LAYER2" not in overlapping


def test_layer_chart_non_primary_title_reserves_no_phantom_band():
    """A title on a non-primary layer is dropped, so it must reserve nothing.

    Suppression-aware pre-pass (GH #89A spec §4.2, amended 2026-08-28). The
    merge seam clears every non-primary layer's scene title; if that layer's
    title band still fed the shared rect, the whole overlay's chrome would be
    pushed down by a gutter nothing is ever drawn in (measured pre-fix: first
    chrome line at y1=37.729 instead of 16). Legends are the opposite case --
    they render, so their gutters still count (asserted by the legend test
    above).
    """
    df = _df()
    base = fm.Chart(df).encode(x="x:Q", y="y:Q")
    line = base.mark_line()
    points = base.mark_point()

    titled = fm.LayerChart(line, points.properties(title="LAYER2")).to_svg()
    plain = fm.LayerChart(line, points).to_svg()

    assert "LAYER2" not in titled, "a non-primary layer's title is not drawn"
    assert set(_line_coord_tuples(titled)) == set(_line_coord_tuples(plain)), (
        "an undrawn title must not move any chrome: the titled overlay's geometry "
        "must equal the untitled one's exactly"
    )


def test_layer_chart_primary_title_still_reserves_its_band():
    """The mirror of the rule above: a drawn title reserves a real band.

    Guards against over-correcting -- the primary layer's title IS rendered, so
    its band must still be reserved and, through the shared rect, push every
    layer's chrome down.
    """
    df = _df()
    base = fm.Chart(df).encode(x="x:Q", y="y:Q")
    line = base.mark_line()
    points = base.mark_point()

    titled = fm.LayerChart(line.properties(title="LAYER1"), points).to_svg()
    plain = fm.LayerChart(line, points).to_svg()

    assert "LAYER1" in titled, "the primary layer's title renders"
    titled_top = min(float(c[1]) for c in _line_coord_tuples(titled))
    plain_top = min(float(c[1]) for c in _line_coord_tuples(plain))
    assert titled_top > plain_top, (
        f"a drawn title must reserve a real band; chrome starts at {titled_top} "
        f"vs {plain_top} without the title"
    )


def test_layer_chart_above_marks_axis_layer_renders_its_axis_once():
    """A ``zindex >= 1`` axis on a non-primary layer renders once, not twice.

    Former refusal door 2 (GH #89A acceptance §9.2). ``zindex >= 1`` routes an
    axis and its gridlines into the ``chrome_above`` panel slot, which the
    merge seam did not clear before #89A -- so deduping the layer's ordinary
    ``axes``/``grid`` would have left a second axis visible, and the fix
    refused to dedup at all. The seam now clears ``chrome_above`` with the
    rest of the chrome.
    """
    df = _df()
    base = fm.Chart(df).encode(x="x:Q", y="y:Q")
    line = base.mark_line()
    above = base.mark_point().configure_axis(zindex=1)

    coord_tuples = _line_coord_tuples(fm.LayerChart(line, above).to_svg())
    assert len(coord_tuples) == len(set(coord_tuples)), (
        "the above-marks axis layer must not add a second copy of the chrome; "
        f"duplicates: {[c for c in coord_tuples if coord_tuples.count(c) > 1]}"
    )
    # The chart still draws a full set of chrome -- dedup removed the copy,
    # not the axis.
    assert len(coord_tuples) == len(_line_coord_tuples(line.to_svg()))


def test_layer_chart_below_marks_annotation_layer_keeps_its_annotation():
    """A below-marks annotation survives its layer's chrome dedup.

    Former refusal door 3 (GH #89A acceptance §9.2). Before GH #89B the
    annotation shared a scene slot with the gridlines, so clearing chrome
    would have deleted it; the fix refused to dedup any layer carrying one.
    The annotation now has its own typed slot, which the merge seam never
    touches.
    """
    from ferrum.annotation.coords import norm
    from ferrum.annotation.primitives import text

    df = _df()
    base = fm.Chart(df).encode(x="x:Q", y="y:Q")
    line = base.mark_line()
    annotated = base.mark_point() + text(norm(0.5), norm(0.5), "BELOWMARK", z="below_marks")

    svg = fm.LayerChart(line, annotated).to_svg()
    coord_tuples = _line_coord_tuples(svg)

    assert len(coord_tuples) == len(set(coord_tuples)), (
        "the annotation-bearing layer's chrome must be deduplicated; duplicates: "
        f"{[c for c in coord_tuples if coord_tuples.count(c) > 1]}"
    )
    assert "BELOWMARK" in svg, "the below-marks annotation is content and must survive"
    assert svg.find("BELOWMARK") < svg.index("<circle"), "and must still paint below the marks"
