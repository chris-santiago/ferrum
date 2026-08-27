"""Regression tests for finding P2 (design review, 2026-08-27).

Static ``LayerChart`` under the default/shared-``y`` resolve mode used to
render every axis line, tick label, grid line, and chart title once PER
LAYER instead of once for the whole overlay: a 2-layer ``LayerChart`` emitted
92 ``<line>`` elements (46 distinct coordinate tuples, every one drawn
twice) and 48 ``<text>`` elements (24 distinct), and two layers binding
different ``y`` fields overprinted both axis titles at the same origin.

Root cause and fix: each overlay child used to render full standalone
chrome over its own independently computed panel rect. Now, when the
per-leaf safety gate in ``crates/ferrum-core/src/render/composite_render.rs``
(``overlay_imposition_safe``) proves it consistent, each non-first overlay
child renders its marks against child 0's real plot rect and
``merge_children`` suppresses its per-panel ``grid``/``axes`` and the
scene-level ``title``; mark content (``marks``/``annotations``/
``strip_title``/legend) is untouched. A leaf the gate refuses (per-leaf
legend, above-marks axis, below-marks annotation) keeps its own rect and
chrome — the pre-fix behavior, never a silent mismatch.

Spec §9.5 acceptance covered here:
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
"""

from __future__ import annotations

import re

import polars as pl

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


def test_layer_chart_per_leaf_legend_refuses_imposition_keeps_both_chromes():
    """A leaf carrying its own legend must refuse the imposition-safety
    gate (``overlay_imposition_safe``) and keep its own rect/chrome -- the
    documented refusal shape (module docstring above: "A leaf the gate
    refuses (per-leaf legend, above-marks axis, below-marks annotation)
    keeps its own rect and chrome -- the pre-fix behavior, never a silent
    mismatch.").

    Mutation-testing gap close (2026-08-27 close-out): every test above is
    a leaf where imposition *succeeds*, so a mutation that always sets
    ``chrome_suppressed[i] = true`` regardless of whether imposition was
    actually applied is indistinguishable from correct code on all four.
    ``mark_point().encode(color="g:N")`` carries a per-leaf legend, which
    the gate must refuse -- under correct code this leaf keeps its own
    full chrome (its axis/grid duplicates layer 0's, since both compute
    the same domain independently), so ``<line>`` elements are *not*
    fully deduplicated the way every gate-succeeds test above asserts.
    Under the always-suppress mutation, this leaf's chrome collapses away
    regardless of the refusal, producing a smaller, fully-deduplicated
    line count indistinguishable from a normal imposition-succeeds merge.
    """
    df = _grouped_df()
    line = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_line()
    points = fm.Chart(df).encode(x="x:Q", y="y:Q").mark_point().encode(color="g:N")

    svg = fm.LayerChart(line, points).to_svg()
    coord_tuples = _line_coord_tuples(svg)
    texts = _text_contents(svg)

    # Structural discriminator, robust to unrelated rendering-detail drift:
    # duplication IS present (contrast with every gate-succeeds test above,
    # which asserts len(coord_tuples) == len(set(coord_tuples))) -- proving
    # the legend-bearing leaf kept its own chrome instead of being
    # suppressed.
    assert len(coord_tuples) > len(set(coord_tuples)), (
        "the legend-bearing leaf's chrome must not be suppressed -- expected "
        "duplicate <line> coordinates from both leaves keeping their own "
        f"axis/grid, got {len(coord_tuples)} lines / {len(set(coord_tuples))} unique"
    )
    # The legend itself must have rendered -- confirms the refusal really
    # was triggered by the per-leaf legend, not some unrelated leaf property.
    assert "g" in texts, "the color legend's title ('g') must render"
    assert "a" in texts and "b" in texts, "the color legend's category swatches must render"
