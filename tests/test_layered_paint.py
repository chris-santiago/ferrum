"""Python-level pin for spec §4.0's hoisted-paint / literal-paint precedence
rules (Batch A Task 14, Lane A).

The mechanism lives in Rust (`LayerPrepared::from_chart_and_layer`'s
chart-level `mark_style` fallback in
``crates/ferrum-core/src/render/prepare/mod.rs``, plus the own-color
exemption in ``crates/ferrum-core/src/render/scene_build.rs``'s
``build_panel_mark_batches``), but the INPUT it operates on --  whether a
layer's own kwargs/encoding reach Rust as "this layer's own" versus
"inherited from chart level" -- is entirely a product of how Python's
``LayerChart`` lowering (``Chart.layer()`` / ``+`` / composite-mark
desugars) shapes the spec tree. Every Rust-side test for this rule builds
its `LayerPrepared`/`ChartSpec` fixtures by hand, so a regression in the
*Python lowering* (e.g. a desugar that stops copying a layer's own
`mark_kwargs` onto that layer, or starts also copying them onto a sibling)
would pass every one of those tests untouched. This module closes that gap:
every fixture here is a real, unmodified `fm.Chart`/figure-function call,
rendered to SVG and asserted on structurally.

Three rules, spec §4.0 + §4.4 (2026-08-28 T4/T5d amendments):

1. A layer with its own user-set literal paint does NOT adopt an inherited
   chart-level color channel -- its literal paint wins (the ROC
   chance-diagonal class: ``fm.roc_chart``'s grey dashed reference line
   must render once, in theme grey, never fanned out per class).
2. A layer with NO literal paint of its own keeps inheriting the
   chart-level color channel exactly as before (the boxplot-hue class:
   ``fm.catplot(kind="box", hue=...)``'s IQR-rect layer, which declares no
   paint of its own, must still vary its fill by hue).
3. A layer's own declared color channel always wins over its own literal
   paint (constructed directly via `Chart.layer()`/`+`, since no shipped
   figure function happens to combine both on one layer).
"""

from __future__ import annotations

import re

import polars as pl

import ferrum
from tests.fixtures import load_dataset, load_fixture


def _stroke(el: str) -> str | None:
    m = re.search(r'stroke="([^"]+)"', el)
    return m.group(1) if m else None


def _dasharray(el: str) -> str | None:
    m = re.search(r'stroke-dasharray="([^"]+)"', el)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Class 1 -- ROC chance-diagonal: layer's own literal paint beats an
# inherited chart-level color channel.
# ---------------------------------------------------------------------------


def test_roc_reference_diagonal_keeps_its_own_literal_grey_not_fanned_per_class():
    """``desugar_roc``'s "reference" layer declares no ``color=`` of its own
    and a literal ``stroke="#AAAAAA"``/``stroke_dash=[4, 4]``; the "line"
    layer declares its OWN ``color=class`` (hoisted onto chart level by
    ``Chart._resolve_pending``'s layered-mode wiring, per every other
    composite mark). Pre-fix, the reference layer had no protection against
    that hoisted chart-level color channel: it adopted the per-class
    palette AND its per-row color grouping fanned the single collinear
    diagonal out into one duplicate polyline per class (3 classes -> 3 grey
    dashes became 3 mis-colored dashes). Post-fix there must be exactly one
    dashed reference polyline, in theme grey, plus exactly one solid
    class-colored polyline per class."""
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    n_classes = len(set(df["y"].to_list()))
    assert n_classes == 3, "fixture assumption: 3-class multiclass_classification"

    chart = ferrum.roc_chart(source, per_class=True, annotate_auc=False)
    svg = chart.to_svg()

    polylines = re.findall(r"<polyline[^>]*>", svg)
    assert len(polylines) == n_classes + 1, (
        f"expected {n_classes} class curves + 1 reference diagonal, got "
        f"{len(polylines)} polylines: {[_stroke(p) for p in polylines]}"
    )

    dashed = [p for p in polylines if _dasharray(p) is not None]
    assert len(dashed) == 1, f"expected exactly one dashed polyline (the diagonal), got {dashed}"
    assert _stroke(dashed[0]) == "#aaaaaa"
    assert _dasharray(dashed[0]) == "4,4"

    solid = [p for p in polylines if _dasharray(p) is None]
    assert len(solid) == n_classes
    solid_strokes = {_stroke(p) for p in solid}
    # One distinct color per class, and none of them is the reference grey --
    # the fanned-out pre-fix bug reused the diagonal's own grey/dash pairing
    # per class, so a solid line inheriting grey would also be diagnostic.
    assert len(solid_strokes) == n_classes
    assert "#aaaaaa" not in solid_strokes


# ---------------------------------------------------------------------------
# Class 2 -- boxplot-hue: a layer with no literal paint of its own keeps
# inheriting the chart-level color channel.
# ---------------------------------------------------------------------------


def _boxplot_hue_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["only"] * 10,
            "hue": ["g1"] * 5 + ["g2"] * 5,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )


def test_catplot_box_hue_iqr_rect_still_inherits_chart_level_color():
    """``desugar_boxplot``'s "box" (IQR rect) layer carries no ``color=``
    and no literal ``fill=``/``stroke=`` of its own -- it relies entirely on
    ``catplot``'s chart-level ``color=hue`` encoding to vary per group. The
    §4.0 hoisted-paint strip must NOT touch this layer (it strips literal
    paint, and this layer inherits a real per-row color channel, not stray
    paint), so each hue group's box must still render in its own color --
    a regression here would silently flatten every box to one fallback
    color, exactly the bug the widened without_paint exemption guards
    against for the opposite case."""
    df = _boxplot_hue_df()
    chart = ferrum.catplot(df, x="cat", y="val", hue="hue", kind="box")
    svg = chart.to_svg()

    # The IQR box renders as a <rect> with a hex fill; the chart-canvas
    # background rect (640x480, id-less) is the only other unconditionally-
    # filled <rect> at this simple chart size, so filter it out by width.
    rects = re.findall(r"<rect[^>]*>", svg)
    box_fills = []
    for r in rects:
        width_m = re.search(r'width="([\d.]+)"', r)
        fill_m = re.search(r'fill="(#[0-9a-fA-F]{6})"', r)
        if width_m is None or fill_m is None:
            continue
        if float(width_m.group(1)) >= 640.0:
            continue  # chart-canvas background rect
        box_fills.append(fill_m.group(1))

    assert len(box_fills) == 2, f"expected one IQR box per hue group, got {box_fills}"
    assert len(set(box_fills)) == 2, f"expected two distinct hue colors, got {box_fills}"


# ---------------------------------------------------------------------------
# Class 3 -- a layer's own color channel always wins over its own literal
# paint (constructed directly; no shipped figure function combines both on
# one layer).
# ---------------------------------------------------------------------------


def test_layer_own_color_channel_beats_its_own_literal_fill():
    """A layer that declares BOTH a literal ``fill=`` AND its own ``color=``
    encoding (``layer.color_is_own`` is ``True``) must resolve per-row via
    the color channel, not the literal -- the exemption that clears an
    INHERITED channel under literal paint (class 1 above) never applies to
    a layer's own declaration (spec §4.4: "a layer's own declared color
    channel always wins over its literal paint"). Built via a real two-layer
    ``Chart`` composition (``+``), the same ``LayerChart`` lowering path
    every figure function goes through, not a hand-built spec tree."""
    df_a = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    df_b = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.5, 1.5, 0.2, 1.2],
            "grp": ["g1", "g1", "g2", "g2"],
        }
    )
    layer_a = ferrum.Chart(df_a).mark_line().encode(x="x", y="y")
    layer_b = ferrum.Chart(df_b).mark_point(fill="magenta").encode(x="x", y="y", color="grp")
    chart = layer_a + layer_b
    svg = chart.to_svg()

    magenta_hex = ferrum.color.to_hex("magenta")
    circle_fills = re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]{6})"', svg)
    assert len(circle_fills) >= 4, f"expected at least 4 point circles, got {circle_fills}"
    assert magenta_hex not in circle_fills, (
        f"literal fill={magenta_hex!r} leaked through despite the layer's own "
        f"color= channel: {circle_fills}"
    )
    # Two groups -> (at most, ignoring any legend swatch duplicates) two
    # distinct per-row fills actually used for the data points themselves.
    data_point_fills = circle_fills[:4]
    assert len(set(data_point_fills)) == 2, f"expected 2 distinct group colors, got {circle_fills}"
