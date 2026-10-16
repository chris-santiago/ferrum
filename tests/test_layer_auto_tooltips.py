"""Per-layer auto-tooltip injection tests (GH #52 Task 10f bug #2).

``Chart._inject_auto_tooltips`` (``src/ferrum/_spec_build.py``) used to read
ONLY the chart-level (post-merge, primary-layer) encoding, so every
non-primary layer of a ``LayerChart`` reported the PRIMARY layer's tooltip
fields instead of its own -- confirmed via headless WASM capture (hovering a
secondary-y-axis layer's mark showed the primary layer's ``x``/``revenue``
fields instead of its own ``margin`` field; see
``.claude/output/secondary-y-captures/NOTES.md``).

The fix derives each layer's auto-tooltip fields from that layer's OWN
merged encoding (``kw["layers"][i]["encoding"]``, already populated per layer
by ``_build_layers_list``) while leaving the chart-level injection in place --
the seam contract with the paired Rust-side fix is that Rust prefers a
layer's own tooltip fields and falls back to the chart-level ones when a
layer carries none.
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm


def _tooltip_wire(chart: fm.Chart) -> dict:
    """Return the parsed-JSON spec dict after auto-tooltip injection.

    Mirrors what ``ferrum._scene._render_scene`` does before handing the spec
    to the Rust interactive renderer.
    """
    spec, _data, _viewport, _theme, _chart_config = chart._render_inputs(_auto_tooltips=True)
    return json.loads(spec.to_json())


def _df():
    return pl.DataFrame(
        {"x": [1, 2, 3], "revenue": [100.0, 200.0, 300.0], "margin": [10.0, 20.0, 30.0]}
    )


# ---------------------------------------------------------------------------
# Unlayered chart: wire unchanged.
# ---------------------------------------------------------------------------


def test_unlayered_chart_tooltip_wire_unchanged():
    """A plain (non-layered) chart's auto-tooltip injection is untouched --
    no ``layers`` key exists to iterate, so the fix is a pure no-op here."""
    chart = fm.Chart(_df()).mark_point().encode(x="x", y="revenue", color="margin")
    kw = _tooltip_wire(chart)

    assert "layers" not in kw
    assert kw["encoding"]["tooltip_fields"] == [
        {"field": "x"},
        {"field": "revenue"},
        {"field": "margin"},
    ]


def test_unlayered_chart_explicit_tooltip_wins():
    """An explicit ``tooltip=`` encoding is never overridden by auto-injection."""
    chart = fm.Chart(_df()).mark_point().encode(x="x", y="revenue", tooltip="margin")
    kw = _tooltip_wire(chart)

    assert "tooltip_fields" not in kw["encoding"]
    assert kw["encoding"]["tooltip"] == {"field": "margin"}


# ---------------------------------------------------------------------------
# Layered chart: per-layer tooltip fields, chart-level fallback preserved.
# ---------------------------------------------------------------------------


def test_layered_chart_each_layer_gets_its_own_tooltip_fields():
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    points = fm.Chart(df).mark_point().encode(x="x", y="margin")
    layered = fm.LayerChart(bars, points, resolve={"y": "independent"})

    kw = _tooltip_wire(layered._build_merged())

    layers = kw["layers"]
    assert len(layers) == 2
    assert layers[0]["mark"] == "bar"
    assert layers[0]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "revenue"}]
    assert layers[1]["mark"] == "point"
    assert layers[1]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "margin"}]

    # Distinct per layer -- this is the exact bug: before the fix both layers
    # carried the primary (bar) layer's fields.
    assert layers[0]["encoding"]["tooltip_fields"] != layers[1]["encoding"]["tooltip_fields"]

    # Chart-level tooltip_fields is kept (seam contract: Rust falls back to
    # it when a layer carries none of its own).
    assert kw["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "revenue"}]


def test_layered_chart_explicit_layer_tooltip_wins_over_auto():
    """A layer with its own explicit ``tooltip=`` encoding keeps it -- the
    per-layer auto-injection only fires when a layer carries no tooltip of
    its own."""
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    points = fm.Chart(df).mark_point().encode(x="x", y="margin", tooltip="margin")
    layered = fm.LayerChart(bars, points, resolve={"y": "independent"})

    kw = _tooltip_wire(layered._build_merged())

    layers = kw["layers"]
    assert layers[0]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "revenue"}]
    assert "tooltip_fields" not in layers[1]["encoding"]
    assert layers[1]["encoding"]["tooltip"] == {"field": "margin"}


def test_layered_chart_explicit_chart_level_tooltip_short_circuits_all_layers():
    """An explicit CHART-level ``tooltip=`` wins for every layer: neither the
    chart-level nor any per-layer auto-injection fires (a per-layer auto
    injection would beat the explicit tooltip in Rust's ``inherit_from``
    merge, silently overriding user intent)."""
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    points = fm.Chart(df).mark_point().encode(x="x", y="margin")
    layered = fm.LayerChart(bars, points, resolve={"y": "independent"})

    merged = layered._build_merged()
    explicit = merged.encode(tooltip="revenue")
    kw = _tooltip_wire(explicit)

    assert "tooltip_fields" not in kw["encoding"]
    assert kw["encoding"]["tooltip"] == {"field": "revenue"}
    for layer in kw["layers"]:
        assert "tooltip_fields" not in layer.get("encoding", {})


def test_shared_y_layered_chart_also_gets_per_layer_tooltips():
    """The per-layer fix is not specific to independent-y LayerCharts -- any
    layered chart's non-primary layer gets its own tooltip fields (this is a
    general multi-layer gap per the Task 10 report, not scoped to GH #52)."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    b = fm.Chart(df).mark_line(stroke="red").encode(x="x", y="revenue")
    merged = a + b

    kw = _tooltip_wire(merged)

    layers = kw["layers"]
    assert len(layers) == 2
    for layer in layers:
        assert layer["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "revenue"}]


# ---------------------------------------------------------------------------
# GH #78: provenance-based short-circuit (mixed layer + genuine chart-wide).
# ---------------------------------------------------------------------------


def test_mixed_layer_explicit_chart_level_tooltip_beside_a_layers_own():
    """GH #78: a ``.encode(tooltip=...)`` written on a composition must
    short-circuit for every tooltip-less layer even when one layer still
    carries its own explicit tooltip: the tooltip-less layer inherits the
    composition's tooltip via Rust's ``inherit_from`` fallback instead of
    getting per-layer auto-injection."""
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    points = fm.Chart(df).mark_point().encode(x="x", y="margin")
    merged = bars + points  # layer 0 (bars) carries its own tooltip
    explicit = merged.encode(tooltip="margin")  # genuine chart-wide override

    kw = _tooltip_wire(explicit)

    assert kw["encoding"]["tooltip"] == {"field": "margin"}
    assert "tooltip_fields" not in kw["encoding"]
    layers = kw["layers"]
    # layer 0 keeps its own explicit tooltip, untouched by the fresh
    # chart-wide value.
    assert layers[0]["encoding"]["tooltip"] == {"field": "revenue"}
    # layer 1 (tooltip-less) gets NO per-layer auto-injection -- it must
    # inherit the fresh chart-wide tooltip via Rust's chart-level fallback.
    assert "tooltip_fields" not in layers[1].get("encoding", {})
    assert "tooltip" not in layers[1].get("encoding", {})


def test_a_primary_layers_own_tooltip_leaves_layer_two_its_own_auto_fields():
    """Sibling of test_bug_hunt_figure_legend's regression test, using plain
    ``__add__`` instead of ``_build_merged``: the non-primary layer collects
    its own auto tooltip fields beside the primary layer's own tooltip."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    merged = a + b

    kw = _tooltip_wire(merged)

    layers = kw["layers"]
    assert layers[0]["encoding"]["tooltip"] == {"field": "revenue"}
    assert layers[1]["encoding"].get("tooltip_fields") == [
        {"field": "x"},
        {"field": "margin"},
    ]


def test_add_then_encode_tooltip_is_the_compositions_own():
    """(a + b).encode(tooltip=...) on the merged chart is written on the
    composition, the parent of every layer: it short-circuits per-layer
    injection and reaches every layer that binds no tooltip of its own."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    merged = (a + b).encode(tooltip="margin")

    kw = _tooltip_wire(merged)

    assert kw["encoding"]["tooltip"] == {"field": "margin"}
    assert "tooltip_fields" not in kw["encoding"]
    # layer 0's own pre-merge tooltip is untouched (its own explicit value),
    # but neither layer gets per-layer tooltip_fields injected.
    layers = kw["layers"]
    assert layers[0]["encoding"]["tooltip"] == {"field": "revenue"}
    assert "tooltip_fields" not in layers[0]["encoding"]
    assert "tooltip_fields" not in layers[1].get("encoding", {})
    assert "tooltip" not in layers[1].get("encoding", {})


# ---------------------------------------------------------------------------
# GH #58: selection-injected chart-level tooltip must union per layer, not
# replace every layer's own fields with one shared set.
# ---------------------------------------------------------------------------


def test_selection_linked_tooltip_unions_per_layer_fields():
    """GH #58: a field-based ``selection_point``'s ``fields`` must union
    into EACH layer's own auto tooltip fields, not replace them with one
    shared chart-level field set. Before the fix, ``_inject_selection_tooltips``
    setting chart-level ``tooltip_fields`` made ``chart_level_explicit`` true
    with no layer carrying its own explicit tooltip, so the short-circuit
    fired and both layers fell back to the single selection-derived field
    (losing their own auto fields entirely)."""
    from ferrum.selection import selection_point

    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    sel = selection_point(fields=["revenue"], name="sel")
    merged = (a + b).add_selection(sel)

    kw = _tooltip_wire(merged)

    layers = kw["layers"]
    assert len(layers) == 2
    # layer 0 (bar) already encodes "revenue" via its own auto fields.
    assert layers[0]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "revenue"}]
    # layer 1 (point) does not encode "revenue" -- the selection field must
    # be unioned in alongside its own x/margin auto fields, not replace them.
    assert layers[1]["encoding"]["tooltip_fields"] == [
        {"field": "x"},
        {"field": "margin"},
        {"field": "revenue"},
    ]


def test_selection_union_applies_beside_an_operands_own_tooltip():
    """GH #58: a field-based selection unions its fields into every
    tooltip-less layer when a left operand carries its own tooltip. V1: that
    tooltip stays on its own layer (it never becomes the chart level), so
    the chart-level ``tooltip_fields`` holds only the selection's field."""
    from ferrum.selection import selection_point

    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    sel = selection_point(fields=["z"], name="sel")
    merged = (a + b).add_selection(sel)

    kw = _tooltip_wire(merged)

    assert "tooltip" not in kw["encoding"]
    assert kw["encoding"]["tooltip_fields"] == [{"field": "z"}]
    layers = kw["layers"]
    # layer 0 keeps its own explicit tooltip, untouched (invariant).
    assert layers[0]["encoding"]["tooltip"] == {"field": "revenue"}
    assert "tooltip_fields" not in layers[0]["encoding"]
    # layer 1 gets its own auto fields UNIONED with the selection field.
    assert layers[1]["encoding"]["tooltip_fields"] == [
        {"field": "x"},
        {"field": "margin"},
        {"field": "z"},
    ]


# ---------------------------------------------------------------------------
# A tooltip written on a composition is the parent of that composition's
# layers only; a later merge's operand keeps its own auto fields.
# ---------------------------------------------------------------------------


def test_a_compositions_tooltip_stays_with_its_layers_after_a_later_merge():
    """Parent rule: ``(a + b).encode(tooltip=...) + c`` reaches a and b, not c.

    ``(a + b)`` is the left operand of the second ``+``, so the tooltip
    written on it is that operand's own and moves onto its layers; c draws
    its own auto fields."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    c = fm.Chart(df).mark_line().encode(x="x", y="z")

    m2 = (a + b).encode(tooltip="margin") + c

    kw = _tooltip_wire(m2)

    assert "tooltip" not in kw["encoding"]
    layers = kw["layers"]
    assert [layer["encoding"].get("tooltip") for layer in layers] == [
        {"field": "margin"},
        {"field": "margin"},
        None,
    ]
    assert layers[2]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "z"}]


def test_an_operands_own_tooltip_stays_on_its_layer_across_merges():
    """V1: ``(a_tt + b) + c`` keeps a's own tooltip on a's layer, and both b
    and c still get their own per-layer auto fields (not a single shared
    chart-level field set)."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    c = fm.Chart(df).mark_line().encode(x="x", y="z")

    m2 = a + b + c

    kw = _tooltip_wire(m2)

    assert "tooltip" not in kw["encoding"]
    layers = kw["layers"]
    assert len(layers) == 3
    assert layers[0]["encoding"]["tooltip"] == {"field": "revenue"}
    assert "tooltip_fields" not in layers[0]["encoding"]
    assert layers[1]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "margin"}]
    assert layers[2]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "z"}]
