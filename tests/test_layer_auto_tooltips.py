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
