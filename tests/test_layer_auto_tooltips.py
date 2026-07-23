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


def test_mixed_layer_explicit_chart_level_tooltip_after_layer_promotion():
    """GH #78: a genuine chart-wide ``.encode(tooltip=...)`` set AFTER a
    merge must short-circuit for every tooltip-less layer even when one
    layer (from an earlier promotion) still carries its own explicit
    tooltip. The old structural proxy (``any_layer_explicit``) treated that
    surviving promoted tooltip as evidence the fresh chart-level value was
    itself a promotion, and ran per-layer auto-injection for the OTHER
    (tooltip-less) layer instead of letting it inherit the new chart-wide
    value via Rust's ``inherit_from`` fallback."""
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    points = fm.Chart(df).mark_point().encode(x="x", y="margin")
    merged = bars + points  # layer 0 (bars) carries its own promoted tooltip
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


def test_promoted_primary_tooltip_layer_two_still_gets_own_auto_fields():
    """Sibling of test_bug_hunt_figure_legend's promotion regression test,
    using plain ``__add__`` instead of ``_build_merged`` -- promotion must
    still let the non-primary layer collect its own auto tooltip fields."""
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


def test_add_then_encode_tooltip_resets_promoted_marker():
    """(a + b).encode(tooltip=...) on the merged chart must behave as a
    genuine chart-wide override -- the ``_tooltip_promoted`` marker carried
    over by ``_clone()`` from the ``+`` must not leak through ``.encode()``.
    Without the reset, this chart's ``_tooltip_promoted`` would still read
    ``True`` after the explicit re-encode, and the loop would run instead of
    short-circuiting."""
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


def test_selection_union_applies_under_promoted_tooltip():
    """GH #58 (narrower combo caught by review): a field-based selection
    must still union its fields into every tooltip-less layer even when the
    chart-level tooltip is PROMOTED (not selection-only). Before the fix,
    ``selection_injected`` was False whenever ``self._encoding`` carried a
    promoted ``tooltip`` (correctly, for the short-circuit gate), but that
    same flag ALSO gated the per-layer union step -- so the selection's
    fields were silently dropped from every tooltip-less layer in this
    combo, even though ``_inject_selection_tooltips`` had already merged
    them into the chart-level ``tooltip_fields``."""
    from ferrum.selection import selection_point

    df = _df()
    # a's own tooltip is promoted onto the merged chart-level encoding.
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    sel = selection_point(fields=["z"], name="sel")
    merged = (a + b).add_selection(sel)

    kw = _tooltip_wire(merged)

    # Chart-level: _inject_selection_tooltips rewrote the promoted single
    # "tooltip" into a merged "tooltip_fields" list once it saw "z" missing.
    assert kw["encoding"]["tooltip_fields"] == [{"field": "revenue"}, {"field": "z"}]
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
# GH #78 (review remediation): chained merges must not re-derive provenance
# structurally on every "+" -- promotion only happens when the LHS of a
# given merge is itself unlayered.
# ---------------------------------------------------------------------------


def test_chained_merge_after_genuine_override_stays_short_circuited():
    """A genuine chart-wide override set on an already-layered chart, then
    extended with a THIRD layer via another ``+``, must still short-circuit
    for every tooltip-less layer. Before this fix, the second ``+`` recomputed
    ``_tooltip_promoted`` structurally from ``"tooltip" in new._encoding``,
    which is still True after the override (the tooltip is still there --
    it's just no longer promoted), flipping the marker back to True and
    reintroducing GH #78 one merge later."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    c = fm.Chart(df).mark_line().encode(x="x", y="z")

    m = (a + b).encode(tooltip="margin")  # genuine chart-wide override
    m2 = m + c  # extend with a third layer AFTER the override

    assert m2._tooltip_promoted is False

    kw = _tooltip_wire(m2)

    assert kw["encoding"]["tooltip"] == {"field": "margin"}
    assert "tooltip_fields" not in kw["encoding"]
    for layer in kw["layers"]:
        # All three layers (a, b, c) are tooltip-less on their own -- none
        # should get per-layer auto-injection; all fall back to the
        # chart-wide "margin" tooltip via Rust's inherit_from.
        assert "tooltip_fields" not in layer.get("encoding", {})
        assert "tooltip" not in layer.get("encoding", {})


def test_promotion_chain_marker_survives_third_merge():
    """A promoted tooltip that is NEVER overridden must stay promoted across
    a chain of merges: ``(a_tt + b) + c`` keeps layer 0's (a's) own tooltip,
    and both b and c must still get their own per-layer auto fields (not a
    single shared chart-level field set)."""
    df = _df()
    a = fm.Chart(df).mark_bar().encode(x="x", y="revenue", tooltip="revenue")
    b = fm.Chart(df).mark_point().encode(x="x", y="margin")
    c = fm.Chart(df).mark_line().encode(x="x", y="z")

    m = a + b
    m2 = m + c

    assert m2._tooltip_promoted is True

    kw = _tooltip_wire(m2)

    layers = kw["layers"]
    assert len(layers) == 3
    assert layers[0]["encoding"]["tooltip"] == {"field": "revenue"}
    assert "tooltip_fields" not in layers[0]["encoding"]
    assert layers[1]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "margin"}]
    assert layers[2]["encoding"]["tooltip_fields"] == [{"field": "x"}, {"field": "z"}]
