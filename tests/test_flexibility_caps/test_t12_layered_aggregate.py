"""T12 — per-layer encoding aggregates must resolve in layered charts.

Flexibility-audit silent-wrong-output bug: a layer built with
``encode(y=Y("v", aggregate="mean"))`` was silently ignored when layered with
another chart. The single-chart path resolves the encoding aggregate into a
top-level ``Aggregate`` transform (via ``_resolve_pending_aggregates``) and
remaps the encoding field to the aggregated output column; the layered path
dropped the aggregate entirely, so the renderer plotted the raw zig-zag through
every row instead of the aggregated mean.

The renderer never executes per-layer ``transforms`` standalone — a layer reads
its input batch from a *named* chart-level transform via ``data_source``. So the
fix routes each aggregating layer through its own named ``Aggregate`` transform
and points the layer's ``data_source`` at it. Two aggregating layers therefore
aggregate independently and never share a groupby.

These tests pin the fix at the spec level (each aggregating layer routes to its
own named ``Aggregate`` transform, encoding field remapped) and behaviorally
(the chart renders against the aggregated columns), plus back-compat
(no-aggregate layered charts and standalone single-chart aggregates unchanged).
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm


def _multi_row_df() -> pl.DataFrame:
    """Two x-groups, multiple rows each, distinct mean vs. max per group."""
    return pl.DataFrame(
        {
            # Float groupby: the Rust stat_aggregate transform requires a
            # Float64/Utf8 groupby column (pre-existing constraint, identical on
            # the single-chart path); orthogonal to the layered-aggregate fix.
            "t": [1.0, 1.0, 2.0, 2.0],
            "v": [10.0, 20.0, 30.0, 50.0],
        }
    )


def _agg_transforms(spec_dict: dict) -> list[dict]:
    return [t for t in (spec_dict.get("transforms") or []) if t.get("type") == "aggregate"]


def _layer_agg(spec_dict: dict, layer: dict) -> dict:
    """Return the named Aggregate transform a layer routes to via data_source."""
    name = layer.get("data_source")
    assert name is not None, "aggregating layer must route to a named transform"
    matches = [t for t in _agg_transforms(spec_dict) if t.get("name") == name]
    assert len(matches) == 1, f"layer data_source {name!r} must map to one Aggregate"
    return matches[0]


def test_two_aggregating_layers_each_route_to_their_own_aggregate():
    """layer(mean, max): each layer routes to its own named Aggregate + remap."""
    df = _multi_row_df()
    mean_layer = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"))
    max_layer = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="max"))

    d = json.loads((mean_layer + max_layer).to_spec().to_json())
    layers = d["layers"]
    assert len(layers) == 2

    # Layer 0 (mean): named Aggregate, fn=mean, groupby=[t], y → mean_v.
    agg0 = _layer_agg(d, layers[0])
    assert agg0["ops"][0]["fn"] == "mean"
    assert agg0["ops"][0]["field"] == "v"
    assert agg0["ops"][0]["as"] == "mean_v"
    assert agg0["groupby"] == ["t"]
    assert layers[0]["encoding"]["y"]["field"] == "mean_v"
    # The raw aggregate= attribute must NOT survive as an ignored encoding key.
    assert "aggregate" not in layers[0]["encoding"]["y"]

    # Layer 1 (max): independent named Aggregate, y → max_v.
    agg1 = _layer_agg(d, layers[1])
    assert agg1["ops"][0]["fn"] == "max"
    assert agg1["ops"][0]["field"] == "v"
    assert agg1["ops"][0]["as"] == "max_v"
    assert agg1["groupby"] == ["t"]
    assert layers[1]["encoding"]["y"]["field"] == "max_v"
    assert "aggregate" not in layers[1]["encoding"]["y"]


def test_two_aggregating_layers_aggregate_independently():
    """The two layers must route to DISTINCT named transforms — no shared groupby."""
    df = _multi_row_df()
    mean_layer = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"))
    max_layer = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="max"))

    d = json.loads((mean_layer + max_layer).to_spec().to_json())
    layers = d["layers"]
    src0 = layers[0]["data_source"]
    src1 = layers[1]["data_source"]
    assert src0 != src1, "each aggregating layer must read its own batch"
    # Different output columns prove the aggregates are independent, not merged.
    assert layers[0]["encoding"]["y"]["field"] == "mean_v"
    assert layers[1]["encoding"]["y"]["field"] == "max_v"
    # Exactly two named Aggregate transforms, one per layer.
    assert len(_agg_transforms(d)) == 2


def test_layer_method_resolves_aggregate():
    """The fix applies via the + operator (which routes through layer wrapping)."""
    df = _multi_row_df()
    base = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"))
    other = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="max"))

    d = json.loads((base + other).to_spec().to_json())
    layers = d["layers"]
    assert _layer_agg(d, layers[0])["ops"][0]["fn"] == "mean"
    assert _layer_agg(d, layers[1])["ops"][0]["fn"] == "max"


def test_single_aggregating_layer_in_layered_chart():
    """One aggregating layer + one raw layer: only the aggregator routes."""
    df = _multi_row_df()
    agg = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"))
    raw = fm.Chart(df).mark_point().encode(x="t", y="v")

    d = json.loads((agg + raw).to_spec().to_json())
    layers = d["layers"]
    assert _layer_agg(d, layers[0])["ops"][0]["fn"] == "mean"
    assert layers[0]["encoding"]["y"]["field"] == "mean_v"
    # Raw layer untouched: no data_source, field stays v.
    assert layers[1].get("data_source") is None
    assert layers[1]["encoding"]["y"]["field"] == "v"
    # Exactly one named Aggregate transform overall.
    assert len(_agg_transforms(d)) == 1


def test_count_shorthand_aggregate_in_layer():
    """count() (field-less) aggregate resolves on a layer too."""
    df = pl.DataFrame({"cat": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
    bars = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y(aggregate="count"))
    line = fm.Chart(df).mark_line().encode(x="cat", y=fm.Y("v", aggregate="mean"))

    d = json.loads((bars + line).to_spec().to_json())
    layers = d["layers"]
    agg0 = _layer_agg(d, layers[0])
    assert agg0["ops"][0]["fn"] == "count"
    # count() has no source field — output column is count_all.
    assert layers[0]["encoding"]["y"]["field"] == "count_all"


def test_aggregating_layer_groupby_includes_color():
    """Non-aggregate encoding fields (color, detail) join the per-layer groupby."""
    df = pl.DataFrame(
        {
            "t": [1.0, 1.0, 2.0, 2.0],
            "g": ["a", "b", "a", "b"],
            "v": [10.0, 20.0, 30.0, 50.0],
        }
    )
    agg = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"), color="g")
    raw = fm.Chart(df).mark_point().encode(x="t", y="v")

    d = json.loads((agg + raw).to_spec().to_json())
    groupby = _layer_agg(d, d["layers"][0])["groupby"]
    assert set(groupby) == {"t", "g"}


def test_aggregating_layer_renders_without_error():
    """Behavioral: the aggregating layered chart renders to SVG."""
    df = _multi_row_df()
    mean_layer = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"))
    max_layer = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="max"))

    svg = (mean_layer + max_layer).show_svg()
    assert svg.lstrip().startswith("<")
    assert "<path" in svg or "<polyline" in svg


def test_no_aggregate_layered_chart_unchanged():
    """Back-compat: a layered chart with no aggregates carries no agg transform.

    The resolution path must be a strict no-op when no layer carries an
    aggregate kwarg: no data_source injected, no Aggregate transform added.
    """
    df = _multi_row_df()
    line = fm.Chart(df).mark_line().encode(x="t", y="v")
    point = fm.Chart(df).mark_point().encode(x="t", y="v")

    d = json.loads((line + point).to_spec().to_json())
    assert not _agg_transforms(d)
    for layer in d["layers"]:
        assert layer.get("data_source") is None
        assert layer["encoding"]["y"]["field"] == "v"


def test_overlap_column_aggregating_layer_resolves():
    """Overlap-column ``+`` path must still resolve a layer's aggregate.

    ``__add__`` pre-assigns a named ``_ident_`` identity ``data_source`` to RHS
    layers when their frame shares a column name with the LHS frame (the
    column-overlap routing path).  The earlier T12 fix skipped any layer that
    already carried a ``data_source``, so an aggregating RHS layer over an
    overlapping-column frame had its ``aggregate=`` silently dropped and plotted
    raw rows.  Here ``t`` overlaps between the two frames, forcing the overlap
    path; the RHS ``y=Y("w", aggregate="mean")`` must still route to a named
    Aggregate and remap its field to the aggregated output column.
    """
    df1 = pl.DataFrame({"t": [1.0, 1.0, 2.0, 2.0], "v": [10.0, 20.0, 30.0, 50.0]})
    df2 = pl.DataFrame({"t": [1.0, 1.0, 2.0, 2.0], "w": [5.0, 7.0, 9.0, 11.0]})

    lhs = fm.Chart(df1).mark_point().encode(x="t", y="v")
    rhs = fm.Chart(df2).mark_line().encode(x="t", y=fm.Y("w", aggregate="mean"))

    d = json.loads((lhs + rhs).to_spec().to_json())
    layers = d["layers"]
    assert len(layers) == 2

    # LHS raw layer: no aggregate, field stays v.
    assert layers[0]["encoding"]["y"]["field"] == "v"

    # RHS aggregating layer: routes to a named mean Aggregate, NOT raw w.
    rhs_layer = layers[1]
    agg = _layer_agg(d, rhs_layer)
    assert agg["ops"][0]["fn"] == "mean"
    # The overlap path renames RHS columns (w -> w__rhs_...), so the aggregated
    # output column and the remapped y field carry that suffix.  The contract is
    # that y is the aggregated mean column, not the raw value column.
    y_field = rhs_layer["encoding"]["y"]["field"]
    assert y_field.startswith("mean_w"), y_field
    assert y_field == agg["ops"][0]["as"]
    assert "aggregate" not in rhs_layer["encoding"]["y"]
    # The aggregate must NOT route to the bare _ident_ identity source anymore.
    assert not rhs_layer["data_source"].startswith("_ident_")
    assert len(_agg_transforms(d)) == 1


def test_standalone_single_chart_aggregate_unchanged():
    """Back-compat: the single-chart aggregate path is unaffected."""
    df = _multi_row_df()
    d = json.loads(
        fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean")).to_spec().to_json()
    )
    agg = _agg_transforms(d)
    assert len(agg) == 1
    assert agg[0]["ops"][0]["fn"] == "mean"
    assert agg[0]["groupby"] == ["t"]
    # Single-chart aggregate stays a plain (unnamed) top-level transform.
    assert "name" not in agg[0]
    assert d["encoding"]["y"]["field"] == "mean_v"
    # No layers key for a plain single-mark chart.
    assert "layers" not in d or not d["layers"]
