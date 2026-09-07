"""FA-4 — per-layer encoding bin= must resolve in layered charts.

Flexibility-audit silent-wrong-output bug: a layer built with
``encode(x=X("v", bin=True))`` was silently incorrect when layered with
another chart. The layered path either dropped the Bin from consideration or
put it in the unnamed chain where it corrupted the shared current batch for
all layers, causing named aggregate transforms to fail with "groupby column
not found" (because the Bin transform replaced the original column schema).

The renderer never executes per-layer ``transforms`` standalone — a layer
reads its input batch from a *named* chart-level transform via ``data_source``.
So the fix routes each binning layer through its own named ``Bin`` transform
and points the layer's ``data_source`` at it.  The layer encoding is
re-pointed at the Bin's own output columns and the ``bin`` kwarg is stripped.

Two binning layers therefore bin independently and never interfere with each
other or with the shared unnamed chain.

**Dated note (2026-09-06, batch-E T12 / spec §6.6).** FA-4 fixed the layered
path only, and this file used to record the single-chart path as deliberately
untouched ("byte-stable with pre-FA4 behavior"). That was wrong on the facts:
pre-FA4 behavior for the single-chart path was already broken, so *every*
standalone ``bin=`` idiom raised at render time (F-L01-01) while this file's
own docstring below claimed the standard histogram pattern "produces valid
output". Both paths now share one lowering
(``_layer_transforms._lower_bin_encoding``), so this suite covers the layered
half and ``tests/test_encode_bin_lowering.py`` covers the single-chart half.
Two behavior changes land here, both disclosed: a binning layer's encoding
gains the ``x2``/``y2`` span bound to ``bin_end`` (so layered bin bars span
their bin instead of collapsing to a default width) and its binned axis
recovers the user's column name as its title instead of reading
``"bin_start"``; and ``bin=`` + ``aggregate=`` on the same layer no longer
raises -- the aggregate folds into that layer's own ``Bin``.

These tests pin the fix at the spec level (each binning layer routes to its
own named ``Bin`` transform, encoding re-pointed) and behaviorally (the
chart renders without error), plus back-compat (no-bin layered charts
unchanged).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm


def _multi_row_df() -> pl.DataFrame:
    return pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})


def _bin_transforms(spec_dict: dict) -> list[dict]:
    return [t for t in (spec_dict.get("transforms") or []) if t.get("type") == "bin"]


def _layer_bin(spec_dict: dict, layer: dict) -> dict:
    """Return the named Bin transform a layer routes to via data_source."""
    name = layer.get("data_source")
    assert name is not None, "binning layer must route to a named transform"
    matches = [t for t in _bin_transforms(spec_dict) if t.get("name") == name]
    assert len(matches) == 1, f"layer data_source {name!r} must map to one Bin"
    return matches[0]


# ---------------------------------------------------------------------------
# Core spec-wiring tests
# ---------------------------------------------------------------------------


def test_binning_layer_routes_to_named_bin_transform():
    """A layer with bin=True gets a named Bin transform + data_source wiring."""
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    d = json.loads((binned + raw).to_spec().to_json())
    layers = d["layers"]
    assert len(layers) == 2

    # Binned layer: data_source set, routes to a named Bin transform.
    bin_xform = _layer_bin(d, layers[0])
    assert bin_xform["field"] == "v"
    # Encoding x remapped to bin_start; bin kwarg stripped.
    assert layers[0]["encoding"]["x"]["field"] == "bin_start"
    assert "bin" not in layers[0]["encoding"]["x"]

    # Raw layer untouched: no data_source, field stays v.
    assert layers[1].get("data_source") is None
    assert layers[1]["encoding"]["x"]["field"] == "v"
    assert "x2" not in layers[1]["encoding"]


def test_binning_layer_gains_the_bin_end_span():
    """Disclosed behavior change (2026-09-06, spec §6.6): a binning layer's
    encoding gains the span channel bound to ``bin_end``, so its bars span
    their bin instead of collapsing to a default width -- and its binned axis
    recovers the user's own column name as the title."""
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    enc = json.loads((binned + raw).to_spec().to_json())["layers"][0]["encoding"]
    assert enc["x2"]["field"] == "bin_end"
    assert enc["x"]["title"] == "v"


def test_horizontal_binning_layer_gains_the_y2_span():
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(y=fm.Y("v", bin=True), x="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    enc = json.loads((binned + raw).to_spec().to_json())["layers"][0]["encoding"]
    assert enc["y"]["field"] == "bin_start"
    assert enc["y2"]["field"] == "bin_end"
    assert "x2" not in enc


def test_two_binning_layers_route_independently():
    """Two layers with bin=True each get their own named Bin transform."""
    df = _multi_row_df()
    binned_a = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    binned_b = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="density")

    d = json.loads((binned_a + binned_b).to_spec().to_json())
    layers = d["layers"]
    assert len(layers) == 2

    src_a = layers[0]["data_source"]
    src_b = layers[1]["data_source"]
    assert src_a != src_b, "each binning layer must read its own named Bin output"

    bin_a = _layer_bin(d, layers[0])
    bin_b = _layer_bin(d, layers[1])
    assert bin_a["name"] != bin_b["name"]
    # Exactly two named Bin transforms.
    assert len(_bin_transforms(d)) == 2


def test_single_binning_layer_in_layered_chart():
    """One binning layer + one raw layer: only the binning layer routes."""
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    d = json.loads((binned + raw).to_spec().to_json())
    layers = d["layers"]

    # Binning layer routed.
    bin_xform = _layer_bin(d, layers[0])
    assert bin_xform["field"] == "v"
    assert layers[0]["encoding"]["x"]["field"] == "bin_start"

    # Raw layer untouched.
    assert layers[1].get("data_source") is None
    assert layers[1]["encoding"]["x"]["field"] == "v"

    # Exactly one named Bin transform overall.
    assert len(_bin_transforms(d)) == 1


def test_bin_kwarg_stripped_from_layer_encoding():
    """The bin= kwarg must not survive in the layer's serialized encoding."""
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    d = json.loads((binned + raw).to_spec().to_json())
    enc_x = d["layers"][0]["encoding"]["x"]
    assert "bin" not in enc_x, f"bin kwarg must be stripped; got {enc_x}"


def test_bin_named_transform_not_in_unnamed_chain():
    """Named Bin must NOT advance the unnamed chain (must carry a name field)."""
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    d = json.loads((binned + raw).to_spec().to_json())
    for t in _bin_transforms(d):
        assert "name" in t, f"Bin in layered chart must be named (got {t})"


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


def test_binning_layer_renders_without_error():
    """Behavioral: a layered chart with bin= renders to SVG without error."""
    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    svg = (binned + raw).to_svg()
    assert svg.lstrip().startswith("<")


# ---------------------------------------------------------------------------
# Overlap-column path test (mirrors T12 test_overlap_column_aggregating_layer_resolves)
# ---------------------------------------------------------------------------


def test_overlap_column_binning_layer_resolves():
    """Overlap-column ``+`` path must still resolve a layer's bin=.

    ``__add__`` pre-assigns a named ``_ident_`` identity ``data_source`` to RHS
    layers when their frame shares a column name with the LHS frame (the
    column-overlap routing path).  The bin resolution must override or work
    alongside that pre-assigned source so the RHS binning layer reads from its
    own named Bin transform, not the bare identity source.
    """
    df1 = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    df2 = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "w": [0.1] * 10})

    lhs = fm.Chart(df1).mark_point().encode(x="v", y="v")
    rhs = fm.Chart(df2).mark_bar().encode(x=fm.X("v", bin=True), y="count")

    d = json.loads((lhs + rhs).to_spec().to_json())
    layers = d["layers"]
    assert len(layers) == 2

    # LHS raw: no bin, field stays.
    assert layers[0].get("data_source") is None or layers[0]["encoding"]["x"]["field"] == "v"

    # RHS binning layer: routes to a named Bin transform, not raw _ident_.
    rhs_layer = layers[1]
    bin_xform = _layer_bin(d, rhs_layer)
    # The overlap path renames RHS columns (v -> v__rhs_...), so the Bin
    # transform must reference the renamed column.
    bin_field = bin_xform["field"]
    assert bin_field.startswith("v"), f"Bin field should start with 'v', got {bin_field!r}"
    # Encoding x remapped to bin_start.
    assert rhs_layer["encoding"]["x"]["field"] == "bin_start"
    assert "bin" not in rhs_layer["encoding"]["x"]
    # Must not route to the bare _ident_ identity source.
    assert not rhs_layer["data_source"].startswith("_ident_")


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


def test_no_bin_layered_chart_unchanged():
    """Back-compat: a layered chart with no bin= carries no named Bin transform."""
    df = _multi_row_df()
    line = fm.Chart(df).mark_line().encode(x="v", y="v")
    point = fm.Chart(df).mark_point().encode(x="v", y="v")

    d = json.loads((line + point).to_spec().to_json())
    assert not _bin_transforms(d)
    for layer in d["layers"]:
        assert layer.get("data_source") is None


def test_standalone_single_chart_bin_stays_unnamed():
    """The single-chart bin's TRANSFORM shape is unchanged by FA-4 and by T12.

    A plain ``Chart().encode(x=X('v', bin=True))`` must still produce an
    unnamed (chain-advancing) ``Bin`` in the spec's top-level ``transforms``
    list -- only named per-layer bins fan out.  The single-chart ENCODING is
    no longer left alone (spec §6.6); that half is covered by
    ``tests/test_encode_bin_lowering.py``.
    """
    df = _multi_row_df()
    d = json.loads(fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True)).to_spec().to_json())
    bins = _bin_transforms(d)
    assert len(bins) == 1
    assert bins[0]["field"] == "v"
    # Single-chart bin stays unnamed (no "name" key).
    assert "name" not in bins[0]
    # No layers key for a plain single-mark chart.
    assert "layers" not in d or not d["layers"]


# ---------------------------------------------------------------------------
# BUG 1 — bin=Bin(...) instance must preserve kwargs (bin_count etc.)
# ---------------------------------------------------------------------------


def test_bin_instance_single_chart_preserves_bin_count():
    """BUG 1a: bin=Bin("v", bin_count=5) must survive the single-chart path.

    Pre-fix: to_implicit_transforms() tries bin_arg.to_json() (absent on PyO3
    Bin), falls through to _bin_dict={}, silently drops bin_count.
    """
    from ferrum._core import Bin as RustBin

    df = _multi_row_df()
    chart = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=RustBin("v", bin_count=5)))
    d = json.loads(chart.to_spec().to_json())
    bins = _bin_transforms(d)
    assert len(bins) == 1
    assert bins[0]["field"] == "v"
    assert bins[0].get("bin_count") == 5, (
        f"bin_count must be 5, got {bins[0].get('bin_count')!r}; "
        f"bin kwargs were silently dropped (Bug 1)"
    )


def test_bin_instance_layered_chart_preserves_bin_count():
    """BUG 1a (layered): bin=Bin("v", bin_count=5) must survive the layered path."""
    from ferrum._core import Bin as RustBin

    df = _multi_row_df()
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=RustBin("v", bin_count=5)), y="count")
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")

    d = json.loads((binned + raw).to_spec().to_json())
    layers = d["layers"]
    bin_xform = _layer_bin(d, layers[0])
    assert bin_xform.get("bin_count") == 5, (
        f"bin_count must be 5 in the named Bin transform, got {bin_xform.get('bin_count')!r}; "
        f"bin kwargs were silently dropped (Bug 1)"
    )


def test_bin_dict_single_chart_preserves_bin_count():
    """BUG 1b: bin={"bin_count": N} must round-trip through the single-chart path.

    This is a byte-stability assertion — bin=True alone cannot catch this
    regression because bin_count would simply be absent (default) not wrong.
    """
    df = _multi_row_df()
    chart = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin={"bin_count": 7}))
    d = json.loads(chart.to_spec().to_json())
    bins = _bin_transforms(d)
    assert len(bins) == 1
    assert bins[0].get("bin_count") == 7, f"bin_count must be 7, got {bins[0].get('bin_count')!r}"


# ---------------------------------------------------------------------------
# BUG 2 — bin= + aggregate= on the same layer folds into that layer's own Bin
# ---------------------------------------------------------------------------


def test_bin_and_aggregate_on_same_layer_folds_into_one_named_bin():
    """Disclosed change (2026-09-06, spec §6.6): the same-layer refusal lifts.

    The refusal this replaced was a consequence of a Bin that could not
    aggregate: ``_resolve_layer_aggregates`` built a named Aggregate, then
    ``_resolve_layer_bins`` overwrote the layer's ``data_source`` with the named
    Bin, leaving encoding y pointed at an aggregate column the Bin's output
    never held.  Now the aggregate folds INTO the layer's own Bin
    (``agg_field``/``agg``), so there is one named transform, no named->named
    chaining, and the layer reads a column that really exists.
    """
    df = _multi_row_df()
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")
    binned_agg = (
        fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y("v", aggregate="mean"))
    )

    d = json.loads((raw + binned_agg).to_spec().to_json())
    layer = d["layers"][1]
    bin_xform = _layer_bin(d, layer)
    assert bin_xform["agg_field"] == "v"
    assert bin_xform["agg"] == "mean"
    # Exactly one named transform for that layer: no separate Aggregate.
    assert not [t for t in (d.get("transforms") or []) if t.get("type") == "aggregate"]
    assert layer["encoding"]["y"]["field"] == "agg_value"
    assert layer["encoding"]["y"]["title"] == "v"
    assert (raw + binned_agg).to_svg().lstrip().startswith("<")


def test_bin_and_count_shorthand_on_same_layer_reads_the_bins_count_column():
    """The field-less ``count()`` idiom needs no fold: the Bin already emits it."""
    df = _multi_row_df()
    raw = fm.Chart(df).mark_point().encode(x="v", y="v")
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count():Q")

    d = json.loads((raw + binned).to_spec().to_json())
    layer = d["layers"][1]
    bin_xform = _layer_bin(d, layer)
    assert "agg" not in bin_xform
    assert not [t for t in (d.get("transforms") or []) if t.get("type") == "aggregate"]
    assert layer["encoding"]["y"]["field"] == "count"


def test_bin_only_layer_and_aggregate_only_layer_compose_fine():
    """Sanity check: bin= on one layer + aggregate= on a DIFFERENT layer is fine.

    Only the combination on the SAME layer is forbidden.
    """
    df = pl.DataFrame(
        {
            "t": [1.0, 1.0, 2.0, 2.0],
            "v": [10.0, 20.0, 30.0, 50.0],
        }
    )
    binned = fm.Chart(df).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    agg = fm.Chart(df).mark_line().encode(x="t", y=fm.Y("v", aggregate="mean"))

    # Should not raise.
    d = json.loads((binned + agg).to_spec().to_json())
    layers = d["layers"]
    assert len(layers) == 2
