"""Layered-transform routing for the :class:`~ferrum.chart.Chart` value class.

This module owns the free functions that resolve per-layer encoding aggregates
and bins into *named* chart-level transforms, plus the ``_NamedTransform``
wrapper and the transform-to-JSON serialisers.  They were extracted from
``chart.py`` (cohesion finding CHART-01) so the chart module keeps only the
fluent surface and ``to_spec`` orchestration.

The renderer never executes a layer's own ``transforms`` standalone; a layer
reads its input batch from a *named* chart-level transform via ``data_source``.
The resolvers here build those named transforms and re-point the layer's
``data_source`` accordingly, keeping serialised output byte-identical to the
pre-extraction behaviour.

These are free functions operating on ``_Layer`` instances and transform lists.
``chart.py`` imports them, so this module must never import ``Chart`` at module
level (that would create an import cycle).
"""

from __future__ import annotations

import json
from dataclasses import replace

from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin
from ferrum._layer import _Layer


class _NamedTransform:
    """Pairs a PyO3 transform object with an explicit name for chart-level serialization.

    When a single-mark chart's transforms are promoted to chart level during
    ``+`` composition, we need them to be *named* in the Rust pipeline so that
    ``FINAL_OUTPUT_KEY`` remains the original input batch (named transforms do
    not advance the unnamed chain).  The corresponding ``_Layer`` sets
    ``data_source`` to the same name so it reads the correct output.
    """

    __slots__ = ("transform", "name")

    def __init__(self, transform: object, name: str) -> None:
        self.transform = transform
        self.name = name


def _infer_agg_groupby(encoding: dict) -> list[str]:
    """Collect groupby fields for an encoding-level aggregate (Altair auto-groupby).

    Returns the fields of every channel that carries *no* ``aggregate`` kwarg,
    in first-seen order with duplicates removed.  Plain string-valued channels
    (e.g. ``encode(x="t")``) and ``ChannelBase`` channels without an aggregate
    both contribute their field; channels with an ``aggregate`` kwarg are the
    measures being summarised and are excluded.

    Shared by the single-chart path (``_resolve_pending_aggregates``, which also
    folds in facet dimensions) and the per-layer path (``_resolve_layer_aggregates``).
    """
    from ferrum.repeat import _RepeatPlaceholder as _RPH

    fields: list[str] = []
    for ch in encoding.values():
        if isinstance(ch, ChannelBase):
            if ch._kwargs.get("aggregate"):
                continue
            f = ch.field
        elif isinstance(ch, str):
            f = ch
        else:
            continue
        if f is None or isinstance(f, _RPH):
            continue
        if f not in fields:
            fields.append(f)
    return fields


def _layer_pending_aggregates(layer: _Layer) -> list[_PendingAggregate]:
    """Collect the encoding-level aggregate sentinels carried by *layer*."""
    pending: list[_PendingAggregate] = []
    for ch in layer.encoding.values():
        if isinstance(ch, ChannelBase):
            for t in ch.to_implicit_transforms():
                if isinstance(t, _PendingAggregate):
                    pending.append(t)
    return pending


def _resolve_layer_aggregates(layers: list) -> tuple[list, list]:
    """Resolve each layer's encoding aggregates into named chart-level transforms.

    Mirrors ``Chart._resolve_pending_aggregates`` for the layered path, but
    respects the render architecture: per-layer ``transforms`` are never executed
    on their own; a layer reads its input batch from a *named* chart-level
    transform via ``data_source``.  So for every layer that carries an
    ``aggregate=`` encoding kwarg (e.g. ``Y("v", aggregate="mean")`` or the
    ``count()`` shorthand) this builds a named ``Aggregate`` transform — groupby
    inferred from the layer's own non-aggregate encoding fields — points the
    layer's ``data_source`` at it, and remaps the aggregated channel's field to
    the transform's output column.

    Each aggregating layer gets its OWN named transform, so two aggregating
    layers aggregate independently and never share a groupby.

    A layer that already carries a ``data_source`` is handled too: the
    ``+`` column-overlap path pre-assigns a named ``_ident_`` identity source to
    RHS layers before this resolver runs.  In the named-transform model every
    named transform reads the same input (the unnamed-chain tail / original
    batch), so the new ``Aggregate`` reads exactly what the ``_ident_`` source
    read, and re-pointing the layer's ``data_source`` to the aggregate keeps the
    ``inherit_non_positional`` routing the overlap path needs.  Only layers
    routed to an explicit non-aggregate ``data_source`` *with no pending
    aggregate* (e.g. composite-mark desugar layers) pass through unchanged,
    keeping serialized output byte-identical.

    Note: per-layer ``bin=`` encodings have the same per-layer-transform-never-run
    gap and are not yet resolved here; only ``aggregate=`` is handled.

    Parameters
    ----------
    layers :
        The chart's resolved ``_Layer`` list.

    Returns
    -------
    (new_layers, named_transforms)
        ``new_layers`` is the layer list with aggregating layers rewritten;
        ``named_transforms`` is the list of ``_NamedTransform`` objects to add to
        the chart-level pipeline (empty when no layer aggregates).
    """
    from ferrum import Aggregate, AggregateOp

    new_layers: list = []
    named_transforms: list = []
    for i, layer in enumerate(layers):
        pending = _layer_pending_aggregates(layer)
        if not pending:
            new_layers.append(layer)
            continue

        groupby = _infer_agg_groupby(layer.encoding)
        ops = [AggregateOp(p.field, p.agg, p.output_col) for p in pending]
        agg_name = f"_layer_agg_{i}"
        named_transforms.append(_NamedTransform(Aggregate(ops, groupby=groupby), agg_name))

        # Remap each aggregated channel's field to its output column.  count()
        # shorthands carry field="" (no source column); key those on "".
        field_remap = {p.field if p.field else "": p.output_col for p in pending}
        new_encoding: dict = {}
        for axis, ch in layer.encoding.items():
            if isinstance(ch, ChannelBase):
                _remap_key = ch.field if ch.field is not None else ""
                if _remap_key in field_remap and ch._kwargs.get("aggregate"):
                    # Rebuild the channel pointing at the aggregated output
                    # column, dropping the now-resolved aggregate kwarg.
                    kwargs = {k: v for k, v in ch._kwargs.items() if k != "aggregate"}
                    new_encoding[axis] = ch.__class__(field_remap[_remap_key], **kwargs)
                    continue
            new_encoding[axis] = ch

        new_layers.append(replace(layer, encoding=new_encoding, data_source=agg_name))

    return new_layers, named_transforms


def _layer_pending_bins(layer: _Layer) -> list[_PendingBin]:
    """Collect the encoding-level bin sentinels carried by *layer*."""
    pending: list[_PendingBin] = []
    for ch in layer.encoding.values():
        if isinstance(ch, ChannelBase):
            for t in ch.to_implicit_transforms():
                if isinstance(t, _PendingBin):
                    pending.append(t)
    return pending


def _resolve_layer_bins(layers: list) -> tuple[list, list]:
    """Resolve each layer's encoding bin sentinels into named chart-level transforms.

    Mirrors ``_resolve_layer_aggregates`` for the bin case.  The renderer never
    executes per-layer transforms standalone; a layer reads its input batch from
    a *named* chart-level transform via ``data_source``.  So for every layer
    that carries a ``bin=`` encoding kwarg this builds a named ``Bin``
    transform — reads from the original input (fan-out semantics, does not
    advance the unnamed chain) — points the layer's ``data_source`` at it, and
    remaps the binned channel's field to ``bin_start`` (the Bin output column),
    stripping the now-resolved ``bin`` kwarg.

    Each binning layer gets its OWN named transform so two binning layers bin
    independently.

    A layer that already carries a ``data_source`` is handled correctly:
    if it also has a pending bin, the bin transform is still created and the
    layer's ``data_source`` is updated to the named bin.  Layers with a
    pre-set ``data_source`` but no pending bin pass through unchanged.

    Parameters
    ----------
    layers :
        The chart's resolved ``_Layer`` list, already processed by
        ``_resolve_layer_aggregates`` (aggregate routing is preserved).

    Returns
    -------
    (new_layers, named_transforms)
        ``new_layers`` is the layer list with binning layers rewritten;
        ``named_transforms`` is the list of ``_NamedTransform`` objects to add
        to the chart-level pipeline (empty when no layer bins).
    """
    from ferrum import Bin

    new_layers: list = []
    named_transforms: list = []
    for i, layer in enumerate(layers):
        pending = _layer_pending_bins(layer)
        if not pending:
            new_layers.append(layer)
            continue

        # Guard: detect a layer whose data_source was already set by the
        # aggregate resolver (_layer_agg_N).  This means the layer had both
        # bin= and aggregate= on different encoding channels.  In the layered
        # named-transform architecture there is no named→named chaining: named
        # transforms all read from the same unnamed-chain tail.  Overwriting the
        # aggregate data_source with the bin data_source would silently point the
        # layer at the original (unaggregated) input while encoding y still
        # references the aggregated output column — a missing-column error at
        # render time.  Raise rather than produce silent garbage.
        if layer.data_source is not None and layer.data_source.startswith("_layer_agg_"):
            raise ValueError(
                "A layer cannot have both bin= and aggregate= encoding kwargs in a layered "
                "chart.  The layered named-transform architecture does not support chaining "
                "a per-layer Bin into a per-layer Aggregate (named transforms cannot read "
                "from other named transforms' outputs).  "
                "Use separate layers: one layer with bin= and one layer with aggregate=."
            )

        # Use the first pending bin (a layer encoding typically has one binned
        # channel).  Multiple bins on the same layer would each create their
        # own named transform; here we use index `i` plus a sub-index.
        new_encoding: dict = {}
        layer_named_transforms: list = []

        for sub_i, pb in enumerate(pending):
            bin_name = f"_layer_bin_{i}" if len(pending) == 1 else f"_layer_bin_{i}_{sub_i}"
            if pb.bin_obj is not None:
                # Pre-built Bin instance — use it directly inside the named
                # wrapper.  The serialiser (_transforms_to_json_list_named)
                # injects the name field from _NamedTransform.name, so the
                # Bin object itself does not need to carry a name.
                layer_named_transforms.append(_NamedTransform(pb.bin_obj, bin_name))
            else:
                bin_kwargs = dict(pb.bin_kwargs)
                bin_kwargs.pop("name", None)  # never inherit a name from the Bin kwargs
                bin_xform = Bin(pb.field, name=bin_name, **bin_kwargs)
                layer_named_transforms.append(_NamedTransform(bin_xform, bin_name))

        named_transforms.extend(layer_named_transforms)

        # The last (or only) bin's name becomes the layer's data_source.
        final_bin_name = layer_named_transforms[-1].name

        # Remap each binned channel's field to bin_start, strip bin kwarg.
        for axis, ch in layer.encoding.items():
            if isinstance(ch, ChannelBase) and ch._kwargs.get("bin"):
                kwargs = {k: v for k, v in ch._kwargs.items() if k != "bin"}
                new_encoding[axis] = ch.__class__("bin_start", **kwargs)
                continue
            new_encoding[axis] = ch

        new_layers.append(replace(layer, encoding=new_encoding, data_source=final_bin_name))

    return new_layers, named_transforms


def _transforms_to_json_list(transforms: list) -> list:
    """Serialize a list of Python transform objects to JSON-safe dicts.

    ``coerce_layers`` in Rust calls ``json.dumps()`` on each layer dict, so
    PyO3 transform objects must be converted to plain dicts first.  We do
    this by round-tripping through ``ChartSpec.to_json()``.
    """
    if not transforms:
        return []
    from ferrum import ChartSpec

    # Build a minimal spec with the transforms; extract the "transforms" array.
    dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=transforms)
    parsed = json.loads(dummy.to_json())
    return parsed.get("transforms", [])


def _transforms_to_json_list_named(transforms: list) -> list:
    """Serialize a mixed transform list to the Rust ``TransformSpec`` wire shape.

    The list may contain two kinds of entries, in any order:

    * **Plain dicts** — the Phase-12 ``transform_*`` functions
      (``transform_bin``, ``transform_aggregate``, ...) each return a dict that
      already matches the ``#[serde(tag = "type")]`` ``TransformSpec`` shape, so
      they pass through unchanged.  This is the *only* construction path for the
      Phase-12 data transforms; they have no typed pyclass (SEAM-02).
    * **Live PyO3 transform objects** — the stat transforms (``Bin``,
      ``Aggregate``, ``Identity``, ...) produced by per-layer aggregate/bin
      resolution and ``+`` composition.  These are serialized via a
      ``ChartSpec`` round-trip (the only way to reach their serde shape).

    Either kind may be wrapped in a :class:`_NamedTransform`, whose ``name`` is
    injected into the serialized entry.  Mixing named and unnamed in one list is
    valid — unnamed transforms chain, named transforms fan out without advancing
    ``FINAL_OUTPUT_KEY``.
    """
    if not transforms:
        return []
    from ferrum import ChartSpec

    # Separate dicts (already JSON-ready) from PyO3 objects (need round-trip).
    # Build the output list preserving order.
    result: list = []
    # Collect runs of PyO3 objects to batch-serialize through ChartSpec.
    pyo3_batch: list = []  # (original_index, transform_item) pairs
    for i, t in enumerate(transforms):
        inner = t.transform if isinstance(t, _NamedTransform) else t
        if isinstance(inner, dict):
            # Flush any pending PyO3 batch first.
            if pyo3_batch:
                pyo3_objs = [item for _, item in pyo3_batch]
                dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=pyo3_objs)
                serialized = json.loads(dummy.to_json()).get("transforms", [])
                for j, (orig_idx, orig_t) in enumerate(pyo3_batch):
                    entry = serialized[j] if j < len(serialized) else {}
                    if isinstance(transforms[orig_idx], _NamedTransform):
                        entry["name"] = transforms[orig_idx].name
                    result.append(entry)
                pyo3_batch = []
            # Dict transform — pass through, inject name if wrapped.
            entry = dict(inner)
            if isinstance(t, _NamedTransform):
                entry["name"] = t.name
            result.append(entry)
        else:
            pyo3_batch.append((i, inner))
    # Flush remaining PyO3 batch.
    if pyo3_batch:
        pyo3_objs = [item for _, item in pyo3_batch]
        dummy = ChartSpec(mark="point", x="__x__", y="__y__", transforms=pyo3_objs)
        serialized = json.loads(dummy.to_json()).get("transforms", [])
        for j, (orig_idx, orig_t) in enumerate(pyo3_batch):
            entry = serialized[j] if j < len(serialized) else {}
            if isinstance(transforms[orig_idx], _NamedTransform):
                entry["name"] = transforms[orig_idx].name
            result.append(entry)
    return result
