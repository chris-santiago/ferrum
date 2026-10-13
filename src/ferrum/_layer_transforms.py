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

The encode-path lowering these resolvers call (auto-groupby inference and
``bin=`` lowering) is shared with the single-chart path and lives in
``_encode_lowering.py``.

These are free functions operating on ``_Layer`` instances and transform lists.
``chart.py`` imports them, so this module must never import ``Chart`` at module
level (that would create an import cycle).
"""

from __future__ import annotations

import json
from dataclasses import replace

from ferrum._encode_lowering import _infer_agg_groupby, _lower_bin_encoding
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


class _Branch:
    """A named sub-pipeline over a chosen input, serialized as one ``branch`` step.

    ``transforms`` runs as its own chain (unnamed steps chain, named steps
    publish) starting from the pipeline's original batch when ``input`` is
    ``None``, or from the output already published under ``input``. The
    sub-chain tail is published under ``name`` and every named or secondary
    output inside it under its own key. A branch never advances the outer
    unnamed chain, so a layer reading ``name`` sees its own chain applied to
    its own input whatever sits before it in the composed pipeline.

    The original transform objects are kept (not pre-serialized) so value
    equality and the identity-derived names ``+`` gives its steps stay valid
    for as long as the branch lives.
    """

    __slots__ = ("name", "input", "transforms")

    def __init__(self, name: str, input: str | None, transforms: list) -> None:
        self.name = name
        self.input = input
        self.transforms = list(transforms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Branch):
            return NotImplemented
        return (
            self.name == other.name
            and self.input == other.input
            and len(self.transforms) == len(other.transforms)
            and all(_same_step(a, b) for a, b in zip(self.transforms, other.transforms))
        )

    __hash__ = None  # type: ignore[assignment]


def _same_step(a: object, b: object) -> bool:
    """Whether two pipeline steps publish the same output under the same key.

    A named step and an unnamed one never match, even when the wrapped
    transform is value-equal: one advances the chain and the other publishes
    a key a layer reads, so collapsing them would strand that layer.
    """
    if a is b:
        return True
    if isinstance(a, _NamedTransform) or isinstance(b, _NamedTransform):
        return (
            isinstance(a, _NamedTransform)
            and isinstance(b, _NamedTransform)
            and a.name == b.name
            and a.transform == b.transform
        )
    return bool(a == b)


def _advances_chain(step: object) -> bool:
    """Whether *step* is an unnamed step, so a later step reads its output as the tail.

    Named steps (a ``_NamedTransform``, a ``_Branch``, a dict or transform
    object carrying its own ``name``) publish beside the chain and leave the
    tail where it was.
    """
    if isinstance(step, (_NamedTransform, _Branch)):
        return False
    entry = step if isinstance(step, dict) else _transforms_to_json_list_named([step])[0]
    return entry.get("name") is None


def _first_chain_step(transforms: list) -> int:
    """Index of the first unnamed step in *transforms*, or its length when there is none.

    Every step inserted at or before this index reads the pipeline's original
    batch, because no unnamed step has run yet.
    """
    for i, step in enumerate(transforms):
        if _advances_chain(step):
            return i
    return len(transforms)


def _layer_pending_aggregates(layer: _Layer) -> list[_PendingAggregate]:
    """Collect the encoding-level aggregate sentinels carried by *layer*."""
    pending: list[_PendingAggregate] = []
    for ch in layer.encoding.values():
        if isinstance(ch, ChannelBase):
            for t in ch.to_implicit_transforms():
                if isinstance(t, _PendingAggregate):
                    pending.append(t)
    return pending


def _layer_pending_bins(layer: _Layer) -> list[_PendingBin]:
    """Collect the encoding-level bin sentinels carried by *layer*."""
    pending: list[_PendingBin] = []
    for ch in layer.encoding.values():
        if isinstance(ch, ChannelBase):
            for t in ch.to_implicit_transforms():
                if isinstance(t, _PendingBin):
                    pending.append(t)
    return pending


def _layer_step(layer: _Layer, transform: object, name: str) -> "_NamedTransform | _Branch":
    """The named step a layer's own aggregate or bin runs as.

    A layer reading the pipeline tail (no ``data_source``) gets a plain named
    step, which reads that tail.  A layer that reads a named output gets a
    one-step ``_Branch`` over that output, so the aggregate or bin sees the
    rows the layer draws rather than the tail of another operand's chain.
    *transform* itself must be unnamed for the branch case, so it advances
    the branch's chain.
    """
    if layer.data_source is None:
        return _NamedTransform(transform, name)
    return _Branch(name, layer.data_source, [transform])


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

    A layer that already carries a ``data_source`` (a ``+`` operand routed to
    its own chain or to the source) aggregates that named output: its step is
    a ``_Branch`` over ``data_source`` (see :func:`_layer_step`), and
    re-pointing the layer's ``data_source`` to the aggregate keeps the
    ``inherit_non_positional`` routing.  Layers routed to an explicit
    ``data_source`` *with no pending aggregate* (e.g. composite-mark desugar
    layers) pass through unchanged.

    A layer that ALSO carries ``bin=`` is skipped here and left to
    :func:`_resolve_layer_bins`, which folds the aggregate into that layer's own
    ``Bin`` (spec §6.6).  Building a separate named ``Aggregate`` for it first
    would strand the layer between two named transforms it cannot chain.

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
        if not pending or _layer_pending_bins(layer):
            new_layers.append(layer)
            continue

        groupby = _infer_agg_groupby(layer.encoding)
        ops = [AggregateOp(p.field, p.agg, p.output_col) for p in pending]
        agg_name = f"_layer_agg_{i}"
        named_transforms.append(_layer_step(layer, Aggregate(ops, groupby=groupby), agg_name))

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


def _resolve_layer_bins(layers: list) -> tuple[list, list]:
    """Resolve each layer's encoding bin sentinels into named chart-level transforms.

    Mirrors ``_resolve_layer_aggregates`` for the bin case.  The renderer never
    executes per-layer transforms standalone; a layer reads its input batch from
    a *named* chart-level transform via ``data_source``.  So for every layer
    that carries a ``bin=`` encoding kwarg this builds a named ``Bin``
    transform — reads the layer's input (the chain tail, or the layer's own
    named output) without advancing the unnamed chain — points the layer's
    ``data_source`` at it, and
    re-points the layer's encoding at the Bin's own output columns via the
    shared :func:`_lower_bin_encoding`.

    Each binning layer gets its OWN named transform so two binning layers bin
    independently.

    A layer that carries ``aggregate=`` alongside ``bin=`` is handled here too,
    not refused: ``_resolve_layer_aggregates`` skipped it, and the aggregate is
    folded into this layer's own ``Bin`` (``agg_field``/``agg``), so no
    named→named chaining is needed.  The pre-``Bin``-aggregation refusal this
    replaced existed only because the ``Bin`` transform could not aggregate.

    A layer that already carries a ``data_source`` bins that named output
    (a ``_Branch`` over it, see :func:`_layer_step`) and its ``data_source``
    is updated to the named bin.  Layers with a pre-set ``data_source`` but no
    pending bin pass through unchanged.

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
    new_layers: list = []
    named_transforms: list = []
    for i, layer in enumerate(layers):
        pending = _layer_pending_bins(layer)
        if not pending:
            new_layers.append(layer)
            continue

        # _lower_bin_encoding refuses a layer that declares more than one bin,
        # so exactly one sentinel survives to build the transform from. A
        # layer's aggregate sentinels never become transforms in the first
        # place (_resolve_layer_aggregates skips any layer that also bins),
        # so there is no transform list of this layer's own to filter --
        # only `build_bin` is needed here, not `filter_transforms` (which the
        # single-chart caller uses, since ITS sentinels DO sit on the chart's
        # transform list).
        lowering = _lower_bin_encoding(layer.encoding, pending, _layer_pending_aggregates(layer))
        bin_name = f"_layer_bin_{i}"
        # Inside a branch the Bin must stay unnamed, so it advances the
        # branch's own chain and its output is what the branch publishes.
        bin_xform = lowering.build_bin(
            pending[0], name=bin_name if layer.data_source is None else None
        )
        named_transforms.append(_layer_step(layer, bin_xform, bin_name))

        new_layers.append(replace(layer, encoding=lowering.encoding, data_source=bin_name))

    return new_layers, named_transforms


def _transforms_to_json_list_named(transforms: list) -> list:
    """Serialize a mixed transform list to the Rust ``TransformSpec`` wire shape.

    The list may contain three kinds of entries, in any order:

    * **Plain dicts** — the Phase-12 ``transform_*`` functions
      (``transform_bin``, ``transform_aggregate``, ...) each return a dict that
      already matches the ``#[serde(tag = "type")]`` ``TransformSpec`` shape, so
      they pass through unchanged.  This is the *only* construction path for the
      Phase-12 data transforms; they have no typed pyclass (SEAM-02).
    * **Live PyO3 transform objects** — the stat transforms (``Bin``,
      ``Aggregate``, ``Identity``, ...) produced by per-layer aggregate/bin
      resolution and ``+`` composition.  These are serialized via a
      ``ChartSpec`` round-trip (the only way to reach their serde shape).
    * **A** :class:`_Branch` — serialized as
      ``{"type": "branch", "name", "input", "transforms"}`` with its own
      sub-chain serialized by this same function.

    A dict or PyO3 entry may be wrapped in a :class:`_NamedTransform`, whose
    ``name`` is injected into the serialized entry.  Mixing named and unnamed in
    one list is valid — unnamed transforms chain, named transforms fan out
    without advancing ``FINAL_OUTPUT_KEY``.
    """
    if not transforms:
        return []
    from ferrum import ChartSpec

    result: list = []
    # Runs of PyO3 objects are batch-serialized through one ChartSpec.
    pyo3_batch: list = []

    def flush() -> None:
        if not pyo3_batch:
            return
        dummy = ChartSpec(
            mark="point",
            x="__x__",
            y="__y__",
            transforms=[t.transform if isinstance(t, _NamedTransform) else t for t in pyo3_batch],
        )
        serialized = json.loads(dummy.to_json()).get("transforms", [])
        for j, original in enumerate(pyo3_batch):
            entry = serialized[j] if j < len(serialized) else {}
            if isinstance(original, _NamedTransform):
                entry["name"] = original.name
            result.append(entry)
        pyo3_batch.clear()

    for t in transforms:
        if isinstance(t, _Branch):
            flush()
            result.append(
                {
                    "type": "branch",
                    "name": t.name,
                    "input": t.input,
                    "transforms": _transforms_to_json_list_named(t.transforms),
                }
            )
            continue
        inner = t.transform if isinstance(t, _NamedTransform) else t
        if isinstance(inner, dict):
            flush()
            entry = dict(inner)
            if isinstance(t, _NamedTransform):
                entry["name"] = t.name
            result.append(entry)
        else:
            pyo3_batch.append(t)
    flush()
    return result
