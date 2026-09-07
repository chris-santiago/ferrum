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

Two helpers here are shared with the SINGLE-CHART path in ``_spec_build.py``
rather than being layer-specific: ``_infer_agg_groupby`` (Altair-style
auto-groupby) and, since 2026-09-06, the encode-path bin lowering
``_lower_bin_encoding`` + ``_build_bin_transform`` (spec §6.6).  Bin lowering
lives here because the layered path already owned it and the two paths having
separate copies is precisely what left every standalone ``bin=`` idiom broken
(F-L01-01); one implementation is what keeps them from drifting again.

These are free functions operating on ``_Layer`` instances and transform lists.
``chart.py`` imports them, so this module must never import ``Chart`` at module
level (that would create an import cycle).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin
from ferrum._layer import _Layer

if TYPE_CHECKING:
    from ferrum._core import Bin as _Bin


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


def _layer_pending_bins(layer: _Layer) -> list[_PendingBin]:
    """Collect the encoding-level bin sentinels carried by *layer*."""
    pending: list[_PendingBin] = []
    for ch in layer.encoding.values():
        if isinstance(ch, ChannelBase):
            for t in ch.to_implicit_transforms():
                if isinstance(t, _PendingBin):
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


class _BinEncodingLowering(NamedTuple):
    """What lowering one encoding's ``bin=`` channel produced (spec §6.6).

    Attributes
    ----------
    encoding : dict
        The rewritten channel dict.  The binned channel points at the ``Bin``
        transform's start column, its span channel (``x2``/``y2``) at the end
        column, and an opposing ``aggregate=`` channel at either the Bin's own
        count column (the ``count()`` idiom) or the aggregate column produced
        by folding that aggregate into the same Bin.
    agg_kwargs : dict of str to str
        Extra ``Bin(...)`` keyword arguments (``agg_field``/``agg``) that fold
        the opposing aggregate into the Bin, or ``{}`` when nothing folded.
    absorbed : tuple of _PendingAggregate
        The ``_PendingAggregate`` sentinels this lowering consumed.  The Bin
        computes those values now, so a surviving ``Aggregate`` would run
        against a schema the Bin has already replaced -- :meth:`filter_transforms`
        is the structural enforcement of that "a caller that put them on a
        transform list MUST drop them" contract; callers should route through
        it rather than filtering this field by hand.
    """

    encoding: dict
    agg_kwargs: dict[str, str]
    absorbed: tuple[_PendingAggregate, ...]

    def build_bin(self, pending_bin: "_PendingBin", *, name: str | None = None) -> "_Bin":
        """Build this lowering's concrete ``Bin`` from *pending_bin* and its
        folded ``agg_kwargs``.

        The one implementation of "how a lowering becomes a real ``Bin``",
        called directly by the layered caller (:func:`_resolve_layer_bins`,
        which has no transform list of its own to filter -- see
        :meth:`filter_transforms`) and by the single-chart caller
        (``SpecBuildMixin._resolve_pending_bins``) before it calls
        :meth:`filter_transforms`.
        """
        return _build_bin_transform(pending_bin, self.agg_kwargs, name=name)

    def filter_transforms(self, effective_transforms: list, bin_xform: "_Bin") -> list:
        """Return a copy of *effective_transforms* with its ``_PendingBin``
        sentinel replaced by *bin_xform* and every absorbed
        ``_PendingAggregate`` dropped.

        The single-chart caller (``SpecBuildMixin._resolve_pending_bins``)
        routes the absorbed-aggregate drop through here, so the "a caller
        that put them on a transform list MUST drop them" contract
        documented on ``absorbed`` is enforced by this one function instead
        of narrated at the call site. The layered caller never calls this --
        a layer's aggregate sentinels never become transforms in the first
        place (``_resolve_layer_aggregates`` skips any layer that also
        bins), so it has no transform list of its own to filter and calls
        only :meth:`build_bin`.

        *effective_transforms* MUST contain the exact ``_PendingBin``
        sentinel *bin_xform* was built from. Silently returning a list that
        lacks the just-built Bin -- because no ``_PendingBin`` was found to
        replace -- is exactly the silent-drop shape this project's
        no-silent-fallback constraint forbids, so a list with no
        ``_PendingBin`` at all raises rather than quietly omitting the Bin.
        """
        absorbed_ids = {id(pa) for pa in self.absorbed}
        resolved: list = []
        replaced = False
        for t in effective_transforms:
            if isinstance(t, _PendingBin):
                resolved.append(bin_xform)
                replaced = True
            elif id(t) not in absorbed_ids:
                resolved.append(t)
        if not replaced:
            raise ValueError(
                "_BinEncodingLowering.filter_transforms: effective_transforms "
                "contains no _PendingBin sentinel to replace with the built "
                "Bin -- the caller's transform list and this lowering have "
                "gone out of sync (returning a filtered list here would "
                "silently drop the Bin from the output)."
            )
        return resolved


def _build_bin_transform(
    pending: _PendingBin, agg_kwargs: dict[str, str], *, name: str | None = None
) -> "_Bin":
    """Build the concrete ``Bin`` that one ``_PendingBin`` sentinel stands for.

    Shared by both encode paths: *name* is set on the layered path (a named
    fan-out transform the layer reads via ``data_source``) and left ``None`` on
    the single-chart path (an unnamed transform that advances the chain).

    A pre-built ``bin=Bin(...)`` instance comes back as-is -- the PyO3 type
    exposes no readable parameters, so it cannot be rebuilt, which is why
    :func:`_lower_bin_encoding` refuses to fold an aggregate into one rather
    than dropping the aggregate silently.  The layered serialiser
    (``_transforms_to_json_list_named``) injects the name from
    ``_NamedTransform.name``, so such an instance need not carry one.
    """
    from ferrum import Bin

    if pending.bin_obj is not None:
        return pending.bin_obj
    bin_kwargs = dict(pending.bin_kwargs)
    bin_kwargs.pop("name", None)  # never inherit a name from the Bin kwargs
    if name is not None:
        bin_kwargs["name"] = name
    # The folded aggregate wins over an agg_field/agg passed through bin={...}:
    # it is the one the encoding's own aggregate= channel names, so it keeps the
    # rewritten axis honest.
    return Bin(pending.field, **{**bin_kwargs, **agg_kwargs})


def _rebind_channel(ch: ChannelBase, column: str) -> ChannelBase:
    """Rebuild *ch* pointing at *column*, keeping the user's other kwargs.

    Drops the now-resolved ``bin=``/``aggregate=`` kwargs (both are consumed by
    the transform, never serialized) and recovers the user-facing column name as
    the axis ``title`` — :func:`ferrum._desugar._apply_remap`'s title-recovery
    convention, so a binned axis reads ``"tip"`` rather than ``"bin_start"``.  An
    explicit ``title=`` always wins; a non-``str`` field (an unresolved Repeat
    placeholder) recovers nothing.
    """
    kwargs = {k: v for k, v in ch._kwargs.items() if k not in ("bin", "aggregate")}
    if isinstance(ch.field, str) and ch.field != column and "title" not in kwargs:
        kwargs["title"] = ch.field
    return ch.__class__(column, **kwargs)


def _lower_bin_encoding(
    encoding: dict,
    pending_bins: list[_PendingBin],
    pending_aggregates: list[_PendingAggregate],
) -> _BinEncodingLowering:
    """Rewrite *encoding* around its ``bin=`` channel (spec §6.6).

    A ``Bin`` transform replaces its input batch's schema outright with its own
    columns, so every channel of the encoding it feeds has to be re-pointed at
    those columns before serialization.  This is the single implementation of
    that rewrite, shared by the single-chart path
    (``SpecBuildMixin._resolve_pending_bins``) and the layered path
    (:func:`_resolve_layer_bins`) so the two cannot drift apart again — the
    layered path having the remap and the single-chart path not having it is
    exactly what made every standalone ``bin=True`` idiom fail (F-L01-01).

    Four idioms are lowered:

    * the binned channel itself → the Bin's start column, plus its span
      channel (``x2``/``y2``) → the Bin's end column, so binned bars span
      their bin rather than collapsing to a default width;
    * ``y="count"`` / ``y="density"`` — a plain field reference to a column the
      Bin really emits, so nothing to do beyond the above;
    * ``y="count():Q"`` — the fieldless ``count`` aggregate, whose answer *is*
      the Bin's count column: the channel is re-pointed there and the pending
      aggregate absorbed;
    * ``y=Y("w", aggregate="mean")`` — an aggregate of a real column on the
      opposing channel, folded into the same Bin via ``agg_field``/``agg`` and
      re-pointed at the Bin's aggregate column.

    Column names come from the transform registry, never from literals, so a
    Rust-side rename raises here naming the stale column (spec §4.6 / M6a).

    Parameters
    ----------
    encoding : dict
        The channel dict to rewrite (never mutated; a copy is returned).
    pending_bins : list of _PendingBin
        The bin sentinels this encoding produced.
    pending_aggregates : list of _PendingAggregate
        The aggregate sentinels available to fold.

    Returns
    -------
    _BinEncodingLowering

    Raises
    ------
    ValueError
        If the encoding declares more than one bin, if one channel carries both
        ``bin=`` and ``aggregate=``, if more than one channel aggregates, if a
        pre-built ``bin=Bin(...)`` would have to absorb an aggregate, or if a
        fieldless non-``count`` aggregate is asked for per bin.
    """
    from ferrum._registry import registry_output_columns
    from ferrum.encoding import X2, Y2

    if not pending_bins:
        return _BinEncodingLowering(encoding, {}, ())
    if len(pending_bins) > 1:
        raise ValueError(
            "an encoding declares more than one bin=; a Bin transform replaces the "
            "batch schema with its own columns, so a second Bin would re-bin the "
            "first one's output. Bin a single channel, or use mark_hex() / "
            "Chart.transform(fm.Bin2D(x=..., y=...)) for two-dimensional binning."
        )

    bin_axes = [
        axis
        for axis, ch in encoding.items()
        if isinstance(ch, ChannelBase) and ch._kwargs.get("bin")
    ]
    if not bin_axes:
        # A stale sentinel: a later .encode() replaced the channel that carried
        # bin=. Nothing to re-point; the Bin still resolves as it always did.
        return _BinEncodingLowering(encoding, {}, ())
    bin_axis = bin_axes[0]

    # Only channels that HONOR aggregate= are scanned. ChannelBase's contract
    # (encoding/base.py, "Honored-kwarg contract") is that an unhonored kwarg is
    # warned once and dropped, so a Tooltip/Size/Color aggregate= must not reach
    # in and set the Bin's agg_field/agg or get re-pointed at agg_value -- that
    # would let a kwarg the library says it drops silently steer the pipeline.
    #
    # The bin= scan above is deliberately NOT gated the same way, and the
    # asymmetry is real rather than an oversight: `to_implicit_transforms` emits
    # the _PendingBin from the raw kwargs either way, so by the time this runs
    # the Bin is already on the transform list and the batch schema is already
    # replaced. Declining to re-point an unhonored bin= channel would not restore
    # the honored contract, only leave the chart pointed at a deleted column.
    # Gating aggregate= restores the contract; gating bin= would not.
    agg_axes = [
        axis
        for axis, ch in encoding.items()
        if isinstance(ch, ChannelBase)
        and "aggregate" in type(ch)._honored_kwargs
        and ch._kwargs.get("aggregate")
    ]
    # Checked before the same-channel refusal below: with the honored gate, only
    # X and Y honor aggregate=, so two aggregating channels means BOTH of them,
    # which necessarily includes the binned one. Refusing the count first names
    # the condition the user has to fix before either channel can work; ordering
    # these the other way round would make this arm unreachable.
    if len(agg_axes) > 1:
        raise ValueError(
            "a binned encoding aggregates on more than one channel "
            f"({', '.join(sorted(agg_axes))}); a Bin computes one aggregate column "
            "per bin. Keep a single aggregate= channel."
        )
    if bin_axis in agg_axes:
        raise ValueError(
            f"encoding channel {bin_axis!r} carries both bin= and aggregate=; one "
            "channel cannot be both the bin position and the aggregated value. "
            "Put the aggregate on the opposing positional channel."
        )

    cols = registry_output_columns("bin", ["bin_start", "bin_end", "count", "agg_value"])
    new_encoding = dict(encoding)
    new_encoding[bin_axis] = _rebind_channel(encoding[bin_axis], cols["bin_start"])

    # Span channel: bin= is honored only on the primary positional channels
    # (X, Y -- see encoding/_honored.PRIMARY_POSITIONAL), so x2/y2 are the only
    # spans a bin lowering can add. An explicitly bound span is left alone.
    # Spelled out per axis, mirroring _desugar._apply_remap's x2/y2 arms: a
    # channel-keyed lookup table would put the *channel* names "x2"/"y2" in a
    # dict-value position, where the M6a consumer sweep reads them as column
    # names (they collide with the linkage transform's segment columns).
    if bin_axis == "x" and "x2" not in new_encoding:
        new_encoding["x2"] = X2(cols["bin_end"], type="Q")
    elif bin_axis == "y" and "y2" not in new_encoding:
        new_encoding["y2"] = Y2(cols["bin_end"], type="Q")

    if not agg_axes:
        return _BinEncodingLowering(new_encoding, {}, ())

    agg_axis = agg_axes[0]
    agg_ch = encoding[agg_axis]
    source_field = agg_ch.field if isinstance(agg_ch.field, str) else ""
    method = agg_ch._kwargs["aggregate"]
    absorbed = tuple(
        pa for pa in pending_aggregates if pa.field == source_field and pa.agg == method
    )

    if not source_field:
        if method != "count":
            raise ValueError(
                f"aggregate={method!r} has no source column, so it cannot be computed "
                "per bin; only count() is field-less. Name the column to aggregate, "
                f"e.g. Y('value', aggregate={method!r})."
            )
        # count() over a binned batch IS the Bin's own count column.
        new_encoding[agg_axis] = _rebind_channel(agg_ch, cols["count"])
        return _BinEncodingLowering(new_encoding, {}, absorbed)

    if pending_bins[0].bin_obj is not None:
        raise ValueError(
            f"bin=Bin(...) on {bin_axis!r} cannot absorb the aggregate= on "
            f"{agg_axis!r}: a pre-built Bin exposes no readable parameters, so it "
            "cannot be rebuilt with the aggregate folded in. Declare it on the Bin "
            f"itself -- Bin({pending_bins[0].field!r}, agg_field={source_field!r}, "
            f"agg={method!r}) -- and bind the channel to {cols['agg_value']!r}."
        )
    new_encoding[agg_axis] = _rebind_channel(agg_ch, cols["agg_value"])
    return _BinEncodingLowering(new_encoding, {"agg_field": source_field, "agg": method}, absorbed)


def _resolve_layer_bins(layers: list) -> tuple[list, list]:
    """Resolve each layer's encoding bin sentinels into named chart-level transforms.

    Mirrors ``_resolve_layer_aggregates`` for the bin case.  The renderer never
    executes per-layer transforms standalone; a layer reads its input batch from
    a *named* chart-level transform via ``data_source``.  So for every layer
    that carries a ``bin=`` encoding kwarg this builds a named ``Bin``
    transform — reads from the original input (fan-out semantics, does not
    advance the unnamed chain) — points the layer's ``data_source`` at it, and
    re-points the layer's encoding at the Bin's own output columns via the
    shared :func:`_lower_bin_encoding`.

    Each binning layer gets its OWN named transform so two binning layers bin
    independently.

    A layer that carries ``aggregate=`` alongside ``bin=`` is handled here too,
    not refused: ``_resolve_layer_aggregates`` skipped it, and the aggregate is
    folded into this layer's own ``Bin`` (``agg_field``/``agg``), so no
    named→named chaining is needed.  The pre-``Bin``-aggregation refusal this
    replaced existed only because the ``Bin`` transform could not aggregate.

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
        bin_xform = lowering.build_bin(pending[0], name=bin_name)
        named_transforms.append(_NamedTransform(bin_xform, bin_name))

        new_layers.append(replace(layer, encoding=lowering.encoding, data_source=bin_name))

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
