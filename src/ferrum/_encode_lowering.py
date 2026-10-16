"""Encode-path lowering shared by the single-chart and layered spec builders.

An encoding can carry work that has to become a transform before the spec is
serialized: an ``aggregate=`` needs its auto-inferred groupby, and a ``bin=``
replaces the batch schema with the Bin's own columns, so every channel of the
encoding it feeds is re-pointed at those columns. The single-chart path
(``_spec_build.py``) and the layered path (``_layer_transforms.py``) both route
through the one implementation here so the two cannot drift apart (spec §6.6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin

if TYPE_CHECKING:
    from ferrum._core import Bin as _Bin


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


def _chart_agg_groupby(encoding: dict, facet) -> list[str]:
    """The groupby of a chart's own encoding aggregate: :func:`_infer_agg_groupby` plus facet fields.

    A faceted chart aggregates within each panel, so its row/column fields
    join the groupby. Shared by the flat chart's lowering and the per-layer
    reuse test, so the two read one rule.
    """
    fields = _infer_agg_groupby(encoding)
    if facet is not None:
        for name in (facet.col, facet.row, facet.field):
            if name and name not in fields:
                fields.append(name)
    return fields


def _bin_groupby(encoding: dict, facet) -> list[str]:
    """The groupby of a ``Bin`` lowered from *encoding*: its grouping channels' fields, plus facet.

    :func:`_chart_agg_groupby` over the channels that carry neither ``bin=``
    nor ``aggregate=``, without the binned channel's own field and without a
    field naming one of the Bin's own output columns (``y="count"`` draws
    the Bin's count, it does not group by it): the rows are binned per
    group, as a grouped ``mark_histogram`` bins them.
    """
    from ferrum._registry import registry_output_columns

    outputs = registry_output_columns(
        "bin", ["bin_start", "bin_end", "count", "density", "agg_value"]
    ).values()
    binned = {
        ch.field
        for ch in encoding.values()
        if isinstance(ch, ChannelBase) and ch._kwargs.get("bin") and isinstance(ch.field, str)
    } | set(outputs)
    grouping = {
        channel: ch
        for channel, ch in encoding.items()
        if not (
            isinstance(ch, ChannelBase) and (ch._kwargs.get("bin") or ch._kwargs.get("aggregate"))
        )
    }
    return [f for f in _chart_agg_groupby(grouping, facet) if f not in binned]


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
    groupby : tuple of str
        The columns the ``Bin`` partitions its rows by (:func:`_bin_groupby`);
        empty for an ungrouped Bin.
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
    groupby: tuple[str, ...] = ()

    def build_bin(self, pending_bin: "_PendingBin", *, name: str | None = None) -> "_Bin":
        """Build this lowering's concrete ``Bin`` from *pending_bin* and its
        folded ``agg_kwargs``.

        The one implementation of "how a lowering becomes a real ``Bin``",
        called directly by the layered caller (:func:`~ferrum._layer_transforms._resolve_layer_bins`,
        which has no transform list of its own to filter -- see
        :meth:`filter_transforms`) and by the single-chart caller
        (``SpecBuildMixin._resolve_pending_bins``) before it calls
        :meth:`filter_transforms`.
        """
        return _build_bin_transform(pending_bin, self.agg_kwargs, self.groupby, name=name)

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
    pending: _PendingBin,
    agg_kwargs: dict[str, str],
    groupby: tuple[str, ...] = (),
    *,
    name: str | None = None,
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

    A non-empty *groupby* bins each group's rows over one set of edges
    (``shared_extent``), as a grouped ``mark_histogram`` does.
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
    if groupby:
        bin_kwargs["groupby"] = list(groupby)
        bin_kwargs["shared_extent"] = True
    return Bin(pending.field, **{**bin_kwargs, **agg_kwargs})


def _rebind_channel(ch: ChannelBase, column: str) -> ChannelBase:
    """Rebuild *ch* pointing at *column*, keeping the user's other kwargs.

    Drops the now-resolved ``bin=``/``aggregate=`` kwargs (both are consumed by
    the transform, never serialized) and recovers the user-facing column name as
    the axis ``title`` — the one re-pointing rule :func:`ferrum._desugar._apply_remap`
    also uses, so a binned axis reads ``"tip"`` rather than ``"bin_start"``.  An
    explicit ``title=`` always wins; a non-``str`` field (an unresolved Repeat
    placeholder) recovers nothing, and neither does a channel that does not
    honor ``title=`` (a tooltip, a text label), which keeps no title.
    """
    kwargs = {k: v for k, v in ch._kwargs.items() if k not in ("bin", "aggregate")}
    if (
        isinstance(ch.field, str)
        and ch.field != column
        and "title" not in kwargs
        and "title" in type(ch)._honored_kwargs
    ):
        kwargs["title"] = ch.field
    return ch.__class__(column, **kwargs)


def _lower_bin_encoding(
    encoding: dict,
    pending_bins: list[_PendingBin],
    pending_aggregates: list[_PendingAggregate],
    groupby: tuple[str, ...] = (),
) -> _BinEncodingLowering:
    """Rewrite *encoding* around its ``bin=`` channel (spec §6.6).

    A ``Bin`` transform replaces its input batch's schema outright with its own
    columns, so every channel of the encoding it feeds has to be re-pointed at
    those columns before serialization.  This is the single implementation of
    that rewrite, shared by the single-chart path
    (``SpecBuildMixin._resolve_pending_bins``) and the layered path
    (:func:`~ferrum._layer_transforms._resolve_layer_bins`) so the two cannot
    drift apart again — the layered path having the remap and the single-chart path not having it is
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
    groupby : tuple of str
        The columns the Bin partitions its rows by: :func:`_bin_groupby` of the
        encoding the binned layer draws with (its own channels over the ones
        it inherits). Empty for an ungrouped Bin.

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
    # warned once and dropped, so a Detail/Shape aggregate= must not reach in
    # and set the Bin's agg_field/agg or get re-pointed at agg_value -- that
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
    # The Bin has one aggregate column. A fieldless count() reads the Bin's own
    # count column and takes no slot; channels aggregating the same (field, op)
    # share the one slot. Checked before the same-channel refusal below:
    # refusing the count first names the condition the user has to fix before
    # either channel can work.
    slot_axes = [
        axis
        for axis in agg_axes
        if not (
            encoding[axis].field in (None, "") and encoding[axis]._kwargs["aggregate"] == "count"
        )
    ]
    slots = {(encoding[axis].field, encoding[axis]._kwargs["aggregate"]) for axis in slot_axes}
    if len(slots) > 1:
        raise ValueError(
            "a binned encoding aggregates on more than one channel "
            f"({', '.join(sorted(slot_axes))}); a Bin computes one aggregate column "
            "per bin. Keep a single aggregate= channel."
        )
    if bin_axis in agg_axes:
        raise ValueError(
            f"encoding channel {bin_axis!r} carries both bin= and aggregate=; one "
            "channel cannot be both the bin position and the aggregated value. "
            "Put the aggregate on the opposing positional channel."
        )

    groupby = tuple(groupby)
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

    absorbed: list = []
    for axis in agg_axes:
        ch = encoding[axis]
        if axis in slot_axes:
            continue
        # count() over a binned batch IS the Bin's own count column.
        new_encoding[axis] = _rebind_channel(ch, cols["count"])
        absorbed += [pa for pa in pending_aggregates if not pa.field and pa.agg == "count"]
    if not slots:
        return _BinEncodingLowering(new_encoding, {}, tuple(absorbed), groupby)

    ((source_field, method),) = slots
    if not isinstance(source_field, str) or not source_field:
        raise ValueError(
            f"aggregate={method!r} has no source column, so it cannot be computed "
            "per bin; only count() is field-less. Name the column to aggregate, "
            f"e.g. Y('value', aggregate={method!r})."
        )
    if pending_bins[0].bin_obj is not None:
        raise ValueError(
            f"bin=Bin(...) on {bin_axis!r} cannot absorb the aggregate= on "
            f"{slot_axes[0]!r}: a pre-built Bin exposes no readable parameters, so it "
            "cannot be rebuilt with the aggregate folded in. Declare it on the Bin "
            f"itself -- Bin({pending_bins[0].field!r}, agg_field={source_field!r}, "
            f"agg={method!r}) -- and bind the channel to {cols['agg_value']!r}."
        )
    for axis in slot_axes:
        new_encoding[axis] = _rebind_channel(encoding[axis], cols["agg_value"])
    absorbed += [pa for pa in pending_aggregates if pa.field == source_field and pa.agg == method]
    return _BinEncodingLowering(
        new_encoding, {"agg_field": source_field, "agg": method}, tuple(absorbed), groupby
    )
