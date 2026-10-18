"""Encode-path lowering shared by the single-chart and layered spec builders.

An encoding can carry work that has to become a transform before the spec is
serialized: an ``aggregate=`` needs its auto-inferred groupby, and a ``bin=``
replaces the batch schema with the Bin's own columns, so every channel of the
encoding it feeds is re-pointed at those columns. The single-chart path
(``_spec_build.py``) and the layered path (``_layer_transforms.py``) both route
through the one implementation here so the two cannot drift apart (spec §6.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin

if TYPE_CHECKING:
    from ferrum._core import Aggregate, Bin

#: The encoding key of a layer's grouping-only bindings (:class:`_GroupingOnly`),
#: written to the wire as the layer's ``group`` field.
GROUPING_KEY = "group"


@dataclass(frozen=True)
class _GroupingOnly:
    """Inherited bindings a layer groups its statistic by but does not draw.

    Not a channel: no reader takes it for a binding. The statistic plan
    (:func:`_statistic_groupby`, :func:`_lower_bin_encoding`) expands it into
    its members in place, so each member groups exactly as the raw binding
    would.
    *bindings* holds ``(channel, binding)`` pairs in the parent's order.
    """

    bindings: tuple[tuple[str, Any], ...]

    def wire_field(self) -> Any:
        """The field the parent draws as color, under the color alias rules (``_ENCODING_ALIASES``)."""
        from ferrum.encoding._aliases import _channel_field, resolve_channel_alias

        return _channel_field(resolve_channel_alias(dict(self.bindings), "color"))


def _expand_grouping(encoding: dict) -> dict:
    """*encoding* with a :class:`_GroupingOnly` replaced in place by its member bindings."""
    if not any(isinstance(value, _GroupingOnly) for value in encoding.values()):
        return encoding
    out: dict = {}
    for channel, value in encoding.items():
        if isinstance(value, _GroupingOnly):
            for member, binding in value.bindings:
                out.setdefault(member, binding)
        else:
            out[channel] = value
    return out


def _grouping_of(encoding: dict) -> "_GroupingOnly | None":
    """*encoding*'s grouping-only key, or ``None`` when it holds none."""
    value = encoding.get(GROUPING_KEY)
    return value if isinstance(value, _GroupingOnly) else None


def _regroup(encoding: dict, grouping: "_GroupingOnly | None") -> dict:
    """Fold the members of *grouping*, as *encoding* now binds them, back into one key.

    The inverse of :func:`_expand_grouping` after a lowering re-pointed a
    member (a binned color re-pointed at the Bin's start column): the key
    sits at the first member's place and keeps the members' order.
    """
    if grouping is None:
        return encoding
    names = [member for member, _ in grouping.bindings]
    out: dict = {}
    for channel, value in encoding.items():
        if channel not in names:
            out[channel] = value
        elif GROUPING_KEY not in out:
            out[GROUPING_KEY] = _GroupingOnly(
                tuple((name, encoding[name]) for name in names if name in encoding)
            )
    return out


#: The ``Bin`` parameters that make it aggregate, group or accumulate its rows,
#: in the order a message names them. Rust refuses each beside ``per_row``.
_BIN_STATISTIC_PARAMETERS = ("groupby", "cumulative", "agg_field/agg", "aggregates")


@dataclass(frozen=True)
class _BinDeclaration:
    """What the ``Bin`` a ``bin=`` channel names declares, read off its wire.

    Attributes
    ----------
    spelling : str
        How the user wrote it: ``"bin=True"``, ``"bin={...}"`` or
        ``"bin=Bin(...)"``.
    written : str
        The binned channel as the user wrote it, e.g. ``x=X('v', bin={...})``,
        the subject of every refusal about this Bin.
    groupby : tuple of str
        The Bin's own groupby columns.
    statistics : tuple of str
        The statistic parameters the user wrote, out of
        :data:`_BIN_STATISTIC_PARAMETERS`.
    outputs : frozenset of str
        The columns this Bin computes, from Rust
        (``ferrum._core.bin_output_columns``).
    wire : dict or None
        A pre-built Bin's wire form, the only way its parameters can be
        extended; ``None`` for the other spellings.
    """

    spelling: str
    written: str
    groupby: tuple[str, ...]
    statistics: tuple[str, ...]
    outputs: frozenset[str]
    wire: dict | None = None


def _bin_declaration(channel: str, ch: ChannelBase) -> _BinDeclaration:
    """The declaration of the ``Bin`` *ch*'s ``bin=`` names, read from its wire on both spellings.

    ``bin={...}`` is read through ``Bin(field, **kwargs)``, which also keeps
    the constructor's own refusal of a malformed parameter; ``bin=True``
    through ``Bin(field)``.

    Raises
    ------
    ValueError
        If the Bin carries a ``name``: an encoding bin is computed for its
        own layer and published to no other, so the name would be ignored.
    """
    from ferrum import Bin
    from ferrum._core import bin_output_columns
    from ferrum._registry import transform_wires

    pending = _own_bin(ch)
    if pending.bin_obj is not None:
        spelling, built = "bin=Bin(...)", pending.bin_obj
    elif pending.bin_kwargs:
        spelling, built = "bin={...}", Bin(pending.field, **dict(pending.bin_kwargs))
    else:
        spelling, built = "bin=True", Bin(pending.field)
    written = f"{channel}={type(ch).__name__}({ch.field!r}, {spelling})"
    (wire,) = transform_wires([built])
    if wire.get("name") is not None:
        raise ValueError(
            f"{written} names its Bin {wire['name']!r}, but an encoding bin is computed for "
            "its own layer and published to no other layer, so name= would be ignored. "
            f"Remove name= from {spelling}; to publish the binned rows under a name another "
            f"layer reads, compute them with Chart.transform(fm.Bin({pending.field!r}, "
            f"name={wire['name']!r}))."
        )
    present = {
        "groupby": bool(wire.get("groupby")),
        "cumulative": bool(wire.get("cumulative")),
        "agg_field/agg": "agg_field" in wire or "agg" in wire,
        "aggregates": "aggregates" in wire,
    }
    return _BinDeclaration(
        spelling,
        written,
        tuple(wire.get("groupby") or ()),
        tuple(name for name in _BIN_STATISTIC_PARAMETERS if present[name]),
        frozenset(bin_output_columns(built)),
        wire if pending.bin_obj is not None else None,
    )


def _binned_channels(encoding: dict) -> list[tuple[str, ChannelBase]]:
    """The ``(channel, binding)`` pairs of *encoding* that carry ``bin=``."""
    return [
        (channel, ch)
        for channel, ch in encoding.items()
        if isinstance(ch, ChannelBase) and ch._kwargs.get("bin")
    ]


def _own_bin(ch: ChannelBase) -> _PendingBin:
    """The ``_PendingBin`` sentinel *ch*'s own ``bin=`` emits."""
    return next(t for t in ch.to_implicit_transforms() if isinstance(t, _PendingBin))


def _statistic_groupby(encoding: dict, facet) -> list[str]:
    """The fields one layer's statistic groups by (spec §6.4), in channel order.

    Every field the encoding draws that its statistic neither aggregates nor
    bins, plus the facet fields, so a faceted layer computes its statistic
    within each panel. Plain string bindings count as fields; a field over an
    unresolved ``Repeat`` placeholder does not. A :class:`_GroupingOnly`
    contributes its members' fields, in its place.

    Under a ``bin=`` the binned field and every column the ``Bin`` declares
    (:func:`_bin_declaration`) are no group keys: ``y="count"`` draws the
    Bin's count and ``y="mean_w"`` an ``aggregates`` entry, neither groups,
    and a ``tooltip="v"`` beside ``x=bin(v)`` reads the binned rows.

    The one groupby rule of every encoding statistic: the flat chart's and
    each layer's aggregate and bin read it alike.
    """
    from ferrum.repeat import _RepeatPlaceholder

    encoding = _expand_grouping(encoding)
    excluded: set = set()
    for channel, ch in _binned_channels(encoding):
        excluded |= _bin_declaration(channel, ch).outputs
        if isinstance(ch.field, str):
            excluded.add(ch.field)
    fields: list[str] = []
    for ch in encoding.values():
        if isinstance(ch, ChannelBase):
            if ch._kwargs.get("aggregate") or ch._kwargs.get("bin"):
                continue
            column = ch.field
        elif isinstance(ch, str):
            column = ch
        else:
            continue
        if column is None or isinstance(column, _RepeatPlaceholder):
            continue
        if column not in excluded and column not in fields:
            fields.append(column)
    if facet is not None:
        for name in (facet.col, facet.row, facet.field):
            if name and name not in excluded and name not in fields:
                fields.append(name)
    return fields


def _polar_channel_targets(coord) -> tuple[tuple[str, str], ...]:
    """The ``(polar, cartesian)`` channel pairs *coord* draws, or ``()`` when not polar.

    Rust's encoding layer only knows x/y; the spec-side coord conversion in
    ``scene_build.rs`` handles the polar->Cartesian pixel transformation. Under
    a ``CoordPolar``, ``theta`` (the angular variable) maps to whichever
    Cartesian axis the coord declares and ``radius`` to the other, with the
    second-extent channels mirroring the primary assignment. Pairs rather
    than a channel-keyed dict: the values are channel names, not columns.
    """
    from ferrum.coord import CoordPolar

    if not isinstance(coord, CoordPolar):
        return ()
    theta_ch = coord.theta  # "x" or "y"
    radius_ch = "y" if theta_ch == "x" else "x"
    return (
        ("theta", theta_ch),
        ("theta2", f"{theta_ch}2"),
        ("radius", radius_ch),
        ("radius2", f"{radius_ch}2"),
    )


def _theta_axis(coord) -> str | None:
    """The Cartesian channel *coord*'s theta draws as, or ``None`` when it is not polar."""
    return dict(_polar_channel_targets(coord)).get("theta")


def _aggregate_ops(pending: list[_PendingAggregate]) -> list:
    """The ``AggregateOp`` list of one ``Aggregate`` computing every sentinel in *pending*.

    One op per distinct output column, in first-seen order: two channels
    asking for the same (field, op) read the one column it computes.
    """
    from ferrum import AggregateOp

    ops: dict = {}
    for p in pending:
        if p.output_col not in ops:
            ops[p.output_col] = AggregateOp(p.field, p.agg, p.output_col)
    return list(ops.values())


def _aggregated_encoding(encoding: dict) -> dict:
    """Return *encoding* with each channel carrying ``aggregate=`` pointed at that aggregate's output.

    Each channel is re-pointed by its own (field, op), at the output column its
    ``_PendingAggregate`` names, so two channels aggregating one field with
    different ops (``x="min(v)", x2="max(v)"``) draw different columns, and a
    channel binding the same field raw (``x="v", x2="max(v)"``) keeps drawing
    it as a group key. The rebuilt channel drops its now-resolved ``aggregate``
    kwarg. A channel over an unresolved ``Repeat`` placeholder is left as is.
    """
    from ferrum.repeat import _RepeatPlaceholder

    out: dict = {}
    for channel, ch in encoding.items():
        pending = None
        if isinstance(ch, ChannelBase) and not isinstance(ch.field, _RepeatPlaceholder):
            pending = next(
                (t for t in ch.to_implicit_transforms() if isinstance(t, _PendingAggregate)),
                None,
            )
        if pending is None:
            out[channel] = ch
            continue
        kwargs = {k: v for k, v in ch._kwargs.items() if k != "aggregate"}
        out[channel] = ch.__class__(pending.output_col, **kwargs)
    return out


def _encoding_statistics(encoding: dict) -> tuple[list, list]:
    """The ``(_PendingAggregate, _PendingBin)`` sentinels the channels of *encoding* carry.

    A grouping-only key's members count as the raw bindings they stand for,
    so a binned inherited color bins the layer as it bins its parent. A
    channel bound under two names (``fill`` aliased onto ``color``) counts
    once.
    """
    aggregates: list = []
    bins: list = []
    seen: set[int] = set()
    for ch in _expand_grouping(encoding).values():
        if isinstance(ch, ChannelBase) and id(ch) not in seen:
            seen.add(id(ch))
            for t in ch.to_implicit_transforms():
                if isinstance(t, _PendingAggregate):
                    aggregates.append(t)
                elif isinstance(t, _PendingBin):
                    bins.append(t)
    return aggregates, bins


@dataclass(frozen=True)
class _StatisticPlan:
    """One layer's statistic, read off its effective encoding (spec §6.4).

    The flat chart and every layer build theirs with :func:`_statistic_plan`
    and run its :meth:`step`, so the two paths compute one statistic per
    layer by one rule.

    Attributes
    ----------
    encoding : dict
        What the layer draws once the statistic has run: each aggregated
        channel at its output column, a binned channel at the ``Bin``'s start
        column and its span at the end column. The input encoding itself when
        the layer computes nothing.
    groupby : tuple of str
        The columns the statistic partitions its rows by
        (:func:`_statistic_groupby`); empty for a per-row ``Bin``. A ``Bin``
        groups by its own declared groupby first, then these.
    aggregates : tuple of _PendingAggregate
        The ops of the layer's one ``Aggregate``; empty when the layer bins
        (its ops fold into the ``Bin``) or computes nothing.
    bin : _PendingBin or None
        The binned channel's sentinel, ``None`` when the layer bins nothing.
    declaration : _BinDeclaration or None
        What that Bin declares itself (:func:`_bin_declaration`).
    bin_kwargs : dict
        The ``Bin`` keyword arguments the plan adds: ``per_row=True`` for a
        positional bin with no aggregate, ``agg_field``/``agg`` for one folded
        aggregate, ``aggregates`` for two or more.
    """

    encoding: dict
    groupby: tuple[str, ...] = ()
    aggregates: tuple[_PendingAggregate, ...] = ()
    bin: _PendingBin | None = None
    declaration: _BinDeclaration | None = None
    bin_kwargs: dict = field(default_factory=dict)

    @property
    def computes(self) -> bool:
        """Whether the layer computes a statistic: it bins or aggregates."""
        return self.bin is not None or bool(self.aggregates)

    def step(self, *, name: str | None = None) -> "Aggregate | Bin | dict":
        """The one transform this plan computes.

        An ``Aggregate`` holding every op, or the ``Bin``
        (:func:`_build_bin_transform`; *name* is set on the layered path's
        named fan-out ``Bin``). The Bin is the ``Bin`` object, except a
        pre-built Bin the plan extends, which is its wire dict with the
        plan's parameters added: the PyO3 ``Bin`` can be extended no other
        way.

        Raises
        ------
        ValueError
            If the plan computes nothing (:attr:`computes`); its caller
            branches on that first.
        """
        if self.bin is not None and self.declaration is not None:
            return _build_bin_transform(
                self.bin, self.declaration, self.bin_kwargs, self.groupby, name=name
            )
        if self.aggregates:
            from ferrum import Aggregate

            return Aggregate(_aggregate_ops(list(self.aggregates)), groupby=list(self.groupby))
        raise ValueError("_StatisticPlan.step: the plan computes no statistic, so it has no step")


def _statistic_plan(encoding: dict, facet) -> _StatisticPlan:
    """The statistic plan of a layer drawing *encoding* under *facet* (spec §6.4).

    A ``bin=`` channel makes the plan a ``Bin`` (:func:`_lower_bin_encoding`);
    otherwise any ``aggregate=`` channel makes it one ``Aggregate``, grouped by
    :func:`_statistic_groupby`, with each aggregated channel drawing its
    (field, op)'s output column. An encoding carrying neither computes nothing.
    """
    aggregates, bins = _encoding_statistics(encoding)
    if bins:
        return _lower_bin_encoding(encoding, bins, aggregates, facet)
    if aggregates:
        return _StatisticPlan(
            _aggregated_encoding(encoding),
            tuple(_statistic_groupby(encoding, facet)),
            tuple(aggregates),
        )
    return _StatisticPlan(encoding)


def _build_bin_transform(
    pending: _PendingBin,
    declaration: _BinDeclaration,
    plan_kwargs: dict,
    groupby: tuple[str, ...] = (),
    *,
    name: str | None = None,
) -> "Bin | dict":
    """Build the concrete ``Bin`` that one ``_PendingBin`` sentinel stands for.

    Shared by both encode paths: *name* is set on the layered path (a named
    fan-out transform the layer reads via ``data_source``) and left ``None`` on
    the single-chart path (an unnamed transform that advances the chain).

    The Bin groups by its own declared groupby, then *groupby* (the plan's);
    when the plan adds a group, every group is counted over one set of edges
    (``shared_extent``), as a grouped ``mark_histogram`` does. A pre-built
    ``bin=Bin(...)`` comes back as the object itself when the plan adds
    nothing, else as its wire dict with ``per_row``, ``groupby`` and
    ``shared_extent`` added; the layered serialiser
    (``_transforms_to_json_list_named``) injects a name from
    ``_NamedTransform.name`` into either form.
    """
    from ferrum import Bin

    additions = dict(plan_kwargs)
    if groupby:
        additions["groupby"] = list(dict.fromkeys((*declaration.groupby, *groupby)))
        additions["shared_extent"] = True
    if declaration.wire is not None:
        if not additions:
            return pending.bin_obj
        return {**declaration.wire, **additions}
    bin_kwargs = dict(pending.bin_kwargs)
    if name is not None:
        bin_kwargs["name"] = name
    return Bin(pending.field, **{**bin_kwargs, **additions})


def _rebind_channel(ch: ChannelBase, column: str, title: str | None = None) -> ChannelBase:
    """Rebuild *ch* pointing at *column*, keeping the user's other kwargs.

    Drops the now-resolved ``bin=``/``aggregate=`` kwargs (both are consumed by
    the transform, never serialized) and titles the channel *title*, by
    default its own field, when that differs from *column*: a binned axis
    reads ``"tip"`` and a folded aggregate ``"mean_tip"``, never
    ``"bin_start"`` or ``"agg_value"``. The one re-pointing rule
    :func:`ferrum._desugar._apply_remap` also uses. An explicit ``title=``
    always wins; a non-``str`` field (none, or an unresolved Repeat
    placeholder) recovers nothing, and neither does a channel that does not
    honor ``title=`` (a tooltip, a text label), which keeps no title.
    """
    if title is None:
        title = ch.field
    kwargs = {k: v for k, v in ch._kwargs.items() if k not in ("bin", "aggregate")}
    if (
        isinstance(title, str)
        and title != column
        and "title" not in kwargs
        and "title" in type(ch)._honored_kwargs
    ):
        kwargs["title"] = title
    return ch.__class__(column, **kwargs)


def _count_axes(encoding: dict, agg_axes: list[str]) -> list[str]:
    """The channels of *agg_axes* asking for a fieldless ``count()``: they read the Bin's count."""
    return [
        axis
        for axis in agg_axes
        if encoding[axis].field in (None, "") and encoding[axis]._kwargs["aggregate"] == "count"
    ]


def _undrawn_bin_statistic(bin_axis: str, encoding: dict, declaration: _BinDeclaration) -> str:
    """The refusal of a ``Bin`` statistic no channel draws beside a positional bin.

    With no channel drawing an aggregate or one of the Bin's columns, each
    row is placed at its bin, so the statistic the user's Bin declares would
    be dropped.
    """
    from ferrum.encoding._aliases import _channel_field

    params = " and ".join(declaration.statistics)
    raw = [
        axis
        for axis, ch in encoding.items()
        if axis != bin_axis and isinstance(_channel_field(ch), str)
    ]
    drawn = (
        f"{' and '.join(repr(axis) for axis in raw)} "
        f"{'draws' if len(raw) == 1 else 'draw'} raw rows"
        if raw
        else "no channel draws a value"
    )
    outputs = sorted(declaration.outputs - {"bin_start", "bin_end"})
    return (
        f"{declaration.written} declares {params}, but no channel draws the Bin's "
        f"statistic: {drawn}, so each row would be placed at its bin and {params} "
        "dropped. Bind a channel to one of the Bin's output columns "
        f"({', '.join(repr(name) for name in outputs)}), or remove {params} from "
        f"{declaration.spelling} to place each row at its bin."
    )


def _own_aggregate(ch: ChannelBase) -> _PendingAggregate:
    """The ``_PendingAggregate`` sentinel *ch*'s own ``aggregate=`` emits."""
    return next(t for t in ch.to_implicit_transforms() if isinstance(t, _PendingAggregate))


def _lower_bin_encoding(
    encoding: dict,
    pending_bins: list[_PendingBin],
    pending_aggregates: list[_PendingAggregate],
    facet,
) -> _StatisticPlan:
    """The statistic plan of an encoding with a ``bin=`` channel (spec §4.2, §6.6).

    A ``Bin`` replaces its input batch's schema with its own columns (or, per
    row, appends its start and end columns), so every channel it feeds is
    re-pointed at those columns:

    * the binned channel → the Bin's start column, titled by its source field,
      plus its span channel (``x2``/``y2``) → the end column, unless the
      span is bound already;
    * a fieldless ``count()`` → the Bin's own count column;
    * each distinct (field, op) aggregate on any channel → computed per bin:
      one folds in through ``agg_field``/``agg`` (its channels drawing the
      Bin's aggregate column), two or more through ``aggregates`` (each
      drawing its own ``<op>_<field>`` column), titled by that output name.

    The Bin aggregates when any channel aggregates or names one of its own
    output columns (``y="count"``), grouped by :func:`_statistic_groupby`.
    A positional bin with neither bins each row in place (``per_row``): every
    row is kept and placed at its bin. A bin on a non-positional channel with
    neither still counts its rows.

    Column names come from the transform registry, never from literals, so a
    Rust-side rename raises here naming the stale column (spec §4.6 / M6a).

    Parameters
    ----------
    encoding : dict
        The channel dict to rewrite (never mutated). A grouping-only key's
        members are lowered as the raw bindings they stand for and folded
        back into the key after.
    pending_bins : list of _PendingBin
        The bin sentinels this encoding produced.
    pending_aggregates : list of _PendingAggregate
        The aggregate sentinels this encoding produced.
    facet :
        The chart's facet, whose fields join an aggregating Bin's groupby.

    Raises
    ------
    ValueError
        If the encoding declares more than one bin, if one channel carries both
        ``bin=`` and ``aggregate=``, if a pre-built ``bin=Bin(...)`` would have
        to absorb an aggregate, or if a fieldless non-``count`` aggregate is
        asked for per bin.
    """
    from ferrum import AggregateOp
    from ferrum._registry import registry_output_columns
    from ferrum.encoding import X2, Y2
    from ferrum.encoding._aliases import _channel_field

    if len(pending_bins) > 1:
        raise ValueError(
            "an encoding declares more than one bin=; a Bin transform replaces the "
            "batch schema with its own columns, so a second Bin would re-bin the "
            "first one's output. Bin a single channel, or use mark_hex() / "
            "Chart.transform(fm.Bin2D(x=..., y=...)) for two-dimensional binning."
        )
    grouping = _grouping_of(encoding)
    encoding = _expand_grouping(encoding)
    bin_axis = next(
        axis
        for axis, ch in encoding.items()
        if isinstance(ch, ChannelBase) and ch._kwargs.get("bin")
    )

    # Only channels that HONOR aggregate= are scanned. ChannelBase's contract
    # (encoding/base.py, "Honored-kwarg contract") is that an unhonored kwarg
    # never steers the pipeline, so it must not fold into the Bin.
    agg_axes = [
        axis
        for axis, ch in encoding.items()
        if isinstance(ch, ChannelBase)
        and "aggregate" in type(ch)._honored_kwargs
        and ch._kwargs.get("aggregate")
    ]
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

    declaration = _bin_declaration(bin_axis, encoding[bin_axis])

    def plan(groupby: tuple[str, ...], kwargs: dict) -> _StatisticPlan:
        return _StatisticPlan(
            _regroup(new_encoding, grouping),
            groupby,
            bin=pending_bins[0],
            declaration=declaration,
            bin_kwargs=kwargs,
        )

    draws_bin_output = any(
        _channel_field(ch) in declaration.outputs
        for axis, ch in encoding.items()
        if axis != bin_axis
    )
    if not agg_axes and not draws_bin_output and bin_axis in ("x", "y"):
        if declaration.statistics:
            raise ValueError(_undrawn_bin_statistic(bin_axis, encoding, declaration))
        return plan((), {"per_row": True})

    groupby = tuple(_statistic_groupby(encoding, facet))
    # A fieldless count() reads the Bin's own count column. Every other
    # aggregate is one (field, op) slot, in channel order; channels asking
    # for the same (field, op) read the one column it computes.
    slots: dict[tuple[Any, str], _PendingAggregate] = {}
    counts = _count_axes(encoding, agg_axes)
    for axis in agg_axes:
        ch = encoding[axis]
        if axis in counts:
            new_encoding[axis] = _rebind_channel(ch, cols["count"])
        else:
            slots.setdefault((ch.field, ch._kwargs["aggregate"]), _own_aggregate(ch))
    if not slots:
        return plan(groupby, {})
    own_aggregate = [
        name for name in declaration.statistics if name in ("agg_field/agg", "aggregates")
    ]
    if own_aggregate and declaration.wire is None:
        slot_axis = next(axis for axis in agg_axes if axis not in _count_axes(encoding, agg_axes))
        raise ValueError(
            f"{declaration.written} declares its own {own_aggregate[0]} and channel "
            f"{slot_axis!r} carries aggregate={encoding[slot_axis]._kwargs['aggregate']!r}: "
            "declare the per-bin statistic in one place. Drop "
            f"{own_aggregate[0]} from {declaration.spelling} to compute the channel's "
            f"aggregate per bin, or drop aggregate= and bind {slot_axis!r} to the Bin's "
            "output column."
        )

    for source_field, method in slots:
        if not isinstance(source_field, str) or not source_field:
            raise ValueError(
                f"aggregate={method!r} has no source column, so it cannot be computed "
                "per bin; only count() is field-less. Name the column to aggregate, "
                f"e.g. Y('value', aggregate={method!r})."
            )
    if pending_bins[0].bin_obj is not None:
        (source_field, method), *_ = slots
        slot_axis = next(axis for axis in agg_axes if encoding[axis].field == source_field)
        raise ValueError(
            f"bin=Bin(...) on {bin_axis!r} cannot absorb the aggregate= on "
            f"{slot_axis!r}: a pre-built Bin computes the statistics it was built "
            "with, and an encoding aggregate is not folded into it. Declare it on the Bin "
            f"itself -- Bin({pending_bins[0].field!r}, agg_field={source_field!r}, "
            f"agg={method!r}) -- and bind the channel to {cols['agg_value']!r}."
        )

    if len(slots) == 1:
        ((source_field, method),) = slots
        columns = {(source_field, method): cols["agg_value"]}
        kwargs: dict = {"agg_field": source_field, "agg": method}
    else:
        columns = {key: pending.output_col for key, pending in slots.items()}
        kwargs = {
            "aggregates": [
                AggregateOp(pending.field, pending.agg, pending.output_col)
                for pending in slots.values()
            ]
        }
    for axis in agg_axes:
        ch = encoding[axis]
        key = (ch.field, ch._kwargs["aggregate"])
        if key in slots:
            new_encoding[axis] = _rebind_channel(ch, columns[key], slots[key].output_col)
    # The folded aggregates are this plan's columns too: a channel drawing one
    # (`text="agg_value"` beside `y=mean(w)`) reads it, it does not group by it.
    folded = set(columns.values())
    return plan(tuple(f for f in groupby if f not in folded), kwargs)
