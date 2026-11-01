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

from ferrum.encoding.base import (
    ChannelBase,
    _PendingAggregate,
    _PendingBin,
    _pending_bin,
    _unset_statistic,
)

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

    def wire(self) -> dict | None:
        """The layer's wire ``group``: what the parent draws as color, under the alias rules.

        The field (``_ENCODING_ALIASES`` picks the binding), plus the binned
        color scale when that binding bins, so dodge and stack group by its
        bins. ``None`` when the parent draws no concrete color field.
        """
        from ferrum.encoding._aliases import _channel_field, resolve_channel_alias

        binding = resolve_channel_alias(dict(self.bindings), "color")
        field = _channel_field(binding)
        if not isinstance(field, str):
            return None
        if isinstance(binding, ChannelBase) and binding._binned_color():
            return {"field": field, "scale": binding._color_bin_scale()}
        return {"field": field}


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
    bin_wire : dict
        The Bin's wire form on every spelling, read for its parameters.
    wire : dict or None
        A pre-built Bin's wire form, the only way its parameters can be
        extended; ``None`` for the other spellings.
    """

    spelling: str
    written: str
    groupby: tuple[str, ...]
    statistics: tuple[str, ...]
    outputs: frozenset[str]
    bin_wire: dict
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

    pending = _pending_bin(ch.field, ch._kwargs["bin"])
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
        wire,
        wire if pending.bin_obj is not None else None,
    )


def _binned_channels(encoding: dict) -> list[tuple[str, ChannelBase]]:
    """The ``(channel, binding)`` pairs of *encoding* whose ``bin=`` bins the rows.

    A binned color is not one: it draws its field through a binned scale.
    """
    return [
        (channel, ch)
        for channel, ch in encoding.items()
        if isinstance(ch, ChannelBase) and ch._lowers_row_bin()
    ]


def _statistic_groupby(encoding: dict, facet) -> list[str]:
    """The fields one layer's statistic groups by, in channel order.

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
            if ch._kwargs.get("aggregate") or ch._lowers_row_bin():
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


#: The channels a composite mark's statistic groups by: color under
#: every spelling that draws as it, detail, shape and stroke_dash.
COMPOSITE_GROUPING_CHANNELS = frozenset(
    ("color", "fill", "stroke", "detail", "shape", "stroke_dash")
)


def _composite_groupby(encoding: dict, facet) -> list[str]:
    """The fields a composite mark's statistic groups by (its composite grouping).

    :func:`_statistic_groupby` restricted to the grouping channels
    (:data:`COMPOSITE_GROUPING_CHANNELS`, a :class:`_GroupingOnly`'s members
    included) plus the facet fields, so a histogram, density, box, violin,
    boxen, qq, contour or smooth computes one statistic per group, exactly as
    the same transform built by hand with ``groupby=`` does. An aggregated
    grouping channel draws a value, not a group, and joins none.

    Empty when no grouping channel binds a field: a faceted panel's rows are
    already its own (the source is partitioned before any statistic runs),
    so the facet fields alone add no group, and the composite's transforms
    stay exactly as they are without a grouping channel.
    """
    grouping = {
        channel: binding
        for channel, binding in _expand_grouping(encoding).items()
        if channel in COMPOSITE_GROUPING_CHANNELS
    }
    if not _statistic_groupby(grouping, None):
        return []
    return _statistic_groupby(grouping, facet)


def _grouping_bindings(encoding: dict) -> dict:
    """The bindings of *encoding* on a grouping channel (:data:`COMPOSITE_GROUPING_CHANNELS`)."""
    return {
        channel: binding
        for channel, binding in encoding.items()
        if channel in COMPOSITE_GROUPING_CHANNELS and binding is not None
    }


def _over(nearer: dict, farther: dict) -> dict:
    """*nearer*'s bindings, then each of *farther*'s on a channel *nearer* binds under no spelling.

    Channel identity is the resolved channel (``fill`` and ``stroke`` draw
    as ``color``), so a nearer ``fill`` shadows a farther ``color``, as an
    own binding shadows its parent's.
    """
    from ferrum.encoding._aliases import resolved_channel

    bound = {resolved_channel(channel) for channel, value in nearer.items() if value is not None}
    return {
        **nearer,
        **{
            channel: value
            for channel, value in farther.items()
            if resolved_channel(channel) not in bound
        },
    }


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
    """One layer's statistic, read off its effective encoding.

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
        named fan-out ``Bin``). ``bin=True``/``bin={...}`` with the plan's
        additions builds a new ``Bin`` object (the PyO3 constructor takes
        every parameter directly). Only two shapes come back as a wire
        dict instead: a pre-built ``bin=Bin(...)`` the plan extends (the
        PyO3 ``Bin`` can be extended no other way), or a Bin that declares
        its own groupby (marked ``written_groupby`` on the wire, whether or
        not the plan adds anything) -- when neither applies and the plan
        adds nothing, a pre-built Bin comes back as the object itself.

        Raises
        ------
        RuntimeError
            If the plan computes nothing (:attr:`computes`); its caller
            branches on that first, so reaching this is an internal
            invariant violation, not a caller mistake.
        """
        if self.bin is not None and self.declaration is not None:
            return _build_bin_transform(
                self.bin, self.declaration, self.bin_kwargs, self.groupby, name=name
            )
        if self.aggregates:
            from ferrum import Aggregate

            return Aggregate(_aggregate_ops(list(self.aggregates)), groupby=list(self.groupby))
        raise RuntimeError("a statistic plan that computes no statistic has no step")


def _statistic_plan(encoding: dict, facet) -> _StatisticPlan:
    """The statistic plan of a layer drawing *encoding* under *facet*.

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

    A Bin that declares its own ``groupby=`` (``declaration.groupby``) always
    comes back as its wire dict, with ``written_groupby`` added naming those
    columns: a line, area or ribbon reading this Bin's output splits by a
    groupby the caller wrote directly, never by one the chart's own statistic
    planner added (which lands in *groupby*/*plan_kwargs* instead, and must
    not split). A Bin declaring no groupby is unaffected by this.
    """
    from ferrum import Bin

    additions = dict(plan_kwargs)
    if groupby:
        additions["groupby"] = list(dict.fromkeys((*declaration.groupby, *groupby)))
        additions["shared_extent"] = True
    if declaration.groupby:
        base = declaration.wire if declaration.wire is not None else declaration.bin_wire
        wire = {**base, **additions, "written_groupby": list(declaration.groupby)}
        if name is not None:
            wire["name"] = name
        return wire
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

    Precondition: *ch*'s ``bin=``/``aggregate=`` was either consumed by the
    statistic the caller's door builds, or refused by that door
    (:func:`_refuse_bin_or_aggregate`) before this runs. Dropping either
    kwarg here is correct only because one of those two already happened.
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


def _rebind_positional(
    column: str,
    orig: ChannelBase | str | None,
    cls: type[ChannelBase],
    *,
    title: str | None = None,
) -> ChannelBase:
    """The bare-name-or-channel face of :func:`_rebind_channel`.

    A figure that re-points a caller's own positional binding at a
    statistic's output column calls this rather than hand-building a fresh
    channel, so the caller's options survive regardless of whether they
    bound a channel object or a plain field name. *orig* is the caller's
    original binding at that slot: a :class:`ChannelBase` is rebuilt through
    :func:`_rebind_channel` (every kwarg kept, *title* forwarded as that
    call's own title default); a bare field name or ``None`` gets a fresh
    quantitative channel titled *title*, or, when *title* is unset, by the
    field itself when it differs from *column*. Shared by
    :func:`ferrum._desugar._apply_remap` (every ``bin=``/``aggregate=``
    lowering, with ``title=None``, so a bare field still titles by itself)
    and a figure that supplies its own default title (e.g. ``residplot``'s
    ``"<y> residual"``).
    """
    if isinstance(orig, ChannelBase):
        return _rebind_channel(orig, column, title=title)._with_options(type="Q")
    if title is None:
        field = orig if isinstance(orig, str) else None
        title = field if (field and column != field) else None
    return cls(column, type="Q", title=title) if title else cls(column, type="Q")


def _quoted_statistic_value(value: Any) -> str:
    """The repr a refusal quotes *value* (a ``bin=``/``aggregate=`` kwarg) with.

    A plain literal (``bool``, ``str``, ``int``, ``float``, ``dict``,
    ``list``, ``tuple``) quotes as its own ``repr``, matching what the
    caller typed (``bin={'bin_count': 3}``). A pre-built object (a
    ``fm.Bin(...)``) quotes by its class name alone, the same spelling
    :func:`_bin_declaration` already gives a pre-built ``Bin``
    (``"bin=Bin(...)"``), never its raw field-by-field repr.
    """
    if isinstance(value, (bool, str, int, float, dict, list, tuple)):
        return repr(value)
    return f"{type(value).__name__}(...)"


def _written_channel(channel: str, ch: ChannelBase, kwarg: str) -> str:
    """*channel*'s binding as the user wrote it, naming the *kwarg* a refusal is about.

    A field-bearing channel quotes its field (``x=X('a', bin=True)``); a
    fieldless one (a histogram/density/ecdf value axis styled with no
    field of its own) quotes none, since the caller wrote none
    (``y=Y(aggregate='count')``, never ``Y(None, aggregate='count')``).
    """
    from ferrum.encoding._aliases import _channel_field

    quoted = _quoted_statistic_value(ch._kwargs[kwarg])
    field = _channel_field(ch)
    if field:
        return f"{channel}={type(ch).__name__}({field!r}, {kwarg}={quoted})"
    return f"{channel}={type(ch).__name__}({kwarg}={quoted})"


def _refuse_bin_or_aggregate(
    entry: str, channel: str, ch: ChannelBase, kwarg: str, *, reason: str, remedy: str
) -> None:
    """Refuse *channel*'s *kwarg* (``"bin"``/``"aggregate"``) a statistic's door cannot honor.

    Raised as :class:`~ferrum._entry.EntryRefusal`, spoken as *entry* and
    re-voiceable by a figure door that calls further in
    (:func:`ferrum._entry.voiced_as`). *reason* names the statistic alone,
    never a call: a re-voiced refusal keeps the same reason text under
    every voice, so a reason naming a call would misquote whichever call
    the caller actually made. *remedy* is a call that renders, the one a
    door choosing it has itself confirmed works -- the same shape
    :func:`ferrum.plots.distribution._refuse_interval_args_on_non_aggregating_kind`
    and the channel-level ``_NO_BIN_MESSAGE`` (:mod:`ferrum.encoding.base`) use.
    """
    from ferrum._entry import EntryRefusal

    written = _written_channel(channel, ch, kwarg)
    raise EntryRefusal(
        lambda voice: (
            f"{voice}({written}) is not supported: {reason}, so {kwarg}= would be ignored. {remedy}"
        ),
        entry,
    )


def _generic_bin_remedy(ch: ChannelBase, *, call_kwargs: str | None = None) -> str:
    """The remedy for a ``bin=`` a statistic does not fold in.

    A field-bearing channel is binned ahead of time, PER ROW
    (``per_row=True``): a default ``Bin`` aggregates its rows and drops
    every column but its own, which breaks a statistic reading the
    field's sibling columns, so the remedy keeps every row and names the
    column the statistic binds instead. A fieldless channel names no
    field to bin, so the only remedy is to drop it. *call_kwargs*, when
    given, names an extra keyword the remedy's own call needs to render
    (``residplot``'s ``dropna=False``, since its default drops the rows
    the remedy's ``.transform()`` step would otherwise read).
    """
    from ferrum.encoding._aliases import _channel_field

    field = _channel_field(ch)
    if not field:
        return "Drop bin=."
    from ferrum._registry import registry_output_columns

    bin_start = registry_output_columns("bin", ["bin_start"])["bin_start"]
    with_clause = f" with {call_kwargs}" if call_kwargs else ""
    return (
        f"Bin the field per row with .transform(fm.Bin({field!r}, per_row=True)) "
        f"and bind its {bin_start!r} column{with_clause}, or drop bin=."
    )


def _generic_aggregate_remedy(ch: ChannelBase, *, call_kwargs: str | None = None) -> str:
    """The remedy for an ``aggregate=`` a statistic does not fold in.

    See :func:`_generic_bin_remedy` for *call_kwargs*.
    """
    from ferrum.encoding._aliases import _channel_field

    if not _channel_field(ch):
        return "Drop aggregate=."
    with_clause = f" with {call_kwargs}" if call_kwargs else ""
    return (
        "Compute the aggregate with .transform(fm.transform_aggregate(...)) and bind its "
        f"output column{with_clause}, or drop aggregate=."
    )


def _histogram_value_axis_remedy(
    kwarg: str, value_ch: ChannelBase, value_axis: str, bin_ch: Any, bin_axis: str
) -> str:
    """The remedy for a ``bin=``/``aggregate=`` on the histogram's value axis.

    The value axis never binds a caller's column -- it draws the Bin's
    own count or density -- so the generic per-row-``Bin``/
    ``transform_aggregate`` remedy would bind that column there and
    silently draw the counts under its name instead. A field-bearing
    channel's remedy computes what the caller asked for over both
    fields (a per-bin aggregate with a bar mark, a two-field count with
    ``mark_hex``) or drops the binding and styles the axis instead. With no
    field bound on the binned axis there is nothing to bin either, so the
    remedy binds one there as well as dropping the value-axis field. A
    fieldless channel's remedy is only to drop the kwarg.
    """
    from ferrum.encoding._aliases import _channel_field

    field = _channel_field(value_ch)
    if not field:
        return "Drop bin=." if kwarg == "bin" else "Drop aggregate=."
    va, ba = value_axis.upper(), bin_axis.upper()
    drop_clause = f"drop {value_axis}= (style the axis with fm.{va}(title=...) and no field)"
    bin_field = _channel_field(bin_ch)
    if not bin_field:
        return f"Bind the field to bin on {bin_axis}= and {drop_clause}."
    if kwarg == "aggregate":
        op = value_ch._kwargs["aggregate"]
        return (
            f"Draw the aggregate per bin with a bar mark, fm.Chart(data).mark_bar().encode("
            f"{bin_axis}=fm.{ba}({bin_field!r}, bin=True), {value_axis}=fm.{va}({field!r}, "
            f"aggregate={op!r})), or {drop_clause}."
        )
    return (
        f"Count rows over both fields with fm.Chart(data).mark_hex().encode("
        f"{bin_axis}={bin_field!r}, {value_axis}={field!r}), or {drop_clause}."
    )


def _refuse_statistic_input(
    entry: str,
    channel: str,
    ch: ChannelBase,
    *,
    reason: str,
    bin_remedy: str | None = None,
    aggregate_remedy: str | None = None,
) -> None:
    """Refuse *channel*'s ``bin=``, then its ``aggregate=``, when *ch* carries either.

    *reason* applies to both alike: the statistic *entry* builds reads
    *channel*'s raw rows, not a bin or aggregate of them. *bin_remedy*/
    *aggregate_remedy* default to the generic, field-aware remedy
    (:func:`_generic_bin_remedy`/:func:`_generic_aggregate_remedy`); a door
    whose own vocabulary gives a better remedy passes its own text.
    """
    if not _unset_statistic("bin", ch._kwargs.get("bin")):
        _refuse_bin_or_aggregate(
            entry,
            channel,
            ch,
            "bin",
            reason=reason,
            remedy=bin_remedy if bin_remedy is not None else _generic_bin_remedy(ch),
        )
    if not _unset_statistic("aggregate", ch._kwargs.get("aggregate")):
        _refuse_bin_or_aggregate(
            entry,
            channel,
            ch,
            "aggregate",
            reason=reason,
            remedy=aggregate_remedy
            if aggregate_remedy is not None
            else _generic_aggregate_remedy(ch),
        )


def _refuse_bin_aggregate_on_channels(
    entry: str,
    encoding: dict,
    channels: tuple[str, ...],
    *,
    reason: str,
    bin_remedy: str | None = None,
    aggregate_remedy: str | None = None,
) -> None:
    """Refuse a ``bin=``/``aggregate=`` on any of *channels* a statistic cannot honor.

    *encoding* may bind a channel bare (a plain field name or ``None``),
    which carries neither kwarg and is skipped. Checked in *channels*
    order, so the first one the caller wrote decides the message.
    """
    for channel in channels:
        ch = encoding.get(channel)
        if isinstance(ch, ChannelBase):
            _refuse_statistic_input(
                entry,
                channel,
                ch,
                reason=reason,
                bin_remedy=bin_remedy,
                aggregate_remedy=aggregate_remedy,
            )


def _refuse_histogram_bin_aggregate(
    entry: str,
    encoding: dict,
    *,
    horizontal: bool,
    density: bool,
    share_spelling: str | None,
    share_remedy: str | None,
    bin_rule_remedy: str,
    density_spelling: str,
    density_remedy: str,
) -> None:
    """Refuse a bin=/aggregate= the histogram's own ``Bin`` cannot honor.

    The binned axis bins through the mark's own ``bins=``/``bin_count=``/
    ``bin_width=``/``extent=``/``nice=``, so a channel ``bin=`` that states
    a rule of its own is a second, conflicting instruction -- refused,
    with *bin_rule_remedy* naming the door's own binning parameters
    instead. ``bin=True`` is a harmless redundancy (asking for exactly
    what the mark already does) and stays honored; an ``aggregate=`` on
    that axis folds into nothing and is always refused. The value axis
    reads no caller column at all -- it draws the Bin's own count or
    density -- so a field-bearing ``bin=``/``aggregate=`` there is
    refused with :func:`_histogram_value_axis_remedy`. A fieldless
    ``aggregate="count"`` matches what the value axis draws and stays
    honored, except under ``density`` (*density_spelling* names how the
    door spells it, *density_remedy* draws the count instead) and under
    a normalized stack (*share_spelling*/*share_remedy*, ``None`` when
    the drawn position does not normalize); every other aggregate on
    that axis is refused with :func:`_histogram_value_axis_remedy`.
    """
    bin_axis, value_axis = ("y", "x") if horizontal else ("x", "y")

    bin_ch = encoding.get(bin_axis)
    if isinstance(bin_ch, ChannelBase):
        bin_value = bin_ch._kwargs.get("bin")
        if not _unset_statistic("bin", bin_value) and bin_value is not True:
            _refuse_bin_or_aggregate(
                entry,
                bin_axis,
                bin_ch,
                "bin",
                reason=f"the histogram bins {bin_axis} by its own rule",
                remedy=bin_rule_remedy,
            )
        if not _unset_statistic("aggregate", bin_ch._kwargs.get("aggregate")):
            _refuse_bin_or_aggregate(
                entry,
                bin_axis,
                bin_ch,
                "aggregate",
                reason=f"the histogram bins {bin_axis} rather than aggregating it",
                remedy=_generic_aggregate_remedy(bin_ch),
            )

    value_ch = encoding.get(value_axis)
    if isinstance(value_ch, ChannelBase):
        from ferrum.encoding._aliases import _channel_field

        if not _unset_statistic("bin", value_ch._kwargs.get("bin")):
            _refuse_bin_or_aggregate(
                entry,
                value_axis,
                value_ch,
                "bin",
                reason=f"the histogram computes {value_axis} itself",
                remedy=_histogram_value_axis_remedy("bin", value_ch, value_axis, bin_ch, bin_axis),
            )
        agg_value = value_ch._kwargs.get("aggregate")
        if not _unset_statistic("aggregate", agg_value):
            fieldless_count = not _channel_field(value_ch) and agg_value == "count"
            if fieldless_count and density:
                _refuse_bin_or_aggregate(
                    entry,
                    value_axis,
                    value_ch,
                    "aggregate",
                    reason=(
                        f"with {density_spelling} the histogram draws density on "
                        f"{value_axis}, not a count"
                    ),
                    remedy=density_remedy,
                )
            elif fieldless_count and share_spelling is not None:
                _refuse_bin_or_aggregate(
                    entry,
                    value_axis,
                    value_ch,
                    "aggregate",
                    reason=(
                        f"with {share_spelling} the histogram draws each bin's share of "
                        f"its stack on {value_axis}, not a count"
                    ),
                    remedy=share_remedy,
                )
            elif not fieldless_count:
                _refuse_bin_or_aggregate(
                    entry,
                    value_axis,
                    value_ch,
                    "aggregate",
                    reason=f"the histogram computes {value_axis} itself",
                    remedy=_histogram_value_axis_remedy(
                        "aggregate", value_ch, value_axis, bin_ch, bin_axis
                    ),
                )


#: The x-channel ``bin=`` remedy on a route whose fit bins before it
#: dispatches on method (``mark_smooth``, and ``lmplot``/``regplot`` on a
#: ``Smooth``-routed method): names the fit's own ``x_bins=`` knob. One
#: constant so the two doors that use it cannot drift apart.
SMOOTH_X_BIN_REMEDY = (
    "Bin the predictor with x_bins=<number of bins>, the fit's own binning, or drop bin=."
)


def _refuse_bin_aggregate_fit(
    entry: str, encoding: dict, *, x_bin_remedy: str | None = None
) -> None:
    """Refuse an x bin=/aggregate= a fit reads raw, and a y one it folds in nowhere.

    x is the predictor; a caller supplies *x_bin_remedy* naming the fit's
    own pre-aggregation knob on a route that has one (``x_bins=``), and
    leaves it unset (the generic, field-aware remedy) on a route that does
    not. x's aggregate= always takes the generic remedy: no route
    aggregates the predictor itself (``x_estimator=`` aggregates the
    RESPONSE within each x bin, never x), so naming it would send the
    caller to a knob that does something else. y is the response the fit
    reads raw, with no knob of its own, so both its kwargs take the
    generic remedy.
    """
    x_ch = encoding.get("x")
    if isinstance(x_ch, ChannelBase):
        _refuse_statistic_input(
            entry, "x", x_ch, reason="the fit reads the raw predictor", bin_remedy=x_bin_remedy
        )
    y_ch = encoding.get("y")
    if isinstance(y_ch, ChannelBase):
        _refuse_statistic_input(entry, "y", y_ch, reason="the fit reads the raw response")


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
    from ferrum.encoding._aliases import _channel_field, _prose_list

    params = _prose_list(list(declaration.statistics))
    raw = [
        axis
        for axis, ch in encoding.items()
        if axis != bin_axis and isinstance(_channel_field(ch), str)
    ]
    drawn = (
        f"{_prose_list([repr(axis) for axis in raw])} "
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
    """The statistic plan of an encoding with a ``bin=`` channel.

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
    row is kept and placed at its bin. A color-family ``bin=`` never reaches
    here: it is a binned color scale, not a row step.

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
    bin_axis = next(axis for axis, _ in _binned_channels(encoding))

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
