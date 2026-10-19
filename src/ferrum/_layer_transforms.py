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

from dataclasses import dataclass, replace

from ferrum._encode_lowering import (
    GROUPING_KEY,
    _GroupingOnly,
    _encoding_statistics,
    _expand_grouping,
    _statistic_plan,
    _theta_axis,
)
from ferrum.encoding._aliases import layered_chart_detail, resolved_channel
from ferrum.encoding._channel_policy import _FACET_CHANNELS, _POLAR_CHANNELS
from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin
from ferrum._layer import (
    _Layer,
    draws_inherited_color,
    inherited_positional_taken,
    layer_mark,
)
from ferrum.position import stack_honored, warn_stack_ignored


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


class _StatisticsMarker:
    """The boundary in one chain between the caller's steps and the statistics ferrum lowers.

    Every step after it in its chain was lowered from a layer's encoding or
    mark (an encoding aggregate or bin, a composite mark's statistics); every
    step before it is the caller's own. Rust groups a statistic after it by a
    binned color's bins when a layer drawing that color reads its output,
    and runs the caller's steps as written. Serialized as an unnamed
    pass-through identity, which never advances the chain (spec §6.7).

    Every instance is the same marker: two compare equal, so a chain
    deduplicated step by step (``_same_step``) keeps one.
    """

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StatisticsMarker)

    def __hash__(self) -> int:
        return hash(_StatisticsMarker)

    def __repr__(self) -> str:
        return "_StatisticsMarker()"


#: The marker's wire step.
_STATISTICS_MARKER_WIRE = {"type": "identity", "statistics": True}


def _with_statistics_marker(steps: list) -> list:
    """*steps*, lowered statistics, behind the chain's statistics marker; empty stays empty."""
    return [_StatisticsMarker(), *steps] if steps else []


def _holds_statistics_marker(chain: list) -> bool:
    """Whether *chain* (its own steps, not its branches') carries lowered statistics."""
    return any(isinstance(step, _StatisticsMarker) for step in chain)


def _statistics_marked(transforms: list, *, keep: bool) -> list:
    """*transforms* with one statistics marker per chain, or none.

    With *keep*, each chain (the top one and every branch's) keeps its first
    marker and drops the later ones, so the boundary sits before the first
    lowered statistic. Without it every marker is dropped: a chart drawing no
    binned color has no statistic to group by bin, and its wire stays as it
    was. Branches are rebuilt, never mutated.

    No step the caller wrote follows a marker in its chain: a chart's own
    ``.transform()`` is inserted before its lowered statistics, and a ``+``
    composition runs an operand chain carrying a marker as its own branch.
    """
    out: list = []
    seen = False
    for step in transforms:
        if isinstance(step, _StatisticsMarker):
            if keep and not seen:
                out.append(step)
            seen = True
            continue
        if isinstance(step, _Branch):
            step = _Branch(step.name, step.input, _statistics_marked(step.transforms, keep=keep))
        out.append(step)
    return out


def _draws_binned_color(encodings: list) -> bool:
    """Whether any of *encodings* binds a binned color, or a grouping key over one."""
    for encoding in encodings:
        for value in encoding.values():
            if isinstance(value, ChannelBase) and value._binned_color():
                return True
            if isinstance(value, _GroupingOnly) and "scale" in (value.wire() or {}):
                return True
    return False


def _source_identity(name: str) -> _NamedTransform:
    """Build a named identity step, which publishes the batch it reads under *name*."""
    from ferrum.marks._desugar_helpers import identity_transform

    return _NamedTransform(identity_transform(name), name)


@dataclass(frozen=True)
class ChartRows:
    """Where a chart's data sits in its pipeline. Never serialized.

    A chart's data is its rows after the steps its caller added with
    ``.transform(...)`` (and the aggregates or bins on its own encoding),
    never the statistics a composite mark appends when it resolves. A layer
    attached to the chart (``_Layer.chart_data``) reads these rows.

    Attributes
    ----------
    input :
        The named output holding the rows *steps* start from; ``None`` is the
        pipeline's original batch.
    steps :
        The chart's chain: the caller's steps, without composite statistics
        or pending encoding sentinels.
    at_tail :
        Whether these rows are exactly the pipeline output the chart's own
        primitive mark reads, so an attached layer reads the tail directly.
    boundary :
        On a flat chart, the index in ``chart._transforms`` where the mark's
        statistics begin, where a later ``.transform(...)`` is inserted;
        ``None`` on a composition, whose later steps append.
    columns :
        The columns of the chart's data before *steps*, when the pipeline's
        input holds other rows too: on a ``+`` composition, its left
        operand's, since the composition's frame also holds the right
        operand's rows and columns. ``None`` when the data is the input
        itself, or its columns are not known.
    """

    input: str | None
    steps: tuple
    at_tail: bool
    boundary: int | None
    columns: tuple[str, ...] | None = None


def _chain_steps(transforms: list) -> tuple:
    """Return the steps of *transforms* that run in the pipeline, without encoding sentinels."""
    return tuple(t for t in transforms if not isinstance(t, (_PendingAggregate, _PendingBin)))


def _chart_rows_of(chart) -> ChartRows:
    """Return *chart*'s data record, the one reader of ``Chart._chart_rows``.

    A chart with no record holds no mark statistic in its ``_transforms``
    (a pending composite, a primitive chart, or a chart assembled from its
    own layers and steps), so its whole chain is its chain and its tail is
    its data.
    """
    if chart._chart_rows is not None:
        return chart._chart_rows
    transforms = list(chart._transforms or [])
    return ChartRows(None, _chain_steps(transforms), True, len(transforms))


def _resolved_chart_rows(pre, resolved) -> ChartRows:
    """Return the data record of *resolved*, the chart *pre*'s pending mark resolved into.

    The resolution appends the mark's statistics after *pre*'s own steps,
    so *pre*'s record still names the chart's data; it stops being the tail
    when an appended step advances the chain.
    """
    rows = _chart_rows_of(pre)
    appended = list(resolved._transforms or [])[len(pre._transforms or []) :]
    if any(_advances_chain(t) for t in appended):
        return replace(rows, at_tail=False)
    return rows


def _settle_chart_data(layers: list, chart) -> tuple[list, list]:
    """Point each layer holding the chart-data role at *chart*'s data.

    When the record is ``at_tail`` the layers read the pipeline tail as they
    are. Otherwise one named ``_Branch`` runs the chart's chain over the
    record's input, and every role layer reads it with ``reads_chart_data``
    set. A chain with no step runs a named identity, so the branch publishes
    its input unchanged. Returns the layers, role cleared, and the branches
    for the caller to add as named steps.

    A layer reading the chart's data through a named output takes its
    x/y/x2/y2 from the written bindings of the chart that owns it when it
    settles (see ``ferrum.chart._written_bindings``), never from the
    resolved encoding a composite mark rewrote, so its ``positional_source``
    is cleared here and its caller marks it ``"written"``: ``Chart.to_spec``
    for the chart's own layers, and ``Chart.__add__`` for a right operand's,
    which fills them from that operand before the composition's written
    bindings fill what it never wrote.
    """
    if not any(layer.chart_data for layer in layers):
        return layers, []
    rows = _chart_rows_of(chart)
    if rows.at_tail:
        return [replace(layer, chart_data=False) for layer in layers], []
    branch = _Branch("", rows.input, list(rows.steps))
    branch.name = f"_auto_{id(branch):x}"
    if not branch.transforms:
        branch.transforms = [_source_identity(f"{branch.name}_rows")]
    settled = [
        replace(
            layer,
            chart_data=False,
            data_source=branch.name,
            reads_chart_data=True,
            positional_source=None,
        )
        if layer.chart_data
        else layer
        for layer in layers
    ]
    return settled, [branch]


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
    if isinstance(step, (_NamedTransform, _Branch, _StatisticsMarker)):
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
    return _encoding_statistics(layer.encoding)[0]


def _layer_pending_bins(layer: _Layer) -> list[_PendingBin]:
    """Collect the encoding-level bin sentinels carried by *layer*."""
    return _encoding_statistics(layer.encoding)[1]


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
    return _Branch(name, layer.data_source, _with_statistics_marker([transform]))


def _resolve_layer_statistics(layers: list, chart) -> tuple[list, list]:
    """Resolve each layer's encoding statistic into a named chart-level transform.

    The renderer never executes per-layer transforms on their own; a layer
    reads its input batch from a *named* chart-level transform via
    ``data_source``. So for every layer whose effective encoding
    (:func:`_effective_encoding` over *chart*'s encoding) carries an
    ``aggregate=`` or a ``bin=``, its own or one it inherits whole from its
    parent, this builds the layer's statistic plan
    (:func:`~ferrum._encode_lowering._statistic_plan`, grouped by that
    encoding plus *chart*'s facet fields), adds the plan's one ``Aggregate``
    or ``Bin`` as a named step, points the layer's ``data_source`` at it and
    draws each re-pointed channel from the step's output columns.

    Each computing layer gets its OWN named step, so two layers compute
    their statistics independently, each at the grain its own effective
    encoding gives it. A binning layer's aggregates fold into its ``Bin``.

    A layer that already carries a ``data_source`` (a ``+`` operand routed to
    its own chain or to the source) computes over that named output: its
    step is a ``_Branch`` over ``data_source`` (see :func:`_layer_step`), and
    re-pointing the layer's ``data_source`` keeps the
    ``inherit_non_positional`` routing. Layers computing nothing pass
    through unchanged.

    Parameters
    ----------
    layers :
        The chart's resolved ``_Layer`` list.
    chart :
        The chart the layers belong to, whose encoding is their parent.

    Returns
    -------
    (new_layers, named_transforms)
        ``new_layers`` is the layer list with computing layers rewritten;
        ``named_transforms`` the steps to add to the chart-level pipeline,
        every ``Aggregate`` before every ``Bin`` (empty when no layer
        computes). The steps reading the chart's chain sit behind one
        statistics marker; a branch carries its own.
    """
    new_layers: list = []
    aggregate_steps: list = []
    bin_steps: list = []
    for i, layer in enumerate(layers):
        effective = _effective_encoding(layer, chart._encoding)
        plan = _statistic_plan(effective, chart._facet)
        if plan.bin is not None:
            name = f"_layer_bin_{i}"
            # Inside a branch the Bin must stay unnamed, so it advances the
            # branch's own chain and its output is what the branch publishes.
            step = plan.step(name=name if layer.data_source is None else None)
            bin_steps.append(_layer_step(layer, step, name))
        elif plan.aggregates:
            name = f"_layer_agg_{i}"
            aggregate_steps.append(_layer_step(layer, plan.step(), name))
        else:
            new_layers.append(layer)
            continue
        carried = _carried_bindings(layer, chart._encoding, plan.encoding)
        new_layers.append(replace(_with_carried(layer, carried), data_source=name))

    steps = aggregate_steps + bin_steps
    if any(not isinstance(step, _Branch) for step in steps):
        steps = _with_statistics_marker(steps)
    return new_layers, steps


_POSITIONAL = ("x", "y", "x2", "y2")
# The channels a composition keeps from its left operand rather than scoping
# them to that operand's layers.
_CHART_CHANNELS = frozenset(_POSITIONAL) | _POLAR_CHANNELS | _FACET_CHANNELS


def _effective_encoding(layer: _Layer, parent: dict) -> dict:
    """The bindings *layer* draws: its own merged over its *parent*'s, channel by channel.

    *parent* is the encoding of the one chart the layer inherits from, the
    chart-level encoding on the wire: the host's written encoding for a layer
    attached with ``Chart.layer`` or ``SecondaryY``, and what the caller wrote
    on the composition itself (plus layer 0's written x/y, the axis
    fallback) for a ``+`` operand's layer, whose operand's own bindings
    already sit on its layers. A layer never inherits from a sibling.

    A channel the layer binds itself, under any spelling that draws as that
    channel (``fill`` and ``stroke`` draw as ``color``,
    :func:`~ferrum.encoding._aliases.resolved_channel`), shadows the
    parent's. Every other parent binding arrives whole: field, type,
    aggregate, bin, scale and title. A layer reading a named output
    (``data_source``) takes no positional channel from *parent*, as Rust's
    ``inherit_non_positional`` gives it none; its x/y fallback is copied onto
    it by ``ferrum.chart._fill_chart_positional``. A parent field names the
    same-named column of the rows the layer reads. A ``None`` value binds
    nothing, so the parent's binding arrives as for a channel the layer
    leaves out. A binding an earlier step carried onto the
    layer from a nearer parent (:func:`_own_channels`) is drawn, but shadows
    no alias spelling of it the way the layer's own does. A parent's x, y, x2
    or y2 arrives only where the layer's mark draws it
    (:func:`~ferrum._layer.inherited_positional_taken`), so the layer's
    statistic never groups by a positional field it does not draw.
    """
    own_channels = _own_channels(layer)
    own = {resolved_channel(channel) for channel in own_channels}
    bound = _bound(layer.encoding)
    # A channel the layer binds under the parent's own spelling keeps the
    # parent's position in the result (the layer's value replaces it below),
    # so the order the effective groupby reads is the parent's, then the
    # layer's own additions.
    inherited = {
        channel: value
        for channel, value in parent.items()
        if (layer.data_source is None or channel not in _POSITIONAL)
        and (channel in bound or resolved_channel(channel) not in own)
    }
    offered = [channel for channel in _POSITIONAL if channel in inherited and channel not in bound]
    taken = inherited_positional_taken(layer.mark, own_channels, offered)
    inherited = {
        channel: value
        for channel, value in inherited.items()
        if channel not in offered or channel in taken
    }
    if not draws_inherited_color(layer.mark) and "color" not in own:
        inherited = _color_as_grouping(inherited)
    return {**inherited, **bound}


def _color_as_grouping(inherited: dict) -> dict:
    """*inherited* with its color bindings replaced by one grouping-only key.

    A layer whose mark draws only a color it binds itself still computes its
    statistic per group of every inherited color spelling, so its labels
    stay one per colored mark. The non-aggregated spellings become one
    :class:`~ferrum._encode_lowering._GroupingOnly` at the place of the first
    of them, in the parent's order; an aggregated color is a value, not a
    group, and joins none. No member, no key. Called only when the layer
    binds no color under any spelling, so none of them is shadowed.
    """
    members = tuple(
        (channel, value)
        for channel, value in inherited.items()
        if resolved_channel(channel) == "color"
        and not (isinstance(value, ChannelBase) and value.option("aggregate"))
    )
    out: dict = {}
    for channel, value in inherited.items():
        if resolved_channel(channel) != "color":
            out[channel] = value
        elif members and GROUPING_KEY not in out:
            out[GROUPING_KEY] = _GroupingOnly(members)
    return out


def _bound(encoding: dict) -> dict:
    """*encoding* without its unset channels: a ``None`` value binds nothing."""
    return {channel: value for channel, value in encoding.items() if value is not None}


def _own_channels(layer: _Layer) -> list[str]:
    """The channels *layer* binds itself: its bound encoding keys, less those it inherited.

    The one reader of ownership: a binding an earlier step carried onto the
    layer from a parent (``inherited_channels``, ``inherited_positional``)
    stays inherited at every later step.
    """
    inherited = set(layer.inherited_channels) | set(layer.inherited_positional)
    return [channel for channel in _bound(layer.encoding) if channel not in inherited]


def _with_carried(layer: _Layer, encoding: dict) -> _Layer:
    """*layer* drawing *encoding*, each non-positional binding it does not own listed as inherited.

    Every step that writes a parent's binding onto a layer goes through
    here, so Rust reads the carried binding as inherited, as it reads the
    same binding completed from the chart-level encoding. A binding the
    layer declared, and one rewritten from it, stays the layer's own.
    """
    own = set(_own_channels(layer))
    # The grouping key is never inherited on the wire: Rust has no binding to
    # complete it from.
    carried = [c for c in encoding if c not in own and c not in _POSITIONAL and c != GROUPING_KEY]
    listed = tuple(dict.fromkeys([*layer.inherited_channels, *carried]))
    if encoding == layer.encoding and listed == layer.inherited_channels:
        return layer
    return replace(layer, encoding=encoding, inherited_channels=listed)


def _composition_encoding(encoding: dict) -> dict:
    """The bindings of a left ``+`` operand's *encoding* that stay with the composition.

    Its positional and polar channels are the composition's axis fallback,
    which a layer binding none of them takes (layer 0's written x/y), and its
    facet channels are the panel's facet. Every other binding is the
    operand's own (:func:`_scoped_to_layers`).
    """
    return {channel: value for channel, value in encoding.items() if channel in _CHART_CHANNELS}


def _scoped_to_layers(layers: list, encoding: dict) -> list:
    """Carry a ``+`` operand's own *encoding* onto the *layers* it contributes.

    The operand's encoding is the parent of its own layers and of no other
    layer of the composition, so each layer takes, whole, every binding of it
    the layer does not bind itself (:func:`_effective_encoding`), except the
    channels that stay with the composition (:func:`_composition_encoding`),
    and lists each as inherited (:func:`_with_carried`).
    A ``detail`` reaches only the layers binding none whose mark draws it,
    as a chart-level one does
    (:func:`~ferrum.encoding._aliases.layered_chart_detail`).
    """
    parent = {
        channel: value
        for channel, value in encoding.items()
        if channel not in _CHART_CHANNELS and channel != "detail"
    }
    details: list = [None] * len(layers)
    unbound = [i for i, layer in enumerate(layers) if "detail" not in layer.encoding]
    if "detail" in encoding and unbound:
        reached = layered_chart_detail(encoding["detail"], [layers[i].mark for i in unbound])
        for i, detail in zip(unbound, reached, strict=True):
            details[i] = detail
    out = []
    for layer, detail in zip(layers, details, strict=True):
        own_parent = parent if detail is None else {**parent, "detail": detail}
        out.append(_with_carried(layer, _effective_encoding(layer, own_parent)))
    return out


def _carried_bindings(layer: _Layer, parent: dict, drawn: dict) -> dict:
    """The bindings *layer* carries on the wire so that it draws *drawn*.

    Rust completes each layer from the chart-level encoding, *parent*, so an
    inherited binding *drawn* takes unchanged from *parent* stays there. The
    layer carries the rest: its own bindings, and each inherited one its
    statistic lowered or whose ``stack=`` its mark cannot honor removed,
    which the chart-level copy does not name.
    """
    bound = _bound(layer.encoding)
    return {
        channel: value
        for channel, value in drawn.items()
        if channel in bound or parent.get(channel) is not value
    }


def _layers_over_parent(layers: list, parent: dict, coord: object = None) -> list:
    """Prepare each of *layers* to draw its effective encoding over *parent*.

    Each layer carries only its own bindings (:func:`_carried_bindings`); the
    rest it inherits from *parent* on the wire. An inherited ``stack=`` is
    judged by the mark of each layer that draws it (:func:`_stack_judged`), as
    *parent* is no layer's drawn encoding. A layer reading the chart's tail
    whose effective encoding computes a statistic will read that statistic's
    named output, where Rust passes it none of the chart's positional
    channels, so it takes them from the chart's encoding
    (``positional_source="encoding"``, filled by
    ``ferrum.chart._fill_chart_positional``, which lists each copy in
    ``inherited_positional``). *coord* is the chart's, the fallback of each
    layer's own.
    """
    out = []
    for layer in layers:
        effective, unstacked = _stack_judged(
            layer, _effective_encoding(layer, parent), layer.coord if layer.coord else coord
        )
        layer = _with_carried(layer, _carried_bindings(layer, parent, effective))
        # A stack-less copy of an inherited x/y/x2/y2 stays inherited on the
        # wire, as the parent's binding it replaces would be.
        positional = [channel for channel in unstacked if channel in _POSITIONAL]
        if positional:
            listed = tuple(dict.fromkeys([*layer.inherited_positional, *positional]))
            layer = replace(layer, inherited_positional=listed)
        if (
            layer.data_source is None
            and layer.positional_source is None
            and any(_encoding_statistics(effective))
        ):
            layer = replace(layer, positional_source="encoding")
        out.append(layer)
    return out


def _stack_judged(layer: _Layer, effective: dict, coord: object) -> tuple[dict, list[str]]:
    """*effective* with each inherited ``stack=`` *layer*'s mark cannot honor removed, and warned.

    A binding the layer draws from its parent arrives whole when its mark
    honors the stack (:func:`ferrum.position.stack_honored`: bar and area on
    any channel, arc and text on the channel a ``CoordPolar`` theta draws
    as), and as a stack-less copy otherwise, so a point or line under a
    stacked parent binding draws every row. The layer's own bindings are
    judged where its wire is built. Returns the judged encoding and the
    channels whose stack was removed.
    """
    mark = layer_mark(layer.mark)
    theta_axis = _theta_axis(coord)
    bound = _bound(layer.encoding)
    out = dict(effective)
    for channel, ch in effective.items():
        if channel in bound or not isinstance(ch, ChannelBase) or not ch.option("stack"):
            continue
        if stack_honored(mark, polar_theta=channel == theta_axis):
            continue
        warn_stack_ignored(ch.option("stack"), mark)
        out[channel] = ch._with_options(stack=None)
    return out, [channel for channel in out if out[channel] is not effective[channel]]


def _statistic_columns(encoding: dict) -> set:
    """The output columns of the aggregates and bins *encoding*'s channels compute."""
    encoding = _expand_grouping(encoding)
    aggregates, bins = _encoding_statistics(encoding)
    columns = {a.output_col for a in aggregates}
    if bins:
        lowered = _statistic_plan(encoding, None).encoding
        columns |= {ch.field for ch in lowered.values() if isinstance(ch, ChannelBase)}
        columns -= {ch.field for ch in encoding.values() if isinstance(ch, ChannelBase)}
    return columns


def _record_columns(data, rows: ChartRows) -> list[str] | None:
    """The columns of the data record *rows* over the frame *data*, when known; else ``None``.

    Known when no step of the record advances the chain: then they are the
    record's own ``columns`` (a composition's left operand's), else, when
    the record starts from the source, *data*'s columns. An encoding
    aggregate or bin is computed per layer and is no chain step. Rows a
    user's ``.transform(...)`` step produced are the engine's to know.
    """
    if data is None or any(_advances_chain(t) for t in rows.steps):
        return None
    if rows.columns is not None:
        return list(rows.columns)
    if rows.input is not None:
        return None
    return list(data.columns)


def _known_chart_data_columns(chart) -> list[str] | None:
    """The columns of *chart*'s data, when Python knows them; else ``None``.

    Read off the chart's data record (:func:`_chart_rows_of`) by
    :func:`_record_columns`: on a ``+`` composition they are its left
    operand's, never the combined frame's.
    """
    return _record_columns(chart._data, _chart_rows_of(chart))


def _known_columns(layer: _Layer, chart) -> list[str] | None:
    """The columns of the rows *layer* reads, when Python knows them; else ``None``.

    A layer reading its chart's data (``chart_data`` / ``reads_chart_data``)
    reads the chart's data record (:func:`_known_chart_data_columns`). A
    layer reading the chart's tail reads the chart's chain: known when no
    step of it advances, and then the chart's data columns. A layer reading
    an operand's named output is the engine's to know.
    """
    if layer.chart_data or layer.reads_chart_data:
        return _known_chart_data_columns(chart)
    if chart._data is None or layer.data_source is not None:
        return None
    if any(_advances_chain(t) for t in _chain_steps(list(chart._transforms or []))):
        return None
    return list(chart._data.columns)


def _refuse_missing_statistic_inputs(layers: list, chart) -> None:
    """Refuse a per-layer statistic whose input or groupby column the layer's rows lack.

    Each layer's statistic is read off its effective encoding over *chart*'s
    (:func:`_effective_encoding`), so a statistic it inherits from its parent
    is checked over the rows it reads too. Checked where Python knows those
    rows (:func:`_known_columns`). A missing column is voiced by the render
    gate's own message (``ferrum._core.layer_unknown_column_message``), naming
    a layer a figure or composite mark built by that call (``_Layer.entry``)
    as the gate does: the
    statistic remedy when it is another layer's statistic output or a column
    the chart's composite mark binds (one its layers reading a named output
    draw), else the plain message. Any other layer is left to the transform
    stage.
    """
    from ferrum._core import layer_unknown_column_message

    effective = [_effective_encoding(layer, chart._encoding) for layer in layers]
    statistics: set = set()
    for layer, encoding in zip(layers, effective):
        statistics |= _statistic_columns(encoding)
        if (
            chart._composite_kind is not None
            and layer.data_source is not None
            and not layer.reads_chart_data
        ):
            statistics |= {
                ch.field for ch in layer.encoding.values() if isinstance(ch, ChannelBase)
            }
    for i, (layer, encoding) in enumerate(zip(layers, effective)):
        aggregates, bins = _encoding_statistics(encoding)
        if not aggregates and not bins:
            continue
        columns = _known_columns(layer, chart)
        if columns is None:
            continue
        inputs = {a.field for a in aggregates if a.field} | {b.field for b in bins}
        inputs |= set(_statistic_plan(encoding, chart._facet).groupby)
        missing = inputs - set(columns)
        # A grouping-only color is refused under the spelling it was bound as.
        for channel, value in _expand_grouping(encoding).items():
            field = value.field if isinstance(value, ChannelBase) else value
            if field in missing:
                remedy = "statistic" if field in statistics else "plain"
                raise ValueError(
                    layer_unknown_column_message(
                        i, channel, field, remedy, layer.independent_y, entry=layer.entry
                    )
                )


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
    from ferrum._registry import transform_wires

    result: list = []
    # Runs of PyO3 objects are batch-serialized through one ChartSpec.
    pyo3_batch: list = []

    def flush() -> None:
        if not pyo3_batch:
            return
        serialized = transform_wires(
            [t.transform if isinstance(t, _NamedTransform) else t for t in pyo3_batch]
        )
        for j, original in enumerate(pyo3_batch):
            entry = serialized[j] if j < len(serialized) else {}
            if isinstance(original, _NamedTransform):
                entry["name"] = original.name
            result.append(entry)
        pyo3_batch.clear()

    for t in transforms:
        if isinstance(t, _StatisticsMarker):
            flush()
            result.append(dict(_STATISTICS_MARKER_WIRE))
            continue
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
