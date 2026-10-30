"""Layer encodings and pipeline steps for the :class:`~ferrum.chart.Chart` value class.

This module owns each layer's effective encoding over its parent, the
resolution of per-layer encoding aggregates and bins into *named* steps, the
pipeline step holders (``_NamedTransform``, ``_Branch``, the statistics
marker, a composition's ``_ComposedStep``) and the transform-to-JSON
serialiser. ``ferrum._compose`` places the steps in their operands' chains.

The renderer never executes a layer's own ``transforms`` standalone; a layer
reads its input batch from a *named* step via ``data_source``. The resolvers
here build those named steps and re-point the layer's ``data_source``
accordingly.

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
from ferrum.encoding._aliases import resolved_channel
from ferrum.encoding._channel_policy import _FACET_CHANNELS, _POLAR_CHANNELS
from ferrum.encoding.base import ChannelBase, _PendingAggregate, _PendingBin
from ferrum._layer import (
    _POSITIONAL_WIRE,
    _Layer,
    draws_inherited_color,
    inherited_positional_taken,
    layer_draws,
    layer_mark,
    reads_named_output,
)
from ferrum.position import stack_honored, warn_stack_ignored


class _NamedTransform:
    """Pairs a PyO3 transform object with an explicit name for serialization.

    A named step publishes its output under the name and does not advance
    its chain's unnamed tail; the ``_Layer`` reading it sets ``data_source``
    to the same name.
    """

    __slots__ = ("transform", "name")

    def __init__(self, transform: object, name: str) -> None:
        self.transform = transform
        self.name = name


class _Branch:
    """A named sub-pipeline over a chosen input, serialized as one ``branch`` step.

    ``transforms`` runs as its own chain (unnamed steps chain, named steps
    publish) starting from the pipeline's original batch when ``input`` is
    ``None``, or from the output already published under ``input``. With a
    ``frame`` (:class:`_OperandFrame`) the branch first keeps one operand's
    own rows and columns of that input, and a nested branch with no ``input``
    reads those rows. The sub-chain tail is published under ``name`` and
    every named or secondary output inside it under its own scoped key
    (``secondary_key(name, key)``). A branch never advances the outer unnamed
    chain, so a layer reading ``name`` sees its own chain applied to its own
    input whatever sits before it in the composed pipeline.

    The original transform objects are kept (not pre-serialized) so value
    equality stays valid for as long as the branch lives.
    """

    __slots__ = ("name", "input", "transforms", "frame")

    def __init__(
        self,
        name: str,
        input: str | None,
        transforms: list,
        frame: "_OperandFrame | None" = None,
    ) -> None:
        self.name = name
        self.input = input
        self.transforms = list(transforms)
        self.frame = frame

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Branch):
            return NotImplemented
        return (
            self.name == other.name
            and self.input == other.input
            and self.frame == other.frame
            and len(self.transforms) == len(other.transforms)
            and all(_same_step(a, b) for a, b in zip(self.transforms, other.transforms))
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _OperandFrame:
    """Which rows of a combined batch are one ``+`` operand's, under their own names.

    A composition over several frames holds them in one batch: each frame's
    rows carry ``marker`` (true on that frame's rows), and each of its
    columns whose name an earlier frame already holds is renamed in the
    batch. Serialized as a branch's ``frame``, which keeps the rows whose
    marker is true, writes each original column from its renamed copy and
    keeps exactly the frame's own ``columns``.
    """

    marker: str
    renames: tuple[tuple[str, str], ...]
    columns: tuple[str, ...]

    def wire(self) -> dict:
        return {
            "marker": self.marker,
            "renames": dict(self.renames),
            "columns": list(self.columns),
        }


class _StatisticsMarker:
    """The boundary in one chain between the caller's steps and the statistics ferrum lowers.

    Every step after it in its chain was lowered from a layer's encoding or
    mark (an encoding aggregate or bin, a composite mark's statistics); every
    step before it is the caller's own. Rust reads the rows at the marker as
    the caller's rows: a statistic after it groups by a binned color's bins
    when a layer drawing that color reads its output, a data-aware sort of a
    statistic's output reads the rows the statistic started from, and a
    column the caller's steps lack is refused with the statistic remedy when
    a step after the marker computes it. Serialized as an unnamed
    pass-through identity, which never advances the chain.

    Every instance is the same marker: two compare equal.
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


class _ComposedStep:
    """One written step of a ``+`` composition, as each reached operand's chain runs it.

    Every copy of one written step is this same object, so they share its
    ``group``: Rust runs each copy where its rows carry the columns the
    step reads, else on the first later rows of the same lineage that do,
    and refuses the step when no copy could run. A named step publishes its
    output inside each operand's branch and leaves the chain where it was.
    Serialized as ``{"type": "composed", "group": ..., "step": ...}``.
    """

    __slots__ = ("step",)

    def __init__(self, step: object) -> None:
        self.step = step

    @property
    def group(self) -> str:
        return f"_auto_{id(self):x}"


def _source_identity(name: str) -> _NamedTransform:
    """Build a named identity step, which publishes the batch it reads under *name*."""
    from ferrum.marks._desugar_helpers import identity_transform

    return _NamedTransform(identity_transform(name), name)


def _chain_steps(transforms: list) -> tuple:
    """Return the steps of *transforms* that run in the pipeline, without encoding sentinels."""
    return tuple(t for t in transforms if not isinstance(t, (_PendingAggregate, _PendingBin)))


def _statistics_start(transforms: list) -> int:
    """Index in *transforms* where the statistics a mark lowered begin: its first marker, else the end.

    A composite mark's statistics follow the caller's steps behind one
    statistics marker (``_desugar._with_mark_statistics``), so a step the caller
    writes after the mark resolved runs before them.
    """
    for i, step in enumerate(transforms):
        if isinstance(step, _StatisticsMarker):
            return i
    return len(transforms)


def _wire_statistic_grouped_fields(steps: list[dict]) -> set[str]:
    """Every field a lowered statistic of the pipeline wire *steps* groups by.

    Walks one chain in order and reads a step's ``groupby`` only when the
    step runs after that chain's own statistics marker (a step equal to
    :data:`_STATISTICS_MARKER_WIRE`): a caller-written step's own ``groupby``
    (``Aggregate``, ``Window``, ``Bin``, ...) never counts, even when it
    shares a field with the chart's own bindings, because it groups rows for
    its own reason, not because some drawn statistic needs the split. Every
    branch step's own ``transforms`` is read as a chain of its own, with its
    own marker deciding, so an operand's own branch prefix is excluded the
    same way and only its lowered statistics count. A composed step (a
    composition's written step) is read for neither, since it carries no
    statistic of its own.
    """
    fields: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            fields.add(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    def walk_chain(chain: list) -> None:
        marker_at = next(
            (i for i, step in enumerate(chain) if step == _STATISTICS_MARKER_WIRE), None
        )
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            if step.get("type") == "branch":
                walk_chain(step.get("transforms") or [])
                continue
            if marker_at is not None and i > marker_at:
                collect(step.get("groupby"))

    walk_chain(list(steps))
    return fields


def _split_chain(transforms: list) -> tuple[tuple, tuple]:
    """*transforms* as ``(the caller's steps, the statistics its marks lowered)``, sentinels dropped.

    The statistics may hold several lowerings, each opened by its own
    :class:`_StatisticsMarker`: the first lowering's marker is kept, as is
    every later one, so the result starts with a marker exactly when it is
    non-empty and callers that append another lowering behind it (``+``
    operand statistics, a figure's own) never collapse two into one.
    """
    steps = _chain_steps(transforms)
    at = _statistics_start(list(steps))
    return steps[:at], steps[at:]


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
    tail where it was; a composed copy advances as its step does.
    """
    if isinstance(step, (_NamedTransform, _Branch, _StatisticsMarker)):
        return False
    if isinstance(step, _ComposedStep):
        return _advances_chain(step.step)
    entry = step if isinstance(step, dict) else _transforms_to_json_list_named([step])[0]
    return entry.get("name") is None


def _layer_pending_aggregates(layer: _Layer) -> list[_PendingAggregate]:
    """Collect the encoding-level aggregate sentinels carried by *layer*."""
    return _encoding_statistics(layer.encoding)[0]


def _layer_pending_bins(layer: _Layer) -> list[_PendingBin]:
    """Collect the encoding-level bin sentinels carried by *layer*."""
    return _encoding_statistics(layer.encoding)[1]


def _layer_step(layer: _Layer, transform: object, name: str) -> "_NamedTransform | _Branch":
    """The named step a layer's own aggregate or bin runs as, inside the chain of the rows it reads.

    A layer reading its chain's tail (no ``data_source``) gets a plain named
    step, which reads that tail. A layer that reads a named output of its
    chain gets a one-step ``_Branch`` over that output, so the aggregate or
    bin sees the rows the layer draws. *transform* itself must be unnamed
    for the branch case, so it advances the branch's chain.
    """
    if layer.data_source is None:
        return _NamedTransform(transform, name)
    return _Branch(name, layer.data_source, _with_statistics_marker([transform]))


@dataclass(frozen=True)
class _LayerStatistic:
    """The step a layer's encoding statistic runs as (:func:`_resolve_layer_statistics`)."""

    step: "_NamedTransform | _Branch"
    bins: bool


def _resolve_layer_statistics(layers: list, chart) -> tuple[list, list]:
    """Resolve each layer's encoding statistic into a named step of the chain it reads.

    The renderer never executes per-layer transforms on their own; a layer
    reads its input batch from a *named* transform via ``data_source``. So
    for every layer whose effective encoding (:func:`_effective_encoding`
    over *chart*'s encoding) carries an ``aggregate=`` or a ``bin=``, its
    own or one it inherits whole from its parent, this builds the layer's
    statistic plan (:func:`~ferrum._encode_lowering._statistic_plan`,
    grouped by that encoding plus *chart*'s facet fields), builds the plan's
    one ``Aggregate`` or ``Bin`` as a named step (:func:`_layer_step`),
    points the layer's ``data_source`` at it and draws each re-pointed
    channel from the step's output columns.

    Each computing layer gets its OWN named step, so two layers compute
    their statistics independently, each at the grain its own effective
    encoding gives it. A binning layer's aggregates fold into its ``Bin``.
    The layers' addresses are relative to the chain each reads, which places
    the step (``ferrum._compose.lower_operands``).

    Returns
    -------
    (new_layers, statistics)
        ``new_layers`` is the layer list with computing layers re-pointed;
        ``statistics`` holds, per layer, its :class:`_LayerStatistic`, or
        ``None`` for a layer computing nothing.
    """
    new_layers: list = []
    statistics: list = []
    for i, layer in enumerate(layers):
        effective = _effective_encoding(layer, chart._encoding)
        plan = _statistic_plan(effective, chart._facet)
        if plan.bin is not None:
            name = f"_layer_bin_{i}"
            # Inside a branch the Bin must stay unnamed, so it advances the
            # branch's own chain and its output is what the branch publishes.
            step = plan.step(name=name if layer.data_source is None else None)
            statistics.append(_LayerStatistic(_layer_step(layer, step, name), bins=True))
        elif plan.aggregates:
            name = f"_layer_agg_{i}"
            statistics.append(_LayerStatistic(_layer_step(layer, plan.step(), name), bins=False))
        else:
            new_layers.append(layer)
            statistics.append(None)
            continue
        carried = _carried_bindings(layer, chart._encoding, plan.encoding)
        new_layers.append(replace(_with_carried(layer, carried), data_source=name))
    return new_layers, statistics


# The channels a composition keeps from its left operand rather than scoping
# them to that operand's layers.
_CHART_CHANNELS = frozenset(_POSITIONAL_WIRE) | _POLAR_CHANNELS | _FACET_CHANNELS


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
    (:func:`~ferrum._layer.reads_named_output`) takes no positional channel
    from *parent*, as Rust's ``inherit_non_positional`` gives it none; its
    x/y fallback is copied onto
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
    # layer's own additions. That holds for a positional channel copied onto
    # a layer reading its chain's tail too; a layer reading a named output
    # takes no other positional channel from the parent.
    named = reads_named_output(layer)
    inherited = {
        channel: value
        for channel, value in parent.items()
        if (
            not named
            or channel not in _POSITIONAL_WIRE
            or (layer.data_source is None and channel in bound)
        )
        and (channel in bound or resolved_channel(channel) not in own)
    }
    offered = [
        channel for channel in _POSITIONAL_WIRE if channel in inherited and channel not in bound
    ]
    taken = inherited_positional_taken(layer.mark, own_channels, offered)
    inherited = {
        channel: value
        for channel, value in inherited.items()
        if channel not in offered or channel in taken
    }
    grouped = set()
    if not draws_inherited_color(layer.mark) and "color" not in own:
        grouped.add("color")
    if "detail" not in own and not layer_draws(layer.mark, "detail"):
        grouped.add("detail")
    if grouped:
        inherited = _as_grouping(inherited, grouped)
    return {**inherited, **bound}


def _as_grouping(inherited: dict, channels: set) -> dict:
    """*inherited* with its bindings on *channels* replaced by one grouping-only key.

    A layer whose mark draws only a color it binds itself, or no detail,
    still computes its statistic per group of every inherited color
    spelling and every inherited detail, so its labels stay one per colored
    mark and at the grain its parent's detail gives. The non-aggregated
    bindings whose resolved channel is in *channels* become one
    :class:`~ferrum._encode_lowering._GroupingOnly` at the place of the first
    of them, in the parent's order; an aggregated color is a value, not a
    group, and joins none. No member, no key. Called only for channels the
    layer binds under no spelling, so none of them is shadowed.
    """
    members = tuple(
        (channel, value)
        for channel, value in inherited.items()
        if resolved_channel(channel) in channels
        and not (isinstance(value, ChannelBase) and value.option("aggregate"))
    )
    out: dict = {}
    for channel, value in inherited.items():
        if resolved_channel(channel) not in channels:
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
    carried = [
        c for c in encoding if c not in own and c not in _POSITIONAL_WIRE and c != GROUPING_KEY
    ]
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
    A ``detail`` reaches every layer binding none, as a chart-level one
    does: a layer whose mark draws it splits by it, and every other layer
    groups its statistic by it (:func:`_effective_encoding`), so a label
    stays at its bars' grain. A detail no layer draws and no statistic
    groups by is reported when the chart is lowered
    (:func:`~ferrum.encoding._aliases.warn_ignored_details`), as a
    chart-level one is.
    """
    parent = {
        channel: value for channel, value in encoding.items() if channel not in _CHART_CHANNELS
    }
    return [_with_carried(layer, _effective_encoding(layer, parent)) for layer in layers]


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
        positional = [channel for channel in unstacked if channel in _POSITIONAL_WIRE]
        if positional:
            listed = tuple(dict.fromkeys([*layer.inherited_positional, *positional]))
            layer = replace(layer, inherited_positional=listed)
        if not reads_named_output(layer) and any(_encoding_statistics(effective)):
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
    judged where its wire is built.
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
      ``{"type": "branch", "name", "input", "transforms"}`` (plus ``frame``
      when it carries one) with its own sub-chain serialized by this same
      function.
    * **A** :class:`_ComposedStep` — serialized as
      ``{"type": "composed", "group", "step"}``, its step by this same
      function.

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
        if isinstance(t, _ComposedStep):
            flush()
            result.append(
                {
                    "type": "composed",
                    "group": t.group,
                    "step": _transforms_to_json_list_named([t.step])[0],
                }
            )
            continue
        if isinstance(t, _Branch):
            flush()
            entry = {
                "type": "branch",
                "name": t.name,
                "input": t.input,
                "transforms": _transforms_to_json_list_named(t.transforms),
            }
            if t.frame is not None:
                entry["frame"] = t.frame.wire()
            result.append(entry)
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
