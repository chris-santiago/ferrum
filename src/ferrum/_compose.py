"""The operands of a ``+`` composition: one record each, lowered once.

A chart drawn from layers is a list of operands. A flat chart, a composite
mark and a host with layers attached are one operand; ``a + b`` (and
``Chart.layer(fm.Layer(data=...))``, which composes the same way) is the
operands of ``a`` followed by those of ``b``, at any nesting depth. Each
operand reads one frame of the chart's frame registry and carries its own
chain in three parts: the caller's steps, the composition steps that reach
it, and the statistics its mark lowered. Each ``_Layer`` names the operand it
belongs to (``_Layer.operand``) and addresses keys inside that operand's
chain.

``Chart.to_spec`` lowers the operands once (:func:`lower_operands`): a chart
of one operand at the pipeline's top level, and each operand of a
composition as one branch whose chain is ``[own steps][composition
steps][statistics marker][its statistics]``, its layers reading keys scoped
inside it. Every operand lowers to exactly one branch, and operands over one
frame whose chains are equal read the same branch: one branch per distinct
(frame, chain). A composition step reaches the left operand and every operand over
the composition's frame (the left operand's), and never an operand over
another frame, as a Vega-Lite parent transform reaches the children that
share its data.

This module never imports ``Chart`` at module level (``chart.py`` imports
it).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ferrum._encode_lowering import (
    COMPOSITE_GROUPING_CHANNELS,
    _encoding_statistics,
    _expand_grouping,
    _grouping_bindings,
    _over,
    _rebind_positional,
    _refuse_categorical_input,
    _refuse_fieldless_categorical_type,
    _refuse_fieldless_positional,
    _refuse_unreadable_option,
    _statistic_plan,
)
from ferrum._layer import _Layer
from ferrum._layer_transforms import (
    _Branch,
    _OperandFrame,
    _advances_chain,
    _effective_encoding,
    _source_identity,
    _split_chain,
    _statistic_columns,
    _transforms_to_json_list_named,
    _with_statistics_marker,
)
from ferrum.encoding._aliases import _channel_field
from ferrum.encoding.base import _FORMAT_TYPE_ALIASES, ChannelBase

#: The key an operand's rows are published under, after its own steps and the
#: composition steps that reach it: the rows a layer attached to the chart
#: reads (``Chart.layer``, ``SecondaryY``).
ROWS_KEY = "_rows"

#: A pass-through step: the chain of an operand branch over the composition's
#: own frame with no step of its own.
_PASS_THROUGH = {"type": "identity"}

#: The channels a statistic reads its columns from, and so the ones whose
#: binding a rebuild records as an input (:attr:`_CompositeSource.inputs`).
_INPUT_CHANNELS = ("x", "y")


def _memoized(rebuild: Callable[..., Any]) -> Callable[..., Any]:
    """*rebuild*, answering a repeated call with arguments equal to an earlier one from its memo.

    *rebuild* is a pure function of its arguments: two calls with equal
    (``==``) arguments build the same chart, so the second call need not run
    it again -- in particular, need not re-invoke a figure function and
    re-raise its build-time warnings. The memo is a short list of ``(args,
    result)`` pairs compared with ``==``, not a ``dict`` keyed by the
    arguments: ``rebuild``'s arguments are themselves plain ``dict``s
    (``own``, ``parent``), which Python refuses as a dict key outright
    (unhashable), so a hash-based cache is not an option here regardless of
    what the bindings inside them define. Held in this call's own
    closure -- never a module-level cache. A :class:`_CompositeSource`
    record owns its own memo through the *rebuild* closure it carries, and a
    chart clone shares that closure (and so the memo) with the record it was
    cloned from, since cloning never rebuilds the record itself.
    """
    memo: list[tuple[tuple, Any]] = []

    @functools.wraps(rebuild)
    def memoized(*args: Any) -> Any:
        for key, result in memo:
            if key == args:
                return result
        result = rebuild(*args)
        memo.append((args, result))
        return result

    return memoized


@dataclass(frozen=True)
class _CompositeSource:
    """The rebuild record of a chart's statistics, built before its grouping was known. Never serialized.

    A composite mark resolves when its chart becomes an operand or takes an
    attached layer, and a figure builds its statistics when it is called;
    either can come before the bindings its statistics should group by
    (a host's later ``.encode()``, a composition's encoding). The record
    builds them again under new ones.

    Attributes
    ----------
    own :
        The grouping bindings (:func:`~ferrum._encode_lowering._grouping_bindings`)
        the statistics were built with as the chart's own, as ``(channel,
        binding)`` pairs.
    grouping :
        The parent grouping bindings they were built with, as ``(channel,
        binding)`` pairs; empty outside any composition.
    layers :
        The layers the build drew, before any merge rewrote them.
    steps :
        The statistics the build lowered, by identity in the chain.
    inputs :
        For each positional channel (:data:`_INPUT_CHANNELS`),
        ``(channel, input, built)``: the binding the statistics read
        (``None`` when the chart left the channel unbound), and the binding
        the build left on the chart's own encoding (``None`` when it left
        none). A binding on the chart that is not the *built*
        object and names another field than the *input* does was written
        after the build (:func:`written_inputs`).
    rebuild :
        ``rebuild(own, parent, facet, written)``: the record of the same build
        under own grouping bindings *own*, parent grouping bindings *parent*
        and the positional bindings *written* after the build (a
        ``{channel: binding}`` mapping, :func:`written_inputs`), within
        *facet*'s panels.
    regroups :
        Whether a change of grouping builds it again: ``False`` for a figure's
        operand that lowered no statistics (a scatter beside a fit), which
        is drawn again only when a positional input is written.
    data :
        The rows the build was drawn over, for a figure that is a chart of one
        operand, else ``None``. A figure that drops the rows null in the
        columns it read (residplot) draws a different frame for another
        column, so the chart that resolves the record on a written input takes
        the rebuilt build's rows, and every door reads them from the resolved
        chart, a ``+`` composition's frame registry included.
    post_build :
        The regression figure's post-build record
        (:class:`_PostBuildPositional`) the build carried, or ``None``: a
        rebuild on a written input replaces the chart's record for the
        channels it rebuilt, so the chart reads the rebuilt build's own
        bindings as the build's, not as written ones.
    """

    own: tuple
    grouping: tuple
    layers: tuple
    steps: tuple
    inputs: tuple
    rebuild: Callable[[dict, dict, Any, dict], "_CompositeSource"]
    regroups: bool = True
    post_build: Any = None
    data: Any = None


def composite_source(pre: Any, resolved: Any, parent: dict | None = None) -> _CompositeSource:
    """The record of *pre*'s pending mark resolving to *resolved* under parent grouping *parent*.

    Its rebuild resolves *pre*'s pending mark again
    (:func:`ferrum._desugar._resolve_pending_impl`, the one place a
    composite's statistics are built), *pre*'s grouping and positional
    bindings replaced by the new ones. The rebuild is memoized
    (:func:`_memoized`), so rendering the same chart more than once under the
    same grouping and inputs resolves the mark again only the first time, not
    once per render.
    """
    from ferrum._desugar import _resolve_pending_impl
    from ferrum.composition import _expand_layers

    def rebuild(own: dict, parent: dict, facet: Any, written: dict) -> _CompositeSource:
        chart = _with_statistic_bindings(pre, own, written)
        return composite_source(
            chart, _resolve_pending_impl(chart, parent=parent, facet=facet), parent
        )

    rebuild = _memoized(rebuild)

    before = len(_split_chain(list(pre._transforms or []))[1])
    lowered = _split_chain(list(resolved._transforms or []))[1]
    return _CompositeSource(
        own=tuple(_grouping_bindings(resolved._encoding).items()),
        grouping=tuple((parent or {}).items()),
        layers=tuple(_expand_layers(resolved)[0]),
        steps=tuple(lowered[before:]),
        inputs=tuple(
            (channel, pre._encoding.get(channel), resolved._encoding.get(channel))
            for channel in _INPUT_CHANNELS
        ),
        rebuild=rebuild,
    )


def _with_statistic_bindings(chart: Any, own: dict, written: dict) -> Any:
    """A clone of *chart* whose grouping bindings are *own* and positional bindings *written*, its others unchanged."""
    new = chart._clone()
    new._encoding = {
        **{
            channel: binding
            for channel, binding in chart._encoding.items()
            if channel not in COMPOSITE_GROUPING_CHANNELS
        },
        **own,
        **written,
    }
    return new


def written_inputs(chart: Any, record: _CompositeSource) -> dict:
    """The positional bindings the caller wrote on *chart* after *record*'s build, as ``{channel: binding}``.

    On a chart of one operand (attached data-less layers included), a live
    x/y binding is a written input when it is not the object the build left
    there and names a field other than the field the statistics read: a
    binding restating the same column, one without a field, or the build's
    own binding with another title is not (the chart keeps drawing what it
    draws). A composition's x/y is never one: it is a parent binding, and
    each operand's layers bind their own. The exception is a regression
    figure's chart, which is a composition of its own scatter and fit and
    carries the figure's post-build record (:class:`_PostBuildPositional`):
    its x/y are the figure's own, and the record names the columns a
    binding may restate without being another variable (the called field,
    the column the figure draws).
    """
    post: _PostBuildPositional | None = chart._post_build_positional
    if chart._operands and post is None:
        return {}
    written = {}
    for channel, input_binding, built in record.inputs:
        live = chart._encoding.get(channel)
        if live is None or live is built:
            continue
        field = _channel_field(live)
        if field is None or field == _channel_field(input_binding):
            continue
        spec = post.channels.get(channel) if post is not None else None
        if spec is not None and field in (spec.called_field, spec.drawn_column):
            continue
        if built is not None and live == built._with_options(title=live.option("title")):
            # The build's own binding retitled (``Chart.labs``), not another column.
            continue
        written[channel] = live
    return written


@dataclass(frozen=True)
class _PostBuildChannel:
    """One x/y channel's contract for a binding a regression figure's caller writes after the call.

    *own* is the channel object the figure itself bound at build time,
    compared by identity against the chart's live encoding, never by
    equality, so only a binding the caller actually wrote later is read.
    *called_field* is the field the caller named this channel with at the
    call; *drawn_column* is the column the figure's layers actually draw
    (the same field for regplot/lmplot, residplot's own batch column for
    residplot). *default_title* is the title
    :func:`~ferrum._encode_lowering._rebind_positional` falls back to when
    the caller's re-pointed binding carries none of its own.
    *honors_fieldless* is ``True`` only for a channel the figure computes
    rather than reads (residplot's residual y): a field-less binding
    styles that axis instead of being refused. *categorical_reads*/
    *categorical_disallowed* are
    :func:`~ferrum._encode_lowering._refuse_categorical_input`'s own
    arguments for this channel, and also gate a field-less
    *honors_fieldless* binding's own declared type (there is no field for
    :func:`_refuse_categorical_input` itself to check). *check_unreadable*
    refuses whichever of *unreadable*'s options the figure's call door
    reads a different way (empty, honoring every one, for every channel
    but residplot's y, since the response column the call named is not
    the residual the figure draws): a post-build ``scale=``/``format=``/
    ``format_type=``/``axis=`` on the called field pushes down onto the
    figure's own layers instead, :func:`_push_down_positional_options`,
    the later write always replacing whatever the layer already carries.
    *call_type* is the
    type the call's own binding declared (``None`` reads as ``"Q"``, the
    positional default); a later binding that declares a *different*
    type is refused rather than silently read or silently dropped
    (``apply_post_build_positional``'s own comparison), with *type_remedy*
    overriding the default ending the same way *scale_remedy* (passed
    straight through to *check_unreadable*) overrides it for ``scale=``.

    *always_honors_drawn_column* is ``True`` only where a binding naming
    the figure's drawn column is a second, legitimate way to spell the
    same column the call named, not a later override of it (residplot's
    predictor, when the caller's own column happens to share its name
    with residplot's internal one): such a binding is read, never
    refused, even though its field also equals *called_field*.
    *restore_declared_type* is the type the call-time binding declared
    (only ever ``"T"``) when :func:`~ferrum._encode_lowering._rebind_positional`
    would otherwise force it back to ``"Q"``; ``None`` leaves the forced
    type alone. Needed only where the chart's own encoding is read
    directly at render time with no intervening layer of its own already
    carrying the declared type forward (flat residplot); harmless
    elsewhere, since a layered figure's own layer keeps its type
    regardless of what this record writes onto the chart's encoding. By
    the time this applies, a differing declared type has already been
    refused above, so the type this restores is always the one the
    binding itself declared, or the call's own.
    """

    own: ChannelBase
    called_field: str
    drawn_column: str
    default_title: str | None
    honors_fieldless: bool
    categorical_reads: Callable[[str, str, str], Any]
    categorical_disallowed: tuple[str, ...]
    check_unreadable: Callable[[str, str, ChannelBase], None]
    unreadable: tuple[str, ...] = ()
    call_type: str | None = None
    type_remedy: Callable[[str], str] | None = None
    always_honors_drawn_column: bool = False
    restore_declared_type: str | None = None


@dataclass(frozen=True)
class _PostBuildPositional:
    """A regression figure's contract for an x/y binding its caller writes after the figure was built.

    *entry* is the public call the caller made (``regplot``, ``lmplot``,
    ``residplot``), the voice every refusal speaks as. *channels* holds
    each positional channel's own :class:`_PostBuildChannel`.
    *statistic_refusal* refuses a ``bin=``/``aggregate=`` on any written
    channel regardless of its field: the call-time check is field-agnostic,
    so this backstop is too, on every form including a pending
    smooth-routed one, so the one check a caller's bare bin=/aggregate=
    trips is always this one, never a later, differently-worded refusal
    from the pending statistic's own resolve-time prep.

    *refuse_undrawn* refuses a channel this figure draws on no layer
    (``x2``/``y2`` on a residual scatter), whatever its field, and runs
    before the per-channel pass below so a binding on such a channel is
    named the same way a ``bin=``/``aggregate=`` is: by this record, not
    by a later render-time error, and on a chart whose x/y the caller
    never touched too.

    Read by :func:`apply_post_build_positional`, which
    ``Chart._resolve_pending`` runs first, before any pending composite
    mark resolves.
    """

    entry: str
    channels: dict[str, _PostBuildChannel]
    statistic_refusal: Callable[[dict[str, ChannelBase]], None]
    #: A figure's own refusal for a channel it draws on no layer, called
    #: with the chart's whole encoding before any x/y is read. ``None``
    #: (every figure whose layers draw every channel a binding could name)
    #: leaves such a channel to render as it already does.
    refuse_undrawn: Callable[[dict[str, ChannelBase]], None] | None = None


def apply_post_build_positional(chart: Any) -> Any:
    """*chart*, its x/y bindings written after a regression figure's build read or refused.

    A channel the figure draws on no layer is refused before anything else
    is read (:attr:`_PostBuildPositional.refuse_undrawn`), so a chart whose
    x/y the caller never touched still names it. A binding equal by
    identity to the figure's own (:attr:`_PostBuildChannel.own`)
    was never rewritten and is left alone. Of the rest: any field's
    ``bin=``/``aggregate=`` is refused first, by
    :attr:`_PostBuildPositional.statistic_refusal`, on every form -- a
    pending smooth-routed form's own resolve-time prep would refuse the
    same thing moments later, so running it here first keeps the text one
    caller's bare ``bin=``/``aggregate=`` always gets, not whichever of
    two differently-worded refusals happens to run first. Then, per
    channel: a binding naming the figure's drawn column is read as-is
    when :attr:`_PostBuildChannel.always_honors_drawn_column` says so,
    even where that column and the called field are spelled the same; a
    binding on the field the caller originally named (and *not* pending --
    a pending form's own resolve-time prep reads this directly) refuses a
    declared ``N``/``O`` (the categorical-input rule), then a declared
    type that differs from the call's own
    (:attr:`~_PostBuildChannel.call_type`), then whichever of
    :attr:`~_PostBuildChannel.unreadable` the call door reads a different
    way, then re-points onto the figure's drawn column
    (:func:`~ferrum._encode_lowering._rebind_positional`), restoring a
    call-time declared type :func:`_rebind_positional` would otherwise
    force away (:attr:`_PostBuildChannel.restore_declared_type` -- safe
    by now, since a *differing* declared type was already refused above);
    a field-less binding either refuses a declared ``N``/``O``/``T`` a
    computed axis cannot place, then styles that axis the same way, or is
    refused naming the column to restate; any other field (another
    variable) is left as it is: it is a written input
    (:func:`written_inputs`), read when the chart's statistics are built
    again from it (:func:`regrouped`). Every
    channel this pass re-points -- never one left alone -- then pushes its own
    ``scale=``/``format=``/``format_type=``/``axis=`` down onto the
    figure's own layers (:func:`_push_down_positional_options`), the
    later write always replacing whatever a layer already carries for
    that option.

    The returned chart's own record points each re-pointed channel's
    :attr:`~_PostBuildChannel.own` at the binding this pass just wrote,
    not the pre-render one: ``Chart._resolve_pending`` runs this function
    more than once reaching one render, and the identity check at the top
    (``binding is spec.own``) must recognize a channel this same pass
    already settled on a later run, rather than reading its own output
    (forced to ``type="Q"`` by :func:`_rebind_positional`, regardless of
    *call_type*) as a fresh caller write that conflicts with the call's
    own declared type.

    Returns *chart* unchanged when it carries no record, or the record's
    channels are all untouched.
    """
    record: _PostBuildPositional | None = chart._post_build_positional
    if record is None:
        return chart
    from ferrum._entry import voiced_as

    # Before any x/y is read, and before the early return below: a record
    # whose own x/y the caller never touched still refuses a channel the
    # figure draws on no layer.
    if record.refuse_undrawn is not None:
        with voiced_as(record.entry):
            record.refuse_undrawn(chart._encoding)
    written: dict[str, ChannelBase] = {}
    for channel, spec in record.channels.items():
        binding = chart._encoding.get(channel)
        if binding is None or binding is spec.own:
            continue
        written[channel] = binding
    if not written:
        return chart

    pending = chart._pending_stat_mark is not None
    deferred_to_pending_prep = False
    with voiced_as(record.entry):
        record.statistic_refusal(written)
        updates: dict[str, ChannelBase] = {}
        for channel, binding in written.items():
            spec = record.channels[channel]
            field = binding.field
            if field is None:
                if spec.honors_fieldless:
                    declared_type = binding.option("type")
                    if declared_type in spec.categorical_disallowed:
                        _refuse_fieldless_categorical_type(
                            record.entry,
                            channel,
                            binding,
                            reads=spec.categorical_reads,
                            subject=spec.drawn_column,
                        )
                    updates[channel] = _rebind_positional(
                        spec.drawn_column, binding, type(binding), title=spec.default_title
                    )
                else:
                    _refuse_fieldless_positional(
                        record.entry,
                        channel,
                        binding,
                        called_field=spec.called_field,
                        unreadable=spec.unreadable,
                        call_type=spec.call_type,
                        pending=pending,
                    )
            elif spec.always_honors_drawn_column and field == spec.drawn_column:
                continue
            elif field == spec.called_field:
                if pending:
                    # A pending smooth-routed form: its own resolve-time
                    # prep, about to run right after this returns, reads
                    # this binding from the final encoding directly, so
                    # there is nothing more to do here. Once that prep
                    # runs it will have read and settled this binding for
                    # good, so the record that deferred to it is cleared
                    # below rather than kept to be read again, wrongly, as
                    # an eager form's on a later call reached by the same
                    # chart (`Chart._resolve_pending` runs more than once
                    # per render) or a mark= override applied before this
                    # call (whose own prep never ran at all).
                    deferred_to_pending_prep = True
                    continue
                _refuse_categorical_input(
                    record.entry,
                    {channel: binding},
                    (channel,),
                    reads=spec.categorical_reads,
                    disallowed=spec.categorical_disallowed,
                )
                declared_type = binding.option("type")
                if declared_type is not None and declared_type != (spec.call_type or "Q"):
                    _refuse_unreadable_option(
                        record.entry, channel, binding, "type", remedy=spec.type_remedy
                    )
                spec.check_unreadable(record.entry, channel, binding)
                rebound = _rebind_positional(
                    spec.drawn_column, binding, type(binding), title=spec.default_title
                )
                if spec.restore_declared_type is not None:
                    rebound = rebound._with_options(type=spec.restore_declared_type)
                updates[channel] = rebound
            # else: another variable. The classifier records nothing: the
            # binding is a written input (:func:`written_inputs`), and the
            # figure is called again with it as its own column.
    if not updates and not deferred_to_pending_prep:
        return chart
    new = chart._clone()
    new._encoding = {**chart._encoding, **updates}
    if deferred_to_pending_prep:
        new._post_build_positional = None
    elif updates:
        from dataclasses import replace

        new._post_build_positional = replace(
            record,
            channels={
                channel: (replace(spec, own=updates[channel]) if channel in updates else spec)
                for channel, spec in record.channels.items()
            },
        )
    if updates and new._layers:
        new._layers = [_push_down_positional_options(updates, layer) for layer in new._layers]
    return new


#: The axis-presentation options a post-build write pushes onto the
#: figure's own layers once a channel is re-pointed, the later write
#: always replacing whatever a layer's own channel already carries for
#: that option. ``title=`` is not here: it already reaches a layer's
#: rendered axis through a separate mechanism (the chart-level encoding's
#: own title, read directly), never through a layer's own channel object.
_PUSHDOWN_OPTIONS = ("scale", "format", "axis")


def _push_down_positional_channel(layer_value: Any, pushed: ChannelBase) -> Any:
    """*layer_value* with *pushed*'s own :data:`_PUSHDOWN_OPTIONS` overlaid, replacing whatever it already carries.

    The later write always wins here, unlike
    :func:`ferrum._desugar._merge_positional_channel_axis` (a layered
    statistic's own construction-time propagation, where a layer that
    states its own presentation keeps it): *pushed* is the caller's own
    re-pointed binding, read after the figure's call already built
    *layer_value*, so it is the newer of the two. Returns ``None`` when
    *layer_value* carries no positional channel to overlay onto, or
    *pushed* sets none of :data:`_PUSHDOWN_OPTIONS` (nor either
    :data:`_FORMAT_TYPE_ALIASES` spelling), so a caller can tell "nothing
    to change" from "changed". When *pushed* carries a ``format_type``
    under either spelling, both of *layer_value*'s own spellings are
    dropped before the canonical one is written -- keeping whichever one
    *layer_value* already carried would leave two keys serializing the
    same wire field, with the later write's own `_SPEC_DICT_ORDER`
    position deciding winner instead of this function.
    """
    if not isinstance(layer_value, ChannelBase):
        return None
    extra = {
        name: pushed.option(name) for name in _PUSHDOWN_OPTIONS if pushed.option(name) is not None
    }
    for alias in _FORMAT_TYPE_ALIASES:
        format_type = pushed.option(alias)
        if format_type is not None:
            extra["format_type"] = format_type
            break
    if not extra:
        return None
    if "format_type" in extra:
        kept = {k: v for k, v in layer_value._kwargs.items() if k not in _FORMAT_TYPE_ALIASES}
        return type(layer_value)(layer_value.field, **{**kept, **extra})
    return layer_value._with_options(**extra)


def _push_down_positional_options(pushed: dict[str, ChannelBase], layer: _Layer) -> _Layer:
    """*layer*, each of *pushed*'s channels' own axis presentation overlaid onto the matching axis.

    Only the channels in *pushed* -- the ones :func:`apply_post_build_positional`
    actually re-pointed this call -- reach *layer*; a binding on another
    variable, which that function leaves out of *pushed*, never does,
    whatever else the same ``.encode()`` call writes.
    """
    from dataclasses import replace

    new_encoding = dict(layer.encoding)
    changed = False
    for axis, value in pushed.items():
        merged = _push_down_positional_channel(new_encoding.get(axis), value)
        if merged is not None:
            new_encoding[axis] = merged
            changed = True
    if not changed:
        return layer
    return replace(layer, encoding=new_encoding)


def figure_recorded(
    chart: Any, call: Callable[[dict], Any], inputs: Callable[[], dict] | None = None
) -> Any:
    """*chart*, a figure's result, carrying the record that builds its statistics again.

    *call* calls the figure again with the given bindings merged into its
    ``encode=`` (the caller's other arguments unchanged). A rebuild adds the
    own grouping bindings that changed since the figure was built (an
    unbound one as ``None``), each parent binding on a channel the figure
    binds under no spelling, on its chart or any of its layers, so the
    figure groups by them as for its own ``encode=``, and each positional
    binding written after the build (:func:`written_inputs`), which the
    call reads as its own ``x=``/``y=``: a figure under a composition's
    color regroups as a composite mark does, and a figure given another
    column afterwards is called with it. A chart of one operand carries the
    record itself; every operand of a composition carries one that reads
    its own statistics and layers from the rebuilt figure. A chart that is
    no figure chart (a pending mark records itself when it resolves), or
    whose operands lowered no statistic, is returned as is.

    *inputs* returns the positional bindings the call read, as
    ``{channel: binding}`` for the figure's chart channels (``None`` for a
    figure that reads no caller x/y): they are what a later binding is
    compared with, and what the chart's own x/y are set against.

    *call* runs through one shared, memoized closure (:func:`_memoized`):
    rendering the same chart more than once rebuilds by the same grouping
    and inputs each time, so the figure is called again only the first time
    a given (grouping, inputs) is seen, not once per render.
    """
    from dataclasses import replace

    from ferrum.encoding._aliases import resolved_channel

    if not hasattr(chart, "_operands") or chart._pending_stat_mark is not None:
        return chart
    built = _grouping_bindings(chart._encoding)
    bound = {resolved_channel(channel) for channel in built} | {
        resolved_channel(channel)
        for layer in chart._layers or []
        for channel in _grouping_bindings(layer.encoding)
    }

    def rebuilt(own: dict, parent: dict, written: dict) -> Any:
        taken = bound | {resolved_channel(channel) for channel in own}
        extra = {
            **{channel: None for channel in built if channel not in own},
            **{
                channel: binding
                for channel, binding in own.items()
                if channel not in built or built[channel] is not binding
            },
            **{
                channel: binding
                for channel, binding in parent.items()
                if resolved_channel(channel) not in taken
            },
            **written,
        }
        return call(extra)

    rebuilt = _memoized(rebuilt)

    if not chart._operands and not _split_chain(list(chart._transforms or []))[1]:
        return chart
    if chart._operands and not any(op.statistics for op in chart._operands):
        return chart

    read = inputs() if inputs is not None else {}
    declared = tuple(
        (channel, read[channel], chart._encoding.get(channel))
        for channel in _INPUT_CHANNELS
        if channel in read
    )
    grouping = tuple(built.items())
    new = chart._clone()
    if not chart._operands:
        new._composite_source = _figure_source(chart, None, built, (), rebuilt, declared, {})
        return new
    new._operands = tuple(
        replace(op, source=_figure_source(chart, i, built, grouping, rebuilt, declared, {}))
        for i, op in enumerate(chart._operands)
    )
    return new


def _figure_source(
    chart: Any,
    operand: int | None,
    own: dict,
    grouping: tuple,
    rebuilt: Callable[[dict, dict, dict], Any],
    declared: tuple,
    base: dict,
) -> _CompositeSource:
    """The record of a figure's chart, or of its *operand* when the figure is a composition.

    *declared* is the record's :attr:`_CompositeSource.inputs` and *base*
    the positional bindings already written into the build it records, which
    every later rebuild keeps unless it writes the channel again.
    """
    from ferrum.composition import _expand_layers

    if (operand is None) != (not chart._operands):
        raise RuntimeError(
            "a figure changed between one chart and a composition when its statistics "
            "were grouped by a binding made after it was built; it cannot be rebuilt"
        )
    if operand is None:
        layers = tuple(_expand_layers(chart)[0])
        steps = _split_chain(list(chart._transforms or []))[1]
    else:
        if operand >= len(chart._operands):
            raise RuntimeError(
                "a figure drew fewer operands when its statistics were grouped by a "
                "binding made after it was built; its operands cannot be rebuilt"
            )
        layers = tuple(layer for layer in chart._layers or [] if layer.operand == operand)
        steps = chart._operands[operand].statistics

    def rebuild(own: dict, parent: dict, facet: Any, written: dict) -> _CompositeSource:
        # A figure takes no facet: its statistics run inside each panel's
        # rows, as they do when the caller facets the figure itself.
        merged = {**base, **written}
        figure = rebuilt(own, parent, merged)
        after = tuple(
            (channel, merged[channel] if channel in merged else read, figure._encoding.get(channel))
            for channel, read, _ in declared
        )
        return _figure_source(figure, operand, own, tuple(parent.items()), rebuilt, after, merged)

    return _CompositeSource(
        own=tuple(own.items()),
        grouping=grouping,
        layers=layers,
        steps=tuple(steps),
        inputs=declared,
        rebuild=rebuild,
        regroups=bool(steps),
        post_build=chart._post_build_positional,
        data=chart._data if operand is None else None,
    )


@dataclass(frozen=True)
class _Operand:
    """One operand of a chart drawn from layers. Never serialized.

    Attributes
    ----------
    frame :
        Index of the frame it reads in the chart's frame registry
        (:func:`chart_frames`); ``0`` is the composition's frame, its left
        operand's.
    own :
        The caller's steps, written with ``.transform(...)`` on the chart
        the operand came from.
    statistics :
        The statistics its mark lowered when it resolved (a composite mark's
        bins, density grid or fitted curve), each lowering led by its own
        statistics marker (:func:`~ferrum._layer_transforms._split_chain`).
    composed :
        The steps written on a composition this operand is reached by, each
        the :class:`~ferrum._layer_transforms._ComposedStep` every reached
        operand shares, innermost composition first.
    source :
        The pending composite mark its statistics came from, or ``None``.
    carried :
        The grouping bindings of the compositions it was an operand of that
        are now themselves operands, nearest first, as ``(channel,
        binding)`` pairs: their encodings moved onto their layers, and stay
        the parent of this operand's statistics.
    """

    frame: int
    own: tuple
    statistics: tuple
    composed: tuple = ()
    source: _CompositeSource | None = None
    carried: tuple = ()


def chart_operands(chart: Any) -> tuple[_Operand, ...]:
    """*chart*'s operands: a composition's own, else the one its chain makes."""
    if chart._operands:
        return chart._operands
    own, statistics = _split_chain(list(chart._transforms or []))
    return (_Operand(0, own, statistics, source=chart._composite_source),)


def carried_by(operands: tuple[_Operand, ...], encoding: dict) -> tuple[_Operand, ...]:
    """*operands* of a composition that becomes an operand, carrying its *encoding*'s grouping.

    The composition's encoding moves onto its layers (``_scoped_to_layers``)
    and stays the parent of its operands' statistics, farther than any
    composition nested inside it.
    """
    from dataclasses import replace

    shared = _grouping_bindings(encoding)
    if not shared:
        return operands
    return tuple(
        replace(op, carried=tuple(_over(dict(op.carried), shared).items()))
        if op.source is not None
        else op
        for op in operands
    )


def regrouped(chart: Any) -> Any:
    """*chart* with its statistics built under the grouping bindings it and its parents now bind.

    Statistics built before their grouping was known (a composite mark that
    resolved when its chart took an attached layer or became an operand, a
    figure's statistics) are built again from their record
    (:class:`_CompositeSource`) whenever the grouping they would be built
    with differs from the one they were built with:

    - a chart of one operand, by its own grouping bindings, so a host's
      ``.encode(color="c")`` after ``.layer(...)`` fits per ``c`` as the
      same binding written before it does;
    - each operand of a composition, by the grouping bindings of every
      composition it is an operand of, the nearest winning, on each channel
      its own chart binds under no spelling (over the effective encoding),
      so ``(fit + points).encode(color="c")`` fits per ``c`` as
      its points color.

    Each layer the build drew takes every field the rebuild draws differently
    (:func:`_redrawn_layers`). Returns *chart* itself when nothing changes,
    else a clone: the caller's chart is never mutated.
    """
    if not chart._operands:
        return _regrouped_alone(chart)
    from dataclasses import replace

    shared = _grouping_bindings(chart._encoding)
    operands = list(chart._operands)
    layers = list(chart._layers or [])
    rebuilt_from: list[tuple[_CompositeSource, _CompositeSource, dict]] = []
    for i, op in enumerate(operands):
        if op.source is None:
            continue
        parent = tuple(_over(dict(op.carried), shared).items())
        written = written_inputs(chart, op.source)
        regroups = op.source.regroups and parent != op.source.grouping
        if not regroups and not written:
            continue
        source = op.source.rebuild(dict(op.source.own), dict(parent), chart._facet, written)
        operands[i] = replace(
            op,
            statistics=_with_run_replaced(op.statistics, op.source.steps, source.steps),
            source=source,
        )
        positions = [k for k, layer in enumerate(layers) if layer.operand == i]
        layers = _redrawn_layers(layers, positions, op.source, source)
        rebuilt_from.append((op.source, source, written))
    if not rebuilt_from:
        return chart
    new = chart._clone()
    new._operands = tuple(operands)
    new._layers = layers
    record, source, written = rebuilt_from[0]
    if written:
        _adopt_rebuilt_inputs(new, record, source, written)
    return new


def _adopt_rebuilt_inputs(
    new: Any, record: _CompositeSource, source: _CompositeSource, written: dict
) -> None:
    """Point *new*'s own x/y at the rebuild *source*'s build, on every channel it rebuilt or left to the build.

    A channel the caller wrote after the build (*written*), and one the
    chart still carries as the build left it (*record*'s own object), takes
    the rebuilt build's binding: the next resolution reads the chart's x/y
    against the rebuilt record's own, and a binding that is not the build's
    own object reads as the caller's. A regression figure's post-build
    record follows the same channels, so the classifier sees the rebuilt
    figure's own bindings as its own.
    """
    from dataclasses import replace

    left = {channel: binding for channel, _, binding in record.inputs}
    encoding = dict(new._encoding)
    adopted = set()
    for channel, _, binding in source.inputs:
        if channel in written or (
            encoding.get(channel) is not None and encoding[channel] is left.get(channel)
        ):
            adopted.add(channel)
            if binding is None:
                encoding.pop(channel, None)
            else:
                encoding[channel] = binding
    new._encoding = encoding
    post = new._post_build_positional
    if post is not None and source.post_build is not None and adopted:
        new._post_build_positional = replace(
            post,
            channels={
                channel: (
                    source.post_build.channels[channel]
                    if channel in adopted and channel in source.post_build.channels
                    else spec
                )
                for channel, spec in post.channels.items()
            },
        )


def _regrouped_alone(chart: Any) -> Any:
    """:func:`regrouped` for a chart of one operand, under its own grouping bindings."""
    from dataclasses import replace

    record = chart._composite_source
    if record is None:
        return chart
    own = _grouping_bindings(chart._encoding)
    written = written_inputs(chart, record)
    if own == dict(record.own) and not written:
        return chart
    source = record.rebuild(own, {}, chart._facet, written)
    new = chart._clone()
    new._transforms = list(_with_run_replaced(tuple(chart._transforms), record.steps, source.steps))
    new._composite_source = source
    if written and source.data is not None:
        new._data = source.data
    _adopt_rebuilt_inputs(new, record, source, written)
    if chart._layers is not None:
        new._layers = _redrawn_layers(
            list(chart._layers), list(range(len(chart._layers))), record, source
        )
        return new
    # A flat chart draws its one mark from its own fields, which a grouping
    # rebuild may change only in the bindings the caller already wrote on it.
    if len(source.layers) != 1 or replace(record.layers[0], encoding={}) != replace(
        source.layers[0], encoding={}
    ):
        if written:
            # A binding on another column can change what the mark is (a
            # density over a second variable draws polygons, not an area):
            # the chart draws the layers the rebuild draws.
            new._layers = list(source.layers)
            new._mark = None
            new._mark_kwargs = {}
            # The style the caller gave the flat mark after it resolved goes
            # through the door that styled it, held to the layers now drawn.
            built = record.layers[0].mark_kwargs or {}
            style = {
                key: value
                for key, value in (chart._mark_kwargs or {}).items()
                if key not in built or built[key] != value
            }
            if style:
                from ferrum._overrides import _apply_mark_overrides

                return _apply_mark_overrides(new, style, entry=chart._entry or "mark")
            return new
        raise RuntimeError(
            "a chart drew a different mark when its statistics were grouped by a "
            "binding made after it was built; it cannot be rebuilt"
        )
    return new


def _with_run_replaced(statistics: tuple, old: tuple, new: tuple) -> tuple:
    """*statistics* with the run of steps *old* (by identity) replaced by *new*."""
    n = len(old)
    for k in range(len(statistics) - n + 1):
        if all(a is b for a, b in zip(statistics[k : k + n], old)):
            return (*statistics[:k], *new, *statistics[k + n :])
    raise RuntimeError(
        "an operand's statistics no longer hold the steps its composite mark lowered; "
        "they cannot be rebuilt"
    )


def _redrawn_layers(
    layers: list, positions: list[int], record: _CompositeSource, source: _CompositeSource
) -> list:
    """*layers* with the build's layers at *positions* redrawn as the rebuild *source* draws them.

    The first ``len(record.layers)`` layers at *positions* are the build's,
    in its order; each is paired with the rebuilt layer of its name
    (:func:`_redrawn`), or by position when the build's layers are not
    uniquely named. A build layer the rebuild no longer draws (a figure's
    single-fit corner metrics with no single value once the fit is per
    group) is removed silently: the rebuild calls the figure again
    (:func:`figure_recorded`), and the figure's own call already reports a
    dropped layer when it was asked to, the same voice whatever door bound
    the grouping. A rebuilt layer the build never drew cannot be placed.
    """
    built = positions[: len(record.layers)]
    if len(built) < len(record.layers):
        raise RuntimeError(
            "a chart lost layers its statistics drew before they were grouped by a "
            "binding made after it was built; its layers cannot be rebuilt"
        )
    pairs = _paired(record.layers, source.layers)
    out = list(layers)
    dropped: list[int] = []
    for k, before, after in zip(built, record.layers, pairs):
        if after is None:
            dropped.append(k)
        else:
            out[k] = _redrawn(out[k], before, after)
    return [layer for k, layer in enumerate(out) if k not in dropped]


def _paired(before: tuple, after: tuple) -> list:
    """Each *before* layer's rebuilt counterpart in *after*, ``None`` where the rebuild draws none.

    Layers pair by name. The build's layers the rebuild draws under no name
    of theirs pair, in order, with the rebuilt layers whose names the build
    never drew, when there are as many of each (a grouped filled density
    draws its one layer as isoline segments); otherwise each is no longer
    drawn. Layers that are not uniquely named pair by position.
    """

    def named(layers: tuple) -> bool:
        names = [layer.name for layer in layers]
        return None not in names and len(set(names)) == len(names)

    if not (named(before) and named(after)):
        if len(before) != len(after):
            raise RuntimeError(
                "a composite mark drew a different number of layers when its statistics "
                "were grouped by a binding made after it was built; its layers cannot be "
                "rebuilt"
            )
        return list(after)
    by_name = {layer.name: layer for layer in after}
    unmatched = [layer.name for layer in before if layer.name not in by_name]
    added = [layer for layer in after if layer.name not in {b.name for b in before}]
    if added and len(added) != len(unmatched):
        raise RuntimeError(
            "a chart drew a layer its statistics never drew before they were grouped by a "
            "binding made after it was built; its layers cannot be rebuilt"
        )
    by_name.update(zip(unmatched, added))
    return [by_name.get(layer.name) for layer in before]


def _redrawn(layer: _Layer, before: _Layer, after: _Layer) -> _Layer:
    """*layer*, built as *before*, with every field *after* draws differently from *before*.

    What a merge wrote onto *layer* (its operand's bindings, positional
    fill, provenance) is kept; an encoding key *after* drops is removed.
    """
    from dataclasses import fields, replace

    if before == after:
        return layer
    changes = {
        f.name: getattr(after, f.name)
        for f in fields(_Layer)
        if f.name != "encoding" and getattr(before, f.name) != getattr(after, f.name)
    }
    encoding = {
        channel: value
        for channel, value in layer.encoding.items()
        if channel not in before.encoding or channel in after.encoding
    }
    encoding.update(
        {
            channel: value
            for channel, value in after.encoding.items()
            if channel not in before.encoding or before.encoding[channel] != value
        }
    )
    return replace(layer, encoding=encoding, **changes)


def chart_frames(chart: Any) -> tuple:
    """*chart*'s frame registry: a composition's, else its own data."""
    return chart._frames or (chart._data,)


def _same_frame(a: Any, b: Any) -> bool:
    """Whether two operand frames are one frame.

    The same object is one frame. Otherwise two frames are one frame when
    they hold the same schema (column names, order and dtypes) and equal
    values, NaN equal to NaN and null equal to null, so a clone, a frame
    rebuilt from the same values and a ``LazyFrame`` collected again by a
    nested ``+`` are the frame they copy.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    from ferrum._coerce import to_polars

    a, b = to_polars(a), to_polars(b)
    return a.schema == b.schema and a.equals(b, null_equal=True)


def merged_frames(lhs: tuple, rhs: tuple) -> tuple[tuple, list[int]]:
    """The registry of ``a + b`` from the operands' registries, and where each *rhs* frame went.

    Each right frame equal to one already registered (:func:`_same_frame`)
    is that frame; any other is appended. The left registry keeps its
    order, so frame ``0`` stays the composition's frame.
    """
    frames = list(lhs)
    at: list[int] = []
    for frame in rhs:
        index = next((i for i, known in enumerate(frames) if _same_frame(known, frame)), None)
        if index is None:
            index = len(frames)
            frames.append(frame)
        at.append(index)
    return tuple(frames), at


def combined_frames(frames: tuple) -> tuple[Any, tuple[_OperandFrame, ...]]:
    """The one batch a composition over *frames* sends, and each frame's selector in it.

    One frame is sent as it is, and needs no selector. Several are
    diagonally concatenated: each frame's rows carry a marker column, true
    on its own rows, and each column whose name an earlier frame already
    holds is renamed (``<column>__rhs_<frame>``), so no frame's rows read
    another's values. Generated names never collide with any frame's
    column. Data that is no supported frame is refused by ``to_polars``.
    """
    if len(frames) == 1:
        return frames[0], ()
    import polars as pl

    from ferrum._coerce import to_polars

    data = [to_polars(frame) for frame in frames]
    taken: set[str] = set().union(*(set(frame.columns) for frame in data))

    def fresh(base: str) -> str:
        name, n = base, 0
        while name in taken:
            n += 1
            name = f"{base}_{n}"
        taken.add(name)
        return name

    held: set[str] = set()
    parts = []
    selectors = []
    for i, frame in enumerate(data):
        renames = {c: fresh(f"{c}__rhs_{i:04d}") for c in frame.columns if c in held}
        marker = fresh(f"__operand_row_{i:04d}")
        part = frame.rename(renames).with_columns(pl.lit(True).alias(marker))
        held |= set(part.columns)
        parts.append(part)
        selectors.append(_OperandFrame(marker, tuple(renames.items()), tuple(frame.columns)))
    return pl.concat(parts, how="diagonal"), tuple(selectors)


def reached_by_composition(operands: tuple[_Operand, ...], steps: tuple) -> tuple[_Operand, ...]:
    """*operands* with *steps*, written on their composition, added to every operand over its frame."""
    from dataclasses import replace

    return tuple(
        replace(op, composed=op.composed + steps) if op.frame == 0 else op for op in operands
    )


# ---------------------------------------------------------------------------
# The columns of the rows a layer reads, where Python knows them
# ---------------------------------------------------------------------------


def _frame_columns_after(chart: Any, op: _Operand, steps: tuple) -> list[str] | None:
    """The columns of *op*'s frame after *steps*, when no step of them advances; else ``None``."""
    data = chart_frames(chart)[op.frame]
    if data is None or any(_advances_chain(t) for t in steps):
        return None
    return list(data.columns)


def known_chart_data_columns(chart: Any) -> list[str] | None:
    """The columns of *chart*'s data, when Python knows them; else ``None``.

    A chart's data is its left operand's rows after that operand's own
    steps and the composition steps that reach it: on a ``+`` composition
    the left operand's frame, never the combined batch. Rows a step
    produced are the engine's to know.
    """
    op = chart_operands(chart)[0]
    return _frame_columns_after(chart, op, op.own + op.composed)


def known_columns(layer: _Layer, chart: Any) -> list[str] | None:
    """The columns of the rows *layer* reads, when Python knows them; else ``None``.

    A layer reading its chart's data (``chart_data`` / ``reads_chart_data``)
    reads its operand's rows; a layer reading its operand's tail reads them
    after the operand's statistics too. A layer reading a named output is
    the engine's to know.
    """
    op = chart_operands(chart)[layer.operand]
    if layer.chart_data or layer.reads_chart_data:
        return _frame_columns_after(chart, op, op.own + op.composed)
    if layer.data_source is not None:
        return None
    return _frame_columns_after(chart, op, op.own + op.composed + op.statistics)


def refuse_missing_statistic_inputs(layers: list, chart: Any) -> None:
    """Refuse a per-layer statistic whose input or groupby column the layer's rows lack.

    Each layer's statistic is read off its effective encoding over *chart*'s
    (:func:`~ferrum._layer_transforms._effective_encoding`), so a statistic
    it inherits from its parent is checked over the rows it reads too.
    Checked where Python knows those rows (:func:`known_columns`). A missing
    column is voiced by the render gate's own message
    (``ferrum._core.layer_unknown_column_message``), naming a layer a figure
    or composite mark built by that call (``_Layer.entry``) as the gate
    does: the statistic remedy when it is another layer's statistic output
    or a column the chart's composite mark binds (one its layers reading a
    named output draw), else the plain message. Any other layer is left to
    the transform stage.
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
        columns = known_columns(layer, chart)
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


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def attach_to_rows(layers: list, chart: Any) -> list:
    """Point each layer attached to *chart* (``_Layer.chart_data``) at its operand's rows.

    On a chart of one operand whose statistics leave the tail where its own
    steps end, the layer reads that tail as it is. Otherwise it reads
    :data:`ROWS_KEY` inside its operand's chain, and takes each x/y/x2/y2 it
    does not bind from the bindings its chart's callers wrote
    (``positional_source="written"``).
    """
    from dataclasses import replace

    if not any(layer.chart_data for layer in layers):
        return layers
    at_tail = not chart._operands and not any(
        _advances_chain(step) for step in chart_operands(chart)[0].statistics
    )
    out = []
    for layer in layers:
        if layer.chart_data and at_tail:
            layer = replace(layer, chart_data=False)
        elif layer.chart_data:
            layer = replace(
                layer,
                chart_data=False,
                data_source=ROWS_KEY,
                reads_chart_data=True,
                positional_source=layer.positional_source or "written",
            )
        out.append(layer)
    return out


def _operand_chain(op: _Operand, layers: list, statistics: list) -> list:
    """*op*'s chain as its *layers* read it: ``[own][composed][rows][op.statistics][layer statistics]``.

    *statistics* holds each layer's ``_LayerStatistic`` (or ``None``):
    *op.statistics* (the operand's mark statistics, already its own
    lowering, or several) comes first, then every aggregate layer statistic
    opens its own lowering, then every bin layer statistic opens its own,
    so two layers computing independently each report their own null-key
    drop rather than sharing the mark's. The rows snapshot is published
    only when a layer, or a layer's statistic, reads it.
    """
    computed = [s for s in statistics if s is not None]
    lowered = [
        *op.statistics,
        *(step for s in computed if not s.bins for step in _with_statistics_marker([s.step])),
        *(step for s in computed if s.bins for step in _with_statistics_marker([s.step])),
    ]
    reads_rows = any(layer.data_source == ROWS_KEY for layer in layers) or any(
        isinstance(s.step, _Branch) and s.step.input == ROWS_KEY for s in computed
    )
    rows = [_source_identity(ROWS_KEY)] if reads_rows else []
    return [*op.own, *op.composed, *rows, *lowered]


def _chain_key(op: _Operand) -> str:
    """One string per distinct operand chain over one frame, so equal operands share a branch."""
    steps = [*op.own, *op.composed, None, *op.statistics]
    wire = ["|" if step is None else _transforms_to_json_list_named([step])[0] for step in steps]
    return json.dumps([op.frame, wire], sort_keys=True)


def lower_operands(chart: Any, layers: list, statistics: list) -> tuple[list, list]:
    """Lower *chart*'s operands to the pipeline its *layers* read.

    *layers* address keys relative to their operand's chain (``None`` its
    tail), and *statistics* holds each layer's encoding statistic
    (``_resolve_layer_statistics``). A chart of one operand runs its chain
    at the pipeline's top level and its layers keep their keys. Each
    operand of a composition runs as one branch, ``_operand_<k>``, over the
    composition's batch, its frame selecting its own rows when the batch
    holds several frames; each of its layers reads the branch's tail or a key
    scoped inside it (``secondary_key``). Operands over one frame with one
    chain share a branch, so a chart composed with itself computes once.
    An operand no layer reads is not lowered.

    Returns the layers with their final keys and the pipeline's steps.
    """
    from dataclasses import replace

    from ferrum._core import secondary_key

    operands = chart_operands(chart)
    if not chart._operands:
        return layers, _operand_chain(operands[0], layers, statistics)
    selectors = chart._frame_selectors
    branch_of: list[int] = []
    shared: dict[str, int] = {}
    for op in operands:
        branch_of.append(shared.setdefault(_chain_key(op), len(shared)))
    members: dict[int, list[int]] = {}
    for i, layer in enumerate(layers):
        members.setdefault(branch_of[layer.operand], []).append(i)
    out = list(layers)
    transforms: list = []
    for k in sorted(members):
        op = operands[branch_of.index(k)]
        indices = members[k]
        name = f"_operand_{k}"
        frame = selectors[op.frame] if selectors else None
        chain = _operand_chain(op, [layers[i] for i in indices], [statistics[i] for i in indices])
        if not chain and frame is None:
            chain = [dict(_PASS_THROUGH)]
        transforms.append(_Branch(name, None, chain, frame=frame))
        for i in indices:
            key = layers[i].data_source
            out[i] = replace(
                layers[i], data_source=name if key is None else secondary_key(name, key)
            )
    return out, transforms
