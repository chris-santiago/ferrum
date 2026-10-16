"""Base classes for encoding channels (internal)."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from typing import Any, ClassVar, Literal

from ferrum._title_sentinel import is_unspecified
from ferrum._validate import validate_choice
from ferrum._warn import warn_once
from ferrum.axis import _normalize_axis
from ferrum.legend import Legend, _normalize_legend
from ferrum.encoding._scale import _scale_to_dict
from ferrum.position import STACK_OFFSETS, _validate_stack_offset

# Falsy stack spellings accepted by the encoding ``stack=`` path only.  These
# map to "no stacking" (Rust's ``Option<String>`` parser returns ``None``).
# ``transform_stack`` and ``position.Stack`` accept ONLY the real offsets in
# ``STACK_OFFSETS``; the encoding path layers these spellings on top.
_STACK_FALSY = frozenset({"false", "null", "none"})

# Accepted ``type=`` spellings → their canonical single-letter code.  The long
# (Vega-Lite) spellings are accepted for convenience but normalized to the short
# code immediately after validation so two channels that differ only in
# type-spelling are equal, hash-equal, and serialize identically (ENC-08).
_TYPE_NORMALIZE = {
    "Q": "Q",
    "N": "N",
    "O": "O",
    "T": "T",
    "quantitative": "Q",
    "nominal": "N",
    "ordinal": "O",
    "temporal": "T",
}


# Per-(channel, kwarg) message override for the honored-kwarg warn_once call
# in ChannelBase.__init__ (#240, 2026-09-11 wave-1 hardening). warn_once's
# generic default -- "accepted but not yet honored; planned for a future
# Phase" -- is true for most un-honored kwargs: no design exists yet and one
# might land later. It is a LIE for ``condition``: the kwarg is stripped from
# ``_honored_kwargs`` by ``__init_subclass__`` for a refused channel, so the
# generic loop below would otherwise describe a future Phase for a
# conditional ``Chart.encode()`` refuses outright and permanently.
def _honored_kwarg_message(channel: str, kwarg: str) -> str | None:
    """Return an explicit warn_once message for an un-honored (channel, kwarg), or None.

    ``None`` falls through to ``warn_once``'s generic "not yet honored"
    default. The one carve-out, ``condition`` on a channel
    :func:`~ferrum.encoding._channel_policy.channel_honors_condition`
    refuses, previews (without duplicating) the ``ValueError``
    ``conditional_wire_channel`` raises at ``Chart.encode()`` time.

    A kwarg that can *never* do anything on a channel -- as opposed to one
    that merely has no design yet -- is not routed through here: it is a
    REFUSE, not a WARN (``ChannelBase._refusal_message``, consulted first in
    ``ChannelBase.__init__``'s loop).
    """
    if kwarg == "condition":
        from ferrum.encoding._channel_policy import (
            _CONDITIONAL_CHANNELS_DISPLAY,
            _CONDITIONAL_REFUSALS,
        )

        if channel in _CONDITIONAL_REFUSALS:
            return (
                f"{channel}(condition=...) is accepted here but permanently unsupported: "
                f"a selection conditional can target only {_CONDITIONAL_CHANNELS_DISPLAY}. "
                "Building the chart raises before this value is used."
            )
    return None


# Refusal templates for a kwarg that can never do anything on a channel.
# Each names the public entry and the value received, filled in with
# `.format(channel=..., kwarg=..., value=...)` at construction.
_FACET_COLUMNS_MESSAGE = (
    "{channel}(columns={value!r}) is grid-mode-inapplicable and permanently refused: "
    "row/column faceting has no wrap-count to apply it to. Use Facet(columns=...) "
    "for wrap-mode panel-per-row counts."
)

_NO_TITLE_MESSAGE = (
    "{channel}(title={value!r}) is not honored: {channel} draws no legend, and "
    "title= is not supported on this channel."
)

_NO_AGGREGATE_MESSAGE = (
    "{channel}(aggregate={value!r}) is not supported: {channel} draws no aggregated "
    "value. Compute the aggregate with .transform(fm.transform_aggregate(...)) and "
    "bind its output column, or drop aggregate=."
)

_NO_LEGEND_MESSAGE = (
    "{channel}({kwarg}={value!r}) is not supported: {channel} draws no legend of "
    "its own, so there is nothing for {kwarg}= to customize."
)

# #176(11) + N3 (spec §6.4): a sort-field object's "field" must be a string
# naming a column. The string/array/channel-shorthand sort forms are
# unvalidated here -- only the object form is malformed-checkable at
# construction; the others still resolve (or warn) at render.
_SORT_FIELD_MESSAGE = (
    '{channel}({kwarg}={value!r}) is not a valid sort object: "field" must be a '
    "string naming a column; got {field!r}"
)


def _validate_sort_object(channel: str, value: Any, *, kwarg: str = "sort") -> None:
    """Refuse a malformed sort-field object (#176(11) + N3, spec §6.4).

    Only the object form ``{"field": ..., "op": ..., "order": ...}`` is
    checked here -- a bare string, an explicit list, or a channel-shorthand
    string (``"-y"``) pass through unvalidated and keep resolving (or
    warning) at render, as before. An object without a string ``field``, or
    with an ``op``/``order`` outside the accepted vocabulary, raises
    ``ValueError`` naming *channel* (the public entry, e.g. ``"Facet"`` or
    ``"Chart.facet"``), *kwarg* (the parameter that took the value) and the
    value received.

    An *absent* ``"op"``/``"order"`` key means "use the default" (Rust's
    ``parse_sort_field_object`` defaults ``op`` to ``sum`` and ``order`` to
    ``ascending``) and is not checked here; a key that is *present* with an
    explicit ``None`` is a malformed value, not an absence, and is checked
    like any other value -- ``validate_choice``, not
    ``validate_optional_choice``, whose "``None`` means unset" convention
    would otherwise let an explicit ``sort={"field": "v", "op": None}``
    construct silently and only refuse later, in ``EncodingSpec``'s voice,
    when Rust's own null refusal is finally reached.

    The op/order vocabulary is read from ``ferrum._core.sort_accepted_vocabulary()``
    (the ``scale_accepted_keys`` precedent) rather than mirrored here as a
    Python literal, and read lazily (inside this function, not at module
    import time) so importing ``ferrum.encoding.base`` never requires the
    compiled extension to already export it.
    """
    if not isinstance(value, dict):
        return
    field = value.get("field")
    if not isinstance(field, str):
        raise ValueError(
            _SORT_FIELD_MESSAGE.format(channel=channel, kwarg=kwarg, value=value, field=field)
        )
    from ferrum._core import sort_accepted_vocabulary

    vocab = sort_accepted_vocabulary()
    if "op" in value:
        validate_choice(f"{channel}({kwarg}=...)", "op", value["op"], vocab["op"])
    if "order" in value:
        validate_choice(f"{channel}({kwarg}=...)", "order", value["order"], vocab["order"])


@lru_cache(maxsize=None)
def _derive_honored(
    role: frozenset[str], legend_surface: str, honors_condition: bool
) -> frozenset[str]:
    """Return a channel class's honored-kwarg set with its derived bits applied.

    *role* is the class body's declared capabilities. Three bits are not
    declared there but derived: ``legend`` is honored iff the channel has a
    legend surface (``own`` or ``cascade``), a legend owner's ``title`` labels
    that legend, and ``condition`` follows the conditional vocabulary.
    Cached, so classes with equal inputs share one set object; returns *role*
    itself when nothing changes.
    """
    derived = set(role) - {"legend", "condition"}
    if legend_surface != "none":
        derived.add("legend")
    if legend_surface == "own":
        derived.add("title")
    if honors_condition:
        derived.add("condition")
    return role if derived == role else frozenset(derived)


def _aux_legend_channels() -> frozenset[str]:
    """Return the channel names whose legend is an auxiliary block (Size, Shape, StrokeDash).

    Read from Rust's ``ferrum._core.aux_legend_channels()`` -- the one list
    ``build_aux_legends`` (``render/prepare/legend.rs``) walks -- rather than
    mirrored by hand, so the two cannot drift (a hand-mirrored literal is
    guarded against by ``tests/test_transform_registry.py``). Used only
    to decide whether a legend value's ``orient=`` sub-key is refused (Q10):
    an aux legend has no independent screen position of its own, so it
    always shares the chart's legend block, placed by
    ``Color(legend=Legend(orient=))`` or ``configure_legend(orient=...)`` --
    ``orient="none"`` still disables the aux block itself.
    """
    from ferrum._core import aux_legend_channels

    return frozenset(aux_legend_channels())


_AUX_LEGEND_ORIENT_MESSAGE = (
    "{channel}(legend=...): orient={orient!r} is not supported on an "
    "auxiliary legend -- Size, Shape and StrokeDash legends always share "
    "the chart's legend block. Place it with Color(legend=Legend(orient=...)) "
    'or configure_legend(orient=...). orient="none" still disables this '
    "channel's own legend block."
)


def _legend_value_orient(value: Any) -> str | None:
    """Return the explicit ``orient=`` a legend kwarg value carries, or None.

    ``value`` is the raw ``legend=`` kwarg (already known not to be
    ``None``/``False`` by the caller): a :class:`~ferrum.legend.Legend`
    instance, a raw dict, or some other truthy value `_normalize_legend`
    treats as "not specified". Mirrors `Legend.to_dict`'s omit-vs-explicit
    contract via `~ferrum._title_sentinel.is_unspecified` so an omitted or
    explicit-``None`` ``orient`` reads identically here to how it reads
    there.
    """
    if isinstance(value, Legend):
        orient = value.orient
    elif isinstance(value, dict):
        orient = value.get("orient")
    else:
        return None
    return None if is_unspecified(orient) else orient


@dataclass(frozen=True)
class _PendingAggregate:
    """Sentinel emitted by ``to_implicit_transforms()`` for aggregate channels.

    Carries the aggregate operation's field, function name, and output column
    name but defers groupby assignment until ``chart.to_spec()`` can inspect
    all sibling encoding channels and infer which fields should be grouped by.

    This is an internal implementation detail — callers outside ``chart.py``
    must not construct or inspect this type directly.
    """

    field: str
    agg: str
    output_col: str


@dataclass(frozen=True)
class _PendingBin:
    """Sentinel emitted by ``to_implicit_transforms()`` for bin= channels.

    Carries the source field and either a pre-built ``Bin`` instance
    (``bin_obj``, when the user passed a ``Bin(...)`` directly) or optional
    Bin constructor kwargs (``bin_kwargs``, for ``bin=True`` / ``bin={...}``).

    The decision of whether to produce an unnamed chain-advancing transform
    (single-chart path) or a named fan-out transform (layered path) is deferred
    until ``chart.to_spec()`` knows the chart structure:

    - Single-chart: ``_resolve_pending_bins`` converts these to unnamed ``Bin``
      objects — byte-identical to the pre-FA4 behaviour.  When ``bin_obj`` is
      present, it is used directly (already a ``Bin``).  When ``bin_kwargs`` is
      set, a new ``Bin`` is constructed from ``field`` + those kwargs.
    - Layered: ``_resolve_layer_bins`` builds one named ``Bin`` per layer and
      points the layer's ``data_source`` at it.  When ``bin_obj`` is present,
      it is wrapped directly inside a ``_NamedTransform`` (the serialiser injects
      the name field); when ``bin_kwargs`` is set, a ``Bin`` is constructed with
      ``name=bin_name``.

    The PyO3 ``Bin`` class exposes no Python-readable attributes (no ``field``,
    no ``bin_count``, etc.).  The ``field`` stored here always comes from the
    encoding channel's ``self.field`` — never from introspecting the ``Bin``
    object.  ``bin_kwargs`` is always empty when ``bin_obj`` is not ``None``.

    This is an internal implementation detail — callers outside ``chart.py``
    must not construct or inspect this type directly.
    """

    field: str
    # Optional Bin constructor kwargs stored as a tuple of (key, value) pairs
    # so the dataclass stays hashable (frozen=True + dict would fail).
    bin_kwargs: tuple = ()
    # Pre-built Bin instance when the user passed bin=Bin(...) directly.
    # When set, bin_kwargs is always empty.  The PyO3 Bin type is not hashable,
    # so this field is excluded from the auto-generated __hash__ and __eq__
    # via hash=False, compare=False.  Equality and hashing use field + bin_kwargs
    # only; bin_obj identity is intentionally not compared.
    bin_obj: Any = dataclass_field(default=None, hash=False, compare=False)


# ---------------------------------------------------------------------------
# Spec-dict serialization (ENC-04: driven off the channel's own honored set)
# ---------------------------------------------------------------------------
#
# Each honored kwarg that maps to an EncodingSpec field has a handler here:
# ``handler(value, out)`` mutates the output dict.  ``to_encoding_spec_dict``
# iterates the channel's ``_honored_kwargs`` and dispatches through this table,
# so a kwarg is serialized iff the channel honors it.  ``bin`` and ``aggregate``
# are honored (for the warn guard) but intentionally absent here — they are
# consumed via ``to_implicit_transforms``, not the spec dict.


def _emit_type(value: Any, out: dict) -> None:
    # PyO3 EncodingSpec emits the data-type under "type_" (Python convention to
    # avoid shadowing the builtin); the wire form is "type".
    out["type_"] = value


def _emit_scale(value: Any, out: dict) -> None:
    # Convert Python Scale objects → JSON-serializable dict so Rust's json_round
    # helper (json.dumps) can serialize them.
    out["scale"] = _scale_to_dict(value)


def _emit_title(value: Any, out: dict) -> None:
    # title=None (explicitly passed) → suppress the axis/legend title by
    # forwarding an empty string to Rust, which renders a blank title instead of
    # falling back to the field name.  title omitted (key absent) keeps the
    # field-name default.  title="Foo" passes through as-is.
    out["title"] = "" if value is None else value


def _emit_axis(value: Any, out: dict) -> None:
    # Accept Axis instances, False (suppression), or raw dicts.
    normalized = _normalize_axis(value)
    if normalized is not None:
        out["axis"] = normalized


def _emit_legend(value: Any, out: dict) -> None:
    # For a channel that owns a legend surface (Color, Size, Shape, ...):
    # accept Legend instances, None/False (suppress THIS channel's legend),
    # or raw dicts.
    normalized = _normalize_legend(value)
    if normalized is not None:
        out["legend"] = normalized


def _emit_legend_cascade_only(value: Any, out: dict) -> None:
    """Emit ``legend=`` for a channel with no legend surface of its own (X, Y).

    X/Y contribute *style overrides* to the shared color-legend cascade
    (NF-B13) — precedence ``color`` > ``x`` > ``y`` — but render no legend
    block themselves. On Color/Size/Shape, bare ``None``/``False`` means
    "suppress THIS channel's own legend"; X/Y have no "this channel's
    legend" for that spelling to mean, so reusing :func:`_emit_legend`
    here made ``X(legend=None)`` reach across channels and suppress
    color's legend — the exact cross-channel leak the None-means-unspecified
    rule this batch codifies everywhere else was written to prevent.

    Bare ``None``/``False`` is therefore a no-op: nothing is serialized,
    exactly as if ``legend=`` had been omitted. An explicit ``Legend(...)``
    instance or a raw dict — including one that spells
    ``{"disabled": True}`` itself — is an unambiguous, typed ask and still
    routes through the normal cascade via :func:`_normalize_legend`.
    """
    if value is None or value is False:
        return
    normalized = _normalize_legend(value)
    if normalized is not None:
        out["legend"] = normalized


def _emit_format(value: Any, out: dict) -> None:
    # Resolve a named preset (or pass an already-raw d3-format/strftime spec
    # through unchanged) so a preset name never reaches Rust unresolved
    # (NF-B1). The derived format_type is provisional: an explicit
    # format_type/formatType kwarg — processed later in _SPEC_DICT_ORDER —
    # overwrites it (explicit-format-wins).
    if value is None:
        return
    from ferrum.format_presets import resolve_format_or_raw

    spec, format_type = resolve_format_or_raw(value)
    out["format"] = spec
    if format_type is not None:
        out["format_type"] = format_type


def _emit_format_type(value: Any, out: dict) -> None:
    # Both the camelCase "formatType" (Vega-Lite compat) and snake_case
    # "format_type" honored keys serialize to the snake_case EncodingSpec field.
    if value is not None:
        out["format_type"] = value


def _emit_passthrough(key: str):
    """Return a handler that copies a non-None kwarg verbatim under *key*."""

    def handler(value: Any, out: dict) -> None:
        if value is not None:
            out[key] = value

    return handler


# Canonical emission order, preserved from the previous hard-coded serializer so
# the resulting dict layout is byte-stable.  ``to_encoding_spec_dict`` walks this
# list and skips keys the channel does not honor.
#
# ``condition`` is deliberately absent (#237, 2026-09-20): a per-channel
# ``condition=`` kwarg is still honored -- ``ChannelBase.__init_subclass__``
# derives the honored-kwarg bit from the conditional vocabulary, so
# construction still accepts it without warning -- but it is never emitted
# here. Every reachable spelling of a conditional (``encode(channel=cond)``,
# ``encode(channel=Channel(field, condition=cond))``, and their ``Layer(...)``
# equivalents) is stripped from the channel object before serialization and
# routed instead through the chart-level ``ChartSpec.conditionals`` list --
# see ``chart.py``'s ``_append_conditional``/``_strip_layer_conditionals``.
# The per-channel wire key this used to feed (``EncodingSpec.condition`` in
# Rust) had no reader; keeping two mechanisms for one feature was the
# defect. The Rust field itself stays, tolerated and consumer-less, because
# ``EncodingSpec`` denies unknown fields and dropping the field would turn a
# spec saved before this fix into a parse error.
_SPEC_DICT_ORDER = (
    "type",
    "scale",
    "title",
    "axis",
    "legend",
    "sort",
    "stack",
    "impute",
    "scheme",
    "format",
    "format_type",
    "formatType",
)

_SPEC_DICT_HANDLERS = {
    "type": _emit_type,
    "scale": _emit_scale,
    "title": _emit_title,
    "axis": _emit_axis,
    "legend": _emit_legend,
    "sort": _emit_passthrough("sort"),
    "stack": _emit_passthrough("stack"),
    "impute": _emit_passthrough("impute"),
    "scheme": _emit_passthrough("scheme"),
    "format": _emit_format,
    "format_type": _emit_format_type,
    "formatType": _emit_format_type,
}


class ChannelBase:
    """Base class for all encoding-channel value objects.

    Subclasses set ``_channel_name`` and ``_honored_kwargs``. The constructor
    accepts a ``field`` positional arg plus arbitrary keyword arguments.

    Honored-kwarg contract
    ----------------------
    ``_honored_kwargs`` is the single, machine-readable source of truth for the
    kwargs a channel honors. A kwarg in this set is accepted silently and (if it
    has a serializer) forwarded to the EncodingSpec by
    `to_encoding_spec_dict`; a kwarg NOT in the set triggers a one-time
    ``warn_once`` and is dropped. Per-channel membership is assembled from the
    named role constants in `_honored`; do not narrate
    render status (``"honored"`` / ``"reserved"`` / ``"no-op today"``) in
    per-channel docstrings, as that prose drifts from this set.
    """

    _channel_name: ClassVar[str] = "_unknown_"
    _honored_kwargs: ClassVar[frozenset[str]] = frozenset(["type"])
    # Stacking channels (X, Y, Theta, Radius) set this True so the base
    # ``_validate`` normalizes their ``stack=`` kwarg.  Non-stacking channels
    # leave it False and the normalization is skipped.
    _stack_kwarg: ClassVar[bool] = False
    # This channel's legend surface, the one fact every legend and
    # legend-title rule derives from (`_derive_honored`, `_validate_legend_value`,
    # `_refusal_message`):
    #
    # - "own" (Color, Fill, Stroke, Size, Shape, StrokeDash): the channel
    #   renders a legend block of its own; `legend` and `title` are honored.
    #   A bare ``legend=None``/``False`` suppresses that block; a
    #   ``Legend(...)``/dict configures it (`_emit_legend`).
    # - "cascade" (X, Y): no legend block of its own; an explicit legend value
    #   contributes style overrides to the shared color-legend cascade, and a
    #   bare ``None``/``False`` is a no-op (`_emit_legend_cascade_only`).
    # - "none" (every other channel): no legend exists. ``legend=None``/``False``
    #   is the silent "no legend here" idiom and puts nothing on the wire; any
    #   other legend value, and a ``title=`` the class does not declare for
    #   another purpose, is refused.
    _legend_surface: ClassVar[Literal["own", "cascade", "none"]] = "none"
    # Kwargs this channel permanently refuses for a reason other than having
    # no legend, keyed to a message template (see `_FACET_COLUMNS_MESSAGE`).
    _refused_kwargs: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Apply the derived honored-kwarg bits once the class body has run.

        A subclass declares its role set without ``legend``, the legend's
        ``title`` or ``condition``: ``legend``/``title`` follow the class's
        ``_legend_surface`` and ``condition`` follows the conditional
        vocabulary (``_channel_policy.channel_honors_condition``), so neither
        fact is kept by hand beside its owner. See `_derive_honored`.
        """
        super().__init_subclass__(**kwargs)
        from ferrum.encoding._channel_policy import channel_honors_condition

        cls._honored_kwargs = _derive_honored(
            cls._honored_kwargs,
            cls._legend_surface,
            channel_honors_condition(cls._channel_name),
        )

    def __init__(self, field: Any = None, **kwargs: Any) -> None:
        # Phase 9: accept _RepeatPlaceholder as a sentinel value alongside str.
        # The placeholder rides through encoding verbatim; RepeatChart.expand()
        # replaces it with a concrete field name at expand time.
        from ferrum.repeat import _RepeatPlaceholder

        if field is not None and not isinstance(field, (str, _RepeatPlaceholder)):
            raise TypeError(
                f"{self.__class__.__name__}: field must be str, _RepeatPlaceholder, or None, "
                f"got {type(field).__name__}"
            )
        # Normalize type_ → type so fm.X("hp", type_="Q") and fm.X("hp", type="Q")
        # are identical. The trailing underscore form avoids shadowing the builtin
        # but both spellings must produce the same internal state.
        if "type_" in kwargs:
            kwargs = dict(kwargs)  # copy before mutation so we don't alias caller's state
            _type_val = kwargs.pop("type_")
            if "type" not in kwargs:
                kwargs["type"] = _type_val
            # else: "type" wins; type_ is discarded.
        # Parse shorthand from the field string so Channel("col:Q") works the same
        # as Channel("col", type_="Q"). This lets users pass shorthand strings
        # directly to channel constructors without manually splitting field and type.
        if isinstance(field, str) and ":" in field:
            from ferrum._shorthand import parse_shorthand

            try:
                parsed_field, parsed_type, _ = parse_shorthand(field)
                if parsed_field is not None:
                    field = parsed_field
                    if parsed_type is not None and "type" not in kwargs:
                        kwargs = {**kwargs, "type": parsed_type}
            except ValueError:
                pass  # non-shorthand field containing ":" — keep as-is
        self.field = field
        self._kwargs = dict(kwargs)
        self._validate()

        for k in self._kwargs:
            if k in self._honored_kwargs:
                continue
            refusal = self._refusal_message(k)
            if refusal is not None:
                raise ValueError(refusal)
            if k == "legend":
                # No legend surface: `_validate_legend_value` has already
                # refused every value except the silent None/False.
                continue
            warn_once(self._channel_name, k, _honored_kwarg_message(self._channel_name, k))

    def _refusal_message(self, kwarg: str) -> str | None:
        """Return the refusal for an un-honored *kwarg* that can never work here, or None."""
        template = self._refused_kwargs.get(kwarg)
        if template is None and kwarg == "title":
            template = _NO_TITLE_MESSAGE
        # A channel whose mark draws no aggregated value would still lower
        # its aggregate= into a step that rewrites the chart's rows, so it
        # is refused rather than warned about.
        if template is None and kwarg == "aggregate" and self._kwargs[kwarg] is not None:
            template = _NO_AGGREGATE_MESSAGE
        if template is None:
            return None
        return template.format(
            channel=self.__class__.__name__, kwarg=kwarg, value=self._kwargs[kwarg]
        )

    def _validate(self) -> None:
        """Enforce kwarg-value constraints; subclasses may override."""
        type_ = self._kwargs.get("type")
        if type_ is not None:
            validate_choice(self.__class__.__name__, "type", type_, _TYPE_NORMALIZE)
            # Normalize the long spellings to their single-letter codes so a
            # channel built with ``type_="quantitative"`` compares, hashes, and
            # serializes identically to one built with ``type_="Q"`` (ENC-08).
            self._kwargs["type"] = _TYPE_NORMALIZE[type_]
        scheme = self._kwargs.get("scheme")
        if scheme is not None and "scheme" in self._honored_kwargs:
            # Validate the named color set at declaration time against the single
            # palette registry (D-COLOR-1).  A bogus scheme name fails here with
            # the canonical ``palette()`` message shape rather than at render.
            from ferrum.color import _validate_scheme

            _validate_scheme(scheme, where=f"{self.__class__.__name__}(scheme=...)")
        if self._stack_kwarg:
            stack = self._kwargs.get("stack")
            if stack is not None:
                self._kwargs["stack"] = self._normalize_stack(stack)
        sort = self._kwargs.get("sort")
        if sort is not None and "sort" in self._honored_kwargs:
            _validate_sort_object(self.__class__.__name__, sort)
        self._validate_legend_value()

    def _explicit_scale_range(self) -> Any:
        """Return the ``range`` of an explicit ``scale=`` kwarg, or ``None`` when it sets none."""
        scale = self._kwargs.get("scale")
        if scale is None:
            return None
        spec = _scale_to_dict(scale)
        return spec.get("range") if isinstance(spec, dict) else None

    def _validate_legend_value(self) -> None:
        """Refuse a ``legend=`` value this channel's legend surface cannot honor (#123).

        Value-dependent, so it runs for every channel rather than through the
        honored-kwarg gate in ``__init__``.

        - ``None``/``False``: always silent, on every legend surface.
        - ``_legend_surface == "none"``: any other value is refused -- there
          is no legend for a dict/``Legend(...)`` to configure.
        - An auxiliary-legend channel (`_aux_legend_channels`: Size, Shape,
          StrokeDash) whose legend value carries an explicit ``orient=``
          other than ``"none"`` is refused: an aux legend has no independent
          screen position, so it always shares the chart's legend block.
          ``orient="none"`` still disables the aux block itself.
        """
        if "legend" not in self._kwargs:
            return
        value = self._kwargs["legend"]
        if value is None or value is False:
            return
        if self._legend_surface == "none":
            raise ValueError(
                _NO_LEGEND_MESSAGE.format(
                    channel=self.__class__.__name__, kwarg="legend", value=value
                )
            )
        if self._channel_name in _aux_legend_channels():
            orient = _legend_value_orient(value)
            if orient is not None and orient != "none":
                raise ValueError(
                    _AUX_LEGEND_ORIENT_MESSAGE.format(
                        channel=self.__class__.__name__, orient=orient
                    )
                )

    def _normalize_stack(self, value: Any) -> str:
        """Normalize a ``stack=`` kwarg value to a string for Rust's ``Option<String>``.

        Rust's ``EncodingSpec.stack`` is ``Option<String>``; passing a Python
        bool crashes PyO3 with ``TypeError: argument 'stack': 'bool' object is
        not an instance of 'str'``.  This helper converts booleans, accepts the
        falsy spellings (``"false"``/``"null"``/``"none"`` → no stacking), and
        validates real offsets against the single canonical ``STACK_OFFSETS``
        set via :func:`~ferrum.position._validate_stack_offset`.

        Parameters
        ----------
        value :
            The raw ``stack=`` argument supplied by the caller.

        Returns
        -------
        str
            ``"zero"`` for ``True``; ``"false"`` for ``False``; the original
            string when it is a recognized stack strategy or falsy spelling.

        Raises
        ------
        TypeError
            When *value* is neither a bool nor a string.
        ValueError
            When *value* is a string that is not a recognized stack strategy.
        """
        channel = self.__class__.__name__
        if value is True:
            return "zero"
        if value is False:
            return "false"
        if not isinstance(value, str):
            raise TypeError(
                f"{channel}(stack={value!r}): stack= must be a bool or one of "
                f"{sorted(STACK_OFFSETS)} or a falsy value {sorted(_STACK_FALSY)}; "
                f"got {type(value).__name__!r}"
            )
        lowered = value.lower()
        if lowered in _STACK_FALSY:
            return value
        if lowered not in STACK_OFFSETS:
            # Surface the user's original spelling, but route through the single
            # canonical validator so all three entry points fail identically.
            _validate_stack_offset(value, func_name=channel, param="stack")
        return value

    def to_encoding_spec_dict(self) -> dict:
        """Return kwargs for the Rust EncodingSpec constructor / serde JSON.

        Iterates this channel's own ``_honored_kwargs`` (the single source of
        truth) rather than a separate hard-coded key list, so a kwarg is
        serialized iff the channel honors it.  Each honored key routes through
        `_SPEC_DICT_HANDLERS` to its serializer; keys with no handler
        (``bin``/``aggregate``) are honored for the warn guard but consumed via
        `to_implicit_transforms`, so they contribute nothing here.  ``legend``
        is the one key whose serializer depends on more than its own value,
        branching on `_legend_surface`: ``"own"`` routes through
        `_emit_legend`; ``"cascade"`` (X, Y) routes through
        `_emit_legend_cascade_only` instead, so a bare ``None``/``False``
        cannot suppress a different channel's legend (see that function's
        docstring); on ``"none"`` ``legend`` is not honored, so nothing is
        emitted -- `_validate_legend_value` has already refused every value
        except the silent ``None``/``False``.

        Keys are emitted in `_SPEC_DICT_ORDER` so the dict layout is
        stable regardless of ``frozenset`` iteration order.
        """
        out: dict = {"field": self.field}
        for key in _SPEC_DICT_ORDER:
            if key not in self._honored_kwargs or key not in self._kwargs:
                continue
            if key == "legend" and self._legend_surface == "cascade":
                _emit_legend_cascade_only(self._kwargs[key], out)
                continue
            handler = _SPEC_DICT_HANDLERS.get(key)
            if handler is None:
                continue
            handler(self._kwargs[key], out)
        return out

    def to_implicit_transforms(self) -> list:
        """Return a list of transform objects derived from kwargs (bin, aggregate).

        Both bin and aggregate transforms are returned as pending sentinels
        rather than concrete Rust objects.  ``chart.to_spec()`` resolves them:

        - ``_PendingBin``: single-chart path converts to an unnamed ``Bin``
          (byte-stable); layered path converts to a named ``Bin`` fan-out so
          it does not corrupt the shared unnamed chain.
        - ``_PendingAggregate``: defers groupby assignment until all sibling
          encoding channels are known (Altair-style auto-groupby).
        """
        out: list = []
        bin_arg = self._kwargs.get("bin")
        if bin_arg:
            if isinstance(bin_arg, dict):
                out.append(_PendingBin(field=self.field, bin_kwargs=tuple(bin_arg.items())))
            elif isinstance(bin_arg, bool):
                out.append(_PendingBin(field=self.field))
            else:
                # Bin instance passed directly.  The PyO3 Bin object exposes no
                # Python-readable attributes (no .field, .bin_count, etc.), so
                # we cannot extract kwargs from it.  Store the instance in
                # bin_obj alongside the channel's own field name so resolvers
                # can use the pre-built Bin directly without guessing internals.
                out.append(_PendingBin(field=self.field, bin_obj=bin_arg))
        agg = self._kwargs.get("aggregate")
        if agg:
            out.append(
                _PendingAggregate(
                    field=self.field or "",
                    agg=agg,
                    output_col=f"{agg}_{self.field or 'all'}",
                )
            )
        return out

    def _with_options(self, **options: Any) -> "ChannelBase":
        """Return a copy of this channel, same class and field, with *options* set over its own."""
        return type(self)(self.field, **{**self._kwargs, **options})

    def option(self, name: str, default: Any = None) -> Any:
        """Return the value of encoding option *name*, or *default* if unset.

        Parameters
        ----------
        name : str
            The option key to look up (e.g. ``"sort"``, ``"axis"``).
        default : Any, optional
            Value returned when *name* is not present.  Defaults to ``None``.

        Returns
        -------
        Any
            The option value, or *default* when the key is absent.
        """
        return self._kwargs.get(name, default)

    def __repr__(self) -> str:
        """Return a string representation of this channel."""
        kw_parts = [f"{k}={v!r}" for k, v in self._kwargs.items()]
        body = ", ".join([repr(self.field)] + kw_parts)
        return f"{self.__class__.__name__}({body})"

    def __eq__(self, other: object) -> bool:
        """Return True if *other* is the same channel class, field, and kwargs."""
        if not isinstance(other, ChannelBase):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.field == other.field
            and self._kwargs == other._kwargs
        )

    def __hash__(self) -> int:
        """Return a hash based on class, field, and kwargs."""
        return hash(
            (
                self.__class__,
                self.field,
                tuple(sorted((k, repr(v)) for k, v in self._kwargs.items())),
            )
        )
