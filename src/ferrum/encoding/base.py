"""Base classes for encoding channels (internal)."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, ClassVar

from ferrum._warn import warn_once
from ferrum.axis import _normalize_axis
from ferrum.legend import _normalize_legend
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
    # Accept Legend instances, None/False (suppression), or raw dicts.
    normalized = _normalize_legend(value)
    if normalized is not None:
        out["legend"] = normalized


def _emit_format_type(value: Any, out: dict) -> None:
    # Both the camelCase "formatType" (Vega-Lite compat) and snake_case
    # "format_type" honored keys serialize to the snake_case EncodingSpec field.
    if value is not None:
        out["format_type"] = value


def _emit_condition(value: Any, out: dict) -> None:
    # Serialize ConditionalSpec or raw dict as opaque JSON for Rust.
    if hasattr(value, "to_spec_dict"):
        out["condition"] = value.to_spec_dict()
    elif isinstance(value, dict):
        out["condition"] = value


def _emit_passthrough(key: str):
    """Return a handler that copies a non-None kwarg verbatim under *key*."""

    def handler(value: Any, out: dict) -> None:
        if value is not None:
            out[key] = value

    return handler


# Canonical emission order, preserved from the previous hard-coded serializer so
# the resulting dict layout is byte-stable.  ``to_encoding_spec_dict`` walks this
# list and skips keys the channel does not honor.
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
    "condition",
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
    "format": _emit_passthrough("format"),
    "format_type": _emit_format_type,
    "formatType": _emit_format_type,
    "condition": _emit_condition,
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
    :meth:`to_encoding_spec_dict`; a kwarg NOT in the set triggers a one-time
    ``warn_once`` and is dropped. Per-channel membership is assembled from the
    named role constants in :mod:`ferrum.encoding._honored`; do not narrate
    render status (``"honored"`` / ``"reserved"`` / ``"no-op today"``) in
    per-channel docstrings, as that prose drifts from this set.
    """

    _channel_name: ClassVar[str] = "_unknown_"
    _honored_kwargs: ClassVar[frozenset[str]] = frozenset(["type", "condition"])
    # Stacking channels (X, Y, Theta, Radius) set this True so the base
    # ``_validate`` normalizes their ``stack=`` kwarg.  Non-stacking channels
    # leave it False and the normalization is skipped.
    _stack_kwarg: ClassVar[bool] = False

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
            if k not in self._honored_kwargs:
                warn_once(self._channel_name, k)

    def _validate(self) -> None:
        """Enforce kwarg-value constraints; subclasses may override."""
        type_ = self._kwargs.get("type")
        if type_ is not None:
            if type_ not in _TYPE_NORMALIZE:
                raise ValueError(
                    f"{self.__class__.__name__}(type={type_!r}): "
                    f"expected one of Q, N, O, T, quantitative, nominal, ordinal, temporal"
                )
            # Normalize the long spellings to their single-letter codes so a
            # channel built with ``type_="quantitative"`` compares, hashes, and
            # serializes identically to one built with ``type_="Q"`` (ENC-08).
            self._kwargs["type"] = _TYPE_NORMALIZE[type_]
        if self._stack_kwarg:
            stack = self._kwargs.get("stack")
            if stack is not None:
                self._kwargs["stack"] = self._normalize_stack(stack)

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
            _validate_stack_offset(value, where=channel)
        return value

    def to_encoding_spec_dict(self) -> dict:
        """Return kwargs for the Rust EncodingSpec constructor / serde JSON.

        Iterates this channel's own ``_honored_kwargs`` (the single source of
        truth) rather than a separate hard-coded key list, so a kwarg is
        serialized iff the channel honors it.  Each honored key routes through
        :data:`_SPEC_DICT_HANDLERS` to its serializer; keys with no handler
        (``bin``/``aggregate``) are honored for the warn guard but consumed via
        :meth:`to_implicit_transforms`, so they contribute nothing here.

        Keys are emitted in :data:`_SPEC_DICT_ORDER` so the dict layout is
        stable regardless of ``frozenset`` iteration order.
        """
        out: dict = {"field": self.field}
        for key in _SPEC_DICT_ORDER:
            if key not in self._honored_kwargs or key not in self._kwargs:
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
