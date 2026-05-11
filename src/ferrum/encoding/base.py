"""Base classes for encoding channels (internal)."""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from ferrum._warn import warn_once


def _scale_to_dict(scale: Any) -> Any:
    """Convert a Python Scale object to a JSON-serializable dict.

    Converts LogScale, LinearScale, TimeScale, SymlogScale, and OrdinalScale
    instances to the dict shape expected by Rust's ScaleSpec serde deserialiser.
    If ``scale`` is already a dict or ``None``, return it unchanged.
    """
    if scale is None or isinstance(scale, dict):
        return scale

    # Import here to avoid circular imports at module load time.
    try:
        from ferrum._core import (  # type: ignore[attr-defined]
            LogScale, LinearScale, TimeScale, SymlogScale, OrdinalScale,
        )
    except ImportError:
        return scale  # can't convert, pass through and let Rust raise

    if isinstance(scale, LogScale):
        d: dict = {"type": "log", "base": scale.base, "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, LinearScale):
        d = {"type": "linear", "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, TimeScale):
        d = {"type": "time", "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, SymlogScale):
        d = {"type": "symlog", "constant": scale.constant, "clamp": scale.clamp}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        if (p := scale.padding) is not None:
            d["padding"] = p
        return d
    if isinstance(scale, OrdinalScale):
        d = {"type": "ordinal", "padding": scale.padding}
        if scale.domain:
            d["domain"] = list(scale.domain)
        if scale.range:
            d["range"] = list(scale.range)
        return d

    # Unknown scale type — return as-is and let Rust surface the error.
    return scale


class ChannelBase:
    """Base class for all encoding-channel value objects.

    Subclasses set _channel_name, _renders_in_phase_8a, and _honored_kwargs.
    Constructor accepts a `field` positional arg + arbitrary keyword arguments;
    unknown kwargs trigger warn_once.
    """

    _channel_name: ClassVar[str] = "_unknown_"
    _renders_in_phase_8a: ClassVar[bool] = False
    _honored_kwargs: ClassVar[frozenset[str]] = frozenset(["type"])

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
        self.field = field
        self._kwargs = dict(kwargs)
        self._validate()

        for k in self._kwargs:
            if k not in self._honored_kwargs:
                warn_once(self._channel_name, k)

    def _validate(self) -> None:
        """Enforce kwarg-value constraints; subclasses may override."""
        type_ = self._kwargs.get("type")
        if type_ is not None and type_ not in ("Q", "N", "O", "T",
                                                 "quantitative", "nominal", "ordinal", "temporal"):
            raise ValueError(
                f"{self.__class__.__name__}(type={type_!r}): "
                f"expected one of Q, N, O, T, quantitative, nominal, ordinal, temporal"
            )

    def to_encoding_spec_dict(self) -> dict:
        """Return kwargs for the Rust EncodingSpec constructor / serde JSON."""
        out: dict = {"field": self.field}
        if (t := self._kwargs.get("type")) is not None:
            out["type_"] = t
        # scale: convert Python Scale objects → JSON-serializable dict so that
        # Rust's json_round helper (which calls json.dumps) can serialize them.
        if (v := self._kwargs.get("scale")) is not None:
            out["scale"] = _scale_to_dict(v)
        for k in ("title", "axis", "legend", "sort", "stack",
                  "impute", "scheme", "format"):
            if (v := self._kwargs.get(k)) is not None:
                out[k] = v
        # PyO3 EncodingSpec.__new__ expects snake_case param name `format_type`
        # even though the user-facing kwarg and JSON serde key is "formatType".
        if (v := self._kwargs.get("formatType")) is not None:
            out["format_type"] = v
        return out

    def to_implicit_transforms(self) -> list:
        """Return a list of transform objects derived from kwargs (bin, aggregate)."""
        out: list = []
        bin_arg = self._kwargs.get("bin")
        if bin_arg:
            from ferrum import Bin
            if isinstance(bin_arg, dict):
                out.append(Bin(self.field, **bin_arg))
            elif isinstance(bin_arg, bool):
                out.append(Bin(self.field))
            else:
                # Bin instance passed directly
                out.append(bin_arg)
        agg = self._kwargs.get("aggregate")
        if agg:
            from ferrum import Aggregate, AggregateOp
            out.append(Aggregate([AggregateOp(self.field or "", agg, f"{agg}_{self.field or 'all'}")]))
        return out

    def __repr__(self) -> str:
        """Return a string representation of this channel."""
        kw_parts = [f"{k}={v!r}" for k, v in self._kwargs.items()]
        body = ", ".join([repr(self.field)] + kw_parts)
        return f"{self.__class__.__name__}({body})"

    def __eq__(self, other: object) -> bool:
        """Return True if *other* is the same channel class, field, and kwargs."""
        if not isinstance(other, ChannelBase):
            return NotImplemented
        return (self.__class__ == other.__class__
                and self.field == other.field
                and self._kwargs == other._kwargs)

    def __hash__(self) -> int:
        """Return a hash based on class, field, and kwargs."""
        return hash((self.__class__, self.field,
                     tuple(sorted((k, repr(v)) for k, v in self._kwargs.items()))))
