"""Base classes for encoding channels (internal)."""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from ferrum._warn import warn_once
from ferrum.encoding._scale import _scale_to_dict


class ChannelBase:
    """Base class for all encoding-channel value objects.

    Subclasses set _channel_name and _honored_kwargs.
    Constructor accepts a `field` positional arg + arbitrary keyword arguments;
    unknown kwargs trigger warn_once.
    """

    _channel_name: ClassVar[str] = "_unknown_"
    _honored_kwargs: ClassVar[frozenset[str]] = frozenset(["type", "condition"])

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
        if type_ is not None and type_ not in (
            "Q",
            "N",
            "O",
            "T",
            "quantitative",
            "nominal",
            "ordinal",
            "temporal",
        ):
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
        for k in ("title", "axis", "legend", "sort", "stack", "impute", "scheme", "format"):
            if k == "legend" and "legend" in self._kwargs:
                # Schwabish SB3 (2026-05-11): distinguish "legend not
                # specified" from "legend explicitly suppressed".
                # ``legend=None`` / ``legend=False`` from the Color encoding
                # signals "hide the legend"; serialize as a dict with the
                # ``disabled: true`` flag the Rust renderer recognizes.
                v = self._kwargs["legend"]
                if v is None or v is False:
                    out["legend"] = {"disabled": True}
                elif isinstance(v, dict):
                    out["legend"] = v
                # Other truthy values are reserved; drop silently.
                continue
            if (v := self._kwargs.get(k)) is not None:
                out[k] = v
        # PyO3 EncodingSpec.__new__ expects snake_case param name `format_type`.
        # Accept both "formatType" (camelCase, Vega-Lite compat) and "format_type"
        # (snake_case, Python idiomatic) as aliases.
        if (v := self._kwargs.get("formatType")) is not None:
            out["format_type"] = v
        elif (v := self._kwargs.get("format_type")) is not None:
            out["format_type"] = v
        # condition: serialize ConditionalSpec or raw dict as opaque JSON for Rust.
        if (cond := self._kwargs.get("condition")) is not None:
            if hasattr(cond, "to_spec_dict"):
                out["condition"] = cond.to_spec_dict()
            elif isinstance(cond, dict):
                out["condition"] = cond
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

            out.append(
                Aggregate([AggregateOp(self.field or "", agg, f"{agg}_{self.field or 'all'}")])
            )
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
