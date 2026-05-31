"""Base classes for encoding channels (internal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ferrum._warn import warn_once
from ferrum.axis import _normalize_axis
from ferrum.legend import _normalize_legend
from ferrum.encoding._scale import _scale_to_dict


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
            if k == "axis" and "axis" in self._kwargs:
                # Accept Axis instances, False (suppression), or raw dicts.
                normalized = _normalize_axis(self._kwargs["axis"])
                if normalized is not None:
                    out["axis"] = normalized
                continue
            if k == "legend" and "legend" in self._kwargs:
                # Accept Legend instances, None/False (suppression), or raw dicts.
                normalized = _normalize_legend(self._kwargs["legend"])
                if normalized is not None:
                    out["legend"] = normalized
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
        """Return a list of transform objects derived from kwargs (bin, aggregate).

        Aggregate transforms are returned as ``_PendingAggregate`` sentinels
        rather than Rust ``Aggregate`` objects.  The ``chart.to_spec()`` method
        resolves them into concrete ``Aggregate`` objects once all sibling
        encoding channels are known, allowing it to infer the correct groupby
        from non-aggregate fields (Altair-style auto-groupby).
        """
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
