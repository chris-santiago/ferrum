"""Shared "not passed" sentinel for ``Axis`` and ``Legend`` value objects, plus
the two contracts built on top of it.

``_UNSET`` marks a dataclass field the caller never touched, distinguishable
at runtime from every value the field can legitimately hold (including
``None``). Two different fields shapes consume it, with two different
contracts — do not conflate them:

1. **The three-way ``title`` contract** (``serialize_title``): ``title`` has
   no "renderer default" to collide with, so its three states each mean
   something different. ``_UNSET`` (omitted) → do not emit the ``"title"``
   key; Rust falls back to the field name. ``None`` (explicit) → emit
   ``title: ""``; Rust treats an empty string as "suppress" (no title, no
   margin). ``"Foo"`` → emit ``title: "Foo"`` verbatim.

   Three surfaces share it, at two levels of one cascade: per-channel
   ``Axis.to_dict`` and ``Legend.to_dict``, and — since batch B task 8 gave
   the chart-level axis title a consumer — ``AxisConfig.to_dict``
   (``configure_axis(title=...)`` / ``configure(axis_x=AxisConfig(title=...))``).
   That third surface routes through this same function deliberately:
   ``AxisConfig(title=None)`` must mean what ``Axis(title=None)`` means, or
   the same kwarg name would carry opposite meanings at two levels of one
   cascade. Rust honors it through the matching single rule
   (``layout::axis::resolve_axis_title``), with per-channel winning over
   chart-level *including* when the per-channel value is the suppression.

2. **The two-way omit-vs-explicit contract** (``is_unspecified``), used by
   nine fields across the two classes whose Python default is a *concrete*
   value that matches the renderer's own default (``Axis``'s ``ticks``,
   ``tick_extra``, ``grid``, ``labels``, ``label_flush``, ``label_overlap``,
   ``domain``; ``Legend``'s ``orient``, ``direction``). A concrete default
   like ``ticks: bool = True`` cannot tell "the caller didn't pass this" apart
   from "the caller passed ``True``", so these fields default to ``_UNSET``
   instead (NF-B3, F-L04-04, 2026-09-02/03 batch B task 7). Two spellings
   count as "not specified" here, and must be treated identically everywhere
   the contract is enforced (construction-time token validation *and*
   ``to_dict()`` serialization):

   - ``_UNSET`` (the field's own default): the caller never touched the kwarg.
   - ``None`` (explicit): every *other* optional field on these two classes
     already accepts ``None`` to mean "unset" — these nine fields honor the
     same convention rather than singling themselves out as the one shape
     that raises on it.

   Any other value — including one textually equal to the renderer's own
   default (``Axis(ticks=True)``, ``Legend(direction="vertical")``) — is
   "specified": it is validated (where the field has a closed vocabulary)
   and always serialized, even though it looks like a no-op. That asymmetry
   (the concrete default value reaches the wire; ``_UNSET``/``None`` do not)
   is the entire point of the contract: it is what lets an explicit value
   beat a conflicting chart-level/theme fallback instead of being silently
   swallowed by it.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    pass


class _UnsetType:
    """Singleton sentinel class for the "field not passed" state."""

    _instance: "_UnsetType | None" = None

    def __new__(cls) -> "_UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_UNSET"


#: Sentinel value used as the default for every field carrying either
#: contract described in the module docstring above.
_UNSET: _UnsetType = _UnsetType()

#: Type alias for the three valid states of the ``title`` parameter.
TitleParam = Union[str, None, _UnsetType]


def serialize_title(title: TitleParam) -> str | None:
    """Convert a ``title`` value to the string to include in a serialized dict.

    Returns ``None`` when the key should be omitted entirely (sentinel ``_UNSET``).
    Returns ``""`` when the title should be suppressed (explicit ``None``).
    Returns the string verbatim otherwise.

    This is the single implementation of the three-way title contract shared
    between ``Axis.to_dict``, ``Legend.to_dict`` and ``AxisConfig.to_dict``
    (the chart-level axis title — see contract 1 in the module docstring). Do
    not reuse this for any other field: :func:`is_unspecified` is the two-way
    contract every other ``_UNSET``-defaulted field on these classes follows.
    """
    if title is _UNSET:
        return None  # omit key — Rust will use the field name
    if title is None:
        return ""  # explicit suppress — Rust treats "" as "no title"
    return title  # type: ignore[return-value]


def is_unspecified(value: Any) -> bool:
    """Return ``True`` if *value* counts as "not specified" under the
    two-way omit-vs-explicit contract (module docstring, contract 2).

    The single gate for every ``_UNSET``-defaulted field that is NOT
    ``title`` — ``Legend.orient``/``direction`` and ``Axis``'s seven
    concrete-default fields. A value is unspecified iff it is the sentinel
    itself (the caller never passed the kwarg) or explicit ``None`` (the
    caller passed it, meaning "unset", matching every sibling optional field
    on these two classes). Both the construction-time validator gate
    (``Legend.__post_init__``, ``LegendConfig.__post_init__``, the raw
    legend-dict path) and the ``to_dict()`` serialization gate
    (``Legend.to_dict``, ``Axis.to_dict``) call this one function, so a field
    cannot end up validated-but-not-serialized (or the reverse) and ``None``
    cannot mean "raise" on one surface and "unset" on a sibling surface.
    """
    return value is _UNSET or value is None
