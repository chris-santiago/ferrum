"""Repeat — typed placeholder sentinels for RepeatChart templates.

Usage:
    from ferrum import Repeat
    Chart(data).mark_point().encode(x=Repeat.column, y=Repeat.row, color="species")

The placeholders are resolved by RepeatChart.expand() into concrete field names
based on the chart's `row=` / `column=` / `layer=` lists. JSON serialization
(via to_repeat_dict) emits `{"$repeat": "<axis>"}`.
"""
from __future__ import annotations
from typing import Final


class _RepeatPlaceholder:
    """Immutable sentinel naming a Repeat axis ('column' | 'row' | 'layer')."""
    __slots__ = ("_field",)

    def __init__(self, field: str) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ guard.
        object.__setattr__(self, "_field", field)

    @property
    def field(self) -> str:
        return self._field

    def to_repeat_dict(self) -> dict:
        return {"$repeat": self._field}

    def __repr__(self) -> str:
        return f"Repeat.{self._field}"

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(
            f"_RepeatPlaceholder is immutable; cannot set {name!r}"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, _RepeatPlaceholder):
            return NotImplemented
        return self._field == other._field

    def __hash__(self) -> int:
        return hash(("_RepeatPlaceholder", self._field))


class Repeat:
    """Namespace for typed RepeatChart template sentinels.

    Access the three sentinels via class attributes:

        Repeat.column   # cell's column-axis field
        Repeat.row      # cell's row-axis field
        Repeat.layer    # cell's layer-axis field
    """
    column: Final[_RepeatPlaceholder] = _RepeatPlaceholder("column")
    row:    Final[_RepeatPlaceholder] = _RepeatPlaceholder("row")
    layer:  Final[_RepeatPlaceholder] = _RepeatPlaceholder("layer")
