"""Typed sentinel namespace for RepeatChart templates.

Usage::

    from ferrum import Repeat
    Chart(data).encode(x=Repeat.column, y=Repeat.row, color="species").mark_point()

The placeholders are resolved by ``RepeatChart.expand()`` into concrete field
names based on the chart's ``row=`` / ``column=`` / ``layer=`` lists.  JSON
serialisation (via ``to_repeat_dict``) emits ``{"$repeat": "<axis>"}``.
"""
from __future__ import annotations
from typing import Final


class _RepeatPlaceholder:
    """Immutable sentinel naming a Repeat axis (``'column'`` | ``'row'`` | ``'layer'``)."""

    __slots__ = ("_field",)

    def __init__(self, field: str) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ guard.
        object.__setattr__(self, "_field", field)

    @property
    def field(self) -> str:
        """Str : The axis name this placeholder represents."""
        return self._field

    def to_repeat_dict(self) -> dict:
        """Serialise to the ``{"$repeat": "<axis>"}`` wire format.

        Returns
        -------
        dict
            A dict with a single ``"$repeat"`` key whose value is the axis
            name (``"column"``, ``"row"``, or ``"layer"``).
        """
        return {"$repeat": self._field}

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"Repeat.{self._field}"

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(
            f"_RepeatPlaceholder is immutable; cannot set {name!r}"
        )

    def __eq__(self, other) -> bool:
        """Return True when *other* is the same placeholder axis."""
        if not isinstance(other, _RepeatPlaceholder):
            return NotImplemented
        return self._field == other._field

    def __hash__(self) -> int:
        """Return a stable hash based on the axis name."""
        return hash(("_RepeatPlaceholder", self._field))


class Repeat:
    """Namespace for typed ``RepeatChart`` template sentinels.

    Access the three sentinels via class attributes — do not instantiate
    ``Repeat`` directly:

    Attributes
    ----------
    column : _RepeatPlaceholder
        Sentinel that resolves to the column-axis field for each cell.
    row : _RepeatPlaceholder
        Sentinel that resolves to the row-axis field for each cell.
    layer : _RepeatPlaceholder
        Sentinel that resolves to the layer-axis field for each cell.

    Examples
    --------
    >>> import ferrum as fm
    >>> base = (
    ...     fm.Chart(df)
    ...     .encode(x=fm.Repeat.column, y=fm.Repeat.row)
    ...     .mark_point()
    ... )
    >>> grid = fm.RepeatChart(base, row=["mpg", "hp"], column=["mpg", "hp"])
    """

    column: Final[_RepeatPlaceholder] = _RepeatPlaceholder("column")
    row:    Final[_RepeatPlaceholder] = _RepeatPlaceholder("row")
    layer:  Final[_RepeatPlaceholder] = _RepeatPlaceholder("layer")
