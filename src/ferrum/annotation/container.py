"""Annotate container — collects annotation primitives for attachment to a chart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Annotate:
    """A collection of annotation primitives to attach to a chart.

    Parameters
    ----------
    items : annotation primitive or list of annotation primitives
        A single primitive or a list of primitives created by the factory
        functions in `annotation`.

    Examples
    --------
    >>> import ferrum.annotation as ann
    >>> from ferrum.annotation import Annotate
    >>> annotations = Annotate([
    ...     ann.text(1.0, 2.0, "peak"),
    ...     ann.span("x", 0, 1, fill="#eee"),
    ... ])
    """

    items: list

    def __post_init__(self) -> None:
        # Accept a single item or a list; always store as a fresh copy.
        raw = self.items
        if not isinstance(raw, list):
            normalized: list = [raw]
        else:
            normalized = list(raw)
        object.__setattr__(self, "items", normalized)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialize all items to a list of dicts for renderer transport."""
        return [item.to_dict() for item in self.items]
