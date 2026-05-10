"""Layer value class — used internally by Chart.__add__."""
from __future__ import annotations


class Layer:
    """Internal layer wrapper; users construct Layer rarely.

    Most multi-layer charts are built via ``chart1 + chart2``.
    """

    def __init__(self, data=None, mark=None, *, encoding=None, transforms=None):
        self.data = data
        self.mark = mark
        self.encoding = encoding or {}
        self.transforms = transforms or []

    def __repr__(self) -> str:
        return f"Layer(mark={self.mark!r})"
