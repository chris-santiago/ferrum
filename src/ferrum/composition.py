"""Composition wrappers: HConcatChart, VConcatChart."""
from __future__ import annotations

from typing import List


class _CompositeBase:
    """Base for HConcat/VConcat. Holds a list of children + spacing."""

    def __init__(self, charts: List, *, spacing: float = 10.0) -> None:
        self.charts = list(charts)
        self.spacing = spacing

    def __or__(self, other):
        return HConcatChart([self, other])

    def __and__(self, other):
        return VConcatChart([self, other])


class HConcatChart(_CompositeBase):
    """Horizontal concatenation of two or more charts."""

    def show_svg(self) -> str:
        from ferrum._core import compose_svg_horizontal
        svgs = [c.show_svg() for c in self.charts]
        return compose_svg_horizontal(svgs, spacing=self.spacing, align="top")

    def show_png(self) -> bytes:
        raise NotImplementedError(
            "HConcatChart.show_png not yet wired in Phase 8a; "
            "use .save('out.svg') instead (Phase 8a follow-up)."
        )

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(
                f"HConcatChart.save({fmt!r}) not yet supported in Phase 8a"
            )

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return f"HConcatChart([{', '.join(repr(c) for c in self.charts)}])"


class VConcatChart(_CompositeBase):
    """Vertical concatenation of two or more charts."""

    def show_svg(self) -> str:
        from ferrum._core import compose_svg_vertical
        svgs = [c.show_svg() for c in self.charts]
        return compose_svg_vertical(svgs, spacing=self.spacing, align="left")

    def show_png(self) -> bytes:
        raise NotImplementedError(
            "VConcatChart.show_png not yet wired in Phase 8a; "
            "use .save('out.svg') instead."
        )

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(
                f"VConcatChart.save({fmt!r}) not yet supported in Phase 8a"
            )

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return f"VConcatChart([{', '.join(repr(c) for c in self.charts)}])"
