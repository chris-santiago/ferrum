"""Composition wrappers: HConcatChart, VConcatChart, JointChart, RepeatChart, ClusterMapChart."""
from __future__ import annotations

from typing import List, Optional


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


# --------------------------------------------------------------------------
# Phase 9 compound views: JointChart, RepeatChart, ClusterMapChart
# --------------------------------------------------------------------------


def _embed_chart_spec(c) -> Optional[dict]:
    """Helper: convert a Chart's .to_spec() output to an embedded JSON dict."""
    import json as _json
    if c is None or not hasattr(c, "to_spec"):
        return None
    return _json.loads(c.to_spec().to_json())


class JointChart:
    """Joint distribution view: center + optional top / right marginal Charts.

    Layout: 2x2 grid where bottom-left = center, top-left = top, bottom-right = right,
    top-right = empty. Cell sizing: marginal cells get 1/(ratio+1), center gets
    ratio/(ratio+1). x-axis shared across (center, top); y across (center, right).
    """
    __slots__ = ("center", "top", "right", "ratio", "spacing", "_theme")

    def __init__(
        self,
        center,
        *,
        top=None,
        right=None,
        ratio: int = 5,
        spacing: float = 0.02,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0; got {ratio}")
        self.center = center
        self.top = top
        self.right = right
        self.ratio = ratio
        self.spacing = spacing
        self._theme = None

    @property
    def charts(self) -> list:
        return [c for c in (self.center, self.top, self.right) if c is not None]

    @property
    def spec(self) -> dict:
        share_x = ["center"]
        if self.top is not None:
            share_x.append("top")
        share_y = ["center"]
        if self.right is not None:
            share_y.append("right")
        return {
            "kind": "joint",
            "center": _embed_chart_spec(self.center),
            "top": _embed_chart_spec(self.top),
            "right": _embed_chart_spec(self.right),
            "ratio": self.ratio,
            "spacing": self.spacing,
            "share": {"x": share_x, "y": share_y},
        }

    def theme(self, t):
        new = JointChart(
            self.center.theme(t),
            top=(self.top.theme(t) if self.top is not None else None),
            right=(self.right.theme(t) if self.right is not None else None),
            ratio=self.ratio,
            spacing=self.spacing,
        )
        new._theme = t
        return new

    def properties(self, **kwargs):
        new = JointChart(
            self.center.properties(**kwargs),
            top=self.top, right=self.right,
            ratio=self.ratio, spacing=self.spacing,
        )
        new._theme = self._theme
        return new

    def show_svg(self) -> str:
        try:
            from ferrum._core import compose_svg_grid  # type: ignore[attr-defined]
        except ImportError as e:
            raise NotImplementedError(
                "JointChart.show_svg() requires compose_svg_grid; "
                "wire-up lands in Phase 9a Task 12"
            ) from e
        center_svg = self.center.show_svg()
        cells = [None, self.top.show_svg() if self.top is not None else None,
                 center_svg, self.right.show_svg() if self.right is not None else None]
        marginal_share = 1.0 / (self.ratio + 1)
        center_share = self.ratio / (self.ratio + 1)
        return compose_svg_grid(
            cells, rows=2, cols=2,
            row_ratios=[marginal_share, center_share],
            col_ratios=[center_share, marginal_share],
            spacing=self.spacing,
            share_x=[[2, 0]],
            share_y=[[2, 3]],
        )

    def show_png(self) -> bytes:
        raise NotImplementedError("JointChart.show_png — Phase 9 follow-up")

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"JointChart.save({fmt!r}) not yet supported in Phase 9")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return (
            f"JointChart(center={self.center!r}, top={self.top!r}, "
            f"right={self.right!r}, ratio={self.ratio})"
        )


class RepeatChart:
    """Repeat a template chart over a grid of row/column field combinations.

    Use `Repeat.column` / `Repeat.row` / `Repeat.layer` typed sentinels in the
    template's `.encode(...)` call to mark which encoding channel gets the
    per-cell field substitution. `RepeatChart.expand()` returns a list of
    `(row_field, col_field, Chart)` tuples — fully resolved Charts.

    `diagonal=` provides an alternate template for cells where row_field ==
    col_field (n×n symmetric repeat). `corner=True` filters the expanded grid
    to the lower triangle (including diagonal).
    """
    __slots__ = (
        "template", "row", "column", "layer", "diagonal", "corner",
        "spacing", "columns", "resolve", "_theme",
    )

    def __init__(
        self,
        template,
        *,
        row=None, column=None, layer=None,
        diagonal=None,
        corner: bool = False,
        spacing: float = 0.02,
        columns: Optional[int] = None,
        resolve=None,
    ) -> None:
        if diagonal is not None and (row is None or column is None):
            raise ValueError(
                "RepeatChart: diagonal= requires both row= and column= to be set"
            )
        self.template = template
        self.row = list(row) if row is not None else None
        self.column = list(column) if column is not None else None
        self.layer = list(layer) if layer is not None else None
        self.diagonal = diagonal
        self.corner = corner
        self.spacing = spacing
        self.columns = columns
        self.resolve = resolve
        self._theme = None

    @property
    def spec(self) -> dict:
        return {
            "kind": "repeat",
            "template": _embed_chart_spec(self.template),
            "row": self.row,
            "column": self.column,
            "layer": self.layer,
            "diagonal": _embed_chart_spec(self.diagonal),
            "corner": self.corner,
            "columns": self.columns,
            "resolve": self.resolve,
            "spacing": self.spacing,
        }

    def expand(self) -> list:
        """Materialize the template into a list of (row_field, col_field, Chart) tuples."""
        import warnings

        rows = self.row or []
        cols = self.column or []

        asymmetric = (
            self.diagonal is not None
            and self.row is not None
            and self.column is not None
            and self.row != self.column
        )
        if asymmetric:
            warnings.warn(
                "RepeatChart: diagonal= ignored because row != column (asymmetric repeat).",
                UserWarning, stacklevel=2,
            )

        out = []
        use_diagonal_match = (
            self.diagonal is not None and not asymmetric
            and len(rows) == len(cols)
        )

        for ri, row_field in enumerate(rows):
            for ci, col_field in enumerate(cols):
                if self.corner and ri < ci:
                    continue
                source = self.template
                if use_diagonal_match and row_field == col_field:
                    source = self.diagonal
                cell = self._resolve_template(source, row_field, col_field)
                out.append((row_field, col_field, cell))
        return out

    def _resolve_template(self, source, row_field: str, col_field: str):
        """Clone source (a Chart) and substitute Repeat placeholders in encoding."""
        from ferrum.repeat import _RepeatPlaceholder
        from ferrum.encoding.base import ChannelBase

        new = source._clone()
        for axis, ch in list(new._encoding.items()):
            if isinstance(ch, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field, row_field, col_field)
                from ferrum.chart import _channel_class_for
                cls = _channel_class_for(axis) or _channel_class_for("x")
                new._encoding[axis] = cls(concrete)
            elif isinstance(ch, ChannelBase) and isinstance(ch.field, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field.field, row_field, col_field)
                new._encoding[axis] = ch.__class__(concrete)
        return new

    @staticmethod
    def _concrete_field(placeholder_axis: str, row_field: str, col_field: str) -> str:
        if placeholder_axis == "column":
            return col_field
        if placeholder_axis == "row":
            return row_field
        if placeholder_axis == "layer":
            return row_field
        raise ValueError(f"unknown Repeat placeholder axis '{placeholder_axis}'")

    def theme(self, t):
        new = RepeatChart(
            self.template.theme(t),
            row=self.row, column=self.column, layer=self.layer,
            diagonal=(self.diagonal.theme(t) if self.diagonal is not None else None),
            corner=self.corner, spacing=self.spacing,
            columns=self.columns, resolve=self.resolve,
        )
        new._theme = t
        return new

    def show_svg(self) -> str:
        try:
            from ferrum._core import compose_svg_grid  # type: ignore[attr-defined]
        except ImportError as e:
            raise NotImplementedError(
                "RepeatChart.show_svg() requires compose_svg_grid; "
                "wire-up lands in Phase 9a Task 12"
            ) from e
        cells = self.expand()
        rows = self.row or []
        cols = self.column or []
        n_rows, n_cols = len(rows), len(cols)
        grid: list = [None] * (n_rows * n_cols)
        for row_field, col_field, chart in cells:
            ri = rows.index(row_field)
            ci = cols.index(col_field)
            grid[ri * n_cols + ci] = chart.show_svg()
        return compose_svg_grid(
            grid, rows=n_rows, cols=n_cols,
            row_ratios=[1.0] * n_rows,
            col_ratios=[1.0] * n_cols,
            spacing=self.spacing,
            share_x=[],
            share_y=[],
        )

    def show_png(self) -> bytes:
        raise NotImplementedError("RepeatChart.show_png — Phase 9 follow-up")

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"RepeatChart.save({fmt!r}) not yet supported")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return (
            f"RepeatChart(row={self.row}, column={self.column}, "
            f"diagonal={'set' if self.diagonal is not None else 'None'}, corner={self.corner})"
        )


class ClusterMapChart:
    """Clustered heatmap with optional row/column dendrograms.

    Layout: 2x2. Top-left empty; top-right = col_dendrogram; bottom-left =
    row_dendrogram (rotated 90°); bottom-right = heatmap. Dendrogram value-axes
    are hidden; categorical axes align with the heatmap row/column labels.
    """
    __slots__ = (
        "heatmap", "row_dendrogram", "col_dendrogram",
        "dendrogram_ratio", "spacing", "_theme",
    )

    def __init__(
        self,
        heatmap,
        *,
        row_dendrogram=None,
        col_dendrogram=None,
        dendrogram_ratio: float = 0.2,
        spacing: float = 0.02,
    ) -> None:
        if not (0.0 < dendrogram_ratio < 1.0):
            raise ValueError(
                f"dendrogram_ratio must be in (0, 1); got {dendrogram_ratio}"
            )
        self.heatmap = heatmap
        self.row_dendrogram = row_dendrogram
        self.col_dendrogram = col_dendrogram
        self.dendrogram_ratio = dendrogram_ratio
        self.spacing = spacing
        self._theme = None

    @property
    def charts(self) -> list:
        return [c for c in (self.heatmap, self.col_dendrogram, self.row_dendrogram) if c is not None]

    @property
    def spec(self) -> dict:
        return {
            "kind": "cluster_map",
            "heatmap": _embed_chart_spec(self.heatmap),
            "row_dendrogram": _embed_chart_spec(self.row_dendrogram),
            "col_dendrogram": _embed_chart_spec(self.col_dendrogram),
            "dendrogram_ratio": self.dendrogram_ratio,
            "spacing": self.spacing,
        }

    def theme(self, t):
        new = ClusterMapChart(
            self.heatmap.theme(t),
            row_dendrogram=(self.row_dendrogram.theme(t) if self.row_dendrogram is not None else None),
            col_dendrogram=(self.col_dendrogram.theme(t) if self.col_dendrogram is not None else None),
            dendrogram_ratio=self.dendrogram_ratio,
            spacing=self.spacing,
        )
        new._theme = t
        return new

    def show_svg(self) -> str:
        try:
            from ferrum._core import compose_svg_grid  # type: ignore[attr-defined]
        except ImportError as e:
            raise NotImplementedError(
                "ClusterMapChart.show_svg() requires compose_svg_grid; "
                "wire-up lands in Phase 9a Task 12"
            ) from e
        d = self.dendrogram_ratio
        h = 1.0 - d
        cells = [
            None,
            self.col_dendrogram.show_svg() if self.col_dendrogram is not None else None,
            self.row_dendrogram.show_svg() if self.row_dendrogram is not None else None,
            self.heatmap.show_svg(),
        ]
        return compose_svg_grid(
            cells, rows=2, cols=2,
            row_ratios=[d, h],
            col_ratios=[d, h],
            spacing=self.spacing,
            share_x=[[1, 3]],
            share_y=[[2, 3]],
        )

    def show_png(self) -> bytes:
        raise NotImplementedError("ClusterMapChart.show_png — Phase 9 follow-up")

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"ClusterMapChart.save({fmt!r}) not yet supported")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return (
            f"ClusterMapChart(heatmap=set, row_dendrogram={'set' if self.row_dendrogram else 'None'}, "
            f"col_dendrogram={'set' if self.col_dendrogram else 'None'}, "
            f"ratio={self.dendrogram_ratio})"
        )
