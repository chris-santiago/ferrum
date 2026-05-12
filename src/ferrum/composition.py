"""Multi-chart composition primitives (HConcat, VConcat, Joint, Repeat, ClusterMap)."""
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
    """Horizontal concatenation of two or more charts.

    Each sub-chart retains its own scales, axes, and legend.  Construct via
    the ``|`` operator on ``Chart`` instances or directly with a list.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to concatenate left-to-right.
    spacing : float, default 10.0
        Horizontal pixel gap between adjacent charts.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.Chart(df).encode(x="hp", y="mpg").mark_point() | fm.Chart(df).encode(x="hp").mark_histogram()
    >>> combined.save("side_by_side.svg")
    """

    def show_svg(self) -> str:
        """Render the horizontally concatenated charts to an SVG string.

        Returns
        -------
        str
            SVG markup with sub-charts placed left-to-right.
        """
        from ferrum._core import compose_svg_horizontal
        svgs = [c.show_svg() for c in self.charts]
        return compose_svg_horizontal(svgs, spacing=self.spacing, align="top")

    def show_png(self) -> bytes:
        """Render to PNG bytes.

        .. note::
            Not yet implemented in Phase 8a.  Use ``.save('out.svg')`` instead.

        Raises
        ------
        NotImplementedError
            Always raised until PNG wiring is complete.
        """
        raise NotImplementedError(
            "HConcatChart.show_png not yet wired in Phase 8a; "
            "use .save('out.svg') instead (Phase 8a follow-up)."
        )

    def save(self, path: str, *, format=None, **kwargs):
        """Save the composition to a file.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : str, optional
            ``"svg"`` is the only supported value.  Other formats raise
            ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            If *format* is not ``"svg"``.
        """
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
        """Print the SVG markup to stdout."""
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.show_svg()

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"HConcatChart([{', '.join(repr(c) for c in self.charts)}])"


class VConcatChart(_CompositeBase):
    """Vertical concatenation of two or more charts.

    Each sub-chart retains its own scales, axes, and legend.  Construct via
    the ``&`` operator on ``Chart`` instances or directly with a list.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to stack top-to-bottom.
    spacing : float, default 10.0
        Vertical pixel gap between adjacent charts.

    Examples
    --------
    >>> import ferrum as fm
    >>> stacked = fm.Chart(df).encode(x="hp", y="mpg").mark_point() & fm.Chart(df).encode(x="hp").mark_histogram()
    >>> stacked.save("stacked.svg")
    """

    def show_svg(self) -> str:
        """Render the vertically concatenated charts to an SVG string.

        Returns
        -------
        str
            SVG markup with sub-charts stacked top-to-bottom.
        """
        from ferrum._core import compose_svg_vertical
        svgs = [c.show_svg() for c in self.charts]
        return compose_svg_vertical(svgs, spacing=self.spacing, align="left")

    def show_png(self) -> bytes:
        """Render to PNG bytes.

        .. note::
            Not yet implemented in Phase 8a.  Use ``.save('out.svg')`` instead.

        Raises
        ------
        NotImplementedError
            Always raised until PNG wiring is complete.
        """
        raise NotImplementedError(
            "VConcatChart.show_png not yet wired in Phase 8a; "
            "use .save('out.svg') instead."
        )

    def save(self, path: str, *, format=None, **kwargs):
        """Save the composition to a file.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : str, optional
            ``"svg"`` is the only supported value.  Other formats raise
            ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            If *format* is not ``"svg"``.
        """
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
        """Print the SVG markup to stdout."""
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.show_svg()

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"VConcatChart([{', '.join(repr(c) for c in self.charts)}])"


# --------------------------------------------------------------------------
# Phase 9 compound views: JointChart, RepeatChart, ClusterMapChart
# --------------------------------------------------------------------------


def _embed_chart_spec(c) -> Optional[dict]:
    """Convert a Chart's ``.to_spec()`` output to an embedded JSON dict."""
    import json as _json
    if c is None or not hasattr(c, "to_spec"):
        return None
    return _json.loads(c.to_spec().to_json())


class JointChart:
    """Joint distribution view: center chart plus optional top and right marginals.

    Lays out a 2 × 2 grid: center chart occupies the bottom-left cell,
    *top* marginal goes top-left, *right* marginal goes bottom-right, and the
    top-right corner is empty.  The x-axis is shared between the center and
    top charts; the y-axis is shared between the center and right charts.

    The cell size ratio between the center and each marginal is controlled by
    ``ratio``.  A ratio of 5 gives the center 5/(5+1) of each dimension and
    each marginal 1/(5+1).

    Most users obtain a ``JointChart`` from `ferrum.jointplot` rather than
    constructing one directly.

    Parameters
    ----------
    center : Chart
        Primary scatter / distribution chart occupying the main panel.
    top : Chart, optional
        Marginal chart drawn above the center (e.g. a histogram of the x
        variable).
    right : Chart, optional
        Marginal chart drawn to the right of the center (e.g. a histogram
        of the y variable).
    ratio : int, default 5
        Size ratio of the center panel to each marginal panel.  Must be > 0.
    spacing : float, default 0.02
        Fractional gap between adjacent cells (0 = no gap, 1 = full width).

    Raises
    ------
    ValueError
        If *ratio* is not > 0.

    Examples
    --------
    >>> import ferrum as fm
    >>> joint = fm.jointplot(df, x="hp", y="mpg")
    >>> joint.save("joint.svg")
    """

    __slots__ = ("center", "top", "right", "ratio", "spacing", "_theme")

    def __init__(
        self,
        center,
        *,
        top=None,
        right=None,
        ratio: int = 5,
        spacing: float = 10.0,
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
        """List of Chart : All non-None sub-charts (center, top, right)."""
        return [c for c in (self.center, self.top, self.right) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
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
        """Apply a theme to all sub-charts and return a new ``JointChart``.

        Parameters
        ----------
        t : Theme
            Theme value to apply.

        Returns
        -------
        JointChart
            A new instance with *t* applied to every sub-chart.
        """
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
        """Forward ``properties(**kwargs)`` to the center chart.

        Parameters
        ----------
        **kwargs
            Keyword arguments accepted by ``Chart.properties`` (e.g.
            ``width``, ``height``, ``title``).

        Returns
        -------
        JointChart
            A new instance with updated center-chart properties.
        """
        new = JointChart(
            self.center.properties(**kwargs),
            top=self.top, right=self.right,
            ratio=self.ratio, spacing=self.spacing,
        )
        new._theme = self._theme
        return new

    def show_svg(self) -> str:
        """Render the joint chart to an SVG string.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.

        Raises
        ------
        NotImplementedError
            If ``compose_svg_grid`` is not available in the compiled
            ``ferrum._core`` extension.
        """
        try:
            from ferrum._core import compose_svg_grid  # type: ignore[attr-defined]
        except ImportError as e:
            raise NotImplementedError(
                "JointChart.show_svg() requires compose_svg_grid; "
                "compose_svg_grid not available in this build"
            ) from e
        # F20: the Rust grid compositor now honors row_ratios/col_ratios via
        # viewBox-scaled per-cell wrappers, so marginals can be passed at
        # their native size and the compositor handles proportional sizing.
        # The marginals still suppress their own axis decoration — the
        # data axis is redundant against the centre cell and the marginal-
        # only axis (count/density on a thin strip) is illegible at marginal
        # size.
        top_chart = self.top.axis(show=False) if self.top is not None else None
        right_chart = self.right.axis(show=False) if self.right is not None else None
        top_svg = top_chart.show_svg() if top_chart is not None else None
        right_svg = right_chart.show_svg() if right_chart is not None else None
        cells = [top_svg, None, self.center.show_svg(), right_svg]
        marginal_share = 1.0 / (self.ratio + 1)
        center_share = self.ratio / (self.ratio + 1)
        return compose_svg_grid(
            cells, rows=2, cols=2,
            row_ratios=[marginal_share, center_share],
            col_ratios=[center_share, marginal_share],
            spacing=self.spacing,
        )

    def show_png(self) -> bytes:
        """Render to PNG bytes.

        Raises
        ------
        NotImplementedError
            Always raised; PNG output is not yet implemented.
        """
        raise NotImplementedError("JointChart.show_png — not yet implemented")

    def save(self, path: str, *, format=None, **kwargs):
        """Save the joint chart to a file.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : str, optional
            ``"svg"`` is the only supported value.

        Raises
        ------
        NotImplementedError
            If *format* is not ``"svg"``.
        """
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"JointChart.save({fmt!r}) not yet supported")

    def show(self):
        """Print the SVG markup to stdout."""
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.show_svg()

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"JointChart(center={self.center!r}, top={self.top!r}, "
            f"right={self.right!r}, ratio={self.ratio})"
        )


class RepeatChart:
    """Repeat a template chart over a grid of row / column field combinations.

    Use ``Repeat.column``, ``Repeat.row``, or ``Repeat.layer`` typed sentinels
    in the template's ``.encode(...)`` call to mark which encoding channel
    receives the per-cell field substitution.  ``RepeatChart.expand()``
    materializes the grid into fully-resolved ``(row_field, col_field, Chart)``
    tuples.

    ``diagonal=`` provides an alternate template for cells where
    ``row_field == col_field`` (symmetric n × n repeat).  ``corner=True``
    filters the expanded grid to the lower triangle including the diagonal.

    Most users obtain a ``RepeatChart`` through ``Chart.repeat()`` or
    ``ferrum.pairplot``.

    Parameters
    ----------
    template : Chart
        Template chart whose ``Repeat.*`` placeholders are substituted per
        cell.
    row : list of str, optional
        Field names assigned to the row axis.
    column : list of str, optional
        Field names assigned to the column axis.
    layer : list of str, optional
        Field names assigned to the layer axis (for non-grid repeat layouts).
    diagonal : Chart, optional
        Alternate template used when ``row_field == col_field``.  Requires
        both *row* and *column* to be set.
    corner : bool, default False
        When ``True``, only the lower-triangle cells (``ri >= ci``) are
        rendered, giving a half-matrix layout.
    spacing : float, default 0.02
        Fractional gap between adjacent cells.
    columns : int, optional
        Maximum number of columns for a wrapped 1-D repeat layout (no-op
        for 2-D row/column repeat).
    resolve : object, optional
        Reserved for future scale-resolution overrides (no-op today).

    Raises
    ------
    ValueError
        If *diagonal* is set but *row* or *column* is not.

    Examples
    --------
    >>> import ferrum as fm
    >>> base = fm.Chart(df).encode(x=fm.Repeat.column, y=fm.Repeat.row).mark_point()
    >>> grid = fm.RepeatChart(base, row=["mpg", "hp"], column=["mpg", "hp"])
    >>> grid.save("pair_grid.svg")
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
        spacing: float = 10.0,
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
        """Dict : Serializable layout spec consumed by the SVG compositor."""
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
        """Materialize the template into fully-resolved chart cells.

        Iterates every ``(row_field, col_field)`` combination, substitutes
        ``Repeat.*`` placeholders, and applies the *diagonal* template where
        applicable.

        Returns
        -------
        list of tuple
            Each element is ``(row_field, col_field, Chart)`` with all
            ``Repeat.*`` placeholders replaced by concrete field names.

        Warns
        -----
        UserWarning
            If *diagonal* is set but ``row != column`` (asymmetric repeat),
            the diagonal template is silently skipped and a warning is issued.
        """
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
        """Map a Repeat placeholder axis name to the concrete field string."""
        if placeholder_axis == "column":
            return col_field
        if placeholder_axis == "row":
            return row_field
        if placeholder_axis == "layer":
            return row_field
        raise ValueError(f"unknown Repeat placeholder axis '{placeholder_axis}'")

    def theme(self, t):
        """Apply a theme to all sub-charts and return a new ``RepeatChart``.

        Parameters
        ----------
        t : Theme
            Theme value to apply.

        Returns
        -------
        RepeatChart
            A new instance with *t* applied to the template and diagonal.
        """
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
        """Render the repeated grid to an SVG string.

        Returns
        -------
        str
            SVG markup containing all materialized cell charts in a grid.

        Raises
        ------
        NotImplementedError
            If ``compose_svg_grid`` is not available in the compiled
            ``ferrum._core`` extension.
        """
        try:
            from ferrum._core import compose_svg_grid  # type: ignore[attr-defined]
        except ImportError as e:
            raise NotImplementedError(
                "RepeatChart.show_svg() requires compose_svg_grid; "
                "compose_svg_grid not available in this build"
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
        )

    def show_png(self) -> bytes:
        """Render to PNG bytes.

        Raises
        ------
        NotImplementedError
            Always raised; PNG output is not yet implemented.
        """
        raise NotImplementedError("RepeatChart.show_png — not yet implemented")

    def save(self, path: str, *, format=None, **kwargs):
        """Save the repeated grid to a file.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : str, optional
            ``"svg"`` is the only supported value.

        Raises
        ------
        NotImplementedError
            If *format* is not ``"svg"``.
        """
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"RepeatChart.save({fmt!r}) not yet supported")

    def show(self):
        """Print the SVG markup to stdout."""
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.show_svg()

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"RepeatChart(row={self.row}, column={self.column}, "
            f"diagonal={'set' if self.diagonal is not None else 'None'}, corner={self.corner})"
        )


class ClusterMapChart:
    """Clustered heatmap with optional row and column dendrograms.

    Lays out a 2 × 2 grid: the heatmap occupies the bottom-right cell,
    the column dendrogram goes top-right, the row dendrogram (rotated 90°)
    goes bottom-left, and the top-left corner is empty.  Dendrogram value
    axes are hidden; categorical axes align with the heatmap row/column labels.

    Cell size is split by ``dendrogram_ratio``: dendrograms receive that
    fraction of the total width/height, the heatmap receives the remainder.

    Most users obtain a ``ClusterMapChart`` from `ferrum.clustermap` rather
    than constructing one directly.

    Parameters
    ----------
    heatmap : Chart
        The central heatmap chart.
    row_dendrogram : Chart, optional
        Dendrogram chart for the row axis.  Displayed to the left of the
        heatmap, rotated 90°.
    col_dendrogram : Chart, optional
        Dendrogram chart for the column axis.  Displayed above the heatmap.
    dendrogram_ratio : float, default 0.2
        Fraction of the total width/height allocated to each dendrogram panel.
        Must be in the open interval (0, 1).
    spacing : float, default 0.02
        Fractional gap between adjacent cells.

    Raises
    ------
    ValueError
        If *dendrogram_ratio* is not in the open interval (0, 1).

    Examples
    --------
    >>> import ferrum as fm
    >>> cm = fm.clustermap(df, method="ward", cmap="rdbu")
    >>> cm.save("clustermap.svg")
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
        spacing: float = 10.0,
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
        """List of Chart : All non-None sub-charts (heatmap, col_dendrogram, row_dendrogram)."""
        return [c for c in (self.heatmap, self.col_dendrogram, self.row_dendrogram) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        return {
            "kind": "cluster_map",
            "heatmap": _embed_chart_spec(self.heatmap),
            "row_dendrogram": _embed_chart_spec(self.row_dendrogram),
            "col_dendrogram": _embed_chart_spec(self.col_dendrogram),
            "dendrogram_ratio": self.dendrogram_ratio,
            "spacing": self.spacing,
        }

    def theme(self, t):
        """Apply a theme to all sub-charts and return a new ``ClusterMapChart``.

        Parameters
        ----------
        t : Theme
            Theme value to apply.

        Returns
        -------
        ClusterMapChart
            A new instance with *t* applied to every sub-chart.
        """
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
        """Render the cluster map to an SVG string.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.

        Raises
        ------
        NotImplementedError
            If ``compose_svg_grid`` is not available in the compiled
            ``ferrum._core`` extension.
        """
        try:
            from ferrum._core import compose_svg_grid  # type: ignore[attr-defined]
        except ImportError as e:
            raise NotImplementedError(
                "ClusterMapChart.show_svg() requires compose_svg_grid; "
                "compose_svg_grid not available in this build"
            ) from e
        d = self.dendrogram_ratio
        h = 1.0 - d
        # Pre-resize each component so the heatmap fills (h × h) of the grid
        # and dendrograms occupy the remaining (d) on the row/col axis they
        # sit beside. Post-F20 the compositor honors row_ratios/col_ratios,
        # but we still pre-resize because the dendrogram tree topology depends
        # on the panel viewport at SVG-emit time — letting the compositor
        # rescale after-the-fact would distort branch positions.
        hm_w = self.heatmap._width or 600.0
        hm_h = self.heatmap._height or 400.0
        dendro_w = hm_w * d / h
        dendro_h = hm_h * d / h
        # Dendrograms have no meaningful axes (only the tree structure
        # matters). clustermap() already calls .axis(show=False) on each
        # dendrogram chart at construction time, so spec-level suppression
        # is in effect here — no post-render SVG mangling needed.
        col_dendro = (self.col_dendrogram.properties(width=hm_w, height=dendro_h)
                      if self.col_dendrogram is not None else None)
        row_dendro = (self.row_dendrogram.properties(width=dendro_w, height=hm_h)
                      if self.row_dendrogram is not None else None)
        col_svg = col_dendro.show_svg() if col_dendro is not None else None
        row_svg = row_dendro.show_svg() if row_dendro is not None else None
        cells = [None, col_svg, row_svg, self.heatmap.show_svg()]
        return compose_svg_grid(
            cells, rows=2, cols=2,
            row_ratios=[d, h],
            col_ratios=[d, h],
            spacing=self.spacing,
        )

    def show_png(self) -> bytes:
        """Render to PNG bytes.

        Raises
        ------
        NotImplementedError
            Always raised; PNG output is not yet implemented.
        """
        raise NotImplementedError("ClusterMapChart.show_png — not yet implemented")

    def save(self, path: str, *, format=None, **kwargs):
        """Save the cluster map to a file.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : str, optional
            ``"svg"`` is the only supported value.

        Raises
        ------
        NotImplementedError
            If *format* is not ``"svg"``.
        """
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"ClusterMapChart.save({fmt!r}) not yet supported")

    def show(self):
        """Print the SVG markup to stdout."""
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.show_svg()

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"ClusterMapChart(heatmap=set, row_dendrogram={'set' if self.row_dendrogram else 'None'}, "
            f"col_dendrogram={'set' if self.col_dendrogram else 'None'}, "
            f"ratio={self.dendrogram_ratio})"
        )
