"""Feature-ranking mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ferrum.chart import Chart


class RankingMarksMixin:
    """Mixin providing feature-ranking mark methods for Chart."""

    def mark_rank1d(
        self,
        *,
        orient: str = "horizontal",
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a univariate feature-ranking bar chart.

        Ranks features by a univariate score (e.g. mutual information, ANOVA
        F-statistic) and displays them as a bar chart sorted by rank.  Data must
        carry the schema emitted by ``ModelSource.rank1d()``: ``feature``,
        ``score``, ``rank``.

        Parameters
        ----------
        orient : {"horizontal", "vertical"}, default "horizontal"
            Bar orientation.  ``"horizontal"`` places features on the y axis
            with score on x; ``"vertical"`` places features on x.
        color_field : str or None, optional
            Column name driving per-feature colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for rank-1D rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.rank1d()).mark_rank1d()
        Chart(mark='bar', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_rank1d

        return self._set_composite_mark(
            "rank1d",
            desugar_rank1d,
            {
                "orient": orient,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="bar",
            position=position,
        )

    def mark_rank2d(
        self,
        *,
        annot: bool = True,
        color_field: str = "correlation",
        text_field: str = "correlation_fmt",
        cmap: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a pairwise feature-ranking correlation heatmap.

        Displays pairwise feature correlation scores as a colour-coded matrix.
        When ``annot=True`` a text overlay renders each cell's value to 2 dp.
        Data must carry the schema emitted by ``ModelSource.rank2d()``:
        ``feature_x``, ``feature_y``, ``correlation``.  The chart builder
        appends a ``correlation_fmt`` (Utf8) column when ``annot=True``.

        Parameters
        ----------
        annot : bool, optional
            Whether to overlay per-cell correlation value text.  Default is
            ``True``.
        color_field : str, optional
            Column driving the heatmap colour scale.  Default is
            ``"correlation"``.
        text_field : str, optional
            Column read by the text layer.  Default is ``"correlation_fmt"``.
        cmap : str or None, optional
            Diverging colormap name for correlation cells.  ``None`` (default)
            defers to the theme's diverging scheme.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for rank-2D rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.rank2d()).mark_rank2d(annot=True)
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_rank2d

        return self._set_composite_mark(
            "rank2d",
            desugar_rank2d,
            {
                "annot": annot,
                "color_field": color_field,
                "text_field": text_field,
                "cmap": cmap,
                **mark_kwargs,
            },
            placeholder="rect",
            position=position,
        )

    def mark_parallel_coordinates(
        self,
        *,
        alpha: float = 0.5,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a parallel coordinates plot.

        Draws one polyline per sample across normalised feature axes.  Data
        must carry the long-form schema produced by
        ``_parallel_coords_chart_from_dataframe``: ``feature`` (Utf8),
        ``value`` (Float64), ``sample_id`` (Utf8), and (optionally) a hue
        column passed via ``color_field``.  The line layer uses
        ``mark_style.detail = "sample_id"`` so each sample renders as its own
        polyline.

        Parameters
        ----------
        alpha : float, optional
            Line opacity.  Default is ``0.5``.
        color_field : str or None, optional
            Column name driving per-sample (or per-class) line colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for parallel-coordinates rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"feature": ["a","a","b","b"],
        ...                    "value": [0.5, 0.3, 0.8, 0.2],
        ...                    "sample_id": ["s0", "s1", "s0", "s1"]})
        >>> fm.Chart(df).mark_parallel_coordinates(color_field="sample_id")
        Chart(mark='line', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_parallel_coordinates

        return self._set_composite_mark(
            "parallel_coordinates",
            desugar_parallel_coordinates,
            {
                "alpha": alpha,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="line",
            position=position,
        )
