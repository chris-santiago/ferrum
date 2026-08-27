"""Clustering / manifold mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ferrum.chart import Chart


class ClusteringMarksMixin:
    """Mixin providing clustering / manifold mark methods for Chart."""

    def mark_silhouette(
        self,
        *,
        zero_line: bool = True,
        color_field: str | None = "cluster",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Rousseeuw silhouette plot.

        Displays one horizontal bar per sample whose width encodes its
        silhouette coefficient, coloured by cluster assignment.  Samples within
        each cluster are stacked vertically in descending coefficient order.
        Data must carry the schema emitted by ``ModelSource.silhouette()``:
        ``sample_id``, ``y_position``, ``cluster``, ``silhouette_value``.

        When ``zero_line=True`` a sentinel ``_ref_zero`` column is injected
        so the downstream ``mark_rule`` renders a single vertical rule at x=0.

        The method pre-computes ``_silhouette_x_lo``, ``_silhouette_x_hi``,
        ``_silhouette_y_lo``, and ``_silhouette_y_hi`` columns from the raw
        data so the renderer can draw rect marks directly.

        Parameters
        ----------
        zero_line : bool, optional
            Whether to draw a vertical reference rule at silhouette = 0.
            Default is ``True``.
        color_field : str or None, optional
            Column name driving per-cluster colour.  Default is ``"cluster"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for silhouette rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(kmeans, X)
        >>> fm.Chart(src.silhouette()).mark_silhouette()
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_silhouette
        from ferrum._constant_columns import _inject_constant

        def _silhouette_prep(df):
            import polars as pl

            if "y_position" not in df.columns or "silhouette_value" not in df.columns:
                return df
            df = df.with_columns(
                [
                    pl.min_horizontal(pl.lit(0.0), pl.col("silhouette_value")).alias(
                        "_silhouette_x_lo"
                    ),
                    pl.max_horizontal(pl.lit(0.0), pl.col("silhouette_value")).alias(
                        "_silhouette_x_hi"
                    ),
                    (pl.col("y_position").cast(pl.Float64) - 0.5).alias("_silhouette_y_lo"),
                    (pl.col("y_position").cast(pl.Float64) + 0.5).alias("_silhouette_y_hi"),
                ]
            )
            if zero_line:
                df = _inject_constant(df, "_ref_zero", 0.0)
            return df

        return self._set_composite_mark(
            "silhouette",
            desugar_silhouette,
            {
                "zero_line": zero_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="rect",
            position=position,
            data_transform=_silhouette_prep,
        )

    def mark_pca_scree(
        self,
        *,
        cumulative_line: bool = True,
        threshold_line: float | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a PCA scree plot.

        Displays explained variance ratio per component as bars and optionally
        overlays the cumulative variance ratio as a line.  Data must carry the
        schema emitted by ``ModelSource.pca_variance()``: ``component``,
        ``explained_variance_ratio``, ``cumulative_variance_ratio``.

        When ``threshold_line`` is non-None a sentinel ``_threshold_line``
        column is injected so the downstream ``mark_rule`` draws a single
        horizontal reference line at the threshold value.

        Parameters
        ----------
        cumulative_line : bool, optional
            Whether to overlay a cumulative explained variance line.  Default
            is ``True``.
        threshold_line : float or None, optional
            Y-position of an optional horizontal threshold reference line
            (e.g. ``0.95`` for 95% explained variance).  ``None`` (default)
            omits the line.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for PCA scree rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(pca_model, X)
        >>> fm.Chart(src.pca_variance()).mark_pca_scree(threshold_line=0.95)
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import (
            desugar_pca_scree,
            desugar_pca_scree_with_threshold,
        )
        from ferrum._constant_columns import _inject_constant

        def _pca_scree_prep(df):
            import polars as pl

            if "component" not in df.columns or "explained_variance_ratio" not in df.columns:
                return df
            df = df.with_columns(
                pl.col("component").cast(pl.Utf8).alias("component"),
            )
            # Scale-resolution anchor: render/prepare.rs:265 feeds layer-0's
            # y+y2 into the y-axis domain computation. Layer-0 here is the
            # cumulative line (y range ~ [evr[0], sum(evr)]); the bar baseline
            # at 0 and any threshold rule (0.95 default) sit outside that
            # range. Stash both anchor values into a y2 column on layer-0 so
            # the scale union covers [0, threshold].
            anchor_hi = max(
                float(threshold_line) if threshold_line is not None else 0.0,
                float(df["cumulative_variance_ratio"].max() or 0.0),
            )
            n = df.height
            anchor_vals = [0.0, float(anchor_hi)] + [None] * max(0, n - 2)
            df = df.with_columns(
                pl.Series(
                    "_y_axis_anchor",
                    anchor_vals[:n],
                    dtype=pl.Float64,
                ),
            )
            if threshold_line is not None:
                df = _inject_constant(
                    df,
                    "_threshold_line",
                    float(threshold_line),
                )
            return df

        fn = desugar_pca_scree_with_threshold if threshold_line is not None else desugar_pca_scree
        return self._set_composite_mark(
            "pca_scree",
            fn,
            {"cumulative_line": cumulative_line, **mark_kwargs},
            placeholder="rect",
            position=position,
            data_transform=_pca_scree_prep,
        )

    def mark_intercluster_distance(
        self,
        *,
        label_clusters: bool = True,
        color_field: str | None = "cluster",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a 2-D intercluster distance (MDS) plot.

        Embeds cluster centres into 2-D using MDS and visualises each centre
        as a point whose area encodes cluster size.  With ``label_clusters=True``
        a ``mark_text`` overlay labels each point by its cluster id.  Data must
        carry the schema emitted by ``ModelSource.intercluster_distance()``:
        ``cluster``, ``x``, ``y``, ``size``.

        Parameters
        ----------
        label_clusters : bool, optional
            Whether to overlay cluster-id text labels.  Default is ``True``.
        color_field : str or None, optional
            Column name driving per-cluster colour.  Default is ``"cluster"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the point layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for intercluster-distance
            rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(kmeans, X)
        >>> fm.Chart(src.intercluster_distance()).mark_intercluster_distance()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_intercluster_distance

        return self._set_composite_mark(
            "intercluster_distance",
            desugar_intercluster_distance,
            {
                "label_clusters": label_clusters,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_decision_boundary(
        self,
        *,
        proba: bool = False,
        color_field: str = "z",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a decision-boundary heatmap.

        Colours a pixel grid by the model's predicted class (``proba=False``)
        or class probability (``proba=True``) at each grid point.  Data must
        carry pre-computed cell bound columns ``x``, ``x2``, ``y``, ``y2``
        and a prediction column ``z``.  The chart builder helper
        ``_decision_boundary_chart_from_source`` produces these columns from a
        ``ModelSource``.

        ``proba`` is informational at the mark layer -- the grid's ``z``
        values are already computed (class index or probability) by the time
        they reach this mark, and the continuous colour scale renders either
        kind identically.  It is the ``decision_boundary_chart`` figure
        function that gives ``proba`` its effect: passing
        ``proba=True``/``False`` there selects which grid ``z`` gets
        computed upstream.  Passing ``proba=True`` directly to
        ``mark_decision_boundary`` on data you assembled yourself has no
        effect unless the ``z`` column was already computed to match, and
        emits a one-time warning naming the no-op.

        Parameters
        ----------
        proba : bool, optional
            Whether to colour by predicted probability rather than class index.
            Only has an effect via the ``decision_boundary_chart`` figure
            function; see above.  Default is ``False``.
        color_field : str, optional
            Column name for the colour encoding.  Default is ``"z"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for decision-boundary rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.decision_boundary()).mark_decision_boundary()
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_decision_boundary

        if proba:
            # `proba` is registered in ferrum.marks._informational_kwargs as
            # an informational-only parameter for this mark: it has zero
            # effect at this call -- the grid's z values were already
            # computed by the time they reach this mark. Routing the warning
            # through warn_informational_kwarg (rather than calling
            # ferrum._warn.warn_once directly) is what makes the registry
            # load-bearing: the call raises if this (mark, param) pair isn't
            # registered, and the AST guard in
            # tests/test_mark_kwargs_no_silent_drop.py verifies the converse
            # -- every registered pair has a matching call site here.
            from ferrum.marks._informational_kwargs import warn_informational_kwarg

            warn_informational_kwarg(
                "decision_boundary",
                "proba",
                (
                    "mark_decision_boundary(proba=True) has no effect here -- "
                    "the grid's z values are already computed by the time they "
                    "reach this mark. Use decision_boundary_chart(proba=True), "
                    "which computes P(class=1) upstream before building the grid."
                ),
            )

        return self._set_composite_mark(
            "decision_boundary",
            desugar_decision_boundary,
            {
                "proba": proba,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="rect",
            position=position,
        )
