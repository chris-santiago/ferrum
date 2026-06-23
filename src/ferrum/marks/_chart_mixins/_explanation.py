"""Model-explanation mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ferrum.chart import Chart


class ExplanationMarksMixin:
    """Mixin providing model-explanation mark methods for Chart."""

    def mark_importance(
        self,
        *,
        orient: str = "horizontal",
        error_bars: bool = True,
        top_k: int | None = None,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a feature-importance bar chart.

        Renders one bar per feature, sorted descending by importance.  When
        ``error_bars=True`` and the data carries ``imp_lower``/``imp_upper``
        columns a second errorbar layer is added.  Data must carry the schema
        emitted by ``ModelSource.importances()``: ``feature``, ``importance``,
        ``std``.  The chart builder computes the bound columns and truncates to
        ``top_k`` rows before calling this method.

        Parameters
        ----------
        orient : {"horizontal", "vertical"}, default "horizontal"
            Bar orientation.  ``"horizontal"`` places features on the y axis
            with importance on x (default); ``"vertical"`` swaps axes.
        error_bars : bool, optional
            Whether to draw error bars from ``imp_lower``/``imp_upper``.
            Default is ``True``.
        top_k : int or None, optional
            Limit results to the top-*k* features by importance.  Truncation
            is applied by the figure-level chart builder
            (``importance_chart``); when ``mark_importance`` is called
            directly this parameter is forwarded to the desugar function but
            the desugar layer treats it as informational -- actual filtering
            must be done on the DataFrame before passing it to ``Chart``.
        color_field : str or None, optional
            Column name to drive per-feature colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for feature-importance rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.importances()).mark_importance(top_k=10)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_importance

        def _importance_filter(df):
            if top_k is not None and "importance" in df.columns:
                return df.sort("importance", descending=True).head(top_k)
            return df

        return self._set_composite_mark(
            "importance",
            desugar_importance,
            {
                "orient": orient,
                "error_bars": error_bars,
                "top_k": top_k,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_importance_filter,
        )

    def mark_shap_beeswarm(
        self,
        *,
        max_display: int = 20,
        color_bar: bool = True,
        order: str = "abs_mean",
        zero_line: bool = True,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP beeswarm summary plot.

        Visualises the distribution of SHAP values across samples for each
        feature as a swarm of points, coloured by the feature's original value.
        Data must carry the long-form schema from ``ModelSource.shap_values()``
        pre-filtered to the top ``max_display`` features by the chart builder.

        When ``zero_line=True`` (default) a sentinel ``_ref_zero`` column is
        injected and the downstream desugar appends a dashed ``mark_rule``
        layer at ``x=0``.

        Parameters
        ----------
        max_display : int, optional
            Maximum number of top features to show.  Default is ``20``.
        color_bar : bool, optional
            Whether to render a colour bar for the feature-value scale.
            Default is ``True``.
        order : {"abs_mean", "mean", "none"}, default "abs_mean"
            Feature ordering: by mean absolute SHAP (``"abs_mean"``), signed
            mean SHAP (``"mean"``), or original order (``"none"``).
        zero_line : bool, optional
            Whether to overlay a dashed vertical reference rule at
            ``shap_value = 0``.  Default is ``True`` (Schwabish SB-followup
            2026-05-12).
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the point layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for SHAP beeswarm rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.shap_values()).mark_shap_beeswarm(max_display=15)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_shap_beeswarm
        from ferrum._constant_columns import _inject_constant

        def _shap_beeswarm_prep(df):
            import polars as pl

            # Filter to top max_display features by mean |SHAP value|.
            if max_display is not None and "shap_value" in df.columns and "feature" in df.columns:
                ranked = (
                    df.group_by("feature")
                    .agg(pl.col("shap_value").abs().mean().alias("_score"))
                    .sort("_score", descending=True)
                    .head(max_display)
                )
                keep = ranked["feature"].to_list()
                df = df.filter(pl.col("feature").is_in(keep))
            if zero_line and "shap_value" in df.columns:
                df = _inject_constant(df, "_ref_zero", 0.0)
            return df

        return self._set_composite_mark(
            "shap_beeswarm",
            desugar_shap_beeswarm,
            {
                "max_display": max_display,
                "color_bar": color_bar,
                "order": order,
                "zero_line": zero_line,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_shap_beeswarm_prep,
        )

    def mark_shap_bar(
        self,
        *,
        max_display: int = 20,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP aggregated-bar feature importance chart.

        Shows mean absolute SHAP values per feature as a horizontal bar chart.
        Data must carry the long-form schema from ``ModelSource.shap_values()``
        pre-filtered to the top ``max_display`` features by the chart builder.

        Parameters
        ----------
        max_display : int, optional
            Maximum number of top features to show.  Default is ``20``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for SHAP bar rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.shap_values()).mark_shap_bar(max_display=10)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_shap_bar

        def _shap_bar_filter(df):
            import polars as pl

            if max_display is not None and "shap_value" in df.columns and "feature" in df.columns:
                ranked = (
                    df.group_by("feature")
                    .agg(pl.col("shap_value").abs().mean().alias("_score"))
                    .sort("_score", descending=True)
                    .head(max_display)
                )
                keep = ranked["feature"].to_list()
                df = df.filter(pl.col("feature").is_in(keep))
            return df

        return self._set_composite_mark(
            "shap_bar",
            desugar_shap_bar,
            {"max_display": max_display, **mark_kwargs},
            placeholder="point",
            position=position,
            data_transform=_shap_bar_filter,
        )

    def mark_shap_waterfall(
        self,
        *,
        sample_idx: int = -1,
        max_display: int = 20,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP waterfall chart for one sample.

        Shows how each feature pushes the model output from the base value
        toward the final prediction for a single observation.  Data must carry
        the long-form schema from ``ModelSource.shap_values()`` for the chosen
        ``sample_idx``.

        Parameters
        ----------
        sample_idx : int
            Row index of the sample to explain.  Must be provided explicitly;
            the default ``-1`` is a guard sentinel that raises ``ValueError``
            immediately so callers get a clear error at call time rather than
            at render time.
        max_display : int, optional
            Maximum number of features to show (smallest-magnitude features
            are collapsed into an ``"other"`` row).  Default is ``20``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for SHAP waterfall rendering.

        Raises
        ------
        ValueError
            If ``sample_idx`` is not provided (left at its default of ``-1``).

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.shap_values()).mark_shap_waterfall(sample_idx=0)
        Chart(mark='point', encoding=[])
        """
        if sample_idx == -1:
            raise ValueError(
                "mark_shap_waterfall requires sample_idx=<int>; "
                "pass an integer index (e.g. sample_idx=0) to select the sample to explain."
            )
        from ferrum.marks.diagnostic import desugar_shap_waterfall

        def _shap_waterfall_filter(df):
            import polars as pl

            if max_display is not None and "shap_value" in df.columns and "feature" in df.columns:
                ranked = (
                    df.group_by("feature")
                    .agg(pl.col("shap_value").abs().mean().alias("_score"))
                    .sort("_score", descending=True)
                    .head(max_display)
                )
                keep = ranked["feature"].to_list()
                df = df.filter(pl.col("feature").is_in(keep))
            return df

        return self._set_composite_mark(
            "shap_waterfall",
            desugar_shap_waterfall,
            {
                "sample_idx": sample_idx,
                "max_display": max_display,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_shap_waterfall_filter,
        )

    def mark_pdp(
        self,
        *,
        kind: str = "average",
        ice_alpha: float = 0.2,
        center: bool = False,
        color_field: str | None = "feature",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render partial-dependence plots (PDP / ICE).

        Visualises how the model's output varies as a function of one feature
        while marginalising over all others.  Data must carry the long-form
        schema from ``ModelSource.partial_dependence()``: ``feature``,
        ``feature_value``, ``pd_value``.  The chart builder pre-sorts ascending
        by ``feature_value``.

        Parameters
        ----------
        kind : {"average", "individual", "both"}, default "average"
            What to render.  ``"average"`` draws the mean PD line;
            ``"individual"`` draws ICE lines (one per sample);
            ``"both"`` overlays average on top of ICE lines.
        ice_alpha : float, optional
            Opacity of individual ICE lines when ``kind`` is ``"individual"``
            or ``"both"``.  Default is ``0.2``.
        center : bool, optional
            Whether to centre ICE lines at their first value (centred ICE).
            Default is ``False``.
        color_field : str or None, optional
            Column name driving per-feature colour.  Default is ``"feature"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for PDP / ICE rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.partial_dependence()).mark_pdp(kind="both", center=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_pdp

        return self._set_composite_mark(
            "pdp",
            desugar_pdp,
            {
                "kind": kind,
                "ice_alpha": ice_alpha,
                "center": center,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )
