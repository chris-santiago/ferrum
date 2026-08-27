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
        order : {"abs_mean", "mean", "max", "none"}, default "abs_mean"
            Feature ordering: by mean absolute SHAP (``"abs_mean"``), max
            absolute SHAP (``"max"``, surfaces high-impact-outlier
            features), signed mean SHAP (``"mean"``), or original order
            (``"none"``).
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
        from ferrum._validate import validate_choice
        from ferrum.marks._desugar_helpers import (
            SHAP_BEESWARM_COLOR_FIELD,
            SHAP_ORDER_VALUES,
            shap_beeswarm_color_channel,
        )

        validate_choice("mark_shap_beeswarm", "order", order, SHAP_ORDER_VALUES)

        def _shap_beeswarm_prep(df):
            import polars as pl

            has_shap = "shap_value" in df.columns and "feature" in df.columns
            # Filter to top max_display features by mean |SHAP value|.
            if max_display is not None and has_shap:
                ranked = (
                    df.group_by("feature")
                    .agg(pl.col("shap_value").abs().mean().alias("_score"))
                    .sort("_score", descending=True)
                    .head(max_display)
                )
                keep = ranked["feature"].to_list()
                df = df.filter(pl.col("feature").is_in(keep))
            # Arrange rows so `feature`'s first-appearance (encounter) order
            # matches the requested display order -- the ordinal y-domain is
            # built from row-encounter order (see plots/explanation.py's
            # matching pl.Enum sort), so reordering rows is how `order`
            # actually controls the axis. order="none" leaves rows as-is.
            if order != "none" and has_shap:
                if order == "abs_mean":
                    agg_expr = pl.col("shap_value").abs().mean()
                elif order == "max":
                    agg_expr = pl.col("shap_value").abs().max()
                else:  # "mean"
                    agg_expr = pl.col("shap_value").mean()
                order_list = (
                    df.group_by("feature")
                    .agg(agg_expr.alias("_score"))
                    .sort("_score", descending=True)["feature"]
                    .to_list()
                )
                # maintain_order=True: this row order feeds the seeded
                # Jitter(seed=42) below and the point draw order, both of
                # which land in the rendered SVG. Polars' default
                # maintain_order=False may permute rows within a tied
                # `_feature_order` group, which would make the render
                # depend on an unguaranteed sort property instead of only
                # on `order`.
                df = (
                    df.with_columns(
                        pl.col("feature").cast(pl.Enum(order_list)).alias("_feature_order")
                    )
                    .sort("_feature_order", maintain_order=True)
                    .drop("_feature_order")
                )
            if zero_line and "shap_value" in df.columns:
                df = _inject_constant(df, "_ref_zero", 0.0)
            # Rename the internal `feature_value_normalized` schema column
            # to a presentable label before the chart holds the data, so
            # the colorbar-legend title fallback (Rust ignores
            # `Color(title=)` for legends -- a pre-existing, package-wide
            # gap; see design-docs/superpowers/followups/2026-05-15-code-archaeology.md)
            # renders something a user would want to see instead of the
            # raw schema name (2026-08-27 close-out). `desugar_shap_beeswarm`
            # and the chart-level mirror below both read
            # `SHAP_BEESWARM_COLOR_FIELD`, so the rename and the field
            # reference cannot drift apart.
            if "feature_value_normalized" in df.columns:
                df = df.rename({"feature_value_normalized": SHAP_BEESWARM_COLOR_FIELD})
            return df

        chart = self._set_composite_mark(
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
        # 2026-08-27 close-out: also mirror the point layer's color channel
        # onto the *chart-level* encoding. `desugar_shap_beeswarm` already
        # sets this exact `Color(...)` (field, scheme, and `legend=`) on the
        # layer itself, which drives the point fill correctly -- but the
        # Rust renderer's colorbar-legend construction for a layered/composite
        # chart reads its `legend=` config from the chart-level
        # `encoding.color`, not from any per-layer color channel. Without
        # this, `color_bar=False` was silently discarded (a colorbar
        # rendered regardless) and the `color_bar=True` "Low"/"High" tick
        # labels never appeared either -- both are wired to a real Rust
        # colorbar decision now, verified by rendering, not merely by
        # inspecting the emitted spec JSON. This mark has no user-facing
        # `color_field=` parameter, so nothing else can set this key; a
        # caller's own later `.encode(color=...)` still wins normally, as
        # with any chart-level channel. `shap_beeswarm_color_channel` is the
        # single authority for this config -- `desugar_shap_beeswarm` calls
        # the same factory for the layer-level copy, so the two cannot
        # diverge.
        return chart.encode(color=shap_beeswarm_color_channel(color_bar=color_bar))

    def mark_shap_bar(
        self,
        *,
        max_display: int | None = 20,
        orient: str = "horizontal",
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a SHAP aggregated-bar feature importance chart.

        Shows mean absolute SHAP values per feature as a horizontal bar chart.
        Data must carry ``feature`` (Utf8) and ``abs_mean_shap`` (Float64) --
        the aggregated schema ``desugar_shap_bar`` renders, one row per
        feature. ``max_display`` truncates to the top-scoring features by
        descending ``abs_mean_shap`` (2026-08-27 close-out).

        Parameters
        ----------
        max_display : int, optional
            Maximum number of top features to show, ranked by descending
            ``abs_mean_shap``.  Default is ``20``.
        orient : {"horizontal", "vertical"}, default "horizontal"
            Bar orientation.  ``"horizontal"`` (default) places the value on
            x and feature on the ordinal y axis, matching the single-model
            layout.  ``"vertical"`` swaps the axes (feature on ordinal x,
            value on y) -- used internally by the ``compare=`` dodge-by-model
            builder, which requires an ordinal-x band axis and re-applies
            ``CoordFlip`` to restore the horizontal visual.
        color_field : str or None, optional
            Column name driving per-bar colour (e.g. ``"model"`` under
            ``compare=``).
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

            # Re-keyed on this mark's actual data contract, `abs_mean_shap`
            # (2026-08-27 close-out) -- the prior `"shap_value" in df.columns`
            # guard checked a long-form column this mark's documented input
            # never carries (see `desugar_shap_bar`'s data contract), so the
            # truncation silently never fired on documented input.
            if (
                max_display is not None
                and "abs_mean_shap" in df.columns
                and "feature" in df.columns
            ):
                ranked = (
                    df.group_by("feature")
                    .agg(pl.col("abs_mean_shap").max().alias("_score"))
                    .sort("_score", descending=True)
                    .head(max_display)
                )
                keep = ranked["feature"].to_list()
                df = df.filter(pl.col("feature").is_in(keep))
            return df

        return self._set_composite_mark(
            "shap_bar",
            desugar_shap_bar,
            {
                "max_display": max_display,
                "orient": orient,
                "color_field": color_field,
                **mark_kwargs,
            },
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
        ``feature`` (Utf8), ``x0``/``x1`` (cumulative start/end, Float64), and
        ``shap_sign`` (Utf8) -- the schema ``desugar_shap_waterfall`` renders,
        one row per feature for the chosen sample. ``max_display`` truncates
        to the top-scoring features by descending contribution magnitude
        ``|x1 - x0|`` (2026-08-27 close-out).

        Parameters
        ----------
        sample_idx : int
            Row index of the sample to explain.  Must be provided explicitly;
            the default ``-1`` is a guard sentinel that raises ``ValueError``
            immediately so callers get a clear error at call time rather than
            at render time.
        max_display : int, optional
            Maximum number of features to show, ranked by descending
            contribution magnitude ``|x1 - x0|``.  Default is ``20``.
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

            # Re-keyed on this mark's actual data contract, `x0`/`x1`
            # (2026-08-27 close-out) -- the prior `"shap_value" in df.columns`
            # guard checked a column this mark's documented input never
            # carries (see `desugar_shap_waterfall`'s data contract), so the
            # truncation silently never fired on documented input. Ranks by
            # descending contribution magnitude `|x1 - x0|`.
            if max_display is not None and {"feature", "x0", "x1"} <= set(df.columns):
                ranked = (
                    df.group_by("feature")
                    .agg((pl.col("x1") - pl.col("x0")).abs().max().alias("_score"))
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
            Informational at the mark layer -- ICE polylines are already
            re-based to start at 0 by the time they reach this mark. Pass
            it to ``pdp_chart(center=...)`` instead. A truthy value passed
            directly here emits a one-time warning naming the no-op.
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
        >>> fm.Chart(src.partial_dependence()).mark_pdp(kind="both")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_pdp

        if center:
            # `center` is registered in ferrum.marks._informational_kwargs
            # as an informational-only parameter for this mark: it has zero
            # effect at this call -- ICE polylines are already re-based to
            # start at 0 by the time they reach this mark. See
            # mark_decision_boundary for the full rationale of routing this
            # through warn_informational_kwarg rather than warn_once
            # directly.
            from ferrum.marks._informational_kwargs import warn_informational_kwarg

            warn_informational_kwarg(
                "pdp",
                "center",
                (
                    "mark_pdp(center=True) has no effect here -- ICE "
                    "polylines are already re-based to start at 0 by the "
                    "time they reach this mark. Use pdp_chart(center=True) "
                    "to control centering."
                ),
            )

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
