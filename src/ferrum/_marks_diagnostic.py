"""Diagnostic and model-selection mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations


class DiagnosticMarksMixin:
    """Mixin providing diagnostic / model-selection / clustering mark methods."""

    # ---- Marks (diagnostic -- Phase 10) ----

    def mark_residuals(
        self,
        *,
        kind: str = "studentized",
        reference_line: bool = True,
        cook_threshold: float | str | None = None,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a residuals diagnostic plot.

        Plots fitted values (``y_pred``) on x against residuals (raw or
        studentized) on y.  Data must carry the schema emitted by
        ``ModelSource.predictions()``: ``y_true``, ``y_pred``, ``residual``,
        ``studentized_residual``, and ``cooks_distance``.

        When ``reference_line=True`` a sentinel ``_ref_zero`` column is
        injected so the downstream ``mark_rule`` draws a single horizontal line
        at y=0.

        When ``cook_threshold`` is set, high-leverage points (Cook's D above
        the threshold) are highlighted as a second ``mark_point`` layer drawn
        in red with a black outline.

        Parameters
        ----------
        kind : {"studentized", "raw"}, default "studentized"
            Residual type to plot on the y axis.  ``"studentized"`` uses
            ``studentized_residual``; ``"raw"`` uses ``residual``.
        reference_line : bool, optional
            Whether to draw a horizontal reference line at y=0.  Default is
            ``True``.
        cook_threshold : float, "auto", or None, optional
            Cook's Distance threshold for outlier highlighting.  ``"auto"``
            uses the conventional ``4 / n`` rule.  ``None`` (default) disables
            outlier highlighting.
        color_field : str or None, optional
            Column name to drive per-group colour on the scatter layer.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the scatter layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for residuals rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.predictions()).mark_residuals(cook_threshold="auto")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_residuals
        from ferrum._constant_columns import _inject_constant

        return self._set_composite_mark(
            "residuals",
            desugar_residuals,
            {
                "kind": kind,
                "reference_line": reference_line,
                "cook_threshold": cook_threshold,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=(
                (lambda df: _inject_constant(df, "_ref_zero", 0.0)) if reference_line else None
            ),
        )

    def mark_prediction_error(
        self,
        *,
        reference_line: bool = True,
        ci: float | None = None,
        reference_band: bool = False,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render an actual-vs-predicted plot.

        Plots ``y_true`` on x against ``y_pred`` on y as scatter points.
        When ``reference_line=True`` the data is pre-sorted ascending by
        ``y_true`` so the downstream ``mark_line`` renders a monotonic y=x
        diagonal.  Data must carry ``y_true`` and ``y_pred`` columns (schema
        from ``ModelSource.predictions()``).

        Parameters
        ----------
        reference_line : bool, optional
            Whether to overlay a y=x identity reference line.  Default is
            ``True``.
        ci : float or None, optional
            Confidence level for a prediction-interval band (e.g. ``0.95``).
            ``None`` (default) omits the band.
        reference_band : bool, optional
            Whether to draw a shaded reference band around the identity line.
            Default is ``False``.
        color_field : str or None, optional
            Column name to drive per-group colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the scatter layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for prediction-error rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_test, y_test)
        >>> fm.Chart(src.predictions()).mark_prediction_error(ci=0.95)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_prediction_error
        from ferrum.plots._helpers import _sort_by

        return self._set_composite_mark(
            "prediction_error",
            desugar_prediction_error,
            {
                "reference_line": reference_line,
                "ci": ci,
                "reference_band": reference_band,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=((lambda df: _sort_by(df, "y_true")) if reference_line else None),
        )

    def mark_roc(
        self,
        *,
        average: str | None = None,
        reference_line: bool = True,
        annotate_auc: bool = False,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Receiver Operating Characteristic (ROC) curve.

        Plots false-positive rate (``fpr``) on x against true-positive rate
        (``tpr``) on y as a line per class.  Data must carry the schema emitted
        by ``ModelSource.roc_curve()``: ``fpr``, ``tpr``, ``threshold``,
        ``class``, ``auc``.  When ``reference_line=True`` the data is
        pre-sorted ascending by ``fpr`` before rendering.

        Parameters
        ----------
        average : str or None, optional
            When the data contains a macro/micro average row with
            ``class=average``, pass ``"macro"`` or ``"micro"`` to keep only
            that average line.  ``None`` (default) renders all classes.
        reference_line : bool, optional
            Whether to overlay a diagonal chance-level reference line
            (TPR=FPR).  Default is ``True``.
        annotate_auc : bool, optional
            Whether to annotate each curve with its AUC value.  Default is
            ``False``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for ROC rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.roc_curve()).mark_roc(annotate_auc=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_roc
        from ferrum.plots._helpers import _sort_by

        return self._set_composite_mark(
            "roc",
            desugar_roc,
            {
                "average": average,
                "reference_line": reference_line,
                "annotate_auc": annotate_auc,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=((lambda df: _sort_by(df, "fpr")) if reference_line else None),
        )

    def mark_pr(
        self,
        *,
        average: str | None = None,
        annotate_ap: bool = False,
        iso_lines: bool = False,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Precision-Recall (PR) curve.

        Plots recall on x against precision on y as a line per class.  Data
        must carry the schema emitted by ``ModelSource.pr_curve()``:
        ``precision``, ``recall``, ``threshold``, ``class``, ``ap``.

        Parameters
        ----------
        average : str or None, optional
            Filter to a specific average type row (e.g. ``"macro"``).
            ``None`` (default) renders all classes.
        annotate_ap : bool, optional
            Whether to annotate each curve with its average-precision (AP)
            value.  Default is ``False``.
        iso_lines : bool, optional
            Whether to draw iso-F1 reference lines in the background.
            Default is ``False``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for PR-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.pr_curve()).mark_pr(annotate_ap=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_pr

        return self._set_composite_mark(
            "pr",
            desugar_pr,
            {
                "average": average,
                "annotate_ap": annotate_ap,
                "iso_lines": iso_lines,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_calibration(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        reference_line: bool = True,
        color_field: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a calibration (reliability) curve.

        Plots mean predicted probability on x against fraction of positive
        outcomes (empirical probability) on y.  A perfectly calibrated model
        lies on the diagonal.  Data must carry the schema emitted by
        ``ModelSource.calibration_curve()``: ``mean_predicted``,
        ``fraction_positive``, ``count``.

        When ``reference_line=True`` the data is pre-sorted ascending by
        ``mean_predicted`` before rendering.

        Parameters
        ----------
        n_bins : int, optional
            Number of calibration bins.  Default is ``10``.
        strategy : {"uniform", "quantile"}, default "uniform"
            Binning strategy.  ``"uniform"`` uses equally-spaced bins;
            ``"quantile"`` uses equal-frequency bins.
        reference_line : bool, optional
            Whether to overlay a perfect-calibration diagonal reference line.
            Default is ``True``.
        color_field : str or None, optional
            Column name to drive per-group colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for calibration rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.calibration_curve()).mark_calibration(n_bins=15)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_calibration
        from ferrum.plots._helpers import _sort_by

        return self._set_composite_mark(
            "calibration",
            desugar_calibration,
            {
                "n_bins": n_bins,
                "strategy": strategy,
                "reference_line": reference_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=(
                (lambda df: _sort_by(df, "mean_predicted")) if reference_line else None
            ),
        )

    def mark_gain(
        self,
        *,
        reference_line: bool = True,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a cumulative gain chart.

        Plots percent of population contacted on x against the cumulative gain
        (fraction of positive cases captured) on y.  Data must carry the schema
        emitted by ``ModelSource.cumulative_gain()``: ``percent_population``,
        ``gain``, ``class``.  The no-skill diagonal baseline is encoded as rows
        with ``class='baseline'``.

        Parameters
        ----------
        reference_line : bool, optional
            Whether to draw the no-skill baseline diagonal.  Default is
            ``True``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for cumulative-gain rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.cumulative_gain()).mark_gain()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_gain

        return self._set_composite_mark(
            "gain",
            desugar_gain,
            {
                "reference_line": reference_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_lift(
        self,
        *,
        reference_line: bool = True,
        color_field: str | None = "class",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a lift curve chart.

        Plots percent of population targeted on x against lift (ratio of
        positive-case density to baseline) on y.  Data must carry the schema
        emitted by ``ModelSource.lift_curve()``: ``percent_population``,
        ``lift``, ``class``.  The no-skill lift=1 baseline is encoded as rows
        with ``class='baseline'``.

        Parameters
        ----------
        reference_line : bool, optional
            Whether to draw the lift=1 baseline rule.  Default is ``True``.
        color_field : str or None, optional
            Column name to drive per-class colour.  Default is ``"class"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for lift-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.lift_curve()).mark_lift()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_lift

        return self._set_composite_mark(
            "lift",
            desugar_lift,
            {
                "reference_line": reference_line,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_discrimination_threshold(
        self,
        *,
        metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
        n_thresholds: int = 50,
        threshold_line: bool = False,
        optimum_label: bool = True,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a discrimination-threshold sweep plot.

        Sweeps the decision threshold from 0 to 1 and plots multiple metrics
        (precision, recall, F1, queue rate) as lines against the threshold
        value.  Data must be in long form with columns ``threshold``, ``metric``,
        ``value`` -- the figure builder handles unpivoting from
        ``ModelSource.discrimination_threshold()`` output.

        Parameters
        ----------
        metrics : tuple of str, optional
            Metric names to include.  Default is
            ``("precision", "recall", "f1", "queue_rate")``.
        n_thresholds : int, optional
            Number of evenly-spaced threshold steps to evaluate.  Default is
            ``50``.
        threshold_line : bool, optional
            Whether to draw a vertical rule at the optimal threshold.  Default
            is ``False``.
        optimum_label : bool, optional
            Whether to overlay a text annotation at the F1-optimum point
            showing ``"max F1 = {f1:.3f} @ t={threshold:.2f}"``.
            Default ``True`` (Schwabish C7 audit-rework, 2026-05-12).
            The mark's data_transform injects ``_optimum_x`` /
            ``_optimum_y`` / ``_optimum_text`` sentinel columns from
            the long-form data; the desugar emits a ``mark_text`` layer.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for discrimination-threshold
            rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.discrimination_threshold()).mark_discrimination_threshold()
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_discrimination_threshold

        def _disc_threshold_prep(df):
            import polars as pl

            if "threshold" not in df.columns or "metric" not in df.columns:
                return df
            n = df.height
            if n == 0:
                return df
            # Find F1-optimum row from the long-form data. ``f1`` lives
            # as a ``metric`` value with ``value`` carrying the score.
            f1_rows = df.filter(pl.col("metric") == "f1")
            if f1_rows.height == 0:
                return df
            best_idx = int(f1_rows["value"].arg_max() or 0)
            best_t = float(f1_rows["threshold"][best_idx])
            best_f1 = float(f1_rows["value"][best_idx])
            new_cols = []
            if threshold_line:
                col = [best_t] + [None] * (n - 1)
                new_cols.append(pl.Series("_threshold_best", col, dtype=pl.Float64))
            if optimum_label:
                opt_x = [best_t] + [None] * (n - 1)
                opt_y = [best_f1] + [None] * (n - 1)
                opt_text = [f"max F1 = {best_f1:.3f} @ t={best_t:.2f}"] + [None] * (n - 1)
                new_cols.extend(
                    [
                        pl.Series("_optimum_x", opt_x, dtype=pl.Float64),
                        pl.Series("_optimum_y", opt_y, dtype=pl.Float64),
                        pl.Series("_optimum_text", opt_text, dtype=pl.Utf8),
                    ]
                )
            return df.with_columns(new_cols) if new_cols else df

        return self._set_composite_mark(
            "discrimination_threshold",
            desugar_discrimination_threshold,
            {
                "metrics": metrics,
                "n_thresholds": n_thresholds,
                "threshold_line": threshold_line,
                "optimum_label": optimum_label,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_disc_threshold_prep,
        )

    def mark_confusion(
        self,
        *,
        normalize: str | None = None,
        annotate: bool = True,
        color_field: str = "value",
        cmap: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a confusion matrix as an annotated heatmap.

        Renders an ordinal heatmap with actual class on one axis and predicted
        class on the other.  Data must carry the long-form schema emitted by
        ``ModelSource.confusion_matrix()``: ``actual``, ``predicted``,
        ``value``, ``value_fmt``.  When ``annotate=True``, a second
        ``mark_text`` layer reads ``value_fmt`` for per-cell text labels.

        Parameters
        ----------
        normalize : {"true", "pred", "all"} or None, optional
            Normalise cell counts.  ``None`` (default) shows raw counts.
        annotate : bool, optional
            Whether to overlay per-cell count / percentage text.  Default is
            ``True``.
        color_field : str, optional
            Column name driving the heatmap colour scale.  Default is
            ``"value"``.
        cmap : str or None, optional
            Sequential colormap name for the heat cells.  ``None`` (default)
            defers to the theme's sequential scheme.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the rect layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for confusion-matrix rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.confusion_matrix()).mark_confusion(normalize="true")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_confusion

        return self._set_composite_mark(
            "confusion",
            desugar_confusion,
            {
                "normalize": normalize,
                "annotate": annotate,
                "color_field": color_field,
                "cmap": cmap,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_class_prediction_error(
        self,
        *,
        normalize: bool = False,
        color_field: str = "predicted",
        show_counts: bool = True,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a class prediction error bar chart.

        Renders one stacked bar per actual class (x-axis), with segments
        coloured by predicted class.  This orientation surfaces which classes
        are confused with which -- for each actual class you can see how the
        model's predictions distribute across the predicted classes.  Data must
        carry long-form columns ``(actual, predicted, value)`` -- same shape as
        ``ModelSource.confusion_matrix(normalize=None)``.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalise each bar to 100%.  Default is ``False``.
        color_field : str, optional
            Column driving the segment colour.  Default is ``"predicted"``.
        show_counts : bool, optional
            Whether to overlay per-segment count text at the segment
            centre.  Default is ``True`` (Schwabish SB-followup
            2026-05-12).  Empty segments are skipped.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the bar layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for class-prediction-error
            rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(clf, X_test, y_test)
        >>> fm.Chart(src.confusion_matrix()).mark_class_prediction_error(normalize=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_class_prediction_error

        def _cpe_prep(df):
            if not show_counts or "value" not in df.columns:
                return df
            import polars as pl

            return df.with_columns(
                pl.when(pl.col("value") > 0)
                .then(pl.col("value").cast(pl.Int64).cast(pl.Utf8))
                .otherwise(None)
                .alias("_count_text"),
            )

        return self._set_composite_mark(
            "class_prediction_error",
            desugar_class_prediction_error,
            {
                "normalize": normalize,
                "color_field": color_field,
                "show_counts": show_counts,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_cpe_prep,
        )

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

    def mark_learning_curve(
        self,
        *,
        ci_style: str = "band",
        color_field: str | None = "split",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a learning curve (train size vs. CV score).

        Plots training set size on x against CV score on y, with separate
        lines for training and validation splits.  Data must carry the schema
        emitted by ``ModelSource.learning_curve()``, pre-deduped per
        ``(train_size, split)`` by the chart builder.

        Parameters
        ----------
        ci_style : {"band", "bars", "none"}, default "band"
            How to display cross-validation variance.  ``"band"`` draws a
            shaded ribbon; ``"bars"`` draws error bars; ``"none"`` omits CI.
        color_field : str or None, optional
            Column name to drive per-split line colour.  Default is
            ``"split"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for learning-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_train, y_train)
        >>> fm.Chart(src.learning_curve()).mark_learning_curve(ci_style="band")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_learning_curve

        return self._set_composite_mark(
            "learning_curve",
            desugar_learning_curve,
            {"ci_style": ci_style, "color_field": color_field, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_validation_curve(
        self,
        *,
        log_scale: bool = False,
        ci_style: str = "band",
        color_field: str | None = "split",
        param_label: str | None = None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a validation curve (hyperparameter vs. CV score).

        Plots a swept hyperparameter value on x against CV score on y, with
        separate lines for training and validation splits.  Data must carry the
        schema emitted by ``ModelSource.validation_curve()``, pre-deduped per
        ``(param_value, split)`` by the chart builder.

        Parameters
        ----------
        log_scale : bool, optional
            Whether to use a log scale on the x axis.  Useful for parameters
            like regularisation strength that span orders of magnitude.
            Default is ``False``.
        ci_style : {"band", "bars", "none"}, default "band"
            How to display cross-validation variance.
        color_field : str or None, optional
            Column name to drive per-split line colour.  Default is
            ``"split"``.
        param_label : str or None, optional
            Human-readable x-axis title for the hyperparameter being swept.
            The chart builder forwards the user's ``param`` argument here.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the line layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for validation-curve rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_train, y_train)
        >>> fm.Chart(src.validation_curve(param="C")).mark_validation_curve(
        ...     log_scale=True, param_label="C"
        ... )
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_validation_curve

        return self._set_composite_mark(
            "validation_curve",
            desugar_validation_curve,
            {
                "log_scale": log_scale,
                "ci_style": ci_style,
                "color_field": color_field,
                "param_label": param_label,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_cv_scores(
        self,
        *,
        kind: str = "box",
        split: str = "both",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a per-fold cross-validation score summary.

        Shows the distribution of CV scores across folds as a box plot, strip
        plot, or bar chart.  Data must carry the schema emitted by
        ``ModelSource.cv_scores()``.  The chart builder pre-aggregates per
        split when ``kind="bar"`` and passes raw per-fold rows for ``"box"``
        or ``"strip"``.

        Parameters
        ----------
        kind : {"box", "strip", "bar"}, default "box"
            Summary plot type.
        split : {"train", "test", "both"}, default "both"
            Which CV split(s) to display.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to constituent layers.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for CV-score rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(model, X_train, y_train)
        >>> fm.Chart(src.cv_scores()).mark_cv_scores(kind="strip", split="test")
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_cv_scores

        return self._set_composite_mark(
            "cv_scores",
            desugar_cv_scores,
            {"kind": kind, "split": split, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_alpha_selection(
        self,
        *,
        log_scale: bool = True,
        highlight_best: bool = True,
        ci_style: str = "band",
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a regularisation-strength (alpha) selection curve.

        Sweeps the regularisation parameter ``alpha`` and plots CV score as a
        function of alpha, with variance bands.  When ``highlight_best=True``
        a vertical rule is drawn at the alpha that maximises ``mean_score``.
        Data must carry the schema emitted by ``ModelSource.alpha_selection()``:
        ``alpha``, ``mean_score``, ``score_lo``, ``score_hi``, ``split``.

        Parameters
        ----------
        log_scale : bool, optional
            Whether to use a log scale on the x axis.  Default is ``True``
            (regularisation parameters typically span orders of magnitude).
        highlight_best : bool, optional
            Whether to draw a vertical reference rule at the optimal alpha.
            Default is ``True``.
        ci_style : {"band", "bars", "none"}, default "band"
            How to display CV variance.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides forwarded to constituent layers.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for alpha-selection rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> src = fm.ModelSource(lasso, X_train, y_train)
        >>> fm.Chart(src.alpha_selection()).mark_alpha_selection(log_scale=True)
        Chart(mark='point', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_alpha_selection
        from ferrum._constant_columns import _inject_constant

        def _inject_best_alpha(df):
            import polars as pl

            if "alpha" not in df.columns or "mean_score" not in df.columns:
                return df
            agg = (
                df.group_by("alpha")
                .agg(pl.col("mean_score").first())
                .sort("mean_score", descending=True)
            )
            if agg.height == 0:
                return df
            return _inject_constant(df, "_best_alpha", float(agg["alpha"][0]))

        return self._set_composite_mark(
            "alpha_selection",
            desugar_alpha_selection,
            {
                "log_scale": log_scale,
                "highlight_best": highlight_best,
                "ci_style": ci_style,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_inject_best_alpha if highlight_best else None,
        )

    # ---- Marks (clustering / manifold -- Phase 10f) ----

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

        Parameters
        ----------
        proba : bool, optional
            Whether to colour by predicted probability rather than class index.
            Default is ``False``.
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
        >>> fm.Chart(src.decision_boundary()).mark_decision_boundary(proba=True)
        Chart(mark='rect', encoding=[])
        """
        from ferrum.marks.diagnostic import desugar_decision_boundary

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
