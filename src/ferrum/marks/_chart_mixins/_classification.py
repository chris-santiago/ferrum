"""Classification-diagnostic mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ferrum.marks._desugar_helpers import _sort_by

if TYPE_CHECKING:
    from ferrum.chart import Chart


class ClassificationMarksMixin:
    """Mixin providing classification-diagnostic mark methods for Chart."""

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
            ``False``: a raw mark stays un-annotated.  (The ``roc_chart``
            figure function defaults to ``True`` and owns the overlay via
            the shared ``_metric_labels`` helper; the divergent defaults
            are intentional.)
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
            value.  Default is ``False``: a raw mark stays un-annotated.
            (The ``pr_chart`` figure function defaults to ``True`` and owns
            the overlay via the shared ``_metric_labels`` helper; the
            divergent defaults are intentional.)
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
