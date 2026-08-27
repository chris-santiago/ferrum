"""Model-selection mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ferrum.marks._desugar_helpers import _utf8_col

if TYPE_CHECKING:
    from ferrum.chart import Chart


class SelectionMarksMixin:
    """Mixin providing model-selection mark methods for Chart."""

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
        ci_style : {"band", "errorbar"}, default "band"
            How to display cross-validation variance.  ``"band"`` draws a
            shaded ribbon; ``"errorbar"`` draws a vertical rule per point.
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
        ci_style : {"band", "errorbar"}, default "band"
            How to display cross-validation variance.  ``"band"`` draws a
            shaded ribbon; ``"errorbar"`` draws a vertical rule per point.
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
        color_field: str | None = None,
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
        color_field : str or None, optional
            Column to drive per-group colour (and, for ``kind="box"``, the
            transform groupby).  ``None`` (default) keeps the single-model
            behaviour: box groups by ``split`` alone; strip colours by
            ``split`` and jitters within its band.  Passing a distinct
            column (e.g. ``"model"``, the GH #42 compare= dodge path)
            groups box stats by ``(split, color_field)`` and, for
            ``kind="strip"``, drops the default jitter so a chart-level
            ``position=Dodge(...)`` can offset points instead (position
            adjustments are not composable).
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
        from ferrum._validate import validate_choice

        validate_choice("mark_cv_scores", "split", split, ("train", "test", "both"))

        def _cv_scores_filter(df):
            if split == "both" or "split" not in df.columns:
                return df
            return df.filter(_utf8_col("split") == split)

        return self._set_composite_mark(
            "cv_scores",
            desugar_cv_scores,
            {"kind": kind, "split": split, "color_field": color_field, **mark_kwargs},
            placeholder="point",
            position=position,
            data_transform=_cv_scores_filter,
        )

    def mark_alpha_selection(
        self,
        *,
        log_scale: bool = True,
        highlight_best: bool = True,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a regularisation-strength (alpha) selection curve.

        Sweeps the regularisation parameter ``alpha`` and plots mean CV
        score as a single line.  When ``highlight_best=True`` a vertical
        rule is drawn at the alpha that maximises ``mean_score``.  Data
        must carry the schema emitted by ``ModelSource.alpha_selection()``:
        ``alpha``, ``fold``, ``score``, ``mean_score``, ``std_score`` (one
        row per (alpha, fold), deduped to one line point per alpha by
        the desugar's ``x``/``y`` encoding).

        Unlike ``mark_learning_curve``/``mark_validation_curve``,
        ``mark_alpha_selection`` renders a single mean-score line with no
        CI band -- the data contract carries no lower/upper variance
        columns (``std_score`` is present but unused for a band; a caller
        wanting to visualize it can layer its own errorbar) -- so it has
        no ``ci_style`` parameter; passing one raises ``TypeError``.

        Parameters
        ----------
        log_scale : bool, optional
            Whether to use a log scale on the x axis.  Default is ``True``
            (regularisation parameters typically span orders of magnitude).
        highlight_best : bool, optional
            Whether to draw a vertical reference rule at the optimal alpha.
            Default is ``True``.
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
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
            data_transform=_inject_best_alpha if highlight_best else None,
        )
