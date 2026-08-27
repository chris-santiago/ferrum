"""Regression-diagnostic mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` -- they call ``self._set_composite_mark``
and related Chart internals via ``self``.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ferrum.marks._desugar_helpers import _sort_by

if TYPE_CHECKING:
    from ferrum.chart import Chart


class RegressionMarksMixin:
    """Mixin providing regression-diagnostic mark methods for Chart."""

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
