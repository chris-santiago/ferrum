"""Phase 10g — feature-ranking visualizers (no-model variants).

Mirrors yellowbrick's ``Rank1D`` / ``Rank2D`` / ``ParallelCoordinates``
surfaces. None of these need a fitted estimator — they operate on the
raw feature matrix X (with optional y for ``Rank1D(algorithm="covariance")``)
and skip the ``ModelSource`` round-trip. The chart is materialized
directly from the in-house compute helpers in ``stats.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .base import FerrumVisualizer
from ferrum.plots.ranking import (
    _parallel_coords_chart_from_dataframe,
    _rank1d_chart_from_dataframe,
    _rank2d_chart_from_dataframe,
)
from .._rank_helpers import rank1d_compute, rank1d_compute_with_y, rank2d_compute


class Rank1DVisualizer(FerrumVisualizer):
    """Rank features by a univariate statistic and display as a bar chart.

    A no-model visualizer (``model=None``) that computes a scalar score
    for each feature column and renders them as a ranked horizontal or
    vertical bar chart.  ``fit`` is overridden directly — the base-class
    ``ModelSource`` round-trip is bypassed entirely.

    Records ``top_feature_score`` (the score of the highest-ranked
    feature) in ``_metrics``.

    Parameters
    ----------
    algorithm : {"shapiro", "variance", "covariance"}, default "shapiro"
        Scoring algorithm.

        - ``"shapiro"`` — Shapiro-Wilk W statistic (normality score).
        - ``"variance"`` — variance of each feature column.
        - ``"covariance"`` — absolute covariance with the target ``y``.
          Requires ``y`` to be passed at ``fit`` time; raises
          ``ValueError`` when ``y`` is ``None``.
    orient : {"horizontal", "vertical"}, default "horizontal"
        Bar orientation of the resulting chart.
    top_k : int, optional
        If given, display only the top ``top_k`` features.  ``None``
        shows all features.
    color_field : str, optional
        Column name forwarded to the chart's color encoding.  ``None``
        produces a single-color chart.
    random_state : int, optional
        Accepted for API symmetry with model-backed visualizers but
        intentionally never consumed — this visualizer bypasses
        ``ModelSource`` entirely. Documented as a permanent no-op so
        callers that script over visualizers don't have to special-case
        which ones accept the kwarg.
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Raises
    ------
    ValueError
        When ``algorithm="covariance"`` and ``y`` is ``None`` at ``fit``
        time.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.read_csv("wine.csv")
    >>> X, y = df.drop("target"), df["target"]
    >>> viz = fm.Rank1DVisualizer(algorithm="variance").fit(X)
    >>> viz.show()
    """

    def __init__(
        self,
        *,
        algorithm: str = "shapiro",
        orient: str = "horizontal",
        top_k: int | None = None,
        color_field: str | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.orient = orient
        self.top_k = top_k
        self.color_field = color_field

    def fit(self, X: Any, y: Any = None) -> "Rank1DVisualizer":
        if self.algorithm == "covariance":
            if y is None:
                raise ValueError("Rank1DVisualizer(algorithm='covariance') requires y.")
            df = rank1d_compute_with_y(X, y, algorithm="covariance")
        else:
            df = rank1d_compute(X, algorithm=self.algorithm)
        self._metrics["top_feature_score"] = float(df["score"][0])
        self._chart = _rank1d_chart_from_dataframe(
            df,
            algorithm=self.algorithm,
            orient=self.orient,
            top_k=self.top_k,
            color_field=self.color_field,
            theme=self.theme,
        )
        self._fitted = True
        return self


class Rank2DVisualizer(FerrumVisualizer):
    """Rank feature pairs by pairwise correlation and display as a heatmap.

    A no-model visualizer (``model=None``) that computes an N×N
    correlation matrix and renders it as an annotated heatmap.  ``fit``
    is overridden directly — the base-class ``ModelSource`` round-trip is
    bypassed entirely.

    The ``"kendall"`` algorithm routes pairwise computation through
    ``ferrum._core.kendall_tau_b`` (Rust) for performance.

    Records ``max_abs_corr`` (the largest absolute off-diagonal
    correlation value) in ``_metrics`` — useful for detecting
    multicollinearity at a glance.

    Parameters
    ----------
    algorithm : {"pearson", "spearman", "kendall", "covariance"}, default "pearson"
        Pairwise association measure.

        - ``"pearson"`` — Pearson linear correlation coefficient.
        - ``"spearman"`` — Spearman rank correlation.
        - ``"kendall"`` — Kendall tau-b, computed via
          ``ferrum._core.kendall_tau_b`` for performance.
        - ``"covariance"`` — raw covariance (not normalized to [-1, 1]).
    annot : bool, default True
        Whether to annotate each heatmap cell with its numeric value.
    random_state : int, optional
        Accepted for API symmetry with model-backed visualizers but
        intentionally never consumed — this visualizer bypasses
        ``ModelSource`` entirely. Documented as a permanent no-op so
        callers that script over visualizers don't have to special-case
        which ones accept the kwarg.
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.read_csv("wine.csv")
    >>> X = df.drop("target")
    >>> viz = fm.Rank2DVisualizer(algorithm="spearman").fit(X)
    >>> viz.show()
    """

    def __init__(
        self,
        *,
        algorithm: str = "pearson",
        annot: bool = True,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.annot = annot

    def fit(self, X: Any, y: Any = None) -> "Rank2DVisualizer":
        del y
        df = rank2d_compute(X, algorithm=self.algorithm)
        off_diag = df.filter(pl.col("feature_x") != pl.col("feature_y"))
        if off_diag.height > 0:
            self._metrics["max_abs_corr"] = float(off_diag["correlation"].abs().max() or 0.0)
        else:
            self._metrics["max_abs_corr"] = 0.0
        self._chart = _rank2d_chart_from_dataframe(
            df,
            algorithm=self.algorithm,
            annot=self.annot,
            theme=self.theme,
        )
        self._fitted = True
        return self


class ParallelCoordinatesVisualizer(FerrumVisualizer):
    """Visualize multivariate samples as a parallel-coordinates chart.

    A no-model visualizer (``model=None``) that draws one polyline per
    sample across a set of parallel vertical axes (one per feature).
    ``fit`` is overridden directly — the base-class ``ModelSource``
    round-trip is bypassed entirely.

    Records ``n_samples`` and ``n_features`` in ``_metrics`` so the
    ``repr`` surfaces the chart's shape.

    Parameters
    ----------
    features : list of str, optional
        Subset of column names to include as axes.  When ``None``, all
        non-``hue`` columns are used.
    hue : str, optional
        Column name used to color-encode each sample line. ``None``
        produces a single-color chart unless ``fit(X, y)`` is called
        with ``y`` — in that case ``y`` is attached as the hue column
        automatically.
    rescale : {"minmax", "zscore", None}, default "minmax"
        Per-axis normalization applied before plotting.

        - ``"minmax"`` — rescale each axis to [0, 1].
        - ``"zscore"`` — standardize each axis to zero mean, unit variance.
        - ``None`` — no rescaling; raw values are plotted.
    alpha : float, default 0.5
        Opacity of each sample line (0 = fully transparent, 1 = opaque).
    random_state : int, optional
        Accepted for API symmetry with model-backed visualizers but
        intentionally never consumed — this visualizer bypasses
        ``ModelSource`` entirely. Documented as a permanent no-op so
        callers that script over visualizers don't have to special-case
        which ones accept the kwarg.
    theme : Theme, optional
        Per-chart theme override.  Falls back to the global default
        when ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> import polars as pl
    >>> df = pl.read_csv("iris.csv")
    >>> viz = fm.ParallelCoordinatesVisualizer(hue="species", rescale="minmax")
    >>> viz.fit(df.drop("species"), df["species"])
    >>> viz.show()
    """

    def __init__(
        self,
        *,
        features: list[str] | None = None,
        hue: str | None = None,
        rescale: str | None = "minmax",
        alpha: float = 0.5,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.features = features
        self.hue = hue
        self.rescale = rescale
        self.alpha = alpha

    def fit(self, X: Any, y: Any = None) -> "ParallelCoordinatesVisualizer":
        # When the caller passes y but didn't set hue, route y as the
        # color encoding under a reserved column name. Matches the
        # sklearn convention that y is the supervisory signal — using it
        # as the visual grouping is the natural default when no explicit
        # hue column is named.
        effective_hue = self.hue
        if self.hue is None and y is not None:
            X = _attach_hue_from_y(X, y)
            effective_hue = "_hue"
        # n_samples / n_features bookkeeping for the repr.
        if isinstance(X, pl.DataFrame):
            n_samples = X.height
            n_features = (
                len(self.features)
                if self.features
                else (X.width - (1 if effective_hue in X.columns else 0))
            )
        elif hasattr(X, "shape"):
            n_samples = int(X.shape[0])
            n_features = len(self.features) if self.features else int(X.shape[1])
        else:
            arr = np.asarray(X)
            n_samples = int(arr.shape[0])
            n_features = len(self.features) if self.features else int(arr.shape[1])
        self._metrics["n_samples"] = float(n_samples)
        self._metrics["n_features"] = float(n_features)
        self._chart = _parallel_coords_chart_from_dataframe(
            X,
            features=self.features,
            hue=effective_hue,
            rescale=self.rescale,
            alpha=self.alpha,
            theme=self.theme,
        )
        self._fitted = True
        return self


def _attach_hue_from_y(X: Any, y: Any) -> Any:
    """Attach ``y`` as a ``_hue`` column on ``X`` for the parallel
    coordinates color encoding.

    Handles polars DataFrame, pandas DataFrame, and 2D numpy. Returns a
    new DataFrame; never mutates the input.
    """
    if isinstance(y, pl.Series):
        hue_vals = y.to_list()
    elif hasattr(y, "to_list"):
        hue_vals = list(y.to_list())
    else:
        hue_vals = list(np.asarray(y).tolist())

    if isinstance(X, pl.DataFrame):
        return X.with_columns(pl.Series("_hue", hue_vals))
    if hasattr(X, "assign") and hasattr(X, "columns"):
        # pandas DataFrame.
        return X.assign(_hue=hue_vals)
    # 2D numpy → polars first.
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {arr.shape}")
    df = pl.DataFrame({f"f{j}": arr[:, j].tolist() for j in range(arr.shape[1])})
    return df.with_columns(pl.Series("_hue", hue_vals))
