"""Phase 10g — feature-ranking visualizers (no-model variants).

Mirrors yellowbrick's ``Rank1D`` / ``Rank2D`` / ``ParallelCoordinates``
surfaces. None of these need a fitted estimator — they operate on the
raw feature matrix X (with optional y for ``Rank1D(algorithm="covariance")``)
and skip the ``ModelSource`` round-trip. The chart is materialized
directly from the in-house compute helpers in ``stats.py``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import polars as pl

from ._base import FerrumVisualizer
from ferrum.plots.ranking import (
    _parallel_coords_chart_from_dataframe,
    _rank1d_chart_from_dataframe,
    _rank2d_chart_from_dataframe,
)
from .._internal._rank_helpers import rank1d_compute, rank1d_compute_with_y, rank2d_compute


class _ParallelSource(NamedTuple):
    """What a parallel-coordinates chart is drawn from: the samples and the column coloring them."""

    X: Any
    hue: str | None


class Rank1DVisualizer(FerrumVisualizer):
    """Rank features by a univariate statistic and display as a bar chart.

    A no-model visualizer (``model=None``) that computes a scalar score
    for each feature column and renders them as a ranked horizontal or
    vertical bar chart.  Its source is the score frame itself, so no
    ``ModelSource`` is built.

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
        Column of the computed ranking forwarded to the chart's color
        encoding: one of ``"feature"``, ``"score"`` or ``"rank"``. The fitted data's own columns are gone
        by the time the bars are encoded, so naming one fails with
        ``ValueError: unknown column`` when the chart renders.  ``None``
        produces a single-color chart.
    mark : dict, optional
        Per-layer mark overrides. See ``rank1d_chart``.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``.
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
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
    >>> df = pl.DataFrame({
    ...     "alcohol": [13.2, 12.4, 14.1, 12.0, 13.7, 12.9],
    ...     "ash": [2.4, 2.1, 2.6, 2.0, 2.5, 2.2],
    ...     "hue": [1.0, 0.9, 1.1, 0.8, 1.05, 0.95],
    ...     "target": [0, 1, 0, 1, 0, 1],
    ... })
    >>> X, y = df.drop("target"), df["target"]
    >>> viz = fm.Rank1DVisualizer(algorithm="variance").fit(X)
    >>> viz.show()
    Chart(mark='bar', encoding=['x', 'y'])
    """

    def __init__(
        self,
        *,
        algorithm: str = "shapiro",
        orient: str = "horizontal",
        top_k: int | None = None,
        color_field: str | None = None,
        mark: dict | None = None,
        encode: dict | None = None,
        properties: dict | None = None,
        layers: list | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.orient = orient
        self.top_k = top_k
        self.color_field = color_field
        self.mark = mark
        self.encode = encode
        self.properties = properties
        self.layers = layers

    def _build_source(self, X: Any, y: Any) -> pl.DataFrame:
        """Return the ranked per-feature score frame; only ``algorithm="covariance"`` reads ``y``."""
        if self.algorithm == "covariance":
            if y is None:
                raise ValueError("Rank1DVisualizer(algorithm='covariance') requires y.")
            return rank1d_compute_with_y(X, y, algorithm="covariance")
        return rank1d_compute(X, algorithm=self.algorithm)

    def _materialize(self) -> None:
        self._metrics["top_feature_score"] = float(self._source["score"][0])

    def _build_chart(self) -> Any:
        return _rank1d_chart_from_dataframe(
            self._source,
            algorithm=self.algorithm,
            orient=self.orient,
            top_k=self.top_k,
            color_field=self.color_field,
            mark=self.mark,
            encode=self.encode,
            properties=self.properties,
            layers=self.layers,
            theme=self.theme,
        )


class Rank2DVisualizer(FerrumVisualizer):
    """Rank feature pairs by pairwise correlation and display as a heatmap.

    A no-model visualizer (``model=None``) that computes an N×N
    correlation matrix and renders it as an annotated heatmap.  Its source
    is the correlation frame itself, so no ``ModelSource`` is built.

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
    mark : dict, optional
        Per-layer mark overrides. See ``rank2d_chart``.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``.
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
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
    >>> df = pl.DataFrame({
    ...     "alcohol": [13.2, 12.4, 14.1, 12.0, 13.7, 12.9],
    ...     "ash": [2.4, 2.1, 2.6, 2.0, 2.5, 2.2],
    ...     "hue": [1.0, 0.9, 1.1, 0.8, 1.05, 0.95],
    ...     "target": [0, 1, 0, 1, 0, 1],
    ... })
    >>> X = df.drop("target")
    >>> viz = fm.Rank2DVisualizer(algorithm="spearman").fit(X)
    >>> viz.show()
    Chart(mark='rect', encoding=[])
    """

    def __init__(
        self,
        *,
        algorithm: str = "pearson",
        annot: bool = True,
        mark: dict | None = None,
        encode: dict | None = None,
        properties: dict | None = None,
        layers: list | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.annot = annot
        self.mark = mark
        self.encode = encode
        self.properties = properties
        self.layers = layers

    def _build_source(self, X: Any, y: Any) -> pl.DataFrame:
        """Return the pairwise association frame; ``y`` is unused."""
        del y
        return rank2d_compute(X, algorithm=self.algorithm)

    def _materialize(self) -> None:
        off_diag = self._source.filter(pl.col("feature_x") != pl.col("feature_y"))
        if off_diag.height > 0:
            self._metrics["max_abs_corr"] = float(off_diag["correlation"].abs().max() or 0.0)
        else:
            self._metrics["max_abs_corr"] = 0.0

    def _build_chart(self) -> Any:
        return _rank2d_chart_from_dataframe(
            self._source,
            algorithm=self.algorithm,
            annot=self.annot,
            mark=self.mark,
            encode=self.encode,
            properties=self.properties,
            layers=self.layers,
            theme=self.theme,
        )


class ParallelCoordinatesVisualizer(FerrumVisualizer):
    """Visualize multivariate samples as a parallel-coordinates chart.

    A no-model visualizer (``model=None``) that draws one polyline per
    sample across a set of parallel vertical axes (one per feature).  Its
    source is the samples themselves, so no ``ModelSource`` is built.

    Records ``n_samples`` and ``n_features`` in ``_metrics`` so the
    ``repr`` surfaces the chart's shape.

    Parameters
    ----------
    features : list of str, optional
        Subset of column names to include as axes.  When ``None``, all
        non-``hue`` columns are used.
    hue : str, optional
        Column of the ``X`` passed to ``fit`` used to color-encode each
        sample line; it is not drawn as an axis. ``None``
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
    mark : dict, optional
        Per-layer mark overrides. See ``parallel_coordinates_chart``.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``.
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
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
    >>> df = pl.DataFrame({
    ...     "sepal_length": [5.1, 4.9, 7.0, 6.4, 6.3, 5.8],
    ...     "sepal_width": [3.5, 3.0, 3.2, 3.2, 3.3, 2.7],
    ...     "petal_length": [1.4, 1.4, 4.7, 4.5, 6.0, 5.1],
    ...     "species": ["setosa", "setosa", "versicolor", "versicolor",
    ...                 "virginica", "virginica"],
    ... })
    >>> viz = fm.ParallelCoordinatesVisualizer(hue="species", rescale="minmax")
    >>> viz.fit(df)
    ParallelCoordinatesVisualizer(n_samples=6.0000, n_features=3.0000)
    >>> viz.show()
    Chart(mark='line', encoding=[])

    Without ``hue``, the ``y`` passed to ``fit`` colors the lines:

    >>> viz = fm.ParallelCoordinatesVisualizer().fit(
    ...     df.drop("species"), df["species"]
    ... )
    >>> viz.show()
    Chart(mark='line', encoding=[])
    """

    def __init__(
        self,
        *,
        features: list[str] | None = None,
        hue: str | None = None,
        rescale: str | None = "minmax",
        alpha: float = 0.5,
        mark: dict | None = None,
        encode: dict | None = None,
        properties: dict | None = None,
        layers: list | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.features = features
        self.hue = hue
        self.rescale = rescale
        self.alpha = alpha
        self.mark = mark
        self.encode = encode
        self.properties = properties
        self.layers = layers

    def _build_source(self, X: Any, y: Any) -> _ParallelSource:
        """Return ``X`` and the column that colors its lines.

        When the caller passes y but didn't set hue, y is routed as the
        color encoding under a reserved column name. Matches the sklearn
        convention that y is the supervisory signal -- using it as the
        visual grouping is the natural default when no explicit hue column
        is named.
        """
        if self.hue is None and y is not None:
            return _ParallelSource(_attach_hue_from_y(X, y), "_hue")
        return _ParallelSource(X, self.hue)

    def _materialize(self) -> None:
        # n_samples / n_features bookkeeping for the repr.
        X, hue = self._source
        if isinstance(X, pl.DataFrame):
            n_samples = X.height
            n_features = (
                len(self.features) if self.features else (X.width - (1 if hue in X.columns else 0))
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

    def _build_chart(self) -> Any:
        return _parallel_coords_chart_from_dataframe(
            self._source.X,
            features=self.features,
            hue=self._source.hue,
            rescale=self.rescale,
            alpha=self.alpha,
            mark=self.mark,
            encode=self.encode,
            properties=self.properties,
            layers=self.layers,
            theme=self.theme,
        )


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
