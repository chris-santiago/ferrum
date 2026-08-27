"""Ranking and multivariate-exploration figure functions and their private builders.

Public API
----------
rank_chart, rank1d_chart, rank2d_chart, parallel_coordinates_chart,
decision_boundary_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located builder that produces a
fully-formed ``Chart``.

Internal-only builders
----------------------
_rank1d_chart_from_dataframe, _rank2d_chart_from_dataframe,
_resolve_pc_features, _apply_pc_rescale,
_parallel_coords_chart_from_dataframe,
_decision_boundary_chart_from_source, _resolve_decision_boundary_features,
_build_decision_boundary_grid, _build_decision_boundary_unified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ferrum import Chart

from ferrum._coerce import to_polars
from ferrum._validate import validate_choice
from ferrum.encoding import X, Y
from ferrum.plots._helpers import (
    _UNSET,
    _finalize_chart,
    _resolve_first_param,
    _resolve_source,
    _warn_deprecated_dispatcher,
    _zero_anchored_domain,
)


# ---------------------------------------------------------------------------
# rank_chart  (deprecated dispatcher)
# ---------------------------------------------------------------------------


def rank_chart(
    data_or_source: Any = _UNSET,
    X: Any = None,
    y: Any = None,
    *,
    source: Any = _UNSET,  # deprecated keyword alias for ``data_or_source``
    rank: str = "2d",
    algorithm: str | None = None,
    top_k: int | None = None,
    annot: bool = True,
    orient: str = "horizontal",
    color_field: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Feature-ranking chart: univariate bar or pairwise heatmap.

    Computes a ranking score for each feature (or each feature pair)
    and renders either a ranked bar chart (``rank="1d"``) or a pairwise
    correlation heatmap (``rank="2d"``). Accepts a fitted estimator,
    ``ModelSource``, or a raw DataFrame / 2D array (no model required
    for most algorithms).

    Parameters
    ----------
    data_or_source : estimator, ModelSource, DataFrame, or array-like
        Input data. When a fitted estimator or ``ModelSource`` is
        supplied, the feature matrix is taken from the bound data.
        When a DataFrame or 2D array is supplied, ``X`` is used as
        the feature matrix if provided. (Family-canonical first-param
        name; the legacy keyword ``source=`` is accepted as a deprecated
        alias.)
    X : array-like, optional
        Feature matrix. Used when ``data_or_source`` is a raw estimator
        (not a ``ModelSource``) or when ``data_or_source`` is a raw
        DataFrame and ``X`` overrides it.
    y : array-like, optional
        Target vector. Required only for
        ``algorithm="covariance"`` which routes through
        ``ModelSource.rank1d``.
    rank : {"1d", "2d"}, default "2d"
        Ranking mode. ``"1d"`` computes a univariate score per feature
        and renders a horizontal bar chart (or vertical with
        ``orient="vertical"``). ``"2d"`` computes pairwise scores and
        renders a heatmap.
    algorithm : str or None, default None
        Ranking algorithm. When ``None``, defaults to ``"shapiro"``
        for ``rank="1d"`` and ``"pearson"`` for ``rank="2d"``.
    top_k : int or None, default None
        For ``rank="1d"``, truncate to the top-k features by score.
        Has no effect for ``rank="2d"``.
    annot : bool, default True
        For ``rank="2d"``, overlays the correlation value (2 decimal
        places) as a text label in each heatmap cell. Has no effect for
        ``rank="1d"``.
    orient : {"horizontal", "vertical"}, default "horizontal"
        Bar orientation for ``rank="1d"``; ignored for ``rank="2d"``.
    color_field : str or None, default None
        Column name to use for bar fill color in ``rank="1d"``; when
        ``None``, a single color is used.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Ranked bar chart (``rank="1d"``) or pairwise heatmap
        (``rank="2d"``).

    Raises
    ------
    ValueError
        If ``rank`` is not ``"1d"`` or ``"2d"``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.rank_chart(X_train, rank="2d")
    >>> fm.rank_chart(X_train, rank="1d", algorithm="shapiro", top_k=10)

    .. deprecated:: 2026-05-12
        Use [rank1d_chart][ferrum.rank1d_chart] or [rank2d_chart][ferrum.rank2d_chart] directly. This
        dispatcher remains as a shim that forwards to the appropriate
        sibling and will be removed in a future major release.
    """
    data_or_source = _resolve_first_param(
        data_or_source,
        source,
        canonical_name="data_or_source",
        alias_name="source",
        func_name="rank_chart",
    )
    if data_or_source is _UNSET:
        raise TypeError("rank_chart() missing required argument: 'data_or_source'")
    validate_choice("rank_chart", "rank", rank, {"1d", "2d"})

    _warn_deprecated_dispatcher("rank_chart", "rank", "rank1d_chart / rank2d_chart")
    if rank == "1d":
        return rank1d_chart(
            data_or_source,
            X,
            y,
            algorithm=algorithm,
            top_k=top_k,
            orient=orient,
            color_field=color_field,
            random_state=random_state,
            mark=mark,
            encode=encode,
            properties=properties,
            layers=layers,
            theme=theme,
        )
    return rank2d_chart(
        data_or_source,
        X,
        y,
        algorithm=algorithm,
        annot=annot,
        random_state=random_state,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# rank1d_chart
# ---------------------------------------------------------------------------


def rank1d_chart(
    data_or_source: Any = _UNSET,
    X: Any = None,
    y: Any = None,
    *,
    source: Any = _UNSET,  # deprecated keyword alias for ``data_or_source``
    algorithm: str | None = None,
    top_k: int | None = None,
    orient: str = "horizontal",
    color_field: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Univariate feature-ranking bar chart.

    Computes a per-feature ranking score and renders a horizontal (or
    vertical with ``orient="vertical"``) bar chart sorted by score.
    Accepts a fitted estimator, ``ModelSource``, or a raw DataFrame /
    2D array; the ``"covariance"`` algorithm requires ``y``.

    Parameters
    ----------
    data_or_source : estimator, ModelSource, DataFrame, or array-like
        Input data. When a fitted estimator or ``ModelSource`` is supplied,
        the feature matrix is taken from the bound data. (Family-canonical
        first-param name; the legacy keyword ``source=`` is accepted as a
        deprecated alias.)
    X, y : optional
        Feature matrix / target -- forwarded to ``_resolve_source`` when
        ``data_or_source`` is a raw estimator.
    algorithm : str or None, default None
        Ranking algorithm. ``None`` selects ``"shapiro"``.
    top_k : int or None, optional
        Limit the chart to the top-k features by score.  ``None`` shows
        all features.
    orient : {"horizontal", "vertical"}, default "horizontal"
        Bar orientation.
    color_field : str or None, optional
        Column name to map to bar color.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Ranked bar chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.rank1d_chart(model, X_train, algorithm="shapiro")
    """
    import ferrum

    data_or_source = _resolve_first_param(
        data_or_source,
        source,
        canonical_name="data_or_source",
        alias_name="source",
        func_name="rank1d_chart",
    )
    if data_or_source is _UNSET:
        raise TypeError("rank1d_chart() missing required argument: 'data_or_source'")

    algo = algorithm or "shapiro"
    if isinstance(data_or_source, ferrum.ModelSource):
        df = data_or_source.rank1d(algorithm=algo)
    elif algo == "covariance":
        ms = _resolve_source(
            data_or_source,
            X,
            y,
            random_state=random_state,
        )
        df = ms.rank1d(algorithm=algo)
    else:
        from ferrum.diagnostics._internal._rank_helpers import rank1d_compute

        input_data = data_or_source if X is None else X
        df = rank1d_compute(input_data, algorithm=algo)
    return _rank1d_chart_from_dataframe(
        df,
        algorithm=algo,
        orient=orient,
        top_k=top_k,
        color_field=color_field,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# rank2d_chart
# ---------------------------------------------------------------------------


def rank2d_chart(
    data_or_source: Any = _UNSET,
    X: Any = None,
    y: Any = None,
    *,
    source: Any = _UNSET,  # deprecated keyword alias for ``data_or_source``
    algorithm: str | None = None,
    annot: bool = True,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Pairwise feature-correlation heatmap.

    Computes pairwise feature correlation (or covariance) and renders a
    heatmap. Accepts a fitted estimator, ``ModelSource``, or a raw
    DataFrame / 2D array (no model required).

    Parameters
    ----------
    data_or_source : estimator, ModelSource, DataFrame, or array-like
        Input data. (Family-canonical first-param name; the legacy keyword
        ``source=`` is accepted as a deprecated alias.)
    X, y : optional
        Feature matrix / target.
    algorithm : str or None, default None
        Ranking algorithm. ``None`` selects ``"pearson"``.
    annot : bool, default True
        Overlay the correlation value (2 decimals) on each cell.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Pairwise correlation heatmap.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.rank2d_chart(model, X_train, algorithm="pearson")
    """
    import ferrum

    data_or_source = _resolve_first_param(
        data_or_source,
        source,
        canonical_name="data_or_source",
        alias_name="source",
        func_name="rank2d_chart",
    )
    if data_or_source is _UNSET:
        raise TypeError("rank2d_chart() missing required argument: 'data_or_source'")

    algo = algorithm or "pearson"
    if isinstance(data_or_source, ferrum.ModelSource):
        df = data_or_source.rank2d(algorithm=algo)
    else:
        from ferrum.diagnostics._internal._rank_helpers import rank2d_compute

        input_data = data_or_source if X is None else X
        df = rank2d_compute(input_data, algorithm=algo)
    return _rank2d_chart_from_dataframe(
        df,
        algorithm=algo,
        annot=annot,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# parallel_coordinates_chart
# ---------------------------------------------------------------------------


def parallel_coordinates_chart(
    data: Any,
    *,
    features: list[str] | None = None,
    hue: str | None = None,
    rescale: str | None = "minmax",
    alpha: float = 0.5,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Parallel coordinates chart for multivariate data.

    Renders one polyline per sample, with each feature mapped to a
    vertical axis. Features are optionally rescaled to a common range
    before plotting so all axes are visually comparable. Samples are
    colored by a grouping column when ``hue`` is provided.

    Parameters
    ----------
    data : polars.DataFrame, pandas.DataFrame, or array-like
        Input data. Polars and pandas DataFrames are used directly;
        2D numpy arrays are auto-named ``f0``, ``f1``, etc.
    features : list of str or None, default None
        Column names to use as parallel axes. When ``None``, all
        columns except ``hue`` are used.
    hue : str or None, default None
        Column name to color samples by (e.g. a target class or cluster
        id). Pass ``None`` for monochrome lines.
    rescale : {"minmax", "zscore"} or None, default "minmax"
        Per-feature rescaling applied before rendering so axes share a
        common visual range. ``"minmax"`` maps to ``[0, 1]``;
        ``"zscore"`` standardizes to zero mean and unit variance;
        ``None`` uses raw feature values.
    alpha : float, default 0.5
        Opacity of individual polylines; lower values reduce overplot
        in dense datasets.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Parallel coordinates chart with one polyline per sample.

    Raises
    ------
    ValueError
        If any name in ``features`` is not a column in ``data``.
    ValueError
        If ``rescale`` is not one of ``"minmax"``, ``"zscore"``, or
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.parallel_coordinates_chart(X_df, hue="species", rescale="minmax")
    """
    return _parallel_coords_chart_from_dataframe(
        data,
        features=features,
        hue=hue,
        rescale=rescale,
        alpha=alpha,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# decision_boundary_chart
# ---------------------------------------------------------------------------


def decision_boundary_chart(
    model: Any,
    X: Any,
    y: Any = None,
    *,
    features: tuple = (0, 1),
    grid_resolution: int = 200,
    proba: bool = False,
    scatter: bool = True,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Decision-boundary heatmap for a classifier over a 2D feature slice.

    Builds a ``grid_resolution x grid_resolution`` grid over two
    selected features, holds all other features fixed at their column
    means, and colors each cell by the model's predicted class (or
    probability). Optionally overlays training-point scatter.

    Parameters
    ----------
    model : estimator or ModelSource
        Fitted sklearn-compatible classifier or an explicit
        ``ferrum.ModelSource``.
    X : array-like
        Feature matrix. Must be provided (not optional) so the grid
        bounds and column means can be computed.
    y : array-like or None, default None
        True labels. Used only for the scatter overlay when
        ``scatter=True``; not required otherwise.
    features : tuple of (int or str, int or str), default (0, 1)
        Two feature indices or column names to use for the x and y axes
        of the grid. All other features are fixed at their column means.
        Exactly 2 features are required.
    grid_resolution : int, default 200
        Number of grid points along each axis; total cells =
        ``grid_resolution**2``.
    proba : bool, default False
        When ``True`` and the model exposes ``predict_proba``, the color
        channel uses ``predict_proba[:, 1]`` (continuous probability).
        When ``False``, the color channel uses ``predict`` (discrete
        class index).
    scatter : bool, default False
        When ``True``, overlays a scatter of training points colored by
        ``y`` on top of the boundary heatmap via the ``+`` compositor.
        Note: the overlay currently renders as horizontal concatenation
        per the ``ChartSpec`` one-batch contract.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.

    Returns
    -------
    Chart
        Decision-boundary heatmap, optionally with training-point
        scatter overlay.

    Raises
    ------
    ValueError
        If ``features`` does not contain exactly 2 elements.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.svm import SVC
    >>> fm.decision_boundary_chart(SVC().fit(X_train, y_train), X_train, y_train, features=(0, 1))
    """
    source = _resolve_source(model, X, y, random_state=random_state)
    return _decision_boundary_chart_from_source(
        source,
        features=tuple(features),
        grid_resolution=int(grid_resolution),
        proba=bool(proba),
        scatter=bool(scatter),
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _rank1d_chart_from_dataframe(
    df: pl.DataFrame,
    *,
    algorithm: str = "shapiro",
    orient: str = "horizontal",
    top_k: int | None = None,
    color_field: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Univariate rank1d chart over a pre-computed rank1d DataFrame.

    Truncates to ``top_k`` rows before rendering (the ModelSource has
    already sorted rows by descending score). ``algorithm`` is accepted
    for signature parity with ``rank_chart``; it doesn't affect the
    render -- the DataFrame already carries the scores.

    Pins the score axis to ``[0, max_score * 1.05]`` (or ``[0, 1]`` for
    shapiro -- W is bounded above by 1). Without an explicit zero
    baseline, ``mark_bar``'s ordinal-y path renders bars from the
    panel's left edge to ``to_pixel_f64(score)`` -- when scores are
    tightly clustered above zero (typical for Shapiro W in [0.98, 1.0]
    on well-behaved features), the auto-derived x domain starts at
    ``min(score)`` and the smallest-scored feature has a zero-width
    bar. Anchoring the domain at zero matches yellowbrick's default
    and makes relative magnitudes legible.
    """
    import ferrum

    if top_k is not None:
        df = df.head(int(top_k))
    max_score = float(df["score"].max() or 0.0)
    min_score = float(df["score"].min() or 0.0)
    if algorithm == "shapiro":
        x_domain = [0.0, 1.0]
    elif min_score < 0.0:
        # Negative scores (rare -- only seen for raw covariance without
        # abs); anchor at ``min - pad`` instead.
        pad = max(abs(min_score), abs(max_score)) * 0.05
        x_domain = [min_score - pad, max_score + pad]
    else:
        x_domain = list(_zero_anchored_domain(pl.Series([0.0]), df["score"]))

    chart = ferrum.Chart(df).mark_rank1d(
        orient=orient,
        color_field=color_field,
    )
    if orient == "horizontal":
        chart = chart.encode(
            x=X("score", scale={"type": "linear", "domain": x_domain}),
            y=Y("feature"),
        )
    else:
        chart = chart.encode(
            x=X("feature"),
            y=Y("score", scale={"type": "linear", "domain": x_domain}),
        )
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _rank2d_chart_from_dataframe(
    df: pl.DataFrame,
    *,
    algorithm: str = "pearson",
    annot: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Pairwise rank2d heatmap chart over a pre-computed rank2d DataFrame.

    When ``annot=True``, appends a ``correlation_fmt`` (Utf8) column
    holding ``"{:.2f}".format(correlation)`` per row so the text-overlay
    layer can render compact 2-dp labels without invoking Rust-side
    number formatting per cell.
    """
    import ferrum

    if annot and "correlation_fmt" not in df.columns:
        df = df.with_columns(
            pl.col("correlation")
            .map_elements(
                lambda v: f"{v:.2f}",
                return_dtype=pl.Utf8,
            )
            .alias("correlation_fmt"),
        )
    chart = ferrum.Chart(df).mark_rank2d(annot=annot)
    chart = chart.properties(title=ferrum.Title(f"Feature Correlation ({algorithm.title()})"))
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Parallel coordinates helpers
# ---------------------------------------------------------------------------


def _coerce_to_polars(data: Any) -> pl.DataFrame:
    """Coerce ``parallel_coordinates``' data argument into a polars DataFrame.

    Every input type :func:`ferrum._coerce.to_polars` accepts (polars,
    pyarrow, narwhals-compatible pandas/modin/cuDF/dask/ibis, dict,
    list[dict], numpy 2D) is normalized identically to every other coercion
    call site — including the datetime/categorical normalization
    ``to_arrow_table`` applies, which a pandas frame would otherwise skip.

    A bare ``list``/``tuple`` input (list-of-lists) carries no column
    metadata, and ``to_arrow_table`` rejects it outright (it requires a list
    of dicts), so those two types alone get a ``col_0, col_1, ...`` fallback
    frame — matching the ``to_arrow_table`` numpy-2D naming convention, so
    those names become the parallel-coordinates axis labels when
    ``features=None``. Every other input, including a 2D numpy array (which
    ``to_arrow_table`` already accepts directly, column-name-free), goes
    through ``to_polars`` alone.

    The fallback is gated on *input type*, not on "``to_polars`` raised a
    ``TypeError``": ``pyarrow.lib.ArrowTypeError`` (and narwhals' own failure
    surface) are both ``TypeError`` subclasses, so a genuinely named frame
    whose conversion fails (e.g. a pandas frame with an extension dtype
    ``to_arrow_table`` can't natively convert) must raise loudly instead of
    silently discarding its real column names for ``col_0, col_1, ...``.
    """
    if isinstance(data, (list, tuple)):
        try:
            return to_polars(data)
        except TypeError as exc:
            import numpy as np

            try:
                arr = np.asarray(data, dtype=np.float64)
            except (ValueError, TypeError):
                # Re-raise the original to_polars failure (not `from None`),
                # so this numpy attempt's own exception stays visible on
                # __context__ for diagnosis rather than being suppressed.
                raise exc
            if arr.ndim != 2:
                raise exc
            return pl.DataFrame({f"col_{j}": arr[:, j].tolist() for j in range(arr.shape[1])})
    return to_polars(data)


def _resolve_pc_features(
    df: pl.DataFrame,
    features: list[str] | None,
    hue: str | None,
) -> list[str]:
    """Resolve the parallel-coordinates feature list and validate it exists."""
    if features is None:
        features = [c for c in df.columns if c != hue]
    else:
        features = [str(c) for c in features]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"parallel_coordinates: features {missing!r} are not in the "
            f"data (available columns: {df.columns!r})."
        )
    return features


def _apply_pc_rescale(
    df: pl.DataFrame,
    features: list[str],
    rescale: str | None,
) -> pl.DataFrame:
    """Apply per-feature minmax / zscore rescale (or pass through on None)."""
    validate_choice("parallel_coordinates", "rescale", rescale, ("minmax", "zscore", None))
    if rescale is None:
        return df
    if rescale == "minmax":
        for c in features:
            col = df[c]
            vmin = col.min()
            vmax = col.max()
            if vmin is None or vmax is None or vmin == vmax:
                continue
            df = df.with_columns(
                ((pl.col(c) - float(vmin)) / (float(vmax) - float(vmin))).alias(c),
            )
        return df
    # rescale == "zscore" (validated above)
    for c in features:
        col = df[c]
        mu = col.mean()
        sd = col.std()
        if sd is None or sd == 0.0:
            continue
        df = df.with_columns(
            ((pl.col(c) - float(mu)) / float(sd)).alias(c),
        )
    return df


def _parallel_coords_chart_from_dataframe(
    data,
    *,
    features: list[str] | None = None,
    hue: str | None = None,
    rescale: str | None = "minmax",
    alpha: float = 0.3,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Parallel coordinates chart from a wide DataFrame.

    Reshapes ``data`` (a polars DataFrame, pandas DataFrame, or 2D
    numpy array) into long form (``sample_id``, ``feature``, ``value``)
    plus an optional ``hue`` column when provided, then renders one
    polyline per sample with ``mark_parallel_coordinates``.

    ``rescale`` in ``{"minmax", "zscore", None}``: per-feature rescaling
    applied before unpivot so all features share a common y axis.
    """
    import ferrum

    df = _coerce_to_polars(data)
    features = _resolve_pc_features(df, features, hue)
    df = _apply_pc_rescale(df, features, rescale)

    # Reshape to long form.
    id_cols = ["sample_id"] + ([hue] if hue is not None else [])
    df = df.with_row_index("sample_id").with_columns(
        pl.col("sample_id").cast(pl.Utf8),
    )
    long = df.unpivot(
        index=id_cols,
        on=features,
        variable_name="feature",
        value_name="value",
    )
    # Preserve feature order so the ordinal x scale lays out features in
    # the user-supplied (or default) sequence rather than alphabetical.
    long = (
        long.with_columns(
            pl.col("feature").cast(pl.Enum(features)),
        )
        .sort("sample_id", "feature")
        .with_columns(
            pl.col("feature").cast(pl.Utf8),
        )
    )

    if hue is not None:
        # Cast hue to Utf8 so the categorical color scale routes it
        # correctly (same Int64->continuous gotcha as silhouette.cluster).
        long = long.with_columns(pl.col(hue).cast(pl.Utf8))

    chart = ferrum.Chart(long).mark_parallel_coordinates(
        alpha=alpha,
        color_field=hue,
    )
    chart = chart.properties(title=ferrum.Title("Parallel Coordinates"))
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Decision boundary helpers
# ---------------------------------------------------------------------------


def _decision_boundary_chart_from_source(
    source: Any,
    *,
    features: tuple = (0, 1),
    grid_resolution: int = 200,
    proba: bool = False,
    scatter: bool = True,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Decision-boundary heatmap of model predictions over a 2D feature grid.

    Pre-computes a ``grid_resolution x grid_resolution`` grid of
    ``x / x2 / y / y2`` cell bounds and the model's prediction
    (``z`` = class index when ``proba=False``, ``P(class=1)`` when
    ``proba=True``). The grid is fed to ``mark_decision_boundary``
    (rect-based).

    When ``scatter=True`` and ``y`` is available, the training scatter
    is composed as a true overlay layer (not horizontal concat) by
    constructing a single unified DataFrame: grid rows hold
    ``x/x2/y/y2/z`` with the scatter coordinates null, and scatter rows
    hold ``scatter_x/scatter_y/scatter_z`` (= true class label mapped to
    the same numeric domain as ``z``) with the grid columns null. Both
    layers share the same DataFrame identity (avoiding unnecessary
    null-pad merge) so ``Chart.__add__`` layers them and both color
    encodings resolve
    against the same continuous color scale -- matching boundary and
    point colors means the cell-color directly says "predicted class /
    probability" while the point's color says "true class", so a
    misclassified point pops against its neighborhood.

    Non-numeric class labels (e.g. string labels) are mapped to a
    zero-based class-index float via the model's ``classes_`` attribute
    when present, otherwise via lexicographic order. The black point
    stroke is uniform; ``size=80`` ensures visibility against any
    background color including same-color cells.
    """
    import ferrum

    feat_idx = _resolve_decision_boundary_features(source, features)
    grid_info = _build_decision_boundary_grid(
        source,
        feat_idx,
        grid_resolution,
        proba=proba,
    )

    if not (scatter and source.y is not None):
        # Pure-boundary path: no overlay, no padding columns, no row mixing.
        grid_df = pl.DataFrame(
            {
                "x": [v - grid_info["dx"] / 2 for v in grid_info["flat_x"]],
                "x2": [v + grid_info["dx"] / 2 for v in grid_info["flat_x"]],
                "y": [v - grid_info["dy"] / 2 for v in grid_info["flat_y"]],
                "y2": [v + grid_info["dy"] / 2 for v in grid_info["flat_y"]],
                "z": [float(v) for v in grid_info["z"]],
            }
        )
        # Non-proba: cast class indices to String for categorical color scale.
        if not proba:
            grid_df = grid_df.with_columns(
                pl.col("z").cast(pl.Int64).cast(pl.Utf8).alias("z"),
            )
        # `proba` is not forwarded to mark_decision_boundary: its effect
        # (which grid `z` column got computed) already happened above, in
        # `_build_decision_boundary_grid`; the mark itself treats `proba`
        # as informational and warns if it is passed directly.
        chart = ferrum.Chart(grid_df).mark_decision_boundary()
        chart = chart.properties(title=ferrum.Title("Decision Boundary"))
        return _finalize_chart(
            chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
        )

    from ferrum._layer import _Layer

    unified = _build_decision_boundary_unified(source, grid_info)
    # Non-proba: cast class indices to String for categorical color scale.
    if not proba:
        unified = unified.with_columns(
            pl.col("z").cast(pl.Int64).cast(pl.Utf8).alias("z"),
            pl.col("scatter_z").cast(pl.Int64).cast(pl.Utf8).alias("scatter_z"),
        )
    # `proba` is not forwarded to mark_decision_boundary here either -- see
    # the pure-boundary path above for why.
    chart = ferrum.Chart(unified).mark_decision_boundary()
    chart = chart.layer(
        _Layer(
            mark="point",
            encoding={"x": "scatter_x", "y": "scatter_y", "color": "scatter_z"},
            mark_kwargs={"stroke": "#000000", "stroke_width": 1.0, "size": 80.0},
            name="scatter",
        )
    )
    chart = chart.properties(title=ferrum.Title("Decision Boundary"))
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _resolve_decision_boundary_features(
    source: Any,
    features: tuple,
) -> tuple[int, int]:
    feat_idx = tuple(
        source.feature_names.index(f) if isinstance(f, str) else int(f) for f in features
    )
    if len(feat_idx) != 2:
        raise ValueError(
            f"decision_boundary_chart requires exactly 2 features; got {len(feat_idx)}."
        )
    return feat_idx  # type: ignore[return-value]


def _build_decision_boundary_grid(
    source: Any,
    feat_idx: tuple[int, int],
    grid_resolution: int,
    *,
    proba: bool,
) -> dict:
    """Compute the prediction grid + cell bounds for a 2-feature
    decision-boundary chart. Returns a dict with ``flat_x``, ``flat_y``,
    ``dx``, ``dy``, ``z``, and the two feature vectors ``x_col``, ``y_col``.
    """
    import numpy as np

    # numpy required: 2D positional column slicing (X_np[:, i]) and meshgrid operations.
    X_np = source.X.to_numpy()
    x_col = X_np[:, feat_idx[0]].astype(np.float64)
    y_col = X_np[:, feat_idx[1]].astype(np.float64)
    pad_x = (x_col.max() - x_col.min()) * 0.05
    pad_y = (y_col.max() - y_col.min()) * 0.05
    xs = np.linspace(
        x_col.min() - pad_x,
        x_col.max() + pad_x,
        int(grid_resolution),
    )
    ys = np.linspace(
        y_col.min() - pad_y,
        y_col.max() + pad_y,
        int(grid_resolution),
    )
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    xx, yy = np.meshgrid(xs, ys)
    grid = np.tile(X_np.mean(axis=0), (xx.size, 1))
    grid[:, feat_idx[0]] = xx.ravel()
    grid[:, feat_idx[1]] = yy.ravel()
    if proba and "predict_proba" in source.capabilities:
        z = source.model.predict_proba(grid)[:, 1].astype(np.float64)
    else:
        z = np.asarray(source.model.predict(grid)).astype(np.float64)
    return {
        "flat_x": [float(v) for v in xx.ravel()],
        "flat_y": [float(v) for v in yy.ravel()],
        "dx": dx,
        "dy": dy,
        "z": z,
        "x_col": x_col,
        "y_col": y_col,
    }


def _build_decision_boundary_unified(source: Any, g: dict) -> pl.DataFrame:
    """Build the layered grid+scatter DataFrame for the overlay path.

    Maps ``y`` labels to the same numeric domain as ``z``. For
    ``proba=False`` ``z`` is the predicted class index (integer-cast
    prediction); ``scatter_z`` is the true class index in the same
    encoding so matching colors = correct prediction. For ``proba=True``
    ``z`` is ``P(class=1) in [0, 1]`` and the true label is 0 or 1.
    Prefers the model's ``classes_`` attribute when present; falls back
    to lex sort.

    Grid rows hold ``x/x2/y/y2/z`` (scatter columns null); scatter rows
    hold ``scatter_x/scatter_y/scatter_z`` (grid columns null). mark_rect
    skips null x/x2/y/y2 cells and mark_point skips null
    scatter_x/scatter_y, so each layer renders only its intended rows.
    The shared ``z`` color scale resolves coherently because both ``z``
    and ``scatter_z`` live on the same numeric domain.
    """
    import numpy as np

    y_raw = np.asarray(source.y)
    if hasattr(source.model, "classes_"):
        class_order = list(source.model.classes_)
    else:
        class_order = sorted({v for v in y_raw.tolist()})
    label_to_idx = {c: float(i) for i, c in enumerate(class_order)}
    scatter_z = np.array(
        [label_to_idx.get(v, float("nan")) for v in y_raw.tolist()],
        dtype=np.float64,
    )

    flat_x, flat_y = g["flat_x"], g["flat_y"]
    dx, dy, z = g["dx"], g["dy"], g["z"]
    x_col, y_col = g["x_col"], g["y_col"]
    n_grid = len(flat_x)
    n_scatter = len(x_col)
    nulls_grid = [None] * n_grid
    nulls_scatter = [None] * n_scatter

    return pl.DataFrame(
        {
            "x": [v - dx / 2 for v in flat_x] + nulls_scatter,
            "x2": [v + dx / 2 for v in flat_x] + nulls_scatter,
            "y": [v - dy / 2 for v in flat_y] + nulls_scatter,
            "y2": [v + dy / 2 for v in flat_y] + nulls_scatter,
            "z": [float(v) for v in z] + nulls_scatter,
            "scatter_x": nulls_grid + [float(v) for v in x_col],
            "scatter_y": nulls_grid + [float(v) for v in y_col],
            "scatter_z": nulls_grid + [float(v) for v in scatter_z],
        },
        schema={
            "x": pl.Float64,
            "x2": pl.Float64,
            "y": pl.Float64,
            "y2": pl.Float64,
            "z": pl.Float64,
            "scatter_x": pl.Float64,
            "scatter_y": pl.Float64,
            "scatter_z": pl.Float64,
        },
    )
